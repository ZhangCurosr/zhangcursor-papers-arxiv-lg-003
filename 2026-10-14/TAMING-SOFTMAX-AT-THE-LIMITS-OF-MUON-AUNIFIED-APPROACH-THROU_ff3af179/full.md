# TAMING SOFTMAX AT THE LIMITS OF MUON: AUNIFIED APPROACH THROUGH HILBERT GEOMETRY

Aditya Somasundaram, Charles Guille-Escuret, Alexander Moreno Zhengzhong Liu, Eric Xing

Institute of Foundation Models, Mohamed bin Zayed University of Artificial Intelligence (MBZUAI) aditya.somasundaram@mbzuai.ac.ae

## ABSTRACT

Muon controls weight updates through the RMS movement of a linear layer’s output. This implicitly treats the linear output as a proxy for the full module, including the nonlinearity that follows. This proxy can be misleading for softmax, where the RMS shift in activations does not faithfully measure the change in the output distribution. We propose Hilbert distance as a natural metric for bounding changes in these output distributions. Hilbert distance bounds changes in crossentropy loss and limits probability redistribution. For softmax, it takes the form of the variation seminorm of the preactivation perturbation, allowing probability constraints to be expressed as parameter constraints. Formulating the steepest descent problem with a Hilbert distance constraint brings two recently proposed methods under a unified mathematical framework. For the LM head, solving a row-wise approximation to the resulting diameter-constrained problem leads us to RowNorm (Pethick et al., 2025). For scaled dot-product attention, the same principle yields the first-order query–key steepest descent problem underlying Compositional Muon (Keigwin et al., 2026). These connections motivate using Hilbert distance as a guiding principle for optimizer design in softmax modules.

## 1 INTRODUCTION

An important consideration for stable training is the extent to which a single update changes the network’s intermediate representations. Each intermediate representation is both the output of one layer and the input to the next. This dependence is especially important in deep sequential models, where changing one representation alters the inputs processed by every layer that follows. Without proper control, these representation shifts can accumulate across layers and destabilize training. Muon formalizes this idea for linear layers by controlling the worst-case RMS movement of the layer’s output over inputs of bounded size (Jordan et al., 2024; Bernstein, 2025a). This requirement leads to a suitably scaled spectral norm constraint on the weight update.

This spectral constraint is not arbitrary: it follows from choosing RMS to measure the functional change produced by the update. More broadly, it illustrates how the geometry of an optimizer depends on the notion of change assigned to a module. The Modula project and Scion generalize this perspective by advocating update geometries that reflect the function of each module (Bernstein, 2025b; Pethick et al., 2025).

Muon’s standard update geometry, however, is derived for the linear map and does not explicitly adapt to architectural operations such as normalization and nonlinear activations. As a result, Muon does not yet provide a single optimization rule for every parameter group in the model. In standard Muon training recipes, the LM head, embeddings, and scalar or vector parameters are typically optimized with AdamW (Jordan et al., 2024). Large-scale Muon training can also require additional stabilization. For example, Kimi K2 introduced MuonClip to address excessive attention-score growth by rescaling query and key weights after an update (Team et al., 2026).

We focus on how optimizer design changes when a linear map is followed by softmax. The linear layer’s output is an intermediate state, and its RMS displacement does not faithfully measure the change in the probability distribution produced by softmax. We instead directly constrain the change in the output probability distribution. We propose using Hilbert’s projective metric, or Hilbert distance, for this purpose.

Our contributions are threefold:

• We formulate the steepest descent problem using Hilbert distance to constrain changes in softmax output distributions. This single formulation brings two seemingly independent optimization methods under a unified mathematical framework.

• For the LM head, we show that this probability constraint becomes a row diameter constraint. Its exact dual scales quadratically with vocabulary size. A tighter common-ball constraint instead leads to RowNorm in closed form (Pethick et al., 2025).

• For standard attention, we show that this probability constraint produces a spectral constraint on the first-order change in the QK product, recovering the steepest descent problem posed in Compositional Muon (Keigwin et al., 2026).

Beyond unifying these recent methods, we provide a guiding principle for optimizer design for any neural network module that outputs a probability distribution.

We first expose the mismatch between RMS and softmax (Section 2). Next, we introduce Hilbert geometry and establish its suitability for measuring changes in softmax output distributions (Section 3). We subsequently formulate the corresponding steepest descent problems for the LM head and attention. For the LM head, a tighter approximation leads to RowNorm in closed form. For attention, our formulation recovers the QK steepest descent problem studied by Compositional Muon (Sections 4 and 5).

## 2 PRELIMINARIES ON STEEPEST DESCENT

Steepest descent chooses an update that gives the greatest first-order decrease in the loss while remaining small under a chosen measure of size. For a minibatch gradient $G = \nabla _ { W } \mathcal { L }$ , an update $\Delta W$ changes the loss to first order by $\langle G , \Delta W \rangle _ { F }$ . Let ${ \mathcal { D } } _ { W } ( \Delta W )$ measure the size of this update at the current weights W. We consider

$$
\operatorname* { m i n } _ { \Delta W } \big \langle G , \Delta W \big \rangle _ { F } \qquad \mathrm { s u b j e c t ~ t o } \qquad \mathcal { D } _ { W } \big ( \Delta W \big ) \leq \rho ,\tag{1}
$$

where $\rho > 0$ sets the permitted update size. We leave $\mathcal { D } _ { W }$ unspecified for now, allowing its choice to reflect the module being optimized.

The importance of the underlying choice of metric cannot be overstated. Choosing the Frobenius norm measures the update directly in parameter space and gives the SGD direction (Robbins & Monro, 1951). Muon takes a functional perspective instead: for a linear layer, it controls the worstcase change in the output produced by the update (Bernstein, 2025a). Measuring this change with RMS leads to a spectral norm constraint on the weight update, whose steepest descent solution gives the Muon direction. The Modula project generalizes this perspective by advocating update geometries that reflect the function of each module (Bernstein, 2025b). Around this principle, Scion develops a systematic optimization framework, formulating updates as Linear Minimization Oracles under $\bar { \| } { \cdot } \| _ { p \to q }$ operator norm constraints (Pethick et al., 2025). Here, the norms $\left\| \cdot \right\| _ { p }$ and $\left\| \cdot \right\| _ { q }$ describe the input and output geometries of the module. The shared ethos of these works is to build optimizers whose geometry reflects the function of each module.

This functional perspective becomes more delicate when a linear layer is followed by a nonlinearity. The linear output is then only an intermediate state in the computation of the full module. Controlling the RMS movement of this intermediate state does not account for the geometry introduced by the subsequent nonlinearity. For a softmax module, the linear output is the preactivation vector, which softmax maps to a probability distribution on the simplex. We therefore ask whether RMS displacement of the preactivation faithfully measures changes in the output distribution.

## 2.1 GEOMETRIC MISMATCH BETWEEN MUON AND SOFTMAX

We argue that Muon is inadequate for modules involving softmax. Muon’s spectral geometry can allocate updates poorly for such modules. The distinction lies between the nature of RMS and softmax: RMS measures how much the preactivation vector moves on average, while softmax measures how much one coordinate gains relative to others.

![](images/3d1d1f53578b64c89979e256627a8bcffcbe73224d266375af56748ce6899d01.jpg)  
Figure 1: A common RMS bound permits very different changes in softmax probabilities. All cases share the same centered $z ~ \in ~ \mathbb { R } ^ { 3 2 }$ and initial distribution $p = \operatorname { s o f t m a x } ( z )$ The centered perturbations produce $q = \mathrm { s o f t m a x } ( z + \Delta z )$ and all satisfy $\mathrm { \tilde { R M S } } ( \Delta z ) \leq \mathrm { \tilde { 1 } }$ . The pairs $( \mathrm { R M S } ( \Delta z ) , \mathcal { H } ( p , q ) )$ are $( 1 , 4 )$ for blue, (1, 2) for purple, and $( 1 / 2 , 4 )$ for red, where H denotes Hilbert distance. Blue substantially reshapes the distribution, while purple barely changes it. Red matches blue’s Hilbert distance at half its RMS.

A simple example. Consider an update where all the rows are equal, $\Delta W = { \mathbf { 1 } } b ^ { \top }$ . In this case, $\Delta z = { \bar { \Delta } } W x = { \bar { ( } } b ^ { \top } x ) \mathbf { 1 }$ , a scaled ones vector. Such a perturbation to the preactivation vector does not alter the probability distribution, despite incurring an arbitrarily large RMS norm. This particular case is not problematic in practice because softmax modules receive row-centered gradients, causing Muon to consider only centered directions; see Section 6. It nevertheless illustrates a broader geometric mismatch between RMS bounds and softmax that can produce practical pathologies.

Figure 1 shows that this discrepancy persists under centered perturbations. Unlike the preceding example, all three cases share the same initial distribution and use centered perturbations satisfying a common RMS bound. The blue and purple updates have equal RMS, yet the blue perturbation substantially reshapes the distribution while the purple perturbation barely changes it. The red perturbation has half their RMS but still causes a substantial probability redistribution.

This mismatch compels direct control of the output distribution. Designing an effective update rule for softmax modules requires choosing a suitable metric on the output distribution and formulating the induced steepest descent problem. This raises a fundamental question:

## What is the appropriate metric for perturbations of the softmax probabilities?

## 3 HILBERT GEOMETRY OF SOFTMAX

We turn to Hilbert’s projective metric, or simply Hilbert distance (Birkhoff, 1957; De La Harpe, 1993). For strictly positive vectors $x , y \in \mathbb { R } _ { > 0 } ^ { n } .$ , it is defined by $\begin{array} { r } { \mathcal { H } ( x , y ) = \log \frac { \operatorname* { m a x } _ { i } x _ { i } / y _ { i } } { \operatorname* { m i n } _ { j } x _ { j } / y _ { j } } } \end{array}$ . We apply this metric to strictly positive probability distributions on the simplex $\mathcal { P } = \{ x \in \bar { \mathbb { R } } _ { > 0 } ^ { n } : \sum _ { i } x _ { i } = 1 \}$ } (Nielsen & Sun, 2018; 2023). For $p , q \in \mathcal { P }$ , Hilbert distance can equivalently be written as

$$
\mathcal { H } ( p , q ) = \operatorname* { m a x } _ { i , j } \left| \log \frac { q _ { i } / q _ { j } } { p _ { i } / p _ { j } } \right| .\tag{2}
$$

Thus, Hilbert distance measures the largest change in pairwise odds.

Hilbert’s projective metric has several useful geometric properties on positive cones and probability simplices (Lemmens & Nussbaum, 2012; Papadopoulos & Troyanov, 2014). Bounding the Hilbert distance directly controls the change in cross-entropy loss and limits probability redistribution in total variation. For softmax, the distance also admits a closed-form expression in terms of the preactivation update: it coincides with the range between the largest and smallest coordi nates (the variation seminorm). Consequently, uniform shifts leave the distance invariant, mirroring the translation invariance of the softmax itself. Motivated by these geometric properties, we adopt the Hilbert distance to quantify changes in output distributions and formulate the resulting steepest descent problem.

Hilbert distance bounds pairwise odds ratios. Throughout, let $z \in \mathbb { R } ^ { n }$ denote the preactivation supplied to softmax, and let $p = \operatorname { s o f t m a x } ( z )$ be the resulting output distribution. A perturbation $\Delta z$ produces the updated distribution $q = \mathrm { s o f t m a x } ( z + \Delta z )$ . By Equation 2, bounding Hilbert distance uniformly bounds the odds ratio for every pair of outcomes:

$$
\mathcal { H } ( p , q ) \leq \varepsilon \quad \Longleftrightarrow \quad e ^ { - \varepsilon } \leq \frac { q _ { i } / q _ { j } } { p _ { i } / p _ { j } } \leq e ^ { \varepsilon } \qquad \forall i , j .
$$

In particular, for softmax, Hilbert distance has the closed form

$$
\mathcal { H } \big ( \mathrm { s o f t m a x } ( z ) , \mathrm { s o f t m a x } ( z + \Delta z ) \big ) = \operatorname* { m a x } _ { i , j } | \Delta z _ { i } - \Delta z _ { j } | = \operatorname* { m a x } _ { i } \Delta z _ { i } - \operatorname* { m i n } _ { i } \Delta z _ { i } = \left\| \Delta z \right\| _ { \mathrm { v a r } } .\tag{3}
$$

The variation seminorm $\| \Delta z \| _ { \operatorname { v a r } } ,$ also called the oscillation osc $( \Delta z )$ , is the difference between the largest and smallest entries of $\Delta z$ . It vanishes precisely when $\Delta z$ is constant, matching the translation invariance of softmax.

Hilbert distance bounds probability redistribution. Bounding pairwise odds ratios also limits the redistribution of probability mass. We measure this redistribution using total variation,

$$
\mathrm { T V } ( p , q ) = { \frac { 1 } { 2 } } \sum _ { i } \left| p _ { i } - q _ { i } \right| = \operatorname* { m a x } _ { A \subseteq \{ 1 , \ldots , n \} } \left| \sum _ { i \in A } ( p _ { i } - q _ { i } ) \right| .
$$

Thus, $\mathrm { T V } ( p , q ) \leq \delta$ guarantees that no set of outcomes can gain or lose more than $\delta$ probability mass. Hilbert distance gives a sharp bound on total variation (Cohen & Fausti, 2024, Theorem 5.1),

$$
\mathrm { T V } ( p , q ) \leq \operatorname { t a n h } \left( { \frac { { \mathcal { H } } ( p , q ) } { 4 } } \right) .
$$

The bound is sharp even when the preactivation perturbation is fixed. For every $\Delta z \in \mathbb { R } ^ { n }$

$$
\operatorname* { s u p } _ { z \in \mathbb { R } ^ { n } } \mathrm { T V } \big ( \mathrm { s o f t m a x } ( z ) , \mathrm { s o f t m a x } ( z + \Delta z ) \big ) = \mathrm { t a n h } \bigg ( \frac { \| \Delta z \| _ { \mathrm { v a r } } } { 4 } \bigg ) .
$$

The variation seminorm therefore determines the largest probability redistribution that a given preactivation update can produce across all starting distributions. Appendix $_ { \mathrm { A } . 2 }$ proves this sharpness statement.

Hilbert distance bounds change in cross-entropy (CE) loss. Let p and $q$ be the prediction distributions before and after an update. For a target outcome i, the CE loss is $\ell _ { i } ( p ) = - \log p _ { i }$ so its change is $\begin{array} { r } { \Delta \ell _ { i } = \ell _ { i } ( q ) \dot { - } \ell _ { i } ( p ) = \log \frac { p _ { i } } { q _ { i } } } \end{array}$ . Hilbert distance is the range of these changes, $\begin{array} { r } { \mathcal { H } ( p , q ) = \operatorname* { m a x } _ { i } \Delta \ell _ { i } - \operatorname* { m i n } _ { i } \Delta \ell _ { i } } \end{array}$ . It therefore measures how differently the update affects the loss across possible targets. Moreover, $\begin{array} { r } { \sum _ { i } p _ { i } = \sum _ { i } q _ { i } = 1 } \end{array}$ implies min<sub>i</sub> $\begin{array} { r } { \frac { \tilde { p } _ { i } } { q _ { i } } \leq 1 \leq \operatorname* { m a x } _ { i } \frac { p _ { i } } { q _ { i } } } \end{array}$ . Taking logarithms gives min<sub>i</sub> $\Delta \ell _ { i } \leq 0 \leq$ max<sub>i</sub> $\Delta \ell _ { i }$ . Consequently, $\mathcal { H } ( p , q ) \leq \dot { \varepsilon }$ guarantees that the change in CE loss is at most ε,

$$
| \Delta \ell _ { i } | \le \operatorname* { m a x } _ { j } \Delta \ell _ { j } - \operatorname* { m i n } _ { j } \Delta \ell _ { j } \le \mathcal { H } ( p , q ) \quad \forall i .
$$

Steepest descent problem. Together, these results show that Hilbert distance controls pairwise odds, probability redistribution, and changes in cross-entropy loss. Having established these properties, we now formulate the steepest descent problem using Hilbert distance to measure the size of an update. We return to Equation 1 and express the update constraint in terms of this distance. Let $W$ denote the module’s parameters and let $p _ { W } ( x )$ be the output distribution produced at input x. An update $\Delta W$ changes this distribution from $p _ { W } ( x )$ to $p _ { W + \Delta W } ( x )$ . Requiring these distributions to remain within Hilbert distance $\varepsilon$ for every input in a prescribed set X gives

$$
\operatorname* { m i n } _ { \Delta W } \big \langle \nabla _ { W } \mathcal { L } , \Delta W \big \rangle _ { F } \qquad \mathrm { s u b j e c t ~ t o } \qquad \operatorname* { s u p } _ { x \in \mathcal { X } } \mathcal { H } \big ( p _ { W } ( x ) , p _ { W + \Delta W } ( x ) \big ) \leq \varepsilon .\tag{4}
$$

## 4 THE LM HEAD

For the LM head, bounding Hilbert distance over the hidden state space is equivalent to bounding the diameter of the update rows. The resulting steepest descent problem has an exact dual whose size grows quadratically with the vocabulary. A tighter common-ball approximation avoids this coupling and leads to RowNorm in closed form.

We first specialize Equation 4 to the LM head. For a hidden state $h \in \mathbb { R } ^ { d }$ , the LM head produces the logit vector $z = W h ^ { \bullet } \in \mathbb { R } ^ { V }$ and the token distribution $p = \operatorname { s o f t m a x } ( z )$ . We consider an untied head without bias, trained with CE and no other loss term depending on W. Throughout this section, we compare the head before and after an update at the same hidden state.

Hilbert distance induces a row diameter constraint. Let $\Delta W$ be a weight update with rows $\Delta w _ { i } ^ { \top }$ . The update changes the logit for token i by $\Delta w _ { i } ^ { \top } h .$ . For two tokens i and $j ,$ their logit difference therefore changes by $( \Delta w _ { i } - \Delta w _ { j } ) ^ { \top } h$ . Equation 3 gives

$$
\mathcal { H } \big ( p _ { W } ( h ) , p _ { W + \Delta W } ( h ) \big ) = \| \Delta W h \| _ { \mathrm { v a r } } = \operatorname* { m a x } _ { i , j } \big | ( \Delta w _ { i } - \Delta w _ { j } ) ^ { \top } h \big | .
$$

For a fixed pair of rows, the largest value of this inner product over $\left\| h \right\| _ { 2 } \leq R { \mathrm { i s } } R \left\| \Delta w _ { i } - \Delta w _ { j } \right\| _ { 2 }$ Taking the largest value over all row pairs gives

$$
\operatorname* { s u p } _ { \| h \| _ { 2 } \leq R } \mathcal { H } \big ( p _ { W } ( h ) , p _ { W + \Delta W } ( h ) \big ) = R \operatorname* { m a x } _ { i , j } \| \Delta w _ { i } - \Delta w _ { j } \| _ { 2 } = R D ( \Delta W ) .\tag{5}
$$

The maximum pairwise distance is precisely the diameter $D ( \Delta W )$ of the row cloud $\{ \Delta w _ { 1 } , \ldots , \Delta w _ { V } \}$ (Rudin, 1976, Definition 3.9). For this input ball, the general constraint therefore specializes to ${ \mathcal { D } } _ { W } ( \Delta W ) = R D ( \Delta W )$ ). Choosing $R = { \sqrt { d } }$ gives the RMS unit ball. Thus, requiring a Hilbert distance of at most ε throughout this ball is equivalent to imposing $D ( \Delta W ) \leq \rho _ { \mathrm { h e a d } } .$ where $\rho _ { \mathrm { h e a d } } = \varepsilon / R$

This single constraint on the update rows controls both the predicted distribution and the crossentropy loss. By Section 3, no group of tokens can gain or lose more than tanh $. ( \varepsilon / 4 )$ probability mass, and the CE loss changes by at most ε for any target token.

Steepest descent under the row diameter constraint. We seek the update with the greatest firstorder loss decrease under the row diameter bound. Let $G = \nabla _ { W } \mathcal { L }$ have rows $g _ { i } ^ { \top }$ . Using the row diameter constraint in Equation 5, we obtain

$$
\operatorname* { m i n } _ { \Delta W } \langle G , \Delta W \rangle _ { F } \qquad \mathrm { s u b j e c t t o } \qquad D ( \Delta W ) \leq \rho _ { \mathrm { h e a d } } .\tag{6}
$$

Since the loss depends on $W$ only through the softmax output, its gradient satisfies $\mathbf { 1 } ^ { \top } G = 0$ Adding the same vector to every update row produces a common shift of the logits and therefore leaves both the output distribution and the loss unchanged. Because pairwise row differences are also unchanged, the diameter constraint places no restriction on the common translation of the update rows.

Appendix ${ \mathrm { A } } . 3$ gives an exact dual characterization of this problem. The dual is a vector-valued optimal transport problem (Ciosmak, 2021). In our setting, this transport problem introduces a ddimensional flow variable for each of the $\binom { V } { 2 }$ token pairs. This quadratic growth with vocabulary size makes solving the flow problem computationally expensive. We proceed on a quest to find a cheaper approximation.

A common-ball approximation leads to RowNorm. We ask for a tighter constraint, requiring all update rows to lie in a common closed Euclidean ball $B ( \zeta , \rho _ { \mathrm { h e a d } } / 2 )$ . With this restriction, Equation 6 becomes

$$
\operatorname* { m i n } _ { \Delta W , \zeta } \left. G , \Delta W \right. _ { F } \qquad \mathrm { s u b j e c t t o } \qquad \Delta w _ { i } \in \mathcal { B } ( \zeta , \rho _ { \mathrm { h e a d } } / 2 ) \quad \forall i .\tag{7}
$$

Since the gradient rows sum to zero, translating every update row by the same vector leaves the objective unchanged. Using this freedom, we invoke a change of variable $\widehat { \Delta W } = \Delta W - \mathbf { 1 } \zeta ^ { \top }$ which expresses each row relative to the ball’s center. Each row $\widehat { \Delta w _ { i } }$ then lies in $B ( 0 , \rho _ { \mathrm { h e a d } } / 2 )$ Equation 7 therefore has the same minimum as

$$
\operatorname* { m i n } _ { \widehat { \Delta W } } \Big \langle G , \widehat { \Delta W } \Big \rangle _ { F } \qquad \mathrm { s u b j e c t ~ t o } \qquad \widehat { \Delta w } _ { i } \in \mathcal { B } ( 0 , \rho _ { \mathrm { h e a d } } / 2 ) \quad \forall i .\tag{8}
$$

Conversely, any $\widehat { \Delta W }$ satisfying this constraint gives a feasible common-ball update by choosing $\Delta W = { \widehat { \Delta W } }$ and $\zeta = 0$

A natural curiosity would be the suboptimality gap introduced by the tighter ball constraint. Let $\Delta ,$ and $\Delta _ { \mathrm { b a l l } }$ denote the greatest first-order decreases under the diameter and common-ball constraints, respectively. These are the negatives of the minimum objective values in Equations 6 and 7. For $G \neq 0$ and $\rho _ { \mathrm { h e a d } } > 0$ , let $k = { \mathrm { { r a n k } } } ( G )$ . Jung’s theorem (Jung, 1901) bounds the radius needed to enclose a set of rows with a given diameter. Applying it to the two optimization problems bounds the relative suboptimality gap by

$$
0 \leq \frac { \Delta _ { \star } - \Delta _ { \mathrm { b a l l } } } { \Delta _ { \star } } \leq 1 - \sqrt { \frac { k + 1 } { 2 k } } .\tag{9}
$$

The bound is sharp. Appendix $\mathrm { A . 4 }$ provides its proof and elaborates on the equivalence brought about by the change of variable.

To find an update with this guarantee, we solve Equation 8 one row at a time. Each row has its own constraint and contributes independently to the objective. For a feasible row $\widehat { \Delta w _ { i } }$ , the Cauchy– Schwarz inequality and the radius constraint give

$$
\left. g _ { i } , \widehat { \Delta w } _ { i } \right. \geq - \left\| g _ { i } \right\| _ { 2 } \left\| \widehat { \Delta w } _ { i } \right\| _ { 2 } \geq - \frac { \rho _ { \mathrm { h e a d } } } { 2 } \left\| g _ { i } \right\| _ { 2 } .
$$

For $g _ { i } \neq 0 ,$ equality is attained by choosing $\widehat { \Delta w _ { i } }$ opposite to $g _ { i }$ with norm $\rho _ { \mathrm { h e a d } } / 2$ . For a zero gradient row, we choose $\widehat { \Delta w _ { i } } = 0$ . Let RN normalize each nonzero row to unit Euclidean norm and leave zero rows unchanged. Choosing $\Delta W = { \widehat { \Delta W } }$ and $\zeta = 0$ , we obtain one minimizing update for the common-ball problem,

$$
\Delta W _ { \mathrm { R N } } = - \frac { \rho _ { \mathrm { h e a d } } } { 2 } \mathsf { R N } ( G ) .\tag{10}
$$

This update is feasible for the original diameter problem and attains the common-ball optimum. Its first-order decrease is $\Delta _ { \mathrm { b a l l } }$ , so Equation 9 applies directly to RowNorm.

A practical version subtracts the mean row from $\Delta W _ { \mathrm { R N } }$ to obtain projected RowNorm. This common translation preserves the objective, the row diameter, and the output probabilities. Our Hilbert distance argument therefore independently recovers the direction derived by Lau & Su (2026) from the symmetries of the LM head. Appendix $_ { \mathrm { A . 5 } }$ gives an implementation with momentum and numerical stabilization.

## 5 THE ATTENTION MODULE

We extend our principle from the LM head to attention. Attention presents a more intricate setting: its softmax probabilities arise from bilinear scores that jointly depend on the query and key weights. Bounding the Hilbert distance between attention distributions leads us to the QK constraint underlying Compositional Muon.

Attention distributions. Attention uses softmax to distribute probability across the positions visible to each query. We study how this distribution changes when the query and key weights are updated, holding the input hidden states $h _ { 1 } , \ldots , h _ { T } \in \mathbb { R } ^ { \tilde { d } }$ fixed. For a single head with query and key dimension $d _ { h }$ , the weights $W _ { Q } , W _ { K } \in \mathbb { R } ^ { d \times d _ { h } }$ determine the attention scores through

$$
M = W _ { Q } W _ { K } ^ { \top } \in \mathbb { R } ^ { d \times d } , \qquad s _ { i j } = \frac { h _ { i } ^ { \top } M h _ { j } } { \sqrt { d _ { h } } } .
$$

We consider scaled dot-product attention without rotary position transforms or query–key normalization. Under causal masking, let $p _ { i j }$ denote the probability assigned by query i to position $j \in \mathcal { I } _ { i } = \{ 1 , \ldots , i \}$ . We measure Hilbert distance on this visible set, where all probabilities are positive.

Hilbert distance bounds changes in score differences. Let $\Delta M$ denote the change in the QK product, so that the updated product is $M + \Delta M$ . At position i, write $p _ { i } = ( p _ { i j } ) _ { j \in \mathcal { T } _ { i } }$ for the original attention distribution and $p _ { i } ^ { + }$ for the distribution after this change. Holding the hidden states fixed, the score $s _ { i j }$ changes by

$$
\Delta s _ { i j } = \frac { h _ { i } ^ { \top } ( M + \Delta M ) h _ { j } - h _ { i } ^ { \top } M h _ { j } } { \sqrt { d _ { h } } } = \frac { h _ { i } ^ { \top } \Delta M h _ { j } } { \sqrt { d _ { h } } } .
$$

Within row $i ,$ the score difference between positions $j$ and k therefore changes by $h _ { i } ^ { \top } \Delta M ( h _ { j } -$ $h _ { k } ) / \sqrt { d _ { h } }$ . Applying Equation 3 within each attention row gives

$$
\mathcal { H } ( p _ { i } , p _ { i } ^ { + } ) = \operatorname* { m a x } _ { j , k \in \mathcal { I } _ { i } } \left| \Delta s _ { i j } - \Delta s _ { i k } \right| = \frac { 1 } { \sqrt { d _ { h } } } \operatorname* { m a x } _ { j , k \in \mathcal { I } _ { i } } \left| h _ { i } ^ { \top } \Delta M ( h _ { j } - h _ { k } ) \right| .\tag{11}
$$

Intuitively, the probability constraint, specific to the attention mechanism, asks how much an update can change a query’s preference between two keys.

The induced spectral norm constraint. Assume that every hidden state satisfies $\| h _ { \ell } \| _ { 2 } \le R .$ with $R \ = \ { \sqrt { d } }$ corresponding to the RMS unit ball. The hidden state $h _ { i }$ has norm at most $R ,$ while the difference $h _ { j } \mathrm { ~ - ~ } h _ { k }$ has norm at most 2R. Consequently, $\left| h _ { i } ^ { \top } \Delta M ( h _ { j } - h _ { k } ) \right| \ \leq$ $\left. h _ { i } \right. _ { 2 } \left. \Delta M \right. _ { \mathrm { o p } } \left. h _ { j } - h _ { k } \right. _ { 2 } \leq 2 R ^ { 2 } \left. \Delta M \right. _ { \mathrm { o p } }$

For causal sequences with $T \geq 3 ,$ , the worst-case Hilbert distance over this input set is

$$
\operatorname* { s u p } _ { \| h _ { \ell } \| _ { 2 } \leq R } \operatorname* { m a x } _ { 1 \leq i \leq T } \mathcal { H } ( p _ { i } , p _ { i } ^ { + } ) = \frac { 2 R ^ { 2 } } { \sqrt { d _ { h } } } \left\| \Delta M \right\| _ { \mathrm { o p } } .\tag{12}
$$

Requiring a Hilbert distance of at most ε for every attention row throughout this input set is therefore equivalent to

$$
\left. \Delta M \right. _ { \mathrm { o p } } \leq \rho _ { \mathrm { Q K } } , \qquad \rho _ { \mathrm { Q K } } = \frac { \varepsilon \sqrt { d _ { h } } } { 2 R ^ { 2 } } .\tag{13}
$$

Why does the same Hilbert distance principle give a row diameter for the LM head and a spectral norm for attention? In the LM head, each comparison involves a single hidden state, whereas attention involves a query input and a difference between key inputs. Under the stated norm bounds, these two vectors can be aligned independently to attain the spectral norm of $\Delta M$ . The bound is attained when $h _ { i }$ is R times a unit leading left singular vector of $\Delta M$ , and $h _ { j } - h _ { k }$ is 2R times the corresponding unit right singular vector.

Steepest descent over query and key updates. Let $\Delta W _ { Q }$ and $\Delta W _ { K }$ denote updates to the query and key weights $W _ { Q }$ and $W _ { K }$ , respectively. Expanding the updated product gives

$$
\Delta M = ( W _ { Q } + \Delta W _ { Q } ) ( W _ { K } + \Delta W _ { K } ) ^ { \top } - W _ { Q } W _ { K } ^ { \top } = \underbrace { \Delta W _ { Q } W _ { K } ^ { \top } + W _ { Q } \Delta W _ { K } ^ { \top } } _ { \Delta M _ { \mathrm { i n } } } + \Delta W _ { Q } \Delta W _ { K } ^ { \top } .
$$

The first two terms give the first-order change $\Delta M _ { \mathrm { l i n } } ( \Delta W _ { Q } , \Delta W _ { K } )$ , while $\Delta W _ { Q } \Delta W _ { K } ^ { \top }$ is second order in the updates and is therefore omitted. Applying Equation 13 to this first-order change gives

$$
\left\| \Delta W _ { Q } W _ { K } ^ { \top } + W _ { Q } \Delta W _ { K } ^ { \top } \right\| _ { \mathrm { o p } } \leq \rho _ { \mathrm { Q K } } .
$$

We assume that these weights affect the loss only through the attention probabilities. Write $G _ { Q } =$ $\nabla _ { \boldsymbol { W _ { O } } } \mathcal { L }$ and $G _ { K } = \nabla _ { W _ { K } } \mathcal { L }$ for their gradients. Constructing our steepest descent problem under this Hilbert distance bound, we obtain

$$
\begin{array} { r l } { \underset { \Delta W _ { Q } , \Delta W _ { K } } { \operatorname* { m i n } } } & { { } \left. G _ { Q } , \Delta W _ { Q } \right. _ { F } + \left. G _ { K } , \Delta W _ { K } \right. _ { F } } \\ { \mathrm { s u b j e c t ~ t o } } & { { } \left\| \Delta W _ { Q } W _ { K } ^ { \top } + W _ { Q } \Delta W _ { K } ^ { \top } \right\| _ { \mathrm { o p } } \leq \rho _ { \mathrm { Q K } } . } \end{array}\tag{14}
$$

Equation 14 is precisely the first-order QK steepest descent problem posed by Keigwin et al. (2026, Section 1.2). Their derivation controls changes in the composed bilinear map through the spectral norm of its first-order perturbation. Our starting point is a bound on the Hilbert distance between attention distributions. Translating this probability bound through softmax and the bilinear scores recovers the same constraint on the first-order QK product change. Our Hilbert distance derivation therefore provides a natural prelude to Compositional Muon, motivating its QK constraint through the stability of the attention distributions.

The LM head and attention affect optimization through both forward probability distributions and the gradients propagated to the hidden states. Our main analysis bounds changes in the former and derives the corresponding steepest descent problems. Holding the inputs and incoming gradients fixed, Appendix B bounds the resulting perturbations of the latter.

## 6 DISCUSSION

Can Muon be adapted to softmax simply by removing the common shifts that softmax ignores? Because softmax is invariant to common shifts, one might try adapting Muon by centering the update rows before measuring their spectral norm. We show that the resulting steepest descent problem has the same minimum and that the Muon update is already row centered. Translation symmetry alone therefore does not alter Muon, motivating the stronger relative control provided by Hilbert distance.

Centering leaves Muon unchanged. Consider a linear module with n outputs and a parameter matrix whose influence on the loss is entirely through the softmax output distribution. Let $P =$ $I _ { n } - \mathbf { 1 1 } ^ { \top } / n$ denote the orthogonal projection that subtracts the mean row. Since gradient rows sum to zero, $P { \dot { G } } = G$ . Since $P$ is an orthogonal projection, $\begin{array} { r } { \| P \Delta W \| _ { \mathrm { o p } } \leq \| P \| _ { \mathrm { o p } } \| \Delta \tilde { W } \| _ { \mathrm { o p } } \leq \| \Delta W \| _ { \mathrm { o p } } . } \end{array}$ Thus, every update satisfying $\| \Delta W \| _ { \mathrm { o p } } \le \rho$ also satisfies $\begin{array} { r } { \| \boldsymbol { P } \boldsymbol { \Delta } \bar { \boldsymbol { W } } \| _ { \mathrm { o p } } \leq \bar { \boldsymbol { \rho } } . } \end{array}$ Conversely, take any update $\Delta W$ satisfying $\begin{array} { r } { \| P \Delta W \| _ { \mathrm { o p } } \leq \dot { \rho } . } \end{array}$ . The centered update $P \Delta W$ has spectral norm at most $\rho .$ It also preserves the objective, since $\langle G , P \Delta W \rangle _ { F } = \langle P G , \Delta W \rangle _ { F } = \langle G , \Delta W \rangle _ { F }$ . The additional freedom therefore leaves the minimum unchanged,

$$
\operatorname* { m i n } _ { \lVert P \Delta W \rVert _ { \mathrm { o p } } \leq \rho } \langle G , \Delta W \rangle _ { F } = \operatorname* { m i n } _ { \lVert \Delta W \rVert _ { \mathrm { o p } } \leq \rho } \langle G , \Delta W \rangle _ { F } .
$$

The Muon update is already centered. Indeed, if $G = A \Sigma B ^ { \top }$ is the compact SVD of a nonzero gradient, then every left singular vector satisfies $\mathbf { 1 } ^ { \top } a _ { j } = ( \mathbf { 1 } ^ { \top } G b _ { j } ) / \sigma _ { j } = 0$ , giving

$$
\Delta W _ { \mathrm { M u o n } } = - \rho A B ^ { \top } , \qquad P \Delta W _ { \mathrm { M u o n } } = - \rho ( P A ) B ^ { \top } = \Delta W _ { \mathrm { M u o n } } .
$$

Accounting for translation symmetry alone therefore recovers the standard Muon solution. Our Hilbert distance approach goes further by bounding the largest change in the difference between any two preactivation coordinates. For a centered preactivation perturbation $\Delta z \in \mathbb { R } ^ { n }$ , with $p =$ softmax(z) and $q = \mathrm { s o f t m a x } ( z + \Delta z )$ , we have $2 \left\| \Delta z \right\| _ { \mathrm { R M S } } \leq \mathcal { H } ( p , q ) \leq \sqrt { 2 n } \left\| \Delta z \right\| _ { \mathrm { R M S } } .$ . Since the upper bound is sharp, a fixed bound on centered RMS permits Hilbert distance to grow with the output width. Appendix A.1 proves these bounds and discusses their sharpness.

Limitations. The main objective of this work is to provide a unifying mathematical foundation for softmax modules. Through Hilbert distance, we independently recover RowNorm for the LM head and the first-order QK constraint underlying Compositional Muon. These mathematical connections naturally raise the question of what this perspective offers in practice. We leave that question to future experimental work and make no empirical claim of improved training speed or final model quality. The work of Lau & Su (2026) on RowNorm and that of Keigwin et al. (2026) on Compositional Muon report improvements in language-model pretraining and include ablations of their optimizer choices. These experiments provide existing empirical evidence for the practical relevance of the methods connected by our framework.

A second limitation arises from the relative nature of Hilbert distance. Because it controls pairwise odds, it can be sensitive to large relative changes in low probability outcomes even when little probability mass moves. For example, suppose an update raises the probability of one outcome from $1 0 ^ { - 5 } ~ \mathrm { t o } ~ 1 0 ^ { - 6 }$ while leaving the rest of the distribution nearly unchanged. The probability of this outcome increases by a factor of 1000, producing a large Hilbert distance despite the small movement of probability mass, $\mathrm { T V } ( p , q ) = 9 . 9 \dot { 9 } \times 1 0 ^ { - \bar { 7 } } , \mathcal { H } ( \bar { p , q } ) \approx 6 . 9 1$ . A small Hilbert distance constraint can therefore exclude updates that move very little probability mass. Future work could investigate metrics that preserve the useful guarantees developed here while being less sensitive to low probability outcomes.

## 7 RELATED WORK

From Adam to module specific optimizers. Adam has long served as the default optimizer across deep learning (Kingma & Ba, 2014). Muon has recently challenged this convention (Jordan et al., 2024). For a linear layer, Muon controls the worst-case change in its output as measured by RMS, resulting in a bound on the spectral norm of the weight update. The resulting steepest descent problem forms the mathematical foundation of the Muon update (Bernstein, 2025a). The Modula project generalizes this perspective and advocates constructing optimizers whose geometry reflects the function of each module (Bernstein, 2025b; Large et al., 2024). Modular duality and Scion provide systematic ways to construct updates from such geometries (Bernstein & Newhouse, 2024; Pethick et al., 2025). We extend this principle to modules involving softmax.

Geometry on probability distributions. A related line of work connects parameter updates to the probability distributions they induce. Natural gradient equips a family of probability distributions with the Fisher geometry and chooses updates according to the resulting notion of steepest descent (Amari, 1998; Martens, 2020). TRPO makes this connection explicit by constraining the expected KL divergence between the action distributions induced by the original and updated policies (Schulman et al., 2015). Hilbert’s projective metric offers a different geometry for positive vectors. Beginning with the contraction theory of positive operators, this geometry has subsequently appeared in matrix scaling, clustering, nonlinear embeddings, and the study of probability simplices (Birkhoff, 1957; De La Harpe, 1993; Cuturi, 2013; Nielsen & Sun, 2018; 2023). More recently, Cohen & Fausti (2024) established a sharp relationship between Hilbert distance and total variation. We draw on this literature to study the parameter geometry induced by Hilbert distance on the probability distributions produced by softmax.

Optimizers for the LM head and attention. The LM head and attention have each motivated specialized optimization methods. For the LM head, Scion (Pethick et al., 2025) includes RowNorm and SCALE (Glentis et al., 2026) proposes a momentum-based variant; Lau & Su (2026) derive projected RowNorm from its permutation and shared-shift symmetries. Attention has motivated a different collection of interventions. Query and key normalization controls the scale of attention scores, while spectral reparameterization addresses attention entropy collapse (Henry et al., 2020; Zhai et al., 2023). MuonClip monitors the maximum attention logits and conditionally rescales the query and key projection weights after a Muon update (Team et al., 2026). Compositional Muon goes further by constraining the first-order change in the product of the query and key matrices (Keigwin et al., 2026). Although row normalization for the LM head and product control for attention appear to be independent design choices, we show that they share common ground in bounding changes in the probability distributions produced by softmax.

## 8 CONCLUSION

We develop a unified framework for optimizing composed softmax modules by bounding changes in their output probabilities and deriving the corresponding parameter constraints. This addresses a limitation of Muon’s RMS control, which can conceal substantial probability redistribution when applied to the intermediate linear output. We propose using Hilbert distance to constrain these changes in the output distribution directly. Hilbert distance respects the common-shift invariance of softmax and controls both the redistribution of probability mass and changes in cross-entropy loss. Its representation as the variation seminorm of the preactivation perturbation lets us translate a common constraint on the output distribution into a parameter geometry for each module. For the LM head, this gives a row diameter constraint. A tighter ball approximation leads to RowNorm, with centering giving its projected form. For attention, the same principle recovers the first-order QK steepest descent problem underlying Compositional Muon. A single probability constraint thus connects the separately developed RowNorm update and Compositional Muon’s QK optimization problem. We offer this framework as a contribution to the broader Modula perspective, demonstrating how the output geometry of composed softmax modules can guide optimizer design. We hope these connections spark new ideas for optimizers and architectures built around composed modules. Hilbert distance provides a starting point for softmax, and we encourage researchers to ask which output geometries could play that role for other compositions.

## AI USE STATEMENT

The idea of bounding changes in softmax output probabilities originated with the authors. The choice of Hilbert distance was found by studying many existing distances in the literature. AIassisted mathematical exploration also informed the selection of Hilbert distance. AI tools were extensively used to survey literature, identify related work, check mathematical arguments, and verify correctness. In particular, the authors initially obtained a weaker approximation bound for RowNorm relative to the diameter-constrained optimum. AI tools subsequently identified Jung’s theorem and its applicability in tightening this bound.

The initial manuscript was written by the authors. AI tools were subsequently used to improve wording, technical exposition, and presentation, and to provide critical feedback on the manuscript. The authors reviewed all AI assisted editorial changes before incorporating them. We, the authors, take responsibility for the final content of the paper, including its mathematical claims, proofs, and references.

## ETHICS STATEMENT

This work develops a unified framework for optimizing modules which involve softmax. The research presented here does not involve human participants, personal data, or the collection or release of datasets. The framework is general purpose, and all ethical considerations in downstream applications depend on the training data, models, and deployment contexts in which it is used.

## REPRODUCIBILITY STATEMENT

The assumptions underlying our theoretical results are stated alongside the derivations in the main text. Appendix A provides the sharpness argument for the total variation bound, the exact dual formulation of the diameter-constrained problem, and the proof of the sharp RowNorm approximation guarantee. Appendix B derives the forward and backward perturbation bounds for the LM head and attention. Appendix A.5 specifies the practical RowNorm update, including centering, momentum, and numerical stabilization. To reproduce Figure 1, Appendix C gives deterministic formulas for the initial logits and all three updates, together with their RMS norms, Hilbert distances, and total variation values.

## AUTHOR CONTRIBUTIONS

If you’d like to, you may include a section for author contributions as is done in many journals. This is optional and at the discretion of the authors.

## ACKNOWLEDGMENTS

Use unnumbered third level headings for the acknowledgments. All acknowledgments, including those to funding agencies, go at the end of the paper.

## REFERENCES

Shun-Ichi Amari. Natural gradient works efficiently in learning. Neural computation, 10(2):251– 276, 1998.

Jeremy Bernstein. Deriving muon, 2025a. URL https://jeremybernste.in/writing/ deriving-muon.

Jeremy Bernstein. The modula docs, 2025b. URL https://docs.modula.systems/.

Jeremy Bernstein and Laker Newhouse. Modular duality in deep learning. arXiv preprint arXiv:2410.21265, 2024.

Garrett Birkhoff. Extensions of jentzsch’s theorem. Transactions of the American Mathematical Society, 85(1):219–227, 1957.

Krzysztof J Ciosmak. Optimal transport of vector measures. Calculus of Variations and Partial Differential Equations, 60(6):230, 2021.

Samuel N. Cohen and Eliana Fausti. Hyperbolic contractivity and the hilbert metric on probability measures, 2024. URL https://arxiv.org/abs/2309.02413.

Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013.

Pierre De La Harpe. On hilbert’s metric for simplices. Geometric group theory, 1:97–119, 1993.

Athanasios Glentis, Jiaxiang Li, Andi Han, and Mingyi Hong. Memory-efficient llm pretraining via minimalist optimizer design, 2026. URL https://arxiv.org/abs/2506.16659.

Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 4246–4253, 2020.

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https: //kellerjordan.github.io/posts/muon/.

Heinrich Jung. Ueber die kleinste kugel, die eine raumliche figur einschliesst.¨ Journalfur die reine¨ und angewandte Mathematik (Crelles Journal), 1901(123):241–257, 1901.

Ben Keigwin, Li Yang, Yunzhe Zhang, Dhruv Pai, and Alec DeWulf. Towards compositional steepest descent. 2026. URL https://tilderesearch.com/blog/ compositional-muon.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Tim Large, Yang Liu, Minyoung Huh, Hyojin Bahng, Phillip Isola, and Jeremy Bernstein. Scalable optimization in the modular norm, 2024. URL https://arxiv.org/abs/2405.14813.

Tim Tsz-Kit Lau and Weijie Su. Symmetry-compatible principle for optimizer design: Embeddings, lm heads, swiglu mlps, and moe routers, 2026. URL https://arxiv.org/abs/2605. 18106.

Bas Lemmens and Roger D Nussbaum. Nonlinear Perron-Frobenius Theory. Cambridge University Press, 2012.

James Martens. New insights and perspectives on the natural gradient method. Journal ofMachine Learning Research, 21(146):1–76, 2020.

Frank Nielsen and Ke Sun. Clustering in Hilbert’s Projective Geometry: The Case Studies of the Probability Simplex and the Elliptope of Correlation Matrices, pp. 297–331. Springer International Publishing, November 2018. ISBN 9783030025205. doi: 10.1007/978-3-030-02520-5 11. URL http://dx.doi.org/10.1007/978-3-030-02520-5\_11.

Frank Nielsen and Ke Sun. Non-linear embeddings in hilbert simplex geometry. In Topological, Algebraic and Geometric Learning Workshops 2023, pp. 254–266. PMLR, 2023.

Athanase Papadopoulos and Marc Troyanov. Handbook ofHilbert geometry. European Mathematical Society, 2014.

Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, and Volkan Cevher. Training deep learning models with norm-constrained lmos, 2025. URL https: //arxiv.org/abs/2502.07529.

Herbert Robbins and Sutton Monro. A stochastic approximation method. The annals of mathematical statistics, pp. 400–407, 1951.

Walter Rudin. Principles ofMathematical Analysis. International Series in Pure and Applied Mathematics. McGraw-Hill, 3 edition, 1976. URL https://books.google.com/books?id= kwqzPAAACAAJ.

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International conference on machine learning, pp. 1889–1897. Pmlr, 2015.

Kimi Team, Yifan Bai, Yiping Bao, Y. Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, Zhuofu Chen, Jialei Cui, Hao Ding, Mengnan Dong, Angang Du, Chenzhuang Du, Dikang Du, Yulun Du, Yu Fan, Yichen Feng, Kelin Fu, Bofei Gao, Chenxiao Gao, Hongcheng Gao, Peizhong Gao, Tong Gao, Yuyao Ge, Shangyi Geng, Qizheng Gu, Xinran Gu, Longyu Guan, Haiqing Guo, Jianhang Guo, Xiaoru Hao, Tianhong He, Weiran He, Wenyang He, Yunjia He, Chao Hong, Hao Hu, Yangyang Hu, Zhenxing Hu, Weixiao Huang, Zhiqi Huang, Zihao Huang, Tao Jiang, Zhejun Jiang, Xinyi Jin, Yongsheng Kang, Guokun Lai, Cheng Li, Fang Li, Haoyang Li, Ming Li, Wentao Li, Yang Li, Yanhao Li, Yiwei Li, Zhaowei Li, Zheming Li, Hongzhan Lin, Xiaohan Lin, Zongyu Lin, Chengyin Liu, Chenyu Liu, Hongzhang Liu, Jingyuan Liu, Junqi Liu, Liang Liu, Shaowei Liu, T. Y. Liu, Tianwei Liu, Weizhou Liu, Yangyang Liu, Yibo Liu, Yiping Liu, Yue Liu, Zhengying Liu, Enzhe Lu, Haoyu Lu, Lijun Lu, Yashuo Luo, Shengling Ma, Xinyu Ma, Yingwei Ma, Shaoguang Mao, Jie Mei, Xin Men, Yibo Miao, Siyuan Pan, Yebo Peng, Ruoyu Qin, Zeyu Qin, Bowen Qu, Zeyu Shang, Lidong Shi, Shengyuan Shi, Feifan Song, Jianlin Su, Zhengyuan Su, Lin Sui, Xinjie Sun, Flood Sung, Yunpeng Tai, Heyi Tang, Jiawen Tao, Qifeng Teng, Chaoran Tian, Chensi Wang, Dinglu Wang, Feng Wang, Hailong Wang, Haiming Wang, Jianzhou Wang, Jiaxing Wang, Jinhong Wang, Shengjie Wang, Shuyi Wang, Si Wang, Xinyuan Wang, Yao Wang, Yejie Wang, Yiqin Wang, Yuxin Wang, Yuzhi Wang, Zhaoji Wang, Zhengtao Wang, Zhengtao Wang, Zhexu Wang, Chu Wei, Qianqian Wei, Haoning Wu, Wenhao Wu, Xingzhe Wu, Yuxin Wu, Chenjun Xiao, Jin Xie, Xiaotong Xie, Weimin Xiong, Boyu Xu, Jinjing Xu, L. H. Xu, Lin Xu, Suting Xu, Weixin Xu, Xinran Xu, Yangchuan Xu, Ziyao Xu, Jing Xu, Jing Xu, Junjie Yan, Yuzi Yan, Hao Yang, Xiaofei Yang, Yi Yang, Ying Yang, Zhen Yang, Zhilin Yang, Zonghan Yang, Haotian Yao, Xingcheng Yao, Wenjie Ye, Zhuorui Ye, Bohong Yin, Longhui Yu, Enming Yuan, Hongbang Yuan, Mengjie Yuan, Siyu Yuan, Haobing Zhan, Dehao Zhang, Hao Zhang, Wanlu Zhang, Xiaobin Zhang, Yadong Zhang, Yangkun Zhang, Yichi Zhang, Yizhi Zhang, Yongting Zhang, Yu Zhang, Yutao Zhang, Yutong Zhang, Zheng Zhang, Haotian Zhao, Yikai Zhao, Zijia Zhao, Huabin Zheng, Shaojie Zheng, Longguang Zhong, Jianren Zhou, Xinyu Zhou, Zaida Zhou, Jinguo Zhu, Zhen Zhu, Weiyu Zhuang, and Xinxing Zu. Kimi k2: Open agentic intelligence, 2026. URL https://arxiv.org/abs/2507.20534.

Lieven Vandenberghe and Stephen Boyd. Convex optimization, volume 1. Cambridge university press Cambridge, 2004.

Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, and Joshua M Susskind. Stabilizing transformer training by preventing attention entropy collapse. In International conference on machine learning, pp. 40770–40803. PMLR, 2023.

## A PROOFS AND FURTHER CONNECTIONS

## A.1 COMPARING CENTERED RMS AND HILBERT DISTANCE

This subsection proves the comparison used in Section 6. For centered preactivation perturbations, Hilbert distance bounds RMS independently of the output width, whereas the reverse bound necessarily grows with the width.

Let $z \in \mathbb { R } ^ { n }$ be a preactivation vector and let $\Delta z \in \mathbb { R } ^ { n }$ satisfy $\mathbf { 1 } ^ { \top } \Delta z = 0$ . Define $p = \operatorname { s o f t m a x } ( z )$ $q = \mathrm { s o f t m a x } ( z + \Delta z )$ , and $\left\| \Delta z \right\| _ { \mathrm { R M S } } = \left\| \Delta z \right\| _ { 2 } / \sqrt { n }$ . We prove

$$
2 \left\| \Delta z \right\| _ { \mathrm { R M S } } \leq \mathcal { H } ( p , q ) \leq \sqrt { 2 n } \left\| \Delta z \right\| _ { \mathrm { R M S } } .\tag{15}
$$

The softmax identity gives $\begin{array} { r } { \mathcal { H } ( p , q ) = \operatorname* { m a x } _ { i , j } | \Delta z _ { i } - \Delta z _ { j } | } \end{array}$ . Applying the Cauchy–Schwarz inequality gives

$$
| \Delta z _ { i } - \Delta z _ { j } | \leq \sqrt { 2 } \sqrt { \Delta z _ { i } ^ { 2 } + \Delta z _ { j } ^ { 2 } } \leq \sqrt { 2 \sum _ { k = 1 } ^ { n } \Delta z _ { k } ^ { 2 } } = \sqrt { 2 n } \left\| \Delta z \right\| _ { \mathrm { R M S } } , \qquad i \neq j .
$$

Taking the maximum over i and $j$ proves the upper bound. Equality is attained by $\begin{array} { r l } { \Delta z } & { { } = } \end{array}$ $( a , - a , 0 , \ldots , 0 )$ for any $a > 0$ , so the factor $\sqrt { 2 n }$ is sharp.

For the lower bound, let $m = \mathrm { m i n } _ { i } \Delta z _ { i } , M = \mathrm { m a x } _ { i } \Delta z _ { i }$ , and $c = ( m + M ) / 2$ . Since $\begin{array} { r } { \sum _ { i } \Delta z _ { i } = 0 } \end{array}$

$$
\left\| { \Delta z } \right\| _ { \mathrm { R M S } } ^ { 2 } \leq \left\| { \Delta z } \right\| _ { \mathrm { R M S } } ^ { 2 } + c ^ { 2 } = \frac { 1 } { n } \sum _ { i } ( \Delta z _ { i } - c ) ^ { 2 } \leq \operatorname* { m a x } _ { i } ( \Delta z _ { i } - c ) ^ { 2 } = \left( \frac { M - m } { 2 } \right) ^ { 2 } = \frac { \mathcal { H } ( p , q ) ^ { 2 } } { 4 } .
$$

Dropping the nonnegative term $c ^ { 2 }$ proves the lower bound. When n is even, equality is attained by taking half the coordinates equal to a and half equal $\mathbf { t o } - a$

Equation 15 reveals an asymmetry between the two measures. Hilbert distance bounds centered RMS with a constant independent of $n ,$ , but a fixed RMS bound permits Hilbert distance to grow as $\sqrt { n }$ . A similar dependence appears in the standard comparison between the spectral and Frobenius norms: for a matrix A of rank r,

$$
\begin{array} { r } { \| A \| _ { \mathrm { o p } } \leq \| A \| _ { F } \leq \sqrt { r } \ \| A \| _ { \mathrm { o p } } . } \end{array}
$$

In both comparisons, an aggregate Euclidean measure can conceal an extremal change whose magnitude grows with dimension or rank. Appendix $\mathrm { A } . 2$ shows that, for softmax, this separation can persist at the level of the output distribution: centered RMS can vanish while total variation approaches $1 / 2$

## A.2 PROBABILITY REDISTRIBUTION UNDER SOFTMAX

This subsection establishes two consequences of the Hilbert geometry developed in the main text. First, we show that the total variation bound is sharp for every fixed preactivation perturbation. We then construct centered perturbations whose RMS tends to zero while their probability redistribution remains bounded away from zero as the output width grows.

The sharp total variation bound. Recall that the variation seminorm is $\begin{array} { r c l } { \gamma } & { = } & { \| \Delta z \| _ { \mathrm { v a r } } } & { = } \end{array}$ max $\Delta z _ { \ell } - \mathrm { m i n } _ { \ell } \Delta z _ { \ell }$ . For $p = \operatorname { s o f t m a x } ( z )$ and $q = \mathrm { s o f t m a x } ( z + \Delta z )$ , Equation 3 and the sharp bound of Cohen & Fausti $( 2 0 2 4$ , Theorem 5.1) give $\mathrm { T V } ( p , \dot { q } ) \leq$ tanh $( \gamma / 4 )$ . To approach equality when $\gamma > 0 .$ , choose $i , j$ with $\Delta z _ { i } - \Delta z _ { j } = \gamma$ and set $z _ { i } = - \gamma / 2 , z _ { j } = 0$ . The difference between these preactivation coordinates therefore changes from $- \gamma / 2$ to $\gamma / 2 .$ Let all other preactivations tend $\mathrm { t o } - \infty$ , concentrating both distributions on $i , j$ . The probability gained by coordinate i then gives

$$
\mathrm { T V } ( p , q ) \longrightarrow \frac { e ^ { \gamma / 2 } } { 1 + e ^ { \gamma / 2 } } - \frac { 1 } { 1 + e ^ { \gamma / 2 } } = \operatorname { t a n h } \Bigl ( \frac { \gamma } { 4 } \Bigr ) .
$$

This approaches the upper bound through strictly positive softmax distributions, proving the claimed supremum. For $\gamma = 0$ , the update is a common shift and both sides are zero.

Vanishing RMS with nonvanishing probability redistribution. We now show that centering alone does not make RMS a uniform measure of probability change. Take $n \geq 3$ , initial preactivation vector $z = 0$ , and $a _ { n } = \log ( n - 1 )$ ). The initial probabilities and centered update are

$$
p = { \frac { \mathbf { 1 } } { n } } , \qquad \Delta z = a _ { n } \left( 1 - { \frac { 1 } { n } } , - { \frac { 1 } { n } } , \ldots , - { \frac { 1 } { n } } \right) ^ { \top } , \qquad { \mathbf { 1 } } ^ { \top } \Delta z = 0 .
$$

By softmax’s common-shift invariance, $q = \mathrm { s o f t m a x } ( \Delta z )$ satisfies

$$
q _ { 1 } = \frac { e ^ { a _ { n } } } { e ^ { a _ { n } } + n - 1 } = \frac { 1 } { 2 } , \qquad q _ { i } = \frac { 1 } { e ^ { a _ { n } } + n - 1 } = \frac { 1 } { 2 ( n - 1 ) } \quad ( i > 1 ) .
$$

Yet, as $n \to \infty$

$$
\left\| \Delta z \right\| _ { \mathrm { R M S } } = \frac { \sqrt { n - 1 } } { n } \log ( n - 1 ) \longrightarrow 0 , \qquad \mathrm { T V } ( p , q ) = \frac { 1 } { 2 } - \frac { 1 } { n } \longrightarrow \frac { 1 } { 2 } .
$$

Even after centering, vanishing RMS does not ensure vanishing probability redistribution uniformly across widths. Meanwhile, $\mathcal { H } ( p , q ) = \log ( n - 1 )$ captures the growing relative change.

## A.3 THE EXACT DUAL OF THE DIAMETER CONSTRAINT

This subsection derives the exact dual of the row diameter problem in Equation $6 .$ The dual represents each gradient row through antisymmetric vectors assigned to pairs of vocabulary rows and minimizes their total Euclidean norm. We first establish the balance constraints and the dual identity, then relate the resulting problem to vector transport.

Let $G \in \mathbb { R } ^ { V \times d }$ have rows $g _ { i } ^ { \top }$ , where $V \geq 2$ and $\textstyle \sum _ { i } g _ { i } = 0$ . For $\rho _ { \mathrm { h e a d } } > 0 .$ , consider

$$
\operatorname* { m i n } _ { \Delta W } \langle G , \Delta W \rangle _ { F } \qquad \mathrm { s u b j e c t ~ t o } \qquad \| \Delta w _ { i } - \Delta w _ { j } \| _ { 2 } \leq \rho _ { \mathrm { h e a d } } \quad \forall i < j .\tag{16}
$$

Pairwise dual variables. For every pair $i < j ,$ , introduce a vector $\varphi _ { i j } \in \mathbb { R } ^ { d }$ . Extend this collection by setting $\varphi _ { j i } = - \varphi _ { i j }$ and $\varphi _ { i i } = 0$ . The dual feasibility conditions are

$$
\sum _ { j = 1 } ^ { V } \varphi _ { i j } = g _ { i } , \qquad i = 1 , \ldots , V .\tag{17}
$$

Antisymmetry implies

$$
\sum _ { i } \sum _ { j } \varphi _ { i j } = \sum _ { i < j } ( \varphi _ { i j } + \varphi _ { j i } ) = 0 .
$$

Hence, the condition $\textstyle \sum _ { i } g _ { i } = 0$ is necessary for Equation 17. It is also sufficient. For example, setting $\varphi _ { i V } ~ = ~ g _ { i }$ for $i \ < \ V$ and every other independently indexed variable to zero satisfies all balance equations because $\begin{array} { r } { g _ { V } = - \sum _ { i < V } g _ { i } } \end{array}$

The exact dual identity. The primal problem in Equation 16 satisfies

$$
\operatorname* { m i n } _ { D ( \Delta W ) \leq \rho _ { \mathrm { h e a d } } }  G , \Delta W  _ { F } = - \rho _ { \mathrm { h e a d } } \operatorname* { m i n } _ { \Phi } \sum _ { i \mathrm { ~ } \varphi \textit { i j } = g _ { i } }  _ { \forall i }   \varphi _ { i j }   _ { 2 } .\tag{18}
$$

Both minima are attained. Consequently, the greatest first order decrease under the diameter constraint is

$$
\Delta _ { \star } = \rho _ { \mathrm { h e a d } } \operatorname* { m i n } _ { \begin{array} { c } { \Phi } \\ { \sum _ { j } \varphi _ { i j } = g _ { i } \forall i } \end{array} } \sum _ { i < j } \left\| \varphi _ { i j } \right\| _ { 2 } .
$$

Proof. We first establish attainment of the primal minimum. Given any feasible $\Delta W$ , subtract $\Delta w _ { V }$ from every update row. This translation preserves all pairwise differences and changes the objective by

$$
- \left. \sum _ { i } g _ { i } , \Delta w _ { V } \right. = 0 .
$$

We may therefore restrict attention to representatives satisfying $\Delta w _ { V } = 0$ . Feasibility then gives $\lVert \Delta w _ { i } \rVert _ { 2 } \leq \rho _ { \mathrm { h e a d } }$ for every $i \ < \ V$ . The resulting feasible set is nonempty and compact, so the continuous objective attains a finite minimum.

To derive the dual, introduce an auxiliary variable $\xi _ { i j }$ ffor each pair $i < j$ and impose

$$
\xi _ { i j } = \Delta w _ { i } - \Delta w _ { j } , \qquad \lVert \xi _ { i j } \rVert _ { 2 } \leq \rho _ { \mathrm { h e a d } } , \qquad i < j .
$$

Associate the multiplier $\varphi _ { i j }$ with the equality constraint $\xi _ { i j } - \Delta w _ { i } + \Delta w _ { j } = 0$ . The Lagrangian is

$$
\begin{array} { r } { \Lambda ( \Delta W , \xi , \Phi ) = \langle G , \Delta W \rangle _ { F } + \displaystyle \sum _ { i < j } \langle \varphi _ { i j } , \xi _ { i j } - \Delta w _ { i } + \Delta w _ { j } \rangle } \\ { = \displaystyle \sum _ { i = 1 } ^ { V } \left. g _ { i } - \sum _ { j = 1 } ^ { V } \varphi _ { i j } , \Delta w _ { i } \right. + \displaystyle \sum _ { i < j } \langle \varphi _ { i j } , \xi _ { i j } \rangle . } \end{array}
$$

The rows of $\Delta W$ are unrestricted in the inner minimization. Its infimum is therefore finite only if $\textstyle \sum _ { j } \varphi _ { i j } = g _ { i }$ for all i. Under these balance equations, the terms containing $\Delta W$ vanish. Minimizing over the Euclidean balls gives

$$
\begin{array} { c } { \displaystyle \operatorname* { i n f } _ { \boldsymbol { \xi } } \sum _ { i < j } \left. \varphi _ { i j } , \xi _ { i j } \right. = - \rho _ { \mathrm { h e a d } } \sum _ { i < j } \left\| \varphi _ { i j } \right\| _ { 2 } . } \\ { \| \xi _ { i j } \| _ { 2 } \leq \rho _ { \mathrm { h e a d } } \forall i < j } \end{array}
$$

For $\varphi _ { i j } \neq 0 .$ , the infimum for that pair is attained at

$$
\xi _ { i j } = - \rho _ { \mathrm { h e a d } } \frac { \varphi _ { i j } } { \left\| \varphi _ { i j } \right\| _ { 2 } } ;
$$

when $\varphi _ { i j } = 0 $ , any feasible $\xi _ { i j }$ is optimal. The Lagrange dual therefore maximizes

$$
- \rho _ { \mathrm { h e a d } } \sum _ { i < j } \| \varphi _ { i j } \| _ { 2 }
$$

subject to Equation 17. Since $- \rho _ { \mathrm { h e a d } } < 0$ , this is equivalent to the minimization on the right hand side of Equation 18.

The reformulated primal problem is convex. Moreover, $\Delta W = 0$ and $\xi _ { i j } = 0$ satisfy every equality constraint and lie strictly inside every Euclidean ball. Slater’s condition therefore gives strong duality and attainment of the dual optimum (Vandenberghe & Boyd, 2004, Section 5.2.3). This proves Equation 18. □

Connection to vector transport. Let $\mathcal { X } = \{ 1 , \ldots , V \}$ carry the discrete metric $d ( i , j ) = 1$ for $i \neq j$ , and define the vector measure

$$
\mu = \sum _ { i = 1 } ^ { V } g _ { i } \delta _ { i } .
$$

The condition $\textstyle \sum _ { i } g _ { i } = 0$ gives $\mu ( \mathcal { X } ) = 0$ . Equation 17 represents this measure through an antisymmetric vector flow on the complete graph, with transport cost

$$
\sum _ { i < j } \| \varphi _ { i j } \| _ { 2 } .
$$

The primal variables provide the corresponding Lipschitz functions. In particular, define

$$
u _ { i } = - \frac { \Delta w _ { i } } { \rho _ { \mathrm { h e a d } } } .
$$

The diameter constraint implies

$$
\begin{array} { r } { \| u _ { i } - u _ { j } \| _ { 2 } \leq 1 = d ( i , j ) \qquad \forall i \neq j , } \end{array}
$$

so $u : \mathcal { X }  \mathbb { R } ^ { d }$ is 1 Lipschitz. Equation 18 is therefore the finite discrete form of vector transport duality (Ciosmak, 2021, Theorem 2).

This dual formulation characterizes the optimal value without requiring the primal minimizer to be unique. It contains one vector in $\mathbb { R } ^ { d }$ for each of the $\binom { V } { 2 }$ unordered pairs of vocabulary rows. The resulting quadratic growth in the number of dual variables motivates the common ball approximation studied next.

## A.4 THE ROWNORM APPROXIMATION

The main text obtains RowNorm by requiring all update rows to lie in a common ball and then moving that ball to the origin. We first justify why this change of variable preserves the minimum. We then bound the suboptimality introduced by replacing the diameter constraint with the ball constraint.

## A.4.1 CENTERING THE ENCLOSING BALL PRESERVES THE OPTIMUM

Fix a gradient $G \in \mathbb { R } ^ { V \times d }$ with rows $g _ { i } ^ { \top }$ satisfying $\textstyle \sum _ { i } g _ { i } = 0$ , and let $\rho _ { \mathrm { h e a d } } > 0$ . Recall that $B ( \zeta , \rho _ { \mathrm { h e a d } } / 2 )$ denotes the closed Euclidean ball with center $\zeta$ and radius $\rho _ { \mathrm { h e a d } } / 2 .$ . In Equation $^ { 7 , }$ we choose both the update $\Delta W$ and the center $\zeta ,$

$$
\operatorname* { m i n } _ { \Delta W , \zeta } \left. G , \Delta W \right. _ { F } \qquad \mathrm { s u b j e c t t o } \qquad \Delta w _ { i } \in \mathcal { B } ( \zeta , \rho _ { \mathrm { h e a d } } / 2 ) \quad \forall i .
$$

Any center that encloses all the update rows is allowed. In particular, the center is not required to equal the mean row, and the mean row remains unrestricted.

Since the gradient rows sum to zero, translating every update row by the same vector leaves the objective unchanged. Using this freedom, we invoke a change of variable

$$
\widehat { \Delta W } = \Delta W - \mathbf { 1 } \zeta ^ { \top } .
$$

For any feasible pair $( \Delta W , \zeta )$ , the translated rows satisfy

$$
\left\| \widehat { \Delta w } _ { i } \right\| _ { 2 } = \| \Delta w _ { i } - \zeta \| _ { 2 } \leq \frac { \rho _ { \mathrm { h e a d } } } { 2 } \qquad \forall i .
$$

Thus, every row now lies in a ball centered at the origin. The objective is preserved because

$$
\left. G , \Delta W \right. _ { F } = \left. G , \widehat { \Delta W } \right. _ { F } + \underbrace { \left. \sum _ { i } g _ { i } , \zeta \right. } _ { = 0 } = \left. G , \widehat { \Delta W } \right. _ { F } .
$$

After this change of variable, neither the row constraints nor the objective depends on $\zeta .$ Equation 7 therefore has the same minimum as

$$
\operatorname* { m i n } _ { \widehat { \Delta W } } \Big \langle G , \widehat { \Delta W } \Big \rangle _ { F } \qquad \mathrm { s u b j e c t ~ t o } \qquad \widehat { \Delta w } _ { i } \in \mathcal { B } ( 0 , \rho _ { \mathrm { h e a d } } / 2 ) \quad \forall i ,
$$

which is Equation 8. To complete the equivalence, we also need to return from this problem to the original one. Given any feasible $\widehat { \Delta W }$ , choose any $\zeta \in \mathbb { R } ^ { d }$ and set

$$
\Delta W = \widehat { \Delta W } + \mathbf { 1 } \zeta ^ { \top } .
$$

The resulting rows lie in $B ( \zeta , \rho _ { \mathrm { h e a d } } / 2 )$ , and the same objective identity shows that their value is unchanged. Choosing $\zeta = 0$ simply gives $\Delta W = { \widehat { \Delta W } }$ . The two problems therefore attain exactly the same objective values.

The feasible set for the problem centered at the origin is a finite product of closed, bounded balls. It is nonempty and compact, so the continuous linear objective attains its minimum. The reverse construction then provides a minimizing pair for the original problem as well. Consequently, every minimizer of the problem centered at the origin gives a minimizer of the common-ball problem, and every minimizing pair of the common ball problem translates back to one.

This change of variable centers the enclosing ball. The rows inside it can still have a nonzero mean. The RowNorm update in Equation 10 is therefore already one minimizing choice for the common ball problem, with $\zeta = 0 .$ . Subtracting its mean row is optional and preserves the objective by the same translation argument.

Moving the ball to the origin introduces no additional suboptimality. The gap considered next comes entirely from requiring the update rows to fit inside a ball of radius $\rho _ { \mathrm { h e a d } } / 2$ instead of imposing only the original diameter constraint.

## A.4.2 APPROXIMATION GUARANTEE AND SHARPNESS

Moving the ball to the origin preserves the minimum. We now bound the first-order decrease lost by replacing the diameter constraint with the common-ball constraint. Assume $G \neq 0 , \mathbf { 1 } ^ { \top } G = 0$ , and ρ<sub>head</sub> > 0. Let $\Delta _ { \cdot }$ <sub>⋆</sub> and $\Delta _ { \mathrm { b a l l } }$ denote the greatest first-order decreases under these two constraints, respectively. Since RowNorm attains the common-ball optimum,

$$
\Delta _ { \mathrm { b a l l } } = \frac { \rho _ { \mathrm { h e a d } } } { 2 } \sum _ { i } \| g _ { i } \| _ { 2 } > 0 .
$$

Bounding the gap. Jung’s theorem bounds the radius needed to enclose a set using its diameter and dimension. Here we can work in the space spanned by the gradient rows, $s \ =$ span $\left\{ g _ { 1 } , \ldots , g _ { V } \right\}$ . By definition of matrix rank, the dimension of this row space is $k = { \mathrm { r a n k } } ( G )$

To see why this space suffices, consider the part of an update row perpendicular to ${ \mathcal { S } } .$ . It has zero inner product with every gradient row and therefore contributes nothing to the objective. Removing these parts from all rows is an orthogonal projection, which cannot increase pairwise distances. Every feasible update can thus be replaced by one whose rows lie in S, with the same objective value and no larger diameter.

Jung’s theorem (Jung, 1901) guarantees that any such rows with diameter at most $\rho _ { \mathrm { h e a d } }$ lie in a ball of radius at most

$$
r _ { k } = \rho _ { \mathrm { h e a d } } { \sqrt { \frac { k } { 2 ( k + 1 ) } } } .
$$

Let $\zeta$ be the center of such a ball, so that $\| \Delta w _ { i } - \zeta \| _ { 2 } \leq r _ { k }$ for every row. We only need to know that this center exists. As in the preceding subsection, $\textstyle \sum _ { i } g _ { i } = 0$ allows us to subtract $\zeta$ without changing the objective. Cauchy–Schwarz then gives

$$
- \left. G , \Delta W \right. _ { F } = - \sum _ { i } \left. g _ { i } , \Delta w _ { i } - \zeta \right. \leq r _ { k } \sum _ { i } \left\| g _ { i } \right\| _ { 2 } .
$$

This bounds $\Delta _ { \star }$ because every feasible update can be brought into $s$ without changing its objective value. Every common-ball update also satisfies the diameter constraint, so

$$
0 < \Delta _ { \mathrm { b a l l } } \leq \Delta _ { \star } \leq \sqrt { \frac { 2 k } { k + 1 } } \Delta _ { \mathrm { b a l l } } .
$$

Rearranging proves the relative suboptimality bound in Equation $9 .$

Sharpness. A regular simplex gives a gradient for which equality holds. Choose $g _ { 1 } , \ldots , g _ { k + 1 }$ as the unit vertices of a regular k-dimensional simplex centered at the origin in $\mathbb { R } ^ { d }$ . For a vocabulary with $V > k + 1$ , set $g _ { i } = 0$ for every $i > k + 1$ . The simplex rows satisfy

$$
\sum _ { i = 1 } ^ { k + 1 } g _ { i } = 0 , \qquad \left\| g _ { i } \right\| _ { 2 } = 1 , \qquad \langle g _ { i } , g _ { j } \rangle = - \frac { 1 } { k } \quad ( i \neq j ) .
$$

The full gradient therefore satisfies $\mathbf { 1 } ^ { \top } G = 0$ and $\operatorname { r a n k } ( G ) = k .$ . Consider the update $\Delta W = - r _ { k } G$ For every $i > k + 1$ , this sets $\Delta w _ { i } = - r _ { k } g _ { i } = 0$ . Distinct simplex update rows have distance

$$
\big \| \Delta w _ { i } - \Delta w _ { j } \big \| _ { 2 } = r _ { k } \sqrt { \frac { 2 ( k + 1 ) } { k } } = \rho _ { \mathrm { h e a d } } .
$$

Each simplex update row is also at distance $r _ { k } \le \rho _ { \mathrm { h e a d } }$ from every zero update row. The entire update is therefore feasible. It achieves a decrease of $r _ { k } ( k + 1 )$ ), matching the upper bound on $\Delta _ { \star }$ . Since $\Delta _ { \mathrm { b a l l } } = \rho _ { \mathrm { h e a d } } ( k + 1 ) / 2$ , the relative gap is

$$
\frac { \Delta _ { \star } - \Delta _ { \mathrm { b a l l } } } { \Delta _ { \star } } = 1 - \frac { \rho _ { \mathrm { h e a d } } } { 2 r _ { k } } = 1 - \sqrt { \frac { k + 1 } { 2 k } } .
$$

This proves sharpness for every permissible rank $1 \leq k \leq \operatorname* { m i n } ( d , V - 1 )$ . RowNorm therefore attains the optimal first-order decrease when $k = 1$ and retains at least $1 / \sqrt { 2 }$ of it for every rank. The next subsection describes the practical update with momentum and numerical stabilization.

## A.5 PRACTICAL ROWNORM

In training, we apply row normalization to a first-moment estimate of the gradient. Let $P = I _ { V } -$ $\mathbf { 1 1 } ^ { \top } / V$ denote the projection that subtracts the mean row of a matrix. Starting from $B _ { - 1 } = 0 .$ , for $t = 0 , 1 , \ldots$ . let

$$
B _ { t } = \beta B _ { t - 1 } + ( 1 - \beta ) G _ { t } , \qquad { \widehat { B } } _ { t } = { \frac { B _ { t } } { 1 - \beta ^ { t + 1 } } } , \qquad { \widetilde { B } } _ { t } = P { \widehat { B } } _ { t } ,
$$

where $0 \leq \beta < 1$ . For numerical stability, define $\mathsf { R N } _ { \delta _ { \mathrm { n u m } } }$ by dividing row $\widetilde { b } _ { t , i } ^ { \top }$ by $\left\| \widetilde { b } _ { t , i } \right\| _ { 2 } + \delta _ { \mathrm { n u m } }$ with $\delta _ { \mathrm { n u m } } > 0$ . With learning rate $\eta _ { t } \geq 0$ , the update is

$$
\Delta W _ { t } = - \eta _ { t } P \mathsf { R M } _ { \delta _ { \mathrm { n u m } } } \bigl ( \widetilde { B } _ { t } \bigr ) , \qquad W _ { t + 1 } = W _ { t } + \Delta W _ { t } .\tag{19}
$$

Under our assumptions, the momentum estimate is already centered in exact arithmetic, so the first projection only removes any accumulated numerical drift. Row normalization can introduce a nonzero mean, which the final projection removes. Every normalized row has norm at most one, and centering preserves pairwise differences, so $D ( \Delta W _ { t } ) \stackrel { \cdot } { \leq } 2 \eta _ { t }$ . Consequently, $0 \leq \eta _ { t } \leq \varepsilon / ( 2 R )$ ensures the required Hilbert distance bound for every $\left\| { \dot { h } } \right\| _ { 2 } \leq R .$ The computation takes ${ \dot { O } } ( V d )$ time, uses one first-moment buffer, and implements each projection by subtracting the mean row. Momentum and numerical stabilization preserve the diameter bound on the applied update. The forward and backward estimates therefore remain applicable under their stated assumptions.

## B FORWARD AND BACKWARD PERTURBATION BOUNDS

A spectral constraint, as used by Muon, controls perturbations in both the forward and backward computations of a linear layer. For a fixed input h and incoming gradient $r ,$

$$
\left\| \Delta W h \right\| _ { 2 } \leq \left\| \Delta W \right\| _ { \mathrm { o p } } \left\| h \right\| _ { 2 } , \qquad \left\| \Delta W ^ { \top } r \right\| _ { 2 } \leq \left\| \Delta W \right\| _ { \mathrm { o p } } \left\| r \right\| _ { 2 } .
$$

We establish analogous bounds for our softmax modules below, comparing each module before and after an update at the same inputs. Write $\tau = \operatorname { t a n h } ( \varepsilon / 4 )$ . The inequality relating Hilbert distance and total variation gives $\mathrm { T V } ( p , q ) \leq \tau$ whenever $\mathcal { H } ( p , q ) \leq \varepsilon$ (Cohen & Fausti, 2024, Theorem 5.1).

## B.1 THE LM HEAD

The diameter constraint controls the head’s probability output and also bounds the change in the gradient passed to its hidden state. Let $p = p _ { W } ( h )$ and $q = p _ { W + \Delta W } ( h )$ at a fixed input $\| \bar { h } \| _ { 2 } \leq R$ The forward bound is

$$
\begin{array} { r } { \mathcal { H } ( p , q ) \leq R D ( \Delta W ) \leq \varepsilon . } \end{array}
$$

For a target token $c ,$ denote the gradients passed to h by $\lambda _ { h }$ and ${ \lambda } _ { h } ^ { + }$ . Cross-entropy gives

$$
\lambda _ { h } = W ^ { \top } ( p - e _ { c } ) , \qquad \lambda _ { h } ^ { + } = ( W + \Delta W ) ^ { \top } ( q - e _ { c } ) ,
$$

where $e _ { c }$ is the target’s one-hot vector. Writing $\boldsymbol { w _ { j } ^ { \intercal } }$ and $\Delta w _ { j } ^ { \top }$ for the rows of W and $\Delta W$ , subtraction gives

$$
\lambda _ { h } ^ { + } - \lambda _ { h } = \sum _ { j } q _ { j } ( \Delta w _ { j } - \Delta w _ { c } ) + \sum _ { j } ( q _ { j } - p _ { j } ) w _ { j } .
$$

The first sum averages differences between update rows, so its norm is at most $D ( \Delta W )$ . The second sum accounts for the redistributed probability. Because $p$ and q each have unit mass, the positive and negative coefficients $q _ { j } - p _ { j }$ have the same total mass, namely $\mathrm { T V } ( p , q )$ . After normalization, they form two weighted averages of the head rows. Their distance cannot exceed $D ( W )$ , giving

$$
\bigl \| \lambda _ { h } ^ { + } - \lambda _ { h } \bigr \| _ { 2 } \leq D ( \Delta W ) + D ( W ) \mathrm { T V } ( p , q ) \leq \frac { \varepsilon } { R } + \tau D ( W ) .
$$

The current head appears through $D ( W )$ because it transmits the changed probability gradient.

## B.2 ATTENTION

For attention, we again seek bounds on the change in the output distributions and the resulting change in the gradients passed to the hidden states. The backward calculation now includes both the query and key uses of each hidden state. We keep the inputs $\| h _ { \ell } \| _ { 2 } \le R$ and visible sets $\mathcal { I } _ { i }$ fixed. Let $M = W _ { Q } W _ { K } ^ { \top }$ , and let $\Delta M$ denote its change. Equation 12 gives

$$
\operatorname* { m a x } _ { i } \mathcal { H } ( p _ { i } , p _ { i } ^ { + } ) \leq \frac { 2 R ^ { 2 } } { \sqrt { d _ { h } } } \left\| \Delta \boldsymbol { M } \right\| _ { \mathrm { o p } } \leq \varepsilon .
$$

The total variation bound recalled above then gives $\mathrm { T V } ( p _ { i } , p _ { i } ^ { + } ) \leq \tau$ , where $\tau = \operatorname { t a n h } ( \varepsilon / 4 )$

To connect this probability bound to backpropagation, let $r _ { i }$ denote the incoming gradient on $p _ { i }$ . We hold these incoming gradients fixed when comparing the module before and after the QK update. Differentiating softmax gives the gradient with respect to the scores,

$$
a _ { i j } = p _ { i j } ( r _ { i j } - \mu _ { i } ) , \qquad \mu _ { i } = \sum _ { j \in \mathcal { J } _ { i } } p _ { i j } r _ { i j } .
$$

Each score $s _ { i j } = h _ { i } ^ { \top } M h _ { j } / \sqrt { d _ { h } }$ sends this gradient to both its query input and its key input. Denote the corresponding contributions by $\lambda _ { i } ^ { \mathrm { Q } }$ and $\lambda _ { i j } ^ { \mathrm { K } }$ :

$$
\lambda _ { i } ^ { \mathrm { Q } } = \frac { M \sum _ { j \in \mathcal { I } _ { i } } a _ { i j } h _ { j } } { \sqrt { d _ { h } } } , \qquad \lambda _ { i j } ^ { \mathrm { K } } = \frac { a _ { i j } M ^ { \top } h _ { i } } { \sqrt { d _ { h } } } .
$$

The total gradient passed to $h _ { \ell }$ adds its query contribution and its key contributions from all queries that can see it,

$$
\lambda _ { \ell } = \lambda _ { \ell } ^ { \mathrm { Q } } + \sum _ { i : \ell \in \mathcal { I } _ { i } } \lambda _ { i \ell } ^ { \mathrm { K } } .
$$

There are two sources of change in these expressions: the QK product changes, and so does the score gradient. Writing $\Delta a _ { i } = a _ { i } ^ { + } - a _ { i }$ and $\Delta \lambda = \lambda ^ { + } - \lambda ,$ , we obtain

$$
\begin{array} { r l } & { \Delta \lambda _ { i } ^ { \mathrm { Q } } = \frac { \Delta M \sum _ { j \in \mathcal { I } _ { i } } a _ { i j } ^ { + } h _ { j } + M \sum _ { j \in \mathcal { I } _ { i } } \Delta a _ { i j } h _ { j } } { \sqrt { d _ { h } } } , } \\ & { \Delta \lambda _ { i j } ^ { \mathrm { K } } = \frac { a _ { i j } ^ { + } \Delta M ^ { \top } h _ { i } + \Delta a _ { i j } M ^ { \top } h _ { i } } { \sqrt { d _ { h } } } . } \end{array}
$$

Since every hidden state has norm at most $R ,$ the remaining quantities to bound are $\left\| a _ { i } ^ { + } \right\| _ { 1 }$ and $\left\| \Delta \boldsymbol { a } _ { i } \right\| _ { 1 }$ . The first accompanies the product change, and the second accompanies the current product.

We establish these estimates for one row, temporarily omitting the index i. Write $a _ { j } ( p ) = p _ { j } ( r _ { j } - \mu )$ with $\begin{array} { r } { \mu = \sum _ { j } p _ { j } r _ { j } } \end{array}$ . Recall that osc $\begin{array} { r } { \mathbf { \Psi } ( r ) = \operatorname* { m a x } _ { j } r _ { j } - \operatorname* { m i n } _ { j } r _ { j } } \end{array}$ . For the size of $a ( p )$ , Cauchy–Schwarz reduces the calculation to the variance of the entries of r under $p .$ With $m = ( \mathrm { m a x } _ { j } r _ { j } + \mathrm { m i n } _ { j } r _ { j } ) / 2$ denoting the midpoint of their range,

$$
\| a ( p ) \| _ { 1 } ^ { 2 } \leq \sum _ { j } p _ { j } ( r _ { j } - \mu ) ^ { 2 } = \sum _ { j } p _ { j } ( r _ { j } - m ) ^ { 2 } - ( \mu - m ) ^ { 2 } \leq { \frac { \mathrm { o s c } ( r ) ^ { 2 } } { 4 } } .
$$

For the change in $^ { a , }$ we use the probability redistribution measured by total variation. Any change from $p$ to q can be decomposed into transfers from coordinates that lose probability to coordinates that gain it. The total mass transferred in this way is $\mathrm { T V } ( p , q )$ . It therefore suffices to bound the effect of one transfer and add these bounds.

Suppose an amount α moves from coordinate k to coordinate j, giving $q = p + \alpha ( e _ { j } - e _ { k } )$ . Here $e _ { j } , e _ { k }$ are coordinate vectors and $0 \leq \alpha \leq p _ { k }$ . Subtracting the score gradients at fixed r gives

$$
a ( q ) - a ( p ) = \alpha \bigl [ ( r _ { j } - \mu ) e _ { j } - ( r _ { k } - \mu ) e _ { k } - ( r _ { j } - r _ { k } ) q \bigr ] .
$$

Because q has unit mass,

$$
\begin{array} { r } { \| a ( q ) - a ( p ) \| _ { 1 } \leq \alpha \big ( | r _ { j } - \mu | + | r _ { k } - \mu | + | r _ { j } - r _ { k } | \big ) \leq 2 \alpha \operatorname { o s c } ( r ) . } \end{array}
$$

The last step uses the fact that the three absolute differences sum to twice the range of $r _ { j } , r _ { k } , \mu$ . All three values lie within the range of r, since µ is a weighted average of its entries. Adding the bounds over the transfers proves $\| a ( { \bar { q } } ) - a ( p ) \| _ { 1 } \leq 2 \sec ( r ) { \mathrm { \bar { T } V } } ( p , q )$

Returning to the attention rows, the two estimates become

$$
\big \| \boldsymbol { a } _ { i } ^ { + } \big \| _ { 1 } \leq \frac { \operatorname { o s c } ( \boldsymbol { r } _ { i } ) } { 2 } , \qquad \| \Delta \boldsymbol { a } _ { i } \| _ { 1 } \leq 2 \operatorname { o s c } ( \boldsymbol { r } _ { i } ) \tau .
$$

Apply these to the query and key expressions above, using $\| h _ { \ell } \| _ { 2 } \leq R$ and the triangle inequality:

$$
\begin{array} { r l r } {  { \sum _ { \ell } \| \Delta \lambda _ { \ell } \| _ { 2 } \leq \frac { 2 R } { \sqrt { d _ { h } } } \sum _ { i } \Big ( \| \Delta M \| _ { \mathrm { o p } } \| a _ { i } ^ { + } \| _ { 1 } + \| M \| _ { \mathrm { o p } } \| \Delta a _ { i } \| _ { 1 } \Big ) } } \\ & { } & { \leq \frac { R } { \sqrt { d _ { h } } } ( \| \Delta M \| _ { \mathrm { o p } } + 4 \| M \| _ { \mathrm { o p } } \tau ) \sum _ { i } \mathrm { o s c } ( r _ { i } ) . } \end{array}
$$

The factor of two in the first line accounts for the query and key contributions. Both are included when a position attends to itself. For fixed M and incoming gradients, the bound tends to zero as ε tends to zero under the stated spectral constraint. For finite weight updates, $\Delta M$ includes the quadratic term $\Delta W _ { Q } \Delta W _ { K } ^ { \top }$ . The comparison concerns the QK probability module and does not include changes in the incoming gradients or the value pathway.

## C REPRODUCING FIGURE 1

Figure 1 holds the initial distribution fixed and varies only the centered update. We construct the initial logits by centering a quadratic profile over $i = 1 , \ldots , 3 2$

$$
a _ { i } = - \left( \frac { i - 1 6 . 5 } { 4 } \right) ^ { 2 } , \qquad z _ { i } = a _ { i } - \frac { 1 } { 3 2 } \sum _ { j = 1 } ^ { 3 2 } a _ { j } , \qquad p = \mathrm { s o f t m a x } ( z ) .
$$

The three updates below either shift a central block together or perturb individual preactivation coordinates within it. Here, $\mathbf { 1 } _ { \{ \cdot \} }$ is the indicator function, equal to one when its condition holds and zero otherwise:

$$
\begin{array} { r l } & { \Delta z _ { i } ^ { \mathrm { b l u e } } = 2 \mathbf { 1 } _ { \left\{ 1 3 \leq i \leq 1 6 \right\} } - 2 \mathbf { 1 } _ { \left\{ 1 7 \leq i \leq 2 0 \right\} } , } \\ & { \Delta z _ { i } ^ { \mathrm { p u r p l e } } = 2 \mathbf { 1 } _ { \left\{ 9 \leq i \leq 2 4 \right\} } - 1 , } \\ & { \Delta z _ { i } ^ { \mathrm { r e d } } = 2 \mathbf { 1 } _ { \left\{ i = 1 6 \right\} } - 2 \mathbf { 1 } _ { \left\{ i = 1 9 \right\} } . } \end{array}
$$

Each update has zero mean and produces $q = \mathrm { s o f t m a x } ( z + \Delta z )$ . Using

$$
\mathrm { R M S } ( \Delta z ) = \sqrt { \frac { 1 } { 3 2 } \sum _ { i = 1 } ^ { 3 2 } ( \Delta z _ { i } ) ^ { 2 } } , \qquad \mathcal { H } ( p , q ) = \operatorname* { m a x } _ { i } \Delta z _ { i } - \operatorname* { m i n } _ { i } \Delta z _ { i } ,
$$

we obtain the following values, together with the total variation $\begin{array} { r } { \mathrm { T V } ( p , q ) = \frac { 1 } { 2 } \sum _ { i } | p _ { i } - q _ { i } | \colon } \end{array}$

$$
\begin{array} { r l } &  \frac { \mathrm { B l u e } } { \mathrm { R M S } ( \Delta z ) } \left| \begin{array} { c c c } { \mathrm { B l u e } } & { \mathrm { P u r p l e } } & { \mathrm { R e d } } \\ { 1 } & { 1 } & { 1 / 2 } \\ { 4 } & { 2 } & { 4 } \\ { \mathrm { T V } ( p , q ) } & { \right| \approx 0 . 5 1 4 } & { \approx 0 . 0 0 4 } \end{array} . \end{array}
$$

Purple adds a common increment to coordinates 9–24, which initially carry more than 99.5% of the probability mass. The pairwise odds among coordinates $9 , \ldots , 2 4$ remain unchanged, and only a small amount of probability mass moves from the tails into this block. Blue and red instead alter these ratios, producing much larger redistribution under the same RMS bound.