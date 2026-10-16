# Two Sample Sizes in Test-Time LLM Ensembling: Bufered ERM and Minimax Theory

Zhi Zhang and Arash A. Amini<sup>∗</sup> Department of Statistics and Data Science, UCLA

## Abstract

Test-time scaling can improve the performance of large language models by generating multiple answers to a question and aggregating them. Weighted ensembles extend this idea by drawing answers from several models, with the weights determining how often each model is used. Learning these weights from a training dataset involves two sample sizes: Q labeled questions and N generations per model and question. We study their interplay through the clean risk, the population error of the ensemble’s infinite-generation voting rule. Under a margin condition with exponent β, we show that empirical risk minimization (ERM) with a calibrated margin bufer achieves excess risk of order $\bar { Q } ^ { - 1 / 2 } + N ^ { - \beta / 2 }$ , up to logarithmic factors, for fixed numbers of models and candidate answers. To investigate the optimality of this dependence on Q and N, we analyze a simplified submodel with two models and two possible answers per question. We characterize its minimax excess risk up to constants by a Hellinger modulus for a binomial-mixture experiment. Polynomial approximation and spectral analysis reveal three qualitatively diferent regimes. When questions are relatively scarce, the $Q ^ { - 1 / 2 }$ term governs learning and the general bufered ERM is optimal. As questions become more abundant, we reach an intermediate regime in which pooling generation counts across questions improves upon estimating each answer distribution separately, with only logarithmic gains from additional questions. With suficiently many questions, we enter the third regime where the minimax excess risk reaches an $N ^ { - \beta }$ identification floor that persists even with infinitely many questions. The submodel’s lower bounds also apply to general ensembles, matching the bufered-ERM upper bound up to logarithmic factors when questions are scarce and in parts of the intermediate regime. Questions remain about the transitions between regimes and the construction of matching pooled estimators for general ensembles.

## 1 Introduction

Generating more than one response at inference time has become a standard way to improve the performance of large language models. A natural way to combine repeated generations is by plurality voting, that is, returning the final answer that appears most often. This voting rule is used in self-consistency (Wang et al., 2023). Alternatively, a verifier or reward model can score the candidates and select the highest-scoring one, as in best-of-N selection (Cobbe et al., 2021; Huang et al., 2025; Mukherjee et al., 2026). Although these procedures use the samples diferently, they share a statistical question: how much of the model’s infinite-generation behavior can be recovered from finitely many generations? One way to study this question is to consider test-time procedures with parameters learned from data; we introduce one such procedure below.

For such a learning problem, a first sample size is N, the number of generations used to estimate a model’s answer distribution on each training question. There is, however, a second sample size that is easy to overlook. A test-time procedure is learned or evaluated on a finite benchmark, but its intended performance concerns a broader population of questions. We model the benchmark questions as $Q$ draws from a population distribution Q. Thus $Q$ controls generalization beyond the observed questions, whereas N controls how well the answer distribution on each question is known. The two sampling layers are nested: a benchmark score averages over questions, but each term in that average is itself computed from finitely many generations.

We study this interaction in the weighted multi-model setting of Best-of-∞ (Komiyama et al., 2026), which combines answers by voting without a verifier at deployment. Models may favor diferent wrong answers to the same question. A mixture can then make the correct answer the most probable even when it is not the most probable under any individual model (see Table 1 for an empirical illustration).

Suppose K fixed LLMs can be queried on a question q. The question has a finite candidate-answer set $\mathcal A ( q )$ , whose size is uniformly bounded by A, and model i has an infinite-generation answer distribution $p _ { i } ( \cdot \mid q )$ on $\scriptstyle A ( q )$ . Given weights $w \in \Delta _ { K }$ , the ensemble distribution is

$$
p _ { w } ( a \mid q ) : = \sum _ { i = 1 } ^ { K } w _ { i } p _ { i } ( a \mid q ) .\tag{1}
$$

Practically, this corresponds to choosing a model independently according to w for each generation and pooling the resulting answers; equivalently, one may allocate asymptotic fractions $w _ { i }$ of the generation budget to the models. The single-model problem is recovered as a special case.

The weights also determine generation cost. If a generation from model i costs $c _ { i } .$ , the mean cost per generation is $\begin{array} { r } { \bar { c } ( w ) : = \sum _ { i = 1 } ^ { K } w _ { i } c _ { i } } \end{array}$ . An ensemble that retains good population voting accuracy while shifting weight toward cheaper models reduces this mean cost.

Assume each question q has a designated gold answer $a ^ { \star } ( q ) \in \mathcal { A } ( q )$ . For fixed weights $w ,$ the associated ideal voting rule selects the most probable answer under the infinite-generation mixture distribution $p _ { w } ( \cdot \mid q )$ . The rule is declared correct when $a ^ { \star } ( q )$ is the unique maximizer, counting ties as errors. Define the clean margin

$$
M _ { q } ( w ) : = p _ { w } ( a ^ { \star } ( q ) \mid q ) - \operatorname* { m a x } _ { a \neq a ^ { \star } ( q ) } p _ { w } ( a \mid q ) .\tag{2}
$$

The error probability of this ideal rule over the population of questions is therefore

$$
R ( w ) : = \mathbb { P } _ { q \sim \mathcal { Q } } \{ M _ { q } ( w ) \leq 0 \} .\tag{3}
$$

We call (3) the clean risk because it treats the answer distributions $p _ { i } ( \cdot \mid q )$ as known. Our statistical problem is to learn weights with small clean risk from finite benchmark data. Our target is a clean-risk minimizer

$$
w ^ { \star } \in \underset { w \in \Delta _ { K } } { \mathrm { a r g m i n } } R ( w ) .\tag{4}
$$

Neither layer of this risk is observed directly. In practice, we observe labeled questions

$$
q _ { 1 } , \ldots , q _ { Q } \stackrel { \mathrm { i i d } } { \sim } \mathcal { Q }
$$

where “labeled” means that the benchmark supplies each $q _ { j }$ together with its gold answer $a ^ { \star } ( q _ { j } )$ We also observe N sampled answers from every model on every question. The raw generations for model i on question $q _ { j }$ can be summarized by the multinomial count vector

$$
X _ { j , i } : = ( X _ { j , i , a } : a \in \mathcal { A } ( q _ { j } ) ) \sim \operatorname { M u l t i n o m i a l } \bigl ( N , p _ { i } ( \cdot \mid q _ { j } ) \bigr ) , \qquad \widehat { p _ { i } } ( \cdot \mid q _ { j } ) = \frac { X _ { j , i } } { N } .
$$

<table><tr><td></td><td colspan="10">Answer (empirical frequencies in %)</td><td colspan="3">Observed</td></tr><tr><td>Distribution</td><td>70</td><td>147</td><td>273</td><td>294*</td><td>495</td><td>810</td><td>1225</td><td>1820</td><td>2730</td><td>4116</td><td>7350</td><td>12870</td><td>support 44100</td></tr><tr><td>Gemma 3 12B</td><td>4</td><td>12</td><td>0</td><td>20</td><td>8</td><td>4</td><td>24</td><td>4</td><td></td><td>4</td><td></td><td></td><td>12</td></tr><tr><td>Llama 4 Maverick</td><td>0</td><td>0</td><td>4</td><td>44</td><td>0 52</td><td></td><td>0</td><td>0</td><td>0</td><td></td><td>0</td><td>0</td><td>3</td></tr><tr><td>Equal mixture</td><td>2</td><td>6</td><td>2</td><td>32</td><td>4</td><td>28</td><td>12</td><td>2</td><td></td><td></td><td></td><td>2</td><td>13</td></tr></table>

Table 1: Empirical answer frequencies for AIME 2024 I, Problem 6, with $N = 2 5$ generations per model at temperature 0.8. Both models were asked to reason step by step and return a boxed final answer; all 50 responses contained complete boxed answers and finished below their token caps. Every observed answer value is shown. The star marks the gold answer, zeros denote unobserved answers, and boldface marks each empirical mode. The last row is the equal-weight mixture.

Here, $X _ { j , i , a }$ is the number of times model i generated answer a on question $q _ { j }$

To make these ideas concrete, Table 1 gives an illustrative example from a real experiment. The selected question, Problem 6 of the 2024 American Invitational Mathematics Examination (AIME I), asks for the number of monotone paths across an $8 \times 8$ grid with exactly four changes of direction; its gold answer is 294. In $N = 2 5$ generations per model, Gemma 3 12B produced 12 distinct answers and Llama 4 Maverick produced 3. Their empirical modes are diferent wrong answers, 1225 and 810, while their equal-weight empirical mixture has the gold answer as its unique mode. The clean rankings remain unknown and must be inferred from such finite samples.

If the clean distributions $p _ { i }$ were known, ordinary empirical risk minimization over the sampled questions would minimize

$$
\widehat { R } _ { Q } ^ { \mathrm { c l e a n } } ( w ) : = \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } \mathbf { 1 } \{ M _ { q _ { j } } ( w ) \leq 0 \} .\tag{5}
$$

In practice, $p _ { i }$ is replaced by $\widehat { p _ { i } }$ , giving the empirical margin $\widehat { M } _ { q _ { j } } ( w )$ which replaces the clean margin in (5). Finite-N noise can then change whether a question is considered solved at a given w. In particular, a favorable fluctuation may produce $\widehat { M } _ { q _ { j } } ( w ) > 0$ even when $M _ { q _ { j } } ( w ) \leq 0 .$ , causing zero-margin ERM to record an empirical success where the clean ensemble makes an error. In other words, zero-margin ERM can exploit favorable fluctuations in the empirical margins and efectively overfit to the finite-N generation noise.

This motivates the bufered estimator

$$
\widehat { w } _ { \varepsilon } \in \mathop { \mathrm { a r g m i n } } _ { w \in \Delta _ { K } } \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } \mathbf { 1 } \{ \widehat { M } _ { q _ { j } } ( w ) \leq \varepsilon \} .\tag{6}
$$

The choice $\varepsilon = 0$ recovers the Best-of-∞ ERM introduced by Komiyama et al. (2026). When $\varepsilon > 0$ a training question is counted as solved only if its empirical margin exceeds $\varepsilon ;$ empirical margins in (0, ε] are treated as errors. On the event

$$
\operatorname* { s u p } _ { j \leq Q } \operatorname* { s u p } _ { w \in \Delta _ { K } } \left| \widehat { M } _ { q _ { j } } ( w ) - M _ { q _ { j } } ( w ) \right| \leq \varepsilon _ { N } ,\tag{7}
$$

we have, simultaneously for every training question and every weight vector,

$$
\mathbf { 1 } \{ M _ { q _ { j } } ( w ) \leq 0 \} \leq \mathbf { 1 } \{ \widehat { M } _ { q _ { j } } ( w ) \leq \varepsilon _ { N } \} \leq \mathbf { 1 } \{ M _ { q _ { j } } ( w ) \leq 2 \varepsilon _ { N } \} .\tag{8}
$$

The left inequality prevents the learned rule from claiming empirically certified successes that fail under the clean margin. The right inequality shows that the price of this protection is confined to

a narrow clean-margin band. This bufered sandwich is the basic mechanism behind our general result. We also record a corollary for deployment with finitely many generations (Theorem 2.4).

## 1.1 Overview of the results

Let us now give an overview of the main results, starting with a general bound on the excess risk of the bufered ERM; questions about the optimality of this bound lead to a scalar submodel and its minimax analysis.

Bufered ERM for general ensembles. Take

$$
\varepsilon _ { N } \asymp \sqrt { \frac { \log ( K A Q / \delta ) } { N } } ,
$$

where A bounds the number of candidate answers, and $\delta > 0$ is some specified confidence level. Under a one-sided margin condition at the clean-risk minimizer,

$$
\mathbb { P } \{ 0 < M _ { q } ( w ^ { \star } ) \le t \} \le C _ { m } t ^ { \beta } ,
$$

the estimator in (6) $\left( { \mathrm { w i t h ~ } } \varepsilon = \varepsilon _ { N } \right)$ satisfies

$$
R ( \widehat { w } _ { \varepsilon _ { N } } ) - R ( w ^ { \star } ) \lesssim \sqrt { \frac { K \log ( A Q ) + \log ( 1 / \delta ) } { Q } } + C _ { m } \left( \frac { \log ( K A Q / \delta ) } { N } \right) ^ { \beta / 2 } .\tag{9}
$$

The first term is generalization across questions. The second is the cost of replacing population answer distributions by finite-generation estimates. The proof combines (8) with a hyperplanearrangement bound for the clean margin classes. Treating $K , A ,$ , and the margin constants as fixed and suppressing logarithmic and confidence factors, the dependence on the two sample sizes is

$$
R ( \widehat { w } _ { \varepsilon _ { N } } ) - R ( w ^ { \star } ) = \widetilde O \big ( Q ^ { - 1 / 2 } + N ^ { - \beta / 2 } \big ) .\tag{10}
$$

This bound already exhibits a fast-rate efect in $N .$ . Although each clean margin is estimated at the usual $N ^ { - 1 / 2 }$ scale, the margin condition converts this estimation error into an excess-risk contribution of order $N ^ { - \beta / 2 }$ . The exponent $\beta$ is fixed as N and $Q$ grow, but it may be arbitrarily large; consequently, this term can decay faster than $N ^ { - 1 / 2 }$ , or even $N ^ { - 1 }$ . This does not contradict standard parametric estimation limits: the excess risk depends only on questions near the zero-margin boundary, and the margin condition makes such questions increasingly rare.

The overall bound, of course, still contains the $Q ^ { - 1 / 2 }$ term. This raises the key optimality question: to what extent is the dependence on $( Q , N )$ in (10) intrinsic, and in particular can the $N ^ { - \beta / 2 }$ term be improved by pooling information across questions?

A scalar model for the optimality question. To investigate this question, we isolate the simplest nontrivial submodel: querying two models producing two possible answers each. Because the answer set may depend on $q ,$ we may relabel the gold answer on every question as “gold” and the other answer as “other” without loss of generality, that is, $\mathcal { A } ( \boldsymbol { q } ) = \{ \mathrm { g o l d } , \mathrm { o t h e r } \}$ for all $q$ and $a ^ { \star } ( q ) = \operatorname { g o l d }$ . Let $Z _ { q } \in [ - 1 , 1 ]$ be a latent signed advantage and suppose

$$
p _ { 1 } ( \mathrm { g o l d } \mid q ) = \frac { 1 + Z _ { q } } { 2 } , \qquad p _ { 2 } ( \mathrm { g o l d } \mid q ) = \frac { 1 - Z _ { q } } { 2 } .\tag{11}
$$

For $w = ( t , 1 - t )$ , the clean margin is $( 2 t - 1 ) Z _ { q }$ . Thus every $t > 1 / 2$ gives the same clean decisions: the ensemble is correct exactly when $Z _ { q } > 0$ . Every $t < 1 / 2$ gives the opposite decisions and is correct exactly when $Z _ { q } < 0$ , while $t = 1 / 2$ ties on every question. The learning problem therefore reduces to choosing between the two halves of the weight simplex, rather than estimating the magnitude of $t - 1 / 2$

Let G denote the distribution of $Z _ { q }$ when $q \sim \mathcal { Q } .$ , that is, the distribution induced by the map $q \mapsto Z _ { q }$ . Write $\sigma = + 1$ for the action of choosing any $t > 1 / 2$ and $\sigma = - 1$ for choosing any $t < 1 / 2$ and let $R _ { + } ( G )$ and R<sub>−</sub>(G) denote their respective population risks. Their diference is the linear functional measuring the imbalance of population mass across zero:

$$
R _ { - } ( G ) - R _ { + } ( G ) = \theta ( G ) : = \int \operatorname { s g n } ( z ) \ \mathrm { d } G ( z ) .\tag{12}
$$

Hence the sign of $\theta ( G )$ determines which half of the simplex is optimal, and $| \theta ( G ) |$ is the excess risk, or regret, incurred by choosing the wrong half. Section 3 gives the exact decision reduction; the minimax analysis below asks how well this sign can be learned from finite generations.

The general multinomial observation also simplifies. If $X _ { 1 q }$ and $X _ { 2 q }$ count the gold answers from the two models, then

$$
S _ { q } : = X _ { 1 q } + N - X _ { 2 q }
$$

is suficient for $Z _ { q }$ . For the sampled questions, write $Z _ { j } : = Z _ { q _ { j } }$ and $S _ { j } : = S _ { q _ { j } }$ . The reduced experiment can then be written as

$$
Z _ { 1 } , \dots , Z _ { Q } \stackrel { \mathrm { i i d } } { \sim } G , \qquad S _ { j } \mid Z _ { j } = z \sim \mathrm { B i n } \left( n , \frac { 1 + z } { 2 } \right) , \quad j = 1 , \dots , Q , \qquad n : = 2 N .\tag{13}
$$

Thus the raw generations first reduce to multinomial answer counts and then, under antisymmetry, to one binomial count per question. We write $K _ { n } G$ for the resulting marginal distribution of each $S _ { j } .$ , where $K _ { n }$ denotes the n-trial binomial channel. The observed counts $S _ { 1 } , \ldots , S _ { Q }$ are iid from this nonparametric binomial-mixture distribution, and the ensemble problem has become indirect estimation of (12) through the binomial channel.

The original learner also observes the question identities $q _ { 1 } , \ldots , q _ { Q }$ . We begin by studying the reduced count-only experiment, in which only $S _ { 1 } , \ldots , S _ { Q }$ are retained. This reduction is immediate for upper bounds, since an estimator may ignore the question identities, but it requires justification for lower bounds. We later prove in Theorems 4.4 and 4.6 that, over an unrestricted class of question-to-advantage maps, the count-only lower bounds remain valid when the questions are observed and therefore transfer to the general ensemble problem.

Exact scalar minimax characterization. On the two-sided margin class

$$
\mathcal { G } _ { \beta , L } = \{ G : G ( [ - t , t ] ) \leq L t ^ { \beta } , 0 < t \leq 1 \} ,
$$

let $\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L )$ denote the minimax expected regret for choosing between the two actions from $S _ { 1 } , \ldots , S _ { Q } \colon$

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) : = \operatorname* { i n f } _ { \widehat { \sigma } } \operatorname* { s u p } _ { G \in \mathcal { G } _ { \beta , L } } \mathbb { E } _ { G } [ | \theta ( G ) | \mathbf { 1 } \{ \widehat { \sigma } \theta ( G ) < 0 \} ] , \qquad \widehat { \sigma } \in \{ - 1 , + 1 \} .
$$

Writing h for Hellinger distance, define

$$
\omega _ { n , \beta , L } ( \varepsilon ) : = \operatorname* { s u p } \left\{ | \theta ( G ) - \theta ( H ) | : G , H \in { \mathcal G } _ { \beta , L } , \ h ( K _ { n } G , K _ { n } H ) \le \varepsilon \right\} .
$$

This modulus measures the largest ambiguity in the functional θ among two mixing distributions whose observable count laws are within Hellinger distance ε. We prove the constant-factor characterization

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \asymp \omega _ { n , \beta , L } ( Q ^ { - 1 / 2 } )\tag{14}
$$

for every pair $( Q , n )$ . Thus, up to constants, the minimax regret is exactly the ambiguity in θ remaining at Hellinger resolution $Q ^ { - 1 / 2 }$ after passage through the n-trial binomial channel.

Direct evaluations of the modulus (14) help identify three regimes. We begin with the two endpoints, in which one of $Q$ or n remains fixed while the other tends to infinity. At the finite-question endpoint, consider the hard-margin subclass of mixing distributions G supported on $\{ - 1 , + 1 \}$ ; this is equivalent to assuming that there are only “two types” of questions. The count reveals the sign of Z without error for every $n ,$ just as if infinitely many generations were available for each question. Nevertheless, from only $Q$ sampled questions the learner must determine which of the two question types is more prevalent. Evaluating the modulus on this subclass gives the unavoidable root-Q lower bound uniformly in $( Q , n )$ $\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \gtrsim Q ^ { - 1 / 2 }$

At the opposite endpoint, fix n and let $Q  \infty$ . The observable count law $K _ { n } G$ can then be learned arbitrarily accurately, so uncertainty from sampling questions disappears. Yet $K _ { n } G$ depends on G only through its first n moments. Diferent mixing distributions can therefore induce exactly the same count law while having diferent values of θ. We show that the resulting zero-radius ambiguity is

$$
\omega _ { n , \beta , L } ( 0 ) \asymp n ^ { - \beta } .\tag{15}
$$

Hence finitely many generations per question impose an $n ^ { - \beta }$ floor, no matter how large $Q$ becomes. By monotonicity of the modulus and (14), the same lower bound holds uniformly for all $( Q , n )$ $\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \gtrsim n ^ { - \beta }$

Between these two endpoints lies a subtler middle region. Its behavior is governed by spectral properties of the Bernstein–Durrmeyer operator associated with the Bernstein polynomials $b _ { k , n } ( p ) =$ $\binom { n } { k } p ^ { k } ( 1 - p ) ^ { n - k }$ defining the binomial law. We first approach this region from the upper-bound side, showing that the two endpoint rates $Q ^ { - 1 / 2 }$ and $n ^ { - \beta }$ continue to describe the minimax risk over nontrivial portions of the $( Q , n )$ plane. We then use the same spectral structure to obtain matching lower bounds in a broad middle regime.

Pooling and the spectrum of the binomial channel. The endpoint bounds are attained by two diferent procedures. A sign-then-aggregate estimator replaces each count $S _ { j }$ by sgn $( S _ { j } - n / 2 )$ and then chooses the majority side. It satisfies

$$
\begin{array} { r } { \mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \lesssim _ { \beta , L } Q ^ { - 1 / 2 } + n ^ { - \beta / 2 } , } \end{array}
$$

the same dependence on $( Q , n )$ as bufered ERM in the general problem, up to logarithmic factors.   
This procedure compresses every question to one estimated sign before aggregating across questions.

With more questions, it can be better to retain the full count values. A pooled polynomialmoment estimator approximates sgn by a degree-d polynomial $P _ { d }$ and uses each count $S _ { j }$ to form an unbiased estimate of $P _ { d } ( Z _ { j } )$ , so their average estimates the functional $\int P _ { d } \ \mathrm { d } G$ . Its approximation bias is of order $d ^ { - \beta }$ . Its statistical variability is controlled by how strongly inversion of the binomial conditional-expectation operator $\begin{array} { r } { T _ { n } a : = \sum _ { k = 0 } ^ { n } a _ { k } b _ { k , n } ( \cdot ) } \end{array}$ amplifies the polynomial modes of $P _ { d } ;$ we call this inverse amplification the estimator’s stability cost. As shown in Section 5.3, the amplification factors are the inverse singular values of $T _ { n }$ , or equivalently the inverse square roots of the eigenvalues of $D _ { n } = T _ { n } T _ { n } ^ { * }$ , the Bernstein–Durrmeyer operator. Writing $\kappa _ { d , n }$ for the largest

inverse singular value through degree $d ,$

$$
\kappa _ { d , n } : = \left[ \prod _ { r = 0 } ^ { d - 1 } { \frac { n + r + 2 } { n - r } } \right] ^ { 1 / 2 } ,
$$

we obtain the spectral upper bound

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \lesssim _ { \beta , L } \operatorname* { i n f } _ { 1 \leq d \leq n } \left\{ d ^ { - \beta } + \frac { \sqrt { n + 1 } \kappa _ { d , n } } { \sqrt { Q } } \right\} .\tag{16}
$$

For $d \leq n / 2$ , log $\kappa _ { d , n } \asymp d ^ { 2 } / n \colon$ : higher-degree information is progressively more unstable, but it becomes estimable as the number of questions grows.

The same spectrum also produces matching lower bounds. We construct pairs of mixing distributions whose first d moments agree, whose targets difer by order $d ^ { - \beta }$ , and whose remaining higher-order modes are strongly attenuated by the channel. For $0 ~ < ~ \beta ~ \leq ~ 1$ , optimizing the cancellation degree gives the rigorous middle-regime result

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \asymp _ { \beta , L , \eta } [ n \log Q ] ^ { - \beta / 2 } , \qquad n ^ { \beta + 1 + \eta } \le Q \le e ^ { c _ { \eta } n } ,\tag{17}
$$

for every fixed $\eta > 0$ . The upper bound is attained by the pooled polynomial-moment estimator with degree $d \asymp { \sqrt { n \log Q } }$ . At the other endpoint, taking $d = n$ and using $\kappa _ { n , n } \asymp 2 ^ { n } n ^ { - 1 / 4 }$ shows that the same estimator reaches the $n ^ { - \beta }$ floor once

$$
Q \gtrsim 4 ^ { n } n ^ { 2 \beta + 1 / 2 } .
$$

The rigorous rate picture and its scope. These results reveal three diferent sources of dificulty. We call the range in which the unavoidable $Q ^ { - 1 / 2 }$ term dominates the question-limited regime. Once Q is larger, pooling the count information can improve on per-question thresholding; we call this the spectral-pooling regime. Finally, when the count law is estimated accurately enough that finite-moment nonidentifiability becomes the bottleneck, the problem is identification-limited. The regions in which matching scalar rates are currently proved are summarized below; recall that $n = 2 N$
<table><tr><td>Proved region</td><td>Minimax rate</td><td>Achieving procedure</td><td>Scope</td></tr><tr><td> $Q \lesssim N ^ { \beta }$ </td><td> $Q ^ { - 1 / 2 }$ </td><td>Buffered ERM; scalar sign-then-aggregate</td><td>Matching result for general ensembles, up to logarithmic</td></tr><tr><td> $N ^ { \beta + 1 + \eta } \lesssim Q \lesssim e ^ { c _ { \eta } N }$ </td><td> $[ N \log Q ] ^ { - \beta / 2 }$ </td><td>Scalar pooled polynomial-moment estimator</td><td>factors Scalar matching for  $0 < \beta \leq 1$  ; lower bound transfers generally</td></tr><tr><td> $Q \gtrsim 4 ^ { 2 N } \mathrm { p o l y } ( N )$ </td><td> $N ^ { - \beta }$ </td><td>Scalar pooled polynomial-moment estimator</td><td>Scalar matching; lower bound transfers generally</td></tr></table>

The first row is a result for general ensembles: the bufered-ERM upper bound matches the transferred root-Q lower bound, showing that bufered ERM is rate-optimal up to logarithmic factors in the question-limited regime. In the other two rows, the matching upper procedures currently rely on the antisymmetric scalar structure. There is nevertheless a useful general consequence in the polynomial-Q portion of the middle regime when $0 < \beta \leq 1$ . The transferred spectral lower bound is then of order $( N \log N ) ^ { - \beta / 2 }$ , while bufered ERM has excess risk of order $( \log N / N ) ^ { \beta / 2 }$ ; the two difer by only a factor $( \log N ) ^ { \beta }$ . Thus bufered ERM is minimax optimal up to logarithmic factors there.

This logarithmic near-optimality does not extend uniformly over the entire subexponential range. For example, still for $0 < \beta \le 1$ , if $Q = e ^ { N ^ { a } }$ with $0 < a < 1$ , the transferred lower bound and the bufered-ERM upper bound scale respectively as $N ^ { - \beta ( 1 + a ) / 2 }$ and $N ^ { - \beta ( 1 - a ) / 2 }$ . Their ratio is $N ^ { a \beta } = ( \log Q ) ^ { \beta } ;$ : it is polylogarithmic in $Q ,$ but polynomial in $N$

The third row begins when Q is exponential in N. Indeed, for $Q = e ^ { c N }$ , the finite-generation term in the bufered-ERM bound is of constant order, so the bufered-ERM theorem does not even guarantee a vanishing excess risk. By contrast, in the antisymmetric scalar model, the pooled polynomial-moment estimator attains the $N ^ { - \beta }$ identification floor once the explicit exponential threshold in the table is met. This rate does not follow from the bufered-ERM analysis, and constructing a matching pooled procedure for general ensembles remains open.

Conjectured completion of the scalar picture. The spectral upper and lower bounds suggest that the intrinsic stability cost at degree d is $\kappa _ { d , n }$ itself. This leads to the conjecture

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \stackrel { ? } { \sim } _ { \beta , L } \operatorname* { i n f } _ { 1 \leq d \leq n } \left\{ d ^ { - \beta } + \frac { \kappa _ { d , n } } { \sqrt { Q } } \right\} .
$$

If true, it gives the following three-phase rate picture:

$$
\begin{array} { r } { \mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \asymp _ { \beta , L } \left\{ \begin{array} { l l } { Q ^ { - 1 / 2 } , } & { Q \lesssim n ^ { \beta } , } \\ { \left[ n \log ( e Q / n ^ { \beta } ) \right] ^ { - \beta / 2 } , } & { n ^ { \beta } \lesssim Q \lesssim e ^ { c n } , } \\ { n ^ { - \beta } , } & { Q \gtrsim e ^ { c n } . } \end{array} \right. } \end{array}\tag{18}
$$

The notation $e ^ { c n }$ records only the exponential scale. To be more precise, write

$$
Q = n ^ { \beta } e ^ { u _ { n } } , \qquad u _ { n } = \log ( Q / n ^ { \beta } ) .
$$

The question-limited regime corresponds to $u _ { n }$ remaining bounded above, the genuinely spectralpooling regime has $u _ { n } \to \infty$ with $u _ { n } = o ( n )$ , and the identification-limited regime has $u _ { n } \gtrsim n$ . The conjectured middle-region rate is $( n u _ { n } ) ^ { - \beta / 2 }$ ; at the boundary $u _ { n } \asymp n$ , this becomes $n ^ { - \beta }$ , matching the third-region rate.

The rigorous middle theorem currently begins at $u _ { n } \geq ( 1 + \eta )$ log n for fixed $\eta > 0$ . The conjecture predicts the middle rate for every $u _ { n } \to \infty$ , however slowly, and for all $\beta > 0$ . Thus the main open questions for the scalar model are to resolve the transition immediately above $Q \asymp n ^ { \beta }$ , to sharpen the spectral upper and lower bounds to the same scale, which includes removing the extra $\sqrt { n + 1 }$ factor from the upper bound in (16), and to extend the matching spectral lower theory beyond $\beta = 1$ . For bounded $u _ { n }$ , the stabilized factor $1 + u _ { n } = \log ( e Q / n ^ { \beta } )$ connects the middle expression continuously to the root-Q phase.

## 1.2 Related work

Our work follows the voting approach of self-consistency (Wang et al., 2023), which uses agreement among sampled final answers without requiring a verifier at deployment. Experiments of Brown et al. (2024) illustrate the gap between coverage, the probability that a sample collection contains a correct answer, and the accuracy of the selected answer. Eventual coverage requires only positive probability on the correct answer; our ideal voting rule requires it to be the unique mode. For a fixed model and dataset, Wu et al. (2025) analyze the limiting accuracy of voting and use concentration bounds to study its convergence as the number of generations grows.

The direct starting point for our work is Best-of-∞ (Komiyama et al., 2026). That paper formulates the infinite-generation voting objective for mixtures of LLMs, expresses optimization over a benchmark as a mixed-integer linear program, and empirically studies learning and transferring ensemble weights. It also develops a Bayesian stopping rule for approximating the modal answer with finitely many generations. Whereas their study of learning and transferring ensemble weights is empirical, we develop a finite-sample theory of learning and generalization. Our theory identifies the diferent statistical roles of the benchmark size Q and generation depth N, and shows how their interplay governs the attainable learning rates. The max-margin refinement in Best-of-∞ can still exploit favorable fluctuations in empirical margins. Our statistically calibrated bufer, on the other hand, controls their efect on learning, leading to clean excess-risk guarantees. Our scalar minimax analysis uncovers phase transitions in the attainable learning rates as Q and N vary. It reveals regimes in which pooling across questions improves learning, and an identification floor at finite N that persists even with infinitely many questions.

Learned verifiers provide an alternative to voting by scoring candidate solutions (Cobbe et al., 2021). With the generator and scoring model taken as given, Huang et al. (2025) and Mukherjee et al. (2026) study how selection performance depends on the generator’s coverage of high-quality responses and errors in their scores. Di et al. (2025) also use a reward model: they discard responses with low empirical frequency and return up to k of the remaining responses with the highest predicted rewards. For binary correctness rewards, their pass@k criterion counts success whenever at least one returned response is correct. Their minimax bounds concern this selection problem under coverage and reward-error assumptions; they do not address the joint (Q, N) problem of learning ensemble weights across questions.

The two sampling layers also arise in distribution regression, where each covariate is a probability measure observed through a finite sample (Szab´o et al., 2016). In ensemble learning, the underlying answer distributions determine a discontinuous loss through their ensemble rankings. Our general-K bound combines uniform control of their empirical estimates with a growth-function bound for the resulting polytopal decision rules. Margin assumptions also play a central role in classification theory (Audibert and Tsybakov, 2007; Massart and N´ed´elec, 2006). Here the one-sided condition at w<sup>⋆</sup> limits the mass of questions on which the correct answer wins by a small amount; it controls the finite-N term, while generalization across questions contributes the separate root-Q term.

The scalar reduction leads to the binomial-mixture experiment studied by Tian et al. (2017) and Vinayak et al. (2019), who establish optimal recovery rates under Wasserstein-1 loss. Basu et al. (2026) construct exact pointwise confidence intervals for the mixing distribution’s cumulative distribution function and quantiles. Our target is the sign-imbalance functional θ(G), whose integrand is discontinuous at zero. Wasserstein recovery guarantees do not directly provide our minimax rates, which depend on the mixing mass near this decision boundary. We study estimation of this functional and the associated ensemble decision under a power-law bound on that mass.

The general framework for this scalar problem is indirect estimation of a linear functional over a convex model. The modulus-via-testing theory of Donoho and Liu (1991) relates minimax estimation to Hellinger continuity of the target. Juditsky and Nemirovski (2009) construct near-minimax estimators by convex programming for afine observation models, and Polyanskiy and Wu (2026) characterize minimax quadratic risk through convex duality. We give a self-contained finite-sample specialization for expected absolute loss in our categorical experiment, and use reflection symmetry to connect functional estimation to the ensemble decision. Evaluating the resulting modulus as Q and N vary then reveals how sampling error interacts with the ambiguity left by observing only finitely many moments.

To obtain explicit rates, we relate this moment ambiguity to weighted polynomial approximation of the sign function. The matching lower approximation bound uses the inverse theorem of Kopotun (2015) for doubling weights with finitely many zeros and singularities. For finite $Q ,$ , we use the classical Bernstein–Durrmeyer spectrum (Derriennic, 1981; Abel and Berdysheva, 2010) to quantify how estimating higher polynomial moments amplifies sampling error. Combining this spectral information with the margin-dependent approximation error gives the pooling bounds and the rigorous intermediate regime described above.

## 2 Bufered ERM for general ensembles

We now return to the general K-model ensemble and formalize the observations used to learn its weights. The $Q$ sampled questions and the N generations from each model on each question produce two diferent sources of randomness. We then define bufered ERM and establish its excess-risk guarantee. Whether its dependence on $( Q , N )$ is optimal leads to the scalar submodel studied in the next section.

## 2.1 Model and observations

We retain the notation $p _ { i } , p _ { w } , M _ { q }$ , and $w ^ { \star }$ from the introduction. Write $[ m ] = \{ 1 , \dots , m \}$ and $\Delta _ { K } : = \{ w \in \mathbb { R } _ { + } ^ { K } : \sum _ { i = 1 } ^ { K } w _ { i } = 1 \}$ . We assume throughout that $a ^ { \star } ( q ) \in \mathcal { A } ( q )$ and $2 \leq | { \mathcal { A } } ( q ) | \leq A < \infty$ For later reference, define the per-question clean loss

$$
\ell _ { q } ( w ) = \mathbf { 1 } \{ M _ { q } ( w ) \leq 0 \} , \qquad R ( w ) = \mathbb { E } _ { q \sim \mathcal { Q } } \ell _ { q } ( w ) = \mathbb { P } _ { q \sim \mathcal { Q } } \{ M _ { q } ( w ) \leq 0 \} .\tag{19}
$$

The observed data consist of the labeled questions and the raw generations

$$
\begin{array} { r } { S _ { \boldsymbol { Q } , N } : = \left\{ \big ( q _ { j } , \boldsymbol { a } ^ { \star } ( q _ { j } ) , ( Y _ { j , i , r } ) _ { ( i , r ) \in [ K ] \times [ N ] } \big ) : j \in [ \boldsymbol { Q } ] \right\} . } \end{array}\tag{20}
$$

Here $q _ { 1 } , \ldots , q _ { Q } \stackrel { \mathrm { i i d } } { \sim } \mathcal { Q }$ , and, conditional on the questions, the variables

$$
Y _ { j , i , r } \sim p _ { i } ( \cdot \mid q _ { j } ) , \qquad ( j , i , r ) \in [ Q ] \times [ K ] \times [ N ] ,
$$

are independent. For each question and model, reduce these raw generations to the answer counts

$$
X _ { j , i , a } : = \sum _ { r = 1 } ^ { N } \mathbf { 1 } \{ Y _ { j , i , r } = a \} , \qquad X _ { j , i } : = ( X _ { j , i , a } : a \in \mathcal { A } ( q _ { j } ) ) .
$$

Then

$$
X _ { j , i } \mid q _ { j } \sim \mathrm { M u l t i n o m i a l } \big ( N , p _ { i } ( \cdot \mid q _ { j } ) \big ) , \qquad ( j , i ) \in [ Q ] \times [ K ] ,\tag{21}
$$

and $\widehat { p _ { i } } ( { \cdot } \mid q _ { j } ) = X _ { j , i } / N$ . The counts are suficient for the answer distributions, so all subsequent procedures may be expressed in terms of $X _ { j , i }$ rather than the ordered raw generations. Define the empirical ensemble distribution and margin by

$$
\widehat { p } _ { w } ( a \mid q _ { j } ) : = \sum _ { i = 1 } ^ { K } w _ { i } \widehat { p } _ { i } ( a \mid q _ { j } ) , \qquad \widehat { M } _ { q _ { j } } ( w ) : = \widehat { p } _ { w } ( a ^ { \star } ( q _ { j } ) \mid q _ { j } ) - \operatorname* { m a x } _ { a \neq a ^ { \star } ( q _ { j } ) } \widehat { p } _ { w } ( a \mid q _ { j } ) .
$$

The empirical answer distributions ${ \widehat { p } } _ { i } ( \cdot \mid q _ { j } )$ are observable and therefore make $\widehat { M } _ { q _ { j } } ( w )$ computable for any w; the corresponding clean margin $M _ { q _ { j } } ( w )$ is not observed. The target R(w) $R ( w )$ is defined by these clean margins and the population distribution of questions. Thus the analysis must account both for generalization across the $Q$ questions and for estimation of each question’s margin from N generations.

## 2.2 Estimator and guarantee

Raw ERM minimizes the number of training questions with nonpositive empirical margin. It can exploit weights for which a question appears solved only because of a favorable finite-N fluctuation. We instead define, for a bufer $\varepsilon > 0$

$$
\widehat { R } _ { \varepsilon } ( w ) : = \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } \mathbf { 1 } \{ \widehat { M } _ { q _ { j } } ( w ) \leq \varepsilon \} , \qquad \widehat { w } _ { \varepsilon } \in \mathop { \mathrm { a r g m i n } } _ { w \in \Delta _ { K } } \widehat { R } _ { \varepsilon } ( w ) .\tag{22}
$$

Because $\widehat { R } _ { \varepsilon }$ takes values in the finite set $\{ 0 , Q ^ { - 1 } , \ldots , 1 \}$ , its minimum is attained.

For a confidence level $\delta \in ( 0 , 1 )$ , choose the bufer to be

$$
\varepsilon _ { N } : = \sqrt { \frac { 2 \log ( 8 K A Q / \delta ) } { N } } .\tag{23}
$$

The logarithm comes from requiring uniform control over all training questions, models, and candidate answers. The key is the $N ^ { - 1 / 2 }$ scale.

The bufer prevents favorable finite-N fluctuations from creating empirical successes, but it may also count as errors questions that $w ^ { \star }$ solves with only a small winning margin. We control the population mass of such questions through the following one-sided condition.

Assumption 2.1 (One-sided margin at $w ^ { \star } )$ . There are constants $C _ { m } > 0 , \beta > 0$ , and $t _ { 0 } > 0$ such that

$$
\mathbb { P } \{ 0 < M _ { q } ( w ^ { \star } ) \le t \} \le C _ { m } t ^ { \beta } , \qquad 0 < t \le t _ { 0 } .\tag{24}
$$

This condition is imposed only at $w ^ { \star } ;$ ; no margin condition is required at the data-dependent estimator. Generalization across the $Q$ sampled questions contributes a separate error scale, which we denote by

$$
\mathfrak { v } _ { Q } ( \delta ) : = C _ { 0 } \sqrt { \frac { ( K - 1 ) \log ( e A Q ) + \log ( 8 / \delta ) } { Q } } ,\tag{25}
$$

where $C _ { 0 }$ is a universal constant.

Theorem 2.2 (Bufered ERM). Assume Theorem ${ \it 2 . 1 , }$ let $\varepsilon _ { N }$ be given by (23), and suppose $2 \varepsilon _ { N } \le t _ { 0 }$ . Let $\widehat { w } _ { \varepsilon _ { N } }$ be any minimizer in (22). Then, with probability at least $1 - \delta$ over both sources of randomness,

$$
R ( \widehat w _ { \varepsilon _ { N } } ) - R ( w ^ { \star } ) \leq 2 \mathfrak { v } _ { Q } ( \delta ) + C _ { m } ( 2 \varepsilon _ { N } ) ^ { \beta } .\tag{26}
$$

Substituting (23) and (25) gives the explicit rate stated in (9).

Proof. For $t \geq 0$ , define the clean population and empirical threshold risks

$$
R _ { t } ( w ) : = \mathbb { P } \{ M _ { q } ( w ) \leq t \} , \qquad \widetilde { R } _ { t } ( w ) : = \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } \mathbf { 1 } \{ M _ { q _ { j } } ( w ) \leq t \} .
$$

In particular, $\widetilde { R } _ { 0 }$ is the ordinary empirical risk that would be available if the clean answer distributions were known. It contains the randomness from sampling $Q$ questions, but not the finite-N generation noise present in $\widehat { R } _ { \varepsilon _ { N } }$ . Write ${ \mathfrak { v } } _ { Q } = { \mathfrak { v } } _ { Q } ( \delta )$ for brevity. The result follows from a chain of six inequalities. By Theorem A.2, with probability at least $1 - \delta / 4$ over the sampled questions, $| \widetilde { R } _ { t } ( w ) - R _ { t } ( w ) | \leq \mathfrak { v } _ { Q }$ simultaneously for all $w \in \Delta _ { K }$ and $t \in \{ 0 , 2 \varepsilon _ { N } \}$ . In particular,

$$
R _ { 0 } ( \widehat { w } _ { \varepsilon _ { N } } ) \stackrel { ( a ) } { \leq } \widetilde { R } _ { 0 } ( \widehat { w } _ { \varepsilon _ { N } } ) + \mathfrak { v } _ { Q } , \quad \widetilde { R } _ { 2 \varepsilon _ { N } } ( w ^ { \star } ) \stackrel { ( e ) } { \leq } R _ { 2 \varepsilon _ { N } } ( w ^ { \star } ) + \mathfrak { v } _ { Q } .
$$

Separately, Theorem A.1 shows, conditionally on the sampled questions, that event (7) holds with probability at least $1 - \delta / 4$ over the generations. On this event, the bufered sandwich (8) holds for every $w \in \Delta _ { K }$ and $j \in [ Q ]$ . Summing over $j$ and dividing by Q gives

$$
\widetilde { R } _ { 0 } ( \widehat { w } _ { \varepsilon _ { N } } ) \stackrel { ( b ) } { \leq } \widehat { R } _ { \varepsilon _ { N } } ( \widehat { w } _ { \varepsilon _ { N } } ) \stackrel { ( c ) } { \leq } \widehat { R } _ { \varepsilon _ { N } } ( w ^ { \star } ) \stackrel { ( d ) } { \leq } \widetilde { R } _ { 2 \varepsilon _ { N } } ( w ^ { \star } ) .
$$

The middle inequality (c) follows from the minimality of $\widehat { w } _ { \varepsilon _ { N } }$ for the bufered empirical risk. Finally, Theorem 2.1 gives

$$
R _ { 2 \varepsilon _ { N } } ( w ^ { \star } ) - R _ { 0 } ( w ^ { \star } ) = \mathbb { P } \{ 0 < M _ { q } ( w ^ { \star } ) \leq 2 \varepsilon _ { N } \} \stackrel { ( f ) } { \leq } C _ { m } ( 2 \varepsilon _ { N } ) ^ { \beta } .
$$

By a union bound, the two preceding events hold simultaneously with probability at least $1 - \delta$ Combining (a)–(f) in alphabetic order and using $R = R _ { 0 }$ proves (26). □

The two terms in (9) correspond to generalization across questions and estimation of their clean margins from finite generations. The core of the proof is the chain $\mathrm { ( b ) - ( d ) }$ , which compares the clean empirical risk of the data-dependent $\widehat { w } _ { \varepsilon _ { N } }$ directly with the 2ε -thresholded clean empirical risk of $w ^ { \star }$ . The outer inequalities follow from the two sides of the bufered sandwich, while the middle inequality is the empirical optimality of $\widehat { w } _ { \varepsilon _ { N } }$ . Consequently, no population margin condition is required at $\widehat { w } _ { \varepsilon _ { N } } .$ , or uniformly over w. The only margin-band cost occurs at the fixed $w ^ { \star }$ , where the identity preceding (f) identifies it with the probability of $0 < M _ { q } ( w ^ { \star } ) \leq 2 \varepsilon _ { N }$ , which Theorem 2.1 bounds. For the zero-bufer estimator, the analogue of inequality (b) would fail because a favorable fluctuation can make a question with clean margin in $( - \varepsilon _ { N } , 0 ]$ appear solved. A direct analysis would then have to control this negative-margin band at the data-dependent selected weight, typically through a population margin condition uniform over candidate weights or additional structure on the selection step.

Remark 2.3 (Computation). For fixed $\varepsilon ,$ , the bufered objective is piecewise constant on the arrangement of the empirical hyperplanes

$$
\widehat { p } _ { w } ( a ^ { \star } ( q _ { j } ) \mid q _ { j } ) - \widehat { p } _ { w } ( a \mid q _ { j } ) = \varepsilon , \qquad a \neq a ^ { \star } ( q _ { j } ) .
$$

It can be optimized by the same mixed-integer formulation used for weighted Best- $o f \mathrm { - } \infty$ , with the right-hand side shifted by ε. If a numerical solver returns $\widetilde { w } _ { \varepsilon }$ satisfying

$$
\widehat { R } _ { \varepsilon } ( \widetilde { w } _ { \varepsilon } ) \leq \operatorname* { i n f } _ { w \in \Delta _ { K } } \widehat { R } _ { \varepsilon } ( w ) + \eta ,
$$

the same proof adds only η to (26). Thus a computational tolerance no larger than the statistical bound in (9) does not change the rate.

## 2.3 Deployment with a finite generation budget

The clean risk concerns the population voting rule. If instead one is interested in performance at deployment under a limited generation budget, an additional source of sampling error comes into play. With finitely many generations on a new question, the magnitude of a positive (clean) margin also afects the reliability of the vote. We can account for this additional sampling error by increasing the training bufer.

For a fixed weight w and a fresh question $q \sim \mathcal { Q }$ , draw $Y _ { 1 } , \dots , Y _ { T }$ independently from $p _ { w } ( \cdot \mid q )$ independently of the training experiment. This is the independent-mixture implementation: each generation first selects a model according to $w ,$ and then samples from it. Form the empirical answer frequencies (or counts) based on $Y _ { 1 : T }$ and consider the plurality vote based on them. Define

$$
R _ { T } ^ { \mathrm { \tiny { d e p } } } ( w ) : = \mathbb { P } _ { q , Y _ { 1 : T } } \{ a ^ { \star } ( q ) \mathrm { ~ i s ~ n o t ~ t h e ~ u n i q u e ~ e m p i r i c a l ~ p l u r a l i t y } \} .\tag{27}
$$

Thus $T$ is the total number of deployment generations, whereas N counts training generations from each model on each training question. As in the clean loss, ties count as errors.

Corollary 2.4 (Finite deployment of bufered ERM). Assume Theorem 2.1, and $l e t \ \varepsilon _ { N }$ be given by (23). Fix an integer $T \geq 1$ and a reserve $r > 0$ before observing the training data, and suppose $2 \varepsilon _ { N } + r \le t _ { 0 }$ . Let $\widehat { w } _ { \varepsilon _ { N } + r }$ be any minimizer in (22) with bufer $\varepsilon _ { N } + r$ . Then, with probability at least $1 - \delta$ over the training experiment,

$$
R _ { T } ^ { \mathrm { { d e p } } } ( \widehat { w } _ { \varepsilon _ { N } + r } ) - R ( w ^ { \star } ) \leq 2 \mathfrak { v } _ { Q } ( \delta ) + C _ { m } ( 2 \varepsilon _ { N } + r ) ^ { \beta } + ( A - 1 ) e ^ { - T r ^ { 2 } / 2 } .\tag{28}
$$

The additional reserve controls questions whose positive margins are too small for the deployment vote to be reliable. The margin assumption is still needed only at $w ^ { \star }$ ; the proof is given in Section A.3. Taking $r _ { T } = \sqrt { 2 / T } \log ( e A T )$ makes the deployment tail decay faster than any fixed power of $T ^ { - 1 }$ without requiring knowledge of $\beta .$ . With K, A, the margin constants, and the confidence level fixed, the corollary gives

$$
R _ { T } ^ { \mathrm { { d e p } } } ( \widehat { w } _ { \varepsilon _ { N } + r _ { T } } ) \leq R ( w ^ { \star } ) + \widetilde O \Big ( Q ^ { - 1 / 2 } + N ^ { - \beta / 2 } + T ^ { - \beta / 2 } \Big ) ,\tag{29}
$$

whenever $2 \varepsilon _ { N } + r _ { T } \le t _ { 0 }$ . The comparison is with optimal clean risk. Learning weights that optimize finite-deployment accuracy or cost is a further problem.

## 3 A scalar antisymmetric two-model core

Bufered ERM thresholds the empirical margin of each training question before averaging across questions. To ask whether weaker finite-generation evidence can instead be pooled across questions, we now study a submodel in which the global statistical target becomes one-dimensional and can be analyzed sharply.

## 3.1 Model and exact decision reduction

There are two models and two answers {gold, other}. Each question q has a latent signed advantage $Z _ { q } \in [ - 1 , 1 ]$ and

$$
p _ { 1 } ( \mathrm { g o l d } \mid q ) = \frac { 1 + Z _ { q } } { 2 } , \qquad p _ { 2 } ( \mathrm { g o l d } \mid q ) = \frac { 1 - Z _ { q } } { 2 } .\tag{30}
$$

Let G denote the distribution of $Z _ { q }$ when $q \sim \mathcal { Q }$ . It is a probability measure supported on [−1, 1]. An ensemble weight can be parameterized by a single parameter as $w = ( t , 1 - t )$ with $0 \leq t \leq 1$ . In the two-answer case, the (clean) margin of the ensemble is the same weighted average of the models’ individual margins. Since the margin of model 1 is $Z _ { q }$ and that of model 2 is $- Z _ { q }$ , the ensemble margin is

$$
M _ { q } ( w ) = t ( Z _ { q } ) + ( 1 - t ) ( - Z _ { q } ) = ( 2 t - 1 ) Z _ { q } .\tag{31}
$$

Recalling the definition of the clean risk in (19), we have

$$
R ( w ) = \int \mathbf { 1 } \{ ( 2 t - 1 ) z \leq 0 \} ~ \mathrm { d } G ( z ) .
$$

Here and throughout, sg $\mathfrak { l } ( z ) = \mathbf { 1 } \{ z > 0 \} - \mathbf { 1 } \{ z < 0 \}$ , so that $\operatorname { s g n } ( 0 ) = 0$ . The following lemma formalizes that there are essentially two decision rules to consider.

Lemma 3.1 (Only the side of one half matters). Define

$$
\theta ( G ) : = \int _ { - 1 } ^ { 1 } \operatorname { s g n } ( z ) \mathrm { d } G ( z ) = G ( ( 0 , 1 ] ) - G ( [ - 1 , 0 ) ) .\tag{32}
$$

Every $t > 1 / 2$ has risk $G ( [ - 1 , 0 ] )$ , every $t < 1 / 2$ has risk $G ( [ 0 , 1 ] )$ , and $t = 1 / 2$ has risk one. Hence the optimal side is the sign of $\theta ( G )$ when $\theta ( G ) \neq 0 ;$ when $\theta ( G ) = 0$ , the two sides have equal risk. Choosing the wrong side incurs excess risk $| \theta ( G ) |$

Proof. The stated risks for the three decisions $t > 1 / 2 \ ( \mathrm { t h e } + \mathrm { d e c i s i o n } ) , t < 1 / 2 \ ( \mathrm { t h e } - \mathrm { d e c i s i o n } )$ and $t = 1 / 2$ follow from (31). The decision $t = 1 / 2$ is weakly dominated. Moreover,

$$
R ( - ) - R ( + ) = G ( [ 0 , 1 ] ) - G ( [ - 1 , 0 ] ) = G ( ( 0 , 1 ] ) - G ( [ - 1 , 0 ) ) = \theta ( G ) ,
$$

where the mass at zero cancels. Thus, when $\theta ( G ) > 0$ , the + decision is optimal and choosing − incurs excess risk $\theta ( G ) = | \theta ( G ) |$ . The case $\theta ( G ) < 0$ is similar, while the two decisions have equal risk when $\theta ( G ) = 0$

Thus, after excluding the weakly dominated choice $t = 1 / 2$ , the ensemble problem is to choose a side $\widehat { \sigma } \in \{ - 1 , + 1 \}$ . The + side is optimal when $\theta ( G ) > 0$ , the − side is optimal when $\theta ( G ) < 0$ and either side is optimal when $\theta ( G ) = 0$

The loss producing the regret (i.e. excess-risk) of a decision rule $\widehat { \sigma }$ is

$$
\begin{array} { r } { \mathcal { L } _ { G } ( \widehat { \sigma } ) : = | \theta ( G ) | \mathbf { 1 } \{ \widehat { \sigma } \theta ( G ) < 0 \} . } \end{array}\tag{33}
$$

In particular, the loss is zero when $\theta ( G ) = 0$ , because the two sides then have equal risk.

Although the original ensemble problem only requires choosing a side, it will be useful to consider the companion problem of estimating $\theta ( G )$ under absolute-error loss. Given any real-valued estimator ${ \widehat { \theta } } ,$ let its induced decision be $\widehat { \sigma } ( \widehat { \theta } ) = + \mathrm { 1 }$ when $\widehat { \theta } \geq 0$ and −1 otherwise. Then, for every $G ,$

$$
{ \mathcal { L } } _ { G } ( { \widehat { \sigma } } ( { \widehat { \theta } } ) ) \leq | { \widehat { \theta } } - \theta ( G ) | .\tag{34}
$$

Indeed, the left-hand side is zero when $\theta ( G ) = 0 ;$ otherwise, whenever the induced decision chooses the wrong side, $\widehat { \theta }$ lies on the opposite side of zero from $\theta ( G )$ . Thus any absolute-error bound for $\widehat { \theta }$ is also an upper bound on the regret of the induced decision rule.

## 3.2 What the generations reveal

We now formalize the reduced binomial-mixture experiment previewed in (13). Let $X _ { 1 q }$ and $X _ { 2 q }$ be the numbers of gold answers among the N generations of models 1 and 2 on question $q .$ Conditional on $Z _ { q } = z$ 2

$$
X _ { 1 q } \sim \mathrm { { B i n } } \left( N , { \frac { 1 + z } { 2 } } \right) , \qquad X _ { 2 q } \sim \mathrm { { B i n } } \left( N , { \frac { 1 - z } { 2 } } \right) , \quad \mathrm { { i n d e p e n d e n t l y . } }
$$

Define

$$
S _ { q } : = X _ { 1 q } + N - X _ { 2 q } .\tag{35}
$$

The joint likelihood of $( X _ { 1 q } , X _ { 2 q } )$ depends on z only through $S _ { q }$ , so $S _ { q }$ is suficient for z in this antisymmetric experiment. For the sampled questions, write $Z _ { j } : = Z _ { q _ { j } }$ and $S _ { j } : = S _ { q _ { j } }$ , and set $n : = 2 N$ . The complete hierarchical model is

$$
Z _ { 1 } , \dots , Z _ { Q } \stackrel { \mathrm { i i d } } { \sim } G , \qquad S _ { j } \mid Z _ { j } = z \sim \mathrm { B i n } \left( n , \frac { 1 + z } { 2 } \right) , \quad j = 1 , \dots , Q ,\tag{36}
$$

![](images/23db73b62f9d8493c8b04fb53a62d1522474a0dfda11d4aaada5d77cb864301a.jpg)

![](images/7e9026a5ee72ddeebb4b0b1f7985517d4a5138a2713cb8bf3a44168f9e03fa79.jpg)

![](images/38adae8b7aaad6026e2a05110d49745f20d3b3883f3152d62b861d7792420931.jpg)  
Figure 1: The scalar binomial channel and its two inferential targets, shown for the family in (39) at $\eta = 1 / 2$ and with $n = 1 2$ . Orange and blue distinguish the negative and positive latent mass of $G _ { \eta } ,$ , whose signed diference is $\theta ( G _ { \eta } )$ . The channel maps the continuous mixing law to the discrete count law $K _ { n } G _ { \eta }$ . The two statistical tasks are to estimate this imbalance or decide which side is larger; neither requires reconstructing $G _ { \eta }$ itself.

By suficiency, the raw pair of model counts can be reduced to $S _ { j }$ conditional on the observed question $q _ { j }$ , so the full antisymmetric data reduce to $( ( q _ { j } , S _ { j } ) ) _ { j = 1 } ^ { Q }$ . Passing from these pairs to $S _ { 1 } , \ldots , S _ { Q }$ alone is an additional observation coarsening. We analyze the resulting count-only experiment first and prove in Section 4.5 that its minimax lower bounds transfer, up to universal constants, to a fully nonparametric observed-question model.

For $k = 0 , \ldots , n .$ , let $b _ { k , n } ( p ) : = { \binom { n } { k } } p ^ { k } ( 1 - p ) ^ { n - k }$ denote the kth Bernstein basis polynomial of degree n. Let $K _ { n } G$ denote the marginal law of $S _ { j }$ after mixing over $Z _ { j } \sim G$ . As usual on a finite sample space, we identify this distribution with its pmf, so that

$$
( K _ { n } G ) ( k ) = \int _ { - 1 } ^ { 1 } b _ { k , n } \left( { \frac { 1 + z } { 2 } } \right) \mathrm { d } G ( z ) , \qquad k = 0 , \ldots , n .\tag{37}
$$

This is the nonparametric n-trial binomial-mixture distribution with mixing law G. Equivalently, the marginal observation model is

$$
S _ { 1 } , \ldots , S _ { Q } \stackrel { \mathrm { i i d } } { \sim } K _ { n } G .\tag{38}
$$

From these observations, the goal is either to estimate $\theta ( G )$ in absolute error or decide its sign.

More precisely, $K _ { n }$ is the Markov kernel from [−1, 1] to $\{ 0 , \ldots , n \}$ that assigns to each z the distribution $\mathrm { B i n } ( n , ( 1 + z ) / 2 )$ ; we refer to it as the n-trial binomial channel. We use the same notation for the induced map on probability measures, so that $K _ { n } G$ is the output distribution when the input $Z$ has distribution G, as displayed in (37). Thus (38) can be viewed as observing the infinite-dimensional mixing distribution G through the binomial channel.

To make the channel and its inferential targets concrete, consider the smooth family $G _ { \eta }$ with density

$$
g _ { \eta } ( z ) : = \frac { 1 0 5 } { 1 6 } z ^ { 2 } ( 1 - z ^ { 2 } ) ^ { 2 } ( 1 + \eta z ) , \qquad z \in [ - 1 , 1 ] , \quad | \eta | \le 1 .\tag{39}
$$

The odd tilt leaves the normalizing constant unchanged, and a direct calculation gives $G _ { \eta } ( [ - t , t ] ) \leq$ $( 3 5 / 8 ) t ^ { 3 }$ and $\theta ( G _ { \eta } ) = ( 3 5 / 6 4 ) \eta$ Figure 1 shows how the channel transforms $G _ { \eta }$ into the less informative observation law $K _ { n } G _ { \eta }$ , while the two inferential goals depend only on the imbalance between its colored lobes.

The following lemma describes an inherent limitation of this channel: it reveals only the first n moments of G.

Lemma 3.2. For probability measures G, H on $[ - 1 , 1 ]$ , the following are equivalent:

(a) $K _ { n } G = K _ { n } H ,$

(b) $\textstyle \int P \ d G = \int P$ dH for every algebraic polynomial P of degree at most $n ;$

$$
\begin{array} { r } { ( c ) \int z ^ { j } \mathrm { d } G ( z ) = \int z ^ { j } \mathrm { d } H ( z ) f o r j = 0 , \ldots , n . } \end{array}
$$

Proof. Equality of $K _ { n } G$ and $K _ { n } H$ is equivalent to equality of the integrals of $b _ { k , n } ( ( 1 + z ) / 2 )$ for $k = 0 , \ldots , n$ . The Bernstein polynomials form a basis for the polynomials of degree at most n in $p ,$ and composition with the invertible afine map $p = ( 1 + z ) / 2$ gives a basis for the polynomials of degree at most n in z. Monomials $z ^ { j } , j = 0 , \dots , n$ form another basis. □

In the count-only experiment, letting $Q  \infty$ identifies only $K _ { n } G$ , equivalently the first n moments of G. Consequently, the count-only observations need not identify $G ,$ or even $\theta ( G )$ . This is the source of the finite-generation identification floor quantified below. The observed-question experiment requires a separate minimax transfer argument, developed in Section 4.5.

## 4 Scalar minimax theory

The preceding section reduced the antisymmetric two-model problem to observing independent draws $S _ { 1 } , \ldots , S _ { Q }$ from the binomial-mixture $K _ { n } G .$ , with the goal of determining the sign of $\theta ( G )$ We now study the minimax dificulty of this decision problem under a two-sided margin condition on G.

We begin by showing that the Hellinger modulus of the binomial channel characterizes the minimax decision risk up to universal constants. We then evaluate its zero-radius limit to obtain the finite-generation identification floor, derive finite-Q obstructions, and show that these scalar lower bounds transfer when the questions themselves are observed and to the general ensemble problem. The next sections develop explicit estimators and use the Durrmeyer spectrum to prove matching rates not only at the two endpoints but also in a broad intermediate regime for $0 < \beta \leq 1$ the precise rate immediately above the first transition remains open.

## 4.1 Margin class and minimax risks

Fix $\beta > 0$ and $L \geq 1$ . Define the two-sided margin class

$$
\mathcal { G } _ { \beta , L } : = \left\{ G \in \mathcal { P } ( [ - 1 , 1 ] ) : G ( [ - t , t ] ) \leq L t ^ { \beta } \mathrm { ~ f o r ~ e v e r y ~ } 0 < t \leq 1 \right\} .\tag{40}
$$

The condition implies $G ( \{ 0 \} ) = 0$ . It is stronger than the one-sided condition at $w ^ { \star }$ used in Theorem 2.2; the symmetry is convenient for the minimax identification problem. The illustrative family in ${ \mathrm { F i g } } .$ . 1 belongs to $\mathcal { G } _ { 3 , 3 5 / 8 }$ for every $| \eta | \le 1$

For decision rules $\widehat { \sigma } \in \{ - 1 , \dot { + } 1 \}$ based on $S _ { 1 } , \ldots , S _ { Q }$ , define the minimax decision risk

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) : = \operatorname* { i n f } _ { \widehat { \sigma } } \operatorname* { s u p } _ { G \in \mathcal { G } _ { \beta , L } } \mathbb { E } _ { G } \mathcal { L } _ { G } ( \widehat { \sigma } ) .\tag{41}
$$

For estimators $\widehat { \theta }$ based on the same observations, define the minimax absolute-error risk

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { a b s } } ( \beta , L ) : = \operatorname* { i n f } _ { \widehat { \theta } } \operatorname* { s u p } _ { G \in \mathcal { G } _ { \beta , L } } \mathbb { E } _ { G } | \widehat { \theta } - \theta ( G ) | .
$$

The pointwise reduction (34) shows that

$$
\begin{array} { r } { \mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \leq \mathcal { R } _ { Q , n } ^ { \mathrm { a b s } } ( \beta , L ) . } \end{array}\tag{42}
$$

## 4.2 Exact characterization by the Hellinger modulus

For two probability vectors p, q on $\{ 0 , \ldots , n \}$ , let $\begin{array} { r } { h ^ { 2 } ( p , q ) = \sum _ { k = 0 } ^ { n } ( \sqrt { p _ { k } } - \sqrt { q _ { k } } ) ^ { 2 } } \end{array}$ denote their squared Hellinger distance. Define the indirect Hellinger modulus

$$
\omega _ { n , \beta , L } ( \varepsilon ) : = \operatorname* { s u p } \big \{ | \theta ( G ) - \theta ( H ) | : G , H \in \mathcal { G } _ { \beta , L } , \ h ( K _ { n } G , K _ { n } H ) \leq \varepsilon \big \} .\tag{43}
$$

This is the largest ambiguity in the target among latent distributions whose observable count laws are ε-indistinguishable.

The following theorem characterizes both scalar minimax risks, up to universal constants, through this modulus.

Theorem 4.1 (Scalar minimax characterization). There are universal constants $0 < c < C < \infty$ such that, for every $Q , n \geq 1$ ，

$$
c \omega _ { n , \beta , L } ( Q ^ { - 1 / 2 } ) \leq \mathcal { R } _ { Q , n } ^ { \operatorname* { d e c } } ( \beta , L ) \leq \mathcal { R } _ { Q , n } ^ { \mathrm { a b s } } ( \beta , L ) \leq C \omega _ { n , \beta , L } ( Q ^ { - 1 / 2 } ) .\tag{44}
$$

The upper bound follows by applying the abstract result in Theorem B.2 to the image of $\mathcal { G } _ { \beta , L }$ under the afine map $G \mapsto ( K _ { n } G , \theta ( G ) )$ , which is a compact convex subset of $\Delta _ { n + 1 } \times [ - 1 , 1 ]$ The proof of that abstract result follows the modulus-via-testing approach of Donoho and Liu (1991): robust tests separate target level sets, and a multiscale interval search turns a sequence of such tests into an estimator. The result does not require the observation map $G \mapsto K _ { n } G$ to be injective, so any residual nonidentifiability is retained through $\omega _ { n , \beta , L } ( 0 )$ . The lower decision bound requires an additional step: an arbitrary pair in the modulus may have target values with the same sign, and hence the same optimal decision. Reflection symmetrization produces an equally indistinguishable pair with opposite optimal decisions, after which a two-point testing argument applies. Both arguments initially evaluate the modulus at fixed numerical multiples of $\scriptstyle ( Q ^ { - 1 / 2 } )$ ; the scaling property in Theorem B.3 absorbs these multiples into the constants in (44). Full details are given in Section B.

Theorem 4.1 characterizes the minimax decision regret $\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L )$ in the scalar model, uniformly over $Q , n \geq 1$ , up to universal constant factors. Determining the rate as the two sample sizes vary therefore reduces to evaluating the modulus at resolution $\bar { Q } ^ { - 1 / 2 }$ . Because the modulus is itself a constrained supremum over pairs of mixing distributions, its scaling with $Q$ and n is not immediately clear. We next derive explicit bounds and identify regimes in which they match.

## 4.3 The $Q = \infty$ identification floor

We first consider the implication of Theorem 4.1 in the limit $Q  \infty$ . As the statistical resolution $Q ^ { - 1 / 2 }$ vanishes, the relevant quantity becomes the zero-radius modulus $\omega _ { n , \beta , L } ( 0 )$ . By Theorem 3.2, equality of the observable laws $K _ { n } G = K _ { n } H$ is equivalent to equality of the moments through degree n. Thus $\omega _ { n , \beta , L } ( 0 )$ measures the remaining ambiguity in $\theta ( G )$ when those moments are known exactly. The next theorem evaluates this identification width sharply.

Theorem 4.2 (Sharp identification floor). For every fixed $\beta > 0$ and $L \geq 1$ , there are constants $0 < c _ { \beta , L } < C _ { \beta , L } < \infty$ such that

$$
c _ { \beta , L } { n ^ { - \beta } } \leq \omega _ { n , \beta , L } ( 0 ) \leq C _ { \beta , L } n ^ { - \beta } \qquad f o r \ e v e r y \ n \geq 1 .\tag{45}
$$

Consequently,

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \ge c _ { \beta , L } n ^ { - \beta } \qquad f o r \ e v e r y \ Q .\tag{46}
$$

Since $n = 2 N$ , the finite-generation identification floor is $\Theta ( N ^ { - \beta } )$

![](images/0b6741e951ab6ee33e60eb900056af349d981ecf230036f0443bc48fc87b04c0.jpg)

![](images/c1adf7acfb43324f025dc7debc0becd03d61a14858130466f4c89457c965308e.jpg)  
(a) A nonidentifiable latent pair.  
(b) Their identical count laws.  
Figure 2: A numerical realization of the dual pair in (48) for $\beta = 2$ and $m = n = 7$ , corresponding to margin constant $L = 2$ . Since the dual witness is bang–bang, the Lebesgue densities of $G _ { 7 , + }$ and $G _ { 7 }$ <sub>−</sub> occupy complementary shells beneath the common envelope $2 | z |$ . The measures agree on every moment through degree seven, so their seven-trial binomial-mixture laws coincide even though their sign-imbalance functionals difer.

The minimax consequence (46) follows directly from Theorem 4.1 and monotonicity of the modulus: $\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \gtrsim \omega _ { n , \beta , L } ( Q ^ { - 1 / 2 } ) \geq \omega _ { n , \beta , L } ( 0 )$ . It therefore remains to establish the two bounds in (45).

For the upper bound, a localized Jackson-kernel construction gives a degree-n polynomial $P _ { n }$ which we call the localized Jackson polynomial, such that

$$
| P _ { n } ( z ) - \mathrm { s g n } ( z ) | \lesssim _ { s } \operatorname* { m i n } \{ 1 , ( n | z | ) ^ { - s } \} , \qquad \| P _ { n } \| _ { \infty } \leq 1 ,
$$

where $s > \beta$ . Integrating this pointwise bound under the margin condition gives $\textstyle { \int | \operatorname { s g n } - P _ { n } | \ \mathrm { d } G \lesssim _ { \beta , L } }$ $n ^ { - \beta }$ uniformly over $G \in { \mathcal { G } } _ { \beta , L }$ . If $K _ { n } G = K _ { n } H$ , then $G$ and H have identical moments through degree n, and hence

$$
\int P _ { n } ~ \mathrm { d } G = \int P _ { n } ~ \mathrm { d } H , \qquad | \theta ( G ) - \theta ( H ) | \leq \int | \operatorname { s g n } - P _ { n } | \ \mathrm { d } G + \int | \operatorname { s g n } - P _ { n } | \ \mathrm { d } H \lesssim _ { \beta , L } \ n ^ { - \beta } .
$$

Taking the supremum over all such pairs proves the upper bound on the identification width.

The lower bound uses a related but diferent approximation statement. The argument above requires a single polynomial whose error integrates uniformly over every $G \in { \mathcal { G } } _ { \beta , L }$ . For the lower bound, it sufices to work under the canonical measure

$$
\mu _ { \beta } ( \mathrm { d } z ) : = ( \beta / 2 ) | z | ^ { \beta - 1 } \mathbf { 1 } _ { [ - 1 , 1 ] } ( z ) \mathrm { ~ d } z
$$

and determine the best weighted approximation error. Writing $\Pi _ { n }$ for the real algebraic polynomials of degree at most $n ,$ Theorem E.3 shows that

$$
E _ { n } ( \beta ) : = \operatorname* { i n f } _ { P \in \Pi _ { n } } \int | \operatorname { s g n } ( z ) - P ( z ) | \mathrm { d } \mu _ { \beta } ( z ) \asymp _ { \beta } n ^ { - \beta } .\tag{47}
$$

The upper inequality here follows from the localized Jackson construction; the reverse inequality uses the weighted inverse theorem of Kopotun (2015, Theorem 9.1).

The $L _ { 1 }$ analogue of linear-programming duality, applied to the optimization problem defining $E _ { n } ( \beta )$ , provides an odd function $h _ { n }$ with $\| h _ { n } \| _ { \infty } \leq 1$ that annihilates $\Pi _ { n }$ and attains the approximation error $E _ { n } ( \beta )$ . The measures

$$
\mathrm { d } G _ { n , + } = \left( 1 + h _ { n } \right) \mathrm { d } \mu _ { \beta } , \qquad \mathrm { d } G _ { n , - } = \left( 1 - h _ { n } \right) \mathrm { d } \mu _ { \beta }\tag{48}
$$

form a reflected pair of probability measures with identical moments through degree n, hence inducing the same binomial-mixture law, while their targets are separated by $2 E _ { n } ( \beta ) \asymp n ^ { - \beta }$ Figure 2 illustrates this exact nonidentifiability mechanism for one numerically constructed pair. Mixing both measures with the same symmetric distribution on $\{ - 1 , + 1 \}$ places them in $\mathcal { G } _ { \beta , L }$ while preserving moment matching and the order of the target separation. Full proofs of the localized Jackson construction, the weighted inverse argument, and the dual moment-matching construction are given in Section E.

The dual construction in Theorem E.4 is formulated for an arbitrary cancellation degree m. The choice $m = n$ made here cancels every moment visible through the n-trial binomial channel and hence gives exact equality of the count laws. The same family reappears in Theorem 6.1 with $m = d < n \colon$ : only the first d moments cancel, and the Durrmeyer spectrum controls the information that leaks through the remaining modes.

## 4.4 Finite-Q lower bounds

The modulus characterization also isolates two distinct finite-sample obstructions. The first is ordinary population uncertainty: even when every latent advantage is separated from zero, Q questions reveal their population imbalance only at the root-Q scale.

Proposition 4.3 (Hard-margin root-Q lower bound). For every $n \geq 1 , \ \beta > 0 , \ L \geq 1$ , and $0 < \varepsilon \le 1$ 2

$$
\omega _ { n , \beta , L } ( \varepsilon ) \geq \varepsilon .\tag{49}
$$

Consequently, for every $Q , n \geq 1$ ，

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \geq c Q ^ { - 1 / 2 }\tag{50}
$$

for a universal constant $c > 0$ . Both claims hold on a two-atom subclass supported on $Z \in \{ - 1 , + 1 \}$ and therefore under a hard margin.

Indeed, under

$$
G _ { \sigma , \eta } = \frac { 1 + \sigma \eta } { 2 } \delta _ { + 1 } + \frac { 1 - \sigma \eta } { 2 } \delta _ { - 1 } , \qquad \sigma \in \{ - 1 , + 1 \} ,
$$

every observed count is deterministically 0 or n. The two output laws difer only through the population imbalance, and their Hellinger distance is of order $\eta ,$ while their targets difer by 2η. Thus finite-generation noise plays no role. The exact modulus calculation is given in Section C.

For comparison, an elementary construction based on a rare population of near-boundary questions with a common orientation gives the uniform lower bound $\mathcal { \bar { R } } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \gtrsim _ { \beta , L } ( Q n ) ^ { - \beta / ( \beta + 2 ) }$ Although it illustrates why pooling weak count evidence can help, it is dominated by the spectral lower bound throughout the asymptotic region in which it improves on both endpoint bounds. We give the construction and comparison in Section C.1.

## 4.5 Observed questions and minimax transfer

The count-only experiment in (38) deliberately discards the question identities. This is harmless for upper bounds, because a procedure in the original experiment can always ignore the questions. It is not harmless for lower bounds: an estimator that observes $q _ { j }$ may try to infer the latent advantage $Z ( q _ { j } )$ from the question itself. We now show that, in a fully nonparametric observed-question model, this extra information does not improve the minimax rate. The proof embeds an arbitrary scalar hard pair into deterministic question-to-advantage maps and randomizes only the assignment of their latent advantages to otherwise uninformative question cells.

We model the question population as a nonatomic standard probability space; see Theorem 4.7 for the role of this assumption and a countable alternative. Every such space is measure-isomorphic modulo null sets to $( [ 0 , 1 ] , B , \lambda )$ , which we therefore use as a canonical representation. No regularity or structural relation between q and $Z ( q )$ is imposed. Let

$$
\mathfrak { Z } _ { \beta , L } : = \{ Z : [ 0 , 1 ] \to [ - 1 , 1 ] \mathrm { ~ m e a s u r a b l e : ~ } Z _ { \# } \lambda \in \mathcal { G } _ { \beta , L } \} ,
$$

where $Z _ { \# } \lambda$ is the pushforward law of $Z ( q )$ for $q \sim \lambda$ . For $Z \in 3 _ { \beta , L }$ , the observed-question antisymmetric experiment is

$$
q _ { 1 } , \ldots , q _ { Q } \stackrel { \mathrm { i i d } } { \sim } \lambda , \qquad S _ { j } \mid q _ { j } \sim \mathrm { B i n } \biggl ( n , \frac { 1 + Z ( q _ { j } ) } { 2 } \biggr ) , \quad j = 1 , \ldots , Q .\tag{51}
$$

The estimator sees the full pairs $( q _ { j } , S _ { j } )$ . Define

$$
\theta _ { Z } : = \int _ { 0 } ^ { 1 } \mathrm { s g n } ( Z ( q ) ) \ \mathrm { d } q = \theta ( Z _ { \# } \lambda )
$$

and

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { o b s } } ( \beta , L ) : = \operatorname* { i n f } _ { \widehat { \boldsymbol { \sigma } } } \operatorname* { s u p } _ { Z \in \mathfrak { I } _ { \beta , L } } \mathbb { E } _ { Z } [ | \theta _ { Z } | \mathbf { 1 } \{ \widehat { \boldsymbol { \sigma } } \theta _ { Z } < 0 \} ] ,\tag{52}
$$

where the infimum ranges over all, possibly randomized, rules based on $( ( q _ { j } , S _ { j } ) ) _ { j = 1 } ^ { Q }$ . The raw pair of model counts may be used instead of $S _ { j }$ without changing the experiment, because $S _ { j }$ is suficient conditional on $q _ { j }$ in the antisymmetric family.

Every count-only rule is admissible in (52). Moreover, every $G \in { \mathcal { G } } _ { \beta , I }$ <sub>L</sub> can be realized as $Z _ { \# } \lambda$ by a quantile map. Hence

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { o b s } } ( \beta , L ) \leq \mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) .\tag{53}
$$

The reverse comparison is the nontrivial direction.

The reverse comparison uses two auxiliary constructions. First, an arbitrary scalar mixing distribution is approximated by an equal-weight discrete measure using outward quantiles. Each mass block is moved to its endpoint farther from zero, preserving the margin bound, while the target and induced count law converge to their original values. Equal weights then allow the M atoms to be assigned exactly to M equal-mass question cells. We average over uniform random permutations of this assignment. After conditioning away collisions among both the sampled question cells and the assigned atom indices, the resulting augmented experiment agrees with its count-only counterpart; the total discrepancy is $O ( Q ^ { 2 } / M )$ . Applying this construction to a reflected pair from the Hellinger modulus and letting M grow gives the lower bound. Details are provided in Section D.

Theorem 4.4 (Transfer from the scalar count experiment to observed questions). There are universal constants $c _ { 0 } , c _ { 1 } > 0$ such that, for every $Q , n \geq 1 , \beta > 0$ , and $L \geq 1$ 2

$$
c _ { 1 } \omega _ { n , \beta , L } ( c _ { 0 } Q ^ { - 1 / 2 } ) \leq \mathcal { R } _ { Q , n } ^ { \mathrm { o b s } } ( \beta , L ) \leq \mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) .\tag{54}
$$

One may take $c _ { 0 } = 1 / 6 4$ and $c _ { 1 } = 1 / 1 6$

Combining Theorems 4.4 and 4.1 with Lemma B.3 yields the desired equivalence.

Corollary 4.5 (Count-only and observed-question minimax risks are equivalent). For fixed β, L and every $Q , n \geq 1$

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { o b s } } ( \beta , L ) \asymp \mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \asymp \omega _ { n , \beta , L } ( Q ^ { - 1 / 2 } ) ,\tag{55}
$$

where the implicit constants are universal.

Corollary 4.6 (Transfer to general ensembles). Fix $K \geq 2$ and $A \geq 2$ . For each $Z \in 3 _ { \beta , L }$ , consider the binary-answer K-model environment on $( [ 0 , 1 ] , \lambda )$ in which models 1 and 2 obey (30), while every model $i \geq 3$ returns either answer with probability $1 / 2$ . Then

$$
\operatorname* { i n f } _ { \hat { w } } \operatorname* { s u p } _ { Z \in \mathfrak { I } _ { \beta , L } } \mathbb { E } _ { Z } \bigg [ R _ { Z } ( \widehat { w } ) - \operatorname* { i n f } _ { w \in \Delta _ { K } } R _ { Z } ( w ) \bigg ] \geq \mathcal { R } _ { Q , 2 N } ^ { \mathrm { o b s } } ( \beta , L ) \gtrsim \omega _ { 2 N , \beta , L } ( Q ^ { - 1 / 2 } ) .\tag{56}
$$

Consequently, the root-Q, identification-floor, and spectral lower bounds obtained in the scalar model hold for any larger ensemble class containing this subclass. The proof verifies the zero-margin case, the suficiency reduction, and the one-sided margin condition; see Section D.

Remark 4.7 (Fresh questions and finite strings). The nonatomic formulation keeps the two sample sizes conceptually separate: $Q$ counts fresh questions, while N is the generation budget attached to each question. If questions are instead represented literally as finite strings, let $\mathcal { X } = \Sigma ^ { \star }$ with the discrete sigma-field $2 ^ { \Sigma ^ { \star } }$ . The same constant-factor transfer rate holds for the distribution-free class

$$
\mathfrak { P } _ { \beta , L } ^ { \mathrm { o b s } } : = \{ ( \mathcal { Q } , Z ) : \mathcal { Q } \in \mathcal { P } ( \mathcal { X } ) , Z : \mathcal { X } \to [ - 1 , 1 ] , Z _ { \# } \mathcal { Q } \in \mathcal { G } _ { \beta , L } \} .
$$

Indeed, we can use the same construction, taking Q to be uniform on M distinct strings, randomly assigning the atoms of each equal-weight approximation to those strings, and then choosing $M \gg Q ^ { 2 }$ The proof otherwise repeats the discretization, permutation coupling, and collision argument above and is omitted.

## 5 Scalar estimators and spectral upper bounds

We now turn from the minimax characterization and lower bounds to explicit procedures. The two estimators developed below use the per-question generation counts in diferent ways. The plug-in estimator first converts each count into an estimated sign and then averages these decisions across questions. The afine moment estimator instead retains the count values and pools their information through a polynomial approximation to sgn.

The plug-in estimator matches the root-Q lower bound in the question-limited regime. For the afine moment estimator, we derive a spectral upper bound that reaches the $n ^ { - \beta }$ identification floor when $Q$ is suficiently large and also yields sharper guarantees at intermediate degrees. The matching spectral lower bound and the resulting middle regime are developed in Section 6.

## 5.1 Per-question plug-in estimator

For each question, define the majority-side score and its average by

$$
\widehat { \xi } _ { j } : = { \bf 1 } \{ S _ { j } > n / 2 \} - { \bf 1 } \{ S _ { j } < n / 2 \} , \qquad \widehat { \theta } _ { \mathrm { p l u g } } : = \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } \widehat { \xi } _ { j } .
$$

Thus $\widehat { \xi } _ { j }$ estimates $\operatorname { s g n } ( Z _ { j } )$ and equals zero at an exact tie. Set $\widehat \sigma _ { \mathrm { p l u g } } = + 1$ when $\widehat { \theta } _ { \mathrm { p l u g } } \geq 0$ and $\widehat \sigma _ { \mathrm { p l u g } } = - 1$ otherwise.

Theorem 5.1 (Plug-in estimator). For a constant $C = C ( \beta , L )$ ),

$$
\operatorname* { s u p } _ { G \in { \mathcal G } _ { \beta , L } } \mathbb { E } _ { G } \vert \widehat \theta _ { \mathrm { p l u g } } - \theta ( G ) \vert \leq C \left( Q ^ { - 1 / 2 } + n ^ { - \beta / 2 } \right) .\tag{57}
$$

The same expression upper-bounds the decision regret.

The proof, a direct conditional Hoefding bound integrated under the margin condition, is deferred to Section C.2.

This estimator makes a hard sign decision separately for each question. It is optimal in the question-limited regime but pays the larger finite-generation scale $n ^ { - \beta / 2 }$

Corollary 5.2 (Question-limited regime). $I f Q \leq c _ { 0 } n ^ { \beta }$ for a fixed constant c<sub>0</sub>, then

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \asymp _ { \beta , L , c _ { 0 } } Q ^ { - 1 / 2 } .\tag{58}
$$

Indeed, the plug-in upper bound is $O ( Q ^ { - 1 / 2 } )$ in this regime, while Theorem 4.3 gives the matching lower bound.

At the opposite endpoint, the pooled polynomial estimator developed next reaches the $n ^ { - \beta }$ identification floor once $Q$ is suficiently large.

## 5.2 The afine moment-estimation idea

A diferent estimator avoids per-question thresholding. Recall that $b _ { k , n } ( p ) : = { \binom { n } { k } } p ^ { k } ( 1 - p ) ^ { n - k }$ . Given $a = ( a _ { 0 } , \ldots , a _ { n } ) \in \mathbb { R } ^ { n + 1 }$ , define the statistic and its associated polynomial together by

$$
\widehat { \theta } _ { a } : = \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } a _ { S _ { j } } , \qquad P _ { a } ( z ) : = \sum _ { k = 0 } ^ { n } a _ { k } b _ { k , n } \left( \frac { 1 + z } { 2 } \right) .\tag{59}
$$

Since $\mathbb { P } ( S _ { j } = k \mid Z _ { j } = z ) = b _ { k , n } ( ( 1 + z ) / 2 )$ 2

$$
\mathbb { E } [ a _ { S _ { j } } \mid Z _ { j } = z ] = P _ { a } ( z ) , \qquad \mathbb { E } _ { G } \widehat { \theta } _ { a } = \int P _ { a } ( z ) \mathrm { d } G ( z ) .\tag{60}
$$

If the polynomial $P _ { a }$ approximates sgn, $\widehat { \theta }$ has a small bias; if the coeficients $a _ { k }$ are stable, in the sense of having a small spread, it has small variance. More precisely, define the coeficient spread

$$
A ( a ) : = \operatorname* { m a x } _ { k } a _ { k } - \operatorname* { m i n } _ { k } a _ { k } .
$$

The variable $a _ { S _ { j } }$ is supported on an interval of length $A ( a )$ , so a small spread controls its fluctuation uniformly over $G .$ . Designing a good estimator $\widehat { \theta } _ { a }$ is thus the problem of finding a stable polynomial approximation to a discontinuous function.

For analysis it is more convenient to reverse this parametrization and design an estimator by choosing its polynomial.

Lemma 5.3 (Bernstein parametrization). The map $a \mapsto P _ { a }$ is a linear bijection from $\mathbb { R } ^ { n + 1 }$ onto $\Pi _ { n }$ . Equivalently, every $P \in \Pi _ { n }$ has unique coeficients $a _ { 0 } ( P ) , \ldots , a _ { n } ( P )$ such that

$$
P ( z ) = \sum _ { k = 0 } ^ { n } a _ { k } ( P ) b _ { k , n } \bigg ( \frac { 1 + z } { 2 } \bigg ) .\tag{61}
$$

Proof. The functions $b _ { k , n } ( ( 1 + z ) / 2 ) , k = 0 , \dots , n$ , form a basis of $\Pi _ { n } \colon$ the Bernstein polynomials form a basis in $p ,$ and $p = ( 1 + z ) / 2$ is an invertible afine change of variable. □

For $P \in \Pi _ { n }$ , define

$$
\widehat { \theta } _ { P } : = \widehat { \theta } _ { a ( P ) } = \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } a _ { S _ { j } } ( P ) , \qquad A ( P ) : = A ( a ( P ) ) .\tag{62}
$$

Thus $\widehat { \theta } _ { a }$ and $\widehat { \theta } _ { P }$ are the same estimator family under two parametrizations: the coeficient parametrization makes the statistic transparent, while the polynomial parametrization is better suited to estimator design. Their performance is captured by the tradeof between approximation bias and coeficient stability.

Proposition 5.4 (Bias–stability bound). For every $P \in \Pi _ { n }$ and $G \in { \mathcal { G } } _ { \beta , L }$

$$
\mathbb { E } _ { G } | \widehat { \theta } _ { P } - \theta ( G ) | \leq \int | P ( z ) - \mathrm { s g n } ( z ) | \mathrm { d } G ( z ) + \frac { A ( P ) } { 2 \sqrt { Q } } .\tag{63}
$$

Proof. The conditional identity (60) gives $\begin{array} { r } { \mathbb { E } _ { G } \widehat { \theta } _ { P } = \int P ( z ) \ \mathrm { d } G ( z ) } \end{array}$ , and hence

$$
| \mathbb { E } _ { G } \widehat { \theta } _ { P } - \theta ( G ) | \leq \int | P ( z ) - \mathrm { s g n } ( z ) | ~ \mathrm { d } G ( z ) .
$$

Moreover, $a _ { S _ { j } } ( { \boldsymbol { P } } )$ is supported on an interval of length $A ( P )$ , so $\operatorname { V a r } _ { G } ( a _ { S _ { j } } ( P ) ) \leq A ( P ) ^ { 2 } / 4$ . Independence across questions and Jensen’s inequality give

$$
\mathbb { E } _ { G } \lvert \widehat { \theta } _ { P } - \mathbb { E } _ { G } \widehat { \theta } _ { P } \rvert \leq \sqrt { \mathrm { V a r } _ { G } ( \widehat { \theta } _ { P } ) } \leq \frac { A ( P ) } { 2 \sqrt { Q } } .
$$

Combining the bias and fluctuation bounds proves (63).

By (34), the same bound applies to the regret of the decision induced by $\widehat { \theta } _ { P }$ . Estimator design has therefore become a bias–stability tradeof: choose $P \in \Pi _ { r }$ that approximates sgn over the margin class while keeping the spread of its degree-n Bernstein coeficients small.

A self-contained Chebyshev-to-Bernstein calculation in Section F gives the following conservative coeficient bound.

Lemma 5.5 (Explicit Bernstein coeficient bound). There are universal constants $C _ { 0 } < \infty$ and $B _ { 0 } < 8$ such that, if P has degree at most d and $\| P \| _ { L _ { \infty } [ - 1 , 1 ] } \le 1$ , then every degree-n Bernstein coeficient of $P ( 2 p - 1 )$ , for every $n \geq d ,$ , satisfies

$$
| a _ { k } ( P ) | \leq C _ { 0 } B _ { 0 } ^ { d } .
$$

One may take $B _ { 0 } = 7 . 5$

Combining this lemma with the localized Jackson polynomial gives

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \lesssim \operatorname* { i n f } _ { 1 \leq d \leq n } \left\{ d ^ { - \beta } + \frac { B _ { 0 } ^ { d } } { \sqrt { Q } } \right\} .\tag{64}
$$

This already proves that the $n ^ { - \beta }$ identification floor is constructively attainable once $Q$ is exponential in n.

## 5.3 The Bernstein–Durrmeyer spectrum and estimator stability

The preceding coeficient bound $A ( P ) \lesssim B _ { 0 } ^ { d }$ is uniform over all trial budgets $n \geq d ;$ thus n enters only by restricting the admissible degree to $d \leq n$ . A sharper bound on $A ( P )$ , depending jointly on $( d , n )$ , follows from the spectral properties of the operator implicit in (60). More precisely, introduce the auxiliary reference experiment

$$
U \sim \operatorname { U n i f } [ 0 , 1 ] , \qquad S \mid U = u \sim \operatorname { B i n } ( n , u ) .
$$

The marginal law of S is uniform on $\{ 0 , \ldots , n \}$ , since $\begin{array} { r } { \mathbb { P } ( S = k ) = \int _ { 0 } ^ { 1 } b _ { k , n } ( u ) \mathrm { d } u = 1 / ( n + 1 ) } \end{array}$ . The uniform law of $U$ is a reference measure for the operator calculation, not an assumption on the unknown mixing distribution G. Accordingly, consider $\mathbb { R } ^ { n + 1 }$ with normalized inner product, i.e.,

$$
\mathsf { H } _ { n } : = L _ { 2 } ( \{ 0 , \ldots , n \} , \nu _ { n } ) , \qquad \langle a , b \rangle _ { \nu _ { n } } : = \frac { 1 } { n + 1 } \sum _ { k = 0 } ^ { n } a _ { k } b _ { k } ,
$$

where $\nu _ { n }$ is the uniform distribution on $\{ 0 , \ldots , n \}$ . The conditional-expectation operator

$$
T _ { n } : \mathsf { H } _ { n } \longrightarrow L _ { 2 } [ 0 , 1 ] , \qquad ( T _ { n } a ) ( u ) = \mathbb { E } [ a _ { S } \mid U = u ] = \sum _ { k = 0 } ^ { n } a _ { k } b _ { k , n } ( u )\tag{65}
$$

has range $\Pi _ { n }$ . Its adjoint reverses the conditioning:

$$
T _ { n } ^ { * } : L _ { 2 } [ 0 , 1 ] \longrightarrow \mathsf { H } _ { n } , \qquad ( T _ { n } ^ { * } f ) ( k ) = \mathbb { E } [ f ( U ) \mid S = k ] = ( n + 1 ) \int _ { 0 } ^ { 1 } f ( u ) b _ { k , n } ( u ) \mathrm { d } u ,\tag{66}
$$

where $U \mid S = k \sim \operatorname { B e t a } ( k + 1 , n - k + 1 )$ . Thus

$$
D _ { n } : = T _ { n } T _ { n } ^ { * }
$$

is the classical Bernstein–Durrmeyer operator, with $( D _ { n } f ) ( u ) = \mathbb { E } [ \mathbb { E } [ f ( U ) \mid S ] \mid U = u ]$ . Equivalently, $D _ { n }$ is the Markov smoothing operator obtained by drawing $S \sim \mathrm { B i n } ( n , u )$ and then $U ^ { \prime } \sim \mathrm { B e t a } ( S +$ $1 , n - S + 1 )$ ; its value at u is $\mathbb { E } [ f ( U ^ { \prime } ) \mid U = u ]$ . It is a positive, self-adjoint operator of rank $n + 1$ with range $\Pi _ { n }$

Let $L _ { 0 } , L _ { 1 } , \ldots$ . be the orthonormal shifted Legendre polynomials, obtained by orthogonalizing $1 , u , u ^ { 2 } , \ldots$ . in $L _ { 2 } [ 0 , 1 ]$ ; they turn out to be the eigenfunctions of $D _ { n }$ (Derriennic, 1981). More precisely:

Proposition 5.6 (Durrmeyer spectral decomposition). For $0 \leq j \leq n$

$$
D _ { n } L _ { j } = \lambda _ { j , n } L _ { j } , \qquad \lambda _ { j , n } = \frac { n ! ( n + 1 ) ! } { ( n - j ) ! ( n + j + 1 ) ! } = \prod _ { r = 0 } ^ { j - 1 } \frac { n - r } { n + r + 2 } .\tag{67}
$$

Define $v _ { j } : = \lambda _ { j , n } ^ { - 1 / 2 } T _ { n } ^ { * } L _ { j }$ . Then $v _ { 0 } , \ldots , v _ { n }$ form an orthonormal basis of $\mathsf { H } _ { n }$ , and

$$
T _ { n } a = \sum _ { j = 0 } ^ { n } { \sqrt { \lambda _ { j , n } } } \left. a , v _ { j } \right. _ { \nu _ { n } } L _ { j } .\tag{68}
$$

The eigenvalue formula is classical; Section F.2 gives the short proof, which relies on degree preservation— $\Pi _ { j }$ is an invariant subspace of $D _ { n } -$ and orthogonality of $L _ { j }$ to $\Pi _ { j - 1 }$ . Equation (68) is the singular value decomposition (SVD) of $T _ { n }$ . The singular values $\sqrt { \lambda _ { j , n } }$ quantify how strongly the binomial channel attenuates polynomial mode $j ,$ while $\kappa _ { j , n } = \lambda _ { j , n } ^ { - 1 / 2 }$ is the corresponding inverse amplification.

The SVD of $T _ { n }$ can be used to control $a ( P )$ . We continue to write $\Pi _ { d }$ for the vector space of polynomials of degree at most $d ;$ the ambient interval will be clear from the argument. The afine change of variables below identifies its realizations on [−1, 1] and [0, 1]. For $P$ on [−1, 1], define

$$
{ \widetilde P } ( u ) : = P ( 2 u - 1 ) ,\tag{69}
$$

which is a polynomial on [0, 1] of the same degree. Then the degree-n Bernstein coeficient vector introduced in (61) is characterized by

$$
T _ { n } a ( P ) = { \widetilde { P } } .
$$

Every such polynomial $\widetilde { P }$ of degree at most d has a unique expansion in $L _ { 0 } , \ldots , L _ { d }$

Proposition 5.7. Let $P \in \Pi _ { d } , d \leq n$ , and write its shifted-Legendre expansion as $\begin{array} { r } { \widetilde { P } = \sum _ { j = 0 } ^ { d } c _ { j } L _ { j } } \end{array}$ Then its degree-n Bernstein coeficient vector satisfies

$$
{ \frac { 1 } { n + 1 } } \sum _ { k = 0 } ^ { n } a _ { k } ( P ) ^ { 2 } = \sum _ { j = 0 } ^ { d } { \frac { c _ { j } ^ { 2 } } { \lambda _ { j , n } } } .\tag{70}
$$

In particular, since $\kappa _ { d , n } : = \lambda _ { d , n } ^ { - 1 / 2 }$ , one has

$$
A ( P ) \leq 2 \sqrt { n + 1 } \kappa _ { d , n } \| \widetilde { P } \| _ { L _ { 2 } [ 0 , 1 ] } .\tag{71}
$$

Identity (70) is Parseval’s identity in $\mathsf { H } _ { n } { \mathrm { : } }$ although a degree-d polynomial may have all $n + 1$ degree-n Bernstein coeficients nonzero, its coeficient vector has only $d + 1$ nonzero coordinates in the singular basis $v _ { 0 } , \ldots , v _ { n }$

The inverse singular values have two useful asymptotic regimes. For $d \leq n / 2$

$$
c \frac { d ( d + 1 ) } { n } \leq \log \kappa _ { d , n } \leq C \frac { d ( d + 1 ) } { n } ,\tag{72}
$$

where $c , C > 0$ are numerical constants. Thus

$$
\kappa _ { d , n } = \exp \{ \Theta ( d ^ { 2 } / n ) \} \qquad ( d \ll n ) ,
$$

while $\kappa _ { n , n } \asymp 2 ^ { n } n ^ { - 1 / 4 }$ . The elementary calculation is given in Section F.2.

Applying (71) to the bounded Jackson polynomial gives the refinement

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \lesssim \operatorname* { i n f } _ { 1 \leq d \leq n } \left\{ d ^ { - \beta } + \frac { \operatorname* { m i n } \{ B _ { 0 } ^ { d } , \sqrt { n + 1 } \kappa _ { d , n } \} } { \sqrt { Q } } \right\} .\tag{73}
$$

The improvement is present even at full degree: the endpoint asymptotic above gives the spectral stability factor $\sqrt { n + 1 } \kappa _ { n , n } \asymp 2 ^ { n } n ^ { 1 / 4 }$ , rather than $B _ { 0 } ^ { n }$ . For smaller d, (72) captures how stability varies jointly with d and $n .$ Taking $d = n$ , the moment estimator reaches risk $O ( n ^ { - \beta } )$ whenever

$$
Q \gtrsim 4 ^ { n } n ^ { 2 \beta + 1 / 2 } .\tag{74}
$$

Combined with Theorem 4.2, this gives a second endpoint regime.

Corollary 5.8 (Identification-limited regime). There is $C = C ( \beta , L )$ such that, $i f \ ( 7 4 )$ holds with a suficiently large constant, then

$$
\begin{array} { r } { \mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \asymp _ { \beta , L } n ^ { - \beta } . } \end{array}
$$

For intermediate degrees, (73) yields rates depending jointly on $Q$ and $n .$ . We make the degree choice explicit in the next section, where the resulting upper bound can be compared directly with the spectral moment-cancellation lower bound.

## 6 Spectral lower bounds and the middle regime

The preceding section used the Durrmeyer spectrum to control the stability of the polynomialmoment estimator. We now use its singular decomposition to quantify how dificult it is for the binomial channel to distinguish moment-matched alternatives. The resulting lower bound can be compared directly with the spectral upper bound, giving explicit rates and a matching middle regime.

## 6.1 Spectral lower bound

The construction reuses the dual family from Theorems 4.2 and E.4. At cancellation degree $d ,$ the two alternatives have the same moments up to degree d, while their values of θ difer by order $d ^ { - \beta }$ . In the shifted-Legendre basis, this cancellation removes modes $0 , \ldots , d$ from the observable perturbation. Every surviving mode $j \geq d + 1$ is multiplied by a singular value no larger than $\sqrt { \lambda _ { d + 1 , n } } = \kappa _ { d + 1 , n } ^ { - 1 }$ . When $d = n$ , no observable mode remains and the count laws coincide, recovering the identification floor. For $d < n$ , the following theorem quantifies the residual distinguishability.

Theorem 6.1 (Spectral moment-cancellation lower bound). Fix $0 < \beta \leq 1$ and $L \geq 1$ . There are constants $0 < c _ { \beta , L } < C _ { \beta , L } <$ ∞ such that, for every $n \geq 2$ and every integer $1 \leq d \leq n - 1$ ，

$$
\omega _ { n , \beta , L } \Bigg ( C _ { \beta , L } \frac { n + 1 } { \kappa _ { d + 1 , n } } \Bigg ) \geq c _ { \beta , L } d ^ { - \beta } .\tag{75}
$$

$I f 1 / 2 < \beta \leq 1$ , the following stronger bound holds:

$$
\omega _ { n , \beta , L } \left( \frac { C _ { \beta , L } } { \kappa _ { d + 1 , n } } \right) \geq c _ { \beta , L } d ^ { - \beta } .\tag{76}
$$

The factor $n + 1$ in (75) comes from controlling the spectral tail of an integrable, potentially non- $. L _ { 2 } .$ , perturbation using supremum norms of the shifted Legendre polynomials. When $\beta > 1 / 2$ the perturbation is square-integrable and Bessel’s inequality removes this factor, giving (76). A complete proof is given in Section G.

## 6.2 Explicit rates and the middle regime

The upper and lower spectral results use (72) in opposite directions. To display this comparison, define

$$
x _ { Q , n } : = \log \left( \frac { Q } { n ^ { \beta + 1 } } \right) , \qquad y _ { Q , n } : = \log \bigl ( e ( n + 1 ) \sqrt { Q } \bigr ) , \qquad y _ { Q } ^ { ( 2 ) } : = \log ( e \sqrt { Q } ) .\tag{77}
$$

The quantity $x _ { Q , n }$ measures the logarithmic question budget remaining in the upper bound after accounting for its polynomial prefactor. The quantity $y _ { Q , n }$ is the spectral attenuation required to place the lower-bound alternatives within Hellinger resolution $Q ^ { - 1 / 2 } ; y _ { Q } ^ { ( 2 ) }$ is its square-integrable refinement.

Proposition 6.2 (Explicit spectral rate bounds). Fix $\beta > 0$ and $L \geq 1$ . There are constants $c _ { 0 } , c , C > 0$ and $n _ { 0 } \geq 2$ , depending only on $( \beta , L )$ , such that, for every $n \geq n _ { 0 }$ , the following statements hold.

(a) If $0 < \beta \leq 1$ and $y _ { Q , n } \leq c _ { 0 } n$ , then

$$
\begin{array} { r } { \mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \geq c \left( n y _ { Q , n } \right) ^ { - \beta / 2 } . } \end{array}\tag{78}
$$

If $1 / 2 < \beta \leq 1$ , the same assertion holds with $y _ { Q } ^ { ( 2 ) }$ in place of $y _ { Q , n }$ , both in the condition and in the bound.

(b) $I f 1 \leq x _ { Q , n } \leq c _ { 0 } n .$ , then

$$
\begin{array} { r } { \mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \le C ( n x _ { Q , n } ) ^ { - \beta / 2 } . } \end{array}\tag{79}
$$

The bound is achieved by the polynomial-moment estimator with degree $d \asymp { \sqrt { n x _ { Q , n } } } .$

The degree choices make the comparison transparent. For the upper bound, take $d _ { \operatorname* { u p } } \asymp \sqrt { n x _ { Q , n } }$ with a suficiently small proportionality constant. The upper half of (72) then gives $\kappa _ { d _ { \mathrm { u p } } , n } \leq e ^ { x _ { Q , n } / 4 }$ and the stochastic term in (73) satisfies

$$
\frac { \sqrt { n + 1 } \kappa _ { d _ { \mathrm { u p } } , n } } { \sqrt { Q } } \lesssim n ^ { - \beta / 2 } e ^ { - x _ { Q , n } / 4 } \lesssim _ { \beta } ( n x _ { Q , n } ) ^ { - \beta / 2 } .
$$

For the lower bound, take $d _ { \mathrm { l o w } } \asymp \sqrt { n y _ { Q , n } }$ with a suficiently large proportionality constant. The lower half of (72) then makes $\kappa _ { d _ { \mathrm { l o w } } + 1 , n }$ at least a fixed multiple of $( n + 1 ) { \sqrt { Q } }$ . Thus (75) places alternatives separated by order $d _ { \mathrm { l o w } } ^ { - \beta }$ within Hellinger radius $Q ^ { - 1 / 2 }$ . The version using $y _ { Q } ^ { ( 2 ) }$ follows identically from (76). The complete calculations, including integer rounding and constants, are given in Section G.

When the two logarithmic scales $x _ { Q , n }$ and $y _ { Q , n }$ are comparable, the preceding upper and lower bounds match. This gives the following rigorous middle regime.

Theorem 6.3 (A rigorous middle regime). Fix $0 < \beta \leq 1 , L \geq 1$ , and $\eta > 0$ . There are constants $0 < c _ { \beta , L , \eta } < C _ { \beta , L , \eta } < \infty , c _ { \eta } > 0$ , and $n _ { 0 }$ such that, for every $n \geq n _ { 0 }$ and

$$
n ^ { \beta + 1 + \eta } \leq Q \leq e ^ { c _ { \eta } n } ,\tag{80}
$$

$$
c _ { \beta , L , \eta } [ n \log Q ] ^ { - \beta / 2 } \leq \mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \leq C _ { \beta , L , \eta } [ n \log Q ] ^ { - \beta / 2 } .\tag{81}
$$

The upper bound is attained by the pooled polynomial-moment estimator with degree $d \asymp { \sqrt { n \log Q } }$

Proof. Under the lower inequality in (80),

$$
x _ { Q , n } \asymp _ { \beta , \eta } y _ { Q , n } \asymp _ { \beta , \eta } \log Q .
$$

The upper inequality in (80), with $c _ { \eta }$ suficiently small, places both logarithmic scales in the ranges required by Theorem 6.2. Combining its lower and upper assertions proves (81). □

## 6.3 First transition and spectral conjecture

The middle theorem deliberately stays away from the transition at the end of the question-limited regime. Above that endpoint, the currently proved bounds include

$$
\operatorname* { m a x } \left\{ Q ^ { - 1 / 2 } , ( n y _ { Q , n } ) ^ { - \beta / 2 } \right\} \lesssim _ { \beta , L } \mathcal { R } _ { Q , n } ^ { \operatorname* { d e c } } ( \beta , L ) \lesssim _ { \beta , L } Q ^ { - 1 / 2 } + n ^ { - \beta / 2 } ,\tag{82}
$$

provided $0 < \beta \leq 1$ and $y _ { Q , n } \leq c _ { 0 } n$ , with $c _ { 0 }$ as in Theorem 6.2. When $1 / 2 < \beta \leq 1$ $y _ { Q , n }$ may be replaced by $y _ { Q } ^ { ( 2 ) }$ . For polynomial $Q$ with $n ^ { \beta } \ll Q \lesssim n ^ { \beta + 1 }$ , this leaves at most a logarithmic gap between $( n \log n ) ^ { - \beta / 2 }$ and $n ^ { - \beta / 2 }$ once the spectral term dominates the root-Q term.

The exact spectrum suggests that the intrinsic inverse dificulty at degree d is $\kappa _ { d , n }$ itself.

Conjecture 6.4 (Spectral minimax formula). For fixed $\beta , L$ , the scalar minimax risk satisfies

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \asymp _ { \beta , L } \Psi _ { \beta } ( Q , n ) , \qquad \Psi _ { \beta } ( Q , n ) : = \operatorname* { i n f } _ { 1 \leq d \leq n } \left\{ d ^ { - \beta } + \frac { \kappa _ { d , n } } { \sqrt { Q } } \right\} .\tag{83}
$$

Optimizing (83) using (72) gives the three-phase rate picture in (18); its interpretation and relation to the rigorous results are discussed in the introduction.

To see what the conjecture requires of the modulus, write $\varepsilon = Q ^ { - 1 / 2 }$ . By Theorems B.3 and 4.1, (83) predicts

$$
\omega _ { n , \beta , L } ( \varepsilon ) \asymp _ { \beta , L } \operatorname* { i n f } _ { 1 \leq d \leq n } \left\{ d ^ { - \beta } + \varepsilon \kappa _ { d , n } \right\} .
$$

At an interior optimizing degree, the two terms balance, so $\varepsilon \asymp d ^ { - \beta } / \kappa _ { d , n }$ and the infimum is of order $d ^ { - \beta }$ . Thus the conjecture predicts

$$
\omega _ { n , \beta , L } \bigg ( \frac { d ^ { - \beta } } { \kappa _ { d , n } } \bigg ) \asymp _ { \beta , L } d ^ { - \beta } ,\tag{84}
$$

where multiplicative constants in the argument of the modulus are absorbed into $\asymp _ { \beta , L }$ by Theorem B.3. By the definition of the modulus, the lower bound in (84) requires alternatives whose target values difer by order $d ^ { - \beta }$ while their count laws are within Hellinger distance of order $d ^ { - \beta } / \kappa _ { d , n } .$ The current moment-cancellation construction has the required target separation, but Theorem 6.1 places its count laws only within radius $( n + 1 ) / \kappa _ { d + 1 , n } .$ , improved to $1 / \kappa _ { d + 1 , n }$ when $\beta > 1 / 2$

The upper side has a separate stability gap. For a polynomial $P ,$ define the robust stability norm

$$
\mathsf { S } _ { n } ( P ) : = \operatorname* { s u p } _ { G \in \mathcal { G } _ { \beta , L } } \sqrt { \operatorname { V a r } _ { S \sim K _ { n } G } ( a _ { S } ( P ) ) } .
$$

An exact upper proof would follow from degree-d approximants satisfying

$$
\operatorname* { s u p } _ { G } \int | P _ { d } - \mathrm { s g n } | \mathrm { d } G \lesssim d ^ { - \beta } , \qquad 5 _ { n } ( P _ { d } ) \lesssim \kappa _ { d , n } .
$$

The present coeficient-range argument instead gives the larger stability factor $\sqrt { n + 1 } \kappa _ { d , n }$ . Closing these upper and lower gaps would establish the conjectured formula.

## 7 Discussion

We studied the problem of learning LLM ensemble weights from labeled questions and sampled answers, with performance measured by the clean risk. The interplay between Q and N afects how the observations should be used. Our general theory shows that estimating each question’s answer distribution separately is enough to attain the optimal rate, up to logarithmic factors, when questions are relatively scarce. Our scalar analysis shows that pooling generation counts can improve the rate once suficiently many questions are available. The root-Q obstruction persists even under a hard margin; improving it uniformly requires additional assumptions, such as a Bernstein condition linking loss disagreement to excess risk (Massart and N´ed´elec, 2006).

We study what sampled answers reveal without assuming an exploitable relation between question features and model answer distributions. That is, the LLM itself processes the question text, but we do not assume that its answer distribution can be predicted from that text alone, without generating answers. This provides a baseline for learning ensembles from sampled answers. Additional assumptions describing the predictive value of question texts could permit further gains.

We considered the scalar antisymmetric model mainly as a device to study the fundamental limits of learning from sampled answers, and it already supplies lower bounds for general ensembles (Theorem 4.6). However, it would be of interest to study two models with two possible answers per question without imposing antisymmetry, and determine whether matching upper bounds can be obtained. In this setting, a question has two latent signed advantages $( Z _ { 1 } , Z _ { 2 } )$ and the clean margin of the ensemble with weights $( t , 1 - t )$ on question q is $M _ { q } ( t ) = t Z _ { 1 } + ( 1 - t ) Z _ { 2 }$ . The learner must estimate and optimize the risk curve, $t \mapsto \mathbb { P } \{ t Z _ { 1 } + ( 1 - t ) Z _ { 2 } \leq 0 \}$ , over $t \in [ 0 , 1 ]$ and, for general K and multiple wrong answers, a family of polytopal decision risks. One possible approach is to approximate the loss indicators by multivariate polynomials and minimize the estimated surrogate risk over w. Our current scalar construction balances approximation error versus the variability induced by the polynomial’s Bernstein coeficients (Theorems 5.4 and 5.7). Extending this argument to multivariate surrogate losses would require similar control uniformly over the candidate weights. For general $K .$ , this could provide the matching upper bound when $Q$ is exponential in N or larger, addressing an open problem outlined in Section 1.1.

Choosing the approximation degree also raises an adaptation problem. The general bufer $\varepsilon _ { N }$ does not require knowledge of $\beta _ { \cdot }$ , while the pooling analysis is tuned to a margin class with known $\beta .$ In the general theorem, $\beta$ controls the frequency of small positive margins at $w ^ { \star } ;$ in the scalar class, it controls latent mass on both sides of zero. Inferring this behaviour from empirical margin histograms requires accounting for generation noise and, when the weights are fitted, their dependence on the training data. Adapting pooled estimators to unknown margin behaviour could also inform how a training budget is divided between more questions and more generations per question.

Deployment introduces an additional sampling problem. Theorem 2.4 gives a glimpse of what is possible, extending the bufered learning guarantee to $T$ deployment generations by reserving an additional margin. The result still compares with the optimal clean risk $( \mathrm { i . e . }$ , infinite-generation performance). Ensembles with similar clean risk may difer in both their mean generation cost $\bar { c } ( w )$ and the number of generations needed for reliable voting under deployment constraints. Establishing savings at a prescribed accuracy therefore requires accounting for both. To explore this trade-of, one could compare the deployment performance of the learned ensemble with that of the best ensemble under the same fixed generation budget. Another possibility is to allow the number of generations to vary across questions, subject to an average budget, and study adaptive stopping in that setting. We leave these questions for future work.

Acknowledgements. This work was supported in part by NSF grant DMS-1945667. We made use of large language models from OpenAI and Anthropic in developing this work. Their assistance included literature searches, estimator design, proof development and checking, and drafting and revising the manuscript. Both the mathematical arguments and their presentation were refined through sustained, iterative exchanges between the authors and these models. The authors guided this process and take full responsibility for the results and the final manuscript.

## References

U. Abel and E. E. Berdysheva. Complete asymptotic expansion for multivariate Bernstein–Durrmeyer operators and quasi-interpolants. Journal of Approximation Theory, 162(1):201–220, 2010. doi:10.1016/j.jat.2009.04.004.

J.-Y. Audibert and A. B. Tsybakov. Fast learning rates for plug-in classifiers. The Annals of Statistics, 35(2):608–633, 2007. doi:10.1214/009053606000001217.

P. Basu, B. Brill, and D. Yekutieli. Exact confidence intervals for the mixing distribution from binomial mixture distribution samples. Journal of Computational and Graphical Statistics, 35(2):880–890, 2026. doi:10.1080/10618600.2025.2573147.

B. Brown, J. Juravsky, R. Ehrlich, R. Clark, Q. V. Le, C. R´e, and A. Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv:2407.21787, 2024.

K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. arXiv:2110.14168, 2021.

M. M. Derriennic. Sur l’approximation de fonctions int´egrables sur [0, 1] par des polynˆomes de Bernstein modifi´es. Journal of Approximation Theory, 31(4):325–343, 1981. doi:10.1016/0021- 9045(81)90101-5.

Q. Di, K. Ji, X. Li, H. Zhao, and Q. Gu. Best-of-majority: Minimax-optimal strategy for pass@k inference scaling. arXiv:2510.03199, 2025.

D. L. Donoho and R. C. Liu. Geometrizing rates of convergence, II. The Annals of Statistics, 19(2):633–667, 1991. doi:10.1214/aos/1176348114.

A. Huang, A. Block, Q. Liu, N. Jiang, A. Krishnamurthy, and D. J. Foster. Is Best-of-N the best of them? Coverage, scaling, and optimality in inference-time alignment. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 25075–25126, 2025. arXiv:2503.21878.

A. B. Juditsky and A. S. Nemirovski. Nonparametric estimation by convex programming. The Annals of Statistics, 37(5A):2278–2300, 2009. doi:10.1214/08-AOS654.

J. Komiyama, D. Oba, and M. Oyamada. Best-of-∞: Asymptotic performance of test-time LLM ensembling. In The Fourteenth International Conference on Learning Representations, 2026. arXiv:2509.21091.

K. A. Kopotun. Polynomial approximation with doubling weights having finitely many zeros and singularities. Journal of Approximation Theory, 198:24–62, 2015. doi:10.1016/j.jat.2015.05.003.

P. Massart and E. N´ed´elec. Risk bounds for statistical learning. The Annals of Statistics, 34(5):2326– 2366, 2006. doi:10.1214/009053606000000786.

A. Mukherjee, M. Bullo, D. Basu, and D. G¨und¨uz. Test-time verification via optimal transport: Coverage, ROC, & sub-optimality. In The Fourteenth International Conference on Learning Representations, 2026. arXiv:2510.18982.

Y. Polyanskiy and Y. Wu. Dualizing Le Cam’s method for functional estimation I: General theory. The Annals of Statistics, 54(1):1–24, 2026. doi:10.1214/25-AOS2498.

Z. Szab´o, B. K. Sriperumbudur, B. P´oczos, and A. Gretton. Learning theory for distribution regression. Journal of Machine Learning Research, 17(152):1–40, 2016.

K. Tian, W. Kong, and G. Valiant. Learning populations of parameters. In Advances in Neural Information Processing Systems, volume 30, 2017. arXiv:1709.02707.

R. K. Vinayak, W. Kong, G. Valiant, and S. M. Kakade. Maximum likelihood estimation for learning populations of parameters. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 6448–6457, 2019.

X. Wang, J. Wei, D. Schuurmans, Q. V. Le, E. H. Chi, S. Narang, A. Chowdhery, and D. Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023. arXiv:2203.11171.

Y. Wu, Z. Sun, S. Li, S. Welleck, and Y. Yang. Inference scaling laws: An empirical analysis of compute-optimal inference for LLM problem-solving. In The Thirteenth International Conference on Learning Representations, 2025. arXiv:2408.00724.

T. Zaslavsky. Facing up to arrangements: Face-count formulas for partitions of space by hyperplanes. Memoirs of the American Mathematical Society, 1(154):vii+102, 1975. doi:10.1090/memo/0154.

## A Technical lemmas for bufered ERM

## A.1 Uniform accuracy of the empirical margins

For the proof, express the margins as afine comparisons in w. Set

$$
p . ( \mathfrak { a } \mid q ) : = \bigl ( p _ { 1 } ( \mathfrak { a } \mid q ) , \dotsc , p _ { K } ( \mathfrak { a } \mid q ) \bigr ) , \qquad \widehat { p } . ( \mathfrak { a } \mid q _ { j } ) : = \bigl ( \widehat { p } _ { 1 } ( \mathfrak { a } \mid q _ { j } ) , \dotsc , \widehat { p } _ { K } ( \mathfrak { a } \mid q _ { j } ) \bigr ) ,
$$

and define

$$
\Delta _ { q , a } : = p . ( a ^ { \star } ( q ) \mid q ) - p . ( a \mid q ) , \qquad \widehat { \Delta } _ { q _ { j } , a } : = \widehat { p } . ( a ^ { \star } ( q _ { j } ) \mid q _ { j } ) - \widehat { p } . ( a \mid q _ { j } ) .\tag{85}
$$

Then

$$
M _ { q } ( w ) = \operatorname* { m i n } _ { a \neq a ^ { \star } ( q ) } \left. w , \Delta _ { q , a } \right. , \qquad \widehat { M } _ { q _ { j } } ( w ) = \operatorname* { m i n } _ { a \neq a ^ { \star } ( q _ { j } ) } \left. w , \widehat { \Delta } _ { q _ { j } , a } \right. .\tag{86}
$$

Lemma A.1 (Uniform margin accuracy). With probability at least $1 - \delta / 4$ over the generations, conditional on $q _ { 1 } , \ldots , q _ { Q }$ 2

$$
\operatorname* { s u p } _ { j \leq Q } \operatorname* { s u p } _ { w \in \Delta _ { K } } \left| { \widehat M } _ { q _ { j } } ( w ) - M _ { q _ { j } } ( w ) \right| \leq \varepsilon _ { N } .\tag{87}
$$

Consequently, on this event, for every j and w,

$$
\mathbf { 1 } \{ M _ { q _ { j } } ( w ) \leq 0 \} \leq \mathbf { 1 } \{ \widehat { M } _ { q _ { j } } ( w ) \leq \varepsilon _ { N } \} \leq \mathbf { 1 } \{ M _ { q _ { j } } ( w ) \leq 2 \varepsilon _ { N } \} .\tag{88}
$$

Proof. For each $( j , i , a )$ , Hoefding’s inequality gives

$$
\mathbb { P } \left( \left| \widehat { p } _ { i } ( a \mid q _ { j } ) - p _ { i } ( a \mid q _ { j } ) \right| > \frac { \varepsilon _ { N } } { 2 } \right) \leq 2 \exp \left( - \frac { N \varepsilon _ { N } ^ { 2 } } { 2 } \right) .
$$

A union bound over at most $K A Q$ triples makes the probability of any violation at most $\delta / 4$ . On the complementary event,

$$
\left| \left. w , \widehat { \Delta } _ { q _ { j } , a } - \Delta _ { q _ { j } , a } \right. \right| \leq \varepsilon _ { N }
$$

for every $w \in \Delta _ { K }$ . Taking minima over wrong answers preserves this bound and gives (87); the indicator sandwich follows. □

## A.2 Uniform convergence across questions

Lemma A.2 (Uniform convergence of clean threshold risks). Fix two deterministic thresholds $s _ { 1 } , s _ { 2 } \geq 0$ . With probability at least $1 - \delta / 4$ over the questions,

$$
\operatorname* { s u p } _ { w \in \Delta _ { K } , t \in \{ s _ { 1 } , s _ { 2 } \} } \left. \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } \mathbf { 1 } \{ M _ { q _ { j } } ( w ) \leq t \} - \mathbb { P } \{ M _ { q } ( w ) \leq t \} \right. \leq \mathfrak { v } _ { Q } ( \delta ) .\tag{89}
$$

We prove the lemma by first identifying the empirical process that appears in (89). For a binary function f on the question space, write

$$
P f : = \mathbb { E } _ { q \sim \mathcal { Q } } f ( q ) , \qquad P _ { \cal Q } f : = \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } f ( q _ { j } ) .
$$

For a class H of binary functions, its growth function is

$$
\Pi _ { \mathcal { H } } ( m ) : = \operatorname* { s u p } _ { q _ { 1 } , \ldots , q _ { m } } \left| \left\{ ( h ( q _ { 1 } ) , \ldots , h ( q _ { m } ) ) : h \in \mathcal { H } \right\} \right| .\tag{90}
$$

For $t \geq 0$ , define

$$
f _ { w , t } ( q ) : = \mathbf { 1 } \{ M _ { q } ( w ) \leq t \} , \qquad \mathcal { F } _ { t } : = \{ f _ { w , t } : w \in \Delta _ { K } \} ,\tag{91}
$$

and let $\mathcal { F } _ { \mathrm { b u f } } : = \mathcal { F } _ { s _ { 1 } } \cup \mathcal { F } _ { s _ { 2 } }$ . With this notation, the left side of (89) is exactly

$$
\operatorname* { s u p } _ { f \in { \mathcal { F } } _ { \mathrm { b u f } } } | P _ { Q } f - P f | .\tag{92}
$$

We next bound the growth function of this class. The following classical face-count bound for hyperplane arrangements goes back to Zaslavsky (1975).

Lemma A.3 (Weak sign patterns of afine functions). Fix afine functions $g _ { 1 } , \dotsc , g _ { H } : \mathbb { R } ^ { p } \dotsc \mathbb { R }$ Their weak sign patterns are the vectors

$$
\left( \operatorname { s g n } g _ { 1 } ( x ) , \ldots , \operatorname { s g n } g _ { H } ( x ) \right) \in \{ - 1 , 0 , + 1 \} ^ { H } , \qquad x \in \mathbb { R } ^ { p } .
$$

The number of distinct weak sign patterns is at most

$$
B ( H , p ) : = \sum _ { r = 0 } ^ { p } 2 ^ { r } { \binom { H } { r } } .\tag{93}
$$

If $H \geq p \geq 1$ , then $B ( H , p ) \leq ( 2 e H / p ) ^ { p }$ . If $H < p$ , then $B ( H , p ) = 3 ^ { H }$ . In either case, log $B ( H , p ) \leq$ $C p \log ( e H )$ for a universal constant C.

Proof. Constant afine functions may be discarded because they do not increase the number of patterns. The hyperplanes $\{ g _ { h } = 0 \}$ partition $\mathbb { R } ^ { p }$ into relatively open faces of all dimensions. The weak sign vector is constant on each face, so it sufices to bound the total number of faces.

Let $F ( H , p )$ be the largest possible number of such faces. We prove the bound by induction on H. After arranging the first $H - 1$ hyperplanes, add the Hth. The preceding hyperplanes induce an arrangement of at most $H - 1$ hyperplanes on the new hyperplane, and hence at most $F ( H - 1 , p - 1 )$ faces there. Each induced face can split one old face into two while itself becoming a new face, producing a net increase of at most two. Therefore

$$
F ( H , p ) \leq F ( H - 1 , p ) + 2 F ( H - 1 , p - 1 ) ,
$$

with $F ( 0 , p ) = F ( H , 0 ) = 1$ . The quantity $B ( H , p )$ obeys the same recurrence by Pascal’s identity, so induction proves $F ( H , p ) \leq B ( H , p )$ . Finally, when $H \geq p$

$$
B ( H , p ) \leq \sum _ { r = 0 } ^ { p } { \binom { 2 H } { r } } \leq \left( { \frac { 2 e H } { p } } \right) ^ { p } .
$$

If $H < p ,$ then $\binom { H } { r } = 0$ for $r > H$ , and the binomial theorem gives $\begin{array} { r } { B ( H , p ) = \sum _ { r = 0 } ^ { H } 2 ^ { r } \binom { H } { r } = 3 ^ { H } } \end{array}$ The stated logarithmic bound follows from these two cases. □

Lemma A.4 (Growth of the clean margin classes). Let $p = K - 1$ . For every $m \geq 1$ and $t \geq 0$

$$
\Pi _ { { \mathcal { F } } _ { t } } ( m ) \leq B \big ( m ( A - 1 ) , p \big ) .\tag{94}
$$

Consequently,

$$
\Pi _ { \mathcal { F } _ { \mathrm { b u f } } } ( m ) \le 2 B \big ( m ( A - 1 ) , p \big ) , \qquad \log \Pi _ { \mathcal { F } _ { \mathrm { b u f } } } ( m ) \le C \big ( 1 + p \log ( e A m ) \big ) ,\tag{95}
$$

where C is universal.

Proof. Fix $q _ { 1 } , \ldots , q _ { m }$ and t. For every wrong answer $a \neq a ^ { \star } ( q _ { j } )$ , consider the afine function

$$
g _ { j , a } ( w ) : = \langle w , \Delta _ { q _ { j } , a } \rangle - t
$$

on the $( K - 1 )$ )-dimensional afine hull of $\Delta _ { K }$ . Since

$$
f _ { w , t } ( q _ { j } ) = 1 \quad \Longleftrightarrow \quad g _ { j , a } ( w ) \leq 0 { \mathrm { ~ f o r ~ a t ~ l e a s t ~ o n e ~ } } a \neq a ^ { \star } ( q _ { j } ) ,
$$

the label vector $( f _ { w , t } ( q _ { 1 } ) , \ldots , f _ { w , t } ( q _ { m } ) )$ is determined by the weak sign pattern of at most $m ( A - 1 )$ afine functions. Theorem A.3 proves (94).

The restrictions realized by a union of two classes are contained in the union of their two sets of restrictions. Hence $\Pi _ { \mathcal { F } _ { \mathrm { b u f } } } ( m ) \le \Pi _ { \mathcal { F } _ { s _ { 1 } } } ( m ) + \Pi _ { \mathcal { F } _ { s _ { 2 } } } ( m )$ , which gives the first bound in (95). The second follows from Theorem $\mathrm { A . 3 } { } _ { \cdot }$ , with the additive log 2 accounted for by the constant term. This also covers $p = 0$ , when $B ( H , 0 ) = 1$ □

We use the following standard consequence of symmetrization.

Lemma A.5 (Growth-function deviation inequality). Let H be a class of binary functions and let $P _ { Q }$ be the empirical measure of Q independent observations from P. For every $u > 0$

$$
\mathbb { P } \left\{ \operatorname* { s u p } _ { h \in \mathcal { H } } \left| P _ { Q } h - P h \right| > u \right\} \leq 2 \Pi _ { \mathcal { H } } ( 2 Q ) \exp \left( { - \frac { Q u ^ { 2 } } { 2 } } \right) .\tag{96}
$$

Proof. Write $Z : = \operatorname* { s u p } _ { h \in \mathcal { H } } | P _ { Q } h - P h |$ . Introduce an independent ghost sample with empirical measure $\begin{array} { r } { P _ { Q } ^ { \prime } = \frac { 1 } { Q } \sum _ { j = 1 } ^ { Q } \delta _ { q _ { i } ^ { \prime } } } \end{array}$ , and write $Z ^ { \prime } : = \operatorname* { s u p } _ { h \in \mathcal { H } } | P _ { Q } h - P _ { Q } ^ { \prime } h |$ . Conditional on the original sample, $P _ { Q } h - P h = \dot { \mathbb { E } } [ P _ { Q } h - \dot { P _ { Q } ^ { \prime } } h \ | \ q _ { 1 } , \dots , q _ { Q } ]$ for every h. Convexity of the supremum norm and of $x \mapsto e ^ { \lambda x }$ therefore gives $\mathbb { E } \overset { \cdot } { e } ^ { \lambda Z } \le \mathbb { E } e ^ { \lambda Z ^ { \prime } }$ for every $\lambda > 0$

By exchangeability of each original–ghost pair, $Z ^ { \prime }$ has the same distribution after independent random swaps. Encode the swaps by Rademacher signs $\sigma _ { 1 } , \ldots , \sigma _ { Q }$ . Conditional on the combined sample, a fixed restriction of h gives coeficients $d _ { j } = h ( q _ { j } ) - h ( q _ { j } ^ { \prime } ) \in \{ - 1 , 0 , + 1 \}$ , and the randomized diference is $\begin{array} { r } { Q ^ { - 1 } \sum _ { j = 1 } ^ { Q } \sigma _ { j } d _ { j } } \end{array}$ . There are at most $\Pi _ { \mathcal { H } } ( 2 Q )$ such restrictions. Hoefding’s exponentialmoment bound, together with $e ^ { \lambda | x | } \leq e ^ { \lambda x } + e ^ { - \lambda x }$ , thus gives

$$
\mathbb { E } e ^ { \lambda Z } \le 2 \Pi _ { \mathcal { H } } ( 2 Q ) \exp \left( \frac { \lambda ^ { 2 } } { 2 Q } \right) .
$$

Markov’s inequality with $\lambda = Q u$ now proves (96).

Proof of Theorem A.2. By (92), the supremum in (89) is $\mathrm { s u p } _ { f \in \mathcal { F } _ { \mathrm { b u f } } } \left| P _ { Q } f - P f \right|$ . Apply Theorem A.5 to $\mathcal { F } _ { \mathrm { b u f } }$ , take $u = \mathfrak { v } _ { Q } ( \delta )$ and use (95) with $m = 2 Q$ . Taking the universal constant $C _ { 0 }$ in (25) suficiently large makes the resulting probability at most $\delta / 4$ □

## A.3 Proof of the finite-deployment corollary

Proof of Theorem ${ \it 2 . 4 }$ . We first bound the error of the deployment vote on questions with margin greater than r. The enlarged training bufer will then control the population mass of the remaining questions.

Fix w and q with $M _ { q } ( w ) > r$ . For each wrong answer $a \neq a ^ { \star } ( q )$ , define the vote diferences

$$
D _ { t } ( a ) : = { \bf 1 } \{ Y _ { t } = a ^ { \star } ( q ) \} - { \bf 1 } \{ Y _ { t } = a \} , \qquad t \in [ T ] .
$$

Conditional on $q .$ these variables are independent, take values in $[ - 1 , 1 ]$ , and have mean $p _ { w } ( a ^ { \star } ( q ) \ |$ $q ) - p _ { w } ( a \mid q ) \geq M _ { q } ( w ) > r$ . Hoefding’s inequality gives

$$
\mathbb { P } \Bigg \{ \sum _ { t = 1 } ^ { T } D _ { t } ( a ) \leq 0 \Bigg | q \Bigg \} \leq e ^ { - T M _ { q } ( w ) ^ { 2 } / 2 } \leq e ^ { - T r ^ { 2 } / 2 } .
$$

The gold answer fails to be the unique empirical plurality only if this event occurs for some wrong answer. A union bound over at most $A - 1$ wrong answers therefore gives the conditional error bound $( A - 1 ) e ^ { - T r ^ { 2 } / 2 }$ . On questions with $M _ { q } ( w ) \leq r$ , we bound the conditional error probability by one. Averaging over questions gives

$$
R _ { T } ^ { \mathrm { d e p } } ( w ) \leq R _ { r } ( w ) + ( A - 1 ) e ^ { - T r ^ { 2 } / 2 } , \qquad R _ { r } ( w ) : = \mathbb { P } \{ M _ { q } ( w ) \leq r \} .\tag{97}
$$

This bound holds for every w. Conditional on the training data, the learned weight is fixed, and the fresh deployment question and generations have the law used above.

The rest of the argument follows the proof of Theorem 2.2, and we provide a brief sketch below. Let $\widetilde { R } _ { s }$ be the clean empirical threshold risk defined there. Apply Theorem $\mathrm { A . 2 }$ at the fixed thresholds r and $2 \varepsilon _ { N } + r$ . With probability at least $1 - \delta / 4$ over the questions, $| \widetilde { R } _ { s } ( w ) - R _ { s } ( w ) | \le \mathfrak { v } _ { Q } ( \delta )$ simultaneously for all w and both thresholds. Separately, Theorem A.1 bounds the uniform trainingmargin error by $\varepsilon N$ with probability at least $1 - \delta / 4$ , conditional on the questions. On this latter event, the bufered sandwich (8), with thresholds shifted by $^ { r , }$ holds for every $j \in [ Q ]$ and $w \in \Delta _ { K }$

$$
\mathbf { 1 } \{ M _ { q _ { j } } ( w ) \leq r \} \leq \mathbf { 1 } \{ \widehat { M } _ { q _ { j } } ( w ) \leq \varepsilon _ { N } + r \} \leq \mathbf { 1 } \{ M _ { q _ { j } } ( w ) \leq 2 \varepsilon _ { N } + r \} .
$$

Averaging over the training questions gives

$$
\widetilde { R } _ { r } ( w ) \leq \widehat { R } _ { \varepsilon _ { N } + r } ( w ) \leq \widetilde { R } _ { 2 \varepsilon _ { N } + r } ( w ) , \qquad w \in \Delta _ { K } .\tag{98}
$$

The two events hold jointly with probability at least $1 - \delta .$ . The same comparison argument now applies at these shifted thresholds: the sandwich and empirical optimality compare the clean empirical risks at $\widehat { w } _ { \varepsilon _ { N } + r }$ and $w ^ { \star }$ , while uniform convergence adds ${ \mathfrak { v } } _ { Q } ( \delta )$ at each end. Consequently,

$$
R _ { r } ( \widehat { w } _ { \varepsilon _ { N } + r } ) \leq R _ { 2 \varepsilon _ { N } + r } ( w ^ { \star } ) + 2 \mathfrak { v } _ { Q } ( \delta ) .
$$

Since $2 \varepsilon _ { N } + r \le t _ { 0 }$ , the one-sided margin condition gives

$$
R _ { 2 \varepsilon _ { N } + r } ( w ^ { \star } ) - R ( w ^ { \star } ) = \mathbb { P } \{ 0 < M _ { q } ( w ^ { \star } ) \leq 2 \varepsilon _ { N } + r \} \leq C _ { m } ( 2 \varepsilon _ { N } + r ) ^ { \beta } .
$$

Combining this inequality with (97) proves (28).

For (29), take $r _ { T } = \sqrt { 2 / T } \log ( e A T )$ . The last term in (28) becomes $( A - 1 ) \exp \{ - \log ^ { 2 } ( e A T ) \}$ , which, for fixed $A _ { i }$ , decreases faster than any fixed power of $T ^ { - 1 }$ . Also,

$$
( 2 \varepsilon _ { N } + r _ { T } ) ^ { \beta } \leq C _ { \beta } ( \varepsilon _ { N } ^ { \beta } + r _ { T } ^ { \beta } ) , \qquad r _ { T } ^ { \beta } = ( 2 / T ) ^ { \beta / 2 } \log ^ { \beta } ( e A T ) .
$$

Here one can take $C _ { \beta } = 3 ^ { \beta }$ for every $\beta > 0$ , since $2 \varepsilon _ { N } + r _ { T } \le 3 \operatorname* { m a x } \{ \varepsilon _ { N } , r _ { T } \}$ . Substitution of (23) and (25) now gives (29). □

## B Technical details for the scalar modulus

## B.1 Geometry of the margin class

Lemma B.1 (Basic properties of the margin class). The class $\mathcal { G } _ { \beta , L }$ is convex, invariant under reflection $G \mapsto G ^ { \dagger }$ , where ${ \cal G } ^ { \dagger } ( B ) = { \cal G } ( - B )$ , and compact for weak convergence. Every $G \in { \mathcal { G } } _ { \beta , L }$ satisfies $G ( \{ 0 \} ) = 0$ . The functional $G \mapsto \theta ( G )$ and the channel map $G \mapsto K _ { n } G$ are afine and continuous on $\mathcal { G } _ { \beta , L }$ . Moreover,

$$
\theta ( G ^ { \dagger } ) = - \theta ( G ) , \qquad ( K _ { n } G ^ { \dagger } ) ( k ) = ( K _ { n } G ) ( n - k ) , \quad k = 0 , \ldots , n .\tag{99}
$$

Proof. Convexity and reflection invariance are immediate from the definition of the class. Since $[ - 1 , 1 ]$ is compact, the set of all probability measures on it is weakly compact. It remains to show that the margin class is weakly closed. Suppose $G _ { m } \Rightarrow G$ . Fix $0 < t < 1$ and choose $\eta > 0$ such that $t + \eta < 1$ . By the Portmanteau theorem applied to the open interval $( - t - \eta , t + \eta )$ ,

$$
G ( [ - t , t ] ) \leq G ( ( - t - \eta , t + \eta ) ) \leq \operatorname* { l i m i n f } _ { m \to \infty } G _ { m } ( ( - t - \eta , t + \eta ) ) \leq L ( t + \eta ) ^ { \beta } .
$$

Letting $\eta \downarrow 0$ proves the margin inequality for $t < 1 ;$ at t = 1 it is automatic because $G ( [ - 1 , 1 ] ) =$ $1 \leq L$ . Thus $\mathcal { G } _ { \beta , L }$ is weakly closed and hence compact. Taking $t \downarrow 0$ shows that $G ( \{ 0 \} ) = 0$

For continuity of $\theta ,$ let $\psi _ { \delta }$ be the continuous clipped-sign function that equals −1 on $[ - 1 , - \delta ]$ equals $z / \delta$ on $[ - \delta , \delta ]$ , and equals 1 on [δ, 1]. Uniformly over the margin class,

$$
\bigg | \theta ( G ) - \int \psi _ { \delta } ~ \mathrm { d } G \bigg | \leq 2 G ( [ - \delta , \delta ] ) \leq 2 L \delta ^ { \beta } .
$$

The integral of $\psi _ { \delta }$ is weakly continuous. First letting $m  \infty$ and then $\delta \downarrow 0$ shows that $\theta ( G _ { m } ) $ $\theta ( G )$

Each coordinate of $K _ { n } G$ is the integral of a bounded continuous polynomial in $z ,$ so the channel map is weakly continuous; both the channel and target maps are afine by linearity of integration. Finally, reflection changes the sign of the target, and the Bernstein identity

$$
b _ { k , n } \bigg ( \frac { 1 - z } { 2 } \bigg ) = b _ { n - k , n } \bigg ( \frac { 1 + z } { 2 } \bigg )
$$

gives the channel identity in (99).

## B.2 A finite-sample Hellinger-modulus upper bound

The scalar characterization requires only an upper bound for absolute-error estimation. We give the needed result directly for compact convex categorical models. The proof follows the classical modulus-via-testing method of Donoho and Liu (1991): robust tests separate target level sets, and a multiscale interval search turns those tests into an estimator. Splitting the observations across scales makes the finite-sample expected-loss calculation straightforward.

Let X be a nonempty compact convex subset of $\Delta _ { m } \times \mathbb { R }$ , where $\Delta _ { m }$ denotes the probability simplex on a finite alphabet of size m. Write $x = ( p , \tau )$ for a generic element and define

$$
\omega _ { \mathcal { X } } ( r ) : = \operatorname* { s u p } \left. | \tau - \tau ^ { \prime } | : ( p , \tau ) , ( p ^ { \prime } , \tau ^ { \prime } ) \in \mathcal { X } , \ h ( p , p ^ { \prime } ) \leq r \right. .\tag{100}
$$

Theorem B.2 (Finite-sample Hellinger-modulus upper bound). Under $( p , \tau ) \in \mathcal { X }$ , observe $Y _ { 1 } , \ldots , Y _ { Q } \stackrel { \mathrm { i i d } } { \sim } p$ . Then

$$
\operatorname* { i n f } _ { \widehat { \tau } } \operatorname* { s u p } _ { ( p , \tau ) \in \mathcal { X } } \mathbb { E } _ { p } | \widehat { \tau } - \tau | \leq \frac { 7 } { 2 } \omega _ { \mathcal { X } } ( 3 2 Q ^ { - 1 / 2 } ) .\tag{101}
$$

The infimum is over all possibly randomized estimators $\widehat { \tau }$ based on these observations. The projection $( p , \tau ) \mapsto p$ need not be injective, so the result allows $\omega _ { \mathcal { X } } ( 0 ) > 0$

Two elementary ingredients precede the proof. The first controls how rapidly the modulus can change with its radius. It is a consequence of convexity, rather than an additional regularity assumption.

Lemma B.3 (Scaling of the convex Hellinger modulus). For $0 < \varepsilon _ { 1 } \le \varepsilon _ { 2 }$ 2

$$
\omega \chi ( \varepsilon _ { 1 } ) \leq \omega \chi ( \varepsilon _ { 2 } ) \leq \left( \frac { \varepsilon _ { 2 } } { \varepsilon _ { 1 } } \right) ^ { 2 } \omega \chi ( \varepsilon _ { 1 } ) .\tag{102}
$$

Consequently, replacing an argument of the modulus by any fixed positive multiple changes its value by at most a fixed multiplicative factor.

Proof. The first inequality follows because the set of pairs admissible at radius $\varepsilon _ { 1 }$ is contained in the set admissible at radius $\varepsilon _ { 2 }$ . For the second, take any $x _ { i } = ( p _ { i } , \tau _ { i } ) \in \mathcal { X } , i = 0 , 1$ , satisfying $h ( p _ { 0 } , p _ { 1 } ) \leq \varepsilon _ { 2 }$ . Let $\overline { { x } } = ( x _ { 0 } + x _ { 1 } ) / 2$ , and set

$$
\lambda = \left( \frac { \varepsilon _ { 1 } } { \varepsilon _ { 2 } } \right) ^ { 2 } , \qquad x _ { i } ^ { \lambda } = ( 1 - \lambda ) \overline { { x } } + \lambda x _ { i } , \quad i = 0 , 1 .
$$

Convexity keeps $x _ { 0 } ^ { \lambda } , x _ { 1 } ^ { \lambda }$ in X . Their target separation is $\lambda | \tau _ { 1 } - \tau _ { 0 } |$ , while afinity of the observation coordinate and joint convexity of squared Hellinger distance give

$$
h ^ { 2 } ( p _ { 0 } ^ { \lambda } , p _ { 1 } ^ { \lambda } ) \leq \lambda h ^ { 2 } ( p _ { 0 } , p _ { 1 } ) \leq \varepsilon _ { 1 } ^ { 2 } .
$$

Therefore $\lambda | \tau _ { 1 } - \tau _ { 0 } | \leq \omega _ { \mathcal { X } } ( \varepsilon _ { 1 } )$ . Dividing by λ and taking the supremum over pairs admissible at radius $\varepsilon _ { 2 }$ proves (102). □

The second ingredient constructs one test that is uniformly valid over two convex sets of categorical laws. This is stronger than testing only the pair of laws that are closest in Hellinger distance.

Lemma B.4 (Robust afinity test). Let $\mathcal { P } _ { 0 } , \mathcal { P } _ { 1 } \subset \Delta _ { m }$ be nonempty compact convex sets, and put

$$
\begin{array} { r } { r _ { * } : = \operatorname* { i n f } \{ h ( p , q ) : p \in \mathcal { P } _ { 0 } , \ q \in \mathcal { P } _ { 1 } \} . } \end{array}
$$

For every integer $k \geq 1$ , there is a possibly randomized test $\phi$ based on observations $Y _ { 1 } , \dots , Y _ { k }$ such that

$$
\operatorname* { s u p } _ { p \in \mathcal { P } _ { 0 } } \mathbb { E } _ { p ^ { \otimes k } } \phi \vee \operatorname* { s u p } _ { q \in \mathcal { P } _ { 1 } } \mathbb { E } _ { q ^ { \otimes k } } ( 1 - \phi ) \leq \left( 1 - \frac { r _ { * } ^ { 2 } } { 2 } \right) ^ { k } \leq \exp \left( - \frac { k r _ { * } ^ { 2 } } { 2 } \right) ,\tag{103}
$$

where $\phi = 1$ means choosing $\mathcal { P } _ { 1 }$

Proof. First suppose that every distribution in $\mathcal { P } _ { 0 } \cup \mathcal { P } _ { 1 }$ has strictly positive coordinates. Let $( p _ { 0 } , q _ { 1 } )$ maximize the Hellinger afinity

$$
\rho ( p , q ) : = \sum _ { i = 1 } ^ { m } \sqrt { p _ { i } q _ { i } }
$$

over $\mathcal { P } _ { 0 } \times \mathcal { P } _ { 1 }$ . For fixed $q _ { 1 }$ , the map $p \mapsto \rho ( p , q _ { 1 } )$ is concave. First-order optimality at $p _ { 0 }$ along every segment from p<sub>0</sub> to $p \in \mathcal { P } _ { 0 }$ gives

$$
\sum _ { i = 1 } ^ { m } p _ { i } { \sqrt { \frac { q _ { 1 i } } { p _ { 0 i } } } } \leq \sum _ { i = 1 } ^ { m } p _ { 0 i } { \sqrt { \frac { q _ { 1 i } } { p _ { 0 i } } } } = \rho ( p _ { 0 } , q _ { 1 } ) .\tag{104}
$$

Similarly, for every $q \in \mathcal { P } _ { 1 }$ 2

$$
\sum _ { i = 1 } ^ { m } q _ { i } \sqrt { \frac { p _ { 0 i } } { q _ { 1 i } } } \leq \rho ( p _ { 0 } , q _ { 1 } ) .\tag{105}
$$

Use the likelihood-ratio test between $p _ { 0 } ^ { \otimes k }$ and $q _ { 1 } ^ { \otimes k }$ , namely, $\begin{array} { r } { \phi = \mathbf { 1 } \left\{ \prod _ { j = 1 } ^ { k } q _ { 1 } ( Y _ { j } ) / p _ { 0 } ( Y _ { j } ) \geq 1 \right\} } \end{array}$ . For $p \in \mathcal { P } _ { 0 }$ , Markov’s inequality and (104) yield

$$
\mathbb { E } _ { p ^ { \otimes k } } \phi \le \mathbb { E } _ { p ^ { \otimes k } } \prod _ { j = 1 } ^ { k } \left( \frac { q _ { 1 } ( Y _ { j } ) } { p _ { 0 } ( Y _ { j } ) } \right) ^ { 1 / 2 } = \left[ \sum _ { i = 1 } ^ { m } p _ { i } \sqrt { \frac { q _ { 1 i } } { p _ { 0 i } } } \right] ^ { k } \le \rho ( p _ { 0 } , q _ { 1 } ) ^ { k } .
$$

The inverse likelihood ratio and (105) give the same bound for the other error. Since maximizing afinity is equivalent to minimizing Hellinger distance,

$$
\rho ( p _ { 0 } , q _ { 1 } ) = 1 - \frac { 1 } { 2 } r _ { * } ^ { 2 } ,
$$

which proves (103) under strict positivity.

To remove that assumption, let u be the uniform law and, for $\eta \in ( 0 , 1 )$ , set

$$
\mathcal { P } _ { j } ^ { \eta } : = \{ ( 1 - \eta ) p + \eta u : p \in \mathcal { P } _ { j } \} , \qquad j = 0 , 1 .
$$

Apply the preceding construction to obtain a randomized test $\phi _ { \eta } .$ , viewed as a vector in $[ 0 , 1 ] ^ { m ^ { k } }$ Along a sequence $\eta _ { \ell } \downarrow 0 .$ , compactness gives coordinatewise convergence to a randomized test $\phi .$ Uniform continuity of Hellinger afinity shows that the maximal afinities of the smoothed sets converge to the maximal afinity of the original sets. For each fixed $p \in \mathcal { P } _ { 0 }$ and $q \in \mathcal { P } _ { 1 }$ , the corresponding smoothed product laws converge in total variation to $p ^ { \otimes k }$ and $q ^ { \otimes k }$ . Together with coordinatewise convergence of the tests on the finite sample space, this implies convergence of their testing risks. Passing to the limit proves the result on the boundary of the simplex. □

Proof of Theorem B.2. Let

$$
\tau _ { - } : = \operatorname* { m i n } _ { ( p , \tau ) \in \mathcal { X } } \tau , \qquad \tau _ { + } : = \operatorname* { m a x } _ { ( p , \tau ) \in \mathcal { X } } \tau , \qquad D : = \tau _ { + } - \tau _ { - } .
$$

Because all categorical laws are within Hellinger distance ${ \sqrt { 2 } } ,$ we have $D = \omega _ { } \chi ( \sqrt { 2 } )$ . Set

$$
A : = 3 2 , \qquad \varepsilon _ { 0 } : = \frac { A } { \sqrt { Q } } , \qquad \delta : = \omega _ { \mathcal { X } } ( \varepsilon _ { 0 } ) .
$$

Our goal is to show an estimator that has expected absolute error at most $7 \delta / 2$ , uniformly over $\mathcal { X } .$ $\mathrm { ~ I f ~ } D = 0$ , the target is constant, and the corresponding constant estimator achieves zero error. If $\varepsilon _ { 0 } \geq \sqrt { 2 }$ , then $\delta = D$ , and the midpoint of $[ \tau _ { - } , \tau _ { + } ]$ has error at most $D / 2 = \delta / 2$ . We may therefore assume $\varepsilon _ { 0 } < \sqrt { 2 }$ . The scaling lemma gives

$$
D \leq \frac { 2 \delta } { \varepsilon _ { 0 } ^ { 2 } } .\tag{106}
$$

In particular, $\delta = 0$ would imply $D = 0 . { \mathrm { ~ H ~ } } \delta \geq D / 6$ , the midpoint again has error at most $D / 2 \leq 3 \delta$ It remains to consider

$$
0 < \delta < D / 6 ,\tag{107}
$$

which, together with (106), implies $Q > 3 A ^ { 2 }$

We now construct a sample-split interval-search estimator. It proceeds through stages $s =$ $0 , \ldots , S - 1$ and maintains a nested sequence of intervals $I _ { s }$ intended to localize $\tau .$ Start from $I _ { 0 } = [ \tau _ { - } , \tau _ { + } ]$ . The observations are divided into $S$ disjoint blocks, one per stage, with block sizes specified below. At stage $s ,$ write $I _ { s } = [ a _ { s } , b _ { s } ]$ and $\ell _ { s } = b _ { s } - a _ { s }$ . Using only block $s ,$ test the lower third of $I _ { s }$ against the upper third through the composite hypotheses

$$
H _ { s } ^ { - } : \tau \leq a _ { s } + \frac { \ell _ { s } } { 3 } \qquad \mathrm { a n d } \qquad H _ { s } ^ { + } : \tau \geq a _ { s } + \frac { 2 \ell _ { s } } { 3 } .
$$

If the test chooses $H _ { s } ^ { - }$ , retain the lower two thirds, $I _ { s + 1 } = [ a _ { s } , a _ { s } + 2 \ell _ { s } / 3 ]$ ; if it chooses $H _ { s } ^ { + }$ , retain the upper two thirds, $I _ { s + 1 } = [ a _ { s } + \ell _ { s } / 3 , b _ { s } ]$ . The two candidate intervals overlap in the middle third of $I _ { s } .$ . Thus, when $\tau$ lies in that middle third, either test outcome retains it.

The interval lengths are deterministic and satisfy

$$
\ell _ { s } = \left( \frac { 2 } { 3 } \right) ^ { s } D .
$$

Stop at the first integer S for which $\ell _ { S } \leq 6 \delta$ , and return the midpoint $\widehat { \tau }$ of $I _ { S }$ . It remains to choose the stagewise tests and block sizes. For $s < S$ , define

$$
g _ { s } : = \frac { \ell _ { s } } { 3 } , \qquad r _ { s } ^ { 2 } : = \frac { \varepsilon _ { 0 } ^ { 2 } \ell _ { s } } { 6 \delta } , \qquad u _ { s } : = \frac { \ell _ { s } } { \delta } > 6 .
$$

Here $g _ { s }$ is the separation between the two outer thirds, while $u _ { s }$ measures the current interval length in units of the target accuracy δ. Since $\ell _ { s } > 6 \delta$ , we have $r _ { s } \geq \varepsilon _ { 0 }$ , while (106) gives $r _ { s } ^ { 2 } \leq 1 / 3 < 2$ Thus $\varepsilon _ { 0 } \leq r _ { s } < \sqrt { 2 }$ , and Theorem B.3, applied at radii $\varepsilon _ { 0 }$ and $r _ { s }$ , gives

$$
\omega \chi ( r _ { s } ) \leq \left( \frac { r _ { s } } { \varepsilon _ { 0 } } \right) ^ { 2 } \delta = \frac { \ell _ { s } } { 6 } = \frac { g _ { s } } { 2 } < g _ { s } .\tag{108}
$$

For every possible interval $I _ { s } ,$ the observation laws p generated under $H _ { s } ^ { - }$ and $H _ { s } ^ { + }$ form two compact convex sets. Indeed, each is the image, under the coordinate projection $( p , \tau ) \mapsto p ,$ of the intersection of X with the corresponding closed halfspace. By (108), every pair across these sets is separated by more than $r _ { s }$ in Hellinger distance. If either set is empty, use the corresponding deterministic decision. Otherwise, apply the test supplied by Theorem B.4 using a fresh block of

$$
k _ { s } : = \left\lceil \frac { 4 } { r _ { s } ^ { 2 } } \log u _ { s } \right\rceil\tag{109}
$$

observations.

The use of a fresh block makes the testing guarantee valid conditional on the preceding outcomes. Let $\mathcal { F } _ { s }$ be the sigma-field generated by blocks $0 , \ldots , s - 1$ and any auxiliary randomization used through stage $s - 1$ , so that $I _ { s }$ is $\mathcal { F } _ { s }$ -measurable. Call the update at stage s unsafe if $I _ { s + 1 }$ does not contain $\tau ,$ and let $A _ { s }$ be the event that all preceding updates are safe. Since $I _ { 0 }$ contains $\tau$ and the intervals are nested, $A _ { s } = \{ \tau \in I _ { s } \}$ . On $A _ { s } ,$ either $\tau$ lies in the middle third, in which case an unsafe update is impossible, or the true law $p$ belongs to one of the two tested composite hypotheses. Since block s is independent of $\mathcal { F } _ { s }$ and has law $p ^ { \otimes k _ { s } }$ conditional on that history, the robust test bound gives

$$
\mathbb { P } _ { p } ( \tau \mathbin { \not \in } I _ { s + 1 } | \mathcal { F } _ { s } ) \leq \exp \left( - \frac { k _ { s } r _ { s } ^ { 2 } } { 2 } \right) \leq u _ { s } ^ { - 2 } = \left( \frac { \delta } { \ell _ { s } } \right) ^ { 2 } \qquad \mathrm { o n } ~ A _ { s } .\tag{110}
$$

We now bound the risk of this estimator. Let $E _ { s } : = A _ { s } \cap \{ \tau \notin I _ { s + 1 } \}$ be the event that the first unsafe update occurs at stage s. If no update is unsafe, then $\tau \in I _ { S }$ and $| \widehat { \tau } - \tau | \leq \ell _ { S } / 2 \leq 3 \delta$ . On $E _ { s }$ , all preceding updates were safe, so $\tau \in I _ { s }$ immediately before the unsafe update. Every later interval, and hence ${ \widehat { \tau } } ,$ remains inside $I _ { s } ;$ therefore the loss on $E _ { s }$ is at most $\ell _ { s }$ . Moreover, (110) gives

$$
\mathbb { P } _ { p } ( E _ { s } ) = \mathbb { E } _ { p } \left[ \mathbf { 1 } _ { A _ { s } } \mathbb { P } _ { p } \{ \tau \notin I _ { s + 1 } \mid \mathcal { F } _ { s } \} \right] \leq \left( { \frac { \delta } { \ell _ { s } } } \right) ^ { 2 } .
$$

The events $E _ { s }$ are disjoint, and consequently

$$
\operatorname* { s u p } _ { ( p , \tau ) \in \mathcal { X } } \mathbb { E } _ { p } | \widehat { \tau } - \tau | \leq 3 \delta + \sum _ { s = 0 } ^ { S - 1 } \ell _ { s } \left( \frac { \delta } { \ell _ { s } } \right) ^ { 2 } .
$$

Finally, $\ell _ { S - 1 } > 6 \delta$ and the interval lengths form a geometric sequence, so

$$
\sum _ { s = 0 } ^ { S - 1 } \frac { 1 } { \ell _ { s } } \le \frac { 1 } { \ell _ { S - 1 } } \sum _ { j = 0 } ^ { \infty } \left( \frac { 2 } { 3 } \right) ^ { j } < \frac { 1 } { 2 \delta } .
$$

Thus the expected loss is at most $7 \delta / 2$

It remains to verify that the planned independent blocks fit within the original sample. From (109),

$$
\sum _ { s = 0 } ^ { S - 1 } k _ { s } \le S + \frac { 2 4 Q } { A ^ { 2 } } \sum _ { s = 0 } ^ { S - 1 } \frac { \log u _ { s } } { u _ { s } } .\tag{111}
$$

Minimality of S gives $6 < u _ { S - 1 } \le 9$ . Writing $j = S - 1 - s$ , we have $u _ { s } = u _ { S - 1 } ( 3 / 2 ) ^ { j }$ , and hence

$$
\sum _ { s = 0 } ^ { S - 1 } \frac { \log u _ { s } } { u _ { s } } \leq \frac { 1 } { 6 } \sum _ { j = 0 } ^ { \infty } \left( \frac { 2 } { 3 } \right) ^ { j } \left[ \log 9 + j \log \left( \frac { 3 } { 2 } \right) \right] = \log \left( \frac { 9 } { 2 } \right) .\tag{112}
$$

Also,

$$
S \le 1 + \log _ { 3 / 2 } \left( \frac { D } { 6 \delta } \right) \le 1 + \frac { D } { 2 \delta } \le 1 + \frac { Q } { A ^ { 2 } } ,
$$

where the second inequality uses $\log _ { 3 / 2 } x \le 3 x$ for $x \geq 1$ . Combining this with Eqs. (111) and (112) gives

$$
\sum _ { s = 0 } ^ { S - 1 } k _ { s } \le 1 + \frac { Q } { A ^ { 2 } } + \frac { 2 4 Q } { A ^ { 2 } } \log \left( \frac { 9 } { 2 } \right) < Q ,
$$

using $A = 3 2$ and $Q > 3 A ^ { 2 }$ . Any unused observations are ignored. The estimator is therefore feasible with Q observations, which completes the proof of (101). □

## B.3 Proof of the scalar minimax characterization

Proof of Theorem 4.1. By Theorem B.1, the set

$$
\mathcal { X } _ { n } : = \{ ( K _ { n } G , \theta ( G ) ) : G \in \mathcal { G } _ { \beta , L } \} \subset \Delta _ { n + 1 } \times [ - 1 , 1 ]
$$

is compact and convex, and both coordinates are continuous afine functions of G. Its indirect modulus is exactly (43). Applying Theorem B.2 with $\mathcal { X } = \mathcal { X } _ { n }$ , followed by Theorem B.3, gives

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { a b s } } ( \beta , L ) \leq C \omega _ { n , \beta , L } ( Q ^ { - 1 / 2 } )
$$

for a universal constant C. This proves the rightmost inequality in (44); the pointwise reduction (42) gives the middle inequality.

It remains to prove the lower bound for decision regret. Fix $\varepsilon = ( 2 \sqrt { Q } ) ^ { - 1 }$ and take any $G , H \in { \mathcal { G } } _ { \beta , L }$ satisfying $h ( K _ { n } G , K _ { n } H ) \le \varepsilon$ . After interchanging G and H if necessary, write

$$
\delta : = \theta ( G ) - \theta ( H ) \geq 0 .
$$

The original targets may have the same sign, in which case the same side is optimal under both distributions and the pair need not be dificult for the decision problem. Reflection symmetrization produces a pair with opposite optimal decisions. Specifically, define

$$
G _ { + } : = { \frac { 1 } { 2 } } ( G + H ^ { \dagger } ) , \qquad G _ { - } : = { \frac { 1 } { 2 } } ( H + G ^ { \dagger } ) .
$$

By Theorem B.1, $G _ { + , } G _ { - } \in { \mathcal { G } } _ { \beta , L }$ and

$$
\theta ( G _ { + } ) = \frac { \delta } { 2 } , \qquad \theta ( G _ { - } ) = - \frac { \delta } { 2 } .
$$

Writing $P _ { + } = K _ { n } G _ { + }$ and $P _ { - } = K _ { n } G _ { - }$ , joint convexity of squared Hellinger distance and invariance under reversal of the count labels give

$$
\begin{array} { l } { { \displaystyle h ^ { 2 } ( P _ { + } , P _ { - } ) \leq \frac { 1 } { 2 } h ^ { 2 } ( K _ { n } G , K _ { n } H ) + \frac { 1 } { 2 } h ^ { 2 } ( K _ { n } H ^ { \dagger } , K _ { n } G ^ { \dagger } ) } } \\ { { \displaystyle \quad = h ^ { 2 } ( K _ { n } G , K _ { n } H ) \leq \varepsilon ^ { 2 } . } } \end{array}\tag{113}
$$

For probability measures P and R, write

$$
\rho ( P , R ) : = \sum _ { x } \sqrt { P ( x ) R ( x ) } = 1 - \frac { 1 } { 2 } h ^ { 2 } ( P , R )
$$

for their Hellinger afinity. Afinity factorizes over product measures, so

$$
\begin{array} { r l r } {  { h ^ { 2 } ( P ^ { \otimes Q } , R ^ { \otimes Q } ) = 2 \{ 1 - \rho ( P , R ) ^ { Q } \} } } \\ & { } & { = 2 \{ 1 - ( 1 - \frac { 1 } { 2 } h ^ { 2 } ( P , R ) ) ^ { Q } \} \le Q h ^ { 2 } ( P , R ) . } \end{array}\tag{114}
$$

The last inequality is Bernoulli’s inequality. Under the present Hellinger convention, Cauchy–Schwarz also gives

$$
\mathrm { T V } ( P , R ) = { \frac { 1 } { 2 } } \sum _ { x } | { \sqrt { P ( x ) } } - { \sqrt { R ( x ) } } | ( { \sqrt { P ( x ) } } + { \sqrt { R ( x ) } } )
$$

$$
\leq h ( P , R ) \sqrt { 1 - \frac { 1 } { 4 } h ^ { 2 } ( P , R ) } \leq h ( P , R ) .\tag{115}
$$

Combining (113)–(115) with $\varepsilon = ( 2 \sqrt { Q } ) ^ { - 1 }$ yields

$$
\mathrm { T V } ( P _ { + } ^ { \otimes Q } , P _ { - } ^ { \otimes Q } ) \leq \frac { 1 } { 2 } .\tag{116}
$$

Now fix any, possibly randomized, decision rule ${ \widehat { \sigma } } .$ . Because a wrong decision costs $\delta / 2$ under either $G _ { + }$ or $G _ { - }$ <sub>−</sub> (recall (33)),

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { F \in \mathcal { G } _ { \beta , L } } \mathbb { E } _ { F } \mathcal { L } _ { F } ( \widehat { \sigma } ) \geq \frac { 1 } { 2 } \left\{ \mathbb { E } _ { G _ { + } } \mathcal { L } _ { G _ { + } } ( \widehat { \sigma } ) + \mathbb { E } _ { G _ { - } } \mathcal { L } _ { G _ { - } } ( \widehat { \sigma } ) \right\} } & { } \\ { \displaystyle } & { \quad = \frac { \delta } { 4 } \left\{ P _ { + } ^ { \otimes Q } ( \widehat { \sigma } = - 1 ) + P _ { - } ^ { \otimes Q } ( \widehat { \sigma } = + 1 ) \right\} } \\ { \displaystyle } & { \geq \frac { \delta } { 4 } \left\{ 1 - \mathrm { T V } ( P _ { + } ^ { \otimes Q } , P _ { - } ^ { \otimes Q } ) \right\} \geq \frac { \delta } { 8 } . } \end{array}\tag{117}
$$

The penultimate inequality is Le Cam’s two-point inequality. Since (117) holds for every admissible pair G, H and every decision rule, taking first the infimum over $\widehat { \sigma }$ and then the supremum over all pairs in the definition of the modulus gives

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \ge \frac { 1 } { 8 } \omega _ { n , \beta , L } \bigg ( \frac { 1 } { 2 \sqrt { Q } } \bigg ) .
$$

Finally, Theorem B.3 gives

$$
\omega _ { n , \beta , L } \mathopen { } \mathclose \bgroup \left( \frac 1 { 2 \sqrt { Q } } \aftergroup \egroup \right) \geq \frac 1 4 \omega _ { n , \beta , L } \mathopen { } \mathclose \bgroup \left( Q ^ { - 1 / 2 } \aftergroup \egroup \right) ,
$$

which proves the lower bound in (44).

## C Finite-sample scalar bounds

Proof of Theorem $4 . 3 .$ Take the two distributions $G _ { \sigma , \eta }$ displayed after the proposition. Conditional on $Z = + 1$ , the count is deterministically $S = n$ , while conditional on $Z = - 1$ , it is $S = 0$ . Hence $K _ { n } G _ { + 1 , \eta }$ assigns masses $( 1 - \eta ) / 2$ and $( 1 + \eta ) / 2$ to 0 and $n ,$ , respectively, while $K _ { n } G _ { - 1 , \eta }$ interchanges these masses. Therefore

$$
h ^ { 2 } ( K _ { n } G _ { + 1 , \eta } , K _ { n } G _ { - 1 , \eta } ) = 2 - 2 \sqrt { 1 - \eta ^ { 2 } } \leq 2 \eta ^ { 2 } .
$$

For $0 < \varepsilon \le 1$ , take $\eta = \varepsilon / 2$ . The two channel laws are then within Hellinger distance $\varepsilon ,$ whereas

$$
| \theta ( G _ { + 1 , \eta } ) - \theta ( G _ { - 1 , \eta } ) | = 2 \eta = \varepsilon .
$$

Both measures belong to $\mathcal { G } _ { \beta , L }$ for every $\beta > 0$ and $L \geq 1$ , proving (49). The minimax conclusion (50) follows from the lower bound in Theorem 4.1. □

## C.1 An elementary rare-boundary construction

The following construction places a small fraction of the population near the decision boundary, with a common orientation that can be detected only by pooling its weak count evidence.

Theorem C.1 (Pooled rare-boundary lower bound). For every fixed $\beta > 0$ and $L \geq 1$ , there are constants $c _ { 0 } , c _ { 1 } > 0$ such that, for every $n \geq 1$ and $0 < \varepsilon \le c _ { 0 }$

$$
\omega _ { n , \beta , L } ( \varepsilon ) \geq c _ { 1 } \left( \frac { \varepsilon ^ { 2 } } { n } \right) ^ { \beta / ( \beta + 2 ) } .\tag{118}
$$

Consequently, there is a constant $c _ { \beta , L } > 0$ such that

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \ge c _ { \beta , L } ( Q n ) ^ { - \beta / ( \beta + 2 ) }\tag{119}
$$

for every $Q , n \geq 1$

Proof of Theorem C.1. Consider the fixed family

$$
\mathfrak { G } _ { \beta } : = \{ G _ { \sigma , m } : \sigma \in \{ - 1 , + 1 \} , \ 0 < m \leq 1 / 2 \} , \qquad G _ { \sigma , m } = \frac { 1 - m ^ { \beta } } { 2 } ( \delta _ { - 1 } + \delta _ { + 1 } ) + m ^ { \beta } \delta _ { \sigma m } .
$$

For $0 < t < m$ there is no mass in $[ - t , t ]$ , while for $m \leq t < 1$ the mass is $m ^ { \beta } \leq t ^ { \beta }$ . Hence the family lies in $\mathcal { G } _ { \beta , L }$ , independently of $Q , n ,$ , and $\theta ( G _ { \sigma , m } ) = \sigma m ^ { \beta }$

Let $P _ { \sigma , m } = K _ { n } G _ { \sigma , m }$ . Augmenting the observation by revealing whether the latent draw came from the central component and then applying data processing gives

$$
\mathrm { K L } ( P _ { + 1 , m } | | P _ { - 1 , m } ) \leq m ^ { \beta } \mathrm { K L } \bigg ( \mathrm { B i n } \left( n , \frac { 1 + m } { 2 } \right) \bigg | \bigg | \mathrm { B i n } \left( n , \frac { 1 - m } { 2 } \right) \bigg ) \leq 4 n m ^ { \beta + 2 } ,
$$

where the last inequality uses

$$
\mathrm { K L } \Bigg ( \mathrm { B e r } \left( { \frac { 1 + m } { 2 } } \right) \Bigg | \Bigg | \mathrm { B e r } \left( { \frac { 1 - m } { 2 } } \right) \Bigg ) = m \log { \frac { 1 + m } { 1 - m } } \leq 4 m ^ { 2 }
$$

for $m \le 1 / 2$ . Since squared Hellinger distance is bounded by Kullback–Leibler divergence, taking

$$
m = \left( \frac { \varepsilon ^ { 2 } } { 1 6 n } \right) ^ { 1 / ( \beta + 2 ) }
$$

for $\varepsilon \le c _ { 0 } ( \beta )$ ensures both $m \le 1 / 2$ and $h ( P _ { + 1 , m } , P _ { - 1 , m } ) \leq \varepsilon$ . Meanwhile,

$$
| \theta ( G _ { + 1 , m } ) - \theta ( G _ { - 1 , m } ) | = 2 m ^ { \beta } \asymp _ { \beta } \left( \frac { \varepsilon ^ { 2 } } { n } \right) ^ { \beta / ( \beta + 2 ) } .
$$

This proves (118). Its minimax consequence follows from Theorem 4.1, monotonicity of the modulus, and a fixed reduction of the constant multiplying $Q ^ { - 1 / 2 }$ when necessary. □

The lower bound describes a shared weak orientation among a rare set of near-boundary questions. Unlike sign-then-aggregate, the optimal test for this pair pools the weak count information. Combined with the two endpoint lower bounds, it gives

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \gtrsim _ { \beta , L } \operatorname* { m a x } \left\{ Q ^ { - 1 / 2 } , n ^ { - \beta } , ( Q n ) ^ { - \beta / ( \beta + 2 ) } \right\}\tag{120}
$$

for every $Q , n \geq 1$

The rare-boundary term can exceed both endpoint terms only when $0 < \beta < 1$ , in the intermediate range

$$
n ^ { 2 \beta / ( 2 - \beta ) } \lesssim Q \lesssim n ^ { \beta + 1 } .
$$

To compare it with the spectral obstruction, let $r _ { \mathrm { p o o l } } = ( Q n ) ^ { - \beta / ( \beta + 2 ) }$ and $r _ { \mathrm { s p } } = ( n y _ { Q , n } ) ^ { - \beta / 2 }$ . Direct calculation gives $r _ { \mathrm { s p } } \geq r _ { \mathrm { p o o l } }$ whenever

$$
Q \gtrsim n ^ { \beta / 2 } y _ { Q , n } ^ { ( \beta + 2 ) / 2 } .
$$

For $0 < \beta < 1$ , this condition holds throughout the asymptotic region in which the rare-boundary term can exceed both $Q ^ { - 1 / 2 }$ and $n ^ { - \beta }$ . Thus the moment-canceling alternatives in Section 6, rather than the single rare weak atom, determine the stronger known middle-regime obstruction.

## C.2 Proof of the plug-in bound

Proof of Theorem 5.1. Let $( Z , S , { \widehat { \xi } } )$ denote a generic copy of $( Z _ { j } , S _ { j } , \widehat { \xi } _ { j } )$ , where $\widehat { \xi } = \mathbf { 1 } \{ S > n / 2 \} -$ $1 \{ S < n / 2 \}$ . Then $\mathbb { E } _ { G } \widehat { \theta } _ { \mathrm { p l u g } } = \mathbb { E } _ { G } \widehat { \xi }$ , and the bias–fluctuation decomposition gives

$$
\mathbb { E } _ { G } \lvert \widehat { \theta } _ { \mathrm { p l u g } } - \theta ( G ) \rvert \leq \lvert \mathbb { E } _ { G } \widehat { \xi } - \theta ( G ) \rvert + \mathbb { E } _ { G } \lvert \widehat { \theta } _ { \mathrm { p l u g } } - \mathbb { E } _ { G } \widehat { \theta } _ { \mathrm { p l u g } } \rvert .
$$

Because the $\widehat { \xi } _ { j }$ are iid and bounded by one, the fluctuation term satisfies

$$
\mathbb { E } _ { G } | \widehat { \theta } _ { \mathrm { p l u g } } - \mathbb { E } _ { G } \widehat { \theta } _ { \mathrm { p l u g } } | \leq \sqrt { \mathrm { V a r } _ { G } ( \widehat { \theta } _ { \mathrm { p l u g } } ) } \leq Q ^ { - 1 / 2 } .
$$

It remains to control the bias. The margin condition implies $G ( \{ 0 \} ) = 0$ , so the value of sgn at zero is immaterial. For $Z \neq 0$ , the pointwise inequality $| { \widehat { \xi } } - \operatorname { s g n } ( Z ) | \leq 2 1 \{ { \widehat { \xi } } \neq \operatorname { s g n } ( Z ) \}$ gives

$$
\begin{array} { r l } & { | \mathbb { E } _ { G } \widehat { \xi } - \theta ( G ) | \leq \mathbb { E } _ { G } \big | \mathbb { E } [ \widehat { \xi } \mid Z ] - \mathrm { s g n } ( Z ) \big | } \\ & { \qquad \leq \mathbb { E } _ { G } \mathbb { E } \big [ | \widehat { \xi } - \mathrm { s g n } ( Z ) | \mid Z \big ] } \\ & { \qquad \leq 2 \mathbb { E } _ { G } \mathbb { P } \{ \widehat { \xi } \neq \mathrm { s g n } ( Z ) \mid Z \} . } \end{array}
$$

For $z \neq 0$ , conditional Hoefding bounds the probability of an incorrect sign or a tie by

$$
\mathbb { P } _ { z } \{ \widehat { \xi } \neq \operatorname { s g n } ( z ) \} \le e ^ { - n z ^ { 2 } / 2 } .
$$

Consequently,

$$
| \mathbb { E } _ { G } \widehat { \xi } - \theta ( G ) | \leq 2 \mathbb { E } _ { G } e ^ { - n Z ^ { 2 } / 2 } .\tag{121}
$$

To integrate this bound, put $a = n ^ { - 1 / 2 }$ and split the range of $| Z |$ into

$$
A _ { - 1 } : = \{ z \in [ - 1 , 1 ] : | z | \leq a \} , \qquad A _ { j } : = \{ z \in [ - 1 , 1 ] : 2 ^ { j } a < | z | \leq 2 ^ { j + 1 } a \} , \quad j \geq 0 .
$$

The margin condition, together with the trivial probability bound once $2 ^ { j + 1 } a > 1$ , gives

$$
G ( A _ { - 1 } ) \leq L n ^ { - \beta / 2 } , \qquad G ( A _ { j } ) \leq L 2 ^ { ( j + 1 ) \beta } n ^ { - \beta / 2 } .
$$

On $A _ { j }$ , one has $e ^ { - n Z ^ { 2 } / 2 } \le e ^ { - 2 ^ { 2 j - 1 } }$ . Therefore

$$
\mathbb { E } _ { G } e ^ { - n Z ^ { 2 } / 2 } \le L n ^ { - \beta / 2 } \left[ 1 + 2 ^ { \beta } \sum _ { j \ge 0 } 2 ^ { j \beta } e ^ { - 2 ^ { 2 j - 1 } } \right] \ \le C _ { \beta , L } n ^ { - \beta / 2 } ,
$$

because the series converges. Combining this estimate with (121) and the fluctuation bound proves (57). Finally, (34) gives the same upper bound for the regret of $\widehat { \sigma } _ { \mathrm { p l u g } }$ □

## D Transfer from count-only to observed-question experiments

The transfer proof has two auxiliary steps. First, we approximate each scalar mixing distribution by an equal-weight discrete measure that remains in the same margin class while preserving the target and observable count law asymptotically. Second, we realize such a measure as the pushforward of a deterministic question-to-advantage map by assigning its atoms to a fine equal-mass partition of the question space. A uniform random permutation of this assignment produces an observedquestion experiment that, after averaging, is close to the corresponding count-only experiment; the discrepancy is controlled by the probability of cell collisions. We apply these constructions to a reflected pair from the Hellinger modulus and use the resulting permutation priors to prove the minimax lower bound. Finally, embedding the two informative models among $K - 2$ neutral models transfers the bound to the general ensemble problem.

## D.1 Equal-weight discretization

Lemma D.1 (Equal-weight discretization preserving the margin class). For every $G \in { \mathcal { G } } _ { \beta , L }$ and every integer $M \geq 1$ , there is an equal-weight measure

$$
G ^ { ( M ) } = \frac { 1 } { M } \sum _ { r = 1 } ^ { M } \delta _ { z _ { r } } \in \mathcal { G } _ { \beta , L }
$$

such that

$$
| \theta ( G ^ { ( M ) } ) - \theta ( G ) | \leq M ^ { - 1 } , \qquad G ^ { ( M ) } \Rightarrow G \quad a s \ M \to \infty .
$$

Consequently, for every fixed $n _ { \mathrm { : } }$

$$
\mathrm { T V } ( K _ { n } G ^ { ( M ) } , K _ { n } G ) \to 0 .
$$

If H is the reflection of G under $z \mapsto - z$ , the approximations may be chosen so that $H ^ { ( M ) }$ is the reflection $o f G ^ { ( M ) }$

Proof of Lemma D.1. Because $G \in { \mathcal { G } } _ { \beta , L } , G ( \{ 0 \} ) = 0$ . Write

$$
p _ { + } = G ( ( 0 , 1 ] ) , \qquad p _ { - } = G ( [ - 1 , 0 ) ) , \qquad p _ { + } + p _ { - } = 1 ,
$$

and define the one-sided small-ball functions

$$
F _ { + } ( t ) = G ( ( 0 , t ] ) , \qquad F _ { - } ( t ) = G ( [ - t , 0 ) ) , \qquad 0 < t \leq 1 .
$$

The construction treats the two sides of zero separately. Starting at zero, divide as much of the mass on each side as possible into blocks of mass $1 / M$ , and replace each block by an atom at its endpoint farther from zero. Any atom needed to round the number of blocks is placed at $+ 1 \mathrm { o r } - 1$ Moving mass outward in this way is what preserves the margin bound.

To make this precise, choose an integer $m _ { + }$ nearest to $M p _ { + }$ and put $m _ { - } = M - m _ { + }$ . These are the numbers of atoms assigned to the positive and negative sides. They satisfy

$$
\left| \frac { m _ { + } } { M } - p _ { + } \right| \leq \frac { 1 } { 2 M } , \qquad \left| \frac { m _ { - } } { M } - p _ { - } \right| \leq \frac { 1 } { 2 M } .\tag{122}
$$

Let

$$
k _ { + } = \lfloor M p _ { + } \rfloor , \qquad k _ { - } = \lfloor M p _ { - } \rfloor , \qquad e _ { + } = m _ { + } - k _ { + } , \qquad e _ { - } = m _ { - } - k _ { - } .
$$

Here $k _ { \pm }$ count the complete mass- $1 / M$ blocks and $e _ { \pm }$ account for the possible rounding atom. Since $M p _ { + } + M p _ { - } = M$ , we have $e _ { + } , e _ { - } \in \{ 0 , 1 \}$ and $e _ { + } + e _ { - } \in \{ 0 , 1 \}$ . For $j = 1 , \ldots , k _ { + }$ and $j = 1 , \dots , k _ { - } .$ respectively, define the outward quantiles

$$
\begin{array} { r } { x _ { j } ^ { + } = \operatorname* { i n f } \{ t \in ( 0 , 1 ] : F _ { + } ( t ) \geq j / M \} , \qquad x _ { j } ^ { - } = \operatorname* { i n f } \{ t \in ( 0 , 1 ] : F _ { - } ( t ) \geq j / M \} . } \end{array}
$$

Thus $x _ { j } ^ { + }$ and $- { x } _ { j } ^ { - }$ are the endpoints farther from zero of the jth cumulative blocks on the two sides. Set

$$
G ^ { ( M ) } = \frac { 1 } { M } \left\{ \sum _ { j = 1 } ^ { k _ { + } } \delta _ { x _ { j } ^ { + } } + \sum _ { j = 1 } ^ { k _ { - } } \delta _ { - x _ { j } ^ { - } } + e _ { + } \delta _ { + 1 } + e _ { - } \delta _ { - 1 } \right\} .\tag{123}
$$

For $0 < t < 1$ , the generalized-inverse property gives $x _ { j } ^ { + } \leq t$ if and only if $j / M \le F _ { + } ( t )$ . Consequently, $\# \{ j : x _ { j } ^ { + } \leq t \} = \lfloor M F _ { + } ( t ) \rfloor$ ; the same argument applies on the negative side. Because the possible extra atoms at ±1 are not counted, we obtain

$$
G ^ { ( M ) } ( ( 0 , t ] ) = \frac { \lfloor M F _ { + } ( t ) \rfloor } { M } , \qquad G ^ { ( M ) } ( [ - t , 0 ) ) = \frac { \lfloor M F _ { - } ( t ) \rfloor } { M } .\tag{124}
$$

In particular, the discretization places no more mass near zero than the original distribution:

$$
G ^ { ( M ) } ( [ - t , t ] ) \leq F _ { + } ( t ) + F _ { - } ( t ) = G ( [ - t , t ] ) \leq L t ^ { \beta } .
$$

$\mathrm { A t } ~ t = 1$ , the required inequality is $1 \leq L$ . Hence ${ \cal G } ^ { ( M ) } \in { \mathcal G } _ { \beta , L }$ with exactly the same margin constant.

The positive mass of $G ^ { ( M ) }$ is $m _ { + } / M$ , so (122) gives

$$
\left| \theta ( G ^ { ( M ) } ) - \theta ( G ) \right| = 2 \left| \frac { m _ { + } } { M } - p _ { + } \right| \leq M ^ { - 1 } .
$$

Equation (124) shows directly that, for every $0 < t < 1$ 2

$$
0 \le F _ { + } ( t ) - G ^ { ( M ) } ( ( 0 , t ] ) < \frac { 1 } { M } , \qquad 0 \le F _ { - } ( t ) - G ^ { ( M ) } ( [ - t , 0 ) ) < \frac { 1 } { M } .
$$

$\mathrm { A t } ~ t = 1$ , the corresponding errors are at most $1 / ( 2 M )$ by (122). Thus the one-sided distribution functions of $G ^ { ( M ) }$ converge to those of G, which implies $G ^ { ( M ) } \Rightarrow G$ . Each coordinate of $K _ { n } G ^ { ( M ) }$ is the integral of a bounded continuous polynomial in z, so weak convergence implies that $K _ { n } G ^ { ( M ) }$ converges coordinatewise to $K _ { n } G$ . Since the output alphabet is finite, this implies

$$
\mathrm { T V } ( K _ { n } G ^ { ( M ) } , K _ { n } G ) \to 0 .
$$

Finally, if H is the reflection of $G ,$ define $H ^ { ( M ) }$ to be the reflection of $G ^ { ( M ) }$

## D.2 Random-permutation coupling

Lemma D.2 (Random-permutation embedding). Let $\begin{array} { r } { G _ { M } = M ^ { - 1 } \sum _ { r = 1 } ^ { M } \delta _ { z _ { r } } } \end{array}$ be an equal-weight measure. Partition [0, 1] into M intervals $I _ { 1 } , \ldots , I _ { M }$ of equal length. Draw a uniform random permutation Π of [M] and define the deterministic map

$$
Z _ { \Pi } ( q ) = z _ { \Pi ( r ) } , \qquad q \in I _ { r } .
$$

Let $\overline { { \mathbb { P } } } _ { G _ { M } , M } ^ { Q , n }$ be the law of the full sample in (51) after averaging over Π. Then

$$
\mathrm { T V } \Bigl ( \widetilde { \mathbb P } _ { G _ { M } , M } ^ { Q , n } , \lambda ^ { \otimes Q } \otimes ( K _ { n } G _ { M } ) ^ { \otimes Q } \Bigr ) \le 3 \alpha _ { Q , M } , \qquad \alpha _ { Q , M } : = 1 - \frac { ( M ) _ { Q } } { M ^ { Q } } \le \frac { Q ( Q - 1 ) } { 2 M } ,\tag{125}
$$

where $( M ) _ { Q } = M ( M - 1 ) \cdots ( M - Q + 1 )$ , with $( M ) _ { Q } = 0$ when $Q > M$

Proof of Lemma D.2. If $Q > M$ , the right side of (125) is at least one and the claim is trivial. Assume henceforth that $Q \leq M$

We compare two joint laws on the common augmented variables $( q , R , J , S ) = ( ( q _ { j } , R _ { j } , J _ { j } , S _ { j } ) ) _ { j = 1 } ^ { Q }$ Under the permutation law, first draw $R _ { 1 } , \ldots , R _ { Q }$ iid uniformly from [M] and, conditional on $R ,$ draw the $q _ { j }$ independently and uniformly within $I _ { R _ { j } }$ . Independently draw a uniform permutation Π of [M], set $J _ { j } = \Pi ( R _ { j } )$ , and then draw the counts conditionally independently according to

$$
S _ { j } \mid J _ { j } \sim \mathrm { B i n } \biggl ( n , \frac { 1 + z _ { J _ { j } } } { 2 } \biggr ) .
$$

This construction generates $q _ { 1 } , \ldots , q _ { Q } \stackrel { \mathrm { i i d } } { \sim } \lambda$ , and its (q, S)-marginal is $\overline { { \mathbb { P } } } _ { G _ { M } , M } ^ { Q , n } .$

Under the iid law, generate $( R , q )$ in the same way, but draw $J _ { 1 } , \ldots , J _ { Q }$ iid uniformly from $[ M ]$ , independently of $( R , q )$ , and then use the same conditional distribution of S given J. Its $( q , S )$ -marginal is $\lambda ^ { \otimes Q } \otimes ( K _ { n } G _ { M } ) ^ { \otimes Q }$ . Since projection onto $( q , S )$ cannot increase total variation, it is enough to compare these two augmented laws.

Let

$$
\mathcal { D } _ { Q , M } : = \{ ( r _ { 1 } , \ldots , r _ { Q } ) \in [ M ] ^ { Q } : r _ { 1 } , \ldots , r _ { Q } \mathrm { ~ a r e ~ d i s t i n c t } \} .
$$

The uniform law on $\mathcal { D } _ { Q , M }$ is exactly the law obtained by drawing Q labels uniformly without replacement from [M], with the jth coordinate recording the label drawn for observation $j$

Let $A : = \{ R \in \mathcal { D } _ { Q , M } \}$ under the permutation law. Conditional on $A ,$ , the vector R is uniform on $\mathcal { D } _ { Q , M }$ . For every fixed $r \in \mathcal { D } _ { Q , M }$ , the vector $\Pi ( r ) : = ( \Pi ( r _ { 1 } ) , \ldots , \Pi ( r _ { Q } ) )$ is also uniform on $\mathcal { D } _ { Q , M } .$ and its distribution does not depend on r. Consequently, under the permutation law conditional on A, the vectors R and J are independent and each is uniform on $\mathcal { D } _ { Q , M }$

Under the iid law, let $E : = \{ R \in \mathcal { D } _ { Q , M } , \ J \in \mathcal { D } _ { Q , M } \}$ . Since R and J are independent and each has iid uniform coordinates before conditioning, conditional on E they remain independent and each is uniform on $\mathcal { D } _ { Q , M }$ . Given $( R , J )$ , both laws draw $q _ { j }$ uniformly within $I _ { R _ { j } }$ and generate $S _ { j }$ through the same binomial channel indexed by $J _ { j }$ . Hence their conditional laws agree exactly:

$$
{ \mathcal { L } } _ { \mathrm { p e r m } } ( q , R , J , S \mid A ) = { \mathcal { L } } _ { \mathrm { i i d } } ( q , R , J , S \mid E ) .
$$

The probability of a collision in one iid sample of Q indices is

$$
\alpha _ { Q , M } = 1 - \frac { ( M ) _ { Q } } { M ^ { Q } } \leq \binom { Q } { 2 } \frac { 1 } { M } .
$$

Using the elementary inequality ${ \mathrm { T V } } ( P , P ( \cdot \mid B ) ) \leq P ( B ^ { c } )$ and the triangle inequality,

$$
\begin{array} { r l } & { \mathrm { T V } ( \mathcal { L } _ { \mathrm { p e r m } } ( q , R , J , S ) , \mathcal { L } _ { \mathrm { i i d } } ( q , R , J , S ) ) \leq \mathbb { P } _ { \mathrm { p e r m } } ( A ^ { c } ) + \mathbb { P } _ { \mathrm { i i d } } ( E ^ { c } ) } \\ & { \quad \quad \quad \quad = \alpha _ { Q , M } + \{ 1 - ( 1 - \alpha _ { Q , M } ) ^ { 2 } \} } \\ & { \quad \quad \quad \quad \leq 3 \alpha _ { Q , M } . } \end{array}
$$

Projecting both augmented laws onto $( q , S )$ now proves (125).

## D.3 Transfer theorem and general-ensemble embedding

Proof of Theorem $4 { \cdot } 4 .$ The upper inequality is (53). We prove the lower inequality.

We first record the Bayes reduction that guides the construction. Suppose $\nu _ { + }$ and $\nu _ { - }$ are finitely supported priors on $3 _ { \beta , L }$ such that, for some $b > 0 , \theta _ { Z } = \pm b$ for every $Z \in \mathrm { s u p p } ( \nu _ { \pm } )$ , respectively.

Let $\overline { { \mathbb { P } } } _ { + }$ and $\overline { { \mathbb { P } } } _ { - }$ denote the corresponding prior-averaged laws of the full observed sample. Then every decision rule $\widehat { \sigma }$ satisfies

$$
\begin{array} { r l } & { \underset { Z \in \mathfrak { I } _ { \beta , L } } { \operatorname* { s u p } } \mathbb { E } _ { Z } \mathcal { L } _ { Z } ( \widehat { \sigma } ) \geq \frac { b } { 2 } \left\{ \overline { { \mathbb { P } } } _ { + } ( \widehat { \sigma } = - 1 ) + \overline { { \mathbb { P } } } _ { - } ( \widehat { \sigma } = + 1 ) \right\} } \\ & { \qquad \geq \frac { b } { 2 } \left\{ 1 - \mathrm { T V } ( \overline { { \mathbb { P } } } _ { + } , \overline { { \mathbb { P } } } _ { - } ) \right\} . } \end{array}\tag{126}
$$

The first inequality bounds the maximum risk below by its average under the prior that first chooses the sign uniformly and then draws $Z$ from the corresponding $\nu _ { \pm } ;$ the second is the usual two-point testing inequality. It therefore sufices to construct two such priors with $b$ comparable to the modulus and whose prior-averaged data laws have total variation bounded away from one.

Fix $\varepsilon _ { Q } : = ( 6 4 \sqrt { Q } ) ^ { - 1 }$ and take any $0 < \Delta < \omega _ { n , \beta , L } ( \varepsilon _ { Q } )$ . By the reflection symmetrization used in the proof of Theorem 4.1, there exist $G _ { + } , G _ { - } \in { \mathcal { G } } _ { \beta , L }$ such that

$$
G _ { - } = G _ { + } ^ { \dagger } , \qquad \theta ( G _ { + } ) = a , \qquad \theta ( G _ { - } ) = - a , \qquad a > \Delta / 2 , \qquad h ( K _ { n } G _ { + } , K _ { n } G _ { - } ) \leq \varepsilon _ { Q } ,\tag{127}
$$

where $\dagger$ denotes reflection under $z \mapsto - z$

Apply Lemma D.1 to $G _ { + }$ and reflect the result to obtain equal-weight measures $G _ { + } ^ { ( M ) }$ and $G _ { - } ^ { ( M ) }$ Write

$$
P _ { \pm } ^ { ( M ) } : = K _ { n } G _ { \pm } ^ { ( M ) } , \qquad a _ { M } : = \theta ( G _ { + } ^ { ( M ) } ) .
$$

Because the targets and channel laws converge, and because M may be taken arbitrarily large, choose M so that

$$
a _ { M } \geq \Delta / 4 , \qquad h ( P _ { + } ^ { ( M ) } , K _ { n } G _ { + } ) \leq \varepsilon _ { Q } , \qquad 6 \alpha _ { Q , M } \leq \frac { 1 } { 8 } .\tag{128}
$$

Reflection gives the analogous channel bound for the negative hypothesis and $\theta ( G _ { - } ^ { ( M ) } ) = - \theta ( G _ { + } ^ { ( M ) } )$ The triangle inequality gives

$$
h ( P _ { + } ^ { ( M ) } , P _ { - } ^ { ( M ) } ) \leq 3 \varepsilon _ { Q } .
$$

Consequently, the product-Hellinger and total-variation inequalities (114) and (115) give

$$
\mathrm { T V } \Bigl ( ( P _ { + } ^ { ( M ) } ) ^ { \otimes Q } , ( P _ { - } ^ { ( M ) } ) ^ { \otimes Q } \Bigr ) \leq \frac { 3 } { 6 4 } .\tag{129}
$$

Each discretized alternative is now lifted to a family of deterministic question-to-advantage maps. For each sign, let $\nu _ { \pm }$ be the uniform permutation prior from Lemma D.2, constructed using $G _ { \pm } ^ { ( \hat { M } ) }$ Every map in the support of $\nu _ { \pm }$ belongs to $3 _ { \beta , I }$ and has target $\pm a _ { M }$ . Denote the corresponding prior-averaged full-data laws by $\overline { { \mathbb { P } } } _ { + }$ and $\overline { { \mathbb { P } } } _ { - } .$ . The permutation lemma gives

$$
\mathrm { T V } \Bigl ( \bar { \mathbb { P } } _ { \pm } , \lambda ^ { \otimes Q } \otimes ( P _ { \pm } ^ { ( M ) } ) ^ { \otimes Q } \Bigr ) \leq 3 \alpha _ { Q , M } .
$$

Since adjoining the common factor $\lambda ^ { \otimes Q }$ does not change total variation, the triangle inequality and (129) yield

$$
\mathrm { T V } ( \overline { { \mathbb { P } } } _ { + } , \overline { { \mathbb { P } } } _ { - } ) \leq 6 \alpha _ { Q , M } + \frac { 3 } { 6 4 } < \frac { 1 } { 2 } .
$$

Applying (126) with $b = a _ { M }$ and then taking the infimum over $\widehat { \sigma }$ gives

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { o b s } } ( \beta , L ) \geq \frac { a _ { M } } { 4 } \geq \frac { \Delta } { 1 6 } .
$$

Since this holds for every $0 < \Delta < \omega _ { n , \beta , L } ( \varepsilon _ { Q } )$ , this proves (54) with the stated universal constants.

Proof of Corollary 4.6. In this submodel the clean margin of w is

$$
M _ { q } ( w ) = ( w _ { 1 } - w _ { 2 } ) Z ( q ) .
$$

Put $p _ { + } : = \mathbb { P } \{ Z > 0 \}$ and $p _ { - } : = \mathbb { P } \{ Z < 0 \}$ . The margin condition implies that Z has no mass at zero, so $p _ { + } + p _ { - } = 1$ and $\theta _ { Z } = p _ { + } - p _ { - }$ . Accordingly, the risk of w is $p _ { - }$ <sub>−</sub> when $w _ { 1 } > w _ { 2 } , p _ { + }$ when $w _ { 1 } < w _ { 2 }$ , and one when $w _ { 1 } = w _ { 2 }$ . Hence $\begin{array} { r } { \operatorname* { i n f } _ { w \in \Delta _ { K } } R _ { Z } ( w ) = ( 1 - \vert \theta _ { Z } \vert ) / 2 } \end{array}$ , attained at e when $\theta _ { Z } > 0$ and at $e _ { 2 }$ when $\theta _ { Z } < 0$ . At either choice,

$$
\mathbb { P } \{ 0 < M _ { q } ( w ^ { \star } ) \leq t \} \leq ( Z _ { \# } \lambda ) ( [ - t , t ] ) \leq L t ^ { \beta } .
$$

Thus the embedded environments satisfy the one-sided margin condition at $w ^ { \star }$ used in the general theorem.

Given any estimator ${ \widehat { w } } ,$ define the induced scalar decision $\widehat { \sigma }$ to be +1 when $\widehat { w } _ { 1 } \geq \widehat { w } _ { 2 }$ and −1 otherwise. If $\widehat { w } _ { 1 } \neq \widehat { w } _ { 2 }$ , its clean excess risk equals $| \theta _ { Z } |$ when $\widehat { \sigma }$ chooses the wrong side and is zero otherwise. If $\widehat { w } _ { 1 } = \widehat { w } _ { 2 }$ , the ensemble margin is zero on every question, so $R _ { Z } ( \widehat { w } ) = 1$ and its excess risk is $( 1 + | \theta _ { Z } | ) / 2 \geq | \theta _ { Z } |$ . Therefore, pointwise in the data,

$$
R _ { Z } ( \widehat { w } ) - \operatorname* { i n f } _ { w } R _ { Z } ( w ) \geq | \theta _ { Z } | \mathbf { 1 } \{ \widehat { \sigma } \theta _ { Z } < 0 \} .
$$

For models 1 and $2 , S _ { j }$ is suficient for their raw generations; the generations of models $i \geq 3$ have a law independent of $Z .$ . Thus the induced $\widehat { \sigma }$ is statistically equivalent to a possibly randomized rule based on $( ( q _ { j } , S _ { j } ) ) _ { j = 1 } ^ { Q }$ , and such randomization is already allowed in (52). Taking expectations, suprema, and infima proves the first inequality; the second follows from Corollary 4.5. □

## E Proof of the identification floor

Recall that $\Pi _ { n }$ denotes the space of real algebraic polynomials of degree at most n.

## E.1 Localized polynomial approximation and the upper bound

Lemma E.1 (Localized Jackson polynomial). For every $s > 0$ , there is $C _ { s }$ such that, for every integer $n \geq 1$ , there is $P _ { n } \in \Pi _ { n }$ satisfying

$$
\| P _ { n } \| _ { L _ { \infty } [ - 1 , 1 ] } \leq 1 , \qquad | P _ { n } ( x ) - \mathrm { s g n } ( x ) | \leq C _ { s } \operatorname* { m i n } \{ 1 , ( n | x | ) ^ { - s } \}\tag{130}
$$

for every $x \in [ - 1 , 1 ] \setminus \{ 0 \}$

Proof. The key idea is to smooth the discontinuous sign function with a positive, normalized, finite-frequency Jackson kernel. The resulting convolution is a bounded trigonometric polynomial that closely follows the sign function away from its jumps; evenness then converts it into an algebraic polynomial in $x = \cos \vartheta$

Let $f ( \vartheta ) : = \mathrm { s g n } ( \cos \vartheta )$ on the circle $\mathbb { R } / ( 2 \pi \mathbb { Z } )$ , and choose an integer r such that $2 r - 1 > s$ . For $m \geq 2$ , define

$$
J _ { m , r } ( u ) : = c _ { m , r } \left( \frac { \sin ( m u / 2 ) } { \sin ( u / 2 ) } \right) ^ { 2 r } ,
$$

where the quotient is interpreted continuously at zero. The constant $c _ { m , r }$ is chosen to give normalization $\begin{array} { r } { { \frac { 1 } { 2 \pi } } \int _ { - \pi } ^ { \pi } J _ { m , r } ( u ) } \end{array}$ du = 1.

Kernel properties. Recall that the Fej´er kernel satisfies

$$
F _ { m } ( u ) : = \frac { 1 } { m } \left( \frac { \sin ( m u / 2 ) } { \sin ( u / 2 ) } \right) ^ { 2 } = \sum _ { k = - ( m - 1 ) } ^ { m - 1 } \left( 1 - \frac { | k | } { m } \right) e ^ { i k u } .
$$

Since $J _ { m , r } = c _ { m , r } m ^ { r } F _ { m } ^ { r }$ , it is nonnegative, even, and a trigonometric polynomial of degree $d : =$ $r ( m - 1 )$

To determine its normalization scale, put $K _ { m , r } ^ { 0 } ( u ) : = ( \sin ( m u / 2 ) / \sin ( u / 2 ) ) ^ { 2 r } . \mathrm { ~ O n ~ } | u | \leq 1 / m$ , one has $K _ { m . r } ^ { 0 } ( u ) \asymp _ { r } m ^ { 2 r }$ , while for $0 < | u | \le \pi$ , the inequalities $| \sin ( u / 2 ) | \geq | u | / \pi$ and $| \sin ( m u / 2 ) | \leq 1$ give $K _ { m , r } ^ { 0 ^ { \prime } } ( u ) \lesssim _ { r } | u | ^ { - 2 r }$ . Splitting the integral at $1 / m$ therefore yields

$$
\frac { 1 } { 2 \pi } \int _ { - \pi } ^ { \pi } K _ { m , r } ^ { 0 } ( u ) \mathrm { d } u \asymp _ { r } m ^ { 2 r - 1 } , \qquad c _ { m , r } \asymp _ { r } m ^ { - ( 2 r - 1 ) } .
$$

Only the resulting upper bound on $c _ { m , r }$ is needed below. We next bound the tail. If $a \le 1 / m$ normalized total mass bounds it by one; if $a > 1 / m$ , the preceding pointwise and normalization bounds give

$$
\frac { 1 } { 2 \pi } \int _ { | u | \geq a } J _ { m , r } ( u ) \mathrm { d } u \lesssim _ { r } m ^ { - ( 2 r - 1 ) } \int _ { a } ^ { \pi } u ^ { - 2 r } \mathrm { d } u \lesssim _ { r } ( m a ) ^ { - ( 2 r - 1 ) } .
$$

Thus, for every $0 < a \le \pi$

$$
\frac { 1 } { 2 \pi } \int _ { | u | \geq a } J _ { m , r } ( u ) \mathrm { ~ d } u \leq C _ { r } \operatorname* { m i n } \{ 1 , ( m a ) ^ { - ( 2 r - 1 ) } \} , \qquad 0 < a \leq \pi .\tag{131}
$$

Finite-frequency smoothing. Define

$$
T _ { m } ( \vartheta ) : = \frac { 1 } { 2 \pi } \int _ { - \pi } ^ { \pi } f ( \vartheta - u ) J _ { m , r } ( u ) \ : \mathrm { d } u .
$$

Since $J _ { m , r } \geq 0$ and has normalized mass one, $T _ { m } ( \vartheta )$ is a weighted average of values in $[ - 1 , 1 ]$ , so $| T _ { m } ( \vartheta ) | \leq 1$ . Write $\begin{array} { r } { J _ { m , r } ( u ) = \sum _ { | k | \leq d } j _ { k } e ^ { i k u } } \end{array}$ . The convolution identity gives

$$
T _ { m } ( \vartheta ) = \sum _ { | k | \leq d } j _ { k } { \widehat { f } } ( k ) e ^ { i k \vartheta } ,
$$

where ${ \widehat { f } } ( k )$ is the kth Fourier coeficient of $f .$ Thus $T _ { m }$ is a trigonometric polynomial of degree at most d. Because both $f$ and $J _ { m , r }$ are even, so is $T _ { m }$ , and hence it has a cosine expansion $\begin{array} { r } { T _ { m } ( \vartheta ) = a _ { 0 } + \sum _ { k = 1 } ^ { d } a _ { k } \cos ( k \vartheta ) } \end{array}$ . If ${ \sf T } _ { k }$ denotes the kth Chebyshev polynomial, define $P ( x ) : =$ $\begin{array} { r } { a _ { 0 } + \sum _ { k = 1 } ^ { d } a _ { k } \mathsf { T } _ { k } ( x ) } \end{array}$ . Then $P$ has degree at most d and

$$
T _ { m } ( \vartheta ) = P ( \cos \vartheta ) , \qquad \| P \| _ { L _ { \infty } [ - 1 , 1 ] } \leq 1 .
$$

Localized approximation error. Let $a ( \vartheta )$ denote the circular distance from $\vartheta$ to the nearest discontinuity of $f ,$ namely $\pi / 2$ or $3 \pi / 2$ . Whenever $| u | < a ( \vartheta )$ , the values $f ( \vartheta - u )$ and $f ( \vartheta )$ agree. Therefore (131) implies

$$
\begin{array} { l } { | T _ { m } ( \vartheta ) - f ( \vartheta ) | \leq \displaystyle \frac { 2 } { 2 \pi } \int _ { | u | \geq a ( \vartheta ) } J _ { m , r } ( u ) \mathrm { d } u } \\ { \leq C _ { r } \operatorname* { m i n } \{ 1 , ( m a ( \vartheta ) ) ^ { - ( 2 r - 1 ) } \} . } \end{array}
$$

If $x = \cos \vartheta$ , then $| x | = \sin a ( \vartheta ) \leq a ( \vartheta )$ . For $n \geq 2 r$ , take $m = 1 + \lfloor n / r \rfloor$ . Then $d = r ( m - 1 ) \leq n$ and $m \geq n / r$ . Since $2 r - 1 > s ,$ the last display implies (130), after enlarging the constant. For the finitely many integers $1 \leq n < 2 r$ , take $P _ { n } \equiv 0$ and enlarge $C _ { s }$ □

The estimate in Lemma E.1 is intentionally nonuniform: it gives only a constant bound in the boundary layer $| x | \lesssim n ^ { - 1 }$ , but outside that layer the error decays at any prescribed fixed polynomial order in n|x|.

Lemma E.2 (Integrated approximation under a margin condition). If $G \in { \mathcal { G } } _ { \beta , i }$ <sub>L</sub> and $s > \beta$ in Theorem E.1, then, for every $n \geq 1$

$$
\int | P _ { n } ( z ) - \mathrm { s g n } ( z ) | ~ \mathrm { d } G ( z ) \leq C _ { \beta , L , s } n ^ { - \beta } .\tag{132}
$$

Proof. For $| z | \leq n ^ { - 1 }$ , the integrand is bounded by two and the margin condition gives probability at most $L n ^ { - \beta }$ . On the dyadic shell

$$
\frac { 2 ^ { j } } { n } < | z | \leq \operatorname* { m i n } \left\{ \frac { 2 ^ { j + 1 } } { n } , 1 \right\} ,
$$

the pointwise error is at most $C _ { s } 2 ^ { - j s }$ and the shell probability is at most $L ( 2 ^ { j + 1 } / n ) ^ { \beta }$ until the upper endpoint reaches one. Consequently,

$$
\int | P _ { n } - \mathrm { s g n } | ~ \mathrm { d } G \leq L n ^ { - \beta } \bigg [ 2 + C _ { s } 2 ^ { \beta } \sum _ { j \geq 0 } 2 ^ { j ( \beta - s ) } \bigg ] \leq C _ { \beta , L , s } n ^ { - \beta } ,
$$

because $s > \beta ,$ , proving (132).

Proof of the upper bound in Theorem 4.2. Suppose $K _ { n } G \ = \ K _ { n } H$ . By Theorem 3.2, G and H integrate every polynomial of degree at most n equally. Fix $s = \beta + 1$ and let $P _ { n }$ be the corresponding polynomial from Theorem E.1. Then

$$
\begin{array} { l } { \displaystyle | \theta ( G ) - \theta ( H ) | = \left| \int ( \mathrm { s g n } - P _ { n } ) \mathrm { d } G - \int ( \mathrm { s g n } - P _ { n } ) \mathrm { d } H \right| } \\ { \displaystyle \qquad \leq \int | \mathrm { s g n } - P _ { n } | \mathrm { d } G + \int | \mathrm { s g n } - P _ { n } | \mathrm { d } H \leq C _ { \beta , L } n ^ { - \beta } } \end{array}
$$

by Theorem E.2. Taking the supremum over all pairs with $K _ { n } G = K _ { n } H$ proves the upper bound on $\omega _ { n , \beta , L } ( 0 )$ □

## E.2 Weighted $L _ { 1 }$ approximation of the jump

Let

$$
\mu _ { \beta } ( \mathrm { d } x ) = \frac { \beta } { 2 } | x | ^ { \beta - 1 } \mathbf { 1 } _ { [ - 1 , 1 ] } ( x ) \ \mathrm { d } x .
$$

The next lemma gives the matching approximation rate. Its upper bound follows from the localized Jackson construction above; the lower bound uses a weighted inverse theorem of Kopotun (2015).

Lemma E.3 (Weighted $L _ { 1 }$ approximation of a jump). There are $0 < c _ { \beta } < C _ { \beta } < \infty$ such that

$$
c _ { \beta } n ^ { - \beta } \leq E _ { n } ( \beta ) : = \operatorname* { i n f } _ { P \in \Pi _ { n } } \int | \operatorname { s g n } ( x ) - P ( x ) | \mathrm { d } \mu _ { \beta } ( x ) \leq C _ { \beta } n ^ { - \beta } .\tag{133}
$$

Proof. The upper bound follows from Theorems E.1 and E.2, because $\mu _ { \beta } ( [ - t , t ] ) = t ^ { \beta }$

For the lower bound, we use only the inverse theorem of Kopotun (2015, Theorem 9.1). The power weight $w _ { \beta } ( x ) : = | x | ^ { \beta - 1 }$ is the generalized-Jacobi weight with one designated point $z _ { 1 } = 0$ and

exponent $\beta - 1 > - 1$ , and belongs to the class treated there. Fix an integer $r > \beta _ { \mathrm { {  } } }$ , which will be the order of the modulus, and a suficiently large constant A. Specializing Kopotun’s theorem to $p = 1$ , and absorbing the normalization of $\mu _ { \beta }$ into the constant, gives

$$
\omega _ { \varphi } ^ { r } ( f , A , m ^ { - 1 } ) _ { 1 , w _ { \beta } } \leq C m ^ { - r } \sum _ { k = 1 } ^ { m } k ^ { r - 1 } E _ { k } ( f ) _ { 1 , w _ { \beta } } ,\tag{134}
$$

Here $\begin{array} { r } { \| g \| _ { 1 , w _ { \beta } } : = \int _ { - 1 } ^ { 1 } | g ( x ) | w _ { \beta } ( x ) } \end{array}$ dx, $\begin{array} { r } { E _ { k } ( f ) _ { 1 , w _ { \beta } } : = \operatorname* { i n f } _ { P \in \Pi _ { k } } \| f - P \| _ { 1 , w _ { \beta } } . } \end{array}$ , and $\omega _ { \varphi } ^ { r } ( f , A , m ^ { - 1 } ) _ { 1 , w _ { \beta } }$ is the complete weighted modulus of smoothness defined in Kopotun (2015, Section 3). The normalization of $\mu _ { \beta }$ gives $E _ { k } ( \mathrm { s g n } ) _ { 1 , w _ { \beta } } = ( 2 / \beta ) E _ { k } ( \beta )$ . Kopotun uses $\Pi _ { k }$ for polynomials of degree less than $k ,$ whereas our $\Pi _ { k }$ contains polynomials of degree at most $k ;$ this one-degree shift only afects the constants.

For $f = \mathrm { s g n }$ , the definition of the complete modulus simplifies substantially. Its main-part finite-diference term is evaluated away from a neighborhood of zero and vanishes because sgn is constant on each remaining component. Only the local best-approximation term remains:

$$
\omega _ { \varphi } ^ { r } ( \mathrm { s g n } , A , t ) _ { 1 , w _ { \beta } } = \operatorname* { i n f } _ { P \in \Pi _ { r - 1 } } \int _ { - b _ { t } } ^ { b _ { t } } | \mathrm { s g n } ( x ) - P ( x ) | | x | ^ { \beta - 1 } \mathrm { d } x , \qquad b _ { t } : = 2 A \rho ( t , 0 ) = 2 A ( t + t ^ { 2 } ) ,
$$

for all suficiently small t. The zero polynomial gives an upper bound of order $t ^ { \beta } .$ . For the reverse bound, the interval $[ - b _ { t } , b _ { t } ]$ contains $[ - 2 A t , 2 A t ]$ , and the substitution $x = t u$ gives

$$
\operatorname* { i n f } _ { P \in \Pi _ { r - 1 } } \int _ { - 2 A t } ^ { 2 A t } \left| \operatorname { s g n } ( x ) - P ( x ) \right| | x | ^ { \beta - 1 } \mathrm { ~ d } x = t ^ { \beta } \operatorname* { i n f } _ { Q \in \Pi _ { r - 1 } } \int _ { - 2 A } ^ { 2 A } \left| \operatorname { s g n } ( u ) - Q ( u ) \right| | u | ^ { \beta - 1 } \mathrm { ~ d } u .
$$

The last infimum is strictly positive: $\Pi _ { r - 1 }$ is a finite-dimensional closed subspace of the weighted $L _ { 1 }$ space, whereas sgn is not equal almost everywhere to a polynomial. We have therefore shown that

$$
\omega _ { \varphi } ^ { r } ( \mathrm { s g n } , A , t ) _ { 1 , w _ { \beta } } \asymp t ^ { \beta } , \qquad 0 < t \leq t _ { \beta } .\tag{135}
$$

Together, (134) and (135) give a lower bound of order $m ^ { - \beta }$ on a weighted average of $E _ { 1 } ( \beta ) , \dots , E _ { m } ( \beta )$ for all suficiently large m. It remains to extract a pointwise lower bound for $E _ { n } ( \beta )$

Write $E _ { k } : = E _ { k } ( \beta )$ . By the upper bound already established, $E _ { k } \lesssim k ^ { - \beta }$ . Apply (134) with $m = M n$ , where $M \geq 2$ is a fixed integer to be chosen. By (135),

$$
c ( M n ) ^ { - \beta } \leq C ( M n ) ^ { - r } \sum _ { k = 1 } ^ { M n } k ^ { r - 1 } E _ { k } .
$$

Split the sum at $n .$ Using $E _ { k } \lesssim k ^ { - \beta }$ on the first part and monotonicity $E _ { k } \leq E _ { n }$ for $k \geq n$ on the second gives

$$
\sum _ { k = 1 } ^ { n } k ^ { r - 1 } E _ { k } \lesssim n ^ { r - \beta } , \qquad \sum _ { k = n + 1 } ^ { M n } k ^ { r - 1 } E _ { k } \lesssim ( M n ) ^ { r } E _ { n } .
$$

Consequently,

$$
c M ^ { - \beta } n ^ { - \beta } \leq C _ { 3 } M ^ { - r } n ^ { - \beta } + C _ { 4 } E _ { n } .
$$

Since $r > \beta$ , choose the fixed M large enough that $C _ { 3 } M ^ { - r } \leq ( c / 2 ) M ^ { - \beta }$ . The first term on the right can then be absorbed into the left, giving $E _ { n } \gtrsim n ^ { - \beta }$ . Decreasing the implicit constant handles the finitely many small values of n not covered by (135). □

## E.3 Duality and moment-matched alternatives

Lemma E.4 $( L _ { 1 }$ duality produces a moment-canceling family). For every integer $m \geq 1$ , there is an odd function $h _ { m } \in L _ { \infty } ( \mu _ { \beta } )$ satisfying

$$
\| h _ { m } \| _ { \infty } \leq 1 , \qquad \int P ( z ) h _ { m } ( z ) \mathrm { d } \mu _ { \beta } ( z ) = 0 \quad f o r e v e r y P \in \Pi _ { m } ,\tag{136}
$$

and

$$
\int \mathrm { s g n } ( z ) h _ { m } ( z ) \mathrm { d } \mu _ { \beta } ( z ) = E _ { m } ( \beta ) .\tag{137}
$$

Define the raw alternatives $G _ { m , + } , G _ { m , - } \ b _ { i }$ y

$$
\mathrm { d } G _ { m , + } : = ( 1 + h _ { m } ) \mathrm { d } \mu _ { \beta } , \qquad \mathrm { d } G _ { m , - } : = ( 1 - h _ { m } ) \mathrm { d } \mu _ { \beta } .\tag{138}
$$

They are probability measures on $[ - 1 , 1 ]$ such that:

(a) their moments agree through degree m;

(b) $G _ { m , \pm } ( [ - t , t ] ) \leq 2 t ^ { \beta }$ for every $0 < t \leq 1$ ;

(c) $G _ { m , - } = G _ { m , + } ^ { \dag }$ and $\theta ( G _ { m , + } ) = - \theta ( G _ { m , - } ) = E _ { m } ( \beta )$

Moreover, for every $L \geq 1$ , let

$$
\rho _ { L } : = \operatorname* { m i n } \{ 1 , L / 2 \} , \qquad H _ { 0 } : = \frac 1 2 ( \delta _ { - 1 } + \delta _ { + 1 } ) , \qquad H _ { m , \pm } : = \rho _ { L } G _ { m , \pm } + ( 1 - \rho _ { L } ) H _ { 0 } .\tag{139}
$$

Then $H _ { m , \pm } \in \mathcal { G } _ { \beta , L }$ , their moments agree through degree m, $H _ { m , - } = H _ { m , + } ^ { \dagger }$ , and

$$
\theta ( H _ { m , + } ) = - \theta ( H _ { m , - } ) = \rho _ { L } E _ { m } ( \beta ) .\tag{140}
$$

Proof. The polynomial-approximation error admits a dual certificate $h _ { m }$ that annihilates every polynomial in $\Pi _ { m }$ while being maximally correlated with sgn. Perturbing the reference measure in the directions $\pm h _ { m }$ will then produce moment-matched distributions with oppositely separated targets.

Let $f : = \mathrm { s g n }$ and use the dual pairing

$$
\langle g , h \rangle : = \int g ( x ) h ( x ) \mathrm { d } \mu _ { \beta } ( x ) , \qquad g \in L _ { 1 } ( \mu _ { \beta } ) , \quad h \in L _ { \infty } ( \mu _ { \beta } ) .
$$

Since $E _ { m } ( \beta )$ is the $L _ { 1 }$ distance from $f$ to the subspace $\Pi _ { m }$ , it has the following dual representation;   
see Theorem E.5.

$$
\begin{array} { r } { E _ { m } ( \beta ) = \underset { h \in L _ { \infty } ( \mu _ { \beta } ) , \ \| h \| _ { \infty } \leq 1 } { \operatorname* { m a x } } \langle f , h \rangle . } \end{array}\tag{141}
$$

The maximum is attained by weak-<sup>∗</sup> compactness. We may choose a maximizer that is odd. Indeed, if $h$ is feasible, then

$$
h _ { \mathrm { o d d } } ( x ) : = \frac { h ( x ) - h ( - x ) } { 2 }
$$

is feasible, has $L _ { \infty }$ norm at most one, and has the same objective value because $\mu _ { \beta }$ is symmetric, $\Pi _ { m }$ is reflection invariant, and sgn is odd.

Let $h _ { m }$ be such an odd maximizer and define $G _ { m , \pm }$ by (138). These measures are nonnegative because $\| h _ { m } \| _ { \infty } \leq 1$ . Orthogonality to the constant polynomial makes their total masses one; orthogonality to $x ^ { j } , j = 0 , \dotsc , m$ , makes their moments equal. Moreover,

$$
G _ { m , \pm } ( [ - t , t ] ) \leq 2 \mu _ { \beta } ( [ - t , t ] ) = 2 t ^ { \beta } .
$$

Oddness gives $G _ { m , - } = G _ { m , + } ^ { \dag }$ . Finally, symmetry of $\mu _ { \beta }$ gives R sgn $\mathrm { d } \mu _ { \beta } = 0$ , so (141) gives

$$
\theta ( G _ { m , + } ) = \int \operatorname { s g n } ( x ) h _ { m } ( x ) \ \mathrm { d } \mu _ { \beta } ( x ) = E _ { m } ( \beta ) ,
$$

and reflection gives the target of $G _ { m , - } .$

It remains to verify the claims for the adjusted pair. For every $0 < t < 1$ , the outer atoms do not contribute and

$$
H _ { m , \pm } ( [ - t , t ] ) \leq 2 \rho _ { L } t ^ { \beta } \leq L t ^ { \beta } ;
$$

at $t = 1$ , the required inequality is $1 \leq L$ . Thus $H _ { m , \pm } \in \mathcal { G } _ { \beta , L }$ . Adding the same symmetric measure to the two raw alternatives preserves their moment matching, and symmetry of $H _ { 0 }$ together with reflection of the raw pair gives $H _ { m , - } = H _ { m , + } ^ { \dagger }$ . Finally, $\theta ( H _ { 0 } ) = 0$ , so

$$
\theta ( H _ { m , + } ) = - \theta ( H _ { m , - } ) = \rho _ { L } E _ { m } ( \beta ) .
$$

Remark E.5 (Duality interpretation). Identity (141) is the infinite-dimensional analogue of linearprogramming duality. Indeed, put $\begin{array} { r } { E : = \operatorname* { i n f } _ { P \in \Pi _ { m } } \lVert f - P \rVert _ { L _ { 1 } ( \mu _ { \beta } ) } } \end{array}$ . For every feasible h and every $P \in \Pi _ { m }$

$$
\begin{array} { r } { \langle f , h \rangle = \langle f - P , h \rangle \leq \| f - P \| _ { L _ { 1 } ( \mu _ { \beta } ) } , } \end{array}
$$

so the dual value is at most E. Hahn–Banach supplies a norm-one functional on $L _ { 1 } ( \mu _ { \beta } )$ that annihilates $\Pi _ { m }$ and takes the value E at $f .$ . Identifying $L _ { 1 } ( \mu _ { \beta } ) ^ { * }$ with $L _ { \infty } ( \mu _ { \beta } )$ represents this functional by a feasible h attaining equality, and hence establishes strong duality.

Proof of the lower bound in Theorem 4.2. Let $H _ { n , + } , H _ { n }$ be the adjusted degree-n alternatives supplied by Theorem E.4. Their targets are opposite with

$$
\theta ( H _ { n , + } ) = - \theta ( H _ { n , - } ) = \rho _ { L } E _ { n } ( \beta ) \geq c _ { \beta , L } n ^ { - \beta } .
$$

By Theorem 3.2, $K _ { n } H _ { n , + } = K _ { n } H _ { n , - }$ so this pair proves the lower bound on $\omega _ { n , \beta , L } ( 0 )$

Remark E.6 (Direct minimax consequence). The same pair also gives a direct proof of (46) by a two-point argument. Let

$$
\mathbb { P } _ { \pm } : = ( K _ { n } H _ { n , \pm } ) ^ { \otimes Q }
$$

denote the laws of the full Q-question experiment. Moment equivalence gives $\mathbb { P } _ { + } = \mathbb { P } _ { - }$ . Hence, for every decision rule $\widehat { \sigma }$

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { G \in \mathcal { G } _ { \beta , L } } \mathbb { E } _ { G } \mathcal { L } _ { G } ( \widehat { \sigma } ) \geq \frac { \rho _ { L } E _ { n } ( \beta ) } { 2 } \left[ \mathbb { P } _ { + } ( \widehat { \sigma } = - 1 ) + \mathbb { P } _ { - } ( \widehat { \sigma } = + 1 ) \right] } & { } \\ { \displaystyle \quad } & { = \frac { \rho _ { L } E _ { n } ( \beta ) } { 2 } . } \end{array}
$$

The equality uses $\mathbb { P } _ { + } = \mathbb { P } _ { - }$ and $\widehat { \sigma } \in \{ - 1 , + 1 \}$ . The prefactor $\rho _ { L } E _ { n } ( \beta )$ is the cost of a wrong decision under either alternative, since $| \theta ( H _ { n , \pm } ) | = \rho _ { L } E _ { n } ( \beta )$ . Taking the infimum over $\widehat { \sigma }$ gives $\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \gtrsim _ { \beta , L } n ^ { - \beta }$ , independently of Q.

## F Coeficient bounds and the Durrmeyer calculation

## F.1 Proof of the explicit exponential coeficient bound

Proof of Theorem 5.5. Expand $\begin{array} { r } { P ( z ) = c _ { 0 } + \sum _ { i = 1 } ^ { d } c _ { j } T _ { j } ( z ) } \end{array}$ in Chebyshev polynomials. The cosine coeficient formula and $\| P \| _ { \infty } \leq 1 { \mathrm { ~ g i v e ~ } } | c _ { 0 } | \leq { \dot { 1 } }$ and $| c _ { j } | \le 2$ for $j \geq 1$ . Let $L _ { j }$ be the sum of the absolute values of the monomial coeficients of $T _ { j }$ . The recurrence $T _ { j + 1 } ( z ) = 2 z T _ { j } ( z ) - T _ { j - 1 } ( z )$ gives $L _ { j + 1 } \leq 2 L _ { j } + L _ { j - 1 } ;$ since $L _ { 0 } = L _ { 1 } = 1$ , it follows that $L _ { j } \leq C ( 1 + \sqrt { 2 } ) ^ { j }$ . After substituting $z = 2 p - 1$ , each monomial $z ^ { \ell }$ has absolute coeficient sum $3 ^ { \ell }$ , so the corresponding sum for $T _ { j } ( 2 p - 1 )$ is at most $C \{ 3 ( 1 + \sqrt { 2 } ) \} ^ { j }$

Write $\begin{array} { r } { P ( 2 p - 1 ) = \sum _ { j = 0 } ^ { d } r _ { j } p ^ { j } } \end{array}$ , so the $r _ { j }$ are its coeficients in the ordinary power basis. The preceding bounds and a geometric sum give $\begin{array} { r } { \sum _ { j = 0 } ^ { d } | r _ { j } | \leq C _ { 0 } B _ { 0 } ^ { d } } \end{array}$ , with $B _ { 0 } = 7 . 5 > 3 ( 1 + \sqrt { 2 } )$

For any $n \geq d ,$ , the identity $\begin{array} { r } { p ^ { j } = \sum _ { k = j } ^ { n } \frac { \binom { k } { j } } { \binom { n } { j } } b _ { k , n } ( p ) } \end{array}$ gives the degree-n Bernstein coeficients directly:

$$
a _ { k } ( P ) = \sum _ { j = 0 } ^ { \operatorname * { m i n } ( k , d ) } r _ { j } { \frac { \binom { k } { j } } { \binom { n } { j } } } , \qquad 0 \leq k \leq n .
$$

Every ratio in this sum lies in [0, 1], so $\begin{array} { r } { | \boldsymbol { a } _ { k } ( \boldsymbol { P } ) | \leq \sum _ { j = 0 } ^ { d } | \boldsymbol { r } _ { j } | \leq C _ { 0 } B _ { 0 } ^ { d } . } \end{array}$

## F.2 Proofs for the Durrmeyer spectrum and coeficient bound

Proof of Theorem 5.6. The operator $D _ { n } = T _ { n } T _ { n } ^ { * }$ is self-adjoint and positive. It also preserves each polynomial space $\Pi _ { j }$ . Indeed, beta integration gives

$$
( T _ { n } ^ { * } u ^ { j } ) ( k ) = { \frac { ( n + 1 ) ! } { ( n + j + 1 ) ! } } ( k + 1 ) ( k + 2 ) \cdot \cdot \cdot ( k + j ) ,
$$

which is a degree-j polynomial in $k .$ . Averaging this polynomial under $k \sim \mathrm { B i n } ( n , p )$ shows that $D _ { n } u ^ { j } \in \Pi _ { j }$ and that its leading coeficient is

$$
\lambda _ { j , n } = \frac { n ! ( n + 1 ) ! } { ( n - j ) ! ( n + j + 1 ) ! } .
$$

Here we use the fact that the expectation of a monic degree-j polynomial in a $\mathrm { B i n } ( n , p )$ variable has leading coeficient $n ! / ( n - j ) !$ as a polynomial in $p .$

For $j = 0$ , the claim follows directly from $D _ { n } 1 = 1$ . For $j \geq 1$ , because $L _ { 0 } , \ldots , L _ { j }$ are obtained by orthogonalizing $1 , u , \dotsc , u ^ { j }$

$$
\Pi _ { j } = \Pi _ { j - 1 } \oplus \mathrm { s p a n } \{ L _ { j } \} .
$$

Now $D _ { n } L _ { j } \in \Pi _ { j }$ by degree preservation. For every $q \in \Pi _ { j - 1 }$ , self-adjointness and degree preservation give

$$
\left. D _ { n } L _ { j } , q \right. _ { L _ { 2 } } = \left. L _ { j } , D _ { n } q \right. _ { L _ { 2 } } = 0 ,
$$

because $D _ { n } q \in \Pi _ { j - 1 }$ and $L _ { j } \perp \Pi _ { j - 1 }$ . Hence $D _ { n } L _ { j }$ lies in the one-dimensional space $\Pi _ { j } \cap \Pi _ { j - 1 } ^ { \perp } =$ span $\{ L _ { j } \}$ . Comparing leading coeficients proves (67).

Define $v _ { j } = \lambda _ { j , n } ^ { - 1 / 2 } T _ { n } ^ { * } L _ { j }$ . Then

$$
\left. v _ { i } , v _ { j } \right. _ { \nu _ { n } } = \frac { 1 } { \sqrt { \lambda _ { i , n } \lambda _ { j , n } } } \left. L _ { i } , D _ { n } L _ { j } \right. _ { L _ { 2 } } = \mathbf { 1 } \{ i = j \} ,
$$

and $T _ { n } v _ { j } = \sqrt { \lambda _ { j , n } } L _ { j }$ . There are $n + 1$ orthonormal vectors $v _ { j }$ in the $( n + 1 )$ -dimensional space $\mathsf { H } _ { n } ,$ so they form a basis. Thus $( v _ { j } , L _ { j } , \sqrt { \lambda _ { j , n } } )$ is a singular system and (68) follows. □

Proof of Theorem 5.7. If $P \in \Pi _ { d }$ , write $\begin{array} { r } { \widetilde { P } = \sum _ { j = 0 } ^ { d } c _ { j } L _ { j } } \end{array}$ . Since $T _ { n } a ( P ) = { \widetilde P } .$ comparison of the Legendre coeficients in (68) gives

$$
\left. a ( P ) , v _ { j } \right. _ { \nu _ { n } } = \left\{ { \begin{array} { l l } { c _ { j } / \sqrt { \lambda _ { j , n } } , } & { 0 \leq j \leq d , } \\ { 0 , } & { d < j \leq n . } \end{array} } \right.
$$

Consequently,

$$
a ( P ) = \sum _ { j = 0 } ^ { d } \frac { c _ { j } } { \sqrt { \lambda _ { j , n } } } v _ { j } .
$$

Parseval’s identity in $\mathsf { H } _ { n }$ now gives

$$
\frac { 1 } { n + 1 } \sum _ { k = 0 } ^ { n } a _ { k } ( P ) ^ { 2 } = \| a ( P ) \| _ { \nu _ { n } } ^ { 2 } = \sum _ { j = 0 } ^ { d } \frac { c _ { j } ^ { 2 } } { \lambda _ { j , n } } ,
$$

which proves (70). Since $j \mapsto \lambda _ { j , n }$ is decreasing, Parseval in $L _ { 2 } [ 0 , 1 ]$ also gives

$$
\| a ( P ) \| _ { \nu _ { n } } \leq \kappa _ { d , n } \left( \sum _ { j = 0 } ^ { d } c _ { j } ^ { 2 } \right) ^ { 1 / 2 } = \kappa _ { d , n } \| \widetilde { P } \| _ { L _ { 2 } [ 0 , 1 ] } .
$$

Finally,

$$
\operatorname* { m a x } _ { k } | a _ { k } ( P ) | \leq \sqrt { \sum _ { k = 0 } ^ { n } a _ { k } ( P ) ^ { 2 } } = \sqrt { n + 1 } \| a ( P ) \| _ { \nu _ { n } } .
$$

Applying the same bound to the maximum and minimum proves (71).

Asymptotics of the inverse singular values. For $d \leq n / 2$

$$
\log \kappa _ { d , n } = \frac { 1 } { 2 } \sum _ { r = 0 } ^ { d - 1 } \log \left( 1 + \frac { 2 r + 2 } { n - r } \right) .
$$

Using $x / ( 1 + x ) \leq \log ( 1 + x ) \leq x$ and $n - r \asymp n$ gives (72). At $d = n$

$$
\kappa _ { n , n } ^ { 2 } = \frac { ( 2 n + 1 ) ! } { n ! ( n + 1 ) ! } = { \binom { 2 n + 1 } { n } } ,
$$

and Stirling’s formula gives the endpoint asymptotic.

## G Proofs for the spectral lower bound and rate extraction

We prove Theorems 6.1 and 6.2. The lower-bound argument combines the degree-d dual alternatives from Section E with the Durrmeyer singular system from Section F.2.

Lemma G.1 (Adjoint tail identity for integrable perturbations). Let $g \in L _ { 1 } [ 0 , 1 ]$ and define

$$
u _ { k } : = \int _ { 0 } ^ { 1 } b _ { k , n } ( p ) g ( p ) ~ \mathrm { d } p , \qquad k = 0 , \ldots , n .
$$

Then

$$
( n + 1 ) \sum _ { k = 0 } ^ { n } u _ { k } ^ { 2 } = \sum _ { j = 0 } ^ { n } \lambda _ { j , n } \left( \int _ { 0 } ^ { 1 } g ( p ) L _ { j } ( p ) \mathrm { d } p \right) ^ { 2 } .\tag{142}
$$

Proof. Define the coeficient vector $T _ { n } ^ { * } g$ algebraically by

$$
( T _ { n } ^ { * } g ) ( k ) = ( n + 1 ) u _ { k } .
$$

The vectors $v _ { 0 } , \ldots , v _ { n }$ in Section F.2 form an orthonormal basis of $( \mathbb { R } ^ { n + 1 } , \nu _ { n } )$ and satisfy $T _ { n } v _ { j } =$ $\sqrt { \lambda _ { j , n } } L _ { j }$ . Although g need not belong to $L _ { 2 }$ , every integral below is finite because $g \in L _ { 1 }$ and $L _ { j }$ is bounded. Hence

$$
\langle T _ { n } ^ { * } g , v _ { j } \rangle _ { \nu _ { n } } = \int _ { 0 } ^ { 1 } g ( p ) ( T _ { n } v _ { j } ) ( p ) \ \mathrm { d } p = { \sqrt { \lambda _ { j , n } } } \int _ { 0 } ^ { 1 } g ( p ) L _ { j } ( p ) \ \mathrm { d } p .
$$

Parseval in the finite-dimensional coeficient space gives (142), because

$$
\| T _ { n } ^ { * } g \| _ { \nu _ { n } } ^ { 2 } = \frac { 1 } { n + 1 } \sum _ { k = 0 } ^ { n } ( n + 1 ) ^ { 2 } u _ { k } ^ { 2 } = ( n + 1 ) \sum _ { k = 0 } ^ { n } u _ { k } ^ { 2 } .
$$

Proof of Theorem 6.1. Let $h _ { d } , G _ { d , \pm }$ , and $H _ { d , \pm }$ be, respectively, the degree-d dual certificate, raw alternatives, and margin-adjusted alternatives supplied by Theorem E.4. By Eq. (140) and Theorem E.3,

$$
| \theta ( H _ { d , + } ) - \theta ( H _ { d , - } ) | = 2 \rho _ { L } E _ { d } ( \beta ) \geq c _ { \beta , L } d ^ { - \beta } .\tag{143}
$$

We now bound the Hellinger distance between their count laws. Write $p = ( 1 + z ) / 2$ . Under this transformation, $\mu _ { \beta }$ has density

$$
w _ { \beta } ( p ) = \beta | 2 p - 1 | ^ { \beta - 1 } , \qquad 0 < p < 1 ,
$$

with respect to Lebesgue measure. Put

$$
g _ { d } ( p ) : = h _ { d } ( 2 p - 1 ) w _ { \beta } ( p ) ,
$$

and define

$$
q _ { 0 } ( k ) : = \int _ { 0 } ^ { 1 } b _ { k , n } ( p ) w _ { \beta } ( p ) \mathrm { d } p , \qquad u _ { d } ( k ) : = \int _ { 0 } ^ { 1 } b _ { k , n } ( p ) g _ { d } ( p ) \mathrm { d } p .
$$

Then

$$
K _ { n } G _ { d , + } = q _ { 0 } + u _ { d } , \qquad K _ { n } G _ { d , - } = q _ { 0 } - u _ { d } .
$$

If $r _ { 0 } : = K _ { n } H _ { 0 }$ and $m _ { 0 } : = \rho _ { L } q _ { 0 } + ( 1 - \rho _ { L } ) r _ { 0 }$ , the final pair has probability mass functions

$$
P _ { d , + } : = K _ { n } H _ { d , + } = m _ { 0 } + \rho _ { L } u _ { d } , \qquad P _ { d , - } : = K _ { n } H _ { d , - } = m _ { 0 } - \rho _ { L } u _ { d } .
$$

Because $m _ { 0 } \geq \rho _ { L } q _ { 0 }$ coordinatewise,

$$
\begin{array} { r l } & { \displaystyle h ^ { 2 } ( P _ { d , + } , P _ { d , - } ) = \sum _ { k = 0 } ^ { n } \left( \sqrt { m _ { 0 } ( k ) + \rho _ { L } u _ { d } ( k ) } - \sqrt { m _ { 0 } ( k ) - \rho _ { L } u _ { d } ( k ) } \right) ^ { 2 } } \\ & { \qquad \le 2 \rho _ { L } ^ { 2 } \displaystyle \sum _ { k = 0 } ^ { n } \frac { u _ { d } ( k ) ^ { 2 } } { m _ { 0 } ( k ) } \le 2 \rho _ { L } \displaystyle \sum _ { k = 0 } ^ { n } \frac { u _ { d } ( k ) ^ { 2 } } { q _ { 0 } ( k ) } . } \end{array}\tag{144}
$$

For $0 < \beta \leq 1 , w _ { \beta } ( p ) \geq \beta$ for almost every $p \in ( 0 , 1 )$ , and $\begin{array} { r } { \int _ { 0 } ^ { 1 } b _ { k , n } ( p ) \ \mathrm { d } p = \frac { 1 } { n + 1 } } \end{array}$ . Thus

$$
q _ { 0 } ( k ) \geq { \frac { \beta } { n + 1 } } .\tag{145}
$$

Combining (144) and (145) gives

$$
h ^ { 2 } ( P _ { d , + } , P _ { d , - } ) \leq \frac { 2 \rho _ { L } ( n + 1 ) } { \beta } \sum _ { k = 0 } ^ { n } u _ { d } ( k ) ^ { 2 } = \frac { 2 \rho _ { L } } { \beta } \| T _ { n } ^ { * } g _ { d } \| _ { \nu _ { n } } ^ { 2 } .\tag{146}
$$

The cancellation in (136) removes the first d singular modes. Indeed, $L _ { j } ( ( 1 + z ) / 2 )$ is a polynomial of degree $j$ in $z ,$ so

$$
\int _ { 0 } ^ { 1 } g _ { d } ( p ) L _ { j } ( p ) \ : \mathrm { d } p = 0 , \qquad j = 0 , \ldots , d .
$$

By Theorem G.1 and the monotonicity of $j \mapsto \lambda _ { j , n }$ ,

$$
\begin{array} { r l r } {  { \| T _ { n } ^ { * } g _ { d } \| _ { \nu _ { n } } ^ { 2 } = \sum _ { j = d + 1 } ^ { n } \lambda _ { j , n } ( \int _ { 0 } ^ { 1 } g _ { d } ( p ) L _ { j } ( p ) \mathrm { d } p ) ^ { 2 } } } \\ & { } & { \leq \lambda _ { d + 1 , n } \displaystyle \sum _ { j = d + 1 } ^ { n } ( \int _ { 0 } ^ { 1 } g _ { d } ( p ) L _ { j } ( p ) \mathrm { d } p ) ^ { 2 } . } \end{array}
$$

Since $\| g _ { d } \| _ { 1 } \leq 1$ and the orthonormal shifted Legendre polynomial satisfies $\| L _ { j } \| _ { \infty } = \sqrt { 2 j + 1 }$

$$
\left| \int _ { 0 } ^ { 1 } g _ { d } ( p ) L _ { j } ( p ) ~ \mathrm { d } p \right| \leq \sqrt { 2 j + 1 } .
$$

Therefore

$$
\| T _ { n } ^ { * } g _ { d } \| _ { \nu _ { n } } ^ { 2 } \leq \lambda _ { d + 1 , n } \sum _ { j = 0 } ^ { n } ( 2 j + 1 ) = \lambda _ { d + 1 , n } ( n + 1 ) ^ { 2 } .\tag{147}
$$

Equations (146) and (147) imply

$$
h ( P _ { d , + } , P _ { d , - } ) \leq C _ { \beta , L } \frac { n + 1 } { \kappa _ { d + 1 , n } } .
$$

Together with (143), this proves (75).

If $\beta > 1 / 2$ , then $g _ { d } \in L _ { 2 } [ 0 , 1 ]$ and $\begin{array} { r } { \| g _ { d } \| _ { 2 } ^ { 2 } \le \int _ { 0 } ^ { 1 } w _ { \beta } ( p ) ^ { 2 } ~ \mathrm { d } p = \frac { \beta ^ { 2 } } { 2 \beta - 1 } } \end{array}$ . Bessel’s inequality replaces (147) by

$$
\begin{array} { r } { \| T _ { n } ^ { * } g _ { d } \| _ { \nu _ { n } } ^ { 2 } \leq \lambda _ { d + 1 , n } \| g _ { d } \| _ { 2 } ^ { 2 } \leq C _ { \beta } \lambda _ { d + 1 , n } . } \end{array}
$$

Substitution into (146) proves (76).

Proof of the lower assertion in Theorem 6.2. For $j \le n / 2$ , the lower half of (72) gives

$$
\begin{array} { r } { \log \kappa _ { j , n } \geq c _ { \kappa } j ( j + 1 ) / n . } \end{array}
$$

Let

$$
\Lambda _ { Q , n } : = \log \bigl ( A _ { \beta , L } ( n + 1 ) \sqrt { Q } \bigr ) ,
$$

where $A _ { \beta , L }$ is large enough to absorb the constant in (75), and choose

$$
d + 1 = \left\lceil { \sqrt { \frac { 2 n \Lambda _ { Q , n } } { c _ { \kappa } } } } \right\rceil .
$$

Because $y _ { Q , n } \ \geq \ 1$ and $A _ { \beta , L }$ is fixed, $\Lambda _ { Q , n } \asymp _ { \beta , L } y _ { Q , n } .$ . If the constant $c _ { 0 }$ in Theorem $6 . 2 ( \mathrm { a } )$ is suficiently small and n is suficiently large, then $d + 1 \leq n / 2$ and

$$
\kappa _ { d + 1 , n } \geq A _ { \beta , L } ( n + 1 ) \sqrt { Q } .
$$

Consequently, the Hellinger radius in (75) is at most $Q ^ { - 1 / 2 }$ . Monotonicity of the modulus, Theorem 4.1, and $d \asymp { \sqrt { n \Lambda _ { Q , n } } }$ yield

$$
\mathcal { R } _ { Q , n } ^ { \mathrm { d e c } } ( \beta , L ) \gtrsim _ { \beta , L } { ( n \Lambda _ { Q , n } ) ^ { - \beta / 2 } } .
$$

The comparison $\Lambda _ { Q , n } \asymp _ { \beta , L } y _ { Q , n }$ proves (78). When $\beta > 1 / 2$ , apply (76) instead and set $\Lambda _ { Q } : =$ log $\scriptstyle ; ( A _ { \beta , L } { \sqrt { Q } } )$ . Then $\Lambda _ { Q } \asymp _ { \beta , L } y _ { Q } ^ { ( 2 ) }$ , and the same argument proves the refined assertion. □

Proof of the upper assertion in Theorem 6.2. For $d \leq n / 2$ , the upper half of (72) gives log $\kappa _ { d , n } \leq$ $C _ { \kappa } d ( d + 1 ) / n$ . Choose

$$
d = \left\lfloor a \sqrt { n x _ { Q , n } } \right\rfloor \vee 1
$$

with $a > 0$ small enough that $C _ { \kappa } d ( d + 1 ) / n \leq x _ { Q , n } / 4$ . Reducing $c _ { 0 }$ if necessary ensures $d \leq n / 2$ while $x _ { Q , n } \geq 1$ gives $d \asymp { \sqrt { n x _ { Q , n } } }$ . The spectral upper bound (73) therefore yields a bias term of order $( n x _ { Q , n } ) ^ { - \beta / 2 }$ and a stochastic term bounded by

$$
\frac { \sqrt { n + 1 } \kappa _ { d , n } } { \sqrt { Q } } \lesssim \frac { \sqrt { n } e ^ { x _ { Q , n } / 4 } } { n ^ { ( \beta + 1 ) / 2 } e ^ { x _ { Q , n } / 2 } } = n ^ { - \beta / 2 } e ^ { - x _ { Q , n } / 4 } .
$$

Since $e ^ { - x / 4 } \leq C _ { \beta } x ^ { - \beta / 2 }$ for $x \geq 1$ , the stochastic term has the same stated order, proving (79).