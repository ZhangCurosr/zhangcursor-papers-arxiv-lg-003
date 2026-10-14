# Understanding Circulant Permutation to Extend the CMinHash Estimator

Keegan Kang Department of Mathematics and Statistics Bucknell University Lewisburg, PA 17837 kk054@bucknell.edu

Avery Hood Department of Mathematics and Statistics Bucknell University Lewisburg, PA 17837 AveryHood1013@gmail.com

Benedict H.W. Wong Institute for Infocomm Research (I<sup>2</sup>R) A\*STAR Singapore, 138632 Benedict\_Wong@a-star.edu.sg

## Abstract

CMinHash promises a new direction for designing MinHash algorithms with reduced variance and storage space via circulant permutations. However, the original variance analysis is difficult to extend to other MinHash estimators. We present a new approach that gives explicit and easily comparable approximate variance expressions, and use it to analyze the Minner estimator. Numerical experiments show that our variance expression matches the empirical mean square error (MSE) of estimates. Our work thereby demonstrates how to apply circulant permutation to other MinHash estimators and compute their approximate variance.

## 1 Introduction

Consider two unweighted sets $S _ { 1 } , S _ { 2 } \subseteq \Omega$ , where $\Omega = \{ 1 , 2 , \dots , D \}$ . Let $| S _ { 1 } | = f _ { 1 } , | S _ { 2 } | = f _ { 2 }$ and $a = | S _ { 1 } \cap S _ { 2 } |$ , and $f = | S _ { 1 } \cup S _ { 2 } | . \operatorname { A }$ common measure of similarity between these two sets is the Jaccard similarity, given by $\begin{array} { r } { J = \frac { | \mathcal { S } _ { 1 } \cap \mathcal { S } _ { 2 } | } { | \mathcal { S } _ { 1 } \cup \mathcal { S } _ { 2 } | } = \frac { a } { f _ { 1 } + f _ { 2 } - a } } \end{array}$ . As D is usually large, directly computing the pairwise similarity between these sets is not computationally feasible, and the task is to instead estimate the Jaccard similarity of these sets via hashing algorithms. Measuring this similarity is useful in many applications, even in non statistical (or computer science) disciplines [16, 17, 20]. The traditional MinHash algorithm [3] uses random permutations to estimate J between any two sets, and since then there has been a large body of work on estimating J, which we divide into three categories.

New MinHash Approaches: These are examples of MinHash algorithms that describe a novel (at the time) approach to MinHash. For example, k-min sketches / Conditional Random Sampling records more information with one random permutation [4, 11], b-bit minwise hashing which stores the last b bits of information [8, 9], MinMaxHash uses the “other end" of the random permutation [5], One Permutation Hashing uses just one permutation [12, 18, 19], the Minner estimator and records more data for asymmetric hashing [1]. These contributions usually involve some kind of trade-off (e.g. data is required to be sparse for a better estimate, or greater computational time for reduced storage space), i.e. these methods do not work uniformly well for all datasets.

“Framework" that complements existing MinHash methods: These are techniques that can be applied to an existing algorithm to always improve its performance. The technique of control variates and maximum likelihood estimation can be applied to MinHash (and other similar algorithms) to always reduce their variance [6, 10], while CMinHash shows that “circulantly" moving rightwards across a permutation always reduces variance of estimates [14, 15].

Theoretical work which gives insight on the structure of MinHash: For example, [2] gives a concise theorem on query time, space usage, and update time of similarity search algorithms, [13] explains how to construct a MinHash type estimator across continuous probability distributions.

Our work belongs to the last two categories, where we re-examine the approach taken in the CMinHash paper [15], and propose a way to extend circulant permutations to other MinHash estimators.
<table><tr><td>Permutation</td><td>6</td><td>1</td><td>7</td><td>4</td><td>5</td><td>8</td><td>3</td><td>2</td><td>Draw Order</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td>Set 1</td><td>●</td><td>O</td><td>O</td><td>●</td><td>O</td><td>●</td><td>O</td><td>●</td><td>Set 1</td><td>o</td><td>●</td><td>O</td><td>●</td><td>O</td><td>●</td><td>O</td><td>●</td></tr><tr><td>Set 2</td><td>O</td><td>●</td><td>●</td><td>C</td><td>O</td><td>O</td><td>●</td><td>●</td><td>Set 2</td><td>●</td><td>●</td><td>●</td><td>0</td><td>O</td><td>O</td><td>●</td><td>O</td></tr></table>

Figure 1: We have $| S _ { 1 } | = 4 , | S _ { 2 } | = 5$ . By treating a random permutation (left) as the indices where we sample without replacement (right), min $\{ \pi ( \bar { S _ { 1 } } ) \} = 2 , \operatorname* { m i n } \{ \pi ( S _ { 2 } ) \} = 1$ , which is just the index where the first • appears.

## 1.1 Related Work

Without loss of generality, let $S _ { 1 } , S _ { 2 } \subseteq \Omega$ , and we want to estimate the Jaccard similarity J between these sets. These sets can be represented as D sequential objects consisting of one of four types: $\left\{ : , : , : , \ : \circ , \circ \ \right\}$ , where the • in the corresponding row denotes an element of that set.

Theorem 1. (Broder’s MinHash) Suppose we have K independent random permutations $\pi _ { 1 } , \ldots , \pi _ { K }$ on [D]. Let $\begin{array} { r c l } { \theta ^ { \mathrm { ( M H ) } } } & { = } & { \sum _ { t = 1 } ^ { K } \mathbb { 1 } _ { \left\{ \operatorname* { m i n } \{ \pi _ { t } ( S _ { 1 } ) \} = \operatorname* { m i n } \{ \pi _ { t } ( S _ { 2 } ) \} \right\} } \Big / K } \end{array}$ Then $\begin{array} { r l r } { { \mathbb E } \left( \theta ^ { ( \mathrm { M H } ) } \right) } & { { } = } & { J , } \end{array}$ , with Var $\big ( \theta ^ { ( \mathrm { M H } ) } \big ) = J ( 1 - J ) / K$

A random permutation π on $[ D ]$ can be thought of as a sequence of draws under sampling without replacement, where min $\{ \pi ( \bar { S _ { 1 } } ) \}$ is the draw where the first • appears in the first row, and min $\{ \pi ( S _ { 2 } ) \}$ as the draw where the first • appears in the second row. Figure 1 shows an example where $D = 8$ . Let the set $\mathcal { F } _ { + } = \left\{ : \right\}$ , and $\mathcal { F } _ { - } = \left\{ \begin{array} { l l } { \circ } \\ { \bullet } \end{array} , \begin{array} { l } { \bullet } \\ { \circ } \end{array} \right\}$ . Each permutation in Broder’s MinHash is equivalent to a trial where a “success" is drawing an element in $\mathcal { F } _ { + }$ before an element in $\mathcal { F } _ { - }$ , and an estimator for J is found by finding the probability that an element from $\mathcal { F } _ { + }$ is seen before an element in ${ \mathcal { F } } _ { - }$ Hence by Theorem 1, we get an estimate for J by recording the proportion of “successes" for K independent random permutations (trials), $1 \leq t \leq K$

More generally, other MinHash algorithms can be represented by different D sequential objects, $\mathcal { F } _ { + }$ and $\mathcal { F } _ { - }$ , and an estimator for J can be found by finding the probability that $n _ { 1 }$ elements from $\mathcal { F } _ { + }$ are seen before $n _ { 2 }$ elements in $\mathcal { F } _ { - }$ (we give an example in Section 3).

<table><tr><td>Permutation</td><td>3</td><td>2</td><td>6</td><td>1</td><td>7</td><td>4</td><td>5</td><td>8</td><td>Draw Order</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td>Set 1</td><td>●</td><td>O</td><td>O</td><td>●</td><td>O</td><td>●</td><td>O</td><td>●</td><td>Set 1</td><td>●</td><td>O</td><td>●</td><td>●</td><td>o</td><td>O</td><td>O</td><td></td></tr><tr><td>Set 2</td><td>O</td><td>C</td><td>●</td><td></td><td>O</td><td>O</td><td>●</td><td>●</td><td>Set 2</td><td>●</td><td>●</td><td>O</td><td>O</td><td></td><td></td><td>O</td><td></td></tr></table>

Figure 2: The permutation π in Figure 2 is shifted rightward from Figure 1 by 2 units. Then min $\{ \pi _ {  2 } ( S _ { 1 } ) \} = \operatorname* { m i n } \{ \pi _ {  2 } ( S _ { 2 } ) \} = 1$

CMinHash extends Broder’s MinHash by reducing the K independent permutations needed to two independent permutations, given by $\sigma , \pi$ , and we write this as CMinHash $( \sigma , \pi )$ following the notation in [15]. First, the indices in the sets $S _ { i }$ are permuted by the random permutation $\sigma .$ Second, the random permutation π is generated, and is re-used K times by shifting π circulantly rightwards by 1 unit K times. More formally, let $\pi  t$ denote shifting the permutation t units to the right. Figure 2 gives an example of shifting the permutation in Figure 1 rightwards by 2 units. By linearity of expectations, $\mathbb { E } _ { \sigma , \pi } ( \sum _ { t = 1 } ^ { K } \mathbb { 1 } _ { \{ \operatorname* { m i n } \{ \pi _ {  t } ( S _ { 1 } ) \} = \operatorname* { m i n } \{ \pi _ {  t } ( S _ { 2 } ) \} \} } / K ) = J$ , hence recording the proportion of “successes" under K shifts gives an estimate of J. A less compact version is on Page 13 in Appendix A.1. However, the variance of this estimate is slightly complicated, although lower than Broder’s MinHash (see Page 14 in Appendix A.2 for expression).

## 1.2 Our Contributions And Paper Organization

CMinHash(σ, π) promises a new direction to design hashing algorithms using the 2 permutations (σ, π) and K circulant shifts of π (saving on storage space), but having a smaller variance than K independent permutations under traditional MinHash (and similar algorithms).

While the algorithm behind CMinHash(σ, π) is straightforward to implement (see Algorithm 2 in Appendix A.1), the theoretical analysis of the variance forms a substantial bulk of the original paper.

Moreover, the approach the original paper adopts in terms of random permutations makes the proofs of their Lemma 2.4 (used to compute expectations under σ) and Theorem 3.1 (used to compute the variance of the estimates under σ and π) difficult to replicate in other settings due to the random permutation paradigm and combinatorial calculations. Moreover, the result of Theorem 3.1 (variance of the CMinHash estimator) in [15] gives the variance as a combinatorial sum under constraints. Appendix A.2 notes that one equality used in their proof does not hold in general. This makes it difficult to theoretically compare the CMinHash variance against variance of other algorithms, since Theorem 3.4 and Proposition 3.6 show that the variance (given by the combinatorial sum) is only lower than Broder’s MinHash estimate, and says nothing about other MinHash algorithms. The ful statements of relevant lemmas, theorems, and propositions in [15] can be found in Appendix A.2.

Hence, researchers using a circulant permutation to improve other MinHash estimators would need to independently show that using a circulant permutation theoretically gives a lower variance compared to different MinHash algorithms, which is non-trivial iffollowing similar approaches as in the original paper. Secondly, researchers may also want to compare the performance of two algorithms both using a circulant permutation framework. This is difficult to do theoretically, as the circulant permutation variance is given as a combinatorial sum under constraints if using the approach in [15].

While simulations can be run to compare the empirical performances of a proposed circulant permutation estimator against baselines and sidestep theory, there is no guarantee that a proposed estimator always has reduced variance in all simulated cases, since estimators can have different performance when datasets vary (e.g. sparsity of dataset). This leads us to give the following contributions.

Contribution One: We give a new proof of Lemma 2.4 in [15], using a different paradigm in Section 2. This allows simpler computation of expectations under the permutation σ. This approach can be applied to other MinHash estimators, and we give an example in Section 3.

Contribution Two: We derive a simpler expression for the variance in Theorem 3.1 in [15] under the assumption that $D \to \infty$ , which is common (e.g. see [9, 12, 19]). Our result does not require the equality discussed in Appendix A.2, and is slightly stronger in this respect. This allows easier analysis and comparison of the variance under the permutations σ and π. More specifically, this gives a simpler variance expression which is easier to analyze (and compare) with a remainder of order $f / \bar { D } ^ { 2 }$ when $f / D \to 0$ , and of order $D ^ { - 2 }$ when $f / \dot { D }$ is bounded away from zero, which we explain in Section 2. We remark that this gives approximately accurate variances even for small $D = 1 2 8$ . This approach can also be applied to other MinHash estimators to get variance expressions for the corresponding circulant permutation estimator, with an example in Section 3.

Contribution Three: We show how our approach can be used to analyze MinHash estimators based on circulant permutation, which we demonstrate on an asymmetric hashing setup with the Minner estimator. We pick the Minner estimator for two reasons: a) to show how to do variance analysis when the sets $\mathcal { F } _ { + }$ and $\mathcal { F } _ { - }$ are different, and b) how to extend the circulant permutation setup with counts, i.e. finding the expected number of elements seen in $\mathcal { F } _ { - }$ before an element in $\mathcal { F } _ { + }$

Overall, our paper provides a direct way to compute expectations and variance under circulant permutation, making it easier to analyze and extend MinHash estimators.

Section 2 (and A.1-A.2) give a broader overview of CMinHash. Section 3 applies our approach to the Minner estimator involving counts of events. Section 4 discusses the importance of our results.

## 2 CMinHash in Detail

The sets $S _ { 1 } , S _ { 2 } \subseteq \{ 1 , 2 , \dots , D \}$ described on Page 1 can be visualized as D objects in sequential order (see Figure 1). Let $a = \# \ : \mathrm { o f } \ : : : \ : , \ : f - a$ be the $\#$ of $\overset { \circ } { \cdot }$ and $\mathbf { \Phi } _ { \mathrm { { o } } }$ , and $D - f$ be the $\#$ of $\begin{array} { c } { { { \circ } } } \\ { { { \circ } } } \end{array}$

Definition 1. Definition 2.1 in [15] (modified). Define the location vector ${ \vec { x } } ,$ where $x _ { i } \in \{ \widetilde { O } , \widetilde { \times } , \widetilde { - } \}$ $x _ { i }$ corresponds to the type of the $i ^ { t h }$ object, which is a $\widetilde { O } \ : i f \ : : \widetilde { \times } \ : i f \ : \bullet \ : o r \ : \ : \bullet \ : , a n d \widetilde { - } \ : i f \ : \ : \mathring { \ : } \ : \circ$

Let $\begin{array} { r } { \theta ^ { \mathrm { ( C M H ) } } : = \sum _ { t = 1 } ^ { K } \mathbb { 1 } _ { \{ \operatorname* { m i n } \{ \pi _ {  t } ( S _ { 1 } ) \} = \operatorname* { m i n } \{ \pi _ {  t } ( S _ { 2 } ) \} \} } / K } \end{array}$ which is the CMinHash estimator. We write $\mathbb { 1 } _ { t } \equiv \mathbb { 1 } _ { \{ \operatorname* { m i n } \{ \pi _ {  t } ( S _ { 1 } ) \} = \operatorname* { m i n } \{ \pi _ {  t } ( S _ { 2 } ) \} \} }$ for compactness. $\mathbb { E } \left( \theta ^ { \mathrm { ( C M H ) } } \right)$ is J due to linearity of expectations (equal probability of J regardless of starting point after applying π), but computing Var $\left( \theta ^ { \mathrm { ( C M H ) } } \right)$  requires knowing $\operatorname { C o v } _ { \sigma , \pi } { ( \mathbb { 1 } _ { s } , \mathbb { 1 } _ { t } ) } , 1 \le s , t \le K , \mathrm { a s } \ \mathbb { 1 } _ { s } , \mathbb { 1 } _ { t }$ <sub>t</sub> are correlated.

Definition 2. Definition 2.2 in [15]. For $A , B \in \{ \widetilde { O } , \widetilde { \times } , \widetilde { = } \}$ , let $\{ ( A , B ) \mid \Delta \}$ be the set $\{ ( i , j )$ $( x _ { i } , x _ { j } ) = ( A , B ) , j - i = \Delta \} . \ { \mathrm { ~ } } F o r 1 \leq \Delta \leq K - \mathrm { ~ \bar { 1 } , ~ } d e f i n e$

$$
\mathcal { L } _ { 0 } ( \Delta ) = \{ ( \widetilde O , \widetilde O ) | \Delta \} , ~ \mathcal { L } _ { 1 } ( \Delta ) = \{ ( \widetilde O , \widetilde \times ) | \Delta \} , ~ \mathcal { L } _ { 2 } ( \Delta ) = \{ ( \widetilde O , \widetilde - ) | \Delta \} ,
$$

$$
\mathcal { G } _ { 0 } ( \Delta ) = \{ ( \widetilde { - } , \widetilde { O } ) | \Delta \} , ~ \mathcal { G } _ { 1 } ( \Delta ) = \{ ( \widetilde { - } , \widetilde { \times } ) | \Delta \} , ~ \mathcal { G } _ { 2 } ( \Delta ) = \{ ( \widetilde { - } , \widetilde { - } ) | \Delta \} ,
$$

$$
\mathcal { H } _ { 0 } ( \Delta ) = \{ ( \widetilde { \times } , \widetilde { O } ) | \Delta \} , \mathcal { H } _ { 1 } ( \Delta ) = \{ ( \widetilde { \times } , \widetilde { \times } ) | \Delta \} , \mathcal { H } _ { 2 } ( \Delta ) = \{ ( \widetilde { \times } , \widetilde { - } ) | \Delta \} ,
$$

with thefollowing constraintsfor all $1 \le \Delta \le K - 1$ -，

$$
\begin{array} { r l } & { \left| \mathcal { L } _ { 0 } \right| + \left| \mathcal { L } _ { 1 } \right| + \left| \mathcal { L } _ { 2 } \right| = \left| \mathcal { L } _ { 0 } \right| + \left| \mathcal { G } _ { 0 } \right| + \left| \mathcal { H } _ { 0 } \right| = a , } \\ & { \left| \mathcal { G } _ { 0 } \right| + \left| \mathcal { G } _ { 1 } \right| + \left| \mathcal { G } _ { 2 } \right| = \left| \mathcal { L } _ { 2 } \right| + \left| \mathcal { G } _ { 2 } \right| + \left| \mathcal { H } _ { 2 } \right| = D - f , } \\ & { \left| \mathcal { H } _ { 0 } \right| + \left| \mathcal { H } _ { 1 } \right| + \left| \mathcal { H } _ { 2 } \right| = \left| \mathcal { L } _ { 1 } \right| + \left| \mathcal { G } _ { 1 } \right| + \left| \mathcal { H } _ { 1 } \right| = f - a . } \end{array}
$$

The nine tuples $( A , B )$ in Definition 2 correspond to symbols at $x _ { i } , x _ { i + \Delta }$ in the location vector ${ \vec { x } } .$   
Lemma 1. Lemma 2.4 in [15]. For any $1 \leq s < t \leq K$ , with $t - s = \Delta$ , then $\mathbb { E } _ { \pi } \left( \mathbb { 1 } _ { s } \mathbb { 1 } _ { s + \Delta } \right) =$   
$| { \mathcal { L } } _ { 0 } ( \Delta ) | + ( | { \mathcal { G } } _ { 0 } ( \Delta ) | + | { \mathcal { L } } _ { 2 } ( \Delta ) | ) J$ f+|G<sub>0</sub>(∆)|+|G<sub>1</sub>(∆)|

We give a simpler proof in Appendix A.3 with a pictorial intuition in Table 4, and explain where our approach differs to [15]. Following the sampling without replacement paradigm where a random permutation can be thought of as the draw order, then for a fixed $\Delta .$ , then applying π means we are drawing the indices $( x _ { s } , x _ { s + \Delta } )$ from the location vector $\vec { x }$ in some random order, and we know how many of each $( A , B ) = ( x _ { s } , x _ { s + \Delta } )$ indices there are using Definition 2. The first entry in the tuple $( A , { \dot { B } } )$ then corresponds to what is seen when $\pi$ is shifted s times to the right, and the second entry corresponds to what is seen when π is shifted $s + \Delta$ times to the right. Since a $\widetilde O$ in the location vector means that the hashed values collide, then $\mathbb { E } _ { \pi } \left( \mathbb { I } _ { s } \mathbb { I } _ { s + \Delta } \right) > 0$ can be thought of as the probability of a sequence of draws of the nine different types of D tuples where the first $\widetilde O$ appears before the first ×e for both entries of the tuple, as the random permutation π is a sequence of draws without replacement. We call such a sequence of draws a success, with two examples below.

$$
\bullet ( \widetilde { - } , \widetilde { - } ) , ( \widetilde { - } , \widetilde { \bf O } ) , ( \widetilde { \bf O } , \widetilde { - } )
$$

$( \widetilde { - } , \widetilde { - } ) , ( \widetilde { - } , \widetilde { \mathbf { O } } ) , ( \widetilde { - } , \widetilde { \times } ) , ( \widetilde { \mathbf { O } } , \widetilde { O } )$ . The first $\widetilde O$ for the second entry of the tuples appeared before the first $\begin{array} { l } { \widetilde { \times } } \end{array}$ for the second entry of the tuples.

More examples are given by Table 2 and Table 3 in Appendix A.3. Hence, we just need to compute the probabilities of seeing the tuple we want before other invalid tuples, which is much simpler. The intuition behind this simplification can be seen behind this following analogy: Consider a bag with one blue marble, two red marbles, and N green marbles. The probability of drawing a blue marble before a red marble is always $\frac { 1 } { 3 }$ , regardless of N. We can do the same with $\widetilde O$ and $\widetilde { \times }$ , regardless of $\underline { { \sim } }$

Next, we compute the variance of the $\mathrm { C M i n H a s h } ( \sigma , \pi )$ estimator under K shifts adopting a different approach, which we outline here. Suppose D is large. Under the random permutation $\sigma ,$ both elements in the tuple $( x _ { s } , x _ { s + \Delta } )$ can be thought of drawing entries $x _ { s } , x _ { s + \Delta }$ without replacement from a population of size D with three different objects $\tilde { O } , \tilde { \times } , \tilde { = }$ . More specifically, each set $\{ ( A , B ) \ \bar { | } \ \hat { \Delta } \} , 1 \leq \Delta \leq K - 1$ can be expressed as

$$
| | \{ ( A , B ) \mid \Delta \} | = \sum _ { s = 1 } ^ { D } \mathbb { 1 } _ { \{ ( x _ { s } , x _ { s + \Delta } ) = ( A , B ) \} } .\tag{1}
$$

Therefore, we can take expectations across (functions of) (1). For example,

$$
\mathbb { E } _ { \sigma , \pi } \left( \vert \mathcal { L } _ { 0 } ( \Delta ) \vert \right) = \sum _ { s = 1 } ^ { D } \mathbb { E } _ { \sigma , \pi } \left( \mathbb { 1 } _ { \left\{ ( x _ { s } , x _ { s + \Delta } ) = ( \widetilde { O } , \widetilde { O } ) \right\} } \right) = \frac { a ( a - 1 ) } { D - 1 } .
$$

For the sparse case, we first condition on the positions of the f entries which are not $\underline { { \sim } }$ . This gives an exact expression whose first term can be computed by considering when two such entries are next to each other. For the non sparse case, we use the delta method on the same exact expression. The details are in Appendix $\mathbf { A . } 4 .$

Lemma 2. Suppose we pre-process the location vector with the random permutation $\sigma .$ Let $B _ { D } =$ $2 D - f - 1$ , and suppose $0 \textless J < 1$ . For all $1 \leq \Delta \leq K - 1 , i f f / D  0 ,$ , then

$$
\begin{array} { r l } & { \mathbb { E } _ { \sigma , \pi } \left( 1 _ { s } \mathbb { 1 } _ { s + \Delta } \right) = \left\{ \begin{array} { l l } { J ^ { 2 } - \displaystyle \frac { J ( 1 - J ) f } { 2 ( f - 1 ) ( D - 1 ) } + { \cal O } \left( \frac { f } { D ^ { 2 } } \right) , } & { D e \nu e n a n d \Delta = D / 2 , } \\ { J ^ { 2 } - \displaystyle \frac { J ( 1 - J ) f } { ( 2 f - 1 ) ( D - 1 ) } + { \cal O } \left( \frac { f } { D ^ { 2 } } \right) , } & { o t h e r w i s e , } \end{array} \right. } \end{array}
$$

$$
\begin{array} { r } { \mathrm { C o v } _ { \sigma , \pi } \left( \mathbb { 1 } _ { s } , \mathbb { 1 } _ { s + \Delta } \right) = \left\{ \begin{array} { l l } { - \displaystyle \frac { J ( 1 - J ) f } { 2 ( f - 1 ) ( D - 1 ) } + { \cal O } \left( \frac { f } { D ^ { 2 } } \right) , } & { D e \nu e n a n d \Delta = D / 2 , } \\ { - \displaystyle \frac { J ( 1 - J ) f } { ( 2 f - 1 ) ( D - 1 ) } + { \cal O } \left( \frac { f } { D ^ { 2 } } \right) , } & { o t h e r w i s e . } \end{array} \right. } \end{array}
$$

$H f / D$ is bounded away from zero, then

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } _ { \sigma , \pi } \left( \mathbb { 1 } _ { s } \mathbb { 1 } _ { s + \Delta } \right) = J ^ { 2 } - \frac { J ( 1 - J ) } { B _ { D } } + O \left( D ^ { - 2 } \right) , } \\ & { \displaystyle \mathrm { C o v } _ { \sigma , \pi } \left( \mathbb { 1 } _ { s } , \mathbb { 1 } _ { s + \Delta } \right) = - \frac { J ( 1 - J ) } { B _ { D } } + O \left( D ^ { - 2 } \right) . } \end{array}
$$

We prove this in Appendix A.4.

Definition 3. For $0 < K \le D$ , let $\lambda _ { D , K } = \mathbb { 1 } _ { \left\{ D \ e v e n \right\} } \frac { ( 2 K - D ) _ { + } } { K } , w h e r e \ ( x ) _ { + } : = \operatorname* { m a x } \{ x , 0 \}$

Definition 3 accounts for the proportion of terms in the covariance where $\Delta = D / 2$

Theorem 2. Let $0 < a < f \leq D$ . Suppose we pre-process the location vector with the random permutation σ, and generate an independent permutation π. Let $B _ { D } = 2 D - f - 1 . \ I f f / D \to 0 ,$

$$
\operatorname { V a r } _ { \sigma , \pi } \left( \theta ^ { ( \mathrm { C M H } ) } \right) = \frac { J ( 1 - J ) } { K } \left[ 1 - \frac { f } { D - 1 } \left\{ \frac { K - 1 - \lambda _ { D , K } } { 2 f - 1 } + \frac { \lambda _ { D , K } } { 2 ( f - 1 ) } \right\} \right] + O \left( \frac { f } { D ^ { 2 } } \right) .
$$

$H f / D$ is bounded awayfrom zero, then

$$
\mathrm { V a r } _ { \sigma , \pi } \left( \theta ^ { ( \mathrm { C M H } ) } \right) = \frac { J ( 1 - J ) } { K } \left( 1 - \frac { K - 1 } { B _ { D } } \right) + { \cal O } \left( D ^ { - 2 } \right) .
$$

Proof. We give the proof in Appendix A.4.

The proof of Lemma 2 also gives an exact negative expression for every covariance. Hence, for $K > 1$ , the variance under CMinHash $( \sigma , \pi )$ is smaller than the variance $J ( 1 - J ) / K$ under K independent permutations. If $f / D$ is bounded away from zero, the first-order variance reduction can be thought of as approximately scaled by $( 1 - \bar { ( K ^ { \prime } - 1 ) } / ( 2 D - f - 1 ) )$ ). The leading variance ratio relative to MinHash is independent of a when $\dot { \cal D } , f , \dot { \cal K }$ are fixed, which also verifies the Uniform Superiority (Theorem 3.4) and Consistent Improvement (Proposition 3.6) statements in [15].

![](images/f8b6959ef941be00d571eae22381047188a28af024ce6a0ea5cd26dbd5fe97be.jpg)  
Figure 3: Empirical MSE vs theoretical approximate variance in Theorem 2 of MinHash and CMinHash (σ, π). The dotted lines overlap almost perfectly (up to Monte Carlo error), which shows our theoretical approximate variance matches the empirical variance.

We first construct a similar experiment to [15], generating synthetic vectors with varying $D , f , a$ in Figure 3, averaging over 10000 iterations, and overlaying the theoretical approximation with the empirical mean square error (MSE) of the estimates. The dotted lines overlap almost perfectly with the solid lines at $\bar { D } = 1 2 8$ and $D = 2 5 6$

We do more experiments in Appendix A.5 (Figure 6) to investigate if the approximation is accurate in edge cases when $f / D \to { \bar { 0 } } .$ . At D = 128, for $f = 2 , 4$ , the variance approximation is slightly higher than the empirical MSE. But at $D = 1 0 2 4$ for $f = 2 , 4$ , both lines overlap each other almost perfectly. We do more experiments on the Words dataset [7] (Figure 7 in Appendix A.5), where the theoretical variance approximation and empirical MSE lines also overlap each other almost perfectly.

Hence, we believe our approximation of the theoretical variance is practical enough to do comparisons.

We argue that our approach in Lemma 2 should be how MinHash algorithms are analyzed, especially with large D, as large D necessitated the need of MinHash in the first place.

## 3 Applying the Circulant Permutation Framework To Asymmetric Hashing

A natural question to ask is if our work is generalizable to other MinHash algorithms. We show our approach can be applied to the Minner estimators [1] (asymmetric hashing), and theoretically demonstrate a circulant permutation framework with reduced variance.

The asymmetric hashing setup considers a baseline set $S _ { 1 }$ relative to $S _ { 2 } , \ldots , S _ { n }$ , and estimates the set overlap $a _ { 1 i }$ (which can then be used to estimate the Jaccard similarity $J _ { 1 i } ,$ since $\begin{array} { r } { a _ { 1 i } = \frac { ( f _ { 1 } + f _ { i } ) J _ { 1 i } } { 1 + J _ { 1 i } } ) } \end{array}$ under the assumption that the set cardinalities $f _ { 1 } , \ldots , f _ { n }$ are all known. We give more details of the Minner estimators in Appendix B.1.

<table><tr><td>Draw Order</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td>Set 1</td><td>O</td><td>●</td><td>O</td><td>●</td><td>●</td><td>●</td><td>O</td><td>●</td></tr><tr><td>Set 2</td><td>O</td><td>o</td><td>O</td><td>O</td><td>O</td><td>O</td><td>o</td><td></td></tr></table>

Figure 4: We have min $\{ \pi _ { t } ( S _ { 2 } ) \} = 8 .$ , with $y _ { t } = 1$ , which can be thought of as the presence (or absence) of a • at draw 8 for $ { \boldsymbol { S } } _ { 1 }$ , and $m _ { t } = 4$ , which can be thought of as counting how many • were seen in previous draws 1-7 for $S _ { 1 }$

Theorem 3. Minner estimators (heavily modified from [1]) Suppose we have K independent random permutations $\pi _ { 1 } , \ldots , \pi _ { K }$ on $[ D ]$ , and we want to estimate the set overlap a between $\bar { \cal S } _ { 1 }$ and ${ \cal S } _ { 2 } ,$ , where

$S _ { 1 }$ is the baseline set. Let $y _ { t } = \mathbb { 1 } _ { \{ \operatorname* { m i n } ( \pi _ { t } ( S _ { 2 } ) ) \in \pi _ { t } ( S _ { 1 } ) \} }$ , and let $m _ { t }$ be the count ofelements in $\pi _ { t } ( S _ { 1 } )$ strictly less than min $\left( \pi _ { t } ( S _ { 2 } ) \right)$ . Then the two estimators $f _ { 2 } \sum _ { t = 1 } ^ { K } { \frac { y _ { t } } { K } }$ and $\begin{array} { r } { f _ { 1 } - ( f _ { 2 } + 1 ) \sum _ { t = 1 } ^ { K } \frac { m _ { t } } { K } } \end{array}$ both have expectation $a .$ . Their variances are, respectively, $\begin{array} { r } { \frac { a ( f _ { 2 } - a ) } { K } \overset { \_ } { a n d } \frac { f _ { 2 } ( f _ { 1 } - a ) ( f _ { 1 } + f _ { 2 } - a + 1 ) } { K ( f _ { 2 } + 2 ) } } \end{array}$

We remark that thinking of the random permutation as the draw order in sampling without replacement (see Figure 4) makes understanding the Minner estimators straightforward.

## 3.1 Applying the Circulant Permutation framework to a different setup of the location vector

Without loss of generality, consider estimating the set overlap between $S _ { 1 }$ and $S _ { 2 }$ , where we look at the random variable $y _ { t }$ . The question we aim to resolve is: How can the circulant permutation framework be applied when we are not recording both min $\{ \pi _ {  t } ( S _ { i } ) \} \ 2$

In Section 2, we introduced $\mathcal { F } _ { + }$ and $\mathcal { F } _ { - }$ , where we find the probability that an element from $\mathcal { F } _ { + }$ is seen before an element from $\mathcal { F } _ { - }$ . In the context of $y _ { t }$ , we set $\mathcal { F } _ { + } = \left\{ : \right\}$ , and $\mathcal { F } _ { - } = \left\{ \begin{array} { l } { { \circ } } \\ { { \bullet } } \end{array} \right\}$ as we continue drawing if we see a $\mathrm { ~  ~ { ~ \mathsf ~ { ~ o ~ r ~ } ~ } ~ } _ { 0 } ^ { \bullet }$ . Let $C _ { s } = \#$ of elements of $\mathcal { F } _ { + }$ in $D$ (success), $C _ { f } = \#$ of elements of $\mathcal { F } _ { - }$ in D (failure), and $C _ { n } = D - C _ { s } - C _ { f }$ (neutral). There are $C _ { s } = a \widetilde { O } \mathrm { s }$ $C _ { f } = f _ { 2 } - a \ { \stackrel { \sim } { \times } } \ s$ , and $C _ { n } = D - f _ { 2 } { \stackrel { \sim } { - } } \mathbf { s }$ . To generalize this, the constraints of the sets are now

$$
\begin{array} { r } { \left| \mathcal { L } _ { 0 } \right| + \left| \mathcal { L } _ { 1 } \right| + \left| \mathcal { L } _ { 2 } \right| = \left| \mathcal { L } _ { 0 } \right| + \left| \mathcal { G } _ { 0 } \right| + \left| \mathcal { H } _ { 0 } \right| = C _ { s } , } \\ { \left| \mathcal { G } _ { 0 } \right| + \left| \mathcal { G } _ { 1 } \right| + \left| \mathcal { G } _ { 2 } \right| = \left| \mathcal { L } _ { 2 } \right| + \left| \mathcal { G } _ { 2 } \right| + \left| \mathcal { H } _ { 2 } \right| = C _ { n } , } \\ { \left| \mathcal { H } _ { 0 } \right| + \left| \mathcal { H } _ { 1 } \right| + \left| \mathcal { H } _ { 2 } \right| = \left| \mathcal { L } _ { 1 } \right| + \left| \mathcal { G } _ { 1 } \right| + \left| \mathcal { H } _ { 1 } \right| = C _ { f } . } \end{array}
$$

It is clear that under one permutation, the variance of drawing an element from $\mathcal { F } _ { + }$ before $\mathcal { F } _ { - }$ is $\frac { C _ { s } C _ { f } } { ( C _ { s } + C _ { f } ) ^ { 2 } }$ regardless of the size of $C _ { n }$ . Let $V _ { \mathrm { i n d } }$ be the variance of 1 trial, and $V _ { K }$ the variance of K independent trials (i.e. under $K$ independent permutations). This leads to the following corollary.

Corollary 1. Suppose we pre-process the location vector with a random permutation $\sigma ,$ , and let $\mathbb { 1 } _ { s } \equiv \mathbb { 1 } _  \{ E l e m e n t i n \mathcal { F } _ { + }$ is seen before element in $\mathcal { F } _ { - } \ : a t \ : \pi _ {  s } \}$ . Then for any $1 \leq s < t \leq K$ , with $t - s = \Delta$

$$
\mathbb { E } _ { \pi } \left( \mathbb { 1 } _ { s } \mathbb { 1 } _ { s + \Delta } \right) = \frac { \vert \mathcal { L } _ { 0 } ( \Delta ) \vert + ( \vert \mathcal { G } _ { 0 } ( \Delta ) \vert + \vert \mathcal { L } _ { 2 } ( \Delta ) \vert ) \frac { C _ { s } } { C _ { s } + C _ { f } } } { C _ { s } + C _ { f } + \vert \mathcal { G } _ { 0 } ( \Delta ) \vert + \vert \mathcal { G } _ { 1 } ( \Delta ) \vert } .
$$

Let $\lambda _ { D , K }$ be defined as in Definition $^ { 3 , }$ and let $B _ { D } = D + C _ { n } - 1 . \ I f ( C _ { s } + C _ { f } ) / D \to 0 ,$ , then

$$
\operatorname { V a r } _ { \sigma , \pi } \left( \sum _ { t = 1 } ^ { K } \frac { 1 } { K } \right) = V _ { K } \left[ 1 - \frac { C _ { s } + C _ { f } } { D - 1 } \left\{ \frac { K - 1 - \lambda _ { D , K } } { 2 ( C _ { s } + C _ { f } ) - 1 } + \frac { \lambda _ { D , K } } { 2 ( C _ { s } + C _ { f } - 1 ) } \right\} \right] + O \left( \frac { C _ { s } + C _ { f } } { D ^ { 2 } } \right) .
$$

$I f ( C _ { s } + C _ { f } ) / D$ is bounded away from zero, then

$$
\mathrm { V a r } _ { \sigma , \pi } \left( \sum _ { t = 1 } ^ { K } \frac { \mathbb { 1 } _ { t } } { K } \right) = V _ { K } \left( 1 - \frac { K - 1 } { B _ { D } } \right) + O ( D ^ { - 2 } ) .
$$

Proof. We substitute $a = C _ { s }$ and $f = C _ { s } + C _ { f }$ in Lemma 1, Lemma 2, and Theorem 2. Since $D - f = C _ { n }$ , the quantity $2 D - f - 1$ becomes $D + C _ { n } - 1 = B _ { D }$ □

By the exact covariance identity (16), the variance under $K > 1$ circulant shifts is always reduced compared to K independent permutations. The size of the reduction depends on $\left( C _ { s } + C _ { f } \right)$ . In particular, MinHash algorithms where $\mathcal { F } _ { + }$ and $\mathcal { F } _ { - }$ are small relative to $\bar { D }$ have a small variance reduction when CMinHash $( \sigma , \pi )$ is applied.

With respect to the Minner estimator $y _ { t } .$ , we can apply Corollary 1 and immediately get the theoretical approximation of the variance of $y _ { t }$ by letting $C _ { s } ^ { - } = { a , C _ { f } } = \mathsf { \bar { f } } _ { 2 } - { a }$

## 3.2 Applying the Circulant Permutation framework to counts

Now consider estimating the set overlap between $S _ { 1 }$ and $S _ { 2 }$ , where we look at the random variable $m _ { t }$ . The question we aim to resolve is: How can the circulant permutation framework be applied when we are recording counts ofelements in $\mathcal { F } _ { - }$ before thefirst element in ${ \dot { \mathcal { F } } } _ { + }$ is $s e e n ?$

We still construct the two sets $\mathcal { F } _ { + } , \mathcal { F } _ { - }$ , but give a different meaning for elements in $\mathcal { F } _ { - }$ . Let $\mathcal { F } _ { + } = \left\{ : \frac { \circ } { \bullet } , : \frac { \circ } { \bullet } \ \right\}$ , and $\mathcal { F } _ { - } = \left\{ \begin{array} { l } { \bullet } \\ { \circ } \end{array} \right\}$ with $C _ { s } = f _ { 2 }$ and $C _ { f } = f _ { 1 } - a$ . Then the setup becomes: “count how many elements in $\mathcal { F } _ { - }$ are seen before the first element in $\mathcal { F } _ { + } "$ . Equivalently this would be “count how many $\widetilde { \times }$ are seen before the first $\widetilde O$ is drawn" for the location vector ${ \vec { x } } .$

From Lemma 5 in Appendix B.2, the variance of observing θ elements from $\mathcal { F } _ { - }$ before the first $\mathcal { F } _ { + }$ is $\frac { C _ { s } C _ { f } ( C _ { s } + C _ { f } + 1 ) } { ( C _ { s } + 1 ) ^ { 2 } ( C _ { s } + 2 ) }$ . Let $V _ { \mathrm { i n d } }$ be the variance of 1 trial, and $V _ { K }$ the variance of K independent trials.

Lemma 3. Suppose we pre-process the location vector with a random permutation σ. Let the random variable $\theta _ { s } \equiv$ number $y \tilde { \times }$ seen before thefirst $\widetilde O$ under the random permutation $\pi  s \cdot$ Let $B _ { \Delta } = 2 C _ { s } - \left. \mathcal { L } _ { 0 } \right. + 1$ . Then for any $1 \leq s < t \leq K$ , with $t - s = \Delta$

$$
\mathbb { E } _ { \pi } \left( \theta _ { s } \theta _ { s + \Delta } \right) = \frac { \left| \mathcal { H } _ { 1 } \right| } { B _ { \Delta } } + \frac { 2 C _ { f } ^ { 2 } - C _ { f } ( \left| \mathcal { L } _ { 1 } \right| + \left| \mathcal { H } _ { 0 } \right| ) - 2 \left| \mathcal { H } _ { 1 } \right| } { ( C _ { s } + 1 ) ( B _ { \Delta } + 1 ) } + \frac { C _ { f } ( \left| \mathcal { L } _ { 1 } \right| + \left| \mathcal { H } _ { 0 } \right| ) - 2 \left| \mathcal { L } _ { 1 } \right| \left| \mathcal { H } _ { 0 } \right| } { ( C _ { s } + 1 ) B _ { \Delta } ( B _ { \Delta } + 1 ) } .
$$

$I f C _ { s }$ and $C _ { f }$ are fixed as $D \to \infty ,$ , then, when D is even and $\Delta = D / 2$

$$
\mathrm { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = - V _ { \mathrm { i n d } } \frac { ( C _ { s } + 1 ) ( C _ { s } + 2 ) } { 2 C _ { s } ( 2 C _ { s } + 1 ) } \frac { 2 C _ { s } + C _ { f } } { ( C _ { s } + C _ { f } + 1 ) ( D - 1 ) } + O ( D ^ { - 2 } ) ,
$$

and otherwise,

$$
\mathrm { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = - V _ { \mathrm { i n d } } \frac { ( C _ { s } + 2 ) ( C _ { s } + C _ { f } ) } { ( 2 C _ { s } + 1 ) ( C _ { s } + C _ { f } + 1 ) ( D - 1 ) } + O ( D ^ { - 2 } ) .
$$

If both $C _ { s } / D$ and $C _ { f } / D$ are bounded away from zero, then

$$
\begin{array} { r } { \mathrm { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = \left\{ \begin{array} { l l } { - V _ { \mathrm { i n d } } \left( \frac { C _ { s } \left( 2 D - C _ { s } \right) + C _ { f } \left( D - C _ { s } \right) } { \left( C _ { s } + C _ { f } \right) \left( 2 D - C _ { s } \right) ^ { 2 } } \right) + O \left( \frac { 1 } { D ^ { 2 } } \right) , } & { D e \nu e n a n d \Delta = D / 2 , } \\ { - V _ { \mathrm { i n d } } } & { \left( \frac { 1 } { 2 D - C _ { s } } \right) } & { + O \left( \frac { 1 } { D ^ { 2 } } \right) , } \end{array} \right. } \end{array}
$$

Theorem 4. Suppose we pre-process the location vector with a random permutation σ. Let the random variable $\theta _ { t } \equiv$ number of $\widetilde { \times }$ seen before thefirst $\widetilde O$ under the random permutation $\pi  t \cdot$ Let $K > 1$ , and let $\lambda _ { D , K }$ be defined as in Definition 3. $I f C _ { s }$ and $C _ { f }$ arefixed as $D \to \infty ,$ then

$$
\begin{array} { r l r } {  { \operatorname { V a r } _ { \sigma , \pi } ( \sum _ { t = 1 } ^ { K } \frac { \theta _ { t } } { K } ) = V _ { K } [ 1 - \frac { 1 } { D - 1 } \{ ( K - 1 - \lambda _ { D , K } ) \frac { ( C _ { s } + 2 ) ( C _ { s } + C _ { f } ) } { ( 2 C _ { s } + 1 ) ( C _ { s } + C _ { f } + 1 ) }   } } \\ & { } & {   + \lambda _ { D , K } \frac { ( C _ { s } + 1 ) ( C _ { s } + 2 ) } { 2 C _ { s } ( 2 C _ { s } + 1 ) } \frac { 2 C _ { s } + C _ { f } } { C _ { s } + C _ { f } + 1 } \} ] + O ( D ^ { - 2 } ) . } \end{array}
$$

If both $C _ { s } / D$ and $C _ { f } / D$ are bounded away from zero, then

$$
\mathrm { V a r } _ { \sigma , \pi } \left( \sum _ { t = 1 } ^ { K } \frac { \theta _ { t } } { K } \right) = V _ { K } \left[ 1 - \frac { K - 1 } { 2 D - C _ { s } } \left\{ 1 - \frac { \lambda _ { D , K } C _ { f } D } { ( K - 1 ) ( C _ { s } + C _ { f } ) ( 2 D - C _ { s } ) } \right\} \right] + O ( D ^ { - 2 } ) .
$$

Proof. The proof is similar to Theorem 2, where we add up the covariance terms.

Each term within the square brackets in Theorem 4 is $< 1$ . Hence the circulant permutation estimate gives a lower variance asymptotically. Substituting $C _ { s } = f _ { 2 }$ and $C _ { f } = f _ { 1 } - a$ gives the variance approximation of $m _ { t }$

To verify the approximations, we run experiments on synthetic data at $D = 1 2 8$ with varying $f _ { 1 } , f _ { 2 } ,$ a values over 10000 iterations, where we similarly overlay the approximate theoretical variance when $\begin{array} { r } { \frac { C _ { s } + C _ { f } } { D } = \frac { f _ { 1 } + f _ { 2 } - a } { D } } \end{array}$ do not go to zero with the empirical MSEs in Figure 5. Unlike Figure 3, where the approximate theoretical variance overlaps with the empirical MSE, we see that at $\bar { f _ { 1 } } = 1 8 , f _ { 2 } = 3 2$ we see the theoretical variance curves do not overlap the empirical MSEs. However, how good is our variance approximation for ${ \frac { C _ { s } + C _ { f } } { D } } \to 0 { \mathrm { ~ i f ~ } } D$ is large, and $f _ { 1 } , f _ { 2 }$ are small?

![](images/5ceed9bf6403528f38a33fe34591770d946b02ed26ba32290ead57761d0fd291.jpg)  
Figure 5: Empirical MSE vs theoretical approximate variance in Theorem 4 for count data. The dotted lines overlap almost perfectly for the last three plots.

In Appendix B.3, we run another experiment averaged over 10000 iterations where we fix $( f _ { 1 } , \bar { f _ { 2 } } , a ) \ = \ ( 6 , 2 4 , 6 )$ , and increase this tuple up to $( f _ { 1 } , f _ { 2 } , a ) \ = \ ( 9 6 , 1 2 8 , 1 3 2 )$ , with $D \in$ $\{ 5 1 2 , 1 0 2 4 , 2 0 4 8 , 4 0 9 6 \}$ . In general, as $D$ increases, both our approximation variance ratios match the empirical MSEs. Finally, we do experiments with the Words dataset as in Section 2 (Figure 14 in Appendix B). For the word pairs we choose, both theoretical approximate variance curves overlap the empirical MSE curves perfectly, even for the HONG-KONG word pair, where $\begin{array} { r } { \frac { f _ { 1 } - a } { D } = \frac { 4 1 } { 6 5 5 3 6 } } \end{array}$ is sparse.

## 4 Discussion and Conclusion

Each observation $\theta _ { s }$ recorded by the circulant permutation estimator is correlated with another observation $\theta _ { t } ,$ , due to the circulant shift. Thus in order to theoretically compute (or compare) the variance of such an estimate, the corresponding covariance terms $\operatorname { C o v } _ { \pi } \left( \theta _ { s } , \mathbf { \dot { \theta } } _ { t } \right) = \mathbb { E } _ { \pi } \left( \theta _ { s } \mathbf { \dot { \theta } } _ { t } \right) -$ $\mathbb { E } _ { \pi } \left( \theta _ { s } \right) \mathbb { E } _ { \pi } \left( \theta _ { t } \right)$ need to be computed. Computing $\mathbb { E } _ { \pi } ^ { \mathrm { ~ - ~ } } ( \theta _ { s } \theta _ { t } )$ is the first hurdle in circulant permutation algorithms, yet is needed to compute the variances for such estimators.

But by viewing MinHash under the paradigm of the draw order under sampling without replacement, we simplify the proof to compute $\bar { \mathbb { E } } _ { \pi } \left( \theta _ { s } \bar { \theta } _ { t } \right)$ in [15] via our approach in Lemma 1. Moreover, by using indicator functions and considering exchangeability, we can further compute $\mathbb { E } _ { \pi } \left( \theta _ { s } \theta _ { t } \right)$ for counts as well (our Lemma 3), showing our approach applies beyond the original CMinHash setting.

The second hurdle with circulant permutations is finding the variance of the corresponding estimate. While Theorem 3.1 in [15] states a combinatorial expression for the variance, one equality in its proof does not hold in general (see Appendix A.2). Even apart from this point, the expression is not easily tractable, nor easy to analyze or compare against other MinHash algorithms, e.g. [15] had Theorem 3.4, Proposition $3 . 6 ,$ and many, many experiments in their paper just to compare CMinHash with MinHash (without even comparing against other MinHash algorithms). Researchers building on the circulant permutation framework would face similar technical difficulties when comparing variances.

Our approach suggests that exact accuracy is not needed, particularly because MinHash algorithms are used when $\bar { D _ { } }$ is large in the first place. Lemma 2 shows that by letting D be large, we can do a straightforward computation to compute approximations of the variance. We demonstrate how to compute an approximation when set sizes are sparse relative to $D _ { \colon }$ , and when they are non sparse. Both approximations are accurate for practical purposes, allowing theoretical comparison of variances with other MinHash algorithms in the large $D$ scenario.

Furthermore, our approach is not limited to CMinHash and applies to other MinHash estimators based on circulant permutation. The location vector setup of $\tilde { O } , \tilde { \times } , \tilde { = }$ can be characterized as: “stop at the $s ^ { \mathrm { t h } } \widetilde { O } ,$ do something at ${ \tt a } \stackrel { \sim } { \times } ,$ and continue at $\mathbf { a } \ { \overset { \sim } { - } } \ "$ . Our proof approach in Lemma 1 can be modified to a “stop at the $s ^ { \mathrm { t h } } \widetilde O ^ { \prime \prime }$ scenario, and our approach with indicator random variables in Lemma $^ 3$ can account for a wide range of do somethings, by conditioning on the element ×e itself, providing a direct way to compute expectations and variance for such estimators.

Thus, we believe our work makes circulant permutation estimators easier to analyze and extend. While state of the art algorithms are important, it is equally important to be able to understand and extend these algorithms. We provide a clear method to compute expectations and variance, and our proof techniques can be used directly to extend circulant permutations for other MinHash algorithms.

Limitations: Generally, D needs to be large for accurate approximations. Our experiments show that even D = 65536 (words data) gives good approximations. In practice, D is much larger (e.g. 2<sup>32</sup>).

## Acknowledgments and Disclosure of Funding

We would like to thank Bucknell’s Presidential Fellow program, which paid a stipend to Avery Hood working on this paper.

## References

[1] Thomas Ahle. Minner: Improved similarity estimation and recall on MinHashed databases, 2021. URL https://thomasahle.com/papers/minner.pdf. Manuscript.

[2] Thomas Dybdahl Ahle and Jakob Bæk Tejs Knudsen. Subsets and supermajorities: Optimal hashing-based set similarity search. In 2020 IEEE 61st Annual Symposium on Foundations of Computer Science, pages 728–739. IEEE, 2020. doi: 10.1109/FOCS46700.2020.00073.

[3] Andrei Z. Broder. On the resemblance and containment of documents. In Proceedings of the Conference on Compression and Complexity of Sequences, pages 21–29. IEEE, 1997. doi: 10.1109/SEQUEN.1997.666900.

[4] Edith Cohen and Haim Kaplan. Summarizing data using bottom-k sketches. In Proceedings of the Twenty-Sixth Annual ACM Symposium on Principles of Distributed Computing, pages 225–234. Association for Computing Machinery, 2007. doi: 10.1145/1281100.1281133.

[5] Jianqiu Ji, Jianmin Li, Shuicheng Yan, Qi Tian, and Bo Zhang. Min-max hash for Jaccard similarity. In 2013 IEEE 13th International Conference on Data Mining, pages 301–309. IEEE, 2013. doi: 10.1109/ICDM.2013.119.

[6] Keegan Kang, Sergey Kushnarev, Wei Pin Wong, Rameshwar Pratap, Haikal Yeo, and Chen Yijia. Improving hashing algorithms for similarity search via MLE and the control variates trick. In Proceedings of the 13th Asian Conference on Machine Learning, volume 157 of Proceedings of Machine Learning Research, pages 814–829. PMLR, 2021. URL https: //proceedings.mlr.press/v157/kang21a.html.

[7] Ping Li and Kenneth W. Church. Using sketches to estimate associations. In Proceedings of Human Language Technology Conference and Conference on Empirical Methods in Natural Language Processing, pages 708–715, Vancouver, British Columbia, Canada, October 2005. Association for Computational Linguistics. URL https://aclanthology.org/H05-1089/.

[8] Ping Li and Arnd Christian König. b-bit minwise hashing. In Proceedings ofthe 19th International Conference on World Wide Web, pages 671–680. Association for Computing Machinery, 2010. doi: 10.1145/1772690.1772759.

[9] Ping Li and Arnd Christian König. Theory and applications of b-bit minwise hashing. Communications ofthe ACM, 54(8):101–109, 2011. doi: 10.1145/1978542.1978566.

[10] Ping Li and Christian König. Accurate estimators for improving minwise hashing and b-bit minwise hashing. arXiv:1108.0895, Aug. 2011. Submitted 3 Aug. 2011. DOI: 10.48550/arXiv.1108.0895.

[11] Ping Li, Kenneth W. Church, and Trevor J. Hastie. Conditional random sampling: A sketchbased sampling technique for sparse data. In Advances in Neural Information Processing Systems 19, pages 873–880, 2006. URL https://proceedings.neurips.cc/paper/2006/hash/ aa6b7ad9d68bf3443c35d23de844463b-Abstract.html.

[12] Ping Li, Art B. Owen, and Cun-Hui Zhang. One permutation hashing. In Advances in Neural Information Processing Systems 25, pages 3122–3130, 2012. URL https://proceedings.neurips.cc/paper\_files/paper/2012/hash/ eaa32c96f620053cf442ad32258076b9-Abstract.html.

[13] Ping Li, Xiaoyun Li, and Gennady Samorodnitsky. P-MinHash algorithm for continuous probability measures: Theory and application to machine learning. In Proceedings ofthe 31st ACM International Conference on Information & Knowledge Management, pages 1177–1187. Association for Computing Machinery, 2022. doi: 10.1145/3511808.3557413.

[14] Xiaoyun Li and Ping Li. C-OPH: Improving the accuracy of one permutation hashing (OPH) with circulant permutations. arXiv preprint arXiv:2111.09544, 2021. URL https://arxiv. org/abs/2111.09544.

[15] Xiaoyun Li and Ping Li. C-MinHash: Improving minwise hashing with circulant permutation. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pages 12857–12887. PMLR, 2022. URL https: //proceedings.mlr.press/v162/li22m.html.

[16] Brian D. Ondov, Todd J. Treangen, Páll Melsted, Adam B. Mallonee, Nicholas H. Bergman, Sergey Koren, and Adam M. Phillippy. Mash: Fast genome and metagenome distance estimation using MinHash. Genome Biology, 17(1):132, 2016. doi: 10.1186/s13059-016-0997-x.

[17] Mahmudur Rahman Hera and David Koslicki. Estimating similarity and distance using FracMin-Hash. Algorithmsfor Molecular Biology, 20(1):8, 2025. doi: 10.1186/s13015-025-00276-8.

[18] Anshumali Shrivastava. Optimal densification for fast and accurate minwise hashing. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 3154–3163. PMLR, 2017. URL https: //proceedings.mlr.press/v70/shrivastava17a.html.

[19] Anshumali Shrivastava and Ping Li. Densifying one permutation hashing via rotation for fast near neighbor search. In Proceedings ofthe 31st International Conference on Machine Learning, volume 32 of Proceedings ofMachine Learning Research, pages 557–565. PMLR, 2014. URL https://proceedings.mlr.press/v32/shrivastava14.html.

[20] Tom Tseng, Laxman Dhulipala, and Julian Shun. Parallel index-based structural graph clustering and its approximation. In Proceedings ofthe 2021 International Conference on Management of Data, pages 1851–1864. Association for Computing Machinery, 2021. doi: 10.1145/3448016. 3457278.

## A Appendix: CMinHash and Alternative Proofs

## A.1 The CMinHash $( \sigma , \pi )$ algorithm

CMinHash $( \sigma , \pi )$ is simple to implement, as it just requires a small tweak from Broder’s original MinHash. We place Algorithm 1 and Algorithm 2 side by side for comparison. The pre-processing stage for CMinHash (and by extension, algorithms using circulant permutation) consists of lines 1-4 in Algorithm 2, which replaces the independent permutation within the loop (e.g. Line 6 in Algorithm 1).

Algorithm 1: Broder’s MinHash Algorithm 2: CMinHash(σ, π)   
Input: $\overline { { S _ { 1 } , S _ { 2 } \subseteq \{ 1 , 2 , \dots , D \} } }$ Input: $\overline { { S _ { 1 } , S _ { 2 } \subseteq \{ 1 , 2 , \dots , D \} } }$   
$\theta ^ { \mathrm { ( M H ) } } = 0$ $\theta ^ { \mathrm { ( C M H ) } } = 0$   
Output: $\theta ^ { \mathrm { ( M H ) } }$ Output: $\theta ^ { \mathrm { ( C M H ) } }$   
1 1 Generate permutation $\sigma : [ D ]  [ D ]$   
2 2 $S _ { 1 }  \sigma ( \bar { S } _ { 1 } )$   
3 3 $S _ { 2 }  \sigma ( S _ { 2 } )$   
4 4 Generate permutation $\pi : [ D ] \to [ D ]$   
5 for $t = 1$ to $K$ do 5 for $t = 1$ to K do   
6 Generate permutation $\pi _ { t } : [ D ] \to [ D ]$ 6   
7 $\theta _ { t } \gets \mathbb { 1 } _ { \{ \operatorname* { m i n } \{ \pi _ { t } ( S _ { 1 } ) \} = \operatorname* { m i n } \{ \pi _ { t } ( S _ { 2 } ) \} \} }$ 7 $\theta _ { t } \gets \mathbb { 1 } _ { \{ \operatorname* { m i n } \{ \pi _ {  t } ( S _ { 1 } ) \} = \operatorname* { m i n } \{ \pi _ {  t } ( S _ { 2 } ) \} \} }$   
8 $\boldsymbol { \theta } ^ { ( \mathrm { M H } ) } + = \boldsymbol { \dot { \theta } } _ { t }$ 8 $\partial ^ { ( \mathrm { C M H } ) } + = \dot { \theta } _ { t }$   
9 $\theta ^ { \mathrm { ( M H ) } }  \theta ^ { \mathrm { ( M H ) } } / K$ 9 $\theta ^ { \mathrm { ( C M H ) } }  \theta ^ { \mathrm { ( C M H ) } } / K$

As both E $\left( \boldsymbol { \theta } ^ { \mathrm { ( M H ) } } \right)$ and E $\left( \theta ^ { \mathrm { ( C M H ) } } \right)$ are unbiased estimators of J due to linearity of expectation, the main result of [15] is to show that Var $\left( \theta ^ { \mathrm { ( C M H ) } } \right) < \mathrm { V a r } \left( \theta ^ { \mathrm { ( M H ) } } \right)$ for $K < D$ (always satisfied, since if $K > D$ , computing J directly is more efficient), provided $0 \overset { \prime } { < } J < 1$

This is a surprising and impactful result, since K in Broder’s MinHash is the number of independent permutations, while K in CMinHash $( \sigma , \pi )$ is the number of circular shifts of one permutation. This means that the two independent permutations $\sigma , \pi$ can lead to a lower variance compared to K independent permutations, and storing these two independent permutations in memory is cheaper than storing K independent permutations. Conventional thought assumes that because the circulant shift is used for one permutation π, the variance necessarily needs to be higher.

However, we have not seen many papers building on the work of [15] (e.g. using circulant permutations for other MinHash type algorithms), and we hypothesize this is due to the technical difficulty of the approaches in the paper, e.g. deriving a variance theorem equivalent to Theorem 3.1 in [15], and then showing that the theoretical variance is indeed lower than other benchmark estimates.

## A.2 Lemmas, Theorems and Propositions in the CMinHash $( \sigma , \pi )$ paper

We give the original lemmas, theorems and propositions used in the seminal CMinHash paper, since we refer to them in the main text, and only modify them slightly to fit our notation.

We start with the following definition.

Definition 4. Definition 2.2 in [15]

For $A , B \in \{ \widetilde { O } , \widetilde { \times } , \widetilde { = } \}$ , let $\{ ( A , B ) \mid \Delta \}$ be the set $\{ ( i , j ) : ( x _ { i } , x _ { j } ) = ( A , B ) , j - i = \Delta \}$ . For $1 \leq \Delta \leq K { \stackrel { \cdot } { - } } 1 , d e f i n e$

$$
\mathcal { L } _ { 0 } ( \Delta ) = \{ ( \widetilde O , \widetilde O ) | \Delta \} , \quad \mathcal { L } _ { 1 } ( \Delta ) = \{ ( \widetilde O , \widetilde \times ) | \Delta \} , \quad \mathcal { L } _ { 2 } ( \Delta ) = \{ ( \widetilde O , \widetilde \cdot ) | \Delta \} ,
$$

$$
\mathcal { G } _ { 0 } ( \Delta ) = \{ ( \widetilde { - } , \widetilde { O } ) | \Delta \} , ~ \mathcal { G } _ { 1 } ( \Delta ) = \{ ( \widetilde { - } , \widetilde { \times } ) | \Delta \} , ~ \mathcal { G } _ { 2 } ( \Delta ) = \{ ( \widetilde { - } , \widetilde { - } ) | \Delta \}
$$

$$
\mathcal { H } _ { 0 } ( \Delta ) = \{ ( \widetilde { \times } , \widetilde { O } ) | \Delta \} , \quad \mathcal { H } _ { 1 } ( \Delta ) = \{ ( \widetilde { \times } , \widetilde { \times } ) | \Delta \} , \quad \mathcal { H } _ { 2 } ( \Delta ) = \{ ( \widetilde { \times } , \widetilde { - } ) | \Delta \} .
$$

with thefollowing constraintsfor all $1 \le \Delta \le K - 1$

$$
| \mathcal { L } _ { 0 } | + | \mathcal { L } _ { 1 } | + | \mathcal { L } _ { 2 } | = | \mathcal { L } _ { 0 } | + | \mathcal { G } _ { 0 } | + | \mathcal { H } _ { 0 } | = a ,
$$

$$
| \mathcal { G } _ { 0 } | + | \mathcal { G } _ { 1 } | + | \mathcal { G } _ { 2 } | = | \mathcal { L } _ { 2 } | + | \mathcal { G } _ { 2 } | + | \mathcal { H } _ { 2 } | = D - f ,
$$

$$
\left| \mathcal { H } _ { 0 } \right| + \left| \mathcal { H } _ { 1 } \right| + \left| \mathcal { H } _ { 2 } \right| = \left| \mathcal { L } _ { 1 } \right| + \left| \mathcal { G } _ { 1 } \right| + \left| \mathcal { H } _ { 1 } \right| = f - a .
$$

With this definition, [15] gave an expression for Var $\left( \theta ^ { \mathrm { ( C M H ) } } \right)$ .

Theorem 5. Theorem 3.1 in [15] (as stated)

Suppose $0 < a < f \leq D$ and $J \not \in \{ 0 , 1 \}$ . Then

$$
\mathrm { V a r } \left( \theta ^ { ( \mathrm { C M H } ) } \right) = \frac { J } { K } + \frac { ( K - 1 ) \widetilde { \mathcal { E } } } { K } - J ^ { 2 } ,
$$

where $l = \operatorname* { m a x } ( 0 , D - 2 f + a )$ and

$$
\begin{array} { l } { \displaystyle \widetilde { \mathcal { E } } = \sum _ { \Xi } \left( \frac { l _ { 0 } } { f + g _ { 0 } + g _ { 1 } } + \frac { a ( g _ { 0 } + l _ { 2 } ) } { ( f + g _ { 0 } + g _ { 1 } ) f } \right) \left( \sum _ { s = l } ^ { D - f - 1 } \frac { \left( \begin{array} { c } { D - f } \\ { s } \end{array} \right) \left( \begin{array} { c } { f - a - 1 } \\ { D - f - s - 1 } \end{array} \right) } { \left( \begin{array} { c } { D - a - 1 } \\ { D - f - 1 } \end{array} \right) } \right. } \\ { \displaystyle \left. \times \frac { \left( \begin{array} { c } { s } \\ { n _ { 1 } } \end{array} \right) \left( \begin{array} { c } { D - f - s } \\ { n _ { 2 } } \end{array} \right) \left( \begin{array} { c } { D - f - s } \\ { n _ { 3 } } \end{array} \right) \left( \begin{array} { c } { f - a - ( D - f - s ) } \\ { n _ { 4 } } \end{array} \right) \left( \begin{array} { c } { a - 1 } \\ { a - l _ { 1 } - l _ { 2 } } \end{array} \right) } { \left( \begin{array} { c } { D - 1 } \\ { a } \end{array} \right) } \right) , } \end{array}
$$

with thefeasible set $\Xi = \{ l _ { 0 } , l _ { 2 } , g _ { 0 } , g _ { 1 } \}$ satisfies the constraints in Definition 4 and

$$
n _ { 1 } = g _ { 0 } - ( D - f - s - g _ { 1 } ) ,
$$

$$
n _ { 2 } = D - f - s - g _ { 1 } ,
$$

$$
n _ { 3 } = l _ { 2 } - g _ { 0 } + ( D - f - s - g _ { 1 } ) ,
$$

$$
n _ { 4 } = l _ { 1 } - ( D - f - s - g _ { 1 } ) .
$$

The proof of this theorem can be found in [15].

Remark: During the rebuttal period of NeurIPS 2026, we wanted to further convince reviewers that our approach for the Minner estimator (Section 3) was still simpler to understand, compared to the combinatorial expression we would get, following Li’s approach. We never were able to replicate Li’s approach for the Minner estimator, and we thought this was because we did not understand enough combinatorics.

Since ChatGPT 5.6 SOL was released, we asked it to apply Li’s approach to the Minner estimator, in order to craft a better rebuttal for reviewers. Unfortunately, SOL found an error in Theorem 3.1, invalidating their approach.

The following italics is output (slightly modified) from SOL, and the authors take full responsibility for it.

In the proofofTheorem 3.1, it is stated that $\mathbb { E } _ { \sigma , \pi } \left( \mathbb { 1 } _ { s } \mathbb { 1 } _ { s + \Delta } \right)$ is the samefor all $1 \le \Delta \le K - 1$ because σ is random. This does not hold in general. For example, take $D = 4 , K = 3 , S _ { 1 } =$ $\{ 1 , 2 \} , S _ { 2 } = \{ 1 \}$ . Then $a = 1 , f = 2$ and $J = 1 / 2 .$ . After fixing the position of ${ \widetilde { O } } ,$ the three possible location vectors are $( \tilde { O } , \tilde { \times } , \tilde { - } , \tilde { - } ) , ( \widetilde { O } , \tilde { \bar { - } } , \tilde { \times } , \tilde { - } )$ and $( \boldsymbol { \tilde { O } } , \boldsymbol { \tilde { \overline { { \jmath } } } } , \boldsymbol { \tilde { \overline { { \jmath } } } } , \boldsymbol { \tilde { \times } } )$ . We compute the expectation and variance by first principles. By Lemma 1,

$$
\mathbb { E } _ { \sigma , \pi } \left( \mathbb { 1 } _ { 1 } \mathbb { 1 } _ { 2 } \right) = \frac { 1 } { 3 } \left( \frac { 1 } { 6 } + \frac { 1 } { 4 } + \frac { 1 } { 6 } \right) = \frac { 7 } { 3 6 } ,
$$

$$
\mathbb { E } _ { \sigma , \pi } \left( \mathbb { 1 } _ { 1 } \mathbb { 1 } _ { 3 } \right) = \frac { 1 } { 3 } \left( \frac { 1 } { 4 } + 0 + \frac { 1 } { 4 } \right) = \frac { 1 } { 6 } ,
$$

The variance is hence

$$
\begin{array} { l } { \displaystyle \mathrm { V a r } _ { \sigma , \pi } \big ( \theta ^ { \mathrm { ( C M H ) } } \big ) = \frac { J } { 3 } + \frac { 2 } { 9 } \left( 2 \mathbb { E } _ { \sigma , \pi } \left( \mathbb { 1 } _ { 1 } \mathbb { 1 } _ { 2 } \right) + \mathbb { E } _ { \sigma , \pi } \left( \mathbb { 1 } _ { 1 } \mathbb { 1 } _ { 3 } \right) \right) - J ^ { 2 } } \\ { \displaystyle \quad = \frac { 1 3 } { 3 2 4 } . } \end{array}
$$

On the other hand, applying Theorem 3.1 gives $\operatorname { V a r } _ { \sigma , \pi } ( \theta ^ { \left( \mathrm { C M H } \right) } ) = 5 / 1 0 8 .$

In contrast, the approach in our paper considers every value of $\Delta$ . To be precise, the derivation in the proof of Theorem 3.1 uses $\Delta = 1 .$ , but the variance also involves the corresponding expectations for $\bar { 2 } \le \Delta \le K - 1$ . Even apart from this point, the variance is expressed as a combinatorial sum under several constraints, which makes it difficult to compare theoretically against the variance given by other MinHash algorithms. Moreover, the current approach taken to derive the result in Theorem 3.1 would need to be replicated for other MinHash type algorithms using circulant permutation, which is not straightforward.

In fact, in order for [15] to show that Var $\left( \theta ^ { \mathrm { ( C M H ) } } \right) \ < \ \mathrm { V a r } \left( \theta ^ { \mathrm { ( M H ) } } \right)$ , they state and prove the following lemma and theorem.

Lemma 4. Lemma 3.3 (Strict Increment) in [15]

Let $f > a > 0$ and K be arbitrary and fixed. Denote $\widetilde { \mathcal { E } } _ { D }$ as in Theorem $3 . I ,$ with D as a parameter. Then $\tilde { \mathcal { E } } _ { D + 1 } > \tilde { \mathcal { E } } _ { D }$ for all $D \geq f .$

Theorem 6. Theorem 3.4 (Uniform Superiority) in [15] (rephrased slightly)

For any two sets $s _ { 1 } , s _ { 2 } ,$ with $J \ne 0$ or $J \ne 1$ , and $K > 1$ , it holds that Var $\left( \theta ^ { \mathrm { ( C M H ) } } \right) ~ <$ Var $\left( \theta ^ { \mathrm { ( M H ) } } \right)$

Finally, the CMinHash $\iota ( \sigma , \pi )$ estimator has some nice properties, which [15] gives in the following two propositions.

Proposition 1. Proposition 3.2 (Symmetry) in [15] $\mathrm { V a r } \mathbf { \bar { \Gamma } } \big ( \theta ^ { \mathrm { ( C M H ) } } \big )$  is the same for the $( D , f , a )$ data pair and the $( D , f , f - a )$ data pair, for $0 \leq a \leq$ $f \leq \dot { \cal D }$

Proposition 2. Proposition 3.6 (Consistent Improvement) in [15]

Suppose f isfixed. In terms ofa, the variance ratio $\frac { \mathrm { V a r } \big ( \theta ^ { ( \mathrm { M H } ) } \big ) } { \mathrm { V a r } \big ( \theta ^ { ( \mathrm { C M H } ) } \big ) }$ is a constantfor any $0 < a < f .$

Contribution Two of our paper presents an alternative approach to Theorem 3.1, giving a very accurate approximation to Var $\left( \boldsymbol { \theta } ^ { \mathrm { \bar { ( } C M H ) } } \right)$ , which can be replicated across different MinHash type algorithms which we discuss in Section 3 and Section 4 (our Contribution 3).

Moreover, the exact covariance identity in (16) makes Theorem 3.4, Proposition 3.2, and Proposition 3.6 immediate, while the approximations avoid the constrained combinatorial sum in Theorem 3.1. We discuss this in Section 2.

## A.3 A Simpler Approach to Lemma 2.4

We give a simpler approach for the following lemma.

Lemma 1. Lemma 2.4 in [15]. For any $1 \leq s < t \leq K$ , with $t - s = \Delta$ , then $\mathbb { E } _ { \pi } \left( \mathbb { 1 } _ { s } \mathbb { 1 } _ { s + \Delta } \right) =$   
$| { \mathcal { L } } _ { 0 } ( \Delta ) | + ( | { \mathcal { G } } _ { 0 } ( \Delta ) | + | { \mathcal { L } } _ { 2 } ( \Delta ) | ) J$ f+|G<sub>0</sub>(∆)|+|G<sub>1</sub>(∆)|

Before we embark on a proof, we give intuition on how we prove this lemma, which helps if we want to generalize this approach beyond MinHash.

Following the sampling without replacement paradigm where a random permutation can be thought of as the “draw order", then for a fixed ∆, we can imagine drawing the indices $( x _ { s } , x _ { s + \Delta } )$ in some random order. The first entry in the tuple $( A , B ) = ( x _ { s } , x _ { s + \Delta } )$ then corresponds to what is seen sequentially when π is shifted s times to the right, and the second entry corresponds to what is seen when π is shifted $s + \Delta$ times to the right.

For example, take $D = K = 5 , s = 1$ and $\Delta = 4$ . After applying $\sigma ,$ suppose that the location vector is $\vec { x } = ( \widetilde { - } , \widetilde { - } , \widetilde { O } , \widetilde { \times } , \widetilde { O } )$ . Suppose one random permutation π draws the indices in the order $1 , 2 , \ldots , 5$ under π<sub>→</sub> . Shifting this same permutation rightwards by another 4 units gives $\pi _ {  s + \Delta }$ under which the indices are drawn in the order 5, 1, 2, 3, 4. We take the first entry at the index drawn under $\pi _ {  s }$ and the second entry at the index 4 units to its right, returning to index 1 after index 5. Thus Draw 1 gives $( x _ { 1 } , x _ { 5 } )$ , Draw 2 gives $( x _ { 2 } , x _ { 1 } )$ , and the remaining draws follow the same order. Table 1 gives all five tuples. We bold the first $\widetilde O$ in each entry.

The first $\widetilde O$ in the first entry appears at Draw 3, before the first ×e at Draw 4. The first $\widetilde O$ in the second entry appears at Draw 1, before the first $\begin{array} { l } { \widetilde { \times } } \end{array}$ at Draw 5. Hence $\mathbb { 1 } _ { s } = \mathbb { 1 } _ { s + \Delta } = 1 , \mathrm { s o } \ \mathbb { 1 } _ { s } \mathbb { 1 } _ { s + \Delta } = 1$ for this π. The first $\widetilde O$ occurs at different draws for the two entries.

<table><tr><td>Draw</td><td>First entry  $\mathrm { u n d e r } \pi _ {  s }$ </td><td>Second entry  $\mathrm { u n d e r } \pi _ {  s + \Delta }$ </td><td>Tuple</td></tr><tr><td>Draw 1</td><td>二</td><td>0</td><td>(−,0)</td></tr><tr><td>Draw 2</td><td>二</td><td>二</td><td>(−, 二)</td></tr><tr><td>Draw 3</td><td>õ</td><td>21</td><td>(, −)</td></tr><tr><td>Draw 4</td><td>x</td><td>õ</td><td>(x,ð)</td></tr><tr><td>Draw 5</td><td> $\widetilde O$ </td><td>x</td><td>(õ, x)</td></tr></table>

Table 1: Five successive tuples from one random permutation, for $D = 5 , s = 1$ and $\Delta = 4 .$ . The first entry corresponds to $\pi _ {  s }$ and the second entry to $\pi _ {  s + \Delta }$

Each draw gives the next tuple from this same random permutation. The subscripts s and $s + \Delta$ in the two indicators denote the two circulant shifts. Each Example column in Tables 2 and 3 shows a sequence of tuple draws from one random permutation.

Since a $\widetilde O$ in the location vector means that the hashed values collide, then $\mathbb { E } _ { \pi } \left( \mathbb { I } _ { s } \mathbb { I } _ { s + \Delta } \right) > 0$ means that both min $\{ \pi _ {  s } ( S _ { 1 } ) \} = \operatorname* { m i n } \{ \pi _ {  s } ( S _ { 2 } ) \}$ and min $\{ \pi _ {  s + \Delta } ( S _ { 1 } ) \} = \operatorname* { m i n } \{ \pi _ {  s + \Delta } ( S _ { 2 } ) \}$ } must hold.

Hence, $\mathbb { E } _ { \pi } \left( \mathbb { I } _ { s } \mathbb { I } _ { s + \Delta } \right) > 0$ can be thought of as the probability of a sequence of draws of the nine different types of $D$ tuples where the first $\widetilde O$ appears before the first $\begin{array} { l } { \widetilde { \times } } \end{array}$ in both entries of the tuple, as the random permutation π is a sequence of draws without replacement. We call such a sequence of draws a success, otherwise a failure. Five examples of successes and failures are in Table 2. We bold the first $\widetilde O$ which occurs.

<table><tr><td></td><td>Example 1</td><td>Example 2</td><td>Example 3</td><td>Example 4</td><td>Example 5</td></tr><tr><td>Draw 1</td><td>(−, −)</td><td>(−, −)</td><td> $( \widetilde { \mathbf { O } } , \widetilde { \mathbf { \Lambda } } ^ { \sim } )$ </td><td>(0,−)</td><td> $( \widetilde { \mathbf { O } } , \widetilde { \mathbf { \Lambda } } ^ { \sim } )$ </td></tr><tr><td>Draw 2</td><td>(−,ð)</td><td>(−,ð)</td><td>(x, x)</td><td>(, −)</td><td>(=, 二)</td></tr><tr><td>Draw 3</td><td>(, −)</td><td>(=, x)</td><td></td><td>(,~)</td><td>(=, 二)</td></tr><tr><td>Draw 4</td><td></td><td>(, )</td><td></td><td>(x,õ)</td><td>(=,õ)</td></tr></table>

Table 2: Five examples of sequences of draws. Each example is a vertical column.

Example 1 and Example 2 are functionally the same (and are both successes). While Example 2 has a $\widetilde { \times }$ in the second entry at Draw 3, it does not matter, because the first $\widetilde O$ in the second entry was already drawn in Draw 2.

Example 3 is an example of a failure, as a $\begin{array} { l } { \sim } \\ { \times } \end{array}$ appears before an $\widetilde O$ in the second entry at Draw 2.

Example 4 and Example 5 are also functionally the same. We also do not worry if an $\begin{array} { l } { \widetilde { \times } } \end{array}$ appears next to an $\widetilde O$ (as in Draw 4) for Example 4.

In fact, since it does not matter if a $\begin{array} { l } { \widetilde { \times } } \end{array}$ appears after an ${ \cal \widetilde O } ,$ we do not need to worry about elements appearing after the first ${ \cal \widetilde O } ,$ , and hence can view our examples as in Table 3.

By treating the tuples $( A , B )$ as sampling without replacement, and computing the appropriate probability of success, we can now prove Lemma 1 easily.

Proof. Similar to Li’s approach, we consider three cases, and condition on $\mathbb { 1 } _ { s } .$ . More specifically, we distinguish cases where the success for the first entry of the tuple happens when we draw

<table><tr><td></td><td>Example 1</td><td>Example 2</td><td>Example 3</td><td>Example 4</td><td>Example 5</td></tr><tr><td>Draw 1</td><td> $\overline { { ( \widetilde { \mathbf { \Gamma } } - , \widetilde { \mathbf { \Gamma } } ) } }$ </td><td> $\overline { { ( \widetilde { \mathbf { \Gamma } } - , \widetilde { \mathbf { \Gamma } } ) } }$ </td><td> $\overline { { ( \widetilde { \mathbf { O } } , \widetilde { \mathbf { \Lambda } } ) } }$ </td><td> $\overline { { ( \widetilde { \mathbf { O } } , \widetilde { \mathbf { \Lambda } } ) } }$ </td><td> $\widehat { ( \mathbf { O } , \overline { { \mathbf { \Lambda } } } ) }$ </td></tr><tr><td>Draw 2</td><td> $( \widetilde { \mathbf { - } } , \widetilde { \mathbf { O } } )$ </td><td> $( \widetilde { \mathbf { - } } , \widetilde { \mathbf { O } } )$ </td><td> $( \mathrm { ~  ~ \xi ~ } , \widetilde { \times } )$ </td><td> $( \quad , \widetilde { - } )$ </td><td>(, 二)</td></tr><tr><td>Draw 3</td><td> $( \widetilde { \mathbf { O } } , \mathbf { \Lambda } )$ </td><td>(=, )</td><td></td><td> $( \quad , \widetilde { - } )$ </td><td>(, 二)</td></tr><tr><td>Draw 4</td><td></td><td> $( \widetilde { \mathbf { O } } , \mathbf { \Lambda } )$ </td><td></td><td>(,ō)</td><td> $( \mathrm { ~ \bf ~ \epsilon ~ } , \tilde { \bf O } )$ </td></tr></table>

Table 3: Five examples of sequences of draws, where we do not record the type of entry after the first $\widetilde O$ is drawn. Each example is a vertical column.

$\{ ( \widetilde { O } , \widetilde { \times } ) , ( \widetilde { O } , \widetilde { O } ) , ( \widetilde { O } , \widetilde { \overline { { { - } } } } ) \}$ (before any $\begin{array} { l } { \widetilde { \times } } \end{array}$ for the first entry), and consider the probability for a success where where the first $\widetilde O$ is drawn before a $\begin{array} { l } { \widetilde { \times } } \end{array}$ for the second entry of the tuple. Table 4 gives a pictorial intuition of the five cases.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Case 1</td><td rowspan=1 colspan=1>Case 2a</td><td rowspan=1 colspan=1>Case 2b</td><td rowspan=1 colspan=1>Case 3a</td><td rowspan=1 colspan=1>Case 3b</td></tr><tr><td rowspan=1 colspan=1>Draw t</td><td rowspan=1 colspan=1>(−,ō):(=, ):(ō, x)</td><td rowspan=1 colspan=1>:(−,二)：::(õ,õ)</td><td rowspan=1 colspan=1>·..(=,õ)(=, )•(õ,õ)</td><td rowspan=1 colspan=1>· $( \widetilde { \mathbf { - } } , \widetilde { \mathbf { O } } )$ ·(ō, 二)</td><td rowspan=1 colspan=1>:(ō, 二)：(, 二)</td></tr></table>

Table 4: Pictorial intuition of the five (sub)cases of successes we consider, as we condition on the first $\widetilde O$ seen in the first entry of the tuple. A bold Oe denotes the first $\widetilde O$ seen for that entry. A blank entry in a tuple means it does not matter what that entry is since an $\widetilde O$ was drawn before.

Case One and Case Two follow along similar lines in their proof, but Case Three is vastly simplified.   
For each case, we show the product of probabilities “sequentially" in our reasoning.

We remind the reader that tuples which do not affect the hash colliding are not involved in the below probability calculations. As an analogy, consider a bag with one blue marble, two red marbles, and N green marbles. The probability of drawing a blue marble before a red marble is always $\textstyle { \frac { 1 } { 3 } }$ , regardless of N.

Case One: The first $\widetilde O$ for the first entry of the tuple is seen at $( \widetilde O , \widetilde { \times } )$ . A success for the second entry of the tuple can only happen before $( \widetilde O , \widetilde { \times } )$ , hence we need the probability of seeing the first $\widetilde O$ appearing before any $\begin{array} { l } { \sim } \\ { \times } \end{array}$ for the second entry of the tuple. This is the product of the probabilities of i) drawing a $( \widetilde { - } , \widetilde { O } )$ before any invalid tuples in $\{ ( \widetilde { \mathbf { \Gamma } } , \widetilde { \mathbf { \times } } ) , ( \widetilde { \mathbf { \times } } , \widetilde { O } ) , ( \widetilde { \mathbf { \times } } , \widetilde { \mathbf { \Gamma } } ) , ( \widetilde { \mathbf { \times } } , \widetilde { \mathbf { \times } } ) , ( \widetilde { \mathbf { \times } } , \widetilde { \mathbf { \times } } ) , ( \widetilde { O } , \widetilde { O } ) , ( \widetilde { O } , \widetilde { \mathbf { \sim } } ) , ( \widetilde { O } , \widetilde { \mathbf { \times } } ) \}$ , and ii) drawing a $( \bar { \tilde { O } } , \tilde { \times } )$ before any invalid tuples in $\{ ( \widetilde { \times } , \widetilde { O } ) , ( \widetilde { \times } , \widetilde { - } ) , ( \widetilde { \times } , \widetilde { \times } ) , ( \widetilde { O } , \widetilde { O } ) , ( \widetilde { O } , \widetilde { - } ) \}$ , which is given by

$$
p _ { 1 } = \frac { | \mathcal { G } _ { 0 } | } { f + | \mathcal { G } _ { 0 } | + | \mathcal { G } _ { 1 } | } \frac { | \mathcal { L } _ { 1 } | } { f } = \frac { | \mathcal { G } _ { 0 } | | \mathcal { L } _ { 1 } | } { ( f + | \mathcal { G } _ { 0 } | + | \mathcal { G } _ { 1 } | ) f } .\tag{2}
$$

Case Two: The first $\widetilde O$ for the first entry of the tuple is seen at $( \widetilde { O } , \widetilde { O } )$ . A success is always guaranteed once $( \widetilde { O } , \widetilde { O } )$ is drawn, provided there are no invalid $\begin{array} { l } { \widetilde { \times } } \end{array}$ before an $\widetilde O$ for the second entry of the tuple. Hence there are two possible disjoint sequences ending with $( \widetilde { O } , \widetilde { O } )$ that are successes.

a) A sequence where no $( \widetilde { - } , \widetilde { \times } )$ is seen before $( \widetilde O , \widetilde O )$ , where we draw $( \widetilde O , \widetilde O )$ before invalid tuples $\{ ( \widetilde { - } , \widetilde { \times } ) , ( \widetilde { \times } , \widetilde { O } ) , ( \widetilde { \times } , \widetilde { - } ) , ( \widetilde { \times } , \widetilde { \times } ) , ( \widetilde { O } , \widetilde { - } ) , ( \widetilde { O } , \widetilde { \times } ) \}$ . The probability is given by

$$
p _ { 2 } = \frac { | { \mathcal { L } } _ { 0 } | } { f + | { \mathcal { G } } _ { 1 } | } .\tag{3}
$$

b) A sequence where at least one $( \widetilde { - } , \widetilde { \times } )$ is seen before $( \widetilde O , \widetilde O )$ . This means a $( \widetilde { - } , \widetilde { O } )$ must be seen before it. This is the product of the probabilities of i) first drawing $( \widetilde { - } , \widetilde { O } )$ before invalid tuples in $\{ ( \widetilde { \mathbf { \Gamma } } , \widetilde { \mathbf { \times } } ) , ( \widetilde { \mathbf { \times } } , \widetilde { \mathbf { \cal { O } } } ) , ( \widetilde { \mathbf { \times } } , \widetilde { \mathbf { \Gamma } } ) , ( \widetilde { \mathbf { \times } } , \widetilde { \mathbf { \times } } ) , ( \widetilde { \mathbf { \cal { O } } } , \widetilde { \mathbf { \cal { O } } } ) , ( \widetilde { \mathbf { \cal { O } } } , \widetilde { \mathbf { \Gamma } } ) , ( \widetilde { \mathbf { \cal { O } } } , \widetilde { \mathbf { \Gamma } } ) \}$ , ii) next drawing $( \widetilde { - } , \widetilde { \times } )$ before invalid tuples in $\{ ( \widetilde { \times } , \widetilde { O } ) , ( \widetilde { \times } , \widetilde { - } ) , ( \widetilde { \times } , \widetilde { \times } ) , ( \widetilde { O } , \widetilde { O } ) , ( \widetilde { O } , \widetilde { - } ) , ( \widetilde { O } , \widetilde { \times } ) \}$ , and iii) finally drawing $( \widetilde { O } , \widetilde { O } )$ before invalid tuples in $\{ ( \widetilde { \times } , \widetilde { O } ) , ( \widetilde { \times } , \widetilde { - } ) , ( \widetilde { \times } , \widetilde { \times } ) , ( \widetilde { O } , \widetilde { - } ) , ( \widetilde { O } , \widetilde { \times } ) \}$ , which is given by

$$
p _ { 3 } = \frac { | \mathcal { G } _ { 0 } | } { f + | \mathcal { G } _ { 0 } | + | \mathcal { G } _ { 1 } | } \frac { | \mathcal { G } _ { 1 } | } { f + | \mathcal { G } _ { 1 } | } \frac { | \mathcal { L } _ { 0 } | } { f } = \frac { | \mathcal { G } _ { 0 } | | \mathcal { G } _ { 1 } | | \mathcal { L } _ { 0 } | } { ( f + | \mathcal { G } _ { 0 } | + | \mathcal { G } _ { 1 } | ) ( f + | \mathcal { G } _ { 1 } | ) f } .\tag{4}
$$

Case Three: The first $\widetilde O$ for the first entry of the tuple is seen at $( \widetilde O , \widetilde { \overline { { - } } } )$ . Then a success occurs when either the first $\widetilde O$ for the second entry of the tuple appears before the first $( \widetilde O , \widetilde { \overline { { - } } } )$ , or it appears after, together with no $\begin{array} { l } { \widetilde { \times } } \end{array}$ appearing before the first $\widetilde O$ for the first entry of the tuple. Hence there are two possible disjoint sequences which are successes.

a) A sequence which ends at $( \widetilde O , \widetilde { \overline { { \mathbf { \Lambda } } } } )$ , which means an $( \widetilde { - } , \widetilde { O } )$ must be seen before it. This is the product of probabilities of i) first drawing $( \widetilde { - } , \widetilde { O } )$ before any invalid tuples $\{ ( \widetilde { \mathbf { \Gamma } } , \widetilde { \mathbf { \times } } ) , ( \widetilde { \mathbf { \times } } , \widetilde { O } ) , ( \widetilde { \mathbf { \times } } , \widetilde { \mathbf { - } } ) , ( \widetilde { \mathbf { \times } } , \widetilde { \mathbf { \times } } ) , ( \widetilde { O } , \widetilde { O } ) , ( \widetilde { O } , \widetilde { \mathbf { - } } ) , ( \widetilde { O } , \widetilde { \mathbf { \times } } ) \}$ , and ii) drawing $( \widetilde O , \widetilde { \overline { { - } } } )$ before invalid tuples in $\{ ( \widetilde { \times } , \widetilde { \stackrel { \sim } { - } } ) , ( \widetilde { \times } , \widetilde { O } ) , ( \widetilde { \times } , \widetilde { \times } ) , ( \widetilde { O } , \widetilde { O } ) , ( \widetilde { O } , \widetilde { \times } ) \}$ , given by

$$
p _ { 4 } = \frac { | \mathcal { G } _ { 0 } | } { f + | \mathcal { G } _ { 0 } | + | \mathcal { G } _ { 1 } | } \frac { | \mathcal { L } _ { 2 } | } { f } = \frac { | \mathcal { G } _ { 0 } | | \mathcal { L } _ { 2 } | } { ( f + | \mathcal { G } _ { 0 } | + | \mathcal { G } _ { 1 } | ) f }\tag{5}
$$

b) A sequence which ends after $( \widetilde O , \widetilde { \overline { { \mathbf { \Lambda } } } } )$ , which means the first $\widetilde O$ for the second entry of the tuple appears after it. This is the product of probabilities of i) first drawing $( \widetilde O , \widetilde { \overline { { - } } } )$ before invalid tuples $\{ ( \widetilde { \mathbf { \Gamma } } , \widetilde { O } ) , ( \widetilde { \mathbf { \Gamma } } , \widetilde { \mathbf { \times } } ) , ( \hat { \widetilde { \mathbf { \times } } } , \widetilde { O } ) , ( \widetilde { \mathbf { \times } } , \widetilde { \widetilde { \mathbf { \Gamma } } } ) , ( \widetilde { \mathbf { \times } } , \widetilde { \mathbf { \times } } ) , ( \widetilde { O } , \widetilde { O } ) , ( \widetilde { O } , \widetilde { \mathbf { \times } } ) \}$ , and ii) drawing an element in $\{ ( \widetilde { O } , \widetilde { O } ) , ( { \widetilde { - } } , \widetilde { O } ) , ( \widetilde { \times } , \widetilde { O } ) \}$ before invalid tuples in $\{ ( \widetilde { O } , \widetilde { \times } ) , ( \widetilde { - } , \widetilde { \times } ) , ( \widetilde { \times } , \widetilde { \times } ) \}$ given by

$$
p _ { 5 } = { \frac { \left| { \mathcal { L } } _ { 2 } \right| } { f + \left| { \mathcal { G } } _ { 0 } \right| + \left| { \mathcal { G } } _ { 1 } \right| } } { \frac { a } { f } } = { \frac { \left| { \mathcal { L } } _ { 2 } \right| a } { \left( f + \left| { \mathcal { G } } _ { 0 } \right| + \left| { \mathcal { G } } _ { 1 } \right| \right) f } } .\tag{6}
$$

Adding (2), (3), (4), (5), (6) gives $\frac { | \mathcal { L } _ { 0 } | + ( | \mathcal { G } _ { 0 } | + | \mathcal { L } _ { 2 } | ) J } { f + | \mathcal { G } _ { 0 } | + | \mathcal { G } _ { 1 } | }$

## A.4 An Alternative Approach to Theorem 3.1

Lemma 2. Suppose we pre-process the location vector with the random permutation σ. Let $B _ { D } =$ $2 D - f - 1$ , and suppose $0 \textless J < 1$ . For all $1 \leq \Delta \leq K - 1 , i f f / D \to 0 ,$ , then

$$
\begin{array} { r l } & { \mathbb { E } _ { \sigma , \pi } \left( 1 _ { s } \mathbb { 1 } _ { s + \Delta } \right) = \left\{ \begin{array} { l l } { J ^ { 2 } - \displaystyle \frac { J ( 1 - J ) f } { 2 ( f - 1 ) ( D - 1 ) } + { \cal O } \left( \frac { f } { D ^ { 2 } } \right) , } & { D e \nu e n a n d \Delta = D / 2 , } \\ { J ^ { 2 } - \displaystyle \frac { J ( 1 - J ) f } { ( 2 f - 1 ) ( D - 1 ) } + { \cal O } \left( \frac { f } { D ^ { 2 } } \right) , } & { o t h e r w i s e , } \end{array} \right. } \end{array}
$$

$$
\begin{array} { r } { \mathrm { C o v } _ { \sigma , \pi } \left( \mathbb { 1 } _ { s } , \mathbb { 1 } _ { s + \Delta } \right) = \left\{ \begin{array} { l l } { - \displaystyle \frac { J ( 1 - J ) f } { 2 ( f - 1 ) ( D - 1 ) } + { \cal O } \left( \frac { f } { D ^ { 2 } } \right) , } & { D e \nu e n a n d \Delta = D / 2 , } \\ { - \displaystyle \frac { J ( 1 - J ) f } { ( 2 f - 1 ) ( D - 1 ) } + { \cal O } \left( \frac { f } { D ^ { 2 } } \right) , } & { o t h e r w i s e . } \end{array} \right. } \end{array}
$$

$H f / D$ is bounded awayfrom zero, then

$$
\begin{array} { c } { { \displaystyle \mathbb { E } _ { \sigma , \pi } \left( \mathbb { 1 } _ { s } \mathbb { 1 } _ { s + \Delta } \right) = J ^ { 2 } - \frac { J ( 1 - J ) } { B _ { D } } + O ( D ^ { - 2 } ) , } } \\ { { \displaystyle \mathrm { C o v } _ { \sigma , \pi } \left( \mathbb { 1 } _ { s } , \mathbb { 1 } _ { s + \Delta } \right) = - \frac { J ( 1 - J ) } { B _ { D } } + O ( D ^ { - 2 } ) . } } \end{array}
$$

Proof. Lemma 1 gives $\mathbb { E } _ { \pi } \left( \mathbb { 1 } _ { s } \mathbb { 1 } _ { s + \Delta } \right)$ . Define

$$
\begin{array} { l } { { N _ { \Delta } = \left| \mathcal { L } _ { 0 } ( \Delta ) \right| + J \left( \left| \mathcal { G } _ { 0 } ( \Delta ) \right| + \left| \mathcal { L } _ { 2 } ( \Delta ) \right| \right) , } } \\ { { M _ { \Delta } = f + \left| \mathcal { G } _ { 0 } ( \Delta ) \right| + \left| \mathcal { G } _ { 1 } ( \Delta ) \right| . } } \end{array}\tag{7}
$$

Thus

$$
\mathbb { E } _ { \pi } \left( \mathbb { 1 } _ { s } \mathbb { 1 } _ { s + \Delta } \right) = \frac { N _ { \Delta } } { M _ { \Delta } } .\tag{8}
$$

We first condition on the $f$ positions of the location vector whose entries are $\widetilde { O } \mathrm { o r } \widetilde { \times }$ . Let

$$
X _ { \Delta } = \sum _ { i = 1 } ^ { D } \mathbb { 1 } _ { \left\{ x _ { i } \neq - , ~ x _ { i + \Delta } \neq - \right\} } .\tag{9}
$$

The quantity $X _ { \Delta }$ counts the tuples $( x _ { i } , x _ { i + \Delta } )$ for which both entries are $\widetilde { O } \mathrm { o r } \widetilde { \times }$ . There are $f - X _ { \Delta }$ tuples whose first entry $\mathrm { i s ~ \overset { \sim } { - } }$ and whose second entry is $\widetilde { O } \mathrm { o r } \widetilde { \times }$ . To see this, each of the $f$ entries equal to $\widetilde { O } \mathrm { o r } \widetilde { \times }$ has one entry at distance $\Delta$ before $\mathrm { i t } ,$ and $X _ { \Delta }$ of those preceding entries are also $\widetilde O$ or $\begin{array} { l } { \sim } \\ { \times } \end{array}$ . Therefore

$$
| \mathcal { G } _ { 0 } ( \Delta ) | + | \mathcal { G } _ { 1 } ( \Delta ) | = f - X _ { \Delta } .\tag{10}
$$

Substitution of (10) into (8) gives

$$
M _ { \Delta } = 2 f - X _ { \Delta } .\tag{11}
$$

Conditional on the positions counted by $( 9 ) .$ , the a entries equal to $\widetilde O$ are placed among the $f$ positions equal to $\widetilde { O } \mathrm { o r } \widetilde { \times } \mathrm { b y }$ sampling without replacement. For a tuple counted by $X _ { \Delta }$ , the probability that both entries are Oe is $a ( a - 1 ) / ( f ( f - 1 ) )$ . Also, $f - X _ { \Delta }$ tuples go from −e to either $\widetilde O$ or $\begin{array} { l } { \bar { \widetilde { \times _ { \mathrm { ~  ~ } } } } } \end{array}$ , and another $f - X _ { \Delta }$ tuples go from either $\widetilde { O } \mathrm { o r } \widetilde { \times } \mathrm { t o } \widetilde { - }$ . In each case, the entry which is not $\underline { { \sim } }$ equals $\widetilde O$ with probability $a / f$ . Hence

$$
\mathbb { E } _ { \sigma } \left( | { \mathcal { L } } _ { 0 } ( \Delta ) | \mid X _ { \Delta } \right) = X _ { \Delta } { \frac { a ( a - 1 ) } { f ( f - 1 ) } } ,\tag{12}
$$

$$
\mathbb { E } _ { \sigma } \left( \vert \mathcal { G } _ { 0 } ( \Delta ) \vert + \vert \mathcal { L } _ { 2 } ( \Delta ) \vert \vert X _ { \Delta } \right) = 2 ( f - X _ { \Delta } ) \frac { a } { f } .\tag{13}
$$

Using $J = a / f$ in (7), (12) and (13) give

$$
\mathbb { E } _ { \sigma } \left( N _ { \Delta } \mid X _ { \Delta } \right) = J ^ { 2 } ( 2 f - X _ { \Delta } ) - \frac { J ( 1 - J ) } { f - 1 } X _ { \Delta } .\tag{14}
$$

The denominator in (11) is fixed once $X _ { \Delta }$ is fixed. Dividing (14) by (11) and then averaging over the positions gives

$$
\mathbb { E } _ { \sigma , \pi } \left( \mathbb { 1 } _ { s } \mathbb { 1 } _ { s + \Delta } \right) = J ^ { 2 } - \frac { J ( 1 - J ) } { f - 1 } \mathbb { E } _ { \sigma } \left( \frac { X _ { \Delta } } { 2 f - X _ { \Delta } } \right) ,\tag{15}
$$

$$
\operatorname { C o v } _ { \sigma , \pi } \left( \mathbb { 1 } _ { s } , \mathbb { 1 } _ { s + \Delta } \right) = - \frac { J ( 1 - J ) } { f - 1 } \mathbb { E } _ { \sigma } \left( \frac { X _ { \Delta } } { 2 f - X _ { \Delta } } \right) .\tag{16}
$$

We next approximate the expectation in (16). Write $Y _ { i } = \mathbb { 1 } _ { \{ x _ { i } \neq - , x _ { i + \Delta } \neq - \} }$ , so that $\begin{array} { r } { X _ { \Delta } = \sum _ { i } Y _ { i } } \end{array}$ by (9). For each tuple $( x _ { i } , x _ { i + \Delta } )$ , the probability that both entries are $\widetilde O$ or ×e is $f ( f - 1 ) / ( D ( D - 1 ) )$ ). Summing this probability over the D possible values of i gives

$$
\mathbb { E } _ { \sigma } \left( X _ { \Delta } \right) = D \frac { f ( f - 1 ) } { D ( D - 1 ) } = \frac { f ( f - 1 ) } { D - 1 } .\tag{17}
$$

Suppose first that $D$ is odd or $\Delta \neq D / 2$ . In the sum $\begin{array} { r } { X _ { \Delta } ( X _ { \Delta } - 1 ) = \sum _ { i \neq j } Y _ { i } Y _ { j } } \end{array}$ , there are 2D choices of $( i , j )$ for which the tuples $( x _ { i } , x _ { i + \Delta } )$ and $( x _ { j } , x _ { j + \Delta } )$ have one common position. The remaining $D ( D - 3 )$ choices use four distinct positions. If $u ^ { \underline { { \tilde { r } } } } = u ( u - 1 ) \cdot \cdot \cdot ( u - r + 1 )$ , sampling without replacement therefore gives

$$
\mathbb { E } _ { \sigma } \left( X _ { \Delta } ( X _ { \Delta } - 1 ) \right) = 2 D \frac { f ^ { \underline { { 3 } } } } { D ^ { \underline { { 3 } } } } + D ( D - 3 ) \frac { f ^ { \underline { { 4 } } } } { D ^ { \underline { { 4 } } } } = \frac { f ( f - 1 ) ^ { 2 } ( f - 2 ) } { ( D - 1 ) ( D - 2 ) } .\tag{18}
$$

For every integer $0 \leq x \leq f$

$$
{ \frac { x } { 2 f - x } } = { \frac { x } { 2 f - 1 } } + { \frac { x ( x - 1 ) } { ( 2 f - 1 ) ( 2 f - x ) } } .\tag{19}
$$

Apply (19) to $X _ { \Delta }$ , use (17) for the first term, and use $2 f - X _ { \Delta } \geq f$ together with (18) for the second term. This gives

$$
\frac { 1 } { f - 1 } \mathbb { E } _ { \sigma } \left( \frac { X _ { \Delta } } { 2 f - X _ { \Delta } } \right) = \frac { f } { ( 2 f - 1 ) ( D - 1 ) } + R _ { \Delta } ,\tag{20}
$$

$$
0 \leq R _ { \Delta } \leq \frac { ( f - 1 ) ( f - 2 ) } { ( 2 f - 1 ) ( D - 1 ) ( D - 2 ) } = O \left( \frac { f } { D ^ { 2 } } \right) .\tag{21}
$$

Now suppose that D is even and $\Delta = D / 2$ . The tuple $\left( { { x } _ { i } } , { { x } _ { i + D / 2 } } \right)$ is accompanied by the tuple $\left( { { x } _ { i + D / 2 } } , { { x } _ { i } } \right)$ , so $X _ { \Delta }$ is even. Let $Q _ { \Delta } = X _ { \Delta } / 2$ . Two different pairs of these tuples have four different positions. There are $( D / 2 ) ( D / 2 - 1 )$ choices of the first and second pair, and hence

$$
\begin{array} { r l } & { \mathbb { E } _ { \sigma } \left( X _ { \Delta } ( X _ { \Delta } - 2 ) \right) = 4 \mathbb { E } _ { \sigma } \left( Q _ { \Delta } ( Q _ { \Delta } - 1 ) \right) } \\ & { \quad \quad \quad = 4 \displaystyle \frac { D } { 2 } \left( \frac { D } { 2 } - 1 \right) \frac { f ^ { \underline { { 4 } } } } { D ^ { \underline { { 4 } } } } = \frac { f ( f - 1 ) ( f - 2 ) ( f - 3 ) } { ( D - 1 ) ( D - 3 ) } . } \end{array}\tag{22}
$$

For every even integer $0 \leq x \leq f$

$$
{ \frac { x } { 2 f - x } } = { \frac { x } { 2 ( f - 1 ) } } + { \frac { x ( x - 2 ) } { 2 ( f - 1 ) ( 2 f - x ) } } .\tag{23}
$$

Equations (17), (22), and (23) give

$$
\frac { 1 } { f - 1 } \mathbb { E } _ { \sigma } \left( \frac { X _ { \Delta } } { 2 f - X _ { \Delta } } \right) = \frac { f } { 2 ( f - 1 ) ( D - 1 ) } + R _ { D / 2 } ,\tag{24}
$$

$$
0 \leq R _ { D / 2 } \leq \frac { ( f - 2 ) ( f - 3 ) } { 2 ( f - 1 ) ( D - 1 ) ( D - 3 ) } = O \left( \frac { f } { D ^ { 2 } } \right) .\tag{25}
$$

Substituting (20) and (24), with their respective bounds (21) and (25), into (15) and (16) proves the two expressions when $f / D \to 0$ . The two bounds hold for every $1 \le \Delta \le K - 1$ because the pair count only distinguishes the case in (18) from the case in (22).

Now suppose that $f / D$ is bounded away from zero. Define

$$
h ( x ) = \frac { x } { 2 f - x } , \quad \mu _ { \Delta } = \mathbb { E } _ { \sigma } \left( X _ { \Delta } \right) = \frac { f ( f - 1 ) } { D - 1 } ,
$$

where the value of $\mu _ { \Delta }$ is from (17). Direct algebra gives

$$
\frac { h ( \mu _ { \Delta } ) } { f - 1 } = \frac { 1 } { 2 D - f - 1 } = \frac { 1 } { B _ { D } } .\tag{26}
$$

The same pair count used in (18) and (22) gives

$$
\mathrm { V a r } _ { \sigma } \left( X _ { \Delta } \right) = \left\{ \begin{array} { l l } { \frac { 2 f ( D - f ) ( f - 1 ) ( D - f - 1 ) } { ( D - 1 ) ^ { 2 } ( D - 3 ) } , } & { D \mathrm { ~ e v e n ~ a n d ~ } \Delta = D / 2 , } \\ { \frac { f ( D - f ) ( f - 1 ) ( D - f - 1 ) } { ( D - 1 ) ^ { 2 } ( D - 2 ) } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{27}
$$

For $0 \leq x \leq f ,$

$$
h ^ { \prime \prime } ( x ) = \frac { 4 f } { ( 2 f - x ) ^ { 3 } } , \quad 0 < h ^ { \prime \prime } ( x ) \leq \frac { 4 } { f ^ { 2 } } .\tag{28}
$$

We Taylor expansion about $\mu _ { \Delta }$ , take expectations, and using the fact $\mathbb { E } _ { \sigma } \left( X _ { \Delta } - \mu _ { \Delta } \right) = 0$ and (28) gives

$$
0 \leq \mathbb { E } _ { \sigma } \left( h ( X _ { \Delta } ) \right) - h ( \mu _ { \Delta } ) \leq { \frac { 2 } { f ^ { 2 } } } \mathrm { V a r } _ { \sigma } \left( X _ { \Delta } \right) .\tag{29}
$$

If $f / D$ is bounded below by a positive constant, the two expressions for the variance in (27) are $O ( \dot { D } )$ . Dividing (29) by $f - 1$ therefore gives

$$
0 \leq \frac { \mathbb { E } _ { \sigma } \left( h ( X _ { \Delta } ) \right) - h ( \mu _ { \Delta } ) } { f - 1 } \leq \frac { 2 \mathrm { V a r } _ { \sigma } \left( X _ { \Delta } \right) } { f ^ { 2 } ( f - 1 ) } = O ( D ^ { - 2 } ) .\tag{30}
$$

Combining (26) and (30) gives

$$
\frac { 1 } { f - 1 } \mathbb { E } _ { \sigma } \left( \frac { X _ { \Delta } } { 2 f - X _ { \Delta } } \right) = \frac { 1 } { B _ { D } } + O ( D ^ { - 2 } ) .\tag{31}
$$

Finally, substituting (31) into (15) and (16) gives the two expressions when $f / D$ is bounded away from zero. □

Theorem 2. Let $0 < a \le f \le D$ . Suppose we preprocess the location vector with the random permutation σ, and generate an independent permutation π. Let $B _ { D } = 2 D - f - 1 . \ I f f / D \to 0 \quad$

$$
\operatorname { V a r } _ { \sigma , \pi } \left( \theta ^ { ( \mathrm { C M H } ) } \right) = \frac { J ( 1 - J ) } { K } \left[ 1 - \frac { f } { D - 1 } \left\{ \frac { K - 1 - \lambda _ { D , K } } { 2 f - 1 } + \frac { \lambda _ { D , K } } { 2 ( f - 1 ) } \right\} \right] + O \left( \frac { f } { D ^ { 2 } } \right) .
$$

I $f f / D$ is bounded away from zero, then

$$
\mathrm { V a r } _ { \sigma , \pi } \left( \theta ^ { ( \mathrm { C M H } ) } \right) = \frac { J ( 1 - J ) } { K } \left( 1 - \frac { K - 1 } { B _ { D } } \right) + { \cal O } ( D ^ { - 2 } ) .
$$

Proof. This follows from expanding $\begin{array} { r } { \operatorname { V a r } _ { \sigma , \pi } \left( \sum _ { t = 1 } ^ { K } \mathbb { 1 } _ { t } / K \right) = \frac { 1 } { K ^ { 2 } } \sum _ { s = 1 } ^ { K } \sum _ { t = 1 } ^ { K } \operatorname { C o v } _ { \sigma , \pi } \left( \mathbb { 1 } _ { s } , \mathbb { 1 } _ { t } \right) } \end{array}$ using covariance terms in Lemma 2. The term $\dot { \lambda } _ { D , K }$ accounts for the proportion of covariance terms where D is even and $\Delta = D / 2$ □

## A.5 Additional Experiments to Verify Variance Approximation

While Figure 3 shows that our variance approximation for $\begin{array} { r } { \frac { f } { D }  0 } \end{array}$ does match the empirical MSE even at small $D = 1 2 8$ , we also need to understand how it behaves at edge cases $( \mathrm { i . e }$ . when $f _ { 1 } , f _ { 2 }$ approaches zero). While this usually isn’t the case in practical settings, we want to investigate how well we can rely on the given approximation.

Hence, we plot $( f , a )$ pairs of $( 2 , 1 ) , ( 4 , 2 ) , ( 6 , 3 ) , ( 8 , 4 )$ at $D = 1 2 8$ and $D = 1 0 2 4$ respectively. We see that at $f = 2 , 4 ,$ , the approximate theoretical variance lines are slightly off, but still extremely close to the empirical MSE, but at $f = 6 , 8$ , both lines overlap each other almost perfectly. Hence, we believe our approximation of the theoretical variance is practical enough to do comparisons.

![](images/18f2103cc94eee488b5fc1fa9bee6140fc8946496cbbd5d363581b15773972db.jpg)

![](images/b50cadef3a78e1e8a72902c43da921d95259c07a2546a1a6e95ab5498fe04dd3.jpg)

![](images/d7cd9b0a9523da60f446aa378800c4c30228ec964d3433a1525750ca8a8675a5.jpg)

![](images/4016e1a293cad67dd5b8b8876a77d533dbb64f76c43b8a5259da2a6227a1292c.jpg)

![](images/e6fd658aa54561e0c0b1d657d41fda493f55cc18532a5363a19ba006fb48b30b.jpg)

![](images/6f9199614319d1074c819f2ea9bf1a70f7094a51e7ff4dec6555d54156ffe9d9.jpg)

![](images/b2f447b05f6736e41e113dc50db87bd964cfbe4213b13a224a5dc6f0000162d3.jpg)

![](images/4d8a0d40669f5d53ee124bdfe1bfdf353e6d37579efa79eac757ac22d2050258.jpg)  
Figure 6: Empirical MSE vs theoretical approximate variance in Theorem 2 of MinHash and CMinHash $( \sigma , \pi )$ , at edge cases when $f$ is small. $\operatorname { A t } f = 2 , 4 ,$ , the dotted lines do not overlap perfectly, but they do at $\bar { f } = 6 , 8$

Finally, we plot both theoretical approximations for the variance in Thereom 2 and the empirical MSE for select word pairs from the words dataset [7] in Figure 7. Here, $D = 6 5 5 3 6$ , while the word pairs have $\textstyle { \frac { f _ { i } } { D } } \in [ 0 . 0 1 4 3 , 0 . 3 2 2 6 ]$ , i.e. where $\begin{array} { r } { \frac { f } { D }  0 } \end{array}$ and where $\textstyle { \frac { f } { D } }$ does not tend to zero.

We average across 10,000 iterations as well. We can see that both theoretical approximations as well as the empirical MSE lines overlap perfectly.

![](images/fdaf77595377a44782a73fed5bbad546369e1c3e4d2cfc97be1acbbee378f7bd.jpg)  
Figure 7: Empirical MSE versus both theoretical approximate variances in Theorem 2 of MinHash and CMinHash $( \sigma , \pi )$ for selected word pairs. CMH Var (S) is the the sparse case where $\begin{array} { r } { \frac { f _ { 1 } } { D }  0 } \end{array}$ . CMH Var (NS) is the the non-sparse case where $\textstyle { \frac { f _ { 1 } } { D } }$ does not tend to zero. Both theoretical approximations and empirical MSE lines overlap perfectly.

## B Appendix: Circulant Permutation for The Minner Estimators

This appendix covers Section 3 in more detail.

## B.1 Understanding the Minner Estimators

To better understand the Minner estimators, we give the following lemma before explaining in detail. Lemma 5. Suppose we have a random permutation π on $[ D ]$ . Let $y = \mathbb { 1 } _ { \{ \operatorname* { m i n } ( \pi ( S _ { 2 } ) ) \in \pi ( S _ { 1 } ) \} }$ , and m to be the count of elements in π $\cdot ( S _ { 1 } )$ strictly lesser than min $\left( \pi ( S _ { 2 } ) \right)$ ). Then $\begin{array} { r } { \mathbb { E } \left( y \right) = \frac { a } { f _ { 2 } } , \mathbb { E } \left( m \right) = \frac { f _ { 1 } - a } { f _ { 2 } + 1 } } \end{array}$ Var $\begin{array} { r } { ( m ) = \frac { ( f _ { 1 } - a ) ( f _ { 1 } + f _ { 2 } - a + 1 ) f _ { 2 } } { ( f _ { 2 } + 1 ) ^ { 2 } ( f _ { 2 } + 2 ) } } \end{array}$

<table><tr><td>Draw Order</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td>Set 1</td><td>O</td><td>●</td><td>O</td><td>●</td><td>●</td><td>●</td><td>O</td><td>●</td></tr><tr><td>Set 2</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>o</td><td></td></tr></table>

Figure 4: We have min $\{ \pi _ { t } ( S _ { 2 } ) \} = 8$ , with $y _ { t } = 1$ , which can be thought of as the presence (or absence) of a • at draw 8 for $S _ { 1 }$ , and $m _ { t } = 4$ , which can be thought of as counting how many • were seen in previous draws 1-7 for $S _ { 1 }$

We also replicate Figure 4 here, as viewing the random variables under the “drawing objects" from the sampling without replacement paradigm makes the proof straightforward.

Proof. Set $\mathcal { F } _ { + } = \left\{ : \right\}$ , and $\mathcal { F } _ { - } = \left\{ \begin{array} { l } { \circ } \\ { \bullet } \end{array} \right\}$ , with $| { \mathcal { F } } _ { + } | = a { \mathrm { ~ a n d ~ } } | { \mathcal { F } } _ { - } | = f _ { 2 } - a$ . Then y is simply the probability of seeing an element from $\mathcal { F } _ { + }$ drawn before an element in $\mathcal { F } _ { - }$ , which is $\begin{array} { r } { \frac { a } { f _ { 2 } - a + a } = \frac { a } { f _ { 2 } } } \end{array}$

Now, set $\mathcal { F } _ { + } = \left\{ \ : : , \ : \frac { \circ } { \bullet } \ : \right\}$ , and $\mathcal { F } _ { - } = \left\{ \begin{array} { l } { \bullet } \\ { \circ } \end{array} \right\}$ , with $| \mathcal { F } _ { + } | = f _ { 2 }$ and $| { \mathcal { F } } _ { - } | = f _ { 1 } - a$ . Then m is the count of the number of elements in $\mathcal { F } _ { - }$ seen before the first element in $\mathcal { F } _ { + }$

Suppose we index each element in $\mathcal { F } _ { - }$ by $; 1 \leq i \leq f _ { 1 } - a$ . We can write the random variable m as

$$
m = \sum _ { i = 1 } ^ { f _ { 1 } - a } \mathbb { 1 } _ { \{ { \mathrm { e l e m e n t ~ } } i { \mathrm { ~ i s ~ d r a w n ~ b e f o r e ~ a n y ~ e l e m e n t ~ i n ~ } } \mathcal { F } _ { + } \} } .
$$

There are $f _ { 2 }$ elements in $\mathcal { F } _ { + }$ , and we require the single element i in $\mathcal { F } _ { - }$ to be drawn first, hence

$$
\mathbb { E } \left( \mathbb { 1 } _ { \left\{ { \mathrm { e l e m e n t ~ } } i \mathrm { ~ i s ~ d r a w n ~ b e f o r e ~ a n y ~ e l e m e n t ~ i n ~ } \mathcal { F } _ { + } \right\} } \right) = \frac { 1 } { | \mathcal { F } _ { + } | + 1 } = \frac { 1 } { f _ { 2 } + 1 } ,
$$

$$
\begin{array} { r } { \mathbb { E } \left( m \right) = \sum _ { i = 1 } ^ { f _ { 1 } - a } \frac { 1 } { f _ { 2 } + 1 } = \frac { f _ { 1 } - a } { f _ { 2 } + 1 } } \end{array}
$$

To find Var (m), we need to find E $\left( m ^ { 2 } \right)$ . Certainly,

$$
\begin{array} { r l } & { \mathbb { E } \left( m ^ { 2 } \right) = \displaystyle \sum _ { i = 1 } ^ { f _ { 1 } - a } \mathbb { E } \left( \mathbb { 1 } _ { \{ \mathrm { \sf ~ c l e m e n t ~ } i \mathrm { \ i s ~ d r a w n ~ b e f o r c ~ a n y ~ e l e m e n t ~ i n ~ } \mathcal { F } _ { + } \} } \right) } \\ & { \qquad + \displaystyle \sum _ { i \neq j } \mathbb { E } \left( \mathbb { 1 } _ { \{ \mathrm { \sf ~ c l e m e n t ~ } i \mathrm { \ i s ~ d r a w n ~ b e f o r c ~ a n y ~ e l e m e n t i n ~ } \mathcal { F } _ { + } \} } \mathbb { 1 } _ { \{ \mathrm { \sf ~ c l e m e n t ~ } j \mathrm { \ i s ~ d r a w n ~ b e f o r c ~ a n y ~ c l e m e n t i n ~ } \mathcal { F } _ { + } \} } \right) . } \end{array}
$$

Now $\mathbb { E } ( \mathbb { 1 } _ { \mathrm { \{ e l e m e n t ~ } } i $ is drawn before any element in $\mathcal { F } _ { + } \} \mathbb { 1 }$ {element j is drawn before any element in $\begin{array} { r } { \mathcal { F } _ { + } \} \big ) = \frac { 2 } { ( f _ { 2 } + 1 ) ( f _ { 2 } + 2 ) } . } \end{array}$ as this can be thought of as out of the $f _ { 2 }$ elements in $\mathcal { F } _ { + }$ , and the two elements $i , j$ the $i , j$ elements must be drawn first. Therefore,

$$
\operatorname { \mathbb { E } } { \left( m ^ { 2 } \right) } = { \frac { f _ { 1 } - a } { f _ { 2 } + 1 } } + { \frac { 2 ( f _ { 1 } - a ) ( f _ { 1 } - a - 1 ) } { ( f _ { 2 } + 1 ) ( f _ { 2 } + 2 ) } } ,
$$

and

$$
\mathrm { V a r } \left( m \right) = \mathbb { E } \left( m ^ { 2 } \right) - \mathbb { E } \left( m \right) ^ { 2 } = \frac { \left( f _ { 1 } - a \right) f _ { 2 } \left( f _ { 1 } + f _ { 2 } - a + 1 \right) } { ( f _ { 2 } + 1 ) ^ { 2 } ( f _ { 2 } + 2 ) } .
$$

We remark that both these estimators have different performance based on the sizes of the sets $s _ { 1 } , s _ { 2 }$ However, it is not in the scope of this paper to analyze which estimator is better given $a , f _ { 1 } , f _ { 2 } , D$ or propose better Minner estimators. The paper focuses on extending the circulant permutation framework when counts are involved.

## B.2 Proof of Lemma 3

We recall the definition of the nine sets again, but this time using our notation $C _ { s } , C _ { f }$ and $C _ { n }$ (success, failure, neutral) for generalizability.

For $A , B \in \{ \widetilde { O } , \widetilde { \times } , \widetilde { = } \}$ , let $\{ ( A , B ) \mid \Delta \}$ be the set $\{ ( i , j ) : ( x _ { i } , x _ { j } ) = ( A , B ) , j - i = \Delta \}$ . For $1 \le \Delta \le K - 1$ , define

$$
\mathcal { L } _ { 0 } ( \Delta ) = \{ ( \widetilde O , \widetilde O ) | \Delta \} , ~ \mathcal { L } _ { 1 } ( \Delta ) = \{ ( \widetilde O , \widetilde { \times } ) | \Delta \} , ~ \mathcal { L } _ { 2 } ( \Delta ) = \{ ( \widetilde O , \widetilde { - } ) | \Delta \} ,
$$

$$
\mathcal { G } _ { 0 } ( \Delta ) = \{ ( \widetilde { - } , \widetilde { O } ) | \Delta \} , ~ \mathcal { G } _ { 1 } ( \Delta ) = \{ ( \widetilde { - } , \widetilde { \times } ) | \Delta \} , ~ \mathcal { G } _ { 2 } ( \Delta ) = \{ ( \widetilde { - } , \widetilde { - } ) | \Delta \} ,
$$

$$
\mathcal { H } _ { 0 } ( \Delta ) = \{ ( \widetilde { \times } , \widetilde { O } ) | \Delta \} , \quad \mathcal { H } _ { 1 } ( \Delta ) = \{ ( \widetilde { \times } , \widetilde { \times } ) | \Delta \} , \quad \mathcal { H } _ { 2 } ( \Delta ) = \{ ( \widetilde { \times } , \widetilde { - } ) | \Delta \} ,
$$

with the following constraints for all $1 \le \Delta \le K - 1$

$$
| \mathcal { L } _ { 0 } | + | \mathcal { L } _ { 1 } | + | \mathcal { L } _ { 2 } | = | \mathcal { L } _ { 0 } | + | \mathcal { G } _ { 0 } | + | \mathcal { H } _ { 0 } | = C _ { s } ,
$$

$$
| \mathcal { G } _ { 0 } | + | \mathcal { G } _ { 1 } | + | \mathcal { G } _ { 2 } | = | \mathcal { L } _ { 2 } | + | \mathcal { G } _ { 2 } | + | \mathcal { H } _ { 2 } | = C _ { n } ,
$$

$$
\begin{array} { r } { | \mathcal { H } _ { 0 } | + | \mathcal { H } _ { 1 } | + | \mathcal { H } _ { 2 } | = | \mathcal { L } _ { 1 } | + | \mathcal { G } _ { 1 } | + | \mathcal { H } _ { 1 } | = C _ { f } . } \end{array}
$$

Lemma 3. Suppose we pre-process the location vector with a random permutation σ. Let the random variable $\theta _ { s } \equiv$ number $o f \widetilde { \times }$ seen before the first $\widetilde O$ under the random permutation $\pi  s \cdot$ Let $B _ { \Delta } = 2 C _ { s } - \left. \mathcal { L } _ { 0 } \right. + 1$ . Then for any $1 \leq s < t \leq K _ {  }$ , with $t - s = \Delta$

$$
\mathbb { E } _ { \pi } \left( \theta _ { s } \theta _ { s + \Delta } \right) = \frac { \left| \mathcal { H } _ { 1 } \right| } { B _ { \Delta } } + \frac { 2 C _ { f } ^ { 2 } - C _ { f } ( \left| \mathcal { L } _ { 1 } \right| + \left| \mathcal { H } _ { 0 } \right| ) - 2 \left| \mathcal { H } _ { 1 } \right| } { ( C _ { s } + 1 ) ( B _ { \Delta } + 1 ) } + \frac { C _ { f } ( \left| \mathcal { L } _ { 1 } \right| + \left| \mathcal { H } _ { 0 } \right| ) - 2 \left| \mathcal { L } _ { 1 } \right| \left| \mathcal { H } _ { 0 } \right| } { ( C _ { s } + 1 ) B _ { \Delta } ( B _ { \Delta } + 1 ) } .
$$

$I f C _ { s }$ and $C _ { f }$ arefixed as $D \to \infty$ , then, when $D$ is even and $\Delta = D / 2$

$$
\mathrm { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = - V _ { \mathrm { i n d } } \frac { ( C _ { s } + 1 ) ( C _ { s } + 2 ) } { 2 C _ { s } ( 2 C _ { s } + 1 ) } \frac { 2 C _ { s } + C _ { f } } { ( C _ { s } + C _ { f } + 1 ) ( D - 1 ) } + O ( D ^ { - 2 } ) ,
$$

and otherwise,

$$
\mathrm { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = - V _ { \mathrm { i n d } } \frac { ( C _ { s } + 2 ) ( C _ { s } + C _ { f } ) } { ( 2 C _ { s } + 1 ) ( C _ { s } + C _ { f } + 1 ) ( D - 1 ) } + O ( D ^ { - 2 } ) .
$$

If both $C _ { s } / D$ and $C _ { f } / D$ are bounded awayfrom zero, then

$$
\begin{array} { r } { \mathrm { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = \left\{ \begin{array} { l l } { - V _ { \mathrm { i n d } } \left( \frac { C _ { s } \left( 2 D - C _ { s } \right) + C _ { f } \left( D - C _ { s } \right) } { \left( C _ { s } + C _ { f } \right) \left( 2 D - C _ { s } \right) ^ { 2 } } \right) + O \left( \frac { 1 } { D ^ { 2 } } \right) , } & { D e \nu e n a n d \Delta = D / 2 , } \\ { - V _ { \mathrm { i n d } } } & { \left( \frac { 1 } { 2 D - C _ { s } } \right) } & { + O \left( \frac { 1 } { D ^ { 2 } } \right) , } \end{array} \right. } \end{array}
$$

Proof. Our proof will follow a similar approach to the proof of Lemma 1 with respect to sequence of draws.

For ease of notation, we define the sets

$$
\mathcal { O } _ { 1 } = \mathcal { L } _ { 0 } \cup \mathcal { L } _ { 1 } \cup \mathcal { L } _ { 2 } ( \mathrm { t u p l e s ~ w h e r e ~ f i r s t ~ e n t r y ~ i s ~ } \widetilde { O } ) ,
$$

$$
\mathcal { O } _ { 2 } = \mathcal { L } _ { 0 } \cup \mathcal { G } _ { 0 } \cup \mathcal { H } _ { 0 } \mathrm { ~ ( t u p l e s ~ w h e r e ~ s e c o n d ~ e n t r y ~ i s ~ \widetilde { \mathcal { O } } ) , }
$$

$\mathcal { X } _ { 1 } = \mathcal { H } _ { 0 } \cup \mathcal { H } _ { 1 } \cup \mathcal { H } _ { 2 }$ (tuples where first entry is $\widetilde { \times } )$

$\mathcal { X } _ { 2 } = \mathcal { L } _ { 1 } \cup \mathcal { G } _ { 1 } \cup \mathcal { H } _ { 1 }$ (tuples where second entry is $\widetilde { \times } )$ ,

and define the variable

$$
U = \left| \mathcal { O } _ { 1 } \cup \mathcal { O } _ { 2 } \right| = 2 C _ { s } - \left| \mathcal { L } _ { 0 } \right| .
$$

For an element i in $\mathcal { X } _ { 1 }$ , we write the random variable $I _ { i } ^ { ( 1 ) } \equiv \mathbb { 1 } _  \{ :$ <sub>{i</sub> <sub>appears</sub> <sub>before</sub> <sub>elements</sub> <sub>in</sub> <sub>O1}</sub>. Similarly, $\mathcal { O } _ { 1 } \}$ for an element $j$ in $\mathcal { X } _ { 2 }$ , we write $I _ { j } ^ { ( 2 ) } \equiv \mathbb { 1 } _ { \{ j \} }$ appears before elements in $\mathcal { O } _ { 2 } \}$ . The two random variables can therefore be written as

$$
\theta _ { s } = \sum _ { i \in \mathcal { X } _ { 1 } } I _ { i } ^ { ( 1 ) } , \quad \theta _ { s + \Delta } = \sum _ { j \in \mathcal { X } _ { 2 } } I _ { j } ^ { ( 2 ) } .\tag{32}
$$

Multiplying the two sums in (32) gives

$$
\mathbb { E } _ { \boldsymbol { \pi } } \left( \theta _ { s } \theta _ { s + \Delta } \right) = \sum _ { i \in \mathcal { X } _ { 1 } \cap \mathcal { X } _ { 2 } } \mathbb { E } _ { \boldsymbol { \pi } } \left( I _ { i } ^ { ( 1 ) } I _ { i } ^ { ( 2 ) } \right) + \sum _ { \underset { i \neq j } { \sum _ { i , j \in \mathcal { X } _ { 2 } } } } \mathbb { E } _ { \boldsymbol { \pi } } \left( I _ { i } ^ { ( 1 ) } I _ { j } ^ { ( 2 ) } \right) .\tag{33}
$$

For the first sum in (33), the equality $i = j$ requires the tuple $( \widetilde { \times } , \widetilde { \times } )$ , so there are $| \mathcal { H } _ { 1 } |$ choices. For any one of them, $I _ { i } ^ { ( 1 ) } I _ { i } ^ { ( 2 ) } = 1$ when i is the first tuple among $\mathcal { O } _ { 1 } \cup \mathcal { O } _ { 2 } \cup \{ i \}$ . Exchangeability of these $U + 1$ tuples gives probability $1 / ( U + 1 )$ . Hence

$$
\sum _ { i \in \mathcal { X } _ { 1 } \cap \mathcal { X } _ { 2 } } \mathbb { E } _ { \boldsymbol \pi } \left( I _ { i } ^ { ( 1 ) } I _ { i } ^ { ( 2 ) } \right) = \frac { | \mathcal { H } _ { 1 } | } { U + 1 } .\tag{34}
$$

We now evaluate the second sum in (33). As in the proof of Lemma 1, we consider four cases depending on whether i is in $\mathcal { O } _ { 2 }$ and whether j is in $\mathcal { O } _ { 1 }$

Case One: Suppose $i \notin \mathcal { O } _ { 2 }$ and $j \not \in { \mathcal { O } } _ { 1 }$ . Let E be the earliest tuple seen out of $\mathcal { O } _ { 1 } \cup \mathcal { O } _ { 2 } \cup \{ i \} \cup \{ j \}$ Since both i and j are outside $\mathcal { O } _ { 1 } \cup \mathcal { O } _ { 2 }$ , we have

$$
\mathbb { P } \left( E = i \right) = \mathbb { P } \left( E = j \right) = \frac { 1 } { U + 2 } .
$$

A valid sequence occurs in two disjoint ways. First, i is drawn before any tuple in $\mathcal { O } _ { 1 } \cup \mathcal { O } _ { 2 } \cup \{ j \}$ and then $j$ is drawn before any tuple in $\mathcal { O } _ { 2 } .$ Second, j is drawn before any tuple in $\mathcal { O } _ { 1 } \cup \mathcal { O } _ { 2 } \cup \{ i \}$ and then i is drawn before any tuple in $\mathcal { O } _ { 1 }$ . Thus the probability for each ordered pair in this case is

$$
\frac { 1 } { U + 2 } \frac { 1 } { C _ { s } + 1 } + \frac { 1 } { U + 2 } \frac { 1 } { C _ { s } + 1 } = \frac { 2 } { ( C _ { s } + 1 ) ( U + 2 ) } .
$$

There are

$$
( | \mathcal { H } _ { 1 } | + | \mathcal { H } _ { 2 } | ) ( | \mathcal { G } _ { 1 } | + | \mathcal { H } _ { 1 } | ) - | \mathcal { H } _ { 1 } | = ( C _ { f } - | \mathcal { H } _ { 0 } | ) ( C _ { f } - | \mathcal { L } _ { 1 } | ) - | \mathcal { H } _ { 1 } |
$$

ordered pairs with $i \notin \mathcal { O } _ { 2 } , j \notin \mathcal { O } _ { 1 }$ , and $i \neq j$ . Hence the contribution from Case One is

$$
\frac { 2 \left( ( C _ { f } - | \mathcal { H } _ { 0 } | ) ( C _ { f } - | \mathcal { L } _ { 1 } | ) - | \mathcal { H } _ { 1 } | \right) } { ( C _ { s } + 1 ) ( U + 2 ) } .\tag{35}
$$

Case Two: Suppose $i \in \mathcal { O } _ { 2 }$ and $j \not \in { \mathcal { O } } _ { 1 }$ . Since $i \in \mathcal { X } _ { 1 } \cap \mathcal { O } _ { 2 }$ , this means $i \in \mathcal { H } _ { 0 }$ . A valid sequence must first draw j before any tuple in $\mathcal { O } _ { 2 }$ , and then draw i before any tuple in $\mathcal { O } _ { 1 }$ . Since only j is added to $\mathcal { O } _ { 1 } \cup \mathcal { O } _ { 2 }$ , the probability for each ordered pair in this case is

$$
\frac { 1 } { U + 1 } \frac { 1 } { C _ { s } + 1 } = \frac { 1 } { ( C _ { s } + 1 ) ( U + 1 ) } .
$$

There are $\left| \mathcal { H } _ { 0 } \right| \left( C _ { f } - \left| \mathcal { L } _ { 1 } \right| \right)$ such ordered pairs. Hence the contribution from Case Two is

$$
\frac { | \mathcal { H } _ { 0 } | ( C _ { f } - | \mathcal { L } _ { 1 } | ) } { ( C _ { s } + 1 ) ( U + 1 ) } .\tag{36}
$$

Case Three: Suppose i $\notin \mathcal { O } _ { 2 }$ and $j \in \mathcal { O } _ { 1 }$ . By the same reasoning as Case Two, $j \in \mathcal { L } _ { 1 }$ , and the contribution is

$$
\frac { ( C _ { f } - | \mathcal { H } _ { 0 } | ) | \mathcal { L } _ { 1 } | } { ( C _ { s } + 1 ) ( U + 1 ) } .\tag{37}
$$

Case Four: Suppose $i \in \mathcal { O } _ { 2 }$ and $j \in \mathcal { O } _ { 1 }$ . This case contributes zero. Since $i \in \mathcal { O } _ { 2 } .$ , the event $\mathbb { 1 } _ { \{ j \mathrm { ~ b e f o r e } \mathcal { O } _ { 2 } \} } = 1$ requires j to appear before i. However, since $j \in \mathcal { O } _ { 1 }$ , the event $\mathbb { 1 } _ { \{ i \mathrm { ~ b e f o r e } \mathcal { O } _ { 1 } \} } = 1$ requires i to appear before $j$

Adding (34), (35), (36), and (37), and substituting $U = 2 C _ { s } - | \mathcal { L } _ { 0 } | .$ , gives

$$
\begin{array} { r l } & { \mathbb { E } _ { \boldsymbol \pi } \left( \theta _ { s } \theta _ { s + \Delta } \right) = \frac { \left| \mathcal { H } _ { 1 } \right| } { 2 C _ { s } - \left| \mathcal { L } _ { 0 } \right| + 1 } + \frac { 2 C _ { f } ^ { 2 } - C _ { f } ( \left| \mathcal { L } _ { 1 } \right| + \left| \mathcal { H } _ { 0 } \right| ) - 2 \left| \mathcal { H } _ { 1 } \right| } { ( C _ { s } + 1 ) ( 2 C _ { s } - \left| \mathcal { L } _ { 0 } \right| + 2 ) } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad  \end{array}\tag{38}
$$

This proves the first expression in Lemma 3.

Now suppose we pre-process the location vector with the random permutation σ. Then we want to find the expectation of (38) with respect to σ. For ease of notation, let

$$
\boldsymbol { Z } = \left( Z _ { 0 } , Z _ { 1 } , Z _ { 2 } , Z _ { 3 } \right) = \left( \left| \mathcal { L } _ { 0 } \right| , \left| \mathcal { L } _ { 1 } \right| , \left| \mathcal { H } _ { 0 } \right| , \left| \mathcal { H } _ { 1 } \right| \right) ,\tag{39}
$$

and let $F ( Z )$ denote the right hand side of (38). When $Z = ( 0 , 0 , 0 , 0 )$

$$
F ( 0 , 0 , 0 , 0 ) = \frac { 2 C _ { f } ^ { 2 } } { ( C _ { s } + 1 ) ( 2 C _ { s } + 2 ) } = \left( \frac { C _ { f } } { C _ { s } + 1 } \right) ^ { 2 } .\tag{40}
$$

First suppose that $C _ { s }$ and $C _ { f }$ are fixed while $D \to \infty .$ , and that D is odd or $\Delta \neq D / 2 .$ Let

$$
R _ { \Delta } = Z _ { 0 } + Z _ { 1 } + Z _ { 2 } + Z _ { 3 } .
$$

Thus $R _ { \Delta }$ is the number of tuples $( x _ { i } , x _ { i + \Delta } )$ whose two entries are $\widetilde { O } \mathrm { o r } \widetilde { \times }$ . When $R _ { \Delta } = 1$ , the four possible values of $Z$ and their changes from (40) are

$$
\begin{array} { r l r } & { } & { \frac { \mathrm {  ~ { \cal ~ P a i r } ~ } \tilde { \cal \Gamma } } { ( \tilde { \cal O } , \tilde { \cal O } ) } \left| \begin{array} { c } { { \cal Z } } \\ { { \bf ( 1 , 0 , 0 , 0 ) } } \\ { { \bf ( \tilde { \cal C } _ { s } + 1 ) ^ { 2 } ( 2 { \cal C } _ { s } + 1 ) } } \end{array} \right| \frac { { \cal F } ( \cal Z ) - { \cal F } ( 0 , 0 , 0 , 0 ) } { ( { \cal C } _ { s } + 1 ) ^ { 2 } ( 2 { \cal C } _ { s } + 1 ) } } \\ & { } & { \stackrel { \mathrm {  ~ { \cal ~ ( \tilde { \cal ~ O } , \tilde { \cal ~ X } ) ~ } } } { = } \left| \begin{array} { c } { { \bf 0 } , 1 , 0 , 0 , 0 } \\ { { \bf ( 0 , 0 , 1 , 0 ) } } \end{array} \right| \frac { { \cal F } ( \cal Z ) - { \cal F } ( 0 , 0 , 0 , 0 , 0 ) } { ( { \cal C } _ { s } + 1 ) ^ { 2 } ( 2 { \cal C } _ { s } + 1 ) } } \\ & { } & { \stackrel { \mathrm {  ~ { \cal ~ C } ~ } } { = } \left| \begin{array} { c } { { \cal C } _ { s } \tilde { \cal C } _ { f } } \\ { { \bf ( 1 , 0 , 0 , 1 ) } } \end{array} \right| \frac { { \cal C } _ { s } ^ { 2 } } { ( { \cal C } _ { s } + 1 ) ^ { 2 } ( 2 { \cal C } _ { s } + 1 ) } } \\ & { } & { \stackrel { \mathrm {  ~ { \cal ~ C } ~ } } { = } \left| \begin{array} { c } { { \cal C } _ { s } ^ { 2 } } \\ { { \bf ( 1 , 0 , 0 , 1 ) } } \end{array} \right| \frac { { \cal C } _ { s } ^ { 2 } } { ( { \cal C } _ { s } + 1 ) ^ { 2 } ( 2 { \cal C } _ { s } + 1 ) } . } \end{array}\tag{41}
$$

The four rows in (41) are obtained by substituting $Z = ( 1 , 0 , 0 , 0 ) , Z = ( 0 , 1 , 0 , 0 ) , Z = ( 0 , 0 , 1 , 0 )$ and $Z = ( 0 , 0 , 0 , 1 )$ into the right hand side of (38), and then subtracting (40). For example, the $( \widetilde O , \widetilde { \times } )$ row is

$$
\begin{array} { l } { F ( 0 , 1 , 0 , 0 ) - F ( 0 , 0 , 0 , 0 ) = \displaystyle \frac { 2 C _ { f } ^ { 2 } - C _ { f } } { 2 ( C _ { s } + 1 ) ^ { 2 } } + \displaystyle \frac { C _ { f } } { 2 ( C _ { s } + 1 ) ^ { 2 } ( 2 C _ { s } + 1 ) } - \displaystyle \frac { C _ { f } ^ { 2 } } { ( C _ { s } + 1 ) ^ { 2 } } } \\ { \displaystyle = - \displaystyle \frac { C _ { s } C _ { f } } { ( C _ { s } + 1 ) ^ { 2 } ( 2 C _ { s } + 1 ) } . } \end{array}
$$

For each $\Delta ,$ , sampling without replacement gives

$$
\begin{array} { r l r l } & { \mathbb { E } _ { \sigma } \left( | { \mathcal { L } } _ { 0 } | \right) = \displaystyle \frac { C _ { s } \left( C _ { s } - 1 \right) } { D - 1 } , \qquad } & & { \mathbb { E } _ { \sigma } \left( | { \mathcal { L } } _ { 1 } | \right) = \displaystyle \frac { C _ { s } C _ { f } } { D - 1 } , } \\ & { \mathbb { E } _ { \sigma } \left( | { \mathcal { H } } _ { 0 } | \right) = \displaystyle \frac { C _ { s } C _ { f } } { D - 1 } , \qquad } & & { \mathbb { E } _ { \sigma } \left( | { \mathcal { H } } _ { 1 } | \right) = \displaystyle \frac { C _ { f } \left( C _ { f } - 1 \right) } { D - 1 } . } \end{array}\tag{42}
$$

The same calculation as in (18), with $f$ replaced by $C _ { s } + C _ { f }$ gives

$$
\mathbb { E } _ { \sigma } \left( R _ { \Delta } ( R _ { \Delta } - 1 ) \right) = O ( D ^ { - 2 } ) .
$$

Therefore P $( R _ { \Delta } \ge 2 ) = O ( D ^ { - 2 } )$ . When $R _ { \Delta } = 0 .$ , the change in $F$ is zero by (40). When $R _ { \Delta } = 1$ it is the corresponding entry of (41). When $R _ { \Delta } \geq 2 , F$ is bounded because $\dot { C } _ { s }$ and $C _ { f }$ are constant. Multiplying the four entries in (41) by the four expectations in (42) and adding them gives

$$
\begin{array} { r l } { \mathrm { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = \frac { 1 } { ( C _ { s } + 1 ) ^ { 2 } ( 2 C _ { s } + 1 ) ( D - 1 ) } \left\{ C _ { f } ^ { 2 } C _ { s } ( C _ { s } - 1 ) \right. } & { } \\ { \left. - 2 C _ { s } ^ { 2 } C _ { f } ^ { 2 } + C _ { s } ^ { 2 } C _ { f } ( C _ { f } - 1 ) \right\} + O ( D ^ { - 2 } ) } & { } \\ { = - \frac { C _ { s } C _ { f } ( C _ { s } + C _ { f } ) } { ( C _ { s } + 1 ) ^ { 2 } ( 2 C _ { s } + 1 ) ( D - 1 ) } + O ( D ^ { - 2 } ) . } \end{array}\tag{43}
$$

Now suppose that D is even and $\Delta = D / 2$ . The tuple $\left( { { x } _ { i } } , { { x } _ { i + D / 2 } } \right)$ is accompanied by $\left( { { x } _ { i + D / 2 } } , { { x } _ { i } } \right)$ Let $Q _ { \Delta }$ be the number of pairs of positions $\{ i , i + D / 2 \}$ for which both entries are $\widetilde { O } \mathrm { o r } \widetilde { \times }$ . When $Q _ { \Delta } = 1$ , the possible values of $Z$ and their changes are

$$
\begin{array} { r l } { \frac { \mathrm { e n t r i e s ~ i n ~ t h e ~ p a i r ~ } \bigl | \qquad Z } { ( \widetilde { O } , \widetilde { O } ) } } & { \left| \begin{array} { c } { F ( Z ) - F ( 0 , 0 , 0 , 0 ) } \\ { \qquad ( \widetilde { O } , \widetilde { O } ) } \end{array} \right| } \\ { ( \widetilde { O } , \widetilde { \times } ) } & { \left| \begin{array} { c } { ( 0 , 1 , 1 , 0 ) } \\ { ( 0 , 1 , 1 ) ^ { 2 } \bigl | \qquad \frac { 2 C _ { s } C _ { f } + 1 } { ( C _ { s } + 1 ) ^ { 2 } ( 2 C _ { s } + 1 ) } } \\ { \qquad ( \widetilde { \times } , \widetilde { \times } ) } \end{array} \right| } & { } \end{array}\tag{44}
$$

For example, a pair containing $\widetilde O$ and $\boldsymbol { \widetilde { \times } } \mathrm { g i v e s }$ the two tuples $( \widetilde O , \widetilde { \times } )$ and $( \widetilde { \times } , \widetilde { O } )$ , so $Z = ( 0 , 1 , 1 , 0 )$ Substitution into (38) gives

$$
\begin{array} { l } { F ( 0 , 1 , 1 , 0 ) - F ( 0 , 0 , 0 , 0 ) = \displaystyle \frac { 2 C _ { f } ^ { 2 } - 2 C _ { f } } { 2 ( C _ { s } + 1 ) ^ { 2 } } + \displaystyle \frac { 2 C _ { f } - 2 } { 2 ( C _ { s } + 1 ) ^ { 2 } ( 2 C _ { s } + 1 ) } - \displaystyle \frac { C _ { f } ^ { 2 } } { ( C _ { s } + 1 ) ^ { 2 } } } \\ { \displaystyle = - \displaystyle \frac { 2 C _ { s } C _ { f } + 1 } { ( C _ { s } + 1 ) ^ { 2 } ( 2 C _ { s } + 1 ) } . } \end{array}
$$

There are $D / 2$ pairs of positions of the form $\{ i , i + D / 2 \}$ . Sampling the two entries without replacement gives respective expectations

$$
\frac { C _ { s } ( C _ { s } - 1 ) } { 2 ( D - 1 ) } , \quad \frac { C _ { s } C _ { f } } { D - 1 } , \quad \frac { C _ { f } ( C _ { f } - 1 ) } { 2 ( D - 1 ) }\tag{45}
$$

for $( \tilde { O } , \tilde { O } ) , ( \tilde { O } , \tilde { \times } )$ , and $( \widetilde { \times } , \widetilde { \times } )$ , respectively. The calculation in (22) with $f$ replaced by $C _ { s } + C _ { f }$ gives

$$
\mathbb { E } _ { \sigma } \left( Q _ { \Delta } ( Q _ { \Delta } - 1 ) \right) = O ( D ^ { - 2 } ) .
$$

Thus the cases with $Q _ { \Delta } \geq 2$ affect the expectation only by $O ( D ^ { - 2 } )$ . Multiplying the entries in (44) by the three expectations in (45) and adding them gives

$$
\operatorname { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = - \frac { C _ { f } ( 2 C _ { s } + C _ { f } ) } { 2 ( C _ { s } + 1 ) ( 2 C _ { s } + 1 ) ( D - 1 ) } + O ( D ^ { - 2 } ) .\tag{46}
$$

Lemma $^ { 5 }$ gives the one shift variance

$$
V _ { \mathrm { i n d } } = \frac { C _ { s } C _ { f } ( C _ { s } + C _ { f } + 1 ) } { ( C _ { s } + 1 ) ^ { 2 } ( C _ { s } + 2 ) } .\tag{47}
$$

Dividing the leading term of (43) by (47) gives

$$
\mathrm { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = - V _ { \mathrm { i n d } } \frac { ( C _ { s } + 2 ) ( C _ { s } + C _ { f } ) } { ( 2 C _ { s } + 1 ) ( C _ { s } + C _ { f } + 1 ) ( D - 1 ) } + O ( D ^ { - 2 } ) ,\tag{48}
$$

while dividing (46) by (47) gives

$$
\mathrm { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = - V _ { \mathrm { i n d } } \frac { ( C _ { s } + 1 ) ( C _ { s } + 2 ) } { 2 C _ { s } ( 2 C _ { s } + 1 ) } \frac { 2 C _ { s } + C _ { f } } { ( C _ { s } + C _ { f } + 1 ) ( D - 1 ) } + O ( D ^ { - 2 } ) .\tag{49}
$$

Equations (48) and (49) give the next two expressions in Lemma 3 when $C _ { s }$ and $C _ { f }$ are fixed.

Finally, we now suppose that both $C _ { s } / D$ and $C _ { f } / D$ are bounded away from zero. We continue to use $Z$ from (39) and F from (38). The expectation of $Z$ is

$$
\mu = \mathbb { E } _ { \sigma } \left( Z \right) = \frac { 1 } { D - 1 } \left( C _ { s } ( C _ { s } - 1 ) , C _ { s } C _ { f } , C _ { s } C _ { f } , C _ { f } ( C _ { f } - 1 ) \right) .\tag{50}
$$

We use $P ( A _ { 1 } , \ldots , A _ { r } )$ to denote the probability that a sequence of r draws without replacement from the location vector gives $A _ { 1 } , \ldots , A _ { i }$ <sub>r</sub> in this order, where every $A _ { i } \mathrm { i s } \widetilde { O } , \widetilde { \times } , \mathrm { o r } \widetilde { \stackrel { \sim } { - } }$ . For example,

$$
P ( \widetilde { O } , \widetilde { \times } ) = \frac { C _ { s } C _ { f } } { D ( D - 1 ) }
$$

For any two entries $A , B \in \{ \widetilde { O } , \widetilde { \times } , \widetilde { = } \}$ , let

$$
Z _ { A B } = \sum _ { i = 1 } ^ { D } \mathbb { 1 } _ { \{ ( x _ { i } , x _ { i + \Delta } ) = ( A , B ) \} } .\tag{51}
$$

Thus $Z _ { A B }$ is the number of times $( A , B )$ occurs among the tuples $( x _ { i } , x _ { i + \Delta } )$

Suppose first that D is odd or $\Delta \neq D / 2$ . There are $D ^ { 2 }$ terms after expanding $Z _ { A B } Z _ { C E }$ in (51). If the two starting positions are equal, there are $D$ terms, and they are nonzero only if $A = C$ and $B = E$ . If the second starting position is $\Delta$ after the first, there are $D$ terms, and the common entry requires $B = C$ . If the first starting position is $\Delta$ after the second, there are $D$ terms, and the common entry requires $E = A$ . The remaining $D ( D - 3 )$ terms use four distinct positions. Adding these four cases gives

$$
\begin{array} { r l } & { \mathbb { E } _ { \sigma } \left( Z _ { A B } Z _ { C E } \right) = D \mathbb { 1 } _ { \{ A = C , B = E \} } P ( A , B ) + D \mathbb { 1 } _ { \{ B = C \} } P ( A , B , E ) } \\ & { \qquad + D \mathbb { 1 } _ { \{ E = A \} } P ( C , A , B ) + D ( D - 3 ) P ( A , B , C , E ) . } \end{array}\tag{52}
$$

Now suppose that $D$ is even and $\Delta = D / 2$ . There are $D$ terms with the same starting position. There are another $D$ terms in which $( x _ { j } , x _ { j + \Delta } ) = \left( x _ { i + \Delta } , x _ { i } \right)$ , which requires $A = E$ and $B = C$ . The remaining $D ( D - 2 )$ terms have four distinct positions. Therefore

$$
\begin{array} { r l } & { \mathbb { E } _ { \sigma } \left( Z _ { A B } Z _ { C E } \right) = D \mathbb { 1 } _ { \left\{ A = C , B = E \right\} } P ( A , B ) + D \mathbb { 1 } _ { \left\{ A = E , B = C \right\} } P ( A , B ) } \\ & { \qquad + D ( D - 2 ) P ( A , B , C , E ) . } \end{array}\tag{53}
$$

Equations (52) and (53), together with their corresponding draw probabilities, give the sixteen second moments of the four entries in (39). For example, when $\mathbf { \bar { \boldsymbol { D } } }$ is odd or $\Delta \neq D / 2$

$$
\mathbb { E } _ { \sigma } \left( \left. \mathcal { L } _ { 0 } \right. \left. \mathcal { L } _ { 1 } \right. \right) = D P ( \widetilde { O } , \widetilde { O } , \widetilde { \times } ) + D ( D - 3 ) P ( \widetilde { O } , \widetilde { O } , \widetilde { O } , \widetilde { \times } ) ,
$$

because the second $\widetilde O$ in $( \widetilde { O } , \widetilde { O } )$ can be the first $\widetilde O$ in $( \widetilde O , \widetilde { \times } )$ . When D is even and $\Delta = D / 2$

$$
\mathbb { E } _ { \sigma } \left( \left. \mathcal { L } _ { 1 } \right. \left. \mathcal { H } _ { 0 } \right. \right) = D P ( \widetilde { O } , \widetilde { \times } ) + D ( D - 2 ) P ( \widetilde { O } , \widetilde { \times } , \widetilde { \times } , \widetilde { O } ) ,
$$

because $( \widetilde O , \widetilde { \times } )$ and $( \widetilde { \times } , \widetilde { O } )$ can be the two tuples formed from the same two positions.

Let $W = Z - \mu$ . Taylor expansion of $F ( \mu + W )$ and taking expectations give

$$
\begin{array} { r l } & { \displaystyle \mathbb { E } _ { \sigma } \left( F ( Z ) \right) = F ( \mu ) + \frac { 1 } { 2 } \sum _ { r , q = 0 } ^ { 3 } F _ { r q } ( \mu ) \mathrm { C o v } _ { \sigma } \left( Z _ { r } , Z _ { q } \right) } \\ & { \displaystyle \quad \quad + \frac { 1 } { 6 } \sum _ { r , q , v = 0 } ^ { 3 } F _ { r q v } ( \mu ) \mathbb { E } _ { \sigma } \left( W _ { r } W _ { q } W _ { v } \right) + R _ { D } . } \end{array}\tag{54}
$$

Here $F _ { r q }$ and $F _ { r q v }$ denote the respective partial derivatives. Because $| \mathcal { L } _ { 0 } | \le C _ { s }$ , every denominator in (38) is at least of order $C _ { s } .$ Since $C _ { s } / \dot { D }$ is bounded away from zero, the derivatives of orders two, three, and four are ${ \cal O } ( D ^ { - 2 } ) , { \cal O } ( D ^ { - 3 } )$ , and $O ( D ^ { - 4 } )$ , respectively. Expanding products of three and four sums of the form (51), and separating the common index cases in the same way as in (52) and (53), gives

$$
\begin{array} { r } { \mathbb { E } _ { \sigma } \left( W _ { r } W _ { q } W _ { v } \right) = O ( D ) , \mathbb { E } _ { \sigma } \left( \| W \| ^ { 4 } \right) = O ( D ^ { 2 } ) . } \end{array}\tag{55}
$$

It follows from (55) that the third order term in (54) is $O ( D ^ { - 2 } )$ . The fourth order Taylor remainder satisfies $R _ { D } = \dot { O } ( D ^ { - 2 } )$ for the same reason. Thus, up to $O ( \dot { D } ^ { - 2 } )$ ), only $F ( \mu )$ and the covariance term remain in (54).

Now, we can use a computer to find (54). First, compute every covariance term as

$$
\operatorname { C o v } _ { \sigma } \left( Z _ { A B } , Z _ { C E } \right) = \mathbb { E } _ { \sigma } \left( Z _ { A B } Z _ { C E } \right) - \mathbb { E } _ { \sigma } \left( Z _ { A B } \right) \mathbb { E } _ { \sigma } \left( Z _ { C E } \right) ,
$$

using (50) with (52) or (53). Substitute these covariances into the second term of (54), use $C _ { n } =$ $D - C _ { s } - C _ { f }$ , and collect the terms of order $D ^ { - 1 }$ . This gives

$$
\mathrm { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = \left\{ \begin{array} { l l } { \displaystyle - \frac { C _ { f } \{ C _ { s } \left( 2 D - C _ { s } \right) + C _ { f } ( D - C _ { s } ) \} } { C _ { s } ^ { 2 } ( 2 D - C _ { s } ) ^ { 2 } } + O ( D ^ { - 2 } ) , \quad } & { D \mathrm { ~ e v e n ~ a n d } } \\ { \displaystyle - \frac { C _ { f } \left( C _ { s } + C _ { f } \right) } { C _ { s } ^ { 2 } ( 2 D - C _ { s } ) } + O ( D ^ { - 2 } ) , \quad } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{56}
$$

Equation (56) follows by combining (50), the respective draw probabilities, the second moments (52) and (53), and the expansion (54), and noting that $\begin{array} { r } { \operatorname { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = \mathbb { E } _ { \sigma } \left( F ( Z ) \right) - \left( \frac { C _ { f } } { C _ { s } + 1 } \right) ^ { 2 } } \end{array}$ Next, the variance in (47) satisfies

$$
V _ { \mathrm { i n d } } = \frac { C _ { f } ( C _ { s } + C _ { f } ) } { C _ { s } ^ { 2 } } + O ( D ^ { - 1 } ) ,\tag{57}
$$

$$
\frac { V _ { \mathrm { i n d } } } { C _ { s } + C _ { f } } = \frac { C _ { f } } { C _ { s } ^ { 2 } } + O ( D ^ { - 2 } ) .\tag{58}
$$

Substituting (57) and (58) into (56) gives

$$
\mathrm { C o v } _ { \sigma , \pi } \left( \theta _ { s } , \theta _ { s + \Delta } \right) = \left\{ \begin{array} { l l } { - V _ { \mathrm { i n d } } \displaystyle \frac { C _ { s } ( 2 D - C _ { s } ) + C _ { f } ( D - C _ { s } ) } { ( C _ { s } + C _ { f } ) ( 2 D - C _ { s } ) ^ { 2 } } + O ( D ^ { - 2 } ) , \quad } & { D \mathrm { ~ e v e n ~ a n d } } \\ { - V _ { \mathrm { i n d } } \displaystyle \frac { 1 } { 2 D - C _ { s } } + O ( D ^ { - 2 } ) , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

## B.3 Additional Experiments to Verify Variance Approximation

Since the Minner estimator depends on $C _ { s } = f _ { 2 }$ and $C _ { f } = f _ { 1 } - a$ , we investigate edge cases when $f _ { 1 } - a , f _ { 2 }$ are small relative to $D ,$ as $D$ increases. All experiments are done over 1,000 iterations (an order of magnitude less than the plots in Appendix $\mathrm { A . } 5 ,$ , since dealing with large K means more expensive permutations). We can see that the theoretical variance approximations match the empirical MSEs well, as in Figures 9-13.

In the legend of these plots, Counts MSE denotes the empirical MSE of the Minner count estimator, Counts Var denotes the theoretical variance of the Minner count estimator, CP Counts MSE denotes the empirical MSE of the count estimator with circulant permutations, CP Counts Var (S) denotes the theoretical variance approximation when $\begin{array} { r } { \frac { C _ { s } + C _ { f } } { D } \equiv \frac { f _ { 1 } + f _ { 2 } - a } { D }  0 } \end{array}$ , and CP Counts Var (NS) denotes the theoretical variance approximation when $\frac { f _ { 1 } + f _ { 2 } - a } { D }$ does not tend to zero.

Since we are looking at counts when $\frac { f _ { 1 } + f _ { 2 } - a } { D }$ is small, CP Counts Var (S) should be a better approximation. However, we find that both CP Counts Var (S) and CP Counts Var (NS) are equally good estimators as they are indistinguishable from each other.

Unlike the experiments in Appendix A.5, where the theoretical variance curves are almost indistinguishable from the empirical MSEs, we do not expect the count approximations here to overlap perfectly for moderate $D ,$ especially when $\frac { f _ { 1 } + f _ { 2 } - a } { D }$ is small, given that we use a log scale for the y axes in these plots. However, the approximation is still accurate for practical comparisons. In particular, Counts Var gives a useful reference for the amount of Monte Carlo discrepancy expected from 1,000 iterations.

We also plot the theoretical approximation and the empirical MSE for select word pairs from the words dataset [8] in Figure 14.

![](images/0f57fca48f98b66beed37056fef771137464bb5431331e97e27c8c46fe272a71.jpg)

![](images/dcfd08b38159dc24031388190c6ec3a4fca19ec449c08c093b7fec37c0e6c131.jpg)

![](images/883534c30cd250f6a118b8e0f90dff3e4b788b30b371071dd2eddff407623eb5.jpg)

![](images/a3412c2f9dbbf8242416b103483cfc1333bb7f2d18d7dca2424f6ea69d78761f.jpg)  
Figure 9: Empirical MSE vs theoretical approximate variance in Theorem 4 at edge cases when $f _ { i }$ are small respective to D. We use $( f _ { 1 } , { f _ { 2 } } , { \bar { a } } ) = ( 6 , 8 , 2 )$ .

![](images/dd155970ddd44bf444fb6a629b4fb51e1b240a66eb569b0af6f1b1fc1dc396b5.jpg)

![](images/62ee9fbb90e7ed2db6945461fc6511c2718cb36cf3baac3db82b037c8bfc44cf.jpg)

![](images/cf3381b9b7ecab505a00e54a996bfcea3728993679e195eb90f2e2567bb8bd88.jpg)

![](images/7ae884a2709bc5619688727b8779f04f3cd847ce66735e116c0048ea48ca2a02.jpg)  
Figure 10: Empirical MSE vs theoretical approximate variance in Theorem 4 at edge cases when $f _ { i }$ are small respective to D. We use $( f _ { 1 } , f _ { 2 } , \bar { a } ) = ( 1 8 , 2 4 , 6 )$

Both the theoretical approximate variance curves overlaps with the empirical MSEs for all word pairs, even the HONG-KONG word pair, where $\textstyle { \frac { f _ { 1 } - a } { D } } = { \frac { 4 1 } { 6 5 5 3 6 } } \approx 6 . 3 \times 1 0 ^ { - 4 }$ is small.

![](images/b75b172a9bcca37d6566ce90d96951b1c0fad3ad18277953596a7d554a2c4e51.jpg)

![](images/c7a0ee0b8d52e4afb2d00ce915ce10c1e19a738ab720a4ae135307ede251332e.jpg)

![](images/0a87a4de52b38342372b8db5c963e67d27aad9188ef66521530dc5c4d4b0847e.jpg)

![](images/9b128288c5e49005d8b397f1c3f818572b25bbd7abbff4b887531483dcb1f2a7.jpg)  
Figure 11: Empirical MSE vs theoretical approximate variance in Theorem 4 at edge cases when $f _ { i }$ are small respective to D. We use $( f _ { 1 } , f _ { 2 } , a ) = ( 2 4 , 3 2 , 8 )$

![](images/117a7dd3ac097b198ce986e1c0e2a89e57d1bf5782056dfdae5aa1b44fcd39b9.jpg)

![](images/3c8d38544ae0d762e201770c63d7411ee9e7580eaef077d43fa975185ad420de.jpg)

![](images/237cba130a801796e1d317875bd11a7475e9f896bc11572f70d9b109d98f8b13.jpg)

![](images/047dc64f32934899ee6b3fd64a63296e2ca506910bc06766192e8277b9eaf574.jpg)  
Figure 12: Empirical MSE vs theoretical approximate variance in Theorem 4 at edge cases when $f _ { i }$ are small respective to D. We use $( f _ { 1 } , f _ { 2 } , a ) = ( 4 8 , 6 4 , 1 6 )$

![](images/a2fe9e956a09bc9ef53d4df95e28cf4b39cc141aa85a8c1d334b64c0e265edfb.jpg)

![](images/3cb1530a757deb0208edbce1ca20c6bbb909ada2241cbe4535a813fccdd96532.jpg)

![](images/af09d90ccfa7e75dc4170febb64b700ca4865127ce033a9615c6120e906770b8.jpg)

![](images/220c41ee1c751c89643136e9fafb27fdaf9a5da60455466bf6a4a0f507d65be4.jpg)  
Figure 13: Empirical MSE vs theoretical approximate variance in Theorem 4 at edge cases when $f _ { i }$ are small respective to D. We use $( f _ { 1 } , f _ { 2 } , \bar { a } ) = ( 9 6 , 1 2 8 , 3 2 )$ .

![](images/3a41cd610d3c119312b8d87569961cbe08768a328504ee931be7d3a9f894f9b6.jpg)

![](images/767ad0b300909d38b2a436061fb6f02fb8673e938cb73d1a79a1e8cbeb6edd49.jpg)

![](images/ae020c4c924af56de91a8d2df707caa75cea3ed1eb83fe88883d07f756670323.jpg)

![](images/569ffc06fbf19780585924f369d141fdc997d006ed41fbc0c61bdcd271ed5cb7.jpg)

![](images/5588da2a036f198e1a537763f824cabdb88ebbf469ba9f52a0ac389f7eaad5e5.jpg)

![](images/0ce40e53b03de413bfcf6ee5231a35344f6dd342a2ff6ba031ee1433b8444cc7.jpg)

![](images/33a9a9b1ca62d5fa512f5f6693dca2c16cd86ca707f30659d950a0b9627ef1cd.jpg)

![](images/fb909c87c6c8f78e6dabe135746219c9755e5431f790cd2cab5c84025989377e.jpg)

![](images/a8326128210bc163cc8aa31af0f2bcd3a39be529db2216efd592059f7d50e29c.jpg)  
Figure 14: Empirical MSE vs theoretical approximate variance in Theorem 4 for selected word pairs. Both theoretical approximations and empirical MSE lines overlap perfectly.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: Our abstract claims a new approach to approximate the variance of MinHash under circulant permutations. We demonstrate this in Section 2. We also claim that we describe how to apply our approach to different MinHash estimators. We describe how to do so in Section 4 for asymmetric hashing (Minner estimator).

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: The limitations are in the last section of the paper.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

## Answer: [Yes]

Justification: A full and complete proof for all lemmas, theorems, and corollaries are in the appendix. The assumption that D is large is clearly stated.

## Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: Our experiments verify if our approximate variances are close to the empirical MSEs of existing estimators (CMinHash, and circulant permutation applied to the Minner estimator). We review how circulant permutations work in Appendix A1, and give the approximate variances in Theorems 2 and 4.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: Our data is mostly synthetic, and the words dataset is an existing dataset used by [13]. We describe our experiments clearly in Sections 2 and 3. Our experiments are mostly similar to [13], hence code is not provided.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Our data is mostly synthetic, and the words dataset is an existing dataset used by [13]. We describe our experiments clearly in Sections 2 and 3.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [N/A]

Justification: Because we are showing an approximation (which is a fixed value given f , f , a, D) of the actual variance of estimators, no error bars are needed. We compute the respective MSEs for MinHash and Minner estimates averaged over 10,000 iterations. If our approximation was correct, the respective average of the empirical MSEs should overlap with our theoretical variance. Any confidence intervals (or error bars) around the average of the 10,000 empirical MSEs would be of the form MSE ± ϵ. But if our approximate variance overlap the empirical MSEs, then the value of ϵ doesn’t matter in this case (in fact, with 10,000 iterations, any ϵ would be sufficiently small to be mathematically negligible for visual comparison).

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No]

Justification: Our experiments mainly verify that our approximate variance matches empiri cal results, and no large scale compute resources are needed.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We confirm to the NeurIPS Code of Ethics.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

## Answer: [N/A]

Justification: This work serves to make the analysis of the variance of randomized algorithms involving circulant permutations easier.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: This is a theoretical paper.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We cite the algorithm given and use the Words dataset in [13]. To the best of our knowledge, the words dataset first appears in [7].

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [N/A]

Justification: Our approach is theoretical.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: We do not use human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: We do not use human subjects.

## Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

## Answer: [Yes]

Justification: Originally, we did not use an LLM for our research (apart from formatting, phrasing, etc). However, during our rebuttal period (also mentioned on Page 13), we asked 5.6 SOL for a combinatorial implementation of Minner, in order to better craft a rebuttal for our reviewers. It turned out that Li and Li’s variance proof for CMinHash had a slight error. We included the counter example in the paper.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.