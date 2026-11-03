# Training Autoregressive Language Models on Rounded, Bounded, and Censored Numerical Labels via Exact Likelihood

Jingquan Yan, Yuwei Miao, Junzhou Huang

University of Texas at Arlington {jingquan.yan, yuwei.miao, jzhuang}@uta.edu

## Abstract

Numerical labels often specify a set of possible values: a measurement is rounded, a laboratory test reports only a bound, or a program is stopped at a timeout. We introduce boundary marginalization, the first method that trains an autoregressive language model on the exact likelihood of such observations through its existing digit tokens. A fixed-width, order-preserving numeral representation groups consecutive grid values under shared digit prefixes. An interval is a union of such groups, so its exact probability and gradient follow from at most two boundary paths, the successive digit prefixes of its endpoints, each of length logarithmic in the number of grid values. The resulting objective unifies precise, rounded, bounded, and censored supervision and trains digit decisions at the resolution supplied by each observation. The same computation supports exact distributional queries and unbiased gradients for the continuous ranked probability score on exact labels. Experiments show recovery of a conditional distribution from rounded and censored reports alone, improved likelihood on naturally censored molecular measurements, and more accurate code latency distributions on two backbones. Exact interval updates remain tractable as the numerical support grows to one million values.

## 1 Introduction

Numerical observations often report a set of compatible values, which we call a numerical event. A measurement rounded to one decimal place and reported as 1.2 represents the interval [1.15, 1.25). A drug-safety test may report only that a compound’s half-maximal inhibitory concentration (IC50), the concentration at which it inhibits its target by 50%, exceeds the highest concentration tested. A program unfinished at a timeout c has a completion time greater than c. A report that provides only a bound is a censored observation.

Such reports are common and informative. In the cardiac-safety measurements we study, about a third of the labels are bounds (Section 6), and Svensson et al. report that a third or more of the labels are censored in several drug-discovery datasets [20]. Cross-entropy training requires a target numeral, so standard practice either discards these reports or replaces each one with a representative numeral, which the model then learns to reproduce. Training on the total probability of the compatible values uses each report exactly as observed.

Autoregressive language models (LMs) already define probability distributions over numerical values through their digit tokens. LLMTime interprets digit sequences as a hierarchical numerical distribution, and decoding-based regression develops numerical decoder representations and likelihood training [9, 18]. The probability of an individual numeral is directly available, but a reported interval can contain thousands of numerals. How can an LM evaluate and diferentiate their total probability without enumerating them?

Numerical order makes this possible. In a fixed-width numeral written with the most significant digit first, every value below a threshold has a unique first digit at which it becomes smaller. After that digit, all sufixes are allowed, and their conditional probabilities sum to one. Numerals sharing an initial digit sequence form a prefix subtree. Its total probability is just the probability of that prefix. To evaluate a threshold, we need only the next-digit distributions along its boundary path: the successive prefixes of the threshold numeral. Teacher forcing supplies these known prefixes as inputs and reads their next-digit probabilities in one forward pass. We call this computation boundary marginalization: evaluating and diferentiating the exact probability of a numerical interval from at most two teacher-forced boundary paths (Figure 1). For M grid values in base b, the scalar arithmetic is $O ( b \log _ { b } M )$ , independent of the number of compatible values.

A single objective, the likelihood of the observed event, therefore handles precise labels, intervals, and bounds. A singleton (one grid value) recovers sequence cross-entropy (CE), the log score of the complete numeral under the same decoder. An interval supervises its total probability: only prefixes whose subtrees lie partly inside and partly outside the interval receive a direct gradient, while subtrees lying entirely inside it are summed out and need no target digit.

The same computation gives cumulative distribution function (CDF) values for continuous ranked probability score (CRPS) training and exact distributional queries. CRPS integrates the squared diference between the predicted CDF and the step function at the observed value, so it scores predictions by numerical distance: a prediction near the observed value incurs a small loss even when its digits difer. It is a strictly proper score: its expected loss is uniquely minimized by the true distribution. Its training gradient is unbiased when estimated from a few thresholds sampled independently of the model.

Contributions. We introduce boundary marginalization, the first method that trains a pretrained language model on the exact likelihood of rounded, bounded, and censored numerical observations through its existing digit tokens, with no auxiliary output head, imputed target, or sampled rollout.

(i) Boundary marginalization: a stable log-domain algorithm evaluates and diferentiates interval likelihoods using at most two paths of length $d = \log _ { b } M$ (Proposition 1).

(ii) Learning at the observed resolution: a single likelihood handles precise labels, intervals, and bounds, with a local gradient that characterizes the digit decisions supervised by each observation (Proposition 2).

(iii) Exact queries and proper scores: the same computation supports exact tail and quantile queries and unbiased threshold-sampled CRPS gradients (Section 5.4).

Experiments establish the value of incomplete evidence in controlled recovery, naturally censored molecular measurements, and code latency. They demonstrate learning without precise labels, reproduce the latency gains on a second backbone, and measure precision scalability.

## 2 Related Work

Numerical decoders and training through the numeric distribution. LLMTime treats digit sequences as a hierarchical numerical density and samples forecasts [9]. Decoding-based regression studies numerical serialization, cross-entropy training, and decoder-based density estimation [18], and LLM Processes condition numerical prediction on natural-language context [16]. Regression-aware inference estimates Bayes-optimal point predictions from sampled answers [13], and regression-aware fine-tuning (RAFT) diferentiates a squared loss through the expectation of the numeric token distribution over an enumerated grid of candidate values [12]. Reinforcement learning has been used to optimize sequence-level numeric rewards over sampled outputs [6] and CRPS-based distributional rewards over rollouts [14]. Piskorz et al. recover distributional summaries from hidden states with probes trained on sampled outputs [15]. Arithmetic Sampling uses autoregressive prefix masses for arithmetic-code decoding [22]. Our main contribution is exact likelihood learning from incomplete numerical observations through boundary marginalization. CRPS uses the same computation to diferentiate queried CDFs directly, without generating model outputs.

![](images/b161a20a76773625631d926d888bbee49b0af3f0a5e3e64ce334945ab1b18547.jpg)  
Figure 1: Boundary marginalization for $2 3 7 \leq Y \leq 6 8 4$ on the three-digit grid $v _ { k } = k$ , where ∗ denotes an arbitrary digit. Every numeral in the event either is a boundary numeral or first departs from a boundary path at one digit and then continues with an arbitrary sufix. Since the sufix probabilities of a subtree sum to one, the event probability is a sum of prefix masses times sibling digit probabilities, all evaluated at prefixes on the two boundary paths (colored). The model is never evaluated inside a shaded subtree, so the cost is independent of the 448 values in the event.

Digit-level numerical losses. Number Token Loss adds a Wasserstein or squared-error term over the number-token distribution at each position [27]. Numerical Token Integrity Loss (NTIL) combines token-level distance with a diferentiable numerical point construction [7]. Digit Entropy Loss (DEL) reformulates digit-level entropy optimization with binary cross-entropy on digit conditionals [28], and Smooth Maximum Mean Discrepancy (SMMD) matches predicted and target numeric-token distributions using kernels based on distances between numerical values [29]. Strictly proper nonlogarithmic scoring rules have been applied to token conditionals [17], and Calibration Fine-Tuning derives prefix targets from a specified output distribution [2]. This literature improves numerical token supervision and point constructions. Our objective instead assigns probability to the complete observed numerical event, including intervals and bounds for which no exact target numeral is supplied.

Coarsened labels. Heitjan and Rubin formalize observed-data likelihoods for coarsened data and the ignorability conditions under which they are valid [10]. Candidate-set likelihoods of the same form appear in learning with multiple labels [11]. Earlier numerical language models integrate a separate continuous mixture density over a numeral’s rounding interval [19]. Censored likelihood heads have been studied for uncertainty estimation in molecular property prediction [20]. Recent language-model approaches to censored survival outcomes learn relative survival orderings from comparisons between subjects [23, 24]. We train the model’s numerical distribution directly on the censored likelihood through its existing digit tokens and compare against separate distributional output heads trained on the same likelihood (event-aware heads) as primary baselines.

Tree-structured numerical distributions. Hierarchical softmax variants over ordered bins compute leaf probabilities along a path and smooth neighboring leaves [21]. PATH, a concurrent tabular method, learns hierarchical output distributions for interval prediction by combining a base leaf distribution with autoregressive branch refinement [3]. We instead fine-tune a pretrained numeral decoder on observed intervals and bounds, computing their event likelihood by boundary marginalization without constructing the leaf distribution. NumericJev decodes numbers from a structured choice interface at inference time without training [26].

## 3 Learning from Numerical Observations

We represent each observation by the grid values compatible with it and train the LM to assign probability to that set. A fixed-width digit representation places these values in numerical order.

## 3.1 An ordered, fixed-width numerical distribution

We represent numerical outputs on an ordered grid and encode each grid index with a fixed number of digits, starting with the most significant digit. Comparing these strings from left to right preserves numerical order. Let the output support be

$$
\mathcal { V } = \{ v _ { 0 } < \cdots < v _ { M - 1 } \} , \qquad M = b ^ { d } ,\tag{1}
$$

where b is the number of allowed digits, d is the fixed output length, and M is the number of possible grid values. We write K for the grid index and $Y = v _ { K }$ for its numerical value. The uniform grid $v _ { k } = a + \Delta k$ with $\Delta > 0$ is the default. The same computation supports any strictly increasing grid, including a log-spaced grid for positive targets spanning several orders of magnitude. Let $z ( k ) = ( z _ { 1 } , \dots , z _ { d } )$ be the base-b expansion of $k ,$ padded with leading zeros, with $z _ { 1 }$ the most significant digit.

We choose b existing token identifiers (IDs) representing the digits. For input x and digit prefix s (the digits already supplied), let $\ell _ { \theta } ( x , s ) _ { a }$ be the corresponding LM logit. The next-digit distribution is a softmax restricted to the b digit tokens,

$$
p _ { \theta } ( a \mid x , s ) = { \frac { \exp \ell _ { \theta } ( x , s ) _ { a } } { \sum _ { c = 0 } ^ { b - 1 } \exp \ell _ { \theta } ( x , s ) _ { c } } } , \qquad a \in \{ 0 , \ldots , b - 1 \} ,\tag{2}
$$

and the probability of a complete numeral is the product of its digit probabilities,

$$
p _ { \theta } ( k \mid x ) = \prod _ { j = 1 } ^ { d } p _ { \theta } ( z _ { j } ( k ) \mid x , z _ { < j } ( k ) ) .\tag{3}
$$

Normalization over the b digit tokens and deterministic termination after d digits ensure that (3) sums to one over the M grid values. We call it the LM’s constrained numeral distribution. The task prompt declares the mapping from grid indices to numerical values, the support, and the units. Numerals are built directly from token IDs, independently of whether the tokenizer would merge adjacent characters. The LM predicts the target numeral directly using its existing backbone and vocabulary.

## 3.2 Observations as compatible sets

A training observation is $( x , A )$ , where $A \subseteq \{ 0 , \ldots , M - 1 \}$ contains the grid indices compatible with what was observed. We call A the observed event and focus on contiguous intervals $A = [ l , u ] \cap \mathbb { Z }$ Table 1 defines the common cases on a uniform grid. An exact value is represented at the declared modeling resolution. When the reporting precision is coarser than that resolution, the observation is represented by its rounding interval.
<table><tr><td>Observation</td><td>Numerical event</td><td colspan="4">Compatible grid indices</td></tr><tr><td>Exact grid label  $v _ { r }$ </td><td> $Y = v _ { r }$ </td><td></td><td></td><td> $l = u = r$ </td><td></td></tr><tr><td>Closed interval</td><td> $L \leq Y \leq U$ </td><td></td><td> $l = \lceil ( L - a ) / \Delta \rceil$ </td><td></td><td> $u = \lfloor ( U - a ) / \Delta \rfloor$ </td></tr><tr><td>Rounded to width h</td><td> $r - h / 2 \leq Y < r + h / 2$ </td><td></td><td></td><td> $l = \lceil ( r - h / 2 - a ) / \Delta \rceil$ </td><td> $\boldsymbol { u } = \lceil ( \boldsymbol { r } + h / 2 - \boldsymbol { a } ) / \Delta \rceil - 1$ </td></tr><tr><td>Strict lower bound</td><td> $Y > c$ </td><td></td><td></td><td> $l = \lfloor ( c - a ) / \Delta \rfloor + 1$ </td><td> $u = M - 1$ </td></tr><tr><td>Strict upper bound</td><td> $Y < c$ </td><td></td><td> $l = 0$ </td><td></td><td> $u = \lceil ( c - a ) / \Delta \rceil - 1$ </td></tr></table>

Table 1: Event construction. Indices are intersected with the declared support. Rounding uses the half-open convention shown, and other tie rules are handled the same way. Endpoints are converted with decimal or rational arithmetic (Appendix C).

## 3.3 One likelihood for all observations

An event’s probability is the sum of the probabilities of its compatible grid values. We call this sum the event likelihood and, for $A = [ l , u ] \cap \mathbb { Z }$ , write $\operatorname { L o g M a s s } _ { \theta } ( l , u \mid x )$ for its logarithm. We train by minimizing the negative log event likelihood:

$$
\operatorname { \mathcal { L } } _ { \mathrm { e v e n t } } ( \theta ; x , A ) = - \log \sum _ { k \in A } p _ { \theta } ( k \mid x ) = - \operatorname { \mathrm { L o g M a s s } } _ { \theta } ( l , u \mid x ) .\tag{4}
$$

A singleton reduces exactly to sequence negative log likelihood (NLL), rounded measurements supervise bins, and bounds supervise tails. All observation types can appear in the same minibatch with one loss. Section 4 computes this objective and its gradient without enumerating the compatible numerals.

The event likelihood and cross-entropy on an imputed numeral have diferent targets. Suppose a reporting rule maps an observation A to a representative $r ( A )$ . Cross-entropy on $r ( A )$ learns the distribution of the representatives, whereas (4) learns the probabilities of the compatible events. For fixed rounding bins, both objectives can match the bin masses, but imputation additionally concentrates targets at the reported centers. Event likelihood imposes only the observed probability constraints. Mixed reporting resolutions and heterogeneous bounds make this distinction central to learning a numerical distribution.

Likelihood interpretation. Conditionally ignorable reporting means that, for fixed x, all values compatible with a report have the same probability of producing it [10]. The probability of report A given the underlying index k then has the form

$$
q ( A \mid k , x ) = c ( A , x ) \mathbf { 1 } \{ k \in A \} , \qquad \sum _ { A \ni k } c ( A , x ) = 1 ,\tag{5}
$$

where c is independent of θ. The probability of observing report A is then $c ( A , x ) \sum _ { k \in A } p _ { \theta } ( k \mid x )$ , whose negative log likelihood equals (4) up to the constant $- \log c ( A , x )$ . This includes deterministic rounding and conditionally independent censoring. Known reporting conditions can enter x. Appendix B.3 characterizes the distributional information identified by these reports.

## 4 Exact Boundary Marginalization

We evaluate (4) by summing complete prefix subtrees, using only the event’s boundary paths.

## 4.1 One path evaluates a tail probability

For a threshold index k with digit string $z = z ( k )$ , define the prefix mass, the probability that a numeral begins with the first $j - 1$ threshold digits,

$$
P _ { \theta } ( z _ { < j } ) = \prod _ { i < j } p _ { \theta } ( z _ { i } \mid x , z _ { < i } ) , \qquad P _ { \theta } ( \emptyset ) = 1 ,\tag{6}
$$

where the input x is omitted from symbols whose dependence on it is clear from context. The inclusive CDF is $F _ { \theta } ( k \mid x ) = P _ { \theta } ( Y \le v _ { k } \mid x )$ . Every numeral below the threshold matches z up to some position $j$ , takes a smaller digit there, and then continues with an unrestricted sufix whose total probability is one. Grouping numerals by that position and adding the threshold numeral itself gives

$$
F _ { \theta } ( k \mid x ) = \sum _ { j = 1 } ^ { d } P _ { \theta } ( z _ { < j } ) \sum _ { a < z _ { j } } p _ { \theta } ( a \mid x , z _ { < j } ) + p _ { \theta } ( k \mid x ) ,\tag{7}
$$

and the same grouping with larger digits gives the strict survival probability, the probability of exceeding the threshold,

$$
S _ { \theta } ( k \mid x ) = P _ { \theta } ( Y > v _ { k } \mid x ) = \sum _ { j = 1 } ^ { d } P _ { \theta } ( z _ { < j } ) \sum _ { a > z _ { j } } p _ { \theta } ( a \mid x , z _ { < j } ) .\tag{8}
$$

All d conditional digit distributions in these expressions come from a single teacher-forced path, which a causal transformer evaluates in one forward pass. The threshold is given by the observation or query, so no model output is sampled.

## 4.2 Stable interval mass from two paths

Computing $F _ { \theta } ( u ) - F _ { \theta } ( l - 1 )$ erases a small interval probability whenever both CDFs round to the same floating-point number. We therefore write the interval as a disjoint union of prefix subtrees and sum their masses in the log domain.

For $l < u .$ , let ${ \mathit { l } } ^ { \prime } = z ( l )$ and $u ^ { \prime } = z ( u )$ be the boundary digit strings, and let r be their first difering position, indexed from 1. As illustrated in Figure 1, the event consists of four disjoint groups: (1) the two boundary numerals, (2) subtrees that share the boundary prefix before r and choose a digit strictly between the two boundary digits at $r , ~ ( 3 )$ subtrees that first depart from the lower boundary by choosing a larger digit at a position after r, and (4) subtrees that first depart from the upper boundary by choosing a smaller digit at a position after r. Each included subtree admits every sufix, so its total probability is the probability of reaching its prefix. For a digit string t, define its log prefix masses and the log probability of the complete string,

$$
H _ { j } ( t ) = \sum _ { i < j } \log p _ { \theta } ( t _ { i } \mid x , t _ { < i } ) , \qquad H _ { d + 1 } ( t ) = \log p _ { \theta } ( t \mid x ) .\tag{9}
$$

For sibling subtrees at a shared prefix, the total probability is the prefix mass times the sum of the included next-digit probabilities. Collecting the log masses of the boundary numerals and these

disjoint subtree groups in the same order gives

$$
\begin{array} { l } { T ( l , u ) = \big ( H _ { d + 1 } ( l ^ { \prime } ) , H _ { d + 1 } ( u ^ { \prime } ) \big ) } \\ { \displaystyle \qquad \mathrm { ~ } \mathrm { ~ } \forall l \left( H _ { r } ( l ^ { \prime } ) + \log \underset { a < l _ { r } ^ { \prime } + 1 } { \overset { u ^ { \prime } - 1 } { \sum } } p _ { \theta } ( a \mid x , l _ { < r } ^ { \prime } ) \right) } \\ { \displaystyle \qquad \mathrm { ~ } \forall l \left( H _ { j } ( l ^ { \prime } ) + \log \underset { a \leq l _ { j } ^ { \prime } } { \sum } p _ { \theta } ( a \mid x , l _ { < j } ^ { \prime } ) \right) \displaylimits _ { j = r + 1 } ^ { d } } \\ { \displaystyle \qquad \mathrm { ~ } \forall l \left( H _ { j } ( u ^ { \prime } ) + \log \underset { a < l _ { j } ^ { \prime } } { \sum } p ( a \mid x , u _ { < j } ^ { \prime } ) \right) \displaybreaks _ { j = r + 1 } ^ { d } } \\ { \displaystyle \qquad \mathrm { ~ } \forall l \left( H _ { j } ( u ^ { \prime } ) + \log \underset { a < l _ { j } ^ { \prime } } { \sum } p _ { \theta } ( a \mid x , u _ { < j } ^ { \prime } ) \right) \displaybreaks _ { j = r + 1 } ^ { d } , } \end{array}\tag{10}
$$

where ∥ denotes tuple concatenation and empty digit ranges are omitted. With LSE denoting log-sum-exp, the interval log probability is

$$
\operatorname { L o g M a s s } _ { \boldsymbol { \theta } } ( l , u \mid x ) = \operatorname { L S E } ( { \mathcal { T } } ( l , u ) ) .\tag{11}
$$

For $l = u ,$ the log probability is $H _ { d + 1 } ( z ( l ) )$ . For the full support it is zero. One-sided intervals use $\operatorname { L o g M a s s } _ { \boldsymbol { \theta } } ( 0 , u ) = \log F _ { \boldsymbol { \theta } } ( u )$ or $\operatorname { L o g M a s s } _ { \boldsymbol { \theta } } ( l , M - 1 ) = \log S _ { \boldsymbol { \theta } } ( l - 1 )$ for $l > 0 .$ , so one path sufices. An empty event is rejected. For $2 3 7 \leq Y \leq 6 8 4$ in Figure 1, these four groups are {237, 684}, $\{ 3 * * , 4 * * , 5 * * \} , \{ 2 4 * , \ldots , 2 9 * , 2 3 8 , 2 3 9 \}$ , and $\left\{ 6 0 * , \ldots , 6 7 * , 6 8 0 , \ldots , 6 8 3 \right\}$ . Every required next-digit distribution is evaluated at a prefix of 237 or 684.

Proposition 1 (Exactness and query complexity). For (3), equations $( 7 ) - ( 1 1 )$ compute the exact event probabilities in real arithmetic. One tail uses at most d prefix conditionals and one interval at most 2d. Grouped interval evaluation uses at most $2 d + 1$ log-mass terms and $O ( b d ) = O ( b \log _ { b } M )$ scalar arithmetic, with no enumeration of the support or interval interior. Automatic diferentiation gives the exact gradients of the event probability and event log likelihood.

## 4.3 Which digits an interval supervises

On the three-digit grid $v _ { k } = k$ , the event $4 4 0 \leq Y \leq 4 4 9$ consists of every completion of 44. Its probability is simply the prefix probability: the loss supervises the first two digits but supplies no target for the third.

More generally, let $p _ { \theta } ( k \mid x , A )$ be the model distribution restricted to compatible indices and renormalized to sum to one. The gradient of the event loss is the negative average gradient of the compatible numerals’ log probabilities, weighted by their probabilities conditional on A:

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { e v e n t } } = - \mathbb { E } _ { K \sim p _ { \theta } ( \cdot \vert x , A ) } [ \nabla _ { \theta } \log p _ { \theta } ( K \mid x ) ] .\tag{12}
$$

Boundary marginalization computes this average exactly along the boundary paths without enumerating or sampling compatible numerals.

For a prefix s, let $C _ { s }$ contain the indices whose numerals extend s, and let sa denote that prefix followed by digit a. Write $\ell _ { s } ( a ) = \ell _ { \theta } ( x , s ) _ { a }$ for its next-digit logit. Conditioned on the event $A _ { i }$ , let $w _ { s }$ be the posterior probability of reaching prefix s and $t _ { s } ( a )$ the posterior probability that the next digit after s is $a ,$

$$
w _ { s } = P _ { \theta } ( C _ { s } \mid A , x ) , \qquad t _ { s } ( a ) = P _ { \theta } ( C _ { s a } \mid C _ { s } , A , x ) ,\tag{13}
$$

with the convention $t _ { s } = p _ { \theta } ( \cdot \mid x , s )$ when $w _ { s } = 0$ . The derivative with respect to the next-digit logits at s then has the usual softmax cross-entropy form, with $t _ { s }$ as the target distribution and $w _ { s }$ as its weight.

Proposition 2 (Gradient at the observed resolution). Holding the logits at other prefixes fixed, the derivative with respect to the next-digit logit $\ell _ { s } ( a )$ is

$$
\frac { \partial \mathcal { L } _ { \mathrm { e v e n t } } } { \partial \ell _ { s } ( a ) } = w _ { s } \big [ p _ { \theta } ( a \mid x , s ) - t _ { s } ( a ) \big ] .\tag{14}
$$

If $C _ { s } \subseteq A$ or $C _ { s } \cap A = \emptyset$ , this derivative is zero. Consequently, an interval event directly supervises only prefix nodes whose subtrees straddle a boundary, and $| \partial \mathcal { L } _ { \mathrm { e v e n t } } / \partial \ell _ { s } ( a ) | \le w _ { s } \le 1$ . The chain rule propagates these derivatives to the shared LM parameters θ.

The two boundary paths contain every prefix that can receive a nonzero local gradient. Boundary marginalization therefore preserves the exact gradient without visiting the interval interior.

What the labels identify. Event likelihood matches the distribution of the observed reports. Reports from fixed, disjoint rounding bins identify the total probability of each bin, and finer or overlapping reports can additionally distinguish probabilities within a bin. Appendix B.3 states this as an identity for the expected loss and gives the identifiability condition. This motivates our controlled experiment, which varies the precise-label probability down to zero while retaining rounding intervals and bounds $Y \le c$ or $Y > c$ at thresholds sampled independently of Y.

## 5 Implementation, Inference, and CRPS Training

The same boundary computation supports event likelihoods, numerical queries, and CRPS training.   
We describe how teacher forcing supplies the required digit probabilities and how each task uses them.

## 5.1 Teacher-forced probability extraction

Given prompt tokens $x _ { 1 } , \ldots , x _ { n }$ and threshold digits $z _ { 1 } , \ldots , z _ { d } .$ , feed the prompt followed by $z _ { 1 } , \dotsc , z _ { d - 1 }$ Read causal next-token logits at zero-based positions $n - 1 , \ldots , n + d - 2$ , select the b digit-token columns, and apply log-softmax in 32-bit floating-point (FP32) arithmetic. This provides all d conditional distributions for one path. The selected paths are evaluated together in a single batched forward pass. Appendix D verifies this computation and its gradients against exhaustive enumeration.

Singleton and one-sided events require one path, interior intervals at most two, and full-support events contribute zero loss. Dropout is disabled so shared prefixes define one coherent distribution, while automatic diferentiation remains enabled.

Cost. Boundary marginalization combines one or two paths of depth $d = \log _ { b } M$ using $O ( b d )$ scalar arithmetic. Our reference implementation repeats the prompt for each path and uses ordinary causal attention. A layout that shares the prompt and the common digit prefix between the two paths adds only $d - r \leq d - 1$ input positions for the second boundary, where r is the first difering digit. The method supports full fine-tuning and parameter-eficient updates such as low-rank adaptation (LoRA).

## 5.2 Algorithmic summary

Algorithm 1 summarizes one training update from observed events.

Algorithm 1 Event-supervised numerical update   
1. Convert each observation into a nonempty compatible interval $[ l , u ]$ on the declared grid.   
2. Return zero loss for the full support. Otherwise select the singleton path, the tail threshold, or the   
two interval boundaries.   
3. Teacher-force the selected paths with repeated prompts and extract locally normalized digit log   
probabilities.   
4. Use the appropriate log tail or sum the subtree masses in (10) with log-sum-exp.   
5. Average − $- \operatorname { L o g M a s s } _ { \theta } ( l , u \mid x )$ over examples and backpropagate through the selected paths.   
Event-likelihood training by boundary marginalization. The CRPS update uses the same computation   
(Algorithm 2, Appendix C.2).

## 5.3 Exact numerical queries at inference

The trained model answers distributional queries without sampling. A tail probability $P ( Y > c \mid x )$ for example the probability that a program exceeds a latency budget, is one boundary path in one forward pass. An interval probability uses at most two paths. The α-quantile is the smallest grid value with cumulative probability at least α. We find it by choosing digits from most to least significant. Initialize $\tau = \alpha$ for the desired level $\alpha \in ( 0 , 1 )$ . At each prefix, choose the first digit a whose cumulative child probability reaches τ . To express the target level within the chosen subtree, subtract the mass of preceding sibling subtrees and divide by the chosen digit’s probability:

$$
\tau  { \frac { \tau - \sum _ { c < a } p _ { \theta } ( c \mid x , s ) } { p _ { \theta } ( a \mid x , s ) } } .\tag{15}
$$

After d sequential digit steps, this traversal returns the lower α-quantile, with a fixed convention at probability boundaries. The same quantile can be obtained by deterministic CDF bisection with at most $\lceil \log _ { 2 } M \rceil$ boundary queries, as used for the reported medians. An equal-tailed prediction interval uses two quantile searches.

## 5.4 Threshold-sampled CRPS training

Boundary marginalization also supports CRPS training on exact labels. CRPS integrates the squared diference between the predicted CDF and the step function that jumps from zero to one at the observed value. For an exact grid label $y = v _ { r }$ , define $h _ { k } ( r ) = { \bf 1 } \{ r \leq k \}$ and $\Delta _ { k } = v _ { k + 1 } - v _ { k }$ . The CRPS of the discrete output distribution is

$$
\mathcal { L } _ { \mathrm { C R P S } } ( \theta ; x , v _ { r } ) = \sum _ { k = 0 } ^ { M - 2 } \Delta _ { k } \bigl [ F _ { \theta } ( k \mid x ) - h _ { k } ( r ) \bigr ] ^ { 2 } .\tag{16}
$$

We estimate this sum by sampling a few thresholds and weighting each term by the inverse of its sampling probability. Choose a sampling distribution $\pi ( k \mid x , y ) > 0$ over thresholds $k \in \{ 0 , \ldots , M - 2 \}$ that is fixed with respect to θ. For independent thresholds $K _ { 1 } , \ldots , K _ { m } \sim \pi$ , use

$$
\widehat { \mathcal { L } } _ { \mathrm { C R P S } } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \frac { \Delta _ { K _ { j } } } { \pi ( K _ { j } \mid x , y ) } \big [ F _ { \theta } ( K _ { j } \mid x ) - h _ { K _ { j } } ( r ) \big ] ^ { 2 } .\tag{17}
$$

Each sampled threshold requires one teacher-forced boundary path. Both the loss estimator and its ordinary backpropagation gradient are unbiased for the full CRPS (Proposition 4). Appendix C.2

gives the numerical-distance interpretation, the sampling choices, stable residual evaluation, and Algorithm 2. Proposition 5 establishes the variance reduction relative to a matched estimator that samples model outputs.

## 6 Experiments

We test how event likelihood converts incomplete observations into distributional supervision. Three settings cover recovery of a known conditional distribution, naturally censored molecular reports, and code latency on the subset of Code-Regression labeled KBSS, which contains Triton kernels for GPUs [1]. The latency study uses constructed training timeouts and fully observed test outcomes. We then test a second backbone, compare proper scores on exact labels, and measure precision scalability.

The primary backbone is Qwen2.5-1.5B, and the cross-backbone study uses Qwen3-4B-Base. All learning comparisons use rank-16 LoRA, a learning rate of $1 0 ^ { - 4 }$ , and a four-digit numerical grid.

We compare five methods on incomplete labels (Table 6, Appendix E). Event likelihood is the proposed objective. Complete-case CE trains the same decoder with sequence CE on the precise labels only, and imputed-label CE trains it on one representative numeral per report. The two event-aware heads keep the same backbone and the same event likelihood but replace the digit decoder with a five-component Gaussian mixture or a 256-bin histogram output. The CE controls test the value of incomplete reports and their event representation, and the heads compare output parameterizations given the same reports.

Within each comparison, methods share backbone initialization, adapter placement, optimizer settings, update budget, and validation schedule. The head baselines add their declared output parameters, and complete-case CE repeats its precise subset to match the update budget. The incomplete-label studies select checkpoints by event NLL across all validation reports. The exact-label score comparison selects by validation CRPS. Appendix F provides supporting scores, individual runs, and integration checks.

## 6.1 Learning from incomplete numerical evidence

Controlled distribution recovery. We construct $8 , 0 0 0 / 1 , 0 0 0 / 2 , 0 0 0$ training/validation/test examples with inputs drawn uniformly on $[ - 1 , 1 ]$ and numerical outcomes drawn from an inputdependent two-component Gaussian mixture truncated to [−2, 2]. The same latent draws are reported with nominal precise-label probabilities $\rho \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 5 , 1 \}$ . The remaining observations provide rounding intervals or one-sided bounds at thresholds sampled independently of the outcome. Integrated squared CDF error (ISE) measures recovery by integrating the squared diference between the predicted and true conditional CDFs over the support.

Figure 2 compares recovery across precise-label probabilities, with ISE values reported below in units of $1 0 ^ { - 3 }$ . As precise labels become scarce, complete-case CE degrades while event likelihood remains within a factor of two of the full-label reference, 3.819 at $\rho = 1$ . At $\rho = 0 . 1$ , event likelihood achieves mean ISE 6.993, compared with 15.923 for complete-case CE and 29.156 for imputed-label CE, reducing error by 56.1% relative to complete-case CE. It improves on both CE controls in all three paired seeds. With no precise training labels $( \rho = 0 )$ , rounding and threshold reports alone yield mean ISE 6.801, an 80.1% reduction from imputed-label CE (34.121). Complete-case CE is unavailable in this setting.

Event likelihood has lower mean ISE than both event-aware heads at every $\rho \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 5 \}$ while using the model’s existing digit tokens. At $\rho = 0 . 5$ , its mean ISE is 4.936, compared with 6.037 for the mixture head and 5.312 for the histogram head. Imputation performs worse than discarding incomplete reports at $\rho = 0 . 1 , 0 . 2$ , and 0.5, consistent with Section 3: cross-entropy on imputed numerals learns the distribution of the representatives. The five-seed follow-up at $\rho = 0 . 2$ gives mean ISE 4.818 for event likelihood and 7.906 for complete-case CE, a 39.1% reduction.

At the test input nearest $x = 0$ , event likelihood captures both rises and the central plateau of the true conditional CDF in all three seeds (Figure 3). Complete-case CE produces central plateaus near 0.73, 0.28, and 0.29 for seeds 0, 1, and 2, compared with the true 0.50. Imputed-label CE shows large jumps near rounding bin centers and broader tails, consistent with its use of representative numerals as targets (Section 3).

Naturally censored molecular measurements. We retain 14,850 IC50 measurements of human ether-à-go-go-related gene (hERG) channel inhibition, a standard cardiac safety assay, from ChEMBL 37 [5]. We transform concentrations and bounds to pIC50, the negative base-10 logarithm of molar IC50. We split the reports into 11,880 training, 1,485 validation, and 1,485 test reports so that no parent molecule or scafold (core molecular structure) is shared across splits. Bounds constitute 29.8% of training reports and 36.2% of test reports, and the remaining reports are equalities, which give an exact value.

Event likelihood incorporates 3,535 bounded training reports excluded by complete-case CE and achieves the lowest NLL over all test reports, 4.675, improving over complete-case CE by 0.408 nats per report and over both event-aware heads (Table 2). Its NLL on censored reports is 0.823 versus 2.076 for complete-case CE, a 60.3% reduction. On equality reports, its NLL is below imputed-label CE and both heads, and 0.071 nats higher than complete-case CE (6.857 versus 6.786), which trains only on equality reports. Table 10 (Appendix F) reports NLL by bound type, equality CRPS, and the mean absolute error (MAE) of predicted medians.

Code latency from timeout supervision. KBSS pairs each Triton GPU kernel with its measured latency. We split it by code template into 5,949 training, 744 validation, and 744 test examples. Independently assigned timeouts censor 45.2% of training outcomes. Every test latency is retained exactly.

Event likelihood obtains the lowest CRPS on both scales: $8 . 2 6 2 \times 1 0 ^ { - 3 }$ ms and 0.0878 in $\log _ { 1 0 } ( \mathrm { m s } )$ Physical and logarithmic CRPS decrease by 6.8% and 10.5% relative to complete-case CE, and by 8.1% and 1.9% relative to the event-aware histogram. Both CRPS scores improve against every control in each of the three paired seeds (Figure 6, Appendix F). For the event that a kernel exceeds a fixed latency budget, the Brier score measures the mean squared error between the predicted probability and the observed binary outcome. It is 0.1281 versus 0.1550 for complete-case CE, showing more accurate predictions for this tail event (Table 11).

## 6.2 Transfer to a newer, larger backbone

We repeat the event, complete-case CE, and histogram comparison on Qwen3-4B-Base [25], preserving the KBSS split, timeout assignments, prompts, numerical support, 558-update budget, and validation rule. Event likelihood again achieves the lowest CRPS on both scales (Table 3). Physical and logarithmic CRPS decrease by 5.3% and 11.5% relative to complete-case CE, and by 4.3% and 4.0% relative to the histogram head. Its budget Brier score is also lowest. These results reproduce the benefit of event supervision on a second pretrained backbone.

![](images/f7a54eaad70a31637041ad7d16dc50f932c8d4ff8ccbff251e24eced9f46e85f.jpg)  
Figure 2: Distribution recovery across precise-label probabilities. Large markers show mean ISE and error bars show one sample standard deviation (SD) over seeds 0, 1, and 2. Faint markers show individual seeds, ofset horizontally for readability. Lower is better. The horizontal axis is linear and the vertical axis logarithmic. Event-aware heads are evaluated at $\rho \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 5 \}$ , while complete-case CE is unavailable at $\rho = 0 . { \mathrm { ~ A t ~ } } \rho = 1$ , the three decoder objectives share one full-label reference, whose mean is marked by the dashed line. ISE uses 64 shared stratified thresholds per test input (Appendix E). Results for individual seeds appear in Figure 5.

## 6.3 Exact-label score comparison

Boundary marginalization also makes CRPS trainable (Section 5.4), so we test whether CRPS should replace the likelihood as the training objective when every label is exact. With exact labels, event likelihood reduces to sequence CE, which supervises all d digit conditionals of the observed numeral without sampling thresholds. On the fully precise controlled task, we compare sequence CE with threshold-sampled CRPS and binary cross-entropy on threshold events (CDF-BCE) through the same decoder at a common 2,250-update budget, with checkpoints selected by validation CRPS for every method.

Sequence CE attains the lowest mean test CRPS (0.2165), ISE, and point NLL (the NLL of the complete precise test numeral), supporting event likelihood as the primary training objective (Table 13, Appendix F). Querying only $m = 2$ of the $M - 1 = 9 { , } 9 9 9$ grid thresholds per observation, threshold CRPS achieves mean test CRPS 0.2197, about 1.5% above sequence CE. It has lower means than CDF-BCE (0.2219 test CRPS) on all three metrics. The same boundary computation thus supports direct CRPS training without generated rollouts.

## 6.4 Precision scalability

We hold physical support and interval observations fixed while increasing decimal depth from $d = 2$ to 6, expanding the output support from $1 0 ^ { 2 }$ to $1 0 ^ { 6 }$ values. Event update time and peak memory remain nearly constant across the tested depths. With 256 attended prompt tokens, updates take 0.872–0.888 seconds and use 11.224–11.343 GiB of peak allocated memory. With 1,024 tokens, they take 2.102–2.144 seconds and use 27.008–27.458 GiB. Exact interval training therefore remains tractable across this $1 0 ^ { 4 } .$ -fold increase in support size.

![](images/dc6ec07cdacab742de2b298e663bfa65a70b30df3e439990c5959fa55dbb8fb3.jpg)

Figure 3: Conditional CDFs at $\rho = 0 . 1$ for the test input nearest $x = 0 \ ( x = - 0 . 0 0 0 4 1 6 )$ , with one panel per training seed. Curves use checkpoints selected by validation event NLL and connect evaluations at 512 common grid thresholds. The dashed curve is the true grid CDF.
<table><tr><td></td><td colspan="2">hERG: natural reports</td><td colspan="2">KBSS: exact test outcomes</td></tr><tr><td>Method</td><td>Full NLL</td><td>Equality NLL</td><td> $\mathrm { C R P S } ~ ( 1 0 ^ { - 3 } ~ \mathrm { m s } )$ </td><td>CRPS  $( \log _ { 1 0 }$  ms)</td></tr><tr><td>Event likelihood (ours)</td><td> $\mathbf { 4 . 6 7 5 { \scriptstyle \pm 0 . 0 1 2 } }$ </td><td> $6 . 8 5 7 { \scriptstyle \pm 0 . 0 4 9 }$ </td><td> $\mathbf { 8 . 2 6 2 \pm 0 . 2 9 9 }$ </td><td> $\mathbf { 0 . 0 8 7 8 { \scriptstyle \pm 0 . 0 0 1 3 } }$ </td></tr><tr><td>Complete-case CE</td><td> $5 . 0 8 3 { \pm } 0 . 0 5 5$ </td><td> ${ \bf 6 . 7 8 6 \pm 0 . 0 6 2 }$ </td><td> $8 . 8 7 0 { \scriptstyle \pm 0 . 2 3 5 }$ </td><td> $0 . 0 9 8 0 { \pm } 0 . 0 0 2 4$ </td></tr><tr><td>Imputed-label CE</td><td> $4 . 9 5 8 { \pm } 0 . 0 3 2$ </td><td> $6 . 9 9 8 { \pm } 0 . 0 8 6$ </td><td> $1 0 . 2 2 9 { \pm } 0 . 1 3 5$ </td><td> $0 . 1 2 0 5 { \pm } 0 . 0 0 2 6$ </td></tr><tr><td>Event-aware mixture</td><td> $5 . 3 3 8 { \pm } 0 . 0 2 3$ </td><td> $7 . 8 8 0 { \pm } 0 . 0 2 7$ </td><td> $1 1 . 3 7 4 { \scriptstyle \pm 0 . 6 2 6 }$ </td><td> $0 . 0 9 7 3 { \scriptstyle \pm 0 . 0 0 3 1 }$ </td></tr><tr><td>Event-aware histogram</td><td> $5 . 4 1 3 { \pm } 0 . 0 8 7$ </td><td> $7 . 9 3 4 { \pm } 0 . 0 5 3$ </td><td> $8 . 9 9 0 { \scriptstyle \pm 0 . 7 8 0 }$ </td><td> $0 . 0 8 9 5 { \pm } 0 . 0 0 2 2$ </td></tr></table>

Table 2: Incomplete-label learning on real inputs with Qwen2.5-1.5B. Entries are means ± sample standard deviation (SD) over seeds 0, 1, and 2. Lower is better, and bold marks the lowest mean. hERG full NLL covers all 1,485 test reports (948 equalities and 537 bounds), and equality NLL covers the equalities only. KBSS uses all 744 exact test latencies. Methods share prompts, grid, update budget, and checkpoint selection by validation event NLL within each task.

At $d = 6 .$ , the dense per-value output, which predicts one probability per grid value, uses 35.849 and 47.513 GiB for the two prompt lengths. Event likelihood uses 68.4% and 43.2% less memory, respectively (Figure 4). Event likelihood updates take 1.07–1.10 times as long as imputed-label CE updates with 256 prompt tokens and 1.45–1.47 times as long with 1,024 tokens. Appendix F.5 reports the full grid, including fixed-size event-aware heads.

1,024 attended prompt tokens
<table><tr><td>Method</td><td> $\mathrm { C R P S } ~ ( 1 0 ^ { - 3 } ~ \mathrm { m s } )$ </td><td> $\mathrm { C R P S } ~ ( \log _ { 1 0 } \mathrm { m s } )$ </td><td>Budget Brier</td></tr><tr><td>Event likelihood (ours)</td><td> ${ \bf 8 . 3 9 5 \pm 0 . 2 7 4 }$ </td><td> $\mathbf { 0 . 0 8 7 8 { \scriptstyle \pm 0 . 0 0 1 8 } }$ </td><td> $\mathbf { 0 . 1 3 2 7 { \scriptstyle \pm 0 . 0 0 5 9 } }$ </td></tr><tr><td>Complete-case CE</td><td> $8 . 8 6 2 { \pm } 0 . 1 1 9$ </td><td> $0 . 0 9 9 2 { \scriptstyle \pm 0 . 0 0 3 1 }$ </td><td> $0 . 1 5 4 1 { \scriptstyle \pm 0 . 0 0 6 7 }$ </td></tr><tr><td>Event-aware histogram</td><td> $8 . 7 7 6 { \pm } 0 . 4 8 8$ </td><td> $0 . 0 9 1 5 { \pm } 0 . 0 0 0 8$ </td><td> $0 . 1 4 0 9 { \pm } 0 . 0 0 3 6$ </td></tr></table>

Table 3: KBSS transfer to Qwen3-4B-Base. Entries are means ± sample SD over seeds 0, 1, and 2 on all 744 exact test outcomes. Both CRPS scales and the evaluation budget fixed from the training data match the Qwen2.5 evaluation. Lower is better, and bold marks the lowest mean. Table 12 reports median MAE on both scales.

![](images/1a69bc5995c9ff263b94d07e6e8a32f993182428bde0a5aa97637429fa4ad885.jpg)

![](images/c0f9d4f88263128e8250c7312481b0f74b8a24f132b12337b150a30657ead513.jpg)  
Figure 4: Precision scaling without dense output growth. Depths 2, 4, 6 represent $1 0 ^ { 2 } , 1 0 ^ { 4 } , 1 0 ^ { 6 }$ values on the same support. Methods share inputs and optimizer settings. Event likelihood and the three heads use the same interval reports, and imputed-label CE trains with sequence CE on one representative numeral per report. Memory is maximum peak allocated memory across three benchmark replicates, including weights, gradients, and Adam state. The dense per-value output predicts one probability per grid value, and the histogram and mixture heads have fixed output dimensions.

## 7 Conclusion

Boundary marginalization makes numerical events directly trainable in an autoregressive numeral distribution. Precise labels, rounding intervals, and censored outcomes contribute their exact likelihood through at most two boundary paths. Controlled experiments show gains over complete-case training when precise labels are scarce and over imputation when only rounding and threshold reports are available. Naturally censored molecular measurements and code latency further demonstrate the value of event supervision, with the latency gains reproduced on a second backbone. The same computation provides exact distributional queries and direct CRPS training, and the precision benchmark establishes tractable interval updates across a million-value support. Numerical supervision can therefore preserve the information supplied by each observation throughout learning and prediction.

## Appendix

## A Limitations

Finite representation and shared parameters. The declared grid fixes the support and resolution of the predictive distribution. Proposition 2 characterizes derivatives with respect to individual prefix logits. Shared LM parameters can still couple predictions across prefix subtrees. Threshold-sampled scores retain sampling variance. The variance reduction in Proposition 5 compares with a matched estimator that samples model outputs.

Reporting and identifiability. The likelihood interpretation of (4) assumes conditionally ignorable reporting (5). Nonignorable reporting requires modeling the reporting mechanism. Observed reports identify the event probabilities characterized by Proposition 3. At $\rho = 0$ , the controlled rounding and threshold design leaves some individual grid probabilities in the tails unidentified.

Empirical scope. Predictive rankings depend on the score: the mixture head has lower point NLL on the exact test values at $\rho = 0$ (Table 8). Outcomes behind natural hERG bounds are unobserved, and the assay reporting process selects the hERG equality subset. KBSS uses constructed timeouts and code templates as proxies for related programs. The two Qwen backbones difer in both generation and size, and seed variation on fixed splits does not quantify uncertainty from sampling new data. Predictive accuracy across grid depths remains untested.

## B Statistical Properties and Proofs

## B.1 Proof of Proposition 1

For a threshold string z, partition the complete strings smaller than z by their first difering position $j .$ . Such a string matches $z _ { < j }$ , takes a digit $a < z _ { j }$ , and has an arbitrary sufix. Its partition mass is

$$
P _ { \theta } ( z _ { < j } ) p _ { \theta } ( a \mid x , z _ { < j } ) \sum _ { t \in \{ 0 , \ldots , b - 1 \} ^ { d - j } } p _ { \theta } ( t \mid x , z _ { < j } , a ) .
$$

The sufix sum is one by repeated normalization of (2). These partitions are disjoint and exhaustive.   
Adding the threshold leaf proves (7), and using $a > z _ { j }$ proves (8).

For two diferent boundaries, all compatible strings share the boundary prefix at positions before r. A digit strictly between $l _ { r } ^ { \prime }$ and $u _ { r } ^ { \prime }$ admits every sufix. A digit equal to $l _ { r } ^ { \prime }$ must remain at least the lower sufix: it either equals the lower leaf or first becomes larger at a later position. The upper branch is symmetric. This is exactly (10), with no overlap. The largest collection has two leaves, one middle group, and at most $2 ( d - r )$ later groups, hence at most $2 d + 1$ terms.

Only conditional distributions on the two boundary strings occur, so there are at most 2d queried prefix nodes. Each grouped sibling sum examines at most b digits, which establishes $O ( b d )$ arithmetic. The sums and products are finite diferentiable expressions for the same function of $\theta ,$ so diferentiating the identity proves gradient equality. Log-sum-exp changes the numerical representation and leaves the mathematical function unchanged.

## B.2 Proof of Proposition 2

Holding the logits at other prefixes fixed, the log probability of a complete numeral k has derivative

$$
\frac { \partial \log p _ { \theta } ( k ) } { \partial \ell _ { s } ( a ) } = \mathbf { 1 } \{ k \in C _ { s } \} \big [ \mathbf { 1 } \{ k \in C _ { s a } \} - p _ { \theta } ( a \mid x , s ) \big ] .
$$

Diferentiating the log sum over A gives (12). Taking the negative posterior expectation of the displayed derivative yields

$$
- P ( C _ { s a } \mid A ) + P ( C _ { s } \mid A ) p _ { \theta } ( a \mid x , s ) = w _ { s } [ p _ { \theta } ( a \mid x , s ) - t _ { s } ( a ) ] .
$$

If $C _ { s } \subseteq A$ , conditioning on A imposes no further restriction within $C _ { s }$ , so $t _ { s } ( a ) = p _ { \theta } ( a \mid x , s )$ . If $C _ { s } \cap A = \varnothing , w _ { s } = 0$ . For an ordered interval, a subtree can be partially inside the event only if it contains a boundary. This identifies which prefixes can receive nonzero local gradients. The chain rule propagates these derivatives to shared model parameters.

## B.3 Observed-distribution consistency

For fixed $x ,$ let $B _ { A , k } = c ( A , x ) \mathbf { 1 } \{ k \in A \}$ be the matrix that maps a distribution over values to the distribution over reported events, and let $p ^ { * }$ be the true grid distribution, so $B p ^ { * }$ is the observable event distribution. Let $\mathcal { R } ( \boldsymbol { p } )$ be the expected event loss and KL denote Kullback–Leibler divergence.

Proposition 3 (Observed-distribution consistency). The excess expected event loss equals the KL divergence between the true and predicted report distributions:

$$
\begin{array} { r } { \mathcal { R } ( p ) - \mathcal { R } ( p ^ { * } ) = \mathrm { K L } ( B p ^ { * } \| B p ) . } \end{array}\tag{18}
$$

Population minimizers therefore match the observable event distribution. The report probabilities uniquely determine the grid distribution exactly when every change in the grid distribution changes at least one report probability:

$$
\ker B \cap \Big \{ \delta : \sum _ { k = 0 } ^ { M - 1 } \delta _ { k } = 0 \Big \} = \{ 0 \} .\tag{19}
$$

Proof. The expected loss is

$$
{ \mathcal { R } } ( p ) = - \sum _ { A } ( B p ^ { * } ) _ { A } \log \sum _ { k \in A } p _ { k } .
$$

Since $( B p ) _ { A } = c ( A , x ) \sum _ { k \in A } p _ { k }$ , the diference from the corresponding risk at $p ^ { * }$ cancels every nonzero reporting factor and becomes

$$
\sum _ { A } ( B p ^ { * } ) _ { A } \log { \frac { ( B p ^ { * } ) _ { A } } { ( B p ) _ { A } } } = \mathrm { K L } ( B p ^ { * } \| B p ) .
$$

Terms with zero true probability contribute zero. Thus minimizers match $B p ^ { * }$ . Two simplex distributions have the same observable event distribution exactly when their diference is in ker $B$ and has zero sum. Condition (19) makes this diference zero. If that condition fails, a nonzero null vector whose entries sum to zero perturbs an interior simplex point in both directions without changing the observable event distribution, so the map is not injective on the whole simplex.

## B.4 CRPS propriety and unbiased diferentiation

Proposition 4 (Proper objective and unbiased optimization). Let $F ^ { * }$ be the CDF of the conditional data distribution. The excess expected CRPS is the integrated squared diference between the predicted and true CDFs:

$$
\mathbb { E } _ { Y \sim p ^ { * } } \mathcal { L } _ { \mathrm { C R P S } } ( p , Y ) - \mathbb { E } _ { Y \sim p ^ { * } } \mathcal { L } _ { \mathrm { C R P S } } ( p ^ { * } , Y ) = \sum _ { k = 0 } ^ { M - 2 } \Delta _ { k } ( F ( k ) - F ^ { * } ( k ) ) ^ { 2 } ,\tag{20}
$$

so the score is strictly proper on the finite ordered support. Moreover, (17) is unbiased for (16), and its ordinary backpropagation gradient is unbiased for the gradient of (16).

Proof. At threshold $k ,$ the random indicator $H _ { k } = \mathbf { 1 } \{ Y \leq v _ { k } \}$ has expectation $F ^ { * } ( k )$ . Expanding a squared loss gives

$$
\begin{array} { r } { \mathbb { E } ( F ( k ) - H _ { k } ) ^ { 2 } = ( F ( k ) - F ^ { * } ( k ) ) ^ { 2 } + F ^ { * } ( k ) ( 1 - F ^ { * } ( k ) ) . } \end{array}
$$

The second term does not depend on $F$ . Weighted summation proves (20). All weights are positive and all $M - 1$ cumulative probabilities determine the M grid masses, which establishes strict propriety.

For the sampled objective, each threshold’s proposal probability cancels its inverse-probability weight:

$$
\mathbb { E } _ { K } \frac { \Delta _ { K } } { \pi ( K ) } ( F _ { \theta } ( K ) - h _ { K } ) ^ { 2 } = \sum _ { k } \Delta _ { k } ( F _ { \theta } ( k ) - h _ { k } ) ^ { 2 } .
$$

The support is finite and $\pi$ is independent of $\theta ,$ so diferentiation commutes with this finite expectation. Averaging independent draws preserves the identities. The same cancellation applies within each stratum in (26).

## B.5 Matched output-sampling comparison

At a fixed threshold K, the CRPS contribution has gradient

$$
g _ { K } = 2 \frac { \Delta _ { K } } { \pi ( K ) } \big ( F _ { \theta } ( K \mid x ) - h _ { K } ( r ) \big ) \nabla _ { \theta } F _ { \theta } ( K \mid x ) .\tag{21}
$$

Boundary marginalization integrates out the randomness of sampled model outputs in the following matched estimator (Proposition 5).

Fix $x , y , \theta$ , sample a threshold $K \sim \pi .$ , and draw two independent complete model outputs $Y _ { 1 } , Y _ { 2 } \sim p _ { \theta }$ Set $I _ { i } = { \bf 1 } \{ Y _ { i } \leq v _ { K } \} , h = { \bf 1 } \{ y \leq v _ { K } \} , w = \Delta _ { K } / \pi ( K )$ , and $s _ { i } = \nabla _ { \theta } \log p _ { \theta } ( Y _ { i } \mid x )$ . The score-function estimator of the CRPS gradient at threshold K from two output samples is

$$
G = w ( I _ { 1 } - h ) ( I _ { 2 } - h ) ( s _ { 1 } + s _ { 2 } ) .\tag{22}
$$

The directly diferentiated gradient is (21).

Proposition 5 (Conditional-expectation variance reduction). For the matched estimator (22), averaging over model outputs at a fixed threshold gives the direct gradient:

$$
\mathbb { E } [ G \mid K , x , y ] = g _ { K } .\tag{23}
$$

With finite second moments, direct diferentiation has no greater gradient variance in any parameter direction:

$$
\operatorname { C o v } ( G ) - \operatorname { C o v } ( g _ { K } ) = \mathbb { E } _ { K } [ \operatorname { C o v } ( G \mid K ) ] \succeq 0 .\tag{24}
$$

Proof. Condition on a threshold K. The two model outputs are independent, and

$$
\begin{array} { r } { \mathbb { E } ( I _ { i } - h ) = F _ { \theta } ( K ) - h , \qquad \mathbb { E } [ ( I _ { i } - h ) s _ { i } ] = \nabla _ { \theta } F _ { \theta } ( K ) , } \end{array}
$$

because $\mathbb { E } s _ { i } = 0$ and $\mathbb { E } [ I _ { i } s _ { i } ] = \nabla _ { \theta } P ( Y _ { i } \leq v _ { K } )$ . Therefore

$$
\begin{array} { r l } & { \mathbb { E } [ G \mid K ] = w \mathbb { E } [ ( I _ { 1 } - h ) s _ { 1 } ] \mathbb { E } [ I _ { 2 } - h ] + w \mathbb { E } [ I _ { 1 } - h ] \mathbb { E } [ ( I _ { 2 } - h ) s _ { 2 } ] } \\ & { \qquad = 2 w ( F _ { \theta } ( K ) - h ) \nabla _ { \theta } F _ { \theta } ( K ) = g _ { K } . } \end{array}
$$

The law of total covariance gives

$$
\operatorname { C o v } ( G ) = \operatorname { C o v } ( \mathbb { E } [ G \mid K ] ) + \mathbb { E } [ \operatorname { C o v } ( G \mid K ) ] ,
$$

which proves (24). Every conditional covariance matrix is positive semidefinite.

## C Implementation Details

## C.1 Numeral paths and event likelihoods

Digit tokens and alignment. Numerals are constructed from the fixed token IDs representing digits $0 , \ldots , b - 1$ , ordered by digit value, independently of multi-character number tokens. With n prompt tokens, the first $d - 1$ path digits follow the prompt, and causal logits are read at zero-based positions $n - 1 , \ldots , n + d - 2$ . Padding ofsets are handled before gathering logits. We project the required hidden states onto the ten existing vocabulary rows in FP32 with autocast disabled locally, then apply log-softmax over the digit tokens. The backbone uses bfloat16 (BF16), while verification uses 64-bit floating-point (FP64) precision. Each path repeats the prompt and uses ordinary causal attention. For q paths, this layout processes $q ( n + d - 1 )$ input tokens per observation, compared with $n + d - 1$ for sequence CE.

One coherent decoder distribution. Dropout is disabled in the probability computation while automatic diferentiation remains enabled, so paths that share a prefix describe the same decoder distribution. Trainable parameters continue to receive gradients in evaluation mode.

Log-domain operations. Log prefix masses are cumulative sums of digit log probabilities. Grouped sibling masses and interval unions use log-sum-exp. Empty digit ranges are skipped, singleton and full-support events have explicit branches, and no probability floor or imputed target is introduced into the objective. Non-finite loss values trigger a diagnostic that records the support, interval, and boundary logits.

Endpoint conversion. Measurement intervals are converted to grid indices with decimal or rational arithmetic, preserving endpoint conventions and reporting precision. Empty events are rejected.

## C.2 CRPS estimator details

Both sequence NLL and CRPS are strictly proper scores of the complete constrained numeral distribution [8]. Sequence NLL sums the autoregressive digit log losses. CRPS weights CDF errors by numerical distance, so adjacent values such as 0999 and 1000 remain close despite difering at every digit. The following details specify stable residual evaluation and the stratified estimator.

Stable residual evaluation. We compute both log F and log S directly from their prefix decompositions and evaluate the squared residual as

$$
\left( F _ { \theta } ( k ) - h _ { k } \right) ^ { 2 } = \left\{ \begin{array} { l l } { \mathrm { e x p } ( 2 \log F _ { \theta } ( k ) ) , } & { h _ { k } = 0 , } \\ { \mathrm { e x p } ( 2 \log S _ { \theta } ( k ) ) , } & { h _ { k } = 1 , } \end{array} \right.\tag{25}
$$

which avoids subtracting a CDF that has rounded to one. All accumulation is performed in at least FP32.

Stratified sampling. A stratified design partitions thresholds into fixed contiguous strata $\begin{array} { l } { { \displaystyle { \cal S } _ { g } } , } \end{array}$ sets $\begin{array} { r } { W _ { g } = \sum _ { k \in S _ { g } } \Delta _ { k } . } \end{array}$ , and samples $m _ { g }$ thresholds with probability $\Delta _ { k } / W _ { g }$ inside each stratum. The estimator

$$
\sum _ { g } \frac { W _ { g } } { m _ { g } } \sum _ { j = 1 } ^ { m _ { g } } \bigl [ F _ { \theta } ( K _ { g j } ) - h _ { K _ { g j } } \bigr ] ^ { 2 }\tag{26}
$$

is unbiased and samples every stratum. We split thresholds into two contiguous strata of approximately equal total numerical width and draw one threshold from each.

Algorithm 2 Threshold-sampled CRPS update   
1. For each exact label, draw m thresholds from a fixed sampling distribution that assigns positive   
probability to every threshold, or use a stratified design.   
2. Teacher-force the threshold paths and compute each log CDF and log survival probability.   
3. Evaluate (25) and combine terms with the sampling weights in (17) or (26).   
4. Backpropagate through the threshold paths and update the same trainable parameters.

## D Verification of the Probability Computation

The probability computation is verified in FP64 on a three-digit decimal tree with 1,000 outcomes and 70 parameters shared across prefix logits. Checks compare every CDF and 87 interval log probabilities with exhaustive enumeration, test event and CRPS gradients and finite diferences, and enumerate the matched score-function expectation on an eight-leaf tree (Table 4). Integration checks cover decoding from digit token IDs, causal logit alignment, shared-prefix consistency, endpoint conversion across carry and rounding boundaries, full-support zero loss and gradient, empty-event handling, and boundary versus enumerated losses and gradients in small causal models. In an FP32 singleton stress test with log mass −120, CDF subtraction returns zero while boundary evaluation returns the log mass directly. The minimum eigenvalue of the matched conditional covariance is $- 4 . 4 1 \times 1 0 ^ { - 1 9 }$ consistent with roundof around a positive-semidefinite matrix.

## E Experimental Details

Controlled family. The reference family is a truncated mixture on [−2, 2],

$$
\begin{array} { r l } & { \widetilde { p } ( y \mid x ) = w ( x ) \mathcal { N } ( y ; \mu _ { 1 } ( x ) , \sigma _ { 1 } ( x ) ^ { 2 } ) + ( 1 - w ( x ) ) \mathcal { N } ( y ; \mu _ { 2 } ( x ) , \sigma _ { 2 } ( x ) ^ { 2 } ) , } \\ & { \quad w ( x ) = \mathrm { s i g m o i d } ( 3 x ) , \quad \mu _ { 1 } ( x ) = 0 . 6 \sin ( \pi x ) - 0 . 5 , \quad \mu _ { 2 } ( x ) = 0 . 6 \sin ( \pi x ) + 0 . 5 , } \\ & { \quad \sigma _ { 1 } ( x ) = 0 . 0 6 + 0 . 1 2 ( x + 1 ) / 2 , \qquad \sigma _ { 2 } ( x ) = 0 . 0 8 + 0 . 1 0 ( 1 - x ) / 2 . } \end{array}\tag{27}
$$

The density is conditioned on [−2, 2], exact bin masses are computed, and the grid-valued target is sampled. The allocation is 8,000/1,000/2,000 train/validation/test inputs.

<table><tr><td>Verification quantity</td><td>Maximum absolute discrepancy</td></tr><tr><td>CDF versus enumeration</td><td> $6 . 6 6 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td>Interval log probability versus enumeration</td><td> $8 . 8 8 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td>Event gradient versus enumeration</td><td> $9 . 9 9 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td>Event gradient versus finite differences</td><td> $2 . 1 7 \times 1 0 ^ { - 1 1 }$ </td></tr><tr><td>CRPS objective versus threshold expectation</td><td> $5 . 6 8 \times 1 0 ^ { - 1 4 }$ </td></tr><tr><td>CRPS gradient versus threshold expectation</td><td> $1 . 2 8 \times 1 0 ^ { - 1 3 }$ </td></tr><tr><td>Matched score-function mean versus direct gradient</td><td> $6 . 9 4 \times 1 0 ^ { - 1 7 }$ </td></tr></table>

Table 4: Checks of the probability computation and its derivatives in FP64 on CPU.

Reporting mechanisms. For the controlled experiment, the reporting mode is drawn independently of the sampled outcome. The nominal precise-label fraction $\rho$ is the probability of reporting the precise grid value. Otherwise, a rounding report or a threshold report is chosen with equal probability. Rounding widths are drawn uniformly from {0.2, 0.4} with a fixed origin and half-open convention. A threshold c is drawn uniformly from [−1.5, 1.5] and the report retains only $Y \leq c \ \mathrm { o r } \ Y > c .$ , whichever occurred. These mechanisms satisfy (5). The same latent samples and reporting random numbers are reused across methods and fractions, with nested precise-label masks for $\rho \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 5 , 1 \}$ . Thus $\rho = 0$ retains rounding and threshold reports throughout training and validation. The mixture and histogram heads are evaluated at $\rho \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 5 \}$ . The three decoder objectives are compared at $\rho \in \{ 0 . 1 , 0 . 2 , 0 . 5 \}$ and share one reference fit per seed at $\rho = 1$

For KBSS, each synthetic budget is drawn independently from the empirical distribution of training grid indices: with probability 0.9, a sampled index defines a budget at the midpoint above it on the logarithmic grid, and with probability 0.1, no timeout is imposed. This allows uncensored observations throughout the support and gives an approximate censoring rate of 0.45 when ties are rare. One censoring configuration is fixed before comparison, and its realized censoring fraction is reported. A separate evaluation budget is fixed from the training data before test evaluation.

Molecular cohort and provenance. The hERG cohort uses ChEMBL 37 target CHEMBL240, IC50 measurements in nanomolar (nM) units, positive finite standard\_value, and standard\_relation in $\{ = , < , \leq , > , \geq \}$ [5]. We require direct single-protein assignment (assay confidence score 9), retain records with no validity flag or with Manually validated status, and exclude flagged potential duplicates [4]. The text forms $< = / > =$ and mathematical-symbol equivalents are normalized. The split manifest records activity, molecule, assay, and document IDs, the release, query, filters, and extraction checksum. The saved manifest records split-specific equality and censoring counts, bound directions, and threshold distributions.

Molecular event conversion. Let C be IC50 in nM and $Y = g ( C ) = 9 - \log _ { 1 0 } { C } .$ Then $C > c$ becomes $Y < g ( c )$ $C \geq c$ becomes $Y \leq g ( c )$ , and the other inequalities reverse analogously. We transform the standardized concentration and relation directly: ChEMBL’s pchembl\_value is computed for equality records and is not required for inclusion [4]. The declared support is [0, 14] pIC50 with $1 0 ^ { 4 }$ grid values. Equality records use the nearest grid value with half-up ties. Rounding intervals are used only when reporting precision is documented. Out-of-support equalities and empty or full-support bounds are excluded with recorded counts. Independent measurements remain separate. Parent-molecule and nonchiral Murcko-scafold groups are assigned by a deterministic allocation that does not use activity labels, targeting $8 0 / 1 0 / 1 0$ train/validation/test proportions. Prompts contain the parent molecule’s simplified molecular input line entry system (SMILES) string, structured assay type, the target name, and the numerical mapping.

Code data and common supports. We use the pinned Code-Regression revision and the KBSS latency field in milliseconds. Formatting, name, and literal variants share an abstract syntax tree (AST) template fingerprint of the code. Unparseable inputs use whitespace-normalized text. A seeded group shufle reserves the largest group for training and targets $8 0 / 1 0 / 1 0$ row proportions without splitting groups. Code exceeding 2,048 tokenizer tokens is excluded without truncation. A log-spaced grid is fixed from training extrema with a margin of 0.5 in $\log _ { 1 0 } ( \mathrm { m s } )$ . The Qwen3 comparison reuses the saved dataset and prompt strings, tokenized by its own pinned tokenizer. Split counts, support, and timeout assignments are unchanged.

Cohorts and reporting. Table 5 reports the realized report counts by split. Left bounds report that the target lies below a threshold (the upper bounds of Table 1), and right bounds report that it lies above one (the lower bounds).
<table><tr><td>Dataset</td><td>Split</td><td>Exact</td><td>Rounded</td><td>Left bounds</td><td>Right bounds</td></tr><tr><td>Controlled (ρ = 0)</td><td>Train</td><td>0</td><td>3,916</td><td>2,091</td><td>1,993</td></tr><tr><td></td><td>Validation</td><td>0</td><td>514</td><td>250</td><td>236</td></tr><tr><td></td><td>Test</td><td>0</td><td>999</td><td>519</td><td>482</td></tr><tr><td>Controlled (ρ = 0.1)</td><td>Train</td><td>853</td><td>3,482</td><td>1,877</td><td>1,788</td></tr><tr><td></td><td>Validation</td><td>98</td><td>470</td><td>226</td><td>206</td></tr><tr><td></td><td>Test</td><td>183</td><td>906</td><td>462</td><td>449</td></tr><tr><td>Controlled (ρ = 0.2)</td><td>Train</td><td>1,654</td><td>3,095</td><td>1,667</td><td>1,584</td></tr><tr><td></td><td>Validation</td><td>184</td><td>424</td><td>208</td><td>184</td></tr><tr><td></td><td>Test</td><td>403</td><td>796</td><td>398</td><td>403</td></tr><tr><td>hERG</td><td>Train</td><td>8,345</td><td>0</td><td>3,288</td><td>247</td></tr><tr><td></td><td>Validation</td><td>966</td><td>0</td><td>495</td><td>24</td></tr><tr><td></td><td>Test</td><td>948</td><td>0</td><td>518</td><td>19</td></tr><tr><td>KBSS</td><td>Train</td><td>3,263</td><td>0</td><td>0</td><td>2,686</td></tr><tr><td></td><td>Validation</td><td>397</td><td>0</td><td>0</td><td>347</td></tr><tr><td></td><td>Test</td><td>744</td><td>0</td><td>0</td><td>0</td></tr></table>

Table 5: Realized report counts for the fixed controlled-data seed and real-data splits. The precise-label probability $\rho$ is nominal: $\rho = 0 . 1$ yields 853 precise training reports out of 8,000. Controlled-test latent outcomes and the true conditional distribution are retained separately for scoring. Every KBSS test outcome remains exact. hERG bounds are expressed after the pIC50 transformation.

The hERG extraction retains 14,850 of 17,099 candidate activities. Exclusions are 1,339 potential duplicates, 587 records below confidence 9, 316 validity flags, five unsupported relations, and two missing structures. The split has 3,860/1,485/1,485 parent/scafold groups, and every validation and test group contains one report.

KBSS retains 7,437 of 12,615 latency records after the code length limit of 2,048 tokens, excluding 5,178 longer programs without truncation. The training, validation, and test splits contain 2,578/347/349 template groups, respectively. The largest groups contain 1.56%, 2.02%, and 2.02% of the examples in the training, validation, and test splits, respectively. Median full prompt lengths in the training, validation, and test splits are 1,200, 1,209, and 1,326 tokens, respectively, with maxima of 2,140/2,136/2,140. The log-grid support is [−2.40812210037, 2.19253743808] in $\log _ { 1 0 } ( \mathrm { m s } )$ , determined from training extrema with margin 0.5. There are no out-of-support outcomes in any split. Timeouts

follow the reporting rule described above. The evaluation budget is 0.03351313227795 ms, fixed from the training 75th-percentile grid index. The source dataset revision is 2410c301318408ea80eeacd6 41d887d039b95dc0. Group splits and timeout assignments are fixed across methods.
<table><tr><td>Method</td><td>Supervision and role</td></tr><tr><td>Event likelihood (ours)</td><td>Exact probability of each observed interval through the numerical decoder.</td></tr><tr><td>Complete-case CE</td><td>Same decoder trained only on precise observations</td></tr><tr><td>Imputed-label CE</td><td>Same decoder trained on midpoint representatives for bounded intervals and admissible representatives adjacent to the bounds for censored labels.</td></tr><tr><td>Event-aware mixture head</td><td>Five-component Gaussian mixture on the declared target scale, trained on the same event likelihood using the same backbone.</td></tr><tr><td>Event-aware histogram head</td><td>Fixed 256-bin head using the same backbone and evidence. The cost study also includes a separate dense output with one probability per grid value.</td></tr></table>

Table 6: Core incomplete-label comparison. The distributional heads compare the numeral decoder with other output parameterizations trained on the same incomplete evidence.

Baseline configuration. Table 6 summarizes the compared methods. The five-component mixture head predicts means, positive scales, and normalized weights from the final prompt representation. Its input-independent initialization uses equal mixture weights, means evenly spaced across the central 70% of the declared support, and scale max $\{ 0 . 0 7 5 ( U - L ) , 2 \Delta \}$ , where $\Delta$ is the grid spacing. This preserves the synthetic initializer on [−2, 2] and scales it consistently to pIC50. The histogram uses 256 bins. Imputation uses an admissible endpoint for one-sided bounds and a midpoint index for finite intervals.

Common probability evaluation. The primary metrics are hERG NLL over all reports and over the equality subset, and KBSS CRPS on the prespecified physical and logarithmic scales. For comparable event NLL, continuous heads are integrated over the common grid cells (the interval of values assigned to each grid value) and normalized over the declared support. Histogram mass is allocated by its piecewise-uniform density. Full NLL retains the empirical measurement weights, and separate equality and censoring scores expose the contributions of each report type. For hERG, equality CRPS uses the same grid target across methods, while median MAE uses the transformed reported equality value. For code latency, both CRPS scales and median errors use every original unrounded test value, including any out-of-support outcome, whose CRPS contribution is integrated exactly.

Training configuration. Qwen2.5-1.5B is pinned at revision 8faed761d45a263340a0528343f099c0 5c9a4323, and Qwen3-4B-Base at 906bfd4b4dc7f14ee4320094d8b41684abff8539. Both use rank-16 LoRA with α = 32 on attention and feed-forward projections. AdamW uses a constant learning rate 10<sup>−4</sup>, $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ , zero weight decay, gradient-norm clipping at 1.0, and efective batch size 32. No search over learning rates or head resolutions is performed. Dropout is disabled while automatic diferentiation remains active. Model/tokenizer revisions, digit IDs, source hashes, dataset manifests, support, and conversion rules are saved with each run. Table 7 lists the fit counts, update budgets, and checkpoint selection rules.

The incomplete-label studies run for three epochs, and the exact-label score study runs for nine. Validation includes step zero and epoch/final ends. Synthetic validation occurs every 50 updates, and hERG and both KBSS studies validate every 20 updates. Each checkpoint is selected separately for its method and seed using only observed validation labels. Complete-case CE repeats its precise subset to match the optimizer update budget of the full dataset.

<table><tr><td>Study</td><td>Backbone</td><td>Fits</td><td>Max updates</td><td>Validation selection</td></tr><tr><td>Controlled incomplete labels</td><td>Qwen2.5</td><td>64</td><td>750</td><td>Event NLL</td></tr><tr><td>hERG</td><td>Qwen2.5</td><td>15</td><td>1,116</td><td>Event NLL</td></tr><tr><td>KBSS</td><td>Qwen2.5</td><td>15</td><td>558</td><td>Event NLL</td></tr><tr><td>KBSS transfer</td><td>Qwen3</td><td>9</td><td>558</td><td>Event NLL</td></tr><tr><td>Exact-label scores</td><td>Qwen2.5</td><td>9</td><td>2,250</td><td>CRPS</td></tr></table>

Table 7: The 112 distinct learning fits represented in the reported comparisons. The 64 controlled fits include 60 core fits and four additional event likelihood and complete-case CE fits for seeds 3 and 4 at $\rho = 0 . 2$ . The core allocation is 12, 15, 15, 15, and 3 fits at $\rho = 0 , 0 . 1 , 0 . 2 , 0 . 5 , 1$ , respectively. The shared full-label reference is counted once per seed.

Evaluation budget. Synthetic ISE and CRPS use 64 shared stratified threshold queries per input, each query being a gap between adjacent grid values, with a separate integration check comparing 64 and 256 queries at $\rho \in \{ 0 . 2 , 0 . 5 , 1 \}$ . hERG equality CRPS uses 64 fixed, physically weighted stratified queries per equality. KBSS uses 64 shared threshold queries for both CRPS scales. The gap widths on each scale and the exact position of each unrounded test value inside its gap give the physical and logarithmic contributions. Out-of-support CRPS contributions are integrated exactly. The same queries are reused across methods and training seeds. Event NLL and budget probabilities use direc event queries. Reported medians use deterministic CDF bisection. The precision study is separate: 30 configurations, three benchmark replicates each, and five warmup plus 20 measured updates per replicate.

## E.1 Exact-label score protocol

The completed study compares sequence CE, threshold-sampled CRPS, and CDF-BCE on the existing fully precise controlled data: 8,000/1,000/2,000 training/validation/test inputs, support [−2, 2], and 10,000 grid values. All three methods use Qwen2.5-1.5B, seeds 0, 1, and 2, efective/micro batch sizes 32/8, and the common 2,250-update budget. Checkpoints are selected by validation CRPS on the observed labels at steps 0, 50, . . . , 2250, using 64 fixed per-input validation queries. Test evaluation uses a separate fixed set of 64 stratified queries per input.

CRPS and CDF-BCE share two contiguous threshold strata, with one fresh threshold per stratum on every presentation of an observation. Thresholds are drawn for the full efective batch before microbatch slicing. The sampled training scores are divided by the fixed support width W = 4. Reported test CRPS and ISE retain their original units. For independent threshold draws, the unnormalized CDF-BCE estimator is

$$
\widehat { \mathcal { L } } _ { \mathrm { C D F - B C E } } = - \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \frac { \Delta _ { K _ { j } } } { \pi ( K _ { j } ) } \left[ h _ { K _ { j } } \log F _ { \theta } ( K _ { j } ) + ( 1 - h _ { K _ { j } } ) \log S _ { \theta } ( K _ { j } ) \right] .\tag{28}
$$

The stratified implementation uses the weights of (26) and direct log tails. Expected excess CDF-BCE is a sum of Bernoulli KL divergences weighted by the positive grid spacings $\Delta _ { k } .$ , so it is also strictly proper on the grid. The comparison holds the decoder and, between the two threshold objectives, the query design fixed. Sequence CE remains the log score of the complete numeral.

## F Completed Results and Reproducibility

## F.1 Controlled recovery and seed replication

Table 8 gives all 20 three-seed configurations (60 fits). Complete-case CE is unavailable at $\rho = 0 .$ where every training observation is a rounding interval or a one-sided threshold report. The full-label reference at $\rho = 1$ is one shared decoder fit per seed. Table 9 reports the five-seed extension, and Figure 5 shows the individual fits behind the three-seed results. Figure 6 shows the paired diferences between event likelihood and each control at $\rho = 0 . 1$ , where event likelihood improves on both CE controls in every seed, together with the five-seed diferences at $\rho = 0 . 2$
<table><tr><td> $\rho$ </td><td>Method</td><td>ISE (10−3)</td><td>Event NLL</td><td>CRPS</td><td>Point NLL</td></tr><tr><td>0</td><td>Event likelihood (ours)</td><td> $6 . 8 0 1 { \scriptstyle \pm 2 . 3 2 4 }$ </td><td> $0 . 8 5 2 { \pm } 0 . 0 0 9$ </td><td> $0 . 2 2 0 5 { \pm } 0 . 0 0 0 8$ </td><td> $8 . 1 6 9 { \pm } 0 . 0 6 7$ </td></tr><tr><td>0</td><td>Imputed-label CE</td><td> $3 4 . 1 2 1 { \pm } 3 . 7 2 9$ </td><td> $1 . 0 2 4 { \pm } 0 . 0 1 7$ </td><td> $0 . 2 4 8 6 { \pm } 0 . 0 0 3 6$ </td><td> $9 . 5 6 3 { \pm } 0 . 0 8 6$ </td></tr><tr><td>0</td><td>Event-aware mixture</td><td> $7 . 0 9 9 { \pm } 1 . 0 5 3 $ </td><td> $0 . 8 6 4 { \pm } 0 . 0 1 0$ </td><td> $0 . 2 2 2 1 { \pm } 0 . 0 0 1 8$ </td><td> $7 . 9 3 2 { \pm } 0 . 0 0 6$ </td></tr><tr><td>0</td><td>Event-aware histogram</td><td> $9 . 0 8 5 { \pm } 0 . 9 7 7$ </td><td> $0 . 8 6 2 { \scriptstyle \pm 0 . 0 2 2 }$ </td><td> $0 . 2 2 3 8 { \pm } 0 . 0 0 1 9$ </td><td> $8 . 2 7 8 { \pm } 0 . 0 2 6$ </td></tr><tr><td>0.1</td><td>Event likelihood (ours)</td><td> $6 . 9 9 3 { \pm } 1 . 1 4 3$ </td><td> $1 . 4 8 1 { \pm } 0 . 0 0 5$ </td><td> $0 . 2 2 0 8 { \pm } 0 . 0 0 0 6$ </td><td> $7 . 9 0 8 { \pm } 0 . 0 1 8$ </td></tr><tr><td>0.1</td><td>Complete-case CE</td><td> $1 5 . 9 2 3 { \pm } 5 . 1 3 7$ </td><td> $1 . 5 4 4 { \pm } 0 . 0 1 0$ </td><td> $0 . 2 3 2 2 { \pm } 0 . 0 0 3 8$ </td><td> $8 . 1 5 3 { \pm } 0 . 0 2 9$ </td></tr><tr><td>0.1</td><td>Imputed-label CE</td><td> $2 9 . 1 5 6 { \pm } 2 . 4 7 2$ </td><td> $1 . 7 4 0 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 2 4 3 8 { \pm } 0 . 0 0 2 1$ </td><td> $9 . 1 5 3 { \pm } 0 . 0 5 2$ </td></tr><tr><td>0.1</td><td>Event-aware mixture</td><td> $7 . 6 9 7 { \scriptstyle \pm 0 . 3 0 3 }$ </td><td> $1 . 5 0 0 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 2 2 1 9 { \pm } 0 . 0 0 2 5$ </td><td> $7 . 9 3 3 { \pm } 0 . 0 2 2$ </td></tr><tr><td>0.1</td><td>Event-aware histogram</td><td> $7 . 1 9 2 { \pm } 1 . 3 3 3$ </td><td> $1 . 5 2 2 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 2 2 1 1 { \pm } 0 . 0 0 0 2$ </td><td> $8 . 3 1 5 { \pm } 0 . 0 1 3$ </td></tr><tr><td>0.2</td><td>Event likelihood (ours)</td><td> $4 . 2 8 7 { \pm } 1 . 0 0 0$ </td><td> $2 . 2 6 3 { \pm } 0 . 0 0 7$ </td><td> $0 . 2 1 9 3 { \scriptstyle \pm 0 . 0 0 1 7 }$ </td><td> $7 . 9 0 0 { \pm } 0 . 0 1 4$ </td></tr><tr><td>0.2</td><td>Complete-case CE</td><td> $6 . 3 0 4 { \pm } 1 . 2 3 0 $ </td><td> $2 . 2 7 3 { \pm } 0 . 0 0 2$ </td><td> $0 . 2 2 1 6 { \pm } 0 . 0 0 1 2$ </td><td> $7 . 9 4 9 { \pm } 0 . 0 1 7$ </td></tr><tr><td>0.2</td><td>Imputed-label CE</td><td> $2 5 . 6 1 6 { \scriptstyle \pm 5 . 3 9 2 }$ </td><td> $2 . 5 6 9 { \pm } 0 . 0 3 5$ </td><td> $0 . 2 4 2 0 { \scriptstyle \pm 0 . 0 0 5 7 }$ </td><td> $8 . 8 6 9 { \scriptstyle \pm 0 . 0 7 7 }$ </td></tr><tr><td>0.2</td><td>Event-aware mixture</td><td> $6 . 5 7 4 { \scriptstyle \pm 2 . 0 0 5 }$ </td><td> $2 . 2 7 1 { \pm } 0 . 0 0 8$ </td><td> $0 . 2 2 1 4 { \pm } 0 . 0 0 1 6$ </td><td> $7 . 9 0 5 { \pm } 0 . 0 2 8$ </td></tr><tr><td>0.2</td><td>Event-aware histogram</td><td> $5 . 3 8 8 { \pm } 0 . 9 3 5$ </td><td> $2 . 3 0 1 { \pm } 0 . 0 1 3$ </td><td> $0 . 2 2 0 5 { \pm } 0 . 0 0 2 4$ </td><td> $8 . 1 3 3 { \pm } 0 . 0 1 5$ </td></tr><tr><td>0.5</td><td>Event likelihood (ours)</td><td> $4 . 9 3 6 { \pm } 1 . 7 8 0$ </td><td> $4 . 2 8 5 { \pm } 0 . 0 2 1$ </td><td> $0 . 2 1 9 5 { \pm } 0 . 0 0 1 5$ </td><td> $7 . 8 8 1 { \pm } 0 . 0 2 8$ </td></tr><tr><td>0.5</td><td>Complete-case CE</td><td> $5 . 1 8 2 { \pm } 0 . 4 4 8$ </td><td> $4 . 2 9 4 { \pm } 0 . 0 1 2$ </td><td> $0 . 2 2 0 6 { \pm } 0 . 0 0 2 2$ </td><td> $7 . 8 9 7 { \scriptstyle \pm 0 . 0 1 3 }$ </td></tr><tr><td>0.5</td><td>Imputed-label CE</td><td> $1 2 . 2 7 0 { \scriptstyle \pm 1 . 7 0 6 }$ </td><td> $4 . 5 3 7 { \pm } 0 . 0 2 8$ </td><td> $0 . 2 2 6 3 { \pm } 0 . 0 0 1 2$ </td><td> $8 . 3 4 4 { \pm } 0 . 0 5 0$ </td></tr><tr><td>0.5</td><td>Event-aware mixture</td><td> $6 . 0 3 7 { \pm } 2 . 6 2 9$ </td><td> $4 . 2 8 5 { \pm } 0 . 0 0 3$ </td><td> $0 . 2 2 0 7 { \scriptstyle \pm 0 . 0 0 1 1 }$ </td><td> $7 . 8 7 9 { \pm } 0 . 0 0 3$ </td></tr><tr><td>0.5</td><td>Event-aware histogram</td><td> $5 . 3 1 2 { \scriptstyle \pm 0 . 7 9 9 }$ </td><td> $4 . 3 3 8 { \pm } 0 . 0 2 8$ </td><td> $0 . 2 1 9 8 { \pm } 0 . 0 0 1 7$ </td><td> $8 . 0 1 4 { \pm } 0 . 0 3 1$ </td></tr><tr><td>1</td><td>Shared full-label reference</td><td> $3 . 8 1 9 { \pm } 0 . 4 9 9$ </td><td> $7 . 8 4 8 { \pm } 0 . 0 0 7$ </td><td> $0 . 2 1 7 4 { \scriptstyle \pm 0 . 0 0 1 2 }$ </td><td> $7 . 8 4 8 { \pm } 0 . 0 0 7$ </td></tr></table>

Table 8: All controlled configurations, mean ± sample SD over seeds 0, 1, and 2. Event NLL compares methods within the same precise-label fraction, and ISE compares recovery across fractions. CRPS and point NLL score the precise test outcomes. All scores use the fixed 2,000-input test cohort.

<table><tr><td>Method  $( \rho = 0 . 2 )$ </td><td>Seeds</td><td> $\mathrm { I S E \ ( 1 0 ^ { - 3 } ) }$ </td><td>Event NLL</td></tr><tr><td>Event likelihood (ours)</td><td>5</td><td> $4 . 8 1 8 { \pm } 1 . 0 8 5$ </td><td> $2 . 2 5 8 { \pm } 0 . 0 0 8$ </td></tr><tr><td> $\mathrm { C o m p l e t e - c a s e ~ C E }$ </td><td>5</td><td> $7 . 9 0 6 { \pm } 2 . 9 5 3$ </td><td> $2 . 2 8 8 { \pm } 0 . 0 2 2$ </td></tr></table>

Table 9: Five-seed follow-up comparing event likelihood with complete-case CE, extending seeds 0, 1, and 2 with seeds 3 and 4. Entries are means ± sample SD. The mean ISE diference (event likelihood minus complete-case CE) is $- 3 . 0 8 8 \times 1 0 ^ { - 3 }$

The paired ISE diferences $( 1 0 ^ { - 3 } )$ , event likelihood minus complete-case CE, for seeds $0 { - } 4$ are $- 4 . 2 0 1 , \ : - 2 . 0 5 6 , \ : + 0 . 2 0 7 , \ : - 7 . 7 5 0 , \ : \mathrm { a n d } \ : - 1 . 6 4 0$

Repeated integration checks preserve the direction and magnitude of the gain. On the fixed 256-input subset, the five-seed mean ISE diferences are −2.971, −2.960, −2.976, −2.936, and −2.937

$( \times 1 0 ^ { - 3 } )$ for the primary 64-query draw, two additional 64-query draws, and two 256-query draws.   
Their range is $4 . 0 8 \times 1 0 ^ { - 5 }$ in the unscaled diference.

![](images/12315aa9719ef8cf8ba67d5cf61521087ce03c1e38d3549dd33f3b48ce464d9a.jpg)  
Figure 5: Individual fits behind Figure 2. Each marker is one training run, identified by its fine-tuning seed, and the bar is the seed mean. All 60 core fits are shown, with the shared full-label reference drawn once per seed. The vertical axis is logarithmic and shows ISE multiplied by $1 0 ^ { 3 }$

Conditional CDFs. Figure 3 uses test input test\_001617, selected by the smallest |x| with ties broken by saved ID. The nine fits use their existing checkpoints selected by validation event NLL. We query 512 common grid indices spanning the support, including both endpoints, and compute the true grid CDF through each cell’s upper edge, matching the ISE evaluation. The quoted central CDF levels use the shared threshold $v _ { 4 9 9 0 } \simeq - 0 . 0 0 3 8$ between the two rises.

![](images/9791d8a24cf535ab9224c3a3b039a1f0a313345288a2c55f4ba331f62186284e.jpg)

(b) Controlled, ρ = 0.2  
![](images/9f1428c8733f2522cd41f55f94e3aa9c8b2ebf97e010174756d7487dd78b78df.jpg)

(c) KBSS: physical scale  
![](images/514178106f0e3a28468ac219c8a22c90851c6237264cf63c05182411a8190a01.jpg)  
Negative differences favor event likelihood

(d) KBSS: log<sub>10</sub> (ms)  
![](images/20bbbcf886525dfc91628dfbd130c05620102b73aae378c4f3293ddf825e0618.jpg)  
Seed 0 Seed 1 Seed 2  
Figure 6: Paired diferences across training seeds. (a) ISE diferences, event likelihood minus each of the four controls, at $\rho = 0 . 1$ for seeds 0–2. (b) ISE diferences, event likelihood minus complete-case CE, at $\rho = 0 . 2$ for all five seeds. $( \mathrm { c } , \mathrm { d } )$ KBSS CRPS diferences, event likelihood minus each of the four controls, for all three Qwen2.5 seeds on physical and logarithmic scales. Negative diferences favor event likelihood, and points show individual seed diferences.

## F.2 Supporting hERG and KBSS metrics

Table 10 reports hERG NLL by bound type, equality CRPS, and the MAE of predicted medians. Table 11 reports the KBSS budget Brier score and median errors on both scales.
<table><tr><td>Method</td><td>Censored NLL</td><td>Left-bound NLL</td><td>Right-bound NLL</td><td>Equality CRPS</td><td>Equality median MAE</td></tr><tr><td>Event likelihood (ours)</td><td> $0 . 8 2 3 { \pm } 0 . 0 9 1$ </td><td> $0 . 8 0 1 { \scriptstyle \pm 0 . 1 0 2 }$ </td><td> $1 . 4 3 3 { \pm } 0 . 2 9 6$ </td><td> $0 . 4 7 3 { \pm } 0 . 0 2 3$ </td><td> $0 . 6 5 5 { \pm } 0 . 0 2 9$ </td></tr><tr><td>Complete-case CE</td><td> $2 . 0 7 6 { \pm } 0 . 1 1 2$ </td><td> $2 . 1 0 5 { \pm } 0 . 1 1 9$ </td><td> $1 . 2 7 3 { \pm } 0 . 1 4 5$ </td><td> $0 . 4 3 6 { \pm } 0 . 0 1 1$ </td><td> $0 . 6 2 5 { \pm } 0 . 0 1 9$ </td></tr><tr><td>Imputed-label CE</td><td> $1 . 3 5 6 { \pm } 0 . 1 7 5$ </td><td> $1 . 3 5 2 { \pm } 0 . 1 8 9$ </td><td> $1 . 4 6 8 { \pm } 0 . 1 9 0$ </td><td> $0 . 4 5 4 { \pm } 0 . 0 2 6$ </td><td> $0 . 6 4 7 { \scriptstyle \pm 0 . 0 3 9 }$ </td></tr><tr><td>Event-aware mixture</td><td> $0 . 8 5 0 { \pm } 0 . 0 3 2$ </td><td> $0 . 8 3 1 { \pm } 0 . 0 3 5$ </td><td> $1 . 3 7 7 { \scriptstyle \pm 0 . 1 3 2 }$ </td><td> $0 . 4 8 0 { \pm } 0 . 0 0 6$ </td><td> $0 . 6 8 0 { \pm } 0 . 0 0 8$ </td></tr><tr><td>Event-aware histogram</td><td> $0 . 9 6 3 { \pm } 0 . 1 9 4$ </td><td> $0 . 9 5 0 { \pm } 0 . 2 0 3$ </td><td> $1 . 2 9 8 { \pm } 0 . 0 6 9$ </td><td> $0 . 5 1 3 { \pm } 0 . 0 3 0$ </td><td> $0 . 6 9 0 { \scriptstyle \pm 0 . 0 3 0 }$ </td></tr></table>

Table 10: Supporting hERG scores, mean ± sample SD over three seeds. There are 518 left bounds and 19 right bounds in the test set. Equality CRPS and the mean absolute error of predicted medians use pIC50 units. Full and equality NLL remain visible in Table 2.

<table><tr><td>Method</td><td>Budget Brier</td><td>Median MAE  $\left( 1 0 ^ { - 3 } ~ \mathrm { m s } \right)$ </td><td>Median MAE  $\left( \log _ { 1 0 } \mathrm { m s } \right)$ </td></tr><tr><td>Event likelihood (ours)</td><td> $0 . 1 2 8 1 { \pm } 0 . 0 0 4 6$ </td><td> $9 . 9 6 7 { \scriptstyle \pm 0 . 2 2 6 }$ </td><td> $0 . 1 1 5 3 { \pm } 0 . 0 0 2 7$ </td></tr><tr><td>Complete-case CE</td><td> $0 . 1 5 5 0 { \pm } 0 . 0 0 8 2$ </td><td> $1 1 . 2 2 1 { \pm } 0 . 3 1 5$ </td><td> $0 . 1 3 3 7 { \pm } 0 . 0 0 5 2$ </td></tr><tr><td>Imputed-label CE</td><td> $0 . 1 9 3 6 { \pm } 0 . 0 0 2 0$ </td><td> $1 3 . 7 4 8 { \pm } 0 . 3 3 5$ </td><td> $0 . 1 8 2 0 { \scriptstyle \pm 0 . 0 0 7 9 }$ </td></tr><tr><td>Event-aware mixture</td><td> $0 . 1 5 0 5 { \pm } 0 . 0 0 2 6$ </td><td> $1 1 . 2 6 4 { \pm } 0 . 3 3 5$ </td><td> $0 . 1 3 4 0 { \scriptstyle \pm 0 . 0 0 7 5 }$ </td></tr><tr><td>Event-aware histogram</td><td> $0 . 1 3 4 2 { \pm } 0 . 0 0 3 6$ </td><td> $1 0 . 1 8 5 { \pm } 0 . 3 3 7$ </td><td> $0 . 1 1 7 4 { \scriptstyle \pm 0 . 0 0 5 3 }$ </td></tr></table>

Table 11: Supporting KBSS metrics on all 744 exact outcomes, mean $\pm$ sample SD over seeds 0, 1, and 2. Brier evaluates strict exceedance of the evaluation budget fixed from the training data, using a direct survival query.

## F.3 Cross-backbone supporting scores

Table 12 reports the median errors of the Qwen3-4B-Base study.
<table><tr><td>Method</td><td>Median MAE  $( 1 0 ^ { - 3 } ~ \mathrm { m s } )$ </td><td>Median MAE  $( \log _ { 1 0 }$  ms)</td></tr><tr><td>Event likelihood (ours)</td><td> $9 . 9 7 2 { \scriptstyle \pm 0 . 2 9 3 }$ </td><td> $0 . 1 1 5 9 { \pm } 0 . 0 0 3 8$ </td></tr><tr><td>Complete-case CE</td><td> $1 1 . 3 6 6 { \pm } 0 . 4 9 4$ </td><td> $0 . 1 3 6 4 { \pm } 0 . 0 0 9 6$ </td></tr><tr><td>Event-aware histogram</td><td> $1 0 . 8 1 4 { \scriptstyle \pm 0 . 3 0 8 }$ </td><td> $0 . 1 2 2 6 { \pm } 0 . 0 0 3 3$ </td></tr></table>

Table 12: Supporting Qwen3-4B-Base median errors on KBSS, mean ± sample SD over seeds 0, 1, and 2. Both scores use the same 744 exact outcomes and CDF bisection procedure as the Qwen2.5 study. CRPS and budget Brier scores appear in Table 3.

## F.4 Exact-label score results

Table 13 reports the exact-label test scores. The updates selected by validation CRPS for seeds 0, 1, and 2 are (800, 800, 750) for CE, (2050, 1900, 2050) for CRPS, and (2050, 1900, 1850) for CDF-BCE. Figure 7 summarizes the validation trajectories. The full-label reference in Table 8 uses the separate incomplete-label protocol, with 750 updates and checkpoint selection by validation event NLL.
<table><tr><td>Training objective</td><td>Test CRPS</td><td> $\mathrm { I S E \ ( 1 0 ^ { - 3 } ) }$ </td><td>Point NLL</td></tr><tr><td>Sequence CE</td><td> $\mathbf { 0 . 2 1 6 4 7 { \pm } 0 . 0 0 0 2 3 }$ </td><td> $\mathbf { 3 . 3 9 9 } \pm 0 . 9 1 0$ </td><td> $\mathbf { 7 . 8 4 2 \pm 0 . 0 1 0 }$ </td></tr><tr><td>Threshold CRPS</td><td> $0 . 2 1 9 7 4 { \pm } 0 . 0 0 1 6 2$ </td><td> $5 . 7 7 7 { \scriptstyle \pm 1 . 1 7 1 }$ </td><td> $8 . 1 5 6 { \pm } 0 . 0 6 2$ </td></tr><tr><td>Threshold CDF-BCE</td><td> $0 . 2 2 1 8 9 { \pm } 0 . 0 0 2 0 2$ </td><td> $6 . 8 5 6 { \pm } 1 . 5 1 6$ </td><td> $8 . 1 8 0 { \pm } 0 . 2 7 3$ </td></tr></table>

Table 13: Exact-label controlled comparison at a common 2,250-update budget. Entries are means ± sample SD over seeds 0, 1, and 2, and lower is better. Every method selects its checkpoint by validation CRPS. Test CRPS and ISE use 64 common stratified queries on 2,000 test inputs. Point NLL scores the complete precise numeral.

## F.5 Complete precision measurements

The benchmark uses the first 32 input IDs from the controlled training set and four fixed physical intervals: [−1.73, −0.41], [−0.37, 0.83], [0.19, 1.71], and [−1.41, 1.39]. All remain interior intervals requiring two boundary paths at every depth. Attended neutral context gives exactly 256 or 1,024 prompt tokens. The efective batch is 32 and microbatch is four observations. Every method uses rank-16/alpha-32 LoRA, BF16 backbone computations, and the production probability arithmetic. All three timing replicates use the same random seed. Each fresh process performs five warmup updates followed by 20 synchronized full optimizer updates. The timer includes gradient clearing, input assembly/transfers, all microbatch forwards, likelihood evaluation, backward, clipping, and AdamW. Replicate case order is randomized, and ratios of update times to those of imputed-label CE are paired within the same replicate and physical GPU.

![](images/e0f311e48fa74120318939b1ae993bfdbe15aeac8acdd4434c7c5ed749802487.jpg)  
Figure 7: Validation CRPS during the exact-label score study, shown from update 100 through the common 2,250-update budget. Curves are unsmoothed means over seeds 0, 1, and 2, and shading denotes one sample SD across seeds. Every checkpoint uses the same 64 fixed stratified validation queries per input. Test results use the checkpoint with the lowest validation CRPS, selected separately for each method and seed.

The dense per-value output predicts one probability per grid value, with $( H + 1 ) 1 0 ^ { d }$ added output parameters for hidden width H. The practical mixture and 256-bin histogram retain fixed parameter counts but store the existing FP64 cell-edge array (about 8 MB at d = 6). Update times are means of medians from three timing replicates, and memory is the maximum observed peak allocation across replicates. Figure 8 shows the update times, and Table 14 lists every timing and memory measurement.

![](images/4b22c8cca9e38c149e48196bb35ffb1f01a22369ddcdc75a1dab599fa8df4b0b.jpg)

![](images/cf9a16b1bd44cc778f3329b71f5b0193adce5c14a81af18a7ec5be033436c6e1.jpg)

Figure 8: Training update time for the same workloads as Figure 4. The event implementation processes two paths per observation, each with its own copy of the prompt, and imputed-label CE and all heads process one prompt. Curves show means of the medians from the three timing replicates.
<table><tr><td></td><td></td><td colspan="3">256 prompt tokens</td><td colspan="3">1,024 prompt tokens</td></tr><tr><td>Method</td><td>d</td><td>Update (s)</td><td>GiB</td><td>×CE</td><td>Update (s)</td><td>GiB</td><td>×CE</td></tr><tr><td>Event likelihood</td><td>2</td><td>0.888158</td><td>11.224</td><td>1.075</td><td>2.114650</td><td>27.458</td><td>1.458</td></tr><tr><td>Event likelihood</td><td>4</td><td>0.873143</td><td>11.245</td><td>1.071</td><td>2.143870</td><td>27.071</td><td>1.472</td></tr><tr><td>Event likelihood</td><td>6</td><td>0.872044</td><td>11.343</td><td>1.098</td><td>2.102270</td><td>27.008</td><td>1.446</td></tr><tr><td>Imputed-label CE</td><td>2</td><td>0.826942</td><td>7.235</td><td>1.000</td><td>1.450010</td><td>19.034</td><td>1.000</td></tr><tr><td>Imputed-label CE</td><td>4</td><td>0.816379</td><td>7.245</td><td>1.000</td><td>1.456340</td><td>19.064</td><td>1.000</td></tr><tr><td>Imputed-label CE</td><td>6</td><td>0.795354</td><td>7.257</td><td>1.000</td><td>1.453370</td><td>19.095</td><td>1.000</td></tr><tr><td>Histogram (256)</td><td>2</td><td>0.833546</td><td>7.234</td><td>1.009</td><td>1.433360</td><td>19.023</td><td>0.989</td></tr><tr><td>Histogram (256)</td><td>4</td><td>0.817460</td><td>7.234</td><td>1.002</td><td>1.436490</td><td>19.023</td><td>0.986</td></tr><tr><td>Histogram (256)</td><td>6</td><td>0.811750</td><td>7.241</td><td>1.021</td><td>1.444380</td><td>19.031</td><td>0.994</td></tr><tr><td>Mixture (5)</td><td>2</td><td>0.806047</td><td>7.228</td><td>0.976</td><td>1.446600</td><td>19.017</td><td>0.998</td></tr><tr><td>Mixture (5)</td><td>4</td><td>0.829752</td><td>7.228</td><td>1.017</td><td>1.442720</td><td>19.017</td><td>0.991</td></tr><tr><td>Mixture (5)</td><td>6</td><td>0.813540</td><td>7.235</td><td>1.023</td><td>1.448850</td><td>19.025</td><td>0.997</td></tr><tr><td>Dense per-value output</td><td>2</td><td>0.802479</td><td>7.230</td><td>0.973</td><td>1.431710</td><td>19.019</td><td>0.987</td></tr><tr><td>Dense per-value output</td><td>4</td><td>0.811784</td><td>7.472</td><td>0.996</td><td>1.435780</td><td>19.246</td><td>0.986</td></tr><tr><td>Dense per-value output</td><td>6</td><td>0.840514</td><td>35.849</td><td>1.057</td><td>1.562770</td><td>47.513</td><td>1.075</td></tr></table>

Table 14: Complete precision scaling measurements. GiB denotes peak allocated memory, and ratios of update times to those of imputed-label CE are computed within each timing replicate and then averaged.

## References

[1] Yash Akhauri, Xingyou Song, Arissa Wongpanich, Bryan Lewandowski, and Mohamed S Abdelfattah. Regression language models for code. arXiv preprint arXiv:2509.26476, 2025.

[2] Davide Baldelli, Sruthi Kuriakose, Maryam Hashemzadeh, Amal Zouaq, and Sarath Chandar. Probabilistic calibration is a trainable capability in language models. arXiv preprint arXiv:2605.11845, 2026.

[3] Pengxiang Cai, Wanchen Lian, Chenyang Liu, Xiaohan Li, Qingyuan Zeng, Jinhong Wang, and Jintai Chen. Path: Next-interval prediction via autoregressive tree hierarchy on tabular data. arXiv preprint arXiv:2608.08078, 2026.

[4] ChEMBL Team. Assay and activity questions. ChEMBL Interface Documentation.

[5] ChEMBL Team. ChEMBL database, release 37. EMBL-EBI, May 2026.

[6] Ming Chen, Sheng Tang, Rong-Xi Tan, Ziniu Li, Jiacheng Chen, Ke Xue, and Chao Qian. Beyond token-level supervision: Unlocking the potential of decoding-based regression via reinforcement learning. arXiv preprint arXiv:2512.06533, 2025.

[7] Xiang Fei, Jinghui Lu, Qi Sun, Hao Feng, Yanjie Wang, Wei Shi, An-Lan Wang, Jingqun Tang, and Can Huang. Advancing sequential numerical prediction in autoregressive models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 562–574, 2025.

[8] Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association, 102(477):359–378, 2007.

[9] Nate Gruver, Marc Finzi, Shikai Qiu, and Andrew G Wilson. Large language models are zero-shot time series forecasters. Advances in neural information processing systems, 36:19622–19635, 2023.

[10] Daniel F Heitjan and Donald B Rubin. Ignorability and coarse data. The annals of statistics, pages 2244–2253, 1991.

[11] Rong Jin and Zoubin Ghahramani. Learning with multiple labels. Advances in neural information processing systems, 15, 2002.

[12] Michal Lukasik, Zhao Meng, Harikrishna Narasimhan, Yin-Wen Chang, Aditya Krishna Menon, Felix Yu, and Sanjiv Kumar. Better autoregressive regression with llms via regression-aware fine-tuning. In International Conference on Learning Representations, volume 2025, pages 69871–69901, 2025.

[13] Michal Lukasik, Harikrishna Narasimhan, Aditya Krishna Menon, Felix Yu, and Sanjiv Kumar. Regression aware inference with llms. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 13667–13678, 2024.

[14] Jungsoo Park, Hyungjoo Chae, Ethan Mendes, Jay DeYoung, Varsha Kishore, Wei Xu, and Alan Ritter. Reinforcement learning over predictive distributions for llm regression. arXiv preprint arXiv:2605.20740, 2026.

[15] Julianna Piskorz, Katarzyna Kobalczyk, and Mihaela van der Schaar. Eliciting numerical predictive distributions of llms without autoregression. arXiv preprint arXiv:2603.02913, 2026.

[16] James Requeima, John Bronskill, Dami Choi, Richard E Turner, and David Duvenaud. Llm processes: Numerical predictive distributions conditioned on natural language. Advances in Neural Information Processing Systems, 37:109609–109671, 2024.

[17] Chenze Shao, Fandong Meng, Yijin Liu, and Jie Zhou. Language generation with strictly proper scoring rules. arXiv preprint arXiv:2405.18906, 2024.

[18] Xingyou Song and Dara Bahri. Decoding-based regression. arXiv preprint arXiv:2501.19383, 2025.

[19] Georgios Spithourakis and Sebastian Riedel. Numeracy for language models: Evaluating and improving their ability to predict numbers. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2104–2115, 2018.

[20] Emma Svensson, Hannah Rosa Friesacher, Susanne Winiwarter, Lewis Mervin, Adam Arany, and Ola Engkvist. Enhancing uncertainty quantification in drug discovery with censored regression labels. Artificial Intelligence in the Life Sciences, 7:100128, 2025.

[21] Wesley Tansey, Karl Pichotta, and James G Scott. Deep nonparametric estimation of discrete conditional distributions via smoothed dyadic partitioning. arXiv preprint arXiv:1702.07398, 2017.

[22] Luke Vilnis, Yury Zemlyanskiy, Patrick Murray, Alexandre Tachard Passos, and Sumit Sanghai. Arithmetic sampling: parallel diverse decoding for large language models. In International conference on machine learning, pages 35120–35136. PMLR, 2023.

[23] Yishu Wei, Hexin Dong, Yi Lin, Jiahe Qian, Yi Liu, and Yifan Peng. Towards end-to-end llm-based censoring-aware survival analysis. arXiv preprint arXiv:2605.25399, 2026.

[24] Tianqi Xiang, Qixiang Zhang, Xinpeng Ding, Yi Li, and Xiaomeng Li. Cacsurv: Concordancealigned comparative learning with large language models for cancer survival prediction. arXiv preprint arXiv:2608.16594, 2026.

[25] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[26] Weiwei Ye, Hangchen Liu, and Renhe Jiang. Numericjev: Jev-like llm numerical decoding with multiway decision trees. arXiv preprint arXiv:2609.28587, 2026.

[27] Jonas Zausinger, Lars Pennig, Anamarija Kozina, Sean Sdahl, Julian Sikora, Adrian Dendorfer, Timofey Kuznetsov, Mohamad Hagog, Nina Wiedemann, Kacper Chlodny, et al. Regress, don’t guess–a regression-like loss on number tokens for language models. arXiv preprint arXiv:2411.02083, 2024.

[28] Zhaohui Zheng, Chenhang He, Shihao Wang, Yuxuan Li, Ming-Ming Cheng, and Lei Zhang. Del: Digit entropy loss for numerical learning of large language models. arXiv preprint arXiv:2605.20369, 2026.

[29] Zhuo Zuo, Li Yue, Wenhao Zheng, Chenpeng Wang, and Xianggen Liu. Enhancing numerical prediction in llms via smooth mmd alignment. arXiv preprint arXiv:2606.27731, 2026.