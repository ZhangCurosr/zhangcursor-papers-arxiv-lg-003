# WUSH-KV: KV Cache Quantization with Data-Adaptive Transforms

Jiale Chen Jiale.Chen@ist.ac.at

Institute of Science and Technology Austria (ISTA)

Vage Egiazarian

Eldar Kurtić

Torsten Hoefler

Dan Alistarh<sup>∗</sup>

Institute of Science and Technology Austria (ISTA)

Institute of Science and Technology Austria (ISTA) & Red Hat AI

ETH Zürich

Institute of Science and Technology Austria (ISTA)

## Abstract

KV cache memory and bandwidth costs grow with context length and batch size, which limits efficient long-context inference. To address this bottleneck, we introduce WUSH-KV for low-bit KV-cache quantization. It adapts WUSH, which constructs a data-aware transform from the second-order statistics of both factors in a matrix product to reduce quantization error. WUSH-KV uses calibration data to construct separate key and value transforms, with the value transform folded into the model weights and the key transform applied after RoPE. The transforms can be paired with clipped quantizers. For one such quantizer, QuEST INT, we show that, under mild assumptions, the WUSH transform is near-optimal. With this quantizer, WUSH-KV reduces layerwise reconstruction error and achieves the lowest end-to-end perplexity among other tested transforms. For end-to-end evaluation, we integrate WUSH-KV into SGLang using OSCAR-style percentile-clipped affine quantization. At 2-bit, WUSH-KV performs comparably to or outperforms the OSCAR transform across all evaluated models and downstream tasks.

## 1 Introduction

Large language models are increasingly used with long contexts and large serving batches. This puts growing pressure on the memory capacity and bandwidth of inference systems. One important source of this cost is the key-value (KV) cache. During autoregressive generation, each attention layer stores the keys and values of previous tokens so that they do not need to be recomputed. The cache grows linearly with sequence length and batch size, and it is repeatedly read as new tokens are generated. As a result, the KV cache can become a major bottleneck for long-context inference. Low-bit quantization provides a simple way to reduce its memory footprint and data movement. Yet, aggressive quantization can introduce large errors into both attention scores and outputs.

Changes of basis (linear transforms) can make cached vectors easier to quantize, but the quality of the transform should also reflect how quantization errors propagate through attention. Key errors affect query-key products, whereas value errors affect projected attention outputs. This motivates separate transforms that account for both cache statistics and downstream sensitivity. WUSH (Chen et al., 2026) was recently introduced as a principled approach to joint weight-activation quantization. It constructs a closed-form invertible transform from the second-order statistics of both factors in a matrix product. This product-aware view naturally extends to the KV cache.

In this work, we adapt WUSH from weight-activation quantization to KV-cache quantization. We call the resulting method WUSH-KV. For each KV head, WUSH-KV constructs separate transforms for keys and values using calibration-time statistics. The key transform accounts for the query heads that consume the cached keys. The value transform accounts for the output-projection blocks that consume the cached values. The value-side transform can be folded into the model weights and therefore adds no online transform cost. The key-side transform and its query-side compensation are applied after headwise normalization (when present) and rotary positional encoding. The WUSH transforms operate independently of the scalar quantizer and can be paired with per-token clipped quantizers. WUSH-KV also retains sink and recent cache entries in full precision. Our theoretical analysis focuses on the QuEST (Panferov et al., 2025) quantizer. Under an additive rounding-noise model and explicit conditions on normalized tails and clipping errors, we show that the ideal WUSH transform is near-optimal among transforms with balanced coordinate sensitivity. The controlled reconstruction study uses the same quantizer to isolate transform quality, while the end-to-end evaluations use the percentile-clipped affine quantizer from the OSCAR (Zhou et al., 2026) concurrent work. In our experiments, WUSH-KV consistently reduces attention reconstruction error and achieves the lowest perplexity among all existing methods. In 2-bit downstream benchmarks, it scores higher than OSCAR on all four Qwen3-8B tasks and is competitive at 4B and 32B.

## 2 Related Work

KV cache quantization. There is a long line of work on this topic. KIVI (Liu et al., 2024) observes that keys and values have different quantization characteristics, so it quantizes keys per channel and values per token. Recent residual keys and values remain in full precision. KVQuant (Hooper et al., 2024) also exploits the channel-wise structure of keys. It quantizes keys before RoPE, uses sensitivity-weighted nonuniform quantization, and handles outliers with a per-vector dense-and-sparse representation. It keeps the first token in FP16 to protect the attention sink. AQUA-KV (Shutova et al., 2025) exploits cross-layer dependencies with compact adapters to predict cached keys and values and quantizes the remaining residual information. These methods focus on quantizer design, outlier handling, and selective high-precision retention.

Transform-based quantization. Another line of work changes the representation before quantization. QuaRot (Ashkboos et al., 2024) redistributes outliers with randomized Hadamard transforms. This enables end-to-end 4-bit quantization of weights, activations, and the KV cache. SpinQuant (Liu et al., 2025) shows that quantized accuracy can vary substantially across rotations. It learns two mergeable orthogonal rotations by minimizing a task loss on calibration data. For low-bit activations and KV caches, it also uses efficient fixed online Hadamard transforms. FlatQuant (Sun et al., 2025) learns Kronecker-structured transforms through calibration and applies head-wise transforms to keys and values in the KV cache. RotateKV (Su et al., 2025) specializes in rotations for aggressive 2-bit cache compression. It combines calibration-based channel reordering with grouped-head key rotations applied before RoPE. It also identifies additional attention-sink tokens and retains their KV entries in FP16. TurboQuant (Zandieh et al., 2026) randomly rotates vectors and applies scalar quantizers matched to the resulting coordinate distribution. For unbiased inner-product estimation, the published method additionally applies a one-bit-per-coordinate QJL sketch to the quantization residual. Subsequent analysis (Ben-Basat et al., 2026) notes that randomized Hadamard transforms can replace uniform random rotations in this family of quantizers. Practical implementations $( \mathrm { e . g . , \ v L L M ^ { 1 } ) }$ likewise use efficient Hadamard transforms but drop the QJL sketch as harmful. This motivates the normalized Hadamard transform as our practical TurboQuant-style fixed-transform baseline. Concurrent with our work, OSCAR (Zhou et al., 2026) estimates attention-aware covariance statistics offline, and constructs separate fixed orthogonal transforms for keys and values. It also calibrates per-layer clipping parameters that determine token-wise clipping thresholds. The method is implemented in SGLang (Zheng et al., 2024), a serving system compatible with paged KV caches. OSCAR restricts its key and value transforms to be orthogonal, whereas WUSH-KV allows general invertible transforms. This additional freedom permits anisotropic rescaling to jointly balance cache statistics and downstream sensitivity, which an orthogonal transform cannot generally realize.

## 3 Background on the WUSH Transform

Post-training quantization replaces a model’s weights, and sometimes its activations, with low-precision codes that share a scale within each group of elements. A handful of outliers carry far more amplitude than the rest, so they set the scale, and many of the available quantization levels go unused. Transforms are one of the remedies: for a matrix X and an invertible transform matrix ${ \mathbf { } } T ,$ quantize T X rather than X itself, and undo the transform with $\pmb { T } ^ { - 1 }$ afterwards. What makes a transform good is not a property of X alone: the error only matters through the matrix product $( \mathrm { e . g . } , W ^ { \top } X )$ in which X is one factor. The WUSH transform (Chen et al., 2026) therefore builds $\mathbf { T }$ in closed form from second-order statistics of both factors of that product. According to their analysis, this choice is provably optimal for floating-point formats and asymptotically optimal for integer ones. The transform is block diagonal, and its block width is the same as the quantization group size. This keeps it cheap to apply to activations at inference time. We follow the original WUSH construction and express it as a function of a Gram matrix and a Hessian, which is the form used throughout the rest of this paper.<sup>2</sup>

The loss a transform has to control. Let $W \in \mathbb { R } ^ { d \times d _ { \mathrm { o u t } } }$ and $\pmb { X } \in \mathbb { R } ^ { d \times d _ { \mathrm { b a t c h } } }$ be the weights and activations of a linear layer, restricted to the d input coordinates one block acts on. Chen et al. (2026) quantizes both factors at once. We will only ever quantize one of them, which is also the case to which their derivation reduces. We take that factor to be $X ,$ keep W exact, and let $\mathcal { Q }$ be a quantizer, specified in Section 4. The round trip through the invertible transform $\pmb { T } \in \mathbb { R } ^ { d \times d }$ leaves behind a perturbation ${ \pmb \varepsilon } = { \pmb T } ^ { - 1 } { \pmb \mathcal { Q } } \left( { \pmb T } { \pmb X } \right) - { \pmb X }$ . With $\pmb { H } = 2 \pmb { W } \pmb { W } ^ { \top }$ ， the loss this perturbation causes at the layer output is $\begin{array} { r } { \ell = \left\| \boldsymbol { W } ^ { \top } \boldsymbol { \varepsilon } \right\| _ { \mathrm { F } } ^ { 2 } = 2 ^ { - 1 } \sum _ { j } \varepsilon _ { j } ^ { \top } H \varepsilon _ { j } } \end{array}$ , a quadratic form in the columns $\varepsilon _ { j }$ of $\varepsilon .$ . The matrix H is the Hessian of ℓ with respect to a column of $\varepsilon .$

The WUSH construction. Let $M = X X ^ { \top }$ be the Gram matrix of X, $\pmb { \mathscr { H } } \in \left\{ \pm d ^ { - \frac { 1 } { 2 } } \right\}$ d×d a normalized (orthogonal) Hadamard matrix,<sup>3</sup> I the $d \times d$ identity, and $\gamma \geq 0$ a small damping ratio. The transform T applied to X is built as

$$
\begin{array} { r l r } & { } & { { \pmb { L } } { \pmb { L } } ^ { \top } = { \pmb { H } } + \gamma d ^ { - 1 } \mathrm { t r } \left( { \pmb { H } } \right) { \bf I } , \qquad { \pmb { U } } { \pmb { \Lambda } } { \pmb { U } } ^ { \top } = { \pmb { L } } ^ { \top } \left( { \pmb { M } } + \gamma d ^ { - 1 } \mathrm { t r } \left( { \pmb { M } } \right) { \bf I } \right) { \pmb { L } } , } \\ & { } & { c = ( 1 + \gamma ) ^ { \frac { 1 } { 2 } } \left( \mathrm { t r } \left( { \pmb { M } } \right) \right) ^ { \frac { 1 } { 2 } } \left( \mathrm { t r } \left( { \pmb { \Lambda } } ^ { \frac { 1 } { 2 } } \right) \right) ^ { - \frac { 1 } { 2 } } , \qquad { \pmb { T } } = c { \pmb { \mathcal { H } } } { \pmb { \Lambda } } ^ { - \frac { 1 } { 4 } } { \pmb { U } } ^ { \top } { \pmb { L } } ^ { \top } , \qquad } \end{array}\tag{1}
$$

where $\pmb { L } \pmb { L } ^ { \top }$ is the Cholesky decomposition and $U \pmb { \Lambda } \pmb { U } ^ { \top }$ is the symmetric eigendecomposition. The scalar c is introduced so that $\| \mathbfcal { T } \pmb { X } \| _ { \mathrm { F } } = \| \pmb { X } \| _ { \mathrm { F } }$ when $\gamma = 0$ , and approximately so otherwise.

Definition 1 For symmetric positive semidefinite M, $H \in \mathbb { R } ^ { d \times d }$ and a damping ratio $\gamma \geq 0$ that makes both damped matrices positive definite, write Wush $( M , H , \gamma )$ for the transform T that $E q . \ ( 1 )$ builds from them, in which M is the Gram matrix of the tensor being quantized and H is the Hessian of the loss with respect to that tensor’s perturbation.

We hold $\gamma$ fixed throughout and abbreviate Wush $( M , H , \gamma )$ to Wush $( M , H )$ . This construction is invariant to independent positive rescaling of its arguments, Wush $( \rho _ { \mathrm { M } } M , \rho _ { \mathrm { H } } H ) =$ Wush (M, H) for every $\rho _ { \mathrm { M } } , \rho _ { \mathrm { H } } > 0$ , so how the two matrices are normalized never has to be tracked. What remains, for any tensor we wish to quantize, is to write down the Hessian of the loss with respect to its perturbation.

## 4 WUSH-KV

This section specializes in WUSH within the key/value cache of grouped-query attention. We first identify the cached tensors and formulate their transforms, then specify the quantizer and its guarantees, and finally describe cache management, storage, and computation costs.

## 4.1 KV Cache in Grouped-Query Attention

Grouped-query attention (GQA). In a model of dimension $d _ { \mathrm { m o d e l } }$ , an attention module has $n _ { \mathrm { q } }$ query heads and $n _ { \mathrm { k v } }$ key/value heads of dimension $d ,$ so each key/value head is read by $n _ { \mathrm { q } } / n _ { \mathrm { k v } }$ query heads. A query head is indexed by the pair $( h , g )$ : the key/value head h it reads and its position $g = 1 , \ldots , n _ { \mathrm { q } } / n _ { \mathrm { k v } }$ within that group. Let $\boldsymbol { X } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times \boldsymbol { S } }$ be the attention module input over a context of S positions, one column each. Each head projects it with $W _ { \mathrm { Q } ( h , g ) } , W _ { \mathrm { K } ( h ) } , W _ { \mathrm { V } ( h ) } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d }$ . Let Norm denote any headwise normalization applied over the d coordinates of each projected query or key. This includes root mean square (RMS) normalization and reduces to the identity when no such normalization is used. The query and key normalizations may have distinct parameters, which we suppress in the notation. The rotary embedding RoPE $( \pmb { A } ) = [ R _ { 1 } \pmb { a } _ { 1 } , \dots , \pmb { R } _ { S } \pmb { a } _ { S } ]$ rotates each column by an orthogonal $\pmb { R _ { s } } \in \mathbb { R } ^ { d \times d }$ fixed by its position s. Write τ for the softmax temperature and $\pmb { \mathcal { M } } \in \{ 0 , - \infty \} ^ { S \times S }$ for the causal mask, zero where a query is allowed to attend to a position and −∞ elsewhere. The output projection $W _ { \mathrm { O } } \in \mathbb { R } ^ { n _ { \mathrm { q } } d \times d _ { \mathrm { m o d e l } } }$ splits into one row block $W _ { \mathrm { O } ( h , g ) } \in \mathbb { R } ^ { d \times d _ { \mathrm { m o d e l } } }$ per query head. The attention module output is then

$$
\begin{array} { c } { { \displaystyle Q _ { h , g } = \mathrm { R o P E } \left( \mathrm { N o r m } \left( W _ { \mathrm { Q } ( h , g ) } ^ { \top } \boldsymbol { X } \right) \right) , \qquad K _ { h } = \mathrm { R o P E } \left( \mathrm { N o r m } \left( W _ { \mathrm { K } ( h ) } ^ { \top } \boldsymbol { X } \right) \right) , } } \\ { { \displaystyle V _ { h } = W _ { \mathrm { V } ( h ) } ^ { \top } \boldsymbol { X } , \qquad P _ { h , g } = \mathrm { s o f t m a x } \left( \tau ^ { - 1 } K _ { h } ^ { \top } Q _ { h , g } + \boldsymbol { \mathcal { M } } \right) , \qquad } } \\ { { \displaystyle Q _ { h , g } = V _ { h } P _ { h , g } , \qquad \quad \boldsymbol { Y } = \sum _ { h , g } W _ { \mathrm { O } ( h , g ) } ^ { \top } O _ { h , g } , } } \end{array}\tag{2}
$$

where the softmax acts on each column.

Cached keys and values. Autoregressive generation grows the context one position at a time and re-evaluates Eq. (2) at each new S. Of the three projections $Q _ { h , g } , K _ { h }$ , and $V _ { h }$ only the keys and values must persist because every later position attends to all earlier ones, whereas the queries are used once and discarded. Each key/value head therefore caches $K _ { h } = [ k _ { h , 1 } , \ldots , k _ { h , S } ]$ and $V _ { h } = [ \boldsymbol { v } _ { h , 1 } , \ldots , \boldsymbol { v } _ { h , S } ]$ rather than recomputing them. Each new token appends one column to both caches, and later queries reuse the previously cached columns unchanged. Keys are stored after headwise normalization (when present) and rotary embedding, so $k _ { h , s }$ is read without further normalization or rotation.

## 4.2 WUSH-KV Transforms

Transform placement. A cached key is read only through ${ K } _ { h } ^ { \top } { Q } _ { h , g }$ and a cached value only through $W _ { \mathrm { O } ( h , g ) } ^ { \top } V _ { h }$ , and neither product changes when an invertible matrix is inserted between its factors,

$$
\begin{array} { r } { K _ { h } ^ { \top } Q _ { h , g } = \left( \mathbf T _ { \mathrm { K } ( h ) } K _ { h } \right) ^ { \top } \left( \mathbf T _ { \mathrm { K } ( h ) } ^ { - \top } Q _ { h , g } \right) , \quad W _ { \mathrm { O } ( h , g ) } ^ { \top } V _ { h } = \left( \mathbf T _ { \mathrm { V } ( h ) } ^ { - \top } W _ { \mathrm { O } ( h , g ) } \right) ^ { \top } \left( \mathbf T _ { \mathrm { V } ( h ) } V _ { h } \right) . } \end{array}\tag{3}
$$

With the quantizer of Section 4.3, we therefore cache $\mathcal { Q } \left( \pmb { T } _ { \mathrm { K } ( h ) } \pmb { K } _ { h } \right)$ and $\mathcal { Q } \left( \pmb { T } _ { \mathrm { V } ( h ) } \pmb { V } _ { h } \right)$ in place of the two tensors themselves. For each key/value head, we build one $d \times d$ transform for the keys and another for the values. The value-side transform is folded into the value and output projection weights offline before inference: we replace $W _ { \mathrm { V } ( h ) }$ with ${ \widehat { W } } _ { \mathrm { V } ( h ) } = { W } _ { \mathrm { V } ( h ) } { \cal T } _ { \mathrm { V } ( h ) } ^ { \top }$ and every $W _ { \mathrm { O } ( h , g ) }$ with $\widehat { W } _ { \mathrm { O } ( h , g ) } = T _ { \mathrm { V } ( h ) } ^ { - \top } W _ { \mathrm { O } ( h , g ) }$ . The key side does not disappear into the projection weights because the inserted transforms act after Norm and the position-dependent RoPE, and a dense $\pmb { T } _ { \mathrm { K } ( h ) }$ cannot generally be moved across these operations.

This post-RoPE placement is deliberate. As detailed in Section A, fully folding the key transform and its query-side compensation into the projection weights requires each to commute with RoPE and its respective learned RMS normalization. These joint commutation constraints leave only paired sign-flip transforms, which are trivial because they do not redistribute coordinate magnitudes. A separate alternative is to place an unrestricted transform before RoPE rather than require it to commute with RoPE. This retains the full $d \times d$ freedom, but its compensation depends on the cached position. At each decoding step, this would require restoring and rotating every cached key or applying a different compensation at every cache position, adding position-dependent work across the cache and complicating the attention kernel. We therefore retain the full $d \times d$ transform after $\mathrm { R o P E }$ accepting fixed online transforms for each new key and query in exchange for a simple cache representation.

Transform formulation. For the construction, we use local surrogate Hessians that retain the direct bilinear partner of each cached tensor: the queries of the group for a key and the output projection for a value. The transforms are

$$
\begin{array} { r l r } & { } & { { \cal H } _ { \mathrm { K } ( h ) } = 2 \displaystyle \sum _ { g } { \cal Q } _ { h , g } { \cal Q } _ { h , g } ^ { \top } , \qquad { \cal H } _ { \mathrm { V } ( h ) } = 2 \displaystyle \sum _ { g } { \cal W } _ { \mathrm { O } ( h , g ) } { \cal W } _ { \mathrm { O } ( h , g ) } ^ { \top } , } \\ & { } & { { \cal T } _ { \mathrm { K } ( h ) } = \mathrm { W U S H } \left( { \cal K } _ { h } { \cal K } _ { h } ^ { \top } , { \cal H } _ { \mathrm { K } ( h ) } \right) , \qquad { \cal T } _ { \mathrm { V } ( h ) } = \mathrm { W U S H } \left( { \cal V } _ { h } { \cal V } _ { h } ^ { \top } , { \cal H } _ { \mathrm { V } ( h ) } \right) , } \end{array}\tag{4}
$$

with $\pmb { K } _ { h } \pmb { K } _ { h } ^ { \top }$ and $V _ { h } V _ { h } ^ { \top }$ the Gram matrices of the tensors being quantized. Algorithm 1 summarizes the offline calibration procedure for each attention module. We accumulate statistics over all token positions in each calibration sequence, then sum them across sequences to avoid storing all calibration activations and projections at once.

Algorithm 1: WUSH-KV calibration   
Input: Weights $W _ { \mathrm { Q } } , W _ { \mathrm { K } } , W _ { \mathrm { V } } , W _ { \mathrm { O } }$ , calibration sequences $\left\{ X ^ { ( i ) } \right\} _ { i = 1 } ^ { n _ { \mathrm { c a l } } }$   
Output: Key transforms $\mathbf { \mathit { T } } _ { \mathrm { K } }$ and folded weights $\widehat { W } _ { \mathrm { V } } , \widehat { W } _ { \mathrm { O } }$   
1: for $i = 1 , \ldots , n _ { \mathrm { c a l } }$ independently do   
2: $\mathbf { } Q ^ { ( i ) }  \operatorname { R o P E } ( \operatorname { N o r m } ( W _ { \mathrm { Q } } ^ { \top } \mathbf { X } ^ { ( i ) } ) ) ; \quad \mathbf { K } ^ { ( i ) }  \operatorname { R o P E } ( \operatorname { N o r m } ( W _ { \mathrm { K } } ^ { \top } \mathbf { X } ^ { ( i ) } ) )$   
3: $V ^ { ( i ) }  W _ { \mathrm { V } } ^ { \top } X ^ { ( i ) }$   
4: $M _ { \mathrm { K } ( h ) } ^ { ( i ) } \gets \dot { \boldsymbol { K } } _ { h } ^ { ( i ) } \boldsymbol { K } _ { h } ^ { ( i ) \top } \quad \forall h ; \quad M _ { \mathrm { V } ( h ) } ^ { ( i ) } \gets \boldsymbol { V } _ { h } ^ { ( i ) } \boldsymbol { V } _ { h } ^ { ( i ) \top } \quad \forall h$   
5: Compute Hessian matrices ${ \pmb H } _ { \mathrm { K } ( h ) } ^ { ( i ) } , { \pmb H } _ { \mathrm { V } ( h ) } ^ { ( i ) }$ from $Q ^ { ( i ) } , W _ { \mathrm { O } }$ and optionally   
${ \cal K } ^ { ( i ) } , { \cal V } ^ { ( i ) } \quad \forall h$   
6: end for   
7: $\begin{array} { r } { \pmb { T } _ { \mathrm { K } ( h ) }  \mathrm { W u s H } ( \sum _ { i } \pmb { M } _ { \mathrm { K } ( h ) } ^ { ( i ) } , \sum _ { i } \pmb { H } _ { \mathrm { K } ( h ) } ^ { ( i ) } ) } \end{array}$ ∀ h   
8: $\begin{array} { r } { \mathbf { T } _ { \mathrm { V } ( h ) }  \mathrm { W u s H } ( \sum _ { i } M _ { \mathrm { V } ( h ) } ^ { ( i ) } , \sum _ { i } \pmb { H } _ { \mathrm { V } ( h ) } ^ { ( i ) } ) } \end{array}$ ∀ h   
9: $\widehat { W } _ { \mathrm { V } ( h ) }  W _ { \mathrm { V } ( h ) } \mathbf { \widetilde { T } } _ { \mathrm { V } ( h ) } ^ { \top } \quad \forall h ; \quad \widehat { W } _ { \mathrm { O } ( h , g ) }  T _ { \mathrm { V } ( h ) } ^ { - \top } W _ { \mathrm { O } ( h , g ) } \quad \forall h , g$

## 4.3 Quantizer and Near-Optimality

We use per-token quantization: the d channels of a key or value head form one group and share a scale. To analyze clipping, we use the projection step of QuEST (Panferov et al., 2025). For a nonzero group x, it sets the step $\Delta$ from the group’s root mean square (RMS)

and reconstructs at bin centers,

$$
\Delta = 2 \alpha _ { b } \left( 2 ^ { b } - 1 \right) ^ { - 1 } d ^ { - \frac { 1 } { 2 } } \left\| { \pmb x } \right\| , \qquad Q \left( { \pmb x } \right) = \Delta \left( \left\lfloor \Delta ^ { - 1 } { \pmb x } \right\rfloor + 2 ^ { - 1 } \right) ,\tag{5}
$$

with the floor clamped to $\left\lceil - 2 ^ { b - 1 } , 2 ^ { b - 1 } - 1 \right\rceil$ and ${ \mathcal { Q } } \left( \mathbf { 0 } \right) = \mathbf { 0 }$ . The $2 ^ { b }$ levels are evenly spaced between $\pm \alpha _ { b }$ times the group RMS, and entries outside this range are clipped. The constant $\alpha _ { b }$ minimizes the mean squared error of the corresponding fixed-scale grid on a standard Gaussian.

For the analysis, write $\ell ( \pmb { T } )$ for the quadratic output loss from Section 3 after quantizing T X and undoing T. We keep clipping exact and model rounding within the range by independent uniform noise, as specified in Section B.1. We call an invertible transform sensitivity-balanced when ${ \pmb T } ^ { - \top } { \pmb H } { \pmb T } ^ { - 1 }$ has equal diagonal entries.

Theorem 2 Let $M = X X ^ { \top }$ and H be symmetric positive definite, and let $\pmb { T } _ { \star }$ be $E q$ $( 1 )$ with $\gamma = 0$ . Under the Gaussian-tail and clipping-alignment conditions in Section $B . \mathcal { B } ,$ with bounds independent of d and $b ,$ every sensitivity-balanced invertible T satisfies

$$
\begin{array} { r } { \mathbb { E } [ \ell ( \pmb { T } _ { \star } ) ] \leq ( 1 + O ( \alpha _ { b } ^ { - 2 } ) ) \mathbb { E } [ \ell ( \pmb { T } ) ] \qquad \ \mathit { a s \ b }  \infty . } \end{array}\tag{6}
$$

The expectation is over rounding noise, and the constant depends only on the assumption bounds.

The undamped WUSH transform is sensitivity-balanced by Eq. (20). Section B proves the result and gives an explicit finite-bit bound for $\alpha _ { b } > 1$ . The guarantee concerns the ideal transform, while the experiments use damping.

## 4.4 Full-Precision Cache Window

We combine the quantized cache with two full-precision windows and quantize newly eligible entries in batches. The first $S _ { \mathrm { s i n k } }$ consecutive cache positions form a fixed full-precision sink window throughout generation. Among the remaining positions, the newest $S _ { \mathrm { k e e p } }$ form a rolling recent window. Each new chunk is first appended in full precision and is used to compute its attention output. We then quantize the largest multiple of $S _ { \mathrm { H u s h } }$ among the entries that lie outside both windows under the current cache length. Thus, $S _ { \mathrm { s i n k } } + S _ { \mathrm { k e e p } }$ entries per head remain persistently in full precision once the two windows no longer overlap, and fewer than $S _ { \mathrm { H u s h } }$ additional entries can remain temporarily in full precision. Write $S _ { \mathrm { q u a n t } }$ for the inclusive right boundary of the quantized middle region, initialized to zero for an empty cache. Algorithm 2 gives this update for one attention module (some symbols are overloaded for transformed-coordinate counterparts).

Algorithm 2: WUSH-KV inference   
Input: Weights $W _ { \mathrm { Q } } , W _ { \mathrm { K } } , \widehat { W } _ { \mathrm { V } } , \widehat { W } _ { \mathrm { O } }$ , transforms $\mathbf { { \mathit { T } } _ { K } }$ , cache K, V in transformed   
coordinates with length $S$ and quantized boundary $S _ { \mathrm { q u a n t } }$ , new activation chunk $X ^ { \prime }$ of   
length $S ^ { \prime } ,$ , window sizes $S _ { \mathrm { s i n k } } , S _ { \mathrm { k e e p } }$ with flush size $S _ { \mathrm { H u s h } }$   
Output: Attention output $\mathbf { { \mathbf { } } Y ^ { \prime } } ,$ updated cache $\kappa , V$ of length S and quantized boundary   
S<sub>quant</sub>   
1: $Q ^ { \prime } \gets \mathrm { R o P E }$ Norm $( { \pmb W } _ { \mathrm { Q } } ^ { \top } { \pmb X } ^ { \prime } ) )$ ; K<sup>′</sup> ← RoPE  Norm $( W _ { \mathrm { K } } ^ { \top } \pmb { X } ^ { \prime } ) )$   
2: $V ^ { \prime }  \widehat { W } _ { \mathrm { V } } ^ { \top } X ^ { \prime }$   
3: $\begin{array} { r } { \pmb { Q } _ { h , g } ^ { \prime }  \dot { \pmb { T } } _ { \mathrm { K } ( h ) } ^ { - \top } \pmb { Q } _ { h , g } ^ { \prime } \quad \forall h , g ; \quad \pmb { K } _ { h } ^ { \prime }  \pmb { T } _ { \mathrm { K } ( h ) } \pmb { K } _ { h } ^ { \prime } \quad \forall h } \end{array}$   
4: Append $\mathbf { \nabla } K ^ { \prime } , V ^ { \prime }$ to $\kappa , V$ in full precision; $S \gets S + S ^ { \prime }$   
5: Compute $O ^ { \prime }$ from $Q ^ { \prime } , K , V$ using grouped-query attention (mixed precision)   
6: $Y ^ { \prime } \gets \widehat { \pmb { W } } _ { \mathrm { 0 } } ^ { \top } O ^ { \prime }$   
7: $S _ { \mathrm { s t a r t } } \gets \operatorname* { m a x } \left( S _ { \mathrm { q u a n t } } , S _ { \mathrm { s i n k } } \right)$   
8: $S _ { \mathrm { q u a n t } }  \mathrm { m a x } ( \dot { S } _ { \mathrm { q u a n t } } , S _ { \mathrm { s t a r t } } + \lfloor ( S - S _ { \mathrm { k e e p } } - S _ { \mathrm { s t a r t } } ) / S _ { \mathrm { f u s h } } \rfloor S _ { \mathrm { f u s h } } )$   
9: if $S _ { \mathrm { q u a n t } } > S _ { \mathrm { s t a r t } }$ then   
10: $k _ { h , s } \gets \mathcal { Q } \left( k _ { h , s } \right) \quad \forall h , S _ { \mathrm { s t a r t } } < s \leq S _ { \mathrm { q u a n t } }$   
11: $\pmb { v } _ { h , s }  \mathscr { Q } ( \pmb { v } _ { h , s } ) \quad \forall h , S _ { \mathrm { s t a r t } } < s \leq S _ { \mathrm { q u a n t } }$   
12: end if

## 4.5 Storage and Computation Costs

For a model with $n _ { \mathrm { l a y e r } }$ attention modules, at sequence length S, the model caches $2 n _ { \mathrm { l a y e r } } n _ { \mathrm { k v } } d S$ key/value elements. Because the full-precision windows typically cover less than 1% of the maximum sequence length in the downstream tasks, they increase the effective cache quantization bitwidth only slightly. The method stores $n _ { \mathrm { l a y e r } } n _ { \mathrm { k v } }$ key transforms of size $d \times d ,$ fixed once at calibration time and shared by every position. Their storage overhead is small relative to both the weights and the KV cache at practical sequence lengths. The value-side transform adds no online cost because both halves are folded into the weights. On the key side, each new key costs one d × d product, and so does each query, or $d ^ { 2 }$ multiply-accumulates per position for each of the $n _ { \mathrm { k v } }$ key heads and the $n _ { \mathrm { q } }$ query heads.

## 5 Experiments

We evaluate WUSH-KV at three levels: controlled attention-stage reconstruction error, WikiText-2 (Merity et al., 2017) perplexity, and downstream reasoning accuracy.

## 5.1 Attention-Stage Quantization Error

End-to-end metrics do not isolate the contribution of the KV transforms. We therefore conduct a controlled ablation of the transform choices, measuring attention-stage reconstruction error with keys and values quantized separately and jointly.

Setup. We study all 36 attention modules of Qwen3-8B. We calibrate on 128 FineWeb-Edu (Penedo et al., 2024) sequences and evaluate on 32 disjoint sequences, each of length 1024. We evaluate reconstruction at 2, 3, and 4 bits in a prefill-like setting. Every transform uses the QuEST quantizer in Eq. (5), which holds the quantization rule fixed and isolates the effect of the transform. The entire KV sequence is quantized without the full-precision cache windows in Section 4.4, which isolates the effect of the transforms on reconstruction error. Reference activations follow the clean BF16 model trajectory.

Transforms. We compare six key/value transform pairs: identity (I), random orthogonal (R), normalized Hadamard (H), OSCAR (Zhou et al., 2026), WUSH-KV (WUSH), and an attention-aware WUSH-KV variant (WUSH-A). This H choice reflects the practical TurboQuant-style implementations. H does not reproduce the complete TurboQuant codec because every method in this experiment uses the same QuEST quantizer, isolating the effect of the transform. WUSH, our proposed method, learns separate $\mathbf { z } _ { \mathrm { K } }$ and $\mathbf { \mathit { 1 } } _ { \mathrm { V } }$ from the Gram matrices and simple Hessians in Eq. (4). We use $\gamma = 1 0 ^ { - 2 }$ for WUSH. WUSH-A uses the same construction and damping but replaces the simple Hessians with the attentionaware Hessians derived in Section C. The attention-aware Hessians require only forward-pass quantities, so WUSH-A needs neither backpropagation nor Hessian estimation through automatic differentiation. An empirical-Fisher alternative could use language-model-loss gradients, but we do not evaluate it because backward-pass calibration is too expensive.

2-Bit Module Output (Quantizing Keys and Values)  
![](images/6dba77c4bc71b76df0261709ff8a877b3f41bf5da020084eb0886f766575af68.jpg)  
Figure 1: Relative (squared) $\mathrm { L _ { 2 } }$ module-output errors for I, R, H, OSCAR, WUSH, and WUSH-A with both keys and values quantized at 2-bit.

Results. For each module and measured quantity A, we report $\| \widehat { A } - A \| _ { \mathrm { F } } ^ { 2 } / \| A \| _ { \mathrm { F } } ^ { 2 }$ , summing the numerator and denominator separately over evaluation sequences. Figure 1 measures module-output error after the output projection and before the residual connection, with both keys and values quantized. The largest layerwise reductions occur in the first few attention modules. At 2-bit, when both keys and values are quantized, the geometric-mean module-output errors are 0.208 for WUSH and 0.195 for WUSH-A, compared with 0.325 for H and 0.309 for OSCAR. In Section D.1, the additional 2-bit plots in Figure 2 show that WUSH reduces error in the query-key dot products, softmax probabilities, and module output when only keys are quantized, as well as module-output error when only values are quantized. The 3-bit and 4-bit plots appear in Figure 3. Across bitwidths, WUSH and WUSH-A achieve lower module-output error than H and OSCAR for most attention modules in this setting and provide the strongest overall results among the evaluated transforms. WUSH-A offers only a modest improvement over WUSH here. We therefore use the simpler WUSH Hessians for the main method. WUSH-A remains a practical alternative for future models or settings in which attention-aware sensitivity provides a larger gain.

## 5.2 Perplexity Evaluations

We first test whether the lower attention-stage reconstruction error of WUSH translates into better language-model quality by measuring Qwen3-8B perplexity on WikiText-2.

Setup. For WUSH and OSCAR calibration, we use 128 FineWeb-Edu sequences of length 32,768. We use YaRN with factor 4 for calibrations and evaluations on Qwen3- 8B. For WUSH, following Algorithm 1, an unquantized forward pass accumulates key Gram matrices after RMS normalization and RoPE, value Gram matrices, and simple per-head Hessians defined in Eq. (4). We then construct one key and one value transform for every attention module and KV head with $\gamma = 1 0 ^ { - 2 }$ . Calibration is performed once, and the resulting transforms are reused across all cache bitwidths. On a single NVIDIA L40S GPU, this onetime Qwen3-8B calibration takes about 12 minutes and peaks at 39 GiB of GPU memory. All quantized runs use the $\mathrm { Q u E S T }$ quantizer in Eq. (5), matching our theoretical analysis. We compare identity (I), random orthogonal

Table 1: Qwen3-8B WikiText-2 perplexity. The full-precision baseline is 9.72.
<table><tr><td>Transform</td><td>4-Bit 3-Bit</td><td>2-Bit</td></tr><tr><td>I</td><td>11.36 15.29</td><td>38.08</td></tr><tr><td>R</td><td>9.77 10.18</td><td>14.22</td></tr><tr><td>H</td><td>9.77 10.06</td><td>258.29</td></tr><tr><td>OSCAR</td><td>9.77 10.19</td><td>13.74</td></tr><tr><td>WUSH</td><td>9.73 9.82</td><td>10.51</td></tr></table>

(R), normalized Hadamard (H), OSCAR, and WUSH-KV (WUSH) at 4, 3, and 2 bits. For evaluation, we concatenate and tokenize the WikiText-2 test split, then divide it into non-overlapping sequences of length 2,048. Each sequence starts with an empty cache and is processed in 16-token forward chunks. Perplexity is computed from the average next-token negative log-likelihood. For the quantized runs, we use the full-precision cache windows in Section 4.4 with $S _ { \mathrm { s i n k } } = 1 6 , S _ { \mathrm { k e e p } } = 1 2 8$ , and $S _ { \mathrm { f l u s h } } = 1 6$

Results. Table 1 shows that WUSH achieves the lowest perplexity among the quantized transforms at every bitwidth. OSCAR remains competitive at higher bitwidths, but WUSH’s advantage grows under more aggressive quantization: at 2-bit, WUSH obtains 10.51 perplexity compared with OSCAR’s 13.74. WUSH otherwise remains close to the full-precision baseline, while the other transforms degrade more sharply as the bitwidth decreases. The extreme 2-bit perplexity for H arises from spikes in the first attention module’s key vectors that are concentrated in essentially one coordinate before the transform. The Hadamard transform spreads these spikes into nearly equal-magnitude coefficients close to the midpoints between adjacent 2-bit reconstruction levels, causing dense quantization error.

## 5.3 Downstream Benchmarks

We next evaluate whether WUSH-KV’s relative advantage carries over to downstream reasoning accuracy. For these end-to-end evaluations, we integrate WUSH-KV into SGLang with the OSCAR quantizer (Zhou et al., 2026). This quantizer combines percentile clipping with asymmetric affine scaling and therefore does not readily admit the exact same proof as $\mathrm { Q u E S T }$ . Its adaptive zero point can use the available quantization levels more efficiently for shifted or asymmetric groups, while percentile clipping suppresses rare large errors. In our experiments, this tradeoff yields better end-to-end quality than symmetric QuEST, despite slightly higher average local squared error. Using the same quantizer also enables a direct comparison with the concurrent OSCAR method.

Setup. We evaluate Qwen3-4B-Thinking-2507, Qwen3-8B, and Qwen3-32B on AIME 2025 (Zhang $\&$ Team, 2025), MATH-500 (Lightman et al., 2024), GPQA Diamond (Rein et al., 2024), and LiveCodeBench v6 (Jain et al., 2025) with stochastic sampling over three seeds. We calibrate WUSH separately for each model on FineWeb-Edu, as in the perplexity evaluation, and reuse its transforms across downstream tasks. We use YaRN with factor 4 for calibrations and evaluations on Qwen3-8B and Qwen3-32B, and keep it disabled on Qwen3-4B-Thinking-2507. For the OSCAR baseline, we use its released GPQA Diamond calibrated transforms but were unable to exactly reproduce all scores reported in the concurrent preprint (Zhou et al., 2026), so we report our runs under the same SGLang evaluation pipeline. For a transformed token-head group $\pmb { x } \in \mathbb { R } ^ { d }$ at bitwidth $b ,$ the OSCARstyle quantizer clips the coordinates at their κ-quantile, derives an affine scale $\Delta$ and zero point z from the clipped range, rounds, and reconstructs,

$$
\begin{array} { r l } & { \quad x _ { \operatorname* { m i n } } = \operatorname* { m a x } \left\{ \operatorname* { m i n } x _ { i } , - \mathrm { q u a n t i l e } _ { \kappa } \left( \left| x _ { i } \right| \right) \right\} , \quad x _ { \operatorname* { m a x } } = \operatorname* { m i n } \left\{ \operatorname* { m a x } x _ { i } , \mathrm { q u a n t i l e } _ { \kappa } \left( \left| x _ { i } \right| \right) \right\} , } \\ & { \Delta = \left( 2 ^ { b } - 1 \right) ^ { - 1 } \left( x _ { \operatorname* { m a x } } - x _ { \operatorname* { m i n } } \right) , \quad z = - \Delta ^ { - 1 } x _ { \operatorname* { m i n } } , \quad \mathcal { Q } \left( x \right) = \Delta \left( \left| \Delta ^ { - 1 } x + z \mathbf { 1 } \right| - z \mathbf { 1 } \right) . } \end{array}\tag{7}
$$

with quantile (|x<sub>i</sub>|) taken over $i = 1 , \ldots , d$ and ⌊·⌉ denoting round-to-nearest clamped to $\left[ 0 , 2 ^ { b } - 1 \right]$ . We use $\kappa _ { \mathrm { K } } = 0 . 9 6$ for all three models and $\kappa \mathrm { v } = 0 . 9 2$ for Qwen3-4B-Thinking-2507 and Qwen3-8B, increasing $\kappa _ { \mathrm { V } }$ to 0.96 for Qwen3-32B. We reuse OSCAR’s tuned clipping ratios for WUSH-KV without retuning them for the WUSH-transformed K/V distributions. These ratios may therefore be suboptimal for WUSH-KV, leaving potential headroom from WUSH-specific clipping optimization. We provide a quantizer ablation and calibration controls in Section D.2. All quantized benchmark runs use $S _ { \mathrm { s i n k } } = 6 4 , S _ { \mathrm { k e e p } } = 2 5 6$ , and $S _ { \mathrm { f l u s h } } = 8$ , matching OSCAR’s BF16 sink and recent windows with page-sized INT2 flushing. The OSCAR SGLang integration updates cache storage before attention, while still using the current token or prefill chunk’s K/V in BF16.

Results. Table 2 reports accuracy and generation behavior across model sizes. The budgetexhaustion rate is the fraction of samples that reach the task-specific generation limit without a valid answer. The limit is 65,536 tokens for MATH-500 and GPQA Diamond and 92,160 tokens for AIME 2025 and LiveCodeBench v6. For a strong direct comparison, we use OSCAR’s selected 2-bit quantizer and cache policy without tuning them specifically for WUSH-KV. Even in this OSCAR-favored setup, WUSH-KV and OSCAR are close when OSCAR maintains strong accuracy, including across the 4B tasks and most 32B tasks. At

Table 2: Downstream results with 2-bit KV caches. Each cell gives the task score (%) with its standard deviation over three seeds. The gray second line gives the mean generated-token count in thousands (k) and the budget-exhaustion rate. Bold marks the best quantized mean score and the shortest mean output for each model and task.
<table><tr><td>Model</td><td>Method</td><td>AIME 2025</td><td>MATH-500</td><td></td><td>GPQA Diamond LiveCodeBench v6</td></tr><tr><td rowspan="2">T7hg-k2507 -4B-</td><td>BF16 TurboQuant-Style H Transform</td><td>77.8±5.1 21.8k 0% 0.0 ±0.0 91.8k 100%</td><td>89.5±1.5 5.8k 0% 18.7±1.0 52.9k 76%</td><td>69.9±2.0 9.0k 0% 16.2±1.8 56.4k 68%</td><td>57.1 ± 0.6 19.2k 0% 0.0±0.0 91.7k 99% 50.3±4.5</td></tr><tr><td>OSCAR WUSH-KV WUSH Transform BF16</td><td>65.6 ± 1.9 24.2k 0% 71.1±6.9 24.3k 0% 67.8±1.9 45.4k 11%</td><td>89.3±0.5 6.6k 0% 90.1 ± 0.5 6.6k 0% 95.1 ± 0.3</td><td>65.3±2.8 10.1k 1% 66.3±2.8 10.0k 0% 57.4±3.8</td><td>25.0k 2% 52.4± 0.3 23.0k 0% 47.8±2.2</td></tr><tr><td>e-8B</td><td>TurboQuant-Style H Transform OSCAR WUSH-KV WUSH Transform</td><td>0.0±0.0 92.1k 98% 34.4±8.4 63.0k42% 46.7±3.3 53.4k 12%</td><td>10.8k 1% 21.9±0.1 52.3k 75% 89.3±1.1 16.0k 6% 92.9± 0.1 12.2k 2%</td><td>16.5k 1% 30.1±2.8 28.6k 29% 52.9±3.0 18.3k 1% 54.5±2.3 16.3k 1%</td><td>45.7k 11% 1.0 ± 0.3 87.8k 89% 20.4±0.3 73.8k 71% 35.2±0.7 53.9k 26%</td></tr><tr><td>-32B</td><td>BF16 TurboQuant-Style H Transform OSCAR WUSH-KV WUSH Transform</td><td>74.4±5.1 37.1k 6% 1.1 ±1.9 89.2k 91% 63.3± 3.3 47.2k 9% 60.0±5.8 47.7k 9% 10.6k 2%</td><td>94.7±1.2 7.1k 1% 0.0±0.0 65.5k 100% 93.1 ± 0.7 9.7k 2% 91.9 ±1.4</td><td>66.7±2.7 10.5k 0% 0.0±0.0 65.5k 100% 60.6±3.0 13.6k 2% 59.3±2.0</td><td>61.9±0.3 40.9k 3% 1.0 ±1.2 80.2k 73% 27.8±1.4 71.4k 53%</td></tr></table>

8B, WUSH-KV scores higher than OSCAR on all four benchmarks. When OSCAR degrades sharply, WUSH-KV can be substantially better, as on LiveCodeBench v6 for both 8B and 32B. Relative to BF16, WUSH-KV’s degradation is comparatively consistent across model sizes, whereas OSCAR’s varies more by model and task. WUSH-KV generates at most about 1.5× as many tokens as BF16 in these tasks, well below the nominal 8× reduction in storage per quantized KV entry from BF16 to INT2. Thus, the longer outputs do not erase the expected cache-memory benefit. Hadamard’s large score losses and frequent budget exhaustion show that this fixed transform alone is insufficient for reliable 2-bit KV-cache quantization across these tasks.

We additionally evaluate long-context RULER NIAH (Hsieh et al., 2024) and MRCR (Vodrahalli et al., 2024), finding that WUSH-KV retains higher accuracy than OSCAR at the longest tested lengths. The detailed results are presented in Section D.3.

## 6 Conclusion

We presented WUSH-KV for low-bit KV-cache quantization in grouped-query attention. The method uses calibration data to construct WUSH transforms that are applied to cached keys and values before quantization. These transforms can be paired with per-token clipped quantizers. For QuEST, our analysis provides a clipping-aware near-optimality guarantee among sensitivity-balanced transforms under the stated noise model and assumptions. Our end-to-end downstream benchmarks pair the transforms with an OSCAR-style percentileclipped affine quantizer. The results show that the WUSH transform can improve the quality of low-bit KV cache quantization. One limitation is that the dense key-side transform must remain online because it is applied after RoPE, which adds extra computation during inference. The value-side transform can be folded into the model weights, though. WUSH-KV supports the use of a more general calibration Hessian to construct the transforms. Future work could study cheaper structured transforms, stronger attention-aware sensitivity measures, and system-level optimization and throughput evaluation.

## Acknowledgements

This research was funded in part by the Austrian Science Fund (FWF) 10.55776/COE12 and was partially supported by a generous grant from NVIDIA. The authors would like to thank Verda Cloud for computational support, and in particular, Paul Chang for his consistent, prompt, and generous help throughout the project.

## References

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-free 4- bit inference in rotated LLMs. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 100213–100240. Curran Associates, Inc., 2024. doi: 10.52202/079017-3180. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/b5b93943678 9f76f08b9d0da5e81af7c-Paper-Conference.pdf.

Ran Ben-Basat, Yaniv Ben-Itzhak, Gal Mendelson, Michael Mitzenmacher, Amit Portnoy, and Shay Vargaftik. A note on TurboQuant and the earlier DRIVE/EDEN line of work, 2026. URL https://arxiv.org/abs/2604.18555.

Jiale Chen, Vage Egiazarian, Roberto L. Castro, Torsten Hoefler, and Dan Alistarh. WUSH: Near-optimal adaptive transforms for LLM quantization. In Forty-third International

Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=ZsEC xUkbKB.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. KVQuant: Towards 10 million context length LLM inference with KV cache quantization. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 1270–1303. Curran Associates, Inc., 2024. doi: 10.522 02/079017-0040. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ file/028fcbcf85435d39a40c4d61b42c99a4-Paper-Conference.pdf.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? In First Conference on Language Modeling, 2024. URL https: //openreview.net/forum?id=kIoBbc76Sy.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview .net/forum?id=chfJJYC3iL.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 39578–39601, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/aca97732 e30bcf1303bc22ac3924fd16-Paper-Conference.pdf.

Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. SpinQuant: LLM quantization with learned rotations. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 92009–92032, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/f ile/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 32332–32344. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235 /liu24bz.html.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=Byj72udxe.

Andrei Panferov, Jiale Chen, Soroush Tabesh, Mahdi Nikdan, and Dan Alistarh. QuEST: Stable training of LLMs with 1-bit weights and activations. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 47820–47836. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/panferov25a.html.

Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 30811–30849. Curran Associates, Inc., 2024. doi: 10.52202/07901 7-0970. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/370 df50ccfdf8bde18f8f9c2d9151bda-Paper-Datasets\_and\_Benchmarks\_Track.pdf.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level Googleproof Q&A benchmark. In First Conference on Language Modeling, 2024. URL https: //openreview.net/forum?id=Ti67584b98.

Alina Shutova, Vladimir Malinovskii, Vage Egiazarian, Denis Kuznedelev, Denis Mazur, Surkov Nikita, Ivan Ermakov, and Dan Alistarh. Cache me if you must: Adaptive key-value quantization for large language models. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 55451–55473. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/shutova25a.html.

Zunhai Su, Hanyu Wei, Zhe Chen, Wang Shen, Linge Li, Huangqi Yu, and Kehong Yuan. RotateKV: Accurate and robust 2-bit KV cache quantization for LLMs via outlier-aware adaptive rotations. In James Kwok (ed.), Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25, pp. 6200–6208. International Joint Conferences on Artificial Intelligence Organization, 8 2025. doi: 10.24963/ijcai.2025/690. URL https://www.ijcai.org/proceedings/2025/690. Main Track.

Yuxuan Sun, Ruikang Liu, Haoli Bai, Han Bao, Kang Zhao, Yuening Li, Jiaxin Hu, Xianzhi Yu, Lu Hou, Chun Yuan, Xin Jiang, Wulong Liu, and Jun Yao. FlatQuant: Flatness matters for LLM quantization. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 57587–57613. PMLR, 13–19 Jul 2025. URL https: //proceedings.mlr.press/v267/sun25l.html.

Kiran Vodrahalli, Santiago Ontanon, Nilesh Tripuraneni, Kelvin Xu, Sanil Jain, Rakesh Shivanna, Jeffrey Hui, Nishanth Dikkala, Mehran Kazemi, Bahare Fatemi, Rohan Anil, Ethan Dyer, Siamak Shakeri, Roopali Vij, Harsh Mehta, Vinay Ramasesh, Quoc Le, Ed Chi, Yifeng Lu, Orhan Firat, Angeliki Lazaridou, Jean-Baptiste Lespiau, Nithya Attaluri, and

Kate Olszewska. Michelangelo: Long context evaluations beyond haystacks via latent structure queries, 2024. URL https://arxiv.org/abs/2409.12640.

Amir Zandieh, Majid Daliri, Majid Hadian, and Vahab Mirrokni. TurboQuant: Online vector quantization with near-optimal distortion rate. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 56418–56439, 2026. URL https://proceedings.iclr .cc/paper\_files/paper/2026/file/5c802ef38ab6e366c2ea06eee554c088-Paper-Con ference.pdf.

Yifan Zhang and Math-AI Team. American invitational mathematics examination (AIME) 2025, 2025. URL https://huggingface.co/datasets/math-ai/aime25.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. SGLang: Efficient execution of structured language model programs. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 62557–62583. Curran Associates, Inc., 2024. doi: 10.52202/079017-2000. URL https://proceedings.neurips.cc/pap er\_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference. pdf.

Zhongzhu Zhou, Donglin Zhuang, Jisen Li, Ziyan Chen, Shuaiwen Leon Song, Ben Athiwaratkun, and Xiaoxia Wu. OSCAR: Offline spectral covariance-aware rotation for 2-bit KV cache quantization, 2026. URL https://arxiv.org/abs/2605.17757.

## Appendix A. Alternative Key-Transform Placements

This appendix expands the design tradeoff summarized in Section 4. We first characterize transforms that can pass through RoPE and then consider unrestricted transforms placed before RoPE.

## A.1 RoPE-commuting transforms

On the basis that groups the rotary coordinate pairs, standard RoPE has the form

$$
\mathbf { { \cal R } } _ { s } = \mathrm { b l o c k d i a g } _ { j } \left( \mathbf { { \cal R } } \left( s \omega _ { j } \right) \right) , \qquad \mathbf { { \cal R } } \left( \theta \right) = \left( \cos \theta \mathrm { ~  ~ \left. ~ - \sin \theta \right) ~ } \right) .\tag{8}
$$

For distinct nondegenerate rotary frequencies, the real invertible transforms that commute with every $\mathbf { \delta } _ { R _ { s } }$ are

$$
T _ { \mathrm { K } ( h ) } = \mathrm { b l o c k d i a g } _ { j } \left( \rho _ { h , j } R \left( \theta _ { h , j } \right) \right) , \qquad T _ { \mathrm { K } ( h ) } R _ { s } = R _ { s } T _ { \mathrm { K } ( h ) } , \qquad \rho _ { h , j } > 0 .\tag{9}
$$

Each two-dimensional block lies in the span of $\big (  _ { 0 } ^ { 1 0 } \big )$ and $\left( { \begin{array} { l l } { 0 } & { - 1 } \\ { 1 } & { 0 } \end{array} } \right)$ , while the distinct frequencies prevent mixing between rotary pairs. In the absence of post-projection normalization, such

transforms and their inverse-transposes could pass through RoPE and be folded into the key and query projections.

A learned coordinatewise normalization further restricts exact folding. For example, write RMS normalization with gain ν as

$$
\operatorname { N o r m } \left( \pmb { x } \right) = d ^ { \frac { 1 } { 2 } } \left\| \pmb { x } \right\| ^ { - 1 } \operatorname { d i a g } \left( \pmb { \nu } \right) \pmb { x } .\tag{10}
$$

Moving a linear transform through this fixed normalization for every input requires

$$
T \operatorname { N o r m } \left( x \right) = \operatorname { N o r m } \left( T x \right) \quad \forall x \neq \mathbf { 0 } \quad \implies \quad T ^ { \top } T = \mathbf { I } , \qquad T \operatorname { d i a g } \left( \nu \right) = \operatorname { d i a g } \left( \nu \right) T .\tag{11}
$$

For nonzero gain entries, equality for every nonzero input forces T to commute with diag (ν) and preserve the RMS denominator $( \pmb { T } ^ { \top } \pmb { T } = \mathbf { I } )$ . The corresponding conditions must hold for the key transform and its inverse-transpose under the respective key and query normalizations. For generic learned gains, combining these conditions with RoPE commutation leaves only $\begin{array} { r } { \mathbf { T } _ { \mathrm { K } ( h ) , j } \in \left\{ \left( \begin{array} { l } { 1 \mathrm { ~ 0 ~ } } \\ { 0 \mathrm { ~ 1 ~ } } \end{array} \right) , \left( \begin{array} { l l } { - \bar { 1 } } & { 0 } \\ { 0 } & { - 1 } \end{array} \right) \right\} } \end{array}$ . These paired sign flips leave coordinate magnitudes unchanged and therefore do not redistribute quantization difficulty for a symmetric quantizer such as Eq. (5).

If the learned normalization gains ν are also allowed to change as part of the KV-cache quantization reparameterization, additional transforms may become foldable. However, such reparameterizations remain constrained by RoPE and by the structure of the normalization, and introduce additional design choices beyond folding a transform into fixed projection and normalization parameters. We therefore restrict our discussion to the fixed-normalization setting above.

## A.2 Full pre-RoPE transforms

A generic transform placed before RoPE restores the full d × d freedom but introduces position-dependent compensation. Let $k _ { h , s }$ and $^ { q _ { h , g , t } }$ denote the key at token position s and the query at token position t, respectively. Since $\mathbf { \delta } _ { R _ { s } }$ is orthogonal, the normalized pre-RoPE key is ${ \cal R } _ { s } ^ { \top } k _ { h , s }$ . If the cache instead stores $\pmb { T } _ { \mathrm { K } ( h ) } \pmb { R } _ { s } ^ { \top } \pmb { k } _ { h , s } ,$ , then

$$
\begin{array} { r } { \pmb { k } _ { h , s } ^ { \top } \pmb { q } _ { h , g , t } = \left( \pmb { T } _ { \mathrm { K } ( h ) } \pmb { R } _ { s } ^ { \top } \pmb { k } _ { h , s } \right) ^ { \top } \pmb { T } _ { \mathrm { K } ( h ) } ^ { - \top } \pmb { R } _ { s } ^ { \top } \pmb { q } _ { h , g , t } . } \end{array}\tag{12}
$$

Decoding must therefore restore each cached key before applying its positional rotation or incorporate a different compensation for every cache position. The current post-RoPE placement instead retains a full transform and a simple cache representation at the cost of fixed online transforms for each new key and query.

## Appendix B. Clipping-Aware Near-Optimality for the QuEST Quantizer

We prove Theorem 2 by starting from the unclipped rounding loss, which WUSH minimizes, and then controlling the effect of clipping. Clipping adds error outside the range and removes rounding noise from those coordinates. Both effects must be included in the comparison.

## B.1 Rounding and clipping errors

We keep X and H fixed, with $M = X X ^ { \top }$ as in Section 3. Each column $\mathbf { \Delta } _ { \mathbf { x } _ { j } }$ is one quantization group. Following Chen et al. (2026), we approximate rounding with independent uniform noise, but keep clipping exact. This is a model of the rounding error, not an identity for the deterministic quantizer.

Fix an invertible T . For group j, its RMS after the transform and its quantization step are

$$
r _ { j } = d ^ { - \frac { 1 } { 2 } } \left\| T { \pmb x } _ { j } \right\| , \qquad \Delta _ { j } = 2 \alpha _ { b } \left( 2 ^ { b } - 1 \right) ^ { - 1 } r _ { j } .\tag{13}
$$

The clipping endpoints are $\pm \alpha _ { b } r _ { j }$ . The error caused by clipping alone, before undoing the transform, is

$$
\begin{array} { r } { e _ { j } = \mathrm { c l i p } \left( { \cal T } x _ { j } , - \alpha _ { b } r _ { j } , \alpha _ { b } r _ { j } \right) - { \cal T } x _ { j } . } \end{array}\tag{14}
$$

Here clip acts coordinatewise, and $e _ { j , i }$ denotes coordinate i of group $j$ . Inside the range, $e _ { j , i } = 0$ and we replace the rounding error with $\Delta _ { j } \xi _ { j , i }$ , where all $\xi _ { j , i }$ are independent and uniformly distributed on $[ - 2 ^ { - 1 } , 2 ^ { - 1 } ]$ . Outside the range, the error is exactly $e _ { j , i }$ . Thus, the total round-trip perturbation $\varepsilon _ { j }$ satisfies

$$
\left( \mathbf { T } \pmb { \varepsilon } _ { j } \right) _ { i } = \left\{ \begin{array} { l l } { \Delta _ { j } \xi _ { j , i } , } & { | ( \mathbf { T } \pmb { x } _ { j } ) _ { i } | \leq \alpha _ { b } r _ { j } , } \\ { e _ { j , i } , } & { | ( \mathbf { T } \pmb { x } _ { j } ) _ { i } | > \alpha _ { b } r _ { j } . } \end{array} \right.\tag{15}
$$

Zero groups have $r _ { j } = \Delta _ { j } = 0$ and zero error. All group quantities depend on $\mathbf { T } .$ We suppress that dependence in their subscripts.

To measure errors in transformed coordinates, write $B _ { T } = { T } ^ { - \top } { \pmb { H } } { \pmb { T } } ^ { - 1 }$ . Then $\varepsilon _ { j } ^ { \top } H \varepsilon _ { j } =$ $( T \varepsilon _ { j } ) ^ { \top } B _ { T } ( T \varepsilon _ { j } )$ . The diagonal entry $( B _ { T } ) _ { i i }$ therefore measures sensitivity to an error in coordinate i. The modeled output loss is $\begin{array} { r } { \ell ( \pmb { T } ) = 2 ^ { - 1 } \sum _ { j } \pmb { \varepsilon } _ { j } ^ { \top } \pmb { H } \pmb { \varepsilon } _ { j } } \end{array}$ . Expectations below are over the rounding noise.

An unclipped reference. Uniform rounding noise with step $\Delta _ { j }$ has variance $1 2 ^ { - 1 } \Delta _ { j } ^ { 2 }$ Applying it to every coordinate without clipping would give the expected loss

$$
\begin{array} { l } { { \displaystyle \ell _ { \mathrm { r n d } } ( { \pmb T } ) = 2 4 ^ { - 1 } \sum _ { j } \Delta _ { j } ^ { 2 } \mathrm { t r } ( { \pmb B } _ { { \pmb T } } ) } } \\ { ~ } \\ { { \displaystyle ~ = 6 ^ { - 1 } d ^ { - 1 } \alpha _ { b } ^ { 2 } ( 2 ^ { b } - 1 ) ^ { - 2 } \mathrm { t r } ( { \pmb T } { \pmb M } { \pmb T } ^ { \top } ) \mathrm { t r } ( { \pmb B } _ { { \pmb T } } ) } . } \end{array}\tag{16}
$$

The last equality uses $\begin{array} { r } { \sum _ { j } r _ { j } ^ { 2 } = d ^ { - 1 } \mathrm { t r } ( \pmb { T M T } ^ { \top } ) } \end{array}$ . This reference loss is a deterministic quantity, distinct from the clipped loss.

The clipped loss. Independence and zero mean remove all cross terms involving rounding noise, giving

$$
\begin{array} { l } { { \displaystyle \mathbb { E } [ \ell ( { \pmb T } ) ] = 2 4 ^ { - 1 } \sum _ { j } \Delta _ { j } ^ { 2 } \sum _ { i : | ( { \pmb T } { \pmb x } _ { j } ) _ { i } | \leq \alpha _ { b } r _ { j } } ( B _ { { \pmb T } } ) _ { i i } } } \\ { { \displaystyle ~ + 2 ^ { - 1 } \sum _ { j } e _ { j } ^ { \top } B _ { { \pmb T } } e _ { j } } . } \end{array}\tag{17}
$$

The first term counts rounding only inside the clipping range. The second measures the exact clipping error. Clipped coordinates no longer receive rounding noise, so the total can be smaller than the unclipped reference.

## B.2 The unclipped optimum

Only the trace product in Eq. (16) depends on the transform. The following result shows that WUSH minimizes it over all invertible transforms.

Theorem 3 Let M, $\pmb { H } \in \mathbb { R } ^ { d \times d }$ be symmetric positive definite, and let $\sigma _ { 1 } , \ldots , \sigma _ { d }$ be the singular values of $M ^ { \frac { 1 } { 2 } } H ^ { \frac { 1 } { 2 } } . ^ { 4 }$ Every invertible T satisfies

$$
\mathrm { t r } ( \mathbf { \boldsymbol { T } } \mathbf { \boldsymbol { M } } \mathbf { \boldsymbol { T } } ^ { \top } ) \mathrm { t r } ( \mathbf { \boldsymbol { T } } ^ { - \top } \mathbf { \boldsymbol { H } } \mathbf { \boldsymbol { T } } ^ { - 1 } ) \geq \left( \sum _ { i } \sigma _ { i } \right) ^ { 2 } ,\tag{18}
$$

and $E q . \ ( 1 )$ with $\gamma = 0$ attains the bound.

Proof Take a singular value decomposition $\begin{array} { r } { M ^ { \frac { 1 } { 2 } } H ^ { \frac { 1 } { 2 } } = \sum _ { i } \sigma _ { i } { \pmb { u } } _ { i } { \pmb { v } } _ { i } ^ { \top } } \end{array}$ , where $\{ { \pmb u } _ { i } \}$ and $\{ v _ { i } \}$ are orthonormal bases. Inserting $\pmb { T } ^ { \top } \pmb { T } ^ { - \top } = \mathbf { I }$ and applying Cauchy-Schwarz gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { i } \sigma _ { i } = \sum _ { i } ( { \pmb T } { \pmb M } ^ { \frac 1 2 } { \pmb u } _ { i } ) ^ { \top } ( { \pmb T } ^ { - \top } { \pmb H } ^ { \frac 1 2 } { \pmb v } _ { i } ) } \\ & { \qquad \le \left( \sum _ { i } \| { \pmb T } { \pmb M } ^ { \frac 1 2 } { \pmb u } _ { i } \| ^ { 2 } \right) ^ { \frac 1 2 } \left( \sum _ { i } \| { \pmb T } ^ { - \top } { \pmb H } ^ { \frac 1 2 } { \pmb v } _ { i } \| ^ { 2 } \right) ^ { \frac 1 2 } . } \end{array}\tag{19}
$$

The two sums of squared norms are the traces in Eq. (18), because the singular-vector bases are orthonormal. Squaring proves the lower bound.

For attainment, set $\gamma = 0$ in $\operatorname { E q . } \left( 1 \right)$ , so that $\pmb { L } \pmb { L } ^ { \top } = \pmb { H }$ and $U \mathbf { A } U ^ { \top } = \pmb { L } ^ { \top } M \pmb { L }$ . Substituting the WUSH transform gives

$$
\pmb { T M T } ^ { \top } = c ^ { 2 } \pmb { \mathcal { H } } \pmb { \Lambda } ^ { \frac { 1 } { 2 } } \pmb { \mathcal { H } } ^ { \top } , \qquad \pmb { T } ^ { - \top } \pmb { H } \pmb { T } ^ { - 1 } = c ^ { - 2 } \pmb { \mathcal { H } } \pmb { \Lambda } ^ { \frac { 1 } { 2 } } \pmb { \mathcal { H } } ^ { \top } .\tag{20}
$$

The diagonal of $\pmb { \Lambda } ^ { \frac 1 2 }$ contains the singular values of $M ^ { \frac { 1 } { 2 } } L .$ . These are the $\sigma _ { i } ,$ since $M ^ { \frac { 1 } { 2 } } L$ and $M ^ { \frac { 1 } { 2 } } H ^ { \frac { 1 } { 2 } }$ have the same product with their respective transposes. The trace product is therefore $\begin{array} { r } { ( \operatorname { t r } ( \mathbf { A } ^ { \frac { 1 } { 2 } } ) ) ^ { 2 } = ( \sum _ { i } \sigma _ { i } ) ^ { 2 } } \end{array}$

An orthogonal transform leaves both traces unchanged and gives $\operatorname { t r } ( M ) \operatorname { t r } ( H )$ . This equals the minimum only when M and H are proportional.

For the clipped comparison, a transform is sensitivity-balanced when

$$
( \pmb { { \cal B } } _ { T } ) _ { i i } = d ^ { - 1 } \operatorname { t r } ( \pmb { { \cal B } } _ { T } ) \qquad i = 1 , \ldots , d .\tag{21}
$$

For the undamped WUSH transform this follows from Eq. (20), since every entry of H has magnitude $d ^ { - \frac { 1 } { 2 } }$ . Equal diagonal entries do not mean that $B _ { T }$ is diagonal.

## B.3 The effect of clipping

We first bound clipping error for a scalar Gaussian. Two assumptions then transfer this bound to the WUSH-transformed groups. Finally, we bound how much clipping can reduce the rounding contribution of a competing transform.

Gaussian clipping error. Let $\zeta \sim \mathcal { N } ( 0 , 1 )$ . Clipping $\zeta \ \mathrm { t o } \ [ - \alpha , \alpha ]$ incurs squared error $( | \zeta | - \alpha ) _ { + } ^ { 2 }$ , where $( \cdot ) _ { + }$ takes the positive part. The next lemma concerns clipping alone, not the full quantization error.

Lemma 4 For the Gaussian-optimal clip constant $\alpha _ { b }$ in $E q . \ ( 5 )$

$$
\mathbb { E } \left[ ( | \zeta | - \alpha _ { b } ) _ { + } ^ { 2 } \right] \leq ( 2 ^ { b } - 1 ) ^ { - 2 } .\tag{22}
$$

Moreover, $\alpha _ { b }  \infty$ as $b \to \infty$

Proof For $\alpha > 0$ , let ${ \mathcal { Q } } _ { \alpha }$ round to the nearest of the $2 ^ { b }$ equally spaced levels from $- \alpha$ to $\alpha$ . This scalar grid is fixed across samples, without a sample-dependent RMS scale. Write $\widehat { \zeta } = \mathcal { Q } _ { \alpha _ { b } } ( \zeta )$ and $\Delta = 2 \alpha _ { b } ( 2 ^ { b } - 1 ) ^ { - 1 }$ for the reconstruction and spacing at the optimal endpoint. All expectations below are over $\zeta$ . We first use optimality to bound $\mathbb { E } [ \zeta { \widehat { \zeta } } ]$ . We then relate this expectation to a weighted squared error that bounds the clipping error.

1. Bound the expectation using the optimal scale. If we change the endpoint from $\alpha _ { b }$ to $\alpha ,$ each grid level is multiplied by $\alpha \alpha _ { b } ^ { - 1 }$ . Thus $\alpha \alpha _ { b } ^ { - 1 } \widehat { \zeta }$ is a valid level on the new grid, although it need not be the nearest one to $\zeta .$ . For every $\alpha > 0$

$$
\begin{array} { r } { \mathbb { E } \left[ ( \widehat { \zeta } - \zeta ) ^ { 2 } \right] \leq \mathbb { E } \left[ ( \mathcal { Q } _ { \alpha } ( \zeta ) - \zeta ) ^ { 2 } \right] \leq \mathbb { E } \left[ ( \alpha \alpha _ { b } ^ { - 1 } \widehat { \zeta } - \zeta ) ^ { 2 } \right] . } \end{array}\tag{23}
$$

The first inequality uses the optimality of $\alpha _ { b }$ among all endpoints. The second uses the fact that choosing the nearest level cannot be worse than choosing the rescaled old level. Both inequalities are equalities at $\alpha = \alpha _ { b }$ . Consequently, the last expectation is minimized there. Keeping $\widehat { \zeta }$ fixed and expanding that expectation gives the quadratic

$$
\begin{array} { r } { \mathbb { E } \left[ ( \alpha \alpha _ { b } ^ { - 1 } \widehat { \zeta } - \zeta ) ^ { 2 } \right] = \alpha ^ { 2 } \alpha _ { b } ^ { - 2 } \mathbb { E } [ \widehat { \zeta } ^ { 2 } ] - 2 \alpha \alpha _ { b } ^ { - 1 } \mathbb { E } [ \zeta \widehat { \zeta } ] + 1 . } \end{array}\tag{24}
$$

Here the last term is $\mathbb { E } [ \zeta ^ { 2 } ] = 1$ . The derivative with respect to α vanishes at the positive minimizer $\alpha _ { b } .$ , so

$$
0 = 2 \alpha _ { b } ^ { - 1 } \left( \mathbb { E } [ \widehat { \zeta } ^ { 2 } ] - \mathbb { E } [ \zeta \widehat { \zeta } ] \right) \qquad \Longrightarrow \qquad \mathbb { E } [ \widehat { \zeta } ^ { 2 } ] = \mathbb { E } [ \zeta \widehat { \zeta } ] .\tag{25}
$$

This equality is what makes Cauchy-Schwarz useful. Both second moments are finite, since $\zeta$ is standard Gaussian and $| \widehat { \zeta } | \leq \alpha _ { b }$ . Cauchy-Schwarz does not require the two variables to be independent and gives

$$
\left| \mathbb { E } [ \zeta \widehat { \zeta } ] \right| ^ { 2 } \leq \mathbb { E } [ \zeta ^ { 2 } ] \mathbb { E } [ \widehat { \zeta } ^ { 2 } ] = \mathbb { E } [ \widehat { \zeta } ^ { 2 } ] .\tag{26}
$$

Substituting Eq. (25) into the left-hand side yields $( \mathbb { E } [ \widehat { \zeta } ^ { 2 } ] ) ^ { 2 } \leq \mathbb { E } [ \widehat { \zeta } ^ { 2 } ]$ . This second moment is nonnegative. If it is zero, it is already at most one. Otherwise, dividing by it gives $\mathbb { E } [ \widehat { \zeta } ^ { 2 } ] \leq 1$ Using Eq. (25) once more, we obtain

$$
0 \leq \mathbb { E } [ \zeta \widehat { \zeta } ] = \mathbb { E } [ \widehat { \zeta } ^ { 2 } ] \leq 1 .\tag{27}
$$

Thus, the upper bound of one follows from optimality and Cauchy-Schwarz together, not from Cauchy-Schwarz alone.

2. Relate this expectation to a squared error. The moment bound in Eq. (27) does not yet bound clipping. For that, we use the Gaussian density $\phi ,$ whose derivative satisfies $\phi ^ { \prime } ( x ) = - x \phi ( x )$ . Consider one quantization bin with reconstruction level a. On this bin, a is constant, and the quantization error is $a - x$ . Define

$$
F _ { a } ( x ) = a \left( 8 ^ { - 1 } \Delta ^ { 2 } - 2 ^ { - 1 } ( x - a ) ^ { 2 } \right) , \qquad F _ { a } ^ { \prime } ( x ) = a ( a - x ) .\tag{28}
$$

The constant $8 ^ { - 1 } \Delta ^ { 2 }$ is chosen to remove the boundary terms in integration by parts. Indeed, every finite endpoint of the bin is a midpoint between adjacent levels, where $| x - a | = 2 ^ { - 1 } \Delta$ and hence $F _ { a } ( x ) = 0$ . For the two outer bins, $F _ { a } ( x ) \phi ( x )$ also tends to zero at the infinite endpoint because the Gaussian density decays faster than this quadratic grows. It follows that

$$
\begin{array} { l } { \displaystyle \int _ { \mathrm { b i n } } a ( a - x ) \phi ( x ) \mathrm { d } x = [ F _ { a } ( x ) \phi ( x ) ] _ { \mathrm { e n d p o i n t s } } - \int _ { \mathrm { b i n } } F _ { a } ( x ) \phi ^ { \prime } ( x ) \mathrm { d } x } \\ { = \displaystyle \int _ { \mathrm { b i n } } x F _ { a } ( x ) \phi ( x ) \mathrm { d } x } \\ { = 8 ^ { - 1 } \Delta ^ { 2 } \displaystyle \int _ { \mathrm { b i n } } x a \phi ( x ) \mathrm { d } x - 2 ^ { - 1 } \int _ { \mathrm { b i n } } x a ( x - a ) ^ { 2 } \phi ( x ) \mathrm { d } x . } \end{array}\tag{29}
$$

Now sum this equality over all bins. On the bin with level $^ { a , }$ the reconstruction $\widehat { \zeta }$ equals a. The three integrals therefore become

$$
\begin{array} { r } { \mathbb { E } [ \widehat { \zeta } ( \widehat { \zeta } - \zeta ) ] = 8 ^ { - 1 } \Delta ^ { 2 } \mathbb { E } [ \zeta \widehat { \zeta } ] - 2 ^ { - 1 } \mathbb { E } \left[ \zeta \widehat { \zeta } ( \widehat { \zeta } - \zeta ) ^ { 2 } \right] . } \end{array}\tag{30}
$$

The left-hand side is zero by Eq. (25). Moving the last term to the other side and multiplying by two gives the equality below. The inequality then follows by substituting the upper bound $\mathbb { E } [ \zeta { \widehat { \zeta } } ] \leq 1$ from Eq. (27).

$$
\begin{array} { r } { \mathbb { E } \left[ \zeta \widehat { \zeta } ( \widehat { \zeta } - \zeta ) ^ { 2 } \right] = 4 ^ { - 1 } \Delta ^ { 2 } \mathbb { E } [ \zeta \widehat { \zeta } ] \leq 4 ^ { - 1 } \Delta ^ { 2 } . } \end{array}\tag{31}
$$

This is a bound on the total squared quantization error weighted by $\zeta \widehat { \zeta }$ . The weight lets us extract the clipping error in the next step.

3. Keep the contribution from clipped samples. The symmetric grid reconstructs each nonzero input with the same sign, so $\zeta { \widehat { \zeta } } \geq 0$ everywhere. When $| \zeta | > \alpha _ { b }$ , the reconstruction is the endpoint with that sign. Thus $\zeta \widehat { \zeta } = \alpha _ { b } | \zeta | \geq \alpha _ { b } ^ { 2 }$ and $( \widehat { \zeta } - \dot { \zeta } ) ^ { 2 } = ( | \zeta | - \alpha _ { b } ) ^ { 2 }$ . When $| \zeta | \le \alpha _ { b }$ ， the clipping error is zero and the weighted squared error is nonnegative. Combining these two cases gives the pointwise inequality

$$
\alpha _ { b } ^ { 2 } ( | \zeta | - \alpha _ { b } ) _ { + } ^ { 2 } \leq \zeta \widehat { \zeta } ( \widehat { \zeta } - \zeta ) ^ { 2 } .\tag{32}
$$

Taking expectations, applying Eq. (31), and then multiplying by $\alpha _ { b } ^ { - 2 }$ yields

$$
\begin{array} { r } { \mathbb { E } \left[ ( | \zeta | - \alpha _ { b } ) _ { + } ^ { 2 } \right] \leq \alpha _ { b } ^ { - 2 } \mathbb { E } \left[ \zeta \widehat { \zeta } ( \widehat { \zeta } - \zeta ) ^ { 2 } \right] \leq 4 ^ { - 1 } \alpha _ { b } ^ { - 2 } \Delta ^ { 2 } = ( 2 ^ { b } - 1 ) ^ { - 2 } . } \end{array}\tag{33}
$$

The last equality uses $\Delta = 2 \alpha _ { b } ( 2 ^ { b } - 1 ) ^ { - 1 }$ and proves Eq. (22).

$\it 4 .$ Show that the optimal endpoint grows with bitwidth. Consider the candidate endpoint $\alpha = \sqrt { b }$ . Inside its clipping range, the error is at most half the grid spacing, or $\sqrt { b } ( 2 ^ { b } - 1 ) ^ { - 1 }$ Outside the range, the error is exactly the distance to the nearest endpoint. Optimality of $\alpha _ { b }$ therefore gives

$$
\begin{array} { r } { \mathbb { E } \left[ ( \widehat { \zeta } - \zeta ) ^ { 2 } \right] \leq \mathbb { E } \left[ ( \mathcal { Q } _ { \sqrt { b } } ( \zeta ) - \zeta ) ^ { 2 } \right] \leq b ( 2 ^ { b } - 1 ) ^ { - 2 } + \mathbb { E } \left[ ( | \zeta | - \sqrt { b } ) _ { + } ^ { 2 } \right] \longrightarrow 0 . } \end{array}\tag{34}
$$

The first term tends to zero because the number of levels grows exponentially in b. The second tends to zero by dominated convergence since its integrand tends pointwise to zero and is bounded by the integrable random variable $\zeta ^ { 2 }$ . If $\alpha _ { b }$ stayed bounded along a subsequence, the Gaussian tail beyond a fixed finite endpoint would give a positive lower bound on the clipping error along that subsequence. The total quantization error is at least its clipping contribution, contradicting Eq. (34). Hence $\alpha _ { b }  \infty$ 7

Conditions on the transformed groups. For the next two conditions, evaluate $r _ { j }$ and $e _ { j }$ at $\pmb { T } _ { \star }$ and write $B _ { \star } = B _ { T _ { \star } }$ . Dividing each nonzero transformed group by its own RMS gives

$$
\widetilde { \pmb { x } } _ { j } = r _ { j } ^ { - 1 } \pmb { T } _ { \star } \pmb { x } _ { j } , \qquad \| \widetilde { \pmb { x } } _ { j } \| ^ { 2 } = d .\tag{35}
$$

Set $\widetilde { \pmb { x } } _ { j } = \pmb { 0 }$ for a zero group. Let $C _ { \mathrm { t a i l } } , C _ { \mathrm { a l i g n } } \geq 1$ be fixed bounds for the following conditions. The first condition bounds the fraction of normalized entries above each threshold by a multiple of the Gaussian tail,

$$
\left( d \sum _ { j } r _ { j } ^ { 2 } \right) ^ { - 1 } \sum _ { j } r _ { j } ^ { 2 } \sum _ { i = 1 } ^ { d } \mathbb { 1 } \{ | \widetilde { x } _ { j , i } | > \alpha \} \leq C _ { \mathrm { t a i l } } \mathbb { P } ( | \zeta | > \alpha ) \qquad \forall \alpha \geq \alpha _ { b } .\tag{36}
$$

The weights $r _ { j } ^ { 2 }$ account for each group’s contribution to squared error. The second condition bounds how strongly clipping errors align with sensitive directions,

$$
\sum _ { j } e _ { j } ^ { \top } B _ { \star } e _ { j } \leq C _ { \mathrm { a l i g n } } d ^ { - 1 } \operatorname { t r } ( B _ { \star } ) \sum _ { j } \| e _ { j } \| ^ { 2 } .\tag{37}
$$

This controls the direction of the clipping error, not its magnitude. It does not require independent errors and does not follow from equal diagonal entries alone. Both conditions concern WUSH only. A competing transform needs only to be sensitivity-balanced. The bounds are independent of $d ,$ and must also remain fixed across bitwidths for the asymptotic statement in Theorem 2.

Upper bound at WUSH. Integrating Eq. (36) against $2 ( \alpha - \alpha _ { b } )$ dα over $\alpha \geq \alpha _ { b }$ converts the tail bound into a bound on squared clipping error. Then Eq. (22) gives

$$
\begin{array} { l } { { \displaystyle \sum _ { j } \| e _ { j } \| ^ { 2 } = \sum _ { j } r _ { j } ^ { 2 } \sum _ { i } ( | \widetilde { x } _ { j , i } | - \alpha _ { b } ) _ { + } ^ { 2 } } } \\ { { \displaystyle \qquad \leq C _ { \mathrm { t a i l } } d \sum _ { j } r _ { j } ^ { 2 } \mathbb { E } \left[ ( | \zeta | - \alpha _ { b } ) _ { + } ^ { 2 } \right] } } \\ { { \displaystyle \qquad \leq C _ { \mathrm { t a i l } } d ( 2 ^ { b } - 1 ) ^ { - 2 } \sum _ { j } r _ { j } ^ { 2 } } . } \end{array}\tag{38}
$$

By Eq. (37), the clipping contribution to the loss is therefore at most

$$
\begin{array} { c } { { 2 ^ { - 1 } \displaystyle \sum _ { j } e _ { j } ^ { \top } B _ { \star } e _ { j } \leq 2 ^ { - 1 } C _ { \mathrm { t a i l } } C _ { \mathrm { a l i g n } } ( 2 ^ { b } - 1 ) ^ { - 2 } \mathrm { t r } ( B _ { \star } ) \displaystyle \sum _ { j } r _ { j } ^ { 2 } } } \\ { { { } } } \\ { { = 3 C _ { \mathrm { t a i l } } C _ { \mathrm { a l i g n } } \alpha _ { b } ^ { - 2 } \ell _ { \mathrm { r n d } } ( T _ { \star } ) . } } \end{array}\tag{39}
$$

The rounding contribution cannot exceed the unclipped reference, so Eq. (17) yields

$$
\begin{array} { r } { \mathbb { E } [ \ell ( \pmb { T } _ { \star } ) ] \leq \left( 1 + 3 C _ { \mathrm { t a i l } } C _ { \mathrm { a l i g n } } \alpha _ { b } ^ { - 2 } \right) \ell _ { \mathrm { r n d } } ( \pmb { T } _ { \star } ) . } \end{array}\tag{40}
$$

Lower bound for a competing transform. Now take any sensitivity-balanced invertible T . For each nonzero group, $\| \pmb { T x } _ { j } \| ^ { 2 } = d r _ { j } ^ { 2 }$ , so at most $d \alpha _ { b } ^ { - 2 }$ coordinates can exceed $\alpha _ { b } r _ { j }$ in magnitude. At least a fraction $1 - \alpha _ { b } ^ { - 2 }$ therefore retain their rounding noise. All coordinates have the same sensitivity by Eq. (21), so they retain at least that fraction of the reference loss. The clipping contribution is nonnegative, giving

$$
\begin{array} { r } { \mathbb { E } [ \ell ( \pmb { T } ) ] \geq \left( 1 - \alpha _ { b } ^ { - 2 } \right) \ell _ { \mathrm { r n d } } ( \pmb { T } ) . } \end{array}\tag{41}
$$

Zero groups contribute no loss to either side. Finally, Theorem 3 gives $\ell _ { \mathrm { r n d } } ( T _ { \star } ) \leq \ell _ { \mathrm { r n d } } ( T )$ Combining the upper and lower bounds proves, for $\alpha _ { b } > 1$ ，

$$
\begin{array} { r } { \mathbb { E } [ \ell ( \mathbf { T } _ { \star } ) ] \leq \left( 1 + ( 1 + 3 C _ { \mathrm { t a i l } } C _ { \mathrm { a l i g n } } ) ( \alpha _ { b } ^ { 2 } - 1 ) ^ { - 1 } \right) \mathbb { E } [ \ell ( \mathbf { T } ) ] . } \end{array}\tag{42}
$$

For fixed assumption bounds, this factor is independent of the group width. Since $\alpha _ { b }  \infty ,$ it is $1 + O ( \alpha _ { b } ^ { - 2 } )$ , proving Theorem 2. The comparison remains restricted to sensitivity-balanced transforms because balancing an arbitrary transform can change its clipped loss.

## Appendix C. Exact Hessians for Grouped-Query Attention

This appendix derives attention-aware Hessians for perturbations of the queries, keys, and values in grouped-query attention. Unlike the per-head Hessians in Eq. (4), these exact expressions depend on the position of the perturbed vector. The final subsection aggregates the key and value Hessians to construct WUSH-A. The query case is included for completeness.

## C.1 Setup

Fix a key/value head h, and let $g = 1 , \ldots , n _ { \mathrm { q } } / n _ { \mathrm { k v } }$ index the query heads that read it. We allow the query/key and value head dimensions to differ in this appendix, writing them as $d _ { \mathrm { k } }$ and $d _ { \mathrm { v } }$ (the main text takes both to be d). Write

$$
\begin{array} { c } { q _ { h , g } = \left[ { q _ { h , g , 1 } } , \dots , { q _ { h , g , S } } \right] \in \mathbb { R } ^ { d _ { \mathrm { k } } \times S } , } \\ { K _ { h } = \left[ { k _ { h , 1 } } , \dots , { k _ { h , S } } \right] \in \mathbb { R } ^ { d _ { \mathrm { k } } \times S } , \qquad V _ { h } = \left[ { v _ { h , 1 } } , \dots , { v _ { h , S } } \right] \in \mathbb { R } ^ { d _ { \mathrm { v } } \times S } . } \end{array}\tag{43}
$$

The query and key vectors are taken after Norm and RoPE, as in Eq. (2). Let ${ \mathbf { } } p _ { h , g , t }$ be column t of $P _ { h , g }$ , with entry $p _ { h , g , t , s }$ at cache position s, and let $o _ { h , g , t } = V _ { h } p _ { h , g , t }$ be the corresponding

head output. The causal mask gives $p _ { h , g , t , s } = 0$ whenever $s > t$ . The output-projection blocks $W _ { \mathrm { O } ( h , g ) } \in \mathbb { R } ^ { d _ { \mathrm { v } } \times a }$ <sup>d</sup>model induce the head-coupling matrices

$$
\begin{array} { r } { \boldsymbol { \Gamma } _ { h , g , g ^ { \prime } } = \boldsymbol { W } _ { \mathrm { O } ( h , g ) } \boldsymbol { W } _ { \mathrm { O } ( h , g ^ { \prime } ) } ^ { \top } \in \mathbb { R } ^ { d _ { \mathrm { v } } \times d _ { \mathrm { v } } } . } \end{array}\tag{44}
$$

The diagonal block $\Gamma _ { h , g , g }$ measures the sensitivity of query head $( h , g )$ after output projection, while $\Gamma _ { h , g , g ^ { \prime } }$ for $g \neq g ^ { \prime }$ couples two query heads that share h. Moreover, $\Gamma _ { h , g , g ^ { \prime } } ^ { \top } = \Gamma _ { h , g ^ { \prime } , g } .$

## C.2 Hessian from Attention-Output Derivatives

To measure the local sensitivity of the attention output to quantization error, perturb one query, key, or value vector by δ. Write $\pmb { Y } = [ \pmb { y } _ { 1 } , \dots , \pmb { y } _ { S } ]$ and ${ \pmb Y } ^ { \prime } \left( \pmb \delta \right) = \left[ { \pmb y } _ { 1 } ^ { \prime } \left( \pmb \delta \right) , \ldots , { \pmb y } _ { S } ^ { \prime } \left( \pmb \delta \right) \right]$ for the unperturbed and perturbed attention outputs, respectively, and define

$$
\ell \left( \pmb { \delta } \right) = \left\| \pmb { Y } ^ { \prime } \left( \pmb { \delta } \right) - \pmb { Y } \right\| _ { \mathrm { F } } ^ { 2 } .\tag{45}
$$

Let $y _ { t , j } ^ { \prime } \left( \delta \right)$ and $y _ { t , j }$ denote coordinate $j$ of ${ \pmb y } _ { t } ^ { \prime } \left( \delta \right)$ and $\mathbf { \mathscr { y } } _ { t }$ , respectively. Differentiating Eq. (45) once gives

$$
\frac { \partial \ell } { \partial \pmb { \delta } } = 2 \sum _ { t = 1 } ^ { S } \left( \frac { \partial \pmb { y } _ { t } ^ { \prime } } { \partial \pmb { \delta } ^ { \top } } \right) ^ { \top } \left( \pmb { y } _ { t } ^ { \prime } - \pmb { y } _ { t } \right) .\tag{46}
$$

Differentiating again gives the full Hessian

$$
\frac { \partial ^ { 2 } \ell } { \partial \delta \partial \delta ^ { \top } } = 2 \sum _ { t = 1 } ^ { S } \left( \left( \frac { \partial y _ { t } ^ { \prime } } { \partial \delta ^ { \top } } \right) ^ { \top } \left( \frac { \partial y _ { t } ^ { \prime } } { \partial \delta ^ { \top } } \right) + \sum _ { j = 1 } ^ { d _ { \mathrm { m o d e l } } } \left( y _ { t , j } ^ { \prime } - y _ { t , j } \right) \frac { \partial ^ { 2 } y _ { t , j } ^ { \prime } } { \partial \delta \partial \delta ^ { \top } } \right) .\tag{47}
$$

At zero perturbation,

$$
\begin{array} { r } { Y ^ { \prime } \left( \mathbf { 0 } \right) - Y = \mathbf { 0 } . } \end{array}\tag{48}
$$

This single fact forces the gradient in Eq. (46) to vanish and, independently, removes the second term in Eq. (47). Therefore,

$$
\begin{array} { c } { { \displaystyle \left. \frac { \partial \ell } { \partial \delta } \right| _ { \delta = { \bf 0 } } = { \bf 0 } , } } \\ { { \displaystyle \left. \frac { \partial ^ { 2 } \ell } { \partial \delta \partial \delta ^ { \top } } \right| _ { \delta = { \bf 0 } } = 2 \sum _ { t = 1 } ^ { S } \left( \frac { \partial y _ { t } ^ { \prime } } { \partial \delta ^ { \top } } \right) ^ { \top } \left( \frac { \partial y _ { t } ^ { \prime } } { \partial \delta ^ { \top } } \right) \Bigg | _ { \delta = { \bf 0 } } . } } \end{array}\tag{49}
$$

## C.3 Query Hessian

Perturb the query ${ q } _ { h , g , t }$ by $\pmb { \delta } \in \mathbb { R } ^ { d _ { \mathrm { k } } }$ 2

$$
\pmb q _ { h , g , t } \mapsto \pmb q _ { h , g , t } + \delta .\tag{50}
$$

Only the query head $( h , g )$ at position t changes. The derivative of the logit vector with respect to $\delta ^ { \top }$ is $\tau ^ { - 1 } K _ { h } ^ { \top }$ , while the softmax derivative is diag $( { p } _ { h , g , t } ) - { p } _ { h , g , t } { p } _ { h , g , t } ^ { \top }$ . Chaining

them gives the head-output and attention-output derivatives

$$
\begin{array} { c } { \displaystyle \frac { \partial \boldsymbol { o } _ { h , g , t } ^ { \prime } } { \partial \delta ^ { \top } } | _ { \delta = \mathbf { 0 } } = \boldsymbol { \tau } ^ { - 1 } V _ { h } ( \mathrm { d i a g } ( \boldsymbol { p } _ { h , g , t } ) - \boldsymbol { p } _ { h , g , t } \boldsymbol { p } _ { h , g , t } ^ { \top } ) \boldsymbol { K } _ { h } ^ { \top } , } \\ { \displaystyle  \frac { \partial \boldsymbol { y } _ { t } ^ { \prime } } { \partial \delta ^ { \top } } | _ { \delta = \mathbf { 0 } } = \boldsymbol { \tau } ^ { - 1 } W _ { \mathrm { O } ( h , g ) } ^ { \top } V _ { h } ( \mathrm { d i a g } ( \boldsymbol { p } _ { h , g , t } ) - \boldsymbol { p } _ { h , g , t } \boldsymbol { p } _ { h , g , t } ^ { \top } ) \boldsymbol { K } _ { h } ^ { \top } . } \end{array}\tag{51}
$$

Substituting the attention-output derivative into Eq. (49) gives the $d _ { \mathrm { k } } \times d _ { \mathrm { k } }$ query Hessian

$$
\begin{array} { l } { { \displaystyle { \cal H } _ { \mathrm { Q } ( h , g , t ) } = 2 \left. \left( \frac { \partial y _ { t } ^ { \prime } } { \partial \delta ^ { \top } } \right) ^ { \top } \left( \frac { \partial y _ { t } ^ { \prime } } { \partial \delta ^ { \top } } \right) \right. _ { \delta = 0 } } } \\ { { \displaystyle = 2 \tau ^ { - 2 } { \cal K } _ { h } \left( \mathrm { d i a g } \left( p _ { h , g , t } \right) - p _ { h , g , t } p _ { h , g , t } ^ { \top } \right) V _ { h } ^ { \top } \mathbf { T } _ { h , g , g } V _ { h } \left( \mathrm { d i a g } \left( p _ { h , g , t } \right) - p _ { h , g , t } p _ { h , g , t } ^ { \top } \right) K _ { h } ^ { \top } } . }  \end{array}\tag{52}
$$

## C.4 Key Hessian

Perturb the shared key $k _ { h , s }$ by $\pmb { \delta } \in \mathbb { R } ^ { d _ { \mathrm { k } } }$

$$
\pmb { k } _ { h , s } \mapsto \pmb { k } _ { h , s } + \delta .\tag{53}
$$

This changes every query head that reads h and every causal query position $t \geq s$ . Differentiating the softmax with respect to the logit at cache position s and summing against $V _ { h }$ gives the residual ${ \pmb v } _ { h , s } - { \pmb O } _ { h , g , t }$ . For one such head and position, the head-output derivative is

$$
\left. \frac { \partial \pmb { o } _ { h , g , t } ^ { \prime } } { \partial \pmb { \delta } ^ { \top } } \right| _ { \pmb { \delta } = \mathbf { 0 } } = \tau ^ { - 1 } p _ { h , g , t , s } \left( \pmb { v } _ { h , s } - \pmb { o } _ { h , g , t } \right) \pmb { q } _ { h , g , t } ^ { \top } .\tag{54}
$$

The corresponding attention-output derivative is

$$
\left. \frac { \partial \pmb { y } _ { t } ^ { \prime } } { \partial \pmb { \delta } ^ { \top } } \right| _ { \pmb { \delta = 0 } } = \tau ^ { - 1 } \sum _ { g } p _ { h , g , t , s } \pmb { W } _ { \mathrm { O } ( h , g ) } ^ { \top } \left( \pmb { v } _ { h , s } - \pmb { o } _ { h , g , t } \right) \pmb { q } _ { h , g , t } ^ { \top } .\tag{55}
$$

Substituting the attention-output derivative into $\operatorname { E q } .$ (49) and expanding the head interactions gives the $d _ { \mathrm { k } } \times d _ { \mathrm { k } }$ key Hessian

$$
\begin{array} { l } { { \displaystyle { \cal H } _ { \mathrm { K } ( h , s ) } = 2 \sum _ { t = 1 } ^ { S } \left. \left( \frac { \partial y _ { t } ^ { \prime } } { \partial \delta ^ { \top } } \right) ^ { \top } \left( \frac { \partial y _ { t } ^ { \prime } } { \partial \delta ^ { \top } } \right) \right. _ { \delta = 0 } } } \\ { { \displaystyle = 2 \tau ^ { - 2 } \sum _ { t = 1 } ^ { S } \sum _ { g , g ^ { \prime } } p _ { h , g , t , s } p _ { h , g ^ { \prime } , t , s } \left( v _ { h , s } - \sigma _ { h , g , t } \right) ^ { \top } \Gamma _ { h , g , g ^ { \prime } } \left( v _ { h , s } - \sigma _ { h , g ^ { \prime } , t } \right) q _ { h , g , t } q _ { h , g ^ { \prime } , t } ^ { \top } } . } \end{array}\tag{56}
$$

The residual bilinear form is scalar, so each summand is a scaled query outer product.

## C.5 Value Hessian

Perturb the shared value $_ { v _ { h , s } }$ <sub>s</sub> by $\pmb { \delta } \in \mathbb { R } ^ { d _ { \mathrm { v } } }$

$$
\begin{array} { r } { \pmb { v } _ { h , s } \mapsto \pmb { v } _ { h , s } + \pmb { \delta } . } \end{array}\tag{57}
$$

Values do not enter the softmax, so the attention probabilities remain fixed. For query head $( h , g )$ at position $t \geq s$ , the head-output derivative is

$$
\left. \frac { \partial \pmb { o } _ { h , g , t } ^ { \prime } } { \partial \delta ^ { \top } } \right| _ { \delta = \bf { 0 } } = p _ { h , g , t , s } \mathbf { I } _ { d _ { v } } .\tag{58}
$$

The attention-output derivative is, therefore,

$$
\left. \frac { \partial \pmb { y } _ { t } ^ { \prime } } { \partial \pmb { \delta } ^ { \top } } \right| _ { \pmb { \delta } = \mathbf { 0 } } = \sum _ { g } p _ { h , g , t , s } \pmb { W } _ { \mathrm { O } ( h , g ) } ^ { \top } .\tag{59}
$$

Substituting into Eq. (49) gives the $d _ { \mathrm { v } } \times d _ { \mathrm { v } }$ value Hessian

$$
\begin{array} { r l } & { H _ { \mathrm { V } ( h , s ) } = 2 \displaystyle \sum _ { t = 1 } ^ { S } \left( \frac { \partial y _ { t } ^ { \prime } } { \partial \delta ^ { \top } } \right) ^ { \top } \left. \left( \frac { \partial y _ { t } ^ { \prime } } { \partial \delta ^ { \top } } \right) \right. _ { \delta = \bf { 0 } } } \\ & { = 2 \displaystyle \sum _ { t = 1 } ^ { S } \sum _ { g , g ^ { \prime } } p _ { h , g , t , s } p _ { h , g ^ { \prime } , t , s } { \bf \Gamma } _ { h , g , g ^ { \prime } } . } \end{array}\tag{60}
$$

## C.6 Aggregation for WUSH-A

For the calibration sequence i, let ${ \pmb { H } } _ { \mathrm { K } ( h , s ) } ^ { ( i ) }$ and $\pmb { H } _ { \mathrm { V } ( h , s ) } ^ { ( i ) }$ denote the per-position Hessians above. Using the Gram matrices from Algorithm 1, the resulting transforms are

$$
\begin{array} { r l } & { \pmb { T } _ { \mathrm { K } ( h ) } = \mathrm { W U S H } \left( \displaystyle \sum _ { i = 1 } ^ { n _ { \mathrm { c a l } } } M _ { \mathrm { K } ( h ) } ^ { ( i ) } , \displaystyle \sum _ { i = 1 } ^ { n _ { \mathrm { c a l } } } \sum _ { s = 1 } ^ { S } \pmb { H } _ { \mathrm { K } ( h , s ) } ^ { ( i ) } \right) , } \\ & { \pmb { T } _ { \mathrm { V } ( h ) } = \mathrm { W U S H } \left( \displaystyle \sum _ { i = 1 } ^ { n _ { \mathrm { c a l } } } M _ { \mathrm { V } ( h ) } ^ { ( i ) } , \displaystyle \sum _ { i = 1 } ^ { n _ { \mathrm { c a l } } } \sum _ { s = 1 } ^ { S } \pmb { H } _ { \mathrm { V } ( h , s ) } ^ { ( i ) } \right) . } \end{array}\tag{61}
$$

Thus, WUSH-A differs from WUSH only in the Hessian supplied to the transform construction.

## Appendix D. Additional Experimental Results

## D.1 Additional Attention-Stage Errors

The 2-bit diagnostic plots from Section 5.1 appear in Figure 2, and the 3-bit and 4-bit module-output errors appear in Figure 3. For the diagnostics, query-key dot products are measured before scaling and masking, while softmax uses the model’s native scaling and causal mask. The single-sided module-output plots quantize only keys or only values.

![](images/81c138c6d84b905f6d733bb52310e1ded5a4e80f8c98f594764687d79a15beec.jpg)

2-Bit Softmax Score (Only Quantizing Keys)  
![](images/44cae898313798dc2734f6ea81e48c546a7f9b961b87c1d0e1c458e11321336e.jpg)

2-Bit Module Output (Only Quantizing Keys)  
![](images/d2d089780db7069f34b0c3b96862a2dd57191550b96535401cbb78905a744af7.jpg)

2-Bit Module Output (Only Quantizing Values)  
![](images/d0ab9cc63dce640a1788ee6671d209293634a30e1f0441985c6a859518ab4d99.jpg)

Figure 2: Relative (squared) L<sub>2</sub> errors at 2-bit for I, R, H, OSCAR, WUSH, and WUSH-A. The top row shows query-key dot-product and softmax errors with only keys quantized. The bottom row shows module-output errors with only keys or only values quantized.  
![](images/d9ab98a5bd4a5b532fd0bd3a7c7e6241b4e7c3e0bffb73a4d15430c846463485.jpg)

4-Bit Module Output (Quantizing Keys and Values)  
![](images/93e5c93066324812afb6aeaea10f5b125408f39daa4af187cce3bda56212fa74.jpg)  
Figure 3: Relative (squared) L<sub>2</sub> module-output errors at 3-bit and 4-bit for I, R, H, OSCAR, WUSH, and WUSH-A.

## D.2 Quantizer and Calibration Choices

Table 3 compares the main Qwen3-8B configurations with two alternatives under the same 2-bit benchmark protocol. The main rows repeat the scores in Table 2, and the alternative rows come from separate control evaluations.

Quantizer choice. With the WUSH-KV transform and FineWeb-Edu calibration setup held fixed, the OSCAR-style quantizer yields higher end-to-end scores than QuEST on all four tasks. In a separate single-run AIME 2025 diagnostic, OSCAR calibrated on GPQA

Table 3: Qwen3-8B downstream scores $( \% )$ over three seeds for the main configurations and quantizer or calibration alternatives. The gray second line gives the mean generated-token count in thousands (k) and the budget-exhaustion rate. Main configurations are marked in bold.
<table><tr><td>Calibration, Quantizer</td><td colspan="4">AIME 2025 MATH-500 GPQA Diamond LiveCodeBench v6</td></tr><tr><td>WUSH-KV</td><td></td><td></td><td></td><td></td></tr><tr><td>FineWeb-Edu, OSCAR</td><td> ${ \bf 4 6 . 7 \pm 3 . 3 }$   $5 3 . 4 \mathrm { k } 1 2 \%$ </td><td> ${ \bf 9 2 . 9 \pm 0 . 1 }$   $1 2 . 2 \mathrm { k } \ 2 \%$ </td><td> ${ \bf 5 4 . 5 \pm 2 . 3 }$   $1 6 . 3 \mathrm { k } 1 \%$ </td><td> ${ \bf 3 5 . 2 \pm 0 . 7 }$   $5 3 . 9 \mathrm { k } \ 2 6 \%$ </td></tr><tr><td>FineWeb-Edu, QuEST</td><td> $2 8 . 9 \pm 6 . 9$  63.4k 42%</td><td> $8 8 . 7 \pm 1 . 0$ </td><td> $5 0 . 5 \pm 2 . 5$ </td><td> $3 3 . 1 \pm 0 . 6$ </td></tr><tr><td>OSCAR</td><td></td><td> $1 1 . 7 \mathrm { k } \ \mathrm { ~ 4 \% ~ }$ </td><td> $2 2 . 1 \mathrm { k } \ 4 \%$ </td><td> $6 3 . 7 \mathrm { k } 4 6 \%$ </td></tr><tr><td></td><td> ${ \bf 3 4 . 4 \pm 8 . 4 }$ </td><td> ${ \bf 8 9 . 3 \pm 1 . 1 }$ </td><td> ${ \bf 5 2 . 9 \pm 3 . 0 }$ </td><td> ${ \bf 2 0 . 4 \pm 0 . 3 }$ </td></tr><tr><td rowspan="2">GPQA Diamond, OSCAR</td><td> $6 3 . 0 \mathrm { k } 4 2 \%$ </td><td> $1 6 . 0 \mathrm { k } ~ 6 \%$ </td><td> $1 8 . 3 \mathrm { k } 1 \%$ </td><td> $7 3 . 8 \mathrm { k } 7 1 \%$ </td></tr><tr><td> $2 2 . 2 \pm 1 . 9$ </td><td> $7 6 . 9 \pm 1 . 3$ </td><td> $4 8 . 7 \pm 0 . 3$ </td><td></td></tr><tr><td rowspan="2">FineWeb-Edu, OSCAR</td><td></td><td></td><td></td><td> $1 3 . 1 \pm 1 . 2$ </td></tr><tr><td> $7 0 . 8 \mathrm { k } ~ 6 0 \%$ </td><td> $2 1 . 2 \mathrm { k } 1 8 \%$ </td><td> $1 9 . 2 \mathrm { k } \ \ 4 \%$ </td><td> $7 9 . 1 \mathrm { k } 8 0 \%$ </td></tr></table>

Diamond with QuEST quantizer scored $0 . 0 \%$ . In our local layerwise and attention-modulewise measurements, QuEST instead has lower $\mathrm { L _ { 2 } }$ quantization error for both transforms. Thus local $\mathrm { L _ { 2 } }$ reconstruction error need not predict downstream autoregressive quality, motivating the OSCAR-style quantizer in the main evaluation.

Calibration choice. WUSH-KV uses 128 FineWeb-Edu sequences of length 32,768, or about 4.2 million calibration tokens. By comparison, OSCAR’s Qwen3-8B calibration targets about 8k GPQA Diamond tokens. Two preliminary AIME 2025 runs with GPQA-calibrated WUSH-KV and OSCAR-style quantizer scored 16.7% and 10.0%, suggesting that this setup was suboptimal for WUSH-KV in our tests. We also recalibrated OSCAR on a FineWeb-Edu set matched in size to WUSH-KV’s calibration set to test whether more calibration data would improve its results. This alternative scores lower than the released OSCAR transforms calibrated on the GPQA Diamond on all four tasks, so we retain the released transforms in the main comparison.

## D.3 Long-Context Evaluations

We evaluate Qwen3-8B and Qwen3-32B on long-context tasks with 2-bit KV caches and BF16 baselines using the SGLang setup of Section 5.3. For Qwen3-8B, Table 4 reports RULER NIAH results from 4k to 128k, and Tables 5 and 6 reports OpenAI MRCR results across 0k-128k. We report the normalized area under the MRCR score vs context-length curve (AUC), following Context Arena. The separate Qwen3-32B MRCR results appear in Table 7. Scores in this subsection are averaged over multiple seeds, and ± denotes standard deviation when shown.

In the RULER NIAH results, WUSH-KV remains close to BF16 at 4k and 8k but degrades as the context length increases. WUSH-KV scores above OSCAR at every tested length, with the largest gap at 128k.

MRCR is difficult for all quantized methods, with substantial losses relative to BF16. On Qwen3-8B, WUSH-KV generally degrades more gradually across context lengths and retains higher scores at long contexts than the OSCAR baseline. On Qwen3-32B, OSCAR has slightly higher aggregate scores, while WUSH-KV scores higher at 64k-128k for all three needle counts.

Table 4: Qwen3-8B RULER NIAH accuracy (%) over eight subtasks by context length.
<table><tr><td>Method</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>128k</td><td>Overall</td></tr><tr><td>BF16</td><td>99.71</td><td>97.79</td><td>92.85</td><td>94.29</td><td>85.09</td><td>79.17</td><td>91.48</td></tr><tr><td>WUSH-KV</td><td>97.44 91.92</td><td></td><td>80.75</td><td>76.87</td><td>68.36</td><td>57.26</td><td>78.77</td></tr><tr><td>OSCAR</td><td></td><td>96.63 89.79 80.00</td><td></td><td>74.00</td><td>64.36</td><td>25.69</td><td>71.74</td></tr></table>

Table 5: Qwen3-8B MRCR summary accuracy (%).
<table><tr><td>Method</td><td>2-Needle</td><td>4-Needle</td><td>8-Needle</td><td>Overall AUC</td><td></td></tr><tr><td>BF16</td><td> $3 6 . 8 9 \pm 1 . 9 6$ </td><td> $2 2 . 6 6 \pm 1 . 8 2$ </td><td> $1 6 . 3 8 \pm 0 . 4 4$ </td><td>25.31</td><td>21.95</td></tr><tr><td>WUSH-KV</td><td> ${ \bf 1 0 . 5 5 \pm 0 . 2 4 }$ </td><td> ${ \bf 9 . 6 6 \pm 0 . 3 7 }$ </td><td> ${ \bf 8 . 4 5 \pm 0 . 2 4 }$ </td><td>9.55</td><td>8.35</td></tr><tr><td>OSCAR</td><td> $7 . 2 7 \pm 0 . 6 1$ </td><td> $6 . 9 8 \pm 0 . 3 0$ </td><td> $6 . 1 1 \pm 0 . 0 5$ </td><td>6.79</td><td>4.72</td></tr></table>

Table 6: Qwen3-8B MRCR accuracy (%) by needle count and context-length bucket.
<table><tr><td>Method</td><td>0k-8k</td><td>8k-16k</td><td>16k-32k</td><td>32k-64k</td><td>64k-128k</td></tr><tr><td>2-Needle BF16</td><td> $5 4 . 7 2 \pm 2 . 4 0$ </td><td> $4 1 . 1 4 \pm 1 . 0 1$ </td><td> $3 7 . 6 9 \pm 3 . 4 7$ </td><td> $2 7 . 4 8 \pm 2 . 1 7$ </td><td> $2 3 . 1 6 \pm 3 . 3 1$ </td></tr><tr><td>WUSH-KV OSCAR</td><td> ${ \bf 1 8 . 0 5 \pm 0 . 2 6 }$   $1 5 . 3 6 \pm 1 . 3 0$ </td><td> ${ \bf 1 0 . 8 2 \pm 0 . 9 2 }$   $8 . 8 8 \pm 0 . 5 6$ </td><td> ${ \bf 1 0 . 1 4 \pm 0 . 1 8 }$   $7 . 0 5 \pm 0 . 4 0$ </td><td> ${ \bf 8 . 1 4 \pm 0 . 7 9 }$   $4 . 6 9 \pm 0 . 6 9$ </td><td> ${ \bf 5 . 4 9 \pm 0 . 1 4 }$   $0 . 2 9 \pm 0 . 1 1$ </td></tr><tr><td>4-Needle</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BF16</td><td> $2 9 . 6 4 \pm 1 . 7 1 $ </td><td> $2 2 . 7 3 \pm 1 . 1 6$ </td><td> $2 3 . 5 9 \pm 1 . 2 6$ </td><td> $1 9 . 4 8 \pm 3 . 1 7$ </td><td> $1 7 . 1 4 \pm 2 . 1 0$ </td></tr><tr><td>WUSH-KV</td><td> ${ \bf 1 2 . 8 3 \pm 0 . 3 6 }$ </td><td> $9 . 1 5 \pm 0 . 0 8$ </td><td> ${ \bf 1 0 . 9 3 \pm 1 . 2 1 }$ </td><td> ${ \bf 7 . 2 6 \pm 0 . 2 5 }$ </td><td> ${ \bf 7 . 8 7 \pm 0 . 4 1 }$ </td></tr><tr><td>OSCAR</td><td> $1 1 . 6 3 \pm 1 . 1 8$ </td><td> ${ \bf 9 . 4 4 \pm 0 . 6 5 }$ </td><td> $8 . 6 6 \pm 0 . 2 5$ </td><td> $3 . 6 1 \pm 0 . 2 4$ </td><td> $1 . 0 6 \pm 0 . 1 2$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>8-Needle</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BF16</td><td> $1 8 . 2 2 \pm 1 . 3 9$ </td><td> $2 0 . 6 7 \pm 1 . 3 9$ </td><td> $1 7 . 0 4 \pm 1 . 3 8$ </td><td> $1 3 . 7 8 \pm 1 . 3 1$ </td><td> $1 2 . 2 5 \pm 0 . 5 6$ </td></tr><tr><td>WUSH-KV</td><td> $8 . 9 9 \pm 0 . 1 9$ </td><td> ${ \bf 9 . 7 3 \pm 0 . 7 4 }$ </td><td> ${ \bf 8 . 3 6 \pm 0 . 4 6 }$ </td><td> ${ \bf 8 . 1 9 \pm 0 . 2 4 }$ </td><td> ${ \bf 6 . 9 8 \pm 0 . 1 5 }$ </td></tr><tr><td>OSCAR</td><td> ${ \bf 9 . 3 4 \pm 0 . 3 7 }$ </td><td> $8 . 6 3 \pm 0 . 3 8$ </td><td> $7 . 1 1 \pm 0 . 7 8$ </td><td> $4 . 2 6 \pm 0 . 7 9$ </td><td> $1 . 0 9 \pm 0 . 2 7$ </td></tr></table>

Table 7: Qwen3-32B MRCR accuracy (%) by needle count, including context-length buckets, overall scores, and AUC.
<table><tr><td>Method</td><td></td><td></td><td></td><td></td><td>0k-8k 8k-16k 16k-32k 32k-64k 64k-128k</td><td>Overall</td><td>AUC</td></tr><tr><td>2-Needle</td><td colspan="7"></td></tr><tr><td>BF16 WUSH-KV</td><td>70.98</td><td>60.14</td><td>50.88</td><td>38.27</td><td>36.87</td><td> $5 1 . 4 9 \pm 2 . 0 3$ </td><td>43.70</td></tr><tr><td>OSCAR</td><td>16.39 24.83</td><td>10.79 14.39</td><td>8.93</td><td>10.48</td><td>6.18</td><td> $1 0 . 5 7 \pm 0 . 6 9$ </td><td>9.25</td></tr><tr><td>4-Needle</td><td></td><td></td><td>9.01</td><td>10.87</td><td>3.92</td><td> ${ \bf 1 2 . 6 4 \pm 0 . 7 9 }$ </td><td>9.46</td></tr><tr><td>BF16</td><td colspan="7"></td></tr><tr><td>WUSH-KV</td><td>47.80</td><td>27.65</td><td>23.84</td><td>28.62</td><td>24.22</td><td> $3 0 . 7 5 \pm 1 . 7 6$ </td><td>27.03</td></tr><tr><td>OSCAR</td><td>12.25</td><td>4.79</td><td>8.26</td><td>8.41</td><td>8.44</td><td> $8 . 5 0 \pm 0 . 5 4$ </td><td>8.15</td></tr><tr><td></td><td>14.84</td><td>12.59</td><td>9.41</td><td>7.48</td><td>5.51</td><td> ${ \bf 1 0 . 0 6 \pm 0 . 6 0 }$ </td><td>8.10</td></tr><tr><td>8-Needle</td><td colspan="7"></td></tr><tr><td>BF16</td><td>29.41</td><td>21.58</td><td>17.50</td><td>25.01</td><td>15.32</td><td> $2 1 . 8 6 \pm 1 . 4 5$ </td><td>20.72</td></tr><tr><td>WUSH-KV</td><td>7.79</td><td>7.39</td><td>7.25</td><td>8.43</td><td>6.95</td><td> $7 . 5 7 \pm 0 . 4 5$ </td><td>7.68</td></tr><tr><td>OSCAR</td><td>9.87</td><td>9.67</td><td>8.98</td><td>8.29</td><td>4.62</td><td> ${ \bf 8 . 2 9 \pm 0 . 4 1 }$ </td><td>7.64</td></tr></table>