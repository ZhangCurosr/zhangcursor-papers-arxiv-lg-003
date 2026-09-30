# SCALE SENSITIVITY IN LOW-BIT POST-TRAINING QUANTIZATION: CURVATURE OF THE QUANTIZATION ERROR LANDSCAPE

Jonas von Berg\*& Massimiliano Datres\*& Carlo Kneißl & Gitta Kutyniok†   
Ludwig-Maximilians-Universität München   
Munich Center for Machine Learning (MCML)   
Konrad Zuse School of Excellence in Reliable AI (DAAD)   
{berg,datres,kneissl,kutyniok}@math.lmu.de

## ABSTRACT

Post-training quantization (PTQ) methods in the GPTQ family minimize a layerwise reconstruction error on a uniform grid whose scale must be chosen; the common max-based choice degrades sharply at low bit-widths. We study how sensitive this objective is to the scale. For a layer with i.i.d. Gaussian weights and calibration activations of sufficiently large effective rank, we prove that, as the width grows, the normalized round-to-nearest loss converges with high probability, uniformly over all scales, to the mean-squared error of a uniform quantizer applied to a standard Gaussian; we verify the effective-rank condition for wide, randomly initialized MLPs with odd Lipschitz activations and isotropic Gaussian calibration data. The limiting objective has a unique nondegenerate minimizer, whose scale decreases strictly with the number of levels and whose curvature with respect to relative scale errors decays approximately exponentially with the bitwidth. GPTQ experiments on five LLMs show the same trend: the scale rule changes perplexity substantially at 2-3 bits and negligibly from 6 bits on, and a local measure of GPTQ scale sensitivity decreases with bit-width in line with the Gaussian curvature. The Gaussian-optimal scale fails on raw weights; after Hadamard incoherence processing it matches the best searched rule at 3 bits and above without any search, but remains clearly worse at 2 bits.

## 1 INTRODUCTION

Quantization has emerged as a key approach to reduce the computational cost and memory footprint of modern machine learning models. As these models grow larger, methods that compress them and accelerate inference without retraining have attracted considerable interest in the machine learning community. The success of small-scale language models (LLMs) that can be deployed locally, such as Llama 3.2 (Dubey et al., 2024), DeepSeek (Liu et al., 2024), and BitNet (Ma et al., 2025), has further intensified this interest.

Among various compression strategies, Post-training quantization (PTQ) is especially attractive because it can be applied to a trained model using only a small calibration set, without the need of retraining. It therefore offers a practical route to efficient deployment with modest additional overhead. Despite its widespread adoption, the theoretical understanding of PTQ remains limited. A wide variety of PTQ methods have been proposed, such as GPTQ Frantar et al. (2022) and AWQ Lin et al. (2024), but a mathematical analysis of these methods has only recently begun to emerge Tseng et al. (2024); Chen et al. (2026); Zhang et al. (2026). All of these, however, consider the quantization problem on a fixed grid. A key parameter shared by these algorithms is the quantization scale, which determines the spacing and range of the quantization grid. Although many rules have been proposed for selecting it Banner et al. (2019); Nagel et al. (2021), see also Table 5, it remains unclear which choices best preserve model performance and how sensitive quantization algorithms are to the quantization scale choice. Independently of how the scale is chosen, these methods are surprisingly accurate at 4 or 8 bits. At lower precision, however, performance degradation seems unavoidable Dettmers & Zettlemoyer (2023).

Preprocessing techniques can further reduce the degradation caused by quantization. A prominent example is Hadamard incoherence processing (HIP) (Tseng et al., 2024), which applies a random orthogonal transformation to the weights so as to spread their magnitudes evenly across coordinates. As a consequence, the entries of the transformed weights become approximately Gaussian.

In this work, we take a step towards a mathematical understanding of how scale choice affects these algorithms. We view PTQ as the problem of approximating the linear map of a network layer, in the layerwise reconstruction sense, at a fixed quantization granularity. Our main observation is that, for networks at initialization and under suitable conditions on the calibration data, the reconstruction loss under round-to-nearest (RTN) quantization, viewed as a function of the scale, is well approximated by a deterministic Gaussian quantization objective. We then analyze the Gaussian quantization problem, characterizing its loss landscape around the optimal scale and its dependence on the bit width. We use this analysis to shed light on the scale sensitivity of existing PTQ methods, especially when combined with HIP preprocessing.

Paper Roadmap. Section 2 reviews related work, and Section 3 introduces the notation and quantization setup. Section 4 establishes uniform convergence to a Gaussian limiting loss, characterizes its landscape, and verifies the convergence assumptions for randomly initialized MLPs with Gaussian data. Section 5 evaluates scale sensitivity in LLM quantization. Proofs and additional experiments appear in the appendix.

## Contributions. We summarize our contributions below.

1. Uniform Gaussian limit. Under Gaussian weight and effective-rank assumptions, we prove uniform convergence over scales of the normalized RTN reconstruction loss to a scalar Gaussian objective, with explicit finite-dimensional, high-probability bounds (Theorem 1). We verify the rank condition for wide MLPs at initialization (Theorem 5).

2. Optimal scales and sensitivity. For an analytic extension of the limiting objective, we prove a unique, nondegenerate minimizer and a strictly decreasing optimal scale as quantizer resolution increases (Proposition 1), extending the discrete monotonicity of Na & Neuhoff (2018). We derive its scale-invariant curvature and numerically observe approximately exponential decay with bit-width.

3. Implications for LLM quantization. Across several LLMs, GPTQ exhibits greater sensitivity to scale selection at 2–3 bits than at 6–8 bits, with local sensitivity trends consistent with the Gaussian model. HIP substantially improves the performance of the resulting analytic scale rule, which requires no scale search. We also introduce Proxy-static, an efficient GPTQ-aware scale-selection objective that performs strongly in our experiments, particularly without preprocessing.

## 2 RELATED WORK

Layerwise post-training quantization. For scalability, PTQ often minimizes the layerwise reconstruction error ∥(W — Î)X| on a fixed quantization grid. GPTQ Frantar et al. (2022), building on Optimal Brain Quantization Frantar & Alistarh (2022) and OBS Hassibi & Stork (1992), quantizes columns sequentially while compensating errors through the remaining floating-point weights. GPTAQ Li et al. (2025), ResComp Li et al. (2026), and QRoNoS Zhang et al. (2026) modify reconstruction and error compensation. Orthogonal preprocessing (Chee et al., 2023; Tseng et al., 2024; Ashkboos et al., 2024; Liu et al., 2025), activation smoothing (Xiao et al., 2023), activationaware weight scaling Lin et al. (2024), and magnitude reduction (Zhang et al., 2024) make weights and activations easier to quantize. In particular, QuIP Tseng et al. (2024) uses random orthogonal preprocessing to obtain incoherence-based error guarantees for adaptive rounding on a fixed grid. Scale selection remains often heuristic; we study its influence, especially at low precision, without changing the projection procedure. BRECQ Li et al. (2021) combines blockwise reconstruction with learned rounding based on AdaRound Nagel et al. (2020), improving low-bit accuracy at an optimization cost that can be prohibitively expensive for large models.

Scale and clipping selection. OmniQuant Shao et al. (2024) optimizes clipping parameters and equivalent transformations through blockwise reconstruction, avoiding per-weight rounding optimization. PiSO (Amboage et al., 2026) exploits the piecewise-quadratic objective to compute exact channelwise RTN scales, but reports collapse at 3 bits in the data-free setting. ACIQ (Banner et al., 2018) derives Gaussian and Laplacian clipping thresholds, applying them only to activations after finding no benefit for weights. LLAPQ Nahshan et al. (2021) studies the full-model task-loss landscape over quantization parameters. We instead study layerwise reconstruction, establish a scalar asymptotic limit under coordinatewise RTN, and empirically examine scale sensitivity for RTN and GPTQ.

Optimal uniform scalar quantization. Classical scalar quantization theory studies optimal quantizers for specific distributions. Max (Max, 1960) derived optimality conditions and numerical solutions, including for Gaussian sources. For symmetric uniform quantization, Na and Neuhoff (Na & Neuhoff, 2017; 2018) study distortion convexity in the grid scale and prove that the Gaussianoptimal scale decreases when two levels are added. We connect this theory to layerwise reconstruction through a rigorous asymptotic limit, derive explicit curvature formulas, and compare them with empirical estimates from real quantization landscapes.

## 3 NOTATION AND PRELIMINARIES

Notation. We use standard notation for sets, Gaussian distributions, and differentiability classes, with $\mathbb { N } = 1 , 2 , . . .$ . and $[ n ] = 1 , \ldots , n$ . Write $\operatorname { I d } _ { d }$ for the identity matrix and $\mathbb { S } ^ { p - 1 }$ for the Euclidean unit sphere in RP. Vectors, including matrix rows $M _ { i : }$ , are viewed as columns. The norms $\| \cdot \| _ { o p } ,$ $\| \cdot \| _ { F }$ , and $\| \cdot \| _ { \psi _ { 2 } }$ denote the operator, Frobenius, and sub-Gaussian norms, respectively. For a nonzero positive semidefinite matrix M, the effective rank is given by

$$
r _ { \mathrm { e f f } } ( M ) : = \mathrm { T r } ( M ) / \| M \| _ { o p } .
$$

Symmetric rounding is $\lfloor t \rceil : = \mathrm { s i g n } ( t ) \lfloor | t | + 1 / 2 \rfloor$ , with $\mathrm { s i g n } ( 0 ) = 0$ We denote the standard Gaussian density and cumulative density function by $\varphi$ and Φ. Subscripts on E, IP specify the variable integrated over; primes and subscripts denote ordinary and partial derivatives respectively, e.g., $f _ { s s m } = \partial _ { m } \partial _ { s } ^ { 2 } f$ . Constants $\tilde { C } , C , c > 0$ may change from line to line.

Post-training quantization. Post-training quantization $( \mathrm { P T Q } )$ maps the weights of a trained neural network onto a low-bit grid without retraining, aiming to preserve its input-output behavior using a small calibration set. Let $\vartheta \in \mathbb { R } ^ { d }$ be a vector of full-precision weights, let $B \geq \bar { 2 }$ denote the target bit precision, and set $M _ { B } : = 2 ^ { B - 1 } - 1$ . We define the symmetric quantization cone $\Gamma _ { B } \subset \mathbb { R } ^ { d }$ as the union of all symmetric uniform grids with $2 ^ { B } - 1$ levels per coordinate:

$$
\Gamma _ { B } : = \left\{ s v : s > 0 , v \in \{ - M _ { B } , \ldots , M _ { B } \} ^ { d } \right\} .
$$

For a fixed scale $s > 0$ , coordinatewise round-to-nearest (RTN) quantization is the Euclidean projection onto the corresponding grid:

$$
\Pi _ { s , M _ { B } } ( \vartheta ) : = s \left\lfloor \frac { \vartheta } { s } \right\rceil _ { B } , \qquad \left\lfloor t \right\rceil _ { B } : = \left\{ \begin{array} { l l } { - M _ { B } } & { t < - M _ { B } , } \\ { \left\lfloor t \right\rceil } & { \left\lfloor t \right\rfloor \le M _ { B } , } \\ { M _ { B } } & { t > M _ { B } . } \end{array} \right.\tag{1}
$$

where $\lfloor \cdot \rceil _ { B }$ is applied componentwise. We write $\mathcal { Q } _ { B } : = \{ \Pi _ { s , M _ { B } } : s > 0 \}$ for the resulting family of quantization maps.

We study layerwise PTQ through the empirical squared $L _ { 2 }$ distance between a full-precision linear map and its quantized approximation, evaluated on the same input activations. Let $\dot { \mathcal { X } } \subseteq \mathbb { R } ^ { d _ { 0 } }$ denote the input space, equipped with a probability measure $\mu ,$ and let $\mathbb { D } _ { \mathrm { c a l } } = \{ X _ { j } ^ { \prime } \} _ { j = 1 } ^ { N _ { c } }$ consist of $N _ { c }$ i.i.d. samples from $\mu .$ For a layer with weights $\vartheta ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { \ell } \times d _ { \ell - 1 } }$ , we consider the RTN scale-selection problem of minimizing, over $s > 0$ , the reconstruction error

$$
\begin{array} { r } { \mathcal { E } ^ { ( \ell ) } ( s , \vartheta ^ { ( \ell ) } , X ^ { \ell - 1 } ) : = \left\| \left( \vartheta ^ { ( \ell ) } - \Pi _ { s , M _ { B } } ( \vartheta ^ { ( \ell ) } ) \right) X ^ { \ell - 1 } \right\| _ { F } ^ { 2 } , } \end{array}\tag{2}
$$

where the quantization map acts entrywise. Here, $X ^ { \ell - 1 } \in \mathbb { R } ^ { d _ { \ell - 1 } \times N _ { c } }$ denotes the input activation matrix used to calibrate layer l. Its columns are obtained by propagating the calibration samples through the preceding layers, which may be full-precision or already quantized, depending on the PTQ procedure. We hold $\dot { X } ^ { \ell - 1 }$ fixed when optimizing the scale of the current layer. We write $\mathcal { E } ^ { ( \ell ) } ( s )$ when the dependence on $\vartheta ^ { ( \ell ) }$ and $X ^ { \ell - 1 }$ is clear.

This formulation also covers query, key, value, and output projections in attention, using their respective input activations. While (2) uses one scale per layer, other granularities are commonly used. For instance, per-channel quantization assigns one scale per channel of $\vartheta ^ { ( \ell ) }$ , yielding de independent row-wise problems. Finer granularities, such as group quantization, are also used.

A simple scale-selection rule is the symmetric min-max choice $\begin{array} { r } { s ^ { e } : = \frac { \operatorname* { m a x } _ { i , j } | \vartheta _ { i j } ^ { ( \ell ) } | } { M _ { B } } } \end{array}$ , with the maximum taken over the relevant channel or group for finer granularities. This choice uses only the weight range and does not optimize (2). Our experiments show that it can perform well at higher precision, while becoming substantially less reliable at lower bit-widths.

## 4 THEORETICAL RESULTS

In this section, we study how sensitive the reconstruction error (2) is to the quantization scale s across different bit-widths B. We focus on one output channel and write $\vartheta \in \mathbb { R } ^ { d }$ for its full-precision weight vector, where $d = d _ { \ell - 1 }$ is the input dimension. Let $X \in \mathbb { R } ^ { d \times N _ { c } }$ denote the corresponding calibration activations. For $\mathrm { t r } ( X X ^ { \top } ) > \mathbf { \bar { 0 } }$ , we consider the normalized reconstruction loss

$$
\mathcal { E } ( s , \vartheta , X ) = \mathcal { E } ( s ) : = \frac { \| \vartheta ^ { \top } X - \Pi _ { s , M _ { B } } ( \vartheta ) ^ { \top } X \| _ { 2 } ^ { 2 } } { \operatorname { t r } ( X X ^ { \top } ) } = \frac { \big ( \vartheta - \Pi _ { s , M _ { B } } ( \vartheta ) \big ) ^ { \top } X X ^ { \top } \big ( \vartheta - \Pi _ { s , M _ { B } } ( \vartheta ) \big ) } { \operatorname { t r } ( X X ^ { \top } ) } .
$$

The normalization by $\operatorname { t r } ( X X ^ { \top } )$ removes the overall scale of the calibration activations and, under our assumptions, yields a loss of order one as d grows. For instance, when $X X ^ { \top } = \operatorname { I d } _ { d }$ , the normalized loss reduces to $\begin{array} { r } { \mathcal { E } ( s ) = \frac { 1 } { d } \Vert \vartheta - \Pi _ { s , M _ { B } } ( \vartheta ) \Vert _ { 2 } ^ { 2 } } \end{array}$ , the mean squared approximation error per coordinate. For fixed $s > 0$ and bit-width B, this admits a nontrivial limit as $d \to \infty$ when $\vartheta \sim \mathcal { N } ( 0 , \mathrm { I d } _ { d } )$ . In the rest of this section, we assume that

(H1) the parameters $\vartheta \stackrel { i i d } { \sim } \mathcal { N } ( 0 , 1 )$

(H2) the effective rank of $X X ^ { \top }$ satisfies $r _ { \mathrm { e f f } } ( X X ^ { \top } ) \ge C \log ^ { 8 } { c }$ l for some constant $C > 0$ not dependent on d.

Assumptions (H1) is standard in the literature on wide random neural networks. (H2) asks that the inputs to the group of parameters being quantized are not concentrated on a small number of directions. Section A.4 establishes (H2) with high probability for an MLP at initialization under the stated assumptions on its architecture and isotropic input distribution.

We study the asymptotic limit of $s \mapsto { \mathcal { E } } ( s )$ to gain insight into its loss landscape. In Section 4.1, we establish a high-probability bound yielding uniform convergence in s to the deterministic limiting objective

$$
\mathcal { E } ^ { \infty } ( s ) : = \mathbb { E } _ { Z \sim \mathcal { N } ( 0 , 1 ) } \big [ ( Z - \Pi _ { s , M _ { B } } ( Z ) ) ^ { 2 } \big ] .
$$

We then analyze the landscape of $s \mapsto { \mathcal { E } } ^ { \infty } ( s )$ in Section 4.2.

## 4.1 CONVERGENCE TO THE LIMIT

Since our goal is to study the loss as a function of s, we seek an approximation that holds simultaneously over all $s > 0$ . Such a uniform guarantee also applies when s is chosen based on the realization of  and the calibration data X. The following theorem provides this guarantee with an explicit error bound.

Theorem 1. Let $B \geq 2 , X \in \mathbb { R } ^ { d \times N _ { c } }$ be nonzero matrix satisfying (H2), and θ satisfying (H1) Then, for all $d \geq 2 ,$ it holds

$$
\mathbb { P } \left( \operatorname* { s u p } _ { s > 0 } | { \mathcal { E } } ( s ) - { \mathcal { E } } ^ { \infty } ( s ) | > { \frac { 2 + 1 2 M _ { B } } { \log d } } \right) \leq C M _ { B } d e ^ { - c \log ^ { 2 } d } ,
$$

where $c , C > 0$ are absolute constants independent on d.

A detailed proof of Theorem 1 appears in Appendix $\mathrm { A } . 2$ For fixed B, the theorem bounds the approximation error by a constant multiple of $( \log d ) ^ { - 1 }$ with probability tending to one as $d \to \infty$ In particular,

$$
\operatorname* { s u p } _ { s > 0 } | { \mathcal E } ( s ) - { \mathcal E } ^ { \infty } ( s ) | \xrightarrow [ d  \infty ] { \mathbb P } 0 .
$$

Let $m : = \operatorname* { i n f } _ { s > 0 } \mathcal { E } ( s )$ and $m ^ { \infty } : = \operatorname* { i n f } _ { s > 0 } \mathcal { E } ^ { \infty } ( s )$ denote the optimal empirical and limiting reconstruction losses. We note that both m and $m ^ { \infty }$ are finite, since ${ \mathcal { E } } , { \mathcal { E } } ^ { \infty } \geq 0$ . Uniform convergence immediately implies $m \stackrel { \mathbb { P } } {  } m ^ { \infty }$ . Convergence of minimizers is established in Corollary 1 in $\mathsf { A p - }$ pendix A.3.

Remark 1. The uniform approximation of Theorem 1 relies on (H2), namely a lower bound on the effective rank of $X X ^ { \top }$ . In Appendix $\mathrm { A . 4 }$ we show that this condition holds, with high probability, in the setting of a random multi-layer perceptron with odd activation functions and isotropic inputs.

## 4.2 LIMITING FUNCTION LANDSCAPE

The uniform convergence established in Theorem 1 motivates the study of the limiting functional

$$
\mathcal { E } ^ { \infty } ( M _ { B } , s ) : = \int _ { \mathbb { R } } ( s \ \lfloor t / s \rceil _ { B } - t ) ^ { 2 } \varphi ( t ) d t .\tag{3}
$$

Since our goal is to understand the loss landscape of ${ \mathcal { E } } ^ { \infty }$ as a function of the bit precision B, which enters only through $M _ { B }$ , we make this dependence explicit in the notation, that is from now on we write $\mathcal { E } ^ { \infty } \dot { ( } M _ { B } , s \dot { ) }$ in place of ${ \mathcal { E } } ^ { \infty } ( s )$ used above. We start by noting that $\mathcal { E } ^ { \infty } ( M _ { B } , s )$ splits into the rounding error of an infinite grid, which depends on s only, and a nonnegative clipping term, which is the only part through which the bit precision $B$ enters. More precisely, $\bar { \mathcal { E } } ^ { \infty } ( \bar { M _ { B } } , s ) =$ $\begin{array} { r } { 1 - 4 s \sum _ { j = 0 } ^ { \infty } g \big ( ( j + \frac { 1 } { 2 } ) s \big ) + \bar { 4 } s \sum _ { j = 0 } ^ { \infty } g \big ( ( M _ { B } + j + \frac { 1 } { 2 } ) s \big ) } \end{array}$ , where $g ( x ) : = \varphi ( x ) - x \left( 1 - \Phi ( x ) \right)$ . The derivation of this expression, together with the proof of its well-posedness, is deferred to Corollary 2 in Appendix A.5. For the analysis that follows it is convenient to let the second argument vary continuously. We therefore consider the extension $\bar { \mathcal { E } } ^ { \infty } : [ 1 , \infty ) \times ( 0 , \infty )  ( 0 , \infty )$ of ${ \mathcal { E } } ^ { \infty }$ , defined by $\begin{array} { r } { \bar { \mathcal { E } } ^ { \infty } ( m , \bar { s } ) : = 1 - 4 s \sum _ { j = 0 } ^ { \infty } g \big ( ( j + \frac { 1 } { 2 } ) s \big ) + 4 s \sum _ { j = 0 } ^ { \infty } g \big ( ( m + \dot { j } + \frac { 1 } { 2 } ) s \big ) } \end{array}$ , so that $\bar { \mathcal { E } } ^ { \infty } ( M _ { B } , s ) =$ $\mathcal { E } ^ { \infty } ( M _ { B } , s )$ for every $B \geq 2 .$

The following proposition shows that $\bar { \mathcal { E } } ^ { \infty }$ is smooth, that for every m the function $s \mapsto { \bar { \mathcal { E } } } ^ { \infty } ( m , s )$ has a unique critical point, which is a nondegenerate $\mathrm { g l }$ obal minimizer $s ^ { * } ( m )$ , and that the optimal scale $s ^ { * } ( m )$ is a smooth, strictly decreasing function of $m$ , bounded below by ${ \sqrt { 2 } } / ( m + { \frac { 1 } { 2 } } )$ •

Proposition 1. The following properties hold:

(i) For every $B \geq 2 ,$ the function $\bar { \mathcal { E } } ^ { \infty } ( m , s )$ is smooth, $e . g . \ \bar { \mathcal { E } } ^ { \infty } ( m , s ) \in \mathcal { C } ^ { \infty } ( [ 1 , \infty ) \times ( 0 , \infty ) )$

(ii) For every real $m \geq 1$ , the function $s \mapsto { \bar { \mathcal { E } } } ^ { \infty } ( m , s )$ has a unique critical point on $( 0 , \infty )$ This point is its unique global minimizer $s ^ { * } ( \dot { m } )$ and is nondegenerate:

$$
\partial _ { s } ^ { 2 } \bar { \mathcal { E } } ^ { \infty } ( m , s ^ { * } ( m ) ) > 0 .
$$

(iii) The map m $\mapsto \ s ^ { * } ( m )$ is continuously differentiable and strictly decreasing on $\lbrack 1 , \infty )$ Moreover, for every $m \geq 1$

$$
\frac { d s ^ { * } } { d m } ( m ) = - \frac { \bar { \mathcal { E } } _ { m s } ^ { \infty } \bigl ( m , s ^ { * } ( m ) \bigr ) } { \bar { \mathcal { E } } _ { s s } ^ { \infty } \bigl ( m , s ^ { * } ( m ) \bigr ) } < 0 , \qquad s ^ { * } ( m ) > \frac { \sqrt { 2 } } { m + \frac { 1 } { 2 } } .
$$

Proposition 1 describes the landscape of the limiting energy $\bar { \mathcal { E } } ^ { \infty } ( m , \cdot )$ as a function of the scale s for a fixed level parameter m. Item (ii) shows that it has a unique critical point, which is a nondegenerate global minimum. Item (iii) shows that increasing the number of levels strictly decreases the optimal scale $s ^ { * } ( m )$ . However, at fixed $m ,$ decreasing the scale trades finer resolution for a narrower representable range and hence greater clipping error. The optimal scale balances these effects and satisfies $s ^ { * } ( m ) > \sqrt { 2 } / ( m + \textstyle { \frac { 1 } { 2 } } )$ 1

Beyond existence and uniqueness of the optimal scale $s ^ { * } ( m )$ , we are interested in how sensitive the energy is to scale choice. Since the optimal scale changes with the number of levels, we consider relative rather than absolute scale perturbations. The ordinary second derivative measures sensitivity to additive changes in s; multiplying it by $s ^ { 2 }$ instead measures sensitivity to proportional changes.

Definition 1. Let $m \in [ 1 , \infty )$ be such that $\bar { \mathcal { E } } ^ { \infty } ( m , \cdot )$ is twice differentiable at $s ^ { * } ( m )$ . We define its scale-invariant curvature by $\kappa ( m , s ^ { * } ( m ) ) : = s ^ { 2 } \bar { \mathcal { E } } _ { s s } ^ { \infty } ( m , s ^ { * } ( m ) )$

At the minimizer $s ^ { * } = s ^ { * } ( m )$ , a relative perturbation $s = s ^ { * } ( 1 + \varepsilon )$ gives $\bar { \mathcal { E } } ^ { \infty } ( m , s ^ { * } ( 1 + \varepsilon ) ) -$ $\begin{array} { r } { \bar { \mathcal { E } } ^ { \infty } ( m , s ^ { * } ) = \frac { 1 } { 2 } \kappa ( m , s ^ { * } ) \bar { \varepsilon } ^ { 2 } + \overset { . } { o } ( \varepsilon ^ { 2 } ) } \end{array}$ . We invite the reader to Remark A.5.2 for a more detailed discussion on the role of $s ^ { ( } * ) ( m )$ in the definition of κ. Thus, $\kappa ( m , s ^ { * } )$ quantifies the local increase in reconstruction loss caused by a relative error in the scale. An explicit computation of $\kappa ( m , s ^ { * } ( m ) )$ for an integer $m > 0$ , leads to

$$
\kappa ( m , s ) = 4 s \sum _ { j = 0 } ^ { m - 1 } \Big [ 2 \big ( j + \textstyle { \frac { 1 } { 2 } } \big ) s \Big ( 1 - \Phi \big ( ( j + \textstyle { \frac { 1 } { 2 } } ) s \big ) \Big ) - \big ( j + \textstyle { \frac { 1 } { 2 } } \big ) ^ { 2 } s ^ { 2 } \varphi \big ( ( j + \textstyle { \frac { 1 } { 2 } } ) s \big ) \Big ] .
$$

The derivation of this formula can be found in Appendix A.5.1. A numerical evaluation for $B =$ $2 , \ldots , 8$ shows that $\kappa ( M _ { B } , s ^ { * } ( M _ { B } ) )$ is exponentially decreasing with B as shown in Figure 4. We refer to Algorithm 1 in Appendix A.5.1 for the algorithm we have used for the numerical evaluation.

This shows that the loss landscape is sharper near its minimum at low bit-widths and flatter at high bit-widths, when measured with respect to relative changes in the quantization scale.

## 5 EXPERIMENTS

The previous section showed that, under suitable assumptions, the layerwise PTQ reconstruction loss converges to the simpler asymptotic loss (3). We now test whether predictions from this limiting landscape carry over to trained models. In particular, we want to answer the following questions: 1) Does the greater local scale sensitivity of the limiting objective at lower precision translate into greater sensitivity to scale choice under GPTQ? 2) Does the analytic scale $\mathbf { \bar { \Psi } } { s } = { s } ^ { * } ( { M } _ { B } ) \hat { \sigma }$ remain near-optimal for the reconstruction loss of trained, finite-width layers? Here, ô is the RMS of the weights in the channel being quantized, and $s ^ { * } ( M _ { B } )$ is computed by solving $\mathcal { E } _ { s } ^ { \infty } ( M _ { B } , s ^ { * } ( M _ { B } ) ) =$ 0, using the explicit expression in (28).

## 5.1 SCALE SENSITIVITY OF LLM POST-TRAINING QUANTIZATION

We quantize several LLMs using GPTQ, with a calibration set of 128 randomly sampled windows of 2,048 tokens from the C4 training split. We use per-channel quantization and vary only the scaleselection rule. Our theory concerns coordinatewise RTN, whereas GPTQ updates the remaining weights to compensate for quantization errors. The two procedures coincide on the same fixed grid when $X X ^ { \top }$ is diagonal, but GPTQ is not generally covered by our theory. These experiments therefore test whether the predicted scale sensitivity extends to GPTQ. We consider five different scaleselection rules, ranging from no search (Min-max) through search without activation data (WMSE, Shrink-2.4) to Hessian-weighted search (Proxy-static, RTNH), as specified in Table 5.

Remark 2. Existing scale rules target the RTN loss, which need not align with GPTQ's compensated loss without preprocessing. Our Proxy-static rule, see Appendix B.1.3, reuses GPTQ's inverse-Hessian factorization to account for the compensation: it is the best rule without HIP (Table 1) but gives no clear gain with HIP (Table 2).

For each model and precision, Figure 1 reports the performance range: the difference between the maximum and minimum NLL across the five scale-selection rules. This measures how much the quantized model performance depends on the choice of scale-selection rule. We observe, that from W6 onward, all scale rules give essentially the same NLL. The scale matters most at W2 and W3, consistent with κ decreasing exponentially in the number of bits, i.e., the theoretical loss landscape being more sensitive to relative scale perturbations at low precision. Amboage et al. (2026) reports a related trend, i.e. data-aware scale optimization pays off increasingly as the bit-width decreases. We note that, for group size 128, the performance differences between scale-selection rules diminish even more rapidly with bit-width, consistent with the greater flexibility of fitting scales to smaller groups of weights.

Remark 3. The NLL spread across scale-selection rules reflects both the model's sensitivity to scale changes and the differences between the scales selected by the rules. It shows the practical importance of scale selection in low-precision settings, but does not establish higher curvature of the local

![](images/f27b8021414b61b466c617b1c9ef01a77a37de5328dc7d357da5679912d02e57.jpg)  
Figure 1: Log-scale GPTQ sensitivity to the scale rule across bit per channel (left) and group size 128 (right) for different models. NLL spread is the max minus min WikiText-2 NLL across Minmax, Shrink-2.4, WMSE, Proxy-static, and RTNH.

GPTQ reconstruction loss. Appendix B.2.4 complements this comparison by measuring sensitivity under fixed relative scale perturbations.

Table 1: Llama-3.1-8B, W3 per-channel GPTQ without HIP. Lower perplexity and higher accuracy indicate better performance. Proxy-static, the GPTQ-aware scale-selection rule introduced in Appendix B.1.3, consistently outperforms the other tested rules in this setting.
<table><tr><td>Rule</td><td>WikiText-2 PPL</td><td>C4 PPL PIQA</td><td></td><td>ARC-E</td><td></td><td>ARC-C HellaSwag</td><td>WinoGrande BoolQ</td><td></td><td>Mean</td></tr><tr><td>Shrink-2.4</td><td>14.050</td><td>17.177</td><td>0.705</td><td>0.553</td><td>0.369</td><td>0.708</td><td>0.666</td><td>0.795</td><td>0.632</td></tr><tr><td>Proxy-static</td><td>11.756</td><td>16.328</td><td>0.729</td><td>0.635</td><td>0.412</td><td>0.721</td><td>0.685</td><td>0.798</td><td>0.663</td></tr><tr><td>Min-max</td><td>41.473</td><td>45.506</td><td>0.624</td><td>0.408</td><td>0.282</td><td>0.472</td><td>0.515</td><td>0.524</td><td>0.471</td></tr><tr><td>Analytic</td><td>930,372.777643,344.609</td><td></td><td>0.514</td><td>0.254</td><td>0.268</td><td>0.267</td><td>0.507</td><td>0.622</td><td>0.405</td></tr></table>

Table 1 contrasts the best and worst of the five heuristic scale-selection rules for Llama-3.1-8B at W3 without HIP, together with zero-shot accuracy on six benchmarks to assess whether perplexity differences carry over to downstream tasks. To address the second question, we also include the analytic scale $s ^ { * } ( M _ { B } ) \hat { \sigma }$

## 5.2 PREPROCESSING AND SCALE SENSITIVITY

Scale selection strongly affects performance, and the asymptotically optimal scale fails on the unprocessed model. Since trained weights need not be iid Gaussian, we ask whether preprocessing can preserve the model's function while bringing its weights closer to the assumptions of Theorem 1 and making $s ^ { * } ( M _ { B } ) \hat { \sigma }$ near-optimal for the finite-width reconstruction objective.

Figure 2 shows that, after HIP (see Appendix B.4 for a short description), the analytic rule $s ^ { * } ( M _ { B } ) \hat { \sigma }$ consistently matches the best tested heuristic in NLL across the tested models at W3 and above, without activation-based scoring or scale search.  
Table 2: Llama-3.1-8B, W3 per-channel GPTQ with HIP. Lower is better for perplexity; higher is better for accuracy. Best results are shown in bold.
<table><tr><td>Rule</td><td>WikiText-2 PPL</td><td>C4 PPL</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>HellaSwag</td><td>WinoGrande BoolQ</td><td></td><td>Mean</td></tr><tr><td>Shrink-2.4</td><td>9.095</td><td>14.452</td><td>0.785</td><td>0.744</td><td>0.508</td><td>0.740</td><td>0.717</td><td>0.817</td><td>0.718</td></tr><tr><td>Proxy-static</td><td>9.229</td><td>14.618</td><td>0.781</td><td>0.757</td><td>0.488</td><td>0.739</td><td>0.733</td><td>0.817</td><td>0.719</td></tr><tr><td>Min-max</td><td>15.39622.775</td><td></td><td>0.705</td><td>0.544</td><td>0.332</td><td>0.620</td><td>0.595</td><td>0.621</td><td>0.569</td></tr><tr><td>Analytic</td><td></td><td>9.21414.602</td><td>0.786</td><td>0.761</td><td>0.506</td><td>0.737</td><td>0.711</td><td>0.824</td><td>0.721</td></tr></table>

![](images/642ee52131d5b5dbdc37f6d78e9ec798a6e0462f9966ac00472d6cc49296f2a8.jpg)  
Figure 2: HIP substantially reduces the NLL gap between the analytic rule $s ^ { * } ( M _ { B } ) \hat { \sigma }$ and the best of five heuristics under per-channel GPTQ across tested models and precisions. We report the positive part of this gap, setting it to zero whenever the analytic rule matches or outperforms all five heuristics.

QuIP# Tseng et al. (2024) likewise selects the scales of its vector codebooks from the Gaussian quantization error. Our analysis makes this connection rigorous for uniform scalar quantization of the layerwise reconstruction objective, and our experiments show that the resulting analytic scale transfers, after HIP, to GPTQ (and to QRoNoS and ResComp, see the appendix) with competitive performance and no scale search.

## 5.3 LOCAL RECONSTRUCTION SENSITIVITY

We now examine scale sensitivity directly at the level of the reconstruction objective. For each weight matrix, we evaluate 32 predetermined rows using the calibration activations from a sequential GPTQ run. We repeat the experiment across five models, five precisions, and three calibration draws.

For each row, let ê denote the best searched GPTQ scale and $L ( s )$ its RMS-normalized, Hessianweighted reconstruction loss. We measure sensitivity to relative scale perturbations through

$$
C _ { \mathrm { s y m } } ( \delta ) : = \frac { L ( \hat { s } ( 1 + \delta ) ) + L ( \hat { s } ( 1 - \delta ) ) - 2 L ( \hat { s } ) } { \delta ^ { 2 } } .\tag{4}
$$

For a twice-differentiable loss, this converges to $\hat { s } ^ { 2 } L ^ { \prime \prime } ( \hat { s } )$ as $\delta \  \ 0$ For the empirical GPTQ landscapes, we interpret it as a finite-window sensitivity measure.

Figure 3 shows that the median empirical GPTQ sensitivity¹ decreases approximately exponentially with bit-width over the tested range, consistently across all three perturbation windows. The theoretical Gaussian curvature exhibits the same trend, and at $\delta = 0 . { \overset { - } { 0 } } 5$ , the empirical and theoretical quantities also agree closely in magnitude across models. This behavior mirrors the global sensitivity trend in Figure 1. Together, these observations suggest that the Gaussian landscape provides a useful model for the bit-width dependence of GPTQ sensitivity even without HIP preprocessing, despite substantial differences between the empirically optimal scales and their Gaussian predictions.

Figure 15 illustrates the reconstruction landscape for one weight row across precisions, with and without HIP preprocessing. Figure 13 further shows that median empirical GPTQ sensitivity after HIP follows a similar bit-width dependence to that observed without preprocessing. This suggests that the reduced performance spread across scale-selection rules after HIP could reflect better agreement among the scales selected by these rules, and not a flatter local landscape.

Note that, HIP's rotation alone preserves the effective rank of the input Gram matrix $H = X X ^ { \top }$ Motivated by the tighter convergence bound at higher effective rank (Theorem 1), we test whether transformations that change the spectrum improve analytic scale prediction.

We study W2 per-channel GPTQ on Llama-3.1-8B-Instruct, using a single unrotated sequential quantization stream. We partition the model's 32 blocks into ten fixed depth bins. Within each bin, we select the matrix with the largest normalized reconstruction error after per-row scale search, then its highest-error row. We compare four settings: no preprocessing, HIP, diagonal scaling followed by HIP, and whitening followed by HIP (B.4). All settings use the same pre-target weights and incoming activations; preprocessing is applied only to the target.

![](images/2fe5b37e1f768767d1ecbb40bf68d3b130ee0773b699d20fd8aea51cef4560ae.jpg)  
Figure 3: Colored curves: median finite-window GPTQ sensitivity $C _ { \mathrm { s y m } } ( \delta )$ in logscale at rowwise scales optimized by grid search for different δ. Dashed black: Gaussian curvature $\mathsf { \Gamma } _ { \kappa \left( M _ { B } , s ^ { \star } \left( M _ { B } \right) \right) }$ computed using Algorithm 1. Same exponential decay with number of bits as the empirical GPTQ sensitivity. The dependence of $C _ { \mathrm { s y m } } ( \delta )$ on the perturbation window indicates that these finitewindow measurements cannot be interpreted as a single local quadratic curvature.

Relative to HIP alone, diagonal scaling and whitening reduce the median scale gap $\begin{array} { r l } { \left| \log \left( \frac { s _ { \mathrm { s e a r c h } } } { s _ { \mathrm { a n } } } \right) \right| ~ } & { { } } \end{array}$ where $s _ { \mathrm { a n } } ~ = ~ s ^ { * } ( M _ { 2 } ) \hat { \sigma }$ , from 0.187 to 0.0298 and 0.00406, improving agreement in $8 / 1 0$ and $1 0 / 1 0$ selected rows, respectively (Figure 10). Here, $s _ { \mathrm { s e a r c h } }$ is the best searched GPTQ scale, not a certified global optimum. However, using a common normalization across settings, the best searched reconstruction error increases in every row, by median factors of 3.63 and 20.8 relative to HIP alone (Table 9).

These results reveal a trade-off in the selected cases: the transformations bring the best searched scale closer to the analytic prediction, but worsen the reconstruction error attained by GPTQ even after scale optimization.

## 6 LIMITATION AND FUTURE WORK

We establish the first rigorous connection between Gaussian scalar quantization and layerwise posttraining quantization at random initialization. Under Gaussian weights and an effective-rank condition on calibration activations, the normalized round-to-nearest reconstruction loss converges uniformly in scale to the Gaussian objective (Theorem 1); this condition holds with high probability for wide MLPs (Theorem 5). The objective has a unique, nondegenerate minimizer whose scaleinvariant curvature decreases with bit-width (Proposition 1; Section 4.2), explaining strong scale sensitivity at 2–3 bits and weaker sensitivity from 6 bits onward, consistent with LLM experiments. The analysis also shed light on why HIP improves quantization: most scale-selection rules target near-optimal scales for Gaussian weights, and HIP brings weights closer to this regime. We stress the fact that convergence is not guarantee but Gaussianity of the weights alone for iterative PTQ. Finally, our experiments reveal that naively increasing the effective rank improves scale predictability but removes low-eigenvalue directions that GPTQ exploits for error compensation.

Limitations & Future works. Our theory concerns randomly initialized networks and does not model training or its effects on weight distributions and activation geometry. We therefore do not establish whether the Gaussian and effective-rank assumptions hold after training; the experiments on pretrained LLMs provide empirical evidence beyond our theoretical guarantees. Moreover, the analysis is specific to round-to-nearest (RTN) quantization and does not cover other strategies, including error-compensation methods such as GPTQ. Extending the theory to trained networks and more general quantization methods remains an open direction. In our experiments HIP makes weight distributions more Gaussian, improving alignment with (H1) and GPTQ performance, whereas the transformations we tested to increase the effective rank, motivated by (H2), degrade performance when combined with HIP. This calls for preprocessing that improves Gaussianity and effective rank jointly without sacrificing GPTQ's error compensation. On the theoretical side, extending our analysis to trained weights and to compensated methods such as GPTQ would clarify how scale selection, spectral structure and error compensation interact. Finally, the approximately exponential decay of relative-scale sensitivity in the Gaussian model provides a quantitative reference for future PTQ methods: a natural goal is lower sensitivity at low precision, under matched bit-widths and loss normalization, without loss of reconstruction quality or end-to-end performance.

## AI USE STATEMENT

In this work, we used generative AI tools for polishing code, improving sentence clarity, and refining grammar. Moreover, discussion with AI has been used to provide insight on how to prove positiveness and decreasing behaviour of (ii) and (iii) in Proposition 1. The conversation with AI tools has not produced the proof but just the idea, that has been carefully written by the authors. Therefore, we have not used generative AI tools for creating results and proof in a copy and paste manner. We have carefully reviewed all AI-assisted work: in particular, the LLM-generated part of the code used in the experiments was verified and tested for correctness. We take responsibility for the final content of this work, including text and claims or artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This work focuses on the theoretical analysis of scale sensitivity and quantization objective in machine learning model quantization and does not involve experiments on human subjects, sensitive personal data, or applications with direct societal risks. The datasets referenced are publicly available, and no private or restricted data was used. Potential ethical concerns related to misuse are minimal, as the contributions are mainly theoretical.

## REPRODUCIBILITY STATEMENT

We have taken multiple steps to ensure reproducibility of our results. All theoretical claims are accompanied by rigorous proofs, presented in detail in the appendix. Assumptions underlying the theorems are explicitly stated, and definitions are given in full to allow independent verification. The code used for the experiment is provided as anonymous supplementary material at https : //anonymous.4open.science/r/SOQ-345F.

## ACKNOWLEDGMENTS

Jonas von Berg, Massimiliano Datres and Gitta Kutyniok acknowledge support by the project "Next Generation AI Computing (gAIn)," funded by the Bavarian Ministry of Science and the Arts and the Saxon Ministry for Science, Culture, and Tourism as well as by the Hightech Agenda Bavaria.

Jonas von Berg, Carlo Kneißl and Gitta Kutyniok are also grateful for partial support from the Konrad Zuse School of Excellence in Reliable AI (DAAD).

Additionally, Jonas von Berg, Massimiliano Datres, Carlo Kneißl and Gitta Kutyniok acknowledge support by the Munich Center for Machine Learning (MCML).

Carlo Kneissl and Gitta Kutyniok acknowledge support by the project "Genius Robot" (01IS24083), funded by the Federal Ministry of Education and Research (BMBF), as well as the ONE Munich Strategy Forum (LMU Munich, TU Munich, and the Bavarian Ministery for Science and Art)

Gitta Kutyniok furthermore acknowledges support by the German Research Foundation under Grants DFG-SPP-2298, KU 1446/31-1 and KU 1446/32-1, and by the Bavarian Ministry for Digital Affairs.

## REFERENCES

Juan Amboage, Pablo Monteagudo-Lago, Ian Colbert, Giuseppe Franco, and Nicholas Fraser. Optimal post-training quantization scales and where to find them. arXiv preprint arXiv:2606.10890, 2026.

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. Quarot: Outlier-free 4-bit inference in rotated llms. Advances in Neural Information Processing Systems, 37:100213–100240, 2024.

Ron Banner, Yury Nahshan, Elad Hoffer, and Daniel Soudry. Aciq: Analytical clipping for integer quantization of neural networks. 2018.

Ron Banner, Yury Nahshan, and Daniel Soudry. Post training 4-bit quantization of convolutional networks for rapid-deployment. Advances in neural information processing systems, 32, 2019.

Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher M De Sa. Quip: 2-bit quantization of large language models with guarantees. Advances in neural information processing systems, 36:4396–4429, 2023.

Jiale Chen, Yalda Shabanzadeh, Elvir Crnčević, Torsten Hoefler, and Dan Alistarh. The geometry of llm quantization: Gptq as babai's nearest plane algorithm. In International Conference on Learning Representations, volume 2026, pp. 122653–122695, 2026.

Tim Dettmers and Luke Zettlemoyer. The case for 4-bit precision: k-bit inference scaling laws. In International Conference on Machine Learning, pp. 7750–7774. PMLR, 2023.

Abhimanyu Dubey, Abhinav Jauhri, and et al. The llama 3 herd of models. CoRR, abs/2407.21783, 2024.URLhttps://doi.org/10.48550/arXiv.2407.21783.

Elias Frantar and Dan Alistarh. Optimal brain compression: A framework for accurate post-training quantization and pruning. Advances in Neural Information Processing Systems, 35:4475–4488, 2022.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022.

Babak Hassibi and David Stork. Second order derivatives for network pruning: Optimal brain surgeon. Advances in neural information processing systems, 5, 1992.

Béatrice Laurent and Pascal Massart. Adaptive estimation of a quadratic functional by model selection. Annals of Statistics, 28(5):1302–1338, 2000.

Shuaiting Li, Juncan Deng, Kedong Xu, Rongtao Deng, Hong Gu, Minghan Jiang, Haibin Shen, and Kejie Huang. Rethinking residual errors in compensation-based llm quantization. arXiv preprint arXiv:2604.07955, 2026.

Yuhang Li, Ruihao Gong, Xu Tan, Yang Yang, Peng Hu, Qi Zhang, Fengwei Yu, Wei Wang, and Shi Gu. Brecq: Pushing the limit of post-training quantization by block reconstruction. arXiv preprint arXiv:2102.05426, 2021.

Yuhang Li, Ruokai Yin, Donghyun Lee, Shiting Xiao, and Priyadarshini Panda. Gptaq: Efficient finetuning-free quantization for asymmetric calibration. arXiv preprint arXiv:2504.02692, 2025.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. Proceedings of machine learning and systems, 6:87–100, 2024.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. Spinquant: Llm quantization with learned rotations. In International Conference on Learning Representations, volume 2025, pp. 92009–92032, 2025.

Shuming Ma, Hongyu Wang, Shaohan Huang, Xingxing Zhang, Ying Hu, Ting Song, Yan Xia, and Furu Wei. Bitnet b1. 58 2b4t technical report. arXiv preprint arXiv:2504.12285, 2025.

Joel Max. Quantizing for minimum distortion. IRE Transactions on Information Theory, 6(1):7–12, 1960.

Sangsin Na and David L Neuhoff. On the convexity of the mse distortion of symmetric uniform scalar quantization. IEEE Transactions on Information Theory, 64(4):2626–2638, 2017.

Sangsin Na and David L Neuhoff. Monotonicity of step sizes of mse-optimal symmetric uniform scalar quantizers. IEEE Transactions on Information Theory, 65(3):1782–1792, 2018.

Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. Up or down? adaptive rounding for post-training quantization. In International conference on machine learning, pp. 7197–7206. PMLR, 2020.

Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart Van Baalen, and Tijmen Blankevoort. A white paper on neural network quantization. arXiv preprint arXiv:2106.08295, 2021.

Yury Nahshan, Brian Chmiel, Chaim Baskin, Evgenii Zheltonozhskii, Ron Banner, Alex M Bronstein, and Avi Mendelson. Loss aware post-training quantization. Machine Learning, 110(11): 3245–3262, 2021.

Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Gao Peng, Yu Qiao, and Ping Luo. Omniquant: Omnidirectionally calibrated quantization for large language models. In International Conference on Learning Representations, volume 2024, pp. 45472–45496, 2024.

Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. Quip#: Even better llm quantization with hadamard incoherence and lattice codebooks. Proceedings of machine learning research, 235:48630, 2024.

Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2018.

Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge university press, 2019

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International conference on machine learning, pp. 38087–38099. PMLR, 2023.

Aozhong Zhang, Naigang Wang, Yanxia Deng, Xin Li, Zi Yang, and Penghang Yin. Magr: Weight magnitude reduction for enhancing post-training quantization. Advances in neural information processing systems, 37:85109–85130, 2024.

Shihao Zhang, Haoyu Zhang, Ian Colbert, and Rayan Saab. Qronos: Correcting the past by shaping the future... in post-training quantization. In International Conference on Learning Representations, volume 2026, pp. 87775–87798, 2026.

## APPENDIX

A Theory 14   
A.1 Known results 14   
A.2 Proofs of Theorem 1 17   
A.3 Proof of Corollary 1 22   
A.4 Verifying (H2) for Multi-Layer Perceptrons at initialization 23   
A.5 Proof of Proposition 1 29   
A.5.1 Computation of κ(m) for m > 0 integer 38   
A.5.2 Why relative curvature κ? 39   
A.6 Technical Lemmas 40   
B Experiments 41   
B.1 Experimental Details 42   
B.1.1 Models 42   
B.1.2 Quantization setup 42   
B.1.3 Scale-selection rules 43   
B.1.4 Compute resources and software 44   
B.2 Global scale-sensitivity 44   
B.2.1 Sensitivity to scale selection with HIP preprocessing 45   
B.2.2 Sensitivity to scale selection with GPTQ extensions 46   
B.2.3 Scale sensitivity without min-max 47   
B.2.4 End-to-end sensitivity to controlled scale perturbations 47   
B.2.5 Calibration-Set Size 48   
B.3 Zero-shot evaluation 48   
B.4 Preprocessing and effective rank 49   
B.5 Quantization error landscapes 53   
B.5.1 Gaussian landscapes and convergence 53   
B.6 Local Reconstruction Sensitivity with HIP 53

## A THEORY

## A.1 KNOWN RESULTS

In this section we collect the probabilistic tools used throughout the paper. Throughout the paper the sub-gaussian and sub-exponential norms are understood in the following Orlicz sense, which is the one used in Vershynin (2018). Most of the results appearing in this section are taken from Vershynin (2018).

Definition 2. For a real random variable Y we set

$$
\| Y \| _ { \psi _ { 2 } } : = \operatorname* { i n f } \left\{ u > 0 : \mathbb { E } \left[ e ^ { Y ^ { 2 } / u ^ { 2 } } \right] \leq 2 \right\} ,
$$

with the convention inf $\varnothing = + \infty ,$ , and we call Y sub-gaussian $i f \| Y \| _ { \psi _ { 2 } } < \infty .$

Definition 3. For a real random variable Y we set

$$
\Vert Y \Vert _ { \psi _ { 1 } } : = \operatorname* { i n f } \left\{ u > 0 : \mathbb { E } \left[ e ^ { | Y | / u } \right] \leq 2 \right\} ,
$$

with the convention inf $\varnothing = + \infty ,$ , and we call Y sub-exponential $i f \| Y \| _ { \psi _ { 1 } } < \infty$

Gaussian random variables are sub-gaussian (Vershynin, 2018, Ex. 2.5.8); since the value of their ψ2-norm enters the constants of our bounds, we compute it exactly

Lemma 1. $I f Y \sim { \mathcal { N } } ( 0 , \tau ^ { 2 } )$ with $\tau > 0 ,$ then $\| Y \| _ { \psi _ { 2 } } = \sqrt { 8 / 3 } \tau$ . Moreover, $\| a Y \| _ { \psi _ { 2 } } = | a | \| Y \| _ { \psi _ { 2 } }$ for every $a \in \mathbb { R }$

Proof. For $u > \sqrt { 2 } \tau$ one has $\mathbb { E } [ e ^ { Y ^ { 2 } / u ^ { 2 } } ] = ( 1 - 2 \tau ^ { 2 } / u ^ { 2 } ) ^ { - 1 / 2 }$ , and $\mathbb { E } [ e ^ { Y ^ { 2 } / u ^ { 2 } } ] = + \infty$ otherwise. Hence $\mathbb { E } [ e ^ { Y ^ { 2 } / u ^ { 2 } } ] \le 2$ if and only if $1 - 2 \tau ^ { 2 } / u ^ { 2 } \geq 1 / 4$ i.e. $u ^ { 2 } \geq \frac { 8 } { 3 } \tau ^ { 2 }$ . The homogeneity is immediate from the definition. □

The square of a sub-gaussian random variable is sub-exponential, with an exact relation between the two norms.

Lemma 2. A random variable X is sub-gaussian if and only $i f X ^ { 2 }$ is sub-exponential. Moreover,

$$
\| X ^ { 2 } \| _ { \psi _ { 1 } } = \| X \| _ { \psi _ { 2 } } ^ { 2 } .
$$

The following lemma shows that centering does not increase the sub-exponential norm by more than a constant factor, which we make explicit.

Lemma 3. Let X be a sub-exponential random variable. Then $X - \mathbb { E } X$ is sub-exponential as well, and

$$
\| X - \mathbb { E } X \| _ { \psi _ { 1 } } \leq \left( 1 + \frac { 1 } { \log 2 } \right) \| X \| _ { \psi _ { 1 } } \leq 3 \| X \| _ { \psi _ { 1 } } .
$$

Proof. Since $\| \cdot \| _ { \psi _ { 1 } }$ is a norm, the triangle inequality gives $\| X - \mathbb { E } X \| _ { \psi _ { 1 } } \leq \| X \| _ { \psi _ { 1 } } + \| \mathbb { E } X \| _ { \psi _ { 1 } }$ For a deterministic $a \in \mathbb { R }$ we have $e ^ { | a | / t } \leq 2$ if and only if $t \geq | a | / \log 2$ hence $\| a \| _ { \psi _ { 1 } } = | a | / \log 2 .$ It remains to bound |EX| by $\| X \| _ { \psi _ { 1 } }$ . Let $t > \| X \| _ { \psi _ { 1 } }$ , so that $\mathbb { E } e ^ { | X | / t } \le 2$ Using $e ^ { x } \geq 1 + x$ we get $2 \geq \mathbb { E } e ^ { | X | / t } \geq 1 + \mathbb { E } | X | / t , { \mathrm { i . e . ~ } } \mathbb { E } | X | \leq t$ Letting $t \downarrow \parallel X \parallel _ { \psi _ { 1 } }$ yields $| \mathbb { E } X | \leq \mathbb { E } | X | \leq \| X \| _ { \psi _ { 1 } }$ Combining the three estimates,

$$
\| X - \mathbb { E } X \| _ { \psi _ { 1 } } \leq \| X \| _ { \psi _ { 1 } } + \frac { | \mathbb { E } X | } { \log 2 } \leq \Big ( 1 + \frac { 1 } { \log 2 } \Big ) \| X \| _ { \psi _ { 1 } } ,
$$

and $1 + 1 / \log 2 .$

We shall use the following concentration inequalities for sums and quadratic forms of independent sub-gaussian or sub-exponential random variables.

Theorem 2 (Bernstein's inequality). Let $X _ { 1 } , \ldots , X _ { N }$ be independent, mean-zero, sub-exponential random variables, and let $K \mathbf { \bar { : } } = \operatorname { m a x } _ { i } \| X _ { i } \| _ { \psi _ { 1 } }$ . Then, for every $t \geq 0 ,$

$$
\mathbb { P } \left( \Big | \sum _ { i = 1 } ^ { N } X _ { i } \Big | \ge t \right) \ \le \ 2 \exp \left[ - c \operatorname* { m i n } \left( \frac { t ^ { 2 } } { K ^ { 2 } N } , \frac { t } { K } \right) \right] .
$$

Theorem 3 (Hanson-Wright inequality). Let $X = ( X _ { 1 } , \ldots , X _ { n } ) \in \mathbb { R } ^ { n }$ be a random vector with independent components satisfying $\mathbb { E } [ \mathrm { \bar { X } } _ { i } ] = 0$ and $\| X _ { i } \| _ { \psi _ { 2 } } \le K$ for every $i \in [ n ]$ , and let $A \in$ $\mathbb { R } ^ { n \times n }$ . Then, for every $t \geq 0$

$$
\mathbb { P } \left( \left| X ^ { \top } A X - \mathbb { E } \left[ X ^ { \top } A X \right] \right| > t \right) \leq 2 \exp \left( - c \mathrm { m i n } \left\{ \frac { t ^ { 2 } } { K ^ { 4 } \| A \| _ { F } ^ { 2 } } , \frac { t } { K ^ { 2 } \| A \| _ { o p } } \right\} \right) ,
$$

where $\| A \| _ { F }$ and $\| A \| _ { o p }$ denote the Frobenius and the operator norm of A.

We recall some tail bounds for the sup-norm and the Euclidean norm of a standard Gaussian vector. The first is a union bound over the standard Gaussian tail $\mathbb { P } ( | g | > t ) \le 2 e ^ { - t ^ { 2 } / 2 } , t \ge 0$ (Vershynin, 2018, Prop. 2.1.2); the second is the sharp $\chi ^ { 2 }$ tail bound of Laurent and Massart (Laurent & Massart, 2000, Lemma 1).

Lemma 4. Let $X \sim \mathcal { N } ( 0 , \operatorname { I d } _ { d } )$ . Then, for every $d \geq 2 ,$

$$
\begin{array} { r } { \mathbb { P } \left( \| X \| _ { \infty } > \log d \right) \leq 2 d e ^ { - \log ^ { 2 } d / 2 } , } \end{array}
$$

and, for every $x > 0 ,$

$$
\begin{array} { r } { \mathbb P \left( \| X \| _ { 2 } ^ { 2 } \geq d + 2 \sqrt { d x } + 2 x \right) \leq e ^ { - x } , \qquad \mathbb P \left( \| X \| _ { 2 } ^ { 2 } \leq d - 2 \sqrt { d x } \right) \leq e ^ { - x } . } \end{array}
$$

In particular, choosing $x = d / 1 6 ,$ each of the two inequalities $\begin{array} { r } { \frac 1 2 d \leq \| X \| _ { 2 } ^ { 2 } \leq 2 d } \end{array}$ fails with probability at most $e ^ { - d / 1 6 }$

Proof. For the first claim, by a union bound and the Gaussian tail bound,

$$
\mathbb { P } \left( \| X \| _ { \infty } > \log d \right) \le \sum _ { i = 1 } ^ { d } \mathbb { P } \left( | X _ { i } | > \log d \right) \le 2 d e ^ { - \log ^ { 2 } d / 2 } .
$$

The two-sided bound on $\| X \| _ { 2 } ^ { 2 } \sim \chi _ { d } ^ { 2 }$ is in (Laurent & Massart, 2000, Lemma 1). For the last claim, with $x = d / 1 6$ one has $d - 2 \sqrt { d x } = d / 2$ and $\begin{array} { r } { d + 2 \sqrt { d x } + 2 x = \frac { 1 3 } { 8 } d \leq 2 d . } \end{array}$ □

$\mathrm { N e x t , }$ we recall that the operator norm of an $m \times n$ random matrix with independent sub-gaussian entries is of order ${ \sqrt { m } } + { \sqrt { n } }$ with high probability.

Theorem 4. Let A be an m × n random matrix with independent, mean-zero, sub-gaussian entries $A _ { i j }$ , and let $K : = \operatorname* { m a x } _ { i , j } \| A _ { i j } \| _ { \psi _ { 2 } }$ . Then, for any $t > 0 ,$

$$
\| A \| _ { o p } \leq C K \left( { \sqrt { m } } + { \sqrt { n } } + t \right)
$$

with probability at least $1 - 2 \exp ( - t ^ { 2 } )$

Finally, we recall the Gaussian concentration inequality for Lipschitz functions. These references state the inequality for Euclidean-Lipschitz functions of a standard Gaussian vector, but, since the Frobenius norm is the Euclidean norm on $\mathbb { R } ^ { m \times n } \cong \mathbb { R } ^ { m n }$ and the entries of G are i.i.d. $\mathcal { N } ( 0 , 1 )$ , the matrix version below is the same statement with the same constants

Lemma 5 (Theorem 2.26 in Wainwright (2019)). Let $F \colon \mathbb { R } ^ { m \times n } $ R be K-Lipschitz with respect to the Frobenius norm, i.e. $| F ( A ) - F ( { \bar { B } } ) | \leq K \| A - B \| _ { F } .$ for all $A , B \in \mathbb { R } ^ { m \times n }$ , and let $G \in \mathbb { R } ^ { \tilde { m } \times n }$ be a random matrix with i.i.d. $\mathcal { N } ( 0 , 1 )$ entries. Then, for every $t \geq 0$

$$
\mathbb { P } \left\{ | F ( G ) - \mathbb { E } F ( G ) | > t \right\} \leq 2 \exp \left( - \frac { t ^ { 2 } } { 2 K ^ { 2 } } \right) .
$$

Finally, we recall a known result on randomly initialized MLPs, which says that, with high probability over the weights, the Euclidean norm of each hidden layer grows at most like the square root of the width.

Lemma 6. Under $( H I ) , ( H 3 )$ and (H4), for all $x \in \mathcal { X }$ and $\ell \in [ L - 1 ]$ , there exists a constant $C > 0$ not dependent on d such that

$$
\begin{array} { r } { \mathbb { P } _ { \vartheta } \left( \| \alpha ^ { ( \ell ) } ( x , \vartheta ^ { ( \ell ) } ) \| _ { 2 } \geq C \sqrt { d ^ { \prime } } \right) \leq 2 \ell e ^ { - d ^ { \prime } } . } \end{array}
$$

Proof. We proceed by induction. We start by the base case $\ell = 1$ . We notice that, using Lemma 4 and Theorem 4, for $x \in \mathcal { X }$ , it holds

$$
\begin{array} { l } { \displaystyle \left\| \alpha ^ { ( 1 ) } \left( x , \vartheta ^ { ( 1 ) } \right) \right\| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { d _ { 1 } } \sigma \left( \frac { 1 } { \sqrt { d _ { 0 } } } \vartheta _ { i \colon } ^ { ( 1 ) } x \right) ^ { 2 } } \\ { \displaystyle \qquad \leq L _ { \sigma } ^ { 2 } \sum _ { i = 1 } ^ { d _ { 1 } } \left( \frac { 1 } { \sqrt { d _ { 0 } } } \vartheta _ { i \colon } ^ { ( 1 ) } x \right) ^ { 2 } } \\ { \displaystyle \qquad \leq L _ { \sigma } ^ { 2 } \left\| \frac { 1 } { \sqrt { d _ { 0 } } } \vartheta ^ { ( 1 ) } \right\| _ { \sigma p } ^ { 2 } \| x \| _ { 2 } ^ { 2 } \leq L _ { \sigma } ^ { 2 } \left\| \vartheta ^ { ( 1 ) } \right\| _ { \rho p } ^ { 2 } \| x \| _ { 2 } ^ { 2 } . } \end{array}
$$

Since the entries of $\vartheta ^ { ( 1 ) }$ are independent, centered and, by Lemma 1 applied with $\tau ^ { 2 } ~ = ~ 1$ $\| \vartheta _ { i j } ^ { ( 1 ) } \| _ { \psi _ { 2 } } = \sqrt { 8 / 3 }$ for all $i \in [ d _ { 1 } ] , j \in [ d _ { 0 } ]$ . Combining Theorem 4 with $K = \sqrt { 8 / 3 }$ and $t = \sqrt { d _ { 1 } }$ and Lemma 4, we obtain that there exists a constant $C > 0$ not dependent on d such that

$$
\begin{array} { r } { \mathbb { P } _ { \vartheta } \left( \left\| \alpha ^ { ( 1 ) } \left( x , \vartheta ^ { ( 1 ) } \right) \right\| _ { 2 } \geq C \sqrt { d } \right) \leq 2 e ^ { - d } . } \end{array}
$$

Inductively, assume that, for all $x \in \mathcal { X }$ , there exists constants $C , c > 0$ not dependent on d such that $\left\| \alpha ^ { ( \ell - 1 ) } \left( \dot { x } , \vartheta ^ { ( \ell - 1 ) } \right) \right\| _ { 2 } \leq C \sqrt { d }$ with probability at least $1 - 2 ( L - 1 ) e ^ { - c d }$ . Then, using Theorem 4with $K = \sqrt { 8 / 3 }$ and $t = \sqrt { d _ { \ell } }$ as in the base case, together with the independence of $\vartheta ^ { ( \ell ) }$ from $\alpha ^ { ( \ell - 1 ) }$ , we conclude that

$$
\begin{array} { r l } & { \mathbb { P } _ { \vartheta } \left( \left\| \alpha ^ { ( \ell ) } \left( x , \vartheta ^ { ( \ell ) } \right) \right\| _ { 2 } \geq C \sqrt { d } \right) } \\ & { \qquad \leq \mathbb { P } _ { \vartheta } \left( L _ { \sigma } \left\| \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \right\| _ { \sigma p } \| \alpha ^ { ( \ell - 1 ) } \| _ { 2 } \geq C \sqrt { d } \right) } \\ & { \qquad \leq \mathbb { P } _ { \vartheta } \left( \left\| \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \right\| _ { \sigma } \geq C \right) + 2 ( L - 1 ) e ^ { - d ^ { \prime } } } \\ & { \qquad \leq 2 L e ^ { - d } , } \end{array}
$$

for some constant $C > 0$ not dependent on $d .$

## A.2 PROOFS OF THEOREM 1

To improve readability, we sometimes omit the constants which do not depend on d and $N _ { c }$ form the formulas, since they don't affect the asymptotic behavior. We define, for each $\ell \in \ [ L ]$ , the quantization residual $r ^ { ( \ell ) } : ( 0 , \infty ) \times \mathbb { R } $ R as

$$
r \left( s , t \right) : = t - s \left\lfloor t / s \right\rceil _ { B } .\tag{5}
$$

Whenever r is applied to a multidimensional object, it acts componentwise. In the next lemma, we show some basic properties of the quantized residual evaluated at initialization.

Lemma 7. Let $\vartheta \in \mathbb { R } ^ { d }$ with $\vartheta _ { i } \stackrel { i i d } { \sim } \mathcal { N } ( 0 , 1 )$ . Then, for all $s > 0 _ { i }$ the residual satisfies the following properties

1. $r ( s , \vartheta _ { i } )$ is odd, i.e. $r ( s , \vartheta _ { i } ) = - r ( s , - \vartheta _ { i } ) , f o r a l l i \in [ d ] .$

2. $| r ( s , \vartheta _ { i } ) | \leq | \vartheta _ { i } | f o r a l l i \in [ d ] .$

3. For any fixed $x \in \mathbb { R } .$ the function s → $r ^ { 2 } ( s , x )$ is $2 M _ { B } | x |$ -Lipschitz, continuous.

$4 . \ r ( s , \vartheta _ { i } ) , \ i \ \in \ [ d ]$ , are independent, centered and sub-Gaussian; more precisely, for all $i \in [ d ]$ , it holds

$$
\mathbb { E } _ { \vartheta } \left[ r \left( s , \vartheta _ { i } \right) \right] = 0 , \qquad \left\| r \left( s , \vartheta _ { i } \right) \right\| _ { \psi _ { 2 } } \leq \sqrt { \frac { 8 } { 3 } } .
$$

Proof. We prove each property separately.

1. Property 1. follows immediately by oddness and symmetry of $\lfloor \cdot \rceil _ { B }$ . Indeed, for all $i \in [ d ]$ we have

$$
r ( s , - \vartheta _ { i } ) = - \vartheta _ { i } - \Pi _ { s , M _ { B } } ( - \vartheta _ { i } ) = - \vartheta _ { i } + \Pi _ { s , M _ { B } } ( \vartheta _ { i } ) = - r ( s , \vartheta _ { i } ) .
$$

2. For all $i \in [ d ]$ , it holds

$$
\left| r ( s , \vartheta _ { i } ) \right| = \left| \vartheta _ { i } - \Pi _ { s , M _ { B } } ( \vartheta _ { i } ) \right| \stackrel { ( * ) } { = } \mathrm { d i s t } \big ( \vartheta _ { i } , s \left\{ - M _ { B } , \ldots , M _ { B } \right\} \big ) \leq \left| \vartheta _ { i } \right| ,\tag{6}
$$

where $( * )$ follows from the definition (1) of $\lfloor \cdot \rceil _ { B } ,$ while the last inequality holds because 0 is a quantization level, i.e. $0 \in s \{ - M _ { B } , . . . , \mathbf { \bar { M } } _ { B } \}$

3. Let us assume without loss of generality that $x > 0$ . The case with $x \ < \ 0$ follows by oddness of $\Pi _ { s , M _ { B } }$ . Let us fix $n \in [ M _ { B } ]$ , and denote with $\textstyle \sigma _ { n } = { \frac { x } { n - { \frac { 1 } { 2 } } } }$ . On $\left( \sigma _ { n + 1 } , \sigma _ { n } \right]$ (with $\sigma _ { M _ { B } + 1 } : = 0 )$ we have $\lfloor x / s \rceil _ { B } = n .$ hence $r ^ { 2 } ( s , x ) = ( x - n s ) ^ { \mathbf { \bar { 2 } } }$ there; on $\left( \sigma _ { n } , \sigma _ { n - 1 } \right]$ (with $\sigma _ { 0 } : = \infty )$ we have $\lfloor x / s \rceil _ { B } = n - 1 ,$ hence $r ^ { 2 } ( \dot { s , x } ) = \smash { \big ( x - ( n - 1 ) s \big ) ^ { 2 } }$ there. Both expressions are polynomials in $s ,$ SO $r ^ { 2 } ( \cdot , x )$ is continuous on each of the two halfopen intervals, and in particular left-continuous at $\sigma _ { n }$ . For right-continuity at $\sigma _ { n }$ we use $\begin{array} { r } { x = ( n - \frac { 1 } { 2 } ) \sigma _ { n } } \end{array}$ to compute

$$
\operatorname* { l i m } _ { s \to \sigma _ { n } ^ { + } } r ^ { 2 } ( s , x ) = \left( x - ( n - 1 ) \sigma _ { n } \right) ^ { 2 } = \left( { \frac { \sigma _ { n } } { 2 } } \right) ^ { 2 } = \left( x - n \sigma _ { n } \right) ^ { 2 } = r ^ { 2 } ( \sigma _ { n } , x ) .
$$

Hence $r ^ { 2 } ( \cdot , x )$ is continuous at every $\sigma _ { n }$ , and therefore on all of $( 0 , \infty )$ . On each interval $( \sigma _ { n + 1 } , \sigma _ { n } ) , r ^ { 2 } ( s , x )$ is differentiable, with derivative

$$
\left| { \frac { d } { d s } } r ^ { 2 } ( s , x ) \right| = 2 n { \big | } x - s n { \big | } \leq 2 M _ { B } | x | ,
$$

where we used $n \le M _ { B }$ and $| x - s n | = | r ( s , x ) | \leq | x |$ by Property 2. A continuous function on an interval that is ${ \dot { \mathcal { C } } } ^ { 1 }$ off finitely many points with derivative bounded by L is L-Lipschitz (apply the mean value theorem between consecutive breakpoints and sum), which gives the claim.

4. Independence follows from the independence of the components of $\vartheta$ and from the pointwise nature of $r ( s , \vartheta )$ . Since the Gaussian distribution is symmetric and $\Pi _ { s , M _ { B } }$ is odd, we have that

$$
\begin{array} { r } { \mathbb { E } _ { \pmb { \vartheta } } \left[ r ( s , \vartheta _ { i } ) \right] = \mathbb { E } _ { \pmb { \vartheta } } \left[ \vartheta _ { i } \right] - \mathbb { E } _ { \pmb { \vartheta } } \left[ \Pi _ { s , M _ { B } } ( \vartheta _ { i } ) \right] = 0 . } \end{array}
$$

From the point-wise bound in Property 2, we get $\mathbb { E } _ { \vartheta } \left[ e ^ { r ^ { 2 } ( s , \vartheta _ { i } ) / u ^ { 2 } } \right] \leq \mathbb { E } _ { \vartheta } \left[ e ^ { \vartheta _ { i } ^ { 2 } / u ^ { 2 } } \right]$ for every $u > 0$ , hence

$$
\begin{array} { r l r } {  { \| r ( s , \vartheta _ { i } ) \| _ { \psi _ { 2 } } = \operatorname* { i n f } \{ u > 0 : \mathbb { E } _ { \vartheta _ { i } } [ e ^ { r ^ { 2 } ( s , \vartheta _ { i } ) / u ^ { 2 } } ] \leq 2 \} } } \\ & { } & { \leq \operatorname* { i n f } \{ u > 0 : \mathbb { E } _ { \vartheta _ { i } } [ e ^ { \vartheta _ { i } ^ { 2 } / u ^ { 2 } } ] \leq 2 \} = \| \vartheta _ { i } \| _ { \psi _ { 2 } } = \sqrt { \frac { 8 } { 3 } } , } \end{array}
$$

where the last equality is Lemma 1 applied with $\tau ^ { 2 } = 1$ . As i was arbitrary, this concludes the proof.

For clarity, we recall here Theorem 1.

Theorem 1. Let $B \geq 2 , X \in \mathbb { R } ^ { d \times N _ { c } }$ be nonzero matrix satisfying (H2), and θ satisfying (H1). Then, for all $d \geq 2 ,$ it holds

$$
\mathbb { P } \left( \operatorname* { s u p } _ { s > 0 } | { \mathcal { E } } ( s ) - { \mathcal { E } } ^ { \infty } ( s ) | > { \frac { 2 + 1 2 M _ { B } } { \log d } } \right) \leq C M _ { B } d e ^ { - c \log ^ { 2 } d } ,
$$

where $c , C > 0$ are absolute constants independent on d.

Proof. Let us set $H : = X X ^ { \top } \in \mathbb { R } ^ { d \times d }$ and abbreviate $r _ { \mathrm { e f f } } : = r _ { \mathrm { e f f } } ( H )$ . We note that

$$
\begin{array} { l } { \displaystyle \mathcal { E } ( s ) = \frac { 1 } { \mathrm { T r } ( H ) } \sum _ { i = 1 } ^ { d } H _ { i i } \big ( \vartheta _ { i } - \Pi _ { s , M _ { B } } ( \vartheta _ { i } ) \big ) ^ { 2 } } \\ { \displaystyle + \frac { 1 } { \mathrm { T r } ( H ) } \sum _ { i = 1 } ^ { d } \sum _ { j \neq i } ^ { d } H _ { i j } \big ( \vartheta _ { i } - \Pi _ { s , M _ { B } } ( \vartheta _ { i } ) \big ) \big ( \vartheta _ { j } - \Pi _ { s , M _ { B } } ( \vartheta _ { j } ) \big ) } \\ { \displaystyle = D _ { B } ( s , \vartheta , X ) + O _ { B } ( s , \vartheta , X ) , } \end{array}\tag{7}
$$

where

$$
\begin{array} { l } { \displaystyle D _ { B } \big ( s , \vartheta , X \big ) : = \frac { 1 } { \mathrm { T r } ( H ) } \displaystyle \sum _ { i = 1 } ^ { d } H _ { i i } \big ( \vartheta _ { i } - \Pi _ { s , M _ { B } } \big ( \vartheta _ { i } \big ) \big ) ^ { 2 } } \\ { \displaystyle O _ { B } \big ( s , \vartheta , X \big ) : = \frac { 1 } { \mathrm { T r } ( H ) } \displaystyle \sum _ { i = 1 } ^ { d } \sum _ { j \neq i } ^ { d } H _ { i j } \big ( \vartheta _ { i } - \Pi _ { s , M _ { B } } \big ( \vartheta _ { i } \big ) \big ) \big ( \vartheta _ { j } - \Pi _ { s , M _ { B } } \big ( \vartheta _ { j } \big ) \big ) . } \end{array}
$$

Since the $\vartheta _ { i }$ are independent and, by Lemma 7, the residuals $r ( s , \vartheta _ { i } )$ are centered, we have $\mathbb { E } _ { \vartheta } [ O _ { B } ( s , \vartheta , X ) ] = 0$ . Hence, it holds

$$
\begin{array} { r l } & { \mathcal { E } ^ { \infty } ( s ) = \mathbb { E } _ { \vartheta } \big [ D _ { B } ( s , \vartheta , X ) \big ] } \\ & { \mathcal { E } ( s ) - \mathcal { E } ^ { \infty } ( s ) = D _ { B } ( s , \vartheta , X ) - \mathbb { E } _ { \vartheta } [ D _ { B } ( s , \vartheta , X ) ] + O _ { B } ( s , \vartheta , X ) . } \end{array}
$$

Let us start by controlling $D _ { B } ( s , \vartheta , X )$ . Let us fix $s > 0$ . We note that

$$
\mathbb { E } _ { \vartheta } [ D _ { B } ( s , \vartheta , X ) ] = \frac { 1 } { \mathrm { T r } ( H ) } \sum _ { i = 1 } ^ { d } H _ { i i } \mathbb { E } _ { \vartheta } \big [ r ( s , \vartheta _ { i } ) ^ { 2 } \big ] = \mathbb { E } _ { \vartheta \sim \mathcal { N } ( 0 , 1 ) } \big [ r ( s , \vartheta ) ^ { 2 } \big ] ,
$$

since the $\vartheta _ { i }$ are identically distributed. We note that H is positive semidefinite, so $0 \leq H _ { i i } \leq \| H \| _ { o p }$ for every $i \in [ d ]$ , and therefore

$$
\| \mathrm { d i a g } ( H ) \| _ { F } ^ { 2 } = \sum _ { i = 1 } ^ { d } H _ { i i } ^ { 2 } \leq \| H \| _ { o p } \mathrm { T r } ( H ) , \qquad \| \mathrm { d i a g } ( H ) \| _ { o p } \leq \| H \| _ { o p } .\tag{8}
$$

By Lemma 7 the coordinates of $r ( s , \vartheta )$ are independent, centered and sub-Gaussian with $\| r ( s , \vartheta _ { i } ) \| _ { \psi _ { 2 } } \leq K : = \sqrt { \frac { 8 } { 3 } }$ . Set $\varepsilon _ { d } : = \sqrt { \log ^ { 6 } d / r _ { \mathrm { e f f } } }$ . Therefore, by Theorem 3 applied to the matrix diag(H) with threshold $\operatorname { T r } ( H ) \varepsilon _ { d } .$ , and (8), for all $s > 0$ , we have that

$$
\begin{array} { r l } & { \mathbb { P } _ { \vartheta } \left( \left| D _ { B } ( s , \vartheta , X ) - \mathbb { E } _ { \vartheta } [ D _ { B } ( s , \vartheta , X ) ] \right| \geq \varepsilon _ { d } \right) } \\ & { \quad = \mathbb { P } _ { \vartheta } \left( \left| r ( s , \vartheta ) ^ { \top } \mathrm { d i a g } ( H ) r ( s , \vartheta ) - \mathbb { E } _ { \vartheta } \left[ r ( s , \vartheta ) ^ { \top } \mathrm { d i a g } ( H ) r ( s , \vartheta ) \right] \right| \geq \mathrm { T r } ( H ) \varepsilon _ { d } \right) } \\ & { \quad \leq 2 \exp \left( - c \operatorname* { m i n } \left\{ \frac { \mathrm { T r } ( H ) ^ { 2 } \varepsilon _ { d } ^ { 2 } } { K ^ { 4 } \left\| H \right\| _ { o p } \mathrm { T r } ( H ) } , \frac { \mathrm { T r } ( H ) \varepsilon _ { d } } { K ^ { 2 } \left\| H \right\| _ { o p } } \right\} \right) } \\ & { \quad = 2 \exp \left( - c \operatorname* { m i n } \left\{ \frac { 9 } { 6 4 } r _ { \mathrm { e f f } } \varepsilon _ { d } ^ { 2 } , \frac { 3 } { 8 } r _ { \mathrm { e f f } } \varepsilon _ { d } \right\} \right) } \\ & { \quad \leq 2 \exp \left( - c ^ { \prime } \operatorname* { m i n } \left\{ \log ^ { 6 } d , \sqrt { r _ { \mathrm { e f f } } \log ^ { 3 } d } \right\} \right) , } \end{array}\tag{9}
$$

where $c ^ { \prime } = 3 c / 6 4$ . Moreover, by Lemma 7, the function $s \mapsto r ( s , \vartheta ) ^ { 2 }$ is Lipschitz continuous, that is

$$
| r ( s , \vartheta ) ^ { 2 } - r ( t , \vartheta ) ^ { 2 } | \leq 2 M _ { B } | \vartheta | | t - s | .
$$

Therefore, using $\begin{array} { r } { H _ { i i } \ge 0 , \sum _ { i } H _ { i i } = \mathrm { T r } ( H ) } \end{array}$ and $\mathbb { E } _ { \vartheta \sim { \cal N } ( 0 , 1 ) } [ | \vartheta | ] = \sqrt { 2 / \pi }$ , we obtain that

$$
\vert D _ { B } ( s , \vartheta , X ) - D _ { B } ( t , \vartheta , X ) \vert \leq \vert s - t \vert \frac { 2 M _ { B } } { \mathrm { T r } ( H ) } \sum _ { i = 1 } ^ { d } H _ { i i } \vert \vartheta _ { i } \vert \leq 2 M _ { B } \Vert \vartheta \Vert _ { \infty } \vert s - t \vert ,\tag{10}
$$

$$
\begin{array} { r } { \vert \mathbb { E } _ { \vartheta } [ D _ { B } ( s , \vartheta , X ) ] - \mathbb { E } _ { \vartheta } [ D _ { B } ( t , \vartheta , X ) ] \vert \leq \vert s - t \vert 2 M _ { B } \mathbb { E } \vert \vartheta _ { 1 } \vert \leq 2 M _ { B } \vert s - t \vert . } \end{array}
$$

Let us denote, for simplicity, β := log d. Let us consider the partition of $( 0 , 2 \beta ]$ constructed via the grid $s _ { k } : = 2 k \frac { \beta } { d }$ for $k = 0 , \ldots , d ,$ whose mesh is $\begin{array} { r } { \eta : = s _ { k + 1 } - s _ { k } = \frac { 2 \beta } { d } } \end{array}$ . Let us now fix $k \in \{ 0 , \ldots , d - 1 \}$ and $s \in [ s _ { k } , s _ { k + 1 } ]$ . Then, using (10), it holds

$$
\begin{array} { r l } & { | D _ { B } ( s , \vartheta , X ) - \mathbb { E } _ { \vartheta } [ D _ { B } ( s , \vartheta , X ) ] | } \\ & { \quad \le | D _ { B } ( s , \vartheta , X ) - D _ { B } ( s _ { k + 1 } , \vartheta , X ) | + | D _ { B } ( s _ { k + 1 } , \vartheta , X ) - \mathbb { E } _ { \vartheta } [ D _ { B } ( s _ { k + 1 } , \vartheta , X ) ] | } \\ & { \quad \quad + | \mathbb { E } _ { \vartheta } [ D _ { B } ( s _ { k + 1 } , \vartheta , X ) ] - \mathbb { E } _ { \vartheta } [ D _ { B } ( s , \vartheta , X ) ] | } \\ & { \quad \le 2 \eta M _ { B } \| \vartheta \| _ { \infty } + 2 \eta M _ { B } + | D _ { B } ( s _ { k + 1 } , \vartheta , X ) - \mathbb { E } _ { \vartheta } [ D _ { B } ( s _ { k + 1 } , \vartheta , X ) ] | } \\ & { \quad = 4 M _ { B } \frac { \beta } { d } \| \vartheta \| _ { \infty } + 4 M _ { B } \frac { \beta } { d } + | D _ { B } ( s _ { k + 1 } , \vartheta , X ) - \mathbb { E } _ { \vartheta } [ D _ { B } ( s _ { k + 1 } , \vartheta , X ) ] | \ . } \end{array}
$$

Since, by Lemma $4 , \| \vartheta \| _ { \infty } \leq \beta$ with probability at least $1 - 2 d e ^ { - \log ^ { 2 } d / 2 }$ , and $\beta / d = \log d / d \leq \beta ^ { 2 }$ for $d \geq 3 .$ , on the event $\left\{ \| \vartheta \| _ { \infty } \leq \beta \right\}$ it holds

$$
\begin{array} { r l } {  { \operatorname* { s u p } _ { s \in ( 0 , 2 \beta ] } \vert D _ { B } ( s , \vartheta , X ) - \mathbb { E } _ { \vartheta } [ D _ { B } ( s , \vartheta , X ) ] \vert } } \\ & { \le 8 M _ { B } \frac { \log ^ { 2 } d } { d } + \operatorname* { m a x } _ { k \in [ d ] } \vert D _ { B } ( s _ { k } , \vartheta , X ) - \mathbb { E } _ { \vartheta } [ D _ { B } ( s _ { k } , \vartheta , X ) ] \vert . } \end{array}\tag{11}
$$

It remains to cover $s ~ > ~ 2 \beta$ Since $\Pi _ { s , M _ { B } } ( t ) ~ = ~ 0$ whenever $s \ > \ 2 | t |$ , on $\{ \| \vartheta \| _ { \infty } < \beta \}$ we have $r ( s , \vartheta _ { k } ) = \vartheta _ { k } = r ( 2 \beta , \vartheta _ { k } )$ for every $k \in [ d ]$ and every $s > 2 \beta$ , hence $D _ { B } ( s , \vartheta , X ) =$ $D _ { B } ( s _ { d } , \vartheta , X )$ for all $s > 2 \beta$ , where $s _ { d } = 2 \beta$ . Moreover, by Lemma $7 , 0 \leq r ( s , \vartheta _ { k } ) ^ { 2 } \leq \vartheta _ { k } ^ { 2 }$ for all $s > 0$ , and $r ( s , \vartheta _ { k } ) = \vartheta _ { k } = r ( 2 \beta , \vartheta _ { k } )$ when $| \vartheta _ { k } | < \beta$ and $s > 2 \beta$ , so that $| r ( s , \dot { \vartheta } _ { k } ) ^ { 2 } - r ( s _ { d } , \ddot { \vartheta } _ { k } ) ^ { 2 } | \leq$ $\vartheta _ { k } ^ { 2 } \mathbf { 1 } _ { \{ | \vartheta _ { k } | \geq \beta \} }$ . Therefore

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { s > 2 \beta } \left| \mathbb { E } _ { \boldsymbol { \vartheta } } \big [ D _ { B } \big ( s , \boldsymbol { \vartheta } , \boldsymbol { X } \big ) \big ] - \mathbb { E } _ { \boldsymbol { \vartheta } } \big [ D _ { B } \big ( s _ { d } , \boldsymbol { \vartheta } , \boldsymbol { X } \big ) \big ] \right| \leq \sum _ { k = 1 } ^ { d } \mathbb { E } \big [ \boldsymbol { \vartheta } _ { k } ^ { 2 } \mathbf { 1 } _ { \{ | \partial _ { k } | \geq \beta \} } \big ] = d \mathbb { E } _ { G \sim N ( 0 , 1 ) } \big [ G ^ { 2 } \mathbf { 1 } _ { \{ | G | \geq \log d \} } \big ] } & { } \\ { \quad } & { \leq 4 d \log d e ^ { - \log ^ { 2 } d / 2 } \leq 4 \frac { \log ^ { 2 } d } { d } } \end{array}\tag{12}
$$

for $d \geq 2 7$ , where we used ${ \mathbb E } [ G ^ { 2 } \mathbf { 1 } _ { \{ | G | \geq a \} } ] \leq 4 a e ^ { - a ^ { 2 } / 2 }$ for $a \geq 1$ with $a = \log d .$ and in the last step that $e ^ { 2 u - u ^ { 2 } / 2 } \leq u$ for u = log $d \geq 4 .$ Hence, on $\{ \| \vartheta \| _ { \infty } < \beta \}$ ，

$$
\begin{array} { r l } & { \underset { s > 0 } { \operatorname* { s u p } } [ D _ { B } ( s , \vartheta , X ) - \mathbb { E } _ { \phi } [ D _ { B } ( s , \vartheta , X ) ] ] } \\ & { \quad = \operatorname* { m a x } \Big \{ \underset { s \in ( 0 , 2 \beta ] } { \operatorname* { s u p } } \big \vert D _ { B } ( s , \vartheta , X ) - \mathbb { E } _ { \theta } [ D _ { B } ( s , \vartheta , X ) ] \big \vert , \underset { s > 2 \beta } { \operatorname* { s u p } } \big \vert D _ { B } ( s , \vartheta , X ) - \mathbb { E } _ { \vartheta } [ D _ { B } ( s , \vartheta , X ) ] \big \vert \Big \vert \Big \} } \\ & { \quad \le \operatorname* { m a x } \Big \{ \underset { s \in ( 0 , 2 \beta ] } { \operatorname* { s u p } } \big \vert D _ { B } ( s , \vartheta , X ) - \mathbb { E } _ { \vartheta } \big [ D _ { B } ( s , \vartheta , X ) \big ] \big \vert , \big \vert D _ { B } \big ( s _ { d } , \vartheta , X \big ) - \mathbb { E } _ { \vartheta } \big [ D _ { B } ( s _ { d } , \vartheta , X ) \big ] \big \vert } \\ & { \quad \quad + \underset { s > 2 \beta } { \operatorname* { s u p } } \big \vert \mathbb { E } _ { \vartheta } \big [ D _ { B } ( s _ { d } , \vartheta , X ) \big ] - \mathbb { E } _ { \vartheta } \big [ D _ { B } ( s , \vartheta , X ) \big ] \big \vert \Big \} } \\ & { \quad \overset { ( * ) } { \le } 1 2 M _ { B } \frac { \log ^ { 2 } d } { d } + \underset { k \in [ d ] } { \operatorname* { m a x } } \big \vert D _ { B } ( s _ { k } , \vartheta , X ) - \mathbb { E } _ { \vartheta } \big [ D _ { B } ( s _ { k } , \vartheta , X ) \big ] \big \vert , } \end{array}
$$

where in the second step we used that $D _ { B } ( s , \vartheta , X ) = D _ { B } ( s _ { d } , \vartheta , X )$ for $s > 2 \beta$ on $\{ \| \vartheta \| _ { \infty } < \beta \}$ and in (\*) we used (11) for the first term, (12) for the last term, $M _ { B } \geq 2 ,$ and the fact that sd is one of the grid points, so that $\begin{array} { r } { | D _ { B } ( s _ { d } , \vartheta , X ) - \mathbb { E } _ { \vartheta } [ D _ { B } ( s _ { d } , \vartheta , X ) ] | \le \operatorname* { m a x } _ { k \in [ d ] } | D _ { B } ( s _ { k } , \vartheta , X ) - \vartheta _ { k } ( \vartheta _ { k } , \vartheta , X ) | , } \end{array}$ $\mathbb { E } _ { \vartheta } [ D _ { B } ( s _ { k } , \vartheta , X ) ] |$ . Combining (9) and (11), we can conclude that

$$
\begin{array} { r l } {  { \mathbb { P } _ { \vartheta } ( \operatorname* { s u p } _ { s > 0 } \vert D _ { B } ( s , \vartheta , X ) - \mathbb { E } _ { \vartheta } [ D _ { B } ( s , \vartheta , X ) ] \vert > \varepsilon _ { d } + 1 2 M _ { B } \frac { \log ^ { 2 } d } { d } ) } } \\ & { \le \mathbb { P } _ { \vartheta } \big ( \vert \vartheta \vert \vert _ { \infty } > \beta \big ) + \displaystyle \sum _ { k = 1 } ^ { d } \mathbb { P } _ { \vartheta } ( \vert D _ { B } ( s _ { k } , \vartheta , X ) - \mathbb { E } _ { \vartheta } [ D _ { B } ( s _ { k } , \vartheta , X ) ] \vert > \varepsilon _ { d } ) } \\ & { \le 2 d e ^ { - \log ^ { 2 } d / 2 } + 2 d \exp ( - c ^ { \prime } \operatorname* { m i n } \{ \log ^ { 6 } d , \sqrt { r _ { \mathrm { e f f } } } \log ^ { 3 } d \} ) } \\ & { \le 4 d \exp ( - c ^ { \prime \prime } \operatorname* { m i n } \{ \log ^ { 2 } d , \sqrt { r _ { \mathrm { e f f } } } \log ^ { 3 } d \} ) , } \end{array}\tag{13}
$$

which concludes the bound on $D _ { B } ( s , \vartheta , X )$

It remains now to bound $O _ { B } ( s , \vartheta , X )$ . Let $\xi \in \{ - 1 , 1 \} ^ { d }$ have i.i.d. $\operatorname { R a d } ( 1 / 2 )$ coordinates, independent of v. Since the coordinates of θ are independent and symmetric, $\xi \odot \vartheta \overset { d } { = } \vartheta$ as random vectors; moreover $r ( s , \xi _ { i } \vartheta _ { i } ) = \xi _ { i } r ( s , \vartheta _ { i } )$ by item 1 of Lemma 7. Writing ${ \widetilde { H } } : = H - \mathrm { d i a g } ( H )$ , it holds, for every $s > 0$

$$
\begin{array} { l } { { \displaystyle \sum _ { i = 1 } ^ { d } \sum _ { j \neq i } ^ { d } H _ { i j } r ( s , \xi _ { i } \vartheta _ { i } ) r ( s , \xi _ { j } \vartheta _ { j } ) = \sum _ { i , j = 1 } ^ { d } \xi _ { i } r ( s , \vartheta _ { i } ) \widetilde H _ { i j } r ( s , \vartheta _ { j } ) \xi _ { j } } } \\ { { = \xi ^ { \top } \operatorname { d i a g } \bigl ( r ( s , \vartheta ) \bigr ) \widetilde H \operatorname { d i a g } \bigl ( r ( s , \vartheta ) \bigr ) \xi , } } \end{array}
$$

and consequently, as processes indexed by $s > 0$

$$
\bigl ( O _ { B } ( s , \vartheta , X ) \bigr ) _ { s > 0 } \stackrel { d } { = } \Bigl ( \frac { d } { \mathrm { T r } ( H ) } \xi ^ { \top } \mathrm { d i a g } ( r ( s , \vartheta ) ) \widetilde { H } \mathrm { d i a g } ( r ( s , \vartheta ) ) \xi \Bigr ) _ { s > 0 } ,
$$

so that it suffices to bound the supremum of the right-hand side. Since $\widetilde { H }$ is equal to H on the offdiagonal terms and equal to zero on the diagonal terms, and H is positive semidefinite with diagonal entries in $[ 0 , \| H \| _ { o p } ]$ , it follows that $\| \widetilde { H } \| _ { F } \le \| H \| _ { F }$ and, by Weyl's inequality, $\| \widetilde H \| _ { o p } \le \| H \| _ { o p } .$ For fixed $\vartheta \in \mathbb { R } ^ { d }$ , there are at most $M _ { B } d$ positive breakpoints,

$$
\sigma _ { i , k } = \frac { | \vartheta _ { i } | } { k + 1 / 2 } , \qquad 1 \le i \le d , \quad 0 \le k \le M _ { B } - 1 ,
$$

which we sort increasingly as $0 < \tau _ { 1 } < \cdot \cdot \cdot < \tau _ { K } , K \leq M _ { B } d _ { \cdot }$ and we set $\tau _ { 0 } : = 0$ . Let us define

$$
p ( s ) : = \sum _ { i , j = 1 } ^ { d } \xi _ { i } r ( s , \vartheta _ { i } ) \widetilde { H } _ { i j } r ( s , \vartheta _ { j } ) \xi _ { j } \qquad s > 0 .
$$

For each $s \in \left( \tau _ { k } , \tau _ { k + 1 } \right)$ the residual $r ( s , \vartheta _ { i } )$ is affine, and therefore

$$
p ( \boldsymbol { s } ) = p _ { k } ( \boldsymbol { s } ) : = \sum _ { i , j = 1 } ^ { d } \widetilde { H } _ { i j } \xi _ { i } \xi _ { j } \left( \vartheta _ { i } - \boldsymbol { s } n _ { i } \right) \left( \vartheta _ { j } - \boldsymbol { s } n _ { j } \right) ,
$$

where $n _ { i } : = \lfloor \vartheta _ { i } / s \rceil _ { B } ,$ which is constant for $s \in ( \tau _ { k } , \tau _ { k + 1 } )$ . Thus $p _ { k }$ is a polynomial of degree at most 2, defined on all of R, and it coincides with p on the open interval $\left( \tau _ { k } , \tau _ { k + 1 } \right)$ . Therefore, it holds

$$
\operatorname* { s u p } _ { s > 0 } \vert p ( s ) \vert = \operatorname* { m a x } \Big \{ \operatorname* { m a x } _ { k = 0 , \ldots , K - 1 } \operatorname* { s u p } _ { s \in ( \tau _ { k } , \tau _ { k + 1 } ) } \vert p _ { k } ( s ) \vert , \ \operatorname* { m a x } _ { k = 1 , \ldots , K } \vert p ( \tau _ { k } ) \vert , \ \operatorname* { s u p } _ { s > \tau _ { K } } \vert p ( s ) \vert \Big \} .
$$

For $s > \tau _ { K } = 2 \| \vartheta \| _ { \infty }$ every rounding vanishes, so $r ( s , \vartheta ) = \vartheta$ and $\begin{array} { r } { p ( s ) = \sum _ { i , j } \widetilde { H } _ { i j } \xi _ { i } \xi _ { j } \partial _ { i } \vartheta _ { j } } \end{array}$ is constant. By Lemma 10, applied to $p _ { k }$ on the closed interval $[ \tau _ { k } , \tau _ { k + 1 } ]$ , we have

$$
\operatorname* { s u p } _ { s \in ( \tau _ { k } , \tau _ { k + 1 } ) } | p _ { k } ( s ) | \leq 3 \operatorname* { m a x } \big \{ | p _ { k } ( \tau _ { k } ) | , | p _ { k } ( \tau _ { k + 1 } ) | , | p _ { k } ( ( \tau _ { k } + \tau _ { k + 1 } ) / 2 ) | \big \} .
$$

Therefore, we have that

$$
\begin{array} { r l } {  { \operatorname* { s u p } _ { s > 0 } \vert p ( s ) \vert } } \\ & { \le 3 \operatorname* { m a x } \Big \{ \operatorname* { m a x } _ { k < K } \operatorname* { m a x } \big \{ | p _ { k } ( \tau _ { k } ) | , | p _ { k } ( \tau _ { k + 1 } ) | , | p _ { k } ( \frac { \tau _ { k } + \tau _ { k + 1 } } { 2 } ) | \big \} , \operatorname* { m a x } _ { k \le K } \vert p ( \tau _ { k } ) \vert , \Big \vert \sum _ { i , i = 1 } ^ { d } \widetilde H _ { i j } \xi _ { i } \xi _ { j } \vartheta _ { i } \vartheta _ { j } \Big \} \Big \} . } \end{array}
$$

Each quantity inside the maximum is of the form $\xi ^ { \top } C \xi \mathrm { w i t h } C = \mathrm { d i a g } ( b ) \widetilde { H } \mathrm { d i a g } ( b )$ , where $b \in \mathbb { R } ^ { d }$ collects either the residuals $r ( s _ { 0 } , \vartheta _ { i } )$ at a node $s _ { 0 }$ , or their one-sided limits at a breakpoint, or $\vartheta _ { i }$ itself. Since 0 is a quantization level, $| r ( s , \vartheta _ { i } ) | \leq | \vartheta _ { i } | ,$ and at a breakpoint the one-sided limits equal $\pm \sigma _ { i , k } / 2$ with $\sigma _ { i , k } \leq 2 | \vartheta _ { i } |$ ; hence $| b _ { i } | \leq | \vartheta _ { i } |$ in all cases. Thus there are at most $4 M _ { B } d + 1$ matrices $C _ { j } \in \mathbb { R } ^ { d \times d }$ of the form $C _ { j } = \mathrm { d i a g } ( b ^ { ( j ) } )$ H di $\arg ( b ^ { ( j ) } )$ with $| b _ { i } ^ { ( j ) } | \leq | \vartheta _ { i } |$ , such that

$$
\operatorname* { s u p } _ { s > 0 } \left| \xi ^ { \top } \ \mathrm { d i a g } ( r ( s , \vartheta ) ) \widetilde { H } \ \mathrm { d i a g } ( r ( s , \vartheta ) ) \xi \right| \le 3 \operatorname* { m a x } _ { j \in [ 4 M _ { B } d + 1 ] } | \xi ^ { \top } C _ { j } \xi | .\tag{14}
$$

This includes rounding ties. Since $\widetilde { H }$ has zero diagonal, so does every $C _ { j }$ , and hence

$$
\mathbb { E } _ { \xi } [ \xi ^ { \top } C _ { j } \xi ] = \mathrm { T r } C _ { j } = 0 .
$$

On the event $\{ \| \vartheta \| _ { \infty } \leq \beta \}$ , which has probability at least $1 - 2 d e ^ { - \log ^ { 2 } d / 2 }$ , using $| b _ { i } ^ { ( j ) } | \leq \| \vartheta \| _ { \infty } \leq$ $\beta ,$ the submultiplicativity of the operator norm and $\| H \| _ { F } ^ { 2 } \le \| H \| _ { o p } \operatorname { T r } ( H )$ , it holds that

$$
\| C _ { j } \| _ { o p } \leq \| \vartheta \| _ { \infty } ^ { 2 } \| \widetilde { H } \| _ { o p } \leq \beta ^ { 2 } \| H \| _ { o p } = \log ^ { 2 } d \| H \| _ { o p } ,
$$

$$
\begin{array} { r } { \| C _ { j } \| _ { F } ^ { 2 } \leq \| \vartheta \| _ { \infty } ^ { 4 } \| \widetilde { H } \| _ { F } ^ { 2 } \leq \beta ^ { 4 } \| H \| _ { o p } \operatorname { T r } ( H ) = \log ^ { 4 } d \| H \| _ { o p } \operatorname { T r } ( H ) . } \end{array}
$$

We note that the $\xi _ { i }$ are independent, centered, with $\| \xi _ { i } \| _ { \psi _ { 2 } }$ an absolute constant. Hence, on $\{ \| \vartheta \| _ { \infty } \leq$ $\beta \}$ , by Theorem 3 applied conditionally on θ, (14) and a union bound, it holds

$$
\begin{array} { r l } & { \mathbb { P } _ { \xi } \Big ( \underset { s  0 } { \operatorname* { s u p } } \frac { 1 } { \operatorname { T r } ( H ) } \big | \xi ^ { \top } \operatorname { d i a g } ( r ( s , \vartheta ) ) \widetilde { H } \operatorname { d i a g } ( r ( s , \vartheta ) ) \xi \big | > \varepsilon _ { d } \Big ) } \\ & { \quad \leq \displaystyle \sum _ { j = 1 } ^ { 4 M _ { d } \varkappa 2 + 1 } \mathbb { P } _ { \xi } \Big ( \big | \xi ^ { \top } C _ { j } \xi \big | > \frac { \operatorname { T r } ( H ) } { 3 } \varepsilon _ { d } \Big ) } \\ & { \quad \leq 2 ( 4 M _ { B } d + 1 ) \exp \Big ( - { \varepsilon } \operatorname* { m i n } \{ \frac { \operatorname { T r } ( H ) ^ { 2 } \varepsilon _ { d } ^ { 2 } / 9 } { \log ^ { 4 } ( | H | | _ { \infty } \cap { \Gamma ( H ) } } , \frac { \operatorname { T r } ( H ) \varepsilon _ { d } / 3 } { \log ^ { 2 } d | H | | _ { \infty } \rangle } \} \Big ) } \\ & { \quad = 2 ( 4 M _ { B } d + 1 ) \exp \Big ( - { \varepsilon } \operatorname* { m i n } \{ \frac { \frac { \varepsilon _ { \mathrm { e f f } } \varepsilon _ { d } ^ { 2 } } { 9 \log ^ { 4 } d } } { 9 \log ^ { 4 } d ^ { 3 } \cdot \frac { \frac { \varepsilon _ { \mathrm { e f f } } \varepsilon _ { d } } { 3 0 \log ^ { 2 } d } } { 3 \log ^ { 2 } d } } \} \Big ) } \\ & { \quad = 2 ( 4 M _ { B } d + 1 ) \exp \Big ( - { \varepsilon } \operatorname* { m i n } \{ \frac { \log ^ { 2 } d } { 9 } , \frac { \log d \sqrt { \mathrm { r e r } } } { 3 } \} \Big ) . } \end{array}\tag{15}
$$

Integrating over θ and using the equality in law established above,

$$
\begin{array} { r l } & { \mathbb { P } \Big ( \underset { s > 0 } { \operatorname* { s u p } } | O _ { B } ( s , \vartheta , X ) | > \varepsilon _ { d } \Big ) } \\ & { \quad \le \mathbb { P } _ { \vartheta } \big ( \| \vartheta \| _ { \infty } > \beta \big ) + 2 ( 4 M _ { B } d + 1 ) \exp \Big ( - \frac { c } { 9 } \operatorname* { m i n } \big \{ \log ^ { 2 } d , \log d \sqrt { r _ { \mathrm { e f f } } } \big \} \Big ) . } \end{array}
$$

Combining (7), (13) and (15), we conclude that

$$
\begin{array} { r l } & { \mathbb { P } \left( \underset { s > 0 } { \operatorname* { s u p } } | \mathcal { E } ( s ) - \mathcal { E } ^ { \infty } ( s ) | > 2 \sqrt { \frac { \log ^ { 6 } d } { r _ { \mathrm { e f f } } ( H ) } } + 1 2 M _ { B } \frac { \log ^ { 2 } d } { d } \right) } \\ & { \leq C M _ { B } d \exp \left\{ - c \operatorname* { m i n } \left\{ \log ^ { 2 } d , \log d \sqrt { r _ { \mathrm { e f f } } ( H ) } \right\} \right\} } \end{array}
$$

for absolute constants $c , C > 0$ In particular, if $r _ { \mathrm { e f f } } ( H ) \geq \log ^ { 8 } d ,$ then $\operatorname* { m i n } \{ \log ^ { 2 } d , \log d \sqrt { r _ { \mathrm { e f f } } } \} =$ $\log ^ { 2 } d$ and $\sqrt { \log ^ { 6 } d / r _ { \mathrm { e f f } } } \leq 1 /$ log $d ,$ so that

$$
\mathbb { P } \left( \operatorname* { s u p } _ { s > 0 } | { \mathcal { E } } ( s ) - { \mathcal { E } } ^ { \infty } ( s ) | > { \frac { 2 + 1 2 M _ { B } } { \log d } } \right) \leq C M _ { B } d e ^ { - c \log ^ { 2 } d } ,
$$

where $c , C > 0$ are universal constants independent of $d .$

## A.3 PROOF OF COROLLARY 1

Corollary 1 follows directly from the uniform approximation in Theorem 1, which controls the difference between the infima of $\mathcal { E }$ and ${ \mathcal { E } } ^ { \infty }$ and ensures that any approximate minimizer of $\mathcal { E }$ also approximately minimizes ${ \mathcal { E } } ^ { \infty }$

Corollary 1. Under the assumptions of Theorem $^ { l , }$ set $\begin{array} { r } { \varepsilon _ { d } : = \frac { 2 + 1 2 M _ { B } } { \log d } } \end{array}$ . Then, with probability at least $1 - C M _ { B } d e ^ { - c \log ^ { 2 } d }$ , the following statements hold simultaneously:

1. The optimal values satisfy

$$
\left| m - m ^ { \infty } \right| \leq \varepsilon _ { d } .
$$

2. For every $\eta \geq 0$ and every $\widehat s > 0$ satisfying ${ \mathcal { E } } ( { \widehat { s } } ) \leq m + \eta$ one has

$$
0 \ \leq \ ( \mathcal { E } ^ { \infty } ( \widehat { s } ) - m ^ { \infty } ) \ \leq \ 2 \varepsilon _ { d } + \eta .
$$

In particular, any exact minimizer $\widehat { s } o f \mathcal { E }$ satisfies this bound with $\eta = 0 .$

Proof. By Theorem 1, with probability at least $1 - C M _ { B } d e ^ { - c \log ^ { 2 } d }$ it holds

$$
\operatorname* { s u p } _ { s > 0 } \left. { \mathcal E } ( s ) - { \mathcal E } ^ { \infty } ( s ) \right. \ \le \ \varepsilon _ { d } .
$$

We work on this event. For every $s > 0 ,$ , we have

$$
\begin{array} { r } { { \mathcal { E } } ^ { \infty } ( s ) - \varepsilon _ { d } \leq { \mathcal { E } } ( s ) \leq { \mathcal { E } } ^ { \infty } ( s ) + \varepsilon _ { d } . } \end{array}
$$

Taking infima over $s > 0$ yields

$$
m ^ { \infty } - \varepsilon _ { d } \leq m \leq m ^ { \infty } + \varepsilon _ { d } ,
$$

which proves the first point of the corollary.

Now let $\eta \geq 0$ and let $\widehat s > 0$ satisfy $\mathcal { E } ( \widehat { s } ) \leq m + \eta$ The uniform error bound and the first assertion give

$$
\begin{array} { c } { { \mathcal { E } ^ { \infty } ( \widehat { s } ) \leq \mathcal { E } ( \widehat { s } ) + \varepsilon _ { d } } } \\ { { \leq m + \eta + \varepsilon _ { d } } } \\ { { \leq m ^ { \infty } + \eta + 2 \varepsilon _ { d } . } } \end{array}
$$

The observation that $\mathcal { E } ^ { \infty } ( \widehat { s } ) \geq m ^ { \infty }$ concludes the proof.

## A.4 VERIFYING (H2) FOR MULTI-LAYER PERCEPTRONS AT INITIALIZATION

The uniform approximation of Theorem 1 relies on (H2), namely a lower bound on the effective rank of $X X ^ { \top }$ . In this section we show that this condition holds, with high probability, in the setting of a random multi-layer perceptron with isotropic data for channelwise quantization, i.e., for layer $\ell ,$ we quantize independently the rows $\vartheta _ { i : } ^ { ( \ell ) }$ and X is given by the output of the previous layer (not quantized). We begin by defining the considered model.

Definition 4. Let $L > 1$ be fixed. A multi-layer perceptron (MLP) of depth L is defined as the parametric function $f _ { \vartheta } : \mathbb { R } ^ { d _ { 0 } } $ R obtained via the recursion

$$
\begin{array} { l } { { \displaystyle { \alpha ^ { ( 0 ) } ( x ) : = x , \quad \alpha ^ { ( \ell ) } ( x ) : = \sigma \left( \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \alpha ^ { ( \ell - 1 ) } ( x ) \right) \quad f o r \ell = 1 , \dots , L - 1 , } } } \\ { { \displaystyle { f _ { \vartheta } ( x ) : = \frac { 1 } { \sqrt { d _ { L - 1 } } } \vartheta ^ { ( L ) \top } \alpha ^ { ( L - 1 ) } ( x ) , } } } \end{array}\tag{16}
$$

Here, $\vartheta ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { \ell } \times d _ { \ell - 1 } } f o r \ell = 1 , \ldots , L - 1 ,$ and $\vartheta ^ { ( L ) } \in \mathbb { R } ^ { d _ { L - 1 } }$ . The nonlinearity $\sigma : \mathbb { R }  \mathbb { R }$ acts componentwise, is $L _ { \sigma ^ { - } } L i p s c h i t z ,$ and satisfies $\sigma ( 0 ) = 0 $

Rather than assuming (H2) directly, we give explicit conditions on the network architecture and calibration distribution under which it holds with high probability

## Assumptions We assume that:

(H3) $\begin{array} { l } { { d _ { k } } } \end{array} = \begin{array} { l } { { \alpha _ { k } } { d } } \end{array}$ for some constant $\alpha _ { \ell } \ > \ 0$ and $\ell \in \ [ L ]$ , and we denote with $d \quad : = { }$ mir $\ v { a } _ { k = 1 , \ldots , L - 1 } \ \ v { d } _ { k }$

(H4) the calibration dataset $\mathbb { D } _ { c a l } ~ = ~ \{ X _ { i } ^ { \prime } \} _ { i = 1 } ^ { N _ { c } }$ is the realization of $N _ { c }$ iid copies of $X ^ { \prime } \sim$ $\mathcal { N } ( 0 , \mathrm { I d } _ { d _ { 0 } } )$ We denote with with $\ddot { X ^ { \prime } } ~ = ~ [ X _ { 1 } ^ { \prime } \ldots , X _ { N _ { c } } ^ { \prime } ] ~ \in ~ \mathbb { R } ^ { d _ { 0 } \times N _ { c } }$ . We assume that $N _ { c } \geq C \log ^ { 8 }$ d for some constant $C > 0$ not dependent on d.

Assumption (H3) places the hidden-layer widths in a proportional regime. Assumption (H4) models the calibration inputs as isotropic Gaussian and requires the sample size to grow at least polylogarithmically with the minimum hidden width. This permits calibration sets that are asymptotically smaller than the layer widths. We additionally assume that the activation is odd and Lipschitz covering choices such as tanh but excluding ReLU.

The results of Theorem 1 relies on the feature Gram matrix $\alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top }$ having large effective rank. The following theorem shows that this condition is fulfilled, with high probability, for the (iterative) PTQ problem of a MLP with Gaussian weights and Gaussian calibration inputs, as soon as the width is large enough.

In the rest of this section, to improve readability, we sometimes omit the constants which do not depend on d and $N _ { c }$ form the formulas, since they don't affect the asymptotic behavior. We start by controlling, with high probability, the operator norm $\| \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } \dot { ( X ^ { \prime } ) } ^ { \top } \| _ { \mathrm { o p } }$ from above and the trace tr $\left( \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \right)$ from below, which are the two quantities entering the effective rank. Lemma 8. Let us assume (H1), (H3) and $( H 4 )$ and let $\ell \in [ L - 1 ]$ . Then, there exists universal constants $C , c > 0$ such that

$$
\begin{array} { r } { \mathbb { P } _ { \vartheta , X ^ { \prime } } \left( \left\| \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \right\| _ { \mathrm { o p } } > C ( d + N _ { c } ) \right) \le 2 \cdot 9 ^ { - c ( d + N _ { c } ) } + 2 \ell e ^ { - d } . } \end{array}
$$

Proof. Let us fix $\ell \in [ d ] , u \in \mathbb { S } ^ { d _ { \ell } - 1 }$ and $v \in \mathbb { S } ^ { N _ { c } - 1 }$ . Let us fix $\vartheta ^ { ( k ) } \in \mathbb { R } ^ { d _ { k } \times d _ { k - 1 } }$ for $k \in [ \ell ]$ and define $F _ { u , v } : \mathbb { R } ^ { d _ { 0 } \times N _ { c } } \stackrel { \bullet } {  }$ R as follows

$$
F _ { u , v } ( X ) : = u ^ { \top } \sigma \left( \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \alpha ^ { ( \ell - 1 ) } ( X ) \right) v ,
$$

where $\alpha ^ { ( \ell - 1 ) } ( X ) \in \mathbb { R } ^ { d _ { \ell - 1 } \times N _ { c } }$ is the output of the (l—1)-th layer evaluated on the calibration dataset (with the convention $\alpha ^ { ( 0 ) } ( X ) = X )$ . We note that $F _ { u , v }$ is Lipschitz. Indeed, since $\| u \| _ { 2 } = \| v \| _ { 2 } = 1$ and σ is $L _ { \sigma ^ { - 1 } }$ ipschitz and applied entrywise, for all $\boldsymbol { X } , \boldsymbol { Y } \in \mathbb { R } ^ { d _ { 0 } \times N _ { c } }$ it holds

$$
\begin{array} { r l } & { \left| F _ { u , v } ( X ) - F _ { u , v } ( Y ) \right| \leq \left\| \sigma \left( \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \alpha ^ { ( \ell - 1 ) } ( X ) \right) - \sigma \left( \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \alpha ^ { ( \ell - 1 ) } ( Y ) \right) \right\| _ { F } } \\ & { \qquad \leq L _ { \sigma } \left\| \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \right\| _ { o p } \| \alpha ^ { ( \ell - 1 ) } ( X ) - \alpha ^ { ( \ell - 1 ) } ( Y ) \| _ { F } } \\ & { \qquad \leq L _ { \sigma } ^ { \ell } \displaystyle \prod _ { k = 1 } ^ { \ell } \left\| \frac { 1 } { \sqrt { d _ { k - 1 } } } \vartheta ^ { ( k ) } \right\| _ { o p } \| X - Y \| _ { F } = : \Lambda ( \vartheta ) \| X - Y \| _ { F } , } \end{array}
$$

where we used $\lVert M N \rVert _ { F } \le \lVert M \rVert _ { o p } \lVert N \rVert _ { F }$ and, in the last step, we iterated the previous inequality over the layers $\ell - 1 , \ldots , 1$ . Moreover, since the activation $\sigma$ is odd, it follows that $F _ { u , v } ( - X ) =$ $- F _ { u , v } ( X )$ . Now, since by (H4), $X _ { i j } ^ { \prime } \stackrel { i i d } { \sim } \mathcal { N } ( 0 , 1 )$ which is symmetric, it holds that

$$
\mathbb { E } _ { X ^ { \prime } } [ F _ { u , v } ( X ^ { \prime } ) ] = 0 .
$$

Therefore, using Lemma 5, for all $t > 0$ , it holds

$$
\mathbb { P } _ { X ^ { \prime } } \big ( | F _ { u , v } ( X ^ { \prime } ) | > t \big ) \ \leq \ 2 \exp \left\{ - \frac { t ^ { 2 } } { 2 \Lambda ( \vartheta ) ^ { 2 } } \right\} .\tag{17}
$$

We now control $\Lambda ( \vartheta )$ . Since the entries of $\vartheta ^ { ( k ) }$ are i.i.d. $\mathcal { N } ( 0 , 1 )$ , we have $K = \operatorname* { m a x } _ { i , j } \| \vartheta _ { i j } ^ { ( k ) } \| _ { \psi _ { 2 } } =$ ${ \sqrt { 8 / 3 } }$ , and Theorem 4 with $t = \sqrt { d }$ gives, for every $k \in [ \ell ]$ (recall $d _ { k } = \alpha d )$

$$
\begin{array} { r } { \mathbb { P } _ { \vartheta } \left( \| \vartheta ^ { ( k ) } \| _ { o p } > 3 C K \sqrt { d } \right) \le \mathbb { P } _ { \vartheta } \left( \| \vartheta ^ { ( k ) } \| _ { o p } > C K \big ( \sqrt { d _ { k } } + \sqrt { d _ { k - 1 } } + \sqrt { d } \big ) \right) \le 2 e ^ { - d } . } \end{array}
$$

Hence, by a union bound, the event

$$
\mathcal { E } : = \left. \operatorname* { m a x } _ { k \in \left[ \ell \right] } \| \vartheta ^ { ( k ) } \| _ { o p } \leq 3 C K \sqrt { d } \right.
$$

satisfies $\begin{array} { r l r } { \mathbb { P } _ { \vartheta } ( \mathcal { E } ^ { c } ) } & { { } \le } & { 2 \ell e ^ { - d } } \end{array}$ , and on $\mathcal { E } ,$ under hypothesis (H3), it holds $\begin{array} { r c l c r } { \Lambda ( \vartheta ) } & { \leq } & { \Lambda } & { : = } \end{array}$ $\begin{array} { r } { ( 3 C K L _ { \sigma } ) ^ { \ell } \prod _ { k = 1 } ^ { \ell } \sqrt { \frac { 1 } { \alpha _ { k - 1 } } } } \end{array}$ . In particular, by (17), for every $\vartheta \in \mathcal { E }$ and every $t > 0$

$$
\mathbb { P } _ { X ^ { \prime } } \big ( | F _ { u , v } ( X ^ { \prime } ) | > t \big ) \ \leq \ 2 \exp \left\{ - \frac { t ^ { 2 } } { 2 \Lambda ^ { 2 } } \right\} .\tag{18}
$$

Let $\mathcal { U } \subset \mathbb { S } ^ { d _ { \ell } - 1 } , \mathcal { V } \subset \mathbb { S } ^ { N _ { c } - 1 }$ be the $\textstyle { \frac { 1 } { 4 } }$ -nets of Lemma 11, so that $| \mathcal { U } | \le 9 ^ { d _ { \ell } }$ and $| \mathcal { V } | \le 9 ^ { N _ { c } }$ . By Lemma 11,

$$
\left. \sigma \left( \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \alpha ^ { ( \ell - 1 ) } ( X ^ { \prime } ) \right) \right. _ { \mathrm { o p } } \leq 2 \operatorname* { m a x } _ { ( u , v ) \in \mathcal { U } \times \mathcal { V } } | F _ { u , v } ( X ^ { \prime } ) | ,
$$

so, for every $t > 0$ , we have

$$
\left\{ \left\| \sigma \left( \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \alpha ^ { ( \ell - 1 ) } ( X ^ { \prime } ) \right) \right\| _ { \mathrm { o p } } > 2 t \right\} \subseteq \bigcup _ { ( u , v ) \in \mathcal { U } \times \mathcal { V } } \{ | F _ { u , v } ( X ^ { \prime } ) | > t \} ,
$$

and by (18) and a union bound, for every $\vartheta \in { \mathcal { E } } .$

$$
\begin{array} { r } { \mathbb { P } _ { X } \left( \left\| \sigma \left( \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \alpha ^ { ( \ell - 1 ) } ( X ^ { \prime } ) \right) \right\| _ { \mathrm { o p } } > 2 t \right) \le \vert \mathcal { U } \vert \vert \mathcal { V } \vert \cdot 2 \exp \left\{ - \frac { t ^ { 2 } } { 2 \Lambda ^ { 2 } } \right\} } \\ { \le 2 \cdot 9 ^ { d _ { \ell } + N _ { c } } \exp \left\{ - \frac { t ^ { 2 } } { 2 \Lambda ^ { 2 } } \right\} . } \end{array}
$$

Let us choose $t ^ { 2 } : = 4 \log 9 \left( d _ { \ell } + N _ { c } \right) \Lambda ^ { 2 }$ Then $\frac { t ^ { 2 } } { 2 \Lambda ^ { 2 } } = 2 ( d _ { \ell } + N _ { c } ) \log 9$ , so the right-hand side equals $2 \cdot 9 ^ { d _ { \ell } + N _ { c } } \cdot 9 ^ { - 2 \left( d _ { \ell } + N _ { c } \right) } = 2 \cdot 9 ^ { - \left( d _ { \ell } + N _ { c } \right) }$ . Finally, since $X ^ { \prime }$ and θ are independent,

$$
\begin{array} { r l } & { \mathbb { P } _ { \vartheta , X ^ { \prime } } \left( \bigg \| \sigma \left( \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \alpha ^ { ( \ell - 1 ) } ( X ^ { \prime } ) \right) \bigg \| _ { \mathrm { o p } } > 2 t \right) } \\ & { \quad \leq \mathbb { E } _ { \vartheta } \left[ \mathbf { 1 } _ { \mathcal { E } } \mathbb { P } _ { X ^ { \prime } } \left( \bigg \| \sigma \left( \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \alpha ^ { ( \ell - 1 ) } ( X ^ { \prime } ) \right) \bigg \| _ { \mathrm { o p } } > 2 t \right) \right] + \mathbb { P } _ { \vartheta } ( \mathcal { E } ^ { c } ) } \\ & { \quad \leq 2 \cdot 9 ^ { - ( d _ { \ell } + N _ { c } ) } + 2 \ell e ^ { - d } . } \end{array}
$$

Finally, since X and θ are independent,

$$
\begin{array} { r l } & { \mathbb { P } _ { \vartheta , X ^ { \prime } } \left( \bigg \| \sigma \left( \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \alpha ^ { ( \ell - 1 ) } ( X ^ { \prime } ) \right) \bigg \| _ { \mathrm { o p } } > 2 t \right) } \\ & { \quad \leq \mathbb { E } _ { \vartheta } \left[ \mathbf { 1 } _ { \mathcal { E } } \mathbb { P } _ { X ^ { \prime } } \left( \bigg \| \sigma \left( \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta ^ { ( \ell ) } \alpha ^ { ( \ell - 1 ) } ( X ^ { \prime } ) \right) \bigg \| _ { \mathrm { o p } } > 2 t \right) \right] + \mathbb { P } _ { \vartheta } ( \mathcal { E } ^ { c } ) } \\ & { \quad \leq 2 \cdot 9 ^ { - ( d _ { \ell } + N _ { c } ) } + 2 \ell e ^ { - d } . } \end{array}
$$

We note that $\| \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \| _ { o p } = \| \alpha ^ { ( \ell ) } ( X ) \| _ { o p } ^ { 2 }$ . Therefore, under (H3), there exists universal constants $C , c > 0$ such that

$$
\begin{array} { r l } & { \mathbb { P } _ { \vartheta , X ^ { \prime } } \left( \left\| \boldsymbol { \alpha } ^ { ( \ell ) } ( X ^ { \prime } ) \boldsymbol { \alpha } ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \right\| _ { \mathrm { o p } } > C \left( d + N _ { c } \right) \right) \leq \mathbb { P } _ { \vartheta , X ^ { \prime } } \left( \left\| \boldsymbol { \alpha } ^ { ( \ell ) } ( X ^ { \prime } ) \right\| _ { \mathrm { o p } } > 2 t \right) } \\ & { \qquad \leq 2 \cdot 9 ^ { - ( d _ { \ell } + N _ { c } ) } + 2 \ell e ^ { - d } } \\ & { \qquad \leq 2 \cdot 9 ^ { - c ( d + N _ { c } ) } + 2 \ell e ^ { - d } , } \end{array}
$$

which concludes the proof.

The next lemma shows that, with high probability, a lower bound on the trace t $\mathrm {  ~ \cdot ~ } \mathrm {  ~ r ~ } \big ( \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \big )$ for all $\ell \in [ L - 1 ]$

Lemma 9. Let us assume $( H l ) , ( H 3 )$ and (H4). Then there exist universal constants $a , C , c > 0$ not depending on d, such that for every $\ell \in \{ 0 , \ldots , L - 1 \}$ , it holds

$$
\begin{array} { r } { { \mathbb P } \left( \mathrm { t r } \left( \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \right) < a d N _ { c } \right) \le C N _ { c } e ^ { - c d } . } \end{array}
$$

Proof. Let $\ell \in \{ 0 , \ldots , L - 1 \}$ . We note that

$$
\mathrm { t r } \left( { \boldsymbol { \alpha } } ^ { ( \ell ) } ( X ^ { \prime } ) { \boldsymbol { \alpha } } ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \right) = \| { \boldsymbol { \alpha } } ^ { ( \ell ) } ( X ^ { \prime } ) \| _ { F } ^ { 2 } = \sum _ { j = 1 } ^ { N _ { c } } \| { \boldsymbol { \alpha } } _ { : j } ^ { ( \ell ) } ( X ^ { \prime } ) \| _ { 2 } ^ { 2 } = \sum _ { j = 1 } ^ { N _ { c } } \| { \boldsymbol { \alpha } } ^ { ( \ell ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } ^ { 2 } ,
$$

so that a lower bound on the trace follows from a lower bound on $\| \alpha ^ { ( \ell ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } ^ { 2 }$ holding simultaneously for all $j \in [ N _ { c } ]$ . For $\ell \geq 1 , i \in [ d _ { \ell } ]$ and $j \in [ N _ { c } ]$ , let us define the random variables

$$
\begin{array} { l } { { Z _ { i } ^ { j } : = \displaystyle \frac { 1 } { \sqrt { d _ { \ell - 1 } } } \vartheta _ { i : } ^ { ( \ell ) } \alpha ^ { ( \ell - 1 ) } ( X _ { : j } ^ { \prime } ) \ : , } } \\ { { Y _ { i } ^ { j } : = \sigma \big ( Z _ { i } ^ { j } \big ) ^ { 2 } \ : . } } \end{array}
$$

We note that $\begin{array} { r c l } { \| \alpha ^ { ( \ell ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } ^ { 2 } } & { = } & { \sum _ { i = 1 } ^ { d _ { \ell } } Y _ { i } ^ { j } } \end{array}$ Let $\mathcal { F } _ { \ell - 1 }$ denote the σ-algebra generated by $X ^ { \prime } , \vartheta ^ { ( 1 ) } , \ldots , \vartheta ^ { ( \ell - 1 ) }$ . Since $\vartheta ^ { ( \ell ) }$ is independent of $\mathcal { F } _ { \ell - 1 }$ , the random variables $Y _ { 1 } ^ { j } , \dots , Y _ { d _ { \ell } } ^ { j }$ are i.i.d., and, using (H3), we have

$$
0 \leq Y _ { i } ^ { j } \leq L _ { \sigma } ^ { 2 } \left( Z _ { i } ^ { j } \right) ^ { 2 } .
$$

Moreover, conditioned on $\mathcal { F } _ { \ell - 1 }$ , we have that $\begin{array} { r } { Z _ { i } ^ { j } \sim { \mathcal { N } } \left( 0 , \frac { \| \alpha ^ { ( \ell - 1 ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } ^ { 2 } } { d _ { \ell - 1 } } \right) } \end{array}$ . Let us define following iterative constants

$$
\begin{array} { l } { { a _ { 0 } : = \displaystyle \frac { 1 } { 2 } , \quad b _ { 0 } : = 2 } } \\ { { m _ { \ell } = \displaystyle \operatorname* { i n f } _ { h \in [ a _ { \ell - 1 } , b _ { \ell - 1 } ] } \mathbb { E } _ { Z \sim \mathcal { N } ( 0 , h ) } \left[ \sigma ^ { 2 } ( Z ) \right] , } } \\ { { b _ { \ell } : = 2 L _ { \sigma } ^ { 2 } b _ { \ell - 1 } , } } \\ { { a _ { \ell } : = \frac { 1 } { 2 } m _ { \ell } . } } \end{array}
$$

and the event

$$
E _ { \ell } : = \bigcap _ { j = 1 } ^ { N _ { c } } \left\{ a _ { \ell } d _ { \ell } \leq \Vert \alpha ^ { ( \ell ) } ( X _ { : j } ^ { \prime } ) \Vert _ { 2 } ^ { 2 } \leq b _ { \ell } d _ { \ell } \right\} .
$$

We estimate now the probability of $E _ { \ell }$ . We proceed by induction over $\ell .$ The base case $\ell = 0$ follows from Lemma 4. Indeed, since $\alpha ^ { ( 0 ) } ( \bar { X _ { : j } ^ { \prime } } ) = \bar { X _ { : j } ^ { \prime } } \sim \mathcal { N } ( 0 , \mathrm { I d } _ { d _ { 0 } } )$ by (H4), for each $j \in [ N _ { c } ]$ each of the two inequalities

$$
\frac { 1 } { 2 } d _ { 0 } \leq \| \alpha ^ { ( 0 ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } ^ { 2 } = \| X _ { : j } ^ { \prime } \| _ { 2 } ^ { 2 } \leq 2 d _ { 0 }
$$

fails with probability at most $e ^ { - d _ { 0 } / 1 6 }$ . Therefore, by a union bound,

$$
\mathbb { P } \left( E _ { 0 } ^ { c } \right) \leq \sum _ { j = 1 } ^ { N _ { c } } \left[ \mathbb { P } \left( \| X _ { : j } ^ { \prime } \| _ { 2 } ^ { 2 } < \frac { 1 } { 2 } d _ { 0 } \right) + \mathbb { P } \left( \| X _ { : j } ^ { \prime } \| _ { 2 } ^ { 2 } > 2 d _ { 0 } \right) \right] \leq 2 N _ { c } e ^ { - d _ { 0 } / 1 6 } .
$$

Let now $\ell \geq 1 . \ \mathrm { A s }$ inductive hypothesis, we assume that there exists $c _ { 1 } , \ldots , c _ { \ell - 1 } > 0 .$ , depending only on the activation σ and on the layer index, such that the event

$$
E _ { \ell - 1 } = \bigcap _ { j = 1 } ^ { N _ { c } } \left\{ a _ { \ell - 1 } d _ { \ell - 1 } \leq \| { \boldsymbol { \alpha } } ^ { ( \ell - 1 ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } ^ { 2 } \leq b _ { \ell - 1 } d _ { \ell - 1 } \right\}
$$

satisfies

$$
\mathbb { P } \left( E _ { \ell - 1 } ^ { C } \right) \le 2 N _ { c } e ^ { - d _ { 0 } / 1 6 } + N _ { c } \sum _ { k = 1 } ^ { \ell - 1 } \left( 2 e ^ { - c _ { k } d _ { k } } + e ^ { - d _ { k } / 1 6 } \right) .
$$

Then, we have that

$$
\begin{array} { r l r } {  { \mathbb { P } ( E _ { \ell } ^ { C } ) \le \mathbb { P } ( E _ { \ell } ^ { C } \Big | E _ { \ell - 1 } ) + \mathbb { P } ( E _ { \ell - 1 } ^ { C } ) } } \\ & { } & { \quad \le \displaystyle \sum _ { j = 1 } ^ { N _ { c } } [ \mathbb { P } ( \| \alpha ^ { ( \ell ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } ^ { 2 } < a _ { \ell } d _ { \ell } \Big | E _ { \ell - 1 } ) + \mathbb { P } ( \| \alpha ^ { ( \ell ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } ^ { 2 } > b _ { \ell } d _ { \ell } \Big | E _ { \ell - 1 } ) ] } \\ & { } & { \quad + \mathbb { P } ( E _ { \ell - 1 } ^ { C } ) . } \end{array}\tag{19}
$$

Since $E _ { \ell - 1 } \in \mathcal { F } _ { \ell - 1 }$ , for every event A we have $\mathbb { P } ( A \mid E _ { \ell - 1 } ) = \mathbb { E } \left[ \mathbf { 1 } _ { E _ { \ell - 1 } } \mathbb { P } ( A \mid \mathcal { F } _ { \ell - 1 } ) \right] / \mathbb { P } ( E _ { \ell - 1 } )$ , SO it suffices to bound $\mathbb { P } ( A | \mathcal { F } _ { \ell - 1 } )$ uniformly on the event $E _ { \ell - 1 }$

By Lemma 4, we notice that on $E _ { \ell - 1 }$ , it holds

$$
\begin{array} { r l } & { \mathbb { P } \left( \| \boldsymbol { \alpha } ^ { ( \ell ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } ^ { 2 } > b _ { \ell } d _ { \ell } \Big | \mathcal { F } _ { \ell - 1 } \right) \leq \mathbb { P } \left( L _ { \sigma } ^ { 2 } \displaystyle \sum _ { i = 1 } ^ { d _ { \ell } } \big ( Z _ { i } ^ { j } \big ) ^ { 2 } > 2 L _ { \sigma } ^ { 2 } b _ { \ell - 1 } d _ { \ell } \Big | \mathcal { F } _ { \ell - 1 } \right) } \\ & { \qquad \leq \mathbb { P } \left( \displaystyle \sum _ { i = 1 } ^ { d _ { \ell } } \sqrt { d _ { \ell - 1 } } \Big ( \frac { Z _ { i } ^ { j } } { \| \boldsymbol { \alpha } ^ { ( \ell - 1 ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } } \Big ) ^ { 2 } > 2 d _ { \ell } \Big | \mathcal { F } _ { \ell - 1 } \right) } \\ & { \qquad \leq e ^ { - d _ { \ell } / 1 6 } , } \end{array}\tag{20}
$$

where we used $Y _ { i } ^ { j } \le L _ { \sigma } ^ { 2 } ( Z _ { i } ^ { j } ) ^ { 2 }$ and $b _ { \ell } = 2 L _ { \sigma } ^ { 2 } b _ { \ell - 1 }$ in the first step, $s _ { j } ^ { 2 } \leq b _ { \ell - 1 }$ in the second step, i and the fact that, conditioned on $\begin{array} { r } { \mathcal { F } _ { \ell - 1 } , d _ { \ell - 1 } \frac { Z _ { 1 } ^ { j } } { \| \alpha ^ { ( \ell - 1 ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } } , \dots , d _ { \ell - 1 } \frac { Z _ { d _ { \ell } } ^ { j } } { \| \alpha ^ { ( \ell - 1 ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } } } \end{array}$ are iid $\mathcal { N } ( 0 , 1 )$ in the last step.

Let us define

$$
m _ { \ell } : = \operatorname* { i n f } _ { \substack { h \in [ a _ { \ell - 1 } , b _ { \ell - 1 } ] } } \mathbb { E } _ { Z \sim \mathcal { N } ( 0 , h ) } \big [ \sigma ^ { 2 } ( Z ) \big ] .\tag{21}
$$

We note that $m _ { \ell } > 0$ . Indeed, the map $h \mapsto \mathbb { E } _ { Z \sim \mathcal { N } ( 0 , h ) } [ \sigma ^ { 2 } ( Z ) ] = \mathbb { E } _ { g \sim \mathcal { N } ( 0 , 1 ) } [ \sigma ^ { 2 } ( \sqrt { h } g ) ]$ is continuous on $( 0 , \infty )$ , and it is strictly positive for every $h > 0 \cdot$ otherwise $\sigma ( { \sqrt { h } } g ) = 0$ almost surely, i.e. $\sigma = 0$ Lebesgue-almost everywhere, which is excluded by assumption. Hence the infimum over the compact interval $[ a _ { \ell - 1 } , b _ { \ell - 1 } ] \subset ( 0 , \infty )$ is attained and positive.

Note also that $m _ { \ell } \le L _ { \sigma } ^ { 2 } b _ { \ell - 1 }$ , since $\mathbb { E } _ { Z \sim \mathcal { N } ( 0 , h ) } [ \sigma ^ { 2 } ( Z ) ] \le L _ { \sigma } ^ { 2 } h$ . We notice that on $E _ { \ell - 1 }$

$$
\mathbb { E } \left[ Y _ { i } ^ { j } \mid \mathcal { F } _ { \ell - 1 } \right] = \mathbb { E } _ { Z \sim \mathcal { N } ( 0 , s _ { j } ^ { 2 } ) } \left[ \sigma ^ { 2 } ( Z ) \right] \ge m _ { \ell } ,
$$

by definition of $m _ { l }$ . Conditioned on $\mathcal { F } _ { \ell - 1 }$ , we note that $Y _ { 1 } ^ { j } , \dots , Y _ { d _ { \ell } } ^ { j }$ are sub-exponential i.i.d. random variables with

$$
\begin{array} { r l } & { \left\| Y _ { i } ^ { j } - \mathbb { E } [ Y _ { i } ^ { j } \mid \mathcal { F } _ { \ell - 1 } ] \right\| _ { \psi _ { 1 } } \leq C \left\| Y _ { i } ^ { j } \right\| _ { \psi _ { 1 } } } \\ & { \qquad \leq C L _ { \sigma } ^ { 2 } \left\| ( Z _ { i } ^ { j } ) ^ { 2 } \right\| _ { \psi _ { 1 } } } \\ & { \qquad = C L _ { \sigma } ^ { 2 } \| Z _ { i } ^ { j } \| _ { \psi _ { 2 } } ^ { 2 } } \\ & { \qquad = \frac { 8 } { 3 } C L _ { \sigma } ^ { 2 } \frac { \left\| \alpha ^ { ( \ell - 1 ) } ( X _ { : j } ^ { \prime } ) \right\| _ { 2 } ^ { 2 } } { d _ { \ell - 1 } } } \\ & { \qquad \leq \frac { 8 } { 3 } C L _ { \sigma } ^ { 2 } b _ { \ell - 1 } , } \end{array}
$$

where $C \geq 1$ is an absolute constant, the ψ1- and ψ2-norms are taken with respect to the conditional distribution given $\mathcal { F } _ { \ell - 1 }$ , and we used Lemma 2, Lemma 3, and Lemma 1 with $\begin{array} { r } { \tau = \frac { \| \alpha ^ { ( \ell - 1 ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } ^ { 2 } } { d _ { \ell - 1 } } } \end{array}$ By applying Theorem 2 conditioned on $\mathcal { F } _ { \ell - 1 }$ to the centered variables $Y _ { i } ^ { j } - \mathbb { E } [ Y _ { i } ^ { j } | \mathcal { F } _ { \ell - 1 } ]$ , with $\begin{array} { r } { t : = \bar { \frac { m _ { \ell } } { 2 } } \bar { d _ { \ell } } } \end{array}$ , it follows that on $E _ { \ell - 1 }$

$$
\begin{array} { r l r } {  { \mathbb { P } ( \| \alpha ^ { ( \ell ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } ^ { 2 } < a _ { \ell } d _ { \ell } \Big | \mathcal { F } _ { \ell - 1 } ) \le \mathbb { P } ( | \sum _ { i = 1 } ^ { d _ { \ell } } Y _ { i } ^ { j } - d _ { \ell } \mathbb { E } [ Y _ { 1 } ^ { j } \mid \mathcal { F } _ { \ell - 1 } ] | \ge \frac { m _ { \ell } } { 2 } d _ { \ell } \Big | \mathcal { F } _ { \ell - 1 } ) } } \\ & { } & { \le 2 \exp \{ - c \operatorname* { m i n } ( \frac { t ^ { 2 } } { d _ { \ell } K _ { \ell } ^ { 2 } } , \frac { t } { K _ { \ell } } ) \} , } \end{array}
$$

where $c > 0$ is an absolute constant, and in the first step we used that, on $\begin{array} { r } { E _ { \ell - 1 } , a _ { \ell } d _ { \ell } = \frac { m _ { \ell } } { 2 } d _ { \ell } \le } \end{array}$ $\begin{array} { r } { d _ { \ell } \mathbb { E } [ Y _ { 1 } ^ { j } \vert \mathcal { F } _ { \ell - 1 } ] - \frac { m _ { \ell } } { 2 } d _ { \ell } . } \end{array}$ since $\mathbb { E } [ Y _ { 1 } ^ { j } \mid \mathcal { F } _ { \ell - 1 } ] \ge m _ { \ell }$ . We now compute the minimum in the exponent. Plugging in $\begin{array} { r } { t = \frac { m _ { \ell } } { 2 } \dot { d } _ { \ell } } \end{array}$ , we have

$$
\operatorname* { m i n } \left( \frac { t ^ { 2 } } { d \varepsilon K _ { \ell } ^ { 2 } } , \frac { t } { K _ { \ell } } \right) = d _ { \ell } \operatorname* { m i n } \left\{ \left( \frac { m _ { \ell } } { 2 K _ { \ell } } \right) ^ { 2 } , \frac { m _ { \ell } } { 2 K _ { \ell } } \right\} = d _ { \ell } \left( \frac { m _ { \ell } } { 2 K _ { \ell } } \right) ^ { 2 } ,
$$

where the last equality holds because $m _ { \ell } \le L _ { \sigma } ^ { 2 } b _ { \ell - 1 } \le K _ { \ell }$ gives $\begin{array} { r } { \frac { m _ { \ell } } { 2 K _ { \ell } } \le \frac { 1 } { 2 } < 1 } \end{array}$ , so that the square is the smaller of the two terms. Therefore, on $E _ { \ell - 1 }$ , we have

$$
\begin{array} { r } { \mathbb { P } \left( \| \boldsymbol { \alpha } ^ { ( \ell ) } ( X _ { : j } ^ { \prime } ) \| _ { 2 } ^ { 2 } < a _ { \ell } d _ { \ell } \Big | \mathcal { F } _ { \ell - 1 } \right) \le 2 \exp \left\{ - c _ { \ell } d _ { \ell } \right\} , } \end{array}\tag{22}
$$

where $\begin{array} { r } { c _ { \ell } = c \left( \frac { m _ { \ell } } { 2 K _ { \ell } } \right) ^ { 2 } > 0 } \end{array}$ depends only on σ and l, and not on $d _ { \ell }$ or $N _ { c }$

Using (19), (20), (22) and taking the union over $j \in [ N _ { c } ]$ , we obtain

$$
\mathbb { P } \left( E _ { \ell } ^ { C } \right) \le 2 N _ { c } e ^ { - d _ { 0 } / 1 6 } + N _ { c } \sum _ { k = 1 } ^ { \ell } \left( 2 e ^ { - c _ { k } d _ { k } } + e ^ { - d _ { k } / 1 6 } \right) .
$$

Under (H3), there exist universal constants $C , c > 0$ which depends only on σ and κ such that

$$
\mathbb { P } \left( E _ { \ell } ^ { c } \right) \le C N _ { c } e ^ { - c d } .
$$

Finally, let $a : = \alpha _ { \ell }$ min $\cdot 0 { \le } \ell { \le } L { - } 1 \ : a _ { \ell } > 0$ . Since on the event $E _ { \ell }$ we have $\Vert \alpha ^ { ( \ell ) } ( X _ { : j } ^ { \prime } ) \Vert _ { 2 } ^ { 2 } \geq a _ { \ell } d _ { \ell } \geq$ a $d _ { \ell }$ for every $j \in [ N _ { c } ]$ , the first display of the proof yields the inclusion

$$
E _ { \ell } \subseteq \left\{ \mathrm { t r } \left( \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \right) \geq a d N _ { c } \right\} ,
$$

and therefore

$$
\mathbb { P } \left( \mathrm { t r } \left( \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \right) < a d _ { \ell } N _ { c } \right) \le \mathbb { P } \left( E _ { \ell } ^ { c } \right) \le C N _ { c } e ^ { - c d } .
$$

This concludes the proof.

Theorem 5. Let us assume (H1), (H3) and (H4), and let $\ell \in [ L - 1 ]$ . Then there exist constants $C , c , \bar { C } > 0$ not dependent on d, and an absolute constant $D > 0$ , such that for all $d > D$ , with probability at least $\dot { \mathrm { ~ 1 ~ - ~ } } \bar { C } N _ { c } e ^ { - c d }$ over θ and X,

$$
r _ { e f f } \bigl ( \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \bigr ) \ \geq C \log ^ { 8 } d .
$$

Proof. By definition of the effective rank,

$$
r _ { \mathrm { e f f } } \big ( \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \big ) = \frac { \mathrm { t r } \left( \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \right) } { \left\| \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \right\| _ { \mathrm { o p } } } ,
$$

By Lemma 8, there exist constants $C , c > 0$ such that

$$
\begin{array} { r } { \mathbb { P } _ { \vartheta , X ^ { \prime } } \left( \left\| \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \right\| _ { \mathrm { o p } } > C ( d + N _ { c } ) \right) \le 2 \cdot 9 ^ { - c ( d + N _ { c } ) } + 2 \ell e ^ { - d } . } \end{array}
$$

On the other hand, by Lemma 9, there exist constants $a , c ^ { \prime } , C ^ { \prime } > 0$ such that

$$
\begin{array} { r } { \mathbb { P } _ { \vartheta , X ^ { \prime } } \left( \mathrm { t r } \left( \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \right) < a d N _ { c } \right) \le C ^ { \prime } N _ { c } e ^ { - c ^ { \prime } d } . } \end{array}
$$

Therefore, by a union bound, with probability at least

$$
1 - 2 \cdot 9 ^ { - c ( d + N _ { c } ) } - 2 \ell e ^ { - d } - C ^ { \prime } N _ { c } e ^ { - c ^ { \prime } d } \geq 1 - \bar { C } N _ { c } e ^ { - \bar { c } d }
$$

with $\bar { c } : =$ min{clog $9 , 1 , c ^ { \prime } \} , \bar { C } : = 2 + 2 { \cal L } + C ^ { \prime } :$ it holds that

$$
r _ { \mathrm { e f f } } \big ( \alpha ^ { ( \ell ) } ( X ^ { \prime } ) { \alpha ^ { ( \ell ) } ( X ^ { \prime } ) } ^ { \top } \big ) = \frac { \mathrm { t r } \big ( \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \big ) } { \big \| \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \big \| _ { \mathrm { o p } } } \ge \tilde { C } \frac { d N _ { c } } { d + N _ { c } } \ge \frac { \tilde { C } } { 2 } \operatorname* { m i n } \{ d , N _ { c } \} ,
$$

with $\tilde { C } : = a \kappa / C$ , where in the last step we have used the fact that $\begin{array} { r } { \frac { d N _ { c } } { d + N _ { c } } \geq \frac { 1 } { 2 } } \end{array}$ min $\{ d , N _ { c } \}$ . Therefore, since by (H4) $N _ { c } \geq \log ^ { 8 } d ,$ and $\log ^ { 8 } d \leq d$ for d large enough, we conclude that, with probability at least $1 - \bar { C } N _ { c } e ^ { - \bar { c } d }$

$$
r _ { \mathrm { e f f } } \bigl ( \alpha ^ { ( \ell ) } ( X ^ { \prime } ) \alpha ^ { ( \ell ) } ( X ^ { \prime } ) ^ { \top } \bigr ) \ \geq \ \frac { \tilde { C } } { 2 } \ \log ^ { 8 } d .
$$

## A.5 PROOF OF PROPOSITION 1

We start by proving that ${ \mathcal { E } } ^ { \infty }$ separates the two sources of quantization error. The first is the rounding error, which does not depend on the bit precision and would be present even with an infinite grid; the second is the clipping error, due to inputs falling outside the representable range $[ - M _ { B } s , \bar { M _ { B } } s ]$ and it is the only term through which $M _ { B }$ enters. We record this decomposition in the following corollary, which will be the starting point of our analysis of the loss landscape.

Corollary 2. For every $B \geq 2$ and $s > 0 ,$

$$
\begin{array} { l } { \displaystyle { \mathcal E ^ { \infty } ( M _ { B } , s ) = \mathcal I _ { \infty } ( s ) + 4 s \sum _ { j = 0 } ^ { \infty } g \big ( \big ( M _ { B } + j + \frac { 1 } { 2 } \big ) s \big ) , } } \end{array}
$$

with

$$
\mathcal { I } _ { \infty } ( s ) : = \mathbb { E } _ { Z \sim \mathcal { N } ( 0 , 1 ) } [ ( Z - s \lfloor Z / s \rfloor ) ^ { 2 } ] = 1 - \sum _ { j = 0 } ^ { \infty } 4 s g \bigl ( ( j + \frac { 1 } { 2 } ) s \bigr ) ,
$$

where $g ( x ) : = \varphi ( x ) - x ( 1 - \Phi ( x ) )$ , and both series converge absolutely

Proof. For a natural number $n \geq 0 ,$ let $\Pi _ { s , n } ( z ) : = s \operatorname* { m a x } \{ - n , \operatorname* { m i n } \{ n , \lfloor z / s \rceil \} \}$ denote the quantizer with 2n + 1 levels, so that $\mathcal { E } ^ { \infty } ( n , s ) = \mathbb { E } [ ( Z - \Pi _ { s , n } ( Z ) ) ^ { 2 } ]$ . Let us fix $n \geq 0 .$ , and notice that the quantizers $\Pi _ { s , n }$ and $\Pi _ { s , n + 1 }$ agree on $\{ | z | \leq ( n + { \frac { 1 } { 2 } } ) s \}$ . On $\{ z > ( n + \frac { 1 } { 2 } ) s \}$ one has $\Pi _ { s , n } ( z ) = n s .$ while $\lfloor z / s \rceil \geq n + 1$ there, so $\Pi _ { s , n + 1 } ( z ) = ( n + 1 ) s ;$ the case $z < - ( n + \textstyle { \frac { 1 } { 2 } } ) s$ is symmetric. It holds

$$
\begin{array} { l } { \displaystyle \mathcal { E } ^ { \infty } ( n , s ) - \mathcal { E } ^ { \infty } ( n + 1 , s ) = 2 \int _ { ( n + 1 / 2 ) s } ^ { \infty } \biggl [ ( z - n s ) ^ { 2 } - \bigl ( z - ( n + 1 ) s \bigr ) ^ { 2 } \biggr ] \varphi ( z ) d z } \\ { \displaystyle \stackrel { ( * ) } { = } 4 s \int _ { ( n + \frac { 1 } { 2 } ) s } ^ { \infty } ( z - ( n + \frac { 1 } { 2 } ) s ) \varphi ( z ) d z } \\ { = 4 s \bigl [ \varphi \bigl ( ( n + \frac { 1 } { 2 } ) s \bigr ) - ( n + \frac { 1 } { 2 } ) s \left( 1 - \Phi \left( ( n + \frac { 1 } { 2 } ) s \right) \right) \bigr ] } \\ { = 4 s g \left( ( n + \frac { 1 } { 2 } ) s \right) , } \end{array}
$$

where in (\*) we have used the fact that

$$
( z - n s ) ^ { 2 } - { \bigl ( } z - ( n + 1 ) s { \bigr ) } ^ { 2 } = 2 z s - 2 n s ^ { 2 } - s ^ { 2 } = 2 s { \Bigl ( } z - { \bigl ( } n + { \frac { 1 } { 2 } } { \bigr ) } s { \Bigr ) } ,
$$

and $\textstyle \int _ { x } ^ { \infty } z \varphi ( z ) d z \ = \ \varphi ( x )$ and $\begin{array} { r } { \int _ { x } ^ { \infty } \varphi ( z ) \ d z \ = \ 1 - \Phi ( x ) } \end{array}$ . Let us fix $N > M _ { B }$ . Summing on $n = \mathrm { \tilde { \it M } } _ { B } , \dots , N - 1$ gives

$$
\begin{array} { l } { \displaystyle \mathcal { E } ^ { \infty } ( M _ { B } , s ) - \mathcal { E } ^ { \infty } ( N , s ) = \sum _ { n = M _ { B } } ^ { N - 1 } 4 s g \big ( ( n + \frac { 1 } { 2 } ) s \big ) . } \end{array}
$$

Since $0 \leq g ( x ) \leq \varphi ( x )$ for $x \geq 0$ , the series converges absolutely. Moreover $\begin{array} { r } { \left( Z - \Pi _ { s , N } ( Z ) \right) ^ { 2 } \le } \end{array}$ $Z ^ { 2 } \in L ^ { 1 }$ and $\Pi _ { s , N } ( Z ) ~  ~ s \lfloor Z / s \rceil$ point-wise as $N \uparrow \infty$ , so dominated convergence gives $\mathcal { E } ^ { \infty } ( N , s ) \to \mathcal { T } _ { \infty } ( \dot { s } )$ . Letting $N \uparrow$ ∞ and substituting $j = n - M _ { B }$ concludes the proof. Finally, since $\Pi _ { s , 0 } ~ \equiv ~ 0$ and hence $\mathcal { E } ^ { \infty } ( 0 , s ) = \mathbb { E } [ \breve { Z } ^ { 2 } ] = 1$ , we can use the above formula for $\mathcal { E } ^ { \infty } ( M _ { B } , s ) - \mathcal { E } ^ { \infty } ( N , s )$ with $M _ { B }$ replaced by 0 and $N \to \infty$ to prove that

$$
\begin{array} { r } { \mathcal { T } _ { \infty } ( s ) = 1 - { \displaystyle \sum _ { n = 0 } ^ { \infty } } 4 s g \big ( ( n + \frac { 1 } { 2 } ) s \big ) . } \end{array}
$$

For clarity, we restate Proposition 1 below.

Proposition 1. The following properties hold:

(i) For every $B \geq 2 ,$ the function $\bar { \mathcal { E } } ^ { \infty } ( m , s )$ is smooth, e.g. $\bar { \mathcal { E } } ^ { \infty } ( m , s ) \in \mathcal { C } ^ { \infty } ( [ 1 , \infty ) \times ( 0 , \infty ) )$

(ii) For every real $m \geq 1$ , the function $s \mapsto { \bar { \mathcal { E } } } ^ { \infty } ( m , s )$ has a unique critical point on $( 0 , \infty )$ This point is its unique global minimizer $s ^ { * } ( m )$ and is nondegenerate:

$$
\partial _ { s } ^ { 2 } \bar { \mathcal { E } } ^ { \infty } ( m , s ^ { * } ( m ) ) > 0 .
$$

(iii) The map $m \mapsto s ^ { * } ( m )$ is continuously differentiable and strictly decreasing on $\lbrack 1 , \infty )$ Moreover, for every $m \geq 1$

$$
\frac { d s ^ { * } } { d m } ( m ) = - \frac { \bar { \mathcal { E } } _ { m s } ^ { \infty } \bigl ( m , s ^ { * } ( m ) \bigr ) } { \bar { \mathcal { E } } _ { s s } ^ { \infty } \bigl ( m , s ^ { * } ( m ) \bigr ) } < 0 , \qquad s ^ { * } ( m ) > \frac { \sqrt { 2 } } { m + \frac { 1 } { 2 } } .
$$

Proof. We prove each of the four points separately.

(i) The function g is ${ \mathcal { C } } ^ { \infty }$ , with $g ^ { \prime } = - ( 1 - \Phi ) , g ^ { \prime \prime } = \varphi$ and $g ^ { ( k ) } = p _ { k } \varphi$ for a polynomial $p _ { k }$ when $k \geq 2 ;$ in particular $| g ^ { ( k ) } ( x ) | \le C _ { k } ( 1 + x ) ^ { k } \varphi ( x )$ for $x \ge 1$ and all $k \geq 0 ,$ , using $g ( x ) \leq \varphi ( x ) / x ^ { 2 }$ and $1 - \Phi ( x ) \leq \varphi ( x ) / x$ by Lemma 12. Hence every partial derivative in $( m , s )$ of the summand $s g \big ( ( m + j + \textstyle { \frac { 1 } { 2 } } ) s \big )$ is bounded by a summable function, on $\{ s \ge s _ { 0 } , m \ge 1 \}$ . All series of partial derivatives thus converge uniformly on compact subsets of $[ 1 , \infty ) \times ( 0 , \infty )$ , and term-wise differentiation gives $\bar { \mathcal { E } } ^ { \infty } \in \mathcal { C } ^ { \infty } \big ( [ 1 , \overset { \cdot } { \infty } ) \times ( 0 , \overset { \cdot } { \infty } ) \big )$

## (ii) We start by noticing that

$$
\begin{array} { l } { { \displaystyle g ( a s ) = \varphi ( a s ) - a s \big ( 1 - \Phi ( a s ) \big ) = \int _ { a s } ^ { \infty } t \varphi ( t ) d t - a s \int _ { a s } ^ { \infty } \varphi ( t ) d t } } \\ { ~ } \\ { { \displaystyle ~ = \int _ { a s } ^ { \infty } ( t - a s ) \varphi ( t ) d t } } \\ { { \displaystyle ~ = s ^ { 2 } \int _ { a } ^ { \infty } ( u - a ) \varphi ( s u ) d u } } \\ { { \displaystyle ~ = s ^ { 2 } \int _ { 0 } ^ { \infty } ( u - a ) _ { + } \varphi ( s u ) d u } . } \end{array}
$$

Therefore, by definition of $\bar { \mathcal { E } } ^ { \infty }$ (cf. Corollary 2), we can write

$$
\begin{array} { l } { \displaystyle \bar { \mathcal { E } } ^ { \infty } ( m , s ) = 1 - 4 s \sum _ { j = 0 } ^ { \infty } \left[ g \big ( ( j + \frac 1 2 ) s \big ) - g \big ( ( j + m + \frac 1 2 ) s \big ) \right] } \\ { \displaystyle \qquad = 1 - 4 s \sum _ { j = 0 } ^ { \infty } s ^ { 2 } \int _ { 0 } ^ { \infty } \left[ \big ( u - ( j + \frac 1 2 ) \big ) _ { + } - \big ( u - ( j + m + \frac 1 2 ) \big ) _ { + } \right] \varphi ( s u ) d u } \\ { \displaystyle \qquad = 1 - 4 s ^ { 3 } \int _ { 0 } ^ { \infty } \sum _ { j = 0 } ^ { \infty } \left[ \big ( u - ( j + \frac 1 2 ) \big ) _ { + } - \big ( u - m - ( j + \frac 1 2 ) \big ) _ { + } \right] \varphi ( s u ) d u } \\ { \displaystyle \qquad = 1 - 4 s ^ { 3 } \int _ { 0 } ^ { \infty } D _ { m } ( u ) \varphi ( s u ) d u , } \end{array}
$$

where $\begin{array} { r } { D _ { m } ( u ) : = \sum _ { j = 0 } ^ { \infty } \left[ \left( u - ( j + \frac { 1 } { 2 } ) \right) _ { + } - \left( u - m - ( j + \frac { 1 } { 2 } ) \right) _ { + } \right] } \end{array}$ . Differentiating in s, it holds

$$
\begin{array} { r l } { \partial _ { s } \bar { \mathcal { E } } ^ { \infty } ( m , s ) = - 1 2 s ^ { 2 } \displaystyle \int _ { 0 } ^ { \infty } D _ { m } ( u ) \varphi ( s u ) d u + 4 s ^ { 4 } \int _ { 0 } ^ { \infty } u ^ { 2 } D _ { m } ( u ) \varphi ( s u ) d u } \\ & { = 4 s ^ { 2 } \displaystyle \int _ { 0 } ^ { \infty } ( s ^ { 2 } u ^ { 2 } - 3 ) D _ { m } ( u ) \varphi ( s u ) d u } \\ & { = - 4 s ^ { 2 } \displaystyle \int _ { 0 } ^ { \infty } ( 1 - s ^ { 2 } u ^ { 2 } ) D _ { m } ( u ) \varphi ( s u ) d u - 8 s ^ { 2 } \displaystyle \int _ { 0 } ^ { \infty } D _ { m } ( u ) \varphi ( s u ) d u } \\ & { \overset { ( c ) } { = } - 4 s ^ { 2 } \displaystyle \int _ { 0 } ^ { \infty } D _ { m } ( u ) \frac { d } { d u } ( u \varphi ( s u ) ) d u - 8 s ^ { 2 } \displaystyle \int _ { 0 } ^ { \infty } D _ { m } ( u ) \varphi ( s u ) d u } \\ & { = - 4 s ^ { 2 } [ D _ { m } ( u ) u \varphi ( s u ) | _ { 0 } ^ { \infty } - \displaystyle \int _ { 0 } ^ { \infty } u D _ { m } ^ { \prime } ( u ) \varphi ( s u ) d u ] - 8 s ^ { 2 } \displaystyle \int _ { 0 } ^ { \infty } D _ { m } ( u ) \varphi ( s u ) d v } \\ & { = 4 s ^ { 2 } \displaystyle \int _ { 0 } ^ { \infty } [ u D _ { m } ^ { \prime } ( u ) - 2 D _ { m } ( u ) ] \varphi ( s u ) d u } \end{array}
$$

where in (\*) we used the fact that

$$
\frac { d } { d u } \left( u \varphi ( s u ) \right) = ( 1 - s ^ { 2 } u ^ { 2 } ) \varphi ( u s ) .
$$

Integrating by parts we have

$$
\begin{array} { l } { \displaystyle \int [ u D _ { m } ^ { \prime } ( u ) - 2 D _ { m } ( u ) ] d u } \\ { = u D _ { m } ( u ) - 3 \int D _ { m } ( u ) d u } \\ { = u D _ { m } ( u ) - 3 \displaystyle \sum _ { j = 0 } ^ { \infty } \int \left[ \big ( u - ( j + \frac 1 2 ) \big ) _ { + } - \big ( u - m - ( j + \frac 1 2 ) \big ) _ { + } \right] d u } \\ { = u D _ { m } ( u ) - \frac 3 2 \displaystyle \sum _ { j = 0 } ^ { \infty } \left[ \big ( u - ( j + \frac 1 2 ) \big ) _ { + } ^ { 2 } - \big ( u - m - ( j + \frac 1 2 ) \big ) _ { + } ^ { 2 } \right] . } \end{array}\tag{23}
$$

Using again integration by parts, we get that

$$
\begin{array} { l } { \displaystyle \partial _ { s } \bar { \mathcal { E } } ^ { \infty } ( m , s ) } \\ { \displaystyle = - 4 s ^ { 2 } \int _ { 0 } ^ { \infty } \left( u D _ { m } ( u ) - \frac { 3 } { 2 } \sum _ { j = 0 } ^ { \infty } \left[ \left( u - ( j + \frac 1 2 ) \right) _ { + } ^ { 2 } - \left( u - m - ( j + \frac 1 2 ) \right) _ { + } ^ { 2 } \right] \right) \frac d { d u } \varphi ( s u ) d u } \\ { \displaystyle = 4 s ^ { 4 } \int _ { 0 } ^ { \infty } u P _ { m } ( u ) \varphi ( s u ) d u } \end{array}
$$

where $\begin{array} { r } { P _ { m } ( u ) : = u D _ { m } ( u ) - \frac { 3 } { 2 } \sum _ { j = 0 } ^ { \infty } \Big [ \left( u - ( j + \frac { 1 } { 2 } ) \right) _ { + } ^ { 2 } - \left( u - m - ( j + \frac { 1 } { 2 } ) \right) _ { + } ^ { 2 } \Big ] } \end{array}$ . Now we notice that, for $\begin{array} { r } { 0 < u < m + \frac { 1 } { 2 } } \end{array}$ , it holds

$$
\begin{array} { c l } { { } } & { { { \displaystyle P _ { m } ( u ) = \sum _ { j = 0 } ^ { \infty } u \left( u - ( j + \frac { 1 } { 2 } ) \right) _ { + } - u \left( u - m - ( j + \frac { 1 } { 2 } ) \right) _ { + } - \frac { 3 } { 2 } \left( u - ( j + \frac { 1 } { 2 } ) \right) _ { + } ^ { 2 } } } } \\ { { } } & { { { \displaystyle + \frac { 3 } { 2 } \left( u - m - ( j + \frac { 1 } { 2 } ) \right) _ { + } ^ { 2 } } } } \\ { { } } & { { { \displaystyle = \sum _ { j = 0 } ^ { \infty } \left( u - \frac { 3 } { 2 } \left( u - ( j + \frac { 1 } { 2 } ) \right) _ { + } \right) \left( u - j - \frac { 1 } { 2 } \right) _ { + } } } } \\ { { } } & { { { \displaystyle = \sum _ { j = 0 } ^ { \lfloor u - 1 / 2 \rfloor } \left( u - \frac { 3 } { 2 } \left( u - j - \frac { 1 } { 2 } \right) \right) \left( u - j - \frac { 1 } { 2 } \right) } } } \\ { { } } & { { { \displaystyle \mid \sum _ { j = 0 } ^ { \lfloor u - 1 / 2 \rfloor + 1 } \left( - \frac { u } { 2 } + \frac { 3 } { 2 } j + \frac { 3 } { 4 } \right) \left( u - j - \frac { 1 } { 2 } \right) } } } \\ { { } } & { { { \displaystyle = \sum _ { j = 0 } ^ { \lfloor u - 1 / 2 \rfloor + 1 } \left( - \frac { u } { 2 } + \frac { 3 } { 2 } j + \frac { 3 } { 4 } \right) \left( u - j - \frac { 1 } { 2 } \right) , } } } \end{array}
$$

where $\textstyle \sum _ { j = 0 } ^ { \lfloor u - 1 / 2 \rfloor + }$ denotes the sum over the indices $j \geq 0$ with $u - j - { \textstyle { \frac { 1 } { 2 } } } > 0$ (empty if $u < \frac { 1 } { 2 } )$ . Writing $n : = \lfloor u - 1 / 2 \rfloor _ { + } + 1$ for the number of such indices, the last sum equals $\begin{array} { r } { \frac { n } { 2 } \left( \frac { 1 } { 4 } - ( u - n ) ^ { 2 } \right) } \end{array}$ , which is nonnegative since $\begin{array} { r } { | u - n | \leq \frac { 1 } { 2 } ; } \end{array}$ hence $P _ { m } \ge 0 \mathrm { o n } ( 0 , m + \textstyle { \frac { 1 } { 2 } } )$ and it vanishes exactly at the half-integers.

On the other hand, for all $m > { \frac { 1 } { 2 } } , u > m + 1$ , calling $n = \lfloor u - 1 / 2 \rfloor _ { + } + 1$ and $r =$ $\lfloor u - m - 1 / 2 \rfloor _ { + } + 1$ we have that

$$
\begin{array} { r l } { \exp _ { \tau } \exp _ { \tau } \left( - \frac { 1 } { 2 } \right) } & { = \left( \frac { 1 } { 2 } - \gamma \right) ^ { 2 } , \quad \forall \left( \frac { 1 } { 2 } - \gamma \right) ^ { 2 } , \quad \forall \left( \frac { 1 } { 2 } - \gamma \right) ^ { 3 } , \quad \forall \left( \frac { 1 } { 2 } - \gamma \right) ^ { 2 } , } \\ & { = \left( \frac { 1 } { 2 } - \gamma \right) ^ { 2 } , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad }  \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad }  \\ & { = \left( \frac { 1 } { 2 } - \gamma \right) ^ { 2 } \left( - \frac { 1 } { 2 } - \gamma \right) ^ { 2 } \left( \left( - \frac { 1 } { 2 } - \gamma \right) \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { = \left( \frac { 1 } { 2 } - \gamma \right) ^ { 2 } \left( - \frac { 1 } { 2 } - \gamma \right) ^ { 2 } \left( \frac { 1 } { 2 } - \frac { 1 } { 2 } \right) \left( \frac { 1 } { 2 } - \gamma \right) ^ { 2 } \left( \frac { 1 } { 2 } - \frac { 1 } { 2 } \right) , } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \ \end{array}
$$

where in (\*) we have noticed that

$$
\begin{array} { c } { { \displaystyle - \frac { 1 } { 2 } \le u - m - r = u - m - \left( \lfloor u - m - 1 / 2 \rfloor _ { + } + 1 \right) \le \frac 1 2 } } \\ { { \displaystyle \qquad \Longrightarrow \ \frac { r } { 2 } \left( \frac { 1 } { 4 } - \left( u - m - r \right) ^ { 2 } \right) \ge 0 } } \\ { { \displaystyle \left( r ( u - m ) - \frac { r ^ { 2 } } { 2 } \right) = \sum _ { j = 0 } ^ { r - 1 } \left( u - m - j - \frac 1 2 \right) \ge u - m - \frac 1 2 , } } \end{array}
$$

and in the following step we used $\begin{array} { r } { n \leq u + \frac { 1 } { 2 } } \end{array}$ . We consider now what happens when $\begin{array} { l c r } { { m + \frac 1 2 < u < \operatorname* { m i n } \{ m + 1 , \lfloor m \rfloor + \frac 3 2 \} } } \end{array}$ . On this interval, it holds

$$
\begin{array} { l } { { \displaystyle { D _ { m } ( u ) = \sum _ { j = 0 } ^ { \infty } \left[ \left( u - ( j + \frac { 1 } { 2 } ) \right) _ { + } - \left( u - m - ( j + \frac { 1 } { 2 } ) \right) _ { + } \right] } } } \\ { { \mathrm { } } } \\ { { \displaystyle { \quad = \sum _ { j = 0 } ^ { \lfloor m \rfloor } ( u - j - \frac { 1 } { 2 } ) - ( u - m - \frac { 1 } { 2 } ) } } } \\ { { \mathrm { } } } \\ { { \displaystyle { \quad = \left( \lfloor m \rfloor + 1 \right) u - \frac { \left( \lfloor m \rfloor + 1 \right) ^ { 2 } } { 2 } - u + m + \frac { 1 } { 2 } } } } \\ { { \mathrm { } } } \\ { { \displaystyle { \quad = \lfloor m \rfloor u - \frac { \lfloor m \rfloor \left( \lfloor m \rfloor + 2 \right) } { 2 } + m } } } \end{array}
$$

and, in particular $D _ { m } ^ { \prime } ( u ) = \lfloor m \rfloor$ . Note also that

$$
\frac { d } { d u } \sum _ { j = 0 } ^ { \infty } \Big [ \left( u - ( j + \textstyle { \frac { 1 } { 2 } } ) \right) _ { + } ^ { 2 } - \left( u - m - ( j + \textstyle { \frac { 1 } { 2 } } ) \right) _ { + } ^ { 2 } \Big ] = 2 D _ { m } ( u )
$$

Let us compute

$$
\begin{array} { r l } & { \frac { d } { d u } P _ { m } ( u ) = D _ { m } ( u ) + u D _ { m } ^ { \prime } ( u ) - 3 D _ { m } ( u ) } \\ & { \qquad = u D _ { m } ^ { \prime } ( u ) - 2 D _ { m } ( u ) } \\ & { \qquad = u | m | - 2 \left( | m | \displaystyle u - \frac { | m | ( | m | + 2 ) } { 2 } + m \right) } \\ & { \qquad = | m | ( | m | + 2 - u ) - 2 m } \\ & { \qquad \le | m | ( | m | + \frac { 3 } { 2 } - m ) - 2 m } \\ & { \qquad = | m | \left( \frac { 3 } { 2 } - ( m - | m | ) \right) - 2 | m | - 2 ( m - | m | ) } \\ & { \qquad = - \frac { | m | } { 2 } - ( | m | + 2 ) ( m - | m | ) < 0 } \end{array}
$$

for all $m > 0$ . We notice that $\begin{array} { r } { P _ { m } ( m + \frac { 1 } { 2 } ) = \frac { \lfloor m \rfloor + 1 } { 2 } ( m - \lfloor m \rfloor ) ( \lfloor m \rfloor + 1 - m ) } \end{array}$ is positive for non-integer m and 0 for integer m.

Now, it follows that

$$
- \mathrm { I f } \mathrm { m i n } \{ m + 1 , \lfloor m \rfloor + \frac { 3 } { 2 } \} = m + 1 , \mathrm { t h e n }
$$

$$
P _ { m } ( m + 1 ) \leq \frac { 3 } { 1 6 } ( 1 - 2 m ) < 0 .
$$

$$
- \mathrm { ~ I f ~ } \operatorname* { m i n } \{ m + 1 , \lfloor m \rfloor + \frac { 3 } { 2 } \} = \lfloor m \rfloor + \frac { 3 } { 2 } , \mathrm { t h e n ~ }
$$

$$
\begin{array} { r l } { \operatorname* { P r } _ { 0 \leq t } \{ \ln | \pm 1 + \frac { q } { q } | = \frac { 1 } { 2 } \{ \ln | \pm 1 + \frac { q } { q } \} \} \frac { \sum _ { i = 1 } ^ { n } ( \ln | \pm 1 + 1 - \frac { q } { q } | - 1 ( \ln | \pm 1 + 1 - \ln - \frac { \bar { \eta } } { 2 } |  ]  } { -    ( \ln | \pm 1 + 1 - \ln - \frac { \bar { \eta } } { 2 } |  ]  } } \\ { - \frac { 2 } { q } \sum _ { i = 1 } ^ { n } \{ [ ( 1 - 1 - \lambda ) \frac { q } { q } ] - \frac { 1 } { 2 } [ \ln | \pm 1 + 1 - \ln - \bar { \eta } _ { i }  ]  } \\ {   - ( \ln | \pm 1 ) \frac { \sqrt { \eta } } { q } \} \frac { [ ( \ln | \pm 1  ]   } {   \sqrt { \eta } }  } \\  - \frac { 4 } { q } \{ \ln | \pm 1 + 1 - \frac { q } { q } \} \} \frac { [ ( \ln | \pm 1  ] ) -   \sqrt { \eta } - ( \ln | \pm 1 + 1 - \ln \frac { \bar { \eta } } { 2 } ) ] } {     ( \ln | \pm 1 + 1 - \frac { q } { q } )   } \\  - \frac { 4 } { q } \frac { [ \ln | \pm 1  ] \} { q } \frac { [ \ln | \pm 1  ] ( \ln | \pm 1 + \frac { q } { q } ) } { -   \sqrt { \eta } - ( \ln | \pm 1 + 1 - \ln  ) ] } } \\  - \frac { 4 } { q } \frac  [ \ln | \pm 1  ] ( \ln | \pm  \end{array}
$$

where in the second equality we used that $0 < \lfloor m \rfloor + 1 - m \leq 1$ , so that $( \lfloor m \rfloor + 1 -$ $m - j ) .$ vanishes for $j \geq 1 ,$ , and the last inequality holds since $\lfloor m \rfloor + 1 - { \bar { m } } { \bar { > } } 0$ and $3 m - \lfloor m \rfloor \geq 2 m > 0 .$

It remains to check what happens for $\begin{array} { r } { \lfloor m \rfloor + \frac { 3 } { 2 } \leq u \leq m + 1 } \end{array}$ . On the interval, it holds

$$
\begin{array} { r l } { \sum _ { k \in \{ 0 , 1 \} } \sum _ { i = 1 } ^ { \infty } \sum _ { j = 1 } ^ { \infty } \Big | \frac { 1 } { \beta } \Big | ( \partial _ { t } \phi - \partial _ { t } \phi ) - \frac { 1 } { \beta } \Big | ^ { 2 } \Big | } & { } \\ { \sum _ { k \in \{ 0 , 1 \} } ^ { \infty } \sum _ { i = 1 } ^ { \infty } \Big | \frac { 1 } { \beta } \sum _ { k \in \{ 0 , 1 \} } ^ { i } \frac { 1 } { \beta } \Big | \frac { 1 } { \beta } \Big | \Delta _ { t } ^ { \beta } = \beta - \frac { 1 } { \beta } \Big | \frac { 1 } { \beta } \Big | } & { } \\ { \sum _ { k \in \{ 0 , 1 \} } ^ { \infty } \Big [ \sin ( \phi - \phi ) - \frac { 1 } { 2 } \sin ^ { 2 } ( \phi - \phi ) ^ { 2 } - \sin ^ { 2 } ( \phi - \phi ) ^ { 2 } \Big ] \Big | ^ { 2 } } & { } \\ { \frac { 1 } { \beta } \Big | ( \partial _ { t } \phi - \partial _ { t } \phi ) - \frac { 1 } { \beta } \Big | ^ { 2 } \Big | ( \partial _ { t } \phi - \partial _ { t } \phi ) \Big | ^ { 2 } } & { } \\ { - \frac { 1 } { \beta } \Big ( \sin ( \phi - \phi ) \Big ) ^ { 2 } \Big | ^ { 2 } \Big | ( \partial _ { t } \phi - \partial _ { t } \phi ) - \frac { 1 } { \beta } \Big | ^ { 2 } \Big | ( \partial _ { t } \phi - \partial _ { t } \phi ) \Big | ^ { 2 } } & { } \\ { - ( \sin ( \phi - \phi ) \Big ) ^ { 2 } \Big | ^ { 2 } \Big | ^ { 2 } \Big | ( \partial _ { t } \phi - \partial _ { t } \phi ) - \frac { 1 } { \beta } \Big | ^ { 2 } \Big | ^ { 2 } \Big | ^ { 2 } } & { } \\  - \frac { 1 } { \beta } \cos ( \phi - \phi ) - \sin ( \phi - \phi ) \Big | ^   \end{array}
$$

where in (\*) we used

$$
\begin{array} { l } { \displaystyle 0 \leq u - \lfloor m \rfloor - \frac 3 2 = ( u - m - \frac 1 2 ) - ( \lfloor m \rfloor + 1 - m ) } \\ { \displaystyle \quad \leq ( u - m - \frac 1 2 ) - ( \lfloor m \rfloor + 1 - m ) ( u - m - \frac 1 2 ) } \\ { \displaystyle = [ 1 - ( \lfloor m \rfloor + 1 - m ) ] ( u - m - \frac 1 2 ) } \\ { \displaystyle = ( m - \lfloor m \rfloor ) ( u - m - \frac 1 2 ) } \end{array}
$$

together with

$$
\begin{array} { l } { 0 \leq \lfloor m \rfloor + \frac 5 2 - u \leq 1 } \\ { m + \frac 3 2 - u \geq m + \displaystyle \frac 3 2 - ( m + 1 ) = \displaystyle \frac 1 2 . } \end{array}
$$

Therefore, for all $m > 1 / 2 , P _ { m }$ has a unique zero $\begin{array} { c } { { u _ { m } ^ { * } \ \mathrm { i n } \left[ m + \frac { 1 } { 2 } , \mathrm { m i n } \{ m + 1 , \lfloor m \rfloor + \frac { 3 } { 2 } \} \right] } } \end{array}$ and moreover $P _ { m } \ge 0 \mathrm { o } \dot { \mathrm { \Omega } } ( 0 , u _ { m } ^ { \ast } )$ and $P _ { m } < 0$ on $( u _ { m } ^ { * } , \infty )$ . Then

$$
\begin{array} { l } { \displaystyle \partial _ { s } \bar { \mathcal { E } } ^ { \infty } ( m , s ) = 4 s ^ { 4 } \int _ { 0 } ^ { \infty } u P _ { m } ( u ) \varphi ( s u ) d u } \\ { \displaystyle = \frac { 4 s ^ { 4 } } { \sqrt { 2 \pi } } e ^ { - s ^ { 2 } { u _ { m } ^ { * } } ^ { 2 } / 2 } \int _ { 0 } ^ { \infty } u P _ { m } ( u ) e ^ { - s ^ { 2 } ( u ^ { 2 } - { u _ { m } ^ { * } } ^ { 2 } ) / 2 } d u } \end{array}
$$

Notice now that $\frac { 4 s ^ { 4 } } { \sqrt { 2 \pi } } e ^ { - s ^ { 2 } { u _ { m } ^ { * } } ^ { 2 } / 2 } > 0$ for all $m , s ,$ and $\begin{array} { r } { \int _ { 0 } ^ { \infty } u P _ { m } ( u ) e ^ { - s ^ { 2 } ( u ^ { 2 } - { u _ { m } ^ { * } } ^ { 2 } ) / 2 } d u } \end{array}$ is strictly increasing in $s ,$ since its derivative in s is $\begin{array} { r } { \int _ { 0 } ^ { \infty } u s ( u _ { m } ^ { * ^ { 2 } } - u ^ { 2 } ) P _ { m } ( u ) e ^ { - s ^ { 2 } ( u ^ { 2 } - { u _ { m } ^ { * } } ^ { 2 } ) / 2 } d u } \end{array}$ and the integrand is nonnegative and not identically zero, as $( u _ { m } ^ { * ^ { 2 } } - u ^ { 2 } )$ and $P _ { m } ( u )$ have

the same sign. Therefore, for all $m > 1 / 2 , s \mapsto \bar { \mathcal { E } } _ { s } ^ { \infty } ( m , s )$ has at most one zero. For all $m > 1 / 2$ , a minimum exists because

$$
\begin{array} { r l } & { \bar { \mathcal { E } } ^ { \infty } ( m , s ) = 1 - 4 s ^ { 3 } \displaystyle \int _ { 0 } ^ { \infty } D _ { m } ( u ) \varphi ( s u ) d u } \\ & { \qquad \le 1 - 4 s ^ { 3 } \displaystyle \int _ { 1 / 2 } ^ { \infty } D _ { m } ( u ) \varphi ( s u ) d u } \\ & { \qquad \le 1 - 4 s ^ { 3 } \displaystyle \int _ { m + \frac { 1 } { 2 } } ^ { \infty } m \varphi ( s u ) d u } \\ & { \qquad < 1 } \end{array}
$$

while lim $\begin{array} { r } { \iota _ { s \to 0 } \bar { \mathcal { E } } ^ { \infty } ( m , s ) = \operatorname* { l i m } _ { s \to \infty } \bar { \mathcal { E } } ^ { \infty } ( m , s ) = 1 ; } \end{array}$ indeed, $D _ { m } ( u ) = 0$ for $\begin{array} { r } { u < \frac { 1 } { 2 } } \end{array}$ and $D _ { m } ( u ) \leq m ( u + \frac { 1 } { 2 } )$ , so that $\begin{array} { r } { 4 s ^ { 3 } \int _ { 0 } ^ { \infty } D _ { m } ( u ) \varphi ( s u ) d u \ \leq \ 4 m s ^ { 3 } \int _ { 1 / 2 } ^ { \infty } ( u + \frac { 1 } { 2 } ) \varphi ( s \bar { u } ) d u , } \end{array}$ which tends to 0 both as $s \to 0$ and as $s \to \infty$

We note now that at the critical point $s ^ { * } ( m )$ we have

$$
\begin{array} { l } { { 0 = \bar { \mathcal { E } } _ { s } ^ { \infty } ( m , s ^ { * } ( m ) ) } } \\ { { { } = \displaystyle \frac { 4 s ^ { * } ( m ) ^ { 4 } } { \sqrt { 2 \pi } } e ^ { - s ^ { * ^ { 2 } } ( m ) { u _ { m } ^ { * } } ^ { 2 } / 2 } \int _ { 0 } ^ { \infty } u P _ { m } ( u ) e ^ { - s ^ { * ^ { 2 } } ( m ) ( { u ^ { 2 } } - { u _ { m } ^ { * } } ^ { 2 } ) / 2 } d u } } \\ { { { } \Longrightarrow \displaystyle \int _ { 0 } ^ { \infty } u P _ { m } ( u ) e ^ { - s ^ { * } ( m ) ^ { 2 } ( { u ^ { 2 } } - { u _ { m } ^ { * } } ^ { 2 } ) / 2 } d u = 0 . } } \end{array}
$$

Finally, at $s ^ { * } ( m )$ , it holds

$$
\begin{array} { l } { \displaystyle \bar { \mathcal { E } } _ { s s } ^ { \infty } ( m , s ^ { * } ( m ) ) } \\ { \displaystyle = \frac { 4 s ^ { * } ( m ) ^ { 4 } } { \sqrt { 2 \pi } } e ^ { - s ^ { * ^ { 2 } } ( m ) u _ { m } ^ { * ^ { 2 } } / 2 } \frac { d } { d s } \left( \int _ { 0 } ^ { \infty } u P _ { m } ( u ) e ^ { - s ^ { 2 } ( u ^ { 2 } - u _ { m } ^ { * ^ { 2 } } ) / 2 } d u \right) \Big | _ { s = s ^ { * } ( m ) } } \\ { \displaystyle = \frac { 4 s ^ { * } ( m ) ^ { 4 } } { \sqrt { 2 \pi } } e ^ { - s ^ { * ^ { 2 } } ( m ) u _ { m } ^ { * ^ { 2 } } / 2 } \int _ { 0 } ^ { \infty } u s ^ { * } ( m ) ( { u _ { m } ^ { * ^ { 2 } } - u ^ { 2 } } ) P _ { m } ( u ) e ^ { - s ^ { * ^ { 2 } } ( m ) ( u ^ { 2 } - { u _ { m } ^ { * ^ { 2 } } } ) / 2 } d u } \\ { \displaystyle > 0 } \end{array}
$$

since $( { u _ { m } ^ { * } } ^ { 2 } - { u ^ { 2 } } )$ and $P _ { m } ( u )$ have the same sign for all $u \in ( 0 , \infty )$ , and their product is not identically zero.

(iii) The function $z \mapsto \left( z - s \lfloor z / s \rceil \right) ^ { 2 }$ is even and s-periodic, and equals $z ^ { 2 } \ { \mathrm { o n } } \ ( - { \frac { s } { 2 } } , { \frac { s } { 2 } } )$ ; its Fourier expansion is therefore the rescaled expansion of $x ^ { 2 } \mathrm { o n } \left( - \pi , \pi \right)$ , namely

$$
\bigl ( z - s \lfloor z / s \rceil \bigr ) ^ { 2 } = \frac { s ^ { 2 } } { 1 2 } + \frac { s ^ { 2 } } { \pi ^ { 2 } } \sum _ { k = 1 } ^ { \infty } \frac { ( - 1 ) ^ { k } } { k ^ { 2 } } \cos { \Bigl ( \frac { 2 \pi k z } { s } \Bigr ) } ,
$$

with absolute and uniform convergence. Taking expectations with respect to $Z \sim { \mathcal { N } } ( 0 , 1 )$ and using $\mathbb { E } [ \cos ( t Z ) ] = e ^ { - t ^ { 2 } / 2 }$ with $t = 2 \pi k / s$ yields

$$
\begin{array} { l } { { \displaystyle { \mathcal J } _ { \infty } ( s ) = { \mathbb E } \Big [ \big ( Z - s \lfloor Z / s \rfloor \big ) ^ { 2 } \Big ] = \frac { s ^ { 2 } } { 1 2 } + \frac { s ^ { 2 } } { \pi ^ { 2 } } \displaystyle \sum _ { k = 1 } ^ { \infty } \frac { ( - 1 ) ^ { k } } { k ^ { 2 } } e ^ { - 2 \pi ^ { 2 } k ^ { 2 } / s ^ { 2 } } } } \\ { { \displaystyle \quad = \frac { s ^ { 2 } } { 1 2 } + \frac { s ^ { 2 } } { \pi ^ { 2 } } \displaystyle \sum _ { k = 1 } ^ { \infty } \frac { ( - 1 ) ^ { k } } { k ^ { 2 } } e ^ { - k ^ { 2 } \tau ( s ) } , } } \end{array}
$$

where $\textstyle \tau ( s ) : = { \frac { 2 \pi ^ { 2 } } { s ^ { 2 } } }$ . Therefore, for all $s > 0$ , we have that

$$
\begin{array} { r l } { \mathcal { J } _ { \mathrm { c s } } ^ { \prime } ( s ) = \frac { s } { 6 } + \frac { 1 } { \pi ^ { 2 } } \displaystyle \sum _ { k = 1 } ^ { \infty } \frac { ( - 1 ) ^ { k ! } } { k ^ { 2 } } \frac { d } { d s } \left( s ^ { 2 } e ^ { - k ^ { 2 } \tau ( s ) } \right) } \\ & { = \frac { s } { 6 } + \frac { 1 } { \pi ^ { 2 } } \displaystyle \sum _ { k = 1 } ^ { \infty } \frac { ( - 1 ) ^ { k ! } } { k ^ { 2 } } \left[ 2 s e ^ { - k ^ { 2 } \tau ( s ) } - s ^ { 2 } e ^ { - k ^ { 2 } \tau ( s ) } k ^ { 2 } \tau ^ { \prime } ( s ) \right] } \\ & { = \frac { s } { 6 } + \frac { 1 } { \pi ^ { 2 } } \displaystyle \sum _ { k = 1 } ^ { \infty } \frac { ( - 1 ) ^ { k ! } } { k ^ { 2 } } \left[ 2 s e ^ { - k ^ { 2 } \tau ( s ) } + s ^ { 2 } e ^ { - k ^ { 2 } \tau ( s ) } \frac { 4 \pi ^ { 2 } k ^ { 2 } } { s ^ { 3 } } \right] } \\ & { = \frac { s } { 6 } + \frac { 1 } { \pi ^ { 2 } } \displaystyle \sum _ { k = 1 } ^ { \infty } \frac { ( - 1 ) ^ { k ! } } { k ^ { 2 } } 2 s \left[ 1 + \frac { 2 \pi ^ { 2 } k ^ { 2 } } { s ^ { 2 } } \right] e ^ { - k ^ { 2 } \tau ( s ) } } \\ & { = \frac { s } { 6 } + \frac { 1 } { \pi ^ { 2 } } \displaystyle \sum _ { k = 1 } ^ { \infty } \left( - 1 \right) ^ { k } 2 s \left[ \frac { 1 } { k ^ { 2 } } + \tau ( s ) \right] e ^ { - k ^ { 2 } \tau ( s ) } } \\ & { < \frac { s } { 6 } , } \end{array}
$$

since $\begin{array} { r } { \left\lceil \frac { 1 } { k ^ { 2 } } + \tau ( s ) \right\rceil e ^ { - k ^ { 2 } \tau ( s ) } } \end{array}$ is strictly decreasing to 0 in k an $\displaystyle 1 - ( 1 + \tau ) e ^ { - \tau } < 0$ . Recall now that $g ( \ddot { x } ) = \varphi ( x ) \dot { - } x ( 1 - \Phi ( x ) )$ and $g ^ { \prime } ( x ) = - x \varphi ( x ) - ( 1 - \Phi ( x ) ) + x \varphi ( x ) = - ( 1 - \Phi ( x ) )$ hence

$$
\begin{array} { l } { \displaystyle \bar { \mathcal { E } } _ { s } ^ { \infty } ( m , s ) = \mathcal { I } _ { \infty } ^ { \prime } ( s ) + 4 \sum _ { j = 0 } ^ { \infty } \left[ \varphi \left( \left( m + \frac 1 2 + j \right) s \right) - \right. } \\ { \displaystyle \left. 2 s \left( m + \frac 1 2 + j \right) \left( 1 - \Phi \left( \left( m + \frac 1 2 + j \right) s \right) \right) \right] . } \end{array}\tag{24}
$$

We note that, whenever $y \geq { \sqrt { 2 } }$ , we have

$$
\varphi ( y ) - 2 y ( 1 - \Phi ( y ) ) < { \frac { 1 - y ^ { 2 } } { 1 + y ^ { 2 } } } \varphi ( y ) \leq - { \frac { 1 } { 3 } } \varphi ( y )\tag{25}
$$

where we have used Lemma 12. $\begin{array} { r } { \mathrm { A t } \ s = \frac { \sqrt { 2 } } { m + \frac { 1 } { 2 } } } \end{array}$ , it holds $\begin{array} { r } { ( m + \frac { 1 } { 2 } + j ) \frac { \sqrt { 2 } } { m + \frac { 1 } { 2 } } \geq \sqrt { 2 } } \end{array}$ for all $j \geq 0$ . Therefore, combining the fact that alí the terms in the sum appearing in (24) are negative and (25), and keeping only the term $j = 0$ , we obtain

$$
\bar { \mathcal { E } } _ { s } ^ { \infty } \left( m , \frac { \sqrt { 2 } } { m + \frac { 1 } { 2 } } \right) < \frac { \sqrt { 2 } } { 6 m + 3 } - \frac { 4 } { 3 } \varphi ( \sqrt { 2 } ) \leq \frac { \sqrt { 2 } } { 9 } - \frac { 4 } { 3 } \varphi ( \sqrt { 2 } ) < 0
$$

for all $m \geq 1$ , since $\frac { \sqrt { 2 } } { 9 }$ and $\begin{array} { r } { { \frac { 4 } { 3 } } \varphi ( { \sqrt { 2 } } ) = { \frac { 4 } { 3 } } { \frac { e ^ { - 1 } } { \sqrt { 2 \pi } } } } \end{array}$ . By (ii) of Proposition 1, the function $\bar { \mathcal { E } } _ { s } ^ { \infty } ( m , \cdot )$ has a unique zero in $s ^ { * } ( m )$ with $\bar { \mathcal { E } } _ { s s } ^ { \infty } ( m , s ^ { * } ( m ) ) > 0$ . By continuity it holds that

$$
\begin{array} { l l } { \bar { \mathcal { E } } _ { s } ^ { \infty } ( m , s ) < 0 \qquad s < s ^ { * } ( m ) } \\ { \bar { \mathcal { E } } _ { s } ^ { \infty } ( m , s ) > 0 \qquad s > s ^ { * } ( m ) , } \end{array}
$$

and, since $\begin{array} { r } { \bar { \mathcal { E } } _ { s } ^ { \infty } \left( m , \frac { \sqrt { 2 } } { m + \frac { 1 } { 2 } } \right) \ < \ 0 . } \end{array}$ , we know that $\frac { \sqrt { 2 } } { m + \frac { 1 } { 2 } } < s ^ { * } ( m )$ . Let us call now $y _ { j } =$ $( m + j + { \textstyle \frac { 1 } { 2 } } ) s ^ { * } ( m )$ . Since $\begin{array} { r } { y _ { j } = ( m + j + \frac { 1 } { 2 } ) s ^ { * } ( m ) > \frac { \sqrt { 2 } } { m + \frac { 1 } { 2 } } ( m + j + \frac { 1 } { 2 } ) \geq \sqrt { 2 } , } \end{array}$ for all $j \geq 0$ , we have that

$$
y _ { j } \varphi ( y _ { j } ) - 2 \bigl ( 1 - \Phi ( y _ { j } ) \bigr ) \stackrel { ( * ) } { > } \left( y _ { j } - \frac { 2 } { y _ { j } } \right) \varphi ( y _ { j } ) > 0 ,
$$

where in $( ^ { \ast } )$ we have used again Lemma 12. Since $g ^ { \prime } = - ( 1 - \Phi )$ , we have $\bar { \mathcal { E } } _ { m } ^ { \infty } ( m , s ) =$ $- 4 s ^ { 2 } \textstyle \sum _ { j = 0 } ^ { \infty } \left( 1 - \Phi \big ( ( m + ) + \frac { 1 } { 2 } ) s \right) \big )$ , and differentiating in s gives $\bar { \mathcal { E } } _ { m s } ^ { \infty } ( m , s ) =$

4s $\begin{array} { r } { \sum _ { j = 0 } ^ { \infty } \left[ x _ { j } \varphi ( x _ { j } ) - 2 ( 1 - \Phi ( x _ { j } ) ) \right] } \end{array}$ with $\begin{array} { r } { x _ { j } = ( m + j + \frac { 1 } { 2 } ) s } \end{array}$ . Therefore,

$$
\bar { \mathcal { E } } _ { m s } ^ { \infty } ( m , s ^ { * } ( m ) ) = 4 s ^ { * } ( m ) \sum _ { j = 0 } ^ { \infty } \left[ y _ { j } \varphi ( y _ { j } ) - 2 ( 1 - \Phi ( y _ { j } ) ) \right] > 0 .\tag{26}
$$

By the implicit function theorem, since $\bar { \mathcal { E } } _ { s s } ^ { \infty } ( m , s ^ { * } ( m ) ) > 0$ for every $m \geq 1$ , we have that $s ^ { * } ( \cdot ) \in \mathcal { C } ^ { \infty } ( [ 1 , \infty ) )$ ), and

$$
\frac { d s ^ { * } } { d m } ( m ) = - \frac { \bar { \xi } _ { m s } ^ { \infty } \left( m , s ^ { * } ( m ) \right) } { \bar { \xi } _ { s s } ^ { \infty } \left( m , s ^ { * } ( m ) \right) } .\tag{27}
$$

Combining (26), (27) and $\bar { \mathcal { E } } _ { s s } ^ { \infty } ( m , s ^ { * } ( m ) ) > 0$ , we conclude that

$$
\frac { d s ^ { * } } { d m } ( m ) = - \frac { \bar { \mathcal { E } } _ { m s } ^ { \infty } \big ( m , s ^ { * } ( m ) \big ) } { \bar { \mathcal { E } } _ { s s } ^ { \infty } \big ( m , s ^ { * } ( m ) \big ) } < 0 .
$$

□

## A.5.1 COMPUTATION OF $\kappa ( m )$ FOR $m > 0$ INTEGER

Let $y _ { j } : = j + { \frac { 1 } { 2 } }$ . We recall that

$$
\bar { \mathcal { E } } ^ { \infty } ( m , s ) = 1 - 4 \sum _ { j = 0 } ^ { \infty } s g ( y _ { j } s ) + 4 \sum _ { j = 0 } ^ { \infty } s g \big ( ( m + j + { \textstyle \frac { 1 } { 2 } } ) s \big ) ,
$$

where $g ( x ) = \varphi ( x ) - x { \bigl ( } 1 - \Phi ( x ) { \bigr ) }$ . Since we are interested in integer m, we note that the two series telescope, because $\begin{array} { r } { m + j + \frac { 1 } { 2 } = y _ { j + m } . } \end{array}$ and the expression reduces to the finite sum

$$
\mathcal { E } ^ { \infty } ( m , s ) = 1 - 4 s \sum _ { j = 0 } ^ { m - 1 } g ( y _ { j } s ) .
$$

Moreover, it holds $g ^ { \prime } ( x ) = - ( 1 - \Phi ( x ) )$ and $g ^ { \prime \prime } ( x ) = \varphi ( x )$ . Then, for all $c > 0 .$ , it holds that

$$
\begin{array} { c } { { \left[ s g ( c s ) \right] ^ { \prime } = g ( c s ) + c s g ^ { \prime } ( c s ) } } \\ { { { } } } \\ { { { } = \varphi ( c s ) - c s \bigl ( 1 - \Phi ( c s ) \bigr ) - c s \bigl ( 1 - \Phi ( c s ) \bigr ) } } \\ { { { } } } \\ { { { } = \varphi ( c s ) - 2 c s \bigl ( 1 - \Phi ( c s ) \bigr ) , } } \end{array}
$$

and

$$
\begin{array} { c } { { \left[ s g ( c s ) \right] ^ { \prime \prime } = c g ^ { \prime } ( c s ) + c g ^ { \prime } ( c s ) + c ^ { 2 } s g ^ { \prime \prime } ( c s ) } } \\ { { { } } } \\ { { { } = - 2 c \bigl ( 1 - \Phi ( c s ) \bigr ) + c ^ { 2 } s \varphi ( c s ) } } \\ { { { } } } \\ { { { } = c \Bigl [ c s \varphi ( c s ) - 2 \bigl ( 1 - \Phi ( c s ) \bigr ) \Bigr ] . } } \end{array}
$$

In particular, a direct computation leads to

$$
\begin{array} { c l } { { \partial _ { s } { \mathcal E } ^ { \infty } ( m , s ) = - 4 \displaystyle \sum _ { j = 0 } ^ { m - 1 } \left[ s g ( y _ { j } s ) \right] ^ { \prime } } } \\ { { = 4 \displaystyle \sum _ { j = 0 } ^ { m - 1 } \left[ 2 y _ { j } s \big ( 1 - \Phi ( y _ { j } s ) \big ) - \varphi ( y _ { j } s ) \right] . } } \end{array}\tag{28}
$$

and

$$
\begin{array} { l } { { \displaystyle { \mathcal E } _ { s s } ^ { \infty } ( m , s ) = - 4 \sum _ { j = 0 } ^ { m - 1 } \left[ s g ( y _ { j } s ) \right] ^ { \prime \prime } } \ ~ } \\ { { \displaystyle ~ = - 4 \sum _ { j = 0 } ^ { m - 1 } y _ { j } \left[ y _ { j } s \varphi ( y _ { j } s ) - 2 \big ( 1 - \Phi ( y _ { j } s ) \big ) \right] } , } \end{array}
$$

and, consequently,

$$
\begin{array} { l } { \displaystyle \kappa ( m , s ) = s ^ { 2 } \mathcal { E } _ { s s } ^ { \infty } ( m , s ) } \\ { \displaystyle = 4 s \sum _ { j = 0 } ^ { m - 1 } \left[ 2 y _ { j } s \big ( 1 - \Phi ( y _ { j } s ) \big ) - ( y _ { j } s ) ^ { 2 } \varphi ( y _ { j } s ) \right] . } \end{array}
$$

The following algorithm is used for the numerical evaluation of the scale-invariant curvature $\kappa ( M _ { B } , s ^ { * } ( \bar { M _ { B } } ) )$ reported in Figure 4.

Algorithm 1 Computation of $\kappa ( M _ { B } , s ^ { * } ( M _ { B } ) )$ for $B \in \{ 2 , 3 , 4 , 5 , 8 , 9 \}$ . The functions E\_s and kappa implement the finite sums of Section A.5.1; s\_star locates the unique zero of $\mathcal { E } _ { s } ^ { \infty } ( m , \cdot )$ by Brent's method on the bracket $[ \sqrt { 2 } / ( m + \textstyle { \frac { 1 } { 2 } } ) , 2 0 ]$

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
phi, Q = norm.pdf, norm.sf # Q = 1 − Phi
def E_s (m, s) :
t = (np.arange(m) + 0.5) * s
return -4 * np.sum(phi(t) − 2 * t * Q(t))
def kappa (m, s) :
t = (np.arange(m) + 0.5) * s
return 4 * s * np.sum(2 * t * Q(t) − t**2 * phi(t))
def s_star(m) :
1o, hi = np.sqrt(2) / (m + 0.5), 20.0
return brentq(lambda s: E_s(m, s), lo, hi, xtol=1e-12)
bits = [2, 3, 4, 5, 8, 9]
for B in bits:
m = 2 ** (B − 1) − 1
s = s_star(m)
print (B, m, s, kappa(m, s))
```

The following figure shows the scale-invariant curvature κ of the limiting objective $s \mapsto$ $\mathcal { E } ^ { \infty } ( M _ { B } , s )$ , evaluated at its minimizer $s ^ { * } ( M _ { B } )$ , as a function of the bit-width B. In particular, we note that κ decreases monotonically with B, so the objective is sharply peaked around the optimal scale at low precision and increasingly flat as B grows, which is why the choice of scale matters most at W2 and W3.

## A.5.2 WHY RELATIVE CURVATURE κ?

The scale-invariant curvature κ is a natural quantity for studying sensitivity to relative scale errors. To illustrate this, fix the bit-width and let $\bar { \vartheta } \sim \bar { \mathcal { N } } ( 0 , I _ { d } )$ and $\vartheta _ { \sigma } = \sigma \vartheta$ , with $\sigma > 0$ . Define the variance-normalized loss

$$
\mathcal { E } _ { \sigma } ( s ) : = \frac { 1 } { d \sigma ^ { 2 } } \mathbb { E } \Big [ \big \| \vartheta _ { \sigma } - s \lfloor \vartheta _ { \sigma } / s \big \rceil _ { B } \big \| _ { 2 } ^ { 2 } \Big ] .
$$

Since $\vartheta _ { \sigma } / s = \vartheta / ( s / \sigma )$ , we obtain

$$
\begin{array} { r } { \mathcal { E } _ { \sigma } ( s ) = \mathcal { E } _ { 1 } ( s / \sigma ) , \qquad s _ { \sigma } ^ { * } = \sigma s ^ { * } , } \end{array}
$$

where $s ^ { * }$ minimizes $\mathcal { E } _ { 1 }$ . Consequently, the same relative scale error ε produces the same variancenormalized loss:

$$
\begin{array} { r } { \mathcal { E } _ { \sigma } ( s _ { \sigma } ^ { * } ( 1 + \varepsilon ) ) = \mathcal { E } _ { 1 } ( s ^ { * } ( 1 + \varepsilon ) ) . } \end{array}
$$

Although the ordinary second derivative depends on σ,

$$
\mathcal { E } _ { \sigma } ^ { \prime \prime } ( s _ { \sigma } ^ { * } ) = \sigma ^ { - 2 } \mathcal { E } _ { 1 } ^ { \prime \prime } ( s ^ { * } ) ,
$$

![](images/3456e689df5eba939b3d9d6e550bf5eb0886e0df352c1bf6727b4886b68ca27b.jpg)  
Figure 4: Scale-invariant curvature $\kappa ( M _ { B } , s ^ { * } ( M _ { B } ) )$ of $\mathcal { E } ^ { \infty } ( M _ { B } , s ^ { * } ( M _ { B } ) )$ . κ is decreasing as B increases.

the scale-invariant curvature does not:

$$
\kappa = ( s _ { \sigma } ^ { * } ) ^ { 2 } { \mathcal { E } } _ { \sigma } ^ { \prime \prime } ( s _ { \sigma } ^ { * } ) = ( s ^ { * } ) ^ { 2 } { \mathcal { E } } _ { 1 } ^ { \prime \prime } ( s ^ { * } ) .
$$

It therefore governs the local loss increase under relative scale perturbations:

$$
\mathcal { E } _ { \sigma } ( s _ { \sigma } ^ { * } ( 1 + \varepsilon ) ) - \mathcal { E } _ { \sigma } ( s _ { \sigma } ^ { * } ) = \frac { \kappa } { 2 } \varepsilon ^ { 2 } + o ( \varepsilon ^ { 2 } ) .
$$

This invariance motivates using relative perturbations to assess sensitivity to scale choices, including those produced by the rules in Table 5.

In practice, the scale-selection rules in Table 5 return different positive scales for the same quantization setting. Their discrepancies can be expressed as relative deviations from a common reference scale:

$$
s _ { \mathrm { r u l e } } = s _ { \mathrm { r e f } } ( 1 + \varepsilon _ { \mathrm { r u l e } } ) .
$$

Relative perturbations therefore provide a common way to assess the effect of these differences across weight magnitudes. When the reference scale minimizes the smooth normalized loss, κ governs the resulting loss increase up to second order.

## A.6 TECHNICAL LEMMAS

The following elementary fact allows us to control the supremum of a quadratic polynomial over an interval by its values at three points.

Lemma 10. Let $\textstyle a < b , m : = { \frac { a + b } { 2 } }$ , and let p be a real polynomial of degree at most 2. Then

$$
\operatorname* { s u p } _ { s \in [ a , b ] } | p ( s ) | \leq 3 \operatorname* { m a x } \big \{ | p ( a ) | , | p ( m ) | , | p ( b ) | \big \} .
$$

Proof. Define the Lagrange basis polynomials associated with the nodes $a , m , b ,$

$$
L _ { a } ( s ) : = { \frac { ( s - m ) ( s - b ) } { ( a - m ) ( a - b ) } } , \qquad L _ { m } ( s ) : = { \frac { ( s - a ) ( s - b ) } { ( m - a ) ( m - b ) } } , \qquad L _ { b } ( s ) : = { \frac { ( s - a ) ( s - m ) } { ( b - a ) ( b - m ) } } .
$$

Each has degree 2, equals 1 at its own node and 0 at the other two. Hence $q : = p ( a ) L _ { a } + p ( m ) L _ { m } +$ $p ( b ) L _ { b }$ is a polynomial of degree at most 2 that agrees with p at a, m and $b ;$ the difference $p - q$ has degree at most 2 and three distinct roots, so $p = q$ identically, i.e.

$$
p ( s ) = p ( a ) L _ { a } ( s ) + p ( m ) L _ { m } ( s ) + p ( b ) L _ { b } ( s ) \qquad { \mathrm { f o r ~ a l l ~ } } s \in \mathbb { R } .\tag{29}
$$

We now bound the basis polynomials on $[ a , b ]$ . The affine change of variable $\begin{array} { r } { s = m + \frac { b - a } { 2 } x } \end{array}$ maps $[ a , b ]$ onto $x \in [ - 1 , 1 ]$ and the nodes $a , m , b { \mathrm { t o } } - 1 , 0 , 1$ , and transforms the basis polynomials into

$$
L _ { a } = \frac { x ( x - 1 ) } { 2 } , \qquad L _ { m } = 1 - x ^ { 2 } , \qquad L _ { b } = \frac { x ( x + 1 ) } { 2 } .
$$

On [-1, 1] the function $x \mapsto x ( x - 1 )$ attains its minimum $- { \frac { 1 } { 4 } }$ at $\textstyle x = { \frac { 1 } { 2 } }$ and its maximum 2 at $x = \stackrel { \cdot } { - } 1 , \stackrel { \cdot } { \mathrm { s o } } | L _ { a } | \leq 1 ;$ clearly $L _ { m } \in [ 0 , 1 ]$ ; and $| L _ { b } | \leq 1$ by the symmetry ${ \bar { x } } \mapsto - x$ . Therefore, by (29) and the triangle inequality, for every $s \in [ a , b ]$ ，

$$
\begin{array} { r l r } {  { \vert p ( s ) \vert \leq \vert p ( a ) \vert \vert L _ { a } ( s ) \vert + \vert p ( m ) \vert \vert L _ { m } ( s ) \vert + \vert p ( b ) \vert \vert L _ { b } ( s ) \vert } } \\ & { } & { \leq \vert p ( a ) \vert + \vert p ( m ) \vert + \vert p ( b ) \vert } \\ & { } & { \leq 3 \operatorname* { m a x } \big \{ \vert p ( a ) \vert , \vert p ( m ) \vert , \vert p ( b ) \vert \big \} , } \end{array}
$$

and the claim follows by taking the supremum over $s \in [ a , b ]$

We will control operator norms through the following standard ε-net argument, which reduces the supremum over the unit spheres to a maximum over a finite set of cardinality exponential in the dimension (see (Vershynin, 2018, Sec. 4.4)).

Lemma 11. Let $p , q \geq 1$ . There exist $\textstyle { \frac { 1 } { 4 } }$ -nets $\mathcal { U } \subset \mathbb { S } ^ { p - 1 }$ and $\mathcal { V } \subset \mathbb { S } ^ { q - 1 }$ (with respect to the Euclidean norm) with $| \mathscr { U } | \leq 9 ^ { p }$ and $| \nu | \leq 9 ^ { q }$ , and for every $M \in \mathbb { R } ^ { p \times q }$

$$
\| M \| _ { \mathrm { o p } } ~ \leq ~ 2 \operatorname* { m a x } _ { ( u , v ) \in \mathcal { U } \times \mathcal { V } } | u ^ { \top } M v | .
$$

Proof. The cardinality bound is the standard volumetric estimate $\mathcal { N } ( \mathbb { S } ^ { n - 1 } , \varepsilon ) \leq ( 1 + 2 / \varepsilon ) ^ { n }$ with $\textstyle \varepsilon = { \frac { 1 } { 4 } }$ . Let $u \in \mathbb { S } ^ { p - 1 } , v \in \mathbb { S } ^ { q - 1 }$ and pick $u _ { 0 } \in \mathcal { U } , v _ { 0 } \in \mathcal { V }$ with $\begin{array} { r } { \| u - u _ { 0 } \| _ { 2 } \leq \frac { 1 } { 4 } , \| v - v _ { 0 } \| _ { 2 } \leq \frac { 1 } { 4 } } \end{array}$ Since u $M v - u _ { 0 } ^ { \top } M v _ { 0 } = ( u - u _ { 0 } ) ^ { \top } M v + u _ { 0 } ^ { \top } M ( v - v _ { 0 } )$

$$
\begin{array} { r } { | u ^ { \top } M v | \ \leq \ | u _ { 0 } ^ { \top } M v _ { 0 } | + \left( \| u - u _ { 0 } \| _ { 2 } + \| v - v _ { 0 } \| _ { 2 } \right) \| M \| _ { \mathrm { o p } } \ \leq \ \underset { ( a , b ) \in \mathcal { U } \times \mathcal { V } } { \operatorname* { m a x } } | a ^ { \top } M b | + \frac { 1 } { 2 } \left\| M \right\| _ { \mathrm { o p } } . } \end{array}
$$

Taking the supremum over u, v and rearranging (all quantities are finite) gives the claim.

Our analysis of the loss landscape of ${ \mathcal { E } } ^ { \infty }$ repeatedly requires sharp two-sided bounds on the Gaussian tail $1 - { \dot { \Phi } }$ , which are most conveniently expressed through the inverse Mills ratio $\lambda : = \varphi / ( 1 - \Phi )$ The lemma below collects the elementary properties of λ that we shall need in the following lemma.

Lemma 12. Let $\lambda ( t ) : = \varphi ( t ) / \bigl ( 1 - \Phi ( t ) \bigr ) f o r t \geq 0 .$ Then

$$
( i ) \lambda \in \mathcal { C } ^ { \infty } ( [ 0 , \infty ) ) a n d \lambda ( 0 ) = 2 \varphi ( 0 ) = \sqrt { 2 / \pi } ;
$$

$$
( i i ) \lambda ^ { \prime } ( t ) = \lambda ( t ) \bigl ( \lambda ( t ) - t \bigr ) f o r a l l t \geq 0 ;
$$

$$
( i i i ) \ t < \lambda ( t ) < t + \frac { 1 } { t } f o r \ a l l \ t > 0 ; e q u i v a l e n t l y , \frac { t } { 1 + t ^ { 2 } } \varphi ( t ) < 1 - \Phi ( t ) < \frac { \varphi ( t ) } { t } .
$$

Proof. (i) is clear since $1 - \Phi > 0$ is smooth. For (ii), using $\varphi ^ { \prime } ( t ) = - t \varphi ( t ) { \mathrm { a n d } } ( 1 - \Phi ) ^ { \prime } = - \varphi$

$$
\lambda ^ { \prime } ( t ) = { \frac { - t \varphi ( t ) { \bigl ( } 1 - \Phi ( t ) { \bigr ) } + \varphi ( t ) ^ { 2 } } { { \bigl ( } 1 - \Phi ( t ) { \bigr ) } ^ { 2 } } } = - t \lambda ( t ) + \lambda ( t ) ^ { 2 } .
$$

For (iii), the upper bound on the tail follows from $\begin{array} { r } { 1 - \Phi ( t ) = \int _ { t } ^ { \infty } \varphi ( u ) d u < \int _ { t } ^ { \infty } \frac { u } { t } \varphi ( u ) d u = } \end{array}$ $\varphi ( t ) / t$ . For the lower bound let $\begin{array} { r } { h ( t ) : = 1 - \Phi ( t ) - \frac { t } { 1 + t ^ { 2 } } \varphi ( t ) ; } \end{array}$ then $h ( t ) \to 0 { \mathrm { a s } } t \to \infty$ and a direct computation gives $\begin{array} { r } { h ^ { \prime } ( t ) = - \frac { 2 \varphi ( t ) } { ( 1 + t ^ { 2 } ) ^ { 2 } } < 0 } \end{array}$ , so $h > 0$ on $( 0 , \infty )$ . Rearranging the two tail bounds gives $t < \lambda ( t ) < t + 1 / t$ □

## B EXPERIMENTS

In this section, we recall some quantization algorithm and preprocessing that are used for the experiments. Moreover, we report additional experiments that assess what our theory captures

## B.1 EXPERIMENTAL DETAILS

This appendix specifies the setup of all experiments in Section 5 and Appendix B. The released code (see the Reproducibility Statement) contains a frozen configuration file for every run, and its REPRODUCE . md gives the commands for the five experiment families below. Table 3 lists where the results of each family are reported. As in Section 5, WB denotes weight-only quantization at B bits.

Table 3: Experiment families and their reported results.
<table><tr><td>Section</td><td>Content</td><td>Reported in</td></tr><tr><td>B.2</td><td>Global scale sensitivity and abla- Figures 1, 2, 5, 6, 7, 8, and 9 tions</td><td></td></tr><tr><td>B.3</td><td>Scale selection and zero-shot ac- curacy</td><td>Tables 1, 2, 6, 7, and 8</td></tr><tr><td>B.4</td><td>Preprocessing and effective rank</td><td>Figure 10 and Table 9</td></tr><tr><td>B.5</td><td>Gaussian landscapes and finite- Figures 11 and 12 width convergence</td><td></td></tr><tr><td>B.6</td><td>and empirical landscapes</td><td>Local reconstruction sensitivity Figures 3, 13, 14, and 15</td></tr></table>

## B.1.1 MODELS

We quantize five publicly available decoder-only models (Table 4). Each is loaded in fp16 at a pinned Hugging Face revision with a context length of 2,048 tokens. OPT-125M is converted locally from the pinned PyTorch checkpoint to safetensors. Every linear layer inside the transformer blocks is quantized: the attention projections q, k, v, o, and the MLP projections (gate, up and down for Llama and Qwen; fc1 and $\mathtt { f } _ { \mathbf { C } 2 }$ for OPT). The quantization of each of this module exactly matches the quantization of $\vartheta ^ { ( \ell ) }$ in the sense of Section 3, calibrated on its own input activations. The token embeddings and the language-model head stay in fp16. The short names in Table 4 are those used in the figures.

Table 4: Models, short names used in the figures, and pinned checkpoint revisions.
<table><tr><td>Model</td><td>Short name</td><td>Hugging Face repository</td><td>Revision</td></tr><tr><td>OPT-125M</td><td>OPT-125M</td><td> $\mathtt { f a c e b o o k } / \mathrm { o p t } - 1 2 5 \mathrm { m }$ </td><td>27dcfa74d334</td></tr><tr><td>Llama-3.2-1B-Instruct</td><td>Llama-1B</td><td> $\mathtt { m e t a - l 1 a m a / L 1 a m a - 3 . 2 \mathrm { - 1 B - I n s t r u c t } }$ </td><td>9213176726f5</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>Llama-3B</td><td> $\mathrm { m e t a - 1 1 a m a / L 1 a m a - 3 . 2 \mathrm { - 3 B - I n s t r u c t } }$ </td><td>0cb88a4f764b</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>Llama-8B</td><td>meta-1lama/Llama-3.1-8B-Instruct</td><td> $0 \mathsf { e 9 e 3 9 f 2 4 9 a 1 }$ </td></tr><tr><td>Qwen3-8B</td><td>Qwen3-8B</td><td>Qwen/Qwen3-8B</td><td>b968826d9c46</td></tr></table>

## B.1.2 QUANTIZATION SETUP

Grid. All experiments are weight-only; activations remain in fp16. A WB weight is quantized with the round-to-nearest map $\Pi _ { s , M _ { B } }$ of (1), i.e. onto the symmetric uniform grid $s \{ - M _ { B } , \ldots , M _ { B } \}$ with $M _ { B } = 2 ^ { B - 1 } - 1$ levels per side and $2 ^ { B } - 1$ levels in total, with clipping to $\therefore \dot { \bf \Phi } \bot  { M _ { B } } s . \ \mathrm { A t } \  { B } = 2$ this is the ternary grid $\{ - s , 0 , s \}$ . We use $B \in \{ 2 , 3 , 4 , 6 , 8 \}$ . One fp16 scale is stored per quantization group, and there are no zero-points. Per-channel W3 on Llama-3.1-8B therefore costs 3.003 bits per weight.

Granularity. Per-channel quantization uses one scale per output row, which gives the row-wise problems studied in Section 4. Grouped quantization uses one scale per contiguous group of $g \in$ {512, 256, 128, 64} input columns. With activation ordering, the groups are formed after the column permutation.

GPTQ. Layers are quantized sequentially, block by block. Within a block the order is $\{ \mathbb { q } , \mathbb { k } , \mathbb { v } \} $ $\circ  \{ \mathsf { u p } , \mathsf { g a t e } \} $ down. Each block is calibrated on the outputs of the already-quantized preceding blocks $( ^ { 6 6 } \mathrm { c a r r y } ^ { 3 } )$ propagation), so the activations $X ^ { \ell - 1 }$ in (2) come from the partially quantized network. The Hessian proxy is

$$
H = \frac { 2 } { N _ { c } } \sum _ { n = 1 } ^ { N _ { c } } { X _ { n } X _ { n } ^ { \top } } ,
$$

where $X _ { n } \in \mathbb { R } ^ { d \times 2 0 4 8 }$ holds the layer inputs for calibration window n. Up to the constant factor, which cancels in the normalized loss of Section 4 and in every scale rule below, this is the Gram matrix $X X ^ { \top }$ of the theory. H is accumulated in fp32 over $ { \bar { N } } _ { c } ~ = ~ 1 2 8$ windows with a forward batch of 4 windows. Before inverting H we add $\lambda \operatorname { I d } _ { d }$ with $\lambda = 0 . 0 1$ · mean(diag H). Columns are processed in order of decreasing $H _ { i i }$ (activation ordering) in lazy batches of 128 columns, using the inverse-Cholesky update of the reference implementation (Frantar et al., 2022). Columns with $H _ { i i } = 0$ are set to zero. For grouped quantization the scale of a group is selected on the errorcompensated weights at the point where the algorithm enters the group.

Other engines. ResComp (Li et al., 2026) runs GPTAQ-style (Li et al., 2025) paired compensation plus a residual correction toward the full-precision weights, with $\alpha _ { \mathrm { G P T A Q } } = \alpha _ { \mathrm { R e s C o m p } } = 0 . 2 5$ It uses the stability mode of the reference implementation: org at W2 and allw otherwise. QRoNoS (Zhang et al., 2026) re-solves the full layer at the first column and uses “reset" propagation: each block is calibrated on full-precision inputs. Both engines use the damping, activation ordering and block size above; for grouped rows the compensation block is min(128, g). Because these engines use paired inputs, they additionally admit the rule RTNH-cross, which searches s over the grid of Appendix B.1.3 to minimize $\lVert W \dot { X _ { \mathrm { f p } } } - \Pi _ { s , M _ { B } } ( W ) X _ { q } \rVert _ { F } ^ { 2 }$ , where $X _ { \mathrm { f p } }$ and $X _ { q }$ are the full-precision and quantized-prefix inputs. We run it for $B \leq 3$ only.

Hadamard incoherence processing (HIP). HIP (Appendix B.4) is applied independently to the input dimension of every matrix, $\bar { W } \mapsto W R$ with $R \ = \ \deg ( \xi ) Q$ , where Q is a normalized Hadamard matrix $( Q ^ { \top } Q ^ { ' } = \operatorname { I d } _ { d } )$ from the QuIP# implementation (Tseng et al., 2024) at a pinned commit, and $\xi \in \{ - 1 , 1 \} ^ { d }$ are Rademacher signs drawn from a seed and the matrix name. The Hessian is transformed as $H \mapsto R ^ { \top } H R .$ , which leaves its spectrum, and hence $r _ { \mathrm { e f f } } ( H )$ , unchanged (Appendix B.4). At inference the layer input is rotated online by $R ^ { \top }$ , so the network function is unchanged before quantization. Unless stated otherwise the rotation seed is 0.

## B.1.3 SCALE-SELECTION RULES

Let w denote the (error-compensated) weights of one row or group, and let $s ^ { e } = \operatorname* { m a x } _ { q } | w _ { q } | / M _ { B }$ be its min-max scale (Section 3). Every rule outputs one scale s per row or group (Table 5). Min-max and Analytic require no search; Shrink-2.4 and WMSE search without activation data; Proxy-static and RTNH use the Hessian.

Table 5: Scale-selection rules. $e ( s ) = \Pi _ { s , M _ { B } } ( w ) - w$ is the rounding error at scale $s ,$ and $H _ { g g }$ is the undamped Hessian block of the row or group. The search window for η depends on the experiment (Appendix B.1.3).
<table><tr><td>Rule</td><td>Scale</td><td>Objective</td></tr><tr><td>Min-max</td><td> $s ^ { e }$ </td><td>none</td></tr><tr><td>Shrink-2.4</td><td> $p s ^ { e } , p \in \{ 1 . 0 0 , 0 . 9 9 , \ldots , 0 . 2 1 \}$ </td><td> $\begin{array} { r } { \sum _ { q } | e _ { q } ( p s ^ { e } ) | ^ { 2 . 4 } } \end{array}$ </td></tr><tr><td>WMSE</td><td> $s ^ { e } e ^ { \eta }$  , grid search</td><td> $\textstyle \sum _ { q } ^ { * } e _ { q } ( s ) ^ { 2 }$ </td></tr><tr><td>Proxy-static (ours)</td><td> $s ^ { e } e ^ { \eta }$  , grid search</td><td> $\sum _ { q } ^ { \mathbf { \bar { \alpha } } } e _ { q } ( s ) ^ { 2 } / U _ { q q } ^ { 2 }$  (GPTQ step loss)</td></tr><tr><td>RTNH</td><td> $s ^ { e } e ^ { \eta }$  , grid search</td><td> $e ( s ) ^ { \top } H _ { g g } e ( s )$ </td></tr><tr><td>Analytic</td><td> $s ^ { * } ( M _ { B } ) \hat { \sigma }$ </td><td>none; ô is the RMS of w</td></tr></table>

Min-max and Shrink-2.4. Min-max is the symmetric min-max choice $s ^ { e }$ of Section 3. Shrink-2.4 is the clipping search of the GPTQ reference quantizer, applied to our symmetric $( 2 ^ { B } - 1 )$ -level grid: it scores the 80 shrink fractions $p _ { i } = 1 - i \bar { / } 1 0 0 , i = \bar { 0 } , . . . , 7 9$ , of the min-max range by the $\breve { \ell } ^ { 2 . 4 }$ rounding error.

Analytic scale. $s ^ { * } ( M _ { B } )$ is the unique minimizer of the limiting objective $\mathcal { E } ^ { \infty } ( M _ { B } , \cdot )$ (Proposition 1), obtained as the zero of $\partial _ { s } \mathcal { E } ^ { \infty } ( M _ { B } , \cdot )$ in the closed form (28) by bisection. This gives $s ^ { * } ( M _ { B } ) = 1 . 2 2 4 0 , 0 . 6 5 0 8 , 0 . 3 5 3 4$ , 0.1055, 0.0309 for $B = 2 , 3 , 4 , 6 , 8 \ : ( M _ { B } = 1 , 3 , 7 , 3 1$ , 127). ô is computed on the current compensated weights and excludes input columns with $H _ { i i } = 0$

Proxy-static. Let U be the upper Cholesky factor of the inverse damped Hessian in activation order, and let $e _ { q } ( s ) = \Pi _ { s , M _ { B } } ( w _ { q } ) - w _ { q }$ . For each row or group, Proxy-static selects

$$
s _ { \mathrm { P S } } \in \arg \operatorname* { m i n } _ { s \in \mathcal { S } } \sum _ { q } \frac { e _ { q } ( s ) ^ { 2 } } { U _ { q q } ^ { 2 } } ,
$$

where S is the candidate grid described below. Candidate scores use a fixed snapshot of the weights at group entry, including compensation from previous groups. The objective uses $\mathrm { { G P T Q ^ { \circ } s } }$ percolumn loss weights without simulating candidate-dependent error feedback. After scale selection, quantization proceeds with the usual GPTQ updates. Algorithm 2 summarizes the procedure. We motivate proxy-static by analysing the GPTQ. Our selector is motivated by the loss incurred at each GPTQ quantization step. Let A denote the positive-definite, damped Gram matrix in activation order, and consider a quadratic loss ${ \mathcal { L } } ( z ) = { \bar { z } } ^ { \top } A z$ . At step q, let $F _ { q } = \{ q , \dots , d \}$ denote the remaining free coordinates, and let $w ^ { ( q ) }$ be their current compensated weights. Fixing coordinate q to its quantized value introduces the residual

$$
\epsilon _ { q } ( s ) = w _ { q } ^ { ( q ) } - Q _ { s } ( w _ { q } ^ { ( q ) } ) .
$$

The optimal compensating update solves

$$
\operatorname* { m i n } _ { \Delta \in \mathbb { R } ^ { | F _ { q } | } } \Delta ^ { \top } A _ { F _ { q } F _ { q } } \Delta \quad \mathrm { s u b j e c t t o } \quad \Delta _ { q } = - \epsilon _ { q } ( s ) .
$$

where $A _ { F _ { q } F _ { q } }$ is the principal submatrix of A indexed by $F _ { q }$ Writing $e _ { q }$ for the corresponding coordinate vector, the solution and minimum are

$$
\Delta ^ { \star } = - \frac { \epsilon _ { q } ( s ) } { [ A _ { F _ { q } F _ { q } } ^ { - 1 } ] _ { q q } } A _ { F _ { q } F _ { q } } ^ { - 1 } e _ { q } , \qquad \Delta \mathcal { L } _ { q } ( s ) = \frac { \epsilon _ { q } ( s ) ^ { 2 } } { [ A _ { F _ { q } F _ { q } } ^ { - 1 } ] _ { q q } } .
$$

Here the increment is measured from the conditional quadratic minimizer given the previously fixed coordinates.

If $A ^ { - 1 } = U ^ { \top } U$ with U upper triangular, then

$$
[ A _ { F _ { q } F _ { q } } ^ { - 1 } ] _ { q q } = U _ { q q } ^ { 2 } .
$$

Thus GPTQ's step loss is $\epsilon _ { q } ( s ) ^ { 2 } / U _ { q q } ^ { 2 }$ . Evaluating these losses for every candidate scale would require recomputing the candidate-dependent compensation trajectory.

Proxy-static instead freezes the weights at group entry. For a row or group g with snapshot $v ,$ it selects

$$
s _ { \mathrm { P S } } \in \mathop { \arg \operatorname* { m i n } } _ { s \in \mathcal { S } } \sum _ { q \in g } \frac { ( v _ { q } - Q _ { s } ( v _ { q } ) ) ^ { 2 } } { U _ { q q } ^ { 2 } } .
$$

This retains GPTQ's per-column loss weights while approximating the sequential residuals by static rounding residuals. The factor U is already available from GPTQ, so candidate scoring requires no additional factorization or candidate-specific compensation pass. After scale selection, the usual GPTQ updates are applied. Algorithm 2 summarizes the procedure.

## B.1.4 COMPUTE RESOURCES AND SOFTWARE

Every run uses a single NVIDIA H100 (94 GB) or H200 (141 GB) GPU.

## B.2 GLOBAL SCALE-SENSITIVITY

In this section we add further experiments to understand how different scale-selection rules impact the performance of PTQ of LLMS. This section presents additional experiments examining how scale-selection rules affect the performance of quantized LLMs.

Algorithm 2 GPTQ with Proxy-static scale selection   
Require: Weights $W \in \mathbb { R } ^ { m \times d } .$ , input Gram $H ,$ bit-width $B \geq 2 ,$ group width $G ,$ candidate count   
$C ,$ interval $[ \eta _ { \mathrm { m i n } } , \eta _ { \mathrm { m a x } } ] ,$ damping fraction λ   
Ensure: Quantized weights $\widehat { W }$ and group scales $\left\{ s _ { r g } ^ { \star } \right\}$   
1: $K \gets 2 ^ { B - 1 } - 1$   
2: Define $Q _ { s } ( x ) = s \ \mathrm { c l i p } ( \mathrm { r o u n d } ( x / s ) , - K , K )$   
3: $\mathcal { D }  \{ j : H _ { j j } = 0 \}$   
4: $W _ { : , \mathcal { D } } \stackrel {  } {  } 0 ; \ddot { H } _ { j j } \stackrel {  } {  } 1 \mathrm { f o r } j \in \mathcal { D }$   
5: Let $P$ sort $\mathrm { d i a g } ( H )$ in descending order   
6: $\widetilde { W }  W P ^ { \top } ; \widetilde { H }  P H P ^ { \top }$   
7: $\widetilde { H }  \widetilde { H } + \lambda \operatorname* { m e a n } ( \mathrm { d i a g } ( \widetilde { H } ) ) I$   
8: Compute upper triangular $U$ satisfying $\widetilde { H } ^ { - 1 } = U ^ { \top } U$   
9: $( \eta _ { k } ) _ { k = 1 } ^ { \hat { C } } \gets \mathrm { \ddot { l i n s p a c e } } ( \eta _ { \mathrm { m i n } } , \eta _ { \mathrm { m a x } } , \dot { C } )$   
10: Replace the $\eta _ { k }$ closest to zero by 0   
11: for each contiguous group $g$ in activation order do   
12: $V  \mathrm { c o p y } ( \widetilde { W } _ { : , g } )$ $\triangleright$ freeze compensated group-entry weights   
13: for each output row r do   
14: $a \gets \operatorname* { m a x } _ { j \in g } | V _ { r j } |$   
15: if $a = 0$ then   
16: $s _ { r g } ^ { \star }  1$ any positive scale quantizes zero exactly   
17: else   
18: $s _ { r g } ^ { ( 0 ) }  a / K$   
19: for $k = \mathrm { i } , \ldots , C$ do   
20: $s _ { r g k }  s _ { r g } ^ { ( 0 ) } \exp ( \eta _ { k } )$   
21: $L _ { r g k }  \sum _ { j \in q } \frac { ( \dot { V } _ { r j } - Q _ { s _ { r g k } } ( V _ { r j } ) ) ^ { 2 } } { U _ { j j } ^ { 2 } }$   
22: end for   
23: Select $k ^ { \star }$ minimizing $L _ { r g k }$ , breaking numerical ties toward smallest $| \eta _ { k } |$   
24: $s _ { r g } ^ { \star }  s _ { r g k ^ { \star } }$   
25: end $\mathbf { i } \mathbf { f } ^ {  }$   
26: end for   
27: for columns $j \in g$ in activation order do   
28: $( q _ { j } ) _ { r }  Q _ { s _ { r g } ^ { \star } } ( \widetilde { W } _ { r j } )$ for every row $r$   
29: $e _ { j } \gets ( \widetilde { W } _ { : , j } - q _ { j } ) / U _ { j j }$   
30: $\widetilde { W } _ { : , j : d } \gets \widetilde { W } _ { : , j : d } - e _ { j } U _ { j , j : d }$   
31: $\widehat { W } _ { : , j } ^ { \mathrm { o r d } } \gets q _ { j }$   
32: end for   
33: end for   
34: ${ \widehat { W } }  { \widehat { W } } ^ { \mathrm { o r d } } P$   
35: return $\widehat { W } , \{ s _ { r g } ^ { \star } \}$ , and the group assignment induced by $P$

## B.2.1 SENSITIVITY TO SCALE SELECTION WITH HIP PREPROCESSING

HIP preprocessing appears to produce a steeper decrease in NLL spread with bit-width, qualitatively similar to the effect of reducing the group size to 128 in GPTQ without HIP. However, as discussed in Remark 3, this observation does not directly imply lower curvature of the GPTQ loss landscape. The spread also depends on how far apart the scales selected by the different rules are. HIP may increase agreement among these choices: suppressing extreme weight coordinates can reduce the influence of outliers on min-max scaling, while a more Gaussian weight distribution may improve agreement between WMSE-based scales and the analytic prediction. This provides a possible additional explanation for the reduced variation in end-to-end performance.

![](images/b9cf4c11663f703167e298d7f06339958d52ddd19a14a503877d80c93bac5b86.jpg)  
Figure 5: WikiText-2 NLL spread across scale-selection rules for GPTQ with HIP preprocessing, plotted against bit-width for different models. Left: per-channel quantization. Right: group size 128. The spread is the difference between the maximum and minimum NLL across Min-max, Shrink-2.4, WMSE, Proxy-static, and RTNH. The vertical axis uses a logarithmic scale.

## B.2.2 SENSITIVITY TO SCALE SELECTION WITH GPTQ EXTENSIONS

To assess whether the trend in Figure 1 extends beyond GPTQ, we repeat the scale-selection comparison with ResComp (Li et al., 2026) and QRoNoS (Zhang et al., 2026). We evaluate both methods at W2, W4, and W8 on Llama-3.2-1B and Llama-3.1-8B, with and without HIP preprocessing.

![](images/ae3f1aece292fe4bab40fdc5b743d95758313ecde787cf899622df849b766ecf.jpg)  
Figure 6: WikiText-2 NLL spread across scale-selection rules for ResComp and QRoNoS, with and without HIP preprocessing, as a function of bit-width. The spread is the difference between the maximum and minimum NLL across the tested scale-selection rules.

Both methods exhibit a decrease in NLL spread with increasing bit-width, qualitatively matching the trend observed for GPTQ. HIP preprocessing is associated with an even steeper decrease. These results suggest that the reduced sensitivity to scale selection at higher precision extends beyond GPTQ to related PTQ methods in the tested settings.

## B.2.3 SCALE SENSITIVITY WITHOUT MIN-MAX

To understand whether the bit-width dependence of NLL spread is mainly due to the poor performance of min-max at low precision, we repeat the comparison using only the search based rules Shrink-2.4, WMSE, Proxy-static, and RTNH.

![](images/1ede5c3c6a2bc6cf14aa05ff45226970f7dfcbd895752a445224a65e7876e8a4.jpg)  
Figure 7: WikiText-2 NLL spread across Shrink-2.4, WMSE, Proxy-static, and RTNH for perchannel GPTQ without HIP (left) and with HIP (right). The spread is the maximum minus minimum NLL across these four rules; min-max is excluded. The vertical axis uses a logarithmic scale.

Figure 7 shows that the broad reduction in NLL spread with increasing precision persists after excluding min-max. Without HIP, the dependence is nonmonotonic between W2 and W4, followed by a pronounced decrease at higher precisions. With HIP, the decrease is more consistent across the tested bit-widths, with a particularly large reduction between W2 and W3. Thus, the greater practical importance of scale selection at low precision persists among the search-based rules and can not be explained by the bad performance of min-max. Performing a scale search alone does not ensure comparable end-to-end performance: the criterion used to select the scale remains important.

## B.2.4 END-TO-END SENSITIVITY TO CONTROLLED SCALE PERTURBATIONS

As discussed in Remark 3 the scale-sensitivity plots based on different scale rules do not account for the possibility that discrepancy of the scale rules themselves change with the precision. We therefore add a controlled global sensitivity analysis at a fixed relative perturbation amplitude. We perturb the scales returned by the WMSE search and measure the resulting change in end-to-end performance in NLL. For each bit-width B and channel c, we set

$$
s _ { B , c } ^ { ( r , \pm ) } = s _ { B , c } ^ { 0 } ( 1 \pm \delta \epsilon _ { r , c } ) , \qquad \delta = 0 . 0 5 ,
$$

where $\epsilon _ { r , c } \in \{ - 1 , + 1 \}$ are independent random signs. We use five random directions, matched across bit-widths within each model, and evaluate both signs of each direction. Each perturbed scale configuration is applied in a fresh sequential GPTQ run from the original floating-point weights, recomputing downstream activations, Gram matrices, and error compensation.

Let $\Delta _ { B , r , \pm }$ denote the change in NLL relative to the corresponding unperturbed baseline. We summarize the magnitude of the response by

$$
C _ { B } ( \delta ) = \frac { 1 } { 2 R } \sum _ { r = 1 } ^ { R } \left( | \Delta _ { B , r , + } | + | \Delta _ { B , r , - } | \right) , \qquad R = 5 .
$$

This statistic captures both improvements and degradations without cancellation between opposite perturbation arms. We interpret $C _ { B } ( \delta )$ as finite-perturbation scale sensitivity, not directly as curvature. Away from a stationary point its leading contribution is the directional NLL gradient; near a stationary point, second-order effects can dominate. The absolute values retain improvements as well as degradations, so this statistic measures response magnitude rather than the signed penalty from perturbing the scales. We evaluate per-channel GPTQ without HIP at W2, W3, W4, W6, and W8 on five models, using one calibration draw and evaluating on WikiText-2 and C4.

![](images/78f67aab73eb4abddf08da46e51ad7e7910b837bc75836573b3963a4934ca4e3.jpg)  
Figure 8: End-to-end sensitivity to controlled scale perturbations. Each channel's WMSE-selected scale is increased or decreased by $5 \% ,$ followed by a complete sequential GPTQ run. Each point reports the mean absolute NLL change relative to the unperturbed WMSE baseline, averaged over ten perturbed runs corresponding to five random sign patterns and their opposites. Experiments use per-channel GPTQ and one calibration draw; random directions are matched across bit-widths within each model.

Figure 8 shows substantially larger responses at W2–W4 than at W6–W8. Across the Llama and Qwen models and the two evaluation corpora, the reported W4-to-W6 reduction is approximately 8-96-fold. OPT-125M exhibits a weaker reduction and remains more sensitive at high precision. The dependence on bit-width is not monotonic: for example, Llama-3.1-8B is more sensitive at W4 than at W3 on WikiText-2.

These results further support greater practical sensitivity to relative scale perturbations at low precision in the tested setting. They complement the local reconstruction-loss analysis, while showing that its regular bit-width dependence need not neccessarily translate into a monotonic end-to-end response. The interpretation also depends on baseline quality: some W2 models are severely degraded, and several W4 baselines perform worse than their W3 counterparts.

## B.2.5 CALIBRATION-SET SIZE

Our main experiments use 128 calibration sequences of 2048 tokens each. To assess whether our observations depend on this choice, we vary the number of calibration sequences from 16 to 256 while keeping the sequence length fixed. We compare Min-max, Analytic, and Proxy-static on Llama-1B and OPT-125M at W2 and W3, with and without HIP.

Figure 9 shows that calibration-set size affects NLL, sometimes nonmonotonically, but substantial differences between scale-selection rules persist across the tested sizes. In particular, increasing the calibration set does not consistently close the performance gaps between rules. Thus, the practical importance of scale selection at low precision is not specific to our default calibration-set size.

## B.3 ZERO-SHOT EVALUATION

We complement perplexity evaluation with six zero-shot tasks: PIQA, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, and BoolQ. All quantized results use per-channel weight-only quantization, calibration seed 0, and 128 C4 sequences of 2048 tokens each. The symmetric grid has $K _ { B } = 2 ^ { B - 1 } - 1$ , so W2 is ternary. Raw denotes no preprocessing; HIP uses rotation seed 0. We report acc\_norm for PIQA, ARC-Easy, ARC-Challenge, and HellaSwag, and acc for WinoGrande and BoolQ. Accuracies are percentages, while meanis computed as their unweighted average on the six tasks. WikiText-2 and C4 columns report perplexity. Lower perplexity and higher accuracy are better. Bold entries identify column-wise best reported values within each matched model, precision, preprocessing, engine, and campaign panel, including ties at the displayed precision. The FP16 reference is not bolded. These are single-calibration-seed measurements; differences of a few tenths of an accuracy point do not establish resolved rankings.

![](images/43405ef9654bce73b54e44d0ce215f3a25ec6d387eeac5007cbf140611502a70.jpg)  
Figure 9: Effect of calibration-set size on WikiText-2 NLL (top) and C4 NLL (bottom). Each plot compares Min-max, Analytic, and Proxy-static for Llama-1B and OPT-125M at W2 and W3, with and without HIP. Each calibration sequence contains 2048 tokens.

W3 scale selection. Table 6 compares four scale-selection rules for GPTQ. With HIP, the searchfree analytic rule achieves mean accuracies of 72.08%, 63.75%, and 72.36% on Llama-3.1-8B, Llama-3.2-3B, and Qwen3-8B, respectively, comparable to the searched rules and substantially above min-max. Without HIP, the analytic rule fails severely on all three models. Proxy-static achieves the lowest perplexity on both corpora and the highest six-task mean in each raw W3 comparison. Thus, the relative performance of the selectors depends strongly on preprocessing.

W2 scale selection. Table 7 shows substantial degradation in several W2 configurations, especially without HIP and with min-max. The proxy-static results use the search interval $\bar { \eta } \in [ - 1 , 1 ]$ . Proxystatic achieves lower perplexity on both corpora than Shrink-2.4 in all six model-preprocessing comparisons, but higher six-task mean accuracy in only three. For example, on Llama-3.1-8B with HIP, proxy-static achieves WikiText-2/C4 perplexity of 31.38/43.81, compared with 39.16/53.99 for Shrink-2.4, while their mean accuracies are 48.50% and 48.69%, respectively. Scale selection therefore affects both metrics, but their rankings need not coincide.

Transfer across quantization engines. Table 8 compares Shrink-2.4 and proxy-static for GPTQ, ResComp, and QRoNoS at W2 with HIP preprocessing. This focused campaign uses 128 log-spaced proxy candidates over $\eta \in [ - 1 . 0 , 0 . 1 ]$ . Its search settings differ from the main GPTQ campaign, so overlapping configurations are reported separately. Proxy-static lowers perplexity on both corpora in all six model-engine comparisons and achieves higher mean accuracy in five, although some accuracy differences are small. We omit the corresponding results without HIP because all tested configurations exhibit severe degradation. These findings support proxy-static as a practical scale selector beyond GPTQ, while showing that perplexity improvements do not always translate into accuracy gains.

## B.4 PREPROCESSING AND EFFECTIVE RANK

HIP Consider a fixed nonzero weight row $w \in \mathbb { R } ^ { d }$ and a uniformly random orthogonal matrix R. A random rotation distributes wR uniformly on the sphere of radius $\lVert \boldsymbol { w } \rVert _ { 2 }$ Equivalently,

$$
w R \overset { d } { = } \| w \| _ { 2 } \frac { g } { \| g \| _ { 2 } } , \qquad g \sim \mathcal { N } ( 0 , \mathrm { I d } _ { d } ) .
$$

Since $\| g \| _ { 2 } / { \sqrt { d } }$ concentrates around one, individual coordinates are approximately Gaussian in high dimensions, with standard deviation $\| w \| _ { 2 } / { \sqrt { d } } .$ the row's RMS. Applying the corresponding inverse

Table 6: W3 per-channel GPTQ with a symmetric seven-level grid $\{ k s : k = - 3 , \ldots , 3 \}$ . The proxy-static search uses $\eta \in [ - 1 , 1 ]$ . Panels (a) and (b) show raw and HIP results. FP16 references are excluded from bolding, which identifies the best quantized rule within each model and panel. Lower perplexity and higher accuracy are better.
<table><tr><td>Model</td><td>Rule</td><td>Wiki2 PPL</td><td>C4 PPL</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>HellaSwag</td><td>WinoGrande BoolQ</td><td></td><td>Mean</td></tr><tr><td colspan="9">(a) Without preprocessing</td></tr><tr><td>Llama-3.1-8B FP16 reference</td><td></td><td>7.213</td><td>11.388</td><td>80.96</td><td>79.59</td><td>55.12</td><td>79.26</td><td>74.03</td><td>84.10</td><td>75.51</td></tr><tr><td></td><td>Shrink-2.4</td><td>14.05</td><td>17.18</td><td>70.46</td><td>55.30</td><td>36.86</td><td>70.76</td><td>66.61</td><td>79.48</td><td>63.25</td></tr><tr><td></td><td>Proxy-static</td><td>11.76</td><td>16.33</td><td>72.91</td><td>63.51</td><td>41.21</td><td>72.09</td><td>68.51</td><td>79.85</td><td>66.34</td></tr><tr><td></td><td>Analytic</td><td>9.3 × 105</td><td>6.43 × 105</td><td>51.36</td><td>25.42</td><td>26.79</td><td>26.73</td><td>50.75</td><td>62.17</td><td>40.54</td></tr><tr><td></td><td>Min-max</td><td>41.47</td><td>45.51</td><td>62.35</td><td>40.78</td><td>28.24</td><td>47.20</td><td>51.54</td><td>52.42</td><td>47.09</td></tr><tr><td>Llama-3.2-3B FP16 reference</td><td></td><td>11.048</td><td>16.486</td><td>75.52</td><td>67.85</td><td>46.16</td><td>70.44</td><td>67.32</td><td>78.62</td><td>67.65</td></tr><tr><td></td><td>Shrink-2.4</td><td>42.48</td><td>47.93</td><td>69.91</td><td>53.75</td><td>34.22</td><td>59.46</td><td>55.96</td><td>66.85</td><td>56.69</td></tr><tr><td></td><td>Proxy-static</td><td>21.16</td><td>27.55</td><td>70.29</td><td>58.46</td><td>35.92</td><td>60.93</td><td>61.96</td><td>70.95</td><td>59.75</td></tr><tr><td></td><td>Analytic</td><td>2310.17</td><td>1027.36</td><td>51.52</td><td>26.68</td><td>25.68</td><td>26.68</td><td>52.96</td><td>39.94</td><td>37.24</td></tr><tr><td></td><td>Min-max</td><td>120.61</td><td>73.61</td><td>56.20</td><td>33.16</td><td>26.11</td><td>41.56</td><td>52.72</td><td>52.84</td><td>43.77</td></tr><tr><td>Qwen3-8B</td><td>FP16 reference</td><td>9.715</td><td>15.363</td><td>77.75</td><td>80.93</td><td>56.48</td><td>74.92</td><td>67.64</td><td>86.61</td><td>74.05</td></tr><tr><td></td><td>Shrink-2.4</td><td>37.95</td><td>33.39</td><td>66.49</td><td>43.39</td><td>31.57</td><td>50.36</td><td>55.56</td><td>75.23</td><td>53.77</td></tr><tr><td></td><td>Proxy-static</td><td>18.51</td><td>21.39</td><td>73.23</td><td>60.14</td><td>40.61</td><td>62.68</td><td>64.40</td><td>81.83</td><td>63.82</td></tr><tr><td></td><td>Analytic</td><td>3.13 × 1010</td><td>5.99 × 1010</td><td>52.12</td><td>25.72</td><td>27.22</td><td>26.55</td><td>49.17</td><td>37.83</td><td>36.43</td></tr><tr><td></td><td>Min-max</td><td>22.39</td><td>26.90</td><td>63.22</td><td>41.46</td><td>26.54</td><td>53.57</td><td>52.64</td><td>58.65</td><td>49.35</td></tr><tr><td colspan="9">(b) HIP preprocessing</td><td></td><td></td></tr><tr><td>Llama-3.1-8B FP16 reference</td><td></td><td>7.213</td><td>11.388</td><td>80.96</td><td>79.59</td><td>55.12</td><td>79.26</td><td>74.03</td><td>84.10</td><td>75.51</td></tr><tr><td></td><td>Shrink-2.4</td><td>9.09</td><td>14.45</td><td>78.45</td><td>74.41</td><td>50.77</td><td>74.00</td><td>71.67</td><td>81.74</td><td>71.84</td></tr><tr><td></td><td>Proxy-static</td><td>9.23</td><td>14.62</td><td>78.13</td><td>75.67</td><td>48.81</td><td>73.87</td><td>73.32</td><td>81.65</td><td>71.91</td></tr><tr><td></td><td>Analytic</td><td>9.21</td><td>14.60</td><td>78.56</td><td>76.14</td><td>50.60</td><td>73.65</td><td>71.11</td><td>82.45</td><td>72.08</td></tr><tr><td></td><td>Min-max</td><td>15.40</td><td>22.77</td><td>70.46</td><td>54.38</td><td>33.19</td><td>62.02</td><td>59.51</td><td>62.11</td><td>56.94</td></tr><tr><td></td><td>Llama-3.2-3B FP16 reference</td><td>11.048</td><td>16.486</td><td>75.52</td><td>67.85</td><td>46.16</td><td>70.44</td><td>67.32</td><td>78.62</td><td>67.65</td></tr><tr><td></td><td>Shrink-2.4</td><td>14.96</td><td>20.21</td><td>71.76</td><td>60.40</td><td>38.31</td><td>64.33</td><td>64.40</td><td>75.90</td><td>62.52</td></tr><tr><td></td><td>Proxy-static</td><td>15.01</td><td>20.56</td><td>72.52</td><td>62.63</td><td>38.05</td><td>63.76</td><td>64.09</td><td>75.66</td><td>62.79</td></tr><tr><td></td><td>Analytic</td><td>14.59</td><td>20.06</td><td>72.09</td><td>63.68</td><td>39.59</td><td>64.64</td><td>65.04</td><td>77.46</td><td>63.75</td></tr><tr><td></td><td>Min-max</td><td>30.21</td><td>33.48</td><td>59.19</td><td>37.25</td><td>27.13</td><td>54.46</td><td>55.49</td><td>55.66</td><td>48.20</td></tr><tr><td>Qwen3-8B</td><td>FP16 reference</td><td>9.715</td><td>15.363</td><td>77.75</td><td>80.93</td><td>56.48</td><td>74.92</td><td>67.64</td><td>86.61</td><td>74.05</td></tr><tr><td></td><td>Shrink-2.4</td><td>10.76</td><td>16.78</td><td>75.68</td><td>73.82</td><td>48.55</td><td>70.65</td><td>68.11</td><td>85.26</td><td>70.35</td></tr><tr><td></td><td>Proxy-static</td><td>10.68</td><td>16.70</td><td>76.88</td><td>73.82</td><td>49.49</td><td>70.91</td><td>69.30</td><td>85.50</td><td>70.98</td></tr><tr><td></td><td>Analytic</td><td>10.87</td><td>16.93</td><td>76.71</td><td>77.82</td><td>52.47</td><td>70.90</td><td>70.32</td><td>85.93</td><td>72.36</td></tr><tr><td></td><td>Min-max</td><td>15.20</td><td>20.99</td><td>69.80</td><td>48.32</td><td>32.76</td><td>61.49</td><td>56.59</td><td>68.04</td><td>56.17</td></tr></table>

transformation to the activations preserves the original linear map exactly:

$$
\begin{array} { r } { \widetilde { W } = W R , \qquad \widetilde { X } = R ^ { \top } X , \qquad \widetilde { W } \widetilde { X } = W X . } \end{array}
$$

Although the full-precision output is unchanged, coordinatewise quantization in the rotated basis leads to different errors.

For fixed activations X, the orthogonal rotation gives $\tilde { H } = R ^ { \top } H R$ This changes the coordinate system but preserves the Hessian's eigenvalues. Since $H = X X ^ { \top }$ is positive semidefinite, its trace is the sum of its eigenvalues and its operator norm is the largest eigenvalue. Both are therefore unchanged, so

$$
r _ { \mathrm { e f f } } ( \widetilde { H } ) = \frac { \mathrm { t r } ( \widetilde { H } ) } { \Vert \widetilde { H } \Vert _ { \mathrm { o p } } } = \frac { \mathrm { t r } ( H ) } { \Vert H \Vert _ { \mathrm { o p } } } = r _ { \mathrm { e f f } } ( H ) .
$$

Thus, rotation alone cannot improve the effective rank; changes in upstream quantization may still affect it by changing X. Rotation therefore acts only on the weight distribution, not on the conditioning of the objective. QuIP (Chee et al., 2023) uses exactly such structured random orthogonal preprocessing to reduce weight outliers and obtain incoherence guarantees, substantially improving quantization performance.

Our HIP transform uses efficient randomized Hadamard mixing based on QuIP# Tseng et al. (2024) rather than a full uniformly random rotation. Here, each transformed coordinate is a normalized random signed sum of the original weights, motivating a Gaussian approximation through a centrallimit argument when no individual contribution dominates.

Table 7: W2–ternary per-channel GPTQ with grid $\{ - s , 0 , s \}$ . Panels (a) and (b) show raw and HIP results. FP16 references are excluded from bolding, which identifies the best quantized rule within each model and panel. Lower perplexity and higher accuracy are better.
<table><tr><td>Model</td><td>Rule</td><td>Wiki2 PPL</td><td>C4 PPL</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>HellaSwag</td><td>WinoGrande</td><td>BoolQ</td><td>Mean</td></tr><tr><td colspan="9">(a) Without preprocessing</td></tr><tr><td>Llama-3.1-8B FP16 reference</td><td></td><td>7.213</td><td>11.388</td><td>80.96</td><td>79.59</td><td>55.12</td><td>79.26</td><td>74.03</td><td>84.10</td><td>75.51</td></tr><tr><td></td><td>Shrink-2.4</td><td>158.59</td><td>110.10</td><td>53.37</td><td>28.58</td><td>24.57</td><td>34.08</td><td>50.12</td><td>62.54</td><td>42.21</td></tr><tr><td></td><td>Proxy-static</td><td>157.51</td><td>105.48</td><td>53.50</td><td>29.00</td><td>23.70</td><td>31.30</td><td>50.70</td><td>55.30</td><td>40.60</td></tr><tr><td></td><td>Analytic</td><td>2.18 × 105</td><td>2.37 × 105</td><td>51.14</td><td>24.12</td><td>27.30</td><td>26.09</td><td>51.14</td><td>56.61</td><td>39.40</td></tr><tr><td></td><td>Min-max</td><td>7.87 × 105</td><td>6.73 × 105</td><td>50.76</td><td>24.62</td><td>26.19</td><td>26.66</td><td>51.22</td><td>48.13</td><td>37.93</td></tr><tr><td>Llama-3.2-3B</td><td>FP16 reference</td><td>11.048</td><td>16.486</td><td>75.52</td><td>67.85</td><td>46.16</td><td>70.44</td><td>67.32</td><td>78.62</td><td>67.65</td></tr><tr><td></td><td>Shrink-2.4</td><td>981.90</td><td>639.95</td><td>52.23</td><td>23.86</td><td>26.28</td><td>26.93</td><td>50.20</td><td>53.79</td><td>38.88</td></tr><tr><td></td><td>Proxy-static</td><td>442.39</td><td>315.76</td><td>52.30</td><td>25.80</td><td>25.80</td><td>27.70</td><td>50.50</td><td>54.40</td><td>39.40</td></tr><tr><td></td><td>Analytic</td><td>9291.88</td><td>3934.12</td><td>51.25</td><td>25.13</td><td>25.68</td><td>26.47</td><td>50.91</td><td>38.04</td><td>36.25</td></tr><tr><td></td><td>Min-max</td><td>2.59 × 105</td><td>2.47 × 105</td><td>51.41</td><td>24.16</td><td>27.56</td><td>26.40</td><td>52.25</td><td>47.46</td><td>38.21</td></tr><tr><td>Qwen3-8B</td><td>FP16 reference</td><td>9.715</td><td>15.363</td><td>77.75</td><td>80.93</td><td>56.48</td><td>74.92</td><td>67.64</td><td>86.61</td><td>74.05</td></tr><tr><td></td><td>Shrink-2.4</td><td>135.89</td><td>85.64</td><td>57.02</td><td>32.03</td><td>24.23</td><td>33.43</td><td>52.25</td><td>62.11</td><td>43.51</td></tr><tr><td></td><td>Proxy-static</td><td>71.14</td><td>54.13</td><td>57.70</td><td>31.10</td><td>24.70</td><td>36.40</td><td>50.60</td><td>56.60</td><td>42.80</td></tr><tr><td></td><td>Analytic</td><td>2.16 × 1011</td><td>3.58 × 1011</td><td>52.12</td><td>26.30</td><td>26.79</td><td>26.27</td><td>48.46</td><td>37.83</td><td>36.30</td></tr><tr><td></td><td>Min-max</td><td> $3 . 9 3 \times 1 0 ^ { 5 }$ </td><td> $6 . 4 5 \times 1 0 ^ { 4 }$ </td><td>50.27</td><td>24.58</td><td>26.28</td><td>26.12</td><td>49.17</td><td>48.93</td><td>37.56</td></tr><tr><td colspan="9">(b) HIP preprocessing</td><td></td><td></td></tr><tr><td>Llama-3.1-8B FP16 reference</td><td></td><td>7.213</td><td>11.388</td><td>80.96</td><td>79.59</td><td>55.12</td><td>79.26</td><td>74.03</td><td>84.10</td><td>75.51</td></tr><tr><td></td><td>Shrink-2.4</td><td>39.16</td><td>53.99</td><td>62.68</td><td>39.86</td><td>24.40</td><td>42.90</td><td>54.93</td><td>67.37</td><td>48.69</td></tr><tr><td></td><td>Proxy-static</td><td>31.38</td><td>43.81</td><td>62.00</td><td>39.90</td><td>25.10</td><td>42.00</td><td>56.50</td><td>65.20</td><td>48.50</td></tr><tr><td></td><td>Analytic</td><td>119.45</td><td>75.27</td><td>58.43</td><td>36.78</td><td>23.63</td><td>38.12</td><td>52.57</td><td>58.53</td><td>44.68</td></tr><tr><td></td><td>Min-max</td><td>6.82 × 105</td><td>6.6 × 105</td><td>52.45</td><td>25.25</td><td>26.02</td><td>26.75</td><td>48.15</td><td>52.11</td><td>38.45</td></tr><tr><td>Llama-3.2-3B</td><td>FP16 reference</td><td>11.048</td><td>16.486</td><td>75.52</td><td>67.85</td><td>46.16</td><td>70.44</td><td>67.32</td><td>78.62</td><td>67.65</td></tr><tr><td></td><td>Shrink-2.4</td><td>123.56</td><td>123.84</td><td>56.80</td><td>34.93</td><td>23.98</td><td>34.70</td><td>51.93</td><td>60.58</td><td>43.82</td></tr><tr><td></td><td>Proxy-static</td><td>94.12</td><td>95.11</td><td>57.30</td><td>34.30</td><td>24.10</td><td>34.50</td><td>52.20</td><td>62.30</td><td>44.10</td></tr><tr><td></td><td>Analytic</td><td>179.82</td><td>196.67</td><td>57.24</td><td>31.40</td><td>23.29</td><td>31.42</td><td>52.33</td><td>56.73</td><td>42.07</td></tr><tr><td></td><td>Min-max</td><td>3.43 × 105</td><td>2.65 × 105</td><td>52.23</td><td>26.05</td><td>25.34</td><td>26.24</td><td>50.12</td><td>46.39</td><td>37.73</td></tr><tr><td>Qwen3-8B</td><td>FP16 reference</td><td>9.715</td><td>15.363</td><td>77.75</td><td>80.93</td><td>56.48</td><td>74.92</td><td>67.64</td><td>86.61</td><td>74.05</td></tr><tr><td></td><td>Shrink-2.4</td><td>32.46</td><td>36.32</td><td>68.39</td><td>52.82</td><td>33.11</td><td>48.17</td><td>59.83</td><td>68.99</td><td>55.22</td></tr><tr><td></td><td>Proxy-static</td><td>22.65</td><td>29.75</td><td>68.80</td><td>57.60</td><td>34.60</td><td>51.20</td><td>62.90</td><td>74.40</td><td>58.20</td></tr><tr><td></td><td>Analytic</td><td>46.98</td><td>45.13</td><td>65.07</td><td>50.34</td><td>29.44</td><td>45.04</td><td>57.93</td><td>67.13</td><td>52.49</td></tr><tr><td></td><td>Min-max</td><td>5.06 × 104</td><td>1.91 × 104</td><td>49.46</td><td>24.24</td><td>26.96</td><td>25.55</td><td>49.88</td><td>43.21</td><td>36.55</td></tr></table>

Whitening. Whitening follows the same template as HIP, a change of basis $T$ that preserves the linear map, but drops the orthogonality constraint and thereby changes the geometry of the reconstruction objective. Let $T$ be an invertible matrix and set

$$
\widetilde { W } = W T , \qquad \widetilde { X } = T ^ { - 1 } X , \qquad \widetilde { H } = T ^ { - 1 } H T ^ { - \top } .
$$

Unlike orthogonal rotations, such transformations alter the Hessian spectrum and hence its effective rank. For positive definite $H ,$ choosing $T$ with $T T ^ { \top } = H , { \bf e . g . } T = H ^ { 1 / 2 }$ , whitens the activations: $\widetilde { H } = I , r _ { \mathrm { e f f } } ( \widetilde { H } ) = d ,$ and the reconstruction objective reduces to the plain weight error $\| \widetilde { W } - \widetilde { Q } \| _ { F } ^ { 2 }$ for which coordinatewise rounding is optimal. The cost is that the whitening moves the anisotropy from the Hessian into the weights: the rows of $\widetilde { W } = W H ^ { 1 / 2 }$ have covariance proportional to H rather than being isotropic, and the quantization grid, mapped back to the original basis, is no longer a cube lattice but its image under $\dot { H } ^ { - 1 / 2 }$ . Coordinatewise quantization in the whitened basis therefore optimizes an isotropic objective over a different set of representable weights.

Diagonal rescaling. Between rotation and full whitening sits the per-channel rescaling used by SmoothQuant (Xiao et al., 2023) and AWQ (Lin et al., 2024). Let $T \stackrel { \cdot } { = } D$ be diagonal with positive entries, so that

$$
\widetilde { W } = W D , \qquad \widetilde { X } = D ^ { - 1 } X , \qquad \widetilde { H } = D ^ { - 1 } H D ^ { - 1 } .
$$

The natural choice $D = \mathrm { d i a g } ( H ) ^ { 1 / 2 }$ divides each input channel by its root-mean-square activation, turning $\widetilde { H }$ into the correlation matrix of the input channels. The resulting matrix has $\operatorname { t r } ( { \widetilde { H } } ) = d ,$

![](images/f936c060d932a28310e24c36acad4359c4f0ff2d7b40c8422727ae8acfd0ac8a.jpg)

![](images/5c8a7150841f47417ba61cef1225e57d3a233c78b9ce16c2074d75bf8146d2a9.jpg)  
Figure 10: Target-local preprocessing on ten difficult W2 GPTQ rows selected across the 32 blocks of Llama-3.1-8B. Top: absolute log ratio between the best-searched scale and the Gaussian analytic scale. Bottom: best-searched local reconstruction loss on a logarithmic axis. Each point represents one matrix-row pair selected from a prespecified depth bin; horizontal positions indicate the actual block indices. All four conditions share the same unrotated, quantized upstream stream and pre-target data. Transforms are applied only to the selected target, and losses use a common normalization based on the raw row and Gram. Minima are the best evaluated values, not certified global optima.

while the off-diagonal entries are the correlation coefficients $H _ { i j } / \sqrt { H _ { i i } H _ { j j } }$ . Consequently

$$
r _ { \mathrm { e f f } } ( \widetilde { H } ) = \frac { d } { \Vert \widetilde { H } \Vert _ { \mathrm { o p } } } ,
$$

which equals d exactly when the channels are uncorrelated, in which case diagonal rescaling coincides with whitening.

Matched target-local comparison. To distinguish scale agreement from reconstruction quality, we examine ten difficult W2 targets from Llama-3.1-8B, selecting one matrix-row pair from each of ten fixed depth bins. In each of the bins, we select the matrix with the largest ratio of its summed row-wise best-searched GPTQ reconstruction error to signal energy. Within that matrix, we select the output row with the largest normalized best-searched GPTQ loss.

For each target, the raw, HIP, diagonal-scaling+HIP, and whitening+HIP conditions share the same pre-target weights and activation Hessian. Preprocessing is applied only to the selected target, followed by local GPTQ evaluation across candidate scales. Figure 10 and Table 9 show that whitening brings the best-searched scale closer to the Gaussian analytic prediction in these rows, but generally increases the best-searched reconstruction loss relative to HIP alone. Improved scale agreement therefore need not translate into improved reconstruction quality.

## B.5 QUANTIZATION ERROR LANDSCAPES

## B.5.1 GAUSSIAN LANDSCAPES AND CONVERGENCE

We illustrate the finite-width behavior of the normalized RTN loss using W3 symmetric quantization with levels $\{ - 3 , \ldots , 3 \}$ . For $z \sim \mathcal { N } ( 0 , I _ { d } )$ , we consider the identity Gram and random orthogonal projectors of rank k, yielding

$$
L _ { d } ( s ) = \frac { \| z - Q _ { s } ( z ) \| _ { 2 } ^ { 2 } } { d } , \qquad L _ { d , k } ( s ) = \frac { \| U ^ { \top } ( z - Q _ { s } ( z ) ) \| _ { 2 } ^ { 2 } } { k } ,
$$

respectively, where $U ^ { \top } U = I _ { k }$ and U is independent of z. To understand if the effective rank condition (H2) of Theorem 1 is necessary we choose $k \in \{ d , \lceil \sqrt { d } \rceil , \lceil \log d \rceil \}$ , and constant = 8.

Figure 11 compares individual landscapes at increasing widths with the Gaussian reference. While $k = d , \lceil \sqrt { d }$ converge at decreasing rates towards the gaussian landscape, $k = \lceil \log d \rceil$ and $k = 8$ keep their unregular form even at high d..

This matches the theoretical prediction, as $k = d , \lceil \sqrt { d }$ satisfy the Theorem 1 asymptotic rank condition, whereas the logarithmic and fixed-rank constructions probe regimes outside that sufficient condition. These curves illustrate individual realizations; the uniform error plot in Figure 12 summarizes variability across 20 repetitions and validates Theorem 1.

![](images/64a8a9805f1e7596571d36e73cd0ca3c5a2d4df3a50ea93723c1259b6226a75e.jpg)  
Figure 11: Finite-width W3 RTN landscapes for four effective-rank constructions. Each panel overlays $d \in \{ 4 0 9 6 , 1 6 3 8 4 , 6 5 5 3 6 \}$ and the Gaussian reference (dashed). Curves show one realization per configuration (repetition 0). The horizontal coordinate is $s / s ^ { \star } ( 3 )$ , and all losses are divided by $\mathcal { E } _ { \infty } ( 3 , s ^ { \star } ( 3 ) ,$ ). Only $0 . 3 5 \leq s / s ^ { \star } ( 3 ) \leq 2 . 5$ is displayed.

## B.6 LOCAL RECONSTRUCTION SENSITIVITY WITH HIP

We repeat the local reconstruction sensitivity analysis of Section 5.3 with HIP preprocessing. Figure 13, the HIP counterpart of Figure 3, shows a similar approximately exponential decrease in median sensitivity with increasing bit-width.

To compare sensitivity magnitudes directly, Figure 14 overlays the results with and without HIP at $\delta = 5 \%$ For each model and bit-width, we also report the median paired ratio $C _ { \mathrm { s y m } } ^ { \mathrm { H I P } } / C _ { \mathrm { s y m } } ^ { \mathrm { n o H I P } }$ matching calibration seed, matrix, and row ID. Only pairs with positive, finite sensitivity in both conditions are included. Each condition is evaluated around its own best-searched scale.

![](images/a21121335508450845e284401ed76ceb2caa6d8e40dada0d7def6517450b4d27.jpg)  
Figure 12: Uniform-error estimates for W3 Gaussian quantization landscapes. Median maximum error over the evaluated scale grid S $\begin{array} { r } { \mathrm { , \operatorname* { m a x } } _ { s \in \mathcal { S } } | L _ { d } ( s ) - \mathcal { E } _ { \infty } ( M _ { 3 } , s ) | } \end{array}$ , as a function of width d. Shaded regions show the 10th–90th percentiles across 20 repetitions. Curves correspond to effective ranks $k = d , \lceil \sqrt { d } \rceil$ , [log d], and 8. Errors decrease in the identity and square-root-rank settings, while the logarithmic- and fixed-rank settings show no comparable decrease over the tested widths.

An illustrative reconstruction landscape. Figure 15 compares the Gaussian reference with empirical RTN and GPTQ landscapes for one fixed row of Llama-3.1-8B at W2, W3, and W8, with and without HIP. The RTN landscapes resemble the Gaussian reference in overall shape, although their minimizing scales can differ. The basin insets illustrate how HIP brings the analytic scale closer to the empirical minimizer in this example.

GPTQ compensation also changes the shape of the local basin. However, this single-row comparison does not establish a general flattening effect. Moreover, the landscapes are displayed after division by the bit-dependent Gaussian minimum, whereas our sensitivity statistic excludes this additional normalization. Cross-precision sensitivity comparisons therefore rely on $C _ { \mathrm { s y m } }$ under the common loss normalization, rather than on the visual sharpness of these curves.

![](images/9d83d54f00c4e34652940eb466780a028f7b8284a5132cc9f82bf24822800e03.jpg)  
Figure 13: Local GPTQ scale sensitivity after HIP preprocessing. Colored curves show the median symmetric finite-window sensitivity $C _ { \mathrm { s y m } }$ around each row's best-searched scale at the tested precisions W2, W3, W4, W6, and W8. Panels use relative scale perturbations $\delta \in \{ 2 \% , 5 \% , 1 0 \% \}$ . The black dashed curve shows the Gaussian scale-invariant differential curvature $\kappa ,$ which is identical across panels. Medians pool the selected rows across all studied matrices and three calibration seeds for each model and bit-width.

![](images/e4b778e71738e83623f23bd23663cf67ff4e47d0864910c632f3ca16cdf9014e.jpg)  
Llama-3.2-1BLlama-3.2-3BLlama-3.1-8BQwen3-8BOPT-125MNo HIP-HIP  
Figure 14: Effect of HIP on local GPTQ scale sensitivity at $\delta = 5 \%$ . Left: median symmetric finitewindow sensitivity $C _ { \mathrm { s y m } }$ by model and bit-width, with filled markers for no HIP and open markers for HIP. Right: median of the paired HIP-to-no-HIP sensitivity ratios, matching model, bit-width, calibration seed, matrix, and row ID. Only pairs with positive, finite sensitivity in both conditions are included. This statistic is the median of individual ratios, rather than the ratio of the pooled medians.

Table 8: W2-ternary per-channel cross-engine comparison with HIP and grid $\{ - s , 0 , s \}$ . Bold values compare selectors within each model and engine, excluding FP16 references. These measurements are separate from the main GPTQ campaign. Lower perplexity and higher accuracy are better.
<table><tr><td>Engine</td><td>Rule</td><td>Wiki2 PPL</td><td>C4 PPL</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>HellaSwag</td><td>WinoGrande</td><td>BoolQ</td><td>Mean</td></tr><tr><td colspan="9">(a) Llama-3.1-8B</td><td></td><td></td></tr><tr><td>FP16 reference</td><td></td><td>7.213</td><td>11.388</td><td>80.96</td><td>79.59</td><td>55.12</td><td>79.26</td><td>74.03</td><td>84.10</td><td>75.51</td></tr><tr><td>GPTQ</td><td>Shrink-2.4</td><td>60.17</td><td>60.98</td><td>60.83</td><td>37.58</td><td>23.63</td><td>41.15</td><td>52.17</td><td>61.25</td><td>46.10</td></tr><tr><td></td><td>Proxy-static</td><td>33.71</td><td>45.22</td><td>63.00</td><td>40.45</td><td>25.51</td><td>43.82</td><td>55.33</td><td>66.76</td><td>49.14</td></tr><tr><td>ResComp</td><td>Shrink-2.4</td><td>38.59</td><td>43.39</td><td>62.79</td><td>36.11</td><td>23.72</td><td>42.37</td><td>55.72</td><td>65.90</td><td>47.77</td></tr><tr><td></td><td>Proxy-static</td><td>29.69</td><td>35.45</td><td>61.59</td><td>36.28</td><td>23.81</td><td>44.93</td><td>55.96</td><td>64.25</td><td>47.80</td></tr><tr><td>QRoNoS</td><td>Shrink-2.4</td><td>61.07</td><td>59.03</td><td>66.54</td><td>45.71</td><td>27.22</td><td>45.12</td><td>56.67</td><td>65.38</td><td>51.11</td></tr><tr><td></td><td>Proxy-static</td><td>38.17</td><td>43.70</td><td>56.58</td><td>32.32</td><td>22.35</td><td>46.87</td><td>57.38</td><td>67.49</td><td>47.17</td></tr><tr><td colspan="9">(b) Qwen3-8B</td><td></td><td></td></tr><tr><td>FP16 reference</td><td></td><td>9.715</td><td>15.363</td><td>77.75</td><td>80.93</td><td>56.48</td><td>74.92</td><td>67.64</td><td>86.61</td><td>74.05</td></tr><tr><td>GPTQ</td><td>Shrink-2.4</td><td>34.16</td><td>38.39</td><td>67.63</td><td>54.50</td><td>31.74</td><td>48.65</td><td>59.43</td><td>70.40</td><td>55.39</td></tr><tr><td></td><td>Proxy-static</td><td>24.41</td><td>32.99</td><td>68.72</td><td>55.47</td><td>32.68</td><td>51.59</td><td>58.80</td><td>67.00</td><td>55.71</td></tr><tr><td>ResComp</td><td>Shrink-2.4</td><td>105.93</td><td>101.35</td><td>56.91</td><td>32.62</td><td>22.70</td><td>34.46</td><td>53.43</td><td>62.91</td><td>43.84</td></tr><tr><td></td><td>Proxy-static</td><td>90.31</td><td>88.28</td><td>59.09</td><td>32.66</td><td>23.63</td><td>36.52</td><td>52.33</td><td>60.98</td><td>44.20</td></tr><tr><td>QRoNoS</td><td>Shrink-2.4</td><td>76.01</td><td>57.94</td><td>67.03</td><td>48.32</td><td>28.24</td><td>44.60</td><td>58.09</td><td>63.36</td><td>51.61</td></tr><tr><td></td><td>Proxy-static</td><td>34.25</td><td>32.72</td><td>68.93</td><td>52.44</td><td>30.03</td><td>50.89</td><td>61.17</td><td>69.17</td><td>55.44</td></tr></table>

<table><tr><td>Preprocessing</td><td>Median scale gap  $\vert \log ( s _ { \mathrm { s e a r c h } } / s _ { \mathrm { a n } } \bar { ) } \vert$ </td><td>Gap lower than HIP</td><td>Median best-loss ratio relative to HIP</td><td>Loss lower than HIP</td></tr><tr><td>None</td><td>0.445</td><td>0/10</td><td>3.46</td><td>0/10</td></tr><tr><td>HIP</td><td>0.187</td><td></td><td>1.00</td><td></td></tr><tr><td>Diagonal + HIP</td><td>0.0298</td><td>8/10</td><td>3.63</td><td>0/10</td></tr><tr><td>Whitening + HIP</td><td>0.00406</td><td>10/10</td><td>20.8</td><td>0/10</td></tr></table>

Table 9: Target-local W2 GPTQ preprocessing on ten selected difficult rows of Llama-3.1-8B-Instruct. Counts compare each target against HIP alone. Lower is better in both numeric columns.

Llama-3.1-8B reconstruction landscapes at W2, W3, and W8 RTN local GPTQ  
![](images/956ba99aacb6096875e74b46f1e9faea74e0f034e98c2883125945c42e8b4b1b.jpg)  
Figure 15: Scale-dependent reconstruction landscapes for Llama-3.1-8B, block 0, k-pro j, output row 150. Columns show W2, W3, and W8; rows show the Gaussian reference and empirical RTN and GPTQ losses without and with HIP. Scales and losses are normalized by the Gaussian optimal scale and minimum loss, respectively. Dashed vertical lines mark the analytic scale; markers identify best-searched scales. Insets show the local basins. Empirical losses use inputs from the upstream GPTQ-quantized network at the corresponding precision. This figure illustrates one fixed row rather than an aggregate over the representative cohort.