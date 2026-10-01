# WinoTS: Wavelet-based Self-Distillation for Time Series Models

Noam Major Kathy Razmadze Yoli Shavit

Bar-Ilan University, Ramat-Gan, Israel

## Abstract

Self-supervised pre-training of time series models is currently dominated by nexttoken prediction and reconstruction objectives. In continuous-valued domains, these paradigms often waste model capacity on high-frequency, point-wise noise at the expense of learning invariant structure. While invariance-based self-distillation has proven highly effective in computer vision, its application to temporal data remains largely underexplored. Effectively adapting such methods to time series requires carefully designed augmentations: spatial operations like cropping can shift the timing of repeating cycles or distort the signal, while basic jittering may provide limited variation. We introduce Wavelet-based self-distillation for time series (WinoTS), an invariance-based pre-training paradigm designed specifically for temporal signals. At its core, WinoTS leverages time-frequency augmentations to construct multi-scale structural views without distorting underlying signal dynamics. Across extensive evaluations, WinoTS outperforms state-of-the-art baselines in long-term forecasting, cross-domain zero-shot transfer, and unsupervised anomaly detection. Notably, linear probing on frozen WinoTS representations frequently surpasses fully supervised models trained from scratch. Systematic ablations demonstrate that WinoTS is a flexible, architecture-agnostic framework yielding gains across time series backbones, and establish that time-frequency transformations provide a principled alternative to vision-style spatial augmentations.

## 1 Introduction

Self-supervised learning (SSL) has become the primary driver for learning general-purpose representations across modalities [Devlin et al., 2019, Chen et al., 2020, Caron et al., 2021, Bommasani et al., 2021]. In computer vision, joint-embedding self-distillation frameworks such as DINO [Caron et al., 2021] construct semantic representations by enforcing consistency across augmented views using a momentum teacher. These methods operate on the principle of semantic invariance: a sample’s core structure should remain recognizable across label-preserving transformations. However, the efficacy of self-distillation depends on identifying augmentations that induce meaningful invariance without destroying structural integrity.

Currently, self-supervised pre-training for time series remains heavily dominated by generative paradigms, specifically Next Token Prediction (NTP) Ansari et al. [2024], Das et al. [2024] and Masked Auto Encoding (MAE) Nie et al. [2023], Dong et al. [2024]. In continuous-valued domains, however, these objectives often force models to dedicate significant capacity to predicting highfrequency, point-wise noise rather than capturing underlying structural invariants. While jointembedding frameworks based on semantic invariance offer a compelling alternative, their application to temporal data remains largely underexplored.

A central challenge in extending invariance-based self-supervision to time series lies in the design of effective data augmentations. In computer vision, operations such as localized cropping or color jitter alter surface appearance while reliably preserving object identity. Applying analogous transformations to temporal sequences, however, can disrupt fundamental signal properties. For example, cropping a segment from a multi-cycle sequence, such as an electrocardiogram (ECG) heartbeat or an industrial sensor trace, can truncate recurring seasonal patterns, distort global trends, or shift phase alignment. Conversely, basic point-wise operations like Gaussian noise or minor jittering alter individual values without inducing meaningful structural variation, which may yield a weak self-supervisory signal. Effective adaptation therefore requires augmentation strategies that create semantic variation while explicitly preserving the structural and temporal integrity of the underlying signal.

To bridge this gap, we introduce Wavelet-based self-distillation for time series (WINOTS), an invariance-based pre-training paradigm designed specifically for continuous temporal signals. WINOTS leverages the Discrete Wavelet Transform (DWT) to establish a principled augmentation mechanism localized simultaneously in time and frequency. Given an input window, WINOTS decomposes each channel into multi-resolution approximation and detail components to construct asymmetric, full-length views without altering sequence length or discarding temporal context. Specifically, low-frequency approximation coefficients (capturing macro trends and seasonality) remain intact across views, while high-frequency detail subspaces are asymmetrically perturbed: "easy" views smooth these details via soft-thresholding, whereas "hard" views corrupt them with controlled noise. The student network is then optimized to match the teacher’s centered and sharpened output distribution across augmented view pairs through a standard DINO projection head. To avoid overfitting to the boundary behaviors or phase characteristics of a single basis, WINOTS stochastically samples wavelet functions across the Daubechies, Symlets, and Coiflets families Daubechies [1988, 1992].

We evaluate WINOTS through a comprehensive suite of experiments spanning 13 standard forecasting benchmarks, 12 cross-domain transfer scenarios, and 5 multivariate anomaly detection datasets. Across these tasks, WINOTS enhances downstream representation quality, achieving the top average rank on in-domain forecasting and outperforming direct supervised baselines as well as specialized SSL models like TimeSiam Dong et al. [2024] and TS2Vec Yue et al. [2022]. Under linear probing, frozen WINOTS features demonstrate strong linear separability, frequently surpassing models trained end-to-end from scratch. Furthermore, controlled ablations confirm that wavelet-based view generation provides a principled alternative to aggressive vision primitives (e.g., cropping), delivers competitive performance relative to generative objectives such as MAE and NTP, and yields error reductions across diverse backbone architectures.

In summary, our key contributions are:

• We introduce WINOTS, an invariance-based self-distillation paradigm tailored for continuous temporal signals that addresses the limitations of generative pre-training by optimizing for structural invariants rather than point-wise noise prediction.

• We propose a multi-resolution wavelet view generator leveraging the Discrete Wavelet Transform (DWT) that preserves low-frequency trend and seasonal components while asymmetrically perturbing high-frequency detail coefficients via soft-thresholding and noise corruption, constructing context-preserving view pairs for self-distillation.

• We demonstrate across 13 forecasting benchmarks, 12 cross-domain transfer tasks, and 5 multivariate anomaly detection datasets that WINOTS achieves superior performance over state-of-the-art baselines. Controlled ablations confirm that wavelet-based view generation provides a principled alternative to vision primitives, rivals generative objectives, and delivers gains across diverse time series backbones.

## 2 Related Work

Generative vs. Latent-Alignment Pre-training for Time Series Models. Self-supervised pretraining for temporal data broadly falls into two primary paradigms: generative modeling and latent alignment. Generative approaches learn representations by predicting, denoising, or reconstructing raw temporal signals in the target data domain. These include masked autoencoders Nie et al. [2023], autoregressive next-token prediction Ansari et al. [2024], Das et al. [2024], diffusion frameworks Wang et al. [2025], and past-to-present target-domain reconstruction via Siamese networks Dong et al. [2024]. Although effective for forecasting, these point-level objectives force models to dedicate significant capacity to predicting fine-grained numerical values and high-frequency noise at the expense of capturing broader invariant structure Major et al. [2026]. Conversely, latentalignment paradigms directly optimize representation space by enforcing consistency across different views of the same signal without signal-space decoders. Early work relied primarily on contrastive learning and Siamese instance matching Oord et al. [2018], Yue et al. [2022], whereas recent advances have embraced non-contrastive self-distillation and momentum teacher–student objectives inspired by DINO Pieper et al. [2023], Zhao et al. [2024], Moakher et al. [2026]. However, existing latentalignment approaches predominantly rely on basic spatial masking or cropping, and their empirical validation remains largely constrained to narrow benchmark sets or single architectural backbones.

Time-Series Augmentations for Latent Alignment. The representation quality of latent-alignment frameworks depends fundamentally on the choice of view generation. Existing methods predominantly rely on spatial or time-domain transformations, such as cropping, temporal jittering, or channel masking Yue et al. [2022], Moakher et al. [2026]. However, these vision-inspired operations can severely disrupt temporal dynamics: aggressive cropping removes essential seasonal and macro-trend context, while point-wise noise degrades high-frequency phase and amplitude relationships. Recent studies have highlighted this trade-off, analyzing the precision–invariance spectrum under standardized benchmarks Major et al. [2026]. Rather than applying spatial or heuristic transformations in the time domain, WINOTS introduces a principled wavelet-based view generation framework that operates natively across time and frequency, constructing multi-scale views that induce invariance while strictly preserving signal dynamics.

## 3 Method

We present WINOTS, a self-supervised pre-training framework that adapts joint-embedding selfdistillation to temporal data via multi-resolution time-frequency view generation. Instead of manipulating time-domain samples directly, WINOTS maps input sequences into wavelet coefficient space via the Discrete Wavelet Transform (DWT), constructing asymmetric, full-length views that preserve macroscopic trend and seasonal context while selectively perturbing high-frequency details. In the following subsections, we formalize the wavelet-based view generation mechanism, detail the asymmetric teacher–student self-distillation objective, and describe our stochastic basis sampling scheme.

## 3.1 Background

We first provide the necessary mathematical background on self-distillation and multi-resolution signal representations required to ground our framework.

Self-Distillation with No Labels (DINO). Originally introduced in computer vision, DINO Caron et al. [2021] serves as a general joint-embedding framework for learning semantic representations without negative pairs or labels. The framework consists of two networks sharing identical structural topologies: a student network $g _ { \theta _ { s } , \phi _ { s } }$ optimized via gradient descent, and a teacher network $g _ { \theta _ { t } , \phi _ { t } }$ whose parameters are updated as an exponential moving average (EMA) of the student weights. Each network is parameterized as the composition of an encoder backbone $f _ { \theta }$ and a non-linear projection head $h _ { \phi } \colon$

$$
g _ { \theta , \phi } = h _ { \phi } \circ f _ { \theta } .\tag{1}
$$

Given two different augmented views of an input sample, denoted $x _ { s }$ and $x _ { t } ,$ the student and teacher process them to produce output logits $z _ { s } = g _ { \theta _ { s } , \phi _ { s } } ( x _ { s } )$ and $z _ { t } ~ = ~ g _ { \theta _ { t } , \phi _ { t } } ( x _ { t } )$ . These logits are converted into probability distributions using a softmax operator. The student’s predictive distribution is parameterized by a temperature hyperparameter $\tau _ { s } \mathrm { : }$

$$
P _ { s } ^ { ( i ) } ( z _ { s } ) = \frac { \exp ( z _ { s } ^ { ( i ) } / \tau _ { s } ) } { \sum _ { j } \exp ( z _ { s } ^ { ( j ) } / \tau _ { s } ) } .\tag{2}
$$

To prevent representation collapse across channels, the teacher’s target distribution is centered by a running mean vector c and sharpened by a distinct target temperature $\tau _ { t } < \tau _ { s }$

$$
P _ { t } ^ { ( i ) } ( z _ { t } ) = \frac { \exp ( ( z _ { t } ^ { ( i ) } - c ^ { ( i ) } ) / \tau _ { t } ) } { \sum _ { j } \exp ( ( z _ { t } ^ { ( j ) } - c ^ { ( j ) } ) / \tau _ { t } ) } .\tag{3}
$$

![](images/f76ed1f21781a51342d85f89d31f2238874a95f1df0ffba08c6c23956e887519.jpg)  
Figure 1: Architectural pipeline of WINOTS. The wavelet generator produces $Q$ easy and V hard views. The teacher processes only easy views, while the student processes both. Each teacher easy view is matched to all student views except its identical easy counterpart; every hard view is matched to all teacher views.

The framework is optimized by minimizing the cross-entropy loss between the teacher’s centered and sharpened target distribution and the student’s distribution:

$$
\mathcal { L } _ { \mathrm { D I N O } } = - \sum _ { i } P _ { t } ^ { ( i ) } ( z _ { t } ) \log P _ { s } ^ { ( i ) } ( z _ { s } ) .\tag{4}
$$

Gradients are backpropagated strictly through the student network. After each optimization step, the teacher’s parameters are updated via a momentum schedule: $\theta _ { t } \gets \lambda \theta _ { t } + ( 1 - \lambda ) \theta _ { s }$ and $\phi _ { t } \gets$ $\lambda \phi _ { t } + ( 1 - \lambda ) \phi _ { s }$

Discrete Wavelet Decomposition. The Discrete Wavelet Transform (DWT) provides a standard mathematical framework for analyzing non-stationary signals across multiple resolutions simultaneously. Formally, let $x \in \mathbb { R } ^ { T }$ denote a single-channel temporal sequence. A J-level DWT parameterized by a chosen wavelet basis w recursively decomposes the signal into a low-frequency approximation band and multiple high-frequency detail bands. This decomposition is achieved by applying a low-pass filter h and a high-pass filter $^ { g , }$ followed by dyadic downsampling at each scale:

$$
\begin{array} { l } { { a _ { j } = \left( h \star a _ { j - 1 } \right) \downarrow 2 , } } \\ { { d _ { j } = \left( g \star a _ { j - 1 } \right) \downarrow 2 , \qquad j = 1 , \ldots , J , } } \end{array}\tag{5}
$$

where $a _ { 0 } ~ = ~ x$ . This forward projection maps the raw signal into an orthogonal wavelet space $\mathcal { W } _ { w } x = ( a _ { J } , d _ { J } , \dots , d _ { 1 } )$ . For an orthonormal basis, the original signal can be perfectly recovered via the inverse reconstruction formula:

$$
\boldsymbol { x } = \sum _ { k } a _ { J , k } \phi _ { J , k } + \sum _ { j = 1 } ^ { J } \sum _ { k } d _ { j , k } \psi _ { j , k } ,\tag{6}
$$

where $\phi _ { J , k }$ and $\psi _ { j , k }$ represent the scaling and wavelet functions, respectively. The first term yields the coarse approximation projection $P _ { V _ { I } x }$ , while the second term captures multi-scale localized temporal fluctuations in the orthogonal complement space $V _ { J } ^ { \perp }$

## 3.2 WINOTS: Self-Distillation with Wavelet-based View Construction

WINOTS adapts the general DINO framework to time series by introducing a novel view-generation strategy natively tailored to the temporal domain. Instead of relying on spatial or spatial-spectral approximations, our framework exploits the time-frequency locality of the DWT to construct asymmetric, full-length temporal views (Fig. 1).

Easy View Generation. WINOTS constructs $Q$ easy views $\mathcal { E } ( X ) = \{ X _ { q } ^ { \mathrm { e a s y } } \} _ { q = 1 } ^ { Q }$ by applying nonlinear soft-thresholding to the high-frequency detail bands while leaving each view’s low-frequency approximation coefficients unchanged.

For a given easy view, the modified coefficients are reconstructed in the time domain using the inverse discrete wavelet transform (IDWT), $\mathcal { W } _ { w } ^ { - 1 }$

$$
x ^ { \mathrm { e a s y } } = \mathcal { W } _ { w } ^ { - 1 } \big ( a _ { J } , \eta _ { \tau _ { J } } ( d _ { J } ) , \dots , \eta _ { \tau _ { 1 } } ( d _ { 1 } ) \big ) .\tag{7}
$$

Here, the approximation component $a _ { J }$ remains unchanged, while the detail shrinkage operator $\eta _ { \tau }$ is defined independently for each band using its maximum absolute coefficient:

$$
\eta _ { \tau } ( d ) = \mathrm { s i g n } ( d ) \operatorname* { m a x } ( | d | - \tau , 0 ) , \qquad \tau _ { j } = \rho \operatorname* { m a x } _ { k } | d _ { j , k } | .\tag{8}
$$

This operation sets detail coefficients below $\tau _ { j }$ to zero and attenuates larger coefficients, retaining dominant localized fluctuations without discarding the entire high-pass subspace. The relationship between this operator and classical wavelet shrinkage is discussed in the Supplementary Material, and sensitivity to the shrinkage parameter is evaluated in Table A13.

Hard View Generation. WINOTS further generates V hard views $\{ X _ { v } ^ { \mathrm { h a r d } } \} _ { v = 1 } ^ { V } . \mathrm { A }$ hard view is constructed by injecting controlled Gaussian noise into the multi-scale detail bands while keeping the approximation coefficients $a _ { J }$ unchanged, followed by IDWT reconstruction:

$$
x ^ { \mathrm { h a r d } } = \mathcal { W } _ { w } ^ { - 1 } \big ( a _ { J } , d _ { J } + \varepsilon _ { J } , \ldots , d _ { 1 } + \varepsilon _ { 1 } \big ) , \qquad \varepsilon _ { j , k } \sim \mathcal { N } ( 0 , s ^ { 2 } ) .\tag{9}
$$

For a multivariate hard view, the perturbation scale is sampled independently for each channel. Specifically, for hard view v and channel $^ { c , }$

$$
\begin{array} { r } { s _ { v , c } \sim \mathcal { U } ( s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } ) , \qquad \varepsilon _ { v , j , k , c } \sim \mathcal { N } ( 0 , s _ { v , c } ^ { 2 } ) . } \end{array}\tag{10}
$$

The sampled scale $s _ { v , c }$ is shared across all detail levels of channel c within that hard view. The effect of the perturbation design is further analyzed through the augmentation and transform ablations in Tables A10 and A12.

Multi-Resolution Basis Randomization. A single wavelet basis introduces specific mathematical artifacts rooted in its unique phase alignment, compact support limits, and boundary behaviors. To prevent the model from overfitting to these basis-specific traits, WINOTS samples distinct wavelet bases independently for the teacher and student views from an orthogonal pool :

$$
w ^ { \mathrm { e a s y } } , w ^ { \mathrm { h a r d \stackrel { i . i . d . } { \sim } } \mathcal { U } ( \mathcal { P } ) , }\tag{11}
$$

where the pool consists of six compactly supported wavelets spanning three distinct mathematical families: $\mathcal { P } = \mathrm { \{ s y m 4 } $ , sym6, sym8, db4, db6, coif2 . These families introduce diverse signal processing properties into the augmentation pipeline. Daubechies (db) wavelets provide a minimum-phase asymmetric design, Symlets (sym) optimize for near-symmetric phase behavior to minimize phase distortion, and Coiflets (coif) balance symmetry by enforcing vanishing moment constraints on both the scaling and wavelet functions. The properties of the sampled wavelet families are summarized in Supplementary Table A1. The impact of alternative wavelet families, decomposition depths, and transform variants is evaluated in Supplementary Tables A12–A15.

Architecture-Agnostic Integration. For a multivariate input window $X \in \mathbb { R } ^ { T \times C }$ , WINOTS applies the wavelet transform channel-wise. Within a single view, all channels share the same sampled wavelet basis w. For each hard view $v ,$ however, every channel c independently samples a perturbation scale $s _ { v , c } \sim \mathcal { U } ( s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } )$ shared across its detail levels, along with an independent noise realization $\varepsilon _ { v , j , k , c } \sim \mathcal { N } ( 0 , s _ { v , c } ^ { 2 } )$ . This design preserves cross-channel view coordination while preventing artificial high-frequency co-dependence.

Crucially, because view generation operates in the wavelet coefficient domain prior to full-length reconstruction, the sequence length T remains strictly invariant across all views. This makes WINOTS natively model-agnostic: any time series backbone $f _ { \theta }$ designed for uniform sequence lengths can be seamlessly embedded without modifying its structural parameters or positional encodings (empirically demonstrated across backbone families in Supplementary Table $\mathbf { A } \bar { 9 } )$

The WINOTS Objective. For an input window X, let ${ \mathcal E } ( X ) = \{ X _ { q } ^ { \mathrm { e a s y } } \} _ { q = 1 } ^ { Q }$ and $ \mathcal { H } ( X ) ~ =$ $\{ X _ { v } ^ { \mathrm { h a r d } } \} _ { v = 1 } ^ { V }$ denote the sets of $Q$ easy views and V hard views, respectively. The momentum teacher processes exclusively easy views from ${ \mathcal { E } } ( X )$ , whereas the student network processes views from both ${ \mathcal { E } } ( X )$ and $\mathcal { H } ( X )$

Let $P _ { t , q } ^ { \mathrm { e a s y } }$ denote the teacher’s target distribution for easy view $q ,$ and let $P _ { s , u } ^ { \mathrm { e a s y } }$ and $P _ { s , v } ^ { \mathrm { h a r d } }$ denote the student’s output distributions for easy view u and hard view v. Following standard cross-view self-distillation, valid easy-teacher/easy-student pairs exclude identical view realizations:

$$
\mathcal { T } _ { \mathrm { e e } } = \{ ( q , u ) : 1 \leq q , u \leq Q , \ q \neq u \} ,\tag{12}
$$

whereas all easy-teacher/hard-student pairs are valid:

$$
\mathcal { T } _ { \mathrm { e h } } = \left\{ ( q , v ) : 1 \leq q \leq Q , 1 \leq v \leq V \right\} .\tag{13}
$$

The WINOTS objective minimizes cross-entropy across all valid view combinations:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { W I N o T S } } = \frac { 1 } { \left| \mathcal { T } _ { \mathrm { e e } } \right| + \left| \mathcal { T } _ { \mathrm { e h } } \right| } \Bigg [ \displaystyle \sum _ { ( q , u ) \in \mathcal { T } _ { \mathrm { e e } } } H \big ( \mathrm { s g } \left[ P _ { t , q } ^ { \mathrm { e a s y } } \right] , P _ { s , u } ^ { \mathrm { e a s y } } \big ) } \\ { + \displaystyle \sum _ { ( q , v ) \in \mathcal { T } _ { \mathrm { e h } } } H \big ( \mathrm { s g } \left[ P _ { t , q } ^ { \mathrm { e a s y } } \right] , P _ { s , v } ^ { \mathrm { h a r d } } \big ) \Bigg ] , } \end{array}\tag{14}
$$

where sg[ ] denotes the stop-gradient operator. Here, $| \mathcal { T } _ { \mathrm { e e } } | = Q ( Q - 1 )$ and $| \mathcal { T } _ { \mathrm { e h } } | = Q V$ , yielding an average over $Q ( Q + V - \mathrm { \bar { 1 } } )$ valid pair combinations (alternative pairing configurations are evaluated in Supplementary Table A16).

Crucially, because wavelet bases $w ^ { \mathrm { e a s y } }$ and $w ^ { \mathrm { h a r d } }$ are sampled independently per view pair, the teacher and student evaluate input representations across distinct time-frequency coordinate frames. This basis mismatch introduces subtle phase and boundary shifts across shared low-frequency bands, forcing the encoder to learn representations invariant to coordinate choices and cross-basis phase leakage. We detail the projection head specifications, teacher momentum schedules, and complete hyperparameter setups in Supplementary Tables A1 and A2.

Table 1: Average in-domain forecasting over horizons 96, 192, 336, 720 with context length 336 (lower is better). Red and blue indicate best and second-best results. WINOTS uses full fine-tuning; WINOTS-LP freezes the backbone and trains only the forecasting head. TimeSiam and TS2Vec are self-supervised; the remaining baselines are supervised.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Ours</td><td colspan="2">Self-Supervised</td><td colspan="9">Supervised</td></tr><tr><td>WINOTS (ours)</td><td>WINOTS-LP (ours)</td><td>TimeSiam Dong et al. [2024]</td><td>TS2Vec Yue et al. [2022]</td><td>TimeMixer Wang et al. [2024]</td><td>TimeBase Huang et al. [2025]</td><td>SparseTSF Lin et al. [2024]</td><td>PatchTST Nie et al. [2023]</td><td>DLinear Zeng et al. [2023]</td><td>iTransformer Liu et al. [2023]</td><td>FEDformer Zhou et al. [2022]</td><td>TimesNet Wu et al. [2023]</td><td>Autoformer Wu et al. [2021]</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td></tr><tr><td>ETTh1</td><td>0.411 0.423 0.347 0.386</td><td>0.416 0.423</td><td>0.426 0.443</td><td>0.817 0.669</td><td>0.436 0.444</td><td>0.409 0.415</td><td>0.423 0.430</td><td>0.437 0.438</td><td>0.465</td><td>0.469 0.448</td><td>0.446 0.484</td><td>0.490 0.471</td><td>0.465 0.568 0.518</td></tr><tr><td>ETTh2</td><td>0.346 0.379</td><td>0.347 0.385 0.362</td><td>0.362 0.402</td><td>1.957 1.108</td><td>0.349 0.391</td><td>0.358 0.399</td><td>0.372 0.406</td><td>0.380 0.407</td><td>0.458 0.462</td><td>0.379 0.397</td><td>0.419 0.460</td><td>0.408 0.405</td><td>0.526 0.505</td></tr><tr><td>ETTml</td><td></td><td>0.386</td><td>0.348 0.384</td><td>0.670 0.583</td><td>0.362 0.388</td><td>0.367 0.386</td><td>0.368 0.394</td><td>0.393 0.401</td><td>0.377 0.396</td><td>0.408 0.410</td><td>0.441 0.459</td><td>0.403 0.412</td><td>0.665 0.545</td></tr><tr><td>ETTm2</td><td>0.250 0.308</td><td>0.252 0.308</td><td>0.260 0.321</td><td>0.912 0.676</td><td>0.261 0.321</td><td>0.260 0.317</td><td>0.264 0.320</td><td>0.282 0.327</td><td>0.317 0.371</td><td>0.291 0.334</td><td>0.329 0.377</td><td>0.296 0.332</td><td>0.356 0.401</td></tr><tr><td>Weather</td><td>0.224 0.262</td><td>0.238 0.272</td><td>0.228 0.264 0.249</td><td>1.021 0.727</td><td>0.226 0.265</td><td>0.254 0.290</td><td>0.235 0.277</td><td>0.266 0.284</td><td>0.249 0.300</td><td>0.261 0.281</td><td>0.311 0.363</td><td>0.259 0.287</td><td>0.331 0.373</td></tr><tr><td>Electricity</td><td>0.163 0.254 0.377 0.407</td><td>0.167 0.260 0.397 0.419</td><td>0.158 0.412 0.431</td><td>0.373 0.451 0.831 0.639</td><td>0.163 0.255 0.458 0.454</td><td>0.176 0.261 0.450 0.461</td><td>0.166 0.257 0.440 0.458</td><td>0.365 0.299 0.365 0.405</td><td>0.173 0.275 0.317 0.414</td><td>0.183 0.274 0.358 0.404</td><td>0.256 0.360 0.825</td><td>0.201 0.298 0.424 0.449</td><td>0.321 0.392</td></tr><tr><td>Exchange</td><td>0.204 0.258</td><td>0.260 0.315</td><td>0.213 0.273</td><td>0.274 0.365</td><td>0.213 0.277</td><td>0.292 0.307</td><td>0.196 0.246</td><td>0.270 0.304</td><td>0.258 0.319</td><td>0.272 0.300</td><td>0.675 0.310 0.402</td><td>0.274 0.294</td><td>0.793 0.656</td></tr><tr><td>Solar</td><td>0.411 0.276</td><td>0.420 0.282</td><td>0.407 0.280</td><td>0.938 0.545</td><td>0.430 0.310</td><td>0.447 0.293</td><td>0.415 0.267</td><td>0.471 0.304</td><td>0.451 0.320</td><td>0.479 0.326</td><td>0.640 0.391</td><td>0.638 0.335</td><td>0.740 0.631 0.445</td></tr><tr><td>Traffic</td><td>0.678 0.503</td><td>0.684 0.502</td><td>0.680 0.504</td><td>0.672 0.515</td><td>0.686 0.504</td><td>0.705 0.518</td><td>0.695 0.518</td><td>0.775 0.529</td><td>0.695 0.527</td><td>0.780 0.528</td><td>0.799 0.562</td><td>0.754 0.525</td><td>0.745 0.799 0.563</td></tr><tr><td>AQShunyi AQWan</td><td>0.776 0.494</td><td>0.784 0.496</td><td>0.778 0.495</td><td>0.765 0.509</td><td>0.784 0.497</td><td>0.811 0.511</td><td>0.795 0.509</td><td>0.873 0.517</td><td>0.798 0.520</td><td>0.883 0.519</td><td>0.814 0.526</td><td>0.846 0.511</td><td>0.870 0.552</td></tr><tr><td>CzeLan</td><td>0.223 0.262 0.230</td><td>0.267</td><td>0.232 0.287</td><td>0.333 0.413</td><td>0.230 0.282</td><td>0.274 0.325</td><td>0.238 0.286</td><td>0.266 0.299</td><td>0.354 0.385</td><td>0.267 0.295</td><td>0.333 0.386</td><td>0.307 0.325</td><td>0.655 0.580</td></tr><tr><td>PM2.5</td><td>0.418 0.420</td><td>0.421 0.421</td><td>0.419 0.426</td><td>0.428 0.478</td><td>0.424 0.429</td><td>0.445 0.450</td><td>0.423 0.450</td><td>0.426 0.426</td><td>0.423 0.467</td><td>0.428 0.427</td><td>0.455 0.493</td><td>0.438 0.434</td><td>0.475 0.471</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 4 Experimental Setup

Datasets and evaluation protocols. We evaluate forecasting on 13 benchmarks: ETTh1, ETTh2, ETTm1, ETTm2<sup>1</sup>, Weather<sup>2</sup>, Electricity<sup>3</sup>, Traffic<sup>4</sup>, Exchange and Solar Energy [Lai et al., 2018], AQShunyi and AQWan [Zhang et al., 2017], CzeLan [Poyatos et al., 2020], and PM2.5. We report MSE and MAE averaged over horizons 96, 192, 336, 720 . For anomaly detection, we follow the TSLib evaluation protocol and report point-adjusted precision, recall, and F1; complete detector and evaluation details are provided in the supplementary material.

WINOTS is evaluated under two downstream adaptation protocols: by default, WINOTS employs full fine-tuning, where all model parameters are updated end-to-end on the downstream task. To evaluate representation quality, we also test under linear probing (denoted by WINOTS-LP), where the pre-trained backbone remains frozen and only a task-specific head is optimized. We compare our framework against eleven state-of-the-art baselines for forecasting and seven baselines for anomaly detection. Unless specified otherwise, all methods use their optimal default backbone architecture (when applicable), optimizer configuration, and total number of optimization steps. For WINOTS, this budget is divided between self-supervised pre-training and downstream adaptation, whereas supervised baselines use their default optimal optimization budget entirely for supervised training.

Unless otherwise stated, WINOTS utilizes in-domain self-supervised pre-training, optimizing on the unlabeled training split of the downstream dataset prior to task-specific adaptation. For cross-domain zero-shot transfer, pre-training and task configuration are performed strictly on the source dataset, and the model is evaluated directly on the target dataset without any target-domain parameter updates.

Implementation Details. WINOTS pre-trains a TimeMixer backbone using symmetric DWT boundary extension, decomposition depth $J { = } 3 $ , shrinkage ratio $\rho { = } 0 . 6$ , and the wavelet pool listed in Supplementary Table A1. For its longest filter (sym8, L=16), this depth satisfies $T \geq \dot { 2 } ^ { J } ( L - 1 ) =$ 120. Each input yields $Q { = } 1$ easy view and $V { = } 1$ hard view. Both views are forwarded through the student, but the same-index easy–easy pair is excluded; hence $\mathcal { T } _ { \mathrm { e e } } = \emptyset$ , and the reported loss reduces to $H ( \mathrm { s g } [ P _ { t } ^ { \mathrm { e a s y } } ] , P _ { s } ^ { \mathrm { h a r d } } )$ . Alternative pairing rules and view counts are evaluated in Supplementary Tables A16 and A18. Inputs are standardized channel-wise using training-split statistics. For each hard view, the noise scale is sampled independently per channel from (0.2, 0.5) and shared across that channel’s detail levels.

The DINO head is a three-layer MLP with hidden width 2048, a 256-dimensional $\ell _ { 2 } \cdot$ -normalized bottleneck, and a weight-normalized output layer with K=1024; the choice of K is evaluated in Supplementary Table A4. We use $\tau _ { s } { = } 0 . 1 , \tau _ { t } { = } 0 . 0 4$ , center momentum 0.9, and a cosine teacher-EMA schedule from 0.9996 to 1.0. After pre-training, the projection head is discarded and the EMA teacher backbone is used for downstream adaptation. Complete pre-training optimization and architecture settings are reported in Supplementary Table A2; the wavelet pool and projection output-dimension study are provided in Supplementary Tables A1 and A4.

Architectural and Design Ablations. To isolate the factors driving performance, we evaluate core design choices in Section 5, including view generation primitives, loss objectives, and encoder backbones. Comprehensive supplementary evaluations, covering synthetic pre-training, backbone transferability, wavelet family choices, view pairing rules, and seed robustness, are detailed in Supplementary Tables A6 and A9–A17.

## 5 Results and Discussion

We systematically evaluate WINOTS across three primary downstream tasks: in-domain forecasting, cross-domain zero-shot transfer, and unsupervised anomaly detection. Through extensive empirical comparisons, we demonstrate that wavelet-based self-distillation consistently outperforms supervised baselines, generalizes across diverse backbone architectures, and offers clear advantages over both spatial edits and generative pre-training objectives.

## 5.1 In-Domain Forecasting, Linear Separability, and Backbone Generalization

Table 1 reports in-domain forecasting performance across 13 standard benchmarks (complete perhorizon results are provided in Supplementary Table A5). We evaluate two downstream protocols using our pre-trained TimeMixer backbone: WINOTS (full end-to-end fine-tuning) and WINOTS-LP (linear probing over the frozen pre-trained backbone). All baseline methods, including the supervised TimeMixer baseline, are fully trained end-to-end from scratch. WINOTS consistently improves upon its supervised TimeMixer counterpart, achieving the top average rank across all evaluated methods. Compared directly against supervised TimeMixer, WINOTS yields clear error reductions on 12 datasets by MSE and all 13 by MAE, indicating that wavelet-based self-distillation captures structural regularities that complement the backbone’s native inductive bias. While specialized architectures retain localized advantages on specific benchmarks—such as DLinear on Exchange or SparseTSF on Solar—WINOTS delivers the most consistent gains across the benchmark suite, ranking among the top two performing methods on 12 out of 13 datasets.

To isolate representation quality from fine-tuning dynamics, we evaluate the frozen backbone under WINOTS-LP, where only a lightweight linear forecasting head is trained. WINOTS-LP outperforms fully supervised TimeMixer on 7 datasets by MSE and 10 by MAE, demonstrating that pre-training extracts highly separable, informative temporal features without requiring full weight updates.

To assess whether these gains generalize beyond TimeMixer, Table 2 compares WINOTS against identical backbones trained from scratch under their default optimal supervised configurations. Across five datasets, WINOTS consistently reduces both MSE and MAE for all tested MLP, Transformer, and TCN-based backbones, confirming that its benefits are broadly model-agnostic. A detailed per-dataset breakdown for each backbone family is provided in Supplementary Table A9.

Table 2: Relative MSE/MAE reduction of WINOTS fine-tuning compared to training the same backbone architectures from scratch under their optimal supervised configurations. Results are averaged across five datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather) and four forecasting horizons (H 96, 192, 336, 720 ). Positive values indicate performance gains.
<table><tr><td>Backbone Family</td><td>Encoder Backbone</td><td>Avg. MSE / MAE Reduction (%)</td></tr><tr><td>MLP</td><td>TimeMixer</td><td>+3.2% / +2.7%</td></tr><tr><td rowspan="2">Transformer</td><td>PatchTST</td><td>+3.5% / +2.4%</td></tr><tr><td>iTransformer</td><td>+2.6% / +2.9%</td></tr><tr><td>TCN</td><td>TS2Vec</td><td>+58.0% / +44.0%</td></tr></table>

Table 3: Cross-domain zero-shot forecasting transfer on ETT datasets (lower is better). Best and second-best results are highlighted in red and blue, respectively. Each source target row denotes pre-training on the source dataset followed by direct evaluation on the target dataset, without targetdomain parameter updates.
<table><tr><td rowspan="3">Transfer</td><td colspan="2">WINOTS (Ours)</td><td colspan="2">WINOTS-LP (Ours)</td><td colspan="2">TimeMixer Wang et al. [2024]</td><td colspan="2">TimeBase Huang et al. [2025]</td><td colspan="2">SparseTSF PatchTST Lin et al. Nie et al.</td><td colspan="2">Autoformer Wu et al. [2021]</td><td colspan="2">FEDformer Zhou et al. [2022]</td><td colspan="2">iTransformer Liu et al. [2023]</td><td colspan="2">TimesNet Wu et al. [2023]</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>[2024] MSE</td><td>MAE</td><td>[2023] MSE MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1→ETTh2</td><td>0.3496</td><td>0.3888</td><td>0.3520</td><td>0.3902</td><td>0.3778</td><td>0.4000</td><td>0.3560</td><td>0.3991 0.3729</td><td>0.4075</td><td>0.3790</td><td>0.4020 0.4721</td><td>0.4810</td><td>0.4573</td><td>0.4696</td><td>0.3752</td><td>0.3991</td><td>0.4229</td><td>0.4307</td></tr><tr><td>ETTh1→ETTm1</td><td>0.7027</td><td>0.5461</td><td>0.7032</td><td>0.5457</td><td>0.7866</td><td>0.5738</td><td>0.7420 0.5647</td><td>0.8111</td><td>0.5644</td><td>0.8000 0.5890</td><td>0.7730</td><td>0.5870</td><td>0.7627</td><td>0.5802</td><td>0.8307</td><td>0.5851</td><td>0.9413</td><td>0.6222</td></tr><tr><td>ETTh1→ETTm2</td><td>0.2951</td><td>0.3481</td><td>0.2955</td><td>0.3482</td><td>0.3143</td><td>0.3559</td><td>0.3060 0.3611</td><td>0.3096</td><td>0.3616</td><td>0.3140 0.3570</td><td>0.3650</td><td>0.4070</td><td>0.3532</td><td>0.3902</td><td>0.3216</td><td>0.3632</td><td>0.3524</td><td>0.3837</td></tr><tr><td>ETTh2→ETTh1</td><td>0.4473</td><td>0.4512</td><td>0.4513</td><td>0.4536</td><td>0.6379</td><td>0.5457</td><td>0.4150</td><td>0.4156 0.5039</td><td>0.4739</td><td>0.6410 0.5490</td><td>0.7140</td><td>0.5820</td><td>0.6856</td><td>0.5760</td><td>0.6673</td><td>0.5676</td><td>0.8393</td><td>0.6436</td></tr><tr><td>ETTh2→ETTm1</td><td>0.7586</td><td>0.5666</td><td>0.7675</td><td>0.5676</td><td>0.8627</td><td>0.5956</td><td>0.7620</td><td>0.5709 1.4956</td><td>0.6973</td><td>0.9680 0.6170</td><td>0.7300</td><td>0.5730</td><td>0.7273</td><td>0.5720</td><td>0.9493</td><td>0.6235</td><td>1.3997</td><td>0.7322</td></tr><tr><td>ETTh2→ETTm2</td><td>0.2955</td><td>0.3495</td><td>0.2956</td><td>0.3504</td><td>0.3311</td><td>0.3708</td><td>0.3101 0.3649</td><td>0.3267</td><td>0.3759</td><td>0.3280 0.3680</td><td>0.3490</td><td>0.3840</td><td>0.3293</td><td>0.3718</td><td>0.3291</td><td>0.3689</td><td>0.3803</td><td>0.4030</td></tr><tr><td>ETTm1→ETTh1</td><td>0.5392</td><td>0.4945</td><td>0.5281</td><td>0.4919</td><td>0.7253</td><td>0.5786</td><td>0.6320 0.5417</td><td>0.5546</td><td>0.5066</td><td>0.6210 0.5390</td><td>0.9560</td><td>0.6590</td><td>1.057</td><td>0.704</td><td>0.7165</td><td>0.5676</td><td>1.0033</td><td>0.6814</td></tr><tr><td>ETTm1→ETTh2</td><td>0.3908</td><td>0.4208</td><td>0.3976</td><td>0.4236</td><td>0.4412</td><td>0.4407</td><td>0.3820</td><td>0.4191 0.4049</td><td>0.4257</td><td>0.4380 0.4380</td><td>0.4660</td><td>0.4730</td><td>0.4517</td><td>0.4605</td><td>0.4572</td><td>0.4499</td><td>0.5039</td><td>0.4820</td></tr><tr><td>ETTm1→ETTm2</td><td>0.2697</td><td>0.3216</td><td>0.2724</td><td>0.3228</td><td>0.2994</td><td>0.3353</td><td>0.2760</td><td>0.3302 0.2746</td><td>0.3264</td><td>0.2970 0.3340</td><td>0.3655</td><td>0.4072</td><td>0.323</td><td>0.366</td><td>0.2991</td><td>0.3341</td><td>0.3332</td><td>0.3648</td></tr><tr><td>ETTm2→ETTh1</td><td>0.5011</td><td>0.4842</td><td>0.4844</td><td>0.4733</td><td>0.7596</td><td>0.6059</td><td>0.7560 0.5918</td><td>0.6203</td><td>0.5419</td><td>0.6360 0.5590</td><td>0.7150</td><td>0.5770</td><td>1.265</td><td>0.743</td><td>0.8905</td><td>0.6348</td><td>1.0088</td><td>0.6678</td></tr><tr><td>ETTm2→ETTh2</td><td>0.3632</td><td>0.3972</td><td>0.3660</td><td>0.3988</td><td>0.4185</td><td>0.4322</td><td>0.3870 0.4197</td><td>0.3926</td><td>0.4138</td><td>0.4060 0.4210</td><td>0.4231</td><td>0.4363</td><td>0.429</td><td>0.445</td><td>0.4337</td><td>0.4441</td><td>0.4712</td><td>0.4595</td></tr><tr><td>ETTm2→ETTm1</td><td>0.4791</td><td>0.4552</td><td>0.4318</td><td>0.4301</td><td>0.5936</td><td>0.5067</td><td>0.5500</td><td>0.4880 1.0778</td><td>0.6149</td><td>0.6060 0.5110</td><td>0.7200</td><td>0.5653</td><td>0.720</td><td>0.565</td><td>0.6440</td><td>0.5193</td><td>0.8046</td><td>0.5813</td></tr></table>

## 5.2 Cross-Domain Zero-Shot Forecasting Transfer

To test whether WINOTS captures transferable temporal representations we further evaluate crossdomain forecasting transfer on the ETT benchmarks. In this setting, models are pre-trained on a source dataset and evaluated directly on an unseen target dataset without any target-domain fine-tuning.

As shown in Table 3, WINOTS exhibits particularly strong cross-domain transfer within the ETT benchmark family, achieving the best MSE on nine of the twelve source target pairs and the best MAE on ten, and the top average rank among all evaluated methods (1.42 by MSE and 1.17 by MAE). Relative to its direct TimeMixer base, WINOTS yields an average relative improvement of 16.4% in MSE and 8.7% in MAE, outperforming it across all transfer pairs.

## 5.3 Unsupervised Anomaly Detection

We further evaluate WINOTS representations on five standard multivariate anomaly-detection datasets under the TSLib evaluation protocol (Table 4). Among the evaluated methods, WINOTS achieves the highest point-adjusted F1 on all five benchmarks. These results indicate that the multi-scale representations learned by WINOTS transfer effectively to reconstruction-based anomaly detection. In particular, representations learned by matching detail-perturbed student views to denoised teacher targets appear well suited to modeling regular temporal structure and identifying deviations through reconstruction error.

## 6 Ablation Studies

In this section, we conduct systematic ablation experiments to isolate the impact of core framework design choices. Unless specified otherwise, all ablation models are pre-trained on the source dataset and evaluated downstream under the frozen linear probing (WINOTS-LP) protocol across four standard forecasting horizons (H 96, 192, 336, 720 ).

Table 4: Anomaly detection results. We report point-adjusted precision (P), recall (R), and F1-score (%). Best and second-best results are shown in red and blue, respectively.
<table><tr><td>Dataset</td><td colspan="4">WINOTS (Ours)</td><td colspan="4">iTransformer Liu et al. [2023]</td><td colspan="4">DLinear Zeng et al.</td><td colspan="4">Autoformer Wu et al.</td><td colspan="4">TimesNet FEDformer Wu et al. Zhou et al.</td><td colspan="4">Crossformer Zhang and Yan [2023]</td><td colspan="4">Reformer [Kitaev et al., 2020]</td></tr><tr><td></td><td colspan="2">P R</td><td colspan="2"></td><td colspan="2">R</td><td colspan="2">Fl</td><td colspan="2">[2023] R</td><td colspan="2"></td><td colspan="2">[2021]</td><td colspan="2">[2023]</td><td colspan="2"></td><td colspan="2">[2022]</td><td colspan="2"></td><td colspan="2">F1</td><td colspan="2">P</td><td colspan="2">F1</td></tr><tr><td></td><td></td><td></td><td>F1</td><td>P</td><td></td><td></td><td></td><td></td><td></td><td>F1</td><td>P</td><td>R</td><td>F1</td><td>P</td><td></td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td><td></td><td>P</td><td>R</td><td></td><td></td><td></td><td>R</td><td></td></tr><tr><td>SMD</td><td>84.04 77.34</td><td></td><td>80.55</td><td>68.22</td><td>63.79</td><td>65.93</td><td>70.13</td><td></td><td>69.34</td><td>69.73</td><td>67.91</td><td>41.63</td><td>51.62</td><td></td><td>79.28</td><td>54.20</td><td>64.39</td><td>60.86</td><td>52.23</td><td></td><td>56.22</td><td>62.99</td><td>62.32</td><td>62.65</td><td></td><td>64.03</td><td>61.56</td><td>62.77</td></tr><tr><td>MSL</td><td>88.47</td><td>69.39</td><td>77.78</td><td>53.62</td><td>13.79</td><td>21.94 14.75</td><td>69.45</td><td></td><td>25.38</td><td>37.18</td><td>82.14</td><td>41.25</td><td>54.92</td><td>62.38</td><td></td><td>19.03</td><td>29.17 22.38</td><td>81.72</td><td>39.34 16.94</td><td>53.11</td><td></td><td>77.20 70.42</td><td>27.22 16.24</td><td>40.25 26.39</td><td>78.22 77.41</td><td></td><td>35.51 22.56</td><td>48.85 34.94</td></tr><tr><td>SMAP SWaT</td><td>92.36 34.72</td><td>64.56 8.59</td><td>76.00 13.78</td><td>57.84 6.67</td><td>8.45 1.34</td><td>2.23</td><td></td><td>69.09 5.96</td><td>14.14 1.19</td><td>23.47 1.98</td><td>73.57 9.83</td><td>20.13 1.95</td><td>31.61 3.26</td><td>68.28 4.17</td><td></td><td>13.39 0.84</td><td>1.39</td><td>70.91 10.07</td><td>2.00</td><td>27.35 3.34</td><td></td><td>28.92</td><td>7.37</td><td>11.75</td><td>9.76</td><td>1.93</td><td></td><td>3.23</td></tr><tr><td>PSM</td><td>99.33</td><td>86.51</td><td>92.48</td><td>93.38</td><td>30.87</td><td></td><td>46.40</td><td>96.79</td><td>42.52</td><td>59.08</td><td>98.65</td><td>14.01</td><td>24.53</td><td></td><td>81.76</td><td>31.17</td><td>45.14</td><td>98.05</td><td>14.66</td><td>25.51</td><td></td><td>93.49</td><td>34.85</td><td>50.77</td><td>91.50</td><td></td><td>24.71</td><td>38.92</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Augmentation Strategy. To examine the role of our wavelet-based view generation, we substitute our proposed wavelet augmentations with augmentations commonly used in vision self-distillation, including noise jitter, cropping, and their combinations (Table 5, visualized in Supplementary Figure A1). Overall, preserving full temporal length and phase integrity yields consistently better linear probing performance.

Specifically, cropping combined with jitter (jitter+crop and gaussian+crop) leads to noticeable performance degradation across most benchmarks, particularly on ETTm1 (MSE 0.438 vs. 0.364) and Weather (MSE 0.303 vs. 0.238). While simple time-domain noise (jitter) preserves sequence length and achieves competitive results on specific streams, our wavelet-domain view generation provides superior performance on four out of six datasets, with notable margins on ETTm1 and Weather. This suggests that modulating coefficients in the wavelet domain provides an effective mechanism for augmenting temporal data while keeping overall sequence continuity and long-term structure intact.

Table 5: Wavelet versus vision-like augmentations. We compare wavelet-based augmentations to jitter and cropping. Best results in red, second-best in blue.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">wavelets</td><td colspan="2">jitter</td><td colspan="2">jitter+crop</td><td colspan="2">gaussian+crop</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.416</td><td>0.423</td><td>0.417</td><td>0.426</td><td>0.422</td><td>0.426</td><td>0.424</td><td>0.428</td></tr><tr><td>ETTh2</td><td>0.347</td><td>0.385</td><td>0.349</td><td>0.387</td><td>0.349</td><td>0.388</td><td>0.360</td><td>0.392</td></tr><tr><td>ETTm1</td><td>0.364</td><td>0.386</td><td>0.397</td><td>0.414</td><td>0.438</td><td>0.443</td><td>0.398</td><td>0.418</td></tr><tr><td>ETTm2</td><td>0.252</td><td>0.308</td><td>0.247</td><td>0.308</td><td>0.253</td><td>0.312</td><td>0.274</td><td>0.328</td></tr><tr><td>Weather</td><td>0.238</td><td>0.273</td><td>0.242</td><td>0.278</td><td>0.303</td><td>0.320</td><td>0.263</td><td>0.294</td></tr><tr><td>Electricity</td><td>0.167</td><td>0.260</td><td>0.165</td><td>0.257</td><td>0.165</td><td>0.257</td><td>0.165</td><td>0.256</td></tr></table>

Pre-training Objective. We analyze the impact of the pre-training objective by contrasting the invariance-based joint-embedding distillation of WINOTS against a hybrid formulation as well as generative and reconstruction-based alternatives (Table 6). Specifically, we evaluate: (i) our invariance-based approach (ii) a multi-task hybrid objective (Invariance-based + MAE), (iii) MAE, (iv) NTP, and (v) a joint-embedding predictive architecture (JEPA) Assran et al. [2023].

The results demonstrate that incorporating a localized reconstruction target in (invariancebased+MAE) provides no systematic improvement over pure self-distillation and occasionally degrades downstream performance. Furthermore, purely generative and reconstruction-based objectives (MAE and NTP) trail behind the invariance-based objective of WINOTS on the majority of benchmarks. While NTP exhibits localized advantages on specific datasets (such as ETTm1 and Weather) where its autoregressive forecasting bias aligns closely with the downstream task, it fails to match the overall consistency of WINOTS.

## 7 Conclusion

WINOTS demonstrates that wavelet-domain joint-embedding distillation offers a flexible, modelagnostic paradigm for self-supervised time-series learning, shifting the focus from point-wise reconstruction to multi-scale structural invariants.

Table 6: Ablation of WINOTS’s objective (Invariance-based self distillation). In-domain forecasting MSE/MAE averaged over horizons 96, 192, 336, 720 . Lower is better; best in red, second-best in blue.
<table><tr><td rowspan="3">Dataset</td><td colspan="2">Invariance- based (Ours)</td><td colspan="2">Invariance- based+MAE</td><td colspan="2">MAE</td><td colspan="2">NTP</td><td colspan="2">JEPA</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>0.416</td><td>0.423</td><td>0.418</td><td>0.424</td><td>0.465</td><td>0.465</td><td>0.428</td><td>0.435</td><td>0.442</td><td>0.447</td></tr><tr><td>ETTh1 ETTh2</td><td>0.347</td><td>0.385</td><td>0.355</td><td>0.389</td><td>0.449</td><td>0.457</td><td>0.407</td><td>0.427</td><td>0.390</td><td>0.426</td></tr><tr><td>ETTm1</td><td>0.364</td><td>0.386</td><td>0.370</td><td>0.388</td><td>0.349</td><td>0.387</td><td>0.346</td><td>0.381</td><td>0.375</td><td>0.393</td></tr><tr><td>ETTm2</td><td>0.252</td><td>0.308</td><td>0.258</td><td>0.313</td><td>0.294</td><td>0.344</td><td>0.263</td><td>0.320</td><td>0.274</td><td>0.332</td></tr><tr><td>Weather</td><td>0.238</td><td>0.273</td><td>0.241</td><td>0.274</td><td>0.266</td><td>0.294</td><td>0.226</td><td>0.265</td><td>0.229</td><td>0.266</td></tr></table>

Limitations & Future Work. While our global joint-embedding objective excels at capturing stable multi-scale patterns, it can be outperformed on highly volatile streams by specialized autoregressive priors (e.g., NTP) that explicitly optimize step-by-step local transitions. Future work will explore adaptive, learnable wavelet bases and extend joint-embedding distillation to scale-free, multi-modal temporal systems.

## References

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023.

Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9650–9660, 2021.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PmLR, 2020.

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. In Forty-first International Conference on Machine Learning, 2024.

Ingrid Daubechies. Orthonormal bases of compactly supported wavelets. Communications on pure and applied mathematics, 41(7):909–996, 1988.

Ingrid Daubechies. Ten Lectures on Wavelets, volume 61 of CBMS-NSF Regional Conference Series in Applied Mathematics. Society for Industrial and Applied Mathematics, Philadelphia, PA, 1992. ISBN 978-0-89871-274-2.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.

Jiaxiang Dong, Haixu Wu, Yuxuan Wang, Yun-Zhong Qiu, Li Zhang, Jianmin Wang, and Mingsheng Long. Timesiam: A pre-training framework for siamese time-series modeling. In International Conference on Machine Learning (ICML), 2024.

David L. Donoho and Iain M. Johnstone. Ideal spatial adaptation by wavelet shrinkage. Biometrika, 81(3):425–455, 1994. doi: 10.1093/biomet/81.3.425.

Qihe Huang, Zhengyang Zhou, Kuo Yang, Zhongchao Yi, Xu Wang, and Yang Wang. Timebase: The power of minimalism in efficient long-term time series forecasting. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 26227–26246. PMLR, 2025.

Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In International Conference on Learning Representations (ICLR), 2020.

Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long- and short-term temporal patterns with deep neural networks. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, pages 95–104, 2018.

Shengsheng Lin, Weiwei Lin, Wentai Wu, Haojun Chen, and Junjie Yang. Sparsetsf: Modeling long-term time series forecasting with 1k parameters. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 30211–30226. PMLR, 2024.

Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. itransformer: Inverted transformers are effective for time series forecasting. In The Twelfth International Conference on Learning Representations, 2023.

Noam Major et al. Quantifying the pre-training dividend: Generative versus latent self-supervised learning for time series foundation models. arXiv preprint arXiv:2605.19462, 2026.

Konstantin Mishchenko and Aaron Defazio. Prodigy: An expeditiously adaptive parameter-free learner. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 35779–35804. PMLR, 2024. URL https://proceedings.mlr.press/v235/mishchenko24a.html.

Yessin Moakher et al. Utica: Multi-objective self-distllation foundation model pretraining for time series classification. arXiv preprint arXiv:2603.01348, 2026.

Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations (ICLR), 2023.

Aaron van den Oord et al. Representation learning with contrastive predictive coding. In arXiv preprint arXiv:1807.03748, 2018.

Felix Pieper, Konstantin Ditschuneit, Martin Genzel, Alexandra Lindt, and Johannes S. Otterbach. Self-distilled representation learning for time series, 2023.

Rafael Poyatos, Víctor Granda, Víctor Flo, Mark A. Adams, Balázs Adorján, David Aguadé, Marcos P. M. Aidar, Scott Allen, M. Susana Alvarado-Barrientos, Kristina J. Anderson-Teixeira, et al. Global transpiration data from sap flow measurements: the SAPFLUXNET database. Earth System Science Data Discussions, 2020:1–57, 2020.

Daoyu Wang, Mingyue Cheng, Zhiding Liu, and Qi Liu. Timedart: A diffusion autoregressive transformer for self-supervised time series representation. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 62627–62651. PMLR, 2025.

Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Y Zhang, and Jun Zhou. Timemixer: Decomposable multiscale mixing for time series forecasting. In International Conference on Learning Representations (ICLR), 2024.

Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In Advances in Neural Information Processing Systems, volume 34, pages 22419–22430, 2021.

Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet: Temporal 2d-variation modeling for general time series analysis. In International Conference on Learning Representations, 2023.

Zhihan Yue, Haoyue Liu, Yan Zhou, Huan Yu, and Wenwu Sun. Ts2vec: Towards universal representation of time series. In AAAI Conference on Artificial Intelligence, 2022.

Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? In Proceedings ofthe AAAI conference on artificial intelligence, volume 37, pages 11121–11128, 2023.

Shuyi Zhang, Bin Guo, Anlan Dong, Jing He, Ziping Xu, and Song Xi Chen. Cautionary tales on air-quality improvement in Beijing. Proceedings ofthe Royal Society A: Mathematical, Physical and Engineering Sciences, 473(2205):20170457, 2017.

Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In The eleventh international conference on learning representations, 2023.

Shubao Zhao, Ming Jin, Zhaoxiang Hou, Chengyi Yang, Zengxiang Li, Qingsong Wen, and Yi Wang. Himtm: Hierarchical multi-scale masked time series modeling with self-distillation for longterm forecasting. In Proceedings ofthe 33rd ACM International Conference on Information and Knowledge Management, pages 3352–3362, 2024.

Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International conference on machine learning, pages 27268–27286. PMLR, 2022.

## Appendix

This appendix provides additional methodological details, implementation specifications, and extended experimental results supporting the main paper. We first expand the WINO-TS methodology, including the teacher–student optimization, wavelet view construction, signal-processing interpretation, and architectural implementation details omitted from the main text. We then present extended experimental protocols, optimization and reproducibility settings, detailed per-horizon forecasting results, additional evaluations on synthetic pre-training and classification, and comprehensive ablation studies covering the pre-training objective, augmentation strategy, wavelet design choices, backbone generalization, view pairing, and robustness to random initialization. Together, these materials provide sufficient detail to reproduce and further analyze the proposed framework.

## A Additional Method Details

The main paper defines the WINO-TS wavelet views and self-distillation objective. This section records additional teacher-update, signal-processing, architectural, and implementation details that are useful for reproducing and interpreting the method.

## A.1 Teacher–Student Optimization and Cross-View Objective

At the beginning of pre-training, the teacher parameters are initialized from the student parameters. Gradients are applied only to the student. After each student update, the teacher backbone and projection head are updated by exponential moving average (EMA):

$$
\theta _ { t } \gets \lambda \theta _ { t } + ( 1 - \lambda ) \theta _ { s } , \qquad \phi _ { t } \gets \lambda \phi _ { t } + ( 1 - \lambda ) \phi _ { s } .\tag{15}
$$

The momentum coefficient follows a cosine schedule from $\lambda _ { 0 } = 0 . 9 9 9 6$ to 1.0 over pre-training.

$$
Q
$$

$$
\overline { { { z } } } _ { t } = \frac { 1 } { B Q } \sum _ { b = 1 } ^ { B } \sum _ { q = 1 } ^ { Q } z _ { t , b } ^ { q } .\tag{16}
$$

The center is updated as

$$
c  m _ { c } c + ( 1 - m _ { c } ) \overline { { { z } } } _ { t } , \qquad m _ { c } = 0 . 9 .\tag{17}
$$

The student temperature is fixed at $\tau _ { s } = 0 . 1$ . The teacher temperature is scheduled from 0.06 to its final value $\tau _ { t } = 0 . 0 4$ over the first five pre-training epochs.

During pre-training, WINO-TS attaches a three-layer DINO projection MLP to the backbone. The hidden layers have width 2048 and use GELU activations, and the MLP terminates in a 256-dimensional bottleneck. The bottleneck is $\ell _ { 2 } \cdot$ -normalized and passed through a weight-normalized linear output layer of dimension $K = 1 0 2 4$ . The projection head is discarded after pre-training, and the EMA teacher backbone is transferred to the downstream task.

For an input window X, the wavelet view generator constructs $Q$ easy views

$$
\mathcal { E } ( X ) = \left\{ X _ { q } ^ { \mathrm { e a s y } } \right\} _ { q = 1 } ^ { Q }\tag{18}
$$

and V hard views

$$
\begin{array} { r } { \mathcal { H } ( X ) = \left\{ X _ { v } ^ { \mathrm { h a r d } } \right\} _ { v = 1 } ^ { V } . } \end{array}\tag{19}
$$

The teacher processes only the easy views, whereas the student processes both the easy and hard views. The same easy-view realizations are therefore forwarded through both networks, while the hard views are forwarded only through the student.

Let $P _ { t , q } ^ { \mathrm { e a s y } }$ denote the teacher distribution for easy view $q ,$ and let $P _ { s , u } ^ { \mathrm { e a s y } }$ and $P _ { s , v } ^ { \mathrm { h a r d } }$ denote the corresponding student distributions. The valid easy-teacher/easy-student pairs are

$$
\mathcal { T } _ { \mathrm { e e } } = \{ ( q , u ) : 1 \leq q , u \leq Q , q \neq u \} ,\tag{20}
$$

and all easy-teacher/hard-student pairs are valid:

$$
\mathcal { T } _ { \mathrm { e h } } = \left\{ ( q , v ) : 1 \leq q \leq Q , 1 \leq v \leq V \right\} .\tag{21}
$$

The objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { W I N O } } = \frac { 1 } { \left| \mathcal { T } _ { \mathrm { e e } } \right| + \left| \mathcal { T } _ { \mathrm { e h } } \right| } \Bigg [ \displaystyle \sum _ { ( q , u ) \in \mathcal { T } _ { \mathrm { e e } } } H \big ( \mathrm { s g } \left[ P _ { t , q } ^ { \mathrm { e a s y } } \right] , P _ { s , u } ^ { \mathrm { e a s y } } \big ) } \\ { + \displaystyle \sum _ { ( q , v ) \in \mathcal { T } _ { \mathrm { e h } } } H \big ( \mathrm { s g } \left[ P _ { t , q } ^ { \mathrm { e a s y } } \right] , P _ { s , v } ^ { \mathrm { h a r d } } \big ) \Bigg ] . } \end{array}\tag{22}
$$

Consequently, both student view types contribute directly to optimization. Each easy student view is supervised by the teacher outputs from the other easy-view realizations, while every hard student view is supervised by all easy teacher outputs. Only the teacher–student pair corresponding to the identical easy-view index is excluded.

## A.2 Multiresolution Interpretation

For an orthonormal wavelet basis, a J-level discrete wavelet transform decomposes a single-channel signal into an approximation space $V _ { J }$ and its multi-scale orthogonal complement:

$$
x = \underbrace { \sum _ { k } a _ { J , k } \phi _ { J , k } } _ { P _ { V J } x } + \underbrace { \sum _ { j = 1 } ^ { J } \sum _ { k } } _ { \in V _ { J } ^ { \perp } } d _ { j , k } \psi _ { j , k } .\tag{23}
$$

The approximation coefficients encode coarse temporal structure, whereas $d _ { j , k }$ describes localized variation at a temporal scale on the order of $2 ^ { j }$ around location $2 ^ { j } k$ . Unlike Fourier atoms, which have global temporal support, wavelet atoms are localized in both time and scale. This makes the representation suitable for non-stationary signals whose local frequency content changes over time.

A wavelet with N vanishing moments annihilates polynomials of degree below N:

$$
\psi \perp \{ 1 , t , \ldots , t ^ { N - 1 } \} .\tag{24}
$$

Equivalently, the corresponding scaling filter contains an N-fold zero at $\omega = \pi \colon$

$$
H ( \omega ) = \left( \frac { 1 + e ^ { - i \omega } } { 2 } \right) ^ { N } Q ( e ^ { - i \omega } ) ,\tag{25}
$$

$$
H ( \omega ) = { \frac { 1 } { \sqrt { 2 } } } \sum _ { k } h _ { k } e ^ { - i k \omega } .\tag{26}
$$

with the orthonormal perfect-reconstruction condition

$$
| H ( \omega ) | ^ { 2 } + | H ( \omega + \pi ) | ^ { 2 } = 1 .\tag{27}
$$

The number of vanishing moments therefore controls how strongly low-order trends are suppressed by the detail filters and affects the sparsity of wavelet coefficients on smooth signals.

The critically sampled DWT is not shift-invariant: a small temporal shift can alter coefficient alignment across levels. WINO-TS treats this basis sensitivity as an additional source of view diversity rather than attempting to remove it. At the default depth $J \ = \ 3$ , the approximation emphasizes temporal structure at scales longer than roughly $2 ^ { J + 1 }$ samples, while the detail bands retain progressively finer local fluctuations.

Why full-length temporal views. DINO-style learning depends strongly on the augmentation family. Image transformations such as multi-crop, color jitter, and blur generally alter nuisance factors while retaining object identity, but they do not have direct semantic analogues for ordered temporal signals. Random cropping can remove the context needed to identify trend, seasonality, or regime; time warping can distort phase relationships; and unrestricted amplitude jitter or additive noise can erase discriminative local structure. WINO-TS therefore realizes view diversity through localized wavelet coefficients while reconstructing every view at the original length.

— teacher (dwt\_soft)  student (dwt\_hard ×1.6) — teacher (full + noise)  student (0.3 crop + contrast/jitter)

![](images/731e923ad16989c27fa0158593b75464b661352671bf41bbde1f8eda6aa09d98.jpg)

![](images/dbf50109f346921ac3d7051a15bb62d2514a2f622e5e3a6427a1759642febb64.jpg)

![](images/92d304b279813b22502ad52bc784e8e6dbb150aa6fe7e96ebfcfe0b70b4d2e1a.jpg)

Figure A1: Qualitative comparison of wavelet-domain and vision-style view generation on a standardized periodic input window. The left panel shows the original input signal. In the middle panel, the WINO-TS easy teacher view, produced through wavelet-detail soft thresholding, and the hard student view, produced through wavelet-detail perturbation, retain the dominant period, temporal ordering, and full sequence support. Their differences are concentrated primarily in localized amplitude and high-frequency fluctuations. In the right panel, a representative vision-style augmentation pipeline based on temporal cropping, contrast modification, jitter, and additive noise alters the support and point-wise alignment of the two views, making their periodic correspondence less direct. The example illustrates the intended inductive bias of WINO-TS: preserving coarse temporal organization and global context while varying localized fine-scale content. It is provided as a qualitative sanity check rather than as general evidence of semantic preservation.

Qualitative visualization of temporal-structure preservation. Figure A1 illustrates the difference between the proposed wavelet-domain view generator and a representative vision-style augmentation pipeline. Under WINO-TS, the easy teacher and hard student views remain temporally aligned and preserve the dominant periodic structure of the original window, despite differing in their localized high-frequency content. This produces a non-trivial self-distillation task without removing temporal context or changing the sequence support. In contrast, cropping and unstructured time-domain perturbations can alter support, alignment, and phase correspondence between teacher and student views. The visualization therefore illustrates why full-length, frequency-localized perturbations provide a more natural invariance mechanism for periodic temporal signals.

## A.3 Easy View and Its Relation to Classical Wavelet Shrinkage

For a selected wavelet basis, the easy view preserves the approximation coefficients and applies soft-threshold shrinkage independently to each detail band:

$$
x ^ { \mathrm { e a s y } } = \mathcal { W } _ { w } ^ { - 1 } \left( a _ { J } , \eta _ { \tau _ { J } } ( d _ { J } ) , \dots , \eta _ { \tau _ { 1 } } ( d _ { 1 } ) \right) ,\tag{28}
$$

where

$$
\eta _ { \tau } ( d ) = \mathrm { s i g n } ( d ) \operatorname* { m a x } ( | d | - \tau , 0 ) , \qquad \tau _ { j } = \rho \operatorname* { m a x } _ { k } | d _ { j , k } | .\tag{29}
$$

Equivalently,

$$
x ^ { \mathrm { { e a s y } } } = P _ { V _ { J } } x + \sum _ { j , k } \eta _ { \tau _ { j } } ( d _ { j , k } ) \psi _ { j , k } .\tag{30}
$$

The easy view is therefore not a pure low-pass projection: it retains attenuated versions of dominant detail coefficients while suppressing weak transients.

This operation is inspired by classical wavelet shrinkage Donoho and Johnstone [1994], but it does not use the Donoho–Johnstone noise-calibrated threshold. WINO-TS instead uses a per-band threshold scaled by the largest absolute coefficient in that band. This avoids estimating a separate observation-noise model for every dataset and adapts the threshold to the dynamic range of each level.

Soft-thresholding is nonlinear and generally not idempotent:

$$
\eta _ { \tau } ( \eta _ { \tau } ( d ) ) \neq \eta _ { \tau } ( d ) .\tag{31}
$$

Accordingly, it does not project directly onto $V _ { J }$ . The default $\rho = 0 . 6$ applies strong but partial shrinkage. When $\rho \geq 1$ , every detail coefficient is mapped to zero and the construction reaches the hard low-pass limit $P _ { V _ { J } } x$

## A.4 Hard View and Standardized Perturbation Scale

The hard view preserves the approximation coefficients and injects Gaussian perturbations into all detail levels:

$$
x ^ { \mathrm { h a r d } } = \mathcal { W } _ { w } ^ { - 1 } \left( a _ { J } , d _ { J } + \varepsilon _ { J } , \dots , d _ { 1 } + \varepsilon _ { 1 } \right) ,\tag{32}
$$

(33)

$$
s _ { v , c } \sim \mathcal { U } ( 0 . 2 , 0 . 5 ) , \qquad \varepsilon _ { v , j , k , c } \sim \mathcal { N } ( 0 , s _ { v , c } ^ { 2 } ) .\tag{34}
$$

For each hard view v, a scale $s _ { v , c }$ is sampled independently for every channel c and shared across all detail levels of that channel. The construction can also be written as

$$
x ^ { \mathrm { { h a r d } } } = x + \mathcal { W } _ { w } ^ { - 1 } M _ { \mathrm { H P } } \varepsilon ,\tag{35}
$$

where $M _ { \mathrm { H P } }$ selects the high-pass detail coefficients.

Inputs are standardized channel-wise using statistics computed only from the training split before the wavelet transform is applied. The perturbation scale s is therefore expressed in standardized signal units. For multivariate inputs, the same wavelet basis is shared across channels within a view, but each channel receives an independent realization of ε. This preserves a coordinated view-level transform without imposing artificial high-frequency co-dependence between variables.

## A.5 Boundary Handling, Full-Length Reconstruction, and Admissible Depth

Finite windows require an extension rule at their boundaries. WINO-TS uses symmetric boundary extension. The ideal decomposition into $P _ { V _ { J } }$ x and $V _ { J } ^ { \perp }$ therefore holds exactly in the interior, with boundary-dependent coefficients confined to a region whose order scales as $\mathcal { O } ( 2 ^ { J } L )$ per side for a filter of length $L .$ This is an order statement rather than an exact count, because the affected region depends on the implementation and wavelet support.

When the same basis is used for two views, their unmodified approximation coefficients agree up to the boundary convention. Under the default independent basis sampling, the easy and hard views instead preserve comparable low-frequency bands; their approximation coefficients need not be identical.

The decomposition depth must be compatible with the input length and filter support. The admissible depth satisfies

$$
J \leq \left\lfloor \log _ { 2 } \left( { \frac { T } { L - 1 } } \right) \right\rfloor .\tag{36}
$$

For the longest filter in the default pool, sym8 with $L = 1 6$ , the default depth $J = 3$ requires

$$
T \geq 2 ^ { 3 } ( L - 1 ) = 1 2 0 .\tag{37}
$$

All views are reconstructed to the original length $T ,$ so the augmentation does not require changes to a backbone’s positional encoding, patch layout, or input-shape parameters.

## A.6 Wavelet Pool and Basis Diversity

$$
w ^ { \mathrm { e a s y } } , w ^ { \mathrm { h a r d \stackrel { i . i . d . } { \sim } } \mathcal { U } ( \mathcal { P } ) , }\tag{38}
$$

$$
\mathcal { P } = \{ \mathrm { s y m 4 } , \mathrm { s y m 6 } , \mathrm { s y m 8 } , \mathrm { d b 4 } , \mathrm { d b 6 } , \mathrm { c o i f 2 } \} .\tag{39}
$$

The pool varies support length, smoothness, vanishing moments, and phase behavior. Daubechies wavelets are compactly supported and minimum phase; Symlets reduce phase asymmetry; and Coiflets impose moment constraints on both the scaling and wavelet functions. Independent sampling prevents the encoder from binding its representation to one basis and introduces mild cross-basis phase and boundary variation while retaining comparable coarse temporal content.

Table A1: The WINO-TS wavelet pool . VM denotes the number of vanishing moments of the wavelet function ψ, and L denotes filter length in samples.
<table><tr><td>Wavelet</td><td>Family</td><td>VM</td><td>L</td><td>Primary property</td></tr><tr><td>sym4</td><td>Symlet</td><td>4</td><td>8</td><td>Reduced phase asymmetry</td></tr><tr><td>sym6</td><td>Symlet</td><td>6</td><td>12</td><td>Reduced phase asymmetry</td></tr><tr><td>sym8</td><td>Symlet</td><td>8</td><td>16</td><td>Reduced phase asymmetry</td></tr><tr><td>db4</td><td>Daubechies</td><td>4</td><td>8</td><td>Minimum-phase design</td></tr><tr><td>db6</td><td>Daubechies</td><td>6</td><td>12</td><td>Minimum-phase design</td></tr><tr><td>coif2</td><td>Coiflet</td><td>4</td><td>12</td><td>Moment constraints on φ and ψ</td></tr></table>

## A.7 Model-Agnostic Transfer

At the objective level, WINO-TS requires only a backbone that maps a full-length time-series window to a representation accepted by the projection head. The backbone may be an MLP, Transformer, or temporal convolutional network. After pre-training, the projection head is removed and the EMA teacher backbone is used for downstream adaptation.

Two forecasting adaptation modes are considered. In linear probing, the backbone is frozen and only a task-specific forecasting head is optimized. In full fine-tuning, the backbone and task head are optimized jointly. Classification uses full fine-tuning: the pretrained backbone is unfrozen and optimized jointly with the classification head. In zero-shot cross-domain forecasting, pre-training and task adaptation are performed only on the source dataset, followed by direct target evaluation withou target-domain parameter updates.

## B Additional Positioning and Related Work

WINO-TS is a pre-training and view-construction method rather than a new supervised forecasting architecture. It is therefore complementary to backbones such as PatchTST, iTransformer, and TimeMixer: the DINO projection head is attached only during pre-training and is removed before downstream adaptation.

A related controlled study examined the pre-training dividend in time-series foundation models Major et al. [2026]. It compared generative objectives, including masked reconstruction, next-token prediction, and diffusion, with latent-alignment objectives, including JEPA, Le-JEPA, and DINO, under a shared evaluation protocol, and identified a task-dependent precision–invariance trade-off. WINO-TS addresses a different question: it develops the DINO direction into a dedicated time-series pre-training recipe based on wavelet-domain view construction, basis-family sampling, and fulllength multivariate adaptation, and evaluates the resulting representations across forecasting, transfer, anomaly detection, and classification.

Recent time-series self-distillation methods also use non-contrastive teacher–student learning, but differ in how they define the prediction task. TimeSiam uses past/current subseries and masked past-to-current reconstruction. Self-Distilled Representation Learning for Time Series follows a data2vec-style latent-prediction objective from masked inputs. HiMTM combines hierarchical masked time-series modeling with self-distillation, and UTICA adapts DINOv2-style training to time-series classification. WINO-TS instead constructs full-length views in the wavelet domain and varies localized detail coefficients through shrinkage, perturbation, and basis sampling rather than cropping, subseries selection, or reconstruction-only targets.

## C Extended Experimental Protocols

## C.1 Pre-Training and Adaptation

For in-domain forecasting, the backbone is first pre-trained on the unlabeled training split of the same dataset. The projection head is then discarded and the EMA teacher backbone is adapted using either full fine-tuning or linear probing.

For cross-domain zero-shot forecasting, pre-training and task adaptation are performed strictly on the source dataset. The resulting source model is evaluated directly on the target dataset without target-domain fine-tuning or other target-domain parameter updates.

Table A2: WINO-TS pre-training hyperparameters.  
Hyperparameter Value   
Optimizer AdamW   
Base learning rate 5 10<sup>−4</sup> (linear batch scaling, /256)   
LR schedule 3-epoch warmup  cosine to 1  10 6   
Weight decay cosine 0.04  0.1   
Gradient clipping max-norm 3.0   
Epochs 80   
Batch size (per GPU) 128   
Precision FP32   
EMA teacher momentum cosine 0.9996 1.0   
Teacher temperature warmup 0.06  0.04 (5 epochs)   
Student temperature 0.1   
Freeze last layer 1 epoch   
Easy (teacher) views 1   
Hard (student) views 1   
Input window / patch 336   
Prototypes (K) 1024   
Backbone TimeMixer, 4 layers, d<sub>model</sub>=128,   
d =256, dropout 0.1   
Attention pooling multi-head attention, d =128,   
4 heads, dropout 0.0

For classification, a classification head is attached to the WINO-TS-pretrained backbone, and the complete model is fine-tuned end-to-end with supervision. Both the pretrained backbone and the classification head are updated. Forecasting results use MSE and MAE, averaged over horizons 96, 192, 336, 720 unless a table states otherwise. Classification results use accuracy.

## C.2 Dataset Naming

The main paper reports results on the standard long-term forecasting benchmarks ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Exchange, Solar, and Traffic. The appendix adds further datasets for a broader robustness evaluation.

For readability, we use the following short names throughout: AirQuality-Shunyi and AirQuality-Wan (also written AQShunyi and AQWan), PM2.5, JapaneseVowels, and SelfRegulationSCP1/2. “Solar Energy” is abbreviated to Solar; CzeLan keeps the capitalization of the main tables. All other forecasting names match the main experimental tables.

## D Optimization and Reproducibility

For completeness, we report the optimization and compute configuration used for pre-training and downstream evaluation. All runs use seed 42 unless a table explicitly reports multiple seeds.

## D.1 Self-Supervised Pre-Training

The WINO-TS self-distillation objective is optimized with AdamW. The base learning rate is $5 \times 1 0 ^ { - 4 }$ and is scaled linearly with the effective batch size,

$$
\eta _ { \mathrm { e f f } } = \eta _ { \mathrm { b a s e } } \frac { B N _ { \mathrm { G P U } } } { 2 5 6 } .\tag{40}
$$

The learning rate is warmed up linearly over the first three epochs and then decayed to $1 \times 1 0 ^ { - 6 }$ with a cosine schedule. Weight decay follows a cosine schedule from 0.04 to 0.1, and gradients are clipped to a maximum norm of 3.0. The EMA teacher momentum is cosine-annealed from 0.9996 to 1.0. The teacher temperature is scheduled from 0.06 to 0.04 over the first five epochs, while the student temperature remains fixed at 0.1. The final prototype layer is frozen during the first epoch to stabilize early training.

Pre-training runs for 80 epochs in full precision (FP32) with a per-GPU batch size of 128. Each input contains 336 time steps, and the projection head outputs K = 1024 prototypes.

## D.2 Forecasting

Downstream forecasting uses Adam with a one-cycle learning-rate schedule comprising a 30% warmup phase followed by cosine decay, together with weight decay $1 \times 1 0 ^ { - 4 }$ . We report two adaptation protocols. In linear probing, the backbone is frozen and only the forecasting head is trained. In full fine-tuning, the backbone is unfrozen and the encoder and head use a shared learning rate. The learning rate is $2 \times 1 0 ^ { - 5 }$ for linear probing and $1 \times 1 0 ^ { - 4 }$ for full fine-tuning. A subset of fine-tuning runs instead uses the learning-rate-free Prodigy [Mishchenko and Defazio, 2024] optimizer with $d _ { \mathrm { c o e f } } \in \{ 0 . 5 , 1 . 0 \}$

Fine-tuning runs for approximately 20 epochs with early stopping on a held-out validation split; the checkpoint with the best validation performance is retained. The default forecasting batch size is 128 and is reduced for the highest-channel datasets to fit in memory: 32 for PM2.5, 16 for Electricity, and 8 for Traffic. Head dropout is selected from 0.0, 0.1 . The same optimization configuration is used for horizons 96, 192, 336, 720 .

## D.3 Anomaly Detection

The anomaly detector consists of a WINO-TS-pretrained encoder followed by a linear reconstruction decoder. The encoder and decoder are jointly fine-tuned, using normal windows from the training split. The detector is optimized with Adam for 10 epochs using a learning rate of $1 \times 1 0 ^ { - 3 }$ and a mean-squared reconstruction loss over non-overlapping windows of length 100, represented as 10 patches of length 10.

Given an input X and its reconstruction $\widehat { X }$ , the anomaly score at timestamp t is the channel-averaged squared reconstruction error:

$$
e _ { t } = \frac { 1 } { C } \left\| X _ { t } - \widehat { X } _ { t } \right\| _ { 2 } ^ { 2 } .\tag{41}
$$

Following the TSLib evaluation protocol, the decision threshold is computed from the combined training and unlabeled test reconstruction scores:

$$
\gamma _ { d } = Q _ { 1 - r _ { d } / 1 0 0 } \left( \mathcal { E } _ { \mathrm { t r a i n } } ^ { d } \cup \mathcal { E } _ { \mathrm { t e s t } } ^ { d } \right) ,\tag{42}
$$

where $\mathscr { E } _ { \mathrm { t r a i n } } ^ { d }$ and $\mathcal { E } _ { \mathrm { t e s t } } ^ { d }$ denote the timestamp-level reconstruction-energy distributions for dataset d. We use $r _ { \mathrm { S M D } } = 0 . \mathfrak { s }$ 5 and $r _ { d } = 1 . 0$ for MSL, SMAP, SWaT, and PSM, corresponding to the 99.5th and 99th percentiles, respectively. These dataset-specific anomaly ratios are predefined by the evaluation configuration and are not estimated from the empirical test anomaly rate. Test labels are not used to determine the threshold.

A test timestamp is initially classified as anomalous when $e _ { t } ^ { \mathrm { t e s t } } > \gamma _ { d } .$ . We then apply the standard segment-level point-adjustment procedure before computing precision, recall, and F1. All methods in the anomaly comparison use the same scoring, thresholding, and point-adjustment procedure.

## D.4 Classification

Classification uses full end-to-end fine-tuning. A linear classification head is attached to the pretrained backbone, and both the backbone and classification head are jointly optimized with Adam at learning rate $1 \times 1 0 ^ { - 3 }$ for 20 epochs using batch size 16 on fixed-length windows.

## D.5 Hardware and Software

All experiments run on a server with 8 NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs with 96 GB of memory per GPU. Each dataset–model run uses one GPU; no multi-GPU training is used. The software stack is Python 3.9.25 and PyTorch 2.8.0 with CUDA 12.8 and cuDNN 9.10.

Pre-training cost scales with the number of channels. On one GPU, an epoch takes approximately 24 s for ETTh1 (7 channels) and approximately 26 min for Electricity (321 channels); a complete ETT pre-training run takes approximately 32 min. Downstream fine-tuning takes approximately 10–15 min on smaller datasets and up to a few hours on the largest datasets.

Table A3: Forecasting (MSE/MAE, avg. over horizons 96, 192, 336, 720 ) with synthetic pretraining. Ours (FT) = best fine-tuned DINO config per dataset; Ours (LP) and survey Major et al. [2026] are linear-probe. Each MSE/MAE pair is from a single run. Lower is better.
<table><tr><td></td><td colspan="2">Ours (synth, FT)</td><td colspan="2">Ours (synth, LP)</td><td colspan="2">Survey (synth, LP)</td></tr><tr><td>Dataset</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.417</td><td>0.425</td><td>0.521</td><td>0.489</td><td>0.438</td><td>0.446</td></tr><tr><td>ETTh2</td><td>0.365</td><td>0.402</td><td>0.410</td><td>0.428</td><td>0.362</td><td>0.401</td></tr><tr><td>ETTm1</td><td>0.351</td><td>0.378</td><td>0.361</td><td>0.389</td><td>0.357</td><td>0.384</td></tr><tr><td>ETTm2</td><td>0.250</td><td>0.310</td><td>0.259</td><td>0.319</td><td>0.253</td><td>0.311</td></tr><tr><td>Weather</td><td>0.227</td><td>0.261</td><td>0.242</td><td>0.276</td><td>0.235</td><td>0.272</td></tr></table>

Table A4: Effect of the DINO head output dimension K (out\_dim) on in-domain forecasting (linear probe, context 336, MSE/MAE averaged over horizons 96, 192, 336, 720 ). Ours uses $K { = } 1 0 2 4$ best per row in bold.
<table><tr><td></td><td colspan="2">K=512</td><td colspan="2">K=1024 (Ours)</td><td colspan="2">K=2048</td><td colspan="2">K=8192</td></tr><tr><td>Dataset</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.430</td><td>0.431</td><td>0.416</td><td>0.423</td><td>0.431</td><td>0.433</td><td>0.435</td><td>0.434</td></tr><tr><td>ETTh2</td><td>0.369</td><td>0.395</td><td>0.347</td><td>0.385</td><td>0.377</td><td>0.399</td><td>0.378</td><td>0.400</td></tr><tr><td>ETTm1</td><td>0.364</td><td>0.387</td><td>0.364</td><td>0.386</td><td>0.359</td><td>0.383</td><td>0.357</td><td>0.384</td></tr><tr><td>ETTm2</td><td>0.248</td><td>0.311</td><td>0.252</td><td>0.308</td><td>0.248</td><td>0.310</td><td>0.249</td><td>0.309</td></tr><tr><td>Weather</td><td>0.238</td><td>0.273</td><td>0.238</td><td>0.273</td><td>0.269</td><td>0.298</td><td>0.241</td><td>0.275</td></tr></table>

## E Extended Forecasting Results

## E.1 Complete In-Domain Averages

The complete in-domain forecasting comparison over all 13 datasets is already reported in the main paper in Table 1. That table includes WINO-TS under full fine-tuning and linear probing, the selfsupervised TimeSiam and TS2Vec baselines, and all supervised forecasting baselines, with MSE and MAE averaged over horizons 96, 192, 336, 720 . We refer to the main-paper table rather than reproducing the same large table in the appendix.

## E.2 Horizon-wise Forecasting Results

Table A5 provides the complete per-horizon breakdown for all 13 in-domain forecasting benchmarks. It reports horizons 96, 192, 336, 720 under the same input context length of 336 and includes both WINO-TS adaptation protocols, the self-supervised baselines, and all supervised baselines used in the main comparison.

Using the displayed values and counting ties, at least one WINO-TS adaptation protocol ranks among the two lowest-error methods in 40 of the 52 dataset–horizon rows by MSE and 47 of 52 by MAE; it attains the best displayed value in 28 rows by MSE and 31 rows by MAE. The strongest results are concentrated on the ETT benchmarks, Weather, CzeLan, and many of the shorter-horizon air-quality settings. The full breakdown also exposes meaningful exceptions: TimeSiam is strongest on several Electricity horizons, SparseTSF leads Solar and much of Traffic, DLinear and iTransformer lead Exchange at horizon 720, and non-WINO baselines are strongest on some long-horizon AQShunyi and AQWan settings. Thus, the horizon-averaged gains are not driven by one prediction length, but the preferred adaptation protocol and strongest competing inductive bias remain dataset- and horizon-dependent.

## F Synthetic Pre-Training

We investigate whether WINO-TS can learn transferable temporal representations when the real, in-domain unlabeled pre-training split is replaced with a synthetic time-series corpus. This setting tests whether wavelet-based self-distillation can capture reusable multi-scale temporal primitives without relying exclusively on the statistics of the downstream dataset. Table A6 compares in-domain WINO-TS fine-tuning with two synthetic pre-training variants.

Table A5: In-domain forecasting performance per prediction length 96, 192, 336, 720 (lower is better). All methods use the same input context length of 336. Best and second-best per row are shown in red and blue. Ours: WINO-TS (full fine-tune) and WINO-TS-LP (frozen backbone, trained head). Self-supervised: TimeSiam and TS2Vec. Supervised: end-to-end baselines.
<table><tr><td rowspan=1 colspan=11>ETT1 16 0369 0309 0370 0396 0376 0405 061 0578</td><td rowspan=1 colspan=16></td></tr><tr><td rowspan=1 colspan=5>7200.447 0.456 0.446 0.459</td><td rowspan=1 colspan=2>0.474 0.488</td><td rowspan=1 colspan=4>1.031 0.783</td><td rowspan=1 colspan=7>0.509 0.502 0.446 0.449 0.467 0.471 0.471 0.476</td><td rowspan=1 colspan=2>0.573 0.561</td><td rowspan=1 colspan=4>0.487 0.484 0.592 0.566</td><td rowspan=1 colspan=3>0.498 0.488 0.652 0.560</td></tr><tr><td rowspan=1 colspan=5>96 0.271 0.331 0.270 0.330</td><td rowspan=1 colspan=2>0.303 0.358</td><td rowspan=1 colspan=4>0.934 0.758</td><td rowspan=1 colspan=7></td><td rowspan=1 colspan=2>0.309.0.373</td><td rowspan=1 colspan=4>0.302 0.350.0.408 0.452</td><td rowspan=1 colspan=3>0.323,0.366,0.383,0.425</td></tr><tr><td rowspan=1 colspan=5>ETTh2 192 0.355 0.382 0.342 0.376</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=5>720 0.396 0.429 0.403 0.433</td><td rowspan=1 colspan=2>0.412 0.443</td><td rowspan=1 colspan=4>2.708 1.387</td><td rowspan=1 colspan=5>0.406 0.438 0.417 0.454 0.418 0.452</td><td rowspan=1 colspan=2>0.431 0.453</td><td rowspan=1 colspan=2>0.692 0.591</td><td rowspan=1 colspan=4>0.424 0.444 0.481 0.503</td><td rowspan=1 colspan=3>0.441 0.454 0.905 0.673</td></tr><tr><td rowspan=1 colspan=5>96 0.291 0.343 0.293.0.346</td><td rowspan=1 colspan=2>0.290 0.347</td><td rowspan=1 colspan=4>0.609 0.542</td><td rowspan=1 colspan=5>0.299 0.350 0.315 0.356 0.308 0.357</td><td rowspan=1 colspan=2>0.329 0.367</td><td rowspan=1 colspan=2>0.308 0.350</td><td rowspan=1 colspan=7></td></tr><tr><td rowspan=1 colspan=1>ETTml</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>720 0.412 0.419 0.424 0.419</td><td rowspan=1 colspan=1>0.416</td><td rowspan=1 colspan=1>0.428</td><td rowspan=1 colspan=4>0.760 0.642</td><td rowspan=1 colspan=2>0.434 0.428 0.431</td><td rowspan=1 colspan=1>0.423</td><td rowspan=1 colspan=2>0.439 0.436</td><td rowspan=1 colspan=2>0.464 0.442</td><td rowspan=1 colspan=2>0.465 0.456</td><td rowspan=1 colspan=3>0.483 0.453 0.519</td><td rowspan=1 colspan=1>0.504</td><td rowspan=1 colspan=3>0.477 0.453 0.671 0.556</td></tr><tr><td rowspan=1 colspan=5>960.161.0.247.0.161.0.247</td><td rowspan=1 colspan=1>0.171</td><td rowspan=1 colspan=1>0.262</td><td rowspan=1 colspan=4>0.339 0.418</td><td rowspan=1 colspan=3>0.172, 0.264 0.1690.259</td><td rowspan=1 colspan=2>0.170 0.254</td><td rowspan=1 colspan=1>0.178</td><td rowspan=1 colspan=1>0.259</td><td rowspan=1 colspan=2>0.201 0.297</td><td rowspan=1 colspan=3>0.183, 0.266, 0.262</td><td rowspan=1 colspan=1>0.344</td><td rowspan=1 colspan=3>0.190 0.266 0.287 0.358</td></tr><tr><td rowspan=1 colspan=3>ETTm2 192 0.216 0.285</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>0.351</td><td rowspan=1 colspan=1>0.378</td><td rowspan=1 colspan=1>0.364</td><td rowspan=1 colspan=1>0.386</td><td rowspan=1 colspan=1>1.983</td><td rowspan=1 colspan=3>1.074</td><td rowspan=1 colspan=2>0.365 0.388 0.372</td><td rowspan=1 colspan=1>0.386</td><td rowspan=1 colspan=1>0.373</td><td rowspan=1 colspan=1>0.392</td><td rowspan=1 colspan=1>0.402</td><td rowspan=1 colspan=1>0.401</td><td rowspan=1 colspan=1>0.451</td><td rowspan=1 colspan=1>0.459</td><td rowspan=1 colspan=1>0.414</td><td rowspan=1 colspan=1>0.406</td><td rowspan=1 colspan=1>0.425</td><td rowspan=1 colspan=1>0.430</td><td rowspan=1 colspan=1>0.412</td><td rowspan=1 colspan=1>0.404</td><td rowspan=1 colspan=1>0.452 0.457</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0.1460.197</td><td rowspan=1 colspan=1>0.160</td><td rowspan=1 colspan=1>0.209</td><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=1>0.197</td><td rowspan=1 colspan=1>0.745</td><td rowspan=1 colspan=3>0.607</td><td rowspan=1 colspan=2>0.1490.2000.187</td><td rowspan=1 colspan=1>0.242</td><td rowspan=1 colspan=1>0.158</td><td rowspan=1 colspan=1>0.214</td><td rowspan=1 colspan=1>0.181</td><td rowspan=1 colspan=1>0.221</td><td rowspan=1 colspan=1>0.180</td><td rowspan=1 colspan=1>0.239</td><td rowspan=1 colspan=1>0.174</td><td rowspan=1 colspan=1>0.214</td><td rowspan=1 colspan=1>0.238</td><td rowspan=1 colspan=1>0.313</td><td rowspan=1 colspan=1>0.167</td><td rowspan=1 colspan=1>0.2170</td><td rowspan=1 colspan=1>.2750.346</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>7200.3180.332</td><td rowspan=1 colspan=1>0.324</td><td rowspan=1 colspan=1>0.337</td><td rowspan=1 colspan=1>0.326</td><td rowspan=1 colspan=1>0.337</td><td rowspan=1 colspan=1>1.492</td><td rowspan=1 colspan=3>0.916</td><td rowspan=1 colspan=2>0.3210.3380.337</td><td rowspan=1 colspan=1>0.347</td><td rowspan=1 colspan=1>0.326</td><td rowspan=1 colspan=1>0.344</td><td rowspan=1 colspan=1>0.363</td><td rowspan=1 colspan=1>0.350</td><td rowspan=1 colspan=1>0.327</td><td rowspan=1 colspan=1>0.362</td><td rowspan=1 colspan=1>0.360</td><td rowspan=1 colspan=1>0.350</td><td rowspan=1 colspan=1>0.391</td><td rowspan=1 colspan=1>0.418</td><td rowspan=1 colspan=1>0.361</td><td rowspan=1 colspan=1>0.355</td><td rowspan=1 colspan=1>0.416 0.425</td></tr><tr><td rowspan=1 colspan=3>960.1320.225</td><td rowspan=1 colspan=1>0.135</td><td rowspan=1 colspan=1>0.230</td><td rowspan=1 colspan=1>0.127</td><td rowspan=1 colspan=1>0.220</td><td rowspan=1 colspan=1>0.361</td><td rowspan=1 colspan=3>0.443</td><td rowspan=1 colspan=2>0.1310.2260.150</td><td rowspan=1 colspan=1>0.243</td><td rowspan=1 colspan=1>0.138</td><td rowspan=1 colspan=1>0.230</td><td rowspan=1 colspan=1>0.189</td><td rowspan=1 colspan=1>0.281</td><td rowspan=1 colspan=1>0.147</td><td rowspan=1 colspan=1>0.248</td><td rowspan=1 colspan=1>0.150</td><td rowspan=1 colspan=1>0.245</td><td rowspan=1 colspan=1>0.215</td><td rowspan=1 colspan=1>0.329</td><td rowspan=1 colspan=1>0.169</td><td rowspan=1 colspan=1>0.272</td><td rowspan=1 colspan=1>0.2240.337</td></tr><tr><td rowspan=1 colspan=1>Electricity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>7200.2060.293</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.293</td><td rowspan=1 colspan=1>0.197</td><td rowspan=1 colspan=1>0.286</td><td rowspan=1 colspan=1>0.398</td><td rowspan=1 colspan=3>0.467</td><td rowspan=1 colspan=1>0.2020.2940</td><td rowspan=1 colspan=1>.216</td><td rowspan=1 colspan=1>0.300</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.293</td><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>0.331</td><td rowspan=1 colspan=1>0.211</td><td rowspan=1 colspan=1>0.310</td><td rowspan=1 colspan=1>0.238</td><td rowspan=1 colspan=1>0.321</td><td rowspan=1 colspan=1>0.292</td><td rowspan=1 colspan=1>0.385</td><td rowspan=1 colspan=1>0.248</td><td rowspan=1 colspan=1>0.331</td><td rowspan=1 colspan=1>0.3740.439</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>960.0840.202</td><td rowspan=1 colspan=1>0.087</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.093</td><td rowspan=1 colspan=1>0.218</td><td rowspan=1 colspan=1>0.186</td><td rowspan=1 colspan=3>0.315</td><td rowspan=1 colspan=1>0.1060.231</td><td rowspan=1 colspan=1>0.113</td><td rowspan=1 colspan=1>0.239</td><td rowspan=1 colspan=1>0.110</td><td rowspan=1 colspan=1>0.242</td><td rowspan=1 colspan=1>0.085</td><td rowspan=1 colspan=1>0.203</td><td rowspan=1 colspan=1>0.127</td><td rowspan=1 colspan=1>0.259</td><td rowspan=1 colspan=1>0.087</td><td rowspan=1 colspan=1>0.208</td><td rowspan=1 colspan=1>0.363</td><td rowspan=1 colspan=1>0.452</td><td rowspan=1 colspan=1>0.115</td><td rowspan=1 colspan=1>0.244</td><td rowspan=1 colspan=1>0.3600.451</td></tr><tr><td rowspan=1 colspan=1>Exchange</td><td rowspan=1 colspan=2>1920.1780.3013360.3240.411</td><td rowspan=1 colspan=1>0.1920.360</td><td rowspan=1 colspan=1>0.3090.433</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>7200.9210.715</td><td rowspan=1 colspan=1>0.951</td><td rowspan=1 colspan=1>0.728</td><td rowspan=1 colspan=1>0.980</td><td rowspan=1 colspan=1>0.736</td><td rowspan=1 colspan=1>1.251</td><td rowspan=1 colspan=3>0.878</td><td rowspan=1 colspan=1>78</td><td rowspan=1 colspan=2>1.1450.819</td><td rowspan=1 colspan=1>1.080</td><td rowspan=1 colspan=1>0.806</td><td rowspan=1 colspan=1>1.040</td><td rowspan=1 colspan=1>0.783</td><td rowspan=1 colspan=1>0.865</td><td rowspan=1 colspan=1>0.700</td><td rowspan=1 colspan=1>0.586</td><td rowspan=1 colspan=1>0.610</td><td rowspan=1 colspan=1>0.845</td><td rowspan=1 colspan=1>0.695</td><td rowspan=1 colspan=1>1.520</td><td rowspan=1 colspan=1>0.958</td><td rowspan=1 colspan=1>0.972</td></tr><tr><td rowspan=1 colspan=1>Solar</td><td rowspan=1 colspan=2>960.1920.2531920.2030.256</td><td rowspan=1 colspan=1>0.232</td><td rowspan=1 colspan=1>0.309</td><td rowspan=1 colspan=1>0.196</td><td rowspan=1 colspan=1>0.252</td><td rowspan=1 colspan=1>0.244</td><td rowspan=1 colspan=2>0</td><td rowspan=1 colspan=1>0.3570.354</td><td rowspan=1 colspan=1>0.1890.26000.2280.2810</td><td rowspan=1 colspan=1>.260.284</td><td rowspan=1 colspan=1>0.2770.292</td><td rowspan=1 colspan=1>0.184</td><td rowspan=1 colspan=1>0.235</td><td rowspan=1 colspan=1>0.233</td><td rowspan=1 colspan=1>0.282</td><td rowspan=1 colspan=1>0.225</td><td rowspan=1 colspan=1>0.298</td><td rowspan=1 colspan=1>0.237</td><td rowspan=1 colspan=1>0.280</td><td rowspan=1 colspan=1>0.285</td><td rowspan=1 colspan=1>0.377</td><td rowspan=1 colspan=1>0.231</td><td rowspan=1 colspan=1>0.279</td><td rowspan=1 colspan=1>0.7250.592</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>7200.2120.265</td><td rowspan=1 colspan=1>0.269</td><td rowspan=1 colspan=1>0.305</td><td rowspan=1 colspan=1>0.230</td><td rowspan=1 colspan=1>0.291</td><td rowspan=1 colspan=1>0.303</td><td rowspan=1 colspan=3>0.377</td><td rowspan=1 colspan=2>0.2210.2860.294</td><td rowspan=1 colspan=1>0.294</td><td rowspan=1 colspan=1>0.204</td><td rowspan=1 colspan=1>0.251</td><td rowspan=1 colspan=1>0.285</td><td rowspan=1 colspan=1>0.316</td><td rowspan=1 colspan=1>0.278</td><td rowspan=1 colspan=1>0.332</td><td rowspan=1 colspan=1>0.297</td><td rowspan=1 colspan=1>0.314</td><td rowspan=1 colspan=1>0.357</td><td rowspan=1 colspan=1>0.452</td><td rowspan=1 colspan=1>0.294</td><td rowspan=1 colspan=1>0.306</td><td rowspan=1 colspan=1>0.7590.674</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>960.3840.263192 0.4050.271</td><td rowspan=1 colspan=1>0.3960.410</td><td rowspan=1 colspan=1>0.271</td><td rowspan=1 colspan=1>0.382</td><td rowspan=1 colspan=1>0.269</td><td rowspan=1 colspan=1>0.933</td><td rowspan=1 colspan=3>0.5440</td><td rowspan=1 colspan=2>.4030.2990.424</td><td rowspan=1 colspan=1>0.281</td><td rowspan=1 colspan=1>0.379</td><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>0.451</td><td rowspan=1 colspan=1>0.295</td><td rowspan=1 colspan=1>0.424</td><td rowspan=1 colspan=1>0.307</td><td rowspan=1 colspan=1>0.508</td><td rowspan=1 colspan=1>0.355</td><td rowspan=1 colspan=1>0.591</td><td rowspan=1 colspan=1>0.363</td><td rowspan=1 colspan=1>0.600</td><td rowspan=1 colspan=1>0.314</td><td rowspan=1 colspan=1>0.6770.407</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>7200.4420.293</td><td rowspan=1 colspan=1>0.450</td><td rowspan=1 colspan=1>0.297</td><td rowspan=1 colspan=1>0.439</td><td rowspan=1 colspan=1>0.297</td><td rowspan=1 colspan=1>0.956</td><td rowspan=1 colspan=3>0.5500</td><td rowspan=1 colspan=2>.4640.3250.478</td><td rowspan=1 colspan=2>0.3090.463</td><td rowspan=1 colspan=1>0.285</td><td rowspan=1 colspan=1>0.506</td><td rowspan=1 colspan=1>0.321</td><td rowspan=1 colspan=1>0.4820</td><td rowspan=1 colspan=1>.336</td><td rowspan=1 colspan=1>0.490</td><td rowspan=1 colspan=1>0.326</td><td rowspan=1 colspan=1>0.658</td><td rowspan=1 colspan=1>0.394</td><td rowspan=1 colspan=1>0.681</td><td rowspan=1 colspan=1>0.360</td><td rowspan=1 colspan=1>0.8060.481</td></tr><tr><td rowspan=1 colspan=1>AQShunyi</td><td rowspan=1 colspan=2>960.6250.4741920.6630.494</td><td rowspan=1 colspan=1>0.632</td><td rowspan=1 colspan=1>0.475</td><td rowspan=1 colspan=1>0.627</td><td rowspan=1 colspan=1>0.478</td><td rowspan=1 colspan=1>0.628</td><td rowspan=1 colspan=3>0.491</td><td rowspan=1 colspan=2>0.6470.4820.669</td><td rowspan=1 colspan=2>0.5010.667</td><td rowspan=1 colspan=1>0.504</td><td rowspan=1 colspan=1>0.717</td><td rowspan=1 colspan=1>0.504</td><td rowspan=1 colspan=1>0.652</td><td rowspan=1 colspan=1>0.511</td><td rowspan=1 colspan=1>0.719</td><td rowspan=1 colspan=1>0.502</td><td rowspan=1 colspan=1>0.674</td><td rowspan=1 colspan=1>0.515</td><td rowspan=1 colspan=1>0.722</td><td rowspan=1 colspan=1>0.505</td><td rowspan=1 colspan=1>0.7380.542</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7200.7420</td><td rowspan=1 colspan=2>.5370.739</td><td rowspan=1 colspan=1>0.532</td><td rowspan=1 colspan=1>0.741</td><td rowspan=1 colspan=1>0.533</td><td rowspan=1 colspan=1>0.710</td><td rowspan=1 colspan=5>0.5310.7320.5290.758</td><td rowspan=1 colspan=2>0.5440.730</td><td rowspan=1 colspan=1>0.538</td><td rowspan=1 colspan=1>0.828</td><td rowspan=1 colspan=1>0.554</td><td rowspan=1 colspan=1>0.746</td><td rowspan=1 colspan=1>0.547</td><td rowspan=1 colspan=3>0.8350.5540.772</td><td rowspan=1 colspan=1>0.566</td><td rowspan=1 colspan=1>0.783</td><td rowspan=1 colspan=2>0.5450.9610.615</td></tr><tr><td rowspan=2 colspan=2>960.71301920.753AQWan 3360.782</td><td rowspan=1 colspan=2>.4650.719</td><td rowspan=1 colspan=1>0.466</td><td rowspan=1 colspan=1>0.716</td><td rowspan=1 colspan=1>0.467</td><td rowspan=1 colspan=1>0.721</td><td rowspan=1 colspan=5>0.4870.7300.4750.765</td><td rowspan=1 colspan=2>0.4930.757</td><td rowspan=1 colspan=1>0.495</td><td rowspan=1 colspan=1>0.798</td><td rowspan=1 colspan=1>0.489</td><td rowspan=1 colspan=1>0.745</td><td rowspan=1 colspan=1>0.504</td><td rowspan=1 colspan=3>0.8110.4920.757</td><td rowspan=1 colspan=1>0.502</td><td rowspan=1 colspan=1>0.795</td><td rowspan=1 colspan=2>0.4890.8260.537</td></tr><tr><td rowspan=1 colspan=2>0.4850.7620.4990.788</td><td rowspan=1 colspan=1>0.4860.500</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=6>720 0.8560.5290.8680.5330.857</td><td rowspan=1 colspan=10>0.5290.8110.5270.8500.5230.8820.5400.8430.531</td><td rowspan=1 colspan=1>0.944</td><td rowspan=1 colspan=3>0.5440.8650.541</td><td rowspan=1 colspan=7>0.9540.5470.8950.5590.8980.5360.9830.591</td></tr><tr><td rowspan=1 colspan=6>960.1740.2210.176 0.2230.1831920.2060.2460.2080.2470.213CzeLan 3360.2350.2720.2400.2750.243</td><td rowspan=1 colspan=1>0.247</td><td rowspan=1 colspan=6>0.2890.3750.1890.2500.239</td><td rowspan=1 colspan=3>0.3080.1910.249</td><td rowspan=1 colspan=1>0.212</td><td rowspan=1 colspan=3>0.2590.2070.288</td><td rowspan=1 colspan=3>0.2120.2570.294</td><td rowspan=1 colspan=4>0.3600.2200.2730.5840.542</td></tr><tr><td rowspan=1 colspan=7>7200.2790.3080.2940.3190.2890.333</td><td rowspan=1 colspan=6>0.3910.4630.2740.3310.327</td><td rowspan=1 colspan=3>0.3610.2940.332</td><td rowspan=1 colspan=1>0.336</td><td rowspan=1 colspan=3>0.3490.5550.499</td><td rowspan=1 colspan=3>0.3390.3460.415</td><td rowspan=1 colspan=4>0.4400.4750.4170.6920.603</td></tr><tr><td rowspan=1 colspan=7>960.4240.4230.4250.4230.4220.426192 0.4260.4230.4300.4240.426PM2.50.4313360.4210.4200.4280.4210.4230.425</td><td rowspan=1 colspan=6>0.4500.4750.4330.4300.448</td><td rowspan=1 colspan=3>0.4470.4370.442</td><td rowspan=1 colspan=1>0.433</td><td rowspan=1 colspan=3>0.4280.4260.452</td><td rowspan=1 colspan=3>0.4350.4310.453</td><td rowspan=1 colspan=4>0.4530.4590.4450.4790.4670.4480.4480.4400.4630.462</td></tr><tr><td rowspan=1 colspan=27>7200.4020.4160.4020.4140.4040.4240.4050.4910.4090.4290.4360.4480.4100.4700.410 0.4230.426 0.5040.4110.4220.4610.4930.4080.4240.493 0.507</td></tr></table>

Synthetic WINO-TS pre-training remains competitive with in-domain pre-training on several benchmarks, showing that the framework can learn useful temporal structure from generated data. Its performance is nevertheless less consistent, with the largest degradation occurring on Exchange. This result suggests that generic synthetic temporal primitives do not fully capture all domain-specific dynamics, making real in-domain data the more reliable default.

The Synthetic DINO+MAE variant improves over in-domain WINO-TS-FT on five of the nine datasets by MSE and ties or improves on six by MAE. However, this comparison changes both the pre-training corpus and the objective. It should therefore be interpreted as a joint data-source/objective ablation rather than as evidence for the effect of synthetic data alone. Overall, the results identify synthetic pre-training as a promising scaling direction, while also showing that its effectiveness depends on the match between the generated temporal structure and the downstream domain.

Table A6: Effect of the pre-training data source and objective. Each entry reports MSE/MAE; lower is better. Best values for each dataset and metric are shown in bold.
<table><tr><td></td><td colspan="2">In-domain WINO-TS-FT</td><td colspan="2">Synthetic WINO-TS-FT</td><td colspan="2">Synthetic DINO+MAE</td></tr><tr><td>Dataset</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.411</td><td>0.423</td><td>0.417</td><td>0.425</td><td>0.418</td><td>0.423</td></tr><tr><td>ETTh2</td><td>0.347</td><td>0.386</td><td>0.365</td><td>0.402</td><td>0.359</td><td>0.404</td></tr><tr><td>ETTm1</td><td>0.346</td><td>0.379</td><td>0.350</td><td>0.378</td><td>0.345</td><td>0.377</td></tr><tr><td>ETTm2</td><td>0.250</td><td>0.308</td><td>0.250</td><td>0.309</td><td>0.249</td><td>0.308</td></tr><tr><td>Weather</td><td>0.224</td><td>0.262</td><td>0.226</td><td>0.261</td><td>0.227</td><td>0.261</td></tr><tr><td>Electricity</td><td>0.163</td><td>0.254</td><td>0.164</td><td>0.256</td><td>0.164</td><td>0.256</td></tr><tr><td>Exchange</td><td>0.377</td><td>0.407</td><td>0.843</td><td>0.540</td><td>0.469</td><td>0.469</td></tr><tr><td>Solar</td><td>0.204</td><td>0.258</td><td>0.238</td><td>0.277</td><td>0.199</td><td>0.252</td></tr><tr><td>Traffic</td><td>0.411</td><td>0.276</td><td>0.423</td><td>0.283</td><td>0.414</td><td>0.278</td></tr></table>

Synthetic WINO-TS-FT changes only the pre-training corpus, whereas Synthetic DINO+MAE changes both the pre-training corpus and the pre-training objective.

Table A7: Classification datasets and reported sample counts.
<table><tr><td>Dataset</td><td>Reported samples</td></tr><tr><td>EthanolConcentration</td><td>261</td></tr><tr><td>SpokenArabicDigits</td><td>6599</td></tr><tr><td>FaceDetection</td><td>5890</td></tr><tr><td>JapaneseVowels</td><td>270</td></tr><tr><td>SelfRegulationSCP1</td><td>268</td></tr><tr><td>SelfRegulationSCP2</td><td>200</td></tr><tr><td>UWave</td><td>120</td></tr><tr><td>Heartbeat</td><td>204</td></tr><tr><td>Handwriting</td><td>150</td></tr></table>

## G Extended Classification Results

Classification fine-tunes the full WINO-TS-pretrained backbone jointly with a supervised classification head. Table A7 presents the dataset metadata, and Table A8 reports the results.

WINO-TS exceeds iTransformer on five of the nine datasets, including SpokenArabicDigits, FaceDetection, UWave, EthanolConcentration, and Heartbeat, where class identity may be associated with robust multi-scale morphology. Its weaker performance on JapaneseVowels, SelfRegulationSCP1/2, and especially Handwriting is consistent with the interpretation that some classification tasks depend on fine local detail that can be attenuated by the easy-view shrinkage. This is an interpretation of the observed pattern rather than a direct causal measurement.

## H Additional Ablation Details

The following ablations supplement the objective, backbone, and vision-augmentation analyses reported in the main paper. The vision-like augmentation comparison remains in Table 5 of the main paper and is not duplicated here.

Unless stated otherwise, each entry is MSE/MAE averaged over forecasting horizons 96, 192, 336, 720 , and lower values are better. Because the tables correspond to separate controlled runs, each should be interpreted as a within-table comparison rather than by comparing small numerical differences across different ablation tables.

## H.1 Backbone Ablation

The backbone ablation evaluates whether WINO-TS depends on the TimeMixer encoder or whether the wavelet self-distillation recipe can be used with other backbone families. The DINO objective and full wavelet-augmentation pool are held fixed, and only the encoder is changed. Table A9 summarizes the results. Among the currently reported entries, TimeMixer gives the strongest overall performance across the evaluated forecasting datasets, while PatchTST remains competitive on several benchmarks.

Table A8: Classification accuracy after full end-to-end fine-tuning initialized from a WINO-TSpretrained backbone. Both the backbone and classification head are updated. Higher is better; best results are in bold.
<table><tr><td>Dataset</td><td>WINO-TS (Ours)</td><td>iTransformer Liu et al. [2023]</td></tr><tr><td>EthanolConcentration</td><td>0.2970</td><td>0.2810</td></tr><tr><td>SpokenArabicDigits</td><td>0.9900</td><td>0.9827</td></tr><tr><td>FaceDetection</td><td>0.6700</td><td>0.6592</td></tr><tr><td>JapaneseVowels</td><td>0.9570</td><td>0.9811</td></tr><tr><td>SelfRegulationSCP1</td><td>0.8770</td><td>0.9113</td></tr><tr><td>SelfRegulationSCP2</td><td>0.5000</td><td>0.5611</td></tr><tr><td>UWave</td><td>0.8594</td><td>0.8531</td></tr><tr><td>Heartbeat</td><td>0.7805</td><td>0.7463</td></tr><tr><td>Handwriting</td><td>0.0376</td><td>0.2565</td></tr></table>

Table A9: Backbone ablation for WINO-TS (DINO pre-training,full augmentation family). In-domain forecasting using linear probe, with MSE/MAE averaged over horizons 96, 192, 336, 720 . The DINO objective and augmentation are held fixed — all backbones use the full wavelet-augmentation family (pool sym4, sym6, sym8, db4, db6, coif2 ) — while only the encoder backbone is swapped. Lower is better; best in red, second-best in blue.
<table><tr><td rowspan="3">Dataset</td><td colspan="2">TimeMixer(MLP) Wang et al. [2024]</td><td colspan="2">PatchTST Nie et al. [2023]</td><td colspan="2">iTransformer TS2Vec(TCN) Liu et al. [2023]</td><td colspan="2">Yue et al. [2022]</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.416</td><td>0.423</td><td>0.422</td><td>0.432</td><td>0.656</td><td>0.558</td><td>0.541</td><td>0.505</td></tr><tr><td>ETTh2</td><td>0.347</td><td>0.385</td><td>0.357</td><td>0.392</td><td>0.427</td><td>0.447</td><td>0.388</td><td>0.423</td></tr><tr><td>ETTm1</td><td>0.364</td><td>0.386</td><td>0.370</td><td>0.388</td><td>0.427</td><td>0.423</td><td>0.431</td><td>0.432</td></tr><tr><td>ETTm2</td><td>0.252</td><td>0.308</td><td>0.252</td><td>0.309</td><td>0.290</td><td>0.344</td><td>0.286</td><td>0.341</td></tr><tr><td>Weather</td><td>0.238</td><td>0.272</td><td>0.240</td><td>0.274</td><td>0.269</td><td>0.303</td><td>0.288</td><td>0.305</td></tr><tr><td>Electricity</td><td>0.167</td><td>0.260</td><td>0.1700.263</td><td></td><td>0.334</td><td>0.419</td><td>0.236</td><td>0.336</td></tr></table>

This supports the interpretation of WINO-TS as a transferable pre-training recipe while also showing that the choice of encoder remains consequential.

## H.2 Multi-View Ablation

We additionally evaluate a multi-view variant of WINO-TS with Q=2 easy views and $V \in \{ 2 , 4 , 6 \}$ hard views per window, following the pipeline and hyperparameters of the default configuration. The wavelet basis, noise scale, and noise realization are sampled independently for each view. With $Q = 2$ , the easy–easy matching term $\mathcal { T } _ { \mathrm { e e } }$ in Eq. (16) is active. As reported in Table A18, these configurations underperform the default $Q { = } 1 , V { = } 1$ setting across the evaluated datasets despite their higher per-window cost, and we therefore adopt the single-view-pair default throughout the paper.

## H.3 Objective-Ablation Definitions

The main-paper objective ablation compares the default WINO-TS loss with a hybrid maskedreconstruction objective and three non-default pre-training objectives. The default WINO-TS model uses only the DINO-style cross-view self-distillation loss. WINO-TS+MAE uses

$$
\mathcal { L } _ { \mathrm { h y b r i d } } = \phi \mathcal { L } _ { \mathrm { D I N O } } + ( 1 - \phi ) \mathcal { L } _ { \mathrm { M A E } } , \qquad \phi = 0 . 6 ,\tag{43}
$$

where $\mathcal { L } _ { \mathrm { M A E } }$ reconstructs masked patches against the raw signal. Under this convention, pure WINO-TS corresponds to $\phi = 1$

The MAE baseline uses masked reconstruction without a teacher. NTP uses autoregressive next-token prediction, and JEPA predicts a target latent representation rather than reconstructing raw observations. The backbone and wavelet augmentation are held fixed for the WINO-TS and WINO-TS+MAE comparison.

Table A10: Augmentation-family ablation for WINO-TS with linear probe on the DINO objective. Each cell reports MSE/MAE averaged over horizons 96, 192, 336, 720 . Lower is better; best results are in bold.
<table><tr><td rowspan="2"></td><td colspan="2">Daub.</td><td colspan="2">Zero-out</td><td colspan="2">Full pool</td></tr><tr><td>Dataset MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.416</td><td>0.423</td><td>0.415</td><td>0.422</td><td>0.416</td><td>0.423</td></tr><tr><td>ETTh2</td><td>0.347</td><td>0.385</td><td>0.348</td><td>0.387</td><td>0.347</td><td>0.385</td></tr><tr><td>ETTm1</td><td>0.363</td><td>0.386</td><td>0.359</td><td>0.385</td><td>0.364</td><td>0.386</td></tr><tr><td>ETTm2</td><td>0.253</td><td>0.308</td><td>0.257</td><td>0.312</td><td>0.252</td><td>0.308</td></tr><tr><td>Weather</td><td>0.237</td><td>0.272</td><td>0.238</td><td>0.273</td><td>0.238</td><td>0.273</td></tr></table>

Daub.: Daubechies-only pool. Zero-out: finest-scale coefficient zeroing. Full pool: default WINO-TS pool.

Table A11: Pre-training objective ablation (WINO-TS, TimeMixer backbone, full augmentation pool, Linear probe). In-domain forecasting MSE/MAE averaged over horizons 96, 192, 336, 720 . DINO: pure self-distillation — the student matches the teacher’s centered/sharpened prototype distribution across augmented views, with no reconstruction (ϕ=1). DINO+MAE: adds a maskedautoencoding term in which the student reconstructs masked patches against the raw signal, blended as ϕ DINO + (1 ϕ) MAE (ϕ=0.6). MAE, NTP and JEPA are non-distillation baselines: pure masked autoencoding (reconstruct masked patches, no teacher) , next-token prediction (autoregressive forecasting-style pre-training) and joint-embedding predictive architecture respectively. Lower is better; best in red, second-best in blue.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">WINO-TS</td><td colspan="2">WINO- TS+MAE</td><td colspan="2">MAE</td><td colspan="2">NTP</td><td colspan="2">JEPA</td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.416</td><td>0.423</td><td>0.418</td><td>0.424</td><td>0.465</td><td>0.465</td><td>0.428</td><td>0.435</td><td>0.442</td><td>0.447</td></tr><tr><td>ETTh2</td><td>0.347</td><td>0.385</td><td>0.355</td><td>0.389</td><td>0.449</td><td>0.457</td><td>0.407</td><td>0.427</td><td>0.390</td><td>0.426</td></tr><tr><td>ETTm1</td><td>0.364</td><td>0.386</td><td>0.370</td><td>0.388</td><td>0.349</td><td>0.387</td><td>0.346</td><td>0.381</td><td>0.375</td><td>0.393</td></tr><tr><td>ETTm2</td><td>0.252</td><td>0.308</td><td>0.258</td><td>0.313</td><td>0.294</td><td>0.344</td><td>0.263</td><td>0.320</td><td>0.274</td><td>0.332</td></tr><tr><td>Weather</td><td>0.238</td><td>0.273</td><td>0.241</td><td>0.274</td><td>0.266</td><td>0.294</td><td>0.226</td><td>0.265</td><td>0.229</td><td>0.266</td></tr></table>

## H.4 Augmentation-Family Ablation

This ablation compares the default wavelet pool with a Daubechies-only pool and a hard-view variant that zeros the finest-scale coefficients. All variants retain the default asymmetric network routing: the teacher processes only easy views, whereas the student processes both easy and hard views under the same cross-view self-distillation objective. The results are depicted in Table A10

No variant dominates every dataset. Zero-out is strongest on ETTh1 and ETTm1, the Daubechies-only pool is strongest on Weather, and the full pool is strongest or tied on ETTh2 and ETTm2. The default full pool is retained because it provides phase, support, and smoothness diversity while remaining competitive across datasets.

## H.5 Pre-training Objective Ablation

The objective ablation evaluates whether adding reconstruction-style auxiliary objectives improves WINO-TS. We compare the default DINO-only objective against variants that add reconstruction or masked-token auxiliary terms while keeping the backbone and wavelet augmentation fixed. The results, depicted in Table A11 show that the pure DINO objective is the most reliable choice overall. Auxiliary reconstruction-style terms introduce mixed effects, suggesting that the main gains of WINO-TS come from wavelet-domain self-distillation rather than from adding reconstruction losses.

## H.6 Wavelet Pool and Transform Ablation

We next separate the effects of the sampled wavelet family and the transform. The pool comparison fixes the decimated DWT and compares the mixed pool with db4, db6, db8 and sym4, sym6, sym8 . The transform comparison fixes the mixed pool and replaces DWT with the stationary wavelet transform (SWT) or maximal-overlap DWT (MODWT). The results are depicted in Table A12.

Table A12: Wavelet augmentation ablation for WINO-TS (DINO pre-training, TimeMixer backbone, Linear probe). In-domain forecasting MSE/MAE averaged over horizons 96, 192, 336, 720 . The first column is the default WINO-TS (DWT with the Mixed pool $\{ \mathrm { s y m 4 , s y m 6 , s y m \bar { 8 } , d b 4 , d b 6 , c o i f 2 } \}$ and is the shared reference. Wavelet pool fixes the DWT and sweeps the pool $( \mathrm { d } \mathsf { b } = \{ \mathrm { d } \mathsf { b } 4 , \mathrm { d } \mathsf { b } 6 , \mathrm { d } \mathsf { \bar { b } } 8 \}$ ; sym = sym4,sym6,sym8 ). Transform fixes the Mixed pool and swaps the transform (SWT / MODWT). Lower is better; best in red, second-best in blue.
<table><tr><td></td><td colspan="2">Ours</td><td colspan="4">Wavelet pool</td><td colspan="4">Transform</td></tr><tr><td></td><td colspan="2">DWT/Mixed</td><td colspan="2">db</td><td colspan="2">sym</td><td colspan="2">SWT</td><td colspan="2">MODWT</td></tr><tr><td>Dataset</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.416</td><td>0.423</td><td>0.416</td><td>0.423</td><td>0.416</td><td>0.423</td><td>0.415</td><td>0.422</td><td>0.417</td><td>0.423</td></tr><tr><td>ETTh2</td><td>0.347</td><td>0.385</td><td>0.347</td><td>0.386</td><td>0.347</td><td>0.386</td><td>0.347</td><td>0.386</td><td>0.346</td><td>0.385</td></tr><tr><td>ETTm1</td><td>0.364</td><td>0.386</td><td>0.364</td><td>0.386</td><td>0.365</td><td>0.387</td><td>0.360</td><td>0.385</td><td>0.364</td><td>0.386</td></tr><tr><td>ETTm2</td><td>0.252</td><td>0.308</td><td>0.253</td><td>0.309</td><td>0.258</td><td>0.312</td><td>0.257</td><td>0.312</td><td>0.254</td><td>0.309</td></tr><tr><td>Weather</td><td>0.238</td><td>0.273</td><td>0.237</td><td>0.272</td><td>0.238</td><td>0.272</td><td>0.238</td><td>0.273</td><td>0.239</td><td>0.273</td></tr><tr><td>Electricity</td><td>0.167</td><td>0.260</td><td>0.165</td><td>0.256</td><td>0.166</td><td>0.257</td><td>0.165</td><td>0.257</td><td>0.166</td><td>0.258</td></tr></table>

Table A13: Shrinkage-strength (ρ) ablation for WINO-TS (DINO pre-training, TimeMixer backbone, DWT with the Mixed pool, Linear probe). We vary the soft-threshold shrinkage ratio $\rho$ that sets the per-level denoising threshold of the teacher’s easy view: detail coefficients below ρ max( detail ) are shrunk toward zero, so small $\rho$ preserves high-frequency detail (weak invariance) while large ρ enforces aggressive low-pass smoothing (strong invariance); $\rho = 0 . 6$ is the WINO-TS default. All other factors—transform, wavelet pool, objective, and backbone—are held fixed. Each cell reports MSE/MAE averaged over horizons 96, 192, 336, 720 ; lower is better, best per row in bold.
<table><tr><td rowspan="2">Dataset</td><td colspan="2"> $\rho = 0 . 2$ </td><td colspan="2"> $\rho = 0 . 4$ </td><td colspan="2"> $\rho = 0 . 6$ </td><td colspan="2"> $\rho = 0 . 8$ </td><td colspan="2"> $\rho = 1 . 0$ </td></tr><tr><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.428</td><td>0.430</td><td>0.464</td><td>0.458</td><td>0.416</td><td>0.423</td><td>0.419</td><td>0.426</td><td>0.465</td><td>0.458</td></tr><tr><td>ETTh2</td><td>0.348</td><td>0.387</td><td>0.370</td><td>0.396</td><td>0.347</td><td>0.385</td><td>0.370</td><td>0.396</td><td>0.370</td><td>0.396</td></tr><tr><td>ETTm1</td><td>0.371</td><td>0.391</td><td>0.383</td><td>0.403</td><td>0.364</td><td>0.386</td><td>0.382</td><td>0.404</td><td>0.384</td><td>0.405</td></tr><tr><td>ETTm2</td><td>0.250</td><td>0.310</td><td>0.249</td><td>0.310</td><td>0.252</td><td>0.308</td><td>0.250</td><td>0.310</td><td>0.250</td><td>0.310</td></tr><tr><td>Weather</td><td>0.243</td><td>0.273</td><td>0.244</td><td>0.273</td><td>0.238</td><td>0.273</td><td>0.243</td><td>0.278</td><td>0.243</td><td>0.278</td></tr></table>

The shift-invariant transforms do not provide a uniform advantage. SWT is strongest on ETTh1 and ETTm1, whereas the decimated DWT is strongest on ETTm2 and remains close to the best result elsewhere. The differences among the mixed, Daubechies, and Symlet pools are likewise small and dataset-dependent. These results support the standard DWT as a simple and competitive default rather than establishing it as uniformly superior.

## H.7 Shrinkage-Strength Ablation

The parameter $\rho$ controls the easy-view threshold $\tau _ { j } = \rho \operatorname* { m a x } _ { k } \left| d _ { j , k } \right|$ . Small values keep the teacher view close to the original signal, whereas large values suppress a greater fraction of the detail coefficients. We evaluate $\rho \in \{ 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 , 1 . 0 \}$ while holding the backbone, objective, wavelet pool, and hard-view noise range fixed.

Table A13 reports the results. The ablation shows that moderate shrinkage provides the most stable performance across datasets. Weak shrinkage produces smaller gains, indicating that the teacher view remains too similar to the student input and provides a weaker invariance signal. Overly aggressive shrinkage also degrades performance on several datasets, suggesting that high-frequency detail bands contain task-relevant information that should be attenuated rather than fully removed. The default value $\rho = 0 . 6$ achieves the best overall trade-off between denoising and detail preservation, and we use it in all main experiments.

## H.8 Wavelet Depth and Wavelet Pool

We next ablate two design choices in the wavelet view generator: the decomposition depth J and the wavelet sampling strategy. The depth J determines the scale at which the signal is split into the shared low-frequency approximation and the editable detail subspace. A shallow decomposition leaves more local variation in the approximation, while a deeper decomposition makes the teacher view more aggressively coarse. We compare $J \in \{ 2 , 3 , 4 \}$ using the same DINO objective and backbone.

Table A14: Wavelet-depth ablation using linear probe. Each cell reports MSE/MAE averaged over horizons 96, 192, 336, 720 . Lower is better; best results are in bold.
<table><tr><td colspan="3"> $J = 2$ </td><td colspan="2"> $J = 3$ </td><td colspan="2"> $J = 4$ </td></tr><tr><td>Dataset</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.426</td><td>0.429</td><td>0.416</td><td>0.423</td><td>0.438</td><td>0.440</td></tr><tr><td>ETTh2</td><td>0.370</td><td>0.395</td><td>0.347</td><td>0.385</td><td>0.361</td><td>0.391</td></tr><tr><td>ETTm1</td><td>0.361</td><td>0.386</td><td>0.364</td><td>0.386</td><td>0.362</td><td>0.385</td></tr><tr><td>ETTm2</td><td>0.255</td><td>0.312</td><td>0.252</td><td>0.308</td><td>0.248</td><td>0.308</td></tr><tr><td>Weather</td><td>0.238</td><td>0.272</td><td>0.238</td><td>0.273</td><td>0.239</td><td>0.273</td></tr></table>

Table A15: Wavelet-pool and basis-sampling ablation (Linear probe). Fixed uses a single wavelet basis for both views. Shared sampled draws one wavelet from $\dot { \mathcal { P } }$ and uses it for both easy and hard views. Independent sampled is the WINO-TS default, drawing $w ^ { \mathrm { e a s y } }$ and $w ^ { \mathrm { h a r d } }$ independently. Each cell reports MSE/MAE averaged over horizons 96, 192, 336, 720 . Lower is better; best results are in bold.
<table><tr><td></td><td colspan="2">Fixed basis</td><td colspan="2">Shared sampled</td><td colspan="2">Independent sampled</td></tr><tr><td>Dataset</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.429</td><td>0.431</td><td>0.420</td><td>0.429</td><td>0.416</td><td>0.423</td></tr><tr><td>ETTh2</td><td>0.369</td><td>0.394</td><td>0.363</td><td>0.391</td><td>0.347</td><td>0.385</td></tr><tr><td>ETTm1</td><td>0.362</td><td>0.387</td><td>0.363</td><td>0.387</td><td>0.364</td><td>0.386</td></tr><tr><td>ETTm2</td><td>0.247</td><td>0.308</td><td>0.275</td><td>0.328</td><td>0.252</td><td>0.308</td></tr><tr><td>Weather</td><td>0.243</td><td>0.279</td><td>0.243</td><td>0.279</td><td>0.238</td><td>0.273</td></tr></table>

Table A14 shows that intermediate depth provides the most stable performance. With $J = 2$ , the easy and hard views remain too close in scale, weakening the self-distillation signal. With $J = 4 ,$ , the approximation becomes overly coarse on several datasets, causing the teacher target to discard useful local structure. The default $J = 3$ provides the best overall trade-off and is used throughout the main experiments.

We also evaluate the effect of wavelet-basis sampling. Specifically, we compare a fixed-basis setting, a shared-sampled setting in which the easy and hard views use the same randomly sampled wavelet, and the default independent-sampling setting in which $w ^ { \mathrm { e a s y } }$ and $w ^ { \mathrm { h a r d } }$ are drawn independently from the wavelet pool. Table A15 shows that independent sampling is the most robust overall. Using a fixed basis performs competitively on some datasets but is less stable, suggesting that the encoder can overfit to basis-specific phase or boundary behavior. Shared sampling improves over a fixed basis by adding wavelet diversity, while independent sampling further improves robustness by introducing mild cross-basis leakage between the easy and hard views. This supports the default design of sampling the two view bases independently.

## H.9 View-Pairing Ablation

The default WINO-TS configuration follows an asymmetric DINO-style view-pairing rule. The teacher processes only the easy views, whereas the student processes both easy and hard views. Every student view is matched to every easy teacher view, except that the teacher and student outputs corresponding to the identical easy-view realization are not paired. Thus, both student easy and student hard views contribute to the default objective.

We compare this default rule against two alternative pairing strategies. The same-view variant additionally includes the matching easy-teacher/easy-student pairs that are excluded by the default objective. The symmetric variant allows both the teacher and student to process easy and hard views and applies cross-view matching across non-identical view pairs.

Table A16 shows that the default asymmetric DINO-style pairing provides the most stable performance across the evaluated datasets. Including identical easy–easy pairs does not yield consistent gains, suggesting that direct same-view matching can weaken the desired cross-view invariance pressure. The fully symmetric formulation is competitive on ETTm1 and ETTm2 but is less stable on ETTh1, ETTh2, and Weather.

Table A16: View-pairing ablation using linear probing. Default denotes the WINO-TS pairing rule: the teacher processes easy views, the student processes both easy and hard views, and every student view is matched to every teacher easy view except the identical easy-view index. Same-view additionally includes the matching easy–easy pairs. Symmetric allows both networks to process easy and hard views. Each cell reports MSE/MAE averaged over horizons 96, 192, 336, 720 . Lower is better; best results are in bold.
<table><tr><td colspan="3">Default</td><td colspan="2">Same-view</td><td colspan="2">Symmetric</td></tr><tr><td>Dataset</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.416</td><td>0.423</td><td>0.435</td><td>0.434</td><td>0.420</td><td>0.430</td></tr><tr><td>ETTh2</td><td>0.347</td><td>0.385</td><td>0.363</td><td>0.391</td><td>0.364</td><td>0.392</td></tr><tr><td>ETTm1</td><td>0.364</td><td>0.386</td><td>0.361</td><td>0.385</td><td>0.362</td><td>0.385</td></tr><tr><td>ETTm2</td><td>0.252</td><td>0.308</td><td>0.248</td><td>0.309</td><td>0.248</td><td>0.307</td></tr><tr><td>Weather</td><td>0.238</td><td>0.273</td><td>0.242</td><td>0.279</td><td>0.239</td><td>0.275</td></tr></table>

Table A17: Seed robustness of WINO-TS LP (DINO pre-training, TimeMixer backbone, linear probe). In-domain forecasting MSE/MAE averaged over horizons 96, 192, 336, 720 , reported as mean std over 5 seeds ( 42, 777, 1773, 2024, 3407 ). Lower is better.
<table><tr><td>Dataset</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td> $0 . 4 1 7 \pm 0 . 0 0 1$ </td><td> $0 . 4 2 4 \pm 0 . 0 0 1$ </td></tr><tr><td>ETTh2</td><td> $0 . 3 5 0 \pm 0 . 0 0 4$ </td><td> $0 . 3 8 7 \pm 0 . 0 0 2$ </td></tr><tr><td>ETTm1</td><td> $0 . 3 6 1 \pm 0 . 0 0 3$ </td><td> $0 . 3 8 5 \pm 0 . 0 0 1$ </td></tr><tr><td>ETTm2</td><td> $0 . 2 4 8 \pm 0 . 0 0 2$ </td><td> $0 . 3 0 8 \pm 0 . 0 0 2$ </td></tr><tr><td>Weather</td><td> $0 . 2 3 8 \pm 0 . 0 0 1$ </td><td> $0 . 2 7 2 \pm 0 . 0 0 1$ </td></tr></table>

These results indicate that the benefit of WINO-TS depends not only on generating wavelet-domain views but also on assigning distinct roles to the two networks: the teacher produces targets only from easy views, while the student learns from both easy and hard views through non-trivial cross-view matching.

## H.10 Seed Robustness

We evaluate WINO-TS-LP across five random seeds, 42, 777, 1773, 2024, 3407 . The results are depicted in Table H.10. The standard deviations are small across the tested seeds, indicating limited run-to-run variability for the reported linear-probe setting.

Table A18: Multi-view ablation with a TimeMixer backbone and linear probing. The default uses Q=1 easy view and V=1 hard view; the alternatives use Q=2 and $V \in \{ \bar { 2 } , 4 , 6 \}$ . Each view uses an independently sampled wavelet basis and, for hard views, independently sampled noise. Results are MSE/MAE averaged over horizons 96, 192, 336, 720 ; lower is better and best results are in bold.
<table><tr><td>Dataset</td><td colspan="2">WINO-TS (Q=1, V=1)</td><td colspan="2">Q=2, V=2</td><td colspan="2">Q=2,V=4</td><td colspan="2">Q=2,V=6</td></tr><tr><td></td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>ETTh1</td><td>0.416</td><td>0.423</td><td>0.433</td><td>0.433</td><td>0.429</td><td>0.431</td><td>0.429</td><td>0.430</td></tr><tr><td>ETTh2</td><td>0.347</td><td>0.385</td><td>0.363</td><td>0.391</td><td>0.362</td><td>0.391</td><td>0.362</td><td>0.390</td></tr><tr><td>ETTm1</td><td>0.364</td><td>0.386</td><td>0.361</td><td>0.385</td><td>0.362</td><td>0.385</td><td>0.362</td><td>0.386</td></tr><tr><td>ETTm2</td><td>0.252</td><td>0.308</td><td>0.247</td><td>0.308</td><td>0.248</td><td>0.309</td><td>0.247</td><td>0.308</td></tr><tr><td>Weather</td><td>0.238</td><td>0.273</td><td>0.240</td><td>0.276</td><td>0.239</td><td>0.275</td><td>0.239</td><td>0.275</td></tr></table>