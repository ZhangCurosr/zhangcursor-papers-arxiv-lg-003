# Towards Robust Time Series Learning via Capacity-Centric Modulation

Siru Zhong, Senzhang Wang, James T. Kwok, Fellow, IEEE, and Yuxuan Liang

Abstract—Sample-level reliability heterogeneity is common in deep time series learning. Standard training pipelines apply a uniform regularization setting to all samples, which can underregularize corrupted samples and over-restrict clean samples. Common robustness approaches filter observations in data space or impose priors on latent representations. We propose Capacity Centric Modulation (CCM) as a complementary, sample-adaptive regularization principle. Under this principle, we introduce SACM (Sample-Adaptive Capacity Modulation), a task-agnostic framework that exploits spectral sparsity to assign sample-wise dropout probabilities along internal activation paths. SACM integrates into existing backbones without architectural redesign and preserves the deterministic inference pipeline. Across 301 real-world dataset– backbone pairs covering 9 forecasting, 32 classification, and 4 anomaly-detection datasets, SACM reduces forecasting MSE by 6.7% on average and improves classification accuracy and point-adjusted F1 by 3.04% and 17.05%, respectively, relative to unmodified backbones, with zero test-time overhead.

Index Terms—Time series modeling, robust learning, capacity centric modulation, spectral analysis.

## I. INTRODUCTION

IME series modeling underpins applications in weather, finance, energy demand, and traffic [1, 2]. Temporal learning spans three primary tasks: predictive (forecasting), discriminative (classification), and reconstructive (anomaly detection). The growing volume and complexity of real-world temporal data have driven rapid innovations in deep temporal architectures, ranging from Recurrent Neural Networks (RNNs) [3, 4] and Convolutional Networks (CNNs) [5] to Multi-Layer Perceptrons (MLPs) [6, 7], Transformers [8–10], and diffusionbased generative models [11, 12].

Siru Zhong and Yuxuan Liang are with Data Science & Analytics Thrust, The Hong Kong University of Science and Technology (GZ), Guangzhou, China. Senzhang Wang is with School of Computer Science & Engineering, Central South University, Changsha, China. James T. Kwok is with Dept. of Computer Science & Engineering, The Hong Kong University of Science and Technology, Hong Kong. Corresponding author: Yuxuan Liang (yuxliang@outlook.com).

Despite these architectural advances, deep time series models face sample-level reliability heterogeneity [13–15]. In realworld environments, time series observations often combine diverse signal regimes with unpredictable corruptions. Some samples exhibit clean, stationary periodicity and coherent trends, while others suffer from heavy-tailed measurement noise, transient sensor failures, missing values, or abrupt distributional shifts. Under standard training practices, the model uses a fixed dropout or regularization configuration for every training sample, resulting in the same nominal capacity allocation [16–18]. This creates a learning dilemma: a capacity budget expressive enough to capture intricate temporal dynamics in clean samples overfits spurious artifacts in noisedominated samples, learning incidental fluctuations rather than transferable temporal patterns [19, 20].

To alleviate noise overfitting, existing robust time series methods predominantly follow two conventional paradigms:

• Data-Centric Selection: Approaches such as RobustTSF [19] and Selective Learning [20] frame robustness as a screening problem. They employ auxiliary statistical metrics or proxy networks to identify and exclude corrupted samples or time steps. While effective at filtering outliers, this binary “keep-or-drop” screening can cause information loss, frequently discarding critical tail events or subtle transitions that mimic noise signatures. Furthermore, such screening mechanisms are typically coupled with task-specific loss heuristics, limiting cross-task applicability.

• Prior-Centric Modeling: Approaches such as BayesTSF [21] and RSTIB [22] treat robustness as a representationdisentanglement problem. By incorporating Variational Autoencoders (VAEs) [23] or Bayesian Neural Networks (BNNs), they constrain latent representations to satisfy prior distributions (e.g., Gaussianity). However, real-world temporal dynamics are inherently non-stationary, making rigid distributional assumptions brittle while introducing additional inference overhead and optimization complexity.

![](images/cce2180f1691ae25885ab679094bc7cc2210d3b2550ee6a74c0036c958c7cd29.jpg)  
Fig. 1. Overview of robustness paradigms and intervention levels. a) Data-Centric Selection removes or reweights observations at the data level; b) Prior-Centric Modeling constrains latent representations; c) Capacity-Centric Modulation modulates effective capacity along internal activation paths for predictive (forecasting), discriminative (classification), and reconstructive (anomaly detection) tasks.

TABLE I  
COMPARISON OF ROBUSTNESS PARADIGMS IN TIME SERIES MODELING. SACM INSTANTIATES CAPACITY-CENTRIC MODULATION ON THE ACTIVEHYPOTHESIS SPACE (H), OPERATING COMPLEMENTARILY TO CONVENTIONAL DATA- AND PRIOR-CENTRIC APPROACHES.
<table><tr><td>Paradigm</td><td>Intervention Space</td><td>Core Philosophy</td><td>Representative Methods</td><td>Key Characteristics &amp; Limitations</td></tr><tr><td>Data-Centric Selection (e.g., Hard Selection)</td><td>Data Space (X)</td><td>&quot;Filter Noise&quot; Exclude unreliable observations</td><td>RobustTSF [19] Selective Learning [20]</td><td>• Discrete keep-or-drop screening ● Risks discarding valid rare events; task-coupled</td></tr><tr><td>Prior-Centric Modeling (e.g., Complex Priors)</td><td>Latent Space (Z)</td><td>&quot;Disentangle Noise&quot; Constrain latent distributions</td><td>BayesTSF [21] RSTIB [22]</td><td>• Imposes explicit priors (e.g., Gaussian VAEs/BNNs) • Sensitive to non-stationarity; heavy inference cost</td></tr><tr><td>Capacity-Centric Modulation (Sample-Adaptive Principle)</td><td>Hypothesis Space (H)</td><td>“Coexist with Noise&quot; Match capacity to reliability</td><td>SACM (Ours)</td><td>• Sample-adaptive active capacity allocation • Task-agnostic, zero inference overhead, complementary</td></tr></table>

Both paradigms leave sample-adaptive capacity allocation inside the backbone unaddressed. Applying a uniform regularization strength across heterogeneous samples inevitably leads to a capacity mismatch: it under-regularizes corrupted samples (encouraging spurious noise memorization) while overrestricting the expressive fidelity available to clean ones.

To address this, we propose Capacity-Centric Modulation (CCM) as a complementary principle for robust time series learning (Figure 1 and Table I). Instead of screening observations in the data space (X) or enforcing rigid probabilistic priors in the latent space (Z), CCM operates on the model’s active hypothesis space (H) through sample-adaptive regularization. Along internal activation pathways, CCM scales stochastic regularization conditioned on sample reliability: unreliable samples face stronger pathway dropping to suppress noise memorization, whereas clean samples retain more active capacity to capture intricate temporal dynamics.

Under the CCM principle, we introduce SACM (Sample-Adaptive Capacity Modulation), a unified, task- and modelagnostic framework for robust time series learning. To realize sample-adaptive capacity allocation without expensive noise labels or teacher networks, SACM leverages spectral sparsity as a label-free inductive bias. In the frequency domain, structured temporal signals often concentrate energy into dominant spectral modes, whereas corruptions and perturbations tend to produce less concentrated or more diffuse spectral residuals [24, 25]. SACM features a differentiable spectral residual scorer that detrends the input, applies a Spectral Flatness Measure (SFM)-anchored filter, and computes the reconstruction residual as an unreliability proxy. A learnable rate mapper translates this residual into sample-specific stochastic retention rates along the network’s trainable paths. Via the Straight-Through Estimator (STE), SACM optimizes capacity allocation jointly with the backbone end-to-end. During evaluation, capacity modulation is bypassed, preserving the deterministic inference pipeline with zero test-time overhead.

Relationship to Prior Conference Work. Compared to our preliminary conference paper DROPOUTTS [26], which introduced an empirical spectral dropout heuristic solely for time series forecasting, this journal article systematically advances the formulation into a unified, theoretically motivated framework with three principal extensions:

1) Capacity-Centric Perspective: We reframe sample-level dropout from a heuristic into Capacity-Centric Modulation (CCM), establishing it as a complementary principle alongside data- and prior-centric modeling (§I, §III).

2) Principled Framework & End-to-End Design: Moving beyond heuristic dropping, we develop a systematic, labelfree pipeline featuring an SFM-anchored spectral scorer, a learnable rate mapper, and STE-based joint optimization, supported by a surrogate risk analysis (§IV, §V).

3) Cross-Task & In-Depth Empirical Generalization: We extend the framework beyond forecasting to classification across 32 benchmarks and anomaly detection across 4 benchmarks, covering 220+ matched dataset–backbone configurations (§VI-E), together with zero-shot transfer and controlled ablations (§VI-D, §VI-F).

Experimental Roadmap. We organize the empirical study around five questions: robustness to controlled corruption (RQ1), real-world forecasting and sample-wise capacity allocation (RQ2), data regimes and zero-shot transfer (RQ3), cross-task generality in classification and anomaly detection (RQ4), and ablations, efficiency, and complementarity with data selection (RQ5). Section VI gives the full protocol and reports each question in a dedicated subsection.

In summary, our contributions are summarized as follows:

• Capacity-Centric Regularization Principle: We formulate CCM to address the capacity mismatch in uniform regularization via sample-adaptive capacity allocation.

• Unified Task-Agnostic Framework: We develop SACM, a label-free, spectral-guided instantiation of CCM that adaptively modulates active capacity per sample with training only overhead and zero inference overhead.

• Cross-Task Evaluation: Across 301 dataset–backbone configurations covering forecasting, classification, and anomaly detection, SACM yields widespread empirical gains, faster convergence, and complementarity with data-centric methods.

## II. RELATED WORK

Deep Architectures for Time Series Modeling. Deep learning has advanced time series modeling across multiple tasks by capturing nonlinear temporal dependencies and learning transferable representations [27]. Early sequential architectures based on RNNs [4] and LSTMs [3] suffered from gradient degradation and limited parallelism. Transformer-based architectures overcame these bottlenecks via self-attention and structural decomposition, including Informer [8], Autoformer [28], FEDformer [29], Crossformer [30], PatchTST [9], and iTransformer [10]. In parallel, linear and MLP-based architectures such as DLinear [31], TSMixer [6], and TimeMixer [7] emerged as competitive lightweight alternatives, while diffusionbased and other specialized models have been developed for forecasting and imputation [1, 2, 11, 12]. Beyond forecasting, representation learning underpins time series classification [32– 34] and unsupervised anomaly detection [14, 35–42]. Despite these innovations, standard training pipelines apply a fixed regularization configuration across all samples, leaving models vulnerable to sample-level reliability heterogeneity.

Robustness Paradigms for Time Series. Existing methods addressing time series corruptions predominantly operate in two spaces: $I )$ Data-Centric Selection: Operating in data space $x ,$ RobustTSF [19] identifies outlier samples (each a complete input window), while Selective Learning [20] excludes highuncertainty time steps, and data-centric studies reweight or clean training data [43, 44]. Hard screening suppresses noise but can discard valid rare events and tail dynamics [13]. 2) Prior-Centric Modeling: Operating in latent space ${ \mathcal { Z } } ,$ methods such as BayesTSF [21] and RSTIB [22] leverage Variational Autoencoders [23] or Bayesian Neural Networks to constrain representations. Rigid distributional priors (e.g., Gaus sianity) struggle with non-stationary real-world distributions and incur heavy inference overhead. In parallel, frequencydomain filtering techniques such as FiLM [45], TimeFilter [46], and multi-resolution time-frequency analysis [47] design specialized network layers, but they do not regulate effective capacity conditioned on sample-level temporal reliability. Adaptive Regularization and Capacity Control. Dropout [16] is a standard stochastic regularizer, and Bayesian extensions such as Variational Dropout [48] and Concrete Dropout [18] enable learnable dropout rates across layers or parameters. These formulations optimize static, dataset-level parameters and remain sample-agnostic, sampling stochastic masks from fixed retention distributions regardless of input unreliability. SACM bridges this gap by formalizing Capacity-Centric Modulation, providing an end-to-end regularizer that adapts the model’s active capacity to each sample’s reliability across diverse time series tasks. Compared to our preliminary conference version DROPOUTTS [26], which introduced a forecasting-specific heuristic, this article formalizes CCM into a unified framework spanning forecasting, classification, and anomaly detection.

## III. SPECTRAL SPARSITY AS INDUCTIVE BIAS

## A. Problem Formulation and Capacity Allocation Dilemma

Let $\mathcal { D } = \{ ( \mathbf { x } _ { i } , \mathbf { y } _ { i } ) \} _ { i = 1 } ^ { N }$ be a time series dataset, where each sample is an input window $\mathbf { x } _ { i } \in \mathbb { R } ^ { L \times C }$ and a task-dependent target $\mathbf { y } _ { i } , \mathbf { \xi } N$ denotes the number of training samples, $L$ is the context length, $C$ is the number of channels, and $\mathcal L ( \cdot , \cdot )$ is the task-specific objective loss function; for forecasting, H denotes the prediction horizon. The target $\mathbf { y } _ { i }$ covers diverse downstream tasks: future horizons $\mathbf { y } _ { i } \in \mathbb { R } ^ { H \times C }$ in forecasting, categorical class labels $\mathbf { y } _ { i } \in \{ 1 , \ldots , K \}$ in classification, or self-reconstruction targets $\mathbf { y } _ { i } = \mathbf { x } _ { i }$ in anomaly detection. Contrastive detectors such as DCdetector [39] instead optimize agreement between input-view representations without external target labels. We write the backbone and task head as $f _ { \theta } ( \mathbf { x } ; \mathbf { m } )$ where θ contains trainable parameters and frozen ones are held fixed. The argument m collects dimensionless multipliers applied to internal activations. We abbreviate $f _ { \theta } ( \mathbf { x } ; \mathbf { 1 } )$ as $f _ { \boldsymbol { \theta } } ( \mathbf { x } )$

Conceptually, observed temporal sequences combine dominant structural dynamics with residual variations: $\mathbf x _ { i } = \mathbf x _ { i } ^ { \star } + \mathbf r _ { i }$ where $\mathbf { x } _ { i } ^ { \star }$ represents structured dynamics (e.g., physical trends, cyclic seasonality, and phase-coherent transitions), and $\mathbf { r } _ { i }$ captures residual variations such as measurement noise, local spikes, sensor glitches, missing values, and irregular fluctuations (Figure 2). Let $\sigma _ { i } = \| \mathbf { r } _ { i } \|$ denote the underlying residual magnitude. Since $\mathbf { r } _ { i }$ is unobserved, Section IV-A constructs a spectral residual score $s _ { i }$ as an empirical proxy. Standard empirical risk minimization with fixed regularization solves

![](images/ca2e13f620e9c230fd19bc383a84687937abdd5014aa67cb9f0e07b167228597.jpg)  
Fig. 2. Schematic illustration of representative synthetic temporal components and corruption profiles. Top: Structured dynamics $\mathbf { x } ^ { \star }$ from stationary periodicity to non-stationarity in mean (Trend), frequency (Chirp), and amplitude (AM). Bottom: Residual variations r representing Gaussian noise, heavy-tailed spikes, and randomly scattered missing observations.

$$
\operatorname* { m i n } _ { \theta } \sum _ { i = 1 } ^ { N } \mathcal { L } ( f _ { \theta } ( \mathbf { x } _ { i } ) , \mathbf { y } _ { i } ) + \mathcal { R } ( \theta ) ,\tag{1}
$$

where ${ \mathcal { R } } ( \theta )$ denotes an explicit regularizer (such as an $\ell _ { 2 }$ penalty), typically paired with fixed stochastic regularization (such as uniform dropout) applied identically across all training samples. When $\sigma _ { i }$ varies substantially across samples, this static configuration creates an inherent dilemma:

• Insufficient Model Regularization for Unreliable Samples: When residual variation $\mathbf { r } _ { i }$ is substantial, an insufficiently regularized model tends to fit transient, non-transferable fluctuations, degrading performance on unseen test samples.

• Excessive Model Regularization for Reliable Samples: When a sample exhibits clean, coherent dynamics $( { \bf r } _ { i } \approx { \bf 0 } )$ aggressive static regularization unnecessarily constrains the model’s expressive fidelity, impeding intricate pattern capture.

Resolving this conflict requires modulating the model’s effective capacity during training for each sample. In real-world scenarios, the ground-truth decomposition $\mathbf { r } _ { i }$ and unreliability labels are unavailable, so a label-free proxy is needed to guide sample-level regularization.

## B. Empirical Verification of Spectral Sparsity

To construct a label-free unreliability proxy, SACM exploits the empirical principle of spectral sparsity [24, 49]. This inductive bias suits structured temporal dynamics whose energy concentrates in the dominant spectral modes. In the tested settings, stochastic perturbations, localized spikes, and missingvalue patterns introduce residual components less concentrated than the dominant structure, motivating our spectral scorer.

![](images/49b99f27776853cf6c62ee34c9d1325633a2efcaa68d127c9e4affb64275ba28.jpg)  
Fig. 3. Illustration of Spectral Sparsity. Composite (Periodic + Trend + Chirp + AM) under corruption. Left: Corrupted inputs. Middle: Frequency spectra: structured dynamics concentrate in discrete peaks while corruptions often spread energy across more frequencies. Right: Reconstruction from dominant modes faithfully recovers the underlying ground-truth signal.

By Fourier linearity, the spectrum of an observed sequence satisfies $\mathcal { F } ( \mathbf { x } ) = \mathcal { F } ( \mathbf { x } ^ { \star } ) + \mathcal { F } ( \mathbf { r } )$ . When the energy of $\mathcal { F } ( \mathbf { x } ^ { \star } )$ is sparse, retaining the dominant spectral modes while filtering out non-dominant frequencies reconstructs the underlying structure $\mathbf { x } ^ { \star }$ while capturing nuisance components in the residual. We empirically validate this across varied temporal regimes:

• Evidence across Synthetic and Real-World Data. As demonstrated in Figure 3, retaining only the top 1% dominant Fourier coefficients with the highest energy (τ at the $9 9 ^ { \mathrm { t h } }$ percentile) reconstructs composite temporal signals with high fidelity across diverse synthetic corruptions; similarly, top-10% spectral thresholding across extensive real-world bench marks consistently preserves underlying macro dynamics (detailed in the Supplementary Material).

• Spectral Flatness Measure (SFM) as a Concentration Anchor. To quantify spectral dispersion, we use the SFM, defined as the ratio of the geometric to arithmetic mean of the power spectrum. SFM approaches 1 for flat spectra and decreases as spectral energy becomes concentrated. Figure 4 shows a strong negative correlation $( r = - 0 . 8 4 6 , p < 0 . 0 5 )$ between SFM and a proxy SNR (defined as the ratio of spectral energy in the top 10% highest-amplitude Fourier coefficients to that in the remaining coefficients), supporting SFM as an anchor for automated threshold calibration.

Label-Free Spectral Residual Scorer. Formally, under the working assumption that temporal structure is spectrally sparse, the dominant component xˆ is approximated by spectral filtering:

$$
\hat { \mathbf { x } } = \mathcal { F } ^ { - 1 } ( \mathbf { M } \odot \mathcal { F } ( \mathbf { x } ) ) , \quad \mathbf { M } _ { k } = \mathbb { I } ( | \mathcal { F } ( \mathbf { x } ) | _ { k } > \tau ) ,\tag{2}
$$

where $\mathcal { F }$ and ${ \mathcal { F } } ^ { - 1 }$ denote the Discrete Fourier Transform (DFT) and its inverse, $\odot$ the Hadamard product, and ${ { \bf { M } } _ { k } }$ selects modes above $\tau .$ The residual $\lVert \mathbf { x } - \hat { \mathbf { x } } \rVert _ { 2 }$ measures the energy outside the retained components, motivating the differentiable soft-mask scorer and mean absolute residual in Section $\mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi } \mathrm { \Pi \Pi } \mathrm { \Pi } \mathrm { \Pi \Pi } \mathrm { \Pi \Pi } \mathrm  \Pi \mathrm { \Pi } \mathrm \Pi \mathrm { \Pi } \mathrm { \Pi \Pi } \mathrm \mathrm { \Pi \Pi \Pi } \mathrm \mathrm  \Pi \Pi \Pi \Pi \mathrm \Pi \mathrm { } \mathrm \Pi \Pi \mathrm \Pi \mathrm  \Pi \Pi \Pi \Pi \Pi \Pi \Pi \mathrm \Omega \mathrm \Omega \Omega \mathrm \Omega \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m \ m a m \ m \ m \ m a m a m a m a m \ m a m \ m a m m m \ m a m m m \ m a m m m m \ m a m m m m m a m m m m \ m a m m m m m a m m m m m m a m m$

## IV. THE SACM FRAMEWORK

Figure 5 illustrates SACM’s dual-path architecture. The upper path represents the backbone and task head (such as forecasting projection, classification head, or reconstruction decoder). The lower path executes capacity modulation:

1) Spectral Residual Scorer: Detrends the input, maps it to the frequency domain, applies an SFM-anchored soft spectral filter, and reconstructs the dominant component. The reconstruction error between the original and reconstructed sequences yields a scalar spectral residual score $s _ { i } .$

2) Sample-Adaptive Capacity Mapper: Normalizes residual scores across the mini-batch and maps them to bounded dropout probabilities $p _ { i }$ . Differentiable binary activation masks are generated via the Straight-Through Estimator (STE) [50] to modulate active feature dimensions along trainable paths during forward passes.

Gradients from the primary task loss $\mathcal { L }$ flow end-to-end through the STE path to optimize the scorer and mapper parameters jointly with the backbone. During evaluation, the capacity modulation path is deactivated, ensuring deterministic inference with no additional scorer or mapper computation or latency.

## A. Spectral Residual Scorer

The differentiable spectral residual scorer computes a samplelevel unreliability proxy through a four-stage pipeline:

1. Global Linear Detrending. Non-stationary trends in finite temporal windows introduce sharp boundary discontinuities when computing the DFT, causing severe spectral leakage that smears energy across frequency bins. To suppress boundary artifacts, for an input sample $\mathbf { x } \in \mathbb { R } ^ { L \times C }$ , we remove linear trends via Ordinary Least Squares (OLS):

$$
\begin{array} { r } { \mathbf { x } _ { \mathrm { d e t r e n d } } = \mathbf { x } - \mathbf { x } _ { \mathrm { t r e n d } } , \quad \mathrm { w h e r e } ~ \mathbf { x } _ { \mathrm { t r e n d } } = \mathbf { T } \mathbf { w } ^ { * } + \mathbf { b } ^ { * } , } \end{array}\tag{3}
$$

where $\mathbf { T } = [ 0 , 1 , \ldots , L - 1 ] ^ { \top } \in \mathbb { R } ^ { L \times 1 }$ is the temporal index vector. The slope $\mathbf { w } ^ { * } \in \bar { \mathbb { R } } ^ { 1 \times C }$ and intercept $\mathbf { b } ^ { \mathbf { \bar { * } } } \in \mathbb { R } ^ { 1 \times C }$ are obtained independently for each channel by minimizing $\begin{array} { r l } { ~ } & { { } \sum _ { t = 1 } ^ { L } ( x _ { t , c } - ( t - \dot { 1 } ) w _ { c } - b _ { c } ) ^ { 2 } } \end{array}$ . This detrending isolates baseline trends and reduces spectral leakage.

2. Log-Scale Spectral Normalisation. We compute the DFT of $\mathbf { x } _ { \mathrm { d e t r e n d } }$ via a real-input fast Fourier transform (FFT), implemented with rFFT. For a real-valued input, conjugate symmetry makes the negative-frequency half redundant, so rFFT returns $K _ { f } = \lfloor L / 2 \rfloor + 1$ coefficients $\mathbf { Z } \in \mathbb { C } ^ { K _ { f } \times C }$ and the amplitude spectrum $\mathbf { A } = | \mathbf { Z } |$ . Because spectral amplitudes often span multiple orders of magnitude, we compute the logamplitude $\mathbf { L } = \log ( 1 + \mathbf { A } )$ and perform min-max normalisation across frequency bins for each channel:

![](images/b2fcdc753ca75fc90927a997a67840c13b00f685e4bcf25f25c8de2e9ef86f98.jpg)

![](images/99339ac46dcf6759fa8f3544cdb6e084d497d55355baae5385be9693281b82c1.jpg)

![](images/8de6993a5f8e0981b5bb5b7a8a6e4db9420ad75e47591b9d9cfcdacef58a3872.jpg)  
Fig. 4. Spectral proxy analysis across real-world benchmarks. (a) Proxy SNR, defined as the ratio of spectral energy in the top 10% highest-amplitude Fourier coefficients to that in the remaining coefficients; (b) strong negative correlation between SFM and proxy SNR $( r \bar { = } - 0 . 8 4 6 , p < 0 . 0 5 )$ ; (c) dataset rankings illustrating heterogeneous spectral concentration.

![](images/c6d94d832f9b116429a2a46e06a1544a0090982a301f3e12946aac0619b783f4.jpg)  
Fig. 5. Unified cross-task architecture of SACM across forecasting, classification, and anomaly detection. The spectral scorer extracts spectral residual scores via detrending, log-scale normalization, and SFM-anchored soft filtering; the rate mapper maps scores to bounded dropout rates and differentiable activation masks. As a task-agnostic framework, SACM applies sample-adaptive regularization during training while leaving the inference pipeline unchanged.

$$
\hat { A } _ { k , c } = \frac { L _ { k , c } - \operatorname* { m i n } _ { j } L _ { j , c } } { \operatorname* { m a x } \left( \operatorname* { m a x } _ { j } L _ { j , c } - \operatorname* { m i n } _ { j } L _ { j , c } , \epsilon \right) } ,\tag{4}
$$

where $k \in \{ 1 , \ldots , K _ { f } \}$ indexes frequency bins and $c \in$ $\{ 1 , \ldots , C \}$ channels, and $\epsilon = 1 0 ^ { - 8 }$ prevents division by zero. 3. Learnable SFM-Anchored Spectral Filter. To separate dominant spectral modes from diffuse residual energy without relying on hard, hand-tuned frequency cutoffs, we design a learnable soft filter anchored by SFM. For channel $c ,$ we define the spectral power proxy $P _ { k , c } = L _ { k , c } ^ { 2 } + \epsilon$ and compute:

$$
\mathrm { S F M } _ { c } = \frac { \exp \left( \frac { 1 } { K _ { f } } \sum _ { k = 1 } ^ { K _ { f } } \ln P _ { k , c } \right) } { \frac { 1 } { K _ { f } } \sum _ { k = 1 } ^ { K _ { f } } P _ { k , c } + \epsilon } .\tag{5}
$$

An SFM<sub>c</sub> value close to 1 signifies a flat, noise-like power spectrum, whereas smaller values indicate sharp spectral concentration. We parameterize a channel-adaptive threshold $\tau _ { c }$ as a learnable affine transformation of SFM :

$$
\tau _ { c } = \mathrm { s i g m o i d } \left( w _ { s , c } \cdot \mathrm { S F M } _ { c } + b _ { s , c } \right) ,\tag{6}
$$

where $w _ { s , c }$ and $b _ { s , c }$ are trainable parameters. The continuous spectral retention mask $\mathbf { M } \in [ 0 , 1 ] ^ { \overline { { K } } _ { f } \times C }$ is then computed as:

$$
M _ { k , c } = mathrm { \ s i g m o i d } \left( { \mathrm { s o f t p l u s } } ( \alpha ) \cdot ( \hat { A } _ { k , c } - \tau _ { c } ) \right) ,\tag{7}
$$

where the learnable scalar α controls transition steepness via a strictly positive softplus transform. This soft mask smoothly attenuates non-dominant frequency components while preserving dominant structural harmonics.

4. Residual Scoring. We reconstruct the dominant temporal signal by performing the inverse real-input FFT $( \mathcal { F } ^ { - 1 } )$ on the filtered spectrum and restoring the previously extracted trend:

$$
\mathbf { x } ^ { \prime } = \mathcal { F } ^ { - 1 } ( \mathbf { Z } \odot \mathbf { M } ) + \mathbf { x } _ { \mathrm { t r e n d } } .\tag{8}
$$

We first compute a channel-level residual score and then aggregate it into one sample-level score:

$$
s _ { c } = \frac { 1 } { L } \sum _ { t = 1 } ^ { L } | x _ { t , c } - x _ { t , c } ^ { \prime } | , \qquad s = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } s _ { c } .\tag{9}
$$

The mean s aggregates channel-wise residuals into a label-free proxy for sample unreliability. It yields a shared dropout rate across adaptive layers, allowing sample-level modulation of joint multivariate representations without requiring a correspondence between input channels and hidden features.

## B. Sample-Adaptive Capacity Modulation

Given sample residual scores $\{ s _ { i } \} _ { i = 1 } ^ { | B | }$ in mini-batch $B ,$ the second module dynamically maps s<sub>i</sub> to a sample-specific dropout rate $p _ { i } ,$ , modulating the backbone’s active capacity. Batch-Aware Relative Mapping. Because raw residual magnitudes vary across datasets and mini-batches, we normalize scores relative to the current mini-batch distribution:

$$
\hat { s } _ { i } = \left\{ \begin{array} { l l } { \frac { s _ { i } - \operatorname* { m i n } _ { j } s _ { j } } { \operatorname* { m a x } _ { j } s _ { j } - \operatorname* { m i n } _ { j } s _ { j } } , } & { \mathrm { i f } \operatorname* { m a x } _ { j } s _ { j } - \operatorname* { m i n } _ { j } s _ { j } > \epsilon , } \\ { 0 . 5 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{10}
$$

If score dispersion within a batch falls below ϵ, assigning $\hat { s } _ { i } =$ 0.5 provides a neutral default. We then map score $\hat { s } _ { i } \in [ 0 , 1 ]$ to a dropout probability $p _ { i } \in [ p _ { \operatorname* { m i n } } , p _ { \operatorname* { m a x } } ]$ through a learnable nonlinear sensitivity curve:

$$
\tilde { s } _ { i } = \operatorname { t a n h } \left( \hat { s } _ { i } \cdot \operatorname { s o f t p l u s } ( \gamma ) \right) ,\tag{11}
$$

$$
p _ { i } = p _ { \mathrm { m i n } } + \left( p _ { \mathrm { m a x } } - p _ { \mathrm { m i n } } \right) \cdot \tilde { s } _ { i } ,\tag{12}
$$

where the shared trainable scalar $\gamma$ governs sensitivity to relative unreliability via $a = \mathrm { s o f t p l u s } ( \gamma ) > 0$ . Lower-score samples $( \hat { s } _ { i } \to 0 )$ receive a base dropout rate $p _ { \mathrm { m i n } }$ to retain more active capacity, while higher-score samples $( \hat { s } _ { i } \to 1 )$ receive elevated rates up to $p _ { \operatorname* { m i n } } + ( p _ { \operatorname* { m a x } } - p _ { \operatorname* { m i n } } ) \operatorname { t a n h } ( a ) \leq p _ { \operatorname* { m a x } }$ to restrict capacity and suppress noise memorization.

Differentiable Mask Generation via STE. Stochastic dropout mask sampling is discrete and non-differentiable. To enable end-to-end backpropagation from the task loss $\mathcal { L }$ to the scorer parameters $( \alpha , \mathbf { w } _ { s } , \mathbf { b } _ { s } , \gamma )$ , we employ STE [50]. For sample $i ,$ the forward pass samples a binary mask $\mathbf { B } _ { i } \sim \mathrm { B e r n o u l l i } ( 1 -$ $p _ { i } )$ . A larger $p _ { i }$ zeros out more dimensions, while surviving activations are scaled by inverted-dropout normalization. The differentiable activation mask $\mathbf { M } _ { \mathrm { d r o p } , i }$ is defined as:

$$
{ \bf M } _ { \mathrm { d r o p } , i } = { \bf B } _ { i } + q _ { i } - \mathrm { s g } ( q _ { i } ) , \qquad q _ { i } = 1 - p _ { i } ,\tag{13}
$$

where $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator. In the forward pass, $\mathbf { M } _ { \mathrm { d r o p } , i } = \mathbf { B } _ { i } ,$ while in the backward pass, its surrogate gradient satisfies $\partial \mathbf { M } _ { \mathrm { d r o p } , i } / \partial p _ { i } = - 1$ . For an activation tensor $\mathbf { H } _ { i }$ , inverted dropout scaling produces:

$$
\mathbf { H } _ { i , \mathrm { o u t } } = \mathbf { H } _ { i } \odot \mathbf { m } _ { i } , \qquad \mathbf { m } _ { i } = { \frac { \mathbf { M } _ { \mathrm { d r o p } , i } } { q _ { i } } } .\tag{14}
$$

This operation directly modulates sample-wise active capacity during training while preserving the unbiased conditional expectation of feature representations $( \mathbb { E } [ \mathbf { H } _ { i , \mathrm { o u t } } \mid \mathbf { H } _ { i } ] = \mathbf { H } _ { i } )$ Task-Agnostic End-to-End Optimization. The parameter count added by $\mathbf { S A C M }$ is minimal $( 2 C + 2$ scalar parameters for C channels: $\displaystyle w _ { s , c } , b _ { s , c } , \alpha , \gamma )$ . During training, the scorer and mapper parameters $\phi = \{ \alpha , \mathbf { w } _ { s } , \mathbf { b } _ { s } , \gamma \}$ are optimized jointly with the trainable backbone parameters θ:

$$
\operatorname* { m i n } _ { \theta , \phi } \ \mathbb { E } _ { B } \left[ \frac { 1 } { | \mathcal { B } | } \sum _ { i \in \mathcal { B } } \mathbb { E } _ { \mathbf { B } _ { i } } \left[ \mathcal { L } \big ( f _ { \theta } ( \mathbf { x } _ { i } ; \mathbf { m } _ { i } ) , \mathbf { y } _ { i } \big ) \right] \right] .\tag{15}
$$

Here, m collects the activation multipliers across all adaptive dropout sites for sample i, with rates shared across sites and Bernoulli masks drawn independently. The outer expectation reflects mini-batch sampling, as relative unreliability scores are evaluated across B. Gradients backpropagate end-to-end through the STE to update scorer parameters $\phi$ alongside backbone parameters θ. Because SACM acts strictly on internal activation paths without requiring auxiliary reliability labels, it seamlessly integrates with any differentiable task loss ${ \mathcal { L } } .$

## C. Inference Procedure

During inference, the capacity modulation path is completely bypassed: no spectral decomposition, residual scoring, or rate mapping is executed. The backbone and task heads operate deterministically with all dropout operations disabled $( \mathbf { m } _ { i } \equiv \mathbf { 1 } )$ Consequently, SACM preserves the deterministic inference graph of the underlying backbone, introducing zero additional parameters, memory overhead, or latency at test time.

## V. THEORETICAL CAPACITY ANALYSIS

We relate sample-adaptive dropout to local regularization, then analyze how residual heterogeneity affects capacity allocation under a bias–variance surrogate (Appendix E).

## A. From Dynamic Dropout to Sample-Adaptive Regularization

Fix the backbone parameters, sample $i ,$ and its mini-batch. Analyze one adaptive layer with all other activation multipliers set to one. Let $\mathbf { h } _ { i } \in \mathbb { R } ^ { d }$ be its pre-mask activations, vectorised here, where d is the number of coordinates and $j \in \{ 1 , \ldots , d \}$ indexes them. Conditional on the sample and mini-batch, the retention rate is $q _ { i } = 1 - p _ { i }$ , and independent mask entries satisfy $B _ { i j } \sim$ Bernoulli(q ). Following Sections III and $\mathrm { I V - B } ,$ $\mathbf { m } _ { i }$ collects the multipliers $\mathbf { B } _ { i } / q _ { i }$ at this layer and ones elsewhere. Thus $f _ { \theta } ( \mathbf { x } _ { i } ; \mathbf { m } _ { i } )$ is the stochastic forward mapping, whereas $f _ { \theta } ( \mathbf { x } _ { i } ) = f _ { \theta } ( \mathbf { x } _ { i } ; \mathbf { 1 } )$ is deterministic inference.

The inverted multiplier has mean one and variance

$$
\mathbb { E } [ B _ { i j } / q _ { i } ] = 1 , \qquad \mathrm { V a r } ( B _ { i j } / q _ { i } ) = \frac { p _ { i } } { 1 - p _ { i } } = : \lambda ( p _ { i } ) .\tag{16}
$$

Consequently, the activation perturbation $\pmb { \delta } _ { i } = \mathbf { h } _ { i } \odot ( \mathbf { B } _ { i } / q _ { i } - \mathbf { 1 } )$ has zero mean and covariance $\lambda ( p _ { i } )$ diag $( h _ { i 1 } ^ { 2 } , \ldots , h _ { i d } ^ { 2 } )$ . Let $\ell _ { i } ( \mathbf { h } )$ be the downstream task loss with the target fixed, so $\ell _ { i } ( { \bf h } _ { i } ) = \mathcal { L } ( f _ { \theta } ( { \bf x } _ { i } ) , { \bf y } _ { i } )$ . All mask expectations and variances in this subsection are conditional on the fixed input and mini-batch. A second-order Taylor expansion of $\ell _ { i }$ around $\mathbf { h } _ { i }$ gives [17]

$$
\begin{array} { r } { \mathbb { E } _ { \mathbf { B } _ { i } } \left[ \mathcal { L } \big ( f _ { \theta } ( \mathbf { x } _ { i } ; \mathbf { m } _ { i } ) , \mathbf { y } _ { i } \big ) \right] \approx \mathcal { L } \big ( f _ { \theta } ( \mathbf { x } _ { i } ) , \mathbf { y } _ { i } \big ) + \frac { 1 } { 2 } \lambda ( p _ { i } ) \Omega ( \mathbf { x } _ { i } ; \theta ) , } \end{array}
$$

$$
\Omega ( \mathbf { x } _ { i } ; \theta ) = \sum _ { j = 1 } ^ { d } h _ { i j } ^ { 2 } \frac { \partial ^ { 2 } \ell _ { i } } { \partial h _ { j } ^ { 2 } } ( \mathbf { h } _ { i } ) .\tag{17}
$$

The zero-mean perturbation eliminates the first-order term. Ω is the activation-weighted loss curvature. When $\Omega \geq 0 .$ , the correction penalizes sensitivity to activation perturbations with coefficient $\lambda ( p _ { i } ) / 2$ , which increases with $p _ { i }$ because $\mathrm { d } \lambda / \mathrm { d } p =$ $( 1 - p ) ^ { - 2 } > 0 .$ . The approximation requires a smooth loss and a small Taylor remainder (Appendix E). For fixed mapper parameters, Equation (12) assigns larger $p _ { i }$ and hence larger $\lambda ( p _ { i } )$ to higher normalized residual scores $\hat { s } _ { i }$ within a minibatch. Define the layer’s expected active width as

$$
\mathcal { C } _ { i } ^ { \mathrm { t r a i n } } : = \mathbb { E } \left[ \sum _ { j = 1 } ^ { d } B _ { i j } \right] = ( 1 - p _ { i } ) d .\tag{18}
$$

This counts retained activation coordinates before inverted scaling, with no change in parameter count. Equivalently,

TABLE II  
DEFINITIONS OF CLEAN AND CORRUPTED SIGNALS. OVERVIEW OF THE FOUR CLEAN REGIMES AND THREE CORRUPTION PROFILES $( \tilde { x } _ { t } )$
<table><tr><td>Type</td><td>Category</td><td>Mathematical Formulation</td><td>Physical Interpretation</td><td>Example</td></tr><tr><td rowspan="3"> $\sum _ { i = 1 } ^ { \overline { { \sum } } }$ </td><td>Stationary (Periodic)</td><td> $\begin{array} { r } { x _ { t } = \sum _ { k } A _ { k } \sin ( 2 \pi f _ { k } t + \phi _ { k } ) } \end{array}$ </td><td>Stable equilibrium; constant freq. &amp; amp.</td><td>Power grid voltage</td></tr><tr><td>Non-stat. (Mean)</td><td> $\begin{array} { r } { x _ { t } = \alpha t + \beta + \sum _ { k } A _ { k } \sin ( 2 \pi f _ { k } t ) } \end{array}$ </td><td>Trend &amp; Seasonality; drifts with fluctuations</td><td>Macroeconomic growth (GDP)</td></tr><tr><td>Non-stat. (Freq.)</td><td> $\begin{array} { r } { x _ { t } = A \sin ( 2 \pi ( f _ { 0 } t + \frac { 1 } { 2 } k t ^ { 2 } ) ) } \end{array}$ </td><td>Spectral Drift; time-varying frequency</td><td>Doppler effects (Radar/Sonar)</td></tr><tr><td rowspan="3"></td><td>Non-stat. (Var.)</td><td> $x _ { t } = ( 1 + \mu \sin ( 2 \pi f _ { m } t ) ) \sin ( 2 \pi f _ { c } t )$ </td><td>Time-varying amplitude envelope</td><td>Vibration or communication signals</td></tr><tr><td>Gaussian Noise</td><td> $\tilde { x } _ { t } = x _ { t } + \epsilon _ { t } , \epsilon _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ </td><td>Random measurement noise</td><td>Sensor thermal noise</td></tr><tr><td>Heavy-tail (Student-t)</td><td> $\tilde { x } _ { t } = x _ { t } + \epsilon _ { t } , \epsilon _ { t } \sim t _ { \nu } ( \nu = 2 . 5 )$ </td><td>Heavy-tailed random corruption</td><td>Impulsive sensor disturbances</td></tr><tr><td>Noise</td><td>Missing Values</td><td> $\tilde { x } _ { t } = x _ { t } \odot m _ { t } , \ m _ { t } \sim \mathcal { B } ( 1 - p )$ </td><td>Observation Failures; random data loss</td><td>Wireless packet loss</td></tr></table>

$\mathcal { C } _ { i } ^ { \mathrm { t r a i n } } = d / ( 1 + \lambda ( p _ { i } ) )$ , so a larger coefficient yields a smaller active width. At inference, all activation multipliers are one and the scorer and mapper are bypassed.

## B. A Bias–Variance Surrogate for Capacity Allocation

For dropout rate $p ,$ write $\lambda ~ = ~ \lambda ( p ) ~ = ~ p / ( 1 - p )$ for the corresponding regularization strength. The sample-specific value is $\lambda ( p _ { i } )$ . The bounds $0 ~ \le ~ p _ { \mathrm { m i n } } ~ < ~ p _ { \mathrm { m a x } } ~ < ~ 1$ define $\Lambda = [ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ]$ , with $\lambda _ { \operatorname* { m i n } } = p _ { \operatorname* { m i n } } / ( 1 - p _ { \operatorname* { m i n } } )$ and $\lambda _ { \operatorname* { m a x } } = p _ { \operatorname* { m a x } } / ( 1 - p _ { \operatorname* { m a x } } )$ . Let $\sigma ( \mathbf { x } ) \geq 0$ denote the residual magnitude, with $\sigma ( \mathbf { x } _ { i } ) = \sigma _ { i } = \| \mathbf { r } _ { i } \|$ as defined in Section III. The learned mapper may realize a narrower range within Λ.

We make two assumptions on a scalar prediction error. First, a linear bias response bλ models regularization-induced systematic error, with zero baseline bias and shared slope $b \neq 0 .$ . This first-order model gives squared bias $C _ { 1 } \lambda ^ { 2 }$ , where $C _ { 1 } = b ^ { 2 } > 0$ . Second, model the fitted residual response by a scalar quadratic penalty. Let ε be a random perturbation with $\mathbb { E } [ \varepsilon \mid \sigma ] = 0$ and $\operatorname { V a r } ( \varepsilon \mid \sigma ) = 1$ . For residual input $\sigma \varepsilon .$ , define the fitted scalar response $v _ { \lambda }$ by

$$
v _ { \lambda } = \arg \operatorname* { m i n } _ { v } \left\{ { \frac { 1 } { 2 } } ( v - \sigma \varepsilon ) ^ { 2 } + { \frac { \lambda } { 2 } } v ^ { 2 } \right\} = { \frac { \sigma \varepsilon } { 1 + \lambda } }\tag{19}
$$

The stationarity condition $( v - \sigma \varepsilon ) + \lambda v = 0$ gives the displayed solution. With a fixed output scale ${ \sqrt { C _ { 2 } } } ,$ , where $C _ { 2 } > 0$ , its conditional variance is $C _ { 2 } \operatorname { V a r } ( v _ { \lambda } \mid \sigma ) = C _ { 2 } \sigma ^ { 2 } / ( 1 + \lambda ) ^ { 2 }$ . This assumed fitted-response variance decreases with λ, whereas the forward mask variance in Equation (16) increases. The two describe distinct effects.

Combining the assumed bias and residual response gives $e _ { \lambda } = b \lambda + \sqrt { C _ { 2 } } v _ { \lambda }$ . Its conditional mean squared error is

$$
\mathcal { E } ( \lambda , \sigma ) : = \mathbb { E } [ e _ { \lambda } ^ { 2 } \mid \sigma ] = C _ { 1 } \lambda ^ { 2 } + C _ { 2 } \frac { \sigma ^ { 2 } } { ( 1 + \lambda ) ^ { 2 } } .\tag{20}
$$

The zero conditional mean of $v _ { \lambda }$ removes the cross term, leaving squared bias and variance. Shared constants $C _ { 1 } , C _ { 2 }$ isolate residual magnitude as the source of heterogeneity.

The surrogate is strictly convex for $\lambda \geq 0 ,$ , since $\partial _ { \mathrm { { \bar { \lambda } } } } ^ { 2 } { \mathcal { E } } =$ $2 C _ { 1 } + 6 C _ { 2 } \sigma ^ { \bar { 2 } } / ( 1 + \lambda ) ^ { 4 } > ^ { } 0$ . Its minimizer $\lambda ^ { \star } ( \sigma )$ over $[ \dot { 0 } , \infty )$ satisfies $C _ { 1 } \lambda ^ { \star } ( 1 + \lambda ^ { \star } ) ^ { 3 } = C _ { 2 } \sigma ^ { 2 }$ . The left side increases strictly with $\lambda ^ { \star }$ , so the minimizer increases with σ. The constrained oracle allocation is $g ( \sigma ) ~ = ~ \lambda _ { \Lambda } ^ { \star } ( \sigma ) ~ = ~ \Pi _ { \Lambda } ( \lambda ^ { \star } ( \sigma ) )$ , where $\Pi _ { \Lambda } ( u ) = \operatorname* { m i n } \{ \lambda _ { \operatorname* { m a x } } , \operatorname* { m a x } \{ \lambda _ { \operatorname* { m i n } } , \overset { \cdot } { u } \} \}$ clips a scalar to Λ.

Theorem V.1 (Excess Surrogate Risk of Uniform Capacity Allo cation). Let X be a random input window from the population under study. Under Equation (20), assume $\mathbb { E } [ \sigma ( \mathbf { X } ) ^ { 2 } ] < \infty$ and Va $\mathrm { \Delta } \mathrm { r } [ g ( \sigma ( { \bf X } ) ) ] > 0 .$ . Every sample-independent strength $\lambda _ { \mathrm { f i x } } \in \Lambda$ incurs strictly positive expected excess surrogate risk relative to the oracle allocation:

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbf { X } } [ \mathcal { E } ( \lambda _ { \mathrm { f i x } } , \sigma ( \mathbf { X } ) ) - \mathcal { E } ( g ( \sigma ( \mathbf { { \sigma } } \mathbf { X } ) ) , \sigma ( \mathbf { X } ) ) ] } \\ & { \quad \quad \quad \geq C _ { 1 } \operatorname { V a r } [ g ( \sigma ( \mathbf { X } ) ) ] > 0 . } \end{array}\tag{21}
$$

Proof sketch. The bound $\partial _ { \lambda } ^ { 2 } \mathcal { E } \geq 2 C _ { 1 }$ gives strong convexity. At the minimizer g, the first-order optimality condition yields

$$
\mathcal { E } ( \lambda , \sigma ) - \mathcal { E } ( g ( \sigma ) , \sigma ) \geq C _ { 1 } ( \lambda - g ( \sigma ) ) ^ { 2 } .
$$

Taking expectations with $\mathbb { E } [ ( \lambda _ { \mathrm { f i x } } - g ) ^ { 2 } ] \ge \mathrm { V a r } ( g )$ and $g =$ $g ( \sigma ( \mathbf { X } ) )$ proves it. Appendix E covers boundary optima.

Theorem V.1 assumes distinct oracle rates after clipping, excluding constant allocation at a shared bound. Under this model, larger residual magnitudes favor stronger regularization and smaller expected active widths. This surrogate result motivates SACM’s monotone mapper, which uses spectral residual scores to guide sample-wise regularization.

## VI. EXPERIMENTS

To validate Capacity-Centric Modulation and SACM, our experiments address five research questions (RQs):

• RQ1 (Corruption Robustness): Does SACM reduce forecasting error across corruption scales? (§VI-B)

• RQ2 (Real-World Forecasting): Does SACM improve forecasting accuracy on public benchmarks? (§VI-C)

• RQ3 (Generalization): Does sample-adaptive capacity improve transfer across data regimes and domains? (§VI-D)

• RQ4 (Cross-Task Generality): Does SACM generalize to both classification and anomaly detection tasks? (§VI-E)

• RQ5 (Ablations & Efficiency): Do modules help, is SACM efficient, and does it complement data selection? (§VI-F)

## A. Experimental Settings

Datasets & Benchmark Coverage. Our benchmarks span multiple time series regimes. For predictive modeling (forecasting), we incorporate nine real-world benchmarks: ETTh1/ETTh2, ETTm1/ETTm2 [8], Electricity, Exchange Rate [4], Weather, ILI [28], and ECG waveforms from PhysioNet [53]. To evaluate robustness under controlled reliability heterogeneity, we introduce Synth-12, which combines structured temporal signals with layered corruptions at five scales (Tables II and IV). For discriminative modeling (classification), we evaluate 30 UEA datasets [54], HAR [55], and Sleep-EDF [56]. For reconstructive modeling (anomaly detection), we evaluate SMD [36], MSL/SMAP [35], and PSM [37]. Benchmark statistics and protocols are provided in the Supplementary Material.

TABLE IV  
SYNTH-12 CORRUPTION GRADIENT STATISTICS ACROSS NOISE LEVELS σ ∈ {0.1, . . . , 0.9} (N = 33, 600, CLEAN SFM = 0.003).
<table><tr><td>σ</td><td>SNR (dB)</td><td>SFM</td><td>∆SFM</td><td>MSE</td><td>Stress Regime</td></tr><tr><td>0.1</td><td>23.77</td><td>0.008</td><td>0.005</td><td>0.004</td><td>Mild</td></tr><tr><td>0.3</td><td>16.57</td><td>0.023</td><td>0.020</td><td>0.022</td><td>Moderate</td></tr><tr><td>0.5</td><td>12.39</td><td>0.046</td><td>0.043</td><td>0.058</td><td>High</td></tr><tr><td>0.7</td><td>9.54</td><td>0.076</td><td>0.073</td><td>0.111</td><td>Severe</td></tr><tr><td>0.9</td><td>7.39</td><td>0.109</td><td>0.106</td><td>0.182</td><td>Extreme</td></tr></table>

Backbone Architectures. We evaluate SACM across diverse deep temporal architectures. Forecasting backbones include Transformer-based (Informer [8], Crossformer [30], PatchTST [9], iTransformer [10], MultiPatchFormer [52]), MLP-based (TimeMixer [7], WPMixer [51]), CNN-based (TimesNet [5]), and graph-based (TimeFilter [46]). Classification backbones include iTransformer, PatchTST, NSFormer [57], TimesNet, InceptionTime [32], and MiniROCKET [33]. Anomaly detection backbones include PatchTST, iTransformer, NSFormer, AnomTrans [37], TranAD [38], DCdetector [39], and TimesNet.

TABLE III  
FORECASTING RESULTS ON SYNTH-12 (L = 96, AVERAGED ACROSS H ∈ {96, 192, 336, 720}). +SACM DENOTES BACKBONES AUGMENTED WITH SACM; BOLD MARKS THE LOWER ERROR PER PAIR. AVG. GAIN MACRO-AVERAGES RELATIVE IMPROVEMENTS ACROSS NOISE LEVELS σ $\in [ 0 . 1 , 0 . 9 ] .$
<table><tr><td rowspan="2">σ Metric</td><td colspan="2">Informer (2021)</td><td colspan="2">Crossformer (2023)</td><td colspan="2">PatchTST (2023)</td><td colspan="2">TimesNet (2023)</td><td colspan="2">iTransformer (2024) TimeMixer (2024)</td><td colspan="2">WPMixer (2025)</td><td colspan="2">TimeFilter (2025) MultiPatchFormer (2025)</td></tr><tr><td>Raw</td><td></td><td>+SACM Raw</td><td> $+ { \cal S } \mathbf { A } \mathbf { C } \mathbf { M }$ </td><td>Raw</td><td>+SACM Raw</td><td>+SACM</td><td>Raw +SACM</td><td>Raw</td><td> $+ { \bf S } { \bf A } { \bf C } { \bf M }$  Raw</td><td>+SACM Raw</td><td>+SACM</td><td>Raw</td><td>+SACM</td></tr><tr><td rowspan="2">0.1</td><td>MSE 0.966 MAE</td><td> $\mathbf { 0 . 5 1 4 } _ { ( \uparrow 4 6 . 8 \% ) }$ </td><td>0.450</td><td> $\mathbf { 0 . 3 8 6 } _ { ( \uparrow 1 4 . 2 \% ) }$ </td><td>0.542</td><td> $\mathbf { 0 . 5 2 9 } _ { ( \uparrow 2 . 4 \% ) }$  0.902</td><td> $\mathbf { 0 . 8 4 8 } _ { ( \uparrow 6 . 0 \% ) }$ </td><td>0.523  $\mathbf { 0 . 5 1 9 } _ { ( \uparrow 0 . 8 \% ) }$ </td><td>0.545</td><td> $\mathbf { 0 . 5 4 0 } _ { ( \uparrow 0 . 9 \% ) }$  0.539</td><td> $\mathbf { 0 . 5 1 9 } _ { ( \uparrow 3 . 7 \% ) }$ </td><td>0.543  $\mathbf { 0 . 5 3 0 } _ { ( \uparrow 2 . 4 \% ) }$ </td><td>0.547</td><td> $\mathbf { 0 . 5 2 3 } _ { ( \uparrow 4 . 4 \% ) }$ </td></tr><tr><td>0.775</td><td> $\mathbf { 0 . 5 8 1 } _ { ( \uparrow 2 5 . 0 \% ) }$ </td><td>0.502 0.439</td><td> $\mathbf { 0 . 4 7 1 } _ { ( \uparrow 6 . 2 \% ) }$ </td><td>0.561 0.554(↑1.2%)</td><td>0.763</td><td> $\mathbf { 0 . 7 3 9 } _ { ( \uparrow 3 . 1 \% ) }$  0.544</td><td> $\mathbf { 0 . 5 4 1 } _ { ( \uparrow 0 . 6 \% ) }$ </td><td>0.555  $\mathbf { 0 . 5 5 4 } _ { ( \uparrow 0 . 2 \% ) }$ </td><td>0.552</td><td>0.539 (+2.4%) 0.558</td><td>0.550(↑1.4%)</td><td>0.558</td><td> $\mathbf { 0 . 5 4 6 } _ { ( \uparrow 2 . 2 \% ) }$ </td></tr><tr><td rowspan="4">0.3</td><td>MSE</td><td>0.978</td><td> $\mathbf { 0 . 5 0 7 } _ { ( \uparrow 4 8 . 2 \% ) }$ </td><td> $\mathbf { 0 . 3 7 8 } _ { ( \uparrow 1 3 . 9 \% ) }$ </td><td> $\underline { { 0 . 5 7 1 } } ~ 0 . 5 4 3 _ { ( \uparrow 4 . 9 \% ) }$ </td><td></td><td>0.847  $\mathbf { 0 . 8 2 1 } _ { ( \uparrow 3 . 1 \% ) }$ </td><td>0.554  $\mathbf { 0 . 5 4 8 } _ { ( \uparrow 1 . 1 \% ) }$ </td><td>0.553</td><td> $\mathbf { 0 . 5 5 1 } _ { ( \uparrow 0 . 4 \% ) }$ </td><td>0.556 0.543(2.3%)</td><td>0.577 0.572(0.9%)</td><td>0.559</td><td> $\mathbf { 0 . 5 5 0 } _ { ( \uparrow 1 . 6 \% ) }$ </td></tr><tr><td>MAE</td><td>0.779</td><td>0.495  $\mathbf { 0 . 5 7 7 _ { ( \uparrow 2 5 . 9 \% ) } }$ </td><td> $\mathbf { 0 . 4 6 4 } _ { ( \uparrow 6 . 3 \% ) }$ </td><td>0.5800.564(2.8%)</td><td>0.743</td><td> $\mathbf { 0 . 7 3 0 } _ { ( \uparrow 1 . 7 \% ) }$ </td><td>0.566  $\mathbf { 0 . 5 6 0 } _ { ( \uparrow 1 . 1 \% ) }$ </td><td>0.565  $\mathbf { 0 . 5 6 1 } _ { ( \uparrow 0 . 7 \% ) }$ </td><td>0.563</td><td>0.554(†1.6%) 0.583</td><td>0.581(0.3%))</td><td>0.568</td><td> $\mathbf { 0 . 5 6 2 } _ { ( \uparrow 1 . 1 \% ) }$ </td></tr><tr><td>MSE 0.936</td><td> $\mathbf { 0 . 4 9 4 } _ { ( \uparrow 4 7 . 2 \% ) }$ </td><td>0.431</td><td> $\mathbf { 0 . 3 9 1 } _ { ( \uparrow 9 . 3 \% ) }$ </td><td>0.573 0.556(+3.0%)</td><td>0.816</td><td> $\mathbf { 0 . 7 9 4 } _ { ( \uparrow 2 . 7 \% ) }$ </td><td>0.561  $\mathbf { 0 . 5 5 4 } _ { ( \uparrow 1 . 2 \% ) }$ </td><td>0.558  $\mathbf { 0 . 5 5 2 } _ { ( \uparrow 1 . 1 \% ) }$ </td><td>0.556</td><td>0.550(↑1.1%)</td><td>0.578(↑1.2%)</td><td>0.581</td><td></td></tr><tr><td>MAE</td><td>0.762 0.571(↑25.1%)</td><td>0.492</td><td> $\mathbf { 0 . 4 7 7 } _ { ( \uparrow 3 . 0 \% ) }$ </td><td>0.584</td><td>0.575(↑1.5%) 0.729</td><td>0.721(+1.1%)</td><td>0.575  $\mathbf { 0 . 5 7 0 } _ { ( \uparrow 0 . 9 \% ) }$ </td><td>0.570  $\mathbf { 0 . 5 6 6 } _ { ( \uparrow 0 . 7 \% ) }$ </td><td>0.563</td><td>0.585 0.560(+0.5%) 0.594</td><td>0.589(↑0.8%)</td><td>0.588</td><td> $\mathbf { 0 . 5 5 0 } _ { ( \uparrow 5 . 3 \% ) }$ </td></tr><tr><td rowspan="3">0.7</td><td>MSE</td><td>0.841  $\mathbf { 0 . 4 7 0 } _ { ( \uparrow 4 4 . 1 \% ) }$ </td><td>0.411</td><td>0.370(↑10.0%)</td><td> $\underline { { 0 . 5 7 1 } } ~ 0 . 5 6 \mathbf { 1 } _ { ( \uparrow 1 . 8 \% ) }$ </td><td>0.785</td><td> $\mathbf { 0 . 7 6 3 } _ { ( \uparrow 2 . 8 \% ) }$ </td><td>0.556  $\mathbf { 0 . 5 4 9 } _ { ( \uparrow 1 . 3 \% ) }$ </td><td>0.552</td><td> $\mathbf { 0 . 5 4 8 } _ { ( \uparrow 0 . 7 \% ) }$  0.542</td><td>0.537(0.9%)</td><td>0.576 0.571(↑0.9%)</td><td>0.573</td><td> $\mathbf { 0 . 5 6 9 } _ { ( \uparrow 3 . 2 \% ) }$ </td></tr><tr><td>MAE</td><td>0.726 0.558(+23.1%)</td><td>0.481</td><td> $\mathbf { 0 . 4 3 5 } _ { ( \uparrow 9 . 6 \% ) }$ </td><td>0.5840.577(+1.2%)</td><td>0.721</td><td> $\mathbf { 0 . 7 0 7 } _ { ( \uparrow 1 . 9 \% ) }$ </td><td>0.578  $\mathbf { 0 . 5 7 2 } _ { ( \uparrow 1 . 0 \% ) }$ </td><td>0.567</td><td> $\mathbf { 0 . 5 6 4 } _ { ( \uparrow 0 . 5 \% ) }$  0.559</td><td> $\mathbf { 0 . 5 5 6 } _ { ( \uparrow 0 . 5 \% ) }$ </td><td>0.591 0.588(+0.5%)</td><td>0.585</td><td> $\mathbf { 0 . 5 5 0 } _ { ( \uparrow 4 . 0 \% ) }$ </td></tr><tr><td>MSE 0.828</td><td> $\mathbf { 0 . 4 6 4 } _ { ( \uparrow 4 4 . 0 \% ) }$ </td><td>0.409</td><td> $\mathbf { 0 . 3 6 0 } _ { ( \uparrow 1 2 . 0 \% ) }$ </td><td> $\underline { { 0 . 5 5 2 } } 0 . 5 3 9 _ { ( \uparrow 2 . 4 \% ) }$ </td><td> $\underline { { 0 . 7 4 2 } } 0 . 7 1 7 _ { ( \uparrow 3 . 4 \% ) }$ </td><td>0.535</td><td> $\mathbf { 0 . 5 3 2 } _ { ( \uparrow 0 . 6 \% ) }$ </td><td>0.536  $\mathbf { 0 . 5 3 1 } _ { ( \uparrow 0 . 9 \% ) }$ </td><td></td><td>0.555</td><td>0.550(↑0.9%)</td><td></td><td> $\mathbf { 0 . 5 7 3 } _ { ( \uparrow 2 . 1 \% ) }$ </td></tr><tr><td rowspan="2">0.9 MAE</td><td>0.716</td><td> $\mathbf { 0 . 5 5 0 } _ { ( \uparrow 2 3 . 2 \% ) }$ </td><td>0.479</td><td> $\mathbf { 0 . 4 5 7 } _ { ( \uparrow 4 . 6 \% ) }$ </td><td> $\underline { { 0 . 5 7 5 } } 0 . 5 7 0 _ { ( \uparrow 0 . 9 \% ) }$ </td><td>0.701</td><td> $\mathbf { 0 . 6 8 9 } _ { ( \uparrow 1 . 7 \% ) }$ </td><td>0.570  $\mathbf { 0 . 5 6 8 } _ { ( \uparrow 0 . 4 \% ) }$ </td><td>0.560</td><td> $\mathbf { 0 . 5 5 7 } _ { ( \uparrow 0 . 5 \% ) }$ </td><td> $\underline { { 0 . 5 1 8 } } 0 . 5 1 6 _ { ( \uparrow 0 . 4 \% ) }$   $\underline { { 0 . 5 4 7 } } 0 . 5 4 6 _ { ( \uparrow 0 . 2 \% ) }$ </td><td>0.581</td><td>0.555 0.580</td><td> $\mathbf { 0 . 5 4 1 } _ { ( \uparrow 2 . 5 \% ) }$ </td></tr><tr><td>Avg. Gain</td><td>46.0% /  24.5%</td><td></td><td> $1 1 . 9 \% \mathrm { ~ / ~ } 5 . 9 \%$ </td><td> $2 . 9 \% / \ 1 . 5 \%$ </td><td></td><td> $3 . 6 \% \ / \ 1 . 9 \%$ </td><td> $1 . 0 \% \mathrm { ~ / ~ } 0 . 9 \%$ </td><td> $\mathbf { 0 . 8 \% } \ / \ \mathbf { 0 . 5 \% }$ </td><td> $1 . 7 \% / 1 . 0 \%$ </td><td></td><td>0.577(+0.7%)  $1 . 2 \% / \ 0 . 8 \%$ </td><td> $3 . 6 \% / \ 2 . 1 \%$ </td><td> $\mathbf { 0 . 5 6 9 } _ { ( \uparrow 1 . 9 \% ) }$ </td></tr></table>

TABLE V

REAL-WORLD FORECASTING AVERAGED ACROSS FOUR PREDICTION HORIZONS PER DATASET (SETTINGS IN SEC. VI-A). +SACM DENOTES BACKBONES AUGMENTED WITH SACM; BOLD MARKS THE LOWER ERROR PER PAIR. AVG. GAIN MACRO-AVERAGES RELATIVE IMPROVEMENTS ACROSS DATASETS.
<table><tr><td rowspan=2 colspan=6>Dataset  MetricInformer (2021) Crossformer (2023)Raw +SACM Raw +SACM</td><td rowspan=1 colspan=9>PatchTST (2023) TimesNet (2023) iTransformer (2024)TimeMixer (2024)WPMix</td><td rowspan=1 colspan=6>er (2025)TimeFilter (2025)MultiPatchFormer (2025)</td></tr><tr><td rowspan=1 colspan=1>Raw</td><td rowspan=1 colspan=1>+SACM</td><td rowspan=1 colspan=1>Raw</td><td rowspan=1 colspan=1>+SACM</td><td rowspan=1 colspan=1>Raw</td><td rowspan=1 colspan=1> $+ S \mathbf { A C M }$ </td><td rowspan=1 colspan=1>Raw</td><td rowspan=1 colspan=1>+SACM</td><td rowspan=1 colspan=1>Raw</td><td rowspan=1 colspan=1>+SACM</td><td rowspan=1 colspan=1>Raw</td><td rowspan=1 colspan=1>+SACM</td><td rowspan=1 colspan=1>Raw</td><td rowspan=1 colspan=2>+SACM</td></tr><tr><td rowspan=2 colspan=1>ETThl   MSEMAE0</td><td rowspan=1 colspan=1>1.337</td><td rowspan=1 colspan=3> $\mathbf { 1 . 0 7 7 } _ { ( \uparrow \uparrow 9 . 4 \% ) }$ 0.449</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 4 1 } _ { ( \uparrow 1 . 8 \% ) }$ </td><td rowspan=1 colspan=1>0.459</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 4 6 } _ { ( \uparrow 2 . 8 \% ) }$ </td><td rowspan=1 colspan=1>0.534</td><td rowspan=1 colspan=1> $\mathbf { 0 . 5 2 0 } _ { ( \uparrow 2 . 6 \% ) }$ </td><td rowspan=1 colspan=1>0.448</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 4 5 } _ { ( \uparrow 0 . 7 \% ) }$ </td><td rowspan=1 colspan=1>0.458</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 5 3 } _ { ( \uparrow 1 . 1 \% ) }$ </td><td rowspan=1 colspan=1>0.431</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 3 1 } _ { ( 0 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.428</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 2 7 } _ { ( \uparrow 0 . 2 \% ) }$ </td><td rowspan=1 colspan=1>0.429</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 2 9 }</td><td rowspan=1 colspan=1>_ { ( 0 . 0 \% ) }$ </td></tr><tr><td rowspan=1 colspan=1>.823</td><td rowspan=1 colspan=1> $0 . 7 4 3 _ { ( \upa</td><td rowspan=1 colspan=1>rrow 9 . 7 \% ) }$ </td><td rowspan=1 colspan=1>0.445</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 3 9 } _ { ( \uparrow 1 . 3 \% ) }$ </td><td rowspan=1 colspan=1>0.432</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 2 9 } _ { ( \uparrow 0 . 7 \% ) }$ </td><td rowspan=1 colspan=1>0.492</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 8 3 } _ { ( \uparrow 1 . 8 \% ) }$ </td><td rowspan=1 colspan=1>0.431</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 3 1 } _ { ( 0 . 0 \% ) }$ 0</td><td rowspan=1 colspan=1>.429</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 2 7 } _ { ( \uparrow 0 . 5 \% ) }$ </td><td rowspan=1 colspan=1>0.455</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 5 4 } _ { ( \uparrow 0 . 2 \% ) }$ </td><td rowspan=1 colspan=1>0.452</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 4 9 } _ { ( \uparrow 0 . 7 \% ) }$ </td><td rowspan=1 colspan=1>0.441</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 3 9 } _ { ( \u</td><td rowspan=1 colspan=1>parrow 0 . 5 \% ) }$ </td></tr><tr><td rowspan=2 colspan=1>ETTh2   MSEMAE</td><td rowspan=1 colspan=1>2.657</td><td rowspan=1 colspan=1>1.392</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>.795</td><td rowspan=1 colspan=1> $\mathbf { 0 . 6 2 6 } _ { ( \uparrow 2 1 . 3 \% ) }$ </td><td rowspan=1 colspan=1>0.384</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 7 8 } _ { ( \uparrow 1 . 6 \% ) }$ </td><td rowspan=1 colspan=1>0.480</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 5 6 } _ { ( \uparrow 5 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.384</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 8 1 } _ { ( \uparrow 0 . 8 \% ) }$ </td><td rowspan=1 colspan=1>0.386</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 8 0 } _ { ( \uparrow 1 . 6 \% ) }$ </td><td rowspan=1 colspan=1>0.377</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 7 3 } _ { ( \uparrow 1 . 1 \% ) }$ </td><td rowspan=1 colspan=1>0.381</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 7 7 } _ { ( \uparrow 1 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.395</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 8 4 } _ { ( \u</td><td rowspan=1 colspan=1>parrow 2 . 8 \% ) }$ </td></tr><tr><td rowspan=1 colspan=1>1.120</td><td rowspan=1 colspan=2> $\mathbf { 0 . 8 2 7 _ { ( \uparrow 2 6 . 2 \% ) } }$ </td><td rowspan=1 colspan=1>0.599</td><td rowspan=1 colspan=1> $\mathbf { 0 . 5 2 1 } _ { ( \uparrow 1 3 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.403</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 8 } _ { ( \uparrow 1 . 2 \% ) }$ </td><td rowspan=1 colspan=1>0.460</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 4 7 } _ { ( \uparrow \uparrow 2 . 8 \% ) }$ </td><td rowspan=1 colspan=1>0.400</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 9 } _ { ( \uparrow 0 . 3 \% ) }$ </td><td rowspan=1 colspan=1>0.402</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 9 } _ { ( \uparrow 0 . 7 \% ) }$ </td><td rowspan=1 colspan=1>0.398</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 6 } _ { ( \uparrow 0 . 5 \% ) }$ </td><td rowspan=1 colspan=1>0.401</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 9 } _ { ( \uparrow 0 . 5 \% ) }$ </td><td rowspan=1 colspan=1>0.411</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 0 5 } _ { ( \u</td><td rowspan=1 colspan=1>parrow 1 . 5 \% ) }$ </td></tr><tr><td rowspan=2 colspan=1>MSEETTm1MAE0</td><td rowspan=1 colspan=1>1.480</td><td rowspan=1 colspan=3> $\mathbf { 1 . 1 9 4 } _ { ( \uparrow \uparrow 9 . 3 \% ) }$ 0.413</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 4 } _ { ( \uparrow 4 . 6 \% ) }$ </td><td rowspan=1 colspan=1>0.396</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 3 } _ { ( \uparrow 0 . 8 \% ) }$ </td><td rowspan=1 colspan=1>0.519</td><td rowspan=1 colspan=1> $0 . 5 1 2 _ { ( \uparrow 1 . 3 \% ) }$ </td><td rowspan=1 colspan=1>0.399</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 9 } _ { ( 0 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.393</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 2 } _ { ( \uparrow 0 . 3 \% ) }$ </td><td rowspan=1 colspan=1>0.391</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 8 8 } _ { ( \uparrow 0 . 8 \% ) }$ </td><td rowspan=1 colspan=1>0.394</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 3 _ { ( \uparrow 0 . 3 \% ) } } \ \un</td><td rowspan=1 colspan=1>derline { { 0 . 4 0 2 } }$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 9 } _ { ( \u</td><td rowspan=1 colspan=1>parrow 0 . 7 \% ) }$ </td></tr><tr><td rowspan=1 colspan=1>.855</td><td rowspan=1 colspan=3> $0 . 7 6 0 _ { ( \uparrow 1 1 . 1 \% ) }$ 0.404</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 2 } _ { ( \uparrow 3 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.387</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 8 5 } _ { ( \uparrow 0 . 5 \% ) }$ </td><td rowspan=1 colspan=1>0.474</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 7 0 } _ { ( \uparrow 0 . 8 \% ) }$ </td><td rowspan=1 colspan=1>0.390</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 0 } _ { ( 0 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.385</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 8 3 _ { ( \uparrow 0 . 5 \% ) } }$ </td><td rowspan=1 colspan=1>0.385</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 8 4 } _ { ( \uparrow 0 . 3 \% ) }$ </td><td rowspan=1 colspan=1>0.3860</td><td rowspan=1 colspan=1>.385(0.3%)</td><td rowspan=1 colspan=1>0.397</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 9 5 } _ { ( \u</td><td rowspan=1 colspan=1>parrow 0 . 5 \% ) }$ </td></tr><tr><td rowspan=2 colspan=1>MSEETTm2MAE</td><td rowspan=1 colspan=1>3.357</td><td rowspan=1 colspan=3>1.634-  0.379</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 5 1 } _ { ( \uparrow 7 . 4 \% ) }$ </td><td rowspan=1 colspan=1>0.279</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 7 9 } _ { ( 0 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.348</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 4 1 } _ { ( \uparrow 2 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.284</td><td rowspan=1 colspan=1>0.284(0.0%)</td><td rowspan=1 colspan=1>0.277</td><td rowspan=1 colspan=1>0.277(0.0%)</td><td rowspan=1 colspan=1>0.275</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 7 4 } _ { ( \uparrow 0 . 4 \% ) }$ </td><td rowspan=1 colspan=1>0.283</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 8 3 _ { ( 0 . 0 \% ) } }$ </td><td rowspan=1 colspan=1>0.284</td><td rowspan=1 colspan=1>0.282</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>1.128</td><td rowspan=1 colspan=3>0.738(↑34.6%) 0.395</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 7 8 } _ { ( \uparrow 4 . 3 \% ) }$ </td><td rowspan=1 colspan=1>0.319</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 1 9 } _ { ( 0 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.364</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 6 0 } _ { ( \uparrow 1 . 1 \% ) }$ </td><td rowspan=1 colspan=1>0.321</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 2 1 } _ { ( 0 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.317</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 1 7 _ { ( 0 . 0 \% ) } }$ </td><td rowspan=1 colspan=1>0.317</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 1 6 } _ { ( \uparrow 0 . 3 \% ) }$ </td><td rowspan=1 colspan=1>0.321</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 2 1 } _ { ( 0 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.325</td><td rowspan=1 colspan=1>0.324</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>MSEWeatherMAE</td><td rowspan=2 colspan=1> $\begin{array} { r } { \frac { 0 . 4 8 4 } { 0 . 4 3 2 } \ 0 . 3 8 1 _</td><td rowspan=2 colspan=3>{ ( \uparrow 2 1 . 3 \% ) } \ \frac { 0 . 2 4 1 } { 0 . 2 6 8 } } \\ { \frac { 0 . 4 3 2 } { 0 . 4 3 2 } \ 0 . 3 7 3 _ { ( \uparrow 1 3 . 7 \% ) } \ \frac { 0 . 2 6 8 } { 0 . 2 6 8 } } \end{array}$ </td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 3 9 } _ { ( \uparrow 0 . 8 \% ) }$ </td><td rowspan=1 colspan=1>0.251</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 4 7 } _ { ( \uparrow 1 . 6 \% ) }$ </td><td rowspan=1 colspan=1>0.289</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 8 4 } _ { ( \uparrow 1 . 7 \mathbb { S } ) }$ </td><td rowspan=1 colspan=1>0.268</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 6 5 } _ { ( \uparrow 1 . 1 \% ) }$ </td><td rowspan=1 colspan=1>0.282</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 4 4 } _ { ( \uparrow 1 3 . 5 \% ) }$ </td><td rowspan=1 colspan=1>0.243</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 4 2 } _ { ( \uparrow 0 . 4 \% ) }$ </td><td rowspan=1 colspan=1>0.246</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 4 4 } _ { ( \uparrow 0 . 8 \% ) } ~ \un</td><td rowspan=1 colspan=1>derline { { 0 . 2 5 2 } }$ </td><td rowspan=1 colspan=1> $\underline { { 0 . 2 5 0 _ { ( \u</td><td rowspan=1 colspan=1>parrow 0 . 8 \% ) } } }$ </td></tr><tr><td rowspan=1 colspan=1> $\mathbf { 0 . 2 6 6 } _ { ( \uparrow 0 . 7 \% ) }$ </td><td rowspan=1 colspan=1>0.270</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 6 6 } _ { ( \uparrow 1 . 5 \% ) }$ </td><td rowspan=1 colspan=1>0.305</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 0 2 } _ { ( \uparrow 1 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.280</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 7 6 } _ { ( \uparrow 1 . 4 \% ) }$ </td><td rowspan=1 colspan=1>0.301</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 6 4 } _ { ( \uparrow 1 2 . 3 \% ) }$ </td><td rowspan=1 colspan=1>0.264</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 6 3 } _ { ( \uparrow 0 . 4 \% ) }$ </td><td rowspan=1 colspan=1>0.267</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 6 4 } _ { ( \uparrow 1 . 1 \% ) }$ </td><td rowspan=1 colspan=1>0.270</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 6 8 } _ { ( \u</td><td rowspan=1 colspan=1>parrow 0 . 7 \% ) }$ </td></tr><tr><td rowspan=2 colspan=1>MSEElectricity MAE</td><td rowspan=1 colspan=1>1.528</td><td rowspan=1 colspan=1>0.489</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.213</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 1 1 } _ { ( \uparrow 0 . 9 \% ) }$ </td><td rowspan=1 colspan=1>0.215</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 1 0 } _ { ( \uparrow 2 . 3 \% ) }$ </td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 0 3 } _ { ( \uparrow 1 . 5 \% ) }$ </td><td rowspan=1 colspan=1>0.215</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 1 3 _ { ( \uparrow 0 . 9 \% ) } }$ </td><td rowspan=1 colspan=1>0.211</td><td rowspan=1 colspan=1> $\underline { { 0 . 2 1 2 } } _ { ( \downarrow 0 . 5 \% ) }$ </td><td rowspan=1 colspan=1>0.208</td><td rowspan=1 colspan=1> $\mathbf { 0 . 1 9 8 } _ { ( \uparrow 4 . 8 \% ) }$ </td><td rowspan=1 colspan=1>0.220</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 2 0 } _ { ( 0 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.320</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 9 7 } _ { ( \u</td><td rowspan=1 colspan=1>parrow 7 . 2 \% ) }$ </td></tr><tr><td rowspan=1 colspan=1>0.989</td><td rowspan=1 colspan=1>0.459</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.290</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 8 7 } _ { ( \uparrow 1 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.290</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 8 6 } _ { ( \uparrow 1 . 4 \% ) }$ </td><td rowspan=1 colspan=1>0.304</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 0 1 } _ { ( \uparrow 1 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.290</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 7 3 _ { ( \uparrow 5 . 9 \% ) } }$ </td><td rowspan=1 colspan=1>0.282</td><td rowspan=1 colspan=1> $0 . 2 8 2 _ { ( 0 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.284</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 7 8 } _ { ( \uparrow 2 . 1 \% ) }$ </td><td rowspan=1 colspan=1>0.294</td><td rowspan=1 colspan=1> $\mathbf { 0 . 2 9 4 } _ { ( 0 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.377</td><td rowspan=1 colspan=1> $\mathbf { 0 . 3 6 4 } _ { ( \u</td><td rowspan=1 colspan=1>parrow 3 . 4 \% ) }$ </td></tr><tr><td rowspan=2 colspan=1>MSE $I L I$     MAE</td><td rowspan=1 colspan=1> $\underline { { 7 . 1 3</td><td rowspan=1 colspan=1>0 } } 6 . 2 3 5 _ { ( \upa</td><td rowspan=1 colspan=1>rrow 1 2 . 6 \% ) }$ </td><td rowspan=1 colspan=1>5.035</td><td rowspan=1 colspan=1> ${ \bf 4 . 3 2 1 } _ { ( \uparrow 1 4 . 2 \% ) }$ </td><td rowspan=1 colspan=1>3.321</td><td rowspan=1 colspan=1> $2 . 9 4 2 _ { ( \uparrow 1 1 . 4 \% ) }$ </td><td rowspan=1 colspan=1>6.045</td><td rowspan=1 colspan=1> $4 . 8 7 2 _ { ( \uparrow 1 9 . 4 \% ) }$ </td><td rowspan=1 colspan=1>3.163</td><td rowspan=1 colspan=1> $2 . 9 9 4 _ { ( \uparrow 5 . 3 \% ) }$ </td><td rowspan=1 colspan=1>3.182</td><td rowspan=1 colspan=1> $3 . 1 5 1 _ { ( \uparrow 1 . 0 \% ) }$ </td><td rowspan=1 colspan=1>3.081</td><td rowspan=1 colspan=1> $2 . 9 7 7 _ { ( \uparrow 3 . 4 \% ) }$ </td><td rowspan=1 colspan=1>2.407</td><td rowspan=1 colspan=1>2.223(↑7.6%)</td><td rowspan=1 colspan=1>2.804</td><td rowspan=1 colspan=1> $2 . 5 5 0 _ { ( \uparr</td><td rowspan=1 colspan=1>ow 9 . 1 \% ) }$ </td></tr><tr><td rowspan=1 colspan=1>1.9011</td><td rowspan=1 colspan=1>.772(1</td><td rowspan=1 colspan=1>6.8%)</td><td rowspan=1 colspan=1>1.542</td><td rowspan=1 colspan=1> $\mathbf { 1 . 4 0 6 } _ { ( \uparrow \ 8 . 8 \% ) }$ </td><td rowspan=1 colspan=1>1.110</td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 6 4 } _ { ( \uparrow 4 . 1 \% ) }$ </td><td rowspan=1 colspan=1>1.303</td><td rowspan=1 colspan=1> $1 . 2 2 3 _ { ( \uparrow 6 . 1 \% ) }$ </td><td rowspan=1 colspan=1>1.069</td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 4 8 } _ { ( \uparrow 2 . 0 \% ) }$ </td><td rowspan=1 colspan=1>1.148</td><td rowspan=1 colspan=1> $\mathbf { 1 . 1 4 2 } _ { ( \uparrow 0 . 5 \% ) }$ </td><td rowspan=1 colspan=1>1.065</td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 3 8 } _ { ( \uparrow 2 . 5 \% ) }$ </td><td rowspan=1 colspan=1>0.9650</td><td rowspan=1 colspan=1>.931(↑3.5%)</td><td rowspan=1 colspan=1>0.991</td><td rowspan=1 colspan=1> $\underline { { 1 . 0 0 1 } } _ { (</td><td rowspan=1 colspan=1>\downarrow 1 . 0 \% ) }$ </td></tr><tr><td rowspan=2 colspan=1>MSEExchange RateMAE</td><td rowspan=1 colspan=1>3.533</td><td rowspan=1 colspan=1>1.434</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.933</td><td rowspan=1 colspan=1>0.851(↑8.8%)</td><td rowspan=1 colspan=1>0.458</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 4 4 } _ { ( \uparrow \ 3 . 1 \% ) }$ </td><td rowspan=1 colspan=1>0.478</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 7 7 } _ { ( \uparrow 0 . 2 \% ) }$ </td><td rowspan=1 colspan=1>0.451</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 3 8 } _ { ( \uparrow 2 . 9 \% ) }$ </td><td rowspan=1 colspan=1>0.445</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 4 1 } _ { ( \uparrow 0 . 9 \% ) }$ </td><td rowspan=1 colspan=1>0.445</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 4 1 } _ { ( \uparrow 0 . 9 \% ) }$ </td><td rowspan=1 colspan=1>0.446</td><td rowspan=1 colspan=1>0.444(↑0.4%)</td><td rowspan=1 colspan=1>0.526</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 6 9 } _ { ( \u</td><td rowspan=1 colspan=1>parrow 1 0 . 8 \% ) }$ </td></tr><tr><td rowspan=1 colspan=1>1.410</td><td rowspan=1 colspan=1>0.875</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.657</td><td rowspan=1 colspan=1> $\mathbf { 0 . 6 2 6 } _ { ( \uparrow 4 . 7 \% ) }$ </td><td rowspan=1 colspan=1>0.457</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 5 2 } _ { ( \uparrow 1 . 1 \% ) }$ </td><td rowspan=1 colspan=1>0.478</td><td rowspan=1 colspan=1> $\underline { { 0 . 4 7 9 } } _ { ( \downarrow 0 . 2 \% ) }$ </td><td rowspan=1 colspan=1>0.455</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 4 8 } _ { ( \uparrow 1 . 5 \% ) }$ </td><td rowspan=1 colspan=1>0.446</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 4 5 } _ { ( \uparrow 0 . 2 \% ) }$ </td><td rowspan=1 colspan=1>0.449</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 4 5 } _ { ( \uparrow 0 . 9 \% ) }$ </td><td rowspan=1 colspan=1>0.452</td><td rowspan=1 colspan=1>0.448(↑0.9%)</td><td rowspan=1 colspan=1>0.486</td><td rowspan=1 colspan=1> $\mathbf { 0 . 4 6 3 } _ { ( \</td><td rowspan=1 colspan=1>uparrow 4 . 7 \% ) }$ </td></tr><tr><td rowspan=2 colspan=2>MSE1.306ECG             0MAE0.826</td><td rowspan=1 colspan=3>1.052(↑19.4%)0.711</td><td rowspan=1 colspan=1>0.682(t4.1%)</td><td rowspan=1 colspan=1>0.750</td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 2 2 } _ { ( \uparrow \mathrm { 3 . 7 \mathcal { B } } ) }$ </td><td rowspan=1 colspan=1>0.745</td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 4 0 } _ { ( \uparrow 0 . 7 \% ) }$ </td><td rowspan=1 colspan=1>0.762</td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 4 0 } _ { ( \uparrow \uparrow 2 . 9 \% ) }$ </td><td rowspan=1 colspan=1>0.715</td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 1 0 } _ { ( \uparrow 0 . 7 \% ) }$ </td><td rowspan=1 colspan=1>0.725</td><td rowspan=1 colspan=1> $\mathbf { 0 . 6 9 4 } _ { ( \uparrow 4 . 3 \% ) }$ </td><td rowspan=1 colspan=1>0.737</td><td rowspan=1 colspan=1>0.730(↑0.9%)</td><td rowspan=1 colspan=1>0.750</td><td rowspan=1 colspan=1> $\mathbf { 0 . 7 3 3 } _ { ( \u</td><td rowspan=1 colspan=1>parrow 2 . 3 \% ) }$ </td></tr><tr><td rowspan=1 colspan=3>.693(↑16.1%)0.518</td><td rowspan=1 colspan=1>0.50 (+2.5%))</td><td rowspan=1 colspan=1>0.533</td><td rowspan=1 colspan=1> $0 . 5 1 7 _ { ( \uparrow 3 . 0 \% ) }$ </td><td rowspan=1 colspan=1>0.532</td><td rowspan=1 colspan=1> $\mathbf { 0 . 5 2 9 } _ { ( \uparrow 0 . 6 \% ) }$ </td><td rowspan=1 colspan=1>0.537</td><td rowspan=1 colspan=1> $\mathbf { 0 . 5 2 8 } _ { ( \uparrow 1 . 7 \% ) }$ </td><td rowspan=1 colspan=1>0.515</td><td rowspan=1 colspan=1> $0 . 5 1 2 _ { ( \uparrow 0 . 6 \% ) }$ </td><td rowspan=1 colspan=1>0.518</td><td rowspan=1 colspan=1>0.502( 3.1%))</td><td rowspan=1 colspan=1>0.524</td><td rowspan=1 colspan=1>0.519(↑1.0%)%)</td><td rowspan=1 colspan=1>0.532</td><td rowspan=1 colspan=1> $\mathbf { 0 . 5 2 4 } _ { ( \u</td><td rowspan=1 colspan=1>parrow 1 . 5 \% ) }$ </td></tr><tr><td rowspan=1 colspan=21>Avg. Gain     $3 5 . 4 \% / \ 2 3 . 3 \%$     $7 . 1 \% / 4 . 4 \%$     3.0% / 1.5%     $3 . 8 \% / \ 1 . 7 \%$      $1 . 6 \% / 1 . 4 \%$      $2 . 0 \% / \ 1 . 7 \%$     $1 . 8 \% / \ 1 . 1 \%$     $1 . 3 \% \mathrm { ~ / ~ } 0 . 9 \%$       $3 . 8 \% / \ 1 . 3 \%$ </td></tr></table>

Implementation Details. All models retain their published backbone architecture, default task head, and default optimization hyperparameters. We implement the paired Raw and +SACM variants in PyTorch, optimize with Adam [58], and run experiments on NVIDIA A800 GPUs. Unless stated otherwise, SACM fixes $p _ { \mathrm { m i n } } = 0 . 0 5$ and $p _ { \operatorname* { m a x } } = 0 . 5 0$ , and initializes the learnable parameters at α = 10, $\mathbf { b } _ { s } = 0 ,$ , and $\gamma = 1$ . The scorer and mapper are active only during training; evaluation disables them and all stochastic dropout, so inference follows the deterministic backbone graph. Dataset-specific splits, window construction, corruption generation, decoder usage, controlled baselines, and random-seed protocols are stated with the corresponding RQ below and tabulated in Supplementary Appendices A–B. Evaluation reports MSE/MAE for forecasting, top-1 accuracy for classification, and point-adjusted precision, recall, and F1 for anomaly detection.

and horizons $H \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ . Across all backbones and corruption scales, SACM lowers horizon-averaged MSE; Informer [8] shows the largest reduction, 46.0% lower average MSE and up to 48.2% at σ = 0.3. Crossformer [30] achieves an 11.9% gain, while TimesNet [5] improves by 3.6%. PatchTST, iTransformer, and TimeMixer achieve gains of 0.8%–2.9%. Newer backbones follow: WPMixer [51], TimeFilter [46], and MultiPatchFormer [52] reduce average MSE by 1.7%, 1.2%, and 3.6%, improving at all scales. These gains across backbone designs support a separate capacity-modulation path: sampleadaptive regularization complements temporal feature modeling without changing the backbone’s inference architecture.

## B. RQ1: Robustness under Reliability Heterogeneity

Non-Monotonic Corruption Response. Figure 6 plots the horizon-averaged MSE of Informer and Crossformer against the corruption scale $\sigma ~ \in ~ [ 0 . 1 , 0 . 9 ]$ . Error does not grow monotonically with noise for either baseline: Informer rises to a mid-noise maximum and then decreases at larger $\sigma ,$ while Crossformer decreases throughout. These different baseline trajectories motivate comparing Raw and +SACM at each matched corruption scale. SACM (blue) remains below the corresponding baseline at every plotted level, flattening the Informer mid-noise peak and reducing Crossformer error throughout. The matched gains across these different trajectories provide a more consistent robustness indicator than the slope of either error curve alone.

Performance under Controlled Corruptions. Table III summarizes the forecasting performance on Synth-12 under the layered Gaussian, heavy-tailed, and point-wise missing corruptions of Table II. Each corrupted window is paired with its clean target, so the model trains on the corrupted observation but is evaluated against the clean future. The benchmark contains 33,600 windows with $\sigma \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$

![](images/8083c70bd41bc8c447093bd4a8d397b191d5ec90eda291f08e6a1e9e1b5bac61.jpg)  
Fig. 6. Error trajectories under increasing corruption. (a) Informer exhibits a mid-noise error peak under fixed capacity (red), which is absent with SACM (blue). (b) Crossformer shows decreasing baseline error across the plotted corruption levels; SACM remains lower and smoother.

## C. RQ2: Real-World Forecasting & Capacity Allocation

Long-Term Forecasting Performance. Table V compares nine backbones on nine real-world benchmarks using chronological train/validation/test splits and the forecasting settings of Supplementary Appendix A. SACM achieves gains across many backbones and horizons, with smaller or tied changes on ETTm2. Informer reduces average MSE by 68.0% on Electricity, 59.4% on Exchange Rate, and 47.6% on ETTh2. TimeMixer improves by 13.7% on Weather, while Multi-PatchFormer gains 7.2% on Electricity. On ILI, WPMixer, TimeFilter, and MultiPatchFormer reduce MSE by 3.4%, 7.6%, and 9.1%, respectively. On ECG waveforms, Informer and PatchTST improve MSE by 19.4% and 3.8%. Larger gains on Informer and smaller gains on newer backbones indicate different regularization sensitivity, while improvements across domains support its use with diverse temporal representations. Sample-Wise Capacity Allocation Analysis. Table VI reports observed sample-wise dropout rates $p _ { i }$ from the final convergence epochs on the training windows used for forecasting. ILI has a lower mean dropout rate than ETTm2 (0.267 vs. 0.329), but its variance is fourfold larger (0.0194 vs. 0.0049). Thus, SACM retains more activations on average on ILI and varies retention more across its windows. The distributions illustrate both regularization strength and sample allocation.

TABLE VI  
DISTRIBUTION OF SAMPLE-ADAPTIVE DROPOUT RATES DURING FINAL EPOCHS. STATISTICS AFTER BATCH-RELATIVE RATE MAPPING.
<table><tr><td>Dataset</td><td> $\mathbf { M e a n } \pm \mathbf { S t d } .$ </td><td>Variance</td><td>Rel. Var.</td><td>Allocation</td></tr><tr><td>ETTm2</td><td> $0 . 3 2 9 \pm 0 . 0 7 0$ </td><td>0.0049</td><td>1.0×</td><td>Lower variation</td></tr><tr><td>ILI</td><td> $0 . 2 6 7 \pm 0 . 1 3 9$ </td><td>0.0194</td><td>4.0×</td><td>Higher variation</td></tr></table>

## D. RQ3: Data-Regime and Cross-Domain Generalization

Varying Training-Data Fractions. Table VII evaluates PatchTST on ILI using 10%, 25%, 50%, 75%, and 100% of the chronological training split, with validation and test windows unchanged. SACM lowers both errors at every fraction and improves monotonically as data grows (MAE 1.295 → 1.064, MSE $4 . 0 7 4 \  \ 2 . 9 4 2 )$ , whereas the unregularized model stalls and then regresses: its MSE holds at 3.169 from 50% to 75% and rises to 3.321 at 100%. SACM’s relative gain therefore peaks at full data (4.14% MAE, 11.41% MSE), so more training data strengthens its advantage; the MSEdominant reduction indicates suppression of the large deviations that dominate squared error. Notably, SACM at 50% data already beats the raw model at 100% (MSE 3.143 vs. 3.321). Zero-Shot Cross-Domain Transfer. We test whether sourcedomain residual regularization aids cross-domain generalization. We apply a source-trained PatchTST to the target. Table VII lists eight ETT transfer directions. No target-domain fine-tuning or label selection is used. SACM lowers both errors in all eight directions and never degrades a metric (average MAE 0.402 → $0 . 3 9 5 , \ \mathrm { M S E } \ 0 . 4 0 7 \  \ 0 . 4 0 4 )$ , with its largest gain (MAE $0 . 5 1 9  0 . 5 0 1 )$ on $E T T h 2  E T T h I ,$ the highest-error direction. The scorer and dropout are disabled at evaluation, so these gains add no inference cost. Because the target pass is identical in both settings, the gains stem from source training alone.

TABLE VII  
PATCHTST PERFORMANCE ACROSS TRAINING-DATA FRACTIONS (ILI, TOP) AND ZERO-SHOT OOD TRANSFER (ETT, BOTTOM). SHADED CELLS DENOTE SACM; ARROWS INDICATE RELATIVE CHANGES.
<table><tr><td colspan="5">Training-Data Fractions on ILI</td></tr><tr><td>Scenario</td><td colspan="2">Raw</td><td colspan="2">+SACM</td></tr><tr><td></td><td>MAE ↓ MSE↓</td><td></td><td>MAE↓</td><td>MSE↓</td></tr><tr><td>10% Data</td><td>1.333</td><td>4.185</td><td> $\mathbf { 1 . 2 9 5 } _ { ( \uparrow 2 . 9 \% ) }$ </td><td> $\pm . 0 7 4 _ { ( \uparrow 2 . 7 \% ) }$ </td></tr><tr><td>25% Data</td><td>1.156</td><td>3.201</td><td> $1 . 1 4 4 _ { ( \uparrow 1 . 0 \% ) }$ </td><td> ${ \bf 3 . 1 9 5 } _ { ( \uparrow 0 . 2 \% ) }$ </td></tr><tr><td>50% Data</td><td>1.116</td><td>3.169</td><td> $\mathbf { 1 . 1 0 6 } _ { ( \uparrow 0 . 9 \% ) }$ </td><td> $3 . 1 4 3 _ { ( \uparrow 0 . 8 \% ) }$ </td></tr><tr><td>75% Data</td><td>1.111</td><td>3.169</td><td> $\mathbf { 1 . 0 9 6 } _ { ( \uparrow 1 . 4 \% ) }$ </td><td> $3 . 1 2 8 _ { ( \uparrow 1 . 3 \% ) }$ </td></tr><tr><td>100% Data</td><td>1.110</td><td>3.321</td><td> $\mathbf { 1 . 0 6 4 } _ { ( \uparrow 4 . 1 \% ) }$ </td><td> $2 . 9 4 2 _ { ( \uparrow 1 1 . 4 \% ) }$ </td></tr><tr><td>Average</td><td>1.165</td><td>3.409</td><td> $\mathbf { 1 . 1 4 1 } _ { ( \uparrow 2 . 1 \% ) }$ </td><td> $3 . 2 9 6 _ { ( \uparrow 3 . 3 \% ) }$ </td></tr></table>

Zero-Shot Generalization on ETT
<table><tr><td rowspan="2">Scenario</td><td colspan="2">Raw</td><td colspan="2">+SACM</td></tr><tr><td>MAE↓</td><td>MSE↓</td><td>一  $\overline { { \mathbf { M A E \downarrow } } }$ </td><td>MSE↓</td></tr><tr><td> $\overline { { E T T h I \to E T T h 2 } }$ </td><td>0.391</td><td>0.406</td><td> $\mathbf { 0 . 3 8 7 } _ { ( \uparrow 1 . 0 \% ) }$ </td><td>0.404(↑0.5%)</td></tr><tr><td> $E T T h I \to E T T m 2$ </td><td>0.319</td><td>0.359</td><td> $\mathbf { 0 . 3 1 8 } _ { ( \uparrow 0 . 3 \% ) }$ </td><td> $\mathbf { 0 . 3 5 8 } _ { ( \uparrow 0 . 3 \% ) }$ </td></tr><tr><td> $E T T h 2  E T T h I$ </td><td>0.519</td><td>0.480</td><td> $\mathbf { 0 . 5 0 1 } _ { ( \uparrow 3 . 5 \% ) }$ </td><td> $\mathbf { 0 . 4 7 3 } _ { ( \uparrow 1 . 5 \% ) }$ </td></tr><tr><td>ETTh2 → ETTm2</td><td>0.325</td><td>0.364</td><td> $\mathbf { 0 . 3 1 7 _ { ( \uparrow 2 . 5 \% ) } }$ </td><td> $\mathbf { 0 . 3 5 8 } _ { ( \uparrow 1 . 6 \% ) }$ </td></tr><tr><td> $E T T m I \to E T T h 2$ </td><td>0.457</td><td>0.442</td><td> $\mathbf { 0 . 4 4 8 } _ { ( \uparrow 2 . 0 \% ) }$ </td><td> $\mathbf { 0 . 4 3 9 } _ { ( \uparrow 0 . 7 \% ) }$ </td></tr><tr><td> $E T T m I \to E T T m 2$ </td><td>0.305</td><td>0.331</td><td> $\mathbf { 0 . 3 0 2 } _ { ( \uparrow 1 . 0 \% ) }$ </td><td> $\mathbf { 0 . 3 3 0 } _ { ( \uparrow 0 . 3 \% ) }$ </td></tr><tr><td> $E T T m 2  E T T h 2$ </td><td>0.424</td><td>0.428</td><td> $\mathbf { 0 . 4 2 0 } _ { ( \uparrow 0 . 9 \% ) }$ </td><td> $\mathbf { 0 . 4 2 7 } _ { ( \uparrow 0 . 2 \% ) }$ </td></tr><tr><td> $E T T m 2  E T T m I$ </td><td>0.476</td><td>0.444</td><td> $\mathbf { 0 . 4 7 0 } _ { ( \uparrow 1 . 3 \% ) }$ </td><td> $\mathbf { 0 . 4 4 2 } _ { ( \uparrow 0 . 5 \% ) }$ </td></tr><tr><td>Average</td><td>0.402</td><td>0.407</td><td> $\mathbf { 0 . 3 9 5 } _ { ( \uparrow 1 . 7 \% ) }$ </td><td> $\mathbf { 0 . 4 0 4 } _ { ( \uparrow 0 . 7 \% ) }$ </td></tr></table>

## E. RQ4: Cross-Task Generality

Classification. Classification requires learning temporal features that distinguish classes across samples with varying measurement quality and local irregularity. By assigning stronger dropout to windows with higher residual scores, SACM aims to reduce reliance on sample-specific fluctuations while retaining more activations for lower-score windows. Table VIII evaluates this regularization strategy in 192 matched pairs across six backbones and 32 datasets, with each task trained independently using its train/test split. Macro accuracy increases from 68.82% to 70.91%, a gain of 2.09 percentage points (pp), with 158 wins, 34 ties, and no losses at the reported precision. Notable gains span architectures: InceptionTime improves from 41.98% to 61.83% on EigenWorms (+19.85

CLASSIFICATION ACCURACY (%) ON 32 DATASETS AND SIX BACKBONES. SHADED +SACM CELLS DENOTE SACM AND SHOW CHANGES FROM RAW IN PERCENTAGE POINTS (PP); BOLD MARKS THE HIGHER ACCURACY WITHIN EACH PAIR.

TABLE VIII

$$
+ { \cal S } \mathbf { A } \mathbf { C } \mathbf { M }
$$

$$
+ { \cal S } \mathbf { A } \mathbf { C } \mathbf { M }
$$

$$
+ { \cal S } \mathbf { A } \mathbf { C } \mathbf { M }
$$

$$
{ \pmb 9 7 . 6 7 } _ { ( \uparrow 0 . 3 \mathrm { p p } ) }
$$

$$
\mathbf { 9 8 . 0 0 } _ { ( \uparrow \uparrow 1 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 9 7 . 6 7 _ { ( \uparrow 0 . 3 \mathrm { p p ) } } }
$$

$$
\uparrow 6 . 7 \mathsf { p p } )
$$

$$
{ \bf 9 9 . 0 0 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
{ \bar { 5 } } 3 . 3 3 _ { ( \uparrow 6 . 7 \mathrm { p p ) } }
$$

$$
\mathbf { 9 7 . 6 7 } _ { ( \uparrow 1 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 4 0 . 0 0 } _ { ( \uparrow \uparrow 6 . 7 \mathrm { p p } ) }
$$

$$
( \uparrow 2 . 5 \mathrm { p p } )
$$

$$
\mathbf { 7 0 . 0 0 } _ { ( \uparrow 2 . 5 \mathrm { p p } ) }
$$

$$
3 3 . 3 3 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 4 0 . 0 0 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
3 3 . 3 3 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 8 7 . 5 0 } _ { ( \uparrow 2 . 5 \mathrm { p p } ) }
$$

$$
\$ 100.00
$$

$$
2 5 . \mathbf { 0 0 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 9 2 . 5 0 } _ { ( \uparrow 7 . 5 \mathrm { p p } ) }
$$

$$
\mathbf { y 8 . 6 I } _ { ( \uparrow 0 . 2 \mathrm { p p } ) }
$$

$$
9 8 . 3 3 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 8 8 . 8 9 } _ { ( \uparrow 1 . 4 \mathrm { p p } ) }
$$

$$
{ \bf 9 8 . 5 4 } _ { ( \uparrow 0 . 2 \mathrm { p p } ) }
$$

$$
9 9 . 7 9 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
{ \bf 9 8 . 6 1 } _ { ( \uparrow 8 . 3 \mathrm { p p } ) }
$$

$$
\mathbf { 4 2 . 0 0 } _ { ( \uparrow 4 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 2 6 . 0 0 } _ { ( \uparrow 2 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 8 4 . 7 2 _ { ( \uparrow 5 . 6 \mathrm { p p } ) } }
$$

$$
9 3 . 0 6 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 2 8 . 0 0 } _ { ( \uparrow 4 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { \$ 8 .61 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 5 6 . 0 0 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 6 0 . 0 0 } _ { ( \uparrow \uparrow 6 . 0 \mathrm { p p } ) }
$$

$$
\pmb { 4 9 . 6 2 } _ { ( \uparrow 2 . 3 \mathrm { p p } ) }
$$

$$
\pmb { 5 5 . 7 3 } _ { ( \uparrow 5 . 4 \mathrm { p p } ) }
$$

$$
\mathbf { 5 6 . 4 9 } _ { ( \uparrow 3 . 8 \mathrm { p p } ) }
$$

$$
\mathbf { 4 5 . 0 4 } _ { ( \uparrow 6 . 9 \mathrm { p p } ) }
$$

$$
\mathbf { 6 1 . 8 3 } _ { ( \uparrow 1 9 . 9 \mathrm { p p } ) }
$$

$$
{ \bf 7 8 . 9 9 } _ { ( \uparrow 2 . 2 \mathrm { p p } ) }
$$

$$
\mathbf { 5 6 . 5 2 } _ { ( \uparrow 7 . 2 \mathrm { p p } ) }
$$

$$
\mathbf { 9 6 . 3 8 } _ { ( \uparrow 0 . 7 \mathrm { p p } ) }
$$

$$
\mathbf { 9 2 . 0 3 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
3 4 . 2 2 _ { ( \uparrow 2 . 7 \mathrm { p p } ) }
$$

$$
2 9 . 6 6 _ { ( \uparrow 1 . 5 \mathrm { p p ) } }
$$

$$
3 2 . 3 2 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 3 1 . 9 4 } _ { ( \uparrow 2 . 3 \mathrm { p p } ) }
$$

$$
{ 3 5 . 3 6 } _ { ( \uparrow 2 . 3 \mathrm { p p } ) }
$$

$$
{ \bf 9 4 . 0 7 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 9 4 . 0 7 } _ { ( \uparrow 0 . 4 \mathrm { p p } ) }
$$

$$
\mathbf { 3 0 . 4 2 } _ { ( \uparrow 3 . 0 \mathrm { p p } ) }
$$

$$
9 4 . 4 4 _ { ( \uparrow 0 . 7 \mathrm { p p } ) }
$$

$$
9 3 . 3 3 _ { ( \uparrow 0 . 4 \mathrm { p p ) } }
$$

$$
\mathbf { 6 8 . 1 0 } _ { ( \uparrow 0 . 6 \mathrm { p p } ) }
$$

$$
\mathbf { 6 6 . 0 6 } _ { ( \uparrow 0 . 9 \mathrm { p p } ) }
$$

$$
3 3 . 3 3 _ { ( \uparrow 1 6 . 7 \mathrm { p p ) } }
$$

$$
\mathbf { 6 8 . 7 0 } _ { ( \uparrow 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 6 8 . 5 9 } _ { ( \uparrow 1 . 6 \mathrm { p p } ) }
$$

$$
{ \bf 6 8 . 0 8 } _ { ( \uparrow 2 . 4 \mathrm { p p } ) }
$$

$$
\bar { \mathsf { \mathbf { s } } } \mathbf { 1 . 5 3 } _ { ( \uparrow 1 . 3 \mathrm { p p } ) }
$$

$$
\pmb { 5 7 . 0 0 } _ { ( \uparrow 2 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 5 8 . 0 0 } _ { ( \uparrow 2 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 6 1 . 0 0 } _ { ( \uparrow 6 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 5 2 . 7 0 } _ { ( \uparrow \mathrm { 5 . 4 p p } ) }
$$

$$
\mathbf { 5 8 . 0 0 } _ { ( \uparrow 4 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 5 6 . 7 6 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 6 2 . 1 6 } _ { ( \uparrow 4 . 1 \mathrm { p p } ) }
$$

$$
\mathbf { 5 9 . 4 6 } _ { ( \uparrow 1 . 4 \mathrm { p p } ) }
$$

$$
2 7 . 8 8 _ { ( \uparrow 1 . 3 \mathrm { p p } ) }
$$

$$
\mathbf { 4 8 . 6 5 } _ { ( \uparrow 5 . 4 \mathrm { p p } ) }
$$

$$
\mathbf { 2 8 . 3 5 _ { ( \uparrow 2 . 9 \mathrm { p p ) } } }
$$

$$
4 3 . 2 4 _ { ( \uparrow 5 . 4 \mathrm { p p } ) }
$$

$$
{ \bf 3 4 . 7 1 } _ { ( \uparrow 1 . 2 \mathrm { p p } ) }
$$

$$
\mathbf { 7 6 . 1 0 } _ { ( \uparrow \uparrow . 0 \mathrm { p p } ) }
$$

$$
2 4 . 8 2 _ { ( \uparrow 1 . 5 \mathrm { p p } ) }
$$

$$
7 2 . 6 8 _ { ( \uparrow 0 . 5 \mathrm { p p ) } }
$$

$$
{ \bf 6 5 . 6 5 } _ { ( \uparrow 0 . 7 \mathrm { p p ) } }
$$

$$
\mathbf { 3 0 . 5 9 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
7 2 . 6 8 _ { ( \uparrow 0 . 5 \mathrm { p p ) } }
$$

$$
\mathbf { 7 0 . 9 2 } _ { ( \uparrow 0 . 8 \mathrm { p p } ) }
$$

$$
\mathbf { 5 3 . 5 6 } _ { ( \uparrow 0 . 5 \mathrm { p p } ) }
$$

$$
7 2 . 6 8 _ { ( \uparrow 0 . 5 \mathrm { p p ) } }
$$

$$
\mathbf { 7 8 . 0 5 } _ { ( \uparrow 1 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 5 8 . 2 6 } _ { ( \uparrow 2 . 2 \mathrm { p p } ) }
$$

$$
7 2 . 2 0 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
{ \bf 6 5 . 3 1 } _ { ( \uparrow 2 . 2 \mathrm { p p } ) }
$$

$$
\mathbf { 7 0 . 5 4 } _ { ( \uparrow 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 4 8 . 5 8 } _ { ( \uparrow 0 . 5 \mathrm { p p } ) }
$$

$$
{ \bf 9 1 . 8 9 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
9 4 . 8 6 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 8 8 . 3 3 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 9 7 . 3 0 _ { ( \uparrow 0 . 3 \mathrm { p p ) } } }
$$

$$
{ 7 8 . 8 9 } _ { ( \uparrow 2 . 2 \mathrm { p p } ) }
$$

$$
7 7 . 7 8 _ { ( \uparrow 0 . 6 \mathrm { p p } ) }
$$

$$
{ \bf 6 0 . 0 2 } _ { ( \uparrow 1 . 2 \mathrm { p p } ) }
$$

$$
\mathbf { 8 5 . 5 6 } _ { ( \uparrow 3 . 3 \mathrm { p p } ) }
$$

$$
7 8 . 3 3 _ { ( \uparrow 1 . 7 \mathrm { p p ) } }
$$

$$
\mathbf { 5 6 . 4 9 } _ { ( \uparrow 0 . 9 \mathrm { p p } ) }
$$

$$
\mathbf { 6 7 . 0 0 } _ { ( \uparrow \uparrow 6 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 6 1 . 1 1 _ { ( \uparrow 1 . 7 \mathrm { p p ) } } }
$$

$$
\mathbf { 6 3 . 0 0 } _ { ( \uparrow 2 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 5 7 . 3 0 } _ { ( \uparrow 1 . 1 \mathrm { p p } ) }
$$

$$
5 4 . 3 4 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 6 5 . 0 0 } _ { ( \uparrow \mathrm { 6 . 0 p p } ) }
$$

$$
\mathbf { 5 6 . 4 9 } _ { ( \uparrow 0 . 3 \mathrm { p p } ) }
$$

$$
{ \bf 6 2 . 0 0 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
{ \bf 6 6 . 0 0 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 9 1 . 6 7 } _ { ( \uparrow 1 . 7 \mathrm { p p } ) }
$$

$$
\mathbf { 5 3 . 0 0 } _ { ( \uparrow 1 . 0 \mathrm { p p } ) }
$$

$$
7 7 . 2 2 _ { ( \uparrow 2 . 8 \mathrm { p p } ) }
$$

$$
\mathbf { 8 3 . 3 3 } _ { ( \uparrow 0 . 6 \mathrm { p p } ) }
$$

$$
9 3 . 3 3 _ { ( \uparrow 1 . 7 \mathrm { p p ) } }
$$

$$
9 5 . 5 6 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
{ \bf 9 8 . 1 7 _ { ( \uparrow 0 . 2 \mathrm { p p ) } } }
$$

$$
\mathbf { 9 7 . 6 0 } _ { ( \uparrow 0 . 5 \mathrm { p p } ) }
$$

$$
\mathbf { 9 9 . 0 3 } _ { ( \uparrow 0 . 5 \mathrm { p p } ) }
$$

$$
{ \bf 9 8 . 1 7 _ { ( \uparrow 0 . 0 \mathrm { p p ) } } }
$$

$$
\mathbf { 8 9 . 6 0 } _ { ( \uparrow 1 . 7 \mathrm { p p } ) }
$$

$$
\mathbf { 8 8 . 4 4 } _ { ( \uparrow 3 . 5 \mathrm { p p } ) }
$$

$$
\mathbf { 9 0 . 7 5 } _ { ( \uparrow 3 . 5 \mathrm { p p } ) }
$$

$$
\mathbf { 1 1 . 2 7 } _ { ( \uparrow 0 . 2 \mathrm { p p } ) }
$$

$$
\mathbf { 6 8 . 7 9 } _ { ( \uparrow 4 . 6 \mathrm { p p } ) }
$$

$$
1 1 . 8 4 _ { ( \uparrow 0 . 7 \mathrm { p p } ) }
$$

$$
8 2 . 2 4 _ { ( \uparrow 1 . 3 \mathrm { p p } ) }
$$

$$
7 8 . 9 5 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 1 3 . 8 1 } _ { ( \uparrow 0 . 4 \mathrm { p p } ) }
$$

$$
3 2 . 1 2 _ { ( \uparrow 1 . 2 \mathrm { p p } ) }
$$

$$
\mathbf { 8 4 . 2 1 _ { ( \uparrow 0 . 7 \mathrm { p p ) } } }
$$

$$
\mathbf { 9 2 . 8 3 } _ { ( \uparrow 0 . 7 \mathrm { p p } ) }
$$

$$
\mathbf { 8 3 . 9 6 } _ { ( \uparrow 1 . 7 \mathrm { p p } ) }
$$

$$
7 7 . 6 3 _ { ( \uparrow 0 . 7 \mathrm { p p ) } }
$$

$$
\mathbf { 9 0 . 7 9 } _ { ( 0 . 0 \mathrm { p p } ) } ^ { \cdot }
$$

$$
\mathbf { 6 0 . 5 6 } _ { ( \uparrow 2 . 2 \mathrm { p p } ) }
$$

$$
\mathbf { 8 2 . 5 9 } _ { ( \uparrow 2 . 4 \mathrm { p p } ) }
$$

$$
\mathbf { 5 5 . 0 0 } _ { ( \uparrow 0 . 6 \mathrm { p p } ) }
$$

$$
9 2 . 8 3 _ { ( \uparrow 1 . 7 \mathrm { p p ) } }
$$

$$
\mathbf { 8 7 . 7 1 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
{ \bf 9 8 . 6 4 } _ { ( \uparrow 0 . 2 \mathrm { p p } ) }
$$

$$
5 3 . 8 9 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
{ \bf 9 7 . 5 0 _ { ( \uparrow 0 . 1 \mathrm { p p ) } } }
$$

$$
\mathbf { 5 6 . 1 1 } _ { ( \uparrow 0 . 6 \mathrm { p p } ) }
$$

$$
\mathbf { 5 9 . 4 4 } _ { ( \uparrow 1 . 1 \mathrm { p p } ) }
$$

$$
\mathbf { 4 0 . 0 0 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 9 9 . 5 0 } _ { ( \uparrow 0 . 1 \mathrm { p p } ) }
$$

$$
5 3 . 3 3 _ { ( \uparrow 6 . 7 \mathfrak { p p } ) }
$$

$$
{ \bf 9 8 . 7 3 _ { ( \uparrow 0 . 3 \mathrm { p p ) } } }
$$

$$
{ \bf 9 9 . 8 2 _ { ( \uparrow 0 . 1 \mathrm { p p ) } } }
$$

$$
7 3 . 3 3 _ { ( \uparrow 3 3 . 3 \mathrm { p p } ) }
$$

$$
\mathbf { 8 7 . 5 0 } _ { ( \uparrow 0 . 3 \mathrm { p p } ) }
$$

$$
\mathbf { 8 5 . 6 2 } _ { ( \uparrow 0 . 3 \mathrm { p p } ) }
$$

$$
4 6 . 6 7 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 6 0 . 0 0 } _ { ( \uparrow \cdot 2 0 . 0 \mathrm { p p } ) }
$$

$$
4 6 . 6 7 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
{ \bf 9 5 . 2 2 } _ { ( \uparrow 0 . 7 \mathrm { p p } ) }
$$

$$
\mathbf { 8 4 . 6 9 } _ { ( \uparrow 0 . 3 \mathrm { p p } ) }
$$

$$
\mathbf { 8 3 . 8 5 } _ { ( \uparrow 0 . 2 \mathrm { p p } ) }
$$

$$
\mathbf { 8 9 . 3 8 } _ { ( \uparrow 2 . 5 \mathrm { p p } ) }
$$

$$
\mathbf { 9 4 . 1 0 } _ { ( \uparrow 0 . 6 \mathrm { p p } ) }
$$

$$
\mathbf { 9 5 . 0 1 } _ { ( \uparrow 0 . 6 \mathrm { p p } ) }
$$

$$
\mathbf { 9 3 . 0 8 } _ { ( \uparrow 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 8 9 . 5 8 } _ { ( \uparrow 0 . 2 \mathrm { p p } ) }
$$

$$
\mathbf { 8 6 . 2 1 _ { ( \uparrow 0 . 6 \mathrm { p p } ) } }
$$

$$
\mathrm { W } / \mathrm { T } / \mathrm { L } = 1 5 8 / 3 4 / 0
$$

POINT-ADJUSTED ANOMALY-DETECTION PRECISION (P), RECALL (R), AND F1 SCORES ACROSS FOUR DATASETS (%). SHADED +SACM COLUMNS DENOTE SACM; BOLD MARKS HIGHER VALUES PER PAIR. AVG. ∆F1 REPORTS MACRO-AVERAGED F1 GAINS IN PERCENTAGE POINTS (PP).

$$
+ { \cal S } \mathbf { A } \mathbf { C } \mathbf { M }
$$

$$
{ \bf 7 0 . 2 7 _ { ( \uparrow 2 5 . 5 \mathrm { p p ) } } }
$$

$$
{ \bf 7 9 . 9 8 _ { ( \uparrow 2 6 . 3 \mathrm { p p ) } } }
$$

$$
\underline { { 9 6 . 8 0 } } _ { ( \downarrow 2 . 2 _ { \mathrm { P P } } ) }
$$

$$
\mathbf { 6 7 . 1 0 } _ { ( \uparrow 3 . 4 \mathrm { p p } ) }
$$

$$
\underline { { 8 6 . 8 4 } } _ { ( \downarrow 1 2 . 5 \mathrm { p p } ) }
$$

$$
\mathbf { 7 0 . 8 6 } _ { ( \uparrow 1 0 . 1 \mathrm { p p } ) }
$$

$$
\mathbf { 4 5 . 6 6 } _ { ( \uparrow 0 . 9 \mathrm { p p } ) }
$$

$$
{ \underline { { 6 6 . 1 7 } } } _ { ( \downarrow 0 . 5 \mathrm { p p } ) }
$$

$$
\mathbf { 7 9 . 6 1 } _ { ( \uparrow 1 . 1 \mathrm { p p } ) }
$$

$$
\mathbf { 8 1 . 4 3 } _ { ( \uparrow 1 9 . 8 \mathrm { p p } ) }
$$

$$
8 3 . 2 7 _ { ( \uparrow 1 3 . 5 \mathrm { p p } ) }
$$

$$
7 2 . 8 2 _ { ( \uparrow 2 . 5 \mathrm { p p } ) }
$$

$$
\underline { { 8 3 . 6 3 } } _ { ( \downarrow 1 . 4 \mathrm { p p } ) }
$$

$$
9 9 . 9 2 _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 7 4 . 4 0 } _ { ( \uparrow 0 . 1 \mathrm { p p } ) }
$$

$$
\mathbf { 8 4 . 4 6 } _ { ( \uparrow 1 . 3 \mathrm { p p } ) }
$$

$$
7 6 . 7 2 _ { ( \uparrow 5 . 9 \mathrm { p p } ) }
$$

$$
{ \bf 7 4 . 2 0 } _ { ( \uparrow 0 . 2 \mathrm { p p } ) }
$$

$$
{ \bf 6 2 . 6 8 } _ { ( \uparrow 0 . 8 \mathrm { p p ) } }
$$

$$
\mathbf { 8 8 . 2 2 } _ { ( \uparrow \mathrm { 5 . 2 p p } ) }
$$

$$
\mathbf { 7 0 . 4 4 } _ { ( \uparrow \uparrow 1 4 . 5 \mathrm { p p } ) }
$$

$$
\underline { { 7 7 . 6 1 } } _ { ( \downarrow 1 3 . 5 \mathrm { p p } ) }
$$

$$
{ \bf 7 6 . 1 8 _ { ( \uparrow 3 7 . 3 \mathrm { p p ) } } }
$$

$$
\mathbf { 8 6 . 6 4 } _ { ( \uparrow 4 0 . 9 \mathrm { p p } ) }
$$

$$
\mathbf { 8 2 . 5 6 } _ { ( \uparrow 1 . 6 \mathrm { p p } ) }
$$

$$
\mathbf { 9 6 . 7 7 _ { ( \uparrow 9 2 . 5 \mathrm { p p ) } } }
$$

$$
\mathbf { 8 1 . 7 6 } _ { ( \uparrow \mathrm { 2 8 . 8 p p } ) }
$$

$$
7 7 . 7 0 _ { ( \uparrow 2 7 . 4 \mathrm { p p } ) }
$$

$$
\mathbf { 9 2 . 6 4 } _ { ( \uparrow 1 0 . 0 _ { \mathrm { P p } } ) }
$$

$$
7 2 . 3 3 _ { ( \uparrow 3 8 . 3 \mathrm { p p ) } }
$$

$$
\mathbf { 9 4 . 9 4 } _ { ( \uparrow 4 1 . 2 \mathrm { p p } ) }
$$

$$
7 3 . 5 0 _ { ( \uparrow 6 5 . 7 \mathrm { p p ) } }
$$

$$
{ \bf 6 9 . 6 6 } _ { ( \uparrow 2 4 . 7 \mathrm { p p ) } }
$$

$$
\mathbf { 8 2 . 7 0 } _ { ( \uparrow 4 . 6 \mathrm { p p } ) }
$$

$$
\underline { { 6 1 . 9 3 } } _ { ( \downarrow 1 6 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 6 5 . 7 0 } _ { ( \uparrow \ 5 . 9 \mathrm { p p } ) }
$$

$$
7 8 . 7 2 _ { ( \uparrow 3 . 4 \mathrm { p p } ) }
$$

$$
\mathbf { 8 5 . 9 2 } _ { ( \uparrow \mathrm { 2 5 . 0 p p } ) }
$$

$$
7 6 . 5 4 _ { ( \uparrow 1 8 . 8 \mathrm { p p } ) }
$$

$$
\underline { { 6 7 . 7 7 } } _ { ( \downarrow 6 . 8 \mathrm { p p ) } }
$$

$$
\mathbf { 8 4 . 7 6 } _ { ( \uparrow 4 0 . 7 \mathrm { p p } ) }
$$

$$
\underline { { 6 0 . 6 7 } } _ { ( \downarrow 1 3 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 7 6 . 6 4 } _ { ( \uparrow 1 4 . 3 \mathrm { p p } ) }
$$

$$
7 2 . 5 3 _ { ( \uparrow 2 0 . 3 \mathrm { p p } ) }
$$

$$
\mathfrak { s o } \mathbf { 9 . 9 3 } _ { ( \uparrow 1 8 . 8 \mathrm { p p } ) }
$$

$$
\mathbf { 9 7 . 1 0 } _ { ( \uparrow 4 . 7 \mathrm { p p } ) }
$$

$$
\mathbf { 9 1 . 1 4 } _ { ( \uparrow \cdot 2 8 . 2 \mathrm { p p } ) }
$$

$$
\mathbf { 9 4 . 0 6 } _ { ( 0 . 0 \mathbf { \ p p } ) }
$$

$$
\mathbf { 7 0 . 7 1 } _ { ( \uparrow \uparrow 1 2 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 7 1 . 9 8 } _ { ( \uparrow 3 . 6 \mathrm { p p } ) }
$$

$$
\mathbf { 8 1 . 6 3 } _ { ( \uparrow \mathrm { 2 6 . 0 p p } ) }
$$

$$
7 2 . 8 5 _ { ( \uparrow 5 . 0 _ { \mathrm { P p } } ) }
$$

$$
\mathbf { 8 5 . 6 7 _ { ( \uparrow 1 1 . 2 _ { p p ) } } }
$$

$$
\mathbf { 8 1 . 9 0 } _ { ( \uparrow 1 4 . 7 \mathrm { p p } ) }
$$

$$
\mathbf { 9 7 . 9 3 _ { ( \uparrow 0 . 9 \mathrm { p p ) } } }
$$

$$
{ \underline { { 9 8 . 6 0 } } } ( \downarrow 1 . 1 \mathrm { p p } )
$$

$$
\mathbf { 9 9 . 6 8 } _ { ( \uparrow 0 . 5 \mathrm { p p } ) }
$$

$$
{ \underline { { 9 9 . 9 8 } } } _ { ( \downarrow 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 9 8 . 2 3 } _ { ( \uparrow 7 . 3 \mathrm { p p } ) }
$$

$$
9 3 . 7 7 _ { ( \uparrow 1 0 . 5 \mathrm { p p } ) }
$$

$$
\underline { { 9 9 . 8 8 } } _ { ( \downarrow 0 . 1 \mathrm { p p } ) }
$$

$$
\mathbf { 8 4 . 4 4 } _ { ( \uparrow 1 2 . 9 \mathrm { p p } ) }
$$

$$
\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 \mathrm { p p } ) }
$$

$$
\mathbf { 9 4 . 7 1 _ { ( \uparrow 1 . 6 \mathrm { p p ) } } }
$$

$$
\mathbf { 9 8 . 0 8 } _ { ( \uparrow 4 . 2 \mathrm { p p } ) }
$$

$$
7 6 . 5 9 _ { ( \uparrow 5 . 0 _ { \mathrm { P P } } ) }
$$

$$
\mathbf { 9 6 . 1 3 } _ { ( \uparrow \uparrow 5 . 4 \mathrm { p p } ) }
$$

$$
\mathbf { 9 1 . 4 3 } _ { ( \uparrow 8 . 3 \mathrm { p p } ) }
$$

$$
\mathbf { 6 9 . 7 1 _ { ( \uparrow 4 . 1 \mathrm { p p ) } } }
$$

$$
\mathbf { 7 6 . 5 6 } _ { ( \uparrow 1 2 . 9 \mathrm { p p } ) }
$$

$$
\mathbf { 8 6 . 7 3 } _ { ( \uparrow 3 . 3 \mathrm { p p } ) }
$$

$$
\mathbf { 8 6 . 7 2 _ { ( \uparrow 9 . 0 \mathrm { p p } ) } }
$$

$$
\mathbf { 9 7 . 2 5 } _ { ( \uparrow 0 . 8 \mathrm { p p } ) }
$$

$$
7 . 2 8 \ : \mathrm { p p } \ : \uparrow
$$

pp), NSFormer from 90.28% to 98.61% on Cricket (+8.33 pp), and PatchTST from 83.33% to 89.13% on Epilepsy (+5.80 pp). On FingerMovements, all six backbones improve, with gains of 2.00–6.00 pp, showing the benefit extends across architectures on the same task. Changes are smaller on SpokenArabicDigits, where Raw accuracy exceeds 97% for every backbone.

We evaluate SMD, MSL, SMAP, and PSM using matched splits and the detector’s task objective and anomaly score. For reconstruction-based models, the observed input window is the reconstruction target. Adaptive dropout acts on trainable activation paths and is disabled at evaluation. Raw and +SACM share the benchmark point-adjusted (PA) protocol [37, 39] for paired comparison; Supplementary Appendix A defines PA. Table IX reports precision, recall, and F1 for all seven backbones on four datasets. All 28 pairs improve under

Anomaly Detection. Spectral residual scores vary even among nominally normal training windows. We test whether this labelfree cue improves detection through adaptive regularization.

PA, with mean $\Delta \mathrm { F 1 } = + 1 1 . 8 0$ pp (macro F1: 69.24% to 81.04%). AnomTrans, NSFormer, and TranAD improve by 18.05, 16.05, and 9.61 pp. On MSL, iTransformer F1 rises from 53.00% to 81.76%, while NSFormer improves from 50.32% to 77.70%, showing substantial gains for multiple detectors on the same dataset. Macro means and changes are computed before rounding. The gains across detection objectives support the taskindependent role of the scorer: it supplies a training-time regularization signal while each detector retains its own scoring rule.

## F. RQ5: Ablations, Complementarity & Efficiency

Sample-Adaptive vs. Fixed Capacity Control. Table X compares the original backbone, validation-selected fixed dropout, learnable global dropout, and SACM. We use four backbones on $I L I \left( H = 4 8 \right)$ and Exchange Rate $( H = 9 6 )$ , with matched splits, initialization, and stopping rules. Validation MSE selects the fixed rate from $\{ 0 , 0 . 0 5 , \hdots , 0 . 5 0 \}$ ; the global variant learns one rate shared by all samples within [0.05, 0.50]. SACM improves MSE over fixed dropout in all eight settings and over global dropout in seven, tying TimeMixer on Exchange Rate at 0.101. Its MAE is lower than the learned-global variant in all eight settings. For TimeMixer on ILI, learning a global rate raises MSE from 3.055 to 3.217, whereas sampleadaptive rates yield 3.064. This contrast supports conditioning regularization on individual windows beyond adjusting its strength. Supplementary Appendix B also reports five-seed comparisons on three datasets and four backbones.

Component Attribution Analysis. We use the same Synth-12 benchmark as RQ1 to examine component effects under controlled corruption. All ablation variants are evaluated across its five corruption scales using the same signal construction and forecasting horizons. Table XI summarizes performance across these scales, while Supplementary Appendix B reports the complete per-scale results. The full configuration has MSE/MAE 1.076/0.797. Removing global linear detrending, the SFM anchor, or log-amplitude normalization raises MSE by 52.6%, 41.3%, and 21.6%, respectively, relative to the full configuration. All listed ablations also perform worse than fixed dropout $( p ~ = ~ 0 . 1 )$ on average. Detrending and spectral anchoring have the largest effects, supporting their roles in separating the residual from broad temporal structure and setting the filtering threshold. Their performance below fixed dropout highlights the importance of the scoring pipeline when adapting regularization. Endpoint-based detrending also trails OLS (MSE 1.574 vs. 1.076) with other components fixed.

TABLE XI  
COMPONENT ABLATION ON SYNTH-12 $( \sigma \in [ 0 . 1 , 0 . 9 ] )$ . GAIN DENOTES RELATIVE CHANGE VS. BASELINE $( p = 0 . 1 )$ .
<table><tr><td rowspan="2">Method</td><td colspan="3">Components</td><td colspan="2">MSE↓</td><td colspan="2">MAE↓</td></tr><tr><td>Detrend Norm log-SFM</td><td></td><td></td><td>Avg.</td><td>Gain (%)</td><td>Avg.</td><td>Gain (%)</td></tr><tr><td>Baseline</td><td>-</td><td>-</td><td>-</td><td>1.159</td><td></td><td>0.836</td><td>-</td></tr><tr><td>Minimal Model</td><td>None</td><td>x</td><td>x</td><td>1.514</td><td>↓30.6</td><td>0.960</td><td>↓14.8</td></tr><tr><td>w/o Detrend+Norm</td><td>None</td><td>x</td><td>√</td><td>1.468</td><td>↓26.7</td><td>0.942</td><td>↓12.7</td></tr><tr><td>w/o Detrend</td><td>None</td><td>√</td><td></td><td>1.642</td><td>↓41.7</td><td>1.014</td><td>↓21.3</td></tr><tr><td>Simple Detrend</td><td>Simple</td><td>√</td><td>√√</td><td>1.574</td><td>↓35.8</td><td>0.987</td><td>↓18.1</td></tr><tr><td>w/o Spectral Norm</td><td>OLS</td><td>x</td><td></td><td>1.308</td><td>↓12.9</td><td>0.896</td><td>↓7.2</td></tr><tr><td>w/o log-SFM Anchor</td><td>OLS</td><td>√</td><td>x</td><td>1.520</td><td>↓31.1</td><td>0.970</td><td>↓16.0</td></tr><tr><td>SACM (Ours)</td><td>OLS</td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>1.076</td><td>↑7.2</td><td>0.797</td><td>↑4.7</td></tr></table>

OLS: global ordinary least-squares linear detrending; Simple: endpoint-based linear detrending; Norm: log-amplitude normalization; log-SFM: spectral flatness measure anchoring. The Minimal Model omits all three listed components. The baseline uses a fixed dropout rate $( p = 0 . 1 )$ without adaptive scoring. $\mathrm { { \cdots } } { \mathrm { { \dot { w } } } } / \mathrm { { o } } ^ { \mathrm { { \prime } } \mathrm { { \prime } } }$ denotes “without”; $\checkmark / X$ indicate enabled/disabled components, respectively. Bold marks the lowest average error.

Hyperparameter Sensitivity. A larger γ increases dropout at a fixed normalized residual score and moves the rate mapper toward saturation. Figure 7 compares $\gamma \in \{ 1 , 5 , 1 0 \}$ across five corruption scales. Panel (a) shows non-monotonic responses to corruption, including a sharp MSE reduction for $\gamma = 1 0$ at $\sigma = 0 . 7 .$ . Panel (b) isolates the sensitivity effect: MSE peaks at $\gamma = 5$ for $\sigma \in \{ 0 . 5 , 0 . 7 , 0 . 9 \}$ but increases across settings for $\sigma \in \{ 0 . 1 , 0 . 3 \}$ . Thus, stronger masking affects corruption scales differently. The $\gamma = 1$ setting yields the lowest plotted MSE at four scales; the remaining experiments initialize $\gamma = 1$ Computational Efficiency. Table XII reports added parameters, latency, epochs to stopping, and total training time for matched Raw and +SACM runs. The scorer adds only 16 parameters in these seven-channel configurations. The reported time per epoch increases by 11.1% for TimeMixer and 29.3% for Informer. Both +SACM runs stop after 16 epochs, compared with 20 and 30 for their Raw counterparts. In these runs, the reduction in epoch count offsets the additional per-epoch cost, lowering total training time by 11.0% for TimeMixer (1.12×) and 31.0% for Informer $( 1 . 4 5 \times )$ . The auxiliary path incurs this cost only during training; inference retains the deterministic graph.

![](images/6bd101ab26fb54408da5468a50ebdfe1b61786664eee0676e540a38da7a958ce.jpg)

![](images/0f5ba7bf929eb5511842be0d44bf591ab47a8643896afdef60e2e099db1fde33.jpg)  
Fig. 7. Hyperparameter sensitivity analysis. (a) Effect of sensitivity parameter γ across corruption scales σ ∈ [0.1, 0.9]; (b) MSE grouped by sensitivity setting, with separate markers for the five corruption scales.

TABLE X  
DROPOUT STRATEGY COMPARISON ON FORECASTING BENCHMARKS $( H = 9 6$ FOR EXCHANGE RATE, $H = 4 8$ FOR ILI). ENTRIES REPORT TEST MSE/MAE. $p ^ { \star }$ IS VALIDATION-SELECTED; BOLD MARKS LOWEST MSE.
<table><tr><td colspan="7"> $I L I \ ( H { = } 4 8 )$ </td><td colspan="5">Exchange Rate  $\left( H { = } 9 6 \right)$ </td></tr><tr><td>Backbone</td><td> $p ^ { \star }$ </td><td>Raw</td><td>Best Fixed</td><td>Learned Global</td><td>SACM</td><td> $p ^ { \star }$ </td><td>Raw</td><td>Best Fixed</td><td>Learned Global</td><td>SACM</td></tr><tr><td>TimeFilter</td><td>0.100</td><td>2.466/0.974</td><td>2.619/1.012</td><td>2.469/0.980</td><td>2.461/0.968</td><td>0.000</td><td>0.108/0.232</td><td>0.108/0.232</td><td>0.106/0.230</td><td>0.105/0.229</td></tr><tr><td>PatchTST</td><td>0.000</td><td>2.939/1.099</td><td>2.956/1.111</td><td>2.879/1.095</td><td>2.874/1.093</td><td>0.100</td><td>0.107/0.230</td><td>0.108/0.231</td><td>0.109/0.230</td><td>0.107/0.229</td></tr><tr><td>TimeMixer</td><td>0.000</td><td>3.055/1.130</td><td>3.227/1.207</td><td>3.217/1.205</td><td>3.064/1.129</td><td>0.000</td><td>0.102/0.224</td><td>0.102/0.224</td><td>0.101/0.225</td><td>0.101/0.224</td></tr><tr><td>Informer</td><td>0.450</td><td>7.173/1.920</td><td>6.569/1.807</td><td>6.650/1.936</td><td>6.348/1.784</td><td>0.050</td><td>4.411/1.676</td><td>2.754/1.169</td><td>2.777/1.143</td><td>2.078/1.050</td></tr></table>

TABLE XII  
TRAINING COST FOR THE PAIRED RUNS (RAW → +SACM). EPOCHS DENOTES EPOCHS TO STOPPING; TOTAL TIME INCLUDES SACM OVERHEAD.
<table><tr><td></td><td colspan="2">Overhead (Cost)</td><td colspan="2">Training Cost</td><td colspan="2">Net Gain</td></tr><tr><td>Model</td><td>Params</td><td>s/Epoch</td><td>Epochs</td><td>Total (s)</td><td>Saved</td><td>Speedup</td></tr><tr><td>TimeMixer</td><td> $+ 1 6$ </td><td> $1 0 2 . 3  1 1 3 . 7 $ </td><td> $2 0  1 6$ </td><td> $2 0 4 5  1 8 1 9$ </td><td>11.0%</td><td>1.12×</td></tr><tr><td>Informer</td><td>+16</td><td> $9 5 . 8  1 2 3 . 9$ </td><td> $3 0  1 6$ </td><td> $2 8 7 3  1 9 8 2$ </td><td>31.0%</td><td>1.45×</td></tr></table>

Complementarity with Data-Centric Selection. Table XIII combines Selective Learning (SL) [20] with SACM on ILI using Informer, $L = 2 4$ , and H = 60. Adding SACM to SL reduces MSE from 6.461 to 6.336 (1.93%) and MAE from 1.902 to 1.774 (6.73%). Both variants incur the same 29.0 s selection-pretraining cost. Backbone training takes 15.1 s with SL and 6.5 s with SL+SACM, totaling 44.1 s and 35.5 s. The combined run saves 8.6 s versus SL alone, while remaining above the Raw baseline’s 20.0 s. Accuracy gains after SL support complementary roles: selection determines which time steps enter training, while SACM adjusts activation retention.

TABLE XIII  
COMPATIBILITY WITH SELECTIVE LEARNING (SL) AND COMPUTATIONAL COST ON ILI (L = 24, H = 60, INFORMER). REL. GAIN DENOTES RELATIVE MSE REDUCTION FROM RAW.
<table><tr><td colspan="5">Compatibility with SL</td></tr><tr><td>Method Strategy</td><td>Mechanism</td><td>Error Metric ↓</td><td></td><td>Rel.</td></tr><tr><td></td><td></td><td>MSE</td><td>MAE</td><td> $\Delta _ { \mathrm { { r e l } } } ~ ( \% )$ </td></tr><tr><td>Baseline (Raw)</td><td></td><td>7.140</td><td>1.916</td><td></td></tr><tr><td>SACM</td><td>Capacity Modulation</td><td>6.429</td><td>1.846</td><td>↑10.0</td></tr><tr><td>SL</td><td>Data Selection</td><td>6.461</td><td>1.902</td><td>↑9.5</td></tr><tr><td>SL + SACM (Ours) Combined</td><td></td><td>6.336</td><td>1.774</td><td>↑11.3</td></tr><tr><td colspan="5">Parameter Overhead</td></tr><tr><td>Method</td><td>Base Params</td><td>Additional</td><td>Total</td><td>Overhead (%)</td></tr><tr><td>Raw Baseline</td><td>11,799,047</td><td></td><td>11,799,047</td><td></td></tr><tr><td>SACM (Ours)</td><td>11,799,047</td><td>+16</td><td>11,799,063</td><td>1.36×10-4</td></tr><tr><td>SL</td><td>11,799,047</td><td>+3,000</td><td>11,802,047</td><td>0.025</td></tr><tr><td colspan="5">Training Time</td></tr><tr><td>Method</td><td>Train Time</td><td>Pre-train</td><td>Total</td><td>Time Saved (%)</td></tr><tr><td>Raw Baseline</td><td>20.0s</td><td>0s</td><td>20.0s</td><td></td></tr><tr><td>SACM</td><td>16.8s</td><td>0s</td><td>16.8s</td><td>↑16.0</td></tr><tr><td>SL</td><td>15.1s</td><td>29.0s</td><td>44.1s</td><td>↓120.5</td></tr><tr><td> $\mathrm { S A C M } + \mathrm { S L }$ </td><td>6.5s</td><td>29.0s</td><td>35.5s</td><td>↓77.5</td></tr></table>

Comparison with Data-Space Hard Denoising. Table XIV compares spectral processing as input filtering versus a regularization cue. On ILI, hard denoising (HD) applies lowpass filtering with matched horizons, split, and backbones. HD lowers Informer MAE from 1.901 to 1.869 but raises PatchTST and TimeMixer MAE to 1.162 and 1.197. SACM instead achieves 1.772, 1.064, and 1.142, improving over Raw on all three backbones. These results suggest that filtering can discard useful information, supporting separate scoring and prediction paths: spectral residuals guide training-time masking while the predictor retains the observed input.

TABLE XIV  
MAE COMPARISON WITH HARD DENOISING (HD) ON ILI. ∆ IS THE RELATIVE MAE CHANGE FROM RAW; BOLD MARKS THE LOWER ERROR AND ↑ DENOTES IMPROVEMENT.
<table><tr><td>Backbone</td><td>Raw</td><td>HD</td><td>SACM</td><td>Δ HD</td><td>∆ Ours</td></tr><tr><td>Informer</td><td>1.901</td><td>1.869</td><td>1.772</td><td>↑1.7%</td><td>↑6.8%</td></tr><tr><td>PatchTST</td><td>1.110</td><td>1.162</td><td>1.064</td><td>↓4.7%</td><td>↑4.1%</td></tr><tr><td>TimeMixer</td><td>1.148</td><td>1.197</td><td>1.142</td><td>↓4.3%</td><td>↑0.5%</td></tr><tr><td>Overall</td><td>-</td><td>-</td><td>-</td><td>↓2.4%</td><td>↑3.8%</td></tr></table>

Qualitative Forecast Behavior. Figure 8 shows four test windows from the 96-to-720 SyntheticTS noise-0.1 experiment using WPMixer. The windows were selected from the middle range of baseline errors, rather than the largest improvements. The relative MAE reductions are 7.0%, 8.8%, 17.1%, and 21.3% for panels (a)–(d), respectively. Across these examples, SACM reduces level and amplitude errors while following the broad temporal pattern of the clean target; residual differences remain around sharp spikes, trough depth, and fast oscillations.

![](images/774197efadef7e797ae14c0c93ba5f85ebacbb7f1180b75902232ae3bb207642.jpg)  
Fig. 8. Qualitative comparison of trajectories. Input context (gray), clean ground truth (green), Raw WPMixer (dashed red), and SACM (blue).

## VII. CONCLUSION

We present Capacity-Centric Modulation (CCM), which addresses sample-level reliability differences through adaptive regularization of activation paths. Its task-agnostic implementation, SACM, calibrates sample-wise dropout using labelfree spectral residuals. A bias–variance surrogate characterizes the excess risk of fixed allocation relative to an oracle with heterogeneous optimal rates. Across 301 dataset–backbone pairs, SACM improves forecasting, classification, and anomaly detection, with shorter training times in the profiled runs and no inference overhead. The gains persist across data regimes and transfer to unseen domains without target adaptation.

Limitations and Future Work. Spectral residuals may conflate informative broadband or impulsive dynamics with corruption (Appendix C). Future work includes hybrid time–frequency filtering for such non-harmonic signals, as well as batch-stable calibration and channel-specific modulation.

## REFERENCES

[1] Y. Li et al., “Towards long-term time-series forecasting: Feature, pattern, and distribution,” in Proc. IEEE 39th Int. Conf. Data Eng. (ICDE), 2023, pp. 1611–1624.

[2] R.-G. Cirstea, B. Yang, C. Guo, T. Kieu, and S. Pan, “Towards spatio-temporal aware traffic time series forecasting,” in Proc. IEEE 38th Int. Conf. Data Eng. (ICDE), 2022.

[3] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural Comput., vol. 9, no. 8, 1997.

[4] G. Lai, W.-C. Chang, Y. Yang, and H. Liu, “Modeling long- and short-term temporal patterns with deep neural networks,” in Proc. 41st ACM SIGIR Conf. Res. Dev. Inf. Retr., 2018, pp. 95–104.

[5] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “TimesNet: Temporal 2d-variation modeling for general time series analysis,” in Int. Conf. Learn. Represent., 2023.

[6] V. Ekambaram, A. Jati, N. Nguyen, P. Sinthong, and J. Kalagnanam, “TSMixer: Lightweight MLP-mixer model for multivariate time series forecasting,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2023.

[7] S. Wang et al., “TimeMixer: Decomposable multiscale mixing for time series forecasting,” in Int. Conf. Learn. Represent., 2024.

[8] H. Zhou et al., “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Proc. AAAI Conf. Artif. Intell., vol. 35, no. 12, 2021.

[9] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in Int. Conf. Learn. Represent., 2023.

[10] Y. Liu et al., “iTransformer: Inverted Transformers are effective for time series forecasting,” in Int. Conf. Learn. Represent., 2024.

[11] K. Rasul, C. Seward, I. Schuster, and R. Vollgraf, “Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting,” in Proc. 38th Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 139. PMLR, 2021, pp. 8857–8868.

[12] Y. Tashiro, J. Song, Y. Song, and S. Ermon, “CSDI: Conditional score-based diffusion models for probabilistic time series imputation,” in Adv. Neural Inf. Process. Syst., vol. 34, 2021.

[13] K. Yamanishi and J.-i. Takeuchi, “A unifying framework for detecting outliers and change points from nonstationary time series data,” in Proc. 8th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2002, pp. 676–681.

[14] A. Blazquez-Garc´ ´ıa, A. Conde, U. Mori, and J. A. Lozano, “A review on outlier/anomaly detection in time series data,” ACM Comput. Surv., vol. 54, no. 3, pp. 56:1–56:33, 2022.

[15] Y. Yang, D. Zhang, Y. Liang, H. Lu, G. Chen, and H. Li, “Not all data are good labels: On the self-supervised labeling for time series forecasting,” in Adv. Neural Inf. Process. Syst., vol. 38, 2025.

[16] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A simple way to prevent neural networks from overfitting,” J. Mach. Learn. Res., vol. 15, no. 56, pp. 1929–1958, 2014.

[17] S. Wager, S. Wang, and P. S. Liang, “Dropout training as adaptive regularization,” in Adv. Neural Inf. Process. Syst., vol. 26, 2013.

[18] Y. Gal, J. Hron, and A. Kendall, “Concrete dropout,” in Adv. Neural Inf. Process. Syst., vol. 30, 2017.

[19] H. Cheng, Q. Wen, Y. Liu, and L. Sun, “RobustTSF: Towards theory and design of robust time series forecasting with anomalies,” in Int. Conf. Learn. Represent., 2024.

[20] Y. Fu et al., “Selective learning for deep time series forecasting,” in Adv. Neural Inf. Process. Syst., 2025.

[21] Q. Pan, P. Yang, and J. Zhang, “BayesTSF: Measuring uncertainty estimation in industrial time series forecasting from a bayesian perspective,” in Proc. 20th Int. Conf. Adv. Intell. Comput. Technol. Appl. (ICIC), pt. II, ser. Lecture Notes in Computer Science, vol. 14863. Springer Nature Singapore, 2024, pp. 81–93.

[22] M. Chen, G. Pang, W. Wang, and C. Yan, “Information bottleneck-guided MLPs for robust spatial-temporal forecasting,” in 42nd Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 267. PMLR, 2025, pp. 8821–8855.

[23] D. P. Kingma and M. Welling, “Auto-encoding variational Bayes,” in Int. Conf. Learn. Represent., 2014.

[24] D. L. Donoho, “Compressed sensing,” IEEE Trans. Inf. Theory, vol. 52, no. 4, pp. 1289–1306, 2006.

[25] H. Ren et al., “Time-series anomaly detection service at Microsoft,” in Proc. 25th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2019, pp. 3009–3017.

[26] S. Zhong et al., “DropoutTS: Sample-adaptive dropout for robust time series forecasting,” in Proc. 43rd Int. Conf. Mach. Learn., 2026.

[27] Q. Ma et al., “A survey on time-series pre-trained models,” IEEE Trans. Knowl. Data Eng., vol. 36, no. 12, 2024.

[28] H. Wu, J. Xu, J. Wang, and M. Long, “Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting,” Adv. Neural Inf. Process. Syst., vol. 34, 2021.

[29] T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin, “FEDformer: Frequency enhanced decomposed Transformer for long-term series forecasting,” in Proc. 39th Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 162. PMLR, 2022, pp. 27 268–27 286.

[30] Y. Zhang and J. Yan, “Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting,” in 11th Int. Conf. Learn. Represent., 2023.

[31] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?” in Proc. AAAI Conf. Artif. Intell., vol. 37, no. 9, 2023, pp. 11 121–11 128.

[32] H. Ismail Fawaz et al., “InceptionTime: Finding AlexNet for time series classification,” Data Min. Knowl. Discov., vol. 34, no. 6, pp. 1936–1962, 2020.

[33] A. Dempster, D. F. Schmidt, and G. I. Webb, “MiniRocket: A very fast (almost) deterministic transform for time series classification,” in Proc. 27th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2021, pp. 248–257.

[34] H. Zhang, Y.-F. Zhang, Z. Zhang, Q. Wen, and L. Wang, “LogoRA: Local-global representation alignment for robust time series classification,” IEEE Trans. Knowl. Data Eng., vol. 36, no. 12, pp. 8718–8729, 2024.

[35] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, “Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2018.

[36] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, “Robust anomaly detection for multivariate time series through stochastic recurrent neural network,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2019.

[37] J. Xu, H. Wu, J. Wang, and M. Long, “Anomaly transformer: Time series anomaly detection with association discrepancy,” in Int. Conf. Learn. Represent., 2022.

[38] S. Tuli, G. Casale, and N. R. Jennings, “TranAD: Deep transformer networks for anomaly detection in multivari ate time series data,” Proc. VLDB Endow., 2022.

[39] Y. Yang, C. Zhang, T. Zhou, Q. Wen, and L. Sun, “DCdetector: Dual attention contrastive representation learning for time series anomaly detection,” in Proc. 29th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2023.

[40] T. Kieu et al., “Robust and explainable autoencoders for unsupervised time series outlier detection,” in Proc. IEEE 38th Int. Conf. Data Eng. (ICDE), 2022, pp. 3038–3050.

[41] R. Wu and E. J. Keogh, “Current time series anomaly detection benchmarks are flawed and are creating the illusion of progress,” IEEE Trans. Knowl. Data Eng., vol. 35, no. 3, pp. 2421–2429, 2023.

[42] Q. Zhou, S. He, H. Liu, J. Chen, and W. Meng, “Labelfree multivariate time series anomaly detection,” IEEE Trans. Knowl. Data Eng., vol. 36, no. 7, 2024.

[43] O. Wu and R. Yao, “Data optimization in deep learning: A survey,” IEEE Trans. Knowl. Data Eng., 2025.

[44] X. Liang, Y. Ji, W.-S. Zheng, W. Zuo, and X. Zhu, “SV-Learner: Support-vector contrastive learning for robust learning with noisy labels,” IEEE Trans. Knowl. Data Eng., vol. 36, no. 10, pp. 5409–5422, 2024.

[45] T. Zhou et al., “FiLM: Frequency improved legendre memory model for long-term time series forecasting,” in Adv. Neural Inf. Process. Syst., vol. 35, 2022.

[46] Y. Hu et al., “TimeFilter: Patch-specific spatial-temporal graph filtration for time series forecasting,” in 42nd Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 267, 2025, pp. 24 893–24 911.

[47] K. Yan, C. Long, H. Wu, and Z. Wen, “Multi-resolution expansion of analysis in time-frequency domain for time series forecasting,” IEEE Trans. Knowl. Data Eng., vol. 36, no. 11, pp. 6667–6680, 2024.

[48] D. P. Kingma, T. Salimans, and M. Welling, “Variational dropout and the local reparameterization trick,” in Adv. Neural Inf. Process. Syst., vol. 28, 2015.

[49] E. J. Candes and T. Tao, “Near-optimal signal recovery from random projections: Universal encoding strategies?” IEEE Trans. Inf. Theory, vol. 52, no. 12, 2006.

[50] Y. Bengio, N. Leonard, and A. Courville, “Estimating ´ or propagating gradients through stochastic neurons for conditional computation,” arXiv:1308.3432, 2013.

[51] M. M. N. Murad, M. Aktukmak, and Y. Yilmaz, “WP-Mixer: Efficient multi-resolution mixing for long-term time series forecasting,” in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 18, 2025, pp. 19 581–19 588.

[52] V. Naghashi, M. Boukadoum, and A. B. Diallo, “A multiscale model for multivariate time series forecasting,” Sci. Rep., vol. 15, no. 1, p. 1565, 2025.

[53] A. L. Goldberger et al., “PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals,” Circulation, vol. 101, no. 23, pp. e215–e220, 2000.

[54] A. Bagnall et al., “The UEA multivariate time series classification archive, 2018,” arXiv:1811.00075, 2018.

[55] D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, “A public domain dataset for human activity recognition using smartphones,” in Proc. Eur. Symp. Artif. Neural Netw. (ESANN), 2013, pp. 437–442.

[56] B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen, and J. J. L. Oberye, “Analysis of a sleepdependent neuronal feedback loop: The slow-wave microcontinuity of the EEG,” IEEE Trans. Biomed. Eng., vol. 47, no. 9, pp. 1185–1194, 2000.

[57] Y. Liu, H. Wu, J. Wang, and M. Long, “Non-stationary transformers: Exploring the stationarity in time series forecasting,” Adv. Neural Inf. Process. Syst., vol. 35, 2022.

[58] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Int. Conf. Learn. Represent., 2015.

![](images/4c1bfb2bf00d3ea6f4cb4a5e2ba823e4c3f914a4f3b62cb350bfe198cc24a10b.jpg)  
Siru Zhong is a Ph.D. candidate with the Data Science and Analytics Thrust, HKUST (GZ), Guangzhou, China. His research interests include spatio-temporal intelligence and time series modeling. He has authored or coauthored more than 20 papers in ICML, NeurIPS, KDD, ACM MM, and AAAI. He gave tutorials at ACM MM 2025, ICME 2026, and AAAI 2026, and received the DSA Excellent Research Award in 2025 and 2026.

![](images/15490a18b08ce9e60e311fa081a3743b06a2327f8d7021046e68e2ddfb28e9cf.jpg)

Senzhang Wang (Member, IEEE) received the Ph.D. degree from Beihang University, Beijing, China, in 2015. He is currently a Professor with the School of Computer Science and Engineering, Central South University, Changsha. He has authored or coauthored more than 100 papers in top journals and conferences, such as IEEE TKDE, ACM TOIS, KDD, NeurIPS, ICLR, IJCAI, and AAAI. His research interests include graph mining, spatio-temporal data mining, and time series analysis.

![](images/4716cca665d236d8260490e1d3a9091f60cb3b6034b870ec1eb49f83e4b79223.jpg)

James T. Kwok (Fellow, IEEE) is currently a Professor with the Department of Computer Science and Engineering at HKUST, Hong Kong. His research interests include machine learning and artificial intelligence. He has served as an Associate Editor of several journals including the IEEE Transactions on Neural Networks and Learning Systems, Neural Networks, Neurocomputing, and Artificial Intelligence. He has also served as a Senior Area Chair of major conferences including NeurIPS, ICML, ICLR, and IJCAI. He is a member of the IJCAI Board of Trustees, and served as the Program Chair of IJCAI 2025 and the General Chair of PAKDD 2026.

![](images/efc3b09c4cdccebe4d01491a9dba44b07c40f62227f952231aadc8441e182155.jpg)

Yuxuan Liang is an Assistant Professor at HKUST (GZ). He obtained his Ph.D. from the National University of Singapore. His research focuses on spatio-temporal AI and foundation models, and he has authored over 100 papers with 16,000 citations. He serves as the Vice Chair of the IEEE CIS NNTC Task Force on Spatio-Temporal Data and Time Series, Associate Editor of Neurocomputing, and Area Chair of major AI conferences including NeurIPS, ICML, ICLR, KDD, ACL, and MM. He received Best

Paper (or Runner-up) Awards, e.g., ICDM 2025, the ACM SIGSPATIAL China Rising Star Award, and the SDSC Dissertation Research Fellowship.

APPENDICES OVERVIEW

The appendices provide the experimental specifications, complete forecasting results, spectral diagnostics, detrending analysis, and theoretical proofs supporting the main paper. The entries below are clickable in the electronic version.

<table><tr><td>Part</td><td>Section</td><td>Contents</td><td>Page</td></tr><tr><td>Appendix A</td><td>Experimental Details</td><td>Benchmarks, protocols, notation, and evaluation metrics</td><td>15</td></tr><tr><td>Appendix B</td><td>Complete Results</td><td>Per-horizon forecasting results, fixed-rate sweep, ablations, and batch-size sensitivity</td><td>17</td></tr><tr><td>Appendix C</td><td>Spectral Diagnostics and Real-world Sparsity</td><td>Component-wise diagnostics and top-coefficient reconstructions</td><td>20</td></tr><tr><td>Appendix D</td><td>Detrending Analysis</td><td>Spectral leakage, Edge-MAE, and failure modes</td><td>23</td></tr><tr><td>Appendix E</td><td>Theory and Proofs</td><td>Surrogate-risk assumptions and excess-risk theorem proof</td><td>27</td></tr></table>

## APPENDIX A EXPERIMENTAL DETAILS

## A. Benchmark Details

We evaluate forecasting, classification, and anomaly detection benchmarks spanning energy, meteorology, finance, healthcare, human activity, speech, and operational telemetry, covering both large-scale real-world series and synthetic stress tests.

Forecasting benchmarks. The four ETT variants contain oil temperature and six power-load variables at hourly or 15-minute resolution. Weather, Electricity, and Exchange Rate represent meteorological, high-dimensional demand, and non-stationary financial series, respectively. ILI contains weekly influenza-like-illness ratios, while ECG concatenates both leads from MIT-BIH records 100–105 [1] into a 12-channel series, downsampled to 36 Hz, truncated to 65,000 points. Table XV summarizes dimensions, splits, frequencies, and horizons. Forecasting datasets use the chronological train/validation/test splits from BasicTS [2].

Classification benchmarks. We use 30 multivariate UEA-style datasets [3], together with HAR [4] and Sleep-EDF [1, 5]. The index below lists each benchmark’s modality and prediction target.

Anomaly-detection benchmarks. We use the task-specific train/validation/test splits of four multivariate telemetry datasets: SMD [6] contains server-machine metrics and labeled abnormal intervals; MSL and SMAP [7] contain spacecraft telemetry and anomaly events; and PSM [8] contains production-server metrics with service-level anomalies. Each reconstruction model follows its prescribed training split and benchmark evaluation protocol, with the input window as the reconstruction target. Each backbone retains its native reconstruction head, and SACM is applied to the trainable activation paths before that head. The benchmark-specific window composition, label usage, and thresholding follow the corresponding public benchmark protocol Table XVI summarizes the dimensionality, split sizes, and test-split anomaly rates of the four benchmarks.

## B. Unified Experimental Protocol

Backbones and task settings. The forecasting study covers Informer, Crossformer, PatchTST, iTransformer, MultiPatchFormer, TimeMixer, WPMixer, TimesNet, and TimeFilter. Classification uses iTransformer, PatchTST, NSFormer, TimesNet, InceptionTime, and MiniROCKET; anomaly detection uses PatchTST, iTransformer, NSFormer, AnomTrans, TranAD, DCdetector, and TimesNet. For forecasting, the input length is L = 24 on ILI and $L = 9 6$ on all other datasets. The evaluated horizons are $H \in \{ 2 4 , 3 6 , 4 8 , 6 0 \}$ for ILI, $H \in \{ 3 6 , 7 2 , 1 4 4 , 2 8 8 \}$ for ECG, and $H \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ for the remaining forecasting datasets. On Synth-12, forecasts are evaluated against the clean target trajectory rather than its corrupted observation.

Optimization and integration. All models are implemented in PyTorch, optimized with Adam, and run on NVIDIA A800 GPUs. We retain each backbone’s default hyperparameters so that paired Raw and +SACM variants differ only in the capacity modulation strategy. SACM recursively replaces dropout modules on trainable paths; models without them are unchanged. All adaptive layers share the sample-level probability vector, but draw independent Bernoulli activation masks. The spectral residua scorer and capacity modulation module are active strictly during training and are completely bypassed during evaluation, so SACM follows the same deterministic inference path as the Raw baseline with no additional computation or latency.

Capacity modulation settings and controlled comparisons. Unless varied in the sensitivity study, we fix $[ p _ { \mathrm { m i n } } , p _ { \mathrm { m a x } } ] =$ [0.05, 0.50] and initialize the learnable spectral-mask steepness at $\alpha = 1 0$ , threshold bias at $\mathbf { b } _ { s } = 0 .$ , and rate-mapping sensitivity at $\gamma = 1$ . In the fixed-dropout comparison, candidate rates are $p \in \{ 0 , 0 . 0 5 , \ldots , 0 . 5 0 \}$ and $p ^ { \star }$ is selected by validation MSE only. The paired-seed study uses seeds 2022–2026 with matched initialization, data order, early stopping, and hyperparameters between Raw and +SACM.

Roles of the remaining hyperparameters. The fixed dropout bounds define the range over which the learned mapper allocates regularization. The lower bound $p _ { \mathrm { m i n } } = 0 . 0 5$ maintains mild regularization even at zero normalized residual score. The upper bound $p _ { \operatorname* { m a x } } = 0 . 5 0$ keeps the expected retention fraction $1 - p _ { i }$ at least 0.50 and the inverted-dropout scaling $1 / ( 1 - p _ { i } )$ at most 2. The attainable maximum also depends on $\gamma .$ . At its default initialization, a normalized score of 1 gives $p _ { i } = 0 . 0 5 + 0 . 4 5$ tanh(softplus(1)) ≈ 0.439. These are properties of the rate mapper, rather than measured retention statistics

TABLE XV  
STATISTICS OF THE NINE REAL-WORLD FORECASTING BENCHMARKS. COLUMNS REPORT DIMENSIONALITY, TIME STEPS, CHRONOLOGICAL TRAIN/VAL/TEST SPLIT, SAMPLING FREQUENCY, PREDICTION HORIZONS, AND DOMAIN; ILI AND ECG USE TASK-SPECIFIC SHORTER HORIZONS.
<table><tr><td>Dataset</td><td>Dim.</td><td>Size</td><td>Split</td><td>Frequency</td><td>Prediction Length</td><td>Domain</td></tr><tr><td>ETTh1</td><td>77</td><td>14,400</td><td>6:2:2</td><td>1 hour</td><td>{96, 192, 336, 720}</td><td>Temperature</td></tr><tr><td>ETTh2</td><td></td><td>14,400</td><td>6:2:2</td><td>1 hour</td><td>{96, 192, 336, 720}</td><td>Temperature</td></tr><tr><td>ETTm1</td><td>7</td><td>57,600</td><td>6:2:2</td><td>15 min</td><td>{96, 192, 336, 720}</td><td>Temperature</td></tr><tr><td>ETTm2</td><td></td><td>57,600</td><td>6:2:2</td><td>15 min</td><td>{96, 192, 336, 720}</td><td>Temperature</td></tr><tr><td>Weather</td><td>21</td><td>52,696</td><td>7:1:2</td><td>10 min</td><td>{96, 192, 336, 720}</td><td>Meteorology</td></tr><tr><td>Electricity</td><td>321</td><td>26,304</td><td>7:1:2</td><td>1 hour</td><td>{96, 192, 336, 720}</td><td>Electricity</td></tr><tr><td>Exchange Rate</td><td>8</td><td>7,588</td><td>7:1:2</td><td>1 day</td><td>{96, 192, 336, 720}</td><td>Finance</td></tr><tr><td>ILI</td><td>7</td><td>966</td><td>7:1:2</td><td>1 week</td><td>{24, 36, 48, 60}</td><td>Health</td></tr><tr><td>ECG</td><td>12</td><td>65,000</td><td>7:1:2</td><td>36 Hz</td><td>{36, 72, 144, 288}</td><td>Health</td></tr></table>

TABLE XVI

STATISTICS OF THE FOUR ANOMALY-DETECTION BENCHMARKS. COLUMNS REPORT DIMENSIONALITY, TRAINING AND TEST TIME STEPS, ANOMALY RATIO IN THE TEST SPLIT, AND DOMAIN. EVALUATION FOLLOWS THE POINT-ADJUSTED PROTOCOL DEFINED IN APPENDIX A-C.
<table><tr><td>Dataset</td><td>Dim.</td><td>Train</td><td>Test</td><td>Anomaly Rate</td><td>Domain</td></tr><tr><td>SMD</td><td>38</td><td>708,405</td><td>708,420</td><td>4.16%</td><td>Server machine</td></tr><tr><td>MSL</td><td>55</td><td>58,317</td><td>73,729</td><td>10.72%</td><td>Spacecraft</td></tr><tr><td>SMAP</td><td>25</td><td>135,183</td><td>427,617</td><td>13.13%</td><td>Spacecraft</td></tr><tr><td>PSM</td><td>25</td><td>132,481</td><td>87,841</td><td>27.76%</td><td>Server machine</td></tr></table>

The learnable scalar α controls spectral-mask sharpness. $\mathrm { W i t h ~ } \alpha = 1 0 .$ , the sigmoid moves from 0.1 to 0.9 over a normalized amplitude interval of $2 \ln ( 9 ) / \mathrm { s o f t p l u s } ( 1 0 ) \approx 0 . 4 4$ around $\tau _ { c } ,$ , giving a smooth initial separation of spectral components. A smaller slope weakens this separation, while a larger slope approaches hard thresholding and reduces mask gradients away from $\tau _ { c }$ . Initializing $\mathbf { b } _ { s } = 0$ leaves the initial threshold logits determined by $w _ { s , c } \mathrm { S F M } _ { c }$ . Each channel’s weight and bias, together with α and $\gamma ,$ are updated through the task loss, so the thresholds and sharpness adapt during training.

The constant $\epsilon = 1 0 ^ { - 8 }$ stabilizes spectral normalization and logarithms. It also determines when nearly identical batch scores use the default normalized score 0.5. Learning rate, batch size, and stopping settings follow each backbone’s default configuration and are matched between paired Raw and +SACM runs, as described above. The sensitivity experiment varies γ; the remaining settings describe the shared default configuration.

Result notation. Raw denotes the unmodified backbone and +SACM denotes the same backbone augmented with SACM. For an error metric e (lower is better) and a score metric m (higher is better), relative improvement is computed as

$$
\Delta _ { \mathrm { r e l } } e = 1 0 0 \frac { e ^ { \mathrm { R a w } } - e ^ { \mathrm { S A C M } } } { e ^ { \mathrm { R a w } } } , \qquad \Delta _ { \mathrm { r e l } } m = 1 0 0 \frac { m ^ { \mathrm { S A C M } } - m ^ { \mathrm { R a w } } } { m ^ { \mathrm { R a w } } } .\tag{22}
$$

A positive value therefore denotes an improvement in both cases. For accuracy and F1 reported as percentages, an absolute change is stated in percentage points (pp) rather than relative percent. W/T/L counts pairwise wins, ties, and losses under the metric named in the corresponding table caption, aggregated over the reported model–dataset pairs.

## C. Evaluation Metrics

Forecasting. For $N _ { \mathrm { f } }$ test windows, prediction horizon H, and $C$ target channels, let $y _ { i , h , c }$ and $\hat { y } _ { i , h , c }$ denote the target and prediction for sample i, horizon step $h ,$ and channel c. We report mean squared error (MSE) and mean absolute error (MAE):

$$
\mathrm { M S E } = \frac { 1 } { N _ { \mathrm { f } } H C } \sum _ { i = 1 } ^ { N _ { \mathrm { f } } } \sum _ { h = 1 } ^ { H } \sum _ { c = 1 } ^ { C } \left( y _ { i , h , c } - \hat { y } _ { i , h , c } \right) ^ { 2 } ,\tag{23}
$$

$$
\mathrm { M A E } = \frac { 1 } { N _ { \mathrm { f } } H C } \sum _ { i = 1 } ^ { N _ { \mathrm { f } } } \sum _ { h = 1 } ^ { H } \sum _ { c = 1 } ^ { C } \left| y _ { i , h , c } - \hat { y } _ { i , h , c } \right| .\tag{24}
$$

Lower values indicate better forecasting performance.

Classification. For $N _ { \mathrm { c } }$ test samples with ground-truth class $z _ { i }$ and predicted class $\hat { z } _ { i } .$ , classification accuracy is

$$
\mathrm { A c c u r a c y } = \frac { 1 } { N _ { \mathrm { c } } } \sum _ { i = 1 } ^ { N _ { \mathrm { c } } } \mathbb { I } ( z _ { i } = \hat { z } _ { i } ) ,\tag{25}
$$

where I(·) is the indicator function.

<table><tr><td>Dataset</td><td>Sequence/target</td><td>Dataset</td><td>Sequence/target</td></tr><tr><td>ArticularyWordRecognition</td><td>Articulatory trajectories; spoken words</td><td>AtrialFibrillation</td><td>ECG; cardiac rhythm</td></tr><tr><td>BasicMotions</td><td>Wearable sensors; basic motions</td><td>CharacterTrajectories</td><td>Pen-tip trajectories; characters</td></tr><tr><td>Cricket</td><td>Motion sensors; umpire gestures</td><td>DuckDuckGeese</td><td>Audio features; bird calls</td></tr><tr><td>EigenWorms</td><td>Posture sequences; locomotion</td><td>Epilepsy</td><td>Wrist acceleration; seizure-like motions</td></tr><tr><td>EthanolConcentration</td><td>Spectral sequences; concentration</td><td>ERing</td><td>Ring-sensor trajectories; gestures</td></tr><tr><td>FaceDetection</td><td>Brain signals; face detection</td><td>FingerMovements</td><td>Brain signals; finger movements</td></tr><tr><td>HandMovementDirection</td><td>Brain signals; movement direction</td><td>Handwriting</td><td>Motion trajectories; characters</td></tr><tr><td>Heartbeat</td><td>Cardiac auscultation; heartbeat class</td><td>InsectWingbeat</td><td>Wingbeat signals; insect species</td></tr><tr><td>JapaneseVowels</td><td>Speech trajectories; speakers</td><td>Libras</td><td>Movement trajectories; sign language</td></tr><tr><td>LSST</td><td>Astronomical light curves; object class</td><td>MotorImagery</td><td>Brain signals; motor imagery</td></tr><tr><td>NATOPS</td><td>Motion sequences; handling signals</td><td>PenDigits</td><td>Pen trajectories; digits</td></tr><tr><td>PEMS-SF</td><td>Traffic sensors; traffic state</td><td>PhonemeSpectra</td><td>Speech spectra; phonemes</td></tr><tr><td>RacketSports</td><td>Wearable sensors; sport actions</td><td>SelfRegulationSCP1</td><td>Cortical potentials; self-regulation</td></tr><tr><td>SelfRegulationSCP2</td><td>Cortical potentials; self-regulation</td><td>SpokenArabicDigits</td><td>Speech sequences; digits</td></tr><tr><td>StandWalkJump</td><td>Multichannel sensors; activities</td><td>UWaveGestureLibrary</td><td>Acceleration; gestures</td></tr><tr><td>HAR</td><td>Smartphone inertial signals; activities</td><td>Sleep-EDF</td><td>Polysomnography; sleep stages</td></tr></table>

Anomaly Detection. Let $a _ { t } \in \{ 0 , 1 \}$ be the ground-truth label at time step t, and let $\tilde { a } _ { t } \in \{ 0 , 1 \}$ be the binary prediction from an anomaly score using the detector’s evaluation threshold. For each contiguous ground-truth anomaly interval $I _ { m } .$ , point adjustment marks the entire interval as detected when at least one point in that interval is predicted anomalous:

$$
\begin{array} { r } { \hat { a } _ { t } ^ { \mathrm { P A } } = \left\{ \begin{array} { l l } { 1 , } & { t \in I _ { m } \mathrm { ~ a n d ~ } \sum _ { u \in I _ { m } } \tilde { a } _ { u } > 0 \mathrm { ~ f o r ~ s o m e ~ } m , } \\ { \tilde { a } _ { t } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{26}
$$

Using the adjusted predictions, the point-level counts are

$$
\mathrm { T P } = \sum _ { t } a _ { t } \hat { a } _ { t } ^ { \mathsf { P A } } , \qquad \mathrm { F P } = \sum _ { t } ( 1 - a _ { t } ) \hat { a } _ { t } ^ { \mathsf { P A } } , \qquad \mathrm { F N } = \sum _ { t } a _ { t } ( 1 - \hat { a } _ { t } ^ { \mathsf { P A } } ) .\tag{27}
$$

We then report point-adjusted precision, recall, and F1 score:

$$
\mathrm { P r e c i s i o n = \frac { T P } { T P + \mathrm { F P } } , ~ } \mathrm { R e c a l l = \frac { T P } { T P + \mathrm { F N } } , ~ } \mathrm { F 1 = \frac { 2 \ p r e c i s i o n ~ R e c a l l } { P r e c i s i o n + R e c a l l } . }\tag{28}
$$

For K dataset–backbone configurations, reported macro averages and average gains use unweighted arithmetic means:

$$
\mathrm { M a c r o } ( m ) = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } m _ { j } ,
$$

$$
\mathrm { A v g . ~ } \Delta m = \frac { 1 } { K } \sum _ { j = 1 } ^ { K } \left( m _ { j } ^ { \mathrm { { S A C M } } } - m _ { j } ^ { \mathrm { { R a w } } } \right) ,\tag{29}
$$

where $m _ { j }$ is the relevant metric for configuration j; percentage-point gains are reported when $m _ { j }$ is expressed as a percentage.

## APPENDIX B

## COMPLETE RESULTS

This section reports the horizon-level forecasting measurements underlying the averages in the main paper. Unless stated otherwise, each Raw/+SACM pair uses the same backbone configuration and data split. Up and down arrows in Tables XVII and XVIII follow the relative-improvement convention in Appendix A-B and summarize horizon-averaged changes.

## A. Results of Synth-12

Table XVII expands the main-paper averages into MSE/MAE for each of five corruption scales $\sigma \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ and four horizons $H \in \{ 9 6 , 1 9 2 , 3 3 6 , 7 2 0 \}$ . Evaluation uses clean targets. The horizon-level entries expose variation with forecasting length that is hidden by the main-paper averages.

## B. Results of Open Benchmarks

Table XVIII reports the individual forecasting horizons for nine backbones on nine real-world benchmarks. These entries complement the main-paper summary by showing where gains, ties, and regressions occur at particular horizons.

TABLE XVII  
COMPLETE PER-HORIZON FORECASTING RESULTS ON SYNTH-12 FOR NINE BACKBONES. EACH HORIZON REPORTS RAW AND SACM (+SACM) MSE/MAE; LOWER ERRORS ARE BOLD AND HIGHER ERRORS ARE UNDERLINED. UP AND DOWN ARROWS REPORT RELATIVE IMPROVEMENTS AND REGRESSIONS, RESPECTIVELY, AVERAGED OVER HORIZONS.
<table><tr><td rowspan=2 colspan=25>Informer         Crossformer        PatchTST         TimesNet        iTransformer       TimeMixer        WPMixer        TimeFilter      MultiPatchFormer[9]            [10]            [11]           [12]           [13]           [14]           [15]           [16]           [17]Raw   +SACM   Raw   +SACM   Raw   +SACM  Raw   +SACM  Raw   +SACM  Raw   +SACM  Raw   +SACM   Raw  +SACM  Raw   +SACMMSE MAEMSEMAEMSE MAEMSEMAEMSE MAEMSEMAEMSE MAEMSEMAE</td></tr><tr><td rowspan=1 colspan=14></td></tr><tr><td rowspan=1 colspan=2>96  1.214 0.8540.695 0.683192  0.811 0.7220.4790.561</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>336 0.723 0.6880.43601             0.445720 1.114 0.837</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>Avg 0.966 0.7750.514Improv. (∆)      46.8%↑2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>5.1%↑</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4.3%</td><td rowspan=1 colspan=1>6.1%↑↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.5%1</td><td rowspan=1 colspan=1>1.2%</td><td rowspan=1 colspan=2>5.9% ↑</td><td rowspan=1 colspan=1>3.2%↑</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>.8%↑</td><td rowspan=1 colspan=1>0.6%↑</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>.9% ↑ 0</td><td rowspan=1 colspan=1>.2% ↑</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>.6%↑</td><td rowspan=1 colspan=1>2.3%↑</td><td rowspan=1 colspan=2>2.3%↑</td><td rowspan=1 colspan=1>1.5%↑</td><td rowspan=1 colspan=1>4.3%↑↑</td><td rowspan=1 colspan=1>2.2%↑</td></tr><tr><td rowspan=1 colspan=1>96  1.281 0.8750.693</td><td rowspan=1 colspan=1>0.679</td><td rowspan=1 colspan=1>0.243 0.373</td><td rowspan=1 colspan=1>0.209</td><td rowspan=1 colspan=1>0.334</td><td rowspan=1 colspan=1>0.282 0.401</td><td rowspan=1 colspan=1>0.273</td><td rowspan=1 colspan=1>0.393</td><td rowspan=1 colspan=2>0.486 0.5650.458</td><td rowspan=1 colspan=1>0.545</td><td rowspan=1 colspan=1>0.278 0.392</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=2>03 336 0.718 0.6860.436 0.5337201.102 0.8330.4250.536</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.499 0.554</td><td rowspan=1 colspan=1>0.479</td><td rowspan=1 colspan=1>0.543</td><td rowspan=1 colspan=1>0.664 0.646</td><td rowspan=1 colspan=1>0.656</td><td rowspan=1 colspan=1>0.642</td><td rowspan=1 colspan=2>0.980 0.8020.934</td><td rowspan=1 colspan=1>0.780</td><td rowspan=1 colspan=2>0.667 0.6460.661</td><td rowspan=1 colspan=1>0.636</td><td rowspan=1 colspan=1>0.661 0.641</td><td rowspan=1 colspan=1>0.659</td><td rowspan=1 colspan=1>0.635</td><td rowspan=1 colspan=1>0.663 0.6390</td><td rowspan=1 colspan=1>.653</td><td rowspan=1 colspan=1>0.631</td><td rowspan=1 colspan=2>0.683 0.6550.678</td><td rowspan=1 colspan=1>0.653</td><td rowspan=1 colspan=2>0.654 0.6370.6430.631</td></tr><tr><td rowspan=1 colspan=2>Avg 0.978 0.7790.5070.577</td><td rowspan=1 colspan=1>0.439 0.495</td><td rowspan=1 colspan=1>0.378</td><td rowspan=1 colspan=1>0.464</td><td rowspan=1 colspan=1>0.571 0.580</td><td rowspan=1 colspan=1>0.543</td><td rowspan=1 colspan=1>0.564</td><td rowspan=1 colspan=2>0.847 0.7430.821</td><td rowspan=1 colspan=1>0.730</td><td rowspan=1 colspan=2>0.554 0.5660.548</td><td rowspan=1 colspan=1>0.5600.553</td><td rowspan=1 colspan=1>0.5650</td><td rowspan=1 colspan=1>.551</td><td rowspan=1 colspan=1>0.561</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>Improv. (∆)      48.2% ↑ 26.0% ↑</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3.8% ↑</td><td rowspan=1 colspan=1>6.3% ↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4.9%↑ 2</td><td rowspan=1 colspan=1>.6%↑</td><td rowspan=1 colspan=2>3.1%↑ 1</td><td rowspan=1 colspan=1>.7%↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.0%1</td><td rowspan=1 colspan=1>1.1%</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.5%</td><td rowspan=1 colspan=1>0.7%</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>.3%↑ 1</td><td rowspan=1 colspan=1>.5%↑</td><td rowspan=1 colspan=2>1.0% ↑</td><td rowspan=1 colspan=1>0.4% ↑</td><td rowspan=1 colspan=1>1.6% ↑</td><td rowspan=1 colspan=1>0.9% ↑</td></tr><tr><td rowspan=1 colspan=2>96  1.195 0.8460.6640.665192 0.782 0.7090.4460.541336 0.702 0.6760.4490.549</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>720 1.064 0.8190.418 0.528</td><td rowspan=1 colspan=1>0.468 0.537</td><td rowspan=1 colspan=1>0.493</td><td rowspan=1 colspan=1>0.562</td><td rowspan=1 colspan=1>0.683 0.656</td><td rowspan=1 colspan=1>0.656</td><td rowspan=1 colspan=1>0.643</td><td rowspan=1 colspan=1>0.936 0.783</td><td rowspan=1 colspan=1>0.912</td><td rowspan=1 colspan=1>0.772</td><td rowspan=1 colspan=1>0.666 0.649</td><td rowspan=1 colspan=1>0.663</td><td rowspan=1 colspan=1>0.648</td><td rowspan=1 colspan=1>0.659 0.641</td><td rowspan=1 colspan=1>0.651</td><td rowspan=1 colspan=1>0.632</td><td rowspan=1 colspan=1>0.658 0.634</td><td rowspan=1 colspan=1>0.650</td><td rowspan=1 colspan=1>0.631</td><td rowspan=1 colspan=1>0.691 0.661</td><td rowspan=1 colspan=1>0.683</td><td rowspan=1 colspan=1>0.659</td><td rowspan=1 colspan=1>0.675 0.6550.651</td><td rowspan=1 colspan=1>0.636</td></tr><tr><td rowspan=1 colspan=2>Improv. (∆)      47.2%↑25.1%↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9.3%↑</td><td rowspan=1 colspan=1>3.1%↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.8% ↑</td><td rowspan=1 colspan=1>1.5%↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.6%↑</td><td rowspan=1 colspan=1>1.1%</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.2%</td><td rowspan=1 colspan=1>0.9%</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>.0% ↑</td><td rowspan=1 colspan=1>0.7% ↑</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>.2% ↑</td><td rowspan=1 colspan=1>0.5%↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.1% ↑</td><td rowspan=1 colspan=1>0.7% ↑</td><td rowspan=1 colspan=1>5.4%↑</td><td rowspan=1 colspan=1>3.3%↑</td></tr><tr><td rowspan=1 colspan=2>96  0.943 0.7580.6240.648192 0.687 0.6670.433 0.536</td><td rowspan=1 colspan=1>0.245 0.367</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.338</td><td rowspan=1 colspan=1>0.298 0.409</td><td rowspan=1 colspan=1>0.287</td><td rowspan=1 colspan=1>0.401</td><td rowspan=1 colspan=1>0.476 0.5660</td><td rowspan=1 colspan=1>.444</td><td rowspan=1 colspan=1>0.543</td><td rowspan=1 colspan=1>0.290 0.402</td><td rowspan=1 colspan=1>0.285</td><td rowspan=1 colspan=1>0.397</td><td rowspan=1 colspan=1>0.288 0.397</td><td rowspan=1 colspan=1>0.280</td><td rowspan=1 colspan=1>0.392</td><td rowspan=1 colspan=1>0.280 0.3900</td><td rowspan=1 colspan=1>.277</td><td rowspan=1 colspan=1>0.389</td><td rowspan=1 colspan=1>0.296 0.412</td><td rowspan=1 colspan=1>0.292</td><td rowspan=1 colspan=1>0.407</td><td rowspan=1 colspan=1>0.288 0.4000.287</td><td rowspan=1 colspan=1>0.399</td></tr><tr><td rowspan=1 colspan=1>07  336 0.671 0.6610.4147201.061 0.8180.411</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>Avg  0.841 0.7260.470Improv. (∆)      44.0%↑</td><td rowspan=1 colspan=1>0.558</td><td rowspan=1 colspan=1>0.411 0.481</td><td rowspan=1 colspan=1>0.370</td><td rowspan=1 colspan=1>0.435</td><td rowspan=1 colspan=1>0.571 0.584</td><td rowspan=1 colspan=1>0.561</td><td rowspan=1 colspan=1>0.577</td><td rowspan=1 colspan=1>0.785 0.721</td><td rowspan=1 colspan=1>0.763</td><td rowspan=1 colspan=1>0.707</td><td rowspan=1 colspan=1>0.556 0.578</td><td rowspan=1 colspan=1>0.549</td><td rowspan=1 colspan=1>0.572</td><td rowspan=1 colspan=1>0.552 0.567</td><td rowspan=1 colspan=1>0.548</td><td rowspan=1 colspan=1>0.564</td><td rowspan=1 colspan=1>0.542 0.559</td><td rowspan=1 colspan=1>0.537</td><td rowspan=1 colspan=1>0.556</td><td rowspan=1 colspan=1>0.576 0.591</td><td rowspan=1 colspan=1>0.571</td><td rowspan=1 colspan=1>0.588</td><td rowspan=1 colspan=1>0.573 0.5850.550</td><td rowspan=1 colspan=1>0.573</td></tr><tr><td rowspan=1 colspan=1>23.1%↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9.9%↑</td><td rowspan=1 colspan=1>9.5%↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.8% ↑ 1</td><td rowspan=1 colspan=1>.2% ↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.8% ↑ 1</td><td rowspan=1 colspan=1>.9% ↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.2%</td><td rowspan=1 colspan=1>1.1%</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>.8% ↑</td><td rowspan=1 colspan=1>0.4% ↑</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>.8% ↑</td><td rowspan=1 colspan=1>0.4% ↑</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8%1</td><td rowspan=1 colspan=1>0.5%</td><td rowspan=1 colspan=1>4.0% ↑</td><td rowspan=1 colspan=1>2.0%↑</td></tr><tr><td rowspan=1 colspan=1>96  1.093 0.8080.617192  0.652 0.6500.400</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=2>09  336  0.629 0.6400.386 0.5027200.939 0.7670.4530.545</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>0.516 0.561</td><td rowspan=2 colspan=2>0.4480.537</td><td rowspan=2 colspan=1>0.652 0.643</td><td rowspan=2 colspan=1>0.625</td><td rowspan=2 colspan=1>0.631</td><td rowspan=2 colspan=2>0.850 0.7460.806</td><td rowspan=2 colspan=1>0.727</td><td rowspan=2 colspan=1>0.622 0.629</td><td rowspan=2 colspan=1>0.620</td><td rowspan=2 colspan=1>0.628</td><td rowspan=2 colspan=1>0.626 0.626</td><td rowspan=2 colspan=1>0.619</td><td rowspan=2 colspan=1>0.621</td><td rowspan=2 colspan=1>0.604 0.611</td><td rowspan=2 colspan=1>0.603</td><td rowspan=2 colspan=1>0.610</td><td rowspan=2 colspan=1>0.637 0.638</td><td rowspan=2 colspan=1>0.632</td><td rowspan=2 colspan=1>0.634</td><td rowspan=2 colspan=1>0.628 0.6320.622</td><td></td></tr><tr><td rowspan=1 colspan=1>0.6500.630</td></tr><tr><td rowspan=1 colspan=2>Avg 0.828 0.7160.464 0.550Improv. (∆)      44.0%↑ 23.2% ↑</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

TABLE XVIII

COMPLETE PER-HORIZON REAL-WORLD FORECASTING RESULTS FOR NINE BACKBONES. EACH HORIZON REPORTS RAW AND SACM (+SACM) MSE/MAE; LOWER ERRORS ARE BOLD AND HIGHER ERRORS ARE UNDERLINED. UP AND DOWN ARROWS REPORT RELATIVE IMPROVEMENTS AND REGRESSIONS, RESPECTIVELY, AVERAGED OVER HORIZONS.
<table><tr><td></td><td></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td></td><td>Imp. (∆) 96 8.137 2.167 4.557 1.490</td><td></td><td>0.760 19.3% ↑ 11.0% ↑</td><td>0.413 0.404</td><td>0.394 0.392 4.7%↑ 3.0%↑</td><td>0.396 0.387</td><td>0.393 0.385 0.8%↑ 0.4%↑</td><td>0.519 0.474 0.512 1.4%↑ 0.9%↑</td><td>0.470 0.399 0.390</td><td>0.399 0.390 0.393 0.385 0.2% ↑ 0.1% ↑</td><td>0.392 0.4%↑</td><td>0.383 0.391 0.385 0.6% ↑</td><td>0.388 0.384 0.394 0.386 0.9% ↑ 0.4% ↑</td><td>0.393 0.385 0.4% ↑ 0.2% ↑</td><td colspan="2">0.402 0.397</td></tr><tr><td>Em2</td><td>32 192 0.351 0.429 0.384 0.425 Avg 3.357 1.128</td><td></td><td>5.555 1.739 0.333 0.413 0.268 0.350 0.381 0.451</td><td>0.181 0.264 0.237 0.306 0.313 0.367 0.784 0.642</td><td>0.178 0.264 0.244 0.311 0.331 0.380 0.650 0.556</td><td>0.175 0.252 0.240 0.296 0.301 0.335 0.402 0.394</td><td>0.175 0.252 0.240 0.296 0.301 0.335 0.402 0.394</td><td>0.207 0.282 0.206 0.282 0.289 0.332 0.289 0.332 0.362 0.378 0.359 0.376 0.534 0.463 0.510 0.451</td><td>0.180 0.257 0.245 0.298 0.305 0.336 0.406 0.394</td><td>0.180 0.256 0.173 0.250 0.245 0.298 0.304 0.336 0.300 0.333 0.406 0.394</td><td>0.173 0.239 0.295 0.239 0.300 0.398 0.391 0.398</td><td>0.250 0.172 0.251 0.295 0.237 0.294 0.333 0.295 0.332</td><td>0.172 0.250 0.175 0.254 0.236 0.293 0.240 0.297 0.295 0.332 0.307 0.339</td><td>0.175 0.254 0.240 0.297 0.306 0.338</td><td colspan="2">0.399 0.7%↑ 0.176 0.256 0.175 0.246 0.303 0.243 0.304 0.339</td></tr><tr><td>Wweater</td><td>Imp. (∆) 96 1.094 0.753 192 0.294 0.328 0.260 0.312</td><td>0.236 0.286</td><td>1.634 0.738 51.3% ↑ 34.5%↑ 0.763 0.596</td><td>0.379 0.395 0.166 0.204 0.208 0.244</td><td>0.351 0.378 7.4%↑ 4.3%↑ 0.162 0.200</td><td>0.279 0.319 0.173 0.208 0.215 0.248</td><td>0.279 0.319 0.348 0.364 0.0%↑ 0.0%↑ 0.165 0.202 0.210 0.243</td><td>0.341 2.0%↑ 1.0%↑ 0.181 0.227 0.184 0.230</td><td>0.360 0.284 0.321 0.185 0.217 0.281 0.235 0.258</td><td>0.284 0.321 0.277 0.317 0.1% ↑ 0.1% ↑ 0.177 0.209</td><td>0.277 0.0%↑ 0.180 0.226 0.164</td><td>0.391 0.395 0.391 0.317 0.275 0.317 0.0% ↑ 0.202 0.162 0.200</td><td>0.394 0.390 0.274 0.316 0.283 0.321 0.2%↑ 0.2%↑ 0.162 0.199 0.162 0.201</td><td>0.410 0.396 0.410 0.396 0.283 0.321 0.1% ↑ 0.1% ↑ 0.158</td><td>0.412 0.402 0.284 0.325 0.197 0.168 0.203</td><td>0.302 0.338 0.409 0.401 0.282 0.324 0.7%↑ 0.4%↑ 0.165 0.199 0.214 0.245</td></tr><tr><td></td><td>32 0.289 0.336 Avg 0.484 0.432 Imp. (∆) 96 1.478 0.965</td><td>0.257 0.267 0.305 0.381 0.373 21.4% ↑ 13.8%↑</td><td>0.304</td><td>0.260 0.286 0.259 0.331 0.339 0.330 0.241 0.268 0.239</td><td>0.205 0.240 0.285 0.339 0.266 0.9%↑ 0.8%↑</td><td>0.269 0.288 0.269 0.345 0.337 0.343 0.251 0.270 0.247 1.5% ↑</td><td>0.286 0.334 0.266 0.289 0.305 1.5%↑</td><td>0.246 0.283 0.245 0.311 0.324 0.305 0.321 0.417 0.387 0.403 0.378 0.284</td><td>0.291 0.299 0.362 0.345 0.302 0.268 0.280</td><td>0.232 0.256 0.287 0.296 0.362 0.345 0.265 0.276 0.282 0.301</td><td>0.246 0.282 0.209 0.309 0.323 0.263 0.395 0.375 0.339 0.244</td><td>0.242 0.208 0.241 0.281 0.263 0.282 0.332 0.341 0.333 0.264 0.243 0.264</td><td>0.207 0.240 0.210 0.244 0.262 0.281 0.266 0.285 0.339 0.332 0.346 0.337 0.242 0.263 0.246 0.267</td><td>0.208 0.241 0.265 0.284 0.345 0.336 0.244</td><td colspan="2">0.217 0.248 0.272 0.289 0.271 0.351 0.341 0.264 0.252 0.270</td></tr><tr><td>Electiy</td><td>192 1.540 0.982 320 1.412 0.951 0.263 1.682 1.058 0.267</td><td>1.162 0.832 0.264 0.333 0.332 0.340</td><td></td><td>0.192 0.267 0.198 0.275 0.214 0.295 0.249 0.324</td><td>0.191 0.264 0.194 0.273 0.211 0.290 0.248 0.323</td><td>0.194 0.268 0.199 0.276 0.214 0.292 0.255 0.326</td><td>0.184 0.262 0.189 0.268 0.213 0.291 0.254 0.324</td><td>1.6%↑ 0.9%↑ 0.174 0.279 0.169 0.274 0.188 0.288 0.186 0.287 0.210 0.307 0.208 0.305 0.249</td><td>0.193 0.267 0.198 0.214 0.292 0.257 0.327</td><td>1.4% ↑ 1.2% ↑ 0.187 0.219 0.198 0.212 0.290</td><td>0.195 0.263 0.196 0.256 0.193 0.267 0.194 0.209 0.283 0.209</td><td>13.7% ↑ 12.4%↑ 0.264 0.183 0.262 0.266 0.191 0.270 0.283 0.208 0.286</td><td>0.4%↑ 0.4%↑ 0.174 0.256 0.181 0.263 0.198 0.281 0.217 0.295</td><td>0.9%↑ 0.8%↑ 0.203 0.274 0.203 0.274 0.201 0.277 0.201 0.277 0.217 0.295</td><td>0.319 0.380 0.204 0.278</td><td>0.349 0.340 0.250 0.268 1.0%↑ 0.9%↑ 0.291 0.366 0.203 0.276 0.348 0.406 0.344 0.406</td></tr><tr><td></td><td>Avg 1.528 0.989 Imp. (∆) 7.005 1.868</td><td>0.489 0.459 0.213 0.290 68.0%↑ 53.6%↑ 6.235 1.713</td><td></td><td></td><td>0.211 0.287 1.1% ↑ 0.9%↑</td><td>0.215 0.290</td><td>0.210 0.286 0.206 0.304 2.6%↑ 1.5%↑ 2.599</td><td>0.252 0.341 0.203 0.301 1.5%↑ 0.9%↑ 1.389</td><td>0.338 0.215 0.290</td><td>0.256 0.326 0.213 1.0%↑ 6.0%↑</td><td>0.249 0.315 0.248 0.273 0.211 0.282 0.212 0.1%↓</td><td>0.315 0.250 0.320 0.282 0.208 0.284 0.0%↑</td><td>0.240 0.314 0.198 0.278 0.220 0.294 4.7%↑ 2.1%↑</td><td>0.260 0.330 0.260 0.220 0.0%↑</td><td colspan="2">0.362 0.416 0.394 0.435 0.330 0.294 0.320 0.377 0.0%↑ 2.933 0.938</td></tr><tr><td>m</td><td>2436 48 0 7.201 1.898 7.173 1.920 7.140 1.916</td><td>5.927 1.743 5.153 1.561 6.348 1.784 5.244 1.576 6.429 1.846 5.006 1.550</td><td></td><td>4.736 1.480</td><td>4.555 1.419 4.115 1.375 4.031 1.359 4.584 1.471</td><td>3.633 1.079 4.019 1.192 2.939 1.099 2.695 1.071</td><td>0.955 9.241 3.620 1.152 7.371 2.874 1.093 4.175 2.674 1.057</td><td>7.288 1.336 1.438 5.751 1.336 1.237 3.272 1.120 1.149 3.177 1.101</td><td>3.507 1.071 3.974 1.152 2.513 1.005</td><td>3.068 1.025 3.841 1.133 2.468 0.991 2.598 1.043</td><td>3.124 1.136 3.116 3.538 1.214 3.455 3.055 1.130 3.064</td><td>1.135 3.173 1.022 1.190 3.720 1.129 2.709 1.046</td><td>3.018 0.987 1.903 0.859 1.147 3.701 1.129 2.8961.044 2.596 1.018 .4660.974 2.591</td><td>1.720 0.821 2.369 0.962 2.461 0.968</td><td>3.178 1.044 0.975</td><td>0.297 0.364 7.2%↑ 3.6%↑ 2.110 0.897 2.914 1.056 2.722 1.036 2.452 1.015</td></tr><tr><td></td><td>Avg 7.130 1.901 Imp. (∆) 2.078</td><td>6.235 1.772 12.6%↑ 6.8%↑</td><td>5.035 1.542 4.321 14.2% ↑</td><td></td><td>1.406 8.8%↑</td><td>3.321 1.110 2.942 11.4%</td><td>3.392 1.064 6.045 1.303 4.1%↑</td><td>4.872 1.223</td><td>2.657 1.049 3.163 1.069</td><td>3.010 2.994 1.048</td><td>1.114 2.967 3.182 1.148 3.151 1.0%↑</td><td>1.113 2.722 1.044 1.142 3.081 1.065</td><td>1.017 2.361 0.983 2.977 1.038 2.407 0.965</td><td>2.342 2.223 0.931</td><td colspan="2">2.802 1.001 2.303 0.982 2.804 0.991</td></tr><tr><td>2</td><td>96 4.411 1.676 192 336 720 3.782</td><td>0.287 0.358 1.005 0.708 1.432 1.265 0.855 1.035</td><td></td><td>0.284 0.530 0.839 1.863 1.749</td><td>0.358 0.510 0.659 0.978 0.626</td><td>0.107 0.230 0.216 0.334 0.434 0.477 1.074 0.786</td><td>0.107 0.229 0.215 0.333 0.423 0.473 1.032 0.772</td><td>19.4% ↑ 6.1%↑ 0.129 0.262 0.233 0.361 0.416 0.483</td><td></td><td>5.3%↑ 2.0%↑ 0.103 0.227 0.205 0.326 0.394 0.458</td><td>0.102 0.224 0.101 0.204 0.325 0.204 0.385 0.450 0.384</td><td>0.6%↑ 0.224 0.102 0.224 0.325 0.204 0.325</td><td>3.4%↑ 2.5%↑</td><td>7.6%↑ 3.5%↑ 0.105 0.210</td><td>0.113 0.239 0.225 0.346 0.494 0.509</td><td>2.550 1.001 9.1%↑ 1.0%↓ 0.113 0.239 0.225 0.346 0.413 0.472</td></tr><tr><td rowspan="2">EG</td><td>2.849 1.232 1.178 3.091 1.301 1.216</td><td>1.050 0.782 0.578 0.527 0.813</td><td></td><td></td><td></td><td></td><td>0.130 0.263 0.233 0.361 0.415 0.482 1.135 0.806</td><td>1.128 0.812</td><td>0.103 0.228 0.206 0.327 0.405 0.468 1.090 0.798</td><td>1.052 0.780</td><td>1.087 0.786 1.076</td><td>0.450 0.393 0.454 0.782</td><td>0.098 0.220 0.201 0.320 0.391 0.453 1.073 0.786</td><td>0.108 0.232 0.213 0.332 0.396 0.456 0.394 1.0680.788 1.065</td><td rowspan="2">0.229 0.328 0.455 0.781 1.274</td><td rowspan="2"></td></tr><tr><td>Avg Imp. (∆) 32 1.742 1.109 1.167 0.728 1.183 0.748</td><td>3.533 1.410 1.434 0.875 59.4%↑</td><td>0.933 0.657 38.0%↑ 1.151</td><td>0.723 0.568 0.437 0.634 0.715 0.520 0.692 0.728 0.774 0.553 0.737</td><td>0.851 8.9%↑ 4.7%↑ 0.523</td><td>0.458 0.457 0.415 0.611 0.458 0.509 0.730 0.521 0.537 0.801 0.561 0.775</td><td>0.444 0.452 2.9%↑ 1.1% ↑ 0.574 0.436 0.706 0.509</td><td>0.478 0.478 0.477 0.3%↑ 0.637 0.469 0.627 0.717 0.520 0.716</td><td>0.479 0.451 0.455 0.3% 0.463 0.611 0.455 0.518 0.731 0.521</td><td>0.438 0.448 2.8%↑ 1.6%↑ 0.572 0.439 0.722 0.516 0.688 0.501</td><td>0.445 0.446 0.441 0.7%↑ 0.561 0.434 0.561</td><td>1.081 0.792 0.445 0.445 0.449 0.2% ↑ 0.432 0.597 0.444 0.711 0.510</td><td>0.441 0.445 0.446 0.452 0.444 0.9%↑ 0.9%↑ 0.6%↑ 0.529 0.413 0.600 0.447 0.678 0.492</td><td>0.448 0.8%↑ 0.597 0.443 0.7130.510 0.705 0.506 0.785 0.552 0.778 0.548</td><td>0.850 1.127 0.526 0.486 0.469 10.8%↑ 0.605 0.450 0.564 0.731 0.522 0.722 0.799 0.561 0.792</td><td>0.797 0.463 4.6%↑ 0.433 0.516 0.558 0.866 0.594 0.855 0.590 0.750 0.532 0.733 0.524</td></tr></table>

TABLE XIX  
PAIRED-SEED STABILITY ACROSS FIVE RANDOM SEEDS (2022–2026). MSE ENTRIES SHOW MEAN ± STD. W/T/L DENOTES WINS/TIES/LOSSES.
<table><tr><td></td><td colspan="5">ETTh1 (L=96, H=96)</td><td colspan="5"> $I L I \ ( L { = } 2 4 , H { = } 2 4 )$ </td><td colspan="5">Exchange Rate (L=96, H=96)</td></tr><tr><td>Backbone</td><td>Raw MSE</td><td>+SACM MSE</td><td>Gain (%)</td><td>W/T/L</td><td>p</td><td>Raw MSE</td><td>+SACM MSE</td><td>Gain (%)</td><td>W/T/L</td><td>p</td><td>Raw MSE</td><td>+SACM MSE</td><td>Gain (%)</td><td>W/T/L</td><td>p</td></tr><tr><td>Informer</td><td> $1 . 2 6 5 8 \pm 0 . 1 8 7 9$ </td><td> $\mathbf { 1 . 0 6 5 8 \pm 0 . 0 4 5 3 }$ </td><td>↑15.8</td><td>4/0/1</td><td>0.076</td><td> $6 . 2 6 0 5 \pm 0 . 7 2 9 3$ </td><td> ${ \bf 5 . 8 4 4 2 \pm 0 . 2 1 2 4 }$ </td><td>↑6.7</td><td>3/0/2</td><td>0.301</td><td> $4 . 0 6 7 1 \pm 0 . 8 4 4 9$ </td><td>1.2747 ± 0.3771</td><td>↑68.7</td><td>5/0/0</td><td>0.003</td></tr><tr><td>PatchTST</td><td> $0 . 3 9 4 3 \pm 0 . 0 0 4 7$ </td><td> $\mathbf { 0 . 3 8 9 7 \pm 0 . 0 0 3 9 }$ </td><td>↑1.2</td><td>4/0/1</td><td>0.056</td><td> $3 . 3 9 9 7 \pm 0 . 3 3 7 3$ </td><td> $\mathbf { 2 . 8 3 8 4 \pm 0 . 1 9 5 3 }$ </td><td>↑16.5</td><td>5/0/0</td><td>0.004</td><td> $0 . 1 0 4 0 \pm 0 . 0 0 1 8$ </td><td> $\mathbf { 0 . 1 0 2 2 } \pm 0 . 0 0 1 4$ </td><td>↑1.7</td><td>5/0/0</td><td>0.005</td></tr><tr><td>TimeMixer</td><td> $0 . 3 9 4 6 \pm 0 . 0 0 2 5$ </td><td> $\mathbf { 0 . 3 9 2 0 \pm 0 . 0 0 2 0 }$ </td><td>↑0.7</td><td>5/0/0</td><td>0.007</td><td> $3 . 3 9 0 4 \pm 0 . 1 8 4 4$ </td><td> $\mathbf { 3 . 3 5 8 2 \pm 0 . 3 4 7 3 }$ </td><td>↑1.0</td><td>2/0/3</td><td>0.764</td><td> $0 . 1 0 2 4 \pm 0 . 0 0 0 5$ </td><td> $\mathbf { 0 . 1 0 1 7 \pm 0 . 0 0 0 7 }$ </td><td>↑0.7</td><td>4/0/1</td><td>0.166</td></tr><tr><td>TimeFilter</td><td> $0 . 3 8 9 5 \pm 0 . 0 0 1 0$ </td><td> $\mathbf { 0 . 3 8 9 1 \pm 0 . 0 0 0 9 }$ </td><td>↑0.1</td><td>4/0/1</td><td>0.104</td><td> $1 . 8 6 5 2 \pm 0 . 1 7 3 8$ </td><td> $\mathbf { 1 . 8 2 2 7 \pm 0 . 1 5 6 5 }$ </td><td>↑2.3</td><td>4/0/1</td><td>0.550</td><td>0.1044 ± 0.0014</td><td>0.1033 ± 0.0014</td><td>↑1.0</td><td>5/0/0</td><td>0.124</td></tr></table>

## C. Paired-Seed Stability

Table XIX reports a representative multi-seed check on three datasets and four backbones. Each Raw/+SACM pair uses five matched seeds (2022–2026) with identical initialization, data order, early stopping, and hyperparameters. Mean MSE gains are positive in all twelve settings; several configurations reach $p < 0 . 0 5$ under a paired comparison, while others remain directionally consistent but not significant at this sample size.

## D. Complete Fixed-Dropout Sweep

Table XX reports test results for every candidate rate from 0 to 0.50 in increments of 0.05 on the representative horizons used in the main comparison. Validation MSE selects the bold cell $p ^ { \star }$ , while test values are reported for every completed fixed-rate checkpoint; selection is therefore independent of test performance.

TABLE XX  
FIXED-DROPOUT SWEEP AT $H = 9 6$ FOR ETTH1 AND EXCHANGE RATE AND H = 48 FOR ILI. CELLS REPORT TEST MSE/MAE; $p ^ { \star }$ IS SELECTED SOLELY BY VALIDATION MSE AND MARKED IN BOLD. UNDERLINED CELLS MARK THE SECOND-LOWEST TEST MSE IN EACH ROW.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Backbone</td><td rowspan="2"> $p ^ { \star }$ </td><td colspan="10">Fixed dropout rate p (test MSE/MAE)</td></tr><tr><td>0.00</td><td>0.05</td><td>0.10</td><td>0.15</td><td>0.20</td><td>0.25</td><td>0.30</td><td>0.35</td><td>0.40</td><td>0.45</td><td>0.50</td></tr><tr><td>ETTh1</td><td>Informer</td><td>0.50</td><td>1.478/0.872</td><td>1.508/0.879</td><td>1.367/0.835</td><td>1.425/0.860</td><td>1.360/0.836</td><td>1.432/0.858</td><td>1.362/0.844</td><td>1.506/0.884</td><td>1.314/0.824</td><td>1.284/0.815</td><td>0.975/0.739</td></tr><tr><td>Exchange Rate</td><td> Informer</td><td>0.05</td><td>4.024/1.589</td><td>2.754/1.169</td><td>4.107/1.601</td><td>4.227/1.626</td><td>4.261/1.625</td><td>4.192/1.617</td><td>4.214/1.619</td><td>4.277/1.625</td><td>4.213/1.612</td><td>4.170/1.601</td><td>4.274/1.621</td></tr><tr><td>ILI</td><td>Informer</td><td>0.45</td><td>7.662/2.060</td><td>9.094/2.213</td><td>9.310/2.254</td><td>8.394/2.102</td><td>8.853/2.181</td><td>7.877/2.023</td><td>8.567/2.122</td><td>8.519/2.106</td><td>8.573/2.119</td><td>6.569/1.807</td><td>8.792/2.146</td></tr><tr><td>ETTh1</td><td>PatchTST</td><td>0.05</td><td>0.392/0.392</td><td>0.386/0.390</td><td>0.383/0.390</td><td>0.391/0.390</td><td>0.390/0.390</td><td>0.388/0.390</td><td>0.387/0.390</td><td>0.387/0.391</td><td>0.388/0.391</td><td>0.391/0.393</td><td>0.399/0.396</td></tr><tr><td>Exchange Rate</td><td>PatchTST</td><td></td><td>0.100.107/0.230</td><td>0.102/0.225</td><td>0.108/0.231</td><td>0.101/0.223</td><td>0.101/0.225</td><td>0.103/0.226</td><td>0.103/0.226</td><td>0.103/0.227</td><td>0.104/0.228</td><td>0.105/0.229</td><td>0.106/0.230</td></tr><tr><td>ILI</td><td>PatchTST</td><td></td><td>0.00 2.956/1.111</td><td>2.919/1.096</td><td>2.992/1.110</td><td>2.962/1.110</td><td>3.130/1.171</td><td>3.066/1.158</td><td>3.172/1.197</td><td>3.286/1.234</td><td>3.454/1.273</td><td>3.867/1.366</td><td>3.928/1.380</td></tr><tr><td>ETTh1</td><td>TimeFilter</td><td>0.00</td><td>0.391/0.391</td><td>0.391/0.391</td><td>0.391/0.391</td><td>0.391/0.391</td><td>0.391/0.391</td><td>0.391/0.391</td><td>0.391/0.391</td><td>0.390/0.391</td><td>0.391/0.391</td><td>0.391/0.391</td><td>0.390/0.391</td></tr><tr><td>Exchange Rate</td><td>TimeFilter</td><td>0.00</td><td>0.108/0.232</td><td>0.106/0.229</td><td>0.106/0.229</td><td>0.106/0.229</td><td>0.106/0.229</td><td>0.106/0.229</td><td>0.102/0.225</td><td>0.106/0.229</td><td>0.103/0.226</td><td>0.103/0.226</td><td>0.106/0.229</td></tr><tr><td>ILI</td><td>TimeFilter</td><td>0.10</td><td>2.619/1.012</td><td>2.560/1.019</td><td>2.619/1.012</td><td>2.574/1.020</td><td>2.514/1.010</td><td>2.532/1.011</td><td>2.545/1.014</td><td>2.551/1.015</td><td>2.562/1.018</td><td>2.590/1.022</td><td>2.623/1.028</td></tr><tr><td>ETTh1</td><td>TimeMixer</td><td>0.00</td><td>0.390/0.392</td><td>0.391/0.390</td><td>0.394/0.390</td><td>0.391/0.388</td><td>0.391/0.388</td><td>0.391/0.388</td><td>0.390/0.388</td><td>0.390/0.388</td><td>0.390/0.388</td><td>0.390/0.388</td><td>0.389/0.388</td></tr><tr><td>Exchange Rate</td><td>TimeMixer</td><td>0.00</td><td>0.102/0.224</td><td>0.101/0.223</td><td>0.101/0.223</td><td>0.101/0.223</td><td>0.101/0.223</td><td>0.101/0.223</td><td>0.101/0.223</td><td>0.102/0.224</td><td>0.102/0.224</td><td>0.102/0.224</td><td>0.102/0.224</td></tr><tr><td>ILI</td><td>TimeMixer 0.00</td><td></td><td>3.227/1.207</td><td>3.467/1.268</td><td>3.201/1.200</td><td>3.196/1.201</td><td>3.201/1.206</td><td>3.187/1.205</td><td>3.203/1.210</td><td>3.211/1.214</td><td>3.230/1.220</td><td>3.256/1.225</td><td>3.280/1.233</td></tr></table>

## E. Detailed Ablation Results

TABLE XXI  
COMPLETE COMPONENT ABLATION ON SYNTH-12 ACROSS FIVE CORRUPTION SCALES. ROWS REPORT MSE/MAE AT EACH σ AND ON AVERAGE; ∆ IS RELATIVE TO FIXED DROPOUT. OLS DENOTES GLOBAL ORDINARY LEAST-SQUARES DETRENDING. THE FULL SACM CONFIGURATION IS LABELED “OURS,” AND BOLD MARKS THE LOWEST ERROR IN EACH RESULT COLUMN.
<table><tr><td rowspan="2">Method</td><td colspan="3">Configuration</td><td colspan="2"> $\sigma = 0 . 1$ </td><td colspan="2"> $\sigma = 0 . 3$ </td><td> $\sigma = 0 . 5$ </td><td> $\sigma = 0 . 7$ </td><td colspan="2"> $\sigma = 0 . 9$ </td><td colspan="2">Average  $\Delta \%$  (vs. Base)</td></tr><tr><td>Detrend LogNorm log-SFM</td><td></td><td></td><td>MSE MAE</td><td>MSE</td><td>MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE MAE</td><td>MSE</td><td>MAE</td><td>MSE</td><td>MAE</td></tr><tr><td>Baseline</td><td></td><td>=</td><td>=</td><td>1.228 0.862</td><td>1.189</td><td>0.846</td><td>1.180 0.843</td><td>1.140 0.831</td><td>1.060</td><td>0.798 1.159</td><td>0.836</td><td></td><td></td></tr><tr><td>Minimal Model</td><td>None</td><td>x</td><td>X</td><td>1.698 1.034</td><td>1.907</td><td>1.092</td><td>1.668 1.010</td><td>0.633 0.650</td><td>1.665 1.015</td><td>1.514</td><td>0.960</td><td>30%↓</td><td>14%↓</td></tr><tr><td>w/o Detrend+Norm</td><td>None</td><td>x</td><td>√</td><td>2.041 1.124</td><td>1.159</td><td>0.843</td><td>1.114 0.827</td><td>1.954 1.111</td><td>1.072 0.806</td><td>1.468</td><td>0.942</td><td>26%↓</td><td>12%↓</td></tr><tr><td>w/o Detrend</td><td>None</td><td>√</td><td>√</td><td>1.672 1.017</td><td>1.863</td><td>1.093</td><td>1.721 1.051</td><td>1.442 0.950</td><td>1.513 0.960</td><td>1.642</td><td>1.014</td><td>41%↓</td><td>21%↓</td></tr><tr><td>Simple Detrend</td><td>Simple</td><td>√</td><td>√</td><td>1.416 0.943</td><td>1.757</td><td>1.042</td><td>1.674 1.018</td><td>1.486 0.958</td><td>1.535 0.974</td><td>1.574</td><td>0.987</td><td>35%↓</td><td>18%↓</td></tr><tr><td>w/o Spectral Norm</td><td>OLS</td><td>x</td><td>√</td><td>1.336 0.911</td><td>1.251</td><td>0.879</td><td>1.278 0.889</td><td>1.588 0.988</td><td>1.090 0.811</td><td>1.308</td><td>0.896</td><td>12%↓</td><td>7%↓</td></tr><tr><td>w/o log-SFM Anchor</td><td>OLS</td><td>√</td><td>x</td><td>1.386 0.933</td><td>1.532</td><td>0.968</td><td>1.540 0.977</td><td>1.693 1.029</td><td>1.448 0.944</td><td>1.520</td><td>0.970</td><td>31%↓</td><td>16%↓</td></tr><tr><td>SACM (Ours)</td><td>OLS</td><td>√</td><td> $\checkmark$ </td><td>1.135 0.841</td><td>1.155</td><td>0.746</td><td>1.050 0.808</td><td>1.049 0.803</td><td>0.990 0.785</td><td>1.076</td><td>0.797</td><td>7.2%↑</td><td>4.7%↑</td></tr></table>

Table XXI isolates the three preprocessing choices that turn a residual score into a reliable sample-wise capacity modulation signal: global OLS detrending, log-amplitude normalization, and log-SFM anchoring. The Baseline row is fixed-dropout training without adaptive scoring and is the reference for the final $\Delta \%$ columns. Reading the ablations as a pipeline rather than as independent switches clarifies which stages matter most.

Detrending is the dominant factor. Removing it entirely raises average MSE from 1.076 to 1.642 (52.6% relative degradation vs. the full method) and leaves every σ well above both Baseline and Ours. Endpoint-style Simple Detrend is only a partia remedy: average MSE remains 1.574, and at $\sigma = 0 . 9$ it reaches 1.535 (worse than no detrending at 1.513), while the full OLS configuration obtains 0.990. A slope estimated from two endpoints can be sensitive to corruption at those endpoints, whereas OLS uses the full window. The observed error difference is consistent with this motivation for global detrending.

The other two stages refine an already detrended residual. Dropping the log-SFM anchor raises average MSE from 1.076 to 1.520 (41.3% relative vs. the full method), nearly as severe as removing detrending, consistent with the spectral reference helping calibrate the mask. Removing log-amplitude normalization is milder but still material (1.308 average MSE, 21.6% relative vs. the full method), supporting normalization before computing the residual score. Configurations that strip multiple stages at once (Minimal Model; w/o Detrend+Norm) stay in the 1.47–1.51 average-MSE band and underperform fixed dropou on ∆%, showing that a poorly calibrated adaptive score can hurt more than a constant p.

The full configuration has the lowest average MSE and the lowest MSE at four of the five values of σ, and it is the only row with positive ∆% against Baseline (7.2% MSE / 4.7% MAE). At $\sigma = 0 . 7$ , Minimal Model records 0.633/0.650, but its average MSE of 1.514 remains above both the full method (1.076) and fixed dropout (1.159). Overall, the complete matrix supports a joint design. Removing any one of the three components increases average error, while the isolated advantage of the Minimal Model at $\sigma = 0 . 7$ does not persist across scales. Together, the components provide the strongest average performance among the tested configurations on Synth-12.

## F. Batch-Size Sensitivity

The rate mapper normalizes each residual score against the current mini-batch before mapping it to a dropout probability, so the batch size B determines how the relative unreliability signal is estimated. Table XXII sweeps $B \in \{ 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ for PatchTST on ETTh1 and WPMixer on Synth-12 at $\sigma = 0 . 5$ , both at $H = 9 6$ . Both settings show a smooth interior optimum at conventional batch sizes: MSE is lowest at $B = 3 2$ for PatchTST (0.3833) and at $B = 6 4$ for WPMixer (0.2781), and changes little across the remaining values of B. MAE follows the same pattern, with its minimum at $B = 1 2 8$ for PatchTST (0.3889) and at $B = 6 4$ for WPMixer (0.3880). Over the five batch sizes, the relative MSE range is 2.42% and 3.11%, with a coefficient of variation of 0.97% and 1.24%, respectively. The batch-relative mapping is therefore stable across conventional batch sizes and does not require per-dataset tuning of B. The min–max guard also makes the mapper well defined at the boundary. When the batch is too small to define a range, the dispersion condition is not met and the normalized score defaults to the neutral prior $\hat { s } _ { i } = 0 . 5$ , returning a moderate default dropout probability instead of an ill-defined normalized score. This behavior keeps gradients well behaved for very small B, and it is consistent with the interior optima above: moderate batches provide enough samples for a meaningful min–max range while retaining the benefit of adaptive rates.

TABLE XXII  
BATCH-SIZE SENSITIVITY AT H = 96 FOR TWO REPRESENTATIVE BACKBONES. BECAUSE THE RATE MAPPER NORMALIZES RESIDUAL SCORES WITHIN EACH MINI-BATCH, WE VARY $B \in \{ 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ AND REPORT TEST MSE/MAE; BOLD MARKS THE LOWEST MSE IN EACH SETTING. ALL RUNS WITHIN A SETTING SHARE A FIXED CONTROL CONFIGURATION, SO THE COMPARISON ISOLATES THE EFFECT OF B. THE LAST TWO COLUMNS SUMMARIZE THE MSE SWEEP: RELATIVE RANGE IS (max MSE − $\operatorname* { m i n } _ { B } \mathrm { M S E } ) /$ min MSE, AND CV IS THE POPULATION STANDARD DEVIATION ACROSS THE FIVE BATCH SIZES DIVIDED BY THEIR MEAN.
<table><tr><td rowspan="2">Backbone / dataset</td><td rowspan="2">Metric</td><td colspan="4">Batch size B</td><td colspan="3">MSE variation</td></tr><tr><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td><td>Range (%)</td><td>CV (%)</td></tr><tr><td>PatchTST ETTh1</td><td>MSE</td><td>0.390715</td><td>0.391424</td><td>0.383258</td><td>0.392519</td><td>0.384708</td><td>2.42</td><td>0.97</td></tr><tr><td></td><td>MAE</td><td>0.392903</td><td>0.392274</td><td>0.390247</td><td>0.391016</td><td>0.388892</td><td></td><td></td></tr><tr><td>WPMixer</td><td>MSE</td><td>0.280752</td><td>0.285856</td><td>0.286771</td><td>0.278124</td><td>0.279373</td><td>3.11</td><td>1.24</td></tr><tr><td>Synth-12 (σ=0.5)</td><td>MAE</td><td>0.390588</td><td>0.397123</td><td>0.394220</td><td>0.387984</td><td>0.388969</td><td></td><td></td></tr></table>

## APPENDIX C

## COMPONENT-WISE SPECTRAL DIAGNOSTICS

This section isolates canonical signal components and corruption types to probe the spectral scorer qualitatively. These diagnostic panels are separate from the layered-corruption Synth-12 benchmark and show what the scorer treats as concentrated structure versus residual under different non-stationarities.

Scope of the diagnostic reconstructions. The visualizations use hard coefficient retention to expose where spectral energy is located. The diagnostic threshold is fixed for visualization, whereas SACM learns channel-adaptive thresholds and uses the resulting residual to dynamically modulate active capacity via sample-wise retention probabilities. The model retains the original input in data space X, and the panels stress-test spectral concentration qualitatively.

## A. Stationary Periodic Signals

Figure 9 shows stationary periodic signals with time-invariant frequency and amplitude. In a finite sampled window, their spectra contain narrow peaks near the fundamental frequencies (e.g., 0.5 and 2.0 Hz). Under the tested corruptions, these peaks remain distinguishable from much of the broadband residual. The hard-threshold reconstructions therefore recover the dominant cycles with limited phase distortion, while the residual absorbs most of the additive high-frequency energy. This setting is the easiest for spectral scoring: energy is already sparse, so residual magnitude provides an effective proxy for sample unreliability.

Clean/reference Noisy input/spectrum

Noise threshold

(a) Gaussian Noise

Filtered reconstruction Signal threshold

![](images/2fefbb202f81a573d77514e04c31d79504f4343f4ff8c863b9794d023ccabf71.jpg)  
(d) Heavy-tail Noise

(b) Spectrum  
![](images/e7490c57e2d752ecf805740813b14997a708e3152cd5d6c49e5afb3aece1814f.jpg)  
(e) Spectrum

(c) Reconstruction  
![](images/95be13cc5b0e914fc61a28ac644b0c8761f13b025db151fd2ad5a1ea99418b97.jpg)  
(f) Reconstruction

![](images/3ae8762c6e80f3cc31f0684c8ca83ae1c2f3a2828d515970a1dbbca92e164907.jpg)

![](images/079b8f207649826e4b81e35affbf55b5890107667f6ad31e6b15812cd9e7ac13.jpg)  
(g) Missing Values

![](images/9c816a58ea84ee0998cd1380736a5915c25d0813cff36f6d5bf46d48ae6b1153.jpg)

(h) Spectrum  
![](images/2fbf425bb75fe755f9ab419098a95aa07a973134af66d483f8f433b23104eb63.jpg)  
Time (s)

(i) Reconstruction  
![](images/d315148b6b33fafed300f5f2f138c6869c0213cbd9b01962ecf890cd16d87617.jpg)  
Frequency (Hz)

![](images/d21c7d4e4a4c56d85f26c8a64af1f1a787e54db165e7ce370cfdd903e1fe390b.jpg)  
Time (s)  
Fig. 9. Stationary periodic signals under three isolated corruptions (Gaussian, heavy-tail, point-wise missing values). Columns: corrupted input, amplitude spectrum, and hard-threshold reconstruction. Narrow peaks remain separable from broadband residual energy, illustrating spectral separation in these examples.

## B. Non-stationary Mean (Trend & Seasonality)

Figure 10 combines linear drift with seasonal oscillations. Before detrending, the drift concentrates energy near zero frequency and the seasonal component appears at harmonic peaks. Without this step, a hard retention threshold can either over-preserve the DC ramp or leak trend energy into neighboring bins after windowing. The scorer removes a global linear fit before spectral masking and restores it after reconstruction, reducing boundary leakage while retaining the main seasonal structure. The residual then better reflects corruption and residual non-trend fluctuation rather than the deterministic drift itself, which matches the role of OLS preprocessing in the main method.

## C. Non-stationary Frequency (Chirp)

Figure 11 presents a linear chirp sweeping from 0.5 to 3.0 Hz. Its energy forms a structured band rather than isolated peaks, so a peak-only view of spectral sparsity is incomplete. In these examples, the applied threshold retains enough of the band to

![](images/0c424939caf1a90798102d888d3d89c2a50fbdff3388290d9df0cd43eec1643f.jpg)  
(d) Heavy-tail Noise

![](images/f9809e8d9538b54babb01c2418fb135dcebb12da290e0632a482b31d34eab5a5.jpg)

Noise threshold

(c) Reconstruction  
![](images/65dd671fe4053ca1ef5c2458924edaa8f1f197794d96ea294b5545320aaa4b35.jpg)

(e) Spectrum  
![](images/39e2dcf0de70bfa113a01120039e2532cb289698219c434ad6de78650aba9631.jpg)

(f) Reconstruction  
![](images/8ea24abb7ff0d5b473aa3bc19c839a07194e908476a981c1da2ce5f34609d5bf.jpg)

![](images/11db2ae590adf97c94120f7474c2bddad1bd0e5ae44eedae4bb87848a9bd1cd2.jpg)

(g) Missing Values  
![](images/39c224c2ebccf918620ead0fc0a56a3bb8337fe6ea90d05574dbe169f050bdfe.jpg)  
Time (s)

(i) Reconstruction  
(h) Spectrum  
![](images/9ba984f4c8023111788ab7e9e295f771ccb21b9907c77afc86ff047cdb9374ac.jpg)

![](images/faa449bbe540dc6cc54427d29772b067fcc45acf579f78fe42c82002f95997e5.jpg)  
Time (s)  
Fig. 10. Trend-seasonal signals under three isolated corruptions. Columns: corrupted input, spectrum after global OLS detrending, and reconstruction with trend restoration. Detrending reduces DC leakage so the residual better reflects corruption rather than deterministic drift.

reconstruct the evolving oscillation and the instantaneous frequency trajectory remains visible after inversion. Corruptions raise the broadband floor around the band, which increases residual energy without fully erasing the chirp structure. This supports treating residual magnitude as a soft difficulty cue even when the latent signal is time-varying rather than purely tonal.

## D. Non-stationary Variance (Amplitude Modulation)

Figure 12 depicts amplitude-modulated signals, where a low-frequency envelope changes the carrier amplitude. The spectrum contains a carrier and modulation sidebands; the reconstructions preserve the principal amplitude variation in the tested cases. Because the envelope redistributes energy into nearby bins rather than destroying the carrier, hard retention can still recover the overarching modulation pattern while leaving fine-scale envelope noise in the residual. Residual sideband energy can include both corruption and low-amplitude modulation structure. This motivates retaining the original input for prediction while using the residual to guide regularization.

## E. Spectral Sparsity Analysis on Real-world Benchmarks

Figure 13 retains the top 10% of Fourier coefficients by amplitude on nine forecasting datasets and sets the rest to zero before inversion. Seven displayed correlations exceed 0.95, while Electricity and ECG have correlations of 0.868 and 0.769, respectively. Spectral concentration therefore varies across the examples: a small coefficient subset closely reconstructs some series but loses more structure on Electricity and ECG. The discarded coefficients can contain both nuisance fluctuations and useful signal, motivating the use of spectral residuals as a regularization cue while preserving the original input.

![](images/f5c69b48ee73fe702c67b6fcaf9483c687626a4fb66c30811232eed1586e9b2f.jpg)  
(d) Heavy-tail Noise

![](images/f8df65a8e2544330fb8e9d99f2200b784a4c6b9976e7bb2d3755429c4df00c57.jpg)  
(e) Spectrum

(c) Reconstruction  
![](images/600a99cb0c1abda4c5e3d9a06c443c1ceb208e3dd1cedf7ef7a8c2c8779b7922.jpg)  
(f) Reconstruction

![](images/bc457cabb47ddf9166367e1e28d9575541f111a02cd3df91fc07b996097ca821.jpg)  
(g) Missing Values

![](images/633a8b029ec592fb4297b680db466eed3b4a21b1aa37010cd22bdc155fff4914.jpg)

![](images/b1586c59f608a8986e28fca91614295c574927aee5eb96b2473768f428952cb6.jpg)

(h) Spectrum  
![](images/38f93dcdf247062862c5ea8fd3ddc249c62bccb6b0cad9f799e1ef30403e9a01.jpg)  
Time (s)

(i) Reconstruction  
![](images/80396bb20a3c9da418609865f5ce6b77f5ec066615b27d8d1985f793d0cd13ae.jpg)

![](images/ef085e5167cff5aa9d22f6310e78545cf5e816d75542daa2b2797fa0dcfdd03b.jpg)  
Time (s)  
Fig. 11. Linear chirps sweeping from 0.5 to 3.0 Hz under three isolated corruptions. Columns: corrupted input, amplitude spectrum, and hard-threshold reconstruction. Energy forms a structured band rather than isolated peaks; the retained band preserves the sweep while corruptions raise residual energy.

## APPENDIX D

## IMPACT OF SPECTRAL LEAKAGE AND DETRENDING

The discrete Fourier transform treats a finite observation window as one period of a periodically extended sequence. When a time series contains a trend, the join between the last time step $x _ { L }$ and the first $x _ { 1 }$ can therefore be discontinuous. This is a property of the finite-window DFT, not of the forecasting backbone itself, but it matters because SACM scores residual energy in the spectral domain before modulating sample-wise active capacity.

This discontinuity can produce spectral leakage, spreading boundary-induced energy across the spectrum and introducing ringing artifacts in truncated spectral reconstruction. If left unaddressed, the spectral mask M may attenuate these leakage components, causing the reconstructed signal xˆ to exhibit large boundary errors. Edge-MAE can then reflect periodic-extension artifacts rather than stochastic corruption, which would inflate residual scores for well-structured samples and misallocate capacity. Global linear detrending reduces this confounder in the residual score, while residuals can still contain structure from non-stationary components beyond a linear trend. We compare three preprocessing strategies: (1) no detrending, (2) endpoint-based detrending, and (3) global linear OLS detrending across four stress tests (Figure 14). The panels isolate failure modes that appear when spectral scoring is applied to short windows with trends, outliers, or incomplete seasonal cycles.

## A. Theoretical Comparison

End-to-End Detrending is a local approach that estimates the trend solely from the boundary values: $\hat { \mathbf { x } } _ { t r e n d } ( t ) ~ =$ $\begin{array} { r } { \mathbf { x } _ { 1 } + \frac { t - 1 } { L - 1 } ( \mathbf { x } _ { L } - \mathbf { x } _ { 1 } ) } \end{array}$ for $t \in \{ 1 , \ldots , L \}$ . While computationally trivial, it is highly sensitive to sensor noise at the endpoints (t = 1 or $t = L )$ . Any spike, drop-out, or quantization error at either boundary fully determines the slope, so a single corrupted endpoint can dominate the residual everywhere.

![](images/8e8e4dc00a69edae02e2b5108077a27e62ff1b7657e69815e33f2bb235126561.jpg)  
(d) Heavy-tail Noise

(b) Spectrum  
![](images/8b78ccc7169f4553d6109fabf33bb27f14f824369f1b5dd51c67f25bb7c66344.jpg)

(c) Reconstruction  
![](images/bf3c2ec02b13f58effe0a5a07507465171c3a7ad22e9367608ce9ad662c46b11.jpg)  
(f) Reconstruction

(e) Spectrum  
![](images/0644929bc7d940dda0c822a77f6d68ea2b4d1a2c809561d5abb19a8f3bb5d2f0.jpg)

![](images/bfc6133f5555d45df10e067281f00bb3a8f61eba43804172db431737253f10c8.jpg)

![](images/733d30e9b9ace8225abdaf745c7f97b96b47bc38aec50ebed4001d0d67884ad2.jpg)

(g) Missing Values  
![](images/422ae5a6a85c8554800f44fd1f073044eafda61707a26f27846867e0e2731eab.jpg)  
Time (s)

(h) Spectrum  
(i) Reconstruction  
![](images/1fe146060a2e60744ad5a3f566229e52e29f3189fbbca41b0ba64133abd6ce08.jpg)

![](images/2fe12a6cc1b08e254f73209340f78ecdad810774f9403a570d872b3c81cd651f.jpg)  
Time (s)  
Fig. 12. Amplitude-modulated signals (2.0 Hz carrier, 0.2 Hz envelope) under three corruptions. Columns: corrupted input, spectrum with carrier and sidebands and hard-threshold reconstruction. The reconstruction retains the amplitude variation, while the residual can contain both corruption and modulation detail.

Global Linear Detrending (Ours) estimates $\begin{array} { r } { \mathbf { w } ^ { * } , \mathbf { b } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { w } , \mathbf { b } } \sum _ { t = 1 } ^ { L } \| \mathbf { x } _ { t } - ( \mathbf { w } t + \mathbf { b } ) \| _ { 2 } ^ { 2 } } \end{array}$ from all time points. Compared with endpoint interpolation, a single endpoint has less leverage, while ordinary least squares remains sensitive to outliers. In exchange, the fit is more stable under isolated boundary failures and better matches the role of detrending as a lightweight preprocessing step before soft spectral masking. The residual serves as a training-time difficulty cue while the backbone continues to use the original input.

## B. Analysis of Failure Modes

We evaluate reconstruction fidelity near the sequence boundaries using Edge-MAE. Let $\mathcal { E } \subseteq \{ 1 , \dots , L \}$ denote the boundary indices highlighted in Figure 14. For C channels,

$$
\mathrm { E d g e - M A E } = \frac { 1 } { | \mathcal { E } | C } \sum _ { t \in \mathcal { E } } \sum _ { c = 1 } ^ { C } \left| x _ { t , c } - \hat { x } _ { t , c } \right| .\tag{30}
$$

Lower values indicate better boundary reconstruction. This diagnostic differs from the full-window residual score of SACM: Edge-MAE stresses leakage at t=1 and $t { = } L ,$ whereas the training residual averages over the window after soft masking.

Sensitivity to Outliers (Row 1). In the “Linear + Start Outlier” scenario, a spike at t = 1 directly determines the endpoint-interpolation slope. Global OLS distributes the fit over all points, mitigating outlier leverage without ignoring extreme observations. The practical consequence for capacity modulation is that endpoint detrending can produce an artificially inflated unreliability score from a single boundary outlier, whereas global OLS reduces the influence of that endpoint on the fitted trend.

![](images/31c1dd2ee1bfd5bfac1c627f193900dd2f0d33e91317f176bbeddd5eafa60d47.jpg)  
Fig. 13. Top-10% Fourier reconstruction on nine forecasting datasets. Each pair shows the original and reconstructed series with Pearson correlation, followed by the amplitude spectrum and retention threshold. Seven correlations exceed 0.95; Electricity and ECG yield 0.868 and 0.769. “Illness” denotes ILI. The hard mask is used for this diagnostic; SACM uses a learned soft mask to compute a training-time regularization cue.

Phase Mismatch in Seasonality (Row 2). If the window does not contain an integer number of cycles, $x _ { 1 } \neq x _ { L }$ and the repeated-window interpretation introduces a boundary discontinuity. Both detrending variants reduce the resulting leakage in this example by removing the linear component of the join discontinuity before spectral truncation. Without detrending, residual energy near the edges can be dominated by periodic-extension artifacts even when the interior seasonal pattern is clean.

Instability under Non-linearity (Row 3). For a quadratic trend, endpoint noise changes the interpolation slope. Global OLS gives the least-squares linear approximation, but no detrending has the lowest Edge-MAE in this row, showing that a linear trend model is not uniformly preferable under model mismatch. We retain global OLS as a default because it wins on the other three stress tests and is cheap to apply. Strongly curved trends may leave systematic residual structure that soft masking must absorb.

Robustness to Sensor Failure (Row 4). A step-up regime shift is followed by a failed final observation. Endpoint interpolation connects the initial and failed final values and infers a nearly flat trend, erasing most of the high-state evidence. Global OLS uses the intervening high-state observations and produces a more suitable linear fit in this constructed case. The residual after global OLS therefore better reflects the failed endpoint rather than a systematically wrong slope across the whole window.

Taken together, the four rows favor global OLS as a default preprocessor for spectral residual scoring: it avoids endpoint leverage under outliers and failures (rows 1 and 4), reduces seasonal join discontinuities (row 2), and shows limitations under strong nonlinear trends where no linear model is optimal (row 3). Edge-MAE stresses boundary reconstruction rather than full-window forecasting error, so the figure diagnoses leakage-induced residual bias. In SACM, the residual serves as a label-free unreliability proxy for capacity modulation; global OLS keeps this signal less confounded by boundary extension artifacts

![](images/858fec6fbe5f90c673ebd7c3000ba373d1bfc6d10eb4858eeb9837695cf7c988.jpg)  
Fig. 14. Comparison of no detrending, endpoint interpolation, and global OLS under four boundary stress tests, scored by reconstruction Edge-MAE near the window edges. Each row compares the three detrending strategies; panels show the observed input, clean reference, estimated trend, reconstruction, and Edge-MAE. Row 1 (start-point outlier): endpoint interpolation inherits the spike as the full slope, while global OLS dilutes its leverage. Row 2 (seasonal phase mismatch): $x _ { 1 } \neq x _ { L }$ induces a join discontinuity; both detrending variants reduce leakage relative to no detrending. Row 3 (quadratic trend with endpoint noise): no detrending achieves the lowest Edge-MAE, showing that a linear trend model is not uniformly preferable under model mismatch. Row 4 (regime shift then final-step sensor failure): endpoint interpolation flattens the high-state evidence, whereas global OLS recovers a more suitable linear fit. Global OLS wins rows 1, 2, and 4. The plot label “Robust OLS refers to ordinary global OLS, not a robust-regression estimator.”

## APPENDIX E

## THEORY AND PROOFS

This appendix derives the local dropout approximation, characterizes the oracle allocation, and proves the excess-risk theorem for the stylized bias–variance surrogate introduced in the theory section.

## A. Local dropout approximation

Condition on the input, mini-batch, and backbone parameters. Consider one hidden activation vector h and a mask B with independent Bernoulli entries of retention probability $q = 1 - p .$ . Set all other activation multipliers to one. In inverted dropout, $\widetilde { \mathbf { h } } = \mathbf { B } \odot \mathbf { h } / q$ . Writing $\pmb { \delta } = \widetilde { \mathbf { h } } - \mathbf { h }$ , independence gives

$$
\mathbb { E } [ \delta ] = \mathbf { 0 } , \qquad \mathrm { C o v } ( \delta ) = \frac { p } { q } \mathrm { d i a g } ( h _ { 1 } ^ { 2 } , \dots , h _ { d } ^ { 2 } ) .
$$

Let $\ell ( \mathbf { h } ) = \mathcal { L } ( g _ { \theta } ( \mathbf { h } ) , \mathbf { y } )$ , where $g _ { \theta }$ is the downstream part of the backbone. Assume that ℓ is twice continuously differentiable on the line segments from h to every possible masked activation. A second-order Taylor expansion yields

$$
\mathbb { E } [ \ell ( { \mathbf { h } } + \pmb { \delta } ) ] = \ell ( { \mathbf { h } } ) + \frac { 1 } { 2 } \operatorname { t r } \big ( \nabla _ { { \mathbf { h } } } ^ { 2 } \ell ( { \mathbf { h } } ) \operatorname { C o v } ( { \pmb { \delta } } ) \big ) + R ,
$$

where the first-order term vanishes and the exact remainder is

$$
R = \mathbb { E } \bigg [ \int _ { 0 } ^ { 1 } ( 1 - t ) \pmb { \delta } ^ { \top } \big ( \nabla ^ { 2 } \ell ( \mathbf { h } + t \pmb { \delta } ) - \nabla ^ { 2 } \ell ( \mathbf { h } ) \big ) \pmb { \delta } \mathrm { d } t \bigg ] .
$$

If the Hessian is Lipschitz with constant $K _ { \ell }$ along these segments, then $| R | \leq K \ell \mathbb { E } [ \| \delta \| ^ { 3 } ] / 6$ . Thus a quadratic activation loss has $R = 0$ , while the approximation for a nonlinear loss depends on the size of this remainder. Defining

$$
\Omega ( \mathbf { x } ; \theta ) = \sum _ { j = 1 } ^ { d } \frac { \partial ^ { 2 } \ell } { \partial h _ { j } ^ { 2 } } ( \mathbf { h } ) h _ { j } ^ { 2 }
$$

and using $p / q = \lambda ( p )$ gives

$$
\mathbb { E } _ { \mathbf { B } } [ \ell ( \widetilde { \mathbf { h } } ) ] = \ell ( \mathbf { h } ) + \frac { 1 } { 2 } \lambda ( p ) \Omega ( \mathbf { x } ; \theta ) + R ,
$$

which is the dropout quadratic approximation stated in the main paper after collecting the higher-order terms in $R .$ For a nonlinear loss, neglecting R requires sufficiently small curvature variation over the mask perturbations. The quadratic term acts as a non-negative sensitivity penalty in regions where the relevant curvature term $\Omega ( \mathbf { x } ; \theta )$ is non-negative.

B. Regularization mapping and oracle characterization

For $p \in [ 0 , 1 ) , q = 1 - p$ and $\lambda = p / q ,$ so $q = ( 1 + \lambda ) ^ { - 1 }$ and

$$
\frac { \mathrm { d } \lambda } { \mathrm { d } p } = \frac { 1 } { ( 1 - p ) ^ { 2 } } > 0 .
$$

For the mapper in the adaptive-capacity subsection, letting $a = \mathrm { s o f t p l u s } ( \gamma ) > 0$ gives

$$
\frac { \mathrm { d } p } { \mathrm { d } \hat { s } } = ( p _ { \mathrm { m a x } } - p _ { \mathrm { m i n } } ) a \mathrm { s e c h } ^ { 2 } ( a \hat { s } ) > 0 .
$$

For fixed mapper parameters, $p$ and $\lambda ( p )$ are increasing in the normalized unreliability score within a mini-batch. The corresponding quadratic correction is a non-negative regularization term when $\Omega \geq 0$ . The bounds $[ p _ { \mathrm { m i n } } , p _ { \mathrm { m a x } } ]$ define an allocation envelope; a mapper with finite a attains a maximum rate $p _ { \operatorname* { m i n } } + ( p _ { \operatorname* { m a x } } - p _ { \operatorname* { m i n } } )$ tanh(a) within that envelope. The expected number of retained activation coordinates is $\begin{array} { r } { \mathbb { E } [ \sum _ { j = 1 } ^ { d } B _ { j } ] = ( 1 - p ) d } \end{array}$ for a layer of width d. This quantity does not change the number of trainable parameters.

## C. Assumptions and oracle allocation

For $C _ { 1 } , C _ { 2 } > 0$ and unreliability level $\sigma \geq 0 .$ , define

$$
\mathcal { E } ( \lambda , \sigma ) = C _ { 1 } \lambda ^ { 2 } + C _ { 2 } \sigma ^ { 2 } ( 1 + \lambda ) ^ { - 2 } , \qquad \lambda \in \Lambda = [ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ] ,
$$

where $0 \leq \lambda _ { \operatorname* { m i n } } < \lambda _ { \operatorname* { m a x } } < \infty$ . The interval is induced by the dropout bounds,

$$
\lambda _ { \operatorname* { m i n } } = { \frac { p _ { \operatorname* { m i n } } } { 1 - p _ { \operatorname* { m i n } } } } , \qquad \lambda _ { \operatorname* { m a x } } = { \frac { p _ { \operatorname* { m a x } } } { 1 - p _ { \operatorname* { m a x } } } } .
$$

Assume $\mathbb { E } [ \sigma ( \mathbf { X } ) ^ { 2 } ] < \infty$ and set $g ( \sigma ) = \Pi _ { \Lambda } ( \lambda ^ { \star } ( \sigma ) )$ , where $\lambda ^ { \star }$ minimizes the surrogate over $[ 0 , \infty )$ . The surrogate coefficient $C _ { 1 }$ and $C _ { 2 }$ are fixed positive constants over the population; all sample heterogeneity enters through $\sigma ( { \mathbf { X } } )$ . More explicitly, assume a scalar prediction error $e _ { \lambda } = b \lambda + \sqrt { C _ { 2 } } \sigma \varepsilon / ( 1 + \lambda )$ with $b ^ { 2 } = C _ { 1 } , \mathbb { E } [ \varepsilon \mid \sigma ] = 0$ , and $\mathbb { E } [ \varepsilon ^ { 2 } \mid \sigma ] = 1$ . The conditional mean is bλ, and the conditional variance is $C _ { 2 } \sigma ^ { 2 } / ( 1 + \lambda ) ^ { 2 }$ . Their squared-mean-plus-variance decomposition gives the stated surrogate. The bias term assumes a linear response with zero baseline bias. For a smooth bias function B, the expansion $B ( \lambda ) = B ( 0 ) + B ^ { \prime } ( 0 ) \lambda + o ( \lambda )$ motivates this choice when $B ( 0 ) = 0$ and $b = B ^ { \prime } ( 0 ) \neq 0$ . The surrogate uses this first-order response throughout the comparison interval. The residual factor is obtained from minimizing $\begin{array} { r } { \frac { 1 } { 2 } ( \bar { v } - \sigma \varepsilon ) ^ { 2 } + \frac { \lambda } { 2 } v ^ { 2 } } \end{array}$ , whose minimizer is $\sigma \varepsilon / ( 1 + \lambda )$ . This scalar penalty is a response assumption for the fitted predictor, rather than a consequence of the dropout Taylor expansion. The inverted-dropout multiplier itself has variance λ and unit mean. The allocation theorem is conditional on these response assumptions and the common positive coefficients $C _ { 1 } , C _ { 2 }$

## D. Proof of the main excess-risk theorem

Theorem E.1 (Excess surrogate risk under reliability heterogeneity). Let $\sigma ( { \bf X } ) \geq 0$ satisfy $\mathbb { E } [ \sigma ( \mathbf { X } ) ^ { 2 } ] < \infty$ , let $\Lambda = [ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ]$ be the feasible interval induced by $p \in [ p _ { \operatorname* { m i n } } , p _ { \operatorname* { m a x } } ] ,$ , and let $g ( \sigma ) = \Pi _ { \Lambda } ( \lambda ^ { \star } ( \sigma ) )$ ) be the projected minimizer of $\mathcal { E } ( \lambda , \sigma )$ . If $\mathrm { V a r } [ g ( \sigma ( \mathbf { X } ) ) ] > 0$ , then for every fixed $\lambda _ { \mathrm { f i x } } \in \Lambda$

$$
\begin{array} { r } { \mathbb { E } \big [ \mathcal { E } \big ( \lambda _ { \mathrm { f i x } } , \sigma ( \mathbf { X } ) \big ) - \mathcal { E } \big ( g ( \sigma ( \mathbf { X } ) ) , \sigma ( \mathbf { X } ) \big ) \big ] \geq C _ { 1 } \operatorname { V a r } [ g ( \sigma ( \mathbf { X } ) ) ] > 0 . } \end{array}
$$

Proof. For fixed σ, differentiation gives

$$
\partial _ { \lambda } \mathcal { E } = 2 C _ { 1 } \lambda - 2 C _ { 2 } \sigma ^ { 2 } ( 1 + \lambda ) ^ { - 3 } , \qquad \partial _ { \lambda } ^ { 2 } \mathcal { E } = 2 C _ { 1 } + 6 C _ { 2 } \sigma ^ { 2 } ( 1 + \lambda ) ^ { - 4 } > 0 .
$$

Thus $\mathcal { E } ( \cdot , \sigma )$ is strictly convex and has a unique minimizer on Λ, namely $g ( \sigma )$ . The stationarity equation for the minimizer over $[ 0 , \infty )$ is $C _ { 1 } \lambda ^ { \star } ( 1 + \lambda ^ { \star } ) ^ { 3 } = C _ { 2 } \sigma ^ { 2 } ;$ ; its left-hand side is strictly increasing for $\lambda \geq 0 .$ , so $\lambda ^ { \star } ( \sigma )$ and hence $g ( \sigma )$ are nondecreasing in $\sigma ,$ with strict increase before projection clips the value at a boundary.

Let $m = \mathbb { E } [ g ( \sigma ( \mathbf { X } ) ) ^ { * }$ ]. Boundedness of g and $\mathbb { E } [ \sigma ( \mathbf { X } ) ^ { 2 } ] < \infty$ ensure that the risks and squared allocation differences below have finite expectations. The first-order optimality condition on Λ gives $\partial _ { \lambda } \mathcal { E } ( g ( \sigma ) , \sigma ) ( \lambda - g ( \sigma ) ) \geq 0$ for every $\lambda \in \Lambda$ . Together with the uniform lower bound $\partial _ { \lambda } ^ { 2 } \mathcal { E } \geq 2 C _ { 1 }$ , this implies, for any fixed $\lambda _ { \mathrm { f i x } } \in \Lambda$

$$
\mathcal { E } ( \lambda _ { \mathrm { f i x } } , \sigma ) - \mathcal { E } ( g ( \sigma ) , \sigma ) \geq C _ { 1 } \big ( \lambda _ { \mathrm { f i x } } - g ( \sigma ) \big ) ^ { 2 } .
$$

Taking expectations and using $\mathbb { E } [ ( \lambda _ { \mathrm { f i x } } - g ) ^ { 2 } ] = \mathrm { V a r } ( g ) + ( \lambda _ { \mathrm { f i x } } - m ) ^ { 2 } \ge \mathrm { V a r } ( g ) > 0$ proves the theorem inequality. □

The condition $\mathrm { V a r } [ g ( \sigma ( \mathbf { X } ) ) ] > 0$ captures genuine allocation heterogeneity; when the projected oracle is constant, a uniform configuration coincides with the oracle. The theorem permits saturation at either endpoint of Λ.

## REFERENCES

[1] A. L. Goldberger, L. A. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, “PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals,” Circulation, vol. 101, no. 23, pp. e215–e220, 2000.

[2] Y. Liang, Z. Shao, F. Wang, Z. Zhang, T. Sun, and Y. Xu, “BasicTS: An open source fair multivariate time series prediction benchmark,” in Benchmarking, Measuring, and Optimizing, ser. Lecture Notes in Computer Science, vol. 13852. Springer, 2023, pp. 87–101.

[3] A. Bagnall, H. A. Dau, J. Lines, M. Flynn, J. Large, A. Bostrom, P. Southam, and E. Keogh, “The UEA multivariate time series classification archive, 2018,” arXiv:1811.00075, 2018.

[4] D. Anguita, A. Ghio, L. Oneto, X. Parra, and J. L. Reyes-Ortiz, “A public domain dataset for human activity recognition using smartphones,” in Proc. Eur. Symp. Artif. Neural Netw. (ESANN), 2013, pp. 437–442.

[5] B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen, and J. J. L. Oberye, “Analysis of a sleep-dependent neuronal feedback loop: The slow-wave microcontinuity of the EEG,” IEEE Trans. Biomed. Eng., vol. 47, no. 9, pp. 1185–1194, 2000.

[6] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, “Robust anomaly detection for multivariate time series through stochastic recurrent neural network,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2019.

[7] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, “Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding,” in Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min., 2018.

[8] J. Xu, H. Wu, J. Wang, and M. Long, “Anomaly transformer: Time series anomaly detection with association discrepancy,” in Int. Conf. Learn. Represent., 2022.

[9] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Proc. AAAI Conf. Artif. Intell., vol. 35, no. 12, 2021.

[10] Y. Zhang and J. Yan, “Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting,” in 11th Int. Conf. Learn. Represent., 2023.

[11] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in Int. Conf. Learn. Represent., 2023.

[12] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “TimesNet: Temporal 2d-variation modeling for general time series analysis,” in Int. Conf. Learn. Represent., 2023.

[13] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “iTransformer: Inverted Transformers are effective for time series forecasting,” in Int. Conf. Learn. Represent., 2024.

[14] S. Wang, H. Wu, X. Shi, T. Hu, H. Luo, L. Ma, J. Y. Zhang, and J. Zhou, “TimeMixer: Decomposable multiscale mixing for time series forecasting,” in Int. Conf. Learn. Represent., 2024.

[15] M. M. N. Murad, M. Aktukmak, and Y. Yilmaz, “WPMixer: Efficient multi-resolution mixing for long-term time series forecasting,” in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 18, 2025, pp. 19 581–19 588.

[16] Y. Hu, G. Zhang, P. Liu, D. Lan, N. Li, D. Cheng, T. Dai, S.-T. Xia, and S. Pan, “TimeFilter: Patch-specific spatial-temporal graph filtration for time series forecasting,” in 42nd Int. Conf. Mach. Learn., ser. Proc. Mach. Learn. Res., vol. 267, 2025, pp. 24 893–24 911. [Online]. Available: https://proceedings.mlr.press/v267/hu25ac.html

[17] V. Naghashi, M. Boukadoum, and A. B. Diallo, “A multiscale model for multivariate time series forecasting,” Sci. Rep., vol. 15, no. 1, p. 1565, 2025.