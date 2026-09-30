# ZeroDiff: Zero-Shot Time Series Reconstruction via Informed-Prior Diffusion

Yingda Fan <sup>1</sup> Dan Lu <sup>2</sup> Xiaowei Jia <sup>1</sup>

## Abstract

Time series modeling increasingly demands highquality supervision, yet target observations remain scarce—exogenous inputs are broadly available, but target measurements are often unavailable due to cost, infrastructure, or accessibility constraints. Can models trained on observed locations reconstruct target time series where measurements have never been collected? We term this zero-shot time series reconstruction. A naive approach—directly mapping exogenous inputs to targets—can yield predictions at unobserved locations, but without target signals, such models fail to capture the intrinsic dynamics of the target variable, producing overly smooth outputs that underestimate extremes. This reveals systematic errors that call for explicit modeling and calibration. We propose ZeroDiff, which constructs an informed prior from exogenous variables alone, then learns to calibrate reconstruction errors through diffusion—training on observed locations and generalizing to unobserved ones. Experiments across diverse real-world datasets demonstrate significant improvements over existing approaches. Our code is available at https: //github.com/YingdaFan/ZeroDiff-ICML2026.

## 1. Introduction

As time series modeling scales from specialized architectures (Nie et al., 2023; Liu et al., 2024) to foundation models (Ansari et al., 2024; Das et al., 2024), the need for high-quality supervision has grown—yet in many scenarios, target observations remain scarce or entirely missing. This challenge arises across domains—from flood forecasting (Kratzert et al., 2018), where meteorological data are abundant but streamflow is sparsely observed across space, to solar energy prediction (Paletta et al., 2023), where satellite imagery and reanalysis data are globally available but irradiance observations are measured at few stations. In each case, exogenous inputs are broadly available, but target measurements are observed only at a subset of locations. In such settings, practitioners face a fundamental question: can models learn from locations where targets are observed to reconstruct target time series at locations where they have never been measured? We term this problem zero-shot cross-domain time series reconstruction. Unlike forecasting (Zhou et al., 2021; Salinas et al., 2020), which predicts future values from historical observations at the same location, or imputation (Cao et al., 2018; Du et al., 2023), which fills in missing values given partial observations, zero-shot reconstruction requires the model to infer an entire target time series at locations with no historical target data whatsoever. This setting poses a distinct challenge: the model must generalize across locations, reconstructing temporal dynamics it has never directly observed.

A central difficulty lies in the heterogeneity of target distributions across locations. While the temporal dynamics—how targets respond to exogenous inputs—are often governed by shared physical or statistical principles, the marginal distribution of the target variable can vary dramatically in mean and variance. As a result, models trained on observed locations struggle to transfer to unobserved ones.

In such settings, probabilistic reconstruction is essential. When ground truth targets are unavailable, uncertainty quantification becomes important for downstream decisionmaking (Gneiting & Katzfuss, 2014). Diffusion models (Ho et al., 2020; Song et al., 2021) offer a principled framework for probabilistic modeling (Rasul et al., 2021; Tashiro et al., 2021), but standard diffusion requires access to ${ \bf Y } _ { 0 }$ for both forward corruption and loss computation—impossible when targets are absent. Recent work shows diffusion benefits from informative initialization: CARD (Han et al., 2022) replaces N (0, I) with a regression-based prior, SDEdit (Meng et al., 2022) preserves structure via partial noising. These advances suggest that if we can construct an informed prior Y<sup>ˆ</sup> capturing the expected target, diffusion can shift from generation to calibration—refining a structured estimate rather than generating from noise.

We present ZeroDiff, a diffusion-based framework for zeroshot time series reconstruction. It decomposes the crosslocation generalization problem into two stages connected by a shared informed prior $\hat { \mathbf { Y } } .$ The first stage generalizes from exogenous inputs to target structure: target distributions vary drastically across locations in mean and variance, but their input–output dynamics are shared once expressed in a standardized space. The location-specific statistics needed for that standardization can themselves be inferred from exogenous variables alone. The informed prior Y<sup>ˆ</sup> at unobserved locations is then constructed directly from X, without any target signal.

![](images/4e7582b73749739fd9720a8025703438ba492a0c3898aa9a424c2985a9b8437a.jpg)

(b) ZeroDiff calibration  
![](images/a34d2dc22c8d5aea55d9f6ce75c614c5a12eced2e9cd969dc81091e6c5fb1fd4.jpg)  
Figure 1. In the exogenous-only reconstruction the peaks are oversmoothed and no longer stand out (a); ZeroDiff calibrates it onto the ground truth, restoring their prominence (b).

The second stage generalizes calibration across locations. Because Y<sup>ˆ</sup> is constructed identically at observed and unobserved locations, its systematic errors share the same structure everywhere. A diffusion model trained to calibrate the prior on observed locations therefore transfers to unobserved ones. This shifts the role of diffusion from generation to calibration: the reverse process refines $\hat { \mathbf Y }$ rather than denoising from $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$

In summary, our contributions are:

• We formalize the problem of zero-shot time series reconstruction and identify a structural assumption that makes it tractable: target magnitudes vary across locations, but input–target dynamics are shared.

• We propose informed-prior diffusion: a single prior inferred from exogenous variables both carries crosslocation dynamics and anchors a calibration that transfers from observed to unobserved locations.

• We design a bidirectional denoiser that conditions on both past and future timesteps, improving reconstruction at peaks and sharp transitions.

Conflict of Interest Disclosure. The authors declare no financial conflicts of interest related to this work.

## 2. Related Work

Diffusion Models for Time Series. Diffusion models have recently emerged as a powerful framework for probabilistic time series modeling, achieving strong performance across forecasting, imputation, and generation tasks. TimeGrad (Rasul et al., 2021) first introduces autoregressive denoising diffusion for probabilistic forecasting, generating future values step-by-step conditioned on historical observations. CSDI (Tashiro et al., 2021) extends score-based diffusion to imputation via conditional generation. SSSD (Alcaraz & Strodthoff, 2022) integrates structured state-space models to capture long-range dependencies. TimeDiff (Shen & Kwok, 2023) proposes non-autoregressive diffusion for parallel generation. Diffusion-TS (Yuan & Qiao, 2024) employs encoder-decoder transformers with disentangled temporal representations. mr-Diff (Shen et al., 2024b) leverages multi-resolution structure through seasonal-trend decomposition. Most recently, CNDiff (Rishi et al., 2025) introduces nonlinear data transformations for conditional forecasting. Despite these advances, all existing diffusion-based time series methods share a fundamental assumption: target observations must be available at locations during training. This assumption is embedded at multiple levels—normalization, forward corruption, and loss computation—all of which require access to ground-truth targets. Consequently, when targets are entirely absent at a subset of locations, the diffusion framework cannot be applied. Our work addresses this previously unsupported regime.

Diffusion with Informative Priors. Standard diffusion models assume the forward process terminates at an uninformative prior $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ (Ho et al., 2020; Song et al., 2021), requiring the reverse process to generate from scratch. Recent work in computer vision has challenged this assumption by incorporating data-dependent priors. PriorGrad (Lee et al., 2022) demonstrates that adaptive priors derived from conditional information improve both efficiency and sample quality. CARD (Han et al., 2022) replaces the standard endpoint with a regression-based prior $\mathcal { N } ( f ( \mathbf { x } ) , \sigma )$ , where $f ( \mathbf { x } )$ is a pre-trained predictor; the diffusion process then refines predictions rather than generates from noise. SDEdit (Meng et al., 2022) starts the reverse process from partially noised inputs, preserving structural information from a guide image. Cold Diffusion (Bansal et al., 2023) demonstrates that meaningful degradations—such as blur or downsampling—can replace Gaussian noise entirely, revealing that diffusion’s core mechanism is learning to invert transformations. These approaches share a common insight: diffusion is most effective as a refinement mechanism when initialized from a structured prior, a design principle that has also influenced other prior-guided reconstruction pipelines (Yuan et al., 2025b;a). However, existing methods assume that the prior itself is obtained from supervised learning using target observations at the same domain. In contrast, our setting requires constructing informative priors without any target data at the reconstruction location. ZeroDiff extends the idea of informative priors to a zero-shot regime, where priors are inferred solely from exogenous variables and calibrated via diffusion using supervision from other locations.

## Zero-shot Learning and Cross-location Generalization.

Zero-shot learning enables recognition of classes unseen during training by leveraging auxiliary information (Lampert et al., 2009). Early approaches learn mappings between visual features and semantic attributes (Farhadi et al., 2009) or word embeddings (Frome et al., 2013). Subsequent work explores compatibility functions (Akata et al., 2016), latent embeddings (Xian et al., 2016), and generative approaches (Xian et al., 2018). Xian et al. (Xian et al., 2017) provide a comprehensive benchmark establishing evaluation protocols for the field. The core principle—transferring knowledge from seen to unseen classes via shared semantic representations—motivates our approach of transferring dynamics from observed to unobserved locations.

In spatio-temporal modeling, cross-location generalization has been explored through graph neural networks. STGCN (Yu et al., 2018) and DCRNN (Li et al., 2018) capture spatial dependencies via graph convolutions. Recent work addresses cross-city transfer: Domain Adversarial Spatial-Temporal Networks (Tang et al., 2022) learn cityinvariant representations, while Graph Neural Processes (Hu et al., 2023) enable extrapolation to unobserved locations. However, these methods assume that at least some target observations are available at all locations, focusing on forecasting rather than reconstruction. They do not address the setting where targets are unobserved at test locations.

Time series foundation models such as Chronos (Ansari et al., 2024), TimesFM (Das et al., 2024), and Time-MoE (Shi et al., 2025) demonstrate impressive zero-shot capabilities across datasets. However, their “zero-shot” refers to generalization across datasets—not across locations within a dataset where some locations lack target observations entirely.

In hydrology, the problem of Prediction in Ungauged Basins (PUB) (Kratzert et al., 2019) is conceptually related, aiming to predict streamflow at locations without gauges. While recent machine learning approaches improve point estimates in this setting, they typically lack principled uncertainty quantification.

## 3. Preliminary

## 3.1. Problem Formulation

Zero-shot time series reconstruction. Given exogenous variables X at a new location, the goal is to reconstruct the target time series Y without any historical target observations at that location.

Formally, consider K locations, each with exogenous variables $\bar { \mathbf { X } ^ { ( k ) } } \in \mathbb { R } ^ { L \times D }$ and target time series $\breve { \mathbf { Y } } ^ { ( k ) } \in \mathbb { R } ^ { L }$ where X and Y are temporally aligned with sequence length L (Tashiro et al., 2021; Cao et al., 2018). Locations are partitioned into:

• Observed O: both $\mathbf { X } ^ { ( k ) }$ and $\mathbf { Y } ^ { ( k ) }$ are available for training.

• Unobserved U: only $\mathbf { X } ^ { ( j ) }$ is accessible; the goal is to reconstruct $\mathbf { Y } ^ { ( j ) }$

Both training and inference operate on the same temporal span. This differs from forecasting (Zhou et al., 2021; Wu et al., 2021; Nie et al., 2023; Liu et al., 2024), which predicts future from past at the same location, and from imputation (Tashiro et al., 2021; Cao et al., 2018; Du et al., 2023), which fills gaps given partial observations. Here, the model must generalize across locations to reconstruct targets it has never seen.

The central challenge is that while the input-target dynamics may be shared across locations once properly standardized, the marginal distribution of Y differs substantially in mean and variance across locations. Standard normalization techniques (Kim et al., 2022; Liu et al., 2022) require target observations to compute location-specific statistics $( \mu , \sigma ) -$ precisely what is unavailable for $j \in \mathcal { U }$

## 3.2. Diffusion Models for Time Series

Denoising diffusion probabilistic models (DDPMs) (Ho et al., 2020; Song et al., 2021; Luo, 2022) learn data distributions through a forward process that progressively adds noise and a reverse process that learns to denoise. Given target ${ \bf Y } _ { 0 }$ , the forward process is defined as:

$$
q ( \mathbf { Y } _ { t } | \mathbf { Y } _ { 0 } ) = \mathcal { N } \big ( \sqrt { \bar { \alpha } _ { t } } \mathbf { Y } _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathbf { I } \big )\tag{1}
$$

The reverse process learns to denoise by optimizing $\mathcal { L } _ { \mathrm { { D D P M } } } = \mathbb { E } _ { t , \mathbf { Y } _ { 0 } , \epsilon } \big [ \| \epsilon - \epsilon _ { \theta } ( \mathbf { Y } _ { t } , t ) \| ^ { 2 } \big ]$ . For time series applications, two formulations are common depending on whether temporal windows are employed:

Sliding window formulation. In forecasting tasks (Rasul et al., 2021; Li et al., 2024; Shen et al., 2024a; Yuan $\&$ Qiao, 2024; Gao et al., 2025), the model conditions on historical observations $\mathbf { X } \in \mathbb { R } ^ { N \times D }$ to predict future targets $\mathbf { Y } \in \mathbb { R } ^ { M \times D }$ , where N and M denote the lookback and prediction horizon lengths respectively.

Aligned formulation. In reconstruction or imputation tasks (Tashiro et al., 2021; Alcaraz & Strodthoff, 2023; Kollovieh et al., 2023), $\mathbf { X } \in \mathbb { R } ^ { L \times D }$ and $\mathbf { Y } \in \mathbb { R } ^ { L }$ share the same temporal span L. The model learns to recover Y from temporally aligned exogenous inputs.

A key advancement is the use of informed priors (Han et al., 2022; Meng et al., 2022; Li et al., 2024). Rather than diffusing toward an uninformative endpoint $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , these methods use a regression-based prior $\mathcal { N } ( \hat { \mathbf { Y } } , \sigma ^ { 2 } \mathbf { I } )$ , where ${ \hat { \mathbf { Y } } } = f ( \mathbf { X } )$ . The forward process becomes:

$$
q ( \mathbf { Y } _ { t } | \mathbf { Y } _ { 0 } , \mathbf { X } ) = \mathcal { N } \big ( \sqrt { \bar { \alpha } _ { t } } \mathbf { Y } _ { 0 } + ( 1 - \sqrt { \bar { \alpha } _ { t } } ) \hat { \mathbf { Y } } , \bar { \sigma } _ { t } \mathbf { I } \big )\tag{2}
$$

where $\bar { \sigma } _ { t } = 1 - \bar { \alpha } _ { t }$

The missing-target problem. Crucially, both formulations require access to ground-truth targets ${ \bf Y } _ { 0 }$ during training. The forward process in Eq. (2) constructs $\mathbf { Y } _ { t }$ from $\mathbf { Y } _ { 0 } .$ , and the denoising process computes the loss using $\mathbf { Y } _ { t } .$ . For unobserved locations, neither the forward process nor the training objective can be defined—the diffusion framework fundamentally breaks down.

## 4. Methodology

We propose a framework for zero-shot time series reconstruction, illustrated in Fig. 2, built on three insights: (i) while target marginals $p ( \mathbf { Y } )$ vary drastically across locations in mean and standard deviation, the input-target dynamics are transferable once standardized; (ii) locationspecific statistics $( \mu , \sigma )$ can be inferred from exogenous variables X alone via cross-modal learning, without requiring target observations; (iii) by constructing informed priors for O where ground-truth is available, the diffusion model learns to calibrate estimation errors on O, and this calibration generalizes to U. Motivated by these observations, we first construct an informed prior through moment estimation and dynamics learning, and then refine it through diffusion-based calibration.

## 4.1. Transferable Prior Construction

Recent work on non-stationary time series (Kim et al., 2022; Liu et al., 2022) demonstrates that separating statistics (mean, variance) from temporal dynamics improves generalization. While the marginal distribution of Y varies across locations in magnitude, the temporal dynamics of how X drives Y, once normalized, are shared. This motivates learning in normalized space, but standard normalization requires target observations to compute location-specific $( \mu , \sigma )$ , which is precisely unavailable for $j \in \mathcal { U }$

We address this limitation through cross-modal moment estimation: inferring target statistics $( \mu _ { k } , \sigma _ { k } )$ from exogenous variables $\mathbf { X } ^ { ( k ) }$ alone. Although X and Y represent different physical quantities, the distributional characteristics of X are predictive of the magnitude of Y. In hydrological systems, for instance, a basin’s weather data and catchment properties can affect the magnitude of runoff.

Specifically, we model the marginal moments of $\mathbf { Y } ^ { ( k ) }$ as a learnable function of exogenous summary statistics:

$$
( \mu _ { k } , \sigma _ { k } ) = h ( \mathbf { Z } ^ { ( k ) } )\tag{3}
$$

where $\mathbf { Z } ^ { ( k ) } = [ \bar { \mathbf { X } } ^ { ( k ) } ; \sigma _ { \mathbf { X } } ^ { ( k ) } ] \in \mathbb { R } ^ { 2 D }$ concatenates the temporal mean and standard deviation of exogenous inputs. Since $\left( \mu _ { k } , \sigma _ { k } \right)$ are time-invariant statistics, we condition on summary statistics of X rather than the full sequence.

Cross-modal moment estimation. We estimate the target moments using a conditional VAE (Kingma & Welling, 2014; Sohn et al., 2015)—a form of exogenous-conditioned prior estimation that captures distributional structure across locations. The architecture consists of three components: a latent encoder $q _ { \psi } ( \mathbf { z } | \mu _ { k } , \sigma _ { k } )$ that maps target moments to a latent distribution, capturing location-specific deviations from the mean behavior; a feature encoder $\phi ( \mathbf { Z } ^ { ( k ) } )$ that transforms exogenous statistics into a representation conditioning the decoder; and a decoder $p _ { \psi } ( \mu _ { k } , \sigma _ { k } | \mathbf { z } , \phi ( \mathbf { Z } ^ { ( k ) } ) )$ that reconstructs target moments from both the latent code and the encoded exogenous features.

During training on ${ \mathcal { O } } ,$ we optimize the evidence lower bound:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { V A E } } = \mathbb { E } _ { q _ { \psi } } \left[ \| ( \mu _ { k } , \sigma _ { k } ) - \mathrm { D e c } _ { \psi } ( \mathbf { z } , \phi ( \mathbf { Z } ^ { ( k ) } ) ) \| ^ { 2 } \right] } \\ & { \quad \quad \quad + \lambda \cdot D _ { \mathrm { K L } } \big ( q _ { \psi } ( \mathbf { z } | \mu _ { k } , \sigma _ { k } ) \| \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) \big ) } \end{array}\tag{4}
$$

The KL regularization encourages the latent space to capture shared distributional structure: locations with similar exogenous characteristics are mapped to nearby points, enabling moment estimation to generalize from $\mathcal { O }$ to $\mathcal { U } .$

For $j \in \mathcal { U } ,$ although $\mathbf { Y } ^ { ( j ) }$ is unavailable, we can compute $\mathbf { Z } ^ { ( j ) } = [ \bar { \mathbf { X } } ^ { ( j ) } ; \mathrm { s t d } ( \mathbf { \bar { X } } ^ { ( j ) } ) ]$ directly from $\mathbf { X } ^ { ( j ) }$ . We then decode using the prior mean:

$$
( \hat { \mu } _ { j } , \hat { \sigma } _ { j } ) = \mathrm { D e c } _ { \psi } ( \mathbf { 0 } , \phi ( \mathbf { Z } ^ { ( j ) } ) )\tag{5}
$$

This enables zero-shot moment estimation: using only $\mathbf { X } ^ { ( j ) }$ and the distributional structure learned from ${ \mathcal { O } } _ { : }$ , we infer $( \hat { \mu } _ { j } , \hat { \sigma } _ { j } )$ without any target observations.

Dynamics learning in standardized space We learn a single dynamics model $f _ { \omega } : \mathbb { R } ^ { L \times D }  \mathbb { R } ^ { L }$ shared across all observed locations, performing reconstruction over the same temporal span rather than forecasting. The dynamics model $f _ { \omega }$ can adopt architectures such as LSTM (Kratzert et al., 2018), DLinear (Zeng et al., 2023), Mamba (Gu & Dao, 2023), or TimesNet (Wu et al., 2023); we compare these choices in Table 2.

![](images/cf22f08d0e766697441f159b279eb87b87f3087930e40de0f4ffd7e4fb4015ff.jpg)  
Figure 2. Proposed framework. The informed prior $\hat { \mathbf Y } = \Phi _ { \psi , \omega } ( \mathbf X )$ combines moment estimation (VAE) with dynamics learning in standardized space, generalizing to unobserved locations. The diffusion process starts from $\mathcal { N } ( \hat { \mathbf { Y } } , \bar { \sigma } _ { T } \mathbf { I } )$ rather than pure noise, enabling calibration instead of generation. The estimated moments also guide optimization by weighting training locations based on their proximity to the target in moment space.

The training objective aggregates prediction error over O:

$$
\operatorname* { m i n } _ { \omega } \sum _ { k \in \mathcal { O } } \left\| f _ { \omega } ( \mathbf { X } ^ { ( k ) } ) - \tilde { \mathbf { Y } } ^ { ( k ) } \right\| ^ { 2 }\tag{6}
$$

where $\tilde { \mathbf { Y } } ^ { ( k ) } = ( \mathbf { Y } ^ { ( k ) } - \mu _ { k } ) / \sigma _ { k }$ is the standardized target. By sharing parameters and training in normalized space, $f _ { \omega }$ captures the shared temporal dynamics of how X drives fluctuations in Y, rather than fitting location-specific magnitudes.

Informed prior construction. We use VAE-estimated statistics $( \hat { \mu } , \hat { \sigma } )$ for both O and U—even for $k \in \mathcal { O }$ where ground-truth statistics $( \mu _ { k } , \sigma _ { k } )$ are available. This ensures consistent estimation of error characteristics between training and inference, enabling the downstream diffusion model to learn a calibration that generalizes to U. The informed prior for $k \in \mathcal { O }$

$$
\hat { \mathbf { Y } } ^ { ( k ) } = \hat { \mu } _ { k } + \hat { \sigma } _ { k } \cdot f _ { \omega } ( \mathbf { X } ^ { ( k ) } )\tag{7}
$$

For $j \in \mathcal { U } \colon$

$$
\hat { \mathbf { Y } } ^ { ( j ) } = \hat { \mu } _ { j } + \hat { \sigma } _ { j } \cdot f _ { \omega } ( \mathbf { X } ^ { ( j ) } )\tag{8}
$$

where $( \hat { \mu } _ { j } , \hat { \sigma } _ { j } )$ is obtained via Eq. (5). For unobserved locations, the estimated moments play a critical role: $f _ { \omega }$ captures the normalized response pattern, while $( \hat { \mu } _ { j } , \hat { \sigma } _ { j } )$ rescales it to

the appropriate magnitude. For notational convenience, we denote the composite mapping as $\Phi _ { \psi , \omega } ( \mathbf { X } ) : = \boldsymbol { \hat { \mu } } \mathrm { + } \boldsymbol { \hat { \sigma } } \cdot \boldsymbol { f } _ { \omega } ( \mathbf { X } )$ where $( \hat { \mu } , \hat { \sigma } ) = \mathrm { D e c } _ { \psi } ( \mathbf { 0 } , \phi ( \mathbf { Z } ) )$ .

## 4.2. Generalizable Diffusion Calibration

The informed prior $\hat { \textbf { Y } } ^ { ( k ) }$ and $\hat { \mathbf { Y } } ^ { ( j ) }$ provide an initial reconstruction but inherit estimation error from both moment estimation and dynamics learning. We refine this prior through diffusion, learning the conditional distribution $\hat { \bf \Phi } _ { p } ( { \bf Y } ^ { ( k ) } | \hat { \bf Y } ^ { ( \bar { k } ) } , { \bf X } ^ { ( k ) } )$ on $\mathcal { O }$ . The denoiser $\epsilon _ { \theta }$ is shared across all $k \in \mathcal { O }$ , learning a calibration that generalizes to U. At inference, we apply this learned distribution to $j \in \mathcal { U }$ by conditioning on $\bar { ( } \hat { \mathbf { Y } } ^ { ( j ) } , \mathbf { X } ^ { ( j ) } )$ . Since the estimation procedure is shared, the error characteristics remain consistent across O and U, enabling the calibration to generalize.

Informed-prior forward process. We diffuse toward the informed prior rather than zero:

$$
q ( \mathbf { Y } _ { t } ^ { ( k ) } | \mathbf { Y } ^ { ( k ) } , \mathbf { X } ^ { ( k ) } ) = \mathcal { N } \big ( \sqrt { \bar { \alpha } _ { t } } \mathbf { Y } ^ { ( k ) } + ( 1 - \sqrt { \bar { \alpha } _ { t } } ) \hat { \mathbf { Y } } ^ { ( k ) } , \bar { \sigma } _ { t } \mathbf { I } \big )\tag{9}
$$

The noise schedule $\left( { { { \bar { \alpha } } _ { t } } , { { \bar { \sigma } } _ { t } } } \right)$ is shared across all locations, while each location k has its own informed prior $\hat { \mathbf { Y } } ^ { ( k ) } =$ $\hat { \mu } _ { k } + \hat { \sigma } _ { k } \cdot f _ { \omega } ( \mathbf { X } ^ { ( k ) } )$ as defined in Eq. (7), incorporating $\mathbf { X } ^ { ( k ) }$ through both the VAE-estimated moments and the dynamics model. ${ \mathrm { \bf A t } } \ t \ = \ T$ , this yields the endpoint distribution $\mathbf { Y } _ { T } ^ { ( k ) } \sim \mathcal { N } ( \hat { \mathbf { Y } } ^ { ( k ) } , \bar { \sigma } _ { T } \mathbf { I } )$ , in contrast to the standard $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ used in conventional diffusion models. This formulation follows the informed-prior framework of Han et al. (2022).

Prior-anchored reverse process. The informed-prior forward process (Eq. (9)) induces a modified posterior distribution. At each reverse step, we sample from $p _ { \theta } ( \mathbf { Y } _ { t - 1 } ^ { ( k ) } | \mathbf { Y } _ { t } ^ { ( k ) } , \mathbf { X } ^ { ( k ) } , \hat { \mathbf { Y } } ^ { ( k ) } )$ , whose posterior mean takes the form:

$$
\pmb { \mu } _ { t - 1 } ^ { ( k ) } = \gamma _ { 0 } \mathbf { Y } ^ { ( k ) } + \gamma _ { 1 } \mathbf { Y } _ { t } ^ { ( k ) } + \gamma _ { 2 } \hat { \mathbf { Y } } ^ { ( k ) }\tag{10}
$$

where $\gamma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 }$ are functions of $\alpha _ { t } , { \bar { \alpha } } _ { t - 1 } ,$ , and $\bar { \sigma } _ { t }$ (see $\mathsf { A p - }$ pendix B). Unlike standard DDPM, our posterior includes a third term—anchoring the trajectory to $\hat { \mathbf Y }$

During inference for $j \in \mathcal { U }$ , the clean target $\mathbf { Y } ^ { ( j ) }$ is unavailable. We estimate it via reparameterization:

$$
\hat { \mathbf { Y } } _ { 0 } ^ { ( j ) } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \big ( \mathbf { Y } _ { t } ^ { ( j ) } - \big ( 1 - \sqrt { \bar { \alpha } _ { t } } \big ) \hat { \mathbf { Y } } ^ { ( j ) } - \sqrt { \bar { \sigma } _ { t } } \epsilon _ { \theta } \big )\tag{11}
$$

Substituting $\hat { \mathbf { Y } } _ { 0 } ^ { ( j ) }$ for $\mathbf { Y } ^ { ( j ) }$ in Eq. (10) yields $\mu _ { t - 1 } ^ { ( j ) }$ <sub>1</sub>, ensuring the reverse trajectory remains guided by the informed prior throughout denoising.

Moment-guided optimization. In addition to denormalization in Eq. (8), the estimated moments $( \hat { \mu } _ { k } , \hat { \sigma } _ { k } )$ guide the optimization of the shared denoiser $\epsilon _ { \theta }$ . Since every $k \in \mathcal { O }$ contributes to updating the shared denoiser parameters, we reweight each location’s gradient contribution based on its proximity to the target in moment space. Let $( \hat { \mu } ^ { * } , \hat { \sigma } ^ { * } )$ denote the estimated moments of the target location; for simultaneous reconstruction of multiple targets, we use the centroid (Appendix E.1). We define weights using a Gaussian kernel over the moment space:

$$
w _ { k } = \exp { \left( - \frac { \| ( \hat { \mu } _ { k } , \hat { \sigma } _ { k } ) - ( \hat { \mu } ^ { * } , \hat { \sigma } ^ { * } ) \| ^ { 2 } } { 2 \tau ^ { 2 } } \right) }\tag{12}
$$

where τ controls the bandwidth. The training objective becomes:

$$
\mathcal { L } = \sum _ { k \in \mathcal { O } } w _ { k } \cdot \mathbb { E } _ { t , \epsilon } \left[ \| \epsilon - \epsilon _ { \theta } ( \mathbf { Y } _ { t } ^ { ( k ) } , \mathbf { X } ^ { ( k ) } , \hat { \mathbf { Y } } ^ { ( k ) } , t ) \| ^ { 2 } \right]\tag{13}
$$

Rather than aggregating gradients equally across all $k \in \mathcal { O }$ this formulation upweights locations whose moments are close to $( \hat { \mu } ^ { * } , \hat { \sigma } ^ { * } )$ , yielding a denoiser better suited for $j \in \mathcal { U }$

Bidirectional denoiser. Our calibration setting is fundamentally different from prior diffusion-based forecasting methods (Rasul et al., 2021), which adopts causal masking to prevent information leakage from future to past (Tashiro et al., 2021; Shen & Kwok, 2023). In contrast, during the calibration process $p ( \mathbf { Y } ^ { ( k ) } | \hat { \mathbf { Y } } ^ { ( k ) } , \mathbf { X } ^ { ( k ) } ,$ ), the conditioning signals $\mathbf { X } ^ { ( k ) }$ and $\hat { \mathbf { Y } } ^ { ( k ) }$ are fully observed across all timesteps. This enables bidirectional conditioning—realized through full self-attention, information flows forward from past and backward from future simultaneously:

$$
\epsilon _ { \theta } ( \mathbf { Y } _ { t , \ell } ^ { ( k ) } ) = f _ { \theta } \big ( \mathbf { Y } _ { t , 1 : L } ^ { ( k ) } , \mathbf { X } _ { 1 : L } ^ { ( k ) } , \hat { \mathbf { Y } } _ { 1 : L } ^ { ( k ) } , t \big ) , \quad \forall \ell \in \{ 1 , \dots , L \}\tag{14}
$$

This bidirectional flow facilitates the reconstruction of complex temporal dynamics that are difficult to infer from either direction alone. For example, a peak can be identified when the forward view captures an upward trend before a certain timestep and the backward view captures a downward trend immediately after it. The effect of the bidirectional flow is validated by the ablation results in Table 1.

## 5. Experiments

## 5.1. Experimental Setup

Table 3. Dataset statistics. Each experiment masks one location and trains on the rest.
<table><tr><td></td><td>STREAMFLOW</td><td>SOLAR</td><td>TEMP</td><td>METHANE</td></tr><tr><td>Domain</td><td>Hydrology</td><td>Energy</td><td>Hydrology</td><td>Carbon Emission</td></tr><tr><td>Exog. Dim</td><td>33</td><td>48</td><td>8</td><td>16</td></tr><tr><td>Length</td><td>4745</td><td>4015</td><td>1095</td><td>4015</td></tr><tr><td>Test Loc.</td><td>100</td><td>30</td><td>42</td><td>30</td></tr></table>

Datasets. We evaluate on four datasets spanning four scientific domains (Table 3). Unlike standard forecasting benchmarks that split data temporally, zero-shot reconstruction requires spatial partitioning: we evaluate whether models trained on observed locations can generalize to held-out ones. For each dataset, we adopt a leave-one-location-out protocol—iteratively masking all target observations at one location while retaining its exogenous inputs, training on the remaining locations, and evaluating reconstruction at the held-out location. TEST LOC. indicates how many independent trials are conducted per dataset; final metrics are averaged across all held-out locations.

CAMELS (Newman et al., 2015; Addor et al., 2017) is a widely-used hydrology benchmark containing daily streamflow from 531 gauges across the contiguous US, with catchment attributes and meteorology for large-sample studies. NHD contains daily stream water temperature derived from the National Hydrography Dataset (U.S. Geological Survey, 2019; 2024) at approximately 1-km resolution. Solar (Mc-Govern et al., 2015) is from the AMS 2013–2014 Solar Energy Prediction Contest, with daily solar energy from Oklahoma Mesonet stations and GEFS ensemble forecasts. Methane (Pastorello et al., 2020; Sun et al., 2025) provides daily methane $\mathrm { ( C H _ { 4 } ) }$ flux from FLUXNET.

Fair Experiment Most baselines are designed for time series forecasting with look-back window $L _ { b }$ and prediction horizon H. To fairly evaluate on reconstruction $( L = 3 6 5 )$ we adopt two protocols. First, sliding window reconstruction: We partition the sequence into overlapping segments. For each segment, baselines can use ground-truth Y from observed locations as look-back and predict the subsequent horizon. Specifically, we use $L _ { b } ~ \in ~ \{ 9 6 , 1 9 2 , 3 6 5 \}$ and $H \ \in \ \{ 9 6 , 1 9 2 , 3 6 5 \}$ , sliding with stride H to cover the full sequence. Final predictions are concatenated. Second, exogenous-only input assumption: For methods supporting exogenous variables, we provide $\mathbf { Y } _ { 1 : L }$ reconstructed from $\mathbf { X } _ { 1 : L }$ as input, matching our zero-shot setting. For each baseline, we report the better of the two protocols, selected on a held-out validation set. Further details on experimental fairness are provided in Appendix F.

Table 1. Performance comparison across different datasets. Best results are in bold. ↑ indicates higher is better, ↓ indicates lower is better.
<table><tr><td rowspan="2">Model</td><td colspan="3">Streamflow</td><td colspan="3">Solar</td><td colspan="3">Temp</td><td colspan="3">Methane</td></tr><tr><td>NSE↑</td><td>RMSE↓</td><td>MAE↓</td><td>NSE↑</td><td>RMSE↓</td><td>MAE↓</td><td>NSE↑</td><td>RMSE↓</td><td>MAE↓</td><td>NSE↑</td><td>RMSE↓</td><td>MAE↓</td></tr><tr><td colspan="10">Diffusion Baselines</td></tr><tr><td>CSDI (Tashiro et al., 2021)</td><td>†</td><td>2.814</td><td>1.267</td><td>0.019</td><td>7.896</td><td>6.035</td><td>†</td><td>6.883</td><td>4.525</td><td>†</td><td>1.910</td><td>1.610</td></tr><tr><td> $+ \ f _ { \omega }$ </td><td>0.060</td><td>2.367</td><td>1.214</td><td>0.307</td><td>7.271</td><td>5.926</td><td>0.722</td><td>2.881</td><td>2.286</td><td>†</td><td>1.229</td><td>1.083</td></tr><tr><td>SSSD (Alcaraz &amp; Strodthoff, 2022)</td><td>†</td><td>2.583</td><td>1.161</td><td>0.446</td><td>5.930</td><td>4.468</td><td>0.605</td><td>4.002</td><td>3.303</td><td>†</td><td>0.974</td><td>0.890</td></tr><tr><td> $+ \ f _ { \omega }$ </td><td>0.071</td><td>2.348</td><td>1.201</td><td>0.374</td><td>6.534</td><td>4.865</td><td>0.746</td><td>2.756</td><td>2.113</td><td>†</td><td>1.083</td><td>0.986</td></tr><tr><td>CSBI (Chen et al., 2023)</td><td>†</td><td>2.730</td><td>1.446</td><td>0.449</td><td>5.915</td><td>4.612</td><td>0.243</td><td>6.100</td><td>5.260</td><td>†</td><td>0.976</td><td>0.900</td></tr><tr><td> $+ \ f _ { \omega }$ </td><td>0.084</td><td>2.374</td><td>1.270</td><td>0.402</td><td>6.410</td><td>4.999</td><td>0.725</td><td>2.941</td><td>2.439</td><td>†</td><td>1.083</td><td>0.991</td></tr><tr><td>DiffusionTS (Yuan &amp; Qiao, 2024)</td><td>†</td><td>3.230</td><td>1.670</td><td>0.026</td><td>7.852</td><td>5.980</td><td>0.293</td><td>5.671</td><td>4.461</td><td>†</td><td>1.661</td><td>1.399</td></tr><tr><td> $+ \ f _ { \omega }$ </td><td>0.033</td><td>2.446</td><td>1.269</td><td>0.259</td><td>7.229</td><td>5.965</td><td>0.747</td><td>2.920</td><td>2.213</td><td>†</td><td>1.222</td><td>1.068</td></tr><tr><td>NsDiff (Ye et al., 2025)</td><td>†</td><td>3.907</td><td>2.686</td><td>0.515</td><td>5.546</td><td>4.556</td><td>0.700</td><td>2.902</td><td>2.371</td><td>†</td><td>1.597</td><td>1.372</td></tr><tr><td> $+ \ f _ { \omega }$ </td><td>0.105</td><td>2.488</td><td>1.325</td><td>0.473</td><td>6.131</td><td>4.987</td><td>0.763</td><td>2.562</td><td>2.063</td><td>†</td><td>1.262</td><td>1.085</td></tr><tr><td colspan="10">Ours</td><td colspan="3"></td></tr><tr><td> $f _ { \omega }$  (Kratzert et al., 2018)</td><td>0.131</td><td>2.139</td><td>1.099</td><td>0.345</td><td>6.407</td><td>5.280</td><td>0.813</td><td>2.306</td><td>1.848</td><td>†</td><td>1.074</td><td>0.940</td></tr><tr><td> $\Phi _ { \psi , \omega }$ </td><td>0.481</td><td>1.679</td><td>0.725</td><td>0.486</td><td>5.707</td><td>4.565</td><td>0.852</td><td>2.205</td><td>1.752</td><td>0.481</td><td>0.371</td><td>0.280</td></tr><tr><td>w/o prior</td><td>†</td><td>2.415</td><td>1.440</td><td>0.323</td><td>6.470</td><td>5.355</td><td>0.829</td><td>2.371</td><td>1.902</td><td>†</td><td>0.848</td><td>0.728</td></tr><tr><td>w/o bidir. &amp; wk</td><td>0.365</td><td>1.750</td><td>1.022</td><td>0.518</td><td>5.526</td><td>4.638</td><td>0.860</td><td>2.014</td><td>1.608</td><td>0.671</td><td>0.276</td><td>0.202</td></tr><tr><td>w/o bidir.</td><td>0.463</td><td>1.576</td><td>0.903</td><td>0.519</td><td>5.519</td><td>4.636</td><td>0.859</td><td>2.010</td><td>1.600</td><td>0.623</td><td>0.276</td><td>0.203</td></tr><tr><td>ZeroDiff</td><td>0.596</td><td>1.423</td><td>0.736</td><td>0.846</td><td>3.109</td><td>2.494</td><td>0.868</td><td>1.756</td><td>1.399</td><td>0.788</td><td>0.219</td><td>0.159</td></tr></table>

† NSE < 0 (cross-location transfer failed). $f _ { \omega } { : }$ : LSTM (Kratzert et al., 2018); + f<sub>ω</sub> = pre-trained on $f _ { \omega }$ predictions.  
bidir. = bidirectional denoiser (§ 4.2); w<sub>k</sub> = moment-guided weighting (Eq. 12).

![](images/f36bb56e66c3b35bea30beabb4cbb2fb3bbdb6b829aa3df5910e3dba61d3fc92.jpg)  
Figure 3. Qualitative comparison on Basin 02065500 (Year 1997, 365 days). Top row: reconstruction by each backbone dynamics model alone; bottom row: reconstruction enhanced by ZeroDiff. Blue: ground truth streamflow; red: model prediction. ZeroDiff consistently improves temporal fidelity, particularly at peak flow events.

## 5.2. Main Experiments

Baselines. Since no existing method directly addresses zero-shot time series reconstruction, we compare against representative diffusion-based approaches from two related tasks. From time series imputation: CSDI (Tashiro et al., 2021), which performs conditional score-based diffusion with self-supervised masking; SSSD (Alcaraz & Strodthoff, 2022), which combines structured state-space models with diffusion for long-range temporal modeling; and CSBI (Chen et al., 2023), a Schrodinger bridge-based¨ method for conditional generation. From time series forecasting: Diffusion-TS (Yuan & Qiao, 2024), which uses an encoder-decoder Transformer with disentangled seasonaltrend representations; and NsDiff (Ye et al., 2025), which introduces non-stationary diffusion to handle distribution shift. All baselines require target observations during training; we adapt them using the sliding-window and exogenousconditioned protocols described in Appendix F, selecting the best configuration per baseline via validation.

Table 2. Effect of dynamics model architecture on CAMELS. Upper: dynamics model $f _ { \omega }$ with moment estimation; lower: full ZeroDiff pipeline with each $f _ { \omega }$
<table><tr><td>Model</td><td>NSE↑ RMSE↓</td><td>MAE↓</td></tr><tr><td colspan="3">Dynamics model  $f _ { \omega }$  only</td></tr><tr><td>LSTM (Kratzert et al., 2018) DLinear (Zeng et al., 2023)</td><td>0.481 1.679 1.777</td><td>0.725 0.818</td></tr><tr><td>Mamba (Gu &amp; Dao, 2023)</td><td>0.428 0.557 1.869</td><td>0.892</td></tr><tr><td>TimesNet (Wu et al., 2023)</td><td>0.517 1.613</td><td>0.706</td></tr><tr><td>ZeroDiff with</td><td> $f _ { \omega }$ </td><td></td></tr><tr><td>ZeroDiff (LSTM)</td><td>0.596</td><td></td></tr><tr><td>ZeroDiff (DLinear)</td><td>1.423 1.353</td><td>0.736</td></tr><tr><td>ZeroDiff (Mamba)</td><td>0.612 0.667 1.618</td><td>0.722</td></tr><tr><td></td><td></td><td>0.828</td></tr><tr><td>ZeroDiff (TimesNet)</td><td>0.593 1.427</td><td>0.724</td></tr></table>

Results. Table 1 shows a clear pattern: standard diffusion models fail (†) on Streamflow and Methane—datasets exhibiting high variability both across locations and within each time series. Without moment estimation to capture location-specific statistics, models trained on observed locations cannot generalize to unseen ones.

Providing baselines with $f _ { \omega }$ predictions $\left( \operatorname { r o w s } + f _ { \omega } \right)$ reduces errors. However, these models can only incorporate the prior through pre-training—using $f _ { \omega } ( \mathbf { X } )$ as pseudo-labels. This treats the prior as training data rather than as an integral part of the diffusion process.

ZeroDiff instead integrates the informed prior directly into the forward process, reverse process, and optimization. The consistent gap between ZeroDiff and baselines $( + f _ { \omega } )$ across all datasets demonstrates that tightly coupling the prior with diffusion is essential.

Ablation Study. We ablate ZeroDiff in the lower block of Table 1; effects are most pronounced on Streamflow where magnitudes differ by orders of magnitude across basins. Replacing $\mathcal { N } ( \hat { \mathbf { Y } } , \bar { \sigma } ^ { 2 } \mathbf { I } )$ with $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ causes failure on Streamflow and Methane, while the deterministic prior $\Phi _ { \psi , \omega }$ alone yields positive NSE everywhere; comparing $\Phi _ { \psi , \omega }$ with ZeroDiff shows that diffusion calibration further improves performance by correcting systematic errors from dynamics learning. Removing the VAE collapses performance on high-variability datasets, as the shared dynamics model cannot rescale predictions without estimated $( \hat { \mu } , \hat { \sigma } )$ . Moment-guided weighting $w _ { k }$ provides gains only when cross-location heterogeneity is high—on Solar and Temp where locations cluster in moment space, weighting becomes redundant. The bidirectional denoiser yields consistent improvement across all datasets: unlike forecasting where causality prohibits looking ahead, reconstruction allows information to flow both directions, enabling correction at sharp transitions.

## 5.3. Choice of Dynamics Model

The dynamics model $f _ { \omega }$ is a plug-in component of ZeroDiff. In Table 2, we evaluate four architectures on CAMELS (Streamflow), the most challenging dataset: LSTM (Hochreiter & Schmidhuber, 1997), DLinear (Zeng et al., 2023), Mamba (Gu & Dao, 2023), and TimesNet (Wu et al., 2023).

As shown in Table 2, ZeroDiff consistently improves all four backbones, confirming that the diffusion refinement is architecture-agnostic. Notably, although Mamba achieves the highest standalone NSE (0.557), our default choice is the simplest LSTM—yet ZeroDiff (LSTM) already reaches 0.596, demonstrating that a correct formulation matters more than a powerful backbone. Even a naive dynamics model can capture sufficient temporal structure to provide a good prior for diffusion refinement. This also reveals the potential of ZeroDiff as a bridge between classical time series models and diffusion-based generation: practitioners can plug in any off-the-shelf backbone and obtain calibrated reconstructions. Figure 3 illustrates this qualitatively—across all four backbones, ZeroDiff tracks ground-truth peaks more faithfully than the backbone alone.

## 5.4. Robustness to Prior Estimation Error

The deterministic prior $\Phi _ { \psi , \omega }$ varies substantially in quality across datasets and locations (Table 1), and the reverse posterior mean (Eq. 10) anchors the diffusion trajectory to this prior through the $\gamma _ { 2 }$ term. We analyze how the final reconstruction depends on prior quality from three angles: the role of the ${ \mathrm { V A E } } ,$ the behavior of diffusion under prior error, and the mechanism that decouples the two stages.

Decoupling moment estimation. Zero-shot spatial extrapolation requires estimating both the marginal moments $( \mu , \sigma )$ and the temporal dynamics. The VAE decouples these two estimands, letting each component focus its generalization on one aspect. Table 4 shows that on low-heterogeneity datasets (Solar, Temp), removing the VAE causes negligible change, but on high-heterogeneity datasets it causes severe degradation, collapsing to −0.655 NSE on Methane. An imperfect prior is far better than no prior.

Table 4. Ablation of VAE moment estimation. NSE across four datasets.
<table><tr><td>Pipeline</td><td>Solar</td><td>Temp</td><td>Streamflow</td><td>Methane</td></tr><tr><td>VAE + LSTM + Diffusion</td><td>0.846</td><td>0.868</td><td>0.596</td><td>0.788</td></tr><tr><td> $\mathrm { L S T M } + \mathrm { D i f f u s i o n } \ ( \mathrm { n o } \ \mathrm { V A E } )$ </td><td>0.839</td><td>0.828</td><td>0.377</td><td>-0.655</td></tr></table>

Table 5. Prior estimation error and diffusion recovery. PRIOR NSE<0: locations where the deterministic prior fails. IMPROVED: locations where diffusion increases NSE.
<table><tr><td>Dataset</td><td>Max/Min ratio</td><td>VAE error</td><td>Prior NSE&lt;0</td><td>Improved</td></tr><tr><td>Solar</td><td>1.3×</td><td>2.0%</td><td>0</td><td>32/32</td></tr><tr><td>Temp</td><td>2.0×</td><td>14.1%</td><td>1</td><td>33/43</td></tr><tr><td>Streamflow</td><td>791×</td><td>18.9%</td><td>1</td><td>89/98</td></tr><tr><td>Methane</td><td>728×</td><td>9.9%</td><td>4</td><td>28/31</td></tr></table>

Table 6. Providing VAE-estimated moments $( \hat { \mu } , \hat { \sigma } )$ as input features to the denoiser. Removing this conditioning consistently degrades performance.
<table><tr><td>Dataset</td><td> $( \hat { \mu } , \hat { \sigma } )$  as input</td><td> $( \hat { \mu } , \hat { \sigma } )$  removed</td></tr><tr><td>Solar</td><td>0.846</td><td>0.831</td></tr><tr><td>Temp</td><td>0.868</td><td>0.842</td></tr><tr><td>Streamflow</td><td>0.596</td><td>0.551</td></tr><tr><td>Methane</td><td>0.788</td><td>0.731</td></tr></table>

Diffusion recovery from prior error. The $\gamma _ { 2 }$ anchoring term ties the reverse process to Y<sup>ˆ</sup> , raising the possibility that inaccurate moments propagate into the final reconstruction. Empirically, the opposite holds: the calibration gain is largest precisely where the prior is weakest. Comparing the $\Phi _ { \psi , \omega }$ and ZeroDiff rows of Table 1, the absolute NSE improvement on Methane is +0.307 (0.481 → 0.788), the largest among the four datasets. Table 5 confirms this at the location level: even on Methane, all 4 locations with negative prior NSE are recovered to positive NSE after diffusion, and 28 of 31 locations overall see improvement.

Moment-conditioned correction. The VAE-estimated moments $( \hat { \mu } , \hat { \sigma } )$ are not only used to construct Y<sup>ˆ</sup> via denormalization—they are also explicitly provided as input features to the denoiser: $\boldsymbol { \epsilon } _ { \theta } = f _ { \theta } ( \mathbf { Y } _ { t } , [ \mathbf { X } ; \hat { \boldsymbol { \mu } } ; \hat { \boldsymbol { \sigma } } ] , \hat { \mathbf { Y } } , t )$ . During training, the denoiser observes a spectrum of $( \hat { \mu } , \hat { \sigma } )$ values across observed locations, each paired with the corresponding ground-truth Y, and learns a correction policy parameterized by these moments rather than a fixed offset relative to any specific prior. Table 6 isolates this effect: removing $( \hat { \mu } , \hat { \sigma } )$ from the denoiser’s input consistently drops NSE, with the largest drops on high-heterogeneity datasets (Streamflow −0.045, Methane −0.057). Since the prior is constructed identically at observed and unobserved locations, this correction transfers to U.

Table 7. Multi-Target reconstruction (NSE). Each row holds out a different number of locations per fold; a single model is trained to reconstruct all held-out targets simultaneously.
<table><tr><td>Locations out</td><td>Streamflow</td><td>Solar</td><td>Temp</td><td>Methane</td></tr><tr><td>1</td><td>0.596</td><td>0.846</td><td>0.868</td><td>0.788</td></tr><tr><td>5</td><td>0.551</td><td>0.843</td><td>0.875</td><td>0.735</td></tr><tr><td>10</td><td>0.554</td><td>0.843</td><td>0.865</td><td>0.749</td></tr><tr><td>20</td><td>0.550</td><td>0.847</td><td>0.855</td><td>0.609</td></tr></table>

## 5.5. Multi-Target Reconstruction

In practical deployments such as continental-scale hydrological modeling, the set U of unobserved locations is large, and training a separate model per target is infeasible. ZeroDiff is designed to serve all targets with a single model: the moment-guided weighting (Eq. 12) uses the centroid $( \hat { \mu } ^ { * } , \hat { \sigma } ^ { * } )$ of estimated moments over U (Appendix E.1), upweighting observed locations whose distributional characteristics are representative of the collective target set.

Table 7 varies the number of held-out locations per fold from 1 to 20. On low-heterogeneity datasets (Solar, Temp), performance is essentially unchanged as |U| grows, indicating that the centroid is an effective summary when targets cluster in moment space. On high-heterogeneity datasets, NSE drops modestly (Streamflow $0 . 5 9 6  0 . 5 5 0$ , Methane $0 . 7 8 8  0 . 6 0 9 )$ as the centroid must compromise across more diverse targets. This degradation is a deliberate tradeoff: one trained model serves all |U| locations, providing an |U|-fold reduction in computational cost relative to perlocation training, with the per-location accuracy gap remaining small when targets share a geographic region.

## 6. Conclusion

ZeroDiff enables zero-shot time series reconstruction by moment estimation and dynamics learning, constructing an informed prior from exogenous variables alone, then calibrating systematic errors through diffusion. The calibration learned on observed locations generalizes to unobserved ones. Experiments across four real-world datasets show that ZeroDiff significantly outperforms existing diffusion methods, particularly on datasets with high cross-location variability where baselines fail entirely.

## Acknowledgments

This research is supported by Dan Lu’s Early Career Project, sponsored by the Office of Biological and Environmental Research in the U.S. Department of Energy (DOE). X.J. was partially supported by the National Science Foundation (NSF) grants 2203581, 2239175, 2316305, 2147195, 2425845, and 2530609; the USGS award G22AC00266; and the NASA grants 80NSSC24K1061 and 80NSSC25K0013.

## Impact Statement

This paper presents ZeroDiff for zero-shot time series reconstruction, with direct applications in hydrological modeling. Potential positive societal impacts include improved flood early warning, water resource management, and environmental monitoring in ungauged regions where monitoring infrastructure is limited. As with any predictive model, we encourage practitioners to appropriately communicate uncertainty estimates when informing real-world decisions.

## References

Addor, N., Newman, A. J., Mizukami, N., and Clark, M. P. The CAMELS data set: catchment attributes and meteorology for large-sample studies. Hydrology and Earth System Sciences, 21(10):5293–5313, 2017. doi: 10.5194/hess-21-5293-2017.

Akata, Z., Perronnin, F., Harchaoui, Z., and Schmid, C. Label-embedding for image classification. IEEE Transactions on Pattern Analysis and Machine Intelligence, 38 (7):1425–1438, 2016.

Alcaraz, J. M. L. and Strodthoff, N. Diffusion-based time series imputation and forecasting with structured state space models. TMLR, 2022.

Alcaraz, J. M. L. and Strodthoff, N. Diffusion-based time series imputation and forecasting with structured state space models. Transactions on Machine Learning Research, 2023.

Ansari, A. F., Stella, L., Turkmen, A. C., Zhang, X., Mer-¨ cado, P., Shen, H., Shchur, O., Rangapuram, S. S., Pineda-Arango, S., Kapoor, S., Zschiegner, J., Maddix, D. C., Wang, H., Mahoney, M. W., Torkkola, K., Wilson, A. G., Bohlke-Schneider, M., and Wang, B. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024, 2024.

Bansal, A., Borgnia, E., Chu, H.-M., Li, J., Kazemi, H., Huang, F., Goldblum, M., Geiping, J., and Goldstein, T. Cold diffusion: Inverting arbitrary image transforms without noise. In Advances in Neural Information Processing Systems, 2023.

Cao, W., Wang, D., Li, J., Zhou, H., Li, L., and Li, Y. BRITS: Bidirectional recurrent imputation for time series. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018.

Chen, Y., Deng, W., Fang, S., Li, F., Yang, N. T., Zhang, Y., Rasul, K., Zhe, S., Schneider, A., and Nevmyvaka, Y. Provably convergent schrodinger bridge with applications ¨ to probabilistic time series imputation. In International Conference on Machine Learning, pp. 4485–4513, 2023.

Das, A., Kong, W., Sen, R., and Zhou, Y. A decoder-only foundation model for time-series forecasting. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024), Vienna, Austria, 2024. OpenReview.net.

Du, W., Cote, D., and Liu, Y. Saits: Self-attention-based imputation for time series. Expert Systems with Applications, 219:119619, 2023.

Farhadi, A., Endres, I., Hoiem, D., and Forsyth, D. Describing objects by their attributes. In CVPR, 2009.

Frome, A., Corrado, G. S., Shlens, J., Bengio, S., Dean, J., Ranzato, M., and Mikolov, T. Devise: A deep visualsemantic embedding model. In NeurIPS, 2013.

Gao, J., Cao, Q., and Chen, Y. Auto-regressive moving diffusion models for time series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 16727–16735, 2025.

Gneiting, T. and Katzfuss, M. Probabilistic forecasting. Annual Review of Statistics and Its Application, 1(1): 125–151, 2014. ISSN 2326-831X. doi: 10.1146/ annurev-statistics-062713-085831.

Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Han, X., Zheng, H., and Zhou, M. CARD: Classification and regression diffusion models. In Advances in Neural Information Processing Systems, 2022.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pp. 6840–6851, 2020.

Hochreiter, S. and Schmidhuber, J. Long short-term memory. Neural Computation, 9(8):1735–1780, 1997.

Hu, J., Liang, Y., Fan, Z., Chen, H., Zheng, Y., and Zimmermann, R. Graph neural processes for spatio-temporal extrapolation. In KDD, 2023.

Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., and Choo, J. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2022.

Kingma, D. P. and Welling, M. Auto-encoding variational bayes. In Bengio, Y. and LeCun, Y. (eds.), 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings, 2014.

Kollovieh, M., Ansari, A. F., Bohlke-Schneider, M., Zschiegner, J., Wang, H., and Wang, Y. Predict, refine, synthesize: Self-guiding diffusion models for probabilistic time series forecasting. In Advances in Neural Information Processing Systems, 2023.

Kratzert, F., Klotz, D., Brenner, C., Schulz, K., and Herrnegger, M. Rainfall–runoff modelling using long short-term memory (lstm) networks. Hydrology and Earth System Sciences, 22(11):6005–6022, 2018.

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., and Nearing, G. S. Toward improved predictions in ungauged basins: Exploiting the power of machine learning. Water Resources Research, 2019.

Lampert, C. H., Nickisch, H., and Harmeling, S. Learning to detect unseen object classes by between-class attribute transfer. In CVPR, 2009.

Lee, S.-g., Kim, H., Shin, C., Tan, X., Liu, C., Meng, Q., Qin, T., Chen, W., Yoon, S., and Liu, T.-Y. Priorgrad: Improving conditional denoising diffusion models with data-dependent adaptive prior. In ICLR, 2022.

Li, Y., Yu, R., Shahabi, C., and Liu, Y. Diffusion convolutional recurrent neural network: Data-driven traffic forecasting. In ICLR, 2018.

Li, Y., Chen, W., Hu, X., Chen, B., and Zhou, M. Transformer-modulated diffusion models for probabilistic multivariate time series forecasting. In International Conference on Learning Representations, 2024.

Liu, Y., Wu, H., Wang, J., and Long, M. Non-stationary transformers: Exploring the stationarity in time series forecasting. In Advances in Neural Information Processing Systems, 2022.

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., and Long, M. itransformer: Inverted transformers are effective for time series forecasting. In ICLR, 2024.

Luo, C. Understanding diffusion models: A unified perspective. arXiv preprint arXiv:2208.11970, 2022.

McGovern, A., Gagne II, D. J., Basara, J., Hamill, T. M., and Margolin, D. Solar energy prediction: An international contest to initiate interdisciplinary research on compelling meteorological problems. Bulletin ofthe American Meteorological Society, 96(8), 2015.

Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.-Y., and Ermon, S. SDEdit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2022.

Newman, A. J., Clark, M. P., Sampson, K., Wood, A. W., Hay, L. E., Bock, A., Viger, R. J., Blodgett, D., Brekke, L., Arnold, J. R., Hopson, T., and Duan, Q. Development of a large-sample watershed-scale hydrometeorological data set for the contiguous USA: data set characteristics and assessment of regional variability in hydrologic model performance. Hydrology and Earth System Sciences, 19 (1):209–223, 2015. doi: 10.5194/hess-19-209-2015.

Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. A time series is worth 64 words: Long-term forecasting with transformers. In ICLR, 2023.

Paletta, Q., Terren-Serrano, G., Nie, Y., et al. Advances in´ solar forecasting: Computer vision with deep learning. Advances in Applied Energy, 2023.

Pastorello, G., Trotta, C., Canfora, E., Chu, H., Christianson, D., Cheah, Y.-W., Poindexter, C., Chen, J., Elbashandy, A., Humphrey, M., et al. The FLUXNET2015 dataset and the ONEFlux processing pipeline for eddy covariance data. Scientific Data, 7(1):225, 2020. doi: 10.1038/ s41597-020-0534-3.

Rasul, K., Seward, C., Schuster, I., and Vollgraf, R. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting. In Proceedings of the 38th International Conference on Machine Learning, volume 139, pp. 8857–8868. PMLR, 2021.

Rishi, J., Mothish, G., and Subramani, D. Conditional diffusion model with nonlinear data transformation for time series forecasting. In ICML, 2025.

Salinas, D., Flunkert, V., Gasthaus, J., and Januschowski, T. Deepar: Probabilistic forecasting with autoregressive recurrent networks. International Journal ofForecasting, 36(3):1181–1191, 2020.

Shen, L. and Kwok, J. Non-autoregressive conditional diffusion models for time series prediction. In ICML, 2023.

Shen, L., Chen, W., and Kwok, J. Multi-resolution diffusion models for time series forecasting. In International Conference on Learning Representations, 2024a.

Shen, L., Chen, W., and Kwok, J. T. Multi-resolution diffusion models for time series forecasting. In ICLR, 2024b.

Shi, X., Wang, S., Nie, Y., Li, D., Ye, Z., Wen, Q., and Jin, M. Time-moe: Billion-scale time series foundation models with mixture of experts. In ICLR, 2025.

Sohn, K., Lee, H., and Yan, X. Learning structured output representation using deep conditional generative models. In NeurIPS, 2015.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

Sun, Y., Chen, S., Chen, S., Qiu, C., Liu, L., Oh, Y., Malone, S. L., McNicol, G., Zhuang, Q., Smith, C., Xie, Y., and Jia, X. X-methanewet: A cross-scale global wetland methane emission benchmark dataset for advancing science discovery with ai. arXiv preprint arXiv:2505.18355, 2025.

Tang, Y., Qu, A., Chow, A. H., Lam, W. H., Wong, S., and Ma, W. Domain adversarial spatial-temporal network: a transferable framework for short-term traffic forecasting across cities. In CIKM, 2022.

Tashiro, Y., Song, J., Song, Y., and Ermon, S. CSDI: Conditional score-based diffusion models for probabilistic time series imputation. In Advances in Neural Information Processing Systems, volume 34, 2021.

U.S. Geological Survey. National hydrography dataset (ver. 2.1), 2019. URL https://www.epa.gov/waterdata/ nhdplus-national-data.

U.S. Geological Survey. Access national hydrography products. Website, 2024. URL https://www.usgs.gov/national-hydrography/ access-national-hydrography-products. Accessed February 25, 2025.

Wu, H., Xu, J., Wang, J., and Long, M. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in Neural Information Processing Systems, 34:22419–22430, 2021.

Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., and Long, M. Timesnet: Temporal 2d-variation modeling for general time series analysis. In ICLR, 2023.

Xian, Y., Akata, Z., Sharma, G., Nguyen, Q., Hein, M., and Schiele, B. Latent embeddings for zero-shot classification. In CVPR, 2016.

Xian, Y., Schiele, B., and Akata, Z. Zero-shot learning–the good, the bad and the ugly. In CVPR, 2017.

Xian, Y., Lorenz, T., Schiele, B., and Akata, Z. Feature generating networks for zero-shot learning. In CVPR, 2018.

Ye, W., Xu, Z., and Gui, N. Non-stationary diffusion for probabilistic time series forecasting. In ICML, 2025.

Yu, B., Yin, H., and Zhu, Z. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. In IJCAI, 2018.

Yuan, X. and Qiao, Y. Diffusion-ts: Interpretable diffusion for general time series generation. In ICLR, 2024.

Yuan, Z., Liu, C., Shen, F., Li, Z., Luo, J., Mao, T., and Wang, Z. MSP-MVS: Multi-granularity segmentation prior guided multi-view stereo. In Proceedings of the AAAI Conference on Artificial Intelligence, 2025a.

Yuan, Z., Luo, J., Shen, F., Li, Z., Liu, C., Mao, T., and Wang, Z. DVP-MVS: Synergize depth-edge and visibility prior for multi-view stereo. In Proceedings of the AAAI Conference on Artificial Intelligence, 2025b.

Zeng, A., Chen, M., Zhang, L., and Xu, Q. Are transformers effective for time series forecasting? In AAAI, 2023.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., and Zhang, W. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the 35th AAAI Conference on Artificial Intelligence (AAAI 2021), pp. 11106–11115. AAAI Press, 2021.

## A. Algorithms for Training and Inference

```latex
Algorithm 1 Training
Input: O, U, $\{ \mathbf { X } ^ { ( k ) } \} , \{ \mathbf { Y } ^ { ( k ) } \} _ { k \in \mathcal { O } } .$ , diffusion steps $T$
Train VAE on $\mathcal { O } \mathrm { : }$ encoder $q _ { \psi } ,$ , feature encoder $\phi ,$ decoder ${ \mathrm { D e c } } _ { \psi }$ // cross-modal moment estimation
For all k: $( \hat { \mu } _ { k } , \hat { \sigma } _ { k } ) \gets \mathrm { D e c } _ { \psi } ( { \bf 0 } , \phi ( { \bf Z } ^ { ( k ) } ) )$ // infer magnitudefrom X
Train dynamics model $f _ { \omega }$ on O in standardized space // shared response pattern
Compute $\begin{array} { r } { ( \hat { \mu } ^ { * } , \hat { \sigma } ^ { * } ) \gets \frac { 1 } { | \mathcal { U } | } \sum _ { j \in \mathcal { U } } ( \hat { \mu } _ { j } , \hat { \sigma } _ { j } ) } \end{array}$ // target centroid in moment space
repeat
Sample $k \in \mathcal { O }$ with probability $\propto w _ { k }$ // moment-guided weighting
Compute $\hat { \mathbf { Y } } ^ { ( k ) } \gets \hat { \mu _ { k } } + \hat { \sigma } _ { k } \cdot \hat { f _ { \omega } } ( \mathbf { X } ^ { ( k ) } )$ // pattern + magnitude
Sample t ∼ Uniform $( 1 , \dots , T \} ) , \epsilon \sim { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } )$
${ \bf Y } _ { t } ^ { ( k ) } \gets \sqrt { \bar { \alpha } _ { t } } { \bf Y } ^ { ( k ) } + ( 1 - \sqrt { \bar { \alpha } _ { t } } ) \hat { { \bf Y } } ^ { ( k ) } + \sqrt { \bar { \sigma } _ { t } } \epsilon$ // diffuse toward prior
Update θ via $\nabla _ { \theta } w _ { k } \| \epsilon - \epsilon _ { \theta } ( { \bf Y } _ { t } ^ { ( k ) } , { \bf X } ^ { ( k ) } , \hat { \mathbf { Y } } ^ { ( k ) } , t ) \| ^ { 2 }$ // calibration on O
until convergence
```

Algorithm 2 Inference   
Input: $j \in \mathcal { U } , \mathbf { X } ^ { ( j ) }$ , trained $\mathrm { V A E } ( \phi , \mathrm { D e c } _ { \psi } ) ,$ dynamics model $f _ { \omega } ,$ denoiser $\epsilon _ { \theta }$   
$( \hat { \mu } _ { j } , \hat { \sigma } _ { j } ) \gets \mathrm { D e c } _ { \psi } ( \mathbf { 0 } , \phi ( \mathbf { Z } ^ { ( j ) } ) )$ // zero-shot moment estimation   
$\hat { \mathbf { Y } } ^ { ( j ) }  \hat { \mu } _ { j } + \hat { \sigma } _ { j } \cdot f _ { \omega } ( \mathbf { X } ^ { ( j ) } )$ // pattern + magnitude   
$\mathbf { Y } _ { T } ^ { ( j ) } \sim \mathcal { N } ( \hat { \mathbf { Y } } ^ { ( j ) } , \bar { \sigma } _ { T } \mathbf { I } )$ // warm-start from prior, not noise   
for $t = T$ to 1 do   
$\boldsymbol { \epsilon } _ { \theta } \gets f _ { \theta } ( \mathbf { Y } _ { t } ^ { ( j ) } , \mathbf { X } ^ { ( j ) } , \hat { \mathbf { Y } } ^ { ( j ) } , t )$ // bidirectional conditioning   
$\begin{array} { r } { \hat { \mathbf { Y } } _ { 0 } ^ { ( j ) } \gets \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \big ( \mathbf { Y } _ { t } ^ { ( j ) } - ( 1 - \sqrt { \bar { \alpha } _ { t } } ) \hat { \mathbf { Y } } ^ { ( j ) } - \sqrt { \bar { \sigma } _ { t } } \epsilon _ { \theta } \big ) } \end{array}$ // reparameterization   
$\pmb { \mu } _ { t - 1 } ^ { ( j ) }  \partial _ { 0 } \hat { \mathbf { Y } } _ { 0 } ^ { ( j ) } + \gamma _ { 1 } \mathbf { Y } _ { t } ^ { ( j ) } + \gamma _ { 2 } \hat { \mathbf { Y } } ^ { ( j ) }$ // three-term posterior   
$\mathbf { Y } _ { t - 1 } ^ { ( j ) }  \pmb { \mu } _ { t - 1 } ^ { ( j ) } + \sigma _ { t } \mathbf { z } , \quad \mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) \mathrm { i f } t > 1$   
end for   
Output: $\mathbf { Y } _ { 0 } ^ { ( j ) }$  
Remark. Unlike standard DDPM that initializes from $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , our method performs warm-start diffusion from the informed prior $\hat { \mathbf { Y } } ^ { ( j ) }$ . The prior anchors the reverse process at three levels: (i) initialization of $\mathbf { Y } _ { T } ^ { ( j ) }$ , (ii) the reparameterization of $\hat { \mathbf { Y } } _ { 0 } ^ { ( j ) }$ , and (iii) the $\gamma _ { 2 }$ term in the posterior mean. This transforms diffusion from generation to calibration—refining a reasonable estimate rather than constructing from noise. The coefficients $\gamma _ { 0 } , \gamma _ { 1 } , \gamma _ { 2 }$ , the variance schedule ${ \bar { \sigma } } _ { t } ,$ and the three-term posterior mean are derived in Appendix B.

## B. Bayesian Derivation of the Posterior Distribution

We derive the posterior distribution for our informed-prior diffusion process. For notational simplicity, we omit the location index and write Y for the target, Y<sup>ˆ</sup> for the informed prior, and X for the exogenous input. Unlike standard DDPM which diffuses toward pure noise $\mathbf { Y } _ { T } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , we diffuse toward the informed prior $\hat { \mathbf Y }$ , enabling warm-start initialization for calibration.

## B.1. Forward Process

The one-step forward transition is:

$$
\mathbf { Y } _ { t } = \sqrt { \alpha _ { t } } \mathbf { Y } _ { t - 1 } + ( 1 - \sqrt { \alpha _ { t } } ) \hat { \mathbf { Y } } + \sqrt { \beta _ { t } } \epsilon _ { t } , \quad \epsilon _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )\tag{15}
$$

where $\begin{array} { r } { \alpha _ { t } : = 1 - \beta _ { t } , \bar { \alpha } _ { t } : = \prod _ { i = 1 } ^ { t } \alpha _ { i } , } \end{array}$ , and $\bar { \sigma } _ { t } : = 1 - \bar { \alpha } _ { t }$ . Unrolling recursively:

$$
\begin{array} { r l } & { \mathbf { Y } _ { t } = \sqrt { \alpha _ { t } } \mathbf { Y } _ { t - 1 } + ( 1 - \sqrt { \alpha _ { t } } ) \hat { \mathbf { Y } } + \sqrt { \beta _ { t } } \epsilon _ { t } } \\ & { \quad = \sqrt { \alpha _ { t } } \left[ \sqrt { \alpha _ { t - 1 } } \mathbf { Y } _ { t - 2 } + ( 1 - \sqrt { \alpha _ { t - 1 } } ) \hat { \mathbf { Y } } + \sqrt { \beta _ { t - 1 } } \epsilon _ { t - 1 } \right] + ( 1 - \sqrt { \alpha _ { t } } ) \hat { \mathbf { Y } } + \sqrt { \beta _ { t } } \epsilon _ { t } } \\ & { \quad = \sqrt { \alpha _ { t } \alpha _ { t - 1 } } \mathbf { Y } _ { t - 2 } + \left( 1 - \sqrt { \alpha _ { t } \alpha _ { t - 1 } } \right) \hat { \mathbf { Y } } + \sqrt { \alpha _ { t } \beta _ { t - 1 } } + \beta _ { t } \epsilon } \\ & { \quad \vdots } \\ & { \quad \quad = \sqrt { \bar { \alpha } _ { t } } \mathbf { Y } + ( 1 - \sqrt { \bar { \alpha } _ { t } } ) \hat { \mathbf { Y } } + \sqrt { \bar { \sigma } _ { t } } \epsilon } \end{array}\tag{16}
$$

Thus $q ( \mathbf { Y } _ { t } | \mathbf { Y } , \hat { \mathbf { Y } } ) = \mathcal { N } \left( \sqrt { \bar { \alpha } _ { t } } \mathbf { Y } + ( 1 - \sqrt { \bar { \alpha } _ { t } } ) \hat { \mathbf { Y } } , \bar { \sigma } _ { t } \mathbf { I } \right)$ , and at $t = T \colon \mathbf { Y } _ { T } \sim \mathcal { N } ( \hat { \mathbf { Y } } , \bar { \sigma } _ { T } \mathbf { I } )$ —the warm-start initialization centered at the prior.

## B.2. Reverse Posterior Derivation

We derive $q ( \mathbf { Y } _ { t - 1 } | \mathbf { Y } _ { t } , \mathbf { Y } , \hat { \mathbf { Y } } )$ via Bayes’ rule:

$$
q ( \mathbf { Y } _ { t - 1 } | \mathbf { Y } _ { t } , \mathbf { Y } , { \hat { \mathbf { Y } } } ) \propto q ( \mathbf { Y } _ { t } | \mathbf { Y } _ { t - 1 } , { \hat { \mathbf { Y } } } ) \cdot q ( \mathbf { Y } _ { t - 1 } | \mathbf { Y } , { \hat { \mathbf { Y } } } )\tag{17}
$$

From Eq. (15), the transition density is:

$$
q ( \mathbf { Y } _ { t } | \mathbf { Y } _ { t - 1 } , { \hat { \mathbf { Y } } } ) = { \mathcal { N } } \left( { \sqrt { \alpha _ { t } } } \mathbf { Y } _ { t - 1 } + ( 1 - { \sqrt { \alpha _ { t } } } ) { \hat { \mathbf { Y } } } , \beta _ { t } \mathbf { I } \right)\tag{18}
$$

From Eq. (16) at $t - 1$ :

$$
q ( \mathbf { Y } _ { t - 1 } | \mathbf { Y } , \hat { \mathbf { Y } } ) = \mathcal { N } \left( \sqrt { \bar { \alpha } _ { t - 1 } } \mathbf { Y } + ( 1 - \sqrt { \bar { \alpha } _ { t - 1 } } ) \hat { \mathbf { Y } } , \bar { \sigma } _ { t - 1 } \mathbf { I } \right)\tag{19}
$$

Define $\mathbf { A } : = \mathbf { Y } _ { t } - ( 1 - \sqrt { \alpha _ { t } } ) \hat { \mathbf { Y } }$ and $\mathbf { B } : = \sqrt { \bar { \alpha } _ { t - 1 } } \mathbf { Y } + ( 1 - \sqrt { \bar { \alpha } _ { t - 1 } } ) \hat { \mathbf { Y } }$ . The product of two Gaussians gives:

$$
\begin{array} { l } { { q ( { \bf Y } _ { t - 1 } | { \bf Y } _ { t } , { \bf Y } , \hat { \bf Y } ) \propto \exp \left( - \frac { ( { \bf A } - \sqrt { \alpha _ { t } { \bf Y } _ { t - 1 } } ) ^ { 2 } } { 2 \beta _ { t } } - \frac { ( { \bf Y } _ { t - 1 } - { \bf B } ) ^ { 2 } } { 2 \bar { \sigma } _ { t - 1 } } \right) } } \\ { { \mathrm { } = \exp \left( - \frac { 1 } { 2 } \left[ \left( \frac { \alpha _ { t } } { \beta _ { t } } + \frac { 1 } { \bar { \sigma } _ { t - 1 } } \right) { \bf Y } _ { t - 1 } ^ { 2 } - 2 \left( \frac { \sqrt { \alpha _ { t } } { \bf A } } { \bar { \beta } _ { t } } + \frac { { \bf B } } { \bar { \sigma } _ { t - 1 } } \right) { \bf Y } _ { t - 1 } + \mathrm { c o n s t } \right] \right) } } \end{array}\tag{20}
$$

Completing the square in $\mathbf { Y } _ { t - 1 }$ , the posterior is Gaussian with:

$$
\tilde { \sigma } _ { t - 1 } = \left( \frac { \alpha _ { t } } { \beta _ { t } } + \frac { 1 } { \bar { \sigma } _ { t - 1 } } \right) ^ { - 1 } = \frac { \beta _ { t } \bar { \sigma } _ { t - 1 } } { \alpha _ { t } \bar { \sigma } _ { t - 1 } + \beta _ { t } }\tag{21}
$$

$$
\tilde { \mu } _ { t - 1 } = \tilde { \sigma } _ { t - 1 } \left( \frac { \sqrt { \alpha _ { t } } \mathbf { A } } { \beta _ { t } } + \frac { \mathbf { B } } { \bar { \sigma } _ { t - 1 } } \right) = \frac { \sqrt { \alpha _ { t } } \bar { \sigma } _ { t - 1 } \mathbf { A } + \beta _ { t } \mathbf { B } } { \alpha _ { t } \bar { \sigma } _ { t - 1 } + \beta _ { t } }\tag{22}
$$

Substituting $\mathbf { A } = \mathbf { Y } _ { t } - ( 1 - \sqrt { \alpha _ { t } } ) \hat { \mathbf { Y } } \mathrm { a n d } \mathbf { B } = \sqrt { \bar { \alpha } _ { t - 1 } } \mathbf { Y } + ( 1 - \sqrt { \bar { \alpha } _ { t - 1 } } ) \hat { \mathbf { Y } }$ into Eq. (22):

$$
\begin{array} { r l } & { \tilde { \mu } _ { t - 1 } = \frac { \sqrt { \alpha _ { t } } \bar { \sigma } _ { t - 1 } \left[ \mathbf { Y } _ { t } - ( 1 - \sqrt { \alpha _ { t } } ) \hat { \mathbf { Y } } \right] + \beta _ { t } \left[ \sqrt { \bar { \alpha } _ { t - 1 } } \mathbf { Y } + \left( 1 - \sqrt { \bar { \alpha } _ { t - 1 } } \right) \hat { \mathbf { Y } } \right] } { \alpha _ { t } \bar { \sigma } _ { t - 1 } + \beta _ { t } } } \\ & { \qquad = \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { \alpha _ { t } \bar { \sigma } _ { t - 1 } + \beta _ { t } } \mathbf { Y } + \frac { \sqrt { \alpha _ { t } } \bar { \sigma } _ { t - 1 } } { \alpha _ { t } \bar { \sigma } _ { t - 1 } + \beta _ { t } } \mathbf { Y } _ { t } + \frac { \sqrt { \alpha _ { t } } \left( \sqrt { \alpha _ { t } } - 1 \right) \bar { \sigma } _ { t - 1 } + \left( 1 - \sqrt { \bar { \alpha } _ { t - 1 } } \right) \beta _ { t } } { \alpha _ { t } \bar { \sigma } _ { t - 1 } + \beta _ { t } } \hat { \mathbf { Y } } } \end{array}\tag{23}
$$

Thus the posterior mean has the three-term form:

$$
\boxed { \tilde { \mu } _ { t - 1 } = \gamma _ { 0 } \mathbf { Y } + \gamma _ { 1 } \mathbf { Y } _ { t } + \gamma _ { 2 } \hat { \mathbf { Y } } }\tag{24}
$$

where:

$$
\gamma _ { 0 } = \frac { \sqrt { \bar { \alpha } _ { t - 1 } } \beta _ { t } } { \alpha _ { t } \bar { \sigma } _ { t - 1 } + \beta _ { t } } , \quad \gamma _ { 1 } = \frac { \sqrt { \alpha _ { t } } \bar { \sigma } _ { t - 1 } } { \alpha _ { t } \bar { \sigma } _ { t - 1 } + \beta _ { t } } , \quad \gamma _ { 2 } = \frac { \sqrt { \alpha _ { t } } ( \sqrt { \alpha _ { t } } - 1 ) \bar { \sigma } _ { t - 1 } + ( 1 - \sqrt { \bar { \alpha } _ { t - 1 } } ) \beta _ { t } } { \alpha _ { t } \bar { \sigma } _ { t - 1 } + \beta _ { t } }\tag{25}
$$

This differs from standard DDPM where $\tilde { \mu } _ { t - 1 } ^ { \mathrm { D D P M } } = \gamma _ { 0 } \mathbf { Y } + \gamma _ { 1 } \mathbf { Y } _ { t }$ . The additional $\gamma _ { 2 } \hat { \mathbf Y }$ term anchors each denoising step to the prior prediction.

## B.3. Inference via Reparameterization

During inference, Y is unknown. From Eq. (16), we estimate:

$$
\hat { \mathbf { Y } } _ { 0 } = \frac { 1 } { \sqrt { \bar { \alpha } _ { t } } } \left( \mathbf { Y } _ { t } - ( 1 - \sqrt { \bar { \alpha } _ { t } } ) \hat { \mathbf { Y } } - \sqrt { \bar { \sigma } _ { t } } \epsilon _ { \theta } ( \mathbf { Y } _ { t } , \mathbf { X } , \hat { \mathbf { Y } } , t ) \right)\tag{26}
$$

where $\epsilon _ { \theta }$ is the learned denoiser. The reverse step is:

$$
\mathbf Y _ { t - 1 } = \gamma _ { 0 } \hat { \mathbf Y } _ { 0 } + \gamma _ { 1 } \mathbf Y _ { t } + \gamma _ { 2 } \hat { \mathbf Y } + \sqrt { \tilde { \sigma } _ { t - 1 } } \mathbf z , \quad \mathbf z \sim \mathcal N ( \mathbf 0 , \mathbf I )\tag{27}
$$

The prior $\hat { \mathbf Y }$ anchors the process at three levels: (1) initialization $\mathbf Y _ { T } \sim \mathcal N ( \hat { \mathbf Y } , \bar { \sigma } _ { T } \mathbf I )$ , (2) reparameterization of $\hat { \mathbf { Y } } _ { 0 } .$ , and (3) the $\gamma _ { 2 } \hat { \mathbf Y }$ term in the posterior mean. This triple anchoring transforms diffusion from generation to calibration: the network learns residual corrections rather than the full data distribution.

## C. Case Study: Full-Series Reconstruction on an Ungauged Basin

![](images/07122f405ef0a7a2d565aa8f7725ede70b06c9c5a77f337d4910a482d661bc00.jpg)  
Figure 4. Full time series reconstruction for Basin 02065500 over the entire 13-year training period (1989–2001, 4,745 days).

## D. Dataset Details

We describe the four datasets, their exogenous variables, and train/test splits.

## D.1. Dataset Overview

Table 8. Dataset statistics: temporal coverage, train/test splits, and exogenous variable counts. All datasets use daily resolution.
<table><tr><td>DATASET</td><td>FULL PERIOD</td><td>TRAIN PERIOD</td><td>DAYS</td><td>TEST LOC.</td><td>DYNAMIC</td><td>STATIC</td><td>EXOG.</td></tr><tr><td>CAMELS</td><td>1989-2009</td><td>1989-2001</td><td>4,745</td><td>100</td><td>6</td><td>27</td><td>33</td></tr><tr><td>Temp</td><td>2010-2012</td><td>2010-2012</td><td>1,095</td><td>42</td><td>8</td><td>0</td><td>8</td></tr><tr><td>Solar</td><td>1994–2007</td><td>1994-2004</td><td>4,015</td><td>30</td><td>45</td><td>3</td><td>48</td></tr><tr><td>Methane</td><td>2008-2018</td><td>2008-2018</td><td>4,015</td><td>30</td><td>7</td><td>9</td><td>16</td></tr></table>

## D.2. Dataset Partition

The data splitting strategy for zero-shot reconstruction differs fundamentally from time series forecasting. While forecasting typically partitions data along the temporal dimension, our task requires spatial partitioning: for each experiment, we mask all target observations at a held-out location while retaining its exogenous variables, then leverage data from remaining locations to reconstruct the masked targets.

We reserve a subset of locations as the validation set. Although their targets are masked during training, we use their reconstruction performance to tune hyperparameters. The remaining held-out locations form the test set for final evaluation. This spatial cross-validation ensures that hyperparameters are selected without leaking information from test locations.

For most baselines, we do not adopt the default hyperparameters reported in their original papers, as these were optimized for forecasting tasks. Instead, we conduct task-specific hyperparameter search on our validation set to ensure fair comparison.

Leave-One-Location-Out Evaluation. We adopt a leave-one-location-out protocol to evaluate zero-shot generalization. For each dataset, we iteratively hold out one target location: all target observations at this location are masked while exogenous variables remain available. The model is trained on the remaining locations and evaluated on its ability to reconstruct the held-out target series. This procedure is repeated for all target locations—100 times for CAMELS, 42 for NHD, and 30 for both Solar and Methane. Final metrics are averaged across all held-out locations to provide a robust estimate of zero-shot reconstruction performance.

## E. Efficiency

## E.1. Multi-Target Reconstruction

A natural question arises: why not train a separate model for each target location $j \in \mathcal { U } ?$ Location-specific training would allow moment-guided weighting to focus entirely on $( \hat { \mu } _ { j } , \hat { \sigma } _ { j } )$ , potentially yielding better reconstruction for that specific location.

While this approach maximizes per-location performance, it becomes impractical when |U| is large. Training and storing separate models for each target location incurs computational cost linear in |U|—prohibitive in applications such as continental-scale hydrological modeling where thousands of ungauged locations require reconstruction.

Our framework addresses this via the moment centroid. Let $( \hat { \mu } ^ { * } , \hat { \sigma } ^ { * } )$ denote the centroid of estimated statistics over $j \in \mathcal { U } \colon$

$$
( \hat { \mu } ^ { * } , \hat { \sigma } ^ { * } ) = \frac { 1 } { | \mathcal { U } | } \sum _ { j \in \mathcal { U } } ( \hat { \mu } _ { j } , \hat { \sigma } _ { j } )\tag{28}
$$

By defining $( \hat { \mu } ^ { * } , \hat { \sigma } ^ { * } )$ as the geometric center over all $j \in \mathcal { U } ,$ , a single model serves multiple target locations simultaneously. The moment-guided weighting $w _ { k }$ (Eq. 12) then upweights training locations whose distributions are representative of the collective target set, rather than any individual target.

This design reflects a deliberate trade-off: we accept a modest reduction in per-location accuracy in exchange for an $| \mathscr { U } | \mathrm { - f o l d }$ improvement in efficiency. In practice, when target locations cluster in moment space—as is common within a geographic region—the centroid closely approximates individual targets, and the performance gap is minimal. Table 7 in the main text validates this empirically: as $| \boldsymbol { \mathcal { U } } |$ grows from 1 to 20, performance remains nearly unchanged on low-heterogeneity datasets (Solar, Temp) and degrades only modestly on high-heterogeneity datasets, while training cost is reduced by a factor of $| \boldsymbol { \mathcal { U } } |$

For applications requiring maximum accuracy at a specific location, one can set $\mathcal { U } = \{ j \}$ and train a location-specific model. Our framework supports both modes: efficient batch reconstruction when throughput matters, and focused single-target reconstruction when accuracy is paramount.

## E.2. Diffusion Computational Efficiency Analysis

Table 9. Computational efficiency of six diffusion-based imputation models across all datasets. Models are sorted by total per-fold time (ascending) within each panel. All datasets use 365-day sequences. Training uses early stopping with patience of 20 epochs. Al experiments conducted on H100.
<table><tr><td>Model</td><td>#Params</td><td>T</td><td>Ns</td><td>Train (min)</td><td>Infer (min)</td><td>ms/sample</td><td>Total (min)</td></tr><tr><td colspan="8">(a) Streamflow (531 locations, 33 features, 6,903 samples/split)</td></tr><tr><td>ZeroDiff</td><td>271 K</td><td>20</td><td>100</td><td>4.3</td><td>9.1</td><td>39.4</td><td>13.4</td></tr><tr><td>DiffusionTS†</td><td>2.33M</td><td>100</td><td>1</td><td>12.1</td><td>3.0</td><td>21.4</td><td>15.1</td></tr><tr><td>NsDiff</td><td>270 K</td><td>20</td><td>100</td><td>16.8</td><td>9.2</td><td>39.9</td><td>26.0</td></tr><tr><td>CSDI</td><td>618K</td><td>50</td><td>10</td><td>23.0</td><td>44.4</td><td>192.9</td><td>67.4</td></tr><tr><td>SSSD‡</td><td>29.4M</td><td>200</td><td>10</td><td>12.2</td><td>165.8</td><td>720.5</td><td>178.0</td></tr><tr><td>CSBI</td><td>137K</td><td>100</td><td>20</td><td>19.1</td><td>160.0</td><td>695.4</td><td>179.1</td></tr><tr><td colspan="8">(b) Solar (500 locations, 48 features, 5,500 samples/split)</td></tr><tr><td>DiffusionTS†</td><td>2.32M</td><td>100</td><td>1</td><td>6.3</td><td>4.0</td><td>22.0</td><td>10.3</td></tr><tr><td>ZeroDiff</td><td>270 K</td><td>20</td><td>100</td><td>4.2</td><td>7.2</td><td>39.5</td><td>11.4</td></tr><tr><td>NsDiff</td><td>269 K</td><td>20</td><td>100</td><td>12.4</td><td>7.3</td><td>39.9</td><td>19.7</td></tr><tr><td>CSDI</td><td>617K</td><td>50</td><td>10</td><td>12.3</td><td>35.4</td><td>192.9</td><td>47.7</td></tr><tr><td>CSBI</td><td>137K</td><td>100</td><td>20</td><td>15.2</td><td>127.2</td><td>693.8</td><td>142.4</td></tr><tr><td>SSSD</td><td>29.4M</td><td>200</td><td>10</td><td>10.8</td><td>132.4</td><td>722.2</td><td>143.2</td></tr><tr><td colspan="8">(c) Temp (45 locations, 8 features, 135 samples/split)</td></tr><tr><td>ZeroDiff</td><td>269 K</td><td>20</td><td>100</td><td>0.6</td><td>0.5</td><td>77.0</td><td>1.1</td></tr><tr><td>NsDiff</td><td>269 K</td><td>20</td><td>100</td><td>0.8</td><td>0.5</td><td>76.9</td><td>1.3</td></tr><tr><td>DiffusionTS†</td><td>2.32M</td><td>100</td><td>1</td><td>1.8</td><td>0.3</td><td>41.0</td><td>2.1</td></tr><tr><td>CSDI</td><td>617K</td><td>50</td><td>10</td><td>1.6</td><td>2.2</td><td>332.1</td><td>3.8</td></tr><tr><td>SSSD‡</td><td>29.4M</td><td>200</td><td>10</td><td>1.0</td><td>5.2</td><td>770.0</td><td>6.2</td></tr><tr><td>CSBI</td><td>137K</td><td>100</td><td>20</td><td>1.0</td><td>12.3</td><td>1,821.0</td><td>13.3</td></tr></table>

## F. Fairness Details

Most baselines are designed for time series forecasting with look-back window $L _ { b }$ and prediction horizon H. To fairly evaluate on reconstruction $( L = 3 6 5 )$ , we adopt two protocols.

## F.1. Sliding Window Reconstruction

We partition the sequence into segments where baselines use ground-truth Y from observed locations as look-back to predict subsequent horizons. We consider $L _ { b } , H \in \{ 9 6 , 1 9 2 , 3 6 5 \}$ , including the $L _ { b } = H = 3 6 5$ setting where the mode reconstructs the current window conditioned on itself as a calibration baseline. Predictions are concatenated with stride H to cover the full sequence. The optimal $( L _ { b } , H )$ for each baseline is selected via validation set performance and reported in Table 1.

## F.2. Exogenous-Conditioned Reconstruction

For methods supporting exogenous inputs, we provide the informed prior $\hat { \mathbf { Y } } _ { 1 : L }$ reconstructed from $\mathbf { X } _ { 1 : L }$ as the look-back window, matching our zero-shot setting where no ground-truth targets are available. This protocol evaluates whether baselines can refine an exogenous-based estimate.

Note that protocol (i) provides baselines with oracle access to target observations—an advantage unavailable in true zero-sho scenarios.

## F.3. Hyperparameter Tuning

For zero-shot time series reconstruction, dataset splitting differs fundamentally from time series forecasting. While forecasting typically partitions data along the temporal dimension, our task partitions along the spatial dimension. In each experiment, we mask all target observations at a held-out location while retaining its exogenous variables, then leverage data from other locations to perform zero-shot prediction of the masked target variable. We reserve several locations as a validation set; although their targets are masked during inference, we use their ground-truth values for hyperparameter selection.

For most baselines, we do not use the default hyperparameters reported in their original papers, as those were tuned for time series forecasting rather than our reconstruction task. Instead, we conduct task-specific hyperparameter search. Our experiments reveal that among $L _ { b } , H \in \{ 9 6 , 1 9 2 , 3 6 5 \}$ , the configuration $L _ { b } = H = 3 6 5$ performs best for most baselines—reconstructing a full year from a full year of context. We attribute this to the misalignment artifacts that arise when $L _ { b }$ or H take smaller values, as discontinuous segments must be concatenated to cover the annual sequence.

## G. Gradient Isolation vs. Joint Optimization of the Prior and Denoiser

ZeroDiff trains the prior construction $\Phi _ { \psi , \omega }$ and the diffusion denoiser $\epsilon _ { \theta }$ in two stages (Algorithm 1), so that no gradient flows between the two components. $\mathbf { A }$ natural alternative is joint (end-to-end) optimization, where the diffusion loss backpropagates into the dynamics model through a differentiable denormalization–renormalization chain. We evaluate this variant on all four datasets, initializing from the two-stage solution and jointly fine-tuning with a learning rate of $1 0 ^ { - 5 }$ for the prior components and $1 0 ^ { - 3 }$ for the denoiser.

Table 10. Two-stage training vs. joint optimization (NSE) across all four datasets.
<table><tr><td>Dataset</td><td>Two-stage</td><td>Joint</td><td> $\Delta$ </td></tr><tr><td>Solar</td><td>0.846</td><td>0.855</td><td>+0.009</td></tr><tr><td>Temp</td><td>0.868</td><td>0.886</td><td>+0.018</td></tr><tr><td>Streamflow</td><td>0.596</td><td>0.631</td><td>+0.035</td></tr><tr><td>Methane</td><td>0.788</td><td>0.758</td><td>-0.030</td></tr></table>

Table 10 reveals a dataset-dependent trade-off. Under gradient isolation, the prior construction solves a pure $\mathbf X \to \mathbf Y$ regression problem, independent of the denoiser $\epsilon _ { \theta } .$ . Under joint optimization, its role changes: it is optimized under the diffusion loss and can drift toward producing priors that a specific O-trained denoiser finds easy to calibrate. The two components can thus co-adapt on observed locations $\mathcal { O }$ in ways that are not part of the underlying $\mathbf X \to \mathbf Y$ relationship, and such co-adaptation does not necessarily transfer to unobserved locations $\mathcal { U } .$

How much room exists for co-adaptation depends on how strongly the data constrain the prior. When the X → Y coupling is strong (Solar, Temp, Streamflow), the regression signal pulls the prior toward a narrow band of solutions, and joint optimization delivers modest gains (+0.009 to +0.035 NSE). On Methane—where the flux depends on soil and hydrological factors absent from the exogenous inputs (cf. § 5.4)—the regression signal is weaker, the prior is underdetermined, and co adaptation degrades zero-shot transfer (−0.030). This is consistent with Methane being the dataset on which cross-location transfer is most fragile (Table 1).

Two-stage training enforces gradient isolation between the prior and the denoiser, avoiding co-adaptation by construction—a robust default for zero-shot reconstruction. Joint optimization is a viable refinement when domain knowledge indicates a strong, well-identified $\mathbf X \to \mathbf Y$ coupling. Whether joint schemes can retain these gains while suppressing co-adaptation remains an open direction.

## H. Hyperparameter Search

We conduct a systematic hyperparameter sensitivity analysis for the diffusion component of ZeroDiff across all four datasets. We vary one factor at a time while keeping the remaining settings at their defaults $( d _ { \mathrm { m o d e l } } = 5 1 2 , n _ { \mathrm { l a y e r s } } = 2 , T = 2 0$ $\mathrm { l r } = 1 0 ^ { - 3 }$ , no fusion). Results are summarized in Figure 5.

![](images/cb4e5b113a61ec4f6aa368cb2ce7b38218a72ef86e95f00ffd48ff3cb0a47464.jpg)

![](images/d6f5f87e93055f4f213a194946af8c75a4f149468632547cfe2f3e99fbd6f028.jpg)

![](images/9f3b40646695b96819a2569c4774103f6d4776ca2c73190279bea6538ad07ad6.jpg)

(d) Fusion Type  
![](images/f10bcf9f78d43f630004a41ce24ef9da0741c94a4d85ce000ba594be602e6cd1.jpg)  
Figure 5. Hyperparameter sensitivity of ZeroDiff across four datasets, measured by NSE. Each panel varies one factor: (a) diffusion steps T, (b) denoiser capacity $( d _ { \mathrm { { m o d e l } } } / n _ { \mathrm { { l a y e r s } } } )$ , (c) learning rate, and (d) exogenous fusion strategy.

Diffusion steps. Increasing T from 10 to 50 yields modest but consistent gains on most datasets, with $T = 5 0$ achieving the best NSE on Streamflow, Temp, and Methane. The improvement is marginal relative to the added computational cost, so we adopt $T = 2 0$ as the default for efficiency.

Model capacity. Smaller denoisers $( d _ { \mathrm { m o d e l } } = 2 5 6 )$ perform comparably to or better than larger ones $( d _ { \mathrm { m o d e l } } = 5 1 2 )$ at moderate depth. However, over-parameterization can be harmful: the $5 1 2 / 4$ configuration causes a severe performance collapse on Solar (NSE = 0.05) and a notable drop on Temp, indicating overfitting when the denoiser capacity exceeds what the calibration task requires.

Learning rate. Performance is relatively stable across the tested range $( 1 0 ^ { - 3 } \mathrm { t o } 1 0 ^ { - 4 } )$ . Solar benefits most from a lower learning rate $( 1 0 ^ { - 4 } )$ , while other datasets show mild improvements or remain flat. We use $1 0 ^ { - 3 }$ as the default.

Fusion type. The exogenous fusion strategy has a dataset-dependent effect. Scatter fusion improves Streamflow and Methane, while interference fusion benefits Temp. No single fusion strategy dominates across all datasets, so we default to no fusion for simplicity and leave fusion selection as a dataset-specific tuning option.