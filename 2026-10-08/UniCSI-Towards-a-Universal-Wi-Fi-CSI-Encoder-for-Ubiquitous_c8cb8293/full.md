# UniCSI Towards a Universal Wi-Fi CSI Encoder for Ubiquitous Human Sensing

Daniel Eckhof <sup>1</sup>, Hua Kang <sup>1</sup>, Zhitang Chen <sup>1</sup>, Jie Chuai <sup>1</sup>

<sup>1</sup>Huawei Noah’s Ark Lab

{eckhof.daniel1,kang.hua1,chenzhitang2,chuaijie}@huawei.com

## Abstract

Wi-Fi sensing promises to turn the everyday wireless signals that already surround us into ubiquitous sensors for human sensing. However, a fundamental obstacle is that CSI is acquired under diverse device-specific configurations, including diferent subcarrier counts, bandwidths, and carrier bands. Consequently, the resulting CSI tensors vary in both spectral resolution and tensor shape, making heterogeneous modeling challenging. Standard architectures struggle with such heterogeneity, forcing lossy pre-processing which compromises the underlying signal. To bridge this gap, we present UniCSI, a unified foundation architecture that directly operates on heterogeneous CSI while preserving the integrity of the native waveform. UniCSI hinges on two core innovations: (1) a physics-informed RF tokenizer that encodes each frequency channel based on its fractional position within the physical spectrum rather than rigid array indices. It preserves intrinsic spectral coherence and enables seamless, frequency resolution-agnostic processing across arbitrary sensing configurations. (2) a spectral aggregator that distills variablelength channel sequences into a fixed-size spectral signature, efectively decoupling the feature dimensionality from the physical subcarrier spacing. Extensive evaluations on a large-scale corpus of 25 heterogeneous public datasets, spanning 14 to 2048 subcarriers, 20 to 160 MHz bandwidth, and the 2.4 and 5 GHz bands, demonstrate that native heterogeneous ingestion substantially improves cross-domain transfer under both supervised and self-supervised training schemes, particularly in regimes where fixed-grid architectures fail to generalize.

## Introduction

Wireless sensing has emerged as a promising paradigm for ubiquitous human-centric perception, with commodity Wi-Fi devices supporting applications from activity recognition and gesture interaction to health monitoring and indoor localization (Wang et al. 2014; Zhang et al. 2021; Abdelnasser, Harras, and Youssef 2015; Adib et al. 2014). Compared with vision-based sensing, it preserves privacy, operates under poor lighting and occlusion, and leverages existing infrastructure without wearables or dedicated sensors. Among wireless signals, Channel State Information (CSI) has become the dominant sensing modality due to its fine-grained characterization of wireless propagation and its sensitivity to environmental dynamics.

Despite the remarkable progress of Wi-Fi sensing, existing solutions sufer from poor cross-domain generalization because CSI measurements are inherently sensitive to environments, devices, and wireless configurations. The resulting distribution shifts significantly limit model transferability across deployment scenarios. Large-scale pretraining ofers a promising solution, learning transferable representations from diverse datasets. Unlike vision datasets, however, which share a common representation format, Wi-Fi sensing datasets are inherently heterogeneous, difering in signal dimensions and acquisition configuration such as subcarrier count, bandwidth, carrier frequency, number of devices, and antenna configuration. These diferences are not only observed across devices and datasets, but may also vary over time for the same device as communication parameters are dynamically reconfigured.

Existing solutions (Jiang et al. 2025; Zhu et al. 2026; Luo, Li, and Liu 2026; Kim et al. 2026) handle this heterogeneity through ad hoc preprocessing, such as zero-padding, subcarrier resampling, or per-device backbones, each of which wastes capacity, distorts the signal, or precludes knowledge sharing. None yields a single model that exploits heterogeneous datasets within a unified framework.

To address these limitations, we introduce UniCSI, a carrier-frequency- and bandwidth-agnostic foundation model for Wi-Fi sensing, built on two components. The first is a per-subcarrier tokenizer that accepts any subcarrier count without padding or resampling, encoding each subcarrier by its fractional position within the observed band rather than by a fixed array index, so that the representation depends on where a subcarrier sits in the band rather than on how finely the band is sampled. Because each antenna link is embedded separately by this same tokenizer, the scheme extends to any MIMO configuration without assuming a fixed antenna layout. The second is a cross-attention module that compresses variable-length spectra into a fixed latent representation, independent of input length and sampling density. Together, these enable a single model to ingest CSI of any shape natively.

In particular, this work makes the following contributions. (1) We propose, to the best of our knowledge, the first CSI encoder that natively ingests measurements of arbitrary subcarrier count, bandwidth, and carrier band without padding or resampling, and that extends to varying MIMO configurations through per-link encoding. (2) We show that JEPA pretraining on a large heterogeneous CSI corpus yields transferable representations, and characterize how much this pretraining benefits from heterogeneity across datasets, bands, and bandwidths. (3) We show that on dificult cross-domain benchmarks, fixed-grid architectures collapse toward chance while UniCSI transfers, establishing native heterogeneous ingestion as a prerequisite for cross-domain generalization rather than a convenience.

## Related Work

Wi-Fi sensing has evolved from CNN- and RNN-based models toward Transformer architectures that better capture longrange dependencies (Yang et al. 2023a; Zhang et al. 2021; Yang et al. 2023d; Zhou et al. 2023; Strohmayer, Wödlinger, and Kampel 2024). These models, however, are typically designed for a single task under homogeneous conditions, and transfer poorly once devices, environments, or configurations shift. This fragility has motivated a line of Wi-Fi sensing foundation models, which share a single backbone across tasks but difer in how, and how much, they confront the heterogeneity of CSI.

Jiang et al. (2025) harmonize datasets into a common format before pretraining, converting every sample to amplitude, forcing a canonical single-transmitter three-receiveantenna layout, and resampling the temporal and subcarrier axes to fixed lengths to yield a fixed 600  90 input. This resample-to-a-fixed-grid strategy is precisely what we seek to avoid, since interpolating the subcarrier axis discards the physical spacing of the measurement and normalizes away the heterogeneity a foundation model should exploit.

WiSenseNet (Lyons, Pandey, and Santra 2025) and AM-FM (Zhu et al. 2026) likewise resolve heterogeneity before or beneath the model rather than in it: the former mixes bandwidths in training but relies on subcarrier-selection preprocessing and evaluates each task within a single benchmark, while the latter compresses the frequency axis with cross-attention yet does so on top of a fixed padded grid with subcarrier and antenna dimensions collapsed into one axis.

CSI-JEPA (Luo, Li, and Liu 2026) is the first application of joint-embedding predictive pretraining to Wi-Fi CSI, and contributes a channel variation-aware masking strategy that targets high-variation regions rather than masking uniformly. Like AM-FM, however, it tokenizes on a fixed $\bar { P _ { K } } \times \bar { P _ { T } }$ patch grid with index-based positional embeddings, so heterogeneity in subcarrier count and bandwidth must be resolved before tokenization; its evaluation likewise stays within a single benchmark.

Closest to our own motivation, WiFi-JEPA (Kim et al. 2026) argues against flattening CSI into a 2D grid, since patch embedding then crosses link and amplitude-phase boundaries, and instead keeps the $( C , T , L )$ structure intact. Its central contribution is link masking, which predicts one antennalink view from the others to exploit cross-link redundancy. WiFi-JEPA, however, is specialized to a single fixed acquisition configuration and tokenizes with index-based positional embeddings, so unlike UniCSI it neither ingests varying subcarrier counts and bandwidths nor encodes subcarriers by physical band position.

In contrast, UniCSI treats variation in subcarrier count, bandwidth, and carrier band not as something to be normalized away before the model, as the preceding methods do, but as something the architecture ingests natively, encoding each subcarrier by its fractional position within the observed band so that a single model spans difering shapes rather than one fixed configuration.Since these methods share fixed-grid tokenization as their ingestion substrate, our experiments compare against controlled instantiations of that substrate under matched data and training conditions.

## Method

Preliminaries of Wi-Fi CSI Wi-Fi operates in the 2.4 and 5 GHz bands using orthogonal frequency-division multiplexing (OFDM), which divides the channel into a set of narrow, evenly spaced frequency components called subcarriers. Because each subcarrier probes the channel at a slightly diferent frequency, a CSI measurement over an observation window is a complex-valued tensor

$$
\mathbf { H } \in \mathbb { C } ^ { T \times N _ { \mathrm { t x } } \times N _ { \mathrm { r x } } \times K } ,\tag{1}
$$

whose element $\mathbf { H } _ { t , i , j , k } = a e ^ { \mathrm { j } \phi }$ records the amplitude a and phase ϕ ofthe channel between transmit antenna i and receive antenna j at time step t and subcarrier k. Here T is the number of packets, $N _ { \mathrm { t x } }$ and $N _ { \mathrm { r x } }$ the transmit and receive antenna counts, and K the number of subcarriers. Crucially, both the MIMO configuration $( N _ { \mathrm { t x } } , N _ { \mathrm { r x } } )$ and the subcarrier count K depend on the underlying hardware and bandwidth, so CSI arrives with heterogeneous dimensions across devices, and an efective foundation model must accommodate them without fixed input shapes or device-specific architectures.

Throughout, we operate on the amplitude a and discard the phase $\phi .$ This choice follows from both the data landscape and our design goals. Most public CSI datasets release amplitude only, making it the common denominator across corpora and the feature most used in prior work (Jiang et al. 2025; Zhu et al. 2026). Raw phase, moreover, is corrupted by carrier-frequency, sampling-clock, and packet-detection ofsets, and its sanitization is chipset-specific, precisely the configuration-dependent preprocessing that UniCSI is designed to avoid.

Overview Given a CSI measurement, our goal is to train a unified encoder that produces a fixed-dimensional representation regardless of the underlying Wi-Fi hardware and communication parameters. In particular, the encoder should naturally accommodate heterogeneous CSI measurements with varying numbers of subcarriers and antenna configurations, without requiring zero-padding, resampling, or devicespecific backbones.

We describe UniCSI, a heterogeneous CSI encoder composed of three key components as shown in Figure 1. First, each transmit–receive antenna link is processed independently by a physics-informed RF tokenizer to capture local temporal dynamics for every subcarrier. Next, a spectral aggregator explicitly models dependencies across subcarriers while remaining agnostic to the number of input subcarriers. Finally, the aggregated tokens are fed into a Transformer encoder to produce a fixed-dimensional representation for downstream Wi-Fi sensing tasks. This architecture can be trained under either supervised multi-task learning or selfsupervised JEPA pretraining without modification.

![](images/d3d47fc41034d16f32eb64c23a4639cb37eacf67febde92dfa3a51ef66f37f8e.jpg)

![](images/4f3b63d50818f65a15dc30372b24648f4ba70b357671888825a90f5455335152.jpg)  
Figure 1: Overview of UniCSI. (Top) Per-link encoder: a temporal CNN and time-patch projection tokenize per-subcarrier CSI; learned queries with band-relative positional encoding cross-attend into the subcarrier tokens, compressing the variable-length spectrum into a fixed $Q \times P \times$ d latent for a Transformer encoder. (Bottom) Training regimes: JEPA pretraining, supervised multi-task training, and downstream probing or fine-tuning.

Physics-Informed RF Tokenizer UniCSI processes each transmit–receive antenna link independently and tokenizes each subcarrier separately, rather than partitioning the entire $K \times T \mathrm { C S I }$ matrix into two-dimensional patches. This design is motivated by the asymmetric semantics of CSI: the temporal axis captures channel evolution over time, whereas the subcarrier axis represents frequency-domain responses. Moreover, subcarrier-wise tokenization naturally accommodates heterogeneous CSI with varying numbers of subcarriers, avoiding fixed patch layouts tied to specific hardware configurations. Specifically, a lightweight 1D CNN is first applied to each subcarrier independently to capture local temporal patterns. The resulting features are then divided into $P = \lfloor T / \tau \rfloor$ non-overlapping windows of length τ, each of which is linearly projected into a d-dimensional RF token, yielding $\mathbf { Z } \in \mathbb { R } ^ { \check { K } \times P \check { \times } d }$ . Since tokenization is performed independently for each subcarrier, K only determines the sequence length, allowing UniCSI to process CSI measurements with arbitrary numbers of subcarriers using the same model parameters.

Spectral Aggregator After RF tokenization, each antenna link is represented as a sequence of subcarrier tokens, $\mathbf { Z } \in \mathbb { R } ^ { K \times P \times d }$ , where K varies across Wi-Fi devices and communication parameters. To decouple the downstream encoder from this variable spectral length, we introduce a query-based spectral aggregator that compresses each spectrum into a fixed-size representation. Specifically, a set of Q learnable query tokens $\{ q _ { j } \} _ { j = \mathrm { \ i } } ^ { Q }$ cross-attend to the K subcarrier tokens at each temporal window, yielding a compact latent representation of size $Q \times P$

To give queries and subcarrier tokens a shared frequency coordinate, we incorporate a physics-informed RF positional encoding. For a measurement with bandwidth B and subcarriers indexed $k \in \{ 0 , \ldots , K - 1 \}$ , the k-th subcarrier is assigned the centered frequency ofset

$$
f _ { k } = \left( k - \frac { K - 1 } { 2 } \right) \frac { B } { K } ,\tag{2}
$$

normalized by the half bandwidth as

$$
\tilde { f } _ { k } = \frac { f _ { k } } { B / 2 } \in [ - 1 , 1 ] ,\tag{3}
$$

and projected into the embedding space via a learned afine map,

$$
\operatorname { P E } ( \tilde { f } _ { k } ) = W \tilde { f } _ { k } + b , \qquad W \in \mathbb { R } ^ { d \times 1 } , b \in \mathbb { R } ^ { d } .\tag{4}
$$

Unlike discrete positional embeddings, whose indices are not comparable across bandwidths or channel configurations, this formulation requires neither a lookup table nor a predefined maximum subcarrier count: subcarriers occupying the same relative location within diferent bandwidths receive consistent positional codes, independent of the subcarrier spacing.

The RF positional encoding is added to both the key/value subcarrier tokens and the query tokens. Each query is associated with a fixed fractional anchor $\tilde { a } _ { j } \in [ - 1 , 1 ]$ and augmented with $\mathrm { P E } ( \tilde { a } _ { j } )$ , enabling queries and subcarrier tokens to interact in a shared normalized frequency coordinate system. Cross-attention is interleaved with Transformer blocks for R refinement steps: at each step, the queries attend to the spectrum, are processed by a Transformer group, and then re-attend to the subcarrier tokens to progressively refine the compressed representation.

The resulting $Q \times P$ latent is further augmented with a learned temporal-window encoding and a band embedding that indicates the operating frequency band (2.4 or 5 GHz). The tokens are then flattened, optionally prepended with a class token for supervised learning, and processed by a prenorm Transformer encoder of depth L. To support arbitrary MIMO configurations, each transmit–receive link is encoded independently, and the resulting per-link predictions are aggregated at inference time.

Training Objectives The UniCSI encoder is agnostic to the learning paradigm: the same architecture is optimized under either supervised multi-task learning or self-supervised pretraining, with only the objective changing.

Supervised multi-task learning. Given M Wi-Fi sensing datasets with heterogeneous tasks, we jointly train the shared encoder $f _ { \theta }$ with a lightweight prediction head $g _ { m }$ per dataset, attached to the representation $\mathbf { z } ~ = ~ f _ { \theta } ( \mathbf { H } )$ The objective is a weighted sum of the per-dataset losses, $\begin{array} { r } { \mathcal { L } _ { \mathrm { s u p } } = \sum _ { m = 1 } ^ { M } \lambda _ { m } \mathcal { L } _ { m } ( g _ { m } ( \mathbf { z } ) , \mathbf { y } _ { m } ) } \end{array}$ , where each ${ \mathcal { L } } _ { m }$ is a task-specific loss (cross-entropy for classification) and $\lambda _ { m }$ balances the datasets.

Self-supervised pretraining. To obtain a general-purpose representation, we pretrain with a Joint-Embedding Predictive Architecture (JEPA) (Assran et al. 2023): contiguous temporal-spectral blocks are held out as prediction targets while the remaining tokens form the context, and a lightweight predictor maps the context representation to the target representation. The encoder is trained under a smooth-$L _ { 1 }$ loss on parameter-free layer-normalized targets, with the target representation produced by an EMA copy of the encoder; details of the masking and normalization are given in the JEPA pretraining experiment.

## Experiments

## Datasets

We gathered as many public CSI datasets as we could in order to test the generalization capability of our approach. The resulting collection spans the breadth of human-sensing tasks, including human activity recognition (HAR), pose estimation, gesture recognition, localization, authentication, and fall detection, so that any model evaluated on it must contend not only with configuration heterogeneity but with a diversity of downstream objectives. We applied two inclusion criteria in dataset selection. First, we required a minimum level of annotation, namely that each dataset reports its carrier frequency, bandwidth, MIMO dimension, and sampling rate. Second, we required the CSI to be free of preprocessing artifacts. The datasets that remain (listed in Table 1) still difer substantially in their MIMO configurations, ranging from 1 1 to 6 3, with some setups combining several receiver devices for simultaneous recording. Four of the pretraining datasets ship with a designated cross-domain test domain, which we exclude from the corpus and reserve for evaluation (marked ‡: CSIDA scene 1, HAR-ORT subject 10, MM-Fi scene 3, OPERAnet room 2); CSI-Bench contributes all subsets except HAR (§) and aggregates four tasks with $n { = } 2 / 5 / 7 / 4 \ ( ^ { \dag } )$ . The full collection totals 530.2 hours of recordings, of which the assembled pretraining corpus comprises 425.6 hours over 391,277 measurements after holdouts. To harmonize acquisition settings, we resampled every recording to a common 100 Hz rate. We then segmented each recording into non-overlapping windows of at most 5 seconds, retaining shorter recordings as single samples. Finally, as chipsets can encode CSI with a diferent bit depth, we normalize each segment to zero mean and unit variance, rendering amplitudes comparable across devices before they reach the encoder.

![](images/ca89bba71eef4780e3336ebe9ea69cb76901535f341dc0a01ea7dbcef537508c.jpg)  
Figure 2: Fused F1 of a single encoder trained jointly across eight heterogeneous CSI datasets. Bars denote the mean and whiskers the standard deviation over three seeds. All datasets except NTU-Fi HAR (in-distribution control) use a crossenvironment or cross-subject protocol.

## Implementation Details

All models were trained on a single accelerator with roughly 140 GB of memory, hosted on an x86 Linux machine with 256 GB of system RAM. Pretraining took 14.75 hours on a single GPU, using approximately 41 GiB of memory. The detailed hyperparameter settings are summarized in Table 2. We report either classification accuracy or macro-F1 depending on the benchmark. Macro-F1 is reported for imbalanced datasets to better reflect per-class performance.

## Multi-task Supervised Training

To assess whether a single encoder can benefit from jointly observing heterogeneous CSI configurations, we trained one encoder on eight CSI datasets for HAR spanning four Wi-Fi bandwidths.

We compare UniCSI against baselines chosen to instantiate the shared design axis of prior CSI foundation models, namely a fixed input grid with index-based positional embeddings (Jiang et al. 2025; Zhu et al. 2026; Luo, Li, and Liu 2026; Kim et al. 2026). Rather than reproduce each method verbatim, which would entangle the architectural question with diferences in corpora, preprocessing, and training recipes, we abstract their common ingestion strategies and train each under an identical recipe, corpus, and parameter budget. The Padded ViT zero-pads CSI to the largest subcarrier dimension before a standard ViT, instantiating the fixed-grid tokenization underlying AM-FM and CSI-JEPA, whereas the Resampled ViT interpolates all inputs to a fixed resolution, instantiating the harmonize-then-train strategy of Jiang et al. (2025). A shared ResNet serves as a shape-agnostic convolutional control. This design attributes any performance diference to the ingestion mechanism itself rather than to confounded training choices.

<table><tr><td>Dataset</td><td>Task (n)</td><td>Rx</td><td>Tx</td><td>Sub.</td><td>Band (GHz)</td><td>BW (MHz)</td></tr><tr><td>Wi-Fi Sensing Contest (Han 2024)</td><td>Presence (n=3)</td><td>2</td><td>2</td><td>248,250</td><td>5</td><td>80,160</td></tr><tr><td>Person-in-Wi-Fi 3D (Yan et al. 2024)</td><td>3D Pose (n=4)</td><td> $3 { \times } 3$ </td><td>1</td><td>30</td><td>5</td><td>20</td></tr><tr><td>XRFV2 (Lan et al. 2025)</td><td>HAR (n=30)</td><td>3×3</td><td>1</td><td>30</td><td>5</td><td>20</td></tr><tr><td>Wi-MIR (Islam et al. 2024)</td><td>HAR (n=17)</td><td>3</td><td>3</td><td>30 5</td><td></td><td>20</td></tr><tr><td>Behavior Auth. (Shi et al. 2017)</td><td>User ID (n=12)</td><td>3</td><td>1</td><td>30 5</td><td></td><td>20</td></tr><tr><td>NTU-Fi Human ID (Yang et al. 2022)</td><td>User ID (n=14)</td><td>3</td><td>1</td><td>114 5</td><td></td><td>40</td></tr><tr><td>MM-Fi‡ (Yang et al. 2023c)</td><td>HAR (n=27)</td><td>3</td><td>1</td><td>114 5</td><td></td><td>40</td></tr><tr><td>WiAR (Guo et al. 2019)</td><td>HAR (n=16)</td><td>3</td><td>1</td><td>30 5</td><td></td><td>20</td></tr><tr><td>ARIL (Wang et al. 2019)</td><td>HAR / Loc. (n=6)</td><td>1</td><td>1</td><td></td><td>52 2.4</td><td>20</td></tr><tr><td>CSIDA‡ (Zhang et al. 2022)</td><td>Gesture (n=6)</td><td>3</td><td>1</td><td>114 5</td><td></td><td>40</td></tr><tr><td>OPERAnet‡ (Bocus et al. 2021)</td><td>HAR (n=7)</td><td>3</td><td>3</td><td>305</td><td></td><td>40</td></tr><tr><td>HAR-ORT‡ (Bellizzi and Gili Kouymtchian 2024)</td><td>HAR (n=5)</td><td>1</td><td>1</td><td>242 5</td><td></td><td>80</td></tr><tr><td>EHUNAM (De Armas et al. 2025)</td><td>HAR / Env. ID (n=6)</td><td>1</td><td>1</td><td>56-241</td><td>2.4,5</td><td>20,80</td></tr><tr><td>CSI-BFI-HÀR (Haque 2025)</td><td>HAR (n=21)</td><td>3</td><td>1</td><td>242</td><td>5</td><td>80</td></tr><tr><td>OctoNet (Yuan et al. 2026)</td><td>HAR (n=64)</td><td> $4 \times 2$ </td><td>1</td><td>114</td><td>5</td><td>40</td></tr><tr><td>UT-HAR (Yang et al. 2023b)</td><td>HAR (n=7)</td><td>3</td><td>1</td><td></td><td>305</td><td>20</td></tr><tr><td>CSI-Bench†§(Zhu et al. 2025)</td><td>Fall/HAR/ID/Prox.</td><td>2-4</td><td>1-2</td><td>14-58</td><td>2.4,5</td><td>20-40</td></tr><tr><td>Exposing the CSI (Cominelli, Gringoli, and Restuccia 2023)</td><td>HAR (n=12)</td><td>3×4</td><td>1</td><td>2048</td><td>5</td><td>160</td></tr><tr><td>XRF55 (Wang et al. 2024)</td><td>HAR (n=55)</td><td> $3 { \times } 3$ </td><td>1</td><td>30 5</td><td></td><td>20</td></tr><tr><td>Widar 3.0 (Zheng et al. 2019)</td><td>Gesture (n=22)</td><td> $6 \times 3$ </td><td>1</td><td>30 5</td><td></td><td>20</td></tr><tr><td>WiMANS (Huang et al. 2025)</td><td>HAR (n=9)</td><td>3</td><td>3</td><td></td><td>302.4,5</td><td>20</td></tr><tr><td>FallDar (Yang, Zhang, and Zhang 2023)</td><td>Fall (n=2)</td><td>3</td><td>1</td><td></td><td>30 5</td><td>20</td></tr><tr><td>GaitID (Zhang et al. 2020)</td><td>User ID (n=11)</td><td>6×3</td><td>1</td><td></td><td>30 5</td><td>20</td></tr><tr><td>NTU-Fi HAR (Yang et al. 2022)</td><td>HAR (n=6)</td><td>3</td><td>1</td><td>114 5</td><td></td><td>40</td></tr><tr><td>SHARP (Meneghello et al. 2023)</td><td>HAR (n=8)</td><td>4</td><td>1</td><td>242 5</td><td></td><td>80</td></tr></table>

Table 1: CSI datasets for pretraining and evaluation. Shaded rows form the self-supervised pretraining corpus; unshaded rows are held out for downstream evaluation only. n is the number of class labels, Rx/Tx the antenna counts, Sub. the subcarrier count, Band the carrier band and BW the channel bandwidth; D A denotes D receiver devices of A antennas each. Footnote marks are explained in the text.

Table 2: Training hyperparameters. All runs use AdamW, cosine decay to 0, gradient clip 1.0, and bf16.
<table><tr><td></td><td>Multi-task</td><td>JEPA</td><td>Probe / k-NN</td><td>Fine-tune</td></tr><tr><td>Encoder</td><td>scratch</td><td>EMA</td><td>frozen</td><td>fine-tuned</td></tr><tr><td>Head</td><td>linear</td><td>predictor</td><td>BN + lin.</td><td>BN + lin.</td></tr><tr><td>Objective</td><td>CE</td><td>smooth-L1</td><td>CE / cosine</td><td>CE</td></tr><tr><td>LR</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>WD</td><td> $5 \times 1 0 ^ { - 2 }$ </td><td> $5 \times 1 0 ^ { - 2 }$ </td><td> $1 \times 1 0 ^ { - 5 }$ </td><td> $5 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Budget</td><td>50 epochs</td><td> $5 0 \mathrm { k \ s t e p s }$ </td><td>200 epochs</td><td>50 epochs</td></tr><tr><td>Batch size</td><td>128</td><td>1024</td><td>64</td><td>64</td></tr></table>

Our model achieves the highest macro-F1 across the eight test sets (71.6%), outperforming the strongest baseline (padded ViT, 62.4%) by 9.2 points. This advantage is concentrated in the hardest cross-domain regimes. On Exposing-CSI, a 12-way task under an unseen environment, the padded and resampled ViTs reach only 6.7% and 7.3% and the ResNet collapses to 1.7%, leaving every baseline at or below the chance floor, whereas our model attains 32.5%, marking the diference between transferring to the unseen environment and failing to generalize at all. XRF55 shows the same pattern, with our model at 33.6% against 22.9% for the strongest baseline. On the well-conditioned datasets our model does not sacrifice performance, saturating NTU-Fi HAR (100.0%), GaitID (99.2%), and FallDar (99.4%). The only datasets on which it does not lead are SHARP (62.6% against the padded ViT’s 65.2%), where the two are within overlapping seed spread, and FallDar, where the padded ViT edges ahead by 0.2 points with both models saturated. The ResNet baseline, which shares a single backbone across shapes without frequency information, is the weakest throughout and the least stable across seeds (average 29.7%), underscoring that a shape-agnostic convolutional backbone alone does not solve heterogeneous ingestion.

## JEPA Pretraining

We pretrain with a joint-embedding predictive architecture (JEPA) (Assran et al. 2023) on the unlabeled corpus, drawing measurements with a bandwidth-stratified sampler that enforces a 50/25/25 mix of 20, 40, and $\ge 8 0 \mathrm { M H z }$ links, with the 160 MHz data pooled into the widest stratum. For each batch we mask a single contiguous block covering 80% of the encoder’s latent token grid (Q = 16 frequencies by P = 20 time patches), leaving a thin border of visible tokens as context. The mask is enforced in input space, by zeroing the corresponding subcarrier-by-time region of the raw CSI within each clip’s non-padded extent.

The prediction target is produced by an EMA target encoder whose momentum is annealed linearly from 0.996 to 0.9995, encoding the unmasked input. A narrow predictor (4 layers, width 192) then maps the visible context tokens, together with a learned mask token at each masked grid position, to the target representations at those positions. We regress onto parameter-free layer-normalized targets under a smooth-L<sub>1</sub> loss, since normalizing the targets removes the scale-collapse shortcut that the encoder’s afine output norm would otherwise leave open, by which the predictor could trivially minimize the loss by driving target magnitudes toward zero rather than learning to predict their structure.

Transferability of Pretrained Representations To assess the quality and transferability of the learned representations, we evaluate the pretrained encoder under two frozen-encoder protocols, k-NN classification and linear probing, in which the encoder is held fixed and only a lightweight predictor is fit (Figure 3). For k-NN, we select the best k based on validation set performance and then apply the best k to the test set. Both protocols measure the discriminative power of the representation itself, without updating the backbone. Across most downstream datasets and under both protocols, UniCSI outperforms the pretrained Padded ViT, the strongest supervised baseline. This baseline is itself an informative comparator, since a JEPA-pretrained Padded ViT is CSI-JEPA-like in all but one respect, sharing its fixed patch grid, index-based positional embeddings, and joint-embedding predictive objective, and trained on the same corpus as UniCSI, so that it difers from Luo, Li, and Liu (2026) only in using uniform rather than variation-aware masking. The gains are most pronounced on the harder cross-environment benchmarks, namely WiMANS, SHARP, and Exposing CSI, while performance remains comparable on easier datasets such as Fall-Dar. That the improvement is larger under k-NN is telling, as it suggests UniCSI learns representations that are more transferable and more semantically structured even without any downstream adaptation.

We further evaluate parameter adaptation by allowing the pretrained encoder to be optimized using full fine-tuning. The results are reported in Table 3. We compare three settings: supervised training from scratch using UniCSI, full finetuning of a pretrained Padded ViT, and full fine-tuning of a pretrained UniCSI. As shown in Table 3, pretrained UniCSI achieves the highest average performance (71.58%), outperforming both training UniCSI from scratch (67.97%) and the pretrained Padded ViT (60.68%). The improvement over training from scratch demonstrates that large-scale pretraining provides a stronger initialization for downstream adaptation, while the substantial margin over Padded ViT indicates that preserving the native spectral structure is more efective than relying on zero-padding to accommodate heterogeneous CSI.

Table 3: Comparison between supervised training from scratch with UniCSI, full fine-tuning (FT) from the JEPApretrained Padded ViT, and full FT from the JEPA-pretrained UniCSI on unseen downstream datasets (Metric: Macro F1 score).
<table><tr><td>Dataset</td><td>Split</td><td>Scratch (Ours)</td><td>Full FT (PaddedViT)</td><td>Full FT (Ours)</td></tr><tr><td>WiMANS</td><td>Cross-Env</td><td>55.69%</td><td>36.96%</td><td>55.83%</td></tr><tr><td>(5 GHz)</td><td></td><td></td><td></td><td></td></tr><tr><td>XRF55</td><td>Cross-Env</td><td>39.53%</td><td>15.5%</td><td>39.78%</td></tr><tr><td>SHARP</td><td>Cross-Env</td><td>58.50%</td><td>75.71%</td><td>71.06%</td></tr><tr><td>Exposing-CSI</td><td>Cross-Env</td><td>47.56% 96.78%</td><td>7.6%</td><td>45.16%</td></tr><tr><td>FallDar CSI-Bench</td><td>Random Random</td><td>80.19%</td><td>98.61% 91.16%</td><td>98.95% 90.67%</td></tr><tr><td>(2.4 GHz)</td><td></td><td></td><td></td><td></td></tr><tr><td>NTU-Fi HAR</td><td>Random</td><td>97.52%</td><td>99.23%</td><td>99.61%</td></tr><tr><td>Average</td><td></td><td>67.97%</td><td>60.68%</td><td>71.58%</td></tr></table>

Compared with the baseline pretrained model, UniCSI achieves stronger downstream performance on the majority of benchmarks under both frozen and trainable adaptation protocols, with the largest margins on cross-environment splits.

Sample Eficiency We next evaluate the sample eficiency of UniCSI under limited supervision. Starting from the pretrained checkpoint or random initialization, we fine-tune all model parameters using only 5, 10, or 50 labeled samples per class on eight downstream datasets spanning in-domain, random, and cross-domain evaluation protocols.

Figure 4 summarizes the average performance across all datasets. Pretraining consistently improves downstream performance at every label budget, with the largest gains in the most label-scarce setting. With only 5 labeled samples per class, pretrained UniCSI increases the average fused accuracy from 29.6% to 47.2% and the macro-F1 from 17.3% to 43.7%. The advantage remains substantial at 10-shot and 50-shot, demonstrating that the learned representation can be adapted efectively with very limited supervision. These results indicate that UniCSI is highly sample-eficient, substantially reducing the amount of labeled data required for downstream deployment.

## Ablation

We isolate the contribution of the two main components by removing them one at a time from the full model, under the same training scheme as multi-task supervised training. Both variants share the same training mixture, schedule, optimizer, and parameter budget, and each is run with three seeds; we report the average accuracy across the eight datasets of the multi-task scheme, since several components produce differences comparable to seed noise. Both components contribute, but to very diferent degrees. Removing the RF tokenizer produces the largest drop, from 73.3% to 63.5%: replacing per-subcarrier tokens with fixed two-dimensional patches forces samples onto a common grid and discards the per-subcarrier temporal processing, which suggests that the tokenizer is the component the rest of the architecture is built on. Removing spectral aggregation costs only 1.6 points (73.3% to 71.7%), which may appear to make the component optional; however, the average conceals its role, as the aggregation is what recovers the accuracy that per-subcarrier tokenization sacrifices on low-subcarrier devices, and it is also what fixes the latent size for any subcarrier count, the property the JEPA pretraining depends on.

![](images/7f67726d3bdf19309cda3c9d185ed38f0874b43506b86873c6c0fdc060ceb2f3.jpg)

![](images/62f818df3e4f8cd23b3a321859f8724ceaac0cf847dfdbf51b541a0aa642921b.jpg)  
Figure 3: Linear probe and k-NN evaluation of Padded ViT and UniCSI on eight representative dataset splits. Bars report fused macro-F1; Avg is the mean over the selected splits.

![](images/0ea0142f79e7231d3ca9bfc4db31ce66bcc3467902d1dc24a5066a22d2c72281.jpg)

![](images/9b305320951971672230f3ca51ce08e17f62140b3b2455a5258a92a33292c795.jpg)  
Figure 4: Few-shot downstream fine-tuning from pretrained versus random initialization, averaged over eight dataset– split pairs.

## Discussion

Across both training regimes, our results support a single claim: handling CSI heterogeneity in the architecture, rather than normalizing it away beforehand, is what enables transfer across devices and environments. Under supervised joint training UniCSI is the only model that transfers to unseen environments where the fixed-grid baselines collapse toward chance, and under self-supervised pretraining the same encoder yields representations that transfer under frozen probing and initialize few-shot fine-tuning far better than training from scratch. The ablation locates these gains in the two components that make the architecture shape-agnostic, the per-subcarrier tokenizer and the spectral aggregator, rather than in scale or data alone, suggesting that native heterogeneous ingestion is not merely a convenience but a property that materially improves cross-domain generalization.

We kept the positional encoding deliberately simple, and see richer frequency representations as the most promising follow-up. Encoding absolute carrier frequency rather than band-relative position is a natural next step.

We compare against abstractions of prior ingestion strategies rather than published checkpoints, since existing models are bound to their preprocessing pipelines and cannot ingest our corpus without the normalization our study removes. Head-to-head evaluation on shared benchmarks under each method’s native preprocessing nonetheless remains valuable future work.

## Conclusion

We presented UniCSI, a unified foundation architecture for heterogeneous Wi-Fi sensing that natively accommodates diverse CSI configurations without lossy preprocessing. By combining a physics-informed RF tokenizer with a spectral aggregator, UniCSI enables a single model to process CSI collected under varying subcarrier counts, bandwidths, and carrier bands. Extensive experiments under both supervised and self-supervised settings show that handling heterogeneity in the architecture, rather than in preprocessing, substantially improves cross-device and cross-environment generalization, with the largest gains precisely where fixed-grid approaches collapse.

## References

Abdelnasser, H.; Harras, K. A.; and Youssef, M. 2015. UbiBreathe: A ubiquitous non-invasive WiFi-based breathing estimator. In Proceedings of the 16th ACM international symposium on mobile ad hoc networking and computing, 277–286.

Adib, F.; Kabelac, Z.; Katabi, D.; and Miller, R. C. 2014. 3D Tracking via Body Radio Reflections. In Proceedings of the 11th USENIX Conference on Networked Systems Design and

Implementation, NSDI’14, 317–329. USA: USENIX Association. ISBN 978-1-931971-09-6.

Assran, M.; Duval, Q.; Misra, I.; Bojanowski, P.; Vincent, P.; Rabbat, M.; LeCun, Y.; and Ballas, N. 2023. Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. arXiv:2301.08243.

Bellizzi, B.; and Gili Kouymtchian, M. 2024. HAR-ORT: 5GHz 80MHz Nexmon-Extracted CSI. Kaggle.

Bocus, M. J.; Li, W.; Vishwakarma, S.; Kou, R.; Tang, C.; Woodbridge, K.; Craddock, I.; McConville, R.; Santos-Rodriguez, R.; Chetty, K.; and Piechocki, R. 2021. OP-ERAnet: A Multimodal Activity Recognition Dataset Acquired from Radio Frequency and Vision-based Sensors. arXiv:2110.04239.

Cominelli, M.; Gringoli, F.; and Restuccia, F. 2023. Exposing the CSI: A Systematic Investigation of CSI-based Wi-Fi Sensing Capabilities and Limitations. In 2023 IEEE International Conference on Pervasive Computing and Communications (PerCom), 81–90. Atlanta, GA, USA: IEEE. ISBN 978-1-6654-5378-3.

De Armas, E.; Diaz, G.; Sobron, I.; Eizmendi, I.; Landa, I.; Matias, J. M.; and Velez, M. 2025. EHUNAM, a WiFi CSIbased Dataset for Human and Machine Sensing. Scientific Data, 12(1): 1950.

Guo, L.; Guo, S.; Wang, L.; Lin, C.; Liu, J.; Lu, B.; Fang, J.; Liu, Z.; Shan, Z.; and Yang, J. 2019. Wiar: A Public Dataset for Wifi-Based Activity Recognition. IEEE Access, 7: 154935–154945.

Han, T. X. 2024. Wi-Fi Sensing Contest. Sensing Dataset Platform (SDP8).

Haque, K. F. 2025. CSI-BFI-HAR: Wi-fi Datasets for Human Activity Recognition.

Huang, S.; Li, K.; You, D.; Chen, Y.; Lin, A.; Liu, S.; Li, X.; and McCann, J. A. 2025. WiMANS: A Benchmark Dataset for WiFi-Based Multi-user Activity Sensing. In Leonardis, A.; Ricci, E.; Roth, S.; Russakovsky, O.; Sattler, T.; and Varol, G., eds., Computer Vision – ECCV 2024, volume 15100, 72– 91. Cham: Springer Nature Switzerland. ISBN 978-3-031- 72945-4 978-3-031-72946-1.

Islam, M. S.; Kabir, M. H.; Hasan, M. A.; and Shin, W. 2024. Wi-MIR: A CSI Dataset for Wi-Fi Based Multi-Person Interaction Recognition. IEEE access : practical innovations, open solutions, 12: 67256–67272.

Jiang, C.; Yan, Y.; Wang, Y.; Chou, C. T.; and Hu, W. 2025. Scale What Counts, Mask What Matters: Evaluating Foundation Models for Zero-Shot Cross-Domain Wi-Fi Sensing.

Kim, D.; Lee, J.; Kim, S.; and heum Kim, S. 2026. WiFi-JEPA: Self-supervised Learning for WiFi-CSI 3D Human Pose Estimation. arXiv:2607.11064.

Lan, B.; Li, P.; Yin, J.; Song, Y.; Wang, G.; Ding, H.; Han, J.; and Wang, F. 2025. XRF V2: A Dataset for Action Summarization with Wi-Fi Signals, and IMUs in Phones, Watches, Earbuds, and Glasses.

Luo, X.; Li, Z.; and Liu, Y. 2026. CSI-JEPA: Towards Foundation Representations for Ubiquitous Sensing with Minimal Supervision. arXiv:2605.14171.

Lyons, N.; Pandey, A.; and Santra, A. 2025. WiSenseNet: A Unified Foundation Model for Diverse Wi-Fi Sensing Tasks Using Channel State Information. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 1–5. Hyderabad, India: IEEE. ISBN 979-8-3503-6874-1.

Meneghello, F.; Garlisi, D.; Fabbro, N. D.; Tinnirello, I.; and Rossi, M. 2023. SHARP: Environment and Person Independent Activity Recognition With Commodity IEEE 802.11 Access Points. IEEE Transactions on Mobile Computing, 22(10): 6160–6175.

Shi, C.; Liu, J.; Liu, H.; and Chen, Y. 2017. Smart User Authentication through Actuation of Daily Activities Leveraging WiFi-enabled IoT. In Proceedings of the 18th ACM International Symposium on Mobile Ad Hoc Networking and Computing, 1–10. Chennai India: ACM. ISBN 978-1-4503- 4912-3.

Strohmayer, J.; Wödlinger, M.; and Kampel, M. 2024. Wi-FlexFormer: Eficient WiFi-Based Person-Centric Sensing. arXiv:2411.04224.

Wang, F.; Feng, J.; Zhao, Y.; Zhang, X.; Zhang, S.; and Han, J. 2019. Joint Activity Recognition and Indoor Localization with WiFi Fingerprints. IEEE Access, 7: 80058–80068.

Wang, F.; Lv, Y.; Zhu, M.; Ding, H.; and Han, J. 2024. XRF55: A Radio Frequency Dataset for Human Indoor Action Analysis. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol., 8(1).

Wang, Y.; Liu, J.; Chen, Y.; Gruteser, M.; Yang, J.; and Liu, H. 2014. E-Eyes: Device-Free Location-Oriented Activity Identification Using Fine-Grained WiFi Signatures. In Proceedings of the 20th Annual International Conference on Mobile Computing and Networking, 617–628. Maui Hawaii USA: ACM. ISBN 978-1-4503-2783-1.

Yan, K.; Wang, F.; Qian, B.; Ding, H.; Han, J.; and Wei, X. 2024. Person-in-WiFi 3D: End-to-End Multi-Person 3D Pose Estimation with Wi-Fi. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 969– 978. Seattle, WA, USA: IEEE. ISBN 979-8-3503-5300-6.

Yang, J.; Chen, X.; Wang, D.; Zou, H.; Lu, C. X.; Sun, S.; and Xie, L. 2023a. SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing. arXiv:2207.07859.

Yang, J.; Chen, X.; Zou, H.; Lu, C. X.; Wang, D.; Sun, S.; and Xie, L. 2023b. SenseFi: A Library and Benchmark on Deep-Learning-Empowered WiFi Human Sensing. Patterns, 4(3): 100703.

Yang, J.; Chen, X.; Zou, H.; Wang, D.; Xu, Q.; and Xie, L. 2022. EficientFi: Toward Large-Scale Lightweight WiFi Sensing via CSI Compression. IEEE Internet of Things Journal, 9(15): 13086–13095.

Yang, J.; Huang, H.; Zhou, Y.; Chen, X.; Xu, Y.; Yuan, S.; Zou, H.; Lu, C. X.; and Xie, L. 2023c. MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems, Nips ’23. Red Hook, NY, USA: Curran Associates Inc.

Yang, M.; Zhu, H.; Zhu, R.; Wu, F.; Yin, L.; and Yang, Y. 2023d. WiTransformer: A Novel Robust Gesture Recognition Sensing Model with WiFi. Sensors, 23(5).

Yang, Z.; Zhang, Y.; and Zhang, Q. 2023. Rethinking Fall Detection with Wi-Fi. IEEE Transactions on Mobile Computing, 22(10): 6126–6143.

Yuan, D.; Zhang, X.; Hou, W.; Lyu, S.; Yu, Y.; Yu, L. J.-T.; Li, C.; and Wu, C. 2026. OctoNet: A large-scale multimodal dataset for human activity understanding grounded in motion-captured 3D pose labels. Advances in Neural Information Processing Systems, 38.

Zhang, X.; Tang, C.; Yin, K.; and Ni, Q. 2022. WiFi-Based Cross-Domain Gesture Recognition via Modified Prototypical Networks. IEEE Internet of Things Journal, 9(11): 8584– 8596.

Zhang, Y.; Zheng, Y.; Qian, K.; Zhang, G.; Liu, Y.; Wu, C.; and Yang, Z. 2021. Widar3.0: Zero-Efort Cross-Domain Gesture Recognition with Wi-Fi. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1–1.

Zhang, Y.; Zheng, Y.; Zhang, G.; Qian, K.; Qian, C.; and Yang, Z. 2020. GaitID: Robust Wi-Fi Based Gait Recognition. In Yu, D.; Dressler, F.; and Yu, J., eds., Wireless Algorithms, Systems, and Applications, volume 12384, 730–742. Cham: Springer International Publishing. ISBN 978-3-030- 59015-4 978-3-030-59016-1.

Zheng, Y.; Zhang, Y.; Qian, K.; Zhang, G.; Liu, Y.; Wu, C.; and Yang, Z. 2019. Zero-Efort Cross-Domain Gesture Recognition with Wi-Fi. In Proceedings of the 17th Annual International Conference on Mobile Systems, Applications, and Services, 313–325. Seoul Republic of Korea: ACM. ISBN 978-1-4503-6661-8.

Zhou, Y.; Huang, H.; Yuan, S.; Zou, H.; Xie, L.; and Yang, J. 2023. MetaFi++: WiFi-Enabled Transformer-Based Human Pose Estimation for Metaverse Avatar Simulation. IEEE Internet Things J., 10(16): 14128–14136.

Zhu, G.; Hu, Y.; Gao, W.; Wang, W.-H.; Wang, B.; and Liu, K. J. R. 2025. CSI-Bench: A Large-Scale In-the-Wild Dataset for Multi-task WiFi Sensing. arXiv:2505.21866.

Zhu, G.; Hu, Y.; Jayaweera, S.; Gao, W.; Wang, W.-H.; Zhang, J.; Wang, B.; Wu, C.; and Liu, K. J. R. 2026. AM-FM: A Foundation Model for Ambient Intelligence Through WiFi. arXiv:2602.11200.