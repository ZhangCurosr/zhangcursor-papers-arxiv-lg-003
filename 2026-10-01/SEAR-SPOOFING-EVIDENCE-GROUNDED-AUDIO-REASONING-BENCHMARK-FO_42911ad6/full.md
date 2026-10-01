# SEAR: SPOOFING EVIDENCE-GROUNDED AUDIO REASONING BENCHMARK FOR AUDIO LANGUAGE MODELS

Rong Wan<sup>1</sup>, Suliu Qin<sup>2</sup>, Jiaxi Li<sup>1</sup>, Wei Xie<sup>3</sup>, Wenwu Wang<sup>1</sup>, Xiaolong Han<sup>1</sup>, Lu Yin<sup>4</sup>, Xilu Wang<sup>1</sup>

<sup>1</sup> University of Surrey, <sup>2</sup> Singapore University of Technology and Design, <sup>3</sup> Guangxi University, <sup>4</sup> Shenzhen University of Advanced Technology

## ABSTRACT

Audio language models (ALMs) are increasingly used for audio deepfake detection (ADD), yet existing benchmarks assess their verdicts or rationale plausibility without verifying the underlying acoustic evidence. To address this issue, we first introduce spoofing evidence-grounded audio reasoning (SEAR), a four-task AQA benchmark to evaluate ALM-based ADD through acoustic evidence identification and quantification, deepfake detection, and forensic rationale generation. We further propose a bona-fide-based acoustic evidence agent (BAEA), which equips a frozen ALM with controlled acoustic tools under FIXED or ADAPTIVE evidence-acquisition policies. Experiments with six ALMs reveal a clear gap between plausible rationales and verifiable acoustic evidence reasoning, while BAEA-FIXED improves final verdicts and forensic rationales on both evaluation partitions. Controlled interventions further show that misleading evidence degrades both detection and grounding performance.

Index Terms— Audio question answering benchmark, audio language model, audio deepfake detection

## 1. INTRODUCTION

Audio language models (ALMs) [1] have increasingly been applied to audio deepfake detection (ADD), formulating the distinction between bona-fide and spoofed audio as an audio question-answering (AQA) task. Pioneering work such as ALLM4ADD investigates binary deepfake classification with ALMs [2]. More recent methods, including HIR-SDD [3], FT-GRPO [4], CoLMbo-DF [5], and HoliAntiSpoof [6], incorporate chain-of-thought reasoning or forensic rationale generation to support an ALM’s final verdict.

Despite this progress, it remains unclear whether ALM rationales and verdicts are grounded in acoustic anomalies that are verifiably present in the signal. Conventional ADD benchmarks, such as ASVspoof [7–9] and CodecFake+ [10, 11], primarily evaluate the final detection outcomes. Recent reasoning-aware evaluations, such as TriDF [12] and HIR-SDD [3], additionally assess the divergence between models’ rationale and human-like forensic trajectory. However, these benchmarks do not evaluate whether ALMs can identify and quantify fine-grained acoustic anomalies or use such evidence consistently in forensic rationale generation and deepfake detection. We then ask: Can ALMs identify and quantify signal-level acoustic anomalies and use this evidence to supportforensic rationales and deepfake verdicts?

To bridge this gap, we first introduce spoofing evidencegrounded audio reasoning (SEAR), a four-task AQA benchmark for evaluating such ALM-based ADD. SEAR is constructed through acoustic feature extraction with statistical analysis of 35 acoustic features, controlled AQA generation, and quality control. For the empirically observed difficulty of fine-grained acoustic evidence reasoning, we draw on toolaugmented ALMs for general audio reasoning [13, 14] and present the bona-fide-based acoustic evidence agent (BAEA) as a tool-augmented baseline for SEAR. BAEA equips a frozen ALM with a controlled signal-analysis tool and a reference distribution estimated exclusively from bona-fide training samples. Its FIXED policy ranks deviations across all 35 features, whereas its ADAPTIVE policy lets the ALM itself select acoustic features before tool execution. The resulting structured information with evidence-identification and feature-quantification answers is provided to the ALM for deepfake detection and rationale generation.

Experiments with six ALMs show a clear gap between generating plausible rationales and reasoning with verifiable evidence. On SEAR, BAEA-FIXED improves evidence identification and quantification, rationale grounding, and deepfake detection over its backbone. Analyses show that misleading evidence degrades both detection and grounding, further highlighting the importance of reliable acoustic evidence acquisition and use. Our contributions are threefold:

• We introduce SEAR<sup>1</sup>, a four-task AQA benchmark beyond binary classification for ADD, constructed via a statistically grounded acoustic-evidence pipeline.

• We present BAEA<sup>2</sup>, a tool-augmented ALM that integrates controlled acoustic measurement for the tasks in SEAR without updating the backbone parameters.

![](images/dfdf71ca0697039c6ed07b602eac31b3f77a458d394b18aa9d4c057a6a08ef0e.jpg)  
Fig. 1. The proposed SEAR benchmark tasks and BAEA method.

• Experiments show that BAEA-FIXED improves over its backbone on SEAR, while controlled interventions reveal how evidence quality affects downstream detection and rationale grounding.

## 2. OUR BENCHMARK

## 2.1. Tasks Design

For an audio sample x, SEAR evaluates ALM-based ADD from three aspects, including deepfake detection D, acoustic evidence identification and quantification E, and forensic rationale generation R:

$$
\mathcal { D } ( \boldsymbol { \hat { y } } , \boldsymbol { y } ) , \qquad \mathcal { E } ( \{ \hat { e } _ { \mathrm { i d } } , \hat { e } _ { \mathrm { v a l } } \} , e ^ { * } ) , \qquad \mathcal { R } ( \hat { r } , r ^ { * } ) ,\tag{1}
$$

where $\hat { y } , \big ( \hat { e } _ { \mathrm { i d } } , \hat { e } _ { \mathrm { v a l } } \big )$ , and rˆ denote the ALM-generated detection, acoustic-evidence, and rationale outputs, respectively. Here, y is the provided ground-truth label, while $e ^ { * }$ and $r ^ { * }$ are the reference acoustic evidence and rationale constructed by the SEAR pipeline. As shown in Fig. 1, there are four tasks designed in SEAR covering three aspects:

• Deepfake verdict: binary classification (T1) asks the ALM whether audio x is genuine or spoofed, producing a binary verdict yˆ evaluated against the ground-truth label y.

• Acoustic evidence: forgery cue identification (T2) asks the ALM to identify the acoustic feature that most strongly indicates spoofing, producing $\hat { e } _ { \mathrm { i d } } .$ , while acoustic feature measurement (T3) asks it to select the measured value of a specified anomalous feature, producing $\hat { e } _ { \mathrm { v a l } }$ . Together, T2 and T3 evaluate acoustic evidence identification and quantification against the verified acoustic evidence $e ^ { * }$

• Forensic rationale generation: forensic rationale generation (T4) requires the ALM to generate an open-ended rationale rˆ that explains its verdict using verifiable signal evidence in natural language.

## 2.2. Audio Sources and Acoustic Feature Extraction

SEAR adopts two representative ADD benchmarks as its audio sources. ASVspoof 2019 LA (19LA) [7] provides a controlled setting, covering six synthesis and voice-conversion attacks in its training and development sets and 13 unseen attacks in evaluation. ASVspoof 2021 LA (21LA) [8] retains these 13 evaluation attacks while introducing seven codec and transmission conditions over VoIP and PSTN channels.

To obtain interpretable signal-level measurements, all audio samples are resampled to 16 kHz and represented by 35 utterance-level acoustic features. Frame-level analysis uses an fast Fourier transform (FFT) size of 1,024 and a hop length of 256 samples. The features comprise the time-averaged first 20 mel-frequency cepstral coefficients (MFCCs) and 15 complementary measurements derived from spectral, energy, temporal, and source-related characteristics. The latter include statistics of spectral bandwidth, roll-off, centroid, and flatness; the proportion of high-frequency spectral energy; RMS energy and its temporal variation; silence and zero-crossing ratios. Together, these features characterize spectral distribution, cepstral structure, energy dynamics, temporal variation, and pitch behaviour, forming the candidate feature set for reference evidence construction.

## 2.3. Reference Evidence and AQA Generation

To construct reference evidence for evaluating ALMs’ answers, 35 acoustic features are adopted and screened using absolute Cohen’s d and folded ROC AUC, $\widetilde { A } _ { j } = \operatorname* { m a x } ( A _ { j } , 1 -$ $A _ { j } )$ . The feature $j$ is identified as a global feature if $| d _ { j } | \geq$ 0.5 and $\widetilde { A } _ { j } \geq 0 . 7 0$ , where $d _ { j }$ and $A _ { j }$ are calculated between all spoofed and bona-fide samples. We identify the feature j as specific to attack a if its folded AUC for distinguishing spoofed samples generated by attack a from the bona-fide samples satisfies $\widetilde { A } _ { j , a } \geq 0 . 8 0$

For each feature, we select the low-, high-, or two-sided anomalous region and threshold(s) that maximize balanced accuracy $\mathrm { B A c c } = ( \mathrm { T P R } + \mathrm { T N R } ) / 2 [ 1 5 ]$ , in distinguishing spoofed from bona-fide audio. For an audio sample x, feature $j$ is flagged as anomalous when its value $f _ { j } ( x )$ falls within the selected region. Each flagged feature yields a reference pair $( e _ { \mathrm { i d } } ^ { \ast } , e _ { \mathrm { v a l } } ^ { \ast } ) = ( j , f _ { j } ( x ) )$ . These pairs constitute $e ^ { * }$ and are ranked by the balanced accuracy of their corresponding rules. If no feature is flagged, no anomaly is assigned.

Based on the label y and constructed reference evidence $e ^ { * }$ , AQAs are instantiated. T1 uses y as the target answer. For T2, the highest-ranked triggered feature, or no anomaly if no feature is triggered, serves as the correct option and is paired with measurement-inconsistent distractors. T3 masks the reference measurement and constructs distractors using controlled offsets with matched precision and units. T4 converts the triggered findings into an evidence-grounded reference rationale. T1–T3 use multiple semantically equivalent templates and seeded option permutations, while T4 uses a fixed label-neutral prompt. Partition labels and attack identities are used only for offline reference construction and are not included in the inputs to evaluated models.

## 2.4. Quality Control

Generated AQAs undergo automated checks for answer uniqueness, numerical traceability, and cross-record consistency. Six human reviewers further inspect a stratified subset of 5,400 AQAs covering labels, attack groups, feature categories, and no-anomaly cases, assessing answer correctness, ambiguity, evidence consistency, and linguistic clarity.

## 3. METHOD

As shown in Section 4.2, many existing ALMs struggle to identify and quantify fine-grained acoustic evidence. To address this issue, we propose BAEA, by augmenting a frozen ALM with a controlled signal-analysis tool to obtain acoustic features without training, as shown in Fig.1.

Bona-fide-based acoustic reference. The controlled tool library uses librosa [16] to compute 35 pre-registered spectral, cepstral, prosodic, and energy features. For feature $j ,$ we estimate its median $m _ { j }$ and median absolute deviation MAD<sub>j</sub> exclusively from bona fide samples in 19LA training partition. No evaluation labels, attack identities, or reference answers are used during BAEA inference. The robust deviation of an input audio x is $z _ { j } ( x ) =$ $( v _ { j } ( x ) - m _ { j } ) / ( 1 . 4 8 2 6 \mathrm { M A D } _ { j } + \epsilon )$ , where $v _ { j } ( x )$ is the deterministic measurement. The sign of $z _ { j } ( x )$ indicates the deviation direction, and $| z _ { j } ( x ) |$ measures acoustic rarity; a large deviation is treated as evidence rather than direct proof of spoofing. Each evidence record retains the feature name, measured value, reference median, robust deviation, direction, and status. The ALM can invoke only whitelisted acoustic tools and cannot modify their outputs.

Evidence-acquisition policies. We instantiate BAEA with two levels of ALM control over evidence acquisition. One is

FIXED, which measures all 35 features and returns the topranked, category-balanced deviations under a task-specific evidence budget. This policy provides stable evidence coverage without requiring the ALM to determine which acoustic categories should be inspected. The other is ADAPTIVE, where the ALM selects features from spectral, cepstral, prosodic, and energy tools. A controller validates the request, removes invalid or repeated categories, enforces the evidence budget, and exposes only the permitted measurements.

Task-conditioned inference. The same evidence interface for all tasks is used in BAEA. For T2 and T3, it maps each option o to controlled features $\mathcal { F } ( o )$ and computes

$$
d ( o ) = \operatorname* { m i n } _ { j \in \mathscr { F } ( o ) } \frac { | v _ { j } ( x ) - \widetilde { v } _ { j , o } | } { | \widetilde { v } _ { j , o } | + \epsilon } ,\tag{2}
$$

where $\widetilde { v } _ { j , o }$ is the value stated by o. The minimum-error candidate is selected, preventing the ALM from overriding deterministic measurements. For T1, the frozen ALM jointly uses the audio and selected evidence, with spoof scores averaged over original and A/B-swapped option orders. For T4, BAEA renders an immutable grounded core and retains an ALM interpretation only if it introduces no unmeasured feature, unsupported value, or attack metadata.

## 4. EXPERIMENTS

## 4.1. Experimental Setup

Models. We evaluate four ALMs on GH200 GPUs: Qwen2- Audio-7B-Instruct (Qwen2-Audio) [17], Qwen2.5-Omni-7B (Qwen2.5-Omni) [18], MiniCPM-o-4.5 (MiniCPM-o) [19], and MOSS-Audio-8B-Instruct (MOSS-Audio) [20]; and two proprietary models via APIs: Gemini-3.1-Flash-Lit (Gemini-Flash) [21] and GPT-Audio-1.5 (GPT-Audio) [22].

Metrics. T1 uses macro-F1 (F1) [2] and equal error rate (EER) [23]; T2–T3 use accuracy (ACC) [12]; and T4 uses BERTScore-F1 (B-F1) [24] and reference grounding (Ref-G). Ref-G is scored from 1–5 by a blinded GPT-4o-mini judge [25], while all other metrics are reported as percentages.

Evaluation scale. 2,000 audio are sampled from each partition, totaling 68,004 AQAs. The subsets retain all attack types and closely match the full class and feature distributions, with a maximum Jensen–Shannon divergence of 0.031.

Prompt control. T1–T3 use three seeded, semantically equivalent templates with independently permuted options, while T4 uses a fixed label-neutral instruction. A fixedtemplate ablation on 200 19LA Evaluation recordings yields maximum variations of 5.7 points for T2 ACC and 3.0 points for both T3 ACC and T4 B-F1. As T1 EER varies by up to 15.0 points, we report position-ensembled EER values.

## 4.2. ALMs’ Performance on SEAR

Table 1 reports the performance of six ALMs across the four SEAR partitions. For T1, the best F1 ranges from 45.07% to

Table 1. Zero-shot performance (%) across SEAR tasks and partitions. Note that the training, development and evaluation sets are abbreviated as Tr, Dev and Eval, respectively.
<table><tr><td>ALM</td><td colspan="4">T1 (F1 ↑)</td><td colspan="4">T2 (ACC ↑)</td><td colspan="4">T3 (ACC ↑)</td><td colspan="4">T4 (B-F1 ↑)</td></tr><tr><td></td><td>19Tr</td><td>19Dev</td><td>19Eval</td><td>21Eval</td><td>19Tr</td><td>19Dev</td><td>19Eval</td><td>21Eval</td><td>19Tr</td><td>19Dev</td><td>19Eval</td><td>21Eval</td><td>19Tr</td><td>19Dev</td><td>19Eval</td><td>21Eval</td></tr><tr><td>Qwen2-Audio</td><td>39.28</td><td>40.50</td><td>39.13</td><td>42.92</td><td>23.90</td><td>24.20</td><td>24.20</td><td>23.90 23.61</td><td></td><td>21.10</td><td>22.95</td><td>24.73</td><td>83.74</td><td>83.50</td><td>83.32</td><td>83.78</td></tr><tr><td>Qwen2.5-Omni</td><td>34.19</td><td>30.85</td><td>25.24</td><td>24.00</td><td>20.30</td><td>21.05</td><td>18.65</td><td>23.9015.07</td><td></td><td>12.84</td><td>16.26</td><td>15.67 84.01</td><td></td><td>83.80</td><td>83.60</td><td>83.81</td></tr><tr><td>MiniCPM-o</td><td>13.95</td><td>14.00</td><td>13.76</td><td>16.06</td><td>19.30</td><td>16.65</td><td>14.65</td><td>23.70</td><td>22.16</td><td>20.23</td><td>22.25</td><td>25.77</td><td>84.43</td><td>84.19</td><td>83.97</td><td>84.16</td></tr><tr><td>MOSS-Audio</td><td>33.66</td><td>32.04</td><td>23.52</td><td>25.56</td><td>18.65</td><td>14.55</td><td>11.80</td><td>20.25</td><td>22.55</td><td>21.69</td><td>23.46</td><td>25.79 83.99</td><td></td><td>83.73</td><td>83.61</td><td>83.95</td></tr><tr><td>Gemini-Flash</td><td>49.27</td><td>52.47</td><td>48.61</td><td>45.07</td><td>20.00</td><td>18.10</td><td>14.65</td><td>24.30</td><td>14.79</td><td>13.05</td><td>15.32</td><td>16.11</td><td>85.17</td><td>85.03</td><td>84.60</td><td>84.82</td></tr><tr><td>GPT-Audio</td><td>14.41</td><td>15.09</td><td>13.17</td><td>14.02</td><td>18.25</td><td>18.10</td><td>18.20</td><td>19.75 10.97</td><td></td><td>10.99</td><td>12.52</td><td>9.71 83.83</td><td></td><td>83.70</td><td>83.26</td><td>83.42</td></tr></table>

Table 2. Effects of T2/T3 context on T1 and T4.
<table><tr><td>Split</td><td>Model</td><td>T1 ∆EER Self/Oracle</td><td>T4 Ref-G Self/Oracle</td><td>T4 B-F1 Self/Oracle</td></tr><tr><td>19LA</td><td>Qwen2-Audio Qwen2.5-Omni MiniCPM-o MOSS-Audio</td><td>-3.85/+1.46 -1.02/+0.60 -1.90/-14.71 +5.77/-1.40</td><td>2.542/3.521 1.915/2.550 1.754/2.672 1.851/2.929</td><td>88.23/88.14 89.69/90.48 90.19/90.19 90.82/90.94</td></tr><tr><td>21LA</td><td>Qwen2-Audio Qwen2.5-Omni MiniCPM-o MOSS-Audio</td><td>-11.86/-4.11 -1.53/+1.62 +3.23/+0.91 +3.57/-0.56</td><td>2.935/3.884 2.175/2.943 1.830/3.143 1.900/3.056</td><td>88.61/88.37 90.65/90.91 91.37/91.15 91.79/91.76</td></tr></table>

52.47%, showing that direct spoofing detection remains challenging. Performance is lower on the fine-grained evidence tasks. T2 accuracy remains close to the 25% random-choice baseline, while T3 accuracy never exceeds 25.79%. In contrast, T4 B-F1 is consistently high, ranging from 83.26% to 85.17%, with little variation across models and partitions. This discrepancy shows that ALMs can generate lexically plausible rationales despite being unable to reliably identify or quantify the corresponding acoustic evidence.

We further examine whether T2/T3 answers benefit T1 and T4. In Table 2, Self uses model-generated T2/T3 answers, whereas Oracle uses oracle-assisted context while retaining the model answers and is not a pure gold-only upper bound. We define $\Delta \mathrm { E E R } = \mathrm { E E R } _ { \mathrm { D i r e c t } } - \mathrm { E E R } _ { \mathrm { C o n t e x t } }$ , so positive values indicate improvement. Oracle context increases T4 Ref-G by 0.635–1.313 across all models and partitions, while B-F1 changes by at most 0.79 points. However, neither context consistently improves T1, showing that accurate evidence does not automatically yield reliable detection.

## 4.3. Tool-Augmented Acousitc Evidence Reasoning

Table 3 compares both BAEA variants with Tool-only and vanilla Qwen2.5-Omni. Tool-only integrates its deterministic acoustic measurement for each task in SEAR without ALM inference. FIXED reduces T1 EER from 36.34% to 24.53% on 19LA and from 36.43% to 33.54% on 21LA, while improving T2/T3 and T4 Ref-G. Tool-only reproduces all T2/T3 decisions but shows dataset-dependent T1 performance, indicating that the benefit of integrating tool evidence with the ALM is domain dependent. ADAPTIVE retains the same T2/T3 binding accuracy but underperforms FIXED on T1 and T4 in both partitions, indicating that stable feature coverage is currently more reliable than autonomous evidence selection.

Table 3. BAEA results on SEAR.
<table><tr><td>Split</td><td>Method</td><td>T1 EER↓</td><td>T2 ACC↑</td><td>T3 ACC↑</td><td>T4 Ref-G↑</td></tr><tr><td rowspan="4">19LA</td><td>Vanilla</td><td>36.34</td><td>18.65</td><td>16.26</td><td>1.174</td></tr><tr><td>Tool-only</td><td>17.65</td><td>95.85</td><td>100.00</td><td></td></tr><tr><td>FIXED</td><td>24.53</td><td>95.85</td><td>100.00</td><td>2.026</td></tr><tr><td>ADAPTIVE</td><td>34.33</td><td>95.85</td><td>100.00</td><td>1.672</td></tr><tr><td rowspan="4">21LA</td><td>Vanilla</td><td>36.43</td><td>23.90</td><td>15.67</td><td>1.196</td></tr><tr><td>Tool-only</td><td>41.64</td><td>84.05</td><td>99.99</td><td></td></tr><tr><td>FIXED</td><td>33.54</td><td>84.05</td><td>99.99</td><td>1.356</td></tr><tr><td>ADAPTIVE</td><td>41.58</td><td>84.05</td><td>99.99</td><td>1.176</td></tr></table>

## 4.4. Effect of Evidence Quality

With BAEA-FIXED, we compare no, matched, and swapped evidence between recordings on the same 200 samples from each evaluation partition. Table 4 reports paired improvements with 95% bootstrap confidence intervals. Positive values indicate improvement over the baseline, and a star indicates that the confidence interval excludes zero. Matched evidence improves T4 Ref-G but not T1 consistently across domains. Swapped evidence significantly increases T1 EER by 30.28 and 18.08 points and reduces Ref-G by 0.605 and 0.755, demonstrating that misleading evidence degrades both detection and grounding.

Table 4. Paired evidence intervention.
<table><tr><td>Split</td><td>Comparison</td><td>∆EER</td><td>∆Ref-G</td></tr><tr><td rowspan="2">19LA</td><td>Correct vs. None</td><td>+11.11</td><td>+0.675*</td></tr><tr><td>Swapped vs. Correct</td><td>-30.28*</td><td>-0.605*</td></tr><tr><td rowspan="2">21LA</td><td>Correct vs. None</td><td>-9.04</td><td>+0.935*</td></tr><tr><td>Swapped vs. Correct</td><td>-18.08*</td><td>-0.755*</td></tr></table>

## 5. CONCLUSION

We first introduced SEAR, which benchmarks ALMs across four tasks beyond binary classification in ADD. Results reveal a gap between plausible rationales and fine-grained acoustic evidence. We further presented BAEA, grounding the inference of an ALM by contrasting the test audio with bona fide references via controlled deterministic acoustic measurements, improving evidence grounding without model updates. However, BAEA relies on a relatively stable bona fide reference distribution. Future work should explore an adaptive reference modeling under evolving audio distributions.

## 6. REFERENCES

[1] Sreyan Ghosh et al., “GAMA: A large audio-language model with advanced audio understanding and complex reasoning abilities,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 6288–6313.

[2] Hao Gu et al., “ALLM4ADD: Unlocking the capabilities of audio large language models for audio deepfake detection,” in Proceedings of the 33rd ACM International Conference on Multimedia, New York, NY, USA, 2025, MM ’25, p. 11736–11745, Association for Computing Machinery.

[3] Artem Dvirniak, Evgeny Kushnir, Dmitrii Tarasov, Artem Iudin, Oleg Kiriukhin, Mikhail Pautov, Dmitrii Korzh, and Oleg Y. Rogov, “Towards robust speech deepfake detection via human-inspired reasoning,” 2026.

[4] Yuankun Xie et al., “Interpretable all-type audio deepfake detection with audio llms via frequency-time reinforcement learning,” 2026.

[5] Runkun Chen, Yixiong Fang, Pengyu Chang, Yuante Li, Massa Baali, and Bhiksha Raj, “Audio language model for deepfake detection grounded in acoustic chain-ofthought,” 2026.

[6] Xuenan Xu, Yiming Ren, Liwei Liu, Wen Wu, Baoxiang Li, Chaochao Lu, Shuai Wang, and Chao Zhang, “HoliAntiSpoof: Audio llm for holistic speech antispoofing,” 2026.

[7] Xin Wang et al., “ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech,” Computer Speech & Language, vol. 64, pp. 101114, 2020.

[8] Xuechen Liu et al., “ASVspoof 2021: Towards spoofed and deepfake speech detection in the wild,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2507–2522, 2023.

[9] Xin Wang et al., “ASVspoof 5: Design, collection and validation of resources for spoofing, deepfake, and adversarial attack detection using crowdsourced speech,” Computer Speech & Language, vol. 95, pp. 101825, 2026.

[10] Haibin Wu, Yuan Tseng, and Hung yi Lee, “Codec-Fake: Enhancing Anti-Spoofing Models Against Deepfake Audios from Codec-Based Speech Synthesis Systems,” in Interspeech 2024, 2024, pp. 1770–1774.

[11] Xuanjun Chen et al., “CodecFake+: A large-scale neural audio codec-based deepfake speech dataset,” arXiv preprint arXiv:2501.08238, 2025.

[12] Jian-Yu Jiang-Lin et al., “TriDF: Evaluating perception, detection, and hallucination for interpretable deepfake detection,” 2026.

[13] Siqian Tong, Xuan Li, Yiwei Wang, Baolong Bi, Yujun Cai, Shenghua Liu, Yuchen He, and Chengpeng Hao, “AuTAgent: A reinforcement learning framework for tool-augmented audio reasoning,” 2026.

[14] Gijs Wijngaard, Elia Formisano, Michel Dumontier, and Jenia Jitsev, “AudioToolAgent: An agentic framework for audio-language models,” 2026.

[15] Kay Henning Brodersen, Cheng Soon Ong, Klaas Enno Stephan, and Joachim M. Buhmann, “The balanced accuracy and its posterior distribution,” in 2010 20th International Conference on Pattern Recognition, pp. 3121– 3124.

[16] Brian McFee, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, Oriol Nieto, et al., “librosa: Audio and music signal analysis in python.,” SciPy, vol. 2015, no. 18-24, pp. 7, 2015.

[17] Yunfei Chu et al., “Qwen2-Audio technical report,” arXiv preprint arXiv:2407.10759, 2024.

[18] Jin Xu et al., “Qwen2.5-Omni technical report,” arXiv preprint arXiv:2503.20215, 2025.

[19] Junbo Cui et al., “MiniCPM-o 4.5: Towards real-time full-duplex omni-modal interaction,” 2026.

[20] OpenMOSS Team, “MOSS-Audio technical report,” https://github.com/OpenMOSS/ MOSS-Audio, 2026, GitHub repository.

[21] Google DeepMind, “Gemini 3.1 flash-lite model card,” https://deepmind.google/models/ model-cards/gemini-3-1-flash-lite/, 2026, Accessed: 2026-05-23.

[22] OpenAI, “GPT-Audio-1.5 model,” https: //developers.openai.com/api/docs/ models/gpt-audio-1.5, 2026, Accessed: 2026-09-13.

[23] Xiang Li, Pin-Yu Chen, and Wenqi Wei, “Where are we in audio deepfake detection? a systematic analysis over generative and detection models,” ACM Trans. Internet Technol., vol. 25, no. 3, Aug. 2025.

[24] Tianyi Zhang\*, Varsha Kishore\*, Felix Wu\*, Kilian Q. Weinberger, and Yoav Artzi, “BERTScore: Evaluating text generation with bert,” in International Conference on Learning Representations, 2020.

[25] OpenAI, “gpt-audio-mini model card,” https: //platform.openai.com/docs/models/ gpt-audio-mini, 2025, Accessed: 2026-05-23.