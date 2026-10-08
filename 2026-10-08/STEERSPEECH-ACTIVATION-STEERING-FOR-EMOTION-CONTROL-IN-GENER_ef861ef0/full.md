# STEERSPEECH: ACTIVATION STEERING FOR EMOTION CONTROL IN GENERATED SPEECH

Afsara Benazir<sup>1,2</sup>, Darius Petermann´ <sup>2</sup>, Felix Xiaozhu Lin<sup>1</sup>, Salar Rahili<sup>2</sup>

<sup>1</sup>University of Virginia <sup>2</sup>Netflix, Inc.

## ABSTRACT

Pretrained text-to-speech (TTS) models can generate expressive speech, but reliable inference-time emotion control remains challenging: prompts and reference audio offer coarse, inconsistent control, whereas specialized conditioning and model adaptation require costly training. We present SteerSpeech, a lightweight activation-steering framework that controls emotion by injecting steering vectors into hidden activations. For each target emotion we train a lightweight low-rank transform, using a multi-expert objective that encourages monotonic emotion control while preserving speaker identity and linguistic content, constraining steering drift, and keeping the TTS backbone frozen. To optimize through discrete speech tokens, we introduce a two-pass generation-and-replay pipeline using a straight-through estimator to backpropagate expert supervision through sampled tokens. At inference, a target-emotion steering direction is optimized with its respective transform and injected into the base TTS model. Objective and subjective evaluations with Qwen3-TTS across seen, unseen, and accented speakers show stronger continuous emotion control with limited speaker and content degradation. SteerSpeech achieves 1.08×–7.12× baseline target-emotion scores and for a representative emotion subjectively, it receives 78.1%–96.8% intensity preference and 1.43×–1.46× speaker-identity preservation at high steering strengths.

Index Terms— Text-to-speech, Emotion control, Speech synthesis, Activation steering

## 1. INTRODUCTION

Modern text-to-speech (TTS) systems generate highly natural speech, yet precise inference-time control of emotion remains challenging. Existing approaches condition synthesis on emotion labels, reference speech, natural-language descriptions, or prosodic controls [1, 2, 3]. Such controls can be coarse or inconsistent, while introducing a specialized encoder or adapting the TTS model for emotion controllability requires computationally expensive training and substantial labeled data. A more practical approach should enable downstream modification of attributes such as emotion, accent, and speaking style without retraining the underlying TTS model.

Activation steering [4] provides a lightweight alternative by modifying a pretrained model’s intermediate representations using a steering direction derived from the difference between mean of target and neutral activations. Prior work applies this approach to emotional speech synthesis: EmoSteer-TTS [5] derives steering directions by identifying and modifying activation dimensions most relevant to the target emotion, whereas EmoShift [6] transforms hidden representations into emotion-specific steering directions through learned mappings; EmoKnob [7] directly manipulates speaker embeddings. However, the success of such intervention depends on how the steering direction is constructed, where it is applied, and how strongly it is injected. Steering directions may provide weak or unstable control, capture unrelated properties, alter speaker identity, linguistic content, or naturalness [8] and vary with the steering strength, making continuous control difficult [5].

Our goal is to enable continuous and fine-grained inference-time control of emotion in generated speech without retraining the TTS backbone or altering other attributes such as accent and speaking style. Our key insight is that reliable emotion steering requires more than strengthening the target emotion: the steering direction should be jointly guided by emotion, speaker, and transcription experts to preserve identity and linguistic content.

To that end, we propose SteerSpeech, a lightweight framework that learns a low-rank residual transform using multi-expert supervision enabled by a two-pass generation-and-replay pipeline that backpropagates through discrete speech tokens. Given a coarse emotion contrast, i.e. the activation difference between target and neutral emotions, the transform produces an optimized steering direction that strengthens the target emotion while preserving other speech attributes. We jointly optimize the transform across multiple steering strengths to encourage a monotonic increase in emotion intensity while limiting unpredictable behavior and off-target drift, all while keeping the TTS backbone frozen. Our contributions are:

• We introduce SteerSpeech, a lightweight activation-steering framework for fine-grained, continuous inference-time control of emotion in generated speech without retraining the underlying TTS model.

• We learn a low-rank residual transform of emotion-steering directions using a multi-expert objective that explicitly promotes monotonic emotion control while preserving speaker identity, linguistic content, and speech quality.

• We demonstrate through objective and subjective evaluations that SteerSpeech provides fine-grained emotion controllability compared to baselines and generalizes across seen, unseen, and accented speakers.

## 2. BACKGROUND

Emotion control in TTS. Existing systems guide emotional expression through conditioning signals at multiple levels. Categorical emotion labels provide simple utterance-level control but represent emotion using a small set of predefined classes [9]. Reference speech transfers expressive characteristics from example speech [10], while natural-language descriptions provide a more flexible interface for specifying emotional style [2]. Finer control uses hierarchical emotion distributions [1] or pitch and energy sketches for local prosodic trajectories [3]. However, these methods often require task-specific conditioning modules, joint TTS training, preference-based finetuning [11], or substantial backbone adaptation, increasing parameter and computational costs.

![](images/22616599f2adf8dadc99f8dab71e139438068ca372c4dc772052032679479211.jpg)  
Fig. 1. SteerSpeech overview.

Activation steering for controllable emotion synthesis. Activation steering [4] offers a lightweight mechanism for controlling a pretrained model by modifying its hidden representations while keeping the model parameters frozen. Given a hidden state $\it { h ^ { ( l ) } }$ at layer $l ,$ a target-emotion direction is added with steering strength α:

$$
v _ { e } ^ { ( l ) } = \bar { h } _ { e } ^ { ( l ) } - \bar { h } _ { n } ^ { ( l ) } , \qquad \widetilde { h } ^ { ( l ) } = h ^ { ( l ) } + \alpha v _ { e } ^ { ( l ) } ,\tag{1}
$$

where $\bar { h } _ { e } ^ { ( l ) }$ and $\bar { h } _ { n } ^ { ( l ) }$ denote the mean target-emotion and neutral activations, averaged over all decoding frames in respective utterances. Prior systems realize this intervention differently: EmoSteer-TTS [5] uses neutral-to-emotion activation differences, while EmoShift [6] uses learned emotion-specific offsets. Co-CoEmo [12] identifies effective steering sites and composes meandifference directions for single and mixed emotions, whereas HE-Vector [13] constructs and combines emotion and dialect vectors in model parameter space. In contrast, SteerSpeech uses explicit multiexpert supervision for monotonic emotion intensity and attribute preservation and learns a low-rank transform that optimizes a target steering direction at an internal layer.

## 3. SYSTEM DESIGN

## 3.1. Overview

We propose SteerSpeech, a lightweight, plug-and-play framework that learns to refine activation-steering directions for controllable emotion synthesis.

Workflow, is illustrated in Fig. 1. For each target emotion, we train a lightweight transform. Training consists of four steps: 1 extracting raw emotion contrasts from matched emotion and neutral utterance pairs, 2 transforming the contrast and injecting it into a frozen speech generator, $\textcircled{8}$ scoring the synthesized speech with frozen emotion, speaker, and transcription experts, and 4 updating only the learned transform, $T _ { \theta }$ using their weighted objective function (Eq. 6). Except for $T _ { \theta } ,$ all components of the underlying speech generation pipeline remain frozen (Sec. 3.3).

Inference. Given a raw emotion contrast $v _ { e }$ i.e., mean activation difference between target-emotion and neutral utterances, $T _ { \theta }$ takes $v _ { e }$ as input and produces a refined steering direction. Steering with $T _ { \theta } ( v _ { e } )$ produces progressively stronger target emotion as the steering strength α increases, while preserving speaker identity and linguistic content.

System Components. SteerSpeech consists of the following:

(1) Frozen speech generator. We use a pretrained autoregressive TTS model conditioned on target text and reference speech that predicts discrete speech tokens, which are subsequently decoded into waveform audio. We steer at an intermediate hidden layer following insights from [12].

(2) Low-rank learned transform. The transform is the only trainable component: it normalizes a raw emotion contrast $v _ { e }$ and applies a rank-r normalized-affine residual correction,

$$
\bar { v } _ { e } = \frac { v _ { e } } { \| v _ { e } \| _ { 2 } } , \qquad q = \bar { v } _ { e } + U ( D \bar { v } _ { e } ) + b ,\tag{2}
$$

where $D \in \mathbb { R } ^ { r \times d }$ and $U \in \mathbb { R } ^ { d \times r }$ parameterize the rank-r correction, and $b \in \mathbb { R } ^ { d }$ is the learned bias. With $d = 1 0 2 4$ and $r = 1 6$ the transform contains 33,793 parameters per emotion. $U ( D \bar { v } _ { e } )$ is linear: D is randomly initialized with orthonormal rows, while U and b are zero-initialized, making the initial residual zero and preserving the input direction. The corrected vector q is normalized and rescaled: $q / \lVert q \rVert \mathrm { ; }$ <sub>2</sub> sets the direction and s the magnitude, where $s _ { 0 }$ is the median $\ell _ { 2 } \cdot$ -norm of input contrasts and η a learned scalar.

$$
s = s _ { 0 } ( 1 + \beta \operatorname { t a n h } \eta ) , \qquad v _ { e } ^ { \star } = T _ { \theta } ( v _ { e } ) = s \left( { \frac { q } { \| q \| _ { 2 } } } \right)\tag{3}
$$

We set $\beta = 0 . 2$ , restricting the magnitude to within $\pm 2 0 \%$ of the reference magnitude $s _ { 0 } , \mathrm { i . e . , } 0 . 8 s _ { 0 } < \| T _ { \theta } ( v _ { e } ) \| _ { 2 } < 1 . 2 s _ { 0 }$ . This allows the transform to refine the steering direction while constraining large changes in geometry and magnitude. The direction $\alpha v _ { e } ^ { \star }$ is injected at layer $l ,$ after which the hidden-state norm is restored to its pre-injection value.

(3) Frozen expert scoring modules. We use three frozen expert models to provide supervision: an emotion expert measures targetemotion strength, a speaker expert assesses identity preservation relative to the reference speech, and a transcription expert evaluates consistency with the target text.

## 3.2. Learning Objective

We train the learned transform using complementary objectives for emotion control, speaker preservation, and transcript preservation, together with a regularization term that constrains changes to the original steering vector.

Monotonic emotion supervision. A central goal of SteerSpeech is continuous controllability: increasing the steering strength α should produce a corresponding increase in target-emotion intensity. We therefore train over an ordered set of steering strengths $\alpha _ { 0 } < \alpha _ { 1 } <$ · · · < α<sub>K</sub> .

Let $\ell _ { j } \equiv \ell ( \alpha _ { j } )$ denote the target-emotion-versus-rest log-odds of speech generated at steering strength $\alpha _ { j }$ . We encourage monotonicity by requiring the increase in emotion score between adjacent steering strengths to be proportional to the corresponding increase in α:

$$
\mathcal { L } _ { \mathrm { m o n o } } = \frac { 1 } { K } \sum _ { j = 0 } ^ { K - 1 } \left[ \kappa \Delta \alpha _ { j } - \Delta \ell _ { j } \right] _ { + } ,
$$

where $\Delta \alpha _ { j } = \alpha _ { j + 1 } - \alpha _ { j } , \Delta \ell _ { j } = \ell _ { j + 1 } - \mathrm { s g } ( \ell _ { j } ) , [ z ] _ { + } = \operatorname* { m a x } ( 0 , z ) ,$ κ controls the required increase in emotion score per unit increase in steering strength, and sg(·) denotes stop-gradient. This hinge loss encourages the desired ordering across steering strengths. However, monotonic ordering alone admits a trivial solution in which the emotion score increases with α while remaining weak at all operating points. We therefore include a term, where $p _ { e } ( \alpha _ { \mathrm { m a x } } )$ denotes the target-emotion probability at the maximum strength and λ controls how strongly the model is encouraged to express the target emotion:

$$
\mathcal { L } _ { \mathrm { e m o t i o n } } = \mathcal { L } _ { \mathrm { m o n o } } - \lambda \log p _ { e } ( \alpha _ { \mathrm { m a x } } )\tag{4}
$$

Speaker-preservation loss. To measure speaker changes introduced specifically by steering, we use the corresponding unsteered generation as a reference. We penalize deviations from the unsteered output’s similarity to the reference speaker. Let e<sub>T</sub>, e<sub>U</sub>, and $e _ { R }$ denote the speaker embeddings of the steered output, unsteered output, and reference speech, respectively. We define

$$
s _ { T } = \cos ( e _ { T } , e _ { R } ) , \qquad s _ { U } = \mathrm { s g } ( \cos ( e _ { U } , e _ { R } ) ) , \qquad \mathcal { L } _ { \mathrm { s p e a k e r } } = | s _ { T } - s _ { U } |\tag{5}
$$

Transcript-preservation loss. Given generated audio x and target transcript tokens $y _ { 1 : T }$ , we use the differentiable teacher-forced token negative log-likelihood, $\begin{array} { r } { \mathcal { L } _ { \mathrm { A S R } } = - \frac { 1 } { T } \sum _ { t = 1 } ^ { T } } \end{array}$ log $P ( y _ { t } \mid y _ { < t } , x )$

$$
\mathcal { L } _ { \mathrm { r e g } } = 0 . 0 1 \left[ 1 - \cos ( q , \bar { v } _ { e } ) \right] + 0 . 0 1 \eta ^ { 2 }
$$

Combined learning objective. The per-sample losses are combined as follows: the resulting objective is backpropagated only to the learned steering transform.

$$
\mathcal { L } _ { \mathrm { u p d a t e } } = \mathcal { L } _ { \mathrm { e m o t i o n } } + 3 2 \mathcal { L } _ { \mathrm { s p e a k e r } } + 0 . 2 \mathcal { L } _ { \mathrm { A S R } } + \mathcal { L } _ { \mathrm { r e g } }\tag{6}
$$

## 3.3. Methodology

Training $T _ { \theta }$ is challenging because expert supervision is available only after autoregressive generation and decoding, while discrete codec-token sampling is non-differentiable; thus, gradients from the emotion, speaker, and transcription experts cannot directly propagate back to the transform.

We address this with a two-pass pipeline: (1) we extract emotion-to-neutral activation differences from matched utterances and transform each contrast using T<sub>θ</sub>; (2) in Pass 1, for each contrast and steering strength α, we inject the transformed direction at layer l and autoregressively sample discrete codec tokens using standard inference; (3) in Pass 2, we replay the sampled sequence with teacher forcing [14] and a straight-through estimator (STE) [15], retaining the discrete codec tokens in the forward pass while propagating gradients through their continuous representations, which a differentiable decoder converts to waveforms; and (4) frozen experts evaluate emotion intensity, speaker preservation, and transcript consistency across multiple α values. This joint evaluation encourages monotonic target-emotion increase while constraining speaker and content drift. The combined loss in Eq. 6 updates only $T _ { \theta } ,$ while the TTS backbone, decoder, and experts remain frozen. At inference, the expert evaluators and differentiable replay path are removed.

## 4. RESULTS AND ANALYSIS

## 4.1. Experimental Setup

Datasets. We use the English subset of the Emotional Speech Dataset (ESD) [16], containing 350 parallel utterances per speaker from 10 native English speakers; Audio is sampled at 16 kHz and average duration is 5-10 seconds. Evaluation covers (1) seen: unseen utterances from in-domain speakers (2) unseen: an unseen speaker split with two held-out English speakers, and (3) accented: cross-corpus generalization on four non-native Mandarin speakers from L2-ARCTIC [17]. We reserve a separate held-out speaker split for validation, distinct from the train and test splits.

Models. We use Qwen3-TTS-0.6B [18], a popular TTS model with Qwen language model backbone. Our rank-16 transform has 33K parameters. The expert models used during training were emotion2vec for emotion classification [19], WavLM for speaker verification [20], and Whisper for ASR [21]. For evaluation, we used an XLSR-53-based emotion classifier [22], ECAPA-TDNN for speaker verification [23], wav2vec 2.0 for ASR-based WER [24] and UTMOS [25] for speech-quality assessment.

Implementation Details. We use temperature 0.9, top- $k = 5 0 ,$ , top-$p = 1 . 0$ , a fixed seed, and steer at layer 15. For monotonic supervision, we use $\alpha \in \{ 0 . 3 , 0 . 4 , \hdots , 1 . 0 \}$ . The transform is trained with AdamW (batch $3 2 .$ learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , and 5% warmup with cosine decay). For evaluation, we construct one global difference of means contrast per emotion from held-out speakers and transform it using $T _ { \theta }$

Baselines. We compare against the following baselines: (1) Emotion reference: generates emotional speech using a reference emotional utterance as conditioning signal [26] (2) Naive difference-ofmeans (or Naive): Constructs a steering direction by subtracting the mean neutral representation from the mean representation of the target emotion. (3) SteerSpeech (ours): Our system, evaluated on the three splits described above: seen, unseen and accented. We report both split-level and pooled performance. Additional baselines were omitted because they require model-specific reimplementation and cross-paper results are not directly comparable.

Evaluation Metrics. Objective metrics include (1) Emotion confidence (↑) (2) Top-1 accuracy (↑), the percentage of samples classified as the target emotion (3) Speaker drift (↓) calculated as how much the steered speech deviates from the unsteered synthesized speech to account for any model bias and (4) Word error rate (WER) (↓) between synthesized speech transcript and reference text and (5) UTMOS (↑), predicted mean opinion score (MOS) assessing the naturalness of synthesized speech. Subjective metrics include (1) emotion preference (2) speaker-ID preference, and (3) naturalness in Survey 1; and (1) speaker-ID preservation and (2) perceived emotion increase in Survey 2. All subjective metrics are ↑.

Subjective Evaluation. We conducted two blinded anger listening studies with 55 participants and at least 16 labels per comparison. Survey 1 used 60 A/B comparisons to assess preferences between SteerSpeech and baselines across all splits. Survey 2 compared five adjacent steering-strength (α) transitions for SteerSpeech and Naive using identical speakers and text. Listeners judged whether higher strength increased anger while preserving speaker identity.

## 4.2. End-to-End Results

Overall performance. Table 1 compares end-to-end performance at each method’s best-performing α from grid search (peak achievable performance); Fig. 2 confirms the advantage holds at matched α. SteerSpeech improves target-emotion scores by 3.7-18.4 percentage points (pp) and top-1 accuracy by 12.5-19.0pp across emotions. Others are emotion-dependent: anger reduces both speaker drift (by 0.038) and WER (by 0.9pp), whereas happy and sad incur modestly higher speaker drift (+0.011/+0.027), with sadness exhibiting the lowest emotion score (0.235). This can be explained by emotion-specific acoustic cues: anger exhibits salient high-arousal characteristics that may be easier to steer, whereas happiness relies on subtler valence cues and sadness more strongly on temporal and rhythmic patterns; similar trend appears in recent work [27]. Steer-Speech UTMOS score improves over Naive by 0.103-0.609 across emotions. The emotion reference baseline has the weakest emotion intensity (SteerSpeech is $1 . 3 5 \times - 7 . 1 2 \times$ higher); its higher naturalness and lower speaker drift largely reflect smaller changes from the original speech.

Table 1. End-to-end comparison across target emotions.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Subset</td><td rowspan="2">Emo.↑ Top-1↑</td><td colspan="4">Angry</td><td colspan="5">Happy</td><td colspan="5">Sad</td></tr><tr><td></td><td>Drift↓</td><td>WER↓</td><td>UTMOS↑</td><td>Emo.↑ Top-1↑</td><td></td><td>Drift↓</td><td>WER↓</td><td>UTMOS↑</td><td></td><td>Emo.↑ Top-1↑ Drift↓</td><td></td><td></td><td>WER↓ UTMOS↑</td></tr><tr><td>Emotion reference</td><td>Pooled</td><td>0.617</td><td>0.775</td><td>0.189</td><td>4.9</td><td>4.077</td><td>0.333</td><td>0.394</td><td>0.172</td><td>2.4</td><td>4.131</td><td>0.033</td><td>0.019</td><td>0.158</td><td>6.0</td><td>4.256</td></tr><tr><td>Naive</td><td>Pooled</td><td>0.651</td><td>0.755</td><td>0.233</td><td>6.0</td><td>2.540</td><td>0.487</td><td>0.615</td><td>0.247</td><td>7.1</td><td>2.999</td><td>0.126</td><td>0.100</td><td>0.156</td><td>8.0</td><td>3.452</td></tr><tr><td></td><td>Seen</td><td>0.848</td><td>0.942</td><td>0.188</td><td>6.1</td><td>3.209</td><td>0.589</td><td>0.825</td><td>0.268</td><td>8.0</td><td>3.211</td><td>0.255</td><td>0.283</td><td>0.191</td><td>8.8</td><td>3.503</td></tr><tr><td>SteerSpeech (ours)</td><td>Unseen</td><td>0.807</td><td>0.925</td><td>0.161</td><td>6.1</td><td>3.142</td><td>0.554</td><td>0.800</td><td>0.275</td><td>15.5</td><td>2.955</td><td>0.166</td><td>0.175</td><td>0.151</td><td>10.6</td><td>3.752</td></tr><tr><td></td><td>Accented</td><td>0.824</td><td>0.975</td><td>0.252</td><td>2.3</td><td>2.980</td><td>0.302</td><td>0.425</td><td>0.208</td><td>4.7</td><td>4.146</td><td>0.246</td><td>0.250</td><td>0.192</td><td>4.2</td><td>3.513</td></tr><tr><td></td><td>Pooled</td><td>0.835</td><td>0.945</td><td>0.195</td><td>5.1</td><td>3.149</td><td>0.524</td><td>0.740</td><td>0.258</td><td>9.7</td><td>3.347</td><td>0.235</td><td>0.255</td><td>0.183</td><td>7.8</td><td>3.555</td></tr></table>

Table 2. Subjective Evaluation: A/B preference (%). Ours show maximum emotion intensity with competitive naturalness.
<table><tr><td>Comparison</td><td>Emotion preference ↑</td><td>Speaker ID preference ↑</td><td>Naturalness ↑</td></tr><tr><td>SteerSpeech (ours) vs. Naive</td><td>78.1/21.9</td><td>41.2/58.8</td><td> $4 7 . 7 / 5 2 . 3 $ </td></tr><tr><td>SteerSpeech (ours) vs. Emotion ref.</td><td>96.8/3.2</td><td>12.6/87.4</td><td>16.2/83.8</td></tr><tr><td>Naive vs. Emotion ref.</td><td>83.5/16.5</td><td>20.3/79.7</td><td>21.2/78.8</td></tr></table>

Table 3. Subjective Evaluation: Effect of steering strength. Steer-Speech better preserves speaker identity at high intensity (α).
<table><tr><td colspan="4">Speaker ID preserved (%) Anger increase (%)</td><td colspan="3"></td></tr><tr><td>Intensity (α) Ours Naive</td><td></td><td></td><td> $\Delta$ </td><td></td><td>Ours Naive</td><td> $\Delta$ </td></tr><tr><td> $0 . 0  0 . 3$ </td><td>89.2</td><td>84.2</td><td>+5.0</td><td>58.3</td><td></td><td> $4 7 . 5 \quad + 1 0 . 8 \quad$ </td></tr><tr><td> $0 . 3  0 . 5$ </td><td>87.5</td><td>84.9</td><td>+2.6</td><td>80.8</td><td></td><td> $6 7 . 2 \quad + 1 3 . 6 \phantom { 0 }$ </td></tr><tr><td> $0 . 5  0 . 7$ </td><td>65.8</td><td>67.5</td><td>-1.7</td><td>71.7</td><td>62.5</td><td>+9.2</td></tr><tr><td> $0 . 7  0 . 9$ </td><td>66.7</td><td>46.6</td><td>+20.1</td><td>83.3</td><td>41.4</td><td>+42.0</td></tr><tr><td> $0 . 9  1 . 0$ </td><td>69.2</td><td>47.4</td><td>+21.8</td><td>58.3</td><td></td><td> $3 6 . 2 \quad + 2 2 . 1 \hphantom { 0 0 0 }$ </td></tr></table>

Survey 1 (Table 2) shows that, at $\alpha = 1 . 0 $ , SteerSpeech is preferred for emotion intensity (78.1% vs. 21.9%), while Naive is preferred for speaker-ID preservation (58.8% vs. 41.2%), revealing an emotion–identity trade-off. Survey 2 (Table 3) shows that Steer-Speech better preserves speaker identity at higher intensities, exceeding Naive by 20.1 and 21.8pp and in perceived anger increases by 42.0 and 22.1 pp, demonstrating higher identity preservation at high emotion intensity.

Generalization. SteerSpeech generalizes to unseen and accented speakers. For anger, unseen speakers show modest reductions in emotion and Top-1 accuracy (0.848 to 0.807; 0.942 to 0.925), lower speaker drift (0.188 to 0.161), and unchanged WER. Accented speakers achieve 0.824 emotion, 0.975 Top-1, 2.3% WER, and 0.252 speaker drift. For happy and sad, unseen and accented speakers retain 51–97% of seen-speaker emotion and Top-1. Detailed subjective results confirm this: anger is perceived monotonically in 78.5%, 93.6%, and 65.6% of comparisons for seen, unseen, and accented speakers, respectively; accented speakers show the highest identity (55.6%) and naturalness (61.7%).

![](images/077d0f47f28a22783279fb05713925ffe792f7e292eee5f85b40f7f1d74b7dc6.jpg)  
Fig. 2. Tradeoff between target-emotion (angry) score and speaker drift at increasing steering strength (α). Ours show stronger emotion scores with lower speaker drift, especially at higher α, whereas Naive steering shows increasing drift & declining emotion intensity.

Ablation. We use angry as a representative emotion in ablation.

Emotion controllability/tradeoff. SteerSpeech achieves a consistently better trade-off between emotion intensity and speaker drift than the Naive baseline (Fig. 2). The optimal range is approximately $\alpha = 2 . 0 { \cdot } 2 . 5 ; \mathrm { a t } \alpha = 2 . 5$ , SteerSpeech reaches a peak emotion probability of 0.834 with slightly lower drift than Naive. Beyond $\alpha \approx 3 ,$ drift increases for both methods, but Naive’s emotion intensity declines sharply while SteerSpeech remains comparatively robust. At matched emotion intensity, ours reduces speaker drift by 12–67% versus Naive; WER falls 12–21% on seen/accented speech but rises 3% on unseen speech.

Component contribution. On a validation subset, adding $L _ { \mathrm { s p e a k e r } }$ to $L _ { \mathrm { e m o t i o n } }$ reduced speaker drift by 0.042, while $L _ { \mathrm { A S R } }$ reduced WER by 3.82 pp. Combining both provides a balanced operating point, highlighting the complementary roles of the loss components. Matrix vs. bias. We ablate the trained low-rank correction $U ( D \bar { v } _ { e } )$ and bias b in Eq. 2 at inference, without retraining and with matched steering norms. $\mathrm { A t } \ \alpha \ = \ 0 . 7 $ , unseen-speaker anger accuracy is 92.5% for the full transform, 85.0% with only $U ( D \bar { v } _ { e } )$ , and 80.0% with only b; on seen speakers, either component nearly matches the full transform, indicating overlapping, split-dependent benefits.

## 5. CONCLUSION

In this work, we introduce SteerSpeech, a lightweight framework for continuous, monotonic emotion control with robust generalization to unseen and accented speakers. Unlike prior steering methods that inject a fixed or unsupervised direction, SteerSpeech jointly trains a low-rank transform with emotion, speaker, and transcription experts so that emotion intensity increases monotonically with steering strength while speaker and content drift stay constrained. Our results show that this supervision enables reliable inference-time emotion control without modifying the TTS backbone. Future work will extend SteerSpeech to additional TTS backbones and emotions.

## 6. REFERENCES

[1] S. Inoue, K. Zhou, S. Wang, and H. Li, “Hierarchical emotion prediction and control in text-to-speech synthesis,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2024, pp. 10601–10605.

[2] X. Jing, K. Zhou, A. Triantafyllopoulos, and B. W. Schuller, “Enhancing emotional text-to-speech controllability with natural language guidance through contrastive learning and diffusion models,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2025, pp. 1–5.

[3] W. Chen, S. Yang, G. Li, and X. Wu, “DrawSpeech: Expressive speech synthesis using prosodic sketches as control conditions,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2025, pp. 1–5.

[4] N. Rimsky, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger, and A. Turner, “Steering Llama 2 via contrastive activation addition,” in Proc. 62nd Annu. Meeting Assoc. Comput. Linguistics (ACL), 2024, pp. 15504–15522.

[5] T. Xie, S. Yang, C. Li, D. Yu, and L. Liu, “EmoSteer-TTS: Fine-grained and training-free emotion-controllable text-to-speech via activation steering,” arXiv preprint arXiv:2508.03543, 2025.

[6] L. Zhou, H. Jiang, J. Li, T. Wang, and H. Li, “EmoShift: Lightweight activation steering for enhanced emotion-aware speech synthesis,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2026.

[7] Haozhe Chen, Run Chen, and Julia Hirschberg, “Emoknob: Enhance voice cloning with fine-grained emotion control,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 8170–8180.

[8] D. Tan et al., “Analysing the generalisation and reliability of steering vectors,” in Adv. Neural Inf. Process. Syst., 2024, vol. 37, pp. 139179–139212.

[9] M. Kang, W. Han, S. J. Hwang, and E. Yang, “ZET-Speech: Zero-shot adaptive emotion-controllable text-to-speech synthesis with diffusion and style-based models,” in Proc. Interspeech, 2023, pp. 4339–4343.

[10] Y. Wang et al., “Style tokens: Unsupervised style modeling, control and transfer in end-to-end speech synthesis,” in Proc. Int. Conf. Mach. Learn. (ICML), 2018, pp. 5180–5189.

[11] X. Gao, C. Zhang, Y. Chen, H. Zhang, and N. F. Chen, “Emo-DPO: Controllable emotional speech synthesis through direct preference optimization,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2025, pp. 1–5.

[12] Siyi Wang, Shihong Tan, Siyi Liu, Hong Jia, Gongping Huang, James Bailey, and Ting Dang, “Cocoemo: Composable and controllable human-like emotional TTS via activation steering,” in Forty-third International Conference on Machine Learning, 2026.

[13] P. Feng et al., “Task vector in TTS: Toward emotionally expressive dialectal speech synthesis,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2026, pp. 1–5.

[14] R. J. Williams and D. Zipser, “A learning algorithm for continually running fully recurrent neural networks,” Neural Comput., vol. 1, no. 2, pp. 270–280, 1989.

[15] Y. Bengio, N. Leonard, and A. Courville, “Estimating or prop- ´ agating gradients through stochastic neurons for conditional computation,” arXiv preprint arXiv:1308.3432, 2013.

[16] K. Zhou, B. Sisman, R. Liu, and H. Li, “Seen and unseen emotional style transfer for voice conversion with a new emotional speech dataset,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2021, pp. 920–924.

[17] G. Zhao et al., “L2-ARCTIC: A non-native english speech corpus,” in Proc. Interspeech, 2018.

[18] H. Hu et al., “Qwen3-TTS technical report,” arXiv preprint arXiv:2601.15621, 2026.

[19] Z. Ma et al., “emotion2vec: Self-supervised pre-training for speech emotion representation,” in Findings Assoc. Comput. Linguistics: ACL, 2024, pp. 15747–15760.

[20] S. Chen et al., “WavLM: Large-scale self-supervised pretraining for full stack speech processing,” IEEE J. Sel. Topics Signal Process., vol. 16, no. 6, pp. 1505–1518, 2022.

[21] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. 40th Int. Conf. Mach. Learn. (ICML), 2023, vol. 202, pp. 28492–28518.

[22] A. Conneau, A. Baevski, R. Collobert, A. Mohamed, and M. Auli, “Unsupervised cross-lingual representation learning for speech recognition,” in Proc. Interspeech, 2021, pp. 2426– 2430.

[23] B. Desplanques, J. Thienpondt, and K. Demuynck, “ECAPA-TDNN: Emphasized channel attention, propagation and aggregation in TDNN-based speaker verification,” in Proc. Interspeech, 2020, pp. 3830–3834.

[24] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Adv. Neural Inf. Process. Syst., 2020, vol. 33, pp. 12449–12460.

[25] Takaaki Saeki, Detai Xin, Wataru Nakata, Tomoki Koriyama, Shinnosuke Takamichi, and Hiroshi Saruwatari, “UTMOS: UTokyo-SaruLab system for VoiceMOS challenge 2022,” in Proc. Interspeech 2022, 2022, pp. 4521–4525.

[26] G. Zhang et al., “iEmoTTS: Toward robust cross-speaker emotion transfer and control for speech synthesis based on disentanglement between prosody and timbre,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 31, pp. 1693–1705, 2023.

[27] Hongfei Du, Jiacheng Shi, Sidi Lu, Gang Zhou, and Ashley Gao, “Sparse autoencoders for interpretable emotion control in text-to-speech,” in Forty-third International Conference on Machine Learning, 2026.