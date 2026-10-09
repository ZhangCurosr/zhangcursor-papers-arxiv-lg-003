# Self-Supervised Speech Representations for Cross-Speaker Dysarthria Detection During Awake Craniotomy

Chinmayi Kanthila, Nassib Abdallah, Harrison Misy, Celine Panheleux, Vanessa Saliou, Romuald Seizeur, and Guillaume Dardenne

Abstract—Detecting intra-operative speech impairment during awake craniotomy is essential for preserving language function. However, automated detection remains challenging because operating-room recordings contain substantial acoustic interference, clinically relevant speech events are rare, and available cohorts are small and heterogeneous across speakers. This study presents a systematic component-wise evaluation of a pipeline for distinguishing dysarthric from no-trouble speech in the DATABRASE corpus of awake-craniotomy recordings. The pipeline incorporates speaker diarization to isolate patient speech, a multi-view representation combining handcrafted acoustic descriptors with multilayer wav2vec 2.0 embeddings, speaker-conditional normalization and transferabilitybased feature selection to improve cross-speaker robustness, and a cascaded classifier comprising a gradient-boosted first stage and a neural second stage. Evaluation was conducted under strict speaker-independent conditions using leave-one-speakerout cross-validation. The results show that cross-speaker performance is influenced more strongly by the speech representation than by classifier choice. The AUCs of three classifiers differed by no more than 4.7%, whereas replacing conventional acoustic descriptors with the multilayer self-supervised representation produced AUC improvements of 18.2%–26.1%. Diarizationconditioned feature extraction and the proposed classifier cascade provided additional consistent gains. These findings indicate that reliable patient-specific speech isolation and strong pretrained representations are more important than increased classifier complexity in low-resource intra-operative settings. They also quantify the potential performance gains that may be achieved through patient-specific preoperative calibration.

Index Terms—Awake craniotomy, dysarthria detection, selfsupervised speech representations, speaker diarization, crossspeaker generalization, intraoperative monitoring.

## I. INTRODUCTION

Awake craniotomy (AC) is the standard surgical approach for resecting tumors growing within or near eloquent cortex, where the surgeon must balance maximal resection against the preservation of motor functions including speech. Lesion size and proximity to cortical or subcortical language networks both raise the risk of postoperative cognitive decline, with direct consequences for functional independence and quality of life [1]. Intra-operative monitoring is mainly dependent on direct electrical stimulation (DES) combined with active patient participation in speech and language tasks. DES can also elicit transient speech disturbances such as speech arrest, anomia, dysarthria, semantic and phonemic paraphasia, and comprehension failures [2]. Their clinical significance varies, as a transient dysarthric episode may reflect disruption of a distributed motor-speech network whose long-term consequences cannot be inferred from a namingtask score alone.

Detecting these events in real time still depends entirely on the surgical team. The neurosurgeon, neuropsychologist, and speech-language pathologist must run the language tests and assess the speech output simultaneously. The evaluation is subjective, has limited temporal resolution, and is prone to fatigue over a long procedure [3], [4].

In this context, the machine-learning framework offers the possibility of quantifying subtle variations in speech directly from intra-operative recordings, classifying emerging impairment patterns, and providing the surgical team with objective, decision-relevant feedback during resection. Rather than replacing expert clinical judgment, such a system could function as an assistive monitoring layer that enhances the sensitivity, consistency, and interpretability of intra-operative language assessment. Nevertheless, the intraoperative environment introduces a distinct set of constraints that conventional automatic speech recognition (ASR) frameworks do not adequately address. These include substantial acoustic noise from the operating-room environment and the patient’s fluctuating neurological state, the absence of largescale labeled intra-operative corpora, and the requirement for robust real-time performance under clinically critical conditions [5]. In fact, each of these constraints arises from a different stage of the processing chain. Noise and multispeaker content relate to data quality, corpus scarcity relates to feature extraction and augmentation, and cross-patient robustness relates to classifier choice and evaluation.

The main contributions of this work are as follows:

• A pathology-preserving pitch augmentation using a HiFi-GAN vocoder that improves prosodic diversity while retaining dysarthric cues.

• Patient-only feature extraction, using diarization to compute contamination-sensitive descriptors from patient frames alone and thereby remove clinician-speech confounds.

• Two cascaded classifiers coupling a gradient-boosted first stage with a neural second stage through leakagefree inner cross-validation.

## II. BACKGROUND AND RELATED WORK

Automated detection of dysarthric speech has developed almost entirely on chronic motor-speech disorders recorded under controlled conditions. Existing literature mainly focuses on what is methodologically achievable, including which acoustic properties carry the impairment, which representations expose them, and which evaluation protocols provide an honest estimate of performance on a new patient. However, there is a lack of studies that describe the conditions of an AC, where impairment is transient and stimulation-evoked, the recording contains clinicians as well as the patient, and the cohort is small and severely imbalanced.

## A. Automated dysarthria detection in general conditions

Dysarthria is a group of motor-speech disorders arising from impaired control of the articulators. Most automated work uses the UA-Speech or TORGO corpora [6], [7], which contain speakers with chronic dysarthria of graded severity recorded under controlled conditions.

Early systems described these recordings with interpretable acoustic descriptors. Prabhakera and Alku [8] classified dysarthric speech using glottal-source parameters with a support vector machine, reporting 94.29% under leave-onespeaker-out (LOSO) evaluation, and showing that sourcelevel measures alone carry substantial diagnostic information. Prosodic and voice-quality descriptors have been used in the same way, selected for their correspondence to identifiable production mechanisms [9], [10]. Millet and Zeghidour [11] instead learned the front end directly from the waveform, obtaining roughly a 10% gain over fixed features and demonstrating that hand-designed descriptors do not exhaust the available signal, though at the cost of clinical interpretability.

Self-supervised encoders have since become the dominant approach for speech feature extraction. Javanmardi et al. [12] compared wav2vec 2.0 (W2V2) and HuBERT embeddings against conventional descriptors and found the pretrained representations consistently superior, while noting that the evaluation corpora remain small and English-only. Pasad et al. [13] showed by layer-wise probing that such encoders distribute information with depth, phonetic detail appearing in middle layers and speaker- and task-conditional information towards the output. This property is consequential for detection because the layers that encode intelligibility also encode speaker identity, so that a discriminative embedding may separate speakers rather than conditions. Kadirvelu et al. [14] addressed this directly. They combined a W2V2 encoder with a multi-task objective intended to group embeddings by severity rather than by speaker, and reported

70.5% accuracy and 59.2% F1 under LOSO evaluation. In this work, speaker invariance improved, but variability in severity remained unresolved.

Classifier design has followed a comparable progression, from classical statistical models to recurrent and convolutional architectures and, more recently, to transformer-based transfer learning [15]–[18]. A separate line combines architectures rather than selecting among them, on the reasoning that one inductive bias suits a heterogeneous representation poorly. Shahin et al. [19] placed a Gaussian mixture stage before a deep network for speaker identification under varying emotion, and the two-stage cascade outperformed either model used alone. A later study by the same group, comparing different shallow-deep pairings, found that the benefit came from coupling the two stages rather than from the particular models chosen [20]. Ensemble formulations have recently been applied to dysarthria detection on the same principle [21]. In each case, the models are combined, but the representation they operate on is the same for both stages.

Across this literature, the evaluation protocol determines the reported result as much as the architecture. Tripathi et al. [22] report 97.4% under speaker-dependent evaluation against 53.9% under speaker-independent evaluation on UA-Speech, and Joshy and Rajan [15] report 93.97% against 49.22% with a single architecture across UA-Speech and TORGO. Manoj et al. [17] report 99.0–99.4% with transformer transfer learning, but without speaker-independent or external validation. Schu et al. [23] showed that part of the performance reported on these corpora is attributable to nonspeech and database-specific cues rather than to the impairment, and Roy et al. [24] found that accuracy under crosscorpus evaluation recovers only after adaptation to the target domain. Table I summarizes these studies. These results are obtained on recordings that contain a single speaker, a stable impairment, and material elicited under controlled conditions.

## B. Speech impairment detection during awake craniotomy

Speech assessment during AC has been addressed primarily as functional mapping rather than signal-level classification [1], [2], and only two studies have approached the classification problem directly. Nishimura et al. [25] trained support vector and relevance vector machines on MFCC features from five awake procedures to detect speech arrest, establishing feasibility but without speaker-independent evaluation. Maoudj et al. [26] extended this work with a W2V2 classifier on 1,883 three-second clips from 23 procedures, including a cross-lingual Japanese–French evaluation. Indomain accuracy was strong but degraded under cross-lingual transfer, indicating that the model had learned regularities specific to the training condition rather than the impairment itself.

Both studies formulate the task as abnormal versus normal speech, grouping heterogeneous deficits such as dysarthria, anomia, paraphasia, and perseveration [27]. A correct prediction therefore indicates only that speech has become abnormal, without identifying which deficit occurred, although these impairments reflect different functional networks and are distinguished clinically during stimulation. The present work accordingly restricts the target to dysarthria, so that the predicted label corresponds to a single motor-speech phenomenon rather than to a heterogeneous abnormality class.

TABLE I  
RECENT DYSARTHRIC-SPEECH CLASSIFICATION STUDIES. SI: SPEAKER-INDEPENDENT; LOSO: LEAVE-ONE-SUBJECT-OUT; SSL: SELF-SUPERVISED LEARNING.
<table><tr><td>Ref.</td><td>Method</td><td>Result</td><td>Comment</td></tr><tr><td>[22]</td><td>DeepSpeech + SVM</td><td>53.9% SI; 97.4% SD</td><td>Large SD-SI gap indicates strong speaker dependence; SD results may overestimate clinical generalisation.</td></tr><tr><td>[15]</td><td></td><td>GRU/ LSTM49.22%SI</td><td>DNN/ CNN/ 93.97% SD; Useful baseline study, but the same model drops sharply under SI testing, highlighting poor unseen-speaker transfer.</td></tr><tr><td>[14]</td><td>SSL transformer + MTL</td><td>59.2%F1</td><td>70.5% Acc.; LOSO evaluation is stronger than SD testing; results show improved speaker invariance but severity variability remains difficult.</td></tr><tr><td>[8]</td><td>Glottal + SVM</td><td>94.29% LOSO</td><td>Interpretable source-based features support clinical analysis; however, performance should be interpreted with dataset/task constraints.</td></tr><tr><td>[17]</td><td>Transformer transfer learning</td><td>SD</td><td>99.0–99.4% Near-ceiling SD accuracy is likely optimistic; without SI/LOSO or external validation, generalisability is limited.</td></tr><tr><td>[12]</td><td>W2V2, HuBERT, handcrafted</td><td>SSL &gt; baselines</td><td>Strong comparison of SSL and handcrafted features; confirms benefit of pretrained embeddings, but datasets remain small and English-only.</td></tr><tr><td>[24]</td><td>DSSCNet; cross-corpus</td><td>56.8- 64.2%; 77.8–79.4%</td><td>The cross-corpus setting is deployment-relevant; the improvement after tuning suggests that domain mismatch</td></tr><tr><td>[11]</td><td>Raw-speech end-to-end</td><td>tuned ~10% gain</td><td>remains a major limitation. Early end-to-end evidence; learned front-ends improve over fixed features, but clinical interpretability is weaker.</td></tr><tr><td>[23]</td><td>UA/TG validation analysis</td><td></td><td>Artefact risk Critical control study; shows that non-speech or speaker/database cues can inflate reported performance on UA and TG.</td></tr></table>

The intra-operative setting also breaks assumptions that the corpus literature relies on. Recordings are made in an active operating room, so the patient’s voice is accompanied by clinician speech, equipment noise, and silence. Clinician speech is not incidental, since the examiner prompts, repeats, and follows up in response to the patient, and its presence is therefore correlated with the condition being detected. Isolating the patient is a diarization problem, and diarization is an established preliminary step in clinical speech analysis, used for cognitive assessment recordings before any diagnostic model is applied [28]. Its reliability is not assured for this population, as general-purpose systems degrade on conversations involving speakers with speech disorders [29], and the labels it produces are anonymous, so identifying which track belongs to the patient requires a further decision [30]. In each of these applications the segmentation is retained, and the description of the exchange that produced it is discarded.

Cohort size is a second constraint. AC procedure has few patients, impaired speech occupies a small fraction of each recording, and the resulting class imbalance is severe. Augmentation is the standard response to scarcity, but it is constrained here in a way it is not for ordinary speech. The label of a clip is defined by the acoustic consequences of impairment, so any transformation that improves class balance while altering those consequences removes the property the classifier is meant to detect. Voice-conversion methods, which modify speaker characteristics directly, are unsuitable for this reason. What is required is not more data but data whose label is preserved by construction.

TABLE II  
SPEAKER-WISE DISTRIBUTION OF THE DATA.
<table><tr><td>Speaker ID</td><td>Total clips</td><td>DYS</td><td>NT</td><td>DYS %</td></tr><tr><td>Patient 1</td><td>796</td><td>38</td><td>758</td><td>4.8</td></tr><tr><td>Patient 2</td><td>355</td><td>31</td><td>324</td><td>8.7</td></tr><tr><td>Patient 3</td><td>112</td><td>11</td><td>101</td><td>9.8</td></tr><tr><td>Patient 4</td><td>254</td><td>39</td><td>215</td><td>15.4</td></tr><tr><td>Patient 5</td><td>173</td><td>28</td><td>145</td><td>16.2</td></tr><tr><td>Patient 6</td><td>35</td><td>6</td><td>29</td><td>17.1</td></tr><tr><td>Patient 7</td><td>607</td><td>117</td><td>490</td><td>19.3</td></tr><tr><td>Patient 8</td><td>665</td><td>142</td><td>523</td><td>21.4</td></tr><tr><td>Patient 9</td><td>244</td><td>78</td><td>166</td><td>32.0</td></tr><tr><td>Patient 10</td><td>773</td><td>265</td><td>508</td><td>34.3</td></tr><tr><td>Patient 11</td><td>267</td><td>130</td><td>137</td><td>48.7</td></tr><tr><td>Total</td><td>4,281</td><td>885</td><td>3,396</td><td>20.7</td></tr></table>

These constraints explain why reliable automated dysarthria classification during AC remains largely unexplored. The two prior studies showed that the detection is feasible. However, gaps still exist in the literature concerning the recording conditions, the size of the available cohorts, and the design of the classifier. The recording contains clinician speech that is correlated with the label and must be separated from the patient’s own. The cohort is too small and too imbalanced to train on directly, and the augmentation required to address this must preserve the acoustic evidence of impairment. The resulting representation combines descriptors of different origin and reliability, which no single classifier is well suited to exploit.

## III. MATERIALS AND METHODS

## A. Dataset

The French dataset used for dysarthric classification, DATABRASE, was collected during AC and comprises intraoperative speech recordings from 23 patients undergoing AC at the University Hospital (CHU) of Brest, France. For each procedure, the continuous recording was segmented into 3-second clips. This duration was chosen to capture the patient’s response to a single DES event together with any associated speech disturbance, short enough to localize transient impairment, yet long enough to support stable acoustic feature extraction. Although intra-operative DES can elicit a range of language disturbances, including anomia, paraphasia, and perseveration, the present work restricts the classification target to DYS versus NT speech.

Each clip was annotated independently by the neurosurgeon using Audacity and subsequently reviewed by two additional speech experts. Patients with no DYS segments were removed, since they provided no positive samples. These patient-level exclusion criteria were then applied to support robust speaker-independent modeling. The retained cohort comprised speakers whose dysarthric proportions ranged from approximately 5% to 49%, with the lowest-proportion retained speaker additionally excluded from within-fold validation use, since the near-absent positive class could not provide a meaningful signal for threshold or model selection. The resulting cohort, summarized in Table II, comprised 11 speakers and 4,281 raw 3-second clips.

The classification framework comprises six stages, from raw-audio conditioning to speaker-grouped evaluation. HiFi-GAN pitch augmentation expands the training folds with label-preserving synthetic clips. These clips add prosodic variability while retaining dysarthric cues. Speaker diarization then separates patient speech from clinician speech. It isolates the frames on which contamination-sensitive features are measured, and yields conversational descriptors of the clinician–patient exchange. A quality filter discards clips with insufficient patient occupancy. The retained clips are described by complementary feature families: conventional acoustic descriptors, conversational features from diarization, and multi-layer self-supervised embeddings. These features are refined by per-fold preprocessing. Each patient is normalized against their own baseline, and only features that transfer across speakers are retained. Classification uses three singlestage baselines and two proposed cascades, each coupling a gradient-boosted stage with a neural one. All stages beyond feature extraction run within each cross-validation fold. Heldout speakers therefore influence neither the preprocessing statistics nor the model hyperparameters. Each stage is introduced cumulatively, so its individual effect can be isolated.

## B. Data augmentation

Dysarthric clips are the minority class throughout the cohort. Across the retained speakers, 885 of 4,281 clips are dysarthric, a ratio of 3.8:1 (Table II). The imbalance is also uneven across patients, ranging from 1.1:1 to nearly 20:1. Training on these clips alone therefore risks a classifier biased toward the majority class and fitted to the few speakers that contribute most positive samples. To mitigate this, the training folds are expanded by pitch augmentation using the HiFi-GAN neural vocoder [31]. Unlike conversion-based approaches such as CycleGAN and StarGAN-VC, which can distort pathology-bearing cues, HiFi-GAN resynthesizes along a single perceptual axis [32]. This retains lexical content, response timing, and the dysarthric acoustic structure the classifier must learn. The V2 variant is used, fine-tuned on the training split.

Given an input mel-spectrogram s, a controlled pitch shift in the conditioning domain produces a modified representation $s ^ { \prime } .$ The waveform is then resynthesized as G(s<sup>′</sup>). Pitch offsets are drawn uniformly from ±2 semitones. This range stays clinically plausible for AC while remaining labelpreserving [33]. Three pitch-shifted variants are generated per source clip. Each variant inherits the speaker identity of its source and is bound to the same fold under the GroupKFold partition, which prevents train–test leakage. Augmentation is applied to the raw clips, so both the original and augmented clips then pass through diarization and feature extraction.

## C. Diarization and feature extraction

1) Speaker separation and clip selection: Operating-room audio rarely contains the patient’s voice in isolation. Clinician speech is present in most clips, and its share of each recording varies with the clinical situation. This poses two distinct problems. First, dysarthria is a disorder of motor speech production, so its acoustic markers describe the patient’s articulatory and phonatory behavior. Measured over a mixture of speakers, descriptors such as jitter, shimmer, formant position, and glottal-source behavior no longer characterize the patient at all. Second, a classifier trained on unsegmented audio may separate the classes using recording context rather than patient speech.

Speaker diarization addresses both. Each three-second clip is processed by pyannote.audio [34], which yields the full waveform, the frames attributed to the patient, and turnlevel descriptors of the clinician–patient exchange. These outputs are used in two ways. The patient frames define the region on which contamination-sensitive features are computed, so that each descriptor is measured where it is physiologically meaningful. The turn-level descriptors are retained as features in their own right, since the clinician’s conduct during AC is reactive. The prompts are repeated, rephrased, or followed up when a response appears impaired, so response latency and follow-up behavior carry information about the patient’s state. Diarization therefore serves both as a segmentation step preceding feature extraction and as a source of features, rather than as a preprocessing operation whose output is discarded.

The diarization output also supports a clip-level quality criterion. The attribute patient\_pct gives the proportion of clip duration attributed to the patient. Clips with patient\_pct < 20% were excluded, since at low patient occupancy the acoustic content is dominated by clinician speech and pathology-relevant cues cannot be measured reliably. The criterion retained 2,706 of 4,281 clips (63.2%), with a post-filter dysarthric prevalence of 22.1% (597 DYS, 2,109 NT). The excluded 36.8% were held out for sensitivity analysis and were not used for training.

2) Feature representation: Each clip is represented by three groups of features, organised by the region on which they can be measured validly.

The first group covers spectral and temporal properties of the clip as a whole. These include cepstral coefficients, spectral-shape descriptors, mel-band statistics, energy and rhythm measures, and pause statistics. Together they characterize vocal-tract shaping, loudness, and speech timing, all commonly altered in dysarthria [35]. Cepstral coefficients are the standard baseline in dysarthric speech analysis and serve that role here [5], [12], [15], [36]. These descriptors remain interpretable under brief non-patient speech, since they summarize the acoustic scene rather than one talker’s production, and are therefore computed on the full clip (Table III).

The second group covers source- and articulator-level properties, which describe the patient’s production directly and are consequently computed on patient frames only.

![](images/0e4eff2b48a66e2e44354f17799fdc76eddc905a543f207026d478c87095dc3a.jpg)  
Fig. 1. Overall classification framework, comprising the data quality, feature extraction, augmentation, and classification stages.

![](images/ab0fe731f7b54d7d0f76cf57fa94c8e35719f40be4f1650c44eee25886e285bc.jpg)  
Fig. 2. End-to-end data preprocessing and feature-extraction framework.

It comprises prosodic measures of fundamental frequency and intensity, voice-quality measures of jitter, shimmer, and harmonics-to-noise ratio [10], [15], formant positions and vowel-space area, and glottal-source descriptors obtained from inverse-filtered glottal flow with closure instants supplied by the speech processing tool, REAPER. This group also includes the self-supervised representation described below, which is likewise sensitive to who is speaking (Table III).

The third group is derived from the diarization output rather than from the audio. It encodes the dynamics of the intra-operative exchange: response latency between clinician prompt and patient response, clinician follow-up rate, turn-

FEATURE INVENTORY, GROUPED BY THE REGION ON WHICH EACH DESCRIPTOR IS COMPUTED: THE FULL 3-S CLIP, THE PATIENT FRAMES, OR THE DIARIZATION OUTPUT.

<table><tr><td>Descriptor</td><td></td><td>Dim. Details</td></tr><tr><td>Full-clip descriptors</td><td></td><td></td></tr><tr><td>MFCC (+∆, ∆∆)</td><td></td><td>65 Cepstral baseline</td></tr><tr><td>Spectral shape</td><td></td><td>30 Centroid, roll-off, bandwidth</td></tr><tr><td>Mel-spectrogram</td><td></td><td>23 Mel-band statistics</td></tr><tr><td>Temporal (3 s)</td><td></td><td>24 RMS, ZCR, duration</td></tr><tr><td>Spectral flux</td><td></td><td>6 Frame-to-frame change</td></tr><tr><td>Pause</td><td></td><td>9 Pause and rhythm statistics</td></tr><tr><td>Patient-frame descriptors</td><td></td><td></td></tr><tr><td>Prosody</td><td></td><td>18 F0, intensity contour</td></tr><tr><td>Voice quality</td><td></td><td>12 Jitter, shimmer, HNR</td></tr><tr><td>Formants</td><td></td><td>22 F1–F3, vowel space</td></tr><tr><td>Glottal</td><td>26</td><td>GNE, NAQ, QOQ, CPPS</td></tr><tr><td>W2V2 (FR) layer 6</td><td>1,024</td><td>Phonetic detail</td></tr><tr><td>W2V2 (FR) layer 9</td><td>1,024</td><td>Phonological structure</td></tr><tr><td>W2V2 (FR) final layer</td><td>1,024</td><td>Intelligibility-related</td></tr><tr><td>Conversational descriptors</td><td></td><td></td></tr><tr><td>Diarization</td><td>13</td><td>Turn-taking, latency, overlap</td></tr></table>

taking statistics, and overlap proportions (Table III). As noted above, these quantities reflect the clinician’s real-time assessment of the patient’s response, and they are available only because diarization is part of the framework.

3) Self-supervised representation: Conventional descrip tors encode properties chosen in advance, and they capture dysarthric production only to the extent that those choices are correct. A self-supervised encoder instead learns speech structure from large external corpora, and can represent regularities that no explicit descriptor makes available. A frozen French-pretrained W2V2 encoder [37], [38] is therefore applied to the patient frames of each clip.

Depth determines what such an encoder represents. Layerwise probing shows that phonetic detail emerges in midnetwork layers, phonological and lexical structure in deeper layers, and task- and speaker-conditional information in the final layers [13]. Dysarthria disturbs articulatory precision, phonological realisation, and intelligibility together, so no single layer carries the full profile. Following the aggregation strategy of [39], layers 6, 9, and the final layer are retained. Frame-level activations of each retained layer ℓ are meanpooled over time to $\mathbf { z } _ { \ell } \in \mathbb { R } ^ { 1 0 2 4 }$ , and the three are concatenated,

$$
\begin{array} { r } { { \bf z } _ { \mathrm { S S L } } = { \bf z } _ { 6 } \left\| { \bf z } _ { 9 } \right\| { \bf z } _ { \mathrm { f i n a l } } , } \end{array}\tag{1}
$$

giving the complete representation $\mathbf { x } = \mathbf { x } _ { \mathrm { H C } } \parallel \mathbf { z } _ { \mathrm { S S L } }$

The encoder is kept frozen. With 597 dysarthric clips from eleven speakers, fine-tuning a 95M-parameter encoder would risk fitting speaker identity rather than dysarthric structure. The same sensitivity motivates the further preprocessing since the embedding is discriminative but encodes who is speaking alongside how they are speaking.

4) Per-fold preprocessing: Both feature groups carry speaker identity alongside dysarthric structure. Three operations reduce this before classification, all estimated within the training fold only. Speaker-conditional centering expresses each conventional descriptor relative to the patient’s own baseline. For each training-fold speaker s, a per-feature mean $\mu _ { s }$ is estimated over that speaker’s NT clips and subtracted,

$$
\begin{array} { r } { \tilde { x } _ { \mathrm { H C } , i } = x _ { \mathrm { H C } , i } - \mu _ { s ( i ) } . } \end{array}\tag{2}
$$

This follows the clinical logic of the task. Impairment is judged against how a given patient normally speaks, not against a population mean, and AC provides the pre-operative recording such a baseline requires. The transferability filter then removes descriptors that do not generalize across patients. A feature is retained only if the sign of its withinspeaker Spearman correlation with the label agrees across at least 80% of training speakers,

$$
S _ { j } = \frac { 1 } { | S _ { \mathrm { t r a i n } } | } \sum _ { s \in S _ { \mathrm { t r a i n } } } \mathcal { H } [ \mathrm { s i g n } ( \rho _ { j , s } ) = \mathrm { s i g n } ( \bar { \rho } _ { j } ) ] , \quad S _ { j } \geq 0 . 8 .\tag{3}
$$

A descriptor that rises with impairment in one patient and falls in another is uninformative for an unseen patient, however well it separates the classes within a speaker. The criterion typically retains 80 to 150 of the 251 conventional descriptors. Centring and the transferability filter are applied to the conventional descriptors only. Both require a perspeaker estimate over NT clips, and the self-supervised embedding is high-dimensional and opaque enough that per-feature sign consistency is not a meaningful criterion. The embedding is instead standardized with the rest of the representation. Standardization rescales every feature to zero mean and unit variance using training-fold statistics, which are then applied unchanged to the validation and test folds.

All three operations are estimated inside the crossvalidation loop. No statistic is computed from a held-out speaker, so the evaluation reflects performance on a patient the framework has not seen.

## D. Classification models

Three single-stage classifiers of distinct inductive bias are evaluated, together with two proposed cascades. The representation defined above is high-dimensional and heterogeneous. It mixes frequency-domain descriptors, temporaldomain measures, conversational descriptors, and a large selfsupervised block, in which many dimensions are weakly informative. No single inductive bias suits all of it. Gradientboosted trees are robust to uninformative dimensions but partition the space along individual axes. Neural models represent smooth dependencies among features, but are sensitive to noise when the sample is small. The cascades are designed to combine these behaviors rather than choose between them. All models are trained and evaluated under identical speakergrouped protocols, so that architecture is the only variable.

1) XGBoost baseline: XGBoost [40] serves as a baseline and as the first stage of both cascades. It is trained with the reg:logistic objective, with the positive-class weight set from the training-fold class ratio. Its splits are invariant to the differing numeric ranges of the conventional and selfsupervised blocks, and its directional default splits tolerate sparse inputs. Boosted trees also remain competitive with deep architectures on small tabular problems [41], which matches the regime here. Hyperparameters (max\_depth, eta, reg\_lambda, subsample, colsample\_bytree, min\_child\_weight) were selected by Bayesian search with a Tree-structured Parzen Estimator sampler [42]. The search ran independently within each fold on the validation speaker only, so the test speaker never influenced selection. Thirty trials were used per fold, since pilot runs showed a clear AUC gain from 5 to 15 trials and below 0.005 from 15 to 50. Each trial allowed up to 200 boosting rounds with early stopping on validation AUC.

2) Multilayer perceptron: The MLP provides a nonrecurrent neural counterpart to the tree baseline. It operates on the full concatenated vector and represents smooth interactions among features that axis-aligned splits cannot express. This capacity carries a cost: with few dysarthric clips, the network is more exposed to noisy dimensions than the tree. Each hidden layer is followed by batch normalisation, ReLU, and dropout, and the width is scaled to the input dimensionality. The output head is trained with focal binary cross-entropy,

$$
\mathcal { L } _ { \mathrm { f o c a l } } ( p , y ) = - \alpha _ { y } ( 1 - p _ { t } ) ^ { \gamma } \log p _ { t } , \quad p _ { t } = \left\{ \begin{array} { l l } { p } & { y = 1 } \\ { 1 - p } & { y = 0 , } \end{array} \right.\tag{4}
$$

with $\gamma = 2 . 0 , \alpha _ { 1 }$ derived from the training-fold class ratio, and $\alpha _ { 0 } = 1 [ 4 3 ]$ . The focal term down-weights easy majorityclass clips, which is appropriate given the imbalance reported in Table II. Hyperparameters were tuned under the same protocol as the other models (Table IV), with Adam and early stopping on validation AUC.

3) Bidirectional GRU: The BiGRU is the recurrent baseline. The other two models consume clip-level summaries, in which the order of events within a clip has already been pooled away. A dysarthric episode is a transient event with onset and release. Each three-second clip is therefore divided into overlapping sub-segments, and the resulting sequence is passed through a two-layer bidirectional encoder, so that within-clip temporal structure remains available to the model. Sequence representations are mean-pooled and projected through a linear head with sigmoid activation. Training uses the same focal objective as the MLP, with the positive-class weight set from the training-fold class ratio, Adam optimization, and early stopping on validation AUC.

4) Cascade-MLP: The first cascade places an XGBoost stage before an MLP stage. The two models fail in different ways on this representation. XGBoost is robust to the many weakly informative dimensions of the self-supervised block, but cannot express smooth dependencies among features. The MLP can express them, but must locate them in a 3,323- dimensional space from few positive examples, and tends to fit speaker identity instead. The cascade removes part of that burden. The tree reduces the representation to a single calibrated estimate of dysarthria, and the neural stage receives that estimate alongside the original features,

$$
\mathbf { x } _ { \mathrm { c a s } } = \mathbf { x } \parallel p _ { \mathrm { X G B } } ,\tag{5}
$$

where $p _ { \mathrm { X G B } } ~ = ~ f _ { \mathrm { X G B } } ( \mathbf { x } )$ and $p _ { \mathrm { c a s } } ~ = ~ f _ { \mathrm { M L P } } ( \mathbf { x } _ { \mathrm { c a s } } )$ . The second stage is therefore not required to rediscover the decision boundary from the raw input, and can instead refine a summary that is already stable across speakers.

Two design choices keep the construction valid under speaker-independent evaluation. First, each training clip’s p<sub>XGB</sub> is produced by an inner LOSO procedure within the training fold. A tree evaluated on its own training clips would return overconfident probabilities, and the second stage would learn to trust an input whose character it never sees again at test time. The inner procedure makes trainingtime and test-time probabilities statistically comparable. Second, an auxiliary regression head $g ( \mathbf { h } _ { 2 } ) \in \mathbb { R } ^ { 4 }$ on the second hidden layer is trained jointly with the classification head,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c a s } } = \mathcal { L } _ { \mathrm { f o c a l } } ( p _ { \mathrm { c a s } } , y ) + \alpha \mathcal { L } _ { \mathrm { M S E } } ( g ( \mathbf { h } _ { 2 } ) , \mathbf { c } ) , } \end{array}\tag{6}
$$

with $\mathbf { c } = ( 0 . 5 , 0 . 5 , 0 . 5 , 0 . 5 ) ^ { \top }$ a soft uniform prior and $\alpha =$ 0.1. The auxiliary term discourages the hidden layer from collapsing onto sharp class indicators, which in a small cohort are often speaker-specific.

5) Cascade-BiGRU: The second cascade replaces the feed-forward second stage with a recurrent one, so that the first-stage estimate is combined with a sequence model rather than a static classifier. The first stage and its inner LOSO construction are unchanged.

Each session k is represented by an ordered sequence of segment feature vectors $\mathbf { X } ^ { ( k ) } = ( \mathbf { x } _ { 1 } ^ { ( k ) } , \ldots , \mathbf { x } _ { T _ { k } } ^ { ( k ) } )$ , where $T _ { k }$ is the number of retained segments. The bidirectional encoder maps this sequence to hidden states, which are mean-pooled into a single session representation,

$$
\mathbf { h } ^ { ( k ) } = \frac { 1 } { T _ { k } } \sum _ { t = 1 } ^ { T _ { k } } \mathrm { B i G R U } \Big ( \mathbf { X } ^ { ( k ) } \Big ) _ { t } .\tag{7}
$$

The first-stage estimate enters as a session-level summary, obtained by averaging the segment probabilities of that session,

$$
\bar { p } _ { \mathrm { X G B } } ^ { ( k ) } = \frac { 1 } { T _ { k } } \sum _ { t = 1 } ^ { T _ { k } } f _ { \mathrm { X G B } } \Big ( \mathbf { x } _ { t } ^ { ( k ) } \Big ) .\tag{8}
$$

TABLE IV  
HYPERPARAMETER SEARCH SPACES AND TUNING SETTINGS.
<table><tr><td>Model</td><td>Hyperparameter</td><td>Search space</td></tr><tr><td>MLP / Cascade-MLP</td><td>Hidden layers Base hidden dim. Dropout Learning rate Weight decay Focal-loss γ Auxiliary weight α</td><td>{2, 3, 4} {64, 128, 256, 512} [0.15, 0.55] log-U  $[ 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 3 } ]$  log-U  $[ 1 0 ^ { - 6 } , 1 0 ^ { - 3 } ]$  [1.0, 3.0] [0.0, 0.3]</td></tr><tr><td>BiGRU / Cascade-BiGRU</td><td>Hidden dimension Recurrent layers Max sequence length Dropout Focal-loss γ Learning rate</td><td>{32, 64, 128, 256} {1,2} {10, 15, 20} segments [0.15,0.55] [1.0, 3.0] log-U  $[ 1 \dot { 0 } ^ { - 4 } , 5 \times 1 0 ^ { - 3 } ]$ </td></tr><tr><td colspan="3">Weight decay log-U  $[ 1 0 ^ { - 6 } , 1 0 ^ { - 3 } ]$  Batch size {16, 32, 64} for all neural models.</td></tr></table>

The pooled representation and the summary are concatenated at the classification head, which produces the session logit,

$$
p _ { \mathrm { c a s } } ^ { ( k ) } = \sigma \Big ( \mathbf { w } ^ { \top } [ \mathbf { h } ^ { ( k ) }  \bar { p } _ { \mathrm { X G B } } ^ { ( k ) } ] + b \Big ) .\tag{9}
$$

The placement of $\bar { p } _ { \mathrm { X G B } } ^ { ( k ) }$ is a deliberate choice. The alternative is to append the segment probability to every element of the input sequence, replacing $\mathbf { x } _ { t } ^ { ( k ) }$ by $\mathbf { \bar { \rho } } _ { [ \mathbf { X } _ { t } ^ { ( k ) } | | p _ { t } ^ { ( \bar { k } ) } ] }$ in (7). Both placements were implemented, and the variant of (9) is retained. A scalar repeated across all timesteps is easily exploited by a recurrent encoder and would compete with the temporal structure the encoder exists to model. Admitting the estimate only at the decision boundary keeps the encoder operating on the feature stream, and confines the first-stage contribution to the point where the two sources are combined.

The head is trained with the same focal objective as the single-stage BiGRU, with the positive-class weight set from the training-fold class ratio. Tuning used the same inner speaker-independent protocol, with 30 trials per fold under three-fold inner cross-validation. The retained configuration was evaluated over five random seeds, with predicted probabilities averaged before thresholding to match the Cascade-MLP.

## E. Evaluation protocol and metrics

Performance is reported with four metrics. AUC is the primary metric. It is threshold-independent and measures ranking quality across all operating points, which suits a task with marked class imbalance and a speaker-adaptive decision threshold. The F1-score, balanced accuracy (BAcc), and overall accuracy (Acc) are reported as secondary, thresholddependent metrics, computed at that operating point. Because dysarthric clips are the minority class, accuracy alone is unreliable, and BAcc and F1 are included to reflect performance on the positive class.

Three evaluation protocols are used, and they answer different questions. Speaker-independent (SID) evaluation measures generalization to unseen patients and is reported in two variants that differ in how many speakers are held out at a time. The first, referred to as 5-fold or SID, is the primary protocol. A 5-fold GroupKFold partition was built at the speaker level and stratified by per-speaker dysarthric proportion, so that no clip from a given speaker appeared in more than one of the training, validation, and test sets of a fold. Each fold contained two to three test speakers. One remaining training speaker was held out for hyperparameter and threshold tuning.

Speaker-dependent (SD) evaluation provides an empirical upper bound. Each retained speaker’s clips were split into five stratified folds preserving that speaker’s dysarthric prevalence, and a classifier was trained and evaluated on each speaker independently. Speakers with fewer than ten dysarthric clips were excluded, since stratified five-fold splitting requires at least two positive clips per fold, leaving nine of eleven speakers. Per-speaker scores were averaged. This protocol is clinically meaningful in AC, where a pre-operative recording of the same patient can serve as training data for an intra-operative classifier. The difference between SD and SID therefore quantifies the cost of transferring across speakers rather than calibrating to one.

The LOSO evaluation holds out a single speaker per fold and repeats across all eleven speakers, and is the strictest form of speaker-independent evaluation. With eleven speakers, LOSO produced cross-fold AUC standard deviations near 0.10, against approximately 0.04 under SID grouping. The two protocols agree on point estimates, which indicates that the 5-fold grouping does not bias the reported AUC.

Because per-speaker clip counts and dysarthric proportions vary widely (Table II), metrics are decomposed by speaker so that individual contributions remain visible rather than absorbed into a cohort average. A single global decision threshold is unsuitable for the same reason, and the operating point is instead computed from each held-out speaker’s own NT distribution.

## F. Experimental design

The individual effect of each component cannot be read from the performance of the complete framework. Experiments are therefore organized as an ordered sequence of additions (Fig. 3), each introducing one component while the others are held fixed.

Baseline classifiers on the conventional features establish the reference. Three estimators of different inductive biases are compared, since a component-wise analysis is meaningful only if the choice of estimator does not dominate. Augmentation is examined next under both SD and speaker independent evaluation, which separates its two possible effects, namely a better estimate of a given speaker’s dysarthric range or improved transfer to unseen speakers. It is retained thereafter because it addresses the class imbalance of Table II, independently of its effect on either protocol.

The cascades are introduced with the same features as the baselines, so any difference isolates the two-stage architecture. The remaining experiments act on the representation. Self-supervised extraction from patient frames is compared against full-clip extraction, and the feature families are then added cumulatively, from the cepstral baseline to the multilayer fusion. Reporting the classifier comparison and the representation additions on the same cohort and protocols allows the two to be weighed against each other.

![](images/67fa9fbb9cdd89ff1afce7a99fac3848eab5c9fa8ffdea7068c3edecf1c7d61b.jpg)  
Fig. 3. Cumulative experimental design.

## IV. RESULTS

The framework was evaluated on the retained cohort of eleven patients, comprising 2,706 clips after the quality criterion, with a dysarthric prevalence of 22.1%. Each component is assessed in the order it is introduced in Section III-F, so that its effect is measured against the configuration preceding it. The conventional feature set and the three single-stage classifiers establish the reference, augmentation is assessed on that same configuration, the cascades are introduced without altering the representation, and the representation is varied last. AUC is the primary metric throughout, and results are reported under the protocols defined in Section III-E.

## A. Baseline classifiers and data augmentation

Three classifiers of distinct inductive bias were first evaluated on the conventional feature set under the LOSO protocol (Table V). XGBoost (0.6148) and the MLP (0.6117) perform comparably, while the BiGRU (0.5873) is weakest, and the leading metric is split across the three systems. The spread of 0.028 AUC is small relative to the cross-fold variability of this cohort, indicating that architecture is not the limiting factor on this representation. All three remain close to chance-corrected performance, with F1 below 0.38, which is consistent with a representation that transfers poorly across speakers.

The same systems were then assessed with and without full-clip HiFi-GAN augmentation under the SD and SID protocols (Table VI). Augmentation yields no systematic benefit. It improves the MLP under SD (0.6946 against 0.6550) and the BiGRU under the SID protocol (0.6141 against 0.5597), but degrades XGBoost under the SID protocol (0.5872 against 0.6440) and the BiGRU under SD (0.4699 against 0.5823). The direction of the effect is therefore conditional on both the estimator and the evaluation regime, and no system benefits under both protocols. This is consistent with augmentation acting on within-speaker variability, which the SD regime rewards, without supplying the inter-speaker variability that cross-speaker generalization requires. Perspeaker and per-fold distributions are shown in Fig. 4.

TABLE V  
BASELINE CLASSIFIER PERFORMANCE UNDER THE LOSO PROTOCOL.
<table><tr><td>Classifier</td><td>AUC</td><td>F1</td><td>BAcc</td><td>Acc</td></tr><tr><td>XGBoost</td><td>0.6148</td><td>0.3782</td><td>0.6334</td><td>0.6298</td></tr><tr><td>MLP</td><td>0.6117</td><td>0.3793</td><td>0.5988</td><td>0.5899</td></tr><tr><td>BiGRU</td><td>0.5873</td><td>0.3629</td><td>0.6203</td><td>0.6402</td></tr></table>

The two protocols differ substantially. XGBoost attains 0.8508 under SD against 0.6440 under the SID protocol, and comparable margins hold for the other systems. This gap between calibrating to a patient and generalizing to an unseen one exceeds any effect attributable to the classifier or to augmentation, and it persists across the experiments that follow.

## B. Cascade architectures

The two cascades were evaluated against their singlestage counterparts on the same feature set, under the SID and LOSO protocols (Table VII), so that any difference is attributable to the architecture alone.

The benefit of the cascade is confined to the stricter protocol. Under LOSO, the Cascade-MLP attains 0.6669 against 0.6117 for the single-stage MLP, and the Cascade-BiGRU attains 0.5968 against 0.5873, so both cascades exceed their baseline. Under the SID protocol the pattern does not hold: the Cascade-BiGRU improves marginally (0.5771 against 0.5597) while the Cascade-MLP does not (0.6428 against 0.6434).

This dependence on protocol is informative. The two protocols differ in how many speakers are held out per fold, so LOSO trains on more speakers and tests on a single unseen one. The first-stage estimate is generated by an inner LOSO procedure, and its usefulness to the second stage depends on how well it transfers to a speaker the tree has not seen. The cascade therefore helps where the first stage is estimated from the widest speaker sample, and contributes little when the training set is smaller and the test set more heterogeneous.

The gain is also larger for the feed-forward second stage than the recurrent one. The Cascade-MLP improves on its baseline by a wider margin than the Cascade-BiGRU under LOSO, which is consistent with the difference in how the first-stage estimate is admitted: the MLP receives it as an additional input dimension alongside the features, whereas the BiGRU receives only its session-level mean at the classification head. Even at its best, however, the Cascade-MLP remains near 0.67 AUC under LOSO, well below the 0.85 attained under speaker-dependent evaluation, so the architecture does not close the cross-speaker gap.

## C. Feature representation

Two experiments vary the representation while the classifiers remain fixed: the region from which the self-supervised block is extracted, and the cumulative addition of feature families.

Restricting self-supervised extraction to patient frames improves performance for every system tested (Fig. 5). The Cascade-MLP attains 0.760 against 0.741 for full-clip extraction, the Cascade-BiGRU 0.730 against 0.712, and XGBoost 0.721 against 0.701. The improvement is modest but consistent in direction and comparable in magnitude across three architectures of different inductive bias, which indicates that it originates in the representation rather than in any property of a particular estimator. Since the conventional descriptors are unchanged between conditions, the difference is attributable to the removal of clinician speech from the region over which the embedding is computed.

In the cumulative ablation, AUC increases at each successive configuration for every system (Table VIII). From the cepstral baseline to the proposed multi-layer fusion, the Cascade-MLP rises from 0.643 to 0.760, XGBoost from 0.591 to 0.721, and the Cascade-BiGRU from 0.579 to 0.730. For all three systems, each configuration improves on the previous one, with the Cascade-MLP highest throughout. The multi-layer fusion outperforms the single-layer configuration for each system, supporting the layer-wise argument that articulatory precision, phonological realization, and intelligibility are encoded at different depths, so that no single layer represents the impairment completely.

Taken together, the two sets of experiments locate the source of performance. Across the ablation, the representation accounts for a gain of between 0.117 and 0.151 AUC depending on the system, whereas the three architectures differ by at most 0.070 at any fixed configuration. The representation therefore governs cross-speaker performance to a greater extent than the choice of classifier, and the cascaded architectures improve on their baselines without altering that conclusion.

## V. DISCUSSION

The stage-wise design summarized in Fig. 3 attributes the observed performance to individual pipeline components rather than to the system as a whole. The results show that the classifier is a secondary factor, while the successive representational stages (cascading, diarization, and the selfsupervised feature set) account for the largest cross-speaker gains.

The baseline comparison (Table V) shows that three classifiers which learn in fundamentally different ways, a gradient-boosted tree, a feed-forward network, and a recurrent network, span a cross-speaker AUC range of only 4.7% under the LOSO protocol, from 0.5873 (BiGRU) to 0.6148 (XGBoost). The convergence of such different architectures to within a few percent is hard to reconcile with the classifier being the limiting factor. If the estimator is varied across its full plausible range and performance changes a little, then the ceiling on performance comes from the information present in the input, not from the model’s ability to use it. The results support the same conclusion, since all three systems sit near an AUC of 0.6, far below clinically useful discrimination, which suggests that conventional acoustic descriptors carry only a weak and largely speaker-specific signal that fails to transfer to unseen patients. Raising cross-speaker performance thus depends on improving the representation rather than the classifier, and the remaining pipeline stages should be optimized in turn.

TABLE VI  
EFFECT OF FULL-CLIP DATA AUGMENTATION UNDER THE SD AND SID PROTOCOLS.
<table><tr><td rowspan="2">Classifier</td><td rowspan="2">Augmentation</td><td colspan="4">SD (per-speaker 5-fold CV)</td><td colspan="4">SID (5-fold grouped)</td></tr><tr><td>AUC</td><td>F1</td><td>BAcc</td><td>Acc</td><td>AUC</td><td>F1</td><td>BAcc</td><td>Acc</td></tr><tr><td rowspan="2">XGBoost</td><td>None</td><td>0.8496</td><td>0.6054</td><td>0.7237</td><td>0.8245</td><td>0.6440</td><td>0.3671</td><td>0.6303</td><td>0.6545</td></tr><tr><td>Full-clip</td><td>0.8508</td><td>0.6066</td><td>0.7079</td><td>0.8229</td><td>0.5872</td><td>0.3592</td><td>0.6228</td><td>0.6041</td></tr><tr><td rowspan="2">MLP</td><td>None</td><td>0.6550</td><td>0.3935</td><td>0.6505</td><td>0.6901</td><td>0.6434</td><td>0.3939</td><td>0.6373</td><td>0.6357</td></tr><tr><td>Full-clip</td><td>0.6946</td><td>0.4226</td><td>0.6608</td><td>0.7163</td><td>0.6345</td><td>0.3857</td><td>0.6261</td><td>0.6850</td></tr><tr><td rowspan="2">BiGRU</td><td>None</td><td>0.5823</td><td>0.3698</td><td>0.5357</td><td>0.4007</td><td>0.5597</td><td>0.3450</td><td>0.6181</td><td>0.5427</td></tr><tr><td>Full-clip</td><td>0.4699</td><td>0.3469</td><td>0.5518</td><td>0.3907</td><td>0.6141</td><td>0.3823</td><td>0.6000</td><td>0.6166</td></tr></table>

![](images/9788e3bd02761442a07af6f36821e12b4fdc68bb5f7927e5128b383f649a2957.jpg)  
Fig. 4. Per-unit AUC for the three baseline classifiers under each protocol and augmentation condition, on the conventional feature set.

![](images/21e722bc6a62213695c04e193c67a1bf7da14805b527ce01cb8c9e64b43dcff7.jpg)  
Fig. 5. Diarization impact on model performance.

The gap between evaluation protocols quantifies the difficulty these stages confront. Under the speaker-dependent protocol, XGBoost reaches an AUC 31.9% higher than under the speaker-independent protocol (0.8496 versus 0.6440), and the augmentation results (Table VI) show the betweenclassifier separation present under the SD protocol reducing to a single overlapping band under the SID protocol. The problem is thus one of transfer across speakers, not of separating impaired from normal speech within a speaker. This also explains why full-clip augmentation produced no systematic cross-speaker benefit. Replicating existing clips increases the training set the model already observes but introduces no new inter-speaker variation, which is the axis along which generalization must occur. Augmentation was therefore retained only for correcting class imbalance, and the transfer difficulty is left to the architectural and representational stages that follow.

TABLE VII  
CASCADE ARCHITECTURES PERFORMANCE UNDER THE SID FIVE-FOLD AND LOSO PROTOCOLS.
<table><tr><td>Protocol(SID)</td><td>System</td><td>AUC</td><td>F1</td><td>BAcc</td><td>Acc</td></tr><tr><td rowspan="4">5-fold</td><td>MLP</td><td>0.6434</td><td>0.3939</td><td>0.6373</td><td>0.6357</td></tr><tr><td>Cascade-MLP</td><td>0.6428</td><td>0.3884</td><td>0.6119</td><td>0.5563</td></tr><tr><td>BiGRU</td><td>0.5597</td><td>0.3450</td><td>0.6181</td><td>0.5427</td></tr><tr><td>Cascade-BiGRU</td><td>0.5771</td><td>0.3531</td><td>0.5640</td><td>0.5128</td></tr><tr><td rowspan="4">LOSO</td><td>MLP</td><td>0.6117</td><td>0.3793</td><td>0.5988</td><td>0.5899</td></tr><tr><td>Cascade-MLP</td><td>0.6669</td><td>0.4049</td><td>0.6308</td><td>0.5698</td></tr><tr><td>BiGRU</td><td>0.5873</td><td>0.3629</td><td>0.6203</td><td>0.6402</td></tr><tr><td>Cascade-BiGRU</td><td>0.5968</td><td>0.3653</td><td>0.5958</td><td>0.5374</td></tr></table>

TABLE VIII

CUMULATIVE FEATURE ABLATION UNDER THE SID FIVE-FOLD PROTOCOL.
<table><tr><td>Configuration</td><td>System</td><td>AUC</td><td>F1</td><td>BAcc</td><td>Acc</td></tr><tr><td rowspan="3">MFCC only</td><td>XGBoost</td><td>0.591</td><td>0.381</td><td>0.545</td><td>0.313</td></tr><tr><td>Cascade-MLP</td><td>0.643</td><td>0.407</td><td>0.601</td><td>0.486</td></tr><tr><td>Cascade-BiGRU</td><td>0.579</td><td>0.371</td><td>0.563</td><td>0.455</td></tr><tr><td rowspan="3">+ Full Phase 1 (incl. diarization)</td><td>XGBoost</td><td>0.662</td><td>0.434</td><td>0.635</td><td>0.639</td></tr><tr><td>Cascade-MLP</td><td>0.699</td><td>0.446</td><td>0.654</td><td>0.685</td></tr><tr><td>Cascade-BiGRU</td><td>0.629</td><td>0.405</td><td>0.613</td><td>0.642</td></tr><tr><td rowspan="3">+ wav2vec2 layer 6</td><td>XGBoost</td><td>0.701</td><td>0.453</td><td>0.651</td><td>0.700</td></tr><tr><td>Cascade-MLP</td><td>0.737</td><td>0.472</td><td>0.678</td><td>0.691</td></tr><tr><td>Cascade-BiGRU</td><td>0.664</td><td>0.429</td><td>0.635</td><td>0.648</td></tr><tr><td rowspan="3">+ wav2vec2 {6, 9, final} (proposed)</td><td>XGBoost</td><td>0.721</td><td>0.461</td><td>0.654</td><td>0.671</td></tr><tr><td>Cascade-MLP</td><td>0.760</td><td>0.494</td><td>0.694</td><td>0.710</td></tr><tr><td>Cascade-BiGRU</td><td>0.730</td><td>0.477</td><td>0.669</td><td>0.683</td></tr></table>

The first of these stages is the cascade, which introduces an architectural modification to the baseline classifiers rather than a change in features or training data. The single-stage MLP and BiGRU are restructured into two-stage models in which a first-stage XGBoost feeds a calibrated probability into the neural second stage. Evaluated on the conventional feature set, this structural change recovers discriminative structure that a single-stage model of the same family leaves unused. Under the LOSO protocol (Table VII), the Cascade-MLP improves on the single-stage MLP by 9.0% in AUC (0.6117 to 0.6669) and the Cascade-BiGRU improves on the single-stage BiGRU by 1.6%, with the gain largest under LOSO, the protocol in which every test speaker is unseen and which most closely reflects deployment on a new patient. This gain follows from the modified architecture. The firststage XGBoost reduces the high-dimensional, largely weakly informative feature vector into a single calibrated probability that is robust to noisy and redundant dimensions, and the restructured second-stage neural model then operates on this smoothed, speaker-agnostic summary alongside the original features, rather than rediscovering the decision boundary from the raw high-dimensional input in a small-sample model where it is prone to overfitting speaker identity. The inner LOSO construction that generates the first-stage probabilities is essential to this validity, since it ensures the second stage is trained on out-of-sample probabilities of the same statistical character it receives at test time, preventing an overconfident in-sample signal that would not transfer.

The next stage acts on the representation itself, through diarization-conditioned extraction. Restricting the selfsupervised embedding to diarized patient frames rather than the full clip raises AUC by 2.9% for XGBoost, 2.6% for the Cascade-MLP, and 2.5% for the Cascade-BiGRU (Fig. 5). The gain is smaller than that of the cascade, but its uniformity across three classifiers of different architectures is the more informative feature. An effect that follows the change in extraction region regardless of the downstream model is a property of the representation’s input rather than of any one estimator, and is therefore attributable to the extraction choice itself. Diarization shows that the clinician speech occupies a class-dependent share of each recording, so a full-clip embedding encodes recording-context structure that correlates with the label without reflecting patient pathology. Restricting extraction to patient frames removes this confound at the representation level, before any classifier is applied, which is why the improvement transfers across all three architectures.

The final and largest stage is the complete self-supervised representation, whose contribution the cumulative ablation (Table VIII) makes explicit. Across the full pipeline, moving from the MFCC baseline to the proposed multi-layer configuration raises AUC by 22.0% for XGBoost, 18.2% for the Cascade-MLP, and 26.1% for the Cascade-BiGRU, gains an order of magnitude larger than the 4.7% that separates the classifiers. Performance is therefore governed primarily by the representation rather than the classifier, justifying the intensive feature-extraction effort in this work.

The structure of this gain is technically informative. Adding the full conventional Phase 1 stack over MFCC contributes a relative AUC gain of 8.6% to 12.0%, but this arises from many descriptors acting together and is partly classifierdependent. A single self-supervised wav2vec2 layer, by contrast, contributes a further 5.4% to 5.9% from one feature family alone. Per feature family, the self-supervised representation is considerably more discriminative signal than the conventional descriptors. This is expected, since handspecified descriptors largely encode speaker identity and do not transfer, whereas the W2V2 encoder, pretrained on external corpora, captures phonetic and phonological structure learned independently of this cohort and therefore generalizes to new speakers.

The multi-layer fusion is justified by how these encoders represent speech, not just by empirical gain. Extending the single layer to the {6, 9, final} fusion adds a further 2.9% to 3.1% for the tree and feed-forward cascades and 9.9% for the recurrent cascade. Layer-wise analyses of selfsupervised speech encoders indicate that phonetic detail is concentrated in mid-network layers, phonological structure in intermediate-deep layers, and intelligibility- and speakerrelated information in the final layers. Since dysarthria simultaneously impairs articulatory precision, phonological realization, and intelligibility, no single layer encodes the full profile, and the consistent improvement from fusion across all three classifiers indicates that the selected depths contribute complementary rather than overlapping information.

In practical terms, this hierarchy implies that a deployable monitoring system should be built on clean, patientattributable measurements and a strong pretrained representation, treating the classifier as a largely interchangeable final component. The achievable performance falls between two scenarios. The cross-speaker results correspond to a previously unseen patient evaluated without calibration, whereas the markedly higher speaker-dependent results, exceeding the cross-speaker values by up to 31.9% in AUC, correspond to the case where a pre-operative recording enables patientspecific calibration. Since AC already involves a structured pre-operative assessment, this calibrated regime is clinically realistic, and the difference between the two regimes indicates how much performance a brief pre-operative baseline could recover.

Several limitations remain. The eleven-speaker cohort is small, so the argument rests on the directional consistency of the orderings across classifiers and configurations rather than on individual point estimates, and a larger, more phenotypically diverse cohort would be needed to separate configurations at finer resolution. The binary dysarthriaversus-no-trouble target also collapses severity levels and does not address whether the dysarthric signature varies with the cortical or subcortical stimulation site; a multiclass extension distinguishing dysarthria, anomia, and paraphasia, annotated separately rather than aggregated, would add clinical granularity beyond that of prior AC work. Finally, generalization beyond DATABRASE is untested. The cross-lingual evaluation of [26] showed substantial performance loss under Japanese–French transfer, indicating that language- and center-specific factors interact with the selfsupervised representation in ways this cohort cannot probe. Validation on independent AC cohorts at different centers, ideally with matched-protocol re-evaluation, together with fine-tuning rather than freezing the encoder once cohort size permits, are the natural next steps before stronger deployment claims can be made.

## VI. CONCLUSION

This work approached intra-operative dysarthria detection as a component-wise pipeline-optimization problem, quantifying the contribution of each stage from data quality through feature extraction and classification. The results point to a single conclusion that cross-speaker performance is governed principally by the representation rather than the classifier. Three classifiers of fundamentally different design converge to within 4.7% AUC, whereas enriching the representation from conventional descriptors to a multilayer self-supervised fusion raises AUC by 18.2% to 26.1%, an order of magnitude larger, with diarization-conditioned extraction adding a further consistent gain. The proposed cascaded architecture recovers additional structure a singlestage model leaves unused, most clearly under the strict LOSO protocol that best reflects deployment on a new patient. Together, these results imply that the priority should be clean patient-attributable measurement and strong pretrained representations rather than more complex classifiers for lowresource intra-operative settings. The main limitation is the small speaker cohort, and the most promising directions for further work are the acquisition of a larger and more diverse speaker set and the fine-tuning, rather than freezing of the self-supervised encoder once cohort size makes this safe against overfitting to speaker identity.

## REFERENCES

[1] I. Mart´ın-Monzon, Y. Rivero Ballagas, and S. Arias-S ´ anchez, “Lan-´ guage mapping: A systematic review of protocols that evaluate linguistic functions in awake surgery,” Applied Neuropsychology: Adult, vol. 29, no. 4, pp. 845–854, jul 2022.

[2] E. Collee, A. Vincent, C. Dirven, and D. Satoer, “Speech and Language´ Errors during Awake Brain Surgery and Postoperative Language Outcome in Glioma Patients: A Systematic Review,” Cancers, vol. 14, no. 21, nov 2022.

[3] E. C. Gommers, K. E. Collee, A. J. P. E. Vincent, E. M. Bos, C. M. F.´ Dirven, S. K. Koekkoek, P. Kruizinga, and D. D. Satoer, “P01.12.B Analysis of semi-spontaneous speech before, during and after awake craniotomy: A case study,” Neuro-Oncology, vol. 24, no. Suppl 2, p. ii26, sep 2022.

[4] M. Keough, W. Chow, C. Reilly, A. G. Loron, and I. Parney, “SURG-22. Awake speech mapping coupled with intraoperative MRI is associated with greater extent of resection in language eloquent diffuse glioma surgery,” Neuro-Oncology, vol. 27, no. Supplement 5, p. v399, nov 2025.

[5] C. Bhat and H. Strik, “Speech Technology for Automatic Recognition and Assessment of Dysarthric Speech: An Overview,” Journal of speech, language, and hearing research: JSLHR, vol. 68, no. 2, pp. 547–577, feb 2025.

[6] H. Kim, M. Hasegawa-Johnson, A. Perlman, J. Gunderson, T. S. Huang, K. Watkin, and S. Frame, “Dysarthric speech database for universal access research,” in Interspeech 2008. ISCA, sep 2008, pp. 1741–1744.

[7] F. Rudzicz, A. K. Namasivayam, and T. Wolff, “The TORGO database of acoustic and articulatory speech from speakers with dysarthria,” Language Resources and Evaluation, vol. 46, no. 4, pp. 523–541, dec 2012.

[8] N. N. Prabhakera and P. Alku, “Dysarthric speech classification using glottal features computed from non-words, words and sentences,” in Proceedings of Interspeech. International Speech Communication Association (ISCA), sep 2018, pp. 3403–3407.

[9] K. L. Kadi, S.-A. Selouani, B. Boudraa, and M. Boudraa, “Discriminative prosodic features to assess the dysarthria severity levels,” in Proceedings of the World Congress on Engineering 2013, vol. 3, 2013, pp. 2201–2205.

[10] A. Al-Ali, S. Al-Maadeed, M. Saleh, R. C. Naidu, Z. C. Alex, P. Ramachandran, R. Khoodeeram, and R. K. M, “The detection of dysarthria severity levels using AI models: A review,” IEEE Access, vol. 12, pp. 48 223–48 238, 2024.

[11] J. Millet and N. Zeghidour, “Learning to Detect Dysarthria from Raw Speech,” in ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), may 2019, pp. 5831–5835.

[12] F. Javanmardi, S. R. Kadiri, and P. Alku, “Pre-trained models for detection and severity level classification of dysarthria from speech,” Speech Communication, vol. 158, p. 103047, 2024.

[13] A. Pasad, J.-C. Chou, and K. Livescu, “Layer-Wise analysis of a selfsupervised speech representation model,” in 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, dec 2021, pp. 914–921.

[14] B. Kadirvelu, L. Stumpf, S. Waibel, and A. A. Faisal, “Speakerindependent dysarthria severity classification using self-supervised transformers and multi-task learning,” PLOS Digital Health, vol. 4, no. 11, p. e0001076, 2025.

[15] A. A. Joshy and R. Rajan, “Automated Dysarthria Severity Classification: A Study on Acoustic Features and Deep Learning Techniques,” IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 30, pp. 1147–1157, 2022.

[16] M. Suresh, R. Rajan, and J. Thomas, “Speaker independent dysarthria severity classification using synthesis-based augmentation,” International Journal of Speech Technology, vol. 28, no. 2, pp. 565–580, jun 2025.

[17] A. Venkata Siva Manoj, V. Lakshman, A. Kamuju, V. Pulagam, and G. Jyothish Lal, “Transformer-based Transfer Learning for Enhanced Speech Dysarthria Severity Assessment,” in 2024 15th International Conference on Computing Communication and Networking Technologies (ICCCNT), jun 2024, pp. 1–6.

[18] R. Mahum, A. M. El-Sherbeeny, K. Alkhaledi, and H. Hassan, “Tran-DSR: A hybrid model for dysarthric speech recognition using transformer encoder and ensemble learning,” Applied Acoustics, vol. 222, p. 110019, jun 2024.

[19] I. Shahin, A. B. Nassif, and S. Hamsa, “Novel cascaded gaussian mixture model-deep neural network classifier for speaker identification in emotional talking environments,” Neural Computing and Applications, vol. 32, pp. 2575–2587, 2020.

[20] I. Shahin, A. B. Nassif, N. Nemmour, A. Elnagar, A. Alhudhaif, and K. Polat, “Novel hybrid DNN approaches for speaker verification in emotional and stressful talking environments,” Neural Computing and Applications, vol. 33, no. 23, pp. 16 033–16 055, 2021.

[21] J. C. Prabhala, R. Ragoju, V. Kupilli, and C. Chesneau, “Enhanced early detection of dysarthric speech disabilities using stacking ensemble deep learning model,” Machine Learning with Applications, vol. 21, p. 100721, 2025.

[22] A. Tripathi, S. Bhosale, and S. K. Kopparapu, “Improved Speaker Independent Dysarthria Intelligibility Classification Using Deepspeech Posteriors,” in ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), may 2020, pp. 6114–6118.

[23] G. Schu, P. Janbakhshi, and I. Kodrasi, “On using the UA-Speech and TORGO databases to validate automatic dysarthric speech clas-

sification approaches,” in ICASSP 2023 – 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–5.

[24] A. K. Roy, H. K. Kathania, and P. Sapkota, “Enhancing speakerindependent dysarthric speech severity classification with dsscnet and cross-corpus adaptation,” 2025.

[25] T. Nishimura, T. Nagao, H. Iseki, Y. Muragaki, M. Tamura, and S. Minami, “Classification of patient’s reaction in language assessment during awake craniotomy,” in 2014 IEEE 7th International Workshop on Computational Intelligence and Applications (IWCIA). Hiroshima, Japan: IEEE, Nov. 2014, pp. 207–212.

[26] I. Maoudj, A. Kuwano, C. Panheleux, Y. Kubota, T. Kawamata, Y. Muragaki, K. Masamune, R. Seizeur, G. Dardenne, and M. Tamura, “Classification of speech arrests and speech impairments during awake craniotomy: A multi-databases analysis,” International Journal of Computer Assisted Radiology and Surgery, vol. 20, no. 2, pp. 217– 224, feb 2025.

[27] H. Duffau, “Awake surgery for incidental WHO grade II gliomas involving eloquent areas,” Acta Neurochirurgica, vol. 154, no. 4, pp. 575–584, apr 2012.

[28] S. S. Xu, X. Ke, M.-W. Mak, K. H. Wong, H. Meng, T. C. Y. Kwok, J. Gu, J. Zhang, W. Tao, and C. Chang, “Speaker-turn aware diarization for speech-based cognitive assessments,” Frontiers in Neuroscience, vol. 17, jan 2024.

[29] R. Riad, H. Titeux, L. Lemoine, J. Montillot, A. Sliwinski, J. Hamet Bagnou, X. N. Cao, A.-C. Bachoud-Levi, and E. Dupoux,´ “A comparison study on patient-psychologist voice diarization,” in Proc. 9th Workshop on Speech and Language Processing for Assistive Technologies (SLPAT-2022). Dublin, Ireland: Association for Computational Linguistics, 2022, pp. 30–36.

[30] A. Zolensky, K. J. Jang, J. Sabin, A. Hartzler, B. Alasaly, S. Mopidevi, M. Liberman, and K. Johnson, “Speaker role identification in clinical conversations,” in Biocomputing 2026: Proceedings of the Pacific Symposium. World Scientific, 2026.

[31] J. Kong, J. Kim, and J. Bae, “HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis,” in Advances in Neural Information Processing Systems, vol. 33. Curran Associates, Inc., 2020, pp. 17 022–17 033.

[32] J. Qin, Z. Jiang, Z. Wang, and C. Ji, “A HiFi-GAN method-based algorithm for expanding and classifying Parkinson’s vocal samples,” in 2022 4th International Conference on Frontiers Technology of Information and Computer (ICFTIC), dec 2022, pp. 308–312.

[33] Y. Qian, H. Hu, and T. Tan, “Data augmentation using generative adversarial networks for robust speech recognition,” Speech Communication, vol. 114, pp. 1–9, nov 2019.

[34] H. Bredin, R. Yin, J. M. Coria, G. Gelly, P. Korshunov, M. Lavechin, D. Fustes, H. Titeux, W. Bouaziz, and M.-P. Gill, “Pyannote.Audio: Neural Building Blocks for Speaker Diarization,” in ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), may 2020, pp. 7124–7128.

[35] M. S. Remya, R. Raman, R. Sankaran, V. Namboodiri, and P. Nedungadi, “Artificial Intelligence for Speech Classification and Enhancement of Speech and Language Disorders: Techniques, Applications, and Future Directions,” IEEE Access, vol. 13, pp. 177 136–177 159, 2025.

[36] Z. Qian, K. Xiao, and C. Yu, “A survey of automatic speech recognition for dysarthric speech,” Electronics, vol. 12, no. 20, p. 4278, 2023.

[37] A. Conneau, A. Baevski, R. Collobert, A. Mohamed, and M. Auli, “Unsupervised cross-lingual representation learning for speech recognition,” in Interspeech 2021, 2021, pp. 2426–2430.

[38] Facebook AI, “wav2vec2-large-xlsr-53-french,” https://huggingface.co/facebook/wav2vec2-large-xlsr-53-french, 2021, french fine-tuned checkpoint based on XLSR-53.

[39] A. Angra, H. Muralikrishna, D. A. Dinesh, and V. Thenkanidiyoor, “Exploring Aggregated wav2vec 2.0 Features and Dual-Stream TDNN for Efficient Spoken Dialect Identification,” IEEE Access, vol. 13, pp. 3115–3129, 2025.

[40] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. San Francisco California USA: ACM, aug 2016, pp. 785–794.

[41] R. Shwartz-Ziv and A. Armon, “Tabular data: Deep learning is not all you need,” Information Fusion, vol. 81, pp. 84–90, may 2022.

[42] T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, “Optuna: A next-generation hyperparameter optimization framework,” in Proc.

25th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2019, pp. 2623–2631.

[43] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss´ for dense object detection,” in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2017, pp. 2980–2988.