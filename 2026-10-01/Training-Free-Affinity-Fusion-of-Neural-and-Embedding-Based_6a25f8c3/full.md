# Training-Free Affinity Fusion of Neural and Embedding-Based Speaker Diarization

Yehoshua Dissen<sup>1,\*</sup> Joseph Keshet<sup>2</sup> Eduard Golshtein<sup>1,\*</sup>

<sup>1</sup>Linguana <sup>2</sup>Technion – Israel Institute of Technology

<sup>\*</sup>Equal contribution

Abstract—Speaker diarization systems based on speaker embeddings and neural diarization exploit complementary forms of speaker information, but their intermediate representations are not directly compatible. We introduce Training-Free Affinity Fusion (TFAF), which integrates the speaker structure inferred by a neural diarizer into an embedding-based diarization system. The neural speaker partition is used to condition local speaker representations, from which we construct a continuous affinity matrix and combine it with the embedding-based acoustic affinity before a single global clustering step. The method requires no additional training, shared embedding space, speaker-label alignment, or hard transfer of the neural diarizer’s speaker count. Experiments on AMI and CALLHOME show consistent DER improvements over both constituent systems; on AMI, fusion also improves speaker-attributed transcription. Ablations show that the neural speaker partition accounts for most of the gain, while retaining the continuous embedding-based affinities provides additional benefit over hard partition fusion.

Index Terms—speaker diarization, speaker clustering, affinity fusion, system combination

## I. INTRODUCTION

Speaker diarization, the task of determining who spoke when, is commonly approached through two broad paradigms: clustering speaker embeddings over a recording, or using a neural diarization model to directly estimate speaker activity from the speech sequence [1], [2]. Embedding-clustering systems are particularly well suited to maintaining speaker identity over long recordings. Their main limitation is the tradeoff between speaker discrimination and temporal resolution. Long analysis windows produce more reliable speaker embeddings but blur short turns and speaker changes, whereas short windows provide better temporal localization but weaker speaker representations. Multiscale clustering addresses this tradeoff by combining affinities computed at several window lengths before global spectral clustering [6], [7].

Neural diarization systems make a different tradeoff. By processing sequences of speech jointly, they can use temporal context to resolve short turns, speaker transitions, and overlapping speech rather than treating each segment as an independent speaker observation. However, processing long recordings directly with a neural sequence model can require substantial memory and computation. Long recordings are therefore often processed in shorter chunks, which introduces the additional problem of maintaining consistent speaker identities across chunks. More generally, neural diarization must accommodate a variable number of speakers and the permutation ambiguity of speaker labels [4], [5]. These properties make the two paradigms naturally complementary. Neural diarization provides detailed local speaker information, while embedding clustering provides strong recording-level speaker discrimination and global consistency [5].

Combining the two paradigms is not straightforward because their intermediate speaker representations are generally incompatible. Embedding-based diarization operates on continuous pairwise speaker similarities, whereas neural diarization produces recording-internal speaker assignments and representations that are not assumed to share the embedding system’s representation space. Consequently, the two systems cannot in general be fused by directly comparing their speaker labels or representations. Existing alternatives address related but different settings: DOVER and DOVER-Lap [17], [18] fuse completed diarization outputs, co-association methods combine hard same-cluster relations [19], [20], probabilitylevel fusion requires compatible neural outputs [21], and hybrid neural–clustering systems integrate the two paradigms within a single diarization pipeline [5]. What is missing is a training-free interface that transfers neural speaker structure into an embedding-based affinity graph while preserving its continuous acoustic information.

We introduce Training-Free Affinity Fusion (TFAF), which uses the neural speaker partition to construct a continuous, partition-conditioned affinity that interfaces with the embedding-based graph. The neural diarizer’s recording internal speaker structure conditions its local representations to form soft affinity evidence, which is added to the continuous embedding-based affinity before final clustering. This preserves the acoustic geometry of the embedding system while allowing the neural diarizer to contribute speaker information without label alignment, a shared embedding space, additional training, or hard adoption of its speaker count or final partition. Experiments on AMI and CALLHOME show that TFAF improves over both systems individually. Ablations show that most of the transferable information comes from the neural speaker partition, while preserving the continuous acoustic affinity is important for effective fusion.

## II. RELATED WORK

A wide range of approaches have been explored for embedding-based diarization, from x-vector speaker representations [3] to stronger neural encoders such as TitaNet [11], multiscale methods that combine information across different temporal resolutions [6], [7], and self-supervised approaches that learn speaker representations from unlabeled speech [13], [14].

Neural diarization instead estimates speaker activity directly from sequences of speech [4]. By processing temporal context jointly, these systems can model short turns, speaker transitions, and overlapping speech without treating each segment as an independent speaker observation. Hybrid approaches combine neural local diarization with clustering or speaker tracking to maintain speaker identities over longer recordings and accommodate a variable number of speakers [5], [15].

Several works combine complementary information at different stages of the diarization pipeline. Park et al. [9] combine lexical and acoustic adjacency matrices before spectral clustering, while Turn-to-Diarize [10] uses detected speaker turns to constrain embedding-based diarization. At the systemoutput level, DOVER and DOVER-Lap align speaker labels across completed diarization hypotheses and combine their decisions [17], [18]. Co-association methods instead convert hard partitions into pairwise same-cluster evidence [19], [20], while recent probability-level fusion combines calibrated speaker-activity probabilities from multiple neural diarization systems [21].

Our approach differs from these methods in both the information it combines and the stage at which fusion is performed. The embedding-based system provides a continuous acoustic affinity matrix, while the neural diarizer provides speaker assignments and local neural representations which are not assumed to share a representation space with the embedding system. Rather than aligning completed outputs or converting both systems into hard partitions, we preserve the continuous acoustic affinities and incorporate the neural speaker structure before final clustering. This avoids speaker-label alignment and hard transfer of the neural diarizer’s speaker count or final partition.

## III. METHOD

Embedding-based and neural diarization systems provide complementary speaker evidence. Embedding-based systems compare speaker embeddings over the recording, providing strong global speaker discrimination, but they inherit the tradeoff between short windows for temporal resolution and longer windows for reliable speaker embeddings. One way to overcome this, is by using multiple temporal scales and combine them. Neural diarizers use temporal context to model speaker activity and speaker turns more directly, but expose recording-internal speaker assignments rather than the same continuous affinity structure used by embedding clustering.

Our approach combines these two sources at the affinity level before the final clustering step. Let the embedding-based system define N base segments. At acoustic scale $s ,$ speaker embeddings are extracted from scale-specific windows anchored to the same base segments, producing $A ^ { ( s ) } \in \mathbb { R } ^ { N \times N }$ The embedding-based affinity is

$$
A ^ { \mathrm { e m b } } = \sum _ { s = 1 } ^ { K } w _ { s } A ^ { ( s ) } ,\tag{1}
$$

where K is the number of acoustic scales. We use equal weights $w _ { s } = 1 , { \mathrm { s o ~ } } A ^ { \mathrm { e m b } }$ is the sum of the per-scale affinities.

The neural diarizer supplies local speaker representations $\ell _ { q } ,$ a recording-level assignment $k ( q )$ for each local speaker instance $q ,$ and frame-level speaker activity indicating when each $q$ is active. The labels $k ( q )$ are recording-internal and carry no correspondence to the clusters produced by the embeddingbased system. We do not require pairwise similarity between the local representations to define a globally consistent speaker metric; when it is informative, the fusion exploits it, while the partition-only variant below requires only the speaker assignments.

For each inferred neural speaker $m ,$ let $c _ { m }$ be the centroid of the local representations assigned to that speaker. Before blending, both vectors are normalized, $\hat { \ell } _ { q } = \ell _ { q } / \lVert \ell _ { q } \rVert _ { 2 }$ and $\hat { c } _ { m } = c _ { m } / \lVert c _ { m } \rVert _ { 2 }$ . We form the partition-conditioned representation

$$
u _ { q } = \frac { ( 1 - \alpha ) \hat { \ell } _ { q } + \alpha \hat { c } _ { k ( q ) } } { \left\| ( 1 - \alpha ) \hat { \ell } _ { q } + \alpha \hat { c } _ { k ( q ) } \right\| _ { 2 } } .\tag{2}
$$

Thus $\alpha = 0 . 5$ gives the local and speaker-level representations equal vector weight.

The $u _ { q }$ representations are mapped onto a fixed neural temporal grid by accumulating their contributions over the 20 ms frames in which the corresponding local speaker is active. Frames with exactly one active local speaker contribute its $u _ { q } ;$ frames with no active speaker or multiple active speakers contribute no evidence. The contributions within each neural cell are averaged and normalized to obtain $z _ { i }$ on the basesegment grid. Empty cells are set to $z _ { i } \ = \ 0 ,$ , so the neural branch abstains rather than contributing different-speaker evidence. A segment crossing a neural speaker boundary can therefore contain contributions from both speakers rather than receiving a hard label.

We construct a continuous neural affinity matrix B, with $B _ { i j }$ equal to the cosine similarity between nonzero $z _ { i }$ and $z _ { j }$ and zero if either segment is uncovered. We fuse the two affinity sources as

$$
\boldsymbol { A } ^ { \mathrm { f u s e d } } = \boldsymbol { A } ^ { \mathrm { e m b } } + \lambda \boldsymbol { B } = \sum _ { s = 1 } ^ { K } w _ { s } \boldsymbol { A } ^ { ( s ) } + \lambda \boldsymbol { B } ,\tag{3}
$$

where λ controls the neural contribution. The same clustering procedure used by the embedding-based system is then applied once to $A ^ { \mathrm { f u s e d } }$ to obtain the final speaker partition and speaker count. Meaning, the neural diarizer contributes speaker evidence without imposing its labels or speaker count on the final output.

For the partition-only ablation, B is replaced by a samespeaker affinity derived from the neural frame assignments. The frame-level indicator is $\mathbf { 1 } [ k ( q ( t ) ) = k ( q ( t ^ { \prime } ) ) ]$ when both frames have a single active speaker and zero otherwise, and is aggregated using the same temporal mapping. Boundary cells may therefore yield fractional affinities. In the full system, $\alpha = 0 . 5$ and $\lambda = 1$ unless otherwise stated.

## IV. EXPERIMENTAL SETUP

## A. Systems

Our embedding-based system is the NeMo Equal-w-MS-Clus configuration [7], using TitaNet-L speaker embeddings [11] and NME-SC [8], [12]. For AMI, we use six window durations, [3.0, 2.5, 2.0, 1.5, 1.0, 0.5] s, with half-window shifts. For CALLHOME, we use the corresponding five telephony durations, [1.5, 1.25, 1.0, 0.75, 0.5] s [7]. VBx [16] is used only as the additional hypothesis in the three-system fusion baselines.

Our neural diarization system is DiariZen [15]. We use the frozen public diarizen-wavlm-large-s80-md checkpoint with its default pipeline and speech detection. DiariZen processes the recording in 16 s chunks with a 1.6 s step. Its segmentation model produces speaker activities every 20 ms for up to four chunk-local speakers. For each chunklocal speaker, the speaker-embedding stage extracts a 256- dimensional representation $\ell _ { q }$ from the corresponding activitymasked audio, and DiariZen’s clustering stage assigns that local speaker to a recording-level speaker k(q). For each recording-level speaker m, we compute the centroid $c _ { m }$ from the local representations assigned to that speaker.

To combine the DiariZen output with Equal-w-MS-Clus, we map the neural information onto a 0.5 s window / 0.25 s shift grid. Each 20 ms frame with exactly one active DiariZen speaker contributes the corresponding partition-conditioned representation $u _ { q }$ from Eq. (2) to every grid cell containing that frame. Each cell is represented by the $\ell _ { 2 } \cdot$ -normalized mean of its contributions; cells with no single-speaker frames are assigned the zero vector and therefore contribute no neural affinity. Frames with multiple active speakers are excluded.

The resulting grid is treated as an additional affinity scale. The standard Equal-w-MS-Clus midpoint mapping associates each base-scale segment with the nearest neural grid cell, using the same temporal mapping employed for the acoustic scales. Consequently, a segment spanning a DiariZen speaker change receives a normalized mixture of the corresponding speaker representations rather than a hard speaker label. No additional training or learned alignment is introduced.

## B. Datasets and scoring

On AMI [22], we use the official 16-meeting MixHeadset test split and the only-words reference setup. Following the matched Equal-w-MS-Clus evaluation, we use oracle speech activity, ignore overlap, and score DER with a 0.25s collar. On CALLHOME [23], we use SRE-2000 disc 8 with the matched telephony configuration and oracle speech activity for the main comparisons. DiariZen itself is always run with its default speech detection; for oracle-SAD comparisons, all system outputs are restricted to the same oracle speech regions before scoring.

For speaker-attributed evaluation, the recognized transcript is held fixed across diarization systems. Word err. counts reference words assigned to the wrong speaker after optimal speaker-label mapping, excluding overlap-ambiguous words.

TABLE I  
CROSS-CORPUS PERFORMANCE UNDER IDENTICAL ORACLE SPEECH COVERAGE.
<table><tr><td>Corpus</td><td>System</td><td>DER↓</td><td>Word err. ↓</td></tr><tr><td>AMI</td><td>Equal-w-MS-Clus</td><td>1.08</td><td>739</td></tr><tr><td>AMI</td><td>DiariZen</td><td>1.54</td><td>859</td></tr><tr><td>AMI</td><td>TFAF</td><td>0.85</td><td>465</td></tr><tr><td>CALLHOME</td><td>Equal-w-MS-Clus</td><td>3.85</td><td>一</td></tr><tr><td>CALLHOME</td><td>DiariZen</td><td>4.17</td><td>一</td></tr><tr><td>CALLHOME</td><td>TFAF</td><td>2.74</td><td>一</td></tr></table>

TABLE II

AMI ABLATION UNDER IDENTICAL ORACLE SPEECH COVERAGE. EACH “+ DZ” ROW ADDS THE INDICATED INFORMATION FROM THE SAME DIARIZEN INFERENCE.
<table><tr><td>Setting</td><td>Word err. ↓</td><td>tcpWER↓</td></tr><tr><td>Equal-w-MS-Clus</td><td>739</td><td>30.88</td></tr><tr><td>DiariZen</td><td>859</td><td>31.62</td></tr><tr><td>+ DZ speaker count</td><td>3598</td><td>37.03</td></tr><tr><td>+ DZ local affinity</td><td>531</td><td>30.25</td></tr><tr><td>+ DZ binary partition affinity</td><td>521</td><td>29.92</td></tr><tr><td>+ DZ centroid affinity</td><td>478</td><td>29.93</td></tr><tr><td>+ DZ local+centroid affinity (TFAF)</td><td>465</td><td>29.95</td></tr></table>

We also report time-constrained minimum-permutation WER (tcpWER) with a 5s collar using MeetEval [24]; unlike Word err., tcpWER includes both recognition and speaker-attribution errors. For oracle-SAD experiments, systems that do not natively use oracle speech regions are adapted to the same speech coverage before scoring.

## V. RESULTS

We first evaluate whether TFAF improves over its two constituent systems. On AMI, our Equal-w-MS-Clus baseline closely matches the published result of Park et al. [7] (1.08 vs. 1.06 DER). On CALLHOME, our implementation obtains 3.85 DER, compared with 4.57 reported in [7]. As shown in Table I, TFAF improves over both systems on both corpora.

We next examine which part of the DiariZen output drives this improvement. Table II uses the same DiariZen inference throughout and changes only the information transferred to Equal-w-MS-Clus. Transferring its estimated speaker count as a hard constraint performs poorly, increasing AMI Word err. from 739 to 3598. In contrast, the local neural representations already improve over the baseline, and introducing the speaker partition reduces the error further. The binary partition alone reaches 521 errors, accounting for most of the improvement, while adding speaker-centroid information gives the best result of 465 with TFAF.

CALLHOME shows the same pattern: the binary partition reduces DER from 3.85 to 2.90 and TFAF reaches 2.74 (Table III). DiariZen estimates the exact CALLHOME speaker count on 83.6% of files, compared with 71.7% for Equal-w-MS-Clus, yet imposing that count raises DER to 4.85. The failure therefore comes from using the estimate as a hard global constraint rather than from poor count accuracy. The fusion weight is also not sharply tuned: $\lambda = 0 . 5 , 1$ , and 2 give DERs of 3.04, 2.74, and 2.69, respectively, so the fixed value λ = 1 remains within 0.05 DER points of the best tested value.

TABLE III  
CALLHOME ABLATION UNDER IDENTICAL ORACLE SPEECH COVERAGE.
<table><tr><td>Setting</td><td>DER↓</td></tr><tr><td>Equal-w-MS-Clus</td><td>3.85</td></tr><tr><td>DiariZen</td><td>4.17</td></tr><tr><td>+ DZ speaker count</td><td>4.85</td></tr><tr><td>+ DZ binary partition affinity</td><td>2.90</td></tr><tr><td>+ DZ local+centroid affinity (TFAF)</td><td>2.74</td></tr></table>

To test whether the systems provide complementary information, we condition on the Equal-w-MS-Clus acoustic affinity and ask whether DiariZen still predicts the true speaker relation. For pairs with acoustic affinity between 0.20 and 0.25, pairs assigned to the same DiariZen speaker belong to the same reference speaker with probability 0.902, compared with only 0.017 when DiariZen assigns them to different speakers. This separation remains at least 4.6× for acoustic affinities below 0.40. The complementarity also works in the opposite direction: among pairs that DiariZen assigns to different speakers, the acoustic affinity distinguishes false splits (true same-speaker pairs) from true different-speaker pairs with an AUC of 0.858. Thus, each system provides useful information where the other is uncertain or incorrect.

We next ask which errors in the transferred partition are most harmful. We synthetically split DiariZen speaker clusters and measure how the resulting partition affects the final diarization. We summarize the severity of a split by $r _ { \mathrm { c u t } } =$ $N _ { \mathrm { c u t } } / N _ { \mathrm { s a m e } }$ the fraction of true same-speaker pairwise relations removed by the corrupted partition. Across 432 synthetic splits, the increase in attribution error is strongly correlated with $r _ { \mathrm { c u t } }$ (Spearman $\rho = 0 . 8 0 )$ . Moreover, for similar values of $r _ { \mathrm { c u t } }$ , coherent balanced splits are more damaging than small fragments because they create a plausible competing speaker partition. False merges are less harmful in our experiments, since the incorrect relations they add must compete with the existing acoustic structure rather than cutting through a true speaker cluster.

We then compare these controlled failures with DiariZen’s actual errors. Its pooled $r _ { \mathrm { c u t } }$ is only 0.018, most of its fragmentation consists of small incoherent pieces, and its false-merge rate is 0.77%. DiariZen therefore tends to make errors in the regime that our corruption experiment finds least damaging, which helps explain why its partition can improve the acoustic clustering despite being imperfect.

We next compare TFAF with alternative fusion strategies. Co-association [19], [20] first converts completed diarization outputs into hard same-speaker partitions and then reclusters them, while DOVER-Lap [18] combines completed diarization hypotheses after label alignment. TFAF instead retains the continuous Equal-w-MS-Clus acoustic graph and incorporates the DiariZen information before the final global clustering decision.

TABLE IV  
COMPARISON OF FUSION LEVELS UNDER IDENTICAL ORACLE SPEECH COVERAGE. TWO-SYSTEM CO-ASSOCIATION COMBINES EQUAL-W-MS-CLUS AND DIARIZEN; THREE-SYSTEM METHODS ADDITIONALLY USE VBX.
<table><tr><td>System</td><td>DER.25</td><td>DER0</td><td>Word err.</td><td>tcpWER</td><td>CH DER</td></tr><tr><td>Equal-w-MS-Clus</td><td>1.08</td><td>2.35</td><td>739</td><td>30.88</td><td>3.85</td></tr><tr><td>Co-association (2)</td><td>1.31</td><td>3.02</td><td>889</td><td>31.61</td><td>3.97</td></tr><tr><td>Co-association (3)</td><td>1.24</td><td>2.90</td><td>701</td><td>30.68</td><td>4.35</td></tr><tr><td>DOVER-Lap (3 systems)</td><td>0.81</td><td>2.06</td><td>665</td><td>30.46</td><td>3.18</td></tr><tr><td>TFAF</td><td>0.85</td><td>1.95</td><td>465</td><td>29.95</td><td>2.74</td></tr></table>

The co-association results show that pairwise encoding by itself is not sufficient. In particular, the two-system version first replaces the continuous Equal-w-MS-Clus graph with its hard partition and is worse than the unfused baseline on every reported AMI metric. Adding VBx improves this consensus, but it remains worse than affinity-level fusion. The important distinction is therefore not simply whether speaker relations are represented pairwise, but whether the continuous acoustic geometry of the primary system is preserved.

We also evaluated DOVER-Lap with the same two input systems, Equal-w-MS-Clus and DiariZen, where it gives no measurable improvement over Equal-w-MS-Clus on AMI. Because DOVER-Lap is more meaningful with multiple hypotheses, Table IV reports the stronger three-system configuration with an additional VBx hypothesis, which improves over Equal-w-MS-Clus on both corpora.

On AMI, three-system DOVER-Lap obtains a slightly lower collared DER than TFAF (0.81 versus 0.85). TFAF instead gives substantially fewer wrong-speaker words (465 versus 665) and lower tcpWER (29.95 versus 30.46). Removing the DER collar also reverses the ordering, from 0.81 versus 0.85 to 2.06 versus 1.95. On CALLHOME, TFAF obtains 2.74 DER compared with 3.18 for DOVER-Lap.

The difference between collared DER and the transcription metrics is concentrated near speaker boundaries. Approximately two thirds of the attribution errors occur within 250 ms of a reference speaker boundary, precisely the region excluded by the standard DER collar. Small changes in collared DER therefore need not reflect changes in speaker-attributed transcription quality.

## VI. DISCUSSION AND CONCLUSION

We presented TFAF, a training-free interface for combining neural and embedding-based diarization systems whose speaker labels and representation spaces are otherwise incompatible. TFAF transfers recording-level neural speaker structure as soft affinity evidence while preserving the continuous acoustic graph used for global clustering. It improves over both constituent systems on AMI and CALLHOME, and ablations show that most of the transferable information comes from the neural speaker partition. Future work will extend TFAF to overlapping speech.

## REFERENCES

[1] X. Anguera, S. Bozonnet, N. Evans, C. Fredouille, G. Friedland, and O. Vinyals, “Speaker diarization: A review of recent research,” IEEE Trans. Audio, Speech, Lang. Process., vol. 20, no. 2, pp. 356–370, 2012.

[2] T. J. Park, N. Kanda, D. Dimitriadis, K. J. Han, S. Watanabe, and S. Narayanan, “A review of speaker diarization: Recent advances with deep learning,” Computer Speech & Language, vol. 72, p. 101317, 2022.

[3] D. Snyder, D. Garcia-Romero, G. Sell, D. Povey, and S. Khudanpur, “X-vectors: Robust DNN embeddings for speaker recognition,” in Proc. ICASSP, 2018, pp. 5329–5333.

[4] Y. Fujita, N. Kanda, S. Horiguchi, Y. Xue, K. Nagamatsu, and S. Watanabe, “End-to-end neural speaker diarization with self-attention,” in Proc. ASRU, 2019, pp. 296–303.

[5] K. Kinoshita, M. Delcroix, and N. Tawara, “Integrating end-to-end neural and clustering-based diarization: Getting the best of both worlds,” in Proc. ICASSP, 2021, pp. 7198–7202.

[6] T. J. Park, M. Kumar, and S. Narayanan, “Multi-scale speaker diarization with neural affinity score fusion,” in Proc. ICASSP, 2021, pp. 7173– 7177.

[7] T. J. Park, N. R. Koluguri, J. Balam, and B. Ginsburg, “Multi-scale speaker diarization with dynamic scale weighting,” in Proc. Interspeech, 2022, pp. 5080–5084, doi: 10.21437/Interspeech.2022-991.

[8] T. J. Park, K. J. Han, M. Kumar, and S. Narayanan, “Auto-tuning spectral clustering for speaker diarization using normalized maximum eigengap,” IEEE Signal Process. Lett., vol. 27, pp. 381–385, 2020.

[9] T. J. Park, K. J. Han, J. Huang, X. He, B. Zhou, P. Georgiou, and S. Narayanan, “Speaker diarization with lexical information,” in Proc. Interspeech, 2019, pp. 391–395, doi: 10.21437/Interspeech.2019-1947.

[10] W. Xia, H. Lu, Q. Wang, A. Tripathi, Y. Huang, I. Lopez Moreno, and H. Sak, “Turn-to-Diarize: Online speaker diarization constrained by transformer transducer speaker turn detection,” in Proc. ICASSP, 2022, pp. 8077–8081, doi: 10.1109/ICASSP43922.2022.9746531.

[11] N. R. Koluguri, T. Park, and B. Ginsburg, “TitaNet: Neural model for speaker representation with 1D depth-wise separable convolutions and global context,” in Proc. ICASSP, 2022, pp. 8102–8106, doi: 10.1109/ICASSP43922.2022.9746806.

[12] O. Kuchaiev et al., “NeMo: A toolkit for building AI applications using neural modules,” arXiv:1909.09577, 2019.

[13] Y. Dissen, F. Kreuk, and J. Keshet, “Self-supervised speaker diarization,” in Proc. Interspeech, 2022, pp. 4013–4017, doi: 10.21437/Interspeech.2022-777.

[14] Y. Dissen, S. Harpaz, and J. Keshet, “Label-free speaker diarization using self-supervised speaker embeddings,” IEEE Trans. Audio, Speech, Language Process., 2026.

[15] J. Han, F. Landini, J. Rohdin, A. Silnova, M. Diez, and L. Burget, “Leveraging self-supervised learning for speaker diarization,” in Proc. ICASSP, 2025.

[16] F. Landini, J. Profant, M. Diez, and L. Burget, “Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: Theory, implementation and analysis on standard tasks,” Computer Speech & Language, vol. 71, p. 101254, 2022.

[17] A. Stolcke and T. Yoshioka, “DOVER: A method for combining diarization outputs,” in Proc. ASRU, 2019.

[18] D. Raj, L. P. Garcia-Perera, Z. Huang, S. Watanabe, D. Povey, A. Stolcke, and S. Khudanpur, “DOVER-Lap: A method for combining overlapaware diarization outputs,” in Proc. SLT, 2021, pp. 881–888.

[19] A. Strehl and J. Ghosh, “Cluster ensembles—A knowledge reuse framework for combining multiple partitions,” J. Mach. Learn. Res., vol. 3, pp. 583–617, 2002.

[20] B. Yin, J. Du, L. Sun, X. Zhang, S. He, Z. Ling, G. Hu, and W. Guo, “An analysis of speaker diarization fusion methods for the first DIHARD challenge,” in Proc. APSIPA ASC, 2018, pp. 1473–1477.

[21] J. I. Alvarez-Trejos, S. A. Balanya, D. Ramos, and A. Lozano-Diez, “Probabilistic fusion and calibration of neural speaker diarization models,” arXiv:2511.22696, 2025.

[22] J. Carletta et al., “The AMI meeting corpus: A pre-announcement,” in Machine Learning for Multimodal Interaction, LNCS 3869, 2006.

[23] A. Canavan, D. Graff, and G. Zipperlen, “CALLHOME American English Speech,” LDC97S42, Linguistic Data Consortium, 1997.

[24] T. von Neumann, C. Boeddeker, M. Delcroix, and R. Haeb-Umbach, “MeetEval: A toolkit for computation of word error rates for meeting transcription systems,” in Proc. CHiME, 2023.