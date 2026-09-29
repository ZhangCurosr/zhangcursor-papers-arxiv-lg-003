# ON TEMPORAL BINDING IN LARGE AUDIO LANGUAGE MODELS

Paul Primus<sup>1</sup> and Gerhard Widmer<sup>1,2</sup>

<sup>1</sup>Institute of Computational Perception, <sup>2</sup>LIT Artificial Intelligence Lab Johannes Kepler University Linz, Austria

## ABSTRACT

Reasoning about temporal structure of audio recordings requires Large Audio Language Models (LALMs) to associate sound events with their temporal position. Understanding the underlying mechanisms is a first step toward diagnosing failures and identifying model components that may need improvement. Using mechanistic interpretability, we investigate how temporal information is represented and bound to sound events in three open-source LALMs. We find that across all three, event-specific location becomes concentrated in event name representations at intermediate modality integration layers. These representations encode coarse event position along a low-dimensional, curved relative time trajectory. Steering event name representations along this trajectory systematically shifts before/after beliefs, providing evidence that these representations contribute to coarse temporal reasoning. In contrast, the same interventions do not reliably shift predicted onset timestamps, suggesting that coarse temporal reasoning and precise metric event localization rely on distinct mechanisms.

Index Terms— audio language models, mechanistic interpretability, temporal reasoning, multimodal representation

## 1. INTRODUCTION

Temporal reasoning over an audio recording requires Large Audio Language Models (LALMs) to associate sound events with their corresponding temporal properties. Determining whether footsteps occurred before or after a vacuum cleaner, for example, requires recognizing both events, associating each with a coarse temporal position, and comparing these event-bound representations. More fine-grained localization of events would additionally require identifying event boundaries and mapping them onto a continuous time axis. Recent LALMs have demonstrated progress on temporal audio reasoning benchmarks, performing well on coarse reasoning tasks while still exhibiting notable shortcomings [1, 2, 3, 4]. Despite the progress, it remains unclear how LALMs internally bind temporal positions to events and which hidden activations are primarily involved in temporal reasoning. Understanding these internal mechanisms can help localize the source of temporal reasoning failures and indicate which model components may need improvement. Mechanistic interpretability methods, such as probing [5], causal tracing [6], activation patching [7], and representation steering [8], provide tools for this by revealing where information is encoded and how it causally affects model behavior. Recent studies of LALMs have used mechanistic interventions to investigate audio–text fusion [9] and identify audio-specialized attention heads that strengthen audio grounding [10]. Kang et al. [11] identified low-dimensional spatiotemporal representations in vision- and video-language models, showing that location information becomes bound to textual object activations and causally influences model predictions. We investigate whether analogous event-bound temporal representations emerge in LALMs and whether they support both coarse temporal reasoning and precise event localization. We study AF-Next [12], MOSS-Audio [13], and Qwen3-Omni [14] using activation swapping and representation analysis to characterize these representations and test their causal role. Our results suggest a common mechanism for coarse temporal reasoning, while precise localization appears to rely on a separate mechanism.

## 2. EXPERIMENTAL SETUP

For our controlled experiments, we construct synthetic sound scenes from ESC-50 foreground events [15] and FreeSound backgrounds [16]. Foreground clips are trimmed of leading and trailing silence and mixed into the backgrounds at SNRs between approximately −5 and 0 dB. For each background, ESC-50 classes detected by an audio tagger are excluded from insertion to reduce overlap with preexisting events. We construct disjoint development and test sets for the synthetic experiments. We additionally use subsets of RealDESED [17] for validating our experiments on natural audio. RealDESED is well suited for this purpose, as it contains longer audio recordings of 15–30 s duration with strong temporal annotations for 15 sound event classes. Depending on the experiment, we select recordings containing either two clearly ordered events suitable for before/after reasoning or a clearly identifiable event with limited temporal overlap with other events. Code and details for reproducing our experiments are available at https://github.com/OptimusPrimus/ icassp2027\_temporal\_binding.

![](images/06fa72cec441949fe8d2c6547a0a27518b855140ddb2d78457172f37981602ab.jpg)  
Fig. 1. Activation swapping across layers for three audio language models. Each line shows the belief shift $S _ { L }$ when residual stream activations for a specific token group are replaced with activations from a second forward pass using the same text query but audio with the event order reversed. Values near 1 indicate a strong shift, while values near 0 indicate little effect.

## 3. TEMPORAL INFORMATION IS BOUND TO EVENT NAME REPRESENTATIONS

We first identify where event-specific temporal information enters the text residual stream and then test how precisely event position can be decoded from the representations.

## 3.1. Intermediate Layers Bind Temporal Information to Event Name Tokens

Following Kang et al. [11], we ask where event-specific temporal information enters textual representations and causally contributes to temporal reasoning. Using our synthetic dataset, we construct pairs of 10 s recordings containing the same background and two ESC-50 events in mirrored temporal arrangements. Let $c _ { q }$ and $c _ { r }$ denote the query and reference events. If recording x contains $c _ { q }$ before $c _ { r }$ , its paired recording y contains the same events in the opposite order. Both recordings are paired with the identical prompt: “Does [query event] occur before or after [reference event]?” At each layer, we replace subsets of residual stream activations for x with the corresponding activations from $y$ and continue the forward pass. If the replaced activations contribute event-specific temporal information, the intervention should shift the model’s output from its belief for x toward its belief for $y .$ We quantify this using a normalized belief shift $S _ { L }$ . Under teacher forcing, we condition on the response prefix “[query event] occurs” and measure the probabilities of “before” and “after”. Let $a ^ { \star } \in$ {before, after} denote the ground truth relation in x, and let $p _ { z } ( a ^ { \star } )$ denote the probability assigned to this answer for input z. For an intervention at layer L,

$$
S _ { L } = \frac { p _ { x } ( a ^ { \star } ) - p _ { \widetilde { x } _ { L } } ( a ^ { \star } ) } { p _ { x } ( a ^ { \star } ) - p _ { y } ( a ^ { \star } ) } ,\tag{1}
$$

where $\widetilde { x } _ { L }$ denotes the intervened forward pass. Thus, $S _ { L } = 0$ indicates no effect, whereas $S _ { L } = 1$ reproduces the full belief change induced by the mirrored recording.

Figure 1 compares interventions for selected token groups. Across compared models, a consistent information flow pattern emerges: In early layers, swapping the audio representations (blue) strongly affects the model’s temporal belief, whereas swapping all text tokens (orange) has little effect.

In an intermediate layer range, the audio influence decreases while swapping only the event name tokens (green) produces a substantial belief shift. At later layers, the event name effect fades and interventions on the full textual sequence become dominant, with interventions on the final token (pink) eventually having the strongest impact of all individual tokens. The “before or after” control tokens (grey) remain close to zero. These results suggest that event-specific temporal information becomes bound to textual representations in intermediate event name activations; swapping these representations shifts the predictions, which indicates that they are causally involved in temporal reasoning.

## 3.2. Coarse Temporal Position Is Decodable from Event Name Representations

We next test how precisely an event’s temporal position can be recovered from these event name representations. To this end, we construct 30 s synthetic recordings containing two distinct ESC-50 events at random temporal locations. One event is selected as the query event, and the model is prompted with “Is there [query event]?”. Let $\mathbf { 1 } _ { L , q } ( x ) \in \mathbb { R } ^ { d }$ denote the layer-L residual stream activation of the final token of queried event $c _ { q } ,$ , and $\mathbf { h } _ { L , 1 } ( x )$ that of the first text prompt token, i.e., the one corresponding to $\mathrm { ^ { 6 6 } I s } ^ { , 5 }$ . We train a MLP regressor with one hidden layer to predict event midpoints of the query and the unmentioned event $( t _ { q }$ and $t _ { o } )$ from these activations at each layer on the synthetic training set. To test whether the decoded temporal signal generalizes across event identity, we use a five-fold cross-validation, training on 40 query classes and evaluating on the remaining 10.

Figure 2 shows that the queried event midpoint $t _ { q }$ is most accurately decoded from its own event name representation $\mathbf { h } _ { L , q } ( x )$ (blue), reaching an MAE of roughly 3.5–4 s at intermediate layers. Predicting the unmentioned event’s time $t _ { o }$ from the same representation (orange) or the queried event’s time from the first query token $ { \mathbf { h } } _ { L , 1 } ( x )$ (green) yields consistently higher errors. The peak in decodability overlaps with the modality integration region identified by activation swapping, indicating that intermediate event name representations contain a coarse, event-specific temporal representation that generalizes across sound classes.

![](images/2bcf7b4998baed779cbe411f7bd43dc7a0448a73709c833a3019e65c6081f76b.jpg)  
Fig. 2. Layerwise MAE for decoding query- and unmentioned-event midpoints $( t _ { q } , t _ { o } )$ from token activations ${ \bf h } _ { L , q }$ and h $^ { ! } L , 1$ .

## 4. GEOMETRY OF EVENT-BOUND TEMPORAL ENCODING

We next extract temporal IDs—shared representations of events at the same temporal position—and analyze their geometry.

## 4.1. Extracting Temporal IDs

We derive these IDs from the model’s hidden event name representations when given 30 s synthetic recordings containing a single ESC-50 event, paired with the query “Is there [query event]?”. Let $\mathbf { h } _ { L , q } ( x )$ denote the layer-L representation of the queried event name token, and let $\mu _ { L , c }$ be its training set mean for class c. We group examples into 2.5 s bins $B _ { k }$ based on the event’s midpoint and define the temporal ID of bin k as the mean class-centered activation,

$$
\pmb { \tau } _ { L , k } = \mathbb { E } \Big [ \mathbf { h } _ { L , q } ( \boldsymbol { x } ) - \pmb { \mu } _ { L , c _ { q } } \mid t _ { q } \in B _ { k } \Big ] .\tag{2}
$$

Class centering removes variation dependent on event identity, so the resulting IDs primarily capture variation associated with temporal position. Bins with insufficient support are discarded, and the retained IDs are stacked into $\mathbf { T } _ { L } \in \mathbb { R } ^ { K \times d }$ for the geometric analysis below.

## 4.2. Temporal IDs Lie on a Low-Dimensional Curved Trajectory

To characterize their geometry, we perform principal component analysis on $\mathbf { T } _ { L }$ at representative modality integration layers identified in Section 3.1. Figure 3 shows the resulting geometry. Temporal variation is strongly concentrated in a low-dimensional subspace: the first two principal components explain 83.9%, 86.4%, and 86.5% of temporal ID variance for AF-Next, MOSS-Audio, and Qwen3-Omni, respectively. As event time increases, the temporal IDs trace a curved trajectory through this subspace. Across models, early and late temporal IDs are strongly anti-aligned, with endpoint cosine similarities between −0.71 and −0.73, while ID magnitude decreases toward the recording midpoint and increases again toward the boundaries. This is consistent with a signed early-to-late component that passes close to the class-centered origin, together with weak off-axis variation producing the observed curvature.

![](images/ab50db11bbd6970659ffedb370c9c43979f2bdf3fe2224bb8aec87542f2bd471.jpg)

![](images/a443994e98aa01cebd74c82c011489356840da39756dcf0aaa6f07c658eb9b9f.jpg)

![](images/3b827a42b32442ff46d880cd1178a54c4e1c38ef7f63a7a66c443fbce24570f0.jpg)  
Fig. 3. Temporal ID geometry at representative modality integration layers. Points show temporal IDs projected onto the first two PCs.

Because the temporal IDs are extracted from fixed 30 s recordings, it is unclear whether they represent absolute time or position relative to the recording duration. We test this using variable-duration RealDESED recordings. We classcenter the event name activations, project them onto the first two PCs of the synthetic temporal IDs, and fit their projections as a function of either absolute midpoint $t _ { q }$ or relative position $r _ { q } = t _ { q } / T _ { x }$ . Relative position consistently provides the better fit: multivariate $R ^ { 2 }$ increases from 0.19 to 0.43 for AF-Next, 0.31 to 0.53 for MOSS-Audio, and 0.30 to 0.52 for Qwen3-Omni.

## 5. EVENT-BOUND TEMPORAL ENCODING CAUSALLY MEDIATES COARSE REASONING

We next test whether temporal IDs affect temporal reasoning by steering activations and measuring prediction shifts.

## 5.1. Temporal ID Steering

For each model, we reconstruct the temporal ID trajectory from the fixed-length synthetic training data at the modality integration layer selected above. For a queried event occurring at relative time $r _ { q } = t _ { q } / T$ , we interpolate this trajectory to obtain its temporal $\mathrm { I D } \tau _ { L } ( r _ { q } )$ . We then define an examplespecific steering direction from the queried event’s current position toward either endpoint,

$$
{ \bf d } _ { L } ( x ) = \frac { \tau _ { L } ^ { \mathrm { t a r g e t } } - \tau _ { L } ( r _ { q } ) } { \left. \tau _ { L , 1 } - \tau _ { L , K } \right. _ { 2 } } ,\tag{3}
$$

![](images/741b62df7dfeac0f3e1f56982b8d724f0850edb455cac93d882dd62bbadbea11.jpg)  
Fig. 4. Relation steering. Before (B) and After (A) denote groups which were predicted as before/after prior to the intervention. Steering shifts probability toward the target relation.

where the target is the beginning or end of the synthetic temporal ID trajectory. Forward intervention uses target $\tau _ { L , K }$ to steer the belief of event occurrence to the end of the recording, and a backward intervention uses target $\tau _ { L , 1 }$ to move it toward the beginning. During inference, we add this direction to the query’s event name tokens,

$$
\widetilde { \mathbf { h } } _ { L , i } ( x ) = \mathbf { h } _ { L , i } ( x ) + \alpha \rho _ { L } \mathbf { d } _ { L } ( x ) ,\tag{4}
$$

where α controls intervention strength and $\rho _ { L }$ is the norm of the selected token activations.

## 5.2. Steering Before/After Beliefs

We first evaluate queries of the form “Does [query event] occur before or after [reference event]?”. We use teacher forcing with prefix “[query event] occurs” and measure the change in probability distribution over next tokens. Figure 4 shows that steering shifts probability mass toward the intended relation across all three models: forward steering increases p(after), whereas backward steering increases p(before). Examples already aligned with the intervention—predicted as “after” before forward steering or “before” before backward steering— show only moderate probability shifts, as their predictions are often already saturated. Their original before/after predictions are preserved in 99.5% of cases. In contrast, predictions opposing the intervention shift strongly toward the intended relation and are frequently flipped, with flip rates of 82% for AF-Next, 84.2% for MOSS-Audio, and 70.7% for Qwen3- Omni. Thus, moving the queried event along the temporal ID trajectory consistently changes both probability mass and discrete before/after judgments on natural audio.

## 5.3. Steering Precise Sound Event Detection

We next test whether temporal IDs also mediate precise onset prediction on a subset of RealDESED. To this end we select recordings in which the queried event class occurs exactly once. Baseline onset detection MAEs are 2.01 s for MOSS-Audio, 2.21 s for Qwen3-Omni, and 6.48 s for AF-Next. AF-Next is strongly biased toward a few onset values: 56.1% and

![](images/d57e585f81029a24f0e1a7c2a83fc2b4a20589fb13f0e07fb547787482e4da67.jpg)  
Fig. 5. Onset steering. Early (E) and Late (L) examples were predicted as starting in the first and second half of the recording before the intervention.

13.8% of predictions are $^ { 0 , \mathrm { s } }$ and 1.5,s. We apply the same endpoint-directed intervention and compare predicted onsets before and after steering.

As shown in Figure 5, steering does not induce a consistent timestep displacement. AF-Next and MOSS-Audio respond asymmetrically: forward steering has little effect, whereas backward steering can substantially shift late predictions. For AF-Next, the backward intervention largely resets predictions to 0 s. Qwen3-Omni shows weaker but more bidirectional shifts toward the target. These asymmetric, modeldependent effects suggest that temporal IDs encode a coarse position which is only able to bias or override a separate finegrained detection mechanism.

## 6. DISCUSSION AND CONCLUSION

Our activation swapping experiments showed that across the three investigated LALMs, event-specific temporal information becomes bound to event name representations in intermediate modality integration layers. These representations appear to encode event position on a low-dimensional trajectory that reflects relative position of the event within the recording. Our activation swapping and before/ after steering results provide evidence that this temporal information is involved in temporal reasoning. However, the temporal IDs do not appear to encode precise event timestamps: We find that nonlinear decoding predicts event midpoints with an MAE of around 3.5–4 s, which is too coarse for precise localization. Furhtermore, steering with these IDs does not induce consistent shifts in predicted event onsets. Together, these results suggest that LALMs use event-bound temporal IDs primarily for coarse temporal reasoning, while relying on additional mechanisms for precise event localization.

Our analysis is limited to three models, a small set of query formulations, and relatively simple acoustic scenes with isolated query events and without strong overlaps. Future work should extend this analysis to more challenging acoustic conditions and investigate the mechanisms underlying fine-grained temporal localization.

## 7. ACKNOWLEDGMENT

The LIT AI Lab is supported by the Federal State of Upper Austria. GPT-5.6 Sol was used to assist with language editing and with writing code for the experiments in this manuscript. All AI-generated text and code was reviewed, verified, and revised by the authors.

## 8. REFERENCES

[1] Sreyan Ghosh, Ashish Seth, Sonal Kumar, Utkarsh Tyagi, Chandra Kiran Evuru, Ramaneswaran S, S Sakshi, Oriol Nieto, Ramani Duraiswami, and Dinesh Manocha, “CompA: Addressing the gap in compositional reasoning in audio-language models,” in International Conference on Learning Representations, 2024.

[2] Debarpan Bhattacharya, Apoorva Kulkarni, and Sriram Ganapathy, “Benchmarking and confidence evaluation of LALMs for temporal reasoning,” in Interspeech, 2025, pp. 2068–2072.

[3] S Sakshi, Utkarsh Tyagi, Sonal Kumar, Ashish Seth, Ramaneswaran Selvakumar, Oriol Nieto, Ramani Duraiswami, Sreyan Ghosh, and Dinesh Manocha, “MMAU: A massive multi-task audio understanding and reasoning benchmark,” in International Conference on Learning Representations, 2025.

[4] Apoorva Kulkarni, Kaousheik Jayakumar, Sreyan Ghosh, Sarah Wiegreffe, Dinesh Manocha, and Ramani Duraiswami, “A closer look at failure modes in temporal understanding of large audio-language models,” in Interspeech, 2026, pp. 893–897.

[5] Guillaume Alain and Yoshua Bengio, “Understanding intermediate layers using linear classifier probes,” in International Conference on Learning Representations Workshop, 2017.

[6] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov, “Locating and editing factual associations in GPT,” in Advances in Neural Information Processing Systems, 2022, vol. 35, pp. 17359–17372.

[7] Fred Zhang and Neel Nanda, “Towards best practices of activation patching in language models: Metrics and methods,” in International Conference on Learning Representations, 2024.

[8] Kenneth Li, Oam Patel, Fernanda Viegas, Hanspeter´ Pfister, and Martin Wattenberg, “Inference-time intervention: Eliciting truthful answers from a language model,” in Advances in Neural Information Processing Systems, 2023, vol. 36, pp. 41451–41530.

[9] Wei-Chih Chen, Chien yu Huang, and Hung yi Lee, “Causal tracing of audio-text fusion in large audio language models,” arXiv preprint arXiv:2603.13768, 2026.

[10] Neta Glazer, Lenny Aharon, and Ethan Fetaya, “Are audio-language models listening? audio-specialist heads for adaptive audio steering,” in Interspeech, 2026.

[11] Raphaela Kang, Hongqiao Chen, Georgia Gkioxari, and Pietro Perona, “Linear mechanisms for spatiotemporal reasoning in vision language models,” in International Conference on Learning Representations, 2026.

[12] Sreyan Ghosh, Arushi Goel, Kaousheik Jayakumar, Lasha Koroshinadze, Nishit Anand, Zhifeng Kong, Siddharth Gururani, Sang gil Lee, Jaehyeon Kim, Aya Aljafari, Chao-Han Huck Yang, Sungwon Kim, Ramani Duraiswami, Dinesh Manocha, Mohammad Shoeybi, Bryan Catanzaro, Ming-Yu Liu, and Wei Ping, “Audio flamingo next: Next-generation open audio-language models for speech, sound, and music,” arXiv preprint arXiv:2604.10905, 2026.

[13] Chen Yang, Chufan Yu, Hanfu Chen, Jie Zhu, Jingqi Chen, Ke Chen, Wenxuan Wang, Yang Wang, Yaozhou Jiang, Yi Jiang, Zhengyuan Lin, Ziqi Chen, Zhaoye Fei, Chenghao Liu, Jun Zhan, Kang Yu, Kexin Huang, Mingshu Chen, Qinyuan Cheng, Ruixiao Li, Shimin Li, Songlin Wang, Yang Gao, Yiyang Zhang, and Xipeng Qiu, “MOSS-Audio technical report,” arXiv preprint arXiv:2606.01802, 2026.

[14] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, and Junyang Lin, “Qwen3-Omni technical report,” arXiv preprint arXiv:2509.17765, 2025.

[15] Karol J. Piczak, “ESC: Dataset for environmental sound classification,” in Proceedings of the ACM International Conference on Multimedia, 2015, pp. 1015–1018.

[16] Frederic Font, Gerard Roma, and Xavier Serra, “Freesound technical demo,” in Proceedings of the ACM International Conference on Multimedia, 2013, pp. 411–412.

[17] Florian Schmid, Paul Primus, Alexander Fichtinger, Tara Jadidi, Tobias Morocutti, and Gerhard Widmer, “RealDESED: A real-world domestic sound event detection benchmark,” arXiv preprint arXiv:2607.16736, 2026.