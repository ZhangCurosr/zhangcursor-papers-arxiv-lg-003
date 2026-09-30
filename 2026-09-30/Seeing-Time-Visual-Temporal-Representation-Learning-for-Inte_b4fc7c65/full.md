# Seeing Time: Visual-Temporal Representation Learning for Interpretable Time Series Clustering

Zheng Zhu<sup>1</sup>, Zexi Tan<sup>2,∗</sup>, Yuming Deng<sup>2</sup> and Yiqun Zhang<sup>3</sup>

<sup>1</sup>Huizhou No. 1 High School (High School Student), <sup>2</sup>Guangdong University of Technology <sup>3</sup>South China University of Technology

zhuz3398199086@foxmail.com, {tanzexi, dengyuming}@mails.gdut.edu.cn, yqzhangzyq@gmail.com

Abstract—Multivariate Time Series (MTS) clustering is an important tool in temporal data mining, aiming to discover latent group structures from complex observations without supervision. Although existing deep clustering methods can learn discriminative temporal representations, the resulting latent clusters are often difficult to relate back to waveform characteristics that practitioners can directly inspect and compare, limiting their ability to assess whether the discovered patterns reflect meaningful temporal behaviors. This paper, therefore, proposes WAVE (Waveform Aligned Visual-temporal Embedding), which treats time series and their deterministically rendered waveform plots as complementary views of the same observations. To produce discriminative representations whose cluster structures can be traced to observable waveform characteristics, WAVE aligns and integrates fine-grained temporal variations with holistic visual patterns, while associating each discovered cluster with its centroid-nearest authentic sample. Accordingly, interpretability in this work specifically refers to waveform-level traceability rather than a general explanation of model decisions. Extensive evaluations across 10 real-world public datasets show that WAVE achieves the highest macro-averaged clustering performance and the best average rank among the compared methods, while qualitative case studies illustrate how the discovered clusters can be inspected through authentic waveform records. The source code is available at https://github.com/Zheng-Zhu1/WAVE.

Index Terms—Time Series, Interpretable Clustering, Visual-Semantic Enhancement, Representation Learning

## I. INTRODUCTION

Multivariate Time Series (MTS) clustering [1] is widely applied in industrial monitoring [2], medical diagnosis [3], human activity recognition [4], and environmental sensing [5], aiming to discover latent groups and representative patterns from complex records when expert annotations are scarce or costly to obtain. In recent years, deep clustering methods have substantially improved pattern discovery in complex data by learning discriminative temporal representations [6]. However, these methods primarily rely on time series for representation learning [7], making the resulting abstract cluster structures difficult to relate to waveform characteristics that can be intuitively observed and compared [8]. Consequently, users struggle to determine how different clusters vary in terms of overall contours, peak-valley distributions, and periodic morphology, and also find it difficult to validate the clustering results against the original time series records [9]. Therefore, how to make the discovered cluster structures traceable to observable waveform characteristics while preserving clustering discriminability remains an urgent MTS clustering problem [10]. In this work, interpretability specifically refers to waveform-level traceability: the ability to associate discovered clusters with authentic waveform records that can be directly inspected and compared.

Recent methods primarily improve the discriminability of MTS cluster structures by encoding local variations, long-term dependencies, and cross-channel relationships from raw sequences through reconstruction [11], contrastive learning [12], or clustering objectives [13]. Yet discriminability alone does not make the discovered structures traceable to observable temporal patterns. This limitation motivates two related directions:model interpretation [14] and result visualization [15]. Nevertheless, their optimization processes and final clustering remain confined to the feature space, and the resulting clusters can usually be evaluated only through clustering metrics or dimensionality-reduced distributions [16]. Regarding model interpretation, some studies employ saliency scores [17] or prototype representations [18] to identify local factors that influence model outputs, but they still struggle to understand and compare different clusters in terms of waveform characteristics such as overall contours, peak-valley distributions, and periodic morphology. As for result visualization, waveform plots are typically used only to display clustering results and do not participate in representation learning or the formation of cluster structures [19]. Therefore, existing studies still lack a unified approach that directly incorporates observable waveform characteristics into representation learning and establishes connections among sequence data, cluster structures, and real waveforms [20].

This gap raises a more fundamental question: whether waveform morphology can participate in clustering representation learning rather than remain merely an external basis for interpreting the results [9]. Waveform plots explicitly expose temporal characteristics such as global contours, peakvalley distributions, and periodic morphology [21], providing observable structural evidence complementary to sequence representations. Meanwhile, recent studies have shown that transforming structured non-visual data into images allows pretrained visual models to capture holistic patterns that are difficult to characterize solely from raw sequences [22].

![](images/8ef07728f7f87235c747d7e317d70e34929d736d5ce55bb00e40adadda5c31e7.jpg)  
Fig. 1. Overview of WAVE. (A) Each MTS record, its weakly augmented sequence, and its deterministically rendered waveform image are encoded into temporal, augmented, and visual representations $\mathbf { Z } ^ { t } , \mathbf { Z } ^ { a }$ , and Z<sup>v</sup>. (B) Cross-modal contrastive alignment maps the temporal and visual representations into a shared space, followed by parameter-free normalized equal fusion to obtain Z. The fused representations are clustered, and the real sample nearest to each centroid is selected as an observable cluster representative.

These findings reveal a potential complementarity: sequence encoders preserve fine-grained temporal variations, whereas visual encoders provide a more holistic description of waveform morphology. Realizing this complementarity is nontrivial because the two encoders organize information in distinct representational geometries [23], and the informativeness of their representations may vary across samples [24]. Simply combining them cannot guarantee a coherent clustering representation. Thus, the central challenge is to incorporate observable waveform characteristics into cluster formation while establishing correspondence between the two representational spaces and preserving their respective strengths [25].

According to this observation, we propose WAVE (Waveform Aligned Visual-temporal Embedding), a visual–temporal framework for observable MTS clustering. To preserve finegrained measurements while incorporating waveform morphology, WAVE represents each record as a sequence and a multichannel waveform plot, capturing temporal variations and holistic patterns, respectively. Because the two views are encoded independently, their shared origin does not guarantee representational consistency. We therefore employ cross-modal contrastive learning to align the paired views, while temporal augmentation improves representation stability. Furthermore, the aligned representations are integrated through parameterfree normalized equal fusion. Finally, WAVE selects the record nearest to each cluster center as its representative waveform, making abstract clusters directly observable and comparable. The main contributions are summarized below.

• We propose WAVE, a visual-temporal MTS clustering framework that addresses the limited interpretability of existing methods primarily optimized for clustering performance by incorporating observable waveform morphology directly into representation learning and cluster formation.

• WAVE grounds each learned cluster in an authentic waveform by selecting the real record nearest to its centroid. This association makes cluster structures observable in the original data space and supports direct comparison

among discovered groups.

• By coupling temporal discrimination with observable waveform characteristics, WAVE produces accurate and interpretable cluster structures. Experiments on 10 realworld datasets verify its clustering effectiveness and the visual traceability of the discovered groups.

## II. PROPOSED METHOD

Given an unlabeled MTS dataset ${ \mathcal { X } } ~ = ~ \{ { \bf X } _ { i } \} _ { i = 1 } ^ { N }$ , where $\mathbf { X } _ { i } \in \mathbb { R } ^ { T \times C }$ contains C variables over T time steps, WAVE learns a representation for clustering, as illustrated in Fig. 1. An MTS record contains both fine-grained temporal dynamics and global waveform morphology, which may not be captured equally well by a single encoder. Therefore, WAVE represents each record through two paired views: the original sequence and its annotation-free waveform image. The former preserves sequential dependencies, while the latter makes geometric patterns such as peaks, slopes, and periodicity directly observable.

## A. Temporal-Visual Waveform Representation

Sequence-only representations can capture temporal dynamics, but their latent cluster structures are difficult to relate to observable waveform characteristics. To preserve discriminative temporal information while grounding the learned clusters in directly inspectable patterns, WAVE constructs two complementary views of each record. The sequence view retains temporal ordering and value variations, while the waveform view exposes structural morphology. They are encoded separately and subsequently aligned in a shared space.

For the temporal branch, discriminative patterns may range from short local changes to dependencies spanning the entire sequence. Each variable is first normalized independently to prevent scale differences from dominating the representation. Multi-scale one-dimensional convolutions then capture local patterns at different temporal ranges, and a Transformer encoder models their long-range relationships. Attention pooling summarizes globally informative regions, while max pooling preserves salient local responses. This design produces a representation sensitive to both distributed temporal context and brief but distinctive events:

$$
\mathbf { z } _ { i } ^ { t } = \mathrm { n o r m } \left( p _ { t } \left( f _ { t } ( \mathbf { X } _ { i } ) \right) \right) \in \mathbb { R } ^ { d } ,\tag{1}
$$

where $f _ { t } ( \cdot )$ and $p _ { t } ( \cdot )$ denote the temporal encoder and projection head, respectively.

However, the temporal embedding remains latent and does not directly expose the waveform morphology underlying the learned clusters. WAVE therefore renders each normalized record into an annotation-free waveform image $\mathbf { I } _ { i } = \mathcal { R } ( \mathbf { X } _ { i } )$ using a fixed rendering scheme, such that visual differences mainly reflect waveform structure. A frozen pretrained Open-CLIP ViT-B/32 image encoder extracts morphological patterns, and a trainable projection head maps them into the representation space:

$$
\mathbf { z } _ { i } ^ { v } = \operatorname { n o r m } \left( p _ { v } \left( f _ { v } ( \mathbf { I } _ { i } ) \right) \right) .\tag{2}
$$

Although both temporal and visual views describe the same record, their embeddings are produced by heterogeneous encoders and therefore remain geometrically inconsistent. Thus, explicit cross-modal alignment is required before complementary signatures can be reliably fused for clustering.

## B. Contrastively Aligned Waveform Fusion

To establish the required cross-view correspondence, WAVE treats $( \mathbf { z } _ { i } ^ { t } , \mathbf { z } _ { i } ^ { v } )$ from the same record as a positive pair and cross-record pairs as negatives. A symmetric cross-modal contrastive objective increases the similarity of matched pairs while separating mismatched ones, encouraging each record to preserve its identity across the two modalities. Consequently, proximity in the aligned space reflects agreement between fine-grained temporal dynamics and observable waveform morphology, providing a consistent basis for their subsequent fusion.

The aligned normalized embeddings are then combined with equal coefficients:

$$
\mathbf { z } _ { i } = \mathrm { n o r m } \left( \mathbf { z } _ { i } ^ { t } + \mathbf { z } _ { i } ^ { v } \right) .\tag{3}
$$

This parameter-free fusion allows the clustering space to reflect both fine-grained temporal dynamics and observable waveform morphology. Consequently, waveform structure participates directly in determining sample proximity and cluster formation.

## C. Optimization and Clustering

Training couples the temporal-visual alignment established above with temporal consistency under weak augmentation. The overall objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { t v } } + \lambda \mathcal { L } _ { \mathrm { t a } } , } \end{array}\tag{4}
$$

where $\mathcal { L } _ { \mathrm { t v } } = \mathcal { L } _ { \mathrm { N C E } } ( \mathbf { Z } ^ { t } , \mathbf { Z } ^ { v } )$ establishes temporal-visual correspondence, while ${ \mathcal { L } } _ { \mathrm { t a } } = { \mathcal { L } } _ { \mathrm { N C E } } ( \mathbf { Z } ^ { t } , \mathbf { Z } ^ { a } )$ constrains weakly augmented sequences to remain close to their original representations. The coefficient λ controls the relative strength of $\mathcal { L } _ { \mathrm { t a } }$ .

To specify these contrastive objectives, let $\alpha , \beta \in \{ t , v , a \}$ index the temporal, visual, and augmented representations, and let $i , j \in \{ 1 , \ldots , B \}$ index samples within a batch. The directional loss is

$$
\ell _ { \alpha  \beta } ^ { ( i ) } = - \log \frac { \exp \Big ( s ( \mathbf { z } _ { i } ^ { \alpha } , \mathbf { z } _ { i } ^ { \beta } ) / \tau \Big ) } { \sum _ { j = 1 } ^ { B } \exp \Big ( s ( \mathbf { z } _ { i } ^ { \alpha } , \mathbf { z } _ { j } ^ { \beta } ) / \tau \Big ) } ,\tag{5}
$$

where $s ( \cdot , \cdot )$ denotes cosine similarity and τ is the temperature. WAVE optimizes both directions through the symmetric objective

$$
\mathcal { L } _ { \mathrm { N C E } } ( \mathbf { Z } ^ { \alpha } , \mathbf { Z } ^ { \beta } ) = \frac { 1 } { 2 B } \sum _ { i = 1 } ^ { B } ( \ell _ { \alpha  \beta } ^ { ( i ) } + \ell _ { \beta  \alpha } ^ { ( i ) } ) .\tag{6}
$$

After optimization, the fused representations defined in the preceding subsection are standardized and partitioned by minimizing the k-means objective

$$
\operatorname* { m i n } _ { \{ \pmb { \mu } _ { k } \} , \{ \pmb { q } _ { i } \} } \sum _ { i = 1 } ^ { N } \left\| \widetilde { \mathbf z } _ { i } - \pmb { \mu } _ { q _ { i } } \right\| _ { 2 } ^ { 2 } ,\tag{7}
$$

where $\widetilde { \mathbf { z } } _ { i }$ is the standardized fused representation, $q _ { i }$ is its cluster assignment, and $\pmb { \mu } _ { k }$ denotes the centroid of k-th cluster.

To connect each latent cluster back to an observable waveform, WAVE selects the authentic sample closest to its centroid as the cluster exemplar:

$$
e _ { k } = \arg \operatorname* { m i n } _ { i : q _ { i } = k } \| \widetilde { \mathbf { z } } _ { i } - \pmb { \mu } _ { k } \| _ { 2 } .\tag{8}
$$

The corresponding original record ${ \bf X } _ { e _ { k } }$ provides a directly inspectable waveform for the discovered cluster.

## III. EXPERIMENTS

This paper evaluates WAVE through clustering, ablation, and interpretability experiments on ten UEA MTS datasets [26]: AtrialFibrillation, BasicMotions, Cricket, ERing, Epilepsy, HandMovementDirection, Libras, NATOPS, Racket-Sports, and StandWalkJump. Following the transductive protocol, training and test sets are combined for unsupervised learning, with labels used only for evaluation and the cluster number set to the ground-truth class count. The baselines cover both recent clustering-specific approaches, including EMTC [1], TFMCC [10], FCACC [7], MVCIMTS [16], and k-Graph [19], and general time series representation learners, including GTM [11], FEI [13], TimesURL [12], and UNITS [21]. Performance is measured by ACC, NMI, and ARI. Ablations examine the two representation views and training objectives, while the interpretability study evaluates waveform exemplars associated with the discovered clusters. We report mean and standard deviation over five seeds. WAVE is implemented in PyTorch 2.8.0 and trained for up to 100 epochs on an RTX 5090 with batch size 32 and $\lambda = 0 . 3$ The temporal encoder contains six Transformer layers and eight heads, OpenCLIP ViT-B/32 remains frozen, and the embedding dimension is 256.

TABLE I  
AGGREGATE CLUSTERING RESULTS ON TEN UEA DATASETS. AVG. AND RANK DENOTE MACRO-AVERAGE PERFORMANCE AND AVERAGE RANK, RESPECTIVELY. LOWER RANKS ARE BETTER; DARK AND LIGHT BLUE INDICATE THE BEST AND SECOND-BEST RESULTS.
<table><tr><td rowspan="2">Methods</td><td colspan="2">ACC</td><td colspan="2">NMI</td><td colspan="2">ARI</td></tr><tr><td>Avg.</td><td>Rank</td><td>Avg.</td><td>Rank</td><td>Avg.</td><td>Rank</td></tr><tr><td>TimesURL (AAAI&#x27;24)</td><td>0.499</td><td>5.100</td><td>0.395</td><td>4.950</td><td>0.280</td><td>4.900</td></tr><tr><td>UNITS (NeurIPS’24)</td><td>0.418</td><td>7.300</td><td>0.239</td><td>7.700</td><td>0.134</td><td>7.700</td></tr><tr><td>FEI (AAAI&#x27;25)</td><td>0.511</td><td>5.050</td><td>0.362</td><td>5.300</td><td>0.252</td><td>5.250</td></tr><tr><td>MVCIMTS (INFFUS’25)</td><td>0.291</td><td>9.450</td><td>0.116</td><td>8.050</td><td>0.060</td><td>8.200</td></tr><tr><td>k-Graph (TKDE&#x27;25)</td><td>0.480</td><td>6.050</td><td>0.310</td><td>6.400</td><td>0.208</td><td>6.500</td></tr><tr><td>EMTC (AAAI&#x27;26)</td><td>0.544</td><td>3.500</td><td>0.374</td><td>4.600</td><td>0.230</td><td>4.800</td></tr><tr><td>TFMCC (AAAI&#x27;26)</td><td>0.578</td><td>4.150</td><td>0.453</td><td>4.050</td><td>0.347</td><td>3.700</td></tr><tr><td>FCACC (PR’26)</td><td>0.531</td><td>5.950</td><td>0.400</td><td>4.500</td><td>0.292</td><td>5.200</td></tr><tr><td>GTM (ICLR’26)</td><td>0.448</td><td>6.300</td><td>0.262</td><td>7.350</td><td>0.156</td><td>6.600</td></tr><tr><td>WAVE (Ours)</td><td>0.703</td><td>2.150</td><td>0.587</td><td>2.100</td><td>0.520</td><td>2.150</td></tr></table>

TABLE II

ABLATION RESULTS AVERAGED OVER TEN UEA DATASETS. DROP DENOTES FULL WAVE MINUS THE CORRESPONDING VARIANT. <sup>∗</sup> INDICATES A SIGNIFICANT DEGRADATION ACCORDING TO A TWO-SIDED WILCOXON SIGNED-RANK TEST WITH HOLM CORRECTION (p < 0.05). COMPLETE RESULTS ARE PROVIDED IN THE APPENDIX.
<table><tr><td rowspan="2">Variant</td><td colspan="2">ACC</td><td colspan="2">NMI</td><td colspan="2">ARI</td></tr><tr><td>Avg.</td><td>Drop</td><td>Avg.</td><td>Drop</td><td>Avg.</td><td>Drop</td></tr><tr><td>Full WAVE</td><td>0.703</td><td>一</td><td>0.587</td><td>一</td><td>0.520</td><td>一</td></tr><tr><td>w/o Visual</td><td>0.663</td><td>0.040</td><td>0.543</td><td>0.044</td><td>0.469</td><td>0.052</td></tr><tr><td>w/o Temporal</td><td>0.666</td><td>0.037*</td><td>0.519</td><td>0.068*</td><td>0.451</td><td>0.069*</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { t v } }$ </td><td>0.581</td><td>0.121*</td><td>0.415</td><td>0.172*</td><td>0.346</td><td>0.174*</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { t a } }$ </td><td>0.687</td><td>0.016</td><td>0.569</td><td>0.018*</td><td>0.498</td><td>0.022*</td></tr></table>

## A. Clustering Performance Evaluation

Table I summarizes performance across all ten datasets. WAVE achieves the highest macro-averaged ACC, NMI, and ARI and the best average rank on all three metrics. It ranks first in 15 and within the top two in 23 of the 30 dataset-metric combinations, with particularly clear gains on Epilepsy and Libras. Friedman tests confirm overall differences among the methods for all metrics $( p < 0 . 0 0 1 )$ ), although pairwise advantages over the strongest baseline do not remain significant after Holm correction. These results demonstrate improved average performance while also indicating dataset-dependent benefits. Complete results are provided in Table I of the appendix.

## B. Ablation Study

Table II summarizes the view and objective ablations over all ten datasets. Removing ${ \mathcal { L } } _ { \mathrm { t v } }$ causes the largest and most consistent degradation, reducing ACC, NMI, and ARI by 0.121, 0.172, and 0.174, respectively. Removing the temporal view also produces significant reductions across all three metrics, confirming its importance for preserving fine-grained temporal dynamics. In contrast, removing $\mathcal { L } _ { \mathrm { t a } }$ leads to smaller drops that are significant for NMI and ARI but not ACC, indicating a mainly regularizing role. Removing the visual view decreases all three macro-averaged metrics, although the differences are not statistically significant after Holm correction, suggesting dataset-dependent contributions. Overall, these results identify explicit temporal-visual alignment as WAVE’s most consistent mechanism. Complete dataset-level results are reported in Table II of the appendix.

TABLE III  
MACRO-AVERAGED CLUSTERING PERFORMANCE OF DIFFERENT VISUAL ENCODERS ON TEN UEA DATASETS OVER FIVE SEEDS.
<table><tr><td>Encoder</td><td>Pretraining</td><td>ACC</td><td>NMI</td><td>ARI</td></tr><tr><td>OpenCLIP ViT-B/32</td><td>LAION-2B</td><td>0.703</td><td>0.587</td><td>0.520</td></tr><tr><td>OpenCLIP ViT-B/32</td><td>Random</td><td>0.614</td><td>0.474</td><td>0.394</td></tr><tr><td>ResNet18</td><td>ImageNet-1K</td><td>0.686</td><td>0.567</td><td>0.504</td></tr></table>

![](images/1f0dd424f752cae7bb420904f784ec4ad8501a6adc70921f4831f727b3d80751.jpg)

![](images/3c8c8a79462ee56f5ae48caee7c360a2049b83f69658f8eb9a35142d7216580a.jpg)

![](images/bc356acbf4d291fe102bbd04fa61437f4dde474419929415d7f7082f062097e4.jpg)  
Fig. 2. Qualitative clustering and waveform-traceability results on BasicMotions and Epilepsy. The left panels show t-SNE projections of the learned representations, where stars mark the authentic samples nearest to the cluster centroids. The right panels show the corresponding waveform records for direct morphological inspection.

Beyond these component ablations, Table III examines different visual encoders under the same ten-dataset protocol. Pretrained OpenCLIP substantially outperforms its randomly initialized counterpart, confirming the benefit of visual pretraining for capturing waveform morphology. ResNet18 achieves competitive but slightly lower macro-averaged performance, indicating that the gains arise from pretrained visual representations rather than being exclusive to CLIP.

## C. Interpretability Case Study

Fig. 2 provides a case-level view of how WAVE’s temporal visual mechanism shapes the learned clusters on BasicMotions and Epilepsy. The temporal branch preserves activitydependent dynamics, while the waveform view exposes complementary morphological cues such as periodicity, amplitude, and irregular variations. Through cross-view alignment, samples sharing these temporal and visual characteristics are drawn into consistent regions of the embedding space. The centroid-nearest representatives further show that the learned clusters correspond to distinctive waveform structures. These observations are consistent with WAVE’s mechanism of organizing MTS through complementary temporal dynamics and visual morphology.

![](images/b2003675f8072cb489a5f469778d3b5b244368ce86f205c8aeee6143c2ebc0e1.jpg)  
Fig. 3. Sensitivity to the temporal fusion weight α over ten UEA datasets. The visual weight is $1 - \alpha .$ . Curves report macro-averaged performance, and shaded regions indicate one standard deviation across five seeds.

## D. Fusion-Weight Sensitivity

Fig. 3 evaluates the fixed fusion $z ( \alpha ) = \alpha \operatorname { n o r m } ( z ^ { t } ) + ( 1 -$ α) norm(z<sup>v</sup>). Intermediate weights outperform both singleview endpoints across all three metrics. Equal fusion achieves the highest ACC and ARI and performs comparably to $\alpha =$ 0.75 in NMI. The narrower uncertainty bands around intermediate weights also indicate greater stability across random seeds. The stable performance around $\alpha \ = \ 0 . 5$ supports equal fusion as a robust parameter-free choice without datasetspecific tuning.

## IV. CONCLUDING REMARKS

MTS clustering remains challenging because cluster structure may arise from both temporal dynamics and waveform morphology, while either view alone can be incomplete. WAVE aligns and fuses these complementary views into a unified representation and links each cluster to an authentic waveform sample for inspection. Results on 10 UEA datasets, together with ablation and sensitivity analyses, support its clustering effectiveness and the importance of cross-view alignment. WAVE currently assumes fixed waveform rendering and known cluster numbers, leaving adaptive rendering and cluster-number estimation for future work.

## ACKNOWLEDGEMENTS

This work was supported in part by the National Natural Science Foundation of China (NSFC) and the Guangdong Provincial Special Fund for Science and Technology Innovation Strategy. Zheng Zhu, the high-school student first author, led development, implementation, evaluation, result analysis, and manuscript drafting. Zexi Tan and Yuming Deng provided methodological guidance and result verification. Yiqun Zhang supervised the project, coordinated administration, secured funding, and revised the manuscript.

## REFERENCES

[1] Z. Tan et al., “Mask the redundancy: Evolving masking representation learning for multivariate time-series clustering,” in AAAI, vol. 40, no. 30, 2026, pp. 25 787–25 795.

[2] C. Wang et al., “Drift doesn’t matter: Dynamic decomposition with diffusion reconstruction for unstable multivariate time series anomaly detection,” in NeurIPS, vol. 36, 2023, pp. 10 758–10 774.

[3] Y. Cai et al., “JoLT: Jointly learned representations of language and time-series for clinical time-series interpretation,” in AAAI, vol. 38, no. 21, 2024, pp. 23 447–23 448.

[4] S. G. Dhekane et al., “Transfer learning in sensor-based human activity recognition: A survey,” ACM Comput. Surv., vol. 57, no. 8, pp. 1–39, 2025.

[5] Y. Liang et al., “AirFormer: Predicting nationwide air quality in China with transformers,” in AAAI, vol. 37, no. 12, 2023, pp. 14 329–14 337.

[6] E. Draayer et al., “Deep clustering for large-scale interpretable time series segmentation,” Data Min. Knowl. Discov., vol. 40, no. 1, pp. 1– 36, 2025.

[7] C. Wang et al., “Fuzzy cluster-aware contrastive clustering for time series,” Pattern Recognit., vol. 173, p. 112899, 2026.

[8] D. A. Nguyen et al., “Improving time series encoding with noise-aware self-supervised learning and an efficient encoder,” in ICDM, 2024, pp. 340–349.

[9] Q. Ren et al., “Rank supervised contrastive learning for time series classification,” in ICDM, 2024, pp. 839–844.

[10] C. Wang et al., “Time-frequency augmented multi-level contrastive clustering for time series,” in AAAI, 2026, pp. 26 142–26 150.

[11] C. He et al., “GTM: A general time-series model for enhanced representation learning of time-series data,” in ICLR, 2026. [Online]. Available: https://openreview.net/forum?id=PWM6FERWz9

[12] J. Liu et al., “TimesURL: Self-supervised contrastive learning for universal time series representation learning,” in AAAI, vol. 38, no. 12, 2024, pp. 13 918–13 926.

[13] E. Fu and Y. Hu, “Frequency-masked embedding inference: A noncontrastive approach for time series representation learning,” in AAAI, 2025, pp. 16 639–16 647.

[14] T. Xie et al., “Anchormoe: Interpretable time series classification via anchor-routed moe,” in KDD, 2026, pp. 5720–5731.

[15] K. Zhang et al., “Self-supervised learning for time series analysis: Taxonomy, progress, and prospects,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 10, pp. 6775–6794, 2024.

[16] Y. Li et al., “Contrastive learning-based multi-view clustering for incomplete multivariate time series,” Inf. Fusion, vol. 117, p. 102812, 2025.

[17] M. Moebus et al., “Contimask: Explaining irregular time series via perturbations in continuous time,” in NeurIPS, 2025, pp. 1–26.

[18] T. T. Nguyen et al., “Robust explainer recommendation for time series classification,” Data Min. Knowl. Discov., vol. 38, pp. 3372–3413, 2024.

[19] P. Boniol et al., “k-graph: A graph embedding for interpretable time series clustering,” IEEE Trans. Knowl. Data Eng., vol. 37, no. 5, pp. 2680–2694, 2025.

[20] X. Piao et al., “Fredformer: Frequency debiased transformer for time series forecasting,” in KDD, 2024, pp. 2400–2410.

[21] S. Gao et al., “UniTS: A unified multi-task time series model,” in NeurIPS, vol. 37, 2024, pp. 140 589–140 631.

[22] S. Zhong et al., “Time-VLM: Exploring multimodal vision-language models for augmented time series forecasting,” in ICML, 2025, pp. 78 478–78 497.

[23] T. Kimura et al., “InfoMAE: Pair-efficient cross-modal alignment for multimodal time-series sensing signals,” in WWW, 2025, pp. 3084–3095.

[24] P. Mohapatra et al., “MAESTRO: Adaptive sparse attention and robust learning for multimodal dynamic time series,” in NeurIPS, 2025, pp. 1–32.

[25] Y. Nam et al., “Bi-modal learning for networked time series,” in KDD, vol. 2, 2025, pp. 2162–2173.

[26] A. Bagnall et al., “The uea multivariate time series classification archive, 2018,” arXiv preprint arXiv:1811.00075, 2018.