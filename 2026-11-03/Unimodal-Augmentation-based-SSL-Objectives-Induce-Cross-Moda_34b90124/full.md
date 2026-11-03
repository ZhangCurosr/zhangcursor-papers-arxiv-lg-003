# Unimodal Augmentation-based SSL Objectives Induce Cross-Modal Alignment

Zohar Mashiah<sup>∗</sup> Department of Computer Science and AI Bar-Ilan University Ramat-Gan, Israel mashiahzohar4@gmail.com

Ronen Talmon Electrical and Computer Engineering Technion Haifa, Israel ronen@ee.technion.ac.il

Nir Ben-Ari<sup>∗</sup> Department of Computer Science and AI Bar-Ilan University Ramat-Gan, Israel nirnirba@gmail.com

Uri Shaham Department of Computer Science and AI Bar-Ilan University Ramat-Gan, Israel uri.shaham@biu.ac.il

## Abstract

Recent works have observed implicit cross-modal alignment between independently trained unimodal models. However, the mechanisms underlying the emergence of this phenomenon remain unresolved. In this paper, we demonstrate that unimodal augmentation-based self-supervised objectives implicitly induce cross-modal alignment. We show that low unimodal losses force high cross-modal alignment, whereas a high unimodal loss prevents such alignment. We theoretically connect the widely used InfoNCE loss to local CKA, the popular measure of crossmodal alignment. We further provide extensive empirical evidence supporting this claim, both in controlled settings and across 53 unconstrained frozen text, vision, and audio unimodal encoders. In both cases, unimodal augmentation-based objectives control cross-modal alignment. Overall, our results provide a theoretical and empirical foundation for understanding the emergence of implicit cross-modal alignment by establishing a direct connection between unimodal training objectives and cross-modal structure.<sup>†</sup>

## 1 Introduction

Recent discoveries in representation learning have provided empirical evidence that independently trained unimodal vision and language models converge toward a unified representation, particularly in terms of their local neighborhood structure (Huh et al., 2024; Gröger et al., 2026a). This phenomenon is known as the Platonic Representation Hypothesis (PRH), and has been further supported by extensive subsequent work (Li et al., 2024; Acevedo et al., 2025; He et al., 2025), including extensions to additional modalities (Duraphe et al., 2025; Li et al., 2025; Edamadaka et al., 2025; Ngo & Kim, 2024) and practical applications (Yacobi et al., 2026; Yoon et al., 2025; Zhang et al., 2025; Gröger et al., 2026b; Jha et al., 2026; Gupta et al., 2026; Schnaus et al., 2025; Li & Walsh, 2026).

However, the understanding of the causes underlying this emergent alignment remains limited. Most works either investigate the effect of scaling-related attributes of model size and capacity or the amount of training data on the strength of this phenomenon (Umar & Laio, 2026; Lobashev, 2025), or empirically address this question with explicit cross-modal factors. For example, shared input statistics and architectural biases (Kapoor et al., 2026; Tjandrasuwita et al., 2025), or overlap in datasets and downstream tasks (Li et al., 2025).

Meanwhile, Augmentation-based Self-Supervised Learning (Aug-SSL) has emerged as a leading training paradigm for training large models (Gui et al., 2024). Popular models such as DINO for vision (Oquab et al., 2023), Sentence-BERT for text (Reimers & Gurevych, 2019), and BYOL-A for audio (Niizumi et al., 2021), have all been trained via Aug-SSL. While its power in unimodal settings is well-studied (Wang & Isola, 2020; Balestriero & LeCun, 2022), to the best of our knowledge, its effect on implicit cross-modal alignment has remained unexplored.

In this paper, we show that emergent cross-modal alignment (CMA) is a direct consequence of unimodal Aug-SSL objectives. That is, the unimodal objectives implicitly induce cross-modal alignment, even in the absence of explicit cross-modal supervision. The augmentations act as view-specific information removers, filtering out view-specific variation. The encoders then steer each embedding space toward the shared semantic structure underlying the modalities, thereby promoting cross-modal similarity. To the best of our knowledge, this is the first work to explain implicit cross-modal alignment via unimodal properties, connecting them both theoretically and in practice.

![](images/81439d2215cddae77df2b3758b34f7e9be8f6b8f9310a51d84d96b57418a7d43.jpg)

We begin with a theoretical investigation, directly connecting the widely used InfoNCE objective (Oord et al., 2018) to local CKA, a popular measure of CMA (Cristianini et al., 2001). We prove that low unimodal Aug-SSL losses force high CMA (Thm. 1), whereas even a single high unimodal loss restricts the alignment (Thm. 2). These results hold independently of data size, model architecture, and other factors previously proposed to explain the emergence of cross-modal alignment.

Figure 1: Augmentation-based objectives induce cross-modal alignment. Cross-modal text-vision alignment (mutual k-nearest neighbors) as a function of model size (number of parameters) across different text models to DI-NOv2. Notably, augmentation-based models achieve higher cross-modal alignment for a given model size, and yield superior alignment even with significantly fewer parameters.

We further provide extensive empirical evidence for these results in real-world settings, both in controlled experiments and across large-scale pretrained models and modalities. For example, Fig. 1 shows that pre-trained text augmentation-based models consistently achieve higher CMA with a vision model than models of comparable size trained with next-token prediction (NTP). Moreover, augmentation-based models reach high CMA with substantially fewer parameters.

Contributions. Our contributions are summarized as follows: (1) We show that unimodal selfsupervised objectives implicitly drive cross-modal alignment; (2) We theoretically establish a connection between the unimodal InfoNCE loss and cross-modal alignment as measured by local CKA; and (3) We empirically demonstrate that the unimodal Aug-SSL loss predicts cross-modal alignment across controlled experiments and large-scale pretrained models.

## 2 Related Work

Implicit Cross-Modal Alignment. Numerous studies have documented the emergence of implicit CMA in vision-language models (Huh et al., 2024; Li et al., 2024; Acevedo et al., 2025; Miller, 2026; Fan et al., 2025; He et al., 2025). Subsequent work has shown that similar alignment phenomena arise across a broader range of modalities, including video-text (Ovsjanikov et al., 2026), astronomical data (Duraphe et al., 2025), and scientific data (Li & Walsh, 2026; Edamadaka et al., 2025), among others. Beyond documenting its existence, several studies have examined how CMA depends on properties of the data and training setup, attributing its emergence to factors such as dataset and task overlap (Li et al., 2025) or shared data characteristics (Tjandrasuwita et al., 2025). However, these explanations remain empirical and do not establish a theoretical account of why CMA emerges.

![](images/575d93bc8cc902c7bce2ba80b31d8a00f61be024cbc882638dd0093be59ecfcc.jpg)  
(a)

![](images/5c90809f1592042d515d475b1cb700cc33d15f9e65115a06df7981a4a62f9ddd.jpg)  
(b)  
Figure 2: (a) Theoretical setup visualization. Three manifolds ${ \mathcal { M } } , { \mathcal { U } } ,$ and V generate two observation spaces X and $\mathcal { V }$ with shared semantics $\mathcal { M } .$ . Two trainable encoders then map the observations to the corresponding embedding spaces $\mathcal { E } _ { X }$ and $\mathcal { E } _ { Y }$ . (b) Correlation between CKA-RBF and mkNN. mkNN is highly correlated with CKA-RBF across different choices of k and the corresponding σ. Both Spearman $( \rho _ { s } )$ and Pearson (r) correlations are close to 1.

Theory of Implicit CMA. Several recent works have developed theoretical perspectives on implicit cross-modal alignment. Lu et al. (2026) formalize the alignment hypothesis using category theory. Insulla et al. (2025) establish a connection between representation alignment and downstream task performance. However, these do not characterize the mechanisms underlying its emergence. Other studies investigate potential mechanisms that may give rise to this phenomenon (Bangachev et al., 2026; Ziyin et al., 2026; Ziyin & Chuang, 2025); however, their analyses rely on strong linearity assumptions about the data or learned representations, limiting their applicability to the highly nonlinear models used in practice. Lobashev (2025) further assumes that the two modalities share input statistics, imposing an additional restriction on the data-generating process. In contrast, this paper provides a theoretical account of how implicit cross-modal alignment emerges directly from the unimodal training objective in general settings.

Augmentation-based SSL. Self-supervised learning (SSL) has emerged as a dominant training paradigm across a wide range of domains, including vision (Chen et al., 2020; Oquab et al., 2023; He et al., 2022), language (Devlin et al., 2019; Gao et al., 2021; Liu et al., 2019), audio (Baevski et al., 2020; Hsu et al., 2021; Saeed et al., 2021; Niizumi et al., 2021), video (Tong et al., 2022; Bardes et al., 2024), and scientific data (Wang et al., 2022). In particular, augmentation-based SSL provides a simple and effective mechanism for learning from large numbers of self-generated training examples through different transformations of the same underlying samples. Prior works have investigated the properties induced by SSL training within individual modalities (Wang & Isola, 2020; Balestriero & LeCun, 2022). Conversely, other works have demonstrated the effectiveness of explicit cross-modal SSL objectives for learning aligned representations (Radford et al., 2021; Wu et al., 2022). In contrast, this paper considers the implications of SSL on implicit cross-modal alignment, without any cross-modal objective.

## 3 Theoretical Analysis: From Unimodal InfoNCE to CMA

This section presents our theoretical results connecting unimodal InfoNCE and cross-modal alignment (CMA). We first formally define the setup, CMA, and unimodal augmentation-based self-supervised learning (Aug-SSL) (Sec. 3.1). We then establish their connection through two complementary results (Sec. 3.2). In Thm. 1, we show that low unimodal losses force high CMA. In Thm. 2, we show that a high unimodal loss for even one model restricts CMA, independently of other factors such as model architecture and training data size. Together, these results establish that unimodal augmentation-based training is a fundamental driver of implicit CMA. For brevity, we defer the complete derivations and proofs to App. A.

## 3.1 Setup

Notations (visualized in Fig. 2a). Let $\mathbf { \mathcal { M } } , \boldsymbol { \mathcal { U } } , \boldsymbol { \mathcal { V } }$ be manifolds representing shared semantics and view-specific information, respectively. Observations are generated by $g _ { X } : \mathcal { M } \times \mathcal { U } \to \mathcal { X }$ and $g _ { Y } : \bar { \mathcal { M } } \times \mathcal { V }  \mathcal { V }$ , where $\mathcal { X } \subseteq \dot { \mathbb { R } } ^ { d _ { \boldsymbol { x } } } , \boldsymbol { y } \subseteq \mathbb { R } ^ { d _ { \boldsymbol { y } } }$ . Two trainable unimodal encoders, $e _ { X } : \mathcal { X } \to \mathcal { E } _ { X } , e _ { Y } :$ $\mathcal { V } \to \mathcal { E } _ { Y }$ , map the observations to normalized embedding spaces $\mathcal { E } _ { X } \subseteq \mathbb { S } ^ { \dot { d } _ { X } \dot { - 1 } } , \mathcal { E } _ { Y } \subseteq \dot { \mathbb { S } } ^ { \dot { d } _ { Y } - 1 }$ respectively. Throughout, $0 < \tau \leq 1$ denotes the fixed temperature of the training objective.

The phenomenon of implicit CMA dictates that independently trained unimodal encoders naturally develop highly aligned representations. To measure the alignment between the embedding spaces $\mathcal { E } _ { X }$ and $\mathcal { E } _ { Y }$ , we use the well-established Centered Kernel Alignment (CKA) metric (Kornblith et al., 2019), following recent works on implicit cross-modal alignment (Huh et al., 2024; Gröger et al., 2026a). We denote the observations $\bar { x = { g _ { X } ( m , u ) , x ^ { \prime } = g _ { X } ( m ^ { \prime } , u ^ { \prime } ) , y = g _ { Y } ( m , v ) , y ^ { \prime } = g _ { Y } ( m ^ { \prime } , v ^ { \prime } ) } }$ . That is, x and y share the same underlying semantics $m _ { : }$ , and likewise $x ^ { \prime }$ and $y ^ { \prime }$ share $m ^ { \prime } .$ In practice, this shared semantic structure is obtained through paired data. The CKA is then often defined using the Hilbert-Schmidt Independence Criterion (HSIC), which is

$$
\mathrm { H S I C } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) = \mathbb { E } \left[ k _ { X } ^ { c } \left( e _ { X } ( x ) , e _ { X } ( x ^ { \prime } ) \right) \right. \left. k _ { Y } ^ { c } \left( e _ { Y } ( y ) , e _ { Y } ( y ^ { \prime } ) \right) \right]
$$

where $k ^ { c }$ is the doubly centered kernel $k ^ { c } = k - \mathbb { E } [ k | x ] - \mathbb { E } [ k | x ^ { \prime } ] + \mathbb { E } [ k ]$ . CKA then measures the correlation between the centered kernels, giving:

$$
\operatorname { C K A } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) = \frac { \mathrm { H S I C } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) } { \sqrt { \mathrm { H S I C } ( \mathcal { E } _ { X } , \mathcal { E } _ { X } ) \mathrm { H S I C } ( \mathcal { E } _ { Y } , \mathcal { E } _ { Y } ) } } \in [ 0 , 1 ] ,\tag{1}
$$

Throughout the paper, we use the CKA-RBF version of CKA (Kornblith et al., 2019), with the Gaussian kerne $k ( z , \bar { z ^ { \prime } } ) = e ^ { - \| z - z ^ { \prime } \| ^ { 2 } / 2 \sigma ^ { 2 } }$ , and denote the kernel variances $\begin{array} { r } { \sigma _ { X } ^ { 2 } = \mathrm { H S I C } ( \mathcal { E } _ { X } , \mathcal { E } _ { X } ) , \sigma _ { Y } ^ { 2 } = } \end{array}$ $\mathrm { H S I C } ( \mathcal { E } _ { Y } , \mathcal { E } _ { Y } )$ . In particular, we assume $\sigma _ { X } , \sigma _ { Y } > 0$ so the CKA is defined.

The Gaussian kernel emphasizes local structure, making this measure particularly suitable for studying the local nature of implicit cross-modal alignment observed in recent work (Gröger et al., 2026a). Moreover, as shown in Fig. 2b, CKA-RBF is strongly correlated with the mutual k-nearest-neighbors (mkNN) metric used in recent works, with the kernel bandwidth σ playing a role analogous to the neighborhood parameter k (see App. J).

Augmentation-based SSL. We define an augmentation as a resampling of both the semantics and the view, while preserving the underlying semantic structure. Specifically:

$$
x ^ { + } : = \operatorname { a u g } ( x ) = g _ { X } ( m ^ { + } , u ^ { + } ) , \qquad m ^ { + } \sim P ( m , \cdot ) , \quad u ^ { + } \sim \nu _ { \mathcal { U } } \mathrm { ~ i n d e p e n d e n t }\tag{2}
$$

where $P ( m , \cdot ) : = \mathsf { J } ( m , \cdot ) \mu$ is a Markov kernel on M with density J. This definition aligns with prior theoretical studies of augmentation-based SSL (Von Kügelgen et al., 2021), while not restricting the augmentation to exactly preserve the semantics. We exclude the degenerate case $\mathsf { J } \equiv 1$ , in which the augmentation resamples the semantics independently and carries no signal.

An Aug-SSL model is trained to maximize the similarity between augmented samples, while preventing collapse to a constant representation, for instance by contrasting against randomly sampled observations. A popular objective for this purpose is the InfoNCE loss (Oord et al., 2018), which has been widely adopted in vision (Chen et al., 2020; He et al., 2020), text (Reimers & Gurevych, 2019; Gao et al., 2021), audio (Niizumi et al., 2021), and other domains (Qian et al., 2021; Wang et al., 2022). Let $x , x ^ { \prime } \in { \mathcal { X } }$ denote observations and let $x ^ { + } = \arg ( x )$ be an augmentation of x. The K-negative InfoNCE loss is

$$
\ell _ { X } ^ { ( K ) } = \mathbb { E } \left[ - \log \frac { \exp \left( \langle e _ { X } ( x ) , e _ { X } ( x ^ { + } ) \rangle / \tau \right) } { \exp \left( \langle e _ { X } ( x ) , e _ { X } ( x ^ { + } ) \rangle / \tau \right) + \sum _ { k = 1 } ^ { K } \exp \left( \langle e _ { X } ( x ) , e _ { X } ( x _ { k } ) \rangle / \tau \right) } \right] ,\tag{3}
$$

where K is a hyperparameter controlling the number of negative samples. Following Wang & Isola (2020) (Thm. 1), we use its population form:

$$
\begin{array} { r } { { \cal N } _ { X } = \displaystyle \operatorname* { l i m } _ { K \to \infty } \left( \ell _ { X } ^ { ( K ) } - \log K \right) = - \frac { 1 } { \tau } \mathbb { E } \left[ \langle e _ { X } ( x ) , e _ { X } ( x ^ { + } ) \rangle \right] + \mathbb { E } _ { x } \left[ \log \mathbb { E } _ { x ^ { \prime } } \left[ e ^ { \langle e _ { X } ( x ) , e _ { X } ( x ^ { \prime } ) \rangle / \tau } \right] \right] . } \end{array}
$$

For convenience, we normalize the objective as follows, with both ends attainable (Lemma 8):

$$
L _ { X } = \frac { N _ { X } + 1 / \tau } { R _ { \tau } } \in [ 0 , 1 ] ,
$$

where $\begin{array} { r } { R _ { \tau } = \frac { 2 } { \tau } + } \end{array}$ log cosh $\scriptstyle { \frac { 1 } { \tau } } \cdot L _ { Y }$ is analogously defined for Y.

## 3.2 Connecting InfoNCE and local CKA

We now present the theoretical results that formalize the connection between the unimodal Aug-SSL losses (InfoNCE) and CMA under the setup above. The two results establish complementary directions of this relationship. Specifically, Thm. 1 shows that low losses in both modalities guarantee high CMA, while Thm. 2 shows that high loss in either modality prevents alignment.

Implicit alignment. The bound in Thm. 1 utilizes the concentration strength of the augmentations, $\Vert J - 1 \Vert _ { L ^ { 2 } } = : \chi$ , where $J ( m , m ^ { + } )$ is the likelihood ratio of $m ^ { + }$ being an augmentation of m versus an independent sample. Thus, $\chi$ quantifies the semantic information retained by augmentations: $\chi = 0$ when $J \equiv 1$ and augmentations ignore the original semantics, and $\chi = \infty$ when they preserve them exactly. We consider the practical regime $0 < \chi < \infty$ , where augmentations alter but retain semantics. Together with the InfoNCE temperature $\tau$ and CKA-RBF scale $\sigma ,$ , this allows us to lower-bound CKA in terms of the unimodal losses.

Theorem 1 (Implicit alignment).

$$
\begin{array} { l } { \displaystyle \mathrm { C K A } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) \geq f ( L _ { X } , L _ { Y } \mid \tau , \sigma , \chi ) , } \end{array}
$$

with $f$ non-increasing in both $L _ { X }$ and $L _ { Y }$ , and converging to 1 as both losses approach 0.

This theorem establishes that when both unimodal Aug-SSL losses $( L _ { X }$ and $L _ { Y } )$ are effectively minimized, a high degree of cross-modal alignment is guaranteed. Importantly, the bound depends only on the attained loss values, not on the training procedure. The monotonic convergence of $f$ to 1 strictly forces the CKA toward its maximum value when the losses are sufficiently small. Specifically,

$$
\begin{array} { r } { f ( L _ { X } , L _ { Y } ) = \cos ( \theta _ { X } + \theta _ { Y } ) , \quad \theta ( L ) = \mathrm { c l a m p } \big ( \operatorname { a r c c o s } \big ( \frac { a _ { \sigma } - b _ { \sigma , \tau } L } { \chi \sqrt { D _ { \sigma , \tau } L } } \big ) , 0 , \frac { \pi } { 2 } \big ) , } \end{array}
$$

where clamp restricts $\theta ( L )$ to $[ 0 , \frac { \pi } { 2 } ]$ , and $\theta _ { X } = \theta ( L _ { X } ) , \theta _ { Y } = \theta ( L _ { Y } )$ . In particular, $f$ depends on the augmentations concentration strength $\chi ,$ , and on the InfoNCE temperature τ and the CKA-RBF scale σ through the hyperparameter-dependent constants: $a _ { \sigma } , b _ { \sigma , \tau }$ and $D _ { \sigma , \tau }$ . For brevity, their exact definitions are deferred to App. A.8.

Intuitively, augmentations remove view-specific information, filtering out variations that are not shared across modalities. Low loss then encourages augmented samples to cluster together, guiding each embedding space toward the shared semantic structure underlying the modalities. Thereby, promoting cross-modal similarity, i.e., high CKA.

Implicit misalignment. While the above highlights the geometric convergence driven by successful unimodal optimization, evaluating the implications of high unimodal loss requires the complementary perspective addressed in Thm. 2. For this, we assume the unimodal deep SSL encoders $e _ { X } , e _ { Y }$ exhibit Lipschitz continuity. That is, $\| e _ { X } ( x ) - e _ { X } ( x ^ { \prime } ) \| \leq \ell _ { X } d _ { \mathcal { X } } ( x , x ^ { \prime } )$ for all $x , x ^ { \prime } \in { \mathcal { X } } ,$ , with an analogous bound $\ell _ { Y }$ for $e _ { Y }$ . This is consistent with recent literature, proving both empirically and theoretically that this happens in practice (Virmaux & Scaman, 2018; Lee et al., 2021). This theorem utilizes an augmentation-dependent constant, the semantic displacement:

$$
\varepsilon _ { X } = \mathbb { E } _ { \pi , u } \Big [ d _ { X } \big ( g _ { X } ( m , u ) , g _ { X } ( m ^ { + } , u ) \big ) ^ { 2 } \Big ] ,
$$

where $\pi ( m , m ^ { + } ) = J ( m , m ^ { + } ) \mu ( m ) \mu ( m ^ { + } )$ is the joint law of the semantic pair. Intuitively, $\varepsilon _ { X }$ measures the extent to which the semantic variations caused by the augmentation process are manifested in the observation space.

Theorem 2 (Implicit misalignment). If the encoder $e _ { X }$ is Lipschitz with a constant $\ell _ { X }$ , and $\tau \leq \sigma ^ { 2 }$ then

$$
\begin{array} { r } { \mathrm { C K A } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) \leq g ( L _ { X } \mid \tau , \sigma , \ell _ { X } , \varepsilon _ { X } ) , } \end{array}
$$

where g is strictly decreasing in $L _ { X }$ . This holds symmetrically for $L _ { Y }$ with $\ell _ { Y } , \varepsilon _ { Y }$

Thm. 2 shows that a sufficiently high unimodal loss strictly upper-bounds CMA, preventing convergence. In particular, for practical hyperparameters, the bound remains bounded away from 1 as $L _ { X }$ approaches its maximum (App. K). Consequently, a poorly optimized encoder strictly precludes the emergence of shared geometric structure, regardless of the optimization quality in the counterpart modality. The condition $\tau \leq \sigma ^ { 2 }$ is consistent with standard InfoNCE practice, where τ is a small fraction and $\sigma ^ { 2 }$ is typically larger for normalized embeddings.

The bound is given by

$$
\begin{array} { r } { g ( L _ { X } ) = \frac { d _ { \sigma , \tau } + \ell _ { X } ^ { 2 } \varepsilon _ { X } } { 2 \sigma ^ { 2 } \sigma _ { X } } + \frac { h _ { \sigma , \tau } } { \sigma _ { X } } ( 1 - L _ { X } ) . } \end{array}
$$

In particular, $g$ depends on the Lipschitz constant $\ell _ { X }$ , the semantic displacement $\varepsilon _ { X }$ , and on the InfoNCE temperature $\tau$ and the CKA-RBF scale $\sigma$ via the hyperparameter-dependent constants: $d _ { \sigma , \tau }$ and $h _ { \sigma , \tau } \left( \mathrm { A p p . \ A . 8 } \right)$

![](images/5340b5a731aaa017e8f21d4a3a541f12837fe0cbac225966ac2c126a35adad6e.jpg)  
(a)

![](images/77475a3b00c3efb6e205b31e2e8ca37d8013c9d6aa68e987ef3945fffc6323b4.jpg)  
(b)  
Figure 3: (a) Empirical validation of Thm. 1 and 2. Vision (ViT) and text (SBERT) models trained to minimize/maximize the Aug-SSL loss (Eq. 3). Minimizing both unimodal losses improves CMA (Thm. 1), whereas maximizing either loss prevents alignment, with CMA converging to the chance level (Thm. 2). (b) Augmentations vs. Supervision. CMA throughout training of a vision model (ViT) against frozen text models: an augmentation-based model (SBERT) and a non-augmentationbased model (BERT). Training ViT with augmentations achieves substantially higher CMA, including compared to the BERT model.

Intuitively, only semantic variation can align across modalities, so CKA is bounded by the fraction of embedding variation that is semantic. As the unimodal loss increases, less variation reflects semantics, limiting the achievable CKA. Both bounds are visualized in App. K.

Together, these theoretical results provide a rigorous mathematical foundation, showing that implicit cross-modal alignment is a geometric consequence of the unimodal InfoNCE objective. In the next section, we empirically demonstrate these predictions, showing that Aug-SSL models exhibit higher CMA than their counterparts.

Message of Section 3: Low unimodal augmentation-based SSL losses induce cross-modal convergence, while a high loss in either modality forces misalignment.

## 4 Aug-SSL is Key and Predictor of CMA in Practice

In this section, we empirically validate the core premise of this paper. We consider both controlled experiments (Sec. 4.1) and frozen pretrained models (Sec. 4.2), providing evidence that unimodal augmentation-based objectives play a central role in the emergence of implicit CMA. Further details concerning the experimental setup and implementation are provided in App. L.

Datasets. We utilize four benchmark datasets. Following Huh et al. (2024) and Gröger et al. (2026a), we use the Wikipedia image-text (WIT) dataset (Srinivasan et al., 2021). We additionally use the MS-COCO image-text benchmark (Lin et al., 2014), which provides semantic labels that allow us to more closely examine the effect of augmentations on cross-modal alignment. To further validate the phenomenon across modalities, we use the audio-text CLOTHO (Drossos et al., 2020) and AudioCaps (Kim et al., 2019) benchmarks.

Evaluating Cross-Modal Alignment. Following Gröger et al. (2026a), we evaluate CMA using the mutual k-nearest neighbors (mkNN) metric. Given a set of paired observations $s =$ $( x _ { 1 } , y _ { 1 } ) , \dots , ( x _ { n } , y _ { n } ) \in { \mathcal { E } } _ { X } \times { \mathcal { E } } _ { Y }$ , let $\overset { \cdot } { N } _ { X } ^ { k } ( x )$ and $N _ { Y } ^ { k } ( y )$ denote the sets of the k nearest neighbors of x and y within their respective embedding spaces. We define

$$
\operatorname { m } k \mathrm { N N } ( \mathcal { S } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { \left| N _ { X } ^ { k } ( x _ { i } ) \cap N _ { Y } ^ { k } ( y _ { i } ) \right| } { k } .\tag{4}
$$

As shown in Fig. 2b, mkNN can be viewed as a discrete approximation of the CKA-RBF considered in Sec. 3, making it a natural choice for evaluating CMA in our experiments. Unless stated otherwise, we use $k = 5 0$ . Additional results using different values of $k ,$ and CKNNA are provided in App. G, and exhibit the same qualitative trends.

NTP BLOOM OpenLLaMA LLaMA Pythia GPT-2

Aug MiniLM RoBERTa Sentence-T5

![](images/8acccb074ecb750761bf5bb0ca8a82b4d48776aac34bdacd33ea410b5d50deb5.jpg)  
(a)

![](images/9f4651d8af620f8ce0c58631e9606df4a5d824370f6bdd5a07f7a170c4b3f941.jpg)

(b)  
![](images/2c943a0b3fde8151e0622f2b6396c880973dac544891d53c6c23fab40458def0.jpg)  
Figure 4: Alignment by model/corpus size. CMA between text models and the DINOv2 vision encoder, using augmentation-based (Aug) and next-token prediction-based (NTP) models. Shown as a function of (a) model size (number of parameters), and (b) corpus size (amount of tokens the model was trained on). Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all models jointly, model/corpus sizes are uncorrelated with CMA, indicated by both weak Spearman (ρ ) and Pearson (r) correlations.

## 4.1 Direct Validation of the Theory

In this section, we isolate the effect of unimodal Aug-SSL training by controlling the model architectures and training setup. We train vision and text encoders from scratch using a ViT (Dosovitskiy et al., 2020) for vision and an SBERT architecture (Reimers & Gurevych, 2019) for text, initialized from a pretrained BERT model (Devlin et al., 2019).

Isolating the InfoNCE effect. To empirically validate Thm. 1 and 2, we train two vision and two text encoders on COCO as described above, with one model per modality minimizing and the other maximizing the unimodal Aug-SSL loss. We use the InfoNCE objective from Sec. 3. For vision, we employ standard image augmentations (e.g., random crops, horizontal flips). For text, we use an LLM (Qwen2.5 (Qwen et al., 2025)) to generate five rephrased versions of each text.

Fig. 3a presents the resulting CMA for all pairings of the four encoders. The pairing of the two loss-minimized encoders corresponds to Thm. 1: minimizing the unimodal Aug-SSL objectives leads to increasing CMA. In contrast, every pairing involving a loss-maximized encoder corresponds to Thm. 2, demonstrating that a high unimodal loss in either modality prevents cross-modal alignment, with CMA approaching the chance level. Training on WIT shows the same trends (App. D).

Aug-SSL Outperforms Explicit Supervision. To isolate the contribution of unimodal augmentations to CMA, we compare augmentation-based SSL with supervised learning on the COCO dataset. Specifically, we consider two supervised objectives: Supervised Contrastive Learning (SupCon) (Khosla et al., 2020) and cross-entropy classification, trained using the labels in COCO. In these experiments, we freeze the pretrained text encoder (BERT/SBERT) and train only the vision encoder.

Fig. 3b presents the resulting CMA. Most notably, the Aug-SSL-trained vision encoder achieves the highest CMA with both text encoders. Its alignment with SBERT, which is an augmentation-based model, is consistently stronger than with BERT, further supporting that augmentation-based training promotes alignment. Importantly, however, the Aug-SSL vision encoder also achieves stronger CMA with BERT than the supervised baselines. Thus, the effect cannot be attributed simply to matching the training paradigm across modalities: augmentation-based training promotes CMA even with a text encoder trained using a different objective. These results further support the role of augmentations as a driver of implicit CMA.

## 4.2 Universal Emergence in Foundation Models

We extend our analysis to unconstrained, frozen pre-trained models from diverse model kinds and modalities. We show that augmentations implicitly induce CMA across architectures, training datasets, and model scales.

![](images/2c1c2846e3df671a11bb5201dc197139691ec431f264c2b15817377f3f816370.jpg)

![](images/5233a13f048b8be6b8df75ab141f28b86ea3622fb4289b00b8f5666cdf698fd6.jpg)

![](images/f61fecc39cdabe3461fce191000a26af0cc68152cfcc0290ec7fc57f5ecd0cbf.jpg)  
Figure 5: Alignment by unimodal loss. Image-text CMA between text models and DINOv2, and vision models and BLOOM as a function of unimodal (a) text InfoNCE loss, and (b) image InfoNCE loss. In contrast to model or corpus size, the unimodal Aug-SSL loss serves as a robust predictor of CMA. It cleanly separates low- and high-alignment models, exhibiting strong Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

Table 1: Image-text CMA is highly correlated with unimodal InfoNCE loss. Spearman $( \rho _ { s } )$ and Pearson (r) correlations between mkNN and unimodal properties over 24/14 text/vision models. Results are mean ± std over all second modality models. Notably, CMA is consistently highly correlated with the negative text and vision unimodal InfoNCE loss.
<table><tr><td rowspan="2"> $\rho _ { s } / r$ </td><td colspan="4">MS-COCO</td><td colspan="4">WIT</td></tr><tr><td colspan="2">Text models</td><td colspan="2">Vision models</td><td colspan="2">Text models</td><td colspan="2">Vision models</td></tr><tr><td>Model size</td><td></td><td> $- 0 . 0 5 _ { \pm 0 . 0 2 } / - 0 . 1 4 _ { \pm 0 . 0 6 }$ </td><td> $- 0 . 2 3 _ { \pm 0 . 0 6 } / - 0 . 2 4 _ { \pm 0 . 0 1 }$ </td><td></td><td> $0 . 0 9 _ { \pm 0 . 0 2 } / 0 . 0 8 _ { \pm 0 . 0 4 }$ </td><td></td><td>−0.25±0.12 /</td><td>2/ −0.28±0.10</td></tr><tr><td>Training-corpus size</td><td> $- 0 . 3 2 _ { \pm 0 . 0 4 } / . - 0 . 4 3 _ { \pm 0 . 0 4 }$ </td><td></td><td>0.20±0.05/</td><td>0.58±0.03</td><td> $- 0 . 1 8 { \scriptstyle \pm 0 . 0 1 } \big / - 0 . 2 1 { \scriptstyle \pm 0 . 0 3 }$ </td><td></td><td>0.34±0.06 /</td><td> $0 . 4 5 _ { \pm 0 . 1 6 }$  </td></tr><tr><td> $\operatorname { I n f o N C E } \operatorname { l o s s } \left( - \mathcal { L } \right)$ </td><td></td><td> $\begin{array} { r l } { \mathbf { 0 . 8 5 _ { \pm 0 . 0 3 } } / } & { { } \mathbf { 0 . 9 4 _ { \pm 0 . 0 3 } } } \end{array}$ </td><td>0.70±0.14 /</td><td>0.92±0.02</td><td> $\mathbf { 0 . 8 4 _ { \pm 0 . 0 2 } / } \quad \mathbf { 0 . 8 8 _ { \pm 0 . 0 2 } }$ </td><td></td><td> $\mathbf { 0 . 8 6 _ { \pm 0 . 0 9 } } ^ { } \dot { } /$ </td><td> ${ \bf 0 . 9 0 _ { \pm 0 . 0 3 } }$ </td></tr></table>

Models. We consider 53 pretrained unimodal encoders across text, vision, and audio. Specifically, we use eight families of 24 text encoders from two training paradigms: augmentation-based models, including MiniLM (Wang et al., 2020), RoBERTa (Reimers & Gurevych, 2019), and Sentence-T5 (Ni et al., 2022); and next-token prediction (NTP)-based models, including BLOOM (Le Scao et al., 2022), OpenLLaMA (Geng & Liu, 2023), LLaMA (Touvron et al., 2023), Pythia (Biderman et al., 2023), and GPT-2 (Radford et al., 2019). Also, we consider four families of 14 vision encoders: DINOv2 (Oquab et al., 2023), MAE (He et al., 2022), ImageNet21K (Russakovsky et al., 2015), and CLIP (Radford et al., 2021). For each family, we evaluate multiple models of different sizes. We compare the text models with the DINOv2 vision encoder and the BYOL-A v2 audio encoder (Niizumi et al., 2022), with the other vision encoders and a total of 15 audio encoders shown in App. E and App. F. Further details regarding the models are provided in App. L.1.

Alignment vs. model/corpus size. Fig. 4a (and 1) shows the CMA of text pre-trained models to DINOv2 as a function of model size. Similarly, Fig. 4b shows it as a function of the corpus size the model was trained on. Interestingly, augmentation-based models consistently achieve higher CMA than NTP-based models of comparable size, and often outperform even substantially larger NTP models, trained with significantly more data. This provides evidence that unimodal augmentations play a key role in inducing CMA. Indicated by both Pearson and Spearman correlations, when considering different model families jointly, model or corpus sizes are uncorrelated with CMA.

Unimodal InfoNCE predicts CMA. Fig. 5 and Tab. 1 show the CMA as a function of the unimodal InfoNCE loss. We apply the same standard image and LLM-based text augmentations described in Sec. 4.1 post-hoc to the frozen pre-trained models and evaluate their unimodal InfoNCE loss. Interestingly, lower unimodal InfoNCE loss is highly correlated with higher CMA, indicated by both high Spearman and Pearson correlations. This further validates the theory presented in Sec. 3, showing that InfoNCE can serve as a unimodal predictor of cross-modal alignment.

![](images/aaf62d03b60b3384ac643b8a35c7d48525adab7b2c78fb5c68b3505ae1ec9520.jpg)

![](images/68969ebc9de97bd161d2b5ad788facacd79f77b57fb5348f5b55d0a3c05617eb.jpg)  
ρ<sub>s</sub> = 0.04, r = 0.15  
(a)

![](images/ac413c4b52d3ab49de7fa988a4df6262f28df714b223bed393eea8e0c190b2f3.jpg)  
ρ = -0.20, r = -0.13  
(b)

![](images/facfe2806e28291fb6439a0a7987e7ae9e2c6bae175f61126886936f42907d8e.jpg)  
ρ = 0.92, r = 0.86  
(c)  
Figure 6: Audio-text results. CMA between text models and the BYOL-A audio encoder as a function of (a) model size (number of parameters), (b) corpus size (amount of tokens the model was trained on), and (c) unimodal text InfoNCE loss. Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all models jointly, model/corpus sizes are uncorrelated with CMA, while the unimodal loss is highly correlated, indicated by both Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

Beyond image-text. Fig. 6 and Tab. 2 show that the same trends observed in vision-text hold robustly to audio-text. Specifically, model/corpus sizes are uncorrelated with CMA, showing augmentationbased models experience higher alignment nonetheless. Also, the unimodal InfoNCE loss is a strong predictor of the cross-modal alignment. This provides further evidence for the universality of these findings across modalities.

Table 2: Audio-text CMA is highly correlated with unimodal InfoNCE loss. Spearman $( \rho _ { s } )$ and Pearson (r) correlations between mkNN and text unimodal properties over 24 text models, averaged ± std over 15 audio models. Notably, CMA is consistently highly correlated with the negative unimodal InfoNCE loss.
<table><tr><td> $\rho _ { s } / r$ </td><td colspan="2">Clotho</td><td colspan="2">AudioCaps</td></tr><tr><td>Model size</td><td></td><td> $0 . 0 5 _ { \pm 0 . 0 2 } / \quad 0 . 1 8 _ { \pm 0 . 0 6 }$ </td><td> $- 0 . 0 8 _ { \pm 0 . 0 5 } / . - 0 . 2 1 _ { \pm 0 . 0 3 }$ </td><td></td></tr><tr><td>Training-corpus size</td><td> $- 0 . 1 9 _ { \pm 0 . 0 1 } / - 0 . 1 0 _ { \pm 0 . 0 4 }$ </td><td></td><td>−0.22±0.04 /−0.30±0.04</td><td></td></tr><tr><td>InfoNCE loss (-L)</td><td></td><td>0.92±0.01 / 0.85±0.03</td><td> $\begin{array} { r l } { \mathbf { 0 . 8 0 { } _ { \pm 0 . 0 3 } } / } & { { } \mathbf { 0 . 9 7 { } _ { \pm 0 . 0 1 } } } \end{array}$ </td><td></td></tr></table>

Ablations. Due to lack of space, we defer ablations to the appendix. Specifically, we show the results are consistent across different kinds of text augmentations (App. D), other vision encoders (App. E), other audio encoders (App. F), other CMA metrics (App. G), different test set sizes (App. H), and different choices of temperature τ (App. I).

Message of Section 4: Across controlled training, frozen pre-trained encoders, and modalities, unimodal augmentation-based SSL loss predicts cross-modal alignment.

## 5 Conclusions and Summary

In this paper, we presented an explanation for the emergence of implicit cross-modal alignment through unimodal augmentation-based objectives. We established theoretical connections between the widely used InfoNCE loss and cross-modal alignment, as measured by local CKA. In particular, we showed that minimizing Aug-SSL unimodal objectives promotes high cross-modal alignment (Thm. 1), while sufficiently high unimodal loss restricts cross-modal alignment (Thm. 2). We then empirically validated these results in both controlled experiments (Sec. 4.1) and unconstrained pre-trained models (Sec. 4.2), demonstrating that unimodal augmentations are a strong driving force behind cross-modal alignment.

We believe this work provides a foundation for understanding the emergence of CMA from both theoretical and empirical perspectives. While Aug-SSL is commonly motivated by its ability to leverage large-scale data, our results reveal an additional, unexplored benefit: augmentation-based objectives can induce cross-modal alignment implicitly, and much more efficiently, without explicitly incorporating any cross-modal information into the training objective.

## References

Santiago Acevedo, Andrea Mascaretti, Riccardo Rende, Matéo Mahaut, Marco Baroni, and Alessandro Laio. A quantitative analysis of semantic information in deep representations of text and images. arXiv preprint arXiv:2505.17101, 2025.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460, 2020.

Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A general framework for self-supervised learning in speech, vision and language. In International conference on machine learning, pp. 1298–1312. PMLR, 2022.

Randall Balestriero and Yann LeCun. Contrastive and non-contrastive self-supervised learning recover global and local spectral embedding methods. Advances in Neural Information Processing Systems, 35:26671–26685, 2022.

Kiril Bangachev, Guy Bresler, and Yury Polyanskiy. Representation alignment rests on linear structure. arXiv preprint arXiv:2605.28870, 2026.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International conference on machine learning, pp. 2397–2430. PMLR, 2023.

Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal of Selected Topics in Signal Processing, 16(6): 1505–1518, 2022.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597–1607. PmLR, 2020.

Nello Cristianini, John Shawe-Taylor, Andre Elisseeff, and Jaz Kandola. On kernel-target alignment. Advances in neural information processing systems, 14, 2001.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Konstantinos Drossos, Samuel Lipping, and Tuomas Virtanen. Clotho: An audio captioning dataset. In ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 736–740. IEEE, 2020.

Kshitij Duraphe, Michael J Smith, Shashwat Sourav, John F Wu, et al. The platonic universe: Do foundation models see the same sky? arXiv preprint arXiv:2509.19453, 2025.

Sathya Edamadaka, Soojung Yang, Ju Li, and Rafael Gómez-Bombarelli. Universally converging representations of matter across scientific foundation models. arXiv preprint arXiv:2512.03750, 2025.

David Fan, Shengbang Tong, Jiachen Zhu, Koustuv Sinha, Zhuang Liu, Xinlei Chen, Michael Rabbat, Nicolas Ballas, Yann LeCun, Amir Bar, et al. Scaling language-free visual representation learning. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 1–13. IEEE, 2025.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 6894–6910, 2021.

Xinyang Geng and Hao Liu. Openllama: An open reproduction of llama, May 2023. URL https: //github.com/openlm-research/open\_llama.

Fabian Gröger, Shuo Wen, and Maria Brbic. Revisiting the platonic representation hypothesis: An´ aristotelian view. arXiv preprint arXiv:2602.14486, 2026a.

Fabian Gröger, Shuo Wen, Huyen Le, and Maria Brbic. With limited data for multimodal alignment, let the structure guide you. Advances in Neural Information Processing Systems, 38:151747– 151776, 2026b.

Jie Gui, Tuo Chen, Jing Zhang, Qiong Cao, Zhenan Sun, Hao Luo, and Dacheng Tao. A survey on self-supervised learning: Algorithms, applications, and future trends. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):9052–9071, 2024.

Sharut Gupta, Shobhita Sundaram, Chenyu Wang, Stefanie Jegelka, and Phillip Isola. Better together: Leveraging unpaired multimodal data for stronger unimodal models. In International Conference on Learning Representations, volume 2026, pp. 73970–74025, 2026.

Emman Haider, Daniel Perez-Becker, Thomas Portet, Piyush Madan, Amit Garg, Atabak Ashfaq, David Majercak, Wen Wen, Dongwoo Kim, Ziyi Yang, et al. Phi-3 safety post-training: Aligning language models with a" break-fix" cycle. arXiv preprint arXiv:2407.13833, 2024.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In 2020 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 9726–9735. IEEE, 2020.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In 2022 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 15979–15988. IEEE, 2022.

Zoe Wanying He, Sean Trott, and Meenakshi Khosla. Seeing through words, speaking through pixels: Deep representational alignment between vision and language models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pp. 35645–35660, 2025.

Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM transactions on audio, speech, and language processing, 29:3451–3460, 2021.

Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, and Christoph Feichtenhofer. Masked autoencoders that listen. Advances in neural information processing systems, 35:28708–28720, 2022.

Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. The platonic representation hypothesis. arXiv preprint arXiv:2405.07987, 2024.

Francesco Maria Gabriele Insulla, Shuo Huang, and Lorenzo Rosasco. Towards a learning theory of representation alignment. In International Conference on Learning Representations, volume 2025, pp. 37594–37608, 2025.

Rishi Jha, Collin Zhang, Vitaly Shmatikov, and John Morris. Harnessing the universal geometry of embeddings. Advances in Neural Information Processing Systems, 38:45963–45987, 2026.

Chaitanya Kapoor, Sudhanshu Srivastava, and Meenakshi Khosla. Bridging critical gaps in convergent learning: How representational alignment evolves across layers, training, and distribution shifts. Advances in Neural Information Processing Systems, 38:140352–140366, 2026.

Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in neural information processing systems, 33:18661–18673, 2020.

Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and Gunhee Kim. Audiocaps: Generating captions for audios in the wild. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 119–132, 2019.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International conference on machine learning, pp. 3519–3529. PMlR, 2019.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilic, Daniel Hesslow, Roman´ Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176bparameter open-access multilingual language model. 2022.

Kuang-Huei Lee, Anurag Arnab, Sergio Guadarrama, John Canny, and Ian Fischer. Compressive visual representations. Advances in Neural Information Processing Systems, 34:19538–19552, 2021.

Jiaang Li, Yova Kementchedjhieva, Constanza Fierro, and Anders Søgaard. Do vision and language models share concepts? a vector space alignment study. Transactions of the Association for Computational Linguistics, 12:1232–1249, 2024.

Zeyu Michael Li, Hung Anh Vu, Damilola Awofisayo, and Emily Wenger. Causes and consequences of representational similarity in machine learning models. arXiv preprint arXiv:2505.13899, 2025.

Zhenzhu Li and Aron Walsh. Platonic representation of foundation machine learning interatomic potentials. Nature Machine Intelligence, pp. 1–11, 2026.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740–755. Springer, 2014.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

Alexander Lobashev. An information-geometric view of the platonic hypothesis. In NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations, 2025.

Jianglin Lu, Hailing Wang, Kuo Yang, Yitian Zhang, Simon Jenni, and Yun Fu. The indra representation hypothesis for multimodal alignment. Advances in Neural Information Processing Systems, 38:156980–157000, 2026.

JW Miller. Testing the platonic representation hypothesis via representation-controlled tokenization. 2026.

Krikamol Muandet, Kenji Fukumizu, Bharath Sriperumbudur, and Bernhard Schölkopf. Kernel mean embedding of distributions: A review and beyond. Foundations and Trends® in Machine Learning, 10(1-2):1–141, 2017. ISSN 1935-8245. doi: 10.1561/2200000060. URL http: //dx.doi.org/10.1561/2200000060.

Jerry Ngo and Yoon Kim. What do language models hear? probing for auditory representations in language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 5435–5448, 2024.

Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith Hall, Daniel Cer, and Yinfei Yang. Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models. In Findings of the association for computational linguistics: ACL 2022, pp. 1864–1874, 2022.

Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino. Byol for audio: Self-supervised learning for general-purpose audio representation. In 2021 International Joint Conference on Neural Networks (IJCNN), pp. 1–8. IEEE, 2021.

Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino. Byol for audio: Exploring pre-trained general-purpose audio representations. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:137–151, 2022.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Maks Ovsjanikov, Viorica Patraucean, Leonidas Guibas, Tyler Zhu, and Tengda Han. Dynamic reflections: Probing video representations with text alignment. In International Conference on Learning Representations, volume 2026, pp. 71224–71250, 2026.

Rui Qian, Tianjian Meng, Boqing Gong, Ming-Hsuan Yang, Huisheng Wang, Serge Belongie, and Yin Cui. Spatiotemporal contrastive video representation learning. In 2021 IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 6960–6970. IEEE, 2021.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings ofthe 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pp. 3982–3992, 2019.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal ofcomputer vision, 115(3):211–252, 2015.

Aaqib Saeed, David Grangier, and Neil Zeghidour. Contrastive learning of general-purpose audio representations. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 3875–3879. IEEE, 2021.

Dominik Schnaus, Nikita Araslanov, and Daniel Cremers. It’sa (blind) match! towards visionlanguage correspondence without parallel data. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 24983–24992. IEEE, 2025.

Krishna Srinivasan, Karthik Raman, Jiecao Chen, Michael Bendersky, and Marc Najork. Wit: Wikipedia-based image text dataset for multimodal multilingual machine learning. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval, pp. 2443–2449, 2021.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle Casbon, et al. Gemma 4 technical report.˘ arXiv preprint arXiv:2607.02770, 2026.

Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, and Dilip Krishnan. Stablerep: Synthetic images from text-to-image models make strong visual representation learners. Advances in Neural Information Processing Systems, 36:48382–48402, 2023.

Megan Tjandrasuwita, Chanakya Ekbote, Liu Ziyin, and Paul Pu Liang. Understanding the emergence of multimodal representation alignment. arXiv preprint arXiv:2502.16282, 2025.

Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are dataefficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35:10078–10093, 2022.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023. URL https://arxiv.org/abs/2302.13971.

Ali Hussaini Umar and Alessandro Laio. Signal-to-noise ratio and sample size govern representational alignment in neural networks. arXiv preprint arXiv:2605.26973, 2026.

Aladin Virmaux and Kevin Scaman. Lipschitz regularity of deep neural networks: analysis and efficient estimation. Advances in neural information processing systems, 31, 2018.

Julius Von Kügelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Schölkopf, Michel Besserve, and Francesco Locatello. Self-supervised learning with data augmentations provably isolates content from style. Advances in neural information processing systems, 34:16451–16467, 2021.

Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International conference on machine learning, pp. 9929–9939. PMLR, 2020.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. Advances in neural information processing systems, 33:5776–5788, 2020.

Yuyang Wang, Jianren Wang, Zhonglin Cao, and Amir Barati Farimani. Molecular contrastive learning of representations via graph neural networks. Nature Machine Intelligence, 4(3):279–287, 2022.

Ho-Hsiang Wu, Prem Seetharaman, Kundan Kumar, and Juan Pablo Bello. Wav2clip: Learning robust audio representations from clip. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 4563–4567. IEEE, 2022.

Amitai Yacobi, Nir Ben-Ari, Ronen Talmon, and Uri Shaham. Learning shared representations from unpaired data. Advances in Neural Information Processing Systems, 38:46634–46666, 2026.

Lauren Hyoseo Yoon, Yisong Yue, and Been Kim. Escaping plato’s cave: Jam for aligning independently trained vision and language models. arXiv preprint arXiv:2507.01201, 2025.

Le Zhang, Qian Yang, and Aishwarya Agrawal. Assessing and learning alignment of unimodal vision and language models. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 14604–14614. IEEE, 2025.

Liu Ziyin and Isaac Chuang. Proof of a perfect platonic representation hypothesis. arXiv preprint arXiv:2507.01098, 2025.

Liu Ziyin, Yizhou Xu, and Isaac Chuang. Neural thermodynamics: Entropic forces in deep and universal representation learning. Advances in Neural Information Processing Systems, 38:56057– 56089, 2026.

## A Proofs

## A.1 Rationale

This section gives an informal overview of the argument: the objects it involves, the role played by each lemma, and how they chain into the two theorems. The formal settings are given in Sec. A.2, and the formal statements and proofs follow in Sec. A.3 and Sec. A.4.

Overview. The centered kernel $k _ { X } ^ { c }$ (Eq. 6) lies in $L ^ { 2 }$ with norm $\sigma _ { X }$ , and Eq. 7 expresses CKA as the cosine of the angle between $k _ { X } ^ { c }$ and $k _ { Y } ^ { c }$ . Since the encoders never interact directly, alignment arises through their shared semantic structure: each encoder is trained against its own augmentations, which act as view-specific information removers by filtering out view-specific variation. Minimizing the unimodal losses therefore steers each embedding space toward the shared semantics underlying the modalities, promoting cross-modal similarity. To formalize this connection, we introduce the augmentation gap $\Delta _ { X }$ , which measures the difference between a sample’s similarity to its augmentations and its similarity to random samples. The proofs of both theorems follow chains of inequalities, bounding the loss with the gap, and the gap with the CKA.

Theorem 1: low losses force alignment. To prove Thm. 1, we bound the angle between each kernel and the shared semantics separately, quantifying the fraction of the encoder structure explained by the shared semantics, and then combine the two bounds via the triangle inequality. In details, Lemma 1 involves the basic components underlying both loss and the CKA: the inner products between a sample and its augmentations, and between a sample and independently sampled points, both of which are bounded in terms of the loss. Lemma 2 then lower-bounds the augmentation gap by the loss, while Lemma 3 relates the augmentation gap to the fraction of the encoder structure explained by the semantics (induced by the augmentations). Lemma 4 shows that the kernel variance, which appears in the denominator of CKA, is upper-bounded in terms of the loss. Finally, Lemma 9 and Prop. 1 establish that the bound in Thm. 1 is non-increasing in both losses and attains 1 when both reach their minima.

Theorem 2: a high loss prevents alignment. To prove Thm. 2, we bound the fraction of each encoder’s structure that is able to carry cross-modal alignment at all, and then bound that fraction by the loss. In details, Lemma 5 shows that conditioned on the shared semantics the two embedding spaces are independent, so only the semantic part of each kernel contributes to the CKA. It therefore upper-bounds the CKA by the semantic variance $\beta _ { X }$ , measuring how much the representation varies with the semantics, normalized by the kernel variance. Lemma 6 then relates the semantic variance to the augmentation gap, up to a term measuring how much the augmentations displace the observations, which is where the Lipschitz assumption (Assumption 1) is used. Lemma 7 completes the chain by upper-bounding the augmentation gap in terms of the loss, complementing Lemma 2. Finally, Prop. 2 establishes that the bound in Thm. 2 is strictly decreasing in the loss, and since it involves a single modality, a high loss in either one alone caps the CKA.

## A.2 Setting

Every statement made for X holds verbatim for $Y ,$

Data. We fix a probability measure µ on the semantic manifold $\mathcal { M } ,$ and nuisance laws $\nu _ { \mathcal { U } }$ on $\mathcal { U }$ and $\nu _ { \mathcal { V } }$ on V. Observations are $g _ { X } ( m , u ) \in \mathcal X$ and $g _ { Y } ( m , v ) \in \mathcal { Y }$ , and the trained encoders send them to the unit spheres of the embedding spaces $\mathcal { E } _ { X } , \mathcal { E } _ { Y }$

$$
\begin{array} { r l r l } & { \boldsymbol { x } = e _ { X } \left( g _ { X } ( m , u ) \right) \in \mathcal { E } _ { X } \subseteq \mathbb { S } ^ { d _ { X } - 1 } , } & & { \boldsymbol { y } = e _ { Y } \left( g _ { Y } ( m , v ) \right) \in \mathcal { E } _ { Y } \subseteq \mathbb { S } ^ { d _ { Y } - 1 } . } \end{array}
$$

Note that for readability, the appendix refers to $x \in \mathcal { E } _ { X }$ and $y \in { \mathcal { E } } _ { Y }$ as embedding points rather than observations. We write $\rho _ { X }$ for the law of x and $\bar { x } = \mathbb { E } [ x ]$ . Throughout, $( m , u , v )$ and $( m ^ { \prime } , u ^ { \prime } , v ^ { \prime } )$ are two independent draws, and all four of $x , x ^ { \prime } , y , y ^ { \prime }$ are built from them, so that x, y share the semantics m and $x ^ { \prime } , y ^ { \prime }$ share $m ^ { \prime }$

Every object below is a square-integrable function of the pair $\big ( ( m , u , v ) , ( m ^ { \prime } , u ^ { \prime } , v ^ { \prime } ) \big )$ . We write $L ^ { 2 }$ for the space of such functions, with ⟨·, ·⟩ and ∥·∥ its inner product and norm.

Kernel. Fix ω $> 0$ . On the sphere $\| x - x ^ { \prime } \| ^ { 2 } = 2 - 2 \langle x , x ^ { \prime } \rangle$ , so the Gaussian kernel of bandwidth $\omega ^ { - 1 / 2 }$ is exactly

$$
k ( x , x ^ { \prime } ) = e ^ { \omega ( \langle x , x ^ { \prime } \rangle - 1 ) } .\tag{5}
$$

We abbreviate $k _ { X } = k ( x , x ^ { \prime } )$ and $k _ { Y } = k ( y , y ^ { \prime } )$ , and use ω rather than a bandwidth symbol so that nothing collides with the kernel scales $\sigma _ { X } , \sigma _ { Y }$ . The centred kernel and the alignment are

$$
k _ { X } ^ { c } = k _ { X } - \mathbb { E } [ k _ { X } \mid x ] - \mathbb { E } [ k _ { X } \mid x ^ { \prime } ] + \mathbb { E } [ k _ { X } ] , \qquad \sigma _ { X } ^ { 2 } = \mathbb { E } \big [ ( k _ { X } ^ { c } ) ^ { 2 } \big ] ,\tag{6}
$$

$$
\mathrm { C K A } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) = \frac { \mathrm { H S I C } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) } { \sqrt { \mathrm { H S I C } ( \mathcal { E } _ { X } , \mathcal { E } _ { X } ) \mathrm { H S I C } ( \mathcal { E } _ { Y } , \mathcal { E } _ { Y } ) } } = \frac { \mathbb { E } [ k _ { X } ^ { c } k _ { Y } ^ { c } ] } { \sigma _ { X } \sigma _ { Y } } = \langle \frac { k _ { X } ^ { c } } { \sigma _ { X } } , \frac { k _ { Y } ^ { c } } { \sigma _ { Y } } \rangle ,\tag{7}
$$

the latter defined whenever $\sigma _ { X } \sigma _ { Y } > 0$ . Equation 7 exhibits the CKA as the cosine of an angle in $L ^ { 2 }$ between two unit vectors; both directions exploit this.

Three scalars.

$$
\hat { A } _ { X } : = \mathbb { E } \big [ k ( x , x ^ { + } ) \big ] , \qquad \hat { S } _ { X } : = \mathbb { E } \big [ k ( x , x ^ { \prime } ) \big ] , \qquad \Delta _ { X } : = \hat { A } _ { X } - \hat { S } _ { X } .\tag{8}
$$

${ \hat { A } } _ { X }$ is how similar a sample looks to its own augmentation, $\hat { S } _ { X }$ how similar two unrelated samples look, and the augmentation gap $\Delta _ { X }$ the excess of the former over the latter.

Definition 1 (Augmentation). An augmentation redraws both the semantics and the nuisance, while keeping the semantics close. That is,

$$
x ^ { + } = e _ { X } \bigl ( g _ { X } ( m ^ { + } , u ^ { + } ) \bigr ) , \qquad m ^ { + } \sim P ( m , \cdot ) , \quad u ^ { + } \sim \nu _ { \mathcal { U } } \mathrm { i n d e p e n d e n t } ,\tag{9}
$$

where $P ( m , \cdot ) = \mathsf { J } ( m , \cdot ) \mu$ is a Markov kernel on M with density J.

Write $\pi ( m , m ^ { + } ) = \mathsf { J } ( m , m ^ { + } ) \mu ( m ) \mu ( m ^ { + } )$ for the joint law of the semantic pair.

The concentration strength of the augmentation is

$$
\chi : = \| \mathsf { J } - 1 \| _ { L ^ { 2 } ( \mu \otimes \mu ) } .\tag{10}
$$

We assume $\mathsf { J } \in L ^ { 2 } ( \mu \otimes \mu )$ and $\chi > 0 ;$ ; the degenerate case $\chi = 0$ means $\mathsf { J } \equiv 1$ , i.e. the augmentation resamples the semantics independently and carries no signal. Both marginals of π are $\mu ,$ so x and $x ^ { + }$ each have marginal law $\rho _ { X }$ , a fact used repeatedly below.

The objective. With $0 < \tau \leq 1$ the temperature, the population InfoNCE loss and its normalisation are

$$
N _ { X } = - \frac { A _ { X } } { \tau } + U _ { X } , \qquad A _ { X } : = \mathbb { E } \big [ \langle x , x ^ { + } \rangle \big ] , \qquad U _ { X } : = \mathbb { E } _ { x } \Big [ \log \mathbb { E } _ { x ^ { \prime } } \big [ e ^ { \langle x , x ^ { \prime } \rangle / \tau } \big ] \Big ] ,\tag{11}
$$

$$
L _ { X } = \frac { N _ { X } + 1 / \tau } { R } , \qquad R = \textstyle { \frac { 2 } { \tau } } + \log \cosh \frac { 1 } { \tau } .\tag{12}
$$

The derivations often use

$$
\begin{array} { r } { \tau R = 2 + \tau \log \cosh \frac { 1 } { \tau } ~ > ~ 2 . } \end{array}\tag{13}
$$

The constants of Section A.3 are

$$
\begin{array} { r } { a : = e ^ { - \omega } \sinh \omega \in ( 0 , \frac { 1 } { 2 } ) , \quad b : = \bigl ( \omega + e ^ { - \omega } \sinh \omega \bigr ) \tau R , \quad D : = \bigl ( 1 - e ^ { - \omega } \bigr ) ^ { 2 } \Bigl ( 1 + \frac { 4 } { \log 2 } \Bigr ) \tau R . } \end{array}\tag{14}
$$

## A.3 Implicit Alignment

This section proves the implicit alignment theorem. That is, small unimodal losses restrict the CKA to be large (Thm. 1). The key is the augmentations, which are accessed via $J - 1$ , which depends only on the shared semantics and is therefore common to both modalities. Since the CKA is the cosine of the angle between the centered kernels $k _ { X } ^ { c }$ and $k _ { Y } ^ { c } ~ ( \mathrm { E q . } ~ 7 )$ , it suffices to show that each kernel is close in angle to $J - 1$ . This angle is determined by two quantities: the inner product $\langle k _ { X } ^ { c } , J - 1 \rangle$ measuring how much of $k _ { X } ^ { c }$ points in the direction of $J - \mathrm { i }$ , and the length of $\bar { k _ { X } ^ { c } }$ , namely the kernel scale $\sigma _ { X }$ . Lemma 3 shows that the inner product equals the augmentation gap $\Delta _ { X } ,$ i.e., how much more similar a sample is to its augmentation than to a random sample. Lemma $2$ shows that a low loss forces this gap to be large, while Lemma 4 shows that it forces $\sigma _ { X }$ to be small, both using the moment bounds of Lemma 1. Together, a low loss forces $k _ { X } ^ { c }$ to align with $J - 1$ , and likewise for $k _ { Y } ^ { c }$ . Thm. 1 then combines the two angles via the triangle inequality, forcing the two kernels to align with each other.

Lemma 1 (Moments).

$$
\begin{array} { r } { \| \bar { x } \| ^ { 2 } \leq \tau R L _ { X } , \qquad { \mathbb { E } } \big [ \langle x , x ^ { \prime } \rangle ^ { 2 } \big ] \leq \Big ( 1 + \frac { 4 } { \log 2 } \Big ) \tau R L _ { X } , \qquad A _ { X } \geq 1 - \tau R L _ { X } . } \end{array}
$$

Proof. Write $r ( x ) = \mathbb { E } [ \langle x , x ^ { \prime } \rangle \mid x ] = \langle x , { \bar { x } } \rangle$ and $V ( x ) = \mathrm { V a r } ( \langle x , x ^ { \prime } \rangle \mid x )$ . Conditionally on x, the variable $\delta = \langle \dot { x } , \dot { x } ^ { \prime } \rangle - \dot { r } ( x )$ has mean zero and lies in an interval of length 2, so Lemma 11 with $\lambda = 1 / \tau \geq 1$ gives log $\begin{array} { r } { \mathbb { E } [ e ^ { \delta / \tau } \mid x ] \ge \frac { \log 2 } { 4 \tau } V ( x ) } \end{array}$ . Averaging and using $\mathbb { E } [ r ( x ) ] = \| { \bar { x } } \| ^ { 2 }$

$$
U _ { X } = \frac { \mathbb { E } [ r ( x ) ] } { \tau } + \mathbb { E } \left[ \log \mathbb { E } [ e ^ { \delta / \tau } \mid x ] \right] \ge \frac { \| \bar { x } \| ^ { 2 } } { \tau } + \frac { \log 2 } { 4 \tau } \mathbb { E } [ V ] \ge 0 .\tag{15}
$$

Since $A _ { X } ~ \leq ~ 1$ , Eq. 11 gives $U _ { X } \ = \ N _ { X } + A _ { X } / \tau \ \le N _ { X } + 1 / \tau \ = \ R L _ { X }$ . Both terms on the last inequality of Eq. 15 are nonnegative, so each is at most $R L _ { X } \colon$ this is the first bound, and $\mathbb { E } [ V ] \ \stackrel { * } { \leq } \ \frac { 4 } { \log { 2 } } \tau R L _ { X }$ . The second bound follows from $\mathbb { E } [ \langle x , x ^ { \prime } \rangle ^ { 2 } ] ^ { - } = \mathbb { E } [ V ] + \mathbb { E } [ r ( x ) ^ { 2 } ]$ and $r ( x ) ^ { 2 } \leq \| { \bar { x } } \| ^ { 2 } \leq \tau R L _ { X }$ . For the third, $U _ { X } \geq 0$ in Eq. 11 gives $A _ { X } / \tau = U _ { X } - N _ { X } \geq - N _ { X } =$ $\begin{array} { r } { \frac { 1 } { \tau } - R L _ { X } } \end{array}$ □

Lemma 2 (The loss controls the gap from below). $\Delta _ { X } \ \ge \ a - b L _ { X }$ , with a, b as in Eq. 14.

Proof. For ${ \hat { A } } _ { X }$

$$
\hat { A } _ { X } = \mathbb { E } \big [ e ^ { \omega ( \langle x , x ^ { + } \rangle - 1 ) } \big ] \geq e ^ { \omega ( A _ { X } - 1 ) } \geq e ^ { - \omega \tau R L _ { X } } \geq 1 - \omega \tau R L _ { X } ,
$$

where the first inequality is Jensen, then the third bound of Lemma 1, then $e ^ { - t } \geq 1 - t .$

For $\hat { S } _ { X } = e ^ { - \omega } \mathbb { E } [ e ^ { \omega \langle x , x ^ { \prime } \rangle } ]$ , Lemma 10(i) with $\alpha = \omega , c = \mathbb { E } [ \langle x , x ^ { \prime } \rangle ] = \| \bar { x } \| ^ { 2 }$

$$
\hat { S } _ { X } \leq e ^ { - \omega } \left( \cosh \omega + \| \bar { x } \| ^ { 2 } \sinh \omega \right) .
$$

Applying the first bound of Lemma 1 gives

$$
\hat { S } _ { X } \leq e ^ { - \omega } \cosh \omega + e ^ { - \omega } \tau R L _ { X } \sinh \omega .
$$

Subtracting and using $1 - e ^ { - \omega } \cosh \omega = \frac { 1 } { 2 } ( 1 - e ^ { - 2 \omega } ) = e ^ { - \omega }$ sinh ω gives $\Delta _ { X } \geq a - b L _ { X }$ □

Lemma 3 (Centered kernel and the augmentation gap). $\begin{array} { r } { \langle k _ { X } ^ { c } , \mathsf { J } - 1 \rangle = \Delta _ { X } , a n d \| \mathsf { J } - 1 \| = \chi . } \end{array}$

Proof. J − 1 is a function of $( m , m ^ { \prime } )$ alone, so for any $f \in L ^ { 2 }$ , writing $\bar { f } ( m , m ^ { \prime } ) = \mathbb { E } [ f \mid m , m ^ { \prime } ]$ and using $\mathbb { E } _ { \mu \otimes \mu } [ \bar { f } \mathsf { J } ] = \mathbb { E } _ { \pi } [ \bar { f } ]$

$$
\langle f , \mathsf { J } - 1 \rangle = \mathbb { E } _ { \mu \otimes \mu } { \big [ } { \bar { f } } \left( \mathsf { J } - 1 \right) { \big ] } = \mathbb { E } _ { \pi } [ f ] - \mathbb { E } [ f ] .\tag{16}
$$

Apply Eq. 16 to $f = k _ { X } ^ { c }$ . By centering Eq. $6 , \mathbb { E } [ k _ { X } ^ { c } ] = 0$ . For the π-expectation,

$$
\mathbb { E } _ { \pi } [ k _ { X } ^ { c } ] = \mathbb { E } _ { \pi } [ k _ { X } ] - \mathbb { E } _ { \pi } \mathbb { E } [ k _ { X } \ \vert \ x ] - \mathbb { E } _ { \pi } \mathbb { E } [ k _ { X } \ \vert \ x ^ { + } ] + \mathbb { E } _ { \pi } \mathbb { E } [ k _ { X } ] .
$$

Because the marginals of π are $\mu , \operatorname { \mathbb { E } } _ { \pi } \operatorname { \mathbb { E } } [ k _ { X } ~ | ~ x ] = \operatorname { \mathbb { E } } _ { \pi } \operatorname { \mathbb { E } } [ k _ { X } ~ | ~ x ^ { + } ] = \operatorname { \mathbb { E } } [ k _ { X } ]$ , hence

$$
\mathbb { E } _ { \pi } [ k _ { X } ^ { c } ] = \hat { A } _ { X } - \hat { S } _ { X } - \hat { S } _ { X } + \hat { S } _ { X } = \Delta _ { X } ,
$$

and Eq. 16 gives the claim.

The norm is Eq. 10: the marginal of $L ^ { 2 } \mathrm { o n } ( m , m ^ { \prime } ) \mathrm { i s } \mu \otimes \mu ,$ , so embedding a function of $( m , m ^ { \prime } )$ into $L ^ { 2 }$ preserves its norm. □

Lemma 4 (Kernel scale). $\sigma _ { X } ^ { 2 } ~ \le ~ D L _ { X }$ , with D as in Eq. 14.

Proof. Let $\begin{array} { r } { C g = g - \mathbb { E } [ g \mid x ] - \mathbb { E } [ g \mid x ^ { \prime } ] + \mathbb { E } [ g ] } \end{array}$ , so that $k _ { X } ^ { c } = C k _ { X }$ . Since $x \perp x ^ { \prime }$ we may factor $C = ( \mathrm { I d } - \bar { P } _ { 1 } ) ( \bar { \mathrm { I d } } - \bar { P _ { 2 } } )$ with $\bar { P _ { 1 } g } = \bar { \mathbb { E } } [ g | \mathbf { \bar { \Phi } } x ^ { \prime } ]$ and $P _ { 2 } \boldsymbol { g } ^ { \cdot \cdot } = \mathbb { E } [ \boldsymbol { g } \mid \boldsymbol { x } ] ;$ each $P _ { i }$ is an orthogonal projection, so each Id $. - P _ { i }$ is a contraction and $\| C g \| \overset { \cdot } { \leq } \| g \| . \ \overset { \cdot } { \mathrm { A s } } \overset { \cdot } { C c } = 0$ for every constant c, $\bar { \sigma _ { X } } = \| k _ { X } ^ { c } \| = \| C ( k _ { X } - c ) \| \leq \| k _ { X } - c \|$ for every constant c.

Take $c = e ^ { - \omega }$ . By Eq. 5, $k _ { X } - e ^ { - \omega } = e ^ { - \omega } \big ( e ^ { \omega \langle x , x ^ { \prime } \rangle } - 1 \big )$ , and Lemma 10(ii) with $\alpha = \omega$ gives $| k _ { X } - e ^ { - \omega } | = e ^ { - \omega } | { \bigl ( } e ^ { \omega \langle x , x ^ { \prime } \rangle } - 1 { \bigr ) } | \leq ( 1 - e ^ { - \omega } ) | \langle x , x ^ { \prime } \rangle |$ . Squaring and applying the second bound of Lemma 1 gives $\sigma _ { X } ^ { 2 } \le D L _ { X }$ □

Theorem 1 (Implicit alignment). Suppose $\sigma _ { X } \sigma _ { Y } > 0$ and $\chi > 0$ . Define, with $a , b , D$ as in Eq. 14,

$$
c ( L ) : = \frac { \left[ a - b L \right] _ { + } } { \chi \sqrt { D L } } , \qquad c _ { X } = c ( L _ { X } ) , \quad c _ { Y } = c ( L _ { Y } ) .\tag{17}
$$

Then $c _ { X } , c _ { Y } \in [ 0 , 1 ]$ and

$$
\mathrm { C K A } ( { \mathcal E } _ { X } , { \mathcal E } _ { Y } ) \ge c _ { X } c _ { Y } - \sqrt { ( 1 - c _ { X } ^ { 2 } ) ( 1 - c _ { Y } ^ { 2 } ) } = : f ( L _ { X } , L _ { Y } ) 
$$

Proof. Lemma 4 gives $L _ { X } \ge \sigma _ { X } ^ { 2 } / D > 0$ because $\sigma _ { X } > 0$ , so $c ( L _ { X } )$ is well defined; likewise $c ( L _ { Y } )$

$\mathrm { I f } c _ { X } = 0 \mathrm { o r } c _ { Y } = 0$ the claimed right-hand side $\mathrm { i s } \leq 0 .$ , and the bound follows from $\mathrm { C K A } \geq 0$ (see $\operatorname { E q . 7 } )$ . Assume therefore $a - b L _ { X } > 0$ and $a - b L _ { Y } > 0$ , and work in $L ^ { 2 }$ with the three unit vectors

$$
\hat { a } = \frac { k _ { X } ^ { c } } { \sigma _ { X } } , \qquad \hat { b } = \frac { k _ { Y } ^ { c } } { \sigma _ { Y } } , \qquad \hat { \jmath } = \frac { { \bf J } - 1 } { \chi } ,
$$

and let $\theta _ { X } = \angle ( \hat { a } , \hat { \ j } ) , \theta _ { Y } = \angle ( \hat { b } , \hat { \ j } )$ . Lemma 3 evaluates the cosine, Lemma 2 bounds its numerator below by a positive quantity, and Lemma 4 bounds its denominator above:

$$
\cos \theta _ { X } = \frac { \langle k _ { X } ^ { c } , \mathsf { J } - 1 \rangle } { \chi \sigma _ { X } } = \frac { \Delta _ { X } } { \chi \sigma _ { X } } \ \ge \ \frac { a - b L _ { X } } { \chi \sqrt { D L _ { X } } } = c _ { X } \ > \ 0 ,\tag{18}
$$

and likewise cos $\theta _ { Y } \geq c _ { Y } > 0$ In particular $c _ { X } \ \leq$ cos $\theta _ { X } \ \leq \ 1$ , so $c _ { X } ~ \in ~ ( 0 , 1 ]$ and $\theta _ { X } \ \leq$ arccos $c _ { X } \leq \pi / 2 ;$ the same for Y.

The angle $\angle ( \cdot , \cdot ) = \operatorname { a r c c o s } \langle \cdot , \cdot \rangle$ is the geodesic distance on the unit sphere of $L ^ { 2 }$ , hence satisfies the triangle inequality:

$$
Z ( \hat { a } , \hat { b } ) \leq \angle ( \hat { a } , \hat { \jmath } ) + \angle ( \hat { b } , \hat { \jmath } ) = \theta _ { X } + \theta _ { Y } \leq \operatorname { a r c c o s } c _ { X } + \operatorname { a r c c o s } c _ { Y } \leq \pi .
$$

Apply cos, decreasing on $[ 0 , \pi ] ,$ and expand cos(arccos $\begin{array} { l l l } { { c _ { X } ~ + ~ \operatorname { a r c c o s } c _ { Y } \hfill } } & { { = } } & { { c _ { X } c _ { Y } ~ - \hfill } } \end{array}$ $\sqrt { ( 1 - c _ { X } ^ { 2 } ) ( 1 - c _ { Y } ^ { 2 } ) }$ . Noting cos $\angle ( \hat { a } , \hat { b } ) = \mathrm { C K A } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } )$ gives

$$
\mathrm { C K A } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) \geq c _ { X } c _ { Y } - \sqrt { ( 1 - c _ { X } ^ { 2 } ) ( 1 - c _ { Y } ^ { 2 } ) }
$$

## A.4 Implicit misalignment

This section gives the complement direction and proves the implicit misalignment theorem (Thm. 2). That is, a large unimodal loss restricts the CKA to be small. The key observation is that only semantic variation can align across modalities. Lemma 5 formalizes this: given the shared semantics, the two embeddings are independent, so only the semantic part of each kernel contributes to the CKA. Hence, the CKA is upper-bounded by the semantic variance $\beta _ { X } , \mathrm { i . e }$ ., how much the embedding varies with the semantics, relative to its overall variation $\sigma _ { X }$ . It then remains to show that a large loss forces $\beta _ { X }$ to be small. Lemma 6 bounds the semantic variance by the augmentation gap $\Delta { _ { X } }$ , up to how much the augmentations displace the observations, which is where the Lipschitz assumption (Assumption 1) is used. Lemma 7 shows that a large loss forces this gap to be small. Together, a large loss leaves much of the variation view-specific, and Thm. 2 chains these inequalities into an upper bound on the CKA.

## A.4.1 CKA and the semantic variance

Notations. Let $\phi ( x ) = k ( x , \cdot )$ be the canonical feature map of the Gaussian kernel (Eq. 5) into its RKHS H (Muandet et al., 2017), so that $\langle \phi ( x ) , \phi ( x ^ { \prime } ) \rangle _ { \mathcal { H } } = \bar { k ( x , x ^ { \prime } ) }$ and $\| \phi ( x ) \| _ { \mathcal { H } } = 1$ on the sphere. Put

$$
\mu _ { m } : = \mathbb { E } [ \phi ( x ) \mid m ] , \qquad \bar { \mu } : = \mathbb { E } [ \phi ( x ) ] , \qquad \delta _ { m } : = \mu _ { m } - \bar { \mu } ,\tag{19}
$$

and define the semantic kernel and the semantic variance induced by X

$$
h _ { X } ( m , m ^ { \prime } ) : = \mathbb { E } \bigl [ k _ { X } ^ { c } \mid m , m ^ { \prime } \bigr ] , \qquad \beta _ { X } : = \mathbb { E } _ { m \sim \mu } \bigl [ h _ { X } ( m , m ) \bigr ] .\tag{20}
$$

Lemma 5. $h _ { X } ( m , m ^ { \prime } ) = \langle \delta _ { m } , \delta _ { m ^ { \prime } } \rangle _ { \mathcal { H } }$ and $\beta _ { X } = \mathbb { E } _ { m } \| \delta _ { m } \| _ { \mathcal { H } } ^ { 2 } \geq 0 .$ . Moreover, writing $s _ { X } : = \| h _ { X } \|$

$$
\begin{array} { r } { \mathrm { C K A } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) \leq \frac { s _ { X } } { \sigma _ { X } } \leq \frac { \beta _ { X } } { \sigma _ { X } } . } \end{array}
$$

Proof. Expanding Eq. 6 with $\operatorname { \mathbb { E } } [ k _ { X } \mid x ] = \langle \phi ( x ) , \bar { \mu } \rangle _ { \mathcal { H } }$ gives the pointwise identity $k _ { X } ^ { c } = \left. \phi ( x ) - \right.$ $\bar { \mu } , \phi ( x ^ { \prime } ) { - } \bar { \mu } \rangle _ { \mathcal { H } }$ . Given $( m , m ^ { \prime } )$ the nuisances $u , u ^ { \prime }$ are independent, so taking conditional expectations inside the inner product yields $h _ { X } ( m , m ^ { \prime } ) = \langle \delta _ { m } , \delta _ { m ^ { \prime } } \rangle \bar { \varkappa } ;$ in particular $\bar { h _ { X } } ( m , m ) = \| \bar { \delta } _ { m } \| _ { \mathcal { H } } ^ { 2 }$ and $\beta _ { X } = \mathbb { E } _ { m } \| \delta _ { m } \| _ { \mathcal { H } } ^ { 2 }$

Because $( u , u ^ { \prime } )$ and $( \boldsymbol { v } , \boldsymbol { v } ^ { \prime } )$ are independent of each other and of $( m , m ^ { \prime } )$ , the pairs $( x , x ^ { \prime } )$ and $( y , y ^ { \prime } )$ are conditionally independent given $( m , m ^ { \prime } )$ , so $\mathbb { E } [ k _ { X } ^ { c } k _ { Y } ^ { c } \ | \ m , m ^ { \prime } ] = h _ { X } h _ { Y }$ and $\mathrm { H S I C } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) \stackrel { \textstyle \cdot } { = }$ $\langle h _ { X } , h _ { Y } \rangle$ . Now $h _ { X }$ is the orthogonal projection of $k _ { X } ^ { c }$ onto the functions of $( m , m ^ { \prime } )$ , hence $s _ { X } \leq \sigma _ { X }$ and likewise $s _ { Y } \leq \sigma _ { Y }$ . Cauchy–Schwarz in $L ^ { 2 }$ gives $\mathrm { H S I C } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) \leq s _ { X } s _ { Y } \leq s _ { X } \sigma _ { Y } ;$ divide by $\sigma _ { X } \sigma _ { Y }$

Finally, Cauchy–Schwarz in $\mathcal { H }$ and independence of $m , m ^ { \prime }$ give

$$
s _ { X } ^ { 2 } = \mathbb { E } _ { m , m ^ { \prime } } \langle \delta _ { m } , \delta _ { m ^ { \prime } } \rangle ^ { 2 } \leq \mathbb { E } _ { m } \| \delta _ { m } \| ^ { 2 } \mathbb { E } _ { m ^ { \prime } } \| \delta _ { m ^ { \prime } } \| ^ { 2 } = \beta _ { X } ^ { 2 } .
$$

## A.4.2 Semantic displacement and the augmentation gap

Assumption 1 (Lipschitz encoders). $\| e _ { X } ( p ) - e _ { X } ( q ) \| \le \ell _ { X } d _ { \mathcal { X } } ( p , q )$ for all $p , q \in \mathcal { X }$ , and likewise for $e _ { Y }$ with constant $\ell _ { Y }$

Definition 2 (Semantic displacement).

$$
\varepsilon _ { X } : = \mathbb { E } _ { \pi , u } \Big [ d _ { \mathcal { X } } \big ( g _ { X } ( m , u ) , g _ { X } ( m ^ { + } , u ) \big ) ^ { 2 } \Big ] ,\tag{21}
$$

the mean-square distance the augmentation moves an observation when the nuisance is held fixed, with $d _ { \mathcal { X } }$ a metric on $\mathcal { X }$

Lemma 6 (Semantic variance versus augmentation gap). Under Assumption 1,

$$
\begin{array} { r } { \beta _ { X } \leq \Delta _ { X } + \frac 1 2 \omega \ell _ { X } ^ { 2 } \varepsilon _ { X } . } \end{array}
$$

Proof. By Lemma 3 and Eq. 1 $6 , \Delta _ { X } = \mathbb { E } _ { \pi } [ k _ { X } ^ { c } ] = \mathbb { E } _ { \pi } [ h _ { X } ] = \mathbb { E } _ { \pi } \langle \delta _ { m } , \delta _ { m ^ { + } } \rangle$ . Both marginals of π are $\mu ,$ so $\mathbb { E } _ { \pi } \Vert \delta _ { m } \Vert ^ { 2 } = \mathbb { E } _ { \pi } \Vert \bar { \delta _ { m ^ { + } } } \Vert ^ { 2 } = \beta _ { X }$ and

$$
\begin{array} { r } { \mathbb { E } _ { \boldsymbol \pi } \left\| \mu _ { m } - \mu _ { m ^ { + } } \right\| ^ { 2 } = \mathbb { E } _ { \boldsymbol \pi } \left\| \delta _ { m } - \delta _ { m ^ { + } } \right\| ^ { 2 } = 2 \beta _ { X } - 2 \Delta _ { X } . } \end{array}
$$

For the bound, couple the two mean embeddings through the same nuisance: with $\psi = \phi \circ e _ { X }$ , $\mu _ { m } - \mu _ { m ^ { + } } = \mathbb { E } _ { u } [ \bar { \psi ( } g _ { X } ( m , u ) ) - \psi ( g _ { X } ( m ^ { + } , u ) ) ]$ ], so by Jensen

$$
\begin{array} { r } { \| \mu _ { m } - \mu _ { m ^ { + } } \| \leq \mathbb { E } _ { u } \| \psi ( g _ { X } ( m , u ) ) - \psi ( g _ { X } ( m ^ { + } , u ) ) \| . } \end{array}
$$

On the sphere, $1 - e ^ { - t } \leq t$ gives

$$
\begin{array} { r } { \| \phi ( p ) - \phi ( q ) \| _ { \mathcal { H } } ^ { 2 } = 2 \big ( 1 - e ^ { - \omega \| p - q \| ^ { 2 } / 2 } \big ) \leq \omega \| p - q \| ^ { 2 } , } \end{array}
$$

so $\phi$ is $\sqrt { \omega } - \mathrm { I }$ Lipschitz and $\psi$ is $\sqrt { \omega } \ell _ { X } - \mathrm { I }$ Lipschitz. Hence

$$
\begin{array} { r } { \Vert \mu _ { m } - \mu _ { m ^ { + } } \Vert \leq \sqrt { \omega } \ell _ { X } \mathbb { E } _ { u } d _ { \mathcal { X } } ( g _ { X } ( m , u ) , g _ { X } ( m ^ { + } , u ) ) , } \end{array}
$$

and a second application of Jensen gives $\mathbb { E } _ { \boldsymbol \pi } \| \mu _ { m } - \mu _ { m ^ { + } } \| ^ { 2 } \leq \omega \ell _ { X } ^ { 2 } \mathbb { E } _ { \boldsymbol \pi , \boldsymbol u } [ d _ { \mathcal { X } } ^ { 2 } ] = \omega \ell _ { X } ^ { 2 } \varepsilon _ { X }$

Write

$$
\begin{array} { r } { \Lambda ( \tau ) : = \coth \frac { 1 } { \tau } - \tau + \tau \log \left( \tau \operatorname { t a n h } \frac { 1 } { \tau } \right) - 1 \ \in \ ( - 1 , 0 ) . } \end{array}\tag{22}
$$

Lemma 7 (The loss controls the gap from above). Suppose τ $\omega \leq 1$ . Then,

$$
\Delta _ { X } \ \leq \ a \ \big ( \Lambda ( \tau ) + \tau R ( 1 - L _ { X } ) \big ) .
$$

Proof. The objective controls two quantities: the alignment $A _ { X }$ from above and the uniformity $T : = \mathbb { E } \big [ e ^ { \langle x , x ^ { \prime } \rangle / \tau } \big ] > 0$ from below. Chord bounds convert ${ \hat { A } } _ { X }$ and ${ \hat { S } } _ { X }$ into exactly those two, after which the constraint linking them makes T cancel.

First, Lemma 10(i) with $\alpha = \omega$ gives $e ^ { \omega } { \hat { A } } _ { X } \leq$ cosh $\omega + A _ { X }$ sinh ω. Second, Lemma 10(iii) with $b = \omega \leq \alpha = 1 / \tau$ , gives,

$$
e ^ { \omega \langle x , x ^ { \prime } \rangle } \geq e ^ { - \omega } + \frac { \sinh \omega } { \sinh \frac { 1 } { \tau } } \big ( e ^ { \frac { 1 } { \tau } \langle x , x ^ { \prime } \rangle } - e ^ { - \frac { 1 } { \tau } } \big ) .
$$

After averaging,

$$
e ^ { \omega } \hat { S } _ { X } = \mathbb { E } [ e ^ { \omega \langle x , x ^ { \prime } \rangle } ] \geq e ^ { - \omega } + \frac { \sinh \omega } { \sinh \frac { 1 } { \tau } } \big ( T - e ^ { - 1 / \tau } \big ) .
$$

Subtracting ${ \hat { A } } _ { X } - { \hat { S } } _ { X }$ , and using cosh $\omega - e ^ { - \omega } =$ sinh ω,

$$
\frac { e ^ { \omega } \Delta _ { X } } { \sinh \omega } \leq 1 + A _ { X } - \frac { T - e ^ { - 1 / \tau } } { \sinh \frac { 1 } { \tau } } .\tag{23}
$$

Now eliminate A . By Eq. 11 and Jensen $( U _ { X } \leq \log T )$ ), $A _ { X } = \tau ( U _ { X } - N _ { X } ) \leq \tau$ (log $T - N _ { X } )$ Bound the logarithm by its tangent at $c = \tau$ sinh $\frac { 1 } { \tau }$ , namely log $T \le T / c - 1 + \log c$ , valid for every $T > 0 \mathrm { : }$

$$
A _ { X } \ \leq \ \frac { T } { \sinh \frac { 1 } { \tau } } - \tau + \tau \log \left( \tau \sinh \frac { 1 } { \tau } \right) - \tau N _ { X } .
$$

Using $1 + e ^ { - 1 / \tau } /$ sinh $\begin{array} { r } { \frac { 1 } { \tau } = \coth \frac { 1 } { \tau } } \end{array}$

$$
\begin{array} { r } { \frac { e ^ { \omega } \Delta _ { X } } { \sinh { \omega } } \leq \coth \frac { 1 } { \tau } - \tau + \tau \log \big ( \tau \sinh \frac { 1 } { \tau } \big ) - \tau N _ { X } . } \end{array}
$$

Finally $\begin{array} { r } { N _ { X } = R L _ { X } - \frac { 1 } { \tau } } \end{array}$ and τ log cosh $\scriptstyle { \frac { 1 } { \tau } } = \tau R - 2$ by Eq. 12 and Eq. 13, so τ $\cdot \ln ( \tau \sinh { \textstyle { \frac { 1 } { \tau } } } ) -$ $\tau N _ { X } = \tau \log ( \tau$ tanh $\begin{array} { r } { \frac { 1 } { \tau } \dot { ) } - 1 + \tau R ( 1 - \dot { L _ { X } } ) } \end{array}$ , giving the claim. □

## A.4.3 Connecting a high loss with low CKA

Theorem 2 (Implicit misalignment). Suppose $\sigma _ { X } > 0 , \tau \omega \leq 1$ , and Assumption 1 holds. Then

$$
\mathrm { C K A } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) \leq \frac { d + \ell _ { X } ^ { 2 } \varepsilon _ { X } } { 2 \sigma ^ { 2 } \sigma _ { X } } + \frac { \tau R a } { \sigma _ { X } } ( 1 - L _ { X } ) = : g ( L _ { X } )
$$

with $d = 2 \sigma ^ { 2 } a \left( \coth { \textstyle \frac { 1 } { \tau } } - \tau + \tau \log { \left( \tau \operatorname { t a n h } { \textstyle \frac { 1 } { \tau } } \right) } - 1 \right)$ , and the same statement holds with Y in place of X.

Proof. Chain Lemma 5, Lemma 6 and Lemma 7:

$$
\mathrm { C K A } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) \leq \frac { \beta _ { X } } { \sigma _ { X } } \leq \frac { \Delta _ { X } + \frac { 1 } { 2 } \omega \ell _ { X } ^ { 2 } \varepsilon _ { X } } { \sigma _ { X } } \leq \frac { a \left( \Lambda ( \tau ) + \tau R \left( 1 - L _ { X } \right) \right) + \frac { 1 } { 2 } \omega \ell _ { X } ^ { 2 } \varepsilon _ { X } } { \sigma _ { X } } .
$$

Reorganization gives the bound.

## A.5 The Loss Range

Lemma 8 (Range of the normalized loss). $L _ { X } \in [ 0 , 1 ]$ . Moreover:

(i) $L _ { X } = 0$ if and only $\begin{array} { r } { \vert f \left. x , x ^ { + } \right. = 1 } \end{array}$ and $\langle x , x ^ { \prime } \rangle = 0$ almost surely;

(ii) $L _ { X } ~ = ~ 1$ if and only if there is a unit vector e such that $x \in \{ e , - e \}$ almost surely with $\textstyle \operatorname* { P r } ( x = e ) = { \frac { 1 } { 2 } }$ , and $x ^ { + } = - x$ almost surely;

(iii) If additionally Assumption 1 is satisfied and $\ell _ { X } ^ { 2 } \varepsilon _ { X } < 4 ,$ , then

$$
L _ { X } \ \le \ 1 - \frac { 1 - \frac 1 4 \ell _ { X } ^ { 2 } \varepsilon _ { X } } { \tau R } \ < \ 1 ,
$$

with equality if and only if there is a unit vector e such that $x \in \{ e , - e \}$ almost surely with $\operatorname* { P r } ( x { \bar { = } } e ) { \bar { = } } { \bar { \frac { 1 } { 2 } } }$ , and the augmentation reverses $q ( m ) = \mathbb { E } [ \langle x , e \rangle \mid m ] ( q ( m ^ { + } ) = - q ( m ) )$ π-almost surely.

Proof. Lower bound. $\mathbb { E } [ \langle x , x ^ { + } \rangle ] \leq 1$ and the uniformity term is nonnegative (Eq. 15), so $N _ { X } \geq$ $- 1 / \bar { \tau } , \mathrm { i . e . } L _ { X } \geq 0$

Case (i). Equality forces $\langle x , x ^ { + } \rangle = 1 \mathrm { a . s }$ . and, by Lemma 1, $\| \bar { x } \| ^ { 2 } = 0$ and $\mathbb { E } [ V ] = 0 ;$ then $r ( x ) ^ { 2 } \leq \| { \bar { x } } \| ^ { 2 } = { \dot { 0 } }$ and $V ( x ) = 0 \ \mathrm { \dot { a } . s . , s o \ } \langle x , x ^ { \prime } \rangle = \mathrm { \dot { 0 } } \ \mathrm { a . s }$ . The converse is a direct substitution.

Upper bound. Since both marginals of π are $\mu ,$ the pair $( x , x ^ { + } )$ is a coupling of $\rho _ { X }$ with itself. Writing $z = x - { \bar { x } }$ and $z ^ { + } = x ^ { + } - \bar { x }$ , the cross terms vanish and $A _ { X } ^ { ' } = \Vert \bar { x } \Vert ^ { 2 } + \mathbb { E } \backslash \langle z , z ^ { + } \rangle$ ; as $\| { \boldsymbol x } \| = 1$ we have $\mathbb { E } \| z \| ^ { 2 } = \mathbb { E } \| z ^ { + } \| ^ { 2 } = 1 - \| \bar { x } \| ^ { 2 }$ , so Cauchy–Schwarz gives

$$
A _ { X } \ \geq \ \| { \bar { x } } \| ^ { 2 } - { \sqrt { \mathbb { E } \| z \| ^ { 2 } } } { \sqrt { \mathbb { E } \| z ^ { + } \| ^ { 2 } } } = \| { \bar { x } } \| ^ { 2 } - ( 1 - \| { \bar { x } } \| ^ { 2 } ) \ = \ 2 \| { \bar { x } } \| ^ { 2 } - 1 .\tag{24}
$$

By Jensen and Lemma 10(i) with $\alpha = 1 / \tau , c = \langle x , x ^ { \prime } \rangle$

$$
\begin{array} { r } { U _ { X } = \mathbb { E } _ { x } \Big [ \log \mathbb { E } _ { x ^ { \prime } } \big [ e ^ { \langle x , x ^ { \prime } \rangle / \tau } \big ] \Big ] \leq \log \mathbb { E } \big [ e ^ { \langle x , x ^ { \prime } \rangle / \tau } \big ] \leq \log \Big ( \cosh \frac { 1 } { \tau } + \| \bar { x } \| ^ { 2 } \sinh \frac { 1 } { \tau } \Big ) . } \end{array}\tag{25}
$$

Denote $t = \| \bar { x } \| ^ { 2 }$ . With Eq. 24 this gives $N _ { X } \leq f ( t )$ for $f ( t ) = ( 1 - 2 t ) / \tau + \log ( \cosh \frac { 1 } { \tau } + t \sinh \frac { 1 } { \tau } )$ and $\begin{array} { r } { f ^ { \prime } ( t ) \leq - \frac { 2 } { \tau } + } \end{array}$ tanh $\textstyle { \frac { 1 } { \tau } } < 0$ for $t \geq 0$ , so $\begin{array} { r } { N _ { X } \le f ( 0 ) = \frac { 1 } { \tau } + \log \cosh \frac { 1 } { \tau } = R - \frac { \mathrm { i } } { \tau } } \end{array}$ . This gives $L _ { X } \leq 1$

Case $( i i ) . ~ L _ { X } = 1$ , meaning $\begin{array} { r } { N _ { X } = R - \frac { 1 } { \tau } } \end{array}$ , forces equality throughout. That is, the chord bound in Eq. 25 is tight; Jensen in Eq. 25 is tight; Cauchy–Schwarz in Eq. 24 is tight; and $\| \bar { x } \| ^ { 2 } = 0$

Since $c \mapsto e ^ { c / \tau }$ is strictly convex, it meets its chord on $[ - 1 , 1 ]$ only at the endpoints, so tightness of the chord bound forces $\langle x , x ^ { \prime } \rangle \in \{ - 1 , 1 \} \mathrm { ~ a . e . , ~ i . e . ~ } x ^ { \prime } \in \{ x , - x \}$ . Hence, there is a single e in the support with $\rho _ { X } ( \{ e , - e \} ) = \stackrel { . } { 1 }$ . Writing $p = \operatorname* { P r } ( x = e )$ we get $\bar { \boldsymbol { x } } = ( 2 p - 1 ) \boldsymbol { e } , \mathrm { s o } \| \bar { \boldsymbol { x } } \| ^ { \bar { 2 } } = 0$ gives $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$ . Then $\mathbb { E } _ { x ^ { \prime } } [ e ^ { \langle x , x ^ { \prime } \rangle / \tau } ] = \cosh \frac { 1 } { \tau }$ for both values of x. With $\bar { x } = 0$ we have $z = x ,$ , and equality in Eq. 24 forces $x ^ { + } = - x \operatorname { a . s }$ . Conversely these conditions give $A _ { X } = - 1$ and $\begin{array} { r } { U _ { X } = \log \cosh \frac { 1 } { \tau } } \end{array}$ hence $\begin{array} { r } { N _ { X } = \frac { 1 } { \tau } + } \end{array}$ log cosh $\begin{array} { r } { \frac { 1 } { \tau } = R - \frac { 1 } { \tau } } \end{array}$ by direct substitution.

Case (iii). Let $c _ { m } : = \mathbb { E } [ x \mid m ]$ be the semantic centroid, $d _ { m } : = c _ { m } - \bar { x } , \gamma : = \mathbb { E } _ { m } \| d _ { m } \| ^ { 2 }$ and $\Gamma : = \mathbb { E } _ { \pi } \Vert d _ { m } - d _ { m ^ { + } } \Vert ^ { 2 }$ . Since u and $u ^ { + }$ are drawn independently, $A _ { X } = \mathbb { E } _ { \pi } \langle c _ { m } , c _ { m ^ { + } } \rangle$ , and since both marginals of π are $\mu ,$ decomposing $\mathbb { E } _ { \pi } \langle d _ { m } , d _ { m + } \rangle$ gives,

$$
A _ { X } = \| { \bar { x } } \| ^ { 2 } + \gamma - { \textstyle { \frac { 1 } { 2 } } } \Gamma .\tag{26}
$$

Now we bound Γ in two ways. Coupling the two centroids through the same nuisance, $c _ { m } - c _ { m ^ { + } } =$ $\mathbb { E } _ { u } [ e _ { X } ( g _ { X } ( m , u ) ) - e _ { X } ( g _ { X } ( m ^ { + } , \bar { u } ) ) ]$ , so Jensen and Assumption 1 give

$$
\Gamma = \mathbb { E } _ { \boldsymbol \pi } \Vert c _ { m } - c _ { m ^ { + } } \Vert ^ { 2 } \leq \mathbb { E } _ { \boldsymbol \pi } \mathbb { E } _ { \boldsymbol u } [ \Vert e _ { X } ( g _ { X } ( m , \boldsymbol { u } ) ) - e _ { X } ( g _ { X } ( m ^ { + } , \boldsymbol { u } ) ) \Vert ] ^ { 2 } \leq \ell _ { X } ^ { 2 } \varepsilon _ { X } .
$$

Also, $\| d _ { m } - d _ { m ^ { + } } \| ^ { 2 } \leq 2 \| d _ { m } \| ^ { 2 } + 2 \| d _ { m ^ { + } } \| ^ { 2 }$ gives $\Gamma \leq 4 \gamma$ . Feeding the smaller of the two into Eq. 26,

$$
\begin{array} { r l r } {  { A _ { X } - \| \bar { x } \| ^ { 2 } \geq \gamma - \frac { 1 } { 2 } \operatorname* { m i n } \{ \ell _ { X } ^ { 2 } \varepsilon _ { X } , 4 \gamma \} = \operatorname* { m a x } \{ \gamma - \frac { 1 } { 2 } \ell _ { X } ^ { 2 } \varepsilon _ { X } , - \gamma \} \geq - \frac { 1 } { 4 } \ell _ { X } ^ { 2 } \varepsilon _ { X } , } } \end{array}
$$

the last step because the two branches meet at $\begin{array} { r } { \gamma = \frac { 1 } { 4 } \ell _ { X } ^ { 2 } \varepsilon _ { X } } \end{array}$

Replacing Eq. 24 by this and repeating the argument of the upper bound, $N _ { X } \leq \psi ( \| \bar { x } \| ^ { 2 } )$ for $\begin{array} { r } { \psi ( \hat { t } ) = ( \frac { \mathbb { 1 } } { 4 } \ell _ { X } ^ { 2 } \hat { \varepsilon } _ { X } - t ) \bar { / } \tau + \log ( \cosh \frac { 1 } { \tau } + \bar { t } \sinh \frac { 1 } { \tau } ) } \end{array}$ , and $\begin{array} { r } { \psi ^ { \prime } ( t ) \leq - \frac { 1 } { \tau } + \operatorname { t a n h } \frac { 1 } { \tau } < 0 , } \end{array}$ , so $N _ { X } \leq \dot { \psi } ( 0 ) =$ $\begin{array} { r } { R - \frac { 1 } { \tau } - \overset { 1 } { \tau } \big ( 1 - \frac { 1 } { 4 } \ell _ { X } ^ { 2 } \varepsilon _ { X } \big ) } \end{array}$ , which is the claim; it is $< 1$ because $\ell _ { X } ^ { \dot { 2 } } \varepsilon _ { X } < 4$

Equality forces $\| \bar { x } \| ^ { 2 } = 0$ , both steps of Eq. 25 tight, and $\begin{array} { r } { \gamma = \frac { 1 } { 4 } \ell _ { X } ^ { 2 } \varepsilon _ { X } } \end{array}$ with $\Gamma = 4 \gamma$ . As in case (ii) the first two give $\rho _ { X } ( \{ e , - e \} ) = 1$ with $\textstyle \operatorname* { P r } ( x = e ) = { \frac { 1 } { 2 } }$ , and $d _ { m } = q ( m ) \epsilon$ for $q ( m ) = \mathbb { E } [ \left. x , e \right. \mid m ] ;$ then $\Gamma = 4 \gamma$ holds exactly when $q ( m ^ { + } ) = - q ( m ) \bar { \pi } { \cdot } { \bf { a . s } } .$ ., and $\gamma = \mathbb { E } _ { m } q ( m ) ^ { 2 }$ □

## A.6 The shape of the bounds

## A.6.1 The shape of the implicit alignment bound

This section analyses the shape of the implicit alignment bound from Thm. 1. Specifically, it proves that the bound $f { \bar { ( L _ { X } , L _ { Y } ) } }$ is nonincreasing in both $L _ { X }$ and $L _ { Y }$ , and converges to the desired limits (Prop. 1).

Lemma 9 (Shape of the alignment coefficient). For $L \in ( 0 , 1 ]$ . Then:

(i) c is nonincreasing on (0, 1], and strictly decreasing on the set where $0 < c ( L ) < 1$

(ii) c(L) = 1 for every $L \leq L _ { 1 }$ , where

$$
L _ { 1 } : = \frac { 2 a b + \chi ^ { 2 } D - \sqrt { \chi ^ { 2 } D \left( 4 a b + \chi ^ { 2 } D \right) } } { 2 b ^ { 2 } } \in ( 0 , a / b ) ,
$$

and c(L) < 1 for L > L<sub>1</sub>. In particular lim $_ { { \cal L } \to 0 ^ { + } } c ( { \cal L } ) = 1$

Proof. (i) Since $a , b , D , \chi > 0 , \mathrm { o n } \left( 0 , \infty \right)$ the map $L \mapsto [ a - b L ] _ { + }$ is nonnegative and nonincreasing and $L \mapsto \chi \sqrt { D L }$ is positive and increasing, so their ratio $g ( L )$ is nonincreasing. Composing with the map $t \mapsto$ min $\{ \dot { 1 } , t \}$ preserves this. On $\{ L < a / b \}$ we may drop the positive part and write $g ( L ) = ( a L ^ { - 1 / 2 } - b L ^ { 1 / 2 } ) / ( \chi \sqrt { D } )$ , whence $\begin{array} { r } { g ^ { \prime } ( L ) = - \frac 1 2 ( a L ^ { - 3 / 2 } + b L ^ { - 1 / 2 } ) / ( \chi \sqrt { D } ) < 0 ; } \end{array}$ since $c = g$ wherever $0 < c < 1$ , the strict decrease follows.

(ii) For $L > 0 , c ( L ) = 1$ iff $a - b L \geq \chi { \sqrt { D L } } ,$ , which forces $L < a / b$ and is then equivalent, after squaring, to $q ( L ) \geq 0$ with $q ( L ) : = b ^ { 2 } \dot { L ^ { 2 } } - ( 2 a b + \chi ^ { 2 } D ) L + a ^ { 2 }$ . The parabola q opens upwards, and

$$
q ( 0 ) = a ^ { 2 } > 0 , \qquad q ( a / b ) = a ^ { 2 } - \bigl ( 2 a b + \chi ^ { 2 } D \bigr ) \frac { a } { b } + a ^ { 2 } = - \chi ^ { 2 } D \frac { a } { b } < 0 ,
$$

so q has two positive roots $L _ { 1 } < a / b < L _ { 2 }$ and $\{ q \ge 0 \} \cap ( 0 , a / b ) = ( 0 , L _ { 1 } ]$ . The quadratic formula gives the stated $L _ { 1 } ,$ , using $( \bar { 2 } a b + \dot { \chi } ^ { 2 } D ) \bar { ^ { 2 } } - 4 a ^ { \hat { 2 } } \hat { b } ^ { 2 } = \dot { \chi } ^ { 2 } \hat { D } ( 4 a \dot { b } + \chi ^ { 2 } \dot { D } )$ . Hence, $c \equiv 1 \mathrm { o n } ( 0 , L _ { 1 } ]$ and since $L _ { 1 } > 0 , c \equiv 1$ is attainable, and gives the limit.

Proposition 1 (Monotonicity of the bound). Denote the right-hand side of Thm. $1 f ( L _ { X } , L _ { Y } )$ . With $L _ { 1 }$ as in Lemma 9:

(i) f is nonincreasing in $L _ { X }$ and in $L _ { Y }$ on $( 0 , 1 ] ^ { 2 }$ , and is strictly decreasing in $L _ { X }$ at every point with $c ( L _ { X } ) \in ( 0 , 1 )$ and $c ( L _ { Y } ) > 0$

(ii) $f \equiv 1 \mathrm { o n } ( 0 , L _ { 1 } ] ^ { 2 } .$

Proof. Let $g ( c ( L _ { X } ) , c ( L _ { Y } ) ) = f ( L _ { X } , L _ { Y } )$ . Write $\theta _ { X } = \operatorname { a r c c o s } { c _ { X } } , \theta _ { Y } = \operatorname { a r c c o s } { c _ { Y } } \in [ 0 , \pi / 2 ]$ so that

$$
g ( c _ { X } , c _ { Y } ) = c _ { X } c _ { Y } - \sqrt { ( 1 - c _ { X } ^ { 2 } ) ( 1 - c _ { Y } ^ { 2 } ) } = \cos \theta _ { X } \cos \theta _ { Y } - \sin \theta _ { X } \sin \theta _ { Y } = \cos ( \theta _ { X } + \theta _ { Y } ) .
$$

Now $c _ { X } \mapsto \theta _ { X }$ is decreasing on $[ 0 , 1 ] , \theta _ { X } + \theta _ { Y } \in [ 0 , \pi ]$ , and cos is decreasing on $[ 0 , \pi ] ;$ a decreasing function of a decreasing function is nondecreasing, so g is nondecreasing in $c _ { X }$ , and by symmetry in $c _ { Y }$ . For the strict monotonicity of g on $c _ { X } \in ( 0 , 1 )$ the chain rule gives

$$
{ \frac { \partial g } { \partial c _ { X } } } = { \frac { \sin ( \theta _ { X } + \theta _ { Y } ) } { \sin \theta _ { X } } } ~ \geq ~ 0 ,
$$

and sin $\iota ( \theta _ { X } + \theta _ { Y } ) = 0$ with $\theta _ { X } + \theta _ { Y } \in [ 0 , \pi ]$ forces $\theta _ { X } + \theta _ { Y } \in \{ 0 , \pi \}$ , i.e. $c _ { X } = c _ { Y } = 1$ or $c _ { X } = c _ { Y } = 0 ;$ at $c _ { X } = 0$ the one-sided derivative is s $\mathrm { i n } ( \pi / 2 + \theta _ { Y } ) { \stackrel { . } { = } } \mathrm { c o s } \theta _ { Y } = c _ { Y } > 0$ unless $c _ { Y } = 0$

(i) From Lemma 9(i): f is a nondecreasing function of $c ( L _ { X } )$ , which is nonincreasing in $L _ { X }$ Strictness likewise combines the strict decrease of c on $\{ 0 < \stackrel { \cdot } { c } < 1 \}$ with the strict increase of $g .$

(ii) By Lemma $9 ( \mathrm { i i } ) , c \equiv 1 \mathrm { o n } ( 0 , L _ { 1 } ] \mathrm { a n d } g ( 1 , 1 ) = 1$

## A.6.2 The shape of the implicit misalignment bound

This section analyses the shape of the implicit misalignment bound from Thm. 2. Specifically, it proves that the bound $g ( L )$ is strictly decreasing L and converges to the desired limits (Prop. 2).

Proposition 2 (Monotonicity of the misalignment bound). The bound g is strictly decreasing in $L _ { X }$ and is minimized at the largest possible loss as in Lemma 8(iii), where it equals

$$
g \left( 1 - \frac { 1 - \frac { 1 } { 4 } \ell _ { X } ^ { 2 } \varepsilon _ { X } } { \tau R } \right) = \frac { a \left( 1 + \Lambda ( \tau ) - \frac { 1 } { 4 } \ell _ { X } ^ { 2 } \varepsilon _ { X } \right) + \frac { 1 } { 2 } \omega \ell _ { X } ^ { 2 } \varepsilon _ { X } } { \sigma _ { X } } ,
$$

$\mathrm { i f } \ell _ { X } ^ { 2 } \varepsilon _ { X } < 4$ , and otherwise at $\begin{array} { r } { g ( 1 ) = \left( a \Lambda ( \tau ) + \frac { 1 } { 2 } \omega \ell _ { X } ^ { 2 } \varepsilon _ { X } \right) / \sigma _ { X } } \end{array}$

Proof. Monotonicity is immediate: g is affine in $L _ { X }$ with negative slope $- a \tau R / \sigma _ { X }$ □

## A.7 General inequalities

This section provides general mathematical inequalities used in the proofs above.

Lemma 10 (Chord bounds). Let $\alpha > 0$ and $c \in [ - 1 , 1 ]$ . Then

(i) $e ^ { \alpha c } \leq$ cosh α + c sinh α;

(ii) $| e ^ { \alpha c } - 1 | \leq ( e ^ { \alpha } - 1 ) | c | ;$

(iii) for every b with $0 < b \leq \alpha , \ e ^ { b c } \geq e ^ { - b } + { \frac { \sinh b } { \sinh \alpha } } ( e ^ { \alpha c } - e ^ { - \alpha } ) .$

Proof. (i) By convexity, $c \mapsto e ^ { \alpha c }$ lies below its chord on $[ - 1 , 1 ]$ , which is cosh α + c sinh α.

(ii) The difference quotient $\begin{array} { r } { c \mapsto ( e ^ { \alpha c } - 1 ) / c = \alpha \int _ { 0 } ^ { 1 } e ^ { \alpha c s } d s } \end{array}$ (value α at $c = 0 )$ is positive and nondecreasing, hence at most its value $e ^ { \alpha } - 1$ at $c = 1 ;$ ; multiply by |c|.

(iii) Put $t = e ^ { \alpha c } \in [ e ^ { - \alpha } , e ^ { \alpha } ]$ . Then $e ^ { b c } = t ^ { b / \alpha }$ is concave in t because $b / \alpha \leq 1$ , so it lies above the chord joining $( e ^ { - \alpha } , e ^ { - b } )$ and $( e ^ { \alpha } , e ^ { b } )$ , whose slope is ${ \frac { e ^ { b } - e ^ { - b } } { e ^ { \alpha } - e ^ { - \alpha } } } = { \frac { \sinh b } { \sinh \alpha } }$ . The right side being affine in t, and the left concave, the inequality survives taking expectations. □

Lemma 11 (Variance floor for the log-MGF). Let δ have mean zero and support in an interval of length at most 2, and let $\lambda \geq 1$ . Then log $\begin{array} { r } { \mathbb { E } [ e ^ { \lambda \delta } ] \geq \frac { \log { 2 } } { 4 } \lambda \mathrm { V a r } ( \delta ) } \end{array}$

Proof. Write $V = \operatorname { V a r } ( \delta ) = \mathbb { E } [ \delta ^ { 2 } ]$ and let [ℓ, u] contain the support with $0 \in [ \ell , u ]$ and $u - \ell \leq 2$ so $\delta \geq - 2$

Taylor’s formula with integral remainder gives $e ^ { t } = 1 + t + t ^ { 2 } \psi ( t )$ with $\begin{array} { r } { \psi ( t ) = \int _ { 0 } ^ { 1 } ( 1 - s ) e ^ { s t } d s = } \end{array}$ $( e ^ { t } - 1 - t ) / t ^ { 2 }$ nondecreasing. Since $\lambda \delta \geq - 2 \lambda$

$$
\begin{array} { r } { \mathbb { E } \big [ e ^ { \lambda \delta } \big ] \ \geq \ 1 + \lambda ^ { 2 } \psi ( - 2 \lambda ) V \ \geq \ 1 + \frac { \lambda } { 4 } V , } \end{array}\tag{27}
$$

because $\begin{array} { r } { \lambda ^ { 2 } \psi ( - 2 \lambda ) = \frac { e ^ { - 2 \lambda } - 1 + 2 \lambda } { 4 } \ge \frac { 2 \lambda - 1 } { 4 } \ge \frac { \lambda } { 4 } } \end{array}$ for $\lambda \geq 1$

Case $I \colon \lambda V / 4 \leq 1$ . By concavity log $( 1 + s ) \geq s \log 2$ on [0, 1], so Eq. 27 gives the claim.

Case $2 \colon \lambda V / 4 > 1$ . Let $p = \operatorname* { P r } ( \delta > 0 )$ , positive because $p = 0$ with $\mathbb { E } \delta = 0$ forces $V = 0$ , and let $h = \mathbb { E } [ \dot { \delta } \mathbf { 1 } _ { \delta > 0 } ] = \mathbb { E } [ - \delta \mathbf { 1 } _ { \delta < 0 } ]$ . Bounding $\delta \leq$ u on $\{ \delta > 0 \}$ and $- \delta \leq - \ell \mathrm { o n } \{ \delta < 0 \}$ gives $V \leq ( u - \bar { \ell } ) h \leq \bar { 2 } h$ , so $\lambda h \geq \lambda \bar { V } / 2 > 2 \geq \bar { p } .$ . Discarding the negative part and applying Jensen to the law of δ given $\{ \delta > 0 \}$ , gives

$$
\begin{array} { r } { \mathbb { E } [ e ^ { \lambda \delta } ] \ge p \mathbb { E } [ e ^ { \lambda \delta } \mid \delta > 0 ] \ge p e ^ { \lambda h / p } . } \end{array}
$$

The map $p \mapsto p e ^ { \lambda h / p }$ is nonincreasing on $( 0 , \lambda h ]$ , so at $p \leq 1$ it is at least $e ^ { \lambda h }$ , whence

$$
\begin{array} { r } { \log \mathbb { E } [ e ^ { \lambda \delta } ] \ge \lambda h \ge \frac { \lambda V } { 2 } \ge \frac { \log 2 } { 4 } \lambda V . } \end{array}
$$

## A.8 Constants Details

Thm. 1, as presented in the main text, incorporates several hyperparameter-dependent constants:

$$
a _ { \sigma } : = e ^ { - \frac { 1 } { \sigma ^ { 2 } } } \sinh \frac { 1 } { \sigma ^ { 2 } } ,
$$

which depends on the CKA-RBF scale σ. Recall $\begin{array} { r } { R _ { \tau } = { \frac { 2 } { \tau } } + \log \cosh { \frac { 1 } { \tau } } } \end{array}$

$$
b _ { \sigma , \tau } : = \big ( \frac { 1 } { \sigma ^ { 2 } } + e ^ { - \frac { 1 } { \sigma ^ { 2 } } } \sinh \frac { 1 } { \sigma ^ { 2 } } \big ) \tau R _ { \tau } , \quad D _ { \sigma , \tau } : = \Big ( 1 + \frac { 4 } { \log 2 } \Big ) \big ( 1 - e ^ { - \frac { 1 } { \sigma ^ { 2 } } } \big ) ^ { 2 } \tau R _ { \tau } ,
$$

which also rely on the InfoNCE temperature τ .

Similarly, Thm. 2 incorporates other hyperparameter-dependent constants:

$$
h _ { \sigma , \tau } = \tau R _ { \tau } a _ { \sigma } ,
$$

which depends on the InfoNCE temperature τ , and

$$
\begin{array} { r } { d _ { \sigma , \tau } = 2 \sigma ^ { 2 } a _ { \sigma } \left( \coth \frac { 1 } { \tau } - \tau + \tau \log \left( \tau \operatorname { t a n h } \frac { 1 } { \tau } \right) - 1 \right) , } \end{array}
$$

which also incorporates the CKA-RBF scale σ.

## B AudioCaps Results

In Sec. 4.2, the evaluation of unconstrained, pre-trained audio models focused primarily on the CLOTHO dataset. To further support our claim across audio datasets, we extend our analysis to AudioCaps.

Fig. 7a extends Fig. 6a from the main text to the AudioCaps dataset, while Fig. 7b extends Fig. 6b. Consistent with our primary findings, augmentation-based text models (Aug) consistently achieve higher cross-modal alignment on AudioCaps than next-token prediction (NTP) models of comparable or even substantially larger sizes. These results further support the role of unimodal augmentation objectives as a fundamental driver of implicit cross-modal alignment.

Fig. 7c extends Fig. 6c from the main text to the AudioCaps dataset. The strong correlation between lower unimodal text Aug-SSL InfoNCE loss and higher cross-modal alignment remains robust on AudioCaps. This finding further supports unimodal InfoNCE loss as a predictor of cross-modal alignment quality.

## C Statistical Significance

In Sec. 4.2, the analysis of unconstrained, pre-trained models reports Spearman $( \rho _ { s } )$ and Pearson (r) correlations between CMA and unimodal model properties over the available encoders. To demonstrate that these correlations are not an artifact of the sample size, this section establishes their statistical significance.

For each panel, we test the null hypothesis that the alignment scores are exchangeable with respect to the property under consideration. We obtain a two-sided p-value from 200,000 random permutations of the alignment scores against the property, where model size and training-corpus size are taken on a $\log _ { 1 0 }$ scale and the InfoNCE loss $\operatorname { a s } - { \mathcal { L } } .$ , matching the axes of the corresponding figures. Since the three figures comprise 10 panels tested jointly, we control the family-wise error rate using the Holm correction and consider a correlation significant when its corrected p-value is below 0.05. Values reported as $< 5 . 0 \times 1 0 ^ { - 6 }$ lie at the resolution limit of the permutation test.

Table 3 reports the results for the three pre-trained figures (4, 5, 6). Consistent with our primary findings, every correlation between the unimodal InfoNCE loss and CMA is significant, for both Spearman and Pearson, and across vision–text and audio–text alike, with all p-values below 2.6x10<sup>−4</sup> before correction. This confirms that the unimodal Aug-SSL loss is a reliable predictor of cross-modal alignment, and that this relation is not driven by a small number of individual models.

In contrast, the correlations with model and corpus size are inconsistent. They are consistently insignificant across datasets and modalities. Notably, in no panel do we find evidence that larger models or larger training corpora yield higher CMA. Together with the loss results, this further supports that unimodal augmentation objectives, rather than scale, act as a fundamental driver of implicit cross-modal alignment.

![](images/3145ef777299d753d38c7d51a4fa6896e814d0457ef3c723d7aaad4dd336460f.jpg)

![](images/c09fa42750d05d1f55cb21bd97f81e92c1ff5a5b51c6cf044bd169e592f4ea38.jpg)  
(a)  
ρ<sub>s</sub> = -0.10, r = -0.23

![](images/fbe17b7b0ad00d4c134e728cfde2bc4f87c4363026a1e937351af67a5b7eefca.jpg)  
ρ<sub>s</sub> = -0.24, r = -0.34  
(b)

![](images/007b63a1e6b1ff5ef98e11971221e8b4f78e2a3399f76cfc8a6eddb045836577.jpg)  
ρ = 0.78, r = 0.97  
(c)

Figure 7: Audio-text results on AudioCaps dataset. CMA between text models and the $\mathrm { \bf B Y O L - A }$ audio encoder as a function of (a) model size (number of parameters), (b) corpus size (amount of tokens the model was trained on), and (c) unimodal text InfoNCE loss. Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all models jointly, model/corpus sizes are uncorrelated with CMA, while the unimoda loss is highly correlated, indicated by both Spearman $( \rho _ { s } )$ and Pearson (r) correlations.  
Table 3: Significance of the correlations in the three pre-trained figures 4, 5, 6. Size and corpus size are correlated on $\log _ { 1 0 }$ and the loss $\mathbf { a s } - { \mathcal { L } } . \mathbf { \nabla } p$ is two-sided from 200,000 permutations of the alignment scores against the property; $\checkmark$ marks $p < 0 . 0 5$ after Holm correction over the 10 panels, ✗ that it does not. Only the correlation with the unimodal InfoNCE loss is statistically significant.
<table><tr><td rowspan="2">Panel</td><td rowspan="2">Property</td><td colspan="2">Spearman</td><td colspan="2">Pearson</td></tr><tr><td> $\rho _ { s }$ </td><td>p</td><td>r</td><td>p</td></tr><tr><td colspan="6">Fig. 4 — text encoders against DINOv2 ViT-B/14</td></tr><tr><td>(a) MS-COCO</td><td>Model size  $( \log _ { 1 0 } )$ </td><td>-0.030</td><td>0.889 X</td><td>-0.081</td><td>0.706 X</td></tr><tr><td>(b) MS-COCO</td><td>Training-corpus size  $( \log _ { 1 0 } )$ </td><td>-0.233</td><td>0.282 X</td><td>-0.360</td><td>0.092 x</td></tr><tr><td>(b) WIT</td><td>Training-corpus size  $( \log _ { 1 0 } )$ </td><td>-0.176</td><td>0.421 X</td><td>-0.173</td><td>0.431 X</td></tr><tr><td colspan="6">Fig. 5 — text encoders against DINOv2 ViT-B/14 in (a); image towers against BLOOM-7b1 in (b)</td></tr><tr><td>(a) MS-COCO</td><td>Text InfoNCE loss (−L)</td><td>0.880</td><td> $< 5 . 0 \times \bar { 1 0 } ^ { - 6 } ~ \surd$ </td><td>0.925</td><td> $< 5 . 0 \times 1 0 ^ { - 6 } ~ \surd$ </td></tr><tr><td>(a) WIT</td><td>Text InfoNCE loss (−L)</td><td>0.820</td><td> $1 . 5 \times 1 0 ^ { - 5 } \ : \surd$ </td><td>0.857</td><td> $< 5 . 0 \times 1 0 ^ { - 6 } ~ \surd$ </td></tr><tr><td>(b) MS-COCO</td><td>Vision InfoNCE loss (−L)</td><td>0.855</td><td> $2 . 6 \times 1 0 ^ { - 4 } \ : \surd$ </td><td>0.939</td><td> $< 5 . 0 \times 1 0 ^ { - 6 } ~ \surd$ </td></tr><tr><td>(b) WIT</td><td>Vision InfoNCE loss (−L)</td><td>0.890</td><td> $5 . 0 \times 1 0 ^ { - 5 } \surd$ </td><td>0.896</td><td> $< 5 . 0 \times 1 0 ^ { - 6 } ~ \surd$ </td></tr><tr><td colspan="6">Fig. 6 — text encoders against BYOL-A v2</td></tr><tr><td>(a) Clotho</td><td>Model size  $( \log _ { 1 0 } )$ </td><td>0.043</td><td>0.837 X</td><td>0.149</td><td>0.486 X</td></tr><tr><td>(b) Clotho</td><td>Training-corpus size  $( \log _ { 1 0 } )$ </td><td>-0.196</td><td>0.368 X</td><td>-0.128</td><td>0.561 X</td></tr><tr><td>(c) Clotho</td><td>Text InfoNCE loss (−L)</td><td>0.920</td><td> $< 5 . 0 \times 1 0 ^ { - 6 } ~ \surd$ </td><td>0.860</td><td> $< 5 . 0 \times 1 0 ^ { - 6 } ~ \surd$ </td></tr></table>

## D Additional Text Augmentations

In Sec. 4, our evaluation of the text InfoNCE utilized LLM-augmentations generated by Qwen2.5. However, this holds universally across different text augmentations. This section extends the analysis in the main text to different LLM augmentations and back-translation augmentations. For the LLM augmentations, we used two additional LLMs (Phi-3.5-Mini (Haider et al., 2024) and Gemma 4 12b (Team et al., 2026)) to generate five rephrased versions of each text. For the back-translation, we used bilingual models to transfer the English text to another language (e.g., French) and back to English. Additional details are provided in App. L.

Fig. 8 and 9 extend Fig. 3a from the main text to the two additional LLMs and back-translation augmentations, respectively. Consistent with our primary findings, the results support Thm. 1: minimizing the unimodal Aug-SSL objectives leads to increasing CMA. In contrast, every pairing involving a loss-maximized encoder is consistent with Thm. 2, showing that a high unimodal loss in either modality prevents cross-modal alignment, with CMA approaching chance level.

![](images/c6d63d2b4efb124e4fd593e603b545787efe8d9ce0e2b94021bc22b0c00fde62.jpg)  
Figure 8: Extended Empirical validation of Thm. 1 and 2 with different LLMs augmentations. Vision (ViT) and text (SBERT) models trained to minimize/maximize the Aug-SSL loss (Eq. 3) using Phi-3.5-Mini and Gemma 4 12B text augmentations. Minimizing both unimodal losses improves CMA (Thm. 1), whereas maximizing either loss prevents alignment, with CMA converging to the chance level (Thm. 2).

Fig. 10 and 11 extend Fig. 5a from the main text to two additional LLMs and back-translation augmentations, respectively. The strong correlation between lower unimodal text Aug-SSL InfoNCE loss and higher cross-modal alignment remains robust across LLMs used to generate the augmentations.

## E Different Vision Encoders

In Sec. 4.2, the evaluation of CMA to vision models focused on DINOv2, and to text models on BLOOM. This section shows that the same trends hold across different vision and text architectures and training paradigms. In particular, it extends the analysis to a broader suite of 14 vision models and 24 text models, spanning the model families discussed in Sec. 4.2.

Fig. 12 extends Fig. 4a to the 14 vision models, while Fig. 13 extends Fig. 1. Consistent with our primary findings, the results demonstrate that augmentation-based text models (Aug) consistently achieve higher cross-modal alignment across all tested vision encoders compared to next-token prediction (NTP) models of comparable or even substantially larger sizes.

Fig. 14 and 15 extend Fig. 4b to the 14 vision models, on COCO and WIT respectively. Consistent with our primary findings, the results demonstrate that augmentation-based text models (Aug) consistently achieve higher cross-modal alignment across all tested vision encoders compared to next-token prediction (NTP) models of comparable or even substantially smaller model and corpus sizes.

![](images/bc867f1881356c576e1b10aee71d71d5a50d6a90579a60b6fad736a12e90e230.jpg)  
Figure 9: Extended Empirical validation of Thm. 1 and 2 with back-translation augmentations. Vision (ViT) and text (SBERT) models trained to minimize/maximize the Aug-SSL loss (Eq. 3) using back-translation text augmentations. Minimizing both unimodal losses improves CMA (Thm. 1), whereas maximizing either loss prevents alignment, with CMA converging to the chance level (Thm. 2).

![](images/fee52cd54bb47e856297de97b164f2e12da5b0db4a1cfe51d7cce1c7fc72f4e5.jpg)  
Figure 10: CMA by InfoNCE across LLMs text augmentations on the COCO and WIT datasets. CMA by unimodal text InfoNCE loss across different LLMs (Phi-3.5-mini, Gemma 4 12B) augmentations. Notably, the unimodal Aug-SSL loss is highly correlated with CMA, clearly separating lowand high-alignment models, and having large Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/a22eaf92d906c0d1126f3fd42c9fd81d63c8b51e31cc2e51bc3e4e9c6a43ab66.jpg)  
Figure 11: CMA by InfoNCE across back-translation text augmentations on the COCO and WIT datasets. CMA by unimodal text InfoNCE loss using back-translation augmentations. Notably, the unimodal Aug-SSL loss is highly correlated with CMA, clearly separating low- and high-alignment models, and having large Spearman (ρ<sub>s</sub>) and Pearson (r) correlations.

Fig. 16 and 17 extend Fig. 5a to the 14 vision models, on COCO and on WIT, respectively. The strong correlation between lower unimodal text Aug-SSL InfoNCE loss and higher cross-modal alignment remains robust regardless of the specific vision architecture or its scale.

Similarly, Fig. 18 and 19 extend Fig. 5b to the 24 text models, on COCO and on WIT, respectively. The strong correlation between lower unimodal vision Aug-SSL InfoNCE loss and higher cross-modal alignment remains robust regardless of the specific text architecture or its scale.

## F Different Audio Encoders

In Sec. 4.2, the evaluation of audio models focused on BYOL-A-v2. This section extends our analysis to a broader suite of 15 audio models, spanning the six model families: wav2vec2 (Baevski et al., 2020), HuBERT (Hsu et al., 2021), WavLM (Chen et al., 2022), data2vec (Baevski et al., 2022), AudioMAE (Huang et al., 2022), BYOL-A (Niizumi et al., 2021).

Fig. 20 and 21 extend the model size analysis (Fig. 6a) to these additional audio models, on Clotho and AudioCaps, respectively. Consistent with our primary findings, the results demonstrate that augmentation-based text models (Aug) consistently achieve higher cross-modal alignment across all tested audio encoders compared to next-token prediction (NTP) models of comparable or even substantially larger sizes.

Fig. 22 and 23 extend Fig. 6b to these audio models, on Clotho and AudioCaps respectively. Consistent with our primary findings, the results demonstrate that augmentation-based text models (Aug) consistently achieve higher cross-modal alignment across all tested audio encoders compared to next-token prediction (NTP) models of comparable or even substantially smaller model and corpus sizes.

Fig. 24 and 25 extend our analysis of the unimodal loss (Fig. 6c) to these audio architectures, on Clotho and AudioCaps, respectively. The strong correlations between lower unimodal text Aug-SSL InfoNCE loss and higher cross-modal alignment remain remarkably robust across every evaluated audio encoder family and scale.

![](images/ec3e7e6397bdea8ec8d8f548154fbf27161204eb7d2301a206bf942e81d977b3.jpg)  
Figure 12: CMA across vision architectures on the COCO dataset. CMA by model size (number of parameters) across different vision encoder families (ImageNet21K, MAE, DINOv2, and CLIP) at various capacity scales. Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all models jointly, model/corpus sizes are uncorrelated with CMA, indicated by both weak Spearman (ρ<sub>s</sub>) and Pearson (r) correlations.

![](images/ac85604101ef0ae213d183a23ae591205597fa7da96cc13b91f31054676a2e0b.jpg)  
Figure 13: CMA across vision architectures on the WIT dataset. CMA by model size (number of parameters) across different vision encoder families (ImageNet21K, MAE, DINOv2, and CLIP) at various capacity scales. Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all models jointly, model/corpus sizes are uncorrelated with CMA, indicated by both weak Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/ddf09f0f09214efab138f11e6a5906dd42098bf6cbb77df7fb4d8ca2a5d7466a.jpg)  
Figure 14: CMA by corpus size across vision architectures on the COCO dataset. CMA by corpus size (tokens trained on) across different vision encoder families (ImageNet21K, MAE, DINOv2, and CLIP) at various capacity scales. Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all models jointly, model/corpus sizes are uncorrelated with CMA, indicated by both weak Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/22be40db52ae21224dc28a0133ccefd10ace7375c60a06945eb648d26834b74d.jpg)  
Figure 15: CMA by corpus size across vision architectures on the WIT dataset. CMA by corpus size (tokens trained on) across different vision encoder families (ImageNet21K, MAE, DINOv2, and CLIP) at various capacity scales. Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all models jointly, model/corpus sizes are uncorrelated with CMA, indicated by both weak Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/6694243dcc9e51b024b461cc2abb0f33bd2f1761e0a08304fd94997a46b818f5.jpg)  
Figure 16: CMA by InfoNCE across vision architectures on the COCO dataset. CMA by unimodal text InfoNCE loss across different vision encoder families (ImageNet21K, MAE, DINOv2, and CLIP) at various capacity scales. Notably, the unimodal Aug-SSL loss is highly correlated with CMA, clearly separating low- and high-alignment models, and having large Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/3c6e819c728a238b1a48affcd041b08f01f186aa39129c131dd489c7233b541a.jpg)  
Figure 17: CMA by InfoNCE across vision architectures on the WIT dataset. CMA by unimodal text InfoNCE loss across different vision encoder families (ImageNet21K, MAE, DINOv2, and CLIP) at various capacity scales. Notably, the unimodal Aug-SSL loss is highly correlated with CMA, clearly separating low- and high-alignment models, and having large Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/b4c400280d790f72303c15c2816a4443e305eff83922ffb53a77b503d945ba7f.jpg)  
Figure 18: CMA by Vision InfoNCE across text architectures on the COCO dataset. CMA by unimodal vision InfoNCE loss across different text encoder families (MiniLM, RoBERTa, Sentence-T5, BLOOM, OpenLLaMA, LLaMA, Pythia, and GPT-2) at various capacity scales. Notably, the unimodal Aug-SSL loss is highly correlated with CMA, clearly separating low- and high-alignment models, and having large Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/791a9b1d503a631d0ab88500c9b26b9ab9afb0ba04c0d2246d8ade3a047aeeb8.jpg)  
Figure 19: CMA by Vision InfoNCE across text architectures on the WIT dataset. CMA by unimodal vision InfoNCE loss across different text encoder families (MiniLM, RoBERTa, Sentence-T5, BLOOM, OpenLLaMA, LLaMA, Pythia, and GPT-2) at various capacity scales. Notably, the unimodal Aug-SSL loss is highly correlated with CMA, clearly separating low- and high-alignment models, and having large Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/57726425d2d26d20b50da3d3ec240ccd771e2ddd19099d44f5a13c9658beba77.jpg)  
Figure 20: CMA across audio architectures on the CLOTHO dataset. CMA by model size (number of parameters) across different audio encoder families (wav2vec2, HuBERT, WavLM, data2vec, AudioMAE, and BYOL-A) at various capacity scales. Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all models jointly, model/corpus sizes are uncorrelated with CMA, indicated by both weak Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/639b9e351af944b75456dac7ed2ac10bb4c9fbf90c11ce858c12a8b5c1dfc205.jpg)  
Figure 21: CMA across audio architectures on the AudioCaps dataset. CMA by model size (number of parameters) across different audio encoder families (wav2vec2, HuBERT, WavLM, data2vec, AudioMAE, and BYOL-A) at various capacity scales. Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all models jointly, model/corpus sizes are uncorrelated with CMA, indicated by both weak Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/256a78009765d88a68beac0a6637314646ffb900fc8e0ad2bfd0872ce892f342.jpg)  
Figure 22: CMA by corpus size across audio architectures on the CLOTHO dataset. CMA by corpus size (tokens trained on) across different audio encoder families (wav2vec2, HuBERT, WavLM, data2vec, AudioMAE, and BYOL-A) at various capacity scales. Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all models jointly, model/corpus sizes are uncorrelated with CMA, indicated by both weak Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/e6badb5ada842e3f7a8871888ea84828685e98783dccb86812c6ca7c83aaf778.jpg)  
Figure 23: CMA by corpus size across audio architectures on the AudioCaps dataset. CMA by corpus size (tokens trained on) across different audio encoder families (wav2vec2, HuBERT, WavLM, data2vec, AudioMAE, and BYOL-A) at various capacity scales. Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all models jointly, model/corpus sizes are uncorrelated with CMA, as indicated by both weak Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/ac090994e195f9b35036c9b4a1804de09662f1ac0a5800b936ba3c04c30ec30f.jpg)  
Figure 24: CMA by InfoNCE across audio architectures on the CLOTHO dataset. CMA by unimodal text InfoNCE loss across different audio encoder families (wav2vec2, HuBERT, WavLM, data2vec, AudioMAE, and BYOL-A) at various capacity scales. Notably, the unimodal Aug-SSL loss is highly correlated with CMA, clearly separating low- and high-alignment models, and having large Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/7a384d58a2913e5eafe79cb65d93115d6e62bf4fa378a90451dd9480b07d0a8a.jpg)  
Figure 25: CMA by InfoNCE across audio architectures on the AudioCaps dataset. CMA by unimodal text InfoNCE loss across different audio encoder families (wav2vec2, HuBERT, WavLM, data2vec, AudioMAE, and BYOL-A) at various capacity scales. Notably, the unimodal Aug-SSL loss is highly correlated with CMA, clearly separating low- and high-alignment models, and having large Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/579f5ecd2d5be4a271957a8d61e91264cc4a8866441728e06b0980b368177d87.jpg)  
Figure 26: Mutual k-NN across k values on the COCO and WIT datasets. Mutual k-NN by model size (number of parameters) across different k values for the metric. Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all k values jointly, model/corpus sizes are uncorrelated with CMA, indicated by both weak Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

## G Additional Cross-Modal Alignment Metrics

In Sec. 4.2, the evaluation focused on Mutual k-NN with k=50 for the CMA metric. To demonstrate that the connection between unimodal Aug-SSL objectives and CMA holds universally across different local CMA metrics on different k values, this section extends our analysis to a broader suite of Mutual k-NN and CKNNA on 4 different k values.

CKNNA. CKNNA is a popular local restriction of CKA: it evaluates the same kernel alignment, but only on the pairs that are mutual k-nearest neighbors in both embedding spaces. Given paired observations $\bar { \mathcal { S } } = ( x _ { 1 } , y _ { 1 } ) , \dotsc , ( x _ { n } , y _ { n } ) \in \mathcal { E } _ { X } \times \bar { \mathcal { E } } _ { Y }$ , let $K _ { i j } = k _ { X } ( e _ { X } ( \hat { x _ { i } } ) , \hat { e _ { X } } ( x _ { j } ) )$ and $\bar { L } _ { i j } =$ $k _ { Y } \left( e _ { Y } ( y _ { i } ) , e _ { Y } ( y _ { j } ) \right)$ be the two kernel matrices (usually linear), and define the mutual-neighbor mask

$$
M _ { i j } = \mathbf { 1 } \big [ x _ { j } \in N _ { X } ^ { k } ( x _ { i } ) \big ] \cdot \mathbf { 1 } \big [ y _ { j } \in N _ { Y } ^ { k } ( y _ { i } ) \big ] , \qquad M _ { i i } = 0 .\tag{28}
$$

Replacing each kernel by its masked counterpart in Eq. 7 gives

$$
\mathrm { C K N N A } _ { k } ( \mathcal { E } _ { X } , \mathcal { E } _ { Y } ) \ = \ \frac { \mathrm { H S I C } \big ( M \odot K , M \odot L \big ) } { \sqrt { \mathrm { H S I C } \big ( M \odot K , M \odot K \big ) \mathrm { H S I C } \big ( M \odot L , M \odot L \big ) } } \ \in \ [ 0 , 1 ] .\tag{29}
$$

Fig. 26 and 27 extend Fig. 4a to Mutual k-NN and CKNNA (respectively) with different k values on COCO and WIT. Consistent with our primary findings, the results demonstrate that augmentationbased text models (Aug) consistently achieve higher cross-modal alignment across the tested CMA metrics and across k values compared to next-token prediction (NTP) models of comparable or even substantially larger sizes.

Fig. 28 and 29 extend Fig. 5a to CKNNA with different k values, on COCO and on WIT. The strong correlation between lower unimodal text Aug-SSL InfoNCE loss and higher cross-modal alignment remains remarkably robust regardless of the specific local CMA metric or the k value.

## H Larger Test Set Sizes

In Sec. 4.2, the evaluation used a test set of size 1024. This section extends our analysis to additional larger test set sizes.

![](images/1479aa0211a1ef4314f412708af2c1382544cf18aa53ce39e1b3aa27ad4dfd13.jpg)  
Figure 27: CKNNA across k values on the COCO and WIT datasets. CKNNA by model size (number of parameters) across different k values for the metric. Notably, Aug models consistently achieve higher CMA, even at substantially smaller model and corpus sizes. Also, considering all k values jointly, model/corpus sizes are uncorrelated with CMA, indicated by both weak Spearman (ρ<sub>s</sub>) and Pearson (r) correlations.

![](images/476ee6229f943041ce85808827e60002364a14b481aa3968390d53d2c250cf15.jpg)  
Figure 28: Mutual k-NN by InfoNCE across k values on the COCO and WIT datasets. Mutual k-NN by unimodal text InfoNCE loss across different k values for the metric. Notably, the unimodal Aug-SSL loss is highly correlated with CMA, clearly separating low- and high-alignment models, and having large Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

![](images/02dbf0c00aa87d6694b99e6c257900da13c2a9971c0b8f0d2896082abd7a879c.jpg)

Figure 29: CKNNA by InfoNCE across k values on the COCO and WIT datasets. CKNNA by unimodal text InfoNCE loss across different k values for the metric. Notably, the unimodal Aug-SSL loss is highly correlated with CMA, clearly separating low- and high-alignment models, and having large Spearman $( \rho _ { s } )$ and Pearson (r) correlations.  
![](images/586174d28bbe9a0f3f5208dce23e2bacc1c3f29c0479ef71479ff4234cee320e.jpg)  
Figure 30: CMA by test set size across capacity scales on the COCO dataset. CMA by test size across different DINOv2 capacity scales. Notably, all text models keep about the same relation across different test set size.

Fig. 30 extends Fig. 4 to different test set sizes on COCO. Consistent with our primary findings, the results demonstrate that the text models keep about the same relation between them across different test set sizes and model capacities.

## I Different Temperatures

In Sec. 4.2, the evaluation used InfoNCE loss with temperature τ = 0.1. Fig. 31 extends Fig. 5a to different temperatures on COCO. The strong correlation between lower unimodal text Aug-SSL InfoNCE loss and higher cross-modal alignment remains robust regardless of the specific temperature of the InfoNCE loss.

## J CKA-RBF and mkNN

Fig. 32 extends Fig. 2b from Sec. 3, evaluating the correlation between Mutual k-NN and CKA-RBF, to different k values. The strong correlation between Mutual k-NN and CKA-RBF remains robust regardless of the specific k value, while adapting the scale σ accordingly. This confirms that CKA-RBF can be viewed as a continuous relaxation of the Mutual k-NN.

![](images/c4aaf2f00002700ca2420f94726f80e185216a62762a0ef99228970f49089be1.jpg)  
Figure 31: CMA by InfoNCE across different temperature on the COCO dataset. CMA by unimodal text InfoNCE loss across different temperature for the InfoNCE loss. Notably, the unimodal Aug-SSL loss is highly correlated with CMA, very consistent across different temperature, clearly separating low- and high-alignment models, and having large Spearman $( \rho _ { s } )$ and Pearson (r) correlations.  
Figure 32: Mutual k-NN by CKA-RBF across k values on the COCO dataset. Mutual k-NN by CKA-RBF across different k values for Mutual k-NN. Notably, Mutual k-NN is highly correlated with CKA-RBF, very consistent across different k, and having large Spearman $( \rho _ { s } )$ and Pearson (r) correlations.

## K Visualizing the Bounds

This section empirically visualizes the bounds from Thm. 1 and Thm. 2 across different parameter options.

Fig. 33 visualizes the bounds in $\tau = 0 . 3$ and $\sigma ^ { 2 } = 1$ . As depicted in $\mathrm { F i g }$ . 33a, the lower bound is effective for sufficiently low losses, and in Fig. 33b, the upper bound is effective for sufficiently high losses. The latter confirms that low unimodal augmentation-based losses guarantee alignment, whereas only one high loss prevents alignment.

Fig. 34 and 35 extends 33 to different values of τ and $\sigma ^ { 2 } .$ , on Thm. 1 and Thm. 2, respectively. This show robustness of the bounds across loss and CMA parameters.

![](images/d816b3603f0cca740d3aaa6281ced36eb1c097278251be02a3caf9eddddb307a.jpg)  
(a)

![](images/65db4d63431a1026b5994a7c3318da538bc94d51d636b03a10d997f22a826594.jpg)  
(b)  
excluded by the bound of the same colou  
Figure 33: Visualization of the (a) lower and (b) upper bounds from Thm. 1 and 2.

## L Implementation Details

This section specifies the experimental configurations used in Sec. 4.

## L.1 Models

Pre-trained encoders. Tables 4, 5, and 6 summarize the 24 text, 14 vision, and 15 audio frozen encoders used in our experiments. L denotes the number of hidden layers, excluding the embedding out put, and d denotes the corresponding representation width. The model paraphrase-MiniLM-L3-v2 has no published corpus-size estimate. We therefore exclude it from analyses involving corpus size.

Corpus sizes. We obtained the corpus-sizes in Tables 4 and 5 from published sources and classified them as exact when the source reports the corresponding count, or approximate when the value is derived from a training schedule or byte count. For NTP encoders, corpus sizes reported in bytes are converted using 4.85 bytes per BPE token. For MiniLM students, the reported 3.3B-word corpus is converted using 1.30 WordPiece tokens per word. For augmentation-based encoders, corpus size includes the backbone pre-training corpus and the distinct examples encountered during the contrastive training stage. We estimate the latter using the average: 2.078 texts per training example and 53.9 tokens per text. This quantifies the number of distinct tokens encountered. Sentence-T5 is the only family for which the number of contrastive training steps is unpublished; we therefore use GTR’s 800k steps for its corresponding $2 \times 1 0 ^ { 9 }$ -pair training stage. Vision corpus size is measured as the number of unique training images.

Embedding extraction. Table 7 summarizes the extraction settings.

Controlled architectures and training. Table 8 specifies the controlled architectures and training schedules. Both training objectives operate on the 128-dimensional projection-head representations, and alignment is evaluated in the same space. The vision models used a common preprocessing pipeline consisting of resizing the short side to 256 pixels, centre cropping to 224×224, and ImageNet normalization. The curves in Fig. 3 and the corresponding appendix figures report the mean ± one sample standard deviation over three training seeds. Alignment is evaluated on a fixed held-out draw of 4096 of the COCO 2017 validation set pairs.

Loss maximization and chance level. The loss-maximizing condition differs from the correspond ing minimizing condition only in the sign of the backward pass. That is, initialization, minibatch stream, optimization schedule, and gradient clipping remain unchanged. For mutual k-NN overlap, the chance level is the expected overlap of two independent k-neighborhoods among n instances, which is $k / ( n - 1 )$ ). For $\bar { k } = 5 0$ and $n = 4 0 9 6$ , this gives $5 0 / 4 0 9 5 = 0 . 0 1 2 2$

![](images/cb8eb1c2bbbbf580a23ed2ba4d1e4c777f31afb98ffd0a032d046cf60ec2aabb.jpg)  
Figure 34: Extended visualization of the lower bound from Thm. 1, across different choices of τ and $\sigma ^ { 2 }$

![](images/37e9c6bf227dff1b36a6632edae24deb773d1078c90123c0feb146eb1fe599a4.jpg)  
Figure 35: Extended visualization of the upper bound from Thm. 2, across different choices of $\tau$ and $\sigma ^ { 2 }$

Table 4: The 24 frozen text encoders. Aug denotes augmentation-based sentence encoders and NTP denotes next-token-prediction language models. Corpus size is measured as the number of unique tokens across all training stages, as described below.
<table><tr><td>Checkpoint</td><td>Family</td><td>Type</td><td>Params</td><td> $L / d$ </td><td>Corpus (tokens)</td></tr><tr><td>sentence-transformers/paraphrase-MiniLM-L3-v2</td><td>MiniLM</td><td>Aug</td><td>17.4M</td><td>3 / 384</td><td></td></tr><tr><td>sentence-transformers  $/ \mathsf { \bar { a } } 1 1 - \mathsf { \bar { M } i n i L M - L 6 } - \mathsf { v } 2$ </td><td>MiniLM</td><td>Aug</td><td>22.0M</td><td>6/384</td><td> $1 . 0 3 \times 1 0 ^ { 1 0 }$ </td></tr><tr><td>sentence-transformers/all-MiniLM-L12-v2</td><td>MiniLM</td><td>Aug</td><td>33.0M</td><td>12 / 384</td><td> $1 . 0 3 \times 1 0 ^ { 1 0 }$ </td></tr><tr><td>sentence-transformers/all-distilroberta-v1</td><td>RoBERTa</td><td>Aug</td><td>82.0M</td><td>6/768</td><td> $2 . 5 9 \times 1 0 ^ { 1 0 }$ </td></tr><tr><td>sentence-transformers/all-roberta-large-v1</td><td>RoBERTa</td><td>Aug</td><td>355.4M</td><td>24 / 1024</td><td> $3 . 8 7 \times 1 0 ^ { 1 0 }$ </td></tr><tr><td>sentence-transformers/sentence-t5-base</td><td>Sentence-T5</td><td>Aug</td><td>110.2M</td><td>12 /768</td><td> $1 . 5 5 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>sentence-transformers/sentence-t5-large</td><td>Sentence-T5</td><td>Aug</td><td>335.7M</td><td>24 / 1024</td><td> $1 . 5 5 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>sentence-transformers/sentence-t5-xl</td><td>Sentence-T5</td><td>Aug</td><td>1.24 B</td><td>24 / 1024</td><td> $1 . 5 5 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>sentence-transformers/sentence-t5-xxl</td><td>Sentence-T5</td><td>Aug</td><td>4.87 B</td><td>24 / 1024</td><td> $1 . 5 5 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>bigscience/bloom-560m</td><td>BLOOM</td><td>NTP</td><td>559.2M</td><td>24 / 1024</td><td> $3 . 5 0 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>bigscience/bloom-1b7</td><td>BLOOM</td><td>NTP</td><td>1.72 B</td><td>24 /2048</td><td> $3 . 5 0 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>bigscience/bloom-3b</td><td>BLOOM</td><td>NTP</td><td>3.00 B</td><td>30 / 2560</td><td> $3 . 5 0 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>bigscience/bloom-7b1</td><td>BLOOM</td><td>NTP</td><td>7.07 B</td><td>30 / 4096</td><td> $3 . 5 0 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>openlm-research/open_llama_3b</td><td>OpenLLaMA</td><td>NTP</td><td>3.32 B</td><td>26 / 3200</td><td> $1 . 0 0 \times 1 0 ^ { 1 2 }$ </td></tr><tr><td>openlm-research/open_1lama_7b</td><td>OpenLLaMA</td><td>NTP</td><td>6.61 B</td><td>32 / 4096</td><td> $1 . 0 0 \times 1 0 ^ { 1 2 }$ </td></tr><tr><td>openlm-research/open_1lama_13b</td><td>OpenLLaMA</td><td>NTP</td><td>12.85 B</td><td>40 / 5120</td><td> $1 . 0 0 \times 1 0 ^ { 1 2 }$ </td></tr><tr><td>LLaMA-1 7B</td><td>LLaMA</td><td>NTP</td><td>6.61 B</td><td>32 / 4096</td><td> $1 . 0 0 \times 1 0 ^ { 1 2 }$ </td></tr><tr><td>LLaMA-1 13B</td><td>LLaMA</td><td>NTP</td><td>12.85 B</td><td>40 / 5120</td><td> $1 . 0 0 \times 1 0 ^ { 1 2 }$ </td></tr><tr><td>EleutherAI/pythia-14m</td><td>Pythia</td><td>NTP</td><td>7.6M</td><td>6/128</td><td> $3 . 0 0 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>EleutherAI/pythia-70m</td><td>Pythia</td><td>NTP</td><td>44.7M</td><td>6/512</td><td> $3 . 0 0 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>EleutherAI/pythia-160m</td><td>Pythia</td><td>NTP</td><td>123.7M</td><td>12/ 768</td><td> $3 . 0 0 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>EleutherAI/pythia-410m</td><td>Pythia</td><td>NTP</td><td>353.8M</td><td>24 / 1024</td><td> $3 . 0 0 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>openai-community/gpt2</td><td>GPT-2</td><td>NTP</td><td>124.4M</td><td>12/ 768</td><td> $8 . 2 5 \times 1 0 ^ { 9 }$ </td></tr><tr><td>openai-community/gpt2-medium</td><td>GPT-2</td><td>NTP</td><td>354.8M</td><td>24 / 1024</td><td> $8 . 2 5 \times 1 0 ^ { 9 }$ </td></tr></table>

Table 5: The 14 frozen vision encoders. All models are loaded through timm. Corpus size denotes the number of training images. Fig. 4 and Fig. 5a use vit\_base\_patch14\_dinov2; the full 14- model comparison is reported in App. E.
<table><tr><td>timm checkpoint</td><td>Family</td><td>Params</td><td> $L / d$ </td><td>Corpus (images)</td></tr><tr><td>vit_tiny_patch16_224.augreg_in21k</td><td>ImageNet21K</td><td>9.74M</td><td>12 / 192</td><td> $\mathrm { I m a g e N e t { - } 2 1 k , 1 . 4 2 \times 1 0 ^ { 7 } }$ </td></tr><tr><td>vit_small_patch16_224.augreg_in21k</td><td>ImageNet21K</td><td>30.08M</td><td>12 / 384</td><td> $\mathrm { I m a g e N e t { - } } 2 1 \mathrm { k } , 1 . 4 2 \times 1 0 ^ { 7 }$ </td></tr><tr><td>vit_base_patch16_224.augreg_in21k</td><td>ImageNet21K</td><td>102.60 M</td><td>12 / 768</td><td>ImageNet-21k,  $1 . 4 2 \times 1 0 ^ { 7 }$ </td></tr><tr><td>vit_large_patch16_224.augreg_in21k</td><td>ImageNet21K</td><td>325.69 M</td><td>24 / 1024</td><td>ImageNet-21k, 1.42 × 107</td></tr><tr><td>vit_base_patch16_224.mae</td><td>MAE</td><td>85.80 M</td><td>12 / 768</td><td>ImageNet-1k, 1.28 × 106</td></tr><tr><td>vit_large_patch16_224.mae</td><td>MAE</td><td>303.30M</td><td>24 / 1024</td><td> $\mathrm { I m a g e N e t - 1 k , 1 . 2 8 \times 1 0 ^ { 6 } }$ </td></tr><tr><td>vit_huge_patch14_224.mae</td><td>MAE</td><td>630.76M</td><td>32 / 1280</td><td>ImageNet-1k, 1.28 × 106</td></tr><tr><td>vit_small_patch14_dinov2.lvd142m</td><td>DINOv2</td><td>22.06M</td><td>12 / 384</td><td> $\mathrm { L V } \bar { \mathrm { D } } { - } 1 4 2 \mathrm { M } , 1 . 4 2 \times 1 0 ^ { 8 }$ </td></tr><tr><td>vit_base_patch14_dinov2.lvd142m</td><td>DINOv2</td><td>86.58M</td><td>12 /768</td><td> $\mathrm { L V D - 1 4 2 M , 1 . 4 2 \times 1 0 ^ { 8 } }$ </td></tr><tr><td>vit_large_patch14_dinov2.lvd142m</td><td>DINOv2</td><td>304.37M</td><td>24 / 1024</td><td> $\mathrm { L V D - 1 4 2 M , 1 . 4 2 \times 1 0 ^ { 8 } }$ </td></tr><tr><td>vit_giant_patch14_dinov2.lvd142m</td><td>DINOv2</td><td>1.14B</td><td>40 / 1536</td><td> $\mathrm { L V D - 1 4 2 M , 1 . 4 2 \times 1 0 ^ { 8 } }$ </td></tr><tr><td>vit_base_patch16_clip_224.laion2b</td><td>CLIP</td><td>86.19M</td><td>12 / 768</td><td> $\mathrm { L A I O N  – 2 B – e n , 2 . 3 2 \times 1 0 ^ { 9 } }$ </td></tr><tr><td>vit_large_patch14_clip_224.laion2b</td><td>CLIP</td><td>303.97M</td><td>24 / 1024</td><td> $\mathrm { L A I O N  – 2 B – e n , 2 . 3 2 \times 1 0 ^ { 9 } }$ </td></tr><tr><td> $\mathrm { v i t \_ h u g e \mathrm { _ { - } p a t c h 1 4 \_ c l i p \mathrm { _ { - } 2 2 4 \mathrm { . 1 a i o n 2 b } } } }$ </td><td>CLIP</td><td>632.08M</td><td>32 / 1280</td><td> $\mathrm { L A I O N  – 2 B – e n , 2 . 3 2 \times 1 0 ^ { 9 } }$ </td></tr></table>

Multi-positive InfoNCE objective. Equation 3 contains one positive per anchor, whereas our controlled experiments use five views per item (Tian et al., 2023). Let a batch contain B items with V views per item, let $z _ { 1 } , \dotsc , z _ { B V }$ denote the unit-normalized projection-head outputs, and let $P ( i )$ denote the $V - 1$ other views of the same item as anchor i. All views belonging to other items are treated as negatives. The resulting objective is

$$
\ell ^ { ( V ) } = - \frac { 1 } { B V } \sum _ { i = 1 } ^ { B V } \frac { 1 } { V - 1 } \sum _ { j \in P ( i ) } \log \frac { \exp \left( \langle z _ { i } , z _ { j } \rangle / \tau \right) } { \sum _ { l \ne i } \exp \left( \langle z _ { i } , z _ { l } \rangle / \tau \right) } .\tag{30}
$$

We use $V = 5$ . When $V = 2 $ , the objective reduces to Eq. 3 with $K = 2 B - 2$

Table 6: The 15 frozen audio encoders. d denotes the widest read-out block. BYOL-A models are convolutional and use unequal block widths: 2048/1024/512/512/512 (d512), 2048/1024/512/1024/1024 (d1024), 2048/1024/512/2048/2048 (d2048), and 2048/1024/2048/2048/3072 (v2). BYOL-A checkpoints are obtained from the authors’ repository.
<table><tr><td>Checkpoint</td><td>Family</td><td>Objective</td><td>Params</td><td>L/d</td></tr><tr><td>facebook/wav2vec2-base</td><td>wav2vec2</td><td>contrastive</td><td>94.37M</td><td>12 / 768</td></tr><tr><td>facebook/wav2vec2-large-lv60</td><td>wav2vec2</td><td>contrastive</td><td>315.44M</td><td>24 / 1024</td></tr><tr><td>facebook/hubert-base-ls960</td><td>HuBERT</td><td>masked</td><td>94.37M</td><td>12 / 768</td></tr><tr><td>facebook/hubert-large-ll60k</td><td>HuBERT</td><td>masked</td><td>315.44M</td><td>24 / 1024</td></tr><tr><td>facebook/hubert-xlarge-1l60k</td><td>HuBERT</td><td>masked</td><td>962.50M</td><td>48 / 1280</td></tr><tr><td>microsoft/wavlm-base</td><td>WavLM</td><td>masked</td><td>94.38M</td><td>12 / 768</td></tr><tr><td>microsoft/wavlm-base-plus</td><td>WavLM</td><td>masked</td><td>94.38M</td><td>12/768</td></tr><tr><td>microsoft/wavlm-large</td><td>WavLM</td><td>masked</td><td>315.45M</td><td>24 / 1024</td></tr><tr><td>facebook/data2vec-audio-base</td><td>data2vec</td><td>masked</td><td>93.16M</td><td>12 / 768</td></tr><tr><td>facebook/data2vec-audio-large</td><td>data2vec</td><td>masked</td><td>313.28M</td><td>24 / 1024</td></tr><tr><td>AudioNTT2020-BYOLA-64x96d512</td><td>BYOL-A</td><td>augmentation</td><td>0.60M</td><td>5 / 2048</td></tr><tr><td>AudioNTT2020-BYOLA-64x96d1024</td><td>BYOL-A</td><td>augmentation</td><td>1.65M</td><td>5 /2048</td></tr><tr><td>AudioNTT2020-BY0LA-64x96d2048</td><td>BYOL-A</td><td>augmentation</td><td>5.32M</td><td>5 / 2048</td></tr><tr><td>AudioNTT2022-BY0LA-64x96d2048</td><td>BYOL-A</td><td>augmentation</td><td>6.33 M</td><td>5 /3072</td></tr><tr><td>gaunernst/vit_base_patch16_1024_128.audiomae_as2m</td><td>AudioMAE</td><td>masked</td><td>85.65M</td><td>12 / 768</td></tr></table>

Table 7: Embedding extraction. All hidden layers are retained. Audio is converted to mono and resampled to 16 kHz, then divided into non-overlapping 10 s windows. Each window is embedded independently and the resulting vectors are averaged using weights proportional to the number of valid samples in each window. Clotho clips contain an average of 2.647 windows, whereas AudioCaps clips contain exactly one. The waveform-transformer models receive normalized waveforms; BYOL-A receives 64-band log-mel spectrograms normalized using evaluation-set statistics (mean −5.4703, standard deviation 4.3999); AudioMAE receives a 1024 × 128 spectrogram represented using a 64 × 8 patch grid.
<table><tr><td>Modality</td><td>Read-out per layer</td><td>Input</td></tr><tr><td>Text (Aug)</td><td>mean over real tokens (SBERT pooling)</td><td>128 tokens</td></tr><tr><td>Text (NTP)</td><td>mean over real tokens</td><td>128 tokens</td></tr><tr><td>Vision</td><td>CLS at the second residual add</td><td>each model&#x27;s own preprocessing</td></tr><tr><td>Audio</td><td>mean over time (SUPERB convention)</td><td>10 s windows at 16 kHz</td></tr></table>

Evaluating the unimodal loss of frozen encoders. The loss shown on the x-axis of Fig. 5 is Eq. 30 evaluated on the frozen representations of the five views in the evaluation draw. The complete evaluation draw is processed jointly rather than in mini-batches, so each anchor is compared against all other rows. With n = 1024 and ${ \dot { V } } = 5 ,$ this gives $K = n V - 1 = 5 1 1 9$ negatives. We use $\tau = 0 . 1$ For each encoder, the reported loss is the minimum across layers, mirroring the maximum used for alignment: both read out the model at its best layer. The minimum keeps the loss a property of the encoder alone, identical across all panels. The corresponding chance loss is $\log ( n { \bar { V } } - 1 ) { \stackrel { . } { = } } 8 . 5 4$ nats.

Supervised baselines. MS-COCO instance annotations are reduced to the largest object per image among the 80 categories. The cross-entropy baseline uses a single Linear(384, 80) classification head on the backbone representation. SupCon uses the same 128-dimensional projection head as the augmentation-based SSL objective, with $\tau = 0 . 1$ ; all same-label views in a batch are treated as positives, while anchors without another sample of the same label are omitted. All three objectives are trained from random initialization on the same training pool for 150 epochs (23,250 steps), using batch size 256, learning rate $5 \times 1 0 ^ { - 4 }$ , weight decay 0.05, and 775 warmup steps. Thus, the three baselines differ only in their objectives

To control for label memorization, classes are partitioned according to COCO supercategories. The animal, furniture, food, and kitchen categories constitute the seen set of 33 classes (39,867 images); the remaining 46 classes are held out. The person category is omitted. Fig. 3b evaluates CMA on the 1,716 validation images belonging to held-out classes.

Table 8: The controlled encoders. Both models use AdamW with linear warmup followed by cosine decay to zero, 15 000 training steps, batches of 256 items with five views per item (1280 views per step), $\tau = 0 . 1$ as in Eq. 3, bfloat16 autocasting on compute capability $\geq 8 . 0$ , and float16 with gradient scaling otherwise. Contrastive logits are always computed in float32. No EMA is used. Training uses seeds 5000, 5001, and 5002.
<table><tr><td></td><td>Vision encoder (ViT)</td><td>Text encoder (SBERT)</td></tr><tr><td>Backbone</td><td>ViT-S/16, randomly initialised</td><td>bert-base-uncased, mean pooling</td></tr><tr><td>Architecture</td><td>12 layers, 6 heads, width 384, MLP 1536</td><td>12 layers, width 768</td></tr><tr><td>Input</td><td>2242, patch 16</td><td>40 tokens</td></tr><tr><td>Projection head</td><td>384 → 384 → 128 with GELU</td><td>768 → 768 → 128 with GELU</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay</td><td>0.05</td><td>0.01</td></tr><tr><td>Warmup steps</td><td>1000</td><td>300</td></tr><tr><td>Gradient clip</td><td>none</td><td>global norm 1.0</td></tr></table>

Alignment metrics. The bandwidth of the CKA-RBF, σ, is determined for each modality as the median distance to the k-th nearest neighbor across samples.

Layer aggregation and null calibration. We follow Gröger et al. (2026a). For frozen encoders, the reported alignment score is the maximum across all image–text layer pairs. Since this maximum is affected by the number of layers, we calibrate each encoder pair as presented in Gröger et al. (2026a).

## L.2 Datasets

Evaluation draw. Unless otherwise stated, each dataset is evaluated using a fixed subsample of n = 1024 aligned pairs drawn without replacement using a global seed. The same draw is reused across experimental stages. The larger evaluation sets in App. H are nested supersets, 1024 ⊂ 2048 ⊂ 4096, so that the sample-size analysis varies only n.

MS-COCO. Evaluation uses val2017. Images with all five captions provide 5,000 image–caption pairs; 4,952 of these images contain an instance annotation and therefore have an associated label. The evaluation draw is sampled from these images. Captions are ordered by annotation ID, and the first caption is used. Controlled experiments use train2017, containing 118,287 images.

WIT. Evaluation uses the wit\_1024 benchmark split of Huh et al. (2024), containing 1,024 image– text pairs with one caption per image. For controlled training, we construct a COCO-sized corpus from wikimedia/wit\_base by retaining the first English reference caption for each image. Benchmark images are removed from the training corpus by caption matching. The remaining data are split using a seeded shuffle into 118,287 training images and 5,000 validation images, matching the sizes of train2017 and val2017.

Clotho. We use Clotho v2.1 (Zenodo 10.5281/zenodo.4783391), specifically its evaluation split containing 1,045 clips and 5,225 captions. Clips have durations between 15 and 30 seconds (mean 22.5 seconds) and a native sampling rate of 44.1 kHz. We sample 1,024 clips for evaluation. Audio is converted to mono, resampled to 16 kHz, and stored. Each clip has five human captions in a fixed column order, which defines the view order, and we use the first.

AudioCaps. Among the 1,470 clips in the official validation and test splits, we use all 856 available valid clips, containing a caption and valid WAV files. Each is exactly 10.00 seconds long at 48 kHz and is downsampled by a factor of three to 16 kHz, yielding a single 10-second window per clip.

Image augmentations. For the controlled experiments and frozen-encoder image-loss evaluation, each image is transformed using the following pipeline: random resized crop to 224<sup>2</sup> with area scale (0.5, 1.0); horizontal flip with probability 0.5; color jitter with brightness, contrast, and saturation 0.2 and hue 0.05, applied with probability one; and ImageNet normalization. Five independent draws produce the five views of each image. The label-free control in Fig. 3b uses the DINO-style multi-crop augmentation associated with its baseline. Each image produces two global crops with scale (0.4, 1.0) and four local crops with scale (0.05, 0.4), all resized to $2 2 4 ^ { 2 }$ . Horizontal flipping is applied to all crops. Color jitter (0.4, 0.4, 0.2, 0.1) is applied with probability 0.8, grayscale with probability 0.2, and Gaussian blur with $\sigma \in ( 0 . 1 , 2 . 0 )$ is applied to the first global crop with probability one, to the second global crop with probability 0.1, and to local crops with probability 0.5. Solarization is applied to the second global crop with probability 0.2.

Text augmentations. The main text views are generated using Qwen with a single user message and no system message. The generator receives only the source caption and has no access to the corresponding image, other captions, or labels. The prompt is:

You rewrite a caption. Output only JSON.   
Rewrite the caption below as EXACTLY <N> alternatives with the same meaning.   
Vary word choice, syntax and clause order; the <N> must differ from each   
other and from the original by more than a synonym swap.   
- Keep the original’s register and length (a plain caption, 8-15 words). No   
commentary.   
- No preamble, no numbering.   
Return only a JSON array of exactly <N> strings.   
Caption: "<CAPTION>"

Here <N> = 4, so the source caption constitutes view 0 and the four generated captions constitute views 1–4. Decoding uses temperature 0.9, top-p 0.95, and a fixed generation seed. Outputs are required to contain exactly four English strings. Outputs failing this constraint are regenerated up to two times; after two failed attempts, the source caption is used as fallback. This filtering prevents off-language outputs from artificially increasing within-group variation. For WIT and Clotho, one prompt constraint is modified to account for the properties of their captions. For WIT, whose captions have a median length of seven words, the length instruction is replaced with: Match the original’s register and length, however short it is. Keep proper nouns, titles and technical terms verbatim. No commentary. For Clotho, the instruction is replaced with: Keep the original’s register and length (a plain caption, 8-20 words). Keep the same events in the same order; do not add, drop or reorder one. No commentary. All remaining prompt components are unchanged. App. D repeats the experiment using microsoft/Phi-3.5-mini-instruct and google/gemma-4-12B-it.

Back-translation views. Back-translation provides an alternative text augmentation procedure that does not rely on prompting or stochastic decoding. View 0 is the source caption, while views 1–4 are obtained by translating the source through German, French, Spanish, and Russian, respectively, and translating each result back to English. The translations use eight Helsinki-NLP/opus-mt checkpoints, corresponding to the four pivot languages in both translation directions. Decoding uses beam search with four beams and no sampling. The resulting corpus is deterministic. Each view uses a different pivot language, producing four translation systems rather than four stochastic samples from a single system.

## L.3 The Bounds

Evaluation of the bounds. The bound figures visualize the analytical bounds derived in Sec. 3. No encoder training or sampling is involved. The lower bound depends on both modality losses, but a single curve suffices because it is given by cos $\left( \theta _ { X } + \theta _ { Y } \right)$ and decreases monotonically in each loss. Thus, the worst case under max $\mathcal { L } _ { X } , \dot { \mathcal { L } } _ { Y } \leq \dot { \mathcal { L } }$ occurs on the diagonal. The upper bound depends only on the loss of one modality and is therefore plotted directly against $\mathcal { L } _ { X \cdot } \mathbf { A }$ dotted line marks $\mathrm { { C K A } = 0 }$ . The excluded corner of each bound is also indicated: the lower bound excludes the lower-left region, whereas the upper bound excludes the upper-right region. Each closed-form implementation is additionally verified against a 50-digit evaluation of the same expression.

Parameter grids and valid domains. Both bound figures use $\tau \in \{ 0 . 1 , 0 . 3 , 0 . 5 \}$ and $\sigma ^ { 2 } \in \mathsf { \Gamma }$ {0.8, 1.0, 1.2}. The lower-bound panels additionally use $\chi \in \{ 0 . 5 , 1 , 1 . 5 \}$ . The upper-bound panels use the parameter triples

$$
( \ell _ { X } , \epsilon _ { X } , \sigma _ { X } ) \in ( 0 . 5 , 0 . 2 , 0 . 8 ) , ( 1 , 0 . 4 , 1 . 0 ) , ( 1 . 5 , 0 . 6 , 1 . 2 ) .
$$

The side-by-side comparison uses the slice $\tau = 0 . 3$ and $\sigma ^ { 2 } = 1$ . Curves are plotted only over the region in which the corresponding bound is non-vacuous. The lower bound terminates at its zero

crossing, below which it is weaker than the trivial constraint CKA ≥ 0. The upper bound is cut where it crosses one and terminates at its maximum possible value as in Lemma 8.

## L.4 Hardware and Software

Hardware and Software. All experiments use Python on a single GPU without distributed training or gradient accumulation. GPU selection is performed at runtime. The reported experiments were executed on NVIDIA RTX PRO 6000 Blackwell (98 GB), B200 (183 GB), H200 (144 GB), A100 80 GB (PCIe and SXM4), L40S (46 GB), RTX 6000 Ada (49 GB), and L4 (23 GB) GPUs, with 8–32 CPU cores and 32–150 GB host memory per job.

Determinism and seeding. Evaluation uses fixed random seeds that are independent of the training seeds. These seeds determine evaluation subsampling, loss-evaluation blocks, kernel-statistic samples, alignment subsamples, and frozen augmentation draws. Consequently, different training runs share the same evaluation data and augmentation realizations and differ only through their training seeds. Caption generation uses a separate generation seed, while back-translation is deterministic and requires no random seed.