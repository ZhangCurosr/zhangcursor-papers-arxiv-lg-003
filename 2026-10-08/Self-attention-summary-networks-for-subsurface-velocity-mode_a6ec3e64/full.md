# Self-attention summary networks for subsurface velocity-model building from common-image gathers

Shiqin Zeng, Yunlin Zeng, Abhinav Prakash Gahlot,

Zijun Deng, and Felix J. Herrmann

Georgia Institute of Technology

## Abstract

Common-image gathers (CIGs) contain physically meaningful information about velocitymodel errors through reflector focusing and residual moveout, but in conventional imaging workflows they are typically used only as diagnostic tools. In this work, we propose a multiscale self-attention summary network that maps high-dimensional 3D CIG volumes into compact conditioning embeddings for probabilistic subsurface velocity inversion. These learned em beddings preserve ofset-dependent kinematic structure and spatial coherence while reducing variability caused by background-velocity mismatch. Conditioned on these summary embeddings, a flow-matching model learns a transport from a Gaussian source distribution to the posterior distribution of plausible velocity fields. Numerical experiments show that, compared with direct conditioning on raw CIGs, the proposed summary network improves posterior velocity inference. In particular, the multiscale attention design provides greater robustness to background-model mismatch, yielding more accurate posterior reconstructions and lower predictive uncertainty.

## 1 Introduction

Velocity model building remains a central challenge in seismic imaging because the inverse problem is high-dimensional, nonlinear, and non-unique (Symes, 2008). Conventional optimization based approaches such as full-waveform inversion (FWI) can recover high-resolution subsurface models when supported by adequate low-frequency content and a suficiently accurate initial model, but they remain sensitive to initialization, prone to cycle skipping, and typically yield a single deterministic estimate (Virieux and Operto, 2009; Symes, 2008). In contrast, common-image gathers (CIGs) contain ofset-dependent information about reflector focusing and residual moveout, making them informative indicators of velocity-model mismatch (Biondi, 2006; Biondi and Symes, 2004).

Despite being constructed under imperfect background velocity-models, CIGs retain structured, mismatch-dependent patterns such as residual moveout and defocusing, which remain informative for velocity inference and are also exploited in uncertainty-aware inversion frameworks based on subsurface extensions (Yin et al., 2024). Building on this perspective, we use CIGs as conditioning variables for probabilistic velocity-model inversion by combining a self-attention summary encoder for high-dimensional CIG representations with a conditional flow-matching model for posterior velocity-model generation (Lipman et al., 2023). The resulting framework learns a distribution over plausible velocity fields conditioned on the compact latent representation of the CIG, improving both reconstruction accuracy and uncertainty quantification.

## 2 Method

## 2.1 3D CIG representation

We use subsurface-ofset common-image gathers (CIGs), shown in the middle column of Fig. 3, as the extended-image representation for migration-velocity analysis (Biondi and Symes, 2004; Biondi, 2006). Let the observed seismic data be denoted by $\mathbf { d } _ { \mathrm { { o b s } } }$ , generated from the true subsurface model $\mathbf { m } _ { \mathrm { g t } }$ through the forward operator ${ \mathcal { F } } ,$ i.e., $\mathbf { d } _ { \mathrm { { o b s } } } = \mathcal { F } ( \mathbf { m } _ { \mathrm { { g t } } } )$ . Given a background model $\mathbf { m } _ { 0 }$ , we define the time-domain extended imaging condition as in Eq. 1

$$
I ( \mathbf { x } , \mathbf { h } ; \mathbf { m } _ { 0 } ) = \sum _ { s } \int u _ { s } ( \mathbf { x } - \mathbf { h } , t ; \mathbf { m } _ { 0 } ) u _ { r } ( \mathbf { x } + \mathbf { h } , t ; \mathbf { d } _ { \mathrm { o b s } } ^ { ( s ) } , \mathbf { m } _ { 0 } ) d t ,\tag{1}
$$

where x is the image location, h is the subsurface half-ofset, and s indexes the source experiment. Here, $u _ { s }$ denotes the source wavefield propagated in m<sub>0</sub>, and $u _ { r }$ denotes the receiver wavefield obtained by back-propagating the observed data in the same background model. When $\mathbf { m } _ { 0 }$ is kinematically consistent with $\mathbf { m } _ { \mathrm { g t } }$ , reflected energy focuses near zero ofset $( \mathbf { h } = \mathbf { 0 } )$ ; otherwise, model error produces residual moveout and defocusing across ofsets. In practice, $I ( \mathbf { x } , \mathbf { h } ; \mathbf { m } _ { 0 } )$ is discretized as a real-valued tensor $\mathbf { y } \in \mathbb { R } ^ { D \times H \times W }$ , where D is the number of sampled ofsets and (H, W) are the spatial dimensions. Each spatial location is therefore associated with an ofset-dependent response vector that encodes local kinematic and structural information relative to the background model. This representation, however, remains highly redundant across both ofset and space.

![](images/9b5b159a97162c5bde5c5df9f9ffd4baaa87a8fcee708010c5b11ca0e67cf575.jpg)  
Figure 1: Overall workflow of the proposed method.

## 2.2 Multiscale self-attention summary network

To obtain a compact and robust seismic representation of CIGs, we learn a summary encoder $\Phi _ { \psi } ( \cdot )$ that maps the input CIGs to a low-dimensional latent embedding. We flatten the spatial grid into tokens indexed by $i \in \{ 1 , \dots , H W \}$ , where each index corresponds to a spatial location and is associated with an ofset-response vector from the raw CIGs. Specifically, let $\mathbf { y } _ { i } \in \mathbb { R } ^ { D }$ denote the ofset vector at token i. We then apply local windowed self-attention at multiple window sizes, since the query, key, and value vectors are all computed from the same token set (Vaswani et al., 2017). For a token index i and a neighboring token index $j$ within a local window $\mathcal { W } ( i )$ , we define $\mathbf { q } _ { i } = W _ { Q } \mathbf { y } _ { i } , \mathbf { k } _ { j } = W _ { K } \mathbf { y } _ { j }$ , and $\mathbf { v } _ { j } = W _ { V } \mathbf { y } _ { j }$ , where $W _ { Q } , W _ { K }$ , and $W _ { V }$ are learnable projection matrices. The attention weights from token i to token $j$ are given by

$$
\alpha _ { i j } = \frac { \exp \left( \mathbf { q } _ { i } ^ { \top } \mathbf { k } _ { j } / \sqrt { d _ { a } } \right) } { \sum _ { j ^ { \prime } \in \mathcal { W } ( i ) } \exp \left( \mathbf { q } _ { i } ^ { \top } \mathbf { k } _ { j ^ { \prime } } / \sqrt { d _ { a } } \right) } ,\tag{2}
$$

where $\mathscr { W } ( i )$ denotes the local spatial window centered at token i, and $d _ { a }$ is the attention dimension. The attended feature at token i is then $\begin{array} { r } { \tilde { \mathbf { y } } _ { i } = \sum _ { j \in \mathcal { W } ( i ) } \alpha _ { i j } \mathbf { v } _ { j } } \end{array}$ . This mechanism allows each spatial location to aggregate information from nearby locations with similar ofset-dependent responses, reinforcing spatially coherent reflector structure while suppressing locally inconsistent artifacts. To encode structure across diferent spatial extents, we employ multiple attention window sizes, similar in spirit to window-based self-attention architectures(Liu et al., 2021). This allows the model to capture both local kinematic variations and larger-scale continuity. After attention-based aggregation, the features are compressed through a learned channel-mixing projection to reduce redundancy while preserving physically relevant information. The final embedding, illustrated by the summary-network outputs in Fig. 3, serves as a compact, mismatch-aware conditioning signa for downstream probabilistic velocity generation.

## 2.3 Velocity-model generation via conditional flow matching

Given the summary encoder, we model the conditional distribution of subsurface velocity models using conditional flow matching. We jointly learn the encoder $\Phi _ { \psi }$ and the conditional vector field $u _ { \phi }$ by minimizing Eq. (3):

$$
\mathcal { L } ( \psi , \phi ) = \mathbb { E } _ { t , \mathbf { x } _ { t } , \mathbf { d } _ { \mathrm { s e i s } } ^ { \mathrm { c I G } } } \left[ \lVert u _ { \phi } ( \mathbf { x } _ { t } , t , \Phi _ { \psi } ( \mathbf { y } ) ) - u _ { t } ( \mathbf { x } _ { t } \mid \Phi _ { \psi } ( \mathbf { y } ) ) \rVert _ { 2 } ^ { 2 } \right]\tag{3}
$$

In this work, we use the rectified-flow probability path (Liu et al., 2023; Lipman et al., 2023), defined by the linear interpolation ${ \bf x } _ { t } = ( 1 - t ) { \bf x } _ { 0 } + t { \bf x } _ { 1 }$ , where $\mathbf { x } _ { 0 } \sim \mathcal { N } ( 0 , I )$ and $\mathbf { x } _ { 1 } \sim p _ { \mathbf { m } }$ of subsurface velocity modeling. The corresponding target vector field is $\begin{array} { r } { u _ { t } = \frac { d \mathbf { x } _ { t } } { d t } = \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } } \end{array}$ . While we adopt the linear rectified-flow path in this work, other choices of probability paths or optimal transport constructions can also be used. At inference, we sample $\mathbf { x } _ { 0 } \sim \mathcal { N } ( 0 , I )$ and generate the subsurface velocity model by integrating the learned conditional ODE in Eq. (4):

$$
\mathbf { x } _ { 1 } = \mathbf { x } _ { 0 } + \int _ { 0 } ^ { 1 } u _ { \phi } ( \mathbf { x } _ { t } , t , \Phi _ { \psi } ( \mathbf { y } ) ) \ d t .\tag{4}
$$

Accordingly, ODE-based sampling provides a practical and theoretically justified choice: it yields a deterministic realization of the learned transport while preserving the probabilistic interpretation via random Gaussian initialization. Since the objective is to match the conditional marginal distribution, the probability-flow ODE is suficient provided that it transports the source distribution to the same target marginals (Song et al., 2021; Albergo et al., 2025). Compared with explicitly simulating an SDE, this results in a simpler and often more computationally eficient inference procedure, while still enabling uncertainty quantification through repeated sampling. An overview of the full pipeline is shown in Fig. 1.

## 3 Experiments and results

## 3.1 Compact latent representation from summary encoding

We first examine how the compressed channel size $C \ll D$ , afects latent compactness and downstream predictive performance. As shown in Fig. 2, increasing C from 4 to 10 leads to a monotonic decrease in the singular value ratio $( \sigma _ { \mathrm { m i n } } / \sigma _ { \mathrm { m a x } } )$ , from 7.2% to 0.22%. We use this ratio as a spectral proxy for latent compactness because smaller values indicate stronger suppression of trailing singular directions and hence a more concentrated spectrum with lower efective dimensionality (Roy and Vetterli, 2007; Jing et al., 2020). Across $C = 4 ~ \mathrm { t o } ~ 1 0$ , downstream performance remains stable overall and improves relative to $C = 4 .$ , with lower RMSE, reduced predictive standard deviation, and higher SSIM for $C = 6 , 8 ,$ and 10. These results suggest that $C = 6 { - } 1 0$ provides a favorable balance between latent compactness and predictive quality. In the remainder of this work, we use $C = 8$ as the default latent dimensionality. By contrast, increasing to $C = 1 2$ yields a larger singular value ratio and degraded downstream performance, indicating diminishing returns beyond the preferred range.

![](images/1afecc31ba14ea40afc59eef88501c34a9c73207262da5bdbf1d70556dee60b2.jpg)

![](images/fab990ba2ca82030fcd4bc4025627614baadd2e725ee2c5de778c43dee1e20b9.jpg)  
(a) Singular value spectra across compressed channel sizes. (b) Downstream performance comparison across channel sizes.  
Figure 2: Selection of compressed channel size based on spectral compactness and downstream predictive performance.

## 3.2 Behavior under background-velocity model errors

The summary network behaves diferently under diferent initial background-velocity models. As shown in Fig. 3, with the 2D smoothed initial background, the input CIGs remain relatively well focused near zero ofset, and the attention features preserve broad ofset-dependent structure. With the 1D initial background, the input CIGs show stronger defocusing and noisier responses around zero ofset. In this case, the attention features shift their emphasis away from near-zero-ofset-dominated responses and redistribute it across other ofset-related components.

These observations suggest that the summary network does not directly recover the correct focusing, but instead learns a mismatch-aware redistribution of ofset information. Self-attention adaptively aggregates information from neighboring spatial locations according to the input feature patterns, while reweighting and mixing ofset-dependent responses within each spatial location. As a result, the summary network adjusts its representation to the mismatch reflected in the input CIGs and produces a summary embedding that is more robust for downstream probabilistic velocity generation.

![](images/1adb8a96db462daf7da9087ebc9fb07024e81af9ccb484266a97bc1601e47872.jpg)  
Figure 3: Summary-network response under diferent background models.

## 3.3 Efect of multiscale attention windows

Fig. 4 illustrates how the attention window size controls the scale of information retained in the summary features under the 1D initial background model. At $w s = 1$ , the feature largely preserves high-frequency local detail, since no spatial neighborhood aggregation is introduced. At $w s = 2$ limited neighborhood mixing begins to suppress local fluctuations while retaining fine-scale structure. At $w s = 4$ , the feature becomes smoother and more coherent while still preserving consistent reflector geometry. As the window size increases further, the representation becomes progressivel coarser. At $w s = 8$ , the feature emphasizes broader structural continuity, whereas at $w s = 1 6$ it becomes overly smooth and blocky, with less local detail.

Overall, these feature patterns indicate that diferent window sizes capture complementary information. Smaller windows preserve sharper local responses, while larger windows provide broader structural context. This motivates the use of multiscale attention to combine fine-scale detail with larger-scale contextual aggregation.

## 3.4 Summary conditioning for posterior inference

We next evaluate how the quality of the conditioning representation afects posterior velocity-model inference from CIGs. Fig. 5 compares three configurations under the same 1D initial background model: using the raw CIG directly as conditioning, using a single-scale summary network, and using a multiscale summary network. Moving from raw CIG conditioning to single-scale summary conditioning to multiscale summary conditioning, the posterior mean becomes progressively closer to the ground-truth velocity model, with a monotonic reduction in RMSE and a corresponding increase in SSIM. The spatial RMSE maps further show that, when the raw CIG is used directly as conditioning, reconstruction errors are more broadly distributed and more pronounced near layered interfaces and structurally complex regions. Introducing a summary network makes these errors more localized, while the multiscale summary network yields the cleanest error profile overall. Similarly, the posterior standard-deviation maps indicate that uncertainty is progressively reduced from the raw-CIG conditioning baseline to the single-scale and multiscale summary variants. The multiscale summary representation yields the most concentrated and best-structured uncertainty field, indicating that combining multiple attention scales improves both posterior accuracy and uncertainty control. Together, Fig. 4 and Fig. 5 show that multiscale attention improves the conditioning representation at the feature level and translates this improvement into more accurate and consistent posterior velocity estimates.

![](images/e05638fe75560e7d966846b837e67ad60635e8718cd037a37e53e02ed9d047df.jpg)  
Figure 4: Attention features at the selected peak-energy ofset index (= 8) for diferent attention window sizes.

![](images/b33bdccd7c4e67edb287284651294fe9ac17b26e6f6a3686f3e6662663d55571.jpg)  
Figure 5: Comparison of posterior velocity inference under three conditioning strategies.

## 4 Conclusion and future work

We propose a probabilistic framework for subsurface velocity-model building that combines a multiscale self-attention summary network that acts on subsurface-ofset Common Image Gathers (CIGs) with conditional flow matching. The summary network converts high-dimensional CIGs into compact conditioning representations, and the flow-matching model uses them to generate posterior velocity models. The results show that this combination improves robustness under background model errors and yields more accurate posterior samples and more concentrated uncertainty fields than direct raw CIG conditioning. Overall, multiscale self-attention and conditional flow matching provide an efective approach for uncertainty-aware velocity inference from CIGs. Future work will test generalization across diverse initial background-velocity models by training on CIGs with diferent velocity errors and evaluating on unseen backgrounds. This would provide stronger evidence that the summary network learns adaptive mismatch-aware representations.

## Acknowledgments

This research was carried out with the support of the Georgia Research Alliance and the partners of the ML4Seismic Center.

## References

Michael S. Albergo, Nicholas M. Bofi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and difusions. Journal of Machine Learning Research, 26(209):1–80, 2025.

Biondo L Biondi. 3D seismic imaging. Society of Exploration Geophysicists, 2006.

Biondo L. Biondi and William W. Symes. Angle-domain common-image gathers for migration velocity analysis by wavefield-continuation imaging. Geophysics, 69(5):1283–1298, 2004. doi: 10.1190/1.1801945.

Li Jing, Jure Zbontar, and Yann LeCun. Implicit rank-minimizing autoencoder. In Advances in Neural Information Processing Systems, volume 33, 2020.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.

Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–10022, 2021.

Olivier Roy and Martin Vetterli. The efective rank: A measure of efective dimensionality. In 15th European Signal Processing Conference (EUSIPCO), 2007.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic diferential equations. In International Conference on Learning Representations, 2021.

William W. Symes. Migration velocity analysis and waveform inversion. Geophysical Prospecting, 56(6):765–790, 2008. doi: 10.1111/j.1365-2478.2008.00698.x.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pages 5998–6008, 2017.

Jean Virieux and Stéphane Operto. An overview of full-waveform inversion in exploration geophysics. Geophysics, 74(6):WCC1–WCC26, 2009.

Ziyi Yin, Rafael Orozco, Mathias Louboutin, and Felix J. Herrmann. WISE: Full-waveform variational inference via subsurface extensions. Geophysics, 89(4):A23–A28, 2024. doi: 10.1190/ geo2023-0744.1.