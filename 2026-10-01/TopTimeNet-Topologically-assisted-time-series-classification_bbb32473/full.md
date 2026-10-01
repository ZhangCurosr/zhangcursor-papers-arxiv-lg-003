# TopTimeNet: Topologically-assisted time-series classification model

Sharareh Sayyad<sup>1</sup> and Sophia Bazzi<sup>2</sup>

<sup>1</sup>Department of Mathematics and Statistics, Washington State University,

Pullman, Washington 99164-3113, USA<sup>\*</sup>

<sup>2</sup>European Molecular Biology Laboratory, EMBL Hamburg, c/o DESY,

Building 25A, Notkestraße 85, 22603 Hamburg, Germany

Distinguishing periodic from chaotic dynamics in a time series is a fundamental challenge in both physics and engineering. Yet, end-to-end learned architectures must discover both a representation and a decision boundary from data, at substantial cost. We introduce TopTimeNet, which decouples these tasks: a fixed, non-learned stage extracts a 42-dimensional geometric and topological descriptor from Takens delay embeddings and persistent homology, and a lightweight learnable stage performs classification. On a benchmark of 49 nonlinear dynamical systems, a 1,638-parameter configuration matches the mean accuracy of one with 33× more trainable parameters. Additionally, this approach delivers mean accuracy comparable to convolutional neural networks and surpasses the average performance of converged Transformer models, while requiring three to four orders of magnitude fewer trainable parameters. Robustness also depends sharply on where noise is introduced: TopTimeNet degrades gracefully under perturbations to its precomputed features, but degrades sharply when noise is introduced into the raw signal and the full feature-extraction pipeline is recomputed, showing that robustness to perturbations of the precomputed features does not imply robustness of the complete raw-signal-to-prediction pipeline. These results show that decoupling fixed geometric and topological feature construction from a lightweight discriminative stage can achieve comparable classification accuracy with substantially fewer trainable parameters.

## I. INTRODUCTION

Nonlinear dynamical systems can exhibit qualitatively diferent regimes of motion, ranging from periodic and quasiperiodic oscillations to chaos. In many experiments, however, the governing equations are unknown, or only a subset of the state variables is accessible. The dynamical regime must then be inferred from a finite time series. This inference problem is nontrivial: chaotic and stochastic processes can both generate irregular signals, and finite data length or measurement noise may obscure the distinction between them [1–3].

Classical nonlinear time-series analysis ofers several tools to address this problem. The largest Lyapunov exponent quantifies sensitivity to initial conditions, while correlation dimension estimates the fractal scaling of the attractor, giving a measure of its efective dimensionality [4]. Power spectra and autocorrelation functions provide information about temporal structure. These methods are valuable, but each probes a particular property of the signal. Correlation-dimension estimates, for example, are well known to be sensitive to observational noise [5, 6]. More generally, finite data and limitations in phase-space reconstruction can complicate the interpretation of nonlinear time-series measures [3]. For this reason, a reliable analysis often requires several diagnostics rather than a single scalar measure.

Machine-learning (ML) methods ofer a complementary strategy. Rather than choosing a specific dynamical indicator in advance, they learn features useful for classification directly from data. Deep neural networks have been used to distinguish chaotic from non-chaotic time series across discrete and continuous dynamical systems [7]. Researchers have also studied data-driven learning methods to distinguish deterministic chaos from stochastic behavior [8, 9]. Many ML approaches operate directly on the raw time series [10, 11]. Convolutional neural networks (CNNs) can learn local and mul tiscale temporal patterns through convolutional fil ters [12]. Transformer-based models employ attention to capture dependencies and interactions across longer time scales [13]. While these architectures are flexible, they often require many trainable parameters and substantial computational resources to learn a useful representation from scratch. For dy namical systems, we can instead extract relevant geometric and topological structure before learning begins, allowing the learnable stage to focus on projecting, combining, and classifying these features. This raises a central question we address in this work: how much learnable capacity is needed once we make such structure explicit?

Topological data analysis (TDA) ofers a way to construct such features. A scalar time series can first be mapped to a point cloud in a reconstructed phase space using delay-coordinate embedding [14]. The resulting point cloud retains geometric information about the observed trajectory. Persistent homology then tracks topological features across diferent length scales. Connected components are described by zeroth homology, while loops are described by first homology. Persistence-based summaries have been used to distinguish dynamical states such as periodic and chaotic behavior [15]. Tempelman and Khasawneh further report that persistence-based chaos detection remains tolerant to moderate observational noise in several benchmark systems [16]. In Sec. IV B, we separately examine robustness when noise is introduced before and after feature extraction.

Several representations have been developed to adapt persistent homology information for statistical learning. Persistence landscapes provide functional summaries of persistence diagrams [17]. Persistence images map diagrams to stable finite-dimensional vectors [18]. Signature-based feature maps provide another representation of persistence barcodes for statistical learning [19]. Persistent entropy has also been used to construct stable summary functions that incorporate information from Betti curves [20]. For time-series data, Umeda combined an engineered topological feature with a CNN-inspired learning architecture [21]. Karan and Kaygun later developed persistent-homology-based pipelines for univariate time-series classification [22].

The use of topological features in time-series learning is thus already well established, and prior work has combined several persistence-derived summaries within a single pipeline [22]. For instance, Karan and Kaygun summarize persistence diagrams using diagram distances, persistent entropy, and scalar norms of Betti curves and persistence landscapes; their pipeline does not include persistence images or geometric statistics computed directly from the embedding point cloud.

Our study focuses on how these complementary geometric and topological feature groups can be organized within a compact model. We introduce TopTimeNet, an architecture for classifying periodic and chaotic time series. Each segmented time series is first mapped to a delay-coordinate point cloud. We compute geometric features directly from this point cloud and obtain additional topological descriptors from persistent homology. The resulting 42 features are divided into five groups: geometric, entropy, lifetime, Betti-curve, and persistence-image features. This feature-extraction stage contains no learned weights, although it depends on fixed analysis settings such as the embedding parameters and persistence-image resolution.

The five feature groups are then passed to the learnable part of TopTimeNet. Each group is projected into a common embedding space. A fusion layer combines the projected features, and a classification head produces the final prediction. Learning therefore focuses on interactions between the extracted feature groups rather than reconstructing the full representation from the raw signal.

This design enables us to investigate the amount of trainable capacity needed for the classification task. We compare compact and higher-capacity TopTimeNet configurations selected through the same hyperparameter search. The two configurations difer by a factor of approximately 33 in parameter count, yet achieve nearly identical mean classification accuracy. We also compare TopTimeNet with CNN and Transformer baselines trained directly on the raw time series.

We additionally examine the response of TopTimeNet to Gaussian perturbations introduced at two distinct points in the pipeline: directly on the precomputed feature vectors, and on the raw time series itself, with the full delay-embedding and persistent-homology computation repeated on the corrupted signal.

Under the tested perturbations, accuracy degrades more sharply when noise is introduced in the raw time series and the features are recomputed than when noise is applied directly to the precomputed feature vectors. This shows that robustness to feature-level perturbations does not imply robustness of the complete raw-signal-to-prediction pipeline.

The remainder of this paper is organized as follows. Section II describes the dataset, the geometric and topological feature-extraction pipeline, the TopTimeNet architecture, and the training procedure. Section III illustrates the extracted features using the double pendulum as an example. Section IV presents the classification results, the response to feature-level and raw-signal noise, and comparisons with CNN and Transformer baselines. Finally, Section V summarizes the main findings and outlines directions for future work.

## II. METHODOLOGY

In this section, we present details on dataset generation, the TopTimeNet architecture, and the procedures used to train and evaluate the models for time-series classification.

## A. Dataset generation

To construct the time-series dataset, we used a catalog of 49 nonlinear dynamical systems from the

Teaspoon library [23]. This catalog includes various discrete maps, such as the logistic map; dissipative flows, including the Lorenz attractor [24] and the driven pendulum; and conservative flows, such as the H´enon-Heiles system [25].

We simulated each system in the catalog in both periodic and chaotic parameter regimes, where available, yielding 96 (dataset, state) pairs, up to a fixed number of time steps, �, usually set to 100,000. For systems described by � coupled diferential equations, the simulations produce �-dimensional trajectories. We treat each component of these solutions separately to ensure that each time series in our dataset is univariate, yielding 239 individual signals across the 96 pairs. To further increase the number of samples for each dynamical regime, we segment each of these signals into segments of length �. With $N = 1 0 0 { , } 0 0 0$ and $T = 1 { , } 0 0 0$ , each signal produces 100 segments per signal, leading to a total of 23,900. However, 600 of these segments are discarded for being identically zero. The remaining 23, 300 segments are unbalanced between classes, with 12, 000 labeled as chaotic and 11, 300 as periodic. To achieve balance, we undersample the majority class (chaotic) to match the minority class, resulting in a final, balanced dataset of 22, 600 time series, equally divided between the periodic and chaotic classes.

## B. TopTimeNet model

Our TopTimeNet model, schematically illustrated in Fig. 1, consists of two main components: feature extraction and learnable fusion and classification. We present each component below.

## 1. Feature extraction

Starting from a segmented time series $\boldsymbol { x } ( t ) \in \mathbb { R } ^ { T }$ ， we employ Takens’ delay embedding [14] to map the signal into a reconstructed phase space. The corresponding delay-coordinate vectors are $x _ { \mathrm { e m b d } } ( t ) =$ $[ x ( \bar { t } ) , x ( t + \tau ) , \ldots , x ( t + ( d _ { \mathrm { e m b } } - 1 ) \tau ) ]$ , and their collection forms the point cloud. Here, $d _ { \mathrm { e m b } }$ is the embedding dimension and � is the time delay. As a result of this process, $N _ { \mathrm { p t } } ~ = ~ T - ( d _ { \mathrm { e m b } } - 1 ) \tau$ points, each with dimension $d _ { \mathrm { e m b } }$ , are created in each point cloud. For the results reported here, we set $d _ { \mathrm { e m b } } = 2$ and $\tau = 2 0$ . These embedding parameters are fixed throughout the hyperparameter search and final evaluation and are not included in the random search described in Sec. II C 2.

Once the point clouds for all segmented time series have been constructed, the pipeline splits into two parallel branches that extract geometric and topological features.

The geometric branch operates directly on the point cloud. Here, pairwise Euclidean distances between all $N _ { \mathrm { p t } }$ points in the point cloud are computed using

$$
D _ { i j } = \| x _ { i } - x _ { j } \| _ { 2 } = \sqrt { \sum _ { k = 1 } ^ { d _ { \mathrm { e m b } } } ( x _ { i , k } - x _ { j , k } ) ^ { 2 } } ,\tag{1}
$$

where $i , j \in \{ 1 , \ldots , N _ { \mathrm { p t } } \}$

Given the pairwise distance matrix �, we extract four geometric features. The first is the diameter of the point cloud,

$$
d _ { \mathrm { p t } } = \operatorname* { m a x } _ { 1 \leq i , j \leq N _ { \mathrm { p t } } } D _ { i j } .\tag{2}
$$

We then compute the nearest-neighbor distance of every point in the point cloud as

$$
D _ { n n , i } = \operatorname* { m i n } _ { j \neq i } D _ { i j } ,\tag{3}
$$

for $i \in \{ 1 , \ldots , N _ { \mathrm { p t } } \}$ . The mean and standard deviation of the nearest-neighbor distances $D _ { n n } \in \mathbb { R } ^ { N _ { \mathrm { p t } } }$ constitute the next two geometric features.

Finally, we construct a Grassberger–Procaccia correlation-dimension proxy [26, 27]. The correlation integral at distance � reads

$$
C ( r ) = \frac { 2 } { N _ { \mathrm { p t } } ( N _ { \mathrm { p t } } - 1 ) } \sum _ { i < j } \mathbb { 1 } ( D _ { i j } < r ) ,\tag{4}
$$

where 1 denotes the indicator function. For a fractal set, $C ( \boldsymbol r )$ is expected to exhibit power-law scaling over an appropriate scaling regime as $C ( r ) \propto r ^ { d _ { \mathrm { c r } } }$ where $d _ { \mathrm { c r } }$ denotes the correlation dimension. Given two radii $r _ { s }$ and $r _ { l } ,$ one may approximate $d _ { \mathrm { c r } }$ as

$$
d _ { \mathrm { c r } } \approx \frac { \log ( C ( r _ { l } ) / C ( r _ { s } ) ) } { \log ( r _ { l } / r _ { s } ) } .\tag{5}
$$

We choose $r _ { s }$ and $r _ { l }$ to be the 10th and 50th percentiles of the pairwise-distance distribution, respectively, providing a reproducible, data-adaptive pair of radii. Because the slope is evaluated between these two radii rather than fitted over an identified scaling regime, we treat $d _ { \mathrm { c r } }$ as a correlationdimension proxy rather than a full correlationdimension estimate. This proxy is the final feature extracted by the geometric branch.

The second branch computes Vietoris–Rips persistent homology of each point cloud for a fixed number $H _ { \mathrm { m a x } }$ of homology dimensions, yielding one persistence diagram for each $H _ { k }$ , with $k = 0 , \ldots , H _ { \mathrm { m a x } } -$

![](images/85ca5fa2bedec1a505eb8a6f262c291cf4bf55ddfa74e4a9d49bdf27a189a63d.jpg)  
Figure 1. Schematic illustration of the TopTimeNet algorithm. The segmented time-series data first pass through a non-trainable feature-extraction stage of the algorithm, where point clouds are generated and geometric and topological statistics are extracted. This process creates a complete feature set that will be used in subsequent steps. The five groups of extracted features are then processed through a Group Projector and combined using a Fusion Layer. Finally, the combined features are classified using a Classifier Head to produce the final class logits.

1 [28].

The Vietoris–Rips construction builds a nested sequence of simplicial complexes by growing a ball of radius � around every point in the cloud and increasing � from zero. Whenever two balls overlap, an edge is added between their centers, and whenever all the edges among a set of points are present, forming a triangle, tetrahedron, or higher-dimensional simplex, that simplex is added as well, regardless of whether the corresponding balls share a common intersection. $\mathrm { A t } \ r = 0 .$ , the complex consists only of isolated points, and as � increases, edges and faces are added progressively, connecting the point cloud into an increasingly dense complex.

As � grows, topological features across dimensions appear and disappear: connected components $\left( H _ { 0 } \right)$ loops $\left( H _ { 1 } \right)$ , and, more generally, �-dimensional holes $\left( H _ { k } \right)$ for � up to $H _ { \mathrm { m a x } } { - } 1$ . Each such feature � is born at the smallest radius $r _ { \mathrm { b } }$ at which it first appears, and dies at the radius $r _ { \mathrm { d } }$ at which it merges with an older feature or is filled in by higher-dimensional simplices. Because two balls of radius � first overlap once their centers are separated by distance $2 r$ we record these events in terms of the underlying pairwise-distance threshold rather than the radius itself, setting $b _ { i } = 2 r _ { \mathrm { E } }$ and $d _ { i } = 2 r _ { \mathrm { d } }$ . The persistence diagram for a given homology dimension is then the collection of all such pairs $( b _ { i } , d _ { i } )$ observed as � is swept.

The single essential class in $H _ { 0 }$ (the connected component that never dies) has $d _ { i } = \infty$ in the standard construction; we cap its death time once, at the largest finite death value observed across all tracked homology dimensions for that point cloud, before computing any downstream statistics. This ensures every reported quantity, including the entropy, lifetime, Betti-curve, and persistence-image features de scribed below, is finite-valued by construction [29].

The lifetime (or persistence) of feature � is the duration over which it survives, $l _ { i } = d _ { i } - b _ { i }$ ; longlived features represent structure that persists over a broader range of filtration scales, whereas short-lived features can arise from noise or small-scale structure. Normalizing the lifetimes across all features in a diagram gives a probability distribution $p _ { i } = l _ { i } / \sum _ { j } l _ { j }$ which is used below to summarize the distribution of persistence across topological features through its entropy.

In TopTimeNet, each persistence diagram at a fixed homology dimension � is passed through four branches for further feature extraction. We describe these four branches below.

a. Entropy branch. The Shannon entropy of the lifetimes is computed at each homology dimension as

$$
H _ { \mathrm { e n t } } = - \sum _ { i } p _ { i } \log p _ { i } ,\tag{6}
$$

where $p _ { i } = l _ { i } / \sum _ { j } l _ { j }$ is the normalized lifetime associated with feature �. The entropy $H _ { \mathrm { e n t } }$ summarizes how the total persistence is distributed among the topological features [30]. Low $H _ { \mathrm { e n t } }$ indicates that the persistence is concentrated in a small number of dominant features, as may occur for periodic signals. In contrast, high $H _ { \mathrm { e n t } }$ indicates that the persistence is distributed more evenly across multiple features, as may occur for chaotic signals.

b. Lifetime branch. Let $l = ( l _ { 1 } , \ldots , l _ { L } )$ denote the vector of lifetimes at a given homology dimension. We compute five summary statistics from �. The maximum lifetime, max $( l _ { i } )$ , indicates the presence of the most persistent topological feature in the simplicial complex, while the total lifetime, $\textstyle \sum _ { i } l _ { i } ,$ provides a measure of the overall persistence of the topological features.

The dominance ratio, max $_ i ( l _ { i } ) / \sum _ { j } l _ { j }$ , estimates the fraction of the total persistence carried by the most persistent feature. A value close to 1 indicates that one feature dominates, as may occur for clean periodic signals, while a value close to 0 indicates that persistence is spread across many features, none of which dominates, as may occur for chaotic signals.

The coeficient of variation, $\mathrm { s t d } ( l ) / \mathrm { m e a n } ( l )$ , measures the relative spread of lifetimes. Finally, the number ofsignificant lifetimes is defined as the count of features satisfying $l _ { i } > 0 . 0 5 \times \operatorname* { m a x } _ { j } ( l _ { j } )$ [31]. This provides a proxy for the number of prominent topological features by excluding features with lifetimes that are small relative to the maximum lifetime.

c. Betti curve branch. Given the set of birth– death pairs $( b _ { i } , d _ { i } )$ from the persistence diagram at a fixed homology dimension $k ,$ we define the Betti curve, which counts the number of topological features alive at a given Vietoris–Rips filtration threshold $\varepsilon ,$ as

$$
\beta ( \varepsilon ) = \sum _ { i } \mathbb { 1 } ( b _ { i } \leq \varepsilon < d _ { i } ) .\tag{7}
$$

The function $\beta ( \varepsilon )$ is piecewise constant, increasing by 1 at each birth and decreasing by 1 at each death, or by more than 1 when several features share the same birth or death threshold.

To obtain a fixed-dimensional representation that is comparable across point clouds, we sample the Betti curve at a fixed number of filtration thresholds, $n _ { \mathrm { b i n } } = 5 0$ . Because the point clouds have diferent diameters, the thresholds are placed on a diameterscaled grid,

$$
\varepsilon _ { j } = \frac { j } { n _ { \mathrm { b i n } } - 1 } d _ { \mathrm { p t } } , \qquad j = 0 , \dots , n _ { \mathrm { b i n } } - 1 ,\tag{8}
$$

where $d _ { \mathrm { p t } }$ is the diameter of the point cloud. Thus, the sampled filtration locations correspond to normalized positions $\varepsilon _ { j } / d _ { \mathrm { p t } } \in [ 0 , 1 ]$

Given the sampled Betti curve $\begin{array} { r l } { \beta } & { { } = } \end{array}$ $\left( \beta _ { 0 } , \dots , \beta _ { n _ { \mathrm { b i n } } - 1 } \right)$ , we summarize it with six statistics. The maximum, max $\beta _ { j }$ , records the largest number of topological features alive simultaneously at any sampled filtration threshold. The mean, ${ \overline { { \beta } } } _ { ; }$ is the average value of the Betti curve across the sampled thresholds, while the standard deviation, $\operatorname { s t d } ( \beta )$ , summarizes the variation in feature counts across the filtration. The peak position is defined as

$$
\frac { j ^ { * } } { n _ { \mathrm { b i n } } - 1 } , \qquad j ^ { * } = \arg \operatorname* { m a x } _ { j } \beta _ { j } ,\tag{9}
$$

and gives the normalized filtration location at which the sampled Betti curve reaches its maximum. Because Betti curves are piecewise constant, this maximum can be attained on a plateau spanning several consecutive $j ;$ our implementation follows the standard argmax convention of returning the first such index, so $j ^ { * }$ identifies the earliest filtration threshold at which the maximum is reached, rather than, $\mathrm { e . g . }$ , the plateau’s midpoint or last index. The turning-point count is obtained by first taking the discrete diference of the curve, $\Delta \beta _ { j } = \beta _ { j + 1 } - \beta _ { j }$ for $j = 0 , \dots , n _ { \mathrm { b i n } } - 2$ , and then counting the indices $j = 0 , \ldots , n _ { \mathrm { b i n } } - 3$ for which two consecutive diferences $\Delta \beta _ { j }$ and $\Delta \beta _ { j + 1 }$ have opposite signs, i.e. $\Delta \beta _ { j } \cdot \Delta \beta _ { j + 1 } < 0$ . These sign changes identify turning points of the Betti curve and provide a proxy for how oscillatory or multi-modal the curve is. Because Betti curves are piecewise constant, this adjacent sign-change criterion does not detect turning points separated by one or more zero-valued diferences (plateaus); it therefore provides a conservative lower bound on the number of true direction changes in the curve, rather than an exhaustive count.

Finally, the bimodality coeficient,

$$
\mathrm { B C } = \frac { m ^ { 2 } + 1 } { \kappa } ,\tag{10}
$$

provides a skewness–kurtosis-based summary of the distribution of the sampled $\beta$ values. Here � is the skewness (standardized third moment) and � is the standard, non-excess kurtosis (standardized fourth moment) of $\beta .$ Larger values of BC are sometimes interpreted as being compatible with bimodality, but BC is a heuristic rather than a direct test of the number of modes and can also be large for strongly skewed unimodal distributions.

d. Persistence image branch. An alternative representation of the set of birth–death pairs $( b _ { i } , d _ { i } )$ at a fixed homology dimension is obtained by treating them as a kernel-smoothed surface rather than a set of isolated points. Following the persistenceimage construction [18], each pair is first mapped to birth–persistence coordinates, $( b _ { i } , \ell _ { i } )$ with $\ell _ { i } =$ $d _ { i } - b _ { i }$ , and we define a weighted density surface over the birth–persistence plane as

$$
\rho ( x , y ) = \sum _ { i } \frac { w ( \ell _ { i } ) } { 2 \pi \sigma ^ { 2 } } \exp \left( - \frac { ( b _ { i } - x ) ^ { 2 } + ( \ell _ { i } - y ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,\tag{11}
$$

$$
w ( \ell ) = \ell ,\tag{12}
$$

where $\sigma > 0$ is the width of the Gaussian kernel (we set $\sigma = 0 . 0 5 )$ . The weight $w ( \ell )$ increases linearly with lifetime, so that longer-lived topological features contribute more strongly to the surface than short-lived ones.

Each pixel of the resulting $n _ { \mathrm { p i } } \times n _ { \mathrm { p i } }$ grid (we set $n _ { \mathrm { p i } } ~ = ~ 1 5 )$ is assigned the integral of $\rho$ over that pixel’s area, rather than the value of $\rho$ evaluated at the pixel center, yielding the persistence image (PI). Before it can be evaluated, the grid must be calibrated: its birth and persistence bounds are set once from a training set of diagrams at a given homology dimension and then held fixed for all subsequent evaluations, ensuring that no information from validation or test samples enters the bound calibration. This ensures that all persistence images at a given homology dimension are expressed on an identical pixel grid and are therefore directly comparable, rather than each being computed on its own sample-specific range. From each PI we extract seven summary statistics, described below.

The total mass, defined as the sum of all pixel intensities (equivalently, the integral of $\rho$ over the sampled birth–persistence region), measures the aggregate weighted intensity captured within the sampled birth–persistence region; it is large when the diagram contains features with both long lifetimes and non-negligible weight $w ( \ell )$

The max pixel is the highest pixel intensity in the PI, indicating a region of highly concentrated persistence intensity. The fraction of pixels whose intensities exceed the average intensity of their own PI is collected as the active fraction. This statistic records the fraction of the image with intensity above its own mean value.

Viewing the PI as a mass distribution, the birth centroid and persistence centroid are its centers of mass along the birth and persistence axes, respectively. Denoting the pixel intensities as $\mathrm { P I } _ { m , n } .$ , with � and � indexing the birth and persistence axes, respectively, the two centroids are given by

$$
{ \bar { b } } = { \frac { \sum _ { m , n } \operatorname { P I } _ { m , n } x _ { m } } { M } } ,\tag{13}
$$

$$
{ \overline { { \ell } } } = { \frac { \sum _ { m , n } \operatorname { P I } _ { m , n } y _ { n } } { M } } ,\tag{14}
$$

where $\begin{array} { r } { M = \sum _ { m , n } \mathrm { P I } _ { m , n } } \end{array}$ , and $x _ { m }$ and $y _ { n }$ denote the normalized birth and persistence coordinates associated with pixel row � and column $n ,$ respectively. The two centroids indicate where the PI mass is concentrated along the birth and persistence axes.

For $M > 0$ , the probability mass $q _ { m , n } = \mathrm { P I } _ { m , n } / M$ can further be used to compute the normalized en-

tropy,

$$
E = \frac { - \sum _ { m , n } q _ { m , n } \log q _ { m , n } } { \log \left( n _ { \mathrm { p i } } ^ { 2 } \right) } ,\tag{15}
$$

with the convention 0 log $0 ~ = ~ 0$ Since $n _ { \mathrm { p i } } ^ { 2 }$ is the total number of pixels, $E \in [ 0 , 1 ]$ . The entropy measures how spread out or concentrated the normalized pixel intensities are across the two-dimensional im-$\mathrm { a g e } ,$ with $E = 0$ when all mass is concentrated in a single pixel and $E = 1$ for a uniform distribution over all pixels.

Finally, the 90th-percentile intensity is the pixel intensity value below which 90% of pixel intensities fall. Unlike the maximum intensity, it is not determined by a single extreme pixel and therefore provides a less outlier-sensitive summary of the upper end of the pixel-intensity distribution.

When the PI has zero total mass $( M = 0 )$ , as occurs for an empty persistence diagram, the massnormalized quantities and centroids are mathematically undefined. In this case, we define all seven PI summary statistics to be zero by convention, including $E = 0$

When the PI has zero total mass $( M = 0 )$ , as occurs for an empty persistence diagram, the massnormalized quantities and centroids are mathematically undefined. In this case, we define all seven PI summary statistics to be zero by convention, including $E = 0$

An empty persistence diagram at a given homology dimension, which can occur when no topological features are detected, is handled by convention in the other three branches as well: the entropy is defined as $H _ { \mathrm { e n t } } = 0$ , all five lifetime-branch statistics are defined as $0 ,$ and the Betti curve is defined as identically zero across all sampled thresholds, so that the resulting maximum, mean, standard devi ation, peak position, and turning-point count are each 0. The bimodality coeficient is not covered by this convention: since it is computed as a ratio with a numerically stabilized denominator, an identically zero Betti curve instead yields a large finite value rather than 0.

## 2. Learnable fusion and classification

So far, we have extracted five feature groups from the geometric, entropy, lifetime, Betti, and PI branches, all computed deterministically. Since these feature groups arise from diferent geometric and topological constructions, we give each group its own batch-normalization module and its own projection into a shared embedding space of dimension

$D _ { \mathrm { e m b } }$ , rather than sharing a single normalizationand-projection module across all groups. These two steps are carried out by the group projector, yielding five $D _ { \mathrm { e m b } }$ -dimensional embeddings, or tokens, one per feature group.

The five tokens are then combined by a fusion layer, which allows each group to access information from every other group before classification. We consider five interchangeable fusion strategies, differing in their number of parameters and expressivity, each mapping the five $D _ { \mathrm { e m b } }$ -dimensional tokens $\{ \mathrm { t o k e n _ { 1 } , \dots , t o k e n _ { 5 } } \}$ to a single fused representation.

a. Bilinear pairwise fusion. For every pair of groups (�, �) with $i < j ,$ , a scalar interaction

$$
s _ { i j } = \mathrm { t a n h } \left( \sum _ { m = 1 } ^ { D _ { \mathrm { e m b } } } ( \mathrm { t o k e n } _ { i } ) _ { m } ( w _ { i j } ) _ { m } ( \mathrm { t o k e n } _ { j } ) _ { m } \right) ,\tag{16}
$$

is computed from learned weights $w _ { i j } \in \mathbb { R } ^ { D _ { \mathrm { e m b } } }$ and converted into a shared residual $\delta _ { i j } = s _ { i j } v _ { i j }$ , where $v _ { i j } \in \mathbb { R } ^ { D _ { \mathrm { e m b } } }$ is a second learned vector, added back to both token � and token $j .$ Once all pairwise residuals have been accumulated, each token is passed through its own LayerNorm before the resulting tokens are concatenated into the fused representation. This is the cheapest strategy, capturing direct pairwise group interactions without attention or positional structure.

b. Gated residual fusion. For each token $i ,$ the mean of all other tokens,

$$
{ \mathrm { c o n t e x t } } _ { i } = { \mathrm { m e a n } } ( \{ { \mathrm { t o k e n } } _ { j } : j \neq i \} ) ,\tag{17}
$$

is processed through a learned sigmoid gate:

$$
\mathrm { g a t e } _ { i } = \sigma ( W _ { g , i } \mathrm { c o n t e x t } _ { i } ) ,\tag{18}
$$

$$
\mathrm { v a l u e } _ { i } = \mathrm { t a n h } ( W _ { v , i } \mathrm { c o n t e x t } _ { i } ) ,\tag{19}
$$

with $W _ { g , i } , W _ { v , i } \in \mathbb { R } ^ { D _ { \mathrm { e m b } } \times D _ { \mathrm { e m b } } }$ learned per-token weight matrices. The token is then updated as

$$
\mathrm { L a y e r N o r m } \left( \mathrm { t o k e n } _ { i } + \mathrm { g a t e } _ { i } \odot \mathrm { v a l u e } _ { i } \right) .\tag{20}
$$

When the gate approaches zero, the cross-group residual contribution vanishes and the update approaches LayerNorm(token<sub>�</sub>). The five updated tokens are then concatenated into the fused representation.

c. Linear attention fusion. Query–key–value attention among the five tokens is used, with the softmax replaced by the positive feature map

$$
\phi ( x ) = \mathrm { E L U } ( x ) + 1 ,\tag{21}
$$

following the linear-attention formulation of Katharopoulos et al. [32]. Here, ELU denotes the exponential linear unit [33] The feature map $\phi$ is applied independently to each of $n _ { \mathrm { h e a d } }$ attention heads, where $n _ { \mathrm { h e a d } }$ is chosen such that $D _ { \mathrm { e m b } }$ is divisible by $n _ { \mathrm { h e a d } }$

For each query token $q _ { i } .$ , linear attention is computed per head as

$$
\mathrm { A t t n } ( q _ { i } ) = \frac { \phi ( q _ { i } ) ^ { \top } \left( \sum _ { j = 1 } ^ { N _ { \mathrm { t } } } \phi ( k _ { j } ) v _ { j } ^ { \top } \right) } { \phi ( q _ { i } ) ^ { \top } \left( \sum _ { j = 1 } ^ { N _ { \mathrm { t } } } \phi ( k _ { j } ) \right) } ,\tag{22}
$$

rather than using softmax attention.

Here, $q _ { i } , k _ { i } , v _ { i } \in \mathbb { R } ^ { d _ { \mathrm { h } } }$ are the per-head query, key, and value vectors, $N _ { \mathrm { t } } ~ = ~ 5$ is the number of tokens, and $d _ { \mathrm { h } } = D _ { \mathrm { e m b } } / n _ { \mathrm { h e a d } }$ is the per-head dimension. Each layer applies a pre-norm residual attention block, followed by an output projection and a second pre-norm residual feedforward sublayer. After the final layer, the five tokens are concatenated into the fused representation.

d. Low-rank cross-group $M L P .$ All five tokens are concatenated into $ { \boldsymbol { { x } } } ^ { \mathrm { ~ } } \in \mathbb { R } ^ { 5 D _ { \mathrm { e m b } } }$ This vector is then passed through $n _ { \mathrm { l a y e r s } }$ stacked bottleneck blocks, each with bottleneck dimension $n _ { \mathrm { r a n k } }$ and a residual connection. For $l = 1 , \ldots , n _ { \mathrm { l a y e r s } } .$ , each block computes

$$
\begin{array} { r } { \begin{array} { r } { h ^ { ( l ) } = \mathrm { G E L U } \left( W _ { \mathrm { d o w n } } ^ { ( l ) } x ^ { ( l - 1 ) } \right) , } \end{array} } \end{array}\tag{23}
$$

with $W _ { \mathrm { d o w n } } ^ { ( l ) } \in \mathbb { R } ^ { n _ { \mathrm { r a n k } } \times 5 D _ { \mathrm { e m b } } }$ , followed by

$$
\begin{array} { r } { \boldsymbol { x } ^ { ( l ) } = \mathrm { L a y e r N o r m } \left( \boldsymbol { x } ^ { ( l - 1 ) } + \boldsymbol { W } _ { \mathrm { u p } } ^ { ( l ) } \boldsymbol { h } ^ { ( l ) } \right) , } \end{array}\tag{24}
$$

with $W _ { \mathrm { u p } } ^ { ( l ) } ~ \in ~ \mathbb { R } ^ { 5 D _ { \mathrm { e m b } } \times n _ { \mathrm { r a n k } } }$ Here, $x ^ { ( 0 ) } = x$ is the concatenated token vector and $x ^ { ( n _ { \mathrm { l a y e r s } } ) }$ is the final fused representation. Because the bottleneck layers act on the concatenated representation, they allow information from all five feature groups to be mixed while limiting the number of trainable parameters through the dimension $n _ { \mathrm { r a n k } }$ . Because the bottleneck layers act on the concatenated representation, they allow information from all five feature groups to be mixed while limiting the number of trainable parameters through the dimension $n _ { \mathrm { r a n k } } .$ , and this fusion strategy serves as TopTimeNet’s default.

$\begin{array} { r l } { e . \ } & { { } M u l t i - g r o u p } \end{array}$ token attention. The five tokens {token<sub>1</sub>, . . . , token<sub>5</sub>} are stacked into $\boldsymbol { X } \in \mathbb { R } ^ { 5 \times D _ { \mathrm { e m b } } }$ and ofset by a learned positional embedding $P \in$ $\mathbb { R } ^ { 5 \times D _ { \mathrm { e m b } } }$ , so that $X ^ { ( 0 ) } = \mathbf { \bar { \boldsymbol { X } } } + \boldsymbol { P }$ distinguishes group identity independently of token content, since each of the five positions consistently corresponds to the same feature group across all inputs.

The result is passed through a standard pre-norm

Transformer encoder with $n _ { \mathrm { l a y e r s } }$ layers, each applying multi-head self-attention (with head count as a configurable hyperparameter, subject to $D _ { \mathrm { e m b } }$ divisibility) over the five tokens, followed by an output projection and a feedforward sublayer, both with pre-norm residual connections. This is the same block structure described for linear attention fusion, but using standard softmax attention rather than the kernelized formulation. The five output tokens are then concatenated into the fused representation. This strategy introduces standard softmax self-attention across the five feature-group tokens and generally uses more trainable parameters than the simpler fusion schemes considered above.

All five strategies share a common interface, so the choice of fusion strategy is treated as a hyperparameter. The fused representation is passed to a classification head: a single batch-normalization layer applied to the full fused representation, followed by a multilayer perceptron with a configurable number of hidden layers, each followed by an activation function and dropout, which outputs the final class logits. When no hidden layers are used, the batch-normalized fused representation is mapped directly to the class logits by a single linear layer.

## C. Model training and hyperparameter selection

## 1. Training protocol

The learnable components of TopTimeNet are trained jointly with a cross-entropy loss. The optimizer is selected between Adam [34] and AdamW [35]. The optimizer, label smoothing, gradient-clipping norm, and the input-normalization scheme described below are hyperparameters selected independently for each configuration by the random search of Sec. II C 2; the small and large configurations compared in Sec. IV therefore difer in some of these settings, as detailed in Table VI. Training runs for up to 500 epochs, with early stopping using a patience of 50 epochs and restoration of the best validation checkpoint, and the learning rate is reduced on validation-loss plateaus via ReduceLROnPlateau.

Data are split $8 0 / 1 0 / 1 0$ into training, validation, and test sets, stratified by class. This split is performed at the level of individual segments rather than source trajectories, using stratified random shufling. As a result, segments originating from the same simulated trajectory may appear in more than one partition, which may inflate the reported test performance relative to generalization on entirely unseen trajectories. This protocol also does not assess generalization to dynamical systems excluded entirely from training, which would require a system-level split. However, all benchmarked architectures, including TopTimeNet and every baseline reported in Sec. IV, are trained and evaluated on identical segment-level splits under this same procedure; the use of identical splits nevertheless enables a controlled comparison between architectures under the same segment-level evaluation protocol. Trajectory-level and system-level splitting are therefore important directions for future work when the goal is to estimate generalization to unseen trajectories or unseen dynamical systems, respectively.

The raw time series are not rescaled before the geometric and topological features are computed; each simulated system’s signal retains its native amplitude, so that diameter, nearest-neighbor distances, and other scale-dependent quantities reflect the native scale of the underlying simulated variable. An optional normalization stage, selected by the hyperparameter search, may standardize the resulting 42- dimensional feature vector using training-set statistics (mean and standard deviation per feature dimension) before it is passed to the model; this is independent of, and precedes, the per-group batch normalization already applied inside the group projector (Sec. II B 2). Of the two configurations reported in Table VI, the small configuration applies this per-feature standardization, while the large configuration does not.

After training, a single calibration temperature � is fit on the validation set via LBFGS to minimize the cross-entropy of temperature-scaled logits, with � constrained to [0.5, 5.0] during optimization. This aims to improve the alignment between predicted confidence and empirical accuracy without altering the trained weights [36].

## 2. Hyperparameter search

Hyperparameters are selected via random search over 400 independently sampled configurations, spanning architectural choices (embedding dimen sion, fusion strategy, fusion rank, number of attention layers/heads, classifier head width, activation, dropout), regularization strength (label smooth ing, gradient clipping norm), and optimization settings (learning rate, optimizer, input normalization scheme). Batch size and early-stopping patience are both nominally part of the search space but restricted to a single possible value (128 and 50 epochs, respectively) for every trial, and so are, in efect, fixed throughout the search and final eval uation rather than genuinely varied. Sampled configurations that pair an attention-based fusion strategy (linear attention or the multi-group transformer) with an embedding dimension not divisible by the number of attention heads are rejected and resampled, since multi-head attention requires this divisibility. TDA-specific hyperparameters (Takens embedding dimension and delay, number of Betti bins, persistence-image resolution and kernel width) are fixed at the values described in Sec. II B 1 rather than included in the search, and the corresponding TDA features are precomputed and cached once for all trials; since these parameters do not vary across trials, a single cached feature set is valid throughout the search. The input normalization scheme is included as a tunable hyperparameter during the search; the two selected configurations identified below difer in this setting (Sec. II C 1, Table VI). To keep the search computationally tractable, each trial trains a single model instance for up to 500 epochs, rather than the multiple independent runs used for final evaluation (Sec. II C 1), and both post-hoc temperature scaling and the noise-robustness sweep are turned of during search, as neither afects which configuration is selected.

Each trial is scored by a weighted combination of macro F1 and the geometric mean of sensitivity and specificity, score $= { \alpha } \cdot F 1 + ( 1 - \alpha )$ · G-mean, with $\alpha = 0 . 6 $ chosen to balance macro-F1 performance with sensitivity–specificity balance. The CNN and Transformer baselines of Sec. IV A were selected using the same scoring formula and the same randomsearch, validity-rejection, and Pareto-selection procedures described in this subsection, but over their own architecture-specific search spaces, with 250 trials each and $\alpha = 0 . 7 ;$ we note this diference in objective weight explicitly, as it was not matched to TopTimeNet’s own search. Across all 400 trials, we identify the configuration achieving the highest score and select it as the best-performing (“large”) set of hyperparameters for TopTimeNet. We additionally identify a parameter-eficient (“small”) alternative via a Pareto-dominance search on the same (score, parameter-count) trials, using a score tolerance of 1%: a configuration is retained on the Pareto front only if no other trial simultaneously scores within 1% of it while using strictly fewer parameters, and among the retained configurations we report the highest-scoring one as the parameter-eficient candidate. The 1% tolerance allows configurations with similar predictive performance to be compared on parameter count, thereby favoring smaller models when the reduction in score is limited. The selected small and large configurations are each retrained 30 times with independent random seeds, and we report the mean ± standard deviation of the evaluation metrics across these runs.

## III. ILLUSTRATIVE EXAMPLE

Before presenting the results of TopTimeNet, we illustrate how the proposed features help distinguish periodic from chaotic dynamics using the time evolution of a double pendulum.

The double pendulum is a classical system in mechanics. Its time evolution is governed by

$$
\dot { \theta _ { 1 } } = \omega _ { 1 } ,
$$

$$
\dot { \theta _ { 2 } } = \omega _ { 2 } ,\tag{25}
$$

(26)

$$
\dot { \omega _ { 1 } } = \frac { - g ( 2 m _ { 1 } + m _ { 2 } ) \sin ( \theta _ { 1 } ) - m _ { 2 } g \sin ( \theta _ { 1 } - 2 \theta _ { 2 } ) - 2 \sin ( \theta _ { 1 } - \theta _ { 2 } ) m _ { 2 } ( \omega _ { 2 } ^ { 2 } l _ { 2 } + \omega _ { 1 } ^ { 2 } l _ { 1 } \cos ( \theta _ { 1 } - \theta _ { 2 } ) ) } { l _ { 1 } ( 2 m _ { 1 } + m _ { 2 } - m _ { 2 } \cos ( 2 \theta _ { 1 } - 2 \theta _ { 2 } ) ) } ,\tag{27}
$$

$$
\dot { \omega _ { 2 } } = \frac { 2 \sin ( \theta _ { 1 } - \theta _ { 2 } ) ( \omega _ { 1 } ^ { 2 } l _ { 1 } ( m _ { 1 } + m _ { 2 } ) + g ( m _ { 1 } + m _ { 2 } ) \cos ( \theta _ { 1 } ) + \omega _ { 2 } ^ { 2 } l _ { 2 } m _ { 2 } \cos ( \theta _ { 1 } - \theta _ { 2 } ) ) } { \cos \theta } .
$$

$$
l _ { 2 } ( 2 m _ { 1 } + m _ { 2 } - m _ { 2 } \cos ( 2 \theta _ { 1 } - 2 \theta _ { 2 } ) )\tag{28}
$$

Here, the parameters are $g = 9 . 8 1 \mathrm { m / s ^ { 2 } } , \ m _ { 1 } =$ $m _ { 2 } ~ = ~ 1 \mathrm { k g } ,$ , and $l _ { 1 } ~ = ~ l _ { 2 } ~ = ~ 1 \mathrm { m }$ This system exhibits a periodic response for initial conditions $\theta _ { 1 } ( 0 ) = 0 . 4 \mathrm { r a d } , \theta _ { 2 } ( 0 ) = 0 . 6 \mathrm { r a d } , \omega _ { 1 } ( 0 ) = \omega _ { 2 } ( 0 ) =$ $1 . 0 \mathrm { r a d / s } ,$ , and a chaotic response for $\theta _ { 1 } ( 0 ) = 0 . 0 \mathrm { r a d }$ $\theta _ { 2 } ( 0 ) = 3 . 0 \mathrm { r a d } , \omega _ { 1 } ( 0 ) = \omega _ { 2 } ( 0 ) = 0 . 0 \mathrm { r a d / s }$ , using the parameter presets of the Teaspoon library [23].

These two regimes were further characterized by the largest Lyapunov exponent, estimated using the method of Eckmann et al. $[ 3 7 ] \colon \lambda _ { 1 } \approx 0 . 0 1 5$ for the periodic trajectory and $\lambda _ { 1 }$ ≈ 0.106 for the chaotic trajectory. The former value is close to zero, consistent with regular motion up to numerical and finitetime estimation efects, whereas the latter is clearly positive and indicates sensitive dependence on initial conditions.

The time evolution of $\theta _ { 1 }$ is shown in the left panels of Fig. 2, and the corresponding point cloud for each regime, obtained using the Takens embedding described in Sec. II B 1, is shown in the right panels.

![](images/a58451a01fc203204e7489080c36d928bdfb414707f96f59ac45dd7e3a8e63b0.jpg)  
Figure 2. Time-dependent $\theta _ { 1 }$ and its Takens embedding. Left panels present $\theta _ { 1 } ( t )$ for the periodic (top) and chaotic (bottom) regimes. Right panels show the associated point clouds obtained via a Takens embedding with time delay $\tau = 2 0$ and embedding dimension $d _ { \mathrm { e m b } } = 2$

The contrast between the two regimes is already visible at the level of the raw signal and its embedding. In the periodic case, $\theta _ { 1 } ( t )$ shows regular, repeating oscillations, and its Takens embedding forms a clean, closed one-dimensional loop in phase space, consistent with a periodic orbit. In the chaotic case, $\theta _ { 1 } ( t )$ shows irregular oscillations of varying amplitude, and the corresponding point cloud instead fills a broad, difuse region of the two-dimensional embedding space with no discernible loop structure, reflecting the absence of any single dominant periodic orbit.

Table I. Geometric branch feature values for the double pendulum, periodic vs. chaotic regime.
<table><tr><td>Feature</td><td>Periodic Chaotic</td><td></td></tr><tr><td>Diameter</td><td>1.1552</td><td>3.2792</td></tr><tr><td>Correlation-dimension proxy</td><td>1.0672</td><td>1.6585</td></tr><tr><td>Mean nearest-neighbor dist.</td><td>0.0014</td><td>0.0410</td></tr><tr><td>Std. nearest-neighbor dist.</td><td>0.0005</td><td>0.0293</td></tr></table>

The four features computed by the geometric branch of TopTimeNet for each signal are given in Table I, and they capture this contrast quantitatively. The diameter nearly triples between the two regimes (1.1552 → 3.2792), and the mean nearestneighbor distance increases by a factor of approximately 29 and its standard deviation by a factor of approximately 59, consistent with the broader spatial spread of the chaotic point cloud. The correlation-dimension proxy increases as well, from 1.0672 to 1.6585, consistent with the chaotic point cloud having a higher efective dimension than the near-one-dimensional periodic trajectory. Together, these four features already separate the two regimes in this illustrative example using only the geometry of the point cloud, before any topological information is introduced; the topological branches discussed next provide complementary descriptors of the structure of these point clouds.

Table II. Entropy branch feature values for the double pendulum, periodic vs. chaotic regime.
<table><tr><td>Feature</td><td>Periodic Chaotic</td></tr><tr><td>Entropy,  $\overline { { H _ { 0 } } }$ </td><td>5.5456 6.6991</td></tr><tr><td>Entropy,  $H _ { 1 }$ </td><td>0.0379 5.0058</td></tr></table>

![](images/95212723df345fc7eddb8670e2a901feded64d3099768cbeed4c8daf8859e5bf.jpg)

![](images/1202dab70c4a16640513159e75048ee527440732ca767f4b7bd6c48f59cb3f5d.jpg)  
Figure 3. Persistence diagrams for the two tracked homology dimensions. $H _ { 0 }$ (connected components, circles) and $H _ { 1 }$ (loops, triangles) features are shown for the periodic (left) and chaotic (right) point clouds of Fig. 2; note the diferent axis ranges between panels. Points farther from the diagonal correspond to more persistent topological features.

Figure 3 shows the resulting persistence diagrams [38]. In the periodic case, the $H _ { 1 }$ diagram is dominated by a single point far from the diagonal, at a death value close to 1, associated with a sin $\mathrm { g l e } ,$ , highly persistent loop, consistent with the clean closed orbit seen in the point cloud. In the chaotic case, the $H _ { 1 }$ diagram instead contains hundreds of points clustered close to the diagonal, with only a handful reaching moderately large death values; this reflects a large number of short-lived loops rather than one dominant structure, and is captured by the entropy branch (Sec. II B 1), as reflected in Table II. The $H _ { 0 }$ entropy values are less strongly separated between the two regimes, since both point clouds start with the same number of connected components and contain many short-lived finite $H _ { 0 }$ classes (Fig. 3), so $H _ { 0 }$ entropy primarily reflects the distribution of component-merging scales in this example. The $H _ { 1 }$ entropy, by contrast, shows a far more striking diference. Here, the periodic case has a nearzero value (0.038), consistent with the single dominant loop, whereas the value is 5.01 for the chaotic case, reflecting the many scattered points of comparable lifetime seen in the diagram above and yielding a much stronger contrast than for $H _ { 0 }$ entropy.

The third branch with five features is the lifetime branch, whose values for the periodic and chaotic signals are given in Table III. Unlike entropy, which reduces the full lifetime distribution to a single number, these five statistics decompose it along complementary axes, namely, overall scale (maximum and total lifetime), concentration (dominance ratio), spread (coeficient of variation), and count (number of significant lifetimes).

The $H _ { 0 }$ statistics tell a comparatively modest story. Neither regime is dominated by a single $H _ { 0 }$ lifetime, as indicated by the dominance ratios of 0.1947 and 0.0088, consistent with the dense sampling of nearby trajectory points discussed for the entropy branch; the chaotic case nonetheless still stands out through its far larger total lifetime (52.1417 vs. 4.8949) and number of significant lifetimes (836 vs. 1), consistent with a broader spatial spread of the point cloud rather than any single dominant merging event.

At $H _ { 1 }$ , by contrast, the distinction between regimes is stark and consistent across every feature. The periodic signal’s dominance ratio is 0.9959 and its number of significant lifetimes is exactly 1, indicating that nearly all of the persistence is carried by a single loop, with essentially nothing left over for any other feature. In the chaotic signal, the most persistent loop carries only 2.5% of the total persistence (dominance ratio 0.0254), with persistence distributed across 169 significant bars, suggesting contributions from many additional features. The total lifetime grows more than sixfold (0.9386 → 6.0242) despite the maximum lifetime of any single bar actually shrinking $( 0 . 9 3 4 8 \ \to \ 0 . 1 5 3 3 )$ A larger total lifetime together with a smaller maximum lifetime requires substantial contributions from additional features rather than concentration in a single one; the coeficient of variation further indicates a less dispersed relative distribution of lifetimes, as it drops from 5.0775 in the periodic case to 1.0785 in the chaotic case. This is consistent with the entropy and dominance-ratio results: the periodic $H _ { 1 }$ persistence is dominated by a single feature, whereas the chaotic case distributes persistence across many features.

Together, the five lifetime statistics, particularly evaluated at $H _ { 1 }$ , recover the same periodic/chaotic distinction as the entropy branch, but decomposed into interpretable components (how big, how concentrated, how spread out, how many) rather than a single aggregate number, illustrating the complementary role the lifetime branch plays alongside entropy in TopTimeNet’s topological feature set.

Figure 4 presents the corresponding Betti curves. For $H _ { 0 }$ , both regimes show a sharp initial spike as the 980 point-cloud points merge into progressively fewer connected components as the filtration threshold grows, but the chaotic curve decays more gradually, with an intermediate plateau around a normalized filtration threshold � of 0.05–0.1, reflecting a less uniform spatial distribution of points than in the periodic case. The $H _ { 1 }$ curves show a far more striking contrast. The periodic curve is a single clean rectangular plateau at $\beta = 1$ , persisting across most of the normalized filtration range, indicating a single loop that remains alive for nearly the entire filtration sweep. The chaotic curve instead rises sharply to a peak of nearly 50 simultaneously coexisting loops at small normalized filtration thresholds $\varepsilon ,$ before decaying through several irregular steps to near zero by a normalized filtration threshold of approximately 0.15, reflecting many loops appearing and rapidly disappearing early in the filtration.

![](images/926823e6979e079edfe6f2c3d78520b834abebc89e136516205aba500f9301af.jpg)  
Figure 4. Diameter-normalized Betti curves $\beta ( \varepsilon )$ for $H _ { 0 }$ (top) and $H _ { 1 }$ (bottom), periodic (left) and chaotic (right). Note the diferent vertical scales between panels.

The six summary statistics extracted from these curves, given in Table IV, largely mirror this picture. At $H _ { 0 } ,$ the two regimes are nearly indistin guishable. The maximum, peak position, and number of detected turning points are identical between periodic and chaotic, and the mean and standard deviation difer only modestly (20.40 vs. 25.38 and 137.09 vs. 140.74, respectively), consistent with both point clouds containing the same number of densely sampled trajectory points and many short-lived con nected components. The strongest diferences instead occur at $H _ { 1 }$ , where most statistics change substantially between regimes. The maximum jumps from a single coexisting loop (1) to nearly fifty (49), and the standard deviation grows by more than a factor of 17 (0.40 → 7.19). One turning point is detected in the chaotic curve under our adopted cri-

![](images/920b20aed800f60cc63a88879bb1c0e8bf6e50ac9f9d7709a9a46a30df6f7b77.jpg)

Table III. Lifetime branch feature values for the double pendulum, periodic vs. chaotic regime.
<table><tr><td rowspan="2">Feature</td><td colspan="2"> $\overline { { H _ { 0 } } }$ </td><td colspan="2"> $\overline { { H _ { 1 } } }$ </td></tr><tr><td>Periodic</td><td>Chaotic</td><td>Periodic</td><td>Chaotic</td></tr><tr><td>Maximum lifetime</td><td>0.9532</td><td>0.4576</td><td>0.9348</td><td>0.1533</td></tr><tr><td>Total lifetime</td><td>4.8949</td><td>52.1417</td><td>0.9386</td><td>6.0242</td></tr><tr><td>Dominance ratio</td><td>0.1947</td><td>0.0088</td><td>0.9959</td><td>0.0254</td></tr><tr><td>Coefficient of variation</td><td>6.1556</td><td>0.6769</td><td>5.0775</td><td>1.0785</td></tr><tr><td>Number of significant lifetimes</td><td>1</td><td>836</td><td>1</td><td>169</td></tr></table>

![](images/955dd0f05bc30387e02898e3283a2c49a45e2f2daca79852da63c92e971e4e29.jpg)

![](images/d48b9b12a88c6971d3800fde8a4e4e7d990e94196e1db702b7ae36d9b8414781.jpg)

$$
H _ { 1 } ,
$$

![](images/3efe3a2fbe223ea9b144a9f83518456ba1965fac6c9d746d863406c957e53f64.jpg)  
Figure 5. Persistence images for $H _ { 0 }$ (top) and $H _ { 1 }$ (bottom), periodic (left) and chaotic (right). Color indicates pixel intensity (note the diferent color scales between panels); axes are birth and persistence pixel bins.

terion, which counts sign changes between immediately adjacent nonzero first diferences $\Delta \beta _ { j }$ [39], whereas none is detected in the periodic curve under the same criterion.

The bimodality coeficient, computed from the statistical distribution of the sampled curve values $\beta _ { 0 } , \ldots , \beta _ { n _ { \mathrm { b i n } } - 1 }$ rather than from the number of peaks of $\beta ( \varepsilon )$ as a function of filtration scale, changes only mildly between regimes (from 1.00 to 0.92); as discussed above, this coeficient is a skewness–kurtosisbased heuristic and should not be interpreted as a direct count or test of peaks in the Betti curve. A more direct picture of where these loops are concentrated along the filtration range is instead given by the peak position, which shifts from 0.0204 to 0.0408 between regimes.

Finally, Fig. 5 shows the persistence images. The $H _ { 0 }$ images are qualitatively similar between regimes. Both show intensity concentrated at low persistence values near the common $H _ { 0 }$ birth location, since most components merge almost immediately, but the chaotic image’s peak intensity is roughly an order of magnitude higher (0.0125 → 0.1561), reflecting diferences in the persistence distribution and the resulting weighted density. The $H _ { 1 }$ images show the clearest visual distinction. The periodic image shows its mass concentrated toward large persistence values (persistence centroid 0.89) at its corresponding birth location, mirroring the single long-lived loop seen in the persistence diagram and Betti curve. The chaotic image instead shows a visually difuse region of intensity centered at small-to-moderate birth and persistence values (persistence centroid 0.49), with intensity spread more broadly across the grid. Its active fraction is 0.60, compared with 0.30 for the periodic image.

The seven summary statistics extracted from these images, given in Table $^ \mathrm { V , }$ are consistent with this picture. At $H _ { 0 } .$ the total mass and max pixel both increase sharply from periodic to chaotic $( 1 . 2 8 0 1    3 4 . 7 0 8 1 $ and $0 . 0 1 2 5  0 . 1 5 6 1$ , respectively), reflecting the greater aggregate weighted intensity and higher peak intensity of the chaotic image; the active fraction also increases from 0.25 to 0.50.

The birth centroid is fixed at exactly 0.5 in both regimes; this is a consequence of how the degenerate $H _ { 0 }$ birth axis is handled in our implementation rather than a property of the underlying dynamics. Because every $H _ { 0 }$ feature is born at the start of the Vietoris–Rips filtration, all $H _ { 0 }$ birth values are identical, and the fitted birth axis collapses to zero width. Our implementation pads this degenerate axis to a single pixel before resizing to the target grid resolution; the resulting interpolation distributes the single row of birth-axis mass uniformly across all rows of the resized image, which places the birth centroid exactly at the midpoint of the normalized [0, 1] range by symmetry, regardless of regime. The persistence centroid, by contrast, shifts modestly with regime (0.2542 → 0.3124), since it is computed along the (non-degenerate) persistence axis.

At $H _ { 1 }$ , the persistence centroid shows a clear contrast between the two regimes: it sits at 0.8928 for the periodic case, consistent with the mass being concentrated toward large persistence values as seen in the image, versus 0.4860 for the chaotic case, where mass is instead spread toward more moderate persistence values. The birth centroid also shifts noticeably at $H _ { 1 } \ ( 0 . 5 0 0 0  0 . 4 2 6 5 )$ , where, unlike $H _ { 0 }$ , birth times genuinely vary across features, so this shift reflects the chaotic image’s mass being centered at somewhat earlier birth times than the periodic one. Entropy increases from 0.8127 to 0.9925, consistent with the more difuse intensity distribution visible in the chaotic image compared with the narrower band in the periodic case. The active fraction also increases from 0.3000 to 0.6000.

Table IV. Betti curve branch feature values for the double pendulum, periodic vs. chaotic regime.
<table><tr><td rowspan="2">Feature</td><td colspan="2"> $\overline { { H _ { 0 } } }$ </td><td colspan="2"> $H _ { 1 }$ </td></tr><tr><td>Periodic</td><td>Chaotic</td><td>Periodic</td><td>Chaotic</td></tr><tr><td>Max</td><td>980.0000</td><td>980.0000</td><td>1.0000</td><td>49.0000</td></tr><tr><td>Mean</td><td>20.4000</td><td>25.3800</td><td>0.8000</td><td>1.6400</td></tr><tr><td>Standard deviation</td><td>137.0863</td><td>140.7353</td><td>0.4000</td><td>7.1882</td></tr><tr><td>Peak position</td><td>0.0000</td><td>0.0000</td><td>0.0204</td><td>0.0408</td></tr><tr><td>Turning points</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>Bimodality coefficient</td><td>0.9999</td><td>0.9626</td><td>1.0000</td><td>0.9247</td></tr></table>

Table V. Persistence image branch feature values for the double pendulum, periodic vs. chaotic regime.
<table><tr><td rowspan="2">Feature</td><td colspan="2"> $\overline { { H _ { 0 } } }$ </td><td colspan="2"> $\overline { { H _ { 1 } } }$ </td></tr><tr><td>Periodic</td><td>Chaotic</td><td>Periodic</td><td>Chaotic</td></tr><tr><td>Total mass</td><td>1.2801</td><td>34.7081</td><td>0.2390</td><td>5.0567</td></tr><tr><td>Max pixel</td><td>0.0125</td><td>0.1561</td><td>0.0030</td><td>0.0173</td></tr><tr><td>Active fraction</td><td>0.2500</td><td>0.5000</td><td>0.3000</td><td>0.6000</td></tr><tr><td>Birth centroid</td><td>0.5000</td><td>0.5000</td><td>0.5000</td><td>0.4265</td></tr><tr><td>Persistence centroid</td><td>0.2542</td><td>0.3124</td><td>0.8928</td><td>0.4860</td></tr><tr><td>90th-percentile intensity</td><td>0.0102</td><td>0.1531</td><td>0.0023</td><td>0.0167</td></tr><tr><td>Entropy</td><td>0.8895</td><td>0.9610</td><td>0.8127</td><td>0.9925</td></tr></table>

Taken together, this illustrative example shows how the five branches provide complementary descriptions of the contrast between periodic and chaotic dynamics. The geometric branch shows that the chaotic point cloud occupies a larger, less uniformly sampled region of phase space; the entropy branch shows that its topological features are less dominated by a single persistent structure; the lifetime branch decomposes this same distinction into interpretable measures of scale, concentration, and count; the Betti curve branch reveals the same contrast unfolding across the filtration itself, from a sin gle stable plateau to a sharp, jagged, rapidly decaying peak; and the persistence image branch localizes where in the birth–persistence plane this diference occurs.

In this example, the $H _ { 1 }$ features show the clearest topological contrast between the two regimes, although several $H _ { 0 }$ features also difer substantially, highlighting the complementary information provided by connectivity- and loop-level topology when characterizing periodic versus chaotic dynamics.

Several individual features already distinguish the two trajectories in this illustrative example, but the fact that the distinction appears across geometry, entropy, lifetime statistics, Betti curves, and persistence images motivates combining all 42 features rather than relying on any single representation. Appendix A extends this same double-pendulum example to illustrate how raw-signal noise disrupts these features, complementing the aggregate raw-signal robustness results of Sec. IV B.

## IV. RESULTS

We evaluate TopTimeNet on the full extendedteaspoon benchmark described in Sec. II A, spanning 49 nonlinear dynamical systems simulated in both periodic and chaotic regimes.

Before comparing against learned-representation baselines, we first ask a more basic question: given that the geometric and topological featureextraction stage of TopTimeNet is deterministic and contains no learned weights (Sec. II B 1), how much learnable capacity does the remaining dis criminative stage actually need? Table VI compares two TopTimeNet configurations selected by the hyperparameter search of Sec. II C 2: the globally best-scoring $\left( \mathrm { ^ { 6 6 } l a r g e ^ { 9 7 } } \right)$ configuration, with 54,886 trainable parameters, and the parametereficient (“small”) configuration identified via the 1%-tolerance Pareto search described there, with 1,638 parameters, approximately 33× fewer. Despite this diference in trainable capacity, the two models achieve nearly identical mean test accuracy $( 9 7 . 5 8 \% \pm 0 . 2 9 \% \ \mathrm { v s . \ 9 7 . 5 8 \% \pm 0 . 2 2 \% }$ , over 30 independent training runs each). The two configurations also yield very similar F1, G-mean, precision, and recall values. The large model trained for the full 500-epoch budget in every one of its 30 runs, never triggering early stopping, whereas the small model stopped early at an average of epoch 306 (range 140– 500). Thus, under the adopted early-stopping criterion, training terminated earlier on average for the small configuration. The small configuration therefore attains essentially the same mean predictive performance as the large configuration while using substantially fewer trainable parameters and terminating training earlier on average.

Table VI. Comparison of the small and large TopTimeNet configurations selected by the hyperparameter search of Sec. II C 2. All test metrics are mean $\pm \mathrm { \ s t d }$ over 30 independent training runs. Both configurations use bilinear fusion (Sec. II B 2), which has no rank or attention hyperparameters; the rank/attention-layer/attention-head values sampled elsewhere in the search do not apply to either selected configuration.
<table><tr><td></td><td>Small model</td><td>Large model</td></tr><tr><td>Trainable parameters</td><td>1,638</td><td>54,886</td></tr><tr><td>Embedding dim. D</td><td>16</td><td>128</td></tr><tr><td>Fusion strategy</td><td>Bilinear</td><td>Bilinear</td></tr><tr><td>Classifier head</td><td>()</td><td>(64, 32, 16)</td></tr><tr><td>Activation</td><td>GELU</td><td>Leaky ReLU</td></tr><tr><td>Dropout</td><td>0.05</td><td>0.0</td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W }$ </td><td>Adam</td></tr><tr><td>Learning rate</td><td> $7 . 1 2 \times 1 0 ^ { - 4 }$ </td><td> $1 . 5 2 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Label smoothing</td><td>0.1</td><td>0.0</td></tr><tr><td>Gradient clip norm</td><td>5.0</td><td>5.0</td></tr><tr><td>Input normalization</td><td> $\mathrm { p e r - c h a n n e l }$ </td><td>none</td></tr><tr><td>Accuracy</td><td> $\overline { { 0 . 9 7 5 8 \pm 0 . 0 0 2 9 } }$ </td><td> $\overline { { 0 . 9 7 5 8 \pm 0 . 0 0 2 2 } }$ </td></tr><tr><td>F1 score</td><td> $0 . 9 7 5 8 \pm 0 . 0 0 2 9$ </td><td> $0 . 9 7 5 7 \pm 0 . 0 0 2 2$ </td></tr><tr><td>G-mean</td><td> $0 . 9 7 5 7 \pm 0 . 0 0 2 9$ </td><td> $0 . 9 7 5 6 \pm 0 . 0 0 2 2$ </td></tr><tr><td>Precision</td><td> $0 . 9 7 5 8 \pm 0 . 0 0 2 9$ </td><td> $0 . 9 7 5 7 \pm 0 . 0 0 2 2$ </td></tr><tr><td>Recall</td><td> $0 . 9 7 5 8 \pm 0 . 0 0 2 9$ </td><td> $0 . 9 7 5 9 \pm 0 . 0 0 2 2$ </td></tr><tr><td>Mean epochs trained</td><td> $\overline { { 3 0 5 . 5 \pm 9 3 . 8 } }$ </td><td> $\overline { { 5 0 0 . 0 } }$ </td></tr><tr><td>Mean training time (s)</td><td> $4 3 3 . 2 \pm 1 3 2 . 5$ </td><td> $8 7 3 . 9 \pm 1 1 . 5$ </td></tr></table>

robustness evaluation of Sec. IV B.

Under the present evaluation protocol, increasing the trainable parameter count from 1,638 to 54,886 does not improve the mean test metrics. This result is consistent with the fixed geometric and topological features already providing a representation from which the periodic and chaotic classes can be discriminated efectively with a comparatively compact learnable stage. The comparison does not, however, establish linear separability of the feature represen tation or a general capacity threshold beyond which additional parameters cannot be beneficial. Guided by this result, we adopt the smaller, 1,638-parameter configuration for all subsequent analyses in this section, including the comparison against convolutional and transformer-based baselines in Sec. IV A and the

## A. Comparison against convolutional and transformer-based baselines

To benchmark TopTimeNet’s performance, we compare it against two learned-representation base lines trained directly on the raw segmented time series: a 1D convolutional neural network (CNN) and a Transformer encoder. Both baselines were selected using the same random-search strategy, trial validity rules, and Pareto-based final-selection procedure described in Sec. II C 2, but over their own architecture-appropriate search spaces rather than TopTimeNet’s, and with 250 sampled trials and an objective weight of $\alpha = 0 . 7$ each, versus 400 trials and $\alpha = 0 . 6$ for TopTimeNet itself; early-stopping patience was fixed at 50 epochs for all three architectures, in both the search and the final evaluation reported here. Both baselines are evaluated over 10 independent training runs each, mirroring the repeated-run protocol used for TopTimeNet in Table VI (at reduced scale, given the substantially higher per-run training cost of both baselines).

Table VII summarizes the CNN and Transformer configurations found by the search, alongside the small TopTimeNet configuration from Table VI. The CNN baseline uses a channel-multiplier architecture (base channels 64, multipliers [1, 2, 4, 8], max pooling) trained with Muon [40]; the Transformer baseline uses a 5-layer encoder (embedding dimension 128, 8 attention heads, feedforward dimension 256) with a convolutional input embedding, trained with AdamW [35].

Table VII. TopTimeNet (small configuration) versus CNN and Transformer baselines. TopTimeNet and CNN report mean ± std over 30 and 10 independent training runs, respectively, with no failed or collapsed runs in either case. For the Transformer, 10 independent training attempts were performed; the reported mean ± std values are computed over the 7 runs that converged, while the remaining 3 attempts remained at a near-chance validation-accuracy plateau and are reported as non-converged (see text).
<table><tr><td></td><td>TopTimeNet (small)</td><td> $\overline { { \mathbf { C N N } \mathbf { \Lambda } ( n = 1 0 ) } }$ </td><td>Transformer  $\overline { { ( n = 7 ) } }$ </td></tr><tr><td>Trainable parameters</td><td>1,638</td><td>1,824,898</td><td> $\overline { { 3 3 , 4 3 5 , 5 7 0 } }$ </td></tr><tr><td>Accuracy</td><td> $0 . 9 7 5 8 \pm 0 . 0 0 2 9$ </td><td> $0 . 9 7 0 8 \pm 0 . 0 1 1 0$ </td><td> $0 . 9 4 1 2 \pm 0 . 0 1 3 3$ </td></tr><tr><td>F1 score</td><td> $0 . 9 7 5 8 \pm 0 . 0 0 2 9$ </td><td> $0 . 9 7 0 8 \pm 0 . 0 1 1 0$ </td><td> $0 . 9 4 1 2 \pm 0 . 0 1 3 3$ </td></tr><tr><td>G-mean</td><td> $0 . 9 7 5 7 \pm 0 . 0 0 2 9$ </td><td> $0 . 9 7 0 8 \pm 0 . 0 1 1 1$ </td><td> $0 . 9 4 1 1 \pm 0 . 0 1 3 3$ </td></tr><tr><td>Precision</td><td> $0 . 9 7 5 8 \pm 0 . 0 0 2 9$ </td><td> $0 . 9 7 0 8 \pm 0 . 0 1 1 0$ </td><td> $0 . 9 4 1 2 \pm 0 . 0 1 3 3$ </td></tr><tr><td>Recall</td><td> $0 . 9 7 5 8 \pm 0 . 0 0 2 9$ </td><td> $0 . 9 7 1 0 \pm 0 . 0 1 0 9$ </td><td> $0 . 9 4 1 6 \pm 0 . 0 1 3 2$ </td></tr><tr><td>Epochs trained</td><td> $3 0 5 . 5 \pm 9 3 . 8$ </td><td> $1 0 3 . 4 \pm 2 3 . 1$ </td><td> $1 4 4 . 3 \pm 6 9 . 2$ </td></tr></table>

Across 10 independent runs, the CNN baseline achieves a mean accuracy of 97.08% ± 1.10%, close to TopTimeNet’s $9 7 . 5 8 \% \pm 0 . 3 0 \%$ . All 10 CNN runs achieve accuracies between 94.8% and 98.0%, with no non-converged run observed. The Transformer baseline, in contrast, shows greater run-to-run instability under the tested training protocol: of its 10 independent training attempts, 3 never escaped a near-chance-accuracy plateau (validation accuracy fluctuating around 50% to 51% from early in training onward) and were terminated by early stopping once validation performance failed to improve for the configured patience window, well before the 500-epoch budget was reached.

We classify these 3 attempts as non-converged based on their validation trajectories and report the performance statistics in Table VII over the remaining 7 runs. The three non-converged attempts reached final validation accuracies of 51.4%, 51.3%, and 54.2%, each with F1 and G-mean scores substantially below what their accuracy alone would suggest, consistent with strongly imbalanced class predictions. The remaining, converged runs achieve a mean accuracy of 94.12% ± 1.33%. Thus, 3 of the 10 Transformer training attempts did not converge under the tested protocol, whereas no non-converged runs were observed among the 30 TopTimeNet runs or the 10 CNN runs.

The Transformer’s converged runs stop, on average, after fewer epochs than TopTimeNet (144.3 ± 69.2 vs. 305.5 ± 93.8), but after more epochs than the CNN (144.3 ± 69.2 vs. $1 0 3 . 4 \pm 2 3 . 1 )$ .

Precision, recall, F1, and accuracy are numerically similar for the CNN and TopTimeNet, and the converged Transformer runs show the same qualitative pattern, with recall $( 9 4 . 1 6 \% \pm 1 . 3 2 \% )$ close to the corresponding accuracy and precision.

The parameter counts also show a substantial difference in trainable model size. The CNN baseline requires 1,824,898 trainable parameters, over 1,100× more than TopTimeNet’s 1,638, while achieving a similar mean accuracy; no non-converged runs were observed among its 10 training attempts.

The Transformer baseline requires 33,435,570 parameters (over 20,000× TopTimeNet’s parameter count and roughly 18× the CNN’s) while achieving a lower mean accuracy than either alternative over its 7 converged runs, with 3 of its 10 training attempts classified as non-converged.

This is the central eficiency argument of this paper in concrete terms: under the tested protocol, the small TopTimeNet configuration achieves mean accuracy comparable to the CNN and higher than the mean of the converged Transformer runs while using three to four orders of magnitude fewer train able parameters. For the Transformer specifically, 3 of the 10 training attempts did not converge. These results are consistent with the small-versuslarge TopTimeNet comparison in Sec. IV, in which increasing the trainable parameter count did not improve the mean test metrics.

## B. Robustness evaluation

Having established that the small, 1,638- parameter TopTimeNet configuration achieves mean clean-data accuracy comparable to the CNN and higher than the mean of the converged Transformer runs, we now evaluate all three models robustness to Gaussian noise. For TopTimeNet, we distinguish two points at which noise can be introduced. In the feature-level sweep, for each of the 30 trained instances of the small model, we inject zero-mean Gaussian noise of standard deviation � directly into the held-out test set’s precomputed 42-dimensional feature vectors, reapply the same interquartile-range clipping bounds fit during feature precomputation, and evaluate the trained classifier on the resulting corrupted features. Because features are cached and reused across training runs, this sweep probes only the robustness of the learnable stage to perturbations of the precomputed geometric and topological summary features; it does not exercise the Takens embedding or persistent homology computation under noise, and so cannot by itself establish the noise-sensitivity of the complete raw-signal-to-prediction pipeline.

(a)  
![](images/0c41a3d1b1eff6699b92b827a6201e02f31cea3bb321082d0f0661d6ee242c08.jpg)

(b)  
![](images/014ea0457f3c281559a6642d15d05e69adc0c70243dbfa5a2dcd1275a3c0b770.jpg)  
Figure 6. Noise robustness of all three models. (a) Test accuracy and (b) Expected Calibration Error (ECE, $M = 1 0$ equal-width bins), both as a function of the Gaussian noise standard deviation �. For TopTimeNet, noise is injected at two distinct points: directly into the precomputed 42-dimensional feature vector of Sec. II B 1 (“feature-level”), and into the raw time series itself, with the entire feature-extraction pipeline, Takens embedding, persistent homology, and all five feature branches, recomputed from the corrupted signal (“raw-signal”). The CNN and Transformer baselines have no intermediate feature representation to perturb separately, so only their raw-signal curves are shown. Thick lines show the mean over independent training runs $( n = 3 0$ for TopTimeNet, $n = 1 0$ for the CNN, $n = 7$ for the Transformer’s converged runs), shaded bands show ±1 standard deviation, and thin lines show individual runs.

In the raw-signal sweep, we instead add Gaussian noise of the same standard deviation � directly to the held-out test segments themselves, and recompute the full 42-dimensional feature vector from the noisy signal, including a fresh Vietoris–Rips persistent homology computation, before evaluating the same trained classifier, with the training-setcalibrated persistence-image grid held fixed across all noise levels. The CNN and Transformer baselines have no analogous feature-level regime, since they consume the raw time series directly with no ana ogous precomputed feature representationYou to perturb separately; noise injected into either baseline is therefore comparable in injection point to TopTimeNet’s raw-signal sweep.

All sweeps use the same numerical noise grid $( \sigma \in$ {0, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0}), applied as an absolute Gaussian noise standard deviation added directly to the raw signal or feature vector, rather than a signal-normalized noise level. Because the raw time series retain their native, system-specific amplitudes (Sec. II C 1), a given numerical � can correspond to substantially diferent noise levels relative to each system’s or variable’s own scale, as il lustrated concretely for one example in Appendix A. For TopTimeNet, however, feature-level and rawsignal noise act in diferent spaces, so equal numerical values of � should not be interpreted as equal normalized perturbation strengths in the two sweeps. As illustrated concretely in Appendix A, the same nominal � can correspond to substantially diferent relative noise levels even between the two regimes of a single example: at $\sigma = 0 . 1$ , the injected noise there amounts to 25% and 15% of the periodic and chaotic signal’s own standard deviation, respectively, while at $\sigma = 0 . 5$ and $\sigma = 1 . 0$ it reaches 73%– 127% and 147%–255%. The higher end of this noise grid is therefore not a subtle perturbation in either regime of this worked example: at $\sigma \geq 0 . 5$ , the injected noise is comparable to or exceeds the clean signal’s own scale, so these high-� points should not be interpreted as a uniform test of small-noise robustness across the benchmark.

The smallest tested noise levels are therefore more informative for assessing sensitivity to modest absolute perturbations, although their magnitude relative to the underlying signal still varies across systems and variables. The same independent training runs already reported in Table VI and Table VII are used throughout (30 for TopTimeNet, 10 for the CNN, and 7 converged runs for the Transformer), so comparisons across noise levels do not involve diferent sets of trained instances within each model.

Only TopTimeNet receives training-time noise augmentation, applied at the feature level $( \sigma _ { \mathrm { a u g } } =$ 0.05, doubling the efective training-set size), so it exposes the model only to noise in the 42- dimensional feature space during training, never to noise in the raw signal.

The CNN and Transformer baselines, by contrast, receive no training-time noise augmentation of any kind; their raw-signal robustness sweep therefore evaluates a model trained exclusively on clean data, whereas TopTimeNet’s feature-level sweep evaluates a model that was specifically trained to tolerate the kind of perturbation being tested.

Predicted-class confidences for all three models are calibrated post-hoc via a temperature � fit on each run’s clean validation set (mean $T = 0 . 5 4 5 \pm$ 0.024 across TopTimeNet’s 30 runs) and reused unchanged across all noise levels for that run, so the calibration mapping itself is not refit as the noise level changes.

Fig. 6 reports the resulting accuracy (panel (a)) and ECE (panel (b)) curves. At $\sigma = 0 ,$ , all curves recover each model’s clean-data accuracy from Tables VI and VII. TopTimeNet’s feature-level curve is the only one of the four that degrades gracefully: accuracy remains above 97% through $\sigma = 0 . 1$ $( 9 7 . 0 2 \% \pm \ : 0 . 2 5 \% )$ above 95% at $\sigma ~ = ~ 0 . 2$ and only falls substantially at the highest tested noise levels, reaching $8 9 . 5 5 \% \pm 1 . 1 8 \%$ at $\sigma ~ = ~ 0 . 5$ and $8 2 . 5 8 \% \pm 1 . 9 4 \%$ at $\sigma = 1 . 0 $ All three raw-signal curves, by contrast, degrade sharply within the first few tested noise levels. TopTimeNet’s own rawsignal accuracy falls to 71.27% ± 1.47% already at $\sigma = 0 . 0 2 5$ , a drop of over 26 percentage points from its clean-data value, and plateaus near chance level $( 5 1 . 5 1 \% \pm 0 . 1 2 \% )$ by $\sigma \ = \ 1 . 0$ The CNN shows the largest initial drop among the three raw-signal curves: clean accuracy of $9 7 . 0 8 \% \pm 1 . 1 0 \%$ falls to 57.48% ± 2.23% at $\sigma = 0 . 0 2 5$ and reaches $5 0 . 1 0 \% \pm$ 0.22% by $\sigma \ = \ 1 . 0 .$ The Transformer’s converged runs show the same qualitative collapse but from a lower starting point and a comparatively smaller initial drop $( 9 4 . 1 2 \% \pm 1 . 3 3 \%$ to $8 0 . 8 8 \% \pm 3 . 3 8 \%$ at $\sigma ~ = ~ 0 . 0 2 5 )$ , reaching a similar chance-level floor $( 5 1 . 4 5 \% \pm 0 . 8 2 \% )$ ) by $\sigma = 1 . 0$ . Thus, all three tested pipelines show substantial sensitivity to raw-signal Gaussian perturbations under the present protocol, despite their diferent architectures and parameter counts.

Panel (b) shows the corresponding contrast in calibration. TopTimeNet’s feature-level ECE increases from 0.69% at $\sigma = 0$ to $1 0 . 0 2 \% \pm 1 . 5 4 \%$ at $\sigma ~ = ~ 1 . 0$ All three raw-signal curves climb much more steeply: TopTimeNet’s own raw-signal ECE jumps to $2 3 . 8 6 \% \pm 1 . 7 1 \%$ at $\sigma = 0 . 0 2 5$ and reaches $4 7 . 5 4 \% \pm 0 . 4 5 \%$ by $\sigma ~ = ~ 1 . 0 ;$ the CNN reaches $4 1 . 6 5 \% \pm 2 . 3 5 \%$ already at $\sigma = 0 . 0 2 5$ and $4 9 . 4 2 \% \pm 1 . 6 5 \%$ by $\sigma = 1 . 0 ;$ and the Transformer’s converged runs reach $4 4 . 0 1 \% \pm 5 . 8 6 \%$ by $\sigma = 1 . 0$ . At the highest raw-signal noise levels, all three models therefore combine near-chance accuracy with large ECE values, indicating substantial degradation in the calibration of their predicted probabilities.

For TopTimeNet, this contrast shows that robustness to perturbations of the precomputed feature vector does not imply robustness when noise is introduced in the raw signal and the features are recomputed. Because the feature-level and rawsignal sweeps perturb diferent spaces, and because TopTimeNet is trained with feature-level but not raw-signal noise augmentation, the present experiment does not isolate the source of the diference between these two robustness curves. Independent of any efect of training-time augmentation, Appendix A shows directly, via a single worked exam ple, that the delay-embedded point cloud and its persistence diagram are themselves visibly disrupted by raw-signal noise at these same � levels, before any classifier is involved: raw-signal noise can disrupt the representation itself, independent of the classifier, though this one example does not indicate what fraction of the aggregate, benchmark-level degradation this mechanism accounts for.

For deployment, these results emphasize that robustness to perturbations of precomputed features should not be taken as evidence of robustness to noise present at acquisition time. Such noise should instead be evaluated by propagating the corrupted signal through the complete signal to-prediction pipeline. See Appendix A for a singleexample illustration of this mechanism.

## V. CONCLUSION

We introduced TopTimeNet, a time-series classifi cation architecture that separates fixed feature construction from discrimination: five families of features derived from Takens delay embeddings and persistent homology, geometric, entropy, lifetime, Betti curve, and persistence image statistics, are computed deterministically and without any trainable parameters, leaving only a lightweight learnable stage to solve the resulting classification task. Using the double pendulum as an illustrative example, we showed how the five branches provide complementary descriptions of the contrast between periodic and chaotic trajectories, and showed on the nonlinear-system benchmark described in Sec. II A that a 1,638-parameter classifier achieves the same mean accuracy as a configuration with 33× more trainable parameters, and achieves mean accuracy comparable to the CNN and higher than the mean of the converged Transformer runs, while using three to four orders of magnitude fewer trainable parameters.

Reliability across repeated training runs varied by baseline: no non-converged runs were observed among the 30 TopTimeNet runs or the 10 CNN runs, whereas 3 of the 10 Transformer training attempts did not converge under the tested protocol.

Our noise experiments distinguish robustness to perturbations of precomputed features from robustness of the complete raw-signal-to-prediction pipeline. Perturbing the precomputed feature vectors directly, TopTimeNet degrades gracefully, retaining most of its accuracy across a wide range of injected noise. Perturbing the raw time series instead, and recomputing the full delay-embedding-topersistent-homology pipeline on the corrupted signal, produces a markedly diferent outcome: accuracy degrades sharply at the smallest tested absolute noise levels, although the magnitude of a given � relative to the underlying signal varies across systems and variables (Appendix A).

The CNN and Transformer baselines, which have no analogous precomputed feature representation to perturb separately, show the same qualitative collapse under raw-signal noise. Thus, all three tested pipelines are sensitive to raw-signal Gaussian perturbations under the present protocol, despite their diferent architectures and parameter counts.

For TopTimeNet, robustness to perturbations of the precomputed feature vector therefore does not imply robustness when noise is introduced before feature extraction. Because the two experiments perturb diferent spaces, and because feature-level but not raw-signal noise is used for TopTimeNet’s training augmentation, the present experiments do not isolate the source of the diference between the two robustness curves.

These observations do not conflict with standard stability results for Vietoris–Rips persistence, which control changes in persistence diagrams under suitable perturbations of the underlying metric space [41]. Such results do not guarantee stability of the complete pipeline, including delay embedding, finite-sample summary statistics, preprocessing, and classification. In particular, the Gaussian perturbations considered here are added to the observed time series after trajectory generation and should not be interpreted as perturbations of the initial conditions whose efects are subsequently amplified by chaotic dynamics. Robustness to noise present at the point of raw-signal acquisition therefore cannot be assumed to follow automatically from persistent homology’s stability properties, and should instead be established empirically, as we have done here.

Appendix A illustrates one contributing mechanism concretely: for a single worked example, the periodic regime’s topological signature, a single domi nant, long-lived loop, is more easily destroyed by ob servational noise than the chaotic regime’s already difuse signature, consistent with both the periodic signal’s smaller natural amplitude (making a given noise level proportionally larger) and its simpler, lower-entropy structure. We emphasize that this single-example asymmetry illustrates a candidate mechanism rather than a general claim that periodic dynamics are always more fragile than chaotic dynamics to observational noise.

The present study is deliberately scoped to a bi nary distinction between periodic and chaotic dy namics. Several directions follow naturally from this scope. Most immediately, the same feature set could plausibly extend to distinguishing a broader range of dynamical regimes beyond the periodic/chaotic dichotomy considered here. Quasi-periodic and stochastic dynamics provide natural additional test cases because their reconstructed trajectories may exhibit geometric and topological structure diferent from the periodic and chaotic examples consid ered here. Whether the present fixed embedding and feature set can separate these regimes reliably remains to be established. Testing TopTimeNet on a labeled multi-class benchmark spanning periodic, quasi-periodic, chaotic, and stochastic regimes would be a natural way to evaluate these hypotheses.

A second direction concerns quantum dynamics. Topological data analysis has been applied to quantum dynamics in several contexts, including regular–chaotic discrimination and persistent homology-based monitoring of finite-time quantum engines [42, 43], and we hypothesize that the same TDA-based feature branches used here for classical dynamical systems could be similarly informative for classifying periodic versus chaotic quantum dynamics, for instance from time series of expectation val ues, wavefunction overlaps, or other quantum ob servables whose reconstructed phase-space embed dings might exhibit analogous topological signatures of regularity and chaos. We view this as a promising direction for testing the present framework, rather than an outcome we take for granted.

More broadly, the underlying idea of this work, replacing a portion of a model’s learned representation with a fixed, domain-informed feature extractor and allowing a small number of trainable parameters to focus solely on the discriminative task, is not specific to periodic/chaotic classification, or even to timeseries data. The same representation/discrimination decoupling could be tested in other learning tasks for which informative domain-based feature representations are available. Testing this hypothesis in other domains, and characterizing where the resulting efficiency benefits and robustness behavior do and do not carry over, remains an open and, we believe, worthwhile direction for future work.

## VI. CODE AVAILABILITY STATEMENT

The code that supports the findings of this article, along with the scripts and instructions needed to regenerate the dataset used in our experiments, is openly available in the TopTimeNet GitHub repository.

## VII. AUTHOR CONTRIBUTIONS

S. S. conceived the project, implemented the algorithm, analyzed the data and prepared the initial draft of the manuscript. S. B. contributed to the analysis of the results and together with S. S. finalized the manuscript.

## VIII. ACKNOWLEDGMENT

S. S. thanks B. Krishnamoorthy for helpful communications at early statges of this project. We acknowledge the use of the following open-source packages in the preparation of our code and analysis: Teaspoon [23], Ripser.py [44], Scikit-TDA [45], and GUDHI [46].

## Appendix A: Illustrative raw-signal noise sweep on the double pendulum

Sec. IV B quantifies the raw-signal noise sweep in aggregate, averaged over 30 independently trained classifiers across all 49 systems in the benchmark. Here we illustrate the same sweep on a single example, the double pendulum, to give visual intuition for the underlying mechanism. Using the featureextraction pipeline of Sec. III, we add zero-mean Gaussian noise of standard deviation � directly to one periodic and one chaotic double-pendulum segment, and recompute the full pipeline, Takens em bedding, persistent homology, and all five feature branches, from the corrupted signal at each of five representative noise levels, $\sigma \in \{ 0 , 0 . 1 , 0 . 2 , 0 . 5 , 1 . 0 \}$

Because � is an absolute noise standard deviation while the paper’s SNR convention (Sec. IV B) is defined relative to a unit reference power, it is worth stating what these nominal � values represent relative to this specific example’s own signal amplitude. We define the relative noise level of a given $\sigma _ { \mathrm { { : } } }$ , for a clean signal $x ( t )$ , as $r ( \sigma ) = \sigma / \mathrm { s t d } ( x )$ , the ratio of the injected noise’s own standard deviation to the clean signal’s standard deviation; $r ( \sigma ) = 1$ (i.e., 100%) corresponds to noise exactly as large, in this sense, as the signal itself. The clean periodic segment has std $\left( x \right) = 0 . 3 9$ and the clean chaotic segment has $\mathrm { s t d } ( x ) = 0 . 6 8 ;$ across the four noise levels shown, $r ( \sigma )$ is 25%/15% at $\sigma = 0 . 1$ , 51%/29% at $\sigma = 0 . 2$ , 127%/73% at $\sigma = 0 . 5$ , and 255%/147% at $\sigma = 1 . 0$ (periodic/chaotic, respectively). The injected noise therefore stays below the signal’s own scale for both regimes only at $\sigma = 0 . 1 \mathrm { : }$ by $\sigma = 0 . 5$ it already exceeds the periodic signal’s own scale and approaches the chaotic signal’s, and at $\sigma = 1 . 0$ it exceeds both, but considerably more so for periodic than for chaotic throughout.

Fig. 7 shows the corrupted raw signals themselves. At $\sigma = 0 . 1 ~ ( r ( \sigma ) \leq 2 5 \%$ for both regimes), the underlying oscillatory structure remains clearly visible to the eye in both signals, with only a modest increase in high-frequency texture; at $\sigma \ : = \ : 0 . 2$ this texture is more pronounced but the oscillation is still readily apparent. Only at $\sigma = 0 . 5$ and above, where the injected noise is comparable to or exceeds the signal’s own scale, does the raw signal begin to look obviously noisy, with the periodic signal’s regular oscillation becoming visually dificult to discern by $\sigma = 1 . 0$ . This is precisely the point: the downstream topological representation degrades well before the raw signal appears corrupted, and well before the noise is large relative to the signal itself.

Fig. 8 shows the resulting point clouds. At $\sigma = 0 ,$ the periodic trajectory embeds as a clean, thin ring, a simple limit cycle. At $\sigma = 0 . 1$ , this ring has visibly thickened but a central hole, the signature of a genuine loop, remains discernible; by $\sigma = 0 . 2$ that hole has efectively closed, and the point cloud has already collapsed into a difuse blob with no remaining loop structure, a state essentially unchanged through $\sigma = 0 . 5$ and $\sigma = 1 . 0$ . The chaotic trajectory’s point cloud, by contrast, is already a difuse, space-filling scatter at $\sigma = 0$ , so it has comparatively less clean structure to lose, and its qualitative appearance changes far less over the same noise range. Part of this asymmetry is attributable to the difering relative noise levels noted above (the same nominal � is a proportionally larger perturbation for the periodic signal, given its smaller natural amplitude), though the periodic point cloud’s simpler, lower-dimensional structure, a thin ring rather than a space-filling scatter, plausibly also makes it more visually sensitive to a comparable relative perturbation.

chaotic, = 0.5  
chaotic, = 0.0  
x(t)  
x(t)  
x(t)  
x(t)  
![](images/d5cc45cb110ae002ad74d909797af7587729258e39445763e3edee080212756a.jpg)

![](images/a2fabaa6a0754b6cdd23bf2da7171a742d232e609658d22079e3e15e38c5a6e0.jpg)

![](images/3501076ff9600ea518e738e1b0725151548c22bc07d26d66caf44f053611108a.jpg)

![](images/5c7d5ae0405225f741439360daec3c553e9ab5fadb63efed3f15116d24023ba7.jpg)

![](images/9032e4a5670e5ff840bfd643ef5720967c42e533e3127b9b0f9c89bef0aa191e.jpg)

![](images/b1fd8f08c08a08f9eb889502fea51eb1326b137246f6169b33820e73ef7902da.jpg)

![](images/47fc7cd3fbb32271879a5a8a841b6245878555a1ab160ee99401bbae7107b6cf.jpg)

![](images/b876330f33fc3fd52f2bbaf33d7791da08ea349d219f1061ee42e80b9034fd68.jpg)

![](images/e6810b0683ee9b79d473885d3290128ce2751283b19eea41943e099f8f868ec1.jpg)

![](images/caed5100dcd02717bb34e78d7fe8c5d772030ce17de9157be51c4936c97d2007.jpg)

Figure 7. Raw double-pendulum time series under increasing Gaussian noise. Top: periodic. Bottom: chaotic.  
![](images/4be8acc1ed40ceae6e8688cb98ad7f8faf8b1fe8f0eb70b54e46b5b88f4594de.jpg)

![](images/d8b07cf8a725faebe6b388c84f224541296b6157372fe85af742fc088fa8fff4.jpg)

![](images/74c6e2c3517f0ee78a237f0124785d64d4f10a15cf597d7024575e6fc98f764c.jpg)

![](images/942ba988b296e2a4ebc6aac6329fc26f99f3a8f636f31c72477eb66cacbbf0b4.jpg)

periodic, = 1.0  
![](images/4ad07ed237ce9b0ed16a23e1e6fe62a877ee2d16d1d23ba686521dc3235781ef.jpg)

chaotic, = 0.1  
![](images/4232391ad8a4f804f47477ced40e3b1ee9c8970bc9ad19af1facf61431344aba.jpg)

![](images/656c890237d59038b4e6765b9ddd3a902a8ed35c34403274b3724319b34f129e.jpg)

chaotic, = 0.2  
![](images/daec676d42fc7a77cd016764c6fccbe2c4b7a90d07e30eaf844f4c97fa9b8ee1.jpg)

chaotic, = 1.0  
![](images/f8f2cbf3b5488c5be24f374a87ee84b1829c0b3d447dda6ccedc8ab79406f1e6.jpg)

![](images/92dabc365ed4d01dd241cb94c6bf800f11875a2113f48c853a6853ff90c413f8.jpg)  
Figure 8. Takens-embedded point clouds under raw-signal Gaussian noise. Top: periodic. Bottom: chaotic.

Fig. 9 shows the corresponding persistence diagrams. At $\sigma \ : = \ : 0 .$ , the periodic regime’s diagram contains a single dominant, long-lived $H _ { 1 }$ bar and essentially nothing else, exactly the clean-loop signature visible in Fig. 8. As � increases, this domi nant bar’s persistence steadily shrinks, from a death time near 0.95 at $\sigma = 0$ to roughly 0.5 at $\sigma = 0 . 1$ and roughly 0.3 at $\sigma = 0 . 2$ , while a growing cloud of small, noise-induced $H _ { 1 }$ bars appears near the diagonal; by $\sigma = 0 . 5$ this cloud has grown large enough that the once-dominant bar is only marginally distinguishable from it. The chaotic regime’s diagram already contains many scattered $H _ { 1 }$ bars of comparable persistence at $\sigma = 0$ , and this qualitative pattern is largely preserved across the full noise range, aside from an overall rescaling of the birth/death axes.

Fig. 10 shows the corresponding relative featurevector drift, $\| f _ { \sigma } - f _ { 0 } \| _ { 2 } / \| f _ { 0 } \| _ { 2 }$ , where $f _ { \sigma } \in \mathbb { R } ^ { 4 2 }$ denotes the full feature vector of Sec. II B 1 recomputed from the signal at noise level $\sigma$ (and $f _ { 0 }$ the same quantity computed on the clean signal). The periodic curve rises sharply between $\sigma = 0 . 0 5$ and $\sigma = 0 . 1 5$ , from below 0.1 to above 0.8, then plateaus around 0.7–0.85 for all larger � tested; the chaotic curve remains below 0.13 across the entire range. The periodic example’s feature vector thus moves substantially further from its clean value than the chaotic example’s does at every tested � (e.g., 0.40 vs. 0.03 at $\sigma = 0 . 1$ , 0.86 vs. 0.04 at $\sigma = 0 . 2 ,$ and 0.73 vs. 0.06 at $\sigma ~ = ~ 1 . 0 )$ , consistent with both the relative-noise-level asymmetry and the qualitative point-cloud and persistence-diagram observations above.

We emphasize that this single-example asymmetry, periodic proving more fragile than chaotic here, illustrates a mechanism by which raw-signal noise degrades a distinguishing topological signature, not a claim that periodic dynamics are universally more noise-sensitive than chaotic dynamics; both the specific dynamical system’s signal amplitude and its topological structure plausibly contribute, and either could dominate for a diferent system or parameter regime. The paper’s aggregate finding, that raw-signal noise collapses classification accuracy far more sharply than feature-level noise of the same magnitude, is established in Sec. IV B by averaging over 30 independently trained classifiers and is the result that should be treated as representative; this appendix is included only to make the underlying mechanism visually concrete.

![](images/f189fb3296c98b608438afdf48b486a7251e8d55bbc05ac5eb2d247beceb99fe.jpg)  
Figure 9. Persistence diagrams under raw-signal Gaussian noise. Top: periodic. Bottom: chaotic.

![](images/3bb72da40d64ce6ebe714a8036ccaaa7726e1cd53b844bab1bc7aac9c89bb4f4.jpg)  
Figure 10. Relative feature-vector drift $\parallel f _ { \sigma } \mathrm { ~ - ~ }$ $f _ { 0 } { \bar { \| } } _ { 2 } / \| f _ { 0 } \| _ { 2 }$ under raw-signal Gaussian noise, for the periodic and chaotic double-pendulum examples of Figs. 7–9.

[1] A. Provenzale, L. A. Smith, R. Vio, and G. Murante, Distinguishing between low-dimensional dynamics and randomness in measured time series, Physica D: Nonlinear Phenomena 58, 31 (1992).

[2] A. A. Tsonis and J. B. Elsner, Nonlinear prediction as a way of distinguishing chaos from random fractal sequences, Nature 358, 217 (1992).

[3] M. Cencini, M. Falcioni, E. Olbrich, H. Kantz, and A. Vulpiani, Chaos or noise: Dificulties of a distinction, Physical Review E 62, 427 (2000).

[4] P. Grassberger and I. Procaccia, Characterization of strange attractors, Phys. Rev. Lett. 50, 346 (1983).

[5] A. Casaleggio, A. Corana, and S. Ridella, Correlation dimension estimation from electrocardiograms, Chaos, Solitons & Fractals 5, 713 (1995).

[6] J. Argyris, I. Andreadis, G. Pavlos, and M. Athanasiou, The influence of noise on the correlation dimension of chaotic attractors, Chaos, Solitons & Fractals 9, 343 (1998).

[7] N. Boull´e, V. Dallas, Y. Nakatsukasa, and D. Samaddar, Classification of chaotic time series with deep learning, Physica D: Nonlinear Phenomena 403, 132261 (2020).

[8] M. Zanin, Can deep learning distinguish chaos from

noise? numerical experiments and general considerations, Communications in Nonlinear Science and Numerical Simulation 114, 106708 (2022).

[9] J. Choi, A. L. Chanu, and J.-M. Park, Learning a quantitative criterion for distinguishing chaos from noise, arXiv:2608.07109 (2026).

[10] Z. Wang, W. Yan, and T. Oates, Time series classification from scratch with deep neural networks: A strong baseline, in 2017 International Joint Conference on Neural Networks (IJCNN) (2017) pp. 1578– 1585.

[11] H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar, and P.-A. Muller, Deep learning for time series classification: A review, Data Mining and Knowledge Discovery 33, 917 (2019).

[12] H. Ismail Fawaz, B. Lucas, G. Forestier, C. Pelletier, D. F. Schmidt, J. Weber, G. I. Webb, L. Idoumghar, P.-A. Muller, and F. Petitjean, Inceptiontime: Finding alexnet for time series classification, Data Min ing and Knowledge Discovery 34, 1936 (2020).

[13] Q. Wen, T. Zhou, C. Zhang, W. Chen, Z. Ma, J. Yan, and L. Sun, Transformers in time series: A survey, in Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence (2023) pp. 6778–6786.

[14] F. Takens, Detecting strange attractors in turbulence, in Dynamical Systems and Turbulence, Warwick 1980: proceedings of a symposium held at the University of Warwick 1979/80 (Springer, 2006) pp. 366–381.

[15] A. Myers, E. Munch, and F. A. Khasawneh, Persistent homology of complex networks for dynamic state detection, Phys. Rev. E 100, 022314 (2019).

[16] J. R. Tempelman and F. A. Khasawneh, A look into chaos detection through topological data anal ysis, Physica D: Nonlinear Phenomena 406, 132446 (2020).

[17] P. Bubenik, Statistical topological data analysis using persistence landscapes, The Journal of Machine Learning Research 16, 77 (2015).

[18] H. Adams, T. Emerson, M. Kirby, R. Neville, C. Peterson, P. Shipman, S. Chepushtanova, E. Hanson, F. Motta, and L. Ziegelmeier, Persistence images: A stable vector representation of persistent homology, Journal of Machine Learning Research 18, 1 (2017).

[19] I. Chevyrev, V. Nanda, and H. Oberhauser, Persistence paths and signature features in topological data analysis, IEEE Transactions on Pattern Analysis and Machine Intelligence 42, 192 (2020).

[20] N. Atienza, R. Gonzalez-Diaz, and M. Soriano-Trigueros, On the stability of persistent entropy and new summary functions for topological data analysis, Pattern Recognition 107, 107509 (2020).

[21] Y. Umeda, Time series classification via topological data analysis, Transactions of the Japanese Society for Artificial Intelligence 32, D (2017).

[22] A. Karan and A. Kaygun, Time series classification via topological data analysis, Expert Systems with Applications 183, 115326 (2021).

[23] F. A. Khasawneh, E. Munch, D. Barnes, M. M. Chumley, <sup>˙</sup>I. G¨uzel, A. D. Myers, S. Tanweer, S. Ty-

mochko, and M. Yesilli, Teaspoon: A python package for topological signal processing, Journal of Open Source Software 10, 7243 (2025).

[24] E. N. Lorenz, Deterministic nonperiodic flow, Journal of Atmospheric Sciences 20, 130 (1963).

[25] M. H´enon and C. Heiles, The applicability of the third integral of motion: some numerical experiments, Astronomical Journal, Vol. 69, p. 73 (1964) 69, 73 (1964).

[26] P. Grassberger and I. Procaccia, Measuring the strangeness of strange attractors, Physica D: nonlinear phenomena 9, 189 (1983).

[27] J. Theiler, Eficient algorithm for estimating the correlation dimension from a set of discrete points, Phys. Rev. A 36, 4456 (1987).

[28] For computational reasons, we set $H _ { \mathrm { m a x } } = 2$ and therefore retain $H _ { 0 }$ and $H _ { 1 }$

[29] Because the capped death value depends on the largest finite death observed for a given point cloud, the resulting lifetime assigned to the essential $H _ { 0 }$ class is an implementation-dependent finite value and should not be interpreted as a topological invariant of the underlying attractor.

[30] N. Atienza, R. Gonzalez-Diaz, and M. Rucco, Persistent entropy for separating topological features from noise in vietoris-rips complexes, Journal of Intelligent Information Systems 52, 637 (2019).

[31] The factor 0.05 defines a fixed relative threshold used throughout the analysis. Here, “significant” refers only to this thresholding criterion and does not imply statistical significance.

[32] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, Transformers are rnns: Fast autoregressive transformers with linear attention, in Proceedings of the 37th International Conference on Machine Learning, Vol. 119 (PMLR, 2020) pp. 5156– 5165.

[33] The exponential linear unit reads $\mathrm { E L U } ( x ) = x$ for � > 0 and ELU(�) = �<sup>�</sup> − 1 for $x \leq 0$

[34] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization, arXiv preprint arXiv:1412.6980 10.48550/arXiv.1412.6980 (2014).

[35] I. Loshchilov and F. Hutter, Decoupled weight decay regularization, arXiv preprint arXiv:1711.05101 (2017).

[36] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, On calibration of modern neural networks, in Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML’17 (JMLR.org, 2017) p. 1321–1330.

[37] J. P. Eckmann, S. O. Kamphorst, D. Ruelle, and S. Ciliberto, Liapunov exponents from time series, Phys. Rev. A 34, 4971 (1986).

[38] The single essential $H _ { 0 }$ class, whose death time is formally infinite, is capped at the largest finite death value observed across all tracked homology dimensions for that point cloud before any downstream statistics are computed; this ensures every reported quantity, including the entropy, lifetime, Betti-curve, and persistence-image features, is finite-valued by construction.

[39] Because Betti curves are piecewise constant, this criterion does not count turning points separated by one or more zero-valued diferences (plateaus); it therefore provides a conservative lower bound on the number of true direction changes in the curve, rather than an exhaustive count.

[40] K. Jordan, Y. Jin, V. Boza, Y. Jiacheng, F. Cesista, L. Newhouse, and J. Bernstein, Muon: An optimizer for hidden layers in neural networks (2024).

[41] F. Chazal, V. de Silva, and S. Oudot, Persistence stability for geometric complexes, Geometriae Dedicata 173, 193 (2014).

[42] H. Cao, D. Leykam, and D. G. Angelakis, Unravelling quantum chaos using persistent homology,

Phys. Rev. E 107, 044204 (2023).

[43] M. Kerem Maden, A. Ullah, B. Coskunuzer, and Q. E. Mustecaplıoglu, Topological engine monitor: persistent homology-based fault detection in finitetime quantum engines, Quantum Science and Technology 11, 045012 (2026).

[44] C. Tralie, N. Saul, and R. Bar-On, Ripser.py: A lean persistent homology library for python, The Journal of Open Source Software 3, 925 (2018).

[45] N. Saul and C. Tralie, Scikit-tda: Topological data analysis for python (2019).

[46] P. Dlotko, Persistence representations, in GUDHI User and Reference Manual (GUDHI Editorial Board, 2022) 3.5.0 ed.