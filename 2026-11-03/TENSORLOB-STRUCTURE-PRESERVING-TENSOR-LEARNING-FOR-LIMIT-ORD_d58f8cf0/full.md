# TENSORLOB: STRUCTURE-PRESERVING TENSOR LEARNING FOR LIMIT ORDER BOOK FORECASTING

Yuxuan Gu<sup>1</sup>, Danilo Mandic<sup>1</sup>

<sup>1</sup> Department of Electrical and Electronic Engineering, Imperial College London, United Kingdom

## ABSTRACT

The limit order book (LOB) records outstanding buy and sell orders across multiple price levels. Its state evolves over time across market side, price level, and order attribute, naturally forming a multiway structure. However, existing LOB models typically flatten these dimensions into a matrix model, thus obscuring the existing structural relationships. Starting from first principles and the marketmicrostructure intuition of the LOB, we design TensorLOB to operate entirely in the tensor domain. TensorLOB uses mode-wise tensor contractions for feature extraction and directly contracts the learned representation with low-rank class-weight tensors for prediction, retaining the distinct roles of time, side, level, and attribute modes. Experiments on the FI-2010 benchmark show that TensorLOB achieves the highest accuracy, recall, and F1 score across all four evaluated horizons. This proof of concept highlights the potential of highdimensional modelling of multi-way latent information intrinsic to financial markets, which has so far been largely overlooked.

Index Terms— Limit Order Book, Mid-Price Prediction, Tensor Classification, CP Decomposition, Deep Learning

## 1. INTRODUCTION

The limit order book (LOB) records outstanding buy and sell orders of a traded asset at multiple price levels. Its state evolves whenever orders are submitted, cancelled, or executed, potentially affecting the best bid and ask prices. Predicting short-term movements of the midprice, which is the average of the best ask and bid prices, is therefore important for market making, execution, and high-frequency risk management. Yet this remains a challenging prediction problem since LOB data are noisy and non-stationary, while the resulting movement classes are often highly imbalanced [1].

At each event, the LOB comprises two sides: ask and bid. Each side is arranged over an ordered sequence of price levels, with price and volume providing different attributes. Across successive events, those changes capture how orders on both sides evolve over time. Hence, a LOB history is naturally a higher-order tensor, with time, market side, price level and attribute as tensor modes.

Despite the naturally multiway structure of LOB, most existing deep LOB prediction methods do not consider the full higher-order structure of the book throughout representation learning. Convolutional, recurrent, attention-based, and bilinear architectures have achieved strong predictive performance [2, 3, 4, 5], but operate on vectorized or matricized representations that collapse several economically meaningful LOB dimensions into a single feature axis. Consequently, market side, price level, and order attribute are not retained as distinct modes, and their structural relationships must instead be learned implicitly from data, yielding a weaker inductive bias for LOB modelling. This motivates a prediction framework that preserves and exploits the native multiway structure of the LOB throughout representation learning, as the subject of this work.

![](images/7bd1c2bd03ed7b1bf798f5bca1fc3cfde3699c7a7bb7931852ad131822f2032a.jpg)  
Fig. 1. A snapshot of a limit order book snapshot at a single event.

To this end, we propose TensorLOB, a structure-preserving neural model that operates directly on the tensor representation of LOB data. TensorLOB uses the native LOB modes to guide representation learning throughout the network. Mode-aware normalization captures temporal and cross-sectional variation, while multi-head Kronecker-factorized tensor projections learn modespecific interactions without collapsing the side and level axes. Finally, the tensor classifier combines information across time, side, level and learned features for mid-price movement prediction.

The proposed structure-preserving design is evaluated over comprehensive experiments against the state-of-the-art (SOTA) on the FI-2010 benchmark [1]. Our proposed TensorLOB achieves the highest mean accuracy, weighted recall, and weighted F1 across a range of prediction horizons. Such consistent performance gains indicate that explicitly preserving the native LOB structure provides an effective inductive bias for mid-price movement prediction.

In summary, our main contributions are threefold:

• We derive TensorLOB from first-principles market-microstructure intuition, representing the LOB through its native time, side, level, and attribute modes and performing the prediction pipeline entirely in the tensor domain.

• We develop a structure-preserving architecture based on mode-wise tensor contractions for feature extraction and low-rank class-weight tensors for prediction, enabling modespecific interactions while retaining the native multiway structure of the LOB.

• We conduct extensive experiments on the FI-2010 benchmark, demonstrating consistent improvements in accuracy and weighted F1 across multiple prediction horizons over vectorized, bilinear, recurrent, and conventional baselines.

![](images/758a78f19ef0a8bbed0e6bb10989a3e2876b1b7d07f46e05ce0dc353da586394.jpg)  
Fig. 2. An overview of the TensorLOB architecture with multi-head tensor projection layers and a tensor classifier.

## 2. RELATED WORK

## 2.1. LOB Movement Prediction

DeepLOB [2] combines convolutional neural networks (CNNs) for feature extraction from vectorized LOB inputs with an LSTM for temporal modelling. DeepLOB-Seq2Seq and DeepLOB-Attention [3] extend this architecture to multi-horizon prediction through sequenceto-sequence and attention-based decoding, respectively. An alternative line of work is based on temporal attention-augmented bilinear networks (TABL) [4]. B(TABL) applies bilinear transformations along the flattened feature and temporal axes, while C(TABL) further introduces temporal attention to learn importance of different time steps. BiN-C(TABL) [5] augments this framework with learnable normalization along both flattened feature and temporal axes to account for different input scales. These methods improve feature extraction, temporal modelling, and input normalization, but represent the LOB data as a vector or a matrix, with market side, price level, and any order attribute collapsed to a single feature dimension.

## 2.2. Tensor Methods

Tensor methods have been widely used in various ways with neural networks, such as large language model (LLM) compression [6, 7, 8], parameter-efficient finetuning [9, 10], tensorized neural representation [11, 12], and interpretable financial prediction [13, 14]. In LOB prediction, Roa-Vicens et al. [15] embedded a multi-linear graph filter into tensor-train recurrent networks for high-dimensional LOB data. In contrast, TensorLOB operates directly on the native LOB tensor throughout the network during both feature extraction and classification stages. This provides a more direct structurepreserving inductive bias while remaining parameter-efficient.

## 3. METHODOLOGY

## 3.1. Preliminaries

## 3.1.1. Limit Order Book

A limit order book (LOB) is a collection of outstanding buy and sell orders of a financial instrument at discrete price levels. As Fig. 1 shows, the ask side contains sell orders with prices greater than the mid-price, while the bid side contains buy orders with prices lower than the mid-price. The best ask and bid prices are the lowest ask price and the highest bid price, respectively, and their average defines the mid-price. On each side, there are multiple levels of orders, with each level characterized by its price and volume.

## 3.1.2. Tensor Decomposition

A tensor is a multi-dimensional array whose dimensions are referred to as modes. Following [16], we use a, a, A, and A to denote scalars, vectors, matrices, and tensors, respectively. For an order-N tensor $\mathcal { A } \in \mathbb { R } ^ { I _ { 1 } \times \dots \times I _ { N } }$ , its $( i _ { 1 } , \dots , i _ { N } )$ )th entry is denoted by $a _ { i _ { 1 } \cdots i _ { N } }$

The mode-n product of an Nth-order tensor $\mathcal { A } \in \mathbb { R } ^ { I _ { 1 } \times \cdots \times I _ { N } }$ with a matrix M $\mathbf { \chi } \in \mathbb { R } ^ { J \times I _ { n } }$ is denoted by $\mathcal { A } \times _ { n } \textbf { M }$ . The resulting tensor has shape $I _ { 1 } \times \cdot \cdot \cdot \times I _ { n - 1 } \times J \times I _ { n + 1 } \times \cdot \cdot \cdot \times I _ { N }$ and is defined elementwise as

$$
\left( \mathcal A \times _ { n } \mathbf M \right) _ { i _ { 1 } , \dots , i _ { n - 1 } , j , i _ { n + 1 } , \dots , i _ { N } } = \sum _ { q = 1 } ^ { I _ { n } } a _ { i _ { 1 } , \dots , i _ { n - 1 } , q , i _ { n + 1 } , \dots , i _ { N } } m _ { j q } .\tag{1}
$$

More generally, let $\mathcal { A } \in \mathbb { R } ^ { I _ { 1 } \times \dots \times I _ { N } }$ and $\boldsymbol { B } \in \mathbb { R } ^ { J _ { 1 } \times \cdots \times J _ { M } }$ Their contraction over $P$ pairs of modes $( i _ { p } , j _ { p } )$ , where $I _ { i _ { p } } = J _ { j _ { p } }$ for $p = 1 , \ldots , P$ , is denoted by $\mathcal { A } \times _ { i _ { 1 } , . . . , i _ { P } } ^ { j _ { 1 } , . . . , j _ { P } } \mathcal { B }$ . The elementwise definition is given by

$$
c _ { \mathbf { i } _ { \bar { \alpha } } \mathbf { j } _ { \bar { \beta } } } = \sum _ { q _ { 1 } = 1 } ^ { I _ { \alpha _ { 1 } } } \cdot \cdot \cdot \sum _ { q _ { P } = 1 } ^ { I _ { \alpha _ { P } } } a _ { \mathbf { i } _ { \bar { \alpha } } , q _ { 1 } , \dots , q _ { P } } b _ { \mathbf { j } _ { \bar { \beta } } , q _ { 1 } , \dots , q _ { P } } ,\tag{2}
$$

where $\mathbf { i } _ { \bar { \alpha } }$ and $\mathbf { j } _ { \bar { \beta } }$ denote the uncontracted mode indices, respectively.

The Canonical Polyadic (CP) Decomposition represents an order-N tensor as a sum of rank-one outer products. A rank-R CP factorization of $\mathcal { A } \in \mathbb { R } ^ { I _ { 1 } \times \cdots \times I _ { N } }$ is given by

$$
\mathcal { A } = \sum _ { r = 1 } ^ { R } \mathbf { a } _ { r } ^ { ( 1 ) } \circ \mathbf { a } _ { r } ^ { ( 2 ) } \circ \cdot \cdot \cdot \circ \mathbf { a } _ { r } ^ { ( N ) } ,\tag{3}
$$

where $\mathbf { a } _ { r } ^ { ( n ) } \in \mathbb { R } ^ { I _ { n } }$ is the r-th factor vector associated with mode n. Each term $\mathbf { a } _ { r } ^ { ( 1 ) } \circ \cdots \circ \mathbf { a } _ { r } ^ { ( N ) }$ is therefore a rank-one tensor of size $I _ { 1 } \times \cdots \times I _ { N }$ . This representation requires $R \sum _ { n = 1 } ^ { N } I _ { n }$ factor parameters, compared with $\textstyle \prod _ { n = 1 } ^ { N } I _ { n }$ for an unstructured tensor, which is parameter-efficient when $\begin{array} { r } { \dot { R } \sum _ { n } I _ { n } \ll \Pi _ { n } I _ { n } } \end{array}$

## 3.2. TensorLOB Architecture

TensorLOB takes as input the most recent $T = 1 0 0 ~ \mathrm { L O B }$ events, arranged into a fourth-order tensor $\mathcal { X } \in \mathbb { R } ^ { T ^ { ( 0 ) } \times S ^ { ( 0 ) } \times L ^ { ( 0 ) } \times A ^ { ( 0 ) } } =$ R<sup>100×2×10×2</sup>, with the four modes as time $T ^ { ( 0 ) }$ , market side (ask or bid) $S ^ { ( 0 ) }$ , price level $L ^ { ( 0 ) }$ , and attribute (price or volume) $A ^ { ( 0 ) }$ respectively. Rather than flattening these modes into a single feature dimension, TensorLOB directly operates on the 4-D structure throughout the prediction pipeline. As illustrated in Fig. 2, the architecture consists of a mode-aware input normalization, two multi-head tensor projection layers for feature extraction, and a CP-parameterized tensor classifier for movement prediction.

## 3.2.1. Input Normalization

Following the bilinear normalization (BiN) scheme of [5], we normalize each input tensor from two complementary views. The temporal view normalizes each book entry across the T events, while the cross-sectional view normalizes the book state within each event across the non-temporal modes. The two normalized tensors are then combined using learned non-negative weights, with the normalization statistics computed independently for each input sample and the affine parameters learned during training.

## 3.2.2. Multi-Head Kronecker-Factorized Feature Extraction

Let $\mathcal { H } ^ { ( m - 1 ) } \in \mathbb { R } ^ { T _ { m - 1 } \times S _ { m - 1 } \times L _ { m - 1 } \times A _ { m - 1 } }$ denote the input to feature extraction layer m, with $\mathcal { H } ^ { ( 0 ) } = \widetilde { \mathcal { X } }$ . Each layer performs feature extraction through mode-wise tensor contractions using $H _ { m }$ projection heads. All heads share the same transformation along the temporal mode, while each head learns distinct projections along the remaining modes. This multi-head projection allows the layer to capture complementary mode-specific interaction patterns in parallel, rather than relying on a single projection.

Specifically, these mode-wise projections are parameterized by a temporal factor matrix, $U ^ { ( m ) } ~ \in ~ \mathbb { R } ^ { T _ { m - 1 } \times T _ { m } }$ , together with head-specific factor matrices, $\mathring { V } ^ { { ( m , h ) } } \in \mathbb { R } ^ { S _ { m - 1 } \times \mathring { S } _ { m } } , \ : \ : W ^ { ( m , h ) } \in$ $\mathbb { R } ^ { L _ { m - 1 } \times L _ { m } }$ , and $G ^ { ( m , h ) } \ \in \ \mathbb { R } ^ { A _ { m - 1 } \times A _ { m } }$ , for side, level and attribute modes, respectively. These factor matrices define modespecific linear transformations while preserving the tensor structure of the LOB representation. The resulting pre-activation representation at layer m then becomes

$$
\begin{array} { r } { \boldsymbol { \mathcal { Q } } ^ { ( m ) } = \displaystyle \frac { 1 } { \sqrt { H _ { m } } } \sum _ { h = 1 } ^ { H _ { m } } \Biggl ( \mathcal { H } ^ { ( m - 1 ) } \times _ { 1 } ^ { 2 } \left( \mathbf { U } ^ { ( m ) } \right) ^ { \top } \times _ { 1 } ^ { 2 } \left( \mathbf { V } ^ { ( m , h ) } \right) ^ { \top } } \\ { \times _ { 1 } ^ { 2 } \left( \mathbf { W } ^ { ( m , h ) } \right) ^ { \top } \times _ { 1 } ^ { 2 } \left( \mathbf { G } ^ { ( m , h ) } \right) ^ { \top } \Biggr ) , } \end{array}\tag{4}
$$

where the successive contractions are applied on all modes of $\mathcal { H } ^ { ( m - 1 ) }$ , yielding $\mathcal { Q } ^ { ( m ) } \in \mathbb { R } ^ { T _ { m } \times S _ { m } \times L _ { m } ^ { \star } \times A _ { m } }$ . It is worth noting that the tensor formulation in Eq. (4) also admits a Kroneckerfactorized interpretation.

Observation 1 (Tensor contraction and Kronecker representation) Let $\mathcal { X } \in \mathbb { R } ^ { I _ { 1 } \times \dots \times I _ { N } }$ and let $\mathbf { A } _ { n } \ \in \ \mathbb { R } ^ { J _ { n } \times I _ { n } } \ f o r \ n \ = \ 1 , \dots , N .$ Assume

$$
\begin{array} { r } { \mathcal { Y } = \mathcal { X } \times _ { 1 } \mathbf { A } _ { 1 } \times _ { 2 } \mathbf { A } _ { 2 } \times _ { 3 } \cdot \cdot \cdot \times _ { N } \mathbf { A } _ { N } . } \end{array}\tag{5}
$$

Then, thefollowing holds.

$$
\begin{array} { r } { \mathrm { v e c } ( \mathcal { V } ) = \left( \mathbf { A } _ { N } \otimes \mathbf { A } _ { N - 1 } \otimes \cdots \otimes \mathbf { A } _ { 1 } \right) \mathrm { v e c } ( \mathcal { X } ) , } \end{array}\tag{6}
$$

where ⊗ denotes the Kronecker product.

Observation 1 shows that a sequence of mode-wise tensor contractions is equivalent to a single Kronecker-factorized linear transformation after vectorization. Hence, each feature extraction head in Eq. (4) is equivalent to a Kronecker-factorized linear transformation after vectorization, while TensorLOB performs the computation directly in the tensor form. The following simple example illustrates how this factorization explicitly encodes mode-wise relationships that are absent from an unrestricted vectorized parameterization.

Remark 1 In Eq. (4),for a head h, the contributionfrom $\mathcal { H } _ { t s l a } ^ { ( m - 1 ) } t c$ a fixed output entry $\mathcal { Q } _ { k q r f } ^ { ( m ) }$ is weighted by $U _ { t k } ^ { ( m ) } V _ { s q } ^ { ( m , h ) } W _ { l r } ^ { ( m , h ) } G _ { a f } ^ { ( m , h ) }$ Hence, input entries sharing the same mode index also share the corresponding factor in their effective weights. For example, two entries at the same event index t share $U _ { t k } ^ { ( m ) }$ . If they also share the same level l, they additionally share $W _ { l r } ^ { ( m , h ) }$ . In contrast, a dense transformation after vectorization assigns an independent coefficient to each pair offlattened indices. Thus, TensorLOB explicitly ties coefficients according to the multiway structure, rather than requiring such relationships to be learned from data.

The projected representation is next normalized along the attribute mode, followed by a GELU activation [17] and dropout [18],

$$
\begin{array} { r } { \mathcal { H } ^ { ( m ) } = \mathrm { D r o p o u t } \left( \mathrm { G E L U } \left( \mathrm { L N } _ { A _ { m } } \left( \mathcal { Q } ^ { ( m ) } \right) \right) \right) . } \end{array}\tag{7}
$$

The first feature extraction layer uses $H _ { 1 } = 2 4$ heads and maps a tensor of size $1 0 0 \times 2 \times 1 0 \times 2 \mathrm { t o 3 5 } \times 2 \times 1 0 \times 6 .$ The second layer uses $H _ { 2 } = 4 4$ heads and produces a tensor of size $2 0 \times 2 \times 1 0 \times 1 2$ Thus, the two layers progressively reduce the temporal dimension, from 100 to 12, while expanding the learned attribute dimension, from 2 to 10, with the side and level dimensions preserved.

## 3.2.3. CP-Parameterized Tensor Classification

Given $\mathcal { H } ^ { ( 2 ) } \in \mathbb { R } ^ { T _ { 2 } \times S _ { 2 } \times L _ { 2 } \times A _ { 2 } }$ , where $T _ { 2 } = 2 0 , S _ { 2 } = 2 , L _ { 2 } =$ 10, and $A _ { 2 } \ = \ 1 2 ,$ instead of first collapsing all modes into an intermediate vector representation, we directly map the tensor to the movement class logits. For each class $k = 1 , 2 , 3$ , let $\mathcal { C } _ { k } ~ \in$ $\mathbb { R } ^ { T _ { 2 } \times S _ { 2 } \times L _ { 2 } \times A _ { 2 } }$ denote its classification coefficient tensor. To retain the mode-wise structure also in the prediction layer, we parameterize $\mathcal { C } _ { k }$ using a rank-R CP decomposition,

$$
\mathcal { C } _ { k } = \sum _ { r = 1 } ^ { R } \mathbf { a } _ { k r } ^ { ( T ) } \circ \mathbf { a } _ { k r } ^ { ( S ) } \circ \mathbf { a } _ { k r } ^ { ( L ) } \circ \mathbf { a } _ { k r } ^ { ( A ) } ,\tag{8}
$$

where $\mathbf { a } _ { k r } ^ { ( T ) } \in \mathbb { R } ^ { T _ { 2 } } , \mathbf { a } _ { k r } ^ { ( S ) } \in \mathbb { R } ^ { S _ { 2 } } , \mathbf { a } _ { k r } ^ { ( L ) } \in \mathbb { R } ^ { L _ { 2 } }$ , and $\mathbf { a } _ { k r } ^ { ( A ) } \in \mathbb { R } ^ { A _ { 2 } }$ are the temporal, side, level, and learned-feature factors, respectively.

The logit associated with class k is obtained by contracting $\dot { \mathcal { H } } ^ { ( 2 ) }$ with each rank-one CP component along all four modes to yield

$$
\boldsymbol { \ell } _ { k } = \sum _ { r = 1 } ^ { R } \mathcal { H } ^ { ( 2 ) } \times 1 \left( \mathbf { a } _ { k r } ^ { ( T ) } \right) ^ { \top } \times _ { 2 } \left( \mathbf { a } _ { k r } ^ { ( S ) } \right) ^ { \top } \times _ { 3 } \left( \mathbf { a } _ { k r } ^ { ( L ) } \right) ^ { \top } \times _ { 4 } \left( \mathbf { a } _ { k r } ^ { ( A ) } \right) ^ { \top } + b _ { k } ,\tag{9}
$$

where $b _ { k }$ is the class-specific bias. Equivalently,

$$
\ell _ { k } = \left. \mathcal { H } ^ { ( 2 ) } , \mathcal { C } _ { k } \right. + b _ { k } .\tag{10}
$$

Thus, each rank-one CP component specifies a separable interaction across the modes, while summing over R components allows multiple multilinear interaction patterns to contribute to each class score. Applying Eq. (9) to all classes yields the logit vector $\ell =$ $[ \ell _ { 1 } , \ell _ { 2 } , \ell _ { 3 } ] ^ { T } \in \mathbb { R } ^ { 3 }$ , with the class probabilities are $\mathbf { p } = \operatorname { S o f t M a x } ( \ell )$ and the predicted movement, with $\hat { y } = \arg \operatorname* { m a x } _ { k \in \{ 1 , 2 , 3 \} } p _ { k }$ , corresponding to downward, stationary, and upward movements.

Remark 2 We use CP rank $R \ = \ 1 6 .$ The CP parameterization requires $K R ( T _ { 2 } + S _ { 2 } + L _ { 2 } + A _ { 2 } ) + K$ learnable classification parameters, compared with $K T _ { 2 } S _ { 2 } L _ { 2 } A _ { 2 } + K$ for an unrestricted coefficient tensor. With $K = 3 , T _ { 2 } = 2 0 , S _ { 2 } = 2 , L _ { 2 } = 1 0$ and $A _ { 2 } = 1 2 ,$ , the CP classifier requires 2,115 parameters, compared with 14,403 for the corresponding dense classifier, representing a 85.3% parameter reduction in the classification layer.

## 4. EXPERIMENTS

## 4.1. Experimental Setup

We evaluated TensorLOB on the pre-normalized FI-2010 benchmark using the provided three-class mid-price movement labels (down, stationary, and up) at prediction horizons of 10, 20, 50, and 100 future events [1]. Each sample consists of the most recent 100 LOB events, with the target assigned to the final event of the window. Following the standard setting in the benchmark, the first seven trading days were used for model development and the final three for testing. The development period was split chronologically, with the first 80% for training and the remaining 20% for validation. This produced 203,701 training, 50,851 validation, and 139,488 test samples.

All neural models were trained under a common optimization and model-selection criterion. We minimize cross-entropy with label smoothing of 0.05 using AdamW optimizer [19], with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ , weight decay of $1 0 ^ { - 4 }$ , and cosine learningrate annealing. Training was limited to 60 epochs with a batch size of 1,024, and gradients are clipped to unit norm. Training samples were shuffled within each epoch, whereas the validation and test sets retain their chronological order. The checkpoint with the highest validation weighted-average F1 score was retained, and the training terminates after 12 consecutive epochs without improvement.

We compare TensorLOB with the SOTA neural and bilinear models described in Section 2.1, together with conventional machine-learning methods [20], including multilayer perceptron (MLP), XGBoost, linear support vector machine (SVM), and LSTM. Each experiment was repeated with seeds 7, 42, and 123, and results are reported as the mean ± sample standard deviation. Performance was measured by accuracy and weighted-average precision, recall, and F1 over the three movement classes. Note that for single-label multiclass classification, weighted average recall is algebraically equal to accuracy, and both were retained for consistency.

## 4.2. Numerical Results

Table 1 shows that the advantage of TensorLOB is not limited to a particular metric or prediction horizon. TensorLOB achieves the highest mean accuracy, weighted recall, and weighted F1 across all four prediction horizons, while also attaining the highest weighted precision at three of them. The improvement is most pronounced at the 20-event horizon, where its weighted F1 exceeds that of the secong best method, BiN-C(TABL), by 3.3 percentage points. Although the gap becomes smaller at horizon 50, TensorLOB retains the lead and the margin widens again at horizon 100. More broadly, its gains over DeepLOB become particularly substantial at the 20- and 100-event horizons, indicating that the advantage persists as the prediction task changes in temporal scale.

These results are consistent with the benefit of the structurepreserving design of TensorLOB. While several baselines deteriorate considerably as the horizon changes, and some collapse to predicting only the majority stationary class at horizons 10 and 20, TensorLOB maintains strong performance across all considered metrics. By retaining the time, side, level, and attribute modes, the model is able to exploit mode-specific information that is obscured by flattening through vectorization or matricization as in the current models. In particular, the mode-wise tensor contractions provide structured feature interactions, while the tensor classifier directly combines these interactions for classification without discarding the underlying LOB geometry. The consistent gains over the vectorized baselines support tensor structure as an effective inductive bias for LOB prediction.

Table 1. Comparison of test performance on FI-2010 benchmark of all methods over four prediction horizons h. Results are reported as mean ± sample standard deviation over three random seeds, with the best value for each metric in bold.
<table><tr><td rowspan=1 colspan=8>h    Model    Accuracy ↑ Precision ↑ Recall ↑   F1 ↑</td></tr><tr><td rowspan=1 colspan=8>DeepLOB380.4 ± 0.9 79.0 ± 0.9 80.4 ± 0.9 79.2 ± 0.8</td></tr><tr><td rowspan=1 colspan=8>DeepLOB-Seq 70.7 ± 0.0 50.0 ± 0.0 70.7 ± 0.0 58.6 ± 0.0DeepLOB-Att. 67.5 ± 9.1 63.6 ± 7.5 67.5 ± 9.1 64.4 ± 7.8</td></tr><tr><td rowspan=1 colspan=8>Linear SVM 70.7 ± 0.0 50.0 ± 0.0 70.7 ± 0.0 58.6 ± 0.0</td></tr><tr><td rowspan=4 colspan=8>MLP    70.7 ± 0.0 50.0 ± 0.0 70.7 ± 0.0 58.6 ± 0.0LSTM    70.7 ± 0.0 50.0 ± 0.0 70.7 ± 0.0 58.6 ± 0.010B(TABL)  70.7 ± 0.0 50.0 ± 0.0 70.7 ± 0.0 58.6 ± 0.0C(TABL)  70.7 ± 0.0 50.0 ± 0.0 70.7 ± 0.0 58.6 ± 0.0BiN-C(TABL) 82.2 ± 1.7 81.7 ± 2.1 82.2 ± 1.7 80.9 ± 1.7</td></tr><tr><td rowspan=1 colspan=5></td></tr><tr><td rowspan=1 colspan=3>70.7 ±</td><td rowspan=1 colspan=5></td></tr><tr><td rowspan=1 colspan=4></td></tr><tr><td rowspan=1 colspan=8>XGBoost78.9 ± 0.0 78.9 ± 0.0 78.9 ± 0.0 75.4 ± 0.0</td></tr><tr><td rowspan=1 colspan=8>TensorLOB 83.5 ± 0.1 83.0 ± 0.1 83.5 ± 0.1 82.5 ± 0.1</td></tr><tr><td rowspan=1 colspan=8>DeepLOB 64.1 ± 0.8 62.8 ± 1.0 64.1 ± 0.8 63.1 ± 0.7</td></tr><tr><td rowspan=1 colspan=7>DeepLOB-Seq 62.1 ± 0.0 38.5 ± 0.0 62.1 ± 0.0</td><td rowspan=1 colspan=1>47.6 ± 0.0</td></tr><tr><td rowspan=1 colspan=7>DeepLOB-Att. 56.4 ± 9.9 40.8 ± 3.9 56.4 ± 9.9</td><td rowspan=1 colspan=1>46.2 ± 2.3</td></tr><tr><td rowspan=1 colspan=7>Linear SVM 57.5 ± 1.5 46.4 ± 1.1 57.5 ± 1.5</td><td rowspan=1 colspan=1>48.5 ± 0.4</td></tr><tr><td rowspan=1 colspan=5>MLP    62.1 ± 0.0 3</td><td rowspan=1 colspan=2>8.5 ± 0.0 62.1 ± 0.0 4</td><td rowspan=1 colspan=1>7.6 ± 0.0</td></tr><tr><td rowspan=2 colspan=5>LSTM[62.1 ± 0.0 320            62.1 ± 0.0 3B(TABL)</td><td rowspan=1 colspan=2>8.5 ± 0.0 62.1 ± 0.0 4</td><td rowspan=1 colspan=1>7.6 ± 0.0</td></tr><tr><td rowspan=1 colspan=2>8.5 ± 0.0 62.1 ± 0.0 4</td><td rowspan=1 colspan=1>7.6 ± 0.0</td></tr><tr><td rowspan=1 colspan=2>C(TABL)</td><td rowspan=1 colspan=3>62.1 ± 0.0 3</td><td rowspan=1 colspan=2>8.5 ± 0.0 62.1 ± 0.0 4</td><td rowspan=1 colspan=1>7.6 ± 0.0</td></tr><tr><td rowspan=1 colspan=2>BiN-C(TABL)</td><td rowspan=1 colspan=3>71.4 ± 4.3</td><td rowspan=1 colspan=2>70.2 ± 5.6 71.4 ± 4.3 6</td><td rowspan=1 colspan=1>8.9 ± 4.6</td></tr><tr><td rowspan=1 colspan=5>XGB0ost 70.8 ± 0.0 7</td><td rowspan=1 colspan=1>0.9 ± 0.0 7</td><td rowspan=1 colspan=1>0.8 ± 0.0 6</td><td rowspan=1 colspan=1>6.1 ± 0.0</td></tr><tr><td rowspan=1 colspan=5>TensorLOB 74.0 ± 0.1</td><td rowspan=1 colspan=1>73.2 ± 0.1</td><td rowspan=1 colspan=2>74.0 ± 0.1 72.2 ± 0.0</td></tr><tr><td rowspan=1 colspan=5>DeepLOB 71.7 ± 1.2 7</td><td rowspan=1 colspan=1>1.7 ± 0.8 7</td><td rowspan=1 colspan=2>1.7 ± 1.2 71.6 ± 0.9</td></tr><tr><td rowspan=1 colspan=5>DeepLOB-Seq 44.4 ± 12.1 4</td><td rowspan=1 colspan=1>2.9 ± 12.8</td><td rowspan=1 colspan=1>44.4 ± 12.1</td><td rowspan=1 colspan=1>42.4 ± 12.9</td></tr><tr><td rowspan=1 colspan=7>DeepLOB-Att. 51.1 ± 5.5 47.0 ± 9.9 51.1 ± 5.5</td><td rowspan=1 colspan=1>48.5 ± 7.5</td></tr><tr><td rowspan=1 colspan=7>Linear SVM 43.6 ± 1.2 49.8 ± 2.8 43.6 ± 1.2</td><td rowspan=1 colspan=1>42.1 ± 3.5</td></tr><tr><td rowspan=1 colspan=8>MLP    44.3 ± 3.3 45.1 ± 2.8 44.3 ± 3.3 44.2 ± 3.5</td></tr><tr><td rowspan=2 colspan=7>LSTM   52.2 ± 0.4 51.9 ± 0.0 52.2 ± 0.4 550B(TABL)  54.2 ± 7.7 53.6 ± 7.3 54.2 ± 7.7 5</td><td rowspan=1 colspan=1>1.9 ± 0.1</td></tr><tr><td rowspan=1 colspan=1>3.7 ± 7.4</td></tr><tr><td rowspan=1 colspan=2>C(TABL)</td><td rowspan=1 colspan=3>63.8 ± 0.5 6</td><td rowspan=1 colspan=1>3.1 ± 0.6 6</td><td rowspan=1 colspan=1>3.8 ± 0.5 6</td><td rowspan=1 colspan=1>3.1 ± 0.8</td></tr><tr><td rowspan=1 colspan=8>BiN-C(TABL) 77.5 ± 0.4 78.0 ± 0.3 77.5 ± 0.4 77.3 ± 0.4</td></tr><tr><td rowspan=1 colspan=8>XGBoost 69.8 ± 0.0 69.4 ± 0.0 69.8 ± 0.0 69.2 ± 0.0</td></tr><tr><td rowspan=1 colspan=8>TensorLOB 77.5 ± 0.1 77.5 ± 0.1 77.5 ± 0.1 77.4 ± 0.1</td></tr><tr><td rowspan=1 colspan=8>DeepLOB 69.7 ± 0.5 69.9 ± 0.5 69.7 ± 0.5 69.6 ± 0.5</td></tr><tr><td rowspan=1 colspan=8>DeepLOB-Seq 48.5 ± 4.4 49.4 ± 4.3 48.5 ± 4.4 48.1 ± 5.1</td></tr><tr><td rowspan=1 colspan=8>DeepLOB-Att. 47.5 ± 10.4 46.8 ± 19.7 47.5 ± 10.4 42.8 ± 12.6</td></tr><tr><td rowspan=1 colspan=8>Linear SVM 42.0 ± 0.4 43.2 ± 0.1 42.0 ± 0.4 39.2 ± 1.1</td></tr><tr><td rowspan=1 colspan=8>MLP    43.9 ± 0.4 49.4 ± 0.6 43.9 ± 0.4 38.9 ± 1.1</td></tr><tr><td rowspan=1 colspan=8>LSTM   47.7 ± 0.5 48.1 ± 0.9 47.7 ± 0.5 47.4 ± 0.6100            48.2 ± 4.2 51.4 ± 5.7 48.2 ± 4.2 47.6 ± 4.1B(TABL)C(TABL)  63.2 ± 2.0 63.7 ± 1.7 63.2 ± 2.0 63.1 ± 2.2</td></tr><tr><td rowspan=1 colspan=8>BiN-C(TABL) 77.9 ± 0.2 78.0 ± 0.2 77.9 ± 0.2 78.0 ± 0.2</td></tr><tr><td rowspan=1 colspan=8>XGBoost 61.2 ± 0.0 65.3 ± 0.0 61.2 ± 0.0 60.8 ± 0.0</td></tr><tr><td rowspan=1 colspan=8>TensorLOB 79.1 ± 0.8 79.1 ± 0.8 79.1 ± 0.8 79.1 ± 0.8</td></tr></table>

## 5. CONCLUSION

We have introduced TensorLOB, a tensor structure-preserving model for limit-order-book prediction. Starting from first principles and domain knowledge of LOB mechanics, TensorLOB retains the natural time, side, level, and attribute modes and performs representation learning and prediction entirely in the tensor domain. By combining mode-specific tensor contractions with tensor classification, the model exploits the underlying multi-way structure while remaining parameter efficient. Across a range of FI-2010 prediction horizons and three random seeds, TensorLOB achieved the highest mean accuracy and weighted F1. As a proof of concept, these results opens up new perspective in financial modelling by resorting to the highdimensional modelling of multi-way latent information, native to financial markets, which has so far been largely overlooked.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This is a numerical simulation study for which no ethical approval was required.

## 7. REFERENCES

[1] Adamantios Ntakaris, Martin Magris, Juho Kanniainen, Moncef Gabbouj, and Alexandros Iosifidis, “Benchmark Dataset for Mid-price Forecasting of Limit Order Book Data with Machine Learning Methods,” Journal ofForecasting, vol. 37, no. 8, pp. 852–866, 2018.

[2] Zihao Zhang, Stefan Zohren, and Stephen Roberts, “DeepLOB: Deep Convolutional Neural Networks for Limit Order Books,” IEEE Transactions on Signal Processing, vol. 67, no. 11, pp. 3001–3012, 2019.

[3] Zihao Zhang and Stefan Zohren, “Multi-horizon Forecasting for Limit Order Books: Novel Deep Learning Approaches and Hardware Acceleration using Intelligent Processing Units,” arXiv preprint arXiv:2105.10430, 2021.

[4] Dat Thanh Tran, Alexandros Iosifidis, Juho Kanniainen, and Moncef Gabbouj, “Temporal Attention-augmented Bilinear Network for Financial Time-series Data Analysis,” IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 5, pp. 1407–1418, 2018.

[5] Dat Thanh Tran, Juho Kanniainen, Moncef Gabbouj, and Alexandros Iosifidis, “Data Normalization for Bilinear Structures in High-frequency Financial Time-series,” in 2020 25th International Conference on Pattern Recognition (ICPR). IEEE, 2021, pp. 7287–7292.

[6] Mingxue Xu, Yao Lei Xu, and Danilo P Mandic, “TensorGPT: Efficient Compression of Large Language Models Based on Tensor-Train Decomposition,” arXiv preprint arXiv:2307.00526, 2023.

[7] Yuxuan Gu, Wuyang Zhou, Giorgos Iacovides, and Danilo Mandic, “TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs,” IEEE International Joint Conference on Neural Networks (IJCNN 2025), 2025.

[8] Yiran Luo, Het Patel, Yu Fu, Dawon Ahn, Jia Chen, Yue Dong, and Evangelos E Papalexakis, “TRAWL: Tensor Reduced and Approximated Weights for Large Language Models,” in Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, 2025, pp. 402–413.

[9] Yifan Yang, Jiajun Zhou, Ngai Wong, and Zheng Zhang, “LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2024, pp. 3161–3176.

[10] Yuxuan Gu, Wuyang Zhou, Giorgos Iacovides, and Danilo Mandic, “TeRA: Vector-Based Random Tensor Network for High-Rank Adaptation of Large Language Models,” in Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 2314–2329.

[11] Wuyang Zhou, Yuxuan Gu, Giorgos Iacovides, Yuning Qiu, Qibin Zhao, and Danilo Mandic, “Tensorizing Engram: Sharing Latents Across N-Gram Embeddings is Beneficial in LLMs,” ICML 2026 workshop on Connecting Low-rank Representations in AI, 2026.

[12] Wuyang Zhou, Yuxuan Gu, Giorgos Iacovides, and Danilo Mandic, “KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices,” International Conference on Machine Learning (ICML), 2026.

[13] Yao Lei Xu, Kriton Konstantinidis, and Danilo P Mandic, “Graph-Regularized Tensor Regression: A Domain-Aware Framework for Interpretable Modeling of Multiway Data on Graphs,” Neural Computation, vol. 35, no. 8, pp. 1404–1429, 2023.

[14] Yao Lei Xu, Giuseppe G Calvi, and Danilo P Mandic, “Tensor-Train Recurrent Neural Networks for Interpretable Multi-Way Financial Forecasting,” in 2021 International Joint Conference on Neural Networks (IJCNN). IEEE, 2021, pp. 1–5.

[15] Jacobo Roa-Vicens, Yao Lei Xu, Ricardo Silva, and Danilo P Mandic, “Graph and Tensor-Train Recurrent Neural Networks For High-Dimensional Models of Limit Order Books,” in Proceedings of the Third ACM International Conference on AI in Finance, 2022, pp. 207–213.

[16] Andrzej Cichocki, Danilo Mandic, Lieven De Lathauwer, Guoxu Zhou, Qibin Zhao, Cesar Caiafa, and Huy Anh Phan, “Tensor Decompositions for Signal Processing Applications: From Two-Way to Multiway Component Analysis,” IEEE Signal Processing Magazine, vol. 32, no. 2, pp. 145–163, 2015.

[17] Dan Hendrycks and Kevin Gimpel, “Gaussian Error Linear Units (GELUs),” arXiv preprint arXiv:1606.08415, 2016.

[18] Geoffrey E Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever, and Ruslan R Salakhutdinov, “Improving Neural Networks by Preventing Co-Adaptation of Feature Detectors,” arXiv preprint arXiv:1207.0580, 2012.

[19] Ilya Loshchilov and Frank Hutter, “Decoupled Weight Decay Regularization,” arXiv preprint arXiv:1711.05101, 2017.

[20] Avraam Tsantekidis, Nikolaos Passalis, Anastasios Tefas, Juho Kanniainen, Moncef Gabbouj, and Alexandros Iosifidis, “Using Deep Learning to Detect Price Change Indications in Financial Markets,” in 2017 25th European Signal Processing Conference (EUSIPCO). IEEE, 2017, pp. 2511–2515.