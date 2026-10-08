# Second-order optimization of variable projection SVM models and road abnormality detection

Andrea Angino, Matthias Voigt, Rolf Krause, Tamas D ´ ozsa ´

Author Accepted Manuscript

This document is the accepted version of the manuscript and is not the final published version.

The final published article is available at:

https://doi.org/10.1109/ICASSP55912.2026.11463454.

Please cite the final published version when referring to this work:

A. Angino, M. Voigt, R. Krause, and T. Dozsa. Second-order optimization of variable projection SVM´ models and road abnormality detection. In ICASSP 2026 – 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1641–1645, Barcelona, Spain, 2026.

© 2026 IEEE.

# SECOND-ORDER OPTIMIZATION OF VARIABLE PROJECTION SVM MODELS AND ROAD ABNORMALITY DETECTION

Andrea Angino<sup>⋆</sup> Matthias Voigt<sup>⋆</sup> Rolf Krause<sup>†,⋆</sup> Tamas D´ ozsa´ <sup>⋆,‡</sup>

<sup>⋆</sup> Faculty of Mathematics and Computer Science, UniDistance Suisse, Brig, Switzerland <sup>†</sup> CEMSE Division, KAUST, Thuwal, Saudi Arabia

<sup>‡</sup> Eotv ¨ os Lor ¨ and University, Faculty of Informatics, Department of Numerical Analysis´

## ABSTRACT

We introduce a novel second-order optimization framework for minimizing so-called variable projection functionals. We demonstrate that the proposed framework is especially useful for the training of variable projection based kernel methods. In particular, the problem of efficiently training variable projection support vector machines (VP-SVMs) is considered. We show the effectiveness of the proposed training methodology in a real-world application, namely we demonstrate how second-order trust region algorithms can be used to train VP-SVM models to recognize road surface abnormalities based on 1D signals obtained from a tire sensor.

Index Terms— Variable projection, support vector machines, trust-region, intelligent tires

## 1. INTRODUCTION

Artificial intelligence (AI) methods have become a crucial component of many state-of-the-art signal processing algorithms in recent years. The main advantage of ML methods is their ability to approximate nonlinear operators in a purely data-driven manner. Unfortunately, ML methods, especially those that rely on deep learning, often suffer from a lack of explainability. This is mainly due to the size of current models: Deep neural networks often contain tens of millions of parameters [1]. The lack of interpretable parameters makes the deployment of ML models difficult in safety-critical applications such as fault detection in critical infrastructure.

Although many papers [1–5] focus on introducing new transparent AI architectures to mitigate these problems, training methods that exploit the structural properties associated with them are less frequent. In this work, we demonstrate that second-order trust-region (TR) methods can be an effective tool to optimize the parameters of so-called variable projection support vector machines (VP-SVMs), see [4]. VP-SVMs were recently introduced in [4] to improve the interpretability and performance of classical SVM classifiers. VP-SVMs contain interpretable parameters and therefore fit well with the transparent AI paradigm. The key component of a VP-SVM model is an adaptive feature transformation step, which transforms the data samples to be classified to a lowdimensional space. The (physically meaningful) parameters of this transformation are optimized together with the weights of the SVM classifier.

SVMs (and kernel methods in general) are difficult to train on large datasets. Indeed, the associated kernel matrix scales with the number of training signals [4,6]. In the case of VP-SVM, this problem is even more significant since the kernel also depends on learnable parameters. In order to make the training problem numerically tractable for VP-SVM, in [4], a reduced objective is proposed together with the use of stochastic subgradient methods. This, however, may lead to suboptimal classification and a lack of interpretability.

In this paper, we propose second-order training methods to address these challenges. In particular, we show that the training time of a VP-SVM model that can recognize road surface abnormalities (such as potholes or bumps), see [7, 8], can be greatly reduced with a properly chosen second-order training scheme. In fact, as our experiments show, the generalization abilities of the classifier will improve significantly, despite the reduction to the cost of training.

In particular, we derive analytic formulas for the secondorder derivatives of Moore-Penrose pseudo-inverses and variable projection operators as well as Hessians of primal quadratic VP-SVM objectives. Then, we show how secondorder optimization schemes can be used to efficiently train VP-SVM classifiers to recognize road surface abnormalities. For reproducibility, we refer to our implementation [9] that also contains comparisons between different training schemes. Although our results are stated for VP-SVM, they can be extended to any kernel-based models, where the kernels depend on learnable parameters.

## 2. VARIABLE PROJECTION OPERATORS

First, we consider separable nonlinear least squares (SNLLS) problems, VP operators, and their connection to feature extraction in ML. Let H be a Hilbert space of functions X → R with $X \subseteq \mathbb { R }$ . Suppose $\{ \phi _ { \pmb { \eta } } ^ { 1 } , \dots , \phi _ { \pmb { \eta } } ^ { m } \}$ is a linearly independent system in H for all parameter vectors $\pmb { \eta } \in \mathbb { R } ^ { p }$ . Henceforth, we shall consider the vectors $\varphi _ { k } ( \pmb { \eta } ) : = \left[ \phi _ { \pmb { \eta } } ^ { k } ( x _ { \ell } ) \right] _ { \ell = 1 } ^ { n } \in$ $\mathbb { R } ^ { n }$ where $x _ { \ell } \in X$ for $\ell = 1 , \ldots , n$ with $x _ { 1 } < . . . < x _ { n } . { \mathrm { ~ W e } }$ shall assume that the sampling points $x _ { \ell }$ are chosen such that the vectors $\left\{ \varphi _ { k } ( \eta ) \right\} _ { k = 1 } ^ { m }$ remain linearly independent in $\mathbb { R } ^ { n }$

Since $\varphi _ { k } ( \pmb { \eta } ) , k = 1 , \ldots$ , m are linearly independent, they span an m-dimensional subspace $G ( \pmb { \eta } ) : = \mathrm { s p a n } \{ \varphi _ { k } ( \pmb { \eta } ) \quad :$ $k = 1 , \ldots , m \} \subset \mathbb { R } ^ { n }$ . It is well known (see, e.g., [10]) that for a fixed η there exists ${ \widehat { \pmb f } } \in G ( \pmb \eta )$ , such that $\| { \pmb f } - { \widehat { \pmb f } } \| _ { 2 }$ is minimal. In fact, with the standard Euclidean inner product $\langle \cdot , \cdot \rangle , \langle f - { \widehat { f } } , g \rangle = 0 ( g \in G ( \eta ) )$ holds, that is, $\widehat { \pmb f }$ is the orthogonal projection of f onto $G ( \eta )$ . The projection operator $P _ { \eta } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ which satisfies $P _ { \eta } f = { \widehat { f } }$ can be expressed by

$$
P _ { \eta } f : = \Phi ( \eta ) \Phi ( \eta ) ^ { + } f ,
$$

where $\Phi ( \pmb { \eta } ) : = [ \varphi _ { 1 } ( \pmb { \eta } ) , \dots , \varphi _ { m } ( \pmb { \eta } ) ] \ \in \ \mathbb { R } ^ { n \times m }$ and $\Phi ( \eta ) ^ { + }$ denotes its Moore-Penrose pseudoinverse. Because this projection depends on η, $P _ { \eta }$ is often referred to as a variable projection (VP) operator. In addition to $P _ { \eta } ,$ we will make use of the feature transformation $C _ { \eta } : \mathbb { R } ^ { n }  \dot { \mathbb { R } } ^ { m }$ defined by

$$
C _ { \eta } f = \Phi ( \eta ) ^ { + } f ,\tag{1}
$$

which provides the coordinates of $\widehat { \pmb f }$ with respect to the basis $\{ \varphi _ { 1 } ( \pmb { \eta } ) , \dotsc , \varphi _ { m } ( \pmb { \eta } ) \}$ . This transformation can be interpreted as a dimension reduction and feature extraction step. If some a priori information is known about the behavior of ${ \textbf { \textit { f } } } ( { \mathrm { e . g . } }$ quasi compactness, quasi periodicity, etc.) then the functions $\phi _ { \pmb { \eta } } ^ { 1 } , \dots , \phi _ { \pmb { \eta } } ^ { m }$ can be chosen so that their parameter $\boldsymbol { \eta } ~ \in ~ \mathbb { R } ^ { p }$ contains physically meaningful information $( \sec , \mathsf { e . g . , } [ 3 - 5 ] )$ Given such a family, it is reasonable to ask, if any η parameter minimizes

$$
\varrho ( \pmb { \eta } ) : = \| \pmb { f } - P _ { \pmb { \eta } } \pmb { f } \| _ { 2 } ^ { 2 } .\tag{2}
$$

Minimizing the functional $\varrho$ in Eq. (2) is also known as a SNLLS problem. In [11], Golub and Pereyra provide analytic formulas for the Jacobian of $\Phi ( \eta ) ^ { + }$ and $P _ { \eta } .$ . Thanks to their contribution, minimizing Eq. (2) can be done with gradientbased optimization schemes. This gives rise to a large number of applications (see, e.g., [3] and [12]).

## 3. VP-SVM MODELS

Consider the dataset $\mathcal { X } \ : = \ \{ f _ { 1 } , \dotsc , f _ { q } \} \ \subset \ \mathbb { R } ^ { n }$ and its labeled version $\mathcal { D } \ : = \ \{ ( f _ { 1 } , y _ { 1 } ) , \dotsc , ( \dot { f } _ { q } , y _ { q } ) \} \ \subset \ \mathbb { R } ^ { n } \ \times$ $\{ - 1 , 1 \}$ Our objective is to identify a nonlinear SVM $\mathcal { F } ( \beta ; \cdot )$ $\begin{array} { r } { \mathbb { R } ^ { n } \ \to \ \{ - 1 , 1 \} , \ \mathcal { F } ( \beta ; \pmb { f } ) : = \ \mathrm { s g n } \left( \sum _ { j = 1 } ^ { q } \beta _ { j } K ( \pmb { f } , \pmb { f } _ { j } ) \right) } \end{array}$ where $\beta : = ( \beta _ { 1 } , \ldots , \beta _ { q } ) ^ { \top } \in \mathbb { R } ^ { q }$ and $K ( \cdot , \cdot )$ is a reproducing kernel of a reproducing kernel Hilbert space (RKHS) [13]. The purpose of $K$ is to lift the data samples $\mathbf { \Psi } _ { { \mathbf { { f } } _ { j } } } ~ \in ~ \mathcal { X }$ $j = 1 , \dotsc , q$ to a higher dimensional space, where a separating hyperplane between the classes, defined by $\beta ,$ might be found. In this work, following the reasoning in [4], we restrict our investigation to the radial basis function (RBF) kernel defined by $K ( \pmb { f } , \pmb { g } ) : = \mathrm { e } ^ { - \| \pmb { f } - \pmb { g } \| _ { 2 } ^ { 2 } / \sigma ^ { 2 } } \left( \pmb { f } , \pmb { g } \in \mathbb { R } ^ { n } \right)$ for some $\sigma \in \mathbb { R } \backslash \{ 0 \}$ . It is well-known [14] that an appropriately chosen feature extraction transformation can significantly improve classification performance. In this work we will consider the transformation $C _ { \eta } f _ { : }$ , as defined in (1), which is learned jointly with the weights of the underlying classifier, i.e.,

$$
\begin{array} { r l } & { \mathcal { F } ( \beta , \eta ; f ) = \mathrm { s g n } \left( \displaystyle \sum _ { j = 1 } ^ { q } \beta _ { j } K \left( C _ { \eta } f , C _ { \eta } f _ { j } \right) \right) } \\ & { ( \beta \in \mathbb { R } ^ { q } , \eta \in \mathbb { R } ^ { p } , f \in \mathbb { R } ^ { n } , \pmb { f } _ { j } \in \mathcal { X } ) . } \end{array}\tag{3}
$$

The goal of training a VP-SVM classifier is to find appropriate parameters η and $\beta$ for the model defined in Eq. (3). This can be done, for example (see [4]), by solving

$$
\begin{array} { r l } & { \underset { \beta \in \mathbb { R } ^ { q } , \eta \in \mathbb { R } ^ { p } } { \operatorname* { m i n } } e ( \beta , \eta ) } \\ & { = \underset { \beta \in \mathbb { R } ^ { q } , \eta \in \mathbb { R } ^ { p } } { \operatorname* { m i n } } \lambda _ { 0 } R ( \eta ) + \lambda _ { 1 } A ( \beta , \eta ) + \lambda _ { 2 } B ( \beta , \eta ) , } \end{array}\tag{4}
$$

where

$$
\begin{array} { r l r } {  { R ( \eta ) : = \frac { 1 } { q } \sum _ { j = 1 } ^ { q } \frac { \| f _ { j } - P _ { \eta } f _ { j } \| _ { 2 } ^ { 2 } } { \| f _ { j } \| _ { 2 } ^ { 2 } } , } } \\ & { } & { A ( \beta , \eta ) : = \beta ^ { \top } K ( \eta ) \beta , \ K ( \eta ) : = \big [ K ( C _ { \eta } f _ { i } , C _ { \eta } f _ { j } ) \big ] _ { i , j = 1 } ^ { q } , \ } \\ & { } & { B ( \beta , \eta ) : = \frac { 1 } { q } \displaystyle \sum _ { j = 1 } ^ { q } ( \operatorname* { m a x } \big ( 0 , 1 - y _ { j } ( \beta ^ { \top } K ( \eta ) _ { j } \big ) ) ^ { 2 } , \ ~ } \end{array}
$$

and where $\lambda _ { 0 } , \lambda _ { 1 } , \lambda _ { 2 } > 0$ are weighting constants for the different terms, and $K _ { j }$ denotes the j-th column of the matrix $K$ . The quadratic loss term B defines a classical SVM and is twice differentiable, except for the case where a sample is exactly on the separating hyperplane.

## 4. SECOND-ORDER DIFFERENTIATION OFVP-OPERATORS

In our training approach, we will incorporate curvature information, we use first and second-order derivatives of the objective functional in Eq. (4). While in practice automatic differentiation tools can produce these, we now proceed with providing the required formulas, focusing first on the regularization term $R ( \eta )$ in Eq. (4). The gradient of this term can be found in [3, p. 5, Eq. (9)], while its Hessian is given by

$$
\begin{array} { c } { \displaystyle \frac { \partial ^ { 2 } R ( \pmb { \eta } ) } { \partial \eta _ { j } \partial \eta _ { i } } = \frac { 1 } { q } \sum _ { k = 1 } ^ { q } \frac { 2 } { \| \pmb { f } _ { k } \| _ { 2 } ^ { 2 } } \left( \left. \frac { \partial P _ { \pmb { \eta } } \pmb { f } _ { k } } { \partial \eta _ { j } } , \frac { \partial P _ { \pmb { \eta } } \pmb { f } _ { k } } { \partial \eta _ { i } } \right. - \right. } \\ { \displaystyle \left. \left. \pmb { f } _ { k } - P _ { \pmb { \eta } } \pmb { f } _ { k } , \frac { \partial ^ { 2 } P _ { \pmb { \eta } } \pmb { f } _ { k } } { \partial \eta _ { j } \partial \eta _ { i } } \right. \right) \ ~ ( i , j = 1 , \ldots , p ) . } \end{array}\tag{5}
$$

Notice that first- and second-order derivatives of the variable projection operator $P _ { \eta }$ are needed to evaluate Eq. (5). The

first-order derivatives can be found in [4], while an elementary calculation shows that for $i , j = 1 , \dotsc , p$

$$
\begin{array} { r l r } {  { \frac { \partial ^ { 2 } P _ { \eta } } { \partial \eta _ { j } \partial \eta _ { i } } = \frac { \partial P _ { \eta } ^ { \perp } } { \partial \eta _ { j } } \frac { \partial \Phi ( \eta ) } { \partial \eta _ { i } } \Phi ( \eta ) ^ { + } + P _ { \eta } ^ { \perp } \frac { \partial ^ { 2 } \Phi ( \eta ) } { \partial \eta _ { j } \partial \eta _ { i } } \Phi ( \eta ) ^ { + } } } \\ & { } & { + P _ { \eta } ^ { \perp } \frac { \partial \Phi ( \eta ) } { \partial \eta _ { i } } \frac { \partial \Phi ( \eta ) ^ { + } } { \partial \eta _ { j } } + [ \frac { \partial P _ { \eta } ^ { \perp } } { \partial \eta _ { j } } \frac { \partial \Phi ( \eta ) } { \partial \eta _ { i } } \Phi ( \eta ) ^ { + }  } \\ & { } & {  + P _ { \eta } ^ { \perp } \frac { \partial ^ { 2 } \Phi ( \eta ) } { \partial \eta _ { j } \partial \eta _ { i } } \Phi ( \eta ) ^ { + } + P _ { \eta } ^ { \perp } \frac { \partial \Phi ( \eta ) } { \partial \eta _ { i } } \frac { \partial \Phi ( \eta ) ^ { + } } { \partial \eta _ { j } } ] ^ { \top } , } \end{array}
$$

where $P _ { \eta } ^ { \bot } : = I - P _ { \eta }$ denotes the orthogonal complement operator of $P _ { \eta }$ and its derivative is given by $\begin{array} { r } { \frac { \partial P _ { \eta } ^ { \bot } } { \partial \eta _ { j } } = - \frac { \partial P _ { \eta } } { \partial \eta _ { j } } } \end{array}$ accordingly. Finally, to fully express the Hessian of $R ( \pmb \eta )$ we need to provide the second-order derivatives of the pseudoinverse $\Phi ( \pmb { \eta } ) ^ { + }$ . Like before, the first-order formulas can be found in the literature (see, e.g., [6]), while the corresponding second partial derivatives for $i , j = 1 , \dotsc , p$ are given by

$$
\begin{array} { r l } & { \frac { \partial ^ { 2 } \hat { \mathcal { A } } ( x , y ) } { \partial \hat { \mathcal { C } } ( y , \hat { \mathcal { H } } ) } = - \frac { \partial ^ { 2 } \hat { \mathcal { A } } ( x , y ) } { \partial \hat { \mathcal { C } } ( y , \hat { \mathcal { H } } ) } - \frac { \partial ^ { 2 } \hat { \mathcal { A } } ( y , z ) } { \partial \hat { \mathcal { C } } ( y , \hat { \mathcal { H } } ) } } \\ & { \quad - \frac { \partial ^ { 2 } \hat { \mathcal { A } } ( x , y ) } { \partial \hat { \mathcal { C } } ( y , \hat { \mathcal { H } } ) } \frac { \partial ^ { 2 } \hat { \mathcal { A } } ( x , y ) } { \partial \hat { \mathcal { C } } ( y , \hat { \mathcal { H } } ) } = \hat { \mathcal { C } } ( \hat { \mathcal { H } } _ { 1 } ) \frac { \partial ^ { 2 } \hat { \mathcal { A } } ( x , y ) } { \partial \hat { \mathcal { C } } ( y , \hat { \mathcal { H } } ) } } \\ & { \quad + \frac { \partial ^ { 2 } \hat { \mathcal { A } } ( x , y ) } { \partial \hat { \mathcal { C } } ( y , \hat { \mathcal { H } } ) } \frac { \partial ^ { 2 } \hat { \mathcal { A } } ( x , y ) } { \partial \hat { \mathcal { C } } ( y , \hat { \mathcal { H } } ) } , \quad \hat { \mathcal { H } } _ { 1 } ^ { \hat { \mathcal { A } } } = \frac { \partial ^ { 2 } \hat { \mathcal { A } } ( x , y ) } { \partial \hat { \mathcal { C } } ( y , \hat { \mathcal { H } } ) } } \\ & { \quad + \frac { \partial ^ { 2 } \hat { \mathcal { A } } ( x , y ) } { \partial \hat { \mathcal { C } } ( y , \hat { \mathcal { H } } ) } \frac { \partial ^ { 2 } \hat { \mathcal { A } } ( x , y ) } { \partial \hat { \mathcal { C } } ( y , \hat { \mathcal { H } } ) } } \\ &  \quad + \frac { \partial ^ { 2 } \hat { \mathcal { A } } ( x , y ) }  \partial \ \end{array}
$$

where $\begin{array} { r } { \frac { \partial ^ { 2 } \Phi ( \eta ) } { \partial \eta _ { j } \partial \eta _ { i } } , i , j = 1 \ldots , p , } \end{array}$ are assumed to exist and be known for a given function system which defines $\Phi ( \eta )$

The partial derivatives of A and B from Eq. (4) also need to be determined. For first-order and second-order derivatives with respect to $\beta ,$ we refer to [6]. Assuming that the kernel $K ( \eta )$ is twice differentiable, for $i = 1 , \ldots , p$ we obtain $\begin{array} { r } { \frac { \partial B ( \beta , \eta ) } { \partial \eta _ { i } } = - \frac { 2 } { q } \sum _ { \ell = 1 } ^ { q } } \end{array}$ <sub>1</sub> max $\begin{array} { r } { \left( 0 , 1 - y _ { \ell } \beta ^ { \top } K ( \pmb { \eta } ) _ { \ell } \right) y _ { \ell } \frac { \partial K ( \pmb { \eta } ) } { \partial \eta _ { i } } \beta . } \end{array}$ The second-order partial derivatives of $B ( \beta , \dot { \eta } )$ for $i , j \ =$

$1 , \ldots , p$ are then given by

$$
\begin{array} { r l r } {  { \frac { \partial ^ { 2 } B ( \boldsymbol { \beta } , \boldsymbol { \eta } ) } { \partial \eta _ { j } \partial \eta _ { i } } = - \frac { 2 } { q } \sum _ { \ell = 1 } ^ { q } [ - \operatorname* { m a x } ( 0 , 1 - y _ { \ell } \boldsymbol { \beta } ^ { \top } \frac { \partial K ( \boldsymbol { \eta } ) _ { \ell } } { \partial \eta _ { j } } ) y _ { \ell }  } } \\ & { } & { \times  \frac { \partial K ( \boldsymbol { \eta } ) } { \partial \eta _ { i } } \beta + \operatorname* { m a x } ( 0 , 1 - y _ { \ell } \boldsymbol { \beta } ^ { \top } K ( \boldsymbol { \eta } ) _ { \ell } ) y _ { \ell } \frac { \partial ^ { 2 } K ( \boldsymbol { \eta } ) } { \partial \eta _ { j } \partial \eta _ { i } } \beta ] . } \end{array}
$$

Furthermore, for $i , j = 1 , \dotsc , p$ we obtain

$$
\frac { \partial A ( \beta , \eta ) } { \partial \eta _ { i } } = \beta ^ { \top } \frac { \partial K ( \eta ) } { \partial \eta _ { i } } \beta , ~ \frac { \partial ^ { 2 } A ( \beta , \eta ) } { \partial \eta _ { j } \partial \eta _ { i } } = \beta ^ { \top } \frac { \partial ^ { 2 } K ( \eta ) } { \partial \eta _ { j } \partial \eta _ { i } } \beta .
$$

Finally, the mixed derivatives are expressed by

$$
\begin{array} { l } { { \displaystyle { \frac { \partial ^ { 2 } e ( \beta , \eta ) } { \partial \eta \partial \beta } = 2 \lambda _ { 1 } \frac { \partial K ( \eta ) } { \partial \eta } \beta } } } \\ { { \displaystyle ~ + \frac { 2 \lambda _ { 2 } } { q } \left( \frac { \partial K ( \eta ) } { \partial \eta } \Pi ( K ( \eta ) \beta - y ) + K ( \eta ) \Pi \frac { \partial K ( \eta ) } { \partial \eta } \beta \right) } } \end{array}\tag{6}
$$

In Eq. (6), we assume that the first $r \in \mathbb N$ entries of the label vector $\pmb { y } = ( y _ { 1 } , \dots , y _ { q } ) ^ { \top }$ are exactly the ones for which $1 -$ $y _ { i } \beta ^ { \top } K ( \pmb { \eta } ) _ { i } > 0 .$ . Furthermore, we define the diagonal matrix $\Pi \in \mathbb { R } ^ { q \times q }$ <sup>q</sup> by $\Pi : = \operatorname { d i a g } ( \pi _ { 1 } , \ldots , \pi _ { q } )$ , where

$$
\pi _ { k } = { \left\{ \begin{array} { l l } { 1 , } & { k \leq r } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. } , \quad k = 1 , \ldots , q .
$$

## 5. EXPERIMENTS

We now show how second-order information can be used to train a VP-SVM classifier to recognize road surface abnormalities. We rely on the dataset introduced in [7]. The dataset contains a total of $q = 5 1 7 1 \mathrm { D }$ signals recorded on the public road network of Budapest. The signals correspond to voltage changes in a tire-implanted force sensor and thus to tire deformation. The signals have been segmented into single tire revolutions and zero-padded, producing compactly supported waveforms of equal length. For information about measurements and signal characteristics, we refer to [7].

The first few adaptive Hermite functions [15] can be used to effectively approximate the tire signals [7]. For this reason, we choose $\varphi _ { k } ( \pmb { \eta } ) , k = 1 , \dots , m$ as the discrete, equidistant sampling of the k-th adaptive Hermite function [5] defined by

$$
\begin{array} { r } { \varphi _ { k } ( \eta ) = \left[ \sqrt { \frac { s } { \pi ^ { 1 / 2 } 2 ^ { k } k ! } } h ^ { k } ( s ( x _ { \ell } - t ) ) \mathrm { e } ^ { - ( s ( x _ { \ell } - t ) ) ^ { 2 } / 2 } \right] _ { \ell = 1 } ^ { n } } \\ { ( s > 0 , x _ { \ell } , t \in \mathbb { R } , \eta : = ( s , t ) ^ { \top } \in \mathbb { R } ^ { 2 } ) , } \end{array}
$$

where $h ^ { k }$ denotes the k-th classical Hermite polynomial [16]. Importantly, for adaptive Hermite functions, the existence of a global solution for minimizing Eq. (2) is proven in [5]. Finally, we note that the dilation parameter s can be used to model the support of the signals and thus vehicle speed, making the representation interpretable. In [7], a total of 11 adaptive Hermite functions are proposed to obtain the features. VP-based neural networks then achieve an average of 97% accuracy on the dataset. In contrast, we use the much simpler VP-SVM model and represent the signals using only 5 Hermite-Fourier coefficients. Regardless, as seen in Table 1, our proposed training scheme is able to achieve performance close to the state-of-the-art.

In particular, we compare the performance of gradient descent (GD) and the TR method proposed in Algorithm 6.1.1 of [17] for training VP-SVM (see Eq. (4)). The GD method uses first-order information only. Note that performing a line search along the negative gradient direction would be prohibitive due to the multiple kernel evaluations required at each iteration. Therefore, we fix a small learning rate of $1 0 ^ { - 3 } .$ , as commonly done in stochastic gradient methods. In contrast, the TR approach leverages the second-order derivative discussed earlier to define the model and determine the search direction as an approximate solution to the trust region subproblem (using the Steihaug-Toint method, as outlined in [18]). To the best of our knowledge, this is the first time secondorder methods have been used to train a kernel-based classifier, where the kernel also depends on learnable parameters. A key advantage of TR methods is that when a step is rejected, re-evaluating the Hessian along with 1st and 2nd kernel derivative evaluations is not required. Another advantage over approaches that rely on multiple objective function evaluations is that, in the case of a rejected step, TR only requires a single kernel evaluation per iteration.

All experiments are performed with initial weights $\beta = 0$ and parameters $\pmb { \eta } = ( s , \bar { t } ) ^ { \top } = ( 1 0 , 0 ) ^ { \top }$ for dilation and translation, fixed regularization parameters $\lambda _ { 0 } = 2 , \lambda _ { 1 } = 0 . 1$ , and $\lambda _ { 2 } ~ = ~ 1 . ^ { 1 }$ From Figure 1, we observe that the TR method achieves a faster reduction in the loss and higher accuracy compared to GD on both training and test sets. The bottom plot confirms that the TR method effectively reduces the gradient norm, whereas GD exhibits slower decay.

Table 1: Summary of VP-SVM classification performance.
<table><tr><td colspan="5">Alg. #It Avg Acc. Min Acc. Max Acc. ∥∇e∥ Time (s)</td></tr><tr><td>TR</td><td>15 25</td><td>95.5% 96.3%</td><td>92.2% 93.2%</td><td>98.1% 9.16e-04</td><td>1.73</td></tr><tr><td>GD</td><td></td><td></td><td>68.0%</td><td>98.1% 2.24e-04</td><td>5.39</td></tr><tr><td></td><td>25 125</td><td>72.3% 83.9%</td><td>79.6%</td><td>77.9% 1.57e-02</td><td>0.33</td></tr><tr><td></td><td>250</td><td>88.2%</td><td>85.4%</td><td>88.5% 7.52e-03 93.3% 4.82e-03</td><td>1.68 3.39</td></tr><tr><td></td><td>500</td><td>91.3%</td><td>89.3%</td><td>95.2% 2.10e-03</td><td>6.78</td></tr><tr><td></td><td>1000</td><td>92.1%</td><td>90.3%</td><td>96.2% 8.24e-04</td><td>13.63</td></tr></table>

<sup>1</sup>All experiments were conducted on a workstation with an Intel(R) Xeon(R) W-2123 CPU @ 3.60GHz, 64 GB RAM, and Python 3.10.12 with Pytorch 2.4.0 using automatic differentiation. All models were trained using the CPU.

![](images/3fa9a5af2083440856850a93eaecfa443a2962afb9c4a3a05d4dc4a13ab90d79.jpg)  
Fig. 1: Top and middle: Objective function and accuracy for both TR and GD on training and test sets. Bottom: Gradient norm evolution over time for TR and GD on the training set.

Table 1 reports the classification performance obtained by training VP-SVM. Both the GD and TR solvers are evaluated using a 5-fold cross-validation. The table summarizes the mean gradient norms ∥∇e∥ and accuracies, together with the minimum and maximum values across the different tests. The results show that TR consistently outperforms GD, achieving higher accuracy with substantially fewer iterations and time. TR reaches a given performance level much faster, whereas GD requires more than two times the training time and still fails to achieve TR’s accuracy.

## 6. CONCLUSION

In this paper, we proposed second-order training schemes for VP-SVM classifiers. We provided the necessary analytic formulas and demonstrated the effectiveness of TR training schemes using second-order information.

## DATA AND CODE AVAILABILITY

Data and code to reproduce the results are available in [9].

## ACKNOWLEDGMENT

The research of TD has received funding from the Swiss Government Excellence Scholarship No. 2025.0057. The research of AA was funded in part by the Swiss National Science Foundation (SNSF) grant No. 224943. The project was supported by the University Excellence Fund of Eotv ¨ os¨ Lorand University, Budapest, Hungary (ELTE). Project no.´ K146721 has been implemented with the support provided by the Ministry of Culture and Innovation of Hungary from the National Research, Development and Innovation Fund, financed under the $\mathrm { K } \_ { 2 3 } \cdot \mathrm { \cup T K A } ^ { \prime \prime }$ funding schemes.

## 7. REFERENCES

[1] Z. C. Lipton, “The mythos of model interpretability: In machine learning, the concept of interpretability is both important and slippery.,” Queue, vol. 16, no. 3, pp. 31– 57, 2018.

[2] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and Liu L. Yang, “Physics-informed machine learning,” Nature Reviews Physics, vol. 3, no. 6, pp. 422–440, 2021.

[3] P. Kovacs, G. Bogn´ ar, C. Huber, and M. Huemer, “VP-´ NET: Variable projection networks,” International Journal of Neural Systems, vol. 32, no. 01, pp. 2150054, 2022.

[4] T. Dozsa, F. Deuschle, B. Cornelis, and P. Kov´ acs,´ “Variable projection support vector machines and some applications using adaptive Hermite expansions,” International Journal ofNeural Systems, vol. 34, no. 01, pp. 2450004, 2024.

[5] T. Dozsa, C. B ´ ock, J. Meier, and P. Kov ¨ acs, “Weighted´ Hermite variable projection networks for classifying visually evoked potentials,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 7, pp. 12415–12428, 2025.

[6] O. Chapelle, “Training a support vector machine in the primal,” Neural Computation, vol. 19, no. 5, pp. 1155– –1178, 2007.

[7] T. Dozsa, J. Rad´ o, J. Volk,´ A. Kisari, A. Soumelidis, and<sup>´</sup> P. Kovacs, “Road abnormality detection using piezore-´ sistive force sensors and adaptive signal models,” IEEE Transactions on Instrumentation and Measurement, vol. 71, pp. 1–11, 2022.

[8] T. Dozsa, V. Jurdana, S. B. Segota, J. Volk, J. Rad´ o,´ Alexandros A. Soumelidis, and P. Kovacs, “Road type´ classification using time-frequency representations of tire sensor signals,” IEEE Access, vol. 12, pp. 53361– 53372, 2024.

[9] A. Angino, M. Voigt, R. Krause, and T. Dozsa, “Second-´ order optimization of variable projection support vector machine models and road abnormality detection: implementation,” 2025, doi: 10.5281/zenodo.17129636.

[10] P. P. Petrushev and V. A. Popov, Rational Approximation of Real Functions, vol. 28 of Encyclopedia of Mathematics and its Applications, Cambridge University Press, Cambridge, UK, 1988.

[11] G. H. Golub and V. Pereyra, “The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate,” SIAM Journal on Numerical Analysis, vol. 10, no. 2, pp. 413–432, 1973.

[12] G. H. Golub and V. Pereyra, “Separable nonlinear least squares: the variable projection method and its applications,” Inverse Problems, vol. 19, no. 2, pp. R1, 2003.

[13] M. O. Stitson, J. A. E. Weston, A. Gammerman, V. Vovk, and V. Vapnik, “Theory of support vector machines,” Tech. Rep. CSD-TR-96-19, University of London, Department of Computer Science, Dec. 1996, Available at https://yu.ac.kr/ <sub>˜</sub>shkwon/lectures/ic/svm/svm\_1.pdf.

[14] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning, Adaptive Computation and Machine Learning Series. MIT Press, Cambridge, MA, USA, 2016.

[15] T. Dozsa and P. Kov ´ acs, “ECG signal compression using ´ adaptive Hermite functions,” in ICT Innovations 2015, Cham, Switzerland, 2016, pp. 245–254, Springer International Publishing.

[16] G. Szego,˝ Orthogonal Polynomials, vol. 23 of Colloquium Publications, AMS, Providence, RI, USA, 4th edition, 1975.

[17] A. R. Conn, N. I. M. Gould, and P. L. Toint, Trust Region Methods, MPS-SIAM Series on Optimization. SIAM, Philadelphia, PA, USA, 2000.

[18] J. B. Erway, P. E. Gill, and J. D. Griffin, “Iterative methods for finding a trust-region step,” SIAM Journal on Optimization, vol. 20, no. 2, pp. 1110–1131, 2009.