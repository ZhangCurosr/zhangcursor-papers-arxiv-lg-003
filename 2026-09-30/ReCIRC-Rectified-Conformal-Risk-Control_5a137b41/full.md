# ReCIRC: Rectified Conformal Risk Control

Bruno Marcondes e Resende<sup>∗</sup>

Helton Graziadei

Thiago Rodrigo Ramos<sup>†</sup>

Rafael Izbicki<sup>‡</sup>

Department of Statistics, Federal University of São Carlos

September 30, 2026

## Abstract

Many applications of black-box predictive models require controlling task-relevant error rates, such as missed lesion pixels in segmentation or missed labels in multilabel classification. Conformal risk control (CRC; Angelopoulos et al. 2024) gives distribution-free guarantees for such losses, but it calibrates a single threshold shared by all inputs. Because conditional risk varies with the input, this marginal guarantee often overprotects easy cases and underprotects hard ones. We propose ReCIRC (Rectified Conformal Risk Control), which inverts each input’s estimated local risk curve to reparameterize the calibrated threshold as a risk budget a representing a common target conditional risk, and then applies CRC unchanged to the resulting family. ReCIRC retains CRC’s finite-sample marginal guarantee regardless of the accuracy of the estimated curves, while accurate curves yield approximate conditional risk control and, under additional conditions, asymptotically exact conditional risk control; they also support a risk-calibration diagnostic. Across three synthetic and five real-data settings spanning segmentation, multilabel and multiclass classification, and regression, ReCIRC attained the lowest average worst-group risk and mean positive group excess in every setting, while maintaining marginal risk close to the target, whereas changes in prediction size were application-dependent.

Keywords: conformal prediction; conformal risk control; distribution-free inference; conditional risk control; adaptive calibration

## 1 Introduction

Black-box predictive models are now embedded in many consequential scientific and clinical workflows (Wang et al. 2023, Topol 2019); examples include segmentation networks that delineate polyps in colonoscopy images, multilabel and hierarchical classifiers, and opendomain question-answering systems (Fan et al. 2020, Lin et al. 2014, Deng et al. 2009, Kwiatkowski et al. 2019). In these applications, point predictions alone are not enough. Practitioners need guarantees that errors of specific, task-relevant kinds, such as missed lesion pixels, missed labels, semantically distant predictions, occur at a controlled rate.

Conformal prediction provides such guarantees for the miscoverage of prediction sets: wrapped around any predictive model, it produces sets that contain the true response with a prescribed probability, without assumptions on the data-generating distribution beyond exchangeability (Vovk et al. 2005, Lei et al. 2018, Angelopoulos & Bates 2023). Conformal risk control (CRC; Angelopoulos et al. 2024) extends this idea from miscoverage to arbitrary bounded losses that shrink as predictions become more protective: given a family of decision rules $\{ f _ { \lambda } : \lambda \in \Lambda \}$ ordered so that larger λ yields a smaller loss $\ell ( f _ { \lambda } ( x ) , y )$ , CRC calibrates a data-driven threshold $\widehat { \lambda }$ such that the expected loss of $f _ { \widehat { \lambda } }$ on a new covariate–response pair (X, Y ) is at most a target level α. This covers, among others, the false-negative rate of segmentation masks and label sets, graph-distance losses in hierarchical classification, and F1-based losses in question answering.

Despite its generality, CRC calibrates a single global parameter shared by every input. The core dificulty is that a raw threshold is not a risk scale: the conditional risk $R ( \lambda \mid x ) = \mathbb { E } { \bigl \{ } \ell ( f _ { \lambda } ( X ) , Y ) \mid X = x { \bigr \} }$ typically varies with x, so the same λ that yields nearly zero conditional risk on easy inputs can leave hard inputs far above the target level. Because the guarantee averages over the covariate distribution, it can overprotect easy inputs while leaving dificult inputs above the target.

To address this limitation, we propose risk rectification. The idea is to keep the CRC calibration step untouched but change the parameter on which it operates: instead of calibrating the raw threshold λ, we first convert the original family into a new family indexed by a risk budget a, obtained by inverting each input’s local risk curve. In the oracle version, the same value of a has the same interpretation at every input x: it is the desired conditional expected loss. In practice, the local risk curves are estimated, and ordinary CRC is applied to the resulting reparameterized family to ensure distribution-free guarantees. We call the resulting framework ReCIRC (Rectified Conformal Risk Control). The CRC guarantee does not require the estimated risk curves to be correct: poor estimates can reduce local adaptivity, but they cannot invalidate finite-sample marginal risk control. Accurate estimates, in turn, make the calibrated budget an approximately common conditional-risk scale, and we provide a graphical goodness-of-fit diagnostic that targets exactly this property.

Figure 1 illustrates this phenomenon in the polyp-segmentation experiment. For each image, the false-negative rate (FNR) is the fraction of ground-truth polyp pixels omitted from the predicted mask; the conditional FNR shown in each dificulty bin is the average of this fraction over the images in that bin. Global CRC concentrates these missed pixels among dificult inputs, whereas ReCIRC keeps the conditional FNR closer to the target across dificulty levels.

Paper Outline. Section 2 reviews CRC; Sections 3 and 4 present ReCIRC and its guarantees; and Section 5 evaluates the method. Algorithms, proofs, and worked examples appear in the appendices.

Same global target: α = 0.10  
![](images/4ea41d78e04486f4fe12f1338d7f2659a461227bf7e2b4382ae8f5f10c548f36.jpg)  
Figure 1: Flattening the risk curve. Conditional false-negative rate (FNR) by dificulty in the polyp-segmentation experiment (Section 5; target $\alpha = 0 . 1 0 )$ . Global CRC’s FNR rises sharply on dificult images, whereas ReCIRC keeps it near the target. Curves show means and 95% confidence intervals over 20 repetitions; images illustrate the easiest and hardest bins.

## 1.1 Relation to Other Work

Conformal methods for controlling prediction errors. Conformal prediction uses held-out calibration data to construct prediction sets with finite-sample marginal coverage under exchangeability (Vovk et al. 2005, Lei et al. 2018, Angelopoulos & Bates 2023). CRC extends this idea beyond set coverage: it controls the expected value of a bounded loss that decreases as predictions become more protective (Angelopoulos et al. 2024). Riskcontrolling prediction sets and Learn-then-Test provide related guarantees that hold with high probability over the calibration sample; the latter also allows non-monotone losses and several tuning parameters (Bates et al. 2021, Angelopoulos et al. 2025). These guarantees are marginal: they control error averaged across future inputs, not separately for easy and dificult cases. ReCIRC addresses this heterogeneity by first indexing each input’s prediction by its estimated conditional risk and then applying a standard calibration method. We use CRC, but the same ReCIRC family can be calibrated by the other methods under their respective assumptions.

Local and conditional conformal prediction. Distribution-free conditional coverage cannot be guaranteed in finite samples without admitting trivial prediction sets (Lei & Wasserman 2014, Vovk 2012, Barber et al. 2021). This impossibility has motivated a literature on approximate conditional validity. Mondrian and group-conditional constructions (Vovk 2012) calibrate separately within user-specified categories; multivalid calibration extends this to collections of possibly overlapping groups (Jung et al. 2023); localized calibration reweights the calibration sample around the test point, with randomization restoring robust guarantees (Guan 2023, Hore & Barber 2025); data-driven partitions replace the user’s groups by ones learned from the data (Cabezas et al. 2025); and guarantees indexed by a function class interpolate between the marginal and conditional extremes (Gibbs et al. 2025). These approaches either coarsen the conditioning target or modify how calibration itself is carried out. ReCIRC does neither: it keeps ordinary calibration and instead learns a conditional mean-loss scale, so that its finite-sample guarantee stays marginal and distribution-free, while its pointwise and group-level statements are inherited from the accuracy of the estimated risk curves.

Score rectification and plug-in conditional calibration. A recurring idea in conformal prediction is to place heterogeneous inputs on a common scale before calibrating a global cutof. Examples include normalized scores (Papadopoulos & Haralambous 2011), conformalized quantile regression (Romano et al. 2019), and distribution-based transformations (Chernozhukov et al. 2021, Izbicki et al. 2020, 2022, Han et al. 2022). Most directly, Dheur et al. (2025, Eq. 14) transform a conformity score through its estimated conditional CDF, while EPICSCORE (Cruz Cabezas et al. 2025) uses a posterior predictive CDF to incorporate epistemic uncertainty. Plassier et al. (2025) instead learn a transformation from a conditional quantile at the target coverage level. Section 3.4 shows that ReCIRC recovers the conditional-CDF construction under the miscoverage loss function. Our method extends this principle to bounded losses that are monotone along ordered decision families: it estimates the conditional mean-loss curve, inverts it to define a common risk scale, and calibrates that scale with ordinary CRC. The same fitted curve can be recalibrated at multiple risk targets, and its estimation error afects the conditional-risk bounds but not finite-sample marginal validity.

Adaptive conformal risk control. AA-CRC is the closest method in aim: for a fixed target α, it learns an input-dependent threshold and controls weighted risk over every non-negative weight in a chosen vector space of functions, up to a regularization term (Blot et al. 2025, Gibbs et al. 2025). ReCIRC instead estimates the entire conditional mean-loss curve on independent data, uses it to define a one-parameter local risk scale, and delegates final calibration to ordinary CRC. Its pointwise and group-level conclusions are expressed directly through risk-estimation error (Propositions 4.6 and E.1), while Corollary 4.7 gives conditions under which the deployed conditional risk converges to the target. Fair Risk Control targets multi-group fairness constraints through multicalibration (Zhang et al. 2024). Localized adaptive risk control instead operates online, updating a threshold function in a reproducing kernel Hilbert space (Zecchin & Simeone 2024); ReCIRC is an ofline procedure whose final predictor retains CRC’s finite-sample marginal guarantee.

Risk-based parameterizations for specific tasks. Two task-specific constructions use a similar reparameterization to the one we propose. Xu et al. (2023) study ordinal classification, where prediction sets are contiguous ranges of classes. They index the nested family by a plug-in estimate of the conditional risk, computed from the classifier’s estimated class probabilities, and then calibrate that index with CRC. Luo & Zhou (2025) pursue the same goal in image segmentation, adapting masks through the predicted pixel-probability mass, with a probability-calibrated variant and a stratified variant that assigns group-specific thresholds; only the latter requires a group structure. ReCIRC formulates risk rectification for general ordered families and arbitrary bounded monotone losses.

## 1.2 Novelty

Our contributions are:

• A general risk-scale construction: We formulate local risk-curve inversion for bounded monotone losses along ordered decision families, extending risk-based parameterizations beyond individual tasks.

• Conditional guarantees from estimation accuracy: We relate risk-curve estimation error to pointwise, most-input, and group-conditional risk bounds relative to the budget (Propositions 4.6 and E.1). Under additional attainability and regularity conditions, we obtain a rate for convergence of deployed conditional risk to the target (Corollary 4.7).

• A risk-scale diagnostic: We compare realized losses with advertised budgets, overall and within covariate groups, to assess the fitted scale (Section 3.5). The diagnostic is a by-product of calibration and requires no data beyond the calibration set.

Distribution-free marginal validity follows by applying the existing CRC calibration argument to the rectified family (Theorem 4.4; Angelopoulos et al. 2024), regardless of risk-estimation accuracy.

## 2 Background: Conformal Risk Control

## 2.1 Setting and the CRC procedure

Let $( X _ { 1 } , Y _ { 1 } ) , \dots , ( X _ { n } , Y _ { n } ) , ( X _ { n + 1 } , Y _ { n + 1 } )$ be exchangeable random pairs taking values in $\mathcal { X } \times \mathcal { V }$ , where the first n pairs form a calibration set C and $\left( X _ { n + 1 } , Y _ { n + 1 } \right)$ is a test point whose response is unobserved. Consider a family of decision rules $\{ f _ { \lambda } : \lambda \in \Lambda \}$ indexed by a scalar $\lambda \in \Lambda = [ \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } ]$ , together with a loss $\ell ( f _ { \lambda } ( x ) , y ) \in [ 0 , B ]$ . The family is assumed to be ordered so that larger values of λ are more protective:

$$
\lambda _ { 1 } \leq \lambda _ { 2 } \quad \Longrightarrow \quad \ell ( f _ { \lambda _ { 1 } } ( x ) , y ) \geq \ell ( f _ { \lambda _ { 2 } } ( x ) , y ) \quad { \mathrm { f o r ~ e v e r y ~ } } ( x , y ) .\tag{1}
$$

The prototypical example takes $f _ { \lambda } ( x )$ to be a prediction set that grows with λ and ℓ to be a loss that shrinks as the set grows, such as the false-negative proportion of a segmentation mask or a label set. For example, let $Y \subseteq { \mathcal { U } }$ denote the true positive pixels or labels and let $f _ { \lambda } ( x ) \subseteq \mathcal { U }$ be the selected set. The loss

$$
\ell _ { \mathrm { F N } } ( f _ { \lambda } ( x ) , Y ) : = \frac { | Y \setminus f _ { \lambda } ( x ) | } { | Y | \vee 1 }
$$

is bounded by one and is non-increasing as $f _ { \lambda } ( x )$ expands.

Throughout, the family $\{ f _ { \lambda } \}$ is treated as fixed: when it is built from a trained base model, all expectations and risk curves below are interpreted conditionally on the data used to fit it, which is assumed independent of the calibration and test points.

Given a target risk level α, CRC (Angelopoulos et al. 2024) computes the calibration risk $\begin{array} { r } { \widehat { \mathcal { R } } _ { \mathrm { c a l } } ( \lambda ) = n ^ { - 1 } \sum _ { j = 1 } ^ { n } \ell ( f _ { \lambda } ( X _ { j } ) , Y _ { j } ) } \end{array}$ and selects

$$
\widehat { \lambda } = \operatorname* { i n f } \left\{ \lambda \in \Lambda : \frac { n } { n + 1 } \widehat { \mathcal { R } } _ { \mathrm { c a l } } ( \lambda ) + \frac { B } { n + 1 } \leq \alpha \right\} .
$$

Under exchangeability, monotonicity, and right-continuity of the loss in $\lambda ,$ Angelopoulos et al. (2024) show that

$$
{ \mathbb E } \{ \ell ( f _ { \widehat { \lambda } } ( X _ { n + 1 } ) , Y _ { n + 1 } ) \} \le \alpha ,\tag{2}
$$

and that this bound is nearly tight, the expected loss being at least $\alpha - 2 B / ( n + 1 )$ under mild regularity conditions. When ℓ is the miscoverage loss ${ \bf 1 } \{ y \notin f _ { \lambda } ( x ) \}$ , (2) recovers the marginal coverage guarantee of split conformal prediction.

## 2.2 The limitation: a global threshold is not a risk scale

The guarantee in (2) is marginal: it averages over the covariate distribution. To see what it hides, define the local conditional risk curve at input x,

$$
R ( \lambda \mid x ) : = \mathbb { E } \{ \ell ( f _ { \lambda } ( X ) , Y ) \mid X = x \} ,\tag{3}
$$

which is non-increasing in λ by (1). CRC controls only the average ${ \mathbb E } \{ R ( \widehat \lambda \mid X ) \}$ , and $R ( \lambda \mid x )$ is typically far from constant in x: the same raw threshold λ intersects the risk curves of easy and hard inputs at very diferent risk levels, as illustrated in Figure 2(a). A value of λ that gives low risk for one input may give high risk for another, so risk can fall below the target on easy inputs and exceed it on hard ones (Figure 2(c)). This is the problem that risk rectification addresses.

## 3 Our approach: ReCIRC

The key insight of ReCIRC is to reinterpret the parameter calibrated by CRC as a local risk budget: instead of asking every input to share the same raw threshold λ, we ask every input to share the same target conditional risk $^ { a , }$ and let the threshold adapt to the input. We first describe the oracle construction, which assumes the local risk curves are known, and then the practical procedure, which estimates them.

(a) Same $\lambda ,$ diferent risks  
![](images/91865041d427d1363b99b11abb3cb50f289525493d5dcec2374c5a7cd69843bb.jpg)  
λ

(b) Same risk, diferent λ  
![](images/822c83618279a2796b4a0860b1abbfa4df96454d75474ce177513596452c3c2d.jpg)  
λ

(c) Risk across inputs  
![](images/2a35e8c2854cfe9d3cc088bc767ed6ae6c8575f62a70a466222fd4416ef43f5a.jpg)  
inputs, easy → hard  
Figure 2: Risk rectification as a reparameterization. (a) The same raw threshold $\lambda _ { 0 }$ gives diferent conditional risks at an easy and a hard input. (b) A common risk budget a instead gives each input its own threshold $R ^ { - 1 } ( a \mid x )$ , obtained by inverting its local risk curve. (c) Conditional risk across inputs, with the two inputs of (a) marked. A global threshold whose average risk is $\alpha$ (here $\lambda _ { 0 } )$ leaves easy inputs below $\alpha$ and hard inputs above it; oracle rectification with $a = \alpha$ brings every input to $\alpha .$ . In practice the curves are estimated and CRC calibrates $a$ (Section 3.2). The plots are normalized to $B = 1$

## 3.1 Oracle risk rectification

Suppose for a moment that the local risk curves $R ( \lambda \mid x )$ in (3) were known. For a desired risk budget $a \in [ 0 , B ]$ , define the risk-curve inverse

$$
R ^ { - 1 } ( a \mid x ) : = \operatorname* { i n f } \{ \lambda \in \Lambda : R ( \lambda \mid x ) \leq a \} ,
$$

with the convention inf $\emptyset = \lambda _ { \mathrm { m a x } }$ . Throughout, $R ^ { - 1 } ( a \mid x )$ denotes this generalized inverse of the local curve in the threshold coordinate: its input is a risk level $^ { a , }$ and its output is the raw decision threshold needed at input x to attain conditional risk at most a. Figure 2(b) shows this inversion: a horizontal cut at risk level a produces one threshold for an easy input and a larger, more protective threshold for a hard input.

We define the oracle rectified family by

$$
f _ { a } ^ { R } ( x ) : = f _ { R ^ { - 1 } ( a | x ) } ( x ) , \qquad a \in [ 0 , B ] .\tag{4}
$$

The family $\{ f _ { a } ^ { R } : a \in [ 0 , B ] \}$ is simply a reparameterization of the original family $\{ f _ { \lambda } : \lambda \in$ $\Lambda \}$ : each rectified rule outputs an element of the original nested family, so the geometry of the predictions is inherited from the original CRC construction (Appendix G illustrates this for several losses from the CRC literature). The diference is that the new parameter a has a direct risk interpretation. Assume that $R ( \cdot \mid x )$ is right-continuous, as holds whenever the loss is right-continuous in λ (the standard CRC requirement of Section 2.1). The construction then gives

$$
{ \mathbb E } \{ \ell ( f _ { a } ^ { R } ( X ) , Y ) ~ | ~ X = x \} = R ( R ^ { - 1 } ( a ~ | ~ x ) ~ | ~ x ) \le a ,\tag{5}
$$

so, in the oracle case, a is a common upper bound on the conditional risk at every input x (Figure $2 ( \mathrm { c } ) )$ ). If the local curve is continuous and crosses level $^ { a , }$ , then the achieved risk equals $a ;$ otherwise it may be conservative. Thus risk rectification changes the global CRC knob from a raw threshold λ, which is not comparable across inputs, into a common conditional-risk upper bound and, where the curves cross the target, an exact risk scale.

## 3.2 The ReCIRC procedure

The oracle construction uses the unknown curves $R ( \lambda \mid x )$ . In practice, we replace them by estimated curves and let CRC calibrate the remaining scalar risk budget, as summarized in the pipeline of Figure 3. The procedure is the following:

1. Choose an ordered CRC family $\{ f _ { \lambda } : \lambda \in \Lambda \}$ —typically built from a base predictive model fitted on separate training data—a bounded loss $\ell \in [ 0 , B ]$ , and a target marginal risk level α.

2. Split the labeled data into a risk-training set D and a calibration set $\mathscr { C } = \{ ( X _ { j } , Y _ { j } ) \} _ { j = 1 } ^ { n }$

3. Use D exclusively to train and tune the conditional-risk estimator $\hat { R } _ { D } ( \lambda \mid x )$ by directly regressing the loss on $( x , \lambda )$ (see Section 3.3 for details). An alternative based

on an estimated law of $Y \mid x$ is described in Appendix B.

4. Fix a finite threshold grid $\Lambda _ { M } = \{ \lambda _ { 1 } = \lambda _ { \operatorname* { m i n } } < \cdots < \lambda _ { M } = \lambda _ { \operatorname* { m a x } } \}$ . For each candidate risk budget a, compute

$$
\widehat { R } _ { \mathcal { D } } ^ { - 1 } ( a \mid x ) = \operatorname* { m i n } \{ \lambda \in \Lambda _ { M } : \widehat { R } _ { \mathcal { D } } ( \lambda \mid x ) \leq a \} ,\tag{6}
$$

with the conventions min $\emptyset = \lambda _ { \mathrm { m a x } }$ and $\widehat { R } _ { \mathcal { D } } ^ { - 1 } ( 0 \mid x ) = \lambda _ { \mathrm { m a x } }$ . Then define the rectified decision rules as

$$
\begin{array} { r } { \widehat { f } _ { \mathcal { D } , a } ^ { R } ( x ) = f _ { \widehat { R } _ { \mathcal { D } } ^ { - 1 } ( a | x ) } ( x ) . } \end{array}
$$

5. Fix a finite budget grid $A = \{ 0 = a _ { 0 } < a _ { 1 } < \cdots < a _ { L } \} \subset [ 0 , B ]$ containing the fully protective budget $a ~ = ~ 0$ . On the calibration set, compute $\widehat { \mathcal { R } } _ { \mathrm { c a l } } ^ { R } ( a ) \ =$ $\textstyle n ^ { - 1 } \sum _ { j = 1 } ^ { n } \ell \{ { \widehat { f } } _ { \mathcal { D } , a } ^ { R } ( X _ { j } ) , Y _ { j } \}$ for each $a \in { \mathcal { A } }$ . Because a is a risk budget, larger values of a are less protective, so we choose the largest grid budget that still passes the CRC correction:

$$
\widehat { a } = \operatorname* { m a x } \left\{ a \in \mathcal { A } : \frac { n } { n + 1 } \widehat { \mathcal { R } } _ { \mathrm { c a l } } ^ { R } ( a ) + \frac { B } { n + 1 } \leq \alpha \right\} .\tag{7}
$$

6. For a new input x, output $\widehat { f } _ { \mathcal { D } , \widehat { a } } ^ { R } ( x ) = f _ { \widehat { R } _ { \mathcal { D } } ^ { - 1 } ( \widehat { a } | x ) } ( x )$

Algorithm 1 in Appendix A summarizes the procedure. The reason the method is valid is simple: after D is fixed, the rules $\{ { \widehat { f } } _ { { D } , a } ^ { R } : a \in [ 0 , B ] \}$ form a one-parameter family whose loss is nondecreasing in $^ { a , }$ and CRC is applied to this family using only observed calibration losses. Therefore the standard CRC argument gives ${ \mathbb E } \{ \ell ( \widehat f _ { \mathcal D , \widehat a } ^ { R } ( X _ { n + 1 } ) , Y _ { n + 1 } ) \} \le \alpha$ (Theorem 4.4). The guarantee does not require $\widehat { R } _ { \mathcal { D } }$ to be correct: poor estimated risk curves can make the predictions less eficient or less locally adaptive, but they do not invalidate finite-sample marginal risk control.

Remark 3.1 (Other calibration wrappers). Rectification is a reparameterization of the family, not of the calibration scheme. The rectified family $\{ \widehat { f } _ { \mathcal { D } , a } ^ { R } \}$ can therefore also be calibrated

![](images/2f7804045b8b8c6fb8e00f019c5aeac7bb3c19109f192833215f369e69584dd9.jpg)  
Figure 3: The ReCIRC pipeline. The risk-training set $\mathcal { D }$ is used only to train and tune the conditional-risk estimator; its estimated local risk curves are inverted to define the rectified family indexed by the risk budget $a ;$ the calibration set C is used only by the final CRC step, which selects the budget ab. Because calibration sees only observed losses of a monotone one-parameter family, marginal validity holds regardless of the quality of the estimated curves.  
with other wrappers, such as risk-controlling prediction sets for high-probability guarantees (Bates et al. 2021) or Learn-then-Test for non-monotone losses (Angelopoulos et al. 2025). We focus on CRC for concreteness.

## 3.3 Estimating the local risk curve

The remaining ingredient is the estimator $\hat { R } _ { D } ( \lambda \mid x )$ . Using the risk-training set $\mathcal { D } =$ $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { m }$ , create an augmented loss-training set: for each observation $i ,$ sample thresholds $\lambda _ { i k } \sim q , k = 1 , \ldots , K$ , for a design distribution q supported on Λ (e.g., uniform), and compute the observed losses $Z _ { i k } = \ell ( f _ { \lambda _ { i k } } ( X _ { i } ) , Y _ { i } )$ . Then train the conditional-risk regression

model

$$
g _ { \theta } ( x , \lambda ) \approx \mathbb { E } \{ \ell ( f _ { \lambda } ( X ) , Y ) \mid X = x \} = R ( \lambda \mid x )\tag{8}
$$

on the augmented data $\left\{ \left( X _ { i } , \lambda _ { i k } , Z _ { i k } \right) \right\}$ , and set ${ \hat { R } } _ { D } ( \lambda \mid x ) = g _ { \theta } ( x , \lambda )$ . Notice that only a single regression curve is fitted: x and λ are the covariates used. The true curve is non-increasing in $\lambda ,$ so the fitted surface should respect or approximate this monotonicity; a simple correction applies isotonic regression in λ for each queried x. If the same labeled data are used both to construct $f _ { \lambda }$ and to fit $g _ { \boldsymbol { \theta } }$ , the fitted risk surface may be optimistically biased; this efect can be reduced by separating the base-model and risk-training samples, cross-fitting, or using out-of-sample losses.

For computationally heavier settings, we use an alternative implementation in which the risk curve is estimated at a fixed collection of threshold anchors and then interpolated over the full threshold grid, with monotonicity enforced before inversion. The specific experiments and implementation details are reported in Appendix C.2.

An alternative estimator first models $Y \mid x$ and then averages the loss under that model. Appendix B gives the construction and discusses when this generative route may be useful.

## 3.4 Special case: rectifying conformal prediction scores

Standard conformal prediction with a scalar nonconformity score is a special case of the framework. Let $S ( x , y )$ be a score for which smaller values are better, and define $f _ { \lambda } ( x ) = \{ y : S ( x , y ) \leq \lambda \}$ with the miscoverage loss $\ell ( f _ { \lambda } ( x ) , y ) = 1 \{ S ( x , y ) > \lambda \}$ , so that $B = 1$ . The local risk curve is the conditional upper tail of the score,

$$
R ( \lambda \mid x ) = \mathbb { P } \{ S ( x , Y ) > \lambda \mid X = x \} = 1 - F _ { S \mid x } ( \lambda ) ,
$$

where $F _ { S | x }$ is the conditional distribution function of $S ( x , Y )$ given $X = x$ . The risk budget $a \in [ 0 , 1 ]$ is now a local miscoverage budget, and the oracle threshold is the conditional quantile $R ^ { - 1 } ( a \mid x ) = F _ { S \mid x } ^ { - 1 } ( 1 - a )$ , up to the usual generalized-quantile convention, so the

rectified set is $f _ { a } ^ { R } ( x ) = \{ y : S ( x , y ) \leq F _ { S | x } ^ { - 1 } ( 1 - a ) \}$ . When $F _ { S | x }$ is continuous and strictly increasing on the range of the scores, this set can be rewritten as

$$
f _ { a } ^ { R } ( x ) = \{ y : \widetilde { S } ( x , y ) \leq 1 - a \} , \qquad \widetilde { S } ( x , y ) : = F _ { S | x } ( S ( x , y ) ) ,
$$

which is exactly local score recalibration through the probability integral transform, as used to obtain approximate conditional coverage in conformal prediction (Chernozhukov et al. 2021, Izbicki et al. 2020, 2022; Dheur et al. 2025, Cruz Cabezas et al. 2025).

Appendix E.4 extends this quantile interpretation to general bounded monotone losses: Proposition E.4 represents the local risk curve as the upper tail of a latent variable, so that risk rectification is a local quantile transformation. This latent variable need not be an observed conformity score.

## 3.5 A risk-calibration diagnostic for the risk scale

Rectification treats a as a conditional-risk scale. This interpretation can be assessed on any sample independent of $\mathcal { D }$ , including the calibration set C. For a group $G$ of inputs, fixed without using the responses of that sample, let $I _ { G } = \{ i : X _ { i } \in G \}$ and compute, for every budget on the grid,

$$
{ \widehat { r } } _ { G } ( a ) : = { \frac { 1 } { | I _ { G } | } } \sum _ { i \in I _ { G } } \ell \left( { \widehat { f } } _ { { D } , a } ^ { R } ( X _ { i } ) , Y _ { i } \right) .\tag{9}
$$

Plotting ${ \widehat { r } } _ { G } ( a )$ against a gives a risk-calibration curve. An exact risk scale follows the diagonal; a curve below it is conservative; and a curve above it reveals budgets at which the fitted scale understates risk. Drawing the curve overall and within prespecified dificulty bins shows both global mis-scaling and where it occurs. For fixed a and $G , { \widehat { r } } _ { G } ( a )$ is an unbiased estimate of the group risk bounded in (12). On ${ \mathcal { C } } ,$ the overall curve is the calibration table already computed by Algorithm 1, so the diagnostic requires no extra data. Bootstrap or concentration-based bands can be added to the curves.

Like any grouped diagnostic, (9) assesses risk on average within the chosen bins rather than at each individual x: deviations of opposite sign can cancel in the overall curve, and finer or better-chosen bins sharpen the assessment.

Curves below the diagonal do not necessarily indicate inaccurate risk estimates: even with exact risk curves, discrete prediction sets or limits on how small predictions can become may prevent the risk from reaching the budget.

## 4 Theoretical Guarantees

This section gives two main guarantees. First, finite-sample marginal risk control holds regardless of the quality of the estimated risk curves. Second, accurate curves turn the budget into an approximate conditional-risk scale; when the curves cross the target, the deployed conditional risk approaches α. Most-input and group versions of the second guarantee, and a more general form of its asymptotic version, are shown in Appendix E; all proofs are in Appendix F.

## 4.1 Finite-sample marginal risk control

Recall that D is the risk-training set, $\mathscr { C } = \{ ( X _ { j } , Y _ { j } ) \} _ { j = 1 } ^ { n }$ the calibration set, and $\left( X _ { n + 1 } , Y _ { n + 1 } \right)$ the test point. To avoid clutter, conditioning on D below also conditions on any independent data and randomness used to fit the base family or the risk estimator before calibration. We will, when necessary, assume the following:

Assumption 4.1 (Exchangeability). $( X _ { 1 } , Y _ { 1 } ) , \dotsc , ( X _ { n + 1 } , Y _ { n + 1 } )$ are exchangeable and independent of D.

Assumption 4.2 (Monotone bounded family). The loss satisfies $\ell ( f _ { \lambda } ( x ) , y ) \in [ 0 , B ]$ and (1), and the family contains a fully protective rule: $\ell ( f _ { \lambda _ { \mathrm { m a x } } } ( x ) , y ) = 0$ for every (x, y).

Assumption 4.3 (Pre-calibration finite grid). The budget in (7) is selected from a finite grid $\mathcal { A } = \{ 0 = a _ { 0 } < a _ { 1 } < \cdots < a _ { L } \} \subset [ 0 , B ]$ containing $a _ { 0 } = 0$

Assumptions 4.1 and 4.2 are standard in CRC. The former is its distributional requirement; for the latter, the full label set, the full mask, or the whole response space incurs zero false-negative loss. Assumption 4.3 simply formalizes how the procedure is implemented (Algorithm 1).

Theorem 4.4 (Marginal validity of ReCIRC). Fix $0 < \alpha \le B$ . Suppose Assumptions $\it 4 . 1 \mathrm { - } \it 4 . 3$ hold and $n \geq B / \alpha - 1$ . Then the budget ab in (7) is well defined and

$$
\begin{array} { r } { \mathbb { E } \left\{ \ell \left( \widehat { f } _ { \mathcal { D } , \widehat { a } } ^ { R } ( X _ { n + 1 } ) , Y _ { n + 1 } \right) \big | \mathcal { D } \right\} \leq \alpha . } \end{array}
$$

In particular, the unconditional risk is also at most α.

Theorem 4.4 states that rectification preserves the finite-sample marginal guarantee of CRC, no matter how poor the estimated risk curves are.

## 4.2 Conditional risk control

We now quantify in what sense the budget a controls conditional risk. First, we show that the oracle budget is a conditional-risk upper bound.

Proposition 4.5 (Oracle conditional validity). Fix x. Suppose Assumption 4.2 holds and $R ( \cdot \mid x )$ is right-continuous. Then, for every $a \in [ 0 , B ] , \mathbb { E } \{ \ell ( f _ { a } ^ { R } ( X ) , Y ) | X = x \} \leq a$ . If in addition $R ( \cdot \mid x )$ is continuous and $a \leq R ( \lambda _ { \operatorname* { m i n } } \mid x )$ , the inequality holds with equality.

The equality condition says that the least protective rule must itself have risk at least as large as the requested budget. If it does not, the local scale stops at $R ( \lambda _ { \operatorname* { m i n } } \mid x )$ ; the upper bound still holds, but equality cannot.

In practice the curves are estimated, and the oracle bound degrades by exactly the amount by which the fitted curve understates the true risk:

Proposition 4.6 (Conditional validity from risk-curve accuracy). Fix x and suppose Assumptions 4.1 and 4.2 hold. Define

$$
\varepsilon _ { M } ( x ) : = \operatorname* { m a x } _ { \lambda \in \Lambda _ { M } } [ R ( \lambda \mid x ) - { \widehat R } _ { \mathcal { D } } ( \lambda \mid x ) ] _ { + } .
$$

Then, simultaneously for every $a \in [ 0 , B ]$

$$
\begin{array} { r } { \mathbb { E } \left\{ \ell \left( \widehat { f } _ { \mathcal { D } , a } ^ { R } ( X ) , Y \right) \Big | X = x , \mathcal { D } \right\} \leq a + \varepsilon _ { M } ( x ) . } \end{array}
$$

$I f ,$ in addition, the calibration and test pairs are $i . i . d .$ , Assumption 4.3 holds, $0 < \alpha \le B$ and $n \geq B / \alpha - 1$ , the bound remains valid after conditioning on the calibration set and replacing a by $\widehat { a }$

At the calibrated budget, the proposition controls risk relative to $\widehat { a }$ rather than relative to α itself, and the two can difer when the rectified family is conservative or saturates at some inputs. Proposition E.1 in Appendix E.1 converts the pointwise bound into “most-input” and arbitrary-group statements. Proposition E.2 in Appendix E quantifies how risk-curve estimation error and threshold-grid spacing translate into error in the conditional-risk scale over attainable budgets.

The next result closes the remaining gap, giving conditions under which ${ \widehat { a } } ,$ and with it the deployed conditional risk, converges to the target level.

Corollary 4.7 (Rate for the deployed conditional risk; corollary of Theorem E.3). Suppose Assumptions $\it 4 . 1 \mathrm { - } \it 4 . 3$ hold, with the calibration and test pairs i.i.d. Along a sequence $n  \infty ,$ let the risk-training size satisfy $m = m ( n ) \to \infty$ , let $M = M ( n )$ , and let the threshold grid have maximum spacing $h _ { M } \to 0$ , and let the budget grid $\mathcal { A } _ { n } = \{ 0 = a _ { 0 , n } < \cdot \cdot \cdot < a _ { L _ { n } , n } = B \}$ with maximum spacing $h _ { n } \to 0$ . All quantities below are indexed along this sequence: $\eta _ { m }$ and $h _ { M }$ abbreviate $\eta _ { m ( n ) }$ and the spacing of $\Lambda _ { M ( n ) }$ , and every $O _ { \mathbb { P } }$ statement is taken as $n \to \infty$ . Fix $\alpha \in ( 0 , B )$ . Assume there is a constant $c > 0$ such that $R ( \lambda _ { \operatorname* { m i n } } \mid x ) \geq \alpha + c$ for almost every x. Suppose also that the curves $R ( \cdot \mid x )$ share a finite Lipschitz constant and that, uniformly over inputs and grid thresholds, their estimation error is $O _ { \mathbb { P } } ( \eta _ { m } )$ for some $\eta _ { m }  0 ;$ ; that is,

$$
\operatorname { e s s s u p } _ { x \sim P _ { X } } \operatorname* { m a x } _ { \lambda \in \Lambda _ { M } } \vert R ( \lambda \mid x ) - \widehat { R } _ { \mathcal { D } _ { m } } ( \lambda \mid x ) \vert = O _ { \mathbb { P } } ( \eta _ { m } ) .
$$

Then, provided $n \geq B / \alpha - 1$ , for almost every fixed $x ,$

$$
\left| \mathbb { E } \left\{ \ell \left( \widehat { f } _ { \mathcal { D } _ { m } , \widehat { a } } ^ { R } ( X ) , Y \right) \Big | X = x , \mathcal { D } _ { m } , \mathcal { C } \right\} - \alpha \right| = O _ { \mathbb { P } } ( \eta _ { m } + h _ { M } + n ^ { - 1 / 2 } + h _ { n } ) .
$$

The endpoint condition in the corollary ensures that risk α is attainable for almost every input. The rate combines errors from learning the risk curves, discretizing the threshold and budget, and using a finite calibration sample. Here $\eta _ { m }$ is a standard uniform regression rate for the risk surface $( x , \lambda ) \mapsto R ( \lambda \mid x )$ . For example, if x is d-dimensional and this surface is s-Hölder smooth, local-polynomial or spline/wavelet regression can attain

$$
\eta _ { m } = ( \log m / m ) ^ { s / ( 2 s + d + 1 ) }
$$

under the usual design, moment, and tuning conditions (Stone 1982, Masry 1996, Chen & Christensen 2015); the extra one in the denominator comes from using λ as an additional predictor. Theorem E.3 in Appendix E states the general version, in which the marginal and local risk-scale errors are allowed to have diferent rates and no uniformity over x is required.

The pointwise guarantee in Proposition 4.6 also has a deployment implication: it remains valid if deployment only reweights the inputs. Formally, covariate shift means that the input distribution changes from $P _ { X }$ during calibration to $Q _ { X }$ at deployment, while the response mechanism $Y \mid X$ remains unchanged (Shimodaira 2000, Izbicki et al. 2017).

Corollary 4.8 (Estimated ReCIRC under covariate shift). Fix D and C, and let $\widehat { f } = \widehat { f } _ { \mathcal { D } , \widehat { a } } ^ { R }$ be the deployed rule. $I f Q _ { Y | X } = P _ { Y | X }$ for $P _ { X }$ -almost every x and $Q _ { X } \ll P _ { X }$ , then, under the conditions of Proposition 4.6,

$$
\mathbb { E } _ { Q } \{ \ell ( \widehat { f } ( X ) , Y ) \} - \widehat { a } \leq \| \varepsilon _ { M } \| _ { L ^ { \infty } ( P _ { X } ) } .\tag{10}
$$

Thus the worst local underestimation of the fitted risk curves controls deployed risk under any such shift.

## 5 Experiments

We evaluate ReCIRC across eight synthetic and real-data settings designed to capture diferent forms of risk heterogeneity. We examine whether rectification reduces groupwise risk heterogeneity while maintaining marginal risk close to the target level, and characterize the associated changes in prediction size. The synthetic experiments provide controlled mechanisms of heterogeneity, whereas the real-data applications span image segmentation, multilabel text classification, multiclass prediction, and tabular regression. Within each setting, we compare global CRC, AA-CRC, and ReCIRC, with the latter using TabICLv2 as a regression model to estimate the conditional-risk surface (Qu et al. 2026).

## 5.1 Experimental setup

Detailed experimental settings, including base predictors, adaptive representations, sample sizes, and data splits, are reported in Appendix C. The three synthetic designs introduce distinct sources of heterogeneity. The first is a heteroscedastic regression problem with $X \sim \mathrm { U n i f } ( - 2 , 2 )$ and $Y = \sin ( \pi X / 2 ) + \sigma ( X ) \varepsilon$ , where $\varepsilon \sim N ( 0 , 1 )$ and $\sigma ( x ) = 0 . 2 + 0 . 6 | x |$ The second is a multilabel problem in which a one-dimensional dificulty variable controls both label cardinality and the separation between positive and negative scores. The third introduces interaction efects through an XOR structure, with conditional scale $\sigma ( x ) = 0 . 4 + 2 . 2 \Im \{ \mathrm { s i g n } ( x _ { 1 } ) \neq \mathrm { s i g n } ( x _ { 2 } ) \}$ . These controlled settings are complemented by five real-data benchmarks spanning image segmentation, text classification, multiclass prediction, and tabular regression: polyp segmentation uses PraNet pixelwise scores; RCV1 and Letter Recognition construct prediction sets from probabilistic classifiers; and Medical Insurance and Superconductor use quantile random forests (Meinshausen 2006) to construct regression intervals.

Data allocation and repetitions. As in Section 3.2, ReCIRC uses a risk-training set D to fit the conditional-risk estimator and an independent calibration set C for the final CRC step. Within each repetition, all methods share the same base predictor, nested decision family, and test observations, so their comparisons are paired. Because global CRC does not require risk training, it is calibrated on D ∪ C in the heteroscedastic, latent-dificulty, and RCV1 settings. In the QRF-based settings, D is used to fit the base model, so global CRC is calibrated on C alone; in polyp segmentation and Letter Recognition, it uses the same calibration set C as ReCIRC. The size of the calibration sample afects only the variability of the global threshold, not its inability to adapt to input dificulty, which drives the groupwise comparisons below. AA-CRC follows its original data allocation except in polyp segmentation and Letter Recognition (Appendix C.4).

The QRF-based settings require additional care because the base quantile random forest is fitted on D. If ordinary fitted predictions were used on the same observations to construct the ReCIRC risk-training data, the resulting losses could be optimistically biased. We therefore compute the QRF-derived features and losses for observations in D from out-of-bag quantile predictions, so that each observation is evaluated using only trees for which it was out of bag. Calibration and test observations use ordinary predictions from the fitted QRF. The target risk level is α = 0.10 throughout, and all results are averaged over 20 repetitions.

Methods. Global CRC calibrates a single threshold shared by all inputs. AA-CRC (Blot et al. 2025) instead learns an input-dependent threshold within a chosen adaptive function class, using the objective and gradient of the authors’ implementation. In the RF-based settings, we fit a random forest on D to a task-specific measure of prediction dificulty and use its leaf-membership indicators as the feature map $\Phi ( x )$ . The resulting threshold has the form

$$
u _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) = \boldsymbol { \Phi } ( \boldsymbol { x } ) ^ { \top } \boldsymbol { \theta } ,
$$

with the coeficients fitted on the independent calibration set $\mathcal { C } .$ Because each tree contributes exactly one active leaf indicator, constant functions already belong to the span of $\Phi ( x )$ , so no additional intercept is needed.

AA-CRC and ReCIRC use covariate information diferently. AA-CRC adapts the decision parameter directly through the function $u _ { \theta } ( x )$ . ReCIRC estimates the conditional-risk surface $R ( \lambda \mid x )$ as a function of an input representation and λ, and then inverts the fitted surface to obtain an input-specific threshold for each risk budget. The final scalar budget is calibrated by CRC.

The representations used by the two methods are therefore task-specific and need not coincide. In the heteroscedastic experiment, AA-CRC learns its representation from the covariate and absolute residuals, whereas ReCIRC estimates risk jointly from the covariate and λ. In the synthetic multilabel experiment, both methods use the dificulty covariate and score summaries, with ReCIRC additionally conditioning on λ. In the QRF-based settings, we follow the random-forest representation proposed for tabular data in AA-CRC (Blot et al. 2025): AA-CRC learns an RF representation from the tabular covariates and a minimum-covering-multiplier target, whereas ReCIRC also uses QRF-derived location and scale summaries to estimate the risk surface.

ReCIRC uses TabICLv2 to estimate the local risk curves. Depending on the setting, we either fit a single regression model jointly on the input representation and $\lambda ,$ or fit separate regressors at a collection of threshold anchors and interpolate between the estimated risks, enforcing monotonicity before inversion. For complex inputs such as images, the risk estimator receives compact summaries from the base model rather than the raw covariates. These representations need not recover all information relevant to the conditional loss: as discussed in Section 3.2, an imperfect risk model may reduce local adaptivity but does not afect the finite-sample marginal guarantee.

Appendix C gives the complete specification for each experiment, including the basemodel fitting sample, nested decision family, threshold and budget grids, risk-regression representation, threshold-sampling or anchor construction, AA-CRC regularization, and evaluation groups.

## 5.2 Losses and evaluation metrics

Each experiment uses a bounded loss appropriate to its task. For multilabel prediction and segmentation, we use the missed-positive loss $\ell _ { \mathrm { F N } }$ of Section 2.1, that is, the fraction of positive labels or pixels omitted from the prediction. The corresponding risk is the perobservation average of this loss; in segmentation, this gives equal weight to images rather than pooling positive pixels across the dataset. Letter Recognition uses the miscoverage loss

$$
\ell _ { \mathrm { m c } } ( f _ { \lambda } ( x ) , y ) = { \bf 1 } \{ y \notin f _ { \lambda } ( x ) \} .
$$

The heteroscedastic regression experiment uses symmetric intervals $[ \widehat \mu ( x ) - \lambda , \widehat \mu ( x ) + \lambda ]$ ， where ${ \widehat { \mu } } ( x )$ is the fitted GAM mean, together with the bounded excess loss

$$
\ell _ { \mathrm { e x c } } ( f _ { \lambda } ( x ) , y ) = \operatorname* { m i n } \left\{ 1 , \frac { ( | y - \widehat { \mu } ( x ) | - \lambda ) _ { + } } { 1 . 5 } \right\} .
$$

The QRF-based regression experiments use intervals $[ l _ { \lambda } ( x ) , u _ { \lambda } ( x ) ]$ centered at the estimated conditional median, together with the asymmetric miscoverage loss

$$
\ell _ { \mathrm { a s y m } } ( f _ { \lambda } ( x ) , y ) = 0 . 2 { \bf 1 } \{ y < l _ { \lambda } ( x ) \} + 0 . 8 { \bf 1 } \{ y > u _ { \lambda } ( x ) \} .
$$

Since a response cannot fall below and above the interval simultaneously, this loss is bounded by 0.8. The CRC correction uses this bound in Superconductor and the conservative bound $B = 1$ in Medical Insurance and XOR; both are valid, since B only needs to upper-bound the loss.

For repetition r and method $j ,$ , let $T _ { r }$ denote the test set and $L _ { r j i }$ the observed loss on test observation $i \in T _ { r }$ . The empirical marginal risk is

$$
\bar { L } _ { r j } = \frac { 1 } { | T _ { r } | } \sum _ { i \in T _ { r } } L _ { r j i } .
$$

To assess heterogeneity, let $\mathcal { G } _ { r } = \{ G _ { r 1 } , \ldots , G _ { r J _ { r } } \}$ denote the collection of evaluation groups, common to all methods within a repetition. Groups with too few test observations are omitted (Appendix C.5), so $J _ { r }$ may vary across repetitions. The empirical risk in group $k$ is

$$
\bar { L } _ { r j k } = \frac { 1 } { \left| T _ { r } \cap G _ { r k } \right| } \sum _ { i \in T _ { r } \cap G _ { r k } } L _ { r j i } ,
$$

and we summarize groupwise performance by

$$
W _ { r j } = \operatorname* { m a x } _ { k \leq J _ { r } } \bar { L } _ { r j k } , \qquad E _ { r j } = \frac { 1 } { J _ { r } } \sum _ { k = 1 } ^ { J _ { r } } ( \bar { L } _ { r j k } - \alpha ) _ { + } .
$$

Here, $W _ { r j }$ is the worst-group risk and $E _ { r j }$ is the mean positive group excess above the target. The latter gives equal weight to groups, even when their sizes difer or the groups overlap.

Groups are defined using setting-specific measures of prediction dificulty and are held fixed across methods within each repetition. The synthetic experiments use the known heterogeneity structure, while the real-data benchmarks use predictive uncertainty or task-relevant covariates to construct the groups. Detailed group definitions are given in Appendix C.5. The Superconductor groups were selected using the responses, so this grouped analysis is exploratory.

Prediction size is measured as the number of selected labels, classes, or pixels, or as interval width in regression. Tables 1–4 report the averages across the 20 repetitions of worst-group risk, mean positive group excess, prediction size, and marginal risk. Worstgroup risk is computed separately within each repetition before averaging. These summaries measure heterogeneity across the chosen groups rather than pointwise conditional risk.

## 5.3 Results

Across all eight settings, ReCIRC attains the lowest average worst-group risk and mean positive group excess, although in some settings its advantage over AA-CRC is within sampling variability, while its efect on prediction size varies across applications (Tables 1–4).

Groupwise risk. The synthetic experiments show consistent reductions in risk heterogeneity. In the heteroscedastic setting, ReCIRC attains an average worst-group risk of 0.1130, compared with 0.1312 for AA-CRC and 0.2340 for global CRC. In the latent-dificulty setting, the corresponding values are 0.1179, 0.1303, and 0.2329. The mean positive group excess is also smallest for ReCIRC in both settings. This ordering persists across distinct forms of heterogeneity: the first two designs involve one-dimensional dificulty mechanisms, whereas the XOR experiment introduces interaction-driven heterogeneity. In the latter, ReCIRC attains an average worst-group risk of 0.1130, compared with 0.1166 for AA-CRC and 0.1338 for global CRC, with corresponding mean positive group excesses of 0.0041, 0.0066, and 0.0145.

The same pattern extends to the five real-data benchmarks, where ReCIRC attains the smallest average worst-group risk and mean positive group excess in every setting. The diferences are particularly pronounced in RCV1, where worst-group risk decreases from 0.1722 under global CRC and 0.1410 under AA-CRC to 0.1144 under ReCIRC, and in Medical Insurance, where the corresponding values are 0.2021, 0.2061, and 0.1283; in the latter, AA-CRC does not improve on global CRC. In polyp segmentation and Letter Recognition, the reductions relative to global CRC are also large (from 0.3941 to 0.1092 and from 0.2861 to 0.1848), whereas the improvements over AA-CRC are smaller; in Superconductor, diferences among the three methods are modest. In Letter Recognition, the worst-group risk remains well above α for all methods: because every top-m set contains at least one class, easy inputs cannot reach the target risk, which pushes ab above α (Proposition 4.6).

Marginal risk. The groupwise comparisons should be interpreted together with the marginal operating level of each method (Table 4). The CRC guarantee concerns expected loss over calibration and a new observation and therefore does not require empirical test risk to remain below α in every repetition. Across the eight experiments, the mean marginal risk of ReCIRC ranges from 0.0908 to 0.0985, remaining close to the target $\alpha = 0 . 1 0$ . Global CRC also operates slightly below the target on average, while AA-CRC remains close to the target but slightly exceeds 0.10 in some settings. Relative to global CRC, which operates at a similar marginal level, the reductions in groupwise risk achieved by ReCIRC are therefore not explained by a lower marginal risk. Relative to AA-CRC, part of the advantage in the XOR and Superconductor settings is of the same order as the diference in marginal risk.

Risk-scale calibration. The risk-calibration diagnostics of Section 3.5 are computed on the test set of each repetition, which is used neither to fit the risk curves nor to calibrate the budget. Figure 4 shows the polyp-segmentation experiment. The empirical risk curves approximately follow the diagonal over $a \in [ 0 . 0 5 , 0 . 2 0 ]$ , both overall and within the five uncertainty groups, supporting the interpretation of a as a common group-level risk scale near the target $\alpha = 0 . 1 0$ . At larger budgets, the mean curves generally lie below the diagonal, indicating conservative risk relative to the budget. The remaining diagnostics, reported in Appendix D, show similar behavior near $\alpha ,$ except in the low-entropy Letter Recognition groups; at larger budgets, their departures from the diagonal are mostly conservative and reflect endpoint saturation, with the regression intervals collapsing to a point and the Letter Recognition sets reducing to a single class.

Prediction size. The reduction in risk heterogeneity can have diferent consequences for prediction size (Table 3). In polyp segmentation, the decrease in worst-group risk relative to global CRC is accompanied by an increase in average mask area from 15,538 to 22,455 pixels, approximately 45%. In RCV1, average prediction-set size increases from 5.016 to

![](images/3ac9ea319b44199085e24ac2de3c0375e60e3c2995783aefcc404ced4adca726.jpg)  
Figure 4: Held-out risk-calibration diagnostics for ReCIRC in the polyp-segmentation experiment. Each panel plots empirical average loss ${ \widehat { r } } _ { G } ( a )$ against the risk budget $^ { a , }$ overall or within one of five groups defined by a label-free uncertainty score. Loss is the fraction of positive pixels omitted, averaged across images. Thick blue curves show means over 20 repetitions; faint curves show individual repetitions, with groups fixed within each repetition. The dashed black diagonal marks equality between risk and budget, the red dash-dotted vertical line marks $\alpha = 0 . 1 0$ , and the shaded band highlights $a \in [ 0 . 0 5 , 0 . 2 0 ]$

15.392 labels, and smaller increases occur in the latent-dificulty and XOR experiments. AA-CRC shows increases of similar magnitude in polyp segmentation and RCV1 (20,258 pixels and 13.860 labels), which suggests that this cost reflects the additional protection given to dificult inputs rather than rectification specifically.

In other settings, however, improved groupwise risk is obtained together with smaller predictions. Relative to global CRC, ReCIRC reduces average set size from 3.100 to 2.210 in Letter Recognition, and average interval width by approximately 29% in Medical Insurance and 12% in Superconductor. In the heteroscedastic experiment, ReCIRC also produces slightly shorter intervals than global CRC, although AA-CRC attains the smallest average width. These results show that the size cost of redistributing risk across inputs is strongly application-dependent: in some settings more uniform risk requires substantially larger predictions, whereas in others adaptation allows both lower groupwise risk and smaller average predictions.

Table 1: Worst-group risk at $\alpha = 0 . 1 0$ . Entries report mean values across repetitions, with standard errors in parentheses. Bold indicates the lowest mean within each setting.
<table><tr><td>Setting</td><td>Global CRC</td><td>AA-CRC</td><td>ReCIRC-TabICLv2</td></tr><tr><td>Synthetic settings</td><td></td><td></td><td></td></tr><tr><td>Heteroscedastic</td><td>0.2340 (0.0061)</td><td>0.1312 (0.0047)</td><td>0.1130 (0.0039)</td></tr><tr><td>Latent difficulty</td><td>0.2329 (0.0044)</td><td>0.1303 (0.0038)</td><td>0.1179 (0.0022)</td></tr><tr><td>XOR interaction</td><td>0.1338 (0.0026)</td><td>0.1166 (0.0021)</td><td>0.1130 (0.0016)</td></tr><tr><td>Real benchmarks</td><td></td><td></td><td></td></tr><tr><td>Polyp segmentation</td><td>0.3941 (0.0135)</td><td>0.1148 (0.0042)</td><td>0.1092 (0.0027)</td></tr><tr><td>RCV1</td><td>0.1722 (0.0018)</td><td>0.1410 (0.0021)</td><td>0.1144 (0.0013)</td></tr><tr><td>Letter Recognition</td><td>0.2861 (0.0054)</td><td>0.1937 (0.0026)</td><td>0.1848 (0.0038)</td></tr><tr><td>Medical Insurance</td><td>0.2021 (0.0168)</td><td>0.2061 (0.0190)</td><td>0.1283 (0.0065)</td></tr><tr><td>Superconductor</td><td>0.1462 (0.0096)</td><td>0.1371 (0.0068)</td><td>0.1325 (0.0099)</td></tr></table>

Table 2: Mean positive group excess at $\alpha = 0 . 1 0 .$ . Entries report mean values across repetitions, with standard errors in parentheses. Bold indicates the lowest mean within each setting.
<table><tr><td>Setting</td><td>Global CRC</td><td>AA-CRC</td><td>ReCIRC-TabICLv2</td></tr><tr><td>Synthetic settings</td><td></td><td></td><td></td></tr><tr><td>Heteroscedastic</td><td>0.0371 (0.0019)</td><td>0.0083 (0.0010)</td><td>0.0043 (0.0010)</td></tr><tr><td>Latent difficulty</td><td>0.0331 (0.0012)</td><td>0.0093 (0.0013)</td><td>0.0058 (0.0011)</td></tr><tr><td>XOR interaction</td><td>0.0145 (0.0012)</td><td>0.0066 (0.0008)</td><td>0.0041 (0.0006)</td></tr><tr><td>Real benchmarks</td><td></td><td></td><td></td></tr><tr><td>Polyp segmentation</td><td>0.0588 (0.0027)</td><td>0.0032 (0.0008)</td><td>0.0027 (0.0006)</td></tr><tr><td>RCV1</td><td>0.0311 (0.0009)</td><td>0.0129 (0.0009)</td><td>0.0032 (0.0003)</td></tr><tr><td>Letter Recognition</td><td>0.0413 (0.0013)</td><td>0.0320 (0.0009)</td><td>0.0292 (0.0012)</td></tr><tr><td>Medical Insurance</td><td>0.0243 (0.0049)</td><td>0.0286 (0.0057)</td><td>0.0099 (0.0024)</td></tr><tr><td>Superconductor</td><td>0.0128 (0.0024)</td><td>0.0108 (0.0018)</td><td>0.0093 (0.0027)</td></tr></table>

Table 3: Mean prediction size or interval width at $\alpha = 0 . 1 0$ . Entries report mean values across repetitions, with standard errors in parentheses. Bold indicates the lowest mean within each setting.
<table><tr><td>Setting</td><td>Global CRC</td><td>AA-CRC</td><td>ReCIRC-TabICLv2</td></tr><tr><td>Synthetic settings</td><td></td><td></td><td></td></tr><tr><td>Heteroscedastic</td><td>1.764 (0.018)</td><td>1.615 (0.021)</td><td>1.660 (0.022)</td></tr><tr><td>Latent difficulty</td><td>5.380 (0.030)</td><td>6.528 (0.055)</td><td>6.813 (0.044)</td></tr><tr><td>XOR interaction</td><td>3.948 (0.027)</td><td>4.000 (0.023)</td><td>4.065 (0.016)</td></tr><tr><td>Real benchmarks</td><td></td><td></td><td></td></tr><tr><td>Polyp segmentation</td><td>15,538 (187)</td><td>20,258 (195)</td><td>22,455 (335)</td></tr><tr><td>RCV1</td><td>5.016 (0.027)</td><td>13.860 (0.194)</td><td>15.392 (0.242)</td></tr><tr><td>Letter Recognition</td><td>3.100 (0.069)</td><td>6.060 (0.046)</td><td>2.210 (0.022)</td></tr><tr><td>Medical Insurance</td><td>9,198.6 (282.2)</td><td>8,459.4 (221.6)</td><td>6,558.5 (210.3)</td></tr><tr><td>Superconductor</td><td>28.352 (0.214)</td><td>26.781 (0.102)</td><td>24.865 (0.119)</td></tr></table>

Table 4: Marginal risk at $\alpha = 0 . 1 0$ . Entries report mean values across repetitions, with standard errors in parentheses.
<table><tr><td>Setting</td><td>Global CRC</td><td>AA-CRC</td><td>ReCIRC-TabICLv2</td></tr><tr><td>Synthetic settings</td><td></td><td></td><td></td></tr><tr><td>Heteroscedastic</td><td>0.0963 (0.0022)</td><td>0.0975 (0.0017)</td><td>0.0908 (0.0021)</td></tr><tr><td>Latent difficulty</td><td>0.0924 (0.0017)</td><td>0.0993 (0.0023)</td><td>0.0941 (0.0021)</td></tr><tr><td>XOR interaction</td><td>0.0937 (0.0014)</td><td>0.1018 (0.0009)</td><td>0.0975 (0.0008)</td></tr><tr><td>Real benchmarks</td><td></td><td></td><td></td></tr><tr><td>Polyp segmentation</td><td>0.0917 (0.0027)</td><td>0.0848 (0.0010)</td><td>0.0956 (0.0014)</td></tr><tr><td>RCV1</td><td>0.0989 (0.0010)</td><td>0.1075 (0.0011)</td><td>0.0952 (0.0010)</td></tr><tr><td>Letter Recognition</td><td>0.0952 (0.0017)</td><td>0.0983 (0.0012)</td><td>0.0985 (0.0014)</td></tr><tr><td>Medical Insurance</td><td>0.0917 (0.0041)</td><td>0.0990 (0.0044)</td><td>0.0936 (0.0041)</td></tr><tr><td>Superconductor</td><td>0.0903 (0.0016)</td><td>0.1002 (0.0009)</td><td>0.0967 (0.0011)</td></tr></table>

## 6 Final Remarks

This paper discusses the unevenness of conformal risk control across the covariate space: the parameter calibrated by CRC is a raw threshold, so a single global value can overprotect easy inputs and underprotect hard ones. We proposed ReCIRC, a framework that reparameterizes the calibrated family so that its parameter becomes a local risk budget, obtained by inverting estimated local risk curves, and then applies ordinary CRC unchanged. The framework directly regresses observed losses on the input and threshold and retains the finite-sample marginal guarantee of CRC regardless of the quality of the fitted curves. Accurate curves yield approximate conditional risk control, while a direct risk-calibration curve provides a goodness-of-fit diagnostic for the estimated risk scale. Under uniformly consistent risk-curve estimates, a target attainable at almost every input, finer threshold and budget grids, and a growing calibration sample, the deployed conditional risk converges to the target.

Two consequences of learning the family are worth emphasizing. First, the fitted risk surface does not depend on the target level: a single fit of ${ \hat { R } } _ { { D } } ( \lambda \mid x )$ serves every α, since changing the target only re-scans the calibration table of Algorithm 1, and because the rectified loss is monotone in the budget, the resulting predictors are automatically nested across targets. Methods whose fitting objective is indexed by α, including AA-CRC as formulated here, generally require a separate fit at each level and do not automatically guarantee nesting across levels. Second, the risk-calibration curve of Section 3.5 is possible because a is an advertised conditional-risk level that can be confronted with realized loss; this has no analogue when the calibrated parameter is a raw threshold as in the standard formulation of CRC.

Despite ofering stronger local risk behavior, ReCIRC has some limitations. First, it requires an additional risk-training split to estimate the risk curves, which reduces the data available for the base model and for calibration. Second, the augmented loss-regression data add computation. Third, the finite-sample guarantee remains marginal: the local improvements depend on the quality of the estimated risk curves, and flat or discontinuous local risk curves can make the calibrated budget conservative at some inputs. Fourth, more uniform risk can require larger predictions: in polyp segmentation and RCV1, average prediction size increased substantially relative to global CRC, although in four of the other six settings it decreased.

This work opens several avenues for future research. A first direction is to remove the dedicated risk-training split through cross-fitting or cross-validation-style calibration arguments, which would eliminate rectification’s additional data requirement relative to standard CRC. A second direction is to combine rectification with partition-based calibration, in the spirit of Mondrian conformal methods (Vovk 2012, Cabezas et al. 2025), so as to obtain finite-sample local guarantees on the rectified scale; the price such methods pay in multiplicity may be lower here precisely because the rectified risk profile is already close to flat.

A third direction concerns the calibration wrapper itself. Because rectification acts on the decision family rather than on the calibration step, it composes with any procedure that calibrates a scalar along an ordered family: handing $\{ \widehat { f } _ { \mathcal { D } , a } ^ { R } \}$ to risk-controlling prediction sets (Bates et al. 2021) or to Learn-then-Test (Angelopoulos et al. 2025) requires no modification of either, and would deliver high-probability rather than in-expectation control over a budget that retains its local interpretation. The existing literature reaches non-marginal guarantees under these wrappers by a diferent route, refining where calibration happens: separate thresholds within a prespecified partition or hierarchy, or a reweighting class fixed in advance. Rectification is complementary, and in a useful sense orthogonal.

In conclusion, by turning the knob calibrated by CRC into a common conditional-risk scale, ReCIRC makes distribution-free risk control locally meaningful while keeping its guarantees intact. Code to implement ReCIRC and reproduce the experiments is available at repository on Github.

## 7 Disclosure statement

The authors report there are no competing interests to declare.

## 8 Data Availability Statement

Reproduction code for all experiments is provided as Supplementary Material and is also publicly available at repository on Github. The three synthetic settings described in

Section 5.1 are generated directly by the corresponding scripts using predefined random seeds and therefore require no external data. The real-data benchmarks used in the experiments are publicly available from the sources described below:

• Polyp segmentation: the prepared polyp-segmentation dataset, which pools images from Kvasir-SEG (Jha et al. 2020), CVC-ClinicDB (Bernal et al. 2015), CVC-ColonDB (Tajbakhsh et al. 2016), ETIS-LaribPolypDB (Silva et al. 2014), and CVC-300 (Vázquez et al. 2017), is obtained from the oficial AA-CRC repository, which provides separate training and test archives. Pixelwise prediction scores are produced with the PraNet segmentation model (Fan et al. 2020).

• RCV1 text multilabel classification: the RCV1-v2 feature vectors and topic labels (Lewis et al. 2004) are obtained using the fetch\_rcv1 loader provided by scikit-learn.

• Letter Recognition: the dataset (Frey & Slate 1991) is obtained from the UCI Machine Learning Repository.

• Medical Insurance: the dataset (Lantz 2013) is obtained from the Machine Learning with R datasets repository on GitHub.

• Superconductor: the dataset (Hamidieh 2018) is obtained from the UCI Machine Learning Repository.

## References

Angelopoulos, A. N. (2026), ‘Conformal risk control for non-monotonic losses’, arXiv preprint arXiv:2602.20151 .

https://arxiv.org/abs/2602.20151

Angelopoulos, A. N. & Bates, S. (2023), ‘Conformal prediction: A gentle introduction’, Foundations and Trends in Machine Learning 16(4), 494–591.

Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan, M. I. & Lei, L. (2025), ‘Learn then test: Calibrating predictive algorithms to achieve risk control’, The Annals of Applied Statistics 19(2), 1641–1662.

Angelopoulos, A. N., Bates, S., Fisch, A., Lei, L. & Schuster, T. (2024), Conformal risk control, in ‘The Twelfth International Conference on Learning Representations’.

Barber, R. F., Candès, E. J., Ramdas, A. & Tibshirani, R. J. (2021), ‘The limits of distribution-free conditional predictive inference’, Information and Inference: A Journal of the IMA 10(2), 455–482.

Bates, S., Angelopoulos, A., Lei, L., Malik, J. & Jordan, M. (2021), ‘Distribution-free, risk-controlling prediction sets’, Journal of the ACM 68(6), 1–34.

Bernal, J., Sánchez, F. J., Fernández-Esparrach, G., Gil, D., Rodríguez, C. & Vilariño, F. (2015), ‘WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians’, Computerized Medical Imaging and Graphics 43, 99–111.

Blot, V., Angelopoulos, A. N., Jordan, M. I. & Brunel, N. J.-B. (2025), Automatically adaptive conformal risk control, in ‘Proceedings of the 28th International Conference on Artificial Intelligence and Statistics’, Vol. 258 of Proceedings of Machine Learning Research, PMLR, pp. 19–27.

https://proceedings.mlr.press/v258/blot25a.html

Cabezas, L. M. C., Otto, M. P., Izbicki, R. & Stern, R. B. (2025), ‘Regression trees for fast and adaptive prediction intervals’, Information Sciences 686, 121369.

Chen, X. & Christensen, T. M. (2015), ‘Optimal uniform convergence rates and asymptotic normality for series estimators under weak dependence and weak conditions’, Journal of Econometrics 188(2), 447–465.

Chernozhukov, V., Wüthrich, K. & Zhu, Y. (2021), ‘Distributional conformal prediction’, Proceedings of the National Academy of Sciences 118(48), e2107794118.

Cruz Cabezas, L. M., Silva Santos, V., Ramos, T. & Izbicki, R. (2025), Epistemic uncertainty in conformal scores: A unified approach, in ‘Proceedings of the Forty-first Conference on Uncertainty in Artificial Intelligence’, Vol. 286 of Proceedings of Machine Learning Research, PMLR, pp. 443–470.

https://proceedings.mlr.press/v286/cruz-cabezas25a.html

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K. & Fei-Fei, L. (2009), ImageNet: A largescale hierarchical image database, in ‘IEEE Conference on Computer Vision and Pattern Recognition’, pp. 248–255.

Dheur, V., Fontana, M., Estievenart, Y., Desobry, N. & Ben Taieb, S. (2025), A unified comparative study with generalized conformity scores for multi-output conformal regression, in ‘Proceedings of the 42nd International Conference on Machine Learning’, Vol. 267 of Proceedings of Machine Learning Research, PMLR, pp. 13444–13485. https://proceedings.mlr.press/v267/dheur25a.html

Fan, D.-P., Ji, G.-P., Zhou, T., Chen, G., Fu, H., Shen, J. & Shao, L. (2020), PraNet: Parallel reverse attention network for polyp segmentation, in ‘International Conference on Medical Image Computing and Computer-Assisted Intervention’, Springer, pp. 263–273.

Frey, P. W. & Slate, D. J. (1991), ‘Letter recognition using Holland-style adaptive classifiers’, Machine Learning 6(2), 161–182.

Gibbs, I., Cherian, J. J. & Candès, E. J. (2025), ‘Conformal prediction with conditional guarantees’, Journal of the Royal Statistical Society Series B: Statistical Methodology 87(4), 1100–1126.

Guan, L. (2023), ‘Localized conformal prediction: A generalized inference framework for conformal prediction’, Biometrika 110(1), 33–50.

Hamidieh, K. (2018), ‘A data-driven statistical model for predicting the critical temperature of a superconductor’, Computational Materials Science 154, 346–354.

Han, X., Tang, Z., Ghosh, J. & Liu, Q. (2022), ‘Split localized conformal prediction’, arXiv preprint arXiv:2206.13092 .

https://arxiv.org/abs/2206.13092

Hore, R. & Barber, R. F. (2025), ‘Conformal prediction with local weights: Randomization enables robust guarantees’, Journal of the Royal Statistical Society Series B: Statistical Methodology 87(2), 549–578.

Izbicki, R., Lee, A. B. & Freeman, P. E. (2017), ‘Photo-z estimation: An example of nonparametric conditional density estimation under selection bias’, The Annals of Applied Statistics 11(2), 698–724.

Izbicki, R., Shimizu, G. & Stern, R. (2020), Flexible distribution-free conditional predictive bands using density estimators, in ‘Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics’, Vol. 108 of Proceedings of Machine Learning Research, pp. 3068–3077.

Izbicki, R., Shimizu, G. & Stern, R. B. (2022), ‘CD-split and HPD-split: Eficient conformal regions in high dimensions’, Journal of Machine Learning Research 23(87), 1–32.

Jha, D., Smedsrud, P. H., Riegler, M. A., Halvorsen, P., de Lange, T., Johansen, D. & Johansen, H. D. (2020), Kvasir-SEG: A segmented polyp dataset, in ‘MultiMedia Modeling: 26th International Conference, MMM 2020’, Vol. 11962 of Lecture Notes in Computer Science, Springer, pp. 451–462.

Jung, C., Noarov, G., Ramalingam, R. & Roth, A. (2023), Batch multivalid conformal prediction, in ‘International Conference on Learning Representations’.

Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M., Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q. & Petrov, S. (2019), ‘Natural Questions: A benchmark for question answering research’, Transactions of the Association for Computational Linguistics 7, 453–466.

Lantz, B. (2013), Machine Learning with R, Packt Publishing, Birmingham.

Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J. & Wasserman, L. (2018), ‘Distributionfree predictive inference for regression’, Journal of the American Statistical Association 113(523), 1094–1111.

Lei, J. & Wasserman, L. (2014), ‘Distribution-free prediction bands for non-parametric regression’, Journal of the Royal Statistical Society Series B: Statistical Methodology 76(1), 71–96.

Lewis, D. D., Yang, Y., Rose, T. G. & Li, F. (2004), ‘RCV1: A new benchmark collection for text categorization research’, Journal of Machine Learning Research 5, 361–397.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P. & Zitnick, C. L. (2014), Microsoft COCO: Common objects in context, in ‘European Conference on Computer Vision’, Springer, pp. 740–755.

Luo, R. & Zhou, Z. (2025), ‘Conditional conformal risk adaptation’, arXiv preprint arXiv:2504.07611 .

Masry, E. (1996), ‘Multivariate local polynomial regression for time series: Uniform strong consistency and rates’, Journal of Time Series Analysis 17(6), 571–599.

Meinshausen, N. (2006), ‘Quantile regression forests’, Journal of Machine Learning Research 7, 983–999.

Mossina, L. & Friedrich, C. (2025), ‘Controlling false positives in image segmentation via conformal prediction’, arXiv preprint arXiv:2511.15406 .

https://arxiv.org/abs/2511.15406

Papadopoulos, H. & Haralambous, H. (2011), ‘Reliable prediction intervals with regression neural networks’, Neural Networks 24(8), 842–851.

Plassier, V., Fishkov, A., Dheur, V., Guizani, M., Ben Taieb, S., Panov, M. & Moulines, E. (2025), Rectifying conformity scores for better conditional coverage, in ‘Proceedings of the 42nd International Conference on Machine Learning’, Vol. 267 of Proceedings of Machine Learning Research, pp. 49459–49492.

Qu, J., Holzmüller, D., Varoquaux, G. & Le Morvan, M. (2026), TabICLv2: A better, faster, scalable, and open tabular foundation model, in ‘Proceedings of the 43rd International Conference on Machine Learning’, Proceedings of Machine Learning Research.

Romano, Y., Patterson, E. & Candès, E. (2019), Conformalized quantile regression, in ‘Advances in Neural Information Processing Systems’, Vol. 32.

Romano, Y., Sesia, M. & Candès, E. J. (2020), Classification with valid and adaptive coverage, in ‘Advances in Neural Information Processing Systems’, Vol. 33, pp. 3581– 3591.

Shimodaira, H. (2000), ‘Improving predictive inference under covariate shift by weighting the log-likelihood function’, Journal of Statistical Planning and Inference 90(2), 227–244.

Silva, J., Histace, A., Romain, O., Dray, X. & Granado, B. (2014), ‘Toward embedded detection of polyps in WCE images for early diagnosis of colorectal cancer’, International Journal of Computer Assisted Radiology and Surgery 9(2), 283–293.

Stone, C. J. (1982), ‘Optimal global rates of convergence for nonparametric regression’, The Annals of Statistics 10(4), 1040–1053.

Tajbakhsh, N., Gurudu, S. R. & Liang, J. (2016), ‘Automated polyp detection in colonoscopy videos using shape and context information’, IEEE Transactions on Medical Imaging 35(2), 630–644.

Topol, E. J. (2019), ‘High-performance medicine: the convergence of human and artificial intelligence’, Nature Medicine 25, 44–56.

Vázquez, D., Bernal, J., Sánchez, F. J., Fernández-Esparrach, G., López, A. M., Romero, A., Drozdzal, M. & Courville, A. (2017), ‘A benchmark for endoluminal scene segmentation of colonoscopy images’, Journal of Healthcare Engineering 2017, 4037190.

Vovk, V. (2012), Conditional validity of inductive conformal predictors, in ‘Proceedings of the Asian Conference on Machine Learning’, Vol. 25 of Proceedings of Machine Learning Research, pp. 475–490.

Vovk, V., Gammerman, A. & Shafer, G. (2005), Algorithmic Learning in a Random World, Springer, New York.

Wang, H., Fu, T., Du, Y., Gao, W., Huang, K., Liu, Z., Chandak, P., Liu, S., Van Katwyk, P., Deac, A., Anandkumar, A., Bergen, K., Gomes, C. P., Ho, S., Kohli, P., Lasenby, J., Leskovec, J., Liu, T.-Y., Manrai, A., Marks, D., Ramsundar, B., Song, L., Sun, J., Tang,

J., Veličković, P., Welling, M., Zhang, L., Coley, C. W., Bengio, Y. & Zitnik, M. (2023), ‘Scientific discovery in the age of artificial intelligence’, Nature 620(7972), 47–60.

Xu, Y., Guo, W. & Wei, Z. (2023), Conformal risk control for ordinal classification, in ‘Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence’, Vol. 216 of Proceedings of Machine Learning Research, PMLR, pp. 2346–2355.

https://proceedings.mlr.press/v216/xu23a.html

Xu, Y., Ying, M., Guo, W. & Wei, Z. (2025), Two-stage risk control with application to ranked retrieval, in ‘Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence’, pp. 9104–9111.

https://www.ijcai.org/proceedings/2025/1012

Zecchin, M. & Simeone, O. (2024), Localized adaptive risk control, in ‘Advances in Neural Information Processing Systems’, Vol. 37, pp. 8165–8192. https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ 0f93c3e9b557980d93016671acd94bd2-Abstract-Conference.html

Zhang, L., Roth, A. & Zhang, L. (2024), Fair risk control: A generalized framework for calibrating multi-group fairness risks, in ‘Proceedings of the 41st International Conference on Machine Learning’, Vol. 235 of Proceedings of Machine Learning Research, PMLR, pp. 59783–59805.

https://proceedings.mlr.press/v235/zhang24be.html

## A Algorithms

Algorithm 1 ReCIRC: rectified conformal risk control   
Require: ordered family $\{ f _ { \lambda } : \lambda \in \Lambda \}$ ; threshold grid $\Lambda _ { M } ;$ loss $\ell \in [ 0 , B ]$ ; level $\alpha ;$ risk  
training set $\mathcal { D } ;$ calibration set $\mathscr { C } = \{ ( X _ { j } , Y _ { j } ) \} _ { j = 1 } ^ { n }$ ; budget grid $\mathcal { A } = \{ 0 = a _ { 0 } < a _ { 1 } < \cdot \cdot \cdot <$   
$a _ { L } \big \}$ in $[ 0 , B ]$   
1: Train and tune ${ \hat { R } } _ { { D } } ( \lambda \mid x )$ using only D (Algorithm $2 ;$ see Appendix B for the response  
model alternative)   
2: for $\ell = 0 , \ldots , L$ do   
3: for $j = 1 , \dotsc , n$ do   
4: $\lambda _ { j \ell }  \widehat { R } _ { D } ^ { - 1 } ( a _ { \ell } \mid X _ { j } )$ ▷ grid inversion of (6)   
5: $L _ { j \ell } \gets \ell ( f _ { \lambda _ { j \ell } } ( X _ { j } ) , Y _ { j } )$   
6: end for   
7: $\begin{array} { r } { \widehat { \mathcal { R } } _ { \mathrm { c a l } } ^ { R } ( a _ { \ell } ) \gets n ^ { - 1 } \sum _ { j = 1 } ^ { n } L _ { j \ell } } \end{array}$   
8: end for   
9: $\begin{array} { r } { \widehat { a } \gets \operatorname* { m a x } \left\{ a _ { \ell } : \frac { n } { n + 1 } \widehat { \mathcal { R } } _ { \mathrm { c a l } } ^ { R } ( a _ { \ell } ) + \frac { B } { n + 1 } \leq \alpha \right\} } \end{array}$   
10: return predictor $x \mapsto f _ { \widehat { R } _ { D } ^ { - 1 } ( \widehat { a } | x ) } ( x )$

Algorithm 2 Direct risk-curve estimation via loss regression   
Require: risk-training set $\boldsymbol { \mathcal { D } } \ = \ \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { m } ;$ design distribution $q$ on $\Lambda ;$ number of   
augmentations $K ;$ regression learner $g _ { \boldsymbol { \theta } } ;$ threshold grid $\Lambda _ { M }$   
1: for $i = 1 , \dots ,$ m and $k = 1 , \ldots , K$ do   
2: draw $\lambda _ { i k } \sim q ; Z _ { i k }  \ell ( f _ { \lambda _ { i k } } ( X _ { i } ) , Y _ { i } )$   
3: end for   
4: Train $g _ { \boldsymbol { \theta } } ( x , \lambda )$ on $\{ ( X _ { i } , \lambda _ { i k } , Z _ { i k } ) \}$ to approximate (8)   
5: For each queried $x ,$ let $\{ \widehat { R } _ { \cal D } ( \lambda \mid x ) \} _ { \lambda \in \Lambda _ { \cal M } }$ be the non-increasing (isotonic) regression of   
$\{ g _ { \theta } ( x , \lambda ) \} _ { \lambda \in \Lambda _ { M } }$ on λ   
6: return the monotonized curves $\hat { R } _ { \mathcal { D } } ( \cdot \mid x )$ on $\Lambda _ { M }$

## B Response-model risk-curve estimation

When a model for $Y \ | \ x$ is fitted on the risk-training set D and supplies a predictive distribution ${ \widehat { p } } ( y \mid x )$ from which one can draw samples, the risk curve can instead be estimated by modeling $Y \mid$ x and averaging the loss. On the threshold grid $\Lambda _ { M } .$ , for every calibration or test input $x ,$ draw Monte Carlo responses $Y ^ { ( 1 ) } ( x ) , \dots , Y ^ { ( S ) } ( x ) \sim { \widehat { p } } ( \cdot \mid x )$ and set

$$
\widehat { R } _ { \cal D } ( \lambda \mid x ) = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \ell ( f _ { \lambda } ( x ) , Y ^ { ( s ) } ( x ) ) , \qquad \lambda \in \Lambda _ { \cal M } .
$$

If $\lambda \mapsto \ell ( f _ { \lambda } ( x ) , y )$ is monotone for every $y .$ using the same sampled responses for all $\lambda \in \Lambda _ { M }$ makes the Monte Carlo curve automatically monotone. The estimated curve is then inverted on the grid as in (6). Algorithm 3 summarizes the procedure.

This alternative is useful when the base model already supplies reliable samples or probabilities for $Y \mid$ x: the same samples can be reused across losses and decision families, and pointwise monotonicity is preserved without post-processing. Its disadvantages are the cost of sampling or scoring a large output space and the need to model aspects of the conditional law that may be irrelevant to the loss of interest. Direct loss regression avoids those costs by targeting a scalar response, at the price of a loss-specific fit and explicit monotonicity handling.

Algorithm 3 Alternative risk-curve estimation via a model for $Y \mid x$   
Require: predictive model ${ \widehat { p } } ( y \mid x )$ fitted on $\mathcal { D } ;$ threshold grid $\Lambda _ { M } ;$ Monte Carlo size $S ;$   
query point x   
1: Draw $Y ^ { ( 1 ) } ( x ) , \dots , Y ^ { ( S ) } ( x ) \sim { \widehat { p } } ( \cdot \mid x )$   
2: for $\lambda \in \Lambda _ { M }$ do   
3: $\begin{array} { r } { \widehat { R } _ { \mathcal { D } } ( \lambda \mid x )  S ^ { - 1 } \sum _ { s = 1 } ^ { S } \ell ( f _ { \lambda } ( x ) , Y ^ { ( s ) } ( x ) ) } \end{array}$ ▷ same draws for all λ preserve   
monotonicity   
4: end for   
5: return $\{ \widehat { R } _ { \cal D } ( \lambda \mid x ) \} _ { \lambda \in \Lambda _ { \cal M } }$

## C Experimental settings and implementation details

Table 5: Overview of the experimental settings. All experiments are repeated 20 times. The AA-CRC feature map column summarizes how the input-dependent score threshold is parameterized. In the RF-based settings, a random forest is fitted on D using a task-specific scalar measure of prediction dificulty, and Φ(x) is the concatenation of one-hot indicators of the leaves reached by x. Split sizes refer to the risk-training partition D, calibration partition C, and test partition T of ReCIRC. Base-model fitting, ReCIRC grids and representations, AA-CRC regularization, and evaluation-group definitions are given in the subsections below.
<table><tr><td>Setting</td><td>Base model</td><td>AA-CRC feature map</td><td>Splits (risk train / cal / test)</td></tr><tr><td colspan="4">Synthetic settings</td></tr><tr><td>Heteroscedastic</td><td>GAM mean with absolute-residual intervals (regression)</td><td>RF leaf indicators; RF target: absolute residual</td><td>1,000 / 500 / 1,000</td></tr><tr><td></td><td>Latent difficulty Synthetic label scores, 50 labels Linear: intercept, (multilabel)</td><td>difficulty covariate and score summaries</td><td>1,000 / 500 / 1,000</td></tr><tr><td>XOR interaction Quantile random forest</td><td>(regression)</td><td>(standardized on D) RF leaf indicators; RF target: minimum covering multiplier</td><td>6,400 / 4,800 / 4,800</td></tr><tr><td colspan="4">Real benchmarks</td></tr><tr><td>Polyp segmentation</td><td>PraNet pixelwise scores (segmentation) One-vs-rest logistic on TF-IDF, Score summaries</td><td>90 probability quantiles (label-free)</td><td>700 / 700 / 398 2,000 / 1,500 / 14,500</td></tr><tr><td>RCV1</td><td>103 labels (multilabel)</td><td>and all 103 label scores</td><td></td></tr><tr><td>Letter Recognition</td><td>Logistic classifier, 26 classes (multiclass)</td><td>Input covariates and probability summaries</td><td>6,000 / 4,000 / 4,000</td></tr><tr><td>Medical Insurance</td><td>Quantile random forest (regression)</td><td>RF leaf indicators; RF target: minimum covering multiplier</td><td>535 / 401 / 402</td></tr><tr><td>Superconductor Quantile random forest</td><td>(regression)</td><td>RF leaf indicators; RF target: minimum covering multiplier</td><td>8,505 / 6,378 / 6,380</td></tr></table>

## C.1 Base-model fitting and risk-training predictions

The base predictive models and the samples used to fit them are as follows. In the heteroscedastic experiment, the GAM mean model is fitted on an independent sample of size 500, separate from the 1,000 observations in the risk-training set D. In the synthetic multilabel experiment, the label scores are generated directly from the data-generating mechanism, so there is no separately fitted base predictor. In polyp segmentation, we use the precomputed PraNet probability maps supplied with the benchmark and do not retrain PraNet.

For RCV1, the one-vs-rest logistic classifier is fitted on a separate base sample of 12,000 documents, after which its predicted probabilities are computed on the conformal pool used for risk training, calibration, and evaluation. For Letter Recognition, the logistic classifier is fitted on the 30% base split, corresponding to 6,000 observations; the remaining data are split into risk-training, calibration, and test samples.

Medical Insurance, Superconductor, and XOR use quantile random forests (QRFs). In these three settings, the QRF is fitted on D. For observations in D, however, the QRFderived location and scale features and the losses used to train the ReCIRC risk estimator are computed from out-of-bag quantile predictions. Thus, an observation’s own QRF fit is not used to construct its risk-regression target. For calibration and test observations, ordinary QRF predictions from the model fitted on D are used.

## C.2 ReCIRC implementation details

The heteroscedastic experiment uses 101 equally spaced values of λ in [0, 4], 101 risk budgets in [0, 1], and K = 15 threshold draws from a uniform distribution on [0, 4] per risk-training observation. The latent-dificulty experiment uses 101-point threshold and budget grids on [0, 1], with K = 20 thresholds sampled uniformly from the threshold grid. Polyp segmentation uses 51 thresholds on [0, 1], 101 budget values on [0, 1], and $K = 2 0$ uniform threshold draws on [0, 1]. RCV1 uses all 26 equally spaced threshold values in [0, 1] for every risk-training observation and a 101-point budget grid on [0, 1]. Letter Recognition uses the top-m family, $m = 1 , \ldots , 2 6$ , with K = 8 sampled values of m and an 81-point budget grid on [0, 1]. Medical Insurance, Superconductor, and XOR use 80 thresholds on [0, 4], 201 budgets on [0, 1], and 16 deterministic threshold anchors.

The nested family is the symmetric interval $[ \widehat { \mu } ( x ) - \lambda , \widehat { \mu } ( x ) + \lambda ]$ in the heteroscedastic experiment. The synthetic multilabel experiment includes labels with scores at least $1 - \lambda .$ and the polyp experiment includes pixels whose PraNet probabilities are at least $1 - \lambda$ RCV1 uses the score-threshold family internally; equivalently, writing $\lambda = 1 - \tau$ gives the protective parameterization $\{ k : { \widehat { p } } _ { k } ( x ) \geq 1 - \lambda \}$ . Letter Recognition uses the m classes with largest predicted probabilities.

In Medical Insurance, Superconductor, and XOR, the interval is centered at the estimated conditional median. Its lower and upper scales are given by the diferences between the estimated 0.50 and 0.05 quantiles and between the estimated 0.95 and 0.50 quantiles, respectively, with both scales multiplied by λ. These experiments use the asymmetric miscoverage loss defined in Section 5.2.

Medical Insurance, Superconductor, and XOR use the anchor implementation described in Section 3.3. Separate TabICLv2 regressors are fitted at 16 approximately equally spaced threshold anchors. The predicted anchor risks are clipped and made non-increasing in λ, interpolated to the complete 80-point grid using PCHIP, and monotonized once more before inversion.

In the regression settings, the numerical threshold grid ends at $\lambda = 4$ , which need not be fully protective for every possible response. The theoretical family may be viewed as augmented by a fully protective endpoint, corresponding to the whole response space, which serves only as a fallback ensuring feasibility at $a _ { 0 } = 0$ . This fallback was not required in any of the reported repetitions: the calibrated budget satisfied $\widehat { a } > 0$ throughout, so all reported predictions were obtained from the finite numerical grid.

## C.3 Risk-regression representations

The representation supplied to the conditional-risk estimator is task-specific. The heteroscedastic experiment uses x together with λ. The synthetic multilabel experiment uses the observed covariates together with the mean, standard deviation, maximum, mean of the five largest scores, sum, and gap between the two largest label scores. Polyp segmentation represents each PraNet probability map by 90 empirical quantiles. RCV1 uses the three largest label scores, their sum, mean and standard deviation, summed binary entropy, counts of scores above 0.05, 0.10, 0.20, and 0.50, together with all 103 individual scores. Letter Recognition combines the original covariates with the five largest predicted probabilities, normalized predictive entropy, the top-two probability gap, top-three and top-five probability mass, the proportions of classes with probability at least 0.05 and 0.10, and the standard deviation of the probability vector. The corresponding decision coordinate is appended in the joint risk-regression settings.

For Medical Insurance, Superconductor, and XOR, the representation contains the original covariates, the estimated conditional median, the lower and upper QRF scales, their sum, the relative contribution of the upper scale, and the logarithm of the total scale. For observations in D, these QRF-derived quantities and the corresponding loss targets are computed from out-of-bag predictions.

TabICLv2 is used throughout. RCV1 uses four estimators with a context of 2,000 risk-training observations. Letter Recognition limits the augmented risk-training sample to 6,000 rows. The QRF-based experiments use four TabICLv2 estimators per anchor; Superconductor and XOR limit the risk-training context to 4,000 observations per repetition.

## C.4 AA-CRC implementation

All AA-CRC variants use the objective and gradient of the authors’ implementation and are optimized with SLSQP. The RF-leaf variants (heteroscedastic, Medical Insurance, Superconductor, and XOR) use no ridge penalty and impose box constraints on the coeficients; in the QRF-based settings, the minimum-covering-multiplier target used to train the random forest on D is computed from out-of-bag QRF predictions. The latentdificulty, RCV1, and Letter Recognition variants use ridge coeficient 0.01 on the slope coeficients only, and polyp segmentation uses it on all coeficients; in Letter Recognition, the coeficient is increased by a factor of 10 if the optimizer reaches the numerical boundary.

As in the original procedure, θ is fitted on C and deployed without further calibration, except in two settings. In polyp segmentation, θ is fitted on ${ \mathcal { D } } \cup { \mathcal { C } } .$ , which, if anything, favors AA-CRC; its objective is evaluated on $6 4 \times 6 4$ downsampled masks, excluding empty ones, with the same 90 probability quantiles used by ReCIRC as features. In Letter Recognition, the original configuration gave a marginal risk of about 0.05 at $\alpha = 0 . 1 0$ , because rank-based scores confine the attainable risk to the steps of the top-m family; we therefore use the continuous adaptive-prediction-set score (Romano et al. 2020), fit θ on D, and calibrate a scalar ofset on C by CRC, mirroring the data allocation of ReCIRC and restoring the marginal guarantee of AA-CRC. In all settings except Letter Recognition, because θ is fitted on C and deployed without the test point in the optimization, the marginal guarantee of AA-CRC holds only approximately, with a slack that grows with the dimension of Φ; this accounts for its marginal risk slightly above α in some settings, notably RCV1.

## C.5 Evaluation groups

The groups used for worst-group risk and mean positive group excess are evaluation devices only and are not supplied to any method. The heteroscedastic experiment uses five equal mass bins of the true conditional scale $\sigma ( x ) = 0 . 2 + 0 . 6 | x |$ , while the latent-dificulty experiment uses five equal-mass bins of the known latent dificulty. Polyp segmentation uses five bins of a label-free uncertainty score based on the average proximity of the PraNet probabilities to 0.5, with cutpoints fitted on D. RCV1 uses ten bins of summed binary predictive entropy, also with cutpoints fitted on $D _ { : }$ , while Letter Recognition uses five equal-mass bins of normalized predictive entropy.

Medical Insurance uses eight potentially overlapping groups obtained by crossing smoker status with the indicator $\mathrm { B M I } \geq 3 0$ and, separately, with an indicator for age above its sample median; groups with fewer than 30 test observations are omitted. Superconductor uses the four intersections obtained by median-splitting range\_ThermalConductivity and wtd\_std\_ThermalConductivity; groups with fewer than 50 test observations are omitted. Because these two variables were selected using their association with the response in the full dataset, this grouped analysis is exploratory. XOR uses the four sign quadrants of $( X _ { 1 } , X _ { 2 } )$

## D ReCIRC Calibration Diagnostics

Each figure includes a panel for the overall group $G = \mathcal { X }$ . The additional groups were fixed within each repetition as follows. In the heteroscedastic and latent-dificulty simulations, they are five equal-mass bins of, respectively, the known noise scale $\sigma ( X )$ and the known latent dificulty. For polyp segmentation, they are five bins of the label-free score $\begin{array} { r } { | \mathcal { U } | ^ { - 1 } \sum _ { u \in \mathcal { U } } \{ 1 - } \end{array}$ $2 | \widehat { p } _ { u } ( X ) - 1 / 2 | \}$ , with cutpoints fitted on the risk-training images. For RCV1, they are ten bins of the summed binary entropy of the 103 label scores, with cutpoints fitted on the risk-training split; for Letter Recognition, they are five equal-mass bins of normalized predictive entropy; and for Medical Insurance, five equal-mass bins of the total QRF scale s<sub>−</sub> $( X ) + s _ { + } ( X )$ . The Superconductor groups are the four intersections obtained by median-splitting the two covariates with the largest absolute full-data correlations with the response (range\_ThermalConductivity and wtd\_std\_ThermalConductivity); because this selection used the responses, that diagnostic is exploratory. Finally, the XOR groups are the four sign quadrants of $( X _ { 1 } , X _ { 2 } )$ . Except for the stated Superconductor exception, group construction uses only covariates, base-model outputs, or known simulation quantities, and never the held-out outcomes used to compute ${ \widehat r } _ { G } ( a )$

![](images/bd399ba5a530a4995631fbeb4c545f2b52d838f356e3c65e2e9286a63f060dad.jpg)  
Figure 5: Held-out risk-calibration diagnostics for ReCIRC in the heteroscedastic experiment, overall and within five equal-mass bins of the known noise scale $\sigma ( X )$ . Thick curves show mean empirical loss over 20 repetitions; faint curves show individual repetitions, with groups fixed within each repetition.

![](images/2220c5c3ea41ead5ae82eed36c81cb40e5381db02b85ce1054083dfa6075c053.jpg)  
Figure 6: Held-out risk-calibration diagnostics for ReCIRC in the latent-dificulty experiment, overall and within five equal-mass bins of the known latent dificulty. Thick curves show mean empirical loss over 20 repetitions; faint curves show individual repetitions, with groups fixed within each repetition.

![](images/54b351bbb7f4115d18e305a9af7ecbbb07eaa738da13145fd10a64f4df34ad74.jpg)  
Figure 7: Held-out risk-calibration diagnostics for ReCIRC in RCV1. Results are shown over 20 repetitions, using fixed groups within each repetition; faint curves correspond to individual repetitions.

![](images/ac77a94b73805434288f204253ac051e1b422162d1462bde8c0039faaa52fa92.jpg)  
Figure 8: Held-out risk-calibration diagnostics for ReCIRC in Letter Recognition. Results are shown over 20 repetitions, using fixed groups within each repetition; faint curves correspond to individual repetitions.

![](images/d61afa75eea98ef7c0d905f13027b1d5fd615405a00b43fc9efb7857d6929d58.jpg)  
Figure 9: Held-out risk-calibration diagnostics for ReCIRC in Medical Insurance. Results are shown over 20 repetitions, using fixed groups within each repetition; faint curves correspond to individual repetitions.

![](images/553ee9cd14c4e67107bb875b1ffcabda372ff25012b33c3a063b4b0576bbab0b.jpg)  
Figure 10: Held-out risk-calibration diagnostics for ReCIRC in Superconductor. Results are shown over 20 repetitions, using fixed groups within each repetition; faint curves correspond to individual repetitions.

![](images/f232dc8d3fc24200a2d8ea067504d8bef68f0343cf548b8bfff8f6c82ba7ad87.jpg)  
Figure 11: Held-out risk-calibration diagnostics for ReCIRC in the XOR interaction experiment. Results are shown over 20 repetitions, using fixed groups within each repetition; faint curves correspond to individual repetitions.

Figure 4 in Section 5 and the seven figures in this appendix together show the held-out risk-calibration diagnostics for ReCIRC across the eight experimental settings. Each figure plots empirical average loss ${ \widehat { r } } _ { G } ( a )$ against the risk budget a. Thick blue curves show means over 20 repetitions; faint curves show individual repetitions, with groups fixed within each repetition. The dashed black diagonal marks equality between risk and budget, the red dash-dotted vertical line marks $\alpha = 0 . 1 0$ , and the shaded band highlights $a \in [ 0 . 0 5 , 0 . 2 0 ]$ Curves below the diagonal indicate conservative risk relative to the advertised budget. Such departures can arise from risk-estimation error, discrete decision choices, or saturation at the least protective endpoint. In the regression settings, the plateaus arise because the interval at $\lambda = 0$ collapses to a point whose risk caps the attainable budget: about 0.5 under $\ell _ { \mathrm { a s y m } } ,$ and increasing with the noise scale under $\ell _ { \mathrm { e x c } }$ . In Letter Recognition, ReCIRC selects among nonempty top-m prediction sets. Once $m = 1$ , further increases in the budget leave the prediction unchanged, and the loss equals the top-1 classification error indicator. This explains the low plateaus in the low-entropy groups. Moreover, the discrete choices of m can produce conservative risk even before saturation and even with exact conditional-risk estimates. Departures above the diagonal are small, at most about 0.03, and occur mainly in the noisiest heteroscedastic bins and, at small budgets, in the harder latent-dificulty bins.

## E Additional theoretical results and representations

## E.1 Most-input and group consequences

Proposition E.1 (Most-input and group conditional-risk bounds). Under the conditions of Proposition 4.6, the following conclusions hold conditionally on D, for almost every realization of D; throughout, $\mathbb { P } _ { X }$ and the $L ^ { p } ( P _ { X } )$ norm are evaluated with that realization fixed. Suppose $\varepsilon _ { M } \in L ^ { p } ( P _ { X } )$ for some $p \in [ 1 , \infty )$ . Then, for every $a \in [ 0 , B ]$ and $t > 0$

$$
\mathbb { P } _ { X } \left\{ x : \mathbb { E } \left[ \ell \left( \widehat { f } _ { \mathcal { D } , a } ^ { R } ( X ) , Y \right) \Big | X = x , \mathcal { D } \right] > a + t \right\} \leq \frac { \| \varepsilon _ { M } \| _ { L ^ { p } \left( P _ { X } \right) } ^ { p } } { t ^ { p } } .\tag{11}
$$

For every measurable group $G \subseteq { \mathcal { X } }$ with $\mathbb { P } _ { X } \left( X \in G \right) > 0$

$$
\begin{array} { r } { \mathbb { E } \left\{ \ell \left( \widehat { f } _ { \mathcal { D } , a } ^ { R } ( X ) , Y \right) \Big | X \in G , \mathcal { D } \right\} \leq a + \mathbb { P } _ { X } ( X \in G ) ^ { - 1 / p } \| \varepsilon _ { M } \| _ { L ^ { p } ( P _ { X } ) } . } \end{array}\tag{12}
$$

If the calibration and test pairs are i.i.d., the budget-grid condition holds, $0 < \alpha \le B$ , and $n \geq B / \alpha - 1$ , both conclusions remain valid after conditioning on the calibration set and replacing a by ab.

The group risk on the left side of (12) is estimated directly by the held-out riskcalibration curve (9). Thus the diagnostic provides a practical check of the population quantity controlled by this proposition.

## E.2 Rate transfer from risk-curve estimation

Proposition E.2 (Rate transfer from risk-curve estimation). Suppose Assumptions $\it 4 . 1$ and 4.2 hold, and let $\mathcal { D } _ { m }$ be a risk-training sample. Let $h _ { M } : = \mathrm { m a x } _ { j < M } ( \lambda _ { j + 1 } - \lambda _ { j } )$ . Suppose $R ( \cdot \mid x )$ is Lipschitz and, for a sequence $\eta _ { m }  0$

$$
\operatorname* { m a x } _ { \lambda \in \Lambda _ { M } } \vert R ( \lambda \mid x ) - \widehat { R } _ { \mathcal { D } _ { m } } ( \lambda \mid x ) \vert = O _ { \mathbb { P } } ( \eta _ { m } ) .
$$

Then

$$
\operatorname* { s u p } _ { 0 \leq a \leq R ( \lambda _ { \operatorname* { m i n } } | x ) } |  { \mathbb { E } } \{ \ell  \widehat { f } _ { \mathcal { D } _ { m } , a } ^ { R } ( X ) , Y ) | X = x , \mathcal { D } _ { m } \} - a | = O _ { \mathbb { P } } ( \eta _ { m } + h _ { M } ) .
$$

$I f ,$ in addition, the calibration and test pairs are $i . i . d .$ , Assumption 4.3 holds, $0 < \alpha \le B$ 2 $n \geq B / \alpha - 1$ , and, along the asymptotic sequence under consideration,

$$
\mathbb { P } \{ \widehat { a } \le R ( \lambda _ { \operatorname* { m i n } } \mid x ) \} \longrightarrow 1 ,
$$

then also

$$
\left| \mathbb { E } \left\{ \ell \left( \widehat { f } _ { \mathcal { D } _ { m } , \widehat { a } } ^ { R } ( X ) , Y \right) \Big | X = x , \mathcal { D } _ { m } , \mathcal { C } \right\} - \widehat { a } \right| = O _ { \mathbb { P } } ( \eta _ { m } + h _ { M } ) .
$$

## E.3 A general deployed-risk rate

Theorem E.3 (General rate for the deployed conditional risk). Along a sequence $n  \infty ,$ let the risk-training size satisfy $m = m ( n ) \to \infty$ and let $M = M ( n )$ . All quantities below are indexed along this sequence: $\rho _ { m } , \ \eta _ { m } , \ h _ { M }$ , and $h _ { n }$ abbreviate $\rho _ { m ( n ) } , \ \eta _ { m ( n ) }$ , the spacing of $\Lambda _ { M ( n ) }$ , and the spacing of $\mathcal { A } _ { n }$ , and every $O _ { \mathbb { P } }$ statement is taken as $n  \infty$ . Suppose Assumptions $\it 4 . 1 \mathrm { - } \it 4 . 3$ hold along this sequence, with the calibration and test pairs $i . i . d .$ . Let $\mathcal { D } _ { m }$ be the risk-training sample, and let $\mathcal { A } _ { n } = \{ 0 = a _ { 0 , n } < \cdot \cdot \cdot < a _ { L _ { n } , n } = B \}$ have maximum spacing $h _ { n } \to 0$ . Fix $\alpha \in ( 0 , B )$ and suppose $n \geq B / \alpha - 1$ . For an open interval $I \subset ( 0 , B )$ containing α, suppose that, for some deterministic $\rho _ { m } \to 0$

$$
\operatorname* { s u p } _ { a \in I } \left| \mathbb { E } \left\{ \ell \left( \widehat { f } _ { { D _ { m } } , a } ^ { R } ( X ) , Y \right) \Big | \mathscr { D } _ { m } \right\} - a \right| = O _ { \mathbb { P } } ( \rho _ { m } ) .
$$

Then

$$
| \widehat { \boldsymbol { a } } - \boldsymbol { \alpha } | = O _ { \mathbb { P } } ( \rho _ { m } + n ^ { - 1 / 2 } + h _ { n } ) .
$$

If, in addition, for a fixed x, the interval I lies in $( 0 , R ( \lambda _ { \operatorname* { m i n } } \mid x ) ) , R ( \cdot \mid x )$ is Lipschitz, the threshold grid has maximum spacing $h _ { M } : = \operatorname* { m a x } _ { j < M } ( \lambda _ { j + 1 } - \lambda _ { j } ) \to 0$ , and, for a sequence $\eta _ { m }  0 _ { ; }$

$$
\operatorname* { m a x } _ { \lambda \in \Lambda _ { M } } \vert R ( \lambda \mid x ) - \widehat { R } _ { \mathcal { D } _ { m } } ( \lambda \mid x ) \vert = O _ { \mathbb { P } } ( \eta _ { m } ) ,
$$

then

$$
\begin{array} { r } { \left| \mathbb { E } \left\{ \ell \left( \widehat f _ { \mathcal D _ { m } , \widehat { a } } ^ { R } ( X ) , Y \right) \Big | X = x , \mathcal D _ { m } , \mathcal C \right\} - \alpha \right| = O _ { \mathbb { P } } ( \rho _ { m } + \eta _ { m } + h _ { M } + n ^ { - 1 / 2 } + h _ { n } ) . } \end{array}
$$

The $\rho _ { m }$ condition is a high-level marginal risk-scale condition, not a calibration conclusion. A primitive suficient condition is uniform risk-curve accuracy: if the error bound and Lipschitz constant in Proposition E.2 hold uniformly over $x ,$ then that proposition and integration over X give $\rho _ { m }$ of order $\eta _ { m } + h _ { M }$

## E.4 Rectification as a local quantile transformation

The rectification map depends on the original family and loss only through the local risk curve. This leads to a canonical representation that connects the construction to an ordinary quantile transformation. Normalize B = 1 for this subsection.

Proposition E.4 (Canonical representation). Fix x and assume $\lambda \mapsto R ( \lambda \mid x )$ is nonincreasing and right-continuous, with $R ( \lambda _ { \operatorname* { m a x } } \mid x ) = 0$ . Then:

(a) $F _ { x } ( \lambda ) : = 1 - R ( \lambda \mid x )$ is a cumulative distribution function, so there exists a random variable $Z _ { x }$ with $\mathbb { P } ( Z _ { x } > \lambda ) = R ( \lambda \mid x )$ ;

(b) the risk-curve inverse is a generalized quantile:

$$
R ^ { - 1 } ( a \mid x ) = F _ { x } ^ { - 1 } ( 1 - a ) , \qquad F _ { x } ^ { - 1 } ( u ) : = \operatorname* { i n f } \{ \lambda \in \Lambda : F _ { x } ( \lambda ) \geq u \} .
$$

In particular, this convention gives $F _ { x } ^ { - 1 } ( 0 ) = \lambda _ { \operatorname* { m i n } }$ ;

(c) the oracle rectified rule satisfies $f _ { a } ^ { R } ( x ) = f _ { { F } _ { x } ^ { - 1 } ( 1 - a ) } ( x )$ , and the canonical problem defined by the threshold rules $f _ { \lambda } ^ { \mathrm { c a n } } ( x ) = \{ z : z \leq \lambda \}$ with 0–1 loss $1 \{ z > \lambda \}$ applied to $Z _ { x }$ has exactly the same local risk curve and rectification map as the original problem.

Thus risk rectification can be viewed as replacing the raw threshold by a local quantile of a latent variable whose upper tail is the conditional risk curve. In the miscoverage special case of Section 3.4, this latent variable has the conditional law of the score $S ( x , Y )$ ; for a general bounded monotone loss, it is only a representation of the risk curve and need not correspond to an observed score.

## F Proofs

## F.1 Proof of Theorem 4.4

Condition on $\mathcal { D } _ { : }$ so that the rectified family $\{ \widehat { f } _ { D , a } ^ { R } : a \in [ 0 , B ] \}$ is fixed, and write

$$
L _ { j } ( a ) : = \ell \left( \widehat { f } _ { \mathcal { D } , a } ^ { R } ( X _ { j } ) , Y _ { j } \right) , \qquad j = 1 , \ldots , n + 1 .
$$

We first record two structural facts. First, the map $a \mapsto \widehat { R } _ { \mathcal { D } } ^ { - 1 } ( a \mid x )$ is non-increasing on $[ 0 , B ]$ : for $a > 0$ the sets $\{ \lambda \in \Lambda _ { M } : \widehat { R } _ { \mathcal { D } } ( \lambda \mid x ) \leq a \}$ grow with a, so their minima decrease, and the convention $\widehat { R } _ { \mathcal { D } } ^ { - 1 } ( 0 \mid x ) = \lambda _ { \mathrm { m a x } }$ returns the largest possible threshold. By (1), each $L _ { j }$ is therefore nondecreasing in a, with values in $[ 0 , B ]$ ; no separate regularity assumption is needed. Second, at $a = 0$ the convention deploys $f _ { \lambda _ { \mathrm { m a x } } }$ , so $L _ { j } ( 0 ) = 0$ by Assumption 4.2, and the condition $\begin{array} { r } { n \geq B / \alpha - 1 \ \mathrm { g i v e s } \ \frac { n } { n + 1 } \cdot 0 + \frac { B } { n + 1 } \leq \alpha \colon } \end{array}$ the constraint set in (7) contains $a _ { 0 } = 0$ , so ab is a well-defined maximum over a finite nonempty set, and it satisfies the CRC constraint by construction.

Now define the augmented selection rule

$$
{ \widehat { a } } ^ { \prime } : = \operatorname* { m a x } \left\{ a \in { \mathcal { A } } : { \frac { 1 } { n + 1 } } \sum _ { j = 1 } ^ { n + 1 } L _ { j } ( a ) \leq \alpha \right\} ,
$$

which is well defined for the same reason. If a is feasible in (7), then $\begin{array} { r } { \frac { 1 } { n + 1 } \sum _ { j = 1 } ^ { n + 1 } L _ { j } ( a ) \leq } \end{array}$ $\begin{array} { r } { \frac { n } { n + 1 } \widehat { \mathcal { R } } _ { \mathrm { c a l } } ^ { R } ( a ) + \frac { B } { n + 1 } \leq \alpha } \end{array}$ because $L _ { n + 1 } ( a ) \leq B$ ; hence $\widehat { a } \leq \widehat { a } ^ { \prime }$ , and monotonicity of the losses gives $L _ { n + 1 } ( \widehat { a } ) \leq L _ { n + 1 } ( \widehat { a } ^ { \prime } )$ . Conditional on D, each $L _ { j } ( \cdot )$ is a fixed measurable transformation of $( X _ { j } , Y _ { j } )$ , so the loss paths $L _ { 1 } , \ldots , L _ { n + 1 }$ are exchangeable by Assumption 4.1, and $\widehat { a } ^ { \prime }$ is a symmetric function of them. Therefore

$$
\mathbb { E } \{ L _ { n + 1 } ( \widehat { a } ^ { \prime } ) \mid \mathcal { D } \} = \mathbb { E } \left\{ \frac { 1 } { n + 1 } \sum _ { j = 1 } ^ { n + 1 } L _ { j } ( \widehat { a } ^ { \prime } ) \Bigg | \mathcal { D } \right\} \leq \alpha ,
$$

where the inequality holds pointwise because $\widehat { a } ^ { \prime }$ is itself feasible—this is where the finite grid matters, as it makes the maximum attained. Combining the two displays gives ${ \mathbb E } \{ L _ { n + 1 } ( \widehat { a } ) \mid { \mathcal D } \} \le \alpha$ , which is the claimed inequality; taking expectations over $\mathcal { D }$ gives the unconditional statement. This is the conformal risk control argument of Angelopoulos et al. (2024), specialized to a finite ordered index set.

Finally, when the response-model alternative of Appendix B draws fresh Monte Carlo responses at each evaluated input, attach to each pair $( X _ { j } , Y _ { j } )$ an independent auxiliary variable $\xi _ { j }$ collecting its draws; the triples $( X _ { j } , Y _ { j } , \xi _ { j } )$ remain exchangeable, each $L _ { j } ( \cdot )$ is a fixed measurable transformation of its triple, and the argument applies verbatim. □

## F.2 Proof of Propositions 4.5 and 4.6

For Proposition 4.5, write $\lambda ^ { * } : = R ^ { - 1 } ( a \mid x ) = \operatorname* { i n f } \{ \lambda : R ( \lambda \mid x ) \leq a \}$ . Assumption 4.2 gives $R ( \lambda _ { \operatorname* { m a x } } \mid x ) = 0 \leq a .$ , so the defining set is nonempty. Take $\lambda _ { k } \downarrow \lambda ^ { * }$ with $R ( \lambda _ { k } \mid x ) \leq a ;$ right-continuity gives $R ( \lambda ^ { * } \mid x ) = \operatorname* { l i m } _ { k } R ( \lambda _ { k } \mid x ) \leq a$ . The first display then follows from (5). For the equality claim, suppose $R ( \cdot \mid x )$ is continuous and $a \leq R ( \lambda _ { \operatorname* { m i n } } \mid x )$ , and assume for contradiction that $R ( \lambda ^ { * } \mid x ) < a$ . Since $R ( \lambda _ { \operatorname* { m i n } } \mid x ) \geq a$ , we have $\lambda ^ { * } > \lambda _ { \operatorname* { m i n } }$ , and by continuity there exists $\lambda < \lambda ^ { * }$ with $R ( \lambda \mid x ) < a .$ , contradicting the definition of $\lambda ^ { * }$ as the infimum. Hence $R ( \lambda ^ { * } \mid x ) = a$

For Proposition 4.6, let $\widehat { \lambda } _ { a } ( x ) : = \widehat { R } _ { \mathcal { D } } ^ { - 1 } ( a \mid x )$ . If $a = 0$ or the defining set is empty, then $\widehat { \lambda } _ { a } ( x ) = \lambda _ { \mathrm { m a x } }$ and the desired bound is trivial by Assumption 4.2. Otherwise, the selected threshold belongs to the finite feasible set, so $\widehat { R } _ { \cal D } ( \widehat { \lambda } _ { a } ( x ) \mid x ) \leq a$ . Therefore

$$
\begin{array} { r } { \mathbb { E } \left\{ \ell \left( \widehat { f } _ { \mathcal { D } , a } ^ { R } ( X ) , Y \right) \Big | X = x , \mathcal { D } \right\} = R ( \widehat { \lambda } _ { a } ( x ) \mid x ) \leq \widehat { R } _ { \mathcal { D } } ( \widehat { \lambda } _ { a } ( x ) \mid x ) + \varepsilon _ { M } ( x ) \leq a + \varepsilon _ { M } ( x ) , } \end{array}
$$

where the equality uses that $\widehat { \lambda } _ { a } ( x )$ is D-measurable and $( X , Y )$ is independent of $\mathcal { D } \mathrm { ~ ( A s - }$ sumption 4.1), and the first inequality uses the gridwise one-sided error bound. Under the additional i.i.d. condition, the test pair is independent of C. Since the bound holds simultaneously in a, we may condition on C and substitute $a = \widehat { a }$ . This proves Proposition 4.6.

## F.3 Proof of Proposition E.1

Fix a realization of D for which the pointwise conclusion of Proposition 4.6 holds. Define

$$
\begin{array} { r } { V _ { a } ( x ) : = \left[ \mathbb { E } \left\{ \ell \left( \widehat { f } _ { \mathcal { D } , a } ^ { R } ( X ) , Y \right) \Big | X = x , \mathcal { D } \right\} - a \right] _ { + } . } \end{array}
$$

The pointwise result gives $V _ { a } ( x ) \leq \varepsilon _ { M } ( x )$ . Markov’s inequality applied to ${ \cal V } _ { a } ^ { p } ( X )$ proves (11). For a measurable group G of probability $\pi _ { G } > 0$ , Hölder’s inequality yields

$$
\mathbb { E } \{ \varepsilon _ { M } ( X ) \mid X \in G , { \mathcal { D } } \} = \frac { \mathbb { E } \{ \varepsilon _ { M } ( X ) \mathbf { 1 } ( X \in G ) \mid { \mathcal { D } } \} } { \pi _ { G } } \leq \pi _ { G } ^ { - 1 / p } \| \varepsilon _ { M } \| _ { L ^ { p } ( P _ { X } ) } .
$$

Combining this with the pointwise result and averaging over $X \mid X \in G$ proves (12). Under the additional i.i.d. condition, the test pair is independent of C. Since both bounds hold simultaneously in a, we may condition on C and substitute $a = \widehat { a }$ □

## F.4 Proof of Proposition E.2

Set $\begin{array} { r } { \Delta _ { m } ( x ) : = \operatorname* { m a x } _ { \lambda \in \Lambda _ { M } } \vert R ( \lambda \mid x ) - \widehat { R } _ { \mathcal { D } _ { m } } ( \lambda \mid x ) \vert } \end{array}$ , and let $L _ { x }$ be a Lipschitz constant for $R ( \cdot \mid x )$ . Fix $a \in [ 0 , R ( \lambda _ { \operatorname* { m i n } } \mid x ) ]$ and write $\widehat { \lambda } _ { a } ( x ) = \widehat { R } _ { { D } _ { m } } ^ { - 1 } ( a \mid x )$ . The proposition gives $R ( \widehat { \lambda } _ { a } ( x ) \mid x ) \leq a + \Delta _ { m } ( x )$

For the reverse inequality, first consider the boundary cases. If $a = 0$ , the convention selects $\lambda _ { \mathrm { m a x } }$ and the true risk is zero. If the estimated feasible set is empty, then $\hat { R } _ { D _ { m } } ( \lambda _ { \operatorname* { m a x } } \ |$ $x ) > a$ while $R ( \lambda _ { \operatorname* { m a x } } \mid x ) = 0$ , so $a < \Delta _ { m } ( x )$ . If $\widehat { \lambda } _ { a } ( x ) = \lambda _ { \operatorname* { m i n } }$ , feasibility and $a \leq R ( \lambda _ { \operatorname* { m i n } } \mid x )$ give $0 \le R ( \lambda _ { \operatorname* { m i n } } \mid x ) - a \le \Delta _ { m } ( x )$ . In every boundary case the claimed lower bound follows. Otherwise, let $\lambda ^ { - }$ be the grid point immediately below $\widehat { \lambda } _ { a } ( x )$ . Its infeasibility gives $R ( \lambda ^ { - } \mid x ) > a - \Delta _ { m } ( x )$ , while Lipschitz continuity gives $R ( \widehat { \lambda } _ { a } ( x ) \mid x ) \geq R ( \lambda ^ { - } \mid x ) - L _ { x } h _ { M }$ Thus the absolute error is at most $\Delta _ { m } ( x ) + L _ { x } h _ { M }$ , uniformly over all attainable budgets. Under the additional conditions in the final sentence of the proposition, the same bound applies at $a = \widehat { a }$ on the event $\{ \widehat { a } \leq R ( \lambda _ { \operatorname* { m i n } } \mid x ) \}$ . Since this event has probability tending to one, the claimed $O _ { \mathbb { P } } ( \eta _ { m } + h _ { M } )$ conclusion follows. □

## F.5 Proof of Theorem E.3

For compactness in this proof, write

$$
\begin{array} { r } { R _ { m } ^ { R } ( \boldsymbol { a } \mid \boldsymbol { x } ) : = \operatorname { \mathbb { E } } \left\{ \ell \left( \widehat f _ { \mathcal { D } _ { m } , \boldsymbol { a } } ^ { R } ( \boldsymbol { X } ) , \boldsymbol { Y } \right) \Big | \boldsymbol { X } = \boldsymbol { x } , \mathcal { D } _ { m } \right\} , \qquad \boldsymbol { r } _ { m } ( \boldsymbol { a } ) : = \operatorname { \mathbb { E } } \{ R _ { m } ^ { R } ( \boldsymbol { a } \mid \boldsymbol { X } ) \mid \mathcal { D } _ { m } \} , } \end{array}
$$

and let

$$
q _ { m } : = \operatorname* { s u p } _ { a \in I } | r _ { m } ( a ) - a | , \qquad q _ { m } ( x ) : = \operatorname* { s u p } _ { a \in I } | R _ { m } ^ { R } ( a \mid x ) - a | .
$$

The first accuracy assumption of the theorem gives $q _ { m } = O _ { \mathbb { P } } ( \rho _ { m } )$ . For its final claim, Proposition E.2, together with the assumptions on the original risk-curve estimator and the fact that $I \subset ( 0 , R ( \lambda _ { \operatorname* { m i n } } \mid x ) )$ , gives $q _ { m } ( x ) = O _ { \mathbb { P } } ( \eta _ { m } + h _ { M } )$

For each m, n and $a \in { \mathcal { A } } _ { n }$ , write

$$
{ \widehat { r } } _ { m , n } ( a ) : = { \frac { 1 } { n } } \sum _ { j = 1 } ^ { n } \ell \left( { \widehat { f } } _ { { D } _ { m } , a } ^ { R } ( X _ { j } ) , Y _ { j } \right) .
$$

Conditional on $\mathcal { D } _ { m }$ , this is an average of i.i.d. variables in $[ 0 , B ]$ with mean $r _ { m } ( a )$ . Moreover, the losses are pointwise nondecreasing in a, as established in the proof of Theorem 4.4. This ordering gives a uniform calibration bound without a union bound over $\mathcal { A } _ { n }$ . Indeed, for $t \in [ 0 , B ]$ , the sets

$$
\left\{ ( x , y ) : \ell \left( { \widehat { f } } _ { { D } _ { m } , a } ^ { R } ( x ) , y \right) > t \right\} , \qquad a \in A _ { n } ,
$$

are nested in a. For each fixed t, assign an observation the first grid index at which its loss exceeds t. The displayed events are then ordinary threshold events for this one-dimensional index. Conditional on $\mathcal { D } _ { m }$ , the Dvoretzky–Kiefer–Wolfowitz inequality therefore gives

$$
\begin{array} { r } { \mathbb { P } \{ D _ { m , n } ( t ) > u \mid \mathcal { D } _ { m } \} \le 2 \exp ( - 2 n u ^ { 2 } ) , } \end{array}
$$

where

$$
D _ { m , n } ( t ) : = \operatorname* { m a x } _ { a \in \mathcal { A } _ { n } } \left. \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \mathbf { 1 } \Big \{ \ell \left( \widehat { f } _ { \mathcal { D } _ { m , a } } ^ { R } ( X _ { j } ) , Y _ { j } \right) > t \Big \} - \mathbb { P } \Big \{ \ell \left( \widehat { f } _ { \mathcal { D } _ { m , a } } ^ { R } ( X ) , Y \right) > t \Big | \mathcal { D } _ { m } \Big \} \right. .
$$

It follows that $\mathbb { E } \{ D _ { m , n } ( t ) \mid \mathcal { D } _ { m } \} \le \sqrt { \pi / ( 2 n ) }$ . The layer-cake representation and Tonelli’s inequality now yield

$$
\mathbb { E } \left\{ \operatorname* { m a x } _ { a \in \mathcal { A } _ { n } } \left. \widehat { r } _ { m , n } ( a ) - r _ { m } ( a ) \right. \bigg \vert \mathcal { D } _ { m } \right\} \leq B \sqrt { \frac { \pi } { 2 n } } .
$$

The maximum changes by at most $B / n$ when one calibration observation is replaced, so McDiarmid’s inequality further gives, for every $\delta \in ( 0 , 1 )$ ,

$$
\mathbb { P } \left\{ \underset { a \in A _ { n } } { \operatorname* { m a x } } \left| \hat { r } _ { m , n } ( a ) - r _ { m } ( a ) \right| > B \left( \sqrt { \frac { \pi } { 2 n } } + \sqrt { \frac { \log ( 1 / \delta ) } { 2 n } } \right) \bigg | D _ { m } \right\} \leq \delta .
$$

Consequently,

$$
\operatorname* { m a x } _ { a \in A _ { n } } \left| \widehat { r } _ { m , n } ( a ) - r _ { m } ( a ) \right| = O _ { \mathbb { P } } ( n ^ { - 1 / 2 } ) \longrightarrow 0 \mathrm { i n ~ p r o b a b i l i t y } .\tag{13}
$$

The bound is uniform over every finite $\mathcal { A } _ { n }$ and hence requires no condition on the number of grid points.

The CRC constraint in (7) is equivalently

$$
\widehat { r } _ { m , n } ( a ) \leq \alpha - c _ { n } , \qquad c _ { n } : = \frac { B - \alpha } { n } \longrightarrow 0 .
$$

Let

$$
E _ { m , n } : = q _ { m } + \operatorname* { m a x } _ { a \in \mathcal { A } _ { n } } \left| \widehat { r } _ { m , n } ( a ) - r _ { m } ( a ) \right| + c _ { n } .
$$

By assumption and (13), $E _ { m , n } ~ = ~ O _ { \mathbb { P } } ( \rho _ { m } + n ^ { - 1 / 2 } )$ . With probability tending to one, $\left[ \alpha - E _ { m , n } - h _ { n } , \alpha + E _ { m , n } + h _ { n } \right] \subset I$ . On this event, let $b _ { n } ^ { - }$ be the largest grid point no greater than $\alpha - E _ { m , n }$ and $b _ { n } ^ { + }$ the smallest grid point no less than $\alpha + E _ { m , n }$ . Then

$$
\widehat { r } _ { m , n } ( b _ { n } ^ { - } ) \leq b _ { n } ^ { - } + q _ { m } + \operatorname* { m a x } _ { a \in A _ { n } } \vert \widehat { r } _ { m , n } ( a ) - r _ { m } ( a ) \vert \leq \alpha - c _ { n } ,
$$

so $b _ { n } ^ { - }$ is feasible. Similarly, $\widehat { r } _ { m , n } ( b _ { n } ^ { + } ) \geq \alpha + c _ { n } > \alpha - c _ { n }$ , so $b _ { n } ^ { + }$ is infeasible. Monotonicity makes every grid point above $b _ { n } ^ { + }$ infeasible, and therefore

$$
\alpha - E _ { m , n } - h _ { n } \leq b _ { n } ^ { - } \leq \widehat { a } < b _ { n } ^ { + } \leq \alpha + E _ { m , n } + h _ { n } .
$$

Consequently,

$$
| \widehat { a } - \alpha | \leq E _ { m , n } + h _ { n } = O _ { \mathbb { P } } \bigl ( \rho _ { m } + n ^ { - 1 / 2 } + h _ { n } \bigr ) .
$$

Finally, $\widehat { a } \in I$ with probability tending to one, and on that event

$$
| R _ { m } ^ { R } ( \hat { a } \mid x ) - \alpha | \leq q _ { m } ( x ) + | \hat { a } - \alpha | = O _ { \mathbb { P } } ( \rho _ { m } + \eta _ { m } + h _ { M } + n ^ { - 1 / 2 } + h _ { n } ) .
$$

Because the test pair is independent of ${ \mathcal { C } } ,$ the left-hand side is the conditional-risk error displayed in the theorem. This proves both claims. □

## F.6 Proof of Corollary 4.7

Let

$$
r _ { * } : = \exp _ { x \sim P _ { X } } R ( \lambda _ { \operatorname* { m i n } } \mid x ) .
$$

Because $\alpha < r _ { * }$ , choose an open interval I containing α whose closure lies in $( 0 , r _ { * } )$ . The deterministic argument in the proof of Proposition E.2, now using the common Lipschitz constant and taking the essential supremum over x, gives

$$
\operatorname* { s u p } _ { a \in I } \mathrm { e s s } \operatorname* { s u p } _ { x \sim P _ { X } } \left| \mathbb { E } \left\{ \ell \left( \widehat { f } _ { \mathcal { D } _ { m } , a } ^ { R } ( X ) , Y \right) \Big | X = x , \mathcal { D } _ { m } \right\} - a \right| = O _ { \mathbb { P } } ( \eta _ { m } + h _ { M } ) .
$$

Integrating over X verifies the marginal risk-scale condition of Theorem E.3 with $\rho _ { m } =$ $\eta _ { m } + h _ { M }$ . Its first conclusion therefore gives

$$
| \widehat { a } - \alpha | = O _ { \mathbb { P } } ( \eta _ { m } + h _ { M } + n ^ { - 1 / 2 } + h _ { n } ) ,
$$

and also places ab in I with probability tending to one. On that event, the preceding uniform bound and the triangle inequality give the same rate uniformly over x for the deployed conditional-risk error. Independence of the test pair from C identifies that error with the conditional expectation displayed in the corollary. □

## F.7 Proof of Corollary 4.8

Proposition 4.6 gives

$$
\mathbb { E } _ { P } \{ \ell ( \widehat { f } ( X ) , Y ) \mid X = x , \mathcal { D } , \mathcal { C } \} \leq \widehat { a } + \varepsilon _ { M } ( x )
$$

for $P _ { X }$ -almost every x. Because $Y \mid X$ is unchanged, the same pointwise bound holds under $Q ;$ because $Q _ { X } \ll P _ { X }$ , integrating it over $Q _ { X }$ proves (10). □

## F.8 Proof of Proposition E.4

(a) Since $\lambda \mapsto R ( \lambda \mid x )$ is non-increasing and right-continuous and $R ( \lambda _ { \operatorname* { m a x } } \mid x ) = 0$ the function $F _ { x } ( \lambda ) \ : = \ : 1 \ : - \ : R ( \lambda \mid x )$ is non-decreasing and right-continuous on Λ, with $F _ { x } ( \lambda _ { \operatorname* { m a x } } ) = 1$ . Extending it to the real line by setting $F _ { x } ( \lambda ) = 0 \mathrm { f o r } \lambda < \lambda _ { \operatorname* { m i n } }$ and $F _ { x } ( \lambda ) = 1$ for $\lambda > \lambda _ { \mathrm { m a x } }$ makes it a valid cumulative distribution function. If $R ( \lambda _ { \operatorname* { m i n } } \mid x ) < 1$ , the extension simply places an atom of size $1 - R ( \lambda _ { \operatorname* { m i n } } \mid x )$ at $\lambda _ { \operatorname* { m i n } }$ . Thus there exists $Z _ { x } \sim F _ { x }$ for which $\mathbb { P } ( Z _ { x } > \lambda ) = 1 - F _ { x } ( \lambda ) = R ( \lambda \mid x )$ on Λ.

(b) Directly from the definitions,

$$
\begin{array} { r l } & { R ^ { - 1 } ( a \mid x ) = \operatorname* { i n f } \{ \lambda \in \Lambda : R ( \lambda \mid x ) \leq a \} = \operatorname* { i n f } \{ \lambda \in \Lambda : 1 - R ( \lambda \mid x ) \geq 1 - a \} } \\ & { \quad \quad = \operatorname* { i n f } \{ \lambda \in \Lambda : F _ { x } ( \lambda ) \geq 1 - a \} = F _ { x } ^ { - 1 } ( 1 - a ) , } \end{array}
$$

where $F _ { x } ^ { - 1 }$ denotes the generalized quantile over Λ, including the endpoint convention $F _ { x } ^ { - 1 } ( 0 ) = \lambda _ { \operatorname* { m i n } }$

(c) The first identity follows from (b) and the definition (4). For the canonical problem, if $Z _ { x } \sim F _ { x }$ then

$$
R _ { \mathrm { c a n } } ( \lambda \mid x ) = \mathbb { E } \{ { \bf 1 } ( Z _ { x } > \lambda ) \} = \mathbb { P } ( Z _ { x } > \lambda ) = R ( \lambda \mid x ) ,
$$

so the canonical problem has exactly the same local risk curve as the original problem, and therefore the same generalized inverse and the same rectification map. □

## G Worked examples: rectified predictions across CRCstyle losses

Rectification replaces the global threshold λ by the local threshold $\lambda _ { a } ^ { R } ( x ) = R ^ { - 1 } ( a \mid x )$ , so every rectified prediction $f _ { a } ^ { R } ( x ) = f _ { \lambda _ { a } ^ { R } ( x ) } ( x )$ is a member of the original nested family. The family determines what more protection looks like—a larger mask or label set, a deeper list, a coarser node of a hierarchy, a wider interval, or a smaller mask—and rectification determines how much of it each input receives. The examples below, built from losses used in the CRC literature, also expose the conditions this relies on: a scalar ordering of the family, a fully protective endpoint, attainable budgets, and monotone realized losses. Unless stated otherwise, the base model assigns scores $s _ { x } ( u ) \in ( 0 , 1 )$ to the items of a finite universe U (pixels, labels, documents, or candidate answers), and $f _ { \lambda } ( x ) = \{ u \in \mathcal { U } : s _ { x } ( u ) \geq 1 - \lambda \}$ $\lambda \in [ 0 , 1 ]$ , so the least and most protective rules are $f _ { 0 } ( x ) = \varnothing$ and $f _ { 1 } ( x ) = \mathcal { U }$

Missed positives: masks, label sets, and retrieval sets. The false-negative proportion $\ell _ { \mathrm { F N } }$ of Section 2.1, with $Y \subseteq { \mathcal { U } }$ the true positives, underlies the tumor-segmentation and multilabel examples of Angelopoulos et al. (2024) and the retrieval stage of Xu et al. (2025).

The rectified rule $f _ { a } ^ { R } ( x ) = \{ u : s _ { x } ( u ) \geq 1 - \lambda _ { a } ^ { R } ( x ) \}$ is still a score superlevel set, but its cutof is input-specific. Easy images receive a high cutof and a tight mask, and hard images a lower cutof and a larger mask whose boundary follows the level sets of the segmentation score; ambiguous inputs receive larger label sets; and queries with difuse relevance are retrieved more deeply than queries whose relevant documents are clearly ranked on top.

Modified nDCG. For the ranking stage of Xu et al. (2025, Section 4.2), let the response $Z = ( d _ { ( 1 ) } , \ldots , d _ { ( m ) } )$ list the relevant documents in decreasing order of relevance, let $f _ { \lambda } ( x )$ be the retained documents, and define

$$
\ell _ { \mathrm { n D C G } } ( f _ { \lambda } ( x ) , Z ) = 1 - \frac { \sum _ { j = 1 } ^ { m } \mathbf { 1 } \{ d _ { ( j ) } \in f _ { \lambda } ( x ) \} / \log ( j + 1 ) } { \sum _ { j = 1 } ^ { m } 1 / \log ( j + 1 ) } ,
$$

with $\ell _ { \mathrm { n D C G } } = 0$ when $m = 0$ . The loss is graded: missing the most relevant document costs more than missing a marginal one, so the local depth is short when the ranker confidently places the top documents first and long when it does not. In the full two-stage system both thresholds may be rectified, but CRC still needs a single scalar path through the threshold pairs.

Hierarchical graph distance. Angelopoulos et al. (2024) score a single predicted node of a class tree $\tau$ by its graph distance $d _ { T }$ to the set anc $( y )$ of ancestors of the true leaf $y .$ including y. For a nested set of selected nodes $f _ { \lambda } ( x ) \subseteq \tau$ , a natural extension is

$$
\ell _ { \mathrm { t r e e } } ( f _ { \lambda } ( x ) , y ) = \frac { 1 } { D } \operatorname* { m i n } _ { v \in f _ { \lambda } ( x ) } \operatorname* { m i n } _ { u \in \mathrm { a n c } ( y ) } d _ { \mathcal { T } } ( v , u ) ,
$$

where $D \ { \mathrm { ( e . g . } }$ , the diameter of $\tau _ { \ / { 2 } }$ normalizes the loss to $[ 0 , 1 ]$ and the minimum over an empty set is D. The loss vanishes once a selected node lies on the ancestral path and cannot increase as the set grows. Rectification now controls granularity: easy inputs return a small leaf-level neighborhood, hard inputs larger subtrees or coarser internal nodes.

Token-level F1 for question answering. Let U be the candidate answers generated for question x and Y a set of acceptable reference answers. Writing the best-answer loss of Angelopoulos et al. (2024) for a set of references,

$$
\ell _ { \mathrm { F } 1 } ( f _ { \lambda } ( x ) , Y ) = 1 - \operatorname* { m a x } _ { z \in f _ { \lambda } ( x ) } \operatorname* { m a x } _ { y \in Y } \mathrm { F } 1 _ { \mathrm { t o k e n } } ( z , y ) ,
$$

with the maximum over an empty set equal to zero, rectification makes the number of returned candidates question-specific. This loss also shows that the fully protective endpoint of Assumption 4.2 can fail: $\ell _ { \mathrm { F 1 } } ( f _ { \lambda _ { \mathrm { m a x } } } ( x ) , Y ) = 0$ only if some candidate attains token F1 of one against a reference. One may instead rectify the excess loss $\ell _ { \mathrm { { F 1 } } } ( f _ { \lambda } ( x ) , Y ) \textrm { - }$ $\ell _ { \mathrm { F 1 } } ( f _ { \lambda _ { \mathrm { m a x } } } ( x ) , Y )$ , which has a zero endpoint but changes the controlled quantity: calibrating it at level α bounds the original risk only by $\alpha + \mathbb { E } \{ \ell _ { \mathrm { F 1 } } ( f _ { \lambda _ { \mathrm { m a x } } } ( X ) , Y ) \}$ . Control of the original loss at level α thus requires a bound $\beta$ on this residual risk and calibration at level $\alpha - \beta$

Ordinal losses. For ordinal labels $\mathcal { Y } = \{ 0 , \ldots , K - 1 \}$ and interval outputs $f _ { \lambda } ( x ) =$ $[ l _ { \lambda } ( x ) , u _ { \lambda } ( x ) ] \cap \mathcal { Y }$ , Xu et al. (2023) study the weighted exclusion loss $h ( y ) \mathbf { 1 } \{ y \notin f _ { \lambda } ( x ) \}$ , with weights $h ( y ) \in [ 0 , B ]$ , and the divergence loss $( K - 1 ) ^ { - 1 } \operatorname* { m i n } _ { k \in f _ { \lambda } ( x ) } | y - k |$ ; both are non-increasing as the interval expands. Rectification makes the interval width local, but it only selects a point on the pre-specified interval path and cannot change the path’s geometry: a large $h ( y )$ makes omitting class y costly, yet the interval extends toward y faster only if the family already does so. The divergence loss also illustrates saturation: a singleton next to the true label already has small loss, so $R ( \lambda _ { \operatorname* { m i n } } \mid x )$ can fall well below $B = 1$ , and larger budgets leave the rule at $f _ { \lambda _ { \mathrm { m i n } } }$ , which is then conservative (cf. Proposition 4.5).

False positives and shrinking masks. Mossina & Friedrich (2025) control false positives by shrinking segmentation masks, normalizing by the size of the initial mask. For illustration we instead take the shrinking family $f _ { \lambda } ( x ) = \{ u : s _ { x } ( u ) > \lambda \}$ , which runs from $f _ { 0 } ( x ) = \mathcal { U }$ to $f _ { 1 } ( x ) = \varnothing$ , and the false-positive fraction $\ell _ { \mathrm { F P } } ( f _ { \lambda } ( x ) , Y ) = | f _ { \lambda } ( x ) \setminus Y | / ( | \mathcal { U } \setminus Y | \lor 1 )$ , which is non-increasing in λ. Protection now means removing pixels, and rectification acts as local erosion: images with many likely false positives receive a higher threshold and a smaller mask. The false discovery proportion $| f _ { \lambda } ( x ) \setminus Y | / ( | f _ { \lambda } ( x ) | \vee 1 )$ is often more meaningful but generally not monotone along the family. Since CRC needs monotone realized losses, monotonizing the estimated risk curve does not fix this; one needs a family on which the loss is pointwise monotone, a monotone upper envelope of the loss, or a calibration method for non-monotone losses, such as Learn-then-Test (Angelopoulos et al. 2025) or stability-based conformal risk control (Angelopoulos 2026).