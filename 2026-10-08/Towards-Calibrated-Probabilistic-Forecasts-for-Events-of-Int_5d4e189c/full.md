# Towards Calibrated Probabilistic Forecasts for Events of Interest via Outcome-Conditional Recalibration

Jakob Wessel Institute for Meteorology University of Leipzig

Sam Allen Institute of Statistics Karlsruhe Institute of Technology

## Abstract

Calibration is an essential requirement for probabilistic predictions to be useful for decision making. While state-of-the-art predic tion methods often yield miscalibrated pre dictive distributions, several post-hoc recalibration schemes have been proposed to gener ate calibrated predictions. However, popular recalibration schemes can conceal miscalibra tion in specific regions of the outcome space. Since particular outcomes, such as extreme events, often matter most for decision mak ing, probabilistic predictions should be cali brated when evaluation is restricted to these outcomes. Hence, in this paper, we introduce outcome-conditional recalibration, a post-hoc method to recalibrate probabilistic predictions on user-defined regions of the outcome space. The method is simple, easy to implement, and can be applied to arbitrary predic tive distributions. It works by applying the quantile recalibration approach of Kuleshov et al. (2018) to forecast conditional distribu tions, before rescaling these conditional dis tributions so that forecast event probabilities match empirical occurrence frequencies. This produces valid and continuous predictive dis tributions that are calibrated within each re gion of interest. Across regression benchmarks, we demonstrate that existing recali bration schemes do not necessarily yield cal ibrated predictions when interest is on particular outcomes, and that our approach improves outcome-conditional calibration relative to existing conditional and unconditional recalibration methods, while retaining competitive calibration overall. In an application to day-ahead electricity price forecasting, the approach substantially improves calibration when predicting negative prices, at negligible cost to forecast accuracy.

## 1 INTRODUCTION

Probabilistic predictions are said to be calibrated if they accurately quantify the uncertainty in the unknown outcome variable, in the sense that predicted probabilities align statistically with corresponding outcomes. Calibration is considered a fundamental property that forecasts should satisfy if they are to be useful for decision making (Gneiting et al., 2007). For example, if an event is predicted to occur with a certain probability, then the prediction is calibrated if the event does indeed occur with this probability. Here, we focus on probabilistic predictions in a regression context, where predictions take the form of a cumulative distribution function. While many forms of calibration exist in this case (Gneiting and Resin, 2023; Resin et al., 2026), the most popular notion of calibration corresponds to all prediction intervals derived from the predictive distribution having the correct unconditional coverage (Dawid, 1984; Diebold et al., 1998).

Guo et al. (2017) demonstrate that state-of-the-art prediction methods often do not issue calibrated predictions. Many approaches have thus been proposed to recalibrate predictions for real-valued outcomes. Vovk et al. (2017) introduced conformal predictive systems, which yield sets of predictive distributions that are guaranteed to contain a calibrated prediction out-ofsample (see also Chernozhukov et al., 2021). Several other studies have proposed weakly enforcing calibration during model training by quantifying miscalibration and using this as a regularisation term within the loss function (Wilks, 2018; Utpala and Rai, 2020; Laajil et al., 2026). Alternatively, post-hoc recalibration schemes learn the miscalibration in predictive distribu tions, and then transform or post-process the predictions so that this miscalibration is removed (Kuleshov et al., 2018; Dheur and Taieb, 2024).

However, these recalibration methods typically yield predictive distributions that only satisfy a weak unconditional notion of calibration. It is well-known that unconditional calibration is generally insuficient for efective decision making, since it only requires that predicted probabilities align with true event frequencies on average, and does not ensure that individual predictions are reliable (Christofersen, 1998; Romano et al., 2019; Song et al., 2019). An unconditionally calibrated predictive distribution may also not be calibrated when attention is restricted to a particular subset of the outcome space (Allen et al., 2025b). Standard recalibration methods may therefore yield forecasts that are miscalibrated when attention is restricted to certain outcomes. However, certain outcomes, such as extreme events, are generally more relevant for decision making, making calibrated predictions for these outcomes particularly valuable.

We therefore introduce a post-hoc outcome-conditional recalibration scheme that applies the quantile recali bration method of Kuleshov et al. (2018) to the conditional forecast distribution given that the outcome falls in a particular subset of the outcome space. $\mathrm { B y }$ applying this repeatedly to disjoint subsets that partition the real line, and rescaling so that forecast event probabilities match empirical occurrence frequencies, the resulting recalibrated predictive distributions are valid and continuous distribution functions that satisfy outcome-conditional calibration by construction. We apply the approach to benchmark regression datasets, and find that the standard recalibration scheme of Kuleshov et al. (2018) does not necessarily issue calibrated predictions for extreme events; nor do variants designed to satisfy stronger conditional notions of calibration (Kuleshov and Deshpande, 2022). In contrast, outcome-conditional recalibration yields wellcalibrated predictions for extreme events, whilst also exhibiting competitive calibration overall. In an application to energy price forecasting, outcome-conditional calibration is capable of issuing reliable predictions of negative electricity prices, whereas alternative recali bration methods are not.

## 2 PRELIMINARIES

## 2.1 Notation

Let $Y \in \mathbb { R }$ be a scalar outcome. We denote a probabilistic prediction for Y by a cumulative distribution function $F : \mathbb { R }  [ 0 , 1 ]$ We treat $F$ as a random variable; the randomness in $F$ often comes from dependence on random covariates, which we omit from the notation for simplicity. The joint distribution of $( F , Y )$ is denoted by P. We assume that this joint distribution is continuous, though this can be relaxed via suitable randomisation (Gneiting and Ranjan, 2013). We refer to $F ( Y )$ as the forecast probability integral transform (PIT). The conditional distribution of $F$ on an interval $\mathcal { I } = [ a , b ) \subset \mathbb { R }$ is defined as

$$
F ^ { \mathbb { Z } } ( x ) : = { \frac { F ( x ) - F ( a ) } { F ( b ) - F ( a ) } } , \quad { \mathrm { f o r ~ } } x \in [ a , b ) ,
$$

with $F ^ { \mathcal { I } } ( x ) = 0$ for $x < a ,$ , and $F ^ { \mathcal { T } } ( x ) = 1$ for $x \geq b .$ The conditional PIT (CPIT) on I is defined as $F ^ { \mathbb { Z } } ( Y )$

In practice, we observe a sample of forecast– observation pairs $\{ ( F _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , which can be interpreted as realisations of the random pair $( F , Y )$ . The pairs $\{ ( F _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ can either correspond to the same data used to train the forecasts, or (more commonly) to an out-of-sample calibration dataset. Conditional forecast distributions on I are denoted by $\{ F _ { i } ^ { \underline { { \tau } } } \} _ { i = 1 } ^ { N }$

## 2.2 Related Work

The basis of our outcome-conditional recalibration scheme is the widely used quantile recalibration method of Kuleshov et al. (2018). Several studies have proposed extensions or adaptations of this PIT-based recalibration. Dheur and Taieb (2024) suggest to di rectly learn the optimal quantile recalibrated predic tive distribution, rather than applying the recalibration post-hoc; that is, the recalibration is included as a part of the model, and model parameters are then learned subject to this calibration constraint. Wilks (2018) instead adapt the loss function used to train probabilistic models by adding a regularisation term that quantifies the divergence between the PIT val ues $\{ \bar { F } _ { i } ( y _ { i } ) \} _ { i = 1 } ^ { N }$ and a standard uniform distribution, thus penalising models that produce miscalibrated predictive distributions during training (see also Utpala and Rai, 2020). Laajil et al. (2026) employ a similar approach to yield predictive distributions for multivariate outcomes that satisfy a multivariate notion of calibration (Gneiting et al., 2008; Allen et al., 2024).

Other studies have also proposed extensions of quantile recalibration to multivariate predictive distributions. Chung et al. (2024) use the estimated predictive density function to transform multivariate predictive distributions to univariate predictive distributions, before applying quantile recalibration. This ensures that high-density regions of the multivariate predictive distribution are calibrated. Dheur and Ben Taieb (2026) similarly propose recalibrating latent variables within generative models using quantile recalibration. Feeding these recalibrated latent variables into the estimated model then yields calibrated multivariate predictions. Kock et al. (2026) instead directly estimate the conditional distribution of the PIT values in all dimensions. Quantile recalibration is then applied along each margin, and sampling from these recalibrated dis tributions at observed PIT vectors yields a sample from a calibrated multivariate predictive distribution.

This latter approach is performed conditionally on covariates, and therefore approximates a stronger notion of calibration. Song et al. (2019) and Kuleshov and Deshpande (2022), among others, also propose conditional recalibration schemes whereby the recalibration is performed conditionally on the original predictive distributions, thus making the predictions calibrated conditionally on the forecasts themselves; this is of ten referred to as auto-calibration (Tsyplakov, 2011; Gneiting and Ranjan, 2013) or distribution calibration (Song et al., 2019). However, these approaches typically require making fairly strong modelling assumptions, and therefore do not satisfy any finite-sample calibration guarantees. In Section 5, we use the approach of Kuleshov and Deshpande (2022) as a baseline method against which outcome-conditional recalibration is compared.

However, none of these studies have considered calibration conditional on the observed outcomes. Wessel et al. (2026) alternatively quantify tail miscalibration, and use this to regularise loss functions during model training. This is found to yield more reliable forecasts for extreme outcomes. However, this only weakly enforces tail calibration. The recalibration scheme proposed in this paper instead provides a means to enforce outcome-conditional calibration of predictive distributions post-hoc, leading to predictive distributions that satisfy theoretical in-sample calibration guarantees.

## 3 CALIBRATION

The calibration of a probabilistic prediction is typically assessed by checking whether the forecast probability integral transform (PIT), F(Y), follows a standard uniform distribution (Dawid, 1984; Diebold et al., 1998). This implies that prediction intervals derived from the predictive distribution have the correct unconditional coverage at all nominal levels.

Definition 3.1 (Probabilistic calibration). The prediction F is probabilistically calibrated for Y if $F ( Y ) \sim$ Unif(0, 1). That is, $\mathbb { P } ( F ( Y ) \leq u ) = u$ for all $u \in [ 0 , 1 ]$

Probabilistic calibration is also often referred to as quantile calibration (Song et al., 2019), though we adopt the terminology introduced by Gneiting et al. (2007). Allen et al. (2025b) introduced a generalised notion of probabilistic calibration that focuses on tails of predictive distributions. This definition of tail calibration can be adapted to assess calibration when predicting outcomes in any region of the outcome space. We refer to this as outcome-conditional calibration.

Definition 3.2 (Outcome-conditional calibration). The prediction F is outcome-conditionally calibrated on $\mathcal { T } = [ a , b ) \mathrm { ~ i f ~ } \mathbb { P } ( F ^ { \mathcal { Z } } ( Y ) \leq u \mathrm { ~ | ~ } Y \in \mathcal { T } ) = u$ for all $u \in [ 0 , 1 ]$ , and $\mathbb { P } ( Y \in \mathcal { T } ) = \mathbb { E } [ F ( b ) - F ( a ) ]$

The first condition in Definition 3.2 requires that $F ^ { \mathbb { Z } } ( Y ) \sim { \mathrm { U n i f } } [ 0 , 1 ]$ when $Y \in \mathcal { T }$ . That is, the conditional predictive distribution $F ^ { \mathcal { I } }$ is a probabilistically calibrated prediction for outcomes in $\mathcal { T } .$ We refer to this condition as CPIT calibration. The second condition requires that the unconditional probability that $Y \in \mathcal { T }$ is equal to the average predicted probability that $Y \in \mathcal { Z }$ . This assesses whether predictions are calibrated when forecasting the occurrence of outcomes in I. We refer to this condition as occurrence calibration.

While outcome-conditional calibration requires that the conditional predictive distribution is calibrated when forecasting $Y \in \mathcal { Z }$ , probabilistic calibration generally does not imply outcome-conditional calibration (Allen et al., 2025b). Hence, standard recalibration schemes that enforce probabilistic calibration may not yield outcome-conditionally calibrated predictive distributions. In Section 5, we demonstrate empirically that they often do not. This motivates the introduction of outcome-conditional recalibration schemes.

Remark 3.1 (Conditional calibration). While proba bilistic calibration is most commonly assessed in practice, it corresponds to a fairly weak notion of calibration that only holds on average; it does not assess whether uncertainty is accurately quantified in any single prediction. Instead, several stronger definitions of calibration have been proposed. For example, auto-calibration (or distribution calibration) is a strong conditional notion of calibration that extends the typical notion of conditional calibration for probabilistic classifiers. If a prediction is auto-calibrated, then it is also outcome-conditionally calibrated (Allen et al., 2025b). Thus, one could argue that outcomeconditional recalibration is obsolete if we can achieve auto-calibration. However, while several recalibration schemes have been proposed to achieve autocalibration (Song et al., 2019; Kuleshov and Deshpande, 2022; Kock et al., 2026), these all require making somewhat restrictive modelling assumptions. In Section 5, we demonstrate that outcome-conditional recalibration can improve both outcome-conditional and global probabilistic calibration relative to these methods.

## 4 FROM GLOBAL TO LOCAL RECALIBRATION

## 4.1 Quantile Recalibration

In a regression context, the goal of post-hoc recalibration is to learn a function $T : [ 0 , 1 ] \to [ 0 , 1 ]$ such that

$T \circ F$ is a calibrated predictive distribution. Kuleshov et al. (2018) consider the strictly increasing function $T ( u ) \doteq \mathbb { P } ( \boldsymbol { F } ( \boldsymbol { Y } ) \le u )$ , with inverse $T ^ { - 1 }$ . If $T$ is the identity function, the prediction $F$ is probabilistically calibrated for $Y ;$ if not, the prediction exhibits systematic miscalibration. For any $T$ and $F ,$ we have

$$
\mathbb { P } ( { T } ( F ( Y ) ) \leq u ) = \mathbb { P } ( F ( Y ) \leq T ^ { - 1 } ( u ) ) = u ,
$$

for all $u \in [ 0 , 1 ]$ . That is, $T \circ F$ is a probabilistically calibrated forecast distribution for Y by construction.

In practice, $T$ must be estimated from the finite sample $\{ ( F _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ . Applying the estimated function $\hat { T }$ to the forecasts yields a sequence of predictive distributions $\{ \hat { T } \circ F _ { i } \} _ { i = 1 } ^ { N }$ that are generally calibrated insample, and which should exhibit good calibration outof-sample, assuming these forecasts-observation pairs are also realisations of $( F , Y )$ Following Kuleshov et al. (2018), we estimate T using isotonic regression, though other choices also exist, such as kernel density estimation (Dheur and Taieb, 2023), and the empirical distribution of the PIT values $\{ F _ { i } ( y _ { i } ) \} _ { i = 1 } ^ { N }$

## 4.2 Outcome-conditional Recalibration

An analogous approach can be used to recalibrate the predictions within a subset I of interest. Define $T _ { \mathcal { T } } ( u ) = \mathbb { P } ( F ^ { \mathcal { T } } ( Y ) \le u \ | \ Y \in \mathcal { T } )$ , for $u \in [ 0 , 1 ]$ . The predictive distribution is CPIT calibrated on I if $T _ { \mathcal { I } }$ is the identity function. In any case, we have that

$$
\begin{array} { r l } & { \mathbb { P } \big ( T _ { \mathcal { T } } ( F ^ { \mathcal { Z } } ( Y ) ) \leq u \mid Y \in \mathcal { T } \big ) = } \\ & { \quad \mathbb { P } \big ( F ^ { \mathcal { T } } ( Y ) \leq { T _ { \mathcal { T } } ^ { - 1 } ( u ) } \mid Y \in \mathcal { T } \big ) = T _ { \mathcal { T } } ( { T _ { \mathcal { T } } ^ { - 1 } ( u ) } ) = u , } \end{array}
$$

for all $u \in [ 0 , 1 ]$ . That is, $T _ { \mathcal { I } ^ { \circ } } F ^ { \mathcal { I } }$ is always a probabilistically calibrated predictive distribution for $Y \in \mathcal { Z }$

The predicted probability that $Y \in \mathcal { T }$ can also easily be recalibrated by rescaling $F ( b ) - F ( a )$ by a constant so that it satisfies the second requirement of outcomeconditional calibration in Definition 3.2 (whilst appropriately handling the probability mass on the complement of $\boldsymbol { \mathcal { T } } )$ . This two-step recalibration approach can therefore be used to obtain outcome-conditionally calibrated predictions on particular regions of interest.

While this only recalibrates the restriction of $F$ to $\mathcal { T } ,$ this approach can be applied to several intervals that partition the real line, yielding a valid predictive distribution that is locally calibrated on all chosen intervals. Let $a _ { 0 } = - \infty < a _ { 1 } < \cdots < a _ { G - 1 } < a _ { G } = + \infty .$ and consider the contiguous intervals ${ \cal T } _ { 1 } ~ = ~ ( a _ { 0 } , a _ { 1 } )$ , ${ \mathcal T } _ { g } \ = \ [ a _ { g - 1 } , a _ { g } )$ , for $g = 2 , \ldots , G$ . These G intervals partition the outcome space R. The recalibration scheme of Kuleshov et al. (2018) can then be applied separately to each interval.

Given the predictive distribution $F ,$ define the transformation $T _ { \mathcal { Z } _ { q } } ( u ) \ = \ \mathbb { P } ( F ^ { \mathcal { Z } _ { g } } ( Y ) \ \le \ u \ | \ Y \ \in \ \mathcal { T } _ { g } )$ , for $u \in [ 0 , 1 ]$ and $\mathbf { \bar { \rho } } _ { g } = 1 , \dots , G ,$ and let $c _ { 1 } , \ldots , c _ { G } \in ( 0 , \infty )$ be rescaling constants such that

$$
\mathbb { E } \left[ \frac { c _ { j } [ F ( a _ { j } ) - F ( a _ { j - 1 } ) ] } { \sum _ { g = 1 } ^ { G } c _ { g } [ F ( a _ { g } ) - F ( a _ { g - 1 } ) ] } \right] = \mathbb { P } ( Y \in \mathbb { Z } _ { j } ) ,
$$

for $j = 1 , \dots , G$ . Kull and Flach (2015, Theorem 1) prove that such constants exist. Define $m _ { 0 } = 0$ and $m _ { j } = c _ { j } [ F ( a _ { j } ) - F ( a _ { j - 1 } ) ] / \sum _ { q = 1 } ^ { G } c _ { g } [ F ( a _ { g } ) - F ( a _ { g - 1 } ) ]$ for $j = 1 , \dots , G$ . Then, a recalibrated version of $F$ can be defined as

$$
\widetilde { F } ( x ) = \sum _ { j = 0 } ^ { g - 1 } m _ { j } + m _ { g } \left[ T _ { \mathcal { T } _ { g } } \circ F ^ { \mathcal { T } _ { g } } ( x ) \right] \quad \mathrm { f o r } x \in \mathcal { T } _ { g } .\tag{1}
$$

We term this approach outcome-conditional recalibration. Essentially, the quantile recalibration of Kuleshov et al. (2018) is applied separately to each conditional forecast distribution $F ^ { \mathcal { T } _ { 1 } } , \dotsc , F ^ { \mathcal { \tilde { T } } _ { G } }$ . These recalibrated conditional distributions are then rescaled using the weights $m _ { 1 } , \ldots , m _ { G }$ These weights always sum to one, and $\tilde { F }$ is therefore a piecewise combination of the recalibrated distributions, with knots at the interval bounds. This outcome-conditionally recalibrated predictive distribution is therefore a valid and continuous distribution function that is outcomeconditionally calibrated on the intervals $\mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { G }$ by construction. This is formalised in the following propositions, which are proved in Appendix A.

Proposition 4.1 (Validity). The recalibrated predictive distribution defined at (1) is a valid and continuous distribution function on R. In particular, $\widetilde F ( - \infty ) = 0 , \widetilde F ( + \infty ) = 1$ , and $\widetilde { F }$ is non-decreasing.

Proposition 4.2 (Calibration). The recalibrated predictive distribution defined at (1) is outcomeconditionally calibrated on the intervals $\mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { G }$

In practice, the functions $T _ { \mathcal { L } _ { g } }$ and rescaling constants $c _ { g }$ are estimated from the calibration data $\{ ( F _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ . Similarly to the global recalibration of Kuleshov et al. (2018), the functions $T _ { \mathcal { L } _ { g } }$ are estimated using isotonic regression. However, rather than being estimated using all PIT values, $T _ { \mathcal { I } _ { g } }$ is estimated using the conditional PIT values corresponding to outcomes in ${ \mathcal { T } } ^ { g }$ , that is $\{ F _ { i } ^ { \textstyle { \mathscr { L } } _ { g } } ( y _ { i } ) \}$ for $\{ i = 1 , \ldots , N : y _ { i } \in$ $\mathcal { T } ^ { g } \}$ . This is done separately for each $g = 1 , \ldots , G$ The rescaling parameters $c _ { g }$ are estimated using iterative proportional fitting (Sinkhorn’s algorithm). The resulting recalibrated predictive distributions remain valid, continuous, and locally calibrated. This sample implementation is discussed in detail in Appendix B.

Remark 4.1 (Choice of intervals). Propositions 4.1 and 4.2 do not make any assumptions on the intervals $\mathcal { T } _ { 1 } , \ldots , \mathcal { T } ^ { G }$ , and outcome-conditional recalibration can therefore be applied using any partition of the outcome space. The approach can also readily be extended to use disconnected subsets of R, rather than intervals. In some applications, the choice of intervals to consider can be determined from the forecasting problem at hand. For example, in the following section, we consider an application to electricity price forecasts, where we are particularly interested in predicting negative electricity prices. In other applications, it may be reasonable to use intervals that correspond to quantiles of the outcome distribution. We find that using more intervals generally yields better overall calibration, subject to suficient data being available on which to accurately learn the recalibration functions and parameters for each interval.

Remark 4.2 (Global-Local recalibration). As mentioned in Section 3, probabilistic calibration generally does not imply outcome-conditional calibration, and hence the quantile recalibration approach of Kuleshov et al. (2018) does not necessary yield outcomeconditionally calibrated predictive distributions. Similarly, there is no guarantee that outcome-conditional recalibration yields predictive distributions that are probabilistically calibrated overall; this holds regardless of the choice of intervals used within the recali bration. For comparison, we additionally consider an approach that combines the two recalibration schemes by first implementing outcome-conditional recalibration, before applying quantile recalibration to these locally calibrated predictive distributions. While this does not necessarily enforce outcome-conditional calibration, the prior application of outcome-conditional recalibration may weakly enforce this in practice.

## 5 EXPERIMENTS

We demonstrate the utility of outcome-conditional recalibration on standardised machine learning benchmark datasets from the UCI and OpenML repositories, as well as in an application to day-ahead electricity price forecasting. In both cases, the approach is compared to the global quantile recalibration approach of Kuleshov et al. (2018), as well as a conditional variant introduced by Kuleshov and Deshpande (2022). The conditional variant is designed to recalibrate predictive distributions so that they are auto-calibrated.

## 5.1 Diagnostics

To evaluate the various predictive distributions, we quantify the degree of miscalibration using the distance of the distribution of PIT values from a standard

uniform distribution. For $u _ { j } = j / J$ , for $j = 1 , \dots , J ,$ with $J = 1 0$ , probabilistic calibration is assessed using

$$
\mathrm { C a l \mathrm { - } E r r } = \sum _ { j = 1 } ^ { J } \left( \left[ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { 1 } \{ F _ { i } ( y _ { i } ) \leq u _ { j } \} \right] - u _ { j } \right) ^ { 2 } .
$$

This follows Kuleshov et al. (2018) and corresponds to the squared error between the empirical distribution function of the forecast PIT values and the distribution function of a standard uniform distribution, averaged over a discrete grid. We additionally visualise this distance to uniformity graphically via a quantile-quantile plot that shows the sorted PIT values $\{ F _ { i } ( y _ { i } ) \} _ { i = 1 } ^ { N }$ <sub>1</sub> plotted against reference values $1 / N , 2 / N , \ldots , 1$

Outcome-conditional calibration on an interval $\mathcal { Z } =$ $[ a , b )$ is similarly assessed using the same measure of calibration error applied to the empirical distribution of the conditional PIT values. That is,

$$
\sum _ { j = 1 } ^ { J } \left( \left[ \frac { 1 } { { \cal N } _ { \mathcal { T } } } \sum _ { \{ i : y _ { i } \in \mathcal { I } \} } \mathbb { 1 } \{ F _ { i } ^ { \mathcal { T } } ( y _ { i } ) \leq u _ { j } \} \right] - u _ { j } \right) ^ { 2 } ,
$$

where $N _ { \mathcal { T } } = | \{ i = 1 , \dots , N : y _ { i } \in \mathcal { T } \} |$ is the number of observations that fall in I. We also refer to this as subset calibration error. This only assesses CPIT calibration, the first requirement in Definition 3.2. The second requirement, occurrence calibration, is assessed by computing the diference between the average predicted probability that $Y \in \mathcal { Z }$ and the realised occurrence frequency. That is, for each interval I

$$
{ \mathrm { O c c - E r r } } = \left| { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \left\{ \left[ F _ { i } ( b ) - F _ { i } ( a ) \right] - \mathbb { 1 } \{ y _ { i } \in { \mathcal { I } } \} \right\} \right| .
$$

For both the calibration error and the occurrence error, a value closer to zero suggests that the predictive distributions exhibit better calibration.

## 5.2 UCI and OpenML experiments

Data. The recalibration methods are first compared on a range of datasets from the UCI and OpenML repositories with varying sizes, from $N \ = \ 7 6 8$ to $N = 4 5 7 3 0 .$ . A random 25% of the data is used for testing, and the remaining 75% to train the models and recalibration schemes, with the split stratified so that both sets have similar outcome distributions. Results are averaged over 10 random splits. For each dataset, outcome-conditional recalibration is performed using five intervals, defined using evenly spaced quantiles of the distribution of the outcomes.

Models. Probabilistic predictions are first obtained using Bayesian Ridge regression (MacKay, 1992), as well as MC-Dropout neural networks (Gal and

![](images/28a7689251648c0bbd015268be3091c23cfd1a2ff250f7a0662cce564cd020da.jpg)  
Figure 1: Outcome-conditional (CPIT/subset) vs overall calibration error averaged over all datasets and models for each of the recalibration methods. The vertical and horizontal bars indicate standard error across seeds, datasets and base methods.

Ghahramani, 2016) with two hidden layers of 128 units each, a dropout rate of 0.5, and PReLU activation functions. These predictions are recalibrated using the quantile recalibration of Kuleshov et al. (2018, KFE18), the covariate-dependent recalibration of Kuleshov and Deshpande (2022, KD22), and outcome-conditional recalibration (OC). We additionally consider the two step procedure that implements outcome-conditional recalibration followed by quantile recalibration (OC+QR). Recalibration is performed in-sample on the training data set. We also tested fitting the recalibration methods out-of-sample, reserving 30% of the training data for recalibration (see Appendix D).

Results. Figure 1 shows overall (probabilistic) and outcome-conditional (CPIT/subset) calibration error averaged over all datasets and both prediction models. Outcome-conditional calibration error is averaged across all five intervals. Table 2 in the Appendix shows outcome-conditional calibration error and occurrence calibration error, and Table 3 shows overall calibration error, for each dataset and model.

Without recalibration, both Bayesian Ridge regression and MC-Dropout yield miscalibrated predictive distributions, which are particularly miscalibrated when attention is placed on particular outcomes. Quantile recalibration (Kuleshov et al., 2018) improves overall calibration, but the resulting recalibrated predictive distributions remain severely miscalibrated within the five intervals under consideration. Covariateconditional recalibration (Kuleshov and Deshpande, 2022) improves outcome-conditional calibration whilst marginally deteriorating overall calibration. However, as expected, outcome-conditional recalibration outperforms all other methods in terms of occurrence error and outcome-conditional calibration error; this is the case for all datasets and for both prediction models.

Outcome-conditional recalibration leads to slightly worse overall calibration than quantile recalibration. This can be circumvented by applying quantile recalibration to the outcome-recalibrated predictive distributions; the resulting predictions exhibit the best overall calibration, without a large sacrifice in calibration when attention is restricted to particular outcomes.

Discussion. The results suggest that outcomeconditional recalibration is able to successfully recalibrate predictive distributions in particular regions of the outcome space, whereas the standard quantile recalibration of Kuleshov et al. (2018) does not; this is found to hold across a range of benchmark regression datasets. The predictions are evaluated using the same partition of the outcome space on which outcomerecalibration is trained, which is the canonical approach if the partitions corresponds to subsets of interest. However, the outcome-conditional recalibrated predictive distributions also exhibit good overall calibration, and results in Appendix E demonstrate that the good outcome-conditional calibration also transfers to other partitions. The covariate-conditional recalibration of Kuleshov and Deshpande (2022) is competitive with outcome-conditional recalibration, though it is based on a much slower deep learningbased density fit. In contrast, outcome-recalibration is hyperparameter-free and can be applied out-of-the-box without the need for an additional validation dataset, nor complex density estimation.

## 5.3 Electricity Price Forecasting

Consider now an application to day-ahead forecasting of hourly electricity prices. We apply the recalibration methods to a state-of-the-art distributional neural network model (Lakshminarayanan et al., 2017; Rasp and Lerch, 2018), which was proposed for electricity price forecasting by Marcjasz et al. (2023). We focus in particular on whether the recalibration methods yield calibrated predictions for negative electricity prices, which are rare, impactful events that can occur when electricity generation exceeds electricity demand.

Data. We study German day-ahead electricity prices from the EPEX spot market from 2015–2020. We consider hourly electricity prices, with 24 prices per delivery day. Forecasts are issued based on a set of covariates, including day-ahead load and renewablegeneration predictions; EUA carbon, API2 coal, Brent oil, and TTF gas prices; lagged electricity prices at lags of 1, 2, 3, 7 days; and calendar dummy variables;

these are the same covariates that were used by Marcjasz et al. (2023). The data were obtained from the ENTSO-E platform and investing.com.

Model. The recalibration methods are applied to a distributional deep neural network (DDNN) whose outputs are the parameters of a Johnson’s $S _ { U }$ (JSU) distribution. The JSU distribution is a flexible probability distribution that can model skewed and heavytailed data with support on the entire real line, which Marcjasz et al. (2023) find to be appropriate for German electricity price forecasts. The resulting predictive distribution function is $F ( x ) = \Phi ( \gamma + \delta$ asinh((x− $\xi ) / \lambda ) ) , x \in \mathbb { R }$ , where Φ is the standard Gaussian distribution function, and $( \xi , \lambda , \delta , \gamma )$ are the estimated four parameters of the JSU distribution. Separate parameters are estimated for each hour within the same network, so that the output layer has length $4 \times 2 4 = 9 6$

Following Marcjasz et al. (2023), we fit a neural network with two hidden layers, which is trained using maximum likelihood estimation. An ensemble of four networks are trained with diferent tuned hyperparameter configurations, and the estimated JSU parameters are averaged over the four configurations. The parameters are estimated sequentially using a rolling training window: the DDNN is tuned on the first three years of data (2015–2017), before being retrained weekly on a window of the 1456 days (roughly four years) prior to each evaluation day. The day-ahead forecasts are then evaluated over 2019 and 2020. All results are aggregated across all 24 hours of the day.

The DDNN predictions are recalibrated using quantile recalibration, covariate-conditional recalibration, outcome-conditional recalibration (OC), as well as outcome-conditional recalibration followed by quantile recalibration (OC+QR). Like the base DDNN model, the recalibration models are also estimated adaptively using a rolling window: for each week, the recalibration parameters are estimated on a trailing 182-day (roughly six months) window containing the DDNN’s past out-of-sample predictions, and these parameters are then used to recalibrate the DDNN predictive distributions across the following week. This is designed to mirror operational practice, in which the performance of a model would be monitored sequentially over time, and the recalibration would thus also be time-adaptive.

Outcome-conditional recalibration is performed by partitioning the outcome space into a negative-price group $( - \infty , 0 )$ and four equal-probability quantile groups of the non-negative prices. Electricity prices are negative for 1.9% of all hours between 2015–2020, which grows from ∼1.44% in 2015 to ∼3.4% in 2020.

Results. Table 1 displays the calibration error of all recalibration methods, as well as the raw DDNN model without recalibration. All methods yield predictive distributions that exhibit very good overall calibration; calibration error is always smaller than 0.001. The first panel of Figure 2 confirms that the empirical distribution of the PIT values is essentially uniform for all methods.

Table 1: Rolling (time-adaptive, 182-day) recalibration of the DDNN-JSU, German day-ahead prices, 2019–2020. CRPS in EUR/MWh; cal is the calibration error (lower is better); g<sub>0</sub> is the negative-price group; $\mathrm { O C C } _ { g _ { 0 } }$ is its occurrence ratio (1 = calibrated). Best in each column in bold.
<table><tr><td>Method</td><td>CRPS</td><td> $\mathrm { c a l } _ { \mathrm { a l l } }$ </td><td> $\mathrm { c a l } _ { g _ { 0 } }$ </td><td> $\mathrm { O C C } _ { g _ { 0 } }$ </td></tr><tr><td>Raw</td><td>2.65</td><td>0.0009</td><td>0.0611</td><td>0.939</td></tr><tr><td>KFE18</td><td>2.67</td><td>0.0008</td><td>0.0636</td><td>0.973</td></tr><tr><td>KD22</td><td>2.78</td><td>0.0006</td><td>0.1290</td><td>1.141</td></tr><tr><td>Ours: OC</td><td>2.70</td><td>0.0004</td><td>0.0051</td><td>1.050</td></tr><tr><td>Ours:  $\mathrm { O C } + \mathrm { Q R }$ </td><td>2.71</td><td>0.0010</td><td>0.0082</td><td>1.041</td></tr></table>

However, the DDNN predictive distributions are not calibrated when attention is restricted to negative electricity prices, and this is not corrected by the unconditional quantile recalibration of Kuleshov et al. (2018) or the covariate-conditional recalibration of Kuleshov and Deshpande (2022); this calibration error is also shown in Table 1. In contrast, outcome-conditional recalibration yields predictive distributions that exhibit both good overall calibration, as well as good calibration when predicting negative electricity prices.

The distribution of the conditional PIT values of all methods is shown in the second panel of Figure 2. The CPIT values appear approximately uniform for the outcome-conditional recalibration methods, lying along the diagonal line. The shape of the distribution of CPIT values for the other methods suggests that they yield predictions that over-predict the severity of negative electricity prices, despite being calibrated on average. The final panel of Figure 2 suggests that all methods predict the occurrence of negative electricity prices with roughly the correct probability on average; the miscalibration therefore comes from the JSU distribution not accurately capturing the distribution of negative electricity prices.

Table 1 additionally contains average values of the continuous ranked probability score (CRPS), a proper scoring rule. The raw DDNN produces the most accurate predictions when assessed using the average CRPS, and all recalibration methods yield slightly higher (i.e. worse) CRPS values. This demonstrates that recalibration does not necessarily lead to improved average scores; this is the case for unconditional, covariate-conditional, and outcome-conditional recalibration. However, since the goal of probabilistic forecasting is to obtain predictions that are as sharp as possible subject to calibration, a small decrease in accuracy is often tolerable if the trade-of is improved calibration. This is especially true if calibration is improved when predicting high-impact outcomes.

![](images/9db4de4bdda4367beb25adbe3c217ad90173247c52e3217db74f38e30b3b4e07.jpg)

![](images/b1d0025db5b8ffc921c46a5a84745c5e3d55beeb72ca4dcead1c41bd0b0e848e.jpg)

![](images/2d17beb0ad0781de87cf7a17ba86cc5ee78297a156f46a1d26396a0511e2535d.jpg)  
Figure 2: Rolling (182-day) recalibration of the DDNN-JSU, German day-ahead prices, 2019–2020. Left: empirical CDF of PIT values. Middle: quantile-quantile plot of CPIT values for the negative price group against reference values. Right: occurrence ratio of expected over realised events. For the left and middle panels, a calibrated forecast would lie on the 1-1 line, whereas on the right, the occurrence ratio should be close to one.

## 6 DISCUSSION AND CONCLUSIONS

Probabilistic predictions should be as informative as possible, subject to being calibrated. While state-ofthe-art machine learning models generally do not yield calibrated predictions, several methods have been proposed to recalibrate predictions. Such recalibration methods typically only enforce a weak unconditional notion of calibration, which does not ensure that predictions are calibrated when attention is restricted to particular outcomes of interest. However, in many applications, it is essential to have calibrated probabilistic predictions for certain outcomes, since certain outcomes, such as extreme events, are often particularly important for decision making; calibrated predictions for these outcomes are therefore especially valuable.

In applications to electricity price forecasting and benchmark regression datasets, we demonstrate that state-of-the-art recalibration methods generally do not yield predictive distributions that issue calibrated forecasts for certain outcomes of interest. We therefore introduce a novel post-hoc outcome-conditional recalibration method that ensures probabilistic predictions are calibrated when forecasting particular outcomes. The method is simple and hyperparameter-free, meaning it can be estimated almost instantaneously even for large datasets. It can be implemented in-sample on the training dataset, or (ideally) out-of-sample on an additional calibration dataset. Hence, there is large potential to implement outcome-conditional recalibration in many other application domains.

Outcome-conditional recalibration yields predictive distribution that are outcome-conditionally calibrated in-sample, on the calibration data with which they are trained. Allen et al. (2025a) demonstrate that prediction methods with in-sample calibration guarantees can be used to construct sets of predictions with out-of-sample calibration guarantees, akin to conformal prediction. Connections between quantile recal ibration and conformal prediction have been pointed out by Dheur and Taieb (2023), and since outcomeconditional recalibration yields forecasts that are insample calibrated, there is scope to conformalise the approach to yield prediction sets that satisfy outcomeconditional calibration properties out-of-sample.

To our knowledge, outcome-conditional recalibration is the first post-hoc recalibration method that targets outcome-dependent notions of calibration. While auto-calibration (or distribution calibration) implies outcome-conditional calibration in theory, our empirical results suggest that recalibration schemes designed to enforce auto-calibration (e.g. Kuleshov and Deshpande, 2022) do not necessarily yield predictive distributions that are calibrated when forecasting particular outcomes. In contrast, outcome-conditional recalibration does. Further work is therefore needed to clarify the connections between outcome-conditional calibration and auto-calibration.

## Acknowledgements

We thank Rafael Weinert for providing code and data that was used for the application to electricity price forecasts.

## Code and Data availability

The code to run all experiments and reproduce results in this paper will be provided upon acceptance.

## AI Use Statement

The authors used the generative AI models Claude Opus 4.8 and Claude Opus 5 in Claude Code to support with method implementation, as well as for editing of the manuscript. All outputs have been carefully checked and code has been tested for correctness. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

Allen, S., Gavrilopoulos, G., Henzi, A., Kleger, G.-R., and Ziegel, J. (2025a). In-sample calibration yields conformal calibration guarantees. arXiv preprint arXiv:2503.03841.

Allen, S., Koh, J., Segers, J., and Ziegel, J. (2025b). Tail calibration of probabilistic forecasts. Journal of the American Statistical Association, 120:2796– 2808.

Allen, S., Ziegel, J., and Ginsbourger, D. (2024). Assessing the calibration of multivariate probabilistic forecasts. Quarterly Journal of the Royal Meteorological Society, 150:1315–1335.

Chernozhukov, V., W¨uthrich, K., and Zhu, Y. (2021). Distributional conformal prediction. Proceedings of the National Academy ofSciences, 118:e2107794118.

Christofersen, P. F. (1998). Evaluating interval forecasts. International Economic Review, 39:841–862.

Chung, Y., Char, I., and Schneider, J. (2024). Sampling-based multi-dimensional recalibration. In Proceedings of the 41st International Conference on Machine Learning, pages 8919–8940. PMLR.

Dawid, A. P. (1984). Present position and potential developments: Some personal views statistical theory the prequential approach. Journal of the Royal Statistical Society: Series A (General), 147:278–290.

Dheur, V. and Ben Taieb, S. (2026). Multivariate latent recalibration for conditional normalizing flows. Advances in Neural Information Processing Systems, 38:4946–4988.

Dheur, V. and Taieb, S. B. (2023). A large-scale study of probabilistic calibration in neural network regression. In Proceedings of the 40th International Conference on Machine Learning, pages 7813–7836. PMLR.

Dheur, V. and Taieb, S. B. (2024). Probabilistic calibration by design for neural network regression. In

International Conference on Artificial Intelligence and Statistics, pages 3133–3141. PMLR.

Diebold, F. X., Gunther, T. A., and Tay, A. S. (1998). Evaluating density forecasts with applications to financial risk management. International Economic Review, 39:863–883.

Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. In Balcan, M. F. and Weinberger, K. Q., editors, Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1050–1059, New York, New York, USA. PMLR.

Gneiting, T., Balabdaoui, F., and Raftery, A. E. (2007). Probabilistic Forecasts, Calibration and Sharpness. Journal of the Royal Statistical Society. Series B (Statistical Methodology), 69:243–268.

Gneiting, T. and Ranjan, R. (2013). Combining predictive distributions. Electronic Journal of Statistics, 7:1747–1782.

Gneiting, T. and Resin, J. (2023). Regression diagnostics meets forecast evaluation: conditional calibration, reliability diagrams, and coeficient of determination. Electronic Journal of Statistics, 17:3226– 3286.

Gneiting, T., Stanberry, L. I., Grimit, E. P., Held, L., and Johnson, N. A. (2008). Assessing probabilistic forecasts of multivariate quantities, with an application to ensemble predictions of surface winds. TEST, 17:211–235.

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. In International Conference on Machine Learning, pages 1321–1330. PMLR.

Hebert-Johnson, U., Kim, M., Reingold, O., and Rothblum, G. (2018). Multicalibration: Calibration for the (Computationally-Identifiable) Masses. In Proceedings of the 35th International Conference on Machine Learning, pages 1939–1948. PMLR.

Kock, L., Rodrigues, G., Sisson, S. A., Klein, N., and Nott, D. J. (2026). Calibrating multivariate regression with localized PIT mappings. Journal of Computational and Graphical Statistics, pages 1–22.

Kuleshov, V. and Deshpande, S. (2022). Calibrated and Sharp Uncertainties in Deep Learning via Density Estimation. In Proceedings of the 39th International Conference on Machine Learning, pages 11683–11693. PMLR.

Kuleshov, V., Fenner, N., and Ermon, S. (2018). Accurate Uncertainties for Deep Learning Using Calibrated Regression. In Proceedings of the 35th In-

ternational Conference on Machine Learning, pages 2796–2804. PMLR.

Kull, M. and Flach, P. (2015). Novel decompositions of proper scoring rules for classification: Score adjustment as precursor to calibration. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 68–85. Springer.

Laajil, A., Zhalieva, E., Desobry, N., and Taieb, S. B. (2026). Calibrated multivariate distributional regression with pre-rank regularization. arXiv preprint arXiv:2601.22895.

Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems, 30:6402–6413.

MacKay, D. J. C. (1992). Bayesian Interpolation. Neural Computation, 4:415–447.

Marcjasz, G., Narajewski, M., Weron, R., and Ziel, F. (2023). Distributional neural networks for electricity price forecasting. Energy Economics, 125:106843.

Rasp, S. and Lerch, S. (2018). Neural networks for postprocessing ensemble weather forecasts. Monthly Weather Review, 146:3885–3900.

Resin, J., Yang, L., and Gneiting, T. (2026). Hierarchies of calibration: Classification meets regression. arXiv preprint arXiv:2606.03245.

Romano, Y., Patterson, E., and Candes, E. (2019). Conformalized quantile regression. Advances in neural information processing systems, 32:3538–3548.

Song, H., Diethe, T., Kull, M., and Flach, P. (2019). Distribution calibration for regression. In Proceedings of the 36th International Conference on Machine Learning, pages 5897–5906. PMLR.

Tsyplakov, A. (2011). Evaluating density forecasts: a comment. Available at SSRN 1907799.

Utpala, S. and Rai, P. (2020). Quantile regularization: Towards implicit calibration of regression models. arXiv preprint arXiv:2002.12860.

Vovk, V., Shen, J., Manokhin, V., and Xie, M.- g. (2017). Nonparametric predictive distributions based on conformal prediction. In Conformal and probabilistic prediction and applications, pages 82– 102. PMLR.

Wessel, J. B., Schillinger, M., Kwasniok, F., and Allen, S. (2026). Enforcing tail calibration when training probabilistic forecast models. International Journal of Forecasting, page S0169207026000464.

Wilks, D. S. (2018). Enforcing calibration in ensemble postprocessing. Quarterly Journal of the Royal Meteorological Society, 144:76–84.

# Supplementary Materials

## A PROOFS

Proof of Proposition 4.1. The proof follows easily from the fact that $T _ { \mathcal { T } _ { g } }$ and $F ^ { \pmb { \mathcal { T } } _ { g } }$ are continuous non-decreasing functions from zero to one, for all $g = 1 , \ldots , G$ . That is, $T _ { \mathcal { I } _ { q } }$ is non-decreasing with $T _ { \mathcal { T } _ { g } } ( 0 ) = 0$ and $T _ { \mathcal { T } _ { g } } ( 1 ) = 1$ and $F ^ { \mathcal { T } _ { g } }$ is non-decreasing with $F ^ { \mathcal { Z } _ { g } } ( - \infty ) = 0$ and $F ^ { \mathcal { T } _ { g } } ( + \infty ) = 1$ . Hence,

$$
\widetilde { \cal F } ( - \infty ) = m _ { 0 } + m _ { 1 } \left[ { \cal T } _ { \mathcal { T } _ { 1 } } \circ { \cal F } ^ { \mathcal { T } _ { g } } ( - \infty ) \right] = m _ { 1 } { \cal T } _ { \mathcal { T } _ { 1 } } ( 0 ) = 0
$$

and

$$
\widetilde { F } ( + \infty ) = \sum _ { i = 0 } ^ { G - 1 } m _ { i } + m _ { G } \left[ T _ {  { { \mathbb Z } } _ { G } } \circ F ^ {  { { \mathbb Z } } _ { G } } ( + \infty ) \right] = \sum _ { i = 0 } ^ { G - 1 } m _ { i } + m _ { G } T _ {  { { \mathbb Z } } _ { G } } ( 1 ) = \sum _ { i = 0 } ^ { G } m _ { i } = 1 .
$$

Since $T _ { \mathcal { I } _ { g } }$ and $F ^ { \pmb { \mathscr { T } } _ { g } }$ are non-decreasing, so is $T _ { \mathcal { T } _ { g } } \circ F ^ { \mathcal { T } _ { g } }$ . Hence, for $x _ { 1 } \leq x _ { 2 }$ with $x _ { 1 } , x _ { 2 } \in \mathcal { T } _ { g }$

$$
\widetilde { F } ( x _ { 1 } ) = \sum _ { i = 1 } ^ { g - 1 } m _ { i } + m _ { g } \left[ T _ { \mathcal { Z } _ { g } } \circ F ^ { \mathcal { T } _ { g } } ( x _ { 1 } ) \right] \leq \sum _ { i = 1 } ^ { g - 1 } m _ { i } + m _ { g } \left[ T _ { \mathcal { Z } _ { g } } \circ F ^ { \mathcal { T } _ { g } } ( x _ { 2 } ) \right] = \widetilde { F } ( x _ { 2 } ) .
$$

If $x _ { 1 } \in \mathcal { T } _ { g _ { 1 } }$ <sub>1</sub> and $x _ { 2 } \in \mathcal { I } _ { g _ { 2 } }$ , for $g _ { 1 } < g _ { 2 }$ , then

$$
\widetilde { F } ( x _ { 1 } ) = \sum _ { i = 1 } ^ { g _ { 1 } - 1 } m _ { i } + m _ { g _ { 1 } } \left[ T _ { \mathcal { I } _ { g _ { 1 } } } \circ F ^ { \mathcal { Z } _ { g _ { 1 } } } ( x _ { 1 } ) \right] \leq \sum _ { i = 1 } ^ { g _ { 1 } } m _ { i } \leq \sum _ { i = 1 } ^ { g _ { 2 } - 1 } m _ { i } + m _ { g _ { 2 } } \left[ T _ { \mathcal { I } _ { g _ { 2 } } } \circ F ^ { \mathcal { Z } _ { g _ { 2 } } } ( x _ { 2 } ) \right] = \widetilde { F } ( x _ { 2 } ) .
$$

The recalibrated predictive distribution $\widetilde { F }$ is therefore non-decreasing, and bounded between zero and one.

Continuity within intervals $\mathcal { T } _ { g }$ follows from the continuity of $T _ { \mathcal { I } _ { g } }$ and $F ^ { \mathcal { T } _ { g } }$ . Let ${ \mathcal { T } } _ { g } = [ a _ { g - 1 } , a _ { g } )$ , at the limits one has:

$$
\operatorname* { l i m } _ { x \uparrow a _ { g } } \tilde { F } ( x ) = \sum _ { i = 1 } ^ { g - 1 } m _ { i } + m _ { g } = \sum _ { i = 1 } ^ { g } m _ { i } = \operatorname* { l i m } _ { x \downarrow a _ { g } } \tilde { F } ( x ) ,
$$

as F<sup>I</sup>g (a<sub>g−1</sub>) = 0, F<sup>I</sup>g (a<sub>g</sub>) = 1 and T<sub>I</sub> (0) = 0, T<sub>I</sub> (1) = 1.

Proof of Proposition $4 . 2 .$ . For any $g = 1 , \ldots , G$ , we have $F ^ { \mathcal { Z } _ { g } } ( a _ { g - 1 } ) = 0$ and $F ^ { \mathcal { Z } _ { g } } ( a _ { g } ) = 1$ . Hence, for any $x \in \mathcal { T } _ { g }$ we have

$$
\tilde { F } ^ { Z _ { g } } ( x ) = \frac { \widetilde { F } ( x ) - \widetilde { F } ( a _ { g - 1 } ) } { \widetilde { F } ( a _ { g } ) - \widetilde { F } ( a _ { g - 1 } ) } \quad = \frac { \sum _ { i = 0 } ^ { g - 1 } m _ { i } + m _ { g } \left[ T _ { \widetilde { X } _ { g } } \circ F ^ { T _ { g } } ( x ) \right] - \sum _ { i = 0 } ^ { g - 1 } m _ { i } } { \sum _ { i = 0 } ^ { g } m _ { i } - \sum _ { i = 0 } ^ { g - 1 } m _ { i } } = \frac { m _ { g } \left[ T _ { \widetilde { X } _ { g } } \circ F ^ { T _ { g } } ( x ) \right] } { m _ { g } } = T _ { \widetilde { X } _ { g } } \circ F ^ { T _ { g } } ( x ) .
$$

Conditional on $Y \in \mathcal { T } _ { g } , \tilde { F } ^ { \mathcal { Z } _ { g } } ( Y ) = T _ { \mathcal { T } _ { g } } \circ F ^ { \mathcal { T } _ { g } } ( Y )$ follows a standard uniform distribution by construction. The recalibrated forecast distribution $\widetilde { F }$ therefore satisfies the first requirement of local calibration in Definition 3.2, for all intervals $\mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { G }$ . The second requirement in Definition 3.2 follows since

$$
\mathbb { E } [ \widetilde { F } ( a _ { g } ) - \widetilde { F } ( a _ { g - 1 } ) ] = \mathbb { E } [ m _ { g } ] = \mathbb { E } \left[ \frac { c _ { j } [ F ( a _ { j } ) - F ( a _ { j - 1 } ) ] } { \sum _ { g = 1 } ^ { G } c _ { g } [ F ( a _ { g } ) - F ( a _ { g - 1 } ) ] } \right] = \mathbb { P } ( Y \in \mathbb { Z } _ { j } ) .
$$

## B SAMPLE-LEVEL ESTIMATION

We provide a sample analysis and description of the method and implementation.

Setup. Consider a partition of the outcome space R into G contiguous intervals

$$
\mathbb { R } = \bigcup _ { g = 1 } ^ { G } \mathcal { Z } ^ { ( g ) } , \qquad \mathcal { T } ^ { ( g ) } = [ a ^ { ( g ) } , b ^ { ( g ) } ) ,
$$

with $a ^ { ( 1 ) } = - \infty$ and $b ^ { ( G ) } = + \infty$ . Write $N ^ { ( g ) } : = | \{ i : y _ { i } \in \mathcal { I } ^ { ( g ) } \} |$ for the group sample size, $q ^ { ( g ) } : = \mathbb { P } ( Y \in \mathcal { T } ^ { ( g ) } )$ for the occurrence probability (estimated by $N ^ { ( g ) } / N )$ , and $\Delta _ { i } ^ { ( g ) } : = F _ { i } ( b ^ { ( g ) } )  – F _ { i } ( a ^ { ( g ) } )$ for the forecast probability mass assigned to $\boldsymbol { \mathcal { T } ^ { ( g ) } }$ by forecast i. Note that $\begin{array} { r } { \sum _ { g = 1 } ^ { G } \Delta _ { i } ^ { ( g ) } = 1 } \end{array}$ for every i. Finally, write $F _ { i } ^ { ( g ) } ( \cdot ) : = [ F _ { i } ( \cdot ) - F _ { i } ( a ^ { ( g ) } ] / \Delta _ { i } ^ { ( g ) }$ for the CDF truncated on group g.

Intensity correction. For each group $g = 1 , \ldots , G$ , compute the conditional PIT (CPIT) values $u _ { i } ^ { ( g ) } : = F _ { i } ^ { ( g ) } ( y _ { i } )$ for all i with $y _ { i } \in \mathcal { T } ^ { ( g ) }$ , and fit a non-decreasing map $T ^ { ( g ) } : [ 0 , 1 ]  [ 0 , 1 ]$ via isotonic regression on the tuples

$$
u _ { i } ^ { ( g ) } \longrightarrow \frac { 1 } { N ^ { ( g ) } } \big | \big \{ j : u _ { j } ^ { ( g ) } \le u _ { i } ^ { ( g ) } \big \} \big | ,
$$

with boundary conditions $T ^ { ( g ) } ( 0 ) = 0$ and $T ^ { ( g ) } ( 1 ) = 1$ . These maps correct probabilistic or quantile calibration on the group g.

Occurrence correction. To correct the probability mass assigned to each group, we introduce positive group scalings $c ^ { ( 1 ) } , \ldots , c ^ { ( G ) }$ and define the recalibrated group mass

$$
m _ { i } ^ { ( g ) } = \frac { c ^ { ( g ) } \Delta _ { i } ^ { ( g ) } } { \sum _ { j = 1 } ^ { G } c ^ { ( j ) } \Delta _ { i } ^ { ( j ) } } .\tag{2}
$$

By construction $\begin{array} { r } { \sum _ { q = 1 } ^ { G } m _ { i } ^ { ( g ) } = 1 } \end{array}$ for every i, so the recalibrated forecast defined below is a proper CDF. The scalings are chosen so that the average recalibrated mass in each group matches the empirical occurrence rate,

$$
\frac { 1 } { N } \sum _ { i = 1 } ^ { N } m _ { i } ^ { ( g ) } = q ^ { ( g ) } , \qquad g = 1 , \ldots , G .\tag{3}
$$

Because the normalisation in eq. (2) couples all groups, eq. (3) has no closed-form solution; instead $\{ c ^ { ( g ) } \}$ is obtained by iterative proportional fitting (Sinkhorn’s algorithm). Starting from $c ^ { ( g ) } = 1$ , iterate

$$
c ^ { ( g ) } \ :  \ : c ^ { ( g ) } \frac { q ^ { ( g ) } } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } m _ { i } ^ { ( g ) } } , \qquad g = 1 , \ldots , G ,\tag{4}
$$

recomputing $m _ { i } ^ { ( g ) }$ from (2) after each sweep, until convergence. The overall scale of $\{ c ^ { ( g ) } \}$ cancels in (2) and may be fixed by normalising $\begin{array} { r } { \sum _ { g } c ^ { ( g ) } = 1 } \end{array}$ . Setting $c ^ { ( g ) } \equiv 1$ recovers the uncorrected case $m _ { i } ^ { ( g ) } = \Delta _ { i } ^ { ( g ) }$

The recalibrated CDF is then obtained as combination of the occurrence and intensity correction:

Recalibrated CDF. For a new forecast–observation pair $( F _ { i } , y _ { i } )$ , using the scalings $\{ c ^ { ( g ) } \}$ learned on the training set, define the group-specific intercept

$$
A _ { i } ^ { ( g ) } = \sum _ { j = 1 } ^ { g - 1 } m _ { i } ^ { ( j ) } .\tag{5}
$$

The outcome-conditional recalibrated CDF for $y \in \mathcal { I } ^ { ( g ) }$ is

$$
\widetilde { F } _ { i } ( y ) = { \cal A } _ { i } ^ { ( g ) } + m _ { i } ^ { ( g ) } T ^ { ( g ) } \left( u _ { i } ^ { ( g ) } \right) ,\tag{6}
$$

where $u _ { i } ^ { ( g ) } = \big ( F _ { i } ( y ) - F _ { i } ( a ^ { ( g ) } ) \big ) / \Delta _ { i } ^ { ( g ) }$ is the CPIT value of the new observation. Here $A _ { i } ^ { ( g ) }$ is the recalibrated value of $F _ { i } ( a ^ { ( g ) } )$ and $m _ { i } ^ { ( g ) }$ is the recalibrated probability mass assigned to $\boldsymbol { \mathcal { T } ^ { ( g ) } }$ . This corresponds to (1). Validity, continuity and in-sample outcome-conditional calibration of the resulting CDF is established by Propositions 4.1 and 4.2.

## C PER-DATASET UCI/OPENML RESULTS

The following ten UCI and OpenML datasets were used in Section 5.2: energy (N = 768, UCI-ML id = 242), kinematics (N = 8192, Delve kin8nm), power (N = 9568, UCI-ML id = 294), airfoil (N = 1503, UCI-ML id = 291), superconductivity (N = 21263, UCI-ML id = 464), protein (N = 45730, UCI-ML id = 265), ames (N = 2930, OpenML id = 43926), bank (N = 8192, OpenML bank8FM), elevators (N = 16599, OpenML elevators) and puma8NH (N = 8192, OpenML puma8NH). All experiments were run on a modern laptop (Apple M4 chip, 16Gb RAM).

In this Appendix, we present results per dataset and per model. Table 2 shows outcome-conditional (CPIT/subset) calibration error and occurrence error for each of the competing recalibration methods (uncalibrated, KFE18, KD22, outcome-conditional recalibration: OC, and outcome-conditional + quantile recalibration: OC+QR) and for each dataset and base method (Bayesian Ridge and MC Dropout). Table 3 shows overall calibration error. We find that outcome-conditional recalibration outperforms all other methods in terms of CPIT/subset calibration error and occurrence error, in all ten datasets. In terms of overall calibration, KFE18 and outcome-conditional + quantile recalibration appear best for most datasets. The OC+QR method seems to be a good intermediate that is close on subset and occurrence calibration to the outcome-conditional method, but also has good overall calibration.

Table 2: Outcome-conditional (CPIT/subset) calibration error and occurrence error across datasets and methods. Values report the mean ± sample standard deviation over ten seeds, scaled by 10<sup>2</sup>. Lower is better. Bold indicates the lowest mean among methods for a fixed base model, dataset, and metric.  
Bayesian ridge
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Uncal.</td><td colspan="2">Kuleshov et al. &#x27;18</td><td colspan="2">Kuleshov-Deshpande &#x27;22</td><td colspan="2">Ours: OC</td><td colspan="2">Ours: OC + QR</td></tr><tr><td>Cal-sub.</td><td>Occ-sub.</td><td>Cal-sub.</td><td>Occ-sub.</td><td>Cal-sub.</td><td>Occ-sub. ↓</td><td>Cal-sub. ↓</td><td>Occ-sub.</td><td>Cal-sub.</td><td>Occ-sub. ↓</td></tr><tr><td>airfoil</td><td>2.60 ± 0.47</td><td>1.93 ± 0.34</td><td>3.10 ± 0.58</td><td>2.18 ± 0.37</td><td>1.89 ± 0.69</td><td>1.19 ± 0.28</td><td>1.39 ± 0.35</td><td>0.54 ± 0.30</td><td>1.66 ± 0.29</td><td>0.65 ± 0.32</td></tr><tr><td>ames</td><td>7.28 ± 0.96</td><td>2.56 ± 0.24</td><td>6.16 ± 0.90</td><td>2.19 ± 0.28</td><td>1.40 ± 0.60</td><td>0.62 ± 0.21</td><td>1.06 ± 0.41</td><td>0.43 ± 0.16</td><td>2.79 ± 0.83</td><td>1.02 ± 0.24</td></tr><tr><td>bank</td><td>32.26 ± 1.32</td><td>3.74 ± 0.16</td><td>30.32 ± 1.58</td><td>3.73 ± 0.16</td><td>1.67 ± 1.03</td><td>0.44 ± 0.14</td><td>0.43 ± 0.14</td><td>0.27 ± 0.09</td><td>1.31 ± 0.46</td><td>0.34 ± 0.12</td></tr><tr><td>elevators</td><td>26.56 ± 0.56</td><td>4.79 ± 0.15</td><td>19.05 ± 0.60</td><td>3.03 ± 0.16</td><td>13.82 ± 0.92</td><td>0.54 ± 0.21</td><td>0.20 ± 0.06</td><td>0.28 ± 0.09</td><td>0.42 ± 0.12</td><td>0.66 ± 0.13</td></tr><tr><td>energy</td><td>29.09 ± 5.59</td><td>3.79 ± 0.46</td><td>27.02 ± 6.43</td><td>3.78 ± 0.47</td><td>9.16 ± 3.68</td><td>1.22 ± 0.49</td><td>3.46 ± 1.03</td><td>0.91 ± 0.20</td><td>5.26 ± 1.90</td><td>1.12 ± 0.40</td></tr><tr><td>kinematics</td><td>0.34 ± 0.07</td><td>1.06 ± 0.10</td><td>1.21 ± 0.34</td><td>1.37 ± 0.10</td><td>0.34 ± 0.06</td><td>0.34 ± 0.17</td><td>0.25 ± 0.08</td><td>0.22 ± 0.11</td><td>0.93 ± 0.28</td><td>0.70 ± 0.21</td></tr><tr><td>power</td><td>3.26 ± 0.49</td><td>1.21 ± 0.11</td><td>2.94 ± 0.47</td><td>1.19 ± 0.12</td><td>0.45 ± 0.12</td><td>0.33 ± 0.11</td><td>0.32 ± 0.07</td><td>0.28 ± 0.12</td><td>0.38 ± 0.08</td><td>0.28 ± 0.13</td></tr><tr><td>protein</td><td>44.51 ± 0.44</td><td>8.98 ± 0.05</td><td>32.81 ± 0.53</td><td>7.57 ± 0.06</td><td>0.52 ± 0.28</td><td>0.41 ± 0.19</td><td>0.06 ± 0.02</td><td>0.07 ± 0.02</td><td>0.10 ± 0.03</td><td>0.30 ± 0.04</td></tr><tr><td>puma8NH</td><td>7.75 ± 0.21</td><td>3.77 ± 0.09</td><td>3.85 ± 0.31</td><td>3.86 ± 0.09</td><td>0.83 ± 0.29</td><td>0.85 ± 0.23</td><td>0.24 ± 0.06</td><td>0.19 ± 0.12</td><td>0.27 ± 0.07</td><td>0.21 ± 0.13</td></tr><tr><td>supercon</td><td>49.72 ± 0.74</td><td>6.87 ± 0.10</td><td>50.22 ± 0.78</td><td>6.85 ± 0.10</td><td>1.14 ± 0.88</td><td>0.72 ± 0.21</td><td>0.13 ± 0.04</td><td>0.20 ± 0.08</td><td>0.35 ± 0.05</td><td>0.82 ± 0.11</td></tr></table>

MC dropout
<table><tr><td>Dataset</td><td colspan="2">Uncal.</td><td colspan="2">Kuleshov et al.&#x27;18</td><td colspan="2">Kuleshov-Deshpande &#x27;22</td><td colspan="2">Ours: OC</td><td colspan="2">Ours: OC + QR</td></tr><tr><td></td><td>Cal-sub. ↓</td><td>Occ-sub.</td><td>Cal-sub. ↓</td><td>Occ-sub. 水</td><td>Cal-sub.</td><td>Occ-sub. ↓</td><td>Cal-sub. ↓</td><td>Occ-sub.</td><td>Cal-sub. 火</td><td>Occ-sub. ↓</td></tr><tr><td>airfoil</td><td>3.33 ± 1.21</td><td>2.46 ± 0.67</td><td>5.40 ± 1.45</td><td>3.21 ± 0.65</td><td>2.19 ± 0.37</td><td>1.07 ± 0.32</td><td>1.53 ± 0.39</td><td>0.59 ± 0.30</td><td>3.60 ± 1.64</td><td>1.01 ± 0.50</td></tr><tr><td>ames</td><td>8.69 ± 8.97</td><td>2.84 ± 1.65</td><td>9.84 ± 2.63</td><td>1.20 ± 0.55</td><td>6.74 ± 1.78</td><td>0.74 ± 0.27</td><td>2.74 ± 1.12</td><td>0.46 ± 0.19</td><td>13.16 ± 2.66</td><td>2.58 ± 1.48</td></tr><tr><td>bank</td><td>19.25 ± 1.74</td><td>1.81 ± 0.43</td><td>14.42 ± 1.66</td><td>1.71 ± 0.34</td><td>1.85 ± 1.55</td><td>0.42 ± 0.19</td><td>0.55 ± 0.24</td><td>0.29 ± 0.08</td><td>0.81 ± 0.28</td><td>0.31 ± 0.08</td></tr><tr><td>elevators</td><td>16.14 ± 1.68</td><td>1.72 ± 0.86</td><td>14.76 ± 2.01</td><td>1.04 ± 0.59</td><td>12.43 ± 1.20</td><td>0.48 ± 0.16</td><td>0.26 ± 0.11</td><td>0.32 ± 0.12</td><td>0.39 ± 0.18</td><td>0.47 ± 0.19</td></tr><tr><td>energy</td><td>19.20 ± 3.55</td><td>3.66 ± 1.33</td><td>16.10 ± 4.21</td><td>3.24 ± 1.42</td><td>8.86 ± 2.45</td><td>1.30 ± 0.44</td><td>3.70 ± 1.15</td><td>0.81 ± 0.17</td><td>7.06 ± 1.92</td><td>1.14 ± 0.51</td></tr><tr><td>kinematics</td><td>4.11 ± 1.16</td><td>1.35 ± 0.29</td><td>4.86 ± 1.03</td><td>1.41 ± 0.27</td><td>0.50 ± 0.15</td><td>0.37 ± 0.16</td><td>0.41 ± 0.11</td><td>0.30 ± 0.11</td><td>1.18 ± 0.28</td><td>0.36 ± 0.14</td></tr><tr><td>power</td><td>1.32 ± 0.63</td><td>0.86 ± 0.25</td><td>1.32 ± 0.66</td><td>0.92 ± 0.28</td><td>0.49 ± 0.15</td><td>0.33 ± 0.12</td><td>0.34 ± 0.10</td><td>0.27 ± 0.09</td><td>0.45 ± 0.08</td><td>0.26 ± 0.09</td></tr><tr><td>protein</td><td>43.38 ± 0.62</td><td>9.20 ± 0.40</td><td>37.33 ± 1.33</td><td>8.97 ± 0.25</td><td>0.44 ± 0.17</td><td>0.61 ± 0.18</td><td>0.07 ± 0.02</td><td>0.12 ± 0.06</td><td>0.23 ± 0.07</td><td>0.52 ± 0.09</td></tr><tr><td>puma8NH</td><td>5.97 ± 0.66</td><td>3.49 ± 0.39</td><td>4.50 ± 0.26</td><td>3.66 ± 0.46</td><td>0.63 ± 0.37</td><td>0.54 ± 0.14</td><td>0.35 ± 0.11</td><td>0.27 ± 0.15</td><td>0.58 ± 0.29</td><td>0.44 ± 0.16</td></tr><tr><td>supercon</td><td>38.35 ± 1.96</td><td>5.70 ± 0.50</td><td>27.14 ± 2.21</td><td>5.46 ± 0.59</td><td>0.51 ± 0.15</td><td>0.52 ± 0.28</td><td>0.14 ± 0.04</td><td>0.19 ± 0.06</td><td>0.73 ± 0.19</td><td>1.05 ± 0.14</td></tr></table>

Table 3: Overall calibration error across datasets and methods. Values report the mean ± sample standard deviation over ten seeds, scaled by $1 0 ^ { 2 }$ . Lower is better. Bold indicates the lowest mean among methods for a fixed base model, dataset, and metric.  
Bayesian ridge
<table><tr><td>Dataset</td><td>Uncal.</td><td>Kuleshov et al. &#x27;18</td><td>Kuleshov-Deshpande &#x27;22</td><td>Ours: OC</td><td> $\mathrm { O u r s : ~ O C + Q R }$ </td></tr><tr><td>airfoil</td><td> $0 . 3 7 \pm 0 . 2 1$ </td><td> ${ \bf 0 . 3 1 \pm 0 . 1 3 }$ </td><td> $0 . 5 5 \pm 0 . 4 5$ </td><td> $0 . 7 0 \pm 0 . 3 8$ </td><td> $0 . 3 6 \pm 0 . 2 8$ </td></tr><tr><td>ames</td><td> $3 . 1 9 \pm 0 . 7 4$ </td><td> $0 . 5 8 \pm 0 . 2 8$ </td><td> $0 . 5 5 \pm 0 . 4 1$ </td><td> $1 . 9 7 \pm 0 . 4 5$ </td><td> ${ \bf 0 . 4 2 \pm 0 . 2 2 }$ </td></tr><tr><td>bank</td><td> $1 . 1 6 \pm 0 . 4 6$ </td><td> $0 . 1 5 \pm 0 . 1 9$ </td><td> $0 . 4 1 \pm 0 . 2 9$ </td><td> $1 . 1 8 \pm 0 . 3 3$ </td><td> ${ \bf 0 . 1 5 \pm 0 . 2 0 }$ </td></tr><tr><td>elevators</td><td> $2 . 6 9 \pm 0 . 3 8$ </td><td> $\mathbf { 0 . 0 7 \pm 0 . 0 6 }$ </td><td> $0 . 1 8 \pm 0 . 1 8$ </td><td> $0 . 2 6 \pm 0 . 1 0$ </td><td> $0 . 0 7 \pm 0 . 0 5$ </td></tr><tr><td>energy</td><td> $3 . 3 9 \pm 1 . 5 8$ </td><td> $0 . 9 6 \pm 0 . 5 9$ </td><td> $2 . 0 8 \pm 1 . 2 1$ </td><td> $1 . 9 8 \pm 1 . 0 0$ </td><td> ${ \bf 0 . 5 0 \pm 0 . 3 7 }$ </td></tr><tr><td>kinematics</td><td> $0 . 7 5 \pm 0 . 2 6$ </td><td> $\mathbf { 0 . 0 8 \pm 0 . 0 5 }$ </td><td> $0 . 1 8 \pm 0 . 0 9$ </td><td> $0 . 9 0 \pm 0 . 3 4$ </td><td> $0 . 0 8 \pm 0 . 0 5$ </td></tr><tr><td>power</td><td> $0 . 1 9 \pm 0 . 0 7$ </td><td> $\mathbf { 0 . 0 9 \pm 0 . 0 6 }$ </td><td> $0 . 1 3 \pm 0 . 0 9$ </td><td> $0 . 1 6 \pm 0 . 1 0$ </td><td> $0 . 0 9 \pm 0 . 0 6$ </td></tr><tr><td>protein</td><td> $1 . 7 6 \pm 0 . 1 5$ </td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td> $0 . 0 3 \pm 0 . 0 2$ </td><td> $0 . 0 5 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 0 1 \pm 0 . 0 1 }$ </td></tr><tr><td>puma8NH</td><td> $0 . 2 6 \pm 0 . 1 0$ </td><td> $0 . 0 7 \pm 0 . 0 6$ </td><td> $0 . 2 0 \pm 0 . 1 8$ </td><td> $0 . 1 3 \pm 0 . 0 6$ </td><td> $\mathbf { 0 . 0 6 \pm 0 . 0 5 }$ </td></tr><tr><td>supercon</td><td> $0 . 5 6 \pm 0 . 1 5$ </td><td> $0 . 0 4 \pm 0 . 0 3$ </td><td> $0 . 2 2 \pm 0 . 1 7$ </td><td> $0 . 4 7 \pm 0 . 1 6$ </td><td> $\mathbf { 0 . 0 3 \pm 0 . 0 2 }$ </td></tr></table>

MC dropout
<table><tr><td>Dataset</td><td>Uncal.</td><td>Kuleshov et al. &#x27;18</td><td>Kuleshov-Deshpande &#x27;22</td><td> $\mathrm { O u r s } { \colon } \mathrm { O C }$ </td><td> $\mathrm { O u r s : ~ O C + Q R }$ </td></tr><tr><td>airfoil</td><td> $1 . 9 7 \pm 1 . 0 5$ </td><td> $\mathbf { 0 . 4 6 \pm 0 . 5 0 }$ </td><td> $0 . 9 0 \pm 0 . 5 4$ </td><td> $1 . 6 9 \pm 0 . 4 6$ </td><td> $0 . 6 3 \pm 0 . 6 4$ </td></tr><tr><td>ames</td><td> $1 2 . 9 6 \pm 1 2 . 2 5$ </td><td> $8 . 0 0 \pm 1 . 3 1$ </td><td> $9 . 4 4 \pm 0 . 6 6$ </td><td> ${ \bf 3 . 2 0 \pm 2 . 6 6 }$ </td><td> $4 . 4 7 \pm 2 . 4 6$ </td></tr><tr><td>bank</td><td> $1 . 2 0 \pm 0 . 5 4$ </td><td> $0 . 2 0 \pm 0 . 2 1$ </td><td> $0 . 5 8 \pm 0 . 6 4$ </td><td> $0 . 2 3 \pm 0 . 2 1$ </td><td> ${ \bf 0 . 2 0 \pm 0 . 2 0 }$ </td></tr><tr><td>elevators</td><td> $1 . 4 6 \pm 0 . 9 7$ </td><td> $\mathbf { 0 . 0 6 \pm 0 . 0 3 }$ </td><td> $0 . 0 7 \pm 0 . 0 5$ </td><td> $0 . 1 7 \pm 0 . 1 7$ </td><td> $0 . 0 7 \pm 0 . 0 4$ </td></tr><tr><td>energy</td><td> $5 . 3 4 \pm 3 . 1 7$ </td><td> $0 . 7 6 \pm 0 . 4 8$ </td><td> $1 . 6 3 \pm 1 . 6 8$ </td><td> $2 . 5 3 \pm 1 . 2 0$ </td><td> ${ \bf 0 . 6 3 \pm 0 . 3 2 }$ </td></tr><tr><td>kinematics</td><td> $0 . 3 4 \pm 0 . 4 7$ </td><td> $0 . 1 7 \pm 0 . 0 8$ </td><td> ${ \bf 0 . 1 6 \pm 0 . 1 1 }$ </td><td> $0 . 2 8 \pm 0 . 1 7$ </td><td> $0 . 2 0 \pm 0 . 0 5$ </td></tr><tr><td>power</td><td> $0 . 4 3 \pm 0 . 4 5$ </td><td> $0 . 1 0 \pm 0 . 0 4$ </td><td> $0 . 1 3 \pm 0 . 0 7$ </td><td> $0 . 1 0 \pm 0 . 0 7$ </td><td> $\mathbf { 0 . 0 9 \pm 0 . 0 4 }$ </td></tr><tr><td>protein</td><td> $1 . 5 1 \pm 0 . 4 5$ </td><td> ${ \bf 0 . 0 2 \pm 0 . 0 2 }$ </td><td> $0 . 1 1 \pm 0 . 1 0$ </td><td> $0 . 3 4 \pm 0 . 1 7$ </td><td> $0 . 0 3 \pm 0 . 0 2$ </td></tr><tr><td>puma8NH</td><td> $0 . 3 7 \pm 0 . 2 2$ </td><td> $\mathbf { 0 . 1 4 \pm 0 . 0 6 }$ </td><td> $0 . 2 0 \pm 0 . 1 6$ </td><td> $0 . 1 7 \pm 0 . 0 6$ </td><td> $0 . 1 4 \pm 0 . 0 5$ </td></tr><tr><td>supercon</td><td> $3 . 4 1 \pm 0 . 8 6$ </td><td> $0 . 0 4 \pm 0 . 0 2$ </td><td> $0 . 1 2 \pm 0 . 0 5$ </td><td> $0 . 7 6 \pm 0 . 1 5$ </td><td> $\mathbf { 0 . 0 3 \pm 0 . 0 3 }$ </td></tr></table>

## D OUT-OF-SAMPLE FITTING FOR THE UCI/OPENML EXPERIMENTS

We also tested fitting the recalibration methods out-of-sample, as Kuleshov and Deshpande (2022) note that their conditional recalibration method might not perform well when trained in-sample. As before, we reserve 75% of the data for training and 25% for testing. Of the training dataset, we use 30% (up to 2000 data points) to fit the recalibration models, whilst using the rest to estimate the base models. We then evaluate on the remaining 25%. Table 4 contains the results in terms of subset calibration error and occurrence error, Table 5 presents overall calibration errors. The results are similar to in-sample fitting. The outcome-conditional recalibration method performs best on almost all datasets in terms of outcome-conditional calibration; the exceptions are the ames dataset for the MC-Dropout model, where KD22 marginally outperforms outcome-conditional recalibration, and the kinematics dataset for the Bayesian ridge model, for which the base model is already very well calibrated. In terms of overall calibration error, outcome-conditional+quantile recalibration and the quantile recalibration of Kuleshov et al. (2018) again issue the most calibrated predictive distributions for all datasets.

Table 4: Outcome-conditional calibration (CPIT/subset) error and occurrence error across datasets and methods. Values report the mean ± sample standard deviation over ten seeds, scaled by $1 0 ^ { 2 }$ . Lower is better. Bold indicates the lowest mean among methods for a fixed base model, dataset, and metric.  
Bayesian ridge
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Uncal.</td><td colspan="2">Kuleshov et al. &#x27;18</td><td colspan="2">Kuleshov-Deshpande&#x27;22</td><td colspan="2">Ours: OC</td><td colspan="2">Ours: OC + QR</td></tr><tr><td>Cal-sub. ↓</td><td>Occ-sub. ↓</td><td>Cal-sub. ↓</td><td>Occ-sub. ↓</td><td>Cal-sub. ↓</td><td>Occ-sub. ↓</td><td>Cal-sub.</td><td>Occ-sub. ↓</td><td>Cal-sub.</td><td>Occ-sub. ↓</td></tr><tr><td>airfoil</td><td> $2 . 6 2 \pm 0 . 5 6$ </td><td> $1 . 9 5 \pm 0 . 4 0$ </td><td>3.20 ± 0.98</td><td> $2 . 2 4 \pm 0 . 4 9$ </td><td> $3 . 1 6 \pm 1 . 3 0$ </td><td>1.49 ± 0.35</td><td>2.10 ± 0.31</td><td>0.75 ± 0.37</td><td>2.29 ± 0.51</td><td>0.85 ± 0.38</td></tr><tr><td>ames</td><td> $7 . 6 9 \pm 1 . 0 9$ </td><td> $2 . 5 9 \pm 0 . 2 0$ </td><td>6.68 ± 1.26</td><td> $2 . 3 7 \pm 0 . 2 4$ </td><td>2.05 ± 0.79</td><td>0.76 ± 0.20</td><td>1.81 ± 0.72</td><td>0.52 ± 0.20</td><td>2.66 ± 1.01</td><td>0.84 ± 0.31</td></tr><tr><td>bank</td><td> $3 2 . 1 9 \pm 1 . 3 4$ </td><td> $3 . 7 5 \pm 0 . 1 8$ </td><td> $3 0 . 4 5 \pm 1 . 7 2$ </td><td> $3 . 7 3 \pm 0 . 1 7$ </td><td> $2 . 9 3 \pm 1 . 8 4$ </td><td> $0 . 5 5 \pm 0 . 1 8$ </td><td> $\mathbf { 0 . 6 9 \pm 0 . 3 3 }$ </td><td> ${ \bf 0 . 3 0 \pm 0 . 1 1 }$ </td><td>1.64 ± 0.80</td><td>0.37 ± 0.13</td></tr><tr><td>elevators</td><td>26.56 ± 0.65</td><td> $4 . 7 9 \pm 0 . 1 4$ </td><td> $1 8 . 7 7 \pm 0 . 7 4$ </td><td> $3 . 0 5 \pm 0 . 2 3$ </td><td> $1 4 . 3 7 \pm 0 . 9 5$ </td><td> $0 . 6 6 \pm 0 . 2 6$ </td><td>0.49 ± 0.22</td><td>0.41 ± 0.16</td><td>0.83 ± 0.38</td><td>0.80 ± 0.22</td></tr><tr><td>energy</td><td>29.13 ± 5.55</td><td> $3 . 8 3 \pm 0 . 4 3$ </td><td> $2 6 . 1 2 \pm 7 . 4 6$ </td><td> $3 . 8 2 \pm 0 . 4 7$ </td><td> $1 1 . 4 2 \pm 5 . 9 3$ </td><td> $1 . 6 5 \pm 0 . 7 3$ </td><td> ${ \bf 6 . 3 6 \pm 2 . 4 9 }$ </td><td>1.37 ± 0.58</td><td>9.04 ± 3.54</td><td>1.81 ± 0.77</td></tr><tr><td>kinematics</td><td>0.33 ± 0.07</td><td> $1 . 0 6 \pm 0 . 1 4$ </td><td> $1 . 2 8 \pm 0 . 3 4$ </td><td> $1 . 3 4 \pm 0 . 1 2$ </td><td> $0 . 4 9 \pm 0 . 1 7$ </td><td> $0 . 5 1 \pm 0 . 2 2$ </td><td></td><td>0.35 ± 0.10 0.27 ± 0.14</td><td>1.04 ± 0.35</td><td>0.62 ± 0.31</td></tr><tr><td>power</td><td>3.27 ± 0.49</td><td> $1 . 2 2 \pm 0 . 1 3$ </td><td>2.99 ± 0.56</td><td> $1 . 2 2 \pm 0 . 1 4$ </td><td> $0 . 6 2 \pm 0 . 2 1$ </td><td> $0 . 4 3 \pm 0 . 1 4$ </td><td> ${ \bf 0 . 5 3 \pm 0 . 1 9 }$ </td><td>0.30 ± 0.14</td><td>0.58 ± 0.16</td><td>0.32 ± 0.13</td></tr><tr><td>protein</td><td>44.52 ± 0.45</td><td> $8 . 9 8 \pm 0 . 0 5$ </td><td></td><td>32.42 ± 1.19 7.78 ± 0.20</td><td> $1 . 6 0 \pm 0 . 7 0$ </td><td> $0 . 8 4 \pm 0 . 3 0$ </td><td></td><td>0.23 ± 0.05 0.20 ± 0.14</td><td>0.30 ± 0.12</td><td>0.47 ± 0.14</td></tr><tr><td>puma8NH</td><td>7.72 ± 0.16</td><td> $3 . 7 7 \pm 0 . 0 8$ </td><td></td><td>3.93 ± 0.60 3.83 ± 0.15</td><td> $0 . 8 2 \pm 0 . 2 8$ </td><td></td><td></td><td>0.96 ± 0.26 0.42 ± 0.11 0.27 ± 0.18</td><td>0.42 ± 0.17</td><td>0.28 ± 0.15</td></tr><tr><td>supercon</td><td></td><td></td><td>49.79 ± 0.71 6.88 ± 0.10 49.95 ± 0.72 6.91 ± 0.10</td><td></td><td></td><td>1.01 ± 0.46 0.78 ± 0.39 0.39 ± 0.11 0.28 ± 0.11 0.57 ± 0.10 0.83 ± 0.12</td><td></td><td></td><td></td><td></td></tr></table>

MC dropout
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Uncal.</td><td colspan="2">Kuleshov et al. &#x27;18</td><td colspan="2">Kuleshov-Deshpande &#x27;22</td><td colspan="2">Ours: OC</td><td colspan="2">Ours: OC + QR</td></tr><tr><td>Cal-sub. ↓</td><td>Occ-sub. ↓</td><td>Cal-sub. 水</td><td>Occ-sub. ↓</td><td>Cal-sub. ↓</td><td>Occ-sub. ↓</td><td>Cal-sub. ↓</td><td>Occ-sub. ↓</td><td>Cal-sub. ↓</td><td>Occ-sub. ↓</td></tr><tr><td>airfoil</td><td>4.70 ± 2.26</td><td>3.32 ± 0.82</td><td>4.71 ± 1.73</td><td>3.40 ± 0.68</td><td>3.15 ± 1.43</td><td>1.51 ± 0.56</td><td>2.54 ± 0.82</td><td>0.92 ± 0.37</td><td>2.94 ± 0.91</td><td>1.11 ± 0.41</td></tr><tr><td>ames</td><td>5.08 ± 5.81</td><td>2.11 ± 0.95</td><td>2.71 ± 1.30</td><td>1.42 ± 0.28</td><td>1.55 ± 0.74</td><td>0.68 ± 0.23</td><td>1.74 ± 0.46</td><td>0.47 ± 0.20</td><td>4.22 ± 2.21</td><td>1.24 ± 0.79</td></tr><tr><td>bank</td><td>19.13 3 ± 2.19</td><td>2.33 ± 0.47</td><td>15.18 ± 1.91</td><td>2.02 ± 0.42</td><td>2.31 ± 2.59</td><td>0.58 ± 0.27</td><td>0.79 ± 0.34</td><td>0.34 ± 0.14</td><td>1.00 ± 0.43</td><td>0.35 ± 0.12</td></tr><tr><td>elevators</td><td>16.06 ± 1.38</td><td>1.56 ± 0.70</td><td>14.17 ± 2.23</td><td>0.93 ± 0.46</td><td>12.36 ± 0.86</td><td>0.56 ± 0.23</td><td>0.43 ± 0.13</td><td>0.45 ± 0.15</td><td>0.57 ± 0.13</td><td>0.56 ± 0.20</td></tr><tr><td>energy</td><td>17.56 ± 5.23</td><td>2.98 ± 0.77</td><td>15.28 ± 5.25</td><td>2.69 ± 0.83</td><td>10.73 ± 3.87</td><td>1.55 ± 0.52</td><td>6.45 ± 1.96</td><td>1.33 ± 0.53</td><td>8.67 ± 2.24</td><td>1.61 ± 0.72</td></tr><tr><td>kinematics</td><td>4.01 ± 1.05</td><td>1.37 ± 0.36</td><td>3.72 ± 0.89</td><td>1.37 ± 0.37</td><td>0.57 ± 0.26</td><td>0.52 ± 0.22</td><td>0.52 ± 0.17</td><td>0.35 ± 0.12</td><td>0.79 ± 0.35</td><td>0.38 ± 0.14</td></tr><tr><td>power</td><td>1.16 ± 0.54</td><td>0.90 ± 0.46</td><td>0.85 ± 0.61</td><td>0.88 ± 0.45</td><td>0.65 ± 0.22</td><td>0.38 ± 0.11</td><td>0.54 ± 0.14</td><td>0.30 ± 0.15</td><td>0.65 ± 0.22</td><td>0.29 ± 0.14</td></tr><tr><td>protein</td><td>43.20 ± 0.60</td><td>9.38 ± 0.51</td><td>36.63 ± 2.07</td><td>9.09 ± 0.40</td><td>1.06 ± 0.81</td><td>0.87 ± 0.25</td><td>0.28 ± 0.08</td><td>0.25 ± 0.14</td><td>0.46 ± 0.25</td><td>0.66 ± 0.22</td></tr><tr><td>puma8NH</td><td>6.10 ± 0.74</td><td>3.25 ± 0.27</td><td>4.77 ± 0.48</td><td>3.34 ± 0.28</td><td>0.59 ± 0.35</td><td>0.71 ± 0.33</td><td>0.44 ± 0.17</td><td>0.36 6 ± 0.18</td><td>0.53 ± 0.22</td><td>0.45 ± 0.16</td></tr><tr><td>supercon</td><td>40.22 ± 2.64</td><td>5.58 ± 0.70</td><td>29.06 ± 2.56</td><td>5.29 ± 0.95</td><td>0.98 ± 0.46</td><td>0.57 ± 0.23</td><td>0.43 ± 0.15</td><td>0.25 ± 0.06</td><td>0.87 ± 0.26</td><td>0.95 ± 0.19</td></tr></table>

Table 5: Overall calibration error across datasets and methods. Values report the mean $\pm$ sample standard deviation over ten seeds, scaled by $1 0 ^ { 2 }$ . Lower is better. Bold indicates the lowest mean among methods for a fixed base model, dataset, and metric.  
Bayesian ridge
<table><tr><td>Dataset</td><td>Uncal.</td><td>Kuleshov et al. &#x27;18</td><td>Kuleshov-Deshpande &#x27;22</td><td>Ours: OC</td><td>Ours:  $\mathrm { O C } + \mathrm { Q R }$ </td></tr><tr><td>airfoil</td><td> ${ \bf 0 . 3 7 \pm 0 . 2 3 }$ </td><td> $0 . 5 1 \pm 0 . 3 8$ </td><td> $1 . 3 0 \pm 0 . 7 0$ </td><td> $0 . 9 5 \pm 0 . 6 6$ </td><td> $0 . 6 0 \pm 0 . 6 2$ </td></tr><tr><td>ames</td><td> $2 . 7 4 \pm 1 . 0 2$ </td><td> $\mathbf { 0 . 4 9 \pm 0 . 3 1 }$ </td><td> $0 . 6 6 \pm 0 . 9 9$ </td><td> $1 . 8 0 \pm 0 . 4 7$ </td><td> $0 . 5 0 \pm 0 . 3 9$ </td></tr><tr><td>bank</td><td> $1 . 1 4 \pm 0 . 4 5$ </td><td> ${ \bf 0 . 1 6 \pm 0 . 1 9 }$ </td><td> $0 . 5 8 \pm 0 . 3 6$ </td><td> $1 . 1 7 \pm 0 . 4 2$ </td><td> $0 . 2 1 \pm 0 . 2 8$ </td></tr><tr><td>elevators</td><td> $2 . 7 1 \pm 0 . 3 7$ </td><td> $0 . 1 1 \pm 0 . 1 0$ </td><td> $0 . 1 4 \pm 0 . 1 0$ </td><td> $0 . 2 9 \pm 0 . 1 1$ </td><td> ${ \bf 0 . 1 1 \pm 0 . 1 0 }$ </td></tr><tr><td>energy</td><td> $3 . 3 9 \pm 1 . 6 6$ </td><td> $1 . 4 5 \pm 1 . 5 6$ </td><td> $2 . 8 4 \pm 1 . 4 0$ </td><td> $2 . 7 1 \pm 1 . 4 0$ </td><td> ${ \bf 1 . 3 3 \pm 0 . 9 9 }$ </td></tr><tr><td>kinematics</td><td> $0 . 7 1 \pm 0 . 2 6$ </td><td> ${ \bf 0 . 1 4 \pm 0 . 1 0 }$ </td><td> $0 . 2 0 \pm 0 . 1 4$ </td><td> $1 . 0 3 \pm 0 . 4 0$ </td><td> $0 . 1 6 \pm 0 . 0 9$ </td></tr><tr><td>power</td><td> $0 . 1 8 \pm 0 . 0 5$ </td><td> ${ \bf 0 . 1 6 \pm 0 . 1 1 }$ </td><td> $0 . 1 9 \pm 0 . 0 8$ </td><td> $0 . 2 2 \pm 0 . 1 9$ </td><td> $0 . 1 7 \pm 0 . 1 2$ </td></tr><tr><td>protein</td><td> $1 . 7 5 \pm 0 . 1 5$ </td><td> $\mathbf { 0 . 0 6 \pm 0 . 0 5 }$ </td><td> $0 . 2 5 \pm 0 . 2 3$ </td><td> $0 . 0 7 \pm 0 . 0 3$ </td><td> $0 . 0 7 \pm 0 . 0 5$ </td></tr><tr><td>puma8NH</td><td> $0 . 2 7 \pm 0 . 1 1$ </td><td> $\mathbf { 0 . 1 0 \pm 0 . 0 4 }$ </td><td> $0 . 2 9 \pm 0 . 1 9$ </td><td> $0 . 1 5 \pm 0 . 1 1$ </td><td> $0 . 1 1 \pm 0 . 0 6$ </td></tr><tr><td>supercon</td><td> $0 . 5 7 \pm 0 . 1 6$ </td><td> $0 . 1 1 \pm 0 . 0 7$ </td><td> $0 . 2 1 \pm 0 . 1 3$ </td><td> $0 . 5 2 \pm 0 . 2 4$ </td><td> $\mathbf { 0 . 0 9 \pm 0 . 0 6 }$ </td></tr></table>

MC dropout
<table><tr><td>Dataset</td><td>Uncal.</td><td>Kuleshov et al. &#x27;18</td><td>Kuleshov-Deshpande &#x27;22</td><td> $\mathrm { O u r s } { \colon } \mathrm { O C }$ </td><td> $\mathrm { O u r s : ~ O C + Q R }$ </td></tr><tr><td>airfoil</td><td> $1 . 0 4 \pm 0 . 5 9$ </td><td> ${ \bf 0 . 6 2 \pm 0 . 4 6 }$ </td><td> $1 . 8 6 \pm 1 . 4 3$ </td><td> $1 . 7 2 \pm 0 . 6 6$ </td><td> $0 . 6 7 \pm 0 . 4 3$ </td></tr><tr><td>ames</td><td> $6 . 8 7 \pm 7 . 0 3$ </td><td> $0 . 4 1 \pm 0 . 2 9$ </td><td> $0 . 5 0 \pm 0 . 5 2$ </td><td> $3 . 4 1 \pm 2 . 1 7$ </td><td> ${ \bf 0 . 3 9 \pm 0 . 1 8 }$ </td></tr><tr><td>bank</td><td> $2 . 0 5 \pm 1 . 6 6$ </td><td> ${ \bf 0 . 2 0 \pm 0 . 2 0 }$ </td><td> $0 . 5 7 \pm 0 . 5 5$ </td><td> $0 . 3 4 \pm 0 . 2 9$ </td><td> $0 . 2 0 \pm 0 . 2 3$ </td></tr><tr><td>elevators</td><td> $0 . 9 4 \pm 0 . 5 5$ </td><td> $0 . 0 8 \pm 0 . 0 4$ </td><td> $0 . 1 5 \pm 0 . 1 2$ </td><td> $0 . 1 6 \pm 0 . 1 0$ </td><td> $\mathbf { 0 . 0 8 \pm 0 . 0 4 }$ </td></tr><tr><td>energy</td><td> $4 . 6 4 \pm 2 . 5 8$ </td><td> ${ \bf 1 . 2 9 \pm 0 . 7 8 }$ </td><td> $2 . 3 2 \pm 1 . 2 7$ </td><td> $2 . 3 2 \pm 1 . 0 9$ </td><td> $1 . 4 5 \pm 0 . 8 0$ </td></tr><tr><td>kinematics</td><td> $0 . 2 4 \pm 0 . 1 3$ </td><td> $\mathbf { 0 . 1 4 \pm 0 . 0 9 }$ </td><td> $0 . 2 4 \pm 0 . 1 8$ </td><td> $0 . 5 0 \pm 0 . 2 3$ </td><td> $0 . 1 4 \pm 0 . 0 9$ </td></tr><tr><td>power</td><td> $0 . 7 2 \pm 0 . 4 1$ </td><td> $\mathbf { 0 . 1 3 \pm 0 . 0 8 }$ </td><td> $0 . 2 4 \pm 0 . 1 0$ </td><td> $0 . 2 1 \pm 0 . 1 5$ </td><td> $0 . 1 4 \pm 0 . 0 6$ </td></tr><tr><td>protein</td><td> $1 . 8 2 \pm 0 . 6 4$ </td><td> $\mathbf { 0 . 0 6 \pm 0 . 0 5 }$ </td><td> $0 . 3 8 \pm 0 . 3 5$ </td><td> $0 . 3 5 \pm 0 . 1 8$ </td><td> $0 . 0 8 \pm 0 . 0 7$ </td></tr><tr><td>puma8NH</td><td> $0 . 3 3 \pm 0 . 1 8$ </td><td> ${ \bf 0 . 1 6 \pm 0 . 1 8 }$ </td><td> $0 . 3 0 \pm 0 . 3 2$ </td><td> $0 . 2 5 \pm 0 . 1 7$ </td><td> $0 . 1 6 \pm 0 . 1 7$ </td></tr><tr><td>supercon</td><td> $4 . 1 7 \pm 1 . 5 9$ </td><td> $0 . 0 8 \pm 0 . 0 6$ </td><td> $0 . 3 2 \pm 0 . 2 7$ </td><td> $0 . 8 3 \pm 0 . 1 7$ </td><td> $\mathbf { 0 . 0 7 \pm 0 . 0 4 }$ </td></tr></table>

## E EVALUATING ON PARTITIONS NOT USED FOR FITTING

In the results in the main paper the, evaluation of outcome-conditional calibration was always performed with respect to the partitions that were used for recalibration. In this Appendix, we evaluate to what extent outcomeconditional recalibration transfers across partitions. We assess how a method fitted on G partitions transfers when evaluated on finer partitions, and also assess performance with diferent combinations of partitions.

Table 6 contains the calibration results for diferent implementations of outcome-conditional recalibration. When recalibrating for a fixed number of partitions, e.g. $G = 5 ,$ the calibration error deteriorates when evaluated on a finer partition, though outcome-conditional recalibration still outperforms all baselines. In contrast, training using a finer partition of the outcome space also performs well on coarser partitions. When interest is on calibration for many diferent partitions of the outcome space, an iterated recalibration method performs well; this approach is described in the following subsection. However, this is not much more beneficial than the more parsimonious outcome-conditional recalibration approach covered in the main text.

Table 6: Evaluation of recalibration schemes when calibration is assessed on diferent partitions to those used for training the outcome-conditional recalibration scheme. The recalibration is trained using $G = 5 , 1 6 , 2 0$ partitions, all defined using evenly spaced quantiles of the outcome distribution. Calibration error is then calculated for the following partitions: $^ { 6 6 } \mathrm { C a l - a l l } ^ { \dag }$ is the standard (partition-free) calibration error; $G = 5 , 1 0 ,$ , 20 evenly spaced partitions; $G = 5$ shift sets quantile breaks at $[ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 ] ;$ ; union16 pairs adjacent atoms of the 16-cell partition; tail is a partition at quantile breakpoints 0.5, 0.9, 0.95 and 0.99 quantiles; rand averages three partitions with 4 break levels drawn uniformly per seed; mean averages the seven partition columns. Results are aggregated over all datasets with five random seeds, with Bayesian ridge as the base model. All calibration errors have been scaled by $1 0 ^ { 3 }$ to aid readability.
<table><tr><td>Method</td><td>Cal-all</td><td>G=5</td><td>union16</td><td>G=5 shift</td><td>G=10</td><td>G=20</td><td>tail</td><td>rand</td><td>mean</td></tr><tr><td>Uncalibrated</td><td>11.8</td><td>198.4</td><td>178.4</td><td>235.1</td><td>158.9</td><td>133.8</td><td>252.4</td><td>217.3</td><td>196.3</td></tr><tr><td>KFE18</td><td>2.2</td><td>177.7</td><td>163.9</td><td>218.0</td><td>147.5</td><td>124.5</td><td>225.9</td><td>195.5</td><td>179.0</td></tr><tr><td>KD22</td><td>4.9</td><td>20.7</td><td>37.8</td><td>36.6</td><td>44.7</td><td>71.7</td><td>118.7</td><td>48.0</td><td>54.0</td></tr><tr><td>Ours: OC  $G { = } 5$ </td><td>7.9</td><td>8.7</td><td>32.0</td><td>31.3</td><td>41.5</td><td>80.2</td><td>154.7</td><td>42.3</td><td>55.8</td></tr><tr><td>Ours: OC,  $G { = } 1 6$ </td><td>10.1</td><td>13.2</td><td>19.0</td><td>15.4</td><td>26.3</td><td>56.0</td><td>132.8</td><td>37.9</td><td>42.9</td></tr><tr><td>Ours: OC,  $G { = } 2 0$ </td><td>10.0</td><td>13.2</td><td>20.2</td><td>16.2</td><td>26.9</td><td>54.1</td><td>132.7</td><td>40.3</td><td>43.4</td></tr><tr><td>Ours:  $\operatorname { O C } + \operatorname { Q R } , G { = } 5$ </td><td>1.6</td><td>15.0</td><td>27.6</td><td>36.3</td><td>44.4</td><td>83.7</td><td>188.4</td><td>48.4</td><td>63.4</td></tr><tr><td>Ours:  $\mathrm { O C } + \mathrm { Q R } , G { = } 1 6$ </td><td>1.6</td><td>14.7</td><td>19.9</td><td>20.4</td><td>29.6</td><td>56.7</td><td>148.7</td><td>39.2</td><td>47.0</td></tr><tr><td>Ours:  $\mathrm { O C } + \mathrm { Q R } , G { = } 2 0$ </td><td>1.7</td><td>15.3</td><td>20.4</td><td>20.3</td><td>30.2</td><td>56.0</td><td>135.7</td><td>42.0</td><td>45.7</td></tr><tr><td>Iterated recalibration, 10 rounds</td><td>4.9</td><td>13.5</td><td>19.9</td><td>20.6</td><td>31.1</td><td>56.7</td><td>122.7</td><td>37.1</td><td>43.1</td></tr></table>

## E.1 Iterated recalibration

Any single application of outcome-conditional recalibration is in-sample calibrated on only one family of subsets (the family on which it is trained). Inspired by the multicalibration strategy from Hebert-Johnson et al. (2018), we now introduce a strategy to recalibrate predictive distributions on many subsets of the outcome space. This approach applies outcome-conditional recalibration iteratively on diferent partitions, and composes the resulting transformations.

Setup. Consider $n _ { b }$ break points $b _ { 0 } = - \infty < b _ { 1 } < \cdot \cdot \cdot < b _ { n _ { b } - 1 } = + \infty .$ possibly at equally spaced quantile levels of the outcome space, then define the set of all subsets defined by these breakpoints,

$$
\begin{array} { r } {  { \mathcal { S } } \ = \ \big \{ [ b _ { j } , b _ { k } ) : 0 \leq j < k \leq n _ { b } - 1 \big \} , \qquad |  { \mathcal { S } } | = { \binom { n _ { b } } { 2 } } . } \end{array}
$$

For example, $n _ { b } = 1 7$ yields $\vert { \cal S } \vert = 1 3 6$ subsets, including R itself.

Method. Outcome-conditional recalibration is applied in rounds. In each round, it finds the subset in S with the largest calibration error. It then recalibrates this subset, whilst leaving the predictive distribution on the complement of this subset untouched (up to a rescaling to ensure a valid distribution function). This is then repeated for R rounds, after which the resulting predictive distribution is a composition of the one-subset outcome-conditional recalibrations. The final transformation G produces the iterated recalibrated prediction $\widetilde { F } _ { i } ( y ) = G _ { i } ( F _ { i } ( y ) )$ . Each round of the algorithm is described in more detail below:

For rounds $1 , \ldots , R ,$

1. Find the subset $S = [ b _ { j } , b _ { k } ) \in \mathcal { S }$ with the largest subset calibration error (calibration error over $1 . 6 5 / n _ { S } )$

2. Compute current break point probabilities: $A _ { i } = G _ { i } ( F _ { i } ( b _ { j } ) ) , B _ { i } = G _ { i } ( F _ { i } ( b _ { k } ) ) , \Delta _ { i } = B _ { i } - A _ { i }$ as well as CPIT-values: $v _ { i } = ( G _ { i } ( F _ { i } ( y _ { i } ) ) - A _ { i } ) / \Delta$ for the observations $y _ { i } ~ \in ~ S$ . For round $1 , \ G _ { i }$ is the identity function.

3. Apply outcome-conditional recalibration on $S = [ b _ { j } , b _ { k } )$

• Shape: fit $T ^ { S }$ using isotonic regression of $\{ v _ { i } \}$ on its own normalised ranks (see Appendix B).

• Mass: writing $c = \log ( c ^ { S } / c ^ { S ^ { c } } )$ , the recalibrated mass assigned to $S$ simplifies to,

$$
m _ { i } = \frac { c ^ { S } \Delta _ { i } } { c ^ { S } \Delta _ { i } + c ^ { S ^ { c } } ( 1 - \Delta _ { i } ) } = \mathrm { l o g i t } ^ { - 1 } \big ( \mathrm { l o g i t } ( \Delta _ { i } ) + c \big ) .
$$

The recalibration equation $\begin{array} { r } { \sum _ { i } \mathrm { l o g i t } ^ { - 1 } ( \mathrm { l o g i t } ( \Delta _ { i } ) + c ) / N = q _ { S } } \end{array}$ can be solved by bisection.

4. Update $G _ { i }$ as:

$$
\begin{array} { r } { G _ { i } ( u ) \gets \left\{ \begin{array} { l l } { s _ { i } G _ { i } ( u ) , } & { u \leq F _ { i } ( b _ { j } ) , } \\ { A _ { i } ^ { \prime } + m _ { i } T ^ { S } \big ( ( G _ { i } ( u ) - A _ { i } ) / \Delta _ { i } \big ) , } & { F _ { i } ( b _ { j } ) < u < F _ { i } ( b _ { k } ) , } \\ { 1 - s _ { i } \big ( 1 - G _ { i } ( u ) \big ) , } & { u \geq F _ { i } ( b _ { k } ) , } \end{array} \right. } \end{array}
$$

where $s _ { i } = ( 1 - m _ { i } ) / ( 1 - \Delta _ { i } )$ and $A _ { i } ^ { \prime } = s _ { i } A _ { i }$ . This is (6) for the partition $\{ S , S ^ { c } \}$ with $T ^ { S ^ { c } } = \operatorname { i d }$

We find that the number of rounds is usually of minor importance, and running for a fixed 10 rounds yields good performance in the application above. As single-step outcome-conditional recalibration is extremely fast, this is very feasible computationally. It is also possible to perform early stopping on a held-out dataset, but we find that this does not increase performance much.