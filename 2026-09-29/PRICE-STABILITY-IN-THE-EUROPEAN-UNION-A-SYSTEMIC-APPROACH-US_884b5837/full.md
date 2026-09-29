# PRICE STABILITY IN THE EUROPEAN UNION:A SYSTEMIC APPROACH USING RANDOM MATRIX THEORY

TECHNICAL REPORT

Sami Diaf   
Department of Socioeconomics Universität Hamburg   
sami.diaf@uni-hamburg.de

## ABSTRACT

Price stability remains a pillar in monetary policy practices and carries a special importance within monetary unions. Mainstream economics tried to leverage price stability using price indices and several metrics to shed light on specific dynamics and optimal macroeconomic levels. The wide availability of data led researchers to consider the study of systems using Random Matrix Theory, based on inner correlation patterns. This aims to enhance the multivariate analysis by removing noisy patterns from the signal and improve data quality for further inferences. This work considers the collection of monthly inflation indices in the Eurozone as a system of prices to analyze its eigenvalues’ statistical and asymptotic properties and uncover inner country-level insights. Results confirm the system cannot assumed to be randomly generated, and the data exhibit noise-dominated patterns, due to small and persistent variations at the country-level. The latter make the inter-country correlations more dynamic and the separation of the signal from the noise quiet dificult. Findings identified two countries as distorting inflation dynamics besides three other distinct, regional-based groups of countries. Variability sources might stem from economic episodes fueling inflation spikes in some countries, as well as methodological aspects used to ensure data quality and representativeness in the European Union. Despite being complex, the system demonstrates a certain stability, in terms of self-organization; while large monthly fluctuations cannot be considered as rare events, but part of the data-generating process.

Keywords Inflation  Eurozone  Random Matrix Theory

## 1 Introduction

Price stability has been debated and extensively researched by economists, aiming at finding the optimal level of prices to implement monetary policies. In this context, stability refers to an optimal price level that depends on the policy horizons [1], which was widely thought to be the ultimate objective of monetary policy across several countries [2].

In the case of monetary unions, as for the European Union, price stability refers to the annual objective of keeping the Harmonized Index of Consumer Prices (HICP) within the Eurozone below 2%. The existence of an anchor, as a quantitative target, is believed to be a powerful instrument for anchoring inflation expectations and facilitating the conduct of monetary policy strategies by central banks [2].

The HICP is an aggregation of prices collected within the Eurozone by the Eurostat. It serves two main purposes [3]: quantifying the definition of price stability in the ECB monetary policy strategy and assessing price convergence for country candidates aiming at joining the monetary union.

Price collected at the Eurozone members are indeed heterogeneous and their patterns country-specific [3]. Attempts to harmonizing price data were implemented at the methodology-level during mid-1970s and considerable efort were put later on developing the theory and practice of consumer price indices (CPIs).

Forecasters explored many econometric and machine learning methods to enhance the HICP predictability and successful results were obtained with the use of Deep Neural Networks (DNNs) [4], confirming the dificulty encountered when using classic assumptions.

Country-level indices could be seen as a system of collected indices rendering an overview of price fluctuations within the Eurozone. This collection is indeed a system, where each country reports its index in a given period, yielding a matrix of data entries with a double dimension: country and time.

The study of systems has been first investigated in nuclear physics [5], where statistical properties of these system are essential to study their behavior. For this aim, many tools were derived from statistical mechanics, physics and probability to establish a dedicated brach of analysis: Random Matrix Theory (RMT) [6].

Generally, RMT deals with the behavior of eigenvalues, whose spectrum carries information about embedded patterns in large collections of data, or systems [7]. Eigenvalues are indeed random variables whose distribution depends solely on data properties. Classic applications considered the spectrum of eigenvalues and modeled it using Marčenko-Pastur distribution [8] which plots a bell-like curve over eigenvalues used to capture the signal-noise duality in the original dataset.

Further extensions of RMT were used to assess the goodness-of-fit of large systems, particularly DNNs [9, 10, 11], by deriving eigenvalue-based quality metrics on data quality and system accuracy. Such tools have been applicable on weight matrices of DNNs, as well as at the data-level, on the basis of an information exponent, �, that is empirically associated with optimal or ideal learning.

This works ofers a genuine, novel perspective on inflation analysis by considering Eurostat inflation data as a system, instead of focusing on country-based price index. The latter is generally the weighted product of many sub-indices, hiding heterogeneous information that could be useful in interpreting specific developments and identifying local variations. RMT gives deeper insight on the asymptotic behavior of eigenvalues and ofers extended analysis range, exceeding classic tools for multivariate time series whose scope mainly targets clustering and classification tasks [12].

Applied on the monthly Eurostat inflation variations of 27 country members from January 2021 to July 2026, this approach confirmed the efect of small variations in preventing a clear separation of the signal from the noise. The Power-law index estimated on the Marčenko-Pastur distribution, known as the information correlation, shows a slight overfitting behavior of the system. This validates the persistent nature of small variations within many Eurozone countries, acting as a multivariate attention mechanism dominating the system. Despite this persistent behavior, the system demonstrates slightly overfit nuances close to perfect fit models, and reinforces the hypothesis of a self-organization state with improved robustness against errors.

Persistent small variations in the system further indicate the presence of modular structures in the correlation matrix. This evidence may hint hidden variability and inner clustered properties in the data. The spacing eigenvalues’ distribution [13] succeeded in learning a cutof, under which correlations between country indices are assumed to be noisedriven. This denoising process identified two outliers and three distinct groups of countries with shared similarities in recent inflation developments. These findings suggest the use of regional-based indices to enhance the predictability of HICP, by taking into account a dynamic cluster formation of indices.

The remainder of the paper explains price stability from a monetary perspective in Section 2, then provides a theoretical framework to understand RMT tools (Section 3), before setting the application on Eurostat data in Section 4 and a following discussion of results and suggestions (Section 5).

## 2 Price Stability

The concept of price stability has a long history in modern macroeconomics and is still used as an anchor to guide strategies implemented by central banks and monetary committees around the world.

When conducting its monetary policy, a central bank has little control over short-term shocks afecting prices. It can devise a medium-term policy orientation to avoid excessive volatility in short-term interest rates or the real economy [1]. By doing this, the central bank shows a gradualist response to shocks that threaten price stability, which ofsets such excessive volatility, while maintaining the price stability condition over the medium term

[14] preferred the term neutral policy over price stability because it keeps output at its potential, with a constant markup of price over marginal cost. They recommended price stability as the core monetary strategy to ensure, among other conditions, the reduction of the output gap.

The Treaty on European Union gives the ECB the primary mandate of maintaining price stability over the medium run. Formulating the quantitative objective of price stability was left to the ECB’s Governing Council. The latter defined price stability, during at a Governing Council meeting held in October 1998, as “ the year-on-year increase in the Harmonised Index of Consumer Prices for the euro area of below $2 \%$

Particularly in the Eurozone, price stability was tested using two main criteria: the stability of short- to medium-term inflation expectations and the absence of long-term price-level uncertainty [15].

Even if in many countries price stability represents a primary goal for monetary policy, actual practices vary substantially across countries. They range from no explicit quantitative definition to explicit quantitative definitions and inflation targets (point targets or ranges for admissible inflation outcomes) [2].

As a benchmark, the HCIP is supported by a set of legally binding standards related to the index quality, in addition to the coordination eforts of the Eurostat to harmonize compilation practices [16] across national statistical institutes<sup>1</sup>. A Survey of Professional Forecasters was launched by the ECB in 1999 to help predicting main macroeconomic aggregates throughout the Euro area [17] and have an expert assessment on future inflation dynamics.

The HICP serves as the basis for many subsequent indices computed to help ECB monitoring inflation targeting, which difer in terms of methodology and data used [18]. For instance, [19] computed a domestic inflation index for monetary policy transmission mechanism, based on import intensities of HICP and further information from national accounts and input-output tables. Further attempts to predict HICP used a multivariate approach as well as DNNs to enhance the predictability [4], and confirmed issues due to the data heterogeneity.

## 3 Random Matrix Theory

Data analysis was first a context-based discipline, meaning the tools used to extract insights were relevant to the field of application. One can cite statistics as an earlier approach to gain information from data, which led to the emergence of cross-disciplinary fields as for econometrics.

The growing amount of data encouraged researchers to adopt advanced tools to maximize information extraction. While many practitioners still consider the idea of a model, as a formalism of a solution addressing a given problem, physicists consider the collection of data as a system. The latter could also be seen as a blend of all parameters pertaining to a model, or as a collection of all data entries related to a given task.

The behavior of a system, as for convergence and inner dependence, required advanced elements borrowed from statistical mechanics, physics and geometry. This gave birth to RMT as an inter-disciplinary field of research applying tools on diferent fields [6], ranging from nuclear physics to social sciences.

The statistical properties of a given system are retrieved by the analysis of its eigenvalues, as latent features bearing embedded information about data. This permits to study correlation dynamics and asymptotic properties for future directions as well as the duality signal-noise.

While most datasets have a rectangular, tabular structure; the correlation matrix is used to extract eigenvalues and analyze their variability spectrum via the Marčenko-Pastur distribution [8]. The latter describes, at the origin, the statistical properties of sample covariance matrices stemming from the Gaussian Orthogonal Ensemble [5].

Initially, a T N matrix W is assumed to have its elements $w _ { i j }$ drawn from a normal distribution ${ \cal N } ( 0 , \sigma ^ { 2 } )$ . The Wishart transform, or the covariance matrix, is computed and have negligible non-diagonal elements. In practice, we use the sample correlation matrix on normalized inputs $\tilde { W } ,$ , given by $\begin{array} { r } { { \bf \tilde { \cal X } } = \frac { 1 } { N } \tilde { W } ^ { \top } \tilde { \tilde { W } } } \end{array}$ as the starting point for extracting eigenvalues $\lambda _ { i } .$ The eigenvalues spectrum on the original � matrix has a probability density of Marčenko-Pastur (MP):

$$
\begin{array}{c} \begin{array} { r } { f ( \lambda ) = \left\{ \frac { N } { T } \frac { \sqrt { \left( \lambda _ { + } - \lambda \right) \left( \lambda - \lambda _ { - } \right) } } { 2 \pi \sigma ^ { 2 } } \quad \mathrm { i f } \lambda \in \left[ \lambda _ { - } , \lambda _ { + } \right] , \right.} \\ { 0 \quad \mathrm { i f } \lambda \notin \left[ \lambda _ { - } , \lambda _ { + } \right] . } \end{array}   \end{array}
$$

where $\begin{array} { r } { \lambda _ { - } = \sigma ^ { 2 } ( 1 - \sqrt { \frac { T } { N } } ) ^ { 2 } } \end{array}$ and $\begin{array} { r } { \lambda _ { + } = \sigma ^ { 2 } ( 1 + \sqrt { \frac { T } { N } } ) ^ { 2 } } \end{array}$

The MP distribution considers the spectrum of eigenvalues bounded between $\lambda _ { - }$ and $\lambda _ { + }$ as representing the noise randomness, while eigenvalues falling outside the interval $[ \lambda _ { - } , \lambda _ { + } ]$ are proxies of the signal. The MP paradigm is indeed a special case of the Wigner semicircle law [5], which stands for the asymptotic distribution of eigenvalues.

Because of embedded correlations in many systems, the X matrix is not strictly diagonal and the resulting MP dis tribution is somewhat deformed from its theoretical formula, due to data properties or learning schemes [11]. This deformation, known as heavy-tail, is an additional information on the dificulty to separate the signal from the noise in complex datasets.

[9] devised an empirical evidence, called Heavy-Tailed Self-Regularization, consisting on the use of a Power-law fit on the MP distribution to compute an information correlation index $\alpha ,$ as an empirical goodness-of-fit measure of DNNs without accessing training and test data. Later, further constraints on eigenvalues distribution were brought in the Semi-Empirical Theory of Learning (SETOL) [11]. This extended the self-regularization theory [20], which assumes the generic existence of a self-organized macroscopic state in any large multivariate system [21].

It was found [11] that nearly perfect-fit systems are related to $\alpha \sim 2$ , while values in the range of [2,6] are synonyms of underfitting issues. Overfitting occurs with values below 2, which indicate a heavy-tailed distribution of eigenvalues with infinite sample variance due to memory patterns.

Researchers are usually interested in cleaning big correlation matrices and clustering their elements on specific subgroups. Depending on data complexity, these tasks could be performed if the noise has Gaussian properties and its matrix adds up to a deterministic matrix to reconstruct the original data [22]. Such schemes rely on Truncated Singular Value Decomposition (TSVD), which are not optimal when the noise exhibits departures from the Gaussian orthogonality.

Inner clustered patterns, known as modular structures, emerge in complex datasets after eliminating noisy correlations, usually using a threshold to just keep relevant links. For this aim, the Nearest-Neighbor Spacing Distribution (NNSD) is used to study the probabilistic properties of ordered eigenvalues [13].

For a spacing distribution $s _ { i } = | \bar { \lambda } _ { i } - \bar { \lambda } _ { i - 1 } |$ of ordered eigenvalues ${ \bar { \lambda } } _ { i }$ , the probability distribution related to pure random matrices is the Gaussian Orthogonal Ensemble $\begin{array} { r } { P _ { G O E } ( s ) = \frac { \pi } { 2 } . s . e x p \bar { ( - \frac { \pi } { 4 } s ^ { 2 } ) } } \end{array}$ . The eigenvalues show a repulsive behavior, concentrated around a target far from $s = 0$

Modular structures exist when the entries of the original matrix are not random [13]. This indicates a block decomposition scheme following an exponential distribution $P _ { E x p } ( s ) = e x p ( - s )$ . The absence of repulsion in the exponential distribution helps finding clusters or subgroups in large networks and filter out noise-driven fluctuations.

Genuine clusters could be retrieved via a candidate signal-noise separating threshold on the correlation matrix. The log-likelihood distances of the NNSD to both limiting distributions are iteratively calculated. The threshold is the point where a regime change, or a clear departure, occurs between both distances previously computed [13].

RMT tools often rely on specific metrics built around the eigenvalues spectrum. Considering ordered eigenvalues from highest to lowest values of a matrix X: $\bar { \lambda } _ { 1 } > \bar { \lambda } _ { 2 } > . . . > \bar { \lambda } _ { n } > 0$ , we can compute the condition number $\begin{array} { r } { \kappa _ { E } ( X ) = \sqrt { \frac { \bar { \lambda } _ { 1 } } { \bar { \lambda } _ { n } } } } \end{array}$ to measure how sensitive a linear system is to small errors or noise [23]. For $\kappa \sim 1$ , the system is well-conditioned and low-sensitive, because its output remains stable when its input changes by a small amount. Alternatively, large values of � describes ill-conditioned systems with high sensitivity to noisy patterns.

[24] proposed an alternative measure $\begin{array} { r } { \kappa _ { D } ( X ) ~ = ~ \sqrt { \frac { \sum \bar { \lambda } _ { i } } { \bar { \lambda } _ { n } } } } \end{array}$ to test the dificulty of matrix inversion and to quantify the distance to the nearest ill-posed (singular) matrix. These measures may be used to test the ability of a system to amplify noisy-patterns and track their impact as new data entries are collected. Such iterative methods [25] help quantify the degree of perturbation new datapoints may bring to exacerbate the system, or alternatively how robust are new data entries to the stability of the system.

## 4 Application

Monthly inflation rate variations of 27 Eurozone countries were gathered from the Eurostat website <sup>2</sup>, covering the period January 2001-July 2026. Data of these countries constitute a system of prices, whose inner statistical properties will be investigated using RMT tools. Explicitly, the distribution of eigenvalues will be scrutinized to assess the signalnoise duality and the goodness-of-fit of the whole system. The interdependence of eigenvectors, through their spacing distribution, will be of a key importance to remove noise-related correlation which are not informative. This eases visualization and clustering of the countries, based on their truly informative links.

The system is a data matrix of 27 countries and 307 observations providing the per-country monthly evolution of inflation prices, as measured by the Consumption Price indices. No pre-processing steps or transformations on the system were conducted.

![](images/5dfa2c09111825fa57babef4357496e61fcb1f17b77ee81d559ef98f229f9638.jpg)  
Figure 1: Log-Log Empirical Spectral Density and the Power-law fit for Eurozone prices, yielding $\alpha = 1 . 8 2$ on the basis of $\lambda _ { - } = 3 7 . 2 1$

## 4.1 SETOL

The WeightWatcher tool<sup>3</sup> is used to apply the SETOL methodology [11]. The eigenvalues spectrum is extracted and the empirical spectral density (ESD) is plotted and fitted using a log-log regression to get the information correlation �.

Figure 1 shows the Power-law fit on the basis of estimated $\lambda _ { - } = 3 7 . 2 1$ and $\sigma = 0 . 2$ , rendering an information correlation $\alpha = 1 . 8 2$ with a Kolmogorov-Smirnov distance $D _ { K S } = 0 . 1 0$ computed vis-à-vis the theoretical Power-law. The value of � falls in the range of slighlty overfitted models, meaning many entries in the system are noise-dominated, mostly related to values below $\lambda _ { - }$ . The dispersion of the signal-related eigenvalues hints at potential inter-country correlations that hide inner clusters or modular structures. Knowing the limited size of the dataset, the aspect ratio $\begin{array} { r } { { \dot { Q } } = \frac { T } { N } } \end{array}$ of the original matrix � plays a significant role in the finite-size efect and the value of � [11].

The price system cannot be assumed to be pure random. Figure 2 displays a clear distinction between the empirical ESD from its random theoretical values. The system exhibits small departures in its eigenvalues distribution, called spikes, responsible of driving the value of � far from the ideal learning threshold $( \alpha \sim 2 )$ . The spikes, large eigenvalues in Figure 2 exceeding the $\lambda _ { + } ^ { r a n d }$ , reflect inflation fluctuations that depart from the sample. Such rare events cannot be assumed to be outliers, but are part of the system in what [21] called Dragon-kings. These meaningful outliers distort the PL-fit of � but indicate the system is self-organized [21], having inner mechanisms exhibiting stability after episodes of high inflation.

![](images/5c4a7c20452740ab7b9c3a418041566107ab16759357e302b0da19b62a3b2c07.jpg)  
Figure 2: Diference between the Log-Log Empirical Spectral Density of its theoretical (randomly generated) values (max rand $( \lambda _ { + } ^ { r a n d } )$ is the theoretical largest eigenvalue).

The distance between the ESD and the spikes is considered as a phase-transition, or a regime transition that bifurcates the system, due to the economic status and internal/external factors exacerbating inflation pressures in some countries. These spikes are linked to persistent significant fluctuations recorded in some countries in the system �, rather than spuriously large elements in the correlation matrix �, a mechanism known as correlation trap [11].

The information correlation index �, could be assumed to be a self-organization index [21] and is indeed a measure of self-similarity, similar to the fractal dimension [26, 27] indicating the roughness of the ESD. [11] considers it as a proxy of implicit regularization, in constrast to explicit regularization used in most machine learning and DNNs.

![](images/9c1cf024fa28327b5c22f7bb23d8306bb50fd8f9a67b186c962bc855fde9696a.jpg)  
Figure 3: Variations of $\alpha$ values following diferent thresholds of $\lambda _ { - }$ . The selected $\lambda _ { - } ~ = ~ 3 7 . 2 1$ achieves the lowest Kolmogorov-Smirnov Distance $( D _ { K S } )$ when computing the Power-law fit.

� is indeed a random variable, whose values depend on the determination of � . Figure 3 shows the spectrum of possible � values when diferent $\lambda _ { - }$ are selected. The optimal � is selected on the basis of the lowest $D _ { K S }$ between the log-log fit and the theoretical distribution. For values of � in the range of [100,160] the system could be assumed to have an eficient separation of the signal and noise, in other terms a perfect fit. However, values greater than 200 demonstrate underfitting properties, where the system is not able to eficiently learn from the noise and the signal. In other terms, sidelining more eigenvalues will shrink the available information, resulting in increased values of � toward the underfitting status.

In probabilistic terms, some country-specific elements of the original matrix � may have been drawn from a Power-law distribution rather than the normal distribution [11], which explains the heavy-tailed empirical MP distribution.

The value of � of the price system is slightly below the perfect fit condition, and hints at a potential violation of the Marčenko-Pastur hypothesis. In overfitted systems $( \alpha < 2 )$ one may consider the initial data having an infinite variance and linked to a Power-law distribution, rather than a normal distribution. The goal is to better represent non-linearities of the entries $W _ { i j }$ , which result in heavy-tailed distribution of the eigenvalues. [11] suggested a Pareto distribution with an index (shape parameter) $\mu ,$ following:

$$
w _ { i , j } ( \mu ) \sim \frac { C } { x ^ { \mu + 1 } } ~ ; ~ \rho ( \lambda ) \sim \lambda ^ { - ( a \mu + b ) }
$$

where $\begin{array} { r } { a = \frac { 1 } { 2 } } \end{array}$ and $b = 1$ for ideal conditions $( \alpha \sim 2 )$ and � a given constant. For a retrieved $\alpha = 1 . 8 2$ , one can assume the monthly inflation rates in the European Union as realizations of a Pareto distribution of $\mu = 1 . 6 4$ , confirming the non-linearities and the need of strong learners (DNNs) to predict HICP [4].

Considering the period 2001-2018, the condition number $\pmb { K } \pmb { E }$ is iteratively computed on monthly new data entries in the system W over the period 2019-2026. Figure 4 shows a downward trend in the condition number values, except the year 2022, which indicates the system vulnerability to noisy patterns. This period witnessed generalized price jumps, where annual inflation rates in the European Union rose to double-digit numbers [28]. Despite the relative limited size of the sample, the system exhibits lower values of $\pmb { K } \pmb { E }$ meaning more robustness against errors and a certain systemic stability<sup>4</sup>, as the overall sample correlation (2019-2026) with the HICP was found to be at a level of -0.824.

## 4.2 Nearest-neighbor spacing distribution

The country-based correlation matrix of the system, X, is used to determine a threshold separating the signal from the noise. This denoising process aims to remove hidden, non-informative inter-country links pertaining to the noise. This allows to keep signals that help diferentiate countries based on their inter-linkage and eventually, cluster their similarities in subgroups, or modular structures.

Figure 5 shows the empirical NNSD negative log-Likelihood distances to both Wigner and exponential distributions. The point, at which both distances change their behavior, is assumed to be the threshold used to clean the correlation matrix [13]. The cutof of 0.356, a relatively higher positive correlation, appears to fulfill this condition and is next used to clean the original correlation matrix of the system to leave out links deemed to represent the noise.

![](images/b4577318232e056995241e6f79da76199c7c5847e3663d6be6eda27ae2dbfccd.jpg)  
Figure 4: Plot of the iterative condition number during the period (2019-2026).

Such transformations permit to ease clustering attempts to uncover modular structures, by considering significant correlation patterns in the system. Similar countries are clustered, in terms of communities, based on a deterministic analysis of short random walks [29] called Walktrap [30] and leading eigenvector of the modularity matrix [31]. The latter is a spectral top-down approach, if compared to other bottom-up approaches as for Louvain algorithm [32].

The leading eigenvalue method (Figure 6) identified six distinct communities, based on their inner interactions throughout the study period. Romania and Bulgaria departs from other countries, having their own clusters, due to large inflation spikes in recent years. The third community clusters five east-European countries (Slovakia, Czechia, Hungary, Poland, Latvia and Estonia). Finland has its own community and lies between two other dense communities with geographical similarities.

The clustering ofers interesting insights on regional inflation variability, responsible of fueling noisy patterns in the system. The integration of Romania and Bulgaria might have created spikes in the eigenvalues’ distribution and moved away � from a perfect fit perspective. Based on the uncovered clusters, hidden patterns in the system cannot be imputed to two countries, but to other clusters of countries whose intermediary position might explain the dificulty of signal/noise separation and the higher cutof value of the NNSD.

Another factor that might have contributed to the clustering is the frequent update of the HICP basket for the sake of data quality. This hidden feature cannot be investigated at the country-level, but requires in-depth price collection and potential systemic analysis of the HICP divisions [16].

![](images/da56cdcb848a4dd644e496ecc9ca58645655a348aee32774878ca9b286ad6192.jpg)  
Figure 5: Plot of empirical negative log-likelihood distances to Wigner and Exponential distributions, following selected thresholds. The cutof, or turning point, appears at the value of 0.356 .

![](images/ddbb956d4932273e0ff2a4acf1c5e8932b750995ad4be39bfc98819a3043155b.jpg)  
Figure 6: Communities reconstructed from the denoised country correlation matrix using leading eigenvector community detection [31].

## 5 Discussion

The use of RMT tools ofered new perspectives in studying inflation dynamics in the European Union. Considering the whole system of monthly inflation rates of 27 countries, RMT identified latent non-stochastic patterns, confirming dificulties of predicting the HICP path over the medium run. The relatively small size of the sample does not permit to have robust generalizations, but showed strong evidences of small, but persistent cluster-driven deviations. Despite this hysteresis-efect, the system remains self-organized and assumes the spikes in the eigenvalues distribution as part of the sample, not as outliers.

Separating signal from the noise seems to be dificult in this particular dataset and needed a high positive threshold to filter out noise-related links, highlighting the non-stationary correlation dynamics in the system. The latter render modular structures facilitating a clustering of countries. The application of random walks community detection and further RMT clustering schemes confirmed the existence of regional communities with noticeable distinctions, based on recorded inflation spikes during last years.

The system is far from being random and the determination of stochasticity origins is quiet dificult, as it needs high-level data to identify more granular patterns. Several factors could have altered the system dynamics as for country-weight updates, cyclical inflationary trends and price methodologies performed to harmonize the collected data.

Findings confirm the dificulty to predict HICP, based on existing price collection. The application of machine learning algorithms for predictive modeling will definitely need regularization schemes and transformations, that handle persistent small fluctuations. This challenging task could be easily performed with the help of synthetic data, as for Tabular Prior-Data Fitted Networks (TabPFN) [33], at the expense of explainability and interpretability needed by economists.

Clustering the system with the leading eigenvalue method yielded six distinct groups, highlighting the need to construct regional sub-indices to better apprehend the HICP path over the medium- and long-term. Aside from classic network clustering whose applications were unable to uncover clusters, the non-linear nature of the data entries required advanced RMT tools to get granular insights.

Overall, the system has a self-organization feature similar to a mean-reverting behavior after the occurrence of large deviations from the sample distribution. The robustness against errors, as measured by the condition number, is another aspect of the system’s stability and resilience despite the limited number of observations.

## 6 Conclusion

This paper ofered a diferent way to analyze inflation at the European Union, by assuming inflation indices as components of a system rather than a multivariate dataset. This required advanced analytical tool to handle the cross-country price dynamics and inner correlations, seen as arbitrary structures with asymptotic properties. The system was found to be have persistent, small fluctuations that could be identified as noise-driven patterns similar to slight overfitted DNNs. This alters the separation of the signal from the noise, hence requiring solid inference techniques, especially for predictive exercises. The eigenvalues spectrum, as an informative feature of the system, exhibits heavy-tailed properties that distinguishes it from a pure random state. Overall, the system is dominated by small, memory-based deviations whose variability is group-based, rather than being country-related. The statistical properties of eigenvalues permits to cluster the system into six distinct groups of countries, indicating the need to compute sub-indices to better apprehend medium-term projections of the HICP and alleviate its variability. Despite non-linearities in the spectrum of eigenval ues, the system demonstrates a self-organizing state featuring large spikes that cannot be assumed to be outliers but part of the sample distribution. Self-organization is also a robustness that prevents the system from amplifying errors and reinforcing noisy-patterns. Further details on the HICP divisions will help getting more granular insights on inflation origins and potential cross-country trends, as persistency sources.

## References

[1] Frank Smets. Maintaining price stability: how long is the medium term? Journal of Monetary Economics, 50(6):1293–1309, 2003.

[2] Efrem Castelnuovo, Diego Rodriguez-Palenzuela, and Sergio Nicoletti Altimari. Definition of price stability, range and point inflation targets: the anchoring of long-term inflation expectations. Working Paper Series 273, European Central Bank, 2003.

[3] Eurostat. Harmonised Index ofConsumer Prices (HICP) Methodological Manual. European Commission, Luxembourg, 2024. Manuscript completed in January 2024 and corrected in September 2024.

[4] László Vancsura, Tibor Tatay, and Tibor Bareith. Enhancing policy insights: Machine learning-based forecasting of euro area inflation hicp and subcomponents. Forecasting, 7(4), 2025.

[5] Eugene P. Wigner. Characteristic vectors of bordered matrices with infinite dimensions. Annals ofMathematics, 62(3):548–564, 1955.

[6] Marc Potters and Jean-Philippe Bouchaud. A First Course in Random Matrix Theory: for Physicists, Engineers and Data Scientists. Cambridge University Press, 2020.

[7] Olivier Ledoit and Sandrine Péché. Eigenvectors of some large sample covariance matrix ensembles. Probability Theory and Related Fields, 151(1-2):233–264, 2011.

[8] V. A. Marchenko and L. A. Pastur. Distribution of eigenvalues for some sets of random matrices. Math. USSR-Sb, 1:422–437, 1967.

[9] Charles H. Martin, Tongsu Peng, and Michael W. Mahoney. Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data. Nature Communications, 12(1), jul 2021.

[10] Charles H. Martin and Michael W. Mahoney. Implicit self-regularization in deep neural networks: evidence from random matrix theory and implications for learning. Journal of Machine Learning Research, 22(1), jan 2021.

[11] Charles H Martin and Christopher Hinrichs. Setol: A semi-empirical theory of (deep) learning, 2025.

[12] Ángel López-Oriona and José A. Vilar. Machine learning for multivariate time series with the r package mlmts. Neurocomputing, 537:210–235, 2023.

[13] Uwe Menzel. RMThreshold: Signal-Noise Separation in Random Matrices by using Eigenvalue Spectrum Analysis, 2016. R package version 1.1.

[14] Marvin Goodfriend and Robert G. King. The case for price stability. NBER Working Paper Series, 1(8423), 2001.

[15] Christian Bordes and Laurent Clerc. Price stability and the ecb’s monetary policy strategy\*. Journal of Economic Surveys, 21(2):268–326, 2007.

[16] Eurostat. European Classification of Individual Consumption According to Purpose version 2 (ECOICOP ver. 2). Eurostat Metadata and Classifications, 2026. Accessed: 2026-09-27.

[17] Geof Kenny, Véronique Genre, Carlos Bowles, Roberta Friz, Aidan Meyler, and Tuomas Rautanen. The ecb survey of professional forecasters (spf)—a review after eight years’ experience. Occasional Paper Series 59, European Central Bank, Frankfurt am Main, 2007.

[18] Katalin Bodnár, Bruno Fagandini, Peter Healy, Christian Höynck, and Flavie Rousseau. Domestic and cyclical inflation in the euro area. Statistics Paper Series 54, European Central Bank, August 2026.

[19] Annika Fröhling, Daphne O’Brien, and Sebastian Schaefer. A new indicator of domestic inflation for the euro area. ECB Economic Bulletin, 4, 2022.

[20] Y. Malevergne and D. Sornette. Collective origin of the coexistence of apparent random matrix theory noise and of factors in large sample correlation matrices. Physica A: Statistical Mechanics and its Applications, 331(3):660– 668, 2004.

[21] Didier Sornette. Dragon-kings, black swans and the prediction of crises, 2009.

[22] Xiucai Ding. High dimensional deformed rectangular matrices with applications in matrix denoising. Bernoulli, 26(1):387–417, 2020.

[23] Alan Edelman. Eigenvalues and condition numbers of random matrices. SIAM Journal on Matrix Analysis and Applications, 9(4):543–560, 1988.

[24] James W. Demmel. The probability that a numerical analysis problem is dificult. Mathematics of Computation, 50(182):449–480, 1988.

[25] Hermann Weyl. Das asymptotische verteilungsgesetz der eigenwerte linearer partieller diferentialgleichungen (mit einer anwendung auf die theorie der hohlraumstrahlung). Mathematische Annalen, 71(4):441–479, 1912.

[26] Benoît B. Mandelbrot. Les Objets Fractals: Forme, Hasard et Dimension. Flammarion, Paris, 1975.

[27] Benoît B. Mandelbrot. The fractal geometry of nature. W. H. Freeman and Comp., New York, 1982.

[28] Vicente Ferreira, Alexandre Abreu, and Francisco Louçã. The rise and fall of inflation in the euro area (2021- 2024): A heterodox perspective. Structural Change and Economic Dynamics, 72:103–110, 2025.

[29] David Harel and Yehuda Koren. On clustering using random walks. In Proceedings of the 21st Conference on Foundations of Software Technology and Theoretical Computer Science, FST TCS ’01, page 18–41, Berlin, Heidelberg, 2001. Springer-Verlag.

[30] Pascal Pons and Matthieu Latapy. Computing communities in large networks using random walks (long version), 2005.

[31] Mark E. J. Newman. Finding community structure in networks using the eigenvectors of matrices. Physical Review E, 74(3):036104, 2006.

[32] Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment, 2008(10):P10008, oct 2008.

[33] Noah Hollmann, Samuel Müller, Katharina Eggensperger, and Frank Hutter. Tabpfn: A transformer that solves small tabular classification problems in a second, 2023.