# SynForecast: Controlled Synthetic Time Series for Forecasting Research

Olivier Sprangers<sup>a,∗</sup>, Cristian Challu<sup>a</sup>, Jan Rathfelder<sup>a</sup>, Han Wang<sup>a</sup>, Max Mergenthaler Canseco<sup>a</sup>

<sup>a</sup>Nixtla, San Francisco, CA, United States

## Abstract

Synthetic time series constitute a substantial share of the training corpora of several recent forecasting foundation models. Existing studies report benefits at small data budgets and modest synthetic proportions, but each evaluates individua combinations of model and generator. Studying these benefits systematically requires generators that support controlled configuration, perturbation, reproducibility and mechanism-level ablation. We review 11 existing generators and find gaps in their combined support for these capabilities.

In this paper, we present SynForecast, an open-source Python library that generates synthetic time series from 32 generators, 19 of which implement named mechanisms with a published source in econometrics, quantitative finance, statistical physics and point-process theory. Its design supports controlled experiments: perturbations and families of mechanisms can be varied while everything else is held fixed, output comes in the long format that forecasting libraries accept, and output does not depend on the number of parallel workers.

We evaluate SynForecast on M4 at five frequencies, using a neural network and a gradient-boosted ensemble. We find that a classical baseline attains the lowest error in nine out of ten frequency-history combinations. Among the learned models, a synthetic prior outperforms the same network trained from scratch at a fixed budget by up to 34% when only a few dozen observations are available. However, it loses to the network trained from scratch at full history at every frequency, and none of the three rescalings of history we pre-specified predicts where the sign changes. At the. hourly frequency, where the prior fails at every history, we find that the synthetic corpus lacks daily cycles, and adding them reduces zero-shot error by 43 to 63% for both models. Fine-tuning the pretrained model outperforms training from<sup>[</sup> scratch by up to about 15%, but at short histories a fixed 300-step fine-tune is worse than the pretrained model used as is. Finally, removing the classical family of mechanisms costs about 2% for both models, and every other family has a smaller efect outside the hourly frequency.

Keywords: synthetic data, time series generation, pretraining, open-source forecasting software, reproducibility, eficiency in forecasting methods

## 1. Introduction

Synthetic data play an increasingly important role in training time series foundation models. Chronos augments public datasets with series sampled from Gaussian-process priors (Ansari et al., 2024), while TimesFM assigns synthetic series a 20% sampling weight during pretraining<sup>r</sup> (Das et al., 2024). Chronos-2 relies entirely on synthetic data for its multivariate and covariate-informed training tasks (Ansari et al., 2025). Other models use synthetic data as their sole source of pretraining examples: ForecastPFN and TempoPFN are trained exclusively on generated series (Dooley et al., 2023; Moroshan et al., 2026), and Oreshkin et al. (2026) train neural forecasting models on approximately one billion simulated series, reporting zero-shot accuracy on public benchmarks that exceeds

AutoARIMA and several foundation models. These developments motivate studying how the composition and properties of synthetic training data afect forecasting performance.

Existing studies suggest that the benefits of synthetic training data depend on the forecasting setting. ForecastPFN achieves the most MSE wins among the evaluated methods when competing models receive training budgets of 50–250 observations (Dooley et al., 2023). In the Chronos-T5 Small ablation, the most consistent improvements occur with approximately 10% synthetic data (Ansari et al., 2024), while TimesFM benefits at underrepresented granularities but shows little change on the hourly ETT datasets (Das et al., 2024). Other studies report less favorable results. Cazaux et al. (2026) find that synthetic augmentation increases error in 67% of their full-data augmentation trials across five architectures, with substantial diferences between architectures. Kuvshinova et al. (2024) show that supervised learning from limited real observations can outperform zero-shot forecasting with models trained on larger synthetic corpora. However, these findings concern diferent combinations of generating processes, forecasting models and training protocols. Hence, it remains unclear when synthetic pretraining improves performance on a new dataset, in particular as the available history changes.

Controlled simulation is a natural way to investigate these dependencies, and has an established role in forecasting research (Kang et al., 2017; Hewamalage et al., 2022). Answering when synthetic pretraining helps requires experiments that vary one factor at a time, such as the generating mechanisms, a perturbation or the forecasting model, while holding the others fixed. However, existing generators make such experiments dificult in four ways. First, several return only bare arrays without frequencies, time origins or series identifiers, such as TimeSynth<sup>1</sup>, so researchers must reconstruct forecasting panels by hand, where alignment errors can go unnoticed. Second, where perturbations are available, their isolation from the base process is not guaranteed by design: whether the underlying series stays the same when an anomaly or changepoint is enabled depends on draw order or a shared random stream, so a stress test can confound the perturbation with a diferent base trajectory. Third, determinism under parallel generation is not guaranteed: most generators either generate in a single process, which limits scale, or give no guarantee that output is unchanged across worker counts. Fourth, most pretraining generators were built for a single study and implement a single generating family, so the contribution of a mechanism cannot be measured by removing it from a corpus.

We therefore identify four software requirements, one for each of these gaps: (i) long-format panel output with explicit series identifiers, time origins and frequencies; (ii) perturbations that can be varied while the underlying generating process remains fixed; (iii) identical output for a fixed seed and configuration regardless of the number of parallel workers; and (iv) diverse generating mechanisms organized into named families that can be selectively excluded. Of the 11 existing generators we examined, none fully satisfies more than two of these requirements (Appendix A).

In this paper, we present SynForecast, an open-source Python library that implements these four requirements within a common interface for synthetic time series generation. We use it to examine how the benefits of synthetic pretraining vary with history length, forecasting model, and the composition of the synthetic corpus.

Contributions. We make two main contributions:

1. We introduce SynForecast, a software framework for controlled synthetic time series generation (Section 3). It combines long-format panel output, perturbations applied separately from the generating processes, and randomness sampled before parallel dispatch, so that output is reproducible across worker counts. Its 32 generators include 19 mechanisms drawn from published work, organized into named families that support systematic ablation.

2. We study synthetic pretraining with two small forecasting models on five M4 frequencies (Section 4). We show that a synthetic prior outperforms training from scratch by up to 34% at short histories but not at full history, and we test whether the number of observations, the number of seasonal cycles or the ratio of history length to forecast horizon explains where this transition occurs. We also measure the contribution of individual generator families across models, trace the failure of the prior at the hourly frequency to missing daily cycles, and use controlled perturbations to assess the efect of structural breaks on three forecasting methods.

Scope. We use small, open models so that the experiments can be reproduced on modest hardware. Hence, our findings characterize the tested models and configurations, and their applicability to large forecasting foundation models remains to be evaluated. SynForecast also supports data augmentation (Section 3.6), but we leave its evaluation, including a matched comparison of bootstrap and parametric augmentation, to future work.

SynForecast is released under the Apache 2.0 license at https://github.com/Nixtla/synforecast, with documentation at https://nixtlaverse.nixtla.io/ synforecast/.

## 2. Background and related work

We review three lines of work that generate synthetic time series: for pretraining foundation models, for benchmarking forecasting methods, and for augmenting observed series. For each line of work, we position SynForecast relative to existing approaches. We refer the interested reader to the survey of Liu et al. (2025) for a broader overview of synthetic data for time series foundation models, and to Petropoulos et al. (2022) for the wider forecasting literature. Our focus is on generation from explicitly specified processes. Learned generative models, including difusion models (Yuan and Qiao, 2024) and the methods evaluated in TSGBench (Ang et al., 2023), instead estimate generative distributions from observed data. These methods can support several downstream applications, but a detailed review is outside the scope of this paper.

## 2.1. Synthetic data for pretraining

Synthetic generation provides a scalable way to expand the volume and diversity of forecasting training corpora. Existing approaches difer in their generating assumptions and how synthetic series are incorporated into training.

Chronos samples from Gaussian-process priors over composed kernels (Ansari et al., 2024); TimesFM mixes about three million series built from ARMA processes, seasonal patterns, trends and step functions into a real corpus that includes M4 (Das et al., 2024); and ForecastPFN (Dooley et al., 2023), Mamba4Cast (Bhethanabhotla et al., 2024) and TempoPFN train on synthetic data alone, the last from a released pipeline that includes stochastic diferential equations, Gaussian processes and audio synthesis (Moroshan et al., 2026). SarSim0 (Oreshkin et al., 2026) superposes stable SARIMA trajectories into multiseasonal traces, adds rate-based heavy-tailed noise and supports training on about one billion series generated on the fly. On the public GiftEval benchmark, models pretrained on SarSim0 data improve on AutoARIMA and recent foundation models (Oreshkin et al., 2026), and TempoPFN surpasses most models trained on real data (Moroshan et al., 2026).

Three features of this literature matter for what follows. First, generators difer in breadth: KernelSynth and SarSim0 build primarily on Gaussian processes and SARIMA models, respectively, whereas TimesFM combines four types of process and the released TempoPFN code includes 14 named generators, which its paper groups into ten types. Second, several studies examine the contributions of individual generators or components. TempoPFN reports leave-one-generator-out ablations and finds that excluding its stochastic-diferential-equation generator raises normalized CRPS on GIFT-Eval by about 26% (Moroshan et al., 2026). The SarSim0 study compares training on SarSim0, KernelSynth and ForecastPFN data across the same forecasting models, and separately ablates components of the SarSim0 pipeline (Oreshkin et al., 2026). Third, these pipelines were developed for particular pretraining studies, whereas SynForecast packages generation in a standalone library for reuse across forecasting experiments.

## 2.2. Generation for benchmarking and evaluation

Synthetic generation also supports forecasting evaluation by providing data with controlled characteristics. Forecasting libraries such as GluonTS (Alexandrov et al., 2020) and Darts (Herzen et al., 2022) include utilities for generating artificial series, in their dataset.artificial and utils.timeseries\_generation modules, respectively. More extensive simulation studies investigate how forecasting performance depends on the generating process. Hewamalage et al. (2022), for example, simulate autoregressive and seasonal autoregressive processes, SE-TAR models, logistic maps, and Mackey–Glass systems while varying series length, the number of series, heterogeneity, and model complexity to compare global and local forecasting methods. This empirical work complements theoretical results on the expressiveness and complexity of global forecasting models (Montero-Manso and Hyndman, 2021).

GRATIS (Kang et al., 2020) is a particularly relevant approach to reusable synthetic generation. It uses mixture autoregressive (MAR) models (Wong and Li, 2000), which can capture heavy tails and heteroskedasticity, to generate series with diverse properties, including non-stationarity and multiple seasonalities. Its evaluation demonstrates broad coverage of benchmark datasets, including M4, in a projected feature space. GRATIS also supports generation with target characteristics by using a genetic algorithm to optimize MAR parameters, extending earlier work that directly evolved time series observations (Kang et al., 2017).

SynForecast includes a GRATIS-inspired MAR generator, adapted rather than reproduced: it samples stationary autoregressive components, standardizes series by default, and targets four summary features. The two approaches difer in how a corpus is specified, not in how much of the feature space it can reach. GRATIS targets diversity and features within the MAR family, whereas SynForecast exposes named mechanisms whose contributions can be measured by changing the composition of the corpus (Section 4.6). Mechanism-based specification and feature-based evaluation are complementary, and we use both in Section 4.5.

Separating a base process from perturbations also has precedents, for example in GutenTAG, which configures anomalies separately from base oscillations (Wenig et al., 2022). SynForecast adopts this separation so that perturbations can vary while the base series stays fixed (Section 3.4).

## 2.3. Augmentation of observed series

Augmentation supplements observed datasets with additional series, often obtained through resampling, transformations, or simulation from fitted models. An established application is bagging exponential smoothing models fitted to series reconstructed from a moving block bootstrap of the STL remainder of Box–Cox transformed series (Bergmeir et al., 2016). Petropoulos et al. (2018) attribute much of the improvement to accounting for uncertainty in model selection. For global forecasting models, Bandara et al. (2021) study augmentation through bootstrapping and dynamic time warping barycentric averaging (Petitjean et al., 2011), alongside transfer learning from GRATIS-generated series, and report accuracy gains in settings with limited training data.

Synthetic series also support model selection and pretraining. FFORMS augments the reference set used to train its model-selection classifier with series simulated from fitted models (Talagala et al., 2023). Chronos uses TSMixup to construct convex combinations of real series; an ablation with Chronos-T5 Small finds similar indomain accuracy and improved zero-shot accuracy (Ansari et al., 2024). However, these benefits vary across settings: Cazaux et al. (2026) report that augmentation degrades accuracy in 67% of their trials using the full real training set, with substantial variation across architectures.

SynForecast brings existing generating mechanisms and augmentation methods into a common interface for forecasting research. Its contribution is the software design that allows researchers to vary corpus composition and perturbations, reproduce the resulting data, and compare their efects across forecasting models.

## 3. Design

In this section, we first state the four requirements that guide the design of SynForecast. Next, we describe the generators, the output format, the perturbation overlays and the seeding scheme that makes generation deterministic under parallelism. We end the section with augmentation and throughput.

3.1. Four requirements for controlled forecasting experiments

Four requirements guide the design of SynForecast, one for each gap identified in Section 1: (i) forecastingready output, in which observations carry a frequency, a starting timestamp and a series identity (Wang et al., 2020); (ii) separable perturbations, so that a changepoint, an anomaly or a burst of missing values can vary while the base trajectory stays fixed; (iii) determinism under parallelism, so that a fixed configuration and seed produce the same corpus regardless of worker count or scheduling; and (iv) removable families of mechanisms. Comparing corpora with and without a family requires a defined protocol for corpus size, sampling weights and training budget, and measures the contribution of that family to a particular mixture.

Figure 1 illustrates how SynForecast addresses the first three requirements, which Sections 3.3 to 3.5 examine in turn. Section 3.2 describes the generator families that support the fourth. Augmentation introduces an additional concern when generators are fitted to observed data: preventing leakage requires fitting only on the training portion of each evaluation split. Section 3.6 discusses this protocol.

## 3.2. The generator surface

SynForecast exposes 32 generators organized into five groups:

• Statistical (5): random walk, seasonal, SARIMA, ETS and integer-valued autoregression.

• Stochastic (13): GARCH, Ornstein–Uhlenbeck, geometric Brownian motion, jump difusion, Poisson, cyclic, fractional Brownian motion, Hawkes, stochastic volatility, regime switching, chaotic systems, bounded processes and Lévy processes. These draw mainly on quantitative finance, statistical physics and point-process theory.

• Multivariate (3): copula, VAR and Gaussian process, with copula and VAR generators providing explicit dependence across series.

• Domain (7): intermittent demand, energy load, IoT sensors, vital signs, clickstream, daily active users and state-space models.

• Pretraining (4): TSI and TCM, used as univariate generators by Ansari et al. (2025); KernelSynth, following Ansari et al. (2024); and a GRATIS-inspired mixture-autoregressive generator (Section 2.2). These generators randomize process structure or component composition across series to produce diverse training corpora.

The first four groups contain 28 generators: 19 implement named mechanisms with published sources, four are textbook constructions and five are stylized domain simulators. The supplementary material describes the model or recipe implemented by each generator and documents its provenance.

All 32 generators share a configuration and generation interface through a common base class. The following example uses the default SARIMA parameters and a fixed random seed:

```python
from synforecast.generators import SARIMAGenerator
gen = SARIMAGenerator(
min_length=365, max_length=365, freq="D", seed=42
)
df = gen.generate(n_series=1_000)
```

The call generates 1,000 daily SARIMA series, each containing 365 observations. The parameters min\_length and max\_length specify inclusive length bounds; when they difer, lengths are sampled uniformly, allowing panels with unequal series lengths. The freq parameter accepts pandas ofset aliases, such as "D" for daily observations, or a positive integer specifying the spacing of an integer time index. Figure 2 illustrates three generators from each group and the overlays described in Section 3.4.

## 3.3. Output format

generate returns a long-format dataframe with columns unique\_id, ds and y, following the schema used across the Nixtlaverse forecasting libraries. The default pandas output can therefore be passed to StatsForecast, MLForecast and NeuralForecast without restructuring the data. Column names are configurable at construction to accommodate other project conventions. Outputs that are only bare arrays, such as those of TimeSynth and the TSGM simulators, and the KernelSynth script, which attaches only a fixed dummy start date (Appendix A), require additional code to construct a forecasting panel with timestamps, frequencies, series identifiers and consistent ordering. SynForecast centralizes this step in a shared implementation. gratis already supports keyed

![](images/b6282e03c853a5d96c2b19f68c8815f38c38c07da502b8cb157623f58c4bcb4b.jpg)

Figure 1: From configuration to panel. Before dispatch, a master random stream initialized with seed s samples series lengths n , generation seeds g<sub>i</sub>, and one seed per overlay and series. Workers generate base trajectories x<sub>i</sub> under mechanism configuration θ using their assigned generation seeds. Changepoints C, anomalies A and missingness M are then applied in sequence, each using its own seeded random stream. The resulting series are assembled into a long-format dataframe. Each stage is annotated with the design requirement it addresses.

/ statistical

![](images/0d3dedde22684700a64dc5733f07638f5c052ad5d9d5be988ca103a64efcf593.jpg)  
/ stochastic

![](images/8efa4e604d3df8525e41b2b9477eb731dbbfa7649847e7203e8a923f25b6e79a.jpg)

![](images/bb6a861edaee1464d4e5d9ea1d4aad33976073a407d068caa57ab9a07e6e94eb.jpg)  
/ domain

![](images/188dca28213f0cd48ad7f3adef08e4415b2cc547363bf4e9d610ab468061ae16.jpg)  
/ pretraining

![](images/309c3b1f646fb9026f6f1d5024b71445bdb7bcbb171dd78964cfb65808afee0b.jpg)

![](images/cb5d0770aeba3ac853b335e62f1a04481a06be29481ed36c986f509c40cb85a5.jpg)

![](images/736fc82ecb007aebe0a38de466e2667cf3b9be90413a18aa2316ea80bc4fb073.jpg)

![](images/d380130e21751ca3a5c7dd30036ff0f5e5df2fca4e42d212df9d5cba7f404f15.jpg)

![](images/eca9a62a3dbd45955d624726c5fc9a213675fac002c75400879e3ede83b0cb77.jpg)

![](images/6c21f94182fd507f9b245c73f1f4b4cdb98942c6b0e24bdd4890c6b07a285181.jpg)

![](images/c3b8f0f72ef412fc8259c387acc88133a70fb822a464ca43be1641a4893d940b.jpg)

![](images/c581f1445f6d06594a099c6a1b43a5dae4f7d370440bd5e74065d822bb85ecbb.jpg)

![](images/90bf47c3d4d7d9981ef6a9592ef24f28ee01864a06e9947359c468f00bccbde3.jpg)

![](images/2b2a1733b2448801cfbe92df5b923ff42aad1e9d0ea65b899f5bad1eec19ad21.jpg)

![](images/5747ccf901e0c6ef961f3e8869013ad4fd37b73d0c33122a9b10f922026309ae.jpg)  
/ overlays on one base trajectory (grey)

![](images/a9788476b730db0dcf7b6e9a01683505132e864c27a320935dcab087ad43c254.jpg)

![](images/f092372fc2807e7630e090f10b79c2aa225445f7e78066aa6399122555956149.jpg)

![](images/effc839e61b18a85d3a6ea5b56e2ca0ca107498320bd3aa417be7a30948b8a72.jpg)

![](images/1aebe73b2c80ca89ffe6920ab1609f9ce3b0078abf92f60b8ec00aed2a1f4af2.jpg)  
Figure 2: Example output for monthly series of length 144. Top: three generators from each of the five groups, using default mechanism parameters except for the seasonal settings of SARIMA, ETS and the Gaussian process. Jointly generated panels are illustrated with two series each. Bottom: a seasonal series with each overlay enabled separately, superimposed on the same base trajectory without overlays (gray). Dashed lines indicate changepoint positions, markers identify anomalies, and the gray base trajectory remains visible through the missing block.

temporal output in R; SynForecast adopts this approach for integration with Python forecasting libraries.

Schema compatibility does not replace data validation. Missing values may be intentional, as in sensor-failure simulations or missingness overlays, and their treatment depends on the downstream model and experimental protocol. Numerical divergence can also lead to non-finite values that require detection and handling. In our experiments, we remove series containing non-finite values before training. The common schema reduces the need for custom data preparation while leaving these validity checks explicit.

A shared output-construction path supports pandas, Polars, cuDF, Modin and PyArrow; the full test suite covers pandas and Polars, and the other three are smoketested. Users can select the representation that matches their downstream pipeline, subject to the formats accepted by the consuming library. Polars output is constructed without an intermediate pandas dataframe; selecting cuDF materializes the generated arrays on the GPU.

## 3.4. Perturbations as overlays

Changepoints, anomalies and missingness are configured through the base class and applied after generation. All 32 generators therefore expose the same perturbation interface. Exogenous-variable options are configured through the same base class and add covariates or metadata to the output. The following example combines the three perturbation types:

gen = SARIMAGenerator(   
min\_length=365, max\_length=365, freq="D", seed=42,   
changepoints=True, num\_changepoints=3,   
changepoint\_type="mixed",   
anomalies=True, anomaly\_fraction=0.02,   
anomaly\_types=["spike", "level\_shift"],   
missing\_data=True, missing\_pattern="block",   
missing\_block\_size=7,   
)

Each perturbation has explicit parameters. Changepoints support level, trend, variance and mixed changes, with a configurable count and optional relative locations and magnitudes. Missingness supports random, block and seasonal patterns, controlled by a rate and pattern-specific parameters such as block size or seasonal period. Anomalies have a configurable fraction, a selection of types, pertype magnitudes and a duration for temporary level shifts.

A separate option controls the innovation distribution in generators that use the shared noise sampler. The available distributions are normal, Student’s t, Laplace, uniform and skew-normal, allowing noise tails and asymmetry to vary without introducing a new generator class. Additive anomaly magnitudes are expressed in the units of the series; the default spike-magnitude parameter is ten units. Consequently, a fixed setting has diferent relative efects across series scales. Comparisons intended to hold relative perturbation strength fixed must adjust these magnitudes accordingly, as in Section 4.7.

The overlay implementation is shared across generator classes, avoiding separate implementations for each combination of mechanism and perturbation. For controlled comparisons, the essential property is that overlay settings do not afect the random stream used to generate the base trajectory. Initializing generators with the same base configuration and seed therefore preserves the underlying trajectory when perturbations are enabled or modified. This supports paired stress tests in which forecasting accuracy is compared on matched base data. Section 3.5 describes the seed allocation that ensures this separation.

## 3.5. Determinism under parallelism

Generation is parallel by default. The $\mathtt { n \_ j }$ obs argument controls the worker count, with all available logical cores used unless the runtime configuration specifies otherwise. Work is dispatched to a Rust batch implementation built on Rayon or to a threaded implementation for mechanisms without a batch path. Copula, VAR, IoT sensor, daily active users and TCM in multivariate mode use sequential generation routines: they ignore the worker count and follow a fixed draw order rather than the per-series scheme described below. The design requirement is that changing the worker count or scheduling order must not change the generated output.

For the shared parallel path, series lengths and seeds are allocated before dispatch. For a request of n series, a master random stream initialized with the user’s seed draws, in order, n series lengths, n generation seeds and three perturbation seeds per series, corresponding to changepoints, anomalies and missingness. All three perturbation seeds are allocated even when their overlays are disabled. Each series is then generated using its assigned streams, independently of which worker executes the task.

Formally, series i is

$$
y _ { i } = \left( M _ { m _ { i } } \circ A _ { a _ { i } } \circ C _ { c _ { i } } \right) \left( G _ { \theta } ( g _ { i } , n _ { i } ) \right) ,\tag{1}
$$

where $G _ { \theta }$ denotes the generating mechanism with configuration $\theta , \ n _ { i }$ is the series length, and $g _ { i }$ is its generation seed. The configured changepoint, anomaly and missingness operators are denoted by $C ,$ A and M, with seeds $c _ { i } ,$ $a _ { i }$ and $m _ { i } ,$ respectively. A disabled overlay is the identity. Because seed allocation does not depend on which overlays are enabled, enabling an overlay changes the corresponding operator while preserving the base trajectory and the remaining seed assignments.

Hence, for a newly initialized generator, the output is determined by the seed, n and the configuration within a fixed library version, execution path and numerical environment, and a subsequent call on the same instance continues the random stream. Two properties follow: output is invariant to the worker count, which the test suite checks by comparing one worker with four, and toggling a perturbation preserves the base trajectory and the seed assignments of the other overlays (Section 3.4).

Reproducibility also depends on the numerical environment. KernelSynth factorizes a covariance matrix using multithreaded linear algebra, so its output can change slightly with the BLAS thread count. We have not established a general bound across machines or numerical backends. The threaded and joint generators also use NumPy’s random generator, whose streams are not guaranteed to remain unchanged across releases. Reproducing a corpus therefore requires recording the SynForecast and NumPy versions, numerical backend and thread configuration, alongside a hash identifying the generated corpus.

The test suite checks both properties (Appendix E). Because cross-version output stability is not guaranteed, benchmarks intended to remain comparable must pin both SynForecast and NumPy; our experiments use SynForecast 0.2.0, and the hourly frequency follow-up of Section 4.5 uses a later commit.

The convenience pools seed each instance with the base seed plus a fixed ofset, which reaches about 2,000 in pretraining\_pool. To avoid reusing instance seeds across replicates, base seeds must therefore be separated by more than the largest ofset, as in our experiments.

## 3.6. Augmentation

The generators described so far sample unconditionally. Augmentation generates synthetic series conditional on observed data. For each observed series, SynForecast selects and fits one of nine generators with parameterfitting support, then emits synthetic series identified as {original\_id}\_aug\_{i}, keeping them distinguishable from the originals. Three non-parametric alternatives are also available: a moving block bootstrap of the remainder of a classical decomposition, adapted from the Box–Cox and STL procedure of Bergmeir et al. (2016), averaging under dynamic time warping (Petitjean et al., 2011), and mixup across series following TSMixup (Ansari et al., 2024). When fitting fails, the default behavior is to raise an error. An opt-in fallback generates a matched AR(1) series and reports every substitution, so a failed fit cannot silently change the generating mechanism.

Conditioning on observed data introduces a leakage risk. If augmentation uses observations from the evaluation window, information from that window can enter the training set. An interface can help enforce or audit this boundary by separating fitting from sampling, accepting an explicit cutof or recording provenance, but each approach still depends on the caller specifying the held-out data correctly. SynForecast keeps synthetic series identifiable, but an identifier cannot show that a series was generated from training data alone. Hence, the caller must separate training and evaluation data before augmentation. We leave the evaluation of augmentation to future work.

## 3.7. Throughput

We treat generation speed as a design consideration, since Oreshkin et al. (2026) have already demonstrated on-the-fly generation. Generation is implemented largely in Rust and exposed through PyO3. On a machine with 20 cores, the 54-instance pool peaks at 419 series per second with eight workers, 2.45 times the throughput of one worker. Cost is dominated by series length and by KernelSynth: a daily pretraining draw, with series of up to 9,919 observations, takes 391 seconds against 1.3 seconds for quarterly (Appendix D).

## 4. Results

In this section, we first describe the experimental protocol. Next, we compare the learned models with statistical baselines, evaluate fine-tuning, and study how the benefit of the prior depends on history length. We then analyze the failure of the prior at the hourly frequency and ablate the mechanism families. We end the section with an experiment that uses the overlays to measure the cost of structural breaks.

## 4.1. Protocol

We pretrain small models on synthetic data produced by SynForecast and evaluate them on held-out real observations. No real data enters pretraining, and no evaluation observation enters any fitting step. The decision rules, test families and efect measures were specified before the experiments ran. Appendix C compares each expectation with its outcome and documents all deviations from the plan, the most consequential of which was running the ablation on 20 replicates rather than the planned 10.

Data. We use M4 (Makridakis et al., 2020) at the hourly, daily, weekly, monthly and quarterly frequencies. Yearly data are excluded because they lack the seasonal period required by the analysis in Section 4.4. Each cell evaluates a subsample of 300 series selected using a digest of the frequency and replicate, so all cells within a given frequency and replicate evaluate the same series. Weekly and hourly contain only 359 and 414 series, respectively, so replicates overlap in most of their series. We use natural seasonal periods (hourly 24, daily 7, weekly 52, monthly 12, quarterly 4). Our seasonal MASE values at daily and weekly frequencies are therefore not directly comparable to published M4 results, which treat these frequencies as non-seasonal.

Pretraining corpus. We use the pretraining\_pool of SynForecast 0.2.0, comprising 54 generator instances from 19 classes. Every periodic instance is set to the target frequency’s seasonal period, which the harness verifies for every corpus. Each (frequency, replicate) corpus is generated once and shared by both models. We draw 26 series per instance; the control keeps the first 20 of each and discards those containing non-finite values or too few observations for a training window. One IoT-sensor instance produces non-finite values in almost every series, so 53 instances contribute to all but 7 of the 150 corpora, in which it keeps a single series. The three mixture-autoregressive instances are non-seasonal, and the synthetic length ranges follow those of the M4 training series at each frequency.

Models. NHITS (Challu et al., 2023) and LightGBM (Ke et al., 2017), which we run via NeuralForecast and MLForecast, respectively, represent diferent function classes: a neural sequence model and a tree ensemble over lag features. They share the corpus, per-series standardization, input window and evaluation series, so agreement between them provides replication across function classes within a common pipeline. Both models use an input window equal to the forecast horizon: 48 observations for hourly data, 14 for daily, 13 for weekly, 18 for monthly and 8 for quarterly, with NHITS using this as its input size and LightGBM using lags 1 to h. This window is short by the standards of deep models on M4 and, at weekly, shorter than one seasonal cycle.

Conditions. At each history level we evaluate from\_scratch, trained on the target series for 300 steps; zero\_shot, pretrained for 1,000 steps on synthetic data only; pretrain\_ft, the pretrained model trained for a further 300 steps on the target at a tenth of the learning rate; and Seasonal Naive and AutoETS (Hyndman et al., 2002), computed on the same split. The learned conditions train for fixed budgets, with no validation split, early stopping or tuning of the budget. Every fit records its realized number of training steps, and the harness rejects an NHITS fit that deviates from its budget. On ten replicates NHITS also reports from\_scratch\_long, trained for 1,000 steps (Section 4.4). LightGBM reports from\_scratch and zero\_shot only, because continuing to add trees to a booster fitted on synthetic data is not a comparable analog of fine-tuning.

Metric and comparisons. Accuracy is the mean absolute scaled error (Hyndman and Koehler, 2006) of each series, in the seasonal form used in M4 (Makridakis et al., 2020), which scales by the in-sample error of the seasonal naive forecast,

$$
\mathrm { M A S E } = \frac { \frac { 1 } { h } \sum _ { k = 1 } ^ { h } \left| y _ { T + k } - \hat { y } _ { T + k } \right| } { \frac { 1 } { T - m } \sum _ { t = m + 1 } ^ { T } \left| y _ { t } - y _ { t - m } \right| } ,\tag{2}
$$

in which T denotes the training length, h the horizon and m the seasonal period. Conditions are compared within a replicate r through the relative improvement of condition b over condition a,

$$
\Delta _ { a  b } ( r ) = 1 0 0 \times \frac { \overline { { \mathrm { M A S E } } } _ { a } ( r ) - \overline { { \mathrm { M A S E } } } _ { b } ( r ) } { \overline { { \mathrm { M A S E } } } _ { a } ( r ) } ,\tag{3}
$$

where MASE<sub>c</sub>(r) denotes the mean MASE across the series in a cell for condition c and replicate r, with a positive $\Delta$ indicating that condition b outperforms condition a. Inference is across replicates, at each history level separately, with Wilcoxon signed-rank tests (Wilcoxon, 1945) and Holm correction (Holm, 1979).

We run 30 replicates, each with its own series subsample, corpus and initializations; the ablation arms run on the first 20. Appendix B gives the remaining details.

## 4.2. Comparison with statistical baselines

We begin with the comparison that bounds every other result in this section. Table 1 reports mean seasonal MASE of the statistical baselines and the best learned condition at the shortest and the longest history of each frequency.

We find that a statistical method attains the lowest error in nine of the ten rows: AutoETS in seven and Seasonal Naive in two. The single exception, daily at full history, is won by NHITS trained from scratch by 0.055 MASE with a paired standard error of 0.032, which is not a clear win. At the hourly frequency, Seasonal Naive performs best at both history levels, and the NHITS zero-shot model has 2.4 to 5.4 times its MASE. However, NHITS trained from scratch for 1,000 steps instead of 300 approximately matches Seasonal Naive at full history (1.192 against 1.188 on the ten replicates that ran it), which suggests that the gap between training from scratch and Seasonal Naive is largely a training-budget efect. Hence, at the scale studied here (a neural network of 156K to 191K parameters and a gradient-boosted ensemble), a practitioner should prefer none of the learned conditions. Where we report below that a synthetic prior beats training from scratch, the claim compares two learned conditions and is not a recommendation to use either.

Two aspects of the comparison may favor the statistical baselines. AutoETS selects its model form from the data, while the learned models run at a fixed budget without tuning, and the synthetic M4 timestamps ofer no calendar information that a neural model could exploit. Hence, the margins in Table 1 reflect the evaluated configurations and do not isolate the efect of method class. At monthly and quarterly frequencies, AutoETS leads by 0.20 to 0.45 MASE with paired standard errors of at most 0.015, so the gaps are large relative to their sampling uncertainty. The smaller gaps at weekly and daily frequencies may be more sensitive to tuning.

## 4.3. Fine-tuning after synthetic pretraining

We examine whether fine-tuning a pretrained model improves forecasting accuracy compared with training from scratch on the target data or using the pretrained model without adaptation. Figure 3 shows both comparisons for each frequency and history level, alongside the comparison between the pretrained model and a model trained from scratch examined in Section 4.4. Table S4 in the supplementary material reports the corresponding statistical tests for the fine-tuning comparisons.

Fine-tuning against training from scratch. We find that fine-tuning reduces forecast error for at least some history levels at four of the five frequencies. The largest average reduction occurs at quarterly, at approximately 13%, with statistically significant diferences at all six history levels. At monthly, fine-tuning reduces error at every level. At daily and weekly, the improvement broadly diminishes as history length increases and turns negative at the longest histories. However, at the hourly frequency, fine-tuning increases error by 10 to 26% across all history levels. With daily cycles in the corpus, it reduces error at every level (Section 4.5). Overall, 28 of the 30 comparisons are statistically significant after correction for multiple testing.

Table 1: Mean seasonal MASE, control arm, over 30 replicates and 300 series, with the standard error across replicates in parentheses. “Best learned” is the lowest of from\_scratch, zero\_shot and pretrain\_ft for NHITS, or of the first two for LightGBM. Lower is better; the best value in each row is bold.
<table><tr><td colspan="2"></td><td colspan="2">Statistical</td><td colspan="2">Best learned</td></tr><tr><td>Frequency</td><td>History</td><td>Seas. Naive</td><td>AutoETS</td><td>NHITS</td><td>LightGBM</td></tr><tr><td>Hourly</td><td>96 obs</td><td>1.293 (0.004)</td><td>1.839 (0.006)</td><td>2.065 (0.021)</td><td>1.743 (0.010)</td></tr><tr><td rowspan="3">Daily</td><td>full</td><td>1.195 (0.004)</td><td>1.603 (0.006)</td><td>1.412 (0.012)</td><td>1.443 (0.006)</td></tr><tr><td>28 obs</td><td>1.490 (0.013)</td><td>1.229 (0.013)</td><td>1.418 (0.013)</td><td>1.390 (0.012)</td></tr><tr><td>full</td><td>1.441 (0.015)</td><td>1.205 (0.037)</td><td>1.150 (0.014)</td><td>1.332 (0.065)</td></tr><tr><td rowspan="2">Weekly</td><td>104 obs</td><td>1.151 (0.002)</td><td>0.723 (0.004)</td><td>0.777 (0.004)</td><td>0.793 (0.005)</td></tr><tr><td>full</td><td>1.218 (0.004)</td><td>0.572 (0.002)</td><td>0.581 (0.003)</td><td>0.616 (0.002)</td></tr><tr><td rowspan="2">Monthly</td><td>36 obs</td><td>1.393 (0.011)</td><td>1.152 (0.010)</td><td>1.496 (0.014)</td><td>1.410 (0.013)</td></tr><tr><td>full</td><td>1.260 (0.012)</td><td>0.953 (0.010)</td><td>1.178 (0.013)</td><td>1.149 (0.013)</td></tr><tr><td rowspan="2">Quarterly</td><td>16 obs</td><td>1.522 (0.013)</td><td>1.243 (0.014)</td><td>1.710 (0.013)</td><td>1.695 (0.013)</td></tr><tr><td>full</td><td>1.595 (0.013)</td><td>1.161 (0.010)</td><td>1.552 (0.012)</td><td>1.617 (0.015)</td></tr></table>

![](images/af0a2ce7cd7a9329ac467b1e72a0976eba94ef8dd2891d40bc25025f25bbd940.jpg)

<table><tr><td rowspan=1 colspan=6>PRETRAIN_FT VS FROM_SCRATCH</td></tr><tr><td rowspan=1 colspan=1>+15</td><td rowspan=1 colspan=1>+15</td><td rowspan=1 colspan=1>+15</td><td rowspan=1 colspan=1>+15</td><td rowspan=1 colspan=1>+14</td><td rowspan=1 colspan=1>+3</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>full</td></tr><tr><td rowspan=1 colspan=1>+7</td><td rowspan=1 colspan=1>+9</td><td rowspan=1 colspan=1>+6</td><td rowspan=1 colspan=1>+4</td><td rowspan=1 colspan=1>+3</td><td rowspan=1 colspan=1>+1</td></tr><tr><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>48</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>108</td><td rowspan=1 colspan=1>144</td><td rowspan=1 colspan=1>full</td></tr><tr><td rowspan=1 colspan=1>+4</td><td rowspan=1 colspan=1>+3</td><td rowspan=1 colspan=1>+2</td><td rowspan=1 colspan=1>+1</td><td rowspan=1 colspan=1>-2</td><td rowspan=1 colspan=1>-1</td></tr><tr><td rowspan=1 colspan=1>104</td><td rowspan=1 colspan=1>156</td><td rowspan=1 colspan=1>208</td><td rowspan=1 colspan=1>260</td><td rowspan=1 colspan=1>416</td><td rowspan=1 colspan=1>full</td></tr><tr><td rowspan=1 colspan=1>+4</td><td rowspan=1 colspan=1>+6</td><td rowspan=1 colspan=1>+4</td><td rowspan=1 colspan=1>+4</td><td rowspan=1 colspan=1>+3</td><td rowspan=1 colspan=1>-1</td></tr><tr><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>56</td><td rowspan=1 colspan=1>112</td><td rowspan=1 colspan=1>182</td><td rowspan=1 colspan=1>364</td><td rowspan=1 colspan=1>full</td></tr><tr><td rowspan=1 colspan=1>-10</td><td rowspan=1 colspan=1>-11</td><td rowspan=1 colspan=1>-13</td><td rowspan=1 colspan=1>-15</td><td rowspan=1 colspan=1>-25</td><td rowspan=1 colspan=1>-26</td></tr><tr><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>144</td><td rowspan=1 colspan=1>240</td><td rowspan=1 colspan=1>384</td><td rowspan=1 colspan=1>672</td><td rowspan=1 colspan=1>full</td></tr><tr><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=2>L2  L3</td><td rowspan=1 colspan=1>L4</td><td rowspan=1 colspan=1>L5</td><td rowspan=1 colspan=1>full</td></tr></table>

![](images/cb5ad0832bbc007bdf5672d15b7fbe5f4a7235b1cad4b914dbbf73e341795d7b.jpg)  
Figure 3: Relative improvement $\Delta _ { a  b } ~ ( \mathrm { E q . } ~ 3 , ~ \% )$ of the condition named first over the condition named second, NHITS, mean over 30 replicates, by frequency and history level; the small number in each cell is the history in observations; Table S2 in the supplementary material gives each value with its 95% interval. Bold values are significant after Holm correction within the 30 tests of a panel (Wilcoxon signed-rank test across replicates). Blue favors the first condition, orange the second; the color scale is clipped at ±40%.

Fine-tuning against the pretrained model. Compared with the pretrained model without adaptation, fine-tuning increases error by 26 to 45% at the shortest history for quarterly, monthly and daily data. At full history, it reduces error by 9 to 13% at all frequencies except hourly. These results indicate that a fixed budget of 300 finetuning steps, without early stopping, can substantially reduce the benefits of synthetic pretraining when only a few dozen observations per series are available. Selecting the budget on held-out validation data, with zero steps among the candidates, could mitigate this efect, although we did not evaluate this approach. At the hourly frequency, finetuning reduces error relative to the zero-shot model at every history level, which reflects the poor zero-shot performance discussed in Section 4.5.

Hence, no learned configuration dominates across frequencies and history lengths. At short histories, the pretrained model without adaptation performs best at the non-hourly frequencies. At full history, fine-tuning and training from scratch are equivalent within a 5% margin at these four frequencies (two one-sided tests, $p < 0 . 0 2 )$ although equivalence within a 2% margin is not established at daily and quarterly. The direction of the fine-tuning effect is consistent across replicate subsets. When the replicates are split into two groups of 15, the efect has the same sign in both groups for all 30 comparisons. We prespecified that the quarterly improvement would be treated as a robust result only if fine-tuning were also not significantly worse than the zero-shot model at any history level. This criterion is not met because fine-tuning is worse at short histories.

Appendix B describes how the statistical tests are constructed.

## 4.4. Efect of history length

We find that the benefit of the synthetic prior depends on history length (Figure 3, left). Compared with training from scratch, the prior reduces forecast error by 26 to 34% at the shortest histories for daily, monthly and quarterly data, with improvements in all 30 replicates, but increases error by 7 to 17% at full history. The direction of the efect reverses at four of the five frequencies for both models, although the initial advantage of the prior is negligible for LightGBM at weekly (+0.02%). At the hourly frequency, the prior increases error at every history level. Its error is 51% higher at 96 observations and 4.6 times that of training from scratch at full history. Section 4.5 traces this failure to the corpus, which lacks daily cycles. With these cycles, the NHITS prior changes sign at the hourly frequency as well.

We examine whether the crossover aligns across frequencies under diferent measures of history length.

Within a frequency, observation count and seasonal cycles difer only by a constant factor; comparing frequencies allows these measures to be distinguished. For each frequency f with an observed reversal, we locate the crossover $c _ { f }$ , defined as the realized history length at which $\Delta _ { \mathbf { f r o m } _ { \mathbf { \cdot } } }$ <sub>\_scratch→zero\_shot</sub> crosses zero. We express it in raw observations $c _ { f }$ , seasonal cycles $c _ { f } / m _ { f } ,$ and history relative to the forecast horizon $c _ { f } / h _ { f }$ , where $m _ { f }$ and $h _ { f }$ denote the seasonal period and forecast horizon, respectively.

Let $\mathrm { C V } _ { k }$ denote the coeficient of variation of normalization k across frequencies with an observed crossover, and let $\mathrm { C V } _ { ( 1 ) } \ \leq \ \mathrm { C V } _ { ( 2 ) }$ denote the two smallest values. Under our pre-specified rule, the normalization with the smallest coeficient of variation is considered to govern the crossover if

$$
\mathrm { C V } _ { ( 1 ) } < 0 . 2 5 \quad \mathrm { a n d } \quad \mathrm { C V } _ { ( 2 ) } \geq 2 \mathrm { C V } _ { ( 1 ) } .\tag{4}
$$

The rule and the interpretation of each possible outcome were specified before the experiments, using the nominal history axis. Appendix C reports that analysis, which yields the same conclusion as the analysis based on realized history.

We find that none of the three normalizations satisfies the pre-specified crossover criterion (Figure 4, Table 2). History relative to the horizon has the lowest dispersion, with coeficients of variation of 0.411 for NHITS and 0.371 for LightGBM, both above the threshold of 0.25. The corresponding ratios $\mathrm { C V _ { ( 2 ) } / C V _ { ( 1 ) } }$ are 1.19 and 1.27, also below the required value of $2 .$ Seasonal cycles produce the greatest dispersion, contrary to the original expectation motivated by the need for repeated cycles to identify seasonal patterns. The crossover occurs at 4.2 cycles for weekly data and 45.3 cycles for daily data. This ranking depends partly on weekly, where the 13-observation input window covers only a quarter of the 52-week cycle, so neither model has access to a seasonal lag. Excluding weekly yields similar dispersion for seasonal cycles and raw observations (NHITS CV 0.61 and 0.62, respectively). Because the forecast horizon equals the input-window length, normalization by the horizon is also normalization by the receptive field, so this comparison cannot distinguish their roles.

However, the crossover could also reflect an undertrained comparator under the 300-step budget. On the ten replicates that additionally trained from scratch for 1,000 steps (Table S5 in the supplementary material), the longer budget reduces error by 16 to 23% at every hourly history level from 144 observations onward. At the other frequencies, it increases error by up to 5.3% at most shorter histories, while diferences at full history remain within 2.2%. The longer budget therefore changes the balance between short- and long-history performance, shifting the crossovers to 83, 169, 236 and 328 observations without changing the conclusion of the normalization analysis. For the test used, ten replicates do not permit statistical significance after Holm correction across 30 comparisons, so we report efect estimates and their intervals.

![](images/f97cb9b43dae79f0c58b8b1c5aca5c2f27086be2c15751648b156912ff3ab432.jpg)

![](images/48e70d2e6bed4cbf511d514db7bc091c89ed5de706a728e905a9c194a3851cff.jpg)  
HISTORY / SEASONAL PERIOD

![](images/2bb139862c0a1c2f88e576f2682f13bfd810f40a6e2428e415da0413a9ba802d.jpg)  
Figure 4: Improvement of the synthetic prior (zero\_shot) over target-specific training (from\_scratch) against history, NHITS, under the three candidate normalizations. Bands are 95% intervals over 30 replicates; the dashed line is parity. The vertical scale is linear within ±25% and logarithmic beyond it, with the transition marked by a dotted rule: hourly reaches −359%, which on a linear scale compresses the four frequencies that carry the argument into an illegible band around parity. No point is clipped. All three panels are shown because none of them collapses the five frequencies onto a common curve: in (b), seasonal cycles, the crossings span 4.2 (weekly) to 45.3 (daily). Hourly is negative throughout.

Table 2: Crossover location by frequency and normalization, NHITS, in realized training observations, over 30 replicates. Hourly is excluded because the efect is negative at every level; that exclusion was the pre-specified handling for a frequency with no crossing. CV is the coeficient of variation across the four remaining frequencies, with a bootstrap 95% interval over replicates. Figure 4 is generated independently from the same cells and its interpolated crossings agree with this table to within 0.1 observations.
<table><tr><td>Frequency</td><td>Raw (obs)</td><td>Cycles</td><td>History/horizon</td></tr><tr><td>Quarterly</td><td>83.7</td><td>20.9</td><td>10.5</td></tr><tr><td>Monthly</td><td>171.9</td><td>14.3</td><td>9.5</td></tr><tr><td>Weekly</td><td>220.1</td><td>4.2</td><td>16.9</td></tr><tr><td>Daily</td><td>316.8</td><td>45.3</td><td>22.6</td></tr><tr><td>CV</td><td>0.491</td><td>0.824</td><td>0.411</td></tr><tr><td>95% CI</td><td>[0.459, 0.628]</td><td>[0.783, 0.975]</td><td>[0.375, 0.556]</td></tr></table>

Under the main protocol, the observed crossovers fall between 84 and 317 realized observations for NHITS and between 66 and 263 for LightGBM. This is broadly in line with ForecastPFN, which obtained the most MSE wins against other model classes at training budgets of 50–250 observations (Dooley et al., 2023), although that comparison used diferent comparators, metrics and datasets. The comparator is the same network trained from scratch at a fixed budget and, except at weekly, performs worse than Seasonal Naive at the shortest histories. At quarterly, monthly and daily, the NHITS prior’s error relative to Seasonal Naive remains approximately constant with history length, whereas at weekly it falls from 0.67 to 0.56 times that of Seasonal Naive (Table S6 in the supplementary material). The crossover occurs where the comparator’s learning curve intersects this level, and improved regularization of the comparator could shift its location. At the non-hourly frequencies, the synthetic prior generally performs better with a few dozen observations per series, while training from scratch performs better with several hundred. AutoETS nevertheless outperforms both at this scale (Section 4.2).

## 4.5. Failure of the prior at the hourly frequency

In the main experiment, hourly is the one frequency at which the synthetic prior never helps, and the failure grows with history: the zero-shot error relative to Seasonal Naive rises from 2.4 at 96 observations to 5.4 at full history, although the model only sees the last 48 observations. We tested three explanations. We tested the third, periodicity, in a follow-up run after the main experiment.

Normalization. NHITS inputs are standardized per series over the whole retained history, so a longer history rescales the 48 observations the model sees. When we rerun the hourly grid of the main experiment with each input window standardized instead, pretraining included, the ratio to Seasonal Naive no longer grows (2.1 at 96 observations, 1.9 at full history), and zero-shot error falls by 11 to 65%, significantly at every level. We ran this for NHITS only. LightGBM, which standardizes the same way, shows the same growth (from 3.1 to 6.4 times Seasonal Naive), but we did not rerun it.

Coverage. We placed each frequency’s evaluated series, the pretraining corpus and an equally sized sample of other M4 series in one feature space fitted on the evaluated series, using SynForecast’s coverage module, and counted the evaluated series in grid cells the comparison sample does not reach (Table 3). A size-matched sample of the corpus leaves 72% of the hourly series uncovered, against 24% for real hourly series. At the other frequencies, the gap is 7 to 26 points, and hourly has the largest gap at grid resolutions of 20, 30 and 40 bins per axis. However, the absolute figures are noisy and depend on sample size. Across replicates, the hourly fraction ranges from 0.60 to 0.85, and the full corpus of about 1,060 series, instead of the matched 114, leaves 47% uncovered. Measured instead by the share of all grid cells that real series occupy and the corpus misses, the hourly gap is not the largest.

Normalization explains why the failure grows with history, but not the failure itself: with window standardization, the prior is still worse than training from scratch at every hourly level, and about twice as bad as Seasonal Naive.

Periodicity. Real hourly series are dominated by a daily cycle. Every periodic instance in the hourly corpus has period 24, but few instances are periodic. Measured as the strength of the 24-step seasonality, 1 − Var(remainder)/Var(detrended) under a classical decomposition, the median over the series of a corpus is 0.03 on average across replicates, against 0.92 for the evaluated M4 series. We therefore reran the hourly grid on a second corpus that adds calendar cycles to the control corpus. SynForecast’s calendar\_overlays prior gives each pool instance, with probability 0.8, a time-of-day profile of one to five harmonics and, with probability 0.7, a day-of-week profile. The profile is either added, with a peak of one to eight standard deviations, or, on strictly positive series, multiplied in, with a swing of 20 to 90% of the level. Each instance keeps the series of the control corpus, so the two corpora difer only in the cycles, and the median seasonal strength rises to 0.60. Note that no model receives a calendar feature, so a model can learn that 24- and 168-step cycles occur, but not their phase. The follow-up uses all 30 replicates and six history levels, NHITS with both scalers, and LightGBM. Training from scratch and the baselines are shared with the control corpus of the same run, whose zero-shot error relative to Seasonal Naive is within 3% of the main experiment at every level.

We find that calendar cycles reduce zero-shot error by 43 to 62% for NHITS and by 52 to 63% for LightGBM, at every history level and in every replicate (Table 4). With cycles, the NHITS prior outperforms training from scratch by 16% at 96 observations and by 12% at 144, and loses by 70% at full history. The sign changes at 323 realized observations, just above the range of 84 to 317 observations at the other frequencies. In addition, fine-tuning the prior with cycles outperforms training from scratch at every level, by 9 to 21%. However, the prior with cycles remains 1.3 to 2.5 times worse than Seasonal Naive. For LightGBM, and for NHITS with the window scaler, it also still loses to training from scratch at every level, by 10 to 106% and by 9 to 24%, respectively.

To conclude, the lack of daily cycles in the corpus explains most of the hourly failure, and normalization explains its growth with history. With cycles, the NHITS ratio to Seasonal Naive still rises 1.5-fold from 96 observations to full history with the series scaler, and does not grow with the window scaler. The coverage results are consistent with this, as hourly is the frequency the corpus covers worst. However, the coverage comparison is descriptive, its hourly gap of 0.48 falls just below the prespecified threshold of 0.50, and we did not recompute it for the corpus with cycles. We specified the follow-up after the main experiment and after a five-replicate pilot (Appendix C), and ran it on a commit of the library that is not yet released, without the generic overlays that the library’s pretraining prior can add.

Table 3: Feature-space coverage of the evaluated M4 series, mean over replicates 0–2. “Uncovered” is the fraction of evaluated series lying in grid cells (30 × 30, PCA fitted on the evaluated series) that the comparison sample does not occupy. The comparison samples are the round’s pretraining corpus and other M4 series of the same frequency, subsampled to the same size, n; at weekly and hourly n is limited by the series M4 has outside the evaluated subsample. “Full” scores the whole control corpus instead of a sample of size n. At weekly, 54 to 56 of the 300 evaluated series have undefined features and are dropped.
<table><tr><td rowspan="2">Frequency</td><td colspan="4">Uncovered evaluated series</td><td rowspan="2">Full</td></tr><tr><td>n</td><td>Synthetic</td><td>Real</td><td>Gap</td></tr><tr><td>Quarterly</td><td>1000</td><td>0.27</td><td>0.10</td><td>+0.17</td><td>0.26</td></tr><tr><td>Monthly</td><td>1000</td><td>0.38</td><td>0.11</td><td>+0.26</td><td>0.35</td></tr><tr><td>Weekly</td><td>59</td><td>0.44</td><td>0.29</td><td>+0.14</td><td>0.09</td></tr><tr><td>Daily</td><td>1000</td><td>0.10</td><td>0.03</td><td>+0.07</td><td>0.10</td></tr><tr><td>Hourly</td><td>114</td><td>0.72</td><td>0.24</td><td>+0.48</td><td>0.47</td></tr></table>

Table 4: Hourly follow-up: the pretraining corpus without (control) and with calendar cycles, mean over 30 replicates. Top: zero-shot MASE relative to Seasonal Naive. Bottom: improvement ∆ (Eq. 3, %) of the zero-shot model over the same model trained from scratch; bold values are significant after Holm correction across the six history levels (Wilcoxon signed-rank test across replicates). Series and window name NHITS’s input standardization. The control rows, including those with the window scaler, come from the follow-up run, not from the main experiment, so they difer slightly from the mainexperiment values quoted in the text.
<table><tr><td rowspan="2"></td><td colspan="6">History (observations)</td></tr><tr><td>96</td><td>144</td><td>240</td><td>384</td><td>672</td><td>Full</td></tr><tr><td>Zero-shot MASE relative to Seasonal Naive</td><td colspan="6"></td></tr><tr><td>NHITS, series</td><td>2.34</td><td>2.44</td><td>2.72</td><td>3.28</td><td>4.52</td><td>5.38</td></tr><tr><td>with cycles NHITS, window</td><td>1.34</td><td>1.37</td><td>1.44</td><td>1.56</td><td>1.80</td><td>2.01</td></tr><tr><td></td><td>2.17</td><td>2.11</td><td>2.07</td><td>2.02</td><td>1.98</td><td>1.91</td></tr><tr><td>with cycles</td><td>1.51</td><td>1.48</td><td>1.46</td><td>1.44</td><td>1.42</td><td>1.39</td></tr><tr><td>LightGBM with cycles</td><td>3.17</td><td>3.19</td><td>3.46</td><td>4.25</td><td>5.83</td><td>6.37</td></tr><tr><td></td><td>1.49</td><td>1.51</td><td>1.61</td><td>1.79</td><td>2.15</td><td>2.49</td></tr><tr><td colspan="7">∆from_scratch→zero_shot (%) NHITS, series</td></tr><tr><td>with cycles</td><td>-47 +16</td><td>-57 +12</td><td>-79 +5</td><td>-119 -4</td><td>-245 -37</td><td>-357 -70</td></tr><tr><td>NHITS, window with cycles</td><td>-65 -15</td><td>-54 -9</td><td>-75 -24</td><td>-70 -22</td><td>-70 -23</td><td>-59 -16</td></tr><tr><td>LightGBM</td><td>-135</td><td>-146</td><td>-162</td><td>-196</td><td>-312</td><td>-428</td></tr><tr><td>with cycles</td><td>-10</td><td>-17</td><td>-22</td><td>-25</td><td>-52</td><td>-106</td></tr></table>

## 4.6. Ablation of mechanism families

In this section, we use the capability that distinguishes mechanism specification (Section 2.2): removing a named family. We define seven families by the instances in the pretraining pool, remove each family in turn while keeping the corpus size approximately constant, and measure the change in zero-shot accuracy. Every arm keeps a prefix of the same per-instance draws as the control, and draws more from each remaining instance to keep the corpus size. Hence, an arm difers from the control both in the family removed and in the weight of the remaining families, and an efect measures a family against a reweighted remainder.

The efect of removing family a is

$$
E _ { a } = \frac { 1 0 0 } { | \mathcal { C } | } \sum _ { ( f , l , r ) \in \mathcal { C } } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \log \mathrm { M A S E } _ { i } ^ { ( a ) } - \log \mathrm { M A S E } _ { i } ^ { ( 0 ) } \right) ,\tag{5}
$$

in which C denotes the cells (frequency f, history level l, replicate r) of a population and N the series of a cell, so that $E _ { a }$ is positive when removing the family hurts. Intervals come from a bootstrap over replicates. The primary population excludes hourly, where the zero-shot model fails, and an efect is said to replicate when both models’ intervals exclude zero with the same sign (Figure 5; Table F.7 in Appendix F).

We find that one family replicates on the pre-specified replicates and four on the run as executed, and that all efects are small. We had pre-specified the ablation on replicates 0–9 and ran it on 0–19, a change decided before launch because ten replicate clusters give under-covering intervals (Appendix C); Table F.7 reports both. Removing the classical family costs about 2% for both models on both replicate sets. Removing the Gaussian-process or the heavy-tail and regime family costs about 1%, and removing the remaining stochastic mechanisms – fractional Brownian motion, chaotic systems, bounded processes and cycles – improves accuracy by about 1%. These three effects are supported by both models in the analysis using 20 replicates. On the pre-specified subset of 10, the Light-GBM intervals exclude zero but the NHITS intervals do not, although the point estimates difer from those of the larger analysis by no more than 0.2 points. Each of the four efects keeps its sign at all four frequencies of the primary population for both models, with two exceptions for NHITS. The domain, procedural and MAR families do not replicate.

![](images/97fe4203335aaab113593f31ad2e882766d3b4539d49d1225a8a63c312ff8468.jpg)  
Figure 5: Efect of removing each mechanism family on zero-shot accuracy $( { \mathrm { E q . ~ 5 } } ) ,$ primary population (daily, weekly, monthly and quarterly; three history levels each; 20 replicates), with 95% intervals from a bootstrap over replicates. Positive: the family helps. Families: classical = SARIMA, ETS, INAR; Gaussian process = four GP instances; heavy-tail and regime = GARCH, Lévy, regime switching; other stochastic = fractional Brownian motion, chaotic, bounded, cyclic; domain = intermittent demand, energy load, IoT sensor, vital signs; procedural = TSI, TCM, KernelSynth; MAR = three non-seasonal mixture-autoregressive instances. Table F.7 also reports the pre-specified replicates 0–9.

We highlight three observations. First, mechanism families are not interchangeable: removing the classical family changes downstream accuracy in a direction on which both model classes agree. Second, the efects are modest, at most 5% at any non-hourly frequency, which suggests that corpus composition has a limited influence at this scale. By comparison, TempoPFN’s leave-one-generator-out ablation, with a model of about 40M parameters, found that removing a single generator raised normalized CRPS on GIFT-Eval by 2–14%, and by 26% for its stochastic-diferential-equation generator (Moroshan et al., 2026); the metric and the unit removed difer from ours. Third, the direction of an efect is not always what one would expect. The heavy-tail and regime family (GARCH and Lévy processes, and the Markov-switching autoregression) contributes, whereas the long-memory, chaotic and bounded processes do not. The sign of the heavy-tail efect reversed between this run and our earlier run on a smaller pool, so we consider it provisional. The library makes such an ablation possible outside the training code of a single model, behind one interface. At the hourly frequency, the efects are larger, up to 19%, and follow a diferent pattern, so we do not generalize from them.

![](images/77b7cbda6bc27e2bcf2e0f85fddb31baf90f5b675b83002fce10309cd561bdcf.jpg)  
Figure 6: Change in MASE caused by one level shift of three times the series’ median standard deviation, by its position relative to the forecast origin; monthly SARIMA panels, 5 replicates of 300 series, mean ± one standard error across replicates. The shaded position lies inside the 18-step forecast window.

## 4.7. Efect of structural breaks

Finally, we use the overlays of Section 3.4 for a controlled stress test. We generate monthly SARIMA panels of 300 series of 144 observations and, from the same seeds, the same panels with a single level shift of three times the series’ median standard deviation at a chosen position. Before the break, the perturbed series equal the unperturbed ones bit for bit, which the experiment asserts for every panel. Seasonal Naive, AutoETS and NHITS each forecast the last 18 observations of both versions, and Figure 6 shows the paired change in MASE, measured on the unperturbed series’ scale (values in Table S3 of the supplementary material).

We find that the three methods fail in diferent ways, each traceable to how it uses history. Seasonal Naive is unafected unless the break falls within its last season. AutoETS, which fits the whole history, pays a small cost for any break six or more observations back, and absorbs a break six observations back far better than the other methods. NHITS, which sees only the last 18 observations, is almost unafected by breaks outside that window and strongly afected by breaks inside it. These results are as expected. Because the design holds everything else fixed, the comparison isolates how each method uses history.

## 5. Discussion

Role of synthetic priors. A synthetic prior provides a plausible starting point when target data are scarce, and its value decreases as more target data become available. Our results add two practical refinements. First, finetuning is not a free improvement on the prior. With a few dozen observations per series, a fixed 300-step fine-tune without early stopping removes much of what the prior provides, so in that regime the pretrained model is better used as is, although a classical baseline is better still. Second, the prior helps only when observations per series are few, and practitioners with long histories gain nothing from it. Note that our short histories are the recent tails of mature M4 series, and not the first observations of new products, whose launch ramps and intermittency M4 tails lack. Whether our results carry over to new products remains to be tested.

Diagnosis of a failing prior. The hourly results show that a failing prior can have more than one cause, and that these causes can be separated (Section 4.5). Hence, before concluding that synthetic data does not suit a frequency, it is worth checking how inputs are normalized and whether the corpus contains the dominant structure of the target data. Seasonal strength and feature coverage are cheap checks, and a rerun that changes a single mechanism or overlay tests the diagnosis directly.

Relation to other findings. Cazaux et al. (2026) find that the efect of synthetic augmentation depends mainly on model architecture, with gains at reduced data limited to some combinations of architecture and dataset, whereas our results show that the efect of synthetic pretraining depends on the amount of available target history. The two findings are complementary, and together suggest that whether synthetic data helps has no general answer. Our results also difer from those of Oreshkin et al. (2026) on GiftEval and of Ansari et al. (2025) on fev-bench, where synthetic or partly synthetic pretraining produces models that beat statistical baselines zero-shot. On the M-Series, which include M4, Oreshkin et al. (2026) instead report mixed results against AutoARIMA, which is closer to what we observe. Moreover, our corpora are orders of magnitude smaller, our models are about 700 times smaller than Chronos-2 in parameter count (156K to 191K against 120M), and our input window is one horizon long, against contexts of up to 8,192 observations. Hence, we believe that synthetic pretraining may require a large scale, a long receptive field, or both, before it competes with fitting the target directly.

Limitations. Limitations of our work are that we evaluate two model classes, trained on the same corpus and preprocessing, on one dataset family. Generalization across architectures and to other datasets, such as the non-M4 collections in the Monash archive (Godahewa et al., 2021), is untested. Both learned conditions train for fixed budgets without validation or early stopping, the input window equals the forecast horizon, and no condition pretrains on real series, unlike Kuvshinova et al. (2024), who also pretrain on M3 and M4. Hence, our results compare a synthetic prior with one untuned way of training on the target, and cannot separate the value of synthetic data from that of pretraining in general. Finally, the authors develop the library, so this is not an independent evaluation. The declaration of competing interest at the end of this article states the relationship, and Appendix C and the artifact allow the evaluation to be audited.

Future work. For future work, we see four experiments that follow from the questions this paper leaves open:

1. Varying corpus size while holding mechanism diversity fixed, and vice versa, would separate two explanations our results leave confounded: that the prior is too small, or that it has the wrong shape.

2. Applying a large pretrained forecaster to the same splits would test whether synthetic pretraining escapes the crossover at scale, read as an upper bound on zero-shot accuracy rather than as a matched comparison.

3. Comparing the moving block bootstrap, which preserves the observed trend and seasonality, with parametric augmentation, which replaces the process with a fitted approximation, on matched splits and panel sizes would show which of the two improves forecasts more.

4. Rerunning the full grid on a corpus with calendar cycles, and adding mechanisms targeted at the regions of feature space that the corpus still does not reach, would show how much of the remaining gap to Seasonal Naive is due to the corpus rather than to the scale of the model.

## 6. Conclusion

In this paper, we presented SynForecast, an opensource Python library for synthetic time series designed for forecasting research. Its 32 generators, 19 of which implement named mechanisms with a published source, share one interface. This interface makes it possible to ablate mechanism families outside the training code of a single model, and to stress-test forecasting methods on a shared base trajectory.

Using SynForecast on M4 with two small learned models, we find four results. First, at this scale, an automatically specified classical baseline attains the lowest error in nine of ten frequency-history combinations. Second, a synthetic prior outperforms the same network trained from scratch at a fixed budget by up to 34% with a few dozen observations, and loses to it at full history at every frequency. At four frequencies, the sign changes between 84 and 317 observations, and none of the three rescalings of history we pre-specified predicts where. At the hourly frequency, where the prior fails, the corpus lacks daily cycles: adding them reduces zero-shot error by 43 to 63% for both models under per-series standardization and gives the NHITS prior a crossover at 323 observations, while normalization explains why the remaining error grows with history. Third, fine-tuning the prior outperforms training from scratch at four frequencies, by about 13% at quarterly, but at short histories a fixed 300-step fine-tune is worse than the prior used as is. Fourth, removing the classical family costs about 2% for both models, and three further families have efects of about 1% that replicate across models on 20 replicates, but not on the 10 we had pre-specified.

To conclude, the usefulness of synthetic data for forecasting depends on the setting, in ways that permodel studies have only begun to map. We believe that SynForecast provides a basis for extending these experiments to other architectures, other datasets, augmentation, and the corpus scales at which simulation-only training has recently succeeded.

## CRediT author statement

Olivier Sprangers: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Project administration, Writing – original draft, Writing – review and editing. Cristian Challu and Jan Rathfelder: Writing – original draft, Writing – review and editing. Han Wang: Software. Max Mergenthaler Canseco: Writing – review and editing.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-forprofit sectors.

## Declaration of competing interest

The authors work at Nixtla, the company that develops the software examined as the principal subject of this article. This relationship may be perceived as a competing interest.

## Data and code availability

All datasets used are public and obtained programmatically; no new dataset is released and no source file is redistributed. SynForecast is released under the Apache 2.0 license at https://github.com/Nixtla/synforecast and on PyPI; every experiment ran against synforecast 0.2.0 as published there, except the hourly follow-up of Section 4.5. That follow-up ran against commit b2338c8 of the repository, which adds the calendar cycles and was not yet part of a release.

The accompanying artifact contains the experiment runners, which record the package versions, seeds and realized training steps of every cell; the analysis scripts; the pre-specification; the result rows of every cell; and a manifest mapping each table and figure to the script and inputs that produce it, so that every table and figure can be regenerated without a GPU. It is publicly available at https://doi.org/10.6084/m9.figshare.34874388.

Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the authors used OpenAI Codex and Anthropic Claude Code to assist with literature discovery and drafting. The authors reviewed and edited the content and take full responsibility for the content of the article.

## References

Alexandrov, A., Benidis, K., Bohlke-Schneider, M., Flunkert, V., Gasthaus, J., Januschowski, T., Maddix, D.C., Rangapuram, S., Salinas, D., Schulz, J., Stella, L., Türkmen, A.C., Wang, Y., 2020. GluonTS: Probabilistic and neural time series modeling in Python. Journal of Machine Learning Research 21, 1–6. URL: https://jmlr.org/papers/v21/19-820.html.

Ang, Y., Huang, Q., Bao, Y., Tung, A.K.H., Huang, Z., 2023. TSGBench: Time series generation benchmark. Proceedings of the VLDB Endowment 17, 305– 318. doi:10.14778/3632093.3632097.

Ansari, A.F., Shchur, O., Küken, J., Auer, A., Han, B., Mercado, P., Rangapuram, S.S., Shen, H., Stella, L., Zhang, X., Goswami, M., Kapoor, S., Maddix, D.C., Guerron, P., Hu, T., Yin, J., Erickson, N., Desai, P.M., Wang, H., Rangwala, H., Karypis, G., Wang, Y., Bohlke-Schneider, M., 2025. Chronos-2: From univariate to universal forecasting. arXiv:2510.15821.

Ansari, A.F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S.S., Pineda Arango, S., Kapoor, S., Zschiegner, J., Maddix, D.C., Wang, H., Mahoney, M.W., Torkkola, K., Wilson, A.G., Bohlke-Schneider, M., Wang, Y., 2024. Chronos: Learning the language of time series. Transactions on Machine Learning Research ArXiv:2403.07815.

Bandara, K., Hewamalage, H., Liu, Y.H., Kang, Y., Bergmeir, C., 2021. Improving the accuracy of global forecasting models using time series data augmentation. Pattern Recognition 120, 108148. doi:10.1016/ j.patcog.2021.108148.

Bergmeir, C., Hyndman, R.J., Benítez, J.M., 2016. Bagging exponential smoothing methods using STL decomposition and Box–Cox transformation. International Journal of Forecasting 32, 303–312. doi:10.1016/j. ijforecast.2015.07.002.

Bhethanabhotla, S.K., Swelam, O., Siems, J., Salinas, D., Hutter, F., 2024. Mamba4Cast: Eficient zeroshot time series forecasting with state space models. arXiv:2410.09385.

Cazaux, H., Ásgeirsson, E.I., Stefánsson, H., 2026. Does Synthetic Data Help? Empirical Evidence from Deep Learning Time Series Forecasters. arXiv:2605.06032.

Challu, C., Olivares, K.G., Oreshkin, B.N., Garza Ramirez, F., Mergenthaler Canseco, M., Dubrawski, A., 2023. NHITS: Neural hierarchical interpolation for time series forecasting, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 6989–6997. doi:10.1609/aaai.v37i6.25854.

Das, A., Kong, W., Sen, R., Zhou, Y., 2024. A decoderonly foundation model for time-series forecasting, in: Proceedings of the 41st International Conference on Machine Learning, pp. 10148–10167. ArXiv:2310.10688.

Dooley, S., Khurana, G.S., Mohapatra, C., Naidu, S., White, C., 2023. ForecastPFN: Synthetically-trained zero-shot forecasting, in: Advances in Neural Information Processing Systems 36. ArXiv:2311.01933.

Godahewa, R., Bergmeir, C., Webb, G.I., Hyndman, R.J., Montero-Manso, P., 2021. Monash time series forecasting archive, in: Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks. ArXiv:2105.06643.

Herzen, J., Lässig, F., Piazzetta, S.G., Neuer, T., Tafti, L., Raille, G., Van Pottelbergh, T., Pasieka, M., Skrodzki, A., Huguenin, N., Dumonal, M., Kościsz, J., Bader, D., Gusset, F., Benheddi, M., Williamson, C., Kosinski, M., Petrik, M., Grosch, G., 2022. Darts: User-friendly modern machine learning for time series. Journal of Machine Learning Research 23, 1–6. ArXiv:2110.03224.

Hewamalage, H., Bergmeir, C., Bandara, K., 2022. Global models for time series forecasting: A simulation study. Pattern Recognition 124, 108441. doi:10.1016/j. patcog.2021.108441.

Holm, S., 1979. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6, 65– 70. URL: https://www.jstor.org/stable/4615733.

Hyndman, R.J., Koehler, A.B., 2006. Another look at measures of forecast accuracy. International Journal of Forecasting 22, 679–688. doi:10.1016/j.ijforecast. 2006.03.001.

Hyndman, R.J., Koehler, A.B., Snyder, R.D., Grose, S., 2002. A state space framework for automatic forecasting using exponential smoothing methods. International Journal of Forecasting 18, 439–454. doi:10.1016/ S0169-2070(01)00110-8.

Kang, Y., Hyndman, R.J., Li, F., 2020. GRATIS: GeneR-Ating TIme Series with diverse and controllable characteristics. Statistical Analysis and Data Mining 13, 354–376. doi:10.1002/sam.11461.

Kang, Y., Hyndman, R.J., Smith-Miles, K., 2017. Visualising forecasting algorithm performance using time series instance spaces. International Journal of Forecasting 33, 345–358. doi:10.1016/j.ijforecast.2016.09.004.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.Y., 2017. LightGBM: A highly eficient gradient boosting decision tree, in: Advances in Neural Information Processing Systems 30, Curran Associates, Inc. URL: https: //proceedings.neurips.cc/paper/2017/hash/ 6449f44a102fde848669bdd9eb6b76fa-Abstract. html.

Kuvshinova, K., Tsymboi, O., Kostromina, A., Simakov, D., Kovtun, E., 2024. Towards foundation time series model: To synthesize or not to synthesize? arXiv:2403.02534.

Liu, X., Aksu, T., Liu, J., Wen, Q., Liang, Y., Xiong, C., Savarese, S., Sahoo, D., Li, J., Liu, C., 2025. Empowering time series analysis with synthetic data: A survey and outlook in the era of foundation models. arXiv:2503.11411.

Makridakis, S., Spiliotis, E., Assimakopoulos, V., 2020. The M4 competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting 36, 54–74. doi:10.1016/j.ijforecast.2019.04.014.

Montero-Manso, P., Hyndman, R.J., 2021. Principles and algorithms for forecasting groups of time series: Locality and globality. International Journal of Forecasting 37, 1632–1653. doi:10.1016/j.ijforecast.2021.03.004.

Moroshan, V., Siems, J., Zela, A., Carstensen, T., Hutter, F., 2026. TempoPFN: Synthetic pre-training of linear RNNs for zero-shot time series forecasting, in: Advances in Neural Information Processing Systems 39. To appear. arXiv:2510.25502.

Nikitin, A., Iannucci, L., Kaski, S., 2024. TSGM: A flexible framework for generative modeling of synthetic time series, in: Advances in Neural Information Processing Systems 37 (Datasets and Benchmarks Track), pp. 129042– 129061. doi:10.52202/079017-4098.

Oreshkin, B.N., Jauhari, M., Selvam, R.K., Wolf, M., Pan, W., Ramasubramanian, S., Olivares, K.G., Konstantinova, T., Potapczynski, A., Cao, M., Efimov, D., Mahoney, M.W., Wilson, A.G., 2026. Zero-shot forecasting by simulation alone, in: International Conference on Learning Representations. ArXiv:2601.00970.

Petitjean, F., Ketterlin, A., Gançarski, P., 2011. A global averaging method for dynamic time warping, with applications to clustering. Pattern Recognition 44, 678–693. doi:10.1016/j.patcog.2010.09.013.

Petropoulos, F., Hyndman, R.J., Bergmeir, C., 2018. Exploring the sources of uncertainty: Why does bagging for time series forecasting work? European Journal of Operational Research 268, 545–554. doi:10.1016/j.ejor. 2018.01.045.

Petropoulos, F., et al., 2022. Forecasting: theory and practice. International Journal of Forecasting 38, 705–871. doi:10.1016/j.ijforecast.2021.11.001.

Talagala, T.S., Hyndman, R.J., Athanasopoulos, G., 2023. Meta-learning how to forecast time series. Journal of Forecasting 42, 1476–1501. doi:10.1002/for.2963.

Wang, E., Cook, D., Hyndman, R.J., 2020. A new tidy data structure to support exploration and modeling of temporal data. Journal of Computational and Graphical Statistics 29, 466–478. doi:10.1080/10618600.2019. 1695624.

Wenig, P., Schmidl, S., Papenbrock, T., 2022. TimeEval: A benchmarking toolkit for time series anomaly detection algorithms. Proceedings of the VLDB Endowment 15, 3678–3681. doi:10.14778/3554821.3554873.

Wilcoxon, F., 1945. Individual comparisons by ranking methods. Biometrics Bulletin 1, 80–83. doi:10.2307/ 3001968.

Wong, C.S., Li, W.K., 2000. On a mixture autoregressive model. Journal of the Royal Statistical Society: Series B 62, 95–115. doi:10.1111/1467-9868.00222.

Yuan, X., Qiao, Y., 2024. Difusion-TS: Interpretable diffusion for general time series generation, in: International Conference on Learning Representations. URL: https://openreview.net/forum?id=4h1apFjO99.

## Appendix A. Existing generators

Table A.5 scores 11 existing generators against the four requirements of Section 1. We included generators that produce series from a model rather than transforming observed ones, and that are either packaged for reuse or were used to pretrain a published forecasting model; we excluded learned generative models, which are fitted to data and are closer to augmentation. One included generator, generate\_series in utilsforecast, belongs to the same ecosystem as SynForecast. Every score comes from reading the released code and documentation at the stated version; nothing was executed. SarSim0 has no official code release, so only its design, which the paper describes, is scored. The requirements are SynForecast’s design goals (Section 1), so the table positions SynForecast rather than ranking tools built for other purposes.

Two generators fully meet two requirements each. GutenTAG derives a seed for each dataset from the base seed and the dataset’s name, so its parallel output does not depend on the number of workers, but its output is indexed by position, with no frequency or calendar origin. TempoPFN stores a frequency, start and identity with every series and selects among 14 generators by weight, but its released command line seeds each run from the wall clock, and its per-generator seeds include Python’s salted string hash, so a seed alone does not reproduce a run across interpreter launches. Four existing generators return a forecasting-ready panel: gratis through a tsibble, GluonTS, TempoPFN and utilsforecast. However, none of them combines this panel with perturbations that leave the base series untouched by design. A Python port of gratis in its repository, not yet released, returns bare arrays.

On other axes, other tools lead: gratis can target any user-specified vector of series features (its paper demonstrates up to 12) against our four, KernelSynth, ForecastPFN, TempoPFN and SarSim0 have each trained a published forecasting model, SarSim0 generates a billion series on the fly, and the multivariatizers of Ansari et al. (2025) impose multivariate structure that our joint generators do not attempt.

Table A.5: Existing generators against the requirements of Section 1: (i) output carries a frequency, a time origin and a series identity; (ii) a perturbation can be varied while the base series stays bit-identical; (iii) output is identical under parallel execution; (iv) many named generating processes, each removable. Yes means fully met. For (ii), partly means perturbations act on a generated series but their isolation follows from draw order or a shared random stream and is not guaranteed. For (iii), partly means the library exposes a seed but either generates in one process or gives no guarantee across worker counts; no means no seed is exposed. For (iv), yes means ten or more mechanisms the user can select individually, partly two to nine, or primitives the user composes, and no a single family whose components the user cannot select. A dash marks a property that cannot be assessed without code. TimeSynth has no publication; its repository, last updated in 2020, is https://github.com/TimeSynth/TimeSynth.
<table><tr><td>Generator</td><td>Version inspected</td><td>(i)</td><td>(ii)</td><td>(iii)</td><td>(iv)</td><td>Base mechanisms</td></tr><tr><td>gratis (R) (Kang et al., 2020)</td><td>1.0.8</td><td>yes</td><td>no</td><td>partly</td><td>partly</td><td>3: MAR, ARIMA, ETS</td></tr><tr><td>GluonTS dataset.artificial (Alexandrov et al., 2020)</td><td>commit d60556c</td><td>yes</td><td>partly</td><td>partly</td><td>partly</td><td>2 presets, about 14 primitives</td></tr><tr><td>Darts generators (Herzen et al., 2022)</td><td>0.47.0</td><td>partly</td><td>no</td><td>no</td><td>partly</td><td>6</td></tr><tr><td>TimeSynth</td><td>0.2.4</td><td>no</td><td>partly</td><td>no</td><td>partly</td><td>7</td></tr><tr><td>TSGM simulators (Nikitin et al., 2024)</td><td>0.1.0</td><td>no</td><td>partly</td><td>partly</td><td>partly</td><td>3 simulators</td></tr><tr><td>GutenTAG (Wenig et al., 2022)</td><td>1.5.0</td><td>partly</td><td>partly</td><td>yes</td><td>yes</td><td>13 base oscillations</td></tr><tr><td>KernelSynth (Ansari et al., 2024)</td><td>commit 1224faf</td><td>partly</td><td>no</td><td>no</td><td>no</td><td>1: composed GP kernels</td></tr><tr><td>ForecastPFN prior (Dooley et al., 2023)</td><td>commit 919dda4</td><td>partly</td><td>no</td><td>no</td><td>no</td><td>1: trend × season × noise</td></tr><tr><td>utilsforecast</td><td>0.2.16</td><td>yes</td><td>no</td><td>partly</td><td>no</td><td>1: sawtooth plus noise</td></tr><tr><td>TempoPFN pipeline (Moroshan et al., 2026)</td><td>commit 5969ec6</td><td>yes</td><td>partly</td><td>partly</td><td>yes</td><td>14</td></tr><tr><td>SarSim0 (Oreshkin et al., 2026)</td><td>no official code</td><td></td><td></td><td></td><td>no</td><td>1: SARIMA</td></tr><tr><td>SynForecast</td><td>0.2.0</td><td>yes</td><td>yes</td><td>yes</td><td>yes</td><td>32</td></tr></table>

## Appendix B. Experimental protocol details

Timestamps. M4 series are distributed with an integer index, so we assign synthetic timestamps from a fixed origin at the frequency of the group. These carry no calendar meaning, and no model receives a calendar feature.

Seasonal periods. The natural periods of Section 4.1 serve both in the seasonal MASE denominator and in the cycle normalization. The M4 convention, which treats weekly and daily as non-seasonal, would leave too few frequencies with a defined period to distinguish the normalizations of Section 4.4.

Seeds. Every random seed, for the synthetic pool and for initializing the pretrained, from-scratch and fine-tuned models, is derived from one fixed entropy value and a key naming its role, frequency, replicate and model, so no two replicates or roles share a seed. Two sharings are deliberate: both models pretrain on the same corpus within a (frequency, replicate), and every ablation arm reuses the control’s draws (Section 4.6).

History levels. Every history level satisfies two floors: it must admit at least one training window, and it must be at least twice the seasonal period, because seasonal MASE divides by the mean absolute diference at the seasonal lag and a history equal to the period supplies no such diferences at all.

Replicates and run-to-run variation. The grid has 30 replicates; the ablation arms run on the first 20. Training on a GPU need not be bit-reproducible, so 20 cells, four per frequency, were evaluated twice with identical seeds and checkpoints. Zero-shot forecasts were bit-identical, as no training happens at evaluation. The conditions trained at evaluation were bit-identical in all 16 non-hourly cells and difered in all four hourly ones; the largest diference in a cell mean was 0.12 replicate standard deviations for from\_scratch, below 0.01 for pretrain\_ft, and 0.52 for from\_scratch\_long. One AutoETS cell (daily) difered only by floating-point rounding, of the order of $1 0 ^ { - 1 3 }$ . Pretraining was not repeated, so its run-to-run variation is not measured. Run-to-run variation is part of the replicatelevel variance that the tests already absorb, so it bounds the reproduction of point estimates rather than the inference.

Reproducibility. Experiments install SynForecast 0.2.0 from PyPI, except the hourly follow-up of Section 4.5, which installs commit b2338c8 from the repository. Every cell records the versions of each package it imported, its seeds and its realized step counts. The artifact at https://doi.org/10.6084/m9.figshare.34874388 regenerates every table and figure.

Test construction. Each history level is tested on its own 30 replicates; we do not pool across history levels, because the six levels within a replicate are nested truncations of the same subsample and pooling them would treat strongly dependent observations as independent. Correction is Holm within each comparison’s family of 30 tests, and every significance statement above is post-correction.

Axis. The crossover is located against the realized mean training length of each cell, with full history included as an ordinary point. A nominal history level overstates what a cell trains on, because retaining the last k observations returns all of a series shorter than k; at the largest weekly level the overstatement is 66 observations. The realized lengths were recomputed ofline from the series subsamples.

## Appendix C. Pre-specification, outcomes and deviations

The expectations and decision rules for the experiments of Section 4 were written down and frozen on 2026-10- 13, after smoke runs on reduced budgets had checked the pipeline and before the experiment was launched. The pre-specification is an internal document shipped with the artifact, not an external registration. It was written after earlier runs on the same evaluation subsamples, and several of its choices drew on them: the expectation that history relative to the horizon would be the least dispersed normalization and seasonal cycles the most, the exclusion of hourly from the ablation’s primary population, and the log-ratio efect measure. The rule text, including the conclusion to be drawn under each crossover outcome, is in the artifact. Table S7 in the supplementary material lists every expectation against its outcome.

The hourly follow-up of Section 4.5 has its own prespecification, written on 2026-10-30, after the main experiment and before the follow-up was launched. It was also written after a five-replicate pilot with NHITS alone, on a re-implementation of the pipeline, whose estimates it states. Its efect measure, tests and replication criterion are those of the main experiment. The last four rows of Table S7 score it.

Deviations. Four departures from the pre-specification bear on what is reported.

1. Ablation replicates. The ablation was fixed at replicates 0–9 and run on 0–19. The change was decided before launch, because ten replicate clusters give under-covering bootstrap intervals; the runner whose hash was recorded at launch already uses 20. Table F.7 reports both; on the pre-specified replicates only the classical family replicates across models.

2. Crossover axis. The rule as first fixed located the crossover on the nominal history levels with full history excluded. After an earlier run, in which that placement excluded the crossover itself at monthly and quarterly, we moved to realized training lengths with full history included (Appendix B). On the original axis, this run’s verdict is unchanged – no collapse for either model – but only daily and weekly cross, so the coeficients of variation rest on two frequencies: history relative to the horizon is least dispersed (NHITS 0.17, LightGBM 0.19), only 1.30 and 1.26 times clear of the runner-up against the required 2.

3. Budget check. With ten replicates, the pre-specified Holm family of 30 tests cannot reach significance, so the check is reported as estimates with intervals (Section 4.4; Table S5 in the supplementary material).

4. Run-to-run noise. The from-scratch shifts of 0.12 replicate standard deviations, and 0.52 with the 1,000-step budget, exceed the pre-specified 0.1, so diferences of that size between trained conditions are within run-to-run noise (Appendix B).

The analyses that report the ablation on both replicate sets, the budget check as intervals, the verdict on the original axis, the zero-shot/Seasonal Naive ratios and run-torun noise by frequency were added in revision and are not pre-specified; they are in the artifact as a separate script.

## Appendix D. Generation cost

Throughput against worker count. We hold the total work fixed at 864 series of 1,024 observations from the 54-instance pretraining pool and vary the worker count. On a machine with 20 logical cores (Linux under WSL2), throughput rises from 171 series per second with one worker to a plateau of 370 to 420 between six and twelve workers, peaks at eight with a speedup of 2.45, and falls to 321 at 20 (Table D.6a). A random walk alone generates 34,000 to 40,000 series per second at any worker count, so for cheap mechanisms the cost of a series is fixed per-call work. The pool is limited instead by its three KernelSynth instances, which take about four-fifths of single-worker time and factorize a dense covariance matrix with a multithreaded linear-algebra library, so that worker and library threads compete for cores beyond the plateau. The library’s default of using every core is therefore not the best setting, and a caller generating at scale should set the worker count explicitly.

Cost across mechanisms. On one series of length 1,000, the public generation call costs a median of 0.87 ms across the 32 generators, mostly fixed per-call overhead that amortizes in batch; KernelSynth costs 101 times the median, and without it the spread is sixfold.

Cost concentrates in length. At 9,919 observations, the longest daily series, one KernelSynth series takes 8.2 seconds and 5.7 GB of resident memory (97% of the summed cost of one series from each of the 19 classes in the pool), and its cost grows superlinearly with length. Table D.6b shows the consequence for our experiment: the quarterly draw (lengths 16–866) takes 1.3 seconds and 0.3 GB, the daily draw (93–9,919) 391 seconds and 22.7 GB. Generating each (frequency, replicate) draw once and sharing it across both models and all ablation arms kept the experiment’s 150 draws to 2.5 CPU-container-hours, against 8.4 GPU-hours for the 1,730 pretraining runs.

Table D.6: Generation throughput. Panel (a) holds total work fixed at 864 series of 1,024 observations from the 54-instance pretraining pool and varies the worker count; medians of five interleaved repetitions on 20 logical cores. Panel (b) is the cost of one pretraining draw at each frequency, as the experiments of this section generated it: 54 instances, 26 series each, four workers. Each draw runs in a fresh process, and peak resident memory is that process’s own.  
(a) Throughput against worker count
<table><tr><td>Workers</td><td>Median (s)</td><td>Series/s</td><td>Speedup</td><td>Efficiency</td></tr><tr><td>1</td><td>5.05</td><td>171</td><td>1.00</td><td>1.00</td></tr><tr><td>2</td><td>3.89</td><td>222</td><td>1.30</td><td>0.65</td></tr><tr><td>4</td><td>2.77</td><td>312</td><td>1.82</td><td>0.46</td></tr><tr><td>6</td><td>2.33</td><td>371</td><td>2.17</td><td>0.36</td></tr><tr><td>8</td><td>2.06</td><td>419</td><td>2.45</td><td>0.31</td></tr><tr><td>12</td><td>2.27</td><td>381</td><td>2.23</td><td>0.19</td></tr><tr><td>20</td><td>2.69</td><td>321</td><td>1.87</td><td>0.09</td></tr></table>

(b) Cost of one pretraining draw, by frequency
<table><tr><td>Frequency</td><td>Lengths</td><td>Rows</td><td>Seconds</td><td>Series/s</td><td>Peak RSS</td></tr><tr><td>Quarterly</td><td>16-866</td><td>627,364</td><td>1.3</td><td>1069</td><td>0.31 GB</td></tr><tr><td>Hourly</td><td>700-960</td><td>1,167,877</td><td>3.6</td><td>388</td><td>0.46 GB</td></tr><tr><td>Weekly</td><td>80-2,597</td><td>1,908,521</td><td>12.1</td><td>116</td><td>1.95 GB</td></tr><tr><td>Monthly</td><td>42-2,794</td><td>2,012,590</td><td>14.3</td><td>98</td><td>2.13 GB</td></tr><tr><td>Daily</td><td>93–9,919</td><td>7,120,903</td><td>391.3</td><td>4</td><td>22.67 GB</td></tr></table>

## Appendix E. Overlay orthogonality

Both invariance properties are asserted by the test suite. One test checks that a given seed produces identical values with one and with four workers, for three batch-path generators and a composed pool. Another test checks orthogonality for all 32 generators, using the flag columns that report which observations each overlay touched: every untouched observation is bit-identical to the run without the overlay, series lengths are unchanged, and each overlay lands in the same place whatever other overlays are enabled. For changepoints, which alter everything after them, the test checks the observations before the first changepoint, and a level-shift anomaly, flagged at its start only, is checked over its configured duration.

The orthogonality test exists because the property did not hold everywhere. Before version 0.2.0, the generators outside the batch path (KernelSynth, copula, VAR, multivariate TCM, IoT sensor and daily active users) drew all three overlays from one stream, so enabling one overlay could move another. Version 0.2.0 gives every series three perturbation seeds on every path and adds the test to guard against this failure.

## Appendix F. Ablation results

Table F.7 reports the ablation of Section 4.6 on both replicate sets, the run as executed and the pre-specified one, together with the hourly efects that Figure 5 omits.

Table F.7: Mechanism-family ablation. Efect = 100 × log(geometric-mean zero-shot MASE, arm / control); positive means removing the family made zero-shot accuracy worse, so the family contributes. For each family, the first row is the run as executed (replicates 0–19) and the second the pre-specified replicates 0–9 (Appendix C). Primary population: daily, weekly, monthly and quarterly, three history levels each; 95% interval from a bootstrap over replicates; bold where the interval excludes zero. Hourly is reported separately, as a point estimate on replicates 0–19. Families: classical = SARIMA, ETS, INAR; Gaussian process = four GP instances; heavy-tail and regime = GARCH, Lévy, regime switching; other stochastic = fractional Brownian motion, chaotic systems, bounded processes, cyclic; domain = intermittent demand, energy load, IoT sensor, vital signs, of whose 11 instances one almost never contributes (Section 4.1); procedural = TSI, TCM, KernelSynth; MAR = three non-seasonal mixture-autoregressive instances.
<table><tr><td colspan="3"></td><td colspan="4">Primary population</td><td colspan="2">Hourly</td></tr><tr><td>Family removed</td><td>Replicates</td><td>Inst.</td><td>NHITS</td><td></td><td>LightGBM</td><td></td><td>NHITS</td><td>LightGBM</td></tr><tr><td rowspan="2">Classical</td><td>0-19</td><td>11</td><td>+1.83</td><td>[+1.15, +2.51]</td><td>+2.30</td><td>[+1.93,+2.63]</td><td>+2.2</td><td>-2.3</td></tr><tr><td>0-9</td><td></td><td>+1.69</td><td>[+0.91, +2.51]</td><td>+2.46</td><td>[+1.85, +2.98]</td><td></td><td></td></tr><tr><td rowspan="2">Gaussian process</td><td>0-19</td><td></td><td></td><td>4 +0.88[+0.25, +1.53]</td><td>+0.93</td><td>[+0.71,+1.13]</td><td>+9.8</td><td>+10.8</td></tr><tr><td>0-9</td><td></td><td></td><td>+0.90 [−0.18, +2.06]</td><td>+0.98</td><td>[+0.67, +1.27]</td><td></td><td></td></tr><tr><td rowspan="2">Heavy-tail, regime</td><td>0-19</td><td></td><td></td><td>6 +0.98[+0.47,+1.45]</td><td>+0.83</td><td>[+0.52, +1.11]</td><td>-1.0</td><td>-6.6</td></tr><tr><td>0-9</td><td></td><td></td><td>+0.81 [−0.04, +1.60]</td><td>+0.86</td><td>[+0.49, +1.26]</td><td></td><td></td></tr><tr><td rowspan="2">Other stochastic</td><td>0-19</td><td>10</td><td></td><td>-0.83 [-1.47, -0.22]</td><td>-0.79</td><td>[—1.04, -0.52]</td><td>-13.5</td><td>-1.0</td></tr><tr><td>0-9</td><td></td><td></td><td>-0.66 [−1.61, +0.22]</td><td>-0.80</td><td>[-1.18, -0.41]</td><td></td><td></td></tr><tr><td rowspan="2">Domain</td><td>0-19</td><td></td><td>11 +0.49</td><td>[−0.11,+1.09]</td><td></td><td>−0.25 [−0.54, +0.06]</td><td>+7.6</td><td>+19.2</td></tr><tr><td>0-9</td><td></td><td>+0.26</td><td>[−0.61,+1.17]</td><td></td><td>−0.20 [−0.64, +0.22]</td><td></td><td></td></tr><tr><td rowspan="2">Procedural</td><td>0-19</td><td>9</td><td>-0.35</td><td>[−1.15, +0.43]</td><td>-0.48</td><td>[—0.77, -0.21]</td><td>+4.9</td><td>+6.1</td></tr><tr><td>0-9</td><td></td><td>-0.48</td><td>[-1.87, +0.86]</td><td>-0.48</td><td>[-0.93, -0.07]</td><td></td><td></td></tr><tr><td rowspan="2">MAR</td><td>0-19</td><td>3</td><td>-0.05</td><td>[−0.66, +0.54]</td><td>-0.61 -0.54</td><td>[—0.80, -0.42]</td><td>+0.3</td><td>0.0</td></tr><tr><td>0-9</td><td></td><td>+0.05</td><td>[−0.72,+0.73]</td><td></td><td>[-0.79, -0.27]</td><td></td><td></td></tr></table>

# Supplementary material for “SynForecast: Controlled Synthetic Time Series for Forecasting Research”

Olivier Sprangers<sup>a,∗</sup>, Cristian Challu<sup>a</sup>, Jan Rathfelder<sup>a</sup>, Han Wang<sup>a</sup>, Max Mergenthaler Canseco<sup>a</sup>

<sup>a</sup>Nixtla, San Francisco, CA, United States

Table S1 gives, for each of the 32 generators in synforecast 0.2.0, the model each generator implements and the work that introduced it. The five domain simulators have no canonical source: they compose stylized components (daily and weekly profiles, drifts, events, failures) and their tests pin those designed behaviors rather than a published model’s statistics. The random walk, the seasonal sinusoid, the cyclic generator and the Poisson process are textbook constructions and are listed without a source.

Table S1: Generators in synforecast 0.2.0 (class names without the Generator sufix), grouped as in Section 3 of the article, with the model each implements and its source. A dash marks a textbook construction or a stylized domain simulator.
<table><tr><td>Generator</td><td>Model</td><td>Source</td></tr><tr><td>Statistical</td><td></td><td></td></tr><tr><td>RandomWalk</td><td>Random walk with drift</td><td></td></tr><tr><td>Seasonal</td><td>Sinusoid plus linear trend and noise</td><td></td></tr><tr><td>SARIMA</td><td>Seasonal ARIMA  $( p , d , q ) \times ( P , D , Q ) _ { \ l }$  with</td><td>Box et al., 2008</td></tr><tr><td>ETS</td><td>optional regressors Innovations state space form of exponential smoothing</td><td>Hyndman et al., 2002, 2008</td></tr><tr><td>INAR</td><td>INAR(p) with binomial thinning, Poisson or negative binomial</td><td>Steutel and van Harn, 1979; Al-Osh and Alzaid, 1987; Du and Li, 1991</td></tr><tr><td>Stochastic</td><td>innovations</td><td></td></tr><tr><td>GARCH</td><td>GARCH(p, q)</td><td>Bollerslev, 1986</td></tr><tr><td>OrnsteinUhlenbeck</td><td>Mean-reverting diffusion, Euler-Maruyama</td><td>Uhlenbeck and Ornstein, 1930</td></tr><tr><td>GeometricBrownianMotion</td><td>Log-normal diffusion, exact discretization</td><td>Black and Scholes, 1973</td></tr><tr><td>JumpDiffusion</td><td>Geometric Brownian motion with compound</td><td>Merton, 1976</td></tr><tr><td>PoissonProcess</td><td>Poisson log-normal jumps Homogeneous Poisson counts</td><td></td></tr><tr><td>Cyclic</td><td>Superposed sinusoids with slowly modulated</td><td></td></tr><tr><td>FractionalBrownianMotion</td><td>frequency Fractional Brownian motion; circulant</td><td>Mandelbrot and Van Ness, 1968; Davies and Harte, 1987</td></tr><tr><td>HawkesProcess</td><td>embedding by default Self-exciting point process, exponential or</td><td>Hawkes, 1971</td></tr><tr><td>StochasticVolatility</td><td>power-law kernel Heston; SABR</td><td>Heston, 1993; Hagan et al.,</td></tr><tr><td>RegimeSwitching</td><td>Markov-switching AR(1)</td><td>2002 Hamilton, 1989</td></tr><tr><td>ChaoticSystem</td><td>Lorenz system; logistic map; Mackey-Glass</td><td>Lorenz, 1963; May, 1976; Mackey and Glass, 1977</td></tr><tr><td>BoundedProcess</td><td>equation Beta autoregression;</td><td>Rocha and Cribari-Neto, 2009</td></tr><tr><td>LevyProcess</td><td>logit-normal AR(1) α-stable increments, Chambers-Mallows-Stuck</td><td>Chambers et al., 1976</td></tr><tr><td>Multivariate</td><td>sampling</td><td></td></tr><tr><td>Copula</td><td>Gaussian or t copula with Nelsen, 2006; Demarta and</td><td></td></tr><tr><td>VAR</td><td>arbitrary marginals Stable vector</td><td>McNeil, 2005 Lütkepohl, 2005</td></tr><tr><td>GaussianProcess</td><td>Gaussian-process prior; RBF, Matérn or periodic kernel</td><td>Rasmussen and Williams, 2006</td></tr><tr><td>Domain</td><td></td><td></td></tr><tr><td>IntermittentDemand</td><td>Bernoulli demand occurrence times a</td><td>Croston, 1972</td></tr><tr><td>StateSpace</td><td>demand-size distribution Linear Gaussian state space model, or</td><td>Durbin and Koopman, 2012</td></tr><tr><td>EnergyLoad</td><td>user-supplied dynamics Stylized: daily, weekly and yearly profiles,</td><td></td></tr><tr><td>IoTSensor</td><td>temperature response Stylized: drift, battery degradation, sensor failures</td><td></td></tr><tr><td>VitalSigns</td><td>Stylized: baseline, circadian rhythm, physiological events</td><td></td></tr><tr><td>Clickstream</td><td>Stylized: Poisson sessions, diurnal profile, bot traffic</td><td></td></tr><tr><td>DailyActiveUsers</td><td>Stylized: growth, weekend effect, decaying event boosts</td><td></td></tr><tr><td>Pretraining TSI</td><td>Random trend,</td><td>Bahrpeyma et al., 2021</td></tr><tr><td></td><td>seasonality and irregular composition</td><td></td></tr><tr><td>TCM</td><td>Random temporal structural causal model</td><td>Runge et al., 2023</td></tr><tr><td>KernelSynth</td><td>Gaussian-process prior over randomly composed kernels</td><td>Ansari et al., 2024</td></tr><tr><td>MAR</td><td>Mixture autoregression with feature-targeted search</td><td>Wong and Li, 2000; Kang et al., 2020</td></tr></table>

For TSI and TCM, the pair that Ansari et al. (2025) use as univariate generators, the cited works supply the construction rather than the generator: TSI composes components after Bahrpeyma et al. (2021) with its own component families and sampling distributions, and TCM follows the temporal structural-causal framing of Runge et al. (2023) with its own graph sampler and edge functions.

Table S2: The values of Figure 3 of the article with their uncertainty: relative improvement (%) of the condition named first over the condition named second, NHITS, mean over 30 replicates, with the half-width of the 95% t interval over replicates; history levels from the shortest (left) to full history (right), as in Figure 3 of the article.
<table><tr><td colspan="6">Frequency History level</td></tr><tr><td>(a) zero_shot over from_scratch</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Quarterly  $+ 3 2 . 5 \pm 0 . 9 + 3 3 . 4 \pm 0 . 7 + 3 1 . 5 \pm 1 . 3$ </td><td></td><td></td><td> $+ 2 9 . 1 \pm 1 . 6 $ </td><td> $+ 2 1 . 0 \pm 2 . 0$ </td><td> $- 7 . 3 \pm 2 . 1$ </td></tr><tr><td>Monthly</td><td> $+ 2 6 . 1 \pm 1 . 0 + 2 4 . 5 \pm 1 . 1 + 2 4 . 3 \pm 1 . 1$ </td><td></td><td> $+ 1 9 . 8 \pm 1 . 2$ </td><td> $+ 1 0 . 5 \pm 1 . 8$ </td><td> $- 8 . 9 \pm 1 . 8$ </td></tr><tr><td>Weekly</td><td> $+ 6 . 5 \pm 0 . 9 + 3 . 6 \pm 0 . 8 + 2 . 2 \pm 1 . 0$ </td><td></td><td> $- 0 . 5 \pm 1 . 1$ </td><td> $- 7 . 9 \pm 1 . 3$ </td><td> $- 1 7 . 2 \pm 3 . 6$ </td></tr><tr><td>Daily</td><td> $+ 3 4 . 1 \pm 0 . 9 + 2 9 . 4 \pm 1 . 0 + 1 7 . 5 \pm 1 . 4$ </td><td></td><td> $+ 8 . 8 \pm 1 . 3$ </td><td> $- 1 . 0 \pm 1 . 7$ </td><td> $- 1 7 . 4 \pm 2 . 8$ </td></tr><tr><td>Hourly</td><td> $- 5 1 . 0 \pm 5 . 8 - 6 0 . 5 \pm 5 . 7 - 8 2 . 2 \pm 7 . 0$ </td><td></td><td> $- 1 2 2 . 8 \pm 8 . 7$ </td><td> $- 2 4 8 . 0 \pm 1 3 . 4$ </td><td> $- 3 5 8 . 6 \pm 1 9 . 7$ </td></tr><tr><td>(b) pretrain_ft over from_scratch</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Quarterly</td><td> $+ 1 5 . 3 \pm 1 . 5 + 1 4 . 7 \pm 1 . 6 + 1 5 . 2 \pm 1 . 4$ </td><td></td><td></td><td> $+ 1 5 . 3 \pm 1 . 9$   $+ 1 3 . 8 \pm 2 . 1$ </td><td> $+ 3 . 3 \pm 1 . 5$ </td></tr><tr><td>Monthly</td><td> $+ 6 . 5 \pm 1 . 0$ </td><td> $+ 9 . 1 \pm 0 . 8$   $+ 6 . 3 \pm 0 . 9$ </td><td> $+ 4 . 5 \pm 1 . 0$ </td><td> $+ 2 . 9 \pm 1 . 2$ </td><td> $+ 0 . 6 \pm 1 . 1$ </td></tr><tr><td>Weekly</td><td> $+ 3 . 7 \pm 0 . 8$ </td><td> $+ 3 . 0 \pm 0 . 6$ </td><td> $+ 1 . 9 \pm 0 . 8$   $+ 1 . 1 \pm 0 . 9$ </td><td> $- 1 . 9 \pm 0 . 6$ </td><td> $- 0 . 8 \pm 1 . 0$ </td></tr><tr><td>Daily</td><td> $+ 4 . 3 \pm 0 . 5$ </td><td> $+ 5 . 9 \pm 0 . 9$ </td><td> $+ 4 . 2 \pm 1 . 3$ </td><td> $+ 3 . 8 \pm 1 . 0$   $+ 3 . 1 \pm 1 . 2$ </td><td> $- 1 . 3 \pm 0 . 9$ </td></tr><tr><td>Hourly</td><td> $- 1 0 . 3 \pm 2 . 1$ </td><td>−11.1 ±1.7 −13.5 ±2.3</td><td> $- 1 5 . 3 \pm 2 . 3$ </td><td> $- 2 5 . 2 \pm 2 . 7$ </td><td> $- 2 6 . 4 \pm 3 . 0$ </td></tr><tr><td colspan="2">(c) pretrain_ft over zero_shot</td><td></td><td></td><td></td><td></td></tr><tr><td>Quarterly</td><td></td><td> $- 2 5 . 5 \pm 2 . 5 - 2 8 . 2 \pm 2 . 7 - 2 4 . 0 \pm 2 . 2$ </td><td></td><td> $- 1 9 . 6 \pm 1 . 8$ </td><td> $- 9 . 2 \pm 1 . 8$ </td><td> $+ 9 . 8 \pm 1 . 4$ </td></tr><tr><td> $\mathrm { M o n t h l y }$ </td><td></td><td> $- 2 6 . 6 \pm 1 . 3 - 2 0 . 6 \pm 1 . 3 - 2 3 . 9 \pm 1 . 4$ </td><td></td><td> $- 1 9 . 3 \pm 1 . 4$ </td><td> $- 8 . 6 \pm 1 . 3$ </td><td> $+ 8 . 7 \pm 1 . 1$ </td></tr><tr><td> $\mathrm { W e e k l y }$ </td><td></td><td> $- 3 . 0 \pm 0 . 6 - 0 . 7 \pm 0 . 6 - 0 . 3 \pm 0 . 6$ </td><td></td><td> $+ 1 . 5 \pm 0 . 7$ </td><td> $+ 5 . 5 \pm 1 . 1$ </td><td> $+ 1 3 . 5 \pm 2 . 5$ </td></tr><tr><td>Daily</td><td></td><td> $- 4 5 . 3 \pm 2 . 0 - 3 3 . 4 \pm 1 . 4 - 1 6 . 2 \pm 1 . 2$ </td><td></td><td> $- 5 . 7 \pm 1 . 1$ </td><td> $+ 4 . 0 \pm 1 . 1$ </td><td> $+ 1 3 . 4 \pm 1 . 8$ </td></tr><tr><td>Hourly</td><td></td><td> $+ 2 6 . 4 \pm 2 . 3 + 3 0 . 3 \pm 2 . 1 + 3 7 . 2 \pm 2 . 0$ </td><td></td><td> $+ 4 7 . 8 \pm 1 . 7$ </td><td> $+ 6 3 . 7 \pm 1 . 3$ </td><td> $+ 7 2 . 2 \pm 0 . 9$ </td></tr></table>

## Detailed results

Table S2 gives every value of Figure 3 of the article with its uncertainty, and Table S3 the values plotted in Figure 6 of the article, with the error of each method on the unperturbed panels for scale. Tables S4, S5 and S6 give the fine-tuning tests of Section 4.3 of the article, its budget check of Section 4.4, and the zero-shot error relative to Seasonal Naive on which the reading of the crossover in Section 4.4 rests. Table S7 scores every pre-specified expectation against its outcome, as Appendix C of the article describes.

Table S3: Change in MASE caused by one level shift, by its position relative to the forecast origin; mean over 5 replicates of 300 monthly SARIMA series (standard error across replicates). Breaks “before” fall in the training window; $\mathrm { { ^ { 6 } a f t e r } } ^ { \prime \prime }$ falls in the forecast window. MASE without a break: Seasonal Naive 1.012, AutoETS 0.807, NHITS 0.716.
<table><tr><td>Break position (obs)</td><td>Seas. Naive</td><td>AutoETS</td><td>NHITS</td></tr><tr><td>72 before</td><td>+0.000 (0.000)</td><td>+0.070 (0.006)</td><td>+0.005 (0.002)</td></tr><tr><td>36 before</td><td>+0.000 (0.000)</td><td>+0.070 (0.005)</td><td>+0.014 (0.004)</td></tr><tr><td>18 before</td><td>+0.000 (0.000)</td><td>+0.078 (0.004)</td><td>+0.334 (0.229)</td></tr><tr><td>6 before</td><td>+1.140 (0.105)</td><td>+0.150 (0.024)</td><td>+1.393 (0.037)</td></tr><tr><td>1 before</td><td>+1.620 (0.162)</td><td>+1.441 (0.127)</td><td>+1.903 (0.107)</td></tr><tr><td>6 after</td><td>+1.140 (0.110)</td><td>+1.280 (0.086)</td><td>+1.333 (0.114)</td></tr></table>

Table S4: pretrain\_ft against from\_scratch and against zero\_shot, NHITS, 30 replicates. Positive favors fine-tuning. Each history level is tested on its 30 replicates with a Wilcoxon signed-rank test; “Levels” counts the six levels that reach significance after Holm correction within each family of 30 tests (5 frequencies × 6 history levels), one family per comparison. Mean and win rate are averaged over levels; † marks a frequency whose perlevel efect changes sign with history, for which no mean is given (per-level values with intervals are in Table S2). MDE is the minimum detectable efect at 80% power on the per-replicate percentage diferences, under a normal approximation, as a range across levels.
<table><tr><td rowspan="2"></td><td colspan="4">vs from_scratch</td><td colspan="3">VS zero_shot</td></tr><tr><td>Frequency Mean (%)</td><td>Win rate Levels MDE (%)</td><td></td><td></td><td></td><td>Mean (%) Win rate Levels</td><td></td></tr><tr><td>Quarterly</td><td>+12.96</td><td>0.97</td><td>6/6</td><td>2.0-2.9</td><td>†</td><td>0.17</td><td>6/6</td></tr><tr><td>Monthly</td><td>+4.98</td><td>0.91</td><td>5/6</td><td>1.1-1.6</td><td>†</td><td>0.17</td><td>6/6</td></tr><tr><td>Weekly</td><td>†</td><td>0.68</td><td>5/6</td><td>0.8-1.4</td><td>†</td><td>0.60</td><td>4/6</td></tr><tr><td>Daily</td><td>†</td><td>0.82</td><td>6/6</td><td>0.7-1.7</td><td>†</td><td>0.32</td><td>6/6</td></tr><tr><td>Hourly</td><td>-16.96</td><td>0.01</td><td>6/6</td><td>2.3-4.1</td><td>+46.29</td><td>1.00</td><td>6/6</td></tr></table>

Table S5: Sensitivity check E1b: improvement (%) of NHITS trained from scratch for 1,000 steps over 300 steps, mean over the ten replicates that ran it, by frequency and history level (nominal history level in parentheses). Positive favors 1,000 steps. Bold where the 95% replicate-bootstrap interval excludes zero. With ten replicates and Holm correction over 30 tests no level can reach significance (smallest attainable adjusted $p = 0 . 0 5 9 )$ , so the check is reported as estimates.
<table><tr><td colspan="2">Frequency</td><td colspan="4">History level</td></tr><tr><td>Quarterly</td><td>-2.4 (16)</td><td>-3.3 (20)</td><td>-3.6 (32)</td><td>-5.3 (48)</td><td>-3.1 (64)</td><td>+1.8 (full)</td></tr><tr><td>Monthly</td><td>-2.1 (36)</td><td>-3.4 (48)</td><td>-1.9 (72)</td><td>−0.4 (108)</td><td>-1.3 (144)</td><td>+2.2 (full)</td></tr><tr><td>Weekly</td><td>-2.3 (104)</td><td>-1.7 (156)</td><td>-2.4 (208)</td><td>-1.2 (260)</td><td>+0.5 (416)</td><td>+0.9 (full)</td></tr><tr><td>Daily</td><td>+1.9 (28)</td><td>-1.6 (56)</td><td>-3.0 (112)</td><td>-3.9 (182)</td><td>+0.3 (364)</td><td>-1.3 (full)</td></tr><tr><td>Hourly</td><td></td><td></td><td></td><td></td><td>+3.0 (96) +17.7 (144) +19.7 (240) +22.6 (384) +21.9 (672) +15.6 (full)</td><td></td></tr></table>

Table S6: Mean zero-shot MASE relative to mean Seasonal Naive MASE over 30 replicates, at the shortest and at full history, control arm.
<table><tr><td rowspan="2">Frequency</td><td colspan="2">NHITS</td><td colspan="2">LightGBM</td></tr><tr><td>Shortest</td><td>Full</td><td>Shortest</td><td>Full</td></tr><tr><td>Quarterly</td><td>1.12</td><td>1.08</td><td>1.11</td><td>1.17</td></tr><tr><td>Monthly</td><td>1.07</td><td>1.02</td><td>1.01</td><td>0.99</td></tr><tr><td>Weekly</td><td>0.67</td><td>0.56</td><td>0.69</td><td>0.68</td></tr><tr><td>Daily</td><td>0.95</td><td>0.94</td><td>0.93</td><td>1.12</td></tr><tr><td>Hourly</td><td>2.41</td><td>5.40</td><td>3.13</td><td>6.37</td></tr></table>

Table S7: Pre-specified expectations against outcomes.
<table><tr><td>Item</td><td>Expectation</td><td>Outcome</td><td>Held</td></tr><tr><td>Baselines</td><td>A statistical baseline is lowest in at least 8 of the 10 rows of Table 1 of the article</td><td>9 of 10</td><td>Yes</td></tr><tr><td>Crossover</td><td>Zero-shot beats scratch at the shortest history and loses at full history at four of five frequencies, hourly negative; the rule returns no collapse, history/horizon least</td><td>As expected, for both models</td><td>Yes</td></tr><tr><td rowspan="3">Fine-tuning</td><td>dispersed Fine-tuning beats scratch at quarterly at all six levels, at least four significant</td><td>Six of six significant, +13%</td><td>Yes</td></tr><tr><td>The quarterly effect is presented as reliable only if fine-tuning is nowhere significantly worse than zero-shot</td><td>Significantly worse at short histories</td><td>Not presented</td></tr><tr><td>The sign alternation of an earlier run at daily and hourly does not recur 1,000-step scratch beats</td><td>It does not</td><td>Yes</td></tr><tr><td>Budget (E1b)</td><td>300 at full history at four or more frequencies; the zero-shot advantage survives</td><td>Better at full history at hourly; at the other frequencies within ±2.2%, and worse at most shorter histories; the advantage survives</td><td>No</td></tr><tr><td>Ablation</td><td>No family replicates across both models outside hourly, on replicates 0–9</td><td>Classical replicates on 0–9; four families on 0-19</td><td>No</td></tr><tr><td>Hourly scaler</td><td>The window scaler stops the zero-shot/Seasonal Naive ratio growing (at most 1.2 times)</td><td>0.88 times</td><td>Yes</td></tr><tr><td>Coverage</td><td>Hourly coverage gap above 0.50; below 0.30 elsewhere Zero-shot bit-identical; 7</td><td>0.48; 0.07 to 0.26 elsewhere Zero-shot identical;</td><td>Partly</td></tr><tr><td>Run-to-run</td><td>trained-condition shifts below 0.1 replicate standard deviations</td><td>shifts up to 0.12 (scratch) and 0.52 (scratch, 1,000 steps), all at hourly</td><td>Partly</td></tr><tr><td>Cycles (follow-up)</td><td>Calendar cycles lower hourly zero-shot error at all six levels for</td><td>By 43 to 62% (NHITS) and 52 to 63% (LightGBM): replicates</td><td>Yes</td></tr></table>

## References

Al-Osh, M.A., Alzaid, A.A., 1987. First-order integer-valued autoregressive (INAR(1)) process. Journal of Time Series Analysis 8, 261–275. doi:10. 1111/j.1467-9892.1987.tb00438.x.

Ansari, A.F., Shchur, O., Küken, J., Auer, A., Han, B., Mercado, P., Rangapuram, S.S., Shen, H., Stella, L., Zhang, X., Goswami, M., Kapoor, S., Maddix, D.C., Guerron, P., Hu, T., Yin, J., Erickson, N., Desai, P.M., Wang, H., Rangwala, H., Karypis, G., Wang, Y., Bohlke-Schneider, M., 2025. Chronos-2: From univariate to universal forecasting. arXiv:2510.15821.

Ansari, A.F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S.S., Pineda Arango, S., Kapoor, S., Zschiegner, J., Maddix, D.C., Wang, H., Mahoney, M.W., Torkkola, K., Wilson, A.G., Bohlke-Schneider, M., Wang, Y., 2024. Chronos: Learning the language of time series. Transactions on Machine Learning Research ArXiv:2403.07815.

Bahrpeyma, F., Roantree, M., Cappellari, P., Scriney, M., McCarren, A., 2021. A methodology for validating diversity in synthetic time series generation. MethodsX 8, 101459. doi:10.1016/j.mex.2021.101459.

Black, F., Scholes, M., 1973. The pricing of options and corporate liabilities. Journal of Political Economy 81, 637–654. doi:10.1086/260062.

Bollerslev, T., 1986. Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics 31, 307–327. doi:10.1016/0304-4076(86) 90063-1.

Box, G.E.P., Jenkins, G.M., Reinsel, G.C., 2008. Time Series Analysis: Forecasting and Control. Wiley, Hoboken, NJ. doi:10.1002/9781118619193.

Chambers, J.M., Mallows, C.L., Stuck, B.W., 1976. A method for simulating stable random variables. Journal of the American Statistical Association 71, 340–344. doi:10.1080/01621459.1976.10480344.

Croston, J.D., 1972. Forecasting and stock control for intermittent demands. Operational Research Quarterly 23, 289–303. doi:10.2307/3007885.

Davies, R.B., Harte, D.S., 1987. Tests for Hurst efect. Biometrika 74, 95– 101. doi:10.1093/biomet/74.1.95.

Demarta, S., McNeil, A.J., 2005. The t copula and related copulas. International Statistical Review 73, 111–129. doi:10.1111/j.1751-5823.2005. tb00254.x.

Du, J.G., Li, Y., 1991. The integer-valued autoregressive (INAR(p)) model. Journal of Time Series Analysis 12, 129–142. doi:10.1111/j.1467-9892. 1991.tb00073.x.

Durbin, J., Koopman, S.J., 2012. Time Series Analysis by State Space Methods. 2nd ed., Oxford University Press, Oxford. doi:10.1093/acprof: oso/9780199641178.001.0001.

Hagan, P.S., Kumar, D., Lesniewski, A.S., Woodward, D.E., 2002. Managing smile risk. Wilmott Magazine URL: https://www.wilmott.com/ managing-smile-risk/.

Hamilton, J.D., 1989. A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica 57, 357–384. doi:10.2307/1912559.

Hawkes, A.G., 1971. Spectra of some self-exciting and mutually exciting point processes. Biometrika 58, 83–90. doi:10.1093/biomet/58.1.83.

Heston, S.L., 1993. A closed-form solution for options with stochastic volatility with applications to bond and currency options. The Review of Financial Studies 6, 327–343. doi:10.1093/rfs/6.2.327.

Hyndman, R.J., Koehler, A.B., Ord, J.K., Snyder, R.D., 2008. Forecasting with Exponential Smoothing: The State Space Approach. Springer Series in Statistics, Springer, Berlin. doi:10.1007/978-3-540-71918-2.

Hyndman, R.J., Koehler, A.B., Snyder, R.D., Grose, S., 2002. A state space framework for automatic forecasting using exponential smoothing methods. International Journal of Forecasting 18, 439–454. doi:10.1016/ S0169-2070(01)00110-8.

Kang, Y., Hyndman, R.J., Li, F., 2020. GRATIS: GeneRAting TIme Series with diverse and controllable characteristics. Statistical Analysis and Data Mining 13, 354–376. doi:10.1002/sam.11461.

Lorenz, E.N., 1963. Deterministic nonperiodic flow. Journal of the Atmospheric Sciences 20, 130–141. doi:10.1175/1520-0469(1963)020<0130: DNF>2.0.CO;2.

Lütkepohl, H., 2005. New Introduction to Multiple Time Series Analysis. Springer, Berlin. doi:10.1007/978-3-540-27752-1.

Mackey, M.C., Glass, L., 1977. Oscillation and chaos in physiological control systems. Science 197, 287–289. doi:10.1126/science.267326.

Mandelbrot, B.B., Van Ness, J.W., 1968. Fractional Brownian motions, fractional noises and applications. SIAM Review 10, 422–437. doi:10.1137/ 1010093.

May, R.M., 1976. Simple mathematical models with very complicated dynamics. Nature 261, 459–467. doi:10.1038/261459a0.

Merton, R.C., 1976. Option pricing when underlying stock returns are discontinuous. Journal of Financial Economics 3, 125–144. doi:10.1016/ 0304-405X(76)90022-2.

Nelsen, R.B., 2006. An Introduction to Copulas. Springer Series in Statistics. 2nd ed., Springer, New York. doi:10.1007/0-387-28678-0.

Rasmussen, C.E., Williams, C.K.I., 2006. Gaussian Processes for Machine Learning. MIT Press, Cambridge, MA. doi:10.7551/mitpress/3206. 001.0001.

Rocha, A.V., Cribari-Neto, F., 2009. Beta autoregressive moving average models. TEST 18, 529–545. doi:10.1007/s11749-008-0112-z.

Runge, J., Gerhardus, A., Varando, G., Eyring, V., Camps-Valls, G., 2023. Causal inference for time series. Nature Reviews Earth & Environment 4, 487–505. doi:10.1038/s43017-023-00431-y.

Steutel, F.W., van Harn, K., 1979. Discrete analogues of self-decomposability and stability. The Annals of Probability 7, 893–899. doi:10.1214/aop/ 1176994950.

Uhlenbeck, G.E., Ornstein, L.S., 1930. On the theory of the Brownian motion. Physical Review 36, 823–841. doi:10.1103/PhysRev.36.823.

Wong, C.S., Li, W.K., 2000. On a mixture autoregressive model. Journal of the Royal Statistical Society: Series B 62, 95–115. doi:10.1111/ 1467-9868.00222.