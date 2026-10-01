# The Nixtlaverse: An Open-Source Ecosystem for Forecasting<sup>⋆</sup>

Olivier Sprangers<sup>a,∗</sup>, Max Mergenthaler Canseco<sup>a</sup>, Marco Peixeiro<sup>a</sup>, Saul Caballero Ramirez<sup>a</sup>, Mariana Menchero García<sup>a</sup>, Jing-Qiang Goh<sup>a</sup>, Han Wang<sup>a</sup>, Nikhil Gupta<sup>a</sup>, Rogelio Melo<sup>a</sup>, Senbong Gee<sup>a</sup>, Cristian Challu<sup>a</sup>

<sup>a</sup>Nixtla, San Francisco, CA, United States

## Abstract

Large forecasting applications often combine statistical, machine-learning, and neural models. These model families solve the same problem, but they difer in their fitted state, their training procedures, and the way they parallelize work. Forecasting software must therefore either hide these diferences behind a single estimator interface, or keep the families in separate packages, which forces users to rewrite data preparation and evaluation for every package.

In this paper, we present the Nixtlaverse, an ecosystem of open-source Python libraries for time series forecasting, as2 a case study of a third design: all libraries share the same long-format panel data and keyed forecast outputs, while every model family keeps its own specialized implementation. We demonstrate the behavior and benefits of this design throughe three use cases on the public M5 competition data. First, we evaluate statistical, machine-learning, and neural models,<sup>S</sup> as well as an external engine from a separate ecosystem, in a single rolling-origin evaluation. The shared outputs make<sub>0</sub> per-series and hierarchy-weighted accuracy metrics cheap to compute side-by-side. Second, we profile runtime and peak3

memory from 100 to 30,490 series and locate the bottleneck of each family: statistical fitting scales approximately linearly in the number of series, feature construction dominates machine-learning memory, and neural training time is nearly independent of panel size under a fixed training budget. Third, we reconcile the forecasts of multiple engines, including the external one, over all 42,840 series of the M5 hierarchy, using sparse reconciliation where dense implementations exhausted the memory of our machine.<sup>s</sup>

c These use cases establish the costs and boundaries of the design and demonstrate the utility of shared data and output contracts for cross-family evaluation and reconciliation. The success of these design choices, and by extension the Nixtlaverse itself, has been validated by the substantial public distribution, scholarly reuse, and adoption through other forecasting frameworks. The Nixtlaverse is released under permissive open-source licenses, with public datasets, reproducible examples, and verifiable benchmark artifacts.<sub>4</sub>

Keywords: forecasting software, open-source frameworks, panel time series, hierarchical forecasting, reproducibility,<sub>9</sub> scalability<sub>3</sub>

## 1. Introduction

Large forecasting applications require more than fitting a model: practitioners prepare temporal panels, generate multi-step forecasts, evaluate methods over histori-X cal origins, and often aggregate forecasts for downstreamr decisions, frequently combining local statistical methods, global regressors, and neural models. These families solve the same problem but difer in fitted state, feature construction, training loops, preferred hardware, and natural unit of parallelism. Software can expose them through one estimator abstraction, which must accommodate those differing semantics, or through independent packages, which forces users to rewrite data preparation and evaluation. We study a third design that shares panel and forecastoutput contracts while retaining specialized model-family implementations.

We state three design commitments. First, shared panel and output contracts can support cross-family evaluation even when model construction and behavioral defaults remain engine-specific. Second, scalability should be implemented within each model family because its bottleneck and natural unit of parallelism are properties of the method. Third, downstream operations should consume keyed forecasts rather than fitted models, so that evaluation and reconciliation remain independent of the producing engine. The first commitment applies a general interface principle to forecasting by standardizing a narrow data contract while leaving implementations free behind it. What makes the contract forecasting-specific is its explicit representation of series, time, and forecast

origin.

The Nixtlaverse is a collection of interoperable forecasting libraries composed of StatsForecast, MLForecast, NeuralForecast, CoreForecast, UtilsForecast, HierarchicalForecast, and DatasetsForecast. Together, they implement these commitments through a shared long-panel representation and keyed forecast dataframes.<sup>1</sup> StatsForecast (Garza Ramirez et al., 2022), MLForecast, and NeuralForecast (Olivares et al., 2022) provide specialized forecasting engines; CoreForecast supplies grouped numerical operations; and UtilsForecast and HierarchicalForecast (Olivares et al., 2024) consume forecast outputs for evaluation and reconciliation.

We evaluate these commitments through three concrete use cases on the public M5 dataset (Makridakis et al., 2022). Each represents a task faced by forecasting practitioners and examines the behavior and boundaries of one commitment. The first applies a common rolling-origin evaluation across all three model families and an engine external to the ecosystem (Section 4). The second profiles resource use as a workflow scales from a 100-series pilot to the complete 30,490-series panel (Section 5). The third performs hierarchical reconciliation over all 42,840 series in the M5 hierarchy, independently of the engines that produced the base forecasts (Section 6). The complete experimental protocol, including datasets, method configurations, and measurement procedures, is provided in Appendix A.

The contributions of this work are:

1. a comprehensive model-family boundary that specifies what the shared panel, input and output contracts guarantee, with functional semantics (e.g., training, forecasting, reconciling) deliberately remaining engine-specific;

2. measurements, organized as an evaluation use case (Section 4) and a scaling use case (Section 5), connecting that boundary to common rolling-origin evaluation across heterogeneous engines and to the distinct runtime and memory bottlenecks of local statistical, global machine-learning, and global neural methods;

3. a hierarchical use case (Section 6) demonstrating that output-level interoperability supports common metrics and sparse hierarchical reconciliation while exposing diferences between per-series, weighted, and coherent forecasts, and the memory boundary at which the tested dense reconcilers stop being feasible. In addition, as a direct test of extensibility, we tested an engine external to the ecosystem entering both the evaluation and reconciliation paths through the same keys; and

4. a descriptive contribution: a license and publicinfrastructure inventory of the ecosystem, and an account of the release-coordination and reproducibility obligations its multi-repository design implies.

## 2. Related Work

Open-source forecasting frameworks difer in which parts of the forecasting workflow they standardize. The R forecast package established automatic local modeling (Hyndman and Khandakar, 2008), while tidyverts combines indexed tsibble data with model and forecast tables (Wang et al., 2020; O’Hara-Wild et al., 2026b,a). In Python, Prophet exposes a single decomposable model through a small interface (Taylor and Letham, 2018); GluonTS focuses on probabilistic and neural components (Alexandrov et al., 2020); Darts exposes multiple model families through one interface (Herzen et al., 2022); sktime uses a common forecaster interface with capability tags (Löning et al., 2019); and AutoGluon–TimeSeries places statistical, tree-based, and neural models behind a single automated selection interface (Shchur et al., 2023). Pretrained forecasting models provide another workflow in which the target panel need not be used for training (Ansari et al., 2024; Das et al., 2024); and have to be considered outside the scope of this work.

Statistical, machine-learning and neural forecasting methods are not interchangeable implementations of one identical abstraction. Januschowski et al. (2020) note that the distinction between the aforementioned model families is mostly of tribal nature, and discuss objective and subjective dimensions to categorize forecasting methods, such as local versus global parameterization, or complexity. These dimensions cut across the boundaries a common estimator interface must hide. Petropoulos et al. (2022, §2.7.10) document the distinct data, computational, and tuning requirements the machine-learning and neural families impose relative to statistical methods. The local–global boundary is supported by both theory and empirical evidence. Montero-Manso and Hyndman (2021) establish the conditions under which a single global function fitted across a panel can match or outperform per-series models, while Semenoglou et al. (2021) examine the behavior of such cross-learning empirically. Because this boundary is a property of the method, Section 3 treats the unit of parallelism as a property of the method as well. Evaluation practice for global and deep models likewise requires care over training budgets, input windows, and comparability of tuning efort (Hewamalage et al., 2021, §§4.2.5, 4.5, 5.9); our fixed-budget configurations are motivated by that guidance and inherit its limitations. Our contribution is not to show that these families difer, but to measure where the diference surfaces in software boundaries, and to enable the community to repeat that measurement through the released artifact.

The forecasting frameworks discussed above difer along two independent design axes, one concerning data representation and the other model exposure. A dedicated temporal object such as a tsibble can validate frequency and alignment when constructed, while a generic dataframe integrates directly with existing data systems but leaves those checks to the user. Similarly, a common estimator interface presents every model family uniformly, whereas separate engines preserve the training and scaling behavior of each family. The Nixtlaverse uses the general dataframe on the first axis and separate engines on the second, and recovers what is shared, such as evaluation and reconciliation, at the level of keyed outputs instead. Treating the input of the diferent engines as a generic dataframe abstraction brings positive outcomes like enabling flexibility around diferent instances of that abstraction, such as pandas, Dask, or even Spark dataframes. The consequences of this design choice become visible in Section 4, where behavioral defaults difer across engines and the forecast horizon is specified per call in some engines and per model in others.

Rolling-origin evaluation measures performance over multiple historical origins (Tashman, 2000), with special care required when target-derived features are recreated (Bergmeir et al., 2018). Curated multi-dataset archives such as that of Godahewa et al. (2021) standardize the data side of benchmarking; the boundary we examine is the software side that consumes such data. Hierarchical reconciliation is similarly downstream of model fitting: it adjusts base forecasts to satisfy aggregation constraints (Hyndman et al., 2011; Wickramasuriya et al., 2019); Athanasopoulos et al. (2024) survey the field, including the probabilistic extensions our point-forecast experiments do not reach. Frameworks also difer in where reconciliation attaches. fabletools attaches reconciliation to a table of fitted models, producing coherent forecasts when the table is forecast (O’Hara-Wild et al., 2026b), which requires participating engines to live inside the framework. The design studied here instead reconciles keyed forecasts, which brings independence from the producing engine. We use these tasks to test whether one output representation can support common evaluation and reconciliation across model families. FoReco (Girolimetto and Di Fonzo, 2026) likewise reconciles base forecasts independently of the producing engine, supplied as positionally aligned numeric matrices; the design studied here difers in identifying series by explicit keys in the engines’ native output format rather than by matrix position.

The prior descriptions of the individual libraries (Garza Ramirez et al., 2022; Olivares et al., 2022, 2024) document each package’s models and interface. What they do not contain, and this paper adds, is the ecosystem-level analysis, including an explicit statement of the shared data and output contracts and their boundary, phase-level runtime and memory measurements across panel sizes for all three families under one protocol, cross-engine reconciliation over the complete M5 hierarchy with its feasibility limits, an interoperability test with an engine external to the ecosystem, and a versioned, verifiable benchmark artifact.

## 3. The Nixtlaverse

## 3.1. Design requirements

Local statistical methods learn separate state per series and parallelize across a panel; global machine-learning methods materialize lag and calendar features before fitting a shared regressor; neural methods train from batches of temporal windows, often on an accelerator such as a graphics processing unit (GPU). A common workflow must therefore preserve series, timestamps, covariates, forecast origins, observations, and predictions without requiring these families to share fitted state or a scaling strategy. Historically, forecasting software was designed around automatic modeling of one series at a time (Hyndman and Khandakar, 2008), and forecasting across large panels was recognized as a software problem in its own right only later (Taylor and Letham, 2018); we therefore treat scalability as an explicit, per-family design requirement.

## 3.2. Panel representation

The Nixtlaverse libraries represent a temporal panel as a long dataframe with minimum columns (unique\_id, ds, y) for the series, timestamp, and target. Additional columns are optional and might contain static or timevarying covariates. This format supports unequal history lengths and partitioning with common dataframe and distributed-computing systems.

Explicit series and time keys avoid dependence on row position or a rectangular array and let forecasts be joined directly to observations. A general dataframe cannot, however, infer whether a covariate is available beyond the forecast origin, or prevent every alignment error; the libraries therefore require model-specific declarations for static and future covariates. The trade-of keeps the panel simple while making assumptions explicit.

## 3.3. The contract

The shared contract is the boundary this paper examines, at the versions in Table 1.

Input. The shared input is a Python dataframe, such as pandas, polars, Spark, or Dask, containing a series identifier, a timestamp or integer time column, and a numeric target. Any additional columns represent covariates, whose static or future status is declared through the corresponding engine rather than inferred from the dataframe. UtilsForecast provides the common validation of frame type, required columns, and time and target dtypes, which is the only validation the contract guarantees, while each engine remains responsible for any additional formal or semantic validation and may enforce it to a diferent extent. Maintaining a consistent contract therefore requires engine-independent checks to be consolidated in UtilsForecast, although the separate-library design can allow that convergence to be deferred (Section 3.7).

Output. A dataframe keyed by series and timestamp (plus the forecast origin under rolling-origin evaluation), with one numeric column per model and h rows per series and origin, the identifiers drawn from the input panel and the timestamps continuing it at the declared frequency.

Conformance. An external engine is a conforming producer for the downstream components if it emits exactly the output frame above; the external-engine test of Section 4 is the checklist executed (setup in Appendix A.6). In principle, evaluation requires nothing further, however, specific evaluation scenarios may require additional information. For example, some reconcilers from HierarchicalForecast may require an in-sample frame of the same shape (series, timestamp, observed target, and fitted value per model), and evaluating with some loss functions from UtilsForecast requires distributional outputs or a seasonal period. Thus, beyond the output frame itself, what a producer must supply depends on the furthest downstream component it feeds.

Evolution. The contract serves as the ecosystem’s public interface and remains unchanged across the release range examined here, while its use by six of the seven packages means that any modification to the required columns or key semantics constitutes a breaking change that the multi-repository design must coordinate manually, as discussed in Section 3.7.

## 3.4. Nixtla’s forecasting libraries

StatsForecast implements local statistical and econometric models using compiled numerical routines and parallel or distributed execution over series (Garza Ramirez et al., 2022); the automatic exponential-smoothing model we use follows the state-space selection framework of Hyndman et al. (2002). MLForecast creates temporal features for general regressors and manages recursive or direct multi-step prediction. NeuralForecast samples windows for global neural architectures and supports point and probabilistic losses (Olivares et al., 2022). Their constructors remain family-specific, but each accepts a long panel and returns a keyed dataframe with one column per model.

## 3.5. Shared components

Figure 1 provides an overview of the ecosystem. Datasets and long panel data enter the forecasting libraries from the left. The libraries return keyed forecast dataframes, which are consumed by evaluation and hierarchical reconciliation on the right.

UtilsForecast evaluates keyed outputs and supplies pre-processing utilities for the long panel, most importantly fill\_gaps, which inserts the rows needed to make every series contiguous between its first and last observation, an assumption the engines make of the input panel. These utilities, together with its plotting helpers, operate on any conforming panel and are usable independently of the forecasting libraries. DatasetsForecast supplies benchmark loaders, and HierarchicalForecast provides forecast reconciliation strategies as a post-processing step (Olivares et al., 2024). CoreForecast provides compiled grouped operations that the three forecasting engines build on.

## 3.6. Scaling strategies

The modular architecture reflects how the methods scale. StatsForecast partitions independent series fits and uses compiled routines inside each, so the dominant work grows with the number of series and the complexity of the local model. MLForecast constructs features over the complete panel before fitting a shared estimator, and either stage can dominate depending on the regressor. NeuralForecast samples and batches temporal windows, so its cost depends on the number and length of windows, the architecture, and accelerator utilization. HierarchicalForecast introduces cross-series matrix operations that are typically not independent across series, because the aggregation structure connects them.

The same diferences determine how work is distributed once a panel no longer fits one machine. Because every row carries its keys, the panel can be partitioned without positional structure: StatsForecast partitions series fits through Spark (Zaharia et al., 2016), Dask (Rocklin, 2015), or Ray (Moritz et al., 2018), dispatching to whichever backend holds the dataframe through the engine-agnostic Fugue layer (Wang et al., 2022); MLForecast distributes feature construction but also needs a distributed estimator; NeuralForecast instead supports accelerator, multidevice, and Spark-based multi-node training of one global model. Because these strategies are operationally distinct despite similar workflow methods, we measure preparation, fitting, prediction, evaluation, and reconciliation separately rather than treating one parallelization parameter as a common measure of scalability.

## 3.7. Open-source implementation and coordination

Separate libraries allow users of statistical models to avoid installing a deep-learning stack, preserve modelspecific choices in the public interfaces, and enable shared numerical and evaluation components to be reused without placing every model family on the same release cycle. Table 1 documents the open-source foundations of the case study.

The multi-repository design creates a corresponding coordination burden because data conventions, frequency handling, defaults, output schemas, and dependency versions must be aligned manually across releases, while crosslibrary tests cover only the combinations they exercise. A unified framework, such as the estimator-interface designs discussed in Section 2, centralizes this coordination but also couples every model family to a common dependency stack and release cadence; the design studied here instead accepts manual coordination to preserve their independence.

Table 1: Open-source components used in the paper. Reported package versions and their release licenses were recorded together with the public project assets in July 2026 from the projects’ public repositories (URL on the title page). “Guide” denotes a repository-leve contribution guide. The artifact records the version-pinned dependency closure used by the benchmark.
<table><tr><td>Component</td><td>Version</td><td>Licence</td><td>Public project assets</td></tr><tr><td>StatsForecast</td><td>2.0.3</td><td>Apache-2.0</td><td>source, documentation, tests/CI, guide</td></tr><tr><td>MLForecast</td><td>1.0.31</td><td>Apache-2.0</td><td>source, documentation, tests/CI, guide</td></tr><tr><td>NeuralForecast</td><td>3.2.0</td><td>Apache-2.0</td><td>source, documentation, tests/CI, guide</td></tr><tr><td>HierarchicalForecast</td><td>1.3.1</td><td>Apache-2.0</td><td>source, documentation, tests/CI, guide</td></tr><tr><td>CoreForecast</td><td>0.0.18</td><td>Apache-2.0</td><td>source, documentation, tests/CI</td></tr><tr><td>UtilsForecast</td><td>0.2.16</td><td>Apache-2.0</td><td>source, documentation, tests/CI, guide</td></tr><tr><td>DatasetsForecast</td><td>1.0.0</td><td>MIT</td><td>source and dataset loaders</td></tr></table>

![](images/bd0d009eedc8d9ca04f35b3debc4c0937010ddcf02266a7d97ca87ca2775ec6f.jpg)  
Figure 1: Overview of the open-source Nixtlaverse. Packages are drawn as sharp-cornered boxes and the shared data contracts as rounded boxes; the legend gives the meaning of the fills. The forecasting engines consume the same long panel representation, optionally pre-processed by UtilsForecast (e.g. filling date gaps), and return keyed forecast dataframes, which downstream evaluation and reconciliation consume regardless of the producing engine. The dashed line indicates that the three engines build on the compiled grouped operations of CoreForecast.

## 3.8. Observable reach and downstream adoption

The public infrastructure described above also provides partial evidence about the ecosystem’s reach. To assess the impact of the Nixtlaverse, we measured several indicators: repository stars, package downloads, citations, and public case studies. We report these signals separately and attach each measurement to a fixed snapshot date.

Table 2 reports public indicators for the four principal user-facing libraries as retrieved on 21 August 2026. The download counts should not be interpreted as three million distinct users or summed into an ecosystem-wide adoption figure. They include repeated installations, automated build and test environments, and package caches. Moreover, installations of the forecasting libraries also generate downloads of shared dependencies such as CoreForecast and UtilsForecast. The figures demonstrate sustained distribution activity, not the number of people or organizations using the software.

Reuse outside the Nixtlaverse provides a more structural form of adoption. At least four independently maintained forecasting projects expose Nixtlaverse implementations or data adapters. Darts (Herzen et al., 2022) provides a general wrapper around StatsForecast models and an interface for NeuralForecast models. sktime (Löning et al., 2019) supplies adapters for several StatsForecast statistical models and NeuralForecast architectures. AutoGluon–TimeSeries (Shchur et al., 2023) bases a family of statistical and intermittentdemand models, including AutoARIMA, ETS, CES, Theta, ADIDA, Croston and IMAPA, on StatsForecast implementations. The FEV benchmarking library (Shchur et al., 2025) separately provides adapters that convert evaluation data into the Nixtlaverse representation. These integrations place parts of the ecosystem behind external APIs and benchmark contracts maintained under separate release cycles. They therefore provide stronger evidence of downstream reuse than repository attention alone.

Embedding in independently maintained products extends beyond forecasting libraries. The Databricks Many Model Forecasting solution accelerator builds dedicated StatsForecast, MLForecast, and NeuralForecast pipelines (Databricks Industry Solutions, 2026); the time-series database TDengine implements several of the statistical forecasting algorithms of its analytics component TDgpt, including ETS, Theta, and complex exponential smoothing, on StatsForecast (TAOS Data, 2026); and Lightwood, the AutoML engine behind MindsDB, declares StatsForecast as a required dependency and builds its ARIMA time-series mixer on it (MindsDB, 2026). The AutoML platform PyCaret (Ali, 2020) exposes StatsForecast’s AutoARIMA as an optional engine and, like Lightwood, reaches it through sktime’s adapter, an instance of second-order reuse in which one framework’s adapter becomes another framework’s integration path. Such chains extend into model serving: the KServe inference platform ships an Auto-Gluon model server that declares AutoGluon–TimeSeries as a dependency and thereby carries StatsForecast and MLForecast into its dependency closure (The KServe Authors, 2026). These are declared dependencies and shipped integrations, verified in each project’s public repository on 28 August 2026; like the indicators of Table 2, they evidence embedding, not usage volume.

The Nixtlaverse has also entered the scholarly record. Some examples include retail demand forecasting (Oliveira and Ramos, 2024), electricity-load forecasting (Delgado Fernandez et al., 2025), hydrological-flow forecasting (Muniz et al., 2026), and financial-volatility forecasting (Souto, 2026). The citation counts in Table 2 also illustrate a known weakness in software citation practice: MLForecast had no independently indexed software record on the snapshot date, despite appearing in applied studies and software comparisons.

Independent assessments and downstream reuse provide qualitative context to the telemetry. Thoughtworks placed MLForecast in the “Trial” ring of its November 2025 Technology Radar (Thoughtworks, 2025, p. 42). It reported that MLForecast scaled eficiently to millions of data points and consistently outperformed comparable tools in its evaluation, and described it as a compelling choice for teams operationalizing high-volume forecasting. This is evidence from one organization rather than a representative user survey, but its assessment specifically highlights automated feature construction and distributed execution as practically valuable capabilities. A further single-source signal comes from practitioner education. The textbook Modern Time Series Forecasting with Python (Joseph and Tackes, 2024), for example, uses StatsForecast for its single-step backtesting baselines and introduces NeuralForecast in its chapter on specialized deep-learning architectures.

A practitioner case study from the German data consultancy m2hycon describes selecting StatsForecast for a customer problem involving rare component-failure events, modeled separately by customer and component, because it exposed Croston variants, IMAPA, ADIDA, and TSB through a concise common interface (Windler, 2024). The report states that this enabled rapid model setup and tuning and describes the resulting approach as a robust solution for the intermittent-event problem.

Independent reuse is also visible at the framework level. AutoGluon–TimeSeries (Shchur et al., 2023, pp. 4, 16), developed by researchers at Amazon Web Services, relies on StatsForecast for its local statistical models and on MLForecast to construct its tabular forecasters. These components are included in AutoGluon’s best\_quality preset and therefore participate in its automated modelselection and ensemble workflow.

Taken together, the Nixtlaverse has achieved substantial public distribution, documented scholarly reuse, and adoption through other forecasting frameworks.

Table 2: Public indicators of reach for the principal Nixtlaverse libraries, retrieved on 21 August 2026. Downloads are distribution events recorded by PyPI during the preceding 30 days. Citation counts are those displayed for the canonical software records in Google Scholar; MLForecast did not have a separately indexed software record. These indicators are not necessarily counts of distinct users, deployments, or citing works across the ecosystem.
<table><tr><td>Library</td><td>GitHub stars</td><td>PyPI downloads</td><td>Google Scholar citations</td></tr><tr><td>StatsForecast</td><td>4,876</td><td>1,942,775</td><td>135</td></tr><tr><td>MLForecast</td><td>1,269</td><td>505,473</td><td></td></tr><tr><td>NeuralForecast</td><td>4,248</td><td>285,120</td><td>146</td></tr><tr><td>HierarchicalForecast</td><td>756</td><td>289,221</td><td>29</td></tr></table>

## 4. Use case 1: five models from four engines

A forecasting team selecting a method for a retail demand panel faces a comparison across model families: statistical baselines, a gradient-boosted regressor, a neural architecture, and often a legacy method maintained in another ecosystem entirely. The comparison is meaningful only if every candidate is evaluated under the same protocol on the same panel. In practice, such comparisons are expensive because every framework requires its own data preparation, re-indexing, and schema translation. This use case runs that task across the three resident engines (StatsForecast, MLForecast and NeuralForecast) and one foreign engine, and demonstrates the first commitment: shared panel and output contracts support crossfamily evaluation even when model construction and behavioral defaults remain engine-specific. The benefit of the shared contracts is that practitioners can run this comparison with one merge and one evaluation call.

The panel is the M5 competition data, consisting of daily unit sales from ten Walmart stores in three US states (Makridakis et al., 2022), containing 30,490 bottom-level series and 47.6M observations after leading zero-sales periods are removed per series, with the competition horizon of 28 days. All methods use the same three non-overlapping forecast origins, with re-estimation at every origin. The methods are a small representative set rather than a benchmark of every model in the ecosystem. We include the Seasonal Naive and AutoETS (Hyndman et al., 2002) from StatsForecast; LightGBM (Ke et al., 2017) from MLForecast, the model class that dominated the M5 accuracy competition (Januschowski et al., 2022), with fixed target lags and date features; and NHITS (Challu et al., 2023) from NeuralForecast with a fixed context length and training budget. No method-specific hyperparameter optimization is performed; the exact configurations are in Appendix A.2. The external engine is Croston’s method (Croston, 1972) as implemented in sktime 1.1.0 (Löning et al., 2019), an intermittent-demand method appropriate for M5 from a codebase with no shared history with the libraries under study, run in a separate virtual environment with no Nixtla package installed (Appendix A.6). We compute MASE (Hyndman and Koehler, 2006) and RMSE as unweighted means over series and origins and, at the final origin, the competition’s weighted root mean squared scaled error (WRMSSE) over all 42,840 series at the twelve M5 hierarchy levels (Makridakis et al., 2022). Metric details and the one-origin rationale for WRMSSE are in Appendix A.3.

## 4.1. One panel, three engines

Figure 2 shows the shared part of the rolling-origin workflow. Each engine receives the same long dataframe and returns forecasts keyed by series, timestamp, and origin. The outputs are joined directly and scored in one evaluation call.

Model construction remains specific to each family, with statistical models requiring a seasonal period, the machine-learning engine requiring features and a regressor, and the neural model requiring an architecture, context, horizon, and training budget. To ensure that evaluation workflows are defined consistently across engines, we recommend specifying the origin spacing and refitting policy explicitly, as shown in Figure 2. The shared data contract makes the resulting outputs composable while keeping these behavioral diferences visible.

The workflow in Figure 2 could be executed without engine-specific reshaping or schema translation using three shared contracts, namely a long input panel keyed by series and timestamp, an explicit rolling-origin request, and forecast outputs keyed by series, timestamp, and origin. These outputs could be joined and scored in a single evaluation call, with the remaining code limited to dropping the duplicated observed-target column from two frames before the join. Model-specific requirements remained confined to the constructors, including a seasonal period for statistical models, feature definitions and a regression estimator for machine learning, and the architecture, context length, and training budget for the neural engine. One remaining workflow diference was how the forecast horizon was specified, with two engines accepting it per call and the third encoding it in the model. The contracts therefore did not make the estimators interchangeable.

Shared keys do not eliminate every coordination point because default argument values are not synchronized across libraries, requiring both the spacing between origins and the refitting policy to be specified explicitly, as shown in Figure 2. This reflects the trade-of described in Section 3, whereby the shared data contract guarantees that outputs can be combined while each library retains responsibility for its behavioral defaults.

```python
from datasetsforecast m5 import M5
from statsforecast import StatsForecast
from mlforecast import MLForecast
from neuralforecast import NeuralForecast
from statsforecast.models import AutoETS, SeasonalNaive
from lightgbm import LGBMRegressor
from neuralforecast.models import NHITS
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mase, rmse
from functools import partial
h, season, freq = 28, 7, "D"
df, *_ = M5.load(directory="data") # long panel (unique_id, ds, y)
sf = StatsForecast(freq=freq,
models=[AutoETS(season_length=season),
SeasonalNaive(season_length=season)])
mf = MLForecast(freq=freq,
models=[LGBMRegressor()],
lags=[7, 14, 28], date_features=["dayofweek", "month"])
nf = NeuralForecast(freq=freq,
models=[NHITS(h=h, input_size=2 * h, max_steps=1000)])
cv_sf = sf.cross_validation(
df=df, h=h, n_windows=3, step_size=h, refit=True)
cv_ml = mf.cross_validation(
df=df, h=h, n_windows=3, step_size=h, refit=True)
cv_nf = nf.cross_validation(
df=df, n_windows=3, step_size=h, refit=True)
keys = ["unique_id", "ds", "cutoff"]
cv = (cv_sf.merge(cv_ml.drop(columns="y"), on=keys)
.merge(cv_nf.drop(columns="y"), on=keys))
metrics = [partial(mase, seasonality=season), rmse]
scores = evaluate(cv, metrics=metrics, train_df=df)
```  
Figure 2: The M5 rolling-origin workflow: one panel, three family-specific constructors, shared cross-validation calls, a keyed merge, and one evaluation call. Seeds and logging are omitted; the executable configuration is provided in the benchmark artifact.

We run the same workflow with an external engine, Croston’s method from sktime, to demonstrate suficiency of the output contract. The cost of ‘admission’ is a 121-line adapter that reads the panel, fits the model per series and origin, and emits the keyed frame; the adapter is included in the benchmark artifact. Its forecasts then pass through the merge and evaluation call of Figure 2 unchanged, scoring four engines from two ecosystems side by side.

## 4.2. Benefits of output-level interoperability

Table 3 reports the rolling-origin results on the complete panel. It is not a comparison of methods: the table contains a small, deliberately untuned selection of models, and none approaches the tuned ensembles that won the competition. The table characterizes what one merged frame makes visible about five fixed configurations, one of them produced outside the Nixtlaverse, and its value lies in the columns disagreeing.

NHITS attains the best per-series MASE but the worst hierarchical WRMSSE, while the approximately unbiased AutoETS shows the opposite pattern. This reversal is consistent with the evaluation objectives because absolute-error training targets conditional medians, which are often zero for intermittent retail demand, while an unweighted absolute-error metric can favor behavior that a sales-weighted squared-error metric penalizes (Kolassa, 2016). Across the three origins, NHITS predicts only 66–70% of observed demand in aggregate and produces 11.7–16.7% negative forecasts, causing bottom-level errors to accumulate into downward bias at sales-weighted upper levels. Although this pattern is consistent with its default mean-absolute-error loss, establishing causality would require refitting under a loss targeting a diferent central tendency.

The ordering is largely stable under resampling and retraining. Across models, bootstrap intervals overlap only for AutoETS and Croston, while three additional NHITS seeds change mean MASE by at most 0.002, keep the aggregate ratio between 0.66 and 0.73, and produce WRMSSE values between 1.732 and 1.888, still far from the next engine. The shared output boundary makes both metrics straightforward to compute on the same forecasts, exposing a disagreement that either metric alone would conceal.

The disagreement also reveals a limit of the sharedoutput boundary. MASE and RMSE can be computed from the merged forecast frame through a single UtilsForecast evaluation call, whereas WRMSSE requires a separate workflow that pivots the final origin, joins the oficial product hierarchy, and invokes the competition evaluator. The shared schema makes the per-series and hierarchy-weighted results directly comparable, but the latter requires more efort to obtain, while the more readily available per-series metric ranks NHITS first. Relying on that ranking alone would select a model whose forecasts sum to only 70% of observed demand, a serious limitation for inventory or revenue planning regardless of its per-series score. Output interoperability standardizes the data exchanged between components, but it does not determine which metric or weighting scheme should guide model selection.

The shared frame also makes post-processing sensitivity checks straightforward. We evaluate the raw forecasts to avoid model-specific adjustments, but clipping negative predictions at zero changes WRMSSE only from 0.673 to 0.670 for AutoETS and from 1.888 to 1.885 for NHITS, without altering any model ranking. The disagreement between the metrics is therefore not an artifact of evaluating unadjusted outputs.

Limitations. This use case exercises only target lags and calendar features, not the static- and future-covariate declarations that Section 3 identifies as the contract’s errorprone edge. Because the configurations are untuned and the neural model has a fixed step budget, Table 3 compares equal tuning efort rather than converged or equalexposure training and does not rank the underlying methods (Appendix A.2 states the consequences). The external test covers one method from one engine through an adapter written by us; it establishes that the contract admits a foreign producer, not that every external engine maps this cheaply.

## 5. Use case 2: scaling from a pilot to the full panel

A team moving from a pilot of one hundred series to the complete panel must determine which computational phase will dominate, how much memory to provision, how many workers to use, and when a distributed backend justifies its overhead. Answering these questions requires measurements by phase and model family because the second commitment assigns scalability to each family, making both the bottleneck and the natural unit of parallelism properties of the underlying method. These measurements allow practitioners to provision resources for the phase that will dominate at their target scale.

We create nested subsets of the main panel with an increasing number of series, drawn by seeded sampling proportional in product category and store so that each subset approximates the composition of the complete panel, and reserve the final 28 days of each series as a common holdout. Data preparation, fitting, prediction, and evaluation are measured separately, recording wall-clock time and peak memory as aggregate proportional set size (PSS) over parent and worker processes; the panel-size sweep uses five measured repetitions per configuration, each in a fresh process, and we report medians. These subsets profile resources only; comparative accuracy is reported on the complete panel in Section 4.2. The full measurement protocol, including the subset-composition rationale and the treatment of timeouts and out-of-memory kills, is in Appendix A.4.

Figure 3 separates the runtime of every engine into its computational phases as the panel grows from 100 to

![](images/e005ee0ade932203ae0b243ef482f7e3092e9cd13d9a426ce93e0d2c194de235.jpg)  
Figure 3: Resource profile per engine (columns) as the number of series increases, for subsets drawn proportionally in product category and store. The first four rows show the runtime of each computational phase separately (log–log; the statistical and neural engines have no separate preparation phase), the fifth row the total runtime over all recorded phases including data loading, and the bottom row peak memory as aggregate proportional set size (PSS) during preparation, fitting, and prediction. Medians over five cold-start repetitions; axes are shared across engines within each row.

Table 3: Rolling-origin results on the complete M5 bottom-level panel (30,490 series, three origins, re-estimation at every origin), for the fixed configurations of Appendix A.2. Metrics use raw, unclipped engine outputs. MASE and RMSE are unweighted means over series and origins; the bracketed 95 percent bootstrap intervals resample whole series, and the corresponding RMSE intervals, which are of comparable width, are in the artifact. WRMSSE is the competition’s weighted metric over all twelve hierarchy levels at the final origin. The Tota column reports the aggregate forecast of the final origin relative to the observed total; cross-validation times and the resampling details are in Appendix A.3. The external Croston engine runs in an isolated environment with no Nixtla package installed (Appendix A.6).
<table><tr><td rowspan="2">Model Engine</td><td rowspan="2"></td><td colspan="2">MASE</td><td rowspan="2">WRMSSE</td><td rowspan="2">Total</td></tr><tr><td>mean 95% CI</td><td>RMSE</td></tr><tr><td>SeasonalNaive</td><td>StatsForecast</td><td>1.196 [1.175, 1.229]</td><td>1.781</td><td>0.847</td><td>0.99</td></tr><tr><td>AutoETS</td><td>StatsForecast</td><td>1.050 [1.028, 1.084]</td><td>1.382</td><td>0.673</td><td>0.99</td></tr><tr><td>LightGBM</td><td>MLForecast</td><td>1.568 [1.544, 1.598]</td><td>1.534</td><td>1.045</td><td>1.06</td></tr><tr><td>NHITS</td><td>NeuralForecast</td><td>0.874 [0.856, 0.906]</td><td>1.421</td><td>1.888</td><td>0.70</td></tr><tr><td>Croston</td><td>sktime (external)</td><td>1.073 [1.056, 1.101]</td><td>1.412</td><td>0.957</td><td>1.03</td></tr></table>

30,490 series. All 85 runs across 17 engine–panel-size configurations completed.

The dominant phase confirms the computational structure of Section 3. For the statistical engine, fitting dominates at every size and scales approximately linearly with the number of series, from 3.0 seconds at 100 series to 691 seconds at the full panel. For the machine-learning engine, fitting the shared gradient-boosting estimator dominates (29.8 seconds at full scale), with feature construction adding 7.0 seconds; recursive prediction remains below one second even at the full panel because the lag updates are vectorized across series. For the neural engine, a fixed 1,000-step budget makes training time weakly dependent on panel size by construction (8.8 to 16.2 seconds on one GPU); this is a fixed-compute profile rather than a comparison at equal data exposure or converged training.

Machine learning has the lowest total time from 100 through 5,000 series; at 10,000 series the machine learning and the neural engine are indistinguishable (15.6 against 15.4 seconds), and at the full panel the fixed-budget neural engine is fastest (28 against 47 seconds). These totals sum every recorded phase, including the data-loading and evaluation phases no engine avoids, which together account for about nine of the neural engine’s 28 seconds at full scale, so the engine-specific diference is wider than the totals suggest. The statistical engine is slowest beyond 100 series. Peak memory during preparation, fitting, and prediction is 15.2 GiB for the two-model statistical portfolio and 13.8 GiB for AutoETS alone (retained per-series fitted state), 7.0 GiB for machine learning (the materialized feature table), and 4.1 GiB for the neural engine (streamed window batches), rising to 5.6 GiB for the neural engine if evaluation is included and unchanged for the others.

Efect of parallelization. Runtime growth with panel size can be summarized by a scaling exponent, computed as the slope of fitting time against the number of series on log– log axes, with an exponent of 1 indicating that fitting time increases in direct proportion to the panel size. Table 4 reports this sweep for the statistical engine (StatsForecast fitting AutoETS alone), measured between 100 and 1,000 series as medians of three repetitions. The exponent is

Table 4: Efect of parallelization for the statistical engine: StatsForecast fitting AutoETS alone, medians of three repetitions in fresh processes. Twenty workers is every logical CPU of the machine (Appendix A.7). The exponent is the slope of fitting time against the number of series on log–log axes between the two panel sizes; speedups are relative to one worker at the same panel size.
<table><tr><td rowspan="2">Workers</td><td colspan="2">100 series</td><td colspan="2">1,000 series</td><td rowspan="2">Exponent</td></tr><tr><td>Fit (s)</td><td>Speedup</td><td>Fit (s)</td><td>Speedup</td></tr><tr><td>1</td><td>23.8</td><td></td><td>233.8</td><td></td><td>0.99</td></tr><tr><td>4</td><td>7.0</td><td>3.4×</td><td>62.3</td><td>3.75×</td><td>0.95</td></tr><tr><td>20</td><td>3.4</td><td>6.9×</td><td>23.8</td><td>9.8×</td><td>0.84</td></tr></table>

0.99 on one worker, 0.95 on four, and 0.84 on twenty; the speedups over one worker are 3.4 and 6.9 times at 100 series against 3.75 and 9.8 times at 1,000. Per-series work is thus approximately linear, as the single-worker exponent shows, and parallel execution understates it because perworker overheads (process startup, scheduling, and possible thread contention between workers and the numerical libraries they call, whose thread counts we did not restrict) amortize over more series as the panel grows. Reporting a parallel exponent as a property of the method would conflate the two.

Portfolio versus single model. Every engine accepts several models in one call; our configuration exercises this only in the statistical engine, which fits the SeasonalNaive and AutoETS models. To keep engine comparisons singlemodel, we profiled AutoETS alone at the two largest sizes. Adding SeasonalNaive costs 1.6 to 2.0 GiB of retained state at the full panel (about 0.3 GiB at 10,000 series) and no measurable time. The apparent time diference in the block-ordered pass reversed sign under interleaved, counterbalanced repetitions, identifying it as machinestate drift rather than model count (Appendix A.4 reports the diagnosis). For SeasonalNaive, the cost is not the model’s parameters, which are seven values per series, but the in-sample fitted values, residuals, and stored series that the fitted object retains per model, so it grows with observations rather than with series.

Table 5: Distributed backend against local multiprocessing for the statistical engine only: StatsForecast forecasting the SeasonalNaive–AutoETS portfolio over the full 30,490-series panel through the streaming .forecast interface, one measured run per backend. Construction is the one-time conversion of the in-memory panel into Ray’s distributed object store. Peak memory is aggregate proportional set size over parent and worker processes.
<table><tr><td rowspan="2">Backend</td><td colspan="3">Runtime (s)</td><td rowspan="2">Peak memory (GiB)</td></tr><tr><td>Forecast</td><td>Constr.</td><td>Total</td></tr><tr><td>Multiprocessing</td><td>738</td><td></td><td>738</td><td>3.1</td></tr><tr><td>Ray</td><td>865</td><td>32</td><td>897</td><td>19.2</td></tr></table>

Distributed backend. In this experiment, StatsForecast generated forecasts for the two-model portfolio across the full panel using two execution modes, local multiprocessing and the Ray backend on a local cluster, both accessed through the Fugue dispatch layer described in Section 3. The two backends gave numerically identical forecasts, which follows from the design: partitioning by series changes where each local model is fitted, not how. Table 5 summarizes the single-machine cost. Ray took 897 seconds in total, of which 32 were a one-time conversion of the in-memory panel into Ray’s distributed object store and 865 the forecasting pass itself, against 738 seconds for multiprocessing, and it used 19.2 GiB against 3.1 GiB, reflecting the object store and per-worker copies. We report the conversion separately because it is paid once per dataset, not per forecasting call. Ray’s value lies not in improving single-machine performance but in providing a scale-out path, since the same specification accepts a distributed dataframe in place of a pandas dataframe without requiring changes to the modeling code (McKinney, 2010). The streaming interface peaks at 3.1 GiB while forecasting and 3.4 GiB while loading, whereas the fit-then-predict interface used in the scaling experiment, which retains the fitted state of every series, peaks at 15.2 GiB for the same two-model portfolio. Both choices materially afect memory, through diferent mechanisms: the backend contrast (3.1 against 19.2 GiB) reflects Ray’s object store and per-worker copies, the price of the scale-out path, while the workflow contrast (3.1 against 15.2 GiB) reflects retained per-series fitted state. Because workflow and backend were not crossed in a complete 2 × 2 design, these contrasts do not rank the two efects. Each distributed figure comes from a single run, while the 15.2 GiB result is the median of five runs.

Limitations. We did not compare these family-specific strategies with a family-agnostic scaling primitive, and absolute runtimes and GPU results remain hardwareand driver-dependent. The transferable findings are the dominant phases under these configurations, the approximately linear growth of single-worker per-series fitting, and the mechanisms that would change the profiles: a cheaper regressor shifts machine learning toward feature construction, while an unfixed neural budget recouples

training to panel size.

## 6. Use case 3: coherent forecasts over the complete hierarchy

A demand planner consumes forecasts at every level of a retail hierarchy and requires them to be coherent, since a state-level total that difers from the sum of its stores creates problems for inventory and revenue planning regardless of per-series accuracy. Forecast reconciliation adjusts base forecasts to satisfy the aggregation constraints, and the third commitment says it should consume keyed forecasts rather than fitted models, so that the planner can change the producing engine, or add one from outside the ecosystem, without changing the reconciliation path. The benefit of reconciling keyed forecasts is that practitioners can swap or add base engines without modifying the reconciliation code.

The published twelve-level M5 aggregation structure contains 42,840 series and 67.6M observations. We create base forecasts with AutoETS and LightGBM, store both in the same keyed dataframe, and reconcile with fixed BottomUp and MinT configurations where feasible. The MinT configuration is MinTraceSparse with method="wls\_var", a diagonal approximation of the forecast-error variance estimated from the in-sample residuals of each base engine (Appendix A.5 gives the full configuration). This use case runs a single 28-day holdout at the final origin rather than the three rolling origins of Section 4, so its MASE values are not directly comparable with those of Table 3. The aim is to demonstrate that reconciliation can be implemented independently of the base forecasting method, not that one base model performs best for every hierarchy.

Both base-forecast dataframes entered the same reconciliation implementation directly from the shared keyed representation, so output-level interoperability decoupled reconciliation from model fitting. However, the contract guarantees neither coherent base forecasts, nor improved accuracy, nor computational feasibility. Table 6 reports these boundaries on the complete 42,840-series hierarchy. Base forecasts violate the aggregation constraints by up to 1,051 units for AutoETS, 18,071 for LightGBM and 204 for the external engine over the 28-day horizon. For these reconcilers coherence holds by construction, since upper levels are produced by summing the reconciled bottom level; HierarchicalForecast provides coherence utilities to verify that the emitted frames realize the constraint exactly.

The MASE columns in Table 6 should not be interpreted as a ranking, but as evidence that reconciliation redistributes error rather than consistently reducing it, with the resulting distribution determined by the base forecasts as well as the reconciliation method. For AutoETS, whose forecasts aggregate almost without bias (Section 4.2), every reconciler changes accuracy marginally. For Light-GBM, whose total is heavily biased, bottom-up aggregation repairs the total but carries bottom-level errors upward, degrading the intermediate levels most sharply (departmental MASE moves from 1.149 to 5.694), while variance-weighted MinT redistributes the adjustment and improves every reported level. Coherence is guaranteed by construction while improvement is guaranteed only under the mathematical properties of the reconciliation method.

Table 6: Base and reconciled forecasts on the complete M5 hierarchy (42,840 series, 28-day holdout at the final origin), using sparse reconcili ation implementations. MASE is the mean over series; the total level is the single top series. Runtime covers the reconciliation step only, per base engine, from one measured run each; it excludes the time of hierarchy construction. A second complete pass of the same configurations is retained in the artifact as superseded; Section 6 reports how far it difers. The dense configurations attempted, BottomUp and MinT-shrink, did not complete; dense MinT with the wls\_var estimator was not run (Section 6). Variance-weighted reconciliation is not run for the external engine, which does not supply in-sample fitted values (Appendix A.6).
<table><tr><td></td><td>Runtime (s)</td><td colspan="3">MASE, all levels</td><td colspan="3">MASE, total level</td></tr><tr><td>Forecasts</td><td></td><td>AutoETS</td><td>LightGBM</td><td>Croston</td><td>AutoETS</td><td>LightGBM</td><td>Croston</td></tr><tr><td>Base</td><td></td><td>1.031</td><td>1.476</td><td>1.047</td><td>0.653</td><td>2.163</td><td>1.457</td></tr><tr><td>BottomUp</td><td>60 / 57 / </td><td>1.029</td><td>1.653</td><td>1.047</td><td>0.677</td><td>1.723</td><td>1.439</td></tr><tr><td>MinT (wls_var)</td><td>179 / 197 /</td><td>1.033</td><td>1.321</td><td></td><td>0.672</td><td>0.605</td><td></td></tr></table>

The reconciliation path also passes the external-engine test described in Appendix A.6. The foreign engine’s base forecasts for all 42,840 series entered the same sparse bottom-up reconciler directly from the keyed frame and reproduced the pattern observed for the resident engines, with aggregation violations before reconciliation, an aggregation residual of exactly zero afterward, and only marginal changes in accuracy consistent with its nearly unbiased totals.

At the scale of M5, memory rather than runtime becomes the limiting resource. Constructing the twelve-level hierarchy took 258 seconds and peaked at 28.5 GiB before reconciliation began, while a dense summing matrix would itself require 9.7 GiB in double precision (42,840×30,490× 8 bytes). Both the dense BottomUp and MinT-shrink runs were terminated by the operating system.

Sparse linear algebra removes the expensive matrix but not the data. Although the matrices used by the sparse solver are small, the aggregated panel and the 66-millionrow in-sample fitted frame remain resident because the output contract passes the complete frame to the variance estimator, even though a diagonal estimator mathematically requires only one variance per series. The sparse MinT step therefore still peaked at 43.6 GiB, and was itself terminated by the operating system when both engines’ intermediates were resident in one process; restructured to hold one engine at a time, it completed in 179 and 197 seconds (Table 6).

Limitations. Sparse MinT is not guaranteed to match the dense implementation of the same estimator; at this scale the dense configurations exhausted memory, so we could not bound the diference. Variance-weighted reconciliation remains undemonstrated for external producers because it additionally requires in-sample fitted values through the same keyed contract. The observed feasibility boundary is a single threshold crossing, not a general comparison of workflow or algorithm choices.

## 7. Discussion

The three use cases characterize the commitments within the limits of one ecosystem, one dataset, and the versions in Table 1. Their value lies in separating guarantees checkable at package boundaries from behaviors that cannot be standardized away. This section synthesizes their implications; experiment-specific limitations accompany each use case, and study-wide limitations are collected below.

C1: Share the data contract, not the estimator. The boundary used here is explicit series and time keys on the input, an explicit origin schedule, and series, time, and origin keys on the output. These suficed for common evaluation across families whose fitted states have nothing in common, but not to standardize feature availability, loss functions, refitting defaults, horizon placement, or probabilistic semantics, which remain visible engine responsibilities. The evidence therefore supports a narrow interoperability claim: shared keys standardize exchange, not behavioral semantics.

C2: Implement scalability inside the model family. The phase profiles of Section 5 are consistent with this commitment, and show that partitioning by series is not a general scaling primitive. That these methods have diferent bottlenecks is a property of the methods, established independently (Januschowski et al., 2020); the use case measures where the diferences surface. Memory, however, depends on the complete execution path rather than the model family alone: retaining per-series fitted state (“fitthen-predict”) raised local memory from 3.1 to 15.2 GiB relative to the streaming .forecast interface, while Ray’s object store and per-worker copies raised streaming memory from 3.1 to 19.2 GiB. Both choices matter, and neither contrast was measured on a common design, so the commitment concerns where scaling behavior is implemented, not comparative eficiency; capacity planning must account for both model-state residency and backend overhead.

C3: Let downstream components consume forecasts, not models. A single joined output table allowed both base engines to use the same reconcilers and made their evaluation metrics directly comparable, while the external test demonstrated extensibility by routing an engine from another ecosystem through the same evaluation and reconciliation workflows using a 121-line adapter (Sections 4 and 6). The full-hierarchy use case also establishes that schema-compatible base forecasts need not be coherent, reconciliation need not improve accuracy, and dense methods need not remain computationally tractable. Section 4.2 reveals a related boundary because the contract standardizes the data exchanged between components but not the evaluation objective, allowing the most convenient metric to produce a ranking contradicted by the hierarchy-weighted metric. These results support C3 for the point-forecast evaluation and reconciliation operations tested here.

Scope conditions. These commitments are not universally advantageous. When covariate handling is a primary source of workflow errors, a dedicated time-series object that validates covariates at construction may be preferable to a generic dataframe, weakening the case for C1. When a single small team maintains the entire stack, a multirepository architecture adds coordination costs without providing the benefit of independent release cycles. A unified estimator interface may similarly be preferable when the primary objective is automated search across model families, weakening the case for C2. Finally, when production relies on only one model family, a shared contract has few components to connect, limiting the value of C1 and C3. The commitments are most useful when several engines evolve independently, multiple model families operate in production, and downstream workflows must support all of them.

## 7.1. Limitations

Our evidence comes from a small set of untuned methods, one dataset, one ecosystem, and fixed software versions; it neither ranks model families nor establishes general eficiency. Results depend on dataset size, seasonal patterns, covariates, hardware, and model settings. M5 in particular is daily, highly intermittent, strongly dayof-week seasonal, and hierarchical; the negative-forecast rates of Section 4.2 and the necessity of sparse reconciliation should not be expected to transfer unchanged to smoother or shallower panels.

Every engine we measure was built around the commitments we examine, so we cannot weigh them against the unified-estimator alternative of Section 2, although we expect such an interface to absorb some of the coordination costs of Section 4.1. Whether the keyed contract is minimal remains untested, as does whether every downstream operation can be expressed over keyed outputs; some operations may require fitted models rather than their outputs.

We evaluate point forecasts only: probabilistic forecasting, probabilistic reconciliation in the sense of Panagiotelis et al. (2023), and model-dependent forecast combination remain outside scope. The long-data representation is flexible but less strict than a dedicated time-series object, so errors such as incorrectly labeled future covariates cannot be detected from the dataframe alone, and the opensource inventory records visible project infrastructure at one date; it does not measure governance quality, contributor diversity, maintenance responsiveness, or long-term sustainability.

Finally, this is not an independent evaluation: the authors are not at arm’s length from the software under study, and the declaration of competing interest at the end of this article states the relationship; the published protocol, raw results, and unfavorable outcomes exist so that this can be audited. Descriptions of other frameworks rest on their published documentation, and we invite their maintainers to correct any factual description.

## 8. Conclusion and Future Work

We presented the Nixtlaverse as a case study of three design commitments for open-source forecasting software: share panel and output contracts rather than fitted estimators, place scalability within model-family implementations, and define downstream operations over keyed forecasts. Its multi-repository implementation preserves specialized dependencies and interfaces, but makes compatibility testing and versioned research artifacts part of the software’s coordination burden. The Nixtlaverse has achieved substantial public distribution, documented scholarly reuse, and adoption through other forecasting frameworks.

We’ve demonstrated that explicit input, origin, and output keys suficed to evaluate all three families (and one engine external to the ecosystem) on the complete M5 panel without per-engine reshaping, while leaving their constructors and behavioral semantics distinct. We ofer these boundaries, rather than surface-level interface uniformity, as the transferable lesson. Whether they generalize would require replication on a second ecosystem or a domain with diferent intermittency and hierarchy depth, and both remain open.

Three findings do not depend on the specific libraries a reader adopts. First, capacity planning must account for both workflow and backend. For the per-series engine, retaining fitted state raised peak memory from 3.1 GiB under local streaming to 15.2 GiB under local fit-thenpredict, while Ray’s object store and per-worker copies raised streaming memory to 19.2 GiB. These measurements identify two material mechanisms rather than ranking their efects: deployments must budget separately for fitted-state residency and distributed-execution overhead. Second, sparse linear algebra is necessary at scale: a dense double-precision summing matrix for the M5 hierarchy requires 9.7 GiB before a single forecast is reconciled. Finally, the measurements quantify, for this workload, the property that per-series parallelization does not transfer to global models: one parallelization control across model families describes an interface rather than a scaling behavior.

The next step for the ecosystem is to reduce the coordination burden described in Section 3.7 by consolidating the packages into a single workspace with a shared dependency lockfile, for example through a uv workspace. Cross-package changes could then be developed and tested together while each package remains separately installable, preserving the ecosystem’s modularity while removing much of its manual coordination overhead.

## CRediT author statement

Olivier Sprangers: Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Project administration, Software, Validation, Visualization, Writing–original draft, Writing–review and editing. Max Mergenthaler Canseco, Marco Peixeiro, Saul Caballero Ramirez, Mariana Menchero García, Jing-Qiang (JQ) Goh, Han Wang, Nikhil Gupta, Rogelio Melo, Senbong Gee, and Cristian Challu: Software, Writing–review and editing.

## Acknowledgements

The authors thank José Morales, Kin Gutierrez, and Deven Mistry.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-forprofit sectors.

## Declaration of competing interest

The authors work at Nixtla, the company that develops the software examined as the principal subject of this article. This relationship may be perceived as a competing interest.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the authors used OpenAI Codex and Anthropic Claude Code to assist with literature discovery and drafting. The authors reviewed and edited the content and take full responsibility for the content of the article.

## Data and code availability

The accompanying artifact contains the source code, dependency specifications, integrity manifests, raw outcomes, and commands for every result, including the external-engine adapter and its isolated-environment specification (Appendix A.6). The M5 source files are not redistributed. The artifact is publicly available at https://doi.org/10.6084/m9.figshare.33399445.

## References

Alexandrov, A., Benidis, K., Bohlke-Schneider, M., Flunkert, V., Gasthaus, J., Januschowski, T., Maddix, D.C., Rangapuram, S., Salinas, D., Schulz, J., Stella, L., Türkmen, A.C., Wang, Y., 2020. GluonTS: Probabilistic and neural time series modeling in python. Journal of Machine Learning Research 21, 1–6. URL: https://www.jmlr.org/papers/v21/19-820.html.

Ali, M., 2020. PyCaret: An open source, low-code machine learning library in Python. Software. URL: https:// pycaret.org.

Ansari, A.F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S.S., Pineda Arango, S., Kapoor, S., Zschiegner, J., Maddix, D.C., Wang, H., Mahoney, M.W., Torkkola, K., Wilson, A.G., Bohlke-Schneider, M., Wang, Y., 2024. Chronos: Learning the language of time series. Transactions on Machine Learning Research URL: https: //openreview.net/forum?id=gerNCVqqtR.

Athanasopoulos, G., Hyndman, R.J., Kourentzes, N., Panagiotelis, A., 2024. Forecast reconciliation: A review. International Journal of Forecasting 40, 430–456. doi:10.1016/j.ijforecast.2023.10.010.

Bergmeir, C., Hyndman, R.J., Koo, B., 2018. A note on the validity of cross-validation for evaluating autoregressive time series prediction. Computational Statistics & Data Analysis 120, 70–83. doi:10.1016/j.csda.2017. 11.003.

Challu, C., Olivares, K.G., Oreshkin, B.N., Garza Ramirez, A., Mergenthaler Canseco, M., Dubrawski, A., 2023. NHITS: Neural hierarchical interpolation for time series forecasting, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 6989–6997. doi:10.1609/aaai.v37i6.25854.

Croston, J.D., 1972. Forecasting and stock control for intermittent demands. Operational Research Quarterly 23, 289–303. doi:10.1057/jors.1972.50.

Das, A., Kong, W., Sen, R., Zhou, Y., 2024. A decoderonly foundation model for time-series forecasting, in: Proceedings of the 41st International Conference on Machine Learning, PMLR. pp. 10148–10167. URL: https: //proceedings.mlr.press/v235/das24c.html.

Databricks Industry Solutions, 2026. Many model forecasting: Bootstrap large-scale forecasting solutions on Databricks. Software, accessed 28 August 2026. URL: https: //github.com/databricks-industry-solutions/ many-model-forecasting.

Delgado Fernandez, J., Potenciano Menci, S., Magitteri, A., 2025. Forecasting anonymized electricity load profiles, in: 2025 IEEE PowerTech, pp. 1–6. doi:10.1109/ PowerTech59965.2025.11180602.

Garza Ramirez, A., Mergenthaler Canseco, M., Challu, C., Olivares, K.G., 2022. StatsForecast: Lightning fast forecasting with statistical and econometric models. PyCon US, Salt Lake City, Utah. URL: https: //github.com/Nixtla/statsforecast.

Girolimetto, D., Di Fonzo, T., 2026. FoReco: Forecast reconciliation. R package version 1.2.1. URL: https: //CRAN.R-project.org/package=FoReco.

Godahewa, R., Bergmeir, C., Webb, G.I., Hyndman, R.J., Montero-Manso, P., 2021. Monash time series forecasting archive, in: Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks. URL: https://datasets-benchmarks-proceedings. neurips.cc/paper\_files/paper/2021/hash/ eddea82ad2755b24c4e168c5fc2ebd40-Abstract-round html.

Herzen, J., Lässig, F., Piazzetta, S.G., Neuer, T., Tafti, L., Raille, G., Van Pottelbergh, T., Pasieka, M., Skrodzki, A., Huguenin, N., Dumonal, M., Kościsz, J., Bader, D., Gusset, F., Benheddi, M., Williamson, C., Kosinski, M., Petrik, M., Grosch, G., 2022. Darts: Userfriendly modern machine learning for time series. Journal of Machine Learning Research 23, 1–6. URL: https: //jmlr.org/papers/v23/21-1177.html.

Hewamalage, H., Bergmeir, C., Bandara, K., 2021. Recurrent neural networks for time series forecasting: Current status and future directions. International Journal of Forecasting 37, 388–427. doi:10.1016/j.ijforecast. 2020.06.008.

Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G., Shang, H.L., 2011. Optimal combination forecasts for hierarchical time series. Computational Statistics & Data Analysis 55, 2579–2589. doi:10.1016/j.csda.2011.03. 006.

Hyndman, R.J., Khandakar, Y., 2008. Automatic time series forecasting: The forecast package for R. Journal of Statistical Software 27, 1–22. doi:10.18637/jss.v027. i03.

Hyndman, R.J., Koehler, A.B., 2006. Another look at measures of forecast accuracy. International Journal of Forecasting 22, 679–688. doi:10.1016/j.ijforecast. 2006.03.001.

Hyndman, R.J., Koehler, A.B., Snyder, R.D., Grose, S., 2002. A state space framework for automatic forecasting using exponential smoothing methods. International Journal of Forecasting 18, 439–454. doi:10.1016/ S0169-2070(01)00110-8.

Januschowski, T., Gasthaus, J., Wang, Y., Salinas, D., Flunkert, V., Bohlke-Schneider, M., Callot, L., 2020. Criteria for classifying forecasting methods. International Journal of Forecasting 36, 167–177. doi:10.1016/ j.ijforecast.2019.05.008.

Januschowski, T., Wang, Y., Torkkola, K., Erkkilä, T., Hasson, H., Gasthaus, J., 2022. Forecasting with trees. International Journal of Forecasting 38, 1473– 1481. doi:10.1016/j.ijforecast.2021.10.004.

Joseph, M., Tackes, J., 2024. Modern Time Series Forecasting with Python. 2 ed., Packt Publishing, Birmingham, UK.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.Y., 2017. LightGBM: A highly efficient gradient boosting decision tree, in: Advances in Neural Information Processing Systems, pp. 3146–3154.

Kolassa, S., 2016. Evaluating predictive count data distributions in retail sales forecasting. International Journal of Forecasting 32, 788–803. doi:10.1016/j. ijforecast.2015.12.004.

Löning, M., Bagnall, A., Ganesh, S., Kazakov, V., Lines, J., Király, F.J., 2019. sktime: A unified interface for machine learning with time series. doi:10.48550/arXiv. 1909.07872, arXiv:1909.07872.

Makridakis, S., Spiliotis, E., Assimakopoulos, V., 2022. M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting 38, 1346– 1364. doi:10.1016/j.ijforecast.2021.11.013.

McKinney, W., 2010. Data structures for statistical computing in Python, in: Proceedings of the 9th Python in Science Conference (SciPy), pp. 56–61. doi:10.25080/ Majora-92bf1922-00a.

MindsDB, 2026. Lightwood: An AutoML framework underpinning MindsDB. Software, accessed 28 August 2026. URL: https://github.com/mindsdb/ lightwood.

Montero-Manso, P., Hyndman, R.J., 2021. Principles and algorithms for forecasting groups of time series: Locality and globality. International Journal of Forecasting 37, 1632–1653. doi:10.1016/j.ijforecast.2021.03.004.

Moritz, P., Nishihara, R., Wang, S., Tumanov, A., Liaw, R., Liang, E., Elibol, M., Yang, Z., Paul, W., Jordan, M.I., Stoica, I., 2018. Ray: A distributed framework for emerging AI applications, in: Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI), pp. 561–577.

Muniz, R.N., Buratto, W.G., Gonzalez, G.V., Seman, L.O., Costa, V.J., Nied, A., 2026. Optimized hybrid neural hierarchical interpolation time series with STL for flow forecasting in hydroelectric power plants. Scientific Reports 16, 5002. doi:10.1038/ s41598-025-34847-x.

O’Hara-Wild, M., Hyndman, R.J., Wang, E., 2026a. fable: Forecasting models for tidy time series. R package version 0.5.0. URL: https://CRAN.R-project.org/ package=fable. accessed 21 July 2026.

O’Hara-Wild, M., Hyndman, R.J., Wang, E., 2026b. fabletools: Core tools for packages in the ‘fable’ framework. R package version 0.8.0. URL: https:// CRAN.R-project.org/package=fabletools. accessed 21 July 2026.

Olivares, K.G., Challu, C., Garza Ramirez, A., Mergenthaler Canseco, M., Dubrawski, A., 2022. Neural-Forecast: User friendly state-of-the-art neural forecasting models. PyCon US, Salt Lake City, Utah. URL: https://github.com/Nixtla/neuralforecast.

Olivares, K.G., Garza Ramirez, A., Luo, D., Challu, C., Mergenthaler Canseco, M., Ben Taieb, S., Wickramasuriya, S.L., Dubrawski, A., 2024. Hierarchical-Forecast: A reference framework for hierarchical forecasting in python. doi:10.48550/arXiv.2207.03517, arXiv:2207.03517.

Oliveira, J.M., Ramos, P., 2024. Evaluating the efectiveness of time series transformers for demand forecasting in retail. Mathematics 12, 2728. doi:10.3390/ math12172728.

Panagiotelis, A., Gamakumara, P., Athanasopoulos, G., Hyndman, R.J., 2023. Probabilistic forecast reconciliation: Properties, evaluation and score optimisation. European Journal of Operational Research 306, 693–706. doi:10.1016/j.ejor.2022.07.040.

Petropoulos, F., Apiletti, D., Assimakopoulos, V., Babai, M.Z., Barrow, D.K., Ben Taieb, S., Bergmeir, C., Bessa, R.J., Bijak, J., Boylan, J.E., et al., 2022. Forecasting: Theory and practice. International Journal of Forecasting 38, 705–871. doi:10.1016/j.ijforecast.2021.11. 001.

Rocklin, M., 2015. Dask: Parallel computation with blocked algorithms and task scheduling, in: Proceedings of the 14th Python in Science Conference (SciPy), pp. 126–132. doi:10.25080/Majora-7b98e3ed-013.

Semenoglou, A.A., Spiliotis, E., Makridakis, S., Assimakopoulos, V., 2021. Investigating the accuracy of cross-learning time series forecasting methods. International Journal of Forecasting 37, 1072–1084. doi:10. 1016/j.ijforecast.2020.11.009.

Shchur, O., Ansari, A.F., Turkmen, C., Stella, L., Erickson, N., Guerron, P., Bohlke-Schneider, M., Wang, Y., 2025. fev-bench: A realistic benchmark for time series forecasting arXiv:2509.26468.

Shchur, O., Turkmen, A.C., Erickson, N., Shen, H., Shirkov, A., Hu, T., Wang, B., 2023. AutoGluon– TimeSeries: AutoML for probabilistic time series forecasting, in: Proceedings of the Second International Conference on Automated Machine Learning, PMLR.

Souto, H.G., 2026. Evaluating the eficacy of NHITS for forecasting stock realized volatility: A comparative analysis with established models. Computational Economics 67, 1291–1348. doi:10.1007/ s10614-025-10917-0.

TAOS Data, 2026. TDengine: High-performance timeseries database, TDgpt analytics component. Software, accessed 28 August 2026. URL: https://github.com/ taosdata/TDengine.

Tashman, L.J., 2000. Out-of-sample tests of forecasting accuracy: An analysis and review. International Journal of Forecasting 16, 437–450. doi:10.1016/S0169-2070(00) 00065-0.

Taylor, S.J., Letham, B., 2018. Forecasting at scale. The American Statistician 72, 37–45. doi:10.1080/ 00031305.2017.1380080.

The KServe Authors, 2026. KServe: Standardized model inference platform on Kubernetes. Software, accessed 28 August 2026. URL: https://github.com/kserve/ kserve.

Thoughtworks, 2025. Technology Radar, Volume 33: An Opinionated Guide to Today’s Technology Landscape. Technical Report. Thoughtworks. URL: https://www.thoughtworks.com/content/ dam/thoughtworks/documents/radar/2025/11/tr\_ technology\_radar\_vol\_33\_en.pdf.

Wang, E., Cook, D., Hyndman, R.J., 2020. A new tidy data structure to support exploration and modeling of temporal data. Journal of Computational and Graphical Statistics 29, 466–478. doi:10.1080/10618600.2019. 1695624.

Wang, H., Kho, K., The Fugue Development Team, 2022. Fugue: A unified interface for distributed computing. Software. URL: https://github.com/ fugue-project/fugue.

Wickramasuriya, S.L., Athanasopoulos, G., Hyndman, R.J., 2019. Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. Journal of the American Statistical Association 114, 804–819. doi:10.1080/01621459.2018.1448825.

Wickramasuriya, S.L., Turlach, B.A., Athanasopoulos, G., 2020. Optimal non-negative forecast reconciliation. Statistics and Computing 30, 1167–1182. doi:10.1007/ s11222-020-09930-0.

Windler, T., 2024. Leveraging advanced forecasting techniques with StatsForecast: A case study. m2hycon. URL: https://www.m2hycon.de/en/news/ leveraging-advanced-forecasting-techniques-with

Zaharia, M., Xin, R.S., Wendell, P., Das, T., Armbrust, M., Dave, A., Meng, X., Rosen, J., Venkataraman, S., Franklin, M.J., Ghodsi, A., Gonzalez, J., Shenker, S., Stoica, I., 2016. Apache Spark: A unified engine for big data processing. Communications of the ACM 59, 56–65. doi:10.1145/2934664.

## Appendix A. Experimental protocol

This appendix specifies the datasets, method configurations, and measurement methodology behind the use cases of Sections 4–6. The artifact at https://doi.org/ 10.6084/m9.figshare.33399445 contains the code to reproduce.

## Appendix A.1. Datasets

We use the M5 competition data: daily unit sales from ten Walmart stores in three US states (Makridakis et al., 2022). After leading zero-sales periods are removed per series, the bottom-level panel contains 30,490 series and 47,649,940 observations. It supports the evaluation and scaling use cases. The published twelve-level aggregation structure contains 42,840 series and 67,687,697 observations and supports the hierarchical use case. We use the competition horizon of 28 days throughout; the frozen dataset manifest records the source files and hashes.

## Appendix A.2. Method configurations

We do not perform method-specific hyperparameter optimization, using documented defaults or one pre-specified configuration. The parameters we set are: season length 7 for SeasonalNaive and AutoETS, the latter selecting its error–trend–seasonal form automatically; target lags {7, 14, 28} and dayofweek and month date features for LightGBM, with recursive multi-step prediction; and context length 56 (2h), a 1,000-step training budget, start padding (enabled because some M5 series are shorter than the context length at the earliest forecast origins), and the default mean-absolute-error loss for NHITS. Every engine uses horizon h = 28, daily frequency, and a seed fixed per repetition; all remaining parameters take their library defaults at the versions in Table 1.

Fixing the configurations makes tuning budget irrelevant to the comparison; the absence of tuning penalizes engines whose defaults are weak on intermittent retail data, and the fixed neural step budget insulates the neural engine from the panel-size axis that Section 5 measures. The configurations are therefore comparable in efort, not in fitted quality, and Section 4.2 interprets the accuracy results accordingly.

## Appendix A.3. Evaluation protocol

All methods use the same three non-overlapping forecast origins, 28-day horizon, and observed target values, with a model re-estimated at every origin. We compute <sup>statsforecast-a-case-study/.</sup>MASE (Hyndman and Koehler, 2006) and RMSE at every origin with UtilsForecast from the combined forecast dataframe; the MASE denominator is the in-sample seasonal-naive mean absolute error recomputed from the data preceding each origin, with no floor imposed on it. Both metrics are unweighted means over series and origins. For the final origin, we additionally report the competition’s weighted root mean squared scaled error (WRMSSE), which aggregates accuracy over all 42,840 series at the twelve M5 hierarchy levels (Makridakis et al., 2022). WRMSSE is reported at one origin because the competition weights are defined for a specific evaluation window, so the weighted comparison in Section 4.2 is established at that origin and not across the three; the aggregate forecast ratios reported alongside it do not depend on the competition weights and are given at all three origins.

The intervals in Table 3 are percentile bootstrap intervals over 2,000 resamples of whole series. The intervals condition on one training realization of the neural model; the seed sensitivity reported in Section 4.2 comes from retraining NHITS under three further seeds at full scale, recorded in the artifact.

## Appendix A.4. Resource-measurement protocol

For the scaling use case, we create nested subsets of the main panel with an increasing number of series and reserve the final 28 days as a common holdout. Subsets are drawn by seeded sampling proportional in product category and store, so each approximates the composition of the complete panel while remaining nested across sizes. The sampling seed is fixed across engines and repetitions. The subsets profile resources only; comparative accuracy is reported on the complete panel. We measure data preparation, fitting, prediction, and evaluation separately, recording wall-clock time and peak memory as aggregate proportional set size (PSS) over parent and worker processes, so pages shared between forked processes are counted once. Because the statistical engine can fit several models in one call, we profile both the SeasonalNaive–AutoETS portfolio and AutoETS alone, using the single-model configuration wherever engines are compared with each other, and vary the worker count at two panel sizes so that the parallelization efect can be measured. The evaluation phase times a code path that is identical across engines (the merge of the engine’s forecasts with the common holdout, followed by one evaluation call); it is measured with the engine’s fitted state still resident, so small per-engine diferences in this phase reflect the measurement context rather than engine-specific evaluation code. Thread-count environment variables for the numerical libraries are left unset and recorded per run; parallel configurations may therefore include thread oversubscription between worker processes and the threaded libraries they call. The artifact records the remaining per-run details like software versions, thread settings, device, seed, and CUDA memory for the training process.

The panel-size sweep uses five measured repetitions per configuration and the worker-count sweep three, each in a fresh process so cold-start costs are included and memory is not contaminated by earlier models; we report medians and dispersion. Where two engine configurations are compared with each other rather than across panel sizes, their repetitions are interleaved and their order counterbalanced, so drift in machine state over a long run sequence cannot be mistaken for a diference between configurations. The reconciliation and distributed-backend experiments report one measured run per configuration, because a single configuration takes tens of minutes and, for the dense reconcilers, exhausts machine memory; Sections 5 and 6 state the repetitions behind every figure and, where a second pass exists, how far it difers.

## Appendix A.5. Hierarchical workflow

For the hierarchical dataset, we create base forecasts with AutoETS and LightGBM, store both sets of forecasts in the same keyed dataframe, and reconcile them using fixed BottomUp and MinT configurations where feasible. The MinT configuration is MinTraceSparse with method="wls\_var": a diagonal approximation of the forecast-error variance, estimated from the in-sample residuals of each base engine, with no non-negativity constraint imposed (for which see Wickramasuriya et al., 2020). The experiment uses a single 28-day holdout at the final origin rather than the three rolling origins of the evaluation use case. We report MASE by hierarchy level for the base and reconciled forecasts, the reconciliation runtime, and the maximum aggregation residual; the level-wise results diagnose the efect of reconciliation without weighting large revenue series more heavily, whereas WRMSSE in Section 4.2 uses the complete oficial hierarchy and competition weights.

## Appendix A.6. External-engine setup

The extensibility claim of Section 3, that downstream components accept any engine returning the required keys, cannot be tested with engines built alongside those components. We therefore repeat both downstream tasks with a foreign producer: Croston’s method (Croston, 1972) as implemented in sktime 1.1.0 (Löning et al., 2019), an intermittent-demand method appropriate for M5 from an external codebase. The external engine runs in a separate virtual environment with no Nixtla package installed, so the only objects that cross the boundary are the long panel going out and a keyed forecast dataframe coming back. Its forecasts at the same three origins enter the same single evaluation call as the resident engines (Section 4), and its base forecasts over the complete 42,840-series hierarchy enter the same sparse bottom-up reconciler (Section 6). Variance-weighted reconciliation is not run for the external engine, because MinT additionally requires in-sample fitted values from the producer.

## Appendix A.7. Reproducibility

The accompanying artifact contains the executable benchmark scripts, direct dependency specification, complete version-pinned dependency closure, M5 file manifest with SHA-256 hashes, model configurations, final raw outcomes, and an evidence map from each manuscript result to its producing command. A second SHA-256 manifest covers the publishable artifact itself.

Each JSONL outcome records the observed Python and package versions, platform, CPU and memory capacity, thread settings, accelerator, and seed. The reported runs used Python 3.10.12 on Linux/WSL2 with 20 logical CPUs, 48.0 GiB RAM, and an NVIDIA GeForce RTX 5090. Exact runtimes and GPU outcomes remain hardware- and driver-dependent.