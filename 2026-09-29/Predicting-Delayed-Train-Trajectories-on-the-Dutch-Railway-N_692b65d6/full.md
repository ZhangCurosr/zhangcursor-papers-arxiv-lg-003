# Predicting Delayed Train Trajectories on the Dutch Railway Network: Explainable AI Evaluation of Topological, Operational and Weather Features with Tree Based Ensemble Methods

Jia Long Bao, Ali Mohammed Mansoor Alsahag, Seyed Sahand Mohammadi Ziabari

## Abstract

The reliable prediction of passenger train delays is a critical component of railway management. While contemporary research frequently attempts to maximize absolute accuracy by deploying opaque deep learning architectures, the underlying data mechanics driving longitudinal predictive decay remain underexplored. Consequently, this study provides an explainable temporal robustness analysis of network-wide railway delay prediction. Focusing on the Dutch railway network, this research utilizes interpretable treebased ensembles to integrate granular topological, environmental, and operational features. The overarching finding establishes that while feature-rich tree-based models improve simultaneous (withinmonth) prediction, predictive performance systematically degrades when evaluated across non-simultaneous (future) months. Furthermore, multi-horizon SHAP and dispersion analyses explicitly link this degradation to environmental feature volatility and instability within the statistical target definition. Ultimately, this thesis demonstrates that richer feature sets alone are insuficient to resolve long-term forecasting constraints, underscoring the necessity to transition toward dynamic, season-aware architectures anchored by absolute operational boundaries.

## Keywords

Network-Wide Railway Delay Prediction, Spatiotemporal Data, Explainable Artificial Intelligence, Tree-Based Ensemble Models, Supervised Classification

## Github Repository

https://github.com/jbaonl/MSc-Thesis/tree/publication-revisions

## 1 Introduction

This section examines the economic consequences ofrailway delays, identifies research gaps in network-wide delay prediction, and defines the guiding research questions.

## 1.1 Economic and Operational Context

1.1.1 Financial Costs of Delays. In 2025, the Dutch national railway operator, Nederlandse Spoorwegen (NS), distributed €9.7 million in annual compensation and refunds [33]. Beyond direct fiscal burdens, schedule deviations result in cumulative time losses costing the Dutch economy over €400 million annually [36].

1.1.2 Systemic Vulnerabilities. The Netherlands maintains a high density rail network where high-frequency scheduling amplifies sensitivities to operational failures and environmental factors, such as winter weather or seasonal track debris [2, 43]. Managing this complexity requires a continuous trade-of between mitigating cascading delays and maintaining individual service punctuality [25].

1.1.3 Regulatory AI Transparency in Public Infrastructure. The European Union Artificial Intelligence Act (EU AI Act) mandates transparency and human oversight for automated decision-making systems deployed within high-risk infrastructure domains, such as the Dutch national railway [16]. With transparency rules scheduled for enforcement by August 2026, developing interpretable machine learning pipelines is a practical regulatory necessity [13]. This research utilizes SHapley Additive exPlanations (SHAP) to align network-wide delay modeling with these emerging regulatory requirements [30, 31].

## 1.2 Research Gaps in Network-Wide Train Delay Prediction

1.2.1 The Interpretability and Temporal Robustness Gap. Recent literature indicates a paradigm shift toward short-term, real-time prediction methodologies utilizing complex hybrid architectures to maximize immediate operational accuracy [11, 28, 39, 41, 44, 47]. However, this trajectory frequently neglects spatiotemporal data representation within aggregated, long-term network-wide predictive frameworks [39]. While topological features extracted from spatial graphs have facilitated efective link prediction in alternative domains [26], adapting these techniques to railway networks reveals significant empirical generalization challenges across unseen temporal horizons [22].

1.2.2 Positioning Within Contemporary Literature. This research diverges from recent foundational frameworks across several key dimensions. Compared to the baseline established by Kämper [22], this study adopts a highly granular stop-to-stop network representation, incorporates engineered operational features, and integrates canceled services into the delay target. Unlike prior research utilizing regional weather data [10], this study employs exact station coordinates to minimize spatial noise. Finally, in contrast to black-box deep learning architectures [40], this research prioritizes diagnostic interpretability, utilizing tree-based ensembles to enable explicit SHAP value analysis.

Ultimately, the central contribution of this research is not exclusively the absolute prediction of delays, but an explainable temporal robustness analysis of network-wide railway delay prediction. By trading the raw predictive density of deep learning for the diagnostic capability of tree-based models and SHAP analysis, this approach systematically links predictive degradation to environmental feature volatility and target instability.

## 1.3 Methodological Approach and Research Objectives

Tree-based ensembles are selected for this network-wide analysis because they bypass strict statistical distribution assumptions, capture non-linear feature interactions, and generate explicit feature importance rankings [39]. A network-wide spatial scope is necessary because delays within dense infrastructures rarely remain isolated; this approach ensures cascading delays are accurately modeled across the entire topology.

1.3.1 Temporal Evaluation Framework. To evaluate long-term temporal generalization, operational data is partitioned into discrete monthly snapshots. While this isolates macroscopic network states from real-time operational noise [22, 26, 39], it introduces inherent methodological trade-ofs. Specifically, defining a binary target variable across aggregated monthly periods introduces vulnerability to temporal label shift as seasonal operational frequencies fluctuate [22]. To mitigate this structural limitation, this research prioritizes temporal robustness by incorporating a sensitivity analysis to validate the target definition across alternative operational thresholds. Furthermore, rather than solely optimizing predictive accuracy, this study prioritizes temporal robustness through a comparative evaluation of topological, operational, and environmental feature sets. Finally, SHAP values are utilized to diagnose feature volatility, isolate the primary factors underlying predictive stability, and critically assess the underlying assumptions of the monthly aggregation framework [30].

## 1.4 Research Questions

The following research questions guide this methodological investigation:

Main Research Question: To what extent does the integration of topological, environmental, and operational features within tree-based ensemble models maintain temporal generalizability for network-wide passenger train delayed trajectories within the Dutch railway network operated by NS across subsequent monthly snapshots?

Sub-Questions:

• SRQ1 (Temporal Generalization): How does predictive performance, quantified via paired Wilcoxon statistical tests of balanced accuracy, vary between simultaneous (withinmonth) and non-simultaneous (future unseen months) testing phases?

• SRQ2 (Model Behavior Insight): How does the relative predictive importance of distinct feature categories (topological, environmental, and operational) within the primary XG-Boost (trajectory-based split) fluctuate across non-simultaneous horizons, as measured by SHAP value rankings, and which specific variables demonstrate the highest stability against temporal drift?

The remainder of this paper reviews the relevant literature, details the methodology and experimental setup, and concludes with a discussion of the results.

## 1.5 Code Availability

The code is publicly available on: https://github.com/jbaonl/MSc-Thesis/tree/publication-revisions

## 1.6 Data Availability

Data utilized is publicly available, exact details can be found in the data collection notebook which is provided in the repo.

## 1.7 Ethical and Societal Considerations

The predictive models, software, and findings presented in this study are intended solely for academic and experimental purposes. We explicitly condemn the adaptation of these models in ways that justify, scale, or exacerbate regional and urban disadvantages.

A critical consideration in this thesis is the inherent risk of temporal baselines, such as the naive and seasonal naive models utilized herein. Because these models rely exclusively on historical target variables, they are highly susceptible to reiterating structural inequities already built into the system. For instance, if specific routes face chronic delays due to historical underinvestment or systemic neglect, a temporal model might encode these delays as natural seasonality. Deploying such models without critical oversight risks ‘bias laundering’ [34], providing a mathematical facade of objectivity to systemic ineficiencies. Consequently, this creates plausible deniability for organizations and difuses responsibility for improving service quality.

Furthermore, we must warn against proxy blindness in delay prediction [4]. While models may not explicitly evaluate demographic data, spatial and temporal features, such as route density or high-density urban corridors versus lower-density rural routes, frequently act as heavily correlated proxies for socioeconomic status. A comprehensive fairness audit to evaluate whether delay predictions disparately impact specific communities is a critical avenue for future work prior to any operational deployment.

Finally, while this study acknowledges the value of explainability through feature attribution, it must be noted that explainability is not a substitute for algorithmic fairness. The attribution of a target variable to specific features can be easily manipulated or cherry-picked by organizations to further difuse responsibility, a phenomenon often referred to as fairwashing [3]. True algorithmic fairness requires looking beyond the mathematical mechanisms of the model to the real-world historical context generating the data. Ultimately, it is critical to recognize that the explainability techniques utilized in this study capture correlational dependencies driving the model’s predictions, constrained entirely by the provided input features. These mathematical attributions must not be conflated with physical causality, nor should they be interpreted as definitive explanations for real-world operational failures.

## 2 Related Work

The prediction of long-term transportation dynamics increasingly relies on topological feature extraction and graph-based learning. While efective in aviation, its application to railway delay prediction reveals a fundamental gap between representational richness, temporal generalization, and model interpretability. This section evaluates prior methodologies to ground the diagnostic framework required for SRQ1 and SRQ2.

## 2.1 Topological Representation and Network Dynamics

Modeling spatial dependencies remains a central challenge in machine learning [32]. To circumvent the high-dimensional feature spaces generated by traditional encoding, recent methodologies extract topological features directly from spatial graphs for integration into tabular models [22, 26].

2.1.1 Foundational Representations and Railway Limitations. Dynamic graph topology serves as an efective proxy for spatial relationships. Lei demonstrated that localized topological features, particularly operational edge weights, facilitate efective tabular link prediction in aviation [26]. However, Lei noted severe generalization degradation under non-simultaneous testing in alternative domains such as the Brazil bus network, suggesting topological features struggle under temporal variation.

Adapting this framework to the Dutch railway network, Kämper observed similarly restricted predictive generalization across all temporal testing scenarios [22]. This failure was linked to this limited generalization to two primary limitations. First, trajectories were aggregated to origin-destination pairs, overlooking localized network behavior and inducing severe data sparsity. Second, operational edge weights were explicitly excluded to prevent target leakage. Despite employing node centrality metrics and hyperparameter tuning, performance remained near random baseline levels. Kämper suggested this failure likely associated with data scarcity and a lack of external feature richness rather than inherent methodological flaws.

## 2.2 Model Complexity and Data Integrity

2.2.1 The Interpretability Trade-of. Within the context of prior work on Dutch railway delay prediction, three complementary research directions emerge. Kämper employed a topological, tabular framework on monthly aggregated data, demonstrating limited generalization under simultaneous and non-simultaneous evalu ation constrained by data sparsity and restricted feature richness [22]. Brakenhof adapted this approach by incorporating operational and environmental features, improving predictive performance for disruption and cancellation tasks [10]. In contrast, van der Maas introduced hybrid spatio-temporal deep learning architectures (GAT-GRU and GAT-LSTM), achieving substantially stronger non-simultaneous predictive performance through richer modeling of temporal dependencies [40].

Building upon these contributions, this thesis prioritizes interpretability and feature stability analysis over absolute model complexity. While hybrid deep learning architectures (GATs) capture complex spatiotemporal dependencies, their attention weights func tion primarily as structural proxies and rarely provide explicit feature explainability regarding why a specific prediction is made [7, 19, 20, 37, 42]. Because deployed models within critical public infrastructure must satisfy strict transparency requirements under the EU AI Act [16], this research utilizes tree-based ensembles integrated with SHAP to calculate specific feature attributions [5, 30, 31]. While SHAP values quantify correlational predictive reliance rather than causality, they provide the necessary diagnostic transparency for regulatory compliance [5, 18]. By trading the raw predictive density of deep learning for the diagnostic capability of SHAP, this approach rigorously evaluates feature robustness over time.

Furthermore, while prior work demonstrated the eficacy of multi-year seasonal models to capture recurring patterns [10], this study explicitly restricts its scope to consecutive cross-month generalization to strictly isolate the immediate temporal degradation of the feature space.

2.2.2 Environmental Data Integrity. Prior methodologies also exhibited environmental data limitations. Previous studies utilizing nearest-neighbor approximations of relocated KNMI weather station data likely introduced structural measurement noise and spatial inconsistencies [10, 23]. Consequently, the limited predictive contribution of environmental features in prior work may reflect data aggregation flaws rather than an absence of meteorological signal. To mitigate this noise, this methodology extracts historical weather data utilizing exact station coordinates [40, 43].

## 2.3 Temporal Instability and Explainability

Beyond representational constraints, temporal instability remains a critical challenge. Lei attributed near-random non-simultaneous performance to unstable feature importance for the Brazil bus dataset, where models overfit to the structural patterns ofindividual training snapshots [26]. To systematically diagnose this degradation, SHAP values are employed as a post-hoc interpretability tool to track feature stability across temporal contexts. Crucially, SHAP is restricted to interpretation rather than feature selection, as its use in selection can artificially degrade predictive performance [10, 17]. Recent state-of-the-art studies continue to emphasize SHAP as a critical tool for interpreting model behavior across diverse domains [12, 27].

## 2.4 Research Gap Summary

The existing literature demonstrates that long-term network-wide delay prediction is constrained by data sparsity and feature exclusion [22]. Conversely, deep learning methodologies that successfully capture these complex spatiotemporal dependencies sacrifice the diagnostic interpretability required for deployment in critical public infrastructure [40].

To address this fundamental gap, this research diverges from prior static frameworks by proposing a highly granular, multilayered tabular approach. By integrating dynamic topological graphs, operational edge weights, and exact-coordinate environmental data within interpretable tree-based ensembles, this study navigates the trade-of between representational richness and diagnostic transparency [8, 10, 28, 40, 45].

To statistically evaluate the predictive capacities of this proposed architecture, the following hypotheses are formulated:

• H1 (Simultaneous Validity): Under simultaneous testing (trajectory-based split), tabular ensemble performance will significantly exceed the 50% null baseline, establishing the discriminative validity of the engineered features.

• H2 (Temporal Generalization): Under non-simultaneous testing (trajectory-based split), performance will exhibit a statistically significant degradation compared to simultaneous evaluations.

• H3 (Structural Degradation): Under a strict time-based split, performance will significantly degrade across all evaluated models during non-simultaneous horizons, suggesting that temporal instability is an inherent operational phenomenon rather than an artifact of sample size fluctuations.

While these hypotheses address the statistical significance of performance shifts (SRQ1), the internal feature dynamics underlying these shifts (SRQ2) are subsequently evaluated via post-hoc SHAP analysis.

## 3 Methodology

This section details the analytical framework and feature engineering procedures used to evaluate the temporal predictability of network-wide train delays. Rather than optimizing strictly for absolute algorithmic accuracy, the methodology utilizes comparative evaluation designs to isolate feature performance across consecutive monthly horizons. A comprehensive overview is visualized in Appendix B Figure 1.

## 3.1 Data Acquisition and Pre-processing

3.1.1 Railway Operations Data and Spatial Filtering. The operational dataset, sourced from RijdenDeTreinen [1], comprises historical services from 2019 to 2024, a station distance matrix, and reference spatial data. Freight and international services were excluded to isolate standard domestic passenger operations, yielding an initial dataset of 83,810,933 raw station stop records.

These records were chronologically sequenced into directed trajectories (scheduled connections between consecutive stations), adopting the high-density spatial formulation utilized by Braken hof [10] to systematically increase data density. To ensure topological validity, bidirectional spatial filtering was applied: trajectories containing unreferenced stations, and reference stations lacking historical trajectories, were dropped. This guarantees all topologi cal features rely exclusively on verified spatial distances within the Netherlands (Appendix J Figure 9b).

3.1.2 Imputation andService Cancellations. Missing terminal timestamps were imputed deterministically based on physical network constraints: missing arrivals for starting services and missing departures for terminating services were set to zero delay. Both partially and fully canceled services were retained and assigned an arrival delay penalty $( > 0 )$ . As established in Section 1, this framework explicitly retains both partially and fully canceled services within the predictive target, diverging from prior baselines [22]. This inclusive definition recognizes a cancellation as a literal topological edge removal, thereby capturing actual network-wide disruption [15, 24]. Furthermore, current operational data lacks the ’slowly changing dimensions’ to track the change of the cancellation status over time, which is necessary to diferentiate ad-hoc operational failures from planned maintenance cancellations $( \mathrm { e . g . }$ , exact announcement timestamps). Given this limitation, all cancellation instances are conservatively treated as delays.

3.1.3 Environmental Data Acquisition andCleaning. Historical hourly weather data for exact station coordinates was acquired via the Open-Meteo API [46]. Duplicate timestamp entries, occurring annually during daylight saving time (DST) transitions, were retained.

Because the underlying meteorological values remain distinct despite identical timestamps, retaining these sequential measurements preserves the statistical integrity of subsequent monthly aggregations. However, this assumption could be revisited in future work.

3.1.4 Temporal Aggregation and Target Definition. Trajectory records were aggregated into discrete monthly snapshots to capture macroscopic network states. An activity threshold of at least four scheduled rides per month was applied to filter anomalous routing noise [22]. Preprocessing decisions are summarized in Appendix B Table 2.

To establish baseline operational reliability, the delay ratio for a given trajectory $T$ was defined as:

$$
\mathrm { r a t i o \_ a r r i v a l \_ d e l a y } _ { T } = \frac { \mathrm { s c h e d u l e d \ r i d e s \ w i t h \ a r r i v a l \ d e l a y } \ > 0 } { \mathrm { t o t a l \ s c h e d u l e d \ r i d e s } }
$$

To mitigate majority-class bias during model training, the binary target label $Y _ { T }$ (’Is Significantly Delayed’) was established via a global median threshold. The target definition was adapted for comparability with prior work and controlled class balance [22]:

$$
Y _ { T } = \left\{ \begin{array} { l l } { { 1 } } & { { \mathrm { i f ~ r a t i o \_ a r r i v a l \_ d e l a y } _ { T } > 0 . 2 6 2 9 } } \\ { { 0 } } & { { \mathrm { i f ~ r a t i o \_ a r r i v a l \_ d e l a y } _ { T } \le 0 . 2 6 2 9 } } \end{array} \right.
$$

While the initial median-based threshold ensures algorithmic class balance, it introduces vulnerability to temporal label shift as actual delay ratios fluctuate (Appendix J Figure 11). To verify that temporal degradation is not solely an artifact of this static definition, a sensitivity analysis utilizing XGBoost (under the trajectory-based splitting schema) evaluates two alternative thresholds: an Absolute Operational Threshold (fixed 80% punctuality, in other words a 0.20 threshold), chosen in alignment with railway practice as a conservative variant of the NS 3-minute minimum floor (84.4%) [33], and a Month Relative Historical Seasonal Threshold. Evaluating predictive degradation across these varied definitions enables a diagnosis of whether temporal instability is driven by the global median target formulation itself.

Furthermore, to enhance operational utility, the experimental framework evaluates a regression model across two operational resolution levels as an additional sensitivity analysis: Schedule Adherence (0-minute threshold) and Regulatory Punctuality (3-minute threshold). These regression targets provide a granular assessment of service reliability, complementing the classification tasks. The underlying regression models are optimized using a logistic loss objective function (reg:logistic). This approach applies a logistic transformation to the model outputs, thereby ensuring that all continuous punctuality and adherence ratio predictions are naturally bounded within the interval [0, 1]. Model parameters are subsequently learned by minimizing the negative log-likelihood, providing a mathematically robust framework for fractional response estimation. By comparing predictive performance across these diverse classification and regression formulations, this study demonstrates whether the model’s eficacy is robust regardless of the chosen operational strictness or target architecture.

To evaluate the operational utility of these continuous formulations, the models were assessed using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). These metrics were contrasted against a Null Baseline (predicting the mean target value of the training set) to confirm that the models retain continuous predictive signal extraction capabilities alongside their classification boundaries.

![](images/d934005247468badfb63a83f7234442d84fbd505e5d0c0ca9e4ba140015af676.jpg)  
Figure 1: Research Scope and Methodologies

While exploratory time-series analysis indicates that pure delays and outright cancellations exhibit distinct volumetric trends (Appendix M Figure 20), their combined delays to structural routing integrity necessitates a unified predictive target for macroscopic evaluation.

## 3.2 Feature Engineering

Predictive features across operational, spatial, and environmental domains were engineered at the aggregated monthly trajectory level to align with long-term predictive horizons.

3.2.1 Operational Feature Engineering. Operational context was established using the total service volume of the last active month and the elapsed months since the previous service. Inspired by the network operational features introduced by Brakenhof [10], a decayed operational edge weight $\left( W _ { \mathrm { d e c a y e d } } \right)$ was calculated to approximate sequential pattern recognition without inducing target leakage, as it relies exclusively on past information:

$$
W _ { \mathrm { d e c a y e d } } = V _ { t - 1 } { \frac { 1 } { \Delta m } } \quad \mathrm { f o r } 1 \leq \Delta m \leq 1 2
$$

where $V _ { t - 1 }$ is the most recent active service volume and Δ� is the months elapsed. A maximum 12-month lookback aligns with the annual NS scheduling cycle. The feature behaves as a proxy for the recent service rate because it anchors the model’s focus on the most recent active month while proportionately scaling down its relevance based on elapsed inactivity. If no prior active month was identified, Δ� was imputed to 12.

3.2.2 Spatial and Topological Feature Engineering. Dynamic, directed network graphs $G _ { t } ~ = ~ ( V _ { t } , E _ { t } )$ were constructed for each month (Appendix J Figure 9a). Static graphs were avoided to capture seasonal topological shifts. Using physical distances and the decayed operational edge weights $( W _ { \mathrm { d e c a y e d } } )$ , node-level features (e.g., weighted outbound/inbound degrees, average distances) were extracted to quantify a station’s outward distribution capabilities and localized congestion potential, expanding upon the baseline features established by Brakenhof [10]. Concurrently, trajectorylevel features (e.g., Jaccard Coeficient, Adamic-Adar Index) were calculated to evaluate routing redundancy (Appendix N Table 23).

3.2.3 Environmental WeatherFeatures. Hourly station-level historical weather data was aggregated into monthly station-level features (e.g., air temperature, rain, wind speed, wind gusts, snowfall, snow depth, and soil temperature) to scale the daily predictive validity of these specific variables established by van der Maas [40] into macroscopic seasonal network vulnerabilities (Appendix N Table 22).

3.2.4 Feature Set Construction and Ablation Logic. To isolate specific predictive drivers, a comprehensive ablation logic was established. Feature sets (Topological, Weight, Weather, Operational) are first evaluated independently to establish baselines, followed by incremental integration (e.g., Topology + Weight, Topology +

Weight + Weather, Topology + Weight + Operational, All Features) to assess complex interactions.

Furthermore, explicit feature selection prior to training was omitted. While this risks retaining highly collinear variables, particularly spatially proximate environmental metrics and could increase overfitting under temporal shift, it is a deliberate diagnostic choice. Retaining the unfiltered feature space allows tree-based algorithms to inherently manage multicollinearity during node splitting, while enabling post-hoc SHAP analysis to identify exactly which variables introduce noise or degrade over time. Consequently, aggressive feature pruning optimized strictly for raw accuracy is designated as future work.

3.2.5 Exploratory Data Analysis (EDA). The aggregated dataset comprises 49,793 stable trajectories across 43 engineered features (Appendix N Table 22). Structurally, the network maintains an average of247 station nodes and 692 active trajectory edges per monthly snapshot. While scheduled passenger service volumes exhibited minimal variance across the temporal horizon (Appendix J Figure 12), an upward trend in unique trajectory edges was observed from 2022 onward (Appendix J Figure 14). Finally, although the overarch ing target class distribution is mathematically balanced at a 50/50 ratio (Appendix J Figure 10), the underlying proportion of significantly delayed trajectories exhibits natural operational volatility, fluctuating between 10% and 90% across individual months (Appen dix J Figure 14).

## 3.3 Evaluation Framework and Experimental Setup

This subsection details the validation methodologies, dataset splitting strategies, and model configurations utilized to systematically benchmark predictive temporal generalizability. A universal random seed was uniformly established across all computational environments for reproducibility.

3.3.1 Temporal Validation Phases and Data Spliting. To assess out-of-sample generalizability, models were first evaluated under Simultaneous Testing (trained and tested within the same monthly snapshot) and Non-Simultaneous Testing (trained on historical monthly snapshot, tested on unseen future months) [22, 26].

These phases utilized two data splitting schemas (Appendix B Table 3). A strict time-based split (70/30 chronological division) isolated temporal degradation while controlling for sample size. Ad ditionally, a trajectory-based split with random under-sampling was applied exclusively to the XGBoost model to maintain alignment with prior literature and standardize SHAP evaluations [26].

3.3.2 Model Configuration. While Deep Learning architectures (e.g., GNNs) might ofer high predictive accuracy in spatiotemporal tasks, tree-based ensembles were selected as the primary methodology to prioritize feature interpretability and diagnostic transparency, which are often obscured in black-box deep learning models. The models evaluated include Logistic Regression (serving as a linear baseline) alongside three tree-based ensembles: Random Forest, XGBoost, and LightGBM. These specific non-linear models were selected for their native capacity to manage the complex feature interactions and spatial dependencies inherent in transportation networks [32, 39]. Prior to training, standard scaling normalized the continuous feature space to prevent variables with expansive numerical ranges from disproportionately dominating the linear baseline.

Default hyperparameters were predominantly preserved to prioritize feature evaluation over algorithmic optimization, with minor operational deviations (e.g., max iterations, binary objectives) detailed in Appendix N Table 24. To show baseline conclusions did not hinge on extensive tuning, a randomized search cross-validation was executed on the SHAP-evaluated model (XGBoost, trajectorybased split). To prevent look-ahead bias, this sanity check utilized a Time Series Split (TSCV) constrained to 20 iterations across 5 folds. The search space was defined as subsample: [0.6, 0.8, 1.0], n\_estimators: [100, 200, 300], max\_depth: [3, 5, 7], learning\_rate: [0.01, 0.05, 0.1], and colsample\_bytree: [0.6, 0.8, 1.0].

3.3.3 Metrics and Statistical Inference. Balanced Accuracy serves as the primary evaluation metric. By computing the unweighted average recall across classes, it prevents artificial performance inflation and directly addresses the inherent stochasticity of railway delays:

$$
{ \mathrm { B a l a n c e d ~ A c c u r a c y } } = { \frac { 1 } { 2 } } \left( { \frac { T P } { T P + F N } } + { \frac { T N } { T N + F P } } \right)
$$

In the highly stochastic domain of railway operations, where predictability is inherently constrained by unobserved external variables, even marginal improvements of a few hundredths in balanced accuracy hold significant operational meaning, provided they demonstrate statistical stability over time. F1 Score and Receiver Op erating Characteristic Area Under the Curve (ROC AUC) were calculated as supplementary metrics specifically within the time-based splitting configuration to facilitate comprehensive model benchmarking. Conversely, the trajectory-based evaluation focused primarily on Balanced Accuracy to provide a consistent, unweighted diagnostic signal during the SHAP-based feature stability analysis and target threshold sensitivity tests. Furthermore, to isolate genuine model learning from overfitting, a Null Baseline Model was established by randomly shufling the target labels prior to training to generate a strict, non-informative performance floor equating to random chance.

Predictive performance was evaluated via Global Aggregation (micro-averaging) to establish holistic benchmarks comparable to prior foundational frameworks [22, 26], and Temporal Aggregation (macro-averaging per month) for statistical testing.

To ensure valid paired statistical testing between simultaneous and non-simultaneous performance, results required temporal alignment. For each test month �, the simultaneous configuration yielded a single score $( S _ { t } )$ , whereas the non-simultaneous configuration yielded multiple scores from preceding training months. To resolve this dimensionality mismatch, non-simultaneous scores were aggregated via the arithmetic mean $\begin{array} { r } { ( \bar { N } _ { t } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } N _ { t , i } ) } \end{array}$ , establishing the formal paired observation $( S _ { t } , \bar { N } _ { t } )$ for each test month �. Because subsequent Shapiro-Wilk tests yielded inconsistent normality results across various model and feature configurations, the assumption of a universal normal distribution was rejected. Consequently, the non-parametric Wilcoxon signed-rank test (� = 0.05) was uniformly adopted [14], ensuring observed temporal degradation is statistically verifiable.

Finally, to facilitate temporal analysis, temporal aggregation was utilized to track distribution shifts in both predictive performance and the top three SHAP-identified features. These shifts were quantified using the Interquartile Range (IQR, bounded by the 25th and 75th percentiles). This explicitly isolates the true within-month dispersion of the feature space, while concurrently quantifying the variance in historical predictive stability for each target month across the sequential testing horizons.

While the IQR quantifies within-month dispersion, explicitly evaluating the temporal robustness of the model explanations required further non-parametric analysis. To this end, the Kendall rank correlation coeficient (�) was employed to quantify the ordinal consistency of SHAP feature importance rankings across sequential temporal transitions (period � to � + 1). This established whether the fundamental hierarchy of feature prioritization remained stable over time, with statistical significance evaluated against an alpha threshold of $\alpha = 0 . 0 5$

Furthermore, to isolate the underlying mechanisms driving any observed instability in the SHAP values, a two-sample Kolmogorov-Smirnov (K-S) test was concurrently incorporated into the analytical framework. By applying the K-S test directly to the raw input features across the corresponding temporal periods, the methodology quantifies actual distributional shifts within the environment. This dual-testing strategy systematically diferentiates between stochastic algorithmic variance and true concept drift within the underlying railway data structures.

3.3.4 Temporal Baselines. To rigorously benchmark the generalization capabilities of the models, two temporal baselines were introduced: a Naive baseline (persistence based on the prior month, � − 1) and a Seasonal Naive baseline (SNaive, utilizing the identical calendar month from the prior year, � −12). Because the railway network topology is dynamic and edges may not operate every month, these baselines encounter a cold-start problem. Consistent with standard practices in temporal network analysis, missing observations were handled via zero imputation, conservatively predicting “no delay” for trajectories without historical precedent.

Approximately 3.5% of Naive and 5.2% of SNaive predictions rely on this default imputation (Appendix F Table 17) and Figure 6. Because defaulting to zero artificially inflates True Negatives, Balanced Accuracy is specifically utilized to mitigate survivorship bias by equally weighting both classes. To ensure a fair, equal-� comparison $( N = 6 0 )$ , the first 12 months of the dataset were truncated for these temporal baseline evaluations (Appendix F Table 16).

## 3.4 Hardware and Software Requirements

Experiments were executed on the Snellius supercomputing cluster utilizing NVIDIA A100 GPUs (40GB memory) via the University of Amsterdam [38]. Data preprocessing utilized the DuckDB Python API [35]. All model dependencies were version-controlled and specified in the environment configuration file.

## 4 Results

This section details the empirical evaluation of the network-wide delay prediction models. The findings sequentially address temporal generalization capabilities across distinct testing paradigms (SRQ1) and evaluate temporal feature stability via SHAP diagnostics (SRQ2).

Sample Size Variations Across Configurations. The final number of temporally aligned pairs (�) varies marginally across experimental configurations due to pipeline constraints during non-simultaneous forecasting aggregation. Baseline evaluations within the simultaneous context retain the full dataset of 72 monthly horizons. For paired simultaneous versus non-simultaneous evaluations, the trajectorybased data split yields 71 aligned pairs, whereas the time-based algorithmic comparisons yield 66 aligned pairs. Because the Wilcoxon signed-rank tests are applied independently within each configuration, these minor variations do not compromise statistical validity.

## 4.1 SRQ1: Evaluation of Temporal Generalization

This subsection evaluates the predictive performance of the engineered feature sets across simultaneous and non-simultaneous testing horizons to address Hypotheses 1, 2, and 3.

4.1.1 Hypothesis 1: Simultaneous Predictive Performance. Under trajectory-based simultaneous testing, all XGBoost feature configurations demonstrated discriminative predictive capacity significantly exceeding the null baseline (Table 2a and Appendix K Figure 17a). One-sided Wilcoxon signed-rank tests confirmed the predictive validity across all evaluated feature subsets $( p < . 0 0 1$ Appendix D Table 7), supporting Hypothesis 1. Notably, the isolated topological features generated a significant median accuracy gain of 0.075 over the null expectation $( W = 2 5 9 9 . 0 , \phi < . 0 0 1 , r = 0 . 6 0 1 )$ The fully integrated feature sets (All Features) expanded this median gain to 0.127 $( W = 2 5 9 5 . 0 , \phi < . 0 0 1 , r = 0 . 5 9 9 )$ , demonstrating fundamental predictive validity before temporal evaluation.

4.1.2 Hypothesis 2: Temporal Generalization Degradation. Transitioning to non-simultaneous generalization testing, predictive performance exhibited a structural degradation pattern across most evaluated feature configurations (Table 2b and Appendix K Figure 17b). One-sided paired Wilcoxon signed-rank tests confirmed that these temporal performance decreases were statistically significant across all configurations (� < .001, Appendix D Table 8). Notably, the fully integrated architecture (All Features) experienced a significant median accuracy loss of 0.069 when shifted from simultaneous to non-simultaneous evaluation $( W = 2 5 2 9 . 0 , \phi < . 0 0 1 , r = 0 . 6 0 2 )$

However, this degradation is not universal. Under strict timebased splits, isolated topological feature sets successfully resisted temporal decay across the tree-based models, including XGBoost $( W = 3 8 4 , \dot { p } = 1 . 0 0 0 , r = 0 . 4 0 1 )$ , RandomForest $( W = 3 7 2 . 0 , p <$ $1 . 0 0 , r = 0 . 4 0 8 )$ , and $\operatorname { L G B M } \left( W = 5 5 7 . 0 , p = 1 . 0 0 , r = 0 . 3 0 5 \right)$ (Appendix D Table 9). All remaining configurations, including integrated feature sets, logistic regression baselines, trajectory-based XGBoost models, hyperparameter-tuned variants, and alternative threshold formulations, consistently exhibited statistically significant degradation $( p < . 0 0 1 )$ , while several intermediate models (e.g., Topological + Weight of XGBoost and RandomForest) demonstrated no significant change (Appendix D Tables 9 10 11 13).

Incremental ablation analysis within the trajectory-based XG-Boost configuration further clarifies these architectural dynamics (Appendix D Table 12). The integration of weight-based features into the baseline topological framework yielded a statistically sig nificant performance improvement, generating a median accuracy gain of 0.018 $( W \ = \ 3 6 8 . 0 , \ / p \ < \ . 0 0 1 , r \ = \ 0 . 6 1 1 )$ . Conversely, the inclusion of environmental variables (weather features) actively degraded predictive capacity; the enhancement hypothesis was completely rejected $( p = 1 . 0 0 0 )$ as the median accuracy decreased. In contrast, the addition of operational features consistently enhanced predictive stability when combined with the topological and weight representations $( W = 6 9 9 . 0 , \beta < . 0 0 1 , r = 0 . 2 7 8 )$ . Ultimately, the comparison between the base topological model and the optimal Topological + Weight + Operational architecture confirmed a statistically significant aggregate improvement $( W = 2 1 . 0 , p <$ $. 0 0 1 , r = 0 . 6 0 4 )$ . These findings partially support Hypothesis 2, indicating that temporal robustness is likely configuration-dependent and compromised by meteorological noise.

Figure 2: Trajectory-Based Split Results: XGBoost Performance Across Testing Paradigms (Global Aggregation)
<table><tr><td>Feature Set</td><td>Balanced Acc</td><td>Balanced Acc Null</td></tr><tr><td>Topological Features</td><td>0.584</td><td>0.500</td></tr><tr><td>Weight Features</td><td>0.606</td><td>0.495</td></tr><tr><td>Weather Features</td><td>0.573</td><td>0.495</td></tr><tr><td>Operational Features</td><td>0.592</td><td>0.501</td></tr><tr><td>Topological + Weight</td><td>0.611</td><td>0.511</td></tr><tr><td>Topological + Weight + Weather</td><td>0.630</td><td>0.506</td></tr><tr><td>Topological + Weight + Operational</td><td>0.624</td><td>0.494</td></tr><tr><td>All Features</td><td>0.634</td><td>0.504</td></tr></table>

(a) Simultaneous Testing vs. Null Accuracy

<table><tr><td>Feature Set</td><td>Balanced Acc</td><td>Balanced Acc Null</td></tr><tr><td>Topological Features</td><td>0.556</td><td>0.490</td></tr><tr><td>Weight Features</td><td>0.570</td><td>0.498</td></tr><tr><td>Weather Features</td><td>0.501</td><td>0.500</td></tr><tr><td>Operational Features</td><td>0.541</td><td>0.495</td></tr><tr><td>Topological + Weight</td><td>0.574</td><td>0.492</td></tr><tr><td>Topological + Weight + Weather</td><td>0.556</td><td>0.502</td></tr><tr><td>Topological + Weight + Operational</td><td>0.582</td><td>0.506</td></tr><tr><td>All Features</td><td>0.565</td><td>0.502</td></tr></table>

(b) Non-Simultaneous Testing vs. Null Accuracy

<table><tr><td>Feature Set</td><td>Balanced Acc</td><td>Balanced Acc Null</td></tr><tr><td>All Features (Simultaneous)</td><td>0.653</td><td>0.499</td></tr><tr><td>All Features (Non-Simultaneous)</td><td>0.577</td><td>0.506</td></tr></table>

(c) Hyperparameter Tuned (All Features Set Configuration)

4.1.3 Methodological Sanity Check: Hyperparameter Optimization. targeted hyperparameter optimization was applied to the fully integrated feature set. The tuned configuration yielded a comparable significant degradation pattern between simultaneous and nonsimultaneous testing paradigms (Figure 2c and Appendix D Table 13), confirming that performance decay is not attributable to model mis-specification.

4.1.4 Hypothesis 3: Temporal Degradation Across Time-Based Splits. To isolate temporal drift from sampling efects, a strict time-based split was implemented across multiple models. While global aggregations suggest an overall reduction in predictive performance under non-simultaneous testing (Tables 1a and 1b), Wilcoxon tests reveal a more nuanced pattern.

Statistically significant degradation is observed across the majority of feature configurations and algorithms, particularly within highly integrated sets heavily reliant on environmental data (e.g., All Features, $p \ < \ . 0 0 1 )$ . However, isolated topological representations completely resisted this temporal decay across all evaluated tree-based algorithms (LGBM, Random Forest, and XGBoost); non-simultaneous performance marginally exceeded simultaneous baselines, yielding a complete absence of degradation $( p = 1 . 0 0 0 )$ Furthermore, specific hybrid configurations, such as the Topological + Weight + Operational architecture, demonstrated robust temporal stability within the tree-based frameworks, yielding no statistically significant degradation (e.g., XGBoost $p = . 6 9 6 ,$ , LGBM � = .600) (Appendix D Table 9).

These findings partially support Hypothesis 3, indicating that temporal generalization decay is not uniform but instead depends critically on feature composition and model class.

Table 1: Time-Based Split Results: Balanced Accuracy across Simultaneous and Non-Simultaneous Testing (Global Aggregation).  
(a) Simultaneous Testing Results
<table><tr><td>Classifier</td><td>TOP</td><td>WGT</td><td>WTH</td><td>OPS</td><td>TW</td><td>TWCW</td><td>TWCO</td><td>ALLF</td></tr><tr><td>LGBMClassifier</td><td>0.639</td><td>0.673</td><td>0.637</td><td>0.644</td><td>0.683</td><td>0.697</td><td>0.693</td><td>0.705</td></tr><tr><td>LogisticRegression</td><td>0.579</td><td>0.606</td><td>0.575</td><td>0.586</td><td>0.603</td><td>0.627</td><td>0.628</td><td>0.642</td></tr><tr><td>RandomForest</td><td>0.675</td><td>0.697</td><td>0.665</td><td>0.673</td><td>0.707</td><td>0.716</td><td>0.717</td><td>0.723</td></tr><tr><td>XGBClassifier</td><td>0.646</td><td>0.676</td><td>0.638</td><td>0.654</td><td>0.680</td><td>0.695</td><td>0.692</td><td>0.702</td></tr></table>

(b) Non-Simultaneous Testing Results
<table><tr><td>Classifier</td><td>TOP</td><td>WGT</td><td>WTH</td><td>OPS</td><td>TW</td><td>TWCW</td><td>TWCO</td><td>ALLF</td></tr><tr><td>LGBMClassifier</td><td>0.593</td><td>0.583</td><td>0.495</td><td>0.558</td><td>0.594</td><td>0.555</td><td>0.600</td><td>0.563</td></tr><tr><td>LogisticRegression</td><td>0.541</td><td>0.548</td><td>0.483</td><td>0.559</td><td>0.550</td><td>0.492</td><td>0.573</td><td>0.497</td></tr><tr><td>RandomForest</td><td>0.592</td><td>0.577</td><td>0.499</td><td>0.554</td><td>0.588</td><td>0.535</td><td>0.592</td><td>0.542</td></tr><tr><td>XGBClassifier</td><td>0.595</td><td>0.580</td><td>0.499</td><td>0.557</td><td>0.593</td><td>0.554</td><td>0.599</td><td>0.563</td></tr></table>

4.1.5 Temporal Baseline Comparisons. The evaluation of the temporal baselines explicitly highlights the dificulty ofnon-simultaneous forecasting. Over the $N = 6 0$ evaluated months, the simple Naive persistence model achieved a ���������������� = 0.8080 (������ = 0.8246) (Appendix F Table 16 and Figures 4, 5). The Seasonal Naive model achieved a mean ���������������� = 0.7190 (������ = 0.7257). The strength of the immediate � − 1 Naive baseline suggests that underlying network states are highly auto-correlated on a month-to-month basis. While the tabular XGBoost architectures excel at isolating complex interactions within localized simultaneous snapshots, the raw predictive floor established by simple temporal persistence underscores the likely necessity of anchoring future predictive frameworks in dynamic, season-aware memory rather than static tabular horizons.

## 4.2 SRQ2: Model Behavior and Temporal Feature Stability

Addressing SRQ2, SHAP values are utilized strictly as a correlational diagnostic tool to quantify temporal shifts in predictive reliance, rather than to infer physical network causality.

4.2.1 Static Feature Importance: A Single-Month Snapshot. Singlemonth analysis (Appendix Q Figure 28d, January 2024) reveals diverse feature correlations with predictions. The operational metric Target Weighted Outbound Degree emerges as a primary predictor, with higher values associated with increased predicted delay likeli hood. Environmental variables, such as Target Mean Temperature (2m), also show strong positive associations within this specific winter snapshot, while topological features exhibit mixed directional efects.

4.2.2 Feature Volatility and Long-Term Stability. Across multiple temporal horizons, feature importance rankings exhibit substantial instability (Appendix O Figure 21). All isolated feature sets demonstrate fluctuating importance patterns, with environmental features showing the highest volatility and operational features displaying comparatively greater consistency.

Within the fully integrated model (Appendix O Figure 23b), the feature Decayed edge weight exhibits the highest relative stability in SHAP rankings, followed by Target weighted outbound degree and Source mean soil temperature. Nevertheless, even these top-ranked features display notable temporal variability, indicating the absence of strictly stable predictors.

Temporal analysis of the top three features further highlights this distinction (Appendix J Figure 16). Decayed edge weight (1493.23 [1026.00–1999.00]) and Target weighted outbound degree (6122.92 [2437.00–8304.00]) maintain relatively stable temporal means while exhibiting substantial within-month dispersion. In contrast, Source mean soil temperature (11.60 [6.48–16.85]) demonstrates pronounced temporal shifts in its mean with minimal within-month variance.

These findings indicate that structural and operational features provide comparatively more stable predictive signals, whereas environmental features are disproportionately associated with temporal drift, aligning with the observed degradation patterns in SRQ1.

4.2.3 Quantitative Assessment ofExplanation Stability and Concept Drift. To formalize the descriptive observations of feature volatility, Kendall rank correlation coeficients (�) were calculated across the 71 sequential temporal transitions for each feature set utilizing the trajectory based XGBoost base model configuration. This analy sis reveals substantial variance in explanation stability across the evaluated feature sets, confirming the visual diagnostics.

As detailed in Appendix G Table 18, the OPS configuration demonstrated the highest temporal stability (mean $\tau = 0 . 8 3 1 )$ , with all sequential transitions exhibiting statistical significance $( p < 0 . 0 5 )$ This suggests a consistent prioritization of the operational feature hierarchy over the observation period. Conversely, configurations such as WGT and WTH exhibited severe rank instability. These sets were characterized by lower mean correlations and isolated periods of negative ordinal association, indicating complete inversions of feature importance. Furthermore, the WGT configuration achieved statistical significance in only 16.9% (12 of 71) of transitions, suggesting that the observed ranking variations within this subset are largely attributable to algorithmic noise rather than systematic shifting. Configurations encompassing larger feature spaces, specifically TWCO and ALLF, maintained moderate average stability while demonstrating nearly universal statistical significance across transitions.

To contextualize this explanatory instability, the two-sample Kolmogorov-Smirnov (K-S) test results were evaluated to quantify actual distributional shifts within the raw feature space (Appendix I Tables 19, 20 and 21). This macroscopic analysis revealed a stark categorical divide in environmental stability. Weather features (WTH) exhibited extreme volatility, with statistically significant distributional drift occurring in 84.8% of all sequential temporal transitions. In contrast, Operational (OPS), Topological (TOP), and Weight (WGT) features demonstrated high environmental stability, exhibiting significant drift in only 3.29%, 6.94%, and 1.41% of transitions, respectively. By pairing these explanatory metrics with the distributional analysis, a distinct pattern emerges: periods of significant SHAP rank inversion frequently align with statistically significant distributional shifts in the underlying feature space. Therefore, the explanatory instability recorded in specific configurations, particularly those reliant on environmental variables, does not inherently indicate algorithmic fragility. Instead, it reflects the model dynamically reprioritizing variables in response to true concept drift within the operational environment. This separation of explanation-centric variance from data-centric drift confirms that the algorithm accurately captures structural changes in the modeled railway network.

## 4.3 Robustness Check: Alternative Target Thresholds

Across all threshold definitions, models retained strong simultaneous discriminative performance (e.g., 0.643 to 0.646 for the fully integrated configuration), confirming that predictive signal extraction remains valid within localized temporal snapshots.

However, non-simultaneous performance patterns remained highly dependent on the chosen threshold definition. Under the Month Relative Threshold, which preserves class balance within each calendar month, the fully integrated model continued to exhibit substantial and statistically significant degradation (0.646 to 0.556, $\Delta = - 0 . 0 9 0$ , Appendix D Table 11), indicating that relative normalization does not mitigate temporal drift.

In contrast, the Absolute Threshold (> 20% delayed rides) yielded a diferent pattern for structural feature sets. Isolated Topological (0.570 to 0.580, $\Delta = + 0 . 0 1 0 )$ and Weight (0.624 to 0.618, Δ = −0.006) configurations showed no statistically significant diference between simultaneous and non-simultaneous performance (Appendix D Table 10), suggesting improved temporal consistency under this formulation.

Nevertheless, this efect did not generalize across all feature categories. Isolated Weather features continued to exhibit significant degradation (0.575 to 0.501, Δ = −0.074), and the fully integrated model also demonstrated a statistically significant performance decline under the absolute definition (0.643 to 0.604, Δ = −0.039).

Overall, these findings indicate that while alternative target definitions can reduce apparent temporal sensitivity for specific structural feature sets, they do not eliminate generalization decay in more complex or environmentally enriched configurations.

## 4.4 Robustness Check: Regression Formulations

To verify that the predictive validity ofthe engineered features is not merely an artifact of binary thresholding, model performance was evaluated within a continuous regression space targeting Schedule Adherence (0-minute delay) and Regulatory Punctuality (3-minute delay).

Under simultaneous testing, the models successfully extracted continuous predictive signals, consistently outperforming the null baseline (Appendix E Tables 14 and 15). For Regulatory Punctuality, the optimal Topological + Weight + Operational configuration achieved an $M A E = 0 . 0 5 3 2$ and an $R M S E = 0 . 1 0 7 3$ , notably improving upon the Null baseline $( M A E = 0 . 0 9 3 9 , { \mathrm { R M S E ~ } } R M S E = 0 . 1 5 3 2 ) .$ However, mirroring the classification results, predictive decay occurred under non-simultaneous horizons. For Regulatory Punctuality, the same configuration degraded to an $M A E = 0$ .0944 and an $R M S E = 0 . 1 5 7 1$ , closing the gap with the Null baseline $( M A E = 0 . 1 0 6 2 , R M S E = 0 . 1 6 8 1 )$ . A similar trend was observed under the stricter Schedule Adherence threshold. Ultimately, this indicates that while the feature sets possess continuous predictive capacity within-month, they remain structurally vulnerable to temporal drift across future horizons.

## 5 Discussion

This section contextualizes the empirical findings within the current state of railway delay prediction. To clarify predictive and interpretability trade-ofs, the methodology is positioned against foundational literature. Subsequently, a structural diagnosis of tem poral degradation is provided, empirically demonstrating that while the models learn meaningful signals within-month, this capacity systematically degrades across future months. Crucially, this degradation is diagnosed not as a parametric tuning artifact, but as the consequence of label shift, covariate shift, and concept drift. Finally, methodological limitations regarding validity and generalizability are assessed to guide future research trajectories.

## 5.1 Positioning within the State-of-the-Art

This research implemented theoretical recommendations to evaluate whether granular feature engineering could mitigate the temporal generalization challenges inherent in railway networks. Specifi cally, prior work suggested that integrating external operational and environmental variables within a stop-to-stop topology would yield a more robust predictive model [10, 22]. By adopting this multi-layered framework and expanding the target definition to encompass service cancellations, the proposed methodology successfully elevated simultaneous (within-month) predictive capabilities relative to previous baselines, establishing an average balanced accuracy of 0.650 [22].

However, non-simultaneous generalization remains inferior to contemporary deep learning approaches. While the tabular XG-Boost models achieved modest non-simultaneous balanced accuracies, they remain inferior to the robust thresholds (≈ 0.75) demonstrated by hybrid Graph Attention Networks (GAT-LSTM/GRU) [40]. Because these hybrid architectures natively model sequential spatiotemporal flows, the performance gap suggests that flattening dynamic railway operations into static monthly snapshots omits the critical temporal sequencing required for high-fidelity prediction.

The Diagnostic Value ofInterpretability. While contemporary hybrid deep learning architectures exhibit robust absolute predictive accuracy thresholds for spatiotemporal forecasting, they frequently obscure the underlying mechanics of temporal degradation. Consequently, the primary academic contribution of this manuscript is not the maximization of raw predictive density, but the provision of an explainable temporal robustness analysis of network-wide railway delay prediction. As the Dutch railway network functions as critical public infrastructure, automated decision-making systems within this domain are subject to imminent regulatory scrutiny, necessitating high diagnostic transparency.

By prioritizing interpretable tree-based ensembles and SHAP analysis over opaque hybrid architectures, this methodology establishes a rigorous diagnostic lens for characterizing model degradation. Rather than identifying a singular algorithmic limitation, this transparent framework provides empirical evidence that systemic temporal drift is a multifaceted challenge. Ultimately, this approach demonstrates that feature-rich tree-based models improve withinmonth prediction, but their performance degrades across future horizons. Crucially, this framework explicitly links this degradation to environmental feature volatility and instability in the target definition.

## 5.2 Diagnosing Temporal Degradation

The empirical findings suggest that the temporal generalization failure observed during non-simultaneous testing is not an artifact of hyperparameter tuning limitation, but rather a dual-faceted data vulnerability. Specifically, the predictive degradation appears to be systematically linked to environmental feature volatility and instability in the target definition.

Notably, time-based evaluations revealed that isolated topological configurations completely resisted temporal decay across the primary tree-based ensemble models. Non-simultaneous evaluations marginally outperformed simultaneous baselines, confirming a complete absence of degradation for XGBoost $( W = 3 8 4 . 0 , p =$ $1 . 0 0 0 , r = 0 . 4 0 1 )$ , Random Forest $( W = 3 7 2 . 0 , p = 1 . 0 0 0 , r = 0 . 4 0 8 )$ and LightGBM $( W \ = \ 5 5 7 . 0 , \ p \ = \ 1 . 0 0 0 , r \ = \ 0 . 3 0 5 )$ (Appendix D Table 9). This indicates that the foundational spatial network structure possesses inherent temporal robustness. However, highly integrated feature sets and alternative threshold configurations overwhelmingly exhibited statistically significant temporal degradation $( p < . 0 0 1 )$ ) or yielded mixed robustness (Appendix D Tables 8 9 10 11 13). Incremental ablation testing explicitly isolates the source of this volatility. Supplementing the topological baseline with edge weights and operational metrics does not merely resist degradation; it yields statistically significant performance improvements $( p < . 0 0 1 )$ . In contrast, the integration of environmental (weather) features actively compromises the architecture. The addition of meteorological data did not enhance predictive capacity $( p = 1 . 0 0 0 )$ , instead precipitating a measurable drop in median accuracy, which is associated with a statistically significant reduction in non-simultaneous balanced accuracy observed in the fully integrated models (Appendix D Table 12).

This performance decrease during non-simultaneous testing is strongly associated with temporal label shift. Prior studies utilized a static, global median threshold across multi-year datasets to classify significantly delayed trajectories [22]. However, exploratory analy sis reveals that actual class prevalence fluctuates between 10% and 90% monthly (Appendix J Figure 11). Applying a static percentile threshold to this volatile environment induces severe label shift, as the operational meaning of the positive class changes over time.

This target instability is explicitly supported by the threshold sensitivity analysis (Appendix D Tables 7, 10, and 11). While utilizing the absolute threshold (a static 0.20 delay ratio) did not entirely eliminate temporal decay across the broader feature sets, it uniquely stabilized the core network representations, resulting in no statistically significant predictive change. In contrast, configurations reliant on volatile environmental variables continued to sufer statistically significant collapse $( p < 0 . 0 0 1 )$ . This divergence suggests that statistical target definitions artificially compound predictive decay by disproportionately penalizing otherwise stable structural topologies.

5.2.1 Covariate Shift and Concept Drift. Beyond target formulation, the observed predictive variance is further explained by the interplay between covariate shift and concept drift within the external variables. Temporal analysis revealed that the predictive reliance on all features is highly unstable (Appendix O Figure 28d and Appendix J Figure 16).

Conversely, the primary environmental feature, Source mean soil temperature, exhibited notable covariate shift; while the mean fluctuated seasonally, the within-month variance remained entirely collapsed. This spatial uniformity implies that while the input distribution changes over time, the localized variance required for distinct trajectory prediction is absent, causing the models to overfit to the specific monthly snapshot.

This instability is signaled by the shifting SHAP values across consecutive horizons. The substantial volatility in SHAP importance rankings (Section 4.2) indicates concept drift, where the functional relationship between predictive features and the target variable is inconsistent. As environmental features transition from dominant predictors in winter to noise in summer, the model’s predictive reliance undergoes a fundamental structural realignment.

To contextualize this realignment, the evaluation of explanation stability via the Kendall rank correlation (�) revealed clear statistical divergence across model configurations. While configurations exhibiting high temporal stability, notably OPS, indicate robust pre dictive mechanisms, the rank instability observed in configurations heavily reliant on external variables (such as WGT, WTH, and TW) required further diagnosis.

The application of the two-sample K-S test to the raw input features clarifies the source of this instability. As detailed in Appen dix I, meteorological features underwent statistically significant distributional shifts in nearly 85% of evaluated temporal transitions, whereas core operational and structural network features remained statistically stable in over 93% of transitions. By pairing the explanatory metrics with this distributional analysis, a distinct pattern emerges. Periods of significant SHAP rank inversion frequently align with these measured distributional shifts in the underlying feature space. Therefore, the explanatory instability recorded in environmentally enriched configurations reflects the model dynamically adapting to true concept drift within the operational environment. This methodological separation of explanationcentric variance from data-centric drift confirms that the algorithm accurately captures structural changes, ultimately demonstrating that fully integrated tabular models overfit to this seasonal meteorological noise.

Ultimately, the high dispersion in structural features alongside the collapsed variance in environmental features indicates that while topological architectures maintain underlying stability, fully integrated tabular models overfit to seasonal noise. This confirms that the integration of volatile environmental features actively precipitates the non-simultaneous generalization collapse.

## 5.3 Methodological Limitations and Future Work

Reliability of the Target Variable and Subclass Dynamics. The reliability of the target variable relies on the assumption that all recorded cancellations are ad-hoc operational failures. Exploratory data analysis indicates that pure delays and outright cancellations exhibit distinct volumetric trends over time (Appendix M Figure 20). While their combined impact on structural network integrity justifies a unified predictive target within this study, treating them as a monolithic class is a limitation necessitated by current data constraints. Specifically, the historical data lacks the "slowly changing dimensions" required to diferentiate ad-hoc cancellations from planned maintenance (e.g., explicit announcement timestamps). Future research must prioritize the integration of these slowly changing operational dimensions, allowing for the independent predictive modeling of these isolated sub-classes and refining the target variable exclusively to unexpected disruptions.

Internal Validity. A primary limitation relates to internal validity. Because the granular stop-to-stop topology and the expanded feature space were altered simultaneously from previous baselines [22], isolating the specific physical driver of the simultaneous performance increase is constrained. Future research must conduct systematic ablation studies, applying strictly topological features to the granular network without environmental data, to empirically isolate the predictive power inherent to the structural network alone.

Dimensionality and Feature Selection. While SHAP values were utilized for post-hoc interpretation, strict algorithmic feature selection methodologies were not implemented prior to model training. Because the complete engineered feature space was retained across all primary evaluations, potential noise from irrelevant variables negatively impacted cross-month generalization. Future research must apply formal feature selection techniques to reduce dimensionality prior to training, thereby mitigating the compounding variance associated with unstable external features.

Target Granularity and Temporal Horizons. To systematically address the temporal label shift identified in Section 5.2, future methodologies should replace global percentile thresholds with finer temporal aggregations (e.g., daily or hourly horizons) coupled with absolute operational boundaries (e.g., trajectories delayed > 3 minutes). Alternatively, predictive robustness could be enhanced by adopting dynamic, season-specific training paradigms where models are trained and evaluated strictly within matching temporal periods (e.g., training a model exclusively on historical January data to predict a future January) [10].

Scalability and Forecasting Constraints. The operational scalabil ity of environmental features presents a significant constraint. This study evaluated predictive power utilizing historically recorded meteorological data. However, deploying this architecture in a realtime operational setting necessitates forecasted weather values. Given the inherent inaccuracies of meteorological forecasting beyond a 14-day horizon [9, 29], the predictive robustness of environmental features is expected to degrade further in real-time operational planning.

Generalizability and Literature Comparability. The generalizability of these findings is strictly constrained by the domain characteristics of the dataset. Because the models were evaluated exclusively on domestic passenger services operated by NS, the specific feature dependencies identified herein may not transfer, as freight operations and international corridors operate under divergent scheduling priorities and infrastructural constraints. Furthermore, direct comparative analyses against existing literature are impeded by in herent heterogeneities in dataset compositions, rail operators, and temporal evaluation frames. Consequently, future research must establish standardized benchmarking frameworks across distinct European networks to rigorously evaluate cross-domain adaptability and algorithmic generalizability [6].

Ultimately, while these constraints limit immediate operational deployment, the diagnostic value of identifying threshold instability and feature volatility provides a vital theoretical bridge for developing the dynamic, season-aware architectures proposed in the subsequent conclusion.

## 5.4 Ethical and Societal Considerations

The deployment of long-term predictive models within public infrastructure introduces concerns regarding algorithmic bias and geographic fairness. A spatial bias may emerge if the predictive architecture performs more accurately in high-density urban nodes (e.g., the Randstad) compared to data-sparse rural regions. Allocating structural resources based on these geographically biased predictions risks creating a feedback loop of inferior infrastructural support, potentially reinforcing regional socioeconomic inequalities regarding housing afordability and job accessibility [21]. Therefore, an important direction for future work is conducting a comprehensive fairness audit to evaluate predictive performance disparities across high-density urban corridors versus lower-density rural routes prior to any real-time operational deployment.

## 6 Conclusion

This section synthesizes the empirical findings to establish the primary contribution of this research: an explainable temporal robustness analysis of network-wide railway delay prediction. While contemporary state-of-the-art architectures often obscure temporal degradation mechanics within opaque deep learning structures, this study presents a transparent diagnostic evaluation of predictive instability. The overarching finding establishes that feature-rich tree-based models improve within-month prediction, but their performance experiences a statistically significant reduction when evaluated on future months. Crucially, this degradation is explicitly linked to environmental feature volatility and instability in the statistical target definition.

By utilizing a comprehensive tabular dataset of domestic passenger trajectories, this diagnostic evaluation establishes four supporting operational realities. First, integrating granular external features successfully captures a meaningful predictive signal within localized monthly snapshots, achieving robust baseline balanced accuracies. Second, paired Wilcoxon statistical tests reveal a divided generalization capacity; isolated topological features demonstrate temporal resilience in tree-based models, whereas the broader integration of external features systematically degrades out-of-sample predictive performance. Third, targeted sensitivity analyses confirm that this overarching degradation is an inherent vulnerability of the data environment rather than a parametric tuning artifact. Finally, longitudinal SHAP evaluations confirm that this predictive decay is driven by target threshold instability operating concurrently with temporal volatility, specifically concept drift and covariate shift, within the environmental feature space.

Ultimately, this thesis demonstrates that richer feature sets alone are insuficient to resolve the constraints of long-term networkwide railway delay prediction. The primary barrier to predictive robustness is the highly unstable behavior of both the statistical target definition and the external environment across time. Therefore, achieving robust temporal generalization necessitates transitioning beyond the mere accumulation of tabular variables toward dynamic, season-aware modeling architectures anchored by absolute, operationally grounded target designs.

## References

[1] 2025. Open data. https://www.rijdendetreinen.nl/en/open-data

[2] 2026. About NS | NS. https://www.ns.nl/en/about-ns

[3] Ulrich Aïvodji, Hiromi Arai, Olivier Fortineau, Sébastien Gambs, Satoshi Hara, and Alain Tapp. 2019. Fairwashing: the risk of rationalization. arXiv:1901.09749 [cs.LG] https://arxiv.org/abs/1901.09749

[4] Solon Barocas and Andrew D. Selbst. 2016. Big Data’s Disparate Impact. California Law Review 104, 3 (2016), 671–732. https://www.cs.yale.edu/homes/jf/ BarocasSelbst.pdf

[5] Jasmijn Bastings and Katja Filippova. 2020. The elephant in the interpretability room: Why use attention as explanation when we have saliency methods? doi:10. 48550/arXiv.2010.05607 arXiv:2010.05607 [cs].

[6] Maarten Beltman, Marta Ribeiro, Jasper de Wilde, and Junzi Sun. 2025. Dynami cally forecasting airline departure delay probability distributions for individual flights using supervised learning. Journal ofAir Transport Management 126 (June 2025), 102788. doi:10.1016/j.jairtraman.2025.102788

[7] Adrien Bibal, Rémi Cardon, David Alfter, Rodrigo Wilkens, Xiaoou Wang, Thomas François, and Patrick Watrin. 2022. Is Attention Explanation? An Introduction to the Debate. 3889–3900. doi:10.18653/v1/2022.acl-long.269

[8] Md Emran Biswas, Tangina Sultana, Ashis Kumar Mandal, Md Golam Morshed, and Md Delowar Hossain. 2024. Spatio-Temporal Feature Engineering and Selection-Based Flight Arrival Delay Prediction Using Deep Feedforward Regression Network. Electronics 13, 24 (Dec. 2024). doi:10.3390/electronics13244910

[9] Cristian Bodnar, Wessel P. Bruinsma, Ana Lucic, Megan Stanley, Anna Allen, Johannes Brandstetter, Patrick Garvan, Maik Riechert, Jonathan A. Weyn, Haiyu Dong, Jayesh K. Gupta, Kit Thambiratnam, Alexander T. Archibald, Chun-Chieh Wu, Elizabeth Heider, Max Welling, Richard E. Turner, and Paris Perdikaris. 2025. A foundation model for the Earth system. Nature 641, 8065 (May 2025), 1180–1187. doi:10.1038/s41586-025-09005-y

[10] Brent Brakenhof, Ali Mohammed Mansoor Alsahag, and Seyed Sahand Moham madi Ziabari. 2026. Dynamic GNNs for Predicting Train Cancellations on the Dutch Railway Network: A Multi-Season Study of Environmental and Opera tional Factors. Digital Technologies Research and Applications (Jan. 2026), 32–52. doi:10.54963/dtra.v5i1.1709

[11] Zhihong Chang, Chunsheng Liu, and Jianmin Jia. 2023. STA-GCN: Spatial-Temporal Self-Attention Graph Convolutional Networks for Trafic-Flow Predic tion. Applied Sciences 13, 11 (June 2023). doi:10.3390/app13116796

[12] Jing Chen, Ali Mohammed Mansoor Alsahag, and Seyed Sahand Mohammadi Ziabari. 2025. An analytics framework for interpretable subseasonal forecasting under decadal climate variability. Decision Analytics Journal 17 (Dec. 2025), 100660. doi:10.1016/j.dajour.2025.100660

[13] European Commission. 2026. AI Act | Shaping Europe’s digital future. https: //digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai

[14] Janez Demšar. 2006. Statistical Comparisons of Classifiers over Multiple Data Sets. Journal ofMachine Learning Research 7, 1 (2006), 1–30. http://jmlr.org/ papers/v7/demsar06a.html

[15] European Parliament and Council of the European Union. 2021. Regulation (EU) 2021/782 of the European Parliament and of the Council of 29 April 2021 on rail passengers’ rights and obligations (recast). Oficial Journal of the European Union. http://data.europa.eu/eli/reg/2021/782/oj

[16] European Parliament and Council of the European Union. 2024. Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act). Oficial Journal of the European Union, L Series. https://eur-lex.europa.eu/ eli/reg/2024/1689/oj

[17] Daniel Fryer, Inga Strümke, and Hien Nguyen. 2021. Shapley Values for Feature Selection: The Good, the Bad, and the Axioms. IEEE Access 9 (2021), 144352– 144360. doi:10.1109/ACCESS.2021.3119110

[18] Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux. 2022. Why do tree-based models still outperform deep learning on tabular data? doi:10.48550/arXiv.2207. 08815 arXiv:2207.08815 [cs].

[19] Vikas Hassija, Vinay Chamola, Atmesh Mahapatra, Abhinandan Singal, Divyansh Goel, Kaizhu Huang, Simone Scardapane, Indro Spinelli, Mufti Mahmud, and Amir Hussain. 2024. Interpreting Black-Box Models: A Review on Explainable Artificial Intelligence. Cognitive Computation 16, 1 (Jan. 2024), 45–74. doi:10.1007/s12559- 023-10179-8

[20] Sarthak Jain and Byron C. Wallace. 2019. Attention is not Explanation. https: //arxiv.org/abs/1902.10186v3

[21] Kecen Jing and Wen-Chi Liao. 2023. Small things, big impact: The networkmediated spillover efect through a transport connectivity enhancement project. Regional Science and Urban Economics 101 (July 2023), 103897. doi:10.1016/j. regsciurbeco.2023.103897

[22] Merel Kampere and Ali Mohammed Mansoor Alsahag. 2025. Predicting Delayed Trajectories Using Network Features: A Study on the Dutch Railway Network. doi:10.48550/arXiv.2507.11776 arXiv:2507.11776 [cs] version: 1.

[23] KNMI. 2026. Koninklijk Nederlands Meteorologisch Instituut - Seizoenen. https: //www.knmi.nl/kennis-en-datacentrum/uitleg/seizoenen

[24] Ministerie van Binnenlandse Zaken en Koninkrijksrelaties. 2024. Wet personenvervoer 2000. https://wetten.overheid.nl/BWBR0011470/2024-01-01 Last Modified: 2026-02-18.

[25] Eva König. 2020. A review on railway delay management. Public Transport 12, 2 (June 2020), 335–361. doi:10.1007/s12469-020-00233-1

[26] Weihua Lei, Luiz G. A. Alves, and Luís A. Nunes Amaral. 2022. Forecasting the evolution of fast-changing transportation networks using machine learning. Nature Communications 13, 1 (July 2022), 4252. doi:10.1038/s41467-022-31911-2

[27] Ties Leneman, Ali Mohammed Mansoor Alsahag, and Seyed Sahand Moham madi Ziabari. 2026. Explainable AI for subseasonal forecasting of the north atlantic oscillation. Machine Learning for Computational Science and Engineering 2, 1 (Feb. 2026), 6. doi:10.1007/s44379-026-00055-1

[28] Jianmin Li and Xinyue Xu. 2026. A train delay prediction approach for High-Speed railway considering spatiotemporal logical relationships in train operation. Computers & Industrial Engineering 211 (Jan. 2026), 111601. doi:10.1016/j.cie.2025. 111601

[29] Edward N. Lorenz. 1963. Deterministic Nonperiodic Flow. Journal ofthe Atmospheric Sciences 20, 2 (March 1963), 130–141. doi:10.1175/1520-0469(1963) 020<0130:DNF>2.0.CO;2

[30] Scott Lundberg and Su-In Lee. 2017. A Unified Approach to Interpreting Model Predictions. doi:10.48550/arXiv.1705.07874 arXiv:1705.07874 [cs].

[31] Scott M. Lundberg, Gabriel Erion, Hugh Chen, Alex DeGrave, Jordan M. Prutkin, Bala Nair, Ronit Katz, Jonathan Himmelfarb, Nisha Bansal, and Su-In Lee. 2020. From Local Explanations to Global Understanding with Explainable AI for Trees.

Nature machine intelligence 2, 1 (Jan. 2020), 56–67. doi:10.1038/s42256-019-0138-9

[32] Behnam Nikparvar and Jean-Claude Thill. 2021. Machine Learning of Spatial Data. ISPRS International Journal of Geo-Information 10, 9 (Sept. 2021). doi:10. 3390/ijgi10090600

[33] NS. 2026. Nederlandse Spoorwegen Jaarverslag 2025. https://www.nsjaarverslag. nl/

[34] C. O’Neil. 2016. Weapons of Math Destruction: How Big Data Increases Inequality and Threatens Democracy. Crown. https://books.google.nl/books? id=NgEwCwAAQBAJ

[35] Mark Raasveldt and Hannes Mühleisen. 2019. DuckDB. In Proceedings of the 2019 International Conference on Management ofData. 1981–1984. doi:10.1145/ 3299869.3320212

[36] Fons Savelberg, Pim M.J. Warfemius, and Eric Kroes. 2017. ESTIMATION OF THE SOCIAL COSTS OF TRAIN DELAYS IN THE NETHERLANDS.

[37] Sofia Serrano and Noah A. Smith. 2019. Is Attention Interpretable? doi:10.48550/ arXiv.1906.03731 arXiv:1906.03731 [cs].

[38] SURF. 2025. Snellius: de Nationale Supercomputer | SURF.nl. https://www.surf. nl/diensten/rekenen/snellius-de-nationale-supercomputer

[39] Kah Yong Tiong, Zhenliang Ma, and Carl-William Palmqvist. 2023. A review of data-driven approaches to predict train delays. Transportation Research Part C: Emerging Technologies 148 (March 2023), 104027. doi:10.1016/j.trc.2023.104027

[40] Jonathan van der Maas. 2025. Predicting Train Delay in the Dutch Railway Network: A Hybrid Spatio-Temporal Approach. https://scripties.uba.uva.nl/ search?id=record\_56621

[41] Sebastian Wandelt, Xinyue Chen, and Xiaoqian Sun. 2025. Flight Delay Prediction: A Dissecting Review of Recent Studies Using Machine Learning. IEEE Transactions on Intelligent Transportation Systems 26, 4 (April 2025), 4283–4297. doi:10.1109/ TITS.2025.3528536

[42] Sarah Wiegrefe and Yuval Pinter. 2019. Attention is not not Explanation. https: //arxiv.org/abs/1908.04626v2

[43] Hein de Wilde, Ali Mohammed Mansoor Alsahag, and Pierre Blanchet. 2025. Time series classification of satellite data using LSTM networks: an approach for predicting leaf-fall to minimize railroad trafic disruption. doi:10.48550/arXiv. 2507.11702 arXiv:2507.11702 [cs].

[44] Xingtang Wu, Wenbo Lian, Min Zhou, Hairong Dong, and Fang Fang. 2025. Hybrid Approach to Train Delay Prediction: An Integration of Analytical Model and Deep Learning Techniques. IEEE Transactions on Industrial Electronics 72, 3 (March 2025), 3039–3047. doi:10.1109/TIE.2024.3440505

[45] Qingwei Zhong, Yingxue Yu, Yiru Huang, and Tianhang Zhang. 2025. Prediction and Optimization of Civil Aviation Flight Delays Based on Machine Learning Algorithms. International Journal of Computational Intelligence Systems 18, 1 (July 2025), 189. doi:10.1007/s44196-025-00932-2

[46] Patrick Zippenfenig. 2023. Open-Meteo.com Weather API. doi:10.5281/zenodo. 7970649

[47] Xinlu Zong, Jiawei Guo, Fucai Liu, and Fan Yu. 2025. TSTA-GCN: trend spatio temporal trafic flow prediction using adaptive graph convolution network. Scientific Reports 15, 1 (April 2025), 13449. doi:10.1038/s41598-025-96833-7

## Appendix A Generative AI Statement

In the course of writing this thesis, I made use of ChatGPT as a supplementary resource. This tool was primarily used to proofread text, improve the clarity and organization of my writing, and support my understanding of technical literature. In addition, I utilized the tool to identify and resolve coding bugs during the development process.

All content and analysis in this thesis are my own. Any AI-generated suggestions were carefully reviewed, revised, and only incorporated where appropriate.

I used this tool responsibly and transparently, ensuring it served solely as a support mechanism and that all outputs were critically assessed before being integrated into my work.

## Appendix B Research Scope and Methodology

Table 2: Summary of Key Preprocessing Decisions and Methodological Origin
<table><tr><td>Preprocessing Step</td><td>Rationale / Justification</td><td>Methodological Origin</td></tr><tr><td>vices (Freight &amp; International)</td><td>Filtering non-domestic NS ser- Excludes distinct routing priorities and sched- Novel exclusion criteria uling dynamics to strictly isolate domestic pas- senger patterns.</td><td></td></tr><tr><td>tion (Graph structure)</td><td>Stop-level trajectory formula- Increases observation density and prevents spa- Inherited [10] tial sparsity by treating all intermediate stops as discrete edge trajectories.</td><td></td></tr><tr><td>Bidirectional spatial filtering</td><td>Prevents the generation of isolated, discon- Inherited [10] nected nodes and ensures topological validity within the graph structure.</td><td></td></tr><tr><td>Terminal stop imputation</td><td>Aligns missing terminal timestamps with sched- Novel contribution uled times, reflecting physical starting/ending realities.</td><td></td></tr><tr><td>lations</td><td>Handling of full/partial cancel- Reflects true operational impact. Includes all Methodological extension [10, types of cancellations in the definition for de- 22] layed trajectories.</td><td></td></tr><tr><td>weather duplicates</td><td>Retention of October DST Timestamps are duplicated but meteorological Novel justification values are distinct and valid; retaining them pre- vents data loss and maintains unbiased monthly aggregates.</td><td></td></tr><tr><td>mum 4-ride threshold</td><td>Monthly aggregation and mini- Reduces noise from ad-hoc routes to isolate Inherited [22] overarching, stable network-wide structural states.</td><td></td></tr></table>

Table 3: Summary of experimental configurations detailing split type, imbalance handling, and purpose.
<table><tr><td>Split Type</td><td>Imbalance Handling</td><td>Purpose</td></tr><tr><td>Time-Based (All Models)</td><td>Time-Based Split</td><td>Primary temporal generalizability evaluation. Ensures 70%/30% train data split volume for each month</td></tr><tr><td>Trajectory-Based (XGBoost)Random Under-Sampling</td><td></td><td>Secondary baseline for strict literature comparability and standardized SHAP evaluation. Ensures 70%/30% train data split of trajectories across the whole dataset</td></tr><tr><td>Trajectory-Based Null ModelRandom Under-Sampling</td><td></td><td>Establishes a theoretical performance floor (shuffled labels). If the Null Model performs similar to non-shuffled data, then the non-shuffled data is likely a result of overfitting.</td></tr></table>

Appendix C Supplementary Model Performance Metrics (Global Aggregation) Feature sets abbreviations are TOP = topology, WGT = Weight, WTH = Weather, OPS = Operational, TW = TOP + WGT, TWCW = TOP + WGT + WTH, TWCO = TOP + WGT + OPS, ALLF = TOP + WGT + WTH + OPS

Table 4: Time-Based Split Results: F1 Score across Simultaneous and Non-Simultaneous Testing  
(a) Simultaneous Testing Results
<table><tr><td>Classifier</td><td>TOP</td><td>WGT</td><td>WTH</td><td>OPS</td><td>TW</td><td>TWCW</td><td>TWCO</td><td>ALLF</td></tr><tr><td>LGBMClassifier</td><td>0.640</td><td>0.680</td><td>0.637</td><td>0.641</td><td>0.687</td><td>0.699</td><td>0.694</td><td>0.707</td></tr><tr><td>LogisticRegression</td><td>0.548</td><td>0.580</td><td>0.577</td><td>0.537</td><td>0.585</td><td>0.622</td><td>0.608</td><td>0.634</td></tr><tr><td>RandomForest</td><td>0.675</td><td>0.696</td><td>0.664</td><td>0.675</td><td>0.708</td><td>0.714</td><td>0.717</td><td>0.722</td></tr><tr><td>XGBClassifier</td><td>0.645</td><td>0.673</td><td>0.637</td><td>0.649</td><td>0.682</td><td>0.695</td><td>0.693</td><td>0.702</td></tr></table>

(b) Non-Simultaneous Testing Results
<table><tr><td>Classifier</td><td>TOP</td><td>WGT</td><td>WTH</td><td>OPS</td><td>TW</td><td>TWCW</td><td>TWCO</td><td>ALLF</td></tr><tr><td>LGBMClassifier</td><td>0.596</td><td>0.579</td><td>0.452</td><td>0.550</td><td>0.592</td><td>0.570</td><td>0.596</td><td>0.565</td></tr><tr><td>LogisticRegression</td><td>0.504</td><td>0.517</td><td>0.499</td><td>0.507</td><td>0.525</td><td>0.518</td><td>0.549</td><td>0.515</td></tr><tr><td>RandomForest</td><td>0.593</td><td>0.573</td><td>0.484</td><td>0.553</td><td>0.586</td><td>0.545</td><td>0.590</td><td>0.542</td></tr><tr><td>XGBClassifier</td><td>0.595</td><td>0.573</td><td>0.443</td><td>0.544</td><td>0.588</td><td>0.563</td><td>0.595</td><td>0.561</td></tr></table>

Table 5: Time-Based Split Results: ROC AUC across Simultaneous and Non-Simultaneous Testing

(a) Simultaneous Testing Results
<table><tr><td>Classifier</td><td>TOP</td><td>WGT</td><td>WTH</td><td>OPS</td><td>TW</td><td>TWCW</td><td>TWCO</td><td>ALLF</td></tr><tr><td>LGBMClassifier</td><td>0.700</td><td>0.742</td><td>0.697</td><td>0.702</td><td>0.750</td><td>0.767</td><td>0.762</td><td>0.777</td></tr><tr><td>LogisticRegression</td><td>0.608</td><td>0.653</td><td>0.599</td><td>0.638</td><td>0.647</td><td>0.673</td><td>0.679</td><td>0.695</td></tr><tr><td>RandomForest</td><td>0.730</td><td>0.765</td><td>0.726</td><td>0.738</td><td>0.770</td><td>0.781</td><td>0.783</td><td>0.793</td></tr><tr><td>XGBClassifier</td><td>0.700</td><td>0.741</td><td>0.695</td><td>0.710</td><td>0.742</td><td>0.759</td><td>0.755</td><td>0.770</td></tr></table>

(b) Non-Simultaneous Testing Results
<table><tr><td>Classifier</td><td>TOP</td><td>WGT</td><td>WTH</td><td>OPS</td><td>TW</td><td>TWCW</td><td>TWCO</td><td>ALLF</td></tr><tr><td>LGBMClassifier</td><td>0.619</td><td>0.608</td><td>0.494</td><td>0.576</td><td>0.620</td><td>0.575</td><td>0.627</td><td>0.585</td></tr><tr><td>LogisticRegression</td><td>0.561</td><td>0.579</td><td>0.474</td><td>0.565</td><td>0.577</td><td>0.483</td><td>0.601</td><td>0.489</td></tr><tr><td>RandomForest</td><td>0.613</td><td>0.603</td><td>0.500</td><td>0.573</td><td>0.615</td><td>0.551</td><td>0.620</td><td>0.561</td></tr><tr><td>XGBClassifier</td><td>0.620</td><td>0.607</td><td>0.498</td><td>0.573</td><td>0.620</td><td>0.573</td><td>0.627</td><td>0.585</td></tr></table>

Table 6: Sensitivity Analysis: XGBoost Balanced Accuracy across Alternative Target Thresholds. Note: The average null baseline across all tested configurations remained mathematically stable at approximately 0.500 and is omited for clarity.
<table><tr><td rowspan="2">Feature Set</td><td colspan="2">Absolute Threshold (0.20)</td><td colspan="2">Historical Seasonal Threshold</td></tr><tr><td>Simultaneous</td><td>Non-Simultaneous</td><td>Simultaneous</td><td>Non-Simultaneous</td></tr><tr><td>Topological Features</td><td>0.570</td><td>0.580</td><td>0.577</td><td>0.546</td></tr><tr><td>Weight Features</td><td>0.624</td><td>0.618</td><td>0.613</td><td>0.562</td></tr><tr><td>Weather Features</td><td>0.575</td><td>0.501</td><td>0.577</td><td>0.500</td></tr><tr><td>Operational Features</td><td>0.581</td><td>0.538</td><td>0.592</td><td>0.533</td></tr><tr><td>Topological + Weight</td><td>0.612</td><td>0.608</td><td>0.621</td><td>0.564</td></tr><tr><td>Topological + Weight + Weather</td><td>0.626</td><td>0.596</td><td>0.632</td><td>0.559</td></tr><tr><td>Topological + Weight + Operational</td><td>0.625</td><td>0.621</td><td>0.629</td><td>0.569</td></tr><tr><td>All Features</td><td>0.643</td><td>0.604</td><td>0.646</td><td>0.556</td></tr></table>

Figure 3: Sensitivity Analysis Boxplots: Distribution of XGBoost Balanced Accuracy across alternative target thresholds and temporal configurations. These distributions correspond to the mean values reported in Table 6.  
![](images/e9ae7d760a740d60493e0a64ed7176f1cb25356e217758bc2a39079222ca28f4.jpg)  
(a) Absolute Threshold: Simultaneous

![](images/5e444e3c2278713332ede537a1eea937c9eb08f73b0518c0bc6474c443a416b5.jpg)  
(b) Absolute Threshold: Non-Simultaneous

![](images/6bdee177a4b9be483cbb68c4c01dc9d272731b312669e6be3a683d35062dc7a5.jpg)  
(c) Historical Seasonal: Simultaneous

![](images/23bacaa8950760383be6475aeefe1331489dc5955dc9349cbf19000810ec1b84.jpg)  
(d) Historical Seasonal: Non-Simultaneous

## Appendix D Statistical Significance Tests (Temporal Aggregation)

Table 7: Wilcoxon Signed-Rank Test results comparing simultaneous XGBoost performance against the null baseline (Trajectorybased split, $\alpha = 0 . 0 5 \mathrm { _ { i } }$ , � = 72 per feature set).
<table><tr><td>Feature Set</td><td>Direction</td><td>Mdn (Sim)</td><td>Mdn (Null)</td><td>Mdn Diff</td><td>W</td><td>p-value</td><td>Effect (r)</td></tr><tr><td>Topological Features</td><td> $\mathrm { S i m } > \mathrm { N u l l }$ </td><td>0.582</td><td>0.501</td><td>+0.075</td><td>2599.0</td><td> $< 0 . 0 0 1$ </td><td>0.601</td></tr><tr><td>Weight Features</td><td> $\mathrm { S i m } > \mathrm { N u l l }$ </td><td>0.601</td><td>0.496</td><td>+0.107</td><td>2538.0</td><td> $< 0 . 0 0 1$ </td><td>0.602</td></tr><tr><td>Weather Features</td><td> $\mathrm { S i m } > \mathrm { N u l l }$ </td><td>0.571</td><td>0.490</td><td>+0.082</td><td>2582.0</td><td> $< 0 . 0 0 1$ </td><td>0.593</td></tr><tr><td>Operational Features</td><td> $\mathrm { S i m } > \mathrm { N u l l }$ </td><td>0.584</td><td>0.496</td><td>+0.090</td><td>2520.0</td><td> $< 0 . 0 0 1$ </td><td>0.564</td></tr><tr><td>Topological + Weight</td><td> $\mathrm { S i m } > \mathrm { N u l l }$ </td><td>0.605</td><td>0.513</td><td>+0.104</td><td>2593.5</td><td> $< 0 . 0 0 1$ </td><td>0.598</td></tr><tr><td>Topological + Weight + Weather</td><td> $\mathrm { S i m } > \mathrm { N u l l }$ </td><td>0.630</td><td>0.512</td><td>+0.106</td><td>2627.0</td><td> $< 0 . 0 0 1$ </td><td>0.614</td></tr><tr><td>Topological + Weight + Operational</td><td> $\mathrm { S i m } > \mathrm { N u l l }$ </td><td>0.629</td><td>0.499</td><td>+0.132</td><td>2619.0</td><td> $< 0 . 0 0 1$ </td><td>0.610</td></tr><tr><td>All Features</td><td> $\mathrm { S i m } > \mathrm { N u l l }$ </td><td>0.629</td><td>0.503</td><td>+0.127</td><td>2595.0</td><td> $< 0 . 0 0 1$ </td><td>0.599</td></tr></table>

Table 8: Wilcoxon Signed-Rank Test results comparing simultaneous and non-simultaneous XGBoost performance (Trajectorybased split, $\alpha = 0 . 0 5 ,$ , � = 71 per feature set).
<table><tr><td>Model</td><td>Direction</td><td>Mdn (Sim)</td><td>Mdn (Non-Sim)</td><td>Mdn Diff</td><td>W</td><td>p-value</td><td>Effect (r)</td></tr><tr><td>Topological Features</td><td>Sim &gt; Non-Sim</td><td>0.583</td><td>0.556</td><td>+0.033</td><td>2197.0</td><td>&lt; 0.001</td><td>0.442</td></tr><tr><td>Weight Features</td><td>Sim &gt; Non-Sim</td><td>0.601</td><td>0.569</td><td>+0.041</td><td>2305.0</td><td>&lt; 0.001</td><td>0.494</td></tr><tr><td>Weather Features</td><td>Sim &gt; Non-Sim</td><td>0.571</td><td>0.500</td><td>+0.069</td><td>2556.0</td><td>&lt; 0.001</td><td>0.615</td></tr><tr><td>Operational Features</td><td>Sim &gt; Non-Sim</td><td>0.584</td><td>0.540</td><td>+0.045</td><td>2394.0</td><td>&lt; 0.001</td><td>0.537</td></tr><tr><td>Topological + Weight</td><td>Sim &gt; Non-Sim</td><td>0.606</td><td>0.571</td><td>+0.037</td><td>2308.0</td><td>&lt; 0.001</td><td>0.495</td></tr><tr><td>Topological + Weight + Weather</td><td>Sim &gt; Non-Sim</td><td>0.630</td><td>0.553</td><td>+0.074</td><td>2539.0</td><td>&lt; 0.001</td><td>0.606</td></tr><tr><td>Topological + Weight + Operational</td><td>Sim &gt; Non-Sim</td><td>0.630</td><td>0.578</td><td>+0.043</td><td>2409.0</td><td>&lt; 0.001</td><td>0.544</td></tr><tr><td>All Features</td><td>Sim &gt; Non-Sim</td><td>0.629</td><td>0.561</td><td>+0.069</td><td>2529.0</td><td>&lt; 0.001</td><td>0.602</td></tr></table>

Table 9: Wilcoxon Signed-Rank Test results comparing simultaneous and non-simultaneous performance across all models (Time-based split, � = 0.05, N=66 per feature set.
<table><tr><td>Classifier</td><td>Feature Set</td><td>Direction</td><td>Mdn (Sim)</td><td>Mdn (Non-Sim)</td><td>Mdn Diff</td><td>W</td><td>p-value</td><td>Effect (r)</td></tr><tr><td>LGBM</td><td>Topological Features</td><td>Sim &lt; Non-Sim</td><td>0.576</td><td>0.609</td><td>-0.029</td><td>557.0</td><td>1.000</td><td>0.305</td></tr><tr><td></td><td>Weight Features</td><td>Sim &gt; Non-Sim</td><td>0.619</td><td>0.601</td><td>+0.019</td><td>1363.0</td><td>0.050</td><td>0.143</td></tr><tr><td></td><td>Weather Features</td><td>Sim &gt; Non-Sim</td><td>0.584</td><td>0.500</td><td>+0.079</td><td>2179.0</td><td>&lt; 0.001</td><td>0.597</td></tr><tr><td></td><td>Operational Features</td><td>Sim &gt; Non-Sim</td><td>0.592</td><td>0.574</td><td>+0.014</td><td>1435.0</td><td>0.018</td><td>0.183</td></tr><tr><td></td><td>Topological + Weight</td><td>Sim &gt; Non-Sim</td><td>0.616</td><td>0.616</td><td>+0.000</td><td>1058.0</td><td>0.619</td><td>0.026</td></tr><tr><td></td><td>Top + Weight + Weather</td><td>Sim &gt; Non-Sim</td><td>0.632</td><td>0.568</td><td>+0.062</td><td>1998.0</td><td>&lt; 0.001</td><td>0.496</td></tr><tr><td></td><td>Top + Weight + Operational</td><td>Sim &gt; Non-Sim</td><td>0.627</td><td>0.623</td><td>+0.008</td><td>1066.0</td><td>0.600</td><td>0.022</td></tr><tr><td></td><td>All Features</td><td>Sim &gt; Non-Sim</td><td>0.642</td><td>0.579</td><td>+0.062</td><td>1891.0</td><td>&lt; 0.001</td><td>0.437</td></tr><tr><td>Logistic Regression</td><td>Topological Features</td><td>Sim &gt; Non-Sim</td><td>0.551</td><td>0.544</td><td>+0.004</td><td>1311.0</td><td>0.060</td><td>0.136</td></tr><tr><td></td><td>Weight Features</td><td>Sim &gt; Non-Sim</td><td>0.582</td><td>0.558</td><td>+0.025</td><td>1919.0</td><td>&lt; 0.001</td><td>0.452</td></tr><tr><td></td><td>Weather Features</td><td>Sim &gt; Non-Sim</td><td>0.560</td><td>0.500</td><td>+0.059</td><td>2175.0</td><td>&lt; 0.001</td><td>0.595</td></tr><tr><td></td><td>Operational Features</td><td>Sim &gt; Non-Sim</td><td>0.588</td><td>0.574</td><td>+0.015</td><td>1738.0</td><td>&lt; 0.001</td><td>0.352</td></tr><tr><td></td><td>Topological + Weight</td><td>Sim &gt; Non-Sim</td><td>0.577</td><td>0.559</td><td>+0.019</td><td>1690.0</td><td>&lt; 0.001</td><td>0.325</td></tr><tr><td></td><td>Top + Weight + Weather</td><td>Sim &gt; Non-Sim</td><td>0.611</td><td>0.505</td><td>+0.106</td><td>2207.0</td><td>&lt; 0.001</td><td>0.612</td></tr><tr><td></td><td>Top + Weight + Operational</td><td>Sim &gt; Non-Sim</td><td>0.612</td><td>0.587</td><td>+0.017</td><td>1754.0</td><td>&lt; 0.001</td><td>0.361</td></tr><tr><td></td><td>All Features</td><td>Sim &gt; Non-Sim</td><td>0.618</td><td>0.511</td><td>+0.108</td><td>2211.0</td><td>&lt; 0.001</td><td>0.615</td></tr><tr><td>Random Forest</td><td>Topological Features</td><td>Sim &lt; Non-Sim</td><td>0.575</td><td>0.605</td><td>-0.033</td><td>372.0</td><td>1.000</td><td>0.408</td></tr><tr><td></td><td>Weight Features</td><td>Sim &gt; Non-Sim</td><td>0.598</td><td>0.595</td><td>+0.005</td><td>1040.0</td><td>0.662</td><td>0.036</td></tr><tr><td></td><td>Weather Features</td><td>Sim &gt; Non-Sim</td><td>0.567</td><td>0.500</td><td>+0.065</td><td>2193.0</td><td>&lt; 0.001</td><td>0.605</td></tr><tr><td></td><td>Operational Features</td><td>Sim &lt; Non-Sim</td><td>0.566</td><td>0.572</td><td>-0.000</td><td>1080.0</td><td>0.565</td><td>0.014</td></tr><tr><td></td><td>Topological + Weight</td><td>Sim &lt; Non-Sim</td><td>0.598</td><td>0.611</td><td>-0.011</td><td>804.0</td><td>0.973</td><td>0.168</td></tr><tr><td></td><td>Top + Weight + Weather</td><td>Sim &gt; Non-Sim</td><td>0.619</td><td>0.540</td><td>+0.078</td><td>2091.0</td><td>&lt; 0.001</td><td>0.548</td></tr><tr><td></td><td>Top + Weight + Operational</td><td>Sim &gt; Non-Sim</td><td>0.613</td><td>0.613</td><td>+0.001</td><td>977.0</td><td>0.794</td><td>0.071</td></tr><tr><td></td><td>All Features</td><td>Sim &gt; Non-Sim</td><td>0.623</td><td>0.548</td><td>+0.075</td><td>1982.0</td><td>&lt; 0.001</td><td>0.487</td></tr><tr><td>XGBoost</td><td>Topological Features</td><td>Sim &lt; Non-Sim</td><td>0.580</td><td>0.611</td><td>-0.033</td><td>384.0</td><td>1.000</td><td>0.401</td></tr><tr><td></td><td>Weight Features</td><td>Sim &gt; Non-Sim</td><td>0.616</td><td>0.596</td><td>+0.030</td><td>1286.0</td><td>0.124</td><td>0.100</td></tr><tr><td></td><td>Weather Features</td><td>Sim &gt; Non-Sim</td><td>0.571</td><td>0.499</td><td>+0.075</td><td>2192.0</td><td>&lt; 0.001</td><td>0.604</td></tr><tr><td></td><td>Operational Features</td><td>Sim &gt; Non-Sim</td><td>0.587</td><td>0.570</td><td>+0.003</td><td>1340.0</td><td>0.067</td><td>0.130</td></tr><tr><td></td><td>Topological + Weight</td><td>Sim &lt; Non-Sim</td><td>0.605</td><td>0.611</td><td>-0.002</td><td>942.0</td><td>0.852</td><td>0.091</td></tr><tr><td></td><td>Top + Weight + Weather</td><td>Sim &gt; Non-Sim</td><td>0.628</td><td>0.567</td><td>+0.064</td><td>2028.0</td><td>&lt; 0.001</td><td>0.513</td></tr><tr><td></td><td>Top + Weight + Operational</td><td>Sim &gt; Non-Sim</td><td>0.622</td><td>0.620</td><td>+0.004</td><td>1025.0</td><td>0.696</td><td>0.045</td></tr><tr><td></td><td>All Features</td><td>Sim &gt; Non-Sim</td><td>0.634</td><td>0.580</td><td>+0.057</td><td>1832.0</td><td>&lt; 0.001</td><td>0.404</td></tr></table>

Table 10: Wilcoxon Signed-Rank Test results comparing simultaneous and non-simultaneous XGBoost performance under the Absolute Threshold (Trajectory-based split, � = 0.05, � = 71 per feature set).

<table><tr><td>Model</td><td>Direction</td><td>Mdn (Sim)</td><td>Mdn (Non-Sim)</td><td>Mdn Diff</td><td>W</td><td>p-value</td><td>Effect (r)</td></tr><tr><td>Topological Features</td><td>Sim &lt; Non-Sim</td><td>0.574</td><td>0.570</td><td>-0.000</td><td>1212.0</td><td>0.647</td><td>0.032</td></tr><tr><td>Weight Features</td><td>Sim &gt; Non-Sim</td><td>0.620</td><td>0.607</td><td>+0.024</td><td>1876.0</td><td>&lt; 0.001</td><td>0.288</td></tr><tr><td>Weather Features</td><td>Sim &gt; Non-Sim</td><td>0.569</td><td>0.502</td><td>+0.068</td><td>2538.0</td><td>&lt; 0.001</td><td>0.606</td></tr><tr><td>Operational Features</td><td>Sim &gt; Non-Sim</td><td>0.581</td><td>0.543</td><td>+0.039</td><td>2318.0</td><td>&lt; 0.001</td><td>0.500</td></tr><tr><td>Topological + Weight</td><td>Sim &gt; Non-Sim</td><td>0.606</td><td>0.595</td><td>+0.010</td><td>1695.0</td><td>0.008</td><td>0.201</td></tr><tr><td>Topological + Weight + Weather</td><td>Sim &gt; Non-Sim</td><td>0.619</td><td>0.588</td><td>+0.032</td><td>2314.0</td><td>&lt; 0.001</td><td>0.498</td></tr><tr><td>Topological + Weight + Operational</td><td>Sim &gt; Non-Sim</td><td>0.618</td><td>0.614</td><td>+0.007</td><td>1614.0</td><td>0.027</td><td>0.162</td></tr><tr><td>All Features</td><td>Sim &gt; Non-Sim</td><td>0.636</td><td>0.595</td><td>+0.049</td><td>2408.0</td><td>&lt; 0.001</td><td>0.543</td></tr></table>

Table 11: Wilcoxon Signed-Rank Test results comparing simultaneous and non-simultaneous XGBoost performance under the Month-Relative Threshold (Trajectory-based split, � = 0.05, � = 71 per feature set).
<table><tr><td>Model</td><td>Direction</td><td>Mdn (Sim)</td><td>Mdn (Non-Sim)</td><td>Mdn Diff</td><td>W</td><td>p-value</td><td>Effect (r)</td></tr><tr><td>Topological Features</td><td>Sim &gt; Non-Sim</td><td>0.580</td><td>0.547</td><td>+0.034</td><td>2178.0</td><td> $< 0 . 0 0 1$ </td><td>0.433</td></tr><tr><td>Weight Features</td><td> $\mathrm { S i m } > \mathrm { N o n } { \cdot } \mathrm { S i m }$ </td><td>0.618</td><td>0.562</td><td>+0.053</td><td>2400.0</td><td> $< 0 . 0 0 1$ </td><td>0.539</td></tr><tr><td>Weather Features</td><td> $\mathrm { S i m } > \mathrm { N o n } { \cdot } \mathrm { S i m }$ </td><td>0.580</td><td>0.501</td><td>+0.080</td><td>2495.0</td><td> $< 0 . 0 0 1$ </td><td>0.585</td></tr><tr><td>Operational Features</td><td> $\mathrm { S i m } > \mathrm { N o n } { \cdot } \mathrm { S i m }$ </td><td>0.582</td><td>0.532</td><td>+0.056</td><td>2505.0</td><td>&lt; 0.001</td><td>0.590</td></tr><tr><td>Topological + Weight</td><td> $\mathrm { S i m } > \mathrm { N o n } { \cdot } \mathrm { S i m }$ </td><td>0.619</td><td>0.565</td><td>+0.055</td><td>2406.0</td><td>&lt; 0.001</td><td>0.542</td></tr><tr><td>Topological + Weight + Weather</td><td> $\mathrm { S i m } > \mathrm { N o n } { \cdot } \mathrm { S i m }$ </td><td>0.628</td><td>0.556</td><td>+0.075</td><td>2489.0</td><td>&lt; 0.001</td><td>0.582</td></tr><tr><td>Topological + Weight + Operational</td><td> $\mathrm { S i m } > \mathrm { N o n } { \cdot } \mathrm { S i m }$ </td><td>0.629</td><td>0.565</td><td>+0.058</td><td>2454.0</td><td>&lt; 0.001</td><td>0.565</td></tr><tr><td>All Features</td><td> $\mathrm { S i m } > \mathrm { N o n } { \cdot } \mathrm { S i m }$ </td><td>0.639</td><td>0.553</td><td>+0.090</td><td>2555.0</td><td>&lt; 0.001</td><td>0.614</td></tr></table>

Table 12: Wilcoxon Signed-Rank Test results evaluating incremental feature engineering configurations (XGBoost Trajectory based split, Non-simultaneous testing, � = 0.05).
<table><tr><td>Step</td><td>Base Model</td><td>Enhanced Model</td><td>Direction</td><td>Mdn (Base)</td><td>Mdn (Enh)</td><td>Mdn Diff</td><td>W</td><td>p-value</td><td>Effect (r)</td></tr><tr><td>Adding Weights</td><td>Topological Features</td><td> $\mathrm { T o p o l o g i c a l + W e i g h t }$ </td><td> $\mathrm { E n h } > \mathrm { B a s e }$ </td><td>0.547</td><td>0.565</td><td>+0.018</td><td>368.0</td><td>&lt; 0.001</td><td>0.611</td></tr><tr><td>Adding Weather</td><td>Topological + Weight</td><td> $\mathrm { T o p + W e i g h t + W e a t h e r }$ </td><td> $\mathrm { E n h } < \mathrm { B a s e }$ </td><td>0.565</td><td>0.556</td><td>-0.006</td><td>1931.0</td><td>1.000</td><td>0.314</td></tr><tr><td>Adding Operational</td><td> $\mathrm { T o \bar { p } o l o g \bar { i } c a l + W e i g \bar { h } t }$ </td><td> $\mathrm { T o p + W e i g h t + O p }$ </td><td> $\mathrm { E n h } > \mathrm { B a s e }$ </td><td>0.565</td><td>0.565</td><td>+0.003</td><td>699.0</td><td> $< 0 . 0 0 1$ </td><td>0.278</td></tr><tr><td>Adding Op. to Weather</td><td> $\mathrm { T o p + W e i g h t + W e a t h e r }$ </td><td>All Features</td><td> $\mathrm { E n h } < \mathrm { B a s e }$ </td><td>0.556</td><td>0.553</td><td>-0.006</td><td>1831.0</td><td>0.999</td><td>0.266</td></tr><tr><td>Adding Wth. to Op.</td><td> $\mathrm { T o p + W e i g h t + O p }$ </td><td>All Features</td><td> $\mathrm { E n h } < \mathrm { B a s e }$ </td><td>0.565</td><td>0.553</td><td>-0.014</td><td>2528.0</td><td>1.000</td><td>0.601</td></tr><tr><td>Base vs. Best</td><td>Topological Features</td><td> $\mathrm { T o p + W e i g h t + O p }$ </td><td> $\mathrm { E n h } > \mathrm { B a s e }$ </td><td>0.547</td><td>0.565</td><td>+0.023</td><td>21.0</td><td>&lt; 0.001</td><td>0.604</td></tr></table>

Table 13: Wilcoxon Signed-Rank Test results comparing simultaneous and non-simultaneous tuned XGBoost performance (Trajectory-based split, $\alpha = 0 . 0 5 , N = 7 1 )$
<table><tr><td>Model</td><td>Direction</td><td>Mdn (Sim)</td><td>Mdn (Non-Sim)</td><td>Mdn Diff</td><td>W</td><td>p-value</td><td>Effect (r)</td></tr><tr><td>All Features</td><td> $\mathrm { S i m } > \mathrm { N o n } { \cdot } \mathrm { S i m }$ </td><td>0.650</td><td>0.572</td><td>+0.071</td><td>2555.0</td><td> $< 0 . 0 0 1$ </td><td>0.614</td></tr></table>

## Appendix E Regression Metrics (Regulatory Punctuality & Schedule Adherence)

Table 14: Regression Performance: Regulatory Punctuality Threshold (3-minute delay margin). Performance metrics are evaluated via Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) across both testing paradigms.
<table><tr><td>Feature Set</td><td>MAE</td><td>RMSE</td><td>MAE (Null)</td><td>RMSE (Null)</td></tr><tr><td colspan="5">(a) Simultaneous Testing</td></tr><tr><td>Topological Features</td><td>0.0758</td><td>0.1326</td><td>0.0961</td><td>0.1574</td></tr><tr><td>Weight Features</td><td>0.0600</td><td>0.1196</td><td>0.0942</td><td>0.1542</td></tr><tr><td>Weather Features</td><td>0.0771</td><td>0.1428</td><td>0.0978</td><td>0.1622</td></tr><tr><td>Operational Features</td><td>0.0553</td><td>0.1145</td><td>0.0936</td><td>0.1548</td></tr><tr><td>Topological + Weight</td><td>0.0585</td><td>0.1141</td><td>0.0934</td><td>0.1529</td></tr><tr><td>Topological + Weight + Weather</td><td>0.0542</td><td>0.1090</td><td>0.0935</td><td>0.1550</td></tr><tr><td>Topological + Weight + Operational</td><td>0.0532</td><td>0.1073</td><td>0.0939</td><td>0.1532</td></tr><tr><td>All Features</td><td>0.0498</td><td>0.1031</td><td>0.0930</td><td>0.1548</td></tr><tr><td colspan="5">(b) Non-Simultaneous Testing</td></tr><tr><td>Topological Features</td><td>0.1016</td><td>0.1655</td><td>0.1093</td><td>0.1721</td></tr><tr><td>Weight Features Weather Features</td><td>0.0970</td><td>0.1602</td><td>0.1077</td><td>0.1706</td></tr><tr><td></td><td>0.1179</td><td>0.1739</td><td>0.1139</td><td>0.1670</td></tr><tr><td>Operational Features</td><td>0.0982</td><td>0.1633</td><td>0.1081</td><td>0.1707</td></tr><tr><td>Topological + Weight</td><td>0.0961</td><td>0.1584</td><td>0.1080</td><td>0.1703</td></tr><tr><td>Topological + Weight + Weather</td><td>0.1032</td><td>0.1621</td><td>0.1143</td><td>0.1716</td></tr><tr><td>Topological + Weight + Operational</td><td>0.0944</td><td>0.1571</td><td>0.1062</td><td>0.1681</td></tr><tr><td>All Features</td><td>0.0974</td><td>0.1582</td><td>0.1097</td><td>0.1677</td></tr></table>

Table 15: Regression Performance: Schedule Adherence Threshold (0-minute delay margin). Performance metrics are evaluated via Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) across both testing paradigms.
<table><tr><td>Feature Set</td><td>MAE</td><td>RMSE</td><td>MAE (Null)</td><td>RMSE (Null)</td></tr><tr><td colspan="5">(a) Simultaneous Testing</td></tr><tr><td rowspan="8">Topological Features Weight Features Weather Features Operational Features Topological + Weight Topological + Weight + Weather Topological + Weight + Operational All Features</td><td>0.1366</td><td>0.1864</td><td>0.1645</td><td>0.2200</td></tr><tr><td>0.1228</td><td>0.1758</td><td>0.1617</td><td>0.2169</td></tr><tr><td>0.1486</td><td>0.2028</td><td>0.1673</td><td>0.2241</td></tr><tr><td></td><td>0.1290 0.1827</td><td>0.1588</td><td>0.2123</td></tr><tr><td>0.1209</td><td>0.1721</td><td>0.1620</td><td>0.2185</td></tr><tr><td>0.1196</td><td>0.1711</td><td>0.1622</td><td>0.2182</td></tr><tr><td>0.1191</td><td>0.1701</td><td>0.1618</td><td>0.2169</td></tr><tr><td colspan="4">0.1172 0.1677 0.1620</td></tr><tr><td colspan="14">(b) Non-Simultaneous Testing 0.2243 0.2226 0.2376</td></tr><tr><td rowspan="17">Topological Features Weight Features Weather Features Operational Features</td><td colspan="4">0.1683 0.1631</td><td>0.1800 0.1790</td><td>0.2366</td></tr><tr><td colspan="4">0.1831</td><td></td><td>0.2353 0.2311</td></tr><tr><td colspan="4"></td><td>0.1798</td><td>0.2350</td></tr><tr><td colspan="4">0.1723</td><td>0.1785</td><td></td></tr><tr><td colspan="4">0.1639</td><td>0.1795</td><td>0.2361</td></tr><tr><td colspan="4">Topological + Weight + Weather 0.1687 0.1631</td><td>0.1794</td><td>0.2337</td></tr><tr><td colspan="4">Topological + Weight + Operational</td><td colspan="2">0.1785</td><td>0.2347</td></tr><tr><td colspan="4">All Features 0.1658</td><td colspan="2">0.2225 0.1804</td><td>0.2347</td></tr></table>

## Appendix F Temporal Baselines (Performance & Surviorship Bias)

Table 16: Temporal Baselines: Predictive Performance Summary. Evaluated over � = 60 truncated months to ensure an equal comparison window between Naive (� − 1) and Seasonal Naive (� − 12) models.
<table><tr><td>Baseline Model</td><td>Statistic</td><td>Balanced Accuracy</td><td>F1 Score</td><td>ROC AUC</td></tr><tr><td rowspan="4">Naive (t − 1)</td><td>Mean</td><td>0.8080</td><td>0.7798</td><td>0.8080</td></tr><tr><td>Median</td><td>0.8246</td><td>0.8147</td><td>0.8246</td></tr><tr><td>Q1</td><td>0.7807</td><td>0.7172</td><td>0.7807</td></tr><tr><td>Q3</td><td>0.8459</td><td>0.8703</td><td>0.8459</td></tr><tr><td rowspan="4">Seasonal Naive (t – 12)</td><td>Mean</td><td>0.7190</td><td>0.6531</td><td>0.7190</td></tr><tr><td>Median</td><td>0.7257</td><td>0.6612</td><td>0.7257</td></tr><tr><td>Q1</td><td>0.6739</td><td>0.5352</td><td>0.6739</td></tr><tr><td>Q3</td><td>0.7778</td><td>0.7868</td><td>0.7778</td></tr></table>

Table 17: Temporal Baselines: Survivorship-Bias and Imputation Summary (� = 60). This table details the percentage of predictions relying on zero imputation (Default Rate) due to missing historical trajectories, alongside the resulting Delay Rate Gap and Non-Defaulted Accuracy.
<table><tr><td>Baseline Model</td><td>Metric</td><td>Default Rate</td><td>Default Delay Rate</td><td>Non-Def. Delay Rate</td><td>Delay Rate Gap</td><td>Non-Def. Bal. Acc.</td></tr><tr><td rowspan="2">Naive</td><td>Mean</td><td>0.0356</td><td>0.6983</td><td>0.4953</td><td>0.2030</td><td>0.8274</td></tr><tr><td>Median</td><td>0.0329</td><td>0.7208</td><td>0.5355</td><td>0.2155</td><td>0.8453</td></tr><tr><td rowspan="2">Seasonal Naive</td><td>Mean</td><td>0.0525</td><td>0.6215</td><td>0.4962</td><td>0.1254</td><td>0.7377</td></tr><tr><td>Median</td><td>0.0466</td><td>0.6250</td><td>0.5396</td><td>0.1760</td><td>0.7396</td></tr></table>

![](images/c51fb60df5e9249cea10e6002d0a9009675bfd93e638ad59c32f913c3403f5b6.jpg)  
Figure 4: Temporal Baselines: Balanced Accuracy by Calendar Month. Illustrates the aggregated seasonal variance and interquartile range for the Naive and Seasonal Naive baselines.

Temporal Baselines: Balanced Accuracy Over Time  
![](images/e4b3ca6726fc13b626712a3d83933711b87be9d96a3e801c933062e141e35ce7.jpg)  
Figure 5: Temporal Baselines: Balanced Accuracy Over Time. Displays the chronological predictive stability of the persistence baselines over the 60-month evaluation period.

![](images/47c3ad2b838d5c86a9a84e7b815840e5afb2f3b47c6a509afb2ce991872355c4.jpg)  
Defaulted Minus Non-Defaulted Delay Rate

![](images/116074f89f254ee367c2191320c3e6ea006581e0132a5fb83f5700eef2b8d421.jpg)  
Figure 6: Survivorship-Bias Assessment. The top panel shows the Fallback Usage (default rate) over time due to missing historical trajectories. The bottom panel displays the Delay-Rate Gap between defaulted and non-defaulted predictions.

## Appendix G Kendall Rank Correlation

Table 18: Kendall rank correlation (�) results evaluating sequential SHAP feature importance stability for the XGBoost base model (Trajectory-based split, $\alpha = 0 . 0 5 , T = 7 1$ sequential transitions).
<table><tr><td>Feature Set</td><td>N (Features)</td><td>Mean τ</td><td>Min τ</td><td>Max τ</td><td>Significant Transitions  $( p < 0 . 0 5 )$ </td></tr><tr><td>Topological Features</td><td>13</td><td>0.540</td><td>0.128</td><td>0.769</td><td>59 / 71</td></tr><tr><td>Weight Features</td><td>7</td><td>0.421</td><td>-0.048</td><td>0.905</td><td>12 / 71</td></tr><tr><td>Weather Features</td><td>14</td><td>0.449</td><td>-0.209</td><td>0.846</td><td>48 / 71</td></tr><tr><td>Operational Features</td><td>9</td><td>0.831</td><td>0.556</td><td>1.000</td><td>71 / 71</td></tr><tr><td>Topological + Weight</td><td>20</td><td>0.479</td><td>-0.158</td><td>0.705</td><td>63 / 71</td></tr><tr><td>Topological + Weight + Weather</td><td>34</td><td>0.367</td><td>-0.005</td><td>0.647</td><td>58 / 71</td></tr><tr><td>Topological + Weight + Operational</td><td>29</td><td>0.594</td><td>0.281</td><td>0.749</td><td>71 / 71</td></tr><tr><td>All Features</td><td>43</td><td>0.481</td><td>0.207</td><td>0.681</td><td>70 / 71</td></tr></table>

![](images/6a65272d28eb958262f5349cf95be81e3e1280952ee729168a3dbd12ad45a513.jpg)  
(a) Topological Features

![](images/1a047855f00c689e2ad6655d46d2b964b0bc1c3ee3a3242c093c5b65bf18387b.jpg)  
(b) Weight Features

![](images/ee7067e262de5efec3d156a6a932f0d60a6997482fe8a2d22ae4344ad75003e3.jpg)  
(c) Weather Features

![](images/013bbfa05dc614d720f1e3ca089e152e02230dff7e1fae447f8743c20be4b540.jpg)  
(d) Operational Features

![](images/49969be85252ec28759a79ac0b9fb74ec7352841d45e40ff21a45352e995d509.jpg)  
(e) Topological + Weight

![](images/df3adc22460b6e2d53330f408e5ac69fd32270ab3dea08cef7fdf15df31c3738.jpg)  
(f) Topological + Weight + Weather

![](images/e1462e93d8f8c7807d74a9d92333496d61f71e8edaf678476ba57762d6ea62d8.jpg)  
(g) Topological + Weight + Operational

![](images/5b3d98fa69bcb71d8145473fb5a4e8eac17bb823290ba509318e7b8cbbc3e5c5.jpg)  
(h) All Features  
Figure 7: Temporal trajectory of Kendall rank correlation coeficients across sequential transitions for all XGBoost feature configurations.

Appendix H Non Simultaneous Performance Across Months for the SHAP evaluated XGBoost model (trajectory based split)

Non-Simultaneous Performance: Topological Features  
![](images/e1076d71c395d8ce7a3f56205dec443aa9ddd30b2d0584d2efe2dd39b893e4cb.jpg)  
(a) Topological Features

Non-Simultaneous Performance: Weight Features  
![](images/34f5d233ab9ddbfe7af704a88acd4a31a6fb8dd6cca273d526bd9886e4ea7ac8.jpg)  
(b) Weight Features

Non-Simultaneous Performance: Weather Features  
![](images/9c2729b4f6c3ff677fcc28e3509fb7e13cc18f671d482a56f470b8cc1c582fab.jpg)  
(c) Weather Features  
Non-Simultaneous Performance: Topological + Weight

Non-Simultaneous Performance: Operational Features  
![](images/da5f12161fcdd545f1b0736aabdf6daa7b6588f48cfb575447aefa5e573823e1.jpg)  
(d) Operational Features

![](images/dcad761befe3125b1c3b907680fe72c2ae44e3027a34278c1b518ad187bfedd8.jpg)  
(e) Topological + Weight  
Non-Simultaneous Performance: Topological + Weight + Operationa

Non-Simultaneous Performance: Topological + Weight + Weather  
![](images/ffe150424e4d56dfb716bd5c6d0f6a83764c2e3b02bc9e39e2e905b25908eb22.jpg)

![](images/53edd9eabfeb180c0ff7720e4794d6a5e73cbc7fac85954812b38fd79ee1e284.jpg)  
(g) Topological + Weight + Operational

(f) Topological + Weight + Weather  
Non-Simultaneous Performance: All Features  
![](images/a8e8919d0f795eabb1f7a3bf028f1375410369dc4b69a45001f14f0b53f895c3.jpg)  
(h) All Features  
Figure 8: Non-simultaneous balanced accuracy performance across months for all feature sets (XGBoost, trajectory0based split.

Table 19: Macro-Categorical Kolmogorov-Smirnov (K-S) Drift Summary. This table quantifies the frequency and magnitude (D-Statistic) of statistically significant distributional shifts across the primary feature categories.
<table><tr><td>Category</td><td>Drift Frequency (%)</td><td>Mean D-Stat</td><td>Max D-Stat</td></tr><tr><td>Operational (OPS)</td><td>3.29</td><td>0.028</td><td>0.994</td></tr><tr><td>Topological (TOP)</td><td>6.94</td><td>0.045</td><td>0.190</td></tr><tr><td>Weight (WGT)</td><td>1.41</td><td>0.043</td><td>1.000</td></tr><tr><td>Weather (WTH)</td><td>84.81</td><td>0.645</td><td>1.000</td></tr></table>

Table 20: Feature-Level Kolmogorov-Smirnov (K-S) Diagnostic Summary (Part 1: Topological and Weight Features). Details the significance rate of temporal drift across all 71 sequential transitions.
<table><tr><td>Feature Category Significance Rate (%)</td></tr><tr><td>Mean D-Stat Max D-Stat Topological Features (TOP)</td></tr><tr><td>src_degree TOP 4.23 0.045 0.079</td></tr><tr><td>tgt_degree TOP 1.41 0.044 0.073 src_avg_distance_in TOP 4.23 0.047 0.162</td></tr><tr><td></td></tr><tr><td>common_neighbors TOP 12.68 0.040 0.095</td></tr><tr><td>jaccard_coefficient TOP 12.68 0.045 0.100</td></tr><tr><td>preferential_attachment TOP 4.23 0.042 0.080</td></tr><tr><td>adamic_adar_index TOP 14.08 0.046 0.095</td></tr><tr><td>resource_allocation_index TOP 11.27 0.045 0.095</td></tr><tr><td>src_avg_distance_out TOP 4.23 0.044 0.168</td></tr><tr><td>src_avg_distance_total TOP 7.04 0.049 0.190</td></tr><tr><td>tgt_avg_distance_in TOP 4.23 0.045 0.154</td></tr><tr><td>tgt_avg_distance_out TOP 4.23 0.041 0.157</td></tr><tr><td>tgt_avg_distance_total TOP 5.63 0.046 0.182</td></tr><tr><td>Weight Features (WGT)</td></tr><tr><td>decayed_edge_weight WGT 1.41 0.044 0.994</td></tr><tr><td>src_w_deg_inbound WGT 1.41 0.045 0.968</td></tr><tr><td>src_w_deg_outbound WGT 1.41 0.042 1.000</td></tr><tr><td>src_w_deg_total WGT 1.41 0.044 1.000</td></tr><tr><td>tgt_w_deg_inbound WGT 1.41 0.044 1.000</td></tr><tr><td>tgt_w_deg_outbound WGT 1.41 0.041 1.000</td></tr><tr><td>tgt_w_deg_total WGT 1.41 0.041 1.000</td></tr></table>

Table 21: Feature-Level Kolmogorov-Smirnov (K-S) Diagnostic Summary (Part 2: Weather and Operational Features). Details the significance rate of temporal drift across all 71 sequential transitions.
<table><tr><td>Feature Category Significance Rate (%) Mean D-Stat Max D-Stat</td></tr><tr><td>Weather Features (WTH)</td></tr><tr><td>src_avg_temperature_2m WTH 100.00 0.878 1.000</td></tr><tr><td></td></tr><tr><td>tgt_avg_temperature_2m WTH 100.00 0.878 1.000 src_avg_soil_temperature WTH 100.00 0.913 1.000</td></tr><tr><td>tgt_avg_soil_temperature WTH 100.00 0.914 1.000</td></tr><tr><td>src_avg_rain WTH 100.00 0.784 1.000</td></tr><tr><td>tgt_avg_rain WTH 100.00 0.785 1.000</td></tr><tr><td>src_avg_snowfall WTH 56.34 0.364 1.000</td></tr><tr><td>tgt_avg_snowfall WTH 56.34 0.363 1.000</td></tr><tr><td>src_avg_snow_depth WTH 38.03 0.270 1.000</td></tr><tr><td>tgt_avg_snow_depth WTH 38.03 0.270 1.000</td></tr><tr><td>src_avg_wind_speed_10m WTH 100.00 0.587 1.000</td></tr><tr><td>tgt_avg_wind_speed_10m WTH 98.59 0.584 1.000</td></tr><tr><td>src_avg_wind_gusts_10m WTH 100.00 0.721 1.000</td></tr><tr><td>tgt_avg_wind_gusts_10m WTH 100.00 0.720 1.000</td></tr><tr><td>Operational Features (OPS)</td></tr><tr><td>distance OPS 0.00 0.013 0.042</td></tr><tr><td>last_active_volume OPS 1.41 0.044 0.994</td></tr><tr><td>months_since_last_service OPS 1.41 0.033 0.994</td></tr><tr><td>is_new_or_stale_route OPS 1.41 0.022 0.994</td></tr><tr><td>ratio_intercity OPS 2.82 0.038 0.090</td></tr><tr><td>ratio_intercity_direct OPS 0.00 0.006 0.027</td></tr><tr><td>ratio_international OPS 0.00 0.003 0.021</td></tr><tr><td>ratio_operational_exceptions OPS 11.27 0.043 0.124</td></tr><tr><td>ratio_sprinter OPS 11.27 0.046 0.106</td></tr></table>

![](images/1e4ac6d98043d3f974c3f29bf5d4a46fcf1632557db56707e33f25eed5f81693.jpg)

(a) Geographical representation of the Dutch railway network at 2024-04, highlighting active stations and scheduled trajectory connections stops stations.  
![](images/ed5a72eef298fed564bf6e2e3642a523283c4b1f5129f02a5caa5ef708a7f5ab.jpg)  
(b) Stations from the reference stations dataset with no historical train railway passenger services operated by NS (i.e. present in stations dataset but not in historical services trajectories dataset).  
Figure 9: Spatial overview of the Dutch railway network (April 2024) infrastructure and historical passenger service coverage.

![](images/e2221823051b710e309d9751b2b50e405c89a0269fca8518d0a4c5f6fe26dd1f.jpg)

Figure 10: Binary Class Distribution with Target Label ’Is Significantly Delayed  
![](images/d7618ca315fb2b9b1a5bfb1b8e71df0ffec02cec2d88c92386250be359b77317.jpg)

Figure 11: Monthly class balance illustrating the proportion of on-time versus significantly delayed edges over the observation period.  
![](images/1aa89df0dec372bd4aae4dac6a72069aa3c5bc43a2c98ccdecdad90014b98a46.jpg)  
Figure 12: Total volume of scheduled train services aggregated per month across the entire railway network.

![](images/06c2b04f1ece8db3cc41cb7f36bcb24a073aefa4f7cafa89dd5685d82fb6704c.jpg)

Figure 13: Mean proportion of delayed arrivals per edge over time, measured against the historical global median.  
![](images/bbb93d1c4cc530c2a57da7330c07a46f1703944c96ed08db6b01e1927a5c62d4.jpg)

Figure 14: Distribution of the binary target label across unique edges within the network infrastructure of NS data.  
![](images/ca051bb2f6d78f85fdd5bfff9124bb1c4b12e66ac1198eb5c712a41b3c0bb8f9.jpg)  
Figure 15: Distribution of the binary target label across unique edges within the network infrastructure utilizing only origindestination as trajectories of NS data (Kämper’s).

![](images/7f47fc59edc432a566fa602603026ed8c41b9eaf972bf7ddc5b9093672516264.jpg)  
(a) Decayed Edge Weight

![](images/6d30efaf032746d7fc5559067d9d187e298ef72f7b08f373a5819254747b2d72.jpg)  
(b) Target Weighted Outbound Degree

![](images/a21597718c1c708f84b9fe626df5764e384f35c8b2226da640a4bd13d6858572.jpg)  
(c) Source Average Soil Temperature  
Figure 16: Distribution over time for selected network and environmental features. Shaded regions denote the interquartile range (25th–75th percentiles). Note that variations in src\_avg\_soil\_temperature are present but extremely minor, causing the interquartile range to collapse and appear as a single line.

## Appendix K Model Performance Distributions (XGBoost Trajectory-based split)

![](images/a04ec15f66197eb43220f3c93ab700d2a703702e562ec5e2bbfff892bb62401b.jpg)  
(a) Box plot of balanced accuracy across all feature sets for Simultaneous testing (XGBoost, Trajectory-based split, Global Aggregation).

![](images/53ee113dcdb92031a641d2913c336456bdfb012142deb6600457f656a58dd42c.jpg)  
(b) Figure 3: Box plot of balanced accuracy across all feature sets for Non-Simultaneous testing (XGBoost, Trajectory-based split, Global Aggregation).  
Figure 17: Comparative analysis of balanced accuracy distributions across all evaluated feature sets under Non-Simultaneous and Simultaneous testing conditions.

Global Class Balance (Total N = 49795), Absolute Threshold (> 0.20) Not Significantly Delayed (Class 0): 16614

## Appendix L Alternative Delay Thresholds: Absolute and Relative

![](images/d53b04f1b3ebc28d0de59c7319f68bbd04a136d4ad48e4318fd08bbcd6bcf48e.jpg)  
(a) Global Class Balance Absolute Threshold

Class Balance Over Time (Absolute threshold 0.20)  
![](images/f78765412f998bb4d3abeca6f85894f35803186055c4ed05dacf0da7413c5899.jpg)  
(b) Class Balance Over Time Absolute Threshold

![](images/9a3bd6d3f530e76fb292229ea3b1b0e897fb77dd6c66a08fb31f28337e91465c.jpg)  
(c) XGBoost Balanced Accuracy Boxplots Simultaneous Testing Absolute Threshold

![](images/ed83ab72cde93b7f78964e9e74715a583cbd6208bb650d6a8e3a21891d863779.jpg)  
(d) XGBoost Balanced Accuracy Boxplots Non-Simultaneous Testing Absolute Threshold

Figure 18: Overview of class balance and XGBoost balanced accuracy performance utilizing the absolute delay threshold.

Global Class Balance (Total N = 49795), Month Relative Threshold

![](images/c26a41c9d33723a0355fcefdd8932d75e573b6a12fc122caf6328ecbd94b2f82.jpg)  
(a) Global Class Balance Month Relative Threshold

Class Balance Over Time (Month Relative Threshold)  
![](images/bd4c7a03d1dcc46e536b46451c37700f04467328c800d008b9f60676fad5ef1d.jpg)  
(b) Class Balance Over Time Month Relative Threshold

![](images/b20c65d3d0fbc1933f3707db3ec961829c4ba18371e30610dffcf7a54ad5c320.jpg)  
(c) XGBoost Balanced Accuracy Boxplots Simultaneous Testing Month Relative Threshold

![](images/049923c71686338422d4751dd859c947684ad28505f9be6a1f4d80614907c44b.jpg)  
(d) XGBoost Balanced Accuracy Boxplots Non-Simultaneous Testing Month Relative Threshold  
Figure 19: Overview of class balance and XGBoost balanced accuracy performance utilizing the month relative delay threshold.

## Appendix M Supplementary Operational Metrics

![](images/a824e157de36553d89c78b932dd16a0113b8efb440a0d97727ff6ddcfdec6a0d.jpg)  
Figure 20: Time-Series of Monthly Operational Metrics: Pure Delays and Cancellations Compared Against Total Scheduled Service Volume.

## Appendix N Feature Engineering and Hyperparameter Configurations

Table 22: Overview of engineered feature sets utilized for trajectory delay prediction and their corresponding measurement units. Prefixes [src/tgt] denote source or target station calculations.

<table><tr><td>Feature Name</td><td>Measured In</td></tr><tr><td>Topological Features (Unweighted) [src/tgt] degree</td><td>Count</td></tr><tr><td>[src/tgt] avg distance in [src/tgt] avg distance out [src/tgt] avg distance total common neighbors jaccard coefficient preferential attachment adamic adar index resource allocation index</td><td>Kilometers Kilometers Kilometers Count Ratio (0-1) Score Score Score</td></tr><tr><td>Weighted Topology Features decayed edge weight [src/tgt] weighted degree inbound</td><td>Weight</td></tr><tr><td>[src/tgt] weighted degree outbound [src/tgt] weighted degree total</td><td>Weight Weight Weight</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Environmental (Weather) Features</td><td></td></tr><tr><td></td><td></td></tr><tr><td>[src/tgt] mean temperature 2m</td><td></td></tr><tr><td></td><td>℃</td></tr><tr><td>[src/tgt] sum rain</td><td>Millimeters</td></tr><tr><td>[src/tgt] sum snowfall</td><td></td></tr><tr><td></td><td>Centimeters</td></tr><tr><td>[src/tgt] mean snow depth</td><td></td></tr><tr><td>[src/tgt] mean wind speed 10m</td><td>Meters</td></tr><tr><td>[src/tgt] max wind gusts 10m</td><td>km/h</td></tr><tr><td></td><td>km/h</td></tr><tr><td>[src/tgt] avg soil temperature</td><td></td></tr><tr><td></td><td>℃</td></tr><tr><td>Operational Features</td><td></td></tr><tr><td>distance</td><td>Kilometers</td></tr><tr><td>last active volume</td><td></td></tr><tr><td>months since last service</td><td>Count</td></tr><tr><td>is new or stale route</td><td>Month Counts</td></tr><tr><td></td><td>Boolean (0/1)</td></tr><tr><td>ratio intercity</td><td>Ratio (0-1)</td></tr><tr><td>ratio intercity_direct</td><td></td></tr><tr><td></td><td>Ratio (0-1)</td></tr><tr><td>ratio international</td><td>Ratio (0-1)</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>ratio sprinter</td><td>Ratio (0-1)</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>ratio operational exceptions</td><td>Ratio (0-1)</td></tr><tr><td></td><td></td></tr></table>

Table 23: Topological Feature Definitions and Mathematical Formulations. The monthly railway network is formally defined as a directed graph $G _ { t } = ( V _ { t } , E _ { t } )$ , where the nodes $V _ { t }$ represent individual train stations and the directed edges $E _ { t }$ denote active train trajectories. For any given node $v \in V _ { t } ,$ ���(�) and ����(�) denote the standard degree and weighted degree, respectively. The neighborhood set is defined as $N ( v ) ,$ , and the physical spatial distance between stations is denoted by $d ( u , v )$ . Within this framework, the physical distance between stations was utilized as the edge distance, while the decayed operational edge weight $( W _ { \mathbf { d e c a y e d } } )$ previously defined in Section 3.2.1 was formally assigned as the structural edge weight �(�, �) to map historical service density directly onto the network topology.
<table><tr><td>Feature</td><td>Type</td><td>Formula / Description</td></tr><tr><td>Node Degree</td><td>Node-level</td><td> $d e g ( v ) = | \{ u \in V : ( v , u ) \in E \} |$ </td></tr><tr><td>Out Weighted Degree</td><td>Node-level</td><td> $\begin{array} { r } { w d e g _ { o u t } ( v ) = \sum _ { u \in N ( v ) } w ( v , u ) } \end{array}$  , where w(v, u) is the edge weight.</td></tr><tr><td>In Weighted Degree</td><td>Node-level</td><td> $\begin{array} { r } { w d e g _ { i n } ( v ) = \sum _ { u \in N ( v ) } w ( u , v ) } \end{array}$  , where  $w ( u , v )$  is the edge weight.</td></tr><tr><td>Total Weighted Degree</td><td>Node-level</td><td> $w d e g _ { t o t a l } ( v ) = w d e g _ { o u t } ( v ) + w d e g _ { i n } ( v )$ </td></tr><tr><td>Out Average Distance</td><td>Node-level</td><td> $\begin{array} { r } { \frac { 1 } { | N ( v ) | } \sum _ { u \in N ( v ) } d ( v , u ) } \end{array}$ </td></tr><tr><td>In Average Distance</td><td>Node-level</td><td> $\begin{array} { r } { \frac { 1 } { | N ( v ) | } \sum _ { u \in N ( v ) } d ( u , v ) } \end{array}$ </td></tr><tr><td>Total Average Distance</td><td>Node-level</td><td>Šum of in and out average distances.</td></tr><tr><td>Common Neighbours</td><td>Trajectory-level</td><td> $\left| \Gamma ( u ) \cap \Gamma ( v ) \right|$ </td></tr><tr><td>Jaccard Coefficient</td><td>Trajectory-level</td><td>|Γ(u)∩Γ(v)| |Γ(u)∪Γ(v)|</td></tr><tr><td>Preferential Attachment</td><td>Trajectory-level</td><td> $| \Gamma ( u ) | \cdot | \Gamma ( v ) |$ </td></tr><tr><td>Adamic-Adar Index</td><td>Trajectory-level</td><td> $\sum _ { \substack { \mathbf { \Phi } \mathbf { w } \in \Gamma ( u ) \cap \Gamma ( v ) } } \frac { 1 } { \log | \Gamma ( w ) | }$ </td></tr><tr><td>Resource Allocation Index</td><td>Trajectory-level</td><td> $\begin{array} { r } { \sum _ { w \in \Gamma ( u ) \cap \Gamma ( v ) } \frac { 1 } { | \Gamma ( w ) | } } \end{array}$ </td></tr></table>

Table 24: Utilized Default Hyperparameter Configurations by Classifier
<table><tr><td>Classifier</td><td>Hyperparameters</td></tr><tr><td>LogisticRegression (scikit-learn 1.8.0)</td><td>penalty: {&#x27;l2&#x27;}, C: {1.0}, 11_ratio: {0.0}, fit_intercept: {True}, intercept_scaling: {1}, solver: {&#x27;lbfgs&#x27;}, max_iter: {1000}, tol: {0.0001}, dual: {False}, class_weight: {None}, random_state: {88}, warm_start: {False}, n_jobs: {None}</td></tr><tr><td>RandomForestClassifier (scikit-learn 1.8.0)</td><td>n_estimators: {100}, criterion: {&#x27;gini&#x27;}, max_depth: {None}, min_samples_split: {2}, min_samples_leaf: {1}, min_weight_fraction_leaf: {0.0}, max_features: {&#x27;sqrt&#x27;}, max_leaf_nodes: {None}, min_impurity_decrease: {0.0}, bootstrap: {True}, oob_score: {False}, n_jobs: {None}, random_state: {88}, class_weight: {None}, ccp_alpha: {0.0}, max_samples:</td></tr><tr><td>XGBClassifier (xgboost 3.2.0)</td><td>{None}, monotonic_cst: {None} objective: {&#x27;binary:logistic&#x27;}, n_estimators: {100}, learning_rate: {0.3}, max_depth: {6}, booster: {&#x27;gbtree&#x27;}, tree_method: {&#x27;auto&#x27;}, gamma: {0}, min_child_weight: {1}, max_delta_step: {0}, sub- sample: {1}, colsample_bytree: {1}, colsample_bylevel: {1}, colsample_bynode: {1}, reg_alpha:</td></tr><tr><td>LGBMClassifier (lightgbm 4.6.0)</td><td>{0}, reg_lambda: {1}, scale_pos_weight: {1}, base_score: {auto}, random_state: {88} objective: {&#x27;binary&#x27;}, boosting_type: {&#x27;gbdt&#x27;}, num_leaves: {31}, max_depth: {-1}, learn- ing_rate: {0.1}, n_estimators: {100}, subsample_for_bin: {200000}, scale_pos_weight: {1}, min_split_gain: {0.0}, min_child_weight: {0.001}, min_child_samples: {20}, subsample: {1.0}, subsample_freq: {0}, colsample_bytree: {1.0}, reg_alpha: {0.0}, reg_lambda: {0.0}, ran- dom_state: {88}</td></tr></table>

## Appendix O Year-over-Year (YoY) Temporal Feature Importance

![](images/8bf2583864c068da303ed30fc058051811d1ae314afdda5aaad392afebe5aabd.jpg)  
(a) Topology

![](images/4faebbfa380f75624543e2e5858d1f0dfd14263c31917f79c95ea50efd2a0a6f.jpg)  
(b) Weight

![](images/96efc1f1b4c23a8bffb68b3bb0d3146209988051558d76eddf866bb1041c1569.jpg)  
(c) Weather

![](images/b0eff77d689b02cae6abe6652655b95dddc7dabbe7b7c11841a2a24e7bb6466d.jpg)  
(d) Operational  
Figure 21: Monthly Year-over-Year temporal feature importance for the XGBoost model evaluating isolated baseline feature sets.

![](images/36812e71b1847656c502bde4b6079e7ebb05180043f908fa7b4c2fde263a6389.jpg)  
(a) Topology + Weight

![](images/57caaacb7be2b59df955fcec2a69c4f9b44638f2e98b63624c2689b1048917b4.jpg)  
(b) Topology + Weight + Weather  
Figure 22: Monthly Year-over-Year temporal feature importance for the XGBoost model evaluating initial combined topological + weight and weather feature sets.

(b) All Features  
(a) Topology + Weight + Operational  
![](images/846e7e57d7f459610172a92bf5f5c2da5c9333bf060e58d97b2507cf12d019ae.jpg)

![](images/bd76ce19a780c3ee5126749b172e082ae24fab2b5d808c0192ddc24fe70f348a.jpg)  
Figure 23: Monthly Year-over-Year temporal feature importance for the XGBoost model evaluating combined topological + weight + operational feature and all feature set.

## Appendix P Month-over-Month (MoM) Temporal Feature Importance

![](images/24dfa035ba703a936b6fcd44b234bc4d5d57abda8ab9f7c4dbc7e14ea43b9258.jpg)  
(a) Topology

![](images/ea3e56ac3120b39281d0e9e8d92493d2d636c88fa8dc6311133107f970c98c95.jpg)  
(b) Weight

![](images/6ca20f3c0000b01bfc1812101a16265fa82e96a5104abac4315b64feea8cc8b8.jpg)  
(c) Weather

![](images/4c4d52e7b0b470ee14ddf5ea4c2fe38754a96251dcd23e199989bd511e73192e.jpg)  
(d) Operational  
Figure 24: Aggregated (2019-2024) Month-over-Month SHAP feature importance for the XGBoost model evaluating isolated baseline feature sets.

![](images/d7e5fe4ccd342982f52e6320222ec9143f49daf413f6e23c4925977e8fe39445.jpg)  
(a) Topology + Weight

![](images/e1a14f2959af53e1faf1628ec3a479101ae44ba6550c82d117a06633ccc994da.jpg)  
(b) Topology + Weight + Weather  
Figure 25: Aggregated (2019-2024) Month-over-Month SHAP feature importance for the XGBoost model evaluating initial combined topological + weight and weather feature sets.

![](images/7e17c806c8a545122b6dae158092582f675e27a3341515ca3504afcdfadc9c8f.jpg)  
(a) Topology + Weight + Operational

![](images/a40430c2125ee8446cd7e02f5e5b28d10a37a4f90d6c137e38fbcae1cb3ed78a.jpg)  
(b) All Features  
Figure 26: Aggregated (2019-2024) Month-over-Month SHAP feature importance for the XGBoost model evaluating combined operational features and all features set.

## Appendix Q SHAP Value Feature Impact Distributions

![](images/974f7a5cc39040f2d2a1ef586b4957c1486b276edeaaa6a0e1b1b9dc2fa11059.jpg)  
(a) Topology

![](images/278aef9364ede87fef2839de71b8b26dc056109b32bf62f59ecc68be450248e9.jpg)  
(b) Weight

![](images/170f51585b75e89575bff97d5a6856deee007ca953fabc5508c1bb58c96148e4.jpg)  
(c) Weather

![](images/b0ae4d664ffa064219ee7a041844357369bce3b4a12a0baa54b77e88ebaf3c9a.jpg)  
(d) Operational  
Figure 27: SHAP summary plots illustrating feature impact magnitudes for the XGBoost model utilizing individual feature sets of period 2024-01.

![](images/4ab35e5c124f18d7e50f04ccc715e91fc9e2dc7efbccacb0d8b351731953f5c7.jpg)  
(a) Topology + Weight

![](images/4a17fbe5ecb69e80eb5282b8ad585705dc3cf30a1edd41180458bbeebc0de606.jpg)  
(b) Topological + Weight + Weather

![](images/64ccad419b143e77c7df109b0ec01fb1b887bafdee817e9cb967624cd09baa73.jpg)  
(c) Topological + Weight + Operational

![](images/afc3a404619cdd445b3b18da93db7252ad3380bdd3ac8bf525af415e2d4ac9b5.jpg)  
(d) All Features  
Figure 28: SHAP summary plots illustrating feature impact magnitudes for the XGBoost model utilizing combined feature sets of period 2024-01. Color indicates feature value (red = high, blue = low).

## Appendix R Confusion Matrices

![](images/5a568e7fa7cde6d70da8cdb5b594149f06cfe3403b2c5227cc84f7958d5116b9.jpg)  
(a) Simultaneous: Topological

![](images/09affff77b057dd6cc52ec789b6606316e14902a847d0ef1a5fefe68a517935c.jpg)  
(b) Non-Simultaneous: Topological

![](images/27a5922001ca158310305a600de018152b15102ca48f79c9b9fc187961d1dd4f.jpg)  
(c) Simultaneous: Weight

![](images/a7f1d0271b31a029dc4bd360df535a4e2cbdae82ecdba47a969576d9fc1b684e.jpg)  
(d) Non-Simultaneous: Weight

![](images/2ba5bf28cd87444077397a82147f4585879d5730b5f87eb92917d76b1768aa09.jpg)  
(e) Simultaneous: Weather

![](images/c68e67e4c2463d06e4bff3ba9876e335a968554a448e0f50ebe5d47510001a73.jpg)  
(f) Non-Simultaneous: Weather

![](images/2ae07b609c2a68470c47342e3be88ae9194c707c0db6ce9f6039813e88bef07b.jpg)

![](images/9a14c766c6396da4d9a221c55569d3f4b7b6b709018fbdbd1024446c8bcada13.jpg)

![](images/4f3b87d89d321d3f6c9eb1ab2b8bf47e89536a604f5edc3a710e1990956a612d.jpg)

(a) Simultaneous: Topology + Weight  
![](images/76036b7dfce445649c8dcc3a8400f76fc75c2b17966554149ea8e09eb5517301.jpg)  
(c) Simultaneous: Topology + Weight + Weather

(b) Non-Simultaneous: Topology + Weight  
![](images/d640dd8141db41534d2e51b3499d55718030db99f29cb475adbf8bc5e67da4e5.jpg)

![](images/a76a338c725583554e3ec447a29295bfe4cadd4f39068bd5966e0f94946e51da.jpg)  
(e) Simultaneous: Topology + Weight + Operational

(d) Non-Simultaneous: Topology + Weight + Weather  
![](images/c0aa1fe04b3ef18046ce1a6e64390c0b69c854bf2b4f514cac26cc6653fd3168.jpg)

![](images/08eb79368c281e3a3a6bfec09324399af07af0abae3081d0e54b2a5ad2fcdc86.jpg)

![](images/a4637f60daaa724a2b505fbeefff8c1e19c4eaf85712c2c26747e8641902d634.jpg)  
(g) Simultaneous: All Features

(f) Non-Simultaneous: Topology + Weight + Operational  
![](images/6361d73be9d749018016af1ecc055cb91773f6918ef4ac29378f30df25e8ef9f.jpg)  
(h) Non-Simultaneous: All Features

![](images/197289720834f24be3aa78e195ddfcbc83a5de0a48eac5ebc53102ccfb926ba3.jpg)  
(a) Simultaneous (Tuned Hyperparameters): All Features

![](images/0f02f27a46ed2a1363ab4cc6ec48b22e6634f89457e87ab979a1cfd41c4c4b71.jpg)  
(b) Non-Simultaneous (Tuned Hyperparameters): All Features  
Figure 31: Side-by-side confusion matrix comparisons for Tuned XGBoost models utilizing the complete feature space (Topological + Weight + Weather + Operational) of period 2024-01.