# What Does History Add? Calendar Controls and Information-Timing Audits for University Trajectory Forecasting

Zepei Cao East China Normal University

15 October 2026

## Abstract

University planning systems may benefit from prior observations, but a comparison with a current-state predictor can conflate historical information with additional calendar features. We examine this distinction in a reproducible, rule-based university trajectory simulator with 11 state coordinates, planned activities, resource constraints, and stochastic execution. The study reconstructs 16,750 routes from 3,350 simulated profiles, comprising 804,000 monthly records across training, validation, and two family-level evaluation sets. Seventeen regularized linear specifications and a persistence baseline separate current plans, past-state summaries, past-plan summaries, and history-length indicators. On the original 40-family evaluation, six-month history reduces six-step normalized RMSE from 0.159234 to 0.151522. However, its advantage over a history-length-only control is only −0.000895, with a paired family-bootstrap 95% interval of [−0.001957, 0.000248]. Frozen weights evaluated on 30 previously unused families show the same pattern: 0.156480 versus 0.150074 against the current-state baseline, but a length-controlled diference of −0.000558 with 95% interval [−0.001783, 0.000728]. An exploratory loss decomposition identifies improvements in preparation indices and funds, while fatigue accounts for 97.80% of the history model’s original six-step normalized squared error and has an uncertain length-controlled benefit. Moreover, 28.20% of its six-step fatigue predictions violate the simulator’s actual [5, 90] support. The results support an audited distinction between history summaries, clock information, and state-specific predictive value. They concern conditional simulator prediction given recorded future plans; they do not establish real-student forecasting, causal action efects, autonomous planning, or languagemodel memory performance.

## 1 Introduction

University planning spans several interacting processes: academic preparation, projects, career exploration, limited discretionary time, and available funds. A system that represents only the latest state may discard useful information about recent progress or repeated activity. Conversely, a system with extensive history can appear stronger because it receives additional indicators of elapsed time, because evaluation supplies later plans unavailable at the forecast origin, or because an aggregate score conceals diferences among outcomes. These possibilities must be distinguished before attributing an improvement to memory.

We study a deliberately limited question: what additional predictive value do explicitly defined summaries of earlier states and plans provide, beyond the current state, nominal context, and current recorded plan? A rule-based simulator makes the information order and targetgeneration mechanism inspectable. It also limits the substantive interpretation: its internal preparation and fatigue indices are designed quantities, not validated measurements of actual students. This setting is useful for examining evaluation logic and conditional prediction, while ofering no substitute for empirical educational validation.

The key control is inexpensive. A windowed-history model receives an available-history count and an indicator that history is nonempty. Both are deterministic functions of the calendar. Adding them to a linear model changes its calendar basis even when no earlier observation is supplied. We therefore compare each relevant history model with a length-only counterpart, alongside the conventional current-state baseline. The resulting distinction matters: a favorable comparison against the latter need not imply a favorable comparison against the former.

Our contribution is a task and evaluation audit rather than a new recurrent architecture. We formalize the simulator’s prediction-time contract, compare 17 matched Ridge specifications, preserve scenario-family dependence in evaluation, and inspect target-specific error and output support. An exposure audit found that 22 of the original 40 evaluation families had appeared in earlier local tasks. We retain those preselected results transparently and evaluate locked weights on a separately selected 30-family set that excludes all identified previous uses. The substantive finding is conditional: historical summaries improve several preparation and funds predictions, but the prespecified six-month summary’s six-step aggregate advantage over its clock-aware control remains uncertain in both evaluation sets. All-history summaries retain a clearer advantage over their own count-all control. This distinction is more informative than a general claim that longer memory improves university planning.

## 2 Related Work

Recurrent educational state estimation is established. Deep Knowledge Tracing models student coursework interactions to predict subsequent performance [1]. CSEAL combines LSTM knowledge tracing, prerequisite navigation, and actor–critic learning-path recommendation [2]. Its main learning-gain experiments use both a rule-based simulator and a DKT-based simulator, with explicit discussion of their limitations. Our domain includes several preparation dimensions, fatigue, and funds, but the combination of dynamic state, history, constraints, and educational recommendation is prior work. We do not claim a first education world model or a new theory of student development.

Learned dynamics can support action selection when evaluated in the environment in which actions are executed. PlaNet connects latent dynamics and multi-step prediction with online planning and control [4]. Sequential predictions and policy choices can also alter the distribution of subsequent observations, as analyzed in DAgger [5]; MOPO addresses distribution shift in model-based ofline policy optimization [6]. These works motivate separating conditional prediction from policy evaluation. Our supplied-plan experiment trains and evaluates state predictors, without learning an autonomous policy or optimizing an ofline return.

Architecture terminology also requires care. GRU’s reset and update gates originate in established recurrent-network work [3]; neither a recurrent hidden vector nor a history summary establishes human-like long-term memory. This study instead uses transparent linear predictors to isolate declared information channels. Its conclusions apply to these representations and this data-generating mechanism, rather than to all possible memory architectures.

Finally, feature legitimacy, independence assumptions, and the scientific target of prediction are central to reproducibility [7]. A task can be valid conditional on supplied information while being unsuitable for a diferent deployment-time claim. Real educational outcomes require additional evidence. For example, a real-world educational reinforcement-learning study found no significant reading-competence diference from its expert baseline despite improvements within both groups [8]. Simulator accuracy and educational efectiveness should therefore remain separate conclusions.

## 3 Data Provenance and Information Contract

## 3.1 Inspectable simulator records

The source archive contains 30,000 simulated starting profiles, five policy-generated routes per profile, and 48 months per route. Its 150,000 routes are not 150,000 independently observed students. The generator identifies profiles and monthly histories as synthetic, marks numerical coeficients as uncalibrated design assumptions, and provides deterministic reconstruction under its frozen version and seed. Public factual references inform scenario vocabulary and observable endpoint fragments; they do not supply complete longitudinal observations for these generated profiles.

The predictor targets nine internal preparation indices, an internal fatigue index, and a discretionary-funds balance in scenario yuan. These quantities follow explicit transition equations and stochastic execution. They are not calibrated probabilities of admission or employment, psychometric scores, or verified personal financial histories. Source family groups contain 25 profile configurations. Each profile’s five routes share its starting conditions, and related configurations within a family must remain together when splitting and resampling.

We reconstruct 104 families for the expanded experiment: 48 training, 16 validation, and 40 evaluation families. This yields 1,200 training profiles, 400 validation profiles, and 1,000 original evaluation profiles, or 6,000, 2,000, and 5,000 routes, respectively. The separate 30-family evaluation adds 750 profiles and 3,750 routes. Thus the study processes 134 distinct families, 3,350 profiles, 16,750 routes, and 804,000 monthly records in total. Only the two evaluation sets are used to report held-out prediction performance; the total count includes training and validation.

## 3.2 State, event, plan, and execution order

Figure 1 depicts the source ordering. For monthly record t, the simulator first stores a pre-event state $x _ { t }$ . It then samples events and updates resources, forms the saved plan ${ \boldsymbol { a } } _ { t } ,$ executes activities stochastically, and records the resulting state $y _ { t }$ . Consecutive records satisfy $y _ { t } = x _ { t + 1 }$ for the selected core coordinates. Nominal context $c _ { t }$ contains fixed profile resources, scenario descriptors, and deterministic calendar features.

Our conditional input consists of $x _ { t } , \ c _ { t } .$ , the saved plan when enabled, and permitted earlier observations. Actual hours, realized spending, event fields, execution status, narrative rationale, identifiers, and target states are excluded. Nevertheless, $a _ { t }$ is formed after the current event and resource update. It can consequently carry predictive information about those changes. $\mathrm { A }$ comparison with and without $a _ { t }$ estimates the value of this recorded information under the declared task; it does not identify the efect of forcing an action.

Multi-step evaluation supplies recorded future plans. Those plans were originally formed using later realized states and events. Consequently, the rollout is retrospective conditional state prediction, not a forecast available from all information at the initial month. Excluding future true state vectors from recursion does not make these plans available prospectively. We use the word “forecasting” only with this explicit conditioning qualification and make no claim of autonomous replanning.

Events can influence both the saved plan and the resulting state.

![](images/1a23abaf9ae51092032310fd7aff9df32204db88e90fd9454bcd91853e1374b1.jpg)  
Figure 1: Information timing in the source simulator. One-step prediction is conditional on an already formed saved plan. Recursive evaluation additionally supplies future recorded plans, but not future observed state vectors.

## 4 Prediction Models and Temporal Summaries

## 4.1 Strictly past summary features

Let $t \in \{ 0 , \ldots , 4 7 \}$ and let $k _ { t } ( w ) = \operatorname* { m i n } ( t , w )$ for a finite window w, or $k _ { t } ( \infty ) = t$ for all available history. For $k _ { t } > 0$ , we define

$$
m _ { t } ^ { x } ( w ) = \frac { 1 } { k _ { t } } \sum _ { j = t - k _ { t } } ^ { t - 1 } x _ { j } ,\tag{1}
$$

$$
m _ { t } ^ { \Delta } ( w ) = \frac { x _ { t } - x _ { t - k _ { t } } } { k _ { t } } ,\tag{2}
$$

$$
m _ { t } ^ { a } ( w ) = \frac { 1 } { k _ { t } } \sum _ { j = t - k _ { t } } ^ { t - 1 } a _ { j } .\tag{3}
$$

The past-state mean excludes $x _ { t } .$ , and the past-plan mean excludes $a _ { t }$ . The change summary is the mean of completed state diferences ending at the current observed state. Its telescoping form shows that it is an endpoint slope, not a representation of every transition or its order. At $t = 0$ , all summaries are zero. Summary models additionally receive

$$
\ell _ { t } ( w ) = \left( k _ { t } ( w ) / 4 8 , { \bf 1 } \{ t > 0 \} \right) .\tag{4}
$$

The baseline feature vector is $[ x _ { t } , c _ { t } , a _ { t } ]$ . The full windowed vector appends all three summaries and $\ell _ { t } ( w )$ . We evaluate windows of 1, 3, 6, and 12 months and all available history; componentonly and component-removal specifications; current-plan removal; and two length-only controls. A length-only model appends $\ell _ { t } ( w )$ but no historical states or plans. Although the baseline already contains a month fraction and seasonal indicators, finite-window $\ell _ { t } ( w )$ introduces a clipped linear function and an initial-record indicator. It therefore supplies a diferent calendar basis to a linear model.

The baseline has 94 input coordinates, the full history model 164, and the length-only control 96. For a finite window, the count supplies a saturating calendar function. For all history, $k _ { t } / 4 8 = t / 4 8$ is afine-equivalent to the baseline month fraction $( t + 1 ) / 4 8$ , while the nonempty indicator still adds an initial-record distinction. Duplicating an afine-equivalent column can also change the Ridge penalty geometry. Thus these controls expose specific clock and representation efects rather than perfectly removing every calendar or capacity diference. Comparisons concern additional summary information within the specified regularized linear model family. They do not isolate an architecture-independent value of memory. Means discard temporal order, and the all-history slope can be reconstructed from the current and initial states with elapsed time.

## 4.2 Shared regularized regression

All 17 learned specifications use the same Ridge fitting procedure. Features and target changes $y _ { t } - x _ { t }$ are standardized using training records only. An unpenalized intercept accompanies a squared-error objective with an $L _ { 2 }$ penalty on coeficients. Each specification independently selects its penalty from {0.01, 0.1, 1, 10, 100, 1000} by the validation one-step score; test records never select a penalty or window. Predictions are obtained by adding the predicted change to the current state. We also report persistence, which retains the initial state throughout a rollout. Thus the original evaluation contains 17 learned Ridge specifications and one nonlearned baseline, not 18 trained models.

No projection or clipping is applied to the reported predictions. This preserves model failures outside the source mechanism’s support. It also means that a numerical regression prediction should not be treated as a feasible state estimate merely because its average error is small.

## 5 Evaluation Design and Audits

## 5.1 Origins, targets, and score

One-step evaluation uses all 48 records per route with the contemporaneous observed state. Recursive evaluation uses seven common zero-based origins {0, 6, 12, 18, 24, 30, 36} and a 12-step horizon. At each origin, only the observed state prefix through that origin initializes the predictor. Every subsequent state, including states used in updated history summaries, is a prediction. Recorded future plans and deterministic future context remain supplied. All specifications use identical origins and endpoint records.

A matched teacher-forced reference predicts the same endpoint record from its contemporaneous true state. This distinguishes recursive error accumulation from changing calendar composition. The all-record one-step score is not substituted for the horizon-one score on the seven-origin subset. Curves in Figure 2 should be read under these matched-origin contracts.

For coordinate $d ,$ let $\sigma _ { d }$ denote its post-state standard deviation in the training set. The aggregate score is

$$
\mathrm { n R M S E } = \sqrt { \frac { 1 } { 1 1 N } \sum _ { i = 1 } ^ { N } \sum _ { d = 1 } ^ { 1 1 } \frac { ( \widehat { y } _ { i d } - y _ { i d } ) ^ { 2 } } { \sigma _ { d } ^ { 2 } } } .\tag{5}
$$

We also retain coordinate-wise RMSE and MAE. Group scores separate the nine preparation indices, fatigue, and funds. Normalization removes unit diferences but does not assign educational utility or ensure equal error contributions.

Paired percentile intervals resample families jointly across compared predictors, preserving every route and profile within a sampled family. We use 10,000 replicates with 40 or 30 clusters, respectively. All families have 125 routes, so equally weighted family MSE reproduces the pooled point score before taking its square root. These intervals are conditional on fixed fitted models and the simulator evaluation distribution. They do not include uncertainty about the realism of the source equations. Secondary windows, components, outcomes, and horizons are reported with descriptive intervals without multiplicity adjustment; we do not select a winning specification from those intervals as a new primary hypothesis.

## 5.2 Local freezing and prior exposure

The expanded protocol specifies the six-month-history versus baseline comparison at one step and the six-step endpoint. A pre-evaluation addendum supplies history-length controls and matched teacher-forced references. These files are locally frozen, hash-recorded analysis specifications; they are not a formally registered preregistration. Earlier pilot results informed the study question.

A subsequent provenance check identified prior exposure of 22 original evaluation families across dynamics, stress, action-fork, and supervised-fine-tuning tasks. The original 40-family results are retained rather than selectively replaced. Exclusion analyses are labeled postobservation sensitivities. A further protocol selected 30 families from the source test partition, excluding all identified prior task families and the entire original 40-family set. It locked original weights, preprocessing, and penalty choices before evaluating those families. This fresh same-rule evaluation includes nine selected learned Ridge specifications and persistence; it is not a new training run, a diferent simulator, or an external real-student cohort.

## 6 Results

## 6.1 History compared with the conventional baseline

Table 1 reports the original evaluation. Six-month history lowers one-step nRMSE from 0.058750 to 0.057009, a 2.96% relative reduction. At the six-step endpoint, the score decreases from 0.159234 to 0.151522, a 4.84% reduction; the paired diference is −0.007712, with interval [−0.008996, −0.006443]. Twelve-step scores are 0.191085 and 0.180771. Persistence is substantially weaker, reaching 0.320327 at six steps and 0.526675 at twelve, but this weak baseline alone does not establish the specific value of historical summaries.

Locked evaluation on the fresh families gives a similar baseline comparison (Table 2). One-step scores are 0.058400 and 0.056828, a 2.69% relative reduction. Six-step scores are 0.156480 and 0.150074, a 4.09% reduction, with paired diference −0.006406 and interval [−0.007866, −0.005093]. These observations extend the pattern to documented previously unused families under identical source rules. They do not test transfer to other mechanisms or actual student records.

Table 1: Original 40-family evaluation: aggregate normalized RMSE. One-step prediction uses all records; recursive endpoints use the same seven origins per route. All predictions remain unprojected.
<table><tr><td>Specification</td><td>1 step</td><td>6 steps</td><td>12 steps</td></tr><tr><td>Current state + plan</td><td>0.058750</td><td>0.159234</td><td>0.191085</td></tr><tr><td>History-1</td><td>0.057834</td><td>0.156365</td><td>0.187299</td></tr><tr><td>History-3</td><td>0.057238</td><td>0.153452</td><td>0.183228</td></tr><tr><td>History-6</td><td>0.057009</td><td>0.151522</td><td>0.180771</td></tr><tr><td>History-12</td><td>0.056945</td><td>0.150763</td><td>0.178771</td></tr><tr><td>All history</td><td>0.057132</td><td>0.153006</td><td>0.180879</td></tr><tr><td>Count-6 only</td><td>0.058017</td><td>0.152417</td><td>0.184461</td></tr><tr><td>Count-all only</td><td>0.058639</td><td>0.157927</td><td>0.189845</td></tr><tr><td>Current, no plan</td><td>0.064450</td><td>0.186877</td><td>0.250815</td></tr><tr><td>History-6, no current plan</td><td>0.061598</td><td>0.174394</td><td>0.239969</td></tr><tr><td>All history, no current plan</td><td>0.061853</td><td>0.191677</td><td>0.373956</td></tr><tr><td>Persistence</td><td>0.077200</td><td>0.320327</td><td>0.526675</td></tr></table>

## 6.2 The history-length control changes the interpretation

At six steps, the original length-only control scores 0.152417, close to the full six-month-history score of 0.151522. The remaining relative reduction is 0.59%, and its paired diference −0.000895 has interval [−0.001957, 0.000248]. Fresh-family scores are 0.150632 and 0.150074; the remaining reduction is 0.37%, with diference −0.000558 and interval [−0.001783, 0.000728]. Table 3 separates these comparisons. Both six-step intervals include zero, so the evidence does not resolve an aggregate improvement over the length-only control at this endpoint.

Table 2: Fixed-weight validation on 30 previously unused families under identical simulator rules. No preprocessing, coeficient, or penalty choice is fitted to this set.
<table><tr><td>Specification</td><td>1 step</td><td>6 steps</td><td>12 steps</td></tr><tr><td>Current state + plan</td><td>0.058400</td><td>0.156480</td><td>0.186687</td></tr><tr><td>History-6</td><td>0.056828</td><td>0.150074</td><td>0.176040</td></tr><tr><td>History-12</td><td>0.056719</td><td>0.149420</td><td>0.174548</td></tr><tr><td>All history</td><td>0.056830</td><td>0.150692</td><td>0.175504</td></tr><tr><td>Count-6 only</td><td>0.057741</td><td>0.150632</td><td>0.180099</td></tr><tr><td>Count-all only</td><td>0.058286</td><td>0.155242</td><td>0.185422</td></tr><tr><td>Current, no plan</td><td>0.063900</td><td>0.183541</td><td>0.245420</td></tr><tr><td>History-6, no current plan</td><td>0.061259</td><td>0.172388</td><td>0.234944</td></tr><tr><td>All history, no current plan</td><td>0.061387</td><td>0.189648</td><td>0.373426</td></tr><tr><td>Persistence</td><td>0.077709</td><td>0.326292</td><td>0.543952</td></tr></table>

This is not evidence that history has no predictive value. Against the length-only control, one-step diferences are −0.001008 on the original families and −0.000913 on fresh families; their descriptive intervals lie below zero. Twelve-step diferences are −0.003690 and −0.004059, also with intervals below zero. The horizon dependence and target decomposition matter. Nor is the longest summary uniformly best: the original all-history six-step score is 0.153006, compared with 0.150763 for the 12-month summary; fresh scores are 0.150692 and 0.149420. These are fixed-window observations, not a test-selected recommendation of an optimal window.

The all-history model also improves on its own count-all control at six steps: the original diference is −0.004921, with descriptive interval [−0.006202, −0.003696], and the fresh-family diference is −0.004550, with interval [−0.005387, −0.003731]. These controls have diferent bases. Count-6 introduces a hinge that saturates after six observations, whereas count-all introduces an unsaturated linear count and the initial-record indicator. Consequently, the uncertain six-month comparison must not be generalized to every historical representation or calendar control.

Table 3: Paired family-bootstrap diferences against length-only controls. Negative diferences favor historical content. These 95% intervals are descriptive and unadjusted across specifications and horizons.
<table><tr><td>Set</td><td>Comparison</td><td>Step</td><td>∆ nRMSE</td><td>95% interval</td></tr><tr><td>Original</td><td>History-6 vs count-6</td><td>1</td><td>-0.001008</td><td>[-0.001138, -0.000882]</td></tr><tr><td>Original</td><td>History-6 vs count-6</td><td>6</td><td>-0.000895</td><td>[-0.001957, +0.000248]</td></tr><tr><td>Original</td><td>History-6 vs count-6</td><td>12</td><td>-0.003690</td><td>[-0.006299, -0.000782</td></tr><tr><td>Original</td><td>All history vs count-all</td><td>1</td><td>-0.001507</td><td>[-0.001667, -0.001358]</td></tr><tr><td>Original</td><td>All history vs count-all</td><td>6</td><td>-0.004921</td><td>[-0.006202, -0.003696]</td></tr><tr><td>Original</td><td>All history vs count-all</td><td>12</td><td>-0.008966</td><td>[-0.011098, -0.006899]</td></tr><tr><td>Fresh</td><td>History-6 vs count-6</td><td>1</td><td>-0.000913</td><td>[-0.001021, -0.000805]</td></tr><tr><td>Fresh</td><td>History-6 vs count-6</td><td>6</td><td>-0.000558</td><td>[-0.001783, +0.000728]</td></tr><tr><td>Fresh</td><td>History-6 vs count-6</td><td>12</td><td>-0.004059</td><td>[-0.007459, -0.000661]</td></tr><tr><td>Fresh</td><td>All history vs count-all</td><td>1</td><td>-0.001455</td><td>[-0.001624, -0.001299</td></tr><tr><td>Fresh</td><td>All history vs count-all</td><td>6</td><td>-0.004550</td><td>[-0.005387, -0.003731]</td></tr><tr><td>Fresh</td><td>All history vs count-all</td><td>12</td><td>-0.009917</td><td>[-0.011729, -0.008024]</td></tr></table>

![](images/213f8a1409ed5eebb3ecfa3f7dacf7dea9b5bb17c07e25a382439dfcdc18ce7a.jpg)

![](images/b1d6cf363bdd96ded4191ab5bd3453743de50adbda5ef217cfa635ce051aa261.jpg)  
Figure 2: Recursive errors with future recorded plans supplied. Dashed lines are matched one-step predictions using contemporaneous observed states at exactly the same endpoint months and origins (current and history-6). The all-record one-step table uses a diferent record composition.

## 6.3 Diferent state groups give diferent answers

Figure 3 decomposes the original evaluation’s normalized squared error. For the six-monthhistory predictor, fatigue contributes 98.72% of one-step loss and 97.80% of six-step loss. The nine preparation coordinates contribute only 1.20% and 2.00%, respectively. This reflects their difering predictability relative to training variation, rather than an intentional preference for fatigue.

An independently replayed exploratory comparison against the length-only control finds a six-step preparation-group nRMSE diference of −0.008040, with paired interval [−0.008538, −0.007556]; all 40 families favor the historical summaries on that group score. Funds give −0.002280, with interval [−0.002631, −0.001939] and 39 family wins. Fatigue instead gives +0.001135, with interval [−0.002419, 0.005009]. The aggregate comparison therefore combines clearer preparation and funds improvements with uncertainty in its dominant coordinate. A claim of uniform improvement across the student state would misrepresent this result. These group intervals are exploratory and derived from the original 40 families; they are not presented as independent real-world validation.

![](images/1e65a8a5504265761040af2102d751d31ca1d36ae8caf3e130115f9874b845c4.jpg)  
Figure 3: Additive shares of the aggregate training-standardized squared error. Fatigue dominates the aggregate despite normalization; improvements in other coordinates can therefore be obscured. RMSE itself is not additive.

Table 4: Exploratory six-step group comparison: history-6 minus count-6 on the original 40 families. Group scores use their own coordinate averages; they are not additive RMSE contributions.
<table><tr><td>State group</td><td>∆ group nRMSE</td><td>95% interval</td><td>Family wins</td></tr><tr><td>Preparation (9)</td><td>-0.008040</td><td>[-0.008538, -0.007556]</td><td>40/40</td></tr><tr><td>Fatigue</td><td>+0.001135</td><td>[-0.002419, +0.005009]</td><td>22/40</td></tr><tr><td>Funds</td><td>-0.002280</td><td>[-0.002631, -0.001939]</td><td>39/40</td></tr></table>

## 6.4 Predictive error does not ensure valid support

The source fatigue transition clamps outcomes to [5, 90]. A broad [0, 100] check misses many infeasible predictions. Table 5 retains both checks. On the original evaluation, the baseline violates the strict interval in 31.369% of one-step and 30.786% of six-step predictions. The sixmonth-history model reduces these rates to 28.860% and 28.200%, but its six-step broad-interval violation rate is only 0.123%. All of its strict six-step violations are predictions below 5.

These results preclude describing the unprojected outputs as consistently support-valid. Projection onto a verified target interval cannot increase pointwise squared error when the target belongs to that interval; such an improvement would be an elementary constraint efect rather than a new learning contribution. Recursive projection would change later inputs and require a separate evaluation. We preserve the original unprojected predictions here rather than silently repairing the principal results.

Table 5: Original evaluation fatigue predictions outside the actual [5,90] support versus the broader [0,100] interval. Strict failures are retained in the reported scores.
<table><tr><td>Specification</td><td>Strict: 1</td><td>Strict: 6</td><td>Strict: 12</td><td>Wide: 6</td></tr><tr><td>Current state + plan</td><td>31.369%</td><td>30.786%</td><td>31.500%</td><td>0.137%</td></tr><tr><td>History-6</td><td>28.860%</td><td>28.200%</td><td>28.560%</td><td>0.123%</td></tr><tr><td>Count-6 only</td><td>30.404%</td><td>30.414%</td><td>32.300%</td><td>0.089%</td></tr><tr><td>All history</td><td>28.809%</td><td>27.280%</td><td>28.600%</td><td>0.437%</td></tr></table>

## 7 Reproducibility, Limitations, and Implications

The reproducibility artifacts record source, data, protocol, result, and checkpoint hashes; exact group manifests; the full penalty grid; coordinate and family errors; and independent replay code. Checks verify split disjointness, state continuity, train-only preprocessing, and locked fresh-family weights. Independent inference without importing the training implementation reproduces all five reported task scores for all 17 learned specifications exactly at the recorded numerical precision. Matched-origin teacher-forced versus horizon-one recursive predictions difer by at most $8 . 5 4 \times 1 0 ^ { - 8 }$ in training-standardized units. Poisoning unobserved future states and target labels leaves permitted-origin predictions unchanged. These checks establish implementation consistency for the declared conditional contract, not empirical educational validity.

Several limitations define the scope. First, the mechanism and numeric coeficients are designed assumptions; multiple simulated routes cannot replace authorized longitudinal observations. Second, current and future saved plans can encode event information. Third, only transparent linear summary representations are tested, so conclusions neither characterize all recurrent systems nor evaluate language-model conversational memory. Fourth, prior exposure afected the original family set, although locked fresh-family evaluation addresses the documented overlap within the same mechanism. Fifth, conditional family-bootstrap intervals and unadjusted secondary intervals should not be interpreted as universal or simultaneous guarantees. Finally, prediction gains do not measure policy benefits: no autonomous action-selection claim follows from this experiment.

The methodological implication is practical. Evaluations of history should report the prediction-time information contract, add clock-aware controls when history handling creates calendar features, align recursive and teacher-forced targets, preserve dependence among scenario groups, and inspect each consequential output rather than rely on one aggregate. Here those steps yield a qualified result: compact historical summaries contain useful predictive information for several simulator coordinates, while calendar controls and fatigue error substantially change the interpretation of the aggregate six-step gain. A credible extension to university planning would require independently measured outcomes and a policy evaluation against strong fixed and expert baselines, with the same attention to information timing and failure disclosure.

## A Complete Specifications and Artifact Contract

The accompanying computational material contains deterministic reconstruction code, frozen protocols and amendments, split manifests, fitted model arrays, validation penalty candidates, per-family errors, and independent replay audits. The source generator and its required factualreference pool must remain frozen to reproduce the archived scenario profiles. No archival data license or individual source permission is inferred from public availability. Until a permanent public repository is supplied, this manuscript should be treated as a local reproducibility package rather than a claim of public open-source availability.

The declared information boundary excludes identifiers, route-policy labels, hidden generator assumptions, actual hours, realized spending, completion status, event fields, target outcomes, and post-execution narrative fields. In a recursive forecast, historical states after the origin are model predictions. Saved future plans are nevertheless permitted conditioning variables and may encode subsequent realized information.

## B AI Tools and Human Responsibility

Generative AI tools assisted research organization, code development, literature screening, drafting, and independent code and result checks. Statistical tables and figures were computed from recorded experiment outputs; they were not generated as plausible illustrations of an unrun experiment. A human submitting author must verify every claim, citation, provenance statement, and permission and assumes responsibility for the final work. AI tools are not authors. No human author, afiliation, ethical approval, observational cohort, or data-sharing permission has been invented for this draft.

Table 6: Complete original specifications. Component-only and component-removal rows retain current state, context, current plan, and history-count indicators; only the named past-summary component changes. Explicit current-plan removal rows are separate. All results, including unfavorable outcomes, are retained. Penalties were selected on validation data only.
<table><tr><td>Specification</td><td>1 step</td><td>6 steps</td><td>12 steps</td></tr><tr><td>Current state + plan</td><td>0.058750</td><td>0.159234</td><td>0.191085</td></tr><tr><td>History-1</td><td>0.057834</td><td>0.156365</td><td>0.187299</td></tr><tr><td>History-3</td><td>0.057238</td><td>0.153452</td><td>0.183228</td></tr><tr><td>History-6</td><td>0.057009</td><td>0.151522</td><td>0.180771</td></tr><tr><td>History-12</td><td>0.056945</td><td>0.150763</td><td>0.178771</td></tr><tr><td>All history</td><td>0.057132</td><td>0.153006</td><td>0.180879</td></tr><tr><td>History-6: past-state mean only</td><td>0.057615</td><td>0.152877</td><td>0.185093</td></tr><tr><td>History-6: past slope only</td><td>0.057328</td><td>0.151982</td><td>0.183769</td></tr><tr><td>History-6: past-plan mean only</td><td>0.057945</td><td>0.152914</td><td>0.184963</td></tr><tr><td>History-6: omit past-state mean</td><td>0.057008</td><td>0.151687</td><td>0.181453</td></tr><tr><td>History-6: omit past slope</td><td>0.057503</td><td>0.153343</td><td>0.185511</td></tr><tr><td>History-6: omit past-plan mean</td><td>0.057330</td><td>0.151891</td><td>0.183382</td></tr><tr><td>History-6, no current plan</td><td>0.061598</td><td>0.174394</td><td>0.239969</td></tr><tr><td>All history, no current plan</td><td>0.061853</td><td>0.191677</td><td>0.373956</td></tr><tr><td>Current, no plan</td><td>0.064450</td><td>0.186877</td><td>0.250815</td></tr><tr><td>Count-6 only</td><td>0.058017</td><td>0.152417</td><td>0.184461</td></tr><tr><td>Count-all only</td><td>0.058639</td><td>0.157927</td><td>0.189845</td></tr><tr><td>Persistence</td><td>0.077200</td><td>0.320327</td><td>0.526675</td></tr></table>

## References

[1] Chris Piech, Jonathan Bassen, Jonathan Huang, Surya Ganguli, Mehran Sahami, Leonidas Guibas, Jascha Sohl-Dickstein. Deep Knowledge Tracing. Advances in Neural Information Processing Systems 28, 2015. https://proceedings.neurips.cc/paper/2015/hash/ bac9162b47c56fc8a4d2a519803d51b3-Abstract.html.

[2] Qi Liu, Shiwei Tong, Chuanren Liu, Hongke Zhao, Enhong Chen, Haiping Ma, Shijin Wang. Exploiting Cognitive Structure for Adaptive Learning. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019. DOI: 10.1145/3292500.3330922. https://arxiv.org/html/1905.12470v1.

[3] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, 2014. DOI: 10.3115/v1/D14-1179. https://aclanthology.org/D14-1179/.

[4] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson. Learning Latent Dynamics for Planning from Pixels. Proceedings of the 36th International Conference on Machine Learning, 2019. https://proceedings.mlr.press/ v97/hafner19a.html

[5] Stephane Ross, Geofrey Gordon, Drew Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, 2011. https://proceedings. mlr.press/v15/ross11a.html.

[6] Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Y Zou, Sergey Levine, Chelsea Finn, Tengyu Ma. MOPO: Model-based Ofline Policy Optimization. Advances in Neural Information Processing Systems 33, 2020. https://proceedings.neurips.cc/ paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html.

[7] Sayash Kapoor, Arvind Narayanan. Leakage and the reproducibility crisis in machinelearning-based science. Patterns, 2023. DOI: 10.1016/j.patter.2023.100804. https://pmc. ncbi.nlm.nih.gov/articles/PMC10499856/.

[8] Anna Riedmann, Carlo D’Eramo, Birgit Lugrin. Real-World Testing Matters in Reinforcement Learning for Education. Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems, 2025. https://ifaamas.csc.liv.ac.uk/ Proceedings/aamas2025/pdfs/p1764.pdf.