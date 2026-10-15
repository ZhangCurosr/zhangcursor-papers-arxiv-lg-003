# When Does Task-Aware Imputation Matter? An Analysis of Robustness Under Train–Test Mismatch

Maria V. Salvador INESC TEC University of Porto maria.v.salvador@inesctec.pt

Ana F. Sequeira INESC TEC University of Porto ana.f.sequeira@inesctec.pt

Jaime S Cardoso INESC TEC University of Porto jaime.cardoso@inesctec.pt

## Abstract

Missing data is a pervasive challenge in machine learning, often degrading predictive performance and complicating model deployment. Most existing approaches treat imputation as a preprocessing step independent of the downstream task, a separation that becomes problematic under train–test mismatches in missingness mechanisms or rates. We introduce a unified framework that treats missing entries as variables in a task-conditioned optimization problem. We instantiate this framework in two settings: an amortized approach for scenarios with missing data at both training and test time, and a latent optimization approach for settings where only training data is incomplete. The evaluation is conducted on 98 tabular datasets and robustness under train–test mismatches is assessed on 36 datasets. Results reveal that reconstruction error constitutes a poor proxy for predictive utility (Spearman ρ ≈ 0.16–0.32), and that task-aware imputation becomes critical in scenarios where classical methods fail to recover complete-data performance, achieving win rates of up to 80% in challenging train–test mismatch scenarios.

## 1 Introduction

Missing data is a pervasive problem in real-world machine learning, with significant implications for the validity, reliability, and fairness of predictive models. In practice, data may be incomplete due to collection errors, equipment failures, participant drop-out, or reluctance to disclose sensitive information. Missing values can lead to biased estimates, reduced statistical power, and degraded performance, making their handling a critical component of any data-driven pipeline.

Understanding why data is missing– the missingness mechanism– is essential. There are three main missingness mechanisms: Missing Completely At Random (MCAR), where the probability of missingness is independent of the data; Missing At Random (MAR), where the probability of missingness is independent of the missing values given the observed data; and Missing Not At Random (MNAR), where the probability of missingness depends on the unobserved values themselves [1, 2]. These mechanisms not only influence how missing data should be modeled, but also interact with when missingness occurs in the learning pipeline.

In practice, an important consideration is whether missingness occurs at inference time. In some applications, missing values appear only during training, while deployment data is clean [2]. In others, missingness persists at test time, often under shifts in mechanisms or rates [3, 4]. These scenarios pose distinct challenges: the former primarily affects representation learning, while the latter requires robustness to distribution shift. Despite these differences, most imputation methods treat missing data as a static preprocessing problem, without explicitly accounting for when or how missingness occurs.

This raises a fundamental question: to what extent does improving imputation quality translate into better predictive performance, particularly in realistic settings where missingness persists at test time or differs between training and deployment? This question leads to the primary goal of this work, which is to investigate when task-aware imputation improves predictive performance in supervised learning with missing values.

Our contributions are threefold. First, we show that task-aware imputation is not universally beneficial, but instead becomes critical in challenging regimes such as distribution shift and intrinsically hard missingness settings. Second, we introduce a simple and flexible formulation that treats missing values as learnable variables jointly optimized with the predictive model, providing a common perspective that encompasses both amortized and latent approaches within a single framework. Third, through a comprehensive empirical study, we demonstrate that standard reconstruction metrics are only weakly correlated with downstream performance, highlighting the limitations of conventional evaluation protocols.

## 2 Related Work

Approaches to handling missing data include imputation, where missing entries are filled; omission, where incomplete samples are discarded; and methods that directly operate on incomplete data [2, 5, 6]. In this work, we focus on imputation.

Classical methods such as mean, median, and K-NN [7] are simple and efficient but fail to capture complex dependencies. Iterative approaches such as MissForest [8], MICE [9], and related conditional models [10] model non-linear relationships but are computationally expensive and sensitive to design choices. HyperImpute [11] extends this paradigm through automated model and hyperparameter selection, while low-rank matrix completion methods exploit global structure [12, 13]. Learning-based approaches such as TabPFN [14] and TabImpute [15] leverage pretrained models for fast, zero-shot imputation. Deep learning methods for tabular data, including transformer-based architectures [16, 17], have further advanced this direction.

Several approaches model the joint data distribution using deep generative models [18–26], learning to reconstruct missing entries through reconstruction objectives, often with masking strategies during training. While expressive, these methods typically prioritize data fidelity over downstream predictive performance. Other works incorporate additional structure into the imputation process. For instance, MIRACLE [27] models the missingness mechanism under MNAR using causal assumptions, while optimal transport-based approaches align distributions between observed and missing data [28, 29].

Handling missing data in supervised learning has traditionally relied on two-stage pipelines, where missing values are first imputed and then used for downstream prediction. This separation can lead to suboptimal performance, as imputation is not aligned with the predictive task. Benchmark studies show that the choice of imputation method significantly affects downstream performance, particularly in terms of robustness and interpretability [30]. Moreover, standard metrics such as RMSE and MAE often fail to reflect imputation quality, as they do not capture discrepancies in the underlying data distribution.

Task-aware approaches aim to address this limitation. TARImpute [31] co-optimizes imputation with clustering objectives, while OptImpute [32] formulates imputation as an optimization problem independent of downstream supervision. Deep learning methods such as GRAPE [33] and GEDI [34] propose end-to-end frameworks that jointly perform imputation and prediction, often relying on specific architectural choices such as graph-based representations.

Changes in missingness mechanisms can also induce distribution shift. Domain Adaptation under Missingness Shift (DAMS) [3] shows that differences in missingness alone can invalidate standard assumptions and require adaptation, particularly when missingness indicators are unavailable. Similarly, StableMiss+ [4] studies mask distribution shifts and proposes removing unstable correlations involving the missingness mask.

These works highlight two key challenges: (i) robustness under shifts in missingness mechanisms or rates, and (ii) the misalignment between imputation quality and downstream predictive performance.

We address these challenges by introducing a unified formulation that treats missing values as learnable variables jointly trained with the predictive model. This perspective directly aligns imputation with the task objective, and allows us to characterize when such alignment translates into improved predictive performance. Unlike prior task-aware approaches, our formulation does not depend on specific architectures or specialized designs (e.g., graph-based models), and avoids the need for bilevel optimization. Instead, it provides a simple and flexible lens to study task-aware imputation itself, encompassing both per-sample (latent) and amortized strategies within a single formulation. Furthermore, we explicitly investigate train–test mismatches in missingness mechanisms and rates, a setting that remains largely underexplored in existing work.

## 3 Joint-AmortizedImpute: Task-Aware Imputation via Joint Optimization

Let $\mathbf { \mathcal { D } } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ be a dataset of n samples, where:

$\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ is the input feature vector for the i-th sample,

$y _ { i } \in \mathcal { V }$ is the corresponding label or target value,

• some components of $\mathbf { x } _ { i }$ may be missing.

Let $M \in \{ 0 , 1 \} ^ { n \times d }$ be a binary mask matrix such that:

$$
m _ { i j } = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f } x _ { i j } \mathrm { i s } \mathrm { o b s e r v e d } , } \\ { 0 } & { \mathrm { i f } x _ { i j } \mathrm { i s } \mathrm { m i s s i n g } . } \end{array} \right.
$$

The learning objective is to find a function $f _ { \theta } : \mathbb { R } ^ { d }  \mathcal { V } _ { \mathrm { ~ } }$ , parameterized by $\theta ,$ that minimizes the expected loss:

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { ( \mathbf { x } , y ) \sim \mathcal { P } } \left[ \ell ( f _ { \theta } ( \mathbf { x } ) , y ) \right] ,\tag{1}
$$

where ℓ is a loss function and $\mathcal { P }$ is the underlying data distribution.

However, due to missing values, we only observe $\mathbf { x } _ { i } ^ { \mathrm { { o b s } } } = \mathbf { x } _ { i } \odot \mathbf { m } _ { i }$ , and do not observe

$\mathbf { x } _ { i } ^ { \mathrm { u } } = \mathbf { x } _ { i } \odot \left( \mathbf { 1 } - \mathbf { m } _ { i } \right)$ , where ⊙ denotes element-wise multiplication. The learning problem becomes:

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { ( \mathbf { x } ^ { \mathrm { o b s } } , y ) \sim \mathcal { P } _ { \mathrm { o b s } } } \left[ \ell ( f _ { \theta } ( \mathbf { x } ^ { \mathrm { o b s } } , \hat { \mathbf { x } } ^ { \mathrm { u } } ) , y ) \right] ,\tag{2}
$$

where $\hat { \mathbf { x } } ^ { \mathrm { u } }$ is an estimate of $\mathbf { x } ^ { \mathrm { { u } } }$ with an imputation strategy.

We propose to jointly optimize the predictive model and the missing values in a unified objective. Let $\mathbf { x } ^ { \mathrm { o b s } }$ and $\mathbf { x } ^ { \mathrm { { u } } }$ denote the observed and unobserved components of the input, respectively. We consider the following objective:

$$
\operatorname* { m i n } _ { \theta , \mathbf { x } ^ { \mathrm { u } } } \mathbb { E } _ { ( \mathbf { x } ^ { \mathrm { o b s } } , y ) \sim \mathcal { P } _ { \mathrm { o b s } } } \left[ \ell \big ( f _ { \theta } ( \mathbf { x } ^ { \mathrm { o b s } } , \mathbf { x } ^ { \mathrm { u } } ) , y \big ) \right] .\tag{3}
$$

To regularize the solution, we introduce a penalty on the unknown components:

$$
\operatorname* { m i n } _ { \theta , \mathbf { x } ^ { \mathrm { u } } } \mathbb { E } _ { ( \mathbf { x } ^ { \mathrm { o b s } } , y ) } \left[ \ell ( f _ { \theta } ( \mathbf { x } ^ { \mathrm { o b s } } , \mathbf { x } ^ { \mathrm { u } } ) , y ) \right] + g ( \mathbf { x } ^ { \mathrm { u } } ) ,\tag{4}
$$

where $g$ encodes prior assumptions about plausible imputations.

Reference-based parameterization. We express missing values as deviations from a reference imputation $\hat { \mathbf { x } } ^ { \mathrm { u } }$

$$
\begin{array} { r } { \mathbf { x } ^ { \mathrm { u } } = \hat { \mathbf { x } } ^ { \mathrm { u } } + \pmb { \delta } , } \end{array}\tag{5}
$$

This parameterization can be interpreted as introducing a strong prior over plausible imputations through the reference solution. It stabilizes optimization by anchoring the search in a realistic region of the input space, while allowing task-aware corrections to refine the imputation. Moreover, it reduces the effective search space, enabling the model to focus on adjustments that improve downstream performance rather than reconstructing values from scratch.

This leads to the equivalent objective:

$$
\operatorname* { m i n } _ { \theta , \delta } \mathbb { E } _ { ( \mathbf { x } ^ { \mathrm { o b s } } , y ) } \left[ \ell \big ( f _ { \theta } ( \mathbf { x } ^ { \mathrm { o b s } } , \hat { \mathbf { x } } ^ { \mathrm { u } } + \delta ) , y \big ) \right] + g ( \delta ) .\tag{6}
$$

A simple choice is a quadratic regularization:

$$
g ( \delta ) = \lambda \| \delta \| ^ { 2 } ,\tag{7}
$$

which encourages imputations to remain close to the reference.

Unified imputation operator. We define a general imputation operator:

$$
\mathbf { x } ^ { \mathrm { u } } = \mathcal { T } _ { \psi } ( \mathbf { x } ^ { \mathrm { o b s } } , \mathbf { m } , \hat { \mathbf { x } } ^ { \mathrm { u } } ) ,\tag{8}
$$

where $\psi$ parameterizes how imputations are computed.

This formulation yields a unified objective:

$$
\operatorname* { m i n } _ { \theta , \psi } \mathbb { E } _ { ( \mathbf { x } ^ { \mathrm { o b s } } , y ) } \left[ \ell \big ( f _ { \theta } ( \mathbf { x } ^ { \mathrm { o b s } } , \mathcal { Z } _ { \psi } ( \cdot ) ) , y \big ) \right] + g ( \mathcal { T } _ { \psi } ( \cdot ) ) .\tag{9}
$$

Different choices of ψ recover different instantiations:

• Amortized imputation (Joint-AmortizedImpute): We parameterize the imputation operator as $\mathcal { T } _ { \phi } ,$ , with shared parameters ϕ that predict imputations from inputs. This can be interpreted as learning to approximate the optimal latent corrections δ in a single forward pass, enabling generalization to unseen data and reducing computational cost. The model is trained by minimizing:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { \mathrm { p e n } } \mathcal { L } _ { \mathrm { p e n } } + \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } , } \end{array}\tag{10}
$$

where $\lambda _ { \mathrm { p e n } } , \lambda _ { \mathrm { r e c } } \geq 0 , \mathcal { L } _ { \mathrm { p e n } }$ is a regularization term and ${ \mathcal { L } } _ { \mathrm { r e c } }$ is a reconstruction loss.

• Non-amortized imputation (Joint-LatentImpute): We directly optimize per-sample latent variables $\{ \pmb { \delta } _ { i } \} _ { i = 1 } ^ { n }$ . The objective reduces to Eq. 10 with $\lambda _ { \mathrm { r e c } } = 0$

Full methodological and implementation details of Joint-AmortizedImpute and Joint-LatentImpute are provided in Appendix A.7 and Appendix A.6, respectively.

## 4 Experimental Setup

We evaluate the proposed framework under two complementary settings reflecting different practical scenarios regarding the availability of missing values at inference time. In our experiments, we focus on classification tasks, therefore we use the Cross-Entropy (CE) loss. For more details on the classifier, refer to Appendix A.5.

Joint-AmortizedImpute: missingness at inference time. This constitutes our main evaluation setting. We consider three scenarios to assess robustness under distribution shifts in missingness:

• Robustness under train/test mismatch.

– Same mechanism, different missing rates: The missingness mechanism is fixed (either MCAR, MAR or MNAR), whereas the missing rate differs between training $m _ { r } ^ { \mathrm { t r a i n } } \in \{ 0 . 1 , 0 . 3 \}$ and testing $m _ { r } ^ { \mathrm { t e s t } } \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 \} : m _ { r } ^ { \mathrm { t e s t } } \neq m _ { r } ^ { \mathrm { t r a i n } }$

– Different mechanism, same missing rate: The missing rate is fixed $( m _ { r } ^ { \mathrm { t r a i n } } = m _ { r } ^ { \mathrm { t e s t } } \in$ {0.1, 0.5}), while the missingness mechanism varies between training and testing (MCAR, MAR or MNAR).

• Overall stability. Training and test data share the same missingness mechanism and rate. We evaluate six missignness mechanisms– MCAR; MAR and its sequential variant; MNAR, and its latent-factor and polarization variants– across five missing rates, $m _ { r } \in$ {0.1, 0.3, 0.5, 0.7, 0.8}.

Joint-LatentImpute: no missingness at inference time. This setting isolates the effect of taskaware imputation during training while reflecting scenarios where training data is incomplete but deployment data is clean. We evaluate across the same six missingness mechanisms (MCAR; MAR and its sequential variant; and MNAR with latent-factor and polarization variants) and five missing rates m<sub>r</sub> ∈ {0.1, 0.3, 0.5, 0.7, 0.8}.

Please, refer to Appendix A.3 for more details on data splits and simulated missingness mechanisms.

Baselines. We compare the proposed approach with the following imputation methods : mean/median for continuous variables, mode for categorical variables, K-NN using the mean of the nearest neighbors for numerical variables and mode imputation for categorical ones, MissForest and TabImpute. We also consider the complete data baseline, which provides an upper bound on performance and allows us to quantify the impact of missingness and imputation.

Reference imputation choice. We select the reference imputation xˆ<sup>u</sup> using a task-driven procedure: each candidate method is used to impute the training data, a downstream classifier is trained, and the method yielding the best validation performance is chosen. This selection is performed per dataset and missingness setting and provides a strong initialization for the proposed method, improving stability and focusing learning on task-relevant corrections.

Evaluation metrics. Imputation quality is measured using Root Mean Squared Error (RMSE) for numerical features and error rate for categorical features (Cat. Err.). To assess downstream utility, we train a classifier on the imputed data and report accuracy/F1-score. We report the performance gap relative to the complete-data baseline:

$$
\mathrm { G a p } = \mathrm { A c c u r a c y } _ { \mathrm { c o m p l e t e - d a t a } } - \mathrm { A c c u r a c y } _ { \mathrm { m e t h o d } } .\tag{11}
$$

Lower values indicate better performance, with a gap of zero corresponding to perfect recovery. We additionally measure performance recovery using the recovery ratio, defined as the ratio between the accuracy obtained under missing data and the accuracy achieved on the corresponding complete-data baseline:

$$
\mathrm { R R } = \frac { \mathrm { A c c u r a c y } _ { \mathrm { m e t h o d } } } { \mathrm { A c c u r a c y } _ { \mathrm { c o m p l e t e - d a t a } } } .\tag{12}
$$

A value of RR = 1 indicates full recovery of complete-data performance, while lower values reflect degradation due to missingness. Values greater than 1 indicate that the method outperforms the complete-data baseline. We define the hardness of a scenario based on the recovery ratio as:

$$
{ \mathrm { H a r d n e s s } } = 1 - { \mathrm { R R } } .\tag{13}
$$

We define four hardness levels: Easy (Hardness < 0.02), Mild $( 0 . 0 2 \leq \mathrm { H a r d n e s s } < 0 . 0 5 )$ , Hard (0.05 ≤ Hardness < 0.1), and Severe (Hardness ≥ 0.1).

Best classical method. For reporting purposes, we define the best classical method in each scenario as the imputation approach achieving the highest test accuracy. This definition is used only for post-hoc comparison and does not affect model selection, which is performed using validation performance.

Datasets and data preprocessing. We evaluate our approach on classification datasets from OpenML [35]. For experiments involving distribution shifts (cross-mechanism and varying missingrate), we use a set of 36 datasets with imbalance ratio below 10 to avoid severely skewed class distributions. In this setting, classification performance is evaluated using accuracy. To assess overall robustness and stability, including hardness analysis unrelated to distribution shifts, we extend the evaluation to a broader collection of 98 datasets. All methods are evaluated on identical data splits to ensure fair comparison. Numerical features are standardized using z-score normalization, where statistics are computed using only observed values from the training set. Further details on dataset characteristics and domains are provided in Appendix A.1.

Statistical analysis. To assess statistical significance across multiple datasets, we follow standard protocols for comparing learning algorithms. Methods are ranked per dataset and compared using the Friedman test, followed by the Nemenyi post-hoc test to identify significant differences in average rank. We visualize results using critical difference (CD) diagrams, where lower ranks indicate better performance and horizontal bars connect methods that are not significantly different. Additionally, we report Spearman rank correlations to analyze the relationship between reconstruction quality and downstream performance. All statistical tests are conducted at a significance level of $\alpha = 0 . 0 5$

## 5 Results

## 5.1 Robustness under difficult scenarios

## 5.1.1 Train/test mismatch

## • Different missing rates

As shown in Figure 1, the proposed Joint-AmortizedImpute consistently achieves the highest recovery ratio (0.8181), substantially outperforming all classical baselines (0.69–0.70). The CD diagram across all scenarios reveals that the joint model achieves the best average rank and is separated from the classical methods by statistically significant differences. The results in Table 1 reveal a clear dependence on the magnitude of the train–test missingness shift. When the shift is small or when the test missingness is comparable to (or lower than) the training missingness, Joint-AmortizedImpute underperforms the best classical baseline. In contrast, when the test missingness is substantially higher than the training missingness, the proposed approach consistently provides strong gains: for example, in the 0.1 → 0.5 and 0.3 → 0.7 settings, it achieves positive improvements exceeding 0.10 on average and wins in over 80% of cases.

<table><tr><td>Method</td><td>Recovery Ratio (↑)</td></tr><tr><td>Joint-AmortizedImpute (Ours)</td><td>0.8181</td></tr><tr><td>Column Median</td><td>0.7048</td></tr><tr><td>K-NN</td><td>0.7043</td></tr><tr><td>Column Mean</td><td>0.7038</td></tr><tr><td>MissForest</td><td>0.6915</td></tr><tr><td>TabImpute</td><td>0.6906</td></tr></table>

(a) Average recovery ratio relative to the com-  
plete data baseline.

![](images/2e6e69245dea3b9290c0bb951fa6622d859bd1573a39df2aa0a673fa62f46f05.jpg)  
(b) CD diagram.  
Figure 1: Comparison of imputation methods under varying missingness probabilities. The recovery ratio (left) shows that Joint-AmortizedImpute achieves the best overall performance, while the CD diagram (right) confirms statistically significant ranking differences.

Table 1: Performance of Joint imputation relative to the best classical method under train–test missingness shifts. We report mean accuracy difference, win rate, and degradation rate across scenarios.
<table><tr><td>Train → Test</td><td>Mean (↑)</td><td>Win (↑)</td><td>Degradation (↓)</td></tr><tr><td>0.1 → 0.3</td><td>-0.0151</td><td>25.00%</td><td>47.22%</td></tr><tr><td>0.1 → 0.5</td><td>0.1240</td><td>74.07%</td><td>21.30%</td></tr><tr><td>0.1 → 0.7</td><td>0.0873</td><td>66.67%</td><td>28.21%</td></tr><tr><td>0.3 → 0.1</td><td>-0.0154</td><td>26.39%</td><td>43.06%</td></tr><tr><td>0.3 → 0.5</td><td>0.1428</td><td>81.94%</td><td>13.89%</td></tr><tr><td>0.3 → 0.7</td><td>0.1063</td><td>82.05%</td><td>12.82%</td></tr></table>

## • Different missing mechanisms

The CD diagram (Figure 2, left) shows that Joint-AmortizedImpute achieves the best average rank across datasets compared to classical baselines, with statistically significant differences. This is reinforced by the recovery ratio results (Figure 2, right), where Joint-AmortizedImpute attains a substantially higher value (0.8381), with a clear margin over the second-best method (K-NN at 0.7220).

Moreover, Joint-AmortizedImpute consistently outperforms classical methods when evaluated under MNAR missingness, regardless of the training mechanism (Table 2). In contrast, when evaluated under MAR, the proposed approach exhibits negative mean differences and low win rates, indicating limited generalization under this setting. Interestingly, models trained under MNAR generalize well to MCAR, achieving high win rates and positive mean improvements.

The strong performance observed in MCAR/MAR → MNAR reflects that MNAR test settings are intrinsically more challenging, providing greater opportunity for task-aware imputation to improve performance. When training on MNAR and testing on MCAR, the problem becomes easier, as MCAR introduces only random missingness. On the other hand, MAR introduces structured missingness that depends on observed features, which can conflict with patterns learned under MNAR, where missingness depends on unobserved values, leading to a mismatch in how missingness relates to the predictive features. Similarly, training under MAR may encourage reliance on such structured missingness, which breaks under MCAR, resulting in degraded performance.

![](images/23632d0363750e6ab51e846fd2884f80a9804603e72cafffa636af30170740a3.jpg)  
(a) CD diagram (Accuracy)

<table><tr><td>Method</td><td>Recovery Ratio (↑)</td></tr><tr><td>Joint-AmortizedImpute (Ours)</td><td>0.8381</td></tr><tr><td>K-NN</td><td>0.7220</td></tr><tr><td>Column Mean</td><td>0.7178</td></tr><tr><td>Column Median</td><td>0.7162</td></tr><tr><td>TabImpute</td><td>0.7069</td></tr><tr><td>MissForest</td><td>0.7021</td></tr></table>

(b) Average recovery ratio relative to the complete data baseline.  
Figure 2: Comparison of imputation methods.

Table 2: Performance of Joint imputation relative to the best classical method under train–test missingness mechanism shifts. We report mean accuracy difference (Joint minus Best Classical), win rate, and degradation rate across scenarios.
<table><tr><td>Train → Test</td><td>Mean (↑)</td><td>Win (↑)</td><td>Degradation (↓)</td></tr><tr><td>MAR → MCAR</td><td>-0.0249</td><td>23.53%</td><td>55.29%</td></tr><tr><td>MAR → MNAR</td><td>0.1536</td><td>77.65%</td><td>18.82%</td></tr><tr><td>MCAR → MAR</td><td>-0.0134</td><td>24.32%</td><td>51.35%</td></tr><tr><td>MCAR → MNAR</td><td>0.1783</td><td>79.73%</td><td>13.51%</td></tr><tr><td>MNAR → MAR</td><td>-0.0267</td><td>23.46%</td><td>55.56%</td></tr><tr><td>MNAR → MCAR</td><td>0.1352</td><td>80.00%</td><td>15.00%</td></tr></table>

## 5.1.2 Difficult scenarios not due to train/test mismatch

We further analyze performance in intrinsically difficult missingness settings, defined by high hardness, i.e., scenarios where classical imputation methods fail to recover complete-data performance (see Section 5). These correspond to the “hard” and “severe” regimes in our evaluation.

Joint-AmortizedImpute When focusing on harder missingness settings, JointAmortized-Impute achieves the lowest rank in both accuracy and F1-score. Though in terms of classification accuracy it is not significantly different from Column Mean, for F1-score it is significantly better than the remaining methods.

Joint-LatentImpute. Figure 4 presents the CD diagrams restricted to hard and severe scenarios, where the proposed Joint-LatentImpute is the top-performing method across both accuracy and F1 score (statistically significant, Nemenyi post-hoc, all $p < 1 0 ^ { \bar { - } 6 } )$ . Moreover, we observe a weak but statistically significant negative correlation between improvement and the performance of the selected reference imputation (Spearman, $\rho = - 0 . 0 7 , p < 0 . 0 0 5 )$ .

![](images/231347d860d0ed74fc67f602cdde05aed8a725b3298aa96c7969f30fc44e3fca.jpg)  
(a) Accuracy

![](images/96347b29e8802943119854d7dba8a38db63641321d2868f195dd681c80d2cea5.jpg)  
(b) F1-score

Figure 3: Joint-AmortizedImpute: CD diagrams (scenario-level) for hard and severe scenarios. Lower ranks indicate better performance.  
![](images/a0337efeaa63ba08765708ded16b467613b690c3938d723f80ca75eeef546d91.jpg)  
(a) Accuracy

![](images/60ab28d37c0453528ef5d26fe551c62852d20ef2e311182e7a98f787012be8e6.jpg)  
(b) F1 score  
Figure 4: Joint-LatentImpute: CD diagrams (scenario-level) for hard and severe scenarios.

## 5.2 Reconstruction error & its relationship to downstream performance

Table 3 reports the imputation quality ranking for both Joint-LatentImpute and Joint-AmortizedImpute across numerical (RMSE) and categorical features.

<table><tr><td></td><td colspan="2">Joint-LatentImpute</td><td colspan="2">Joint-AmortizedImpute</td></tr><tr><td>Method</td><td>RMSE Rank (↓)</td><td>Cat. Err. Rank (↓)</td><td>RMSE Rank (↓)</td><td>Cat. Err. Rank(↓)</td></tr><tr><td>Joint (Ours)</td><td>2.96</td><td>1.8</td><td>2.80</td><td>2.01</td></tr><tr><td>Column Mean</td><td>3.11</td><td>1.6</td><td>2.62</td><td>1.77</td></tr><tr><td>K-NN</td><td>3.39</td><td>1.6</td><td>3.15</td><td>一</td></tr><tr><td>Column Median</td><td>4.64</td><td>1.6</td><td>3.85</td><td></td></tr><tr><td>MissForest</td><td>3.02</td><td>2.0</td><td>4.82</td><td>3.12</td></tr><tr><td>TabImpute</td><td>3.87</td><td>3.6</td><td>3.74</td><td>3.65</td></tr></table>

Table 3: Imputation quality ranking (lower is better).

Joint-LatentImpute. We analyze the relationship between imputation quality and downstream classification performance using Spearman rank correlation between RMSE ranks and accuracy ranks across all scenarios. We observe a statistically significant positive correlation (Spearman, $\rho = 0 . 1 6 , p < 0 . 0 0 1 \rangle$ , which indicates that methods achieving better reconstruction quality tend to also perform better in downstream prediction. However, this correlation is weak, suggesting that reconstruction error alone does not fully explain predictive performance. For example, MissForest achieves competitive performance in terms of numerical reconstruction error, ranking second overall. However, this advantage does not translate to downstream classification performance, where simpler methods such as mean and K-NN imputation often perform better.

Joint-AmortizedImpute. We observe a statistically significant positive correlation between RMSE ranks and accuracy ranks across all scenarios (Spearman, $\rho = 0 . 3 2 , p < 0 . 0 0 1 )$ . This reinforces that, though methods achieving better reconstruction quality tend to perform better in downstream prediction, reconstruction quality alone does not fully explain predictive performance.

## 5.3 Overall stability assessment

Joint-AmortizedImpute. Table 4 shows that the proposed method achieves the highest average recovery ratio with respect to the complete-data baseline, highlighting its robustness in mitigating the impact of missing data compared to the benchmark imputation methods. However, on average,

Joint-AmortizedImpute leads to only marginal changes relative to the selected reference imputation method, with small performance degradations (Table 5). The CD diagrams in Figures 5b and 5a show that, for both accuracy and F1-score, the proposed approach is not significantly different from the best ranking approaches, Column Mean and Column Median (Nemenyi post-hoc, $p = 0 . 3 8$ and $p = 0 . 9 6$ for accuracy; $p = 0 . 1 1$ for Column Mean in F1).
<table><tr><td>Method</td><td>Recovery Ratio (↑)</td></tr><tr><td>Joint-AmortizedImpute (Ours)</td><td>0.8932</td></tr><tr><td>Column Mean</td><td>0.8902</td></tr><tr><td>K-NN</td><td>0.8875</td></tr><tr><td>Column Median</td><td>0.8863</td></tr><tr><td>TabImpute</td><td>0.8785</td></tr><tr><td>MissForest</td><td>0.8593</td></tr></table>

Table 4: Mean recovery ratio per method (higher is better).

<table><tr><td></td><td>Accuracy (↑)</td><td>F1 (↑)</td></tr><tr><td>Mean  $\Delta$ </td><td>-0.0039</td><td>-0.0022</td></tr><tr><td>Median  $\Delta$ </td><td>-0.0000</td><td>-0.0003</td></tr><tr><td>Win Rate</td><td>41.5%</td><td>41.9%</td></tr><tr><td>Degradation  $> 5 \%$ </td><td>3.8%</td><td>6.0%</td></tr><tr><td>Degradation  $> 1 0 \%$ </td><td>0.85%</td><td>1.30%</td></tr></table>

Table 5: Joint-AmortizedImpute vs Reference performance. Positive $\Delta$ indicates improvement over the selected reference imputation method.

![](images/2f296179c3d5909d88b3828785330721667f6870ec9c473c9ae2a0d9c4dd4c8e.jpg)  
(a) Accuracy.

![](images/7611dddf3d9f19b325bb56334bc2806e32fa7cd07b2b8e31b3f0d35f1b68880e.jpg)  
(b) F1-score.  
Figure 5: Overall: CD diagrams comparing imputation methods based on performance rankings.

Joint-LatentImpute. Across all scenarios, Table 6 shows that the proposed joint imputation achieves the highest average recovery performance when averaged across all scenarios. Moreover, Joint-LatentImpute improves performance compared to the selected reference imputation in approximately 45% of the cases, while it degrades performance by more than 5% and 10% in, respectively, 2% and 0.3% of the scenarios, suggesting that the proposed method is generally stable and rarely introduces large performance drops (Table 7).

Table 6: Joint-LatentImpute (Ours): recovery of complete data performance.
<table><tr><td>Method</td><td>Recovery Ratio (↑)</td></tr><tr><td>Joint-LatentImpute (Ours)</td><td>0.9641</td></tr><tr><td>K-NN</td><td>0.9595</td></tr><tr><td>Mean</td><td>0.9580</td></tr><tr><td>TabImpute</td><td>0.9579</td></tr><tr><td>MissForest</td><td>0.9577</td></tr><tr><td>Median</td><td>0.9570</td></tr></table>

Table 7: Joint-LatentImpute (Ours) performance difference over the selected reference (Accuracy / F1).
<table><tr><td>Metric</td><td>Acc (↑)</td><td>F1 (↑)</td></tr><tr><td>Mean ∆</td><td>-0.0007</td><td>-0.0022</td></tr><tr><td>Win rate</td><td>0.4548</td><td>0.4051</td></tr><tr><td>Degradation  $> 5 \%$ </td><td>0.0189</td><td>0.0522</td></tr><tr><td>Degradation  $> 1 0 \%$ </td><td>0.0025</td><td>0.0124</td></tr></table>

## 6 Conclusions, Limitations and Future Work

Conclusions. This work studies the role of task-aware imputation in supervised learning with missing data, with a focus on understanding when it improves predictive performance. Our results show that task-aware imputation does not uniformly improve predictive performance, but instead provides benefits primarily in challenging regimes, particularly under distribution shift and in MNAR settings, where classical methods fail to recover complete-data performance. We further find that reconstruction quality is only weakly correlated with downstream performance (Spearman $\rho \approx 0 . 1 6 \ – 0 . 3 2 )$ suggesting that optimizing imputation for reconstruction alone is insufficient. These results indicate that imputation should be treated as a task-dependent problem, rather than a preprocessing step focused solely on data fidelity.

Limitations and Future Work. The proposed Joint-AmortizedImpute approach uses fixed hyperparameters across all datasets. While this ensures consistent evaluation, it may limit adaptability; dataset-specific tuning or automated optimization could improve performance and robustness. The experimental comparison is also limited to a small set of baseline methods, and extending the benchmark would enable a more comprehensive evaluation. Furthermore, experiments focus exclusively on tabular classification, leaving generalization to other settings (e.g., regression or structured data such as time series) as an open question. Finally, the method does not explicitly model the missingness mechanism, which may reduce effectiveness when missingness is structured or informative. Incorporating mechanism-aware components is a promising direction for future work.

Broader impacts. Improved handling of missing data can enhance the reliability of machine learning systems in real-world applications such as healthcare, finance, and scientific analysis. By aligning imputation with predictive objectives, the proposed approach may support better decisionmaking. However, task-aware imputation may also amplify biases in the observed data, leading to unfair or misleading outcomes, particularly in sensitive domains. Future work should explore fairness-aware imputation and develop safeguards to mitigate these risks.

## References

[1] Kristian Kleinke, Jost Reinecke, Daniel Salfrán, and Martin Spiess. Missing Data Mechanism and Ignorability, pages 23–52. Springer International Publishing, Cham, 2020.

[2] Roderick J. A. Little and Donald B. Rubin. Statistical Analysis with Missing Data. Wiley Series in Probability and Statistics. Wiley, 3 edition, 2019.

[3] Helen Zhou, Sivaraman Balakrishnan, and Zachary C. Lipton. Domain adaptation under missingness shift. In Francisco J. R. Ruiz, Jennifer G. Dy, and Jan-Willem van de Meent, editors, International Conference on Artificial Intelligence and Statistics, 25-27 April 2023, Palau de Congressos, Valencia, Spain, Proceedings of Machine Learning Research, pages 9577–9606. PMLR, 2023.

[4] Yichen Zhu and Bo Jiang. Stablemiss+: Prediction with incomplete data under agnostic mask distribution shift. In ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 7190–7194, 2024.

[5] Tianjing Li, Susan Hutfless, Daniel O. Scharfstein, Michael J. Daniels, Joseph W. Hogan, Roderick J.A. Little, Jason A. Roy, Andrew H. Law, and Kay Dickersin. Standards should be applied in the prevention and handling of missing data for patient-centered outcomes research: a systematic review and expert consensus. Journal of Clinical Epidemiology, 67(1):15–32, 2014.

[6] Donald B. Rubin. Inference and missing data. Biometrika, 63(3):581–592, 12 1976.

[7] Gustavo E. A. P. A. Batista and Maria Carolina Monard. An analysis of four missing data treatment methods for supervised learning. Applied Artificial Intelligence, 17(5-6):519–533, 2003.

[8] Daniel J. Stekhoven and Peter Bühlmann. Missforest—non-parametric missing value imputation for mixed-type data. Bioinformatics, 28(1):112–118, 10 2011.

[9] Stef van Buuren and Karin Groothuis-Oudshoorn. mice: Multivariate imputation by chained equations in r. Journal ofStatistical Software, 45(3):1–67, 2011.

[10] E. Albu, S. Gao, L. Wynants, and B. Van Calster. missforestpredict—missing data imputation for prediction settings. PLOS ONE, 20(11):e0334125, 2025.

[11] Daniel Jarrett, Bogdan Cebere, Tennison Liu, Alicia Curth, and Mihaela van der Schaar. Hyperimpute: Generalized iterative imputation with automatic model selection. In Proceedings of the 39th International Conference on Machine Learning (ICML), volume 162, pages 9916– 9937, 2022.

[12] Emmanuel J. Candès and Benjamin Recht. Exact matrix completion via convex optimization. Foundations ofComputational Mathematics, 9(6):717–772, 2009.

[13] Rahul Mazumder, Trevor Hastie, and Robert Tibshirani. Spectral regularization algorithms for learning large incomplete matrices. Journal of Machine Learning Research, 11:2287–2322, 2010.

[14] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Hoo, Robin Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637:319–326, 01 2025.

[15] Jacob Feitelberg, Dwaipayan Saha, Kyuseong Choi, Zaid Ahmad, Anish Agarwal, and Raaz Dwivedi. Tabimpute: Accurate and fast zero-shot missing-data imputation with a pre-trained transformer. arXiv preprint arXiv:2510.02625, 2025.

[16] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Revisiting deep learning models for tabular data. In Proceedings ofthe 35th International Conference on Neural Information Processing Systems, NIPS ’21, Red Hook, NY, USA, 2021. Curran Associates Inc.

[17] Boris Muzellec, Julie Josse, Claire Boyer, and Marco Cuturi. Missing data imputation using optimal transport. In Proceedings of the 37th International Conference on Machine Learning, ICML’20. JMLR.org, 2020.

[18] Tianyu Du, Luca Melis, and Ting Wang. Remasker: Imputing tabular data with masked autoencoding. In International Conference on Learning Representations (ICLR), 2024.

[19] Adriana Fonseca Costa, Miriam Seoane Santos, Jastin Pompeu Soares, and Pedro Henriques Abreu. Missing data imputation via denoising autoencoders: The untold story. In Wouter Duivesteijn, Arno Siebes, and Antti Ukkonen, editors, Advances in Intelligent Data Analysis XVII, pages 87–98, Cham, 2018. Springer International Publishing.

[20] Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings ofthe 25th International Conference on Machine Learning, ICML ’08, page 1096–1103, New York, NY, USA, 2008. Association for Computing Machinery.

[21] Shenfen Kuang, Yewen Huang, and Jie Song. Unsupervised data imputation with multiple importance sampling variational autoencoders. Scientific Reports, 15:3409, 01 2025.

[22] Pierre-Alexandre Mattei and Jes Frellsen. Miwae: Deep generative modelling and imputation of incomplete data sets. In Proceedings of the 36th International Conference on Machine Learning (ICML), pages 4413–4423, 2019.

[23] Alejandro Nazábal, Pablo M. Olmos, Zoubin Ghahramani, and Isabel Valera. Handling incomplete heterogeneous data using vaes. Pattern Recognition, 107:107501, 2020.

[24] Jinsung Yoon, James Jordon, and Mihaela van der Schaar. Gain: Missing data imputation using generative adversarial nets. In Proceedings ofthe 35th International Conference on Machine Learning (ICML), pages 5689–5698, 2018.

[25] Zhichao Chen, Haoxuan Li, Fangyikang Wang, Odin Zhang, Hu Xu, Xiaoyu Jiang, Zhihuan Song, and Hao Wang. Rethinking the diffusion models for missing data imputation: a gradient flow perspective. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA, 2024. Curran Associates Inc.

[26] Alexia Jolicoeur-Martineau, Kilian Fatras, and Tal Kachman. Generating and imputing tabular data via diffusion and flow-based gradient-boosted trees, 2023.

[27] Trent Kyono, Yao Zhang, Alexis Bellot, and Mihaela van der Schaar. Miracle: Causallyaware imputation via learning missing data mechanisms. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

[28] John Lee, Nicholas P. Bertrand, and Christopher J. Rozell. Unbalanced Optimal Transport Regularization for Imaging Problems. IEEE Transactions on Computational Imaging, 6:1219– 1232, 2020.

[29] Gabriel Peyré and Marco Cuturi. Computational Optimal Transport: With Applications to Data Science. Foundations and Trends® in Machine Learning, 11(5-6):355–607, 2019.

[30] Tolou Shadbahr, Michael Roberts, Jan Stanczuk, Julian Gilbey, Philip Teare, Sören Dittmer, Matthew Thorpe, Ramon Viñas Torné, Evis Sala, Pietro Lió, et al. The impact of imputation quality on machine learning classifiers for datasets with missing values. Communications Medicine, 3:139, 2023.

[31] Xiaoou Ding, Yanshuo Liu, Zhounan Chen, Hongzhi Wang, Chen Wang, and Jianmin Wang. Tarimpute: Task-aware auto-recommender system for missing value imputation algorithms with clustering case studies. Proc. VLDB Endow., 18(12):5343–5346, August 2025.

[32] Dimitris Bertsimas, Colin Pawlowski, and Ying Daisy Zhuo. From predictive methods to missing data imputation: An optimization approach. Journal of Machine Learning Research, 18(196):1–39, 2018.

[33] Jiaxuan You, Xiaobai Ma, Daisy Yi Ding, Mykel Kochenderfer, and Jure Leskovec. Handling missing data with graph representation learning. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc.

[34] Katrina Chen, Xiuqin Liang, Zheng Ma, and Zhibin Zhang. Gedi: A graph-based end-to-end data imputation framework. 2023 IEEE 35th International Conference on Tools with Artificial Intelligence (ICTAI), pages 723–730, 2023.

[35] Matthias Feurer, Jan N. van Rijn, Arlind Kadra, Pieter Gijsbers, Neeratyoy Mallik, Sahithya Ravi, Andreas Müller, Joaquin Vanschoren, and Frank Hutter. Openml-python: an extensible python api for openml. Journal ofMachine Learning Research, 22(100):1–5, 2021.

[36] Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, Geeta Chauhan, Anjali Chourdia, Will Constable, Alban Desmaison, Zachary DeVito, Elias Ellison, Will Feng, Jiong Gong, Michael Gschwind, Brian Hirsh, Sherlock Huang, Kshiteej Kalambarkar, Laurent Kirsch, Michael Lazos, Mario Lezcano, Yanbo Liang, Jason Liang, Yinghai Lu, CK Luk, Bert Maher, Yunjie Pan, Christian Puhrsch, Matthias Reso, Mark Saroufim, Marcos Yukio Siraichi, Helen Suk, Michael Suo, Phil Tillet, Eikan Wang, Xiaodong Wang, William Wen, Shunting Zhang, Xu Zhao, Keren Zhou, Richard Zou, Ajit Mathews, Gregory Chanan, Peng Wu, and Soumith Chintala. Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In Proceedings ofthe 29th ACM International Conference on Architectural Supportfor Programming Languages and Operating Systems (ASPLOS), 2024.

## A Experimental setup

All experiments were conducted using Python 3.10.19. The implementation relies on PyTorch [36], scikit-learn, NumPy and pandas. Tabular foundation models are incorporated through the tabimpute library (downloaded from its public GitHub repository). Evaluation metrics such as accuracy and F1-score are computed using scikit-learn.

Experiments were conducted on a machine equipped with an NVIDIA GPU with CUDA support.

## A.1 Datasets and data preprocessing

We restrict our evaluation to OpenML classification datasets without missing values. This allows us to generate missingness in a fully controlled manner, ensuring that the ground-truth complete data is always available.

Tables 8 and 9 summarize, respectively the domains and characteristics of the 98 datasets used in our main evaluation. The collection spans a wide range of domains, dataset sizes, feature dimensionalities, and class cardinalities, ensuring a diverse and representative benchmark. In particular, datasets include both low- and high-dimensional settings, varying proportions of categorical features, and both binary and multi-class classification tasks.

## List of datasets:

<table><tr><td>Property</td><td>Range</td></tr><tr><td>#Datasets</td><td>98</td></tr><tr><td>Samples</td><td>350-7000</td></tr><tr><td>Features</td><td>4-49</td></tr><tr><td># Categorical features</td><td>0-23</td></tr><tr><td>Classes</td><td>2-40</td></tr></table>

Table 8: Dataset characteristics.

<table><tr><td>Domain</td><td>#Datasets</td></tr><tr><td>Synthetic</td><td>19</td></tr><tr><td>Finance</td><td>15</td></tr><tr><td>Medical / Healthcare</td><td>14</td></tr><tr><td>Computer / IT / Security</td><td>14</td></tr><tr><td>Social / Behavioral</td><td>11</td></tr><tr><td>Physical / Environmental</td><td>11</td></tr><tr><td>Biology / Chemistry Industrial / Engineering</td><td>8 6</td></tr></table>

Table 9: Dataset domains (manual annotation).

• 1StudentPerfromance

• Accidents\_Prediction\_Dataset

• Australian\_seed\_1\_nrows\_2000\_nclasses\_10\_ncols\_100\_stratify\_True

• Bank-Note-Authentication-UCI

• Bank\_Customer\_Churn

• CanadaFiresD

• Creditability-German-Credit-Data

• Heart-Disease-Dataset-(Comprehensive)

• Intersectional-Bias-Assessment-(Training-Data)

• JapaneseVowels

• LED-display-domain-7digit

• LEV

• Mobile\_Price

• Multiclass\_Classification\_for\_Corporate\_Credit\_Ratings

• Phishing\_Legitimate\_full

• PieChart2

• PizzaCutter1

• Pumpkin\_Seeds

• SWD

• Satellite

• Titanic

• VulNoneVul

• ada

• air-quality-and-pollution-assessment

• analcatdata\_supreme

• arsenic-female-bladder

• aztrees3

• balance-scale

• bank-marketing\_seed\_1\_nrows\_2000\_nclasses\_10\_ncols\_100\_stratify\_True

• bank8FM

• blood-transfusion-service-center

• boston

• bwin\_amlb

• cardiotocography

• churn

• climate-model-simulation-crashes

• collins

• compas-two-years

• covertype\_seed\_0\_nrows\_2000\_nclasses\_10\_ncols\_100\_stratify\_True

• cpu\_small

• credit-approval

• credit-g

• delta\_ailerons

• diabetes

• dis

• doa\_bwin

• electrical\_grid\_stability\_simulated\_classification

• eucalyptus

• eye\_movements

• glioma\_grading\_clinical\_and\_mutation\_features

• hazelnut-spread-contaminant-detection

• heloc

• hepatitis\_c\_virus\_hcv\_for\_egyptian\_patients

• house\_16H\_seed\_0\_nrows\_2000\_nclasses\_10\_ncols\_100\_stratify\_True

• ibm-employee-performance

• ilpd

• jungle\_chess\_2pcs\_raw\_endgame\_complete\_seed\_4\_nrows\_2000\_nclasses\_10\_ncols\_100\_stratify\_True

• kc2

• kdd\_ipums\_la\_97-small

• mabbob\_ela\_as\_5d\_classify

• maternal\_health\_risk

• mfeat-zernike

• numerai28.6\_seed\_0\_nrows\_2000\_nclasses\_10\_ncols\_100\_stratify\_True

• oil\_spill

• page-blocks

• phoneme

• pollen

• puma8NH

• qsar-biodeg

• red\_wine

• ringnorm

• rl

• rmftsa\_ladata

• road-safety\_seed\_0\_nrows\_2000\_nclasses\_10\_ncols\_100\_stratify\_True

• segment

• seismic-bumps

• shrutime

• sick

• socmob

• steel-plates-fault

• stock

• strikes

• student\_lifestyle\_dataset

• sylvine\_seed\_1\_nrows\_2000\_nclasses\_10\_ncols\_100\_stratify\_True

• texture

• thyroid-allbp

• thyroid-allhyper

• tokyo1

• twonorm

• vehicle

• visualizing\_soil

• vowel

• waveform

• wdbc

• wilt

• wind

• wine-quality-red

• yeast

<table><tr><td>Metric</td><td>Range</td></tr><tr><td># Datasets</td><td>36</td></tr><tr><td>Samples</td><td>350-7000</td></tr><tr><td>Features</td><td>4-48</td></tr><tr><td># Categorical features</td><td>0-15</td></tr><tr><td>Classes</td><td>2-10</td></tr></table>

Table 10: Dataset characteristics.

<table><tr><td>Domain</td><td># Datasets</td></tr><tr><td>Synthetic</td><td>6</td></tr><tr><td>Finance</td><td>11</td></tr><tr><td>Medical / Healthcare</td><td>5</td></tr><tr><td>Computer / IT / Security</td><td>4</td></tr><tr><td>Social / Behavioral</td><td>4</td></tr><tr><td>Physical / Environmental</td><td>4</td></tr><tr><td>Biology / Chemistry Industrial / Engineering</td><td>1 1</td></tr></table>

Table 11: Dataset domains (manual annotation).

For experiments involving distribution shifts, we consider a set of 36 datasets, whose characteristics and domains are reported in Tables 10 and 11 . These datasets are selected to ensure stable evaluation under more challenging settings, such as cross-mechanism and varying missing-rate scenarios, while maintaining diversity across domains.

## List of datasets:

## A.2 Datasets with reported win-rate results

• delta\_ailerons

• compass\_seed\_1\_nrows\_2000\_nclasses\_10\_ncols\_100\_stratify\_True

• electrical\_grid\_stability\_simulated\_classification

• cmc

• Mobile\_Price

• covertype\_seed\_0\_nrows\_2000\_nclasses\_10\_ncols\_100\_stratify\_True

• cpu\_small

• diabetes\_risk\_prediction\_dataset

• compas-two-years

• balance-scale

• LED-display-domain-7digit

• Bank\_Customer\_Churn

• Multiclass\_Classification\_for\_Corporate\_Credit\_Ratings

• JapaneseVowels

• Pumpkin\_Seeds

• Titanic

• analcatdata\_supreme

• Phishing\_Legitimate\_full

• 1StudentPerfromance

• Australian\_seed\_1\_nrows\_2000\_nclasses\_10\_ncols\_100\_stratify\_True

• Bank-Note-Authentication-UCI

• credit

• diabetes

• credit-approval

• Heart-Disease-Dataset-(Comprehensive)

• Intersectional-Bias-Assessment-(Training-Data)

• KDDCup09\_upselling

• bank-marketing\_seed\_1\_nrows\_2000\_nclasses\_10\_ncols\_100\_stratify\_True

• ada

• bank8FM

• CanadaFiresD

• VulNoneVul

• blood-transfusion-service-center

• boston\_corrected

• credit-g

• bwin\_amlb

## A.3 Data splits and missingness

Each dataset is split into training (70%), validation (15%), and test (15%) sets using stratified sampling with respect to the class labels. Splits are generated using a fixed random seed (42) to ensure reproducibility. Categorical variables are encoded as integer category codes prior to missingness generation. Missing values are introduced after splitting the data, independently for the training, validation and test sets.

Other than the classical MCAR, MAR, and MNAR settings, we consider three additional missingness mechanisms capturing different dependency structures. These mechanisms are implemented using the TabImpute framework [15], which provides configurable generators for structured missingness patterns.

MAR Sequential is generated using an ϵ-greedy bandit process, where missingness depends on previously observed values. MNAR latent factor missingness is driven by a low-rank latent representation of the data, making missingness depend on unobserved variables. In particular, the latent rank is randomly sampled between 1 and 5, introducing variability in the underlying dependency structure. MNAR Polarization missingness preferentially removes values in the lower tail of the distribution (below the 25th percentile), simulating censoring effects.

## A.4 Benchmark imputation methods

For each dataset and missingness configuration, we load both the fully observed data and the corresponding corrupted version with missing values. A binary mask indicating observed entries is also provided. Numerical features are standardized using a StandardScaler fitted on the training data and applied to all splits.

We evaluate five imputation methods: mean, median, K-NN, MissForest and TabImpute. Numerical and categorical features are handled separately when required, with categorical variables typically imputed using the mode. If a categorical feature contains only missing values, a default value (0) is assigned.

Numerical features are standardized using a StandardScaler fitted on the training data with missing values. The same transformation is applied to validation and test sets.

Categorical variables are represented as integer-encoded categories during imputation. After imputation, all features are transformed using one-hot encoding before being passed to the classifier.

Each imputation method is applied independently to the training, validation, and test sets. For methods requiring fitting, the imputer is trained on the training data and then applied to validation and test sets.

Specifically:

• Mean and median imputation are computed exclusively on the training data. For numerical features, the mean or median of each feature is calculated using the training set and used to fill missing values. For categorical features, the mode is used. The same statistics are then consistently applied to validation and test sets through feature encoding alignment.

• K-NN imputation (implemented with sklearn.impute.KNNImputer) is fitted on the numerical features of the training data only. Missing numerical values are imputed based on nearest neighbors in the training set with K = 5. Categorical features are imputed separately using the training-set mode.

• MissForest (implemented via the missforest package https://github.com/ yuenshingyan/MissForest) is trained exclusively on the training data. A random forest model is iteratively fitted to predict missing values using observed features, and the trained imputer is then used to reconstruct the training data.

• TabImpute (using tabimpute.interface.TabImputeCategorical [15]) is applied independently to each data split. The same hyperparameters are used across splits, but the model does not reuse information from the training set when imputing validation or test data. TabImpute processes data in chunks, with a maximum of 64 rows per chunk and up to 4 chunks per batch.

After imputation, categorical variables are one-hot encoded, and a classifier is trained on the imputed training data.

For each dataset and missingness configuration, the imputation method yielding the highest validation accuracy is selected. Imputation quality is assessed by comparing imputed values to ground truth using a masked error metric computed only on missing entries. Imputation time is recorded to assess computational efficiency. For reproducibility, results and best-performing imputations are stored for each dataset, mechanism, and missingness level.

## A.5 Classifier

For all experiments, we use a simple feedforward neural network to isolate the effect of the proposed imputation method. The classifier consists of a single hidden layer with ReLU activation:

$$
f _ { \theta } ( x ) = W _ { 2 } \sigma ( W _ { 1 } x + b _ { 1 } ) + b _ { 2 } ,\tag{14}
$$

where $\sigma ( \cdot )$ denotes the ReLU activation.

The hidden dimension is fixed to $h = 3 2$ across all experiments. The classifier is intentionally kept lightweight to ensure that performance differences are primarily attributable to the imputation strategy rather than model capacity.

## A.6 Joint-LatentImpute

For numerical features, additive offsets are learned for missing entries. For categorical features, missing values are represented as learnable logits over category distributions. The model is initialized with reference imputations obtained from the best-performing baseline method (selected via validation). These reference values serve as anchors, and the model learns deviations from them.

Let M denote the set of missing entries. A reference imputation $Z ^ { r e f }$ is computed beforehand using the best-performing baseline method, and is used to initialize missing values.

Let $x _ { i j }$ denote a missing numerical entry. It is parameterized as:

$$
x _ { i j } = z _ { i j } ^ { r e f } + \alpha \cdot \delta _ { i j } ,\tag{15}
$$

where $\delta _ { i j }$ is a learnable parameter and $\alpha = 0 . 1$ . Observed entries remain unchanged throughout training.

Each categorical variable is represented using a smoothed one-hot encoding. For missing entries, logits are defined as:

$$
\ell _ { i j } = \log p _ { i j } ^ { r e f } + \alpha \cdot \delta _ { i j } ,\tag{16}
$$

where $p _ { i j } ^ { r e f }$ is the reference categorical distribution.

For missing entries, perturbations $\delta _ { i j }$ are initialized with small Gaussian noise to avoid degenerate initialization and encourage exploration of the imputation space.

Sampling is performed using the Gumbel-Softmax relaxation:

$$
\begin{array} { r } { \tilde { x } _ { i j } = \mathrm { G u m b e l S o f t m a x } ( \ell _ { i j } , \tau ) , } \end{array}\tag{17}
$$

which enables differentiable sampling of discrete variables.

The model is trained by minimizing:

$$
\mathcal { L } = \mathcal { L } _ { C E } + \lambda _ { p e n } \mathcal { L } _ { p e n } = \mathcal { L } _ { C E } + \lambda _ { n u m } \mathcal { L } _ { n u m } + \lambda _ { c a t } \mathcal { L } _ { c a t } ,\tag{18}
$$

where $\lambda _ { n u m } = 0 . 1$ and $\lambda _ { c a t } = 0 . 5$

• Classification Loss

$$
\mathcal { L } _ { C E } = - \sum _ { i } \log p ( y _ { i } \mid x _ { i } )\tag{19}
$$

• Numerical Regularization

$$
\mathcal { L } _ { n u m } = \frac { 1 } { | \mathcal { M } _ { n u m } | } \sum _ { ( i , j ) \in \mathcal { M } _ { n u m } } \delta _ { i j } ^ { 2 }\tag{20}
$$

• Categorical Regularization

$$
\mathcal { L } _ { c a t } = \frac { 1 } { \left| \mathcal { M } _ { c a t } \right| } \sum _ { ( i , j ) \in \mathcal { M } _ { c a t } } \mathrm { K L } \big ( p _ { \delta _ { i j } } \parallel p _ { i j } ^ { r e f } \big )\tag{21}
$$

These regularization terms constrain imputations to remain close to the reference solution.

Training is performed using mini-batch stochastic gradient descent with the Adam optimizer. Two optimizers are used:

• one for model parameters $\theta ,$

• one for latent variables $\Delta .$ , with a learning rate reduced by a factor of 10.

A temperature annealing schedule is applied:

$$
\tau _ { t } = \operatorname* { m a x } ( \tau _ { \mathrm { m i n } } , \tau _ { 0 } \cdot 0 . 9 9 5 ^ { t } ) ,\tag{22}
$$

with $\tau _ { 0 } = 1 . 0$ and $\tau _ { \mathrm { m i n } } = 0 . 2$

Training proceeds for a maximum of 1000 epochs with batch size 64.

To improve robustness, the following mechanisms are used:

• Early stopping based on validation accuracy (patience = 50);

• Gradient clipping with maximum norm 5.

After training, missing values are reconstructed as follows:

• Numerical

$$
x _ { i j } = z _ { i j } ^ { r e f } + \alpha \cdot \delta _ { i j }\tag{23}
$$

• Categorical

$$
\hat { c } _ { i j } = \arg \operatorname* { m a x } p _ { \delta _ { i j } }\tag{24}
$$

Implementation details. Model parameters are initialized using Kaiming uniform initialization for linear layers. Missing numerical entries are initialized from the reference values with additive Gaussian noise $\mathcal { N } ( 0 , 0 . 1 ^ { \frac { \ d } { 2 } } )$ . For categorical features, missing values are initialized by sampling from a noisy version of the reference distribution, obtained via a softmax over perturbed log-probabilities. Observed categorical values are encoded as smoothed one-hot vectors. Optimization is performed using Adam with separate learning rates for model parameters and latent variables: respectively, $1 \times \bar { 1 } 0 ^ { - 3 } \mathrm { ~ a n d ~ } 1 \times 1 \bar { 0 } ^ { - 4 }$ . Gradients are clipped to a maximum norm of 5 to ensure stability. Early stopping is applied with a patience of 50 iterations. Mini-batch training is used with batch size 64. The Gumbel-Softmax temperature is annealed exponentially during training from 1.0 to a minimum of 0.2. All experiments are conducted using a fixed random seed (42) for reproducibility.

Number of parameters. Let $P _ { \mathrm { c l a s s i f i e r } }$ be the number of parameters of the downstream classification model, $d _ { \mathrm { n u m } }$ the number of missing numerical entries, and $d _ { \mathrm { c a t } }$ the number of latent parameters associated with categorical missing entries. The total number of parameters is given by

$$
\# \mathrm { P a r a m e t e r s } = P _ { \mathrm { c l a s s i f i e r } } + d _ { \mathrm { n u m } } + d _ { \mathrm { c a t } } .\tag{25}
$$

Computational Complexity. Let n be the number of samples, d the input dimensionality, h the hidden dimension of the classifier, and T the number of training iterations. In this approach, missing values are treated as free optimization variables, resulting in $\mathcal { \bar { O } } ( | \mathcal { M } | )$ additional parameters, where M denotes the set of missing entries. Each training iteration requires a forward and backward pass through the classifier, with complexity $\mathcal { O } ( d h + \bar { h ^ { 2 } } )$ per sample, as well as updates to the latent variables, which incur an additional cost proportional to $| { \mathcal { M } } |$

The overall training complexity is therefore:

$$
\mathcal { O } \left( T \cdot \left( \frac { n } { b } \cdot ( d h + h ^ { 2 } ) + \vert \mathcal { M } \vert \right) \right) ,\tag{26}
$$

which becomes $O ( T \cdot n d )$ in the worst case. The space complexity is $O ( n d + | { \mathcal { M } } | )$ , dominated by storing the dataset and the latent variables.

## A.7 Joint-AmortizedImpute

Given an input with missing values $X$ , a mask $M \in \{ 0 , 1 \} ^ { d }$ indicating observed entries, and a reference imputation $Z _ { \mathrm { r e f } } .$ , the imputation network produces a completed representation:

$$
\hat { X } = M \odot X + ( 1 - M ) \odot \left( Z _ { \mathrm { r e f } } + \alpha \cdot \Delta ( X , M , Z _ { \mathrm { r e f } } ) \right) ,
$$

where $\Delta ( \cdot )$ is a learned residual correction and α is a fixed scaling factor. This formulation can be interpreted as learning task-aware deviations from a strong reference imputation, rather than predicting missing values from scratch.

Reference values $Z _ { \mathrm { r e f } }$ are obtained from the best-performing baseline imputation method (selected via validation), and serve as strong initial estimates that the model refines.

Numerical features are used directly after standardization. Categorical variables are encoded as onehot vectors. Missing categorical values are initialized using reference distributions with added noise and refined during training using a Gumbel-Softmax relaxation to enable differentiable sampling.

Cross-entropy loss is applied to the classifier predictions. For naturally missing values, the imputed outputs are regularized toward the reference imputations:

• Mean squared error for numerical features

• KL divergence for categorical distributions

To encourage robustness, artificial masking is applied during training by randomly hiding observed values. The model is then trained to reconstruct these values:

• Smooth L1 loss for numerical features

• Cross-entropy loss for categorical features

The contribution of the penalty and reconstruction terms is scaled dynamically based on prediction confidence. Confidence is estimated using the entropy of the predicted class probabilities, with lower entropy indicating higher confidence. The auxiliary losses are weighted proportionally to this confidence, encouraging stronger regularization when predictions are uncertain.

Training proceeds for a fixed number of iterations with mini-batch optimization. At each iteration:

• A fraction of observed entries is randomly masked (data augmentation)

• Missing values are imputed using either the learned imputer or the reference baseline (with probability 0.3)

• The classifier is trained on the resulting completed data

The Gumbel-Softmax temperature is annealed during training to encourage discrete categorical predictions. Early stopping is applied based on validation accuracy. At test time, missing values are imputed using the learned imputation network and predictions are produced by the classifier.

Number of parameters. Let d denote the input dimensionality and C the number of classes. The model consists of a classifier and a neural imputer. The total number of parameters is:

$$
\# \mathrm { P a r a m e t e r s } = P _ { \mathrm { c l a s s i f i e r } } + P _ { \mathrm { i m p u t e r } } ,
$$

where both terms depend only on d and the network architecture, and are independent of the number of training samples.

Implementation details. We use a hidden dimension of $h = 2 5 6$ for the imputation network. Model parameters are initialized using Kaiming uniform for linear layers. Missing numerical entries are initialized using the reference values with additive Gaussian noise $\mathcal { N } ( 0 , 0 . 0 1 ^ { \breve { 2 } } )$ . For categorical features, missing values are initialized by sampling from a noisy version of the reference distribution via a softmax over perturbed log-probabilities. Observed categorical values are encoded as one-hot vectors. The scaling factor for the residual correction is fixed to $\alpha = 0 . 1$ . We use the Adam optimizer with learning rate $\mathrm { \overline { { 1 0 } } ^ { - 3 } }$ for the classifier and $5 \times 1 0 ^ { - 4 }$ for the imputer. Training is performed for up to 300 iterations with mini-batches of size 64 and early stopping based on validation accuracy with a patience of 50 iterations. Artificial masking is applied with probability proportional to the dataset missing rate, capped at 0.2. The Gumbel-Softmax temperature is initialized at $\tau = 1 . 0$ and annealed multiplicatively with a factor of 0.995 per iteration down to a minimum of 0.3. The penalty and reconstruction weights are set to $\lambda _ { \mathrm { p e n } } = 0 . 1$ and $\lambda _ { \mathrm { r e c } } = 0 . 1$ . Confidence-based scaling is applied using normalized prediction entropy.

Computational Complexity. Let n be the number of samples, d the input dimensionality, h the hidden dimension, b the batch size, and T the number of training iterations. The method jointly trains a neural imputer and a classifier, each requiring forward and backward passes with complexity $\mathcal { O } ( d h + h ^ { 2 } )$ per sample.

The overall training complexity is:

$$
O \left( T \cdot { \frac { n } { b } } \cdot \left( d h + h ^ { 2 } + d \right) \right) ,
$$

where the additional $\mathcal O ( d )$ term accounts for masking, reconstruction losses, and categorical sampling. The space complexity is $\mathcal { O } ( n d + d h + h ^ { 2 } )$ , dominated by storing the dataset, masks, and model parameters.

## B Results

## B.1 Overall stability

These results are obtained through non-parametric factor effect analysis with effect sizes $( \eta ^ { 2 } )$ to quantify the impact of dataset characteristics on downstream performance.

Joint-AmortizedImpute Table 12 shows how dataset characteristics affect the relative performance of Joint-AmortizedImpute compared to the best classical method. The results indicate that feature dimensionality, dataset size, missingness mechanism, number of classes, and imbalance, are statistically significant, though they present small effect sizes. In contrast, missing rate is not significant. Overall, the small effect sizes suggest that standard dataset characteristics explain only a limited portion of the performance differences, reinforcing that missingness structure, including intrinsic hardness, is the dominant factor.

Figure 6 shows the distribution of accuracy differences relative to the best-performing classical method on the test set across datasets. While methods often achieve similar performance (values close to zero), Joint-AmortizedImpute exhibits few extreme deviations, indicating stability.

Figure 7 shows the accuracy gap relative to the complete-data baseline as a function of the missing rate. As expected, performance degrades for all methods as missingness increases. However, Joint-AmortizedImpute consistently exhibits the smallest gap, indicating greater robustness to increasing levels of missingness.

![](images/99757ced52b819cf19a9cf3cdb708d8b1d0273d50c944e688cd722f3c89fe6fe.jpg)  
Figure 6: Accuracy difference to the best performing classical method on the test set.

![](images/0a9ed3204a1d6db92dd70988f9aca42847056e57cb37d9b245485d3845eca1e2.jpg)  
Figure 7: Accuracy gap to complete data baseline.

<table><tr><td>Factor</td><td>Sig.</td><td> $\eta ^ { 2 }$ </td></tr><tr><td>Imbalance</td><td>Yes</td><td>0.005</td></tr><tr><td>Missing Rate</td><td>No</td><td>0.0005</td></tr><tr><td>Dataset Size</td><td>Yes</td><td>0.05</td></tr><tr><td>#Categorical</td><td>Yes</td><td>0.017</td></tr><tr><td>Mechanism</td><td>Yes</td><td>0.015</td></tr><tr><td>#Classes</td><td>Yes</td><td>0.01</td></tr></table>

Table 12: Factor impact on performance differences between Joint-AmortizedImpute and the best classical method.

Joint-LatentImpute. Overall, additional results on Joint-LatentImpute tell a similar story.

Table 13 shows that the number of categorical features, dataset size and missingness mechanism have a statistically significant effect on the performance difference between the proposed Joint-LatentImpute and the best benchmark imputation method. Imbalance, missing rate and number of classes do not show a statistically significant effect. Once again, the small effect sizes suggest that differences in performance are not fully explained by dataset properties.

Figure 8 shows the distribution of accuracy differences relative to the best-performing classical method on the test set across datasets. Joint-LatentImpute remains consistently close to the best method, with a tighter distribution and fewer extreme degradations, suggesting improved robustness compared to classical approaches.

Figure 9 shows the accuracy gap relative to the complete-data baseline as a function of the missing rate. Joint-LatentImpute consistently exhibits the smallest gap, indicating greater robustness to increasing levels of missingness.

![](images/2eacd009bb662ee65ec2383f299b9a737edba15466242108f5c00c44b51ee8fa.jpg)  
Figure 8: Accuracy difference to the best classical method.

![](images/e887a154d623cd2ffa9eea9c6a368f861e6ef17c99486772fdc47b1eebbc4dd5.jpg)  
Figure 9: Accuracy gap vs missing rate (all methods).

<table><tr><td>Factor</td><td>Sig.</td><td> $\eta ^ { 2 }$ </td></tr><tr><td>Imbalance</td><td>No</td><td>0.002</td></tr><tr><td>Missing Rate</td><td>No</td><td>0.0002</td></tr><tr><td>Dataset Size</td><td>Yes</td><td>0.03</td></tr><tr><td>#Categorical</td><td>Yes</td><td>0.04</td></tr><tr><td>Mechanism</td><td>Yes</td><td>0.02</td></tr><tr><td>#Classes</td><td>No</td><td>0.003</td></tr></table>

Table 13: Factor impact on performance differences between Joint-LatentImpute and the best classical method.

## B.2 Computational cost

Joint-AmortizedImpute. The proposed Joint-AmortizedImpute introduces an upfront training cost, with a mean training time of approximately 39.3 seconds due to the joint optimization of the imputation and predictive components. However, once trained, imputation is performed via a single forward pass, resulting in efficient inference.

Joint-LatentImpute. We compare the computational cost of the proposed method with classical imputation approaches. As expected, simple methods such as mean and median imputation incur negligible cost, while more advanced methods such as MissForest and TabImpute require several seconds per dataset. The proposed joint method has a mean imputation time of approximately 31.7 seconds due to the end-to-end optimization of imputations and model parameters. We report wall-clock time required to impute the training data. For iterative methods such as MissForest and TabImpute it includes model fitting.

<table><tr><td>Method</td><td>mean (↓)</td><td>median (↓)</td><td>std (↓)</td></tr><tr><td>Column Mean</td><td>0.0009</td><td>0.000</td><td>0.002</td></tr><tr><td>Column Median</td><td>0.0009</td><td>0.000</td><td>0.003</td></tr><tr><td>K-NN</td><td>0.692</td><td>0.195</td><td>1.4</td></tr><tr><td>MissForest</td><td>12.02</td><td>8.51</td><td>11.23</td></tr><tr><td>TabImpute</td><td>21.94</td><td>7.15</td><td>57.47</td></tr><tr><td>Joint-LatentImpute (Ours)</td><td>31.68</td><td>12.66</td><td>48.75</td></tr></table>

Table 14: Average time (in seconds) required to obtain a fully imputed training dataset across all datasets. For iterative methods (MissForest and TabImpute), this includes model fitting.