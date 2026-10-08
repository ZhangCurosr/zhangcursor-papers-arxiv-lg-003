# THINKING IN DEPTH: RETROSPECTIVE INFERENCE FOR TABULAR FOUNDATION MODELS

Hao-Run Cai, Si-Yang Liu, Zi-Jian Cheng, Kun-Yang Yu, Jin-Hao Sheng, Guo Yu, Chonghan Liu, Zhi Zhou, Jun-Peng Jiang, Lan-Zhe Guo, Han-Jia Ye School of Artificial Intelligence, Nanjing University National Key Laboratory for Novel Software Technology, Nanjing University

## ABSTRACT

Tabular foundation models (TFMs) are pretrained across diverse tabular tasks and make predictions on a new table at inference time using its labeled examples as context. Most recent TFMs perform such in-context prediction with stacked Transformer layers, repeatedly transforming how examples are represented and compared. By tracing individual queries through several strong TFMs, we find that predictive refinement is highly uneven across depth and is often concentrated in later layers. This uneven refinement motivates us to reconsider how intermediate representations are constructed and reused throughout the network. We introduce RETRO, a tabular foundation model based on retrospective inference, where later stages can explicitly revisit and recombine intermediate information produced earlier in the network. RETRO organizes this process around two complementary operations: which intermediate information to revisit, and how the resulting contextual update should be shaped for each query. Attention Residuals address the former by adaptively reweighting contributions from different depths, while query-conditioned Gated Attention addresses the latter by modulating the attention output element-wise across representation dimensions. Our analysis shows that RETRO shifts predictive refinement earlier and more broadly across depth, with different stages revising different subsets of queries in a pattern suggestive of multi-view refinement. Across TabArena, TALENT, and RelArena, RETRO ranks among the top three and lies on the Pareto frontier. These results indicate that directly reusing intermediate representations provides a practical way to better exploit depth in TFMs.

## 1 INTRODUCTION

Tabular foundation models (TFMs) are reshaping tabular learning from training a separate predictor for each dataset toward reusing a pretrained prediction procedure across heterogeneous tables. By pretraining over diverse tasks, TFMs can adapt to a new table at inference time using its labeled examples as context, without requiring a taskspecific model to be learned from scratch (Hollmann et al., 2023; 2025; Qu et al., 2025). Recent models have demonstrated increasingly strong performance across a broad range of tabular tasks (Qu et al., 2026; Grinsztajn et al., 2026; Eo et al., 2026), establishing TFMs as a promising general-purpose paradigm for tabular prediction.

In most recent in-context TFMs, labeled support examples and unlabeled queries pass through a stack of Transformer layers before producing the final prediction. Several works have examined what happens along this layered computation. Ye et al. (2025a) show that representations formed within a TFM contain task-conditioned predictive structure and can support downstream prediction, while Balef et al. (2026) reveal substantial redundancy and non-uniform contributions across layers through probing and layer interventions. These studies begin to characterize how predictions emerge through the internal computation of TFMs rather than only at their final output.

![](images/1d00c1f8a7a4135a1656a34d0775256d346930b159d7178684ae06e8e6c01003.jpg)  
Figure 1: RETRO lies on the Pareto frontier of three mainstream benchmarks.

![](images/a236721af978a159521fee5940c689aa41553b7ca440bb6b58b2fb62b4d682cd.jpg)  
Figure 2: A housing example of complementary neighborhood views and retrospective integration. Top: a person considers size, location, and condition, then revisits these assessments to reach a decision. Bottom: the TFM schematic places the same homes in different neighborhoods across representations and integrates earlier information in the final representation. This illustrates a possible organization of tabular inference.

We take a complementary, sample-level view by tracing how individual queries change as they pass through several strong TFMs. We find that predictive refinement is highly uneven across depth and, in some models, is strongly concentrated in later layers. Different queries are also revised at different stages, revealing a much less uniform process than final aggregate performance alone would suggest. These observations suggest that the issue may not simply be what each layer computes, but also how information produced along the way remains available to later prediction. This motivates a different organization of intermediate computation, in which earlier representations can be reconsidered rather than influencing subsequent stages only through the latest accumulated state.

We call this organization retrospective inference. Instead of viewing prediction as a sequence in which each representation simply gives way to the next, retrospective inference keeps intermediate information available as the prediction evolves. The goal is to let later computation selectively draw on representations formed at different stages, so that useful information can be retained and recombined as different queries are progressively refined.

This perspective leads to two coupled design requirements. First, when several intermediate representations remain available, the model must determine which of them are relevant to the current sample. Second, after contextual information is computed from these representations, the model must determine how that update should modify the current query representation. Accordingly, we design RETRO with a retrospective aggregation path that adaptively reweights contributions from different depths, and a query-conditioned gating mechanism that modulates newly computed contextual information across representation dimensions. We instantiate the former with Attention Residuals (Kimi Team, 2026) and the latter with Gated Attention (Qiu et al., 2025). Together, they allow the model to adapt both the historical information it reuses and the way new contextual information is incorporated for each query.

Our analysis shows that RETRO shifts predictive refinement earlier and distributes it more broadly across depth, with different stages revising different subsets of queries in a pattern suggestive of multi-view refinement. Across TabArena (Erickson et al., 2025), TALENT (Ye et al., 2024), and RelArena (Hayler et al., 2026), RETRO ranks in the top three among the evaluated methods, while lying on the estimated performance–cost Pareto frontier of each benchmark. These results highlight the value of treating intermediate representations as reusable resources for subsequent inference. Our contributions are threefold:

• We characterize sample-level predictive dynamics in strong TFMs, complementing prior representation- and layer-level analyses, and show that refinement is highly uneven across depth and often concentrated in later stages.

• We propose retrospective inference and instantiate it in RETRO through adaptive reuse of intermediate representations and query-conditioned modulation of contextual updates, leading to earlier and more distributed prediction refinement across stages.

• We evaluate RETRO on three benchmarks spanning tabular classification, regression, and relational prediction, obtaining top-three overall Elo rankings among the evaluated methods.

## 2 RELATED WORK

Tabular machine learning. Tabular supervised learning predicts a target for each row from a set of heterogeneous feature columns, typically including numerical and categorical variables. Most dataset-specific tabular predictors fall broadly into two families. Tree-based methods, particularly gradient-boosted decision trees, include XGBoost (Chen & Guestrin, 2016), LightGBM (Ke et al., 2017), and CatBoost (Prokhorenkova et al., 2018). Deep tabular methods learn task-specific predictors with neural architectures, including FT-Transformer (Gorishniy et al., 2021), RealMLP (Holzmuller¨ et al., 2024), ModernNCA (Ye et al., 2025b), TabM (Gorishniy et al., 2025), and TabPack (Gorishniy et al., 2026). The relative strengths of these methods have been systematically examined in large-scale empirical studies and benchmarks (Grinsztajn et al., 2022; McElfresh et al., 2023; Liu et al., 2025; Erickson et al., 2025). Broader surveys further organize the architectures, learning paradigms, and generalization regimes of tabular learning methods (Borisov et al., 2024; Jiang et al., 2026).

Tabular foundation models. Tabular foundation models (TFMs) move beyond dataset-specific fitting by pretraining a prediction procedure across many tasks and applying it to unseen tables using labeled examples as context (van Breugel & van der Schaar, 2024). TabPFN (Hollmann et al., 2023) established this paradigm with a Transformer that performs in-context prediction over labeled and query examples. Subsequent TFMs mainly differ in how they represent a table and exchange information across features and samples. One family retains fine-grained feature or cell states while exchanging information across feature and sample axes, including TabPFN v2 (Hollmann et al., 2025), LimiX (Zhang et al., 2025a), and EXAONE-Tabular (Eo et al., 2026). Another first aggregates feature information into row representations and then performs row-level in-context learning, including TabICL (Qu et al., 2025), TabICLv2 (Qu et al., 2026), TabPFN-3 (Grinsztajn et al., 2026), and TabFM (Kong & Das, 2026). Efficient direct row-wise modeling provides another design, as exemplified by TabSwift (Liu & Ye, 2026). Orthogonal to architecture, richer synthetic priors and downstream adaptation improve transfer across heterogeneous tasks, as explored by TabForestPFN (den Breejen et al., 2024), Mitra (Zhang et al., 2025b) and Mitra-v2 (Tao et al., 2026). RETRO follows the feature-to-row, row-level ICL hierarchy, but redesigns representation flow within the ICL stack so that intermediate contributions remain directly reusable by later stages.

Recent work has also begun to examine how TFMs arrive at their predictions beyond final predictive accuracy. Ye et al. (2025a) analyze task-conditioned representations in TabPFN v2 and study the predictive utility and reuse of intermediate representations. Balef et al. (2026) investigate layer-wise inference dynamics across multiple TFMs using probing and layer interventions, revealing substantial redundancy across depth. Bilos et al. (2026) use causal interventions to study similarity-based in-context prediction mechanisms across contemporary TFMs. In our work, such analysis serves as a design motivation rather than the endpoint: we trace sample-level prediction changes, observe uneven refinement across depth, and use this phenomenon to motivate retrospective reuse of intermediate computation.

Residual connections. The study of inference across depth also raises questions about how intermediate information is passed between layers. Standard residual connections (He et al., 2016) facilitate optimization through additive updates, allowing earlier computations to contribute to later states. Subsequent designs introduce learned gates (Srivastava et al., 2015) or residual scaling (Bachlechner et al., 2021) to control the propagation of information and improve training. Hyper-Connections (Zhu et al., 2024) extend this design to multiple interacting residual streams, while Manifold-Constrained Hyper-Connections (mHC) (Xie et al., 2025) constrain their mixing matrices to stabilize signal propagation across depth. A complementary direction provides explicit access to earlier computations. DenseNet (Huang et al., 2017) concatenates earlier feature maps, while Attention Residuals (Kimi Team, 2026) use input-dependent weights to aggregate contributions from different depths. Delta Attention Residuals (Luo et al., 2026) further investigate the representations used as sources for this aggregation. In tabular modeling, Xiaomi-TabLDM (Wang et al., 2026) incorporates lightweight Attention Residuals into its architecture. RETRO builds on block Attention Residuals to organize information flow within the ICL stack, and examines how retrospective access relates to sample-level prediction changes across depth.

## 3 PRELIMINARIES

Tabular in-context learning. Let $S = \{ ( \mathbf { x } _ { j } , y _ { j } ) \} _ { j = 1 } ^ { n _ { s } }$ be a labeled support set and $Q = \{ \mathbf { x } _ { i } ^ { q } \} _ { i = 1 } ^ { n _ { q } }$ a set of unlabeled queries from the same table. The target space is $\mathcal { Y } = \{ 1 , \ldots , K \}$ for classification and $\mathcal { V } = \mathbb { R }$ for regression. A tabular foundation model (TFM) is pretrained across tasks with different feature schemas and targets. At inference time, it predicts $p _ { \theta } ( y \mid \mathbf { \bar { x } } _ { i } ^ { q } , S )$ using observed support labels while keeping its pretrained parameters θ fixed; query labels are withheld. In architectures such as TabICLv2 (Qu et al., 2026), this process consists of two stages: constructing a representation for each row, then performing in-context prediction over these row representations.

Row representations. Following TabICLv2, we use column-wise embedding and row-wise feature aggregation to construct the inputs to the ICL stack. The column encoder produces context-dependent feature representations, and the row encoder aggregates them into a fixed-width vector $h _ { 0 i } \in \mathbb { R } ^ { d }$ for each sample. For $n = n _ { s } + n _ { q }$ support and query rows, these vectors form $H _ { 0 } \in \mathbb { R } ^ { n \times d }$ . Observed support labels are incorporated into the representations used for ICL, enabling the subsequent layers to condition predictions on the current task.

Multi-layer in-context prediction. An L-block Transformer processes $H _ { 0 }$ using the labeled support set as context. Let $H _ { \ell } \in \mathbb { R } ^ { n \times d }$ collect the row representations after block $\ell ,$ with row i denoted by $h _ { \ell i }$ . Each block applies attention followed by a feed-forward network, both with residual connections. Before attention, layer normalization produces $Z _ { \ell } = \mathrm { L N } ( H _ { \ell - 1 } )$ , where $\mathrm { L N }$ denotes layer normalization applied independently to each row. Let $Z _ { \ell } ^ { S }$ contain the normalized support rows. For clarity, we write the attention computation for a single head and omit layer indices on its parameters:

$$
\begin{array} { r l } & { O _ { \ell } = \operatorname { s o f t m a x } \left( \frac { ( Z _ { \ell } W _ { Q } ^ { \top } ) ( Z _ { \ell } ^ { S } W _ { K } ^ { \top } ) ^ { \top } } { \sqrt { d _ { k } } } \right) Z _ { \ell } ^ { S } W _ { V } ^ { \top } , } \\ & { A _ { \ell } = O _ { \ell } W _ { O } ^ { \top } + \mathbf { b } _ { O } ^ { \top } . } \end{array}\tag{1}
$$

Here $W _ { Q } , W _ { K }$ , and $W _ { V }$ are learned query, key, and value projections; $d _ { k }$ is the query/key dimension. The softmax normalizes each receiving row’s scores over the support rows. $\dot { W } _ { O }$ and $\mathbf { b } _ { O }$ are the output projection and its bias, shared across rows. In multi-head attention, $O _ { \ell }$ concatenates the outputs of all heads before this projection. The update $A _ { \ell }$ is added to $H _ { \ell - 1 }$ , followed by a pre-normalized feed-forward residual update. After L blocks, a prediction head maps the final query representations to outputs.

## 4 HOW DO TFMS REFINE PREDICTIONS ACROSS DEPTH?

Following Ye et al. (2025a), we extract support embeddings in a held-out query role and fit a linear readout on their final-layer representations. Applying the same readout to $h _ { \ell i }$ across depth lets us track each query’s predictive state and true-class probability margin. We analyze TabICLv2, TabPFN-3, and EXAONE-Tabular on the first official split of 38 TabArena classification datasets (Erickson et al., 2025). Extraction, margin definitions, and visualization details are given in Appendix A.

We find that predictive refinement is often concentrated in a small portion of the network. In Figure 3 (left), TabICLv2 shows little visible predictive change through much of its depth, followed by a pronounced late transition. Across datasets, the final third accounts for 84.4% of its absolute margin movement, compared with 51.5% for TabPFN-3 and 61.1% for EXAONE-Tabular. Across all three models, predictive refinement is unevenly distributed across depth, although its concentration varie across architectures.

These observations motivate organizing depth around more differentiated contributions to prediction. Successive representations could emphasize different sample relationships, while keeping their result available for later integration. This diagnostic tracks how predictive information becomes accessible to a common final-layer readout across depth. Our design question is how to make intermediate results explicitly reusable as the focus of subsequent computation changes.

![](images/883fd39413973176974f16c14663253310c251878aed83cb740a51beb92e4f0d.jpg)  
Figure 3: Layer-wise changes in predictive margin for TabICLv2 and a retrospective variant on SDSS17. The left and right grids show query representations across all 12 layers of TabICLv2 and our retrospective variant, respectively. Within each model, representations are projected onto a fixed PCA basis fitted to its final-layer support embeddings; coordinates are therefore comparable across layers within a model, but not between models. Color indicates the change in true-class probability margin from the preceding layer (∆m), with green denoting an increase and purple a decrease. Circles mark queries changing from incorrect to correct, and crosses mark the reverse. L1 is shown in gray because no preceding layer is measured. Insets enlarge L1, L4, and L7; the L4 and L7 insets use separately labeled, narrower color scales to reveal small early-layer changes. TabICLv2 exhibits limited directly readable changes through much of its depth, whereas the retrospective variant shows changes for different queries across more stages. Full four-model visualizations and aggregate results over 38 datasets appear in Appendices A.4 and C.1.

## 5 RETROSPECTIVE INFERENCE FOR TABULAR FOUNDATION MODELS

The findings in Section 4 raise a question about representation learning: how does the way intermediate information is reused affect the contributions learned across depth? We introduce retrospective inference, which makes earlier contributions directly available to later computation. Attention Resid uals implement this reuse, while Gated Attention modulates the contextual updates produced within the resulting stack.

## 5.1 RETROSPECTIVE REUSE WITH ATTENTION RESIDUALS

In a standard residual stack, each sublayer reads a state that accumulates preceding updates. An early contribution can influence subsequent prediction, but only as part of this evolving state; it is no longer available as a separate source that later layers can selectively weight. Consequently, its predictive value depends on how subsequent transformations retain and use it. This organization permits layer-wise specialization, but does not explicitly support selecting an earlier contribution independently of those accumulated after it.

Retrospective inference changes this relationship between intermediate computation and prediction. By retaining earlier contributions as separate sources, it allows a stage to provide information that remains useful when selected later, even if that information is insufficient for prediction on its own. We hypothesize that this reuse pathway favors complementary contributions across depth: different stages can refine different sample relationships without requiring their results to remain separately recoverable from a single accumulated state. For tabular ICL, this means that a representation useful for comparing some queries with their support examples can remain accessible while later stages refine other comparisons.

![](images/89b15d725ecb3e7549c3792e36e336bda7ee2b603bf92ee74f4dfc006a153637.jpg)  
Figure 4: Sample-state flows across depth. Frozen-readout trajectories for TabICLv2 (left) and our ungated retrospective variant (right), averaged equally over 38 classification datasets. Correct merges always-correct and previously-wrong but currently-correct queries; the two wrong states distinguish whether a query was ever correct. Ribbon widths show query fractions across all 12 layers. The displayed average measures correctness switches over L1→L2 through L10→L11; only this summary excludes the final transition.

We realize this access through block Attention Residuals (Kimi Team, 2026). The source bank contains the initial row encoding, sums of residual updates from completed two-layer groups, and the current group’s partial sum. At reading site t, let $B _ { t }$ denote the available sources and $b _ { i }$ the representation of row i in source b. A learned vector $w _ { t }$ produces row-dependent weights:

$$
\alpha _ { t b i } = \frac { \exp ( \langle w _ { t } , \mathrm { R M S N o r m } _ { t } ( b _ { i } ) \rangle ) } { \sum _ { b ^ { \prime } \in \mathcal { B } _ { t } } \exp ( \langle w _ { t } , \mathrm { R M S N o r m } _ { t } ( b _ { i } ^ { \prime } ) \rangle ) } , \qquad \widetilde { h } _ { t i } = \sum _ { b \in \mathcal { B } _ { t } } \alpha _ { t b i } b _ { i } .\tag{2}
$$

The scorer is shared across rows but learned separately at each reading site. Weights are recomputed from each row’s candidate contributions. RMS normalization applies only to scoring; the values remain unnormalized. Independent readers precede every feed-forward sublayer and every attention sublayer after the first. Sublayer outputs accumulate within the current group, and completed groups remain available to later readers. The final reader combines the initial encoding and completed groups before prediction. This retains grouped update contributions rather than every full hidden state; Appendix B gives the recurrence.

Representation dynamics. We examine the resulting trajectories with the fixed final-support readout used in Section 4. In Figure 3 (right), predictive changes appear earlier and involve different queries at successive stages. The aggregate state flows in Figure 4 show revisions distributed across depth, includ ing both corrections of previously incorrect predictions and deteriorations of previously correct ones.

Under this fixed readout, retrospective inference exhibits earlier and more distributed prediction changes, with successive stages revising different subsets of queries. This pattern is consistent with a form of multi-view refinement, where different stages contribute to different subsets of queries. Additional model comparisons and classification and regression cases are reported in Figures 13 and 14 and Appendices C.4 and C.5.

## 5.2 SHAPING CONTEXTUAL UPDATES WITH GATED ATTENTION

Historical reads determine the input to a sublayer; its output becomes a new contribution available for later reuse. We use Gated Attention (Qiu et al., 2025) to modulate this contextual update for each receiving row. Let $z _ { \ell i }$ be the normalized attention-reader output and $o _ { \ell i }$ the concatenated attention-head outputs. Following the row-vector convention of Equation (1),

$$
\begin{array} { r l } & { g _ { \ell i } = \sigma ( z _ { \ell i } W _ { g , \ell } ^ { \top } + \mathbf { b } _ { g , \ell } ^ { \top } ) , } \\ & { A _ { \ell i } = ( g _ { \ell i } \odot o _ { \ell i } ) W _ { O , \ell } ^ { \top } + \mathbf { b } _ { O , \ell } ^ { \top } . } \end{array}\tag{3}
$$

Here $\sigma$ is the sigmoid. Each layer learns a gate projection shared across support and query rows. Ordinary attention is already query-dependent; gating additionally modulates its output channels before projection. It can therefore alter both the direction and magnitude of the update, rather than only its overall strength. The gated update enters the current group’s accumulation, shaping the information retained for subsequent reads.

![](images/e12d3780e3e40b70440b624357963de71e5c0836a899cc553da6f1c8b5989915.jpg)  
Figure 5: Gated-update geometry. Mean-gate replacement effects on 38 classification and 13 regression datasets in TabArena. Whiskers: 95% dataset-bootstrap intervals.

![](images/2a688730755a494347c27d04d52b6c26ffa3cbefd5689895088849a7d4353736.jpg)  
Figure 6: Component ablation. TabArena classification Elo within the four-configuration comparison. Whiskers: reported 95% intervals.

Direction and magnitude. We analyze fixed gated models on 38 classification and 13 regression datasets, using the first official split, one view without ensembling, and native final prediction heads. A counterpart update replaces each query’s gate with the channelwise mean over the unlabeled query batch. We substitute either this full update, its direction while retaining the learned norm, or its norm while retaining the learned direction. Replacements occur after output projection at every ICL layer, on query rows only. All normalization layers remain in place; the updates are recomputed at each intervened state (Appendix D.1).

Replacing direction reduces mean classification accuracy by 0.616 percentage points, compared with −0.002 for replacing magnitude (Figure 5); the paired difference is 0.618 points (95% interval: [0.173, 1.217]). The corresponding regression NRMSE increases are 0.2328 and 0.0011, with NRMSE defined as query RMSE divided by the support-target standard deviation. Thus, under these replacements, the trained models depend more on the learned update direction than on its norm. This supports a role for gating in shaping contextual information, beyond attenuating its strength. Figure 6 provides the complementary component ablation, with dataset-level results and additional controls reported in Appendix D.

## 6 RETRO: MODEL, TRAINING, AND EVALUATION

We combine retrospective access and selective contextual incorporation in RETRO, a tabular foundation model built on a hierarchical table encoder and a row-level ICL stack. We describe its architecture and pretraining procedure, then evaluate its predictive performance, component contributions, and computational cost.

Architecture. As illustrated in Figure 7, the table encoder consists of three column-processing blocks and three row-processing blocks, both with a hidden dimension of 128 and eight attention heads. The column stage uses 128 inducing vectors, while the row stage aggregates feature representations through four CLS tokens. Their outputs are concatenated into a 512-dimensional representation for each row. We retain TabICLv2’s repeated feature grouping, target-aware embeddings, and queryaware scalable softmax (QASSMax). The row representations, augmented with observed support labels, enter a 12-layer ICL stack with eight attention heads and a feed-forward expansion ratio of two. Within this stack, retrospective aggregation constructs the inputs to attention and feed-forward sublayers, and gated attention modulates contextual outputs before projection (Equations (2) and (3)). Residual updates are grouped over pairs of layers. The final aggregation combines the initial encoding and six completed update groups, and a two-layer MLP with hidden dimension 1,024 produces query predictions. Following TabICLv2, classification and regression use separately trained models with task-specific target embeddings and output heads.

Pretraining. We use the synthetic task generator of TabICLv2, which samples structural causal models with diverse graph structures and functional relationships to generate heterogeneous numerical and categorical features. Each task is divided into labeled support examples and held-out queries.

![](images/04e8470c0c188973336f797e4637c9d8fbc8962c686993b30edeb7b9e70947ef.jpg)  
Figure 7: The RETRO architecture. A table encoder produces row representations for a retrospective ICL stack. Within each layer, separate aggregation modules construct attention and feed-forward inputs from available historical contributions. An input-conditioned gate modulates attention outputs before projection. Completed two-layer update groups remain available to subsequent layers and the final aggregation used for prediction.

The classification model learns to predict query class labels, while the regression model predicts 999 quantiles at levels {0.001, 0.002, . . . , 0.999} using the summed pinball loss. Pretraining follows a three-stage curriculum that progressively increases the number of samples per task, with up to 100 features and a batch size of 64 tasks. The first stage uses 500K updates on tasks with 1,024 samples, allocating 30–90% to the support set. The second and third stages use 40K and 10K updates, respectively, with task sizes sampled log-uniformly from 400–10,240 and 400–60,000 samples and an 80% support fraction. We use Muon for matrix-valued parameters and AdamW for one-dimensional parameters, following the optimization setup of TabICLv2. We adopt cosine learning-rate schedules with peak rates of $8 \times 1 0 ^ { - 4 } , 1 0 ^ { - 4 }$ , and $2 \times 1 0 ^ { - 5 }$ across the three stages, and a weight decay of 0.01. Further details of synthetic data generation, filtering, and the remaining optimization settings follow Qu et al. (2026).

Evaluation protocol and predictive performance. At inference time, pretrained parameters remain fixed and labeled support examples provide task-specific context. Classification predictions are obtained from the classification head; regression point predictions are computed by averaging the predicted quantiles. We retain the preprocessing and inference procedures of TabICLv2. The benchmark scope, available evaluation settings, and aggregation procedures are described in Appendix E.

We evaluate on TabArena, TALENT, and RelArena using the comparison pools listed in Appendix E. We report separate classification and regression Elo rankings, together with overall ratings recomputed from pooled results rather than averaged across the two task types.

Figure 8 shows the five highest-rated methods in each ranking. Among the evaluated methods, RETRO ranks third on TabArena, second on TALENT, and first on RelArena by overall Elo. It also ranks above TabICLv2 in the overall rankings on TabArena and TALENT, indicating competitive performance relative to the backbone on which our architecture is based.

The task-specific rankings reveal variation across evaluation settings. On TabArena, RETRO ranks fourth for classification and second for regression. On TALENT, it ranks second for both task types, although its overall Elo is close to those of several competing TFMs. On RelArena, it ranks first for classification and second for regression. These results establish competitive performance across classification and regression, while the relative advantages depend on the benchmark. Complete rankings and uncertainty estimates are provided in Appendix E.

Component ablation. On TabArena classification (Figure 6), the combined model has the highest Elo point estimate (1115.6) in this ablation, while both single-mechanism variants score above the model with neither. These Elo values use a separate comparison pool from the main rankings (Appendix E.2).

![](images/e3e15c74884a0d87659e2ccd711f58aa54b63af311518b5a227d558dfdbf69a6.jpg)  
Figure 8: Elo rankings across three benchmarks. Each panel shows the five highest-rated methods. Candidate sets and scoring protocols differ, so ratings should be compared within panels. Complete rankings appear in Appendix E.

Efficiency. RETRO retains the backbone’s width and depth, adding only historical aggregation modules and gate projections within the ICL stack. The aggregation modules introduce 24,576 parameters. With a d-dimensional gate in each of the L layers, gate projections add $L ( d ^ { 2 } + d )$ parameters, approximately 3.15M for $L = 1 2$ and $d = 5 1 2$ . For n rows and B available historical sources, each aggregation requires $O ( B n d )$ computation, while each gate projection requires $\mathcal { O } ( n d ^ { 2 } )$ These operations complement the existing $\bar { \mathcal { O } } ( n \bar { n } _ { s } d )$ support-conditioned attention and $\mathcal { O } ( n d ^ { 2 } )$ feedforward computation, where $n _ { s }$ is the support-set size. The performance–runtime trade-off in Figure 1 shows that RETRO improves predictive performance over TabIC $\lrcorner \tt V 2$ with nearly unchanged inference time. Relative to the other evaluated methods, it lies on the Pareto frontier: no compared method achieves both better predictive performance and lower runtime under the reported evaluation setting. This positions RETRO as a competitive choice when both predictive quality and inference cost matter.

## 7 DISCUSSION AND CONCLUSION

We studied how individual queries are refined across the depth of contemporary TFMs and found that predictive changes are uneven, often concentrating in later layers. This observation motivates retrospective inference: retaining access to earlier contributions while adapting how newly computed contextual information modifies each query. RETRO instantiates these two operations through Attention Residuals and query-conditioned, element-wise Gated Attention.

Retrospective variants exhibit earlier and more distributed prediction changes, with different stages revising different subsets of queries. Fixed-weight gating interventions further reveal sensitivity to update direction beyond changes in magnitude alone. Native-prediction comparisons place RETRO among the top three methods by overall Elo on TabArena, TALENT, and RelArena within the evaluated candidate sets. Together, these findings support organizing intermediate computation as reusable information rather than only as transient states passed from layer to layer.

## REFERENCES

Thomas Bachlechner, Bodhisattwa Prasad Majumder, Huanru Henry Mao, Gary Cottrell, and Julian J. McAuley. Rezero is all you need: fast convergence at large depth. In UAI, 2021.

Amir Rezaei Balef, Mykhailo Koshil, and Katharina Eggensperger. Is one layer enough? understanding inference dynamics in tabular foundation models. In ICML, 2026.

Marin Bilos, James T. Wilson, Anderson Schneider, and Yuriy Nevmyvaka. A mechanistic study of tabular foundation models. In ICML, 2026.

Vadim Borisov, Tobias Leemann, Kathrin Seßler, Johannes Haug, Martin Pawelczyk, and Gjergji Kasneci. Deep neural networks and tabular data: A survey. IEEE Transactions on Neural Networks and Learning Systems, 35(6):7499–7519, 2024.

Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In KDD, 2016.

Felix den Breejen, Sangmin Bae, Stephen Cha, and Se-Young Yun. Why in-context learning transformers are tabular data classifiers. CoRR, abs/2405.13396, 2024.

Moonjung Eo, Min-Kook Suh, Hye-Seung Cho, Jiwon Kim, Seoyoon Kim, Sangjun Nam, and Soonyoung Lee. EXAONE Tabular 1.0: Technical report. CoRR, abs/2608.25774, 2026.

Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmuller, Prateek Mutalik Desai, David¨ Salinas, and Frank Hutter. TabArena: A living benchmark for machine learning on tabular data. In NeurIPS, 2025.

Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Revisiting deep learning models for tabular data. In NeurIPS, 2021.

Yury Gorishniy, Akim Kotelnikov, and Artem Babenko. Tabm: Advancing tabular deep learning with parameter-efficient ensembling. In ICLR, 2025.

Yury Gorishniy, Akim Kotelnikov, Ivan Rubachev, and Artem Babenko. Tabpack: Efficient hyperparameter ensembles for tabular deep learning. In ICML, 2026.

Leo Grinsztajn, Edouard Oyallon, and Ga ´ el Varoquaux. Why do tree-based models still outperform¨ deep learning on tabular data? In NeurIPS, 2022.

Leo Grinsztajn, Klemens Fl´ oge, Oscar Key, Felix Birkel, Philipp Jund, Brendan Roof, Mihir Manium,¨ Shi Bin Hoo, Magnus Buhler, Anurag Garg, Dominik Safaric, Jake Robertson, Benjamin J¨ ager,¨ Simone Alessi, Adrian Hayler, Vladyslav Moroshan, Lennart Purucker, Philipp Singer, Alan Arazi, Julien Siems, Jan Hendrik Metzen, Georg Grab, Nick Erickson, Siyuan Guo, Eliott Kalfon, Simon Bing, David Salinas, Clara Cornu, Lilly Charlotte Wehrhahn, Diana Kriuchkova, Kursat Kaya, Lydia Sidhoum, Marie Salmon, Jerry Chen, Madelon Hulsebos, Yann LeCun, Samuel Muller,¨ Bernhard Scholkopf, Sauraj Gambhir, Noah Hollmann, and Frank Hutter. TabPFN-3: Technical¨ report. CoRR, abs/2605.13986, 2026.

Adrian Hayler, Klemens Floge, Alan Arazi, Rishabh Ranjan, Jure Leskovec, Felix Birkel, Brendan¨ Roof, Anurag Garg, Kristina Collins, Lydia Sidhoum, Jonas M. Kubler, Siyuan Guo, Oscar¨ Key, Jan Hendrik Metzen, Rylee Grace, David Salinas, Arthur Cahu, Simon Bing, Benjamin Jager, Tuana¨ C¸ elik, Mihir Manium, Vitor Monteiro, Jake Robertson, Jerry Chen, Eliott Kalfon, Tomas Pereda, Lilly Charlotte Wehrhahn, Dominik Safaric, Tobias Schroeder, Georg Grab, Diana´ Kriuchkova, Clara Cornu, Philipp Singer, Nick Erickson, Vahid Balazadeh, Marie Salmon, Simone Alessi, Kur¨ s¸at Kaya, Philipp Jund, Leo Grinsztajn, Yann LeCun, Bernhard Sch´ olkopf, Madelon¨ Hulsebos, Lennart Purucker, Sauraj Gambhir, Frank Hutter, and Noah Hollmann. Advancing open and reproducible relational learning: Relarena-α, tabpfn-rel and RPI. CoRR, abs/2608.16319, 2026.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

Noah Hollmann, Samuel Muller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer¨ that solves small tabular classification problems in a second. In ICLR, 2023.

Noah Hollmann, Samuel Muller, Lennart Purucker, Arjun Krishnakumar, Max K¨ orfer, Shi Bin Hoo,¨ Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 637(8044):319–326, 2025.

David Holzmuller, L ¨ eo Grinsztajn, and Ingo Steinwart. Better by default: Strong pre-tuned mlps and´ boosted trees on tabular data. In NeurIPS, 2024.

Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. In CVPR, 2017.

Jun-Peng Jiang, Si-Yang Liu, Hao-Run Cai, Qi-Le Zhou, and Han-Jia Ye. Representation learning for tabular data: A comprehensive survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(6):6488–6508, 2026.

Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu. LightGBM: A highly efficient gradient boosting decision tree. In NIPS, 2017.

Kimi Team. Attention residuals. CoRR, abs/2603.15031, 2026.

Weihao Kong and Abhimanyu Das. Introducing TabFM: A zero-shot foundation model for tabular data. Google Research Blog, 2026.

Si-Yang Liu and Han-Jia Ye. TabSwift: An efficient tabular foundation model with row-wise attention. In ICML, 2026.

Si-Yang Liu, Hao-Run Cai, Qi-Le Zhou, Huai-Hong Yin, Tao Zhou, Jun-Peng Jiang, and Han-Jia Ye. Talent: A tabular analytics and learning toolbox. Journal ofMachine Learning Research, 26(226): 1–16, 2025.

Cheng Luo, Zefan Cai, and Junjie Hu. Delta attention residuals. CoRR, abs/2605.18855, 2026.

Duncan C. McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C., Ganesh Ramakrishnan, Micah Goldblum, and Colin White. When do neural nets outperform boosted trees on tabular data? In NeurIPS, 2023.

Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. Catboost: Unbiased boosting with categorical features. In NeurIPS, 2018.

Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free. In NeurIPS, 2025.

Jingang Qu, David Holzmuller, Ga¨ el Varoquaux, and Marine Le Morvan. TabICL: A tabular¨ foundation model for in-context learning on large data. In ICML, 2025.

Jingang Qu, David Holzmuller, Ga ¨ el Varoquaux, and Marine Le Morvan. TabICLv2: A better, faster,¨ scalable, and open tabular foundation model. In ICML, 2026.

Rupesh Kumar Srivastava, Klaus Greff, and Jurgen Schmidhuber. Highway networks.¨ CoRR, abs/1505.00387, 2015.

Yefan Tao, Xiyuan Zhang, Xinyi Liu, Boran Han, Danielle Maddix, Haoyang Fang, Zhen Han, Jiading Gai, Xuanqing Liu, Michael Bohlke-Schneider, Yuyang Wang, Gerald Friedland, Kevan Mah, Chris Lee, and Chris Kong. Mitra-v2 technical report. CoRR, abs/2609.04540, 2026.

Boris van Breugel and Mihaela van der Schaar. Position: Why tabular foundation models should be a research priority. In ICML, 2024.

Penghui Wang, Wei Liu, Hong Wang, Chengyue Huang, Yuxi Sun, Zirui Wang, Hongming Huang, Quan Wang, Zhenwei Xin, Ping Hou, Jie Yu, Chunxiao Liu, Erli Meng, and Bin Wang. Xiaomi-TabLDM: A tabular foundation model technical report. CoRR, abs/2609.03880, 2026.

Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, and Wenfeng Liang. mhc: Manifold-constrained hyper-connections. CoRR, abs/2512.24880, 2025.

Han-Jia Ye, Si-Yang Liu, Hao-Run Cai, Qi-Le Zhou, and De-Chuan Zhan. A closer look at deep learning methods on tabular datasets. CoRR, abs/2407.00956, 2024.

Han-Jia Ye, Si-Yang Liu, and Wei-Lun Chao. A closer look at TabPFN v2: Understanding its strengths and extending its capabilities. In NeurIPS, 2025a.

Han-Jia Ye, Huai-Hong Yin, De-Chuan Zhan, and Wei-Lun Chao. Revisiting nearest neighbor for tabular data: A deep tabular baseline two decades later. In ICLR, 2025b.

Xingxuan Zhang, Gang Ren, Han Yu, Hao Yuan, Hui Wang, Jiansheng Li, Jiayun Wu, Lang Mo, Li Mao, Mingchao Hao, Ningbo Dai, Renzhe Xu, Shuyang Li, Tianyang Zhang, Yue He, Yuanrui Wang, Yunjia Zhang, Zijing Xu, Dongzhe Li, Fang Gao, Hao Zou, Jiandong Liu, Jiashuo Liu, Jiawei Xu, Kaijie Cheng, Kehan Li, Linjun Zhou, Qing Li, Shaohua Fan, Xiaoyu Lin, Xinyan Han, Xuanyue Li, Yan Lu, Yuan Xue, Yuanyuan Jiang, Zimu Wang, Zhenlei Wang, and Peng Cui. Limix: Unleashing structured-data modeling capability for generalist intelligence. CoRR, abs/2509.03505, 2025a.

Xiyuan Zhang, Danielle Maddix Robinson, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael W. Mahoney, Tony Hu, Huzefa Rangwala, George Karypis, and Yuyang Wang. Mitra: Mixed synthetic priors for enhancing tabular foundation models. In NeurIPS, 2025b.

Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, and Xun Zhou. Hyper-connections. CoRR, abs/2409.19606, 2024.

## A LAYER-WISE REPRESENTATION ANALYSIS

This appendix specifies the diagnostic used in Sections 4 and 5 and reports aggregate depth statistics. Appendix B describes the retrospective computation, and Appendix C presents the full-depth cases. The fixed-readout analysis here is distinct from the native-head gating interventions in Appendix D and the benchmark evaluations in Appendix E.

## A.1 DATA AND QUERY-ROLE REPRESENTATIONS

We analyze the first official repeat/fold $( r 0 / f 0 )$ of each of the 38 TabArena classification datasets (Erickson et al., 2025). Models use the same outer query rows, a single preprocessing view, and no ensemble.

Following Ye et al. (2025a), support representations are obtained by cross-fitting within the outer support set. Each held-out support fold is encoded as unlabeled queries conditioned on the remaining support rows. The number of stratified folds is min $( 1 0 , n _ { \mathrm { m i n } } )$ , where $n _ { \mathrm { m i n } }$ is the smallest supportclass count; the random seed is 42. Outer queries are encoded against the full outer support set. Their labels enter neither representation extraction nor readout fitting. Cross-fitting removes the support row’s own label from its encoding, although the cross-fitted and outer-query contexts have different sizes.

Capture points. For TabICLv2, each intermediate representation is the input to the next block’s pre-attention normalization, with the final state taken from the encoder output. For the retrospective variants, this point includes the next attention reader; the final state includes the output reader. It is therefore the accessible mixture, not the raw update accumulator. Both sequences contain 12 states of width 512. TabPFN-3 supplies 24 ICL block outputs of width 512 before final RMS normalization. EXAONE-Tabular supplies 12 target-cell CAST outputs of width 192 after block normalization. A depth index denotes a native block, not equal computation across models. The retrospective diagnostics use Attention Residuals without gating.

## A.2 FIXED READOUTS AND PREDICTIVE CHANGES

A StandardScaler and L2-regularized logistic classifier are fitted to the final-layer cross-fitted support representations. We use $C = 1$ and LBFGS with tolerance $1 0 ^ { - 6 }$ , allowing up to 20,000 iterations when the initial 3,000-iteration limit is insufficient for convergence. The scaler and classifier remain fixed across depth.

Let $r _ { L }$ denote this readout and $p _ { \ell i } = r _ { L } ( h _ { \ell i } )$ . For query i with true class $y _ { i } ^ { q } ,$ , define

$$
m _ { \ell i } = p _ { \ell i , y _ { i } ^ { q } } - \operatorname* { m a x } _ { c \neq y _ { i } ^ { q } } p _ { \ell i , c } , \qquad c _ { \ell i } = { \bf 1 } \Bigl [ \mathrm { a r g } \operatorname* { m a x } _ { c } p _ { \ell i , c } = y _ { i } ^ { q } \Bigr ] .\tag{4}
$$

The probability margin and correctness indicator provide a common decision reference across depth.   
They are not predictions from the pretrained model’s native head.

Transitions. At layer $\ell > 1$ , correction and deterioration fractions are $C _ { \ell } = \mathbb { E } _ { i } [ ( 1 - c _ { \ell - 1 , i } ) c _ { \ell i } ]$ and $D _ { \ell } = \mathbb { E } _ { i } [ c _ { \ell - 1 , i } ( \mathrm { \bar { 1 } } - c _ { \ell i } ) ]$ . Writing $a _ { \ell } = \mathbb { E } _ { i } [ c _ { \ell i } ]$ gives

$$
a _ { \ell } - a _ { \ell - 1 } = C _ { \ell } - D _ { \ell } , \qquad R _ { \ell } = C _ { \ell } + D _ { \ell } , \qquad V _ { \ell } = \mathbb { E } _ { i } | m _ { \ell i } - m _ { \ell - 1 , i } | .\tag{5}
$$

A correctness switch does not include a change between two wrong classes. The early-motion share is $\textstyle \sum _ { \ell = 2 } ^ { \lfloor 2 L / 3 \rfloor } V _ { \ell } / \sum _ { \ell = 2 } ^ { L } V _ { \ell } ;$ its complement is the final-third share in Section 4. The early range contains seven of eleven transitions for 12-layer models and fifteen of twenty-three for TabPFN-3. All query rows contribute to each dataset’s statistics, followed by an equal-weight average across datasets.

State-flow plots use three histories: Correct, Currently wrong, previously correct, and Never correct. The second refers to any earlier correct state, not only the preceding layer; the third includes the current layer. The two currently-correct histories of the underlying four-state partition are merged by summing their node counts and transition counts. Every query and native layer is retained. The scalar summary printed on the flow plots averages $R _ { \ell }$ over $\ell = \dot { 2 } , \dots , L - 1$ , excluding the final transition. In contrast, the per-step statistics in Table 1 include all $L - 1$ transitions.

Table 1: Descriptive frozen-readout dynamics, equally averaged over 38 datasets (one official split each). Bold marks the largest value in each column. Early motion is the share of absolute probability-margin change before the final third of depth. TabPFN-3 has 24 layers; the other models have 12. Native-step averages complement the depth-dependent any-revision statistic, but do not equate compute or depth granularity across architectures.
<table><tr><td>Model</td><td>Any revision (%)</td><td>Revision / step (%)</td><td>E|∆m|</td><td>Early motion (%)</td></tr><tr><td>TabICLv2</td><td>63.8</td><td>9.82</td><td>0.197</td><td>15.6</td></tr><tr><td>TabPFN-3</td><td>66.8</td><td>7.55</td><td>0.149</td><td>48.5</td></tr><tr><td>EXAONE-Tabular</td><td>71.8</td><td>16.12</td><td>0.319</td><td>38.9</td></tr><tr><td>Ours</td><td>79.5</td><td>24.66</td><td>0.488</td><td>60.7</td></tr></table>

## A.3 VISUALIZATION AND REGRESSION

For each model and dataset, PCA is fitted to at most 1,200 deterministically selected final-layer support embeddings. Up to 1,200 query rows are shown. Every layer uses the same basis and axis limits within a model; coordinates are not aligned across models. Readouts operate in the original embedding space, so the plots are not two-dimensional decision surfaces.

In classification plots, green and purple denote positive and negative adjacent margin changes. Rings mark corrections and crosses mark deteriorations. The first measured layer is gray because it has no measured predecessor. The main SDSS17 figure includes separately labeled inset scales to resolve small early-layer changes. The remaining classification plates share the same main color scale.

Regression. The superconductivity case in Appendix C.4 contains 14,175 support and 7,088 query rows. Support extraction uses shuffled ten-fold cross-fitting with seed 42. A StandardScaler and Ridge regressor (α = 1, with intercept) are fitted to final-layer cross-fitted support representations and frozen across depth. Colors encode

$$
\delta e _ { \ell i } = \frac { | \widehat { y } _ { \ell - 1 , i } - y _ { i } ^ { q } | - | \widehat { y } _ { \ell i } - y _ { i } ^ { q } | } { s _ { S } } ,\tag{6}
$$

where $s _ { S }$ is the outer-support target standard deviation. Positive values indicate error reduction. The four models share a symmetric-log scale spanning [−10, 10]; no classification-state partition is assigned to regression.

## A.4 QUANTITATIVE DEPTH DIAGNOSTICS

Table 1 summarizes the ungated retrospective variant used in Figure 3. Mean correctness turnover per transition is 24.66% for the retrospective variant and 9.82% for TabICLv2; 79.5% and 63.8% of queries, respectively, change correctness at least once. The early-motion shares are 60.7% and 15.6%, with the retrospective share larger on 36 of 38 datasets. Turnover and absolute margin movement are larger on 33 of 38 datasets. In the state-flow comparison in Figure 4, final frozen-readout accuracies are 87.3% for TabICLv2 and 87.5% for our retrospective model.

The per-step statistics here include all transitions, whereas the scalar summaries in Figure 14 exclude the final transition. The model settings for each analysis are specified in Appendix A.1. The case studies in Appendix C complement both summaries by locating changes within individual query trajectories.

## A.5 INTERPRETATION

The frozen-readout analysis measures how predictive information is expressed relative to a common final-layer decision rule, rather than the total information contained in each representation. We therefore use it to compare layer-wise refinement patterns; the aggregate statistics include all 38 classification datasets, while the visual cases are illustrative.

## B RETROSPECTIVE ICL IMPLEMENTATION

This appendix specifies the grouped accumulation underlying Equations (2) and (3) and the architecture in Figure 7. The table encoder and pretraining procedure are described in Section 6.

## B.1 HISTORICAL READS AND GROUPED UPDATES

Let $P$ be the current group accumulator and M the ordered bank of completed sources. Initially $P = H _ { 0 }$ and M is empty. For layer ℓ, the attention input is

$$
U _ { \ell } = \left\{ { \begin{array} { l l } { P , } & { \ell = 1 , } \\ { { \mathrm { R e a d } } _ { a , \ell } ( M , P ) , } & { \ell > 1 , } \end{array} } \right.\tag{7}
$$

where each Read applies Equation (2) independently to every row. At the start of a two-layer group $( \ell = 1 , 3 , \ldots , 1 1 )$ , this read is followed by appending $P$ to M and resetting $P$ to zero. Thus, the initial encoding becomes the first stored source. Attention and feed-forward updates then proceed as

$$
\begin{array} { r l } & { P \gets P + A _ { \ell } ( \mathrm { L N } ( U _ { \ell } ) ; S ) , } \\ & { V _ { \ell } = \mathrm { R e a d } _ { f , \ell } ( M , P ) , } \\ & { P \gets P + \mathrm { F F N } _ { \ell } ( \mathrm { L N } ( V _ { \ell } ) ) . } \end{array}\tag{8}
$$

Here $A _ { \ell }$ includes the attention output projection and, when enabled, the gate in Equation (3). Both support and query rows receive updates, but only support rows supply attention keys and values. The attention read occurs before a group is stored; the feed-forward read occurs after the attention update is added.

After the last layer, an independent output reader combines (M, P). For 12 layers grouped in pairs, it reads seven sources: the initial encoding and six group sums. The capture points in Appendix A.1 include these learned reads rather than only the accumulator.

## B.2 NORMALIZATION, INITIALIZATION, AND GATING

Each reader has its own length-d scoring vector and RMS gain, shared across rows. The scorer is initialized to zero and the gain to one, giving uniform source weights initially. RMS normalization, scoring, softmax, and weighted reduction are evaluated in FP32; the result is cast to the input dtype. Only scoring uses normalized sources. The attention and feed-forward sublayers retain their pre-LayerNorms.

The scorer is a learned site-specific vector, not a projection of the latest state. Consequently, weights vary with the row’s source values and the reader’s parameters, without inheriting a previous reader’s weights. Gating is computed from the normalized attention input and multiplies the concatenated head outputs before their output projection. It modulates latent channels but does not skip attention or feed-forward computation. Its experimental role is examined in Appendix D.

There are $\left( L - 1 \right) + L + 1 = 2 L$ historical readers, adding 4Ld parameters, or 24,576 at $L = 1 2 , d =$ 512. The L gate projections add $L ( d ^ { 2 } + d )$ parameters. These counts cover the additional modules, not the total model size or measured runtime.

## C FULL-DEPTH REPRESENTATION AND STATE-FLOW ANALYSES

The following case studies extend Sections 4 and 5 using the extraction and readout protocol in Appendices A.1 and A.2. We first compare full-depth SDSS17 representations and state flows, then report benchmark-average flows and additional regression and classification cases. All measured layers are retained. Within each model, every layer uses the final-support PCA basis; neither coordinates nor computation per layer are aligned across models. All trajectories use the fixed readouts defined in Appendix A.2, while Appendix A.4 reports aggregate classification statistics.

## C.1 SDSS17: FULL-DEPTH REPRESENTATIONS

Figures 9–12 extend Figure 3 to all four models on the same SDSS17 split. TabPFN-3 has 24 layers; the other models have 12. These full-depth views reveal when predictive changes emerge and which queries they affect, complementing the comparison of early and late refinement in Section 4.

<table><tr><td>L1</td><td>L2 X</td><td>L3 a</td><td>L4 J</td></tr><tr><td>L5 S</td><td>L6 J</td><td>L7 J</td><td>L8 J</td></tr><tr><td>L9 A</td><td>L10</td><td>L11 Canwd</td><td>L12 O O</td></tr><tr><td colspan="2">oWrong to correct × Correct to wrong</td><td>-2 0 2</td><td colspan="2">8 Probability-margin change ∆m</td></tr></table>

Figure 9: TabICLv2 on SDSS17. All native layers in the fixed final-support PCA basis. Colors show adjacent probability-margin change under the frozen readout; rings and crosses mark corrections and deteriorations. The first layer is gray. Extraction and display conventions are given in Appendices A.1 and A.3.

![](images/db22539c9fab79aed189da595fe59664b01e5498925c644a32f4b68a9388bb69.jpg)  
Figure 10: Our retrospective model on SDSS17. All native layers in the fixed final-support PCA basis. Colors show adjacent probability-margin change under the frozen readout; rings and crosses mark corrections and deteriorations. The first layer is gray. Extraction and display conventions are given in Appendices A.1 and A.3.

![](images/52d239c269d6e8c822fc6160422cde270ab3ad64b7f5eef96f44a7025b6387c4.jpg)  
Figure 11: TabPFN-3 on SDSS17. All native layers in the fixed final-support PCA basis. Colors show adjacent probability-margin change under the frozen readout; rings and crosses mark corrections and deteriorations. The first layer is gray. Extraction and display conventions are given in Appendices A.1 and A.3.

![](images/632c34df2196503bff2408db75474ccdaab8b32847d27706d046594418cbb399.jpg)  
Figure 12: EXAONE-Tabular on SDSS17. All native layers in the fixed final-support PCA basis. Colors show adjacent probability-margin change under the frozen readout; rings and crosses mark corrections and deteriorations. The first layer is gray. Extraction and display conventions are given in Appendices A.1 and A.3.

## C.2 SDSS17: SAMPLE-STATE FLOWS

Figure 13 tracks all 26,018 SDSS17 queries using the three-state partition in Appendix A.2. Unlike the sampled PCA displays, these flows retain every query. They show how corrections of previously incorrect predictions and deteriorations of previously correct ones contribute to the changing query populations across depth.

![](images/985ace076367763f5b5d627fc701f2e1bde91aaa7fb0e506e293c93fa333e97e.jpg)

![](images/71bc69f45288791ba656498a5e7ca5b319e1e7b89c8d0d24633b9885689de70f.jpg)

![](images/3996ea9374dab6c5c79b3790dc44264ff0eae3b81a2bf1b31b1cb29310c38c81.jpg)

![](images/39bc772a89da3472501be06d9e4b218467dc6ebd8f1fe1d14b2f8ca249043a69.jpg)  
Figure 13: Four-model sample-state flows on SDSS17. Node heights and ribbon widths are percentages of all outer queries. Correct merges both currently-correct histories; the two wrong states distinguish whether a query was ever correct at an earlier layer. All native layers are retained.

## C.3 BENCHMARK-AVERAGE SAMPLE-STATE FLOWS

Figure 14 extends Figure 4 to all four models over the 38 classification datasets. Each dataset contributes equally, regardless of query-set size. These flows compare how correctness changes are distributed across depth beyond a single illustrative dataset. Aggregation conventions are given in Appendices A.2 and A.4.

![](images/063ad4457765b17da7220149655720a0cefd3446d6c32e42402d3d5b6545eb1e.jpg)

![](images/a4ba43e8855753c979426de6f4bf0f0d24dce319345a2bee0b159a5d46bf45b7.jpg)

![](images/b3612538ba2df186d7402815a2be8205c4e184eb1ffc9424ba98597dffc978a4.jpg)

![](images/d004758f362a475b100dfcf8cd4539687b7d1fad870c070d4e11136d24d296a3.jpg)  
Figure 14: Four-model sample-state flows averaged over 38 classification datasets. Node heights and ribbon widths are equal-weight means of per-dataset query percentages, using the same three states as Figure 4. All native layers are shown. The average inter-layer prediction change excludes only each model’s final transition from the summary, not from the plotted flows.

## C.4 SUPERCONDUCTIVITY REGRESSION: FULL-DEPTH REPRESENTATIONS

Figures 15–18 extend the analysis to superconductivity. Colors encode the normalized reduction in absolute error defined in Equation (6), using one scale across models. This reveals improvements and deteriorations without discretizing a continuous prediction into correctness states.

![](images/1992dd9df89b40aa1731f2c1a5c801ea98ad53dff1d60ca8c5bc2db0c2eceec9.jpg)  
Figure 15: TabICLv2 on superconductivity. All native layers in the fixed final-support PCA basis. Green/purple indicate reductions/increases in absolute error under the frozen final-support Ridge readout, normalized by the support-target standard deviation. All four models share the same color scale. The first layer is gray; no binary correctness states are assigned.

![](images/5a73cd216073d6fe548e97f7d976e0acae3449e02bae0dc0c48783029ca6e51f.jpg)  
Figure 16: Our retrospective model on superconductivity. All native layers in the fixed finalsupport PCA basis. Green/purple indicate reductions/increases in absolute error under the frozen final-support Ridge readout, normalized by the support-target standard deviation. All four models share the same color scale. The first layer is gray; no binary correctness states are assigned.

![](images/2e13bc19e2e08c01e7ee2342595b723c612c8163d4969729151f0d1e5d3e53df.jpg)  
Figure 17: TabPFN-3 on superconductivity. All native layers in the fixed final-support PCA basis. Green/purple indicate reductions/increases in absolute error under the frozen final-support Ridge readout, normalized by the support-target standard deviation. All four models share the same color scale. The first layer is gray; no binary correctness states are assigned.

![](images/d8667ae93dcf977685bf3c383ba0ce6279cd9cd16e3dacbe55fd8ed2c6c97041.jpg)  
Figure 18: EXAONE-Tabular on superconductivity. All native layers in the fixed final-support PCA basis. Green/purple indicate reductions/increases in absolute error under the frozen final-support Ridge readout, normalized by the support-target standard deviation. All four models share the same color scale. The first layer is gray; no binary correctness states are assigned.

## C.5 ADDITIONAL CLASSIFICATION CASE: IS-THIS-A-GOOD-CUSTOMER

Figures 19–22 provide an additional classification case. Correct predictions can become incorrect at later layers, illustrating that refinement is not monotonic. These query-level trajectories complement the aggregate state-flow summaries by showing both corrections and deteriorations within the same dataset.

![](images/f0a155484f427d14191adf8a95c7ef22a92de4e56e843efa284756a0b2f03081.jpg)  
Figure 19: TabICLv2 on Is-this-a-good-customer. All native layers in the fixed final-support PCA basis. Colors show adjacent probability-margin change under the frozen readout; rings and crosses mark corrections and deteriorations. The first layer is gray. Extraction and display conventions are given in Appendices A.1 and A.3.

![](images/68f58a5dcadb06ffe975f867415b2fbc60ff20eae712f52e15023994913aa3b9.jpg)  
Figure 20: Our retrospective model on Is-this-a-good-customer. All native layers in the fixed final-support PCA basis. Colors show adjacent probability-margin change under the frozen readout; rings and crosses mark corrections and deteriorations. The first layer is gray. Extraction and display conventions are given in Appendices A.1 and A.3.

![](images/6684013180aec4b90e8b31eefd504773890b8caf91635c7bb9d6dace58ddeb02.jpg)  
Figure 21: TabPFN-3 on Is-this-a-good-customer. All native layers in the fixed final-support PCA basis. Colors show adjacent probability-margin change under the frozen readout; rings and crosses mark corrections and deteriorations. The first layer is gray. Extraction and display conventions are given in Appendices A.1 and A.3.

![](images/c78f0d579e6a76b3443724ded2c36002a3f5d18005e5a44a10c14fb1427e6623.jpg)  
Figure 22: EXAONE-Tabular on Is-this-a-good-customer. All native layers in the fixed finalsupport PCA basis. Colors show adjacent probability-margin change under the frozen readout; rings and crosses mark corrections and deteriorations. The first layer is gray. Extraction and display conventions are given in Appendices A.1 and A.3.

## D ANALYSIS OF GATED CONTEXTUAL UPDATES

This appendix extends the update-geometry analysis in Section 5.2. We first test which aspects of the learned update matter to a fixed predictor, then examine whether gates are more interchangeable among nearby samples. The component comparison in Figure 6 is separate from these fixed-model interventions. All results use native final prediction heads, unlike the frozen-readout trajectories in Appendix A.

## D.1 DIRECTION AND MAGNITUDE INTERVENTIONS

Protocol. We evaluate gated classification and regression models with 12 ICL layers, width 512, and two-layer history groups. Each model uses weights averaged over the final 100 updates of the second pretraining stage. The study covers the first official split of all 51 TabArena datasets (38 classification and 13 regression), using every support and query row, one view, and FP32 inference without ensembling. Query labels are used only for evaluation. Dataset means receive equal weight; 95% intervals use 20,000 paired dataset-bootstrap resamples.

Geometric controls. At each intervened state, let $o _ { \ell i }$ and $g _ { \ell i }$ be the concatenated attention output and learned gate for query i. Using the row-vector convention of Equation (3), define

$$
\begin{array} { r l } & { \bar { g } _ { \ell } = | Q | ^ { - 1 } \displaystyle \sum _ { i \in Q } g _ { \ell i } , } \\ & { u _ { \ell i } = ( g _ { \ell i } \odot o _ { \ell i } ) W _ { O , \ell } ^ { \top } + \mathbf { b } _ { O , \ell } ^ { \top } , } \\ & { v _ { \ell i } = ( \bar { g } _ { \ell } \odot o _ { \ell i } ) W _ { O , \ell } ^ { \top } + \mathbf { b } _ { O , \ell } ^ { \top } . } \end{array}\tag{9}
$$

The counterpart $v _ { \ell i }$ uses the channelwise mean gate of the current unlabeled query batch. We compare replacing the update by $v _ { \ell i }$ with two partial replacements:

$$
u _ { \ell i } ^ { \mathrm { d i r } } = v _ { \ell i } \frac { \| u _ { \ell i } \| _ { 2 } } { \| v _ { \ell i } \| _ { 2 } } , \qquad u _ { \ell i } ^ { \mathrm { m a g } } = u _ { \ell i } \frac { \| v _ { \ell i } \| _ { 2 } } { \| u _ { \ell i } \| _ { 2 } } .\tag{10}
$$

The first changes direction while preserving the learned norm; the second changes the norm while preserving direction. Both act on the projected update, including its bias, rather than on the gate vector itself. Only query updates are replaced, at all 12 layers. Support updates, feed-forward computation, and normalization remain unchanged. The pair (u, v) is recomputed along each intervened trajectory.

Results. Figure 23 expands Figure 5 with dataset-level effects. Mean-gate replacement reduces classification accuracy by 0.719 percentage points. Replacing direction loses 0.616 points, versus −0.002 for replacing magnitude. The paired contrast is 0.618 points (95% interval: [0.173, 1.217]), but its median is 0.126: 23 datasets favor preserving direction, seven favor preserving magnitude, and eight tie. On credit-g, direction replacement is less harmful by 1.497 points. The effect is therefore heterogeneous across classification tasks.

For regression, NRMSE is query RMSE divided by the support-target standard deviation. Mean-gate, direction, and magnitude replacement increase NRMSE by 0.2326, 0.2328, and 0.0011, respectively. The direction-minus-magnitude contrast is 0.2317 (95% interval: [0.1579, 0.3070]), positive on all 13 datasets. Under these controls, preserving the learned direction is less disruptive than preserving only update size.

Interpretation. These results support query-conditioned shaping of contextual updates, particularly their direction. Together with the component comparison in Figure 6, they indicate that gating affects the geometry of the contextual update rather than only its overall scale.

## D.2 GATE EXCHANGE WITHIN SAMPLE NEIGHBORHOODS

We next examine whether gates can be exchanged with less disruption between nearby samples. Normalized first-layer attention inputs are standardized using support statistics. Eight MiniBatchKMeans clusters are fitted to support rows, and queries are assigned without their labels. Complete query gate vectors are cyclically exchanged within these groups. The control uses random groups with exactly the same sizes. Both conditions preserve the gate multiset, moved-query count, and singleton count; support gates remain unchanged. Groups are fixed across layers, and results are averaged over three exchange seeds within each dataset.

(a) Classification (n=38)  
![](images/3a39bde20220500ed26dd4e7309113d5b7230d86c57b24de8ab959793398397c.jpg)

(b) Regression (n=13)  
![](images/3216563523db2451af6907f89bc9aa97719fd1ad4f85ebca419cbdfb84aea4ea.jpg)

(c) Paired datasets  
![](images/61e349be50739e686bed826fd83f8515f05a9763f4e13c11e71cb7069b7372f2.jpg)  
Replace magnitude: Accuracy loss (pp)

(d) Paired datasets  
![](images/972989dc7b4be040d1b24e3c475463ab3006348c3e2479b219d2794d2a65e28a.jpg)  
Figure 23: Sensitivity to update direction and magnitude. Top: mean degradation relative to learned gating, with 95% dataset-bootstrap intervals. Bottom: dataset-level effects; points above the diagonal indicate greater harm from replacing direction. The classification inset enlarges the region near zero. Positive values mean lower accuracy or higher NRMSE. Replacement distances are not matched.

In Figure 24, neighborhood-constrained exchange has a mean advantage of 0.441 accuracy points (95% interval: [0.057, 1.031]) and 0.0810 NRMSE ([0.0415, 0.1289]). Classification has 18 improvements, 14 reversals, and six ties, with median advantage zero; all 13 regression datasets favor neighborhood-constrained exchange.

Local exchanges also have smaller gate displacement on every dataset. Mean gate RMS displacement is 0.0839 versus 0.0991 for classification and 0.1163 versus 0.1322 for regression. The smaller displacement helps explain why local exchanges are less disruptive. Together, these observations are consistent with local continuity of sample-conditioned updates, without establishing distinct semantic roles for the neighborhoods.

![](images/463864b3c32071c2e59f623c09c1e499be64f10fdd93423d0cf359828d2090f5.jpg)

![](images/5f277616e8311fc84694c5ff91f19b89458e0f44c3245ab68e61e5bb0045ad97.jpg)  
Figure 24: Neighborhood-constrained versus random gate exchange. Each point is a dataset, averaging three exchange seeds. Below the diagonal, exchange within support-derived neighborhoods is less harmful. Group sizes and moved-row counts are matched, within-neighborhood exchanges also induce smaller gate displacement.

## E BENCHMARK PROTOCOLS AND COMPLETE RANKINGS

This appendix provides the evaluation scope and complete rankings underlying Section 6 and Figure 8. These results evaluate native predictors, not the intermediate readouts in Appendix A or the singleview interventions in Appendix D.

## E.1 EVALUATION SCOPE

TabArena includes 38 classification and 13 regression datasets, comprising 594 and 222 official splits, respectively. TALENT includes 200 classification and 100 regression datasets. RelArena includes 12 classification and nine regression tasks. The comparison pools contain 37/38/39 methods for TabArena (overall/classification/regression), 17 for each TALENT ranking, and 10 for each RelArena ranking.

## E.2 RANKING AND UNCERTAINTY

All tables are ordered by decreasing Elo, with RETRO highlighted in bold. Average Rank is lower-isbetter. The benchmark-specific Elo procedures and comparison pools are retained, so Elo values are not calibrated across benchmarks.

Elo+ and Elo− are the distances from the point estimate to the upper and lower interval endpoints. TabArena uses 200 dataset-bootstrap replicates and RelArena 100 task-bootstrap replicates, with 2.5th and 97.5th percentile endpoints. TALENT uses dataset-level aggregated scores, averaging evaluation seeds. Its Elo is averaged over 30 random permutations of dataset order. The resulting interval reflects order variability, not evaluation-seed uncertainty or a bootstrap confidence interval.

For TabArena, Average Rank is computed from ranks within each split, averaged first within a dataset and then equally across datasets. TALENT and RelArena retain their benchmark-exported average ranks. Overall Elo is fitted to the pooled classification and regression results.

Component comparison. Figure 6 compares both mechanisms, Attention Residuals only, Gated Attention only, and neither on TabArena classification, with Elo point estimates 1115.6, 1093.1, 969.0, and 822.3, respectively. These values belong to the four-configuration comparison, not the full benchmark candidate pool, and should not be compared directly with Figure 8. Whiskers reproduce the reported 95% intervals.

## E.3 TABARENA

Tables 2–4 report the complete overall, classification, and regression comparisons for the TabArena panels of Figure 8.

Table 2: TabArena: overall ranking. All 37 selected methods on 51 datasets.
<table><tr><td>Method</td><td>Elo ↑</td><td>Elo+</td><td>Elo-</td><td>Average Rank ↓</td></tr><tr><td>TabFM (default)</td><td>1882.9</td><td>109.9</td><td>99.5</td><td>4.31</td></tr><tr><td>EXAONE-Tabular (default)</td><td>1849.7</td><td>72.8</td><td>49.7</td><td>4.89</td></tr><tr><td>RETRO</td><td>1741.6</td><td>84.3</td><td>72.5</td><td>7.26</td></tr><tr><td>TabPFN-3 (default)</td><td>1728.9</td><td>76.1</td><td>55.1</td><td>7.58</td></tr><tr><td>Xiaomi-TabLDM (default)</td><td>1673.7</td><td>74.2</td><td>66.0</td><td>9.11</td></tr><tr><td>TabPFN-2.6 (default)</td><td>1669.5</td><td>62.7</td><td>44.9</td><td>9.24</td></tr><tr><td>TabICLv2 (default)</td><td>1650.9</td><td>67.6</td><td>57.6</td><td>9.80</td></tr><tr><td>TabDPT-1.3 (default)</td><td>1601.4</td><td>60.6</td><td>40.1</td><td>11.39</td></tr><tr><td>RealMLP (tuned + ensembled)</td><td>1558.3</td><td>46.1</td><td>44.7</td><td>12.88</td></tr><tr><td>TabDPT-Turbo (default)</td><td>1502.2</td><td>59.1</td><td>50.6</td><td>14.93</td></tr><tr><td>TabM (tuned + ensembled)</td><td>1501.3</td><td>49.8</td><td>45.7</td><td>14.97</td></tr><tr><td>LightGBM (tuned + ensembled)</td><td>1483.0</td><td>32.0</td><td>33.5</td><td>15.66</td></tr><tr><td>RealMLP (tuned)</td><td>1470.7</td><td>46.5</td><td>50.4</td><td>16.13</td></tr><tr><td>CatBoost (tuned + ensembled)</td><td>1467.6</td><td>33.1</td><td>39.8</td><td>16.25</td></tr><tr><td>CatBoost (tuned)</td><td>1452.1</td><td>36.5</td><td>35.0</td><td>16.85</td></tr><tr><td>TabM (tuned)</td><td>1437.1</td><td>49.6</td><td>43.0</td><td>17.43</td></tr><tr><td>ModernNCA (tuned + ensembled)</td><td>1437.1</td><td>67.8</td><td>52.1</td><td>17.43</td></tr><tr><td>LightGBM (tuned)</td><td>1427.2</td><td>31.4</td><td>26.7</td><td>17.82</td></tr><tr><td>XGBoost (tuned + ensembled)</td><td>1417.9</td><td>32.0</td><td>31.9</td><td>18.18</td></tr><tr><td>CatBoost (default)</td><td>1410.3</td><td>47.6</td><td>43.1</td><td>18.48</td></tr><tr><td>ModernNCA (tuned)</td><td>1394.9</td><td>41.1</td><td>40.3</td><td>19.08</td></tr><tr><td>XGBoost (tuned)</td><td>1387.8</td><td>30.3</td><td>30.9</td><td>19.36</td></tr><tr><td>TabSwift (default)</td><td>1387.8</td><td>66.3</td><td>53.8</td><td>19.36</td></tr><tr><td>TabM (default)</td><td>1328.5</td><td>48.2</td><td>47.5</td><td>21.65</td></tr><tr><td>ModernNCA (default)</td><td>1279.4</td><td>39.2</td><td>40.3</td><td>23.50</td></tr><tr><td>ExtraTrees (tuned + ensembled)</td><td>1254.1</td><td>53.4</td><td>55.9</td><td>24.42</td></tr><tr><td>RealMLP (default)</td><td>1253.7</td><td>36.1</td><td>45.8</td><td>24.43</td></tr><tr><td>XGBoost (default)</td><td>1234.3</td><td>37.6</td><td>41.9</td><td>25.12</td></tr><tr><td>ExtraTrees (tuned)</td><td>1218.7</td><td>58.2</td><td>68.1</td><td>25.66</td></tr><tr><td>RandomForest (tuned + ensembled)</td><td>1207.4</td><td>61.8</td><td>58.3</td><td>26.04</td></tr><tr><td>LightGBM (default)</td><td>1202.6</td><td>43.4</td><td>36.9</td><td>26.20</td></tr><tr><td>RandomForest (tuned)</td><td>1163.2</td><td>62.9</td><td>69.3</td><td>27.48</td></tr><tr><td>Linear (tuned + ensembled)</td><td>1032.9</td><td>65.0</td><td>106.2</td><td>31.10</td></tr><tr><td>ExtraTrees (default)</td><td>1019.3</td><td>56.9</td><td>86.3</td><td>31.42</td></tr><tr><td>RandomForest (default)</td><td>1000.0</td><td>55.0</td><td>66.1</td><td>31.86</td></tr><tr><td>Linear (tuned)</td><td>996.6</td><td>70.0</td><td>126.9</td><td>31.93</td></tr><tr><td>Linear (default)</td><td>897.1</td><td>76.7</td><td>148.5</td><td>33.81</td></tr></table>

Table 3: TabArena: classification ranking. All 38 selected methods on 38 datasets.
<table><tr><td>Method</td><td>Elo ↑</td><td>Elo+</td><td>Elo-</td><td>Average Rank ↓</td></tr><tr><td>TabFM (default)</td><td>1833.3</td><td>118.3</td><td>107.4</td><td>4.76</td></tr><tr><td>EXAONE-Tabular (default)</td><td>1821.1</td><td>82.0</td><td>56.0</td><td>5.00</td></tr><tr><td>TabPFN-3 (default)</td><td>1690.7</td><td>74.8</td><td>68.0</td><td>8.12</td></tr><tr><td>RETRO</td><td>1680.4</td><td>79.1</td><td>70.8</td><td>8.42</td></tr><tr><td>TabPFN-2.6 (default)</td><td>1638.4</td><td>56.6</td><td>50.2</td><td>9.70</td></tr><tr><td>Xiaomi-TabLDM (default)</td><td>1631.5</td><td>70.7</td><td>60.1</td><td>9.93</td></tr><tr><td>TabICLv2 (default)</td><td>1627.6</td><td>78.4</td><td>68.7</td><td>10.05</td></tr><tr><td>TabDPT-1.3 (default)</td><td>1566.9</td><td>83.6</td><td>61.4</td><td>12.16</td></tr><tr><td>RealMLP (tuned + ensembled)</td><td>1516.8</td><td>50.0</td><td>41.7</td><td>14.05</td></tr><tr><td>TabM (tuned + ensembled)</td><td>1495.9</td><td>57.1</td><td>45.0</td><td>14.88</td></tr><tr><td>LightGBM (tuned + ensembled)</td><td>1463.0</td><td>51.0</td><td>35.2</td><td>16.21</td></tr><tr><td>TabDPT-Turbo (default)</td><td>1454.3</td><td>72.5</td><td>60.1</td><td>16.57</td></tr><tr><td>TabICL (default)</td><td>1447.4</td><td>49.0</td><td>56.2</td><td>16.85</td></tr><tr><td>CatBoost (tuned + ensembled)</td><td>1442.0</td><td>50.1</td><td>42.7</td><td>17.07</td></tr><tr><td>TabM (tuned)</td><td>1437.8</td><td>55.9</td><td>43.0</td><td>17.25</td></tr><tr><td>RealMLP (tuned)</td><td>1434.3</td><td>56.6</td><td>54.2</td><td>17.40</td></tr><tr><td>CatBoost (tuned)</td><td>1430.9</td><td>49.0</td><td>44.7</td><td>17.54</td></tr><tr><td>LightGBM (tuned)</td><td>1413.3</td><td>41.1</td><td>34.8</td><td>18.27</td></tr><tr><td>XGBoost (tuned + ensembled)</td><td>1407.7</td><td>42.0</td><td>47.0</td><td>18.51</td></tr><tr><td>CatBoost (default)</td><td>1407.2</td><td>48.3</td><td>47.6</td><td>18.53</td></tr><tr><td>ModernNCA (tuned + ensembled)</td><td>1391.6</td><td>71.9</td><td>65.9</td><td>19.18</td></tr><tr><td>ModernNCA (tuned)</td><td>1385.3</td><td>51.3</td><td>35.9</td><td>19.44</td></tr><tr><td>XGBoost (tuned)</td><td>1376.6</td><td>38.1</td><td>46.0</td><td>19.81</td></tr><tr><td>TabSwift (default)</td><td>1366.2</td><td>64.6</td><td>61.1</td><td>20.25</td></tr><tr><td>TabM (default)</td><td>1331.4</td><td>53.4</td><td>48.0</td><td>21.69</td></tr><tr><td>ModernNCA (default)</td><td>1265.9</td><td>45.2</td><td>48.3</td><td>24.34</td></tr><tr><td>RealMLP (default)</td><td>1251.2</td><td>39.7</td><td>45.2</td><td>24.92</td></tr><tr><td>XGBoost (default)</td><td>1242.9</td><td>43.7</td><td>50.1</td><td>25.24</td></tr><tr><td>ExtraTrees (tuned + ensembled)</td><td>1241.7</td><td>63.5</td><td>62.3</td><td>25.28</td></tr><tr><td>RandomForest (tuned + ensembled)</td><td>1210.6</td><td>77.4</td><td>77.2</td><td>26.45</td></tr><tr><td>ExtraTrees (tuned)</td><td>1207.0</td><td>67.6</td><td>73.2</td><td>26.58</td></tr><tr><td>LightGBM (default)</td><td>1192.7</td><td>57.8</td><td>55.7</td><td>27.10</td></tr><tr><td>RandomForest (tuned)</td><td>1166.1</td><td>76.0</td><td>80.3</td><td>28.03</td></tr><tr><td>Linear (tuned + ensembled)</td><td>1088.1</td><td>91.1</td><td>98.4</td><td>30.51</td></tr><tr><td>Linear (tuned)</td><td>1053.4</td><td>91.8</td><td>106.5</td><td>31.48</td></tr><tr><td>RandomForest (default)</td><td>1000.0</td><td>78.3</td><td>78.3</td><td>32.81</td></tr><tr><td>ExtraTrees (default)</td><td>997.5</td><td>90.5</td><td>87.7</td><td>32.87</td></tr><tr><td>Linear (default)</td><td>956.3</td><td>97.9</td><td>123.9</td><td>33.76</td></tr></table>

Table 4: TabArena: regression ranking. All 39 selected methods on 13 datasets.
<table><tr><td>Method</td><td>Elo ↑</td><td>Elo+</td><td>Elo-</td><td>Average Rank ↓</td></tr><tr><td>TabFM (default)</td><td>2165.6</td><td>254.1</td><td>73.4</td><td>3.56</td></tr><tr><td>RETRO</td><td>2082.1</td><td>332.6</td><td>117.5</td><td>4.83</td></tr><tr><td>EXAONE-Tabular (default)</td><td>2043.9</td><td>251.0</td><td>83.9</td><td>5.52</td></tr><tr><td>TabPFN-3 (default)</td><td>1966.4</td><td>320.5</td><td>122.8</td><td>7.15</td></tr><tr><td>Xiaomi-TabLDM (default)</td><td>1928.5</td><td>348.6</td><td>176.1</td><td>8.05</td></tr><tr><td>Nori-30M (default)</td><td>1895.0</td><td>254.2</td><td>76.1</td><td>8.90</td></tr><tr><td>TabPFN-2.6 (default)</td><td>1866.7</td><td>252.9</td><td>51.6</td><td>9.66</td></tr><tr><td>TabICLv2 (default)</td><td>1835.8</td><td>292.9</td><td>124.4</td><td>10.53</td></tr><tr><td>Nori (default)</td><td>1825.1</td><td>242.2</td><td>65.6</td><td>10.84</td></tr><tr><td>TabDPT-1.3 (default)</td><td>1817.0</td><td>299.2</td><td>111.4</td><td>11.07</td></tr><tr><td>RealMLP (tuned + ensembled)</td><td>1786.9</td><td>204.5</td><td>47.5</td><td>11.97</td></tr><tr><td>TabDPT-Turbo (default)</td><td>1762.6</td><td>251.9</td><td>83.4</td><td>12.72</td></tr><tr><td>ModernNCA (tuned + ensembled)</td><td>1684.5</td><td>206.6</td><td>99.9</td><td>15.24</td></tr><tr><td>RealMLP (tuned)</td><td>1678.3</td><td>189.1</td><td>78.3</td><td>15.45</td></tr><tr><td>CatBoost (tuned + ensembled)</td><td>1628.3</td><td>212.6</td><td>76.2</td><td>17.13</td></tr><tr><td>LightGBM (tuned + ensembled)</td><td>1623.3</td><td>202.9</td><td>90.5</td><td>17.30</td></tr><tr><td>CatBoost (tuned)</td><td>1597.8</td><td>219.6</td><td>88.8</td><td>18.17</td></tr><tr><td>TabM (tuned + ensembled)</td><td>1593.0</td><td>240.3</td><td>96.3</td><td>18.34</td></tr><tr><td>LightGBM (tuned)</td><td>1546.3</td><td>227.3</td><td>87.7</td><td>19.94</td></tr><tr><td>TabSwift (default) XGBoost (tuned + ensembled)</td><td>1536.7</td><td>212.2</td><td>140.0</td><td>20.27</td></tr><tr><td>ModernNCA (tuned)</td><td>1523.8</td><td>195.5</td><td>77.9</td><td>20.71</td></tr><tr><td></td><td>1509.2</td><td>199.9</td><td>105.4</td><td>21.21</td></tr><tr><td>TabM (tuned) XGBoost (tuned)</td><td>1505.1</td><td>248.6</td><td>102.7</td><td>21.34</td></tr><tr><td></td><td>1497.4</td><td>217.6</td><td>87.9</td><td>21.61</td></tr><tr><td>CatBoost (default)</td><td>1491.9</td><td>167.0</td><td>126.8</td><td>21.79</td></tr><tr><td>ModernNCA (default)</td><td>1393.3</td><td>192.3</td><td>94.7</td><td>25.01</td></tr><tr><td>TabM (default)</td><td>1384.0</td><td>222.6</td><td>125.6</td><td>25.30</td></tr><tr><td>ExtraTrees (tuned + ensembled)</td><td>1354.7</td><td>168.4</td><td>107.1</td><td>26.20</td></tr><tr><td>RealMLP (default)</td><td>1325.6</td><td>172.1</td><td>125.8</td><td>27.05</td></tr><tr><td>ExtraTrees (tuned) LightGBM (default)</td><td>1316.4</td><td>169.4</td><td>113.4</td><td>27.31</td></tr><tr><td>XGBoost (default)</td><td>1288.8</td><td>168.9</td><td>90.4</td><td>28.08</td></tr><tr><td>RandomForest (tuned + ensembled)</td><td>1259.9</td><td>191.5</td><td>124.5</td><td>28.84</td></tr><tr><td></td><td>1246.1</td><td>118.0</td><td>116.6</td><td>29.19</td></tr><tr><td>RandomForest (tuned)</td><td>1199.1</td><td>119.6</td><td>129.5</td><td>30.31</td></tr><tr><td>ExtraTrees (default)</td><td>1121.8</td><td>133.6</td><td>157.9</td><td>31.91</td></tr><tr><td>RandomForest (default)</td><td>1000.0</td><td>107.5</td><td>137.5</td><td>33.82</td></tr><tr><td>Linear (tuned + ensembled)</td><td>491.6</td><td>142.7</td><td>2134.7</td><td>37.27</td></tr><tr><td>Linear (tuned)</td><td>389.4</td><td>185.5</td><td>2245.4</td><td>37.71</td></tr><tr><td>Linear (default)</td><td>121.4</td><td>220.6</td><td>2245.2</td><td>38.69</td></tr></table>

## E.4 TALENT

Tables 5–7 report the corresponding TALENT comparisons. Their interval columns use the datasetorder variability described in Appendix E.2.

Table 5: TALENT: overall ranking. All 17 selected methods on 300 datasets.
<table><tr><td>Method</td><td>Elo ↑</td><td>Elo+</td><td>Elo-</td><td>Average Rank ↓</td></tr><tr><td>TabFM</td><td>1856.0</td><td>338.5</td><td>339.8</td><td>3.99</td></tr><tr><td>RETRO</td><td>1755.4</td><td>280.9</td><td>443.1</td><td>5.41</td></tr><tr><td>EXAONE-Tabular</td><td>1745.2</td><td>254.4</td><td>239.3</td><td>5.22</td></tr><tr><td>Xiaomi-TabLDM</td><td>1717.2</td><td>223.6</td><td>290.2</td><td>6.05</td></tr><tr><td>TabPFN-3</td><td>1697.7</td><td>357.0</td><td>347.5</td><td>5.68</td></tr><tr><td>TabICLv2</td><td>1679.3</td><td>218.9</td><td>384.4</td><td>6.22</td></tr><tr><td>TabPFN-2.5</td><td>1614.0</td><td>251.5</td><td>334.8</td><td>7.01</td></tr><tr><td>TabPFN-2.6</td><td>1566.5</td><td>277.9</td><td>329.9</td><td>6.91</td></tr><tr><td>TabSwift</td><td>1432.0</td><td>276.8</td><td>295.7</td><td>9.80</td></tr><tr><td>TabM (tuned)</td><td>1394.0</td><td>415.9</td><td>282.2</td><td>11.08</td></tr><tr><td>ModernNCA (tuned)</td><td>1385.8</td><td>342.0</td><td>334.2</td><td>10.85</td></tr><tr><td>CatBoost (tuned)</td><td>1383.4</td><td>287.3</td><td>301.1</td><td>11.18</td></tr><tr><td>LightGBM (tuned)</td><td>1364.5</td><td>341.4</td><td>250.8</td><td>11.74</td></tr><tr><td>RealMLP (tuned)</td><td>1308.6</td><td>318.5</td><td>230.9</td><td>11.64</td></tr><tr><td>XGBoost (tuned)</td><td>1289.7</td><td>324.0</td><td>315.4</td><td>12.32</td></tr><tr><td>RandomForest (tuned)</td><td>1239.0</td><td>546.8</td><td>365.2</td><td>13.80</td></tr><tr><td>MLP (tuned)</td><td>1071.8</td><td>226.8</td><td>239.0</td><td>14.10</td></tr></table>

Table 6: TALENT: classification ranking. All 17 selected methods on 200 datasets.
<table><tr><td>Method</td><td>Elo ↑</td><td>Elo+</td><td>Elo-</td><td>Average Rank ↓</td></tr><tr><td>TabFM</td><td>1829.8</td><td>306.9</td><td>408.4</td><td>4.11</td></tr><tr><td>RETRO</td><td>1719.4</td><td>312.6</td><td>299.9</td><td>5.89</td></tr><tr><td>TabICLv2</td><td>1715.2</td><td>316.5</td><td>362.1</td><td>5.93</td></tr><tr><td>EXAONE-Tabular</td><td>1708.3</td><td>238.5</td><td>255.4</td><td>5.53</td></tr><tr><td>TabPFN-3</td><td>1676.8</td><td>267.1</td><td>228.3</td><td>5.80</td></tr><tr><td>Xiaomi-TabLDM</td><td>1670.8</td><td>304.8</td><td>295.4</td><td>6.37</td></tr><tr><td>TabPFN-2.5</td><td>1633.4</td><td>308.0</td><td>290.4</td><td>6.72</td></tr><tr><td>TabPFN-2.6</td><td>1571.9</td><td>246.6</td><td>340.5</td><td>7.07</td></tr><tr><td>TabM (tuned)</td><td>1464.3</td><td>401.5</td><td>393.3</td><td>10.51</td></tr><tr><td>TabSwift</td><td>1417.8</td><td>272.4</td><td>203.6</td><td>9.60</td></tr><tr><td>CatBoost (tuned)</td><td>1408.6</td><td>325.2</td><td>279.8</td><td>11.37</td></tr><tr><td>LightGBM (tuned)</td><td>1363.1</td><td>362.6</td><td>375.2</td><td>11.69</td></tr><tr><td>ModernNCA (tuned)</td><td>1321.8</td><td>411.2</td><td>252.9</td><td>10.82</td></tr><tr><td>RealMLP (tuned)</td><td>1313.4</td><td>354.4</td><td>238.9</td><td>11.64</td></tr><tr><td>XGBoost (tuned)</td><td>1282.7</td><td>307.1</td><td>206.8</td><td>12.18</td></tr><tr><td>RandomForest (tuned)</td><td>1232.4</td><td>441.7</td><td>340.8</td><td>13.87</td></tr><tr><td>MLP (tuned)</td><td>1170.5</td><td>368.5</td><td>307.7</td><td>13.91</td></tr></table>

Table 7: TALENT: regression ranking. All 17 selected methods on 100 datasets.
<table><tr><td>Method</td><td>Elo ↑</td><td>Elo+</td><td>Elo-</td><td>Average Rank ↓</td></tr><tr><td>TabFM</td><td>1886.2</td><td>254.1</td><td>281.8</td><td>3.76</td></tr><tr><td>RETRO</td><td>1873.7</td><td>172.4</td><td>231.4</td><td>4.44</td></tr><tr><td>EXAONE-Tabular</td><td>1868.3</td><td>290.2</td><td>357.7</td><td>4.60</td></tr><tr><td>Xiaomi-TabLDM</td><td>1780.7</td><td>210.6</td><td>257.1</td><td>5.42</td></tr><tr><td>TabPFN-3</td><td>1729.9</td><td>315.9</td><td>461.8</td><td>5.45</td></tr><tr><td>TabICLv2</td><td>1666.1</td><td>293.8</td><td>274.2</td><td>6.80</td></tr><tr><td>TabPFN-2.6</td><td>1660.5</td><td>286.4</td><td>200.1</td><td>6.61</td></tr><tr><td>TabPFN-2.5</td><td>1592.5</td><td>215.0</td><td>257.3</td><td>7.58</td></tr><tr><td>TabSwift</td><td>1453.5</td><td>262.1</td><td>353.7</td><td>10.21</td></tr><tr><td>CatBoost (tuned)</td><td>1411.8</td><td>355.7</td><td>288.9</td><td>10.80</td></tr><tr><td>ModernNCA (tuned)</td><td>1337.2</td><td>542.5</td><td>253.3</td><td>10.92</td></tr><tr><td>LightGBM (tuned)</td><td>1309.7</td><td>378.9</td><td>285.7</td><td>11.84</td></tr><tr><td>RealMLP (tuned)</td><td>1259.6</td><td>269.9</td><td>256.3</td><td>11.63</td></tr><tr><td>TabM (tuned)</td><td>1255.7</td><td>395.0</td><td>271.3</td><td>12.21</td></tr><tr><td>XGBoost (tuned)</td><td>1228.7</td><td>352.1</td><td>310.8</td><td>12.59</td></tr><tr><td>RandomForest (tuned)</td><td>1131.7</td><td>452.2</td><td>240.8</td><td>13.68</td></tr><tr><td>MLP (tuned)</td><td>1054.4</td><td>413.7</td><td>220.3</td><td>14.46</td></tr></table>

## E.5 RELARENA

Tables 8–10 report the RelArena comparisons. Overall results pool all 21 tasks, while the two task-specific tables retain their respective subsets.

Table 8: RelArena: overall ranking. All 10 selected methods on 21 tasks.
<table><tr><td>Method</td><td>Elo ↑</td><td>Elo+</td><td>Elo-</td><td>Average Rank ↓</td></tr><tr><td>RETRO</td><td>1797.0</td><td>78.4</td><td>60.4</td><td>2.90</td></tr><tr><td>KurveRSC</td><td>1781.2</td><td>92.8</td><td>93.6</td><td>3.05</td></tr><tr><td>TabPFN-rel-local</td><td>1721.8</td><td>116.2</td><td>80.2</td><td>3.62</td></tr><tr><td>GraphSAGE</td><td>1666.4</td><td>90.8</td><td>78.6</td><td>4.19</td></tr><tr><td>RelGT</td><td>1575.2</td><td>126.8</td><td>103.3</td><td>5.17</td></tr><tr><td>RDBLearn</td><td>1555.1</td><td>89.6</td><td>93.1</td><td>5.38</td></tr><tr><td>RelGNN-ES</td><td>1521.4</td><td>90.6</td><td>81.6</td><td>5.74</td></tr><tr><td>LightGBM</td><td>1356.2</td><td>90.2</td><td>120.8</td><td>7.33</td></tr><tr><td>Constant (Per-entity)</td><td>1260.3</td><td>130.4</td><td>150.4</td><td>8.10</td></tr><tr><td>Constant (Global)</td><td>1000.0</td><td>70.9</td><td>130.1</td><td>9.52</td></tr></table>

Table 9: RelArena: classification ranking. All 10 selected methods on 12 tasks.
<table><tr><td>Method</td><td>Elo ↑</td><td>Elo+</td><td>Elo-</td><td>Average Rank ↓</td></tr><tr><td>RETRO</td><td>2185.9</td><td>266.1</td><td>45.3</td><td>3.08</td></tr><tr><td>KurveRSC</td><td>2151.6</td><td>340.0</td><td>111.5</td><td>3.42</td></tr><tr><td>TabPFN-rel-local</td><td>2135.2</td><td>315.9</td><td>122.3</td><td>3.58</td></tr><tr><td>GraphSAGE</td><td>2103.2</td><td>358.6</td><td>135.7</td><td>3.92</td></tr><tr><td>RelGT</td><td>1981.0</td><td>352.6</td><td>124.4</td><td>5.25</td></tr><tr><td>RelGNN-ES</td><td>1958.1</td><td>292.1</td><td>106.1</td><td>5.50</td></tr><tr><td>RDBLearn</td><td>1958.1</td><td>292.1</td><td>89.5</td><td>5.50</td></tr><tr><td>LightGBM</td><td>1770.2</td><td>250.6</td><td>106.4</td><td>7.33</td></tr><tr><td>Constant (Per-entity)</td><td>1754.8</td><td>331.8</td><td>185.6</td><td>7.46</td></tr><tr><td>Constant (Global)</td><td>1000.0</td><td>186.4</td><td>1952.8</td><td>9.96</td></tr></table>

Table 10: RelArena: regression ranking. All 10 selected methods on 9 tasks.
<table><tr><td>Method</td><td>Elo ↑</td><td>Elo+</td><td>Elo-</td><td>Average Rank ↓</td></tr><tr><td>KurveRSC</td><td>1736.4</td><td>321.6</td><td>143.4</td><td>2.56</td></tr><tr><td>RETRO</td><td>1722.3</td><td>302.3</td><td>130.4</td><td>2.67</td></tr><tr><td>TabPFN-rel-local</td><td>1609.0</td><td>247.2</td><td>169.5</td><td>3.67</td></tr><tr><td>GraphSAGE</td><td>1519.0</td><td>103.4</td><td>94.2</td><td>4.56</td></tr><tr><td>RelGT</td><td>1469.8</td><td>208.6</td><td>185.5</td><td>5.06</td></tr><tr><td>RDBLearn</td><td>1453.4</td><td>158.5</td><td>188.5</td><td>5.22</td></tr><tr><td>RelGNN-ES</td><td>1369.7</td><td>162.9</td><td>197.9</td><td>6.06</td></tr><tr><td>LightGBM</td><td>1228.9</td><td>103.2</td><td>211.6</td><td>7.33</td></tr><tr><td>Constant (Per-entity)</td><td>1000.0</td><td>184.3</td><td>353.1</td><td>8.94</td></tr><tr><td>Constant (Global)</td><td>1000.0</td><td>63.7</td><td>147.0</td><td>8.94</td></tr></table>