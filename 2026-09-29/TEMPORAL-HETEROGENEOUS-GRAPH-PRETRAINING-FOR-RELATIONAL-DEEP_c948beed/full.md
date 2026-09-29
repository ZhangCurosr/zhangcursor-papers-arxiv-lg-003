# TEMPORAL HETEROGENEOUS GRAPH PRETRAINING FOR RELATIONAL DEEP LEARNING

Yixin Peng<sup>1⋆</sup>, Er Jin<sup>1</sup>, Diego Collarana<sup>2</sup>, and Stefan Decker<sup>1,2</sup>

<sup>1</sup> RWTH Aachen University, Aachen, Germany

{peng,jin,decker}@dbis.rwth-aachen.de

<sup>2</sup> Fraunhofer FIT, Sankt Augustin, Germany

diego.collarana.vargas@fit.fraunhofer.de

## ABSTRACT

Relational deep learning represents database rows and foreign-key links as a heterogeneous graph, enabling prediction from record attributes and relational context. Two forms of temporal signals play distinct roles in these graphs: record age changes as the prediction cutoff advances, whereas the interval between two observed records remains fixed. Prior work has explored temporal modeling and temporal pretraining for heterogeneous graphs, but typically treats time as a single source of information or focuses on either representation or supervision in isolation. We investigate how explicitly representing both temporal signals affects the benefits of temporal pretraining on downstream tasks. Our framework pairs two complementary temporal encodings: Multi-scale Time Encoding, which captures record age through learnable time scales and type-specific projections, and Rotary Time Encoding, which encodes signed time differences between linked records through rotary transformations during graph propagation. Rather than treating these encodings as architectural additions alone, we train them through three self-supervised objectives: recovering historical relations, forecasting horizon-dependent future relation activity, and contrasting temporally valid historical subgraphs. All model inputs are restricted to information available at their observation cutoffs. We further structure pretraining into two stages: learning neighborhood representations via subgraph contrastive learning, followed by refining these representations through either relation recovery or future activity prediction. We evaluate across five Rel-Bench datasets and 11 classification and regression tasks using representative heterogeneous graph neural network (GNN) and Graph Transformer (GT) backbones. With both temporal encodings, the best evaluated staged schedules for the two backbones outperform direct supervised training using the same encodings by 3.02% and 1.06%, respectively, and controls without pretraining or encodings by 3.24% and 2.37%, respectively.

## 1 INTRODUCTION

Relational databases store records in linked tables, with predictive signals distributed across both record attributes and relationships. For example, predicting whether a customer will buy again may require combining customer details, product information, and purchase times from several tables. Conventional pipelines assemble these signals through task-specific joins and aggregations (Kanter & Veeramachaneni, 2015). Relational deep learning (RDL) instead represents rows as nodes and foreign-key links as typed edges, allowing graph models to learn directly from attributes and relational neighborhoods (Fey et al., 2024).

Learning from relational graphs requires modeling both heterogeneity and time. Customer and product records describe distinct entity types, while purchase records capture interactions between them. Together, they form a heterogeneous graph with type-specific node attributes and multiple relation types. Time provides two complementary signals about these records and their relationships. Record age measures the time elapsed between a record’s observation and the prediction cutoff. A signed inter-record interval measures the difference between the observation times of two linked records, capturing their temporal order and distance. For example, as the prediction cutoff advances, a purchase record becomes older, while the interval between the customer’s registration and that purchase remains fixed.

Temporal encodings make record age and inter-record intervals explicit in model representations (Xu et al., 2020; Hu et al., 2020c; Yu et al., 2023; Cong et al., 2023; Li et al., 2023; Robinson et al., 2024; Dwivedi et al., 2026). Pretraining guides models to integrate these temporal signals with record attributes and graph structure before adaptation to downstream tasks. Existing pretraining approaches use heterogeneous reconstruction and relation- or schema-based contrastive learning (Hu et al., 2020b; Jiang et al., 2021; Sun et al., 2025), while temporal objectives further target interaction timing or statistics of future records (Ma et al., 2026; Truong et al., 2026). Temporal encodings and pretraining objectives thus serve complementary roles: the former specify how time is represented, and the latter guide what and how the model learns from these representations. This motivates us to study how temporal heterogeneity-aware pretraining can be combined with explicit encodings of both temporal signals. Figure 2 summarizes these design choices across prior methods and our framework.

We address this challenge with a framework that integrates temporal encodings with staged pretraining. Our temporal encodings capture both record age and relative timing: Multi-scale time encoding (TIMEMIX) incorporates record-age information into node inputs through learnable time scales and type-specific projections, while Rotary time encoding (TIMEROPE) adapts prior temporal rotary attention (Peng et al., 2026) to use bounded, signed log-intervals and extends it to GNN message construction.

Building on these encodings, we introduce three complementary pretraining objectives. Temporal Subgraph Contrast (TEMP-SUB) aligns corrupted views of historical neighborhoods through typeaware pooling and relation-preserving augmentations, encouraging representations to remain stable under missing attributes or links. Historical Relation Recovery (REL-HIST) recovers hidden typed links using only the context available when those links were observed, whereas Horizon-aware Relation Activity Prediction (REL-FUTURE) predicts whether an entity will receive links of a specified relation and how many events will occur within a future window. We organize these objectives into a staged pretraining scheme: TEMP-SUB first learns neighborhood representations, which are then refined using either REL-HIST or REL-FUTURE. All model inputs respect their observation cutoffs throughout pretraining. The resulting pretrained model is subsequently fine-tuned for individual tasks within the same database.

We evaluate our framework on five RelBench databases (Robinson et al., 2024) across 11 tasks spanning binary classification, multiclass classification, and regression using representative heterogeneous GNN and GT backbones. Controlled ablations assess the complementary benefits of the temporal encodings and the effectiveness of staged pretraining (Figure 4).

Our contributions are:

• Complementary temporal representations. We establish that jointly encoding record age and inter-record intervals improves aggregate performance over either alone on both backbones, with record age contributing more in component ablations.

• Effective staged pretraining. Our two staged schedules yield positive aggregate transfer in all four backbone–encoding settings. With both encodings, the best schedule on each backbone outperforms either constituent objective alone, demonstrating the value of staging complementary supervision.

• Gains beyond temporal encoding alone. Matched comparisons establish additional mean type gains of 3.02% on the GNN and 1.06% on the GT over supervised training with both encodings. Our best schedules also outperform the evaluated pretraining baselines, including their staged variants.

![](images/063b515ed02eadb5f3dfcfaccbeee03d291e81c9d34c015dbfd39ddb92ff99aa.jpg)  
Figure 1: From relational tables to historical subgraphs for RDL. (a) Customer, purchase, and product tables: id denotes primary keys, c id and $_ { \mathrm { p . i d } }$ denote foreign keys, and time records purchase observation times. (b) Rows become nodes whose types follow their source tables; foreignkey references become typed edges with corresponding reverse relations for bidirectional message passing. Purchase nodes carry timestamps $t _ { v } . \ ( \mathrm { c } )$ An illustrative two-hop sample around seed C1, using only information available by cutoff $\tau = \mathrm { d a y }$ 5. S4 is excluded because its timestamp exceeds τ; S3 and C2 lie outside the two-hop neighborhood.

## 2 BACKGROUND AND RELATED WORK

## 2.1 RELATIONAL DEEP LEARNING

RDL provides an end-to-end framework for learning from linked database records (Fey et al., 2024; Dwivedi et al., 2025). It combines multimodal attributes, schema-defined relationships, and temporal observations in a graph representation. RelBench supplies predictive tasks with chronological splits (Robinson et al., 2024).

Definitions. Let $\mathcal { D } = ( \mathcal { T } , \mathcal { R } )$ be a database with tables $\mathcal { T } = \{ T _ { 1 } , \ldots , T _ { n } \}$ and foreign-key relations R. Each row is identified by a primary key and contains attributes and, where applicable, foreign keys referencing other rows. A timestamp, when present, records its observation time. Following the relational entity graph formulation (Fey et al., 2024), we construct $G = ( V , E , \phi , \psi , t )$ , where V contains the rows, E contains their foreign-key links, $\phi : V  \tau$ assigns node types by source table, and $\psi : E  \mathcal { R }$ assigns relation types. The timestamp function t is defined on timestamped nodes, with $t _ { v } = t ( v )$ . We include reverse edges and corresponding relation types in E and R for message passing. Figure 1(a,b) illustrates this mapping: customer, purchase, and product tables become distinct node types, and the two foreign keys in each purchase define its links to a customer and a product.

## 2.2 GRAPH MODELS AND TEMPORAL ENCODING FOR RDL

We consider two encoder families for RDL: GNNs based on relational message aggregation, including HETEROGNN and RelGNN (Robinson et al., 2024; Chen et al., 2025), and GTs based on heterogeneous attention, including HGT, RelGT, and Dual-Branch Temporal Heterogeneous Graph Fusion Model (THGFM) (Hu et al., 2020c; Dwivedi et al., 2026; Peng et al., 2026).

Viewed in the RDL setting, existing methods incorporate record age through temporal features added to node inputs or included in attention and sequence representations. These features use positional encodings with trainable projections in HETEROGNN, learnable trigonometric functions in TGAT, THAN, and DyGFormer, and fixed-frequency cosine functions in GraphMixer (Robinson et al., 2024; Xu et al., 2020; Li et al., 2023; Yu et al., 2023; Cong et al., 2023). RelGT incorporates seed-relative age features during record tokenization (Dwivedi et al., 2026). For inter-record intervals, HGT adds projected sinusoidal encodings of source–target time differences to source representations, while THGFM applies interval-dependent query–key rotations (Hu et al., 2020c; Peng et al., 2026). Additionally, record-age distributions and time scales vary across the evaluated RelBench databases (Appendix B.2). Motivated by these observations and the complementary roles of these two forms of temporal information, we develop two encodings and integrate both into the GNN and GT backbones.

<table><tr><td rowspan="2">Pretraining strategy</td><td colspan="3">Pretraining granularity</td><td colspan="2">Temporal encoding</td><td rowspan="2">Schedule Staged</td><td rowspan="2">Evaluated graph models</td></tr><tr><td>Node;</td><td>Relation; Heterogeneity Heterogeneity Heterogeneity</td><td>(Sub-)Graph;</td><td>Recency</td><td>Interval</td></tr><tr><td>TVE</td><td>●: ●</td><td>- ;</td><td>0;0</td><td>B</td><td></td><td></td><td>GNN</td></tr><tr><td>TGPM</td><td></td><td></td><td></td><td></td><td></td><td></td><td>GT</td></tr><tr><td>CTRL</td><td></td><td>●; 0</td><td></td><td></td><td></td><td></td><td>GNN</td></tr><tr><td>Un-SAGE</td><td></td><td>o;</td><td></td><td></td><td></td><td></td><td>GNN</td></tr><tr><td>AttrMasking</td><td></td><td></td><td></td><td></td><td></td><td></td><td>GNN</td></tr><tr><td>ContextPred</td><td></td><td></td><td>●</td><td></td><td></td><td></td><td>GNN</td></tr><tr><td>GPT-GNN</td><td></td><td></td><td></td><td></td><td>B</td><td></td><td>GNN &amp; GT</td></tr><tr><td>PHE</td><td></td><td></td><td>0;0</td><td></td><td>B</td><td></td><td>GNN &amp; GT</td></tr><tr><td>PT-HGNN</td><td>●; ●</td><td>●; ●</td><td>●; •</td><td></td><td>B</td><td></td><td>GT (HGT)</td></tr><tr><td>Our staged schedules</td><td>•; •</td><td>•;•</td><td>•; •</td><td>●</td><td>●</td><td>●</td><td>GNN &amp; GT</td></tr><tr><td colspan="2">● Direct Indirect/limited coverage</td><td colspan="2">B Backbone-dependent</td><td colspan="2">Not explicit</td><td colspan="2"></td></tr></table>

Figure 2: Comparison of representative graph pretraining strategies. In each of the first three columns, paired symbols indicate coverage of the supervision granularity and explicit use of node or relation types at that granularity, respectively. Filled and half-filled circles denote direct and indirect or limited coverage; B denotes backbone-dependent encoding, and a dash denotes no explicit coverage.

## 2.3 GRAPH PRETRAINING: HETEROGENEITY, TIME, AND STAGING

Graph pretraining provides supervision over nodes, relations, and subgraphs, but coverage of these granularities does not by itself imply explicit use of heterogeneous types or schema (Figure 2). AttrMasking and GraphMAE learn through attribute reconstruction (Hu et al., 2020a; Hou et al., 2022; 2023); Un-SAGE and ContextPred learn from proximity and neighborhood context (Hamilton et al., 2017; Hu et al., 2020a), while GCC and GraphCL contrast subgraph or graph views (Qiu et al., 2020; You et al., 2020). Heterogeneous pretraining makes schema structure explicit in the supervision: GPT-GNN uses type-specific attribute and link generation, PT-HGNN combines relation-wise sampling with schema-instance agreement, and PHE incorporates type-aware schema representations and semantic neighbors (Hu et al., 2020b; Jiang et al., 2021; Sun et al., 2025). HGMAE and MUG alternatively use metapaths in reconstruction or alignment (Tian et al., 2023; Shan et al., 2026). For RDL, this distinction matters because table and relation types determine which entities and links a pretraining target describe. These methods establish ways to learn from typed structure; temporal encoding and temporally defined targets introduce additional design choices.

Time can condition the representation of observed history or define what the model must predict. TGPM combines masked patch reconstruction with next-time prediction (Ma et al., 2026), while CTRL models temporal influence through heterogeneous event-subgraph and edge prediction (Li et al., 2026). In relational databases, Griffin uses completion pretraining and supervised training, while RelPrism constructs attribute-based pseudo-tasks (Wang et al., 2025; Yang et al., 2026). TVE directly targets future relational information: it predicts statistics, including counts, of records reached through schema traversals (Truong et al., 2026). These targets differ in what they predict: interaction timing, event structure, or aggregated relational statistics.

How these objectives are organized is a further question. Hu et al. (2020a) stage node-level and graph-level pretraining, whereas TGPM jointly trains temporal objectives and TVE supports weighted combinations of losses (Ma et al., 2026; Truong et al., 2026). Our schedules study a different transi tion: first learn stable historical representations through contrastive learning, then refine them through relation recovery or future activity prediction, evaluating these schedules with explicit temporal encodings across heterogeneous GNN and GT backbones. Together, seed-node and historical-subgraph contrast, typed relation recovery, and activity prediction cover node, relation, and subgraph supervision, with heterogeneous types explicitly used in pooling, sampling, decoding, and augmentation.

## 3 METHOD

For seed node s and prediction cutoff τ, let $G _ { s , \leq \tau }$ denote a sampled neighborhood containing only information available by τ. We refer to this input as a historical neighborhood or subgraph. We use s to identify its seed and u, v to denote general nodes or edge endpoints. Temporal encoding and graph propagation operate on nodes and edges throughout the sampled neighborhood. In Figure 1(c), S4 is excluded by time, while S3 lies outside C1’s two-hop neighborhood. Static tables contribute attributes available at the cutoff even when they have no timestamp column. The model produces the seed representation $h _ { s } = f _ { \theta } ( G _ { s , \leq \tau } ) \in \mathbb { R } ^ { d }$ , where d is the hidden dimension.

## 3.1 TEMPORAL ENCODING

Record age: TIMEMIX. TIMEMIX represents how old a record is at the prediction cutoff, allowing the model to learn how recency matters for each table type. For each timestamped node v in the sampled neighborhood, we compute its age relative to the same cutoff $\tau { : }$

$$
a _ { v } = \operatorname* { m a x } ( 0 , \tau - t _ { v } ) , \qquad \xi _ { v , k } = \log \left( 1 + \frac { a _ { v } } { s _ { k } } \right) ,\tag{1}
$$

where $\boldsymbol { a } _ { v }$ is the record age and $\xi _ { v , k }$ is its feature at learned positive time scale $s _ { k } .$ , for $k = 1 , \ldots , K _ { \mathrm { t i m e } }$ We parameterize $s _ { k } = \mathrm { c l i p } ( \exp ( \eta _ { k } ) , s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } )$ , where $\eta _ { k } \in \mathbb { R }$ is trainable and $0 < s _ { \mathrm { m i n } } < s$ max are fixed bounds. Here $K _ { \mathrm { t i m e } }$ is the number of scales, and ages and scales use the same time unit. The logarithm preserves variation among recent records while compressing large ages. Future records are excluded by sampling before this encoding is applied.

Let $x _ { v } \in \mathbb { R } ^ { d }$ be the table encoder’s representation of record attributes. For node type $a = \phi ( v )$ , we add a mixture of projected age features:

$$
e _ { v } ^ { \mathrm { t i m e } } = \sum _ { k = 1 } ^ { K _ { \mathrm { t i m e } } } \pi _ { k } g _ { a , k } ( \xi _ { v , k } ) , \qquad \widetilde { x } _ { v } = x _ { v } + e _ { v } ^ { \mathrm { t i m e } } .\tag{2}
$$

Here $g _ { a , k } : \mathbb { R } \to \mathbb { R } ^ { d }$ is a learned linear projection for type a and scale $k ,$ and $\pi _ { k }$ is a learned nonnegative mixture weight, with $\textstyle \sum _ { k } \pi _ { k } = 1$ . The time embedding $e _ { v } ^ { \mathrm { t i m e } }$ augments the attribute representation, giving $\widetilde { x } _ { v }$ as the input to graph propagation. Scales and weights are shared across node types; type-specific projections allow the same age to have different effects in different tables. For records without timestamps, we use $\widetilde { x } _ { v } = x _ { v }$ . Appendix D gives more details.

Time differences: TIMEROPE. TIMEROPE captures the order and spacing of linked records. For a directed edge $u  v$ in the sampled neighborhood, u is the source and v is the destination. When both endpoints have timestamps, define $\Delta t _ { u v } = t _ { v } - t _ { u }$ as the signed time difference. We convert it to a rotation phase:

$$
\rho _ { u v } = \mathrm { c l i p } \left( \mathrm { s i g n } ( \Delta t _ { u v } ) \log \left( 1 + \frac { | \Delta t _ { u v } | } { s _ { \mathrm { R } } } \right) , - \rho _ { \mathrm { m a x } } , \rho _ { \mathrm { m a x } } \right) ,\tag{3}
$$

where $s _ { \mathrm { R } }$ is a fixed positive time scale and $\rho _ { \mathrm { m a x } } > 0$ is a fixed hyperparameter that bounds the magnitude of the rotation phase. The sign distinguishes temporal order; the logarithm and clipping limit the effect of long gaps. Edges with an untimestamped endpoint use the backbone’s message or attention operation without rotation.

For each node $v ,$ let $y _ { v } \in \mathbb { R } ^ { d }$ denote its input to the current graph layer; in the first layer, $y _ { v } = \widetilde { x } _ { v }$ Let $R ( \rho )$ rotate the j-th consecutive pair of feature dimensions by angle $\omega _ { j } \rho$ , where $\omega _ { j }$ is the corresponding RoPE frequency (Su et al., 2023). Our GNN backbone constructs messages using opposite half-phase rotations and aggregates them by relation:

$$
\begin{array} { l } { { \displaystyle m _ { u \to v } = \frac { 1 } { 2 } \left[ R ( - \rho _ { u v } / 2 ) y _ { u } + R ( + \rho _ { u v } / 2 ) y _ { v } \right] } , \ ~ } \\ { { \displaystyle h _ { v } ^ { ( r ) } = W _ { \mathrm { n e i g h } } ^ { ( r ) } \left( \frac { 1 } { | \mathcal { N } _ { r } ( v ) | } \sum _ { u \in \mathcal { N } _ { r } ( v ) } m _ { u \to v } \right) + W _ { \mathrm { s e l f } } ^ { ( r ) } y _ { v } } . } \end{array}\tag{4}
$$

Here $\mathcal { N } _ { r } ( v )$ contains incoming neighbors under relation r, and $h _ { v } ^ { ( r ) }$ is the relation’s output. The learned matrices $W _ { \mathrm { n e i g h } } ^ { ( r ) }$ and $W _ { \mathrm { s e l f } } ^ { ( r ) }$ transform averaged messages and node inputs, respectively.

For our GT backbone, we adapt the query–key rotation mechanism of THGFM’s Rotary Temporal Attention (Peng et al., 2026), using the bounded, signed log-interval $\rho _ { u v }$ defined above as the temporal phase. Thus, TIMEROPE incorporates inter-record timing into GNN message construction and GT attention weighting. Appendix D provides the rotation convention and backbone implementation details.

## 3.2 TEMPORAL PRETRAINING OBJECTIVES

Historical link recovery: REL-HIST. REL-HIST asks whether two records were linked, given the context available when the link was observed. We sample an observed link $u  v$ of relation r and use its assigned observation time as cutoff τ. We then sample $K _ { \mathrm { n e g } }$ negative destinations of the same type as v that are available at τ and are not known destinations of u under r in the pretraining data.

We encode historical neighborhoods centered on the source and on each candidate destination at the same cutoff τ, obtaining their node representations. We remove the target edge and its reverse from every input so that the encoder cannot directly observe the answer. For endpoint representations $h _ { u }$ and $h _ { v }$ , a relation-specific bilinear decoder assigns the score $s _ { \mathrm { h i s t } } ( u , v , r ) \stackrel { \textstyle } { = } h _ { u } ^ { \top } \bar { W } _ { r } h _ { v }$ , where $W _ { r } \in \mathbb { R } ^ { d \times d }$ is a learned matrix for relation r. The mean softplus ranking loss $\mathcal { L } _ { \mathrm { { R e l - H i s t } } }$ encourages the observed destination to score above the sampled negatives. Appendix E gives the timestamp assignment and full loss.

Future relation activity: REL-FUTURE. REL-FUTURE predicts whether a seed node s will receive links of relation r in a future window and how many. Each forward foreign-key link $e = ( u \to s )$ represents an event, with time $t _ { e } = t _ { u }$ given by the source record’s timestamp. Let $\mathcal { E } _ { r } ( s )$ be the set of these events. For cutoff τ and horizon δ, we define the event count and prediction targets as

$$
\begin{array} { c } { { c _ { r } ( s ; \tau , \delta ) = \displaystyle \sum _ { e \in \mathcal { E } _ { r } ( s ) } \mathbf { 1 } \{ \tau < t _ { e } \le \tau + \delta \} , } } \\ { { \ } } \\ { { y _ { \mathrm { a c t } } = \displaystyle \mathbf { 1 } \{ c _ { r } ( s ; \tau , \delta ) > 0 \} , \qquad y _ { \mathrm { c o u n t } } = \log \ ( 1 + c _ { r } ( s ; \tau , \delta ) ) . } } \end{array}\tag{5}
$$

To construct a positive example, we uniformly sample an eligible relation and then an eligible event of that relation. Let $s ^ { + }$ be its destination and $t _ { \mathrm { e v t } }$ its time. We choose a candidate horizon δ and set $\tau = t _ { \mathrm { e v t } } - \delta _ { \mathrm { { \mathrm { t } } } }$ , requiring $t _ { \mathrm { f i r s t } } ( s ^ { + } ) \leq \tau$ , where $t _ { \mathrm { f i r s t } }$ denotes the entity’s first observation time. This ensures that the seed is available at the cutoff and the window contains at least the sampled event. For the same $( r , \tau , \delta )$ , we sample negative seeds s<sup>−</sup> of the same type as $s ^ { + }$ that are available at τ and have no events in the window.

For each seed $s ,$ the encoder uses only $G _ { s , \leq \tau }$ to compute $h _ { s }$ . Two decoders, conditioned on r and δ, map $h _ { \varepsilon }$ to an occurrence logit and a nonnegative log-count prediction. We optimize

$$
\mathcal { L } _ { \mathrm { R e l - F u t u r e } } = \mathcal { L } _ { \mathrm { a c t } } + \lambda _ { \mathrm { c o u n t } } \mathcal { L } _ { \mathrm { c o u n t } } ,\tag{6}
$$

where $\mathcal { L } _ { \mathrm { a c t } }$ is the binary occurrence loss, ${ \mathcal { L } } _ { \mathrm { c o u n t } }$ is the Smooth L1 loss on log-count targets, and $\lambda _ { \mathrm { c o u n t } }$ weights the count loss. Each loss gives equal weight to the positive example and the mean over negatives. Appendix E provides the timestamp assignment, sampling rules, and full losses.

Temporal subgraph contrast: TEMP-SUB. TEMP-SUB learns representations that remain consistent when some features or links are missing from a historical neighborhood. Let $\tau _ { \mathrm { p r e } }$ be the global pretraining cutoff, set to the dataset’s validation timestamp. For timestamped seeds, we require $t _ { s } \leq \tau _ { \mathrm { p r e } }$ and set $\tau = t _ { s } ;$ for seeds without timestamps, we set $\tau = \tau _ { \mathrm { p r e } } .$ . For each input $\textit { G } _ { s , \leq \tau }$ , we create two independently corrupted views sharing the same seed node s and cutoff τ by randomly masking input embedding entries and dropping edges. We either keep or drop each edge together with its corresponding reverse edge, and each view retains at least one edge per represented forward relation.

To summarize each view, we first pool nodes within each type, then combine the type summaries. Let H be an augmented subgraph, ${ \mathsf { \mathcal { T } } } ( H )$ its set of node types, and $V _ { a } ( H )$ its nodes of type a. We compute

$$
\mu _ { H , a } = \frac { 1 } { | V _ { a } ( H ) | } \sum _ { u \in V _ { a } ( H ) } h _ { u } , \qquad c _ { H , a } = \operatorname { t a n h } ( P _ { a } \mu _ { H , a } + e _ { a } ) ,\tag{7}
$$

where $h _ { u }$ is computed within $H , \mu _ { H , a }$ is the mean representation of type $^ { a , }$ and $c _ { H , a }$ is its type summary. The learned projection $P _ { a } \in \mathbb { R } ^ { d \times d }$ and type embedding $e _ { a } \in \mathbb { R } ^ { d }$ adapt the summary to its table type. A shared learned linear scoring function $g _ { \mathrm { p o o l } } : \mathbb { R } ^ { d } \overset { - } {  }$ R then assigns attention weights $\beta _ { H , a }$ to combine the summaries into $z _ { H } \colon$

$$
\beta _ { H , a } = \frac { \exp \bigl ( g _ { \mathrm { p o o l } } \bigl ( c _ { H , a } \bigr ) \bigr ) } { \sum _ { b \in \mathcal { T } ( H ) } \exp \bigl ( g _ { \mathrm { p o o l } } \bigl ( c _ { H , b } \bigr ) \bigr ) } , \qquad z _ { H } = \sum _ { a \in \mathcal { T } ( H ) } \beta _ { H , a } c _ { H , a } .\tag{8}
$$

Averaging within types prevents larger tables from dominating solely through their node counts, while attention learns how much each type contributes to the neighborhood representation.

We apply contrastive learning at two levels: the whole neighborhood and its seed node s. For neighborhood-level contrast, we map the pooled representations of both views through the same projection head. Each neighborhood in the first view is trained to match its counterpart in the second view, with other neighborhoods in the batch serving as negatives. We compute an InfoNCE loss (van den Oord et al., 2019) using cosine similarity divided by temperature T. We then repeat the matching from the second view to the first and average the two losses to obtain $\mathcal { L } _ { \mathrm { g r a p h } }$ . For seed-level contrast, we take the representation of s from each view and apply the same matching procedure through a separate projection head, obtaining $\mathcal { L } _ { \mathrm { s e e d } }$ . To complement contrastive learning in the projection space, we additionally regularize the seed representations with a VICReg penalty (Bardes et al., 2022) that discourages collapse by maintaining variance across seeds and reduces redundancy across feature dimensions. This penalty acts directly on the encoder outputs before the seed projection head and is averaged over the two views, giving ${ \dot { \mathcal { R } } } _ { \mathrm { r e g } } .$ The full objective is

$$
{ \mathcal { L } } _ { \mathrm { T e m p - S u b } } = { \mathcal { L } } _ { \mathrm { g r a p h } } + { \mathcal { L } } _ { \mathrm { s e e d } } + { \mathcal { R } } _ { \mathrm { r e g } } .\tag{9}
$$

Appendix C and Appendix E provide more details.

## 4 EXPERIMENTS AND RESULTS

## 4.1 EXPERIMENTAL SETUP

Datasets and evaluation. We evaluate on 11 tasks from five RelBench databases, Arxiv, Avito, F1, H&M, and Event (Robinson et al., 2024), using the original chronological training, validation, and test splits. We report accuracy for multiclass classification, ROC–AUC for binary classification, and MAE for regression. All experiments are repeated with four random seeds, {42, 43, 44, 45}; reported task metrics are the means and standard deviations across these runs, with each run evaluated at it best validation checkpoint. Appendix A provides dataset and task details.

Backbones and baselines. We benchmark five backbones: two GNNs, HETEROGNN and RelGNN, and three GTs, HGT, RelGT, and THGFM. The controlled encoding and pretraining study focuses on HETEROGNN and THGFM, selected as the best-performing representatives of their respective families by mean validation rank across the 11 tasks (Appendix B.1). Selecting one backbone per family is a trade-off between architectural coverage and computational cost. All backbones use three layers with hidden dimension 128, and GTs use eight attention heads. We compare against nine pretraining baselines: ATTRMASKING, CONTEXTPRED, Un-SAGE, GPT-GNN, PT-HGNN, PHE, CTRL, TGPM, and TVE. Single-stage pretraining and downstream fine-tuning each use 10 epochs with 10,000 steps per epoch. Two-stage pretraining uses 10 epochs with 5,000 steps per epoch in each stage, matching the total single-stage pretraining budget of 100,000 steps. Models trained without pretraining use the same downstream budget. Appendix C provides the full architecture and training configuration.

## 4.2 MAIN RESULTS

Aggregate performance across backbones. We denote the fixed day-based single-scale recordage encoding used by HETEROGNN (Robinson et al., 2024) as DAYPE and use it as the baseline for multi-scale TIMEMIX on both backbones (Appendix B.2). Figure 3 uses the same-backbone supervised control with only DAYPE and without TIMEROPE or pretraining as its reference (A0) throughout, measuring the effects of pretraining and, where enabled, our temporal encodings. In each of the four backbone–encoding settings, one of our two proposed staged schedules achieves the highest mean type gain among the evaluated methods. With both encodings, TS→REL-HIST leads on the GNN at +3.24%, and TS→REL-FUTURE leads on the GT at +2.37%. The strongest baselines, including TS-initialized variants, reach +2.96% on the GNN (ATTRMASKING) and +1.15% on the GT (TS→PT-HGNN). Without our encodings, TS→REL-FUTURE leads on the GNN (+2.51%), while TS→REL-HIST leads on the GT (+1.58%).

Performance under task-balanced aggregation. The benchmark contains one multiclass, six binary, and four regression tasks, so equal weighting of task types gives individual tasks different weights. We therefore also report average ranks with all 11 tasks weighted equally. Our staged schedules rank best in all four panels, with average ranks of 4.73, 7.14, 4.95, and 6.45, respectively. TS→REL-HIST leads on the GNN with both encodings, and TS→REL-FUTURE leads in the other three panels; with both encodings, our two staged schedules occupy the top two positions on each backbone. Thus, the aggregate advantage is supported by both gain and rank summaries.

![](images/24956411cae16168c50727b8291f83a016700b32b21c095e806c0412b30d2e2b.jpg)  
Figure 3: Relative test-performance gains (%) across 11 tasks, computed from task metrics averaged over four random seeds. All panels use the same-backbone supervised control without our time encodings (only with DayPE) or pretraining as reference (A0). Mean type gain averages gains within each task type and then equally across types; average rank weights all tasks equally (lower is better). Colors saturate at ±10%; white dots mark values beyond this range. TS denotes TEMP-SUB; outlined rows identify our two-stage schedules, and bold summary values mark the best result in each panel.

Task coverage and sources of improvement. With both encodings, the best staged configuration for each backbone improves over A0 on 8 of the 11 tasks for the GNN and 7 for the GT, including all four regression tasks on both backbones. Regression provides the largest task-type gains: +5.12% on the GNN and +8.55% on the GT. On the latter, gains span driver position (+8.74%), item sales (+7.54%), and ad CTR (+17.76%). Classification gains are less uniform: the GNN schedule improves author-category accuracy by 3.40% and binary classification by 1.21% on average; the GT schedule gains 0.26% on binary classification but loses 1.70% on author-category accuracy. The positive aggregates therefore reflect benefits across several regression targets alongside more variable classification transfer. Appendix B.3 reports detailed test results and standard deviations.

## 4.3 ABLATION STUDIES

Contributions of the temporal encodings. Removing TIMEMIX restores DAYPE, the single-scale baseline introduced in Section 4.2, regardless of whether TIMEROPE is enabled. In Figure 4a, removing TIMEMIX reduces mean type gain by 0.59 percentage points on the GNN and 1.18 on the GT, compared with 0.28 and 0.46 points from disabling TIMEROPE. Removing both lowers gains by 0.46 and 1.29 points, respectively. The combined configuration therefore achieves the highest mean type gain on both backbones, with removing TIMEMIX causing the larger decrease. Effects vary across task types: disabling TIMEROPE slightly improves multiclass performance but reduces binary and regression gains on both backbones.

Effect of staged pretraining. Figure 4b measures pretraining gains over supervised controls with the same backbone and temporal encodings. Both proposed schedules yield positive mean type gains in all four settings. With both encodings, TS→REL-HIST achieves +3.02% on the GNN, while TS→REL-FUTURE achieves +1.06% on the GT, demonstrating benefits beyond temporal encoding alone. Under the same total pretraining budget, adding an initial TEMP-SUB stage improves over the corresponding single objective in seven of the eight comparisons for REL-HIST and REL-FUTURE. Notably, on the GT with both encodings, staging turns negative transfer into positive gains for both objectives, whereas all six TS-initialized baselines shown remain below the supervised control. The benefit nevertheless depends on the configuration: REL-FUTURE alone slightly outperforms its staged counterpart on the GNN with both encodings (+3.00% versus +2.83%). Among the baselines, staging improves PT-HGNN, CTRL, TGPM and TVE in all four settings, while its effects on AttrMasking and ContextPred are mixed. One possible explanation is that the typeaware representations learned by TEMP-SUB better support relation- and schema-based supervision (Figure 2).

![](images/63dd037d1bbd52b419b61f3a488c27b25e6580b9732faadbd02004527d984334.jpg)  
Figure 4: Contributions of temporal encoding and staged pretraining. (a) Gain differences from ours (TIMEMIX+TIMEROPE), without pretraining. DAYPE is the single-scale baseline for TIMEMIX. (b) Pretraining gains over matched supervised controls (A0: DAYPE; B0: TIMEMIX+TIMEROPE). Arrows connect single- and two-stage results with equal total pretraining steps.

## 4.4 DISCUSSION AND LIMITATIONS

Temporal scales and horizon design. The statistics in Appendix B.2 reveal substantial variation in record-age distributions across databases. Avito includes records observed within one day before the prediction cutoff, with age variation at second, minute, and hour scales. Record ages in Event and H&M are concentrated at scales of days to months, whereas Arxiv and F1 include records dating back years. These differences motivate the multi-scale record-age representation in TIMEMIX.

The supported prediction horizons also vary across databases. Avito supports only 1- and 7-day horizons, while Event and H&M also support windows spanning weeks to months. In Arxiv and F1, 67.7% and 87.1% of eligible events, respectively, support a 365-day horizon. These differences motivate using multiple horizons for REL-FUTURE: short windows provide training examples where history is limited, while longer windows provide supervision on relation activity over months or a year where sufficient history is available. We therefore use candidate horizons from one day to one year.

Limitations. Our framework focuses on within-database training; transfer to unseen databases remains to be explored. Single- and two-stage pretraining use the same total number of training steps, although computational cost per step can vary across objectives. Comparisons under matched computational costs and with alternative stage orders would further clarify the contributions of objective composition and stage order.

## 5 CONCLUSION

We present an RDL framework that encodes record age and inter-record intervals and stages subgraph contrast before historical relation recovery or future activity prediction. Across five RelBench databases and 11 tasks, the two encodings together outperform either alone in aggregate on both backbones, and staged pretraining yields additional gains over supervised controls with the same encodings on both GNN and GT backbones. These results support combining explicit temporal representations with staged self-supervision for within-database prediction. Generalization to unseen databases and comparisons of stage orders under matched computational budgets remain directions for future work.

## AI USE STATEMENT

Generative AI tools assisted with language editing and polishing. The authors are responsible for the final content, claims, and artifacts in this work.

## REPRODUCIBILITY STATEMENT

To facilitate reproducibility, Section 2.1 specifies the graph construction and observation rule, and Section 3 describes the encodings, objectives, and training schedule. Appendix A documents database statistics, task definitions, and temporal splits; Appendix B reports task-level metrics and their variation; Appendix C specifies the architecture, optimization, and four-seed evaluation settings; Appendix D provides temporal implementation details; Appendix E specifies the decoders, losses, and sampling details for the pretraining objectives; and Appendix F presents the optimization curves. The appendix will be provided as a separate PDF in the Supplementary Materials. All code needed to reproduce the reported results will be provided as supplementary code in the Supplementary Materials, including instructions for downloading and processing the publicly available RelBench datasets (Robinson et al., 2024) used in our experiments.

## REFERENCES

Adrien Bardes, Jean Ponce, and Yann LeCun. VICReg: Variance-invariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations, 2022.

Tianlang Chen, Charilaos Kanatsoulis, and Jure Leskovec. RelGNN: Composite message passing for relational deep learning, 2025. URL https://arxiv.org/abs/2502.06784.

Weilin Cong, Si Zhang, Jian Kang, Baichuan Yuan, Hao Wu, Xin Zhou, Hanghang Tong, and Mehrdad Mahdavi. Do we really need complicated model architectures for temporal networks?, 2023. URL https://arxiv.org/abs/2302.11636.

Vijay Prakash Dwivedi, Charilaos Kanatsoulis, Shenyang Huang, and Jure Leskovec. Relational deep learning: Challenges, foundations and next-generation architectures. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 5999–6009, 2025.

Vijay Prakash Dwivedi, Sri Jaladi, Yangyi Shen, Federico Lopez, Charilaos Kanatsoulis, Rishi Puri,´ Matthias Fey, and Jure Leskovec. Relational graph transformer. In International Conference on Learning Representations, volume 2026, pp. 154348–154369, 2026.

Matthias Fey, Weihua Hu, Kexin Huang, Jan Eric Lenssen, Rishabh Ranjan, Joshua Robinson, Rex Ying, Jiaxuan You, and Jure Leskovec. Position: Relational deep learning – graph representation learning on relational databases. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 13592–13607, 2024.

Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. Advances in neural information processing systems, 30, 2017.

Zhenyu Hou, Xiao Liu, Yukuo Cen, Yuxiao Dong, Hongxia Yang, Chunjie Wang, and Jie Tang. GraphMAE: Self-supervised masked graph autoencoders. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, pp. 594–604, 2022.

Zhenyu Hou, Yufei He, Yukuo Cen, Xiao Liu, Yuxiao Dong, Evgeny Kharlamov, and Jie Tang. GraphMAE2: A decoding-enhanced masked self-supervised graph learner. In Proceedings of the ACM web conference 2023, pp. 737–746, 2023.

Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. Strategies for pre-training graph neural networks. In International Conference on Learning Representations, 2020a.

Ziniu Hu, Yuxiao Dong, Kuansan Wang, Kai-Wei Chang, and Yizhou Sun. GPT-GNN: Generative pre-training of graph neural networks. In Proceedings ofthe 26th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 1857–1867, 2020b.

Ziniu Hu, Yuxiao Dong, Kuansan Wang, and Yizhou Sun. Heterogeneous graph transformer. In Proceedings of the web conference 2020, pp. 2704–2710, 2020c.

Xunqiang Jiang, Tianrui Jia, Yuan Fang, Chuan Shi, Zhe Lin, and Hui Wang. Pre-training on largescale heterogeneous graph. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining, pp. 756–766, 2021.

James Max Kanter and Kalyan Veeramachaneni. Deep feature synthesis: Towards automating data science endeavors. In 2015 IEEE international conference on data science and advanced analytics (DSAA), pp. 1–10. IEEE, 2015.

Chenglin Li, Yuanzhen Xie, Chenyun Yu, Junfeng Zhao, Yu Xia, Beibei Kong, Zang Li, and Di Niu. CTRL: Continuous-time representation learning on temporal heterogeneous information network. Know.-Based Syst., 339(C), June 2026. ISSN 0950-7051. doi: 10.1016/j.knosys.2026.115514. URL https://doi.org/10.1016/j.knosys.2026.115514.

Longhai Li, Lei Duan, Junchen Wang, Chengxin He, Zihao Chen, Guicai Xie, Song Deng, and Zhaohang Luo. Memory-enhanced transformer for representation learning on temporal heterogeneous graphs. Data Science and Engineering, 8:98–111, 2023.

Yijun Ma, Zehong Wang, Weixiang Sun, and Yanfang Ye. Temporal graph pattern machine, 2026. URL https://arxiv.org/abs/2601.22454.

Yixin Peng, Diego Collarana, Er Jin, and Stefan Decker. THGFM: Dual-branch temporal heterogeneous graph fusion model, 2026. URL https://arxiv.org/abs/2607.27303.

Jiezhong Qiu, Qibin Chen, Yuxiao Dong, Jing Zhang, Hongxia Yang, Ming Ding, Kuansan Wang, and Jie Tang. GCC: Graph contrastive coding for graph neural network pre-training. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’20, pp. 1150–1160. ACM, August 2020. doi: 10.1145/3394486.3403168. URL http: //dx.doi.org/10.1145/3394486.3403168.

Joshua Robinson, Rishabh Ranjan, Weihua Hu, Kexin Huang, Jiaqi Han, Alejandro Dobles, Matthias Fey, Jan E Lenssen, Yiwen Yuan, Zecheng Zhang, et al. RelBench: A benchmark for deep learning on relational databases. Advances in Neural Information Processing Systems, 37:21330–21341, 2024.

Lianze Shan, Jitao Zhao, Dongxiao He, Yongqi Huang, Zhiyong Feng, and Weixiong Zhang. MUG: Meta-path-aware universal heterogeneous graph pre-training. In Proceedings ofthe AAAI Confer ence on Artificial Intelligence, volume 40, pp. 25260–25268, 2026.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding, 2023. URL https://arxiv.org/abs/2104.09864.

Shengyin Sun, Chen Ma, and Jiehao Chen. PHE: Structure and semantic enhanced pre-training of graph neural networks for large-scale heterogeneous graphs. ACM Transactions on Knowledge Discoveryfrom Data, 20(1):1–26, 2025.

Yijun Tian, Kaiwen Dong, Chunhui Zhang, Chuxu Zhang, and Nitesh V Chawla. Heterogeneous graph masked autoencoders. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 9997–10005, 2023.

Quang Truong, Zhikai Chen, Mingxuan Ju, Tong Zhao, Neil Shah, and Jiliang Tang. A pretraining framework for relational data with information-theoretic principles. Advances in Neural Information Processing Systems, 38:8122–8153, 2026.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding, 2019. URL https://arxiv.org/abs/1807.03748.

Yanbo Wang, Xiyuan Wang, Quan Gan, Minjie Wang, Qibin Yang, David Wipf, and Muhan Zhang. Griffin: Towards a graph-centric relational database foundation model, 2025. URL https://arxiv.or g/abs/2505.05568.

Da Xu, Chuanwei Ruan, Evren Korpeoglu, Sushant Kumar, and Kannan Achan. Inductive representation learning on temporal graphs. In International Conference on Learning Representations, 2020.

Jinyu Yang, Cheng Yang, Junze Chen, Zedi Liu, Muhan Zhang, Hanyang Peng, and Chuan Shi. RelPrism: A multi-faceted pre-training framework with self-generated tasks for relational databases, 2026. URL https://arxiv.org/abs/2605.23241.

Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and Yang Shen. Graph contrastive learning with augmentations. Advances in neural information processing systems, 33: 5812–5823, 2020.

Le Yu, Leilei Sun, Bowen Du, and Weifeng Lv. Towards better dynamic graph learning: New architecture and unified library. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 67686– 67700. Curran Associates, Inc., 2023. doi: 10.52202/075280-2960. URL https://proceedings.neur ips.cc/paper files/paper/2023/file/d611019afba70d547bd595e8a4158f55-Paper-Conference.pdf.

## CONTENTS OF THE APPENDIX

A Datasets and Prediction Tasks 14   
A.1 Database Statistics 14   
A.2 Task Statistics and Temporal Splits 14   
A.3 Database Details and Prediction Targets 14   
B Detailed Results 15   
B.1 Backbone Selection: GNNs and GTs 15   
B.2 Temporal Encoding and Temporal Information Distributions 16   
B.3 Pretraining Detailed Results 18   
C Architecture and Training Configuration . 20   
D Temporal Encoding Details . 22   
E Pretraining Objective Details . 23   
F Pretraining and Fine-Tuning Curves 25

## A DATASETS AND PREDICTION TASKS

The five RelBench databases (Robinson et al., 2024) provide one multiclass classification task, six binary classification tasks, and four regression tasks. The tables below give their sizes; the task definitions specify prediction windows and eligibility rules.

## A.1 DATABASE STATISTICS

Table 1 counts records before temporal filtering, excluding downstream task tables and reverse edges. Each declared foreign-key column counts as one directed schema relation, including separate columns that reference the same table.

Table 1: Stored database statistics. FK denotes foreign-key columns; Tasks counts the tasks evaluated here.
<table><tr><td>Database</td><td>Domain</td><td>Tables</td><td>Rows</td><td>FK relations</td><td>Tasks</td></tr><tr><td>Arxiv</td><td>Scholarly publications</td><td>6</td><td>2,733,846</td><td>6</td><td>2</td></tr><tr><td>Avito</td><td>Online advertising</td><td>8</td><td>24,653,915</td><td>11</td><td>3</td></tr><tr><td>F1</td><td>Motor racing</td><td>9</td><td>97,606</td><td>13</td><td>2</td></tr><tr><td>H&amp;M</td><td>Fashion retail</td><td>3</td><td>16,931,173</td><td>2</td><td>2</td></tr><tr><td>Event</td><td>Social events</td><td>5</td><td>44,822,992</td><td>7</td><td>2</td></tr></table>

## A.2 TASK STATISTICS AND TEMPORAL SPLITS

In Table 2, a sample is an (entity, prediction time) pair; an entity can therefore appear more than once.   
“Entities” counts distinct target identifiers across all three splits.

Table 2: Task sizes in the stored splits. Entities are counted once across training, validation, and test.
<table><tr><td>Database</td><td>Task</td><td>Type</td><td>Train</td><td>Validation</td><td>Test</td><td>Entities</td></tr><tr><td rowspan="2">Arxiv</td><td>author-category</td><td>Multiclass classification</td><td>210,769</td><td>39,015</td><td>39,655</td><td>126,219</td></tr><tr><td>paper-citation</td><td>Binary classification</td><td>534,233</td><td>155,845</td><td>193,696</td><td>193,696</td></tr><tr><td rowspan="3">Avito</td><td>user-clicks</td><td>Binary classification</td><td>59,454</td><td>21,183</td><td>47,996</td><td>66,449</td></tr><tr><td>user-visits</td><td>Binary classification</td><td>86,619</td><td>29,979</td><td>36,129</td><td>63,405</td></tr><tr><td>ad-ctr</td><td>Regression</td><td>5,100</td><td>1,766</td><td>1,816</td><td>4,997</td></tr><tr><td rowspan="2">F1</td><td>driver-dnf</td><td>Binary classification</td><td>11,411</td><td>566</td><td>702</td><td>821</td></tr><tr><td>driver-position</td><td>Regression</td><td>7,453</td><td>499</td><td>760</td><td>826</td></tr><tr><td rowspan="2">H&amp;M</td><td>user-churn</td><td>Binary classification</td><td>3,832,692</td><td>76,556</td><td>74,575</td><td>999,345</td></tr><tr><td>item-sales</td><td>Regression</td><td>5,488,184</td><td>105,542</td><td>105,542</td><td>105,542</td></tr><tr><td rowspan="2">Event</td><td>user-ignore</td><td>Binary classification</td><td>19,239</td><td>2,013</td><td>1,958</td><td>9,694</td></tr><tr><td>user-attendance</td><td>Regression</td><td>19,239</td><td>2,013</td><td>1,958</td><td>9,694</td></tr></table>

We use the cached chronological splits. Validation and test cutoffs are 2022-01-01 and 2023-01- 01 for Arxiv; 2015-05-08 and 2015-05-14 for Avito; 2020-09-07 and 2020-09-14 for H&M; and 2012-11-21 and 2012-11-29 for Event. F1 uses several prediction times. For driver-dnf, validation spans 2005-03-02 to 2008-03-16 and test spans 2010-03-02 to 2013-03-16. For driver-position, the corresponding spans are 2005-03-02 to 2009-10-07 and 2010-03-02 to 2016-05-29. Targets cover (t, t + ∆] after cutoff t, with the task-specific horizons below.

## A.3 DATABASE DETAILS AND PREDICTION TARGETS

The task definitions follow the task-construction code used in this study.

Arxiv (rel-arxiv) contains 222,769 papers, 143,691 authors, 53 categories, 616,585 authorship rows, 155,061 paper–category rows, and 1,595,687 citation rows. Submission dates timestamp papers and associated relation records. author-category predicts an author’s most frequent primary research category over the next 182 days, among 53 classes, and includes only authors who publish in that window. paper-citation predicts whether a paper submitted by the cutoff receives a citation in the next 182 days.

Avito (rel-avito) links 5,960,558 advertisements and 98,250 users to 2,579,289 searches, 9,254,702 search-stream records, 6,454,562 visits, and 302,974 phone requests, with location and category tables as context. All three tasks use a four-day prediction window. user-clicks predicts whether a user records more than one ad click, counting repeated clicks separately. user-visits predicts visits to more than one distinct advertisement. ad-ctr predicts clicks divided by searchstream impressions for advertisements with at least one click in the window.

F1 (rel-f1) records drivers, constructors, circuits, races, qualifying sessions, results, and standings. Its nine tables include 857 drivers, 211 constructors, 77 circuits, 1,101 races, and 26,080 race results; race dates timestamp competition records. driver-dnf predicts whether a driver has a result with statusId ̸= 1 in the next 30 days, following the benchmark rule that any status other than Finished receives a did not finish (DNF) label. driver-position predicts mean finishing order (positionOrder) over the next 60 days.

H&M (rel-hm) contains 1,371,980 customers, 105,542 articles, and 15,453,651 transactions. Dated purchases link customers to articles and record prices; customer attributes and product descriptions provide additional features. user-churn includes customers who purchased in the previous seven days and predicts no purchase in the next seven days. item-sales predicts the sum of an article’s transaction prices over the next seven days, using the database’s price scale and assigning zero when there are no sales.

Event (rel-event) contains 38,209 users, 3,137,972 events, 30,386,403 friendship rows, 11,245,010 attendance rows, and 15,398 interest rows. Event start times, user join times, and interest timestamps provide temporal context. For events starting in the next seven days, user-ignore predicts whether more than two of a user’s attendance records remain invited, and user-attendance counts that user’s yes or maybe records. These targets describe recorded responses rather than verified attendance.

## B DETAILED RESULTS

This section reports the full backbone comparison, temporal encoding ablation, and pretraining results, together with record-age and horizon statistics that supplement Section 4.2. Task metrics are reported as the mean ± standard deviation across four independent runs with seeds {42, 43, 44, 45}. Relative gains and their aggregates are computed from these task means. Table 3 reports validation performance for backbone selection; the encoding and pretraining comparisons report test performance.

## B.1 BACKBONE SELECTION: GNNS AND GTS

Table 3 compares validation performance for five backbones under the same RDL pipeline: HET-EROGNN, RelGNN, HGT, RelGT, and THGFM (Robinson et al., 2024; Chen et al., 2025; Hu et al., 2020c; Dwivedi et al., 2026; Peng et al., 2026). For each task, we rank all five backbones by their mean validation metric, using higher accuracy or ROC–AUC and lower MAE as better performance. We average these ranks equally across all 11 tasks, then select the backbone with the lowest mean rank within each family. Among GNNs, HETEROGNN has a lower mean validation rank than RelGNN (3.18 versus 3.36), with two and three task wins, respectively. Among GTs, THGFM has the lowest mean validation rank (2.27), followed by HGT (2.55) and RelGT (3.64), with four, two, and zero task wins, respectively. We therefore select HETEROGNN and THGFM for the controlled encoding and pretraining study. Task wins are descriptive; mean validation rank determines the family representatives.

Selecting one representative per family is a deliberate trade-off to reduce computational cost: running all encoding ablations, pretraining objectives, and staged schedules on all five backbones would substantially increase the training budget. The subsequent comparisons therefore assess effects on these two selected backbones, and do not establish that the effects extend to RelGNN, HGT, or RelGT. Backbone selection uses validation performance only; test performance is reserved for final

Table 3: Backbone validation performance (mean ± standard deviation) used to select one representative per family. Ranks are computed across all five models for each task and averaged equally over all 11 tasks. Bold marks the best task mean, the highest task-win count, and the lowest mean rank within each family; arrows indicate the preferred direction.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Metric</td><td colspan="2">GNN</td><td colspan="3">GT</td></tr><tr><td>HETEROGNN</td><td>RelGNN</td><td>HGT</td><td>RelGT</td><td>THGFM</td></tr><tr><td>arxiv/author-category</td><td>Acc.↑</td><td>0.1311±0.0020</td><td>0.1298±0.0010</td><td>0.1331±0.0017</td><td>0.1037±0.0050</td><td>0.1324±0.0027</td></tr><tr><td>arxiv/paper-citation</td><td>AUC↑</td><td>0.6495±0.0069</td><td>0.6216±0.0095</td><td>0.6583±0.0074</td><td>0.6550±0.0027</td><td>0.6652±0.0019</td></tr><tr><td>avito/user-clicks</td><td>AUC↑</td><td>0.6435±0.0021</td><td>0.6496±0.0336</td><td>0.6369±0.0032</td><td>0.6298±0.0185</td><td>0.6405±0.0008</td></tr><tr><td>avito/user-visits</td><td>AUC↑</td><td>0.6938±0.0012</td><td>0.6735±0.0014</td><td>0.6763±0.0095</td><td>0.6899±0.0186</td><td>0.6830±0.0075</td></tr><tr><td>f1/driver-dnf</td><td>AUC↑</td><td>0.7442±0.0042</td><td>0.8020±0.0042</td><td>0.7978±0.0064</td><td>0.7830±0.0117</td><td>0.7999±0.0068</td></tr><tr><td>hm/user-churn</td><td>AUC↑</td><td>0.7025±0.0005</td><td>0.7011±0.0108</td><td>0.7042±0.0006</td><td>0.6970±0.0012</td><td>0.7064±0.0015</td></tr><tr><td>event/user-ignore</td><td>AUC↑</td><td>0.8570±0.0022</td><td>0.8618±0.0275</td><td>0.8638±0.0275</td><td>0.8593±0.0297</td><td>0.8661±0.0094</td></tr><tr><td>hm/item-sales</td><td>MAE↓</td><td>0.0656±0.0003</td><td>0.0640±0.0007</td><td>0.0692±0.0021</td><td>0.0662±0.0004</td><td>0.0670±0.0002</td></tr><tr><td>avito/ad-ctr</td><td>MAE↓</td><td>0.0355±0.0004</td><td>0.0398±0.0000</td><td>0.0368±0.0012</td><td>0.0372±0.0014</td><td>0.0380±0.0000</td></tr><tr><td>f1/driver-position</td><td>MAE↓</td><td>3.8399±0.0078</td><td>3.8703±0.0079</td><td>3.0344±0.2518</td><td>3.1994±0.0656</td><td>3.5500±0.0034</td></tr><tr><td>event/user-attendance</td><td>MAE↓</td><td>0.2570±0.0006</td><td>0.2490±0.0015</td><td>0.2470±0.0001</td><td>0.2530±0.0015</td><td>0.2420±0.0022</td></tr><tr><td colspan="2">No. of task-wise best results ↑</td><td>2</td><td>3</td><td>2</td><td>0</td><td>4</td></tr><tr><td colspan="2">Mean rank ↓</td><td>3.18</td><td>3.36</td><td>2.55</td><td>3.64</td><td>2.27</td></tr></table>

Table 4: Temporal encoding ablation (mean ± standard deviation over four runs). Gains are relative to the same backbone with DAYPE (Equation (10)) and without TIMEROPE or pretraining in each panel. TIMEMIX replaces DAYPE; TIMEROPE adds rotary interval encoding to the specified record-age encoder. Mean type gain weights the three task types equally. Bold marks the best entry per panel and column.  
```csv
Multiclass classification Binary classification Regression
Encoding Author Mean task gain Citation Ignore DNF Churn Clicks Visits Mean task gain At end. Position Sales CTR Mean task gain Mean type gain
Acc.↑ ∆% ↑ AUC↑ AUC↑ AUC↑ AUC↑ AUC↑ AUC↑ ∆% ↑ MAE↓ MAE↓ MAE↓ MAE↓ ∆% ↑ ∆% ↑
Panel A: HETEROGNN (GNN backbone
DAYPE (reference) 0.1236±0.0005 +0.00% 0.7101±0.0105 0.6961±0.0019 0.6474±0.0164 0.6627±0.0008 +0.00% 4.3596±0.0168 +0.00% +0.00%
TIMEMIX 0.1265±0.0028 +2.35% 0.6430±0.0043 0.8083±0.0095 0.7316±0.0043 0.6868±0.0028 0.6282±0.0042 0.6355±0.0090 −0.84% 0.2577±0.0083 0.0638±0.0015 0.0408±0.0006 −0.98% +0.18%
DAYPE+TIMEROPE 0.1239±0.0017 +0.24% 0.6527±0.0036 0.8202±0.0051 0.7227±0.0099 0.6896±0.0052 0.6318±0.0022 0.6454±0.0050 −0.14% 0.2616±0.0021 4.1803±0.1225 0.0617±0.0006 0.0421±0.0001 −0.49% −0.13%
TIMEMIX+TIMEROPE 0.1261±0.0026 +2.02% 0.6469±0.0031 0.8789±0.0073 0.6967±0.0231 0.6920±0.0011 0.6174±0.0070 0.6269±0.0080 −0.46% 0.2566±0.0090 4.0603±0.0988 0.0660±0.0002 0.0410±0.0005 −0.18% +0.46%
Panel B: THGFM (GT backbone)
DAYPE (reference) 0.1235±0.0020 +0.00% 0.6603±0.0012 0.8194±0.0326 0.7396±0.0068 0.6969±0.0041 0.6346±0.0023 +0.00% 0.2439±0.0009 0.0613±0.0006 +0.00% +0.00%
TIMEMIX 0.1266±0.0034 +2.51% 0.6482±0.0035 0.8308±0.0195 0.7210±0.0051 0.6913±0.0016 0.6195±0.0021 0.6269±0.0117 −1.28% 0.2422±0.0039 4.2098±0.1526 0.0639±0.0017 0.0412±0.0006 +1.26% +0.83%
DAYPE+TIMEROPE 0.1232±0.0022 −0.24% 0.6543±0.0019 0.8196±0.0250 0.7213±0.0137 0.6926±0.0020 0.6311±0.0019 0.6465±0.0036 −0.50% 0.2415±0.0016 4.3080±0.1579 0.0616±0.0004 0.0421±0.0000 +1.08% +0.11%
TIMEMIX+TIMERoPE. 0.1265+0.0045 +2.43% 0.6940±0.0007 0.6321±0.0077 0.6454±0.0118 −0.05% 0.2456±0.0039 4.2606±0.2064 +1.49% +1.29%
```  
evaluation. Downstream checkpoints are also selected using validation metrics, while pretraining checkpoints follow the objective-specific criteria in Appendix C.

## B.2 TEMPORAL ENCODING AND TEMPORAL INFORMATION DISTRIBUTIONS

Reference encoding and configurations. As introduced in Section 4.2, DAYPE (day-based positional encoding) is the fixed day-based single-scale record-age encoding used by HETEROGNN (Robinson et al., 2024). It serves as the baseline for multi-scale TIMEMIX on both backbones and is defined as

$$
e _ { v } ^ { \mathrm { b a s e } } = W _ { \phi ( v ) } \operatorname { P E } ( a _ { v } / s _ { \mathrm { d a y } } ) + b _ { \phi ( v ) } , \qquad s _ { \mathrm { d a y } } = 8 6 4 0 0 \mathrm { s e c o n d s } .\tag{10}
$$

Here $a _ { v }$ is measured in seconds, PE is sinusoidal positional encoding with fixed frequencies, and $W _ { \phi ( v ) }$ and $b _ { \phi ( v ) }$ are type-specific trainable parameters. The term single-scale refers to the fixed day-based normalization; the sinusoidal positional encoding itself contains multiple frequencies. Thus, “without our encodings” retains DAYPE and disables TIMEROPE; it still provides recordage information. In contrast, TIMEMIX uses multiple learned time scales and mixes type-specific projections of log-transformed record ages (Section 3.1; Appendix D).

Aggregate and task-level effects. Relative to DAYPE, replacing the age encoder with TIMEMIX yields mean type gains of +0.18% on HETEROGNN and +0.83% on THGFM, while adding TIMEROPE to DAYPE yields −0.13% and +0.11%, respectively. Combining TIMEMIX and TIMEROPE gives the highest mean type gain on both backbones, +0.46% and +1.29%. Accordingly, Figure 4a shows mean type gain differences of −0.59, −0.28, and −0.46 percentage points on HET-EROGNN, and −1.18, −0.46, and −1.29 points on THGFM, for DAYPE+TIMEROPE, TIMEMIX, and DAYPE, respectively. The combined gains by task type are +2.02%, −0.46%, and −0.18% on HETEROGNN, and +2.43%, −0.05%, and +1.49% on THGFM, for multiclass classification, binary classification, and regression, respectively. Thus, the positive overall gains do not imply improvements in every task type. In particular, TIMEMIX alone gives slightly higher multiclass gains than the combined configuration on both backbones, while the combination gives higher regression gains than either encoding change alone.

![](images/fb2cbee1c53552ef3611f3afecaefb80ed52f06931e510f84739a614a745f8ad.jpg)  
Figure 5: Record-age distributions by task: (a) all nine intervals and (b) a magnified view of ages up to one day, on the original percentage scale.

Task-level effects also vary. With both encodings, HETEROGNN improves author classification by 2.02%, user-ignore AUC by 9.44%, and MAE by 7.37% for driver position and 2.10% for user attendance, but loses performance on both Avito binary tasks and item sales. On THGFM, the combined encodings improve ad-CTR MAE by 5.38% and also help author classification, user visits, user ignore, and driver position; citation and driver-dnf performance decrease.

Distribution of record ages. Figure 5 summarizes the ages of records available to each task. We first deduplicate the training (entity, cutoff) pairs and sample up to 20,000 pairs per task using seed 42. For each pair, we collect the seed row, if timestamped, and directly linked timestamped rows observed by the cutoff. We convert timestamps to integer Unix seconds and compute record ages as defined in Section 3.1. These statistics cover the seed and its direct neighbors; the model samples neighborhoods that may extend to additional hops. We group ages into nine bins with boundaries at 1 second, 1 minute, 1 hour, 1 day, and 7, 30, 90, and 365 days. The first four boundaries match the initial TIMEMIX scales. Panel (b) enlarges the bins up to one day while retaining their percentages of all observations; bar-end labels show their combined share.

The observed histories span different time scales across databases. In Avito, all sampled record ages fall within 30 days, and 19.7–20.6% of the records are at most one day old. Avito also shows variation at second, minute, and hour scales; minute-scale variation is largely absent from the other databases under integer-second preprocessing. Event records are concentrated between 1 and 90 days, while H&M records mostly fall between 7 and 365 days. Arxiv and F1 contain much older records: fewer than 0.3% are at most one day old, and about 78% of F1 records are more than a year old. These differences provide descriptive motivation for the multi-scale design of TIMEMIX. The distributions alone do not establish its predictive advantage over a single-scale variant or determine the optimal number or initialization of scales. The encoding ablations evaluate TIMEMIX as a complete replacement for DAYPE; a controlled comparison of $\bar { K } _ { \mathrm { t i m e } } = 1$ and $K _ { \mathrm { t i m e } } > 1$ within the same TIMEMIX formulation remains to be evaluated.

![](images/2c6852830558b0c998a28079ddd94a92039c849f3c77612860cbe3ff22247740.jpg)

![](images/672e31163cd5ce3a7f8d371098163e02440de8807c290ce0e293eaa1ba30ec64.jpg)  
Figure 6: Support for REL-FUTURE horizons: (a) the relation-balanced percentage of eligible events supporting each horizon and (b) the resulting horizon sampling distribution.

Support for future horizons. Figure 6 shows which of the candidate horizons {1, 7, 30, 90, 365} days can be used for REL-FUTURE. A horizon is eligible when the entity is already observed at the input cutoff, as specified in Equation (17). We compute these statistics from timestamped forward foreign-key relations up to each database’s validation cutoff, excluding static relations and events with no eligible horizon. Panel (a) reports the percentage of eligible events supporting each horizon, computed within each relation and then averaged equally across relations. An event may support several horizons. Panel (b) shows the expected share of training examples assigned to each horizon when we uniformly sample a relation, an eligible event within it, and an eligible horizon for that event.

Support for longer horizons varies substantially across databases. Avito supports only 1- and 7-day windows. In Event, support falls from 62.6% at 7 days to 31.3% at 30 days and 9.1% at 90 days, with no support at 365 days. H&M supports windows through 90 days, but only 0.05% of eligible events support a 365-day window. By comparison, the 365-day horizon is eligible for 67.7% of Arxiv and 87.1% of F1 events under the same averaging scheme. A shared set of multiple horizons therefore provides short-window supervision in databases with limited history and longer-window supervision where the history permits it.

The sampling distribution reflects these differences. In Avito, 80.5% of training examples are expected to use the 1-day horizon and 19.5% the 7-day horizon; unsupported horizons receive no samples. Thus, using the same candidate set does not imply sampling horizons equally across databases. These statistics describe the availability and frequency of supervision, rather than the predictive value of each horizon.

## B.3 PRETRAINING DETAILED RESULTS

Tables 5 and 6 provide the full results underlying the pretraining comparisons, including task metrics and results for TEMP-SUB alone. Each panel uses a supervised control with the same backbone and temporal encoding configuration: A0 uses DAYPE without TIMEROPE, whereas B0 replaces DAYPE with TIMEMIX and enables TIMEROPE. Both controls are trained directly on downstream tasks without pretraining. The reported gains therefore measure the additional effect of pretraining, matching Figure 4b. The main-results heatmap in Figure 3 instead uses A0 throughout, measuring the effects of pretraining and, where enabled, our temporal encodings. Rows with TEMP-SUB followed by a second-stage objective can be compared with either objective alone. Each panel contains 20 pretraining strategies and one supervised control: nine single-stage baselines, our three single-stage objectives, and eight two-stage schedules. The latter comprise TS followed by ATTRMASKING, CONTEXTPRED, PT-HGNN, CTRL, TGPM, TVE, REL-HIST, or REL-FUTURE. The six baseline pairs and our two proposed pairs are shown in Figure 4b.

Table 5: Pretraining transfer on HETEROGNN (mean ± standard deviation over four runs). Panel A uses DAYPE without TIMEROPE; Panel B uses TIMEMIX with TIMEROPE. Gains are relative to the marked reference within each panel. Bold marks the best value per panel and column.
<table><tr><td></td><td>Pretraining strategy</td><td>Multiclass classification</td><td></td><td></td><td colspan="6">Binary classification</td><td colspan="6"></td></tr><tr><td></td><td>Temporal Graph-level Local stage</td><td>Author Acc.↑</td><td>Task gain Δ%↑</td><td>Citation AUC↑</td><td>Ignore AUC↑</td><td>DNF AUC↑</td><td>Chum AUC↑</td><td>Clicks AUC↑</td><td></td><td>AUC↑ Visits</td><td>Mean task gain Δ%↑ Attend. MAE↓</td><td>Position MAE↓</td><td>Sales MAE↓</td><td></td><td>CTR MAE↓ Δ% ↑</td><td>Mean task gain Mean type gain △%↑</td></tr><tr><td>Panel A: DAYPE without T1MERoPE. All Δ colu w/o</td><td>No pretraining (Ref. A0)</td><td>0.1236±0.0005</td><td>ans use Ref. A0 below +0.00%</td><td>0.6450±0.0047</td><td>0.8031±0.0157</td><td>0.7101±0.0105</td><td>0.6961±0.0019</td><td>0.6474±0.0164</td><td>0.6627±0.0008</td><td>+0.00%</td><td>0.2620±0.0013</td><td>4.3596±0.0168</td><td>0.0604±0.0001</td><td>0.0403±0.0003</td><td>+0.00%</td><td>+0.00%</td></tr><tr><td colspan="9">Baseline pretraining strategies</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o</td><td></td><td>0.1209±0.0039</td><td>-2.18%</td><td>0.5023±0.0053</td><td>0.8206±0.0286</td><td>0.7038±0.0284</td><td>0.6902±0.0011</td><td>0.6482±0.0096</td><td>0.6570±0.0032</td><td>-3.74%</td><td>0.2632±0.0059</td><td>4.3955±0.0202</td><td>0.0565±0.0008</td><td>0.0362±0.0002</td><td>+4.24%</td><td>-0.56%</td></tr><tr><td>w/o</td><td></td><td>Un-SAGE GPT-GNN 0.1203±0.0008</td><td>-2.67%</td><td>0.6226±0.0174</td><td>0.8658±0.0139</td><td>0.7007±0.0012</td><td>0.6895±0.0008</td><td>0.6470±0.0123</td><td>0.6582±0.0042</td><td>+0.22%</td><td>0.2655±0.0016</td><td>4.3884±0.0273</td><td>0.0566±0.0018</td><td>0.0393±0.0012</td><td>+1.82%</td><td>-0.21%</td></tr><tr><td>w/o</td><td></td><td>PT-HGNN 0.1176±0.0060</td><td>-4.85%</td><td>0.6494±0.0580</td><td>0.8182±0.0549</td><td>0.7589±0.0009</td><td>0.6971±0.0043</td><td>0.6579±0.0087</td><td>0.6625±0.0096 0.6548±0.0063</td><td>+1.86% -0.36%</td><td>0.2638±0.0029</td><td>4.3516±0.1148</td><td>0.0557±0.0007</td><td>0.0376±0.0006 0.0383±0.0004</td><td>+3.78%</td><td>+0.26% -1.06%</td></tr><tr><td>w/o w/o</td><td>PHE</td><td>0.1225±0.0010 ATTRMASKING 0.1214±0.0019</td><td>-0.89% -1.78%</td><td>0.6412±0.0014 0.6382±0.0207</td><td>0.8540±0.0200</td><td>0.6801±0.0204</td><td>0.6948±0.0010</td><td>0.6323±0.0049 0.6424±0.0077</td><td>0.6591±0.0057</td><td>+0.36%</td><td>0.2635±0.0092 0.2618±0.0079</td><td>5.3604±0.0053 4.2791±0.1757</td><td>0.0568±0.0001 0.0558±0.0011</td><td>0.0380±0.0009</td><td>-1.92% +4.06%</td><td>+0.88%</td></tr><tr><td>w/o</td><td>CTRL</td><td>0.1240±0.0017</td><td>+0.32%</td><td>0.5399±0.0744</td><td>0.8352±0.0131 0.8274±0.0219</td><td>0.7103±0.0008 0.6675±0.0370</td><td>0.6998±0.0017 0.6962±0.0031</td><td></td><td>0.6386±0.0117</td><td>0.6538±0.0010</td><td>-3.66% 0.2626±0.0045</td><td>3.9744±0.0422</td><td>0.0563±0.0008</td><td>0.0455±0.0001</td><td>+1.33%</td><td>-0.67%</td></tr><tr><td>w/o w/o</td><td>TGPM</td><td>0.1238±0.0025</td><td>+0.16%</td><td>0.6441±0.0161</td><td>0.8124±0.0193</td><td>0.6969±0.0026</td><td>0.6961±0.0007</td><td>0.6534±0.0140</td><td></td><td>0.6594±0.0029</td><td>-0.07% 0.2548±0.0056</td><td></td><td>0.0560±0.0015</td><td>0.0396±0.0011</td><td>+2.36%</td><td>+0.82% -0.05%</td></tr><tr><td>w/o</td><td></td><td>0.1223±0.0019 0.1240±0.0007</td><td>-1.05% +0.32%</td><td>0.6436±0.0036</td><td>0.6870±0.0220</td><td>0.7217±0.0030 0.7081±0.0126</td><td>0.6957±0.0025 0.6953±0.0045</td><td>0.6536±0.0071 0.6460±0.0095</td><td>0.6623±0.0047 0.6554±0.0029</td><td>-2.03% -0.23%</td><td>0.2578±0.0038 0.2582±0.0012</td><td>4.2298±0.0307 4.3304±0.0951</td><td>0.0563±0.0020 0.0561±0.0007</td><td>0.0404±0.0003 0.0392±0.0002</td><td>+2.93% +3.15%</td><td>+1.08%</td></tr><tr><td></td><td>Our pretraining strategies</td><td></td><td></td><td>0.6344±0.0110</td><td>0.8189±0.0083</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o w/o</td><td>REL-HIST</td><td>0.1260±0.0019 +1.94%</td><td></td><td>0.6494±0.0011</td><td>0.7798±0.0276</td><td>0.6951±0.0052</td><td>0.6891±0.0044</td><td>0.6542±0.0067</td><td>0.6632±0.0026</td><td>-0.70%</td><td>0.2634±0.0099</td><td>4.3409±0.0113</td><td>0.0559±0.0008</td><td>0.0373±0.0017</td><td>+4.00%</td><td>+1.75%</td></tr><tr><td>w/o</td><td>REL-FUTURE</td><td>0.1233±0.0027</td><td>-0.24%</td><td>0.6410±0.0060</td><td>0.8464±0.0332</td><td>0.7027±0.0451</td><td>0.6951±0.0039</td><td>0.6469±0.0159</td><td>0.6613±0.0009</td><td>+0.55%</td><td>0.2543±0.0118</td><td>4.3560±0.0052</td><td>0.0556±0.0005</td><td>0.0391±0.0009</td><td>+3.70%</td><td>+1.34%</td></tr><tr><td>w/o</td><td>TEMP-SUB</td><td>0.1178±0.0029</td><td>-4.69%</td><td>0.6576±0.0065</td><td>0.7961±0.0426</td><td>0.6861±0.0146</td><td>0.6963±0.0037</td><td>0.6405±0.0101</td><td>0.6597±0.0026 0.6607±0.0008</td><td>-0.63% +0.13%</td><td>0.2660±0.0059 0.2548±0.0048</td><td>4.1975±0.1987 4.0122±0.1722</td><td>0.0556±0.0002 0.0560±0.0009</td><td>0.0404±0.0009</td><td>+2.69%</td><td>-0.88%</td></tr><tr><td>w/o w/o</td><td>TEMP-SUB ATTRMASKING TEMP-SUB CONTEXTPRED</td><td>0.1232±0.0037 0.1241±0.0022</td><td>-0.32% +0.40%</td><td>0.6461±0.0060 0.6366±0.0021</td><td>0.8227±0.0694 0.7895±0.0456</td><td>0.6834±0.0108 0.6821±0.0280</td><td>0.6982±0.0037 0.6976±0.0054</td><td>0.6600±0.0122 0.6511±0.0163</td><td>0.6536±0.0035</td><td>-1.25%</td><td>0.2583±0.0056</td><td>4.1695±0.2314</td><td>0.0560±0.0001</td><td>0.0443±0.0005 0.0495±0.0011</td><td>+2.58% -1.18%</td><td>+0.80% -0.68%</td></tr><tr><td>w/o</td><td></td><td>0.1208±0.0037</td><td>-2.27%</td><td>0.6535±0.0133</td><td>0.8122±0.0448</td><td>0.7377±0.0020</td><td>0.6985±0.0021</td><td>0.6579±0.0099</td><td>0.6635±0.0044</td><td>+1.40% +0.06%</td><td>0.2516±0.0028 0.2542±0.0028</td><td>4.3849±0.1361 4.1614±0.0564</td><td>0.0558±0.0002 0.0554±0.0005</td><td>0.0386±0.0005</td><td>+5.32%</td><td>+1.48%</td></tr><tr><td>w/o w/o</td><td>TEMP-SUB CTRL TEMP-SUB TGPM</td><td>0.1235±0.0009 0.1220±0.0054</td><td>-0.08% -1.29%</td><td>0.6462±0.0009 0.6460±0.0088</td><td>0.8027±0.0096 0.6776±0.0353</td><td>0.7037±0.0091 0.7287±0.0204</td><td>0.6977±0.0030 0.6976±0.0017</td><td>0.6565±0.0111 0.6571±0.0073</td><td>0.6594±0.0021 0.6625±0.0050</td><td>-1.86%</td><td>0.2544±0.0032</td><td>4.1520±0.0724</td><td>0.0557±0.0004</td><td>0.0374±0.0002 0.0375±0.0002</td><td>+4.89% +5.97%</td><td>+1.62% +0.94%</td></tr><tr><td>w</td><td>TEMP-SUB TVE TEF-SU R-HIsT</td><td>0.1230±0.0040</td><td>-0.49%</td><td>0.6327±0.0015</td><td>0.8044±0.0308</td><td>0.7106±0.0096</td><td>0.6928±0.0054</td><td>0.6452±0.0057</td><td>0.6514±0.0018 0.6596±0.0009</td><td>-0.70%</td><td>0.2563±0.0045 0.2527±0.0092</td><td>4.1328±0.1100 4.2533±0.0597</td><td>0.0560±0.0011 0.0561±0.0012</td><td>0.0369±0.0007</td><td>+5.45%</td><td>+2.01% +1.42%</td></tr><tr><td>w/o</td><td>TEMP-SUB REL-FUTURE</td><td>0.1242±0.0009 0.1234±0.0035</td><td>+0.49% -0.16%</td><td>0.6455±0.0052</td><td>0.7633±0.0244</td><td>0.7162±0.0225</td><td>0.6937±0.0060 0.6997±0.0017</td><td>0.6610±0.0070 0.6579±0.0143</td><td>0.6644±0.0032</td><td>-0.46% +1.03%</td><td>0.2464±0.0035</td><td>4.0057±0.0239</td><td>0.0552±0.0002</td><td>0.0376±0.0004 0.0395±0.0004</td><td>+6.00% +6.65%</td><td>+2.51%</td></tr><tr><td></td><td></td><td></td><td></td><td>0.6569±0.0032</td><td>0.8073±0.0369</td><td>0.7203±0.0062</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>wl</td><td></td><td>Panel B: TiMEMIX with TIMERoPE. All Δ columns use Ref. BO below</td><td></td><td></td><td></td><td></td><td></td><td>0.6174±0.0070</td><td>0.6269±0.0080</td><td>+0.00%</td><td>0.2566±0.0090</td><td></td><td>0.0660±0.0002</td><td>0.0410±0.0005</td><td></td><td>+0.00%</td></tr><tr><td></td><td>No pretraining (Ref. B0)</td><td>0.1261±0.0026</td><td>+0.00%</td><td>0.6469±0.0031</td><td>0.8789±0.0073 0.6967±0.0231</td><td></td><td>0.6920±0.0011</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>+0.00%</td><td></td></tr><tr><td>w/</td><td>Baseline pretraining strategies</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-4.17%</td><td>0.2637±0.0079</td><td>4.3595±0.0482</td><td>0.0565±0.0004</td><td></td><td></td><td></td></tr><tr><td>w/</td><td></td><td>Un-SAGE 0.1235±0.0016</td><td>-2.06%</td><td>0.5011±0.0504</td><td>0.8058±0.0046</td><td>0.6734±0.0012</td><td>0.6756±0.0018</td><td>0.6538±0.0088 0.6484±0.0076</td><td>0.6625±0.0025 0.6638±0.0075</td><td>+0.28%</td><td>0.2645±0.0032</td><td>4.5646±0.0259</td><td>0.0564±0.0019</td><td>0.0413±0.0003 0.0503±0.0001</td><td>+1.63% -3.88%</td><td>-1.53% -0.72%</td></tr><tr><td>w/</td><td></td><td>PT-HGNN GPT-GNN 0.1279±0.0013 0.1238±0.0013</td><td>+1.43% -1.82%</td><td>0.6172±0.0047 0.6081±0.0052</td><td>0.8540±0.0089</td><td>0.6867±0.0013 0.6864±0.0009</td><td>0.6893±0.0023 0.6900±0.0015</td><td>0.6537±0.0130</td><td>0.6614±0.0028</td><td>-0.80%</td><td>0.2637±0.0035</td><td>4.3801±0.0386</td><td>0.0556±0.0006 0.0564±0.0020</td><td>0.0379±0.0002</td><td>+4.22% +0.77%</td><td>+0.53% -0.92%</td></tr><tr><td>w</td><td></td><td>0.1208±0.0042 ATTRMASKING 0.1289±0.0019</td><td>-4.20% +2.22%</td><td>0.6133±0.0045 0.6579±0.0197</td><td>0.8047±0.0524 0.8185±0.0410 0.8280±0.0140</td><td>0.7335±0.0109 0.6355±0.0007</td><td>0.6857±0.0022 0.6936±0.0012</td><td>0.6588±0.0026 0.6367±0.0059</td><td>0.6581±0.0006 0.6588±0.0115</td><td>+0.66% -0.74%</td><td>0.2637±0.0016 0.2504±0.0081</td><td>5.1706±0.2662 3.9469±0.1745</td></table>

Table 6: Pretraining transfer on THGFM (mean ± standard deviation over four runs). Panel A uses DAYPE without TIMEROPE; Panel B uses TIMEMIX with TIMEROPE. Gains are relative to the marked reference within each panel. Bold marks the best value per panel and column.
<table><tr><td></td><td>Pretraining strategy</td><td>Multiclass classification</td><td></td><td></td><td colspan="6">Binary classification</td><td colspan="6"></td></tr><tr><td></td><td>Temporal Graph-level Local stage</td><td>Author Acc.↑</td><td>Task gain Δ%↑</td><td>Citation AUC↑</td><td>Ignore AUC↑</td><td>DNF AUC↑</td><td>Chumn AUC↑</td><td>Clicks AUC↑</td><td>Visits AUC↑</td><td></td><td>Mean task gain Attend. Δ%↑ MAE↓</td><td>Position MAE↓</td><td></td><td>Sales MAE↓</td><td>CTR MAE↓ Δ% ↑</td><td>Mean task gain Mean type gain Δ%↑</td></tr><tr><td>Panel A: DAYPE without TIMERoPE. All Δ columns use Ref. A0 below</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o</td><td>No pretraining (Ref. A0)</td><td>0.1235±0.0020 +0.00%</td><td></td><td>0.6603±0.0012</td><td>0.8194±0.0326</td><td>0.7396±0.0068</td><td>0.6969±0.0041</td><td>0.6346±0.0023</td><td>0.6369±0.0033</td><td>+0.00%</td><td>0.2439±0.0009</td><td>4.3693±0.0132</td><td>0.0613±0.0006</td><td>0.0431±0.0001</td><td>+0.00%</td><td>+0.00%</td></tr><tr><td></td><td>Baseline pretraining strategies</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o</td><td>Un-SAGE</td><td>0.1194±0.0009</td><td>-3.32%</td><td>0.6644±0.0054</td><td>0.8631±0.0352</td><td>0.7052±0.0348</td><td></td><td>0.6297±0.0087</td><td>0.6228±0.0046</td><td>-0.34%</td><td>0.2635±0.0032</td><td>4.6955±0.0240</td><td>0.0581±0.0005</td><td>0.0429±0.0002</td><td>-2.10%</td><td>-1.92%</td></tr><tr><td>w/o w/o</td><td>GPT-GNN PT-HGNN</td><td>0.1197±0.0015 0.1234±0.0011</td><td>-3.08%</td><td>0.6568±0.0058</td><td>0.7367±0.0405</td><td>0.6558±0.0045</td><td>0.6962±0.0022</td><td>0.6513±0.0040</td><td>0.6455±0.0229</td><td>-3.01%</td><td>0.2635±0.0093</td><td>4.3773±0.2472</td><td>0.0640±0.0002</td><td>0.0445±0.0024</td><td>-3.75%</td><td>-3.28%</td></tr><tr><td>w/o</td><td>PHE</td><td>0.1177±0.0009</td><td>-0.08% -4.70%</td><td>0.6493±0.0333</td><td>0.8698±0.0129 0.8376±0.0057</td><td>0.7563±0.0010 0.7219±0.0496</td><td>0.6998±0.0018</td><td>0.6737±0.0024</td><td>0.6370±0.0009</td><td>+2.22% -0.20%</td><td>0.2631±0.0111</td><td>4.3530±0.0658 5.8698±0.0560</td><td>0.0597±0.0003 0.0646±0.0003</td><td>0.0407±0.0005</td><td>+0.41%</td><td>+0.85%</td></tr><tr><td>w/o</td><td>ATTRMASKING</td><td>0.1203±0.0045</td><td>-2.59%</td><td>0.6464±0.0305 0.6502±0.0094</td><td>0.8027±0.0640</td><td>0.6809±0.0151</td><td>0.6965±0.0013</td><td>0.6273±0.0037</td><td>0.6514±0.0017</td><td>-1.27%</td><td>0.2636±0.0070 0.2583±0.0101</td><td></td><td>0.0588±0.0007</td><td>0.0422±0.0011 0.0429±0.0003</td><td>-9.00% -0.70%</td><td>-4.63% -1.52%</td></tr><tr><td>w/o</td><td>CONTEXTPRED</td><td>0.1169±0.0014</td><td>-5.34%</td><td>0.6506±0.0086</td><td>0.7907±0.0327</td><td>0.6480±0.0095</td><td>0.7007±0.0043 0.6939±0.0023</td><td>0.6482±0.0105 0.6414±0.0029</td><td>0.6577±0.0014</td><td></td><td>0.2608±0.0108</td><td>4.3993±0.2041</td><td>0.0579±0.0001</td><td>0.0499±0.0004</td><td>-3.73%</td><td>-3.70%</td></tr><tr><td>w/o</td><td>CTRL TGPM</td><td>0.1213±0.0008</td><td>-1.78%</td><td>0.6440±0.0244</td><td>0.7958±0.0104</td><td>0.6822±0.0186</td><td>0.6766±0.0011</td><td>0.6446±0.0043</td><td>0.6525±0.0029 0.6385±0.0022</td><td>-2.04%</td><td>0.2496±0.0014</td><td>4.2604±0.0554</td><td>0.0605±0.0009</td><td>0.0408±0.0003</td><td>+1.81%</td><td>-0.78%</td></tr><tr><td>w/o w/o</td><td>TVE</td><td>0.1218±0.0054</td><td>-1.38%</td><td>0.6375±0.0093</td><td>0.8017±0.0071</td><td>0.6760±0.0101</td><td>0.6755±0.0011</td><td>0.6342±0.0030</td><td>0.6299±0.0085</td><td>-2.37% -3.07%</td><td>0.2455±0.0093</td><td>4.2300±0.1372</td><td>0.0606±0.0007</td><td>0.0424±0.0002</td><td>+1.36%</td><td>-1.03%</td></tr><tr><td></td><td></td><td>0.1157±0.0018</td><td>-6.32%</td><td>0.6409±0.0023</td><td>0.7701±0.0312</td><td>0.7451±0.0014</td><td>0.6962±0.0049</td><td>0.6436±0.0029</td><td>0.6437±0.0070</td><td>-0.97%</td><td>0.2545±0.0069</td><td>4.2856±0.0534</td><td>0.0576±0.0006</td><td>0.0415±0.0004</td><td>+2.02%</td><td>-1.76%</td></tr><tr><td></td><td>tegies</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o w/o</td><td>REL-HIST REL-FUTURE</td><td>0.1225±0.0008 0.1233±0.0044</td><td>-0.81% -0.16%</td><td>0.6533±0.0042</td><td>0.7932±0.0032 0.8203±0.0141</td><td>0.7118±0.0556</td><td>0.6997±0.0024</td><td>0.6263±0.0070</td><td>0.6459±0.0030</td><td>-1.25%</td><td>0.2423±0.0048</td><td>4.4595±0.0945</td><td>0.0605±0.0006 0.0612±0.0007</td><td>0.0409±0.0010</td><td>+1.33%</td><td>-0.24%</td></tr><tr><td>w/o</td><td></td><td></td><td></td><td>0.6452±0.0010</td><td></td><td>0.6751±0.0407</td><td>0.7020±0.0032</td><td>0.6619±0.0062</td><td>0.6540±0.0016</td><td>-0.53%</td><td>0.2447±0.0079</td><td></td><td></td><td></td><td>+3.63%</td><td>+0.98%</td></tr><tr><td>w/o</td><td>TEMP-SUB TEMP-SUB ATTRMASKING</td><td>0.1194±0.0043 0.1212±0.0051</td><td>-3.32% -1.86%</td><td>0.6156±0.0559 0.6462±0.0107</td><td>0.7996±0.0211 0.8095±0.0162</td><td>0.6857±0.0345 0.6872±0.0202</td><td>0.6973±0.0036 0.7010±0.0030</td><td>0.6203±0.0171</td><td>0.6463±0.0028</td><td>-2.87%</td><td>0.2474±0.0004 0.2503±0.0063</td><td>4.3392±0.1328 4.3736±0.0351</td><td>0.0569±0.0001</td><td>0.0475±0.0007</td><td>-0.56%</td><td>-2.25%</td></tr><tr><td>w/o w/o</td><td>TEMP-SUB CONTEXTPRED</td><td>0.1138±0.0067</td><td>-7.85%</td><td>0.5573±0.0814</td><td>0.7770±0.0256</td><td>0.6519±0.0326</td><td>0.6977±0.0055</td><td>0.6411±0.0063 0.6652±0.0164</td><td>0.6523±0.0032 0.6486±0.0045</td><td>-1.07% -4.31%</td><td>0.2618±0.0038</td><td>4.2555±0.3372</td><td>0.0567±0.0009</td><td>0.0480±0.0008 0.0509±0.0002</td><td>-1.00% -2.84%</td><td>-5.00% -1.31%</td></tr><tr><td>w/o</td><td>TEMP-SUB CTRL TEMP-SUB PT-HGNN</td><td>0.1230±0.0013 0.1212±0.0020</td><td>-0.40% -1.86%</td><td>0.6481±0.0122 0.6440±0.0147</td><td>0.8465±0.0180 0.8011±0.0275</td><td>0.7330±0.0055</td><td>0.6991±0.0010</td><td>0.6660±0.0030</td><td>0.6437±0.0028</td><td>+1.15%</td><td>0.2521±0.0049</td><td>4.2452±0.0648 4.1967±0.0299</td><td>0.0583±0.0005 0.0568±0.0013</td><td>0.0406±0.0003</td><td>+2.74%</td><td>+1.16%</td></tr><tr><td>w/o</td><td>TEMP-SUB TGPM</td><td>0.1209±0.0031</td><td>-2.11%</td><td>0.6326±0.0055</td><td>0.8008±0.0367</td><td>0.6723±0.0294 0.6841±0.0188</td><td>0.6767±0.0029 0.6702±0.0009</td><td>0.6491±0.0108 0.6339±0.0063</td><td>0.6404±0.0077 0.6270±0.0010</td><td>-2.05% -3.51%</td><td>0.2492±0.0015 0.2469±0.0029</td><td>4.2224±0.0673</td><td>0.0607±0.0016</td><td>0.0406±0.0024 0.0384±0.0004</td><td>+4.02% +3.87%</td><td>+0.04% -0.58%</td></tr><tr><td></td><td>TEMP-SUB TVE</td><td>0.1157±0.0034</td><td>-6.32% -0.65%</td><td>0.6407±0.0184</td><td>0.7751±0.0377</td><td>0.7467±0.0043</td><td>0.6960±0.0022</td><td>0.6479±0.0045</td><td>0.6454±0.0130</td><td>-0.69%</td><td>0.2541±0.0039</td><td>4.2228±0.0664</td><td>0.0581±0.0005</td><td>0.0389±0.0001</td><td>+3.94%</td><td>-1.02%</td></tr><tr><td>w/o</td><td>TEF-SU REL-HIsT TEMP-SUB REL-FUTURE</td><td>0.1227±0.0028 0.1228±0.0016</td><td>-0.57%</td><td>0.6468±0.0033 0.6506±0.0021</td><td>0.8231±0.0233</td><td>0.7097±0.0248</td><td>0.6985±0.0007</td><td>0.6582±0.0050</td><td>0.6503±0.0094</td><td>+0.07%</td><td></td><td>0.2412±0.0020~4.1374±0.0560</td><td>−0.0568±0.0007</td><td>0.0404±0.0002</td><td>+5.33%</td><td>+1.58%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.8195±0.0130</td><td>0.6862±0.0092</td><td></td><td>0.6546±0.0071</td><td>0.6593±0.0265</td><td>-0.21%</td><td>0.2440±0.0029</td><td>4.3201±0.1085</td><td>0.0575±0.0002</td><td>0.0383±0.0053</td><td>+5.06%</td><td>+1.43%</td></tr><tr><td></td><td>Panel B: TiMEMiX with TiMERoPE. All Δ columns use Ref. B0 below</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/</td><td>No pretraining (Ref. B0)</td><td>0.1265±0.0045+0.00%</td><td></td><td>0.6515±0.0053</td><td>0.8393±0.0167</td><td>0.7255±0.0204</td><td>0.6940±0.0007</td><td>0.6321±0.0077</td><td>0.6454±0.0118</td><td>+0.00%</td><td>0.2456±0.0039</td><td>4.2606±0.2064</td><td>0.0621±0.0025</td><td>0.0409±0.0011</td><td>+0.00%</td><td>+0.00%</td></tr><tr><td></td><td>Baseline pretraining strategies Ua-SAGE</td><td></td><td></td><td></td><td></td><td></td><td>0.6831±0.0008</td><td>0.6334±0.0026</td><td></td><td>−8.00%</td><td>0.2635±0.0102</td><td>4.7512±0.0520</td><td>0.0761±0.0005</td><td></td><td></td><td></td></tr><tr><td>wl w</td><td></td><td>0.1239±0.0045 0.1209±0.0032</td><td>-2.06% -4.43%</td><td>0.5074±0.0059 0.6382±0.0027</td><td>0.8037±0.0219</td><td>0.6043±0.0038 0.6893±0.0031</td></table>

Mean type gain weights the multiclass, binary, and regression task types equally, using the gain definition in Appendix B.2. We aggregate before rounding and compute differences from the unrounded gains. The average ranks in Figure 3 rank all 20 pretraining strategies within each backbone–encoding setting, excluding the supervised control. Tied task means receive the average of their ranks, and the 11 task ranks are weighted equally.

Comparison with constituent objectives. Using the matched reference within each panel of Tables 5 and 6, the best staged schedule in each backbone–encoding setting achieves a higher mean type gain than either of its constituent objectives alone. With both encodings, TS→REL-HIST reaches +3.02% on HETEROGNN, compared with +0.11% for TEMP-SUB and +1.14% for REL-HIST alone. On THGFM, TS→REL-FUTURE reaches +1.06%, compared with −1.28% for TEMP-SUB and −0.52% for REL-FUTURE alone. Without our encodings, TS→REL-FUTURE leads on HETEROGNN at +2.51%, compared with −0.88% for TEMP-SUB and +1.34% for REL-FUTURE alone. TS→REL-HIST leads on THGFM at +1.58%, compared with −2.25% for TEMP-SUB and −0.24% for REL-HIST alone. The gains over the stronger constituent are thus 1.88 and 1.58 percentage points with our encodings, and 1.17 and 1.83 percentage points without our encodings, for HETEROGNN and THGFM, respectively.

Comparison with baseline pretraining. Under the matched A0/B0 references, TS→CTRL is the strongest baseline by mean type gain on HETEROGNN in both panels, reaching +1.62% without our encodings and +2.71% with both encodings. On THGFM, TS→PT-HGNN leads the baselines without our encodings at +1.16%, while TS→TVE leads with both encodings at −0.02%. Our best schedules exceed these baseline gains by 0.88 and 0.31 percentage points on HETEROGNN, and 0.42 and 1.08 points on THGFM, respectively. All six TS-initialized baselines remain below the matched supervised control on THGFM with both encodings, whereas both of our proposed schedules exceed it.

Dependence on the second-stage objective. Staging does not improve every objective. On HET-EROGNN with both encodings, the mean type gain of REL-FUTURE decreases from +3.00% alone to +2.83% after TEMP-SUB, and that of ATTRMASKING decreases from +2.64% to +2.19%. In the same panel, REL-HIST improves from +1.14% to +3.02%, while PT-HGNN improves from +0.53% to +1.89%. The effect also varies across backbones: with both encodings on THGFM, REL-FUTURE improves from −0.52% alone to +1.06% after TEMP-SUB, unlike the decrease on HETEROGNN. The temporal baselines show the same dependence on the backbone and encoding configuration. With both encodings on HETEROGNN, CTRL, TGPM, and TVE improve from +0.93%, +1.53%, and +1.36% alone to +2.71%, +2.34%, and +2.22% after TEMP-SUB, respectively. On THGFM with both encodings, staging improves CTRL from −1.50% to −0.45% and TVE from −0.54% to −0.02%, while TGPM is nearly unchanged at −1.37% in both cases at the reported precision. These comparisons show that the benefit of TEMP-SUB initialization depends on both the subsequent objective and the backbone, consistent with the variation observed in the main ablation study.

## C ARCHITECTURE AND TRAINING CONFIGURATION

All five backbones use three graph layers with hidden dimension 128; the GTs use eight attention heads. Tables 8 and 9 list the detailed settings for HETEROGNN and THGFM in the controlled study. Both use the same materialized table features and temporally sampled neighborhoods. We sample three hops with fanouts of 128, 64, and 32 at the first, second, and third hops, respectively.

Attribute processing. Primary and foreign keys define graph structure and are excluded from attribute features. The PyTorch Frame encoders in Table 7 map each remaining column to 128 dimensions. Parameters are specific to each table, with separate embeddings or maps for its columns. Tables without attributes receive a constant numerical feature. Text vectors are precomputed with average word embeddings glove.6B.300d and remain fixed; the column-specific maps from 300 to 128 dimensions are trainable. Timestamp attributes encode year, month, day, day of week, hour, minute, and second, with median-timestamp imputation for missing values. These calendar features are separate from the record-age encoding in TIMEMIX.

For a row v in table $a = \phi ( v )$ with $m _ { a }$ attribute columns, let $b _ { v , j } \in \mathbb { R } ^ { 1 2 8 }$ be the embedding of column j. We concatenate these embeddings in a fixed column order and apply the table’s ResNet:

$$
b _ { v } = [ b _ { v , 1 } \rVert \cdot \cdot \cdot \rVert b _ { v , m _ { a } } ] \in \mathbb { R } ^ { 1 2 8 m _ { a } } , \qquad x _ { v } = \mathrm { R e s N e t } _ { a } ( b _ { v } ) \in \mathbb { R } ^ { 1 2 8 } ,\tag{11}
$$

where ∥ denotes concatenation. Each table uses a four-layer ResNet with hidden dimension 128 to produce its row representations. The column encoders, ResNets, graph layers, and enabled temporal modules are trained jointly with the current objective’s head.

Training and evaluation. We optimize all models using AdamW at the constant learning rate and weight decay in Table 9. Single-stage pretraining uses 10 epochs with 10,000 sampled steps per epoch, for a total budget of 100,000 steps. Two-stage pretraining allocates 10 epochs with 5,000 sampled steps per epoch to each stage, giving 50,000 steps per stage and the same total budget of

Table 7: Column encoders used by both backbones. Only types present in a table are instantiated; all outputs have dimension 128.
<table><tr><td>Attribute type</td><td>PyTorch Frame encoder</td><td>Operation</td></tr><tr><td>Categorical</td><td>EmbeddingEncoder</td><td>Learned category lookup.</td></tr><tr><td>Numerical</td><td>LinearEncoder</td><td>Standardize using column statistics, then apply a learned affine map. Missing values are imputed with the column mean.</td></tr><tr><td rowspan="2">Multicategorical</td><td>MultiCategorical</td><td rowspan="2">Average learned embeddings of the categories in a cell.</td></tr><tr><td>EmbeddingEncoder</td></tr><tr><td>Vector / text embedding</td><td>LinearEmbeddingEncoder</td><td>Apply a learned affine map to the input vector.</td></tr><tr><td>Timestamp</td><td>TimestampEncoder</td><td>Encode the year positionally and other calendar fields cyclically, then apply a learned linear map with bias.</td></tr></table>

Table 8: Architecture settings for the selected GNN and GT. Temporal settings apply when the corresponding proposed encoding is enabled.
<table><tr><td>Setting</td><td>GNN (HETEROGNN)</td><td>GT (THGFM)</td></tr><tr><td>Graph layers / hidden width</td><td>3 / 128</td><td>3 /128</td></tr><tr><td>Table feature encoder</td><td>4-layer ResNet, width 128</td><td>4-layer ResNet, width 128</td></tr><tr><td>Table-encoder dropout Text embedding / learned mapping</td><td>0.2 Fixed GloVe / 300 → 128</td><td>0.2 Fixed GloVe / 300 → 128</td></tr><tr><td>Heads per attention branch</td><td>Not applicable</td><td>8 (16 dimensions each)</td></tr><tr><td>Graph-operator dropout</td><td>None</td><td>0.05</td></tr><tr><td>Across-relation aggregation</td><td>Sum</td><td>Sum</td></tr><tr><td>Post-layer normalization / activation</td><td>Node-wise LayerNorm / ReLU</td><td>Node-wise LayerNorm / ReLU</td></tr><tr><td>Downstream head</td><td>Linear, 128 → C</td><td>Linear, 128 → C</td></tr><tr><td>Temporal sampling / hop fanouts</td><td>Uniform / (128, 64, 32)</td><td>Uniform / (128, 64, 32)</td></tr><tr><td>TIMEMIX dimension / fusion</td><td>128 / addition</td><td></td></tr><tr><td>TiMERoPE mode / time unit / phase</td><td></td><td>128 / addition</td></tr><tr><td>cap</td><td>Symmetric / 1 hour / 16</td><td>Symmetric / 1 hour / 16</td></tr></table>

C is the number of classes for multiclass classification and 1 for binary classification or regression. Graphoperator dropout refers to the graph module, separately from the table encoder. The temporal operators are detailed in Appendix D.

Table 9: Training settings shared by the selected GNN and GT. Single- and two-stage pretraining each receive 100,000 steps in total.
<table><tr><td>Setting</td><td>GNN and GT</td></tr><tr><td>Optimizer / learning rate / weight decay</td><td>AdamW / 2 × 10−3 / 10−5</td></tr><tr><td>Learning-rate schedule</td><td>Constant; no warmup or decay</td></tr><tr><td>Single-stage pretraining epochs / steps per epoch</td><td>10 / 10,000</td></tr><tr><td>Two-stage pretraining epochs / steps per epoch (each stage)</td><td>10 / 5,000</td></tr><tr><td>Total pretraining steps (either schedule)</td><td>100,000</td></tr><tr><td>Pretraining batch size</td><td>256; 512 for the H&amp;M curricula</td></tr><tr><td>Downstream epochs / steps per epoch Downstream batch size</td><td>10 / 10,000</td></tr><tr><td></td><td>1,024 for Arxiv and F1; 512 for Avito, H&amp;M, and Event</td></tr><tr><td>Fine-tuning scope Downstream losses</td><td>Full encoder and newly initialized task head</td></tr><tr><td></td><td>Cross-entropy (multiclass), binary cross-entropy with logits (binary),  $L _ { 1 }$  (regression)</td></tr><tr><td>Validation frequency Downstream checkpoint criterion</td><td>Every downstream epoch</td></tr><tr><td></td><td>Maximum validation accuracy or ROC-AUC; minimum validation MAE</td></tr><tr><td>Random seeds</td><td>42, 43, 44, 45</td></tr><tr><td>Reported task statistics</td><td>Mean and standard deviation across four runs</td></tr></table>

100,000 steps. These budgets apply to both our objectives and the pretraining baselines, including their staged variants. Each downstream fine-tuning run uses 10 epochs with 10,000 steps per epoch, totaling 100,000 steps; models trained directly on downstream tasks without pretraining use the same budget. There is no patience-based early stopping. For all pretraining strategies, we select the checkpoint with the lowest training loss. The next stage loads the selected encoder and initializes its objective-specific modules and AdamW optimizer afresh. For each seed in {42, 43, 44, 45}, we run pretraining where applicable, fine-tuning, validation-based checkpoint selection, and test evaluation independently, using the same chronological splits. The mean validation ranks used to choose one study backbone per family, as a trade-off to reduce computational cost, are described in Appendix B.1. Reported means and standard deviations summarize these runs; relative gains are computed from the task means.

Objective settings. REL-HIST uses $K _ { \mathrm { n e g } } ~ = ~ 4$ negative destinations per positive link. REL-FUTURE uses $K _ { \mathrm { f u t } } = 2$ negatives, candidate horizons $\mathcal { H } = \{ 1 , 7 , 3 0 , 9 \bar { 0 } , 3 6 \bar { 5 } \}$ days, count-loss weight $\lambda _ { \mathrm { c o u n t } } = 0 . 2 5$ , and Smooth L1 transition threshold $\beta _ { \mathrm { S L } } = 1$ . TEMP-SUB uses edge-drop probability $p _ { \mathrm { e d g e } } = 0 . 2$ , feature-mask probability $p _ { \mathrm { m a s k } } = 0 . 2$ , and contrastive temperature $\bar { T } = 0 . \bar { 2 }$ Its graph and seed contrastive losses both have unit weight. The seed regularizer uses variance weight $\lambda _ { \mathrm { v a r } } = 1$ , covariance weight $\lambda _ { \mathrm { c o v } } = 0 . 0 1$ , target standard deviation $\gamma = 1$ , and variance stabilizer $\epsilon _ { \mathrm { v a r } } = 1 0 ^ { - 4 }$ ; its full definition is in Appendix E. These settings are shared across backbones and stages. We performed no hyperparameter search; the values here and in Appendix D are fixed experimental choices, and their optimality has not been evaluated.

Horizon eligibility and window boundaries. A horizon δ is eligible only if the seed is available at $\tau = t _ { \mathrm { e v t } } - \delta$ . Events at τ belong to the history and are excluded from the target count; events at $\tau + \delta$ are included. A negative seed must be available at τ and have no events of the selected relation in this window. An event at the seed’s first observation cannot define a positive example for any positive horizon.

## D TEMPORAL ENCODING DETAILS

TIMEMIX parameterization. TIMEMIX uses $K _ { \mathrm { t i m e } } = 4$ trainable log-scales $\eta _ { k }$ , initialized so that the scales exp(η<sub>k</sub>) equal 1 second, 1 minute, 1 hour, and 1 day, respectively. Durations are represented in seconds, with positive scales $s _ { k } = \mathrm { c l i p } ( \exp ( \eta _ { k } ) , s _ { \mathrm { m i n } } , s _ { \mathrm { m a x } } )$ , where $s _ { \operatorname* { m i n } } = 1$ second and $s _ { \mathrm { m a x } } = 2 5 9 2 0 0 0$ seconds (30 days). The bounds constrain the scales, not the record ages. The mixture weights are

$$
\pi _ { k } = \frac { \exp ( \gamma _ { k } ) } { \sum _ { j = 1 } ^ { K _ { \mathrm { t i m e } } } \exp ( \gamma _ { j } ) } ,\tag{12}
$$

where $\gamma _ { k }$ is a trainable logit, initialized equally across scales to give uniform weights. Parameter sharing and missing-timestamp handling follow Section 3.1.

TIMEROPE implementation. TIMEROPE computes the bounded, signed log-interval $\rho _ { u v }$ defined in Section 3.1, with time scale $s _ { \mathrm { R } } = 3 6 0 0$ seconds (1 hour) and phase cap $\rho _ { \mathrm { m a x } } = 1 6$ . For a feature dimension $d ,$ let $d _ { \mathrm { e v e n } }$ be the largest even integer no greater than d. Consecutive dimension pairs are rotated with frequencies $\omega _ { j } = 1 0 0 0 0 ^ { - 2 j / d _ { \mathrm { e v e n } } } , j = 0 , \ldots , d _ { \mathrm { e v e n } } / 2 - 1$ . For pair $\left( x _ { 2 j } , x _ { 2 j + 1 } \right)$ ,

$$
R _ { j } ( \rho ) \left[ \begin{array} { c c c c } { x _ { 2 j } } \\ { x _ { 2 j + 1 } } \end{array} \right] = \left[ \begin{array} { c c c c } { \cos ( \omega _ { j } \rho ) } & { - \sin ( \omega _ { j } \rho ) } \\ { \sin ( \omega _ { j } \rho ) } & { \cos ( \omega _ { j } \rho ) } \end{array} \right] \left[ \begin{array} { c c c c } { x _ { 2 j } } \\ { x _ { 2 j + 1 } } \end{array} \right] .\tag{13}
$$

An unmatched final dimension is unchanged. GNN messages use the feature width; GT rotations use the attention-head width.

GNN message construction. This message construction extends temporal rotation to GNN aggregation without query–key attention.

GT attention weighting. We retain the query–key rotation mechanism of THGFM’s Rotary Temporal Attention (Peng et al., 2026), using our bounded, signed log-interval $\rho _ { u v }$ as its phase input. For a fixed relation r and attention head, let $q _ { v } , k _ { u } , p _ { u } \in \mathbb { R } ^ { \bar { d } _ { h } }$ be the query, key, and value projected

from the current layer inputs $y _ { v }$ and $y _ { u }$ , with any branch-specific endpoint adapters applied before projection. Relation, branch, and head indices are omitted for clarity. The symmetric $\mathrm { q k }$ mode computes

$$
\begin{array} { r l } & { \widehat { q } _ { u v } = R ( + \rho _ { u v } / 2 ) q _ { v } , \qquad \widehat { k } _ { u v } = R ( - \rho _ { u v } / 2 ) k _ { u } , } \\ & { \alpha _ { u v } = \mathrm { s o f t m a x } _ { u \in \mathcal { N } _ { r } ( v ) } \left( \frac { \widehat { q } _ { u v } ^ { \top } \widehat { k } _ { u v } } { \sqrt { d _ { h } } } \right) . } \end{array}\tag{14}
$$

Here $\mathcal { N } _ { r } ( v )$ contains incoming neighbors under relation r. Values remain unrotated, and the head output $\begin{array} { r } { \mathrm { i s } \sum _ { u \in \mathcal { N } _ { r } ( v ) } \alpha _ { u v } p _ { u } } \end{array}$ . For THGFM, rotation is applied separately in the relation-specific attention branch and the shared-space branch with endpoint adapters. Their outputs are then combined by the backbone’s type-conditioned gated sum.

## E PRETRAINING OBJECTIVE DETAILS

This section specifies the decoders, losses, and sampling details for Section 3.2. We retain the main text’s notation, including historical node representations $h _ { v } \in \mathbb { R } ^ { d }$ . Hyperparameter values are in Appendix C.

Historical link recovery. A link’s observation time is the timestamp of the row containing the foreign key, or the other endpoint’s timestamp if the first is missing. Links with no timestamped endpoint use the pretraining cutoff. For each positive link, we draw $K _ { \mathrm { n e g } }$ negative destinations under the constraints in Section 3.2. The bilinear score is

$$
s _ { \mathrm { h i s t } } ( u , v , r ) = h _ { u } ^ { \top } W _ { r } h _ { v } ,\tag{15}
$$

where $W _ { r } \in \mathbb { R } ^ { d \times d }$ is learned for relation $^ { r } \cdot$ The ranking loss is

$$
{ \mathcal { L } } _ { \mathrm { R e l - H i s t } } = \frac { 1 } { K _ { \mathrm { n e g } } } \sum _ { k = 1 } ^ { K _ { \mathrm { n e g } } } \mathrm { s o f t p l u s } \big [ s _ { \mathrm { h i s t } } ( u , v _ { k } ^ { - } , r ) - s _ { \mathrm { h i s t } } ( u , v , r ) \big ] ,\tag{16}
$$

where ${ v } _ { k } ^ { - }$ is the k-th negative destination and softplus $( x ) = \log ( 1 + e ^ { x } )$

Future relation activity. Each forward foreign-key link $e = ( u  s )$ contributes one event of relation r to $\mathcal { E } _ { r } ( s )$ ; reverse edges are not counted again. The timestamp assignment uses $t _ { e } = t _ { u }$ when the source table has a timestamp field, and otherwise uses $t _ { s }$ when the destination table has one. Relations between two tables without timestamps are excluded. We retain only events at or before the pretraining cutoff whose timestamped endpoints are available by $t _ { e }$

For a timestamped seed, $t _ { \mathrm { f i r s t } } ( s ) = t _ { s }$ . Otherwise, availability is estimated from its earliest incident relation time within the cutoff. Seeds without an assigned availability time are excluded.

We use candidate horizons $\mathcal { H } = \{ 1 , 7 , 3 0 , 9 0 , 3 6 5 \}$ days, converted to the same unit as the timestamps. For an event involving s at $t _ { \mathrm { e v t } }$ , the eligible horizons are

$$
\mathcal { D } ( s , t _ { \mathrm { e v t } } ) = \{ \delta \in \mathcal { H } : t _ { \mathrm { f i r s t } } ( s ) \leq t _ { \mathrm { e v t } } - \delta \} .\tag{17}
$$

If time assignment falls back to $t _ { s }$ , then $t _ { e } = t _ { \mathrm { f i r s t } } ( s )$ , so no positive horizon is eligible. Thus, relations retained for this objective use source timestamps, as described in Section 3.2. The eligible event and relation sets are

$$
\begin{array} { r } { \mathcal { E } _ { r } ^ { + } = \{ ( s , e ) : e \in \mathcal { E } _ { r } ( s ) , \mathcal { D } ( s , t _ { e } ) \neq \emptyset \} , \qquad \mathcal { R } _ { \mathrm { f u t } } = \{ r : \mathcal { E } _ { r } ^ { + } \neq \emptyset \} . } \end{array}\tag{18}
$$

We sample a relation, an event, and a horizon uniformly from these sets:

$$
\begin{array} { r l r } & { r \sim \mathrm { U n i f } ( \mathcal { R } _ { \mathrm { f u t } } ) , } & { ( s ^ { + } , e ^ { + } ) \sim \mathrm { U n i f } ( \mathcal { E } _ { r } ^ { + } ) , } \\ & { \delta \sim \mathrm { U n i f } ( \mathcal { D } ( s ^ { + } , t _ { e ^ { + } } ) ) , } & { \tau = t _ { e ^ { + } } - \delta . } \end{array}\tag{19}
$$

Here Unif denotes uniform sampling and $t _ { e ^ { + } } = t _ { \mathrm { e v t } }$ in the main text. Event sampling uses replacement. For the same relation and window, negative seeds are drawn from

$$
\mathcal V ^ { - } ( s ^ { + } , r , \tau , \delta ) = \{ s : \phi ( s ) = \phi ( s ^ { + } ) , \ : t _ { \mathrm { f i r s t } } ( s ) \leq \tau , \ : c _ { r } ( s ; \tau , \delta ) = 0 \} .\tag{20}
$$

We draw $K _ { \mathrm { f u t } }$ negatives, using distinct seeds when possible and reusing valid draws if necessary. Attempts without valid negatives are skipped. Targets follow Equation $( 5 ) ;$ ; window boundaries are specified in Appendix C.

The occurrence and count decoders share a horizon encoder and layer normalization, but use separate relation embeddings and biases. The occurrence logit is

$$
w _ { r , \delta } = w _ { r } + g _ { \mathrm { h o r } } ( \log ( 1 + \delta _ { \mathrm { d a y } } ) ) , \qquad s _ { \mathrm { a c t } } ( s , r , \delta ) = \frac { \mathrm { L N } ( h _ { s } ) ^ { \top } \left( w _ { r , \delta } / \| w _ { r , \delta } \| _ { 2 } \right) } { \sqrt { d } } + b _ { r } ,\tag{21}
$$

where $w _ { r } \in \mathbb { R } ^ { d }$ and $b _ { r }$ are learned relation parameters, $g _ { \mathrm { h o r } } : \mathbb { R }  \mathbb { R } ^ { d }$ is the horizon encoder, $\delta _ { \mathrm { d a y } }$ is the horizon in days, LN is layer normalization, and $\| \cdot \| _ { 2 }$ is the Euclidean norm. The count score $s _ { \mathrm { c o u n t } } ( s , r , \delta )$ uses the same form with its own relation embedding and bias. Its nonnegative prediction is

$$
\widehat { y } _ { \mathrm { c o u n t } } = \mathrm { s o f t p l u s } ( s _ { \mathrm { c o u n t } } ( s , r , \delta ) ) .\tag{22}
$$

Using the targets in Equation (5), the losses for a positive $s ^ { + }$ and negatives $s _ { k } ^ { - }$ at fixed $( r , \tau , \delta )$ are

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { a c t } } = \frac { 1 } { 2 } \left[ \mathrm { s o f t p l u s } ( - s _ { \mathrm { a c t } } ( s ^ { + } , r , \delta ) ) + \frac { 1 } { K _ { \mathrm { f u t } } } \sum _ { k = 1 } ^ { K _ { \mathrm { f u t } } } \mathrm { s o f t p l u s } ( s _ { \mathrm { a c t } } ( s _ { k } ^ { - } , r , \delta ) ) \right] , } \\ { \mathcal { L } _ { \mathrm { c o u n t } } = \frac { 1 } { 2 } \left[ \mathrm { S m o o t h L } 1 ( \widehat { y } _ { \mathrm { c o u n t } } ^ { + } , y _ { \mathrm { c o u n t } } ^ { + } ) + \frac { 1 } { K _ { \mathrm { f u t } } } \sum _ { k = 1 } ^ { K _ { \mathrm { f u t } } } \mathrm { S m o o t h L } 1 ( \widehat { y } _ { \mathrm { c o u n t } , k } ^ { - } , 0 ) \right] , } \end{array}\tag{23}
$$

$$
{ \mathcal { L } } _ { \mathrm { R e l - F u t u r e } } = { \mathcal { L } } _ { \mathrm { a c t } } + \lambda _ { \mathrm { c o u n t } } { \mathcal { L } } _ { \mathrm { c o u n t } } .
$$

Superscripts + and − identify positive and negative examples; negative count targets are zero. SmoothL1 uses threshold $\beta _ { \mathrm { S I } }$ . Each loss equally weights the positive example and the mean over negatives, and $\lambda _ { \mathrm { c o u n t } }$ weights the count loss.

Symmetric subgraph contrast. For each minibatch, we sample seeds of one node type without replacement. For its B historical neighborhoods, let $z _ { i } ^ { ( 1 ) }$ and $z _ { i } ^ { ( 2 ) }$ be neighborhood i’s two pooled views after the shared projection head. The symmetric InfoNCE loss is

$$
S _ { i j } = \frac { \cos ( z _ { i } ^ { ( 1 ) } , z _ { j } ^ { ( 2 ) } ) } { T } , \qquad \mathcal { L } _ { \mathrm { g r a p h } } = \frac { 1 } { 2 B } \sum _ { i = 1 } ^ { B } [ \mathrm { C E } ( S _ { i , : } , i ) + \mathrm { C E } ( S _ { : , i } , i ) ] ,\tag{24}
$$

where $S _ { i j }$ is cosine similarity scaled by temperature $T , \operatorname { C E } ( \cdot , i )$ is cross-entropy with positive index $i ,$ and $S _ { i , i }$ and $S _ { : , i }$ are row i and column i of the score matrix. The same loss applied to seed representations through a separate projection head gives $\mathcal { L } _ { \mathrm { s e e d } }$ . Both projection heads have the architecture Linear $\cdot ( \bar { d , } d )  \dot { \mathrm { L a y e r N o r } } \dot { \mathrm { m } } ( d )  \mathrm { G E L U } \breve { \cup }  \mathrm { L i n e a r } ( d , d )$ . Each head is shared across the two views; the graph and seed heads have separate parameters.

Seed representation regularization. Let $X ^ { ( v ) } \in \mathbb { R } ^ { B \times d }$ contain the encoder’s seed representations in view $v \in \{ 1 , 2 \}$ before the seed projection head. We apply the variance and covariance components of VICReg (Bardes et al., 2022) to these representations, with cross-view agreement provided by $\mathcal { L } _ { \mathrm { s e e d } }$ . For a single view X, define

$$
\begin{array} { r l r } {  { \bar { X } _ { j } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } X _ { i j } , } } \\ & { } & { \mathcal { V } ( X ) = \frac { 1 } { d } \displaystyle \sum _ { j = 1 } ^ { d } \mathrm { m a x } ( 0 , \gamma - \sigma _ { j } ( X ) ) . } \end{array}\tag{25}
$$

The variance uses divisor $B$ and is computed on the unnormalized encoder outputs. This hinge penalizes feature dimensions whose standard deviation across seeds falls below γ.

For the covariance term, we first apply LayerNorm independently to each row of X, without learnable scale or bias and with numerical stabilizer $\epsilon _ { \mathrm { L N } } \doteq 1 0 ^ { - 5 }$ . Let $Y = \mathrm { L a y e r N o r m } ( X )$ and

$\begin{array} { r } { \bar { Y } _ { j } = B ^ { - 1 } \sum _ { i = 1 } ^ { B } Y _ { i j } } \end{array}$ . We then compute

$$
C _ { j k } ( X ) = \frac { 1 } { B - 1 } \sum _ { i = 1 } ^ { B } ( Y _ { i j } - \bar { Y } _ { j } ) ( Y _ { i k } - \bar { Y } _ { k } ) , \qquad \mathcal { C } ( X ) = \frac { 1 } { d } \sum _ { j \neq k } C _ { j k } ( X ) ^ { 2 } .\tag{26}
$$

Thus, covariance is computed across seeds after normalization within each seed; its off-diagonal penalty reduces redundancy between feature dimensions. The sum includes all ordered pairs $j \neq$ k and is normalized by d. Both regularizers are set to zero when $B < 2 ;$ the covariance term is also zero when d $< 2$

We average each term over the two views and form

$$
\mathcal { R } _ { \mathrm { r e g } } = \frac { 1 } { 2 } \sum _ { v = 1 } ^ { 2 } \left[ \lambda _ { \mathrm { v a r } } \mathcal { V } ( X ^ { ( v ) } ) + \lambda _ { \mathrm { c o v } } \mathcal { C } ( X ^ { ( v ) } ) \right] , \qquad \mathcal { L } _ { \mathrm { T e m p } \cdot \mathrm { S u b } } = \mathcal { L } _ { \mathrm { g r a p h } } + \mathcal { L } _ { \mathrm { s e e d } } + \mathcal { R } _ { \mathrm { r e g } } .\tag{27}
$$

The weights and variance parameters are given in Appendix C.

## F PRETRAINING AND FINE-TUNING CURVES

The curves cover both backbones under two settings: DAYPE without TIMEROPE, and TIMEMIX with TIMEROPE. These are the encoding configurations of the A0 and B0 controls, respectively. Pretraining panels are grouped by database; solid lines show single-objective training and dashed lines show stage 2 after TEMP-SUB (TS). Epoch counts restart at each stage. Each epoch comprises 10,000 steps for single-stage pretraining and 5,000 steps for each stage of two-stage pretraining, so equal epoch counts do not indicate equal numbers of steps in these panels. Metrics should be compared within a row because objectives use different scales. Fine-tuning panels use 10,000 steps per epoch and are grouped by task and show validation accuracy or ROC–AUC (higher is better) and MAE (lower is better). Test results are in Tables 5 and 6.

HeteroGNN · without TimeMix + TimeRoPE | Pretraining validation  
![](images/1742370473b7080358b76541d0194de1eafdd89ab5185d3170114fd6fe38f28e.jpg)  
Figure 7: Pretraining metrics for HETEROGNN with DAYPE and without TIMEROPE: single objectives.

HeteroGNN · without TimeMix + TimeRoPE | Pretraining validation  
![](images/bbed62f58eefc38fde3dc7ab11a8d20e4002f1e1444b8ee5e7f99b5dcde3867b.jpg)  
Figure 8: Pretraining metrics for HETEROGNN with DAYPE and without TIMEROPE: single objectives and stage 2 after TEMP-SUB.

HeteroGNN · without TimeMix + TimeRoPE | Downstream validation  
![](images/23d0310ee3105878128fc85eecc9b9a44cbece7953d915d08c57266ab9b878ca.jpg)  
Figure 9: Fine-tuning validation metrics for HETEROGNN with DAYPE and without TIMEROPE: single-stage baselines.

![](images/a0c9300385ac48af8563fa8b08e0b27a948c52886225d0d7d5bbabb92b3eae9a.jpg)  
Figure 10: Fine-tuning validation metrics for HETEROGNN with DAYPE and without TIMEROPE: single and staged objectives.

HeteroGNN · DayPE, without TimeRoPE | Pretraining loss (seed 43)  
![](images/bc22006f6e02da59bea613c0857aed907ea26139e613201bcf39f5b21479354d.jpg)  
Figure 11: Pretraining losses for HETEROGNN with DAYPE and without TIMEROPE: CTRL, TGPM, TVE, and the shared TEMP-SUB first stage.

HeteroGNN · DayPE, without TimeRoPE | Downstream validation (seed 43)  
![](images/99bb66abf748e26dc31d5497dc1432b3d4fefdafad30f267b0d566d197e5c354.jpg)  
Figure 12: Fine-tuning validation metrics for HETEROGNN with DAYPE and without TIMEROPE: CTRL, TGPM, TVE, and their TS-initialized variants.

HeteroGNN · with TimeMix + TimeRoPE | Pretraining validation  
![](images/b580099b80aa85f0a68e49430723c89b7a2ae8d5321e616283453b654d79a828.jpg)  
Figure 13: Pretraining metrics for HETEROGNN with TIMEMIX and TIMEROPE: single objectives.

HeteroGNN · with TimeMix + TimeRoPE | Pretraining validation  
![](images/a47da1427fc89ee4c88e48ad95c2558bf9edf96ff068172b7dcc62db8e83d59e.jpg)  
Figure 14: Pretraining metrics for HETEROGNN with TIMEMIX and TIMEROPE: single objectives and stage 2 after TEMP-SUB.

![](images/ddfe1f32a91151d96149f4864b6a8b52241150b176be95bec235e1c7ebd2f999.jpg)  
Figure 15: Fine-tuning validation metrics for HETEROGNN with TIMEMIX and TIMEROPE: singlestage baselines.

![](images/85a69923dd99382535387a55c6e648819800e4cc66d3ef5dea6384ad8e18fde4.jpg)  
Figure 16: Fine-tuning validation metrics for HETEROGNN with TIMEMIX and TIMEROPE: single and staged objectives.

![](images/2552f39ccbab8896d9b6822a12606251546031809e3e8a11e72eeb351802b7ef.jpg)  
Figure 17: Pretraining losses for HETEROGNN with TIMEMIX and TIMEROPE: CTRL, TGPM, TVE, and the shared TEMP-SUB first stage.

![](images/66b5eb3afb5e7475ecfd5624c0d5aa97563df35745bd109cfb8e865b9832494a.jpg)  
Figure 18: Fine-tuning validation metrics for HETEROGNN with TIMEMIX and TIMEROPE: CTRL, TGPM, TVE, and their TS-initialized variants.

THGFM · without TimeMix + TimeRoPE | Pretraining validation  
![](images/ae4e8849ef29a25942e70e3e60eec5313aec75b2b15208933f3c169caef9bedf.jpg)  
Figure 19: Pretraining metrics for THGFM with DAYPE and without TIMEROPE: single objectives.

THGFM · without TimeMix + TimeRoPE | Pretraining validation  
![](images/44fb74312b72f87d6fce182201859c8ec6a18331829d3db0e0e07a358f67023e.jpg)  
Figure 20: Pretraining metrics for THGFM with DAYPE and without TIMEROPE: single objectives and stage 2 after TEMP-SUB.

THGFM · without TimeMix + TimeRoPE | Downstream validation  
![](images/2e1084b19fdca8cb42c796e62227b99fec5d9907b17616eed404a2046e837964.jpg)  
Figure 21: Fine-tuning validation metrics for THGFM with DAYPE and without TIMEROPE: single stage baselines.

![](images/c89f381537bc1eab0725970b94a7d53fedf69280d0c856b6288f9aef5c87bb3c.jpg)  
Figure 22: Fine-tuning validation metrics for THGFM with DAYPE and without TIMEROPE: single and staged objectives.

THGFM · DayPE, without TimeRoPE | Downstream validation (seed 43)  
THGFM · DayPE, without TimeRoPE | Pretraining loss (seed 43)  
![](images/aad647873866e2c20db010e27586649508336b736fb03074a75e02229b779e09.jpg)  
Figure 23: Pretraining losses for THGFM with DAYPE and without TIMEROPE: CTRL, TGPM, TVE, and the shared TEMP-SUB first stage.

![](images/b28902092bbbaa150ead071bb68394479d3a2edbfcf32b85e562a6981222e5a7.jpg)  
Figure 24: Fine-tuning validation metrics for THGFM with DAYPE and without TIMEROPE: CTRL, TGPM, TVE, and their TS-initialized variants.

THGFM · with TimeMix + TimeRoPE | Pretraining validation  
![](images/d12714e5ea4e81342ca0126f021dfee7d688a317b5c514bb03822d898e004eb3.jpg)  
Figure 25: Pretraining metrics for THGFM with TIMEMIX and TIMEROPE: single objectives.

THGFM · with TimeMix + TimeRoPE | Pretraining validation  
![](images/93021470c36d27feb59ed6d7eae350eb9b2de46a10fa66b6f14904bd59487c15.jpg)  
Figure 26: Pretraining metrics for THGFM with TIMEMIX and TIMEROPE: single objectives and stage 2 after TEMP-SUB.

![](images/96e993dfab5136c8e2713311f137cd4e6676eddfbfa0b5799f487aaa72b61eda.jpg)  
Figure 27: Fine-tuning validation metrics for THGFM with TIMEMIX and TIMEROPE: single-stage baselines.

![](images/86b6d196a17378a8fad58a3815ffd19dc68a9e6f8d764e47488448418c7cd04c.jpg)  
Figure 28: Fine-tuning validation metrics for THGFM with TIMEMIX and TIMEROPE: single and staged objectives.

THGFM · TimeMix + TimeRoPE | Pretraining loss (seed 43)  
![](images/aeb318dc3a6923f04fd18370027eb9dac4161501abf000cb0737a7fbe71293ba.jpg)  
Figure 29: Pretraining losses for THGFM with TIMEMIX and TIMEROPE: CTRL, TGPM, TVE, and the shared TEMP-SUB first stage.

![](images/e0e1769b0b97ef665074ddb5da83b5c7c0a61f6556aa41594ea20eae49e7d3ea.jpg)  
Figure 30: Fine-tuning validation metrics for THGFM with TIMEMIX and TIMEROPE: CTRL, TGPM, TVE, and their TS-initialized variants.