# STARS: From Spatiotemporal Dynamics to Social Representations in Human-Robot Interaction

Nathan Tsoi<sup>\*</sup>, Michael J. Munje<sup>\*</sup>, Tejas Oberoi, Rishab Maheshwari, Pengen Zheng, Tanush Chauhan, Peter Stone, Joydeep Biswas Department of Computer Science, The University of Texas at Austin \* Equal Contribution

Abstract: Fielding socially competent robots requires joint reasoning over spatial information and the social information it can convey, such as group motion and personal space. While current models excel at either spatial reasoning (e.g., modeling physical dynamics) or social reasoning (e.g., parsing text-based interactions), few natively unify both. Moreover, data scarcity in Human-Robot Interaction (HRI) makes training large models from scratch challenging. To bridge this gap, we introduce STARS: SpatioTemporal Autoencoded Representations for Social-interaction, a method for learning event-level relational representations that models short windows of multi-agent interactions as spatiotemporal graphs. A self-supervised message-passing graph neural network autoencoder then compresses these physical dynamics into a compact latent space for each node. By explicitly modeling agents and their interactions as graph structure, STARS injects a strong relational inductive bias to naturally capture the underlying social situation. To assess the utility of these learned representations, we evaluate STARS’ representations on the SEAN Together dataset, which provides VR-based navigation data annotated with subjective human perceptions, and the SocialNav-SUB benchmark for social scene understanding. When trained on STARS representations, linear probes achieve highly competitive performance on perception prediction and pedestrian action classification tasks. Specifically, STARS exhibits strong data efficiency, particularly on pedestrian action classification, where STARS-He consistently outperforms unstructured baselines across labeled-data fractions, including in low-data settings. Furthermore, qualitative analysis of the learned latent space reveals that STARS representations can capture high-level social semantics, demonstrating cohesive clustering of distinct human action intents without optimizing the encoder with a supervised action classification objective. An overview of this paper along with the code can be found at https://larg.github.io/stars.

Keywords: Human-Robot Interaction, Social Navigation, Representation Learning, Graph Neural Networks

## 1 Introduction

Deploying socially competent robots in human spaces requires autonomous robots to reason jointly using spatial knowledge to plan collision-free paths and social knowledge to perceive context, predict behaviors, and adhere to social norms. A robot navigating through a crowded hallway, for example, must plan collision-free motion in the physical environment while also anticipating pedestrian behavior and reacting in a socially compliant manner [1, 2]. Joint reasoning over spatial and social information is central to Human-Robot Interaction (HRI), yet existing approaches tend to address only one or the other. Foundation spatial models parse LiDAR and visual observations into geometric representations, but lack the capacity to understand social dynamics [3, 4]. At the same time, some autoregressive models conditioned on text (LLMs) or text and vision (VLMs) exhibit strong social perception [3], but struggle to process continuous spatial dynamics [5, 6] and precise temporal information [7]. Bridging this gap by training large models from scratch is challenging due to the scarcity of grounded HRI data. Collecting such data requires time and cost intensive human-subject studies, and yields datasets that are small, fragmented across sensor modalities, and highly prone to overfitting when applied to large, unstructured models.

Although most HRI datasets are small and fragmented, often collected in different environments, with different robots, sensors, tasks, and annotations, the premise of this paper is that their recurring social interaction structure [8] can be used to learn reusable representations for downstream social navigation tasks that require reasoning about agents, interactions, and shared context. This interaction structure is naturally relational, as agents move through shared spaces and influence one another over time. We hypothesize that graph-structured representations are well-suited to learning from small, fragmented HRI datasets because they can encode agents as nodes and interactions as edges, injecting strong relational inductive bias [9]. This inductive bias enables message-passing graph neural networks [10] to learn interaction patterns efficiently in limited-data regimes.

We introduce STARS: SpatioTemporal Autoencoded Representations for Social-interaction, a method for self-supervised learning of event-level relational representations. In contrast to the autoregressive prediction of world models, STARS learns a representation over a short time horizon to capture relevant contextual information. We define an interaction scenario as a fixed-duration sequence of events where agents share a physical space. We encode these interaction scenarios into graphs, where directed edges encode sequences of pairwise spatial features over time.

Though our interaction scenario graphs contain only spatiotemporal information, our target downstream tasks are inherently social. By optimizing a self-supervised reconstruction objective over the fixed-duration interaction scenario, STARS compresses physical dynamics into a compact latent space that captures the implicit social situation [11]. We demonstrate our method using two architectural variants that show the learned representations allow a simple linear probe to extract so cial meaning from physical motion, unlocking data-efficient generalization without requiring taskspecific representation learning.

## Our contributions are threefold:

1. We introduce STARS, a self-supervised representation learning method for social navigation, which compresses datasets with different feature dimensions and graph schemas into a latent representation containing features useful for diverse downstream tasks with limited task-specific labeled data.

2. We demonstrate that self-supervised pretraining of relational representations improves data efficiency. By comparing STARS to unstructured baselines in low-data regimes, we show that pretrained graph representations support robust downstream generalization with a fraction of the labeled data and transfer across the two evaluated datasets.

3. We establish that STARS’ self-supervised representations naturally encode social semantics. By analyzing the latent space topology, we illustrate the model’s inherent ability to distinguish complex social behaviors, such as action intents, without a supervised action classification objective.

## 2 Related Work

Spatial and social reasoning in social navigation. Evaluating multi-agent interactions has historically relied on pedestrian trajectory datasets [12, 13, 14] and egocentric robot observations [15, 16]. To model these complex shared environments, Graph Neural Networks (GNNs) have emerged as the standard architecture, as they naturally mirror the topological structure of human crowds [17, 18]. Recent heterogeneous and attention-based spatiotemporal GNNs excel at weighting relational dependencies between diverse entities to extract continuous kinematic interactions [19, 20]. Current learning efforts tend to focus on trajectory forecasting, which aims to minimize spatial displacement error, and therefore learn to reflect future spatial configurations, rather than reusable socia semantics. Recent works have laid the groundwork to push beyond spatial forecasting. Zhang et al. [21] provide a dataset and baseline for predicting human perceptions of robot performance and Munje et al. [6] established a benchmark for VLM understanding of social interactions in navigation. Still, existing trajectory-focused models struggle to generalize to social tasks because they do not explicitly model the normative behaviors and interpersonal associations that define social interactions [22, 11].

Foundation models and the HRI data bottleneck. Foundation and world models demonstrate impressive predictive capabilities for embodied decision-making [23, 24, 25, 26], including strong zero-shot transfer in Vision-Language-Action (VLA) controllers [27, 28, 29]. However, these models rely on massive, unified datasets. In contrast, grounded HRI data is scarce, geographically constrained, and highly fragmented across varying sensor modalities, survey instruments, and tasks [30, 31, 32, 33, 34, 35, 36, 37]. Applying foundation-scale learning to HRI is therefore hindered by data scarcity, leading unstructured models to overfit or fail to generalize across diverse social scenarios [4].

Relational inductive bias. To learn efficiently in low-data regimes, recent approaches have leveraged structured representations. Relational inductive bias [9] constrains neural architectures to represent entities as nodes and interactions as edges, encouraging networks to deduce underlying physical dynamics without relying on hand-engineered features [38]. While this bias has driven advancements in physical object manipulation, its application to extracting social context from interaction remains underexplored [39, 40]. At the same time, generative self-supervised graph representation learning has shown that semantically meaningful embeddings can be extracted without human annotation by reconstructing masked features or topological structures [41, 42, 43].

## 3 Method

In this section, we detail the STARS framework. We first formalize the problem of extracting representations from heterogeneous social navigation datasets, outline our graph construction process, and describe the unsupervised graph autoencoder used to learn the latent social semantics.

## 3.1 Problem Formulation

Let $\mathcal { D } = \{ \mathcal { D } _ { 1 } , \mathcal { D } _ { 2 } , \ldots , \mathcal { D } _ { K } \}$ denote a collection of social navigation datasets (Fig. 2A), where index k identifies a dataset in the collection. Each dataset $\mathcal { D } _ { k }$ contains a set of interaction scenarios $\{ x _ { i } ^ { ( k ) } \}$ An interaction scenario x is a fixed-duration observation of $T _ { k }$ timesteps containing multiple agents sharing a physical space. Rather than operating directly on arbitrary raw sensor inputs, STARS converts each scenario into a common graph representation with agent node attributes and temporal pairwise edge features. Let $\mathcal { A }$ denote the finite set of node types, $d _ { v } ^ { ( k ) }$ the dimension of the node attributes supplied by dataset $k ,$ and $d _ { e }$ the dimension of the pairwise features. We represent each scenario with N nodes as a tuple $G = ( \pmb { \tau } , \mathbf { X } ^ { V } , \mathbf { m } , \mathbf { X } ^ { E } )$ in the graph space $\mathcal { G } _ { k }$ , defined as:

$$
\mathcal { G } _ { k } = \bigcup _ { N \geq 2 } ~ \underbrace { \mathcal { A } ^ { N } } _ { \mathrm { t y p e s } \tau } ~ \times ~ \underbrace { \mathbb { R } ^ { N \times d _ { v } ^ { ( k ) } } } _ { \mathrm { n o d e ~ a t t r i b u t e s } ~ \mathbf { X } ^ { V } } ~ \times ~ \underbrace { \{ 0 , 1 \} ^ { N \times T _ { k } } } _ { \mathrm { v a l i d } \mathrm { m a s k } \mathbf { m } } \times \underbrace { \mathbb { R } ^ { N \times N \times T _ { k } \times d _ { c } } } _ { \mathrm { e d g e f e a t u r e s } ~ \mathbf { X } ^ { E } }\tag{1}
$$

The disjoint union over $N$ allows the number of agents to vary across scenarios. Each node $v _ { i } \in V$ corresponds to an agent and has type $\tau _ { i } ~ \in ~ A$ . The node set is partitioned as $V ~ = ~ V _ { R } \cup V _ { H }$ where $V _ { R }$ contains robot nodes and $V _ { H }$ contains human nodes, and the type of each directed edge follows from its endpoints, $\rho _ { i j } = ( \tau _ { i } , \tau _ { j } )$ . The valid mask entry $m _ { i , t }$ records whether node i is observed at timestep $t ,$ and edge validity is given by $M _ { i j , t } ^ { E } = m _ { i , t } m _ { j , t } \nVdash [ i \ne j ]$ , so the graph is complete over every ordered pair of nodes observed at the same timestep. Each row $\mathbf { x } _ { i }$ of $\mathbf { X } ^ { V }$ contains the node attributes available for agent $i ,$ such as encoded occupancy grids or trajectory goals, and is zero-initialized otherwise. Each entry $\mathbf { e } _ { i j } \in \mathbb { R } ^ { T _ { k } \times d _ { e } } \mathrm { o f } \mathbf { X } ^ { E }$ is a temporal edge feature sequence containing the relative $S E ( 2 )$ pose of node j in the body frame of node i and its temporal deltas, with $d _ { e } = 8$ (Appendix B.1.2). Datasets may differ in sensor modality, sampling frequency, agent types, and available annotations and in Eq. 1 that variation is confined to $d _ { v } ^ { ( k ) }$ and $T _ { k }$ . To handle these diverse inputs, all raw sensor modalities are projected into a common hidden dimension prior to message passing. The relational message-passing core and latent heads are shared across datasets, while dataset-dependent input projections, temporal edge encoders, and edge decoders accommodate differences in feature dimensions and temporal windows (Appendix B.2.1).

![](images/2d4cbe9a493e57364ac8bc6f3c10a7921cafe474d75f87d965f5568480797b6f.jpg)  
Figure 1: STARS three stages: (A) construct spatiotemporal graphs from social navigation datasets, (B) utilize an unsupervised message-passing graph autoencoder to compress physical dynamics into latent representations, and (C) adapt frozen latents to downstream tasks using lightweight taskspecific prediction heads. See Sec. 3 for details.

The representation-learning problem maps an interaction scenario x, through its graph $G \in \mathcal G _ { k }$ , to compact per-node representations Z that support downstream social navigation tasks. For a downstream task ${ \mathcal { T } } _ { m } ,$ a subset of scenarios is paired with labels $\{ ( x _ { i } , y _ { i } ^ { ( m ) } ) \}$ , where $y _ { i } ^ { ( m ) } \in \mathcal { y } _ { m }$ represents the task-specific annotation, which in our evaluations include subjective human perceptions of robot behavior and discrete pedestrian action categories. A representation is considered better if it yields stronger held-out performance on $\mathcal { T } _ { m }$ under the same labeled-data budget and downstream adaptation protocol. Thus, the goal is to obtain compact and reusable representations that support strong downstream performance when labeled examples for each task are scarce.

## 3.2 STARS Overview

Motivated by a need to learn from diverse social navigation datasets containing spatiotemporal interactions while preserving the relational structure needed for downstream social tasks, STARS is designed to disentangle spatiotemporal dynamics via autoencoding to per-node latents ${ \mathbf z } = f _ { \phi } ( G )$ that capture reusable social context [11].

As shown in Figure 1, STARS is a three-stage framework for learning representations from spa tiotemporal data. First, STARS extracts features from social navigation datasets (examples shown in Figure 2) to construct unified spatiotemporal graphs. While STARS can apply generally to any spatiotemporal relational data, in this work we focus on the application area of social navigation. Explicitly modeling human and robot agents as nodes and their physical interactions as edges imposes a strong relational inductive bias. Second, a pretraining step compresses physical dynamics into a compact latent space via a self-supervised message-passing graph autoencoder. Self-supervised reconstruction reduces overfitting to scarce labels, enabling higher-capacity encoders compared to a supervised learning baseline (Appendix E.4.1). To evaluate the utility of these representations for social tasks, we subsequently apply a lightweight, task-specific linear prediction head to the frozen latents. We instantiate STARS using both homogeneous and heterogeneous graph formulations to study how explicit entity and relation types affect STARS’ learned representations.

## 3.3 Spatiotemporal Relational Graph Construction

Our construction makes two design choices. First, our graphs are edge-feature dense, capturing temporal relationships primarily through edge rather than node attributes. Second, rather than using distance-based thresholds, we construct complete graphs to capture interactions between all observed agents. An example graph construction is shown in Appendix Fig. 3, and full input projection specifications are given in Appendix B.1.

The homogeneous graph represents nodes as generic “agents” and all edges as “interactions.” Robot and human nodes retain distinct input attributes, and entity type enters only through a twoclass node embedding; the graph does not include edge-type embeddings. Every dataset is resampled to a common temporal window, so one input projection and one edge encoder serve both datasets, and minibatches may mix sources.

The heterogeneous graph explicitly distinguishes nodes by entity type. Edges are also typed based on the connected node types. Robot nodes may include robot-specific attributes, such as encoded occupancy grids or trajectory goals when available, while human nodes are associated with a learnable semantic embedding representing the entity type, concatenated with a mean-pooled summary of their incident edge features. The edge set captures typed relations such as robot-human, humanrobot, and human-human interactions, with relation types provided separately through edge-type embeddings.

## 3.4 Unsupervised Graph Autoencoder

The core of STARS is a graph encoder implemented as a Message-Passing Graph Neural Network (MPGNN) that maps the spatiotemporal graph into node-level latent distributions.

Message Passing: The graph encoder performs message passing over the complete interaction graph. In STARS’ heterogeneous graph formulation, message passing uses a HEAT-style update [44]. Let $\mathbf { h } _ { j } ^ { ( \ell ) }$ and $\mathbf { h } _ { i } ^ { ( \bar { \ell } ) }$ denote the source and destination node embeddings at layer $\ell ,$ and $\mathbf { t } _ { i j }$ denote a learned embedding for the specific edge type (e.g., human-robot vs. human-human). Let $\tilde { \mathbf { e } } _ { i j } ^ { ( 0 ) } = \mathrm { E n c } _ { \mathrm { e d g e } } ( \mathbf { e } _ { i j } )$ denote the initial encoded temporal edge representation. We construct a joint interaction context:

$$
\mathbf { u } _ { i j } ^ { ( \ell ) } = [ \mathbf { h } _ { j } ^ { ( \ell ) } | | \mathbf { h } _ { i } ^ { ( \ell ) } | | \tilde { \mathbf { e } } _ { i j } ^ { ( \ell ) } | | \mathbf { t } _ { i j } ]\tag{2}
$$

The representation $\mathbf { u } _ { i j } ^ { ( \ell ) }$ includes the full context of the interaction. During message passing, the network must determine both the content of the information being transmitted and the relative importance of that information. Therefore, we use $\mathbf { u } _ { i j } ^ { ( \ell ) }$ to independently parameterize both the raw attention score $s _ { i j } ^ { ( \ell ) }$ and the message content $\mathbf { m } _ { i j } ^ { ( \ell ) }$ via two learnable functions:

$$
s _ { i j } ^ { ( \ell ) } = f _ { \mathrm { a t t n } } ( \mathbf { u } _ { i j } ^ { ( \ell ) } ) , \qquad \mathbf { m } _ { i j } ^ { ( \ell ) } = f _ { \mathrm { m s g } } ( \mathbf { u } _ { i j } ^ { ( \ell ) } )\tag{3}
$$

Attention weights are normalized via a destination-wise softmax $\alpha _ { i j } ^ { ( \ell ) }$ , and the aggregated message is used to update the node states via a residual LayerNorm step. The heterogeneous formulation also updates $\tilde { \mathbf { e } } _ { i j } ^ { ( \bar { \ell } ) }$ between message-passing layers using the corresponding edge message; full update equations are provided in Appendix B.2.3. Typed-edge attention enables the model to learn which distant interactions matter without needing separate message-passing modules per relation. The homogeneous graph formulation omits edge-type embeddings $\mathbf { t } _ { i j }$ from $\mathbf { u } _ { i j } ^ { ( \ell ) }$ , using only the source node embedding, destination node embedding, and encoded temporal edge features $\tilde { \mathbf { e } } _ { i j }$ in the update.

Latent Encoding: After L message-passing layers, shared linear projection heads parameterize a diagonal Gaussian latent distribution for each node:

$$
q _ { \phi } ( \mathbf { z } _ { i } \mid G ) = { \mathcal { N } } { \big ( } \mu _ { i } , \ \mathrm { d i a g } ( \sigma _ { i } ^ { 2 } ) { \big ) } , \qquad \mu _ { i } = f _ { \mu } { \big ( } \mathbf { h } _ { i } ^ { ( L ) } { \big ) } , \quad \log \sigma _ { i } ^ { 2 } = f _ { \sigma } { \big ( } \mathbf { h } _ { i } ^ { ( L ) } { \big ) }\tag{4}
$$

Latent samples are drawn via the reparameterization trick, $\mathbf { z } _ { i } ~ = ~ \pmb { \mu } _ { i } + \pmb { \sigma } _ { i } \odot \pmb { \epsilon } .$ Let ${ \textbf { Z } } =$ $[ \mathbf { z } _ { 1 } , \hdots , \mathbf { z } _ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times d _ { z } }$ denote the stacked node latents, where $d _ { z }$ is the latent dimension, so that $\begin{array} { r } { q _ { \phi } ( \mathbf { Z } \mid G ) = \prod _ { i } q _ { \phi } ( \mathbf { z } _ { i } \mid G ) } \end{array}$ . This per-node latent formulation preserves the asymmetry of pairwise interactions, as edge reconstruction conditions on both endpoint latent nodes rather than a single pooled graph representation.

Decoding & Objective: Unlike autoregressive models, we employ an unsupervised autoencoding objective over the entire fixed-duration interaction scenario. Specifically, for temporal edge reconstruction, the decoder employs an MLP that takes the concatenated source and destination latents together with the edge-type embedding, $\left[ \mathbf { z } _ { i } \| \mathbf { z } _ { j } \| \mathbf { t } _ { i j } \right]$ , as input.

For node-level reconstruction, we restrict the objective to nodes containing active input attributes. Reconstructing static, zero-initialized features, such as placeholders for entities without observed attributes, would introduce a strong bias toward trivial constant targets, diluting the gradient signal. Let $V _ { \mathrm { f e a t } } \subseteq V$ be the subset of nodes containing active attributes, determined by the dataset and node type rather than by testing whether $\mathbf { x } _ { i }$ is zero (Appendix B.1). In practice, $V _ { \mathrm { f e a t } }$ includes only nodes with observed, non-placeholder attributes, which in our datasets is the SEAN-T robot node alone. For nodes without active input attributes, we omit node reconstruction. Table 3 in Appendix B.1 summarizes the node features and reconstruction targets for each architecture and dataset. Crucially, the absence of human node reconstruction does not exclude human nodes from the learned representations or lead to a training mismatch, because the temporal edge decoder must reconstruct $\hat { \mathbf { e } } _ { i j }$ from the concatenated endpoint latents alone. The model is forced to encode rich spatiotemporal social semantics and joint context into the human node latents to successfully reconstruct these edges.

The model is trained by minimizing the following VAE objective:

$$
\mathcal { L } _ { \mathrm { p r e } } = \lambda _ { v } \sum _ { v _ { i } \in V _ { \mathrm { f a t } } } \mathrm { M S E } ( \mathbf { x } _ { i } , \hat { \mathbf { x } } _ { i } ) \ + \ \sum _ { \rho } \lambda _ { \rho } \sum _ { \rho _ { i j } = \rho } \mathrm { M S E } ( \mathbf { e } _ { i j } , \hat { \mathbf { e } } _ { i j } ) \ + \ \beta \sum _ { v _ { i } \in V } D _ { \mathrm { K L } } \left( q _ { \phi } ( \mathbf { z } _ { i } \mid G ) \parallel \mathcal { N } ( 0 , I ) \right)\tag{5}
$$

where $\lambda _ { v }$ and $\lambda _ { \rho }$ denote weighting factors for node and edge reconstruction, the latter indexed by relation type, allowing the network to learn across imbalanced datasets, for example, where humanhuman interactions are much more common than human-robot interactions.

## 3.5 Task-Agnostic Linear Probing

To evaluate the quality of our learned latent representations and adapt them to downstream tasks with limited labeled data, we apply linear probes [45].

Because STARS learns compressed node-level latent representations $\mathbf { z } _ { i } ,$ it provides flexibility in selecting and aggregating these node latents to represent different structural components relevant to a given downstream task. After freezing the pretrained encoder weights $\phi ,$ STARS selects and aggregates the necessary latents based on a given downstream task, and because N varies across scenarios this aggregation returns a fixed-dimensional vector that is invariant to permutations of nodes of the same type:

$$
\begin{array} { r } { \mathbf { z } _ { \mathrm { p r o b e } } ^ { \mathrm { S E A N } } = [ \mathbf { z } _ { r }  \overline { { \mathbf { z } } } _ { h } ] , \qquad \overline { { \mathbf { z } } } _ { h } = \frac { 1 } {  V _ { H }  } \displaystyle \sum _ { v _ { i } \in V _ { H } } \mathbf { z } _ { i } , \qquad \mathbf { z } _ { \mathrm { p r o b e } } ^ { \mathrm { S N S } } ( i ) = [ \mathbf { z } _ { r }  \mathbf { z } _ { i }  , \quad v _ { i } \in V _ { H } } \end{array}\tag{6}
$$

the first for the graph-level SEAN-T perception tasks and the second for the node-level SNS action task, both of dimension $2 d _ { z }$ . A linear probe with weights W is then optimized using supervised pairs $( G , y )$ , where $y$ is the task label, by minimizing $\mathcal { L } _ { \mathrm { t a s k } } = \mathrm { L o s s F n } ( y , W \mathbf { z } _ { \mathrm { p r o b e } } + b )$ . This lightweight adaptation step keeps the representation fixed, so downstream performance reflects the information encoded in $\mathbf { z } _ { \mathrm { p r o b e } }$ rather than task-specific representation learning.

## 4 Experiments

We evaluate the effectiveness of STARS to learn social semantics from physical dynamics. Specifically, we demonstrate that STARS achieves data-efficient downstream performance, and through analysis of the latent space topology, we show that the model can distinguish social behaviors, such as action intents, without optimizing the encoder with a supervised classification objective during pretraining. Our research questions are:

RQ1: Downstream Performance. How well does STARS work on downstream social navigation tasks compared to unstructured baselines when adapting pretrained representations?

RQ2: Data Efficiency. For a given downstream task, what is the impact of labeled data availability, from 1% of the labeled training set (60 samples) to 100%, when using pretrained STARS representations?

RQ3: Latent Space Semantics. Do STARS’ self-supervised representations naturally encode social semantics (e.g., action intents) without a supervised action classification objective?

![](images/357626476892c41e1778acb22104561b768872d42bd154b2fa1abc1d11dbf113.jpg)  
Figure 2: (A) Dataset Overview. We utilize two human-robot interaction datasets: SEAN-T [21] for evaluating subjective human ratings of robot navigation and SNS [6] for downstream relational action classification (e.g., avoid, follow). (B) Latent Space Analysis (RQ3). 2D Projections of the SNS Human Node Action Latent Space for the avoid and follow actions. We visualize the 128- dimensional representations of nodes extracted from the STARS-trained homogeneous GNN-VAE encoder using linear PCA and non-linear t-SNE. Each point represents a node embedding, colorcoded by action label. See Sec. 4.2 for details.

Datasets and Tasks We evaluate STARS across two distinct datasets to assess unsupervised pretraining and downstream task adaptation. Our pretraining pool consists of multi-agent spatiotemporal data from SEAN Together (SEAN-T) [21], which provides 8-second immersive virtualreality observations of robot navigation, and SocialNav-SUB (SNS) [6], a structured social reasoning benchmark containing pedestrian-robot interactions. Examples of the interaction scenarios are shown in Figure 2 and further details are provided in the Appendix A. To evaluate whether our autoencoder successfully captures reusable social semantics, we freeze the pretrained represen tations and train lightweight linear probes on two distinct downstream tasks: 1) Perception Prediction (SEAN-T): Predicting binary subjective human ratings of a robot’s behavior across three distinct metrics: perceived competence, clarity of intention, and surprise [21]. This task evaluates the model’s ability to extract subjective social impressions directly from physical motion. 2) Action Classification (SNS): Predicting discrete, categorical relational actions taken by pedestrians in dense crowd scenes with up to 30 people (e.g., avoiding,following, yielding) [6]. This task tests the model’s capacity to explicitly classify interaction intents in complex multi-agent environments.

Training and Evaluation All reported results are averaged across 10 random seeds and use early stopping [46] to limit overfitting. Model-specific hyperparameter selection, capacity, and complete training protocols are detailed in Appendix C.

## 4.1 RQ1 & RQ2: Downstream Performance and Data Efficiency

Experiment: To evaluate downstream performance (RQ1) and data efficiency (RQ2), we pretrain STARS on the combined training splits of the datasets using the self-supervised reconstruction objective. After freezing the encoder, we extract the relevant elements of the latent graph representation (details in the Appendix C) and train the linear probe using progressively larger fractions of the labels from the training split. We compare our approach against several unstructured baselines: a fully connected, feed-forward Multi-Layer Perceptron (MLP), an Autoencoder to ablate the graph structure while retaining data compression, and a Random Forest to provide a non-deep-learning baseline. Note that the Random Forest is omitted for SNS due to the dataset’s variable-sized, high dimensional features. The Autoencoder receives the full temporal edge representation and follows the same pooled pretraining and frozen adaptation protocol across labeled-data fractions. We report F -Scores for competence, surprise, and intention for SEAN-T and macro F -Score over avoid, follow, and not consider for SNS, with per-class results reported in the Appendix E.1.

Results: Table 1 shows that the heterogeneous (STARS-He) and homogeneous (STARS-Ho) variants are competitive with unstructured baselines on SEAN-T and achieve stronger performance on most evaluated settings. While both exhibit competitive results, STARS-Ho yields the highest overall scores in most SEAN-T categories. STARS-He achieves the highest SNS macro $F _ { 1 }$ across all labeled-data fractions, with per-class results reported in Appendix E.1. On SNS, unstructured baselines fail to extract relational signal, plateauing near a macro $F _ { 1 }$ -Score of 0.3, regardless of how much labeled data is provided. In contrast, STARS-He rapidly acquires task-relevant information, achieving 0.362 with just 1% of the data and achieves 0.503 at 100%, indicating that the heterogeneous formulation is particularly effective for this relational action classification task. We addition ally compare against the same heterogeneous graph architecture trained end-to-end on SNS labels. STARS-He achieves a macro $F _ { 1 }$ -Score of 0.503 at 100% labeled data, compared with 0.445 for the supervised model; full results are reported in Appendix E.4.1. To assess cross-dataset transfer, we pretrain on one dataset and probe on the other. At 100% labeled data, STARS-He pretrained only on SEAN-T achieves 0.497 macro $F _ { 1 }$ on SNS compared with 0.503 under joint pretraining, while SNS-only pretraining achieves $0 . 8 0 1 / 0 . 7 1 0 / 0 . 7 2 9$ on the SEAN-T competence/surprise/intention tasks compared with 0.794/0.709/0.721 under joint pretraining. STARS-Ho also transfers across datasets, although joint pretraining provides larger gains; full results across labeled-data fractions are reported in Appendix E.3. Finally, the benefit of STARS’s relational structure is most pronounced on SNS, where STARS-He consistently outperforms the temporal Autoencoder across all labeled-data fractions. On SEAN-T, the Autoencoder remains competitive in several settings, suggesting that explicit relational structure provides a greater benefit for pedestrian action classification than for the perception-prediction tasks.

Table 1: Downstream Performance (RQ1) and Data Efficiency (RQ2) on SEAN-T and SNS. Results showcase $F _ { 1 }$ -Scores $( \mu \pm \sigma )$ across progressively scaled fractions of labeled data. We compare unstructured baseline methods against our proposed STARS variants (STARS-He: Heterogeneous, STARS-Ho: Homogeneous). To rigorously evaluate both representation quality and data efficiency, STARS utilizes a frozen pretrained encoder paired with a lightweight linear probe.
<table><tr><td></td><td>Method</td><td>1%</td><td>5%</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td rowspan="4">Commnce SEEANT</td><td>MLP Autoencoder</td><td> $0 . 4 5 0 \pm 0 . 2 2$   $\mathbf { 0 . 6 6 3 \pm 0 . 0 8 }$ </td><td> $0 . 5 4 6 \pm 0 . 1 9$   $0 . 6 7 8 \pm 0 . 0 5$ </td><td> $0 . 6 6 4 \pm 0 . 1 5$   $0 . 7 0 6 \pm 0 . 0 5$ </td><td> $0 . 7 6 1 \pm 0 . 0 3$   $0 . 7 3 4 \pm 0 . 0 2$ </td><td> $0 . 7 7 8 \pm 0 . 0 2$   $0 . 7 4 7 \pm 0 . 0 2$ </td><td> $0 . 7 9 7 \pm 0 . 0 2$   $0 . 7 4 2 \pm 0 . 0 2$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random Forest</td><td> $0 . 5 5 7 \pm 0 . 2 0$ </td><td> $0 . 6 4 1 \pm 0 . 1 2$ </td><td> $0 . 6 8 0 \pm 0 . 1 0$ </td><td> $0 . 7 0 4 \pm 0 . 0 9$ </td><td> $0 . 7 1 5 \pm 0 . 0 9$ </td><td> $0 . 7 1 8 \pm 0 . 0 8$ </td></tr><tr><td>STARS-He (Ours) STARS-Ho (Ours)</td><td> $0 . 6 3 5 \pm 0 . 1 0$   $0 . 6 1 2 \pm 0 . 1 4$ </td><td> $0 . 6 7 3 \pm 0 . 0 7$ </td><td> $0 . 7 2 6 \pm 0 . 0 4$ </td><td> $0 . 7 6 2 \pm 0 . 0 3$ </td><td> $0 . 7 8 1 \pm 0 . 0 3$ </td><td> $0 . 7 9 4 \pm 0 . 0 2$ </td></tr><tr><td rowspan="4">SEEEANT Suprse</td><td>MLP</td><td></td><td> ${ \bf 0 . 7 0 2 \pm 0 . 0 8 }$ </td><td> ${ \bf 0 . 7 3 8 \pm 0 . 0 5 }$ </td><td> ${ \bf 0 . 7 6 5 \pm 0 . 0 4 }$ </td><td> $\mathbf { 0 . 7 8 7 \pm 0 . 0 3 }$ </td><td> ${ \bf 0 . 7 9 9 \pm 0 . 0 2 }$ </td></tr><tr><td>Autoencoder</td><td> $0 . 2 1 7 \pm 0 . 2 2$ </td><td> $0 . 2 5 0 \pm 0 . 1 4$ </td><td> $0 . 4 2 3 \pm 0 . 2 7$ </td><td> $0 . 4 5 0 \pm 0 . 2 1$ </td><td> $0 . 6 3 0 \pm 0 . 2 3$ </td><td> $0 . 7 2 0 \pm 0 . 0 9$ </td></tr><tr><td>Random Forest</td><td> ${ \bf 0 . 5 4 8 \pm 0 . 1 0 }$ </td><td> ${ \bf 0 . 5 9 7 \pm 0 . 1 2 }$ </td><td> $\mathbf { 0 . 6 6 0 \pm 0 . 0 7 }$ </td><td> $\mathbf { 0 . 6 7 4 \pm 0 . 0 9 }$ </td><td> $0 . 7 1 2 \pm 0 . 1 0$ </td><td> $0 . 7 3 9 \pm 0 . 0 6$ </td></tr><tr><td>STARS-He (Ours)</td><td> $0 . 2 9 1 \pm 0 . 2 4$ </td><td> $0 . 3 4 1 \pm 0 . 1 9$ </td><td> $0 . 3 7 3 \pm 0 . 2 0$ </td><td> $0 . 4 4 4 \pm 0 . 1 6$ </td><td> $0 . 4 7 0 \pm 0 . 1 5$ </td><td> $0 . 4 9 6 \pm 0 . 1 5$ </td></tr><tr><td rowspan="4">SEEEANT Intnion</td><td>STARS-Ho (Ours)</td><td> $0 . 4 5 5 \pm 0 . 1 7$  0.440 ± 0.23</td><td> $\begin{array} { c } { 0 . 4 8 1 \pm 0 . 1 7 } \\ { 0 . 5 4 1 \pm 0 . 2 1 } \end{array}$ </td><td> $_ { 0 . 5 4 8 \pm 0 . 2 0 } ^ { 0 . 5 2 6 \pm 0 . 1 5 }$ </td><td> $_ { 0 . 6 1 0 \pm 0 . 1 6 } ^ { 0 . 6 0 3 \pm 0 . 1 2 }$ </td><td> $\begin{array} { c } { 0 . 6 5 5 \pm 0 . 0 8 } \\ { \mathbf { 0 . 7 1 3 \pm 0 . 0 6 } } \end{array}$ </td><td> $0 . 7 0 9 \pm 0 . 0 4$   ${ \bf 0 . 7 4 1 \pm 0 . 0 3 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MLP Autoencoder</td><td> $0 . 4 7 9 \pm 0 . 2 0$ </td><td> $0 . 5 3 3 \pm 0 . 1 3$ </td><td></td><td> $0 . 6 5 9 \pm 0 . 0 3$ </td><td> $0 . 6 9 3 \pm 0 . 0 3$ </td><td> $0 . 7 0 3 \pm 0 . 0 1$ </td></tr><tr><td>Random Forest</td><td> $0 . 5 6 9 \pm 0 . 0 4$   $0 . 5 9 2 \pm 0 . 1 8$ </td><td> $0 . 6 2 2 \pm 0 . 0 3$   $0 . 6 6 5 \pm 0 . 1 0$ </td><td> $\begin{array} { c } { 0 . 6 0 6 \pm 0 . 0 6 } \\ { 0 . 6 5 9 \pm 0 . 0 3 } \\ { 0 . 6 8 5 \pm 0 . 0 9 } \end{array}$ </td><td> $0 . 6 6 5 \pm 0 . 0 5$ </td><td> $0 . 6 6 1 \pm 0 . 0 3$ </td><td> $0 . 6 6 4 \pm 0 . 0 2$ </td></tr><tr><td rowspan="4"></td><td>STARS-He (Ours)</td><td></td><td></td><td></td><td> $0 . 6 9 2 \pm 0 . 0 9$ </td><td> $0 . 7 0 0 \pm 0 . 0 8$ </td><td> $0 . 6 9 8 \pm 0 . 0 9$ </td></tr><tr><td>STARS-Ho (Ours)</td><td> $0 . 5 7 6 \pm 0 . 0 8$ </td><td> $\begin{array} { r } { 0 . 6 1 9 \pm 0 . 0 5 } \\ { \mathbf { 0 . 6 7 1 } \pm \mathbf { 0 . 0 7 } } \end{array}$ </td><td> $\begin{array} { c } { 0 . 6 7 1 \pm 0 . 0 4 } \\ { \mathbf { 0 . 7 0 1 \pm 0 . 0 6 } } \end{array}$ </td><td> $\begin{array} { c } { 0 . 6 9 4 \pm 0 . 0 3 } \\ { \mathbf { 0 . 7 1 5 \pm 0 . 0 4 } } \end{array}$ </td><td> $\begin{array} { c } { 0 . 7 0 5 \pm 0 . 0 3 } \\ { \mathbf { 0 . 7 2 3 \pm 0 . 0 3 } } \end{array}$ </td><td> $0 . 7 2 1 \pm 0 . 0 3$ </td></tr><tr><td></td><td> ${ \bf 0 . 6 0 3 \pm 0 . 1 0 }$ </td><td></td><td></td><td></td><td></td><td> ${ \bf 0 . 7 3 4 \pm 0 . 0 2 }$ </td></tr><tr><td>MLP</td><td> $0 . 2 9 4 \pm 0 . 0 0$ </td><td> $\begin{array} { l } { 0 . 2 9 7 \pm 0 . 0 1 } \\ { 0 . 3 0 9 \pm 0 . 0 2 } \end{array}$ </td><td></td><td></td><td></td><td> $0 . 2 9 3 \pm 0 . 0 0$ </td></tr><tr><td rowspan="4">MM01 SNS</td><td>Autoencoder</td><td> $0 . 3 0 8 \pm 0 . 0 2$ </td><td></td><td> $\begin{array} { c } { 0 . 3 0 0 \pm 0 . 0 1 } \\ { 0 . 3 0 7 \pm 0 . 0 2 } \end{array}$ </td><td> $\begin{array} { l } { 0 . 3 0 6 \pm 0 . 0 2 } \\ { 0 . 3 1 0 \pm 0 . 0 2 } \end{array}$ </td><td> $\begin{array} { l } { 0 . 2 9 5 \pm 0 . 0 1 } \\ { 0 . 3 0 9 \pm 0 . 0 2 } \end{array}$ </td><td> $0 . 3 3 2 \pm 0 . 0 1$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>STARS-He (Ours)</td><td> $\mathbf { 0 . 3 6 2 \pm 0 . 0 6 }$ </td><td></td><td> $\mathbf { 0 . 4 7 0 \pm 0 . 0 4 }$ </td><td> ${ \bf 0 . 4 9 3 \pm 0 . 0 3 }$ </td><td> $\mathbf { 0 . 5 0 1 \pm 0 . 0 3 }$ </td><td> ${ \bf 0 . 5 0 3 \pm 0 . 0 2 }$ </td></tr><tr><td>STARS-Ho (Ours)</td><td> $0 . 3 4 4 \pm 0 . 0 5$ </td><td> $\begin{array} { c } { { \mathbf { 0 . 4 4 7 \pm 0 . 0 5 } } } \\ { { 0 . 4 3 1 \pm 0 . 0 5 } } \end{array}$ </td><td> $0 . 4 5 3 \pm 0 . 0 4$ </td><td> $0 . 4 6 8 \pm 0 . 0 3$ </td><td>0.471 ± 0.02</td><td> $0 . 4 7 5 \pm 0 . 0 2$ </td></tr></table>

## 4.2 RQ3: Latent Space Semantics

Experiment: To qualitatively evaluate whether the learned representations capture high-level social semantics without optimizing the encoder with a supervised classification objective (RQ3), we analyze the topology of the latent spaces. Using the pretrained latent representation from the homogeneous graph encoder reported in Table 1, we extract frozen representations from the 128- dimensional node-level embeddings for SNS. To focus on semantic navigation intents, we conduct our experiments on the dominant action labels in the SocialNav-SUB (SNS) dataset, avoid and follow. We apply both Principal Component Analysis (PCA) for linear projection and t-Distributed

Stochastic Neighbor Embedding (t-SNE) for non-linear dimensionality reduction to map the high dimensional representations to a 2D space, color-coded by downstream class labels.

Results: Latent space projections shown in Figure 2 reveal structural self-organization corresponding to downstream human navigation semantics. The linear PCA projection (left) shows smooth variation across the latent space, while the non-linear t-SNE projection (right) reveals visually distinct groupings associated with action classes. Together, these projections provide qualitative evidence that the learned representations contain action relevant structure, despite the encoder not being optimized with a supervised action classification objective.

## 5 Limitations and Future Work

While STARS generalizes well, its autoencoder and message passing inherently smooth highfrequency pairwise geometries. Therefore, these latents excel at node-level tasks like intent prediction, but are less effective on edge-centric tasks (e.g., group detection) that require exact relational geometry. Additionally, standardizing diverse datasets into unified graphs can discard datasetspecific nuances. Future work can explore more flexible graph construction pipelines and the use of intermediate, pre-message-passing temporal edge embeddings to bypass the autoencoder bottleneck, preserving the precise geometric information necessary for fine-grained relational tasks. Our evaluation assumes fixed-duration interaction windows and processed agent tracks, and is limited to curated social navigation datasets. A promising direction for future work is to evaluate STARS under online segmentation, noisier perception, and more diverse real-world settings.

## 6 Conclusion

In this work, we introduced STARS for self-supervised representation learning in social navigation by bridging continuous physical dynamics and high-level social reasoning. By modeling scenarios as spatiotemporal graphs and using a self-supervised graph autoencoder, STARS injects a strong relational inductive bias. Evaluations show that our learned representations are highly reusable and data-efficient, and can encode complex social semantics without directly optimizing the encoder for downstream tasks. Ultimately, our results provide evidence that structural priors can help offset the scarcity of grounded HRI data, offering a promising direction toward reusable and data-efficient representations for social navigation.

## Acknowledgments

A portion of work has taken place in the Learning Agents Research Group (LARG) at UT Austin. LARG research is supported in part by NSF (FAIN-2019844, NRT-2125858, OIA-2535195), ONR (N00014-24-1-2550), ARO (W911NF-17-2-0181, W911NF-23-2-0004, W911NF-25-1-0065), Lockheed Martin, Lyda Hill, and Good Systems. Peter Stone serves as the Chief Scientist of Sony CTC and receives financial compensation for that role. The terms of this arrangement have been reviewed and approved by the University of Texas at Austin in accordance with its policy on objectivity in research. Another portion of this work has taken place at the Autonomous Mobile Robotics Laboratory (AMRL) at the Artificial Intelligence Laboratory, The University of Texas at Austin. AMRL research is supported in part by the National Science Foundation (OIA-2535195, CAREER-2046955, OIA-2219236, DGE-2125858, CCF-2319471). The authors also acknowledge the Texas Advanced Computing Center (TACC) at The University of Texas at Austin for providing high-performance computing resources that have contributed to the research results reported within this paper. Any opinions, findings, and conclusions expressed in this material are those of the authors and do not necessarily reflect the views of the sponsors.

## References

[1] C. Mavrogiannis, F. Baldini, A. Wang, D. Zhao, P. Trautman, A. Steinfeld, and J. Oh. Core challenges of social robot navigation: A survey. ACM Transactions on Human-Robot Interaction, 12(3):1–39, 2023.

[2] A. Francis, C. Perez-d’Arpino, C. Li, F. Xia, A. Alahi, R. Alami, A. Bera, A. Biswas, J. Biswas,´ R. Chandra, et al. Principles and guidelines for evaluating social robot navigation algorithms. ACM Transactions on Human-Robot Interaction, 14(2):1–65, 2025.

[3] X. Zhou, J. Liu, A. Yerukola, H. Kim, and M. Sap. Social world models. arXiv preprint arXiv:2509.00559, 2025.

[4] M. Lisondra, B. Benhabib, and G. Nejat. Embodied ai with foundation models for mobile service robots: A systematic review. Robotics, 15(3):55, 2026.

[5] S. K. Ramakrishnan, E. Wijmans, P. Kraehenbuehl, and V. Koltun. Does spatial cognition emerge in frontier models? arXiv preprint arXiv:2410.06468, 2024.

[6] M. J. Munje, C. Tang, S. Liu, Z. Hu, Y. Zhu, J. Cui, G. Warnell, J. Biswas, and P. Stone. Socialnav-sub: Benchmarking vlms for scene understanding in social robot navigation. In Proceedings of the Conference on Robot Learning (CoRL), 2025. Project page: https:// larg.github.io/socialnav-sub.

[7] Q. Gao, X. Pi, K. Liu, J. Chen, R. Yang, X. Huang, X. Fang, L. Sun, G. Kishore, B. Ai, et al. Do vision-language models have internal world models? towards an atomic evaluation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 26170–26195, 2025.

[8] M. M. de Graaf, S. Ben Allouch, and J. A. Van Dijk. What makes robots social?: A user’s perspective on characteristics for social human-robot interaction. In International Conference on Social Robotics, pages 184–193. Springer, 2015.

[9] P. W. Battaglia, J. B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi, M. Malinowski, A. Tacchetti, D. Raposo, A. Santoro, R. Faulkner, et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018.

[10] J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl. Message passing neural networks. In Machine learning meets quantum physics, pages 199–214. Springer, 2020.

[11] N. Tsoi, A. Xiang, P. Yu, S. S. Sohn, G. Schwartz, S. Ramesh, M. Hussein, A. W. Gupta, M. Kapadia, and M. Vazquez. Sean 2.0: Formalizing and generating social situations for robot´ navigation. IEEE Robotics and Automation Letters, 7(4):11047–11054, 2022.

[12] S. Pellegrini, A. Ess, K. Schindler, and L. Van Gool. You’ll never walk alone: Modeling social behavior for multi-target tracking. In 2009 IEEE 12th international conference on computer vision, pages 261–268. IEEE, 2009.

[13] A. Lerner, Y. Chrysanthou, and D. Lischinski. Crowds by example. In Computer graphics forum. Wiley Online Library, 2007.

[14] A. Robicquet, A. Sadeghian, A. Alahi, and S. Savarese. Learning social etiquette: Human trajectory understanding in crowded scenes. In European conference on computer vision, pages 549–565. Springer, 2016.

[15] R. Mart´ın-Mart´ın, H. Rezatofighi, A. Shenoi, M. Patel, J. Gwak, N. Dass, A. Federman, P. Goebel, and S. Savarese. Jrdb: A dataset and benchmark for visual perception for navigation in human environments. arXiv preprint arXiv:1910.11792, 2019.

[16] H. Karnan, A. Nair, X. Xiao, G. Warnell, S. Pirk, A. Toshev, J. Hart, J. Biswas, and P. Stone. Socially compliant navigation dataset (scand): A large-scale dataset of demonstrations for social navigation. IEEE Robotics and Automation Letters, 7(4):11807–11814, 2022.

[17] A. Mohamed, K. Qian, M. Elhoseiny, and C. Claudel. Social-stgcnn: A social spatio-temporal graph convolutional neural network for human trajectory prediction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14424–14432, 2020.

[18] V. Kosaraju, A. Sadeghian, R. Mart´ın-Mart´ın, I. Reid, H. Rezatofighi, and S. Savarese. Socialbigat: Multimodal trajectory forecasting using bicycle-gan and graph attention networks. Advances in neural information processing systems, 32, 2019.

[19] X. Li, Y. Liang, Z. Yang, and J. Li. Pedestrian trajectory prediction based on dual social graph attention network. Applied Sciences, 15(8):4285, 2025.

[20] R. Li, T. Qiao, S. Katsigiannis, Z. Zhu, and H. P. Shum. Unified spatial–temporal edgeenhanced graph networks for pedestrian trajectory prediction. IEEE Transactions on Circuits and Systemsfor Video Technology, 35(7):7047–7060, 2025.

[21] Q. Zhang, N. Tsoi, M. Nagib, B. Choi, J. Tan, H.-T. L. Chiang, and M. Vazquez. Pre-´ dicting human impressions of robot performance during navigation tasks. arXiv preprint arXiv:2310.11590, 2023.

[22] S. Thompson, K. Candon, and M. Vazquez. The social context of human–robot interactions.´ Annual Review ofControl, Robotics, and Autonomous Systems, 9, 2025.

[23] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

[24] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

[25] J. Bruce, M. D. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.

[26] J. Parker-Holder, P. Ball, J. Bruce, V. Dasagi, K. Holsheimer, C. Kaplanis, A. Moufarek, G. Scully, J. Shar, J. Shi, S. Spencer, J. Yung, M. Dennis, S. Kenjeyev, S. Long, V. Mnih, H. Chan, M. Gazeau, B. Li, F. Pardo, L. Wang, L. Zhang, F. Besse, T. Harley, A. Mitenkova, J. Wang, J. Clune, D. Hassabis, R. Hadsell, A. Bolton, S. Singh, and T. Rocktaschel. Genie 2: A¨ large-scale foundation world model. URL: https://deepmind. google/discover/blog/genie-2-alarge-scale-foundation-world-model, 2024. URL https://deepmind.google/discover/ blog/genie-2-a-large-scale-foundation-world-model/.

[27] A. O’Neill, A. Rehman, A. Maddukuri, A. Gupta, A. Padalkar, A. Lee, A. Pooley, A. Gupta, A. Mandlekar, A. Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

[28] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023.

[29] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[30] J. A. Duncan, F. Alambeigi, and M. W. Pryor. A survey of multimodal perception methods for human–robot interaction in social environments. ACM Transactions on Human-Robot Interaction, 13(4):1–50, 2024.

[31] J. Ondras, A. Anwar, T. Wu, F. Bu, M. Jung, J. J. Ortiz, and T. Bhattacharjee. Human-robot commensality: Bite timing prediction for robot-assisted feeding in groups. In 6th Annual Conference on Robot Learning, 2022.

[32] F. Bu and W. Ju. Ssup-hri: Social signaling in urban public human-robot interaction dataset. In International Conference on Social Robotics, pages 479–487. Springer, 2024.

[33] J. S. Heinisch, J. Kirchhoff, P. Busch, J. Wendt, O. von Stryk, and K. David. Physiological data for affective computing in hri with anthropomorphic service robots: the affect-hri data set. Scientific Data, 11(1):333, 2024.

[34] S. Heo, Y. Cho, J. Park, S. Cho, Z. Tsoy, H. Lim, and Y. Cha. Diverse humanoid robot pose estimation from images using only sparse datasets. Applied Sciences, 14(19):9042, 2024.

[35] S. Thompson, A. Gupta, A. W. Gupta, A. Chen, and M. Vazquez. Conversational group de- ´ tection with graph neural networks. In Proceedings of the 2021 International Conference on Multimodal Interaction, pages 248–252, 2021.

[36] F. Yang, Y. Gao, R. Ma, S. Zojaji, G. Castellano, and C. Peters. A dataset of human and robot approach behaviors into small free-standing conversational groups. PloS one, 16(2):e0247364, 2021.

[37] S. Shrestha, Y. Zha, S. Banagiri, G. Gao, Y. Aloimonos, and C. Fermuller. Natsgd: A dataset with speech, gestures, and demonstrations for robot learning in natural human-robot interaction. arXiv preprint arXiv:2403.02274, 2024.

[38] F. Ferreira, L. Shao, T. Asfour, and J. Bohg. Learning visual dynamics models of rigid objects using relational inductive biases. arXiv preprint arXiv:1909.03749, 2019.

[39] A. Sanchez-Gonzalez, J. Godwin, T. Pfaff, R. Ying, J. Leskovec, and P. Battaglia. Learning to simulate complex physics with graph networks. In International conference on machine learning, pages 8459–8468. PMLR, 2020.

[40] M. Malik and L. Isik. Relational visual representations underlie human social interaction recognition. Nature Communications, 14(1):7317, 2023.

[41] T. N. Kipf and M. Welling. Variational graph auto-encoders. arXiv preprint arXiv:1611.07308, 2016.

[42] J. Liu, C. Zhang, Z. He, W. Zhang, and N. Li. Network-to-network: Self-supervised network representation learning via position prediction. IEEE Transactions on Knowledge and Data Engineering, 37(3):1354–1365, 2025.

[43] P. Falqueto et al. Human-Aware Robotics: Predict, Assist, and Plan for Seamless Interaction. PhD thesis, University of Trento, 2025.

[44] X. Mo, Y. Xing, and C. Lv. Heterogeneous edge-enhanced graph attention network for multiagent trajectory prediction. In arXiv preprint arXiv:2106.07161, 2021. URL https://arxiv. org/pdf/2106.07161.

[45] G. Alain and Y. Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.

[46] L. Prechelt. Early stopping-but when? In Neural Networks: Tricks of the trade, pages 55–69. Springer, 2002.

[47] Y. Shi, Z. Huang, S. Feng, H. Zhong, W. Wang, and Y. Sun. Masked label prediction: Unified message passing model for semi-supervised classification, 2021. URL https://arxiv.org/ abs/2009.03509.

[48] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

[49] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

## A Dataset and Graph Statistics

## A.1 Datasets

We evaluate our method on two datasets providing physical robot-human interaction data: SEAN Together and SocialNav-SUB. Both datasets are converted into the common graph format described in Section 3.

SEAN Together (SEAN-T). SEAN Together contains 8-second (5 Hz) immersive virtual-reality observations of robot navigation behavior paired with subjective human ratings [21]. Participants rated interactions along three dimensions: perceived competence, clarity of intention, and surprise. We use this to evaluate the extraction of subjective social impressions from physical motion. We organize the SEAN-T dataset by participant and then split it into train, validation, and test subsets corresponding to 80%, 16% and 4% of participants.

SocialNav-SUB (SNS). SocialNav-SUB expands the SCAND dataset [16] into a structured social reasoning benchmark for navigation scenes [6]. The dataset contains robot-centric interaction scenarios with relational action labels, such as avoiding,following, and yielding to. We use this dataset to evaluate action classification from physical interaction dynamics. Training uses automatically generated action labels from the SocialNav-SUB heuristic labeling procedure, with the training portion split into 85% training and 15% validation, while evaluation uses a separate held-out test set annotated by human labelers.

Table 2: Dataset and graph construction details. The temporal window is fixed within each dataset. The maximum number of edges follows from the maximum number of nodes under the fully connected graph construction.
<table><tr><td>Dataset</td><td># Scenarios</td><td>Sample Freq.</td><td>Window Size T</td><td>Window Duration</td><td>Max Nodes N</td><td>Max Edges</td></tr><tr><td>SEAN-T</td><td>2969</td><td>5Hz</td><td>40</td><td>8s</td><td>16</td><td>240</td></tr><tr><td>SocialNav-SUB</td><td>3052</td><td>8Hz</td><td>20</td><td>2.5s</td><td>31</td><td>930</td></tr></table>

## B Implementation Details

## B.1 Graph Construction Details

## B.1.1 Shared Graph Representation

All variants represent each fixed-duration interaction scenario as the spatiotemporal graph G of Eq. 1, where nodes correspond to observed agents and directed edges correspond to ordered pair wise interactions. Each edge stores a temporal sequence of relative spatial and motion features, enumerated in Appendix B.1.2. The homogeneous and heterogeneous variants differ only in how entity and relation type information is represented, and in which components are instantiated per dataset (Appendix B.2.1).

## B.1.2 Edge Feature Channels

Both variants use the same $d _ { e } = 8$ edge channels, differing only in sequence length. The first four are the relative $S E ( 2 )$ pose of node $j$ in the body frame of node i, and the remaining four are their temporal deltas, $\delta _ { t } u = u _ { t } - u _ { t - 1 } \colon$

$$
\mathbf { e } _ { i j , t } = \left[ \Delta x _ { i j , t } , \Delta y _ { i j , t } , \cos \Delta \theta _ { i j , t } , \sin \Delta \theta _ { i j , t } , \delta _ { t } \Delta x _ { i j } , \delta _ { t } \Delta y _ { i j } , \delta _ { t } \cos \Delta \theta _ { i j } , \delta _ { t } \sin \Delta \theta _ { i j } \right] .\tag{7}
$$

Each delta is set to zero at the first observed timestep of a contiguous track segment, so a pedestrian who leaves and re-enters the window contributes no spurious jump across the gap. Because headings enter only through (cos, sin) pairs, the encoding has no wrap-around discontinuity. The deltas are per-timestep differences rather than rates, so their scale depends on the sampling frequency; features are standard-scaled before training, which removes the resulting offset between datasets but not the difference in what one timestep represents physically.

## B.1.3 Homogeneous Graph

Unified Representation: The homogeneous model processes both SEAN Together and SocialNav-SUB (SNS) as single-type graphs, while robot and human identity is included only as a 2-class input node embedding. All data is unified to a 40-timestep window, with SNS interpolated from 20 to 40 steps, and projected into a shared 592-dimensional node feature space.

Feature Sets:

• Node Features: SEAN robot nodes contain 512D map features and 80D goal trajectories. For SNS robot nodes and all human nodes across both datasets, these 592 dimensions are zero-initialized. A bias-free linear projection is used so that zeroed features contribute only through the learnable type embeddings.

• Edge Features: 8-dimensional relative SE(2) poses and temporal deltas (Appendix B.1.2).

Processing: A Temporal Edge Transformer encodes the 40-timestep edge attributes into fixed-size embeddings. Message passing is performed using a homogeneous TransformerConv layer that operates on the entire graph simultaneously. An edge-level padding mask is derived from the valid mask to ensure the model ignores invalid timesteps during temporal pooling.

## B.1.4 Heterogeneous Graph

Multi-Type Topology: The heterogeneous model utilizes a HeteroData structure with two node types (robot, bystander) and three directed edge types: (robot, to, bystander), (bystander, to, robot), and (bystander, to, bystander). Unlike the homogeneous case, SEAN (40 steps) and SNS (20 steps) retain their native temporal resolutions, and batches are restricted to a single source to prevent temporal mismatch. For SEAN-only heterogeneous experiments, we additionally retain the SEANspecific follower node, adding directed edges between the follower and both robot and bystander nodes. This preserves the full interaction structure available in SEAN while keeping the combined heterogeneous model restricted to node and edge types shared across datasets.

Table 3: Node features by architecture and dataset. All configurations use 8D edge features comprising relative position, relative heading (sine and cosine), and their temporal differences. Both architectures use reconstruction pretraining followed by linear probing with a frozen encoder.
<table><tr><td>Model</td><td>Dataset</td><td>Robot features</td><td>Human features</td></tr><tr><td>STARS-He</td><td></td><td>SEAN-T 592D map and goal features</td><td>8D learned embedding + 8D pooled edge features</td></tr><tr><td>STARS-He SNS</td><td></td><td>8D learned embedding + 8D pooled edge features</td><td>8D learned embedding + 8D pooled edge features</td></tr><tr><td>STARS-Ho SEAN-T</td><td></td><td>592D map and goal features + type embedding</td><td>592D zero vector + type embedding</td></tr><tr><td>STARS-Ho SNS</td><td></td><td>592D zero vector + type embedding</td><td>592D zero vector + type embedding</td></tr></table>

## Feature Sets:

• SEAN Nodes: The robot uses 592D features (map + goal). Bystanders use a learnable 8D embedding concatenated with an 8-dimensional positional feature derived from the meanpooled relative pose of the robot-to-bystander edge.

• SNS Nodes: Both robot and bystander nodes use a learnable 8D embedding concatenated with an 8-dimensional positional feature derived from the mean-pooled relative pose of the given node’s incident robot-to-bystander edges.

• Edge Features: 8-dimensional SE(2) relative pose and dynamics, matching the dimensions of the homogeneous case but varying in sequence length.

## B.2 Graph Encoder Details

## B.2.1 Per-Dataset and Shared Components

Because $d _ { v } ^ { ( k ) }$ and $T _ { k }$ differ across datasets, components whose input or output dimensionality depends on these quantities are instantiated per dataset, while the relational message-passing core and latent heads are shared. Table 4 gives the split, which differs between the two variants and is the main structural distinction between them: the homogeneous variant unifies the datasets at conversion time and shares every component, while the heterogeneous variant keeps them separate through the input and output adapters and shares the relational core. In the heterogeneous variant this means four node feature projections, one per dataset and node type, three learned type embeddings, two temporal edge encoders, and two edge decoders.

Table 4: Components shared across datasets in each variant.
<table><tr><td>Component</td><td>Homogeneous</td><td>Heterogeneous</td></tr><tr><td>Node feature projection</td><td>shared</td><td>per dataset and node type</td></tr><tr><td>Temporal edge encoder</td><td>shared (T=40)</td><td>per dataset (40, 20)</td></tr><tr><td>Message-passing layers</td><td>shared</td><td>shared</td></tr><tr><td>Edge-type embedding</td><td>not used</td><td>shared</td></tr><tr><td>Latent heads  $f _ { \mu } , f _ { \sigma }$ </td><td>shared</td><td>shared</td></tr><tr><td>Edge decoder</td><td>shared (T=40)</td><td>per dataset</td></tr><tr><td>Node decoder</td><td colspan="2">SEAN-T robot node only</td></tr></table>

## B.2.2 Feature Preprocessing

Node attributes are first projected into a shared hidden dimension $d _ { h i d d e n }$ via type-specific Multi layer Perceptrons (MLPs):

$$
\mathbf { h } _ { i } ^ { ( 0 ) } = \mathbf { M } \mathbf { L } \mathbf { P } _ { \tau ( i ) } ( \mathbf { x } _ { i } )\tag{8}
$$

where $\tau ( i )$ denotes the node type. To encode temporal edge sequences across datasets, temporal edge sequences $\mathbf { e } _ { i j }$ are passed through a Transformer edge encoder with learned positional embeddings. This compresses the full temporal sequence into a fixed-dimensional relation embedding $\tilde { \mathbf { e } } _ { i j } \in \mathbb { R } ^ { d _ { h i d d e n } }$

$$
\tilde { \mathbf { e } } _ { i j } = \mathrm { E n c _ { \mathrm { e d g e } } } ( \mathbf { e } _ { i j } )\tag{9}
$$

## B.2.3 Message Passing

## Heterogeneous Message Passing

The graph encoder performs message passing over the complete interaction graph. In our heterogeneous variant, we utilize a HEAT-style update [44]. Let $\mathbf { h } _ { j } ^ { ( k ) }$ and ${ \bf h } _ { i } ^ { ( k ) }$ denote the source and destination node embeddings at layer $k ,$ and $\mathbf { t } _ { i j }$ denote a learned embedding for the specific edge type. We construct an edge-conditioned interaction representation:

$$
\mathbf { u } _ { i j } ^ { ( k ) } = \left[ \mathbf { h } _ { j } ^ { ( k ) } \parallel \mathbf { h } _ { i } ^ { ( k ) } \parallel \tilde { \mathbf { e } } _ { i j } ^ { ( k ) } \parallel \mathbf { t } _ { i j } \right]\tag{10}
$$

This representation encapsulates the full context of the interaction. During message passing, the network must determine both the content of the information being transmitted and the relative importance of that information. Therefore, we use $\mathbf { u } _ { i j } ^ { ( k ) }$ to independently parameterize both the raw attention score $s _ { i j } ^ { ( k ) }$ and the message content $\mathbf { m } _ { i j } ^ { ( k ) }$ via two learnable functions:

$$
s _ { i j } ^ { ( k ) } = f _ { \mathrm { a t t n } } \left( \mathbf { u } _ { i j } ^ { ( k ) } \right) , \qquad \mathbf { m } _ { i j } ^ { ( k ) } = f _ { \mathrm { m s g } } \left( \mathbf { u } _ { i j } ^ { ( k ) } \right)\tag{11}
$$

Formally, the destination-wise softmax normalizes the raw attention scores across all incoming edges from the neighborhood $\mathcal { N } ( i )$

$$
\alpha _ { i j } ^ { ( k ) } = \frac { \exp \left( s _ { i j } ^ { ( k ) } \right) } { \sum _ { m \in \mathcal { N } ( i ) } \exp \left( s _ { i m } ^ { ( k ) } \right) }\tag{12}
$$

These normalized weights are used to compute the aggregated neighborhood message m¯ <sup>(k)</sup>:

$$
\bar { \mathbf { m } } _ { i } ^ { ( k ) } = \sum _ { j \in \mathcal { N } ( i ) } \alpha _ { i j } ^ { ( k ) } \mathbf { m } _ { i j } ^ { ( k ) }\tag{13}
$$

The node state is updated via a residual connection and Layer Normalization (LN) to maintain numerical stability during training:

$$
\mathbf { h } _ { i } ^ { ( k + 1 ) } = \mathrm { L N } \Big ( \mathbf { h } _ { i } ^ { ( k ) } + f ^ { \mathrm { u p d } } \left( [ \mathbf { h } _ { i } ^ { ( k ) } \parallel \bar { \mathbf { m } } _ { i } ^ { ( k ) } ] \right) \Big )\tag{14}
$$

To enhance our edges’ influence in the GNN, and for future edge-level downstream task integration, we additionally update edge embeddings at each layer following the Group GNN convention [22]. The message content $\mathbf { m } _ { i j } ^ { ( k ) }$ , which was already computed for node aggregation previously, is reused as a residual update to the edge embedding:

$$
\tilde { \mathbf { e } } _ { i j } ^ { ( k + 1 ) } = \mathrm { L N } _ { \mathrm { e d g e } } \left( \tilde { \mathbf { e } } _ { i j } ^ { ( k ) } + \mathbf { m } _ { i j } ^ { ( k ) } \right)\tag{15}
$$

We perform a similar computation as the node update equation, except we omit a separate update function for $\mathbf { m } _ { i j } ^ { ( k ) }$ since it has already been transformed by an MLP, making an additional transformation redundant.

## Homogeneous Message Passing

For the homogeneous graph encoder, message passing does not use the HEAT-style update above. It instead applies a shared edge-conditioned attention convolution [47] in which the encoded temporal edge features $\tilde { \mathbf { e } } _ { i j }$ enter the key and value terms, with no edge-type embedding.

![](images/65ae704c5913091d7aa843b96b04518dba64d679e9bcf29c432c412dcb0074bd.jpg)  
Figure 3: STARS Heterogeneous architecture overview. Raw inputs (robot occupancy + goal, learned human embeddings, 8D temporal edge features) are projected to a 256D shared hidden space via type-specific preprocessing (MLPs for nodes, a Transformer for edges). The encoder performs HEAT-style message passing (MP) over the fully connected heterogeneous graph (2 layers), with attention and messages parameterized over the concatenated edge-conditioned representation $\mathbf { u } _ { i j } .$ The resulting 256D node states are projected through shared linear heads to per-node Gaussian posteriors and sampled to 128D latents z<sub>i</sub> via the reparameterization trick. The decoder reconstructs edge features (and robot features on SEAN) from latent pairs, trained jointly with a KL divergence term against $\mathcal { N } ( 0 , I )$ . Pretrained encoders are frozen for downstream linear probing (SLA).

## C Training and Evaluation Protocol

All experiments were implemented in PyTorch [48] using mixed precision.

We applied standard feature scaling and fast early stopping [46] with patience 20. All reported results are averaged across 10 random seeds. Models were optimized with AdamW using decoupled weight decay [49]; model-specific learning rates and selected hyperparameters are reported in Appendix C.3.

## C.1 Pretraining Protocol

Models were first pretrained using the ELBO self-supervised VAE objective described in Sec. 3. Graphs were processed independently within each minibatch, allowing each interaction scenario to contain a variable number of human nodes and edges. Optimization details and selected hyperparameters are provided in Appendix C.3.

## C.2 Downstream Adaptation Protocol

After pretraining, encoder weights were frozen and reused for downstream prediction tasks. For supervised linear adaptation (SLA), latent graph representations were extracted from the pretrained encoder and passed into lightweight linear prediction heads. Linear probes were trained independently for each downstream task while keeping the graph encoder fixed.

The probe input follows Eq. 6: for the graph-level SEAN-T tasks it concatenates the robot latent with the mean of the human latents, giving one $2 d _ { z } = 2 5 6$ dimensional vector per scenario, and for the node-level SNS task it concatenates the robot latent with that pedestrian’s own latent, giving one 256- dimensional vector per labeled pedestrian. The three independent binary SEAN-T heads therefore hold $3 \times ( 2 5 6 + 1 ) = 7 7 1$ parameters and the single five-way SNS head $5 \times ( 2 5 6 + 1 ) = 1 2 8 5$ matching Table 5. The SNS classifier retains all five action categories, but we report macro $F _ { 1 }$ over avoid, follow, and not consider. We exclude overtake and yield from the aggregate metric because their limited labeled support does not permit stable per-class evaluation.

## C.3 Hyperparameter Search

We tuned hyperparameters only for the unified SEAN+SNS models. Hyperparameter search was performed with Optuna using a TPE sampler and median pruning. Each trial first trained a unified GNN-VAE, then froze the encoder and trained an SNS linear probe. The Optuna objective was SNS probe validation loss using cross-entropy.

For the homogeneous graph model, the search selected a batch size of 32, dropout of 0.1, weight decay of $1 0 ^ { - 4 }$ , adaptive maximum mixing coefficient of $0 . 7 5 , \beta _ { \mathrm { m a x } } = 0 . 1$ , hidden dimension 256, latent dimension 128, three Transformer layers, and learning rate $1 0 ^ { - 3 }$

For the heterogeneous graph model, we ran three independent Optuna searches with $\beta _ { \mathrm { m a x } }$ fixed at 0.1, 0.5, and 0.9, optimizing the remaining hyperparameters within each search. The $\beta _ { \mathrm { m a x } } = 0 . 9$ search yielded the best downstream validation performance. The selected heterogeneous configuration used hidden dimension 256, latent dimension 128 per node, batch size 32, two HEAT-style message-passing layers, and a temporal edge Transformer with depth 3 and 4 attention heads. Training used a VAE learning rate of $1 . 1 5 6 \times 1 0 ^ { - 4 }$ , KL warmup of 25 epochs, dropout 0.2, adaptive sampling warmup of 20 epochs, adaptive ramp-up of 100 epochs, and maximum mixing coefficient 0.75.

All final unified models used AdamW optimization, VAE weight decay $1 0 ^ { - 4 }$ , probe weight decay $1 0 ^ { - 4 }$ , gradient clipping at 20.0, VAE early stopping patience 20, and probe early stopping patience 20. The unified search used SEAN graphs from combined v1.h5, SNS graphs from social nav v3.h5, and the fixed scene split split indices final.pt. SNS node features were dropped, so SNS encodings used learned node-type embeddings and edge geometry.

For the SEAN-only baseline, we did not run a separate Optuna search. Instead, the SEAN-only model reused the selected unified hyperparameters wherever applicable, including hidden dimension, latent dimension, learning rate, KL weight, KL warmup, dropout, batch size, weight decay, and gradient clipping. Unified-only adaptive SNS sampling parameters were omitted for SEANonly training.

## C.4 Sampling and Class Imbalance

We used stratified and adaptive sampling during unified GNN training for both heterogeneous and homogeneous models to address the large SNS class imbalance, where approximately 82% of training examples correspond to the not consider class. Although the VAE pretraining objective itself is unsupervised, the adaptive sampler uses downstream SNS label statistics to improve the class balance of batches used during unified training. Empirically, this sampling strategy improved downstream $F _ { 1 }$ scores for both SEAN and SNS tasks, with particularly noticeable gains for the homogeneous graph model.

## D Compute and Reproducibility Details

To support reproducibility and clarify the computational requirements of STARS, we report dataset sizes, graph construction details, model capacity, training runtimes, inference runtimes, and hardware details for all experiments.

## D.1 Hardware and Software

Hardware. All experiments were run primarily on the Lonestar6 and Stampede3 systems at the Texas Advanced Computing Center (TACC). Unless otherwise stated, models were trained on a single NVIDIA A100 PCIe GPU with 40 GB of GPU memory. Experiments were executed in an Ubuntu 22.04 containerized environment using Python 3.10, PyTorch 2.0.1, and CUDA-enabled GPU execution.

## D.2 Model Capacity

Table 5 reports the number of trainable parameters for STARS. For STARS, we report the capacity of the pretrained encoder-decoder separately from the lightweight downstream prediction head.

Table 5: Model capacity comparison. For STARS, parameter counts are separated into the pretrained representation model and the downstream task-specific prediction head.
<table><tr><td>Model / Component</td><td>Stage</td><td>Trainable Params.</td><td>Frozen Params.</td><td>Notes</td></tr><tr><td>STARS-He Encoder/Decoder</td><td>Pretraining</td><td>6,420,810</td><td>N/A</td><td>Heterogeneous graph autoencoder</td></tr><tr><td>STARS-Ho Encoder/Decoder</td><td>Pretraining</td><td>2,936,848</td><td>N/A</td><td>Homogeneous graph autoencoder</td></tr><tr><td>Linear Prediction Head</td><td>Downstream adaptation</td><td>1,285 (SNS), 771 (SEAN)</td><td>6,420,810</td><td>Used with frozen STARS encoder</td></tr></table>

## D.3 Runtime and Memory

Training and inference runtime. Table 6 reports the computational cost of each experimental pipeline. Pretraining time refers to the wall-clock time required to train the graph autoencoder. Downstream training time refers to fitting the task-specific prediction head on top of the frozen encoder. Inference runtime is reported per interaction scenario and includes representation extraction and prediction when applicable. Peak GPU memory is measured during training unless otherwise stated.

Table 6: Runtime and memory usage for each experimental pipeline. For STARS-based pipelines, pretraining is performed once and the frozen encoder is reused for downstream tasks. Downstream training time refers to fitting the task-specific prediction head. Inference runtime is reported per interaction scenario.
<table><tr><td>Model / Pipeline</td><td>Dataset / Task</td><td>Pretraining Time</td><td>Downstream Training Time Inference Time / Scenario Peak GPU Memory</td><td></td><td></td></tr><tr><td>STARS-He + Linear Head</td><td>SEAN-T + SNS</td><td>~2.5 hrs / run</td><td>&lt;1 sec</td><td>~12 ms</td><td>~13 GB</td></tr><tr><td>STARS-Ho + Linear Head</td><td>SEAN-T + SNS</td><td>~3 hrs / run</td><td>~12 sec</td><td>~30 ms</td><td>~17 GB</td></tr></table>

Table 7: Extended STARS results on SEAN-T and SNS across labeled-data fractions.
<table><tr><td>Task</td><td>Method</td><td>1%</td><td>5%</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td>Competence</td><td>STARS-He (Ours) STARS-Ho (Ours)</td><td> $0 . 6 3 5 \pm 0 . 1 0$   $0 . 6 1 2 \pm 0 . 1 4$ </td><td> $0 . 6 7 3 \pm 0 . 0 7$   $0 . 7 0 2 \pm 0 . 0 8$ </td><td> $0 . 7 2 6 \pm 0 . 0 4$   $0 . 7 3 8 \pm 0 . 0 5$ </td><td> $0 . 7 6 2 \pm 0 . 0 3$   $0 . 7 6 5 \pm 0 . 0 4$ </td><td> $0 . 7 8 1 \pm 0 . 0 3$   $0 . 7 8 7 \pm 0 . 0 3$ </td><td> $0 . 7 9 4 \pm 0 . 0 2$   $0 . 7 9 9 \pm 0 . 0 2$ </td></tr><tr><td>Surprise</td><td>STARS-He (Ours) STARS-Ho (Ours)</td><td> $0 . 4 5 5 \pm 0 . 1 7$   $0 . 4 4 0 \pm 0 . 2 3$ </td><td> $0 . 4 8 1 \pm 0 . 1 7$   $0 . 5 4 1 \pm 0 . 2 1$ </td><td> $0 . 5 2 6 \pm 0 . 1 5$   $0 . 5 4 8 \pm 0 . 2 0$ </td><td> $0 . 6 0 3 \pm 0 . 1 2$   $0 . 6 1 0 \pm 0 . 1 6$ </td><td> $0 . 6 5 5 \pm 0 . 0 8$   $0 . 7 1 3 \pm 0 . 0 6$ </td><td> $0 . 7 0 9 \pm 0 . 0 4$   $0 . 7 4 1 \pm 0 . 0 3$ </td></tr><tr><td>Intention</td><td>STARS-He (Ours) STARS-Ho (Ours)</td><td> $0 . 5 7 6 \pm 0 . 0 8$   $0 . 6 0 3 \pm 0 . 1 0$ </td><td> $0 . 6 1 9 \pm 0 . 0 5$   $0 . 6 7 1 \pm 0 . 0 7$ </td><td> $0 . 6 7 1 \pm 0 . 0 4$   $0 . 7 0 1 \pm 0 . 0 6$ </td><td> $0 . 6 9 4 \pm 0 . 0 3$   $0 . 7 1 5 \pm 0 . 0 4$ </td><td> $0 . 7 0 5 \pm 0 . 0 3$   $0 . 7 2 3 \pm 0 . 0 3$ </td><td> $0 . 7 2 1 \pm 0 . 0 3$   $0 . 7 3 4 \pm 0 . 0 2$ </td></tr><tr><td>Avoid</td><td>STARS-He (Ours) STARS-Ho (Ours)</td><td> $0 . 0 9 8 \pm 0 . 0 9$   $0 . 1 2 6 \pm 0 . 0 9$ </td><td> $0 . 1 7 3 \pm 0 . 1 1$   $0 . 2 3 4 \pm 0 . 0 6$ </td><td> $0 . 1 9 0 \pm 0 . 1 0$   $0 . 2 5 2 \pm 0 . 0 6$ </td><td> $0 . 2 1 9 \pm 0 . 1 0$   $0 . 2 5 9 \pm 0 . 0 6$ </td><td> $0 . 2 3 0 \pm 0 . 0 9$   $0 . 2 5 7 \pm 0 . 0 5$ </td><td> $0 . 2 4 2 \pm 0 . 0 8$   $0 . 2 6 7 \pm 0 . 0 5$ </td></tr><tr><td>Follow</td><td>STARS-He (Ours) STARS-Ho (Ours)</td><td> $0 . 1 5 5 \pm 0 . 1 5$   $0 . 1 3 2 \pm 0 . 1 3$ </td><td> $0 . 3 2 7 \pm 0 . 1 2$   $0 . 2 8 9 \pm 0 . 1 2$ </td><td> $0 . 3 7 4 \pm 0 . 0 8$   $0 . 3 4 3 \pm 0 . 0 7$ </td><td> $0 . 4 0 4 \pm 0 . 0 4$   $0 . 3 7 4 \pm 0 . 0 4$ </td><td> $0 . 4 1 2 \pm 0 . 0 3$   $0 . 3 8 5 \pm 0 . 0 4$ </td><td> $0 . 4 0 5 \pm 0 . 0 3$   $0 . 3 9 6 \pm 0 . 0 2$ </td></tr><tr><td>Not Consider</td><td>STARS-He (Ours) STARS-Ho (Ours)</td><td> $0 . 8 3 4 \pm 0 . 0 6$   $0 . 7 7 3 \pm 0 . 1 1$ </td><td> $0 . 8 4 0 \pm 0 . 0 4$   $0 . 7 6 8 \pm 0 . 0 5$ </td><td> $0 . 8 4 6 \pm 0 . 0 3$   $0 . 7 6 5 \pm 0 . 0 4$ </td><td> $0 . 8 5 5 \pm 0 . 0 2$   $0 . 7 7 1 \pm 0 . 0 3$ </td><td> $0 . 8 6 1 \pm 0 . 0 2$   $0 . 7 7 0 \pm 0 . 0 4$ </td><td> $0 . 8 6 2 \pm 0 . 0 2$   $0 . 7 6 3 \pm 0 . 0 4$ </td></tr><tr><td>Macro F1</td><td> $\mathrm { S T A R S  – H e ( O u r s ) }$  STARS-Ho (Ours)</td><td> $0 . 3 6 2 \pm 0 . 0 6$   $0 . 3 4 4 \pm 0 . 0 5$ </td><td> $0 . 4 4 7 \pm 0 . 0 5$   $0 . 4 3 1 \pm 0 . 0 5$ </td><td> $0 . 4 7 0 \pm 0 . 0 4$   $0 . 4 5 3 \pm 0 . 0 4$ </td><td> $0 . 4 9 3 \pm 0 . 0 3$   $0 . 4 6 8 \pm 0 . 0 3$ </td><td> $0 . 5 0 1 \pm 0 . 0 3$   $0 . 4 7 1 \pm 0 . 0 2$ </td><td> $0 . 5 0 3 \pm 0 . 0 2$   $0 . 4 7 5 \pm 0 . 0 2$ </td></tr></table>

<table><tr><td>Task</td><td>Model</td><td>1%</td><td>5%</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td>Competence</td><td>STARS-He</td><td> $0 . 5 6 8 \pm 0 . 1 0$ </td><td> $0 . 6 3 6 \pm 0 . 0 7$ </td><td> $0 . 6 8 6 \pm 0 . 0 6$ </td><td> $0 . 7 3 3 \pm 0 . 0 3$ </td><td> $0 . 7 7 3 \pm 0 . 0 3$ </td><td> $0 . 7 9 2 \pm 0 . 0 2$ </td></tr><tr><td>Surprise</td><td>STARS-He</td><td> $0 . 4 5 5 \pm 0 . 1 4$ </td><td> $0 . 4 5 3 \pm 0 . 1 6$ </td><td> $0 . 4 8 0 \pm 0 . 2 2$ </td><td> $0 . 5 5 \pm 0 . 1 5$ </td><td> $0 . 6 5 3 \pm 0 . 0 9$ </td><td> $0 . 7 1 6 \pm 0 . 0 5$ </td></tr><tr><td>Intention</td><td>STARS-He</td><td> $0 . 5 2 8 \pm 0 . 0 9$ </td><td> $0 . 6 0 1 \pm 0 . 0 5$ </td><td> $0 . 6 4 4 \pm 0 . 0 5$ </td><td> $0 . 6 8 3 \pm 0 . 0 3$ </td><td> $0 . 7 0 5 \pm 0 . 0 3$ </td><td> $0 . 7 2 1 \pm 0 . 0 2$ </td></tr></table>

Table 8: SEAN-T downstream performance after SEAN-T-only pretraining. Results report $F _ { 1 }$ -Score $( \mu \pm \sigma )$ across labeled-data fractions.

## E Additional Results and Ablations

## E.1 Extended Main Results

Table 7 provides the full downstream results for STARS across all supervised data fractions. We report SEAN-T performance for the three subjective social-impression tasks and SNS performance using per-class $F _ { 1 }$ scores together with macro- $F _ { 1 }$ . Across both datasets, performance generally improves as the amount of labeled data available for supervised linear adaptation increases. The heterogeneous and homogeneous variants show similar trends on SEAN-T, while the heterogeneous model achieves higher SNS macro- $F _ { 1 }$ at larger label fractions.

## E.2 Pretraining Data Ablations

SEAN-only Pretraining. We compare the jointly pretrained SEAN-T+SNS model against a SEAN-T only model to assess the effect of adding cross-dataset interaction data on SEAN-T downstream performance. The SEAN-T only model remains competitive with joint pretraining, particularly at larger labeled-data fractions. At 100% labeled data, it achieves 0.792/0.716/0.721 on competence, surprise, and intention, compared with 0.794/0.709/0.721 under joint pretraining. These results indicate that adding SNS data does not substantially alter SEAN-T downstream performance.

## E.3 Cross-Dataset Transfer

To evaluate whether STARS learns representations that transfer across datasets, we pretrain on one dataset and evaluate the frozen encoder on the other using the same linear-probing protocol as the main experiments. We compare single-dataset pretraining against joint SEAN-T+SNS pretraining.

At 100% labeled data, STARS-He achieves performance comparable to joint pretraining. SEAN-T-only pretraining achieves 0.497 macro $F _ { 1 }$ on SNS compared with 0.503 under joint pretraining, while SNS-only pretraining achieves $0 . 8 0 1 / 0 . 7 1 0 / 0 . 7 2 9$ on SEAN-T competence, surprise, and intention, compared with 0.794/0.709/0.721 under joint pretraining. STARS-Ho also transfers across datasets, although joint pretraining provides larger gains.

Table 9: Cross-dataset transfer to SNS. Models are pretrained either on SEAN-T only or jointly on SEAN-T and SNS and evaluated on SNS using progressively larger fractions of labeled training data. Results report macro F<sub>1</sub>-Score $( \mu \pm \sigma )$
<table><tr><td>Model</td><td>Pretraining</td><td>1%</td><td>5%</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td>STARS-He</td><td> $\mathbf { S E A N - T o n l y }$ </td><td> $0 . 3 5 5 \pm 0 . 0 6$ </td><td> $0 . 4 4 2 \pm 0 . 0 5$ </td><td> $0 . 4 6 3 \pm 0 . 0 4$ </td><td> $0 . 4 8 9 \pm 0 . 0 3$ </td><td> $0 . 4 9 3 \pm 0 . 0 3$ </td><td> $0 . 4 9 7 \pm 0 . 0 3$ </td></tr><tr><td>STARS-He</td><td>SEAN-T+SNS</td><td> $0 . 3 6 2 \pm 0 . 0 6$ </td><td> $0 . 4 4 7 \pm 0 . 0 5$ </td><td> $0 . 4 7 0 \pm 0 . 0 4$ </td><td> $0 . 4 9 3 \pm 0 . 0 3$ </td><td> $0 . 5 0 1 \pm 0 . 0 3$ </td><td> $0 . 5 0 3 \pm 0 . 0 2$ </td></tr><tr><td>STARS-Ho</td><td>SEAN-T only</td><td> $0 . 2 7 4 \pm 0 . 0 5$ </td><td> $0 . 3 3 2 \pm 0 . 0 4$ </td><td> $0 . 3 5 1 \pm 0 . 0 3$ </td><td> $0 . 3 6 5 \pm 0 . 0 3$ </td><td> $0 . 4 0 7 \pm 0 . 0 2$ </td><td> $0 . 4 1 9 \pm 0 . 0 1$ </td></tr><tr><td>STARS-Ho</td><td> $\mathbf { S E A N - T + S N S }$ </td><td> $0 . 3 4 4 \pm 0 . 0 5$ </td><td> $0 . 4 3 1 \pm 0 . 0 5$ </td><td> $0 . 4 5 3 \pm 0 . 0 4$ </td><td> $0 . 4 6 8 \pm 0 . 0 3$ </td><td> $0 . 4 7 1 \pm 0 . 0 2$ </td><td> $0 . 4 7 5 \pm 0 . 0 2$ </td></tr></table>

Table 10: Cross-dataset transfer to SEAN-T. Models are pretrained either on SNS only or jointly on SEAN-T and SNS and evaluated on the three SEAN-T perception tasks using progressively larger fractions of labeled training data. Results report F -Score $( \mu \pm \sigma )$ .
<table><tr><td>Model</td><td>Pretraining</td><td>Task</td><td>1%</td><td>5%</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td>STARS-He</td><td>SNS only</td><td>Competence</td><td> $0 . 6 1 5 \pm 0 . 1 0$ </td><td> $0 . 6 7 4 \pm 0 . 0 8$ </td><td> $0 . 7 3 1 \pm 0 . 0 5$ </td><td> $0 . 7 7 2 \pm 0 . 0 3$ </td><td> $0 . 7 8 6 \pm 0 . 0 3$ </td><td> $0 . 8 0 1 \pm 0 . 0 1$ </td></tr><tr><td>STARS-He</td><td>SNS only</td><td>Surprise</td><td>0.409 ± 0.20</td><td> $0 . 4 4 6 \pm 0 . 1 8$ </td><td> $0 . 4 9 3 \pm 0 . 1 7$ </td><td> $0 . 5 9 5 \pm 0 . 1 1$ </td><td> $0 . 6 5 2 \pm 0 . 0 8$ </td><td> $0 . 7 1 0 \pm 0 . 0 5$ </td></tr><tr><td>STARS-He</td><td>SNS only</td><td>Intention</td><td>0.571 ± 0.09</td><td>0.619 ± 0.08</td><td>0.668 ± 0.04</td><td>0.700 ± 0.03</td><td> $0 . 7 1 7 \pm 0 . 0 3$ </td><td> $0 . 7 2 9 \pm 0 . 0 3$ </td></tr><tr><td>STARS-He</td><td>SEAN-T+SNS</td><td>Competence</td><td>0.635 ± 0.10</td><td>0.673 ± 0.07</td><td>0.726 ± 0.04</td><td>0.762 ± 0.03</td><td>0.781 ± 0.03</td><td> $0 . 7 9 4 \pm 0 . 0 2$ </td></tr><tr><td>STARS-He</td><td>SEAN-T+SNS</td><td>Surprise</td><td>0.455 ± 0.17</td><td>0.481 ± 0.17</td><td>0.526 ± 0.15</td><td>0.603 ± 0.12</td><td>0.655 ± 0.08</td><td>0.709 ± 0.04</td></tr><tr><td>STARS-He</td><td>SEAN-T+SNS</td><td>Intention</td><td>0.576 ± 0.08</td><td>0.619 ± 0.05</td><td>0.671 ± 0.04</td><td>0.694 ± 0.03</td><td>0.705 ± 0.03</td><td> $0 . 7 2 1 \pm 0 . 0 3$ </td></tr><tr><td>STARS-Ho</td><td>SNS only</td><td>Competence</td><td>0.440 ± 0.27</td><td>0.507 ± 0.28</td><td>0.600 ± 0.24</td><td>0.735 ± 0.02</td><td>0.744 ± 0.03</td><td>0.766 ± 0.01</td></tr><tr><td>STARS-Ho</td><td>SNS only</td><td>Surprise</td><td>0.206 ± 0.28</td><td>0.285 ± 0.25</td><td>0.309 ± 0.26</td><td>0.481 ± 0.23</td><td>0.570 ± 0.14</td><td>0.655 ± 0.04</td></tr><tr><td>STARS-Ho</td><td>SNS only</td><td>Intention</td><td>0.493 ± 0.18</td><td>0.515 ± 0.19</td><td>0.573 ± 0.11</td><td>0.647 ± 0.02</td><td>0.655 ± 0.02</td><td>0.676 ± 0.02</td></tr><tr><td>STARS-Ho</td><td>SEAN-T+SNS</td><td>Competence</td><td>0.612 ± 0.14</td><td>0.702 ± 0.08</td><td>0.738 ± 0.04</td><td>0.765 ± 0.04</td><td>0.787 ± 0.03</td><td> $0 . 7 9 9 \pm 0 . 0 2$ </td></tr><tr><td>STARS-Ho</td><td>SEAN-T+SNS</td><td>Surprise</td><td> $0 . 4 4 0 \pm 0 . 2 3$ </td><td> $0 . 5 4 1 \pm 0 . 2 1$ </td><td>0.548 ± 0.20</td><td>0.610 ± 0.16</td><td>0.713 ± 0.06</td><td> $0 . 7 4 1 \pm 0 . 0 3$ </td></tr><tr><td>STARS-Ho</td><td>SEAN-T+SNS</td><td>Intention</td><td> $0 . 6 0 3 \pm 0 . 1 0$ </td><td> $0 . 6 7 1 \pm 0 . 0 7$ </td><td>0.701 ± 0.06</td><td> $0 . 7 1 5 \pm 0 . 0 4$ </td><td> $0 . 7 2 3 \pm 0 . 0 3$ </td><td> $0 . 7 3 4 \pm 0 . 0 2$ </td></tr></table>

## E.4 Pretraining Objective Ablations

Random Encoder. The random encoder baseline uses the same heterogeneous GNN-VAE architecture as the full model but removes self-supervised pretraining. This isolates whether downstream performance comes from the learned pretrained representation or simply from the architecture and linear probe. The random encoder performs noticeably worse than the full model on most of the F1- scores, indicating that the self-supervised graph reconstruction objective learns task-relevant latent structure.

## E.4.1 Self-Supervised vs. Supervised Heterogeneous Graph

To separate the value of self-supervised pretraining from the heterogeneous graph architecture itself, we compare the frozen STARS encoder with an otherwise similar heterogeneous model trained end-to-end using supervised SNS labels. As shown in Table 11, supervised training improves some minority-class scores such as avoid, but the self-supervised representation achieves stronger macro-$F _ { 1 }$ across all label fractions. This suggests that the reconstruction-based pretraining objective provides a more stable representation for low-data downstream adaptation under the SNS class imbal ance.

## E.5 Representation Component Ablations

Robot-Latent Only. The robot-latent only ablation uses only the learned robot latent representation for downstream prediction, omitting human-node latents from the downstream readout. This tests how much task-relevant information is summarized directly in the robot representation. The full representation performs substantially better on SNS minority action classes and on most SEAN-T perception tasks, although the robot-latent only representation remains competitive on SEAN-T intention.

Bystander-Latent Only. The bystander-latent only ablation uses only the mean pooled bystander latent representation, omitting the robot latent from the downstream readout. This tests how much predictive information can be recovered from surrounding pedestrian context alone. The bystander representation retains task relevant signal, but the full representation performs better overall, indicating that combining robot and pedestrian representations is more informative for downstream tasks.

Table 11: SNS Action Classification: Self-Supervised vs. Supervised GNN-VAE. We compare STARS’ frozen self-supervised encoder with a linear probe against an identical architecture trained end-to-end with supervised classification. Both models use the heterogeneous graph formulation. Results report F1-Score $( \mu \pm \sigma )$ across label fractions.
<table><tr><td>Task</td><td>Method</td><td>1%</td><td>5%</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td>Avoid</td><td>Hetero Self-Sup Hetero Sup</td><td> $0 . 0 9 8 \pm 0 . 0 9$   $0 . 1 7 8 \pm 0 . 0 7$ </td><td> $0 . 1 7 3 \pm 0 . 1 1$   $0 . 2 3 4 \pm 0 . 0 5$ </td><td> $0 . 1 9 0 \pm 0 . 1 0$   $0 . 2 5 8 \pm 0 . 0 4$ </td><td> $0 . 2 1 9 \pm 0 . 1 0$   $0 . 2 9 2 \pm 0 . 0 3$ </td><td> $0 . 2 3 0 \pm 0 . 0 9$   $0 . 2 9 7 \pm 0 . 0 2$ </td><td> $0 . 2 4 2 \pm 0 . 0 8$   $0 . 2 8 4 \pm 0 . 0 2$ </td></tr><tr><td>Follow</td><td>Hetero Self-Sup Hetero Sup</td><td> $0 . 1 5 5 \pm 0 . 1 5$   $0 . 1 1 2 \pm 0 . 1 1$ </td><td> $0 . 3 2 7 \pm 0 . 1 2$   $0 . 2 8 7 \pm 0 . 0 7$ </td><td> $0 . 3 7 4 \pm 0 . 0 8$   $0 . 3 0 5 \pm 0 . 0 6$ </td><td> $0 . 4 0 4 \pm 0 . 0 4$   $0 . 3 5 3 \pm 0 . 0 4$ </td><td> $0 . 4 1 2 \pm 0 . 0 3$   $0 . 3 7 3 \pm 0 . 0 4$ </td><td> $0 . 4 0 5 \pm 0 . 0 3$   $0 . 3 8 4 \pm 0 . 0 4$ </td></tr><tr><td>Not Consider</td><td>Hetero Self-Sup Hetero Sup</td><td> $0 . 8 3 4 \pm 0 . 0 6$   $0 . 2 1 8 \pm 0 . 2 5$ </td><td> $0 . 8 4 0 \pm 0 . 0 4$   $0 . 3 5 1 \pm 0 . 2 2$ </td><td> $0 . 8 4 6 \pm 0 . 0 3$   $0 . 4 2 9 \pm 0 . 2 3$ </td><td> $0 . 8 5 5 \pm 0 . 0 2$   $0 . 6 0 4 \pm 0 . 1 3$ </td><td> $0 . 8 6 1 \pm 0 . 0 2$   $0 . 6 6 2 \pm 0 . 0 8$ </td><td> $0 . 8 6 2 \pm 0 . 0 2$   $0 . 6 6 7 \pm 0 . 0 7$ </td></tr><tr><td>Macro F1</td><td>Hetero Self-Sup Hetero Sup</td><td> $0 . 3 6 2 \pm 0 . 0 6$   $0 . 1 6 9 \pm 0 . 0 9$ </td><td> $0 . 4 4 7 \pm 0 . 0 5$   $0 . 2 9 1 \pm 0 . 1 0$ </td><td> $0 . 4 7 0 \pm 0 . 0 4$   $0 . 3 3 1 \pm 0 . 1 0$ </td><td> $0 . 4 9 3 \pm 0 . 0 3$   $0 . 4 1 7 \pm 0 . 0 7$ </td><td> $0 . 5 0 1 \pm 0 . 0 3$   $0 . 4 4 4 \pm 0 . 0 4$ </td><td> $0 . 5 0 3 \pm 0 . 0 2$   $0 . 4 4 5 \pm 0 . 0 4$ </td></tr></table>

Table 12: Random encoder ablation results.
<table><tr><td></td><td></td><td>Architecture</td><td>1%</td><td>5%</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td rowspan="9">Random encoder:</td><td>Competence</td><td>Heterogeneous</td><td> $0 . 5 3 0 \pm 0 . 2 2$ </td><td> $0 . 5 9 6 \pm 0 . 1 6$ </td><td> $0 . 6 3 5 \pm 0 . 1 2$ </td><td> $0 . 6 7 7 \pm 0 . 0 9$ </td><td>0.691 ± 0.08</td><td> $0 . 7 0 7 \pm 0 . 0 7$ </td></tr><tr><td>Surprise</td><td>Heterogeneous</td><td> $0 . 2 5 1 \pm 0 . 2 0$ </td><td> $0 . 2 9 6 \pm 0 . 2 1$ </td><td>0.372 ± 0.17</td><td> $0 . 4 2 4 \pm 0 . 1 3$ </td><td> $0 . 4 6 7 \pm 0 . 1 2$ </td><td> $0 . 4 8 0 \pm 0 . 1 4$ </td></tr><tr><td>Intention</td><td>Heterogeneous</td><td> $0 . 6 0 6 \pm 0 . 2 0$ </td><td> $0 . 6 4 8 \pm 0 . 1 4$ </td><td> $0 . 6 7 7 \pm 0 . 1 0$ </td><td> $0 . 7 0 0 \pm 0 . 0 8$ </td><td> $0 . 7 0 4 \pm 0 . 0 8$ </td><td> $0 . 7 1 3 \pm 0 . 0 9$ </td></tr><tr><td>F1 Avoid</td><td>Heterogeneous</td><td> $0 . 0 7 0 \pm 0 . 1 1$ </td><td> $0 . 2 5 1 \pm 0 . 0 8$ </td><td> $0 . 3 0 1 \pm 0 . 0 3$ </td><td> $0 . 3 0 4 \pm 0 . 0 2$ </td><td> $0 . 3 0 5 \pm 0 . 0 2$ </td><td> $0 . 3 0 2 \pm 0 . 0 2$ </td></tr><tr><td> $F _ { 1 }$  Follow</td><td>Heterogeneous</td><td> $0 . 0 7 0 \pm 0 . 1 2$ </td><td> $0 . 0 6 5 \pm 0 . 1 1$ </td><td> $0 . 1 0 1 \pm 0 . 1 2$ </td><td> $0 . 1 0 1 \pm 0 . 1 1$ </td><td> $0 . 0 8 2 \pm 0 . 1 0$ </td><td> $0 . 0 9 5 \pm 0 . 1 0$ </td></tr><tr><td>F1 Not Consider</td><td>Heterogeneous</td><td> $0 . 8 6 5 \pm 0 . 0 3$ </td><td> $0 . 8 4 9 \pm 0 . 0 2$ </td><td> $0 . 8 3 3 \pm 0 . 0 2$ </td><td> $0 . 8 3 9 \pm 0 . 0 1$ </td><td> $0 . 8 3 9 \pm 0 . 0 1$ </td><td> $0 . 8 3 9 \pm 0 . 0 0$ </td></tr><tr><td>PA</td><td>Heterogeneous</td><td> $0 . 6 0 1 \pm 0 . 0 3$ </td><td> $0 . 5 9 0 \pm 0 . 0 2$ </td><td> $0 . 5 7 7 \pm 0 . 0 2$ </td><td> $0 . 5 8 3 \pm 0 . 0 1$ </td><td> $0 . 5 8 4 \pm 0 . 0 1$ </td><td> $0 . 5 8 3 \pm 0 . 0 0$ </td></tr></table>

Raw Robot Node Only. The raw robot-node baseline bypasses graph pretraining entirely and trains directly on the 592-dimensional SEAN robot feature vector. This tests whether the SEAN labels can be predicted from robot-centric observations alone without relational message passing. The raw robot-node baseline contains substantial predictive signal, particularly for competence and surprise. At 100% labeled data, it achieves 0.746/0.700/0.688 on competence, surprise, and intention, compared with 0.794/0.709/0.721 for the full heterogeneous STARS representation. Compared with the robot-latent-only ablation, the raw features perform better on competence and surprise but worse on intention, indicating that the different representations retain complementary task-relevant information.

## E.6 Baseline and Lower-Bound Analyses

Weighted Random Baseline. The weighted random baseline samples predictions according to the empirical training-label distribution, providing a label-aware lower bound under class imbalance. Its strong performance on majority labels such as not consider highlights how misleading accuracy or single-class F1 can be on imbalanced SNS labels. However, overall, our full model substantially outperforms this baseline in nearly every F1 metric, showing that it learns beyond label-frequency priors.

## E.7 SNS $F _ { 1 }$ -Score Analysis

We evaluate the SocialNav-SUB vision-language-model (VLM) baselines on the same held-out SNS test split used for our downstream action classification task. Rather than independently rerunning the prompted VLM baselines, we obtained their per-example prediction outputs from the SocialNav-SUB authors and computed metrics on the corresponding examples in our evaluation split. For each scenario, the model predicts a discrete relational action label for the target pedestrian. We compute per-class $F _ { 1 }$ -scores and Probability of Agreement (PA) following the SocialNav-SUB evaluation protocol [6]. Because these models are evaluated zero-shot through prompting and are not fine-

<table><tr><td></td><td></td><td>Architecture</td><td>1%</td><td>5%</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td>128D robot-z only</td><td>Competence</td><td>Heterogeneous</td><td> $0 . 5 9 2 \pm 0 . 1 3$ </td><td> $0 . 6 2 7 \pm 0 . 1 0$ </td><td> $0 . 6 6 3 \pm 0 . 0 8$ </td><td> $0 . 6 9 5 \pm 0 . 0 7$ </td><td> $0 . 7 0 7 \pm 0 . 0 7$ </td><td> $0 . 7 1 1 \pm 0 . 0 8$ </td></tr><tr><td></td><td>Surprise</td><td>Heterogeneous</td><td> $0 . 3 7 3 \pm 0 . 1 2$ </td><td> $0 . 4 0 0 \pm 0 . 1 2$ </td><td> $0 . 4 1 0 \pm 0 . 1 1$ </td><td> $0 . 4 3 2 \pm 0 . 1 2$ </td><td> $0 . 4 5 4 \pm 0 . 1 2$ </td><td> $0 . 4 7 9 \pm 0 . 1 4$ </td></tr><tr><td></td><td>Intention</td><td>Heterogeneous</td><td> $0 . 6 2 3 \pm 0 . 1 2$ </td><td> $0 . 6 6 2 \pm 0 . 0 8$ </td><td> $0 . 6 9 6 \pm 0 . 0 8$ </td><td> $0 . 7 1 6 \pm 0 . 0 7$ </td><td> $0 . 7 3 0 \pm 0 . 0 7$ </td><td> $0 . 7 3 0 \pm 0 . 0 7$ </td></tr><tr><td></td><td>F1 Avoid</td><td>Heterogeneous</td><td> $0 . 0 1 7 \pm 0 . 0 5$ </td><td> $0 . 0 0 4 \pm 0 . 0 2$ </td><td> $0 . 0 0 1 \pm 0 . 0 1$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td></td><td>F1 Follow</td><td>Heterogeneous</td><td> $0 . 0 0 4 \pm 0 . 0 2$ </td><td> $0 . 0 0 8 \pm 0 . 0 3$ </td><td> $0 . 0 0 7 \pm 0 . 0 2$ </td><td> $0 . 0 0 3 \pm 0 . 0 1$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0$ </td></tr><tr><td></td><td>F1 Not Consider</td><td>Heterogeneous</td><td> $0 . 8 7 5 \pm 0 . 0 2$ </td><td> $0 . 8 8 1 \pm 0 . 0 1$ </td><td> $0 . 8 8 3 \pm 0 . 0 0$ </td><td> $0 . 8 8 3 \pm 0 . 0 0$ </td><td> $0 . 8 8 3 \pm 0 . 0 0$ </td><td> $0 . 8 8 3 \pm 0 . 0 0$ </td></tr><tr><td></td><td>PA</td><td>Heterogeneous</td><td> $0 . 6 1 0 \pm 0 . 0 2$ </td><td> $0 . 6 1 6 \pm 0 . 0 1$ </td><td> $0 . 6 1 8 \pm 0 . 0 0$ </td><td> $0 . 6 1 8 \pm 0 . 0 0$ </td><td> $0 . 6 1 8 \pm 0 . 0 0$ </td><td> $0 . 6 1 8 \pm 0 . 0 0$ </td></tr></table>

Table 13: Robot-latent only (128D) ablation results.

Table 14: 128D byst only ablation results.
<table><tr><td colspan="2"></td><td>Architecture</td><td>1%</td><td>5%</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td rowspan="7">128D byst only</td><td>Competence</td><td>Heterogeneous</td><td> $0 . 5 5 1 \pm 0 . 1 7$ </td><td> $0 . 5 3 3 \pm 0 . 1 5$ </td><td> $0 . 5 5 6 \pm 0 . 1 5$ </td><td> $0 . 6 2 1 \pm 0 . 1 0$ </td><td> $0 . 6 4 2 \pm 0 . 0 9$ </td><td> $0 . 6 5 0 \pm 0 . 0 7$ </td></tr><tr><td>Surprise</td><td>Heterogeneous</td><td> $0 . 3 2 3 \pm 0 . 1 5$ </td><td> $0 . 3 3 7 \pm 0 . 1 3$ </td><td> $0 . 3 0 2 \pm 0 . 1 3$ </td><td> $0 . 2 3 2 \pm 0 . 1 4$ </td><td> $0 . 1 6 1 \pm 0 . 1 5$ </td><td> $0 . 1 4 9 \pm 0 . 1 4$ </td></tr><tr><td>Intention</td><td>Heterogeneous</td><td> $0 . 5 7 4 \pm 0 . 1 7$ </td><td> $0 . 5 9 8 \pm 0 . 1 5$ </td><td> $0 . 6 3 4 \pm 0 . 1 3$ </td><td> $0 . 6 9 3 \pm 0 . 0 9$ </td><td> $0 . 7 1 3 \pm 0 . 0 9$ </td><td> $0 . 7 1 7 \pm 0 . 0 9$ </td></tr><tr><td>F1 Avoid</td><td>Heterogeneous</td><td> $0 . 1 3 4 \pm 0 . 1 0$ </td><td> $0 . 1 5 4 \pm 0 . 0 8$ </td><td> $0 . 1 6 7 \pm 0 . 0 8$ </td><td> $0 . 1 4 9 \pm 0 . 0 9$ </td><td> $0 . 1 4 8 \pm 0 . 0 8$ </td><td> $0 . 1 5 3 \pm 0 . 0 9$ </td></tr><tr><td>F1 Follow</td><td>Heterogeneous</td><td> $0 . 2 2 3 \pm 0 . 1 2$ </td><td> $0 . 3 1 7 \pm 0 . 0 6$ </td><td> $0 . 3 4 1 \pm 0 . 0 4$ </td><td> $0 . 3 4 4 \pm 0 . 0 3$ </td><td> $0 . 3 4 4 \pm 0 . 0 3$ </td><td> $0 . 3 4 3 \pm 0 . 0 3$ </td></tr><tr><td>F1 Not Consider</td><td>Heterogeneous</td><td> $0 . 6 6 1 \pm 0 . 0 8$ </td><td> $0 . 7 5 6 \pm 0 . 0 6$ </td><td> $0 . 7 8 6 \pm 0 . 0 4$ </td><td> $0 . 8 0 7 \pm 0 . 0 3$ </td><td> $0 . 8 0 7 \pm 0 . 0 3$ </td><td> $0 . 8 1 1 \pm 0 . 0 2$ </td></tr><tr><td>PA</td><td>Heterogeneous</td><td> $0 . 4 1 9 \pm 0 . 0 6$ </td><td> $0 . 4 9 8 \pm 0 . 0 5$ </td><td> $0 . 5 2 4 \pm 0 . 0 3$ </td><td> $0 . 5 3 9 \pm 0 . 0 3$ </td><td> $0 . 5 3 9 \pm 0 . 0 2$ </td><td> $0 . 5 4 2 \pm 0 . 0 2$ </td></tr></table>

tuned for this task, we report results only on the full test set rather than across labeled-data fractions. Table 17 summarizes the resulting per-class $F _ { 1 }$ and PA scores.

Table 15: 592D robot node only (SEAN-ablation compared with combined model).
<table><tr><td></td><td>Architecture</td><td>1%</td><td>5%</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td>Competence</td><td>Heterogeneous</td><td> $0 . 5 4 1 \pm 0 . 1 1$ </td><td> $0 . 6 4 1 \pm 0 . 0 7$ </td><td> $0 . 6 7 3 \pm 0 . 0 7$ </td><td> $0 . 7 1 4 \pm 0 . 0 5$ </td><td> $0 . 7 2 4 \pm 0 . 0 4$ </td><td> $0 . 7 4 6 \pm 0 . 0 3$ </td></tr><tr><td>Surprise</td><td>Heterogeneous</td><td> $0 . 5 2 8 \pm 0 . 1 0$ </td><td> $0 . 5 9 1 \pm 0 . 0 9$ </td><td> $0 . 6 1 0 \pm 0 . 0 8$ </td><td> $0 . 6 5 2 \pm 0 . 0 5$ </td><td> $0 . 6 8 2 \pm 0 . 0 4$ </td><td> $0 . 7 0 0 \pm 0 . 0 4$ </td></tr><tr><td>Intention</td><td>Heterogeneous</td><td> $0 . 4 9 8 \pm 0 . 0 9$ </td><td> $0 . 5 8 0 \pm 0 . 0 6$ </td><td> $0 . 5 9 0 \pm 0 . 0 5$ </td><td> $0 . 6 3 2 \pm 0 . 0 5$ </td><td> $0 . 6 4 9 \pm 0 . 0 5$ </td><td> $0 . 6 8 8 \pm 0 . 0 5$ </td></tr></table>

Table 16: WRB ablation results.
<table><tr><td></td><td></td><td>Architecture</td><td>1%</td><td>5%</td><td>10%</td><td>25%</td><td>50%</td><td>100%</td></tr><tr><td>WRB:</td><td>Competence</td><td>Heterogeneous</td><td> $0 . 5 2 4 \pm 0 . 1 4$ </td><td> $0 . 5 2 8 \pm 0 . 1 2$ </td><td> $0 . 5 1 7 \pm 0 . 1 0$ </td><td> $0 . 5 1 7 \pm 0 . 1 0$ </td><td> $0 . 5 3 0 \pm 0 . 1 1$ </td><td> $0 . 5 1 3 \pm 0 . 1 0$ </td></tr><tr><td></td><td>Surprise</td><td>Heterogeneous</td><td> $0 . 3 4 5 \pm 0 . 1 3$ </td><td> $0 . 3 6 6 \pm 0 . 1 2$ </td><td> $0 . 3 8 4 \pm 0 . 1 0$ </td><td> $0 . 3 6 0 \pm 0 . 0 8$ </td><td> $0 . 3 6 8 \pm 0 . 0 8$ </td><td> $0 . 3 6 0 \pm 0 . 0 9$ </td></tr><tr><td></td><td>Intention</td><td>Heterogeneous</td><td> $0 . 5 6 3 \pm 0 . 1 1$ </td><td> $0 . 5 7 4 \pm 0 . 0 9$ </td><td> $0 . 5 7 5 \pm 0 . 0 9$ </td><td> $0 . 5 7 8 \pm 0 . 0 8$ </td><td> $0 . 5 7 5 \pm 0 . 0 8$ </td><td> $0 . 5 6 3 \pm 0 . 0 8$ </td></tr><tr><td></td><td>F1 Avoid</td><td>Heterogeneous</td><td> $0 . 0 9 0 \pm 0 . 0 7$ </td><td> $0 . 1 0 4 \pm 0 . 0 5$ </td><td> $0 . 1 0 4 \pm 0 . 0 4$ </td><td> $0 . 1 0 8 \pm 0 . 0 4$ </td><td> $0 . 1 0 5 \pm 0 . 0 5$ </td><td> $0 . 1 0 3 \pm 0 . 0 5$ </td></tr><tr><td></td><td>F1 Follow</td><td>Heterogeneous</td><td> $0 . 0 4 6 \pm 0 . 0 6$ </td><td> $0 . 0 6 2 \pm 0 . 0 5$ </td><td> $0 . 0 6 9 \pm 0 . 0 5$ </td><td> $0 . 0 8 3 \pm 0 . 0 4$ </td><td> $0 . 0 7 0 \pm 0 . 0 5$ </td><td> $0 . 0 8 3 \pm 0 . 0 5$ </td></tr><tr><td></td><td>F1 Not Consider</td><td>Heterogeneous</td><td> $0 . 8 1 2 \pm 0 . 0 5$ </td><td> $0 . 8 0 6 \pm 0 . 0 3$ </td><td> $0 . 8 0 0 \pm 0 . 0 2$ </td><td> $0 . 8 1 1 \pm 0 . 0 2$ </td><td> $0 . 8 1 3 \pm 0 . 0 2$ </td><td> $0 . 8 0 9 \pm 0 . 0 1$ </td></tr><tr><td></td><td>PA</td><td>Heterogeneous</td><td> $0 . 5 4 2 \pm 0 . 0 5$ </td><td> $0 . 5 3 2 \pm 0 . 0 3$ </td><td> $0 . 5 2 7 \pm 0 . 0 2$ </td><td> $0 . 5 3 6 \pm 0 . 0 2$ </td><td> $0 . 5 3 9 \pm 0 . 0 1$ </td><td> $0 . 5 3 6 \pm 0 . 0 1$ </td></tr></table>

<table><tr><td>Model</td><td> $F _ { 1 }$  Avoid</td><td> $F _ { 1 }$  Follow</td><td> $F _ { 1 }$  Overtake</td><td> $F _ { 1 }$  Yield</td><td> $F _ { 1 }$  NC</td><td>PA</td></tr><tr><td>Gemini</td><td>0.223</td><td>0.000</td><td>0.000</td><td>0.222</td><td>0.745</td><td>0.493</td></tr><tr><td>40</td><td>0.241</td><td>0.176</td><td>0.000</td><td>0.000</td><td>0.521</td><td>0.344</td></tr><tr><td>04-mini</td><td>0.222</td><td>0.363</td><td>0.000</td><td>0.111</td><td>0.852</td><td>0.582</td></tr></table>

Table 17: SocialNav-SUB VLM baseline results on the held-out SNS action classification task [6]. Models are evaluated on the full downstream test set. Metrics report per-class $F _ { 1 }$ -scores alongside Probability of Agreement (PA).