# Propagate, Then Sharpen: Post-Hoc Refinement of Frozen Node Classifiers

Preben Johnsen Bentdal Department of Informatics, University of Bergen Norwegian Research Centre (NORCE) pbe027@uib.no

Nello Blaser Department of Informatics, University of Bergen nello.blaser@uib.no

Xue-Cheng Tai Norwegian Research Centre (NORCE) xtai@norceresearch.no

## Abstract

We study post-hoc refinement of frozen node classifiers: given only the graph G and class distributions Q predicted by a frozen model, can we improve accuracy without access to node features, model parameters, or gradients? APPNP answers this by propagating logits with a restart towards the initial predictions, minimizing the anchored Dirichlet energy. Instead, we consider the Potts energy, and decompose it into a Dirichlet term, which penalizes disagreement between neighbouring nodes, and a Gini term, which penalizes indecision within each node. This decomposition motivates Propagate, Then Sharpen (PtS), which alternates between propagation of class probabilities and node-wise, mass-preserving sharpening, with only one additional hyperparameter selected using labelled validation nodes. Across nine homophilic graphs, with a frozen MLP backbone, PtS improves mean test accuracy over independently tuned APPNP by 1.71 percentage points on clean inputs and 3.90 under severe Gaussian feature corruption. Gains over APPNP become smaller, but remain positive with frozen GCN and GraphSAGE backbones. Sharpening also removes most of the accuracy loss of deep propagation: on clean inputs without restart, accuracy falls by 2.2 points between 2 and 100 propagation steps under PtS, compared with 33.8 for APPNP.

## 1 Introduction

Graph-based node classification uses node attributes (features) and relationships between nodes (structure). These two inputs are not equally reliable. Features can be fragile: sensors drift or fail, attributes go missing or stale for part of the graph, or features are perturbed to protect privacy, while the graph structure usually stays intact. When this happens, correcting the model is often not an option, because retraining may be costly or access to parameters or training data might be restricted. These constraints motivate post-hoc methods that reuse the deployed model as a frozen predictor (Boudiaf et al., 2022). We ask how much prediction accuracy can be recovered using only the frozen class distributions Q and the graph G, without access to node features, model parameters, gradients, or training data. The refinement should apply to any backbone, so that another predictor can easily be swapped in. Validation labels should only be used for hyperparameter selection, and refinement should tolerate a mismatch between validation and test conditions. We therefore evaluate refinement both on clean inputs, and under Gaussian feature corruption on datasets with continuous-valued node features for which additive Gaussian noise provides a meaningful controlled perturbation.

A natural way to recover accuracy on G is to propagate predictions along its edges, and APPNP is the standard way to do so: personalized PageRank propagation of the logits, with a restart towards the original prediction at every step (Gasteiger et al., 2019). Propagation exploits informative neighbours, but repeated propagation also removes distinctions between nodes, which can result in oversmoothing (Oono & Suzuki, 2020), i.e. predictions in a connected component converge towards a common value as the number of propagation steps grows. Restart is what prevents this collapse in APPNP, but it only helps if the reference prediction is reliable. If features are missing or corrupted at test time, the restart repeatedly re-injects corrupted predictions, and propagation spreads these errors through the graph. APPNP is therefore sensitive to its depth and restart hyperparameters, and without restart it collapses at large depth (Figure 2).

The space in which predictions are propagated matters. APPNP propagates logits, so large class margins exert great influence, which helps when margins are reliable but amplifies confident errors. Propagating probabilities instead bounds the initial scores in [0, 1], limiting the influence of extreme logits. The trade-of is that averaging neighbouring distributions can weaken class preferences when neighbours disagree. This motivates a refinement method that acts in two directions: smoothing predictions across neighbouring nodes, and sharpening the class distribution within each node. We start from the relaxed Potts interaction used in variational image segmentation (Potts, 1952; Liu et al., 2022; Tai et al., 2024; Liu et al., 2024). On the probability simplex, the interaction splits exactly into a Dirichlet term, which penalizes disagreement between neighbouring nodes, and a Gini term, which penalizes indecision within each node. This decomposition pairs graph propagation with a local sharpening step that reduces indecision within each node.

We call the resulting method Propagate, Then Sharpen (PtS). It alternates between a propagation and a sharpening step. Sharpening is controlled by a single parameter η: at η = 0 we recover propagation without sharpening exactly, so validation can switch sharpening of, as η → ∞ sharpening approaches a one-hot assignment when the most probable class is unique, and in between it acts as a soft thresholding. Sharpening never changes a node’s own predicted class, it changes the distribution used by the next propagation step. PtS uses only Q and G, labels enter only through hyperparameter selection, any backbone that outputs class probabilities can be used, and its gain over APPNP persists at large propagation depth (Figure 2), as well as when hyperparameters are selected a a diferent corruption severity than the one evaluated at (Figure 4).

## Contributions

• We introduce Propagate, Then Sharpen (PtS), which alternates probability-space graph propagation with a mass-preserving categorical sharpening step derived from the Dirichlet–Gini decomposition of the Potts interaction. Sharpening adds one hyperparameter η, is of at η = 0, provably preserves every node’s predicted class, and decreases a local KL–Gini energy at each step (Proposition 1).

• We show that sharpening reduces accuracy degradation with propagation depth. On a constructed two-community graph, restart alone fails while PtS keeps every node correct at every depth (Proposition 2). Across nine homophilic graphs on clean inputs without restart, increasing depth from K = 2 to K = 100 reduces mean accuracy by 5.2 percentage points for PtS and 22.0 for Logit-Sharp (APPNP with sharpening in logit-space) at the same sharpening parameter η = 16. Increasing PtS’s sharpening parameter to η = 200 reduces its drop to 2.2 points (Figure 2).

• On nine homophilic graphs, PtS gains 3.90 points over independently tuned APPNP under severe Gaussian corruption and 1.71 on clean inputs (Table 1), and improves on average over a graph-adapted LAME and a post-hoc Graph-TV baseline (Tables 8 and 9).

Related Work The sharpening step of PtS comes from variational image segmentation, where the Potts model penalizes the boundaries between regions of a soft assignment field. PottsMGNet and Double-well Net connect this formulation with trainable neural architectures by deriving their updates from Potts energies and operator-splitting schemes (Tai et al., 2024; Liu et al., 2024), and soft threshold dynamics adds spatial regularization and shape priors to the output of a segmentation network (Liu et al., 2022). We transfer the same splitting idea from the image domain to a graph, where the assignment field becomes the class distributions of the nodes and the split separates smoothing from local sharpening. The combination already appears on graphs in difuse-interface methods, which pair smoothing with phase separation, and in Merriman–Bence–Osher (MBO) schemes, which alternate difusion and thresholding (Bertozzi & Flenner, 2012; Merkurjev et al., 2013; Garcia-Cardona et al., 2014). PtS is a soft mass-preserving counterpart applied to frozen predictions rather than labels, approaching hard thresholding as η → ∞ when the largest class probability is unique. Allen–Cahn Message Passing (ACMP) and GREAD apply reaction–difusion dynamics to learned node representations inside trained graph neural networks (Wang et al., 2023; Choi et al., 2023), PtS acts on the outputs of an already trained model and has no learnable parameters. Yang et al. (2025) incorporate a graph-based non-local TV-regularized softmax into GNN training. We evaluate its objective as a post-hoc baseline on frozen predictions, denoted Graph-TV (Appendix I).

Post-hoc propagation methods refine predictions by spreading information through the graph structure. PPNP and APPNP combine neural prediction with personalized PageRank propagation in logit space, with APPNP using an iterative approximation with restart (Gasteiger et al., 2019). Correct & Smooth applies label propagation after training, using known training labels to correct prediction errors and smooth the corrected predictions (Huang et al., 2021). PtS uses no labels at inference, but its sharpening step can be inserted into the smoothing stage of Correct & Smooth (Appendix L). Laplacian Adjusted Maximum-Likelihood Estimation (LAME) refines a frozen classifier’s output without updating its parameters. Its objective combines KL fidelity to the original class probabilities with a pairwise agreement term weighted by afinities computed from pretrained feature representations (Boudiaf et al., 2022). Replacing that afinity with the graph operator turns its update into a fixed-point iteration of the full KL-plus-Potts energy (Appendix C), which we evaluate as a further baseline (LAME-Graph). PtS alternates anchored propagation with local, mass-preserving sharpening, using the propagated distribution as the reference for each sharpening step.

## 2 Method

Let $G = ( \nu , \mathcal { E } )$ be a graph with $N = | \nu |$ nodes and let C be the number of classes. We symmetrize directed source graphs, remove duplicate edges and existing self-loops. Let $A \in \{ 0 , 1 \} ^ { N \times N }$ denote the resulting binary adjacency without self-loops, ${ \widetilde { A } } = A + I$ the adjacency with added self-loops, $\widetilde { D }$ the diagonal degree matrix $\begin{array} { r } { \widetilde { D } _ { i i } = \sum _ { j } \widetilde { A } _ { i j } } \end{array}$ , and S the normalised adjacency

$$
S = \widetilde { D } ^ { - 1 / 2 } \widetilde { A } \widetilde { D } ^ { - 1 / 2 } .
$$

The operator S is symmetric, element-wise non-negative, and spec $( S ) \subseteq [ - 1 , 1 ]$ with 1 an eigenvalue.

A frozen classifier $f _ { \theta }$ produces logits $Z \in \mathbb { R } ^ { N \times C }$ and class distributions $Q = \mathrm { s o f t m a x } ( Z ) \in \mathbb { R } ^ { N \times C }$ , with each row on the probability simplex $\Delta ^ { C - 1 } = \{ p \in \mathbb { R } ^ { C } : p \geq 0 , \ \mathbf { 1 } ^ { \top } p = 1 \}$

Refinement produces a sequence of states $U ^ { ( 0 ) } , \dots , U ^ { ( K ) } \in \mathbb { R } ^ { N \times C }$ . We write $U _ { i } ^ { ( k ) } \in \mathbb { R } ^ { C }$ for row i of $U ^ { ( k ) }$ , treated as a column vector.

![](images/c76c22d7aceca83d62773dd3fbb580b8fead518869e4022162806a790871151d.jpg)  
Figure 1: Propagate, Then Sharpen. Node colours show two-class proportions in an illustrative first iteration $( \alpha = 0 . 1 , \eta = 3 . 5 )$ , initialized at $Q .$ Propagation and sharpening update all nodes, and the sharpened state becomes the current state for the next iteration. Propagation includes self-loops and a restart towards the fixed Q. Pie charts show normalised class proportions.

## 2.1 Propagate

We start from the personalized PageRank recursion of APPNP (Gasteiger et al., 2019), but apply it to $Q$ instead of logits:

$$
U ^ { ( 0 ) } = Q , \qquad U ^ { ( k + 1 ) } = \mathcal { T } \bigl ( U ^ { ( k ) } \bigr ) , \qquad \mathcal { T } ( U ) = \alpha Q + \bigl ( 1 - \alpha \bigr ) S U\tag{1}
$$

with restart probability $\alpha \in [ 0 , 1 ]$ and K propagation steps. The only diference from APPNP is the state being propagated: APPNP propagates the logits Z and applies a softmax at the end, whereas this recursion propagates the class distributions Q. We call the variant PPR-Prob and use it to isolate the efect of the propagation space. Since S does not preserve unit row sums, we retain the unnormalized states during propagation and row-normalize $U ^ { ( K ) }$ to obtain the final class distributions. Normalising after every step instead is the PPR-rn variant compared in Appendix N.

## 2.2 Propagate, Then Sharpen

PtS extends this propagation with a node-wise categorical sharpening step. In a reaction–difusion interpretation, this local sharpening plays the role of the reaction step. As illustrated in Figure 1, from $U ^ { ( 0 ) } = Q$ , we alternate

between

$$
\widetilde { U } ^ { ( k + 1 ) } = \alpha Q + ( 1 - \alpha ) S U ^ { ( k ) }
$$

propagate

$$
U _ { i } ^ { ( k + 1 ) } = R _ { \eta } \Big ( \widetilde { U } _ { i } ^ { ( k + 1 ) } \Big ) \quad i = 1 , \dots , N\tag{2a}
$$

sharpen

(2b)

where the reaction $R _ { \eta }$ is defined in (4) below. Symmetric normalization does not preserve unit row sums. We therefore define the sharpening map on the simplex and then lift it to unnormalised rows. For $p \in \Delta ^ { C - 1 }$ with $p > 0$ and a reaction strength $\eta \geq 0$ , let

$$
r _ { \eta } ( p ) = \mathrm { s o f t m a x } ( \log p + \eta p ) ,\tag{3}
$$

which maps $\Delta ^ { C - 1 }$ into itself. For a propagated row $h \in \mathbb { R } _ { > 0 } ^ { C }$ with total mass $m ( h ) = \mathbf { 1 } ^ { \top } h$ , the reaction is

$$
R _ { \eta } ( h ) = m ( h ) r _ { \eta } \biggl ( \frac { h } { m ( h ) } \biggr ) .\tag{4}
$$

At $\eta = 0$ we recover $R _ { 0 } ( h ) = h _ { ; }$ , so PtS reduces exactly to PPR-Prob with the same $( \alpha , K )$ . Every row of $\widetilde { U } ^ { ( k + 1 ) }$ is strictly positive, so (4) is well defined: $Q = \operatorname { s o f t m a x } ( Z ) > 0$ elementwise, and the added self-loops give S a strictly positive diagonal, so positivity is preserved for every $\alpha \in [ 0 , 1 ]$

Writing the reaction in the two stages of (3)–(4) separates what it does. The map $r _ { \eta }$ redistributes probability mass within a node, and the factor $m ( h )$ restores the row total,

$$
\mathbf { 1 } ^ { \top } R _ { \eta } ( h ) \ = \ m ( h ) \mathbf { 1 } ^ { \top } r _ { \eta } ( h / m ( h ) ) \ = \ m ( h ) \ = \ \mathbf { 1 } ^ { \top } h .
$$

The reaction therefore changes a node’s class distribution while preserving the mass the row carries into the next propagation step, which is what determines the node’s total contribution to its neighbours. We compare this choice with normalizing each row’s mass after every iteration in Table 16. Since η multiplies the normalised rows, the sharpening strength is invariant to the row total.

For every $\eta \geq 0$ , the reaction keeps the relative ranking of the classes within each row (Appendix D). It therefore never changes a node’s own predicted class, and influences later predictions only through the state that enters the next propagation step

Without restart $( \alpha = 0 )$ and without sharpening, propagation collapses: on a connected graph, every node converges to the same class distribution, and the predictions of APPNP converge to a single class shared by all nodes (Appendix D.1). The experiments in Figure 2 show this collapse and how sharpening counteracts it.

## 2.3 Energy interpretation and local descent

Motivated by the entropy-regularized Potts formulations in Tai et al. (2024), consider normalised class distributions $u _ { i } \in \Delta ^ { C - 1 }$ and a symmetric afinity matrix W with $W _ { i j } \geq 0$ (for PtS, $W = S )$ . The relaxed Potts interaction is

$$
E ( U ) = \frac { 1 } { 2 } \sum _ { i , j } W _ { i j } \left( 1 - u _ { i } ^ { \top } u _ { j } \right) .
$$

For one-hot assignments, this is a weighted penalty for disagreement between connected nodes. For soft assignments, it also penalizes indecision within each node. Writing $d _ { i } = \textstyle \sum _ { j } W _ { i j }$ gives

$$
E ( U ) = \underbrace { \frac { 1 } { 4 } \sum _ { i , j } W _ { i j } \| u _ { i } - u _ { j } \| ^ { 2 } } _ { E _ { \mathrm { D } } ( U ) } + \underbrace { \frac { 1 } { 2 } \sum _ { i } d _ { i } \left( 1 - \| u _ { i } \| ^ { 2 } \right) } _ { E _ { \mathrm { G } } ( U ) } .
$$

The Dirichlet term $E _ { \mathrm { D } } ( U )$ penalizes diferences between neighbouring distributions, while the Gini term $E _ { \mathrm { G } } ( U )$ penalizes indecision within each node. This decomposition motivates the two operations of PtS. Propagation is a gradient step on an anchored, degree-normalised Dirichlet objective, while sharpening decreases a local KL–Gini energy. This does not establish descent of the joint Potts objective. Figure 6 illustrates how these components interact during propagation on WikiCS.

Propagation descends an anchored, degree-normalised Dirichlet objective. Following the optimization view of APPNP, consider $\begin{array} { r } { J ( U ) = \frac { \alpha } { 2 } \| U - Q \| _ { F } ^ { 2 } + \frac { 1 - \alpha } { 2 } \operatorname { t r } \bigl ( U ^ { \top } ( I - S ) U \bigr ) } \end{array}$ , whose first term anchors the state near Q and whose second term is a degree-normalised Dirichlet energy, so that α balances closeness to the original prediction against smoothness over the graph. Since $\nabla J ( U ) = \alpha ( U - Q ) + ( 1 - \alpha ) ( I - S ) U$ , a gradient step with unit step size gives $U - \nabla J ( U ) = \alpha Q + ( 1 - \alpha ) S U = \mathcal { T } ( U )$ : one propagation step is exactly one unit-step gradient step on J (Zhu et al., 2021).

Sharpening descends the Gini term of a single node. For a propagated class distribution $p _ { i } ,$ , define

$$
E _ { \eta } ( \boldsymbol { u } ; p _ { i } ) = \mathrm { K L } ( \boldsymbol { u } \| p _ { i } ) + \frac { \eta } { 2 } \left( 1 - \| \boldsymbol { u } \| ^ { 2 } \right) , \qquad \boldsymbol { u } \in \Delta ^ { C - 1 }\tag{5}
$$

where the KL term keeps the update close to $p _ { i }$ and the Gini term favours more decisive class distributions. Stationarity gives the fixed-point condition u = softmax $( \log p _ { i } + \eta u )$ , and one fixed-point step initialized at $u = p _ { i }$ as in the limited fixed-point iterations of PottsMGNet (Tai et al., 2024), returns exactly the sharpening map (3). Appendix B gives the full derivation, and that single step already decreases the energy it comes from.

Proposition 1 (Sharpening descends the local energy). Let $p \in \Delta ^ { C - 1 }$ with $p > 0$ , let $\eta \geq 0$ and let $p ^ { * } = r _ { \eta } ( p )$ Then $E _ { \eta } ( p ^ { * } ; p ) \leq E _ { \eta } ( p ; p )$ , and consequently $\begin{array} { r } { \frac { \eta } { 2 } \big ( \| p ^ { * } \| ^ { 2 } - \| p \| ^ { 2 } \big ) \geq \mathrm { K L } ( p ^ { * } \| p ) \geq 0 , } \end{array}$ , so one sharpening step does not increase the Gini term. Moreover, $p ^ { * }$ preserves the ordering of the classes of $p .$

The proof (Appendix D) uses the concavity of the Gini term: sharpening minimizes a linearized upper bound of $E _ { \eta } ( \cdot ; p )$ that is tight at p, so iterating it is a concave–convex procedure for (5) (Yuille & Rangarajan, 2003).

Restart is not enough Sharpening also changes the behaviour of deep propagation. Restart does not by itself protect a community from a confident neighbouring one: on a two-community graph, APPNP and PPR-Prob with $\alpha = 0 . 1$ misclassify one entire community for every depth $K \geq 3$ , while PtS with $\eta = 4$ keeps every node correct at every depth (Proposition 2 and Figure 5 in Appendix D.2). The mechanism there is that propagation cannot push a node’s correct-class probability below 0.546 while it is at least 0.6 everywhere, and one sharpening step lifts 0.546 back above 0.6.

The Potts decomposition motivates combining agreement between neighbours with decisiveness within each node. PtS propagates class probabilities, then applies one local KL–Gini sharpening step while preserving the propagated row mass. Sharpening preserves the current predicted class, so it afects labels only through subsequent propagation. Setting $\eta = 0$ recovers PPR-Prob exactly.

## 3 Experimental protocol

We use nine homophilic graphs: WikiCS (Mernyei & Cangea, 2022), Cora-TAPE, PubMed-TAPE and TAPE-Arxiv23 (He et al., 2024), ogbn-arxiv and ogbn-products (Hu et al., 2020), and Ele-Photo, Ele-Computers and Books-History (Yan et al., 2023). We use Roman-Empire and Amazon-Ratings (Platonov et al., 2023) as heterophilic controls and report them separately. WikiCS uses its first ten supplied training/validation splits and its shared test mask, the OGB graphs use their oficial split, and all other graphs use ten unstratified random 60/20/20 splits. We use three backbone seeds per split, each (graph, split, backbone seed) defines one unit. Dataset statistics are given in Appendix E.

For each unit we train a graph-blind MLP on clean features, minimizing cross-entropy on the training labels, and freeze the checkpoint with the highest clean validation accuracy, architecture and optimizer settings are given in Appendix F. The GCN (Kipf & Welling, 2017) and GraphSAGE (Hamilton et al., 2017) backbones follow the same training and corruption protocol, model sizes are given in Table 5.

At inference, we corrupt all nodes as

$$
X _ { i j } ^ { ( \sigma , r ) } = X _ { i j } + \sigma s _ { j } \xi _ { i j } ^ { ( r ) } , \qquad \xi _ { i j } ^ { ( r ) } \sim \mathcal { N } ( 0 , 1 ) ,
$$

where $s _ { j }$ is the standard deviation of feature $j$ over the training nodes and $\sigma \in \{ 0 , 0 . 5 , 1 , 1 . 5 , 2 \}$ . We draw three independent Gaussian fields $\xi ^ { ( r ) }$ per graph and split, shared across backbone seeds and methods, and reuse them across severities, so that the severities form a nested noise ladder. Cora-TAPE and PubMed-TAPE use RoBERTabase (Liu et al., 2019) CLS embeddings computed without fine-tuning, and the three CS-TAG graphs (Ele-Photo, Ele-Computers, Books-History) use the published RoBERTa-base CLS embeddings. For each corruption, we compute $Z = f _ { \boldsymbol { \theta } } \big ( X ^ { ( \sigma , r ) } \big )$ and $Q = \operatorname { s o f t m a x } ( Z )$ once and use these across methods and experiments. APPNP refers to the propagation stage of Gasteiger et al. (2019), applied post hoc to frozen logits Z, rather than the full model trained end to end.

APPNP, PPR-Prob and PtS are each tuned independently with 250 Optuna trials (Akiba et al., 2019) per unit, severity and noise draw, maximizing validation accuracy. APPNP and PPR-Prob tune $( \alpha , K )$ , and PtS additionally tunes $\eta ,$ including $\eta = 0 ;$ , full search spaces are in Appendix G and the selected values in Appendix H. Unless stated otherwise, hyperparameters are selected at the same severity at which they are evaluated, transfer across severities is reported in Figure 4. For each severity, we average test accuracies over draws, then over seeds, within each split, and report mean ± standard deviation across splits. For the OGB graphs, which have a single oficial split, standard deviations instead describe variation across seed×draw repeats. Diferences between methods are computed within (split, seed, draw) before aggregation.

We compare against two external post-hoc baselines using the same frozen predictions, graph and validation data. LAME-Graph is the graph-adapted LAME of Boudiaf et al. (2022), which replaces the feature afinity by S (Appendix C) and tunes the interaction strength and the number of iterations using 250 Optuna trials. Graph-TV applies the non-local total-variation objective of Yang et al. (2025) post hoc to the frozen predictions, with $\varepsilon = 1$ and the regularization strength selected by validation accuracy over a grid of 26 candidates. Graph-TV is evaluated at $\sigma \in \{ 0 , 2 \}$ only and excludes ogbn-products because of its memory requirements. Both baselines are detailed in Appendix I.

## 4 Results

## 4.1 Accuracy on clean and corrupted inputs

Across the nine main graphs with a frozen MLP, PtS improves mean test accuracy over independently tuned PPR-Prob by 0.57 percentage points on clean inputs and 1.09 at $\sigma = 2 .$ . The corresponding gains over post-hoc APPNP are 1.71 and 3.90 points (Table 1). With the frozen MLP backbone at $\sigma = 2 .$ mean accuracy rises from 41.95% for the unrefined predictions Q to 56.34% for APPNP and 60.24% for PtS.

Table 1: Test accuracy (%) with a frozen MLP backbone. Each method is tuned independently at the evaluation severity. The mean excludes the heterophilic controls.
<table><tr><td></td><td colspan="4"> $\sigma = 0$ </td><td colspan="4"> $\sigma = 2$ </td></tr><tr><td>Dataset</td><td>Q(MLP)</td><td>APPNP</td><td>PtS</td><td> $\mathrm { P t S \mathrm { ~ - ~ } A P P N P }$ </td><td>Q(MLP)</td><td>APPNP</td><td>PtS</td><td>PtS – APPNP</td></tr><tr><td>WikiCS</td><td>72.67</td><td>77.78</td><td>78.42</td><td> $+ 0 . 6 4 \pm 0 . 5 0$ </td><td>46.36</td><td>67.97</td><td>72.15</td><td> $+ 4 . 1 8 \pm 1 . 7 5$ </td></tr><tr><td>Cora-TAPE</td><td>60.51</td><td>76.16</td><td>79.74</td><td> $+ 3 . 5 8 \pm 1 . 2 3 $ </td><td>43.82</td><td>68.16</td><td>72.30</td><td> $+ 4 . 1 3 \pm 1 . 9 9$ </td></tr><tr><td>PubMed-TAPE</td><td>81.59</td><td>84.35</td><td>85.18</td><td> $+ 0 . 8 3 \pm 0 . 6 4$ </td><td>64.56</td><td>78.87</td><td>80.73</td><td> $+ 1 . 8 6 \pm 1 . 4 4$ </td></tr><tr><td>TAPE-Arxiv23</td><td>67.59</td><td>69.53</td><td>69.72</td><td> $+ 0 . 1 9 \pm 0 . 1 0$ </td><td>28.56</td><td>35.63</td><td>38.91</td><td> $+ 3 . 2 9 \pm 0 . 2 6$ </td></tr><tr><td>ogbn-arxiv</td><td>55.95</td><td>66.10</td><td>66.56</td><td> $+ 0 . 4 6 \pm 0 . 5 1$ </td><td>22.65</td><td>33.86</td><td>40.62</td><td> $+ 6 . 7 6 \pm 1 . 7 3$ </td></tr><tr><td>ogbn-products</td><td>59.58</td><td>70.58</td><td>72.17</td><td> $+ 1 . 5 8 \pm 0 . 2 4$ </td><td>22.46</td><td>28.54</td><td>27.33</td><td> $- 1 . 2 1 \pm 0 . 1 7$ </td></tr><tr><td>Ele-Photo</td><td>66.83</td><td>71.87</td><td>74.91</td><td> $+ 3 . 0 4 \pm 0 . 6 3$ </td><td>44.15</td><td>55.97</td><td>61.43</td><td> $+ 5 . 4 6 \pm 1 . 0 4$ </td></tr><tr><td>Ele-Computers</td><td>60.96</td><td>75.37</td><td>80.11</td><td> $+ 4 . 7 4 \pm 0 . 9 6$ </td><td>38.05</td><td>59.94</td><td>69.56</td><td> $+ 9 . 6 3 \pm 1 . 0 0$ </td></tr><tr><td>Books-History</td><td>81.38</td><td>82.59</td><td>82.88</td><td> $+ 0 . 2 9 \pm 0 . 2 3$ </td><td>66.91</td><td>78.11</td><td>79.13</td><td> $+ 1 . 0 2 \pm 0 . 6 1$ </td></tr><tr><td>heterophilic controls</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Roman-Empire Amazon-Ratings</td><td>65.54 49.52</td><td>65.65</td><td>65.54 52.74</td><td> $- 0 . 1 1 \pm 0 . 1 3$ </td><td>19.34 34.17</td><td>19.30</td><td>19.30</td><td> $- 0 . 0 1 \pm 0 . 0 2$ </td></tr><tr><td></td><td></td><td>51.63</td><td></td><td> $+ 1 . 1 2 \pm 0 . 3 5$ </td><td></td><td>36.74</td><td>36.78</td><td> $+ 0 . 0 4 \pm 0 . 0 8$ </td></tr><tr><td>Mean (9 main)</td><td>67.45</td><td>74.93</td><td>76.63</td><td>+1.71</td><td>41.95</td><td>56.34</td><td>60.24</td><td>+3.90</td></tr></table>

PtS beats APPNP on all nine main graphs on clean inputs and on eight out of nine at $\sigma = 2 ,$ where the positive gains range from 1.02 to 9.63 points. The exception is ogbn-products, where $\mathrm { A P P N P }$ is better on noisy data while PtS beats it on clean data. The advantage over APPNP is negligible on the two heterophilic controls at $\sigma = 2$ . For the other baselines, at $\sigma = 2 .$ , PtS also improves mean accuracy over LAME-Graph by 3.75 points across nine graphs. LAME-Graph uses the coupled Potts update, placing the graph interaction inside the softmax while retaining Q as its reference, PtS propagates first and sharpens relative to the propagated distribution (Appendix C). Against Graph-TV, the improvement is 4.24 points on the eight graphs evaluated by both methods (Table 8).

## 4.2 Sharpening and propagation depth

At $\sigma = 2$ , independently tuned PPR-Prob improves mean accuracy over APPNP by 2.81 points, and PtS improves over PPR-Prob by an additional 1.09 points. On clean inputs, the PtS–PPR-Prob gap is 0.57 points. Per-dataset comparisons are given in Table 10, and the full corruption ladder is reported in Table 14. Additionally, to test if the selected PtS configurations depend on sharpening, we keep their $( \alpha , K )$ fixed and set $\eta = 0$ without retuning (Reaction OFF). Mean accuracy at $\sigma = 2$ then falls from 60.24% to 54.93%, a drop of 5.31 points (Table 2). Figure 3 visualizes how the two contributions change across the full corruption ladder, and Table 14 gives the dataset-level values.

Table 2: Mean accuracy (%) over the nine main graphs at $\sigma = 2$ with the frozen MLP. PPR-Prob is independently tuned, Reaction OFF reuses the PtS-selected propagation settings without retuning.
<table><tr><td>Method</td><td>Sharpening</td><td>Accuracy</td></tr><tr><td>PPR-Prob</td><td> $\eta = 0$ </td><td>59.15</td></tr><tr><td>PtS</td><td>Selected η</td><td>60.24</td></tr><tr><td>Reaction OFF</td><td> $\eta = 0$ </td><td>54.93</td></tr></table>

Next, we vary the propagation depth K, while holding each curve’s restart α and sharpening strength η fixed (Figure 2). Logit-Sharp applies the sharpening step on top of logit-space propagation using APPNP.

Without restart, increasing K from 2 to 100 reduces clean accuracy by 33.8 points for APPNP and 31.2 for PPR-Prob, compared with 2.2 points for PtS with $\eta = 2 0 0$ . With noise σ = 2, APPNP and PPR-Prob lose 21.1 and 20.5 points, while PtS with $\eta = 2 0 0$ reaches 57.7% at both depths with a maximum in between. With restart $\alpha = 0 .$ 1 on clean inputs, accuracy decreases from 74.5% to 70.4% for APPNP and from 75.7% to 74.8% for PtS with $\eta = 2 0 0$ between $K = 2$ and $K = 1 0 0$ : losses of approximately 4.1 and 0.9 percentage points. At $K = 1 0 0$ , both PtS curves remain above APPNP and PPR-Prob in every restart and corruption condition shown; complete depth results are reported in Table 17.

![](images/0213f208193474e77c2720ad7f23a57333fdba92b9f3de019120244ca0deacee.jpg)  
Figure 2: Mean accuracy over the nine main graphs at fixed $( \alpha , \eta )$ . Columns vary the restart, rows vary the corruption severity.

## 4.3 Corruption severity

Figure 3 shows positive mean gains for PtS over both APPNP and PPR-Prob at every tested corruption severity. With hyperparameters selected independently for each method at the evaluation severity, the gain over APPNP is 1.71 percentage points on clean inputs and 3.90 at $\sigma = 2 .$ , while the gains over PPR-Prob are 0.57 and 1.09 points. The main comparisons select hyperparameters at the severity used for evaluation. Figure 4 examines how improvements transfer across selection and evaluation severities. The mean PtS-APPNP gain remains positive across all 25 combinations. When both methods are selected using clean validation data, PtS keeps a 2.28 point advantage at $\sigma = 2$ , compared with 3.90 points under severity-matched selection.

![](images/b31e927e32462fc8ccbff1677b45d1ceb12d65ddee5daff9e286ffc65027d1d8.jpg)  
Figure 3: Mean test-accuracy gains in percentage points over the nine main homophilic graphs with a frozen MLP. The curves compare PPR-Prob with APPNP, PtS with PPR-Prob, and PtS with APPNP. Each method is independently selected at the evaluation severity.

![](images/b0e990d9dcd219cd844c2a699df0b6d364888dd3e302a33bcd336786bf3107a5.jpg)  
Figure 4: Mean PtS−APPNP test-accuracy gains in percentage points over the nine homophilic graphs with a frozen MLP. Rows give the severity used for hyperparameter selection and columns the evaluation severity. The first row shows clean-selection transfer.

## 4.4 Backbone

Under the same-severity selection protocol used for the main comparisons, we apply PtS to frozen GCN and GraphSAGE predictions, this still gives positive gains of 0.88 and 1.23 points at $\sigma = 2$ (Table 3). Both backbones show positive gains over APPNP on all nine main graphs at $\sigma = 2 .$ , and on seven (GCN) and nine (GraphSAGE) on clean inputs (Appendix K). Their smaller gains compared to MLPs are consistent with graph-aware backbones profiting less from post-hoc graph propagation. The Correct & Smooth extension is evaluated separately in Appendix L.

Table 3: Mean test-accuracy gain of PtS over independently tuned APPNP across the nine main datasets, in percentage points.
<table><tr><td>Frozen backbone</td><td> $\sigma = 0$ </td><td> $\sigma = 2$ </td></tr><tr><td>MLP</td><td>+1.71</td><td>+3.90</td></tr><tr><td>GCN</td><td>+0.41</td><td>+0.88</td></tr><tr><td>GraphSAGE</td><td>+0.47</td><td>+1.23</td></tr></table>

## 4.5 Calibration and runtime

With an MLP at $\sigma = 2 ,$ , PtS has a higher raw negative log-likelihood (NLL) and expected calibration error (ECE) than APPNP, so its uncalibrated confidences are less trustworthy. Temperature scaling fitted on validation nodes (Guo et al., 2017) reduces the mean NLL of PtS below that of APPNP without changing accuracy, although its ECE stays higher (Table 20). PtS costs between 1.3 and 54 ms per propagation step (Table 19).

PtS improves mean accuracy on clean and corrupted inputs and substantially reduces the accuracy loss from deep propagation. It keeps its advantage under clean-only selection and with graph-aware backbones. Sharpening worsens raw calibration, which temperature scaling largely mitigates.

## 5 Discussion

Findings Sharpening never changes a node’s own predicted class, so its accuracy gain comes from subsequent propagation steps. Turning the sharpening of at PtS-selected (α, K) lowers accuracy by 5.31 points, showing that these selected propagation settings depend on sharpening (Table 10). Gains over APPNP are generally larger in higher local-homophily bins and in lower prediction-confidence quintiles (Table 18). The largest gain is in the 0.6–0.8 local-homophily bin rather than at 0.8–1. One possible explanation is that, when neighbours are confident in their predictions, propagation alone already produces a decisive and mostly correct distribution, so sharpening has less to add, while in more uncertain local regimes, averaging can leave the node with a correct but small margin and one sharpening step restores it (Lemma 1) before the next propagation. The two local-homophily bins below 0.4 have negative PtS-APPNP gains. Post-hoc refinement also recovers a substantial part of the corruption damage: with the MLP backbone the frozen prediction loses 25.5 points between σ = 0 and σ = 2, and PtS cuts that loss to 16.4 points, a 36% reduction against 27% for APPNP (Table 1). The graph-aware backbones lose less to begin with (8 and 14 points) and refinement still recovers 25% and 22% of it (Appendix K). Their smaller headroom might be due to their propagation operators acting as a form of Laplacian smoothing that reduces Dirichlet disagreement (Li et al., 2018).

Limitations The main results select hyperparameters at the severity at which they are evaluated, so they assume validation nodes that reflect test-time conditions. Figure 4 shows that the gain survives clean-only validation, but the gain shrinks. We study Gaussian corruption of node features only, leaving missing features and perturbations of the graph itself to future work. Additionally, sharpening worsens raw calibration, but this is mitigated by temperature scaling (Table 20). All nodes share a single η, and propagation favours agreement between connected nodes so that neither component can adapt to the reliability of a local neighbourhood, and misleading neighbours can reinforce errors rather than correct them. The negligible gains on the heterophilic controls and the loss against APPNP on ogbn-products (Table 1) show that the benefit is not universal. Node- or edge-adaptive parameters are therefore a natural extension, ACMP provides a related example of learned attractive and repulsive interactions (Wang et al., 2023). Finally, PtS does not minimize the full Potts energy on the graph. Propagation is normalised PPR, and we use the propagated distribution as the reference for the local sharpening step, so Proposition 1 is only a loca descent guarantee at a single node. Empirically, the two PtS steps act on complementary parts of the energy. In the example on WikiCS, propagation reduces the Dirichlet term, while sharpening increases it but strongly reduces the Gini term (Figure 6). After a few iterations, both energy terms stabilize and validation accuracy reaches a plateau, while the propagation without sharpening continues to deteriorate. The energy formulation also provides a natural route for extending PtS with additional regularizers or constraints. Variational and Potts-based image segmentation methods have incorporated geometric and topological priors such as convexity, connectivity and volume constraints (Liu et al., 2020, 2022; Li et al., 2026). Similar constraints could be carried over to graphs that carry geometric or topological information that can be exploited.

Conclusion Propagation over the graph can recover accuracy from degraded features, but its standard safeguard against oversmoothing, restart, reinjects the predictions that have become unreliable. Splitting the relaxed Potts interaction into Dirichlet and Gini terms adds an extra safeguard: propagate class probabilities, then sharpen each node. PtS adds one parameter to APPNP-style propagation, needs only the frozen class distribution and the graph and costs a few milliseconds per propagation step. With an MLP backbone, it improves mean test accuracy over independently tuned APPNP by 3.90 points at σ = 2 and 1.71 on clean inputs across nine graphs, and removes most of the accuracy drop of deep propagation. Gains are smaller with graph-aware backbones and negligible on heterophilic graphs. Smoothing across neighbours and decisiveness within nodes can be treated as separate operations, and keeping nodes decisive enables deeper propagation.

## AI use statement

Generative AI tools assisted with editing and polishing the text, including checking the consistency of notation and formula definitions and identifying awkward phrasing and spelling errors. AI tools also assisted with literature search, including identifying LAME as potentially relevant prior work, the main author read and evaluated the original paper before incorporating it into the manuscript and experimental comparison. AI tools were also used for code review and for cleaning up the submitted code for readability, including removing stale experiments not included in the paper, numerical consistency checks, and generating Figures 1 and 5. They were also consulted for feedback on experiments and as discussion partners for exploratory ideas. All AI-assisted material was reviewed and checked for accuracy by the authors, who take responsibility for the final text, code, claims, and results of this work.

## Reproducibility statement

The refinement method is fully specified by (2)–(4), and derived in Appendix B; the proofs of Proposition 1 and Proposition 2 are given in Appendix D and Appendix D.2. The experimental protocol, including the backbones, corruption model, splits, and aggregation, is described in Section 3; dataset statistics are listed in Table 4, model sizes in Appendix F, search spaces in Table 6, and the selected hyperparameters in Appendix H. Baseline objectives and search settings are given in Appendix I. Code is available at: https://github.com/Prebenno/ Propagate-Then-Sharpen.

## Funding

P.J.B. is supported by a PhD fellowship funded by the Research Council of Norway through the STIPINST programme (project no. 342607), hosted by NORCE.

## References

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’19, pp. 2623–2631, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450362016. doi: 10.1145/3292500.3330701. URL https://doi.org/10.1145/ 3292500.3330701.

Andrea L. Bertozzi and Arjuna Flenner. Difuse interface models on graphs for classification of high dimensional data. Multiscale Modeling & Simulation, 10(3):1090–1118, 2012. doi: 10.1137/11083109X. URL https://doi. org/10.1137/11083109X.

Malik Boudiaf, Romain Mueller, Ismail Ben Ayed, and Luca Bertinetto. Parameter-free online test-time adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022. URL https://openaccess.thecvf.com/content/CVPR2022/papers/Boudiaf\_Parameter-Free\_Online\_ Test-Time\_Adaptation\_CVPR\_2022\_paper.pdf.

Jeongwhan Choi, Seoyoung Hong, Noseong Park, and Sung-Bae Cho. GREAD: Graph neural reaction-difusion networks. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 5722–5747. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/ v202/choi23a.html.

Cristina Garcia-Cardona, Ekaterina Merkurjev, Andrea L. Bertozzi, Arjuna Flenner, and Allon G. Percus. Multiclass data segmentation using difuse interface methods on graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 36(8):1600–1613, 2014. doi: 10.1109/TPAMI.2014.2300478.

Johannes Gasteiger, Aleksandar Bojchevski, and Stephan Günnemann. Predict then propagate: Graph neural networks meet personalized pagerank. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=H1gL-2A9Ym.

Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Doina Precup and Yee Whye Teh (eds.), Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 1321–1330. PMLR, 06–11 Aug 2017. URL https: //proceedings.mlr.press/v70/guo17a.html.

Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips. cc/paper\_files/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf.

Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, and Bryan Hooi. Harnessing explanations: LLM-to-LM interpreter for enhanced text-attributed graph representation learning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=RXFVcynVe1.

Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 22118–22133. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/ 2020/file/fb60d411a5c5b72b2e7d3527cfc84fd0-Paper.pdf.

Qian Huang, Horace He, Abhay Singh, Ser-Nam Lim, and Austin Benson. Combining label propagation and simple models out-performs graph neural networks. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=8E1-f3VhX1o.

Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=SJU4ayYgl.

Qimai Li, Zhichao Han, and Xiao-Ming Wu. Deeper insights into graph convolutional networks for semi-supervised learning. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, pp. 3538–3545, 2018. URL https://doi.org/10.1609/aaai.v32i1.11604.

Wenxiao Li, Xue-Cheng Tai, and Jun Liu. Topology-guaranteed image segmentation: Enforcing connectivity, genus, and width constraints. SIAM Journal on Imaging Sciences, 19(2):1137–1173, 2026. doi: 10.1137/25M1765870. URL https://doi.org/10.1137/25M1765870.

Hao Liu, Jun Liu, Raymond H. Chan, and Xue-Cheng Tai. Double-well net for image segmentation. Multiscale Modeling & Simulation, 22(4):1449–1477, 2024. doi: 10.1137/24M1632103. URL https://doi.org/10.1137/ 24M1632103.

Jun Liu, Xue-Cheng Tai, and Shousheng Luo. Convex shape prior for deep neural convolution network based eye fundus images segmentation, 2020. URL https://arxiv.org/abs/2005.07476.

Jun Liu, Xiangyue Wang, and Xue-Cheng Tai. Deep convolutional neural networks with spatial regularization, volume and star-shape priors for image segmentation. Journal of Mathematical Imaging and Vision, 64(6):625–645, 2022. doi: 10.1007/s10851-022-01087-x.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. RoBERTa: A robustly optimized BERT pretraining approach, 2019. URL https://arxiv.org/abs/1907.11692.

Ekaterina Merkurjev, Tijana Kostić, and Andrea L. Bertozzi. An MBO scheme on graphs for classification and image processing. SIAM Journal on Imaging Sciences, 6(4):1903–1930, 2013. doi: 10.1137/120886935. URL https://doi.org/10.1137/120886935.

Péter Mernyei and Cătălina Cangea. Wiki-CS: A Wikipedia-based benchmark for graph neural networks, 2022. URL https://arxiv.org/abs/2007.02901.

Kenta Oono and Taiji Suzuki. Graph neural networks exponentially lose expressive power for node classification. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id= S1ldO2EFPr.

Oleg Platonov, Denis Kuznedelev, Michael Diskin, Artem Babenko, and Liudmila Prokhorenkova. A critical look at the evaluation of GNNs under heterophily: Are we really making progress? In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=tJbbQfw-5wv.

R. B. Potts. Some generalized order-disorder transformations. Mathematical Proceedings of the Cambridge Philosophical Society, 48(1):106–109, 1952. doi: 10.1017/S0305004100027419.

Xue-Cheng Tai, Hao Liu, and Raymond Chan. PottsMGNet: A mathematical explanation of encoder-decoder based neural networks. SIAM Journal on Imaging Sciences, 17(1):540–594, 2024. doi: 10.1137/23M1586355. URL https://doi.org/10.1137/23M1586355.

Yuelin Wang, Kai Yi, Xinliang Liu, Yu Guang Wang, and Shi Jin. ACMP: Allen-Cahn message passing with attractive and repulsive forces for graph neural networks. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=4fZc\_79Lrqs.

Hao Yan, Chaozhuo Li, Ruosong Long, Chao Yan, Jianan Zhao, Wenwen Zhuang, Jun Yin, Peiyan Zhang, Weihao Han, Hao Sun, Weiwei Deng, Qi Zhang, Lichao Sun, Xing Xie, and Senzhang Wang. A comprehensive study on text-attributed graphs: Benchmarking and rethinking. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 17238–17264. Curran Associates, Inc., 2023. doi: 10.52202/075280-0755. URL https://proceedings.neurips.cc/paper\_ files/paper/2023/file/37d00f567a18b478065f1a91b95622a0-Paper-Datasets\_and\_Benchmarks.pdf.

Yiming Yang, Jun Liu, and Wei Wan. Regularizing softmax with graph similarity for enhanced node classification in semisupervised settings. International Journal of Intelligent Systems, 2025(1):8861477, 2025. doi: 10.1155/int 8861477. URL https://onlinelibrary.wiley.com/doi/abs/10.1155/int/8861477.

A. L. Yuille and Anand Rangarajan. The concave-convex procedure. Neural Computation, 15(4):915–936, 2003. doi: 10.1162/08997660360581958.

Meiqi Zhu, Xiao Wang, Chuan Shi, Houye Ji, and Peng Cui. Interpreting and unifying graph neural networks with an optimization framework. In Proceedings of the Web Conference 2021, pp. 1215–1226, 2021. doi: 10.1145/3442381.3449953. URL https://doi.org/10.1145/3442381.3449953.

## Appendix overview

This appendix provides the theoretical derivations, experimental details, and additional analyses supporting the main paper.

Theory and derivations (Appendices A to D). We derive the graph energy underlying PtS, its decomposition into Dirichlet and Gini terms, and the categorical sharpening step. We then connect the construction to Potts and LAME and analyze properties of sharpening and repeated propagation, including class-order preservation, local Gini descent, propagation collapse, and the efect of sharpening at increasing depth.

Experimental setup (Appendices E to H). We provide full details on the datasets, frozen base models, hyperparameter search spaces, and selected hyperparameters used throughout the experiments.

Additional comparisons (Appendices I to M). We compare PtS with the external post-hoc baselines LAME-Graph and Graph-TV, evaluate stronger GCN and GraphSAGE backbones, study how the sharpening step can be incorporated into Correct & Smooth, and report results across the full Gaussian corruption ladder.

Ablations and mechanism (Appendices N to Q). We examine row-mass preservation, propagation depth, the evolution of the Dirichlet and Gini components, performance across nodes with diferent local homophily, degree, and prediction confidence

Calibration and computational cost (Appendices R to S). Finally, we report runtime and computational overhead and evaluate calibration before and after temperature scaling.

## A From the continuous Potts energy to the graph energy

We start from an entropy-regularized Potts formulation and transfer it from a continuous domain to a graph. The derivations show how predictions from a frozen classifier become a KL term, and how the Potts interaction decomposes into a Dirichlet and Gini term. This decomposition motivates combining propagation between nodes with local sharpening of class distributions.

Let

$$
{ \Delta } ^ { C - 1 } = \left\{ u \in \mathbb { R } ^ { C } : u _ { c } \geq 0 , \sum _ { c = 1 } ^ { C } u _ { c } = 1 \right\}
$$

denote the probability simplex. In the multiphase formulation, an assignment field $v ( x ) \in { \Delta } ^ { C - 1 }$ associates each position $x \in \Omega$ with a soft class distribution.

Our motivation comes from the entropy-regularized multiphase formulation in Soft Threshold Dynamics (Liu et al., 2022) and the related two-phase formulation in PottsMGNet (Tai et al., 2024). With the perimeter-scaling constants absorbed into λ, we write the continuous energy as

$$
\begin{array} { c } { { \displaystyle \operatorname* { m i n } _ { v : \Omega \to \Delta ^ { C - 1 } } \left[ \int _ { \Omega } v ( x ) ^ { \top } g ( x ) d x + \epsilon \int _ { \Omega } \sum _ { c = 1 } ^ { C } v _ { c } ( x ) \ln v _ { c } ( x ) d x \right. } } \\ { { \displaystyle \left. + \frac { \lambda } { 2 } \int _ { \Omega } \int _ { \Omega } G _ { \rho } ( x - y ) \left( 1 - v ( x ) ^ { \top } v ( y ) \right) d y d x \right] } } \end{array}\tag{6}
$$

Here $\epsilon > 0 , \lambda \geq 0 .$ , and $G _ { \rho }$ is a Gaussian kernel. The last term is an approximation of the Potts boundary penalty. The first term assigns a cost $g ( x )$ to each class at each position, and the entropy term regularizes the assignments within the simplex, with the convention 0 log $0 = 0$

## A.1 Graph analogue

We replace spatial positions with graph nodes, integrals with sums, and the Gaussian kernel with a symmetric nonnegative afinity matrix.

$$
\begin{array} { r } { W \in \mathbb { R } ^ { N \times N } \qquad W = W ^ { \top } \qquad W _ { i j } \geq 0 . } \end{array}
$$

The graph analogue of Equation (6) is

$$
\operatorname* { m i n } _ { \{ u _ { i } \in \Delta ^ { C - 1 } \} _ { i = 1 } ^ { N } } \left[ \sum _ { i } u _ { i } ^ { \top } g _ { i } + \epsilon \sum _ { i , c } u _ { i c } \ln u _ { i c } + \frac { \lambda } { 2 } \sum _ { i , j } W _ { i j } \left( 1 - u _ { i } ^ { \top } u _ { j } \right) \right] .\tag{7}
$$

For one-hot assignments $u _ { i } = e _ { y }$

$$
1 - u _ { i } ^ { \top } u _ { j } = \mathbf { 1 } [ y _ { i } \neq y _ { j } ] .
$$

and the interaction reduces to a weighted Potts regularization that penalizes disagreement between connected nodes. We obtain the assignment cost from the frozen classifier. Let $q _ { i } \in \Delta ^ { C - 1 }$ denote row i of $Q .$ , the predicted class distribution at node i, with $q _ { i c } > 0$ for every class, and choose $g _ { i c } = - \epsilon \log q _ { i c }$

Then the first two terms in (7) become

$$
\begin{array} { r } { \displaystyle \sum _ { i } u _ { i } ^ { \top } g _ { i } + \epsilon \sum _ { i , c } u _ { i c } \log u _ { i c } = \epsilon \sum _ { i , c } u _ { i c } \log \frac { u _ { i c } } { q _ { i c } } } \\ { = \epsilon \sum _ { i } \mathrm { K L } ( u _ { i } \| q _ { i } ) . } \end{array}
$$

After dividing by ϵ and defining $\gamma = \lambda / \epsilon .$ , we get

$$
E _ { \gamma } ( \boldsymbol { U } ; \boldsymbol { Q } ) = \sum _ { i } \mathrm { K L } ( u _ { i } \| \boldsymbol { q } _ { i } ) + \frac { \gamma } { 2 } \sum _ { i , j } W _ { i j } \left( 1 - u _ { i } ^ { \top } u _ { j } \right) .\tag{8}
$$

The energy therefore combines anchoring to the frozen classifier’s predictions with a relaxed Potts interaction between the nodes.

Gini + Dirichlet For a symmetric W, we define $d _ { i } = \textstyle \sum _ { j } W _ { i j }$ . Using the polarization identity

$$
\begin{array} { r } { \| u _ { i } - u _ { j } \| ^ { 2 } = \| u _ { i } \| ^ { 2 } + \| u _ { j } \| ^ { 2 } - 2 u _ { i } ^ { \top } u _ { j } } \end{array}
$$

and the symmetry of $W$ , we get

$$
\frac { 1 } { 2 } \sum _ { i , j } W _ { i j } \left( 1 - u _ { i } ^ { \top } u _ { j } \right) = \underbrace { \frac { 1 } { 4 } \sum _ { i , j } W _ { i j } \| u _ { i } - u _ { j } \| ^ { 2 } } _ { E _ { \mathrm { D } } \left( U \right) } + \underbrace { \frac { 1 } { 2 } \sum _ { i } d _ { i } \left( 1 - \| u _ { i } \| ^ { 2 } \right) } _ { E _ { \mathrm { G } } \left( U \right) } .
$$

The Dirichlet term $E _ { \mathrm { D } }$ penalizes diferences between the class distributions of connected nodes, and is zero when the class distributions are identical on all connected nodes. The Gini term $E _ { \mathrm { G } }$ penalizes distributions that spread mass over several classes, and is zero when all node distributions are one-hot.

The two terms describe diferent properties. Two neighbouring nodes can carry identical but uncertain class distributions: their connecting edge is then not penalized by $E _ { \mathrm { D } }$ , while both nodes still contribute to $E _ { \mathrm { G } }$

## B Derivation of the sharpening step

Here, we derive the sharpening step from a local energy that combines the KL distance to the propagated distribution with a Gini term. The derivation shows why one fixed-point step yields the reaction used in PtS. When $\eta = 0$ , the reaction becomes an identity. The decomposition above gives

$$
E _ { \gamma } ( U ; Q ) = \sum _ { i } \mathrm { K L } ( u _ { i } | | q _ { i } ) + \gamma E _ { \mathrm { D } } ( U ) + \gamma E _ { \mathrm { G } } ( U ) .
$$

which motivates treating graph smoothing and local categorical sharpening as separate operations.

PtS uses the probability-space propagation defined in Section 2.1,

$$
U ^ { ( 0 ) } = Q , \qquad \widetilde { U } ^ { ( k + 1 ) } = \alpha Q + ( 1 - \alpha ) S U ^ { ( k ) } .
$$

For a positive propagated row $h _ { i } = \widetilde { U } _ { i } ^ { ( k + 1 ) }$ , let $m _ { i } = \mathbf { 1 } ^ { \top } h _ { i }$ and $p _ { i } = h _ { i } / m _ { i }$ . Since the rows of $S$ do not necessarily sum to one, this normalization is what ensures $p _ { i } \in \Delta ^ { C - 1 }$

We hold $p$ fixed and use the assignment cost $g _ { c } = - \log p _ { c }$ . Combining this cost with the entropy regularization gives

$$
E ( u ; p ) = \sum _ { c } u _ { c } \log \frac { u _ { c } } { p _ { c } } + \frac { \eta } { 2 } \big ( 1 - \| u \| ^ { 2 } \big ) , \qquad u \in \Delta ^ { C - 1 } .
$$

The first term keeps the update close to the propagated distribution p and is zero when $u = p ,$ while the second term penalizes indecision and is zero when u is one-hot.

We use one global sharpening strength $\eta \geq 0$ for all nodes. This replaces the degree-dependent weighting in the Gini term. To handle the constraint $\textstyle \sum _ { c } u _ { c } = 1$ , we introduce a Lagrange multiplier ν:

$$
L ( u , \nu ; p ) = E ( u ; p ) + \nu \left( \sum _ { c } u _ { c } - 1 \right) .
$$

Stationarity in the simplex gives

$$
\frac { \partial \mathcal { L } } { \partial u _ { c } } = \log \frac { u _ { c } } { p _ { c } } + 1 - \eta u _ { c } + \nu = 0 .
$$

With $p$ held fixed, the corresponding constrained gradient flow is

$$
\dot { u } _ { c } ( t ) = - \log \frac { u _ { c } ( t ) } { p _ { c } } - 1 + \eta u _ { c } ( t ) - \nu ( t ) , \qquad u ( 0 ) = p ,
$$

where $\dot { u } _ { c } ( t )$ denotes the time derivative of $u _ { c } ( t )$ . The multiplier $\nu ( t )$ is chosen so that $\textstyle \sum _ { c = 1 } ^ { C } { \dot { u } } _ { c } ( t ) = 0$ . Since $\textstyle \sum _ { c = 1 } ^ { C } p _ { c } = 1$ , this preserves $\textstyle \sum _ { c = 1 } ^ { C } u _ { c } ( t ) = 1$

An implicit Euler step of size $\tau > 0$ gives

$$
\frac { u _ { c } - p _ { c } } { \tau } + \log { \frac { u _ { c } } { p _ { c } } } + 1 - \eta u _ { c } + \nu = 0 .
$$

Since the new unknown distribution u appears both in the time-step term and in the gradient, we use a fixed-point iteration similar to the one in PottsMGNet (Tai et al., 2024, Section 5.2). Starting from $v ^ { ( 0 ) } = p ,$ we evaluate the non-logarithmic terms at $v ^ { k }$ and solve for the next iterate:

$$
v ^ { ( k + 1 ) } = \mathrm { s o f t m a x } \left( \log p + \eta v ^ { ( k ) } - \frac { v ^ { ( k ) } - p } { \tau } \right) .
$$

We use one inner iteration. Since $v ^ { ( 0 ) } = p ,$ , the last term is zero and the update becomes

$$
p _ { c } ^ { * } = \frac { p _ { c } \exp ( \eta p _ { c } ) } { \sum _ { b } p _ { b } \exp ( \eta p _ { b } ) } , \qquad p ^ { * } = \mathrm { s o f t m a x } ( \log p + \eta p ) .
$$

Because the fixed-point iteration is initialized at $v ^ { ( 0 ) } = p ,$ the time-step term vanishes in the first iteration, so the step size $\tau$ does not appear in the implemented reaction. We use this single iteration rather than solving to convergence, and restore the row mass afterwards:

$$
R _ { \eta } ( h _ { i } ) = m _ { i } \mathrm { s o f t m a x } ( \log p _ { i } + \eta p _ { i } ) , \qquad { \bf 1 } ^ { \top } R _ { \eta } ( h _ { i } ) = m _ { i } .
$$

The restored mass determines the row’s total contribution in the next propagation step. With $\eta = 0$ we get $R _ { 0 } ( h _ { i } ) = h _ { i } ,$ so that PtS reduces to PPR-Prob with the same $( \alpha , K )$ . The final class distributions are obtained by row-normalizing $U ^ { ( K ) }$

## C Relation to Potts and LAME

This part shows the connection between the graph energy, the coupled fixed-point update, and LAME-Graph. In the coupled update, the neighbourhood contribution is in the softmax, while the original predictions are kept as the reference. PtS instead uses separate steps for propagation and sharpening, with the propagated distribution as the reference in the local sharpening step.

For a symmetric W, every stationary point of the energy in Equation (8) must satisfy

$$
u _ { i } = \mathrm { s o f t m a x } \left( \log q _ { i } + \gamma ( W U ) _ { i } \right) .\tag{9}
$$

Here the neighbourhood interaction sits inside the softmax, while $q _ { i }$ is kept as the reference. PtS instead propagates first and then uses the propagated distribution $p _ { i }$ as the anchor for the reaction. The Potts decomposition motivates this construction, but it does not imply that the PtS iteration reduces the global energy. We evaluate the coupled iteration (9) with $B = \gamma S$ under the same protocol as everything else, reported as LAME-Graph in Tables 8 and 9.

Relationship to LAME Once the constant term $\textstyle { \frac { \gamma } { 2 } } \sum _ { i , j } W _ { i j }$ is removed from Equation (8), the energy has the same KL-plus-afinity form as the objective of LAME (Boudiaf et al., 2022), whose update can be written as

$$
u _ { i } ^ { ( t + 1 ) } = \mathrm { s o f t m a x } \big ( \log q _ { i } + ( B u ^ { ( t ) } ) _ { i } \big ) .
$$

In the original LAME formulation the afinity matrix B is constructed from pretrained feature representations, using a k-nearest-neighbour afinity as well as linear and Gaussian kernels. Setting $B = \gamma W$ recovers the iteration in Equation (9), which connects the coupled update to LAME with a graph-derived afinity; this is the LAME-Graph baseline. PtS uses a diferent update: propagation combines the current state with the original predictions, and a separate mass-preserving reaction then sharpens the propagated distribution $p _ { i }$

## D Properties of sharpening

Here we examine both the properties of a single sharpening step and its behaviour under repeated propagation. We show that sharpening keeps the class ordering and decreases local Gini. The propagation analysis shows why the nodes’ predictions can collapse without restart, while the example in Figure 5 shows that even restart is not always suficient to keep the correct predictions correct.

## D.1 Pure propagation collapses

Assume $\alpha = \eta = 0$ and that the graph is connected and undirected, with a self-loop at every node. Let $Q > 0$ have row sums equal to one. Propagation without restart and without sharpening is then

$$
U ^ { ( K ) } = S ^ { K } Q .\tag{10}
$$

From Oono & Suzuki (2020, Proposition 1), S has a single eigenvalue 1 while all other eigenvalues have absolute value less than 1, the corresponding unit eigenvector $\phi \propto \widetilde { D } ^ { 1 / 2 } \mathbf { 1 }$ is strictly positive. Write $v = \phi ,$ , with $v _ { j }$ denoting its entry at node $j .$

Since S is real and symmetric, we can decompose it into its eigenvectors, where N is the number of nodes and $\lambda _ { 1 } = 1$

$$
\boldsymbol { S } ^ { K } = \lambda _ { 1 } ^ { K } \boldsymbol { \phi } \boldsymbol { \phi } ^ { \top } + \sum _ { r = 2 } ^ { N } \lambda _ { r } ^ { K } \boldsymbol { \phi } _ { r } \boldsymbol { \phi } _ { r } ^ { \top } .
$$

Let $K  \infty$ . Since $| \lambda _ { r } | < 1$ for $r \geq 2$ then $\lambda _ { r } ^ { K }  0$ leaving us with

$$
S ^ { K } \to \phi \phi ^ { \intercal }\tag{11}
$$

Let $p _ { i } ^ { ( K ) }$ be the final normalised prediction at node i.

$$
\begin{array} { r l } { p _ { i } ^ { ( K ) } = \frac { U _ { i } ^ { ( K ) } } { \sum _ { c } U _ { i c } ^ { ( K ) } } \qquad } & { B y \ d e f i n i t i o n } \\ { \quad } & { = \frac { ( S ^ { K } Q ) _ { i } } { \sum _ { c } \left( S ^ { K } Q \right) _ { i c } } \qquad } \\ { \quad } & { \to \ \frac { ( \phi \phi ^ { T } Q ) _ { i } } { \sum _ { c } ( \phi \phi ^ { T } Q ) _ { i c } } \qquad } \\ { \quad } & { = \frac { \phi _ { i } \sum _ { j } v _ { j } Q _ { j } } { \phi _ { i } \sum _ { j } v _ { j } \underbrace { C _ { j } } _ { \textcircled { i } } \sum _ { c } \phi _ { j c } } } \end{array} \qquad \mathrm { l o } \qquad 
$$

Which result in

$$
\operatorname* { l i m } _ { K  \infty } p _ { i } ^ { ( K ) } = \frac { \sum _ { j } v _ { j } Q _ { j } } { \sum _ { j } v _ { j } } .
$$

The limit does not depend on i: it is a weighted average of the initial predictions, so every node converges to the same normalised class distribution. Pure probability propagation therefore collapses, as observed empirically in Figure 2.

The same argument applies to APPNP at $\alpha = 0$ , where the propagated state is the logit matrix Z rather than $Q$ Here $S ^ { K } Z  v v ^ { \top } Z$ , so node i receives the logit row $\phi _ { i } c$ with the common vector $\begin{array} { r } { c = \sum _ { j } \phi _ { j } Z _ { j } } \end{array}$ . Since $\phi _ { i } > 0$ , the softmax of $\phi _ { i } c$ has the same maximizing class at every node: the predicted distributions need not coincide, but the predicted labels all collapse onto a single class.

With $\alpha \in ( 0 , 1 ]$ With restart, the iteration converges to the PPNP solution instead (Gasteiger et al., 2019),

$$
\begin{array} { r } { U ^ { \infty } = \alpha ( I _ { n } - ( 1 - \alpha ) S ) ^ { - 1 } Q , } \end{array}
$$

which depends on i through $Q ,$ so restart prevents collapse. It does so by reinjecting $Q$ at every step, which is what makes the quality of $Q$ decisive, Proposition 2 shows that this is not always enough.

## D.2 Restart alone does not preserve a minority community

Proposition 2. Let G consist of two 9-node cliques A and B joined by a perfect matching, so that with added self-loops every node draws 90% of its propagation weight from its own community. Let the true class be 1 on A and 2 on B, with frozen predictions $Q _ { i } = ( 0 . 9 , 0 . 1 )$ for $i \in \mathcal { A }$ and $Q _ { j } = ( 0 . 4 , 0 . 6 )$ for $j \in B$ , and let $\alpha = 0 . 1$ . Then APPNP and PPR-Prob misclassify every node of B for every depth $K \geq 3 .$ , while PtS with $\eta = 4$ classifies every node correctly at every depth.

![](images/a71c353105bdfa68b3dc13cca9d6d8e3bc9e521cde41a4c15dd82527f131d71d.jpg)  
Figure 5: Class probabilities with increasing depth. All methods use the same graph, initial predictions, and $\alpha = 0 . 1$ PtS uses $\eta = 4$ . The numbers give the correct-class probability in group B. Self-loops are included in the updates but omitted from the drawing.

Each node is joined to one distinct node in the other group, so after adding self-loops every node has ten connections: nine within its community (eight clique neighbours and itself) and one across. All degrees are equal, so $\widetilde { D } = 1 0 I$ and $S = \widetilde { A } / 1 0$ is both symmetric and row-stochastic, with weight 0.9 inside the community and 0.1 across it. Propagation therefore preserves unit row sums, and so does the reaction, so all three iterations stay on the simplex. By symmetry, all nodes of a community carry the same state at every depth, and it sufices to track one node per community.

PPR-Prob Let $a _ { k }$ and $b _ { k }$ be the class-1 probabilities in A and B. The update (1) gives

$$
a _ { k + 1 } = 0 . 0 9 + 0 . 9 ( 0 . 9 a _ { k } + 0 . 1 b _ { k } ) , \qquad b _ { k + 1 } = 0 . 0 4 + 0 . 9 ( 0 . 1 a _ { k } + 0 . 9 b _ { k } ) .
$$

The sum $\Sigma _ { k } = a _ { k } + b _ { k }$ satisfies $\Sigma _ { k + 1 } = 0 . 1 3 + 0 . 9 \Sigma _ { k }$ with $\Sigma _ { 0 } = 1 . 3$ , hence $\Sigma _ { k } = 1 . 3$ for all k. The diference $\Delta _ { k } = a _ { k } - b _ { k }$ satisfies $\Delta _ { k + 1 } = 0 . 0 5 + 0 . 7 2 \Delta _ { k }$ with $\Delta _ { 0 } = 0 . 5$ , hence

$$
\begin{array} { r } { \Delta _ { k } = \frac { 5 } { 2 8 } + \left( 0 . 5 - \frac { 5 } { 2 8 } \right) 0 . 7 2 ^ { k } , } \end{array}
$$

which decreases monotonically to $5 / 2 8 \approx 0 . 1 7 9$ . A node of B is misclassified exactly when $b _ { k } = ( 1 . 3 - \Delta _ { k } ) / 2 > 1 / 2$ that is when $\Delta _ { k } < 0 . 3$ . Since $\Delta _ { 2 } = 0 . 3 4 5 2$ and $\Delta _ { 3 } = 0 . 2 9 8 5 4$ , the nodes of B are classified correctly for $K \leq 2$ and misclassified for every $K \geq 3 .$

APPNP With two classes, the softmax depends only on the logit diference, so let $x _ { k }$ and $y _ { k }$ be the class-1 minus class-2 logit diferences in A and B, starting from $x _ { 0 } = \log 9$ and $y _ { 0 } = \log { \frac { 2 } { 3 } }$ . APPNP applies the same afine recursion to the logits, so $\Sigma _ { k } = x _ { k } + y _ { k }$ satisfies $\Sigma _ { k + 1 } = 0 . 1 \Sigma _ { 0 } + 0 . 9 \Sigma _ { k }$ and stays at $\Sigma _ { 0 } = \log 6 _ { \mathrm { : } }$ , while $\Delta _ { k } = x _ { k } - y _ { k }$ satisfies $\Delta _ { k + 1 } = 0 . 1 \Delta _ { 0 } + 0 . 7 2 \Delta _ { k }$ with $\Delta _ { 0 } =$ log 13.5 and decreases monotonically to $0 . 1 \Delta _ { 0 } / 0 . 2 8 \approx 0 . 9 3 0$ . A node of B is misclassified exactly when $y _ { k } = ( \log 6 - \Delta _ { k } ) / 2 > 0$ , that is when $\Delta _ { k } < \log 6 \approx 1 . 7 9 1 8$ . Since $\Delta _ { 2 } \approx 1 . 7 9 6 9$ and $\Delta _ { 3 } \approx 1 . 5 5 4 1$ , the nodes of B are again classified correctly for $K \leq 2$ and misclassified for every $K \geq 3$

PtS We claim that for $\eta = 4$ every node keeps at least probability 0.6 on its correct class at every depth. This holds at $k = 0$ , where the correct-class probabilities are 0.9 on A and 0.6 on $B .$ Assume it holds at step k and consider a node i with correct class $y .$ Since the rows are distributions, the propagated value satisfies

$$
\widetilde { U } _ { i y } ^ { ( k + 1 ) } = 0 . 1 Q _ { i y } + 0 . 9 \sum _ { j } S _ { i j } U _ { j y } ^ { ( k ) } ~ \ge ~ 0 . 1 \cdot 0 . 6 + 0 . 9 \left( 0 . 9 \cdot 0 . 6 + 0 . 1 \cdot 0 \right) = 0 . 5 4 6 ,
$$

using $Q _ { i y } \geq 0 . 6$ , the induction hypothesis inside the community, and non-negativity across it. In the two-class case the sharpening map satisfies

$$
\frac { r _ { \eta } ( p ) _ { y } } { 1 - r _ { \eta } ( p ) _ { y } } = \frac { p _ { y } } { 1 - p _ { y } } e ^ { \eta ( 2 p _ { y } - 1 ) } ,
$$

which is increasing in $p _ { y }$ , so $\widetilde { U } _ { i y } ^ { ( k + 1 ) } \geq 0 . 5 4 6$ implies

$$
U _ { i y } ^ { ( k + 1 ) } \ \ge \ \frac { 0 . 5 4 6 e ^ { 4 ( 0 . 5 4 6 ) } } { 0 . 5 4 6 e ^ { 4 ( 0 . 5 4 6 ) } + 0 . 4 5 4 e ^ { 4 ( 0 . 4 5 4 ) } } = 0 . 6 3 4 7 \ldots > 0 . 6 .
$$

The invariant is restored, so by induction every node keeps more than probability 0.6 on its correct class, and hence is classified correctly, at every depth. □

## D.3 Sharpening preserves the class ordering

From the update, we define

$$
p _ { c } ^ { * } = \frac { p _ { c } e ^ { \eta p _ { c } } } { \sum _ { b } p _ { b } e ^ { \eta p _ { b } } } ,\tag{12}
$$

where the entries of $p ^ { * }$ are positive and sum to one. For any two classes c and $b ,$

$$
\frac { p _ { c } ^ { * } } { p _ { b } ^ { * } } = \frac { p _ { c } } { p _ { b } } e ^ { \eta ( p _ { c } - p _ { b } ) } .\tag{13}
$$

Since $\eta \geq 0 ,$ , the ratio is at least ${ p _ { c } } / { p _ { b } }$ whenever $p _ { c } \geq p _ { b }$ , so sharpening reinforces a node’s existing class preference and in particular leaves its predicted class unchanged.

The same identity quantifies how much confidence a single step can restore after propagation has eroded it.

Lemma 1 (Recovering a confidence level). Let $p \in \Delta ^ { C - 1 }$ with $p > 0$ have a unique leading class c and margin $\mu = p _ { c } - \operatorname* { m a x } _ { b \neq c } p _ { b } > 0$ . Then

$$
p _ { c } ^ { * } \ge \frac { 1 } { 1 + ( C - 1 ) e ^ { - \eta \mu } } ,
$$

so $f o r$ any target confidence $\beta \in ( 0 , 1 )$ we have $p _ { c } ^ { * } \ge \beta$ as soon as $\eta ~ \geq ~ \mu ^ { - 1 } \log \bigl ( ( C - 1 ) \beta / ( 1 - \beta ) \bigr )$ . In particular $p _ { c } ^ { * } \to 1$ as $\eta  \infty$

Proof. For $b \neq c , ( 1 3 )$ gives $p _ { b } ^ { * } / p _ { c } ^ { * } = ( p _ { b } / p _ { c } ) e ^ { - \eta ( p _ { c } - p _ { b } ) } \leq e ^ { - \eta \mu }$ , since $p _ { b } \leq p _ { c }$ and $p _ { c } - p _ { b } \ge \mu$ . Summing over the $C - 1$ competing classes,

$$
\frac { 1 } { p _ { c } ^ { * } } = 1 + \sum _ { b \neq c } \frac { p _ { b } ^ { * } } { p _ { c } ^ { * } } \leq 1 + ( C - 1 ) e ^ { - \eta \mu } ,
$$

which is the stated bound. It is at least $\beta$ precisely when $( C - 1 ) e ^ { - \eta \mu } \leq ( 1 - \beta ) / \beta$ , that is when $\eta \mu \geq \log ( ( C -$ $1 ) \beta / ( 1 - \beta ) )$ □

A node whose leading class survives propagation with margin $\mu$ can therefore be returned to any confidence level by a large enough $\eta ,$ which is the mechanism behind Proposition 2. The bound also shows the cost: the same $\eta$ is applied to every node, including nodes whose leading class is wrong.

## D.4 Proof of Proposition 1

Fix $p \in \Delta ^ { C - 1 }$ with $p > 0$ and $\eta \geq 0$ , and recall the local energy (5),

$$
E _ { \eta } ( u ; p ) = \mathrm { K L } ( u \| p ) + \frac { \eta } { 2 } \bigl ( 1 - \| u \| ^ { 2 } \bigr ) , \qquad u \in \Delta ^ { C - 1 } .
$$

The KL term is convex in u and the Gini term is concave, so $E _ { \eta }$ is a diference of convex functions. Concavity of $\begin{array} { r } { u \mapsto - \frac { \eta } { 2 } \Vert u \Vert ^ { 2 } } \end{array}$ gives the linear upper bound

$$
- \frac { \eta } { 2 } \| u \| ^ { 2 } \leq - \frac { \eta } { 2 } \| p \| ^ { 2 } - \eta p ^ { \top } ( u - p ) ,
$$

with equality at $u = p$ . Adding $\begin{array} { r } { \mathrm { K L } ( u \| p ) + \frac { \eta } { 2 } } \end{array}$ to both sides defines a majorizer

$$
M ( u ) = \operatorname { K L } ( u \| p ) - \eta p ^ { \top } u + \mathrm { c o n s t } , \qquad E _ { \eta } ( u ; p ) \leq M ( u ) , \qquad E _ { \eta } ( p ; p ) = M ( p ) .
$$

By the Gibbs variational principle, M is minimized over the simplex at $u = \mathrm { s o f t m a x } ( \log p + \eta p ) = r _ { \eta } ( p ) = p ^ { * }$ , which is exactly the sharpening step (3). Therefore

$$
E _ { \eta } ( p ^ { * } ; p ) ~ \le ~ M ( p ^ { * } ) ~ \le ~ M ( p ) ~ = ~ E _ { \eta } ( p ; p ) ,
$$

which proves the descent claim. Writing it out, $\begin{array} { r } { \mathrm { K L } ( p ^ { * } \| p ) + \frac { \eta } { 2 } ( 1 - \| p ^ { * } \| ^ { 2 } ) \le \frac { \eta } { 2 } ( 1 - \| p \| ^ { 2 } ) } \end{array}$ , that is

$$
\frac \eta 2 \big ( \| p ^ { * } \| ^ { 2 } - \| p \| ^ { 2 } \big ) \ \ge \ \mathrm { K L } ( p ^ { * } \| p ) \ \ge \ 0 ,
$$

so for $\eta > 0$ the Gini impurity $1 - \| p \| ^ { 2 }$ does not increase, and it strictly decreases unless $p ^ { * } = p .$ . The ordering claim is Appendix D.3. The same majorization argument applied at an arbitrary iterate v instead of p shows that the full fixed-point iteration $v \mapsto \operatorname { s o f t m a x } ( \log p + \eta v )$ is a concave–convex procedure, so every inner iteration, not only the first, decreases $E _ { \eta } ( \cdot ; p )$ □

## E Datasets

Table 4 summarizes the nine graphs in the main board and the heterophilic control datasets. The datasets vary in nodes, edges, homophily, and classes. The control datasets are reported separately to show how the methods act when neighbouring nodes often belong to separate classes.

Table 4: Dataset statistics. Edges are the reported undirected-pair counts, and h denotes edge homophily.
<table><tr><td>Dataset</td><td>Nodes</td><td>Edges</td><td>Features</td><td>Classes</td><td>h</td></tr><tr><td>WikiCS</td><td>11,701</td><td>215,603</td><td>300</td><td>10</td><td>0.65</td></tr><tr><td>Cora-TAPE</td><td>2,708</td><td>5,278</td><td>768</td><td>7</td><td>0.81</td></tr><tr><td>PubMed-TAPE</td><td>19,717</td><td>44,324</td><td>768</td><td>3</td><td>0.80</td></tr><tr><td>TAPE-Arxiv23</td><td>46,198</td><td>38,863</td><td>300</td><td>40</td><td>0.64</td></tr><tr><td>ogbn-arxiv</td><td>169,343</td><td>1,157,799</td><td>128</td><td>40</td><td>0.65</td></tr><tr><td>ogbn-products</td><td>2,449,029</td><td>61,859,012</td><td>100</td><td>47</td><td>0.81</td></tr><tr><td>Ele-Photo</td><td>48,362</td><td>436,891</td><td>768</td><td>12</td><td>0.74</td></tr><tr><td>Ele-Computers</td><td>87,229</td><td>628,274</td><td>768</td><td>10</td><td>0.82</td></tr><tr><td>Books-History</td><td>41,551</td><td>251,590</td><td>768</td><td>12</td><td>0.64</td></tr><tr><td>Roman-Empire</td><td>22,662</td><td>32,927</td><td>300</td><td>18</td><td>0.05</td></tr><tr><td>Amazon-Ratings</td><td>24,492</td><td>93,050</td><td>300</td><td>5</td><td>0.38</td></tr></table>

## F Frozen backbones

Table 5 shows the architecture of the frozen base models that are used to produce the predictions. The MLP uses node features alone, while GCN and GraphSAGE use the graph structure as well. PtS is applied after these models have been trained and does not add any new trainable parameters to the base models. The MLP has architecture $d  2 5 6  2 5 6  C .$ , with BatchNorm, ReLU, and dropout (0.5) after each hidden linear layer, and is trained with Adam (learning rate 0.01, weight decay $5 \times 1 0 ^ { - 4 } )$ for at most 500 epochs with early-stopping patience 100.

Table 5: Number of parameters of each backbone, with the feature dimension D and the number of classes C. The last column gives the depth and width of the graph backbones.
<table><tr><td>Dataset</td><td>D</td><td>C</td><td>MLP</td><td>GCN</td><td>GraphSAGE</td><td>Conv. layers × width</td></tr><tr><td>WikiCS</td><td>300</td><td>10</td><td>146,442</td><td>80,138</td><td>159,498</td><td>2 × 256</td></tr><tr><td>Cora-TAPE</td><td>768</td><td>7</td><td>265,479</td><td>199,175</td><td>397,575</td><td>2 × 256</td></tr><tr><td>PubMed-TAPE</td><td>768</td><td>3</td><td>264,451</td><td>198,147</td><td>395,523</td><td>2 × 256</td></tr><tr><td>TAPE-Arxiv23</td><td>300</td><td>40</td><td>154,152</td><td>87,848</td><td>174,888</td><td>2 × 256</td></tr><tr><td>ogbn-arxiv</td><td>128</td><td>40</td><td>110,120</td><td>110,120</td><td>218,664</td><td>3 × 256</td></tr><tr><td>ogbn-products</td><td>100</td><td>47</td><td>104,751</td><td>36,015</td><td>71,215</td><td>3 × 128</td></tr><tr><td>Ele-Photo</td><td>768</td><td>12</td><td>266,764</td><td>200,460</td><td>400,140</td><td>2 × 256</td></tr><tr><td>Ele-Computers</td><td>768</td><td>10</td><td>266,250</td><td>199,946</td><td>399,114</td><td>2 × 256</td></tr><tr><td>Books-History</td><td>768</td><td>12</td><td>266,764</td><td>200,460</td><td>400,140</td><td>2 × 256</td></tr><tr><td>Roman-Empire</td><td>300</td><td>18</td><td>148,498</td><td>82,194</td><td>163,602</td><td>2 × 256</td></tr><tr><td>Amazon-Ratings</td><td>300</td><td>5</td><td>145,157</td><td>78,853</td><td>156,933</td><td>2 × 256</td></tr></table>

## G Hyperparameter search spaces

Table 6 shows the search spaces and budget for the propagation methods and variants of correct and smooth. We select hyperparameters using validation accuracy, with a separate Optuna trial per method. APPNP and PPR-Prob choose the restart weight α and number of propagation steps K, while PtS also searches over sharpening strength η. The search space also includes $\eta = 0 ,$ , so validation can choose to turn of the reaction.

Table 6: Search spaces and maximum numbers of hyperparameter evaluations per unit and severity. All methods use Optuna TPE except Graph-TV, which uses a grid search. The PtS search includes the exact sharpening of setting $\eta = 0$ alongside the logarithmic range. For hyperparameter tuning on ogbn-products, the upper limits are reduced to $K = 5 0$ and $T = 5 0$
<table><tr><td>Method</td><td>Search</td><td>Budget</td></tr><tr><td>APPNP / PPR</td><td> $\alpha \in [ 0 , 1 ] , K \in \{ 1 , \ldots , 1 0 0 \}$ </td><td>250</td></tr><tr><td>PtS</td><td> $\mathrm { S a m e } , \eta = 0 \ \mathrm { o r } \ \log _ { 1 0 } \eta \in [ - 2 , 2 . 4 0 8 ]$ </td><td>250</td></tr><tr><td>LAME-Graph</td><td> $T \in \{ 1 , \dots , 1 0 0 \} , \gamma = 0 \mathrm { o r } \log _ { 1 0 } \gamma \in [ - 2 , 2 . 4 0 8 ]$ </td><td>250</td></tr><tr><td>Graph-TV</td><td> $\lambda = 0 { \mathrm { ~ o r ~ } } \log _ { 1 0 } \lambda \in \{ - 3 , \ldots , 3 \}$  (25 log-spaced values),  $\varepsilon = 1$  fixed</td><td>≤ 26</td></tr><tr><td>C&amp;S</td><td> $\alpha _ { c } , \alpha _ { s } \in [ 0 , 1 ]$ </td><td>250</td></tr><tr><td>C&amp;S-PtS</td><td>Same, plus the PtS reaction search</td><td>250</td></tr></table>

## H Selected hyperparameters (MLP)

Table 7: Median selected values over the units of each (graph, severity), where K is the number of propagation steps, α the restart probability, η the sharpening strength, and “of” the percentage of units for which the search selected the exact sharpening of setting η = 0.
<table><tr><td>Dataset</td><td>σ</td><td>KAPPNP</td><td>KPPR KPtS</td><td>αAPPNP</td><td></td><td>αPtS</td><td>ηPtS</td><td>off (%)</td></tr><tr><td rowspan="5">WikiCS</td><td>0</td><td>2</td><td>2</td><td></td><td>0.11</td><td>0.09</td><td>4.11</td><td>7 8</td></tr><tr><td>0.5</td><td>2</td><td>2</td><td>4</td><td>0.08</td><td>0.06</td><td>6.40</td><td></td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>0.04</td><td>0.04</td><td>6.96</td><td></td></tr><tr><td>1.5</td><td>2</td><td>2</td><td>6</td><td>0.03</td><td>0.03</td><td>12.86</td><td></td></tr><tr><td>2</td><td>2</td><td>3</td><td>8 54</td><td>0.02</td><td>0.00</td><td>34.68 2.24</td><td></td></tr><tr><td rowspan="5">Cora-TAPE</td><td>0</td><td>3</td><td>6</td><td></td><td>0.06 0.13</td><td></td><td></td><td>0</td></tr><tr><td>0.5</td><td>3</td><td>6</td><td>38</td><td>0.05</td><td>0.12</td><td>1.78</td><td>6</td></tr><tr><td>1</td><td>4</td><td>6</td><td>34</td><td>0.03</td><td>0.10</td><td>1.42</td><td>7</td></tr><tr><td>1.5</td><td>4</td><td>7</td><td>34</td><td>0.03</td><td>0.08</td><td>1.35</td><td>4</td></tr><tr><td>2</td><td>5</td><td>6</td><td>38 46</td><td>0.02</td><td>0.06</td><td>1.46</td><td>7</td></tr><tr><td rowspan="5">PubMed-TAPE</td><td>0</td><td>30</td><td>28</td><td></td><td>0.32</td><td>0.31</td><td>1.51</td><td>20</td></tr><tr><td>0.5</td><td>26</td><td>24</td><td>52</td><td>0.24</td><td>0.24</td><td>1.33</td><td>30</td></tr><tr><td></td><td>4</td><td>7</td><td>15</td><td>0.11</td><td>0.15</td><td>0.82</td><td>21</td></tr><tr><td>1 1.5</td><td>4</td><td>7</td><td>22</td><td>0.05</td><td>0.13</td><td>1.13</td><td>10</td></tr><tr><td>2</td><td>5</td><td>8</td><td>24</td><td>0.03</td><td>0.10</td><td>1.61</td><td>6</td></tr><tr><td rowspan="5">TAPE-Arxiv23</td><td>0</td><td>26</td><td>4</td><td></td><td>0.35</td><td>0.30</td><td>0.91</td><td>27</td></tr><tr><td>0.5</td><td>7</td><td>8</td><td>58 26</td><td>0.26</td><td>0.21</td><td>0.17</td><td>39</td></tr><tr><td></td><td>4</td><td>8</td><td>16</td><td>0.12</td><td>0.09</td><td>0.00</td><td>53</td></tr><tr><td>1 1.5</td><td>4</td><td>10</td><td>14</td><td>0.04</td><td>0.05</td><td>0.15</td><td>32</td></tr><tr><td></td><td>3</td><td>9</td><td>29</td><td>0.08</td><td>0.05</td><td>0.63</td><td>8</td></tr><tr><td rowspan="5">ogbn-arxiv</td><td>2 0</td><td>2</td><td>2</td><td></td><td>0.12</td><td>0.26</td><td>6.80</td><td>0</td></tr><tr><td>0.5</td><td>2</td><td>2</td><td>3 3</td><td>0.08</td><td>0.23</td><td>15.96</td><td>0</td></tr><tr><td></td><td>2</td><td>3</td><td>7</td><td>0.03</td><td>0.20</td><td>6.70</td><td>0</td></tr><tr><td>1 1.5</td><td>2</td><td>2</td><td>3</td><td>0.03</td><td>0.06</td><td>6.77</td><td>0</td></tr><tr><td>2</td><td>1</td><td>2</td><td>2</td><td>0.03</td><td>0.02</td><td>0.00</td><td>56</td></tr><tr><td rowspan="5">ogbn-products</td><td>0</td><td>2</td><td></td><td></td><td>0.13</td><td>0.36</td><td>3.34</td><td>0</td></tr><tr><td>0.5</td><td>2</td><td>2 2</td><td>17 4</td><td>0.04</td><td>0.31</td><td>42.74</td><td>0</td></tr><tr><td></td><td>2</td><td>1</td><td>1</td><td>0.41</td><td>0.27</td><td>0.00</td><td>56</td></tr><tr><td>1 1.5</td><td>2</td><td>2</td><td>2</td><td>0.48</td><td>0.33</td><td>0.00</td><td>78</td></tr><tr><td>2</td><td>2</td><td>2</td><td>1</td><td>0.47</td><td>0.29</td><td>0.00</td><td>67</td></tr><tr><td rowspan="5">Ele-Photo</td><td>0</td><td>1</td><td></td><td></td><td>0.02</td><td>0.03</td><td>1.70</td><td>37</td></tr><tr><td>0.5</td><td>1</td><td>1 1</td><td>1 1</td><td>0.02</td><td>0.07</td><td>6.59</td><td>22</td></tr><tr><td></td><td>1</td><td>1</td><td></td><td>0.01</td><td>0.03</td><td>7.80</td><td>22</td></tr><tr><td>1 1.5</td><td></td><td>2</td><td>2 2</td><td>0.01</td><td>0.00</td><td>9.37</td><td></td></tr><tr><td>2</td><td>1</td><td>2</td><td>3</td><td>0.02</td><td>0.00</td><td>8.73</td><td>19 13</td></tr><tr><td rowspan="5">Ele-Computers</td><td>0</td><td>1</td><td></td><td></td><td></td><td>0.24</td><td></td><td>0</td></tr><tr><td>0.5</td><td>2</td><td>3 3</td><td>10 7</td><td>0.01 0.00</td><td>0.10</td><td>6.17 7.27</td><td>0</td></tr><tr><td>1</td><td>2</td><td>3</td><td></td><td>0.00</td><td>0.03</td><td>11.10</td><td>0</td></tr><tr><td>1.5</td><td>2</td><td>3</td><td>8 9</td><td>0.00</td><td>0.01</td><td>17.22</td><td>1</td></tr><tr><td>2</td><td>3</td><td>4</td><td>12</td><td>0.00</td><td>0.00</td><td>23.25</td><td>0</td></tr><tr><td rowspan="5">Books-History</td><td>0</td><td>3</td><td></td><td></td><td>0.42</td><td>0.42</td><td>3.37</td><td>20</td></tr><tr><td>0.5</td><td>1 1</td><td>1 1</td><td>41 15</td><td>0.30</td><td>0.34</td><td>3.26</td><td>13</td></tr><tr><td>1</td><td>1</td><td>2</td><td></td><td>0.18</td><td>0.24</td><td>3.87</td><td>7</td></tr><tr><td>1.5</td><td>2</td><td>3</td><td>5 8</td><td>0.10</td><td>0.16</td><td>3.65</td><td>7</td></tr><tr><td>2 0</td><td>2</td><td>3</td><td>14</td><td>0.06</td><td>0.09</td><td>4.77</td><td>2</td></tr><tr></table>

Across multiple graphs, hyperparameter tuning chose more propagation steps for PtS than for PPR-Prob (Table 7). The chosen sharpening strength varies across datasets and noise levels, and in some cases it is completely turned of. This suggests that sharpening is not always helpful. The depth also needs to be interpreted with the restart weight in mind, a large K can have a limited interpretation when α is close to 1. We calculate medians separately for each hyperparameter.

## I External post-hoc baselines

Both baselines refine the same frozen predictions $Q$ on the same observed graph and evaluation units as PtS. Validation labels are used only for hyperparameter selection. The adaptations, graph operators and selection procedures are specified below.

LAME-Graph. LAME (Boudiaf et al., 2022) refines a frozen classifier’s output using a KL fidelity term and an agreement term weighted by afinities computed from pretrained feature representations. We replace these afinities by the self-looped graph operator S and initialize $U ^ { ( 0 ) } = Q$ . The update is

$$
U ^ { ( t + 1 ) } = \mathrm { R o w S o f t m a x } \big ( \log Q + \gamma S U ^ { ( t ) } \big ) .
$$

The original predictions therefore remain the reference throughout the iteration, while the graph interaction enters inside the softmax. This is the coupled Potts update described in Appendix C.

We select the interaction strength $\gamma$ and a fixed iteration count T by validation accuracy using 250 Optuna TPE trials per unit, severity and noise draw. The search uses $T \in \{ 1 , \ldots , 1 0 0 \}$ and $\log _ { 1 0 } \gamma \in [ - 2 , 2 . 4 0 8 ]$ for positive strengths, together with an explicit identity option that returns $Q$ exactly. We run the selected number of iterations rather than using the original energy-based stopping rule.

Graph-TV. We adapt the TV-regularized softmax of Yang et al. (2025) to frozen predictions by minimizing

$$
\sum _ { i } { \mathrm { K L } } ( u _ { i } \| q _ { i } ) + \lambda { \mathrm { T V } } _ { G } ( U ) , \qquad u _ { i } \in \Delta ^ { C - 1 } .
$$

The KL term keeps predictions close to $Q { \mathrm { . } }$ , while TV penalizes disagreement across the graph. We use their classwise neighbourhood TV with loopless, degree-normalised graph weights, setting the input to log Q and $\varepsilon = 1$ . Setting $\lambda = 0$ returns Q.

We select λ by validation accuracy from at most 26 grid candidates (Table 6). Each dual solve uses at most 3000 iterations, and only candidates with relative primal–dual gap at most $\mathrm { 1 0 ^ { - 3 } }$ are eligible for selection. Graph-TV is evaluated at $\sigma \in \{ 0 , 2 \}$ on eight graphs, excluding ogbn-products because of memory requirements

Table 8: Comparison of post-hoc graph refinement methods at $\sigma = 2$ , including the LAME-Graph and the Graph-TV methods.
<table><tr><td>Dataset</td><td> $Q$ </td><td>APPNP</td><td>PPR-Prob</td><td>LAME-Graph</td><td>Graph-TV</td><td>PtS</td></tr><tr><td>WikiCS</td><td>46.36</td><td>67.97</td><td>68.98</td><td>67.80</td><td>65.10</td><td>72.15</td></tr><tr><td>Cora-TAPE</td><td>43.82</td><td>68.16</td><td>70.47</td><td>63.94</td><td>68.72</td><td>72.30</td></tr><tr><td>PubMed-TAPE</td><td>64.56</td><td>78.87</td><td>80.22</td><td>73.87</td><td>79.64</td><td>80.73</td></tr><tr><td>TAPE-Arxiv23</td><td>28.56</td><td>35.63</td><td>38.26</td><td>35.26</td><td>36.54</td><td>38.91</td></tr><tr><td>ogbn-arxiv</td><td>22.65</td><td>33.86</td><td>40.43</td><td>40.38</td><td>35.25</td><td>40.62</td></tr><tr><td>ogbn-products</td><td>22.46</td><td>28.54</td><td>27.24</td><td>27.47</td><td></td><td>27.33</td></tr><tr><td>Ele-Photo</td><td>44.15</td><td>55.97</td><td>61.00</td><td>59.58</td><td>57.06</td><td>61.43</td></tr><tr><td>Ele-Computers</td><td>38.05</td><td>59.94</td><td>66.85</td><td>62.45</td><td>59.64</td><td>69.56</td></tr><tr><td>Books-History</td><td>66.91</td><td>78.11</td><td>78.90</td><td>77.66</td><td>78.90</td><td>79.13</td></tr><tr><td>heterophilic controls</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Roman-Empire</td><td>19.34</td><td>19.30</td><td>19.30</td><td>19.29</td><td>19.31</td><td>19.30</td></tr><tr><td>Amazon-Ratings</td><td>34.17</td><td>36.74</td><td>36.80</td><td>36.73</td><td>36.73</td><td>36.78</td></tr><tr><td>Mean (9 main)</td><td>41.95</td><td>56.34</td><td>59.15</td><td>56.49</td><td></td><td>60.24</td></tr><tr><td>Mean (8 main, no products)</td><td>44.38</td><td>59.81</td><td>63.14</td><td>60.12</td><td>60.11</td><td>64.35</td></tr></table>

Table 9: External $Q + G$ baselines on clean features $( \sigma = 0 )$ , including the LAME-Graph and the Graph-TV methods.
<table><tr><td>Dataset</td><td>Q</td><td>APPNP</td><td>PPR-Prob</td><td>LAME-Graph</td><td>Graph-TV</td><td>PtS</td></tr><tr><td>WikiCS</td><td>72.67</td><td>77.78</td><td>78.24</td><td>78.18</td><td>78.18</td><td>78.42</td></tr><tr><td>Cora-TAPE</td><td>60.51</td><td>76.16</td><td>78.42</td><td>76.78</td><td>76.82</td><td>79.74</td></tr><tr><td>PubMed-TAPE</td><td>81.59</td><td>84.35</td><td>84.96</td><td>85.05</td><td>85.11</td><td>85.18</td></tr><tr><td>TAPE-Arxiv23</td><td>67.59</td><td>69.53</td><td>69.68</td><td>69.82</td><td>69.81</td><td>69.72</td></tr><tr><td>ogbn-arxiv</td><td>55.95</td><td>66.10</td><td>65.96</td><td>67.53</td><td>65.80</td><td>66.56</td></tr><tr><td>ogbn-products</td><td>59.58</td><td>70.58</td><td>71.32</td><td>71.40</td><td></td><td>72.17</td></tr><tr><td>Ele-Photo</td><td>66.83</td><td>71.87</td><td>74.43</td><td>74.84</td><td>72.03</td><td>74.91</td></tr><tr><td>Ele-Computers</td><td>60.96</td><td>75.37</td><td>78.71</td><td>78.32</td><td>76.47</td><td>80.11</td></tr><tr><td>Books-History</td><td>81.38</td><td>82.59</td><td>82.87</td><td>82.95</td><td>82.97</td><td>82.88</td></tr><tr><td>heterophilic controls</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Roman-Empire</td><td>65.54</td><td>65.65</td><td>65.54</td><td>65.55</td><td>65.57</td><td>65.54</td></tr><tr><td>Amazon-Ratings</td><td>49.52</td><td>51.63</td><td>52.77</td><td>52.05</td><td>52.36</td><td>52.74</td></tr><tr><td>Mean (9 main)</td><td>67.45</td><td>74.93</td><td>76.06</td><td>76.10</td><td></td><td>76.63</td></tr><tr><td>Mean (8 main, no products)</td><td>68.43</td><td>75.47</td><td>76.66</td><td>76.68</td><td>75.90</td><td>77.19</td></tr></table>

Tables 8 and 9 shows the results at $\sigma = 2$ and $\sigma = 0 .$ . The comparison with Graph-TV averages over 8 graphs without products, as it requires too much memory. PtS has a higher average accuracy in these comparisons, but is not best on all datasets. On clean data, LAME-Graph is better on the ogbn-arxiv, TAPE-Arxiv23, and Books-History datasets.

## J Full decomposition

Table 10 gives the per-dataset values for the decomposition summarised in Table 2, at $\sigma = 2$ with the frozen MLP

Table 10: Decomposition of PtS−APPNP at $\sigma = 2$ into the change from moving propagation into probability space, $\Delta _ { \mathrm { s p a c e } }$ and the change from adding the reaction, $\Delta _ { \mathrm { P t S } }$ . APPNP, PPR-Prob, and PtS are tuned independently. Reaction OFF sets $\eta = 0$ at the $( \alpha , K )$ selected for PtS, without retuning
<table><tr><td>Dataset</td><td>APPNP</td><td>PPR-Prob</td><td>PtS</td><td>Reaction OFF</td><td> $\Delta _ { \mathrm { s p a c e } }$  PPR-APPNP</td><td> $\Delta _ { \mathrm { P t S } } ~ \mathrm { P t S - P P R }$ </td></tr><tr><td>WikiCS</td><td>67.97</td><td>68.98</td><td>72.15</td><td>53.81</td><td> $+ 1 . 0 1 \pm 1 . 5 9$ </td><td> $+ 3 . 1 7 \pm 0 . 9 1$ </td></tr><tr><td>Cora-TAPE</td><td>68.16</td><td>70.47</td><td>72.30</td><td>64.83</td><td> $+ 2 . 3 1 \pm 1 . 9 0$ </td><td> $+ 1 . 8 3 \pm 0 . 6 5$ </td></tr><tr><td>PubMed-TAPE</td><td>78.87</td><td>80.22</td><td>80.73</td><td>76.80</td><td> $+ 1 . 3 5 \pm 1 . 2 5$ </td><td> $+ 0 . 5 1 \pm 0 . 2 7$ </td></tr><tr><td>TAPE-Arxiv23</td><td>35.63</td><td>38.26</td><td>38.91</td><td>36.90</td><td> $+ 2 . 6 4 \pm 0 . 2 8$ </td><td> $+ 0 . 6 5 \pm 0 . 3 7$ </td></tr><tr><td>ogbn-arxiv</td><td>33.86</td><td>40.43</td><td>40.62</td><td>39.95</td><td> $+ 6 . 5 7 \pm 1 . 7 1$ </td><td> $+ 0 . 1 8 \pm 0 . 2 2$ </td></tr><tr><td>ogbn-products</td><td>28.54</td><td>27.24</td><td>27.33</td><td>27.33</td><td> $- 1 . 3 0 \pm 0 . 2 2$ </td><td> $+ 0 . 0 9 \pm 0 . 1 5$ </td></tr><tr><td>Ele-Photo</td><td>55.97</td><td>61.00</td><td>61.43</td><td>59.55</td><td> $+ 5 . 0 3 \pm 0 . 7 5$ </td><td> $+ 0 . 4 2 \pm 0 . 4 1$ </td></tr><tr><td>Ele-Computers</td><td>59.94</td><td>66.85</td><td>69.56</td><td>58.13</td><td> $+ 6 . 9 1 \pm 0 . 8 4$ </td><td> $+ 2 . 7 2 \pm 0 . 6 4$ </td></tr><tr><td>Books-History</td><td>78.11</td><td>78.90</td><td>79.13</td><td>77.11</td><td> $+ 0 . 7 9 \pm 0 . 4 7$ </td><td> $+ 0 . 2 3 \pm 0 . 1 9$ </td></tr><tr><td>Mean (9 main)</td><td>56.34</td><td>59.15</td><td>60.24</td><td>54.93</td><td> $\mathbf { + 2 . 8 1 }$ </td><td>+1.09</td></tr></table>

Under independent validation selection and severe corruption $\sigma = 2 ,$ moving propagation from logit space to probability space accounts for 2.81 percentage points of the mean gain, and adding sharpening for a further 1.09 (Table 10). The split between the two varies by graph: on WikiCS it is +1.01 and +3.17, while on ogbn-arxiv it is +6.57 and +0.18. In the Reaction OFF column, we keep the $( \alpha , K )$ selected for PtS and set $\eta$ to zero without new tuning. Mean accuracy then falls to 54.93%, below APPNP itself, so the propagation configurations selected for PtS are only viable in combination with sharpening: PtS generally selects deeper propagation than PPR-Prob (Appendix H). Figure 3 shows how the mean gains change with corruption severity, with dataset-level results in Table 14. When hyperparameters are selected on clean data and evaluated at $\sigma = 2$ , PtS retains mean gains of 2.28 percentage points over APPNP in the nine main graphs (Figure 4).

## K Stronger backbones

We also examine whether PtS improves when frozen predictions come from base models that already use the graph structure. We apply the same method to predictions from GCN and GraphSAGE, with model sizes given in Table 5. The comparison explores how much additional improvement we can get when the base model has already exploited neighbourhood information.

## K.1 GCN Backbone

Table 11: Accuracy (%) with the frozen GCN backbone. ∆ is PtS minus APPNP in percentage points. Bold marks the highest displayed accuracy in each condition.
<table><tr><td></td><td colspan="4">Clean  $( \sigma = 0 )$ </td><td colspan="4">Noisy (σ = 2)</td></tr><tr><td>Dataset</td><td>Q(GCN)</td><td>APPNP</td><td>PtS</td><td> $\Delta$ </td><td>Q(GCN)</td><td>APPNP</td><td>PtS</td><td>Δ</td></tr><tr><td>WikiCS</td><td>79.08</td><td>79.05</td><td>79.40</td><td> $+ 0 . 3 5 \pm 0 . 2 1$ </td><td>76.73</td><td>76.91</td><td>77.68</td><td> $+ 0 . 7 7 \pm 0 . 3 0$ </td></tr><tr><td>Cora-TAPE</td><td>81.96</td><td>82.37</td><td>83.10</td><td> $+ 0 . 7 3 \pm 0 . 4 9$ </td><td>76.20</td><td>78.34</td><td>79.94</td><td> $+ 1 . 6 0 \pm 0 . 6 1$ </td></tr><tr><td>PubMed-TAPE</td><td>85.77</td><td>85.74</td><td>85.81</td><td> $+ 0 . 0 6 \pm 0 . 0 5$ </td><td>80.41</td><td>82.27</td><td>82.73</td><td> $+ 0 . 4 6 \pm 0 . 1 7$ </td></tr><tr><td>TAPE-Arxiv23</td><td>69.28</td><td>69.26</td><td>69.26</td><td> $- 0 . 0 0 \pm 0 . 0 2$ </td><td>43.43</td><td>44.04</td><td>44.33</td><td> $+ 0 . 2 9 \pm 0 . 1 1$ </td></tr><tr><td>ogbn-arxiv</td><td>71.83</td><td>71.90</td><td>72.48</td><td> $+ 0 . 5 8 \pm 0 . 2 8$ </td><td>62.73</td><td>64.66</td><td>64.78</td><td> $+ 0 . 1 2 \pm 0 . 3 3$ </td></tr><tr><td>ogbn-products</td><td>76.55</td><td>77.24</td><td>77.73</td><td> $+ 0 . 4 9 \pm 0 . 1 7$ </td><td>63.81</td><td>69.16</td><td>71.01</td><td> $+ 1 . 8 6 \pm 1 . 1 0$ </td></tr><tr><td>Ele-Photo</td><td>81.33</td><td>81.42</td><td>81.87</td><td> $+ 0 . 4 5 \pm 0 . 1 8$ </td><td>78.47</td><td>78.94</td><td>79.67</td><td> $+ 0 . 7 3 \pm 0 . 2 2$ </td></tr><tr><td>Ele-Computers</td><td>84.03</td><td>84.61</td><td>85.60</td><td> $+ 0 . 9 9 \pm 0 . 1 4$ </td><td>77.19</td><td>80.58</td><td>82.49</td><td> $+ 1 . 9 1 \pm 0 . 4 1$ </td></tr><tr><td>Books-History</td><td>82.68</td><td>82.66</td><td>82.66</td><td> $- 0 . 0 0 \pm 0 . 0 3$ </td><td>80.04</td><td>80.27</td><td>80.44</td><td> $+ 0 . 1 6 \pm 0 . 1 2$ </td></tr><tr><td>Roman-Empire</td><td>45.72</td><td>45.70</td><td>45.70</td><td> $+ 0 . 0 0 \pm 0 . 0 2$ </td><td>20.38</td><td>20.36</td><td>20.35</td><td> $- 0 . 0 1 \pm 0 . 0 2$ </td></tr><tr><td>Amazon-Ratings</td><td>50.02</td><td>50.00</td><td>50.00</td><td> $- 0 . 0 1 \pm 0 . 0 5$ </td><td>37.22</td><td>37.93</td><td>37.82</td><td> $- 0 . 1 1 \pm 0 . 2 4$ </td></tr><tr><td>Mean (9 main)</td><td>79.17</td><td>79.36</td><td>79.77</td><td>+0.41</td><td>71.00</td><td>72.80</td><td>73.67</td><td>+0.88</td></tr></table>

With a GCN backbone, performance on clean data improves incrementally on many graphs, but PtS achieves higher accuracy than APPNP on all nine main graphs at $\sigma = 2$ (Table 11) and on seven out of nine graphs on clean graphs.

## K.2 GraphSAGE backbone

Table 12: Accuracy (%) with the frozen GraphSAGE backbone. ∆ is PtS minus APPNP in percentage points. Bold marks the highest displayed accuracy in each condition.
<table><tr><td></td><td colspan="4">Clean (σ = 0)</td><td colspan="4">Noisy (σ = 2)</td></tr><tr><td>Dataset</td><td>Q(SAGE)</td><td>APPNP</td><td>PtS</td><td> $\Delta$ </td><td>Q(SAGE)</td><td>APPNP</td><td>PtS</td><td>∆</td></tr><tr><td>WikiCS</td><td>78.72</td><td>79.08</td><td>79.41</td><td> $+ 0 . 3 3 \pm 0 . 1 8$ </td><td>71.06</td><td>75.12</td><td>76.38</td><td> $+ 1 . 2 7 \pm 0 . 4 3$ </td></tr><tr><td>Cora-TAPE</td><td>73.54</td><td>74.76</td><td>75.96</td><td> $+ 1 . 2 0 \pm 1 . 1 7$ </td><td>68.18</td><td>70.89</td><td>72.69</td><td> $+ 1 . 8 0 \pm 0 . 9 5$ </td></tr><tr><td>PubMed-TAPE</td><td>84.10</td><td>84.47</td><td>84.80</td><td> $+ 0 . 3 3 \pm 0 . 1 8$ </td><td>72.26</td><td>78.63</td><td>80.04</td><td> $\lvert - 1 . 4 1 \pm 0 . 7 7$ </td></tr><tr><td>TAPE-Arxiv23</td><td>70.04</td><td>70.20</td><td>70.35</td><td> $+ 0 . 1 5 \pm 0 . 0 9$ </td><td>39.67</td><td>41.59</td><td>43.09</td><td> $+ 1 . 5 0 \pm 0 . 4 4$ </td></tr><tr><td>ogbn-arxiv</td><td>71.85</td><td>72.17</td><td>72.43</td><td> $+ 0 . 2 6 \pm 0 . 0 6$ </td><td>50.00</td><td>55.30</td><td>57.44</td><td> $+ 2 . 1 4 \pm 0 . 9 4$ </td></tr><tr><td>ogbn-products</td><td>78.18</td><td>78.34</td><td>78.37</td><td> $+ 0 . 0 4 \pm 0 . 1 1$ </td><td>39.56</td><td>39.61</td><td>39.69</td><td> $+ 0 . 0 7 \pm 0 . 1 3$ </td></tr><tr><td>Ele-Photo</td><td>79.44</td><td>79.53</td><td>80.16</td><td> $+ 0 . 6 3 \pm 0 . 2 6$ </td><td>76.38</td><td>76.81</td><td>77.43</td><td> $+ 0 . 6 2 \pm 0 . 2 2$ </td></tr><tr><td>Ele-Computers</td><td>79.71</td><td>80.96</td><td>82.23</td><td> $+ 1 . 2 8 \pm 0 . 3 0$ </td><td>73.70</td><td>77.80</td><td>79.92</td><td> $+ 2 . 1 2 \pm 0 . 3 6$ </td></tr><tr><td>Books-History</td><td>81.86</td><td>81.85</td><td>81.90</td><td> $+ 0 . 0 5 \pm 0 . 0 5$ </td><td>79.68</td><td>80.14</td><td>80.28</td><td> $+ 0 . 1 4 \pm 0 . 1 4$ </td></tr><tr><td>Roman-Empire</td><td>80.00</td><td>79.99</td><td>79.97</td><td> $- 0 . 0 3 \pm 0 . 0 5$ </td><td>21.05</td><td>21.01</td><td>21.01</td><td> $- 0 . 0 0 \pm 0 . 0 5$ </td></tr><tr><td>Amazon-Ratings</td><td>55.07</td><td>55.38</td><td>55.49</td><td> $+ 0 . 1 1 \pm 0 . 1 5$ </td><td>33.29</td><td>37.84</td><td>37.64</td><td> $- 0 . 2 0 \pm 0 . 1 7$ </td></tr><tr><td>Mean (9 main)</td><td>77.49</td><td>77.93</td><td>78.40</td><td>+0.47</td><td>63.39</td><td>66.21</td><td>67.44</td><td>+1.23</td></tr></table>

With GraphSAGE, PtS achieves higher accuracy than APPNP on the nine main homophilic graphs at the selected noise levels Table 12. The gain varies across backbones and increases with noise level. For example, the improvement under noise is more prominent on ogbn-arxiv and Ele-Computers, while it is smaller on ogbn-products.

## L Correct and Smooth comparison

Correct & Smooth uses known labels in its correction and smoothing stages. We evaluate sharpening within this label-assisted pipeline, leaving correction unchanged and modifying only smoothing. The C&S-PtS variant tests whether the introduced sharpening adds gain over other methods that use APPNP propagation.

Table 13: Comparison of Correct and Smooth (C&S) with C&S-PtS, where the categorical sharpening step is applied after each smoothing step. All methods use an MLP backbone. Subscripts give the corruption severity, and Gain is C&S-PtS minus C&S at $\sigma = 2$ in percentage points.
<table><tr><td>Dataset</td><td> $\mathrm { C } \& \mathrm { S } _ { 0 }$ </td><td> $\mathrm { C \& S  – P t S _ { 0 } }$ </td><td>C&amp;S2</td><td>C&amp;S-PtS2</td><td>Gain (pp)</td></tr><tr><td>WikiCS</td><td>76.90</td><td>78.82</td><td>70.06</td><td>75.43</td><td> $+ 5 . 3 8 \pm 0 . 8 5$ </td></tr><tr><td>Cora-TAPE</td><td>87.38</td><td>87.23</td><td>86.18</td><td>86.00</td><td> $- 0 . 1 7 \pm 0 . 2 0$ </td></tr><tr><td>PubMed-TAPE</td><td>86.97</td><td>86.98</td><td>84.44</td><td>84.44</td><td> $0 . 0 0 \pm 0 . 0 5$ </td></tr><tr><td>TAPE-Arxiv23</td><td>70.32</td><td>70.31</td><td>45.54</td><td>45.55</td><td> $+ 0 . 0 2 \pm 0 . 0 4$ </td></tr><tr><td>ogbn-arxiv</td><td>70.91</td><td>71.00</td><td>68.35</td><td>68.61</td><td> $+ 0 . 2 6 \pm 0 . 3 0$ </td></tr><tr><td>ogbn-products</td><td>77.97</td><td>78.67</td><td>75.08</td><td>76.73</td><td> $+ 1 . 6 5 \pm 0 . 5 3$ </td></tr><tr><td>Ele-Photo</td><td>87.26</td><td>87.31</td><td>85.80</td><td>86.13</td><td> $+ 0 . 3 3 \pm 0 . 0 9$ </td></tr><tr><td>Ele-Computers</td><td>90.68</td><td>90.73</td><td>89.87</td><td>90.08</td><td> $+ 0 . 2 1 \pm 0 . 0 8$ </td></tr><tr><td>Books-History</td><td>84.53</td><td>84.53</td><td>82.74</td><td>82.81</td><td> $+ 0 . 0 7 \pm 0 . 0 4$ </td></tr><tr><td>Roman-Empire</td><td>64.63</td><td>64.67</td><td>19.50</td><td>19.51</td><td> $+ 0 . 0 1 \pm 0 . 0 7$ </td></tr><tr><td>Amazon-Ratings</td><td>53.57</td><td>53.53</td><td>45.76</td><td>46.20</td><td> $+ 0 . 4 4 \pm 0 . 2 9$ </td></tr></table>

Table 13 shows the results on clean data and under noise, while Table 15 shows the entire noise ladder. The improvement is especially large on WikiCS and ogbn-products, while many other datasets show small diferences. Sharpening can therefore also be efective here, but with dataset-dependent efects.

## M Full Gaussian ladder

Table 14 shows the results at all the five levels of gaussian noise. Each method chooses hyperparameters on the same noise levels that they are evaluated on.

Table 14: Accuracy across the full Gaussian corruption ladder. APPNP propagates logits, PPR-Prob isolates probability-space propagation, and PtS additionally applies the categorical sharpening step.
<table><tr><td>Dataset</td><td>Method</td><td>σ = 0</td><td>σ = 0.5</td><td>σ = 1</td><td>σ = 1.5</td><td>σ = 2</td></tr><tr><td>WikiCS</td><td>APPNP</td><td>77.78</td><td>77.12</td><td>75.33</td><td>72.19</td><td>67.97</td></tr><tr><td></td><td>PPR-Prob</td><td>78.24</td><td>77.48</td><td>75.69</td><td>73.09</td><td>68.98</td></tr><tr><td></td><td>PtS</td><td>78.42</td><td>77.61</td><td>76.14</td><td>74.38</td><td>72.15</td></tr><tr><td>Cora-TAPE</td><td>APPNP</td><td>76.16</td><td>75.60</td><td>73.87</td><td>71.43</td><td>68.16</td></tr><tr><td></td><td>PPR-Prob</td><td>78.42</td><td>77.92</td><td>76.28</td><td>73.52</td><td>70.47</td></tr><tr><td></td><td>PtS</td><td>79.74</td><td>78.95</td><td>77.32</td><td>75.21</td><td>72.30</td></tr><tr><td>PubMed-TAPE</td><td>APPNP</td><td>84.35</td><td>83.59</td><td>81.98</td><td>80.35</td><td>78.87</td></tr><tr><td></td><td>PPR-Prob</td><td>84.96</td><td>84.17</td><td>82.76</td><td>81.46</td><td>80.22</td></tr><tr><td></td><td>PtS</td><td>85.18</td><td>84.39</td><td>83.05</td><td>81.84</td><td>80.73</td></tr><tr><td>TAPE-Arxiv23</td><td>APPNP</td><td>69.53</td><td>65.29</td><td>55.35</td><td>44.87</td><td>35.63</td></tr><tr><td></td><td>PPR-Prob</td><td>69.68</td><td>65.27</td><td>55.32</td><td>45.87</td><td>38.26</td></tr><tr><td></td><td>PtS</td><td>69.72</td><td>65.28</td><td>55.33</td><td>45.90</td><td>38.91</td></tr><tr><td>ogbn-arxiv</td><td>APPNP</td><td>66.10</td><td>64.53</td><td>58.09</td><td>45.50</td><td>33.86</td></tr><tr><td></td><td>PPR-Prob</td><td>65.96</td><td>65.03</td><td>62.72</td><td>52.24</td><td>40.43</td></tr><tr><td></td><td>PtS</td><td>66.56</td><td>65.92</td><td>63.04</td><td>53.93</td><td>40.62</td></tr><tr><td>ogbn-products</td><td>APPNP</td><td>70.58</td><td>56.83</td><td>38.51</td><td>31.64</td><td>28.54</td></tr><tr><td></td><td>PPR-Prob</td><td>71.32</td><td>57.66</td><td>37.26</td><td>30.27</td><td>27.24</td></tr><tr><td></td><td>PtS</td><td>72.17</td><td>57.12</td><td>37.28</td><td>30.34</td><td>27.33</td></tr><tr><td>Ele-Photo</td><td>APPNP</td><td>71.87</td><td>70.39</td><td>66.70</td><td>61.25</td><td>55.97</td></tr><tr><td></td><td>PPR-Prob</td><td>74.43</td><td>73.31</td><td>70.02</td><td>65.39</td><td>61.00</td></tr><tr><td></td><td>PtS</td><td>74.91</td><td>73.46</td><td>70.18</td><td>65.97</td><td>61.43</td></tr><tr><td>Ele-Computers</td><td>APPNP</td><td>75.37</td><td>74.14</td><td>71.17</td><td>66.25</td><td>59.94</td></tr><tr><td></td><td>PPR-Prob</td><td>78.71</td><td>77.95</td><td>75.21</td><td>71.46</td><td>66.85</td></tr><tr><td>Books-History</td><td>PtS</td><td>80.11</td><td>79.26</td><td>77.07</td><td>73.74</td><td>69.56</td></tr><tr><td></td><td>APPNP</td><td>82.59</td><td>82.20</td><td>81.11</td><td>79.67</td><td>78.11</td></tr><tr><td></td><td>PPR-Prob</td><td>82.87</td><td>82.47</td><td>81.41</td><td>80.21</td><td>78.90</td></tr><tr><td></td><td>PtS</td><td>82.88</td><td>82.49</td><td>81.47</td><td>80.31</td><td>79.13</td></tr><tr><td colspan="7">Heterophilic controls</td></tr><tr><td>Roman-Empire</td><td>APPNP</td><td>65.65</td><td>53.84</td><td>36.19</td><td>25.11</td><td>19.30</td></tr><tr><td></td><td>PPR-Prob</td><td>65.54</td><td>53.88</td><td>36.19</td><td>25.10</td><td>19.30</td></tr><tr><td></td><td>PtS</td><td>65.54</td><td>53.87</td><td>36.18</td><td>25.10</td><td>19.30</td></tr><tr><td>Amazon-Ratings</td><td>APPNP</td><td>51.63</td><td>44.20</td><td>38.70</td><td>37.09</td><td>36.74</td></tr><tr><td></td><td>PPR-Prob</td><td>52.77</td><td>45.27</td><td>38.81</td><td>37.06</td><td>36.80</td></tr><tr><td></td><td>PtS</td><td>52.74</td><td>45.20</td><td>38.81</td><td>37.05</td><td>36.78</td></tr></table>

Table 15: Correct & Smooth across the full Gaussian corruption ladder. C&S-PtS applies the same categorical sharpening step during the smoothing stage.
<table><tr><td>Dataset</td><td>Method</td><td>σ = 0</td><td>σ = 0.5</td><td>σ = 1</td><td>σ = 1.5</td><td>σ = 2</td></tr><tr><td>WikiCS</td><td>C&amp;S</td><td>76.90</td><td>75.64</td><td>73.14</td><td>71.19</td><td>70.06</td></tr><tr><td></td><td>C&amp;S-PtS</td><td>78.82</td><td>78.27</td><td>77.32</td><td>76.31</td><td>75.43</td></tr><tr><td>Cora-TAPE</td><td>C&amp;S</td><td>87.38</td><td>87.14</td><td>86.81</td><td>86.54</td><td>86.18</td></tr><tr><td></td><td>C&amp;S-PtS</td><td>87.23</td><td>87.05</td><td>86.66</td><td>86.42</td><td>86.00</td></tr><tr><td>PubMed-TAPE</td><td>C&amp;S</td><td>86.97</td><td>86.39</td><td>85.48</td><td>84.89</td><td>84.44</td></tr><tr><td></td><td>C&amp;S-PtS</td><td>86.98</td><td>86.38</td><td>85.48</td><td>84.88</td><td>84.44</td></tr><tr><td>TAPE-Arxiv23</td><td>C&amp;S</td><td>70.32</td><td>66.31</td><td>57.91</td><td>50.56</td><td>45.54</td></tr><tr><td></td><td>C&amp;S-PtS</td><td>70.31</td><td>66.32</td><td>57.92</td><td>50.56</td><td>45.55</td></tr><tr><td>ogbn-arxiv</td><td>C&amp;S</td><td>70.91</td><td>69.99</td><td>68.98</td><td>68.49</td><td>68.35</td></tr><tr><td></td><td>C&amp;S-PtS</td><td>71.00</td><td>70.07</td><td>69.16</td><td>68.71</td><td>68.61</td></tr><tr><td>ogbn-products</td><td>C&amp;S</td><td>77.97</td><td>76.55</td><td>75.84</td><td>75.26</td><td>75.08</td></tr><tr><td></td><td>C&amp;S-PtS</td><td>78.67</td><td>78.07</td><td>77.25</td><td>77.04</td><td>76.73</td></tr><tr><td>Ele-Photo</td><td>C&amp;S</td><td>87.26</td><td>87.04</td><td>86.62</td><td>86.20</td><td>85.80</td></tr><tr><td></td><td>C&amp;S-PtS</td><td>87.31</td><td>87.11</td><td>86.73</td><td>86.40</td><td>86.13</td></tr><tr><td>Ele-Computers</td><td>C&amp;S</td><td>90.68</td><td>90.57</td><td>90.35</td><td>90.08</td><td>89.87</td></tr><tr><td></td><td>C&amp;S-PtS</td><td>90.73</td><td>90.65</td><td>90.47</td><td>90.26</td><td>90.08</td></tr><tr><td>Books-History</td><td>C&amp;S</td><td>84.53</td><td>84.33</td><td>83.78</td><td>83.24</td><td>82.74</td></tr><tr><td></td><td>C&amp;S-PtS</td><td>84.53</td><td>84.32</td><td>83.84</td><td>83.27</td><td>82.81</td></tr><tr><td colspan="7">Heterophilic controls</td></tr><tr><td>Roman-Empire</td><td>C&amp;S</td><td>64.63</td><td>53.00</td><td>35.98</td><td>25.22</td><td>19.50</td></tr><tr><td></td><td>C&amp;S-PtS</td><td>64.67</td><td>52.96</td><td>35.95</td><td>25.22</td><td>19.51</td></tr><tr><td>Amazon-Ratings</td><td>C&amp;S</td><td>53.57</td><td>48.18</td><td>46.47</td><td>45.98</td><td>45.76</td></tr><tr><td></td><td>C&amp;S-PtS</td><td>53.53</td><td>48.18</td><td>46.74</td><td>46.33</td><td>46.20</td></tr></table>

## N Preserving mass

The row mass chooses how strongly a node’s class distribution influences the next propagation step. In Table 16, we test whether rescaling this mass helps after sharpening, compared with normalizing each row sum to one. The comparison also includes propagation without sharpening so that we can examine the efect of row normalization in both cases.

We compare PtS with a variant that resets each row’s mass to one after every iteration, $R _ { \eta } ^ { \mathrm { r n } } ( h ) = r _ { \eta } ( h / m ( h ) )$ Its sharpening-of counterpart, PPR-rn, applies row normalisation after every probability propagation step.

Table 16: Efect of preserving the propagated row mass. $\Delta _ { \mathrm { P t S } }$ is PtS minus PtS-rn and $\Delta _ { \mathrm { P P R } }$ is PPR-Prob minus PPR-rn, both computed within (split, seed, draw). Bold marks the higher of the two accuracies in each row.
<table><tr><td>Dataset</td><td>PtS-rn</td><td>PtS</td><td> $\Delta _ { \mathrm { P t S } }$ </td><td> $\Delta _ { \mathrm { P P R } }$ </td></tr><tr><td colspan="5">Clean (σ = 0)</td></tr><tr><td>WikiCS</td><td>78.24</td><td>78.42</td><td> $+ 0 . 1 8 \pm 0 . 2 7$ </td><td> $+ 0 . 1 5 \pm 0 . 2 7$ </td></tr><tr><td>Cora-TAPE</td><td>79.15</td><td>79.74</td><td> $+ 0 . 5 9 \pm 0 . 6 6$ </td><td> $+ 0 . 1 7 \pm 0 . 5 8$ </td></tr><tr><td>PubMed-TAPE</td><td>85.12</td><td>85.18</td><td> $+ 0 . 0 7 \pm 0 . 1 6$ </td><td> $+ 0 . 0 3 \pm 0 . 0 8$ </td></tr><tr><td>TAPE-Arxiv23</td><td>69.75</td><td>69.72</td><td> $- 0 . 0 3 \pm 0 . 0 6$ </td><td> $- 0 . 0 8 \pm 0 . 0 4$ </td></tr><tr><td>ogbn-arxiv</td><td>67.45</td><td>66.56</td><td> $- 0 . 8 9 \pm 0 . 3 4$ </td><td> $- 1 . 1 1 \pm 0 . 3 2$ </td></tr><tr><td>ogbn-products</td><td>71.93</td><td>72.17</td><td> $+ 0 . 2 4 \pm 0 . 1 1$ </td><td> $+ 0 . 6 0 \pm 0 . 0 5$ </td></tr><tr><td>Ele-Photo</td><td>74.92</td><td>74.91</td><td> $- 0 . 0 1 \pm 0 . 3 0$ </td><td> $- 0 . 3 4 \pm 0 . 4 7$ </td></tr><tr><td>Ele-Computers</td><td>79.67</td><td>80.11</td><td> $+ 0 . 4 5 \pm 0 . 1 5$ </td><td> $+ 0 . 0 1 \pm 0 . 3 3$ </td></tr><tr><td>Books-History</td><td>82.90</td><td>82.88</td><td> $- 0 . 0 2 \pm 0 . 1 7$ </td><td> $+ 0 . 0 2 \pm 0 . 0 7$ </td></tr><tr><td>Mean (9 main)</td><td>76.57</td><td>76.63</td><td> ${ \bf + 0 . 0 6 }$ </td><td>-0.06</td></tr><tr><td colspan="5">Heterophilic controls</td></tr><tr><td>Roman-Empire</td><td>65.54</td><td>65.54</td><td> $+ 0 . 0 0 \pm 0 . 0 2$ </td><td> $+ 0 . 0 0 \pm 0 . 0 1$ </td></tr><tr><td>Amazon-Ratings</td><td>52.69</td><td>52.74</td><td> $+ 0 . 0 5 \pm 0 . 1 4$ </td><td> $+ 0 . 0 2 \pm 0 . 1 5$ </td></tr><tr><td colspan="5">Corrupted (σ = 2)</td></tr><tr><td>WikiCS</td><td>69.66</td><td>72.15</td><td> $+ 2 . 4 8 \pm 0 . 4 2$ </td><td> $+ 1 . 0 1 \pm 0 . 4 2$ </td></tr><tr><td>Cora-TAPE</td><td>71.24</td><td>72.30</td><td> $+ 1 . 0 6 \pm 0 . 6 0$ </td><td> $+ 0 . 5 0 \pm 0 . 4 6$ </td></tr><tr><td>PubMed-TAPE</td><td>80.63</td><td>80.73</td><td> $+ 0 . 1 0 \pm 0 . 1 3$ </td><td> $+ 0 . 1 4 \pm 0 . 1 7$ </td></tr><tr><td>TAPE-Arxiv23</td><td>38.69</td><td>38.91</td><td> $+ 0 . 2 2 \pm 0 . 1 6$ </td><td> $+ 0 . 1 9 \pm 0 . 1 0$ </td></tr><tr><td>ogbn-arxiv</td><td>41.68</td><td>40.62</td><td> $- 1 . 0 7 \pm 0 . 2 9$ </td><td> $- 0 . 1 7 \pm 1 . 1 1$ </td></tr><tr><td>ogbn-products</td><td>27.44</td><td>27.33</td><td> $- 0 . 1 2 \pm 0 . 1 4$ </td><td> $- 0 . 2 6 \pm 0 . 2 6$ </td></tr><tr><td>Ele-Photo</td><td>60.73</td><td>61.43</td><td> $+ 0 . 7 0 \pm 0 . 5 5$ </td><td> $+ 0 . 4 0 \pm 0 . 3 2$ </td></tr><tr><td>Ele-Computers</td><td>67.42</td><td>69.56</td><td> $+ 2 . 1 4 \pm 0 . 3 7$ </td><td> $+ 1 . 1 0 \pm 0 . 4 9$ </td></tr><tr><td>Books-History</td><td>79.09</td><td>79.13</td><td> $+ 0 . 0 4 \pm 0 . 0 9$ </td><td> $+ 0 . 0 4 \pm 0 . 0 6$ </td></tr><tr><td>Mean (9 main)</td><td>59.62</td><td>60.24</td><td> ${ \bf + 0 . 6 1 }$ </td><td> ${ \bf + 0 . 3 3 }$ </td></tr><tr><td colspan="5">Heterophilic controls</td></tr><tr><td>Roman-Empire</td><td>19.29</td><td>19.30</td><td> $+ 0 . 0 0 \pm 0 . 0 4$ </td><td> $+ 0 . 0 0 \pm 0 . 0 3$ </td></tr><tr><td>Amazon-Ratings</td><td>36.79</td><td>36.78</td><td> $- 0 . 0 2 \pm 0 . 0 7$ </td><td> $- 0 . 0 1 \pm 0 . 0 6$ </td></tr></table>

The diferences are small on many noise-free datasets while mass preservation gives larger improvements under noise, especially on WikiCS and Ele-Computers. The efect is still not positive everywhere, on ogbn-arxiv, the variant with row normalization is better in both. This means mass preservation influences the results but does not guarantee higher accuracy.

## O Depth sensitivity

Table 17 gives the complete values for the depth analysis in Figure 2. Logit-Sharp updates the logits as $L \gets$ $L + \eta \operatorname { s o f t m a x } ( L )$ after each APPNP propagation step, with final predictions softmax(L). We vary propagation steps K, while keeping α and η fixed. Without restart, APPNP and PPR-Prob are pure difusion methods, so they oversmooth and lose substantial accuracy at greater depths, while strong sharpening reduces this loss substantially. Restart dampens this decrease, but PtS keeps its advantage at greater depths here. A larger η does not always help: $\eta = 2 0 0$ is best at large depth without restart, while $\eta = 1 6$ is better with restart at $\sigma = 2$

Table 17: Depth sensitivity under clean and corrupted inputs.
<table><tr><td>Curve</td><td>K=1</td><td>K=2</td><td>K=3</td><td>K=5</td><td>K=10</td><td>K=20</td><td>K=40</td><td>K=100</td></tr><tr><td colspan="7">σ = 0, α = 0 (no restart)</td><td></td><td></td></tr><tr><td>frozen Q</td><td>67.5</td><td>67.5</td><td>67.5</td><td>67.5</td><td>67.5</td><td>67.5</td><td>67.5</td><td>67.5</td></tr><tr><td>APPNP</td><td>74.5</td><td>74.3</td><td>73.4</td><td>70.7</td><td>64.8</td><td>58.2</td><td>50.2</td><td>40.5</td></tr><tr><td>Logit-Sharp η=16</td><td>74.5</td><td>74.9</td><td>74.7</td><td>74.3</td><td>72.7</td><td>68.0</td><td>60.7</td><td>52.9</td></tr><tr><td>PPR-Prob</td><td>75.1</td><td>75.8</td><td>75.2</td><td>73.3</td><td>68.5</td><td>61.4</td><td>53.7</td><td>44.6</td></tr><tr><td>PtS η=16</td><td>75.1</td><td>75.8</td><td>75.8</td><td>75.3</td><td>74.4</td><td>73.5</td><td>71.9</td><td>70.6</td></tr><tr><td>PtS η=200</td><td>75.1</td><td>75.4</td><td>75.4</td><td>75.1</td><td>74.7</td><td>74.2</td><td>73.5</td><td>73.2</td></tr><tr><td colspan="9">σ = 0, α = 0.1 (restart)</td></tr><tr><td>frozen Q</td><td>67.5</td><td>67.5</td><td>67.5</td><td>67.5</td><td>67.5</td><td>67.5</td><td>67.5</td><td>67.5</td></tr><tr><td>APPNP</td><td>74.3</td><td>74.5</td><td>74.1</td><td>73.0</td><td>71.4</td><td>70.6</td><td>70.4</td><td>70.4</td></tr><tr><td>Logit-Sharp η=16</td><td>74.3</td><td>75.1</td><td>75.0</td><td>74.6</td><td>73.3</td><td>70.2</td><td>64.6</td><td>61.0</td></tr><tr><td>PPR-Prob</td><td>74.8</td><td>75.6</td><td>75.6</td><td>75.0</td><td>74.0</td><td>73.6</td><td>73.5</td><td>73.5</td></tr><tr><td>PtS η=16</td><td>74.8</td><td>75.9</td><td>76.1</td><td>75.8</td><td>75.3</td><td>74.9</td><td>74.5</td><td>74.2</td></tr><tr><td>PtS η=200</td><td>74.8</td><td>75.7</td><td>75.7</td><td>75.5</td><td>75.2</td><td>75.0</td><td>74.9</td><td>74.8</td></tr><tr><td colspan="9">σ = 2, α = 0 (no restart)</td></tr><tr><td>frozen Q</td><td>41.9</td><td>41.9</td><td>41.9</td><td>41.9</td><td>41.9</td><td>41.9</td><td>41.9</td><td>41.9</td></tr><tr><td>APPNP</td><td>54.0</td><td>55.8</td><td>55.5</td><td>53.8</td><td>50.3</td><td>45.9</td><td>40.4</td><td>34.7</td></tr><tr><td>Logit-Sharp η=16</td><td>54.0</td><td>56.0</td><td>56.4</td><td>56.4</td><td>55.6</td><td>53.0</td><td>48.5</td><td>42.5</td></tr><tr><td>PPR-Prob</td><td>54.2</td><td>57.7</td><td>58.5</td><td>57.8</td><td>54.4</td><td>50.2</td><td>44.5</td><td>37.2</td></tr><tr><td>PtS η=16</td><td>54.2</td><td>58.1</td><td>58.8</td><td>59.4</td><td>59.0</td><td>58.1</td><td>57.0</td><td>56.1</td></tr><tr><td>PtS η=200</td><td>54.2</td><td>57.7</td><td>58.0</td><td>58.1</td><td>58.7</td><td>58.5</td><td>57.9</td><td>57.7</td></tr><tr><td colspan="9">σ = 2, α = 0.1 (restart)</td></tr><tr><td>frozen Q</td><td>41.9</td><td>41.9</td><td>41.9</td><td>41.9</td><td>41.9</td><td>41.9</td><td>41.9</td><td>41.9</td></tr><tr><td>APPNP</td><td>53.5</td><td>55.2</td><td>55.4</td><td>54.7</td><td>53.5</td><td>52.8</td><td>52.6</td><td>52.6</td></tr><tr><td>Logit-Sharp η=16</td><td>53.5</td><td>55.8</td><td>56.4</td><td>56.5</td><td>56.0</td><td>54.2</td><td>51.1</td><td>48.0</td></tr><tr><td>PPR-Prob</td><td>53.1</td><td>55.6</td><td>56.3</td><td>56.3</td><td>55.5</td><td>55.0</td><td>54.8</td><td>54.8</td></tr><tr><td>PtS η=16</td><td>53.1</td><td>57.1</td><td>57.9</td><td>58.1</td><td>58.2</td><td>57.8</td><td>57.3</td><td>57.0</td></tr><tr><td>PtS η=200</td><td>53.1</td><td>56.8</td><td>57.2</td><td>57.1</td><td>56.7</td><td>56.4</td><td>56.2</td><td>56.1</td></tr></table>

## P Dirichlet and Gini

Figure 6 shows how the energy components and validation accuracy evolve through the PtS iteration on WikiCS. We use the configuration α = 0.1, η = 16, and σ = 2 to illustrate what happens before and after propagation and sharpening.

![](images/6c5f1eebd9bca637cbe07820a2c72f23c93a618674379ed49d32d354aa5d6d35.jpg)  
Figure 6: Evolution of the Dirichlet energy $E _ { \mathrm { D } }$ (left), Gini energy $E _ { \mathrm { G } }$ (middle), and validation accuracy (right) under gaussian feature corruption at $\sigma = 2$ . Gold and red markers show values after propagation and sharpening. Black markers denote the frozen predictions Q. Propagation reduces disagreement between nodes, but increases indecision within nodes, while sharpening has the opposite efect

In the visualization, we can see that propagation reduces the Dirichlet term and increases the Gini term, while sharpening has the opposite efect. Still, PtS maintains higher validation accuracy at greater depth than the variant without reaction. Since sharpening preserves each node’s predicted class, its efect must occur through subsequent propagation steps. Proposition 1 guarantees local energy descent under sharpening.

## Q Per-node performance

We start from a strong homophily prior: neighbouring nodes should be similar. We group nodes by local homophily, degree quintile, and by the confidence ma $\mathrm { X } _ { c } Q _ { i c }$ . Table 18 shows the test-accuracy diference between PtS and APPNP in each group, using backbone seed 0 for each split and noise draw 0 at $\sigma = 2$ , with the hyperparameters selected in the main experiment.

We can see that PtS gives the greatest improvement in groups with higher local homophily. Most interestingly, PtS helps most in uncertain areas, the accuracy increase is also higher at nodes with lower prediction confidence. This suggests that PtS is most useful when the original predictions are uncertain, and the neighbourhood provides relevant class information.

Table 18: PtS−APPNP accuracy diference in percentage points, with nodes stratified by local homophily, by degree quintile, and by the confidence of the frozen prediction.
<table><tr><td>Node stratum</td><td>Bucket</td><td>PtS-APPNP (σ=0) PtS-APPNP (σ=2)</td></tr><tr><td>Local homophily</td><td>0-.2</td><td>-1.29 -1.75</td></tr><tr><td></td><td>.2-.4</td><td>-0.51 -0.35</td></tr><tr><td></td><td>.4-.6</td><td>+1.88 +4.83</td></tr><tr><td></td><td>.6-.8</td><td>+3.69 +9.12</td></tr><tr><td></td><td>.8-1</td><td>+1.90 +5.76</td></tr><tr><td>Degree quintile</td><td>isolated</td><td>+0.00 +0.00</td></tr><tr><td></td><td>Q1</td><td>+1.27 +3.64</td></tr><tr><td></td><td>Q2</td><td>+1.63 +3.56</td></tr><tr><td></td><td>Q3</td><td>+1.86 +3.77</td></tr><tr><td></td><td>Q4</td><td>+1.95 +4.41</td></tr><tr><td></td><td>Q5</td><td>+2.04 +5.66</td></tr><tr><td>Anchor confidence quintile</td><td>Q1</td><td>+3.41 +5.12</td></tr><tr><td></td><td>Q2</td><td>+2.30 +4.66</td></tr><tr><td></td><td>Q3</td><td>+1.60 +4.39</td></tr><tr><td></td><td>Q4</td><td>+0.95 +3.87</td></tr><tr><td></td><td>Q5</td><td>+0.48 +3.00</td></tr></table>

## R Runtime and overhead

Table 19 shows the runtime for the base models forward pass, as well as the post hoc processing at both 1 and 100 steps. We can see that, generally, for all methods, the runtime is very low.

Table 19: Runtime and overhead: MLP forward-pass and post-hoc call times in milliseconds at σ = 2
<table><tr><td></td><td></td><td>MLP</td><td colspan="3">Per step (ms)</td><td colspan="3">K = 100 (ms)</td><td>PtS/APPNP</td></tr><tr><td>Dataset</td><td>Device fwd (ms)</td><td></td><td>APPNP</td><td>PPR-Prob</td><td>PtS</td><td>APPNP</td><td>PPR-Prob</td><td>PtS</td><td>per step</td></tr><tr><td>WikiCS</td><td>CPU</td><td>23.0</td><td>1.00</td><td></td><td>2.024.72</td><td>100</td><td>202</td><td>472</td><td>4.7</td></tr><tr><td>Cora-TAPE</td><td>CPU</td><td>5.4</td><td>0.09</td><td></td><td>0.44 1.31</td><td>8.8</td><td>43.6</td><td>131</td><td>14.9</td></tr><tr><td>PubMed-TAPE</td><td>CPU</td><td>59.0</td><td>0.18</td><td></td><td>0.903.08</td><td>17.7</td><td>89.8</td><td>308</td><td>17.4</td></tr><tr><td>TAPE-Arxiv23</td><td>CPU</td><td>87.6</td><td>5.45</td><td></td><td>12.1 30.7</td><td>545</td><td>1,211 3,068</td><td></td><td>5.6</td></tr><tr><td>ogbn-arxiv</td><td>A100</td><td>9.9</td><td>0.90</td><td></td><td>1.69 3.85</td><td>90.0</td><td>169</td><td>385</td><td>4.3</td></tr><tr><td>ogbn-products</td><td>A100</td><td>161</td><td>30.7</td><td></td><td>36.7 54.3</td><td>3,072</td><td>3,669</td><td>5,428</td><td>1.8</td></tr><tr><td>Ele-Photo</td><td>CPU</td><td>169</td><td>1.98</td><td></td><td>3.87 9.73</td><td>198</td><td>387</td><td>973</td><td>4.9</td></tr><tr><td>Ele-Computers</td><td>CPU</td><td>217</td><td>4.03</td><td></td><td>6.55 15.6</td><td>403</td><td>655</td><td>1,564</td><td>3.9</td></tr><tr><td>Books-History</td><td>CPU</td><td>138</td><td>1.98</td><td></td><td>3.299.19</td><td>198</td><td>329</td><td>919</td><td>4.6</td></tr><tr><td>Roman-Empire</td><td>CPU</td><td>41.7</td><td>0.41</td><td></td><td>1.64 5.61</td><td>40.8</td><td>164</td><td>561</td><td>13.8</td></tr><tr><td>Amazon-Ratings</td><td>CPU</td><td>45.2</td><td>0.28</td><td></td><td>0.93 3.10</td><td>27.8</td><td>92.7</td><td>310</td><td>11.2</td></tr></table>

## S Calibration

Sharpening can make class distributions more concentrated, but directly influencing probabilities can harm the model’s calibration. We therefore examine the negative log-likelihood (NLL) and expected calibration error (ECE) before and after temperature scaling using validation nodes.

Table 20: Calibration of the frozen prediction, APPNP and PtS with an MLP backbone, before and after temperature scaling fitted on validation nodes. T is the fitted temperature and accuracy drift is the change in accuracy caused by scaling. Values are means over WikiCS, Cora-TAPE, PubMed-TAPE, TAPE-arxiv23, ogbn-arxiv and ogbn-products
<table><tr><td>σ</td><td>Method</td><td>T</td><td>raw NLL</td><td>scaled NLL</td><td>raw ECE</td><td>scaled ECE</td><td>acc. drift (pp)</td></tr><tr><td>0</td><td>Frozen Q</td><td>1.34</td><td>1.226</td><td>1.132</td><td>0.088</td><td>0.040</td><td>0</td></tr><tr><td rowspan="5">2</td><td>APPNP</td><td>0.84</td><td>0.983</td><td>0.907</td><td>0.129</td><td>0.037</td><td>0</td></tr><tr><td>PtS</td><td>2.11</td><td>1.591</td><td>0.932</td><td>0.158</td><td>0.050</td><td>0</td></tr><tr><td>Frozen Q</td><td>3.55</td><td>3.514</td><td>2.112</td><td>0.302</td><td>0.047</td><td>0</td></tr><tr><td>APPNP</td><td>1.47</td><td>2.108</td><td>1.752</td><td>0.178</td><td>0.075</td><td>0</td></tr><tr><td>PtS</td><td>2.92</td><td>3.088</td><td>1.692</td><td>0.254</td><td>0.093</td><td>0</td></tr></table>

PtS has higher uncorrected NLL and ECE than APPNP, both with and without noise. Temperature scaling mitigates this without afecting accuracy.