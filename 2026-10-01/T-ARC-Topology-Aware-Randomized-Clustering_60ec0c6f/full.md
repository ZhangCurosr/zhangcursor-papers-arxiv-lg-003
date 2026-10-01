# T-ARC: Topology-Aware Randomized Clustering

via Distributionally Robust Stochastic Block Models

Serena Grazia De Benedictis <sup>∗1</sup>, Andersen Ang<sup>2</sup>, Nicoletta Del Buono<sup>1</sup>, Flavia Esposito<sup>1</sup>, and Laura Selicato<sup>3</sup>

<sup>1</sup>Department of Mathematics, University of Bari Aldo Moro, Italy <sup>2</sup>Electronics and Computer Science Department, University of Southampton, UK <sup>3</sup>National Research Council (CNR), Water Research Institute (IRSA), Italy

## Abstract

In this work, we introduce a new clustering method, namely T-ARC (Topology-Aware Randomized Clustering), that corrects the geometric bias of K-means by embedding topological information directly into the optimization objective. Building on the assumption that the data admits an underlying hidden structure modeled via a latent graph, the idea is to uncover this information through the interplay between the standard K-means data-fidelity term and a graph-cut penalty, which discourages cluster assignments inconsistent with the connectivity structure of the data.

To render this coupling tractable, the latent graph is modeled as a random realization from a Stochastic Block Model (SBM), whose scalar parameter is optimized within a Distributionally Robust Optimization (DRO) framework, yielding a closed-form proximal update. Both SBM and DRO are informed by a persistence-based similarity matrix derived from zero-dimensional persistent homology $\left( H _ { 0 } \right)$ , which translates the multiscale connectivity structure of the data into a pairwise topological prior. The overall optimization proceeds via Block Coordinate Descent; convergence is established through a global Lyapunov functional: the deterministic blocks satisfy monotonic descent, while the stochastic graph update satisfies descent in expectation, so that the expected energy converges.

Experiments on synthetic datasets with non-convex geometries and on random subsets of Fashion-MNIST show that T-ARC recovers latent topological structures where K-means fails, achieving the highest accuracy on curved and interleaved clusters while remaining competitive, and markedly more stable than K-means, on real data.

## Contents

1 Introduction

2 Mathematical Problem Formulation

Optimization Strategy 4   
3.1 Update of C and µ . 4   
3.2 Update of L . . 5

4 Convergence Analysis

5 Persistence-based Similarity Matrix

6 Numerical Experiments

7 Conclusion and Future Work

A Derivation of Feasibility Constraints on b

## 1 Introduction

In the big data era, the ability to extract meaningful patterns from complex, high-dimensional datasets is a cornerstone of data analysis. Unsupervised learning (clustering in particular) plays a pivotal role by uncovering hidden structures within data without the need for labeled examples. Among clustering algorithms, K-means [13, 10] remains a widely adopted method, owing to its simplicity, scalability, and broad empirical efectiveness. However, K-means and most of its variants rely fundamentally on Euclidean distances and implicitly assume that clusters are convex and roughly spherical in shape. This geometric bias limits their ability to recover clusters that are elongated, curved, or interleaved: configurations that arise routinely in real-world high-dimensional data [8].

This motivates a shift from a purely geometric approach toward a topological perspective in which the connectivity structure of the data becomes the primary object of analysis. Topology ofers a natural lan guage to formalize this intuition: topological methods analyze the shape of data by encoding structural properties such as connectedness and multi-scale organization, features that are invariant under con tinuous deformations and thus robust to data geometric distortions [17]. Incorporating such structural information directly into a clustering objective is thus a promising direction to overcome the limitations of purely metric-based methods [2, 20].

We propose a new clustering method called Topology-Aware Randomized Clustering (T-ARC) that pursues precisely this direction. The key idea of T-ARC is to assume that the data admits a latent graph structure, and to complement the standard K-means data-fidelity term with a graph-cut term [9], penalizing cluster assignments inconsistent with the connectivity structure of the data. This coupling efectively reveals this latent structure by promoting cluster boundaries that respect the topological organization of the data, rather than its metric arrangement alone, thereby correcting the geometric bias of K-means.

Most existing approaches that incorporate graph information into clustering rely on a similarity graph estimated once from the data and held fixed throughout the optimization [1, 20]. T-ARC departs from this paradigm by treating the latent graph itself as a variable: rather than regularizing against a fixed Laplacian, we model it as a random realization from a parametric family and optimize its distribution jointly with the cluster assignments, under a distributionally robust criterion that accounts for uncertainty in the graph itself.

Formalizing this idea in a tractable way is non-trivial. The connectivity graph over the data points is not known a priori, and the space of all admissible graphs is combinatorially intractable, due to its dimension. To address this, we model the latent graph as a random realization from a parametric Stochastic Block Model (SBM) [12], whose parameters are optimized within a Distributionally Robust Optimization (DRO) framework [11]. This formulation yields closed-form update rules, solved eficiently via a proximal point method [14], and ensures robustness to uncertainty in the graph distribution.

Central to the framework is a persistence-based similarity matrix S, constructed from the zerodimensional persistent homology of the data [6, 23]. It plays a twofold role: parametrizing the SBM edge-probability matrix and defining the center of the DRO ambiguity set, acting in both cases as a structural prior on the latent graph distribution. Unlike conventional similarity measures, which capture only local, pairwise relationships, S encodes the multiscale connectivity structure of the data — provably stable with respect to input perturbations [3] — and propagates it, through the graph-cut term, directly into the clustering objective. This approach instantiates the paradigm of Topological Machine Learning (TML) [7] in the context of clustering, where topological information enters not as an auxiliary signal but as a structural prior that shapes the optimization itself.

The overall optimization proceeds via Block Coordinate Descent (BCD) [22], alternating between updates of the cluster assignment matrix, the centroids, and the graph parameters. While the updates for the cluster assignments and centroids enjoy standard convexity properties, the stochastic nature of the graph update requires a dedicated analysis. We provide a convergence analysis based on a global Lyapunov functional: every block update, including the proximal update of the stochastic graph parameter, decreases its expected value, which is therefore convergent, while the successive diferences of all blocks vanish. In summary, this work makes the following contributions:

• A clustering framework (T-ARC) that corrects the geometric bias of K-means by embedding topological information into the optimization objective via a graph-cut term.

• A persistence-based similarity matrix grounded in the zero-dimensional persistent homology, serving as a structural prior that anchors both the SBM parametrization and the DRO ambiguity set to the multiscale connectivity structure of the data.

• A tractable optimization strategy that jointly learns the graph structure and cluster assignments by reformulating graph learning as DRO over SBM, leading to closed-form updates and convergence guarantees in expectation via a global Lyapunov analysis.

Finally, we validate T-ARC on synthetic datasets with complex geometric and topological structures and on random subsets of Fashion-MNIST [21], demonstrating competitive performance against established baselines.

The remainder of this paper is organized as follows. § 2 presents the mathematical formulation of the T-ARC framework, detailing the integration of the graph-cut term with the K-means objective. § 3 describes the block coordinate descent optimization strategy, including the closed-form updates for the cluster assignments, centroids, and the stochastic graph parameter via the proximal point method. § 4 provides the convergence analysis, establishing the descent in expectation of a global Lyapunov functional and the convergence of its expected value. § 5 defines the construction of the persistence-based similarity matrix derived from zero-dimensional persistent homology, which serves as the structural prior for the model. § 6 presents and discusses the experimental results on both synthetic datasets with complex topologies and random subsets of the Fashion-MNIST dataset. Finally, § 7 concludes the paper and outlines promising directions for future work. The feasibility constraints on the Stochastic Block Model parameter b are derived in Appendix A.

## 2 Mathematical Problem Formulation

Let $\pmb { X } = [ \pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n } ] ^ { \top } \in \mathbb { R } ^ { n \times d }$ be the data matrix, where each row $\pmb { x } _ { i } \in \mathbb { R } ^ { d }$ represents a data point, and n is the number of samples. Our goal is to partition the dataset $\{ \pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n } \}$ into k disjoint clusters.

A standard approach is to solve the K-means optimization problem [5], but this formulation is purely geometric: it assumes convex, spherical clusters and relies on local, pairwise information, remaining blind to the global connectivity structure of the data. This limits its applicability when clusters are elongated, curved, or interconnected, scenarios where topological information becomes crucial.

To overcome this limitation, we enrich the geometric perspective with a topological one. Specifically, we assume the dataset is endowed with an underlying latent topological structure, which we approximate by an undirected and unweighted graph $G = ( V , E )$ . In $G ,$ each node corresponds to a data point ${ \mathbf { } } x _ { i } ,$ and the edge set E encodes adjacency or similarity relations. In order to find this underlying latent topological structure, our approach treats the graph itself as a variable: we aim to discover, among all possible graphs on the n vertices—a set that we will denote with G—the specific configuration that best supports the clustering objective. We thus augment the K-means objective with a graph-cut term, obtaining the following combined regularized optimization problem:

$$
\operatorname* { m i n } _ { C , \mu , L } F ( C , \mu , L ) = \underbrace { \frac { 1 } { 2 } \mathrm { T r } \big ( C ^ { \top } L C \big ) } _ { \mathrm { G r a p h - c u t } } + \underbrace { \frac { \lambda _ { \mu } } { 2 } \| X - C \mu \| _ { F } ^ { 2 } } _ { \mathrm { K - m e a n s } } \quad \mathrm { ~ s . t . ~ } \quad C \in \mathbb { R } _ { + } ^ { n \times k } , \mu \in \mathbb { R } _ { + } ^ { k \times d } , L \in \mathcal { L } _ { \mathcal { G } } ,\tag{1}
$$

where $\lambda _ { \mu } > 0$ balances the two contributions and

$$
{ \mathcal { L } } _ { \mathcal { G } } = \left\{ L = D - A \left| { \begin{array} { l } { A \in \{ 0 , 1 \} ^ { n \times n } , A = A ^ { \top } , \operatorname { d i a g } ( A ) = 0 { \mathrm { ~ a d j a c e n c y ~ m a t r i x ~ o f ~ } } G } \\ { D = \operatorname { d i a g } ( A \mathbf { 1 } _ { n } ) { \mathrm { ~ d e g r e e ~ m a t r i x ~ o f ~ } } G { \mathrm { ~ a n d } } } \\ { G = ( V , E ) \in { \mathcal { G } } } \end{array} } \right. \right\}
$$

denotes the set of Laplacians representing all such graphs. The two terms play complementary roles, reflecting the two perspectives we aim to balance:

• Graph-cut term: This is the topological driver of the model. By coupling the cluster assignment matrix C with the graph Laplacian $L ,$ it penalizes assignments that cut across the connectivity structure of G. It biases the solution toward clusters that respect the underlying topological organization.

• K-means term: This acts as a geometric regularizer. It ensures that clusters remain compact in the original feature space, preventing the graph-cut term from producing degenerate or overly fragmented partitions. It accounts for intra-cluster variance without explicitly using graph information, thereby balancing topological coherence with geometric fidelity.

In essence, the geometric perspective (data as points in space) enters in the clustering task through the K-means term, while the topological perspective (a latent graph on the data) enters through the graph-cut term. Their interplay, modulated by $\lambda _ { \mu } .$ , defines a trade-of where the optimal clustering must simultaneously minimize intra-cluster variance in the ambient space and respect the connectivity structure encoded in the graph.

## 3 Optimization Strategy

The convexity properties of the objective function with respect to each variable motivate the optimization strategy adopted in this work. In particular:

• The graph-cut term is a quadratic form in C. As the graph Laplacian L is positive semi-definite (PSD), this term is convex in $C .$

• The K-means term is non-convex in the joint pair $( C , \mu )$ due to the bilinear structure of $C \mu$ . However, it is block-convex in each variable separately.

• Regarding the constraints, the non-negativity sets $C \geq 0$ and $\mu \geq 0$ are convex. The set $\mathcal { L } _ { \mathcal { G } }$ of graph Laplacians is non-convex.

Thus, the full optimization problem is jointly non-convex, due to both the bilinear term $C \mu$ and the non-convex set $\mathcal { L } _ { \mathcal { G } }$ . The problem is block-wise convex, meaning that when all other variables are held fixed, the optimization over each remaining variable is convex, so it is natural to use BCD, in which the variables are updated sequentially:

• Update $C \colon C _ { k + 1 } = \operatorname * { a r g m i n } _ { C \geq 0 } \ F ( C , \mu _ { k } , L _ { k } )$

• Update $\mu : \mu _ { k + 1 } = \underset { \pmb { \mu } \geq 0 } { \mathrm { a r g m i n } } F ( \pmb { C } _ { k + 1 } , \pmb { \mu } , \pmb { L } _ { k } )$

• Update L: $\pmb { L } _ { k + 1 } = \underset { \pmb { L } \in \mathcal { L } _ { \mathcal { G } } } { \mathrm { a r g m i n } } \ F ( \pmb { C } _ { k + 1 } , \pmb { \mu } _ { k + 1 } , \pmb { L } )$

In the following, we will write $F ( C ) , F ( \pmb { \mu } )$ , and $F ( L )$ to denote the objective function restricted to each block of variables, with the remaining blocks held fixed at their current values.

## 3.1 Update of C and $\mu$

For fixed $\pmb { \mu }$ and L, the optimization of C reduces to a regularized Non-Negative Least Squares (NNLS):

$$
\begin{array} { r } { \underset { C \geq 0 } { \operatorname* { m i n } } F ( C ) = \frac { 1 } { 2 } \mathrm { T r } ( C ^ { \top } L C ) + \frac { \lambda _ { \mu } } { 2 } \| X - C \mu \| _ { F } ^ { 2 } . } \end{array}
$$

This subproblem is solved by the projected gradient step $C _ { k + 1 } = [ C _ { k } - \alpha \nabla _ { C } F ( C ) ] _ { + }$ , where the projection $[ \cdot ] _ { + }$ is applied elementwise and denotes projection onto the nonnegative orthant. The gradient $\nabla _ { C } F ( C ) =$ ${ \cal L } C + \lambda _ { \mu } ( C \mu \mu ^ { \top } - X \mu ^ { \top } )$ has Lipschitz constant $\| \pmb { L } \| _ { 2 } + \lambda _ { \mu } \| \pmb { \mu } \pmb { \mu } ^ { \top } \| _ { 2 }$ . This step is iterated, forming an inner loop, which is terminated when either the maximum number of inner iterations is reached, or the relative change of the objective F between two consecutive inner iterations falls below a prescribed tolerance.

When C is fixed, the optimization over µ reduces to the NNLS min $\stackrel { ! } { \mu } \geq 0 F ( \pmb { \mu } ) = \frac 1 2 \| \pmb { X } - \pmb { C } \pmb { \mu } \| _ { F } ^ { 2 }$ . Again, we use the projected gradient iteration $\pmb { \mu } _ { k + 1 } = [ \pmb { \mu } _ { k } - \alpha \nabla _ { \mu } F ( \pmb { \mu } ) ] -$ <sub>+</sub> with gradient $\nabla _ { \mu } \bar { F ( \mu ) } = C ^ { \top } C \mu - C ^ { \top } X$ and Lipschitz constant $\| C ^ { \top } C \| _ { 2 }$ . The same stopping rule and maximum number of inner iterations are used for this update.

## 3.2 Update of L

The set $\mathcal { L } _ { \mathcal { G } }$ is non-convex, which precludes the use of standard convex optimization methods. Furthermore, the space of all possible graphs on n nodes has cardinality $2 ^ { \binom { n } { 2 } }$ , making any explicit search over $\mathcal { L } _ { \mathcal { G } }$ computationally infeasible for moderately large n.

To address this, we replace the deterministic optimization over $\mathcal { L } _ { \mathcal { G } }$ by a stochastic approximation aimed at recovering the latent graph topology. Rather than optimizing over all possible Laplacians, we restrict attention to a parametric family of random graphs generated by SBM. The Laplacian update then proceeds in two stages: a deterministic optimization of the model parameters, followed by a stochastic realization of the graph according to the SBM.

Stochastic Block Model (SBM) [12] Stochastic Block Models are a classical family of probabilistic models for random graphs, in which the probability of an edge between two nodes depends on the community membership of those nodes. We adopt a similarity-modulated variant: for n nodes, the probability of an edge between nodes i and $j$ is:

$$
P ( A _ { i j } = 1 ) = B _ { i j } ( a , b ) = { \left\{ \begin{array} { l l } { a S _ { i j } } & { { \mathrm { i f ~ } } x _ { i } , x _ { j } { \mathrm { ~ a r e ~ i n ~ t h e ~ s a m e ~ c l u s t e r } } } \\ { b S _ { i j } } & { { \mathrm { i f ~ } } x _ { i } , x _ { j } { \mathrm { ~ a r e ~ i n ~ d i f f e r e n t ~ c l u s t e r s } } } \end{array} \right. }
$$

where $a > 0 , b \geq 0$ , and $S \in [ 0 , 1 ] ^ { n \times n }$ is a similarity matrix. The adjacency matrix A is sampled entry-wise as

$$
A _ { i j } \sim \mathrm { B e r } \big ( B _ { i j } ( a , b ) \big ) , \quad i \neq j ,
$$

with no self-loops, where $\operatorname { B e r } ( p )$ denotes the Bernoulli distribution with success probability p. The associated random Laplacian is $L = D - A$ , where $D = \mathrm { D i a g } ( A \mathbf { 1 } _ { n } )$ . Under this construction, the Laplacian is a random variable parameterized by $\theta = ( a , b )$

Distributionally Robust Optimization (DRO) [11] In classical stochastic optimization, decision making under uncertainty is formulated as the problem of minimizing the expected cost under a fixed distribution: min $_ { x \in \mathcal { X } } \mathbb { E } _ { P } [ f ( x , \xi ) ]$ , where x is the decision variables, $\xi$ is a random variables, and $P$ is the probability distribution governing the uncertainty. In practice, $P$ is rarely known and must be estimated from data; inaccurate estimation may lead to solutions that perform poorly when the true distribution deviates from the estimate. DRO addresses this by replacing the single assumed distribution with an ambiguity set B of plausible distributions:

$$
\operatorname* { m i n } _ { x \in \mathcal { X } } \operatorname* { m a x } _ { P \in \mathcal { B } } \mathbb { E } _ { P } [ f ( x , \xi ) ] .
$$

This seeks decisions that minimize the worst-case expected cost, yielding solutions robust to distributional uncertainty.

In our setting, the exact distribution governing the random graph is unknown. We adopt a DRO perspective and, restricting to the SBM parametric family, we define the ambiguity set as

$$
\mathcal { B } = \left\{ B ( a , b ) = a S + b ( \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } - S ) \mid a > 0 , b \geq 0 , \| B ( a , b ) - S \| _ { F } \leq r , B ( a , b ) \mathbf { 1 } _ { n } = \mathbf { 1 } _ { n } \right\}\tag{2}
$$

that is, the collection of SBM edge-probability matrices within distance r of the empirical similarity matrix S, and optimize against the worst-case expectation over $B .$

Let $\pmb { T } = \pmb { C } \pmb { C } ^ { \top }$ . The Laplacian update is formulated as

$$
\operatorname* { m i n } _ { \theta = \left( a , b \right) } \operatorname* { m a x } _ { B \in { \mathcal { B } } } \mathbb { E } _ { \theta } [ \langle L ( \theta ) , T \rangle ] .
$$

Since the objective is linear in $\textstyle B ( a , b )$ and the ambiguity set B imposes proximity constraints on $\textstyle B ( a , b )$ , the worst-case over B translates into explicit bounds on the scalar parameter $b ,$ derived in Appendix A. By the linearity of expectation, the objective then reduces to

$$
\operatorname* { m i n } _ { b \in [ 0 , b _ { \operatorname* { m a x } } ] } \langle \mathbb { E } _ { b } [ \pmb { L } ] , \pmb { T } \rangle\tag{3}
$$

where the expected Laplacian admits a closed-form expression. Under the SBM, each entry of A is Bernoulli, so

$$
\begin{array} { r c l } { B ( a , b ) = B ( b ) ^ { 1 } = \mathbb { E } _ { \theta } [ A ] } & { = } & { a S + b ( \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } - S ) , \quad \mathrm { w i t h ~ } a = 1 + b ( 1 - n ) , } \\ & { = } & { ( 1 - n b ) S + n b \frac { 1 } { n } \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } } \\ & { = } & { \operatorname { C o n v } _ { b } \left( S , \frac { 1 } { n } \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } \right) } \end{array}\tag{4}
$$

where S encodes the community structure. This shows that $B ( b )$ lies on the line through S and $\textstyle { \frac { 1 } { n } } \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top }$ , and in their convex hull, denoted $\mathrm { C o n v } _ { b } ( \cdot , \cdot )$ , whenever $b \leq 1 / n$ . Over the whole admissible range of $b ,$ the entries $B _ { i j } = a S _ { i j } + b ( 1 - S _ { i j } )$ remain in [0, 1], being convex combinations of the scalars $a , b \in [ 0 , 1 ]$

The expected degree matrix is $\mathbb { E } _ { \boldsymbol { \theta } } [ \boldsymbol { D } ] = \mathrm { d i a g } ( \boldsymbol { B } \mathbf { 1 } _ { n } )$ . Combining these, the expected Laplacian is

$$
\begin{array} { r } { \mathbb { E } _ { \theta } [ L ] \ = \ \mathrm { D i a g } \big ( B ( b ) \mathbf { 1 } _ { n } \big ) - B ( b ) \ \stackrel { ( 4 ) } { = } \ \mathrm { D i a g } \big ( \big [ ( 1 - n b ) S + b \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } \big ] \mathbf { 1 } _ { n } \big ) - ( 1 - n b ) S - b \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } . } \end{array}\tag{5}
$$

Assuming a row-stochastic similarity matrix, i.e., $S \mathbf { 1 } _ { n } = \mathbf { 1 } _ { n }$ , one computes $B ( b ) \mathbf { 1 } _ { n } = \mathbf { 1 } _ { n }$ , so the expected degree matrix is constant: $\mathrm { D i a g } ( { \cal B } ( b ) { \bf 1 } _ { n } ) = I$ . The objective function in (3) becomes

$$
\Phi ( b ) = \langle I , \mathbf { T } \rangle - ( 1 - n b ) \langle S , \pmb { T } \rangle - b \langle \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } , \pmb { T } \rangle = \alpha - \gamma b ,\tag{6}
$$

where $\alpha : = \langle I , T \rangle - \langle S , T \rangle$ and $\gamma : = \langle \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } , \pmb { T } \rangle - n \langle \pmb { S } , \pmb { T } \rangle$ are constants independent of b. This is strictly linear in b.

The admissible values of the parameter b (derived in Appendix A) satisfy

$$
\begin{array} { r } { 0 \leq b \leq b _ { \mathrm { m a x } } : = \operatorname* { m i n } \Bigl \{ \frac { 1 } { n - 1 } , \frac { r } { n \sqrt { S _ { 2 } - 1 } } \Bigr \} , } \end{array}\tag{7}
$$

where $\begin{array} { r } { S _ { 2 } : = \sum _ { i , j } S _ { i j } ^ { 2 } } \end{array}$ , thus the final objective is

$$
\operatorname* { m i n } _ { b \in [ 0 , b _ { \mathrm { m a x } } ] } \Phi ( b ) .
$$

Proximal Point Method [14] on b Since $\Phi ( b )$ in (6) is linear, its unconstrained minimization always attains a boundary of the feasible interval, leading to a degenerate solution. To promote stable interior updates, we use a proximal point approach:

$$
\begin{array} { r } { b _ { k + 1 } = \underset { b \in [ 0 , b _ { \operatorname* { m a x } } ] } { \mathrm { a r g m i n } } \left\{ \Phi ( b ) + \frac { 1 } { 2 \tau } ( b - b _ { k } ) ^ { 2 } \right\} , } \end{array}\tag{8}
$$

where $\tau > 0$ is a stepsize. Setting the derivative of the regularized objective to zero gives the unconstrained update $b _ { k + 1 } ^ { \mathrm { u n p r o j } } = b _ { k } + \tau \gamma$ , and projecting onto the feasible interval yields

$$
b _ { k + 1 } = \operatorname* { m i n } \left( \operatorname* { m a x } ( b _ { k } + \tau \gamma , 0 ) , b _ { \operatorname* { m a x } } \right) .
$$

This update has a clear interpretation: when $\gamma > 0$ , the objective decreases as b increases, so the iterate is pushed toward $b _ { \mathrm { m a x } } ;$ when $\gamma < 0$ it moves toward 0. The proximal term ensures controlled updates and keeps $b _ { k + 1 }$ within the admissible interval.

## 4 Convergence Analysis

The BCD scheme is interpreted as a discrete dynamical system on $( C , \mu , b )$ . To analyze its stability and convergence, we introduce a global Lyapunov functional ${ \mathcal { I } } ( C , \mu , b )$ , whose monotonic descent along the BCD iterations drives convergence [4]:

$$
\mathcal { I } ( C , \mu , b ) = \underbrace { \mathrm { T r } ( C ^ { \top } L ( b ) C ) } _ { \mathrm { G r a p h ~ D i r i c h l e t ~ E n e r g y } } + \underbrace { \lambda _ { \mu } \| X - C \mu \| _ { F } ^ { 2 } } _ { \mathrm { D a t a ~ F i d e l i t y } } ,\tag{9}
$$

• Graph Dirichlet Energy. In the continuous setting, the Dirichlet energy $\begin{array} { r } { E [ u ] \ = \ \int _ { \Omega } \| \nabla u \| ^ { 2 } } \end{array}$ dx measures the total variation of $u : \Omega \to \mathbb { R }$ : small values indicate a smooth, slowly varying function, while large values signal rapid oscillations. Its discrete analogue on a weighted graph with Laplacian $\pmb { L } = \pmb { D } - \pmb { A } \ \mathrm { i s } \ \mathrm { T r } ( \pmb { C } ^ { \top } \pmb { L } \pmb { C } )$ . In our stochastic setting, the graph topology is uncertain: replacing L with the expected Laplacian $\pmb { L } ( b ) = \mathbb { E } _ { b } [ \pmb { L } ( B ) ]$ defined in (5) yields a smoothness term robust to edge-level noise, encouraging cluster assignments stable across likely graph realizations.

• Data Fidelity. This term ensures that the cluster centers C remain close to the observed data X, with $\lambda _ { \mu }$ controlling the balance between smoothness and data fit.

Lower Boundedness The functional J is bounded from below.

• The data-fidelity term is non-negative by definition.

• If $b \geq 0$ , then $B ( b )$ has non-negative entries, and it is symmetric since S is. The Laplacian quadratic form satisfies

$$
\begin{array} { r } { \mathrm { T r } ( C ^ { \top } { \cal L } ( b ) C ) = \frac { 1 } { 2 } \sum _ { k } \sum _ { i , j } B _ { i j } ( b ) ( C _ { i k } - C _ { j k } ) ^ { 2 } \geq 0 . } \end{array}
$$

Thus $\mathcal { I } ( C , \pmb { \mu } , b ) \geq 0$ , and the sequence generated by the algorithm cannot diverge to −∞.

Monotonic Descent under BCD Since each block update is performed via projected gradient descent with a step size equal to the inverse Lipschitz constant (§ 3.1), the standard descent lemma for smooth functions gives:

Updating C with (µ, b) fixed:

Updating µ with (C, b) fixed:

$$
\begin{array} { r l } & { \mathcal { I } ( C _ { k + 1 } , \mu _ { k } , b _ { k } ) + \frac { L _ { C } } { 2 } \| C _ { k + 1 } - C _ { k } \| _ { F } ^ { 2 } \leq \mathcal { I } ( C _ { k } , \mu _ { k } , b _ { k } ) , } \\ & { \mathcal { I } ( C _ { k + 1 } , \mu _ { k + 1 } , b _ { k } ) + \frac { L _ { \mu } } { 2 } \| \mu _ { k + 1 } - \mu _ { k } \| _ { F } ^ { 2 } \leq \mathcal { I } ( C _ { k + 1 } , \mu _ { k } , b _ { k } ) . } \end{array}
$$

Expected Descent in b Updating b while keeping $( C , \mu )$ fixed requires special care due to the stochastic nature of the graph update; the monotonic decrease $\mathcal { I } ( C _ { k + 1 } , \mu _ { k + 1 } , b _ { k + 1 } ) \leq \mathcal { I } ( C _ { k + 1 } , \mu _ { k + 1 } , b _ { k } )$ does not hold pathwise, we instead analyze the expected energy.<sup>2</sup> The key observation is that, in expectation, J depends on b only through the function Φ in (6), which is exactly the function minimized by the proximal update (8).

Theorem 1 (Expected descent w.r.t. the graph parameter). Fix a row-stochastic $S \in \mathbb { R } ^ { n \times n } , C _ { k + 1 }$ and $\pmb { \mu } _ { k + 1 }$ , define the Lyapunov energy J as in (9), and assume that the graph Laplacian $\pmb { L } ( b )$ is generated according to an SBM. Let $b _ { k + 1 }$ be given by the proximal update (8) with $\pmb { T } = \pmb { C } _ { k + 1 } \pmb { C } _ { k + 1 } ^ { \top }$ . Then

$$
\begin{array} { r } { \mathbb { E } \big [ \mathcal { I } ( C _ { k + 1 } , \pmb { \mu } _ { k + 1 } , b _ { k + 1 } ) \big ] + \frac { 1 } { 2 \tau } ( \Delta b _ { k } ) ^ { 2 } \leq \mathbb { E } \big [ \mathcal { I } ( C _ { k + 1 } , \pmb { \mu } _ { k + 1 } , b _ { k } ) \big ] , } \end{array}
$$

where $\Delta b _ { k } : = b _ { k + 1 } - b _ { k }$

Proof. The only random term in J is the Laplacian contribution. By linearity of trace and expectation:

$$
\begin{array} { r } { \mathbb E \big [ \mathrm { T r } ( \pmb { C } ^ { \top } \pmb { L } ( b ) \pmb { C } ) \big ] = \mathrm { T r } \big ( \pmb { C } ^ { \top } \mathbb E [ \pmb { L } ( b ) ] \pmb { C } \big ) = \langle \mathbb E [ \pmb { L } ( b ) ] , \pmb { C } \pmb { C } ^ { \top } \rangle . } \end{array}
$$

For $\pmb { C } = \pmb { C } _ { k + 1 } , \mathrm { i . e . , } \pmb { T } = \pmb { C } _ { k + 1 } \pmb { C } _ { k + 1 } ^ { \top }$ , this is exactly Φ(b) in (6). The data-fidelity term does not depend on b, hence

$$
\begin{array} { r } { \mathbb { E } \big [ \mathcal { I } ( C _ { k + 1 } , \pmb { \mu } _ { k + 1 } , b ) \big ] = \Phi ( b ) + \mathrm { c o n s t . } } \end{array}
$$

Since $b _ { k + 1 }$ minimizes $\Phi ( b ) + { \textstyle \frac { 1 } { 2 \tau } } ( b - b _ { k } ) ^ { 2 }$ over $[ 0 , \boldsymbol { b } _ { \mathrm { m a x } } ]$ and $b _ { k } \in [ 0 , { b _ { \operatorname* { m a x } } } ]$ is feasible, comparing the objective at $b _ { k + 1 }$ and at $b _ { k }$ (where the proximal term vanishes) gives

$$
\begin{array} { r } { \Phi ( b _ { k + 1 } ) + \frac { 1 } { 2 \tau } ( \Delta b _ { k } ) ^ { 2 } \leq \Phi ( b _ { k } ) , } \end{array}
$$

which is the claim.

The theorem shows that, in expectation, the stochastic graph update does not increase the Lyapunov functional, and that the decrease is at least quadratic in the step size.

Total Expected Descent Combining the deterministic descent inequalities for C and $\pmb { \mu }$ with Theorem 1, we obtain the final combined inequality<sup>3</sup>:

$$
\begin{array} { r } { \mathbb { E } \big [ \mathcal { I } _ { k + 1 } \big ] + \frac { 1 } { 2 \tau } ( \Delta b _ { k } ) ^ { 2 } + \frac { L _ { \mu } } { 2 } \| \mu _ { k + 1 } - \mu _ { k } \| _ { F } ^ { 2 } + \frac { L _ { C } } { 2 } \| C _ { k + 1 } - C _ { k } \| _ { F } ^ { 2 } \ \leq \ \mathbb { E } \big [ \mathcal { I } _ { k } \big ] , } \end{array}\tag{10}
$$

where $\mathcal { T } _ { k } : = \mathcal { I } ( C _ { k } , \mu _ { k } , b _ { k } )$ . Hence $( \mathbb { E } [ \mathcal { T } _ { k } ] ) _ { k }$ is nonincreasing and, being bounded below by 0, convergent. Moreover, summing (10) over k gives

$$
\begin{array} { r } { \displaystyle \sum _ { k \geq 0 } \Bigl ( \frac { 1 } { 2 \tau } ( \Delta b _ { k } ) ^ { 2 } + \frac { L _ { \mu } } { 2 } \| \mu _ { k + 1 } - \mu _ { k } \| _ { F } ^ { 2 } + \frac { L _ { C } } { 2 } \| C _ { k + 1 } - C _ { k } \| _ { F } ^ { 2 } \Bigr ) \leq \mathbb { E } [ \mathcal { I } _ { 0 } ] < \infty , } \end{array}
$$

so that the successive diferences of all blocks vanish as $k  \infty$

## 5 Persistence-based Similarity Matrix

The framework developed above operates on a generic similarity matrix S, which simultaneously parametrizes the SBM edge-probability matrix and defines the center of the DRO ambiguity set. The quality of this prior is therefore consequential: it determines both the graph distribution from which the latent graph is drawn and the region of distributional uncertainty over which the optimization is robust. We propose to instantiate $\pmb { S }$ from the zero-dimensional persistent homology of the data, grounding both roles in the multiscale connectivity structure of the data.

Let $\{ K _ { \varepsilon } \} _ { \varepsilon \ge 0 }$ be a finite increasing filtration of simplicial complexes on $\{ \pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n } \}$ . The inclusions $K _ { \varepsilon } \hookrightarrow K _ { \varepsilon ^ { \prime } }$ , for $\varepsilon \ \leq \ \varepsilon ^ { \prime }$ , induce linear maps $( i _ { \varepsilon , \varepsilon ^ { \prime } } ) _ { 0 * } : H _ { 0 } ( K _ { \varepsilon } ) \to H _ { 0 } ( K _ { \varepsilon ^ { \prime } } )$ between the zeroth simplicial homology groups, yielding the zero-dimensional persistent homology as the persistence module $\{ H _ { 0 } ( K _ { \varepsilon } ) , ( i _ { \varepsilon , \varepsilon ^ { \prime } } ) _ { 0 * } \} _ { \varepsilon \leq \varepsilon ^ { \prime } }$ . The complete record of the birth and death of connected components across the filtration is encoded in the $H _ { 0 }$ barcode. For each pair $i , j$ define

$$
\varepsilon _ { i j } : = \operatorname* { i n f } \{ \varepsilon \geq 0 : [ \pmb { x } _ { i } ] _ { \varepsilon } = [ \pmb { x } _ { j } ] _ { \varepsilon } \mathrm { ~ i n ~ } H _ { 0 } ( K _ { \varepsilon } ) \}
$$

as the death time of the $H _ { 0 }$ barcode interval corresponding to the merger of the connected components containing $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ and $\mathbf { \boldsymbol { x } } _ { j }$ . By the Structure Theorem [23], the $H _ { 0 }$ barcode is the unique decomposition of the persistence module into interval modules, and $\varepsilon _ { i j }$ is read of unambiguously as the right endpoint of the interval corresponding to the merger of $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ and $\boldsymbol { x } _ { j } ;$ hence $\varepsilon _ { i j }$ is a well-defined invariant of the persistence module. The persistence-based similarity matrix $S \in [ 0 , 1 ] ^ { n \times n }$ is then defined as

$$
S _ { i j } : = 1 - \frac { \varepsilon _ { i j } } { \varepsilon _ { \operatorname* { m a x } } } , \quad \varepsilon _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { i , j } \varepsilon _ { i j } { } ^ { 4 } .
$$

Observation. S is a well-defined similarity matrix: $S _ { i i } = 1$ since $\varepsilon _ { i i } = 0$ , as each point belongs to its own connected component at $\varepsilon = 0 ;$ symmetry follows from $\varepsilon _ { i j } = \varepsilon _ { j i } ,$ since component merger is a symmetric event; and $S _ { i j } \in [ 0 , 1 ]$ by definition of ε<sub>max</sub>.

This construction is the natural choice for the role $\pmb { S }$ plays in the framework for two reasons. First, the similarity encoded in $_ { s }$ is topological: $S _ { i j }$ is large precisely when $\mathbf { \mathcal { x } } _ { i }$ and $\mathbf { \boldsymbol { x } } _ { j }$ merge into the same connected component at an early filtration scale, reflecting the global multiscale connectivity structure of the data rather than proximity at any fixed scale. This aligns directly with the graph-cut regularization, which promotes cluster assignments that respect this connectivity. Second, the Stability Theorem for persistent homology [3] guarantees that the $H _ { 0 }$ barcode varies continuously with the input data in the bottleneck distance, which implies that $\varepsilon _ { i j }$ , and hence $S ,$ inherit this stability: small perturbations of the data produce small perturbations in S. By feeding S into both the SBM parametrization and the DRO ambiguity set, the graph-learning process is anchored to the multiscale connectivity structure of the data, and this structure is propagated, through the graph-cut term, directly into the clustering objective.

$^ 3 \mathrm { B y }$ Theorem 1, $\begin{array} { r } { \mathbb { E } \big [ \mathcal { I } ( C _ { k + 1 } , \pmb { \mu } _ { k + 1 } , b _ { k + 1 } ) \big ] + \frac { 1 } { 2 \tau } ( \Delta b _ { k } ) ^ { 2 } \leq \mathbb { E } \big [ \mathcal { I } ( C _ { k + 1 } , \pmb { \mu } _ { k + 1 } , b _ { k } ) \big ] } \end{array}$ for all k. Then the standard gradient descent lemma for smooth functions with Lipschitz constants $L _ { C }$ and $L _ { \mu }$ gives

$$
\begin{array} { r } { \mathcal { I } ( C _ { k + 1 } , \mu _ { k + 1 } , b _ { k } ) + \frac { L _ { \mu } } { 2 } \| \mu _ { k + 1 } - \mu _ { k } \| _ { F } ^ { 2 } + \frac { L _ { C } } { 2 } \| C _ { k + 1 } - C _ { k } \| _ { F } ^ { 2 } \leq \mathcal { I } ( C _ { k } , \mu _ { k } , b _ { k } ) . } \end{array}
$$

## 6 Numerical Experiments

To validate T-ARC, we conducted experiments on both synthetic and real datasets. Five synthetic datasets were designed to evaluate the method under controlled conditions, each highlighting specific structural properties. Additionally, we assessed performance on random subsets of Fashion-MNIST [21], providing a more realistic benchmark in a high-dimensional setting.

```latex
Algorithm 1: T-ARC
Input: Data matrix $\boldsymbol { X } \in \mathbb { R } ^ { n \times d } ,$ , number of clusters k, parameter $\lambda _ { \mu } ,$ ambiguity radius r
1 Initialize $C \geq 0 , \mu \geq 0$ . Initialize b in (7). Construct persistence-based similarity matrix $\pmb { S }$ as in
Section 5.
2 Apply symmetric Sinkhorn–Knopp normalization to S, so that it is simultaneously
row-stochastic and symmetric.
3 for $t = 0 , 1 , 2 , \ldots$ do
4 Batch update C: Run projected gradient steps on C minimizing $F ( C , { \pmb \mu } ^ { ( t ) } , { \pmb L } ^ { ( t ) } )$ , until a
stopping criterion is met.
5 Batch update µ: Run projected gradient steps on µ minimizing $F ( C ^ { ( t + 1 ) } , \mu , L ^ { ( t ) } )$ , until a
stopping criterion is met.
6 Proximal update on b: Compute $b ^ { ( t + 1 ) }$ as (8), set $a ^ { ( t + 1 ) } = 1 + b ^ { ( t + 1 ) } ( 1 - n )$
7 Monte Carlo Laplacian update: Construct cluster-aware probabilities B from $C ^ { ( t + 1 ) }$
sample m<sub>MC</sub> adjacency matrices $A ^ { ( m ) } \sim \operatorname { B e r } ( B )$ , compute mean ${ \overline { { A } } } ,$ threshold to binary, set
$\pmb { { \cal L } } ^ { ( t + 1 ) } = \mathrm { d i a g } ( \bar { \bf { A 1 } } ) - \bar { \bf { A } } .$
8 end
9 return Cluster assignment matrix $C ^ { ( t + 1 ) }$ , centroids $\pmb { \mu } ^ { ( t + 1 ) }$
```

The core optimization algorithm is described in Algorithm 1. Specifically it firstly applies symmetric Sinkhorn–Knopp normalization [18] to the input similarity matrix S. This normalization iteratively replaces S with $\bar { D ^ { - 1 / 2 } } S D ^ { - 1 / 2 }$ , where $D = \mathrm { d i a g } ( S \mathbf { 1 } _ { n } )$ , until convergence, guaranteeing that S is simultaneously row-stochastic and symmetric before entering the BCD loop, ensuring the theoretical assumption. This normalization step is independent of the choice of S, allowing the framework to be instantiated with any chosen similarity matrix.

As baseline methods, we consider classical clustering approaches spanning diferent modeling assump tions. In particular, we include the standard K-means algorithm, which provides a natural reference point since T-ARC can be interpreted as a topology-aware extension of K-means. We further consider spherical K-means, which is better suited for data where angular similarity is more informative than Euclidean distance, and spectral clustering, which leverages graph-based representations and is therefore closely related to our similarity-driven framework. Moreover, comparing T-ARC with its variant employing a similarity matrix based on the Euclidean norm allows for a direct assessment of the impact of the persistence-based construction. This comparison isolates the contribution of topological information, highlighting whether incorporating persistence-based connectivity leads to improved clustering performance. The experimental design highlights two complementary aspects. First, on simple synthetic datasets, we verify that the proposed method behaves consistently with classical approaches when the cluster structure is well separated. Second, on more challenging settings — synthetic datasets characterized by nonlinear geometries or latent topological structures — we demonstrate that the proposed framework captures relationships that cannot be properly modeled by purely geometric clustering methods such as K-means. In addition, on random subsets of Fashion-MNIST, we assess its behavior on real high-dimensional data. The proposed algorithm was implemented in MATLAB. Synthetic dataset generation and computation of the persistence-based similarity matrices were carried out in Python using the GUDHI library [19]. Baseline comparisons include classical K-means, Spherical K-means, and Spectral clustering and are performed in MATLAB.

Initialization of variables The assignment matrix $C _ { 0 }$ is generated as a random matrix of size $n \times k .$ normalized row-wise to ensure each row sums to one (with a regularization term of 10<sup>−14</sup> to avoid division by zero). Initial centroids are $\pmb { \mu } _ { 0 } = ( \pmb { C } _ { 0 } ^ { \top } \pmb { C } _ { 0 } ) ^ { - 1 } \pmb { C } _ { 0 } ^ { \top } \pmb { X }$ , rectified to enforce non-negativity. The parameter $b _ { 0 }$ is sampled uniformly in $[ 0 , \boldsymbol { b } _ { \mathrm { m a x } } ]$ (from (7)), and $a _ { 0 } = \operatorname* { m a x } ( 1 0 ^ { - 3 } , 1 + b _ { 0 } ( 1 - n ) )$ ). The initial Laplacian $\scriptstyle { L _ { 0 } }$ is constructed via Monte Carlo from the initialized clusters.

General Settings The behavior of T-ARC is controlled by several key parameters. The regularization parameter is $\lambda _ { \mu } = 0 . 0 1$ , emphasizing the topological structure over the K-means component. The Frobenius radius was chosen as $r = 0 . 0 1$ , which defines the feasible region for the optimization of the probability variable. Monte Carlo sampling uses $m _ { \mathrm { M C } } = 3 0$ samples. For the stopping condition, the global stopping threshold is $\epsilon = 1 0 ^ { - 6 }$ ; inner batches use $\epsilon = 1 0 ^ { - 3 }$ . The proximal step-size is $\tau = 1 0 ^ { - 2 }$ . The random seed is fixed to 55 for reproducibility. For the persistence-based similarity matrix, the Vietoris-Rips filtration [16] is used. The resulting similarity matrices for all five experiments are shown in Fig. 1.

![](images/e3f91fcb1588063eb902b1ae9e71d8283dd6f9529dd4b3c283a4c5fed8fff9ff.jpg)  
Figure 1: Persistence-based similarity matrices for the five experimental settings.

Evaluation Metrics We report external and internal clustering validity indices. As external metrics we consider Accuracy and F1-score measure label agreement against ground-truth. For internal metric, we use the Silhouette coeficient [15], defined as the average over all points of

$$
s ( i ) = \frac { b ( i ) - a ( i ) } { \operatorname* { m a x } \{ a ( i ) , b ( i ) \} } ,
$$

where $a ( i )$ is the mean intra-cluster Euclidean distance and $b ( i )$ is the minimum mean Euclidean distance to any other cluster. Values lie in [−1, 1], with higher values indicating better clustering. The combination of internal and external metrics provides a comprehensive assessment of clustering performance, capturing both label agreement and intrinsic cluster structure.

Example 1: Two Sets Dataset We generated a two-dimensional point cloud of 160 points divided equally into two clusters, centered at (0.3, 0.3) and (0.7, 0.7) within the unit square. The distance between the centers is fixed $( d \approx 0 . 5 7 )$ , while the isotropic Gaussian noise is set to $\sigma = 0 . 0 5$ , 0.12 and 0.20, so that the two sets are well separated, closer and overlapping, respectively, as shown in Fig. 2.

![](images/1eb71d2c1178c3a4d39d3f99008befc5f9541aa3627682439a6f3fb34d645fcf.jpg)

![](images/86985bbba28207dfd81937848321f523ee92857442e9d4a941ae32a79730d47d.jpg)

![](images/a20b8da7ff1d66e6b0d3090a5f1de020b50782492f1bd9a95682090f7d4f5c4d.jpg)  
Figure 2: Two sets dataset obtained increasing isotropic Gaussian noise σ, creating well separated sets (left), closer sets (center) and overlapping sets (right)

When the sets are well separated $( \sigma = 0 . 0 5 )$ , T-ARC with persistence-based similarity achieves perfect Accuracy and F1-score (1.00), matching K-means, Spectral clustering and its Euclidean-based variant (Table 2). The Silhouette scores confirm high cluster coherence for all methods (0.97) except Spherical K means (0.02), whose angular-similarity assumption does not match this data structure; Spherical K-means remains near 0.51 Accuracy for all values of σ. We use this configuration as the reference one, illustrated in Fig. 3 and Fig. 4, which show the dataset, initialization, and final assignment, and the objective/error trajectories, respectively. Fig. 4a shows the objective decreasing and stabilizing, in agreement with the theory, while Fig. 4b shows the errors on the variables.

![](images/8354f291d155b559a8e34b95231a30de8e99da22b2a38f4fae3b52ebc4af9fd3.jpg)  
Figure 3: Three-panel illustration of the two separated sets dataset obtained using $\sigma = 0 . 0 5$ . Left: the original data points, colored by ground-truth labels. Center: random initialization with $k = 2$ clusters. Right: final assignment obtained by T-ARC using the persistence-based similarity matrix.

![](images/54be8bab9ce3752cb1a2d766278b0b875921b70b7a740a866401f72418132201.jpg)

![](images/65ff1abcae1c6c12597fe633e9123a365e304dd2fff69a64112736924af2ec4c.jpg)  
(a)

![](images/8c6e535ebe0f9b72f652aaa40559cabc803b622e820a4ea1a0cbef0ec2ea1a31.jpg)  
(b)  
Figure 4: (a) Objective function value during iterations and (b) Frobenius norm error of $C , \mu ,$ and L during iterations for T-ARC with persistence-based similarity matrix on Example 1.

As the sets get closer (σ = 0.12), T-ARC (persistence) still matches K-means and Spectral clustering (0.99), while the Euclidean-based variant is slightly lower (0.96). When the sets overlap $( \sigma = 0 . 2 0 )$ points drawn from the tail of one cluster fall in the region of the other, and the Accuracy of T-ARC (persistence) drops to 0.83, below K-means (0.93), Spectral clustering (0.95) and the Euclidean-based variant (0.92). This behavior reflects the very property that makes the persistence-based prior efective on non-convex structures: since the similarity between two points is determined by the minimax linking path between them in the Vietoris–Rips filtration, it follows the connectivity of the data rather than its average geometry, which allows T-ARC to trace elongated and interleaved clusters (as shown in the following). When the two sets overlap, however, a few points in the overlap region are enough to connect them, and the connectivity signal no longer separates the clusters; in this regime the Euclidean-based variant, which aggregates all pairwise distances, is more robust. The two similarity constructions thus play complementary roles, and this experiment delineates the regime in which the persistence-based prior is most informative, namely when the persistence gap between intra- and inter-cluster merges is preserved.

Example 2: Two-Moons Dataset We generated a two-dimensional point cloud of 160 points forming two non-linearly separable half-moon clusters, centered around (0.3, 0.6) and (0.7, 0.3), with Gaussian noise $\sigma = 0 . 0 2$ . Fig. 5 shows the dataset, initialization, and final assignment.

![](images/3980cfe64516f67c9882e44ed0e436f945af6ae18e097170580ab8abbb852fca.jpg)  
Figure 5: Three-panel illustration of the Two-Moons dataset. Left: the original data points, colored by ground-truth labels. Center: random initialization with k = 4 clusters. Right: final assignment obtained by T-ARC using the persistence-based similarity matrix.

![](images/43f20d3f52cf88f15491529ef222a9c68aac51b87d8997c083401750cd7ea195.jpg)  
(a)

![](images/e7f5509efc2ce45ac60c47a5ec70b0d4cc465ac47a41ad4d80ab933c30c03d98.jpg)  
(b)  
Figure 6: (a) Objective function value during iterations and (b) Frobenius norm error of C, µ, and L during iterations for T-ARC with persistence-based similarity matrix on Example 2.

This dataset is known to be challenging for linear clustering models such as K-means. Incorporating the randomized Laplacian guided by the persistence-based similarity matrix enables T-ARC to follow the topological structure of the data, accurately identifying both connected components. Both T-ARC (persistence) and Spectral clustering achieve perfect Accuracy and F1-score (1.00), while K-means (0.89) and its Euclidean-based T-ARC variant (0.87) fall short. The Silhouette score of T-ARC (persistence) (0.61) is slightly below K-means (0.67), which is expected since the Silhouette coeficient measures cluster compactness in Euclidean space, favoring the convex partitions produced by K-means. Despite this, T-ARC correctly identifies the curved structure of the data, whereas K-means produces geometrically compact but topologically incorrect clusters. The Euclidean-based T-ARC version (0.66) achieves a higher Silhouette than the persistence-based variant, reflecting its more geometric nature, while Spectral clustering (0.61) matches T-ARC (persistence), confirming that the persistence prior guides the algorithm toward a topologically coherent partition rather than a purely geometric one.

## Example 3: Two-Spirals Dataset We generated a two-dimensional point cloud of 120 points (60 per cluster) forming two interleaved Archimedean spirals, the second obtained by rotating the first by π. Each arm is parametrized as (t cos t, t sin t) with $t \in [ \pi / 2 , 2 \pi ]$ , sampled with approximately uniform density along the arc, and isotropic Gaussian noise with $\sigma = 0 . 0 3$ is added to each point. Fig. 7 shows the dataset, initialization, and final assignment.

Interleaved spirals are a classical benchmark for clustering methods, as the two clusters are not linearly separable and each arm winds around the other, so that points belonging to diferent clusters can be closer in Euclidean distance than points at opposite ends of the same arm. Centroid-based methods are therefore structurally unable to recover the correct partition. On this configuration, T-ARC (persistence) achieves the highest Accuracy and F1-score (0.96), clearly outperforming K-means and Spherical K-means (both 0.84), its Euclidean-based variant (0.85), and Spectral clustering (0.78 and 0.76, respectively). The persistence-based similarity encodes the connectivity of each arm at the scale at which the two spirals are still separate connected components, and the randomized Laplacian guided by this similarity allows T-ARC to follow the winding structure of the data. Spectral clustering is more exposed to spurious connections between adjacent arms, particularly near the center of the spirals, where the arms are closest.

![](images/cd6313b02dd172f3189fd9367bcd2702ae7722514c26e5c51ef8a5892dd76785.jpg)  
Figure 7: Three-panel illustration of the Two-Spirals dataset. Left: the original data points, colored by ground-truth labels. Center: random initialization with k = 2 clusters. Right: final assignment obtained by T-ARC using the persistence-based similarity matrix.

![](images/eb51ffd18f7af905187e80dc9f0384c8cf9885d87f60153bd3bac2b5bff85fab.jpg)  
(a)

![](images/03436b6e024e42c226ef94f04598e2441ac56945a1f9ff877c08c2e054f9c881.jpg)  
(b)  
Figure 8: (a) Objective function value during iterations and (b) Frobenius norm error of C, µ, and L during iterations for T-ARC with persistence-based similarity matrix on the Two-Spirals dataset.

As in the Two-Moons case, the Silhouette score shows the opposite ordering: K-means, Spherical K-means and T-ARC (Euclidean) attain the highest values (0.68), while T-ARC (persistence) (0.58) and Spectral clustering (0.45) lag behind. This is expected, since the Silhouette coeficient measures cluster compactness in Euclidean space and therefore rewards the convex partitions produced by centroid-based methods, which here cut across both spirals. The partition found by T-ARC (persistence) follows the arms and is topologically correct, but it is necessarily less compact in the Euclidean sense, which the Silhouette coeficient penalizes.

Example 4: Two-Eights Dataset We generated 160 two-dimensional points arranged as four circles (with 40 points per cluster) of radius $r = 0 . 1 2 .$ grouped into two vertically-aligned pairs forming two “figure-eight” shapes. Circle centers are (0.35, 0.62), (0.35, 0.38), (0.70, 0.62), (0.70, 0.38), with Gaussian noise $\sigma = 0 . 0 1$ . The algorithm is initialized with k = 4 (the number of circles) to probe whether T-

ARC can identify higher-level topological structure. Fig. 9 shows the dataset, initialization, and final assignment. The external indices in Table 2 may initially seem unfavorable for T-ARC (persistence):

![](images/e0a0909deb3b696b17b6f2eb28f5c21cb1e84a4f3579925b2e732a8fa6842c46.jpg)  
Figure 9: Three-panel illustration of the Two-Eights dataset. Left: the original data points, colored by ground-truth labels. Center: random initialization with k = 4 clusters. Right: final assignment obtained by T-ARC using the persistence-based similarity matrix; the algorithm collapses the four initial clusters into two macro-clusters, each corresponding to one of the two figure-eight shapes.

![](images/d7d6fe9043877bdd845bb913fe9a8034f7d6d6f99f61449d620bcfd4c45dadae.jpg)  
(a)

![](images/aa7a5811d6679a8075b56b4839902672ff99bf52f066fa28f4626aa9b0336d3f.jpg)  
(b)  
Figure 10: (a) Objective function value during iterations and (b) Frobenius norm error of C, µ, and L during iterations for T-ARC with persistence-based similarity matrix on Example 4.

Accuracy 0.50 and F1-score 0.33 against K-means (0.88) and Spectral clustering (0.87). However, this comparison is misleading, and the behavior of T-ARC on this dataset is in fact particularly interesting:

• Although initialized with k = 4 clusters, T-ARC (persistence) converges to a partition into two groups, each corresponding to one of the two figure-eight shapes, rather than to the four individual circles. This partition matches the connected components of the data at the scale at which the two eights are separated, and is thus consistent with their intrinsic topological structure.

• This result stems from the topology-aware structure of T-ARC: since the persistence-based prior enters the optimization through the graph-cut term, the algorithm favors partitions that are consistent with the connected components of the data at the relevant filtration scale, and therefore identifies the two figure-eight shapes rather than the four individual circles.

• The Silhouette score supports this interpretation since the two figure-eight shapes are well separated: the topology-driven partition is also geometrically compact, and the two-cluster partition of T-ARC (persistence) attains 0.64, against 0.56 and 0.57 for the four-cluster partitions of K-means and Spectral clustering, although partitions with diferent numbers of clusters are not directly comparable.

This result highlights the ability of the proposed approach to capture the underlying manifold structure rather than forcing a purely geometric partition of the data. Consequently, even though standard classification metrics may appear less favorable, the method successfully uncovers the topological organization of the dataset, demonstrating its robustness in settings where the true structure is governed by nonlin ear relationships. Notably, T-ARC’s behavior reflects a diferent strength: it identifies the higher-level topological structure (the two “eights”) rather than the individual circles.

Example 5: Chain-link Dataset We generated a three-dimensional point cloud of 160 points forming two circular structures (rings) embedded in $\mathbb { R } ^ { 3 }$ , arranged in a chain-link configuration. The first ring lies in the XY -plane and is centered at the origin, while the second lies in the XZ-plane and is shifted so that the two structures touch at a single point. The rings exhibit diferent sampling densities and noise levels: the first is densely sampled with low Gaussian noise $( \sigma = 0 . 0 3 )$ , whereas the second is sparser and afected by higher noise $( \sigma = 0 . 1 2 )$ . This dataset poses a significant challenge for clustering algorithms, as the two structures are nonlinearly separable and intersect at a point, while the imbalance in density and noise further increases the dificulty of the task.

On this dataset, the Euclidean-based variant of T-ARC achieves the best external indices, with an Accuracy and F1-score of 0.91, outperforming Spectral clustering (0.84), K-means and T-ARC (persistence) (both 0.83), and Spherical K-means (0.51). This is a notable departure from the other datasets: the persistence-based prior, which excels when the topological structure dominates the geometry, is here slightly less discriminative than the Euclidean similarity, likely because the two rings intersect at a single point and the persistence prior merges them into a single connected component at the relevant scale. In terms of internal validity, the results are more nuanced. The Silhouette scores are relatively close across methods, with T-ARC (persistence) and K-means tied at the top (0.52), closely followed by Spectral clustering (0.51), while T-ARC (Euclidean) attains 0.41. This behavior is consistent with the geometric nature of the Silhouette coeficient, which favors convex and well-separated clusters even when they do not reflect the underlying topology.

Overall, the experiment highlights a trade-of between geometric compactness and topological correctness. Interestingly, the two T-ARC variants play complementary roles on this dataset: the Euclidean variant recovers the two rings more accurately, while the persistence variant, whose scores coincide with those of K-means, yields a more compact but less faithful partition, since the persistence prior merges the two rings at their contact point. This suggests that the Euclidean construction is preferable in the presence of intersecting structures, consistently with the overlapping Two sets case.

![](images/64d33bccb4ae73b03fde7e4af91875f37c4ac0aa31e6579622e3f6dc4399bf5e.jpg)  
Figure 11: Three-panel illustration of the Chain-link dataset. Left: the original data points, colored by ground-truth labels. Center: random initialization with k = 4 clusters. Right: final assignment obtained by T-ARC using the euclidean-based similarity matrix.

Example 6: Fashion-MNIST Subset As a real-world application, we consider subsets of the Fashion-MNIST [21] dataset. We draw 10 random subsets, each made of 50 samples from each of three Fashion-MNIST classes—Trouser, Sneaker, and Bag—so that every subset contains 150 grayscale 28 × 28 images, each represented as a vector in $\mathbb { R } ^ { 7 8 4 }$ with pixel intensities rescaled to [0, 1]. All methods are run on each subset from the same initialization, and we report mean ± standard deviation over the 10 subsets.

![](images/d77ee1bf3e78506dbe84beb2f9f0eb768c681147c79f2e76221d2fb1d1f88004.jpg)  
(a)

![](images/5e3b59c93d4fe0b0ee0c48de4379845e577cdaa918fb55f0048185f1ff71bcb2.jpg)  
(b)  
Figure 12: (a) Objective function value during iterations and (b) Frobenius norm error of $C , \mu ,$ and L during iterations for $\mathrm { T - A R C }$ with euclidean-based similarity matrix on Example 5.

T-ARC (persistence) attains an Accuracy of $0 . 9 1 \pm 0 . 0 2$ and an F1-score of $0 . 9 1 \pm 0 . 0 2$ , and its Euclidean-based variant $0 . 9 1 { \pm } 0 . 0 3$ and $0 . 9 1 { \pm } 0 . 0 3$ ; the persistence-based prior matches or outperforms the Euclidean one in 9 of the 10 subsets (Table 1). Both variants outperform Spectral clustering $( 0 . 7 9 \pm 0 . 0 3$ Accuracy, $0 . 7 6 \pm 0 . 0 5 \ \mathrm { F 1 - s c o r e } )$ and, on average, K-means $( 0 . 8 9 \pm 0 . 1 3$ Accuracy, $0 . 8 8 \pm 0 . 1 4 ~ \mathrm { F 1 - s c o r e } )$ The comparison with K-means is best read in terms of stability: started from the same initialization, K-means occasionally converges to poor local minima (down to 0.53 Accuracy on one subset), whereas T-ARC never does, with a standard deviation about five to six times smaller. Spherical K-means attains the best external indices $( 0 . 9 5 \pm 0 . 0 4$ Accuracy, $0 . 9 5 \pm 0 . 0 4 \ \mathrm { F 1 - s c o r e } )$ , consistently with the angular structure of normalized pixel intensities. On the internal side, T-ARC (persistence) achieves the highest Silhouette score among all methods $( 0 . 4 9 \pm 0 . 0 2$ , versus $0 . 4 7 \pm 0 . 0 3$ for Spherical K-means and $0 . 4 6 \pm 0 . 0 8$ for K-means), indicating that its partitions are the most compact ones in the Euclidean sense.

Overall, this experiment shows that, on real high-dimensional data, T-ARC is competitive with classical methods, markedly more stable than K-means under the same initialization, and yields the most compact partitions, while Spherical K-means remains the strongest method in terms of label agreement on this dataset.

## 7 Conclusion and Future Work

In this work, we introduced T-ARC, a novel clustering framework that integrates topological information directly into the optimization objective to overcome the geometric limitations of classical methods such as K-means. By complementing the standard data-fidelity term with a graph-cut regularization, T-ARC promotes cluster assignments that respect the connectivity structure of the data. The key methodological contribution lies in the joint learning of the graph structure and the cluster assignments: the latent graph is modeled as a random realization from an SBM, whose parameters are optimized within a DRO framework. This formulation yields closed-form updates, solved via a proximal point method, and ensures robustness to uncertainty in the graph distribution.

To further anchor the framework in topological principles, SBM and DRO are informed by a persistencebased similarity matrix constructed from the zero-dimensional persistent homology. This matrix translates the multiscale connectivity structure of the dataset into a pairwise similarity representation, which serves as a fixed topological prior throughout the optimization. The overall optimization proceeds via BCD, and we provided a convergence analysis showing that the expected value of a Lyapunov functional is nonincreasing along the iterations, including the stochastic graph updates, and therefore convergent, with vanishing successive diferences of all blocks.

Experimental validation on synthetic datasets with complex geometries demonstrated that T-ARC successfully recovers the underlying topological structure where K-means fails, as in the Two-Moons and Two-Spirals configurations. Notably, on the two-eights dataset, T-ARC (with $k = 4 )$ identified the two higher-level clusters corresponding to the two eights, a partition consistent with the connected

Table 1: Performance metrics of T-ARC and baseline methods on each of the 10 random Fashion-MNIST subsets (50 images per class). Best value per metric and subset in bold; the last column reports mean ± standard deviation over the subsets.
<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="10">Subset</td><td rowspan="2"> $\mathbf { M e a n } \pm \mathbf { s t d }$ </td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td rowspan="5">Accuracy</td><td>T-ARC (persistence)</td><td>0.91</td><td>0.91</td><td>0.89</td><td>0.92</td><td>0.95</td><td>0.90</td><td>0.94</td><td>0.92</td><td>0.92</td><td>0.88</td><td> $0 . 9 1 \pm 0 . 0 2$ </td></tr><tr><td>T-ARC (Euclidean)</td><td>0.91</td><td>0.89</td><td>0.89</td><td>0.93</td><td>0.95</td><td>0.89</td><td>0.93</td><td>0.92</td><td>0.90</td><td>0.88</td><td> $0 . 9 1 \pm 0 . 0 3$ </td></tr><tr><td>K-means</td><td>0.93</td><td>0.53</td><td>0.92</td><td>0.95</td><td>0.97</td><td>0.87</td><td>0.93</td><td>0.95</td><td>0.95</td><td>0.87</td><td> $0 . 8 9 \pm 0 . 1 3$ </td></tr><tr><td>Spherical K-means</td><td>0.96</td><td>0.99</td><td>0.87</td><td>0.99</td><td>0.97</td><td>0.94</td><td>0.95</td><td>0.91</td><td>0.95</td><td>0.93</td><td> $\mathbf { 0 . 9 5 \pm 0 . 0 4 }$ </td></tr><tr><td>Spectral clustering</td><td>0.80</td><td>0.81</td><td>0.77</td><td>0.76</td><td>0.83</td><td>0.73</td><td>0.81</td><td>0.77</td><td>0.83</td><td>0.77</td><td> $0 . 7 9 \pm 0 . 0 3$ </td></tr><tr><td rowspan="5">F1-score</td><td>T-ARC (persistence)</td><td>0.91</td><td>0.91</td><td>0.88</td><td>0.92</td><td>0.95</td><td>0.90</td><td>0.94</td><td>0.92</td><td>0.92</td><td>0.88</td><td> $0 . 9 1 \pm 0 . 0 2$ </td></tr><tr><td>T-ARC (Euclidean)</td><td>0.91</td><td>0.88</td><td>0.88</td><td>0.93</td><td>0.95</td><td>0.89</td><td>0.93</td><td>0.92</td><td>0.90</td><td>0.88</td><td> $0 . 9 1 \pm 0 . 0 3$ </td></tr><tr><td>K-means</td><td>0.93</td><td>0.49</td><td>0.92</td><td>0.95</td><td>0.97</td><td>0.86</td><td>0.93</td><td>0.95</td><td>0.95</td><td>0.87</td><td> $0 . 8 8 \pm 0 . 1 4$ </td></tr><tr><td>Spherical K-means</td><td>0.96</td><td>0.99</td><td>0.86</td><td>0.99</td><td>0.97</td><td>0.94</td><td>0.95</td><td>0.91</td><td>0.95</td><td>0.93</td><td> $\mathbf { 0 . 9 5 \pm 0 . 0 4 }$ </td></tr><tr><td>Spectral clustering</td><td>0.78</td><td>0.79</td><td>0.75</td><td>0.73</td><td>0.83</td><td>0.68</td><td>0.80</td><td>0.73</td><td>0.82</td><td>0.75</td><td> $0 . 7 6 \pm 0 . 0 5$ </td></tr><tr><td rowspan="5"></td><td>T-ARC (persistence)</td><td>0.45</td><td>0.50</td><td>0.50</td><td>0.50</td><td>0.51</td><td>0.46</td><td>0.53</td><td>0.48</td><td>0.49</td><td>0.47</td><td> ${ \bf 0 . 4 9 \pm 0 . 0 2 }$ </td></tr><tr><td>T-ARC (Euclidean)</td><td>0.49</td><td>0.50</td><td>0.50</td><td>0.49</td><td>0.50</td><td>0.46</td><td>0.52</td><td>0.49</td><td>0.41</td><td>0.40</td><td> $0 . 4 8 \pm 0 . 0 4$ </td></tr><tr><td>Silhouette K-means</td><td>0.48</td><td>0.23</td><td>0.49</td><td>0.50</td><td>0.51</td><td>0.45</td><td>0.54</td><td>0.49</td><td>0.48</td><td>0.47</td><td> $0 . 4 6 \pm 0 . 0 8$ </td></tr><tr><td>Spherical K-means</td><td>0.46</td><td>0.45</td><td>0.49</td><td>0.48</td><td>0.51</td><td>0.41</td><td>0.52</td><td>0.50</td><td>0.46</td><td>0.44</td><td> $0 . 4 7 \pm 0 . 0 3$ </td></tr><tr><td>Spectral clustering</td><td>0.33</td><td>0.36</td><td>0.33</td><td>0.32</td><td>0.36</td><td>0.17</td><td>0.31</td><td>0.31</td><td>0.33</td><td>0.32</td><td> $0 . 3 1 \pm 0 . 0 5$ </td></tr></table>

Table 2: Performance metrics of T-ARC and baseline methods across all experiments. Best values per metric and dataset are in bold. † T-ARC (persistence) converges to k = 2 macro-clusters (one per eight); external indices computed w.r.t. the $k = 4$ ground truth. Fashion-MNIST values are mean ± standard deviation over the 10 subsets of Table 1.

Several directions for future research emerge from this work:

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="3">Two sets</td><td rowspan="2"></td><td rowspan="2">Two moons Two spirals Two eights†</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">Chain-link Fashion-MNIST</td></tr><tr><td></td><td>Well-separated Closer Overlapping</td><td></td></tr><tr><td rowspan="5">Accuracy</td><td>T-ARC (persistence)</td><td>1.00</td><td>0.99</td><td>0.83</td><td>1.00</td><td>0.96</td><td>0.50</td><td>0.83</td><td> $0 . 9 1 \pm 0 . 0 2$ </td></tr><tr><td>T-ARC (Euclidean)</td><td>1.00</td><td>0.96</td><td>0.92</td><td>0.87</td><td>0.85</td><td>0.60</td><td>0.91</td><td> $0 . 9 1 \pm 0 . 0 3$ </td></tr><tr><td>K-means</td><td>1.00</td><td>0.99</td><td>0.93</td><td>0.89</td><td>0.84</td><td>0.88</td><td>0.83</td><td> $0 . 8 9 \pm 0 . 1 3$ </td></tr><tr><td>Spherical K-means</td><td>0.51</td><td>0.51</td><td>0.51</td><td>0.87</td><td>0.84</td><td>0.53</td><td>0.51</td><td> ${ \bf 0 . 9 5 \pm 0 . 0 4 }$ </td></tr><tr><td>Spectral clustering</td><td>1.00</td><td>0.99</td><td>0.95</td><td>1.00</td><td>0.78</td><td>0.87</td><td>0.84</td><td> $0 . 7 9 \pm 0 . 0 3$ </td></tr><tr><td rowspan="5">F1-score</td><td>T-ARC (persistence)</td><td>1.00</td><td>0.99</td><td>0.83</td><td>1.00</td><td>0.96</td><td>0.33</td><td>0.83</td><td> $0 . 9 1 \pm 0 . 0 2$ </td></tr><tr><td>T-ARC (Euclidean)</td><td>1.00</td><td>0.96</td><td>0.92</td><td>0.87</td><td>0.85</td><td>0.53</td><td>0.91</td><td> $0 . 9 1 \pm 0 . 0 3$ </td></tr><tr><td>K-means</td><td>1.00</td><td>0.99</td><td>0.93</td><td>0.89</td><td>0.84</td><td>0.88</td><td>0.83</td><td> $0 . 8 8 \pm 0 . 1 4$ </td></tr><tr><td>Spherical K-means</td><td>0.51</td><td>0.51</td><td>0.51</td><td>0.87</td><td>0.84</td><td>0.55</td><td>0.51</td><td> ${ \bf 0 . 9 5 \pm 0 . 0 4 }$ </td></tr><tr><td>Spectral clustering</td><td>1.00</td><td>0.99</td><td>0.95</td><td>1.00</td><td>0.76</td><td>0.87</td><td>0.84</td><td> $0 . 7 6 \pm 0 . 0 5$ </td></tr><tr><td rowspan="5">Silhouette K-means</td><td>T-ARC (persistence)</td><td>0.97</td><td>0.85</td><td>0.51</td><td>0.61</td><td>0.58</td><td>0.64</td><td>0.52</td><td> ${ \bf 0 . 4 9 \pm 0 . 0 2 }$ </td></tr><tr><td>T-ARC (Euclidean)</td><td>0.97</td><td>0.80</td><td>0.67</td><td>0.66</td><td>0.68</td><td>0.18</td><td>0.41</td><td> $0 . 4 8 \pm 0 . 0 4$ </td></tr><tr><td></td><td>0.97</td><td>0.85</td><td>0.69</td><td>0.67</td><td>0.68</td><td>0.56</td><td>0.52</td><td> $0 . 4 6 \pm 0 . 0 8$ </td></tr><tr><td>Spherical K-means</td><td>0.02</td><td>0.13</td><td>0.24</td><td>0.66</td><td>0.68</td><td>0.18</td><td>0.40</td><td> $0 . 4 7 \pm 0 . 0 3$ </td></tr><tr><td>Spectral clustering</td><td>0.97</td><td>0.85</td><td>0.69</td><td>0.61</td><td>0.45</td><td>0.57</td><td>0.51</td><td> $0 . 3 1 \pm 0 . 0 5$ </td></tr></table>

components of the data at the relevant scale. On Fashion-MNIST, T-ARC was competitive with the baselines in terms of external validity, outperforming K-means and Spectral clustering on average while Spherical K-means remained the strongest method in terms of label agreement; at the same time, T-ARC proved markedly more stable than K-means under the same initialization and attained the highest average Silhouette score. The persistence-based similarity matrix matched or outperformed its Euclidean counterpart whenever the topological structure dominates the geometry (well-separated and closer Two sets, Two-Moons, Two-Spirals, Fashion-MNIST), while the Euclidean variant proved more robust when clusters overlap or intersect (overlapping Two sets, Chain-link), delineating the regime in which the topological prior is most informative.

• Although we focused on the zero-dimensional persistent homology, the framework could naturally accommodate similarity matrices derived from higher-order persistent homology, potentially cap turing loop- and void-like structures in the data.

• The computation of the persistence-based similarity matrix via Vietoris-Rips filtration becomes demanding for large point clouds; exploring approximate or landmark-based methods for persistent homology could extend the applicability of T-ARC to larger-scale datasets.

• Finally, the SBM represents only one among many possible probabilistic formulations for the latent graph; richer families of random graphs that account for degree heterogeneity or incorporate geometric structure could yield more flexible connectivity representations.

In summary, T-ARC represents a principled step toward embedding topological structure directly into clustering optimization, bridging topological data analysis and unsupervised learning.

Acknowledgment S.G.D., N.D.B., F.E., and L.S. are members of the Gruppo Nazionale Calcolo Scientifico - Istituto Nazionale di Alta Matematica (GNCS-INdAM). S.G.D. is funded by a PhD fellowship within the framework of the Italian “D.M. n. 117, March 2, 2023” - under the National Recovery and Resilience Plan, Msn. 4, Comp. 2, Investment 3.3 - PhD Project “Topological Data Analysis and optimization for industrial processes”, co-supported by “Pirelli Tyre S.p.A.” (CUP H91I23000170007). This work was completed during a visiting research period of S.G.D. at the University of Southampton. F.E. and L.S. are partially supported by "INdAM - GNCS Project" CUP E53C25002010001.

Author Contributions S.G.D.: Conceptualization, Methodology, Software, Validation, Formal analysis, Data Curation, Investigation, Writing - Original Draft, Writing - Review & Editing, Visualization. A.A.: Conceptualization, Methodology, Formal analysis, Writing - Original Draft, Writing - Review & Editing, Supervision. F.E. and L.S.: Con ceptualization, Methodology, Writing - Review & Editing. N.D.B.: Conceptualization, Methodology, Writing - Review & Editing, Supervision.

## References

[1] D. Cai, X. He, J. Han, and T. S. Huang, Graph regularized nonnegative matrix factorization for data representation, IEEE Transactions on Pattern Analysis and Machine Intelligence, 33 (2011), pp. 1548–1560.

[2] G. Carlsson, Topology and data, Bulletin of the American Mathematical Society, 46 (2009), pp. 255–308.

[3] D. Cohen-Steiner, H. Edelsbrunner, and J. Harer, Stability of persistence diagrams, Discrete & Computational Geometry, 37 (2007), pp. 103–120.

[4] M. N. Dao and M. K. Tam, A lyapunov-type approach to convergence of the douglas-rachford algorithm for a nonconvex setting, Journal of Global Optimization, 73 (2018), pp. 83–112.

[5] C. Ding, T. Li, and M. I. Jordan, Convex and semi-nonnegative matrix factorizations, IEEE Transactions on Pattern Analysis and Machine Intelligence, 32 (2010), pp. 45–55.

[6] H. Edelsbrunner and J. L. Harer, Computational Topology: An Introduction, American Mathematical Society, Providence, RI, 2010.

[7] F. Hensel, M. Moor, and B. Rieck, A survey of topological machine learning methods, Frontiers in Artificial Intelligence, 4 (2021), p. 681108.

[8] J. Hong, W. Qian, Y. Chen, and Y. Zhang, A geometric approach to kk-means clustering, IEEE Transactions on Knowledge and Data Engineering, 38 (2026), pp. 1442–1453.

[9] X. Huang, A. Ang, K. Huang, J. Zhang, and Y. Wang, Inhomogeneous graph trend filtering via a l 2, 0}-norm cardinality penalty, IEEE Transactions on Signal and Information Processing over Networks, (2025).

[10] A. K. Jain, Data clustering: 50 years beyond K-means, Pattern Recognition Letters, 31 (2010), pp. 651–666.

[11] D. Kuhn, S. Shafiee, and W. Wiesemann, Distributionally robust optimization, Acta Numerica, 34 (2025), pp. 579– 804.

[12] C. Lee and D. J. Wilkinson, A review of stochastic block models and extensions for graph clustering, Applied Network Science, 4 (2019).

[13] S. P. Lloyd, Least squares quantization in PCM, IEEE Transactions on Information Theory, 28 (1982), pp. 129–137.

[14] N. Parikh and S. Boyd, Proximal algorithms, Foundations and Trends in Optimization, 1 (2014), pp. 127–239.

[15] P. J. Rousseeuw, Silhouettes: A graphical aid to the interpretation and validation of cluster analysis, Journal of Computational and Applied Mathematics, 20 (1987), pp. 53–65.

[16] H. Schenck, Algebraic Foundations for Applied Topology and Data Analysis, Springer International Publishing, 2022.

[17] L. Selicato, A. Pagano, F. Esposito, and M. Icardi, Topological data analysis for resilience assessment of water distribution networks, Mathematics and Computers in Simulation, 231 (2025), pp. 62–70.

[18] R. Sinkhorn and P. Knopp, Concerning nonnegative matrices and doubly stochastic matrices, Pacific Journal of Mathematics, 21 (1967), pp. 343–348.

[19] The GUDHI Project, GUDHI User and Reference Manual, GUDHI Editorial Board, 2015.

[20] U. von Luxburg, A tutorial on spectral clustering, Statistics and Computing, 17 (2007), pp. 395–416.

[21] H. Xiao, K. Rasul, and R. Vollgraf, Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017.

[22] Y. Xu and W. Yin, A block coordinate descent method for regularized multiconvex optimization with applications to nonnegative tensor factorization and completion, SIAM Journal on Imaging Sciences, 6 (2013), pp. 1758–1789.

[23] A. Zomorodian and G. Carlsson, Computing persistent homology, Discrete & Computational Geometry, 33 (2005), pp. 249–274.

## A Derivation of Feasibility Constraints on $b$

Under the smoothed parametric representation $B ( a , b ) = a S + b ( \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } - S )$ , the ambiguity set $\boldsymbol { B }$ (equation (2)) imposes three constraints on a and b. Throughout, we substitute $a = 1 + b ( 1 - n )$ (derived from row-stochasticity) to obtain constraints solely in b.

Row-stochasticity. Imposing $B ( a , b ) \mathbf { 1 } _ { n } = \mathbf { 1 } _ { n }$ and using $S \mathbf { 1 } _ { n } = \mathbf { 1 } _ { n }$

$$
( a S + b ( \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } - S ) ) \mathbf { 1 } _ { n } = a \mathbf { 1 } _ { n } + b ( n - 1 ) \mathbf { 1 } _ { n } = \mathbf { 1 } _ { n } ,
$$

yielding $a + b ( n - 1 ) = 1$ , i.e., $a = 1 + b ( 1 - n )$ . Positivity $a > 0$ then gives

$$
\begin{array} { r } { b < \frac { 1 } { n - 1 } . } \end{array}\tag{11}
$$

Nonnegativity. The condition $B ( a , b ) _ { i j } \ \geq \ 0$ requires $a S _ { i j } + b ( 1 - S _ { i j } ) \geq 0$ for all $i , j$ . The most restrictive bound is attained at $M : = \operatorname* { m a x } _ { i , j } S _ { i j }$ , yielding $a \ge - b ( 1 - M ) / M$ . Substituting $a = 1 + b ( 1 - n )$ gives

$$
\begin{array} { r } { b \leq \frac { 1 } { n - 1 / M } . } \end{array}\tag{12}
$$

Since each row of S is normalized, $1 / n \leq M \leq 1$ , hence $0 \leq n - 1 / M \leq n - 1$ , so $1 / ( n - 1 / M ) \geq 1 / ( n - 1 )$ , and constraint (12) is always dominated by (11).

Distance constraint. The Frobenius constraint $\| B ( a , b ) - S \| _ { F } \leq r$ simplifies (using $a = 1 + b ( 1 - n ) )$ to $\| b ( \mathbf { 1 } _ { n } \mathbf { 1 } _ { n } ^ { \top } - n S ) \| _ { F } \leq r$ . Letting $\begin{array} { r } { S _ { 2 } : = \sum _ { i , j } S _ { i j } ^ { 2 } } \end{array}$ (which satisfies $S _ { 2 } \geq 1$ for row-stochastic $\pmb { S }$ with $S _ { 2 } > 1$ generically), this gives $b \leq r / ( n \sqrt { S _ { 2 } - 1 } )$

Final bounds. Combining and discarding the dominated nonnegativity constraint:

$$
\begin{array} { r } { 0 \leq b \leq b _ { \mathrm { m a x } } : = \operatorname* { m i n } \Bigl \{ \frac { 1 } { n - 1 } , \quad \frac { r } { n \sqrt { S _ { 2 } - 1 } } \Bigr \} . } \end{array}\tag{13}
$$