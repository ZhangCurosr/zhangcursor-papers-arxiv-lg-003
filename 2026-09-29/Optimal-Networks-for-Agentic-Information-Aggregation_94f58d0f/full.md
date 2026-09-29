# Optimal Networks for Agentic Information Aggregation

MohammadHossein Bateni Google Research bateni@google.com

MohammadTaghi Hajiaghayi University of Maryland hajiagha@umd.edu

Zahra Hadizadeh University of California, Irvine zhadizad@uci.edu

Mahdi JafariRaviz University of Maryland mahdij@umd.edu

Shayan Taherijam University of California, Irvine staherij@uci.edu

## Abstract

We study information aggregation in the networked learning model introduced by Kearns, Roth, and Ryu (SODA 2026). There is a fixed distribution over d features and a common label. Agents learn in topological order on a directed acyclic graph. Each observes a subset of the features and its parents’ predictions, fits a linear predictor to minimize mean squared error, and passes only its prediction forward. The global predictor is the best linear predictor using all features. Kearns, Roth, and Ryu show that the output agent’s error approaches the global predictor’s error along suficiently deep paths with suitable feature coverage, while insuficient depth can prevent aggregation even in large networks.

In contrast to their main focus on a given graph and feature allocation, we consider the limits of the model under two settings. In the adaptive designer setting, a designer chooses the graph, feature allocation, and output agent knowing the distribution. In the oblivious designer setting, the designer fixes all three before an adversary chooses the distribution. Each agent observes one feature and receives predictions from a limited number of parents.

We call the aggregation exact when the output agent matches the global predictor exactly. For $d \geq 3$ , we show that no finite depth guarantees exact aggregation for every distribution with one parent per agent, even when the designer knows the distribution.

In contrast, two parents per agent sufice for exact aggregation even in the oblivious designer setting. A fixed graph, feature allocation, and output agent achieve this for every distribution at depth O(d log d). Knowing the distribution reduces the depth to $O ( d )$ . Both constructions use $O ( d ^ { 2 } )$ agents, with a very large constant for two parents. We show the bounds on the depth and number of agents are all optimal up to constant factors.

## 1 Introduction

Social learning in networks studies how parties with diferent information learn from one another [DeG74]. Parties typically learn from the opinions or predictions of others rather than from their private observations. A similar structure appears in multi-agent AI systems, where a task is split across several models and later models build on the outputs of earlier ones [GCW+24]. In such systems the network is a design choice: the designer decides which model sees which information and which models talk to each other. This raises a basic question. If each party passes on only its prediction, can a well-designed network still do as well as a single learner that sees everything? And if so, how many predictions must each learner receive, and how deep and how large must the network be?

Kearns, Roth, and Ryu [KRR26] model this kind of distributed learning as a network of agents, where each agent has access to a subset of the features and learns a model to predict a common label. Agents learn in turn and then forward their predictions to their successors in the network. Each prediction is therefore both an estimate of the label and the summary of an agent’s information that later agents receive. They ask how the output agent’s prediction compares to a global predictor that has access to all the features.

More formally, let $G = ( A , E )$ be a directed acyclic graph (DAG), and let D be a distribution over $( x , Y )$ , with $\boldsymbol { x } = ( x _ { 1 } , \dots , x _ { d } )$ being the vector of features and Y being the label. Each agent has direct access to a subset of the features of x and receives the predictions of its parents in G. Following a topological order, agents fit linear predictors of Y from these inputs to minimize mean squared error (MSE). The excess error of a prediction is the diference between its MSE and the MSE of the best linear global predictor using all d features. We say that a network achieves information aggregation when the output agent’s prediction is competitive with the global predictor, meaning that its excess error is small. We call the aggregation exact when this excess error is zero.

One might expect that observing every feature somewhere in the network would be enough for exact aggregation. However, a feature that does not help predict the label on its own may become useful when combined with another feature. An agent may leave such a feature out of its prediction, so later agents receive no information about it.

Kearns, Roth, and Ryu [KRR26] ask under what conditions on the graph and feature allocation the network can achieve information aggregation. Their main results focus on analyzing an instance of the problem where the distribution, the graph, the feature allocation, and the output agent are already chosen. They show that under a certain condition on the graph and feature allocation, the excess error of the output agent converges to zero as the depth D grows. Here, depth is the number of agents on a longest path ending at the output agent. They also give examples of distributions, graphs, and feature allocations for which the excess error is bounded below by an inverse polynomia in D.

They also show that depth can be necessary even when the graph and feature allocation are chosen in the best possible way for the distribution. Specifically, Kearns, Roth, and Ryu [KRR26, Theorem 5.9] give, for every d, a distribution on d features such that every DAG of depth at most $D <$ d and every allocation of one feature per agent have excess error at least $1 / ( D + 1 )$ at the output agent. We emphasize the restriction $D < d$ in this result, which leaves open what is possible when the depth reaches or exceeds the number of features.

To understand the limits of the model from a practical perspective, in this work we consider two natural settings for choosing the instance. In the adaptive designer setting, an adversary first chooses a distribution D over $( x , Y )$ with d features. A designer then chooses a DAG G of depth at most D, an allocation of one feature per agent, and an output agent with knowledge of D. In the oblivious designer setting, the designer first chooses the DAG G of depth at most D, an allocation of one feature per agent, and the output agent without knowledge of D. The adversary then chooses D with knowledge of the designer’s choices, so the same graph, allocation, and output agent must work for every distribution.

We also require each agent to receive predictions from at most b parents in both settings. Without this restriction, a single agent could be asked to fit a model from many parent predictions, making its learning problem as large as the global one. With this bound, each agent learns from at most $b + 1$ inputs, including its raw feature.

In both settings, we ask how quickly excess error can decrease with depth, and how much depth and how many agents are necessary and suficient for exact aggregation. We denote the optimal worst-case excess errors at depth D by $R _ { b } ( d , D )$ in the adaptive designer setting and $\overline { { R } } _ { b } ( d , D )$ in the oblivious designer setting. We define these quantities formally in Section 2.

## 1.1 Our results

We summarize our main results in Table 1. For the excess-error bounds, we bound each feature’s second moment and the sum of the $\ell _ { 1 }$ norm of the global predictor’s coeficients by one (Definition 2.3). The constructions for exact aggregation require only finite second moments.

Table 1: Bounds for $d \geq 3$ features and fixed parent limit $b .$
<table><tr><td>Parent limit (b) Quantity</td><td></td><td>Adaptive designer</td><td>Oblivious designer</td></tr><tr><td>1</td><td>Excess error at depth D</td><td> $R _ { b } ( d , D ) = \Theta ( 1 / D )$ </td><td> $\overline { { R } } _ { b } ( d , D ) = \Omega ( 1 / D )$ </td></tr><tr><td>2</td><td>Depth for exact aggregation</td><td>Θ(d)</td><td> $\Theta ( d \log d )$ </td></tr><tr><td> $\geq 3$ </td><td>Depth for exact aggregation</td><td>d</td><td> $\Theta ( d \log d )$ </td></tr><tr><td>≥2</td><td>Number of agents for exact aggregation</td><td> $\Theta ( d ^ { 2 } )$ </td><td>Θ(d2)</td></tr></table>

We first consider $b = 1$ , where the ancestors of the output form a path. For every $d \geq 3$ and $D \geq 1$ , we show that $R _ { 1 } ( d , D ) = \Theta ( 1 / D )$ (Theorem 4.1). This also gives $\overline { { R } } _ { 1 } ( d , D ) = \Omega ( 1 / D )$ in the oblivious designer setting. The matching upper bound in the adaptive designer setting uses knowledge of the distribution to choose the feature allocation along a path of D agents in a greedy way. An oblivious designer cannot make these greedy choices. A natural fixed choice is a path whose agents observe $x _ { 1 } , \ldots , x _ { d }$ in cyclic order, so that every d consecutive agents see all features. Bateni et al. [BHH+26b, Theorem 1] show that such a path has error $O ( d ^ { 2 } / D )$ . Thus $\overline { { R } } _ { 1 } ( d , D )$ lies between $\Omega ( 1 / D )$ and $O ( d ^ { 2 } / D )$ , and we leave closing this gap in d open.

In contrast to $b = 1$ , with three parents per agent $\left( b = 3 \right)$ , the designer can fix a graph, allocation, and output agent that achieve exact aggregation for every distribution at depth $O ( d \log d )$ using $O ( d ^ { 2 } )$ agents (Theorem 5.1). When the designer knows the distribution, we reduce the worst-case depth to d while still using $O ( d ^ { 2 } )$ agents (Theorem 5.2).

We show that each three-parent agent can be replaced by a fixed gadget with a constant number of two-parent agents that reproduces its prediction for every distribution (Lemma 6.1). This constant is independent of d and of the distribution, but it is very large, since the gadget runs all small two-parent networks in parallel. Applying this replacement gives exact aggregation with $b = 2 .$ , depth $O ( d )$ in the adaptive designer setting and $O ( d \log d )$ in the oblivious designer setting, using $O ( d ^ { 2 } )$ agents in both (Theorem 6.2).

We then show that these depth bounds are optimal up to constant factors (Theorem 5.3). In the adaptive designer setting, some distributions require depth at least d for exact aggregation, even with no parent limit. Thus the depth-d bound with three parents is exactly optimal in the worst case. In the oblivious designer setting, every fixed parent limit $b \geq 2$ requires depth $\Omega ( d \log d )$ matching the construction for exact aggregation.

Finally, we show that the quadratic number of agents in our constructions is necessary. For every fixed $b \geq 2$ , some normalized distribution requires $\Omega ( d ^ { 2 } )$ agents for exact aggregation, regardless of depth and even when the designer knows the distribution (Theorem 7.1). This lower bound therefore also holds in the oblivious designer setting, so the number of agents is optimal in both settings.

## 2 Preliminaries

Let $( x , Y ) \sim \mathcal { D }$ , where $x = ( x _ { 1 } , \ldots , x _ { d } ) \in \mathbb { R } ^ { d }$ is the feature vector and $Y \in \mathbb { R }$ is the label. We assume that the features and the label have finite second moments. We write $[ d ] = \{ 1 , \ldots , d \}$ , and all expectations are over $\mathcal { D }$

We work in $L ^ { 2 } ( \mathcal { D } )$ , the space of real random variables with finite second moment, and regard two variables as equal if they agree with probability one. For $U , V \in L ^ { 2 } ( { \mathcal { D } } )$ , we write $\langle U , V \rangle = \mathbb { E } [ U V ]$ and $\| U \| ^ { 2 } = \mathbb { E } [ U ^ { 2 } ]$ . We use f for both a predictor and the random variable $f ( x )$ . Its mean squared error is then MSE $( f ) = \mathbb { E } [ ( Y - f ) ^ { 2 } ] = \| Y - f \| ^ { 2 }$ . For subspaces $V , W \subseteq L ^ { 2 } ( { \mathcal { D } } )$ , their sum $V + W = \{ v + w \mid v \in V , w \in W \}$ is the smallest subspace containing both.

We call vectors $u , v \in L ^ { 2 } ( \mathcal { D } )$ orthogonal, written $u \perp v$ , when $\langle u , v \rangle = 0$ . We write $u \perp V$ when u is orthogonal to every vector in $V _ { : }$ , and $V \perp W$ when every vector in V is orthogonal to every vector in W. The orthogonal complement $V ^ { \perp }$ is the subspace of all vectors in $L ^ { 2 } ( \mathcal { D } )$ orthogonal to V.

There are agents $A = \{ A _ { 1 } , \ldots , A _ { n } \}$ in a DAG $G = ( A , E )$ with a designated output agent $A _ { G }$ An edge $( A _ { j } , A _ { i } ) \in E$ means that $A _ { i }$ receives the prediction of $A _ { j }$ . We write $\operatorname { P a } ( A _ { i } ) = \{ A _ { j } \in A \mid$ $( A _ { j } , A _ { i } ) \in E \}$ for the set of its parents. Agent $A _ { i }$ sees the features $x _ { S _ { i } }$ for a set $S _ { i } \subseteq [ d ]$ , together with its parents’ predictions, and fits the best linear predictor from these inputs. Agents fit in a topological order, so every parent has been fitted before its children.

Let $f _ { i }$ be the predictor of agent $A _ { i }$ . The linear combinations of its inputs form the space

$$
V _ { i } = \operatorname { s p a n } \left( \left\{ x _ { \ell } \mid \ell \in S _ { i } \right\} \cup \left\{ f _ { j } \mid A _ { j } \in \operatorname { P a } ( A _ { i } ) \right\} \right) .
$$

The agent therefore chooses

$$
f _ { i } = \underset { f \in V _ { i } } { \arg \operatorname* { m i n } } \mathrm { M S E } ( f ) .\tag{1}
$$

We call this prediction the fit of Y from the agent’s inputs. We write $f _ { G }$ for the output agent $A _ { G } { } ^ { \prime } \mathrm { s }$ prediction.

For a finite-dimensional subspace $V \subseteq L ^ { 2 } ( { \mathcal { D } } )$ and a variable $Z \in L ^ { 2 } ( \mathcal { D } )$ , we write $P _ { V } Z$ for the vector in V that minimizes $\| Z - v \| ^ { 2 }$ over $v \in V$ . This is the orthogonal projection of $Z$ onto V. It is characterized by the condition that the residual $Z - P _ { V } Z$ is orthogonal to $V .$ . The projected vector is unique, even when its coeficients in a given set of inputs are not. Projection is also linear in Z. In this notation, the fit in equation 1 is $f _ { i } = P _ { V _ { i } } Y$ . For finite-dimensional orthogonal subspaces $V , W \subseteq L ^ { 2 } ( { \mathcal { D } } )$ , projection onto their sum splits as $P _ { V + W } Z = P _ { V } Z + P _ { W } Z$ for every $Z \in L ^ { 2 } ( \mathcal { D } )$

The global predictor fits over $H = \operatorname { s p a n } \{ x _ { 1 } , \ldots , x _ { d } \}$ . We write $f ^ { * } = P _ { H } Y$ for its prediction and $r =$ dim H for the feature rank. Every agent’s prediction lies in H, since its raw features lie in H and, by induction, so do its parents’ predictions. For any predictor $f \in H$ , its excess error is $\mathrm { M S E } ( f ) - \mathrm { M S E } ( f ^ { * } )$ . We say that the network achieves exact aggregation when $f _ { G } = f ^ { * }$

## 2.1 Projection

The following identity relates the improvement in mean squared error to the change in the prediction. We will use it both to compare an agent with the global predictor and to track error along the graph. The proof is in Appendix B.

Lemma 2.1. Let $V \subseteq L ^ { 2 } ( { \mathcal { D } } )$ be a finite-dimensional subspace and let $f = P _ { V } Y$ . Then $\langle Y , f \rangle =$ $\| f \| ^ { 2 }$ , and for every $g \in V$

$$
\operatorname { M S E } ( g ) - \operatorname { M S E } ( f ) = \| g - f \| ^ { 2 } .\tag{2}
$$

Applying Lemma 2.1 with $V = H$ and $f = f ^ { * }$ gives $\mathrm { M S E } ( f _ { i } ) - \mathrm { M S E } ( f ^ { * } ) = \| f _ { i } - f ^ { * } \| ^ { 2 }$ . Thus an agent’s excess error is its squared distance from the global prediction. For an edge $( A _ { j } , A _ { i } ) \in E$ the parent prediction $f _ { j }$ belongs to $V _ { i }$ . Applying the lemma with $V \ = \ V _ { i }$ and $f \ = \ f _ { i }$ gives $\mathrm { M S E } ( f _ { j } ) - \mathrm { M S E } ( f _ { i } ) = \| \bar { f } _ { j } - f _ { i } \| ^ { 2 }$ . Error is therefore non-increasing along every edge. By equation $2 ,$ exact aggregation $f _ { G } = f ^ { * }$ is equivalent to having zero excess error.

The part of the label orthogonal to the raw features cannot afect any agent’s fit. The next lemma lets us remove that part when analyzing a network. The proof is in Appendix B.

Lemma 2.2. Replacing Y by $f ^ { * }$ leaves every agent’s prediction unchanged.

## 2.2 Graph constraints and the two settings

We study networks in which each agent sees one raw feature. An allocation a: $A  [ d ]$ specifies ${ \cal { S } } _ { i } = \{ a ( A _ { i } ) \}$ . Diferent agents may observe the same feature. Agents with no path to the output can be deleted without changing its prediction.

The depth $\mathrm { d e p t h } ( A _ { i } )$ of agent $A _ { i }$ is the number of agents on a longest directed path ending at $A _ { i }$ , so a source has depth one. The depth of the network $\mathrm { d e p t h } ( G )$ is $\operatorname { d e p t h } ( A _ { G } )$ , the depth of its output agent. After deleting agents with no path to $A _ { G }$ , this is also the maximum depth in G. We write $\begin{array} { r } { \Delta ^ { - } ( G ) = \operatorname* { m a x } _ { A _ { i } \in A } | \operatorname { P a } ( A _ { i } ) } \end{array}$ | for its maximum in-degree. Edges may skip depths, and an agent may send its prediction to any number of children.

For bounds on excess error, we must also fix the scale of the distribution. Otherwise, multiplying the label by a constant can make any positive excess error arbitrarily large. We bound the feature second moments and the sum of the absolute values of the global predictor’s coeficients.

Definition 2.3 (Normalized distribution). Fix constants $M _ { X } , A ^ { * } > 0$ . A distribution D is normalized at these bounds if $\| x _ { i } \| ^ { 2 } \leq M _ { X } ^ { 2 }$ for every $i \in [ d ]$ and there is a coeficient vector $\boldsymbol { w } ^ { * } \in \mathbb { R } ^ { d }$ such that $\begin{array} { r } { f ^ { * } = \sum _ { i = 1 } ^ { d } w _ { i } ^ { * } x _ { i } } \end{array}$ and $\textstyle \sum _ { i = 1 } ^ { d } | w _ { i } ^ { * } | \leq A ^ { * }$ . We write $\mathcal { C } _ { d }$ for the class of distributions with $M _ { X } = A ^ { * } = 1$

Dividing the features by $M _ { X }$ and the label by $A ^ { * } M _ { X }$ reduces these bounds to $M _ { X } = A ^ { * } = 1$ Thus excess-error bounds for the unit case are multiplied by $( A ^ { * } M _ { X } ) ^ { 2 }$ at the original scale. Our exact-aggregation results require only finite second moments and do not require normalization.

For positive integers $b , d , D$ , we take the infimum over finite DAGs $G = ( A , E )$ , feature allocations $a \colon A \to [ d ]$ , and output agents $A _ { G } \in A$ , subject to $\Delta ^ { - } ( G ) \leq b$ and $\operatorname { d e p t h } ( A _ { G } ) \leq D$ . We define

$$
R _ { b } ( d , D ) = \operatorname* { s u p } _ { \mathcal { D } \in \mathscr { C } _ { d } } \operatorname* { i n f } _ { G , a , A _ { G } } \| f ^ { * } - f _ { G } \| ^ { 2 } ,\tag{3}
$$

$$
\overline { { { R } } } _ { b } ( d , D ) = \operatorname* { i n f } _ { G , a , A _ { G } } \operatorname* { s u p } _ { \substack { ( \Delta ^ { - } ( G ) \leq b , \mathrm { ~ d e p t h } ( A _ { G } ) \leq D } } \| f ^ { * } - f _ { G } \| ^ { 2 } .\tag{4}
$$

In the adaptive designer setting, equation 3 chooses the best graph, allocation, and output agent for each distribution, then takes the worst error over distributions. In the oblivious designer setting, equation 4 first takes the worst error over distributions for each fixed graph, allocation, and output agent, then minimizes over these choices. Agents fit their coeficients from D in both settings.

Proposition 2.4. For all positive integers b, d, D, we have $0 \le R _ { b } ( d , D ) \le \overline { { R } } _ { b } ( d , D ) \le 1$ . Both quantities are non-increasing in b and D, and non-decreasing in d.

The proof is in Appendix B. After deleting agents with no path to the output, there are only finitely many DAGs and allocations under these constraints, up to relabeling. Consequently, $R _ { b } ( d , D ) = 0$ means that, after seeing any distribution, the designer can choose a graph, allocation, and output agent that achieve exact aggregation within these bounds. For $\overline { { R } } _ { b } ( d , D ) = 0$ , the designer can fix these choices before seeing the distribution and achieve exact aggregation for every distribution.

## 3 Related Work

Kearns, Roth, and Ryu [KRR26] introduce the networked information aggregation model. They prove an $O ( M / \sqrt { D } )$ excess-MSE bound on a path of depth D when every M consecutive agents collectively observe all features, under bounds on feature second moments and the global predictor’s coeficient $\ell _ { 1 }$ norm. Bateni et al. [BHH+26a] extend the protocol to binary classification, where agents minimize binary cross-entropy and pass logits, and prove an $O ( M / \sqrt { D } )$ excess-loss bound under the same coverage condition. Pal [Pal26] sharpens the lower bound for cyclic feature allocations and extends it to a class of convex losses, including logistic loss. Bateni et al. [BHH+26b] determine the optimal worst-case covered-path rate under fixed moment and coeficient bounds: excess error can remain constant through depth of order $M ^ { 2 }$ , and the optimal rate beyond that scale is $\Theta ( M ^ { 2 } / D )$ They obtain analogous bounds for logistic classification. These rate results concern given networks and feature allocations under certain conditions. We instead optimize over the graph, feature allocation, and output agent, and study the resulting worst-case excess error as a function of depth and the number of allowed parents.

The closest result to ours is Kearns, Roth, and Ryu [KRR26, Theorem 5.9], discussed in Section 1, which applies only to depth $D \ < \ d .$ . An extended discussion of other related work is in Appendix A.

## 4 One parent per agent

With at most one parent per agent, deleting agents with no path to the output leaves a path $A _ { 1 } , \ldots , A _ { n }$ with $n \leq D$

Kearns, Roth, and Ryu [KRR26, Theorem 5.9] give a $1 / ( D + 1 )$ excess-error lower bound for every graph and single-feature allocation, but require $D < d$ and use an unnormalized distribution. For every depth D, we give a normalized three-feature distribution with error $\Omega ( 1 / D )$ for every path allocation. Knowing the distribution lets the designer achieve a matching upper bound.

Theorem 4.1. For every $d \geq 3$ and $D \geq 1$ -，

$$
\frac { 1 } { 6 4 0 D } \leq R _ { 1 } ( d , D ) \leq \frac { 1 } { D + 1 } , \qquad \overline { { R } } _ { 1 } ( d , D ) \geq \frac { 1 } { 6 4 0 D } .
$$

In particular, $R _ { 1 } ( d , D ) = \Theta ( 1 / D )$

Thus no finite depth guarantees exact aggregation for every distribution with $d \geq 3$ features, even when the designer knows the distribution. We sketch both bounds below. The full proofs are in Appendix C, including the cases of one or two features in Appendix C.1.

## 4.1 Lower bound

The features in our construction share a large common component and have small informative components. Similar geometry is used in the covered-path lower bound of Bateni et al. [BHH+26b].

Fix an integer $D \geq 1$ , choose $\rho > 0$ with $\rho ^ { 2 } = 1 / ( 4 0 D )$ , and let $U , V , Z$ be independent standard Gaussians. Define

$$
x _ { 1 } = Z + \rho U , \qquad x _ { 2 } = Z + \rho V , \qquad x _ { 3 } = Z - \rho ( U + V ) , \qquad Y = \rho U .\tag{5}
$$

Together, the features recover $Y = ( 2 x _ { 1 } - x _ { 2 } - x _ { 3 } ) / 3$ , so $f ^ { * } = Y$ . A constant rescaling gives a normalized distribution, as shown in Appendix C.4.

We show that an arbitrary graph with depth at most D has error at least $1 / ( 3 2 0 D )$ under this distribution. We reduce to the case where the path reaches error $\Omega ( \rho ^ { 2 } )$ and every later agent retains at least a $1 - 1 0 \rho ^ { 2 }$ fraction of its parent’s error, as shown in Appendix C.2. Bernoulli’s inequality and our choice of $\rho$ ensure that a constant fraction remains after at most D further steps, giving error $\Omega ( 1 / D )$

## 4.2 Upper bound in the adaptive designer setting

We give a greedy allocation that selects the feature most correlated with the remaining error, relative to its norm. On a path $A _ { 1 } , \dotsc , A _ { D }$ , set $f _ { 0 } = 0$ and, for $t = 0 , \ldots , D - 1$ , choose

$$
a ( A _ { t + 1 } ) \in \operatorname * { a r g m a x } _ { i \in [ d ] : \| x _ { i } \| > 0 } \frac { \vert \langle Y - f _ { t } , x _ { i } \rangle \vert } { \| x _ { i } \| } .\tag{6}
$$

Break ties arbitrarily. Agent $A _ { t + 1 }$ uses this feature and its parent’s prediction $f _ { t } .$ , with no parent when $t = 0$

For distributions in $\mathcal { C } _ { d }$ , write $e _ { t } = \| f ^ { * } - f _ { t } \| ^ { 2 }$ . The coeficient and feature bounds give $e _ { 0 } \leq 1$ and ensure that the selected feature has enough correlation with the residual. Adding a multiple of this feature to $f _ { t }$ gives the decrease proved in Lemma C.6:

$$
e _ { t + 1 } \leq e _ { t } - e _ { t } ^ { 2 } .
$$

This recurrence gives $e _ { D } \leq 1 / ( D + 1 )$ , as proved for general normalization bounds in Appendix C.3.

## 5 Three parents per agent

We now show that three parents per agent sufice for exact aggregation in both settings, and we find the optimal depth in each. Throughout this section, we use $f ^ { * }$ as the label, as permitted by Lemma 2.2. No normalization is needed for the constructions.

## 5.1 The oblivious designer setting

The designer must choose the same graph and allocation for every distribution. We organize this graph into rounds. Each round first finds a better prediction, if the current prediction is not exact. It then combines that prediction with the predictions from earlier rounds. This second step ensures that progress in one round is preserved in all later rounds.

Theorem 5.1. For every $d \geq 2$ , the designer can fix a graph, a single-feature allocation, and an output agent that achieve exact aggregation for every distribution with finite second moments, using at most $4 d ^ { 2 }$ agents, at most three parents per agent, and depth $O ( d \log d )$ . For $d = 1$ , one agent $s u f f i c e s$

In particular, $\overline { { R } } _ { b } ( d , D ) = 0$ for $b \geq 3$ once $D$ reaches this bound. The full proof is in $\mathrm { A p \mathrm { - } }$ pendix D.1 and below we give a sketch of the construction.

We now describe the construction. Start with a source observing the fixed feature $x _ { 1 }$ , whose prediction is $p _ { 0 } = P _ { \mathrm { s p a n } \{ x _ { 1 } \} } f ^ { * }$ . Let $V _ { 0 } = \operatorname { s p a n } \{ x _ { 1 } \}$ . We build the rest of the graph one round at a time. At round $t + 1 ,$ , we find a vector $q _ { t }$ that improves on the current prediction $p _ { t }$ if it is not already exactly $f ^ { * }$ . This would mean that $q _ { t } \notin V _ { t }$ and thus contains a new direction. We then set $V _ { t + 1 } = V _ { t } +$ span{q<sub>t</sub>} and compute the prediction $p _ { t + 1 } = P _ { V _ { t + 1 } } f ^ { * }$ . We will show that the following invariant holds for every $0 \leq t < d - 1$

$$
V _ { t + 1 } = { \mathrm { s p a n } } \{ x _ { 1 } , p _ { 0 } , \ldots , p _ { t } , q _ { t } \} = { \mathrm { s p a n } } \{ x _ { 1 } , p _ { 0 } , \ldots , p _ { t } , p _ { t + 1 } \} .\tag{7}
$$

Since the remaining $d - 1$ features span H together with $V _ { 0 }$ , the space can grow at most $d - 1$ times. It grows whenever $p _ { t } \neq f ^ { * }$ , so $p _ { d - 1 } = f ^ { * }$ , as proved in Appendix D.1.

To find a new direction $q _ { t }$ , we look for an improvement using each raw feature. For every $i = 2 , \ldots , d ,$ , add an agent observing $x _ { i }$ and receiving $p _ { t }$ . If $p _ { t } \neq f ^ { * }$ , the residual $f ^ { * } - p _ { t }$ is a nonzero vector in H, so it has a nonzero inner product with some raw feature. That feature cannot be $x _ { 1 }$ since the residual is orthogonal to $V _ { t }$ . At least one of these agents therefore strictly improves on $p _ { t }$

To collect this improvement in a single prediction, feed all these agents’ predictions, together with $p _ { t }$ , into a fixed balanced binary tree. We denote the prediction of the root of this tree as $q _ { t }$ Each internal agent observes $x _ { 1 }$ and receives its two child predictions and $p _ { t }$ . By Lemma 2.1, its error is no larger than either child’s error, so the root prediction $q _ { t }$ keeps any improvement found at the leaves. The tree uses $O ( d )$ new agents and adds $O ( \log d )$ to the depth. Since $p _ { t }$ is already the best predictor in $V _ { t } ,$ , a strict improvement also proves that $q _ { t } \notin V _ { t }$

It remains to add this direction to $V _ { t }$ to get $V _ { t + 1 }$ . We want to compute

$$
p _ { t + 1 } = P _ { V _ { t } + \mathrm { s p a n } \{ q _ { t } \} } f ^ { * } .\tag{8}
$$

By equation 7 and the initialization, we have $V _ { t } + \operatorname { s p a n } \{ q _ { t } \} = \operatorname { s p a n } \{ x _ { 1 } , p _ { 0 } , \dots , p _ { t } , q _ { t } \}$ . We use a gadget that combines these vectors through a balanced tree, with at most three parents per agent.

The gadget recursively splits $p _ { 0 } , \ldots , p _ { t }$ into two nearly equal intervals sharing a boundary prediction. Each interval computes the best prediction from its entries and $x _ { 1 } , p _ { t } , q _ { t }$ . An agent observing x<sub>1</sub> merges the two interval predictions with a third fitted from $x _ { 1 } , p _ { t } , q _ { t }$ and their shared boundary. The nested projections $p _ { i } = P _ { V _ { i } } f ^ { * }$ make this merge exact by Lemma D.3. The recursion stops at pairs $p _ { i } , p _ { i + 1 }$ , handled by Lemma D.1. The gadget depends only on t, and its added agents observe $x _ { 1 }$ and have at most three parents. It adds $O ( t )$ agents and $O ( \log t )$ depth (Lemma D.4 in Appendix D.1).

Applying this gadget gives equation 8 and preserves equation 7, as proved in Appendix D.1. We run $d - 1$ rounds and use $p _ { d - 1 }$ as the output. Since each round uses $O ( d )$ agents and adds O(log d) to the depth, the total number of agents is $O ( d ^ { 2 } )$ and the total depth is $O ( d \log d )$

## 5.2 The adaptive designer setting

When the designer knows the distribution, it can choose which feature to use next. This reduces the depth bound for exact aggregation to the feature rank r.

Theorem 5.2. For every distribution with finite second moments and feature rank $r \geq 1$ , the designer can choose a graph, a single-feature allocation, and an output agent achieving exact aggregation with at most three parents per agent, depth at most $r ,$ and at most $1 + { \bigl ( } { \bigl . } { \bigr ) }$ agents. In particular, $R _ { b } ( d , D ) = 0$ for $b \geq 3$ and $D \geq d .$

We sketch the construction below. The proof and further construction details are in $\mathrm { A p \mathrm { - } }$ pendix D.2.

If $f ^ { * } = 0$ , one source sufices. Otherwise, choose linearly independent features $x _ { 1 } , \ldots , x _ { r }$ spanning H. We construct sets $J _ { t } \subseteq [ r ]$ of t selected features, starting with $J _ { 0 } = \emptyset$ . Write

$$
H _ { t } = \operatorname { s p a n } \{ x _ { j } : j \in J _ { t } \} , \qquad p _ { t } = P _ { H _ { t } } f ^ { * } , \qquad q _ { t , i } = P _ { H _ { t } + \operatorname { s p a n } \{ x _ { i } \} } f ^ { * } \quad ( i \notin J _ { t } ) .
$$

We stop as soon as a selected prediction $p _ { t }$ equals $f ^ { * }$ , using its agent as the output.

After round t, we will have agents predicting p<sub>t</sub> and $q _ { t , i }$ for every $i \notin J _ { t }$ . Consider round $t + 1$ . First, compare the improvements $\| q _ { t , i } - p _ { t } \| ^ { 2 }$ and select $j \notin J _ { t }$ with the smallest nonzero improvement. Set $J _ { t + 1 } = J _ { t } \cup \{ j \}$ and $p _ { t + 1 } = q _ { t , j }$ . This choice makes the update below exact, as proved in Appendix D.2.

The agent predicting $q _ { t , j }$ already predicts $p _ { t + 1 }$ by definition. Unless we stop, what remains is constructing the agents predicting $q _ { t + 1 , i }$ for $i \notin J _ { t + 1 }$ . We will show that an agent observing $x _ { i }$ and receiving $p _ { t } , p _ { t + 1 } , q _ { t , i }$ predicts $q _ { t + 1 , i } .$

Each round adds at most $r - t$ new agents and one depth, so the total number of agents is at most $\begin{array} { r } { 1 + \sum _ { t = 0 } ^ { r - 1 } ( r - t ) = 1 + \binom { r } { 2 } } \end{array}$ and the total depth is at most r. The output agent predicts $p _ { r } = f ^ { * }$

## 5.3 Depth lower bounds

We now show that the depth bounds in the two settings are optimal up to constant factors. We use a normalized version of the Gaussian example from Kearns, Roth, and Ryu [KRR26, Theorem 5.9]. Let $Z _ { 0 } , \dots , Z _ { d - 1 }$ be independent standard Gaussians. Define

$$
x _ { i } = \frac { Z _ { i - 1 } - Z _ { i } } { \sqrt 2 } \quad ( 1 \leq i < d ) , \qquad x _ { d } = \frac { Z _ { d - 1 } } { \sqrt 2 } , \qquad Y = \frac { Z _ { 0 } } { \sqrt 2 d } .\tag{9}
$$

The features sum to give $Y = d ^ { - 1 } \textstyle \sum _ { i = 1 } ^ { d } x _ { i }$ , so $f ^ { * } = Y$ and the global coeficient norm is one. Each feature has second moment at most one. Thus the distribution is normalized.

Exact aggregation for this distribution requires a path that observes $x _ { 1 } , \ldots , x _ { d }$ in order. Only $x _ { 1 }$ is correlated with the label. If the incoming predictions lie in span $\{ x _ { 1 } , \ldots , x _ { \ell } \}$ for some $\ell < d ,$ every feature beyond $x _ { \ell + 1 }$ is independent of the label and those predictions. Thus $x _ { \ell + 1 }$ is the only feature that can extend the prediction beyond this span. This refines the propagation argument used to prove Kearns, Roth, and Ryu [KRR26, Theorem 5.9]. The proof is given in Proposition D.6 in Appendix D.3.

Theorem 5.3. In the adaptive designer setting, for every $d \geq 1$ , some normalized rank-d distribution requires depth $D \geq d$ for exact aggregation, even without a parent limit. In the oblivious designer setting, every fixed parent limit $b \geq 2$ requires $D = \Omega ( d \log d )$ for exact aggregation.

## 6 Two parents per agent

We obtain the two-parent constructions by replacing each agent with three parents by a fixed gadget that uses at most two parents per agent. The gadget uses the same raw feature and reproduces the agent’s prediction with only a constant increase in size and depth.

Lemma 6.1. Consider an agent observing a raw feature $x _ { j }$ and receiving three parent predictions $f _ { 1 } , f _ { 2 } , f _ { 3 }$ . There is a fixed gadget that reproduces this agent’s prediction

$$
P _ { \mathrm { s p a n } \{ x _ { j } , f _ { 1 } , f _ { 2 } , f _ { 3 } \} } Y
$$

for every distribution with finite second moments. The gadget uses $O ( 1 )$ agents, each observing $x _ { j }$ and having at most two parents, and adds $O ( 1 )$ depth above the original parents.

We sketch the proof and give the details in Appendix E. We first build a replacement that may depend on the distribution, and then remove this dependence. Every agent in the gadget observes $x _ { j }$ , so the part of each prediction along $x _ { j }$ can be split of, and we ignore $x _ { j }$ here (Lemma E.9). The goal is the three-parent agent’s prediction $g = P _ { \mathrm { s p a n } \{ f _ { 1 } , f _ { 2 } , f _ { 3 } \} } Y$ , but each new agent can fit Y from only two available predictions. Consider the map $n ( \ddot { f } ) = ( \| g \| ^ { 2 } / \| f \| ^ { 2 } ) f - g$ on nonzero predictions $f .$ . Every fit $f$ satisfies $\langle g , f \rangle = \| f \| ^ { 2 }$ by Lemma 2.1, so $n ( f )$ is orthogonal to $g .$ . Thus the map sends every prediction into the two-dimensional space of vectors in span $\left\{ f _ { 1 } , f _ { 2 } , f _ { 3 } \right\}$ orthogonal to $^ { g , }$ and $n ( g ) = 0$ . Fitting from two predictions $f _ { 1 } , f _ { 2 }$ gives the prediction whose image is the point closest to zero on the line through $n ( f _ { 1 } )$ and $n ( f _ { 2 } )$ (Lemma E.1). So we must reach zero using only such steps.

We move to the complex plane by identifying this space with $\mathbb { C } .$ Multiplying all points by a complex number rotates and scales the plane about zero, so it commutes with the step. Hence if some steps turn three points $T = ( T _ { 1 } , T _ { 2 } , T _ { 3 } )$ into $\mu T = ( \mu T _ { 1 } , \mu T _ { 2 } , \mu T _ { 3 } )$ , the same steps turn $\mu T$ into $\mu ^ { 2 } T$ . Combining such steps, we can evaluate expressions in $\mu .$ . We construct a suitable $\mu$ and an expression in $\mu$ that equals zero. Evaluating it takes at most 300 steps, each done by one agent, and gives zero, the image of $g$ (Lemma E.7). To remove the dependence on the distribution, the fixed gadget runs all two-parent graphs with at most 303 agents in parallel and combines their outputs through a binary tree. Its size and depth are still constant but very large. Applying it to Theorems 5.1 and 5.2 gives the same asymptotic bounds for $b = 2$ in both settings (Appendix E.3).

Theorem 6.2. With at most two parents per agent and a single-feature allocation, exact aggregation is possible for every distribution with finite second moments. In the oblivious designer setting, $O ( d ^ { 2 } )$ agents and depth O(d log d) sufice for $d \geq 2$ , and one agent sufices for $d = 1$ . In the adaptive designer setting, $O ( r ^ { 2 } )$ agents and depth $O ( r )$ sufice for every feature rank $r \geq 1$

## 7 Lower bound on the number of agents for exact aggregation

The constructions in Sections 5 and 6 use $O ( d ^ { 2 } )$ agents. We prove a matching lower bound in the adaptive designer setting for every fixed parent limit $b \geq 2$ . The adversary chooses one distribution for which every exact network needs this many agents, even when the designer knows the distribution and the depth is unrestricted.

Theorem 7.1. For every $d \geq 2$ , the adversary can choose a distribution, normalized with $M _ { X } =$ $A ^ { * } = 1 ,$ , with d linearly independent features and $Y ~ = ~ f ^ { * }$ such that every single-feature DAG $G = ( A , E )$ achieving exact aggregation satisfies

$$
\sum _ { i = 1 } ^ { | A | } { \binom { | \operatorname* { P a } ( A _ { i } ) | + 1 } { 2 } } \geq { \binom { d } { 2 } } .\tag{10}
$$

If each agent has at most $b \geq 1$ parents, then each summand in equation 10 is at most $\binom { b + 1 } { 2 }$ giving $| A | = \Omega ( d ^ { 2 } / b ^ { 2 } )$ . For fixed $b \geq 2$ , this proves an $\Omega ( d ^ { 2 } )$ lower bound in the adaptive designer setting. The same bound holds in the oblivious designer setting, since a graph and allocation fixed in advance must also achieve exact aggregation on this distribution. Together with the constructions in Sections 5 and 6, this makes $\Theta ( d ^ { 2 } )$ agents optimal in both settings.

We will first pick a matrix E with certain properties and define the distribution using this matrix. Choose a symmetric $d \times d$ matrix E with $| E _ { i j } | < 1 / ( 8 d )$ for all $i , j \in [ d ]$ such that the upper-triangular entries of $\Sigma = { \textstyle { \frac { 1 } { 2 } } } I + E$ are algebraically independent over $\mathbb { Q } \mathrm { : }$ no nonzero polynomial with rational coeficients vanishes at these entries. Such a matrix exists by Lemma F.1, proved in Appendix F.1. The algebraic independence will be used in Lemma 7.3. Set $\Sigma = { \textstyle { \frac { 1 } { 2 } } } I + E$ and $c = \mathbf { 1 } _ { d } / ( 4 d )$ , with ${ \bf 1 } _ { d }$ the vector of all ones in $\mathbb { R } ^ { d }$ . Define

$$
\boldsymbol { x } \sim \boldsymbol { N } ( 0 , \Sigma ) , \qquad \boldsymbol { w } ^ { * } = \Sigma ^ { - 1 } \boldsymbol { c } , \qquad \boldsymbol { Y } = \boldsymbol { w } ^ { * } \overset { \intercal } { \boldsymbol { x } } .\tag{11}
$$

By Lemma F.2, the distribution is normalized with $M _ { X } = A ^ { * } = 1$ . The definitions also give $\mathbb { E } [ x x ^ { \top } ] = \Sigma$ and $\mathbb { E } [ x Y ] = \Sigma w ^ { * } = c .$ Fix a graph $G = ( A , E )$ and a single-feature allocation on the chosen distribution. Each fitted prediction has a unique representation $f _ { i } = w _ { i } ^ { \top }$ x with $w _ { i } \in \mathbb { R } ^ { d }$ since the features are linearly independent by Lemma F.2. By equation 35, the squared error of $f _ { i }$ is $\mathbb { E } [ Y ^ { 2 } ] + w _ { i } ^ { \top } \Sigma w _ { i } - 2 w _ { i } ^ { \top } c .$ . Let $e _ { 1 } , \ldots , e _ { d }$ be the coordinate vectors. Agent $A _ { i }$ observing x<sub>ℓ</sub> fits a linear combination of this feature and its parents’ predictions. Omitting the constant $\mathbb { E } [ Y ^ { 2 } ]$ therefore gives

$$
w _ { i } = \operatorname * { a r g m i n } _ { \substack { w \in \mathrm { s p a n } ( \{ e _ { \ell } \} \cup \{ w _ { j } : A _ { j } \in \mathrm { P a } ( A _ { i } ) \} ) } } \left( w ^ { \top } \Sigma w - 2 w ^ { \top } c \right) .\tag{12}
$$

We seek a nonzero symmetric matrix $\Delta$ such that replacing Σ by $\Sigma + t \Delta$ , with c fixed, preserves every fitted coeficient vector. For suficiently small |t|, the new matrix remains positive definite and defines a distribution as in equation 11. For fixed parent coeficient vectors, the objective in equation 12 changes by $t w ^ { \top } \Delta w$ . Requiring $u ^ { \top } \Delta v = 0$ for every pair of input coeficient vectors at each agent, including $u = v$ , makes this change zero throughout each allowed span by bilinearity. Induction along the graph then preserves all fitted coeficients. The following lemma constructs such a ∆ under the stated count bound, while Lemma 7.3 rules it out for an exact network on our chosen distribution.

Lemma 7.2. Fix a single-feature DAG on a distribution with d linearly independent features. If $\begin{array} { r } { \sum _ { i = 1 } ^ { | A | } { \binom { | \operatorname { P a } ( A _ { i } ) | + 1 } { 2 } } < { \binom { d } { 2 } } } \end{array}$ , there is a nonzero symmetric matrix $\Delta$ with zero diagonal such that

$$
u ^ { \top } \Delta v = 0\tag{13}
$$

for every agent $A _ { i }$ observing $x _ { \ell }$ and all $u , v \in \{ e _ { \ell } \} \cup \{ w _ { j } : A _ { j } \in \operatorname { P a } ( A _ { i } ) \}$ , including $u = v$

The proof, in Appendix F.2, counts equations. The matrix $\Delta$ has $\binom { d } { 2 }$ unknown entries, and an agent with p parents imposes at most $p + \binom { p } { 2 } = \binom { p + 1 } { 2 }$ homogeneous linear equations on them. The next lemma, proved in Appendix F.3, rules out a nonzero $\Delta$ for an exact network on our distribution. Together with Lemma 7.2, it gives Theorem 7.1 (Appendix F.4).

Lemma 7.3. For the distribution fixed in Equation (11), suppose a single-feature DAG achieves exact aggregation. If a symmetric matrix $\Delta$ satisfies equation 13 at every agent, then $\Delta = 0$

## AI use statement

We used OpenAI’s GPT-6 Astra in Pro mode to help develop critical ingredients for proving mathematical claims, including key proof ideas, and to assist with proof writing. Specifically, it assisted with the construction and proof of the gadget that replaces a three-parent agent by two-parent agents in Section 6 and appendix E. It also assisted with the proof of the lower bound on the number of agents in Section 7 and appendix F. We did not use generative AI tools to develop theoretical models or conceptual frameworks, formulate mathematical claims, propose or refine hypotheses, design or assess research methods or experiments, or interpret results. Generating synthetic datasets, implementing methods, translation, cleaning or reformatting datasets, and qualitative or thematic data analysis are not applicable to this work. We verified the correctness of all AI-assisted proofs. We take responsibility for the final content of this paper, including all AI-assisted work.

## References

[Aum76] Robert J Aumann. “Agreeing to Disagree”. In: The Annals of Statistics 4.6 (1976), pp. 1236–1239.

[BHH+26a] MohammadHossein Bateni, Zahra Hadizadeh, MohammadTaghi Hajiaghayi, Mahdi JafariRaviz, and Shayan Taherijam. “Networked Information Aggregation for Binary Classification”. In: CoRR abs/2605.01082 (2026). doi: 10.48550/ARXIV.2605. 01082. arXiv: 2605.01082.

[BHH+26b] MohammadHossein Bateni, Zahra Hadizadeh, MohammadTaghi Hajiaghayi, Mahdi JafariRaviz, and Shayan Taherijam. “Optimal Rates for Agentic Networked Information Aggregation”. In: arXiv preprint arXiv:2609.05318 (2026).

[CFJ+21] Kewei Cheng, Tao Fan, Yilun Jin, Yang Liu, Tianjian Chen, Dimitrios Papadopoulos, and Qiang Yang. “SecureBoost: A Lossless Federated Learning Framework”. In: IEEE Intell. Syst. 36.6 (2021), pp. 87–98. doi: 10.1109/MIS.2021.3082561.

[CGG+25] Natalie Collina, Surbhi Goel, Varun Gupta, and Aaron Roth. “Tractable agreement protocols”. In: Proceedings of the 57th Annual ACM Symposium on Theory of Computing. 2025, pp. 1532–1543.

[CGG+26] Natalie Collina, Ira Globus-Harris, Surbhi Goel, Varun Gupta, Aaron Roth, and Mirah Shi. “Collaborative prediction: Tractable information aggregation via agreement”. In: Proceedings of the 2026 Annual ACM-SIAM Symposium on Discrete Algorithms (SODA). SIAM. 2026, pp. 4712–4798.

[DeG74] Morris H DeGroot. “Reaching a consensus”. In: Journal of the American Statistical Association 69.345 (1974), pp. 118–121.

[DF03] D.S. Dummit and R.M. Foote. Abstract Algebra. Wiley, 2003. isbn: 9780471433347. url: https://books.google.com/books?id=KJDBQgAACAAJ.

[GCW+24] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. “Large Language Model Based Multiagents: A Survey of Progress and Challenges”. In: Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI 2024, Jeju, South Korea, August 3-9, 2024. ijcai.org, 2024, pp. 8048–8057. doi: 10 . 24963 / IJCAI . 2024/890.

[GJ10] Benjamin Golub and Matthew O Jackson. “Naive learning in social networks and the wisdom of crowds”. In: American Economic Journal: Microeconomics 2.1 (2010), pp. 112–149.

[KGZ19] Michael P. Kim, Amirata Ghorbani, and James Y. Zou. “Multiaccuracy: Black-Box Post-Processing for Fairness in Classification”. In: Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society, AIES 2019, Honolulu, HI, USA, January 27- 28, 2019. 2019, pp. 247–254. doi: 10.1145/3306618.3314287.

[KRR26] Michael Kearns, Aaron Roth, and Emily Ryu. “Networked Information Aggregation via Machine Learning”. In: Proceedings of the 2026 Annual ACM-SIAM Symposium on Discrete Algorithms, SODA 2026, Vancouver, BC, Canada, January 11-14, 2026. SIAM. 2026, pp. 4799–4845. doi: 10.1137/1.9781611978971.173.

[Pal26] Ambar Pal. “Optimal Lower Bounds for Networked Information Aggregation”. In: arXiv preprint arXiv:2608.15472 (2026).

[Wol92] David H. Wolpert. “Stacked generalization”. In: Neural Networks 5.2 (1992), pp. 241– 259. doi: 10.1016/S0893-6080(05)80023-1.

[YLC+19] Qiang Yang, Yang Liu, Tianjian Chen, and Yongxin Tong. “Federated Machine Learning: Concept and Applications”. In: ACM Trans. Intell. Syst. Technol. 10.2 (2019), 12:1–12:19. doi: 10.1145/3298981.

## A Extended Related Work

Prediction exchange and calibration. Two parties with diferent features can learn from one another by taking turns making and revising predictions. Collina et al. [CGG+26] give protocols for this task whose predictions compete with a restricted class of policies on the parties’ joint feature space. The parties never share their raw features. Instead, the protocols call learning algorithms on each party’s own feature space, with guarantees for both online prediction and learning from a fixed distribution. This approach builds on the agreement protocols of Collina et al. [CGG+25], which use calibration conditions to relax the assumptions of Bayesian agreement. In the classical result of Aumann [Aum76], two agents with a common prior must agree on the probability of an event once their posterior probabilities are common knowledge. The calibration conditions support eficient prediction exchange without requiring the parties to know a common prior.

An auditor for multiaccuracy looks for a test function that correlates with a predictor’s errors. Finding one gives a direction in which to correct the predictor. Kim, Ghorbani, and Zou [KGZ19] use this idea to post-process a given predictor until the expected product of its residual with each test is small. An indicator test measures the mean residual within a group, weighted by the group’s probability. The normal equations for least squares give zero inner product between the residual and each input, whether that input is a raw feature or another learner’s prediction.

Social learning and opinion dynamics. Repeated averaging is the update rule in DeGroot [DeG74]: each agent takes a weighted average of its neighbors’ current beliefs, with the weights fixed throughout the process. When the agents reach consensus, the common belief is a weighted average of their initial beliefs. Golub and Jackson [GJ10] study whether this consensus approaches the true state as the network grows, assuming independent noisy initial estimates. They characterize learning through the weights in the final consensus. Convergence to the true state holds precisely when the largest weight on any individual’s initial estimate tends to zero.

Stacking and distributed learning. In stacked generalization, a second learner is trained to combine predictions from other models [Wol92]. Its training inputs are predictions on examples held out when fitting those models. This gives the second learner evidence about their errors on unseen data. Vertical federated learning organizes collaboration around training a shared model from feature columns held by diferent parties [YLC+19]. The parties have records for common examples but keep their raw data local. For example, SecureBoost trains boosted trees by exchanging encrypted gradient statistics [CFJ+21]. These statistics allow the parties to evaluate candidate splits using features held at diferent sites, and the exchanges continue as further splits and trees are added.

## B Proofs for the preliminaries

We restate and prove the lemmas from Section 2.

Lemma 2.1. Let $V \subseteq L ^ { 2 } ( { \mathcal { D } } )$ be a finite-dimensional subspace and let $f = P _ { V } Y$ . Then $\langle Y , f \rangle =$ $\| f \| ^ { 2 }$ , and for every $g \in V$ 2

$$
\operatorname { M S E } ( g ) - \operatorname { M S E } ( f ) = \| g - f \| ^ { 2 } .\tag{2}
$$

Proof. The residual $Y - f$ is orthogonal to V . Taking its inner product with $f \in V$ gives $\langle Y , f \rangle =$ $\langle Y - f , f \rangle + \langle f , f \rangle = \| f \| ^ { 2 }$ . Since $f - g \in V$ , the two terms in $Y - g = ( Y - f ) + ( f - g )$ are orthogonal. The Pythagorean identity gives equation 2. □

Lemma 2.2. Replacing Y by $f ^ { * }$ leaves every agent’s prediction unchanged.

Proof. Let $f _ { i }$ and $f _ { i } ^ { \prime }$ be the fitted predictions for the labels $Y$ and $f ^ { * }$ , respectively. We will show, by induction on a topological order of the agents, that $f _ { i } = f _ { i } ^ { \prime }$ for every agent $A _ { i } .$ . For any agent, assume that its incoming predictions are unchanged for the two labels. Then its input space $V _ { i }$ is unchanged for the two labels. Since $Y - f ^ { * }$ is orthogonal to $V _ { i }$ and projection is linear, we have $f _ { i } = P _ { V _ { i } } Y = P _ { V _ { i } } ( Y - f ^ { * } ) + P _ { V _ { i } } f ^ { * } = P _ { V _ { i } } f ^ { * } = f _ { i } ^ { \prime }$ □

Proposition 2.4. For all positive integers $b , d , D$ , we have $0 \le R _ { b } ( d , D ) \le \overline { { R } } _ { b } ( d , D ) \le 1$ . Both quantities are non-increasing in b and D, and non-decreasing in d.

Proof. Excess error is nonnegative because $f ^ { * }$ minimizes MSE over H. Every agent can use the zero predictor, so its fitted MSE is at most MSE(0). By Lemma 2.1, its excess error is therefore at most $\| f ^ { * } \| ^ { 2 }$ . The unit bounds give $\begin{array} { r } { \| f ^ { * } \| \leq \sum _ { i = 1 } ^ { d } | w _ { i } ^ { * } | \| x _ { i } \| \leq 1 } \end{array}$

For any fixed graph, allocation, and output agent, the worst error over distributions is at least the best achievable error on each distribution. Taking the supremum of the latter over distributions gives $R _ { b } ( d , D )$ . Taking the infimum of the former over graphs, allocations, and output agents then proves $R _ { b } ( d , D ) \leq \overline { { R } } _ { b } ( d , D )$

Increasing b or D allows more choices in each infimum and thus cannot increase either quantity. Increasing d cannot decrease either quantity, since the adversary can set $x _ { d + 1 } = x _ { 1 }$ to recover the d-feature problem. □

## C Proofs for one parent per agent

## C.1 One or two features

Proposition C.1. For every distribution with finite second moments and $d \ \leq \ 2$ features, the designer can choose a feature allocation on a path of d agents that achieves exact aggregation. For $d = 2$ , the fixed allocation x<sub>1</sub>, x<sub>2</sub>, x<sub>1</sub> on a path of three agents achieves exact aggregation for every such distribution.

Proof. For $d = 1$ , a single agent sees the full feature space and predicts $f ^ { * }$ . For $d = 2$ , if both $\langle Y , x _ { 1 } \rangle$ and $\langle Y , x _ { 2 } \rangle$ are zero, then $f ^ { * } = 0$ and every agent predicts zero by Lemma 2.2. Otherwise, choose $i \in \{ 1 , 2 \}$ with $\langle Y , x _ { i } \rangle \neq 0$ and give $x _ { i }$ to the first agent. Its prediction is a nonzero multiple of $x _ { i }$ . Give the other feature to the second agent. Its inputs span $H ,$ so it predicts $f ^ { * }$

Now fix the allocation $x _ { 1 } , x _ { 2 } , x _ { 1 }$ . If $\langle Y , x _ { 1 } \rangle \neq 0 ,$ the first prediction is a nonzero multiple of $x _ { 1 }$ and the second agent predicts $f ^ { * }$ . The third agent also predicts $f ^ { * }$ , because its inputs include $f ^ { * }$ and lie in H. $\mathrm { I f } \left. Y , x _ { 1 } \right. = 0$ but $\langle Y , x _ { 2 } \rangle \neq 0$ , the first prediction is zero and the second is a nonzero multiple of $x _ { 2 }$ . The third agent then has inputs spanning H and predicts $f ^ { * }$ . If both inner products vanish, every prediction and $f ^ { * }$ are zero. □

## C.2 The lower bound

Use the distribution in equation 5, with $\rho ^ { 2 } = 1 / ( 4 0 D )$ . Together, the features determine the label: $f ^ { * } = ( 2 x _ { 1 } - x _ { 2 } - x _ { 3 } ) / 3 = Y$ . The sum of the absolute values of these coeficients is $4 / 3$ . The feature second moments are $1 + \rho ^ { 2 } , 1 + \rho ^ { 2 }$ , and $1 + 2 \rho ^ { 2 }$ , all at most 2. Thus the distribution satisfies constant bounds on the feature moments and global coeficients. Since $f ^ { * } = Y$ , an agent’s MSE is also its excess error.

We first prove the two estimates used to bound error along the path, then prove Theorem C.4.

Lemma C.2. For the distribution in equation 5, let f be the best linear predictor from any two distinct features. Then $\mathrm { M S E } ( f ) \ge \rho ^ { 2 } / 6$ and $\| \boldsymbol { f } \| ^ { 2 } \ge \rho ^ { 2 } / 5$

Proof. We compute the error for each pair by projecting Y onto the direction orthogonal to its span. We then use the error to bound the norm of the fit.

The variables $U , V , Z$ are orthonormal in $L ^ { 2 } ( \mathcal { D } )$ because they are independent standard Gaussians. Each pair of features is linearly independent: both features have $Z$ coeficient one, so they could be proportional only if they were equal, but their $U , V$ coeficients difer. Thus the orthogonal complement of each pair’s span within span $\{ U , V , Z \}$ is a line. Since Y also lies in this three-dimensional space, the residual $Y - f$ is its projection onto that line. For a nonzero vector $\xi$ on the line, this gives

$$
Y - f = { \frac { \langle Y , \xi \rangle } { \| \xi \| ^ { 2 } } } \xi , \qquad \operatorname { M S E } ( f ) = { \frac { \langle Y , \xi \rangle ^ { 2 } } { \| \xi \| ^ { 2 } } } .
$$

For the pair $\{ x _ { 1 } , x _ { 2 } \}$ , take $\xi = U + V - \rho Z$ . Using $x _ { 1 } = Z + \rho U$ and $x _ { 2 } = Z + \rho V$ , we have

$$
\langle \xi , x _ { 1 } \rangle = \rho - \rho = 0 , \qquad \langle \xi , x _ { 2 } \rangle = \rho - \rho = 0 .
$$

Since $Y = \rho U$ , we also have $\langle Y , \xi \rangle = \rho$ and $\| \xi \| ^ { 2 } = 1 + 1 + \rho ^ { 2 } = 2 + \rho ^ { 2 }$ . The error for this pair is therefore

$$
\mathrm { M S E } ( f ) = \frac { \rho ^ { 2 } } { 2 + \rho ^ { 2 } } .
$$

For the pair $\{ x _ { 1 } , x _ { 3 } \}$ , take $\xi = U - 2 V - \rho Z$ . Using $x _ { 3 } = Z - \rho U - \rho V$ , we get

$$
\langle \xi , x _ { 1 } \rangle = \rho - \rho = 0 , \qquad \langle \xi , x _ { 3 } \rangle = - \rho + 2 \rho - \rho = 0 .
$$

Here $\langle Y , \xi \rangle = \rho$ and $\| \xi \| ^ { 2 } = 1 + 4 + \rho ^ { 2 } = 5 + \rho ^ { 2 }$ , so the error is

$$
\mathrm { M S E } ( f ) = \frac { \rho ^ { 2 } } { 5 + \rho ^ { 2 } } .
$$

For the pair $\{ x _ { 2 } , x _ { 3 } \}$ , take $\xi = 2 U - V + \rho Z$ . In this case,

$$
\langle \xi , x _ { 2 } \rangle = - \rho + \rho = 0 , \qquad \langle \xi , x _ { 3 } \rangle = - 2 \rho + \rho + \rho = 0 .
$$

Now $\langle Y , \xi \rangle = 2 \rho$ and $\| \xi \| ^ { 2 } = 4 + 1 + \rho ^ { 2 } = 5 + \rho ^ { 2 }$ , giving

$$
\operatorname { M S E } ( f ) = { \frac { ( 2 \rho ) ^ { 2 } } { 5 + \rho ^ { 2 } } } = { \frac { 4 \rho ^ { 2 } } { 5 + \rho ^ { 2 } } } .
$$

Since $\rho ^ { 2 } \le 1 / 4 0$ , each denominator is at most 6 and each numerator is at least $\rho ^ { 2 }$ , so every pair has error at least $\rho ^ { 2 } / 6$ . The three errors are at most $\rho ^ { 2 } / 2 , \rho ^ { 2 } / 5$ , and $4 \rho ^ { 2 } / 5$ , respectively, so every pair also has error at most $4 \rho ^ { 2 } / 5$

Finally, the fit $f$ and its residual $Y - f$ are orthogonal, so $\| Y \| ^ { 2 } = \| f \| ^ { 2 } + \operatorname { M S E } ( f )$ . The upper bound on the error therefore gives

$$
\| f \| ^ { 2 } = \| Y \| ^ { 2 } - \mathrm { M S E } ( f ) \geq \rho ^ { 2 } - \frac { 4 \rho ^ { 2 } } { 5 } = \frac { \rho ^ { 2 } } { 5 } .
$$

Lemma C.3. For consecutive agents $A _ { t } , A _ { t + 1 }$ on a single-feature path for equation 5, $i f \parallel f _ { t } \parallel ^ { 2 } \geq$ $\rho ^ { 2 } / 5$ , then

$$
\mathrm { M S E } ( f _ { t + 1 } ) \geq ( 1 - 1 0 \rho ^ { 2 } ) \mathrm { M S E } ( f _ { t } ) .
$$

Proof. We show that the next agent can remove at most a $1 0 \rho ^ { 2 }$ fraction of its parent’s error. Let $x _ { i }$ and $x _ { j }$ be the features observed by $A _ { t }$ and $A _ { t + 1 }$ , respectively, and write $e = Y - f _ { t }$ for the parent’s residual. Since $f _ { t }$ is the fit from the parent’s inputs, e is orthogonal to their span. In particular, $e \perp x _ { i }$ and $e \perp f _ { t }$

The next agent receives $f _ { t }$ and observes $x _ { j } ,$ , so it can improve on $f _ { t }$ only through the part of $x _ { j }$ orthogonal to $f _ { t }$ . The hypothesis $\lVert f _ { t } \rVert \geq \rho / \sqrt { 5 } > 0$ lets us define this part as

$$
z = x _ { j } - { \frac { \langle x _ { j } , f _ { t } \rangle } { \| f _ { t } \| ^ { 2 } } } f _ { t } .
$$

Thus $z \perp f _ { t }$ and span $\left\{ f _ { t } , x _ { j } \right\} = \operatorname { s p a n } \{ f _ { t } , z \}$ . I $\therefore z = 0$ , the next agen $\gamma _ { \mathrm { s } }$ input span is just span $\{ f _ { t } \}$ Since $e \perp f _ { t }$ , its fit remains $f _ { t }$ , which proves the claim in this case.

Suppose now that $z \neq 0$ . The projection of Y onto the line spanned by $f _ { t }$ is $f _ { t } ,$ , because $Y = f _ { t } + e$ and $e \perp f _ { t }$ . Its projection onto the orthogonal line spanned by $z { \mathrm { ~ i s ~ } } \langle e , z \rangle z / \| z \| ^ { 2 }$ . The next agent therefore predicts

$$
f _ { t + 1 } = f _ { t } + { \frac { \langle e , z \rangle } { \| z \| ^ { 2 } } } z .
$$

Since $f _ { t }$ belongs to the next agent’s input span, Lemma 2.1 gives the exact decrease in error:

$$
\mathrm { M S E } ( f _ { t } ) - \mathrm { M S E } ( f _ { t + 1 } ) = \Vert f _ { t + 1 } - f _ { t } \Vert ^ { 2 } = \frac { | \langle e , z \rangle | ^ { 2 } } { \Vert z \Vert ^ { 2 } } .
$$

It remains to bound the numerator by $5 \rho ^ { 2 } \| e \| ^ { 2 }$ and the denominator from below by $1 / 2$

For the numerator, we use the fact that the features are close together. Subtracting any two features in equation 5 cancels their common term $Z .$ Since $U , V$ are orthonormal, the three squared distances are

$$
\begin{array} { r l } & { \| x _ { 1 } - x _ { 2 } \| ^ { 2 } = \rho ^ { 2 } \| U - V \| ^ { 2 } = 2 \rho ^ { 2 } , } \\ & { \| x _ { 1 } - x _ { 3 } \| ^ { 2 } = \rho ^ { 2 } \| 2 U + V \| ^ { 2 } = 5 \rho ^ { 2 } , } \\ & { \| x _ { 2 } - x _ { 3 } \| ^ { 2 } = \rho ^ { 2 } \| U + 2 V \| ^ { 2 } = 5 \rho ^ { 2 } . } \end{array}
$$

Thus $\| x _ { j } - x _ { i } \| ^ { 2 } \leq 5 \rho ^ { 2 }$ , also when $i = j$ . Since z difers from $x _ { j }$ by a multiple of $f _ { t }$ and $e \perp f _ { t }$ , we have $\langle e , z \rangle = \langle e , x _ { j } \rangle$ . Using $e \perp x _ { i }$ and then Cauchy–Schwarz gives

$$
\begin{array} { r l r } {  { | \langle e , z \rangle | ^ { 2 } = | \langle e , x _ { j } - x _ { i } \rangle | ^ { 2 } } } \\ & { } & { \leq \| e \| ^ { 2 } \| x _ { j } - x _ { i } \| ^ { 2 } \leq 5 \rho ^ { 2 } \| e \| ^ { 2 } . } \end{array}
$$

For the denominator, we show that removing the component along $f _ { t }$ leaves most of the squared norm of $x _ { j }$ . Since $Y = \rho U$ and $U , V , Z$ are orthonormal,

$$
\langle Y , x _ { 1 } \rangle = \rho ^ { 2 } , \qquad \langle Y , x _ { 2 } \rangle = 0 , \qquad \langle Y , x _ { 3 } \rangle = - \rho ^ { 2 } .
$$

Thus $| \langle Y , x _ { i } \rangle | \leq \rho ^ { 2 }$ for every feature. Since $e = Y - f _ { t }$ is orthogonal to $x _ { i }$ , this also gives $| \langle f _ { t } , x _ { i } \rangle | =$ $| \langle Y , x _ { i } \rangle | \leq \rho ^ { 2 }$

Writing $x _ { j } = x _ { i } + \left( x _ { j } - x _ { i } \right)$ , the triangle inequality followed by Cauchy–Schwarz gives

$$
\begin{array} { r l } & { \left| \langle f _ { t } , x _ { j } \rangle \right| \leq \left| \langle f _ { t } , x _ { i } \rangle \right| + \left| \langle f _ { t } , x _ { j } - x _ { i } \rangle \right| } \\ & { \qquad \leq \left| \langle f _ { t } , x _ { i } \rangle \right| + \left\| f _ { t } \right\| \left\| x _ { j } - x _ { i } \right\| . } \end{array}
$$

Dividing by $\| f _ { t } \|$ and using the hypothesis $\| \boldsymbol { f } _ { t } \| \ge \rho / \sqrt { 5 }$ , together with the bounds just proved, we obtain

$$
\begin{array} { r l r } {  { \frac { | \langle f _ { t } , x _ { j } \rangle | } { \| f _ { t } \| } \leq \frac { | \langle f _ { t } , x _ { i } \rangle | } { \| f _ { t } \| } + \| x _ { j } - x _ { i } \| } } \\ & { } & { \leq \frac { \rho ^ { 2 } } { \rho / \sqrt { 5 } } + \sqrt { 5 } \rho = 2 \sqrt { 5 } \rho . } \end{array}
$$

The left-hand side is the norm of the component removed from $x _ { j }$ to obtain $z .$ The three features have squared norms $1 + \rho ^ { 2 } , 1 + \rho ^ { 2 }$ , and $1 + 2 \rho ^ { 2 }$ , so $\| \boldsymbol { x } _ { j } \| ^ { 2 } \geq 1$ . Since the removed component is orthogonal to z, Pythagoras gives

$$
\| z \| ^ { 2 } = \| x _ { j } \| ^ { 2 } - \frac { | \langle x _ { j } , f _ { t } \rangle | ^ { 2 } } { \| f _ { t } \| ^ { 2 } } \geq 1 - 2 0 \rho ^ { 2 } \geq \frac { 1 } { 2 } .
$$

The last inequality uses $\rho ^ { 2 } = 1 / ( 4 0 D ) \le 1 / 4 0$

Substituting the numerator and denominator bounds into the error decrease formula, and using $\| e \| ^ { 2 } = \mathrm { M S E } ( f _ { t } )$ , we conclude that

$$
\mathrm { M S E } ( f _ { t } ) - \mathrm { M S E } ( f _ { t + 1 } ) \leq \frac { 5 \rho ^ { 2 } \| e \| ^ { 2 } } { 1 / 2 } = 1 0 \rho ^ { 2 } \mathrm { M S E } ( f _ { t } ) .
$$

Rearranging proves the claim.

Theorem C.4. For every integer $D \geq 1$ , the distribution in equation 5 satisfies

$$
\mathrm { M S E } ( f _ { G } ) - \mathrm { M S E } ( f ^ { * } ) \geq { \frac { 1 } { 3 2 0 D } }
$$

for every graph $G$ with $\Delta ^ { - } ( G ) \leq 1$ and depth $( A _ { G } ) \leq D$ , every single-feature allocation, and every output agent $A _ { G }$

Proof. Since every agent has at most one parent, tracing parents backward from the output gives a single path. All other agents can be removed because their predictions do not afect the output. Write the remaining agents as $A _ { 1 } , \ldots , A _ { n }$ , with $A _ { n } = A _ { G }$ . The depth bound gives $n \leq D$ . Since $f ^ { * } = Y$ , it sufices to show that MSE $\ ' ( f _ { n } ) \ge 1 / ( 3 2 0 D )$

We first handle paths that never combine a nonzero prediction with a diferent raw feature. For the other paths, Lemma C.2 will give an initial error bound, and Lemma C.3 will bound the improvement at each later agent.

An agent observing x with no parent or a zero parent prediction has input span span $\{ x _ { 2 } \}$ . Its fit is zero because $\langle Y , x _ { 2 } \rangle = 0$ . By induction along the path, all predictions are therefore zero until an agent observes $x _ { 1 }$ or x . If neither feature appears, the output error is $\mathrm { M S E } ( f _ { n } ) = \| Y \| ^ { 2 } = \rho ^ { 2 } =$ $1 / ( 4 0 D ) \geq 1 / ( 3 2 0 D )$

Otherwise, let $A _ { s }$ be the first agent observing a feature $x _ { i } \in \{ x _ { 1 } , x _ { 3 } \}$ . Its parent, if present, predicts zero, so

$$
f _ { s } = { \frac { \langle Y , x _ { i } \rangle } { \| x _ { i } \| ^ { 2 } } } x _ { i } \neq 0 .
$$

The prediction is nonzero because $\langle Y , x _ { 1 } \rangle = \rho ^ { 2 }$ and $\langle Y , x _ { 3 } \rangle = - \rho ^ { 2 }$ . If the next agent also observes $x _ { i } ,$ , its inputs still span span $\{ x _ { i } \}$ , so its fit is again $f _ { s }$ . This remains true for as long as the path repeats $x _ { i }$

If every agent after $A _ { s }$ observes $x _ { i }$ , then $f _ { n } = f _ { s }$ . The formula for $f _ { s }$ gives

$$
\| f _ { s } \| ^ { 2 } = { \frac { | \langle Y , x _ { i } \rangle | ^ { 2 } } { \| x _ { i } \| ^ { 2 } } } = { \frac { \rho ^ { 4 } } { \| x _ { i } \| ^ { 2 } } } \leq \rho ^ { 4 } ,
$$

since $\| x _ { 1 } \| ^ { 2 } = 1 + \rho ^ { 2 }$ and $\| x _ { 3 } \| ^ { 2 } = 1 + 2 \rho ^ { 2 }$ are at least one. The fit and its residual are orthogonal, so

$$
\operatorname { M S E } ( f _ { n } ) = \| Y \| ^ { 2 } - \| f _ { s } \| ^ { 2 } \geq \rho ^ { 2 } - \rho ^ { 4 } \geq { \frac { \rho ^ { 2 } } { 2 } } = { \frac { 1 } { 8 0 D } } \geq { \frac { 1 } { 3 2 0 D } } .
$$

Here we used $\rho ^ { 2 } = 1 / ( 4 0 D ) \le 1 / 4 0 < 1 / 2 .$

It remains to consider paths that use a diferent feature after $A _ { s }$ . Let $A _ { \tau }$ be the first such agent, and call its feature $x _ { j }$ . Its parent still predicts $f _ { s } ,$ , a nonzero multiple of $x _ { i } .$ , so

$$
\operatorname { s p a n } \{ f _ { \tau - 1 } , x _ { j } \} = \operatorname { s p a n } \{ x _ { i } , x _ { j } \} .
$$

Thus $f _ { \tau }$ is the fit from two distinct raw features. By Lemma C.2,

$$
\mathrm { M S E } ( f _ { \tau } ) \geq \frac { \rho ^ { 2 } } { 6 } , \qquad \Vert f _ { \tau } \Vert ^ { 2 } \geq \frac { \rho ^ { 2 } } { 5 } .
$$

To apply Lemma C.3 at every later step, we must check that the squared norm stays at least $\rho ^ { 2 } / 5$ . Each agent can use its parent’s prediction, so its fit has no larger error. Also, Lemma 2.1 gives $\| f _ { t } \| ^ { 2 } = \| Y \| ^ { 2 } - \mathrm { M S E } ( f _ { t } )$ for every agent. Consequently, for every $\tau \leq t \leq n$

$$
\| f _ { t } \| ^ { 2 } = \rho ^ { 2 } - \mathrm { M S E } ( f _ { t } ) \geq \rho ^ { 2 } - \mathrm { M S E } ( f _ { \tau } ) = \| f _ { \tau } \| ^ { 2 } \geq \frac { \rho ^ { 2 } } { 5 } .
$$

The lemma therefore applies at each of the $n - \tau$ remaining steps. Since its factor $1 - 1 0 \rho ^ { 2 } =$ $1 - 1 / ( 4 D )$ is positive, iterating gives

$$
\operatorname { M S E } ( f _ { n } ) \geq \operatorname { M S E } ( f _ { \tau } ) ( 1 - 1 0 \rho ^ { 2 } ) ^ { n - \tau } \geq \frac { \rho ^ { 2 } } { 6 } \left( 1 - \frac { 1 } { 4 D } \right) ^ { n - \tau } .
$$

Finally, Bernoulli’s inequality bounds the power below by $1 - ( n - \tau ) / ( 4 D )$ . Since $n - \tau \leq D$ , at least three quarters of the error at $A _ { \tau }$ remains. Hence

$$
\begin{array} { l } { \displaystyle \mathrm { M S E } ( f _ { n } ) \geq \frac { \rho ^ { 2 } } { 6 } \left( 1 - \frac { n - \tau } { 4 D } \right) } \\ { \displaystyle \geq \frac { \rho ^ { 2 } } { 6 } \cdot \frac { 3 } { 4 } = \frac { \rho ^ { 2 } } { 8 } = \frac { 1 } { 3 2 0 D } . } \end{array}
$$

This proves the bound for every path and therefore for every graph in the theorem.

## C.3 The upper bound

We prove the upper bound for general normalization constants using the greedy allocation in equation 6.

Theorem C.5. Let D satisfy Definition 2.3 at bounds $M _ { X } , A ^ { * } > 0$ . For every integer $D \geq 1$ , the designer can choose a feature allocation on the path $A _ { 1 } , \dotsc , A _ { D }$ such that

$$
\mathrm { M S E } ( f _ { D } ) - \mathrm { M S E } ( f ^ { * } ) \leq \frac { ( A ^ { * } M _ { X } ) ^ { 2 } } { D + 1 } .
$$

We first bound the decrease in error from one agent to the next, then prove Theorem C.5.

Lemma C.6. Under the assumptions of Theorem $C . 5 ,$ suppose at least one feature has positive norm. The allocation in equation $\delta ,$ starting from $f _ { 0 } = 0$ , satisfies, for every $0 \leq t < D$ ，

$$
\| f ^ { * } - f _ { t + 1 } \| ^ { 2 } \leq \| f ^ { * } - f _ { t } \| ^ { 2 } - \frac { \| f ^ { * } - f _ { t } \| ^ { 4 } } { ( A ^ { * } M _ { X } ) ^ { 2 } } .\tag{14}
$$

Proof. Fix $0 \leq t < D$ , and let $x _ { i }$ be the feature chosen by equation 6. By Lemma 2.2, the prediction $f _ { t }$ is the projection of $f ^ { * }$ onto $V _ { t } .$ , so $f ^ { * } - f _ { t }$ is orthogonal to $f _ { t }$ . This also holds for $f _ { 0 } = 0$ . Since $Y - f ^ { * }$ is orthogonal to every raw feature,

$$
\begin{array} { l } { \displaystyle | | f ^ { * } - f _ { t } | | ^ { 2 } = \langle f ^ { * } - f _ { t } , f ^ { * } \rangle = \displaystyle \sum _ { \ell = 1 } ^ { d } w _ { \ell } ^ { * } \langle Y - f _ { t } , x _ { \ell } \rangle } \\ { \displaystyle \qquad \leq \left( \displaystyle \sum _ { \ell = 1 } ^ { d } | w _ { \ell } ^ { * } | | x _ { \ell } | \right) \frac { | \langle Y - f _ { t } , x _ { i } \rangle | } { \| x _ { i } \| } } \\ { \displaystyle \qquad \leq A ^ { * } { M _ { X } } \frac { | \langle Y - f _ { t } , x _ { i } \rangle | } { \| x _ { i } \| } . } \end{array}
$$

The first inequality uses the maximizing choice of i in equation 6. The last inequality uses the coeficient and feature bounds.

The next agent can use $f _ { t } + \alpha x _ { i }$ for any $\alpha \in \mathbb { R }$ . Its MSE is

$$
\operatorname { M S E } ( f _ { t } + \alpha x _ { i } ) = \operatorname { M S E } ( f _ { t } ) - 2 \alpha \langle Y - f _ { t } , x _ { i } \rangle + \alpha ^ { 2 } \| x _ { i } \| ^ { 2 } .
$$

Choosing $\alpha = \langle Y - f _ { t } , x _ { i } \rangle / \| x _ { i } \| ^ { 2 }$ decreases MSE by $| \langle Y - f _ { t } , x _ { i } \rangle | ^ { 2 } / \| x _ { i } \| ^ { 2 }$ . The fitted prediction has no larger error, so

$$
\operatorname { M S E } ( f _ { t } ) - \operatorname { M S E } ( f _ { t + 1 } ) \geq { \frac { | \langle Y - f _ { t } , x _ { i } \rangle | ^ { 2 } } { \| x _ { i } \| ^ { 2 } } } \geq { \frac { \| f ^ { * } - f _ { t } \| ^ { 4 } } { ( A ^ { * } M _ { X } ) ^ { 2 } } } .
$$

By Lemma 2.1, the left-hand side equals $\| f ^ { * } - f _ { t } \| ^ { 2 } - \| f ^ { * } - f _ { t + 1 } \| ^ { 2 }$ , including when $t = 0$ . Rearranging proves equation 14. □

Proof of Theorem C.5. If all features have norm zero, every agent is exact. Otherwise, use the allocation in equation 6 and let $e _ { t } = \| f ^ { * } - f _ { t } \| ^ { 2 }$ for $0 \leq t \leq D$ . The initial error satisfies $e _ { 0 } =$ $\| f ^ { * } \| ^ { 2 } \leq ( A ^ { * } M _ { X } ) ^ { 2 }$ by the coeficient and feature bounds. If the error reaches zero, it stays zero because each later agent can use its parent’s prediction.

For positive errors, Lemma C.6 gives $e _ { t } - e _ { t + 1 } \geq e _ { t } ^ { 2 } / ( A ^ { * } M _ { X } ) ^ { 2 }$ . Dividing by $e _ { t } e _ { t + 1 }$ and using $e _ { t + 1 } \leq e _ { t }$ shows that $1 / e _ { t + 1 } - 1 / e _ { t } \geq 1 / ( A ^ { * } M _ { X } ) ^ { 2 }$ . Summing over the D agents gives

$$
\frac { 1 } { e _ { D } } \ge \frac { 1 } { e _ { 0 } } + \frac { D } { ( A ^ { * } M _ { X } ) ^ { 2 } } \ge \frac { D + 1 } { ( A ^ { * } M _ { X } ) ^ { 2 } } .
$$

Taking reciprocals gives the desired bound, since $e _ { D }$ is the output’s excess error by Lemma 2.1.

## C.4 The bounds for the unit case

Theorem 4.1. For every $d \geq 3$ and $D \geq 1$

$$
\frac { 1 } { 6 4 0 D } \leq R _ { 1 } ( d , D ) \leq \frac { 1 } { D + 1 } , \qquad \overline { { R } } _ { 1 } ( d , D ) \geq \frac { 1 } { 6 4 0 D } .
$$

In particular, $R _ { 1 } ( d , D ) = \Theta ( 1 / D )$

Proof. The upper bound follows from Theorem C.5 with $M _ { X } = A ^ { * } = 1$

For the lower bound, multiply the features in equation 5 by $2 { \sqrt { 2 } } / 3$ and the label by $1 / \sqrt { 2 }$ . The new global coeficients are $1 / 2 , - 1 / 4 , - 1 / 4$ , whose absolute values sum to one. The feature second moments are at most $( 8 / 9 ) ( 1 + 2 \rho ^ { 2 } ) \le 1 4 / 1 5 < 1$ , so the rescaled distribution belongs to $\mathcal { C } _ { 3 }$

Rescaling the features by a nonzero constant preserves their spans. Rescaling the label by $1 / \sqrt { 2 }$ divides every fitted prediction by ${ \sqrt { 2 } } .$ , by linearity of projection and induction along the path. Every excess error is therefore halved. Thus Theorem C.4 gives $R _ { 1 } ( 3 , D ) \ge 1 / ( 6 4 0 D )$ . Monotonicity in d from Proposition 2.4 extends this bound to all $d \geq 3$ . Finally, $\overline { { R } } _ { 1 } ( d , D ) \geq R _ { 1 } ( d , D )$ by the same proposition. □

## C.5 A fixed allocation at depth below the number of features

A path fixed in advance may omit a feature on which the label depends entirely. This prevents a uniform upper bound below one when the path is shorter than the number of features.

Proposition C.7. For all positive integers d, D with $D < d ,$ , we have $\overline { { R } } _ { 1 } ( d , D ) = 1$

Proof. Fix a graph, allocation, and output agent with at most one parent per agent and output depth at most $D .$ . The path ending at the output contains at most $D < d$ agents, so some feature $x _ { j }$ is not observed on that path. Choose independent standard Gaussian features and let $Y = x _ { j }$ This distribution belongs to $\mathcal { C } _ { d }$ , and the global predictor is $f ^ { * } = Y$

Each feature observed on the path is orthogonal to $Y .$ . Starting at the source, induction shows that every prediction on the path is zero. The output therefore has excess error $\| \boldsymbol { Y } \| ^ { 2 } = 1$ . This proves the lower bound for every fixed graph, allocation, and output agent. The upper bound of one follows from Proposition 2.4. □

## D Proofs for three parents per agent

## D.1 A fixed graph for exact aggregation

We first construct the gadget in Lemma D.4, which computes the fit from the earlier predictions together with one incoming prediction. We then use this gadget to prove Theorem 5.1. We use $f ^ { * }$ as the label throughout, as permitted by Lemma 2.2.

Fix $p _ { 0 } , \ldots , p _ { t } .$ q satisfying the hypotheses of Lemma D.4. The gadget must compute the fit from $x _ { 1 } , p _ { 0 } , \ldots , p _ { t } , q$ . We will do this by computing fits from shorter intervals of the list $p _ { 0 } , \ldots , p _ { t }$ and then joining them. For $0 \leq \ell \leq r \leq t .$ define

$$
W _ { \ell , r } = \mathrm { s p a n } \{ x _ { 1 } , p _ { t } , p _ { \ell } , \ldots , p _ { r } \} , \qquad g _ { \ell , r } = P _ { W _ { \ell , r } + \mathrm { s p a n } \{ q \} } f ^ { * } .
$$

Thus $g _ { \ell , r }$ is the fit from the predictions in the interval $[ \ell , r ]$ , together with $x _ { 1 } , p _ { t } , q .$ . Since $W _ { 0 , t } = V _ { t }$ the gadget’s output must be $g _ { 0 , t }$ . Every interval includes $p _ { t }$ , which is the fit from all of $V _ { t } .$ . In particular, $P _ { W _ { \ell , r } } f ^ { * } = p _ { t } { } .$ the vector $p _ { t }$ belongs to $W _ { \ell , r }$ , and its residual $f ^ { * } - p _ { t }$ is orthogonal to $W _ { \ell , r } \subseteq V _ { t }$

For $i \leq j$ , we have $p _ { i } \in V _ { i } \subseteq V _ { j }$ , while $f ^ { * } - p _ { j } \perp V _ { j }$ . This gives the first equality below, and Lemma 2.1 applied to $p _ { i } = P _ { V _ { i } } f ^ { * }$ gives the second:

$$
\langle p _ { i } , p _ { j } \rangle = \langle p _ { i } , f ^ { * } \rangle = \| p _ { i } \| ^ { 2 } , \qquad \| p _ { j } - p _ { i } \| ^ { 2 } = \| p _ { j } \| ^ { 2 } - \| p _ { i } \| ^ { 2 } .\tag{15}
$$

The squared-distance identity follows by expanding $\| p _ { j } - p _ { i } \| ^ { 2 }$ and substituting the inner-product identity.

We begin with intervals containing two consecutive predictions. The next lemma computes their fit using at most two agents, each with at most three parents.

Lemma D.1. Let $0 \leq \ell < r \leq t$ with $r = \ell + 1$ . The prediction $g _ { \ell , r }$ can be computed by the following agents, all observing x<sub>1</sub>. $I f \ell = 0$ , one agent with parents $p _ { t } , p _ { r } , q$ sufices. $I f r = t ,$ one agent with parents $p _ { t } , p _ { \ell } , q$ sufices. For $0 < \ell < r < t$ , first create an agent with parents $p _ { \ell } , p _ { r } , q$ and call its prediction u. An agent with parents $p _ { r } , p _ { t } , u$ then predicts $g _ { \ell , r }$

Proof. We first handle the two end intervals. If $\ell = 0$ , then $p _ { \ell } = p _ { 0 } \in \operatorname { s p a n } \{ x _ { 1 } \}$ , so

$$
W _ { 0 , r } + \operatorname { s p a n } \{ q \} = \operatorname { s p a n } \{ x _ { 1 } , p _ { t } , p _ { r } , q \} .
$$

An agent observing $x _ { 1 }$ with parents $p _ { t } , p _ { r } , q$ therefore predicts $g _ { 0 , r }$ . If $r = t$ , the corresponding space is span $\{ x _ { 1 } , p _ { t } , p _ { \ell } , q \}$ , which the other one-agent construction uses directly.

Now suppose $0 < \ell < r < t$ . The first agent observes $x _ { 1 }$ and receives $p _ { \ell } , p _ { r } , q$ . Write $K =$ span $\{ x _ { 1 } , p _ { \ell } , p _ { r } \}$ for its input space before adding $q .$ . The best prediction from $K$ , namely $P _ { K } f ^ { * }$ , is $p _ { r } \colon$ it is already the best prediction in the larger space $V _ { r }$ , and it belongs to K. After adding $q ,$ the first agent predicts $u = P _ { K + \mathrm { s p a n } \{ q \} } f ^ { * }$ . We will show why the second agent can use $p _ { r } , p _ { t }$ , u in place of the predictions spanning its required space $p _ { \ell } , p _ { r } , p _ { t } , q$ for $g _ { \ell , r }$

First consider the case $u = p _ { r }$ . Since the first agent receives $q ,$ its prediction $p _ { r }$ has no greater error than $q .$ The agent producing $q$ receives $p _ { t }$ by hypothesis, so $q$ has no greater error than $p _ { t }$ Finally, $p _ { t }$ has no greater error than $p _ { r }$ because it is the best prediction in $V _ { t }$ and $p _ { r } \in V _ { r } \subseteq V _ { t }$ These comparisons give

$$
\| f ^ { * } - p _ { r } \| ^ { 2 } \leq \| f ^ { * } - q \| ^ { 2 } \leq \| f ^ { * } - p _ { t } \| ^ { 2 } \leq \| f ^ { * } - p _ { r } \| ^ { 2 } .
$$

Hence all three errors are equal, and by Lemma 2.1, $u = p _ { r } = q = p _ { t }$ . Thus all three parents of the second agent predict $p _ { r }$ , and $g _ { \ell , r } = P _ { K } f ^ { * } = p _ { r }$ . The second agent retains this fit because its inputs lie in K and include $p _ { r }$

Now suppose $u \neq p _ { r }$ . Then we must have $u \notin K$ because $p _ { r }$ is the best prediction in $K$ . The first agent’s input space extends K by at most one dimension by adding $q ,$ and its prediction u must use that dimension because otherwise $u \in K$ . Consequently,

$$
K + \operatorname { s p a n } \{ u \} = K + \operatorname { s p a n } \{ q \} .
$$

Replacing $q$ with u therefore leaves the full input space for $g _ { \ell , \mathrm { i } }$ <sub>r</sub> unchanged:

$$
g _ { \ell , r } = P _ { K + \mathrm { s p a n } \{ p _ { t } , q \} } f ^ { * } = P _ { K + \mathrm { s p a n } \{ p _ { t } , u \} } f ^ { * } .
$$

It remains to explain why

$$
\begin{array} { r } { P _ { \mathrm { s p a n } \{ x _ { 1 } , p _ { \ell } , p _ { r } , p _ { t } , u \} } f ^ { * } = P _ { \mathrm { s p a n } \{ x _ { 1 } , p _ { r } , p _ { t } , u \} } f ^ { * } , } \end{array}
$$

where the left-hand side is the projection onto $K + \operatorname { s p a n } \{ p _ { t } , u \}$ and the right-hand side is the prediction of the second agent. So it remains to show that the extra $p \ell$ in the left-hand side does not change the projection.

We will show that the component of $g _ { \ell , \ d r }$ <sub>r</sub> in K is still $p _ { r }$ , which the second agent receives. Since $u = P _ { K + \mathrm { s p a n } \{ q \} } f ^ { * }$ and $p _ { r } = P _ { K } f ^ { * }$ , both residuals $f ^ { * } - u$ and $f ^ { * } - p _ { r }$ are orthogonal to $K$ . Their diference is $u ^ { \cdot } - p _ { r } , \mathrm { s o } \ u - p _ { r } \ \bot \ K$ . Also, $p _ { t } - p _ { r } \perp K$ , since it is the diference of the residuals $f ^ { * } - p _ { r }$ and $f ^ { * } - p _ { t }$ , both orthogonal to $K \subseteq V _ { r } \subseteq V _ { t }$ . Because $p _ { r } \in K$ , we can therefore write the full input space as the orthogonal sum

$$
K + \mathrm { s p a n } \{ p _ { t } , u \} = K + \mathrm { s p a n } \{ p _ { t } - p _ { r } , u - p _ { r } \} .
$$

The projection onto an orthogonal sum is the sum of the projections onto its two subspaces. Since the projection onto $K$ is $p _ { r }$ , this gives

$$
g _ { \ell , r } = p _ { r } + P _ { \mathrm { s p a n } \{ p _ { t } - p _ { r } , u - p _ { r } \} } f ^ { * } \in \mathrm { s p a n } \{ p _ { r } , p _ { t } , u \} .
$$

Thus the second agent can form $g _ { \ell , r }$ from its three parents. All of its inputs, including $x _ { 1 }$ , lie in $K + \operatorname { s p a n } \{ p _ { t } , u \}$ , where $g _ { \ell , r }$ already minimizes the error. It therefore also minimizes the error among the second agent’s available predictions, so the agent predicts $g _ { \ell , r }$ □

For a longer interval $[ \ell , r ]$ , we split at an index m and use the fits from $[ \ell , m ]$ and $[ m , r ]$ . The shared prediction $p _ { m }$ lets us separate the two interval spans into a common part and orthogonal remaining parts, as shown in the next lemma.

Lemma D.2. For $0 \leq \ell < m < r \leq t$ , set

$$
L = W _ { \ell , m } , \qquad R = W _ { m , r } , \qquad C = \mathrm { s p a n } \{ x _ { 1 } , p _ { t } , p _ { m } \} .
$$

Then

$$
P _ { L + R } = P _ { L } + P _ { R } - P _ { C } .\tag{16}
$$

Proof. Let $T = L \cap C ^ { \perp }$ be the subspace of vectors in L orthogonal to $C .$ . Since $C \subseteq L$ , we have the orthogonal decomposition $L = C + T$ . We will show that $T$ is also orthogonal to all of R. Because $C \subseteq R$ , this will give the orthogonal decomposition $L + R = R + T$ , and hence

$$
P _ { L } = P _ { C } + P _ { T } , \quad \quad P _ { L + R } = P _ { R } + P _ { T } . 
$$

Rewriting $P _ { T }$ in the second equality using $P _ { T } = P _ { L } - P _ { C }$ from the first equality gives the claimed identity. It remains to prove $T \perp R$

The nested projections imply that $p _ { j } - p _ { m } \perp V _ { m }$ for every $m \leq j \leq t .$ . Indeed, both residuals $f ^ { * } - p _ { m }$ and $f ^ { * } - p _ { j }$ are orthogonal to $V _ { m } \subseteq V _ { j }$ , and their diference is $p _ { j } - p _ { m }$ . In particular, we can write $L$ as the orthogonal sum

$$
L = \operatorname { s p a n } \{ x _ { 1 } , p _ { \ell } , \dots , p _ { m } \} + \operatorname { s p a n } \{ p _ { t } - p _ { m } \} .
$$

The first summand lies in $V _ { m }$ . The second is orthogonal to $V _ { m }$ and lies in $C ,$ , since $p _ { t } , p _ { m } \in C$ Every vector in $T$ is orthogonal to $C ,$ so its component in the second summand must be zero. Consequently, $T \subseteq V _ { m }$

Now take any $w \in T$ . Since $w \in V _ { m }$ , it is orthogonal to every diference $p _ { j } - p _ { m }$ for $m \le j \le t$ It is also orthogonal to $p _ { m } \in C$ . Therefore, for every $m \le j \le t$

$$
\langle w , p _ { j } \rangle = \langle w , p _ { m } \rangle + \langle w , p _ { j } - p _ { m } \rangle = 0 .
$$

This includes the predictions $p _ { m } , \ldots , p _ { r } , p _ { t }$ that generate R together with $x _ { 1 }$ . Since $x _ { 1 } \in C$ , we also have w $\perp x _ { 1 }$ . Thus $w \perp R ,$ proving $T \perp R$ and completing the proof. □

We now turn this identity between spaces into a way to combine their fits. The next agent will receive the fits from both intervals and one more fit from their shared inputs $x _ { 1 } , p _ { t } , p _ { m } , q$

Lemma D.3. For $0 \leq \ell < m < r \leq t$ , an agent observing $x _ { 1 }$ and receiving $g _ { \ell , m } , \ g _ { m , r }$ , and $g _ { m , m }$ predicts $g _ { \ell , r }$

Proof. We will show that $g _ { \ell , r } ~ \in ~ \operatorname { s p a n } \{ g _ { \ell , m } , g _ { m , r } , g _ { m , m } \}$ . This sufices because all of the agent’s inputs lie in $W _ { \ell , r } + \operatorname { s p a n } \{ q \}$ , where $g _ { \ell , r }$ minimizes the error.

First suppose $\langle f ^ { * } - p _ { t } , q \rangle = 0$ . The residual $f ^ { * } - p _ { t }$ is orthogonal to $V _ { t }$ and, in this case, to $q .$ It is therefore orthogonal to all four spaces $W _ { \ell , r } , W _ { m , r } , W _ { \ell , m } , W _ { m , m }$ , even after adding $q .$ Thus $p _ { t }$ remains the best prediction in each case, giving

$$
g _ { \ell , m } = g _ { m , r } = g _ { m , m } = g _ { \ell , r } = p _ { t } .
$$

All three parents already supply $g _ { \ell , r } ,$ which proves the claim in this case.

Suppose $\langle f ^ { * } - p _ { t } , q \rangle \neq 0$ . Since $f ^ { * } - p _ { t } \perp V _ { t }$ , the nonzero inner product with $q$ implies $q \notin V _ { t }$ For each of the four intervals [i, j], set $s _ { i , j } \ = \ \lVert q - P _ { W _ { i , j } } q \rVert ^ { 2 }$ , the squared norm of the part of $q$ outside its interval space. Since $q \notin V _ { t }$ and $W _ { i , j } \subseteq V _ { t }$ , each $s _ { i , j }$ is positive. We will show that:

$$
g _ { \ell , r } = \frac { s _ { \ell , m } g _ { \ell , m } + s _ { m , r } g _ { m , r } - s _ { m , m } g _ { m , m } } { s _ { \ell , r } } .\tag{17}
$$

By showing the above, we conclude that the agent can compute $g _ { \ell , r }$ from its three parents and concludes the proof.

We first compute $g _ { i , j }$ , by separating its input space into two orthogonal subspaces. Since $P _ { W _ { i , j } } q$ already belongs to $W _ { i , j }$ , subtracting it from $q$ leaves the input span unchanged:

$$
W _ { i , j } + \mathrm { s p a n } \{ q \} = W _ { i , j } + \mathrm { s p a n } \{ q - P _ { W _ { i , j } } q \} .
$$

The two subspaces on the right are orthogonal, because the projection residual $q - P _ { W _ { i , j } } q$ is orthogonal to $W _ { i , j }$ . We can therefore compute $g _ { i , j }$ by adding the projections of $f ^ { * }$ onto these two subspaces.

The projection onto $W _ { i , j }$ is $p _ { t } \colon$ we have $p _ { t } \in W _ { i , j } \subseteq V _ { t }$ , and $f ^ { * } - p _ { t } \perp V _ { t }$ . For the other subspace, we use the projection onto the line spanned by $q - P _ { W _ { i , j } } q$ , whose squared norm is $s _ { i , j } > 0$ . This gives

$$
g _ { i , j } = p _ { t } + \frac { \langle f ^ { * } , q - P _ { W _ { i , j } } q \rangle } { s _ { i , j } } ( q - P _ { W _ { i , j } } q ) .
$$

We now simplify the numerator. Since $p _ { t } \in W _ { i , j }$ , it is orthogonal to $q - P _ { W _ { i , j } } q .$ . Also, $f ^ { * } - p _ { t }$ is orthogonal to $P _ { W _ { i , j } } q \in W _ { i , j } \subseteq V _ { t }$ . These two facts give, respectively,

$$
\langle f ^ { * } , q - P _ { W _ { i , j } } q \rangle = \langle f ^ { * } - p _ { t } , q - P _ { W _ { i , j } } q \rangle = \langle f ^ { * } - p _ { t } , q \rangle .
$$

Substituting this numerator and multiplying by $s _ { i , j }$ yields

$$
s _ { i , j } g _ { i , j } = s _ { i , j } p _ { t } + \langle f ^ { * } - p _ { t } , q \rangle ( q - P _ { W _ { i , j } } q ) .\tag{18}
$$

We now combine the three weighted parent predictions using this formula. Applying Lemma D.2 to $q$ gives $P _ { W _ { \ell , r } } q = P _ { W _ { \ell , m } } q + P _ { W _ { m , r } } q - P _ { W _ { m , m } } q$ . Subtracting both sides from q and grouping the terms gives

$$
q - P _ { W _ { \ell , r } } q = ( q - P _ { W _ { \ell , m } } q ) + ( q - P _ { W _ { m , r } } q ) - ( q - P _ { W _ { m , m } } q ) .\tag{19}
$$

This equality also gives a relation between the weights. For each $[ i , j ]$ , we can split $q$ into its projection $P _ { W _ { i , j } } q$ and its residual $q - P _ { W _ { i , j } } q$ . These two vectors are orthogonal, so

$$
\begin{array} { c } { { \langle q , q - P _ { W _ { i , j } } q \rangle = \langle P _ { W _ { i , j } } q , q - P _ { W _ { i , j } } q \rangle + \| q - P _ { W _ { i , j } } q \| ^ { 2 } } } \\ { { = s _ { i , j } . } } \end{array}
$$

Taking inner products with $q$ in equation 19 for $q - P _ { W _ { \ell , r } } q$ therefore yields

$$
s _ { \ell , r } = s _ { \ell , m } + s _ { m , r } - s _ { m , m } .\tag{20}
$$

Now consider $s _ { \ell , m } g _ { \ell , m } + s _ { m , r } g _ { m , r } - s _ { m , m } g _ { m , m }$ and rewrite it using equation 18 and then equation 20:

$$
\begin{array} { r l r } {  { s _ { \ell , m } g _ { \ell , m } + s _ { m , r } g _ { m , r } - s _ { m , m } g _ { m , m } } } \\ & { } & { = \big ( s _ { \ell , m } + s _ { m , r } - s _ { m , m } \big ) p _ { t } } \\ & { } & { \quad + \big \langle f ^ { * } - p _ { t } , q \big \rangle \big ( ( q - P _ { W _ { \ell , m } } q ) + ( q - P _ { W _ { m , r } } q ) - ( q - P _ { W _ { m , m } } q ) \big ) } \\ & { } & { = s _ { \ell , r } p _ { t } + \big \langle f ^ { * } - p _ { t } , q \big \rangle ( q - P _ { W _ { \ell , r } } q ) } \\ & { } & { = s _ { \ell , r } g _ { \ell , r } , } \end{array}
$$

where the last equality uses equation 18 backwards. Since $s _ { \ell , r } > 0$ , dividing by it gives equation 17.

We can now assemble the gadget. Each interval will compute its fit from the two shorter intervals and the shared inputs, stopping at the pairs covered by Lemma D.1.

Lemma D.4. Let $V _ { 0 } \subseteq \cdots \subseteq V _ { t }$ be nested spaces with $V _ { 0 } = \operatorname { s p a n } \{ x _ { 1 } \}$ . Suppose agents predict $p _ { i } = P _ { V _ { i } } f ^ { * }$ , with $V _ { i } = \operatorname { s p a n } \{ x _ { 1 } , p _ { 0 } , \dots , p _ { i } \}$ for every $0 \leq i \leq t$ . Let $q$ be the prediction of an agent observing $x _ { 1 }$ and receiving $p _ { t }$ . A gadget whose graph and allocation depend only on t computes

$$
P _ { V _ { t } + \operatorname { s p a n } \{ q \} } f ^ { * } .
$$

Every added agent observes $x _ { 1 }$ and has at most three parents. For $t \geq 2$ , the gadget adds at most $4 ( t - 1 )$ agents and has $O ( \log t )$ additional depth. For $t = 0 \ o r t = 1 , q$ already equals this prediction, so no agents are added.

Proof. For $t = 0 \mathrm { ~ o r ~ } t = 1$ , the hypothesis on $V _ { t }$ and the fact that $p _ { 0 } ~ \in ~ \operatorname { s p a n } \{ x _ { 1 } \}$ give $V _ { t } ~ =$ $\operatorname { s p a n } \{ x _ { 1 } , p _ { t } \}$ . The agent predicting $q$ observes $x _ { 1 }$ and receives $p _ { t } .$ , so its residual $f ^ { * } - q$ is orthogonal to $x _ { 1 }$ and $p _ { t }$ . The residual is also orthogonal to $q ,$ because $q$ belongs to that agent’s input span. Therefore $q$ belongs to $V _ { t } + \operatorname { s p a n } \{ q \}$ and its residual is orthogonal to this entire space. This proves $q = P _ { V _ { t } + \operatorname { s p a n } \{ q \} } f ^ { * }$ for these two values of t.

Suppose $t \geq 2$ . We recursively construct an agent predicting $g _ { \ell , r }$ for each interval used in the construction. If $r = \ell + 1$ , use the agents in Lemma D.1. For $r - \ell \geq 2$ , set $m = \lfloor ( \ell + r ) / 2 \rfloor$ and construct $g _ { \ell , m }$ and $g _ { m , r }$ in parallel. Add an agent observing $x _ { 1 }$ with parents $p _ { t } , p _ { m } , q .$ It predicts $P _ { \mathrm { s p a n } \{ x _ { 1 } , p _ { t } , p _ { m } , q \} } f ^ { * } = g _ { m , m }$ , the third parent prediction required by Lemma D.3. An agent observing $x _ { 1 }$ and receiving this prediction and the two interval fits then predicts $g _ { \ell , r }$ . Induction on $r - \ell$ proves that the root computes $g _ { 0 , t } = P _ { V _ { t } + \mathrm { s p a n } \{ q \} } f ^ { * }$

The recursion has t leaves, one for each pair $[ i , i + 1 ]$ , and $t - 1$ internal nodes, since every internal node has two children. By Lemma D.1, the two end pairs use one agent each and the other $t - 2$ pairs use two each. Every internal node uses two agents: one to predict $g _ { m , m }$ and one to combine the three predictions using Lemma D.3. Thus the number of added agents is

$$
2 + 2 ( t - 2 ) + 2 ( t - 1 ) = 4 ( t - 1 ) .
$$

The leaves need at most two layers. At every internal node, the agent with parents $p _ { t } , p _ { m } , q$ can be computed in parallel with the child intervals, so merging the three predictions adds one more layer.

Splitting each interval at its midpoint gives at most $\left\lceil \log _ { 2 } t \right\rceil$ levels of merges. The added depth is therefore at most $2 + \lceil \log _ { 2 } t \rceil = O ( \log t )$

All added agents observe $x _ { 1 }$ and have at most three parents. The interval splits, the parent choices, and the choice between the one-agent and two-agent constructions do not depend on the distribution. Hence the same gadget works for every distribution with finite second moments.

We finish by showing that the fixed graph reaches exact aggregation after the prescribed $d - 1$ rounds. The key point is that every round before exact aggregation increases the dimension of the space retained by the predictions.

Theorem 5.1. For every $d \geq 2 ,$ , the designer can fix a graph, a single-feature allocation, and an output agent that achieve exact aggregation for every distribution with finite second moments, using at most $4 d ^ { 2 }$ agents, at most three parents per agent, and depth $O ( d \log d )$ . For $d = 1$ , one agent sufices.

Proof. For $d = 1$ , the global feature space is span $\{ x _ { 1 } \}$ , so a source observing $x _ { 1 }$ already predicts $f ^ { * }$ . Suppose d $\geq 2$ , and use the source and the $d - 1$ rounds described in Section 5.1. Initially,

$$
V _ { 0 } = \mathrm { s p a n } \{ x _ { 1 } \} = \mathrm { s p a n } \{ x _ { 1 } , p _ { 0 } \} , \qquad p _ { 0 } = P _ { V _ { 0 } } f ^ { * } .
$$

Assume that after t rounds, for every $0 \leq i \leq t ,$ the available predictions satisfy $p _ { i } = P _ { V _ { i } } f ^ { * }$ and $V _ { i } = \operatorname { s p a n } \{ x _ { 1 } , p _ { 0 } , \dots , p _ { i } \}$ , with the spaces nested. We show that the next round preserves these properties.

First suppose $p _ { t } \neq f ^ { * }$ . The residual $f ^ { * } - p _ { t }$ is a nonzero vector in H and is orthogonal to $V _ { t } .$ Since the raw features span $H .$ , some feature has a nonzero inner product with $f ^ { * } - p _ { t }$ . Otherwise this residual would be orthogonal to all of $H _ { ; }$ , including itself. This feature cannot be $x _ { 1 } \in V _ { t }$ so call it $x _ { i }$ with $i \geq 2$ . The agent observing $x _ { i }$ and receiving $p _ { t }$ can use any predictor $p _ { t } + \alpha x _ { i }$ Choosing $\alpha = \langle f ^ { * } - p _ { t } , x _ { i } \rangle / \| x _ { i } \| ^ { 2 }$ gives error

$$
\| f ^ { * } - p _ { t } - \alpha x _ { i } \| ^ { 2 } = \| f ^ { * } - p _ { t } \| ^ { 2 } - \frac { | \langle f ^ { * } - p _ { t } , x _ { i } \rangle | ^ { 2 } } { \| x _ { i } \| ^ { 2 } } < \| f ^ { * } - p _ { t } \| ^ { 2 } .
$$

Its prediction has a smaller or equal error, so at least one of the agents testing a raw feature improves strictly on $p _ { t }$ . Each agent in the binary tree can use either child’s prediction, so the error cannot increase on the path from that improving agent to the root. Hence the root prediction $q _ { t }$ also improves strictly on $p _ { t }$

The root observes $x _ { 1 }$ and receives $p _ { t }$ . Thus Lemma D.4 applies and computes

$$
V _ { t + 1 } = V _ { t } + \mathrm { s p a n } \{ q _ { t } \} , \qquad p _ { t + 1 } = P _ { V _ { t + 1 } } f ^ { * } .
$$

The induction hypothesis gives $V _ { t + 1 } = \operatorname { s p a n } \{ x _ { 1 } , p _ { 0 } , \dots , p _ { t } , q _ { t } \}$ , the first equality in equation $^ { 7 . }$ Since $p _ { t }$ is the best prediction in $V _ { t }$ and $q _ { t }$ has smaller error, $q _ { t } \notin V _ { t }$ , so $V _ { t + 1 }$ extends $V _ { t }$ by one dimension. The fit $p _ { t + 1 }$ has no larger error than $q _ { t }$ , which also forces $p _ { t + 1 } \notin V _ { t }$ . Consequently, $V _ { t } + \operatorname { s p a n } \{ p _ { t + 1 } \}$ is a subspace of $V _ { t + 1 }$ with the same dimension. These spaces are equal, proving the second equality in equation $7$ and completing the induction in this case.

If $p _ { t } = f ^ { * }$ , every agent testing a raw feature has $f ^ { * }$ among its inputs and therefore predicts it with zero error. The same holds throughout the binary tree, so $q _ { t } = f ^ { * }$ . The gadget then returns $p _ { t + 1 } = f ^ { * }$ and $V _ { t + 1 } = V _ { t }$ , which preserves the induction hypothesis in this case as well.

All predictions lie in $H$ , so the nested spaces also lie in H. If $x _ { 1 } \neq 0$ , then dim $V _ { 0 } = 1$ and dim $H \leq d .$ Thus there can be at most $d - 1$ dimension increases before the retained space equals

$H ,$ , at which point its fit is $f ^ { * }$ . If $x _ { 1 } = 0$ , then dim $V _ { 0 } = 0$ and dim $H \leq d - 1$ , giving the same bound. Since every round before exact aggregation increases the dimension, and every later round retains $f ^ { * }$ , the prescribed $d - 1$ rounds end with $p _ { d - 1 } = f ^ { * }$

Each improvement tree uses $d - 1$ agents to test the features $x _ { 2 } , \ldots , x _ { d }$ and $d - 1$ internal agents to combine their predictions with $p _ { t }$ . The gadget adds no agents for $t \leq 1$ and at most $4 ( t - 1 )$ otherwise, by Lemma D.4. Including the source and all $d - 1$ rounds gives at most

$$
1 + 2 ( d - 1 ) ^ { 2 } + 4 \sum _ { t = 2 } ^ { d - 2 } ( t - 1 ) \leq 4 d ^ { 2 }
$$

agents, taking the sum to be zero when $d \leq 3$ . The agents testing the features add one layer, and the balanced binary tree adds $O ( \log d )$ layers. The gadget adds $O ( \log t )$ layers for $t \geq 2$ and none otherwise. Since $t < d ,$ every round adds $O ( \log d )$ depth, giving total depth $O ( d \log d )$ □

## D.2 Selecting features and updating predictions

We now prove Theorem 5.2. For $f ^ { * } \neq 0$ , choose linearly independent features $x _ { 1 } , \ldots , x _ { r }$ spanning H. For some set $J _ { t } \subseteq [ r ]$ of t features which we choose later, define

$$
H _ { t } = \operatorname { s p a n } \{ x _ { j } : j \in J _ { t } \} , \qquad p _ { t } = P _ { H _ { t } } f ^ { * } , \qquad q _ { t , i } = P _ { H _ { t } + \operatorname { s p a n } \{ x _ { i } \} } f ^ { * } \quad ( i \notin J _ { t } ) .
$$

The prediction $p _ { t }$ uses the selected features, and $q _ { t , i }$ is the prediction obtained when $x _ { i }$ is also available. We start with $J _ { 0 } = \emptyset , H _ { 0 } = \{ 0 \}$ , and $p _ { 0 } = 0 . \mathrm { ~ H ~ } p _ { t } = f ^ { * }$ , we stop. Otherwise, select an index $j \notin J _ { t }$ with the smallest nonzero value of $\| q _ { t , j } - p _ { t } \| ^ { 2 }$ , and set $J _ { t + 1 } = J _ { t } \cup \{ j \}$ and $p _ { t + 1 } = q _ { t , j }$ For $t \geq 1$ , the agent predicting $q _ { t , j }$ will already exist, so this selection requires no new agent. The remaining task is to produce $q _ { t + 1 , i }$ for every $i \notin J _ { t + 1 }$

For $t \geq 1$ , each new agent will observe $x _ { i }$ and receive $p _ { t } , p _ { t + 1 } , q _ { t , i }$ . We handle the first update separately in the theorem proof below. If $q _ { t , i } = p _ { t }$ , that parent supplies no additional vector, so we need to express $q _ { t + 1 , i }$ using $x _ { i } , p _ { t } , p _ { t + 1 }$ . To make this update exact, we maintain the condition

$$
q _ { t , i } = p _ { t } \quad \Longrightarrow \quad x _ { i } \perp H _ { t } \qquad ( i \notin J _ { t } ) .\tag{21}
$$

The condition holds initially because $H _ { 0 } = \{ 0 \}$ . The next lemma shows that our selection rule preserves it and puts each required prediction in the new agent’s input span. Its proof also explains why we choose the smallest nonzero improvement.

Lemma D.5. Suppose $p _ { t } \neq f ^ { * }$ and equation 21 holds. Then $q _ { t , i } \neq p _ { t }$ for some $i \notin J _ { t }$ . Choose $j \notin J _ { t }$ minimizing $\| q _ { t , j } - p _ { t } \| ^ { 2 }$ among its nonzero values, and set $J _ { t + 1 } = J _ { t } \cup \{ j \}$ and $p _ { t + 1 } = q _ { t , j }$ For every $i \notin J _ { t + 1 }$

$$
q _ { t + 1 , i } \in \left\{ \begin{array} { l l } { \mathrm { s p a n } \{ p _ { t } , p _ { t + 1 } , q _ { t , i } \} , } & { q _ { t , i } \neq p _ { t } , } \\ { \mathrm { s p a n } \{ p _ { t } , p _ { t + 1 } , x _ { i } \} , } & { q _ { t , i } = p _ { t } . } \end{array} \right.
$$

Moreover, equation 21 holds with $t + 1$ in place of t.

Proof. We first show that a selection is possible. The residual $f ^ { * } - p _ { t }$ is a nonzero vector in $H ,$ , so it has a nonzero inner product with some feature among $x _ { 1 } , \ldots , x _ { r }$ . Otherwise the residual would be orthogonal to all of $H$ , including itself. That feature must be unselected, since the residual is orthogonal to $H _ { t }$ . Call its index i. Since $f ^ { * } - q _ { t , i } \perp x _ { i }$ but $\langle f ^ { * } - p _ { t } , x _ { i } \rangle \neq 0$ , the predictions $q _ { t , i }$ and $p _ { t }$ must difer and hence $\| q _ { t , i } - p _ { t } \| ^ { 2 } > 0$

To compare the predictions before and after selecting $j ,$ , separate each remaining feature into its part in $H _ { t }$ and its part orthogonal to $H _ { t }$ . For every $i \notin J _ { t } ,$ define

$$
u _ { i } = x _ { i } - P _ { H _ { t } } x _ { i } .
$$

The vectors $u _ { i }$ are linearly independent. Indeed, any nontrivial linear relation among them would express a nontrivial combination of the unselected features as a combination of the selected features, contradicting the independence of $x _ { 1 } , \ldots , x _ { r }$

We next prove the two span claims. Since $x _ { i } - u _ { i } = P _ { H _ { t } } x _ { i } \in H _ { t }$ , we have $H _ { t } + \operatorname { s p a n } \{ x _ { i } \} =$ $H _ { t } + \operatorname { s p a n } \{ u _ { i } \}$ . The latter sum is orthogonal, and the projection of $q _ { t , i }$ <sub>i</sub> onto $H _ { t }$ is $p _ { t } .$ , so

$$
q _ { t , i } - p _ { t } = P _ { \mathrm { s p a n } \{ u _ { i } \} } f ^ { * } .
$$

Whenever $q _ { t , i } \neq p _ { t }$ , the diference $q _ { t , i } - p _ { t }$ is therefore a nonzero multiple of $u _ { i }$ . In particular, the selected prediction satisfies $p _ { t + 1 } - p _ { t } = q _ { t , j } - p _ { t } \neq 0$

Now fix $i \notin J _ { t + 1 }$ . Since $H _ { t + 1 } = H _ { t } + \operatorname { s p a n } \{ x _ { j } \}$ and both $x _ { j } - u _ { j }$ and $x _ { i } - u _ { i }$ lie in $H _ { t }$ , the space defining $q _ { t + 1 , i }$ can be written as

$$
H _ { t + 1 } + \operatorname { s p a n } \{ x _ { i } \} = H _ { t } + \operatorname { s p a n } \{ u _ { j } , u _ { i } \} .
$$

Both $u _ { j }$ and $u _ { i }$ are orthogonal to $H _ { t } .$ , so projection onto this space gives

$$
q _ { t + 1 , i } = p _ { t } + P _ { \mathrm { s p a n } \{ u _ { j } , u _ { i } \} } f ^ { * } .\tag{22}
$$

If $q _ { t , i } \neq p _ { t }$ , the two diferences $p _ { t + 1 } - p _ { t }$ and $q _ { t , i } - p _ { t }$ span the same space as $u _ { j } , u _ { i }$ . By equation 22, adding $p _ { t }$ to a linear combination of these diferences gives $q _ { t + 1 , i } .$ , so

$$
q _ { t + 1 , i } \in \operatorname { s p a n } \{ p _ { t } , p _ { t + 1 } , q _ { t , i } \} .
$$

If $q _ { t , i } = p _ { t }$ , the condition equation 21 gives $x _ { i } \perp H _ { t } , \mathrm { s o } u _ { i } = x _ { i }$ . We can then use $p _ { t + 1 } - p _ { t }$ and $x _ { i }$ to span $u _ { j } , u _ { i }$ . The same formula gives

$$
q _ { t + 1 , i } \in \mathrm { s p a n } \{ p _ { t } , p _ { t + 1 } , x _ { i } \} ,
$$

which proves the other span claim.

It remains to prove equation 21 after the selection. First take $i \notin J _ { t + 1 }$ with $q _ { t , i } \neq p _ { t }$ . We will show that $q _ { t + 1 , i } \neq p _ { t + 1 }$ , using the smallest nonzero improvement rule. Since $p _ { t }$ belongs to the input spaces of both $q _ { t , i }$ and $p _ { t + 1 }$ , Lemma 2.1 and the choice of $j$ give

$$
\begin{array} { r l } & { \Vert f ^ { * } - q _ { t , i } \Vert ^ { 2 } = \Vert f ^ { * } - p _ { t } \Vert ^ { 2 } - \Vert q _ { t , i } - p _ { t } \Vert ^ { 2 } } \\ & { \qquad \leq \Vert f ^ { * } - p _ { t } \Vert ^ { 2 } - \Vert p _ { t + 1 } - p _ { t } \Vert ^ { 2 } } \\ & { \qquad = \Vert f ^ { * } - p _ { t + 1 } \Vert ^ { 2 } . } \end{array}
$$

Thus $q _ { t , i }$ has no greater error than $p _ { t + 1 }$ . These predictions are distinct: their diferences from $p _ { t }$ are nonzero multiples of the independent vectors $u _ { i } , u _ { j }$ . Also, $q _ { t , i }$ belongs to $H _ { t + 1 } + \operatorname { s p a n } \{ x _ { i } \}$ , since it belongs to $H _ { t } + \operatorname { s p a n } \{ x _ { i } \}$ and $H _ { t } \subseteq H _ { t + 1 }$ . The best prediction in a subspace is unique, so $p _ { t + 1 }$ cannot be optimal in this space when a distinct available prediction has no greater error. Therefore its best prediction $q _ { t + 1 , i }$ difers from $p _ { t + 1 }$ , as claimed.

Now suppose $q _ { t + 1 , i } = p _ { t + 1 }$ . We just proved above that $q _ { t , i } \neq p _ { t }$ implies $q _ { t + 1 , i } \neq p _ { t + 1 } , \mathrm { s o } \ q _ { t , i } = p _ { t }$ here. Both $q _ { t , i }$ and $q _ { t + 1 , i }$ are projections onto spaces containing $x _ { i }$ , so

$$
\left. f ^ { * } - p _ { t } , x _ { i } \right. = 0 , \qquad \left. f ^ { * } - p _ { t + 1 } , x _ { i } \right. = 0 .
$$

Subtracting gives $\langle p _ { t + 1 } - p _ { t } , x _ { i } \rangle = 0$ . Since $p _ { t + 1 } - p _ { t }$ is a nonzero multiple of $u _ { j }$ , this implies $x _ { i } ~ \perp ~ u _ { j }$ . The equality $q _ { t , i } ~ = ~ p _ { t }$ and equation 21 also give $x _ { i } ~ \perp ~ H _ { t }$ . Hence $x _ { i } ~ \perp ~ H _ { t + 1 }$ , since $H _ { t + 1 } = H _ { t } + \mathrm { s p a n } \{ u _ { j } \}$ . This proves the condition at the next step. □

We now use Lemma D.5 to construct the agents and prove the bounds. Only the first selected feature needs a source. All other predictions will be obtained from this source and the agents added after each selection.

Theorem 5.2. For every distribution with finite second moments and feature rank $r \geq 1$ , the designer can choose a graph, a single-feature allocation, and an output agent achieving exact aggregation with at most three parents per agent, depth at most $r _ { ; }$ , and at most $1 + { \bigl ( } { \bigl . } { \bigr ) }$ agents. In particular, $R _ { b } ( d , D ) = 0$ for $b \geq 3$ and $D \geq d$

Proof. If $f ^ { * } = 0$ , every agent predicts zero by Lemma 2.2, so one source sufices. Suppose $f ^ { * } \neq 0$ We follow the selection rule above and stop as soon as a selected prediction equals $f ^ { * }$ , using its agent as the output.

Since $H _ { 0 } = \{ 0 \}$ , equation 21 holds initially. The designer computes the values $q _ { 0 , i }$ to make the first selection, which exists by Lemma D.5, and creates one source observing the selected feature $x _ { j }$ . Its prediction is $p _ { 1 } = q _ { 0 , j }$

Unless we have stopped, create each $q _ { 1 , i }$ using an agent observing $x _ { i }$ and receiving only $p _ { 1 }$ Since $p _ { 1 }$ is a nonzero multiple of $x _ { j }$ , its input space is span $\{ x _ { i } , p _ { 1 } \} = H _ { 1 } + \mathrm { s p a n } \{ x _ { i } \}$ , so it predicts $q _ { 1 , i }$ . By Lemma D.5 at $t = 0$ , equation 21 holds after this first selection.

After each subsequent selection $t \geq 2$ , unless we have stopped, for every $i \notin J _ { t }$ , add an agent observing $x _ { i }$ and receiving $p _ { t - 1 } , p _ { t } , q _ { t - 1 , i }$ . By Lemma D.5, applied at step $t - 1$ , the inputs span $q _ { t , i }$ and equation 21 is preserved. All inputs lie in $H _ { t } + \operatorname { s p a n } \{ x _ { i } \}$ , where $q _ { t , i }$ is the best prediction, so the agent predicts $q _ { t , i }$ . Each selection after the first reuses the agent predicting $q _ { t , j }$ as the agent predicting $p _ { t + 1 }$

Each update adds at most one layer, while selecting $p _ { t + 1 } = q _ { t , j }$ adds no agent. Starting with $p _ { 1 }$ at depth one, induction gives $p _ { t }$ by depth t and $q _ { t , i }$ by depth $t + 1$

Every selection adds one of the r basis features. If the construction has not stopped earlier, after $r$ selections we have $H _ { r } = H$ and $p _ { r } = f ^ { * }$ , so the output has depth at most r. The construction uses one source and at most $r - t$ new agents after selection $t ,$ for $1 \leq t < r$ . Its size is therefore at most $\begin{array} { r } { 1 + \sum _ { t = 1 } ^ { r - 1 } ( r - t ) = 1 + \binom { r } { 2 } } \end{array}$ □

## D.3 The feature order required along a path

The following proof refines the propagation argument underlying Kearns, Roth, and Ryu [KRR26, Theorem 5.9]. Their theorem gives a depth barrier for the same Gaussian construction without normalization. We track the feature order along paths and include the normalization in the error bound.

Proposition D.6. For the distribution in equation 9, fix any graph, single-feature allocation, and output agent $A _ { G }$ . Let ℓ be the largest prefix length such that some directed path ending at $A _ { G }$ contains agents observing $x _ { 1 } , \ldots , x _ { \ell }$ in this order, possibly with other agents between them. If $\ell < d$ , then

$$
\mathrm { M S E } ( f _ { G } ) - \mathrm { M S E } ( f ^ { * } ) \geq { \frac { 1 } { 2 d ^ { 2 } ( \ell + 1 ) } } .
$$

Proof. For each agent $A _ { v }$ , let $k ( A _ { v } )$ be the length of the longest prefix of the feature order $x _ { 1 } , \ldots , x _ { d }$ appearing along a directed path ending at $A _ { v }$ . We first prove by induction in a topological order that its prediction lies in span $\{ x _ { 1 } , \ldots , x _ { k ( A _ { v } ) } \}$

Let L be the maximum prefix length among the parents of $A _ { v }$ , taking $L = 0$ if it has no parents. By induction, all parent predictions lie in the span of the first L features. If $L = d ,$ the induction claim follows because every prediction lies in H. Suppose $L < d .$ A raw feature $x _ { j }$ with $j > L + 1$ involves only Gaussian variables $Z _ { L + 1 } , \ldots , Z _ { d - 1 }$ . It is therefore orthogonal to both $Y$ and the first L feature span. Adding it to the parents’ inputs does not change the projection of Y . A feature with $j \leq L$ is already in that span. Finally, observing $x _ { L + 1 }$ extends a path containing the first $L$ features in order to one containing the first $L + 1$ in order, and the new prediction lies in the first $L + 1$ feature span. These cases conclude the induction.

At the output, $k ( A _ { G } ) = \ell ,$ so the induction gives $f _ { G } \in \operatorname { s p a n } \{ x _ { 1 } , . . . , x _ { \ell } \}$ . It remains to bound the error of predictions in this space when $\ell < d$ . We do this by computing $P _ { \mathrm { s p a n } \{ x _ { 1 } , \ldots , x _ { \ell } \} } Y$ which is the best prediction in the space span $\{ x _ { 1 } , \ldots , x _ { \ell } \}$ for $Y$ , and thus it is not a worse prediction than $f _ { G }$ because $f _ { G }$ belongs to the same space.

The variables $Z _ { 0 } , \ldots , Z _ { \ell }$ are orthonormal in $L ^ { 2 } ( \mathcal { D } )$ , since they are independent standard Gaussians. Within their span, a vector is orthogonal to $x _ { i } = ( Z _ { i - 1 } - Z _ { i } ) / \sqrt { 2 }$ exactly when its coeficients on $Z _ { i - 1 }$ and $Z _ { i }$ are equal. Orthogonality to all of $x _ { 1 } , \ldots , x _ { \ell }$ therefore requires all $\ell + 1$ coeficients to be equal. Thus the vectors in this Gaussian span orthogonal to the first ℓ features form the line spanned by

$$
w = Z _ { 0 } + \cdot \cdot \cdot + Z _ { \ell } .
$$

The label $Y = Z _ { 0 } / ( \sqrt { 2 } d )$ also lies in span $\{ Z _ { 0 } , \ldots , Z _ { \ell } \}$ . Its residual after projection onto the first ℓ features is therefore its projection onto the line spanned by w. Orthonormality gives $\| w \| ^ { 2 } = \ell + 1$ and $\langle Y , w \rangle = 1 / ( \sqrt { 2 } d )$ , so

$$
Y - P _ { \mathrm { s p a n } \{ x _ { 1 } , \ldots , x _ { \ell } \} } Y = { \frac { \langle Y , w \rangle } { \| w \| ^ { 2 } } } w = { \frac { w } { { \sqrt { 2 } } d ( \ell + 1 ) } } .
$$

Since $P _ { \mathrm { s p a n } \{ x _ { 1 } , \ldots , x _ { \ell } \} } Y$ minimizes the error over span $\{ x _ { 1 } , \ldots , x _ { \ell } \}$ and $f _ { G }$ belongs to this space, we obtain

$$
\operatorname { M S E } ( f _ { G } ) \geq \| Y - P _ { \mathrm { s p a n } \{ x _ { 1 } , \ldots , x _ { \ell } \} } Y \| ^ { 2 } = { \frac { \| w \| ^ { 2 } } { 2 d ^ { 2 } ( \ell + 1 ) ^ { 2 } } } = { \frac { 1 } { 2 d ^ { 2 } ( \ell + 1 ) } } .
$$

Finally, $f ^ { * } = Y$ has zero error, so this is also the claimed lower bound on the excess error. □

Theorem 5.3. In the adaptive designer setting, for every $d \geq 1$ , some normalized rank-d distribution requires depth $D \geq d$ for exact aggregation, even without a parent limit. In the oblivious designer setting, every fixed parent limit $b \geq 2$ requires $D = \Omega ( d \log d )$ for exact aggregation.

Proof. For the adaptive designer setting, use the normalized distribution in equation 9. It has rank d because the coeficient matrix of the features in $Z _ { 0 } , \dots , Z _ { d - 1 }$ is triangular with nonzero diagonal. By Proposition D.6, exact aggregation requires a path containing all d features, hence at least d agents. For $D < d .$ , every path in a depth-D network has at most D agents, so Proposition D.6 gives excess error at least $1 / ( 2 d ^ { 2 } ( D + 1 ) )$ . This gives $R _ { b } ( d , D ) > 0$ for every $b \geq 1$

For the oblivious designer setting, fix $b \geq 2$ and a graph, single-feature allocation, and output agent that achieve exact aggregation for every distribution on d features, with at most b parents per agent. Let D be the output depth.

An agent at depth s has at most $b ^ { s - 1 }$ paths from sources to it. A source has one such path. At any other agent, their number is the sum of the path counts at its at most b parents, each of depth at most $s - 1$ . This proves the bound by induction.

There are therefore at most $b ^ { D - 1 }$ source-to-output paths. Each has at most D agents, so choosing d positions on it gives at most $\textstyle { \binom { D } { d } }$ possible feature orders. Every path ending at the output can be extended backward to a source. Applying Proposition D.6 after each relabeling of the features requires these source-to-output paths to contain every permutation of $[ d ]$ . Counting the d! permutations gives

$$
b ^ { D - 1 } { \binom { D } { d } } \geq d ! .\tag{23}
$$

Finally, ${ \binom { D } { d } } \leq 2 ^ { D }$ , so equation 23 implies $( 2 b ) ^ { D } \geq d !$ . Taking logarithms and using $\log ( d ! ) \geq$ d log $d - d$ gives the stated order bound. □

## E Proofs for two parents per agent

## E.1 Combining three predictions exactly

For points $a , b$ in a Euclidean plane, write $F ( \boldsymbol { a } , \boldsymbol { b } )$ for the point closest to zero on the line through $a , b ,$ with $F ( a , a ) = a$ . Minimizing the squared norm of $a + \lambda ( b - a )$ over $\lambda \in \mathbb { R }$ gives

$$
F ( a , b ) = a - { \frac { \langle a , b - a \rangle } { \| b - a \| ^ { 2 } } } ( b - a ) \qquad ( a \neq b ) .\tag{24}
$$

Thus, when $a \neq b , F ( a , b )$ is the unique point on the line through $a , b$ that is orthogonal to its direction $b - a$

To turn pairwise fits into operations on points in a plane, we rescale each prediction so that its component along the desired prediction g is exactly g. Subtracting g then leaves a point orthogona to it. The next lemma shows that a pairwise fit becomes an application of $F ,$ and the desired prediction becomes zero.

Lemma E.1. Let $V \subseteq L ^ { 2 } ( { \mathcal { D } } )$ have dimension three, and let $g \in V$ be nonzero. For every nonzero $f \in V$ satisfying $\langle g , f \rangle = \| f \| ^ { 2 }$ , define

$$
n ( f ) = { \frac { \| g \| ^ { 2 } } { \| f \| ^ { 2 } } } f - g .\tag{25}
$$

Then $n ( f ) \in V \cap \operatorname { s p a n } \{ g \} ^ { \perp }$ and

$$
f = \frac { \| g \| ^ { 2 } } { \| g \| ^ { 2 } + \| n ( f ) \| ^ { 2 } } \big ( g + n ( f ) \big ) .\tag{26}
$$

In particular, $n ( f ) = 0$ exactly when $\begin{array} { l } { \displaystyle { f \ = \ g } } \end{array}$ . For any two such vectors $f _ { 1 } , f _ { 2 }$ , their $\mathit { f i t } ~ q ~ =$ $P _ { \mathrm { s p a n } \{ f _ { 1 } , f _ { 2 } \} } g$ is nonzero and satisfies

$$
n ( q ) = F \bigl ( n ( f _ { 1 } ) , n ( f _ { 2 } ) \bigr ) .
$$

Proof. The vector $n ( f )$ belongs to V , and the hypothesis on f gives

$$
\langle g , n ( f ) \rangle = \frac { \| g \| ^ { 2 } } { \| f \| ^ { 2 } } \langle g , f \rangle - \| g \| ^ { 2 } = 0 .
$$

Taking squared norms in $g + n ( f ) = \| g \| ^ { 2 } f / \| f \| ^ { 2 }$ therefore gives $\| g \| ^ { 2 } + \| n ( f ) \| ^ { 2 } = \| g \| ^ { 4 } / \| f \| ^ { 2 }$ Substitution into the same equality proves equation 26. This formula gives $f = g$ when $n ( f ) = 0$ and the definition gives $n ( g ) = 0$

For the pairwise fit, set $a = n ( f _ { 1 } )$ and $b = n ( f _ { 2 } )$ . If $a = b$ , equation 26 gives $f _ { 1 } = f _ { 2 }$ . The hypothesis $\langle g , f _ { 1 } \rangle = \| f _ { 1 } \| ^ { 2 }$ then gives $q = f _ { 1 } , \mathrm { s o } n ( q ) = a = F ( a , a )$

Suppose $a \neq b ,$ , and let $c = F ( a , b )$ . The line through $a , b$ also passes through $^ { c , }$ so

$$
\mathrm { s p a n } \{ f _ { 1 } , f _ { 2 } \} = \mathrm { s p a n } \{ g + a , g + b \} = \mathrm { s p a n } \{ g + c \} + \mathrm { s p a n } \{ b - a \} .
$$

For the last equality, $g + c$ is a linear combination of $g + a , g + b$ , and each of $g + a , g + b$ difers from $g + c$ by a multiple of $b - a$ . The two spaces on the right are orthogonal: $g \perp b - a$ because

$a , b \in \operatorname { s p a n } \{ g \} ^ { \perp }$ , and $c \perp b - a$ by the definition of $F .$ . Since $g$ is orthogonal to the second space, its projection onto the input span is

$$
q = P _ { \mathrm { s p a n } \{ g + c \} } g = { \frac { \| g \| ^ { 2 } } { \| g \| ^ { 2 } + \| c \| ^ { 2 } } } ( g + c ) .
$$

This vector is nonzero and has squared norm $\| g \| ^ { 4 } / ( \| g \| ^ { 2 } + \| c \| ^ { 2 } )$ . Substituting into the definition of $n ( q )$ gives $n ( q ) = c = F ( a , b )$ □

The space $V \cap \operatorname { s p a n } \{ g \} ^ { \perp }$ in Lemma E.1 has dimension two because dim $V = 3$ and $g \neq 0$ . An orthonormal basis identifies its two coordinates with the real and imaginary parts of a complex number in $\mathbb { C } ,$ , with $\langle z , w \rangle = \operatorname { R e } ( z { \overline { { w } } } )$ and norm $| z |$ . Multiplication by a nonzero complex number rotates and scales both lines and distances to zero, so

$$
F ( \eta z , \eta w ) = \eta F ( z , w ) .\tag{27}
$$

The identity also holds for $\eta = 0$

We continue in the complex plane, starting from the three points associated with the input predictions $f _ { 1 } , f _ { 2 } , f _ { 3 }$ . Each new point must be obtained by applying F to two available points. Our goal is to produce zero, which corresponds to the desired prediction $g .$

We organize the construction by keeping triples obtained from the initial triple by a common rotation and scaling. For a triple $T = ( a , b , c )$ and a complex number η, we write $\eta T = ( \eta a , \eta b , \eta c )$ The following lemma shows how to combine two sequences that produce such triples.

Lemma E.2. Let $T = ( a , b , c )$ be a triple of nonzero complex points, and write $z T = ( z a , z b , z c )$ Suppose fixed sequences of F operations starting from T produce αT and $\beta T$ . Running the first sequence on the output of the second produces $\alpha \beta T$ , using the sum of their numbers of operations. Alternatively, the three operations

$$
F ( \alpha a , \beta a ) , F ( \alpha b , \beta b ) , F ( \alpha c , \beta c )
$$

produce $F ( \alpha , \beta ) T$ , using that sum plus three operations.

Proof. By equation 27, multiplying all three inputs of a fixed sequence by $\beta$ multiplies every intermediate point and every output by $\beta .$ This follows by induction over its $F$ operations. Thus the sequence that produces $\alpha T$ from $T$ produces $\alpha \beta T$ when run on $\beta T .$

For the second construction, run both sequences from $T .$ . Applying equation 27 to the three stated operations gives the triple

$$
( a F ( \alpha , \beta ) , \ b F ( \alpha , \beta ) , \ c F ( \alpha , \beta ) ) = F ( \alpha , \beta ) T .
$$

The first construction uses the operations of both sequences. The second uses those operations and three more. □

Using the above definition, we can work with complex numbers directly rather than triples. Namely, we can represent a complex number z by a corresponding triple $z T .$ . The next lemma shows that we can also multiply and divide complex numbers, through the operations on triples. More specifically, to support division, we will represent each complex number z by a numerator and denominator triple NT and $D T$ , with $z = N / D$ . The next lemma shows that we can multiply and divide complex numbers, through the operations on triples.

Lemma E.3. Let $T = ( a , b , c )$ be a triple of nonzero complex points, and suppose fixed sequences of F operations starting from T produce αT and βT. Every expression z formed from $1 , \alpha , \beta$ using multiplication, division by nonzero values, and F has two fixed sequences of F operations producing

$$
N T = ( N a , N b , N c ) , \qquad D T = ( D a , D b , D c ) , \qquad D \neq 0 , \qquad z = N / D .
$$

$I f z = 0$ , the constructed triple NT consists of three zero points.

Proof. For the value 1, use $T$ as both the numerator and denominator triple. For $\alpha ,$ use αT as the numerator and T as the denominator, and do the same for $\beta .$ . We extend these choices through the expression using the following three rules.

To multiply two values $N _ { 1 } / D _ { 1 }$ and $N _ { 2 } / D _ { 2 }$ , construct the triples $( N _ { 1 } N _ { 2 } ) T$ and $( D _ { 1 } D _ { 2 } ) T$ by composing sequences as in Lemma E.2. They are the numerator and denominator triples for the product because

$$
\frac { N _ { 1 } } { D _ { 1 } } \frac { N _ { 2 } } { D _ { 2 } } = \frac { N _ { 1 } N _ { 2 } } { D _ { 1 } D _ { 2 } } .
$$

The new denominator is nonzero since $D _ { 1 } , D _ { 2 } \neq 0$

To take the reciprocal of a nonzero value $N / D ,$ exchange the two triples and their sequences. The numerator becomes $D T$ and the denominator becomes $N T$ , giving the ratio $D / N$ . The new denominator is nonzero because $N / D \neq 0$ . Division by a nonzero value is multiplication by its reciprocal, so it uses only these two rules.

To apply F to two values, equation 27 gives

$$
F \left( { \frac { N _ { 1 } } { D _ { 1 } } } , { \frac { N _ { 2 } } { D _ { 2 } } } \right) = { \frac { F ( N _ { 1 } D _ { 2 } , N _ { 2 } D _ { 1 } ) } { D _ { 1 } D _ { 2 } } } .\tag{28}
$$

First construct $( N _ { 1 } D _ { 2 } ) T$ and $( N _ { 2 } D _ { 1 } ) T$ by the multiplication rule. The three F operations in Lemma E.2 then produce $F ( N _ { 1 } D _ { 2 } , N _ { 2 } D _ { 1 } ) T$ , the numerator triple. Another use of the multiplication rule produces the denominator triple $( D _ { 1 } D _ { 2 } ) T$ , whose coeficient is nonzero.

These rules give the required sequences for every expression. Finally, $N / D = 0$ with $D \neq 0$ implies $N = 0$ , so each entry of the numerator triple is zero. Thus a zero ratio gives an actual zero point without performing division. □

Next we construct the first two triples, $T$ and $\mu T$ , from the three given points. These two triples are our only building blocks for the rest of the construction.

Lemma E.4. Let ${ \cal T } _ { 0 } ~ = ~ ( a _ { 0 } , b _ { 0 } , c _ { 0 } )$ be three noncollinear points in $\mathbb { C } ,$ , ordered so that $\left| a _ { 0 } \right| \ \geq$ max $\{ | b _ { 0 } | , | c _ { 0 } | \}$ . Starting from $T _ { 0 } ,$ , either zero is available after at most five F operations, or two operations give a noncollinear triple $T = ( a , b , c )$ of nonzero points from which a fixed sequence of three F operations produces $\mu T = ( \mu a , \mu b , \mu c )$ for some µ satisfying

$$
F ( 1 , \mu ) = \mu , \qquad \mu \not \in { \mathbb R } , \qquad 0 < | \mu | < 1 .
$$

Proof. Start from the ordered triple ${ \cal T } _ { 0 } = ( a _ { 0 } , b _ { 0 } , c _ { 0 } )$ , and stop if any initial or newly produced point is zero. Set $a = a _ { 0 }$ and form $T = ( a , b , c )$ using

$$
b = F ( a , b _ { 0 } ) , \qquad c = F ( b , c _ { 0 } ) .
$$

Then perform the three operations

$$
a ^ { \prime } = F ( a , c ) , \qquad b ^ { \prime } = F ( b , a ^ { \prime } ) , \qquad c ^ { \prime } = F ( c , b ^ { \prime } ) .\tag{29}
$$

We will show that the first two operations give a noncollinear $T$ and the last three produce $\mu T$ with $\mu = a ^ { \prime } / a$ satisfying the stated properties.

The ordering of $T _ { 0 }$ ensures $b \neq a$ . Otherwise, $F ( a , b _ { 0 } ) = a$ would give $a \perp b _ { 0 } - a , { \mathrm { ~ s o ~ } } | b _ { 0 } | ^ { 2 } =$ $| a | ^ { 2 } + | b _ { 0 } - a | ^ { 2 } > | a | ^ { 2 }$ , contrary to the choice of $a _ { 0 }$ . Since b lies on the line through $a , b _ { 0 }$ and difers from $^ { a , }$ the points $a , b , c _ { 0 }$ remain noncollinear.

We also have $c \neq b . \ \operatorname { I f } c = b .$ , the definition of F would give $b \perp c _ { 0 } - b$ . We already have $b \perp a - b$ from the first operation. These two directions are independent because $a , b , c _ { 0 }$ are noncollinear, so b would be zero, in which case we would have stopped. Since c lies on the line through $b , c _ { 0 }$ and difers from $b ,$ the triple $T = ( a , b , c )$ is noncollinear. The perpendicular relations $b \perp a - b$ and $c \perp b - c$ also give

$$
F ( a , b ) = b , \qquad F ( b , c ) = c , \qquad 0 < | c | < | b | < | a | ,\tag{30}
$$

where the strict norm inequalities follow from $\mathrm { P y }$ thagoras and $b \neq a , c \neq b .$

Now consider $\boldsymbol { a ^ { \prime } } = \boldsymbol { F } ( \boldsymbol { a } , \boldsymbol { c } )$ and $\mu = a ^ { \prime } / a$ . The relation $a ^ { \prime } \perp a - a ^ { \prime }$ when divided by a gives $F ( 1 , \mu ) = \mu$ , hence $\langle 1 - \mu , \mu \rangle = \operatorname { R e } \mu - | \mu | ^ { 2 } = 0$ , and thus Re $\begin{array} { r } { { \bf \ddot { \rho } } _ { : \mu } = | \mu | ^ { 2 } } \end{array}$ . Since $a ^ { \prime }$ is the closest point to zero on a line containing $c ,$ we have $\left| a ^ { \prime } \right| \leq \left| c \right|$ . Thus

$$
0 < | \mu | = \frac { | a ^ { \prime } | } { | a | } \leq \frac { | c | } { | a | } < 1 .
$$

A real number satisfying Re $\mu = | \mu | ^ { 2 }$ must be zero or one, so $\mu$ is nonreal.

It remains to show that $b ^ { \prime } = \mu b$ and $c ^ { \prime } = \mu c$ . We use the following identity: if distinct z, $w \in \mathbb { C }$ satisfy $F ( 1 , z ) = z$ and $F ( 1 , w ) = w$ , then

$$
F ( z , w ) = z w .\tag{31}
$$

Indeed, $F ( 1 , z ) = z$ gives Re $z = | z | ^ { 2 }$ , including when $z = 1$ , and likewise for $w$ . If $z w = 0$ , one input is zero and the identity follows. Otherwise,

$$
\begin{array} { r } { \langle z w , z \rangle = | z | ^ { 2 } \mathrm { R e } w = | z w | ^ { 2 } , \qquad \langle z w , w \rangle = | w | ^ { 2 } \mathrm { R e } z = | z w | ^ { 2 } . } \end{array}
$$

Thus $z , w$ lie on the line through zw perpendicular to zw. Since they are distinct, this is their line, and its closest point to zero is zw.

To apply this identity to $b ^ { \prime }$ , we have $F ( 1 , b / a ) = b / a$ by equation 30, and $F ( 1 , \mu ) = \mu$ as proved above. These two numbers are distinct: $a ^ { \prime }$ lies on the line through $a , c ,$ whereas b does not because $T$ is noncollinear. Consequently,

$$
b ^ { \prime } = a F ( b / a , a ^ { \prime } / a ) = a ( b / a ) ( a ^ { \prime } / a ) = \mu b .
$$

For $c ^ { \prime } ,$ the two numbers $c / b$ and $b ^ { \prime } / b = \mu$ also satisfy the hypotheses of equation 31. The first satisfies $F ( 1 , c / b ) = c / b$ by equation 30, and they are distinct because

$$
| b ^ { \prime } | = { \frac { | b | | a ^ { \prime } | } { | a | } } \leq { \frac { | b | | c | } { | a | } } < | c | .
$$

Applying the identity gives $c ^ { \prime } = b F ( c / b , b ^ { \prime } / b ) = b ( c / b ) ( b ^ { \prime } / b ) = \mu c .$ . Hence $( a ^ { \prime } , b ^ { \prime } , c ^ { \prime } ) = \mu T$ . We used two operations to obtain $T$ and the fixed three operations in equation 29 to obtain $\mu T .$ □

If zero has not already appeared, the pairs $( T , T )$ and $( \mu T , T )$ now represent the starting complex numbers 1 and $\mu .$ . By Lemma E.3, it remains to find an expression formed from these two values that equals zero.

To obtain zero, we will use the following lemma. We will later show how to construct the inputs $u , v ,$ , and m needed to apply this lemma.

Lemma E.5. Let $u , v \in \mathbb { C }$ be distinct points with Re $u = \mathrm { R e } v = 1$ , and let $m = ( u + v ) / 2$ be their midpoint. Then $q = F ( u v , m ^ { 2 } )$ is purely imaginary, and $F ( 1 , q ^ { 2 } ) = 0$

Proof. The products uv and $m ^ { 2 }$ lie on a horizontal line. Indeed, the midpoint identity gives

$$
u v - m ^ { 2 } = - \frac { ( u - v ) ^ { 2 } } { 4 } > 0 ,
$$

because $u - v$ is nonzero and purely imaginary. Thus $u v , m ^ { 2 }$ are distinct and have the same imaginary part. Their line has its closest point to zero on the imaginary axis, so $q$ is purely imaginary. Hence $q ^ { 2 }$ is real and nonpositive, and the line through $1 , q ^ { 2 }$ contains zero. □

We use the below identity to construct the inputs u, v and m needed to apply Lemma E.5 and produce a zero.

Lemma E.6. For every nonreal $z \in \mathbb { C }$ with Re $z = 1$ , the value $F ( 1 , z ^ { 2 } )$ is nonzero and

$$
{ \frac { 1 } { F ( 1 , z ^ { 2 } ) } } = { \frac { 1 + { \overline { { z } } } } { 2 } } .\tag{32}
$$

Proof. We compute $F ( 1 , z ^ { 2 } )$ directly and show that it equals $2 / ( 1 + { \overline { { z } } } )$ . By equation 24,

$$
\begin{array} { r l r } {  { F ( 1 , z ^ { 2 } ) = 1 - \frac { \langle 1 , z ^ { 2 } - 1 \rangle } { | z ^ { 2 } - 1 | ^ { 2 } } ( z ^ { 2 } - 1 ) } } \\ & { } & { = 1 - \frac { \operatorname { R e } ( z ^ { 2 } - 1 ) } { | z ^ { 2 } - 1 | ^ { 2 } } ( z ^ { 2 } - 1 ) . } \end{array}
$$

Here $\langle 1 , z ^ { 2 } - 1 \rangle = \operatorname { R e } \left( { \overline { { z ^ { 2 } - 1 } } } \right) = \operatorname { R e } ( z ^ { 2 } - 1 )$ , since conjugation leaves the real part unchanged.

We simplify the numerator first. Because Re $z = 1$ and z is nonreal, $z - 1$ is nonzero and purely imaginary and its square is $- | z - 1 | ^ { 2 }$ . Hence

$$
\operatorname { R e } ( z ^ { 2 } - 1 ) = \operatorname { R e } \left( ( z - 1 ) ^ { 2 } + 2 ( z - 1 ) \right) = - | z - 1 | ^ { 2 } .
$$

For the denominator, factor $z ^ { 2 } - 1 = ( z - 1 ) ( z + 1 )$

$$
| z ^ { 2 } - 1 | ^ { 2 } = | ( z - 1 ) ( z + 1 ) | ^ { 2 } = | z - 1 | ^ { 2 } | z + 1 | ^ { 2 } .
$$

Substituting into the formula for $F ( 1 , z ^ { 2 } )$ now gives

$$
\begin{array} { c } { { F ( 1 , z ^ { 2 } ) = 1 - \displaystyle \frac { - | z - 1 | ^ { 2 } } { | z - 1 | ^ { 2 } | z + 1 | ^ { 2 } } ( z ^ { 2 } - 1 ) } } \\ { { = 1 + \displaystyle \frac { z ^ { 2 } - 1 } { | z + 1 | ^ { 2 } } . } } \end{array}
$$

Here we cancel $| z - 1 | ^ { 2 }$ , which is nonzero because $z \neq 1$

To simplify the remaining fraction, use $| z + 1 | ^ { 2 } = ( z + 1 ) ( 1 + \overline { { z } } ) \mathrm { ~ a n d ~ } z ^ { 2 } - 1 = ( z + 1 ) ( z - 1 )$ Bringing the two terms to a common denominator gives

$$
{ \begin{array} { r l } & { 1 + { \frac { z ^ { 2 } - 1 } { | z + 1 | ^ { 2 } } } = { \frac { ( z + 1 ) ( 1 + { \overline { { z } } } ) + ( z + 1 ) ( z - 1 ) } { ( z + 1 ) ( 1 + { \overline { { z } } } ) } } } \\ & { \qquad = { \frac { ( z + 1 ) ( z + { \overline { { z } } } ) } { ( z + 1 ) ( 1 + { \overline { { z } } } ) } } } \\ & { \qquad = { \frac { z + { \overline { { z } } } } { 1 + { \overline { { z } } } } } = { \frac { 2 } { 1 + { \overline { { z } } } } } . } \end{array} }
$$

We can cancel z + 1 because its real part is two and hence it is nonzero, and the last equality uses $z + { \overline { { z } } } = 2 \operatorname { R e } z = 2$ . The final fraction is defined and nonzero because $\operatorname { R e } ( 1 + { \overline { { z } } } ) = 2$ and hence it is nonzero. Taking its reciprocal proves the identity. □

We are now ready to combine the above lemmas to produce zero from any three noncollinear points in a plane.

Lemma E.7. Starting from three noncollinear points in $\mathbb { C } ,$ , applying F to available pairs produces zero using at most 300 operations.

Proof. Use Lemma E.4 to obtain a triple $T = ( a , b , c )$ of nonzero points and a sequence of three F operations producing $\mu T .$ , stopping if zero appears. We use Lemma E.3 with $\alpha = \mu$ and $\beta = 1$ to construct separate numerator and denominator triples for the expressions

$$
u = \frac { 1 } { \mu } , \qquad v = \frac { 1 } { F ( 1 , u ^ { 2 } ) } , \qquad m = \frac { 1 } { F ( 1 , v ^ { 2 } ) } , \qquad q = F ( u v , m ^ { 2 } ) .\tag{33}
$$

For $u = 1 / \mu$ we use the numerator triple T and denominator triple $\mu T$ . For $v ,$ equation 28 gives $v = \mu ^ { 2 } / F ( \mu ^ { 2 } , 1 )$ , so we construct the numerator triple $\mu { } ^ { 2 } T$ and denominator triple $F ( \mu ^ { 2 } , 1 ) T$ using Lemma E.2. We generate other values in a similar manner as Lemma E.3 allows us to do. We first check that all denominators are nonzero and m is the midpoint of $u , v$

To apply Lemma E.6 to $u ,$ we need Re $u = 1$ and u nonreal. Since $\mu \neq 0$ , we have $u = 1 / \mu =$ $\overline { { \mu } } / \vert \mu \vert ^ { 2 }$ . Using Re $\mu = | \mu | ^ { 2 }$ and the fact that $\mu$ is nonreal gives

$$
\mathrm { R e } u = { \frac { \mathrm { R e } \mu } { | \mu | ^ { 2 } } } = 1 , \qquad \mathrm { I m } u = - { \frac { \mathrm { I m } \mu } { | \mu | ^ { 2 } } } \neq 0 .
$$

Thus we have that $F ( 1 , u ^ { 2 } ) \neq 0$ and by applying Lemma E.6, we get $v = ( 1 + \overline { { u } } ) / 2$ . In turn, Re $v = ( 1 + \mathrm { R e } u ) / 2 = 1$ and Im $v = - \operatorname { I m } u / 2 \neq 0$ . The imaginary parts of u, v have opposite signs, so u ̸= v.

We can therefore apply Lemma E.6 to v as well. It gives $F ( 1 , v ^ { 2 } ) \neq 0$ and

$$
m = { \frac { 1 + { \overline { { v } } } } { 2 } } = { \frac { 3 + u } { 4 } } = { \frac { 2 u + 1 + { \overline { { u } } } } { 4 } } = { \frac { u + v } { 2 } } .
$$

Here we substitute $v = ( 1 + \overline { { u } } ) / 2$ and use $u + \overline { { u } } = 2$ , which follows from Re $u = 1$ . This verifies that all reciprocals in equation 33 are defined and that m is the midpoint of the distinct numbers $u , v$ , both with real part one. Applying Lemma E.5 now shows that $q$ is purely imaginary and $F ( 1 , q ^ { 2 } ) = 0$

Applying Lemma E.3 to the expression $F ( 1 , q ^ { 2 } ) = 0$ therefore produces a numerator triple of zero points.

For the size bound, we keep every intermediate triple and count only new operations. We give the counts in Table 2 and explain each step below.

Table 2: Operation counts for the construction.
<table><tr><td>Step</td><td>New F operations (at most)</td></tr><tr><td>Prepare  $T$ </td><td>2</td></tr><tr><td>Prepare  $\mu T$ </td><td>3</td></tr><tr><td>Represent u</td><td>0</td></tr><tr><td>Represent v</td><td>6</td></tr><tr><td>Represent m</td><td>18</td></tr><tr><td>Represent uv</td><td>0</td></tr><tr><td>Represent  $m ^ { 2 }$ </td><td>45</td></tr><tr><td>Represent  $q$ </td><td>27</td></tr><tr><td>Produce the numerator triple for  $F ( 1 , q ^ { 2 } )$ </td><td>162</td></tr><tr><td>Total</td><td>263</td></tr></table>

Preparing T uses the two operations in Lemma E.4. The same lemma gives a sequence of three operations producing $\mu T$ . Together with the available triple $T ,$ this represents $u = 1 / \mu$ . By Lemma E.2, running this sequence on any available triple ηT produces $\mu \eta T$ in three new operations. We will use this sequence throughout the count.

To represent $v ,$ first run the sequence on $\mu T$ to obtain $\mu { } ^ { 2 } T$ , using three new operations. Define $D _ { v } = F ( \mu ^ { 2 } , 1 )$ , so $v = \mu ^ { 2 } / D _ { v }$ by equation 28. The three $F$ operations in Lemma E.2, applied to the triples $\mu ^ { 2 } T$ and $T _ { i }$ , produce $D _ { v } T$ . Thus representing v costs $3 + 3 = 6$ new operations. Starting from $T$ alone, the complete sequence producing $D _ { v } T$ uses $3 + 3 + 3 = 9$ operations.

For $m ,$ the same quotient rule gives

$$
m = \frac { 1 } { F ( 1 , \mu ^ { 4 } / D _ { v } ^ { 2 } ) } = \frac { D _ { v } ^ { 2 } } { F ( D _ { v } ^ { 2 } , \mu ^ { 4 } ) } .
$$

Define $D _ { m } = F ( D _ { v } ^ { 2 } , \mu ^ { 4 } )$ . To obtain the numerator triple $D _ { v } ^ { 2 } T$ , run the nine-operation sequence for $D _ { v } T$ on the stored triple $D _ { v } T$ . This also produces $\mu D _ { v } T$ along the way, which we keep. Next, starting from the stored triple $\mu ^ { 2 } T$ , run the three-operation sequence twice to obtain $\mu ^ { 4 } T$ , using six new operations. Three more F operations on $D _ { v } ^ { 2 } T$ and $\mu ^ { 4 } T$ produce the denominator triple $D _ { m } T$ . The new cost for m is therefore $9 + 6 + 3 = 1 8$ . Including the nine operations used before this step, we have a complete sequence of 27 operations producing $D _ { m } T$ from $T .$

The product $u v = \mu ^ { 2 } / ( \mu D _ { v } )$ needs no new operations. Its numerator triple $\mu { } ^ { 2 } T$ was produced for $v ,$ and its denominator triple $\mu D _ { v } T$ was kept while constructing m.

To represent $m ^ { 2 } = D _ { v } ^ { 4 } / D _ { m } ^ { 2 }$ , start from $D _ { v } ^ { 2 } T$ and run the nine-operation sequence twice, obtaining $D _ { v } ^ { 3 } T$ and then $D _ { v } ^ { 4 } T$ . This costs 18 new operations. Running the complete sequence for $D _ { m } T$ on its stored output $D _ { m } T$ gives $D _ { m } ^ { 2 } T$ in 27 more operations. Thus this step costs $1 8 + 2 7 = 4 5$ new operations.

For $q = F ( u v , m ^ { 2 } )$ , clearing denominators by equation 28 gives

$$
q = \frac { F ( \mu ^ { 2 } D _ { m } ^ { 2 } , \mu D _ { v } ^ { 5 } ) } { \mu D _ { v } D _ { m } ^ { 2 } } .
$$

We first construct its numerator triple. Starting from $D _ { m } ^ { 2 } T$ , two uses of the three-operation sequence give $\mu ^ { 2 } D _ { m } ^ { 2 } T$ , at a cost of six operations. Starting from $D _ { v } ^ { 4 } T$ , the nine-operation sequence gives $D _ { v } ^ { 5 } T$ , and three further operations give $\mu D _ { v } ^ { 5 } T$ . Three $F$ operations on these two triples then produce the numerator triple for q. Its new cost is $6 + ( 9 + 3 ) + 3 = 2 1$

For the denominator of q, we reuse the triples $\mu ^ { 2 } D _ { m } ^ { 2 } T$ and $D _ { m } ^ { 2 } T$ . Three F operations on them produce $D _ { v } D _ { m } ^ { 2 } T$ , since $F ( \mu ^ { 2 } D _ { m } ^ { 2 } , D _ { m } ^ { 2 } ) = D _ { m } ^ { 2 } F ( \mu ^ { 2 } , 1 ) \stackrel { \cdots } { = } D _ { m } ^ { 2 } D _ { v }$ by equation 27. The three-operation sequence then gives $\mu D _ { v } D _ { m } ^ { 2 } T$ . Thus the denominator costs six new operations, and the total new cost for $q$ is $2 1 + 6 = 2 7$

Finally, to represent $F ( 1 , q ^ { 2 } )$ , we need to run the sequences for the numerator and denominator of $q$ on their own output triples. We therefore count the length of each complete sequence starting from $T .$ . The numerator sequence consists of the steps for $u , v , m , m ^ { 2 }$ , followed by the 21 numerator operations for $q ,$ so its length is at most

$$
3 + 6 + 1 8 + 4 5 + 2 1 = 9 3 .
$$

For the denominator sequence, first produce $D _ { m } T$ in $2 7$ operations and then $D _ { m } ^ { 2 } T$ in $2 7$ more. Apply the nine-operation sequence to obtain $D _ { v } D _ { m } ^ { 2 } T$ , followed by three operations to obtain $\mu D _ { v } D _ { m } ^ { 2 } T$ This sequence has length $2 7 + 2 7 + 9 + 3 = 6 6$

Running each of these sequences on its own output squares its coeficient by Lemma E.2, so the two resulting triples represent $q ^ { 2 }$ . This costs at most $9 3 + 6 6$ new operations. Three $F$ operations on those triples give the numerator triple for $F ( 1 , q ^ { 2 } )$ by equation 28. That triple is zero, as proved above. The final step therefore costs at most $9 3 + 6 6 + 3 = 1 6 2$ operations. Adding the entries of Table 2 gives $2 6 3 < 3 0 0$ operations in total. □

The construction now gives an exact fit from three predictions.

Lemma E.8. Let $g , f _ { 1 } , f _ { 2 } , f _ { 3 } \in L ^ { 2 } ( \mathcal { D } )$ satisfy $\langle g , f _ { i } \rangle = \| f _ { i } \| ^ { 2 } f o r i = 1 , 2 , 3$ . Starting from $f _ { 1 } , f _ { 2 } , f _ { 3 }$ at most 300 projections of g onto spans of at most two available vectors produce $P _ { \mathrm { s p a n } \{ f _ { 1 } , f _ { 2 } , f _ { 3 } \} } g .$

Proof. Set $V = \operatorname { s p a n } \{ f _ { 1 } , f _ { 2 } , f _ { 3 } \}$ . Replacing $g$ by $P _ { V } g$ preserves its inner products and projections within V, so assume $g \in V$ . If $g = 0$ , all inputs are zero by the hypothesis. If dim $V \le 2$ , fit from a pair spanning $V .$ , or one input if its dimension is one.

For dim $V = 3$ , it sufices to show that $n ( f _ { 1 } ) , n ( f _ { 2 } ) , n ( f _ { 3 } )$ are noncollinear. Lemma E.7 then reaches zero, and Lemma E.1 implements every step by a pairwise fit, preserving the number of operations.

If the three points lay on a line through a with direction $b ,$ each $g + n ( f _ { i } )$ would lie in span $\{ g +$ $a , b \}$ . By equation 26, so would each $f _ { i } ,$ , contradicting dim $V ~ = ~ 3$ . This proves the required noncollinearity. □

To also account for the raw feature each agent observes, we will need to keep its contribution in every prediction. The following lemma shows that this is possible.

Lemma E.9. Let $f _ { 1 } , f _ { 2 } , f _ { 3 }$ be agent predictions and let $x _ { j }$ be a raw feature with $Y - f _ { i } \perp x _ { j }$ for $i = { 1 , 2 , 3 }$ . At most 300 added agents, all observing $x _ { j }$ and having at most two parents, sufice to produce

$$
P _ { \mathrm { s p a n } \{ x _ { j } , f _ { 1 } , f _ { 2 } , f _ { 3 } \} } Y .
$$

Proof. Let $h _ { 0 } = P _ { \mathrm { s p a n } \{ x _ { j } \} } Y$ and $h _ { i } = f _ { i } - h _ { 0 }$ . Since $Y - f _ { i } \perp x _ { j }$ and $Y - h _ { 0 } \perp x _ { j }$ , their diference also satisfies $h _ { i } \perp x _ { j }$ . The input space is the orthogonal sum of span $\{ x _ { j } \}$ and span $\{ h _ { 1 } , h _ { 2 } , h _ { 3 } \}$ , so the desired prediction is

$$
P _ { \mathrm { s p a n } \{ x _ { j } , f _ { 1 } , f _ { 2 } , f _ { 3 } \} } Y = h _ { 0 } + P _ { \mathrm { s p a n } \{ h _ { 1 } , h _ { 2 } , h _ { 3 } \} } Y .
$$

The residual $Y - f _ { i }$ is orthogonal to $f _ { i }$ by Lemma 2.1, and to $h _ { 0 } \in \operatorname { s p a n } \{ x _ { j } \}$ by hypothesis. It is thus orthogonal to $h _ { i }$ , giving $\langle Y , h _ { i } \rangle = \langle f _ { i } , h _ { i } \rangle$ . Furthermore, $h _ { i } \perp h _ { 0 }$ implies $\langle f _ { i } , h _ { i } \rangle = \| h _ { i } \| ^ { 2 }$

Apply Lemma E.8 with $g = Y$ and the three vectors $h _ { 1 } , h _ { 2 } , h _ { 3 }$ . It produces $P _ { \mathrm { s p a n } \{ h _ { 1 } , h _ { 2 } , h _ { 3 } \} } Y$ using at most 300 projections of $Y$ , each onto the span of at most two initial vectors or vectors produced by earlier projections.

We will inductively replace each projection by an agent. We maintain that, for every available vector $v ,$ an agent already predicts $h _ { 0 } + v$ . This holds initially because the given agent predicting $f _ { i }$ supplies $h _ { 0 } + h _ { i } = f _ { i }$ for each $i = { 1 , 2 , 3 }$

Consider the next projection, which uses available vectors $u , v$ . By induction, agents predicting $h _ { 0 } + u$ and $h _ { 0 } + v$ already exist. Create an agent observing $x _ { j }$ and receiving these two predictions. Since $h _ { 0 } \in \operatorname { s p a n } \{ x _ { j } \}$ , its input space equals span $\{ x _ { j } \} + \operatorname { s p a n } \{ u , v \}$ . We have $u , v \in \operatorname { s p a n } \{ h _ { 1 } , h _ { 2 } , h _ { 3 } \}$ and hence span $\{ x _ { j } \} \perp \operatorname { s p a n } \{ u , v \}$ . The new agent therefore predicts

$$
P _ { \mathrm { s p a n } \{ x _ { j } , h _ { 0 } + u , h _ { 0 } + v \} } Y = h _ { 0 } + P _ { \mathrm { s p a n } \{ u , v \} } Y .\tag{34}
$$

Thus the new projection also has an agent supplying its sum with $h _ { 0 }$ , completing the induction. Since we replaced each projection with one agent, the total number of added agents is at most 300. □

The given parents need not satisfy $Y - f _ { i } \perp x _ { j }$ , as required by Lemma E.9. We show in the next lemma that with at most three extra agents we can satisfy this condition and then apply Lemma E.9.

Lemma E.10. Consider an agent observing a raw feature $x _ { j }$ and receiving three parent predictions $f _ { 1 } , f _ { 2 } , f _ { 3 }$ . For every distribution with finite second moments, there is a gadget that reproduces this agent’s prediction

$$
P _ { \mathrm { s p a n } \{ x _ { j } , f _ { 1 } , f _ { 2 } , f _ { 3 } \} } Y .
$$

The gadget uses at most 303 added agents, each observing $x _ { j }$ and having at most two parents. The graph and output agent may depend on the distribution.

Proof. Let $V = \operatorname { s p a n } \{ x _ { j } , f _ { 1 } , f _ { 2 } , f _ { 3 } \}$ . If $P _ { V } Y$ lies in the span of $x _ { j }$ and any two of $f _ { 1 } , f _ { 2 } , f _ { 3 }$ , one agent using those inputs sufices. Otherwise, we construct three agents, all observing $x _ { j }$ , whose predictions $p _ { 1 } , p _ { 2 } , p _ { 3 }$ together with $x _ { j }$ span $V$

Start with the fit $p _ { 0 } = P _ { \mathrm { s p a n } \{ x _ { j } \} } Y$ . Since $p _ { 0 } \neq P _ { V } Y$ , at least one of $f _ { 1 } , f _ { 2 } , f _ { 3 }$ has a nonzero inner product with $Y - p _ { 0 }$ . Otherwise this residual would be orthogonal to $V$ . Reorder them so $\langle Y - p _ { 0 } , f _ { 1 } \rangle \neq 0$ . An agent observing $x _ { j }$ and receiving $f _ { 1 }$ predicts $p _ { 1 } = P _ { \mathrm { s p a n } \{ x _ { j } , f _ { 1 } \} } Y$ . This improves on p<sub>0</sub>, so its coeficient on $f _ { 1 }$ is nonzero, giving span $\{ x _ { j } , p _ { 1 } \} = \operatorname { s p a n } \{ x _ { j } , f _ { 1 } \}$

Next, $p _ { 1 }$ is the fit from $x _ { j } , f _ { 1 }$ , but it is not $P _ { V } Y$ . Its residual must therefore have a nonzero inner product with $f _ { 2 }$ or $f _ { 3 }$ . Order these two predictions so $\langle Y - p _ { 1 } , f _ { 2 } \rangle \neq 0$ . By the span identity for $p _ { 1 }$ , an agent receiving $p _ { 1 } , f _ { 2 }$ predicts

$$
p _ { 2 } = P _ { \mathrm { s p a n } \{ x _ { j } , p _ { 1 } , f _ { 2 } \} } Y = P _ { \mathrm { s p a n } \{ x _ { j } , f _ { 1 } , f _ { 2 } \} } Y .
$$

The improvement on $p _ { 1 }$ forces a nonzero coeficient on $f _ { 2 }$ , so span $\{ x _ { j } , p _ { 1 } , p _ { 2 } \} = \operatorname { s p a n } \{ x _ { j } , f _ { 1 } , f _ { 2 } \}$

Finally, $p _ { 2 } \neq P _ { V } Y$ because no two of $f _ { 1 } , f _ { 2 } , f _ { 3 }$ sufice. Its residual is orthogonal to $x _ { j } , f _ { 1 } , f _ { 2 }$ so $\langle Y - p _ { 2 } , f _ { 3 } \rangle \neq 0$ . An agent receiving $p _ { 2 } , f _ { 3 }$ therefore predicts $p _ { 3 } = P _ { \mathrm { s p a n } \{ x _ { j } , p _ { 2 } , f _ { 3 } \} } Y$ with smaller error than $p _ { 2 }$ . Without $f _ { 3 } ,$ , the fit would remain $p _ { 2 }$ , since $Y - p _ { 2 }$ is orthogonal to $x _ { j } , p _ { 2 }$ . Thus the coeficient on $f _ { 3 }$ is nonzero, and

$$
\operatorname { s p a n } \{ x _ { j } , p _ { 1 } , p _ { 2 } , p _ { 3 } \} = \operatorname { s p a n } \{ x _ { j } , f _ { 1 } , f _ { 2 } , f _ { 3 } \} = V .
$$

All three agents observe $x _ { j }$ , so $Y - p _ { i } \perp x _ { j }$ . By Lemma E.9, at most 300 further agents combine their predictions to produce $P _ { V } Y$ . Including the three agents constructed above gives the bound of 303. □

## E.2 Replacing a three-parent agent by a fixed gadget

To fix the graph, we run all replacements of at most 303 agents in parallel and combine their outputs.

Lemma 6.1. Consider an agent observing a raw feature $x _ { j }$ and receiving three parent predictions $f _ { 1 } , f _ { 2 } , f _ { 3 }$ . There is a fixed gadget that reproduces this agent’s prediction

$$
P _ { \mathrm { s p a n } \{ x _ { j } , f _ { 1 } , f _ { 2 } , f _ { 3 } \} } Y
$$

for every distribution with finite second moments. The gadget uses $O ( 1 )$ agents, each observing $x _ { j }$ and having at most two parents, and adds $O ( 1 )$ depth above the original parents.

Proof. List all DAGs with between one and 303 added agents, each observing $x _ { j }$ and receiving at most two predictions from earlier agents or the three external inputs. Include every choice of output agent. There are finitely many choices of parent lists and output after indexing the agents in a topological order. Let B be the number of candidates.

Run these candidates in parallel and combine their outputs through a fixed balanced binary tree whose agents also observe $x _ { j }$ . Set $V = \operatorname { s p a n } \{ x _ { j } , f _ { 1 } , f _ { 2 } , f _ { 3 } \}$ and $g = P _ { V } Y$ . By Lemma E.10, at least one candidate predicts $g .$ . Every candidate and tree prediction lies in $V$ , since agents take linear combinations of their inputs.

Whenever a tree agent receives $g$ from a parent, its input space contains g and lies in $V .$ Since $Y - g \perp V$ , its fit is $g .$ . Following the path from the exact candidate to the root therefore proves that the output is $g .$

The candidates use at most 303B agents and the tree uses $B - 1$ , for a total of at most 304B −1. The additional depth is at most $3 0 3 + \lceil \log _ { 2 } B \rceil$ . Since B is fixed independently of the distribution, these are universal bounds, and the entire graph and its output are fixed. □

## E.3 Applying the gadget to the three-parent constructions

Replacing each three-parent agent by the fixed gadget preserves the predictions throughout a network. We record the size and depth bounds in the next lemma, which applies to both settings.

Lemma E.11. Every single-feature graph with N agents, at most three parents per agent, and output depth D can be replaced by a single-feature graph with O(N) agents, at most two parents per agent, and output depth $O ( D )$ , with the same output prediction for every distribution with finite second moments. The replacement graph and allocation depend only on the original graph and allocation.

Proof. Replace each agent with three parents by the fixed gadget in Lemma 6.1, using the same raw feature throughout. Use the gadget’s output wherever that agent’s prediction is required, including at the network output. Leave other agents unchanged. These choices depend only on G and its allocation.

In a topological order, each gadget receives the same parent predictions as the agent it replaces and therefore computes the same fit. Induction thus preserves the output prediction for every distribution. Each replacement has constant size, giving $O ( N )$ agents in total. Contracting each gadget to one vertex maps any path ending at the new output to a path in $G ,$ which has at most D agents. Each gadget contributes only a constant number of agents to the path, so the new depth is O(D). □

Theorem 6.2. With at most two parents per agent and a single-feature allocation, exact aggregation is possible for every distribution with finite second moments. In the oblivious designer setting, $O ( d ^ { 2 } )$ agents and depth $O ( d \log d )$ sufice for $d \geq 2$ , and one agent sufices for $d = 1$ . In the adaptive designer setting, $O ( r ^ { 2 } )$ agents and depth $O ( r )$ sufice for every feature rank $r \geq 1$

Proof. For the oblivious designer setting with $d \geq 2$ , apply Lemma E.11 to the construction in Theorem 5.1. For $d = 1$ , one agent observing x<sub>1</sub> is exact. For the adaptive designer setting, apply Lemma E.11 to the construction in Theorem 5.2. □

## F Proofs for the lower bound on the number of agents

## F.1 The distribution

We need the upper-triangular covariance entries to satisfy no nonzero polynomial relation with rational coeficients. We define this condition before proving that such entries can be chosen in the required intervals.

For $k \geq 1$ , a polynomial with rational coeficients in the variables $z _ { 1 } , \ldots , z _ { k }$ , denoted $p ( z _ { 1 } , \ldots , z _ { k } )$ is a finite sum of terms $q z _ { 1 } ^ { m _ { 1 } } \cdots z _ { k } ^ { m _ { k } }$ , with a rational coeficient $q$ and nonnegative integer exponents $m _ { 1 } , \ldots , m _ { k }$ . After combining terms with the same powers, the polynomial is zero if every coeficient is zero. It is constant if it does not depend on any variable. For a nonzero polynomial, its degree is the largest sum $m _ { 1 } + \cdots + m _ { k }$ among terms with nonzero coeficients. Its degree in a single variable $z _ { j }$ is the largest exponent $m _ { j }$ among those terms.

For polynomials $p , q$ in the same variables, we say that p divides q if $q = p h$ for some polynomial h. A nonconstant polynomial $p$ is irreducible over Q if it cannot be factored into a product of two nonconstant polynomials with rational coeficients. All polynomials in these definitions must have rational coeficients.

Real numbers $a _ { 1 } , \ldots , a _ { k }$ are algebraically independent over Q if $p ( a _ { 1 } , \ldots , a _ { k } ) \neq 0$ for every nonzero polynomial $p$ with rational coeficients. For example, the polynomial $z _ { 2 } - z _ { 1 } ^ { 2 }$ rules out every pair with $a _ { 2 } = a _ { 1 } ^ { 2 }$

Lemma F.1. For every $d \geq 2 ,$ , there is a symmetric $d \times$ d matrix E with $| E _ { i j } | < 1 / ( 8 d )$ for all $i , j \in [ d ]$ such that the upper-triangular entries of $\Sigma = { \textstyle { \frac { 1 } { 2 } } } I + E$ are algebraically independent over $\mathbb { Q } .$ no nonzero polynomial with rational coeficients vanishes at these entries.

Proof. We construct Σ first and then define $\begin{array} { r } { E = \Sigma - \frac { 1 } { 2 } I . } \end{array}$ . To ensure $| E _ { i j } | < 1 / ( 8 d )$ , we choose each diagonal entry of Σ in the interval $\bigl ( \frac { 1 } { 2 } - \frac { 1 } { 8 d } , \frac { 1 } { 2 } + \frac { 1 } { 8 d } \bigr )$ and each of-diagonal entry in $( - \frac { 1 } { 8 d } , \frac { 1 } { 8 d } )$

We choose the upper-triangular entries of Σ one at a time, in any fixed order, and fill the lower triangle by symmetry. We show that at each step, only countably many values in the permitted interval would violate algebraic independence.

Suppose k entries have been chosen, with algebraically independent values $a _ { 1 } , \ldots , a _ { k }$ . At the first step, $k = 0$ and there are no chosen entries. Fix any nonzero polynomial $p ( z _ { 1 } , \ldots , z _ { k } , z )$ with rational coeficients, and let $m$ be its degree in $z .$ . For each $0 \leq j \leq m$ , collect the terms in which $z$ has exponent $j$ and factor out $z ^ { j }$ and call the remaining polynomial $p _ { j } ( z _ { 1 } , \ldots , z _ { k } )$ , so

$$
p ( z _ { 1 } , \ldots , z _ { k } , z ) = \sum _ { j = 0 } ^ { m } p _ { j } ( z _ { 1 } , \ldots , z _ { k } ) z ^ { j } ,
$$

with each $p _ { j }$ having rational coeficients and $p _ { m }$ nonzero by the choice of $m$ . If $k \geq 1$ , algebraic independence gives $p _ { m } ( a _ { 1 } , \ldots , a _ { k } ) \neq 0$ . If $k = 0$ , the leading coeficient $p _ { m }$ is a nonzero rational

constant. Thus substituting the chosen values leaves a polynomial

$$
p ( a _ { 1 } , \ldots , a _ { k } , z ) = \sum _ { j = 0 } ^ { m } p _ { j } ( a _ { 1 } , \ldots , a _ { k } ) z ^ { j }
$$

of degree m in one real variable z. It has at most m real roots. Excluding these roots ensures that the next entry does not make this particular polynomial vanish.

We must exclude the roots for every such p. There are countably many polynomials with rational coeficients: each is specified by a finite list of rational coeficients and nonnegative integer exponents. Each polynomial excludes finitely many values, so the union of all excluded values is countable. The permitted open interval for each entry of Σ is uncountable, so we can choose $a _ { k + 1 }$ outside this union. Then $p ( a _ { 1 } , \ldots , a _ { k + 1 } ) \neq 0$ for every nonzero polynomial $p$ with rational coeficients, so $a _ { 1 } , \ldots , a _ { k + 1 }$ are algebraically independent. After all upper-triangular entries have been chosen, $\begin{array} { r } { E = \Sigma - \frac { 1 } { 2 } I } \end{array}$ has the required properties. □

We next verify that the bound on the entries of E makes the distribution normalized.

Lemma F.2. For every symmetric E with $\lvert E _ { i j } \rvert < 1 / ( 8 d )$ , the matrix $\Sigma = { \textstyle { \frac { 1 } { 2 } } } I + E$ is positive definite. The distribution in equation 11 has linearly independent features and satisfies

$$
\| x _ { i } \| ^ { 2 } < 1 , \qquad \frac { 1 } { 3 d } < w _ { i } ^ { * } < \frac { 2 } { 3 d } \quad f o r e v e r y i \in [ d ] .
$$

It also satisfies

$$
\mathbb { E } [ x x ^ { \top } ] = \Sigma , \qquad \mathbb { E } [ x Y ] = \Sigma w ^ { * } = c .\tag{35}
$$

For every $u \in \mathbb { R } ^ { d }$ , the linear combination $u ^ { \top } x$ of the features satisfies $\langle u ^ { \top } x , Y \rangle = u ^ { \top } c$ .

Proof. We first show that Σ is positive definite, so that both $x \sim N ( 0 , \Sigma )$ and $\boldsymbol { w ^ { * } } = \Sigma ^ { - 1 } \boldsymbol { c }$ are well defined. The bound on the entries of E and Cauchy–Schwarz give, for every $v \in \mathbb { R } ^ { d }$

$$
| v ^ { \top } E v | \leq \sum _ { i = 1 } ^ { d } \sum _ { j = 1 } ^ { d } | E _ { i j } | \left| v _ { i } \right| \left| v _ { j } \right| \leq \frac { 1 } { 8 d } \left( \sum _ { i = 1 } ^ { d } | v _ { i } | \right) ^ { 2 } \leq \frac { 1 } { 8 } v ^ { \top } v .
$$

Since $\Sigma = I / 2 + E$ , this implies $v ^ { \top } \Sigma v = v ^ { \top } v / 2 + v ^ { \top } E v \ge 3 v ^ { \top } v / 8 > 0$ whenever $v \neq 0$ . Hence Σ is positive definite.

Also, $\mathbb { E } [ x x ^ { \top } ] = \Sigma$ because x has mean zero and covariance Σ. Thus $\| \boldsymbol { v } ^ { \top } \boldsymbol { x } \| ^ { 2 } = \boldsymbol { v } ^ { \top } \Sigma \boldsymbol { v } > 0$ for every nonzero v, proving that the features are linearly independent. Their second moments satisfy

$$
\| x _ { i } \| ^ { 2 } = \Sigma _ { i i } < \frac { 1 } { 2 } + \frac { 1 } { 8 d } < 1 .
$$

To bound the coeficients of $w ^ { * }$ , write $\Sigma w ^ { * } = c$ coordinatewise as

$$
w _ { i } ^ { * } = \frac { 1 } { 2 d } - 2 \sum _ { j = 1 } ^ { d } E _ { i j } w _ { j } ^ { * } .
$$

The entry bound gives $\textstyle \sum _ { j } | E _ { i j } | < 1 / 8$ in every row. Taking absolute values in the coordinate equation and then the maximum over i yields

$$
\operatorname* { m a x } _ { i } | w _ { i } ^ { * } | \leq \frac { 1 } { 2 d } + \frac { 1 } { 4 } \operatorname* { m a x } _ { i } | w _ { i } ^ { * } | , \qquad \mathrm { h e n c e } \qquad \operatorname* { m a x } _ { i } | w _ { i } ^ { * } | \leq \frac { 2 } { 3 d } .
$$

Substituting this bound back into the same equation gives

$$
\left| w _ { i } ^ { * } - \frac 1 { 2 d } \right| \leq 2 \sum _ { j = 1 } ^ { d } \left| E _ { i j } \right| \operatorname* { m a x } _ { k } \left| w _ { k } ^ { * } \right| < \frac 1 4 \cdot \frac 2 { 3 d } = \frac 1 { 6 d } .
$$

Thus $1 / ( 3 d ) < w _ { i } ^ { * } < 2 / ( 3 d )$ for every $i ,$ and $\begin{array} { r } { \sum _ { i } | w _ { i } ^ { * } | < 2 / 3 < 1 } \end{array}$

Finally, the definitions of Y and $w ^ { * }$ give

$$
\mathbb { E } [ x Y ] = \mathbb { E } [ x x ^ { \top } ] w ^ { * } = \Sigma w ^ { * } = c .
$$

Together with $\mathbb { E } [ x x ^ { \top } ] = \Sigma$ , this proves equation 35. For every $u \in \mathbb { R } ^ { d }$ , it also gives $\langle u ^ { \top } x , Y \rangle =$ $\boldsymbol { u } ^ { \top } \mathbb { E } [ \boldsymbol { x } \boldsymbol { Y } ] = \boldsymbol { u } ^ { \top } \boldsymbol { c } .$ □

## F.2 Counting the input inner products

Lemma 7.2. Fix a single-feature DAG on a distribution with d linearly independent features. If $\begin{array} { r } { \sum _ { i = 1 } ^ { | A | } { \binom { | \operatorname { P a } ( A _ { i } ) | + 1 } { 2 } } < { \binom { d } { 2 } } } \end{array}$ , there is a nonzero symmetric matrix $\Delta$ with zero diagonal such that

$$
u ^ { \top } \Delta v = 0\tag{13}
$$

for every agent $A _ { i }$ observing $x _ { \ell }$ and all $u , v \in \{ e _ { \ell } \} \cup \{ w _ { j } : A _ { j } \in \operatorname { P a } ( A _ { i } ) \}$ , including $u = v$

Proof. Write $f _ { i } = w _ { i } ^ { \top } x$ , and let $e \ell$ be the ℓth coordinate vector. We count the equations for the raw feature paired with each parent and for pairs of diferent parents. We then show that these also imply the equations $w _ { j } ^ { \top } \Delta w _ { j } = 0$

Set the diagonal entries of $\Delta$ to zero, leaving $\binom { d } { 2 }$ free entries in a symmetric matrix. At every agent $A _ { i }$ observing $x _ { \ell } .$ , impose

$$
\begin{array} { r } { e _ { \ell } ^ { \top } \Delta w _ { j } = 0 } \\ { w _ { j } ^ { \top } \Delta w _ { k } = 0 } \end{array}
$$

$$
\begin{array} { l } { ( A _ { j } \in \operatorname { P a } ( A _ { i } ) ) , } \\ { ( A _ { j } , A _ { k } \in \operatorname { P a } ( A _ { i } ) , \ j < k ) . } \end{array}
$$

These are homogeneous linear equations in the free entries of ∆. An agent with p parents contributes at most $p + \binom { p } { 2 } = \binom { p + 1 } { 2 }$ equations. By hypothesis, the total is less than $\binom { d } { 2 }$ , so the solution space has positive dimension and contains a nonzero matrix $\Delta$ . The equations $e _ { \ell } ^ { \top } \Delta e _ { \ell } = 0$ already hold $\dot { \Delta } w _ { j } = 0$ under the imposed equations.

Fix such a solution. We prove $w _ { i } ^ { \top } \Delta w _ { i } = 0$ in topological order. At a source observing $x _ { \ell } .$ , the vector $w _ { i }$ is a multiple of $\textstyle e _ { \ell } ,$ , so the claim follows from $\Delta _ { \ell \ell } = 0$

Now consider an agent $A _ { i }$ observing $x _ { \ell } .$ , and assume $w _ { j } ^ { \top } \Delta w _ { j } \ = \ 0$ for each parent $A _ { j }$ . Its prediction is a linear combination of $x _ { \ell }$ and its parents’ predictions. Since the features are linearly independent, the same linear combination relates their coeficient vectors: there are real coeficients α and $\beta _ { j }$ such that

$$
w _ { i } = \alpha e _ { \ell } + \sum _ { A _ { j } \in \mathrm { P a } ( A _ { i } ) } \beta _ { j } w _ { j } .
$$

Substituting this expression and using the symmetry of $\Delta$ gives

$$
\begin{array} { r l } & { w _ { i } ^ { \top } \Delta w _ { i } = \alpha ^ { 2 } \Delta _ { \ell \ell } + 2 \alpha \displaystyle \sum _ { A _ { j } \in \mathrm { P a } ( A _ { i } ) } \beta _ { j } e _ { \ell } ^ { \top } \Delta w _ { j } } \\ & { ~ + \displaystyle \sum _ { A _ { j } , A _ { k } \in \mathrm { P a } ( A _ { i } ) } \beta _ { j } \beta _ { k } w _ { j } ^ { \top } \Delta w _ { k } . } \end{array}
$$

The first term is zero because $\Delta$ has zero diagonal. Every term in the first sum is zero by the imposed raw-feature equations. In the double sum, the terms with $j \neq k$ vanish by the imposed parent equations and symmetry, and those with $j = k$ vanish by the induction hypothesis. Thus $w _ { i } ^ { \top } \Delta w _ { i } = 0$ , completing the induction.

The imposed equations already cover every pair of distinct inputs. The zero diagonal and the induction cover each input paired with itself, so equation 13 holds at every agent. □

## F.3 What exact aggregation requires

To prove Lemma 7.3, we will express the output coeficients as ratios of polynomials. We write C for a symmetric d × d matrix whose upper-triangular entries are separate variables. We write $K ( C )$ for a vector of d polynomials in these variables, and $L ( C )$ for a single polynomial. Evaluating them at the entries of Σ gives the vector $K ( \Sigma )$ and the number $L ( \Sigma )$

Let Σ be chosen and fixed as in Lemma F.1. We will define $K ( C )$ and $L ( C )$ for this Σ, requiring that $K ( \Sigma ) / L ( \Sigma )$ gives the output coeficient vector for this distribution. Another matrix $\Sigma ^ { \prime }$ may lead to a diferent choice of polynomials. We do not require $K ( \Sigma ^ { \prime } ) / L ( \Sigma ^ { \prime } )$ to give the output coeficient vector for the distribution using Σ<sup>′</sup>.

Lemma F.3. Fix E satisfying the conditions in Lemma F.1 and define Σ and the distribution as in equation 11. Fix a single-feature DAG on this distribution. There are a polynomial vector $K ( C )$ and a scalar polynomial $L ( C )$ , with rational coeficients, such that the output coeficient vector is $K ( \Sigma ) / L ( \Sigma )$ and $L ( \Sigma ) > 0$ . If a symmetric matrix $\Delta$ satisfies equation 13 at every agent, then for every $t \in \mathbb { R }$

$$
K ( \Sigma + t \Delta ) = K ( \Sigma ) , \qquad L ( \Sigma + t \Delta ) = L ( \Sigma ) .\tag{36}
$$

Proof. We will construct the polynomials $K _ { i } ( C )$ and $L _ { i } ( C )$ for each agent $A _ { i }$ in topological order. The output agent’s polynomials will give the desired $K ( C )$ and $L ( C )$ . We show that

$$
w _ { i } = \frac { K _ { i } ( \Sigma ) } { L _ { i } ( \Sigma ) } , \qquad L _ { i } ( \Sigma ) > 0 .
$$

Fix any symmetric matrix $\Delta$ satisfying equation 13: at each agent, $u ^ { \top } \Delta v = 0$ for every pair of input coeficient vectors $u , v ,$ , including $u = v$ . We will also show that evaluating $K _ { i }$ and $L _ { i }$ at $\Sigma + t \Delta$ gives the same values as at $\Sigma .$ , for every real t.

For a source observing $x _ { \ell } ,$ , the fit uses only a multiple of $x _ { \ell }$ . By equation 35, we have $\mathbb { E } [ x _ { \ell } Y ] = c _ { \ell }$ and $\mathbb { E } [ x _ { \ell } ^ { 2 } ] = \Sigma _ { \ell \ell }$ , so its coeficient vector is

$$
w _ { i } = { \frac { \langle x _ { \ell } , Y \rangle } { \Vert x _ { \ell } \Vert ^ { 2 } } } e _ { \ell } = { \frac { \mathbb { E } [ x _ { \ell } Y ] } { \mathbb { E } [ x _ { \ell } ^ { 2 } ] } } e _ { \ell } = { \frac { c _ { \ell } } { \Sigma _ { \ell \ell } } } e _ { \ell } .
$$

Set $K _ { i } ( C ) = c _ { \ell } e _ { \ell }$ and $L _ { i } ( C ) = C _ { \ell \ell }$ . Their coeficients are rational because $c _ { \ell } = 1 / ( 4 d )$ , and $L _ { i } ( \Sigma ) =$ $\Sigma _ { \ell \ell } > 0$ because Σ is positive definite. The numerator does not depend on C. For the denominator, applying the condition on $\Delta$ with $u = v = e _ { \ell } \mathrm { g i v e s } \Delta _ { \ell \ell } = 0$ . Hence $L _ { i } ( \Sigma + t \Delta ) = \Sigma _ { \ell \ell } + t \Delta _ { \ell \ell } = L _ { i } ( \Sigma )$

Now suppose the polynomials have been constructed for the parents of $A _ { i }$ , which observes $x \ell$ Choose parents $A _ { j _ { 1 } } , \dotsc , A _ { j _ { s } }$ so that the inputs $x _ { \ell } , f _ { j _ { 1 } } , \ldots , f _ { j _ { s } }$ are linearly independent and

$$
\operatorname { s p a n } \{ x _ { \ell } , f _ { j _ { 1 } } , \ldots , f _ { j _ { s } } \} = \operatorname { s p a n } \bigl ( \{ x _ { \ell } \} \cup \{ f _ { j } : A _ { j } \in \operatorname { P a } ( A _ { i } ) \} \bigr ) .
$$

We can keep $x _ { \ell }$ in this selection because $\| x _ { \ell } \| ^ { 2 } = \Sigma _ { \ell \ell } > 0$ . If $x _ { \ell }$ alone spans all the inputs, take $s = 0$ . This selection depends on Σ, but not on $\Delta$

Since the features are linearly independent, the coeficient vectors $e _ { \ell } , w _ { j _ { 1 } } , \ldots , w _ { j _ { s } }$ are also linearly independent and span all coeficient vectors the agent can use. By induction, $K _ { j } ( \Sigma ) =$ $L _ { j } ( \Sigma ) w _ { j }$ with $L _ { j } ( \Sigma ) > 0$ for every parent. Replacing each selected $w _ { j }$ by $K _ { j } ( \Sigma )$ therefore only multiplies that vector by a nonzero scalar, preserving both independence and the span.

Using these selected parent indices, define the $d \times ( s + 1 )$ matrix

$$
B _ { i } ( C ) = \left[ e _ { \ell } \quad K _ { j _ { 1 } } ( C ) \quad \cdots \quad K _ { j _ { s } } ( C ) \right] .
$$

Every entry of $B _ { i } ( C )$ is either 0, 1, or an entry of a selected $K _ { j _ { k } } ( C )$ , so it is a polynomial with rational coeficients.

All allowed coeficient vectors for $A _ { i }$ are of the form $w = B _ { i } ( \Sigma ) \alpha$ for some $\alpha \in \mathbb { R } ^ { s + 1 }$ . We derive the equations for the best fit one input at a time. Name the columns of $B _ { i } ( \Sigma )$ as

$$
b _ { 0 } = e _ { \ell } , \qquad b _ { k } = K _ { j _ { k } } ( \Sigma ) \quad ( 1 \leq k \leq s ) .
$$

The column b represents the raw feature $b _ { 0 } ^ { \top } x = x _ { \ell }$ . Each other column represents a multiple of a parent prediction:

$$
\begin{array} { r } { \boldsymbol { b } _ { k } ^ { \top } \boldsymbol { x } = K _ { j _ { k } } ( \Sigma ) ^ { \top } \boldsymbol { x } = L _ { j _ { k } } ( \Sigma ) \boldsymbol { w } _ { j _ { k } } ^ { \top } \boldsymbol { x } = L _ { j _ { k } } ( \Sigma ) \boldsymbol { f } _ { j _ { k } } . } \end{array}
$$

Now take $w = w _ { i }$ , the agent’s fitted coeficient vector. Its residual $Y - w ^ { \top } ;$ x is orthogonal to every input, and therefore to $b _ { k } ^ { \top } x$ for every $0 \leq k \leq s$

Fix one such column $b _ { k }$ . Expanding this orthogonality condition gives a scalar equation:

$$
\begin{array} { r l } & { 0 = \langle b _ { k } ^ { \top } x , Y - w ^ { \top } x \rangle } \\ & { \quad = \mathbb E [ ( b _ { k } ^ { \top } x ) Y ] - \mathbb E [ ( b _ { k } ^ { \top } x ) ( w ^ { \top } x ) ] } \\ & { \quad = b _ { k } ^ { \top } \mathbb E [ x Y ] - b _ { k } ^ { \top } \mathbb E [ x x ^ { \top } ] w } \\ & { \quad = b _ { k } ^ { \top } c - b _ { k } ^ { \top } \Sigma w . } \end{array}
$$

To obtain the third line, use $( b _ { k } ^ { \top } x ) ( w ^ { \top } x ) = b _ { k } ^ { \top } x x ^ { \top } \boldsymbol { \imath }$ w and take the fixed vectors $b _ { k } , w$ outside the expectations. The last line uses $\mathbb { E } [ x Y ] = c$ and $\mathbb { E } [ x x ^ { \top } ] = \Sigma$ from equation 35.

We have therefore obtained $b _ { k } ^ { \top } \Sigma \dot { w } = b _ { k } ^ { \top } c$ for every $k = 0 , \ldots , s .$ The rows of $B _ { i } ( \Sigma ) ^ { \top }$ are exactly $b _ { 0 } ^ { \top } , \ldots , b _ { s } ^ { \top }$ , so these equations together say $B _ { i } ( \Sigma ) ^ { \top } \Sigma w = B _ { i } ( \Sigma ) ^ { \top } c$ . Finally, substituting $w = B _ { i } ( \Sigma ) \alpha$ yields

$$
B _ { i } ( \Sigma ) ^ { \top } \Sigma B _ { i } ( \Sigma ) \alpha = B _ { i } ( \Sigma ) ^ { \top } c .
$$

Write $M _ { i } ( C ) = B _ { i } ( C ) ^ { \top } C B _ { i } ( C )$ for the matrix on the left, with Σ replaced by C. At Σ it is positive definite: for every nonzero $\alpha .$

$$
\alpha ^ { \top } M _ { i } ( \Sigma ) \alpha = ( B _ { i } ( \Sigma ) \alpha ) ^ { \top } \Sigma ( B _ { i } ( \Sigma ) \alpha ) > 0 .
$$

The inequality holds because the columns of $B _ { i } ( \Sigma )$ are independent, so $B _ { i } ( \Sigma ) \alpha \neq 0$ , and Σ is positive definite. Thus $M _ { i } ( \Sigma )$ is invertible and has positive determinant.

By the definition of $M _ { i }$ , the equation for the fitted weights is $M _ { i } ( \Sigma ) \alpha = B _ { i } ( \Sigma ) ^ { \top } c$ . Multiplying both sides on the left by $M _ { i } ( \Sigma ) ^ { - 1 }$ gives

$$
\alpha = M _ { i } ( \Sigma ) ^ { - 1 } B _ { i } ( \Sigma ) ^ { \top } c .
$$

The entries of α are the weights on the columns of $B _ { i } ( \Sigma )$ . To recover the coeficients on the raw features, substitute this solution into $w _ { i } = B _ { i } ( \Sigma ) \alpha$

$$
w _ { i } = B _ { i } ( \Sigma ) \alpha = B _ { i } ( \Sigma ) M _ { i } ( \Sigma ) ^ { - 1 } B _ { i } ( \Sigma ) ^ { \top } c .
$$

The inverse $M _ { i } ( C ) ^ { - 1 }$ introduces division by det $M _ { i } ( C )$ , so we separate the numerator and denominator to obtain polynomials with rational coeficients.

For a square matrix M, let $\operatorname { a d j } ( M )$ be its adjugate, the transpose of its cofactor matrix. The identity $M \mathrm { a d j } ( M ) = ( \operatorname* { d e t } M ) I$ gives $M ^ { - 1 } = \mathrm { a d j } ( M ) /$ det M when M is invertible. We therefore define

$$
K _ { i } ( C ) = B _ { i } ( C ) \mathrm { a d j } ( M _ { i } ( C ) ) B _ { i } ( C ) ^ { \top } c , \qquad L _ { i } ( C ) = \mathrm { d e t } M _ { i } ( C ) .
$$

The determinant and every entry of the adjugate are polynomials in the matrix entries. Since the entries of $B _ { i } ( C )$ are polynomials with rational coeficients and $c = \mathbf { 1 } _ { d } / ( 4 d )$ is rational, the same holds for $L _ { i } ( C )$ and each entry of $K _ { i } ( C )$ . The formula for the fit now gives $w _ { i } = K _ { i } ( \Sigma ) / L _ { i } ( \Sigma )$ , with $L _ { i } ( \Sigma ) > 0$

Finally, replace Σ by $\Sigma + t \Delta$ in these polynomials. By induction, $K _ { j } ( \Sigma + t \Delta ) = K _ { j } ( \Sigma )$ for every parent. In particular, the selected columns $K _ { j _ { 1 } } , \ldots , K _ { j }$ have the same values at both matrices. The column $e _ { \ell }$ is fixed, so ${ \cal B } _ { i } ( \Sigma + t \Delta ) = { \cal B } _ { i } ( \Sigma )$

Every column of $B _ { i } ( \Sigma )$ is either $e _ { \ell }$ or $L _ { j } ( \Sigma ) w _ { j }$ for a parent $A _ { j }$ . Thus each entry of $B _ { i } ( \Sigma ) ^ { \top } \Delta B _ { i } ( \Sigma )$ is a scalar multiple of $u ^ { \top } \Delta v$ for input coeficient vectors $u , v$ . These products are zero by equation 13, so

$$
\begin{array} { r l } & { M _ { i } ( \Sigma + t \Delta ) = B _ { i } ( \Sigma ) ^ { \top } ( \Sigma + t \Delta ) B _ { i } ( \Sigma ) } \\ & { ~ = M _ { i } ( \Sigma ) + t B _ { i } ( \Sigma ) ^ { \top } \Delta B _ { i } ( \Sigma ) = M _ { i } ( \Sigma ) . } \end{array}
$$

The formulas defining $K _ { i }$ and $L _ { i }$ use only $B _ { i } , \ M _ { i } .$ and the fixed vector $^ { c , }$ so their values are unchanged as well. These equalities hold for every real $t ,$ even when $\Sigma + t \Delta$ is singular, because the formulas defining the polynomials do not require its inverse. This completes the induction. The polynomials for the output agent give K and L. □

A factorization into irreducibles of a nonzero polynomial $f$ is an expression $f = a p _ { 1 } \cdot \cdot \cdot p _ { r }$ , with a a nonzero rational constant and each $p _ { i }$ irreducible over $\mathbb { Q } .$ Factors may repeat. For a nonzero constant $f ,$ we take $r = 0$ and interpret the empty product as 1.

The following standard theorem makes precise how two such factorizations can difer [DF03].

Theorem F.4 (Unique factorization of polynomials). Every nonzero polynomial with rational coefficients in finitely many variables has a factorization into irreducibles. If $f = a p _ { 1 } \cdot \cdot \cdot p _ { r } = b q _ { 1 } \cdot \cdot \cdot q _ { s }$ are two such factorizations, then $r = s$ . After reordering the factors, for each i there is a nonzero rational constant $u _ { i }$ such that $q _ { i } = u _ { i } p _ { i }$

The following lemma is a consequence of Theorem $\mathrm { F . 4 }$

Lemma F.5 (Irreducible divisors of a product). Let $p , f , g$ be polynomials with rational coeficients in the same variables. $I f p$ is irreducible over $\mathbb { Q }$ and divides $f g$ , then p divides f or p divides $g .$

We now prove that the determinant of any symmetric matrix with separate upper-triangular variables meets the irreducibility hypothesis of Lemma F.5. This is also a standard result and we include a proof for completeness.

Lemma F.6. Let C be a symmetric $d \times d$ matrix whose upper-triangular entries are separate variables. Then the polynomial det C in these variables is irreducible over the rational numbers.

Proof. We use induction on $d .$ For $d = 1$ , the determinant is a single variable and is irreducible. For $d \geq 2$ , write

$$
C = \left( \begin{array} { l l } { t } & { v ^ { \top } } \\ { v } & { B } \end{array} \right) .
$$

The coeficient of t in det C is det $B ,$ obtained by deleting the first row and column. Thus det $C$ has degree one in t.

Suppose det $C = p q$ for polynomials $p , q$ in the entries of $C ,$ , with rational coeficients. Since degrees in t add under multiplication, one of $p , q$ must be independent of t. Name the factors so that $p$ is independent of t. Then $q$ has degree one in t. Let $h _ { 0 }$ and $h _ { 1 }$ be its coeficient of t and its constant term, respectively. Both are polynomials in the entries of B and $v ,$ independent of t, and

$$
\begin{array} { r } { \operatorname* { d e t } C = p ( h _ { 0 } t + h _ { 1 } ) . } \end{array}
$$

The coeficient of t on the right is $p h _ { 0 }$ . Comparing with the coeficient of t in det $C$ gives

$$
\operatorname* { d e t } B = p h _ { 0 } .
$$

We will apply the induction hypothesis to this factorization of det B. Since det $B = p h _ { 0 }$ is independent of every entry of $v ,$ both $p$ and $h _ { 0 }$ must also be independent of those entries: degrees in each variable add under multiplication. Thus $p$ and $h _ { 0 }$ are polynomials only in the entries of B.

By induction, det $B$ is irreducible, so $p$ or $h _ { 0 }$ must be constant. If $p$ is constant, we are done. Otherwise, h<sub>0</sub> is a nonzero rational constant. Substituting $p = ( \operatorname* { d e t } B ) / h _ { 0 }$ into the factorization of det C gives

$$
\operatorname* { d e t } C = p ( h _ { 0 } t + h _ { 1 } ) = ( \operatorname* { d e t } B ) \left( t + { \frac { h _ { 1 } } { h _ { 0 } } } \right) .
$$

This identity would force det $C = 0$ at every matrix with det $B = 0$ , because division by the nonzero constant $h _ { 0 }$ is always defined.

To rule this out, consider the numerical matrix

$$
C _ { 0 } = \left( \begin{array} { c c c } { { 0 } } & { { 1 } } & { { 0 } } \\ { { 1 } } & { { 0 } } & { { 0 } } \\ { { 0 } } & { { 0 } } & { { I _ { d - 2 } } } \end{array} \right) ,
$$

using just the upper-left $2 \times 2$ block when $d = 2$ . Let $B _ { 0 }$ be the lower-right $( d - 1 ) \times ( d - 1 )$ block of $C _ { 0 }$ . The first row of $B _ { 0 }$ is zero, so det $B _ { 0 } = 0$ . But det $C _ { 0 } = - 1$ , from the determinant of the upperleft $2 \times 2$ block and the remaining identity block. This contradicts det $C = ( \operatorname* { d e t } B ) ( t + h _ { 1 } / h _ { 0 } )$ , whose right-hand side is zero at $C _ { 0 }$ . Therefore p must be constant, proving that det C is irreducible.

Lemma 7.3. For the distribution fixed in Equation (11), suppose a single-feature $D A G$ achieves exact aggregation. If a symmetric matrix $\Delta$ satisfies equation 13 at every agent, then $\Delta = 0$

Proof. Take $K , L$ from Lemma F.3. We first show that det C divides $L ( C )$ . We then use a singular matrix on the line $\Sigma + t \Delta$ to rule out $\Delta \neq 0$

Exact aggregation means that the output coeficient vector equals $\boldsymbol { w ^ { * } } = \boldsymbol { \Sigma ^ { - 1 } } \boldsymbol { c }$ , since the features are linearly independent. Hence

$$
{ \frac { K ( \Sigma ) } { L ( \Sigma ) } } = \Sigma ^ { - 1 } c ,
$$

and so $\Sigma K ( \Sigma ) = L ( \Sigma ) c$ . Each coordinate of $C K ( C ) - L ( C ) c$ is a polynomial with rational coeficients in the upper-triangular entries of $C _ { i }$ , and the displayed equality says that it vanishes at $\Sigma .$ Those entries of Σ are algebraically independent by Lemma F.1. Each coordinate must therefore be the zero polynomial, giving

$$
C K ( C ) = L ( C ) c .
$$

This identity holds for every symmetric C, including singular matrices.

Let $\widetilde { C }$ be obtained from C by replacing its first column with $C K ( C )$ . Also let $C ^ { \prime }$ be obtained from C by replacing its first column with $\mathbf { 1 } _ { d }$ . Since $C K ( C ) = L ( C ) c = ( L ( C ) / ( 4 d ) ) \mathbf { 1 } _ { d } .$ the first column of $\widetilde { C }$ is $L ( C ) / ( 4 d )$ times the first column of $C ^ { \prime }$ , and all other columns agree. Factoring this scalar out of the first column gives

$$
\operatorname* { d e t } \widetilde { C } = \frac { L ( C ) } { 4 d } \operatorname* { d e t } C ^ { \prime } .
$$

To compute the same determinant another way, write $C = [ C _ { 1 } , \dots , C _ { d } ]$ , so $C _ { j }$ is the jth column, and write $( K ( C ) ) _ { j }$ for the jth coordinate of $K ( C )$ . By definition,

$$
C K ( C ) = \sum _ { j = 1 } ^ { d } ( K ( C ) ) _ { j } C _ { j } .
$$

The first column of $\widetilde { C }$ is this sum, and its remaining columns are $C _ { 2 } , \ldots , C _ { d }$ . Since the determinant is linear in its first column,

$$
\operatorname* { d e t } \widetilde { C } = \sum _ { j = 1 } ^ { d } ( K ( C ) ) _ { j } \operatorname* { d e t } [ C _ { j } , C _ { 2 } , \ldots , C _ { d } ] .
$$

For $j = 1$ , the matrix inside the determinant is C. For every $j \geq 2 ,$ , its first and jth columns are both $C _ { j }$ , so its determinant is zero. Only the $j = 1$ term remains, giving

$$
\operatorname* { d e t } { \widetilde { C } } = ( K ( C ) ) _ { 1 } \operatorname* { d e t } C .
$$

Equating the two expressions for det $\widetilde { C }$ and multiplying by 4d gives

$$
4 d \left( K ( C ) \right) _ { 1 } \operatorname* { d e t } C = L ( C ) \operatorname* { d e t } C ^ { \prime } .
$$

The displayed identity shows that det C divides the product $L ( C )$ det $C ^ { \prime } .$ . All three are polynomials with rational coeficients in the upper-triangular entries of $C ,$ , and det C is irreducible by Lemma F.6. By Lemma F.5, det C therefore divides $L ( C )$ or det $C ^ { \prime }$

It cannot divide det $C ^ { \prime }$ . The first column of $C ^ { \prime }$ consists of ones, so each term in its determinant is a product of $d - 1$ entries of $C .$ Thus det $C ^ { \prime }$ has degree at most $d - 1$ . It is also a nonzero polynomial: at $C = I$ , subtracting the other columns of $C ^ { \prime }$ from its first column gives I without changing the determinant, so det $C ^ { \prime } = 1$ there. In contrast, det C has degree d. Hence det $C$ cannot divide det $C ^ { \prime }$ and must instead divide $L ( C )$

Suppose $\Delta \neq 0$ . We will find a real t for which $\Sigma + t \Delta$ is singular. Since Σ is positive definite, set

$$
M = \Sigma ^ { - 1 / 2 } \Delta \Sigma ^ { - 1 / 2 } .
$$

The matrix M is symmetric because $\Delta$ and $\Sigma ^ { - 1 / 2 }$ are symmetric. It is nonzero because $\Delta =$ $\Sigma ^ { 1 / 2 } M \Sigma ^ { 1 / 2 }$ and $\Delta \neq 0$ . Thus $M \neq 0$ has a nonzero real eigenvalue λ and a nonzero vector v with $M v = \lambda v .$

$$
\mathrm { S e t } \ t = - 1 / \lambda , \mathrm { s o } \ ( I + t M ) v = ( 1 + t \lambda ) v = 0 . \ \mathrm { U s i n g } \ \Delta = \Sigma ^ { 1 / 2 } M \Sigma ^ { 1 / 2 } , \mathrm { w e \ o b t a i n }
$$

$$
( \Sigma + t \Delta ) ( \Sigma ^ { - 1 / 2 } v ) = \Sigma ^ { 1 / 2 } ( I + t M ) v = 0 .
$$

The vector $\Sigma ^ { - 1 / 2 } v$ is nonzero because $\Sigma ^ { - 1 / 2 }$ is invertible and $v \neq 0$ . Hence $\Sigma + t \Delta$ has a nonzero vector in its kernel and is singular. Since det C divides $L ( C )$ , we have $L ( C ) = \operatorname* { d e t } C \cdot H ( C )$ for some polynomial H. Since det $( \Sigma + t \Delta ) = 0$ , this forces $L ( \Sigma + t \Delta ) = 0$ . But Lemma F.3 gives $L ( \Sigma + t \Delta ) = L ( \Sigma ) > 0$ because equation 13 holds. This contradiction proves $\Delta = 0$ □

## F.4 Proof of the lower bound

Proof of Theorem 7.1. Fix d $\geq 2$ and let the adversary choose the distribution from Section 7. It is normalized and has linearly independent features by Lemma F.2. Since Y is a linear combination of the features, $Y = f ^ { * }$ . With knowledge of this distribution, the designer may choose any graph $G = ( A , E )$ , single-feature allocation, and output agent for which $f _ { G } = Y$

Suppose, for contradiction, that G violates equation 10. By Lemma 7.2, there is a nonzero symmetric matrix $\Delta$ satisfying equation 13 at every agent. Since G is exact on the chosen distribution, Lemma 7.3 gives $\Delta = 0$ , a contradiction. Thus G satisfies equation 10. □