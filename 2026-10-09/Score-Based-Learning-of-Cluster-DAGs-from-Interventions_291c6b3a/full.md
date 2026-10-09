# Score-Based Learning of Cluster DAGs from Interventions

Gaetano Tedesco Informatics Institute, University of Amsterdam

## Abstract

Graphical approaches to causal abstraction transform a low-level causal directed acyclic graph (DAG) over many measured variables into a smaller, high-level DAG whose nodes cluster the original variables and whose edges summarize the causal relations between clusters. Such cluster DAGs are easier to interpret, but learning them requires finding the clusters and recovering the edges between them. Madaleno et al. [39] learn the interventional coarsening—the cluster DAG that merges variables the interventions cannot distinguish—in two constraint-based phases: first the clusters, then the edges. We introduce coarse, the first score-based method for this task: it keeps the two-phase structure but, under linear Gaussian assumptions, swaps the constraint-based edge phase for a score-based one. We show that the interventions themselves identify a causal order over the clusters, and learning the edges reduces to a single local search per cluster under a cluster-level BIC score. We prove that the procedure runs in polynomial time and, provided the variables afected by each intervention are correctly identified, that it is consistent. On synthetic and real-world interventional data, coarse matches state-of-the-art edge recovery given enough samples, with an edge phase up to two orders of magnitude faster, including on dense graphs with hundreds of nodes.

## 1 INTRODUCTION

Recovering the causal structure of a system is a central goal across the empirical sciences, and the large body of work on causal discovery shows how to learn such structure from data at the granularity of the recorded variables [13, 60, 61]. That resolution is not always the

Alex Markham Department of Mathematical Sciences, University of Copenhagen

most useful one. Fine-grained features such as pixels, neurons, or individual transcripts are not the units in which one wants to reason, intervene, or communicate [11, 27, 63], and searching over DAGs with thousands of nodes is computationally daunting: the number of DAGs grows super-exponentially in the number of variables, and large-sample learning of Bayesian networks is NP-hard [14]. Both concerns call for coarser causal descriptions that remain causally valid while being more interpretable and statistically better behaved. Causal abstraction formalizes this idea, relating two models of the same system at diferent granularities so that the coarse model preserves the efects of interventions predicted by the fine one [5, 56]. We work with the graphical version of this idea, in which the coarse model is a cluster DAG: a DAG whose nodes are blocks of the recorded variables and whose edges summarize the causal relations between blocks.

The literature on causal abstraction has developed along two complementary lines. The functional view encodes an abstraction as a pair of maps, one on variable values and one on interventions, such that ab stracting and intervening commute. Rubenstein et al. [56] introduced this as exact transformations, Beckers and Halpern [5] refined it into a hierarchy of ab stractions, and later work extended it to the approxi mate [6] and soft-interventional [45] cases. The graphical view operates directly on the causal DAG: clus ter DAGs (C-DAGs) [1] carry the standard graphical calculus over an assumed clustering of the variables, transit clusters [67] collapse nodes while preserving the identifiability of a chosen causal query, and Wahl et al. [75] study when Markov properties and faithfulness transfer under such grouping. Finally, Schooltink and Zennaro [57] show that the C-DAGs of the graph ical view are exactly the constructive abstractions of the functional hierarchy, so the two views describe the same objects. Most of this work is theoretical in spirit, and learning an abstraction from data remains largely open: existing methods either take the partition as input [1, 20, 74], fix it to preserve a chosen estimand [67], or learn abstraction maps under strong parametric assumptions [9, 15, 16, 21, 46, 49, 79, 81].

Closest to us, the coarsening framework of Madaleno et al. [39] gives a lattice-theoretic account of all valid partition-induced DAGs. The framework has since been extended to cyclic models, where the condensation of the fine graph arises as an identifiable coarse summary [40]. Among them, it singles out the interventional coarsening, which merges exactly those variables that the available interventions cannot distinguish, and learns it from interventional data with unknown targets using RePaRe, a constraint-based procedure in two phases: first the clusters, then the edges, decided by conditional independence (CI) tests. App. A gives a fuller account of the related literature.

Constraint-based learning of this kind inherits the familiar fragilities of CI testing. Each edge decision rests on many hypothesis tests whose reliability degrades as conditioning sets grow and whose errors accumulate across decisions [54, 69]. Score-based methods avoid testing altogether, replacing each CI query with a comparison of a decomposable score; this is the route behind the most successful practical structure searches, from greedy equivalence search to recent order-based and discrete searches, most recently from interventional data with known targets [4, 12, 13, 66, 78]. For cluster DAGs, no such method exists. Only the constraint-based RePaRe learns the interventional coarsening; classical score-based searches do not directly apply, since they target a single DAG over the fine variables, not a coarsening thereof.

We introduce coarse, the first score-based method for learning the interventional coarsening from interventional—more generally, heterogeneous—data. The setting is linear Gaussian: each non-observational environment is a soft intervention on an unknown, possibly multi-target set of variables, so all edges are preserved and every environment is Markov to the fine DAG. coarse keeps the two-phase template—clusters first, edges second—and replaces the CI-tested edge phase with score comparisons. Two observations make this work. First, the interventions themselves identify a causal order over the blocks. Each variable carries an intervention signature, the set of environments that shift its marginal; signatures only grow along directed paths, so comparing them yields a topological order of the cluster DAG and pins down each block’s possible parents. No search over orderings is needed. Second, the fine structural equation model induces a Gaussian block-level model: a curved exponential family whose likelihood factorizes over blocks. The Bayesian information criterion (BIC) [58], summed over environments, therefore decomposes per block and is locally consistent, and classical local searches recover the true parents at the block level. With the order and each block’s possible parents settled by the signatures, edge recovery reduces to a single score-based local search per block, in the spirit of grow-shrink [37, 43, 78]. Assuming error-free signatures, we prove the resulting procedure consistent, with worst-case polynomial running time.

In sum, our contributions are: (i) we extend the interventional coarsening of Madaleno et al. [39] to unknown, multi-target interventions—the resulting coarsening merges variables with equal intervention signatures, coincides with theirs under single-target interventions, and remains identifiable from marginal shifts in the multi-target case, where theirs is not; (ii) we show that the same shift information identifies a causal order over the blocks, removing the search over orderings and confining each block’s candidate parents to a set with all of its parents and none of its descendants; (iii) we score coarse parent sets with the BIC pooled over environments, prove the score decomposable and locally consistent, and prove coarse consistent and with worst-case polynomial running time; and (iv) on synthetic and real-world interventional data, coarse matches state-of-the-art edge recovery given enough samples, while its edge phase runs up to two orders of magnitude faster. Code is available at https://github.com/tedescoG/coarse.

## 2 PRELIMINARIES

We review some definitions and results about graphs, probability distributions, our data-generating model, coarsenings, and local structure search that will be useful for describing our new results in Sec. 3.

## 2.1 Graphs and orders

Let $\mathcal { G } = ( V , E )$ be a directed acyclic graph (DAG) with node set $V = [ d ]$ , indexing a random vector $X =$ $( X _ { v } ) _ { v \in V } \in \mathbb { R } ^ { d }$ with joint distribution $P _ { X }$ , and edge set E. Two nodes are adjacent if an edge joins them, and the skeleton sk(G) is the undirected graph obtained by ignoring edge directions. We write $\operatorname { p a } _ { \mathcal { G } } ( v )$ and $\operatorname { c h } _ { \mathcal { G } } ( v )$ for the parents and children of v. The ancestors an<sub>G</sub>(v) (descendants $\deg ( v ) )$ are the nodes w with a directed path from w to v (from v to w), with the convention $v \in \mathrm { a n } _ { \mathcal { G } } ( v ) \cap \mathrm { d e } _ { \mathcal { G } } ( v )$ . The non-descendants are $\mathrm { n d } _ { \mathcal { G } } ( v ) : = V \backslash \backslash \mathrm { d e } _ { \mathcal { G } } ( v )$ , and all notations extend to sets $A \subseteq V$ by union, e.g. $\begin{array} { r } { \mathrm { d e } _ { \mathcal { G } } ( A ) : = \bigcup _ { a \in A } \mathrm { d e } _ { \mathcal { G } } ( a ) } \end{array}$

Definition 1 (Ancestral partial order). For $u , v \in V$ 2 write $u \preceq g v \mathrm { i f f } u \in \mathrm { a n } _ { \mathcal { G } } ( v )$ , and $u \prec _ { \mathcal { G } } v \operatorname { i f f } u \preceq _ { \mathcal { G } }$ v and u ${ \bf \nabla } \cdot \ne v .$

Definition 2 (Topological ordering). A permutation $\tau : V  V$ is a topological (or causal) ordering of G if $\tau ( u ) < \tau ( v )$ for every $u \in \mathrm { a n } _ { \mathcal { G } } ( v ) \setminus \{ v \}$

We write $\mathcal { T } ( \leq )$ for the set of all linear extensions of a partial order $\leq$ , and $\tau _ { \prec v } : = \{ u \mid \tau ( u ) < \tau ( v ) \}$ for the nodes before v in $\tau .$ The topological orders of $\mathcal { G }$ are exactly the elements of $\tau ( { \preceq } _ { \mathcal { G } } )$ , and each satisfies $\mathrm { p a } _ { \mathcal { G } } ( v ) \subseteq \tau _ { \prec v } \subseteq \mathrm { n d } _ { \mathcal { G } } ( v )$

## 2.2 Conditional independence and Markov properties

We write $X _ { A }$ P $X _ { B } ~ \mid ~ X _ { C }$ when $X _ { A }$ and $X _ { B }$ are conditionally independent given $X _ { C }$ under $P _ { X }$ , and $A \perp \perp _ { \mathcal { G } } B \mid C$ when A and B are d-separated by C in ${ \mathcal { G } } .$ Definition 3 (Markov property and faithfulness). Let $\mathcal { G } = ( V , E )$ be a DAG and let $P _ { X }$ be a joint probability distribution over X admitting a strictly positive density $p _ { X }$ . We say that $P _ { X }$ satisfies the Markov $f a c -$ torization property with respect to G if

$$
p _ { X } ( x ) = \prod _ { v \in V } p \big ( x _ { v } \mid x _ { \mathtt { p a } _ { \mathcal G } ( v ) } \big ) .\tag{1}
$$

This is equivalent to the local Markov property, $X _ { v } \perp \perp _ { P }$ $X _ { \mathrm { n d } _ { \mathcal { G } } ( v ) \backslash \mathrm { p a } _ { \mathcal { G } } ( v ) } \mid X _ { \mathrm { p a } _ { \mathcal { G } } ( v ) }$ for every v, and to the global Markov property, A <sub>G</sub> $B \mid S \Rightarrow X _ { A }$ P $X _ { B } ~ \vert ~ X _ { S }$ [38] for all pairwise disjoint $A , B , S \subseteq V$ . Moreover we say that $P _ { X }$ is faithful to $\mathcal { G }$ when every conditional independence in $P _ { X }$ is implied by (1).

Two DAGs $\mathcal { G }$ and $\mathcal { G } ^ { \prime }$ are Markov equivalent, written $\mathcal { G } \sim \mathcal { G } ^ { \prime }$ , if they entail the same conditional independence statements, which holds if and only if they have the same skeleton and the same v-structures [73]. We write [G] for the Markov equivalence class of $\mathcal { G } .$

## 2.3 Local structure search

Global methods search over whole graphs on V. Local methods instead fix a single target $v \in V$ and ask which subset of the remaining variables shields $X _ { v }$ from everything else. We call a set $B { \mathcal { L } } ( v ) \subseteq V \setminus \{ v \}$ a Markov blanket of $X _ { v }$ if $X _ { v } ~ \perp \perp _ { P } ~ X _ { V \setminus ( B \mathcal { L } ( v ) \cup \{ v \} ) }$ $X _ { B \mathcal { L } ( v ) }$ An inclusion-minimal Markov blanket is called a Markov boundary and we denote it by $B ( v )$ . If $\perp _ { P }$ is a graphoid, which holds under strict positivity of $p _ { X }$ , then $B ( v )$ exists and is unique; under the Markov property and faithfulness, it consists of the parents, children, and parents of children of v [51]. Graphoids are defined in App. B.1. In practice we can restrict the search to a candidate pool fixed in advance, typically by a topological order.

Definition 4 (Restricted Markov boundary; 37). Let $v \in V$ and $Z \subseteq V \backslash \{ v \}$ . A set $B ( v , Z ) \subseteq Z$ is a Markov boundary of $X _ { v }$ relative to Z if

$$
\mathrm { ( i ) } \ X _ { v } \ \perp \perp _ { P } X _ { Z \setminus B ( v , Z ) } \ | \ X _ { B ( v , Z ) } ;
$$

(ii) no proper subset $B ^ { \prime } \subsetneq B ( v , Z )$ satisfies (i).

The case $Z = V \backslash \{ v \}$ recovers $B ( v )$ , and existence and uniqueness carry over unchanged. If the pool contains all parents and no descendants, as the prefix $\tau _ { \prec v }$ of any $\tau \in \mathcal { T } ( \preceq _ { \mathcal { G } } )$ does, the restricted boundary collapses to the parents set.

Proposition 5 (Collapse to parents; 37). Let $P _ { X }$ be Markov and faithful to $\mathcal { G }$ $I f \operatorname { p a } _ { \mathcal { G } } ( v ) \subseteq Z \subseteq \operatorname { n d } _ { \mathcal { G } } ( v )$ then $B ( v , Z ) = \mathrm { p a } _ { \mathcal { G } } ( v )$

Once a topological order is known, recovering $\mathcal { G }$ thus reduces to one restricted boundary search per node.

The grow-shrink algorithm [43] computes $B ( v , Z )$ in two phases. The grow phase starts from $S = \emptyset$ and adds each $w \in Z$ with $X _ { v } ~ \subset { \ U } _ { P } ~ X _ { w } ~ \mid ~ X _ { S }$ , ending as a Markov blanket relative to $Z ;$ the shrink phase removes each w $y \in S$ with $X _ { v } ~ \bot \bot _ { P } ~ X _ { w } \mid X _ { S \setminus \{ w \} }$ , discarding redundant early inclusions. If ${ } _ { - P }$ is a compositional graphoid, the output is $B ( v , Z )$ after $O ( | Z | ^ { 2 } )$ CI queries.

Following Lam et al. [37] and the FLOP method of Wien¨obst et al. [78], we replace each CI query by a comparison of a decomposable local score $s ( v , S ; \mathcal { D } )$ , where D consists of n i.i.d. samples. A candidate w is added if $s ( v , S \cup \{ w \} ; \mathcal { D } ) > s ( v , S ; \mathcal { D } )$ and removed if $s ( v , S \setminus \{ w \} ; \mathcal { D } ) \setminus s ( v , S ; \mathcal { D } )$ . Each phase repeats its sweep until no change occurs, so the search is nongreedy and not tied to a fixed visiting order $\left( \mathrm { { A p p . B . 2 } } \right)$ If s is locally consistent, so that as $n \to \infty$ the increment is positive exactly when $X _ { v } \ nparallel X _ { P } \ X _ { w } \mid X _ { S }$ , every decision agrees in the limit with its CI counterpart, and the score-based version returns $B ( v , Z )$

As the local score we use the Bayesian information criterion [58],

$$
s ( v , S ; \mathcal { D } ) = 2 \hat { \ell } ( v , S ) - \lambda \log ( n ) \dim ( v , S ) ,\tag{2}
$$

where $\hat { \ell } ( v , S )$ is the maximized log-likelihood of $X _ { v }$ given $X _ { S } , \mathrm { d i m } ( v , S )$ its number of free parameters, and $\lambda > 0$ a penalty multiplier $( \lambda = 1$ gives the standard BIC). For curved exponential families, the BIC is decomposable, score equivalent, consistent, and locally consistent [13, 29].

## 2.4 Linear Gaussian model and interventions

Score-based methods compare graphs through the likelihood of the data, so they must assume a parametric family for $P _ { X }$ . The most common choice is the linear Gaussian model, used for instance by GES, GIES, GnIES and FLOP [13, 24, 30, 78]. Let $\varepsilon =$ $\{ 0 , 1 , \ldots , m \}$ be a finite set of environments, with $e = 0$ the observational one, and let $X ^ { ( e ) } \in \mathbb { R } ^ { d }$ be the random vector observed in environment $e ,$ with distribution $P ^ { ( e ) }$ . We assume each $X _ { v } ^ { ( e ) }$ given $X _ { \mathrm { p a } _ { \mathcal { G } } ( v ) } ^ { ( e ) }$ Gaussian with mean linear in the parents and constant variance $( \omega _ { v } ^ { ( e ) } ) ^ { 2 }$ . In matrix form,

$$
X ^ { ( e ) } = A ^ { ( e ) } X ^ { ( e ) } + \eta ^ { ( e ) } , \qquad \eta ^ { ( e ) } \sim \mathcal { N } \big ( \nu ^ { ( e ) } , \Omega ^ { ( e ) } \big ) ,\tag{3}
$$

where $A _ { v u } ^ { ( e ) } ~ \neq ~ 0$ only if $u \in \mathrm {  ~ \ p a ~ } _ { \mathcal { G } } ( v )$ and $\begin{array} { r l } { \Omega ^ { ( e ) } } & { { } = } \end{array}$ diag $\big ( ( \omega _ { 1 } ^ { ( e ) } ) ^ { 2 } , \dots , ( \omega _ { d } ^ { ( e ) } ) ^ { 2 } \big )$ . Hence $P ^ { ( e ) }$ is Gaussian, and the BIC (2) of a parent set is a least-squares regression.

Each environment $e \geq 1$ arises from a soft intervention on an unknown target set $I _ { e } \subseteq V .$ , possibly $| I _ { e } | > 1$ changing only the conditionals of the targets, so that (1) becomes

$$
p ^ { ( e ) } ( x ) = \prod _ { t \in I _ { e } } p ^ { ( e ) } \bigl ( x _ { t } \mid x _ { \mathrm { p a } _ { \mathcal G } ( t ) } \bigr ) \prod _ { v \notin I _ { e } } p ^ { ( 0 ) } \bigl ( x _ { v } \mid x _ { \mathrm { p a } _ { \mathcal G } ( v ) } \bigr ) .\tag{4}
$$

In terms of (3), soft interventions change only the rows $A _ { t , \cdot } ^ { ( e ) }$ and the pairs $( \nu _ { t } ^ { ( e ) } , \omega _ { t } ^ { ( e ) } )$ with $t \in I _ { e } ,$ , so all environments share the same DAG ${ \mathcal { G } } .$ . We write $f ^ { ( e ) } ( X _ { v } )$ for the marginal of $X _ { v }$ in environment e. For each e we observe a dataset $\mathcal { D } ^ { ( e ) }$ of $n _ { e }$ i.i.d. samples, independent across environments, with $n : = \textstyle \sum _ { e } n _ { e }$ . The intervention targets $I _ { e }$ are never observed.

## 2.5 Coarsenings

We build on the framework of Madaleno et al. [39], making precise the resolution identifiable under unknown, multi-target interventions.

Definition 6 (Coarsening; 39). A coarsening of a DAG $\mathcal { G } = ( V , E )$ is a DAG $\mathcal { G } ^ { \prime } = ( V ^ { \prime } , E ^ { \prime } )$ with a surjection $\chi : V \to V ^ { \prime }$ such that

$$
E ^ { \prime } = \{ \chi ( v )  \chi ( w ) : v  w \in E , \ \chi ( v ) \neq \chi ( w ) \} .
$$

We identify $V ^ { \prime }$ with the partition $\Pi = \{ \pi _ { 1 } , . . . , \pi _ { q } \}$ of V whose parts group the nodes sharing a common image under χ and refer to each part as a block. We call $\mathcal { G } ^ { \prime }$ a valid coarsening of $\mathcal { G }$ whenever its induced edge set is acyclic. Madaleno et al. [39] show that valid coarsenings form a lattice under refinement (Π refines $\Pi ^ { \prime }$ , written $\Pi \preceq \Pi ^ { \prime }$ , if every part of Π lies inside some part of $\Pi ^ { \prime } )$ , and that every distribution Markov to $\mathcal { G }$ is also Markov to each of its coarsenings. Further, they show that a valid coarsening can be learned by RePaRe, which walks down this lattice from {V}: a Refine step splits blocks and an $I s E d g e$ step decides edges with conditional independence (CI) tests.

For interventional data, the authors target the interventional coarsening $\mathcal { G } ^ { \mathbb { Z } }$ (partition $\Pi ^ { \hat { \tau } } )$ , merging nodes sharing the intervened ancestors $\mathrm { a n } _ { \mathcal { G } } ( v ) \cap T$ $\begin{array} { r } { T : = \bigcup _ { e > 1 } I _ { e } ; } \end{array}$ with single-target interventions (and suitable assumptions), $X _ { v }$ shifts in environment e if $v \in \mathrm { d e } g ( I _ { e } )$ , so RePaRe recovers $\mathcal { G } ^ { \mathbb { Z } }$ by grouping equal shift patterns, without knowing the targets. Sec. 3.1 extends this to multi-target interventions.

## 3 METHOD: COARSE

coarse learns the interventional coarsening in two phases. First, it reads the blocks and a causal order over them of marginal shifts alone (Sec. 3.1 and Sec. 3.2). Second, with the order fixed, it runs a single score-based local search per block (Sec. 3.3 and Sec. 3.4). All proofs are deferred to App. B.

## 3.1 The interventional coarsening

We first characterize the coarsening identifiable from interventional data under multi-target interventions.

Definition 7 (Intervention signature). The signature of a node $v \in V$ is the set of environments that shift its marginal,

$$
\sigma ( v ) : = \{ e \in \mathcal { E } \ \backslash \ \{ 0 \} : f ^ { ( e ) } ( X _ { v } ) \neq f ^ { ( 0 ) } ( X _ { v } ) \} .
$$

Definition 8 (Interventional coarsening). The interventional coarsening $\mathcal { G } ^ { \varepsilon }$ is the coarsening of $\mathcal { G }$ whose surjection satisfies $\chi ( v ) = \chi ( w ) \iff \sigma ( v ) = \sigma ( w )$ We denote its partition by $\Pi ^ { \varepsilon }$ , and call a block’s common signature its support, supp(π).

In words, $\mathcal { G } ^ { \varepsilon }$ merges the variables the available interventions cannot tell apart; $\Pi ^ { \varepsilon }$ is identifiable by construction, since $\sigma$ depends only on marginals. Def. 8 extends the interventional coarsening $\mathcal { G } ^ { \breve { \tau } }$ of Madaleno et al. [39], and the two coincide under single-target interventions. With multi-target interventions, a shift only signals that some target lies upstream of $v ,$ so $\Pi ^ { \boldsymbol { \tau } }$ is no longer identifiable from marginal shifts. In general, $\Pi ^ { \mathcal { Z } } \preceq \Pi ^ { \mathcal { E } } \ ( \mathrm { A p p . \ B . 3 . 2 } )$

Our results rely on the following assumptions.

Assumption 9 (Identifiability assumptions; 39). For the fine-grained DAG $\mathcal { G } ~ = ~ ( V , E )$ over the random variables indexed by V, and the environments $\mathcal { E }$ with targets $\{ I _ { e } \} _ { e \in \mathcal { E } } , ~ I _ { 0 } : = \varnothing$ , the distributions $\{ f ^ { ( e ) } \} _ { e \in \mathcal { E } }$ satisfy:

(i) Coarse Markov: for every $\textit { e } \in \textit { \mathcal { E } }$ the joint law factorizes according to $\mathcal { G } ^ { \varepsilon }$

$$
f ^ { ( e ) } ( X _ { V } ) = \prod _ { { \boldsymbol u } \in \chi ( V ) \setminus \chi ( I _ { e } ) } f ^ { ( 0 ) } \big ( X _ { \boldsymbol u } \mid X _ { \mathrm { p a } ( { \boldsymbol u } ) } \big )\tag{5}
$$

where $X _ { u } : = X _ { \chi ^ { - 1 } \left( u \right) } , X _ { \mathrm { p a } \left( u \right) } : = X _ { \chi ^ { - 1 } \left( \mathrm { p a } _ { \mathcal { G } } \varepsilon \left( u \right) \right) } ;$

(ii) Coarse faithfulness: the only conditional independences among coarse nodes are those implied by the Markov factorization above;

(iii) Interventional soundness: every intervention induces a shift in the marginal distribution of its targets and each downstream node with respect

to the observational one. That is, $v \in \mathrm { d e } _ { \mathcal { G } } ( I _ { e } ) \Rightarrow$ $f ^ { ( e ) } ( X _ { v } ) \neq f ^ { ( 0 ) } ( X _ { v } )$ for all $e \in \mathcal { E } \setminus \{ 0 \}$ and $v \in V$

Every (and only) descendant of an intervention target records a shift, so signatures only grow along directed edges:

Lemma 10 (Signature monotonicity). Under Ass. $^ { g } ,$ $i f u  w$ in G, then $\sigma ( u ) \subseteq \sigma ( w )$

Proposition 11 (Validity). Under Ass. 9 the graph $\mathcal { G } ^ { \varepsilon }$ , induced by $\chi ,$ is a valid coarsening of G.

## 3.2 A causal order from the interventions

By Lem. 10, comparing supports therefore orders the blocks consistently with causation.

Definition 12 (Support order and parent pool). The support order is the strict partial order on $\Pi ^ { \varepsilon }$ given by $\pi _ { a } < _ { \mathrm { s u p p } } \pi _ { b } \Longleftrightarrow \mathrm { s u p p } ( \pi _ { a } ) \subsetneq \mathrm { s u p p } ( \pi _ { b } )$ , and $\tau ( < _ { \mathrm { s u p p } } )$ denotes the set of its linear extensions. The parent pool of a block π is $\operatorname { p a } ^ { \star } ( \pi ) : = \{ \pi ^ { \prime } \in \Pi ^ { \mathcal { E } } : \pi ^ { \prime } < _ { \operatorname { s u p p } } \pi \}$ distinct blocks have distinct supports, so two blocks are either $< _ { \mathrm { s u p p } }$ -ordered or incomparable.

Theorem 13 (Supports encode the causal order). Under Ass. 9, for distinct blocks $\pi _ { a } , \pi _ { b } \in \Pi ^ { \varepsilon }$

(i) Ancestry: $i f \pi _ { a } \prec _ { \mathcal { G } ^ { \varepsilon } } \pi _ { b } , t h e n \pi _ { a } \prec _ { \mathrm { s u p p } } \pi _ { b } ;$

(ii) Parent pool: $\mathrm { p a } _ { \mathcal { G } ^ { \varepsilon } } ( \pi _ { a } ) \subset \mathrm { p a } ^ { \star } ( \pi _ { a } ) \subset \mathrm { n d } _ { \mathcal { G } ^ { \varepsilon } } ( \pi _ { a } ) ,$ in particular, blocks with incomparable supports are non-adjacent and neither is an ancestor of the other;

(iii) Orientation: $i f \pi _ { a }$ and π<sub>b</sub> are adjacent in $\mathcal { G } ^ { \varepsilon }$ , then $\pi _ { a }  \pi _ { b } \in \mathcal G ^ { \mathcal E }$ if and only $i f \pi _ { a } < _ { \mathrm { s u p p } } \pi _ { b }$

Hence $\mathcal { G } ^ { \varepsilon }$ is determined by its skeleton sk $( \mathcal G ^ { \varepsilon } )$ and the supports.

The converse of (i) fails: causally unrelated blocks can have nested supports (Ex. 30). Supports cannot decide adjacency, which is left to the score.

Corollary 14 (No search over orders). Under Ass. 9:

(i) every $\tau \in \mathcal { T } ( < _ { \mathrm { s u p p } } )$ is a topological order of $\mathcal { G } ^ { \varepsilon }$ and $\mathrm { p a } ^ { \star } ( \pi ) \subseteq \tau _ { \prec \pi } f o r$ every block π;

(ii) for every block π and environment e, the Markov boundary of $X _ { \pi }$ relative to $\mathrm { p a } ^ { \star } ( \pi )$ under $P ^ { ( e ) }$ is $\operatorname { p a } _ { g } \varepsilon \left( \pi \right)$

Order-based methods search over causal orders and, for each order, over parent sets. Here the interventions fix the order once and for all, and learning $\mathcal { G } ^ { \varepsilon }$ reduces to one restricted Markov boundary search per block over a pool containing all parents and no descendants; any parent sets chosen within the pools form a DAG.

## 3.3 Scoring coarse parents

Without loss of generality, we assume centered data. For each block write $X _ { j }$ to denote $X _ { \pi _ { j } } : = ( X _ { v } ) _ { v \in \pi _ { j } } \in$ R<sup>rj</sup> with $r _ { j } : = | \pi _ { j } |$ , and let $X _ { \mathrm { P a } _ { j } } \in \mathbb { R } ^ { s _ { j } }$ stack its parent blocks, with $\begin{array} { r } { s _ { j } : = \sum _ { \pi _ { k } \in \mathrm { P a } _ { j } } r _ { k } } \end{array}$

Definition 15 (Coarse block-Gaussian model). In environment e,

$$
X _ { j } ^ { ( e ) } = B _ { j } ^ { ( e ) } X _ { \mathrm { { P a } } _ { j } } ^ { ( e ) } + \varepsilon _ { j } ^ { ( e ) } , \qquad \varepsilon _ { j } ^ { ( e ) } \sim \mathcal { N } \big ( 0 , \Sigma _ { j } ^ { ( e ) } \big ) ,
$$

for $j = 1 , \dotsc , q .$ , with unrestricted block-regression matrix $B _ { i } ^ { ( e ) } \in \mathbb { R } ^ { r _ { j } \times s _ { j } }$ , unrestricted $\Sigma _ { j } ^ { ( e ) } \in \mathbb { R } ^ { r _ { j } \times r _ { j } } \ \succ \ 0 .$ and mutually independent block residuals (i.e. we assume causal suficiency at the coarse level).

Theorem 16 (Curved exponential family). For each $e \in { \mathcal { E } }$ and each DAG H on $\Pi ^ { \varepsilon }$ , the model of Def. 15 on H is a curved exponential family of dimension

$$
\dim ( { \mathcal { H } } ) = \sum _ { j = 1 } ^ { q } d _ { j } , \qquad d _ { j } : = r _ { j } s _ { j } + { \frac { r _ { j } ( r _ { j } + 1 ) } { 2 } } .
$$

The likelihood factorizes over blocks, so the MLE is one least-squares regression per block. We can therefore define the score as

Definition 17 (Pooled BIC). Let H be a coarse DAG on $\Pi ^ { \varepsilon }$ with parent sets $\mathrm { P a } _ { j }$ , and $\mathcal { D } = \{ \mathcal { D } ^ { ( e ) } \} _ { e \in \mathcal { E } }$ . For each environment, the block score is the BIC (2) of the least-squares regression of $X _ { j }$ on $X _ { \mathrm { P a } _ { j } }$ fitted on $\mathcal { D } ^ { ( e ) }$

$$
\mathrm { B I C } _ { j } ^ { \left( e \right) } ( \pi _ { j } , \mathrm { P a } _ { j } ) : = - n _ { e } \log \left| \widehat { \Sigma } _ { j } ^ { \left( e \right) } \right| - \lambda \log ( n _ { e } ) d _ { j } ,
$$

where $\widehat { \Sigma } _ { j } ^ { ( e ) }$ is the residual sample covariance and $d _ { j }$ is as in Thm. 16. The pooled BIC is

$$
\begin{array} { r l } & { \displaystyle \mathrm { B I C } ( \mathcal { H } , \mathcal { D } ) : = \sum _ { j = 1 } ^ { q } \mathrm { B I C } _ { j } ( \pi _ { j } , \mathrm { P a } _ { j } ) , } \\ & { \displaystyle \mathrm { B I C } _ { j } ( \pi _ { j } , \mathrm { P a } _ { j } ) : = \sum _ { e \in \mathcal { E } } \mathrm { B I C } _ { j } ^ { ( e ) } ( \pi _ { j } , \mathrm { P a } _ { j } ) . } \end{array}\tag{6}
$$

Up to a constant, BIC(H, D) equals $\begin{array} { r } { \sum _ { e } \big ( 2 \hat { \ell } ^ { ( e ) } ( \mathcal { H } ) - } \end{array}$ $\lambda \log ( n _ { e } ) \operatorname { d i m } ( \mathcal { H } ) )$ , the sum of the per-environment BICs. It is score equivalent and, by construction, decomposable, with $\mathrm { B I C } _ { j }$ depending on H only through $( \pi _ { j } , \mathrm { P a } _ { j } )$ ; for the derivations see App. B.5. We write $\pi _ { j } \perp \perp _ { \varepsilon } \pi _ { k } \mid S$ when $X _ { \pi _ { j } } \perp \perp X _ { \pi _ { k } } \mid X _ { S }$ holds in every $P ^ { ( e ) }$ and $X _ { S }$ collects the variables of the blocks in S. Asymptotics are taken with $n _ { e } = \gamma _ { e } n , \gamma _ { e } > 0$ fixed, and $n \to \infty$

Proposition 18 (Local consistency). For a block $\pi _ { j }$ a set of blocks S and a block $\pi _ { k } \notin S \cup \{ \pi _ { j } \}$

$$
\begin{array} { r } { \mathrm { B I C } _ { j } ( S \cup \{ \pi _ { k } \} ) - \mathrm { B I C } _ { j } ( S ) \xrightarrow { p } \left\{ \begin{array} { l l } { + \infty } & { i f \pi _ { j } \rlap / \mathcal { X } _ { \varepsilon } \pi _ { k } \mid S , } \\ { - \infty } & { i f \pi _ { j } \rlap / \mathcal { X } _ { \varepsilon } \pi _ { k } \mid S . } \end{array} \right. } \end{array}
$$

Intuitively, a single dependent environment contributes an order-n gain that beats every order-log n penalty; with no dependent environment, only the penalties remain.

## 3.4 COARSE and its guarantees

We now turn to estimation. Following Madaleno et al. [39], we collect the signatures in the intervention $d e \mathrm { - }$ scendant matrix $M \in \{ 0 , 1 \} ^ { d \times m }$ , with $M _ { v , e } = \mathbb { 1 } [ e \in$ $\sigma ( v ) ]$ , and define the blocks of $\Pi ^ { \varepsilon }$ as the classes of equal rows of M. Each entry tests whether the marginal of $X _ { v }$ in environment $e \neq 0$ difers from the observational one. Under Gaussianity we use a Welch t-test [77] at level α:

$$
M _ { v , e } = \mathbb { 1 } \big [ p _ { v , e } < \alpha \big ] , \qquad p _ { v , e } : = \mathrm { W e l c h } _ { t } \big ( X _ { v } ^ { ( e ) } , X _ { v } ^ { ( 0 ) } \big ) .
$$

The Welch t-test only detects mean shifts. An intervention that changes only higher-order moments of $X _ { v } \mathrm { ( e . g . , }$ , its variance) gives $M _ { v , e } = 0$ even though $f ^ { ( e ) } ( \dot { X _ { v } } ) \neq f ^ { ( 0 ) } ( X _ { v } )$ . Ass. 9, however, allows any distributional change. We keep the Welch t-test as a cheap default for the Gaussian linear setting, but it can be swapped for a general two-sample test $( \mathrm { e . g . } $ Kolmogorov-Smirnov, kernel-based [26] or energy distance [65] tests) with no other change to the algorithm. Given $M , { \mathrm { A l g . ~ 3 } }$ refines the partition greedily, moving down the meet-subsemilattice of valid coarsenings to the partition identified by M. Once the partition is returned, we compute the estimated support of each block and its parent pool $\widehat { \mathrm { p a } } ^ { \star } ( \pi )$ as in Def. 12. For each block we then run a single GrowShrink call (Alg. 1) over its parent pool, estimating its coarse parent set $\widehat { \mathrm { P a } } _ { j }$ . The procedure returns the coarse DAG $\hat { \mathcal { G } } = \left( \hat { \Pi } , \{ \pi _ { k } \to \pi _ { j } : \pi _ { k } \in \widehat { \mathrm { P a } } _ { j } \} \right) ( \mathrm { A l g . ~ 4 } )$

As is standard for constraint-based and hybrid methods, including RePaRe [39], we treat the shift tests as an oracle: $\hat { M } _ { v , e } = M _ { v , e }$ for all v and $e .$ Since Π,<sup>ˆ</sup> the supports, and the pools are deterministic functions of M<sup>ˆ</sup> , they then coincide respectively with $\Pi ^ { \varepsilon }$ , supp, and $\mathrm { p a } ^ { \star } ( \cdot )$ , and the order of Sec. 3.2 is exact. What remains is to prove consistency of the score.

Theorem 19 (Consistency of coarse). Under Ass. 9, Def. 15 and oracle tests, as $n \to \infty$ the output of Alg. 4 satisfies $\mathbb { P } ( \hat { \mathcal { G } } = \mathcal { G } ^ { \varepsilon } )  1$ , and it does not depend on the order in which blocks are processed.

To be precise, being score equivalent, the pooled BIC alone identifies $\mathcal { G } ^ { \varepsilon }$ only up to $[ \mathcal { G } ^ { \varepsilon } ]$ , whose members share the skeleton. The heterogeneity of the data enters only through the supports, which fix every orientation, so no interventional equivalence class or interventional score is needed.

Theorem 20 (Worst-case complexity). With the Welch test and $n _ { 0 } = \mathcal { O } ( n / | \mathcal { E } | )$ , coarse runs in time $\mathcal { O } \left( d ^ { 2 } n + | \mathcal { E } | q ^ { 3 } d ^ { 3 } \right)$ , where d is the number of finegrained nodes, $n = \textstyle \sum _ { e } n _ { e }$ the total sample size and $q = | { \widehat { \Pi } } ^ { \varepsilon } | \leq d$ the number of estimated blocks.

Thm. 19 assumes oracle shift tests and enough samples for every block MLE to exist, i.e. $n _ { e } \ge r _ { j } + s _ { j }$ $\left( \mathrm { A p p . ~ B . 4 } \right)$ In finite samples a fixed α over dm tests can split or merge blocks, and large blocks inflate the model parameters, making estimation complex and data hungry. Three variants address these issues (App. C.2; used in Sec. 4): CV-coarse selects α by cross-validation, kPC-coarse compresses blocks to their leading principal components, and avg-coarse summarizes blocks by their standardized mean.

## 4 EXPERIMENTS

We validate the results of Sec. 3 on synthetic and real data: convergence of the recovered partition and edges, scaling in the number of variables (Sec. 4.2), edge recovery and runtime against RePaRe [39] (Sec. 4.3), and the light-tunnel causal chamber [23] (Sec. 4.4). Additional experiments on sensitivity and robustness to assumption violations, together with theoretical results for multi-target interventions and high-dimensional regimes, are reported in App. C.

## 4.1 Setup

We sample linear Gaussian models (3) with sempler [24]. Ground-truth DAGs are Erd˝os–R´enyi (ER) graphs with edge density $\rho .$ Edge weights have magnitude uniform in [0.5, 2] and random sign, and each model is standardized to unit observational variance [50], to avoid variance artifacts that would invalidate the results [55]. Each of the m interventional environments is a soft shift intervention on unknown targets drawn without replacement, adding $\mathcal { N } ( 2 , 1 )$ noise to each targeted variable. All environments have the same sample size $n _ { e } ,$ and every configuration is repeated over 20 seeds. Unless stated otherwise, coarse builds M<sup>ˆ</sup> with a Welch t-test at level $\alpha = 1 0 ^ { - 4 }$ and uses $\lambda = 1 ;$ full details in App. C.1.

Partition recovery is measured by the adjusted Rand index (ARI) between Π and Π <sup>ˆ</sup> <sup>E</sup> [76] (1 for a perfect match, 0 at chance), edge recovery by precision, recall and $F _ { 1 }$ -score on the coarse DAG, and runtime as the wall-clock time of one fit on one CPU core.

## 4.2 Consistency and scalability

Fig. 1 reports runs of the full Alg. 4 for d from 10 to 1000 and $n _ { e }$ from $1 0 ^ { 3 }$ to $1 0 ^ { 5 }$ . At every d, both ARI and F-score increase with $n _ { e } ,$ confirming Thm. 19. For small graphs $( \approx d \leq 2 0 )$ , both the partition and the coarse edges are recovered near-perfectly once $n _ { e } \geq$ $1 0 ^ { 4 }$ . Larger graphs need more data: at $n _ { e } = 1 0 ^ { 5 }$ the F-score varies from 0.95 for d = 50 to 0.79 for d = 200 and 0.67 for $d = 1 0 0 0$ , while ARI drops from 0.93 at $d = 5 0$ to 0.54 at d = 1000. Nonetheless, under the oracle partition, Tab. 1 shows near-perfect F-scores up to d = 500 once $n _ { e } \ge 5 \times 1 0 ^ { 3 }$ , suggesting the loss in performance stems from the partition estimation stage: M<sup>ˆ</sup> rests on d · m Welch tests, and one missed or spurious shift moves a variable to another block, so partition errors grow with d. Runtime instead grows far below the worst-case bound of Thm. 20 in both $n _ { e }$ and d: about 10 seconds for $d = 1 0 0 0$ with 10<sup>5</sup> samples, and about one second for d = 200.

![](images/46788f8c2de5cfbd931bc40954f47045f6a439fe87d3c1e93ff069a2548301dc.jpg)  
(a) partition recovery

![](images/1a0d16d00081606d08742ab0f498acb1f357edc1c9c039c2aa8d712e094f1e50.jpg)  
(b) coarse-edge recovery

![](images/5fc49369d8154b916ce492aa8fea55fa8ecd52a8ca50ee49f5129b56b86ce42c.jpg)  
(c) runtime in $n _ { e }$

![](images/31502ee59178af48d54a01dbf00d62aa8f9c2c676c78f3ee44f516a2d38067b4.jpg)  
(d) runtime in d  
Figure 1: Full coarse pipeline on ER graphs, $\rho = 0 . 2 , m = 5$ . Both recovery metrics improve with $n _ { e }$ at every $d .$

## 4.3 Comparison against RePaRe

RePaRe keeps a coarse edge only if a CI test rejects in every environment, while coarse adds a parent block as soon as the pooled BIC improves. Tab. 1 reports edge recovery under the oracle partition—a stage common to both methods—on ER graphs with $\rho = 0 . 2 ,$ m = 5 and d up to 500 (full grid in Tab. 3). With small samples, RePaRe’s many CI tests lack power, and a single environment that fails to reject is enough to miss an edge. The coarse score instead sufers on large graphs with few interventions and small samples, where blocks of tens of variables make the model parameters hard to estimate. From $n _ { e } = 5 \times 1 0 ^ { 3 }$ on, coarse reaches an F-score of at least 0.99 at every $d ,$ with speed-ups from 9× to 510×.

## 4.4 Real data: the causal chamber

We use measurements from the light-tunnel causal chamber [23], whose causal graph is known by construction: $d \ = \ 2 0$ actuator and sensor variables, one observational dataset and five interventional ones, each perturbing one light intensity (R, G, B) or one polarizer angle. Besides RePaRe, we run three finegrained baselines that also use interventional data to estimate the true graph: GIES [30] (with the intervention target assumed known), GnIES [24] and UT-IGSP [62] (both of which estimate the intervention targets). For each method with hyperparameters, we compare selected values against oracle ones: coarse selects by cross-validation, UT-IGSP by BIC, and RePaRe uses the GnIES score, which needs the intervention targets to pool data across environments.

coarse and RePaRe recover the true partition (ARI = 1) and the same coarse DAG, with 8 of the 10 true coarse edges and no false edge (Tab. 2). Since the fine grained methods learn a diferent object, we score all methods at both resolutions: a coarse edge expands to all variable pairs it connects, and a fine-grained graph collapses onto the true partition. Each directed F-score is paired with a negative control [53] measuring whether a method improves over chance. coarse and RePaRe score highest at both levels and above their null intervals. GIES beats chance at both levels while UT-IGSP only at the variable level. GnIES recovers directions no better than chance, as reported by Gamella et al. [24]: interventions in this dataset mostly shift the noise mean, which GnIES removes by centering the data.

## 5 DISCUSSION

We introduced coarse, the first score-based method for learning a coarsened causal model from heterogeneous data. Intervention signatures fix the partition and a causal order over the blocks, and a BIC pooled over environments drives one local search per block. The target, the interventional coarsening $\mathcal { G } ^ { \bar { \varepsilon } }$ , extends that of Madaleno et al. [39] to unknown, multi-target interventions and is identifiable from marginal shifts alone. The procedure is consistent and runs in polynomial time; it matches RePaRe on edge recovery given enough samples and runs faster in every configuration, by up to 510× under the oracle partition, and on the light-tunnel chamber it recovers the block structure of the physical system in a fraction of a second.

The design carries four limitations. First, the pipeline is hybrid and sequential: the partition rests on d · m

Table 1: Edge recovery under the oracle partition, ER graphs, $\rho = 0 . 2 , m = 5 .$ . F-score as mean ± sample standard deviation over 20 seeds; speed-up is the ratio of mean RePaRe to mean coarse runtime. Bold marks the better method where the two difer at the reported precision.
<table><tr><td>d</td><td>10</td><td>20</td><td>50</td><td>100</td><td>200</td><td>300</td><td>500</td></tr><tr><td colspan="8">F-score ↑,  $n _ { e } = 1 0 ^ { 3 }$ </td></tr><tr><td>COARSE</td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> ${ \bf 1 . 0 0 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 9 9 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 9 8 \pm 0 . 0 2 }$ </td><td> $0 . 9 5 \pm 0 . 0 5$ </td><td> $0 . 8 4 \pm 0 . 0 8$ </td><td> $0 . 7 1 \pm 0 . 0 5$ </td></tr><tr><td>RePaRe</td><td> $1 . 0 0 \pm 0 . 0 1$ </td><td> $0 . 9 6 \pm 0 . 0 3$ </td><td> $0 . 9 5 \pm 0 . 0 5$ </td><td> $0 . 9 3 \pm 0 . 0 7$ </td><td> $0 . 9 5 \pm 0 . 0 8$ </td><td> ${ \bf 0 . 9 5 \pm 0 . 1 0 }$ </td><td> $\mathbf { 0 . 9 7 \pm 0 . 0 4 }$ </td></tr><tr><td colspan="8">F-score ↑,  $n _ { e } = 1 0 ^ { 5 }$ </td></tr><tr><td>COARSE</td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $\mathbf { 1 . 0 0 \ : \pm { \ : 0 . 0 0 } }$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>RePaRe</td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 9 9 \pm 0 . 0 1$ </td><td> $1 . 0 0 \pm 0 . 0 1$ </td></tr><tr><td colspan="8">Speed-up ↑ (RePaRe runtime / coarse runtime)</td></tr><tr><td colspan="8"> $n _ { e } = 1 0 ^ { 3 }$ </td></tr><tr><td></td><td>9×</td><td>21×</td><td>38×</td><td>44×</td><td>37×</td><td>37×</td><td>32×</td></tr><tr><td> $n _ { e } = 1 0 ^ { 5 }$ </td><td>51×</td><td>172×</td><td>342×</td><td>510×</td><td>509×</td><td>424×</td><td>329×</td></tr></table>

Table 2: Light tunnel, ungrouped mode. Directed F-score at the variable and block level, each against its negative control (mean and 95% interval over 1000 random DAGs with the same number of edges). Fit and Search are wall-clock seconds of the final fit and hyperparameter search.
<table><tr><td></td><td></td><td colspan="3">Variable level</td><td colspan="3">Block level</td><td></td><td></td></tr><tr><td>Method</td><td>ARI</td><td>F1 ↑</td><td></td><td>Null</td><td>F1↑</td><td></td><td>Null</td><td>Fit(s)</td><td>Search(s)</td></tr><tr><td>COARSE</td><td>1.00</td><td>0.74</td><td>0.13</td><td>[0.00, 0.38]</td><td>0.89</td><td></td><td>0.17 [0.00, 0.44]</td><td>0.02</td><td>0.40</td></tr><tr><td>RePaRe</td><td>1.00</td><td>0.74</td><td>0.12</td><td>[0.00, 0.35]</td><td>0.89</td><td>0.16</td><td>[0.00, 0.44]</td><td>0.10</td><td>1.54</td></tr><tr><td>GIES</td><td></td><td>0.62</td><td>0.10</td><td>[0.02, 0.21]</td><td>0.80</td><td>0.27</td><td>[0.12, 0.45]</td><td>1.83</td><td></td></tr><tr><td>GnIES</td><td></td><td>0.02</td><td>0.12</td><td>[0.02, 0.23]</td><td>0.07</td><td>0.27</td><td>[0.11, 0.46]</td><td>376</td><td></td></tr><tr><td>UT-IGSP</td><td></td><td>0.44</td><td>0.12</td><td>[0.02, 0.23]</td><td>0.39</td><td></td><td>0.28 [0.11, 0.45]</td><td>0.19</td><td>2.69</td></tr></table>

Welch tests, with their multiple-testing burden and limited power at small samples, and a misread signature moves a variable between blocks before any edge is scored; at large d the loss comes mostly from this stage (Fig. 1). These tests also detect only mean shifts, so with them the oracle-test assumption of Thm. 19 requires every shift in Ass. 9(iii) to move the mean. Second, the pooled BIC is liberal by construction: a coarse edge survives on the evidence of a single environment (Sec. 3.3), whereas the all-environments rule of RePaRe errs conservative (Sec. 4.3). The same score can also err conservative through its penalties, since adding a parent block costs r<sub>j</sub>r<sub>k</sub> parameters per environment and large blocks observed in few environments need many samples before their true edges appear. Third, the resolution of $\mathcal { G } ^ { \varepsilon }$ is bounded by the available interventions: multi-target environments can merge variables that single-target ones would separate. Fourth, the guarantees rest on acyclic fine graphs, a linear Gaussian model without hidden confounders, soft interventions, and an observational baseline: hard interventions would have each environment score a mutilated coarse graph, which we conjecture breaks the guarantee, and without a baseline neither the partition nor the order is defined.

Each limitation suggests future work. Signatures could be read from non-parametric or description-length criteria rather than tests [33, 41, 44], or the partition and the coarse edges estimated jointly under a single objective [49]. A target-aware score in the spirit of GnIES [24] that does not know the targets could pool environments more sample-eficiently, and resampling with stability selection [47] would attach confidence to each coarse edge. Interventions themselves could be chosen to refine the coarsening where it matters most. Further out, natural next steps include cyclic fine graphs, already identifiable in the linear non-Gaussian setting [40], baseline-free regimes, perhaps via context variables [48], and nonlinear or non-Gaussian scores.

The systems that motivate causal abstraction— high-dimensional measurements of cells, brains, and images—face two obstacles at once: the right macrovariables are unknown, and the fine DAG is out of reach. When heterogeneity is available, coarse addresses both: it reads the grouping and the order of the shifts, leaving the data only a small, polynomial skeleton search.

## References

[1] Tara V. Anand, Adele H. Ribeiro, Jin Tian, and Elias Bareinboim. Causal Efect Identification in Cluster DAGs. Proceedings of the AAAI Conference on Artificial Intelligence, 37(10):12172–12179, June 2023. ISSN 2374- 3468, 2159-5399. doi: 10.1609/aaai.v37i10.26435. URL https://ojs.aaai.org/index.php/AAAI/ article/view/26435.

[2] Steen A Andersson and Michael D Perlman. Normal Linear Regression Models With Recursive Graphical Markov Structure. Journal of Multivariate Analysis, 66(2):133–187, August 1998. ISSN 0047259X. doi: 10.1006/jmva.1998. 1745. URL https://linkinghub.elsevier. com/retrieve/pii/S0047259X98917456.

[3] Steen A. Andersson, David Madigan, and Michael D. Perlman. Alternative Markov Properties for Chain Graphs. Scandinavian Journal of Statistics, 28(1):33–85, 2001. ISSN 0303- 6898. URL https://www.jstor.org/stable/ 4616644.

[4] Bryan Andrews, Joseph Ramsey, Rub´en S´anchez-Romero, Jazmin Camchong, and Erich Kummerfeld. Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow-Shrink Trees. Advances in neural information processing systems, 36:63945–63956, December 2023. ISSN 1049-5258. URL https://pmc. ncbi.nlm.nih.gov/articles/PMC11393735/.

[5] Sander Beckers and Joseph Y. Halpern. Abstracting Causal Models. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):2678– 2685, July 2019. ISSN 2374-3468. doi: 10.1609/ aaai.v33i01.33012678. URL https://ojs.aaai. org/index.php/AAAI/article/view/4117.

[6] Sander Beckers, Frederick Eberhardt, and Joseph Y. Halpern. Approximate Causal Abstraction, June 2019. URL https://arxiv.org/abs/ 1906.11583v2.

[7] Yoav Benjamini and Yosef Hochberg. Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. Journal of the Royal Statistical Society Series B: Statistical Methodology, 57(1):289–300, January 1995. ISSN 1369-7412, 1467-9868. doi: 10.1111/j.2517-6161. 1995.tb02031.x. URL https://academic.oup. com/jrsssb/article/57/1/289/7035855.

[8] Yoav Benjamini and Daniel Yekutieli. The control of the false discovery rate in multiple testing under dependency. The Annals of Statistics, 29 (4):1165–1188, August 2001. doi: 10.1214/aos/ 1013699998.

[9] Simon Bing, Jonas Wahl, and Jakob Runge. Structural Causal Bottleneck Models, March 2026. URL http://arxiv.org/abs/2603.08682. arXiv:2603.08682 [stat.ML].

[10] Nathalie Caspard and Bernard Monjardet. The lattices of closure systems, closure operators, and implicational systems on a finite set: a survey. Discrete Applied Mathematics, 127(2):241–269, 2003. doi: 10.1016/S0166-218X(02)00209-3.

[11] Krzysztof Chalupka, Pietro Perona, and Frederick Eberhardt. Visual Causal Feature Learning, December 2014. URL https://arxiv.org/abs/ 1412.2309v2.

[12] Liuting Chen and Alex Markham. I-FLOP: Fast Learning of Order and Parents from Interventional Data, August 2026. URL http:// arxiv.org/abs/2608.28245. arXiv:2608.28245 [stat.ML].

[13] David Maxwell Chickering. Optimal structure identification with greedy search. J. Mach. Learn. Res., 3(null):507–554, March 2003. ISSN 1532-4435. doi: 10.1162/ 153244303321897717. URL https://dl.acm. org/doi/10.1162/153244303321897717.

[14] David Maxwell Chickering, Christopher Meek, and David Heckerman. Large-Sample Learning of Bayesian Networks is NP-Hard, October 2012. URL http://arxiv.org/abs/1212.2468. arXiv:1212.2468 [cs].

[15] Gabriele D’Acunto, Fabio Massimo Zennaro, Yorgos Felekis, and Paolo Di Lorenzo. Causal Abstraction Learning based on the Semantic Embedding Principle, May 2025. URL http://arxiv. org/abs/2502.00407. arXiv:2502.00407 [cs.LG].

[16] Giovanni De Felice, Arianna Casanova Flores, Francesco De Santis, Silvia Santini, Johannes Schneider, Pietro Barbiero, and Alberto Termine. Causally reliable concept bottleneck models. Advances in Neural Information Processing Systems, 38:149099–149139, 2026. URL https: //openreview.net/forum?id=UX143QGvb8.

[17] Mathias Drton and Michael Eichler. Maximum Likelihood Estimation in Gaussian Chain Graph Models under the Alternative Markov Property. Scandinavian Journal of Statistics, 33 (2):247–257, June 2006. ISSN 0303-6898, 1467- 9469. doi: 10.1111/j.1467-9469.2006.00482.x. URL https://onlinelibrary.wiley.com/doi/ 10.1111/j.1467-9469.2006.00482.x.

[18] Olive Jean Dunn. Multiple Comparisons among Means. Journal of the American Statistical Association, 56(293):52–64, March 1961. ISSN 0162- 1459. doi: 10.1080/01621459.1961.10482090.

URL https://www.tandfonline.com/doi/abs/ 10.1080/01621459.1961.10482090.

[19] Frederick Eberhardt. Almost Optimal Intervention Sets for Causal Discovery, June 2012. URL https://resolver.caltech.edu/ CaltechAUTHORS:20190327-085859738.

[20] Doris Entner and Patrik O. Hoyer. Estimating a Causal Order among Groups of Variables in Linear Models, July 2012. URL http://arxiv.org/ abs/1207.1977. arXiv:1207.1977 [stat.ML].

[21] Yorgos Felekis, Theodoros Damoulas, and Paris Giampouras. Distributionally Robust Causal Abstractions, 2025. URL https://arxiv.org/abs/ 2510.04842. Version Number: 3.

[22] Juan L. Gamella and Christina Heinze-Deml. Active Invariant Causal Prediction: Experiment Selection through Stability. In Advances in Neural Information Processing Systems, volume 33, pages 15464–15475. Curran Associates, Inc., 2020. URL https:// proceedings.neurips.cc/paper/2020/hash/ b197ffdef2ddc3308584dce7afa3661b-Abstract html.

[23] Juan L. Gamella, Jonas Peters, and Peter B¨uhlmann. Causal chambers as a real-world physical testbed for AI methodology. Nature Machine Intelligence, 7(1):107–118, January 2025. ISSN 2522-5839. doi: 10.1038/ s42256-024-00964-x. URL https://www.nature. com/articles/s42256-024-00964-x.

[24] Juan L. Gamella, Armeen Taeb, Christina Heinze-Deml, and Peter B¨uhlmann. Characterization and Greedy Learning of Gaussian Structural Causal Models under Unknown Interventions, March 2025. URL http://arxiv.org/ abs/2211.14897. arXiv:2211.14897 [stat].

[25] AmirEmad Ghassami, Saber Salehkaleybar, and Negar Kiyavash. Optimal Experiment Design for Causal Discovery from Fixed Number of Experiments, February 2017. URL https://arxiv. org/pdf/1702.08567.

[26] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Sch¨olkopf, and Alexander Smola. A Kernel Two-Sample Test. Journal of Machine Learning Research, 13(25):723–773, 2012. ISSN 1533-7928. URL http://jmlr.org/ papers/v13/gretton12a.html.

[27] Moritz Grosse-Wentrup, Akshey Kumar, Anja Meunier, and Manuel Zimmer. Neuro-cognitive multilevel causal modeling: A framework that bridges the explanatory gap between neuronal activity and cognition. PLOS Computational Biology, 20(12):e1012674, December 2024. ISSN 1553-

7358. doi: 10.1371/journal.pcbi.1012674. URL https://journals.plos.org/ploscompbiol/ article?id=10.1371/journal.pcbi.1012674.

[28] Jiaying Gu and Qing Zhou. Learning big gaussian bayesian networks: Partition, estimation and fusion. Journal of Machine Learning Research, 21(158):1–31, 2020. URL http://jmlr. org/papers/v21/19-318.html.

[29] Dominique M. A. Haughton. On the Choice of a Model to Fit Data from an Exponential Family. The Annals of Statistics, 16(1):342–355, 1988. ISSN 0090-5364. URL https://www.jstor.org/ stable/2241441.

[30] Alain Hauser and Peter B¨uhlmann. Characterization and Greedy Learning of Interventional Markov Equivalence Classes of Directed Acyclic Graphs, April 2011. URL https://arxiv.org/ abs/1104.2808v2.

[31] Alain Hauser and Peter B¨uhlmann. Two optimal strategies for active learning of causal models from interventional data. International Journal of Approximate Reasoning, 55(4):926–939, June 2014. ISSN 0888613X. doi: 10.1016/j.ijar.2013. 11.007. URL https://linkinghub.elsevier. com/retrieve/pii/S0888613X13002879.

[32] Antti Hyttinen, Frederick Eberhardt, and Patrik O. Hoyer. Experiment Selection for Causal Discovery. Journal of Machine Learning Research, 14(93):3041–3071, 2013. ISSN 1533- 7928. URL http://jmlr.org/papers/v14/ hyttinen13a.html.

[33] Dominik Janzing and Bernhard Schoelkopf. Causal inference using the algorithmic Markov condition, April 2008. URL http://arxiv.org/ abs/0804.3678. arXiv:0804.3678 [math.ST].

[34] Yoshinobu Kawahara, Kenneth Bollen, Shohei Shimizu, and Takashi Washio. GroupLiNGAM: Linear non-Gaussian acyclic models for sets of variables, June 2010. URL http://arxiv.org/ abs/1006.5041. arXiv:1006.5041 [cs.AI].

[35] Neville Kenneth Kitson, Anthony C. Constantinou, Zhigao Guo, Yang Liu, and Kiattikun Chobtham. A survey of Bayesian Network structure learning. Artificial Intelligence Review, 56(8):8721–8814, August 2023. ISSN 0269-2821, 1573-7462. doi: 10.1007/ s10462-022-10351-w. URL https://link. springer.com/10.1007/s10462-022-10351-w.

[36] Gustavo Lacerda, Peter Spirtes, Joseph Ramsey, and Patrik O. Hoyer. Discovering cyclic causal models by independent components analysis. In Conference on Uncertainty in Artificial Intelligence (UAI), 2008.

[37] Wai-Yin Lam, Bryan Andrews, and Joseph Ramsey. Greedy Relaxations of the Sparsest Permuta tion Algorithm, June 2022. URL http://arxiv. org/abs/2206.05421. arXiv:2206.05421 [cs.AI].

[38] Stefen L. Lauritzen. Graphical Models. Oxford Statistical Science Series. Oxford University Press, Oxford, New York, May 1996. ISBN 978- 0-19-852219-5.

[39] Francisco Madaleno, Pratik Misra, and Alex Markham. Coarsening Causal DAG Models, April 2026. URL http://arxiv.org/abs/2601.10531. arXiv:2601.10531 [stat].

[40] Francisco Madaleno, Francisco C. Pereira, and Alex Markham. Coarsening linear Non-Gaussian causal models with cycles. In Advances in Neural Information Processing Systems (NeurIPS). In Press, 2026. URL https://arxiv.org/abs/ 2605.10163.

[41] Sarah Mameche, David Kaltenpoth, and Jilles Vreeken. Discovering Invariant and Changing Mechanisms from Data. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 1242– 1252, Washington DC USA, August 2022. ACM. ISBN 978-1-4503-9385-0. doi: 10.1145/3534678. 3539479. URL https://dl.acm.org/doi/10. 1145/3534678.3539479.

[42] Dimitris Margaritis. Learning Bayesian Network Model Structure from Data. PhD Thesis, 2003.

[43] Dimitris Margaritis and Sebastian Thrun. Bayesian Network Induction via Local Neighborhoods. In Advances in Neural Information Processing Systems, volume 12. MIT Press, 1999. URL https://proceedings. neurips.cc/paper\_files/paper/1999/hash/ 5d79099fcdf499f12b79770834c0164a-Abstract html.

[44] Alexander Marx and Jilles Vreeken. Formally Justifying MDL-based Inference of Cause and Efect, May 2021. URL http://arxiv.org/abs/2105. 01902. arXiv:2105.01902 [cs.IT].

[45] Riccardo Massidda, Atticus Geiger, Thomas Icard, and Davide Bacciu. Causal Abstraction with Soft Interventions, November 2022. URL http://arxiv.org/abs/2211.12270. arXiv:2211.12270 [cs].

[46] Riccardo Massidda, Sara Magliacane, and Davide Bacciu. Learning causal abstractions of linear structural causal models. In Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, volume 244 of UAI ’24, pages 2486– 2515, Barcelona, Spain, July 2024. JMLR.org.

[47] Nicolai Meinshausen and Peter B¨uhlmann. Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(4): 417–473, 2010. ISSN 1467-9868. doi: 10.1111/j. 1467-9868.2010.00740.x.

[48] Joris M. Mooij, Sara Magliacane, and Tom Claassen. Joint Causal Inference from Multiple Contexts. Journal of Machine Learning Research, 21(99):1–108, 2020. ISSN 1533-7928. URL http://jmlr.org/papers/v21/17-123.html.

[49] Xueyan Niu, Xiaoyun Li, and Ping Li. Learning Cluster Causal Diagrams: An Information-Theoretic Approach. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, pages 4871–4877, Vienna, Austria, July 2022. International Joint Conferences on Artificial Intelligence Organization. ISBN 978-1-956792-00-3. doi: 10.24963/ ijcai.2022/675. URL https://www.ijcai.org/ proceedings/2022/675.

[50] Weronika Ormaniec, Scott Sussex, Lars Lorch, Bernhard Sch¨olkopf, and Andreas Krause. Standardizing Structural Causal Models, March 2025. URL http://arxiv.org/abs/2406.11601. arXiv:2406.11601 [cs.LG] version: 3.

[51] Judea Pearl. Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann Publishers Inc., San Francisco, CA, United States, 1988. ISBN 978- 0-08-051489-5. doi: 10.1016/C2009-0-27609-4. ISBN: 9780080514895.

[52] Jonas Peters, Peter B¨uhlmann, and Nicolai Meinshausen. Causal inference using invariant prediction: identification and confidence intervals, November 2015. URL http://arxiv.org/abs/ 1501.01332. arXiv:1501.01332 [stat].

[53] Anne Helby Petersen. Are You Doing Better Than Random Guessing? A Call for Using Negative Controls When Evaluating Causal Discovery Algorithms. In Proceedings of the Forty-first Conference on Uncertainty in Artificial Intelligence, pages 3465–3479. PMLR, July 2025. URL https://proceedings.mlr.press/ v286/petersen25a.html.

[54] Joseph Ramsey, Peter Spirtes, and Jiji Zhang. Adjacency-faithfulness and conservative causal inference. In Proceedings of the Twenty-Second Conference on Uncertainty in Artificial Intelligence, UAI’06, pages 401–408, Arlington, Virginia, USA, July 2006. AUAI Press. ISBN 978-0- 9749039-2-7.

[55] Alexander G. Reisach, Christof Seiler, and Sebastian Weichwald. Beware of the Simulated DAG! Causal Discovery Benchmarks May Be

Easy To Game, November 2021. URL http:// arxiv.org/abs/2102.13647. arXiv:2102.13647 [stat.ML].

[56] Paul K. Rubenstein, Sebastian Weichwald, Stephan Bongers, Joris M. Mooij, Dominik Janzing, Moritz Grosse-Wentrup, and Bernhard Sch¨olkopf. Causal Consistency of Structural Equation Models, July 2017. URL http:// arxiv.org/abs/1707.00819. arXiv:1707.00819 [stat.ML].

[57] Willem Schooltink and Fabio Massimo Zennaro. Aligning graphical and functional causal abstractions. In Biwei Huang and Mathias Drton, editors, Proceedings of the Fourth Conference on Causal Learning and Reasoning, volume 275 of Proceedings of Machine Learning Research, pages 704–730. PMLR, 07–09 May 2025. URL https://proceedings.mlr.press/ v275/schooltink25a.html.

[58] Gideon Schwarz. Estimating the Dimension of a Model. The Annals of Statistics, 6(2):461– 464, 1978. ISSN 0090-5364. URL https://www. jstor.org/stable/2958889.

[59] Liam Solus, Yuhao Wang, and Caroline Uhler. Consistency Guarantees for Greedy Permutation-Based Causal Inference Algorithms, June 2021. URL http://arxiv.org/abs/1702. 03530. arXiv:1702.03530 [math].

[60] Peter Spirtes. Introduction to Causal Inference. Journal of Machine Learning Research, 11(54): 1643–1662, 2010. ISSN 1533-7928. URL http: //jmlr.org/papers/v11/spirtes10a.html.

[61] Peter Spirtes and Clark Glymour. An Algorithm for Fast Recovery of Sparse Causal Graphs. Social Science Computer Review, 9(1): 62–72, April 1991. ISSN 0894-4393. doi: 10. 1177/089443939100900106. URL https://doi. org/10.1177/089443939100900106.

[62] Chandler Squires, Yuhao Wang, and Caroline Uhler. Permutation-Based Causal Structure Learning with Unknown Intervention Targets, October 2019. URL https://arxiv.org/abs/1910. 09007v2.

[63] Chandler Squires, Annie Yun, Eshaan Nichani, Raj Agrawal, and Caroline Uhler. Causal Structure Discovery between Clusters of Nodes Induced by Latent Factors. In Proceedings of the First Conference on Causal Learning and Reasoning, pages 669–687. PMLR, June 2022. URL https://proceedings.mlr.press/ v177/squires22a.html.

[64] Richard P. Stanley. Enumerative Combinatorics,

Volume 1. Cambridge Studies in Advanced Mathematics, second edition, 2011.

[65] G´abor J. Sz´ekely and Maria L. Rizzo. Energy statistics: A class of statistics based on distances. Journal of Statistical Planning and Inference, 143(8):1249–1272, August 2013. ISSN 03783758. doi: 10.1016/j.jspi.2013.03. 018. URL https://linkinghub.elsevier.com/ retrieve/pii/S0378375813000633.

[66] Marc Teyssier and Daphne Koller. Ordering-Based Search: A Simple and Efective Algorithm for Learning Bayesian Networks, July 2012. URL http://arxiv.org/abs/1207.1429. arXiv:1207.1429 [cs].

[67] Santtu Tikka, Jouni Helske, and Juha Karvanen. Clustering and Structural Robustness in Causal Diagrams, August 2023. URL http:// arxiv.org/abs/2111.04513. arXiv:2111.04513 [stat.ML].

[68] D. J. W. Touw, A. Alfons, P. J. F. Groenen, and I. Wilms. Clusterpath Gaussian Graphical Modeling. Journal of Computational and Graphical Statistics, 0(0):1–15, May 2026. ISSN 1061-8600. doi: 10.1080/ 10618600.2026.2653764. URL https://doi. org/10.1080/10618600.2026.2653764. eprint: https://doi.org/10.1080/10618600.2026.2653764.

[69] Ioannis Tsamardinos, Laura Brown, and Constantin Aliferis. The Max-Min Hill-Climbing Bayesian Network Structure Learning Algorithm. Machine Learning, 65:31–78, October 2006. doi: 10.1007/s10994-006-6889-7.

[70] Caroline Uhler. Gaussian Graphical Models: An Algebraic and Geometric Perspective, July 2017. URL http://arxiv.org/abs/1707.04345. arXiv:1707.04345 [math.ST].

[71] A. W. van der Vaart. Asymptotic Statistics. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 1998. ISBN 978-0-521- 78450-4. doi: 10.1017/CBO9780511802256. URL https://www.cambridge.org/ core/books/asymptotic-statistics/ A3C7DAD3F7E66A1FA60E9C8FE132EE1D.

[72] Tom S. Verma and Judea Pearl. Causal Networks: Semantics and Expressiveness, 1988. URL http: //arxiv.org/abs/1304.2379. arXiv:1304.2379 [cs].

[73] Tom S. Verma and Judea Pearl. On the Equivalence of Causal Models, March 2013. URL http: //arxiv.org/abs/1304.1108. arXiv:1304.1108 [cs].

[74] Jonas Wahl, Urmi Ninad, and Jakob Runge. Vector Causal Inference between Two Groups of Vari ables. Proceedings of the AAAI Conference on Artificial Intelligence, 37(10):12305–12312, June 2023. ISSN 2374-3468. doi: 10.1609/aaai.v37i10. 26450. URL https://ojs.aaai.org/index. php/AAAI/article/view/26450.

[75] Jonas Wahl, Urmi Ninad, and Jakob Runge. Foundations of causal discovery on groups of variables. Journal of Causal Inference, 12(1):20230041, July 2024. ISSN 2193- 3685. doi: 10.1515/jci-2023-0041. URL https://www.degruyterbrill.com/document/ doi/10.1515/jci-2023-0041/html.

[76] Matthijs J. Warrens and Hanneke van der Hoef. Understanding the Adjusted Rand Index and Other Partition Comparison Indices Based on Counting Object Pairs. Journal of Classification, 39(3):487–509, November 2022. ISSN 1432-1343. doi: 10.1007/s00357-022-09413-z. URL https: //doi.org/10.1007/s00357-022-09413-z.

[77] B. L. Welch. The Generalization of ‘Student’s’ Problem when Several Diferent Population Variances are Involved. Biometrika, 34 (1/2):28–35, 1947. ISSN 0006-3444. doi: 10. 2307/2332510. URL https://www.jstor.org/ stable/2332510.

[78] Marcel Wien¨obst, Leonard Henckel, and Sebastian Weichwald. Embracing Discrete Search: A Reasonable Approach to Causal Structure Learning, February 2026. URL http://arxiv.org/ abs/2510.04970. arXiv:2510.04970 [stat.ML].

[79] Kevin Xia and Elias Bareinboim. Neural Causal Abstractions. Proceedings of the AAAI Conference on Artificial Intelligence, 38(18):20585–20595, March 2024. ISSN 2374-3468. doi: 10.1609/aaai.v38i18.30044. URL https://ojs.aaai.org/index.php/AAAI/ article/view/30044.

[80] Fabio Massimo Zennaro. Abstraction between structural causal models: A review of definitions and properties. arXiv preprint arXiv:2207.08603, 2022.

[81] Fabio Massimo Zennaro, M´at´e Dr´avucz, Geanina Apachitei, W. Dhammika Widanage, and Theodoros Damoulas. Jointly Learning Consistent Causal Abstractions Over Multiple Interventional Distributions. In Proceedings of the Second Conference on Causal Learning and Reasoning, pages 88–121. PMLR, August 2023. URL https://proceedings.mlr.press/ v213/zennaro23a.html.

# Score-Based Learning of Cluster DAGs from Interventions: Supplementary Materials

## A EXTENDED RELATED WORK

Causal abstraction The functional view formalizes an abstraction as a pair of maps between structural causal models that commute with interventions. Rubenstein et al. [56] introduce exact transformations, which Beckers and Halpern [5] refine into a hierarchy of uniform, τ-, strong τ-, and constructive abstractions, the last of which arises from an explicit partition of the low-level variables; approximate abstractions relax commutativity [6], Massidda et al. [45] extend τ-abstraction to soft interventions, and Zennaro [80] survey the resulting landscape. On the graphical side, Anand et al. [1] introduce C-DAGs—graphs over an assumed clustering with unspecified intra-cluster structure—and prove d-separation, do-calculus, and the ID algorithm sound and complete on them whenever the clustering is admissible. Tikka et al. [67] take the causal query as given and characterize the transit clusters whose collapse preserves identifiability, and Wahl et al. [75] generalize to group directed mixed graphs with cycles and latents, showing that the Markov property transfers under grouping while faithfulness requires internally well-connected clusters. Schooltink and Zennaro [57] prove that C-DAGs correspond exactly to constructive τ-abstractions; a coarsening is the C-DAG induced by its partition, so learning one from data is implicitly learning a constructive functional abstraction. Madaleno et al. [40] extend the same framework to linear non-Gaussian models with cycles: any DAG-coarsening must group strongly connected components, and the resulting condensation is identifiable from observational data alone, invariant across the ICA equivalence class of Lacerda et al. [36].

Learning abstractions from data Existing eforts parameterize the abstraction map with deep architectures [9, 15, 79], learn it under linearity assumptions [46], use optimal transport for alignment [16], or target distributionally robust abstractions [21]. Zennaro et al. [81] jointly learn abstractions consistent over multiple interventional distributions, and Niu et al. [49] learn cluster causal diagrams information-theoretically from observational data. Another thread learns structure given a known clustering [20, 34, 74], and partitions have long been used to scale structure learning of large networks [28, 68]. Most of these works either assume the partition is provided or impose a strong parametric structure; jointly discovering the partition and its abstract DAG from data remains an exception, as does any score-based treatment.

Structure learning paradigms Kitson et al. [35] survey the field. Constraint-based methods, e.g. PC [61], prune a complete graph by CI tests; their order dependence and the deterioration of high-order tests make errors accumulate [54, 69]. Score-based methods optimize a decomposable criterion such as the BIC [58], at the price of an NP-hard search [14]: greedy equivalence search [13], ordering-based search [66], permutation-based methods with consistency guarantees [59] and their greedy relaxations [37], reinsertion-based searches with growshrink trees [4], and discrete-search approaches [78]. Our edge phase uses the score-based restricted grow-shrink backbone [37, 78], with a coarse, multi-environment score in place of the fine, single-environment BIC.

Learning from interventional data Interventions come in hard (graph-mutilating) and soft (mechanismchanging) varieties; soft interventions preserve the DAG and are the natural regime for heterogeneous data whose targets are unknown. Hauser and B¨uhlmann [30] characterize interventional Markov equivalence for known hard targets and learn it greedily (GIES), Gamella et al. [24] treat unknown soft targets (GnIES), Squires et al. [62] estimate targets jointly with the graph (UT-IGSP), and Peters et al. [52] exploit invariance across environments without target knowledge (ICP). Further work addresses experimental design and active learning [19, 22, 25, 31, 32]. Chen and Markham [12] extend the order-based FLOP algorithm [78] to interventional data with known targets, scoring prefix-restricted parent sets with a target-filtered pooled interventional BIC. Our setting is multi-environment with unknown, possibly multi-target soft interventions, and the interventions determine not only the identifiability target but also the search order.

## B ADDITIONAL THEORY AND PROOFS

App. B.1 to App. B.3.2 collect background results; App. B.3.3 to App. B.7 prove the results of Sec. 3.

## B.1 Graph and probability theory

## B.1.1 d-separation.

Given a DAG G, we say a path between u and w in G is blocked by a set $S \subseteq V \setminus \{ u , w \}$ if there exists a node z such that either:

$z \in S$ and z is a non-collider;

• z is a collider but neither z nor any of its descendants is in $S , \mathrm { d e } _ { \mathcal { G } } ( z ) \cap S = \emptyset$

A path that is not blocked by S is active given S. Disjoint sets $A , B \subseteq V$ are d-separated by S in ${ \mathcal { G } } ,$ written $A \perp _ { \mathcal { G } } B \mid S , \operatorname { i f } S$ blocks every path between any node of A and any node of $B \ [ 5 1 ]$

## B.1.2 Graphoids.

A CI relation on a finite set U is a set of triples $( A , B \mid C )$ of pairwise disjoint subsets of U, written $A \perp \perp B \mid C$ Both $\perp _ { P }$ (with $U = V )$ and the pooled block relation $\perp \perp _ { \mathcal { E } }$ of Sec. 3.3 (with $U = \Pi ^ { \mathcal { E } } )$ are CI relations.

Definition 21 (Semi-graphoid, graphoid, compositional graphoid). A CI relation on U is a semi-graphoid if for all pairwise disjoint $A , B , C , D \subseteq U { \mathrm { : } }$

(1) Symmetry: $A \perp B \mid C \Rightarrow B \perp A \mid C ;$

(2) Decomposition: $A \perp B \cup D \mid C \Rightarrow A \perp B \mid C ;$

(3) Weak union: $A \perp B \cup D \mid C \Rightarrow A \perp B \mid C \cup D ;$

(4) Contraction: A $B \mid C$ and A $D \mid B \cup C \Rightarrow A \bot B \cup D \mid C .$

It is a graphoid if it also satisfies

(5) Intersection: A $B \mid C \cup D$ and A $D \mid B \cup C \Rightarrow A \bot B \cup D \mid C .$

and a compositional graphoid if it also satisfies

(6) Composition: A B | C and $A \perp D \mid C \Rightarrow A \perp B \cup D \mid C .$

The semi-graphoid axioms hold for the CI relation of any distribution, intersection holds under a strictly positive density, and the CI relation of a Gaussian distribution with positive definite covariance is a compositional graphoid [38]. In this work we assume the data follow a Gaussian distribution, hence $\perp _ { P }$ will always be regarded as a graphoid.

## B.1.3 Markov boundary and restricted Markov boundary.

We now show the results needed for existence and uniqueness of the Markov boundary and its restricted version. Theorem 22 (Existence and uniqueness of the Markov boundary). If is a graphoid, then for every $v \in V$ the Markov boundary $B ( v )$ exists and is unique.

The proof combines weak union and intersection applied to the symmetric diference of two candidate boundaries. Strict positivity of $P$ delivers the intersection axiom and hence the hypothesis of the theorem, so $B ( v )$ exists and is unique in the regime assumed throughout this work. For the full argument see Verma and Pearl [72].

In practice, we restrict the search procedures for the Markov boundary from all of $V \setminus \{ v \}$ to a candidate pool $Z$ fixed in advance, typically by a topological order. The corresponding refinement is given in Def. 4. In this case, existence and uniqueness extend without modification: the graphoid axioms are closed under restriction to triples $( A , B \mid C )$ with $A \cup B \cup C \subseteq Z \cup \{ v \}$ , so Thm. 22 applied to the restricted relation, with $Z \cup \{ v \}$ in place of $V ,$ , gives existence and uniqueness of $B ( v , Z )$ for every $Z \subseteq V \setminus \{ v \}$

What Prop. 5 requires of the pool is only that it contains all parents and no descendants, $\mathrm { p a } _ { \mathcal { G } } ( v ) \subseteq Z \subseteq \mathrm { n d } _ { \mathcal { G } } ( v )$ The prefix $\tau _ { \prec v }$ of any topological order $\tau$ of $\mathcal { G }$ (Def. 2) is the standard example. In coarse the pool is instead the parent pool $\mathrm { p a } ^ { \star } ( \pi )$ of Def. 12, which satisfies the same property by Thm. 13(ii), so the restricted boundary again collapses to the parents.

Proof of Prop. 5. By assumption, $\begin{array} { l } { \mathrm { p a } _ { \mathcal { G } } ( v ) \subset \mathcal { Z } \subset \mathrm { n d } _ { \mathcal { G } } ( v ) } \end{array}$ The local Markov property yields $X _ { v } \quad \perp \perp _ { P }$ $X _ { \mathrm { n d } _ { \mathcal { G } } ( v ) \backslash \mathrm { p a } _ { \mathcal { G } } ( v ) } \ | \ X _ { \mathrm { p a } _ { \mathcal { G } } ( v ) }$ , and decomposition restricts this to $X _ { v } \perp \perp _ { P } X _ { Z \backslash \mathrm { p a } _ { G } ( v ) } \mid X _ { \mathrm { p a } _ { G } ( v ) } ,$ , so $\operatorname { p a } _ { \mathcal { G } } ( v )$ satisfies (i) of Def. 4. For minimality, suppose $S \subsetneq \mathrm { p a } _ { \mathcal { G } } ( v )$ satisfies (i); decomposition gives $\bar { X } _ { v } \ \bar { \perp } \bar { \mid } _ { P } \ X _ { u } \ | \ \bar { X } _ { S }$ for every $u \in \mathrm { p a } _ { \mathcal { G } } ( v ) \backslash S$ , but faithfulness forces u and v to be d-connected since they are adjacent in ${ \mathcal { G } } ,$ , a contradiction. Uniqueness then gives $B ( v , Z ) = \mathrm { p a } _ { \mathcal { G } } ( v )$ □

## B.2 Grow-shrink algorithm

Alg. 1 is the non-greedy, score-based, restricted grow-shrink of Wien¨obst et al. [78], stated for a generic target v and pool Z. In coarse, the target is a block $\pi _ { j } .$ the pool is $\mathrm { p a } ^ { \star } ( \pi _ { j } )$ , the candidates are blocks, and the local score is the pooled BIC of Def. 17.

Algorithm 1 Non-greedy, score-based, restricted grow-shrink   
Require: target v, pool Z, local score $s ( \boldsymbol { v } , \cdot ; \mathcal { D } )$   
1: $S \gets \emptyset$   
2: repeat ▷ grow   
3: $S _ { \mathrm { o l d } }  S$   
4: for $w \in Z \backslash S$ do   
5: if $s ( v , S \cup \{ w \} ; \mathcal { D } ) > s ( v , S ; \mathcal { D } )$ then $S  S \cup \{ w \}$   
6: end if   
7: end for   
8: until $S = S _ { \mathrm { o l d } }$   
9: repeat ▷ shrink   
10: $S _ { \mathrm { o l d } }  S$   
11: for $w \in S$ do   
12: if $s ( v , S \setminus \{ w \} ; \mathcal { D } ) > s ( v , S ; \mathcal { D } )$ then $S \gets S \setminus \{ w \}$   
13: end if   
14: end for   
15: until $S = S _ { \mathrm { o l d } }$   
16: return S

Lemma 23 (Correctness of grow-shrink). Let be a compositional graphoid on $U , v \in U$ and $Z \subseteq U \setminus \{ v \}$

(i) (Oracle.) If Alg. 1 adds w $i f f \ v \ \not \perp w \mid S$ and removes w $i f f v \perp \mid w \mid S \setminus \{ w \}$ , it returns the Markov boundary of v relative to Z after $\mathcal { O } ( | Z | ^ { 2 } )$ queries ${ \big . } { \big . } { \big . } { \big . } { \big . } { \big . } { \big . }$

(ii) (Score.) $I f s ( v , S \cup \{ w \} ; \mathcal { D } ) - s ( v , S ; \mathcal { D } )$ diverges in probability to +∞ when v ${ \mathrm { ~ , ~ } } \sqcup \ w \mid S$ and $t o \mathrm { ~ -- } \infty$ when v $w \mid S , f o r$ all $S \subseteq Z$ and w $\in Z \backslash S _ { i }$ , then Alg. 1 returns the same set with probability tending to one.

Proof. (i) Each grow sweep that changes S adds an element, so the grow phase stops after at most $| Z | + 1$ sweeps of at most |Z| queries, and the same holds for the shrink phase. At the end of the grow phase, v w | S for every $w \in Z \backslash S .$ , so composition gives v $Z \backslash S \mid S$ . The shrink phase preserves this property: if v $Z \backslash S \mid S$ and $v \perp \perp w \mid S \setminus \{ w \}$ , contraction gives $v \perp \perp ( \vec { Z } \setminus \vec { S } ) \cup \{ w \} \mid S \setminus \{ w \}$ . At termination, v $\mathcal { A } w \mid S \setminus$ {w} for every w $\in S .$ Suppose some $S ^ { \prime } \subsetneq S$ also satisfied v $Z \setminus S ^ { \prime } \mid S ^ { \prime }$ , and take $w \in S \setminus S ^ { \prime }$ . Decomposition gives v $S \setminus S ^ { \prime } \mid S ^ { \prime }$ and weak union gives v $w \mid S \setminus \{ w \}$ , a contradiction. Hence $S$ is minimal, and Thm. 22 identifies it with the Markov boundary. (ii) A removal comparison is the negative of an addition comparison from $S \setminus \{ w \}$ , so every comparison the algorithm can make has the sign of a CI statement in the limit. There are finitely many pairs $( S , w )$ , so with probability tending to one all of them have the correct sign simultaneously, and on this event the run coincides with the oracle run of (i). □

## B.3 Coarsening theory

## B.3.1 The lattice of valid coarsenings

Madaleno et al. [39, Theorem 3] show that the valid coarsenings of a DAG form a lattice under refinement. The theorem statement (and its proof) is correct except for its last phrase (and sentence), which claims that because the set of valid coarsenings is itself a lattice, it is a sublattice of the partition refinement lattice. Prop. 24 is the corrected version of this claim, in case it is of independent mathematical interest. All results that build on the theorem are unafected: they use only the lattice structure of the valid coarsenings, which the proof does establish, and not the false sublattice claim.

Proposition 24. Let $\mathcal { G } = ( V , E )$ be $a \ D A G ,$ , and let ${ \mathcal { L } } ( { \mathcal { G } } )$ be the set of valid coarsening partitions of $V _ { i }$ , ordered by refinement $( \Pi \preceq \Pi ^ { \prime }$ if Π refines $\Pi ^ { \prime } \big )$ . Then ${ \mathcal { L } } ( { \mathcal { G } } )$ is a meet-subsemilattice of the partition refinement lattice Π<sub>V</sub> [10, Definition 1 and Remark 2].

Proof. The partition of V into the non-empty intersections of blocks of Π and $\Pi ^ { \prime }$ induces an acyclic coarsening whenever Π and $\Pi ^ { \prime }$ do, as shown in the proof of Madaleno et al. [39, Theorem 3], and being the greatest partition refining both, it is their meet in ${ \mathcal { L } } ( { \mathcal { G } } )$ . Since $\Pi _ { V }$ is finite and $\{ V \} \in \mathcal { L } ( \mathcal { G } ) , \mathcal { L } ( \mathcal { G } )$ is a lattice [64, Proposition 3.3.1], with join $\Pi \vee _ { \mathcal { L } } \Pi ^ { \prime }$ the finest valid common coarsening of Π and Π<sup>′</sup>. □

The last line of the above proof is why the coarsening lattice is a meet-subsemilattice rather than a sublattice of the partition refinement lattice: the join of two elements of the coarsening lattice may be coarser than their corresponding (ambient) join in the partition refinement lattice, as shown in the following example.

Example 25. Let $\mathcal { G }$ be the DAG $1  2  3$ with node 4 isolated, and consider the partitions $\Pi _ { 1 } = \left\{ 1 4 \ \right|$ $2 \mid 3 \}$ and $\Pi _ { 2 } = \{ 1 ~ | ~ 2 ~ | ~ 3 4 \}$ of $V = [ 4 ]$ . Both are valid: the coarsenings they induce are the directed paths $\{ 1 4 \}  \{ 2 \}  \{ 3 \}$ and $\{ 1 \}  \{ 2 \}  \{ 3 4 \}$ . The ambient join merges the overlapping blocks 14 and 34, giving $\Pi _ { 1 } \vee \Pi _ { 2 } = \{ 1 3 4 \mid 2 \}$ , which is not valid: the fine edges $1  2$ and $2 $ 3 induce the coarse edges $\{ 1 3 4 \}  \{ 2 \}$ and $\{ 2 \}  \{ 1 3 4 \}$ , a directed cycle. The common upper bounds of $\Pi _ { 1 }$ and $\Pi _ { 2 }$ in $\Pi _ { V }$ are $\{ 1 3 4 \mid 2 \}$ and the top {1234}, so the only valid one is the top, giving $\Pi _ { 1 } \vee _ { \mathcal { L } } \Pi _ { 2 } = \{ 1 2 3 4 \}$ , strictly coarser than $\Pi _ { 1 } \lor \Pi _ { 2 }$

## B.3.2 Relation to the interventional coarsening of Madaleno et al. [39]

Madaleno et al. [39, Def. 8] define the interventional coarsening $\mathcal { G } ^ { \mathbb { Z } }$ through intervened ancestors, while Def. 8 defines $\mathcal { G } ^ { \varepsilon }$ through signatures. Here we show that the two coincide for single-target interventions, and that with multi-target interventions $\mathcal { G } ^ { \varepsilon }$ can be strictly coarser and $\mathcal { G } ^ { \mathbb { Z } }$ is not identifiable from marginal shifts.

Proposition 26 (The signature does not determine $\Pi ^ { \mathcal { I } }$ under multi-target interventions). There exist a $D A G$ $\mathcal { G }$ and environments E with a multi-target intervention for which $\Pi ^ { \varepsilon }$ is strictly coarser than $\Pi ^ { \mathcal { I } }$ . Consequently the signature-based Refine oracle of Madaleno et al. $\it { 1 3 9 }$ , Thm. 10] does not recover $\mathcal { G } ^ { \mathbb { Z } }$

Proof. Take $\mathcal { G } : 1  2 , 1  3 , \mathcal { E } = \{ 0 , 1 \}$ and $I _ { 1 } = \{ 1 , 2 \}$ . The intervened-ancestor sets are

$$
\begin{array} { r } { \mathcal { L } \mathrm { - } \mathrm { a n } _ { \mathcal { G } } ( 1 ) = \{ 1 \} , \qquad \mathcal { Z } \mathrm { - } \mathrm { a n } _ { \mathcal { G } } ( 2 ) = \{ 1 , 2 \} , \qquad \mathcal { Z } \mathrm { - } \mathrm { a n } _ { \mathcal { G } } ( 3 ) = \{ 1 \} , } \end{array}
$$

so $\Pi ^ { \mathcal { T } } = \{ \{ 1 , 3 \} , \ \{ 2 \} \}$ . Since $\deg ( I _ { 1 } ) = \{ 1 , 2 , 3 \}$ , Ass. 9(iii) gives $\sigma ( 1 ) = \sigma ( 2 ) = \sigma ( 3 ) = \{ 1 \}$ , so $\Pi ^ { \varepsilon } = \{ V \}$ 2 strictly coarser than $\vec { \Pi } ^ { \mathcal { Z } }$ . In particular $\sigma ( 2 ) = \sigma ( 3 )$ while $\mathcal { T } \mathrm { - } \mathrm { a n } _ { \mathcal { G } } ( 2 ) \neq \mathcal { T } \mathrm { - } \mathrm { a n } _ { \mathcal { G } } ( 3 )$ □

Proposition 27. Under Ass. $\boldsymbol { \mathscr { s } } , \Pi ^ { \mathcal { Z } } \preceq \Pi ^ { \mathcal { E } }$ , and $\mathcal { G } ^ { \varepsilon }$ is a valid coarsening of $\mathcal { G } ^ { \mathbb { Z } }$

Proof. Both partitions are equivalence classes of $V \colon \textit { v } \sim _ { \Pi ^ { \tau } }$ w if ${ \mathcal { T } } \mathrm { - a n } _ { \mathcal { G } } ( v ) = { \mathcal { T } } \mathrm { - a n } _ { \mathcal { G } } ( w )$ , and $v \ \sim _ { \Pi ^ { \varepsilon } }$ w if $\sigma ( v ) = \sigma ( w )$ . It sufices to show that

$$
v \sim _ { \Pi ^ { \tau } } w \implies v \sim _ { \Pi ^ { \varepsilon } } w .
$$

Let $v , w \in V$ with ${ \mathcal { T } } \mathrm { - } \mathrm { a n } _ { \mathcal { G } } ( v ) = { \mathcal { T } } \mathrm { - } \mathrm { a n } _ { \mathcal { G } } ( w )$ . By Def. 7 and fixing the correspondence of each environment $e \in \mathcal { E } \backslash \{ 0 \}$ with a set $I _ { e }$ of intervention targets this implies $\sigma ( v ) = \sigma ( w )$ . Thus ${ \mathcal { T } } \mathrm { - } \mathrm { a n } _ { { \mathcal { G } } } ( v ) = { \mathcal { T } } \mathrm { - } \mathrm { a n } _ { { \mathcal { G } } } ( w )$ gives $\sigma ( v ) = \sigma ( w )$ ), i.e., v ∼ E w. □

On the other hand, if we assume that each environment is represented by a single-target intervention the two partitions agree.

Assumption 28 (Single-target environments). For every $e \in \mathcal { E } \setminus \{ 0 \} , | I _ { e } | = 1$

Proposition 29. Under Ass. 9 and Ass. 28, $\Pi ^ { \mathcal { E } } = \Pi ^ { \mathcal { Z } }$

Proof. Let $\textstyle T : = \bigcup _ { e \in { \mathcal { E } } \setminus \{ 0 \} } I _ { e }$ be the set of all intervention targets. From Madaleno et al. [39, Definition 8] we have,

$$
\begin{array} { r } { \mathcal { T } \mathbf { - } \mathrm { a n } _ { \mathcal { G } } ( v ) = \mathrm { a n } _ { \mathcal { G } } ( v ) \cap T . } \end{array}\tag{7}
$$

Single-target gives $I _ { e } = \{ t _ { e } \}$ for some $t _ { e } \in T$ , hence $T = \{ t _ { e } \ | \ e \in \mathcal { E } \setminus \{ 0 \} \}$ and $v \in \mathrm { d e } g ( I _ { e } ) \iff t _ { e } \in \mathsf { a n } g ( v )$ Together with Ass. 9(iii) this yields

$$
\sigma ( v ) = \{ e \in \mathcal { E } \setminus \{ 0 \} \mid t _ { e } \in \mathrm { a n } _ { \mathcal { G } } ( v ) \} .\tag{8}
$$

Quantifying over environments then coincides with quantifying over targets in $T ,$ and for every $v , w \in V$

$$
\sigma ( v ) = \sigma ( w ) \iff \mathrm { a n } _ { \mathcal { G } } ( v ) \cap T = \mathrm { a n } _ { \mathcal { G } } ( w ) \cap T .
$$

By (7) the right-hand side reads ${ \mathcal { T } } \mathrm { - } \mathrm { a n } _ { \mathcal { G } } ( v ) = { \mathcal { T } } \mathrm { - } \mathrm { a n } _ { \mathcal { G } } ( w )$ . Since $\Pi ^ { \varepsilon }$ and $\Pi ^ { \mathcal { I } }$ partition V into the equivalence classes of equal $\sigma ( \cdot )$ and equal $\begin{array} { r l } {  { \mathcal { T } \mathrm { - a n } _ { \mathcal { G } } ( \cdot ) } } \end{array}$ respectively, this gives $\Pi ^ { \mathcal { E } } = \Pi ^ { \mathcal { Z } }$ □

## B.3.3 Validity of the interventional coarsening and causal orders from interventions

Proof of Lem. 10. We first show that, for every $v \in V$

$$
\sigma ( v ) = \{ e \in \mathcal { E } \setminus \{ 0 \} \mid v \in \deg ( I _ { e } ) \} .\tag{9}
$$

The inclusion $\supseteq$ is given by Ass. 9(iii). For $\subseteq$ , let $v \not \in \mathrm { d e } _ { \mathcal { G } } ( I _ { e } )$ , so that $\mathrm { a n } _ { \mathcal { G } } ( v ) \cap I _ { e } = \emptyset$ . Since $\mathrm { a n } _ { \mathcal { G } } ( v )$ is ancestral, integrating (4) over $x _ { V \backslash \mathrm { a n } _ { \mathcal { G } } ( v ) }$ gives

$$
p ^ { ( e ) } \big ( x _ { \mathrm { a n } _ { \mathcal { G } } ( v ) } \big ) = \prod _ { u \in \mathrm { a n } _ { \mathcal { G } } ( v ) } p ^ { ( 0 ) } \big ( x _ { u } \mid x _ { \mathrm { p a } _ { \mathcal { G } } ( u ) } \big ) = p ^ { ( 0 ) } \big ( x _ { \mathrm { a n } _ { \mathcal { G } } ( v ) } \big ) ,
$$

hence $f ^ { ( e ) } ( X _ { v } ) = f ^ { ( 0 ) } ( X _ { v } )$ and $e \not \in \sigma ( v )$ . Now let $u \to w$ in $\mathcal { G }$ . Then $\operatorname { a n } _ { \mathcal { G } } ( u ) \subseteq \operatorname { a n } _ { \mathcal { G } } ( w )$ , so $u \in \mathrm { d e } _ { \mathcal { G } } ( I _ { e } )$ implies $w \in \mathrm { d e } _ { \mathcal { G } } ( I _ { e } )$ for every $e \in \mathcal { E } \setminus \{ 0 \}$ , and (9) gives $\sigma ( u ) \subseteq \sigma ( w )$ □

Proof of Prop. 11. By Def. 6, only acyclicity of $\mathcal { G } ^ { \varepsilon }$ needs checking. Suppose for contradiction that $\mathcal { G } ^ { \varepsilon }$ contained a directed cycle $\mu _ { 0 } \to \mu _ { 1 } \to \cdot \cdot \cdot \to \mu _ { k } = \mu _ { 0 }$ with $\mu _ { j - 1 } \neq \mu _ { j }$ for every j. Each coarse edge $\mu _ { j - 1 } \to \mu _ { j }$ arises from some fine edge $u _ { j }  v _ { j }$ with $u _ { j } \in \mu _ { j - 1 }$ and $v _ { j } \in \mu _ { j }$ , while consecutive fine endpoints share a block, $v _ { j } , u _ { j + 1 } \in \mu _ { j }$ (with $u _ { k + 1 } : = u _ { 1 } )$ , whence $\sigma ( v _ { j } ) = \sigma ( u _ { j + 1 } )$ . Chaining these equalities with the inclusions $\sigma ( u _ { j } ) \subseteq \sigma ( v _ { j } )$ of Lem. 10 around the cycle yields

$$
\sigma ( u _ { 1 } ) \subseteq \sigma ( v _ { 1 } ) = \sigma ( u _ { 2 } ) \subseteq \cdots = \sigma ( u _ { k } ) \subseteq \sigma ( v _ { k } ) = \sigma ( u _ { 1 } ) ,
$$

so every inclusion is an equality. In particular $\sigma ( u _ { 1 } ) ~ = ~ \sigma ( v _ { 1 } )$ , placing $u _ { 1 }$ and $v _ { 1 }$ in one block of $\Pi ^ { \varepsilon }$ and contradicting $\mu _ { 0 } \neq \mu _ { 1 }$ □

Proof of Thm. $\mathit { 1 3 . } \ : \left( \mathrm { i } \right)$ A coarse edge $\pi  \pi ^ { \prime }$ comes from a fine edge $u \to w$ with u $\in \pi$ and w $\in \pi ^ { \prime }$ , so Lem. 10 gives $\operatorname { s u p p } ( \pi ) = \sigma ( u ) \subseteq \sigma ( w ) = \operatorname { s u p p } ( \pi ^ { \prime } )$ . Chaining along a directed path from $\pi _ { a }$ to π gives $\operatorname { s u p p } ( \pi _ { a } ) \subseteq \operatorname { s u p p } ( \pi _ { b } )$ and the inclusion is strict since $\pi _ { a } \neq \pi _ { b }$ . (ii) Parents are ancestors, so the first inclusion follows from (i). If some $\pi ^ { \prime } \in \mathrm { p a } ^ { \star } ( \pi _ { a } )$ were a descendant of $\pi _ { a } , \mathrm { ( i ) }$ would give $\pi _ { a } < _ { \mathrm { s u p p } } \pi ^ { \prime }$ , contradicting $\pi ^ { \prime } < _ { \mathrm { s u p p } } \pi _ { a }$ . If the supports of $\pi _ { a }$ and $\pi _ { b }$ are incomparable, (i) rules out ancestry in both directions, hence also adjacency. (iii) Adjacent blocks carry exactly one orientation, and by (i) it runs from the smaller support to the larger. □

Example 30 (Counterexample for the converse of Thm. 13(i)). Let G be $a \right. c \left. b$ with targets $I _ { 1 } = \{ a , b \}$ ， $I _ { 2 } = \left\{ b \right\}$ and $I _ { 3 } = \{ c \}$ . Then $\sigma ( a ) = \{ 1 \} , \sigma ( b ) = \{ 1 , 2 \}$ and $\sigma ( c ) = \{ 1 , 2 , 3 \}$ , so $\Pi ^ { \varepsilon }$ is the discrete partition and $\mathcal { G } ^ { \mathcal { E } } = \mathcal { G }$ . Here supp $( \{ a \} ) \subsetneq \operatorname { s u p p } ( \{ b \} )$ , yet a and b are non-adjacent and neither is an ancestor of the other.

Proof of Cor. $1 \% .$ (i) Fix $\tau \in \mathcal { T } ( < _ { \mathrm { s u p p } } )$ . If $\pi _ { a }  \pi _ { b } \in \mathcal G ^ { \mathcal E }$ , then $\pi _ { a } \prec _ { \mathcal { G } ^ { \mathcal { E } } } \pi _ { b } .$ , so $\pi _ { a } < _ { \mathrm { s u p p } } \pi _ { b }$ by Thm. $1 3 ( \mathrm { i } )$ , and $\pi _ { a }$ precedes $\pi _ { b }$ in τ since τ extends $< _ { \mathrm { s u p p } }$ . Hence τ is a topological order of $\mathcal { G } ^ { \varepsilon }$ . Likewise, every $\pi ^ { \prime } \in \mathrm { p a } ^ { \star } ( \pi )$ satisfies $\pi ^ { \prime } < \sum _ { \mathrm { s u p p } } \pi$ by Def. 12, so $\pi ^ { \prime }$ precedes π in $\tau . ~ \mathrm { ( i i ) }$ Under Ass. $9 , P ^ { ( e ) }$ is Markov and faithful to $\mathcal { G } ^ { \varepsilon }$ , and $\mathrm { p a } _ { \mathcal { G } ^ { \varepsilon } } ( \pi ) \subseteq \mathrm { p a } ^ { \star } ( \pi ) \subseteq \mathrm { n d } _ { \mathcal { G } ^ { \varepsilon } } ( \pi )$ by Thm. 13(ii). Prop. 5, read with blocks as nodes, then gives the claim. □

## B.4 The coarse block-Gaussian model

This section gives a short primer on the coarse block-Gaussian model. It is not exhaustive, but it covers what is needed to follow the method. For a fuller account we refer the reader to Andersson and Perlman [2], Drton and Eichler [17].

## B.4.1 Data and sample covariances

We observe $n _ { e }$ i.i.d. realizations of $X ^ { ( e ) }$ in environment $e ,$ collected columnwise into the centered data matrix $\mathbf { X } ^ { ( e ) } \in \mathbb { R } ^ { d \times n _ { e } }$ . Lowercase $\boldsymbol { x } ^ { ( e ) } \in \mathbb { R } ^ { d }$ denotes a generic column and $x _ { j } ^ { ( e ) } : = ( x _ { v } ^ { ( e ) } ) _ { v \in \pi _ { j } } \in \mathbb { R } ^ { r _ { j } }$ its restriction to block $\pi _ { j }$ . Stacking the realizations of $X _ { j } ^ { ( e ) }$ and of $X _ { \mathrm { P a } _ { j } } ^ { ( e ) }$ gives the row-submatrices $\mathbf { X } _ { j } ^ { ( e ) } \in \mathbb { R } ^ { r _ { j } \times n _ { e } }$ and $\mathbf { X } _ { \mathrm { P a } _ { i } } ^ { ( e ) } \in \mathbb { R } ^ { s _ { j } \times n _ { e } }$ ， that is, the rows of $\mathbf { X } ^ { ( e ) }$ indexed by $\pi _ { j }$ and by $\textstyle \bigcup _ { \pi _ { k } \in \mathrm { P a } _ { j } } \pi _ { k }$ . The sample covariance is $\begin{array} { r } { S ^ { ( e ) } = \frac { 1 } { n _ { e } } \mathbf { X } ^ { ( e ) } ( \bar { \mathbf { X } ^ { ( e ) } } ) ^ { \top } } \end{array}$ , with blocks

$$
\begin{array} { r } { S _ { j , j } ^ { ( e ) } : = \frac { 1 } { n _ { e } } \mathbf { X } _ { j } ^ { ( e ) } ( \mathbf { X } _ { j } ^ { ( e ) } ) ^ { \top } , \qquad S _ { j , \mathrm { P a } _ { j } } ^ { ( e ) } : = \frac { 1 } { n _ { e } } \mathbf { X } _ { j } ^ { ( e ) } ( \mathbf { X } _ { \mathrm { P a } _ { j } } ^ { ( e ) } ) ^ { \top } , \qquad S _ { \mathrm { P a } _ { j } , \mathrm { P a } _ { j } } ^ { ( e ) } : = \frac { 1 } { n _ { e } } \mathbf { X } _ { \mathrm { P a } _ { j } } ^ { ( e ) } ( \mathbf { X } _ { \mathrm { P a } _ { j } } ^ { ( e ) } ) ^ { \top } . } \end{array}
$$

For a candidate $B _ { j } ^ { ( e ) } \in \mathbb { R } ^ { r _ { j } \times s _ { j } }$ , the residual sample covariance is

$$
\begin{array} { r l } & { S _ { j } ^ { ( e ) } ( B _ { j } ^ { ( e ) } ) : = \frac { 1 } { n _ { e } } \big ( \mathbf { X } _ { j } ^ { ( e ) } - B _ { j } ^ { ( e ) } \mathbf { X } _ { \mathrm { P a } _ { j } } ^ { ( e ) } \big ) \big ( \mathbf { X } _ { j } ^ { ( e ) } - B _ { j } ^ { ( e ) } \mathbf { X } _ { \mathrm { P a } _ { j } } ^ { ( e ) } \big ) ^ { \top } } \\ & { \qquad = S _ { j , j } ^ { ( e ) } - B _ { j } ^ { ( e ) } S _ { \mathrm { P a } _ { j } , j } ^ { ( e ) } - S _ { j , \mathrm { P a } _ { j } } ^ { ( e ) } ( B _ { j } ^ { ( e ) } ) ^ { \top } + B _ { j } ^ { ( e ) } S _ { \mathrm { P a } _ { j } , \mathrm { P a } _ { j } } ^ { ( e ) } ( B _ { j } ^ { ( e ) } ) ^ { \top } . } \end{array}\tag{10}
$$

## B.4.2 Maximum likelihood estimation

Taking logs of the block-recursive density and summing over the $n _ { e }$ columns yields the decomposable loglikelihood $\begin{array} { r } { \ell ^ { ( e ) } ( \mathcal { G } ^ { \mathcal { E } } , \theta ^ { ( e ) } ) = \sum _ { j = 1 } ^ { q } \ell _ { j } ^ { ( e ) } ( \mathcal { G } ^ { \mathcal { E } } , \theta _ { j } ^ { ( e ) } ) } \end{array}$ , with

$$
\ell _ { j } ^ { ( e ) } ( \mathcal { G } ^ { \varepsilon } , \theta _ { j } ^ { ( e ) } ) = - \frac { n _ { e } r _ { j } } { 2 } \log ( 2 \pi ) - \frac { n _ { e } } { 2 } \log \Bigl \vert \Sigma _ { j } ^ { ( e ) } \Bigr \vert - \frac { n _ { e } } { 2 } \mathrm { t r } \Bigl \lbrack ( \Sigma _ { j } ^ { ( e ) } ) ^ { - 1 } S _ { j } ^ { ( e ) } ( B _ { j } ^ { ( e ) } ) \Bigr \rbrack .\tag{11}
$$

Since the parameter space also factorizes across blocks [17, Sec. 2], the maximum likelihood estimate (MLE) is obtained by maximizing each $\ell _ { j } ^ { ( e ) }$ separately, one block regression at a time. With no constraints on $B _ { j } ^ { ( e ) }$ or $\Sigma _ { j } ^ { ( e ) }$ it is the ordinary two-step least-squares estimator

$$
\widehat { B } _ { j } ^ { ( e ) } = \mathbf { X } _ { j } ^ { ( e ) } ( \mathbf { X } _ { \mathrm { P a } _ { j } } ^ { ( e ) } ) ^ { \top } \left[ \mathbf { X } _ { \mathrm { P a } _ { j } } ^ { ( e ) } ( \mathbf { X } _ { \mathrm { P a } _ { j } } ^ { ( e ) } ) ^ { \top } \right] ^ { - 1 } = S _ { j , \mathrm { P a } _ { j } } ^ { ( e ) } \left( S _ { \mathrm { P a } _ { j } , \mathrm { P a } _ { j } } ^ { ( e ) } \right) ^ { - 1 } ,\tag{12}
$$

$$
\widehat { \Sigma } _ { j } ^ { ( e ) } = S _ { j , j } ^ { ( e ) } - \widehat { B } _ { j } ^ { ( e ) } S _ { \mathrm { P a } _ { j } , j } ^ { ( e ) } = S _ { j , j } ^ { ( e ) } - S _ { j , \mathrm { P a } _ { j } } ^ { ( e ) } \left( S _ { \mathrm { P a } _ { j } , \mathrm { P a } _ { j } } ^ { ( e ) } \right) ^ { - 1 } S _ { \mathrm { P a } _ { j } , j } ^ { ( e ) } .\tag{13}
$$

Substituting (12) and (13) into (11), using $S _ { j } ^ { ( e ) } ( \widehat { B } _ { j } ^ { ( e ) } ) = \widehat { \Sigma } _ { j } ^ { ( e ) }$ so that $\mathrm { t r } [ ( \widehat { \Sigma } _ { j } ^ { ( e ) } ) ^ { - 1 } S _ { j } ^ { ( e ) } ( \widehat { B } _ { j } ^ { ( e ) } ) ] = r _ { j }$ , and summing over $j$ with $\textstyle \sum _ { j } r _ { j } = d$ , gives the profile log-likelihood

$$
\begin{array} { l } { { { \displaystyle { \widehat { \ell } } ^ { ( e ) } ( { \mathcal G } ^ { \mathcal E } ) = - \frac { n _ { e } } { 2 } \sum _ { j = 1 } ^ { q } \log \left| \widehat { \Sigma } _ { j } ^ { ( e ) } \right| - \frac { n _ { e } d } { 2 } \big ( 1 + \log ( 2 \pi ) \big ) } \ ~ } } \\ { { { \displaystyle ~ = - \sum _ { j = 1 } ^ { q } \log \left| \widehat { \Sigma } _ { j } ^ { ( e ) } \right| + \mathrm { c o n s t . } } } \ ~ } \end{array}\tag{14}
$$

## B.4.3 Existence of the MLE

For a Gaussian model, the MLE exists if and only if the partial covariance admits a positive-definite completion; see Uhler [70] for a detailed treatment. Let mlt $( \mathcal { G } ^ { \mathcal { E } } )$ denote the maximum-likelihood threshold, the smallest sample size at which the Gaussian MLE exists with probability one. Since the MLE above is computed block by block, existence reduces to a per-block condition. Write $F _ { j } : = \pi _ { j } \cup \mathrm { P a } _ { j }$ for the family of block $j$ and

$$
S _ { F _ { j } } ^ { ( e ) } = \left( { S _ { j , j } ^ { ( e ) } } ^ { ( e ) } \quad S _ { j , \mathrm { P a } _ { j } } ^ { ( e ) } \right) \in \mathbb { R } ^ { ( r _ { j } + s _ { j } ) \times ( r _ { j } + s _ { j } ) }
$$

for the corresponding family covariance. The estimate ${ \widehat \Sigma } _ { j } ^ { ( e ) }$ in (13) is the Schur complement of $S _ { \mathrm { P a } _ { j } , \mathrm { P a } _ { j } } ^ { ( e ) }$ in $S _ { F _ { i } } ^ { ( e ) }$ Hence $\widehat { \Sigma } _ { j } ^ { ( e ) } \succ 0$ if and only if $S _ { F _ { i } } ^ { ( e ) } \succ 0$ , which holds with probability one if and only if $n _ { e } > | F _ { j } | = r _ { j } + s _ { j }$ . The threshold of the model is therefore the size of its largest family,

$$
\mathrm { m l t } ( \mathcal { G } ^ { \mathcal { E } } ) \ : = \ : \operatorname* { m a x } _ { j } | F _ { j } | \ : = \ : \operatorname* { m a x } _ { j } ( r _ { j } + s _ { j } ) ,
$$

and the block estimates, the profile log-likelihood (14) and the BIC score exist as soon as $n _ { e } ~ \ge ~ \mathrm { m l t } ( \mathcal { G } ^ { \varepsilon } )$ Throughout, we assume that in every environment $e \in { \mathcal { E } }$ the family covariances $S _ { F _ { j } } ^ { ( e ) }$ have full rank, so that the MLE exists.

Since $\pi _ { j }$ and $\mathrm { P a } _ { j }$ are disjoint, ml $\operatorname { t } ( { \mathcal { G } } ^ { \mathcal { E } } ) \leq d ,$ , and typically mlt $( { \mathcal { G } } ^ { \varepsilon } ) \ll d .$ The model is therefore well defined even in high-dimensional regimes with $n _ { e } <$ d: there the pooled covariance $S ^ { ( e ) }$ is singular and the unrestricted Gaussian on all d coordinates has no MLE, yet every block residual covariance ${ \widehat \Sigma } _ { j } ^ { ( e ) }$ remains positive definite.

## B.4.4 Curved exponential family structure

Proof of Thm. ${ 1 6 } ,$ adapted from Drton and Eichler $\big [ 1 \gamma \big ]$ . The model is the Gaussian AMP chain-graph model whose chain components are the blocks, taken complete [3, 17]. It is a subfamily of the regular exponential family of centered Gaussians on $\mathbb { R } ^ { d }$ with positive definite covariance. Its parameter space $\Theta = \{ ( B _ { j } , \Sigma _ { j } ) _ { j = 1 } ^ { q }$ : $B _ { j } \in \mathbb { R } ^ { r _ { j } \times s _ { j } } , \ \Sigma _ { j } \succ 0 \}$ is an open subset of a Euclidean space of dimension dim(H). Let $B ( \theta ) \in \mathbb { R } ^ { d \times d }$ carry $B _ { j }$ in the rows of $\pi _ { j }$ and the columns of its parent blocks, and let $\bar { \Sigma } ( \theta )$ be block-diagonal with blocks $\Sigma _ { 1 } , \ldots , \Sigma _ { q }$ . In a topological order of $\mathcal { H } , B ( { \boldsymbol { \theta } } )$ is block strictly lower triangular, so $\dot { I } - B ( { \boldsymbol { \theta } } )$ is invertible, and the joint concentration is

$$
\boldsymbol { \psi } ( \boldsymbol { \theta } ) = \big ( I - B ( \boldsymbol { \theta } ) \big ) ^ { \top } \hat { \boldsymbol { \Sigma } } ( \boldsymbol { \theta } ) ^ { - 1 } \big ( I - B ( \boldsymbol { \theta } ) \big ) .
$$

The map $\psi$ is injective: from a covariance Σ in its image, the parameters are recovered by the population regressions $B _ { j } = \Sigma _ { \pi _ { j } , \mathrm { P a } _ { j } } \Sigma _ { \mathrm { P a } _ { j } , \mathrm { P a } _ { i } } ^ { - 1 }$ and $\Sigma _ { j } = \Sigma _ { \pi _ { j } , \pi _ { j } } - B _ { j } \Sigma _ { \mathrm { P a } _ { j } , \pi _ { j } }$ . Both $\psi$ and its inverse are rational with nonvanishing denominators on their domains, so ψ is a difeomorphism onto its image, which is therefore a smooth manifold of dimension dim(H). □

## B.5 Pooled BIC score

Throughout, H is a coarse DAG on $\Pi ^ { \varepsilon }$ with parent sets $\mathrm { P a } _ { j }$ . For a set of blocks $T , \widehat { \Sigma } _ { j } ^ { ( e ) } ( T )$ denotes the residual covariance (13) of the regression of $X _ { j }$ on the blocks in T, fitted on $\mathcal { D } ^ { ( e ) }$ , so that $\widehat { \Sigma } _ { j } ^ { ( e ) } = \widehat { \Sigma } _ { j } ^ { ( e ) } ( \mathrm { P a } _ { j } )$ . The regression parameters $( B _ { j } ^ { ( e ) } , \Sigma _ { j } ^ { ( e ) } )$ are fitted separately in each environment; only H is shared. We take $n _ { e }$ large enough for every family covariance to be full rank (App. B.4), so every term below is finite.

Plugging the profile log-likelihood (14) and dim(H) from Thm. 16 into (2) with $n = n _ { e }$ gives the per-environment score

$$
\mathrm { B I C } ( \mathcal { H } , \mathcal { D } ^ { ( e ) } ) : = 2 \hat { \ell } ^ { ( e ) } ( \mathcal { H } ) - \lambda \log ( n _ { e } ) \dim ( \mathcal { H } ) = \sum _ { j = 1 } ^ { q } \mathrm { B I C } _ { j } ^ { ( e ) } ( \pi _ { j } , \mathrm { P a } _ { j } ) - n _ { e } d \big ( 1 + \log ( 2 \pi ) \big ) ,\tag{15}
$$

with $\mathrm { B I C } _ { j } ^ { ( e ) }$ as in Def. 17. The constant does not depend on ${ \mathcal { H } } ,$ so it can be dropped. Summing over $\mathcal { E }$ and exchanging the finite sums over blocks and environments gives

$$
\sum _ { e \in { \mathcal { E } } } \operatorname { B I C } ( { \mathcal { H } } , { \mathcal { D } } ^ { ( e ) } ) = \operatorname { B I C } ( { \mathcal { H } } , { \mathcal { D } } ) + \operatorname { c o n s t } ,
$$

so the pooled BIC of Def. 17 is the sum of per-environment BICs. Decomposability is immediate, since each $\mathrm { B I C } _ { j } ^ { ( e ) }$ depends on H only through $( \pi _ { j } , \mathrm { P a } _ { j } )$

Proposition 31 (Consistency). Let $\mathcal { G } ^ { \varepsilon }$ be the true coarse $D A G$ over $\Pi ^ { \varepsilon }$ and $[ \mathcal { G } ^ { \varepsilon } ]$ its Markov equivalence class. Assuming each environment share the same $D A G ,$ for every coarse DAG H over $\mathrm { \ i } ^ { \varepsilon }$

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { P r } \bigl ( \mathrm { B I C } ( \mathcal G ^ { \mathcal E } , \mathcal D _ { n } ) > \mathrm { B I C } ( \mathcal H , \mathcal D _ { n } ) \bigr ) = 1 \qquad f o r \ e v e r y \ \mathcal H \not \in [ \mathcal G ^ { \mathcal E } ] .
$$

Proof. The coarse Markov and coarse faithfulness clauses of Ass. 9 make $\mathcal { G } ^ { \varepsilon }$ a perfect map of the block-level law $P ^ { ( e ) }$ in every environment, hence the inclusion-minimal I-map common to all of them. Consequently, any H that is an I-map of every environment yet lies outside $[ \mathcal { G } ^ { \varepsilon } ]$ has strictly larger dimension. Fix such a H and set

$$
\Delta _ { e } : = \ell ^ { ( e ) } ( \mathcal { G } ^ { \mathcal { E } } ) - \ell ^ { ( e ) } ( \mathcal { H } ) \geq 0 ,
$$

the population log-likelihood gap at the respective maximisers, with equality exactly when H is also an I-map of $P ^ { ( e ) }$ . As $n _ { e } \to \infty$ , the empirical law converges to $P ^ { ( e ) }$ , giving $\widehat { \ell } ^ { ( e ) } ( \mathcal { G } ^ { \hat { \mathcal { E } } } ) - \widetilde { \ell } ^ { ( e ) } ( \mathcal { H } ) \approx n _ { e } \Delta _ { e }$ . Using that dim(H) is common to all environments, this reads,

$$
\mathrm { B I C } ( \mathcal G ^ { \mathcal E } , \mathcal D ) - \mathrm { B I C } ( \mathcal H , \mathcal D ) \approx 2 \sum _ { e \in \mathcal E } n _ { e } \Delta _ { e } + \lambda \big ( \mathrm { d i m } ( \mathcal H ) - \mathrm { d i m } ( \mathcal G ^ { \mathcal E } ) \big ) \sum _ { e \in \mathcal E } \log n _ { e } .
$$

Two cases cover $\mathcal { H } \not \in [ \mathcal { G } ^ { \varepsilon } ]$ . If $\Delta _ { e ^ { \star } } > 0$ for some $e ^ { \star }$ , then H encodes an independence that $P ^ { ( e ^ { \star } ) }$ violates, the term $2 n _ { e ^ { \star } } \Delta _ { e ^ { \star } }$ is linear in $n _ { e ^ { \star } }$ , and it dominates the logarithmic penalty, so the diference tends to +∞. If instead $\Delta _ { e } = 0$ for every e, then H is an I-map of each environment and, being outside $[ \mathcal { G } ^ { \varepsilon } ]$ , satisfies dim $( \mathcal { H } ) > \dim ( \mathcal { G } ^ { \varepsilon } )$ the likelihood gap vanishes while the penalty term diverges, again to $+ \infty$ . In both cases the stated probability tends to one, so the pooled score prefers the true graph in the large-sample limit. □

Proposition 32 (Score equivalence). If H and $\mathcal { H } ^ { \prime }$ are Markov equivalent coarse DAGs on $\Pi ^ { \varepsilon }$ , then $\begin{array} { r } { \mathrm { B I C } ( \mathcal { H } , \mathcal { D } ) = } \end{array}$ $\mathrm { B I C } ( \mathcal { H } ^ { \prime } , \mathcal { D } )$

Proof. Markov equivalent DAGs impose the same block-level CI statements, so the models of Def. 15 on H and $\mathcal { H } ^ { \prime }$ coincide and attain the same maximized likelihood in every environment. Moreover, dim $\begin{array} { r } { ( \mathcal { H } ) = \sum _ { \pi _ { k } \to \pi _ { i } \in \mathcal { H } } r _ { j } r _ { k } + } \end{array}$ $\begin{array} { r } { \sum _ { j } r _ { j } ( r _ { j } + 1 ) / 2 } \end{array}$ depends only on the skeleton, which H and $\mathcal { H } ^ { \prime }$ share. Hence each summand of (15) agrees, and so does their sum over $\mathcal { E } .$ □

For a set of blocks $S ,$ let $\Sigma _ { \pi _ { j } | S } ^ { ( e ) }$ be the population residual covariance of $X _ { j } ^ { ( e ) }$ given $X _ { S } ^ { ( e ) }$ , and $\Sigma _ { \pi _ { j } , \pi _ { k } | S } ^ { ( e ) }$ the partial cross-covariance of $X _ { j } ^ { ( e ) }$ and $X _ { k } ^ { ( e ) }$ given $X _ { S } ^ { ( e ) }$ . For Gaussian P<sup>(e)</sup>, $X _ { \pi _ { j } } \perp \perp _ { P ^ { ( e ) } } X _ { \pi _ { k } } \mid X _ { S }$ holds if and only if $\Sigma _ { \pi _ { j } , \pi _ { k } | S } ^ { ( e ) } = 0 \ [ 3 8 ]$

Lemma 33 (Per-environment local consistency). Fix $e \in { \mathcal { E } } .$ , a block $\pi _ { j } { \mathrm { : } }$ , a set of blocks $S \nearrow \pi _ { j }$ and a block π<sub>k</sub> $\notin S \cup \{ \pi _ { j } \}$ , and let $\Delta ^ { ( e ) } : = \mathrm { B I C } _ { j } ^ { ( e ) } ( \pi _ { j } , S \cup \{ \pi _ { k } \} ) - \mathrm { B I C } _ { j } ^ { ( e ) } ( \pi _ { j } , S )$ . As $n _ { e }  \infty , \Delta ^ { ( e ) } \stackrel { p } {  } + \infty i f \Sigma _ { \pi _ { j } , \pi _ { k } | S } ^ { ( e ) } \neq 0$ and $\Delta ^ { ( e ) } \stackrel { p } {  } - \infty$ otherwise.

Proof. Write $S ^ { \prime } : = S \cup \{ \pi _ { k } \}$ . Both parent sets share the $r _ { j } ( r _ { j } + 1 ) / 2$ covariance parameters, and the regression block grows by $r _ { j } r _ { k }$ entries, so

$$
\Delta ^ { ( e ) } = n _ { e } \log \frac { \big | \widehat { \Sigma } _ { j } ^ { ( e ) } ( S ) \big | } { \big | \widehat { \Sigma } _ { j } ^ { ( e ) } ( S ^ { \prime } ) \big | } - \lambda \log ( n _ { e } ) r _ { j } r _ { k } .
$$

Since $S ^ { ( e ) } \stackrel { p } {  } \Sigma ^ { ( e ) }$ , the continuous mapping theorem gives $\widehat { \Sigma } _ { j } ^ { ( e ) } ( T ) \ \stackrel { p } {  } \ \Sigma _ { \pi _ { j } | T } ^ { ( e ) }$ for $T \in \{ S , S ^ { \prime } \}$ . By the Schur complement formula,

$$
\Sigma _ { \pi _ { j } | S ^ { \prime } } ^ { ( e ) } = \Sigma _ { \pi _ { j } | S } ^ { ( e ) } - \Sigma _ { \pi _ { j } , \pi _ { k } | S } ^ { ( e ) } \bigl ( \Sigma _ { \pi _ { k } | S } ^ { ( e ) } \bigr ) ^ { - 1 } \Sigma _ { \pi _ { k } , \pi _ { j } | S } ^ { ( e ) } ,
$$

so the population log-ratio is non-negative and vanishes exactly when $\Sigma _ { \pi _ { i } , \pi _ { k } | S } ^ { ( e ) } = 0$ . If it is positive, the likelihood term grows like $n _ { e }$ and dominates the ${ \mathcal { O } } ( \log n _ { e } )$ penalty. Otherwise, Wilks’ theorem for nested Gaussian regressions [71] gives $n _ { e }$ log $\big ( | { \widehat \Sigma } _ { j } ^ { ( e ) } ( S ) | / | { \widehat \Sigma } _ { j } ^ { ( e ) } ( S ^ { \prime } ) | \big ) \ \xrightarrow { d } \ \chi _ { r _ { j } r _ { k } } ^ { 2 }$ , so the likelihood term is $\mathcal { O } _ { p } ( 1 )$ while the penalty diverges. □

Proof of Prop. 18. By decomposability, the increment of $\mathrm { B I C } _ { j }$ is $\textstyle \sum _ { e \in { \mathcal { E } } } \Delta ^ { ( e ) }$ , with $n _ { e } = \gamma _ { e } n$ . If $\pi _ { j } \nmid { \mathbb { X } } \varepsilon \ \pi _ { k } \mid S ,$ at least one environment has $\Sigma _ { \pi _ { j } , \pi _ { k } | S } ^ { ( e ) } \neq 0$ and, by Lem. 33, contributes a term of order $n ,$ which outgrows the finitely many terms of order log n from the others. If $\pi _ { j }$ $\varepsilon \ \pi _ { k } \mid S$ , every $\Delta ^ { ( e ) }$ diverges $\mathrm { t o \ - \infty }$ , and so does their finite sum. □

## B.6 Estimation and guarantees

## B.6.1 Partition estimation and refinement

Algorithm 2 Compute interventional descendants matrix; adapted from 39   
Require: data $\mathcal { D } = \{ X ^ { ( e ) } \} _ { e \in \mathcal { E } } ,$ threshold α, Test   
Ensure: interventional descendants matrix $M \in \{ 0 , 1 \} ^ { | V | \times ( | \mathcal { E } | - 1 ) }$   
1: for $( v , e ) \in V \times ( \mathcal { E } \setminus \{ 0 \} )$ do ▷ in parallel   
2: $p _ { v , e }  \mathrm { T e s t } \big ( X _ { v } ^ { ( e ) } , X _ { v } ^ { ( 0 ) } \big )$ ▷ Welch-t, LRT, etc.   
3: $M _ { v , e } \gets \mathbb { 1 } [ p _ { v , e } < \alpha ]$   
4: end for   
5: return M

Algorithm 3 Refine via interventional descendant patterns; adapted from 39   
Require: partition Π, interventional descendant matrix M   
Ensure: pair $( \pi ^ { * } , \{ \pi _ { a } , \pi _ { b } \} )$ or (∅, ∅)   
1: for $\pi \in \Pi$ with $| \pi | > 1$ do   
2: pick $u \in \pi$   
3: $\overline { { \pi } } _ { a }  \{ v \in \pi \mid M _ { v , \cdot } = M _ { u , \cdot } \}$   
4: $\pi _ { b }  \pi \setminus \pi _ { a }$   
5: if $\pi _ { b } \neq \varnothing$ then return $( \pi , \{ \pi _ { a } , \pi _ { b } \} )$   
6: end if   
7: end for   
8: return (∅, ∅)

## B.6.2 Full COARSE procedure

Each block is searched over its estimated parent pool $\widehat { \mathrm { p a } } ^ { \star } ( \pi ) : = \{ \pi ^ { \prime } \in \widehat { \Pi } ^ { \varepsilon } : \widehat { \mathrm { s u p p } } ( \pi ^ { \prime } ) \subseteq \widehat { \mathrm { s u p p } } ( \pi ) \}$ . The pools are read of the supports alone, so no actual linear extension of ${ < } _ { \mathrm { s u p p } }$ needs to be computed.

Algorithm 4 coarse   
Require: data $\mathcal { D } = \{ X ^ { ( e ) } \} _ { e \in \mathcal { E } } ,$ threshold α, Test, penalty λ   
Ensure: estimated coarse DAG G<sup>ˆ</sup>   
1: $\widehat { M } \gets$ InterventionalDescendants $( \mathcal { D } , \alpha ,$ Test) ▷ Alg. 2   
2: $\widehat { \Pi } ^ { \varepsilon } \gets \{ V \}$   
3: repeat   
4: $( \pi ^ { * } , \{ \pi _ { a } , \pi _ { b } \} )  \mathrm { R E F I N E } ( \widehat { \Pi } ^ { \varepsilon } , \widehat { M } )$ ▷ Alg. 3   
5: if $\pi ^ { * } \neq \varnothing$ then $\widehat { \Pi } ^ { \varepsilon } \gets ( \widehat { \Pi } ^ { \varepsilon } \setminus \{ \pi ^ { * } \} ) \cup \{ \pi _ { a } , \pi _ { b } \}$   
6: end if   
7: until $\pi ^ { * } = \varnothing$   
8: for each $\pi \in { \widehat { \Pi } } ^ { \varepsilon }$ do   
9: $\widehat { \mathrm { s u p p } } ( \pi )  \{ e \in \mathcal { E } \setminus \{ 0 \} : \widehat { M } _ { v , e } = 1 \}$ for any $v \in \pi$   
10: end for   
11: for each $\pi _ { j } \in { \widehat { \Pi } } ^ { \varepsilon }$ do ▷ any order   
12: $\widehat { \mathrm { p a } } ^ { \star } ( \pi _ { j } ) \gets \{ \pi _ { k } \in \widehat { \Pi } ^ { \mathcal { E } } : \widehat { \mathrm { s u p p } } ( \pi _ { k } ) \subsetneq \widehat { \mathrm { s u p p } } ( \pi _ { j } ) \}$   
13: $\widehat { \mathrm { P a } } _ { j } \gets \mathrm { G R O W S H R I N K } \left( \boldsymbol { \pi } _ { j } , \widehat { \mathrm { p a } } ^ { \star } ( \boldsymbol { \pi } _ { j } ) , \mathrm { B I C } _ { j } \right)$ ▷ Alg. 1   
14: end for   
15: return $\hat { \mathcal { G } } = \big ( \widehat { \Pi } ^ { \mathcal { E } } , \{ \pi _ { k } \to \pi _ { j } : \pi _ { k } \in \widehat { \mathrm { P a } } _ { j } \} \big )$

## B.6.3 Consistency of coarse

At block level, the Markov boundary of $\pi _ { j }$ relative to a set of blocks $Z ,$ written $B ( \pi _ { j } , Z )$ , is the inclusion-minimal $S \subseteq Z$ with $\pi _ { j } \perp \perp _ { \varepsilon } Z \setminus S \mid S$

Lemma 34 (Block-level compositional graphoid). $\perp \perp _ { \mathcal { E } }$ is a compositional graphoid on $\Pi ^ { \varepsilon }$ . Hence $B ( \pi _ { j } , Z )$ exists and is unique for every block $\pi _ { j }$ and every $Z \subseteq { \dot { \Pi } } ^ { \varepsilon } \setminus \{ \pi _ { j } \}$

Proof. Each $P ^ { ( e ) }$ is Gaussian with positive definite covariance, so its CI relation is a compositional graphoid on $V .$ . Restricting it to unions of blocks preserves the axioms. The intersection over $e \in { \mathcal { E } }$ is again a compositiona graphoid: whenever the premises of an axiom hold in every $P ^ { ( e ) }$ , so does its conclusion. The second claim follows from Thm. 22. □

Proof of Thm. 19. Under oracle tests, ${ \mathrm { A l g } } .$ 4 runs with $\Pi ^ { \varepsilon }$ and the pools $\mathrm { p a } ^ { \star } ( \cdot )$ . Fix a block $\pi _ { j }$ and let $Z _ { j } : =$ $\mathrm { p a } ^ { \star } ( \pi _ { j } )$ . By Cor. 14(ii), $\mathrm { P a } _ { j }$ is the Markov boundary of $\pi _ { j }$ relative to $Z _ { j }$ in every $P ^ { ( e ) }$ . Hence $\pi _ { j } \perp \perp _ { \varepsilon } Z _ { j } \langle \mathrm { P a } _ { j } | \mathrm { P a } _ { j }$ while no proper subset of $\mathrm { P a } _ { j }$ has this property, so Pa is the Markov boundary of $\pi _ { j }$ relative to $Z _ { j }$ under $\perp \perp _ { \mathcal { E } } .$ unique since $\perp \perp _ { \mathcal { E } }$ is a compositional graphoid (Lem. 34). By Prop. 18, every increment of $\mathrm { B I C } _ { j }$ has, in the limit, the sign of the matching $\perp \perp _ { \mathcal { E } }$ statement, so the score-based GrowShrink returns $\mathrm { P a } _ { j }$ with probability tending to one (Lem. 23). A union bound over the $q$ blocks gives $\mathbb { P } ( \widehat { \mathrm { P a } } _ { j } = \mathrm { P a } _ { j } \ \forall j )  1$ , and on this event the edges of $\hat { \mathcal G }$ are exactly $\pi _ { k } \to \pi _ { j }$ with $\pi _ { k } \in \operatorname { p a } _ { \mathcal { G } ^ { \varepsilon } } ( \pi _ { j } )$ , that is $\hat { \mathcal { G } } = \mathcal { G } ^ { \mathcal { E } }$ . Finally, each GrowShrink call depends only on $\pi _ { j } ,$ $\mathrm { p a } ^ { \star } ( \pi _ { j } )$ and $\mathrm { B I C } _ { j }$ , and not on the order in which blocks are processed. □

## B.7 Worst-case complexity

Proof of Thm. 20. (1) For each $e \neq 0 ,$ the d Welch statistics need the means and variances of $X ^ { ( e ) }$ and $X ^ { ( 0 ) }$ , at cost $O ( d ( n _ { e } + n _ { 0 } ) )$ , so ${ \mathcal { O } } ( d n )$ in total. (2) Each refinement pass compares at most d rows of $\hat { M }$ to a representative, and there are at most $q - 1$ splits: $O ( q d | \mathcal { E } | )$ . (3) Support inclusion for every ordered pair of blocks costs $\mathcal { O } ( q ^ { 2 } | \mathcal { E } | )$ . (4) Centering costs ${ \mathcal { O } } ( d n )$ and the |E| sample covariances $S ^ { ( e ) }$ cost $\mathcal { O } ( d ^ { 2 } n )$ ; they are computed once and cached. (5) Each grow sweep that does not end the grow phase adds a block, so the phase makes at most $| Z | + 1$ sweeps of at most $| Z |$ probes, whatever the scores; the shrink phase is symmetric. With $Z = \widehat { \mathrm { p a } } ^ { \star } ( \pi _ { j } )$ this gives $\mathcal { O } ( q ^ { 2 } )$ probes per block. Each probe slices $S _ { F _ { j } } ^ { ( e ) }$ in every environment and computes a Cholesky factorization, a triangular solve, an $r _ { j } \times r _ { j }$ Schur complement and a log-determinant, for $\mathcal { O } ( ( r _ { j } + s _ { j } ) ^ { 3 } ) = \mathcal { O } ( d ^ { 3 } )$ per environment. Over the $q$ blocks this gives $\mathcal { O } ( | \mathcal { E } | q ^ { 3 } d ^ { 3 } )$ . Since $q \leq d$ and $| \mathcal { E } | \leq n$ , phases (1)–(3) are dominated by $\mathcal { O } ( d ^ { 2 } n )$ , and the total is

$$
\mathscr { O } ( d n ) + \mathscr { O } ( q d | \mathcal { E } | ) + \mathscr { O } ( q ^ { 2 } | \mathcal { E } | ) + \mathscr { O } ( d ^ { 2 } n ) + \mathscr { O } ( | \mathcal { E } | q ^ { 3 } d ^ { 3 } ) = \mathscr { O } \big ( d ^ { 2 } n + | \mathcal { E } | q ^ { 3 } d ^ { 3 } \big ) .
$$

## C ADDITIONAL EXPERIMENTS

We now present additional experiments that test our theoretical results and assess robustness to violations of our assumptions. Each run used a single CPU core on an Apple M3 Max (36 GB RAM).

## C.1 Experimental details

Synthetic data. ER graphs are sampled with sempler with expected degree $\rho ( d - 1 )$ . Scale-free (SF) graphs are Barab´asi–Albert graphs in which each new node attaches to max{1, round $( \rho ( d - 1 ) / 2 ) \}$ existing nodes, matching the expected degree of the ER graph with the same $\rho .$ Edge weights have magnitude uniform in [0.5, 2] and independent random signs. Observational noise means are uniform in [−2, 2] and variances uniform in [0.5, 2]. The model is standardized in causal order to unit observational variance [50] while interventional environments are not re-standardized. Each interventional environment shifts its targets by independent noise with mean 2 and variance 1. With $k = 1$ target per environment, targets are drawn without replacement, with $k > 1$ $\left( \mathrm { A p p . ~ C . 7 } \right)$ each environment draws k targets and targets may repeat across environments. For non-Gaussian families $\left( \mathrm { A p p . ~ C . 9 } \right)$ , the noise of each variable is rescaled to the mean and variance of its Gaussian counterpart.

Pipeline, cross-validated and oracle runs. In the scalability $\left( \mathrm { S e c . ~ 4 . 2 } \right)$ and density (App. C.3) studies, we run the full pipeline of Alg. 4 with a fixed $\alpha = 1 0 ^ { - 4 }$ . All runtimes are wall-clock times of one fit on one

CPU core (Apple M3 Max, 36 GB RAM). While in the multiple testing $\left( \mathrm { { A p p . ~ C . 6 } } \right)$ and multi-target $\left( \mathrm { { A p p . ~ C . 7 } } \right)$ experiments, we choose α from $\{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 0 . 0 5 , 0 . 1 \}$ by 10-fold cross-validation, maximizing the held-out Gaussian log-likelihood of the fitted coarse model. In the remaining experiments, namely the comparison with RePaRe (Sec. 4.3) and the sensitivity analyses for the penalty multiplier $\lambda ~ ( \mathrm { A p p . ~ C . 5 } )$ , noise (App. C.9), and dimensionality $\left( \mathrm { A p p . ~ C . 8 } \right)$ , all methods are given the true partition, so no threshold is needed.

Causal chamber. We use the lt interventions standard v1 dataset of the light tunnel: the actuators R, G, B, the current ${ \tilde { C } } ,$ the infrared and visible sensors $\tilde { I } _ { 1 : 3 } , \tilde { V } _ { 1 : 3 }$ , the polarizer actuators $\theta _ { 1 , 2 }$ and angle sensors $\ddot { \theta } _ { 1 , 2 } ,$ and the six lens readings $L _ { 1 1 } , \ldots , L _ { 3 2 }$ . The observational dataset has $1 0 ^ { 4 }$ samples and each interventional dataset $1 0 ^ { 3 }$ . In the ungrouped mode each of the five interventional datasets is one environment $( m = 5 )$ ; the true coarse DAG has 8 blocks and 10 edges, which expand to 39 fine-grained edges. In the grouped mode the datasets are pooled into two environments, one for the colour actuators and one for the polarizers $( m = 2 )$ ; the true coarse DAG has 4 blocks and 4 edges. Since physical interventions may change the spread of a variable as well as its mean, on these data coarse and RePaRe build M<sup>ˆ</sup> with a Gaussian likelihood-ratio test for a change in mean or variance.

Hyperparameters are selected with diferent approaches. For coarse we select α by 10-fold cross-validation with λ = 1 fixed. RePaRe searches its partition and edge threshold over the same grid but uses the GnIES score on the expanded DAG, as in Madaleno et al. [39]. Importantly, this means that RePaRE actually consumes the intervention target at selection time, while coarse does not. GIES and GnIES run with their default BIC score, and UT-IGSP selects its two test levels by its own score. Oracle rows select the grid cell with the highest ARI, then the highest F-score (F-score alone for UT-IGSP).

Negative controls [53] For a coarse method, a null draw is a random DAG on the estimated blocks with the same number of edges, expanded to the variable level. For a fine-grained method, a null draw is a random DAG on the 20 variables with the same number of adjacencies, mapped to the type of equivalence class the method returns. Each draw is scored at both levels; we report the mean and the 2.5% and 97.5% quantiles over 1000 draws. For adjacencies we also use the exact null, under which the number of true adjacencies among the estimated ones is hypergeometric.

## C.2 Finite-sample variants

In finite samples a fixed α over dm tests can split or merge blocks, invalidating the procedure, and large blocks inflate the model parameters, making the estimation procedure more complex and data hungry. We address both these issues through three variants of the procedure:

1. CV-coarse: Multiple-testing corrections control false shifts, not partition accuracy, and tend to be conservative as d grows. CV-coarse selects α from a grid by K-fold cross-validated held-out Gaussian log-likelihood of the fitted coarse model, without accessing the ground truth.

2. kPC-coarse: The block model is unrestricted, so each block costs $r _ { j } s _ { j } + r _ { j } ( r _ { j } + 1 ) / 2$ parameters. kPCcoarse replaces each block by its first $k _ { j } : = \operatorname* { m i n } \{ k , r _ { j } \}$ principal components, estimated in the observational environment and applied to all others, and scores with $k _ { j }$ in place of $r _ { j }$ . The family size drops from $\boldsymbol { r } _ { j } + \boldsymbol { s } _ { j }$ to $\begin{array} { r } { k _ { j } + \sum _ { \pi _ { k } \in \mathrm { P a } _ { j } } k _ { k } } \end{array}$

3. avg-coarse: Here we summarize each block by the mean of its standardized variables. In this way the score runs on q scalar Gaussian nodes.

## C.3 Graph density and graph family

Fig. 2 and Fig. 3 repeat the study of Sec. 4.2 at d = 10 for ER and SF graphs, with $\rho \in \{ 0 . 2 , 0 . 5 , 0 . 8 \} , m \in \{ 2 , 5 , 8 \}$ and $n _ { e }$ from $1 0 ^ { 2 }$ to $1 0 ^ { 5 }$ . On ER graphs, the median F-score reaches 1 faster than ARI in almost any setting, suggesting that constraint-based partition recovery is the bottleneck of the procedure. Denser graphs and more environments need more samples: more environments give more blocks to separate, and each adds d tests to $\hat { M }$ The hardest setting, $\rho = 0 . 8$ and $m = 8 ,$ , starts from a median ARI of only 0.33 at $n _ { e } = 1 0 ^ { 2 }$ . SF graphs converge faster: both medians reach 1 by $n _ { e } = 2 \times 1 0 ^ { 3 }$ in every setting.

![](images/228189c5863545011a48773c0a013b32e047469794f94b1af8694af9b8646e83.jpg)  
(a) ARI, m = 2

![](images/bdca819d1b1e0f7e956cdbce51e44512ae34c43c5125cdf993d4bef306f69a4a.jpg)  
(b) ARI, m = 5

![](images/522bfb7d1d8985c1dd2c619f9da4d84070723aa1e71bd6b641f88a1116647a7d.jpg)  
(c) ARI, m = 8

![](images/9221552f496ceb064285a15faa9e5b57d4a467c400af2e2438d1c3e38fdc317e.jpg)  
(d) F-score, m = 2

![](images/f4693a11ae06410d4c9e0ae86baf20f4a7f7d647627533275a305ee2f1069bcb.jpg)  
(e) F-score, m = 5

![](images/b0392aaa638748cd839114fa8d372bf7248e1495bb1b3806b1ba4f2ff2060c1a.jpg)  
(f) F-score, m = 8  
Figure 2: Full coarse pipeline on ER graphs, $d = 1 0 ;$ one curve per density ρ. Denser graphs and more environments delay convergence, but every setting converges.

## C.4 Comparison against RePaRe: full grid

Tab. 3 completes Tab. 1 with every sample size of the grid.

## C.5 Penalty multiplier

Fig. 4 reports an ablation on the penalty multiplier λ. We run the edge stage of coarse under the oracle partition on ER graphs with $d = 5 0 , m = 5 , \rho \in \{ 0 . 2 , 0 . 5 \}$ and $n _ { e } \in \{ 1 0 ^ { 2 } , 5 \times \overset { \vartriangle } { 1 0 ^ { 2 } } , \overset { \vartriangle } { 1 0 ^ { 3 } } , 5 \times 1 0 ^ { 3 } \}$ , for $\lambda \in \{ 0 . 2 5 , 0 . 5 , 1 , 2 , 4 \}$ Precision is at least 0.93 in every cell and at least 0.99 for $\lambda \geq 0 . 5$ , so λ mainly afects recall. At $n _ { e } = 1 0 ^ { 2 }$ , the standard BIC $( \lambda = 1 )$ already misses some edges (F-score 0.84 and 0.87), λ = 2 misses between a third and a half of them, and λ = 4 misses most. From $n _ { e } = 5 \times 1 0 ^ { 3 }$ on, every λ reaches an F-score of at least 0.99. Consistently with Prop. 18, the choice of λ matters only at small $n _ { e }$ . There the score is still far from its asymptotic regime, the number of parameters is large relative to the sample size, and the penalty can outweigh the likelihood gain of true edges. In these cases a lighter penalty is the better choice. Across the regimes we tested, λ = 1 is a good default for coarse.

## C.6 Cutof rules for the descendant test

In order to build ${ \hat { M } } , \ \mathrm { A l g . ~ 2 }$ thresholds d · m Welch p-values at a fixed level α. Fig. 5 compares this choice with rules that set the cutof from the data, on one shared p-value matrix per dataset. We tested several well known statistical correction at $\delta = 0 . 0 5$ 5:

• Bonferroni [18]: uses $\alpha = \delta / ( d m )$ and control the probability of at least one false shift under any dependence between tests.

• Benjamini–Hochberg (BH) [7]: uses $\alpha = k ^ { \star } \delta / ( d m )$ , with $k ^ { \star } : = \operatorname* { m a x } \{ i : p _ { ( i ) } \\leq i \delta / ( d m ) \}$ } and bounds the false discovery rate by δ, provided the p-values are independent or positively dependent.

![](images/ad7cabfdbb953a4591f2e39169ea0623b82012e2b35b3371d74a2876de88f8c1.jpg)

![](images/135acb05cb9fc171f39c807c46d2b58394475a019771450b24c55180a705b6f0.jpg)  
(a) ARI, m = 2  
(b) ARI, m = 5

![](images/c1d2aee00c41c5a30d0e96153e5836910d28889570b84f59f7a10b25919c67b0.jpg)  
(c) ARI, m = 8

![](images/667f094f36b1ec89f3228f78bdd2222af4c0bffad41ee80bc1e3ad24d4a8ee42.jpg)  
(d) F-score, m = 2

![](images/9026ee500f2cb2c025caed0472b1236bdd6a96b3f97be7a6e104881da0de2fef.jpg)  
(e) F-score, m = 5

![](images/4bbc8b42eeac272e77ccf96f3c56d635528a7b082ff72e0db947bdb700d6de0c.jpg)  
Figure 3: As Fig. 2, on scale-free graphs.  
(f) F-score, m = 8

• Benjamini–Yekutieli (BY) [8]: which extends BH by keeping the false discovery rate below δ under any dependence.

And compare them against both our implementation of CV-coarse over the α grid of $\mathrm { A p p }$ . C.1 and the oracle α (picked as the grid value with the highest ARI on that dataset). For the study we used ER graphs, $d = 2 5 0$ $m = 5 , \rho \in \{ 0 . 2 , 0 . 5 , 0 . 8 \} , n _ { e } \in \{ 1 0 ^ { 4 } , 1 0 ^ { 5 } \}$ , 20 seeds and scored only the partition recovery stage.

The ARI of a fixed cutof rises slowly from $\alpha = 1 0 ^ { - 8 }$ to a maximum near $\alpha = 1 0 ^ { - 2 }$ and collapses beyond it, where false shifts split true blocks. Bonferroni sits at $\alpha \approx 4 \times 1 0 ^ { - 5 }$ and loses the most (median gap to the oracle −0.10 to −0.15 at $n _ { e } = 1 0 ^ { 4 } )$ ; BH and BY land in the flat region and lose at most 0.07. CV-coarse, the only rule that needs neither a level nor the truth, is within 0.04 of the oracle median at $n _ { e } = 1 0 ^ { 4 }$ and within 0.02 at $n _ { e } = 1 0 ^ { 5 } ~ \mathrm { ( F i g . ~ 6 ) }$ . The fixed $\alpha = 1 0 ^ { - 4 }$ of Sec. 4.1 is conservative at this d: 0.43 against 0.55 for the oracle at $\rho = 0 . 2 , n _ { e } = 1 0 ^ { 4 }$ , and 0.72 against 0.79 at $n _ { e } = 1 0 ^ { 5 }$ . This is the loss Sec. 4.2 attributes to the partition stage, and CV recovers most of it.

## C.7 Multi-target interventions

Sec. 3.1 defines $\Pi ^ { \varepsilon }$ through signatures and shows that it is coarser than $\Pi ^ { \mathcal { I } }$ [39] under multi-target interventions (App. B.3.2). We quantify the gap at the population level and check that coarse estimates $\Pi ^ { \bar { \varepsilon } }$ , not $\Pi ^ { \mathcal { I } }$

Population level. On ER graphs with $d = 5 0$ and $\rho \in \{ 0 . 2 , 0 . 5 , 0 . 8 \}$ (200 seeds) we compute both partitions from the graph and the targets and report their ARI, the block ratio $| \dot { \Pi } ^ { \mathcal { E } } | / | \Pi ^ { \mathcal { I } } |$ and the hidden-edge fraction, the share of true coarse edges of $\mathcal { G } ^ { \mathbb { Z } }$ that fall inside one block of $\Pi ^ { \varepsilon }$ . In the native design (Fig. 7) each of m environments draws k targets. In the overlap design (Fig. 8) the targets of all environments come from a fixed pool of 20 nodes: a cover of the pool by disjoint k-subsets plus random k-subsets, up to m environments.

In the native design the resolution falls quickly with k: at $k = 2$ the ARI is 0.38–0.75 and at $k = 5$ below 0.26, with $\Pi ^ { \varepsilon }$ holding a quarter to a half of the blocks of $\Pi ^ { \mathcal { I } }$ . More environments lower the ARI further, since every new multi-target environment refines $\Pi ^ { \mathcal { I } }$ more than $\Pi ^ { \varepsilon }$ , but they lower the hidden-edge fraction, from 0.40–0.61 at $m = 2$ to $0 . 1 8 \mathrm { - } 0 . 3 3 $ at $m = 2 0$ for $k = 5$ . The overlap design shows the other side: with a fixed target pool, adding overlapping environments recovers resolution, from ARI 0.56 at $m = 1 0$ to 0.86 at $m = 4 0$ for $k = 2$ $( \rho = 0 . 2 )$ , with the hidden-edge fraction dropping towards zero. What the interventions can resolve is set by how their target sets intersect, not by how many there are.

Table 3: Edge recovery under the oracle partition, ER graphs, $\rho = 0 . 2 , m = 5 ,$ all sample sizes. F-score as mean ± sample standard deviation over 20 seeds; speed-up is the ratio of mean RePaRe to mean coarse runtime. Bold marks the better method where the two difer at the reported precision.
<table><tr><td>d</td><td>10</td><td>20</td><td>50</td><td>100</td><td>200</td><td>300</td><td>500</td></tr><tr><td colspan="8">F-score ↑, coarse</td></tr><tr><td> $n _ { e } = 1 0 ^ { 3 }$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> ${ \bf 1 . 0 0 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 9 9 \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 9 8 \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 9 5 \ : \pm { \ : 0 . 0 5 } }$ </td><td> $0 . 8 4 \pm 0 . 0 8$ </td><td> $0 . 7 1 \pm 0 . 0 5$ </td></tr><tr><td> $n _ { e } = 5 \times 1 0 ^ { 3 }$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> ${ \bf 1 . 0 0 \pm 0 . 0 1 }$ </td><td> $\mathbf { 1 . 0 0 \ : \pm { \ : 0 . 0 0 } }$ </td><td> ${ \bf 1 . 0 0 \pm 0 . 0 1 }$ </td><td> ${ \bf 1 . 0 0 \pm 0 . 0 1 }$ </td><td> ${ \bf 1 . 0 0 \pm 0 . 0 1 }$ </td></tr><tr><td> $n _ { e } = 1 0 ^ { 4 }$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $\mathbf { 1 . 0 0 \ : \pm { \ : 0 . 0 0 } }$ </td><td> ${ \bf 1 . 0 0 \pm 0 . 0 1 }$ </td><td> $\mathbf { 1 . 0 0 \ : \pm { \ : 0 . 0 0 } }$ </td><td> ${ \bf 1 . 0 0 \pm 0 . 0 1 }$ </td></tr><tr><td> $n _ { e } = 5 \times 1 0 ^ { 4 }$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $\mathbf { 1 . 0 0 \ : \pm { \ : 0 . 0 0 } }$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td> $n _ { e } = 1 0 ^ { 5 }$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td colspan="8">F-score ↑, RePaRe</td></tr><tr><td> $n _ { e } = 1 0 ^ { 3 }$ </td><td> $1 . 0 0 \pm 0 . 0 1$ </td><td> $0 . 9 6 \pm 0 . 0 3$ </td><td> $0 . 9 5 \pm 0 . 0 5$ </td><td> $0 . 9 3 \pm 0 . 0 7$ </td><td> $0 . 9 5 \pm 0 . 0 8$ </td><td> ${ \bf 0 . 9 5 \pm 0 . 1 0 }$ </td><td> $\mathbf { 0 . 9 7 \pm 0 . 0 4 }$ </td></tr><tr><td> $n _ { e } = 5 \times 1 0 ^ { 3 }$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 1$ </td><td> $0 . 9 9 \pm 0 . 0 2$ </td><td> $0 . 9 7 \pm 0 . 0 4$ </td><td> $0 . 9 7 \pm 0 . 0 5$ </td><td> $0 . 9 7 \pm 0 . 0 7$ </td><td> $0 . 9 9 \pm 0 . 0 3$ </td></tr><tr><td> $n _ { e } = 1 0 ^ { 4 }$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 1$ </td><td> $0 . 9 8 \pm 0 . 0 3$ </td><td> $0 . 9 9 \pm 0 . 0 4$ </td><td> $0 . 9 8 \pm 0 . 0 6$ </td><td> $0 . 9 9 \pm 0 . 0 2$ </td></tr><tr><td> $n _ { e } = 5 \times 1 0 ^ { 4 }$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 9 9 \pm 0 . 0 2$ </td><td> $1 . 0 0 \pm 0 . 0 1$ </td><td> $1 . 0 0 \pm 0 . 0 1$ </td></tr><tr><td> $n _ { e } = 1 0 ^ { 5 }$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 9 9 \pm 0 . 0 1$ </td><td> $1 . 0 0 \pm 0 . 0 1$ </td></tr><tr><td colspan="8">Speed-up ↑ (RePaRe runtime / coarse runtime)</td></tr><tr><td> $n _ { e } = 1 0 ^ { 3 }$ </td><td>9×</td><td>21×</td><td>38×</td><td>44×</td><td>37×</td><td>37×</td><td>32×</td></tr><tr><td> $n _ { e } = 5 \times 1 0 ^ { 3 }$ </td><td>25×</td><td>43×</td><td>84×</td><td>108×</td><td>101×</td><td>94×</td><td>63×</td></tr><tr><td> $n _ { e } = 1 0 ^ { 4 }$ </td><td>36×</td><td>66×</td><td>136×</td><td>146×</td><td>169×</td><td>128×</td><td>101×</td></tr><tr><td> $n _ { e } = 5 \times 1 0 ^ { 4 }$ </td><td>54×</td><td>104×</td><td>282×</td><td>310×</td><td>372×</td><td>337×</td><td>221×</td></tr><tr><td> $n _ { e } = 1 0 ^ { 5 }$ </td><td>51×</td><td>172×</td><td>342×</td><td>510×</td><td>509×</td><td>424×</td><td>329×</td></tr></table>

Estimation. Fig. 9 runs coarse with cross-validated α on ER graphs with $d = 5 0 , m = 1 0 , k \in \{ 1 , \ldots , 5 \}$ targets per environment and $n _ { e } \in \{ 1 0 ^ { 3 } , 1 0 ^ { 4 } , 1 0 ^ { 5 } \} \ ( 2 0 \ \mathrm { s e e d s } )$ , scoring Π against both Π<sup>ˆ</sup> <sup>E</sup> and Π<sup>I</sup>. ARI(Π<sup>ˆ</sup>, Π<sup>E</sup>) is (almost) flat in k and increases with $n _ { e } ,$ reaching 0.85–0.92 at $n _ { e } = 1 0 ^ { 5 }$ , whereas $\mathrm { A R I } ( \hat { \Pi } , \Pi ^ { \mathcal { I } } )$ decreases with k and, for $n _ { e } \ge 1 0 ^ { 4 }$ , follows the population curve $\mathrm { A R I } ( \Pi ^ { \varepsilon } , \Pi ^ { \varepsilon } )$ . Hence the estimate converges to $\Pi ^ { \varepsilon }$ , the object Def. 8 identifies, and its distance to $\Pi ^ { \mathcal { I } }$ reflects the population gap rather than estimation error.

## C.8 High-dimensional regime: reduced-rank variants

The block MLE of Def. 15 exists only when $n _ { e }$ exceeds the family size $| F _ { j } | = r _ { j } + s _ { j } \ ( \mathrm { A p p . \ B . 4 } )$ , which fails when blocks are large and samples few. Two variants reduce the dimension of every block before scoring; both change only the score stage of Alg. 4. After centering, every environment is divided by the per-variable standard deviation of the observational environment, so that the reduction is the same linear map in every environment.

kPC-coarse replaces each block $X _ { j } ^ { ( e ) } \in \mathbb { R } ^ { r _ { j } }$ by $Z _ { j } ^ { ( e ) } : = V _ { j } ^ { \top } X _ { j } ^ { ( e ) } \in \mathbb { R } ^ { k _ { j } }$ , where $V _ { j } ~ \in ~ \mathbb { R } ^ { r _ { j } \times k _ { j } }$ holds the top $k _ { j } : = \operatorname* { m i n } \{ k , r _ { j } \}$ right singular vectors of the observational data matrix of block $\pi _ { j } .$ , that is, its first $k _ { j }$ principal directions, estimated once in the observational environment and applied unchanged to every other one. avgcoarse is the rank-one case $V _ { j } : = \mathbb { 1 } / r _ { j }$ , the mean of the scaled variables of the block. The pooled BIC of Def. 17 and the grow-shrink search are then run on the reduced blocks, with $k _ { j }$ in place of $r _ { j }$ . Consequently, the family size of block j drops from $\boldsymbol { r } _ { j } + \boldsymbol { s } _ { j }$ to $\begin{array} { r } { k _ { j } + \sum _ { \pi _ { k } \in \mathrm { P a } _ { j } } k _ { k } } \end{array}$ , so the reduced score exists for much smaller $n _ { e } .$

We compare coarse with 1PC-, 3PC- and avg-coarse under the oracle partition on ER graphs with $d = 1 0 0 .$ $\rho = 0 . 1 , m \in \{ 5 , 1 0 , 2 0 \}$ and $n _ { e }$ from 10 to $1 0 ^ { 3 }$ (20 seeds).

Fig. 10 shows the edge recovery performance of each method as the sample size moves from the high-dimensional to the classical statistical regime. Fig. 11 explores a similar setting, but focuses on how many samples are needed to recover the correct parent set. For every block with at least one parent, it reports whether the parent set is recovered exactly and the fraction of parents recovered, as a function of $n _ { e } / F _ { j }$ , where $F _ { j }$ is the number of variables in the block and its true parents. Fig. 12 reports the share of blocks whose family fits in $n _ { e }$ samples, measured in each method’s own units.

![](images/ede4cf6376e5a0891aa9db7ef16e93a2807a65160ed5f5a18d4d9e2887f8a757.jpg)  
(a) $\rho = 0 . 2$

![](images/67a9ada61e6ce45da86fed3989fff940883052621643abed0d9674c3c63eecd5.jpg)  
(b) $\rho = 0 . 5$  
Figure 4: Edge F-score under the oracle partition against the BIC penalty multiplier, ER graphs, $d = 5 0$ $m = 5 ;$ one curve per $n _ { e } ,$ , dashed line at the standard BIC. Precision stays above 0.99 for $\lambda \geq 0 . 5$ , so the loss at large λ and small $n _ { e }$ is recall.

For $n _ { e } \leq 3 0 $ , few families of coarse are feasible (degenerate fits score −∞) and its F-score is only 0.17–0.44, while the reduced variants already recover a large part of the edges (F-score 0.4–0.8), with 3PC the best variant throughout. coarse overtakes $\mathrm { 3 P C }$ at $n _ { e } \approx 1 0 0$ for $m = 2 0$ and at $n _ { e } \approx 5 0 0$ for $m = 5$ , reaching an F-score of 0.96–0.98 at $n _ { e } = 1 0 ^ { 3 }$ and precision 1. The variants instead plateau or decline, more so with more environments: 1PC/3PC reach 0.87/0.93 at $m = 5$ but only 0.62/0.78 at $m = 2 0$ , and avg reaches 0.59. The decline is in precision (0.46–0.66 at $m = 2 0 )$ , the discarded within-block directions carry part of the dependence between a block and its parents, so the reduced model is misspecified and, given enough data, admits spurious parents. Reduction thus trades exactness at large $n _ { e }$ for a score that exists when $n _ { e }$ is below the family size.

Per block (Fig. 11), coarse never recovers a parent set exactly while $n _ { e } \leq F _ { j }$ , because grow-shrink only admits parents whose family still fits in $n _ { e }$ samples (the MLE does not exist). Just below $n _ { e } = F _ { j }$ , its recall is 0.31– 0.45, against 0.60–0.72 for 3PC (exact recovery 0.09–0.44). Above $n _ { e } = F _ { j }$ , coarse becomes exact $( \ge 0 . 8 6$ from $n _ { e } / F _ { j } \approx 2 0$ , ≈ 1 from ≈ 45), while 3PC keeps recall at 0.96–0.98 but is exact in only 0.60–0.89 of blocks (0.31–0.94 for 1PC and avg). The gap comes from spurious parents, matching the precision loss in Fig. 10.

## C.9 Noise distributions

The score of Def. 17 is a Gaussian likelihood. Fig. 13 runs the edge stage under the oracle partition on ER graphs with $\rho = 0 . 2$ and $m = 5$ , for $d \in \{ 5 0 , 1 0 0 , 2 0 0 , 5 0 0 \}$ and five noise families with equal mean and variance: Gaussian, uniform, Laplace and Student-t with 3 and 4 degrees of freedom. The curves coincide: at every $( d , n _ { e } )$ the mean F-scores of the five families difer by at most 0.01, and all reach 1 at $n _ { e } = 5 \times 1 0 ^ { 4 }$ . In a linear model, d-separation implies a vanishing partial covariance whatever the noise, and that is what the increments of Prop. 18 detect allowing BIC to correctly rank the models even under misspecification. The partition is fixed to the oracle here, so this does not test the Welch test under heavy tails, where a rank-based or distribution-free two-sample test could replace it.

## C.10 Causal chamber: full results

Tab. 4 reports precision and recall, oracle selection and the grouped mode. In the ungrouped mode the hyperparameters selected without ground truth coincide with the oracle ones for coarse $( \alpha = 1 0 ^ { - 4 } )$ and RePaRe $( \alpha = 1 0 ^ { - 4 } , \beta = 1 0 ^ { - 2 } )$ ; UT-IGSP gains 0.11 in F-score under oracle selection. In the grouped mode $( m = 2 )$ both methods select a looser partition level than the oracle $( \alpha = 1 0 ^ { - 2 }$ for coarse, $1 0 ^ { - 3 }$ for $\mathrm { R e P a R e } )$ and miss the oracle partition by one variable (ARI 0.93); the edges are the same under both selections, 2 of the 4 true coarse edges with no false edge. Tab. 5 complements Tab. 2 with adjacency (undirected) metrics at the variable level and the exact hypergeometric null. Every method recovers adjacencies above chance, including GnIES, whose directed F-score in Tab. 2 is at chance: its errors are in the orientations. coarse and RePaRe have the highest precision, 0.86, and tie with UT-IGSP for the highest recall.

![](images/a33aa8c9ffcbb6762d5ffba3131e19c89280e1268ee5138f4fb30ef4eeeeb9e0.jpg)  
Figure 5: Partition ARI against the per-test cutof, ER graphs, $d = 2 5 0 , m = 5 .$ Grey: fixed α (median and interquartile range); markers: cutof and ARI reached by each rule (medians with interquartile ranges). Every data-driven rule but Bonferroni lands in the flat region of the curve; CV reaches the oracle without ground truth.

![](images/d9e7c1ceffc0477bddda37edb887dd78f1d2647758fb222b2cab95813e211e23.jpg)  
Figure 6: ARI of each rule minus the oracle ARI on the same dataset, over 20 seeds. d stands for density.

![](images/432e0663ae9b41d4a9caca6bcca5fd59b5f0cd2b3907bbb4924c472ada4c1393.jpg)  
Figure 7: $\Pi ^ { \varepsilon }$ against $\Pi ^ { \mathcal { I } }$ at the population level, native design, ER graphs, $d = 5 0 ;$ k targets per environment, one curve per number of environments m (mean and 95% CI over 200 seeds). Resolution is lost as soon as $k > 1$ more environments refine $\Pi ^ { \mathcal { I } }$ faster than $\dot { \Pi } ^ { \varepsilon }$ but hide fewer edges.

![](images/b1148a89a9da2ee791255f198e5155aa6d021afd637f0710cba3848c56e4ad12.jpg)

Figure 8: As Fig. 7, overlap design: all targets come from a pool of 20 nodes and m grows; one curve per $k ,$ dashed line at the single-target value. Overlapping environments recover the resolution lost to multi-target interventions.  
![](images/9ab8f304407eeef607b1f69c3512248ea9ab2a8bdc342d4358d12bea9d710daa.jpg)  
Figure 9: Partition estimated by coarse with cross-validated α against $\Pi ^ { \varepsilon } \left( \mathrm { t o p } \right)$ and $\Pi ^ { \mathcal { I } }$ (bottom), ER graphs, $d = 5 0 , m = 1 0$ ; one curve per $n _ { e }$ , dashed: population $\mathrm { A R I } ( \Pi ^ { \varepsilon } , \breve { \Pi } ^ { \tau } )$ . The estimate tracks $\Pi ^ { \varepsilon }$ at every $k ;$ its distance to $\Pi ^ { \mathcal { I } }$ is the population gap.

![](images/53e126357cd467c691c93c9826e1402f433f50a393a3871cbeac58b3f32482eb.jpg)

Figure 10: Edge recovery under the oracle partition for coarse and its reduced-rank variants, ER graphs, $d = 1 0 0 , \rho = 0 . 1 ;$ dashed line at $n _ { e } = d .$ The variants score where coarse cannot $( n _ { e }$ below the family size) and lose precision where it can.  
![](images/3f2e26805be2d234165cd7554e8552bea88b06bd7e2b4d7a52c3120950c869bc.jpg)  
Figure 11: Per-block parent recovery under the oracle partition for coarse and its reduced-rank variants, ER graphs, $d = 1 0 0 , \rho = 0 . 1 \colon$ exact parent set (top) and parent recall (bottom) against $n _ { e } / F _ { j }$ , with $F _ { j }$ the number of variables in the true family of $\pi _ { j }$ for every method; blocks binned on a $\log _ { 2 }$ scale, dashed line at $n _ { e } = F _ { j }$ coarse near 0 recovery below the line stems from the non-existence of the $\operatorname { M L E } ;$ the variants recover part of the parents below it but add spurious ones above it.

sample size (n)  
![](images/1ace47a63e6d76aae37590cb4224674ab6abdd0c73c4f25abbf1606f734d9ebf.jpg)  
sample size per env (n)

![](images/6e99d572618361dcb273fb71ec55913b12317cde8d2fd4065d032ac67a517aa2.jpg)  
sample size per env (n)

![](images/493c9beef0f78d46ea7c6a7c29c19466d57b69a3f00499ab71dfc1f25a840249.jpg)  
sample size per env (n)  
Figure 12: Share of blocks whose family size, in each method’s own units, is below $n _ { e } ,$ same setting as Fig. 10. The crossover of ${ \mathrm { F i g . } }$ 10 coincides with the point where every family of coarse becomes feasible.

![](images/01caf6b97bb318d79d358a2a071f6be018addb2b92eec9735af7026dd5094e10.jpg)  
sample size (n)  
(a) d = 50

![](images/cf49189592ca158231048da4d284a568b5c80c124e584e197ce947a090df4a18.jpg)  
(b) d = 100

![](images/2d13422deb382686d18a229a47c070b33043a0a45f95a649635a4c645bead598.jpg)  
(c) d = 200

![](images/11a4738af0dd6bb81b09d3623e33e7f72963271ff916007e5cca0580d32a66cb.jpg)  
(d) d = 500  
Figure 13: Edge F-score under the oracle partition for five noise families, ER graphs, $\rho = 0 . 2 , m = 5 .$ . The Gaussian score is insensitive to the noise family in the linear model.

Table 4: Light tunnel, all methods and selections. Score rows select hyperparameters without ground truth (coarse: cross-validation; RePaRe: GnIES score; UT-IGSP: BIC), oracle rows by maximum ARI, then F-score; GIES and GnIES have no hyperparameters. Metrics refer to the coarse DAG for coarse and RePaRe and to the variable-level graph for the others; GnIES is scored on the skeleton. Fit and Search are wall-clock seconds of the final fit and of the hyperparameter search.
<table><tr><td>Method</td><td>Sel.</td><td>ARI↑</td><td>Prec. ↑</td><td>Rec. ↑</td><td>F1 ↑</td><td>Fit(s)</td><td>Search(s)</td></tr><tr><td colspan="8">Ungrouped (m = 5)</td></tr><tr><td>COARSE</td><td>score</td><td>1.00</td><td>1.00</td><td>0.80</td><td>0.89</td><td>0.02</td><td>0.40</td></tr><tr><td rowspan="3">RePaRe</td><td>oracle</td><td>1.00</td><td>1.00</td><td>0.80</td><td>0.89</td><td>0.01</td><td></td></tr><tr><td>score</td><td>1.00</td><td>1.00</td><td>0.80</td><td>0.89</td><td>0.10</td><td>1.54</td></tr><tr><td>oracle</td><td>1.00</td><td>1.00</td><td>0.80</td><td>0.89</td><td>0.10</td><td></td></tr><tr><td>GIES GnIES</td><td></td><td></td><td>0.63 0.41</td><td>0.62 0.56</td><td>0.62 0.47</td><td>1.83</td><td></td></tr><tr><td rowspan="3">UT-IGSP</td><td>score</td><td></td><td>0.38</td><td></td><td></td><td>376</td><td></td></tr><tr><td>oracle</td><td></td><td>0.53</td><td>0.54</td><td>0.45</td><td>0.19</td><td>2.69</td></tr><tr><td></td><td></td><td></td><td>0.59</td><td>0.56</td><td>0.13</td><td></td></tr><tr><td></td><td>Grouped (m = 2)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>COARSE</td><td>score</td><td>0.93</td><td>1.00</td><td>0.50</td><td>0.67</td><td>&lt;0.01</td><td>0.12</td></tr><tr><td rowspan="3">RePaRe</td><td>oracle</td><td>1.00</td><td>1.00</td><td>0.50</td><td>0.67</td><td>&lt;0.01</td><td></td></tr><tr><td>score</td><td>0.93</td><td>1.00</td><td>0.50</td><td>0.67</td><td>0.02</td><td>0.27</td></tr><tr><td>oracle</td><td>1.00</td><td>1.00</td><td>0.50</td><td>0.67</td><td>0.02</td><td></td></tr></table>

Table 5: Light tunnel, ungrouped mode: adjacency (undirected) metrics against the exact negative control. Under the null hypothesis, the number of true adjacencies among the ˆm estimated ones is hypergeometric. Nul F1 is its mean and 95% interval.
<table><tr><td>Method</td><td>Prec. ↑</td><td>Rec. ↑</td><td>F1 ↑</td><td></td><td>Null F1</td><td> $\hat { m }$ </td></tr><tr><td colspan="7">Variable level</td></tr><tr><td>COARSE</td><td>0.86</td><td>0.64</td><td>0.74</td><td></td><td>0.18 [0.06, 0.29]</td><td>29</td></tr><tr><td>RePaRe</td><td>0.86</td><td>0.64</td><td>0.74</td><td>0.18</td><td>[0.06, 0.29]</td><td>29</td></tr><tr><td>GIES</td><td>0.63</td><td>0.62</td><td>0.62</td><td>0.20</td><td>[0.10, 0.31]</td><td>38</td></tr><tr><td>GnIES</td><td>0.41</td><td>0.56</td><td>0.47</td><td>0.24</td><td>[0.13, 0.34]</td><td>54</td></tr><tr><td>UT-IGSP</td><td>0.45</td><td>0.64</td><td>0.53</td><td>0.24</td><td>[0.13, 0.34]</td><td>55</td></tr><tr><td colspan="7">Block level</td></tr><tr><td>COARSE</td><td>1.00</td><td>0.80</td><td>0.89</td><td>0.32</td><td>[0.11, 0.56]</td><td>8</td></tr><tr><td>RePaRe</td><td>1.00</td><td>0.80</td><td>0.89</td><td>0.32</td><td>[0.11, 0.56]</td><td>8</td></tr><tr><td>GIES</td><td>0.80</td><td>0.80</td><td>0.80</td><td>0.36</td><td>[0.10, 0.60]</td><td>10</td></tr><tr><td>GnIES</td><td>0.50</td><td>0.80</td><td>0.62</td><td>0.44</td><td>[0.23, 0.62]</td><td>16</td></tr><tr><td>UT-IGSP</td><td>0.50</td><td>0.90</td><td>0.64</td><td>0.46</td><td>[0.29, 0.64]</td><td>18</td></tr></table>