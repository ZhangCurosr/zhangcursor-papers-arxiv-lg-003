# The Effect of Layer Ordering in Hybrid Architectures

John Cooper<sup>\*1</sup>, Mingchen Ma<sup>\*1</sup>, Jiaqi Mo<sup>\*1</sup>, Shengqi Qiu<sup>\*1</sup>, Nathaniel Bastian<sup>2</sup>, Ilias Diakonikolas<sup>1</sup>, and Frederic Sala<sup>1</sup>

<sup>1</sup>Department of Computer Sciences, University of Wisconsin-Madison <sup>2</sup>Department of Computer Sciences, John Hopkins University {jfcooper2, mingchen}@cs.wisc.edu

## Abstract

Hybrid attention is rapidly becoming the dominant architectural choice for frontier models. Despite their successes, such architectures complicate model design by introducing additional choices, including the types and locations of layers. This combinatorially large design space has been studied empirically, but given the vast expense of training and evaluating models with different configurations, a principled understanding of the effects of layer selection and ordering is needed. We provide the first such analysis, showing that the dominant hybrid architecture is highly task-specific. Concretely, for any layer configuration, we prove that there exists a task where we can construct a model with this configuration that provably outperforms any other model of any different configuration under fixed resource constraints. Empirically, we observe that for standard training, learned hybrid architectures exhibit separations between layer orderings that match our theoretical findings. For instance, under the same memory budget, the optimal architecture predicted by our theory for a segmented retrieval task consistently achieves perfect accuracy, whereas the next strongest plateaus below 0.5. These results extend to deeper models, where “containing” the dominant architecture for a task indicates whether the model attains high performance. In addition, we investigate the mechanisms learned by hybrids, observing that Transformer layers contribute the largest share of performance on associative recall across architectures. This work establishes insights into taskdependent hybrid architecture design relevant to the training and scaling of frontier-scale hybrids.

## 1 Introduction

Hybrid architectures have become the design of choice for frontier models [20, 25, 12]. This is due to their superior scalability: the quadratic complexity of full self-attention limits the vanilla Transformer architecture on long-context tasks [11, 24]. Hybrid models replace many self-attention layers with efficient subquadratic alternatives that require smaller memory and offer faster inference. Popular choices include state-space layers like linear attention, Mamba, Gated DeltaNet, etc. [8, 32]. The use of these layers produces scalable and highly-performant models.

However, as each layer can be selected from among many variations, layer selection and ordering is a critical design challenge [32, 8, 10]. Indeed, each layer type brings different constraints when processing sequences: full-attention is memory-bound by retrieval distance while recurrent alternatives are bound by their state size. The most popular layer ordering among frontier models uses a block design that involves one softmax-attention layer followed by many subquadratic layers. This is supported empirically [29]. While some work has investigated the selection ratio between softmax-attention and linear attention [30, 7] and others have shown a theoretical separation with hybrids utilizing less memory [6], the benefits and drawbacks of various layer orderings are still unclear. Given the combinatorially large space of different layer orderings, a complete empirical sweep across architectures is intractable.

![](images/7eacf39391ebc4653944fe6c71d90340f012dae9ef56e23f06f4076f0e54c4c7.jpg)  
Figure 1: Structure of tasks that exhibit performance differences among different hybrid architectures. Large gaps need SSM layers to communicate each segment’s query, while the dense computation per segment is done by the Transformers.

A theoretical characterization of hybrid architectures would cut down on this combinatorial space. However, identifying differences between hybrid layer orderings is highly challenging. Tasks can be solved by a variety of algorithms, each best suited for a different hybrid architecture. For instance, the greatest difficulty stems from processing order: some architectures focus on reordering the sequence in early layers to allow for a few later layers to solve the task. These shortcuts can reduce model size and depth compared to more direct algorithms with sequential processing.

In this work, we obtain a new theory-driven approach to hybrid architecture design. Inspired by [6], we study the expressivity of different hybrid architectures under computational resource constraints. We identify tasks and associated architecture designs where no other design (i.e., layer ordering) is able to solve the task under identical constraints. We begin by abstracting full-attention (TF) and sub-quadratic attention variants as generalized windowed functions and generalized state-space models (SSMs), respectively. This allows us to focus on the critical elements constraining layer types: the window width, state size, and embedding precision. For a given layer ordering of generalized SSMs and generalized causal windowed models, we show there exists a task and hybrid architecture such that any architecture deviating from the provided one must use more resources at some layer. This indicates that large-scale hybrid architecture design is situation-specific; in other words, structural properties of a desired task determines the optimal hybrid architecture.

In addition, we zoom in on 2- and 3-layer hybrid architectures to characterize and expand the set of such separating tasks. These tasks are natural, mirroring behaviors found in real-world natural language scenarios. Specifically, we establish tasks where the TF → SSM (TS) and TF → SSM → TF (TST) architectures are resource optimal. The task for TST supplies a natural task that requires layer alternation in a hybrid design. For these tasks, we show a separation in learnability between different layer orderings. By modifying the context window for the Transformers and varying the state in Mamba, we constrain the memory of these hybrid under some fixed budget. We empirically show that the learned hybrids exhibit identical separations as predicted theoretically. For segmented retrieval (designed for the TS architectures) and the TST task, we observe a phenomenon crucial for large-scale design: on hybrids of depth 4 and depth 5, containing TS or TST as a sub-architecture determines performance. This suggests that such “containment” is important in deep hybrids.

We empirically study such deeper hybrids as well. We find that a $\boldsymbol { \mathrm { T } } ^ { * } \boldsymbol { \mathrm { S } } ^ { * }$ architecture achieves higher performance than all ${ \mathbf { S } } ^ { \ast } { \mathbf { T } } ^ { \ast }$ architectures for this task. Mechanistically, through linear probing, we observe that the top-performing model, TTSSSSSS, shows a slow increase in decodability in its SSM layers which are not seen in TTTSSSSS. This supports that hybrid design is nuanced, as slightly different architectures fail to use their SSM layers to improve performance.

The paper is structured as follows. In Section 2, we begin with proving that there are tasks and resource constraints for any hybrid architecture that any other architecture must violate. For shallow hybrids, we establish a natural task family capturing important operations in natural language. In Section 3, we experiment on these natural tasks, identifying that these separations appear for learned hybrids. Lastly, in Section 4, we experiment on 8-layer hybrids on associative recall, observing that mildly different architectures have drastically different performances.

## 2 Theoretical Foundations for Hybrid Layer Ordering

In this section, we introduce our main theorem on model superiority, where we show that for each architecture, there is a task on which it is dominant. We begin with a summary of notation in Appendix A before discussing the main theorem separating architectures of differing layer orderings.

Preliminaries and Notation. We consider sequence-to-sequence token prediction problems. Let V be some vocabulary of tokens, $V = | \nu |$ , and $\vec { \bf x } = ( { \bf x } _ { i } ) _ { i = 1 } ^ { L }$ be an input sequence. A language model M a is sequenceto-sequence map $M : \mathcal { V } ^ { L } \to \mathcal { V } ^ { m }$ of the form $M ( \vec { \bf x } ) = F _ { N } \circ F _ { N - 1 } \circ \cdot \cdot \cdot \circ F _ { 1 } ( \vec { \bf x } )$ , where ◦ is function composition and each $F _ { i }$ is a sequence-to-sequence map called a layer. Through the paper, we will consider the input and output of $F _ { i }$ at each position to be in $\{ 0 , 1 \bar  \} ^ { P }$ , for some $P \in \mathbf { N } _ { + }$ , where we call $P$ the precision of the model. We will consider two types of layers as abstractions of state space models (SSM) and causal windowed attention (TF). The embedded input sequence $\vec { \bf x } = ( { \bf x } _ { 1 } , \ldots , { \bf x } _ { L } )$ uses $P$ precision, i.e ⃗x is such that $\mathbf { x } _ { i } \in \{ 0 , 1 \} ^ { P }$

Transformer Layer. We say a layer $F : ( \{ 0 , 1 \} ^ { P } ) ^ { L } \to ( \{ 0 , 1 \} ^ { P } ) ^ { L }$ is a T-layer (TF) with window size W if there exists a map $f$ such that for $i \in [ L ] , F ( \vec { \bf x } ) _ { i } = f ( { \bf x } _ { i - W + 1 } , \ldots , { \bf x } _ { i } ) \in \{ 0 , 1 \} ^ { P }$ . In other words, a T-layer can be an arbitrary $P .$ -precision causal nonlinear function, whose output depends on the past W positions of the input. Standard transformer implementations with the windowed attention mechanism falls into this category (See Appendix A for formal definitions).

State-Space Model Layer. We say a layer $F : ( \{ 0 , 1 \} ^ { P } ) ^ { L } \to ( \{ 0 , 1 \} ^ { P } ) ^ { L }$ is an S-layer (SSM) with bit complexity B if there exists a finite set $s$ of size at most $2 ^ { B }$ , an update rule $u : \mathcal { S } \times \{ 0 , 1 \} ^ { P } \to \mathcal { S }$ and an output function $r : \mathcal { S }  \{ 0 , 1 \} ^ { P }$ such that $S _ { t } = u ( S _ { t - 1 } , \vec { \bf x } _ { t } )$ and $F ( \vec { \bf x } ) _ { t } = r ( S _ { t } )$ hold.

These definitions capture the core difference between these layers: Transformers perfectly recall their recent context, while SSMs store a compressed version of the context [18].

We define $\Sigma : = \{ \mathrm { S } , \mathrm { T } \}$ . An architecture profile A is a finite word $A = A _ { 1 } A _ { 2 } \cdot \cdot \cdot A _ { D } \in \Sigma ^ { * }$ , whose depth is $| A | : = D$ . For two architecture profiles $A , B$ , we write $A \preceq B$ if A is a subsequence of $B { : }$ : there exist indices $1 \leq i _ { 1 } < \cdots < i _ { | A | } \leq | B |$ such that $\forall j \in [ | A | ] , A _ { j } = B _ { i _ { j } }$ . Notice that every nonempty architecture profile A can be written as $A = X _ { 1 } ^ { r _ { 1 } } X _ { 2 } ^ { r _ { 2 } } \cdot . . . X _ { k } ^ { r _ { k } }$ , with $\textstyle \sum _ { i = 1 } ^ { k } r _ { i } = | A |$ and $r _ { i } \in \mathbf { N } _ { + }$ . Here, $X _ { i } \in \Sigma$ and $X _ { i } \neq X _ { i + 1 }$ for $i \in [ k - 1 ]$ . The architecture $X _ { i } ^ { r _ { i } }$ <sup>i</sup> is comprised of r layers of type $X _ { i }$ without a gap. We define comp $( A ) : = X _ { 1 } X _ { 2 } \cdot \cdot \cdot X _ { k }$ as the compression of A. Let $B , P , W \in \mathbf { N } _ { + }$ , and denote $\mathcal { M } _ { A } ( B , P , W )$ for the class of hybrid models with profile A, such that every S-layer has a state space with size at most $2 ^ { B }$ every T-layer has a window size at most $W$ , and each intermediate tokens between layers belong to a token space with size at most $2 ^ { P }$

The empty architecture represents the identity sequence map. Note that since both an S-layer and a

T-layer contain the identity map, when $A \preceq B$ , a model $M \in \mathcal { M } _ { A } ( B , P , W )$ can be simulated by a model $M ^ { \prime } \in \mathcal { M } _ { B } ( B , P , W )$ . In particular, if $| A | = | B |$ , then $A \preceq B$ forces $A = B$

## 2.1 Expressivity and Efficiency Tradeoff under Budget Constraints

Our method to evaluate the choice of different layer orderings is to understand, under a given budget $B , P ,$ W on the state size, precision, and the window length, whether for a given task there exists a hybrid model $M \in \mathcal { M } _ { A } ( B , P , W )$ that can solve the task with good accuracy. If there exists a model $M \in { \mathcal { M } } _ { \tau } ( B , P , W )$ that solves some given task perfectly, while a model with the same depth but with a different profile must require some computational budgets far beyond $B , P , W$ , then τ is called the dominant choice of ordering for this task. Our main theoretical result reveals task-dependence in the choice of hybrid layer ordering, formalized in Theorem 1.

Theorem 1. Let $D \in \mathbf { N } _ { + }$ and $\tau = X _ { 1 } ^ { r _ { 1 } } X _ { 2 } ^ { r _ { 2 } } \cdot \cdot \cdot X _ { h } ^ { r _ { h } } \in \Sigma ^ { * }$ be an arbitrary architecture profile with depth D. Let $\alpha : = \mathrm { c o m p } ( \tau )$ and $\epsilon \in ( 0 , 1 )$ be an arbitrary fixed constant. There exist constants $c _ { \tau } > 0 , C _ { \tau , \epsilon } >$ 0, a family of task $\{ F _ { \tau , \epsilon , m } \} _ { m \ge 1 }$ , where $F _ { \tau , \epsilon , m } : ( \mathcal { V } _ { \epsilon , m } ) ^ { L _ { \epsilon , m } } \to \mathcal { V } _ { \epsilon , m }$ and $| \mathcal { V } _ { \epsilon , m } | \le \mathrm { p o l y } _ { \epsilon } ( m ) , L _ { \epsilon , m } \le$ poly (m), and a family of distributions $\mu _ { m }$ over a sequence of tokens with length $L _ { m }$ parameterized by $m \in \mathbf { N } _ { + }$ , such thatfor all m sufficiently large, thefollows hold.

1. (Completeness) There exists a hybrid model $M _ { \tau , m } \in \mathcal { M } _ { \tau } ( B _ { \tau , m } , P _ { \tau , m } , W _ { \tau , m } )$ such that er $\mathfrak { c } _ { \mu _ { m } } ( M _ { \tau , m } , F _ { \tau , \varepsilon , m } ) =$ 0 and $B _ { \tau , m } \leq C _ { \tau , \epsilon } \log m , P _ { \tau , m } \leq C _ { \tau , \epsilon } \log m , W _ { \tau , m } \leq m + 1 .$

2. (Soundness for wrong ordering) Let $A \in \Sigma ^ { \leq D }$ be an architecture profile with $\alpha \not \simeq A .$ . Then for every hybrid model $M \in \mathcal { M } _ { A } ( B , P , W )$ that satisfies err $( M , F _ { \tau , \epsilon , m } ) \leq O ( 1 / D )$ , must satisfy $W \geq$ $c _ { \tau } m ^ { 1 / \epsilon } o r \operatorname* { m a x } \{ B , P \} \geq c _ { \tau } m .$

3. (Soundness for insufficient profile) Let $A \in \Sigma ^ { \leq D }$ be an architecture profile with $\alpha \preceq A$ but $\tau \not \simeq A .$ Then, for every hybrid model $M \in \mathcal { M } _ { A } ( B , P , W )$ that satisfies err $( M , F _ { \tau , \epsilon , m } ) \leq O ( 1 / D )$ , must satisfy $W > \lambda _ { \tau } ( m + 1 )$ or $B > \lambda _ { \tau } B _ { \tau , m } o r { \cal P } > \lambda _ { \tau } { \cal P } _ { \tau , m } ,$ , where λ<sub>τ</sub> is a constant that is strictly larger than 1.

We give an interpretation of Theorem 1. Given any target profile $\tau \in \Sigma ^ { D }$ , we would like to evaluate models over a family of tasks $\{ F _ { \tau , \epsilon , m } \}$ , where the size of the input of $F _ { \tau , \epsilon , m }$ grows at an order of poly (m), for every fixed $\epsilon \in ( 0 , 1 )$ . In particular, with a resource budget $B _ { \tau , m } , P _ { \tau , m } = O$ <sub>ϵ</sub>(log m) and $W _ { \tau , m } = m { + } 1$ a hybrid model in $\mathcal { M } _ { \tau } ( B _ { \tau , m } , P _ { \tau , m } , W _ { \tau , m } )$ can perfectly solve $F _ { \tau , \epsilon , m }$ On the other hand, we design a distribution $\mu _ { \epsilon , m }$ of inputs such that every model $M \in \mathcal { M } _ { A } ( B , P , W )$ with the same depth and $\tau \not \simeq A$ must require more resources to achieve a reasonable accuracy.

To quantify the resource separation, we partition the $A \in \Sigma ^ { \leq D }$ into 2 categories: (i) architectures satisfying $\mathrm { c o m p } ( \tau ) \not \simeq A$ , where A is still not more expressive than τ with all neighboring layers merged, and (ii) architectures satisfying $\mathrm { c o m p } ( \tau ) \preceq A$ . We say that A has the wrong order when $\mathrm { c o m p } ( \tau ) \not \simeq A .$ which, by Theorem 1, requires qualitatively more resources to solve the task. In particular, a model A that can solve the task either requires exponentially larger state spaces/precisions or needs a polynomially larger window size, but with an arbitrarily large rate of growth. When $\mathrm { c o m p } ( \tau ) \preceq A$ but $\tau \not \simeq A$ , a qualitative separation is no longer possible. This is due to our result in Appendix E showing that consecutive layers of the same type can be merged. However, we show that the resources required by such a profile are still a constant factor larger than that of τ.

We next provide an overview of why Theorem 1 is expected, and the technique we use to prove it. We defer the full technical details of Theorem 1 to Appendix E.

Warm Up: Examples for Shallow Hybrid Models. To start with, we consider a simple associative recall [1] variant, where we show $\mathrm { ~ a ~ T ~ S ~ }$ profile is the dominant architecture. Consider an input sequence I of the form $( \mathbf { x } _ { 1 } , \hdots , \mathbf { x } _ { m - 1 } , q , \vec { 0 } )$ with length $L ,$ where $q \in [ m - 1 ]$ and $\mathbf { x } _ { i }$ is drawn from some finite vocabulary of size $\mathrm { p o l y } ( m )$ . The task input is $I ,$ and the output has $\mathbf { x } _ { q }$ at the last position. The T-layer can perform local computations within a window of $m ,$ , so it can output $\mathbf { x } _ { q }$ at position m. Following, when the S-layer receives $\mathbf { x } _ { q }$ at position m, it only needs $O ( \log m )$ to memorize it until the end of the sequence. This immediately gives a model $M \in \mathcal { M } _ { \mathrm { T S } } ( \log m , \log m , m )$ that solves the task.

In contrast, the ST architecture requires at least one constraint to be significantly larger. Assuming $W \le m$ , standard communication complexity [27] implies that using a T-layer as the last layer requires $\Omega ( m )$ bits from early in the context. Thus, the S-layer must store $\Omega ( m ) \gg$ log m bits to compress sufficient information to perform the task. This establishes a simple example where TS is the dominant architecture. We provide an additional broad family of natural tasks in Section 2.2 where TS is the dominant architecture, solvable using standard implementations of attention with Mamba-formed SSMs.

The most natural, yet flawed, task for a deeper target architecture τ is a task that iteratively solves the recall problem, separated by long 0 vectors. For the $\tau \ = \ \mathrm { T } { \cal S } \mathrm { T }$ architecture, such a task does separate architectures: this simple tensor look-up task makes τ the dominant choice and is studied empirically in Section 3. Specifically, we are given input $I = ( \vec { x } , q _ { 1 } , \vec { 0 } , \vec { F } , q _ { 2 } )$ , where each $\vec { x } \in [ \sqrt { m } ] ^ { m - 1 } , q _ { 1 } \in [ m - 1 ]$ $\vec { F } \in \mathcal { V } ^ { m }$ is a ${ \sqrt { m } } \times { \sqrt { m } }$ matrix, $q _ { 2 } \in [ \sqrt { m } ]$ , we are asked to output $F _ { \mathbf { x } _ { q _ { 1 } } , q _ { 2 } }$ . The intuition is clean: a T-layer extracts $\mathbf { x } _ { q _ { 1 } }$ and sends it through the S-layer to the second T-layer to complete the lookup task.

However, for $\tau = { \mathrm { S T } } { \mathrm { S } } ,$ a similar idea no longer works. For intuition, we modify the tensor lookup task with the input $I = ( q _ { 1 } , \vec { 0 } , \vec { F } , q _ { 2 } , \vec { 0 } )$ and ask a model to output $F _ { q _ { 1 } , q _ { 2 } }$ at the end. Although a small model with an STS profile can solve the task, we find that a small model with a TS profile can also solve the task. The architecture uses a window of a T-layer that entirely contains $\vec { F }$ , identical to what is used for STS. This layer can copy $\vec { F }$ and place it after $q _ { 2 } \ [ 9 ]$ . This implies that the S-layer receives $q _ { 1 }  q _ { 2 }  \vec { F } ,$ , changing the required processing order. This requires an S-layer with $O ( \log m )$ bits to solve the full task, the same as for STS but with one fewer layer. A core takeaway is that each layer is not only solving some specific task but also compressing and rearranging information for future layers to function more efficiently. A simple task like tensor lookup is too simple to prevent this type of information aggregation from being useful, especially when comparing T-layers, which are any possible causal-window nonlinear function. Thus, to establish Theorem 1, more complicated tasks must be considered.

The task we construct is in Definition 1.

Definition 1 (Sequential Function Composition). Let $\mathcal { X } _ { k e y }$ be a finite space of keys and C be a finite space of vocabularies. Let $k \in \mathbf { N } _ { + }$ . A level-k recursive chain $F _ { \tau , \epsilon , m }$ is defined by a sequence of k functions $\{ H _ { i } \} _ { i = 1 } ^ { k }$ . Specifically, for $i \in [ k ] , H _ { i } ( \mathbf { z } _ { i - 1 } , \mathbf { x } _ { i } ) : \mathcal { X } _ { k e y } \times \mathcal { C } ^ { m }  \mathcal { X } _ { k e y }$ , where $\mathbf { z } _ { i } : = H _ { i } ( \mathbf { z } _ { i - 1 } , \mathbf { x } _ { i } )$ . Given a sequence of input $I : = ( \mathbf { z } _ { 0 } , \vec { 0 } _ { 0 } , \mathbf { x } _ { 1 } , \vec { 0 } _ { 1 } , \ldots , \mathbf { x } _ { k } , \vec { 0 } _ { k } )$ of length $L _ { m } , F _ { \tau , \epsilon , m } ( I ) : = \mathbf { z } _ { k }$ , where each $\vec { 0 } _ { i }$ is a sequence of 0 vectors with different lengths.

The task is divided into k stages. In each stage, we take a previous answer $\mathbf { z } _ { i - 1 }$ and fresh independent context $\mathbf { x } _ { i }$ to produce the next answer, $\mathbf { z } _ { i } = H _ { i } ( \mathbf { z } _ { i - 1 } , \mathbf { x } _ { i } )$ . If we use a layer ordering that subsumes comp(τ ), then the role of $H _ { i }$ is similar to the lookup table in the tensor lookup task, where a T-layer computes $\mathbf { z } _ { i }$ and the following S-layer sends it to the next stage after the long zero vectors. Thus, a model with profile τ can use a small resource budget to solve the task. However, when we analyze a profile A with $\alpha \not \simeq A .$ , motivated by the failure of the tensor lookup task discussed earlier, we construct a discrete function $H _ { i }$ that defeats any other information embedding.

For some intuition, consider α¯ starting with S as an example. We group α¯ into $( S _ { i } , \mathbb { T } _ { i } ) _ { i = 1 } ^ { k }$ , where ${ \mathrm { S } } _ { i } \left( \mathrm { r e s p . ~ T } _ { i } \right)$ is the i-th S(resp. T) layer. At stage $t + 1$ , with the new input block $\mathbf x _ { i + 1 }$ , the <sup>⃗</sup>0 prevents $T _ { i }$ from directly accessing information from the previous block. Let $C _ { i }$ denote the past states of the first i SSM layers before $\mathbf x _ { i + 1 }$ and let $\gamma$ denote the the state of $\mathrm { S } _ { i + 1 }$ right before reading $\mathbf { x } _ { i + 1 }$ . Fixing $C _ { i }$ to its current value value $c ,$ the future output of $\mathrm { S } _ { i + 1 }$ is a deterministic function completely determined by γ and $\mathbf x _ { i + 1 }$ and not previous blocks, i.e. $C _ { i + 1 } = G _ { c } ( \gamma , \mathbf { x } _ { i + 1 } )$ . We refer to Lemma 4 and Lemma 5 for the detailed analysis.

Importantly, since models need not compute $\mathbf { z } _ { i }$ directly to succeed at the task, given the current state record $C _ { i } .$ , we will recursively construct a list of possible $L _ { i } ( c ) \subseteq \mathcal { X } _ { k e y }$ such that only when $\mathbf { z } _ { i } \in L _ { i } ( c )$ can the model succeed with a good probability (Lemma 8). The list $L _ { i } ( c )$ contains all possible z such that for some state $\gamma , \mathbf { z } _ { i + 1 }$ has a large probability to fall into the next list $L _ { i + 1 } ( C _ { i + 1 } )$ . Lemma 6 and Lemma 7 use a probabilistic method to show that there exists an $H _ { i }$ such that each of the $L _ { i } ( c )$ cannot have a large size, which means $\mathbf { z } _ { 1 }$ has a small probability of falling into the list; thus, there exists an $H _ { i }$ used for the separating task. Using $H _ { i }$ , the competitive model with a wrong ordering will not succeed with a good probability unless it breaks some resource budget.

A Size-2 Cover Method for Deeper Hybrid Model with Insufficient Layers. We next give a brief overview of the case where $\mathrm { c o m p } ( \tau ) \preceq A$ . We defer to Appendix E.2 for full technical details. As discussed earlier, in this case, a qualitative separation is not possible. Instead, we construct tasks and carefully analyze the bit complexity of solving them. In our construction, we use variants of computing the inner product over a binary field. The main obstacle is that unlike the case with wrong orderings, there may be too many different competitive profiles to analyze. Our analysis uses a novel size-2 cover approach to narrow the structures needed to analyze.

Definition 2. Let $\tau \in \Sigma ^ { D }$ . Let $s ( \tau ) = | \{ i \in [ D ] \mid \tau _ { i } = \mathrm { S } \} |$ | and $t ( \tau ) = | \{ i \in [ D ] \ | \ \tau _ { i } = \mathbb { T } \} |$ . For $i \in [ s ( \tau ) ]$ ] we use $r _ { i } ( \tau )$ to denote the number of T-layers in τ that is strictly after the i-th S-layer and for $i \in [ t ( \tau ) ]$ ], we use $l _ { i } ( \tau )$ to denote the number of S-layers in τ that is strictly after the i-th T-layer. We call the following set of architecture profiles the size-2 cover of τ.

$$
\mathcal { C } _ { 2 } ( \tau ) = \{ \mathrm { S } ^ { s ( \tau ) } \} \cup \{ \mathrm { T } ^ { t ( \tau ) } \} \cup \{ \mathrm { S } ^ { i } \mathrm { T } ^ { r _ { i } ( \tau ) } ~ | ~ i \in [ s ( \tau ) ] \} \cup \{ \mathrm { T } ^ { i } \mathrm { S } ^ { l _ { i } ( \tau ) } ~ | ~ i \in [ t ( \tau ) ] \}
$$

Roughly speaking, the size-2 cover of τ contains maximal profiles, whose compressed version has length at most $2 ,$ subsumed by τ. As we show in Lemma 9 that if $\tau \not \simeq A$ and $A \in \Sigma ^ { \leq D }$ , then there must be some profile in $ { \mathcal { C } } _ { 2 } ( \tau )$ not subsumed by A. Thus, we only need to design hard tasks for profiles of the form in a size-2 cover; a mixture of these creates a single task that is hard for $\tau \not \simeq A .$ . However, although for each profile in $C _ { 2 } ( \tau )$ there exists a model with profile $\tau$ that can solve it exactly, it is not immediate whether there is a model with profile τ that can solve the mixture of these tasks exactly. Since S-layers and T-layers contain more general SSMs and causal windowed functions, we show in Lemma 10 that there exists a “meta” model with profile τ that can combine these submodels in a black-box way to solve the mixture of tasks.

## 2.2 Natural Tasks for 2-layer Hybrid Models

Although the tasks we constructed for Theorem 1 can be solved by a hybrid model with the target profile using small resource budgets, the tasks themselves are explicitly designed to broaden the family of models for the hardness result. These tasks may not be efficiently solvable using commonly used layer types such as Mamba, DeltaNet, or standard attention. Theorem 1 establishes resource separations across architecture profiles. We next illustrate the role of layer ordering through two more natural tasks that admit a separation favoring TS. We establish lower bounds for ST and give TS constructions in the generalized layer model in Appendix F.2, with full technical details in Appendix F. Appendix F.3 accounts for accumulated convolutional lookback in these recurrent-state lower bounds, extending their applicability to SSMs with convolutional front-ends, such as Mamba.

The core requirement for tasks with a dominant TS architecture is captured in Assumption 1. This assumption states that, for each task, there is some position τ in the context such that changes before some position τ require log |Q| bits of information from before $\tau ,$ similar to the TS task from Section 2. Under this assumption, we can show that any model with SSM layers first and TF layers second requires a large state or window to perform.

Theorem 2 (SSM → TF Cut). Let F be a length-L sequence task with output in Y. Suppose there exist a cutposition τ, a nonemptyfamily U oflength-τ prefixes, and a nonemptyfamily Q oflength- $\left( L - \tau \right)$ suffixes, such that every concatenation uq is a legal input and ∀u $\neq u ^ { \prime } \in \mathcal { U } , \exists q \in \mathcal { Q }$ such that $\begin{array} { r } { F ( u q ) \ne F ( u ^ { \prime } q ) } \end{array}$ . Let M be a deterministic model ofarchitecture ST with recurrent state space S and attention window W. IfM computes F correctly on all legal inputs and that $\tau \leq L - W$ , then $| S | \ge | U |$

This result stems from how information can be processed in a sequence. Without a large window, only the SSM in ST communicates with the early part of the sequence, forcing a large state. For a TS architecture, the Transformer performs computations that are then carried by the SSM, reducing state size and window length. We now define the two natural tasks for the TS architecture.

Exact Copy. Given a sequence x, the task of exact copy is to output 1 if there exists some repeated subsequence of length l, and output 0 otherwise.

Definition 3 (Exact Copy). Given a vocabulary V and an input sequence $\mathbf { x } \in \mathcal { V } ^ { L }$ , we say that x contains an exact copy of repetition c and string length l if there exists $i \in [ L ]$ s.t.

$$
\begin{array} { r } { \mathbf { x } _ { i + j + k l } = \mathbf { x } _ { i + j + ( k + 1 ) l } , \quad j \in \{ 0 , \cdots , l - 1 \} , k \in \{ 0 , \cdots , c - 2 \} . } \end{array}
$$

The task of Exact Copy is to output $y _ { L }$ , where $y _ { i } : = \mathbf { 1 } _  \{ \mathbf { x } _ { 1 : i }$ <sub>has</sub> <sub>an</sub> <sub>exact</sub> <sub>copy}</sub>.

This task satisfies Assumption 1 by setting τ to be the end of the exact copy. Thus, Theorem 2 applies, ensuring ${ \mathbf { S } } ^ { \ast } { \mathbf { T } } ^ { \ast }$ architectures require large memory. This task represents situations where a complex operation occurs early in a sequence, but the result needs to be carried much later in the context. For instance, reasoning-style tasks where a model identifies a useful lemma using TF layers early in the sequence can be efficiently propagated to the end of the sequence by SSM layers.

Segmented Retrieval. To define segmented retrieval, we place separator tokens in the sequence to act as a partition. The simplest per-segment operation is a fixed lookup in the past. Concretely, this means that for some fixed $\lambda \geq 1$ , the value at the separator token at position i is $x _ { i - \lambda }$ . The task of segmented retrieval is to output the last of these $x _ { i - \lambda }$ in some subset of the vocabulary.

Definition 4 (Segmented Retrieval). Given a vocabulary V, we specify a non-empty subset $\mathcal { N } \subseteq \mathcal { V }$ . Given an input sequence $( ( \mathbf { x } _ { 1 } , b _ { 1 } ) , \dotsc , ( \mathbf { x } _ { L } , b _ { L } ) ) \in ( { \mathcal { V } } \times \{ 0 , 1 \} ) ^ { L }$ , where $b _ { i }$ indicates whether position i is an internal segment boundary and is revealed only when position i is processed. We define $\{ t _ { 1 } , \cdots , t _ { k - 1 } \} : =$ $\{ i \in [ L - 1 ] : b _ { i } = 1 \} , t _ { 0 } : = 0 , t _ { k } : = L$ and refer an element from $\{ t _ { 1 } , \cdot \cdot \cdot , t _ { k - 1 } \}$ as a boundary and define l := min $j \in [ k ]  \left( t _ { j } - t _ { j - 1 } \right)$ and we set that $l \geq 2$ . Select $1 \leq \lambda < l$ , and define the largest index of the nonempty set $\{ \bar { j } \in [ k - 1 ] : \mathbf { x } _ { t _ { j } - \lambda } \in \mathcal { N } \}$ as λ<sup>⋆</sup>. The Segmented Retrieval task is to output the token $\mathbf { x } _ { t _ { \lambda ^ { \star } } - \lambda }$

Similar to exact copy, this task satisfies Assumption 1. Again, Theorem 2 applies, ensuring ${ \mathbf { S } } ^ { \ast } { \mathbf { T } } ^ { \ast }$ architectures require large memory. However, the source of the cut τ is different. Here, the cut occurs at the partitions, where the model must recall the past λ tokens. The TS architecture uses its TF layer to perform the λ-back recall, and the SSM carries the relevant tokens through the sequence. This is a simplified multidocument task, where information from each document (segment) may need to be carried to the end of the context; this latter component is best suited for SSM layers.

![](images/756e3b5aa08aa87b71fcc09fcff104387817615a6a832c56eeb5ac136813aa5b.jpg)  
(a) The Segmented Retrieval task.

![](images/09a53ec461c61ab0c813b1188d47c586bf37704e9a7fe5a2e97a2f13758e5c82.jpg)  
(b) The Exact Copy task.  
Figure 2: Results for the TS tasks where an inference memory proxy is held constant. The annotations represent the relationship between window storage and state size for Transformer-Mamba hybrids. The only architecture with high performance on the task is the TS architecture. Due to fixed memory, models with larger token dimension have smaller window and state size, making the tasks difficult. Conversely, small token dimension limits expressivity.

## 3 Empirical Results of Natural Tasks

We have established that for certain tasks, there is a theoretical separation in expressivity. However, these theoretical results focus on layerwise resource constraints that may not tie to common constraints in practice. We empirically test small hybrids on these tasks with a total fixed inference memory, showing that hybrids of this specific architecture learn them with the best performance.

## 3.1 Two Layers - Segmented Retrieval and Exact Copy

We begin the experiments with a sweep over the possible 2-layer hybrids on tasks where we expect a TS to be dominant. For two tasks, segmented retrieval and exact copy, we observe that there is a similar separation between architectures as predicted theoretically. These results are conducted under a fixed proxy of inference memory (details in Appendix C.3).

Setup. On both tasks, we use a small vocabulary of around 20 tokens and sequence lengths of around 150. We select Transformers with RoPE embeddings for T-layers and Mamba for S layers [8]. These match the model types used in [9] to compare pure SSMs to pure Transformers. For the TT and SS architectures, a fixed memory constraint induces a trade-off between token embedding dimension and window length or state size. For the ST and TS, there are two degrees of freedom to satisfy the memory budget proxy. We select some window-to-state-size ratios; for each, we execute a sweep over token dimension to blanket the space of possible architectures. The memory calculation is described in Appendix C.3. All experiments are averaged across 5 seeds.

Results. Results for both tasks are found in Figure 2. These results show the performance of each hybrid, plotted against the token embedding dimension. For the TS and ST hybrids, the ratio of window to state dimension is annotated at the end of each associated curve. Each curve ends once the desired ratio is no longer attainable, i.e. the token dimension is so large that the sum of the window and state expansion factors is less than the sum of the ratio components. Moving to the right increases the token embedding size, which requires the window and state expansion factor to decrease. This allows the model to have richer token representations but reduces the available context. Moving to the left increases the window width and state expansion factor but requires a reduction in token precision. While much, if not all, of the context is available, the token embeddings are so small that the model cannot compute with each token sufficiently. The highest average performance of a model other than TS is shown by a dashed horizontal line.

<table><tr><td>Model</td><td>Accuracy</td><td>Model</td><td>Accuracy</td></tr><tr><td>TTTT</td><td>0.871</td><td>TSSS</td><td>1.000</td></tr><tr><td>STTT</td><td>0.706</td><td>STSS</td><td>1.000</td></tr><tr><td>SSTT</td><td>0.516</td><td>SSTS</td><td>1.000</td></tr><tr><td>SSST</td><td>0.345</td><td>TTSS</td><td>1.000</td></tr><tr><td>SSSS</td><td>0.125</td><td>TSTS</td><td>1.000</td></tr><tr><td rowspan="6"></td><td></td><td>TSST</td><td>1.000</td></tr><tr><td></td><td>STTS</td><td>1.000</td></tr><tr><td></td><td>STST</td><td>1.000</td></tr><tr><td></td><td>TSTT</td><td>1.000</td></tr><tr><td></td><td>TTST</td><td>1.000</td></tr><tr><td></td><td>TTTS</td><td>1.000</td></tr></table>

Table 1: Training 4-layer hybrids on segmented retrieval. Left columns are models without TS; the right columns are models with TS as a component. There is a clear dichotomy between hybrids with and without this motif.

![](images/b5a155d8294b8465d7ee1fa2d3ec936822d22c9df24097da02ca6c7e26f59b39.jpg)  
Figure 3: Performance of all possible 5-layer Transformer-Mamba hybrids on the task where TST is expected to dominate. Models with TST as a subarchitecture perform above those without.

Crucially, the only model that achieves high accuracy, and perfect accuracy in one setting, is the TS architecture. This result also confirms that the layer ordering is vital to performance. The ST architectures never achieve high performance, as their ability to capture complex operations early in the sequence is hindered by this layer ordering.

## 3.2 Containing Subarchitectures Predicts Performance

The two-layer experiments in Section 3.1 and the three-layer experiments in Appendix D.4 identify TS and TST, respectively, as the best-performing profiles under the tested settings. Motivated by our theoretical separations, we next examine whether containing these profiles as subsequences predicts performance at greater depth. We evaluate all four-layer profiles on segmented retrieval (Table 1) and all five-layer profiles on tensor lookup (Figure 3).

Setup. For each experiment, we follow the setup for the two- and three-layer architectures. We increase the total allotted memory to account for the models having more layers. For both settings, we test all layer orderings of Transformers and SSM layers. We repeat across 3 seeds for each.

Results. The results for training 4-layer hybrids on segmented retrieval are found in Table 1. This table separates architectures that contain TS as an architecture and those that do not. Accuracy is taken as the highest across seeds for models without TS and the lowest across seeds for models with TS. Even so, all architectures with TS as a sub-architecture achieve perfect performance, while all architectures without TS do not achieve perfect performance on any of the 3 seeds.

![](images/345f9a811266e94681b4121500b0b3cdbd161a60d943a6ff1c77cb18df3a4224.jpg)  
Figure 4: Token accuracy of $\boldsymbol { \mathrm { T } } ^ { * } \boldsymbol { \mathrm { S } } ^ { * }$ and ${ \mathbf S } ^ { \ast } { \mathbf T } ^ { \ast }$ architectures on (2, 1)-AR. Small differences in architectures affect performance dramatically.

![](images/541dbccdff8b8cb00fee0eed76332cd84065b490a8e638b5de80c54aec2ac76e.jpg)  
Figure 5: Probe accuracy from the output stream at each model layer. All models have a significant gain in probe accuracy at their first TF layer, even in the pure TF where all layers are identical. TTTSSSSS and STSTSTST both see little increase at S-layers, while the other models do.

The results for the TST task are shown in Figure 3. The models are positioned based on their number of Transformer layers and are colored based on their containment of TST as a sub-architecture. Aside from the STSTS architecture, models with TST are notably higher in performance than the other architectures.

## 4 Observations on Associative Recall

We have thus far focused on the accuracy of various hybrid architectures. These architectures were trained on tasks where we have clear theoretical results indicating which architecture is predicted to have the highest performance. Here, we probe other aspects of learned hybrids on a variant of associative recall [1], including a mechanistic study of the learned algorithms by different layer orderings.

Setup. We test (2, 1)-AR (two tokens per key, one token per value) on hybrid architectures with 8 layers. This task represents a very compressed version of a multidocument task, where every triplet of tokens is a document identifier and associated value. The chosen architecture families are size-2 cover architectures $\mathrm { T } ^ { * } { \cal S } ^ { * } \ \mathrm { a n d \ S ^ { * } \mathrm { T } ^ { * } }$ . We vary the T-layer count between 0 and 8, repeating each setting with 5 seeds. The median, minimum, and maximum performance are reported. Additionally, for each architecture, we take the last token representations at each layer of the model and linearly probe for the next token. This roughly measures how much of the next token is identified by each layer. Additional details can be found in Appendix C.3.

Results. The results on performance are shown in Figure 4. As expected, hybrid models outperform pure architectures. However, each architecture family behaves differently, including a different peak TF-to-SSM ratio. Models with Transformers in the first few layers benefit most from having a very small subset of layers as Transformer layers. We also observe that the $\boldsymbol { \mathrm { T } } ^ { * } \boldsymbol { \mathrm { S } } ^ { * }$ architecture outperforms the ${ \mathrm { S } } ^ { * } { \mathrm { T } } ^ { * }$ architecture. We predict this is due to the multi-document nature of this task.

The linear probe results in Figure 5 further support this. Consistently, the first TF layer improves decodability significantly. This TF layer retrieves the key-value pairs within its context window, but cannot capture the rest. More subtle is the behavior of models with many SSM layers stacked back-to-back. With these long strings of SSM layers, the model’s decodability slowly increases as depth increases; we hypothesize the SSM layers are bringing these remaining key-value pairs from outside the context window. We also note that models trained on natural language typically exhibit a similar behavior, with TF layers performing the majority of recall [17].

## 5 Conclusion

Hybrid architectures are critical to building frontier models. Their performance and flexibility come at the cost of an expanded design space that cannot be easily navigated without an understanding of the effect of layer order and design on task performance. We provided an analysis taking the first step towards such an understanding. We showed that for each design, there exist tasks that provably favor a model with this type of architecture over all other choices. We do so by explicitly constructing such models, additionally finding that trained hybrids exhibit matching behaviors. Our general theory relies on deliberately constructed tasks and expressive abstractions of SSM and attention layers, and the corresponding optimal architectures need not always be realizable by standard practical models. While our natural tasks and experiments begin to bridge this gap, an important direction is to extend the theory to practical layer classes, richer mixtures of tasks, and natural-language settings. Ultimately, we hope to identify task-level properties that predict effective layer orderings before expensive model training.

## 6 Acknowledgements

John Cooper and Frederic Sala were supported by the National Science Foundation (NSF) (CCF2106707), the Defense Advanced Research Projects Agency (DARPA Young Faculty Award) and the Wisconsin Alumni Research Foundation (WARF). Ilias Diakonikolas was supported by a Guggenheim Fellowship and the ONR (N00014- 25-1-2268). Mingchen Ma was supported by NSF Award CCF-2144298 (CAREER) and the ONR (N00014- 25-1-2268).

## AI use statement

In this work, generative AI models including GPT 5.6 sol and GPT 6 Astra are used interactively as a routine for discussion, helping with proof construction, writing code for some of the tasks as well as checking potential flaws during the preparation of the manuscript. The analysis of Case 4 of Theorem 4 was mainly established and proved by GPT 6 Astra based on a hardness construction provided by the authors that contained a non-trivial mathematical flaw. We include the generated proof at https://github.com/SprocketLab/layerimportance, as supplementary material to support the correctness of the theory. We additionally used generative AI tools to polish some grammar in the text. We did not use generative AI tools for any other required-disclosure task, including the core writing or literature review. We have reviewed all AI-assisted work: AI-generated proofs and code were read, executed, and verified for correctness by at least 2 authors, and all suggested grammatical edits were checked to ensure no claims or technical content were altered. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

[1] Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, and Christopher Re. Zoology: Measuring and improving recall in efficient language models. In The Twelfth International Conference on Learning Representations, 2024.

[2] Sangmin Bae, Bilge Acun, Chien-Yu Lin, Haroun Habeeb, Seungyeon Kim, Liang Luo, Junjie Wang, and Carole-Jean Wu. Hybrid architectures for language models: Systematic analysis and design insights, 2026.

[3] Satwik Bhattamishra, Michael Hahn, Phil Blunsom, and Varun Kanade. Separations in the representational capabilities of transformers and recurrent architectures. Advances in Neural Information Processing Systems, 37:36002–36045, 2024.

[4] Aviv Bick, Eric P. Xing, and Albert Gu. Retrieval-aware distillation for transformer-SSM hybrids. In Forty-third International Conference on Machine Learning, 2026.

[5] Lijie Chen, Binghui Peng, and Hongxun Wu. Theoretical limitations of multi-layer transformer. In 2025 IEEE 66th Annual Symposium on Foundations of Computer Science (FOCS), pages 2631–2653. IEEE, 2025.

[6] John Cooper, Ilias Diakonikolas, Mingchen Ma, and Frederic Sala. Expressivity-efficiency tradeoffs for hybrid sequence models. In Forty-third International Conference on Machine Learning, 2026.

[7] Yonggan Fu, Xin Dong, Shizhe Diao, Matthijs Van keirsbilck, Hanrong Ye, Wonmin Byeon, Yashaswi Karnati, Lucas Liebenwein, Maksim Khadkevich, Alexander Keller, Jan Kautz, Yingyan Celine Lin, and Pavlo Molchanov. Nemotron-flash: Towards latency-optimal hybrid small language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[8] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling, 2024.

[9] Samy Jelassi, David Brandfonbrener, Sham M. Kakade, and Eran Malach. Repeat after me: Transformers are better than state space models at copying. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 21502– 21521. PMLR, 21–27 Jul 2024.

[10] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are RNNs: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pages 5156–5165. PMLR, 13–18 Jul 2020.

[11] Feyza Duman Keles, Pruthuvi Mahesakya Wijewardena, and Chinmay Hegde. On the computational complexity of self-attention. In Proceedings of The 34th International Conference on Algorithmic Learning Theory, volume 201 of Proceedings ofMachine Learning Research, pages 597–619. PMLR, 20 Feb–23 Feb 2023.

[12] Kimi Team, Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, Wentao Li, Enzhe Lu, Weizhou Liu, Yanru Chen, Weixin Xu, Longhui Yu, Yejie Wang, Yu Fan, Longguang Zhong, Enming Yuan, Dehao Zhang, Yizhi Zhang, T. Y. Liu, Haiming Wang, Shengjun Fang, Weiran He, Shaowei Liu, Yiwei Li, Jianlin Su, Jiezhong Qiu, Bo Pang, Junjie Yan, Zhejun Jiang, Weixiao Huang, Bohong Yin, Jiacheng You, Chu Wei, Zhengtao Wang, Chao Hong, Yutian Chen, Guanduo Chen, Yucheng Wang, Huabin Zheng, Feng Wang, Yibo Liu, Mengnan Dong, Zheng Zhang, Siyuan Pan, Wenhao Wu, Yuhao Wu, Longyu Guan, Jiawen Tao, Guohong Fu, Xinran Xu, Yuzhi Wang, Guokun Lai, Yuxin Wu, Xinyu Zhou, Zhilin Yang, and Yulun Du. Kimi linear: An expressive, efficient attention architecture, 2025.

[13] Eyal Kushilevitz. Communication complexity. In Advances in Computers, volume 44, pages 331–360. Elsevier, 1997.

[14] Hyunji Lee, Wenhao Yu, Hongming Zhang, Kaixin Ma, Jiyeon Kim, Dong Yu, and Minjoon Seo. Understanding and enhancing mamba-transformer hybrids for memory recall and language modeling. In Lucas Charpentier, Leshem Choshen, Ryan Cotterell, Mustafa Omer Gul, Michael Y. Hu, Jing Liu, Jaap Jumelet, Tal Linzen, Aaron Mueller, Candace Ross, Raj Sanjay Shah, Alex Warstadt, Ethan Gotlieb Wilcox, and Adina Williams, editors, Proceedings of the First BabyLM Workshop, pages 380–398, Suzhou, China, November 2025. Association for Computational Linguistics.

[15] Barak Lenz, Opher Lieber, Alan Arazi, Amir Bergman, Avshalom Manevich, Barak Peleg, Ben Aviram, Chen Almagor, Clara Fridman, Dan Padnos, Daniel Gissin, Daniel Jannai, Dor Muhlgay, Dor Zimberg, Edden M. Gerber, Elad Dolev, Eran Krakovsky, Erez Safahi, Erez Schwartz, Gal Cohen, Gal Shachaf, Haim Rozenblum, Hofit Bata, Ido Blass, Inbal Magar, Itay Dalmedigos, Jhonathan Osin, Julie Fadlon, Maria Rozman, Matan Danos, Michael Gokhman, Mor Zusman, Naama Gidron, Nir Ratner, Noam Gat, Noam Rozen, Oded Fried, Ohad Leshno, Omer Antverg, Omri Abend, Or Dagan, Orit Cohavi, Raz Alon, Ro’i Belson, Roi Cohen, Rom Gilad, Roman Glozman, Shahar Lev, Shai Shalev-Shwartz, Shaked Haim Meirom, Tal Delbari, Tal Ness, Tomer Asida, Tom Ben Gal, Tom Braude, Uriya Pumerantz, Josh Cohen, Yonatan Belinkov, Yuval Globerson, Yuval Peleg Levy, and Yoav Shoham. Jamba: Hybrid transformer-mamba language models. In The Thirteenth International Conference on Learning Representations, 2025.

[16] William Merrill, Yanhong Li, Tyler Romero, Anej Svete, Caia Costello, Pradeep Dasigi, Dirk Groeneveld, David Heineman, Bailey Kuehl, Nathan Lambert, Chuan Li, Kyle Lo, Saumya Malik, DJ Matusz, Benjamin Minixhofer, Jacob Morrison, Luca Soldaini, Finbarr Timbers, Pete Walsh, Noah A.

Smith, Hannaneh Hajishirzi, and Ashish Sabharwal. Olmo hybrid: From theory to practice and back, 2026.

[17] Kurt Felix Michalak and Steven Abreu. Some attention is all you need for retrieval. In Mechanistic Interpretability Workshop at NeurIPS 2025, 2025.

[18] Elvis Nunez, Luca Zancato, Benjamin Bowman, Aditya Golatkar, Wei Xia, and Stefano Soatto. Expansion span: Combining fading memory and retrieval in hybrid state space models. In Proceedings of the International Conference on Neuro-symbolic Systems, volume 288 of Proceedings of Machine Learning Research, pages 570–596. PMLR, 28–30 May 2025.

[19] NVIDIA. Nemotron-h: A family of accurate and efficient hybrid mamba-transformer models, 2025.

[20] NVIDIA. Nvidia nemotron 3: Efficient and open intelligence, 2025.

[21] Destiny Okpekpe and Antonio Orvieto. Revisiting associative recall in modern recurrent models, 2025.

[22] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. Incontext learning and induction heads. Transformer Circuits Thread, 2022.

[23] Binghui Peng, Srini Narayanan, and Christos Papadimitriou. On limitations of the transformer architecture. In First Conference on Language Modeling, 2024.

[24] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling transformer inference. In D. Song, M. Carbin, and T. Chen, editors, Proceedings of Machine Learning and Systems, volume 5, pages 606–624. Curan, 2023.

[25] Zihan Qiu, Zekun Wang, Xiao Li, Yanpeng Li, Yang Xu, Yixuan Wang, Huaqing Zhang, Rui Men, Bochao Mao, Chengruidong Zhang, Fan Zhou, Hao Luo, Haofeng Huang, Haoran Lian, Haoyan Huang, Hongqing Chen, Jianwei Zhang, Jing Xu, Junjie Wang, Langshi Chen, Liangyu Wang, Linlang Jiang, Man Yuan, Minmin Sun, Peng Jin, Siqi Zhang, Siyu Wang, Xingzhang Ren, Yakai Wang, Yi Zhang, Yiming Dong, Yizhong Cao, Yubo Ma, Yunfei Mao, Bo Zheng, and Dayiheng Liu. On the design of qwen3.8-next architecture: Evaluation, efficiency, and training stability, 2026.

[26] Qwen Team. Qwen3-next-80b-a3b-instruct. https://huggingface.co/Qwen/ Qwen3-Next-80B-A3B-Instruct, 2025. Model card. Accessed 2026-09-08.

[27] Tim Roughgarden. Communication complexity (for algorithm designers). Foundations and Trends in Theoretical Computer Science, 11(3-4):217–404, 2016.

[28] Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Transformers, parallel computation, and logarithmic depth. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 43276–43327. PMLR, 21–27 Jul 2024.

[29] Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert Gu, Ali Hatamizadeh, Sudhakar Singh, Deepak Narayanan, Garvit Kulshreshtha, Vartika Singh, Jared Casper, Jan Kautz, Mohammad Shoeybi, and Bryan Catanzaro. An empirical study of mamba-based language models, 2024.

[30] Dustin Wang, Rui-Jie Zhu, Steven Abreu, Yong Shan, Taylor Kergan, Yuqi Pan, Yuhong Chou, Zheng Li, Jibin Wu, Ge Zhang, Wenhao Huang, and Jason Eshraghian. A systematic analysis of hybrid linear attention, 2026.

[31] Xiaojie Xia, Huigang Zhang, Chaoliang Zhong, Jun Sun, and Yusuke Oishi. Distill-then-replace: Efficient task-specific hybrid attention model construction, 2026.

[32] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations, 2025.

## Table of Contents

A Notation 17   
B Related Work 18   
B.1 Hybrid Models 18   
B.2 Mechanistic Tasks . 19   
C Experiment Details 19   
C.1 Two- and Three- Layer Hybrid Details . 19   
C.2 Learning Rate Selection . 20   
C.3 Memory Proxy Calculation 20   
D Additional Experiments 21   
D.1 Segmented Retrieval 21   
D.2 Different Quantities of Available Memory 21   
D.3 Testing DeltaNet Hybrids . 23   
D.4 Three Layers - Tensor Lookup 24   
D.5 An ST Task 24   
D.6 Associative Recall Variants 26   
D.7 Learning Rate Sensitivity 26   
E Proving Theorem 1 27   
E.1 Wrong use of ordering 28   
E.2 Correct ordering but insufficient layers 34   
E.3 Proof of Theorem 1 46   
F Omitted Details in Section 2.2 47   
F.1 Hardness Results 47   
F.2 Constructions 50   
F.3 SSMs with a Convolutional Front-End 55

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td>V U</td><td>The vocabulary space. The intermediate space used between architecture layers.</td></tr><tr><td>y S</td><td>The output space. An SSM layer.</td></tr><tr><td>T</td><td>A TF layer.</td></tr><tr><td> $\Sigma$   $A , B$ </td><td>Defined to be {S, T}, the possible letters in an architecture.</td></tr><tr><td> $\tau$ </td><td>Model architectures from  $\Sigma ^ { * }$ </td></tr><tr><td> $\mathrm { c o m p } ( \cdot )$ </td><td>The target architecture relating to Theorem 1.</td></tr><tr><td> $\alpha$ </td><td>Compressing all common layers into a single layer.</td></tr><tr><td> $\bar { \alpha }$ </td><td>The compression of  $\tau .$ </td></tr><tr><td></td><td>The flipped architecture,  $( \mathsf { S } , \mathbb { T } ) \mapsto ( \mathbb { T } , \mathsf { S } ) .$ </td></tr><tr><td> $s$   $W$ </td><td>The state space for the SSM layers.</td></tr><tr><td></td><td>The window for the TF layers.</td></tr><tr><td> $B$ </td><td>Number of bits available to an SSM layer.</td></tr><tr><td> $P$ </td><td>Number of bits available to the hidden space between layers.</td></tr><tr><td> $\mathcal { M } _ { A } ( B , P , W )$ </td><td>Model family of architecture A under the constraints B, P, W.</td></tr><tr><td> $d$ </td><td>The dimension of the hidden state.</td></tr><tr><td> $N$ </td><td>The Mamba state expansion factor.</td></tr><tr><td>X</td><td></td></tr><tr><td>f</td><td>An input sequence from V. A binary vector marking the endpoints of partitions for 2-layer tasks.</td></tr><tr><td> $L$ </td><td>The sequence length.</td></tr><tr><td> $\mathbf { W } _ { k } , \mathbf { W } _ { q } , \mathbf { W } _ { v }$   $u , r$ </td><td>Attention parameter matrices.</td></tr></table>

Table 2: Notation

## A Notation

We provide an overview of the notation used throughout the paper in Table 2.

An attention head Attn is defined by matrices $\mathbf { W } _ { k } , \mathbf { W } _ { q } , \mathbf { W } _ { v } \in \mathbb { R } ^ { d \times d }$ such that $\begin{array} { r } { \mathrm { A t t n } ( \vec { \bf x } ) _ { j } = \sum _ { i = 1 } ^ { n } \alpha _ { j i } { \bf W } _ { v } { \bf x } _ { i } } \end{array}$ where

$$
\alpha _ { j i } : = \frac { \exp \left( ( \mathbf { W } _ { q } \mathbf { x } _ { j } ) \cdot ( \mathbf { W } _ { k } \mathbf { x } _ { i } ) \right) } { \sum _ { i = 1 } ^ { n } \exp \left( ( \mathbf { W } _ { q } \mathbf { x } _ { j } ) \cdot ( \mathbf { W } _ { k } \mathbf { x } _ { i } ) \right) } .
$$

An attention layer AT is defined by H attention heads $\mathrm { A t t n } _ { 1 } , \ldots , \mathrm { A t t n } _ { H }$ and a projection matrix ${ \mathbf { W } _ { o } } \in \mathbf { \Sigma }$ $\mathbb { R } ^ { d \times d H }$ . Denote by $\mathbf { O } : = ( \mathrm { A t t n } _ { 1 } ( \vec { \mathbf { x } } ) ^ { \top } , \ldots , \mathrm { A t t n } _ { H } ( \vec { \mathbf { x } } ) ^ { \top } ) ^ { \top } \in \mathbb { R } ^ { d H \times L }$ the concatenation of the outputs of the H attention heads, so the attention layer $\mathrm { A T } ( \vec { \bf x } )$ outputs $\mathbf { W } _ { o } \mathbf { O } \in \mathbb { R } ^ { d \times L }$

A Transformer layer TF is defined by an attention layer AT and an MLP layer. In particular, an MLP layer is defined as $f ( \mathbf { x } ) : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ as $f ( \mathbf { x } ) = \mathbf { U } _ { 2 } \sigma ( \mathbf { U } _ { 1 } \mathbf { x } )$ , where $\mathbf { U } _ { 1 } , \mathbf { U } _ { 2 }$ are matrices and σ is an activation function applied coordinate-wise. Specifically, $\mathrm { T F } ( \vec { \bf x } ) = \mathrm { M L P } ( \mathrm { A T } ( \vec { \bf x } ) )$

An SSM layer SSM is defined with a state $S _ { t } ,$ , an update rule $s ,$ and an emission rule $^ r \cdot$ Let $s _ { 0 } \in { \mathcal { S } }$ be some initial state. Given some sequence $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { L }$ , the state of the model at iteration i is denoted by

$s _ { i } = S _ { i } ( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { i } )$ and the output token is denoted by $r _ { i } = F ( { \vec { \bf x } } ) _ { i } = R _ { i } ( { \bf x } _ { 1 } , . ~ . ~ , { \bf x } _ { i } )$ , where the state and output are defined recursively:

1. $S _ { 0 } ( \varnothing ) = s _ { 0 } .$

2. $S _ { i } ( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { i } ) = u ( S _ { i - 1 } ( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { i - 1 } ) , \mathbf { x } _ { i } ) ,$

3. $R _ { i } ( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { i } ) = r ( S _ { i } ( \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { i } ) ) ,$

## B Related Work

## B.1 Hybrid Models

Empirical Analysis on Hybrid Models. Hybrid designs now appear throughout frontier-scale open models, where attention placement is conventionally fixed and only the mixing ratio is tuned. Jamba [15] is one of the first models to combine attention, Mamba, and mixture-of-experts layers. It interleaves one attention layer for every seven Mamba layers and reports no substantial difference relative to a 1:3 ratio. Nemotron-H [19] allocates roughly 8% of layers to attention, spacing them evenly through the stack; Nemotron 3 [20] retains only a handful among interleaved Mamba-2 and mixture-of-experts layers; and Nemotron-Flash [7] selects its layer sequence via search under a latency budget. Kimi Linear [12], Olmo Hybrid [16], Qwen3- Next [26], and Qwen3.8-Next [25] all place one attention layer in every four. Within each period, attention follows the recurrent layers, except in Jamba where it appears among them.

Systematic studies explore hybrid design more broadly but still leave placement largely unexplored. [29] show at the 8B-parameter scale that a hybrid with 7% attention layers outperforms a Transformer on all twelve standard tasks. [30] sweep six linear-attention variants across five ratios, recommending 3:1 to 6:1 ratios of SSM to TF layers, and find that the best standalone linear model is not the best hybrid component. [14] compare sequential and parallel hybrids on memory recall. Only [2] vary where attention blocks are placed: at every ratio, placing attention in the early layers underperforms pure models, whereas placing it in the middle performs best. Some works have also observed that pure Transformers learn many heads that can be simulated closely by an SSM, allowing for distillation to more efficient architectures [31, 4].

Since each configuration requires a full pre-training run, deployed models each commit to a single interleaving, and the one controlled study of placement [2] explores only a narrow slice of the design space. Our synthetic sweeps cover a broad set of orderings within an 8-layer stack and show that the best ordering depends on the task.

Theoretical Understanding for Hybrid Models. Hybrid architectures have been studied less in a theoretical setting than in empirical settings. The work by [6] established a separation in memory resource constraints between hybrid architectures and pure architectures. We build on this work, studying hybrids of greater depth. Additionally, [16] proved hybrid architectures can implement tasks that exist in $\mathrm { N C ^ { 1 } }$ while both SSMs and Transformers are stuck in $\mathrm { T C ^ { 0 } }$ . This work establishes more tasks and distinguishes between layer orderings, which has yet to be studied using tools from communication complexity Communication complexity is a classical tool for converting information-transfer bottlenecks into computational lower bounds [13, 27]. It has recently become useful for understanding the expressivity of sequence models. [28] use communication-complexity arguments to establish representational limitations of attention, while [3] obtain separations between Transformers and recurrent architectures through reductions from communication problems. Closer to large language models, [23] use communication complexity to prove limitations of Transformers on function composition, and [5] introduce an autoregressive communication model to obtain unconditional depth–width lower bounds for multi-layer decoder-only Transformers and a provable advantage for chain-of-thought reasoning.

## B.2 Mechanistic Tasks

Associative recall. The most common task to test a model’s retrieval abilities is associative recall [22]. It probes a model’s capability to maintain an associative memory, keeping track of facts indexed by prior tokens. The original form of associative recall separated pure SSMs from pure TFs as worse at storing large memories.

This task, however, has grown too simple to the point where many similar architectures perform nearly perfectly. A few variants have been proposed, such as MQAR [1]–eliminating issues from only querying a single key-value pair–and phonebook [9]–where each key and value are made of multiple tokens. We focus on the last of these as a probe into the differences between hybrid architectures.

## C Experiment Details

We provide some additional experimental details for those conducted in the main body. We begin with a description of the layer architectures used. In Section C.1, we provide the experimental details omitted in the main body. In SectionC.2, we detail the learning rate sweeps conducted for each of the experiments. Lastly, in Section C.3, we describe the calculation used to determine the size of memory used by these models.

Transformers. The Transformers come from GPTNeoX and are given RoPE positional embeddings, both in the pure model and in all hybrids. All experiments use Transformer layers with a single head each. We observed that for this scale of experiments, increasing the number of heads by partitioning the square key and query matrices into thin rectangles hindered performance severely.

SSMs. The chosen SSM for all experiments is Mamba. This architecture allows for the state size to be increased independently of token dimension.

## C.1 Two- and Three- Layer Hybrid Details

Here, we detail the different experimental parameters relevant to each of the tasks tested.

Segmented Retrieval. We use a vocabulary of 40, with half of the vocabulary within the set N . We use sequences of length 150 and a partition of 10 segments. The segment boundaries are chosen uniformly at random in a sequence. We use a lookup distance of λ = 8, halfway to the expected segment length of 15. This prevents too many sequences from being so short as to not have any valid lookup, nor is λ so small that it falls within Mamba’s learned convolution. Using a uniform sampling of positions makes the task easier than if all segments were of length λ or greater. In fact, we observed that when λ is within this convolution length, the pure SSM architectures perform the task nearly as well as TS. It is possible to modify this task so the lookup distance is not fixed, though, circumventing the convolution altogether. Each model has 512 dimensions of memory. Due to the output space containing 20 tokens from the vocabulary, random performance is 0.05.

Exact Copy. For sequence lengths of 100, we have 20 vocabulary tokens. The copied sequence length is 5, which is repeated 3 times in the relevant sequences. To assist with training, we train the model to output 1s at every position after the exact copy is observed rather than only at the last token. Each model has 512 dimensions of memory. Due to the task being binary, random performance is 0.5.

TST Task. We use a sequence length of 150 and a vocabulary size of 7. The selection of 7 puts the final matrix in the last 1/3 of the context, and the initial lookup lies within the initial 1/3 of the context. This leaves a comfortable 1/3 of the middle of the context as null tokens, requiring an SSM layer to prevent large memory usage. We use a special copy of the vocabulary for the query token q to perform the first lookup. This helped the task be learnable by separating the tokens that are within the tensors and those which act as queries.

## C.2 Learning Rate Selection

In all experiments, we conduct a learning rate sweep to identify the best learning rate for each set of hyperparameters. For the segmented retrieval and TST tasks, 9 learning rates are sampled multiplicatively evenly between $1 0 ^ { - 4 }$ and $1 0 ^ { - 3 }$ . Each learning rate is run for one epoch and for 2 different sets. The learning rate with the smallest loss is selected as the learning rate for the 5 seeds tested.

For the exact copy experiments, 4 learning rates are multiplicatively spaced between $1 0 ^ { - 3 }$ and $1 0 ^ { - 1 }$ . For each of these, all 5 seeds are run for complete training runs. The best learning rate on average was selected. These experiments are thus more computationally expensive than the other tasks, although this worked in this setting, as we experimented on the other two tasks with deeper hybrids with more settings.

The learning rates for associative recall were selected with 2 epochs of training and a collection of learning rates between $1 0 ^ { - 5 }$ and $1 0 ^ { - 2 }$ . More details are provided in Appendix D.7.

## C.3 Memory Proxy Calculation

We calculate a proxy for the inference memory used, M, by each model in a simplified manner as follows.   
Let d be the hidden dimension of the model.

For a Transformer with window W, we account for the memory needed for the Transformer to store the tokens in its context. Thus, we calculate the memory needed as

$$
M _ { \mathrm { T F } } = d W .
$$

For a Mamba layer (also titled SSM), we use the state expansion factor N. The hidden state of a Mamba layer is a matrix of size $N \times d .$ . Thus, we calculate the memory needed as

$$
M _ { \mathrm { S S M } } = d N .
$$

Lastly, for the few experiments containing Gated DeltaNet, we only need the token dimension. The state for these layers is a square matrix of the same size as the input. Thus, we calculate the memory needed as

$$
M _ { \mathrm { G D N } } = d ^ { 2 } .
$$

For a model with $N _ { \mathrm { T F } }$ Transformer layers, $N _ { \mathrm { S S M } }$ Mamba layers, and $N _ { \mathrm { G D N } }$ GDN layers, the total memory is

$$
M = N _ { \mathrm { T F } } d W + N _ { \mathrm { S S M } } d N + N _ { \mathrm { G D N } } d ^ { 2 } .
$$

Enforcing identical memory. When enforcing that each experiment uses the same allotted memory, we only modify W and N. The desired token dimension d is maintained. Additionally, the ratio W/N is roughly maintained so that a desired weight budget favoring one of these layer types over the other is reflected in the used parameters in the experiment.

When W and N are calibrated, they are always rounded down to stay under the required budget. However, we select memory budgets and token dimensions as powers of 2 so all experiments have precisely equal memory.

## D Additional Experiments

We provide some additional experiments covering other design decisions and further observations relevant to deeper hybrids.

We begin with results directly related to those in the main body. Those include experiments on the tasks with dominant two layer architectures with more memory and more experiments between dimensions that are powers of 2. We then include some experiments with DeltaNet hybrids, noting that the available design space is much more limited as the state is tied only to token dimension. Some other tasks are then investigated, including the TST tensor lookup task comparing al 3-layer hybrids as well as a task expected to be optimal for ST. This final task comes from prior work and confirms that the codebase does not only bias towards the TS architecture. Lastly, we investigate the results of 8-layer hybrids on two variants of associative recall. We also report their learning-rate sensitivities, noting that for these tasks, each of these architectures shows a surprisingly large range of usable learning rates.

## D.1 Segmented Retrieval

In the main body, we showed the results for a sweep of token dimensions, windows, and state sizes. These experiments use token dimensions that are all powers of 2. We conducted additional experiments with more token dimensions, with results shown in Figure 6. These are no longer powers of two and thus vary slightly in memory. However, the results still mimic those in the main body, where only the TS architecture performs the task.

These additional points also allow us to observe an interesting structure between the TT and TS architectures. The TT architecture attains its greatest average performance at a smaller token dimension and larger window than the TS architectures. These optima show a clear improvement in memory efficiency for TS architectures: the TS architectures use their allotted state to increase the computation per token instead of dedicating much of it to a large window to capture only some small components early in the sequence.

## D.2 Different Quantities of Available Memory

We first showed a separation between hybrid architectures on segmented retrieval and exact copy using 512 dimensions of memory. We also test architectures with larger and small quantities of available memory.

The results for double the memory can be seen in Figure 7. We still observe that the TS architecture is the highest-performing architecture. However, for exact copy, unlike the 512 memory experiments, the TT architecture gains significantly in performance at small token dimensions. This is likely due to the increased window compared to the main body, where dimension 8 only leaves a window of 128 now, which is large enough to capture most copied subsequences. For the segmented retrieval experiments, we observe that performance on all architectures increases compared to 512 dimensions of memory.

![](images/c34c4f09b4d553b6ca7b8f3a0b1ecce3e39a5f425b95e40f9c8c5292b2fff03d.jpg)  
Figure 6: Extended results for segmented retrieval.

![](images/08393969a7de439322d3e0482170ac68616a862617d36a48887952ca17234607.jpg)  
(a) The Segmented Retrieval task.

![](images/598e7cbce829b2523fdb1a8a9554e82687bb1a531ba445b290c1743485c5f3a7.jpg)  
(b) The Exact Copy task.  
Figure 7: Hybrid performance on exact copy for 1024 dimensions of memory.

## D.3 Testing DeltaNet Hybrids

The majority of this work focuses on the Mamba SSM architecture. This architecture includes a state-expansion factor that allows the size of the state to be changed independently of other model parameters. This section describes an experiment conducted on hybrids of Transformer and Gated DeltaNet (GDN) layers.

An important note about Gated DeltaNet is that its state matrix, $S _ { t } .$ , exists in $\mathbb { R } ^ { d \times d }$ . This state size is tied to the token dimension. Thus, unlike the Transformer-Mamba hybrids where the window-tostate-size dimension was free, we can only modify the token dimension and window to achieve the desired memory.

The results are in Figure 10. For the Transformer-Mamba hybrids, we fix the ratio of window to state-expansion to be 15 to 1, matching the highest performance seen across ratios. We observe the same behavior for Gated DeltaNet as with Mamba, with TS mirroring TG closely. However,<sup></sup>

![](images/69c95c91bd06d465f855b20611d1103bab8f133abe475de2357f623a6cf88fe4.jpg)  
Figure 9: Results for training various architectures with different token dimensions on the TST task. All models used the same window size and state size. As expected, only the TST model attains a high accuracy.

due to the larger state size of GDN as compared to Mamba, larger token dimensions violate the memory constraints faster; hence, the TG and GT curves end earlier. This confirms that available inference memory is a core bottleneck across a couple of different SSM architectures.

![](images/788399efbfb73b39b2f2d763306915e5a505d28d123a3015ad2336bce499f8dc.jpg)  
(a) The Segmented Retrieval task.

![](images/daa9c99806c074f6e316f9dd0e3f5a591b6bbd04c786dbb3543f972c279c4a97.jpg)  
(b) The Exact Copy task.  
Figure 8: Hybrid performance on exact copy for 256 dimensions of memory.

## D.4 Three Layers - Tensor Lookup

As established by the theory about deeper hybrids, there is a natural tensor-lookup task that is predicted to have the TST architecture be dominant. We investigate this task empirically as well to test the results in a learned setting.

These experiments mirror closely to those for the two-layer architectures. The architectures used are also Transformers with RoPE and Mamba for the SSM. In these experiments, we also hold memory constant. However, we only test on one window-to-state-size ratio for each architecture: 15 to 1. The results are found in Figure 9. As expected, the TST model performs the best, reaching an accuracy above 90%, unlike all competing architectures.

We report the exact average performances from Figure 3 in Table 3. This table lists all possible 5-layer architectures. All models with the TST motif, with the exception of 1, performed better than the remaining models. All models were held to roughly the same memory of 6000. One additional takeaway from this table comes from the architectures of the form $\boldsymbol { \mathrm { T } } ^ { * } \boldsymbol { \mathrm { S } } ^ { * }$ and ${ \mathbf { S } } ^ { * } { \mathbf { T } } ^ { * }$ . For the $\boldsymbol { \mathrm { T } } ^ { * } \boldsymbol { \mathrm { S } } ^ { * }$ models, there is a bias towards more T layers, while for ${ \mathbf { S } } ^ { \ast } { \mathbf { T } } ^ { \ast }$ , there is a bias towards fewer S layers. Differences such as these support the need for a more principled understanding of various hybrid architecture designs.

## D.5 An ST Task

In [6], the selective copying task was used to show a separation in hybrid expressivity over pure model expressivity. Empirically, this task was shown to be better suited for ST rather than TS. To confirm that the code used does not artificially bias the TS architecture towards being the dominant architecture, we run the same suite of experiments as used for the 2-layer TS tasks.

The results are found in Figure 11. We observe that the ST architectures do indeed achieve perfect performance on this task, while the other architectures do not. Thus, the codebase trains architectures in a way that can show various architectures as the highest performing.

Beyond this confirmation, we observe some additional effects in their own right. First, the TF and SSM architectures plateau at different locations in the figure. In fact, the SSM Mamba continues to increase in performance as the dimension increases and the state expansion factor shrinks.

![](images/b075733fd44333f37a8f131c9206bd008bff677cd3424fbd02b43e83a97825f0.jpg)  
Figure 10: Performance on segmented retrieval, including Gated DeltaNet (G) hybrids. Only the 15:1 ratio of window to SSM state for TS and ST hybrids is shown.

<table><tr><td>Model</td><td>Performance</td><td>Model</td><td>Performance</td></tr><tr><td>TTTST</td><td>0.911</td><td>TTTTT</td><td>0.533</td></tr><tr><td>TTSTT</td><td>0.934</td><td>TTTTS</td><td>0.840</td></tr><tr><td>TTSTS</td><td>0.884</td><td>TTTSS</td><td>0.776</td></tr><tr><td>TTSST</td><td>0.890</td><td>TTSSS</td><td>0.787</td></tr><tr><td>TSTTT</td><td>0.922</td><td>STTTT</td><td>0.527</td></tr><tr><td>TSTTS</td><td>0.923</td><td>STTTS</td><td>0.842</td></tr><tr><td>TSTST</td><td>0.926</td><td>TSSSS</td><td>0.659</td></tr><tr><td>TSTSS</td><td>0.888</td><td>STTSS</td><td>0.770</td></tr><tr><td>TSSTT</td><td>0.967</td><td>STSSS</td><td>0.635</td></tr><tr><td>TSSTS</td><td>0.893</td><td>SSTTT</td><td>0.550</td></tr><tr><td>TSSST</td><td>0.949</td><td>SSTTS</td><td>0.696</td></tr><tr><td>STTST</td><td>0.900</td><td>SSSTT</td><td>0.823</td></tr><tr><td>STSTT</td><td>0.948</td><td>SSTSS</td><td>0.650</td></tr><tr><td>STSTS</td><td>0.800</td><td>SSSTS</td><td>0.709</td></tr><tr><td>STSST</td><td>0.959</td><td>SSSST</td><td>0.831</td></tr><tr><td>SSTST</td><td>1.000</td><td>SSSSS</td><td>0.478</td></tr></table>

Table 3: Quantitative results for a deeper hybrids on the TST task. The left architectures contain TST as a sub-architecture, while the other three do not.

Because of these different regimes of success for each of the component architectures, the hybrid architectures also have two local maxima in their performance curves. This may indicate that for the hybrids with larger token dimension and smaller window and state-expansion factor, the pure-SSM algorithm is favored. In the other regime with moderate token dimension, the TS architecture likely implements an algorithm not too different to the pure TF one.

## D.6 Associative Recall Variants

We have focused on experiments pertaining to (2, 1)-AR as this mirrors a compressed version of multidocument tasks. However, we also want to consider the other variant of associative recall, namely (1, 2)-AR. This task instead has only |V| possible keys instead of |V|<sup>2</sup>, making the key space much easier for an SSM to store.

The results for both are shown in Figure 12. The new variant of associative recall, (1, 2)-AR, shows on average a higher performance, likely due to the decreased size in the key space. However, we also see that the top-performing hybrid layer ordering is different for each task. While both tasks favor low Transformerto-SSM ratios, (1, 2) recall sees stronger performance with more T-Layers than for (2, 1)-AR.

## D.7 Learning Rate Sensitivity

Prior work has established that Mamba has a large learning rate sensitivity on the associative recall task [21]. As a result, we ran learning rate sweeps within our own codebase. The results can be seen in Figure 13. Surprisingly, SSMs showed remarkable resilience to learning rate selection. However, other SSMheavy architectures did show a very narrow band of learning rates that resulted in the highest performance.

![](images/183fc28d6cb5c24c396ed07793361129ad1f35677787c78ee8b4ba8476a4d491.jpg)  
Figure 11: Performance of architectures trained on selective copy. We observe that the ST architecture is largely dominant, as expected.

However, the order of the different architectures’ optimal performances remains mostly intact when selecting a learning rate in the optimal band between $1 0 ^ { - 4 }$ and $1 0 ^ { - 3 }$

## E Proving Theorem 1

This section is dedicated to the needed machinery and proofs to prove Theorem 1. We begin with two lemmas from prior work that allow for neighboring layers of the same type to be combined. We then complete proofs for the two parts of Theorem 1: the first focuses on hybrids of the wrong order (Theorem 3, while the second focuses on hybrids of the correct order but with insufficient layers (Theorem 4).

Model Merge. We borrow two lemmas from prior work. These allow neighboring layers of common type to be combined.

Lemma 1 (Lemma 3.5 in [6]). Consider a k-layer model M defined that is a composition of k state space layers $S S M _ { j } , j \in [ k ]$ . There is a model SSM<sup>′</sup> that only consists of a single layer SSM, which behaves the same as M. In particular, denote the state space of $S S M _ { j }$ as $S _ { j } , j \in [ k ]$ , and denote by the state space of SSM<sup>′</sup> as $S ^ { \prime } { } _ { ; }$ , then $\begin{array} { r } { | S ^ { \prime } | \leq \prod _ { j = 1 } ^ { k } | S _ { j } | } \end{array}$

![](images/908e70cb67d58dd4d8d6d85a8cb9d0b2cc36f5b9f393e99a5349cf8cdb13387e.jpg)

![](images/3b30a740f181348a20b8cdaf1ab5630578a75b5e777574e3837b48f8b53b2807.jpg)  
(a) Architecture sweep for the (2, 1)-AR task.  
(b) Architecture sweep for the (1, 2)-AR task.  
Figure 12: Token accuracy of $\boldsymbol { \mathrm { T } } ^ { * } \boldsymbol { \mathrm { S } } ^ { * }$ and ${ \mathbf { S } } ^ { \ast } { \mathbf { T } } ^ { \ast }$ architectures on (2, 1) associative recall. Small differences in architectures affect performance dramatically.

Lemma 2. Consider a k-layer model M defined that is a composition of k causal windowed functions $f _ { j } , j \in [ k ]$ . There is a model M<sup>′</sup> that only consists of a single layer causal windowed function $f ,$ which behaves the same as M. In particular, denote the window size of $f _ { j }$ as $W _ { j } , ~ j ~ \in ~ [ k ]$ , and denote by the window size off as W, then $\begin{array} { r } { W ^ { \prime } \leq \sum _ { j = 1 } ^ { k } W _ { i } - k + 1 } \end{array}$

## E.1 Wrong use of ordering

We start with the case where a competitive model A does not contain comp(τ ) as a subsequence.

Theorem 3. Let $D \in \mathbf { N } _ { + }$ and $\tau = X _ { 1 } ^ { r _ { 1 } } X _ { 2 } ^ { r _ { 2 } } \cdot \cdot \cdot X _ { h } ^ { r _ { h } } \in \Sigma ^ { * }$ be an arbitrary architecture profile with depth $\begin{array} { r } { D = \sum _ { i = 1 } ^ { h } r _ { j } } \end{array}$ . Let $\alpha : = \mathrm { c o m p } ( \tau )$ and $\epsilon \in ( 0 , 1 )$ be an arbitrary fixed constant. There exist constants $c _ { \tau } > 0 , \breve { C } _ { \tau , \epsilon } > 0 .$ , afamily oftasks $\{ F _ { \tau , \epsilon , m } \} _ { m \ge 1 }$ and afamily ofdistributions $\mu _ { m }$ over a sequence oftokens of length $L _ { m }$ parameterized by $m \in \mathbf { N } _ { + }$ , such that for all m sufficiently large, the following hold. Let $\bar { A } \in \bar { \Sigma } ^ { \leq D }$ be an architecture profile with α ̸⪯ A. Then for every hybrid model $M \in \mathcal { M } _ { A } ( B , P , W )$ that satisfies er $\cdot ( M , F _ { \tau , \epsilon , m } ) \leq 1 / 3$ , must satisfy $W \ge c _ { \tau } m ^ { 1 / \epsilon }$ or max $\{ B , P \} \ge c _ { \tau } m$

To begin with, we present the following model simplification lemma.

Lemma 3. $L e t \tau = X _ { 1 } ^ { r _ { 1 } } X _ { 2 } ^ { r _ { 2 } } \cdot \cdot \cdot X _ { h } ^ { r _ { h } } \in \Sigma ^ { * }$ be an arbitrary architecture with depth D and $\alpha = \mathrm { c o m p } ( \tau )$ Denote by ${ \bar { \alpha } = \bar { X } _ { 1 } , \ldots , \bar { X } _ { h } }$ be the complementary of α, where ${ \bar { X } } _ { i } \neq X _ { i } \in \Sigma$ . Let $A \in \Sigma ^ { \leq D }$ such that $\alpha \not \simeq A ,$ , then comp $( A ) \preceq \bar { \alpha }$ . Furthermore, for $B , P , W \in \mathbf { N } _ { + } ,$ , and $M \in \mathcal { M } _ { A } ( B , P , W )$ , there exist a hybrid model M<sup>′</sup> in $\mathcal { M } _ { \bar { \alpha } } ( D B , P , D ( W - 1 ) )$ ) that can represent M.

ProofofLemma 3. Since $\alpha = X _ { 1 } , \ldots , X _ { k }$ and $\mathrm { c o m p } ( A )$ is an alternating architecture profile, we consider two cases. In the first case, comp(A) begins with $X _ { 1 }$ . We claim that it must be the case where $| \mathrm { c o m p } ( A ) | \leq$ $k - 1$ , otherwise, $\alpha \preceq \mathrm { c o m p } ( A )$ . This implies that comp(A) must be a substring of $\bar { \alpha } ,$ starting from the second layer. comp $( A ) \preceq \bar { \alpha }$ . In the second case, comp(A) start with ${ \bar { X } } _ { 1 }$ . In this case, $| \mathrm { c o m p } ( A ) | \le k$ otherwise $\alpha \preceq \mathrm { c o m p } ( A )$ . So, comp $( A ) \preceq \bar { \alpha }$

![](images/b89c804f3978599c6baa43d2a016dffc21529a95e5fb6608c0968bd50f770acd.jpg)  
(a) Sweep for (2, 1)-AR.

![](images/c3ba08d11f6cd6c44f75bbdefe1ab531ddba16cae230a36ef6cee2858904cf71.jpg)  
(b) Sweep for (1, 2)-AR.  
Figure 13: Learning rate sweep for select architectures trained on associative recall.

Since $\mathrm { c o m p } ( A ) \preceq \bar { \alpha }$ , the second part of Lemma 3 follows directly from Lemma 1 and Lemma 2.

Lemma 3 implies that instead of competing with an arbitrary architecture profile, it is sufficient to compete with hybrid models in $\mathcal { M } _ { \bar { \alpha } } ( D B , P , D ( W - 1 ) )$ . A central part of our analysis is to characterize the format and dependence of information through different layers and at different positions.

Before proceeding, we first introduce some notations that we will use in the rest of the analysis to keep track of different types of layers. Let $\bar { \alpha } = \bar { X } _ { 1 } \bar { X } _ { 2 } \ldots \bar { X } _ { h }$ and $M ^ { \prime } \in \mathcal { M } _ { \bar { \alpha } } ( D B , P , D ( W - 1 ) )$ ). Denote by $k = \{ i \in [ h ] \mid \bar { X } _ { i } = S \}$ . We introduce indicator $\delta _ { L } = \mathbf { 1 } \{ \bar { X } _ { 1 } = \mathrm { T } \}$ and $\delta _ { R } = \mathbf { 1 } \{ \bar { X } _ { h } = \mathrm { T } \}$ . That is to say, we can write $\bar { \alpha } = \mathbb { T } _ { 0 } ^ { \delta _ { L } } \mathbb { S } _ { 1 } \mathbb { T } _ { 1 } , \dots , \mathbb { S } _ { k - 1 } \mathbb { T } _ { k - 1 } \mathbb { S } _ { k } \mathbb { T } _ { k } ^ { \delta _ { R } }$ , where $\boldsymbol { \mathrm { T } } _ { 0 } ^ { 0 } , \boldsymbol { \mathrm { T } } _ { s } ^ { 0 }$ mean the corresponding boundary layers are absent. In particular, we have $h = 2 k - 1 + \delta _ { L } + \delta _ { R }$

We introduce the family of tasks we are considering as well as the format of input.

Definition 5. Let $\mathcal { X } _ { k e y }$ be a finite space of keys, and C be a finite space of vocabularies. Let $\epsilon \in ( 0 , 1 )$ and $m \in \mathbf { N } _ { + }$ . Let τ be an architecture profile with $\bar { \alpha } = \mathbb { T } _ { 0 } ^ { \delta _ { L } } \mathrm { S } _ { 1 } \mathrm { T } _ { 1 } , \dots , \mathrm { S } _ { k - 1 } \mathrm { T } _ { k - 1 } \mathrm { S } _ { k } \mathrm { T } _ { k } ^ { \delta _ { R } }$ . A level-k recursive chain $F _ { \tau , \epsilon , m }$ is defined by a sequence of k functions. Specifically, for $i \in [ k ] , H _ { i } ( \mathbf { z } _ { i - 1 } , \mathbf { x } _ { i } ) : \mathcal { X } _ { k e y } \times \mathcal { C } ^ { m } $ $\chi _ { k e y } ,$ , where $Z _ { i } : = H _ { i } ( \mathbf { z } _ { i - 1 } , \mathbf { x } _ { i } )$ . Given a sequence of input $I : = ( \mathbf { z } _ { 0 } , \vec { 0 } _ { 0 } , \mathbf { x } _ { 1 } , \vec { 0 } _ { 1 } , \ldots , \mathbf { x } _ { k } , \vec { 0 } _ { k } )$ of length $L _ { m } .$ $F _ { \tau , \epsilon , m } ( I ) : = \mathbf { z } _ { k }$ , where each $\vec { 0 } _ { i }$ is a sequence of 0 vectors.

Let $\Delta _ { m } : = D \lfloor m ^ { 1 / \epsilon } \rfloor$ . We define position variables in the input sequence I. We consider two cases. If $\delta _ { L } = 0$ , we set $u _ { 1 } = 2 \quad$ , otherwise, we set $u _ { 1 } = \Delta _ { m } + 1 , v _ { 1 } = u _ { 1 } + m - 1$ and for $i \ge 1 , b _ { i } = v _ { i } + i \Delta _ { m } ,$ $u _ { i + 1 } = b _ { i } + \Delta _ { m } + 1 , v _ { i + 1 } = u _ { i + 1 } + m - 1$ . Then, $z _ { 0 } = I _ { 1 } , x _ { i } = I _ { [ u _ { i } , v _ { i } ] }$ and the other positions of S are 0. For the output position, if $\delta _ { R } = 1$ , we ask the output to be at $b _ { k } ;$ ; otherwise, $\vec { o } _ { k }$ does not exist and we ask the output to be as $v _ { k }$

The input distribution $\mu _ { m }$ for I that we are considering samples an input sequence I by sample $\mathbf { z } _ { 0 }$ and each $\mathbf { x } _ { i }$ uniformly and independently from the domains.

We remark that Definition 5 immediately implies the existence of a model in $\mathcal { M } _ { \tau }$ that can efficiently solve the task exactly. To see this, we consider a model M with profile $\alpha = \mathrm { c o m p } ( \tau )$ , for every $i \geq 1$ , since the window size of a T layer is $m + 1$ , we use $O _ { \tau } ( \log m )$ bits to feed $\left( ( \mathbf { z } _ { i - 1 } , \mathbf { x } _ { i , 1 } ) , \ldots , ( \mathbf { z } _ { i - 1 } , \mathbf { x } _ { i , m } ) \right)$ to $\mathrm { T } ,$ which is enough for it to compute $\mathbf { z } _ { i } ,$ precisely. This implies that if $\mathcal { X } _ { k e y }$ has size $\mathrm { p o l y } _ { \tau } ( m )$ , then the next S-layer can use $O _ { \tau } ( \log m )$ to feed $( ( \mathbf { z } _ { i } , \mathbf { x } _ { i + 1 , 1 } ) , \ldots , ( \mathbf { z } _ { i } , \mathbf { x } _ { i + 1 , m } ) )$ to the next T and thus it can solve the task exactly. ${ \mathrm { S o } } ,$ in the rest of the analysis we focus on the hardness part.

Given an input sequence I, we denote by $Y ^ { ( j ) } ( I )$ the intermediate input of $\mathrm { T } _ { j }$ and denote by $h _ { j } ( t ; I )$ the internal state of $\mathrm { S } _ { j }$ after processing position t. By Lemma 3, we without loss of generality assume that $h _ { j } ( t ; I ) \ \in \ \{ 0 , 1 \} ^ { \mathsf { ^ { D B } } }$ . Based on the notation of Definition 1, for $0 \ \leq \ j \ \leq \ i ,$ we define position $a _ { i j } = v _ { i } + j \Delta _ { m }$ . In particular, $a _ { i 0 } = v _ { i }$ and $a _ { i i } = b _ { i }$ . We define the state record at stage i with input I as

$$
C _ { i } ( I ) : = ( h _ { 1 } ( a _ { i 1 } ; I ) , h _ { 2 } ( a _ { i 2 } ; I ) , \dots , h _ { i } ( a _ { i i } ; I ) ) .
$$

Our first lemma shows that if $C _ { i }$ is fixed, then for every $\ell > b _ { i } , Y _ { \ell } ^ { ( i ) }$ is only a deterministic function of the parts of the input after $b _ { i }$

Lemma 4. Let $D \in \mathbf { N } _ { + }$ and $\tau = X _ { 1 } ^ { r _ { 1 } } X _ { 2 } ^ { r _ { 2 } } \cdot \cdot \cdot X _ { h } ^ { r _ { h } } \in \Sigma ^ { * }$ be an arbitrary architecture with $\alpha = \mathrm { c o m p } ( \tau )$ and depth D. Denote by $\bar { \alpha } = \bar { X } _ { 1 } , \dots , \bar { X } _ { h }$ be the complementary of α, where $\bar { X } _ { i } \neq X _ { i } \in \Sigma$ . Let $M ^ { \prime } \in$ $\mathcal { M } _ { \bar { \alpha } } ( D B , P , D ( W - 1 ) )$ be a hybrid model. Let I, I<sup>′</sup> be two sequences of inputs of the level-k recursive chain $F _ { \tau , \epsilon , m }$ and let $i \in [ k ]$ . Suppose that $I _ { \ell } = I _ { \ell } ^ { \prime } ,$ for every $\ell > v _ { i }$ and $C _ { i } ( I ) = C _ { i } ( I ^ { \prime } )$ , then for every $j \in [ i ]$ and for every $\ell > \bar { a _ { i j } } , Y ^ { ( j ) } ( I ) _ { \ell } = Y ^ { ( j ) } ( I ^ { \prime } ) _ { \ell } ,$ and for every $\ell \geq a _ { i j } \ h _ { j } ( \ell ; I ) = h _ { j } ( \ell ; I ^ { \prime } )$

ProofofLemma 4. We prove the statement by induction. For $j = 1$ , we consider two cases. In the first case, $\delta _ { L } = 0 ,$ which implies that α¯ begins with $\mathrm { S } _ { 1 } .$ . In this case, the states of $h _ { 1 }$ only depend on the past states and input and thus we have for every $\ell \geq a _ { i 1 } \ h _ { 1 } ( \ell ; I ) = h _ { 1 } ( \ell ; I ^ { \prime } )$ . In the second case, α¯ begins with $\mathrm { T } _ { 0 } \mathrm { S } _ { 1 }$ . By the choice of $\Delta _ { m } .$ , for every $\ell > v _ { i } + \Delta _ { m } = a _ { i 1 } , \mathrm { T } _ { 0 } ( I ) _ { \ell } = \mathrm { T } _ { 0 } ( I ^ { \prime } ) _ { \ell }$ , which implies that $h _ { 1 } ( \ell ; I ) = h _ { 1 } ( \ell ; I ^ { \prime } )$ The results for the states automatically translate to the results for $Y ^ { ( 1 ) }$

Now suppose the statement holds for $j .$ We prove it for the case of $j + 1$ . Since the window size of a transformer layer is at most $\Delta _ { m }$ , we know that the input of $\mathrm { T } _ { \mathit { j } }$ only depends on positions after $a _ { i j }$ when evaluating after $a _ { i ( j + 1 ) }$ . Since for every $\ell > a _ { i j } , Y ^ { ( j ) } ( I ) _ { \ell } = \overset {  } { Y ^ { ( j ) } } ( I ^ { \prime } ) _ { \ell }$ , we know that $\mathrm { T } _ { j } ( Y ^ { ( j ) } ( { \bar { I } } ) ) _ { \ell } =$ $\mathrm { T } _ { j } \big ( Y ^ { ( j ) } ( I ^ { \prime } ) \big ) _ { \ell }$ for $\ell > a _ { i ( j + 1 ) }$ . On the other hand, since for every $\ell > a _ { i j } , h _ { j } ( \ell ; I ) = h _ { j } ( \ell ; I ^ { \prime } )$ , we know that for every $\ell > a _ { i ( j + 1 ) } , h _ { j + 1 } ( \ell ; I ) = h _ { j + 1 } ( \ell ; I ^ { \prime } )$ and thus $Y ^ { ( j + 1 ) } ( I ) _ { \ell } = Y ^ { ( j + 1 ) } ( I ^ { \prime } ) _ { \ell }$

Next, we characterize the evolution of $C _ { i }$ when i increases. i.e how the record of the hidden states of each SSM layer changes according to the position.

Lemma 5. Let $D \in \mathbf { N } _ { + }$ and $\tau = X _ { 1 } ^ { r _ { 1 } } X _ { 2 } ^ { r _ { 2 } } \cdot \cdot \cdot X _ { h } ^ { r _ { h } } \in \Sigma ^ { * }$ be an arbitrary architecture with $\alpha = \mathrm { c o m p } ( \tau )$ and depth D. Denote by $\bar { \alpha } = \bar { X } _ { 1 } , \ldots , \bar { X } _ { h }$ be the complementary of α, where ${ \bar { X } } _ { i } \neq X _ { i } \in \Sigma$ . Let $M ^ { \prime } \in$ $\mathcal { M } _ { \bar { \alpha } } ( D B , P , D ( W - 1 ) )$ be a hybrid model. Let I be an input of a level-k recursive chain, realized $b y$ $\left( \mathbf { z } _ { 0 } , \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { k } \right)$ . Denote by $J _ { i }$ the prefix $\left( \mathbf { z } _ { 0 } , \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { i } \right)$ . For every $i \in [ k - 1 ]$ and for every c such that $C _ { i } ( I ) = c ,$ there exists a map $H _ { i , c } : { \mathcal { C } } ^ { m }  \{ 0 , 1 \} ^ { i D B } , E _ { i , c } : { \mathcal { C } } ^ { m }  { \mathcal { C } } ^ { b _ { i + 1 } - u _ { i + 1 } + 1 } , \Phi _ { h _ { i + 1 } ( u _ { i + 1 } - 1 ; I ) } :$ $\mathcal { C } ^ { b _ { i + 1 } - u _ { i + 1 } + 1 }  \{ 0 , 1 \} ^ { D B }$ such that $( h _ { 1 } ( a _ { ( i + 1 ) 1 } ; I ) , h _ { 2 } ( a _ { ( i + 1 ) 2 } ; I ) , \dots , h _ { i } ( a _ { ( i + 1 ) i } ; I ) ) = H _ { i , c } ( \mathbf { x } _ { i + 1 } )$ and $h _ { i + 1 } ( a _ { ( i + 1 ) ( i + 1 ) } ; I ) = \Phi _ { h _ { i + 1 } ( u _ { i + 1 } - 1 ; J _ { i } ) } ( E _ { i , c } ( \mathbf { x } _ { i + 1 } ) )$

In other words, once we have fixed the prefix of the input $J _ { i }$ as well as the value of $C _ { i } ( I )$ , there exists maps $H _ { i , c } , E _ { i , c }$ that only depends on the compressed information $C _ { i } ( I )$ and a map $\Phi _ { h _ { i + 1 } \left( u _ { i + 1 } - 1 ; I \right) }$ that only depends on the state $h _ { i + 1 } ( u _ { i + 1 } - 1 ; I )$ such that $C _ { i + 1 } ( I ) = ( H _ { i , c } ( \mathbf { x } _ { i + 1 } ) , \Phi _ { h _ { i + 1 } ( u _ { i + 1 } - 1 ; I ) } ( E _ { i , c } ) ( \mathbf { x } _ { i + 1 } ) )$ . To simplify the notation, for any fixed $i , c$ and $h _ { i + 1 } ( u _ { i + 1 } - 1 ; I ) = \gamma$ , we denote by $C _ { i + 1 } ( I ) = \Gamma _ { i , c , \gamma } ( \mathbf { x } _ { i + 1 } )$

ProofofLemma 5. We start by proving the existence of $H _ { i , c }$ . By Lemma $^ { 4 , }$ we know that once we fix $C _ { i } ( I ) = c _ { \mathrm { { } } }$ , for every $j \le i$ and for every $\ell \geq a _ { i j }$ , the hidden state $h _ { j } ( \ell , I )$ is only a function of $\mathbf x _ { i + 1 }$ . This implies that the vector $( h _ { 1 } ( a _ { ( i + 1 ) 1 } ; I ) , h _ { 2 } ( a _ { ( i + 1 ) 2 } ; I ) , \dots , h _ { i } ( a _ { ( i + 1 ) i } ; I ) )$ only depends on $\mathbf { x } _ { i + 1 }$ , given the current state $C _ { i } ( J _ { i } )$

We next prove the existence of $\Phi _ { J _ { i } }$ and $E _ { i , c }$ . We notice that the state $h _ { i + 1 } ( a _ { ( i + 1 ) ( i + 1 ) } ; I )$ is a function of the state $h _ { i + 1 } ( b _ { i } , I )$ and the input context $\boldsymbol { \mathrm { T } } _ { i } \big ( \boldsymbol { Y } ^ { ( i ) } \big )$ . In particular, if we denote by δ the transition function of the state of $\mathrm { S } _ { i + 1 }$ . Then, it holds that for $j \geq 0$ , we have $h _ { i + 1 } ( u _ { i + 1 } + j ; I ) = \delta ( h _ { i + 1 } ( u _ { i + 1 } + j -$ $1 ; I ) , \mathrm { T } _ { i } ( Y ^ { ( i ) } ( I ) ) _ { u _ { i + 1 } + j } )$

By Lemma 4, we know that for $\ell > a _ { i i } , Y ^ { ( i ) } ( I ) _ { \ell }$ is a function that only depends on $\mathbf x _ { i + 1 }$ . This implies that given $C _ { i } = c , \mathrm { T } _ { i } ( Y ^ { ( i ) } ( I ) ) _ { u _ { i + 1 } + j } )$ , for $j = 0 , \ldots , b _ { i } - v _ { i }$ only depends on $\mathbf { x } _ { i + 1 }$ , while the state $h _ { i + 1 } ( u _ { i + 1 } - 1 ; I )$ can depend on the entire prefix $J _ { i }$ . This implies that there exist maps $\boldsymbol { E } _ { i , c } :$ $\mathcal { C } ^ { m } ~ \to ~ \mathcal { C } ^ { \hat { b } _ { i + 1 } - u _ { i + 1 } + 1 } , \Phi _ { h _ { i + 1 } ( u _ { i + 1 } - 1 ; I ) } ~ : ~ \mathcal { C } ^ { { b } _ { i + 1 } - u _ { i + 1 } + 1 } ~ \to ~ \{ 0 , 1 \} ^ { D B }$ such that $h _ { i + 1 } ( a _ { ( i + 1 ) ( i + 1 ) } ; I ) \ =$ $\Phi _ { h _ { i + 1 } ( u _ { i + 1 } - 1 ; I ) } ( E _ { i , c } ( \mathbf { x } _ { i + 1 } ) )$ □

Based on Lemma 4 and Lemma 5, we now give the construction for the functions $H _ { i }$ that we are using. Our construction follows the following intuition. By Lemma 4 and Lemma 5, conditioned over $C _ { i }$ and $J _ { i } ,$ the behavior of the model is a fixed function of $\mathbf x _ { i + 1 }$ . In particular, the total number of such functions can be bounded. Since each $\mathbf { x } _ { i }$ is drawn uniformly at random, assuming for the time that the model succeed with a good probability at the next stage only if the true key $\mathbf { z } _ { i + 1 }$ falls into a small list of candidates of length l, we will next show that by carefully choosing the size of $\chi _ { k e y } .$ , there must be some Boolean function H such that no matter which function in the class is chosen, there are also at most l possible $\mathbf { z } _ { i }$ that makes $\mathbf { z } _ { i + 1 }$ fall into the list with a good probability. This implies that if we sequentially choose all the $H _ { i }$ based on this, then the model finally will only have a small probability to succeed.

To start with, we first show that for a single step such a function $H _ { i }$ exists. Using a probablistic construction, we show the following lemma.

Lemma 6. Let $\qquad K , \mathcal { X } . \mathcal { Q } ,$ A be finite sets. Denote by $N = | { \cal K } |$ | and $n = | \mathcal { X } |$ . Let $\mathcal { G } : \mathcal { A } \times \mathcal { X }  \mathcal { Q }$ be a finite set ofmaps. For $l \in [ N ] , l e t \mathcal { L } _ { l } = \{ \mathcal { L } : Q \to 2 ^ { \kappa } | | \mathcal { L } ( q ) | \leq i , \forall q \in \mathcal { Q } \} . \mathrm { ~ F o r } \theta \in ( 0 , 1 )$ , if

$$
\binom { N } { l + 1 } | \mathcal { G } | | \mathcal { L } _ { l } | [ | A | 2 ^ { n } ( l / N ) ^ { \theta n } ] ^ { l + 1 } < 1 ,
$$

then there exists a map $H : \mathcal { K } \times \mathcal { X }  \mathcal { K }$ such that for every $G \in { \mathcal { G } }$ and every $\mathcal { L } \in \mathcal { L } _ { l }$ the set

$$
B ( G , { \mathcal { L } } ) : = \{ \mathbf { z } \in K \mid \exists a \in A , { \underset { \mathbf { x } \sim U ( { \boldsymbol { x } } ) } { \operatorname* { P r } } } [ H ( \mathbf { z } , \mathbf { x } ) \in { \mathcal { L } } ( G ( a , \mathbf { x } ) ) ] \geq \theta \}
$$

has cardinality at most l

ProofofLemma 6. Fix $\mathbf { z } \in \mathcal { K } , G \in \mathcal { G } , \mathcal { L } \in \mathcal { L } _ { l } , a \in \mathcal { A }$ . We choose H from all possible maps uniformly at random. Since for each $\mathbf { x } \in { \mathcal { X } } .$ , the list $\mathcal { L } ( G ( a , \mathbf { x } ) )$ has size at most $l ,$ we have

$$
\operatorname* { P r } _ { H } [ H ( \mathbf { z } , \mathbf { x } ) \in \mathcal { L } ( G ( a , \mathbf { x } ) ) ] \leq l / N .
$$

By taking a union bound over $x ,$ we have

$$
\operatorname* { P r } _ { H } [ \{ H : \operatorname* { P r } _ { \mathbf { x } \sim U ( \mathcal { X } ) } [ H ( \mathbf { z } , \mathbf { x } ) \in \mathcal { L } ( G ( a , \mathbf { x } ) ) ] \geq \theta \} ] \leq \sum _ { r \geq \theta n } { \binom { n } { r } } ( l / N ) ^ { r } \leq 2 ^ { n } ( l / N ) ^ { \theta n } .
$$

Now we take a union bound over ${ \mathcal { A } } ,$ for every fixed $\mathbf { z } \in { \mathcal { K } }$ we have

$$
\operatorname* { P r } _ { H } [ \mathbf { z } \in B ( G , \mathcal { L } ) ] \leq | A | 2 ^ { n } ( l / N ) ^ { \theta n } .
$$

Now we consider the probability that a bad event happens over the randomization of the choice of H, which means there exists $l + 1$ possible values in $B ( G , { \mathcal { L } } )$ . Fix a list of $l + 1$ value $\mathbf { z } ^ { ( 1 ) } , \ldots , \mathbf { z } ^ { ( l + 1 ) }$ , since the events $\mathbf { z } ^ { ( i ) } \in B ( G , \mathcal { L } )$ only depend on the function $\dot { H } ( \mathbf { z } ^ { ( i ) } , \cdot )$ and are independent, we know that

$$
\operatorname* { P r } _ { H } [ \mathbf { z } ^ { ( 1 ) } , \ldots , \mathbf { z } ^ { ( l + 1 ) } \in B ( G , { \mathcal { L } } ) ] \leq ( | A | 2 ^ { n } ( l / N ) ^ { \theta n } ) ^ { l + 1 } .
$$

We now take the union bound of $G , { \mathcal { L } }$ and all possible tuples $\mathbf { z } ^ { ( 1 ) } , \ldots , \mathbf { z } ^ { ( l + 1 ) }$ , the assumption of Lemma 6 implies that the probability that we choose an H that creates some $B ( G , { \mathcal { L } } )$ with size more than l is strictly less than 1. This implies the existence of H that satisfies the statement of Lemma 6.

We next fit Lemma 6 into our framework using Lemma 4 and Lemma 5.

Lemma 7. For $\epsilon \in ( 0 , 1 )$ , let $r _ { \epsilon } = \lfloor ( 1 / \epsilon ) \rfloor + 3 , N _ { m } = m ^ { 2 r _ { \epsilon } } , l _ { m } = m ^ { r _ { \epsilon } }$ and let ${ \mathcal { X } } _ { k e y } = [ N _ { m } ] .$ . For all sufficiently large m, there exist functions $H _ { 1 } , \dots , H _ { k } : { \mathcal { X } } _ { k e y } \times { \mathcal { C } } ^ { m }  { \mathcal { X } } _ { k e y }$ such that for every $M \in$ $\mathcal { M } _ { \bar { \alpha } } ( D B , P , D ( W - 1 ) )$ , an arbitrary hybrid model that satisfies $B , P < c _ { \tau } m$ and $W < c _ { \tau } m ^ { 1 / \epsilon } , c _ { \tau } \ll 1$ a constant, the follows hold. Fix a level $0 \leq i < k$ and $c = C _ { i } ,$ , for every $\mathcal { L } : \mathrm { I m } ( C _ { i + 1 } )  2 ^ { \chi _ { k e } }$ <sup>y</sup> such that $| \mathcal { L } ( c ^ { \prime } ) | \leq l _ { m } f o r a l l c ^ { \prime } \in \mathrm { I m } ( C _ { i + 1 } )$ , the set $\{ z \in \mathcal { X } _ { k e y } \mid \exists \gamma \in \{ 0 , 1 \} ^ { D B } , \operatorname* { P r } _ { \mathbf { x } _ { i + 1 } \sim \mathcal { C } ^ { m } } [ H _ { i + 1 } ( \mathbf { z } , \mathbf { x } _ { i + 1 } ) \in$ $\mathcal { L } ( \Gamma _ { i , c , \gamma } ( \mathbf { x } _ { i + 1 } ) ) ] \geq 1 / ( 4 k ) \}$ has size at most $l _ { m } .$

ProofofLemma 7. We apply Lemma 6 to prove Lemma 7. Let $\mathcal { K } : = \mathcal { X } _ { k e y } , N = N _ { m }$ and $\mathcal { X } = \mathcal { C } ^ { m } , n =$ $| \mathcal { C } | ^ { m } , | \mathcal { C } | \ge 2$ . Let $\mathcal { A } = \mathbf { \bar { \{ 0 , 1 \} } } ^ { D B }$ be the state space of $\mathrm { S } _ { i + 1 }$ and $\mathcal { Q } = \mathrm { I m } ( C _ { i + 1 } )$ Define the collection of map $\mathcal { G } = \{ G _ { i , c } ( \gamma , \mathbf { x } _ { i + 1 } ) = \Gamma _ { i , c , \gamma } ( \mathbf { x } _ { i + 1 } ) \mid c \in \operatorname { I m } ( C _ { i } ) \} . \ l = l _ { m }$ and $\theta = 1 / ( 4 k )$ . We show that these choices of parameters satisfy the assumption of Lemma 6.

We first bound the size of G. By Lemma 5, we know that $\Gamma _ { i , c , \gamma } = \big ( H _ { i , c } ( \mathbf { x } _ { i + 1 } ) , \Phi _ { \gamma } ( E _ { i , c } ( \mathbf { x } _ { i + 1 } ) ) \big )$ . We count all possible $H _ { i , c } , E _ { i , c } , \Phi _ { \gamma }$ . Since the total number of output bits of $H _ { i , c }$ is at most $i D B$ , we know that there are at most $2 ^ { i D B n }$ possible $H _ { i , c } .$ . Similarly, denote by $s _ { i + 1 } : = b _ { i + 1 } - u _ { i + 1 } + 1 = ( i + 1 ) \Delta _ { m } + m$ then the total number of output bits of $E _ { i , c }$ is $s _ { i + 1 } P$ and thus there are at most $2 ^ { s _ { i + 1 } P n }$ many $E _ { i , c } .$ . Finally, Φ is determined by the state transition function $\delta : \{ 0 , 1 \} ^ { D B } \times \{ 0 , 1 \} ^ { P } \to \times \{ 0 , 1 \} ^ { D B }$ and has at most $2 ^ { D B 2 ^ { D B + P } }$ many possibilities. So the total number of Φ is at most $\stackrel { \cdot } { 2 } s _ { i + 1 } \stackrel { \cdot } { D } B 2 ^ { D B + P }$ This gives

$$
\begin{array} { r l } & { \log ( | \mathcal { G } | ) \leq n ( i D B + s _ { i + 1 } P ) + s _ { i + 1 } D B 2 ^ { D B + P } } \\ & { \qquad \leq | \mathcal { C } | ^ { m } ( k m ^ { 2 } + m ^ { 1 / \epsilon + 1 } ) + k 2 ^ { 3 c _ { \tau } m } m ^ { 1 / \epsilon + 2 } } \\ & { \qquad = o ( r _ { \epsilon } \log m ( m ^ { r _ { \epsilon } } + 1 ) | \mathcal { C } | ^ { m } / ( 4 k ) ) . } \end{array}
$$

Next, we upper bound the total number of list functions. For any fix $q \in \mathcal { Q }$ , the number of subsets of $| \kappa |$ with size at most l is $\begin{array} { r } { \sum _ { j = 0 } ^ { l } \binom { | \mathcal { K } | } { j } \leq ( l + 1 ) ( e N / l ) ^ { l } } \end{array}$ . This gives the total number list functions is at most $( ( l _ { m } + 1 ) ( e N _ { m } / l _ { m } ) ^ { l _ { m } } ) ^ { | \mathcal { Q } | }$ . In particular, by taking the logorim, we have

$$
\log ( ( ( l _ { m } + 1 ) ( c N _ { m } / l _ { m } ) ^ { l _ { m } } ) ^ { | \mathcal { Q } | } ) = | \mathcal { Q } | \log ( l _ { m } + 1 ) + l _ { m } | \mathcal { Q } | \log ( e N _ { m } / l _ { m } ) = o ( r _ { c } \log m ( m ^ { r _ { c } } + 1 ) | \mathcal { C } | ^ { m } / ( 4 k ) ) .
$$

Finally, we verify the assumption of Lemma 6. By taking the logarithm, the left hand side is.

$$
\log ( | \mathcal { G } | ) + \log ( | \mathcal { L } _ { l _ { m } } | ) + \log ( { \binom { N _ { m } } { l _ { m } + 1 } } ) + ( l _ { m } + 1 ) ( \log ( | \mathcal { A } | ) + n + \theta n \log ( l _ { m } / N _ { m } ) ) .
$$

We notice that the negative term is

$$
( l _ { m } + 1 ) { \theta } n \log ( l _ { m } / N _ { m } ) = - r _ { \epsilon } \log m ( m ^ { r _ { \epsilon } } + 1 ) | \mathcal { C } | ^ { m } / ( 4 k ) ,
$$

which dominates the other terms above. Thus, the assumption of Lemma 6 is satisfied.

Now, in the remainder of the proof, we will translate Lemma 7 to the error analysis of a fixed model.

Lemma 8. For $\epsilon \in ( 0 , 1 )$ , let $r _ { \epsilon } = \lfloor ( 1 / \epsilon ) \rfloor + 3 , N _ { m } = m ^ { 2 r _ { \epsilon } } , l _ { m } = m ^ { r _ { \epsilon } }$ and let ${ \mathcal { X } } _ { k e y } = [ N _ { m } ] .$ . Let $H _ { 1 } , \ldots , H _ { k }$ be functions chosen from Lemma $7 .$ Let $M \in \mathcal { M } _ { \bar { \alpha } } ( D B , P , D ( W - 1 ) )$ , an arbitrary hybrid model that satisfies $B , P < c _ { \tau } m$ and $W < c _ { \tau } m ^ { 1 / \epsilon } , c _ { \tau } \ll 1$ a constant. For $i \in [ k ]$ , denote by $J _ { i }$ the prefix $\left( \mathbf { z } _ { 0 } , \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { i } \right)$ and err $_ i ( J _ { i } ) : = \operatorname* { P r } _ { I \sim \mu _ { m } } [ M ( I ) \neq \mathbf { z } _ { k } \ | \ J _ { i } ]$ . For $i \in [ k - 1 ]$ , and $c \in { \mathrm { I m } } ( C _ { i } )$ , there exists sets $L _ { i } ( c ) \subseteq \mathrm { X } _ { k e y }$ with size $l _ { m }$ such thatfor every $J _ { i } ,$ , err $_ i ( J _ { i } ) \leq \delta _ { i + 1 } : = 1 / 3 + i / ( 6 k )$ implies that $\mathbf { z } _ { i } \in L _ { i } ( C _ { i } ( J _ { i } ) )$

ProofofLemma 8. We prove the lemma using a backward induction on i. The base case is $i = k - 1$ Notice that if $C _ { k }$ is given, then the output of M is a deterministic function of $C _ { k }$ . We define the list function $\mathcal { L } ( C _ { k } )$ to be a set that only contains the output of $M ,$ given $C _ { k }$ . By applying Lemma 7 to ${ \mathcal { L } } ,$ , we know that $L _ { k - 1 } ( c ) : = \{ z \in \mathcal { X } _ { k e y } \mid \exists \gamma \in \{ 0 , 1 \} ^ { D B } , \operatorname* { P r } _ { \mathbf { x } _ { k } \sim \mathcal { C } ^ { m } } [ H _ { k } ( \mathbf { z } , \mathbf { x } _ { k } ) \in \mathcal { L } ( \Gamma _ { k - 1 , c , \gamma } ( \mathbf { x } _ { k } ) ) ] \geq 1 / ( 3 k ) \}$ has a size at most $l _ { m } .$ Now fix any $J _ { k - 1 }$ such that $\mathrm { e r r } _ { k - 1 } ( J _ { k - 1 } ) \leq 1 / 3 + ( k - 1 ) / ( 6 k )$ . By Lemma 4, we know that there there exists some γ such that $C _ { k } = \Gamma _ { k - 1 , c , \gamma } ( \mathbf { x } _ { k } )$ , where $C _ { k - 1 } ( J _ { k - 1 } ) = c .$ . The assumption that $\mathrm { e r r } _ { k - 1 } ( J _ { k - 1 } ) \leq 1 / 3 + ( k - 1 ) / ( 6 k )$ implies that $\operatorname* { P r } _ { \mathbf { x } _ { k } \sim \mathcal { C } ^ { m } } [ H _ { k } ( \mathbf { z } _ { k - 1 } , \mathbf { x } _ { k } ) \in \mathcal { L } ( \Gamma _ { k - 1 , c , \gamma } ( \mathbf { x } _ { k } ) ) ] \geq 1 / 3 k$ which implies that $\mathbf { z } _ { k - 1 } \in L _ { k - 1 } ( C _ { k - 1 } )$

Now suppose that we have proven this for i, we prove this inductively for i−1. For every $c \in \mathrm { I m } ( C _ { i - 1 } )$ We define $L _ { i - 1 } ( c ) = \left\{ z \in \mathcal { X } _ { k e y } \mid \exists \gamma \in \{ 0 , 1 \} ^ { D B } , \operatorname* { P r } _ { \mathbf { x } _ { k } \sim \mathcal { C } ^ { m } } [ H _ { i } ( \mathbf { z } , \mathbf { x } _ { i } ) \in L _ { i } ( \Gamma _ { i - 1 , c , \gamma } ( \mathbf { x } _ { i } ) ) ] \geq 1 / ( 3 k ) \right\}$ . By Lemma 7, we know that $| L _ { i - 1 } ( c ) | \le l _ { m }$ . Now, let $J _ { i - 1 }$ such that er $\mathrm { r } _ { i - 1 } ( J _ { i - 1 } ) \leq \delta _ { i }$ . Since err $\begin{array} { r l } { \ i - 1 \left( J _ { i - 1 } \right) = } \end{array}$ $\mathbb { E } _ { \mathbf { x } _ { i } } [ \mathrm { e r r } _ { i } ( J _ { i - 1 } , \mathbf { x } _ { i } ) ]$ , by Markov’s inequality we know that $\operatorname* { P r } _ { \mathbf { x } _ { i } } [ \operatorname { e r r } _ { i } ( J _ { i - 1 } , \mathbf { x } _ { i } ) > \delta _ { i + 1 } ] \leq \delta _ { i } / \delta _ { i + 1 }$ and thus $\operatorname* { P r } _ { \mathbf { x } _ { i } } [ \operatorname { e r r } _ { i } ( J _ { i - 1 } , \mathbf { x } _ { i } ) \leq \delta _ { i + 1 } ] \geq 1 - \delta _ { i } / \delta _ { i + 1 }$ . By induction when er $\dot { \mathbf { \zeta } } _ { i } ( J _ { i - 1 } , \mathbf { x } _ { i } ) \leq \delta _ { i + 1 }$ , we have $\mathbf { z } _ { i } \in L _ { i } ( C _ { i } )$ which means

$$
\operatorname* { P r } _ { \mathbf { x } _ { i } } [ H _ { i } ( \mathbf { z } _ { i - 1 } , \mathbf { x } _ { i } ) \in L _ { i } ( \Gamma _ { i - 1 , c , \gamma } ( \mathbf { x } _ { i } ) ) ] \geq 1 / ( 3 k ) .
$$

Thus, $\mathbf { z } _ { i - 1 } \in L _ { i - 1 } ( c )$

We now give the full proof of Theorem 3.

ProofofTheorem 3. By Lemma 3, it suffices to consider a model $M ^ { \prime } \in \mathcal { M } _ { \bar { \alpha } } ( D B , P , D ( W - 1 ) )$ . Suppose that M<sup>′</sup> has error at most $1 / 3$ over $\mu _ { m }$ . Then, we have $\mathbb { E } _ { J _ { 1 } } [ \mathrm { e r r } _ { 1 } ( J _ { 1 } ) ] \leq 1 / 3$ . Since $\delta _ { 2 } = 1 / 3 + 1 / ( 6 k )$ by Markov’s inequality, we know that $\Pr _ { J _ { 1 } } [ e _ { 1 } ( J _ { 1 } ) \leq \delta _ { 2 } ] \geq 1 / ( 2 k + 1 )$ . By Lemma $^ { 8 , }$ we know that $z _ { 1 } \in L _ { 1 } ( C _ { 1 } )$ . In particular, this implies that $\mathrm { P r } [ \mathbf { z } _ { 1 } \in L _ { 1 } ( C _ { 1 } ) ] \geq 1 / ( 2 k + 1 )$ . However, by Lemma $^ { 6 , }$ we will show that the choice of $H _ { 1 }$ must also have $\mathrm { P r } [ \mathbf { z } _ { 1 } \in L _ { 1 } ( C _ { 1 } ) ] < 1 / ( 2 k + 1 )$ . To see this, recall that the set of $\mathbf { z } _ { 0 } \in \mathcal { X } _ { k e y }$ such that there exists some internal state γ such that the probability that $H _ { 1 } ( \mathbf { z } _ { 0 } , \mathbf { x } _ { 1 } ) \in$ $L _ { 1 } ( \Gamma _ { \gamma } ( \mathbf { x } _ { 1 } ) )$ is more than $1 / 4 k$ has a size at most $l _ { m }$ . Denote by this set of keys to be $B ,$ then we have

$$
\operatorname* { P r } _ { \mathbf { z } _ { 0 } , \mathbf { x } _ { 1 } } [ \mathbf { z } _ { 1 } \in L _ { 1 } ( C _ { 1 } ) ] \le \operatorname* { P r } _ { \mathbf { z } _ { 0 } \in B } + ( 1 / 4 k ) \le l _ { m } / N _ { m } + ( 1 / 4 k ) < 1 / ( 2 k + 1 ) ,
$$

where m is large enough.

## E.2 Correct ordering but insufficient layers

Theorem 4. Let $D \in \mathbf { N } _ { + }$ and $\tau = X _ { 1 } ^ { r _ { 1 } } X _ { 2 } ^ { r _ { 2 } } \cdot \cdot \cdot X _ { h } ^ { r _ { h } } \in \Sigma ^ { D }$ be an arbitrary architecture profile with depth D. Let $\alpha : = \mathrm { c o m p } ( \tau )$ There exists a constant $\lambda _ { \tau } > 1 , C _ { \tau } > 0 .$ , a family of task $\{ G _ { \tau , m } \} _ { m \geq 1 }$ and a family of distributions $\nu _ { m }$ over a sequence of tokens with length $L _ { m }$ parameterized by $m \in \mathbf { N } _ { + } ,$ , such that for all m sufficiently large, the follows hold. Let $A \in \Sigma ^ { \bar { \leq } D }$ be an architecture profile with $\alpha \preceq A$ but $\tau \not \simeq A .$ . Then for every hybrid model $M \in \mathcal { M } _ { A } ( B , P , W )$ that satisfies er $\cdot ( M , G _ { \tau , m } ) \leq O ( 1 / D )$ must satisfy $W \ge \lambda _ { \tau } ( m + 1 )$ or max $\{ B , P \} \ \ge \ \lambda _ { \tau } C _ { \tau }$ log m. On the other hand, there exists a model $M ^ { \prime } \in { \mathcal { M } } _ { \mathrm { \Gamma } }$ <sub>τ</sub>(C<sub>τ</sub> log m, C<sub>τ</sub> log $m , m + 1 )$ that can solve $G _ { \tau , m }$ exactly.

The main proof of Theorem 4 will focus on the hardness part. We introduce the following definitions

Definition 6. Let $\tau \in \Sigma ^ { D }$ . Let $s ( \tau ) = | \{ i \in [ D ] \mid \tau _ { i } = \mathrm { S } \} |$ and $t ( \tau ) = | \{ i \in [ D ] \ | \ \tau _ { i } = \mathrm { T } \}$ |. For $i \in [ s ( \tau ) ]$ we use $r _ { i } ( \tau )$ to denote the number of T-layers in τ that is strictly after the ith S-layer and for $i \in [ t ( \tau ) ]$ we use $l _ { i } ( \tau )$ to denote the number of S-layers in τ that is strictly after the ith T-layer. We call the set of architecture profiles

$$
\mathcal { C } _ { 2 } ( \tau ) = \{ \mathrm { S } ^ { s ( \tau ) } \} \cup \{ \mathrm { T } ^ { t ( \tau ) } \} \cup \{ \mathrm { S } ^ { i } \mathrm { T } ^ { r _ { i } ( \tau ) } ~ | ~ i \in [ s ( \tau ) ] \} \cup \{ \mathrm { T } ^ { i } \mathrm { S } ^ { l _ { i } ( \tau ) } ~ | ~ i \in [ t ( \tau ) ] \} ,
$$

the size-2 cover of $\tau .$

We first give the following property of the size-2 cover.

Lemma 9. Let $D \in \mathbf { N } _ { + }$ and $\tau \in \Sigma ^ { D }$ . Let $A \in \Sigma ^ { \leq D }$ such that $\tau \not \simeq A ,$ then there exists some $\sigma \in \mathcal { C } _ { 2 } ( \tau )$ such that $\sigma \not \simeq A .$

Proof of Lemma 9. Suppose that instead for every $\sigma \in \mathcal { C } _ { 2 } ( \tau )$ , we have $\sigma \preceq A$ . In particular, we have $\mathbf { S } ^ { s ( \tau ) } \preceq A$ and $\mathbb { T } ^ { t ( \tau ) } \preceq A$ . This implies that it must be the case that $| A | = D$ and $s ( A ) = s ( \tau ) , t ( A ) = t ( \tau )$ We notice that once $s ( A ) , t ( A )$ and $r _ { 1 } ( A ) , \ldots , r _ { s ( A ) } ( A )$ is fixed, A is completely determined. Since for $i \in { s ( \tau ) , \mathbb { S } ^ { i } \mathrm { T } ^ { r _ { i } ( \tau ) } \preceq A }$ , we know that $r _ { i } ( A ) \geq r _ { i } ( \tau )$ . Notice that the total number of $( i , j )$ such that $i < j$ and $A _ { i } = \mathrm { T } , A _ { j } = \mathrm { S }$ is exactly

$$
\sum _ { i = 1 } ^ { s ( A ) } ( t ( A ) - r _ { i } ( A ) ) = \sum _ { i = 1 } ^ { t ( A ) } l _ { i } ( A ) .
$$

A $\neq \tau$ implies that at least one of $r _ { i } ( A ) > r _ { i } ( \tau )$ , which implies that $\begin{array} { r } { \sum _ { i = 1 } ^ { t ( A ) } l _ { i } ( A ) < \sum _ { i = 1 } ^ { t ( A ) } l _ { i } ( \tau ) } \end{array}$ and thus some $l _ { i } ( \tau ) > l _ { i } ( A )$ . However, this implies that $\dot { \mathrm { T } } ^ { i } S ^ { l _ { i } ( \tau ) } \not \cong A$ □

Given Lemma $^ { 9 , }$ we next provide a reduction showing that it is enough to construct a hard task for every architecture profile in the size-2 cover.

Lemma 10. Let $D \in \mathbf { N } _ { + }$ and $\tau = X _ { 1 } ^ { r _ { 1 } } X _ { 2 } ^ { r _ { 2 } } \cdot \cdot \cdot X _ { h } ^ { r _ { h } } \in \Sigma ^ { D }$ be an arbitrary architecture profile with depth D. Let C be a set of R architecture profiles such that for $\begin{array} { r } { \sigma _ { j } \in \mathcal { C } , j \in [ R ] , \sigma _ { j } \preceq \tau . } \end{array}$ . Furthermore, suppose there exists constant $C > 0$ and $\lambda > 1$ such that for $j \in [ R ]$ , there exists a family of task $\{ H _ { j , m } \}$ there exists a model $M ^ { \prime } \in \mathcal { M } _ { \sigma _ { i } } ( C$ log m, C log m, , m + 1) that can solve $H _ { j , m }$ exactly, while every $h y -$ brid model $M \in \mathcal { M } _ { A } ( B , P , \bar { W } )$ that satisfies err $( M , H _ { j , m } ) ~ \leq ~ 1 / 3$ , must satisfy $W \ge \lambda ( m + 1 )$ or max $\{ B , P \} \ge \lambda C$ log m, $i f A \in \Sigma ^ { \leq D } , \sigma _ { j } \not \geq A .$ . Then, there exists a family of task $\{ G _ { \tau , m } \}$ such that for every hybrid model $M \in { \mathcal { M } } _ { A } ( B , P ,$ W) that satisfies err $( M , G _ { \tau , m } ) \leq 1 / ( 3 R )$ , must satisfy $W \geq \lambda ( m + 1 )$ or max $\{ B , P \} \ge \lambda C$ log m, $i f A \in \Sigma ^ { \leq D }$ and $\exists \sigma \in { \mathcal { C } } , \sigma \not \cong A .$ On the other hand, there exists a model $M ^ { \prime } \in \mathcal { M } _ { \tau } ( C$ log m, C log $m , m + 1 )$ that can solve $G _ { \tau , m }$ exactly.

ProofofLemma 10. We start with the design of the task $G _ { \tau , m }$ . Let $\mathcal { X }$ be the context space and let $H _ { j , m } :$ ${ \mathcal { X } } ^ { L _ { m } } \to { \mathcal { X } }$ , where $L _ { m }$ is the length of the input sequence. Let $\nu _ { j , m }$ be the distribution of the input sequence for task $H _ { j , m }$ . We define the distribution $\nu _ { m }$ over $( [ R ] \times \mathcal { X } ) ^ { L _ { m } }$ as follows. First, $, j \sim [ R ]$ is drawn uniformly. Given $j ,$ a sequence $\mathbf { x } \sim \nu _ { j , m }$ is drawn and finally the returned sample has $\bar { \mathbf { x } } \in ( [ R ] \times \mathcal { X } ) ^ { L _ { m } }$ has a form $\bar { \bf x } _ { i } = ( j , { \bf x } _ { i } )$ , where $i \in [ L _ { m } ]$ . And the task $G _ { \tau , m }$ is asked to compute $H _ { j , m } ( \mathbf { x } )$ , given x¯. The hardness part follows immediately. Since The hardness part follows immediately. Since $M _ { A } \in \mathcal { M } ( B , P , W )$ and some and some $\sigma _ { j } \not \subset A$ , which implies , which implies that that $M _ { A }$ fails to solve fails to solve $H _ { j , m }$ over over $\nu _ { j , m }$ with probability with probability $1 / 3 .$ , this implies that condition , this implies that condition $j \sim [ R ]$ ] is sampled, is sampled, there is at least $1 / R$ probability that $H _ { j , m }$ has to be solved and $M _ { A }$ has a failuare probability $1 / ( 3 R )$

We now design the model M that can be used to solve the task $G _ { \tau , m }$ . Since the layers we are considering are generalized linear models and non-linear causal window functions, we will just use a selector between layers to select the correct model to use. Specifially, let $M _ { j , m } = N _ { j , m } ^ { | \sigma _ { j } | } \circ \cdot \cdot \cdot \circ N _ { j , m } ^ { 1 }$ , where $N _ { j , m } ^ { i }$ is the ith layer of $M _ { j , m } ,$ which can solve $H _ { j , m }$ exactly. Importantly, since $\sigma _ { j } \preceq \tau ,$ , we know that there exists a strictly increasing map $P _ { j } : [ | \sigma _ { j } | ]  [ D ]$ , such that $\tau _ { P _ { j } ( i ) } = ( \sigma _ { j } ) _ { \mathrm { ~ } }$ <sub>i</sub>, for $i \in [ | \sigma _ { j } | ]$ . We define the model M as follows for the $i ^ { * } { \mathfrak { t h } }$ layer of $M .$ , it contains identical map and all types of layers $N _ { j , m } ^ { i } \mathrm { i f } P _ { j } ( i ) = i ^ { * }$ as blocks and it will select which block to use when read an input $\left( \boldsymbol { j } , \mathbf { x } _ { t } \right)$ . Specifially, when it reads $\left( { j , { \bf { x } } _ { t } } \right)$ , it will choose to use $N _ { j , m } ^ { P _ { j } ( i ) } \mathrm { i f } P _ { j } ( i ) = i ^ { * }$ and identical map otherwise. By the property of $M _ { j , m }$ , the model M has 0 error over $\nu _ { m } .$ Importantly, the structure of M is defined independent on j and only makes uses one of $M _ { j , m }$ on a single input sequence and thus is in $\mathcal { M } _ { \tau } ( C \log m , C \log m , m + 1 )$ □

Based on Lemma 10, it suffices to construct hard tasks for target profiles $\sigma \in \mathcal { C } _ { 2 } ( \tau )$ . In the rest of the proof, we establish Lemma 11.

Lemma 11. Let $D \in \mathbf { N } _ { + }$ and $\tau = X _ { 1 } ^ { r _ { 1 } } X _ { 2 } ^ { r _ { 2 } } \cdot \cdot \cdot X _ { h } ^ { r _ { h } } \in \Sigma ^ { D }$ be an arbitrary architecture profile with depth D. There exists constant $C > 0$ and $\lambda > 1$ such that for $j \in [ R ]$ , there exists a family of task $\{ H _ { \sigma , m } \}$ $\sigma \in \mathcal { C } _ { 2 } ( \tau )$ , such that there exists a model $M ^ { \prime } \in \mathcal { M } _ { \sigma } ( b _ { m } , b _ { m } , m + 1 )$ that can solve $H _ { \sigma , m }$ exactly, where $b _ { m } = C$ log m, while every hybrid model $M \in \mathcal { M } _ { A } ( B , P , W )$ that satisfies err(M, $H _ { \sigma , m } ) \leq 1 / 3 ,$ , for some input distribution $\nu _ { \sigma , m }$ must satisfy $W \geq \lambda ( m + 1 )$ or max $\{ B , P \} \ge \lambda b _ { m } , i f A \in \Sigma ^ { \le D } , \sigma \ne A$

Before presenting the proof, we first state some technical lemmas that will be used for the analysis.

Lemma 12. Let x, w be two independent vector sampled from $\{ 0 , 1 \} ^ { a }$ for some $a \in \mathbf { Z } _ { + }$ . Let $E : \{ 0 , 1 \} ^ { a } \to$ $\{ 0 , 1 \} ^ { a }$ be an embedding function such that |Im $\mathit { ' } E ) | \leq 2 ^ { b }$ , for some $b < a ,$ then for every function $\psi : \cdot$ $\begin{array} { r } { \dot { \{ 0 , 1 \} ^ { a } } \times \{ 0 , 1 \} ^ { a }  \{ 0 , 1 \} , \operatorname* { P r } _ { \mathbf { x } , \mathbf { w } } [ \psi ( E ( \mathbf { x } ) , \mathbf { w } ) = \oplus _ { i = 1 } ^ { n } \mathbf { x } _ { i } \mathbf { w } _ { i } ] \leq ( 1 / 2 ) + 2 ^ { - ( a - b ) / 2 - 1 } } \end{array}$

ProofofLemma 12. Write $Y = \langle \mathbf { x } , \mathbf { w } \rangle _ { 2 } = \oplus _ { i = 1 } ^ { a } \mathbf { x } _ { i } \mathbf { w } _ { i }$ . Let $h \in \operatorname { I m } ( E )$ and denote by $S _ { h } : = \{ { \bf x } \mid E ( { \bf x } ) =$ h}. Let $s _ { h } = | S _ { h } |$ . For each w $\in \{ 0 , 1 \} ^ { a }$ , define $\beta _ { h } ( \mathbf { w } ) : = \mathbb { E } _ { \mathbf { x } \sim U ( S _ { h } ) } \left. ( - 1 ) ^ { \langle \mathbf { x } , \mathbf { w } \rangle _ { 2 } } \right.$ . Notice that $\beta _ { h } ( \mathbf { w } ) =$ $\operatorname* { P r } _ { \mathbf { x } } [ Y = 0 \mid E ( \mathbf { x } ) = h , \mathbf { w } ] - \operatorname* { P r } _ { \mathbf { x } } [ Y = 1 \mid E ( \mathbf { x } ) = h , \mathbf { w } ]$ . This implies that given $h , \mathbf { w }$ , the Bayesian optimal prediction has an error of $( 1 + | \beta _ { h } ( \mathbf { w } ) | ) / 2$

Taking the average over w, we have

$$
\begin{array} { r l } & { \underset { \mathbf { x } , \mathbf { w } } { \operatorname* { P r } } [ \psi ( E ( \mathbf { x } ) , \mathbf { w } ) = \oplus _ { i = 1 } ^ { a } \mathbf { x } _ { i } \mathbf { w } _ { i } \mid E ( \mathbf { x } ) = h ] \leq ( 1 + \mathbb { E } _ { \mathbf { w } } [ \lvert \beta _ { h } ( \mathbf { w } ) \rvert ] ) / 2 } \\ & { } \\ & { \leq ( 1 + \sqrt { \mathbb { E } _ { \mathbf { w } } [ \lvert \beta _ { h } ( \mathbf { w } ) \rvert ^ { 2 } ] } ) / 2 = ( 1 + 1 / \sqrt { s _ { h } } ) / 2 } \end{array}
$$

Now we take the average over h and get

$$
\begin{array} { l l } { \displaystyle \operatorname* { P r } _ { \mathbf { x } , \mathbf { w } } [ \psi ( E ( \mathbf { x } ) , \mathbf { w } ) = \bigoplus _ { i = 1 } ^ { n } \mathbf { x } _ { i } \mathbf { w } _ { i } ] \leq \frac { 1 } { 2 } + \frac { 1 } { 2 ^ { a + 1 } } \sum _ { h } \sqrt { s _ { h } } \leq \frac { 1 } { 2 } + \frac { 1 } { 2 ^ { a + 1 } } \sqrt { | \mathrm { I m } ( E ) | \sum _ { h } s _ { h } } } \\ { \displaystyle \qquad \leq ( 1 / 2 ) + 2 ^ { - ( a - b ) / 2 - 1 } . } \end{array}
$$

Lemma 13. Let $n , K \in \mathbf { Z } _ { + }$ . Let $\mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) }$ be drawn uniform at random and independently from $\{ 0 , 1 \} ^ { n }$ and let J be drawn uniformly from $[ K ]$ . Let Γ be a discrete function over $( \{ 0 , 1 \bar  \} ^ { n } ) ^ { K }$ that takes at most $2 ^ { a }$ values, for some $a \in \mathbf { Z } _ { + }$ Suppose that for every $\gamma \in \mathrm { I m } ( \Gamma ) , j \in [ K ]$ , there exists a set $L _ { \gamma , j } \subseteq \{ 0 , 1 \} ^ { n }$ with size at most $l < 2 ^ { n }$ . Then we have

$$
\operatorname* { P r } _ { \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) } , J } [ \mathbf { x } ^ { ( J ) } \in L _ { \Gamma ( \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) } ) , J } ] \leq \frac { a / K + 1 } { n - \log l } .
$$

Proof of Lemma 13. Notice that

$$
\begin{array} { r l } {  { \sum _ { i = 1 } ^ { K } I ( { \mathbf x } ^ { ( i ) } ; \Gamma ) = \sum _ { i = 1 } ^ { K } H ( { \mathbf x } ^ { ( i ) } ) - \sum _ { i = 1 } ^ { K } H ( { \mathbf x } ^ { ( i ) } \mid \Gamma ) } } \\ & { \leq K n - H ( { \mathbf x } ^ { ( 1 ) } , \ldots , { \mathbf x } ^ { ( K ) } \mid \Gamma ) } \\ & { = I ( { \mathbf x } ^ { ( 1 ) } , \ldots , { \mathbf x } ^ { ( K ) } ; \Gamma ) } \\ & { \leq H ( \Gamma ) \leq a . } \end{array}
$$

This implies that

$$
\begin{array} { r l } & { I ( \mathbf { x } ^ { ( J ) } ; \Gamma , J ) = I ( \mathbf { x } ^ { ( J ) } ; J ) + I ( \mathbf { x } ^ { ( J ) } ; \Gamma \mid J ) } \\ & { \quad \quad \quad = \displaystyle \frac { 1 } { K } \sum _ { i = 1 } ^ { K } I ( \mathbf { x } ^ { ( i ) } ; \Gamma ) } \\ & { \quad \quad \quad \leq a / K . } \end{array}
$$

Denote the joint random variable $R = ( \Gamma , J )$ . We show that $p : = \mathrm { P r } [ \mathbf { x } ^ { ( J ) } \in L _ { R } ] \leq ( I ( \mathbf { x } ^ { ( J ) } ; R ) + 1 ) / ( n - 1$ log l). Define random variable $B = \mathbf { 1 } \{ \mathbf { x } ^ { ( J ) } \in L _ { R } \}$ , we have

$$
H ( \mathbf { x } ^ { ( J ) } , B \mid R ) = H ( \mathbf { x } ^ { ( J ) } \mid R ) + H ( B \mid \mathbf { x } ^ { ( J ) } R ) = H ( \mathbf { x } ^ { ( J ) } \mid R ) .
$$

This gives

$$
\begin{array} { r l } { H ( { \mathbf x } ^ { ( J ) } \mid R ) = H ( { \mathbf x } ^ { ( J ) } , B \mid R ) } & { } \\ & { \ = H ( B \mid R ) + H ( { \mathbf x } ^ { ( J ) } \mid B , R ) } \\ & { \ \le 1 + H ( { \mathbf x } ^ { ( J ) } \mid B , R ) = 1 + p H ( { \mathbf x } ^ { ( J ) } \mid B = 1 , R ) + ( 1 - p ) H ( { \mathbf x } ^ { ( J ) } \mid B = 0 , R ) } \\ & { \ \le 1 + p \log l + ( 1 - p ) n = n + 1 - p ( n - \log l ) . } \end{array}
$$

Thus, we have $I ( \mathbf { x } ^ { ( J ) } ; R ) \ge p ( n - \log l ) - 1$ , which gives $\operatorname* { P r } [ \mathbf { x } ^ { ( J ) } \in L _ { R } ] \leq ( I ( \mathbf { x } ^ { ( J ) } ; R ) + 1 ) / ( n - \log l )$ Since $I ( { \bf x } ^ { ( J ) } ; R ) \le a / K$ , we get

$$
\operatorname* { P r } _ { \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) } , J } [ \mathbf { x } ^ { ( J ) } \in L _ { \Gamma ( \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) } ) , J } ] \leq \frac { a / K + 1 } { n - \log l } .
$$

Lemma 14 (List Decoding for Hadamard Code). Let $f : \{ 0 , 1 \} ^ { n }  \{ 0 , 1 \}$ be an arbitrary Booleanfunction and let $L : = \{ \mathbf { x } \in \{ 0 , 1 \} ^ { n } \mid \operatorname* { P r } _ { \mathbf { u } \sim U ( \{ 0 , 1 \} ^ { n } ) } [ f ( \mathbf { u } ) = \langle \mathbf { x } , \mathbf { u } \rangle _ { 2 } ] > \epsilon + 1 / 2 \} , \epsilon \in ( 0 , 1 )$ . Then, $| L | \leq 1 / ( 4 \epsilon ^ { 2 } )$ ProofofLemma 11. We will break the proof into 4 cases to consider the cases where σ has the form of $\boldsymbol { \mathrm { S } } ^ { r } , \boldsymbol { \mathrm { T } } ^ { r } , \boldsymbol { \mathrm { T } } ^ { r } \boldsymbol { \mathrm { S } } ^ { s } , \boldsymbol { \mathrm { S } } ^ { r } \boldsymbol { \mathrm { T } } ^ { s }$ , with $r + s \le D$ . Let $\lambda = 1 + ( 1 / 4 D )$

Case 1: $\sigma = \mathrm { S } ^ { r }$ In the first case, we consider the case where $\sigma = \mathrm { S } ^ { r }$ . Let $a _ { m } = b _ { m } - 1$ and let $n = r a _ { m }$ We design the input distribution $\nu _ { \mathrm { S } ^ { r } , m }$ as follows. To draw an input I from $\nu _ { \mathrm { S } ^ { r } , m }$ , we first draw $\mathbf { x } , \mathbf { w }$ uniform at random independently from $\{ 0 , 1 \} ^ { n }$ and return $I = ( \mathbf { x } , { \vec { 0 } } , \mathbf { w } )$ , where <sup>⃗</sup>0 is sequence of 0 of length at least $D \lambda ( m + 1 )$ . The task $H _ { \mathrm { S } ^ { r } , m } ( I )$ asks to compute $Y = < \mathbf { x } , \mathbf { w } > _ { 2 } = \oplus _ { i = 1 } ^ { n } \mathbf { x } _ { i } \mathbf { w } _ { i }$ . By Lemma 1, we show that a single layer SSM with $2 ^ { r b _ { m } }$ states can solve this task. In fact, this can be done since $2 ^ { r b _ { m } }$ bits are enough to memorize x, w exactly.

We next show that if $S ^ { r } \not \simeq A ,$ , then a model from $\mathcal { M } _ { A } ( B , P , W )$ has a bad accuracy unless $B , P , W$ are large. Since $S ^ { r } \not \simeq A$ , we know that A contains at most $r - 1 \mathrm { s s M }$ layers. On the other hand, since <sup>⃗</sup>0 has length at least $D \lambda ( m + 1 )$ , we know that if $W < \lambda ( m + 1 )$ , then the T-layers have no direct access to x. This implies that there exist maps ψ and E such that $M ( I ) = \psi ( E ( \mathbf { x } ) , \mathbf { w } )$ , with $\begin{array} { r } { \log ( | \mathrm { I m } ( E ) | ) \leq ( r - 1 ) \lambda b _ { m } . } \end{array}$ Since $n - ( r - 1 ) \lambda b _ { m } \geq \Omega _ { D } ( \log m )$ , Lemma 12, we know that M has a success probability at most $2 / 3$ when m is large enough.

Case 2: $\sigma = \mathrm { T } ^ { r }$ In the second case, we consider the case where $\boldsymbol { \sigma } = \boldsymbol { \mathrm { T } } ^ { r }$ . Let $n = 2 D b _ { m }$ . We design the input distribution $\nu _ { \mathrm { T } ^ { r } , m }$ as follows. To draw an input I from $\nu _ { \mathrm { { T } } ^ { r } , m }$ , we first draw x, w uniform at random independently from $\{ 0 , 1 \} ^ { n }$ and return $I = ( \mathbf { x } , { \vec { 0 } } , \mathbf { w } )$ , where <sup>⃗</sup>0 is sequence of 0 of length $\lambda ( r - 1 ) m + n <$ $\Delta < r m$ (by the choice of λ, such ∆ exists). The task $H _ { \mathrm { T } ^ { r } , m } ( I )$ asks to compute $Y = \langle \mathbf { x } , \mathbf { w } \rangle _ { 2 } = \oplus _ { i = 1 } ^ { n } \mathbf { x } _ { i } \mathbf { w } _ { i }$ Since a T-layer can be any causal nonlinear function with window size $m + 1$ , by Lemma $^ { 2 , }$ we can design some $M \in \mathcal { M } _ { \mathrm { T } ^ { r } } ( b _ { m } , b _ { m } , m + 1 )$ ) such that it first move x across from ⃗0 and computes $Y$

We next show that if $\Sigma ^ { r } \prec A .$ , then a model from $\mathcal { M } _ { A } ( B , P , W )$ has a bad accuracy unless $B , P , W$ are large. Since $\Sigma ^ { r } \prec A .$ , we know that A contains at most $r - 1 \mathrm { T - l a y e r s }$ , by the choice of $\Delta ,$ we know that no after ${ \vec { 0 } } ,$ no T layer can reach x directly. Since there are at most SSM-layers, this implies that there exist maps ψ and E such that $M ( I ) = \psi ( E ( \mathbf { x } ) , \mathbf { w } )$ , with $\log ( | \mathrm { I m } ( E ) | ) \leq D \lambda b _ { m }$ . Since $n - D \lambda b _ { m } \geq \Omega _ { D } ( \log m )$ Lemma 12, we know that M has a success probability at most $2 / 3 .$ , when m is large enough.

Case 3: $\boldsymbol { \sigma } = \boldsymbol { \mathrm { T } } ^ { r } \boldsymbol { \mathrm { S } } ^ { s }$ In the third case, we consider the case where $\boldsymbol { \sigma } = \boldsymbol { \mathrm { T } } ^ { r } \boldsymbol { \mathrm { S } } ^ { s }$ , with $r + s \le D$ . Let $a _ { m } = b _ { m } - 1 , n = s a _ { m }$ , and $K = 3 2 D$ We design the input distribution $\nu _ { \mathrm { T } ^ { r } \mathrm { S } ^ { s } , m }$ as follows. To draw an input I from $\nu _ { \mathrm { T } ^ { r } \mathrm { S } ^ { s } , m }$ , we first draw $\mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) }$ and w uniform at random independently from $\{ 0 , 1 \} ^ { n }$ and draw some J from [K] uniform at random. The sampled sequence I combines $\mathbf { x } ^ { ( \mathrm { 1 } ) } , \dots , \mathbf { \bar { x } } ^ { ( K ) } , J$ and w with two 0 vectors with different lengths. Specifically, we put $\mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) }$ at the first $K n$ positions. We append $\mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) }$ with a sequence of 0 vectors with length $\Delta _ { 1 }$ that satisfies $\Delta _ { 1 } + 1 > ( r - 1 ) ( \lfloor \lambda ( m + 1 ) \rfloor - 1 )$ and $K n + \Delta _ { 1 } + n - 1 \leq r m$

We explain the choice of such $\Delta _ { 1 }$ before proceeding. Notice that the position of j is $q = K n + \Delta _ { 1 } + 1$ and if $W < \lambda ( m + 1 )$ then at the position $q , \tt a r - 1 \mathrm { - l a y e r T }$ can read input at most $( r - 1 ) ( \lfloor \lambda ( m + 1 ) \rfloor - 1 )$ from q. $\Delta _ { 1 } + 1 > ( r - 1 ) ( | \lambda ( m + 1 ) | - 1 )$ implies that starting from position q no information from $\mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) }$ is known by $\scriptstyle \mathbf { T } ^ { r - 1 }$ . On the other hand, a $\scriptstyle \mathbf { T } ^ { r - 1 }$ is able to copy $\mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) }$ to position $[ 1 + ( r - 1 ) m , K n + ( r - 1 ) m ]$ We would like to require a single layer of T with $m + 1$ window size to see $1 + ( r - 1 { \dot { \bf { \tau } } }$ )m at position $q + n - 1$ , which implies that $K n + \Delta _ { 1 } + n - m \leq 1 + ( r - 1 ) m$ . i.e. $K n + \Delta _ { 1 } + n - 1 \leq r m$ . Notice that $r m - ( r - 1 ) \lambda m = \Omega ( m )$ . So, such a choice of $\Delta _ { 1 }$ exists.

Then, after adding J to position $K n + \Delta _ { 1 } + 1$ , we append another <sup>⃗</sup>0 with size $D ( \lfloor \lambda ( m + 1 ) \rfloor - 1 )$ and append w after this <sup>⃗</sup>0.

The existence of a model $M \in \mathcal { M } _ { \mathrm { T } ^ { r } \mathrm { S } ^ { s } } ( b _ { m } , b _ { m } , m + 1 )$ that can solve the task exactly follows from the construction of I. The first $r - 1$ layer T can move $\mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) } \tan \left[ 1 + ( r - 1 ) m , K n + ( r - 1 ) m \right]$ so that another T layer has an exact access to both $\mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) }$ and $^ { J , }$ so can write $\overrightharpoon { x } ^ { ( J ) }$ after position q. Then, an s-layer SSM with space size $2 ^ { b _ { m } }$ can memorize $x ^ { ( J ) }$ and compute $\langle \mathbf { x } ^ { ( J ) } , \mathbf { w } \rangle _ { 2 }$

It remains to show the lower bound part. Let $M \in \mathcal { M } _ { A } ( B , P , W )$ be a model such that $ { \mathbb { T } } ^ { r } S ^ { s } \not \simeq A$ . To simplify the terminology, for an S-layer that appears after r T-layers, we call it a later S layer, otherwise, we call it an early S-layer. Denote by $\mathrm { \Phi } _ { \mathrm { S } } ( { \bf { \bar { e } } } )$ the set of early S-layers. Let E be the set of states of S layers in $\mathrm { S } ^ { ( e ) }$ evaluated before position q i.e. before $j$ is read. Similarly, denote by $\boldsymbol { \mathrm { S } } ^ { ( l ) }$ the set of late S-layers and let $Z$ be the set of states of S layers in $\mathrm { S } ^ { ( l ) }$ evaluated before w is read.

By the design of $\Delta _ { 1 }$ , we know that E is a deterministic function of $\mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) }$ , while conditioned on $E ,$ starting from position $q ,$ the states of layers in $\mathrm { S } ^ { ( e ) }$ is a deterministic function of $( j , \thinspace \mathbf { w } )$ . In particular, we know that $E$ can take at most $2 ^ { D \lambda b _ { m } }$ values. Similarly, we know that conditioning on $Z ,$ the states of layers in $\mathrm { S } ^ { ( l ) }$ after reading w is a deterministic function of w. In particular, $Z$ can take at most $2 ^ { ( s - 1 ) \lambda b _ { m } }$ values, otherwise, $\mathbb { T } ^ { r } S ^ { s } \preceq A$

Together, this implies that the final output $M = \psi ( E , Z , j , \mathbf { w } )$ for some deterministic function $\psi .$ . Now for $\gamma \in \operatorname { I m } ( E ) , j \in [ K ]$ and $\mathbf { z } \in \operatorname { I m } ( Z )$ , we denote by

$$
L _ { \gamma , j , \mathbf { z } } = \{ \mathbf { u } \in \{ 0 , 1 \} ^ { n } \mid \operatorname* { P r } _ { \mathbf { w } } [ \psi ( \gamma , \mathbf { z } , j , \mathbf { w } ) = \langle \mathbf { u } , \mathbf { w } \rangle _ { 2 } ] \geq ( 7 / 1 2 ) \} .
$$

By Lemma 14, we know that $| L _ { \gamma , j , \mathbf { z } } | \leq 3 6$ . This implies that if we denote by $L _ { \gamma , j } = \cup _ { \mathbf { z } \in \mathrm { I m } ( Z ) }$ , then we have $| L _ { \gamma , j } | \leq 3 6 2 ^ { ( s - 1 ) \lambda b _ { m } }$ . We now apply Lemma 13 to $L _ { \gamma , j }$ and $\mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) }$ and get $\operatorname* { P r } _ { \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( k ) } , J } \big [ \mathbf { x } ^ { ( J ) } \in$ $L _ { \Gamma ( \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( k ) } ) , J } ] < 1 / 5$ . We conclude of the proof of the case by analyzing the probability of failure of $M .$ Let p be the random variable of $\operatorname* { P r } _ { \mathbf { w } } [ M ( I ) = \langle \mathbf { x } ^ { ( J ) } , \mathbf { w } \rangle \mid \mathbf { x } ^ { ( 1 ) } , \dots , \mathbf { x } ^ { ( K ) } , J ]$ . Assume by contradiction that $\mathbb { E } [ p ] \ge 2 / 3$ , then we have

$$
\begin{array} { r l } & { 2 / 3 \le \mathbb { E } [ p ] = \mathbb { E } [ p \mid p < 7 / 1 2 ] \operatorname* { P r } _ { \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) } , J } [ p < 7 / 1 2 ] + \mathbb { E } [ p \mid p \ge 7 / 1 2 ] \operatorname* { P r } _ { \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) } , J } [ p \ge 7 / 1 2 ] } \\ & { \qquad \le 7 \operatorname* { P r } _ { \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) } , J } [ p < 7 / 1 2 ] / 1 2 + \operatorname* { P r } _ { \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) } , J } [ p \ge 7 / 1 2 ] } \\ & { \qquad \le 7 / 1 2 + \operatorname* { P r } _ { \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) } , J } [ p \ge 7 / 1 2 ] . } \end{array}
$$

This implies that $\operatorname* { P r } _ { \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) } , J } [ p \geq 7 / 1 2 ] \geq 1 / 5$ . But $\operatorname* { P r } _ { \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( K ) } , J } [ p \geq 7 / 1 2 ] \leq \operatorname* { P r } _ { \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( k ) } , J } [ \mathbf { x } ^ { ( J ) } \in$ $L _ { \Gamma ( \mathbf { x } ^ { ( 1 ) } , \ldots , \mathbf { x } ^ { ( k ) } ) , J } ] < 1 / 5$ , which gives a contradiction. Thus, M must have a failure probability $1 / 3 .$

Case 4: $\boldsymbol { \sigma } = \boldsymbol { \mathrm { S } } ^ { r } \boldsymbol { \mathrm { T } } ^ { s }$ The proof for the case where $\boldsymbol { \sigma } = \boldsymbol { \mathrm { S } } ^ { r } \boldsymbol { \mathrm { T } } ^ { s }$ is more technical than the other three cases.   
We refer to Lemma 15 for a detailed proof.

Analysis of Case 4: $\boldsymbol { \sigma } = \boldsymbol { \mathrm { S } } ^ { r } \boldsymbol { \mathrm { T } } ^ { s }$ . The case $\boldsymbol { \sigma } = \boldsymbol { \mathrm { S } } ^ { r } \boldsymbol { \mathrm { T } } ^ { s }$ is more technical than the other three cases. We isolate the required construction in the following lemma. For the common constant in Lemma 11, we take $\lambda > 1$ to be the minimum of the constants obtained in the four cases. Due to the length and the techniques involved in the proof, in the rest of the section, we will give a brief and compressed version of the proof. We refer the reader to the supplementary materials for a complete rigorous proof.

Lemma 15. Let $D \in \mathbb { N } _ { + }$ . There exist constants $C > 0$ and $\lambda > 1$ such that for every $r , s \in  { \mathbb { N } } _ { + }$ with $r + s \le D ,$ , there exists a family of tasks $\{ F _ { m } \} _ { m \ge 1 }$ with the following properties. There exists a model $M ^ { \prime } \in \mathcal { M } _ { S ^ { r } \mathrm { T } ^ { s } } ( b _ { m } , b _ { m } , m + 1 )$ , where $b _ { m } : = \ \lceil C \rceil \log _ { 2 } m \rceil$ , that solves $F _ { m }$ exactly. On the other hand, $i f A \in \Sigma ^ { \leq D }$ satisfies ${ \cal S } ^ { r } \mathrm { T } ^ { s } \mathrm { ~ \underline { { { \cal ~ \ / ~ } } } } { \cal A } ,$ then every $M \in \mathcal { M } _ { A } ( B , P , W )$ with error at most $1 / 3$ on the input distribution $\nu _ { m }$ must satisfy $W \geq \lambda ( m + 1 )$ or max $\{ B , P \} \ge \lambda b _ { m }$

We first introduce the notation used in the proof. Fix $D \in  { \mathbb { N } } _ { + }$ and let $d : = \operatorname* { m a x } \{ D , 5 \}$ . We choose the finite constants used in the induction below first, and then choose $\kappa _ { D } > 0$ sufficiently large depending only on D. Set $C _ { D } : = \kappa _ { D } + 1 0 d$ and $\lambda _ { D } : = 1 + d / \kappa _ { D }$ . We require $\kappa _ { D } > 4 d D$ , so $1 < \lambda _ { D } < 1 + 1 / ( 4 D )$

For sufficiently large $m ,$ define $a _ { m } : = \lfloor \kappa _ { D } \log _ { 2 } m \rfloor$ $K : = 2 ^ { a _ { m } }$ , and $b _ { m } : = \left\lceil C _ { D } \log _ { 2 } m \right\rceil$ . We further put $B _ { * } : = P _ { * } : = [ \lambda _ { D } b _ { m } ] , S _ { * } : = 2 ^ { B _ { * } } , Q _ { * } : = 2 ^ { P _ { * } } , W _ { * } : = [ \lambda _ { D } ( m + 1 ) ]$ , and $w : = W _ { \ast } - 1$ . For fixed $r , s \geq 1$ with $r + s \le D ,$ let $q : = m - r$ . The chain task uses a source vector $\mathbf { z } _ { 0 } \in [ K ] ^ { r }$ and independent query blocks $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { s } \in [ K ] ^ { q }$

The proof of Lemma 16 fixes constants $N _ { 1 } , \ldots , N _ { d } ~ > ~ 0$ , depending only on $D ,$ and sets $n _ { j } : =$ $\lfloor N _ { j } m / \kappa _ { D } \rfloor$ . They are chosen so that $n _ { j } = \Theta _ { D } ( m )$ , every replay cut below lies strictly inside its corresponding query block, and consecutive replay cuts are separated by at least $w .$ In particular, we require $N _ { j + 1 } > N _ { j } + d$ for $j \in [ d - 1 ]$

Define $\bar { L } _ { 0 } : = \lfloor K ^ { r } / ( m ^ { 1 0 d } S _ { * } ^ { r - 1 } ) \rfloor$ . By taking $\kappa _ { D }$ sufficiently large, for all sufficiently large $m ,$

$$
\frac { K } { m ^ { 1 2 d ^ { 2 } } } \leq L _ { 0 } \leq \frac { K } { 2 } , \qquad \frac { m ^ { 2 } } { L _ { 0 } } = o ( 1 ) .
$$

We call a consecutive block of one or more S-layers a recurrent run. By Lemma 1, a recurrent run may be viewed as one S-layer whose state space is the product of the state spaces of the layers in the run. We next formalize the residual computation obtained after cutting a competing model inside the guard.

Definition 7 (Physical $j \cdot$ -stage replay). Fix lookup tables $\mathcal { H } = ( H _ { 1 } , \dots , H _ { s } )$ , a stage number $1 \leq j \leq d ,$ an incoming first-query length $n \in [ n _ { j } , q ]$ , a finite source set $\kappa .$ a source variable $\zeta \in \kappa .$ , and a fixed old context c. Let $\mathbf { Y } _ { 1 } \in [ K ] ^ { n }$ and $\mathbf { Y } _ { 2 } , \ldots , \mathbf { Y } _ { j } \in [ K ] ^ { q }$ be the raw query blocks. Write $\overline { { \mathbf { Y } } } : = ( \mathbf { Y } _ { 1 } , \ldots , \mathbf { Y } _ { j } ) =$ $( Y ^ { 1 } , \dots , Y ^ { L _ { j } ( n ) } )$ , where $L _ { j } ( n ) : = n + ( j - 1 ) q$

The replay has exactly these $L _ { j } ( n )$ query positions. It first forms a physical token sequence $\mathbf { X } ^ { ( 0 ) } =$ $( X _ { 1 } ^ { ( 0 ) } , \dots , X _ { { L } _ { i } ( n ) } ^ { ( 0 ) } ) \in [ Q _ { * } ] ^ { L _ { j } ( n ) }$ causally from the raw queries, where $X _ { t } ^ { ( 0 ) } = E _ { t } ( Y ^ { 1 : t } ; \mathcal { H } , c )$ . Thus $E _ { t }$ may depend on the lookup tables, the old context, and every raw query symbol that has arrived by position $t ,$ but not on the particular source value $\zeta .$ No additional token positions are introduced.

The continuation consists of $j$ recurrent runs $R _ { 1 } , \ldots , R _ { j }$ , separated by $j - 1 \textrm { T }$ -layers $T _ { 1 } , \ldots , T _ { j - 1 }$ Every T-layer has window at most $W _ { * }$ , every intermediate token belongs to $\left[ Q _ { * } \right]$ , and, $\operatorname { i f } S _ { i }$ denotes the state space of $R _ { i }$ , we require $\textstyle \prod _ { i = 1 } ^ { j } | S _ { i } | \leq S _ { * } ^ { d }$

The first recurrent run may start from an arbitrary post-table state $s _ { 0 } ^ { ( 1 ) } = \phi ( \zeta ) \in \mathcal { S } _ { 1 }$ . For $1 \leq i < j$ let $t _ { i } ( n ) : = n + i q - n _ { j - i }$ . Immediately before $R _ { i + 1 }$ processes position $t _ { i } ( n ) + 1$ , its state may be set to $s _ { t _ { i } ( n ) } ^ { ( i + 1 ) } = \Xi _ { i } ,$ , where $\Xi _ { i } : = \psi _ { i } ( \zeta , U _ { 0 } , Y ^ { 1 : t _ { i } ( n ) } ; \mathcal { H } , c )$ . Here $U _ { 0 }$ is one common auxiliary seed independent of $( \zeta , { \overline { { \mathbf { Y } } } } )$ and shared by all cuts.

After this initialization, $R _ { i + 1 }$ follows its declared S-layer transition and emission rules. In particular, $\Xi _ { i }$ may depend on the source and on every raw query symbol that has arrived by its cut, but not on any symbol after the cut. We denote the output of the final recurrent run at the last physical position by ${ \widehat { \mathbf { z } } } .$

The definition provides no additional advice tape: only the original $O _ { D } ( m )$ physical positions, one bounded token at each position, and the original raw-input arrival schedule are available.

The main technical ingredient for Case 4 is the following random-chain lemma.

Lemma 16 (Random-chain lemma). There exist a constant $c _ { D } > 0$ andfunctions $H _ { 1 } : [ K ] ^ { r } \times [ K ] ^ { q }  [ K ]$ and $H _ { i } : [ K ] \times [ K ] ^ { q }  [ K ] f o r 2 \leq i \leq s$ such that every $H _ { i } ( \cdot , \mathbf { x } ) , i \geq 2 ,$ , is a permutation $o f [ K ]$ ], and the following hold simultaneously for every $F \subseteq [ K ] ^ { r }$ with $L _ { 0 } \leq | F | \leq K$

Let $\mathbf { z } _ { 0 }$ be uniform on $F ,$ and let $\mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { s }$ be independent and uniform on $[ K ] ^ { q }$ . Define $\mathbf { z } _ { 1 } : =$ $H _ { 1 } ( \mathbf { z } _ { 0 } , \mathbf { x } _ { 1 } )$ and $\mathbf { z } _ { i } : = H _ { i } ( \mathbf { z } _ { i - 1 } , \mathbf { x } _ { i } ) f o r 2 \leq i \leq s .$ Then:

(i) for all but an $m ^ { - 1 0 0 d }$ fraction of $\mathbf { x } _ { 1 } \in [ K ] ^ { q }$ , every fiber of ${ \bf z } \mapsto H _ { 1 } ( { \bf z } , { \bf x } _ { 1 } )$ on F has cardinality at most $m ^ { 2 } { } _ { ; }$ ;

(ii) every physical s-stage replay in Definition 7 with $\boldsymbol { K } = \boldsymbol { F } , \boldsymbol { \zeta } = \mathbf { z } _ { 0 }$ , and full incoming first query $n = q$ satisfies $\operatorname* { P r } [ \widehat { \mathbf { z } } = \mathbf { z } _ { s } ] \leq m ^ { - c _ { D } } + o ( 1 )$

The same table realization works uniformly over all eligible source sets $F ,$ , fixed old contexts, and post-table physical replay maps.

ProofofLemma 15. Let $C _ { D } , \lambda _ { D } , \kappa _ { D }$ and $\mathcal { H } = ( H _ { 1 } , \dots , H _ { s } )$ be as in Lemma 16. We take $C = C _ { D }$ and $\lambda = \lambda _ { D }$ . Recall that $q = m - r$ and $K = 2 ^ { a _ { m } }$ . We construct $F _ { m }$ as the equal mixture of two tagged tasks.

The chain task. Draw $\mathbf { z } _ { 0 } \sim \mathrm { U n i f } ( [ K ] ^ { r } )$ and $\mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { s } \overset { \mathrm { i i d } } { \sim } \operatorname { U n i f } ( [ K ] ^ { q } )$ . Define ${ \bf z } _ { 1 } : = { \cal H } _ { 1 } ( { \bf z } _ { 0 } , { \bf x } _ { 1 } )$ and $\mathbf { z } _ { i } : = H _ { i } ( \mathbf { z } _ { i - 1 } , \mathbf { x } _ { i } )$ for $2 \leq i \leq s$ . The input is $I = ( { \bf z } _ { 0 } , { \bf 0 } ^ { 4 D ( m + 1 ) } , { \bf 0 } ^ { r } , { \bf x } _ { 1 } , \ldots , { \bf x } _ { s } )$ . We call the first zero block the guard and the second zero block the scratch block. The desired output is $\mathbf { z } _ { s }$

The reach task. Let $k _ { 0 } : = \lfloor m / 2 \rfloor$ . Draw $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { k _ { 0 } }$ independently and uniformly from $[ K ] ,$ , and independently draw $J \sim \mathrm { U n i f } ( [ k _ { 0 } ] )$ . Place $\mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { k _ { 0 } }$ at the beginning of the sequence and J at position $s m + 1$ , and ask the model to output $\mathbf { u } _ { J }$ . We use fixed left padding so that the two branches have the same length and include the branch tag in every token. Let $\nu _ { m }$ denote the resulting equal mixture.

Completeness. We first show that $S ^ { r _ { \mathrm { T } } s }$ solves both branches exactly. On the chain branch, write $\mathbf { z } _ { 0 } =$ $( z _ { 0 , 1 } , \ldots , z _ { 0 , r } )$ and number the S-layers from bottom to top. The ith S-layer copies its input until it reads $z _ { 0 , r - i + 1 }$ , stores this coordinate across the guard, releases it at scratch position $r - i + 1$ , and then resumes copying. At each release position, every higher S-layer has already released its own coordinate and acts as a copying layer. Hence, after the $r ~ { \mathrm { S - l a y e r s } }$ , the scratch block contains $\mathbf { z } _ { 0 }$ in its original order, while the query blocks remain unchanged.

Since $r + q = m$ , the first T-layer sees $\left( \mathbf { z } _ { 0 } , \mathbf { x } _ { 1 } \right)$ inside one window of size $m + 1$ and computes $\mathbf { z } _ { 1 }$ . Each later T-layer similarly sees $\left( \mathbf { z } _ { i - 1 } , \mathbf { x } _ { i } \right)$ and computes $\mathbf { z } _ { i }$ . Thus the final output is $\mathbf { z } _ { s }$

On the reach branch, all S-layers act as identities. The first $s - 1$ T-layers shift the database to the right by m positions per layer, and the final T-layer sees the requested entry together with J and outputs $\mathbf { u } _ { J }$ . The counters, position/type tags, release modes, and branch tag require only $O _ { D } ( \log m )$ bits, which are absorbed into $b _ { m }$

Guard reduction. Consider any prefix of a competing architecture containing h S-layers. Place a cut immediately after the source vector $\mathbf { z } _ { 0 }$ . When crossing an S-layer, record its state at the current cut. When crossing an T-layer, move the cut to the right by at most w positions. Since the architecture has depth at most D and $D w < 4 D ( m + 1 )$ , every such cut remains inside the guard.

Consequently, all recorded S-states depend only on $\mathbf { z } _ { 0 }$ . Their tuple defines a map $\Gamma : [ K ] ^ { r } \to \mathcal { G }$ with $| \mathcal { G } | \le S _ { * } ^ { h }$ . For $\gamma \in { \mathcal { G } }$ , let $F _ { \gamma } : = \Gamma ^ { - 1 } ( \gamma )$ . After fixing $\Gamma ( { \bf z } _ { 0 } ) = \gamma$ , replaying the prefix from the recorded states shows that every prefix-output token on the original scratch and query positions is a common causal function of the raw query symbols that have arrived by that position, independent of the particular ${ \bf z } _ { 0 } \in { \cal F } _ { \gamma }$

Fix $A \in \Sigma ^ { \leq D }$ with $\mathsf { S } ^ { r } \mathbb { T } ^ { s } \not \cong A$ and $M \in \mathcal { M } _ { A } ( B _ { * } , P _ { * } , W _ { * } )$ . We distinguish three cases.

Case 1: A contains fewer than s T-layers. We use the reach task. Starting from the final query position and tracing the dependence cone backward, every T-layer moves the cut to the left by at most $w ,$ while every S-layer contributes only its state across the cut. Since there are at most s − 1 T-layers,

$$
s m - ( s - 1 ) w = \bigl ( s - ( s - 1 ) \lambda _ { D } + o ( 1 ) \bigr ) m > \frac { m } { 2 } .
$$

Hence the backward dependence cone does not reach the database positions.

The final prediction is therefore a function of J and a message $\Gamma ( \mathbf { u } _ { 1 } , \ldots , \mathbf { u } _ { k _ { 0 } } )$ containing at most $D B _ { * }$ bits. By independence and entropy subadditivity, $I ( { \mathbf { u } } _ { J } ; \Gamma , J ) \le D B _ { * } / k _ { 0 }$ . The standard error-indicator/Fano estimate gives $p \leq ( D B _ { * } / k _ { 0 } + 1 ) / \log _ { 2 } K = o ( 1 )$ . Thus the reach-task success is $o ( 1 )$ .

Case 2: A contains at least s T-layers but fewer than r S-layers. Apply the guard reduction to the entire model. There are at most $S _ { * } ^ { r - 1 }$ source fibers. By the definition of $L _ { 0 } ,$ , the total source mass in fibers of size less than $L _ { 0 }$ is at most $m ^ { - 1 0 d }$ . Partition every remaining fiber into chunks $F$ satisfying $L _ { 0 } \leq | { \cal F } | < 2 L _ { 0 } \leq K$

Fix one such chunk. Conditional on $\mathbf { z } _ { 0 } \in F$ and on the query blocks, the output of the competing model is independent of the particular choice of $\mathbf { z } _ { 0 } \in F$ . By Lemma $1 6 ( \mathrm { i } )$ , outside an $m ^ { - 1 0 0 d }$ fraction of firstquery blocks, every fiber of ${ \bf z } _ { 0 } \mapsto H _ { 1 } ( { \bf z } _ { 0 } , { \bf x } _ { 1 } )$ on $F$ has size at most $m ^ { 2 }$ . Since every later $H _ { i } ( \cdot , \mathbf { x } _ { i } )$ is a permutation, the same bound holds for $\mathbf { z } _ { 0 } \mapsto \mathbf { z } _ { s }$ . Therefore,

$$
\operatorname* { P r } [ { \mathrm { s u c c e s s ~ } } | \ F ] \leq m ^ { - 1 0 0 d } + { \frac { m ^ { 2 } } { | F | } } \leq m ^ { - 1 0 0 d } + { \frac { m ^ { 2 } } { L _ { 0 } } } = o ( 1 ) .
$$

Adding the discarded source mass gives chain-task success $o ( 1 )$ .

Case 3: A contains at least r S-layers and at least s T-layers. Cut A immediately before its rth $\mathrm { S - }$ layer. The strict prefix contains exactly $r \mathrm { ~ - ~ } 1 \mathrm { ~ S - l a y e r s }$ . By the guard reduction, after discarding the same negligible small-fiber mass and fixing a retained source chunk $F ,$ , the prefix outputs on the original scratch and query positions are common causal functions of the arrived raw query symbols and are independent of the particular source $\mathbf { z } _ { 0 } \in F$

Write the residual profile as $\mathrm { S } ^ { b _ { 0 } } \mathrm { T } S ^ { b _ { 1 } } \mathrm { T } \cdot \cdot \cdot \mathrm { T } S ^ { b _ { j } }$ , where $b _ { 0 } \geq 1$ and $b _ { i } \geq 0$ . Since A avoids $S ^ { r _ { \mathrm { T } } s }$ , we have $j \le s - 1$ . Moreover, since A contains at least s T-layers and has depth at most $D _ { : }$ , the number $\begin{array} { r } { B _ { \mathrm { r e s } } : = \sum _ { i } b _ { i } } \end{array}$ satisfies $B _ { \mathrm { r e s } } \leq D - r - s + 1$

Merge every nonempty recurrent run into one product-state S-layer. Insert identity T-layers until there are exactly $s - 1 \mathrm { \ T - l a y e r s } .$ , and insert a copying S-layer into every empty recurrent slot. A copying S-layer uses at most $Q _ { * } = S _ { * }$ states. Since at most $s - 1$ such layers are inserted,

$$
S _ { * } ^ { B _ { \mathrm { r e s } } + s - 1 } \leq S _ { * } ^ { D - r } \leq S _ { * } ^ { d } .
$$

The padded displayed profile need not itself have depth at most D. The original depth bound has already been used in the guard reduction, while Definition 7 only requires the recurrent-state product bound together with the original token and window bounds.

It remains to verify that the padded residual is a physical s-stage replay. Put the actual state of the first residual recurrent run immediately before the first query position into $\phi ( \mathbf { z } _ { 0 } )$ . Its input tokens on the original query positions are the source-independent causal tokens provided by the guard reduction. I ${ \mathrm { ~ f ~ } } s = 1$ , this already gives the one-stage replay.

Suppose $s \geq 2$ . Let $t _ { 1 } : = 2 q - n _ { s - 1 }$ and $t : = t _ { 1 } - w$ . The parameter choices give $0 < t < q$ . Thus, for $^ { 1 S , }$ every payload-relative position $p \geq t _ { 1 } + 1$ , the local window immediately before the second recurrent run starts at $p - w \geq t + 1 > 0$ . Hence this window uses only outputs produced on query positions and cannot reach the guard or scratch block. All earlier source-dependent effects are therefore contained in the actual state of the second recurrent run at $t _ { 1 }$ , which is a legal granted state in Definition $^ { 7 . }$

Every later prescribed cut is separated from the previous one by at least w. The same argument therefore iterates. Each relevant local window lies entirely after the preceding cut, and every earlier source-dependent effect is contained in the actual state of the next recurrent run. Each such state depends only on $\mathbf { z } _ { 0 }$ , the common latent old-source information, and the raw query prefix already available at its cut. Thus the padded residual is a physical s-stage replay with full incoming first query $n = q$

Lemma 16(ii) gives chain-task success $o ( 1 )$ on every retained source chunk.

Conclusion. Every profile avoiding $S ^ { r _ { \mathrm { T } } s }$ falls into one of the three cases above. In Case 1 its reach-task success is $o ( 1 )$ , while in Cases $_ { 2 - 3 }$ its chain-task success is $o ( 1 )$ . We make no claim on the other branch and upper bound its success by one. Hence the total success is at most $1 / 2 + o ( 1 )$ ), and the error is at least $1 / 2 - o ( 1 ) > 1 / 3$ for sufficiently large m. For an input-independently randomized model, fix its random seed, apply the deterministic bound, and average over the seed. □

Proof of Lemma 16. We use the probabilistic method. Sample $H _ { 1 }$ by choosing all of its entries independently and uniformly from $[ K ]$ . Independently, for every $i \ \geq \ 2$ and every $\mathbf { x } \in [ K ] ^ { q }$ , sample $H _ { i } ( \cdot , \mathbf { x } )$ uniformly from the permutations of $[ K ]$ . We show that with positive probability the sampled tables satisfy all required properties simultaneously.

First-table occupancy and balance. Fix an eligible source set $F$ with $L _ { 0 } \leq | F | \leq K$ and a first-query column $\mathbf { x } \in [ K ] ^ { q }$ . Independence of the entries of $H _ { 1 }$ gives

$$
\begin{array} { r } { \operatorname* { P r } \biggr [ \underset { y \in [ K ] } { \operatorname* { m a x } } \left| \left\{ \mathbf { z } \in F : H _ { 1 } ( \mathbf { z } , \mathbf { x } ) = y \right\} \right| \geq m ^ { 2 } \biggr ] \leq K \binom { | F | } { m ^ { 2 } } K ^ { - m ^ { 2 } } } \\ { \leq K \left( \frac { e } { m ^ { 2 } } \right) ^ { m ^ { 2 } } . } \end{array}
$$

Concentration over the $K ^ { q }$ independent columns and a union bound over all eligible source sets show that, simultaneously for every $F _ { \mathrm { { ; } } }$ , at most an $m ^ { - 1 0 0 d }$ fraction of first-query columns violate the occupancy bound. This proves part (i).

We also use the following simultaneous balance event. Suppose a cut at position $t < q$ leaves a positive linear suffix of the first query unseen. Let $\Gamma = \gamma ( \mathbf { z } _ { 0 } , \mathbf { x } _ { 1 } ^ { 1 : t } )$ have range at most $S _ { * } ^ { d }$ , and let $e : [ K ] ^ { q } \to [ J ]$ be a source-independent label with $J \le m ^ { \chi m }$ for some fixed $\chi < \kappa _ { D }$ . Under the original independent uniform law of $\mathbf { z } _ { 0 } \in F$ and $\mathbf { x } _ { 1 } \in [ K ] ^ { q }$ , the average conditional law of $H _ { 1 } ( \mathbf { z } _ { 0 } , \mathbf { x } _ { 1 } )$ given $( \Gamma , e ( \mathbf { x } _ { 1 } ) )$ has total variation distance at most $m ^ { - \kappa _ { D } / 3 }$ from $\operatorname { U n i f } ( [ K ] )$

For fixed Γ and e, the expected total variation distance over $H _ { 1 }$ is at most $\scriptstyle { \frac { 1 } { 2 } } { \sqrt { K S _ { * } ^ { d } J / ( | F | K ^ { q } ) } }$ . A bounded-differences estimate is strong enough to union bound over all eligible source sets, early-state maps, and source-independent labels used below. The same argument applies when the first table is a random permutation table. For an incoming first-query length $n \in [ n _ { j } , q ]$ , the complete label at level $j$ will have rate strictly below $N _ { j }$ , whereas the shortest incoming query has entropy $n _ { j } \log _ { 2 } K = ( N _ { j } m + o ( m ) ) \log _ { 2 } m$ All balance estimates are taken under the original input law, before conditioning on a successful group or selecting a representative.

The mutual induction. Fix a sufficiently large constant $\rho _ { D }$ depending only on $D _ { \bullet }$ . For every $1 \leq j \leq d ,$ we prove simultaneously a program-hardness statement $G _ { j }$ and a prefix-capture statement $\mathcal { C } _ { j }$

The statement $G _ { j }$ is as follows. The initial live key k is uniform on $[ K ]$ and independent of the raw queries. The computation is a physical j-stage replay whose first query has length $n _ { j }$ , followed by $j - 1$ fresh q-symbol query blocks, and whose target is the corresponding j-fold random-permutation composition. Its success probability is at most $\epsilon _ { j } : = m ^ { - c _ { j } \kappa _ { D } }$ for some constant $c _ { j } > 0$ . The same bound holds for every incoming first-query length n $\in [ n _ { j } , q ]$ , after shifting all later absolute cuts by $n - n _ { j }$

The companion statement $\mathcal { C } _ { j }$ makes the capture property precise. Consider any legal continuation arising in the induction for which the source is uniform on a set of size at least $K / m ^ { \rho _ { D } }$ and the current live-key map has occupancy at most $m ^ { \rho D }$ . Fix any source-independent old context, but leave the bounded sourcedependent ancestor-state row unconditioned under the source law. At the prescribed cut, let $\boldsymbol { \xi }$ be the arrived prefix of the first unresolved raw query and let V denote the corresponding source-independent old physical word. Then there exists a source-independent list map $\mathcal { P }$ such that $\bar { | \mathcal { P } ( \mathbf { v } ) | } \overset { \cdot } { \leq } m ^ { A _ { j } m + o ( \bar { m } ) }$ for every realized v, and

$$
\operatorname* { P r } [ \operatorname { S u c c } \mathbf { \boldsymbol { \cdot } } \ \mathbf { \boldsymbol { \xi } } \not \in { \mathcal { P } } ( \mathbf { V } ) ] \leq \epsilon _ { j } .
$$

Here Succ is the branch-masked success event of the continuation. The source, context, and ancestor-history families that arise in the induction have cardinality at most $K ^ { O _ { D } ( m ) }$ , so all events above can be required simultaneously.

The constants are chosen in the noncircular order $( G _ { 1 } , N _ { 1 } ) , ( { \mathcal { C } } _ { 1 } , A _ { 1 } ) , ( G _ { 2 } , N _ { 2 } ) , ( { \mathcal { C } } _ { 2 } , A _ { 2 } ) , \dots A { \mathrm { t } }$ level j, the finite training-tree size and error thresholds are first chosen from lower-stage rates. We then choose $N _ { j }$ larger than the total rate of the complete source-independent label needed to prove $G _ { j } ,$ and afterward choose $A _ { j }$ larger than the new capture-certificate rate. Since D is fixed, $\kappa _ { D }$ is chosen last so that every cut, entropy, occupancy, and moment inequality has positive slack.

Base case. The base case $G _ { 1 }$ is the single-run random-permutation bound. Fix the initial-state map, transition and emission rules, and output map of one recurrent run. For a fixed query column, every sourceindependent physical encoding produces one of at most $Q _ { * } ^ { n }$ physical words.

Let $\theta : = \textstyle \frac { 1 } { 2 } m ^ { - \kappa _ { D } / 1 6 }$ and $h : = \lceil \theta K \rceil$ . For a fixed physical word, correctness on h distinct keys forces h prescribed distinct constraints on a uniform random permutation. Hence

$$
\operatorname* { P r } [ { \mathrm { a g r e e m e n t ~ o n ~ a t ~ l e a s t } } h { \mathrm { ~ k e y s } } ] \leq { \binom { K } { h } } { \frac { 1 } { ( K ) _ { h } } } = { \frac { 1 } { h ! } } .
$$

Thus one query column is bad for a fixed machine template with probability at most ${ Q } _ { * } ^ { n } / h !$

The query columns are independent. For a sufficiently large fixed polynomial $J _ { \mathrm { b a d } }$ in $m$

$$
\operatorname* { P r } [ \mathrm { a t } \mathrm { l e a s t } J _ { \mathrm { b a d } } \mathrm { b a d } \mathrm { c o l u m n s } ] \leq \left( { K } _ { \mathrm { b a d } } ^ { n } \right) \left( \frac { Q _ { * } ^ { n } } { h ! } \right) ^ { J _ { \mathrm { b a d } } } .
$$

Choosing the degree of $J _ { \mathrm { b a d } }$ sufficiently large makes the union bound over all machine templates valid, while $J _ { \mathrm { b a d } } / K ^ { n } = o ( 1 )$ . This proves $G _ { 1 }$ , uniformly over the permitted incoming lengths and fixed-prefix slices. The same independent-column argument after separating heavy and diffuse prefix atoms proves $\mathcal { C } _ { 1 }$

The capture step. Assume that all lower statements have been proved. We use the following certificate estimate, which is the only property of the analytical training tree needed below.

After applying the lower capture statements from the leaves upward, normalize every later unresolved raw prefix by its capture-list rank, inverse rank, and the finite branch information needed to identify the

corresponding fixed-prefix slice. Suppose the remaining unresolved prefix distribution has every atom of mass at most $1 / N$ . If the current continuation has success mass $p ,$ then for some $R _ { j } = O _ { D } ( 1 )$ ,

$$
p ^ { R _ { j } } \leq \theta _ { j } + \eta _ { \mathrm { b a d } , j } + \eta _ { \mathrm { c o l l } , j } + \frac { M _ { j } } { N } , \qquad M _ { j } \leq m ^ { \widetilde A _ { j } m + o ( m ) } .
$$

Here $\theta _ { j } , \eta _ { \mathrm { b a d } , j }$ , and $\eta _ { \mathrm { c o l l } , j }$ are $m ^ { - \Omega _ { D } ( \kappa _ { D } ) }$ . The quantity $M _ { j }$ counts the normalized certificates consisting of the lower capture-list ranks, inverse-context ranks, branch indices, and the finite training fingerprint.

To see the estimate, sample $R _ { j }$ independent analytical training branches. Outside the bad and collision events, every joint success determines one of the $M _ { j }$ normalized certificates. The representative attached to a certificate is tested only on its own arrived raw prefixes and on legal causal frontier states. In particular, the validity test is performed before representatives are grouped. After a certificate is fixed, a diffuse unresolved prefix contributes mass at most $1 / N$ . Summing over the $M _ { j }$ certificates gives the displayed bound. This is also why replacing a representative inside a normalized group never tests it against a foreign raw-prefix assignment.

We now prove $\mathcal { C } _ { j }$ . At each source-independent old context, place every arrived-prefix value whose posterior mass exceeds $m ^ { - A _ { j } m }$ into the capture list. There are fewer than $m ^ { { \dot { A } } _ { j } m }$ such values. Restrict the remaining source measure to the other values without renormalizing. Every remaining atom then has mass at most $1 / N$ for $N = m ^ { A _ { j } m }$ . Choose $A _ { j } > \widetilde { A } _ { j }$ . The certificate term satisfies

$$
\frac { M _ { j } } { N } \leq m ^ { - ( A _ { j } - \widetilde { A } _ { j } ) m + o ( m ) } = o ( 1 ) .
$$

The certificate estimate shows that the successful mass remaining outside the capture list is at most $\epsilon _ { j }$ after decreasing $c _ { j }$ if necessary. This proves $\mathcal { C } _ { j }$

Closing the program bound. We next prove $G _ { j }$ . Fix an incoming first-query length $n \in [ n _ { j } , q ]$ . Write the second query as $\mathbf { x } _ { 2 } = \left( \pmb { \xi } , \mathbf { y } _ { 2 } \right)$ , where $\pmb { \xi } \in [ K ] ^ { \bar { q } - n _ { j - 1 } }$ and $\mathbf { y } _ { 2 } \in [ K ] ^ { n _ { j - 1 } }$ . Let $c : = n + q - n _ { j - 1 }$ be the first replay cut and put $t : = c - w$ . The parameter choices give $0 < t < n$ , so a positive linear suffix of $\mathbf { x } _ { 1 }$ remains unseen at the balance cut.

Let $\mathbf { v } ( \mathbf { x } _ { 1 } , \pmb { \xi } ) : = \mathbf { X } _ { t + 1 : c } ^ { ( 0 ) } ( \mathbf { x } _ { 1 } , \pmb { \xi } ) \in [ Q _ { * } ] ^ { w }$ . This old physical word may depend on the complete first query and on the arrived second-query prefix $\xi ,$ but not on the source or on the remaining raw future.

$\operatorname { A p p l y } C _ { j - 1 }$ at this cut, leaving the bounded ancestor-state row unconditioned under the source law. For each $\mathbf { v } \in [ Q _ { * } ] ^ { w }$ , let ${ \mathcal { P } } ( { \mathbf { v } } )$ be its captured-prefix list. Put $A _ { < j } : = \operatorname* { m a x } _ { i < j } A _ { i }$ and $A ^ { * } : = m ^ { A < j m + o ( m ) }$ . For each $\xi ,$ define $B _ { \pmb \xi } : = \{ \mathbf v : \pmb \xi \in \mathcal { P } ( \mathbf v ) \}$ }. Double counting gives $\begin{array} { r } { \sum _ { \pmb { \xi } } | B _ { \pmb { \xi } } | \le A ^ { * } Q _ { * } ^ { w } } \end{array}$ . Hence, for every cutoff $L _ { \mathrm { c t x } } \geq 1 , \mathrm { P r } [ | B _ { \xi } | > L _ { \mathrm { c t x } } ] \leq A ^ { * } Q _ { * } ^ { w } / ( K ^ { q - n _ { j - 1 } } L _ { \mathrm { c t x } } )$

All label fields other than the inverse-context rank have rates already fixed by lower stages. We choose $L _ { \mathrm { c t x } }$ and then $N _ { j }$ so that the exceptional prefix mass is $m ^ { - \Omega _ { D } ( \kappa _ { D } ) }$ and the rank of $\mathbf { v } ( \mathbf { x } _ { 1 } , \pmb { \xi } )$ inside $B _ { \xi }$ together with the normalized training fingerprint and all previously fixed alignment ranks, gives a complete source-independent label $e = e _ { \xi } ( \mathbf { x } _ { 1 } )$ satisfying $| \operatorname { I m } ( e ) | \leq m ^ { \chi _ { j } \grave { m } + o ( m ) }$ for some $\chi _ { j } < N _ { j }$ . A sentinel is used on excluded inputs. The branch and training indices are fixed on the current slice and suppressed from the notation.

For each label, choose one full first-query representative from its group. The representative is analytical lookup data indexed by the label; its raw coordinates are not another field of the label. Let $\Gamma = \gamma ( \mathbf { z } _ { 0 } , \mathbf { x } _ { 1 } ^ { 1 : t } )$ be the state of the preceding recurrent run at the balance cut. For a fixed label value $^ { a , }$ early row $^ { g , }$ and live key k $\in ~ [ K ]$ , let $q _ { a , g } ( \mathbf { k } )$ be the maximum fresh branch-masked continuation success of the selected representative over all legal causal choices of the current frontier state and all later frontier policies.

The normalization procedure retains a successful training sample only when its own representative has fresh continuation value greater than a fixed $\delta _ { 0 } > 0$ . Samples failing this test are included in $\eta _ { \mathrm { b a d } , j }$ . Thus every retained successful sample with label a, row $^ { g , }$ , and live key k satisfies $q _ { a , g } ( { \bf k } ) > \delta _ { 0 }$

For each fixed $( a , g )$ and every $\mathbf { k } ,$ choose one deterministic maximizing legal causal policy. Each chosen frontier state depends only on the raw history available at its own cut. Therefore these keywise choices paste into one legal lower-stage physical continuation on the same positions; no choice depends on an unseen raw suffix.

For a fixed analytical training tree, let Cross ${ \bf \Psi } ^ { \mathrm { o p t } } : = \mathbb { E } [ q _ { e , \Gamma } ( \mathbf { K } _ { \mathrm { l i v e } } ) ]$ under the original source and query law, and let H be the retained training-success mass. The pointwise own-source test, the simultaneous balance event, and $G _ { j - 1 }$ give

$$
\delta _ { 0 } H \le \mathrm { C r o s s } ^ { \mathrm { o p t } } \le \epsilon _ { j - 1 } + \delta _ { \mathrm { T V } } .
$$

Indeed, the first inequality follows pointwise. For the second, apply the balance estimate once to the complete label e and early row Γ. After replacing the conditional live-key law by uniform, the pasted optimizing policy is one admissible lower-stage physical replay, so $G _ { j - 1 }$ applies.

Averaging over the analytical training tree and combining this estimate with the certificate bound gives

$$
p ^ { R _ { j } } \leq \frac { \epsilon _ { j - 1 } + \delta _ { \mathrm { T V } } } { \delta _ { 0 } } + \theta _ { j } + \eta _ { \mathrm { b a d } , j } + \eta _ { \mathrm { c o l l } , j } .
$$

All expectations over the old source and first query retain their original weights. Excluded inputs are represented by the sentinel or by validity indicators, and the balance estimate is applied before averaging over fixed-prefix slices.

The recursive parameter choice leaves a positive exponent after taking the $R _ { j }$ th root, and hence proves $G _ { j }$ . Notice that the argument uses the optimizing causal frontier policy only as an analytical upper bound; we do not assert that the competing model’s actual future frontier states have the same fresh continuation success.

Returning to the iid first stage. Fix an eligible source chunk F with $L _ { 0 } \leq | { \cal F } | \leq K$ . By part (i), outside an $m ^ { - 1 0 0 d }$ fraction of first-query columns, the map ${ \bf z } _ { 0 } \mapsto H _ { 1 } ( { \bf z } _ { 0 } , { \bf x } _ { 1 } )$ on $F$ has occupancy at most $m ^ { 2 }$ Every later permutation preserves this occupancy, and the bad-column mass is charged under the original first-query law.

Repeat the $G _ { s }$ certificate construction with full incoming first-query length q. Fix the arrived secondquery prefix $\xi ,$ the inverse-branch indices, and the analytical training tree. On this slice, the complete source-independent label has the form $e _ { \pm } ( \mathbf { x } _ { 1 } )$ and depends only on the first query.

Write $\mathbf { x } _ { 2 } = ( \pmb { \xi } , \mathbf { y } _ { 2 } )$ and let $\mathbf { Y } : = ( \mathbf { y } _ { 2 } , \mathbf { x } _ { 3 } , \ldots , \mathbf { x } _ { s } )$ be the remaining fresh raw future. The only change from the permutation-first-stage argument is the first balance step, where we use the simultaneous iid firsttable balance event proved above.

Conditional on the first live key $\mathbf { z } _ { 1 }$ , the complete label, and the early row, all old variables have a joint conditional law independent of $\mathbf { Y }$ . This conditional law can be represented using the common auxiliary seed allowed in Definition 7. For fixed label and early row, the representative’s physical token sequence and replay rules are fixed. Any remaining old-variable dependence can therefore affect only the permitted frontier states, and each such state depends only on the raw prefix available at its own cut. Thus the continuation remains in the physical replay class.

Applying the optimized crossed-certificate argument on every fixed-prefix slice and then averaging under the original prefix distribution gives $\mathrm { P r } [ \widehat { \mathbf { z } } = \mathbf { z } _ { s } ] \leq m ^ { - c _ { D } } + o ( 1 )$ . For $s = 1$ , the iid-column single-run factorial argument applies directly.

All occupancy, balance, capture, and program events above are simultaneous before the source chunk and the post-table continuation are selected. Their intersection therefore has positive probability. Fixing one lookup-table realization in this intersection proves the lemma. □

Now we are able to give the proof of Theorem 4. Theorem 4 follows immediately from Lemma 10, Lemma 9 and Lemma 11.

ProofofTheorem 4. Let $ { \mathcal { C } } _ { 2 } ( \tau )$ be size-2 cover of the target profile τ. By Lemma 11, for every profile $\sigma \in$ $ { \mathcal { C } } _ { 2 } ( \tau )$ , there exists a family of task $\{ H _ { \sigma , m } \}$ , such that there exists a model $M ^ { \prime } \in \mathcal { M } _ { \sigma } ( b _ { m } , b _ { m } , m + 1 )$ that can solve $H _ { \sigma , m }$ exactly, where $b _ { m } = C \log m$ , while every hybrid model $M \in \mathcal { M } _ { A } ( B , P , W )$ that satisfies err $( M , H _ { \sigma , m } ) \leq 1 / 3$ , for some input distribution $\nu _ { \sigma , m }$ must satisfy $W \geq \lambda ( m + 1 )$ or max $\{ B , P \} \ge \lambda b _ { m } .$ if $A \in \Sigma ^ { \leq D } , \sigma \not \geq A$ . By Lemma 10, by combining the mixture of these tasks, we obtain a single family of tasks $\left\{ \boldsymbol { G } _ { \tau , m } \right\}$ such that for every hybrid model $M \in \mathcal { M } _ { A } ( B , P , W )$ that satisfies err $( M , G _ { \tau , m } ) \leq 1 / ( 4 D )$ must satisfy $W \geq \lambda ( m + 1 )$ or max $\{ B , P \} \ge \lambda C$ log m, if $A \in \Sigma ^ { \leq D }$ and $\exists \sigma \in \mathcal { C } _ { 2 } ( \tau ) , \sigma \not \cong A$ . On the other hand, there exists a model $M ^ { \prime } \in \mathcal { M } _ { \tau } ( C \log m , C \log m , , m + 1 )$ that can solve $G _ { \tau , m }$ exactly. Finally, by Lemma $9 , \operatorname { i f } \tau \not \subset A$ , then there must be some $\sigma \in \mathcal { C } _ { 2 } ( \tau )$ such that $\sigma \not \simeq A$ , which completes the proof of Theorem 4. □

## E.3 Proof of Theorem 1

Given Theorem 3 and Theorem 4, the proof of Theorem 1 is almost immediate.

ProofofTheorem 1. Let $\alpha : = \mathrm { c o m p } ( \tau )$ . Let $\{ F _ { \tau , \epsilon , m } ^ { \mathrm { o r d } } \} _ { m \geq 1 }$ be the task family constructed in Theorem 3 for the wrong-order case, with input distribution $\mu _ { m } ^ { \mathrm { o r d } }$ , and let $\{ F _ { \tau , m } ^ { \mathrm { i n s } } \} _ { m \geq 1 }$ be the task family constructed in Theorem 4 for the insufficient-profile case, with input distribution $\mu _ { m } ^ { \mathrm { i n s } }$ . That common constant factor can be selected so that Theorem 3 and Theorem 4 still hold.

We combine the two task families into a single task family $\{ F _ { \tau , \epsilon , m } \} _ { m \ge 1 }$ . Sample J uniformly from $\{ 0 , 1 \}$ . Conditioned on $J = 0 ;$ , sample an input from $\mu _ { m } ^ { \mathrm { o r d } }$ and ask the model to compute $F _ { \tau , \epsilon , m } ^ { \mathrm { o r d } } ;$ conditioned on $J = 1$ , sample an input from $\mu _ { m } ^ { \mathrm { i n s } }$ and ask the model to compute $F _ { \tau , m } ^ { \mathrm { i n s } }$ . We include J as an additional tag in every input token and pad the two types of inputs to the same sequence length. Denote the resulting input distribution by $\mu _ { m }$

By the completeness constructions of Theorems 3 and 4, both component tasks can be solved exactly by models with profile τ. Using the tag J to select between the corresponding layer maps, as in the selector construction in Lemma 10, we obtain a single model $M _ { \tau , m } \in \mathcal { M } _ { \tau } ( B _ { \tau , m } , P _ { \tau , m } , W _ { \tau , m } )$ . that solves $F _ { \tau , \epsilon , m }$ exactly. After increasing the constant $C _ { \tau , \mathfrak { e } }$ <sub>ϵ</sub> if necessary, $B _ { \tau , m } , P _ { \tau , m } \leq C _ { \tau , \epsilon } \log m , W _ { \tau , m } \leq m + 1$ . Moreover, taking the union of the two vocabularies and padding the inputs preserves $| \mathcal { V } _ { \epsilon , m } | \le \mathrm { p o l y } _ { \epsilon } ( m ) , L _ { \epsilon , m } \le$ poly (m).

It remains to prove soundness. Fix an architecture profile $A \ \in \ \Sigma ^ { \leq D }$ such that $\tau \ \not \simeq \ A$ , and let $M \in \mathcal { M } _ { A } ( B , P , W )$ satisfy $\textstyle \operatorname { e r r } _ { \mu _ { m } } ( M , F _ { \tau , \epsilon , m } ) \leq { \frac { 1 } { 8 D } }$ . Let $e _ { \mathrm { o r d } }$ and $e _ { \mathrm { i n s } }$ denote the conditional errors of M conditioned on $J = 0$ and $J = 1$ , respectively. Since J is uniform,

$$
\mathrm { e r r } _ { \mu _ { m } } ( M , F _ { \tau , \epsilon , m } ) = \frac 1 2 \left( e _ { \mathrm { o r d } } + e _ { \mathrm { i n s } } \right) .
$$

Therefore, $\begin{array} { r } { e _ { \mathrm { o r d } } , e _ { \mathrm { i n s } } \le 2 \mathrm { e r r } _ { \mu _ { m } } ( M , F _ { \tau , \epsilon , m } ) \le \frac { 1 } { 4 D } } \end{array}$

We now distinguish two cases.

Case 1: $\alpha \not \simeq A .$ . Since $\begin{array} { r } { e _ { \mathrm { o r d } } \le \frac 1 { 4 D } \le \frac 1 3 } \end{array}$ , Theorem 3 applies to the wrong-order component and gives

$$
W \geq c _ { \tau } m ^ { 1 / \epsilon } \qquad \mathrm { o r } \qquad \operatorname* { m a x } \{ B , P \} \geq c _ { \tau } m .
$$

Case 2: $\alpha \preceq A$ . Since $\tau \not \simeq A$ by assumption, we have $\alpha \preceq A$ and $\tau \not \simeq A$ . Thus A is in the insufficient-profile case. Since $\begin{array} { r } { e _ { \mathrm { i n s } } \ \leq \ \frac { 1 } { 4 D } } \end{array}$ , Theorem 4 applies to the insufficient-profile component and gives

$$
W \geq \lambda _ { \tau } ( m + 1 ) \qquad \mathrm { o r } \qquad \operatorname* { m a x } \{ B , P \} \geq \lambda _ { \tau } C _ { \tau } \log m .
$$

The two cases are exhaustive, since either α $\nprec A$ or $\alpha \preceq A$ . This proves both soundness statements and completes the proof. □

## F Omitted Details in Section 2.2

We provide proofs to various theorems in the main body. We first prove the core hardness result for tasks with a dominant TS architecture. We then look specifically at the two tasks, segmented retrieval and exact copy, showing that these tasks are indeed in this family of tasks. Lastly, we provide constructions for these two tasks that use less window and state-size than what is provably required for ST architectures.

## F.1 Hardness Results

We begin with the core assumption for all tasks with a TS dominant architecture before quickly following with the statement and proof of the SSM → TF theorem. After establishing this result, we show the specific bounds for exact copy and segmented retrieval, where the majority of the proofs come from defining the hard data distribution and cut position τ .

Assumption 1. Let $F$ be a sequence map from $\mathcal { V } ^ { L }$ to Y. We assume F satisfies the following property: there exists an index $\tau , { \mathrm { a } }$ set of prefixes $\mathcal { U } \subseteq \mathcal { V } ^ { \tau } \times \{ 0 , 1 \} ^ { \tau }$ , and a set of suffixes $\mathcal { Q } \subseteq \mathcal { V } ^ { L - \tau } \times \bar { \{ 0 , 1 \} } ^ { L - \tau }$ such that the map $\Gamma : \mathcal { U } \to \mathcal { V } ^ { | \mathcal { Q } | }$ defined as

$$
\Gamma ( u ) : = ( F ( u , q ) ) _ { q \in \mathcal { Q } } \in \mathcal { y } ^ { | \mathcal { Q } | } ,
$$

is an injection. We call τ the cut.

Theorem 5 (SSM → TF Cut, Restated). Consider any function F that satisfies Assumption 1. Let M be a model of architecture ST modeling F with state S and window W. Suppose that the cut position is bounded $a s \tau \leq L - W$ . If M computes $f : \mathcal { V } ^ { L } \to \mathcal { V }$ correctly on all the inputs, then $| S | \geq | U |$

Proof. Let $S _ { t }$ denote the state at position t. Let $u , u ^ { \prime } \in \mathcal { U }$ be prefixes such that $S _ { \tau } ( u ) = S _ { \tau } ( u ^ { \prime } )$ . Since the SSM is deterministic, at time $t \geq \tau$ , the SSM states remain identical, i.e.

$$
S _ { t } ( u , q ) = S _ { t } ( u ^ { \prime } , q ) , \forall t > \tau ,
$$

and the SSM layer outputs are the same.

Since the TF layer can only access the tokens from position $L - W$ to L, and $\tau \leq L - W$ , we have that the model outputs are also the same, i.e.

$$
M ( u , q ) = M ( u ^ { \prime } , q ) .
$$

Now assume M computes f correctly on all inputs and the map Γ of a set of prefixes U is injective (Assumption 1). If two prefixes $u \ne u ^ { \prime }$ outputs the same state at the cut position τ, then by the theorem we have

$$
M ( u , q ) = f ( u , q ) = f ( u ^ { \prime } , q ) = M ( u ^ { \prime } , q ) , \quad \forall q \in \mathcal { Q } ,
$$

which immediately shows that $\Gamma ( u ) = \Gamma ( u ^ { \prime } )$ . Since the map Γ is injective, we have $u = u ^ { \prime }$ , which contradicts the assumption that $u \ne u ^ { \prime } .$ . Therefore, the map

$$
u \mapsto S _ { t } ( u ) , \quad \forall u \in \mathcal { U } ,
$$

must be injective, which implies that $| S | \ge | U |$

Theorem 6 (Exact Copy Lower Bound). $S e t \left| \mathcal { V } \right| \geq 5 , L \geq c l$ , and define $h : = \operatorname* { m i n } ( l , ( L - W ) _ { + } )$ . If a deterministic hybrid model ST performs the Exact Copy task correctly on all inputs, then log $| S | \geq \Omega ( h \log | \mathcal { V } | )$

Proof. We use the notation ofjuxtaposition to mean sequence concatenation. For example, for any two finite sequences

$$
u = ( u _ { 1 } , \ldots , u _ { m } ) \in \mathcal { V } ^ { m } \qquad \mathrm { a n d } \qquad v = ( v _ { 1 } , \ldots , v _ { n } ) \in \mathcal { V } ^ { n } ,
$$

we write

$$
u v : = ( u _ { 1 } , \ldots , u _ { m } , v _ { 1 } , \ldots , v _ { n } ) .
$$

The result follows immediately if $h = 0$ . Thus, we will assume $h \geq 1$

We first pick two distinct elements $g _ { 1 } , g _ { 2 } \in \mathcal { V }$ , and set $\mathcal { A } : = \mathcal { V } \setminus \left\{ g _ { 1 } , g _ { 2 } \right\}$ . Since $| \nu | \geq 5 ,$ we have $| { \mathcal { A } } | \geq 3$ , thus we may pick another element $\alpha \in { \mathcal { A } }$ . For each $a _ { 1 : h } = ( a _ { 1 } , \cdot \cdot \cdot , a _ { h } ) \in \mathcal { A } ^ { h }$ , we define the prefix map $u : { \mathcal { A } } ^ { h }  { \mathcal { A } } ^ { l }$ by

$$
u ( a _ { 1 : h } ) = a _ { 1 : h } \alpha ^ { l - h } = ( a _ { 1 } , \cdot \cdot \cdot , a _ { h } , \underbrace { \alpha , \cdot \cdot \cdot , \alpha } _ { l - h } ) \in { \mathcal A } ^ { l } .
$$

We also define

$$
\mathbf { g } _ { 1 : L - c l } : = g _ { 1 } ^ { l } g _ { 2 } ^ { l } g _ { 1 } ^ { l } g _ { 2 } ^ { l } \cdot \cdot \cdot ,
$$

where the alternating concatenation is truncated after its first $L - c l$ entries. We refer to $\mathbf { g } _ { 1 : L - c l }$ as the failure segment. By construction, every symbol in $\mathbf { g } _ { 1 : L - c l }$ differs from the symbol l positions later whenever both are defined.

For each $a _ { 1 : h } \in \mathcal { A } ^ { h }$ , we define the suffix map $q : { \mathcal { A } } ^ { h }  \mathcal { V } ^ { L - h }$ by

$$
\begin{array} { l } { { q ( a _ { 1 : h } ) : = \alpha ^ { l - h } u ( a _ { 1 : h } ) ^ { c - 1 } { \bf g } _ { 1 : L - c l } } } \\ { { \qquad = \alpha ^ { l - h } ( a _ { 1 } \cdot \cdot \cdot a _ { h } \alpha ^ { l - h } ) ^ { c - 1 } g _ { 1 } ^ { l } g _ { 2 } ^ { l } g _ { 1 } ^ { l } g _ { 2 } ^ { l } \cdot \cdot \cdot . . } } \end{array}
$$

We define $f _ { \mathrm { E C } } : \mathcal { V } ^ { L } \to \{ 0 , 1 \}$ to be the function

$$
f _ { \mathrm { E C } } ( \mathbf { x } _ { 1 : L } ) : = \mathbf { 1 } _ { \left\{ \mathbf { x } _ { 1 : L } { \mathrm { ~ h a s ~ a n ~ e x a c t ~ c o p y ~ o f ~ r e p e t i t i o n ~ } } c { \mathrm { ~ a n d ~ s t r i n g ~ l e n g t h ~ } } l \right\} } .
$$

For some fixed $a _ { 1 : h } \in \mathcal { A } ^ { h }$ , then it is clear that the sequence

$$
\begin{array} { c } { { a _ { 1 : h } q ( a _ { 1 : h } ) = a _ { 1 } \cdot \cdot \cdot a _ { h } \alpha ^ { l - h } ( a _ { 1 } \cdot \cdot \cdot a _ { h } \alpha ^ { l - h } ) ^ { c - 1 } g _ { 1 } ^ { l } g _ { 2 } ^ { l } g _ { 1 } ^ { l } g _ { 2 } ^ { l } \cdot \cdot \cdot } } \\ { { { } } } \\ { { = ( a _ { 1 } \cdot \cdot \cdot a _ { h } \alpha ^ { l - h } ) ^ { c } g _ { 1 } ^ { l } g _ { 2 } ^ { l } g _ { 1 } ^ { l } g _ { 2 } ^ { l } \cdot \cdot \cdot } } \end{array}
$$

contains an exact copy, i.e. $f _ { \mathrm { E C } } ( a _ { 1 : h } q ( a _ { 1 : h } ) ) = 1$

For any $a _ { 1 : h } ^ { \prime } \in \mathcal { A } ^ { h }$ such that $a _ { 1 : h } ^ { \prime } \neq a _ { 1 : h }$ , then

$$
a _ { 1 : h } ^ { \prime } q ( a _ { 1 : h } ) = a _ { 1 } ^ { \prime } \cdot \cdot \cdot a _ { h } ^ { \prime } \alpha ^ { l - h } \left( a _ { 1 } \cdot \cdot \cdot a _ { h } \alpha ^ { l - h } \right) ^ { c - 1 } g _ { 1 } ^ { l } g _ { 2 } ^ { l } g _ { 1 } ^ { l } g _ { 2 } ^ { l } \cdot \cdot \cdot \ .
$$

For $\mathbf { x } \in \mathcal { V } ^ { L }$ , define

$$
E ( \mathbf { x } ) : = \left\{ m \in [ 1 , L - l ] : \mathbf { x } _ { m } = \mathbf { x } _ { m + l } \right\} .
$$

By the definition of Exact Copy,

$$
f _ { \mathrm { E C } } ( \mathbf { x } ) = 1 \quad \Longleftrightarrow \quad E ( \mathbf { x } ) \mathrm { c o n t a i n s } ( c - 1 ) l \mathrm { c o n s e c u t i v e i n t e g e r s } .\tag{1}
$$

We now show that $| E ( a _ { 1 : h } ^ { \prime } q ( a _ { 1 : h } ) ) | < ( c - 1 ) l$ , which implies this sequence contains no exact copy. Define $\mathbf { x } ^ { \prime } : = a _ { 1 : h } ^ { \prime } q ( a _ { 1 : h } ) $ , we now determine $E ( \mathbf { x } ^ { \prime } )$ . For $1 \leq m \leq h$ , we have $\mathbf { x } _ { m } ^ { \prime } = a _ { m } ^ { \prime } , \mathbf { x } _ { m + l } ^ { \prime } = a _ { m } .$ and hence m $\in E ( \mathbf { x } ^ { \prime } ) \Longleftrightarrow a _ { m } ^ { \prime } = a _ { m }$ . For $h < m \leq l ,$ both $\mathbf { x } _ { m } ^ { \prime }$ and $\mathbf { x } _ { m + l } ^ { \prime }$ are equal to α. Moreover, for $l < m \leq ( c - 1 ) l$ , the positions m and $m + l$ correspond to the same position in two consecutive copies of $a _ { 1 } \cdots a _ { h } \alpha ^ { l - h }$ , and therefore ${ \bf x } _ { m } ^ { \prime } = { \bf x } _ { m + l } ^ { \prime }$

On the other hand, for every remaining $m \in [ 1 , L - l ]$ with $m > ( c - 1 ) l$ , we have $\mathbf { x } _ { m } ^ { \prime } \neq \mathbf { x } _ { m + l } ^ { \prime }$ . Indeed, if m lies in the last copy of $a _ { 1 } \cdots a _ { h } \alpha ^ { l - h }$ , then $\mathbf { x } _ { m } ^ { \prime } \in \mathcal { A }$ while $\mathbf { x } _ { m + l } ^ { \prime } \in \{ g _ { 1 } , g _ { 2 } \}$ ; if m lies in the failure segment $\mathbf { g } _ { 1 : L - c l }$ , then $\mathbf { x } _ { m } ^ { \prime } \neq \mathbf { x } _ { m + l } ^ { \prime }$ by the construction of $\mathbf { g } _ { 1 : L - c l }$

Therefore,

$$
E ( \mathbf { x } ^ { \prime } ) = \{ m \in [ 1 , h ] : a _ { m } ^ { \prime } = a _ { m } \} \cup [ h + 1 , ( c - 1 ) l ] .
$$

In particular, $E ( \mathbf { x } ^ { \prime } ) \subseteq [ 1 , ( c - 1 ) l ]$

Since $a _ { 1 : h } ^ { \prime } \neq a _ { 1 : h } .$ , there exists some $j \in [ 1 , h ]$ such that $a _ { i } ^ { \prime } \neq a _ { j } .$ . Hence $j \not \in E ( \mathbf { x } ^ { \prime } )$ . Thus $E ( \mathbf { x } ^ { \prime } )$ does not contain all $( c - 1 ) l$ integers in $[ 1 , ( c - 1 ) l ]$ . Since $E ( \mathbf { x } ^ { \prime } ) \subseteq [ \dot { 1 } , ( c - 1 ) l ]$ , it cannot contain $( c - 1 ) l$ consecutive integers, i.e. $| E ( a _ { 1 : h } ^ { \prime } q ( a _ { 1 : h } ) ) | < ( c - 1 ) l$ . By equation 1, f<sub>EC</sub> $( a _ { 1 : h } ^ { \prime } q ( a _ { 1 : h } ) ) = 0$

Therefore, we have that

$$
f _ { \mathrm { E C } } \left( a _ { 1 : h } ^ { \prime } q \left( a _ { 1 : h } \right) \right) = \left\{ \begin{array} { l l } { 1 , } & { a _ { 1 : h } ^ { \prime } = a _ { 1 : h } , } \\ { 0 , } & { a _ { 1 : h } ^ { \prime } \neq a _ { 1 : h } . } \end{array} \right.
$$

Define $f : = f _ { \mathrm { E C } } , \mathcal { U } : = \mathcal { A } ^ { h } , \mathcal { Q } : = \{ q ( a _ { 1 : h } ) : a _ { 1 : h } \in \mathcal { A } ^ { h } \} , \tau : = h$ following the setting of Theorem 2. Notice that the map $a _ { 1 : h } \mapsto q ( a _ { 1 : h } )$ is injective, since $q ( a _ { 1 : h } )$ contains $u ( a _ { 1 : h } ) ;$ and Γ is also injective on U because Γ is a one-hot vector at the position in the sequence for each input:

$$
a _ { 1 : h } ^ { \prime } \mapsto \Gamma ( a _ { 1 : h } ^ { \prime } ) = \left( f \left( a _ { 1 : h } ^ { \prime } q \left( a _ { 1 : h } \right) \right) \right) _ { a _ { 1 : h } \in \mathcal { A } ^ { h } } = ( 0 , \cdot \cdot \cdot , 0 , 1 , 0 , \cdot \cdot \cdot 0 ) ,
$$

where the slot with value 1 corresponds to the position $a _ { 1 : h } = a _ { 1 : h } ^ { \prime }$

By Theorem 2, if M computes f correctly on all inputs, then

$$
| S | \geq | \mathcal { U } | = | A | ^ { h } ,
$$

Therefore,

$$
\log | S | \geq \Omega ( h \log | \mathcal { V } | ) .
$$

For the corollary, if $L \geq c l$ and $W \leq L - l ,$ , then $L - W \geq l ,$ Thus $h = \operatorname* { m i n } ( l , ( L - W ) _ { + } ) = l$ and

$$
\log | S | \geq \Omega ( l \log | \mathcal { V } | ) .
$$

Theorem 7 (Segmented Retrieval Lower Bound). Set $| { \mathcal { N } } | \geq 2$ and $L \ge 3 \lambda + 1$ , and $h : = \operatorname* { m i n } ( \lambda , ( L - W ) _ { + } )$ Then any deterministic hybrid model M of architecture ST that performs the Segmented Retrieval task correctly on all inputs must satisfy log $| S | \geq \Omega ( h \log | \mathcal { N } | )$

Proof. For $h = 0$ the bound is vacuous, so we assume $h \geq 1$

Define the prefix map u $: \mathcal { N } ^ { h }  ( \mathcal { N } \times \{ 0 , 1 \} ) ^ { h }$ by

$$
\begin{array} { r } { u ( n _ { 1 : h } ) = \left( \left( n _ { 1 } , 0 \right) , \left( n _ { 2 } , 0 \right) , \cdots , \left( n _ { h } , 0 \right) \right) , \quad \forall n _ { 1 : h } \in \mathcal { N } ^ { h } . } \end{array}
$$

Since $h \leq L - W$ , h can be a candidate for the cut position τ . Therefore, adversarially, we set $\tau = h$ . We fix a position $j \in [ h ]$ that serves as the query position for the segmented retrieval; that is to say, for each $v _ { 1 : L - h } \in \mathcal { V } ^ { L - h }$ , we set the suffix map $q _ { j } : \mathcal { V } ^ { \bar { L } \bar { - } h } \to ( \mathcal { V } \times \{ 0 , 1 \} ) ^ { \bar { L } - h }$ as

$$
q _ { j } \left( v _ { 1 : L - h } \right) : = \left( \left( v _ { 1 } , 0 \right) , \left( v _ { 2 } , 0 \right) , \cdot \cdot \cdot \left( v _ { j + \lambda - h } , 1 \right) , \cdot \cdot \cdot \left( v _ { L - h } , 0 \right) \right) .
$$

Therefore, the Online Segmented Retrieval task for $\mathbf { x } _ { 1 : L } : = u ( n _ { 1 : h } ) q _ { j } ( v _ { 1 : L - h } )$ is to output $n _ { j }$ Pick $n _ { 1 : h } ^ { \prime } , n _ { 1 : h } \in \mathcal { N } ^ { h }$ such that $n _ { 1 : h } ^ { \prime } \neq n _ { 1 : h }$ , and let $j ^ { \prime } \in [ h ]$ be the index such that $n _ { j ^ { \prime } } ^ { \prime } \neq n _ { j ^ { \prime } }$ . Fix $v _ { 1 : L - h } \in$ $\mathcal { V } ^ { L - h }$ . We construct two input sequences ${ \bf x } ^ { \prime } { } _ { 1 : L } : = u ( n _ { 1 : h } ^ { \prime } ) q _ { j ^ { \prime } } ( v _ { 1 : L - h } )$ and $\mathbf { x } _ { 1 : L } : = u ( n _ { 1 : h } ) q _ { j ^ { \prime } } ( v _ { 1 : L - h } )$ Assign an arbitrary deterministic hybrid model M of architecture ST to the task of Segmented Retrieval. Then

$$
M ( \mathbf { x ^ { \prime } } _ { 1 : L } ) = n _ { j ^ { \prime } } ^ { \prime } , \quad M ( \mathbf { x } _ { 1 : L } ) = n _ { j ^ { \prime } } .
$$

Therefore, we must have $S _ { h } \left( u \left( n _ { 1 : h } ^ { \prime } \right) \right) \ne S _ { h } \left( u \left( n _ { 1 : h } \right) \right)$ . Otherwise, by Theorem 2, we have $M ( \mathbf { x } ^ { \prime } _ { 1 : L } ) =$ $n _ { j ^ { \prime } } ^ { \prime } = M ( \mathbf { x } _ { 1 : L } ) = n _ { j ^ { \prime } }$ , which contradicts $n _ { j ^ { \prime } } ^ { \prime } \neq n _ { j ^ { \prime } }$ . Hence, we conclude the map

$$
n _ { 1 : h } \mapsto S _ { h } \left( u \left( n _ { 1 : h } \right) \right) , \quad \forall n _ { 1 : h } \in \mathcal { N } ^ { h } ,
$$

is injective, and thus implies

$$
| S | \geq | { \mathcal { N } } | ^ { h } .
$$

For the corollary, if $L \ge 3 \lambda + 1$ and $W \leq \lambda + 1$ , then $L - W \geq 2 \lambda$ , hence $h : = \operatorname* { m i n } ( \lambda , ( L - W ) _ { + } ) = \lambda$ Thus,

$$
\log | S | \geq \Omega ( \lambda \log | \mathcal { N } | ) .
$$

## F.2 Constructions

## Construction for Exact Copy

We first outline the high-level idea behind the construction. At each position t, we want to examine whether this position can be an ending position of the desired exact copy. This task comes down to finding the number of equalities up to each position t. Let the number of consecutive equalities up to each position t be denoted by $H _ { t }$ . The SSM layer is set to calculate $H _ { t } = \mathbf { 1 } _ { \{ t \geq l + 1 , \mathbf { x } _ { t } = \mathbf { x } _ { t - l } \} } ( H _ { t - 1 } + 1 )$ . In other words, if the current position satisfies ${ \bf x } _ { t } = { \bf x } _ { t - l } ,$ then we add 1 to the counter so $\tilde { H _ { t } } = H _ { t - 1 } + 1 ;$ ; otherwise, we reset the counter. The Transformer layer first extracts $\mathbf { x } _ { t - l }$ at each position t. Lastly, we need another SSM layer to check if $H _ { t } = ( c - 1 ) l$ , that is, we need SSM to output $y _ { t } = \mathbf { 1 } _ { \{ H _ { t } = ( c - 1 ) l \} } \lor \mathbf { 1 } _ { \{ y _ { t - 1 } = 1 \} }$ . Since for any two variables $x , y \in \{ 0 , 1 \}$ we have that $x \vee y = ( 1 - y ) x + y .$ it follows that

![](images/fd348a16dd77117953c732be05569ef9ffdda04fae1b5e0334c11fcc753d3a9c.jpg)  
Figure 14: Structure of Exact Copy. At each position $t > l ,$ the TF layer retrieves $\mathbf { x } _ { t - l }$ . The first SSM layer counts consecutive matches ${ \bf x } _ { t } = { \bf x } _ { t - l }$ in $H _ { t }$ , while the second sets $y _ { t } = 1$ once $H _ { t } = ( c - 1 ) l$ and retains this value thereafter. Here, abc repeats three times consecutively, giving $H _ { 1 0 } = 6$ and $y _ { L } = 1$ despite subsequent mismatches.

$$
y _ { t } = \left( 1 - y _ { t - 1 } \right) \mathbf { 1 } _ { \left\{ H _ { t } = \left( c - 1 \right) l \right\} } + y _ { t - 1 } = \bigl ( 1 - \mathbf { 1 } _ { \left\{ H _ { t } = \left( c - 1 \right) l \right\} } \bigr ) y _ { t - 1 } + \mathbf { 1 } _ { \left\{ H _ { t } = \left( c - 1 \right) l \right\} } .
$$

We now present the detailed hybrid model construction.

Theorem 8 (Exact Copy Construction). There exists a model M ofarchitecture TSS with window $W \le { \cal { O } } ( l )$ and state log $| S | \leq O ( \log | \mathcal { V } | + \log ( c l ) )$ that performs Exact Copy (Definition 3).

Proof. We provide the various components of the TS architecture in distinct sections.

Token Embeddings. We define $\psi ^ { \prime } : \mathcal { M }  \{ \pm 1 \} ^ { \log | \mathcal { M } | }$ to be the binary encoding of the vocabulary in M.

Positional Encoding. We define $\phi ^ { \prime } : [ L ]  \{ \pm 1 \} ^ { \log | L | }$ as the binary encoding of numbers in $[ L ] . \ \mathrm { A f } -$

ter adding the positional encoding, the input context has the following form:

$$
\mathrm { T F i n p u t } ( \vec { \mathbf { x } } ) : = \left( \begin{array} { c c c c } { \psi ^ { \prime } ( \mathbf { x } _ { 1 } ) } & { \cdot \cdot \cdot } & { \psi ^ { \prime } ( \mathbf { x } _ { L - 1 } ) } & { \psi ^ { \prime } ( \mathbf { x } _ { L } ) } \\ { \mathbf { 0 } } & { \cdot \cdot \cdot } & { \mathbf { 0 } } & { \mathbf { 0 } } \\ { \phi ^ { \prime } ( 1 ) } & { \cdot \cdot \cdot } & { \phi ^ { \prime } ( L - 1 ) } & { \phi ^ { \prime } ( L ) } \\ { \mathbf { 0 } } & { \cdot \cdot \cdot } & { \mathbf { 0 } } & { \mathbf { 0 } } \\ { \mathbf { 1 } } & { \cdot \cdot \cdot } & { \mathbf { 1 } } & { \mathbf { 1 } } \end{array} \right)
$$

Transformer Layer. As outlined in the high-level idea above, we need to extract the token $\mathbf { x } _ { t - l }$ at each position t. The required sliding window size is l + 1. We add a positional bias $\mathbf { B } _ { t - l }$ into the attention:

$$
A t t n _ { j } ( T F - i n p u t ( \vec { \mathbf { x } } ) ) : = \sum _ { i = \operatorname* { m a x } ( 1 , j - l ) } ^ { j } \frac { \exp ( ( \mathbf { B } _ { j - l } ) _ { i } ) } { \sum _ { i = \operatorname* { m a x } ( 1 , j - l ) } ^ { j } \exp ( ( \mathbf { B } _ { j - l } ) _ { i } ) } \psi ^ { \prime } ( \mathbf { x } _ { i } ) = \psi ^ { \prime } ( \mathbf { x } _ { j - l } ) ,
$$

where $( \mathbf { B } _ { j - l } ) _ { i } = 0 , i = j - l , ( \mathbf { B } _ { j - l } ) = - \infty , \forall i \neq j - l .$ That is, we set $\mathbf { W } _ { q } = \mathbf { W } _ { k } = 0$ and ${ \bf W } _ { v } ( T F -$ $i n p u t ( \vec { \bf x } ) ) = ( \psi ^ { \prime } ( { \bf x } _ { 1 } ) , \cdot \cdot \cdot , \psi ^ { \prime } ( { \bf x } _ { L } ) ) ^ { \top }$ . Therefore, the output of the transformer layer is as follows:

$$
\begin{array} { r } { \left( \begin{array} { c c c c } { \psi ^ { \prime } ( \mathbf { x } _ { 1 } ) } & { \ldots } & { \psi ^ { \prime } ( \mathbf { x } _ { L - 1 } ) } & { \psi ^ { \prime } ( \mathbf { x } _ { L } ) } \\ { \mathbf { 0 } } & { \ldots } & { \psi ^ { \prime } ( \mathbf { x } _ { L - 1 - l } ) } & { \psi ^ { \prime } ( \mathbf { x } _ { L - l } ) } \\ { \phi ^ { \prime } ( 1 ) } & { \ldots } & { \phi ^ { \prime } ( L - 1 ) } & { \phi ^ { \prime } ( L ) } \\ { \mathbf { 0 } } & { \ldots } & { \mathbf { 0 } } & { \mathbf { 0 } } \\ { \mathbf { 1 } } & { \ldots } & { \mathbf { 1 } } & { \mathbf { 1 } } \end{array} \right) } \end{array}
$$

Here we set that $\forall j \leq l , A t t n _ { j } ( T F - i n p u t ( \vec { \bf x } ) ) = { \bf 0 } .$

SSM Layers. As outlined in the high-level idea of the SSM layer construction, we need two SSM layers to solve the task. Moreover, the first SSM layer is a two-gated SSM layer. The update rule becomes:

$$
H _ { t } = ( I - \Delta _ { A } ( \mathbf { x } _ { t } ) W _ { A } ) H _ { t - 1 } + \Delta _ { B } ( \mathbf { x _ { t } } ) W _ { B } \mathbf { x } _ { t } ,
$$

where $\Delta _ { B } ( { \bf x } _ { t } )$ is a scalar depending on $\mathbf { x } _ { t }$ but different from $\Delta _ { A } ( { \bf x } _ { t } )$   
We denote the number of consecutive equalities up to position t as $H _ { t }$ . Rigorously, $H _ { t } = \operatorname* { m a x } ( \{ 0 \} \cup \{ h \in [ t ] : e _ { t - h + 1 } = \cdot \cdot \cdot = e _ { t } :$   
where $e _ { t } : = \mathbf { 1 } _ { \left\{ t \geq l + 1 , \psi ^ { \prime } ( \mathbf { x } _ { t } ) = \psi ^ { \prime } ( \mathbf { x } _ { t - l } ) \right\} }$   
For the first SSM layer, our goal is $\begin{array} { r } { \dot { H } _ { t } = \mathbf { 1 } _ { \left\{ t \geq l + 1 , \psi ^ { \prime } ( \mathbf { x } _ { t } ) = \psi ^ { \prime } ( \mathbf { x } _ { t - l } ) \right\} } ( H _ { t - 1 } + 1 ) } \end{array}$ , whence we assign

$$
\Delta _ { A } ( \begin{array} { c } { \psi ^ { \prime } ( \mathbf { x } _ { t } ) } \\ { \psi ^ { \prime } ( \mathbf { x } _ { t - l } ) } \\ { \phi ^ { \prime } ( t ) } \\ { \mathbf { 0 } } \\ { \mathbf { 1 } } \end{array} ) ) = 1 - \mathbf { 1 } _ { \{ t \geq l + 1 , \psi ^ { \prime } ( \mathbf { x } _ { t } ) = \psi ^ { \prime } ( \mathbf { x } _ { t - l } ) \} } , \quad \Delta _ { B } ( \left( \begin{array} { c } { \psi ^ { \prime } ( \mathbf { x } _ { t } ) } \\ { \psi ^ { \prime } ( \mathbf { x } _ { t - l } ) } \\ { \phi ^ { \prime } ( t ) } \\ { \mathbf { 0 } } \\ { \mathbf { 1 } } \end{array} \right) ) = \mathbf { 1 } _ { \{ t \geq l + 1 , \psi ^ { \prime } ( \mathbf { x } _ { t } ) = \psi ^ { \prime } ( \mathbf { x } _ { t - l } ) \} } ,
$$

$$
\begin{array} { r } { { \bf W } _ { A } = { \bf W } _ { C } = I , \quad { \bf W } _ { B } ( \left( \begin{array} { c } { \psi ^ { \prime } ( { \bf x } _ { t } ) } \\ { \psi ^ { \prime } ( { \bf x } _ { t - l } ) } \\ { \phi ^ { \prime } ( t ) } \\ { { \bf 0 } } \\ { { \bf 1 } } \end{array} \right) ) = 1 . } \end{array}
$$

Therefore, the output of the first SSM layer is as follows.

$$
\begin{array} { r l } { \left( \begin{array} { c c c c } { \psi ^ { \prime } ( \mathbf { x } _ { 1 } ) } & { \ldots } & { \psi ^ { \prime } ( \mathbf { x } _ { L - 1 } ) } & { \psi ^ { \prime } ( \mathbf { x } _ { L } ) } \\ { \mathbf { 0 } } & { \ldots } & { \psi ^ { \prime } ( \mathbf { x } _ { L - 1 - l } ) } & { \psi ^ { \prime } ( \mathbf { x } _ { L - l } ) } \\ { \phi ^ { \prime } ( 1 ) } & { \ldots } & { \phi ^ { \prime } ( L - 1 ) } & { \phi ^ { \prime } ( L ) } \\ { H _ { 1 } } & { \ldots } & { H _ { L - 1 } } & { H _ { L } } \\ { \mathbf { 1 } } & { \ldots } & { \mathbf { 1 } } & { \mathbf { 1 } } \end{array} \right) } \end{array}
$$

The second SSM layer uses the standard update rule, i.e. $y _ { t } = ( I - \Delta ( \mathbf { x } _ { t } ) W _ { A } ) y _ { t - 1 } + \Delta ( \mathbf { x } _ { t } ) W _ { B } \mathbf { x } _ { t }$ . As outlined in the high-level idea, the second layer is designed to compute

$$
y _ { t } = ( 1 - \mathbf { 1 } _ { \{ H _ { t } = ( c - 1 ) l \} } ) y _ { t - 1 } + \mathbf { 1 } _ { \{ H _ { t } = ( c - 1 ) l \} } .
$$

This final expression is attained with the following weights for the second SSM:

$$
\Delta ( \begin{array} { c } { \psi ^ { \prime } ( \mathbf { x } _ { t } ) } \\ { \psi ^ { \prime } ( \mathbf { x } _ { t - l } ) } \\ { \phi ^ { \prime } ( t ) } \\ { H _ { t } } \\ { \mathbf { 1 } } \end{array} ) ) = \mathbf { 1 } _ { \{ H _ { t } = ( c - 1 ) l \} } ,
$$

$$
\begin{array} { r } { { \bf W } _ { A } = { \bf W } _ { C } = I , \quad { \bf W } _ { B } ( \left( \begin{array} { c } { \psi ^ { \prime } ( { \bf x } _ { t } ) } \\ { \psi ^ { \prime } ( { \bf x } _ { t - l } ) } \\ { \phi ^ { \prime } ( t ) } \\ { H _ { t } } \\ { { \bf 1 } } \end{array} \right) ) = 1 . } \end{array}
$$

## Construction for Segmented Retrieval

We first outline the high-level idea behind the construction.

To solve the Segmented Retrieval task, we can first use a Transformer layer to extract the token $\mathbf { x } _ { t - \lambda }$ at each position t by adding a positional bias $\mathbf { B } _ { t - \lambda }$ . Then, at each position t, we need to check if $t \in \mathbf { T } : =$ $\{ t _ { 1 } , \cdot \cdot \cdot , t _ { k - 1 } \}$ and $\mathbf { x } _ { t - \lambda } \in \mathcal { N }$ are both true. If both conditions are satisfied, then we set the SSM state as $H _ { t } = \mathbf { x } _ { t - \lambda } ;$ ; otherwise, $H _ { t } = H _ { t - 1 }$ , maintaining the previous state. This can be achieved through a standard SSM layer, as we set

$$
H _ { t } = \big ( I - \mathbf { 1 } _ { \{ ( t \in \mathbf { T } ) \wedge ( \mathbf { x } _ { t - \lambda } \in \mathcal { N } ) \} } \big ) H _ { t - 1 } + \mathbf { 1 } _ { \{ ( t \in \mathbf { T } ) \wedge ( \mathbf { x } _ { t - \lambda } \in \mathcal { N } ) \} } \mathbf { x } _ { t - \lambda } .
$$

Theorem 9 (Segmented Retrieval Construction). There exists a model M of architecture TS with window $W \le { \cal { O } } ( \lambda )$ and state log $| s | \leq O ( \log | \mathcal { V } | )$ that performs Segmented Retrieval (Definition 4).

Proof. We provide the various components of the TS architecture in distinct sections.

Token Embeddings. We define $\psi ^ { \prime } : \mathcal { M }  \{ \pm 1 \} ^ { \log | \mathcal { M } | }$ to be the binary encoding of the vocabulary in M.

![](images/c6a1c3b99763cca91469b66f7050b976af53daff5f18c08e0e3a9dc90562f237.jpg)  
Figure 15: Structure of Segmented Retrieval. At each internal boundary $t _ { j } ,$ the TF layer retrieves $v _ { j } =$ $\mathbf { x } _ { t _ { j } - \lambda }$ . The SSM layer retains the most recent retrieved token belonging to $\mathcal { N }$ and outputs it at position $L .$ Here, the retrieved tokens are $A , u , B , v ,$ , so the output is $B .$

Positional Encoding. We define $\phi ^ { \prime } : [ L ]  \{ \pm 1 \} ^ { \log | L | }$ as the binary encoding of numbers in $[ L ] . \ \mathrm { A f } .$ ter adding the positional encoding, the input context has the following form:

$$
\mathrm { T F i n p u t } ( \vec { \mathbf { x } } ) : = \left( \begin{array} { c c c c } { \psi ^ { \prime } ( \mathbf { x } _ { 1 } ) } & { \cdot \cdot \cdot } & { \psi ^ { \prime } ( \mathbf { x } _ { L - 1 } ) } & { \psi ^ { \prime } ( \mathbf { x } _ { L } ) } \\ { \mathbf { 0 } } & { \cdot \cdot \cdot } & { \mathbf { 0 } } & { \mathbf { 0 } } \\ { \phi ^ { \prime } ( 1 ) } & { \cdot \cdot \cdot } & { \phi ^ { \prime } ( L - 1 ) } & { \phi ^ { \prime } ( L ) } \\ { \mathbf { 0 } } & { \cdot \cdot \cdot } & { \mathbf { 0 } } & { \mathbf { 0 } } \\ { \mathbf { 1 } } & { \cdot \cdot \cdot } & { \mathbf { 1 } } & { \mathbf { 1 } } \end{array} \right)
$$

Transformer Layer. As outlined in the high-level idea above, we need to extract the token $\mathbf { x } _ { t - \lambda }$ at each position $t ,$ from which the sliding window size is $\lambda + 1$ , and we add the positional bias $\mathbf { B } _ { t - \lambda }$ into the attention:

$$
A t t n _ { j } ( T F - i n p u t ( \vec { \mathbf { x } } ) ) : = \sum _ { i = \operatorname* { m a x } ( 1 , j - \lambda ) } ^ { j } \frac { \exp ( ( \mathbf { B } _ { j - \lambda } ) _ { i } ) } { \sum _ { i = \operatorname* { m a x } ( 1 , j - \lambda ) } ^ { j } \exp ( ( \mathbf { B } _ { j - \lambda } ) _ { i } ) } \psi ^ { \prime } ( \mathbf { x } _ { i } ) = \psi ^ { \prime } ( \mathbf { x } _ { j - \lambda } ) ,
$$

where $( \mathbf { B } _ { j - \lambda } ) _ { i } = 0 , i = j - \lambda , ( \mathbf { B } _ { j - \lambda } ) = - \infty , \forall i \neq j - \lambda$ . That is to say we set $\mathbf { W } _ { q } = \mathbf { W } _ { k } = 0$ and $\mathbf { W } _ { v } ( T F - i n p u t ( { \vec { \mathbf { x } } } ) ) = ( \psi ^ { \prime } ( \mathbf { x } _ { 1 } { \bar { \mathbf { ) } } } , \cdots , \psi ^ { \prime } ( \mathbf { x } _ { L } ) ) ^ { \top }$ . Therefore, the output of the transformer layer is as follows.

$$
\begin{array} { r l r } { \left( \psi ^ { \prime } ( \mathbf { x } _ { 1 } ) \mathrm { ~  ~ \cdots ~ } } & { \psi ^ { \prime } ( \mathbf { x } _ { L - 1 } ) \mathrm { ~  ~ \psi ~ } \psi ^ { \prime } ( \mathbf { x } _ { L } ) \right) } & { } \\ { \mathbf { 0 } } & { \cdots \mathrm { ~  ~ \psi ~ } \psi ^ { \prime } ( \mathbf { x } _ { L - 1 - \lambda } ) } & { \psi ^ { \prime } ( \mathbf { x } _ { L - \lambda } ) } \\ { \phi ^ { \prime } ( 1 ) } & { \cdots \mathrm { ~  ~ \psi ~ } \phi ^ { \prime } ( L - 1 ) } & { \phi ^ { \prime } ( L ) } \\ { \mathbf { 0 } } & { \cdots \mathrm { ~  ~ \psi ~ } \mathbf { 0 } } & { \mathbf { 0 } } \\ { \mathbf { 1 } } & { \cdots \mathrm { ~  ~ \psi ~ } \mathbf { 1 } } & { \mathbf { 1 } } \end{array}
$$

Here we set that $\forall j \leq \lambda , A t t n _ { j } ( T F - i n p u t ( \vec { \bf x } ) ) = { \bf 0 }$

SSM Layer. As outlined in the high-level idea of the SSM layer construction, we only need one SSM layer to solve the task. At each position $t ,$ we denote the largest index of the set $\{ t _ { j } - \lambda : j \in [ k - 1 ] , \mathbf { x } _ { t _ { j } - \lambda } \in$ $\mathcal { N } , 1 \leq t _ { j } - \lambda \leq t \}$ as $t ^ { \star }$ , and define $H _ { t } : = \mathbf { x } _ { t ^ { \star } }$ . Then, as stated in the high-level idea, the update rule for $H _ { t }$ is

$$
H _ { t } = \big ( I - \mathbf { 1 } _ { \{ ( t \in \mathbf { T } ) \wedge ( \mathbf { x } _ { t - \lambda } \in \mathcal { N } ) \} } \big ) H _ { t - 1 } + \mathbf { 1 } _ { \{ ( t \in \mathbf { T } ) \wedge ( \mathbf { x } _ { t - \lambda } \in \mathcal { N } ) \} } \mathbf { x } _ { t - \lambda } ,
$$

This is achieved by setting

$$
\mathbf { W } _ { A } = \mathbf { W } _ { C } = I , \quad \mathbf { W } _ { B } ( \left( \begin{array} { c } { \psi ^ { \prime } ( \mathbf { x } _ { t } ) } \\ { \psi ^ { \prime } ( \mathbf { x } _ { t - \lambda } ) } \\ { \phi ^ { \prime } ( t ) } \\ { \mathbf { 0 } } \\ { 1 } \end{array} \right) ) = \mathbf { x } _ { t - \lambda } , \quad \Delta ( \left( \begin{array} { c } { \psi ^ { \prime } ( \mathbf { x } _ { t } ) } \\ { \psi ^ { \prime } ( \mathbf { x } _ { t - \lambda } ) } \\ { \phi ^ { \prime } ( t ) } \\ { \mathbf { 0 } } \\ { 1 } \end{array} \right) ) = \mathbf { 1 } _ { \left\{ ( t \in \mathbf { T } ) \wedge ( \mathbf { x } _ { t - \lambda } \in \mathcal { N } ) \right\} } .
$$

## F.3 SSMs with a Convolutional Front-End

A Mamba block applies a causal depthwise convolution before its selective SSM. Each such layer can therefore read a few recent inputs directly, in addition to its recurrent state. This subsection makes the local access explicit, relates it to the S-layers of Section 2, and quantifies its effect on the lower bounds of Section 2.2.

Definition 8 $( \mathrm { S } _ { \kappa } \mathrm { - l a y e r } )$ . Let $\kappa \geq 0$ . A layer $F : ( \{ 0 , 1 \} ^ { P } ) ^ { L } \to ( \{ 0 , 1 \} ^ { P } ) ^ { L }$ is an $\mathrm { S } _ { \kappa }$ -layer with recurrent state space $s$ if there exist an initial state $S _ { 0 } ~ \in ~ S$ and maps $\mu : { \mathcal { S } } \times ( \{ 0 , 1 \} ^ { P } ) ^ { \kappa + 1 } \to { \mathcal { S } }$ and $\psi : \mathcal { S } \times$ $( \{ 0 , 1 \} ^ { \cdot _ { P } } ) ^ { \kappa + 1 }  \{ 0 , 1 \} ^ { P }$ such that

$$
S _ { t } = \mu \bigl ( S _ { t - 1 } , { \bf x } _ { t - \kappa : t } \bigr ) , \qquad F ( \vec { \bf x } ) _ { t } = \psi \bigl ( S _ { t } , { \bf x } _ { t - \kappa : t } \bigr ) ,
$$

where $\mathbf { x } _ { i }$ for $i \leq 0$ is a fixed padding symbol ⋄ whose encoding differs from that of every token. We call $\mathbf { x } _ { t - \kappa : t }$ the local window and $\log _ { 2 } | S |$ the recurrent-state complexity; the local window is not charged to $s .$

Mamba. With finite-precision arithmetic, a Mamba block (input projection, causal depthwise convolution of width $\kappa + 1$ , selective SSM, gating, output projection, and residual connection) is an $\mathrm { S } _ { \kappa }$ -layer with $\kappa = d _ { \mathrm { c o n v } } - 1$ , whose recurrent state is the SSM state: both the state update and the output at position t depend only on the previous state and on $\mathbf { x } _ { t - \kappa : t }$ . Lower bounds for $\mathrm { S } _ { \kappa } – \mathrm { l a y e r s }$ therefore apply to Mamba. The converse does not hold. Since $\mu$ and $\psi$ are arbitrary, a construction in the $\mathrm { S } _ { \kappa }$ class does not by itself yield parameters of a Mamba block.

Relation to the S-layers of Section 2. An S-layer of Section 2 is an S<sub>0</sub>-layer with the same state space (take $\psi ( S , \mathbf { x } _ { t } ) = r ( S ) )$ , and every S -layer is an $\mathrm { S } _ { \kappa }$ -layer with the same state space. The converse fails at equal budget, because $\psi$ may read the current input. For instance, with $| { \cal S } | = 1$ and $\psi ( S , \mathbf { x } _ { t } ) = \mathbf { x } _ { t }$ , an S -layer computes the identity with zero recurrent bits, whereas an S-layer of Section 2 needs $P$ bits of state to do so. Two facts relate the classes.

(i) Simulation. An $\mathrm { S } _ { \kappa }$ -layer with recurrent state space $s$ is an S-layer of Section 2 with state space $\mathcal { S } \times ( \{ 0 , 1 \} ^ { P } ) ^ { \kappa + 1 }$ : store $\widetilde { S } _ { t } = ( S _ { t } , \mathbf { x } _ { t - \kappa : t } )$ , update the window as a shift register, and set $r ( \widetilde { S } _ { t } ) =$ $\psi ( S _ { t } , \mathbf { x } _ { t - \kappa : t } )$ . Its bit complexity is at most $\log _ { 2 } { | S | } + ( \kappa + 1 ) P$

(ii) Configuration at a cut. For every $\tau ,$ the pair $\left( S _ { \tau } , \mathbf { x } _ { \tau - \kappa + 1 : \tau } \right)$ , of at most $\log _ { 2 } { \left| \mathcal { S } \right| } + \kappa P$ bits, determines $F ( \vec { \bf x } ) _ { t }$ for all $t > \tau$ given $\mathbf { x } _ { \tau + 1 : t }$ . This is the case $D _ { \mathrm { S } } = 1$ of Lemma 17 below.

By (ii), the proof of Theorem 2 applies to $\mathrm { S } _ { \kappa } \mathrm { T }$ models with $| S | 2 ^ { \kappa P }$ in place of |S|. Thus, Theorems 6 and 7 continue to hold with $\log _ { 2 } { | S | } + \kappa P$ in place of $\log _ { 2 } | S |$ . By (i) applied layerwise, a hybrid model with profile $A$ whose recurrent layers are $\mathrm { S } _ { \kappa }$ -layers with at most B recurrent bits is a model in $\mathcal { M } _ { A } ( B + ( \kappa + 1 ) P , P , W )$ in the sense of Section 2. Since

$$
B + ( \kappa + 1 ) P \leq ( \kappa + 2 ) \operatorname* { m a x } \{ B , P \} ,
$$

the conclusion of part 2 of Theorem 1 continues to hold for such models, with $c _ { \tau } m / ( \kappa + 2 )$ in place of $c _ { \tau } m$ The constructions need no change, because every S-layer is an $\mathrm { S } _ { \kappa } { \mathrm { - } } \mathrm { l a y e r }$ . The constant-factor separation in part 3 of Theorem 1 does not follow from this conversion.

For the natural tasks of Section 2.2, charging the window to the state is unnecessary. The following lemma and proposition show that each $\mathrm { S } _ { \kappa }$ -layer reduces the lower bounds by at most κ tokens, and that the resulting threshold on the task distance is tight within the $S _ { \kappa }$ abstraction.

Lemma 17 (Cut through a convolutional stack). Let $M = \mathrm { T } \circ F _ { D _ { \mathrm { S } } } \circ \cdot \cdot \cdot \circ F _ { 1 }$ , where each $F _ { i }$ is an $\mathrm { S } _ { \kappa }$ -layer with recurrent state space $\boldsymbol { S } ^ { ( i ) }$ and states $S _ { t } ^ { ( i ) }$ , and T is a T-layer with window $W ;$ write $M ( { \vec { \bf x } } )$ for the output at position L. Let $D _ { \mathrm { { S } } } \kappa < \tau \leq L - W$ and $\tau _ { i } : = \tau - ( D _ { \mathrm { S } } - i ) \kappa f o r i \in [ D _ { \mathrm { S } } ]$ . Let u, u<sup>′</sup> be prefixes of length τ that agree on positions $\tau - D _ { \mathrm { S } } \kappa + 1 , \ldots , \tau$ τ and satisfy $S _ { \tau _ { i } } ^ { ( i ) } ( \mathbf { u } ) = S _ { \tau _ { i } } ^ { ( i ) } ( \mathbf { u } ^ { \prime } )$ for all $i \in [ D _ { \mathrm { S } } ]$ . Then $M ( \mathbf { u q } ) = M ( \mathbf { u ^ { \prime } q } )$ for every suffix q of length $L - \tau .$

Proof. Write $\mathbf { o } ^ { ( 0 ) } = \vec { \mathbf { x } }$ and $\mathbf { o } ^ { ( i ) } = F _ { i } ( \mathbf { o } ^ { ( i - 1 ) } )$ , and compare the inputs uq and u<sup>′</sup>q. We show by induction on i that $S _ { t } ^ { ( i ) }$ and $\mathbf { o } _ { t } ^ { ( i ) }$ coincide on the two inputs for all $t > \tau _ { i }$ . For $i = 1$ and $t > \tau _ { 1 }$ , the window $\mathbf { x } _ { t - \kappa : t }$ starts at position $t - \kappa \geq \tau _ { 1 } - \kappa + 1 = \tau - D _ { \mathrm { S } } \kappa + 1$ , so it lies in the common part of the prefixes or in the common suffix. Together with $S _ { \tau _ { 1 } } ^ { ( 1 ) } ( \mathbf { u } ) = S _ { \tau _ { 1 } } ^ { ( 1 ) } ( \mathbf { u } ^ { \prime } )$ , induction on t gives the claim. For $i \geq 2$ and $t > \tau _ { i } .$ the window $\mathbf { o } _ { t - \kappa : t } ^ { ( i - 1 ) }$ starts at position $t - \kappa > \tau _ { i } - \kappa = \tau _ { i - 1 }$ , where the two inputs agree by the induction hypothesis. Together with $S _ { \tau _ { i } } ^ { ( i ) } ( \mathbf { u } ) = S _ { \tau _ { i } } ^ { ( i ) } ( \mathbf { u } ^ { \prime } )$ , the claim follows as before. Finally, $\tau _ { D _ { \mathrm { S } } } = \tau$ , and T reads only positions $L - W + 1 , \ldots , L ,$ , all larger than τ. Hence the outputs coincide. □

Theorem 10 (Natural tasks with convolutional recurrent layers). Let $D _ { \mathrm { S } } \geq 1$ , let h be as in Theorem 6 (resp. Theorem 7), and set

$$
h _ { D _ { \mathrm { S } } } : = ( h - D _ { \mathrm { S } } \kappa ) _ { + } .
$$

(a) Under the assumptions ofTheorem 6 (resp. Theorem 7), every deterministic model M as in Lemma 17 that performs Exact Copy (resp. Segmented Retrieval) correctly on all inputs satisfies

$$
\sum _ { i = 1 } ^ { D _ { \mathrm { S } } } \log _ { 2 } | S ^ { ( i ) } | \ \geq \ h _ { D _ { \mathrm { S } } } \log _ { 2 } ( | \mathcal { V } | - 2 )
$$

(resp. the same with $\log _ { 2 } | \mathcal { N } |$ in place of $\log _ { 2 } ( | \mathcal { V } | - 2 ) )$

(b) Conversely, suppose $l \le D _ { \mathrm { S } } \kappa ( r e s p . \lambda \le D _ { \mathrm { S } } \kappa )$ and $P \geq 2 \lceil \log _ { 2 } ( | \mathcal { V } | + 1 ) \rceil + 2$ . Then there is a model M as in Lemma 17, with T the identity $( W = 1 )$ , in which $F _ { 1 } , \ldots , F _ { D _ { \mathrm { S } } - 1 }$ have a single recurrent state and $F _ { D _ { \mathrm { S } } }$ has $O ( \log ( c l ) )$ (resp. $O ( \log | \mathcal { N } | ) ,$ recurrent bits, that performs Exact Copy (resp. Segmented Retrieval) correctly on all inputs.

In particular, with $W = 1$ (so that $h = l ,$ , resp. $h = \lambda ) ,$ , the bound in (a) is nonzero exactly when the task distance exceeds $D _ { \mathrm { S } } \kappa$ , and otherwise (b) gives a solution with small recurrent state.

Proof. (a). If $h _ { D _ { \mathrm { S } } } = 0$ there is nothing to prove, so assume $h _ { D _ { \mathrm { S } } } \geq 1$ . Then $h = h _ { D _ { \mathrm { S } } } + D _ { \mathrm { S } } \kappa .$ , and we use the cut $\tau = h$ , which satisfies $D _ { \mathrm { { S } } } \kappa < \tau \leq L - W$ by the definition of h. By Lemma 17, if a family of prefixes of length h shares its last $D _ { \mathrm { S } } \kappa$ positions and any two of its members are mapped to different outputs by some common suffix, then the map u $. \mapsto \left( S _ { \tau _ { i } } ^ { ( i ) } ( \mathbf { u } ) \right) _ { i \in [ D _ { \mathrm { S } } ] }$ is injective on the family. Its image has at most $\prod _ { i } | S ^ { ( i ) } |$ elements, so it suffices to exhibit such a family of size $( | \mathcal { V } | - 2 ) ^ { h _ { D _ { \mathrm { S } } } } \left( \mathrm { r e s p . } ~ | \mathcal { N } | ^ { h _ { D _ { \mathrm { S } } } } \right)$

For Exact Copy, let $\mathcal { A } , \alpha , g _ { 1 } , g _ { 2 }$ and $\mathbf { g } _ { 1 : L - c l }$ be as in the proof of Theorem 6, and write $h ^ { \prime } = h _ { D _ { \mathrm { S } } }$ . For $a \in \mathcal { A } ^ { h ^ { \prime } }$ , use the prefix $a \alpha ^ { D \mathrm { s } \kappa }$ and the suffix $\begin{array} { r } { \alpha ^ { l - h } \left( a \alpha ^ { l - h ^ { \prime } } \right) ^ { c - 1 } \mathbf { g } _ { 1 : L - c l } } \end{array}$ . For $a ^ { \prime } \in \mathcal { A } ^ { h ^ { \prime } }$ , the resulting input is

$$
a ^ { \prime } \alpha ^ { D _ { \mathrm { S } } \kappa } \cdot ( \mathrm { s u f f i x ~ o f } ~ a ) ~ = ~ a ^ { \prime } \alpha ^ { l - h ^ { \prime } } \big ( a \alpha ^ { l - h ^ { \prime } } \big ) ^ { c - 1 } { \bf g } _ { 1 : L - c l } ,
$$

which is the input analysed in the proof of Theorem 6 with h replaced by $h ^ { \prime } \in [ 1 , l ]$ . It therefore contains an exact copy if and only if $\boldsymbol { a } ^ { \prime } = \boldsymbol { a }$

For Segmented Retrieval, fix $\alpha \in { \mathcal { V } }$ and use, for $n \in \mathcal { N } ^ { h ^ { \prime } }$ , the prefix

$$
\big ( ( n _ { 1 } , 0 ) , \dots , ( n _ { h ^ { \prime } } , 0 ) , \underbrace { ( \alpha , 0 ) , \dots , ( \alpha , 0 ) } _ { D _ { \mathrm { S } } \kappa } \big ) .
$$

For $j \in [ h ^ { \prime } ]$ , append the suffix $q _ { j }$ from the proof of Theorem 7. Its only internal boundary is at position $j + \lambda > h$ , the segment-length conditions of Definition 4 hold because $L \ge 3 \lambda + 1$ and $j \le \lambda$ , and the required output is ${ \bf x } _ { j } = n _ { j }$ . Two distinct prefixes differ at some $j \in [ h ^ { \prime } ]$ , and $q _ { j }$ separates them.

(b). Encode each position by its token (or ⋄) and, for Segmented Retrieval, its boundary bit; two such encodings fit in $P$ bits. Let $j _ { 0 } = 0$ and $j _ { i } = \operatorname* { m i n } ( \delta , i \kappa )$ , where $\delta = l ( { \mathrm { r e s p . ~ } } \delta = \lambda )$ is the task distance. For $i < D _ { \mathrm { { S } } }$ , let $F _ { i }$ have a single recurrent state and output $\mathbf o _ { t } ^ { ( i ) } = ( \mathbf x _ { t } , \mathbf x _ { t - j _ { i } } )$ , with ⋄ for nonpositive indices. This is possible because $F _ { 1 }$ reads $\mathbf { x } _ { t - j _ { 1 } }$ directly $( j _ { 1 } \le \kappa )$ , and for $i \geq 2$ the window of $F _ { i }$ contains $\mathbf { o } _ { t } ^ { ( i - 1 ) }$ whose first entry is $\mathbf { x } _ { t } .$ , and $\mathbf { o } _ { t - ( j _ { i } - j _ { i - 1 } ) } ^ { ( i - 1 ) } ,$ whose second entry is $\mathbf { x } _ { t - j _ { i } }$ , since $0 \leq j _ { i } - j _ { i - 1 } \leq \kappa$ . Because $\delta \le D _ { \mathrm { S } } \kappa$ , we have $\begin{array} { r } { \delta - j _ { D _ { \mathrm { S } } - 1 } \le \kappa , } \end{array}$ so the window of $F _ { D _ { \mathrm { S } } }$ contains both $\mathbf { x } _ { t }$ and $\mathbf { x } _ { t - \delta }$ (for $D _ { \mathrm { S } } = 1$ , directly from $\mathbf { x } _ { t - \kappa : t } )$

For Exact Copy, $F _ { D _ { \mathrm { S } } }$ keeps $H _ { t } = \operatorname* { m i n } \{ ( c - 1 ) l , \ \mathbf { 1 } \{ \mathbf { x } _ { t } = \mathbf { x } _ { t - l } \} ( H _ { t - 1 } + 1 ) \}$ } and $y _ { t } = y _ { t - 1 } \vee \mathbf { 1 } \big \{ H _ { t } =$ $( c - 1 ) l \}$ , and outputs y . Since ⋄ differs from every token, ${ \bf x } _ { t } = { \bf x } _ { t - l }$ can hold only for $t \geq l + 1$ . By Definition 3, an exact copy exists if and only if ${ \bf x } _ { t } = { \bf x } _ { t - l }$ for $( c - 1 ) l$ consecutive positions t. So $y _ { L }$ is correct, and $2 ( ( c - 1 ) l + 1 )$ states suffice. For Segmented Retrieval, $F _ { D _ { \mathrm { S } } }$ stores the last value $\mathbf { x } _ { t - \lambda } \in \mathcal { N }$ read at a boundary position t, or ⊥ before the first such value, and outputs it. This uses $| \mathcal { N } | + 1$ states, and the value stored at position L is $\mathbf { x } _ { t _ { \lambda ^ { \star } } - \lambda }$ □