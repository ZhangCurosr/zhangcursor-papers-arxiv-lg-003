# Robust Decentralized Fairness Auditing

Sayan Biswas EPFL, Switzerland sayan.biswas@epfl.ch

Jade Garcia Bourree´ EPFL, Switzerland jade.garcia-bourree@epfl.ch

Palak EPFL, Switzerland palak.palak@epfl.ch

Anne-Marie Kermarrec EPFL, Switzerland anne-marie.kermarrec@epfl.ch

Martijn de Vos EPFL, Switzerland martijn.devos@epfl.ch

Abstract—Emerging legislation requires large language models (LLMs) to be audited to assess their compliance with regulatory standards, particularly with respect to fairness. Such black-box audits typically assume a single auditor with access to a large, representative set of queries. In practice, it can be difficult for an auditor to obtain such a query set, but multiple auditors can together cover the relevant demographic groups by auditing the LLM collaboratively with their individual query sets. However, relying on multiple auditors raises a fundamental trust problem, as they may act on behalf of the LLM provider to portray a misleading appearance of fairness, i.e., fairwashing. We propose AUDITOPUS, a novel approach for robust decentralized fairness auditing. In AUDITOPUS, auditing proceeds in rounds without a central server. In each round, every auditor issues a fixed number of queries to the LLM, and sends only cumulative statistics vectors of its query results to other auditors instead of sensitive queries in clear. The fairness of the audited LLM is then estimated by aggregating all the vectors together. We show both theoretically and empirically that even a single adversary in the network can steer this estimate by fabricating the vectors it sends, making an unfair LLM appear fair. To address this threat, AUDITOPUS relies on a defense mechanism where each honest auditor locally down-weights any auditor whose cumulative statistics vectors are statistically inconsistent with previous ones. We implement AUDITOPUS and compare it to robust aggregation baselines on two datasets with two pre-trained LLMs. Against an attacker that optimizes the vectors it sends to make the LLM appear fair, AUDITOPUS reduces audit error by up to 78% on average relative to no defense and at least 62% relative to the robust aggregation baselines. Even when 49% of auditors are adversarial, AUDITOPUS never lets a very unfair or moderately unfair LLM pass as fair.

Index Terms—fairness auditing, black-box auditing, large language models, decentralized auditing, fairwashing, robustness.

## I. INTRODUCTION

LLMs are increasingly used as classifiers in high-stakes applications, including content moderation [34], clinical decision support [44], hiring [4], financial risk assessment [25], and legal document analysis [27]. In these settings, an LLM’s decisions directly affect individuals, and systematic disparities in how demographic groups are treated by the LLM can cause real harm, ranging from lost economic opportunities (e.g., unjustified loan denials) to personal consequences (e.g., a missed medical diagnosis) [37]. Emerging legislation, such as the EU AI Act [18] and New York City’s Local Law 144 on automated employment decision tools [38], therefore requires such systems to be audited for compliance, in particular for fairness across demographic groups. However, these LLMs are typically proprietary and served through APIs, so auditors have access to neither the LLM’s weights nor its training data. They typically only have black-box access, i.e., they can only submit queries and observe the LLM’s decisions [11]. Fairness must therefore be audited in such black-box settings by querying the LLM and measuring disparities in its outputs [4].

In a black-box fairness audit, a single auditor, such as a regulator, assembles a set of queries, submits them to the LLM, and estimates a fairness metric from the LLM’s decisions. A standard choice, and the one we focus on in this work, is demographic parity (DP) [9], [16] that measures the difference between the rates at which two demographic groups receive a positive decision from the LLM. The reliability of this estimate heavily depends on the query set, which must meet two requirements. First, it must be large: fairness metrics compare decision rates across groups (e.g., male and female), and estimating them precisely requires many queries per group [31]. Second, it must be representative of the population the LLM serves in deployment; otherwise, the audit measures fairness on the wrong distribution and can over- or underestimate the disparity [14]. Meeting both requirements is difficult for a single auditor. Realistic inputs, such as patient records or job applications, are sensitive and protected by dataprotection regulations; public benchmarks are not necessarily representative; and every query to a commercial API has a cost [3]. In practice, the relevant audit queries are typically spread across many parties, such as hospitals, employers, or civil-society organizations, each holding only a small, possibly unrepresentative query set [41].

Individual auditors, however, can collaborate to effectively conduct a fairness audit of an LLM. By auditing the LLM collaboratively, auditors can estimate fairness over the union of their query sets, which is far larger and more representative than any single auditor’s, while sharing the cost of querying. This addresses both requirements. Sharing raw queries, however, is not an option due to their sensitivity. Fortunately, auditors need not send their queries at all. Group fairness metrics such as DP depend only on aggregate statistics, so each auditor can query the LLM with its own queries and send only these aggregated statistics to other auditors. We show an example of this setting in Figure 1 where N auditors audit the fairness of an LLM and send aggregated statistics.

![](images/96514d2e85c775f8cc99b8d26d2f40464e9c633d473f32895d687c82709639f7.jpg)  
Fig. 1. A collaborative fairness audit of a large language model (LLM).

Collaborative fairness audits, however, raise two challenges. First, the auditor query sets are heterogeneous in terms of size, groups, and predicted labels. Each query set reflects its own sub-population. Therefore, local estimates may widely differ from each other [14] and from the population-level metric, making aggregation of local fairness values inaccurate. Second, some auditors may be adversarial. A participant colluding with the LLM provider can send fabricated numbers that steer honest auditors toward a verdict of fairness, a form of fairwashing [1]. Such adversaries may have a stake in the audit outcome, as platform employees, contractors, or paid affiliates, or because they benefit, economically or ideologically, from the discrimination going undetected. These two challenges compound each other. Robust aggregation methods typically discard contributions that deviate from the majority, but under heterogeneity, honest auditors with unusual sub-populations also deviate, so discarding outliers also discards the data that makes the audit representative.

We propose AUDITOPUS, a novel approach for robust decentralized fairness auditing. Rather than aggregating all statistics in a single exchange, AUDITOPUS proceeds in multiple rounds. Each round, every auditor issues a fixed number of queries to the LLM, and sends only statistics vectors to a few other auditors, never the sensitive queries themselves. This round-based structure bounds how much any single auditor can contribute per round, thus limiting the effect an adversary can have in influencing the round-wise estimate. Moreover, it gives each auditor a history of every other auditor’s statistics vectors to check new vectors against. This enables AUDITOPUS to catch subtle fairwashing that a single round would miss (see Section II-B). Each auditor then estimates the LLM’s fairness by aggregating all the vectors it has received. To stress the importance of robustness, we introduce a strong adaptive adversary that attempts fairwashing by fabricating, every round, the statistics vectors that bring the estimated DP as close to 0 as possible. We design a new defense against this threat that is central to AUDITOPUS. Rather than trusting all auditors equally, each honest auditor locally down-weights any peer whose per-round reports are statistically inconsistent with its own past reports. An honest auditor, by contrast, samples uniformly from a fixed query set, so its increments remain consistent over time.

We implement and evaluate AUDITOPUS on two datasets against robust aggregation baselines. AUDITOPUS prevents fairwashing of very unfair and moderate LLM on both datasets, even when 49% of the auditors are adversarial, and reduces the error of the estimated DP by up to 78% on average relative to no defense and by at least 62% relative to the robust aggregation baselines. Near-fair LLMs remain the hardest to protect, since a small shift of the estimated DP suffices to flip their verdict, yet AUDITOPUS still at least halves this error.

Contributions. We make the following four contributions:

1) We introduce a novel round-based decentralized fairness auditing mechanism, named AUDITOPUS, in which auditors query the LLM with their own private queries and only exchange cumulative statistics vectors with their neighbors (Section III).

2) We introduce a strong adaptive adversary that fairwashes the LLM by optimizing its fabricated statistics vectors every round, and we theoretically characterize the conditions under which a single adversary can steer the estimate so that an unfair LLM appears fair (Section IV).

3) We design a defense that is central to AUDITOPUS: each honest auditor locally down-weights any auditor whose statistics vectors are statistically self-inconsistent (Section V). It requires no trusted party, and unlike robust aggregation techniques, it does not discard honest auditors whose query sets are atypical.

4) We evaluate AUDITOPUS on two real-world datasets against robust aggregation baselines (Section VI). AU-DITOPUS effectively neutralizes the threat from adaptive adversaries present in the network. With 23% adversaries in the network, the error of the estimated DP of AUDI-TOPUS stays between 0.02 and 0.05 as the number of honest auditors and the heterogeneity of their query sets vary, while it reaches up to 0.26 for robust aggregation baselines, at negligible communication and compute cost. Even with 49% of the auditors being adversarial, it never lets a very unfair or moderately unfair LLM pass as fair.

## II. BACKGROUND AND PROBLEM DESCRIPTION

We first introduce black-box and collaborative fairness auditing, and then formulate our research problem.

## A. Black-box Fairness Auditing

The goal of a fairness audit is to verify whether a deployed LLM treats different demographic groups equitably, for example, to assess its compliance with regulation. Such an audit involves three entities: (i) the platform, which hosts the LLM and uses it as a classifier; (ii) the users, who interact with the service offered by the platform; and (iii) the auditor, which conducts the audit to verify whether the LLM is fair across all users. The auditor can be, for example, a state regulator, a consulting firm, or a group of users. In a black-box audit, the auditor has no access to the LLM’s internals, such as its architecture, weights, or training data. Regulations such as the EU AI Act [18] do require platforms to disclose information about their systems, e.g., technical documentation or transparency reports. However, this information is selfreported and cannot be independently verified by the auditor, and it may be incomplete or inconsistent with the platform’s actual behavior [46], [20]. Following [20], we therefore treat the query-answer pairs collected by the auditor as the only trusted source of information, and disregard any public or platform-provided information about the system.

Fairness metric. In this work, we consider black-box audits targeting the fairness of an LLM f. Among all fairness metrics, we study Demographic Parity (DP) [9], [16] which is commonly used in the fairness evaluation literature. We consider an LLM used as a binary classifier. For an input $X ,$ it outputs a decision $f ( X ) \in \{ 0 , 1 \}$ , where $f ( X ) = 1$ denotes a positive decision $( e . g . ,$ a comment flagged as toxic). Each input belongs to one of two demographic groups A and $\mathsf { B } ,$ defined by a sensitive attribute such as religion or gender. A multi-class or multi-group classifier reduces to this setting by auditing one class against all others, as we do in Section VI. For an input X drawn from the population served by $f ,$ the DP of $f$ is defined as follows:

$$
\operatorname { D P } ( f ) = \mathbb { P } ( f ( X ) = 1 \mid X \in \mathsf { A } ) - \mathbb { P } ( f ( X ) = 1 \mid X \in \mathsf { B } ) .\tag{1}
$$

We focus on DP as it only depends on the LLM’s outputs, and can thus be estimated from query-answer pairs alone. Nevertheless, our approach only relies on the fact that the metric is a difference between conditional probabilities of the LLM’s output, and therefore extends to other group fairness metrics with the same structure. For instance, Equality of Opportunity [28] is obtained by additionally conditioning on $Y = 1$ in Eq. (1), and Equalized Odds [28] by considering both $Y = 1$ and $Y = 0 ,$ , provided that the auditor knows the ground-truth labels Y of its queries.

Fairness verdict. The auditor estimates $\operatorname { D P } ( f )$ from its query-answer pairs and outputs a verdict. The LLM is deemed fair if the estimate dDP lies in the fair band $[ - \tau , \tau ] , i . e .$ $| \widehat { \mathrm { D P } } | \le \tau$ for a tolerance τ set by the entity commissioning the audit, and unfair otherwise.

## B. Collaborative Fairness Auditing

As discussed in Section I, a single auditor rarely holds a query set that is large and representative of the population. We therefore consider N auditors, all honest for now, where each auditor i holds a private query set $\mathcal { D } _ { i }$ drawn from its own sub-population, such as the patients of a hospital or the applicants of an employer. In this collaborative setting, each auditor i queries the LLM with its own queries in $\mathcal { D } _ { i }$ and summarizes the resulting decisions into a locally aggregated statistics vector (LAS vector) $c _ { i } = ( a _ { i } , A _ { i } , b _ { i } , B _ { i } )$ , where $A _ { i }$ is the number of queries from group $\mathsf { A } , \ a _ { i }$ the number of those receiving a positive decision, and $B _ { i }$ and $b _ { i }$ their group-B counterparts. For any such vector $\boldsymbol { c } = ( a , A , b , B )$ ), the DP of the queries is:

$$
\mathrm { D P } ( c ) = \frac { a } { A } - \frac { b } { B } .\tag{2}
$$

![](images/dc5cae07d775466decd34f5388993e0ca61aa77256a043e5fa0e350369b58534.jpg)  
Fig. 2. Local DP estimations of $N = 2 0$ honest auditors on the Civil Comments dataset as data heterogeneity grows (lower α = more heterogeneous). The solid horizontal line indicates the true DP and the dotted lines indicate the 5th and 95th percentiles. Heterogeneity of private query sets makes local fairness estimates unreliable.

Applied to its own LAS vector, this gives auditor i its local estimate $\mathrm { D P } ( c _ { i } )$ . We also apply it below to sums of LAS vectors. Crucially, DP depends only on these LAS vectors, so auditors can collaborate by exchanging them rather than their queries.

Summing LAS vectors under heterogeneity. Heterogeneity makes local DP estimates unreliable. Figure 2 shows the local DP estimates of 20 honest auditors on the Civil Comments dataset, with QWEN2.5-7B-INSTRUCT as the LLM classifier, when the data is split across auditors with a Dirichlet distribution of concentration α (the smaller α, the more auditors differ in group composition and positive rate). With near-IID data $( \alpha = 1 0 0 )$ , all local estimates lie close to the true DP. $\boldsymbol { \mathrm { A s } } \ \alpha$ decreases, they spread out. The gap between the 5th and 95th percentiles widens from 0.14 at $\alpha = 1 0 0$ to 1.81 at $\alpha = 0 . 3$ , and many auditors even observe a disparity with the opposite sign. Some auditors may still land on the correct verdict, but only by chance. From its own query set alone, an auditor cannot tell whether its local estimate reflects the LLM’s disparity or merely the composition of its query set. Combining the local estimates, $e . g .$ , by their mean or median, does not solve this. It weights all auditors equally in both groups, although their proportions of each group differ. When group proportions differ across sub-populations, rates computed per sub-population and on the aggregated data can disagree, and even point in opposite directions (Simpson’s paradox) [5]. Instead, auditors can sum their LAS vectors into a single globally aggregated statistics vector (GAS vector) $C = \textstyle \sum _ { i } c _ { i }$ and compute a single estimate from it:

$$
\begin{array} { l } { \displaystyle \mathrm { D P } ( C ) = \frac { \sum _ { i } a _ { i } } { \sum _ { i } A _ { i } } - \frac { \sum _ { i } b _ { i } } { \sum _ { i } B _ { i } } } \\ { \displaystyle = \sum _ { i } \frac { A _ { i } } { \sum _ { j } A _ { j } } \frac { a _ { i } } { A _ { i } } - \sum _ { i } \frac { B _ { i } } { \sum _ { j } B _ { j } } \frac { b _ { i } } { B _ { i } } . } \end{array}\tag{3}
$$

This weights each auditor’s positive rate in a group by that auditor’s proportion of the group. It is equivalent to a single auditor holding the union of all query sets, and thus recovers the DP of this union exactly, whatever the data heterogeneity. Collaborative audits should therefore estimate DP from the GAS vector, and AUDITOPUS does so.

![](images/352d6965924f6d71f1ee47f66744793ee85a40de67b38f18f0bb68bb58152206.jpg)  
(on top of N = 20 honest auditors)  
Fig. 3. One-shot auditing on the Civil Comments dataset (Christian vs. Muslim, true $\mathrm { D P - 0 . 2 2 8 ) }$ , with $N = 2 0$ honest auditors holding heterogeneous data $( \alpha = 1 )$ . Each adversary submits fabricated counts as large as the largest honest auditor’s, all non-toxic comments about group B. Mean $\pm \nobreakspace \mathrm { S D }$ over 20 seeds.

Naive collaborative auditing (one-shot). In the simplest collaborative protocol, each auditor queries the LLM with its full query set once, sends its LAS vector to others, sums all LAS vectors it receives into a GAS vector, and computes Equation (3) on it. This one-shot protocol, as used in prior work [13], is accurate when all auditors are honest, but extremely fragile otherwise. An adversary colluding with the platform can send a fabricated LAS vector with arbitrarily large $A _ { i }$ and $B _ { i }$ , and thereby steer the estimate to any value, including one such that the LLM appears fair. This vulnerability comes from two facts: (i) honest auditors cannot verify another auditor’s LAS vector, as they never see its queries; and (ii) nothing bounds the size of a single contribution. Figure 3 illustrates this on the Civil Comments dataset for an audit with $N \ = \ 2 0$ honest auditors. Even two adversaries in a setting with 22 total auditors can move the estimate from −0.23 (the true DP) into the fair band, and a third pushes it past 0. They thus control the estimate rather than merely bias it. Their LAS vectors are also plausible, since heterogeneous data can produce a group-B-only query set (Figure 2), so nothing distinguishes them from honest ones in a single exchange. Standard robust aggregation techniques do not solve this. Methods such as the median or trimmed mean [50] discard contributions that deviate from the majority, but under heterogeneity honest auditors with unusual sub-populations deviate too, so they discard the very data that makes the audit representative. This is why AUDITOPUS proceeds in rounds with a fixed per-round budget n (Section III).

## C. Problem Description

In summary, we seek a decentralized auditing protocol in which every honest auditor (i) estimates the DP of the honest auditors’ GAS vector accurately, even under heterogeneity; (ii) reaches the correct fair/unfair verdict despite adversaries that fabricate their LAS vectors to fairwash the LLM; (iii) does not rely on a trusted party; and (iv) sends only LAS vectors, never its (possibly sensitive) queries.

## III. DESIGN OF AUDITOPUS

In this section, we give a detailed overview of AUDITOPUS, our novel approach for robust decentralized fairness auditing.

## A. AUDITOPUS in a Nutshell

In AUDITOPUS, a set of auditors V collaboratively audit a black-box LLM over several rounds. In each round, every auditor queries the LLM with a few samples from its private query set. It then sends only its LAS vector, i.e., the number of queries and positive predictions per group, never the queries themselves. Each auditor sends its LAS vector to a select few other auditors, and every auditor sums all the LAS vectors it has received into a GAS vector to estimate the LLM’s DP. The auditing algorithm is described in Section III-C.

Using the GAS vector makes the estimate accurate even when auditors hold very different query sets. However, it also lets an attacker fabricate LAS vectors to make an unfair LLM look fair. We elaborate on this threat in Section IV. As a defense, AUDITOPUS therefore checks each auditor against its own past LAS vectors rather than against the others. An honest auditor samples from fixed query sets, so its LAS vector stays consistent over rounds. An attacker must keep adjusting its LAS vectors to cancel the honest ones. Each honest auditor then gives less weight to increments that are inconsistent with the source’s past LAS vector. We describe this defense in detail in Section V.

## B. System and Threat Model

Auditors and rounds. A set of auditors V, of which N are honest and K are adversarial, audits an LLM f for DP between two groups A and B. Each honest auditor i holds a private query set $\mathcal { D } _ { i }$ , and these query sets are heterogeneous in size and group composition (Section II-B). The audit proceeds in synchronous rounds $t = 1 , \dots , T$

Audit goal. We assume that the honest auditors’ query sets jointly form the population served by $f ,$ so that the DP of their GAS vector, i.e., (3) computed over honest auditors only, equals $\operatorname { D P } ( f )$ . At the end of the audit, each honest auditor outputs an estimate of $\operatorname { D P } ( f )$ and the corresponding verdict. The goal is for every honest auditor to reach the same verdict as $\operatorname { D P } ( f )$ . How the individual outcomes are combined into a single audit outcome is left to the entity commissioning the audit, e.g., a regulator. In Section VI we consider the median among all estimates.

Network model. A central aggregator could collect all LAS vectors and compute Equation (3), but it would become a single point of trust and failure. Auditors from different organizations or jurisdictions may not agree on a trusted party, and a compromised aggregator could dictate the audit’s outcome. We therefore consider a decentralized setting, in which auditors communicate over a static, connected graph $\mathcal { G }$ and exchange LAS vectors only with their neighbors $\mathcal { N } ( i )$ The network is permissioned. Auditors are approved before joining and know each other’s identities, so an adversary cannot join under many identities (i.e., a Sybil attack [15]). Auditors stay online during the audit, and we assume reliable communication channels. Each auditor digitally signs its own LAS vector together with the round in which it produced it, and auditors relay these signed entries unchanged, so vectors relayed by other auditors cannot be altered or forged.

![](images/17eb42cd09b3f4ce2d0347147a5620ff45b47426fc50197467407b239f948697.jpg)  
Fig. 4. The workflow of AUDITOPUS, from the perspective of an honest auditor i.

Threat model. Adversaries aim at fairwashing [1], trying to make an unfair LLM appear fair, $e . g .$ , on behalf of the platform (Section I). An adversary follows the protocol and its message format, but sends fabricated LAS vectors instead of those of its own queries, and relays the LAS vectors of other auditors unchanged. Each adversary colludes with the platform, but adversaries do not collude with one another: they do not share their query sets, their LAS vectors, or their strategies, and each one crafts its reports on its own. Finally, we assume that the platform answers audit queries like regular traffic, since they come from the auditors’ private query sets and cannot be distinguished from those of regular users [24]. In particular, the platform cannot single out the honest auditors queries to infer their statistics. We formalize the adversary’s capabilities and strategy in Section IV.

In AUDITOPUS, auditors exchange only LAS vectors, their own and those they relay, but never their queries. This is intuitively more private than sharing queries in clear. Deployments that need stronger, formal privacy guarantees can perturb each LAS vector with a differentially private mechanism [17] before it is sent, or run AUDITOPUS’ checks and aggregations under secure multi-party computation [7]. These remain orthogonal to how AUDITOPUS fundamentally works.

## C. AUDITOPUS Auditing Workflow

We now describe the workflow of AUDITOPUS from the perspective of an honest auditor $i ,$ as illustrated in Figure 4 and detailed in Algorithm 1.

State. Each auditor i maintains a LAS vector $c _ { i }$ , which sums the statistics of all queries it has issued so far. In addition, it maintains a table $\mathcal { M } _ { i }$ that maps every auditor $j \in \mathcal V$ to the latest LAS vector that i has received from $j ,$ stored as an entry $\mathcal { M } _ { i } [ j ] = ( c , r )$ signed by $j ,$ where $\mathcal { M } _ { i } [ j ] . c$ is the LAS vector and $\mathcal { M } _ { i } [ j ] . r$ the round in which j produced it. To score incoming statistics, i also keeps, for every auditor $j ,$ the LAS vector $p r e v _ { j }$ that it held for $j$ at the end of the previous round, and a weighted LAS vector $W _ { j }$ that accumulates $j ^ { \circ } \mathrm { s }$ contributions to i’s estimate. All vectors are initialized to zero, and all entries to round 0 (Lines 1-2). Each round then consists of the following steps.

```latex
Algorithm 1 The AUDITOPUS workflow, as executed by
honest auditor i.
Input: query set $\mathcal { D } _ { i } ;$ auditor set V; neighbors N(i); per-round query
budget n; rounds T; score cap z<sub>max</sub>; fair band $\tau ;$ black-box LLM
f
1: $\dot { c } _ { i } \gets ( 0 , 0 , 0 , 0 )$ ▷ Initialize LAS vector of i
2: $\mathcal { M } _ { i } [ j ]  ( ( 0 , 0 , 0 , 0 ) , 0 ) ; p r e v _ { j } , W _ { j }  ( 0 , 0 , 0 , 0 )$ for every
$j \in \mathcal V$ ▷ table and LAS vectors of all auditors
3: for $t = 1 , \dots , T$ do
4: $Q \sim { \mathcal { D } } _ { i } , | Q | = n ; \ { \mathcal { D } } _ { i } \gets { \mathcal { D } } _ { i } \setminus Q$ ▷ Step 1: sample
5: $c _ { i } \gets c _ { i } + \dot { \mathrm { Q U E R Y L L M } } ( f , Q )$ ▷ Step 2: query LLM
6: $\mathcal { M } _ { i } [ i ]  ( c _ { i } , t )$ ▷ LAS vector and its round
7: Send $\mathcal { M } _ { i }$ to every $u \in \mathcal { N } ( i ) \qquad v \in$ Step 3: send and recv.
8: Receive table $\mathcal { M } _ { u  i }$ from each $u \in \mathcal { N } ( i )$
9: for each $\iota \in \mathcal { N } ( i )$ do ▷ Step 4: merge
10: for each $j \in \dot { \mathcal { V } }$ do
11: if $\mathcal { M } _ { u  i } [ j ] . r > \mathcal { M } _ { i } [ j ] . r$ then
12: $\mathcal { M } _ { i } [ j ] \left. \mathcal { M } _ { u \right. i } [ j ]$ ▷ keep the newest vector
13: end if
14: end for
15: end for
16: for each $j \in \mathcal V$ do ▷ Step 5: score
17: $\delta \gets \mathcal { M } _ { i } [ j ] . c - p r e v _ { j }$ ▷ $\delta = ( \Delta a , \Delta A , \Delta b , \Delta B )$
18: if $\Delta A + \mathsf { \bar { \Delta \Delta \Delta \beta } } > \mathsf { \bar { 0 } }$ then
19: if $p r e v _ { j } = ( 0 , 0 , 0 , 0 )$ then
20: $w \gets 1$ ▷ first increment: no history
21: else
22: S ← SUSPICION(δ, prev <sub>j</sub>)
23: $w  ( 1 + S ) ^ { - 2 }$
24: end if
25: $W _ { j } \gets W _ { j } + w \delta$
26: end if
27: $\mathbf { \Lambda } _ { p r e v _ { j } }  \mathcal { M } _ { i } [ j ] . c$
28: end for
29: $\begin{array} { r } { \widehat { \mathrm { D P } } _ { i } \gets \mathrm { D P } \left( \sum _ { j \in \mathcal { V } } W _ { j } \right) } \end{array}$ ▷ Step 6: estimate
30: end for
31: return fair if $\begin{array} { r } { | \widehat { \mathrm { D P } } _ { i } | \leq \tau , } \end{array}$ else unfair
```

Step 1: Sample queries. At the start of round t, auditor i draws a set $Q$ of n queries uniformly at random from $\mathcal { D } _ { i }$ and removes them from its query set, so that no query is issued twice (Line 4).

Step 2: Query the LLM. Auditor i submits the queries in $Q$ to the black-box LLM $f ,$ records the decisions, and adds the resulting statistics to its LAS vector $c _ { i }$ . It then stores $c _ { i }$ in its own entry $\mathcal { M } _ { i } [ i ]$ , signed and stamped with the current round t (Lines 5-6).

Step 3: Send and receive statistics. Auditor i sends its entire table $\mathcal { M } _ { i }$ to each of its neighbors in G, and receives the table $\mathcal { M } _ { u  i }$ of each neighbor $u \in \mathcal { N } ( i )$ in return (Lines $7 -$ 8). Auditors thus relay the (signed) entries of other auditors in addition to their own, so the statistics of each auditor reach every other auditor, advancing one hop per round. Only LAS vectors are exchanged, never queries.

Step 4: Merge. Auditor i merges each received table into its own, entry by entry. It replaces $\mathcal { M } _ { i } [ j ]$ with the received entry $\mathcal { M } _ { u  i } [ j ]$ if the latter was produced in a later round and carries a valid signature of $j$ (Lines 9-15). Relaying whole signed entries rather than per-round statistics makes the merge robust to gossip. An auditor may receive the same entry along several paths or with a delay of several rounds, and comparing rounds neither aggregates it twice nor lets an older vector overwrite a newer one.

Step 5: Score statistics. Without a defense, i would aggregate the latest vectors of all auditors into $\textstyle \sum _ { j } { \mathcal { M } } _ { i } [ j ] . c ,$ whose DP equals DP(f) once all statistics have arrived, provided every auditor is honest (Section II-B). As we show in Section IV, however, a malicious auditor can fabricate its statistics to steer this estimate into the fair band. AUDITOPUS therefore weighs the statistics of each auditor before summing them. For each auditor $j ,$ , including i itself, i computes the increment $\delta = \mathcal { M } _ { i } [ j ] . c - p r e v _ { j } , i . e .$ , the statistics that $j$ added since i last updated its entry (Line 17). When $j$ sends new LAS vectors, i assigns $\delta$ a suspicion score S that measures how statistically inconsistent δ is with j’s LAS vector $\rho r e v _ { j } ,$ and adds δ to $W _ { j }$ with weight $w = ( 1 + S ) ^ { - 2 }$ (Lines 18- 25). The first increment of $j$ has no history to compare with, so i gives it weight 1 without scoring it (Lines 19-20). The intuition is that an honest auditor samples uniformly from a fixed query set, so its increments remain consistent with its own LAS vector, whereas an attacker must keep adjusting its fabricated statistics to counter the honest ones. Since each auditor is compared to its own LAS vector rather than to the other auditors, honest auditors whose query sets differ from the majority are not penalized. The weight of an increment is fixed once assigned and never revised. We detail the suspicion score and the weighting in Section V.

Step 6: Estimate and verdict. Auditor i computes its estimate $\widehat { \mathrm { D P } } _ { i } = \mathrm { D P } \left( \sum _ { i \in \mathcal { V } } W _ { j } \right)$ from the weighted LAS vectors of all auditors (Line 29). After $T$ rounds, i outputs the verdict fair if $| \widehat { \mathrm { D P } _ { i } } | \leq \tau ,$ , and unfair otherwise (Line 31).

## IV. ADVERSARY

After defining the adaptive attacker, we reduce its problem to a single variable, its budget split between the two groups, characterize when one attacker can fairwash the audit, and extend the analysis to several non-colluding adversaries.

## A. Strategy of an Adversary

An adversary k joins the graph like any other auditor and follows the message format. Each round it adds an increment δ to its LAS vector, $c _ { k } \gets c _ { k } + \delta ,$ but δ is fabricated. Honest auditors cannot verify it, because they never see another auditor’s raw queries. They can only check that the LAS vector is syntactically valid.

K = 1 K = 3 True DP   
K = 2 Honest only Fair band   
0.0   
DP estimate −0.1   
−0.2   
−0.3   
20 40 60 80 100   
Audit progress (% of rounds)  
Fig. 5. Estimate of DP with $K = 1 , 2 , 3$ adversaries and no defense, on Civil Comments (Christian vs. Muslim), with $N = 2 0$ honest auditors, $\alpha = 1$ and $n = 3 0 0$ (20 seeds).

Algorithm 2 Fairwashing attacker k: Algorithm 1 with   
Steps 1–2 replaced by FABRICATESTATS, and Steps 5–6   
omitted.   
Input: auditor set V; neighbors $\mathcal { N } ( k )$ ; per-round query budget n;   
rounds T   
1: $c _ { k } \gets ( 0 , 0 , 0 , 0 ) ; ~ \mathcal { M } _ { k } [ j ] \gets \big ( ( 0 , 0 , 0 , 0 ) , 0 \big )$ for every $j \in \mathcal V$   
2: for $t \overset { \cdot } { = } 1 , \ldots , \overset { \cdot } { T }$ do   
3: δ ← FABRICATESTATS $( \mathcal { M } _ { k } , t )$ ▷ replaces Steps 1–2   
4: $c _ { k }  c _ { k } + \delta ; \mathcal { M } _ { k } [ k ]  ( c _ { k } , t )$   
5: Send $\mathcal { M } _ { k }$ to every $u \in \mathcal { N } ( k )$   
6: Receive and merge as in Algorithm 1, Steps 3–4 ▷ relays   
entries unchanged   
7: end for   
8: function FABRICATESTATS(M<sub>k</sub>, t) ▷ returns increment   
$\delta = ( \Delta a , \Delta A , \Delta b , \Delta B )$   
9: if $t = 1$ then   
10: $\Delta A  \lfloor n / 2 \rfloor ; \ \Delta B  n - \Delta A$ ▷ nothing heard yet   
11: return $\left( \lfloor \dot { \Delta } \bar { A } / 2 \right]$ , ∆A, $\lfloor \Delta B / 2 \rfloor$ , ∆B ▷ neutral   
vector   
12: else   
13: $\begin{array} { r } { C \gets \sum _ { j \in \mathcal { V } } \mathcal { M } _ { k } [ j ] . c } \end{array}$ ▷ attacker’s view of the   
GAS vector   
14: return $\delta ^ { \star }$ ∈ arg mi $\mathsf { 1 } _ { \delta \in { \mathcal { X } } _ { n } } \left| \operatorname { D P } ( C + \delta ) \right|$ ▷ (7); X<sub>n</sub>:   
feasible moves, (4)   
15: end if   
16: end function

Admissible increments. At every round, the increment $\delta =$ $( \Delta a , \Delta A , \Delta b , \Delta B )$ sent by an adversary satisfies $\delta \in { \mathcal { X } } _ { n } ,$ with

$$
\begin{array} { c } { { \mathcal { X } _ { n } = \bigl \{ \delta \in \mathbb Z _ { \geq 0 } ^ { 4 } : \Delta A + \Delta B = n , \right. } } \\ { { \left. \Delta a \leq \Delta A , \ \Delta b \leq \Delta B \bigr \} . } } \end{array}\tag{4}
$$

Within these bounds the entries are arbitrary, the adversary need not query $f$ at all, and may fabricate LAS vectors.

Per-round objective under past information. An adversary sends its LAS vector at every round from $t = 1$ onward. When acting at round t, it knows its table $\mathcal { M } _ { k }$ , whose entries date from rounds strictly before t. We collect them in the GAS vector:

$$
C = \sum _ { j \in \mathcal { V } } \mathcal { M } _ { k } [ j ] . c = ( a , A , b , B ) ,\tag{5}
$$

which includes its own past fabricated LAS vector. Its objective is the estimate it can compute at that moment, namely the GAS vector corrected by its own current LAS vector:

$$
\mathrm { D P } ( C + \delta ) = \frac { a + \Delta a } { A + \Delta A } - \frac { b + \Delta b } { B + \Delta B } .\tag{6}
$$

We instantiate a strong, adaptive attacker. In every round $t \geq 2 ,$ , it picks the increment that brings its objective closest to 0

$$
\delta ^ { \star } \in \underset { \delta \in \mathcal { X } _ { n } } { \arg \operatorname* { m i n } } \big | \operatorname { D P } ( C + \delta ) \big | .\tag{7}
$$

Attacker workflow. Algorithm 2 details how adversary k runs this strategy. It follows the round structure of Algorithm 1, except that it never queries the LLM. Instead of sampling and querying (Steps 1–2), it calls FABRICATESTATS (Line 3). It then adds the returned increment δ to its own LAS vector and stores the result in its entry $\mathcal { M } _ { k } [ k ]$ (Line 4). Finally, it sends and merges tables exactly like an honest auditor (Lines $5 \mathrm { - }$ 6), relaying the other auditors’ entries unchanged, so that its messages have the same format as honest ones. It skips the scoring and estimation steps (Steps 5–6), which play no role in its attack. FABRICATESTATS distinguishes two cases. In the first round, the adversary has not received any table yet and has nothing to steer, so it sends a neutral LAS vector. An even split of its budget between the two groups, with positive rate $1 / 2$ in each group (Lines 9-11). From round 2 onwards, it sums its table into its view C of the GAS vector (5) (Line 13) and returns the move $\delta ^ { \star }$ of (7) (Line 14). Computing $\delta ^ { \star }$ requires no query to the LLM, only a search over the finite set $\mathcal { X } _ { n }$

Remarks. First, the objective is evaluated at each round separately, so the analysis is independent of the stopping rule, be it a fixed horizon or a convergence criterion. We therefore drop the round index from $C$ and δ. Second, the GAS vector (5) accumulates the adversary’s own past fabrications, so its influence does not dilute as rounds pass. Acting from round 1, it owns a proportion of at least $1 / ( N + K )$ of the total mass at every round. Third, (6) is only a proxy for the estimate that honest auditors actually compute, since the attacker does not yet know the most recent honest LAS vectors. These recent LAS vectors are however a fraction of all LAS vectors accumulated so far, so the gap shrinks as the audit proceeds and the attacker’s objective becomes an accurate proxy of the honest estimate.

## B. Reduction to a Budget Split

We assume $A > 0$ and $B > 0$ so that the ratios in (6) are well defined, which holds from round 2 on as soon as the honest auditors cover both groups. We assume, without loss of generality, that

$$
\begin{array} { r } { \mathrm { D P } ( C ) < - \tau , } \end{array}\tag{8}
$$

$i . e .$ , the GAS vector is unfair against group A. The opposite case follows by exchanging the roles of the two groups.

Because the budget constraint ties $\Delta B = n - \Delta A$ , a move has three degrees of freedom: how to split the budget between

the two groups, and how many positives to claim in each. The objective (6) then reads

$$
\mathrm { D P } ( C + \delta ) = \frac { a + \Delta a } { A + \Delta A } - \frac { b + \Delta b } { B + n - \Delta A } .\tag{9}
$$

Lemma 1 (Problem reduction to extreme labels). For a fixed split $\Delta A , \mathrm { D P } ( C + \delta )$ is strictly increasing in $\Delta a$ and strictly decreasing in $\Delta b .$ . Every maximizer of $\mathrm { D P } ( C + \delta )$ on ${ \mathcal { X } } _ { n }$ satisfies $\Delta a = \Delta A$ and $\Delta b = 0$

Theorem 1 (proof postponed to Appendix B) relates to the attacker (7) as follows. By (8), the attacker needs to raise the estimate. If no move in ${ \mathcal { X } } _ { n }$ reaches $\mathrm { D P } ( C + \delta ) \ge 0$ , minimizing $| \operatorname { D P } ( C + \delta ) |$ amounts to maximizing $\mathrm { D P } ( C + \delta )$ , so $\delta ^ { \star }$ claims that every group-A query is positive and every group-B query negative. Otherwise, these extreme labels would overshoot above 0, and $\delta ^ { \star }$ instead uses intermediate labels that bring $\mathrm { D P } ( C + \delta ^ { \star } )$ as close to 0 as possible (Algorithm 2).

The extreme-label move with split $\begin{array} { r l } { \Delta A , } & { { } i . e . , \delta } \end{array} =$ $( \Delta A , \Delta A , 0 , n - \Delta A )$ , yields $\operatorname { D P } ( C + \delta ) = \varphi ( \Delta A )$ , where for $y \in [ 0 , n ]$

$$
\varphi ( y ) = \frac { a + y } { A + y } - \frac { b } { B + n - y } .\tag{10}
$$

Both denominators are positive on $[ 0 , n ] .$

The attacker’s problem thus reduces from four fabricated numbers to a single variable: how to split its budget n between the two groups.

## C. Fairwashing Conditions for a Single Adversary

Fix one adversary k. We look for a necessary and sufficient condition for the existence of an integer split $\Delta A \in \{ 0 , \ldots , n \}$ such that $| \varphi ( \Delta A ) | < \tau$

Theorem 2. Assume

$$
\operatorname* { m i n } ( A , B ) + 1 > \frac { 1 } { 2 \tau } .\tag{11}
$$

Then there exists $\Delta A \in \{ 0 , \ldots , n \}$ such that $| \varphi ( \Delta A ) | < \tau$ if and only if the following two conditions hold:

1) min $( \varphi ( 0 ) , \varphi ( n ) ) < \tau$ , that is,

$$
\mathrm { D P } ( C ) + \operatorname* { m i n } \left( \frac { n b } { B ( B + n ) } , \ \frac { n \left( A - a \right) } { A ( A + n ) } \right) < \tau ;
$$

2) $\varphi ( \Delta A ) > - \tau$ for some $\Delta A \in \{ 0 , \ldots , n \}$

Condition 2 can be checked in closed form. It holds if and only if a quadratic has two roots with an integer split between them (Lemma 3, Appendix B-B). The proof is given in Appendix B. Since the extreme-label moves belong to ${ \mathcal { X } } _ { n } ,$ these conditions are sufficient for the attacker (7) to reach the fair band at round t (Corollary 7).

Takeaway of Theorem 2. The assumption (11) and condition 1 are mild: with $\tau = 0 . 0 5$ , they hold once min $( A , B ) \geq$ 10 and max $( A , B ) \geq 9 n$ (Lemma 5). The binding condition is condition 2: a single LAS vector must close the gap $| \mathrm { D P } ( C ) | - \tau$ to the fair band. Hence, the closer an LLM is to being fair, the easier it is to fairwash. This gap also shrinks over rounds, since $C$ already contains the adversary’s past fabrications. Accordingly, without defense, a single adversary fairwashes every near-fair run on both datasets, but only part of the very unfair ones (Section VI).

## D. Multiple Non-Colluding Adversaries

Theorem 2 describes what one adversary k can reach from its own view C. With $K \ > \ 1$ , the other adversaries are part of this view: their past fabrications are in $C ,$ and their LAS vectors of round t are unknown to k, like the honest LAS vector of that round. The theorem therefore applies to each adversary as is.

More adversaries fairwash faster. Every adversary adds n fabricated queries per round, all pushing the estimate towards 0. With K adversaries, fabricated mass accumulates in the GAS vector K times faster, so the gap that remains to be closed in condition 2 of Theorem $^ 2$ shrinks faster as well. Figure 5 shows this effect on Civil Comments with $N = 2 0$ honest auditors. With honest auditors alone, the estimate stays at $\mathrm { D P } ( f ) \approx - 0 . 2 3$ . A single adversary moves it slowly, and the estimate enters the fair band only after about 80% of the rounds. Two adversaries bring it into the band after about 25% of the rounds, and three after about 10%. Once inside the band, the adversaries keep pushing, because their objective (7) is 0 rather than the edge of the band. As a result, all curves end close to 0.

## E. Takeaway

Fairwashing is cheap: adversaries need not query the LLM, as each finds its best LAS vector by simply trying every split of its budget between the two groups. It is also effective: a single adversary suffices to make an LLM appear fair, all the more easily as it is close to the fair band, and its influence persists since past fabrications keep counting in the estimate. Nor do several adversaries need to coordinate: each attacking from its own view, they can together push the estimate past 0. Worse, each fabricated LAS vector, taken alone, looks like an honest one from an auditor with an unusual query set, so robust aggregation cannot flag it. What does betray an adversary is its sequence of reports: to keep cancelling the honest LAS vectors, it must adapt its own every round, so they stop resembling its past ones. Our defense exploits exactly this (Section V).

## V. DEFENSE

The defense at the core of AUDITOPUS limits the influence of fabricated LAS vectors on each honest auditor’s estimate (Step 5 of Algorithm 1). We first explain why honest and adversarial increments differ, then how each increment is scored, weighted, and aggregated into the defended estimate.

Each honest auditor runs the defense locally, on the messages it receives. It needs no trusted party, no coordination with the other auditors, and no offline calibration. The defense rests on one observation. An honest auditor samples its queries from a fixed query set, so each LAS vector is statistically consistent with its previous ones. The adversary of Section IV instead re-optimizes its LAS vector every round to cancel the honest ones, so its increments keep departing from what it sent before. AUDITOPUS therefore checks each source against its own history, never against the other sources. Under heterogeneity, honest sources legitimately disagree with each other, and this disagreement is what defeats robust aggregation techniques such as the median and the trimmed mean (Section II).

## A. Honest and Adversarial Increments

Fix a source $j ,$ and let $\boldsymbol { c } = ( a , A , b , B )$ be the LAS vector that auditor i last held for it. $\textrm { I f } \ j$ is honest, c counts the queries that $j$ has drawn so far from its query set $\mathcal { D } _ { j }$ , and its next increment $\delta = ( \Delta a , \Delta A , \Delta b , \Delta B )$ is drawn from the same query set. Three statistics of δ then approximately follow binomial distributions, whose parameters c estimates: (i) the number $\Delta A$ of group-A queries among the $\Delta A + \Delta B$ queries of the increment, with the proportion of group A in $\mathcal { D } _ { j }$ as parameter; (ii) the number $\Delta a$ of positive decisions among these $\Delta A$ queries, with the LLM’s positive rate on group A in $\mathcal { D } _ { j }$ as parameter; (iii) likewise, $\Delta b$ among the $\Delta B$ group-B queries. The approximation lies in treating the rates estimated from c as exact, not in sampling without replacement: the increment and the history are two disjoint uniform samples of the same query set, so the variance of the difference between their rates does not depend on how much of the query set has been used. The binomial model is therefore accurate once the history is long compared with the increment, and overstates the deviation of an honest increment only in the first rounds.

The adversary of Section IV follows none of these distributions, because it changes its LAS vector as honest ones arrive. Its first LAS vector is neutral, with rate $1 / 2$ in each group. From round 2 on, it claims that every group-A query is positive and every group-B query negative (Lemma 1), so its rates jump away from its own past. It also adjusts the split of its budget to its current view of the GAS vector, so its group proportion varies from round to round. Once the estimate approaches 0, it switches to intermediate labels, and with several adversaries, each one corrects for the overshoot of the others (Section IV-D). Each of these moves appears as a deviation on at least one of the three statistics.

## B. Suspicion Score

Auditor i scores the increment $\delta \ = \ \mathcal { M } _ { i } [ j ] . c \ - \ p r e v _ { j }$ of $j$ each time it replaces its entry $\mathcal { M } _ { i } [ j ]$ with a newer one (Lines 17-22 of Algorithm 1). When gossip delays an entry, this increment covers several rounds of $j ;$ the score applies to increments of any size. Algorithm 3 computes the suspicion score S of δ against the source’s LAS vector $c = p r e v _ { j }$ . It relies on the binomial z-score, computed as $\begin{array} { r } { z ( s ; m , p ) \^ { \smile } = \ \frac { s - m p } { \sqrt { m p ( 1 - p ) } } } \end{array}$ , of s successes in m trials with probability $p ,$ which measures, in standard deviations, how far s lies from what the binomial model predicts. Algorithm 3 proceeds in four steps.

Estimate the source’s parameters (Line 2). From $c ,$ auditor i estimates the three parameters of Section V-A: the proportion πˆ of group A in the source’s queries, and its positive rates $\hat { q } _ { \mathsf { A } }$ and $\hat { q } _ { \mathsf { B } }$ in each group. We use Krichevsky-Trofimov (KT) estimates, which add $1 / 2$ to each LAS vector. They always lie in (0, 1), even when the history contains no positive or no negative decision, so the z-scores are always defined, and a short history with extreme rates is not treated as a certainty.

```latex
Algorithm 3 Suspicion score of one increment
Input: increment $\delta = ( \Delta a , \Delta A , \Delta b , \Delta B )$ with $\Delta A + \Delta B > 0 ;$
LAS vector $\boldsymbol { c } = ( a , A , b , B )$ of the same source, with $A + B > 0$
Ensure: suspicion score $S \in [ \bar { 0 } , z _ { \operatorname* { m a x } } ]$
1: function SUSPICION(δ, c)
2: $\begin{array} { r l } { \hat { \pi }  \frac { A + 1 / 2 } { A + B + 1 } ; \hat { q } _ { \mathsf { A } }  \frac { a + 1 / 2 } { A + 1 } ; \hat { q } _ { \mathsf { B } }  \frac { b + 1 / 2 } { B + 1 } } & { { } \triangleright K T } \end{array}$ estimates
3: $z _ { 1 } \gets \dot { z } ( \tilde { \Delta } \dot { A } ; \Delta A + \dot { \Delta } \dot { B } , \hat { \pi } )$ ▷ group-A proportion
4: $z _ { 2 } \gets z ( \Delta a ; \ \Delta A , \ \hat { q } _ { \mathsf { A } } ) \ \mathbf { i f } \ \Delta A > 0$ ▷ rate in A
5: $z _ { 3 } \gets z ( \Delta b ; \Delta B , \hat { q } _ { \mathsf { B } } ) \ \mathbf { i f } \ \Delta B > 0$ ▷ rate in B
6: return min $\left( \operatorname* { m a x } _ { \ell } | z _ { \ell } | , \ z _ { \operatorname* { m a x } } \right)$ over the defined $z _ { \ell }$
7: end function
```

Check the group proportion (Line 3). $z _ { 1 }$ compares the number ∆A of group-A queries in the increment with the number predicted by the source’s own query set. $s = \Delta A$ successes in $m = \Delta A + \Delta B$ trials, which equals the budget n for an increment covering one round, with probability πˆ. A large |z | reveals a source that changes how it splits its budget between the groups.

Check the positive rates (Lines 4-5). z<sub>2</sub> compares the number ∆a of group-A positives in the increment with the source’s own rate. $s = \Delta a$ successes in $m = \Delta A$ trials with probability $\hat { q } _ { \mathsf { A } }$ . It is defined only if the increment contains group-A queries $( \Delta A > 0 )$ $z _ { 3 }$ does the same for group B, with $s = \Delta b , m = \Delta B$ and qˆ . A large $| z _ { 2 } |$ or |z | reveals a source that changes its labels. For instance, after its neutral first LAS vector $( \hat { q } _ { \mathsf { A } } \approx 1 / 2 )$ , the adversary claims $\Delta a = \Delta A .$ which gives $z _ { 2 } = \sqrt { \Delta A \left( 1 - \hat { q } _ { \mathrm { A } } \right) / \hat { q } _ { \mathrm { A } } } \approx \sqrt { \Delta A } , i . e .$ , about 12 for 150 group-A queries.

Combine (Line 6). The score is the largest of the defined |z<sub>ℓ</sub>|, capped at $z _ { \operatorname* { m a x } } \colon S = \operatorname* { m i n } \bigl ( \operatorname* { m a x } _ { \ell } | z _ { \ell } | , ~ z _ { \operatorname* { m a x } } \bigr )$

Taking the maximum means that a deviation on a single statistic suffices, $e . g . ,$ an adversary that keeps its split but changes its labels. The cap bounds the score, so that no increment ever receives zero weight. An honest increment that deviates from the binomial model, $e . g . .$ , early in the audit when its source’s history is still short, keeps part of its weight.

## C. Locked Weights

An increment with score S receives the weight $w =$ $( 1 + S ) ^ { - 2 }$ , and auditor i adds w δ to the weighted LAS vector $W _ { j }$ of its source (Lines 23-25 of Algorithm 1). The weighting is soft. There is no threshold to calibrate, and an honest source with an unlucky increment is down-weighted rather than excluded. The weight is fixed once assigned and never revised. An adversary therefore cannot build up trust to rehabilitate what it sent earlier, and trust gained on earlier increments does not carry over to later ones, since each increment is weighted on its own. The first increment of each source has no history to compare with and receives $w = 1$

## D. Defended Estimate

Auditor i sums the weighted LAS vectors of all sources, including its own (Line 29, Algorithm 1) as $\begin{array} { r l } { \widehat { \mathrm { D P } _ { i } } } & { { } = } \end{array}$ $\textstyle \operatorname { D P } \left( \sum _ { j \in \nu } W _ { j } \right)$ , where $W _ { j }$ is the sum of the increments of $j ,$ each multiplied by its weight. Auditor i weights its own LAS vectors like everyone else’s. Trusting its own LAS vectors would anchor its estimate on its own skewed query set, and under heterogeneity, honest auditors would then disagree even without adversaries.

Agreement. Every weight depends only on the source’s own LAS vectors and on how auditor i received them. Two honest auditors that received the same increments for every source therefore compute the same estimate. This holds on the complete graph, where every entry reaches every auditor in the round it is produced. On sparser graphs, gossip delays can merge several rounds into one increment for some auditors but not for others, so their estimates may differ slightly. Section VI combines them through their median.

Cost. Each round, an auditor sends each neighbor its table: |V| signed entries, each a LAS vector and a round number. Scoring takes at most |V| calls to Algorithm 3, each computing three z-scores. The defense shares nothing beyond the LAS vectors that the protocol already exchanges.

## VI. EXPERIMENTAL EVALUATION

We implement AUDITOPUS<sup>1</sup> and perform an extensive experimental evaluation of our system and baselines. Our experiments answer the following two questions:

1) Effectiveness (RQ1). How accurately do AUDITOPUS and baselines preserve the audit’s verdict and its DP estimate under attack as the proportion of adversaries grows (Section VI-B)?

2) Sensitivity (RQ2). How do the number of honest auditors and level of data heterogeneity affect the robustness of AUDITOPUS and baselines (Section VI-C)?

We provide additional experiments in Appendix D that assess how our results generalize to a second dataset (Appendix D-A), and compare against additional variants of the median and trimmed-mean baselines (Appendix D-C). We further validate our design choices, including query sampling without replacement (Appendix D-B) and the query budget per round (Appendix D-D).

## A. Experimental Setup

We next describe our experimental setup, including dataset and baseline descriptions, and protocol parameters.

Setup. We audit two zero-shot LLM classifiers: QWEN2.5- 7B-INSTRUCT [39] for toxicity on Civil Comments [8], and LLAMA-3.1-8B-INSTRUCT [26] for occupation (one-vs-rest) on Bias in Bios [12]. For each dataset, we pick three unfair cases, which we call very unfair, moderate and near-fair, with a large, medium and small true DP, respectively (Table I). Auditors hold non-IID query sets drawn from a Dirichlet distribution $( \alpha = 1$ unless stated otherwise). Further experimental details are given in Appendix C. We fix $N \ = \ 2 0$ honest auditors, unless stated otherwise.

TABLE I  
SUMMARY OF THE SIX AUDIT CASES AND THE TRUE DP OF THE AUDITED LLMS. Case: VERY UNFAIR, MODERATE OR NEAR-FAIR, DEPENDING ON THE MAGNITUDE OF THE TRUE DP: Population: THE INPUTS THE AUDITORS OUERY, i.e.. THE COMMENTS MENTIONING GROUP A OR B (CIVIL COMMENTS) OR THE BIOGRAPHIES IN THE GIVEN SECTOR (BIAS IN BIOS); Positive: THE PREDICTION DEFINED AS POSITIVE; FOR BIAS IN BIOS, THE MULTI-CLASS OCCUPATION CLASSIFIER IS TURNED INTO A ONE-VS-REST CLASSIFIER FOR THE TARGET OCCUPATION (APPENDIX C). |A|, |B|: NUMBER OF INPUTS PER GROUP; Rate: PROPORTION OF POSITIVE PREDICTIONS PER GROUP; TRUE DP = RATE A − RATE B OVER THE WHOLE POPULATION.
<table><tr><td>Case</td><td>Attribute</td><td>A vs. B</td><td>Population</td><td>Positive</td><td>|A|</td><td>|B|</td><td>Rate A</td><td>Rate B</td><td>True DP</td></tr><tr><td colspan="10">Civil Comments QWEN2.5-7B-INSTRUCT, zero-shot toxicity detection</td></tr><tr><td>Very unfair</td><td>Religion</td><td>Christian vs. Muslim</td><td>comments</td><td>toxic</td><td>39,688</td><td>17,622</td><td>0.208</td><td>0.436</td><td>-0.228</td></tr><tr><td>Moderate</td><td>Religion</td><td>Christian vs. Jewish</td><td>comments</td><td>toxic</td><td>39,688</td><td>4,763</td><td>0.208</td><td>0.398</td><td>-0.190</td></tr><tr><td>Near-fair</td><td>Race</td><td>Black vs. White</td><td>comments</td><td>toxic</td><td>9,385</td><td>16,926</td><td>0.476</td><td>0.537</td><td>-0.060</td></tr><tr><td colspan="10">Bias in Bios LLAMA-3.1-8B-INSTRUCT, zero-shot occupation prediction (one-vs-rest)</td></tr><tr><td>Very unfair</td><td>Gender</td><td>male vs. female</td><td>Healthcare</td><td>nurse</td><td>45,742</td><td>48,985</td><td>0.028</td><td>0.300</td><td>-0.272</td></tr><tr><td>Moderate</td><td>Gender</td><td>female vs. male</td><td>Healthcare</td><td>physician</td><td>48,985</td><td>45,742</td><td>0.423</td><td>0.571</td><td>-0.148</td></tr><tr><td>Near-fair</td><td>Gender</td><td>male vs. female</td><td>Education</td><td>teacher</td><td>71,261</td><td>63,018</td><td>0.035</td><td>0.098</td><td>-0.063</td></tr></table>

Protocol parameters. The $N + K$ auditors, attackers included, communicate over an Erdos–R˝ enyi graph with an´ expected average degree of 4. We redraw the graph until it is connected and keep it fixed during the audit, so attackers occupy random positions in the network. In every round, each auditor issues $n = 3 0 0$ queries to the LLM. We run each audit for at most $T = 1 5 0$ rounds. At that point, the honest auditors have jointly queried every input in the population exactly once, so the DP computed from the honest auditors’ LAS vectors alone equals the true DP. Any deviation from it is therefore caused by the attackers. We report results for querying with replacement in Appendix D-B. We use a fair band of $\tau = 0 . 0 5$ across all experiments.

Attack and defense. We vary the number of attackers $K \in$ $\{ 0 , \ldots , 1 9 \}$ . Each attacker runs Algorithm 2 independently. We set $z _ { \operatorname* { m a x } } = 8$ in all our experiments (Algorithm 3).

Baselines. We compare AUDITOPUS with three baselines: (i) no defense, which sums all entries into the GAS vector; (ii) the median [50]; and (iii) the trimmed mean [50], which drops the ⌊βm⌋ smallest and largest of the m values before averaging. Baselines (ii) and (iii) aggregate the positive rate of each group separately $( i . e . , a / A$ and $b / B )$ , and take DP as the difference of the two results. We use $\beta = 2 0 \%$ . Appendix D-C reports results for $\beta \in \{ 1 0 , 2 0 , 3 0 , 4 0 \} \%$ and for both robust aggregators applied directly to each source’s local DP.

Metrics. For our experiments, we combine the auditors final estimates into a single verdict as follows. At the end of the audit, every auditor sends its DP estimate (Line 29 of Algorithm 1) to a single party, e.g., a regulator. An adversary always sends 0, to make the LLM appear fair. The global DP estimate $\widehat { \mathrm { D P } }$ is the median of the $N + K$ DP estimates. Every LLM we audit in our experiments is unfair, and we order the groups so that $\mathrm { D P } ( f ) < - \tau$ (Table I). Hence any estimate in the fair band is a wrong verdict, and any positive error moves the estimate towards the band. We focus on two metrics: (i) attack success rate (ASR), the proportion of runs where $| \widehat { \mathrm { D P } } | \leq \tau ,$ , i.e., where an unfair LLM is declared fair;

(ii) the DP estimation error $\widehat { \mathrm { D P } } - \mathrm { D P } ( f )$ , which flips the verdict once it exceeds $| \operatorname { D P } ( f ) | - \tau .$ . An estimate pushed to 0 has error $| \operatorname { D P } ( f ) |$ .

For every configuration, we run 20 seeds, where a seed fixes the data partition, the communication graph and the query order. We average all results and present error bars showing standard deviation where applicable.

## B. RQ1: Effectiveness of AUDITOPUS and baselines

We first quantify the robustness of AUDITOPUS and baselines on the Bias in Bios dataset (we provide results for Civil Comments in Appendix D-A). Table II reports the mean ASR and mean absolute DP estimation error of AUDITOPUS and baselines, over every run with $K = 1$ to 19 adversaries. Its first three rows correspond to the three Bias in Bios cases of Table I, from the very unfair to the near-fair LLM, and the last row combines them. This last row highlights the effectiveness of AUDITOPUS. Without any defense, the attack succeeds in 97% of the audits, and the median and the trimmed mean let it through in 24% and 41% of the audits, respectively. In stark contrast, AUDITOPUS almost eliminates the attack, which succeeds in only 3% of the audits. AUDITOPUS also substantially improves accuracy, reducing the mean absolute error from 0.158 to $0 . 0 3 4 \ ( - 7 8 \% )$ , and by 75% and 62% relative to the median and the trimmed mean. These results demonstrate that AUDITOPUS remains robust against adversaries and consistently outperforms all baselines.

Figure 6 breaks down these results by number of adversaries K, up to $K ~ = ~ 1 9$ (top: ASR, bottom: estimation error). Without a defense, fairwashing is remarkably easy: a single adversary already succeeds in 100%, 40%, and 5% of the runs for the near-fair, moderate, and very unfair LLM, respectively, $i . e . .$ , more often the closer the LLM is to the fair band, as Theorem 2 predicts. Two adversaries suffice to succeed in every run, driving the estimate all the way to 0 (dashed line) and making even the very unfair LLM appear perfectly fair. Robust aggregation does not solve this. The median and the trimmed mean aggregate the auditors’ individual positive rates, which differ across heterogeneous query sets, so their estimate is biased even before any attack: with $K = 0$ , they already declare the moderate LLM fair in 15% and 20% of the runs, and 46% and 71% under attack. The median’s low ASR on the very unfair LLM (5%) is no sign of robustness either: up to K = 16, the adversaries push its estimate away from the band, so that it overstates the disparity, and beyond, it breaks down (55% at K = 19). We analyze both baselines in more detail in Appendix D-C.

TABLE II  
THE ATTACK SUCCESS RATE (ASR) AND MEAN ABSOLUTE DP ESTIMATION ERROR ON THE BIAS IN BIOS DATASET, OVER EVERY RUN WITH 1–19 ADVERSARIES (5–49% OF ALL AUDITORS; 380 RUNS PER CASE). WE PROVIDE SIMILAR RESULTS FOR CIVIL COMMENTS IN TABLE IV.
<table><tr><td colspan="5">Attack success rate (%)</td><td colspan="4">Average DP estimation error</td></tr><tr><td>LLM fairness</td><td>No-defense</td><td>Median</td><td>Trimmed mean</td><td>AUDITOPUS</td><td>No-defense</td><td>Median</td><td>Trimmed mean</td><td>AUDITOPUS</td></tr><tr><td>Very unfair</td><td>95</td><td>5</td><td>0</td><td>0</td><td>0.267</td><td>0.137</td><td>0.079</td><td>0.060</td></tr><tr><td>Moderate</td><td>97</td><td>46</td><td>71</td><td>0</td><td>0.144</td><td>0.189</td><td>0.155</td><td>0.034</td></tr><tr><td>Near-fair</td><td>100</td><td>20</td><td>53</td><td>9</td><td>0.062</td><td>0.079</td><td>0.037</td><td>0.006</td></tr><tr><td>All three cases</td><td>97</td><td>24</td><td>41</td><td>3</td><td>0.158</td><td>0.135</td><td>0.090</td><td>0.034</td></tr></table>

![](images/ad8b0ae8b28a0a23984265463f08ec976205ee2601f41163d7e94e5b3097f15f.jpg)  
Number of adversaries (on top of N = 20 honest auditors)  
Fig. 6. The attack success rate (top row) and DP estimation error (in the bottom row) for Bias in Bios, while varying the number of adversaries K from 0 to 19. We consider a very unfair (left column), moderate (middle column) and near-fair LLM (right column). The dashed horizontal line (in the bottom row) corresponds to a DP estimate of 0. Results for the Civil Comments dataset are provided in Appendix D-A.

AUDITOPUS, in contrast, keeps the ASR at 0% on the very unfair and moderate LLMs, even when K = 19 of the 39 auditors (49%) are adversarial. Its estimate does drift slowly towards the fair band as K grows, since AUDITOPUS downweights the fabricated LAS vectors rather than discarding them entirely, but this drift remains modest: even at K = 19, the error reaches only +0.088 on the very unfair LLM, i.e., 40% of the 0.222 gap between its true DP and the fair band, and about half of that gap on the moderate LLM. The nearfair LLM is the hardest case since, as seen above, a small shift suffices to flip its verdict. The ASR of AUDITOPUS in this case remains 0% up to K = 3 but reaches 20% at K = 19. Even then, AUDITOPUS keeps the estimation error below +0.008 for every K, about 8x lower than the +0.061 without defense.

In answer to RQ1, AUDITOPUS remains effective across all fairness levels in Bias in Bios, even when nearly half of the auditors (49%) attempt to fairwash the LLM.

## C. RQ2: Sensitivity to honest auditors and heterogeneity level

We next explore how the number of honest auditors N and the heterogeneity of their query sets affect the robustness of AUDITOPUS and baselines. In Figure 7, we report the DP estimation error on the moderate LLM of Bias in Bios for N ∈ {10, 20, 30} and heterogeneity levels α ranging from 100 (near-homogeneous query sets) to 0.2 (highly heterogeneous), with the number of adversaries K fixed at 30% of N. Some cells in Figure 7 are left out since, for some low values of α, some seeds have no feasible data split in which every auditor holds at least 30 queries of each group.

Figure 7 shows that AUDITOPUS is rather insensitive to changing the number of honest auditors and heterogeneity level, with values ranging from 0.019 to 0.045, thus changing by at most 2.1x across N at a fixed α. At the same time, both robust aggregation baselines break down as the query sets become more heterogeneous, i.e., as α decreases. With nearhomogeneous query sets (α = 100), the median is as accurate as AUDITOPUS, with an error less than 0.03 for every N. As α decreases, however, the errors of both the median and the trimmed mean grow steadily: for N = 10 and α = 0.2, they reach 0.26 and 0.17, respectively. This is because robust aggregators assume that the honest values are concentrated. Adversaries can then only shift the aggregate from one honest positive rate to another, which barely changes the estimate when these rates are close to each other. Heterogeneity spreads the honest rates apart, so the same shift moves the estimate much further. AUDITOPUS, which judges each source only against its own history, does not rely on this assumption.

![](images/6b512b7097c1707feff8d74a12f074f9e5d2f0abea1810b7a7d39196db6a2029.jpg)  
Fig. 7. The DP estimation error on the Bias in Bios dataset, for AUDITOPUS and baselines, on the moderate LLM. We vary the number of honest auditors (N) and heterogeneity level (α). Lower is better.

Answer to RQ2: AUDITOPUS is insensitive to both the number of honest auditors and the heterogeneity of their query sets, with a DP estimation error between 0.02 and 0.05 in every setting, whereas the error of the median and the trimmed mean grows with heterogeneity, up to 0.25 and 0.21, respectively.

## VII. RELATED WORK

Manipulation of black-box audits. Black-box fairness auditing is usually framed as estimating a metric from the model’s answers to a limited number of queries [40]. Such audits are easy to game when the platform is adversarial. For example, it can justify unfair decisions with fair-looking explanations [42], present a biased sample as evidence of fairness [19], [36], or answer audit queries differently from regular traffic [1]. Defenses check the answers against the auditor’s prior knowledge [20] or against an independent trusted source [21]. Others hide the audit set inside a larger candidate set using private information retrieval [23], or replace querying with cryptographic proofs of fairness [43], [49]. All these works treat the platform as the adversary and the auditor as trusted. AUDITOPUS, however, addresses a complementary threat where the platform acts through a colluding auditor.

Collaborative auditing and poisoned aggregates. Alongside decentralized methods for training fair machine learning (ML) models [6], a recent line of work explore how several parties can collaborate to audit a model for fairness, though not yet what happens when some of them cannot be trusted. Auditors can aggregate their query budgets and coverage [13]. In collaborative active learning, FAIR [48] coordinates how several parties select their queries so that each one benefits from the collaboration (individual rationality) and the gains are shared equitably. FAAS [45] lets many parties compute a fairness metric jointly, with zero-knowledge proofs that the submitted cryptograms are well formed. Such proofs show that the computation is correct, not that an auditor’s inputs reflect the model’s real answers, so such works assume honest auditors. The closest threat studied so far is poisoning of distributed statistics. Fake users of a local differential privacy protocol can craft reports that shift estimated frequencies at will [10]. Our attacker plays the same game against an aggregated fairness metric, but AUDITOPUS exploits what local differential privacy protocols lack. Each auditor reports over many rounds, which gives honest auditors a history to check every new LAS vector against.

Byzantine-robust aggregation and gossip. Robust aggregators like the median and trimmed mean [50] discard inputs lying far from the majority. Robust gossip extends this to peer-to-peer networks [30], [22] but with weakened guarantees under data heterogeneity [33], [2]. Robust gossip also typically uses the receiving node’s own value as its trusted reference [30], [22]. AUDITOPUS keeps every auditor’s LAS vectors and never compares auditors with each other, so honest but atypical auditors are not penalized.

History-based detection. Robust distributed learning has also exploited participants’ past behavior, either through momentum [32] or by predicting each client’s next model update and flagging clients whose updates keep deviating from it [51]. AUDITOPUS applies this idea to LAS vectors, where honest behavior is known exactly. An honest auditor samples from a fixed query set, so each new increment follows a binomial distribution given its own past. Unlike these detectors, AU-DITOPUS therefore needs no learned predictor and no trusted server, and it down-weights LAS vectors with locked weights instead of removing participants. Attackers that stay consistent across rounds can evade history-based detectors [47], but in our setting such an attacker gives up the per-round adaptivity it needs to cancel the honest LAS vectors.

## VIII. CONCLUSION

We studied collaborative fairness auditing of black-box LLMs, where auditors with small, heterogeneous query sets estimate demographic parity together by exchanging only locally aggregated statistics vectors. We showed, theoretically and empirically, that globally aggregated statistics vectors are accurate under heterogeneity but fragile. A single adversary can fabricate its LAS vectors to make an unfair LLM appear fair, and standard robust aggregation fails because honest auditors with atypical data deviate from the majority as well. AUDITOPUS is a novel round-based decentralized auditing protocol in which each honest auditor down-weights sources whose LAS vectors are statistically inconsistent with their own history, without a trusted party and without penalizing honest but atypical auditors. Across two datasets and six audit cases, AUDITOPUS reduces attack success rate and error in estimation compared against the no defense and robust aggregation baselines.

## IX. ACKNOWLEDGMENTS

This work has been funded by the Swiss National Science Foundation, under the project “FRIDAY: Frugal, PrivacyAware and Practical Decentralized Learning”, SNSF proposal No. 10.001.796.

## OPEN SCIENCE

We release all artifacts needed to reproduce our results in an anonymized repository. It contains the source code of AUDITOPUS and its baselines, the datasets and prompts used in our experiments, and the scripts and configuration files to run every experiment in Section VI and Appendix D. It also includes the scripts that generate all figures and tables from the raw experimental outputs. The evaluated models are publicly available through their original providers.

## LLM USAGE CONSIDERATIONS

LLMs were used for editorial purposes in this manuscript, and all outputs were inspected by the authors to ensure accuracy and originality. We also used LLMs as coding assistants; we manually reviewed and tested all generated code for correctness and accuracy. We did not use generative AI to propose hypotheses, design the methodology or experiments, or interpret results. The authors take full responsibility for all content of this paper.

LLMs are also the subject of our study since AUDITOPUS audits the fairness of LLMs, so our experiments necessarily query them. To support reproducibility, we use open-weight models, and document the exact configurations and parameters. To limit the environmental footprint, we query each model only once per prompt and store the responses. All experiments then reuse these cached responses rather than repeating inference, so the total LLM compute is independent of the number of experiments we run. We include these cached responses in our artifact, so our results can be reproduced without re-querying the models. All experiments ran on our compute cluster, totaling approximately 100 GPU-hours.

## ETHICAL CONSIDERATIONS

This work involves no human subjects and no personal data.   
All used datasets and pre-trained LLMs are publicly available.   
Our goal is to make fairness audits of LLMs more trustworthy.

We study adversarial behavior to design defenses against it. We do not introduce new attack capabilities against deployed systems or their users. We note that no auditing procedure, including ours, can certify with absolute certainty that a model is fair. Audit outcomes depend on the fairness metrics and demographic groups considered, which do not capture every form of harm.

## REFERENCES

[1] Ulrich A¨ıvodji, Hiromi Arai, Olivier Fortineau, Sebastien Gambs,´ Satoshi Hara, and Alain Tapp. Fairwashing: the risk of rationalization. In International conference on machine learning, pages 161–170. PMLR, 2019.

[2] Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, and Geovani Rizk. Robust distributed learning: Tight error bounds and breakdown point under data heterogeneity. Advances in neural information processing systems, 36:45744–45776, 2023.

[3] McKane Andrus, Elena Spitzer, Jeffrey Brown, and Alice Xiang. What we can’t measure, we can’t understand: Challenges to demographic data procurement in the pursuit of fairness. In Proceedings of the 2021 ACM conference on fairness, accountability, and transparency, pages 249– 260, 2021.

[4] Lena Armstrong, Abbey Liu, Stephen MacNeil, and Danae Metaxa.¨ The silicon ceiling: Auditing GPT’s race and gender biases in hiring. In Proceedings of the 4th ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization, pages 1–18, 2024.

[5] Peter J. Bickel, Eugene A. Hammel, and J. William O’Connell. Sex bias in graduate admissions: Data from Berkeley. Science, 187(4175):398– 404, 1975.

[6] Sayan Biswas, Anne-Marie Kermarrec, Rishi Sharma, Thibaud Trinca, and Martijn De Vos. Fair decentralized learning. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pages 714–734. IEEE, 2025.

[7] Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. Practical secure aggregation for privacy-preserving machine learning. In proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, pages 1175–1191, 2017.

[8] Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. Nuanced metrics for measuring unintended bias with real data for text classification. In Companion proceedings of the 2019 world wide web conference, pages 491–500, 2019.

[9] Toon Calders, Faisal Kamiran, and Mykola Pechenizkiy. Building classifiers with independency constraints. In 2009 IEEE international conference on data mining workshops, pages 13–18. IEEE, 2009.

[10] Xiaoyu Cao, Jinyuan Jia, and Neil Zhenqiang Gong. Data poisoning attacks to local differential privacy protocols. In 30th USENIX Security Symposium (USENIX Security 21), pages 947–964, 2021.

[11] Stephen Casper, Carson Ezell, Charlotte Siegmann, Noam Kolt, Taylor Lynn Curtis, Benjamin Bucknall, Andreas Haupt, Kevin Wei, Jer´ emy´ Scheurer, Marius Hobbhahn, et al. Black-box access is insufficient for rigorous AI audits. In Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, pages 2254–2272, 2024.

[12] Maria De-Arteaga, Alexey Romanov, Hanna Wallach, Jennifer Chayes, Christian Borgs, Alexandra Chouldechova, Sahin Geyik, Krishnaram Kenthapadi, and Adam Tauman Kalai. Bias in bios: A case study of semantic representation bias in a high-stakes setting. In proceedings of the Conference on Fairness, Accountability, and Transparency, pages 120–128, 2019.

[13] Martijn de Vos, Akash Dhasade, Jade Garcia Bourree, Anne-Marie´ Kermarrec, Erwan Le Merrer, Benoˆıt Rottembourg, and Gilles Tredan. Fairness auditing with multi-agent collaboration. In ECAI 2024: 27th European Conference on Artificial Intelligence, 19–24 October 2024, Santiago de Compostela, Spain–Including 13th Conference on Prestigious Applications ofIntelligent Systems (PAIS 2024), pages 1116–1123. SAGE Publications Pvt. Ltd 1 Oliver’s Yard, 55 City Road, London, EC1Y 1SP, 2024.

[14] Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. Retiring Adult: New datasets for fair machine learning. Advances in Neural Information Processing Systems, 34, 2021.

[15] John R Douceur. The sybil attack. In International workshop on peerto-peer systems, pages 251–260. Springer, 2002.

[16] Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness through awareness. In Proceedings of the 3rd innovations in theoretical computer science conference, pages 214–226, 2012.

[17] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of cryptography conference, pages 265–284. Springer, 2006.

[18] European Parliament and Council of the European Union. Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (artificial intelligence act). Official Journal of the European Union, 2024. http://data.europa.eu/eli/reg/2024/1689/oj.

[19] Kazuto Fukuchi, Satoshi Hara, and Takanori Maehara. Faking fairness via stealthily biased sampling. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 412–419, 2020.

[20] Jade Garcia Bourree, Augustin Godinot, Sayan Biswas, Anne-Marie´ Kermarrec, Erwan Le Merrer, Gilles Tredan, Martijn De Vos, and Milos Vujasinovic. Robust ML auditing using prior knowledge. In Proceedings of the 42nd International Conference on Machine Learning, ICML’25. JMLR.org, 2025.

[21] Jade Garcia Bourree, Erwan Le Merrer, Gilles Tredan, and Beno´ ˆıt Rottembourg. Leveraging imperfect sources to detect fairwashing in black-box auditing. Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 2026.

[22] Renaud Gaucher, Aymeric Dieuleveut, and Hadrien Hendrikx. Unified breakdown analysis for Byzantine robust gossip. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research. PMLR, 13–19 Jul 2025.

[23] Augustin Godinot, Sofiane Azogagh, Julien Ferry, and Sebastien Gambs.´ Manipulation-proof oblivious audits against deceptive model providers. arXiv preprint arXiv:2608.04365, 2026.

[24] Augustin Godinot, Erwan Le Merrer, Gilles Tredan, Camilla Penzo, and´ Franc¸ois Ta¨ıani. Under manipulations, are some AI models harder to audit? In 2024 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pages 644–664. IEEE, 2024.

[25] Muhammed Golec and Maha AlabdulJalil. Interpretable LLMs for credit risk: A systematic review and taxonomy. Expert Systems with Applications, page 130941, 2025.

[26] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

[27] Neel Guha, Julian Nyarko, Daniel Ho, Christopher Re, Adam Chilton,´ Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel Rockmore, Diego Zambrano, et al. Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models. Advances in neural information processing systems, 36:44123–44279, 2023.

[28] Moritz Hardt, Eric Price, and Nati Srebro. Equality of opportunity in supervised learning. Advances in neural information processing systems, 2016.

[29] David Hartmann, Lena Pohlmann, Lelia Hanslik, Noah Gießing, Bettina Berendt, and Pieter Delobelle. Audit me if you can: Query-efficient active fairness auditing of black-box LLMs. In Findings of the Association for Computational Linguistics: ACL 2026, pages 33673–33698, 2026.

[30] Lie He, Sai Praneeth Karimireddy, and Martin Jaggi. Byzantinerobust decentralized learning via clippedgossip. arXiv preprint arXiv:2202.01545, 2022.

[31] Disi Ji, Padhraic Smyth, and Mark Steyvers. Can I trust my fairness metric? assessing fairness with unlabeled data and bayesian inference. Advances in Neural Information Processing Systems, 33:18600–18612, 2020.

[32] Sai Praneeth Karimireddy, Lie He, and Martin Jaggi. Learning from history for byzantine robust optimization. In International conference on machine learning, pages 5311–5319. PMLR, 2021.

[33] Sai Praneeth Karimireddy, Lie He, and Martin Jaggi. Byzantinerobust learning on heterogeneous datasets via bucketing. In The Tenth International Conference on Learning Representations ICLR 2022, apr 2022.

[34] Deepak Kumar, Yousef Anees AbuHashem, and Zakir Durumeric. Watch your language: Investigating content moderation with large language models. In Proceedings of the International AAAI Conference on Web and Social Media, volume 18, pages 865–878, 2024.

[35] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.

[36] Valentin Lafargue, Adriana Laurindo Monteiro, Emmanuelle Claeys, Laurent Risser, and Jean-Michel Loubes. Exposing the illusion of fairness: Auditing vulnerabilities to distributional manipulation attacks. Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 2026.

[37] Sean McGregor. Preventing repeated real world AI failures by cataloging incidents: The AI incident database. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 15458–15463, 2021.

[38] New York City Department of Consumer and Worker Protection. Automated employment decision tools (AEDT). https://www.nyc.gov/site/ dca/about/automated-employment-decision-tools.page, 2023. Local Law 144 of 2021; enforcement began July 5, 2023. Accessed: 2026-09-26.

[39] Qwen Team, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

[40] Bashir Rastegarpanah, Krishna Gummadi, and Mark Crovella. Auditing black-box prediction models for data minimization compliance. Advances in Neural Information Processing Systems, 34:20621–20632, 2021.

[41] Nicola Rieke, Jonny Hancox, Wenqi Li, Fausto Milletari, Holger R Roth, Shadi Albarqouni, Spyridon Bakas, Mathieu N Galtier, Bennett A Landman, Klaus Maier-Hein, et al. The future of digital health with federated learning. NPJ digital medicine, 3(1):119, 2020.

[42] Ali Shahin Shamsabadi, Mohammad Yaghini, Natalie Dullerud, Sierra Wyllie, Ulrich A¨ıvodji, Aisha Alaagib, Sebastien Gambs, and Nicolas´ Papernot. Washing the unwashable: On the (im)possibility of fairwashing detection. Advances in Neural Information Processing Systems, 35:14170–14182, 2022.

[43] Ali Shahin Shamsabadi, Sierra Calanda Wyllie, Nicholas Franzese, Natalie Dullerud, Sebastien Gambs, Nicolas Papernot, Xiao Wang,´ and Adrian Weller. Confidential-PROFITT: Confidential proof of fair training of trees. In The Eleventh International Conference on Learning Representations, 2022.

[44] Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez, Ting Fang Tan, and Daniel Shu Wei Ting. Large language models in medicine. Nature medicine, 29(8):1930–1940, 2023.

[45] Ehsan Toreini, Maryam Mehrnezhad, and Aad van Moorsel. Fairness as a Service (FaaS): verifiable and privacy-preserving fairness auditing of machine learning systems. International Journal of Information Security, 23(2):981–997, 2024.

[46] Amaury Trujillo, Tiziano Fagni, and Stefano Cresci. The dsa transparency database: Auditing self-reported moderation actions by social media. Proceedings of the ACM on Human-Computer Interaction, 9(2):1–28, 2025.

[47] Yueqi Xie, Minghong Fang, and Neil Zhenqiang Gong. Model poisoning attacks to federated learning via multi-round consistency. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15454–15463. IEEE, 2025.

[48] Xinyi Xu, Zhaoxuan Wu, Arun Verma, Chuan Sheng Foo, and Bryan Kian Hsiang Low. FAIR: Fair collaborative active learning with individual rationality for scientific discovery. In International Conference on Artificial Intelligence and Statistics, pages 4033–4057. PMLR, 2023.

[49] Chhavi Yadav, Amrita Roy Chowdhury, Dan Boneh, and Kamalika Chaudhuri. FairProof : Confidential and certifiable fairness for neural networks. In Proceedings of the 41st International Conference on Machine Learning, pages 55682–55705, 21–27 Jul 2024.

[50] Dong Yin, Yudong Chen, Kannan Ramchandran, and Peter Bartlett. Byzantine-robust distributed learning: Towards optimal statistical rates. In International conference on machine learning, pages 5650–5659. PMLR, 2018.

[51] Zaixi Zhang, Xiaoyu Cao, Jinyuan Jia, and Neil Zhenqiang Gong. FLDetector: Defending federated learning against model poisoning attacks via detecting malicious clients. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining, pages 2545–2555, 2022.

## APPENDIX A NOTATIONS

TABLE III NOTATIONS USED THROUGHOUT THE PAPER.
<table><tr><td>Symbol</td><td>Meaning</td><td>Defined in</td></tr><tr><td>Protocol</td><td></td><td></td></tr><tr><td> $f$ </td><td>black-box LLM under audit</td><td>Sec. II-A</td></tr><tr><td> $\bar { \mathsf { A } } , \mathsf { B }$ </td><td>the two demographic groups</td><td>Sec. II-A</td></tr><tr><td> $\tau$ </td><td>half-width of the fair band  $[ - \tau , \tau ]$ </td><td>Sec. II-A</td></tr><tr><td> $N , K$ </td><td>number of honest auditors and of adversaries</td><td>Sec. III-B</td></tr><tr><td> $\nu$ </td><td>set of all auditors,  $| \mathcal { V } | = N + K$ </td><td>Sec. III-B</td></tr><tr><td> $\mathcal { G } , \mathcal { N } ( i )$ </td><td>communication graph (static, connected); neighbors of auditor i</td><td>Sec. III-B</td></tr><tr><td> $\mathcal { D } _ { i }$ </td><td>private query set of auditor i</td><td>Sec. II-B</td></tr><tr><td> $T , t$ </td><td>number of rounds; round index</td><td>Sec. III-B</td></tr><tr><td> $n$ </td><td>per-round query budget of every auditor</td><td>Sec. III-C</td></tr><tr><td> $Q$   $\mathcal { M } _ { i }$ </td><td>queries sampled by an auditor in one round,  $| Q | = n$ </td><td>Sec. III-C</td></tr><tr><td></td><td>table of auditor  $i ; \mathcal { M } _ { i } [ j ] = ( c , r )$  : latest LAS vector of j known by i, and the round r in which j produced it</td><td>Sec. III-C</td></tr><tr><td>Statistics</td><td></td><td></td></tr><tr><td> $A , B$ </td><td>number of queries in group A, B</td><td>Sec. II-B</td></tr><tr><td> $a , b$ </td><td>number of positive decisions among them</td><td>Sec. II-B</td></tr><tr><td> $c _ { i } = ( a _ { i } , A _ { i } , b _ { i } , B _ { i } )$ </td><td>LAS vector of auditor ¿, cumulative over rounds</td><td>Sec. II-B</td></tr><tr><td> $C = ( a , A , b , B )$ </td><td>GAS vector, sum of LAS vectors; in Sec. IV, the attacker&#x27;s view of it</td><td>Eq. (3), (5)</td></tr><tr><td> $\delta = ( \dot { \Delta } a , \Delta A , \Delta b , \Delta B )$ </td><td>increment of a LAS vector: fabricated in one round by an adversary; for honest auditor i,  $\mathcal { M } _ { i } [ j ] . c - \mathrm { p r e v } _ { j }$  , which spans several rounds under gossip delays</td><td>Sec. III-C, IV-A</td></tr><tr><td>α</td><td>concentration of the Dirichlet partition (lower = more heterogeneous)</td><td>Sec. II-B</td></tr><tr><td>Demographic parity</td><td></td><td></td></tr><tr><td> $\operatorname { D P } ( f { \bar { ) } }$ </td><td>true DP of f on the population, equal to the DP of the honest auditors’GAS vector</td><td>Eq. (1), Sec. III-B</td></tr><tr><td> $\mathrm { D P } ( c )$ </td><td> $a / A - b / B \bar { , }$  for any vector  $c = ( \bar { a } , A , b , B )$  with  $A , B > 0$ </td><td>Eq. (2)</td></tr><tr><td> $\mathrm { D P } ( { \dot { C } } )$ </td><td>estimate from the GAS vector</td><td>Eq. (3)</td></tr><tr><td> ${ \widehat { \mathrm { D P } } } _ { i }$ </td><td>defended estimate of honest auditor ¿</td><td>Sec. V-D</td></tr><tr><td>DP</td><td>median of the final estimates of all  $N + K$  auditors</td><td>Sec. VI</td></tr><tr><td>Adversary</td><td></td><td></td></tr><tr><td> $\mathcal { X } _ { n }$ </td><td>admissible increments</td><td>Eq. (4)</td></tr><tr><td> $\delta ^ { \star }$ </td><td>best response of the adversary</td><td>Eq. (7)</td></tr><tr><td>y</td><td>budget split to group A (real variable; its integer values are the splits ∆A)</td><td>Sec. IV-B</td></tr><tr><td> $\varphi ( y )$ </td><td>attacker&#x27;s objective under extreme labels</td><td>Eq. (10)</td></tr><tr><td>Defense</td><td></td><td></td></tr><tr><td> $p r e v _ { j }$ </td><td>LAS vector of j held by i at the end of the previous round</td><td>Sec. III-C</td></tr><tr><td> $z ( s ; \dot { m } , p )$ </td><td>binomial z-score of s successes in m trials with probability p</td><td>Sec. V-B</td></tr><tr><td> $\hat { \pi } , \hat { q } _ { \mathsf { A } } , \hat { q } _ { \mathsf { B } }$ </td><td>KT estimates of a source&#x27;s proportion of A and of its positive rates</td><td>Alg. 3</td></tr><tr><td> $z _ { 1 } , z _ { 2 } , z _ { 3 }$ </td><td>z-scores of the group proportion and of the rates in A and B</td><td>Alg. 3</td></tr><tr><td> $S , z _ { \mathrm { m a x } }$ </td><td>suspicion score of an increment,  $S \in [ 0 , z _ { \operatorname* { m a x } } ] ;$  score cap</td><td>Alg. 3</td></tr><tr><td> $w$ </td><td>locked weight of an increment: 1 for a source&#x27;s first increment,  $( 1 + S ) ^ { - 2 }$ </td><td>Sec. V-C</td></tr><tr><td> $W _ { j }$ </td><td>otherwise weighted LAS vector of j held by  $i \colon j ^ { \flat } \mathbf { s }$  increments, each multiplied by its weight (equals  $j ^ { \prime } s \bar { \mathrm { \ L A S } }$  vector when all weights are 1)</td><td>Sec. V-C</td></tr><tr><td>Experiments</td><td></td><td></td></tr><tr><td> $\beta$ </td><td>trimming fraction of the trimmed mean</td><td>Sec. VI</td></tr><tr><td>Analysis (Appendix B)</td><td></td><td></td></tr><tr><td> $I _ { + } , I _ { - }$ </td><td>integer splits with  $\varphi > - \tau , \mathrm { r e s p . } \ \varphi < \tau$ </td><td>App. B</td></tr><tr><td> $\Gamma$ </td><td>quadratic  $- ( 1 + \tau ) y ^ { 2 } + \gamma _ { 1 } y + \gamma _ { 0 } ,$  of the sign of  $\varphi + \tau$ </td><td>Eq. (12)</td></tr><tr><td> $\gamma _ { 1 } , \gamma _ { 0 }$ </td><td>coefficient of y and constant term of Γ</td><td>App. B-B</td></tr><tr><td> $\Lambda$ </td><td>discriminant of Γ</td><td>App. B-B</td></tr><tr><td> $\rho _ { 1 } < \rho _ { 2 }$ </td><td>roots of Γ</td><td>App. B-B</td></tr><tr><td> $y ^ { - } , y ^ { + }$ </td><td>smallest and largest integer splits in  $\left( \rho _ { 1 } , \rho _ { 2 } \right) \cap [ 0 , n ] ; I _ { + } = \{ y ^ { - } , \ldots , y ^ { + } \}$ </td><td>Lemma 3</td></tr></table>

## APPENDIX B PROOFS OF SECTION IV

Throughout, $C = ( a , A , b , B )$ is the attacker’s GAS vector, with $A > 0$ and $B > 0$ (Section IV-B). Every LAS vector, honest or fabricated, counts at most as many positives as queries in each group (for fabricated ones, by definition of X ${ \mathcal { X } } _ { n } ,$ Section $\mathrm { I V } { \cdot } \mathrm { A } )$ , so $0 \leq a \leq A$ and $0 \leq b \leq B$ . Recall from (10) that, for $y \in [ 0 , n ]$ 4

$$
\varphi ( y ) = \frac { a + y } { A + y } - \frac { b } { B + n - y } ,
$$

where both denominators are positive. The integer values of $y$ are the splits $\Delta A \in \{ 0 , \ldots , n \}$ of Section IV-B.

a) Roadmap: Theorem 2 asks for an integer $y \in \{ 0 , \ldots , n \}$ with $- \tau < \varphi ( y ) < \tau$ . We treat the two inequalities separately, through the sets

$$
I _ { + } = \{ y \in \{ 0 , \ldots , n \} : \varphi ( y ) > - \tau \} , \qquad I _ { - } = \{ y \in \{ 0 , \ldots , n \} : \varphi ( y ) < \tau \} ,
$$

so that the theorem asks whether $I _ { + } \cap I _ { - } \neq \emptyset$ . Appendix B-B describes $I _ { + }$ exactly, through a quadratic. Appendix B-C shows that, by concavity, I<sub>−</sub> is nonempty if and only if it contains 0 or n. Appendix B-D shows that $\varphi$ moves by small steps, so that the sequence $\left( \varphi ( y ) \right) _ { y }$ cannot jump over the fair band $( - \tau , \tau )$ . Appendix B-E combines the three. We first prove Lemma 1, which reduces the attacker’s problem to $\varphi .$

## A. Proof of Lemma 1

Fix a split $\Delta A \in \{ 0 , \ldots , n \}$ , so that $\Delta B = n - \Delta A$ . By (9), $\mathrm { D P } ( C + \delta )$ is affine in $\Delta a$ and in $\Delta b .$ . Increasing $\Delta a$ by one, all else fixed, changes it by

$$
\frac { 1 } { A + \Delta A } > 0 , \qquad \mathrm { a n d ~ i n c r e a s i n g ~ } \Delta b \mathrm { ~ b y ~ o n e ~ c h a n g e s ~ i t ~ b y ~ } \qquad - \frac { 1 } { B + n - \Delta A } < 0 .
$$

Hence $\mathrm { D P } ( C + \delta )$ is strictly increasing in $\Delta a$ and strictly decreasing in $\Delta b .$ Let $\delta = ( \Delta a , \Delta A , \Delta b , \Delta B ) \in \mathcal { X } _ { n }$ . If $\Delta a < \Delta A ,$ raising $\Delta a$ by one keeps $\delta$ in $\mathcal { X } _ { n }$ and strictly increases $\mathrm { D P } ( C + \delta ) ; \mathrm { i f } \ \Delta b > 0 ,$ , lowering $\Delta b$ by one does the same. So every maximizer satisfies $\Delta a = \Delta A$ and $\Delta b \ = \ 0$ . Such a move is $\delta = ( \Delta A , \Delta A , 0 , n - \Delta A )$ , and by (10) it satisfies $\operatorname { D P } ( C + \delta ) = \varphi ( \Delta A )$ , hence

$$
\operatorname* { m a x } _ { \delta \in { \mathcal X } _ { n } } \mathrm { D P } ( C + \delta ) = \operatorname* { m a x } _ { y \in \{ 0 , \ldots , n \} } \varphi ( y ) .
$$

## B. Where $\varphi > - \tau ,$ , the quadratic Γ

a) Reduction to a sign: To locate the splits y with $\varphi ( y ) > - \tau$ , we reduce $\varphi ( y ) + \tau { \mathrm { ~ t o ~ } }$ the common denominator $( A + y ) ( B + n - y )$ . Multiplying the first term of $\varphi$ by $B + n - y$ , the second by $A + y$ , and τ by both, we get

$$
\varphi ( y ) + \tau = { \frac { \Gamma ( y ) } { ( A + y ) ( B + n - y ) } } , \qquad \Gamma ( y ) = ( a + y ) ( B + n - y ) - b ( A + y ) + \tau ( A + y ) ( B + n - y ) .\tag{12}
$$

The denominator is positive on $[ 0 , n ]$ , so $\varphi ( y ) + \tau$ has the sign of $\Gamma ( y )$ :

$$
\varphi ( y ) > - \tau \iff \Gamma ( y ) > 0 .\tag{13}
$$

b) Expanding Γ: The three terms of $\Gamma ( y )$ expand as

$$
\begin{array} { c } { { ( a + y ) ( B + n - y ) = - y ^ { 2 } + \left( B + n - a \right) y + a \left( B + n \right) , } } \\ { { - b \left( A + y \right) = - b y - b A , } } \\ { { \tau \left( A + y \right) ( B + n - y ) = - \tau y ^ { 2 } + \tau \left( B + n - A \right) y + \tau A \left( B + n \right) . } } \end{array}
$$

Collecting the coefficients of $y ^ { 2 }$ , of $y$ and the constant terms shows that Γ is a quadratic in $y \colon$

$$
\Gamma ( y ) = - ( 1 + \tau ) y ^ { 2 } + \gamma _ { 1 } y + \gamma _ { 0 } ,
$$

where

$$
\begin{array} { l } { { \gamma _ { 1 } = \left( B + n - a - b \right) + \tau \left( B + n - A \right) , } } \\ { { \gamma _ { 0 } = a \left( B + n \right) - b A + \tau A \left( B + n \right) . } } \end{array}
$$

c) Roots: The leading coefficient of Γ is $- ( 1 + \tau ) < 0 ,$ so Γ is positive exactly between its roots, if it has two distinct ones. Its discriminant is

$$
\Lambda = \gamma _ { 1 } ^ { 2 } + 4 ( 1 + \tau ) \gamma _ { 0 } .
$$

If $\Lambda \leq 0$ , Γ is positive nowhere. If $\Lambda > 0$ , the quadratic formula gives the two roots

$$
\rho _ { 1 } = \frac { \gamma _ { 1 } - \sqrt { \Lambda } } { 2 ( 1 + \tau ) } < \rho _ { 2 } = \frac { \gamma _ { 1 } + \sqrt { \Lambda } } { 2 ( 1 + \tau ) } ,
$$

and $\Gamma ( y ) > 0$ if and only $\mathrm { i f } ~ y \in \mathsf { \Gamma } \left( \rho _ { 1 } , \rho _ { 2 } \right)$

d) Integer splits: The splits are the integers of $[ 0 , n ]$ , so we need the integers of $( \rho _ { 1 } , \rho _ { 2 } ) \cap [ 0 , n ]$ . The smallest integer strictly greater than $\rho _ { 1 } \ \mathrm { i s } \ \lfloor \rho _ { 1 } \rfloor + 1$ , and the largest integer strictly smaller than $\rho _ { 2 }$ is $\lceil \rho _ { 2 } \rceil - 1$ . Clipping them to $[ 0 , n ]$ , we set

$$
y ^ { - } = \operatorname* { m a x } \big ( 0 , ~ \lfloor \rho _ { 1 } \rfloor + 1 \big ) , \qquad y ^ { + } = \operatorname* { m i n } \big ( n , ~ \lceil \rho _ { 2 } \rceil - 1 \big ) ,
$$

with the convention that $\{ y ^ { - } , \dots , y ^ { + } \} = \emptyset { \mathrm { ~ i f ~ } } y ^ { - } > y ^ { + }$ . Combining with (13), we have shown:

Lemma 3 (Where $\varphi > - \tau ) . \ I _ { + } = \{ y ^ { - } , \ldots , y ^ { + } \} \ \mathrm { i f } \ \Lambda > 0 ,$ , and $I _ { + } = \emptyset$ otherwise. In particular, $I _ { + }$ is a set of consecutive integers, and condition 2 of Theorem 2 holds if and only if $\Lambda > 0$ and $y ^ { - } \leq y ^ { + }$

## C. Where $\varphi < \tau ,$ concavity

Lemma 4 (Concavity and endpoints). φ is concave on $[ 0 , n ]$ , hence $\varphi ( y ) \geq \operatorname* { m i n } \left( \varphi ( 0 ) , \varphi ( n ) \right)$ for every $y \in [ 0 , n ]$ . In particular, $I _ { - } \neq \emptyset$ if and only if min $( \varphi ( 0 ) , \varphi ( n ) ) < \tau$ . Moreover,

$$
\varphi ( 0 ) = \mathrm { D P } ( C ) + \frac { n b } { B ( B + n ) } , \qquad \varphi ( n ) = \mathrm { D P } ( C ) + \frac { n \left( A - a \right) } { A ( A + n ) } .
$$

Proof. Writing $\begin{array} { r } { \frac { a + y } { A + y } = 1 - \frac { A - a } { A + y } } \end{array}$ , we have $\begin{array} { r } { \varphi ( y ) = 1 - \frac { A - a } { A + y } - \frac { b } { B + n - y } } \end{array}$ , hence

$$
\varphi ^ { \prime } ( y ) = { \frac { A - a } { ( A + y ) ^ { 2 } } } - { \frac { b } { ( B + n - y ) ^ { 2 } } } , \qquad \varphi ^ { \prime \prime } ( y ) = - { \frac { 2 ( A - a ) } { ( A + y ) ^ { 3 } } } - { \frac { 2 b } { ( B + n - y ) ^ { 3 } } } \leq 0 ,
$$

since $A - a \geq 0 , b \geq 0$ and both denominators are positive. Every $y \in [ 0 , n ]$ is $y = \lambda n$ with $\lambda \in [ 0 , 1 ] ,$ , and concavity gives $\varphi ( y ) \geq ( 1 - \lambda ) \varphi ( 0 ) + \lambda \varphi ( n ) \geq \operatorname* { m i n } \left( \varphi ( 0 ) , \varphi ( n ) \right)$ . So if both endpoints are at least τ , then $\varphi \geq \tau$ everywhere and $I _ { - } = \varnothing ;$ conversely, if one endpoint is below τ, it belongs to $I _ { - }$ . For the endpoint values, using $\mathrm { D P } ( C ) = a / A - b / B$

$$
\varphi ( 0 ) - \mathrm { D P } ( C ) = \frac { b } { B } - \frac { b } { B + n } = \frac { n b } { B ( B + n ) } , \qquad \varphi ( n ) - \mathrm { D P } ( C ) = \frac { a + n } { A + n } - \frac { a } { A } = \frac { n ( A - a ) } { A ( A + n ) } .
$$

In practice, condition 1 holds automatically once the GAS vector is large compared with the attacker’s budget.

Lemma 5 (Condition 1 in practice). If $\mathrm { D P } ( C ) < - \tau$ and $\operatorname* { m a x } ( A , B ) \geq n \left( 1 / ( 2 \tau ) - 1 \right)$ , then condition 1 of Theorem 2 holds. Proof. By Lemma 4, condition 1 reads min $\big ( \varphi ( 0 ) - \mathrm { D P } ( C ) , ~ \varphi ( n ) - \mathrm { D P } ( C ) \big ) < \tau - \mathrm { D P } ( C )$ . Since $b \leq B$ and $A - a \leq A .$ the endpoint values of Lemma 4 give

$$
\varphi ( 0 ) - \mathrm { D P } ( C ) = \frac { n b } { B ( B + n ) } \leq \frac { n } { B + n } , \qquad \varphi ( n ) - \mathrm { D P } ( C ) = \frac { n \left( A - a \right) } { A ( A + n ) } \leq \frac { n } { A + n } ,
$$

so the left-hand side is at most $\frac { n } { \operatorname* { m a x } ( A , B ) + n }$ . This is at most $2 \tau$ if and only if max $( A , B ) \geq n ( 1 - 2 \tau ) / ( 2 \tau ) = n \left( 1 / ( 2 \tau ) - 1 \right)$ which holds by assumption. Finally, $\tau - \operatorname { D P } ( C ) > 2 \tau$ since $\mathrm { D P } ( C ) < - \tau$

## D. Small jumps

Lemma 6 (Small jumps). For every $y \in \{ 0 , \ldots , n - 1 \}$

$$
| \varphi ( y + 1 ) - \varphi ( y ) | \leq { \frac { 1 } { \operatorname* { m i n } ( A , B ) + 1 } } .
$$

Proof. We compute the variation of each term of φ when y increases by one. For the first term,

$$
{ \frac { a + y + 1 } { A + y + 1 } } - { \frac { a + y } { A + y } } = { \frac { ( a + y + 1 ) ( A + y ) - ( a + y ) ( A + y + 1 ) } { ( A + y ) ( A + y + 1 ) } } = { \frac { A - a } { ( A + y ) ( A + y + 1 ) } } = : u _ { y } ,
$$

since the numerator simplifies to $( A + y ) - ( a + y ) = A - a$ . For the second term,

$$
{ \frac { b } { B + n - y - 1 } } - { \frac { b } { B + n - y } } = { \frac { b } { ( B + n - y - 1 ) ( B + n - y ) } } = : v _ { y } .
$$

Hence $\varphi ( y + 1 ) - \varphi ( y ) = u _ { y } - v _ { y }$ with $u _ { y } , v _ { y } \ge 0 .$ so that $| \varphi ( y + 1 ) - \varphi ( y ) | \leq \operatorname* { m a x } ( u _ { y } , v _ { y } )$ . The denominator of $u _ { y }$ grows with y, so $u _ { y }$ is largest at $y = 0 ,$ , where $\begin{array} { r } { u _ { 0 } = \frac { A - a } { A ( A + 1 ) } \le \frac { 1 } { A + 1 } } \end{array}$ since $A - a \leq A$ . The denominator of $v _ { y }$ decreases with y, so $v _ { y }$ is largest at $y = n - 1$ , where $\begin{array} { r } { v _ { n - 1 } = \frac { b } { B ( B + 1 ) } \le \frac { 1 } { B + 1 } } \end{array}$ since $b \leq B$

## E. Proof of Theorem 2

Proof of Theorem 2. By the roadmap, we must show that $I _ { + } \cap I _ { - } \neq \emptyset$ if and only if conditions 1 and 2 hold.

Necessity. Let $y \in I _ { + } \cap I _ { - }$ . Then $I _ { - } \neq \emptyset$ , which is condition 1 by Lemma 4, and $y \in I _ { + }$ is condition 2. This direction doe not use (11).

Sufficiency. $I _ { + }$ is nonempty by condition 2, and $I _ { - }$ is nonempty by condition 1 and Lemma 4. Assume by contradiction that $I _ { + } \cap I _ { - } = \emptyset$ . Then $\varphi \geq \tau \ \mathrm { o n } \ I _ { + }$ , and $I _ { + } \ne \{ 0 , \ldots , n \}$ since I is nonempty. As $I _ { + }$ is a set of consecutive integers (Lemma 3), there exist two neighbors $y \in I _ { + }$ and $y ^ { \prime } \notin I _ { + } , | y - y ^ { \prime } | = 1$ , with $\varphi ( y ) \geq \tau$ and $\varphi ( y ^ { \prime } ) \leq - \tau$ . Hence

$$
| \varphi ( y ) - \varphi ( y ^ { \prime } ) | \geq 2 \tau > \frac { 1 } { \operatorname* { m i n } ( A , B ) + 1 }
$$

by (11), which contradicts Lemma 6.

Corollary 7 (The attacker reaches the fair band). Under the assumptions of Theorem 2, if conditions 1 and 2 hold, the attacker (7) satisfies $| \operatorname { D P } ( C + \delta ^ { \star } ) | < \tau$

Proof. By Theorem 2, there exists $y \in \{ 0 , \ldots , n \}$ with $| \varphi ( y ) | < \tau$ . The extreme-label move $\delta = ( y , y , 0 , n - y )$ belongs to ${ \mathcal { X } } _ { n }$ and satisfies $\mathrm { D P } ( C + \delta ) = \varphi ( y )$ . Since $\delta ^ { \star }$ minimizes $| \operatorname { D P } ( C + \delta ) |$ over ${ \mathcal { X } } _ { n }$ , we $\mathrm { g e t } \ | \operatorname { D P } ( C + \delta ^ { \star } ) | \leq | \varphi ( y ) | < \tau$

You are a content moderation system. Decide whether the following comment is toxic: rude, disrespectful or unreasonable language that is likely to make someone leave a discussion. Reply with JSON of the form "items": ["yes": Y, "no": N], where Y is your confidence in percent that the comment is toxic, N your confidence that it is not, and Y + N = 100.

Fig. 8. Prompt used to classify each comment in Civil Comments with QWEN2.5-7B-INSTRUCT.  
Prompt: occupation classification (Bias in Bios)   
You are an occupation classification system. You are given a short biography of a person. Answer   
with the person’s occupation, chosen from this list: accountant, architect, attorney, chiropractor,   
comedian, composer, dentist, dietitian, dj, filmmaker, journalist, model, nurse, painter, paralegal,   
pastor, photographer, physician, poet, professor, psychologist, software\_engineer, surgeon, teacher,   
yoga\_teacher. Reply with the occupation name only, exactly as written in the list.   
Biography: {biography}   
Occupation:  
Fig. 9. Prompt used to classify each biography in Bias in Bios with LLAMA-3.1-8B-INSTRUCT. {biography} is replaced by the biography text. The answer is constrained to the 25 occupation names in the list.

## APPENDIX C ADDITIONAL EXPERIMENTAL DETAILS

This appendix complements the setup of Section VI-A with additional details: how we obtain the audited LLMs’ predictions on each dataset (Appendices C-A and C-B), how we partition the inputs across auditors, and our implementation and hardware specifications. In both datasets we use the audited LLM as a zero-shot classifier. Every input is sent as a separate request, with no in-context examples and no fine-tuning.

## A. Civil Comments dataset

The audited LLM is QWEN2.5-7B-INSTRUCT, which we ask whether each comment is toxic. A positive prediction means the LLM flags the comment as toxic. The LLM’s answer could not be parsed for 4 of the 182,176 comments, and we drop these, leaving 182,172 comments. Each comment is annotated with one of 16 identity groups. Figure 8 shows the prompt.

## B. Bias in Bios dataset

Data. We use the LabHC release of Bias in Bios [12], stacking the train, test, and dev splits in that order (257,478 + 99,069 + 39,642 biographies). We keep the same 25 occupations as [29] and drop interior designer, personal trainer, and rapper. This leaves 391,895 biographies. Each biography has a binary gender label.

LLM and prompt. The audited LLM is LLAMA-3.1-8B-INSTRUCT, served with vLLM. Each biography is sent as a single user message formatted with the LLM’s chat template (Figure 9). We add no separate system prompt, so that the same message can be used with every LLM we evaluate.

Decoding. We decode greedily (temperature 0, fixed seed) with at most 16 new tokens. We use vLLM’s structured outputs to restrict the answer to exactly one of the 25 occupation names. Every biography therefore gets a valid label, and unlike in Civil Comments, no input is dropped. The audit uses only the predicted label; we do not ask the LLM for per-class scores.

Audits. We group the 25 occupations into sectors following the U.S. Standard Occupational Classification. An audit fixes a sector and a target occupation within it: its population consists of all biographies whose true occupation lies in that sector, and a prediction is positive if and only if the LLM predicts the target occupation (Table I). The audit thus measures whether, among people working in the same sector, the LLM assigns the target occupation at different rates to men and women.

## C. Data partition

For every combination of group and predicted label, the inputs are split across auditors with proportions drawn from a Dirichlet distribution of concentration α. We redraw the partition if any auditor holds fewer than 30 inputs of either group. The resulting local estimates are strongly heterogeneous: across cases, the 5th percentile of an auditor’s local DP ranges from −0.55 to −0.84, and the 95th percentile from +0.08 to +0.66.

## D. Implementation details and hardware

We implement AUDITOPUS and the audit simulation in Python using the NUMPY and PANDAS libraries. The simulation is fully seeded. Each run is determined by its configuration and seed, so every result can be reproduced exactly. The audited LLMs’ predictions are computed once, before the audits. We serve the LLMs with VLLM [35]. We generate the predictions using NVIDIA A100 GPUs (one model replica per GPU) available on our compute cluster. The audit simulations all run on CPUs.

TABLE IV  
THE ATTACK SUCCESS RATE (ASR) AND MEAN ABSOLUTE DP ESTIMATION ERROR ON THE CIVIL COMMENTS DATASET, OVER EVERY RUN WITH 1–19 ADVERSARIES (5–49% OF ALL AUDITORS; 380 RUNS PER CASE; AS TABLE II). BOLD: THE LOWEST IN EACH ROW.
<table><tr><td colspan="5">Attack success rate (%)</td><td colspan="4">Average DP estimation error</td></tr><tr><td>LLM fairness</td><td>No-defense</td><td>Median</td><td>Trimmed mean</td><td>AUDITOPUS</td><td>No-defense</td><td>Median</td><td>Trimmed mean</td><td>AUDITOPUS</td></tr><tr><td>Very unfair</td><td>95</td><td>42</td><td>63</td><td>0</td><td>0.221</td><td>0.226</td><td>0.221</td><td>0.050</td></tr><tr><td>Moderate</td><td>99</td><td>33</td><td>35</td><td>0</td><td>0.186</td><td>0.235</td><td>0.219</td><td>0.022</td></tr><tr><td>Near-fair</td><td>97</td><td>68</td><td>80</td><td>78</td><td>0.061</td><td>0.081</td><td>0.073</td><td>0.028</td></tr><tr><td>All three cases</td><td>97</td><td>48</td><td>59</td><td>26</td><td>0.156</td><td>0.180</td><td>0.171</td><td>0.034</td></tr></table>

![](images/c71c045b109f69e2ffaa945bab249ce842a4f17f1a7dbe202737f5214ba0f8fc.jpg)  
Number of adversaries (on top of N = 20 honest auditors)  
Fig. 10. The attack success rate (top row) and DP estimation error (in the bottom row) for Civil Comments, while varying the number of adversaries K from 0 to 19. We consider a very unfair (left column), moderate (middle column) and near-fair LLM (right column). The dashed horizontal line in Figure 10 (in the bottom row) corresponds to a DP value of 0.

## APPENDIX D

## ADDITIONAL EXPERIMENTS

We complement Section VI with additional experiments that answer the following questions:

• Generalization (RQ3). How accurately do AUDITOPUS and baselines preserve the audit’s verdict and its DP estimate on the Civil Comments dataset as the proportion of adversaries grows (Appendix D-A)?

• Query sampling (RQ4). How does querying with replacement, instead of without replacement, affect the accuracy of an audit with honest auditors only, and how does this depend on data heterogeneity (Appendix D-B)?

• Baseline configuration (RQ5). Do the robust aggregation baselines become robust when they aggregate each auditor’s local DP instead of per-group positive rates, or when we change the trimming fraction β of the trimmed mean (Appendix D-C)?

• Query budget (RQ6). How does the per-round query budget, which sets the size of each update that AUDITOPUS scores, affect the accuracy and robustness of AUDITOPUS and the baselines (Appendix D-D)?

## A. RQ3: Effectiveness of AUDITOPUS and baselines on Civil Comments Dataset

We repeat the effectiveness experiment of Section VI-B on a second model and dataset: QWEN2.5-7B-INSTRUCT as a zeroshot toxicity classifier on Civil Comments, with the same protocol, attack, baselines, and metrics. Each case compares two identity groups of the same category: its population consists of all comments about either group, and a positive prediction flags a comment as toxic. The three cases are Christian vs. Muslim (very unfair, true DP −0.228), Christian vs. Jewish (moderate, −0.190), and Black vs. White (near-fair, −0.060), the last only 0.010 outside the fair band (Table I). The honest auditors data is as heterogeneous as on Bias in Bios, and Figure 10 and table IV show the same picture. Across all runs with at least one adversary, the attack succeeds in 97% of the audits without a defense, in 48% and 59% with the median and the trimmed mean, and in 26% with AUDITOPUS, all on the near-fair case. AUDITOPUS lowers the mean DP estimation error from 0.156 without a defense to 0.034 (−78%), whereas the median (0.180) and the trimmed mean (0.171) are even less accurate than no defense at all. Without a defense, a single adversary fairwashes every run on the moderate and near-fair cases. AUDITOPUS, in contrast, is never fooled on the very unfair and moderate cases, but it fails on the near-fair one. Already without adversaries it declares the model fair in 50% of the runs: down-weighting honest auditors whose updates deviate from their own history by chance leaves a mean error of $+ 0 . 0 1 0$ , exactly the distance to the band. The median and the trimmed mean do the same (50% at $K = 0 )$ . Under attack, AUDITOPUS is fooled in 55–80% of the runs up to 8 adversaries and in every run at $K = 1 9$ Its estimate nevertheless stays closest to the truth of all methods (mean error +0.015 to +0.036, against +0.061 without a defense; Table IV), but an error of a few hundredths suffices to cross a band a hundredth away. A disparity this close to the threshold is at the limit of what an audit on heterogeneous data can resolve. Robust aggregation is pushed past the band. All three Civil Comments cases have mid-range positive rates, so the median and the trimmed mean behave as on the moderate Bias in Bios case: they are fooled in 33–68% (median) and 35–80% (trimmed mean) of the attacked runs, and they do not stop at the band. At $K = 1 9$ the median’s mean error exceeds the true disparity on both the very unfair and the moderate cases, i.e., the estimate is pushed past 0.

## B. RQ4: Sampling without replacement

Throughout the paper, each auditor samples its query set without replacement. By the end of the audit it has queried every one of its rows once. Here we justify this choice by comparing it with querying with replacement, where each query is drawn uniformly from the auditor’s query set. Both modes use the same data splits, graphs, seeds and number of rounds as RQ1, and we compare with the true DP.

With replacement, even honest auditors never reach the true DP. Figure 11 follows the honest auditors’ aggregated estimate through the audit, with no adversary. In both modes the first rounds give the same estimate, off the true DP. Every auditor contributes 300 queries per round, so a small query set counts as much as a large one. Without replacement, the small query sets run out as the audit proceeds while the large ones keep contributing, and at the end the estimate equals the true DP in every run. With replacement, every auditor keeps drawing 300 queries per round from its query set, so the estimate stays where it started. On Christian vs. Muslim, for instance, it goes from −0.151 after the first round to −0.157 at the end, against a true DP of −0.228, a third of the disparity lost. On the near-fair Black vs. White case, the offset even puts the estimate inside the fair band. The bias grows with heterogeneity. With near-homogeneous data (α = 100), all query sets look alike and weighting them equally changes nothing. The error is at most 0.001. As the query sets become more heterogeneous it grows, e.g., on Christian vs. Muslim from +0.012 at α = 5 to +0.072 at α = 1 and $+ 0 . 1 1 7$ at $\alpha = 0 . 3$ . Heterogeneous query sets are the setting of this paper, so querying with replacement would bias every audit, independently of any adversary.

## C. RQ5: Robust-aggregation baselines

In the main text, the median and the trimmed mean aggregate each group’s positive rate over the sources of an auditor’s table $( a / A$ and $b / B$ separately), and the trimmed mean cuts $\beta = 2 0 \%$ of the sources from each end. Here we test whether a different choice rescues them: aggregating each source’s local DP $a / A { - } b / B$ instead, and other trimming fractions $\beta \in \{ 1 0 , 2 0 , 3 0 , 4 0 \} \%$ Everything else is as in RQ1.

Aggregating local DPs. Figure 12 applies the median and the trimmed mean to each source’s local DP instead of its per-group positive rates. Heterogeneity biases both aggregators even without adversaries: the honest auditors’ local DPs are spread widely around the true DP (Figure 2), so their median is not the pooled DP. With no adversary, it already declares the model fair in 60% of the audits on teacher and 35% on physician.

Changing the trimming fraction. Trimming more makes the trimmed mean closer to the median (Figure 13). No fraction comes close to AUDITOPUS. Even choosing the best β for each case separately, which an auditor could not do without knowing the true DP, the trimmed mean is fooled in 47% of the runs on Christian vs. Muslim, 32% on Christian vs. Jewish, 48% on physician and 25% on teacher, where AUDITOPUS is fooled in at most 9%. Only on nurse (both 0%) and on the near-fair Black vs. White (73% against 78%) does it match AUDITOPUS. Larger fractions also inherit the median’s other failure: on nurse at $K = 1 0$ , the estimate is pushed away from the band, overstating the disparity; and without adversaries, the false fair verdicts on teacher rise from 20% to 35% as $\beta$ grows.

Neither choice makes robust aggregation robust. Aggregating local DPs trades being pushed away from the band for being pushed into it, and no trimming fraction closes the gap to AUDITOPUS. Both fail for the same reason as in the main text. On heterogeneous data the honest auditors’ values are not concentrated, which is the assumption robust aggregation relies on.

## D. RQ6: Varying the query budget

In the main text every auditor issues $n = 3 0 0$ queries per round. Here we vary n from 100 to 3,000, with $K = 6$ adversaries (30% of the $N = 2 0$ honest auditors), who fabricate n queries per round as well; everything else is as in RQ1. Since auditors query without replacement, every honest auditor still queries its whole query set, whatever the budget. The budget only changes how the audit is split into rounds. The number of rounds is $\lceil \operatorname* { m a x } _ { i } \left| \mathcal { D } _ { i } \right| / n \rceil$ , from 100–172 at $n = 1 0 0$ to only 1–6 at $n = 3 , 0 0 0$ and with it the size of each update that AUDITOPUS scores. $T = 1 5 0$ cap defined in Section VI-A is for $n = 3 0 0$

![](images/f657fca06bddf2df5e682fad65ccc11c87335cce318692020968a0e0dafe5d97.jpg)

![](images/49da8c1b00feb10ad2a99ae75b39d885c435831a0a4863ba822e61e065791599.jpg)

![](images/3bd5cbe2d9f7310d52bdddd22758f1c888ee7bc6f9699441676f63382b011a54.jpg)  
(b) Bias in Bios  
Fig. 11. Honest auditors only $( N = 2 0 ,$ no adversary), querying without and with replacement (mean ± SD over 20 runs; top rows: Civil Comments, bottom rows: Bias in Bios). Without replacement the estimate converges to the true DP; with replacement it stays where it started, off the true DP by an amount that grows with heterogeneity.

AUDITOPUS is unaffected as long as the audit has several rounds. Figure 14 shows attack success on both datasets. On the very unfair and moderate cases of both datasets, AUDITOPUS is never fooled at any budget up to 3,000. On the near-fair case in Bias in Bios it is fooled in at most 15% of the runs up to $n = 2 { , } 0 0 0$ , and in 40% at 3,000, where the audit is down to about six rounds. With so few updates per source, AUDITOPUS has little history to judge each update against. On Black vs. White, which lies only 0.010 outside the band, AUDITOPUS is fooled in 65–95% of the runs at every budget, as in RQ3.

![](images/5d70244845db3fa1bf0f13b17797e8b0ff4c7416030cf48b9b427fcb38477d60.jpg)  
Number of adversaries (on top of N = 20 honest auditors)

(a) Civil Comments  
![](images/c9e5bdbb873b3b19bbb200e10b0dedbf4dbaea03f515d4037d864c18f71fa941.jpg)  
Number of adversaries (on top of N = 20 honest auditors)  
(b) Bias in Bios  
Fig. 12. The median and the 20% trimmed mean applied to each source’s local DP estimate, against no defense and AUDITOPUS, in the setting of RQ1 (top rows: Civil Comments; bottom rows: Bias in Bios).

The undefended audit is fooled less often at the largest budgets. Without a defense, the attack succeeds in at least 80% of the runs of every case up to $n = 2 { , } 0 0 0$ (70% on Christian vs. Jewish), but less often at 3,000 (e.g., 60% on physician and 45% on Christian vs. Jewish). This is not a defense but a shorter audit: with only a few rounds, gossip cannot deliver the adversaries’ latest counts to every honest auditor before the audit ends, so the adversaries’ fabricated counts reach fewer auditors, whose estimates stay further from 0. AUDITOPUS’s advantage does not depend on the query budget as long as the audit spans several rounds. When a large budget shrinks the audit to a handful of rounds, AUDITOPUS has little history per source to compare against, the same limitation that makes a one-shot audit undefendable (Figure 3).

![](images/fce234640b388b41329fc0a02edf5e4c5ab449c7aabafb9c6236c67fd37d32fb.jpg)

![](images/dbe80f3cf8f560201bb1a43620d0e45680818b61f2f0ec4e3cc4dd793a217b61.jpg)  
Number of adversaries (on top of N = 20 honest auditors)

(a) Civil Comments  
![](images/96aef0ebee3cd881b7c1db2206ff626d45f9eeb313825999be878bba6773c2b3.jpg)

![](images/4be5be84e9efae29e587d49a640cf32b2cf158ee4918995ba78fbab04f9800aa.jpg)  
Number of adversaries (on top of N = 20 honest auditors)  
(b) Bias in Bios  
Fig. 13. The trimmed mean of the sources’ group rates at trimming fractions $\beta = 1 0 , 2 0 , 3 0$ and 40%, against AUDITOPUS, in the setting of RQ1.

![](images/83789d8b4b12dbcd34818d2bebe40c2cd3cd8635507304c3e24081089bdddf01.jpg)

![](images/172c20b41136c0e1b4a5d4aae6b7384bcbaec13c6265c1fb5f0cfe5fd291f386.jpg)  
(b) Bias in Bios  
Fig. 14. Attack success as the number of queries per auditor per round grows from 100 to 3,000, with K = 6 adversaries next to $N = 2 0$ honest auditors (20 runs per point); Civil Comments (top) and Bias in Bios (bottom).