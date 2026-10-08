# Self-Consuming Generative Models with Co-Evolving Human Preferences

Xiukun Wei<sup>1∗</sup>, Tian Xie<sup>1∗</sup>, Ding Zhu<sup>1</sup>, Xueru Zhang<sup>1</sup>

<sup>1</sup>The Ohio State University

{wei.1418, xie.1379, zhu.3723, zhang.12807}@osu.edu

## Abstract

Generative models are increasingly trained in self-consuming iterative loops, where users curate preferred samples from model-generated candidates and the curated samples are used to train future generations of the model. Prior work has largely assumed fixed user preferences, but in practice exposure to model outputs gradually reshapes what users perceive as desirable, creating a feedback loop in which model distributions and user preferences co-evolve. We take a first step toward understanding the long-term behavior of such coupled dynamics. We show that when training relies entirely on user-curated synthetic data, iterative curation amplifies initial biases and drives the system toward one of multiple singleton equilibria in which the instance holding an initial advantage eventually dominates. In contrast, injecting reference data into training at a sufficiently large rate fundamentally changes the dynamics and yields a unique globally attracting equilibrium. Building on this insight, we study how reference-data injection can be used to control long-term outcomes, and propose an efficient algorithm that jointly selects a reference distribution and its mixing weight to steer the coupled system toward equilibria that preserve desired attributes while minimizing data collection costs.

## 1 Introduction

Modern generative models are increasingly trained on data that contains synthetic outputs produced by earlier versions of themselves, creating a self-consuming training loop. Recent studies have shown that such loops can lead to model collapse, training instability, and bias amplification [2, 3, 12, 14, 39, 40, 44], and have proposed various mitigation strategies [6, 14, 15]. In many practical settings, the synthetic data reused for retraining is further filtered through human curation [53]: model-generated samples are presented to users, who express preferences by selecting, rating, editing, reposting, or otherwise amplifying certain outputs over others, and the retained samples are subsequently incorporated into future training datasets. Prior work has shown that repeated retraining on such human-curated synthetic data can implicitly steer models toward distributions aligned with user preferences [11, 45].

However, existing works largely treat the preference signal as fixed or exogenous: users are assumed to consistently favor the same attributes, and human curation simply filters samples accordingly. In practice, this assumption is often violated because users and the model interact in a closed loop: the model itself reshapes the environment from which future preferences are formed, as its outputs shape what users are exposed to, which in turn shapes what they find desirable. Such model-influenced preference shifts are supported by recent empirical evidence: LLM-assisted writing systematically alters the tone, vocabulary, and meaning of human-produced text [1]; exposure to biased AI writing assistants shifts users’ subsequent opinions on the underlying topic [19]; and AI-assisted creation more broadly homogenizes the diversity of content users produce [10]. In each case, synthetic outputs are not merely consumed by future models; they reshape the preference landscape that guides future curation, creating a closed feedback loop in which the model and its users co-evolve.

![](images/6dab6684087f58e52b808c6ea36427d9dedb7915ac54f637f2694c36668b4040.jpg)  
Figure 1: Overview of the coupled model-user dynamics. Top row (model update): the current model $p _ { t }$ generates $K$ candidates, of which one is retained via preference-guided $\hat { K } \mathrm { - w a y }$ Luce choice; the curated set $\mathcal { D } _ { \mathrm { c u r } }$ is mixed with a reference dataset ${ \mathcal { D } } _ { \mathrm { r e f } }$ at rate $\eta$ to form $\mathcal { D } _ { t } .$ , on which $p _ { t + 1 }$ is trained. Bottom row (preference update): the user assesses how each candidate aligns with the current preference $w _ { t }$ via the reward $\boldsymbol { r } _ { t } ( \boldsymbol { x } ) = \langle \boldsymbol { w } _ { t } , \varphi ( \boldsymbol { x } ) \rangle$ ⟩, where $\varphi$ is the feature map, and the preference vector then drifts toward the mean feature direction $\bar { \varphi } ( p _ { t + 1 } )$ of the updated model.

This paper sets out to study the long-term behavior of coupled user-model dynamics. We formulate a dynamical system over the joint state $( p _ { t } , w _ { t } )$ in which the model distribution $p _ { t }$ and the underlying user preference vector $w _ { t }$ co-evolve over discrete time steps, as illustrated in Fig. 1. At each round t, the model generates $K$ candidates and the user curates one according to their current preferences, with samples whose features align more closely with $w _ { t }$ selected with higher probability. The model is then retrained on the curated synthetic samples, optionally mixed with a fixed reference distribution, while the user preferences simultaneously drift toward the feature direction of the updated model.

Our analysis shows that the system exhibits qualitatively different long-run behavior depending on whether reference data is included in training. In the purely synthetic regime, where retraining relies entirely on user-curated synthetic samples, repeated curation amplifies local reward advantages: any item that is even slightly preferred under the current reward eventually dominates, driving the system toward one of multiple singleton equilibria. In contrast, when a sufficient fraction of reference data is mixed in at each round, the coupled update becomes a strict contraction on the joint state space under an explicit condition on the mixing weight, implying a unique globally attracting equilibrium that every trajectory converges to geometrically, regardless of initialization.

The uniqueness of the equilibrium under reference-data mixing opens a natural design question: how should the reference distribution and mixing weight be jointly chosen so that a specified level of diversity is preserved at equilibrium at minimum data-collection cost? We formalize this as a constrained optimization problem, which is bilevel and intractable since the constraint involves the equilibrium of a nonlinear coupled map with no closed form. To tackle this, we identify a worst-case relaxation that yields a linear sufficient condition on the reference distribution, decoupling the bilevel structure and reducing the problem to a family of linear programs indexed by the mixing weight. Experiments verify the theoretical results and demonstrate the effectiveness of the proposed algorithm.

Tbl. 1 summarizes how our paper extends the existing line of self-consuming generative models with human curation. In Appx. A, we discuss more related work, and our contributions can be summarized as follows:

• We propose a dynamic model that captures the co-evolution of user preferences and the model under self-consuming iterative training (Sec. 2).

• We theoretically analyze the convergence of the coupled dynamics and characterize the properties of its equilibria, both with and without reference data mixed into the retraining loop (Sec. 3).

• We formulate equilibrium steering as a constrained optimization problem and show that a worstcase sufficient condition reduces it to a family of linear programs, providing a computationally lightweight algorithm for jointly choosing the reference distribution and mixing weight (Sec. 4).

• We empirically verify the theoretical results and the effectiveness of the proposed algorithm on both image generation (CFM and DDPM on CIFAR-10) and text generation (GPT-2 on AG News). (Sec. 5)

Table 1: Comparison with prior work on self-consuming generative models with curation.
<table><tr><td>Work</td><td>Preference</td><td>#Models</td><td>Core conclusion</td><td>Risk</td><td>Role of real data</td></tr><tr><td>Ferbach et al. [11]</td><td>Fixed</td><td>Single</td><td>Curation implicitly optimizes a fixed preference signal.</td><td>Bias amplification Stabilizer</td><td></td></tr><tr><td>Wei and Zhang [45] Fixed</td><td></td><td>Single</td><td>Adversarial curation can systematically misalign the model.</td><td>Malicious feedback</td><td>Improves robustness</td></tr><tr><td>Zhao et al. [59]</td><td>Heterogeneous fixed</td><td>Single</td><td>Reference mixing yields convergence and stability under noisy heterogeneous curation.</td><td>Reward perturbation instability</td><td>Stabilizer and regularizer</td></tr><tr><td>Zhang et al. [58]</td><td>Fixed</td><td>Multi</td><td>Benign curation may backfire under model interaction.</td><td>Cross-model coupling</td><td>Convergence aid</td></tr><tr><td>This work</td><td>Co-evolving</td><td>Single</td><td>Preferences co-evolve with the Preference drift; Stabilizer; model, altering equilibrium structure.</td><td>diversity collapse Rreference data</td><td>design</td></tr></table>

## 2 Problem Formulation

Consider a platform that iteratively trains generative models using data curated by human users. Let $\mathcal { X } \subset \mathbb { R } ^ { m }$ be the instance space and let $p _ { t } \in \mathcal { P } ( \mathcal { X } )$ denote the output distribution of the generative model at round t of the retraining loop. We present the model and results in the discrete setting with finite ${ \mathcal { X } } = \{ x _ { 1 } , \ldots , x _ { n } \}$ , motivated by the fact that in many platforms, outputs fall into a finite number of distinguishable categories (e.g., topic or object classes); the extension to continuous spaces is deferred to Appx. D.

User preference & human curated data. At each round t, the platform presents $K \geq 2$ synthetic samples $\{ x _ { 1 } , \ldots , x _ { K } \}$ drawn from the current model $p _ { t }$ to users, who then select a preferred sample $\hat { x }$ that is used to train the next generation of models. Specifically, let $\varphi : \mathcal { X }  \mathbb { R } ^ { d }$ be a fixed feature map satisfying $\| \varphi ( x ) \| = 1$ , which extracts features of an instance $x \in \mathcal { X }$ relevant to user preferences as a unit-norm representation<sup>1</sup>. User preferences over these features at round t are characterized by a vector $w _ { t } \in \mathcal { W } : = \{ w \in \mathbb { R } ^ { d } : \lVert w \rVert \stackrel { - } { \leq } 1 \}$ . Given this preference vector, the reward of x at round t is defined as $r _ { w _ { t } } ( x ) : = r ( x ; w _ { t } ) : = \langle w _ { t } , \bar { \varphi } ( x ) \rangle \in [ - 1 , \bar { 1 } ]$ , which measures the alignment between the instance features and the user’s preference direction. Based on these rewards, users select xˆ from the candidate set $\{ x _ { 1 } , \ldots , x _ { K } \}$ according to the $K \cdot$ -way Luce rule:

$$
\operatorname* { P r } ( \hat { x } = x _ { k } \mid x _ { 1 } , \dotsc , x _ { K } ) = \frac { e ^ { \tau \cdot r _ { w _ { t } } ( x _ { k } ) } } { \sum _ { j = 1 } ^ { K } e ^ { \tau \cdot r _ { w _ { t } } ( x _ { j } ) } }\tag{1}
$$

where $\tau > 0$ controls the sharpness of user choice. When $K = 2 ,$ , this reduces to Bradley-Terry model [4, 30]. We denote this sampling process by $\hat { x } \sim \operatorname { L u c e } _ { K } ( x _ { 1 } , \dots , x _ { K } )$

Recursive training loop. Given a fixed reference (real data) distribution $p _ { \mathrm { r e f } } \in \mathcal { P } ( \mathcal { X } )$ and user curated synthetic data, the platform updates its model at round $t + 1$ using either only user-curated synthetic data $( \eta = 1 )$ ) or a mixture of synthetic and reference data $\eta \in ( 0 , 1 )$ :

$$
p _ { t + 1 } \in \arg \operatorname* { m a x } _ { p \in \mathcal { P } ( \mathcal { X } ) } ( 1 - \eta ) \mathbb { E } _ { x \sim p _ { \mathrm { r e f } } } [ \log p ( x ) ] + \eta \mathbb { E } _ { \hat { x } \sim \mathrm { L u c e } _ { K } ( x _ { 1 } , \dots , x _ { K } \sim p _ { t } ) } [ \log p ( \hat { x } ) ]\tag{2}
$$

where mixing parameter $\eta \in [ 0 , 1 ]$ controls the fraction of synthetic data used in training. Following Ferbach et al. [11], we show that the retraining step (2) admits a multiplicative reweighting of $p _ { t }$ (proof in Appx. C.1):

$$
p _ { t + 1 } ( x ) = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p _ { t } ( x ) H _ { p _ { t } , w _ { t } } ^ { K } ( x ) ,\tag{3}
$$

where $H _ { p _ { t } , w _ { t } } ^ { K } ( x )$ is referred to as the K-way curation operator, which quantifies the amplification of x induced by K-way preference-based selection from $p _ { t }$

Definition 2.1 (Curation operator). The K-way curation operator $H _ { p _ { t } } ^ { K } : \mathcal { X } \to \mathbb { R } _ { + }$ is defined as

$$
H _ { p _ { t } , w _ { t } } ^ { K } ( x ) : = \mathbb { E } _ { x _ { 1 } , \ldots , x _ { K - 1 } \sim p _ { t } } \left[ \frac { K e ^ { \tau \cdot r _ { w _ { t } } ( x ) } } { e ^ { \tau \cdot r _ { w _ { t } } ( x ) } + \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau \cdot r _ { w _ { t } } ( x _ { i } ) } } \right]
$$

Moreover, for any fixed $x \in \mathcal { X }$ , this reweighting converges, as $K  \infty$ , to an exponential tilt of $p _ { t }$ , $\begin{array} { r } { { \mathrm { i . e . , } } H _ { p _ { t } , w _ { t } } ^ { K } ( x ) \xrightarrow { K  \infty } \frac { e ^ { \tau \cdot r _ { w _ { t } } ( x ) } } { \mathbb { E } _ { z \sim p _ { t } } [ e ^ { \tau \cdot r _ { w _ { t } } ( z ) } ] } . } \end{array}$

Preference drift & coupled dynamics. When users are exposed to model-generated content on the platform, their preferences may evolve over time. To capture this effect, we model the preference vector $w _ { t }$ as being gradually steered toward the feature distribution induced by the model. Specifically, define $\bar { \varphi } ( p ) : = \bar { \mathbb { E } } _ { x \sim p } [ \varphi ( x ) ]$ as the mean feature vector under distribution $p .$ After retraining at round $t + 1$ , user preferences are updated according to

$$
w _ { t + 1 } = \mathrm { P r o j } _ { \mathcal { W } } \left( ( 1 - \beta ) w _ { t } + \beta \bar { \varphi } ( p _ { t + 1 } ) \right)\tag{4}
$$

where $\beta \in ( 0 , 1 ]$ is the preference adaptation rate and $\mathrm { P r o j } _ { \mathcal { W } }$ denotes Euclidean projection onto W.

Eq. (3) and (4) together define a closed-loop dynamical system over the state $( p _ { t } , w _ { t } )$ . We define an equilibrium $( p _ { K } ^ { \star } , w _ { K } ^ { \star } )$ as a fixed point of the joint dynamics, i.e., a state that remains unchanged under data generation and preference adaptation. Formally, $( p _ { K } ^ { \star } , w _ { K } ^ { \star } )$ satisfies

$$
p _ { K } ^ { \star } ( x ) = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p _ { K } ^ { \star } ( x ) H _ { p _ { K } ^ { \star } , w _ { K } ^ { \star } } ^ { K } ( x ) \qquad w _ { K } ^ { \star } = \mathrm { P r o j } _ { \mathcal { W } } ( ( 1 - \beta ) w _ { K } ^ { \star } + \beta \bar { \varphi } ( p _ { K } ^ { \star } ) )\tag{5}
$$

Objective. We study the evolution of the coupled dynamics $( 3 ) \AA - \textcircled { 4 }$ and characterize the long-run behavior of the model $p _ { t }$ and user preferences $w _ { t } \left( \mathsf { S e c } . 3 \right)$ . Specifically, we ask: under what conditions does the system converge and admit an equilibrium; how do the choice set size $K$ , the synthetic-data weight $\eta ,$ and the geometry of the feature map $\varphi$ govern this behavior; and when do the dynamics collapse toward degenerate outcomes versus stabilize under reinjection of reference data.

Beyond stability analysis, we investigate the reference distribution $p _ { \mathrm { r e f } }$ as a steering mechanism. We ask how $p _ { \mathrm { r e f } }$ and the synthetic-data weight η can be jointly chosen so that the resulting equilibrium preserves desired attributes while keeping the cost of intervention small (Sec. 4).

## 3 Long-Term Behavior of User–Model Dynamics

Next, we study the evolution of $( p _ { t } , w _ { t } )$ . We begin with the purely synthetic curation setting $( \eta = 1 )$ , and then extend to the setting in which each round mixes synthetic and reference samples $( \eta \in ( 0 , 1 ) )$ .

## 3.1 Retraining with purely synthetic user-curated data

When $\eta = 1$ , update (3) reduces to $p _ { t + 1 } ( x ) = p _ { t } ( x ) H _ { p _ { t } , w _ { t } } ^ { K } ( x )$ . In this case, the dynamics are driven entirely by repeated preference-guided reweighting of the current distribution via the curation operator $H _ { p _ { t } , w _ { t } } ^ { K }$ . Before analyzing the full dynamics, we first study $H _ { p _ { t } , w _ { t } } ^ { K }$ and identify its key properties.

Proposition 3.1 (Basic properties of curation operator). For any $p \in \mathcal { P } ( \mathcal { X } ) , w \in \mathcal { W } ,$ , and $K \geq 2 ,$ the curation operator $H _ { p , w } ^ { K }$ satisfies the following properties:

1. Strict monotonicity. For any $x _ { i } , x _ { j } \in \mathcal { X } , r _ { w } ( x _ { i } ) > r _ { w } ( x _ { j } ) \Longleftrightarrow H _ { p , w } ^ { K } ( x _ { i } ) > H _ { p , w } ^ { K } ( x _ { j } )$

2. Normalization. $\begin{array} { r } { \sum _ { i = 1 } ^ { n } p ( x _ { i } ) H _ { p , w } ^ { K } ( x _ { i } ) = 1 } \end{array}$

3. Odds contraction. $I f r _ { w } ( x _ { i } ) - r _ { w } ( x _ { j } ) \geq r _ { \Delta } > 0$ , then

$$
\frac { H _ { p , w } ^ { K } ( x _ { j } ) } { H _ { p , w } ^ { K } ( x _ { i } ) } \leq \lambda _ { K } ( r _ { \Delta } ) : = 1 - \frac { ( 1 - e ^ { - \tau \cdot r _ { \Delta } } ) ( K - 1 ) } { e ^ { 2 \tau } + K - 1 } ,
$$

with $\lambda _ { K } ( r _ { \Delta } ) \in ( 0 , 1 )$ and $\lambda _ { K } ( \boldsymbol { r } _ { \Delta } ) \xrightarrow { K  \infty } e ^ { - \tau \cdot \boldsymbol { r } _ { \Delta } }$

See proof in Appx. C.2. Strict monotonicity implies that $H _ { p , w } ^ { K }$ preserves the reward ordering induced by user preferences, ensuring higher-reward instances always receive higher selection weight. Moreover, odds contraction implies lower-reward items are disproportionately suppressed relative to higher-reward ones, leading to an increasingly sharp separation in their selection probabilities. As a result, curation does not merely favor better items at each round; instead, it compounds their advantage over time by repeatedly amplifying higher-reward instances and pushing weaker ones further behind.

Equilibrium and convergence analysis. Building on Prop. 3.1, we next analyze the dynamics. Since each retraining step shifts mass toward higher-reward instances at the expense of lower-reward ones, so any distribution that places positive mass on instances with differing rewards is unstable.

Theorem 3.2 (Equilibrium characterization under purely synthetic setting). When $\eta = 1 , a$ distribution $p \in { \mathcal { P } } ( { \mathcal { X } } )$ is a fixed point of the retraining step under preference w if and only if all instances in its support share the same reward:

$$
p ( x ) > 0 , p ( x ^ { \prime } ) > 0 \implies r _ { w } ( x ) = r _ { w } ( x ^ { \prime } ) .
$$

Consequently, for every instance $x _ { i } \in { \mathcal { X } } ,$ , the pair $( p ^ { \star } = \delta _ { x _ { i } } , w ^ { \star } = \varphi ( x _ { i } ) )$ is an equilibrium: the model has collapsed to a point mass on $x _ { i }$ while user preferences have locked onto its features.

See proof in Appx. C.3, where $\delta _ { x _ { i } }$ denotes the Dirac delta centered at $x _ { i }$ . Thm. 3.2 shows that training exclusively on user-curated synthetic data admits n distinct singleton equilibria, one centered at each instance $x _ { i } .$ in which the model concentrates all mass on a single instance and user preferences align entirely with its features. Since every instance supports a singleton equilibrium, the natural question is: which ofthese equilibria are locally attracting? The next theorem shows that any instance with a sufficient initial advantage will eventually dominate, a phenomenon we term "winner lock-in".

Theorem 3.3 (Local convergence to singleton equilibria). Fix $x _ { i } \in \mathcal X$ and suppose $\varphi ( x _ { j } ) \neq$ $\varphi ( x _ { i } )$ for all $j \neq i .$ Define the reward gap $\begin{array} { r } { \gamma _ { i } = \operatorname* { m i n } _ { j \neq i } \langle \varphi ( x _ { i } ) , \varphi ( x _ { i } ) - \varphi ( x _ { j } ) \rangle > 0 } \end{array}$ and let $\lambda _ { K } ( \cdot )$ be the odds-contraction factor from Prop. 3.1. If the initial state satisfies

$$
1 - p _ { 0 } ( x _ { i } ) \leq \varepsilon _ { i } , \qquad \| w _ { 0 } - \varphi ( x _ { i } ) \| \leq \frac { \gamma _ { i } } { 4 }
$$

for some $\varepsilon _ { i } \in ( 0$ , min $\left\{ \frac { 1 } { 2 } , 1 - \lambda _ { K } \Big ( \frac { \gamma _ { i } } { 2 } \Big ) , \frac { \gamma _ { i } } { 8 } \right\} )$ , then for all $t \geq 0$

$$
1 - p _ { t } ( x _ { i } ) \leq \left( \frac { \lambda _ { K } ( \frac { \gamma _ { i } } { 2 } ) } { 1 - \varepsilon _ { i } } \right) ^ { t } \bigl ( 1 - p _ { 0 } ( x _ { i } ) \bigr ) .
$$

Consequently, $p _ { t } ( x _ { i } ) \xrightarrow { t  \infty }$ 1 and $w _ { t } \xrightarrow { t  \infty } \varphi ( x _ { i } )$

See proof in Appx. C.4. Thm. 3.3 shows that if the initial distribution and preferences are sufficiently biased toward $x _ { i }$ , this bias is self-reinforcing: the mass on all other instances $1 - p _ { t } ( x _ { i } )$ decays geometrically and $p _ { t }$ converges to the singleton equilibrium $\delta _ { x _ { i } }$ . Reward gap $\gamma _ { i }$ is the minimum reward advantage that $x _ { i }$ holds over any competitor $x _ { j }$ when preferences w are fully aligned with $\varphi ( x _ { i } )$ : a larger gap expands the set of initial conditions from which convergence to $\delta _ { x _ { i } }$ is guaranteed.

Remark 3.4 (Role of the choice set size $K )$ . Thm. 3.3 holds for every $K \geq 2$ , but K governs the speed of convergence. Since $\lambda _ { K }$ decreases with K, larger choice sets make curation more selective and accelerate the decay of mass on competing instances. In the limit $K  \infty$ , the update approaches exponential tilting, the most aggressive form of preference-guided reweighting.

Note that the condition $\varphi ( x _ { j } ) \neq \varphi ( x _ { i } )$ in Thm. 3.3 is necessary. When two distinct instances share the same feature direction, curation cannot distinguish between them on the basis of rewards alone, and transferring mass from $x _ { i }$ to such a duplicate leaves both the reward geometry and the induced preference direction unchanged. Local stability therefore fails, as we show below.

Corollary 3.5 (Failure of local stability). Fix $x _ { i } ~ \in ~ { \mathcal { X } }$ . If there exists $x _ { j }$ with $j \neq i$ such that $\varphi ( x _ { j } ) = \varphi ( x _ { i } )$ , then the singleton equilibrium $\delta _ { x _ { i } }$ is not locally asymptotically stable.

See proof in Appx. C.5. Thm. 3.2, Thm. 3.3 and Cor. 3.5 together characterize the long-term behavior of the dynamics when models are retrained exclusively on human-curated synthetic data: it admits n singleton equilibria, one for each instance; under generic feature separation, any instance with a sufficient initial advantage in model mass and preference alignment locks in as the unique long-run outcome; and when two instances share the same feature direction, the local stability fails.

## 3.2 Retraining on mixed synthetic and reference data

We next consider $\eta \in ( 0 , 1 )$ , where each retraining round mixes curated synthetic data with a fixed reference distribution $p _ { \mathrm { r e f } }$ according to (3). Unlike purely synthetic setting, $p _ { \mathrm { r e f } }$ prevents the dynamics from collapsing to a single instance and instead drives them toward a globally stable equilibrium.

To analyze convergence and establish contraction of the dynamics, the following lemma quantifies how perturbations in the model distribution p and user preference w propagate through the retraining step $\begin{array} { r } { \dot { p } \to ( 1 - \eta ) p + \eta p H _ { p , w } ^ { K } } \end{array}$ and serves as the key technical tool for the convergence results below.

Lemma 3.6 (Sensitivity of the retraining step). $\begin{array} { r } { L e t d _ { \mathrm { T V } } ( p , q ) : = \frac { 1 } { 2 } \sum _ { x \in \mathcal { X } } | p ( x ) - q ( x ) | } \end{array}$ denote total variation distance. For any $p , q \in { \mathcal { P } } ( { \mathcal { X } } )$ , w, $v \in \mathcal W$ , and $K \geq 2 ,$ the following hold:

$$
\begin{array} { r } { d _ { \mathrm { T V } } \left( p H _ { p , w } ^ { K } , q H _ { q , w } ^ { K } \right) \leq L _ { p } d _ { \mathrm { T V } } ( p , q ) ; d _ { \mathrm { T V } } \left( p H _ { p , w } ^ { K } , p H _ { p , v } ^ { K } \right) \leq L _ { w } \Vert w - v \Vert ; \Vert \bar { \varphi } ( p ) - \bar { \varphi } ( q ) \Vert \leq 2 d _ { \mathrm { T V } } ( p , q ) } \end{array}
$$

where $L _ { p } : = 2 e ^ { 2 \tau } ( 2 + e ^ { 2 \tau } )$ and $L _ { w } : = 2 \tau e ^ { 4 \tau }$

See proof in Appx. C.6. Lemma 3.6 implies that the retraining step is Lipschitz continuous in both $p$ and w: the first bound controls how perturbations in p propagate through the curation step; the second controls the sensitivity to changes in $w ;$ and the third shows that distributional perturbations translate to bounded perturbations in the mean feature vector entering the preference update (4).

Equilibrium and convergence analysis. The coupled system $( 3 ) \AA - \thinspace ( 4 )$ is difficult to analyze directly: the model update depends on the current preference $w _ { t }$ , while the preference update depends on the updated distribution $p _ { t + 1 }$ . We therefore begin by fixing w and studying the model update in isolation. This decoupling breaks the circularity and allows us to characterize the long-term model distribution for any given w. Lem. 3.7 below shows that, unlike the purely synthetic setting which admits n singleton equilibria with long-run outcome determined by initial conditions (Thm. 3.2), mixing with p<sub>ref</sub> yields a unique long-run distribution, provided $\eta$ is small enough to make the update contractive.

Lemma 3.7 (Unique equilibrium under frozen-w). Fix $w \in \mathcal { W }$ , and consider the thefrozen-w model update $p _ { t + 1 } ( x ) = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p ( x ) H _ { p , w } ^ { K } ( x )$ $I f \eta L _ { p } < 1$ , the update is a strict contraction on $( \mathcal { P } ( \mathcal { X } ) , d _ { \mathrm { T V } } )$ . Consequently, there exists a unique equilibrium $p _ { w } ^ { K , \eta }$ and every trajectory converges to it geometrically:

$$
d _ { \mathrm { T V } } ( p _ { t } , p _ { w } ^ { K , \eta } ) \leq ( \eta L _ { p } ) ^ { t } d _ { \mathrm { T V } } ( p _ { 0 } , p _ { w } ^ { K , \eta } ) .
$$

See proof in Appx. C.7. In general, the equilibrium $p _ { w } ^ { K , \eta }$ must be computed iteratively. In the large-K limit, however, it admits a closed-form expression.

Proposition 3.8 (Closed-form equilibrium). As $K  \infty$ , the frozen-w equilibrium takes the form

$$
p _ { w } ^ { \infty , \eta } ( x ) = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) \frac { Z _ { w } ^ { \eta } } { Z _ { w } ^ { \eta } - \eta e ^ { \tau \cdot r _ { w } ( x ) } } .
$$

where $Z _ { w } ^ { \eta } : = \mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } } [ e ^ { \tau \cdot r _ { w } ( x ) } ]$ is the unique scalar satisfying

$$
\sum _ { i = 1 } ^ { n } ( 1 - \eta ) p _ { \mathrm { r e f } } ( x _ { i } ) \frac { e ^ { \tau \cdot r _ { w } ( x _ { i } ) } } { Z _ { w } ^ { \eta } - \eta e ^ { \tau \cdot r _ { w } ( x _ { i } ) } } = 1 , Z _ { w } ^ { \eta } > \eta \operatorname* { m a x } _ { i } e ^ { \tau \cdot r _ { w } ( x _ { i } ) } .
$$

See proof in Appx. C.7. The closed form reveals how the equilibrium balances three forces: $p _ { \mathrm { r e f } }$ provides a baseline, synthetic-data weight η amplifies the preference-induced exponential tilt $e ^ { \tau r _ { w } ( \cdot ) }$ and concentrates mass on high-reward instances, while the constant $Z _ { w } ^ { \eta }$ ensures the result is a valid probability distribution. As $\eta  0 .$ , the equilibrium recovers $p _ { \mathrm { r e f } } ;$ as $\eta  1$ , the tilt dominates and mass concentrates on the highest-reward instance, aligned with results of the purely synthetic setting.

With the frozen-w equilibrium characterized, we now return to the full coupled system in which both the model distribution and user preferences evolve simultaneously.

Theorem 3.9 (Convergence of coupled dynamics). Consider the coupled dynamics (3)-(4). If $\eta ( L _ { p } + 2 L _ { w } ) < 1$ , then there exists $\begin{array} { r } { \xi \in \left( \frac { \eta L _ { w } } { \beta ( 1 - 2 \eta L _ { w } ) } , \frac { 1 - \eta L _ { p } } { 2 \beta \eta L _ { p } } \right) } \end{array}$ such that the coupled update is a strict contraction on $( \mathcal { P } ( \mathcal { X } ) \times \mathcal { W } , d _ { \xi } )$ , where $d _ { \xi } \big ( ( p , w ) , ( q , v ) \big ) : = d _ { \mathrm { T V } } ( p , q ) + \xi \lVert w - v \rVert .$

Consequently, there exists a unique equilibrium $( p _ { K } ^ { \star } , w _ { K } ^ { \star } )$ satisfying (5), and every trajectory converges to it geometrically.

See proof in Appx. C.8. The metric $d _ { \xi }$ combines distributional and preference distances into a single metric on $\mathcal { P } ( \boldsymbol { \bar { \mathcal { X } } } ) \times \mathcal { W } ;$ ; the window condition on $\xi$ ensures both coordinates contract simultaneously, and the contraction condition $\eta ( L _ { p } + 2 L _ { w } ) < 1$ ensures the reference mixing damps both feedback loops sufficiently for the Banach fixed-point theorem to apply. Compared with the frozen-preference condition $\eta L _ { p } \dot { < } 1$ in Lem. 3.7, the extra $L _ { w }$ term captures the stability cost of preference drift.

Thm. 3.9 highlights a key difference from the purely synthetic setting: whereas pure self-consumption leads to path-dependent collapse driven by initial conditions, mixing with $p _ { \mathrm { r e f } }$ yields a unique equilibrium to which all trajectories converge, regardless of initialization. The resulting equilibrium depends entirely on $p _ { \mathrm { r e f } }$ and η, which raises a natural design question: how should $p _ { \mathrm { r e f } }$ and η be chosen so that the equilibrium preserves desired attributes? We address this in Sec. 4.

## 4 Steering Equilibria via Reference Data Mixing

The uniqueness of the equilibrium under reference data mixing, established in Thm. 3.9, motivates the use of $p _ { \mathrm { r e f } }$ and η as design variables: by choosing them appropriately, one can steer the coupled dynamics toward a desired equilibrium. Specifically, denote $( p _ { K } ^ { \star } ( p _ { \mathrm { r e f } } , \eta ) , w _ { K } ^ { \star } ( p _ { \mathrm { r e f } } , \eta ) )$ as the unique equilibrium attained under reference distribution $p _ { \mathrm { r e f } }$ and mixing weight $\eta ,$ the corresponding mean feature vector is $\bar { \varphi } ( p _ { K } ^ { \star } ( p _ { \mathrm { r e f } } , \eta ) ) : = \mathbb { E } _ { x \sim p _ { K } ^ { \star } ( p _ { \mathrm { r e f } } , \eta ) } \mathrm { \hat { [ \varphi ( x ) ] } }$

To specify which attributes should be preserved at equilibrium, suppose the platform identifies $L$ features of interest, each represented by a unit-norm direction $v _ { \ell } \in \mathbb { R } ^ { d }$ in the feature space. For example, $v _ { \ell }$ might point toward the feature vectors of instances belonging to a particular genre, topic, or demographic group. The quantity $\left. v _ { \ell } , \bar { \varphi } ( p _ { K } ^ { \star } ) \right.$ then measures how well the equilibrium distribution represents feature ℓ on average. Given thresholds $\theta _ { 1 } , \ldots , \theta _ { L } > 0$ specifying the minimum acceptable representation for each feature, we say that $( p _ { \mathrm { r e f } } , \eta )$ satisfies the preservation constraint if $\left. v _ { \ell } , \bar { \varphi } ( p _ { K } ^ { \star } ( p _ { \mathrm { r e f } } , \eta ) ) \right. \geq \theta _ { \ell } , \forall \ell \in \{ 1 , \cdots , L \}$ . For example, in an image generation platform with three categories, setting L = 1, $v _ { 1 } = ( \underbrace { 1 } _ { \mathrm { b i r d } } , \underbrace { 0 } _ { \mathrm { c a t } } , \underbrace { 0 } _ { \mathrm { d o g } } )$ and $\theta _ { 1 } = 0 . 0 5$ requires that bird content

constitutes at least 5% of the equilibrium output.

Let N denote the number of training samples used at each training round. The reference distribution $p _ { \mathrm { r e f } } = ( p _ { \mathrm { r e f , 1 } } , \cdot \cdot \cdot , p _ { \mathrm { r e f } , n } )$ specifies the allocation of reference samples across instances: at each round, $p _ { \mathrm { r e f } , i } \cdot ( 1 - \eta ) \cdot N$ fresh samples of instance $x _ { i }$ are collected and added to training. Collecting these samples is not free. Rare or specialized instances (e.g., those representing underrepresented languages or domain-specific content) require more effort to source, label, and verify than common ones. We capture this heterogeneity by assigning each instance x<sub>i</sub> a per-sample cost $c _ { i } > 0$ , so that the total collection cost per round is $\begin{array} { r } { J ( p _ { \mathrm { r e f } } , \eta ) : = ( 1 - \eta ) \cdot N \cdot \sum _ { i = 1 } ^ { n } c _ { i } p _ { \mathrm { r e f } , i } } \end{array}$

Design objective. Our goal is to find $( p _ { \mathrm { r e f } } , \eta )$ that minimizes the total collection cost subject to the preservation constraint at the long-run equilibrium. This is naturally formulated as a bilevel problem: the outer level optimizes over $( p _ { \mathrm { r e f } } , \eta )$ , and the inner level computes the equilibrium that determines whether the preservation constraint is met:

$$
\operatorname* { m i n } _ { p _ { \mathrm { r e f } } \in \mathcal { P } ( \mathcal { X } ) , \eta \in ( 0 , 1 ) } \quad J ( p _ { \mathrm { r e f } } , \eta ) \quad \mathrm { s . t . } \quad \left. v _ { \ell } , \bar { \varphi } ( p _ { K } ^ { \star } ( p _ { \mathrm { r e f } } , \eta ) ) \right. \geq \theta _ { \ell } , \quad \ell = 1 , \ldots , L\tag{6}
$$

However, (6) is difficult to solve directly: evaluating the preservation constraint requires computing $p _ { K } ^ { \star } ( p _ { \mathrm { r e f } } , \eta )$ , which is the fixed point of a nonlinear coupled map and has no closed form. To tackle this, we propose a worst-case relaxation of the preservation constraint that depends directly on $p _ { \mathrm { r e f } }$ and η, bypassing the need to compute the equilibrium explicitly. Enforcing this relaxed condition decouples the bilevel structure and reduces (6) to a collection of linear programs indexed by $\eta .$

Proposition 4.1 (Sufficient condition for preservation). Fix $\begin{array} { r } { \eta ~ \in ~ ( 0 , \frac { 1 } { ( L _ { p } + 2 L _ { w } ) } ) } \end{array}$ $I f \forall \ell \in$ $\{ 1 , \ldots , L \}$

$$
\sum _ { i = 1 } ^ { n } p _ { \mathrm { r e f } , i } \left. v _ { \ell } , \varphi ( x _ { i } ) \right. \geq \frac { \theta _ { \ell } - \eta \cdot \operatorname* { m i n } _ { i = 1 , \cdots , n } \left. v _ { \ell } , \varphi ( x _ { i } ) \right. } { 1 - \eta } ,
$$

then the equilibrium satisfies the preservation constraint $\left. v _ { \ell } , \bar { \varphi } ( p _ { K } ^ { \star } ( p _ { \mathrm { r e f } } , \eta ) ) \right. \geq \theta _ { \ell } ,$ ∀ℓ.

See proof in Sec. C.9. Replacing the Algorithm 1 Cost-Optimal Reference Data Mixing   
equilibrium constraint in (6) by the suf  
ficient condition of Prop. 4.1 eliminates Require: Data size N, feature vectors $\{ \varphi ( x _ { i } ) \} _ { i = 1 } ^ { n } .$ , di  
the bilevel structure entirely. Since η ap- rections $\{ v _ { \ell } \} _ { \ell = 1 } ^ { L } .$ , thresholds $\{ \theta _ { \ell } \} _ { \ell = 1 } ^ { L } ,$ costs $\{ c _ { i } \} _ { i = 1 } ^ { n }$   
pears only in the right-hand side and in the Ensure: Reference distribution $p _ { \mathrm { r e f } } ^ { \star } ,$ mixing weight $\eta ^ { \star }$   
cost factor $( 1 - \eta )$ , fixing η decouples the 1: $J _ { \operatorname* { m i n } }  + \infty$   
problem into a linear program over $p _ { \mathrm { r e f } }$ 2: for η on a grid in $( 0 , \eta _ { \mathrm { m a x } } )$ do   
3: Solve $\bar { \mathrm { L P } } \left( 7 \right) \stackrel { } { \to } p _ { \mathrm { r e f } } \stackrel { } { ( \eta ) } , \stackrel { } { C } ^ { \star } ( \eta )$   
Algorithm. The complete design proce- 4: if LP feasible and $J ( \eta ) < J _ { \mathrm { m i n } }$ then   
dure combines an outer search over η with 5: $J _ { \mathrm { m i n } }  J ( \eta ) , \eta ^ { \star }  \eta , p _ { \mathrm { r e f } } ^ { \star }  p _ { \mathrm { r e f } } ( \eta )$   
an inner linear program (LP) solve for $p _ { \mathrm { r e f } } ,$ 6: end if   
as summarized in Alg. 1. The search range 7: end for   
is $( 0 , \eta _ { \mathrm { m a x } } )$ , where $\eta _ { \mathrm { m a x } }$ upper-bounds 8: return $p _ { \mathrm { r e f } } ^ { \star } , \eta ^ { \star }$   
mixing weight. Thm. 3.9 provides a con  
servative range $( 0 , \frac { 1 } { ( L _ { p } + 2 L _ { w } ) } )$ for $\eta ,$ but in practice, $\eta _ { \mathrm { m a x } }$ can be set larger since the Lipschitz   
constants are worst-case. The procedure is performed entirely offline prior to deployment and remains   
computationally lightweight: the inner problem is a linear program with n variables and $L + 1$   
constraints, and the outer problem is one-dimensional.

Inner problem: optimal reference distribution for fixed η. For any fixed $\eta \in ( 0 , \bar { \eta } )$ , Prop. 4.1 reduces the design task to the following linear program:

$$
C ^ { \star } ( \eta ) : = \operatorname* { m i n } _ { p _ { \mathrm { r e f } } \in \mathcal { P } ( \mathcal { X } ) } \sum _ { i = 1 } ^ { n } p _ { \mathrm { r e f } , i } c _ { i } \quad \mathrm { s . t . } \quad \sum _ { i = 1 } ^ { n } p _ { \mathrm { r e f } , i } \langle v _ { \ell } , \varphi ( x _ { i } ) \rangle \ \geq \ \frac { \theta _ { \ell } - \eta \operatorname* { m i n } _ { i } \langle v _ { \ell } , \varphi ( x _ { i } ) \rangle } { 1 - \eta } , \ \forall \ell\tag{7}
$$

The solution $p _ { \mathrm { r e f } } ( \eta )$ is the cheapest reference distribution that guarantees the preservation constraint at equilibrium for the given $\eta .$ . By standard linear programming theory, an optimal solution always exists with at most $L \bar { + } 1$ strictly positive entries, meaning the optimal $p _ { \mathrm { r e f } }$ concentrates on at most $L + 1$ instances. The platform therefore needs to actively source only a small number of distinct instance types, regardless of how large $\mathcal { X }$ is. Each inner problem can be solved efficiently offline using standard LP solvers such as simplex or interior-point methods.

Outer problem: selecting the mixing weight. Given $C ^ { \star } ( \eta )$ , the per-round collection cost is $J ( \eta ) : =$ $\overline { { ( 1 - \eta ) N C ^ { \star } ( \eta ) } }$ . The outer problem

$$
\eta ^ { \star } \in \arg \operatorname* { m i n } _ { \eta \in ( 0 , \bar { \eta } ) } ( 1 - \eta ) N C ^ { \star } ( \eta ) .\tag{8}
$$

balances two competing effects. A larger η means fewer reference samples per round, reducing the direct cost $( 1 - \eta ) \bar { N } \bar { C } ^ { \star } ( \eta )$ ; but it simultaneously tightens the preservation constraint, since the right-hand side of (7) grows with $\eta ,$ forcing $C ^ { \star } ( \eta )$ to increase. The optimal $\eta ^ { \star }$ resolves this tension, and since the problem is one-dimensional, it can be found efficiently by a simple grid search.

## 5 Experiments

This section aims to empirically illustrate our theoretical results in a realistic training condition, where finite samples, imperfect model training, and a concrete model architecture introduce errors absent from the abstract analysis<sup>2</sup>.

Datasets. We conduct experiments on two datasets (i) Image dataset: CIFAR-10 [24]: It contains 60,000 images from 10 classes {airplane $: = 0 .$ automobile := 1, bird := 2, cat := 3, deer := 4, dog := 5, frog := 6, horse := 7, ship := 8, truck := 9}. (ii) Text dataset: AG News [54]: It contains 127,600 texts from 4 classes {world := 0, sports := 1, business := 2, science/technology := 3}

Models. We experiment with three generative architectures: OT-CFM [28, 43] and DDPM [17, 35] for image generation, and GPT-2 [37] for text generation. We present OT-CFM results in this section; DDPM and GPT-2 results appear in Appx. E.2 and E.3, respectively. All models are adopted from public GitHub repositories and are used with their default hyperparameters. Each model is initialized from its released pretrained checkpoint, keeping architectures and optimization settings unchanged.

Feature map φ and preference $w _ { t }$ . For CIFAR-10, we use a pretrained ResNet-56 [16] classifier (accuracy $\geq 9 3 \% )$ . The feature map $\varphi ( x ) \in \mathbb { R } ^ { 1 0 }$ is the normalized softmax output, where $\varphi ( x ) _ { c }$ represents the predicted probability of class c. The preference vector $w _ { t } \in \mathbb { R } ^ { 1 0 }$ is updated via Eq. (4) with projection onto the unit ball. For AG News, we use a pretrained BERT [8] classifier with 4-dimensional normalized softmax features; see Appx. E.3 for details.

Iterative Retraining. For OT-CFM on CIFAR-10, each round t proceeds as follows. The current model generates $1 0 { , } 0 0 0$ candidate images, from which 5,000 are selected via K-way Luce curation with reward $\boldsymbol { r } _ { t } ( \boldsymbol { x } ) = \langle \boldsymbol { w } _ { t } , \varphi ( \boldsymbol { x } ) \rangle$ . The curated samples are mixed with reference data at ratio $\eta$ to form a training set of size 5,000, on which the model is fine-tuned for 1,000 steps. After training, 10,000 fresh samples are generated for evaluation: we compute per-class proportions and update $w _ { t }$ . We run each experiment for 20 rounds. We set $\tau = 5 . 0 , K = \infty .$ , and $\beta = 0 . 0 1$ for the main experiments. The effect of each parameter is discussed in Appx. E.1.1. Due to the computational cost of training and sampling from generative models, we report a single run for each setting, with the random seed varying across rounds. To verify reproducibility, Appx. E.2 reports three independent runs under the same setting, confirming that the trajectories are consistent.

Different initializations, different outcomes $( \eta = 1 )$ . Under pure synthetic training, the system exhibits initialization-dependent lock-in. When $w _ { 0 } = [ 1 , 0 , 0 , 0 , . . . ]$ (prefer airplane), curation progressively favors airplane images. The model’s airplane proportion rises from ∼10% to >95% over 20 rounds as shown in Fig. 2(a), while all other classes collapse below 5%. When $w _ { 0 } = [ 0 , 0 , 0 , 1 , \ldots ]$ (prefer cat), the same dynamics drive the model toward cat instead (Fig. 2(c)). In both cases, the curation signal is strong enough $( \tau = 5 . 0 , K = \infty )$ to steer the entire generated distribution. Appx. E.1.1 varies $\beta$ and shows that preference adaptation mainly changes the speed of preference drift, while the qualitative lock-in behaviors remain consistent.

![](images/2ad93cde98c714fa5bf5a6bfc2a9676a25f835f99367f4a3da82cc5f1b15eeff.jpg)

![](images/0f10bb0e5b893f20ce4f5ea31a329200ea8d7214f7fa9ab1f306f0f36de6d6b4.jpg)

![](images/f3101278e2e8489b7653406505cf6c51517ad4129a8e91792646603e2bd79115.jpg)

![](images/3c3e293384aa96c75e7eb5775749f5c84bb8993b1e5690fe66b4f7cec5575ab0.jpg)  
Figure 2: Co-evolution dynamics under different initializations and mixing ratios. Top: per-class generation proportions. Bottom: preference vector $w _ { t }$ .(a) The model locks into airplane, reaching ${ > } 9 5 \%$ by round 20. (b) Uniform reference data stabilizes the distribution near equilibrium. (c) A different initialization leads to a different outcome. (d) The system converges to a similar equilibrium, independent of initialization.

Enough reference data yields a unique equilibrium $( \eta < 1 )$ . When 90% of each training batch comes from the uniform CIFAR-10 training set $( \eta = 0 . 1 )$ , regardless of whether $w _ { 0 } = [ 1 , 0 , 0 , 0 , . . . ]$ (Fig. 2(b)) or $w _ { 0 } = [ 0 , 0 , 0 , 1 , . . . ]$ (Fig. 2(d)), the class proportions remain close to uniform across all 20 rounds. The two runs converge to a similar equilibrium, consistent with the uniqueness guarantee of Thm. 3.9: for sufficiently small $\eta ,$ the reference distribution stabilizes the system to a single fixed point independent of initialization.

Reference data design. We evaluate the effectiveness of Alg. 1 for preserving minority classes under co-evolution. We set the goal is to guarantee that the last four classes (frog, horse, ship, truck) each maintain a proportion of at least $\theta = 0 . 0 8$ at equilibrium. We assign heterogeneous costs reflecting class rarity: cheap classes (airplane, automobile: $c = 1 )$ are abundant, while protected classes are expensive (frog: 30, horse: 28, ship: 27, truck: 25). The preference is initialized to $w _ { 0 } = [ 1 , 0 , 0 , 0 , . . . ]$ (airplane).

Tbl. 2 compares our designed reference with uniform reference and Fig. 3 shows the class proportions under our designed reference. Our designed reference achieves two advantages simultaneously. First, it uses a higher synthetic ratio $( \eta = 0 . 6 7 )$ than either uniform baseline, reducing the total amount of reference data required. Second, at round 20, all protected classes exceed the threshold $\theta _ { \ell } = 0 . 0 8$ suggesting that the designed reference distribution steers the trajectory toward a feasible equilibrium. The partial LP search results over a uniform grid of 50 points in $\eta \in [ 0 . 2 , 0 . 9 ]$ and results under a different setting are provided in Appx. E.1.2.

Table 2: Comparison of reference data strategies. $C ^ { * }$ : per-sample cost; $J \colon$ total cost $( 1 - \eta ) \cdot N \cdot C ^ { * }$ . Last four columns: proportion of each protected class at round 20. $\theta _ { \ell } = 0 . 0 8$ for all protected classes.
<table><tr><td> $p _ { \mathrm { r e f } }$ </td><td>η</td><td> $C ^ { * }$ </td><td>J</td><td>frog</td><td>horse</td><td>ship</td><td>truck</td></tr><tr><td>Uniform</td><td>0.50</td><td>16.4</td><td>41,000</td><td>.067</td><td>.077</td><td>.059</td><td>.067</td></tr><tr><td>Uniform</td><td>0.20</td><td>16.4</td><td>65,600</td><td>.112</td><td>.113</td><td>.097</td><td>.105</td></tr><tr><td>Ours</td><td>0.67</td><td>26.81</td><td>44,043</td><td>.084</td><td>.113</td><td>.088</td><td>.092</td></tr></table>

![](images/559fbd93fd6254d9ce4ebbbde2dfd83e8417ad8b76068d35d21e6ad0da227e9f.jpg)  
Figure 3: Class proportions under designed reference.

## 6 Discussion

Our analysis focuses on a basic co-evolutionary setting in which a single aggregate preference state interacts with a self-consuming generative model through preference-guided curation. The same framework naturally extends to richer forms of preference dynamics. In particular, we consider nonlinear gap-response updates that include saturating and state-dependent gated responses; the model-update branch remains unchanged, while the stability analysis becomes response-dependent. Beyond this, stochastic individual-level adaptation provides a natural micro-level interpretation of the deterministic update studied here, while heterogeneous user populations lead to coupled groupspecific preferences interacting through a shared generative distribution. We discuss these extensions and the resulting theoretical questions in Appx. F.

## 7 Conclusion

We studied self-consuming generative retraining in which the model distribution and user preferences co-evolve. The system was formulated as a coupled dynamic over $( p _ { t } , w _ { t } )$ , with the model updated by reference-mixed user-curated data and the preference vector drifting toward the model’s mean feature. Under purely synthetic curation $( \eta = 1 )$ , the dynamics admit singleton equilibria, and any instance with a sufficient initial advantage locks in. With reference mixing satisfying $\eta ( L _ { p } + 2 L _ { w } ) < 1$ , the coupled update becomes a strict contraction with a unique, globally attracting equilibrium. Building on this, we further formulate equilibrium steering as a bilevel problem and reduce it to a family of linear programs indexed by η. Experiments empirically validate our analysis and confirm the effectiveness of our design. Several directions remain open: relaxing the exact observability of φ and $w _ { t }$ to partially observed quantities, extending the contraction analysis to nonlinear rewards, and generalizing the dynamics to multi-platform settings with heterogeneous user populations.

## Acknowledgments and Disclosure of Funding

This material is based upon work supported by the U.S. National Science Foundation under Grant Nos. IIS-2202699, IIS-2416895, IIS-2301599, CMMI-2301601, DMS-2529302, and OIA-2616279. The authors declare no competing interests.

## References

[1] Marwa Abdulhai, Isadora White, Yanming Wan, Ibrahim Qureshi, Joel Leibo, Max Kleiman-Weiner, and Natasha Jaques. How llms distort our written language, 2026. URL https://arxiv.org/abs/2603. 18161.

[2] Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, and Richard Baraniuk. Self-consuming generative models go MAD. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=ShjMHfmPs0.

[3] Quentin Bertrand, Joey Bose, Alexandre Duplessis, Marco Jiralerspong, and Gauthier Gidel. On the stability of iterative retraining of generative models on their own data. In The Twelfth International Conference on Learning Representations, 2024.

[4] Ralph Allan Bradley and Milton Edward Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

[5] Gavin Brown, Shlomi Hod, and Iden Kalemaj. Performative prediction in a stateful world. In International conference on artificial intelligence and statistics, pages 6045–6061. PMLR, 2022.

[6] Zhongteng Cai, Yaxuan Wang, Yang Liu, and Xueru Zhang. Stabilizing self-consuming diffusion models with latent space filtering. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 19844–19852, 2026.

[7] Sarah Dean and Jamie Morgenstern. Preference dynamics under personalized recommendations. In Proceedings ofthe 23rd ACM Conference on Economics and Computation, EC ’22, page 795–816, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450391504. doi: 10.1145/ 3490486.3538346. URL https://doi.org/10.1145/3490486.3538346.

[8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N19-1423/.

[9] Elvis Dohmatob, Yunzhen Feng, Pu Yang, Francois Charton, and Julia Kempe. A tale of tails: Model collapse as a change of scaling laws. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=KVvku47shW

[10] Anil R Doshi and Oliver P Hauser. Generative ai enhances individual creativity but reduces the collective diversity of novel content. Science advances, 10(28):eadn5290, 2024.

[11] Damien Ferbach, Quentin Bertrand, Avishek Joey Bose, and Gauthier Gidel. Self-consuming generative models with curated data provably optimize human preferences, 2024. URL https://arxiv.org/abs/ 2407.09499.

[12] Shi Fu, Yingjie Wang, Yuzhu Chen, Xinmei Tian, and Dacheng Tao. A theoretical perspective: How to prevent model collapse in self-consuming training loops. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=WttfQGwpES.

[13] Etienne Gauthier, Francis Bach, and Michael I. Jordan. Explaining and preventing alignment collapse in iterative rlhf, 2026. URL https://arxiv.org/abs/2605.04266.

[14] Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, Tomasz Korbak, Henry Sleight, Rajashree Agrawal, John Hughes, Dhruv Bhandarkar Pai, Andrey Gromov, Dan Roberts, Diyi Yang, David L. Donoho, and Sanmi Koyejo. Is model collapse inevitable? Breaking the curse of recursion by accumulating real and synthetic data. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=5B2K4LRgmz

[15] Nate Gillman, Michael Freeman, Daksh Aggarwal, Chia-Hong Hsu, Calvin Luo, Yonglong Tian, and Chen Sun. Self-correcting self-consuming loops for generative model training. In ICML, 2024. URL https://openreview.net/forum?id=i0nVanexij.

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016. doi: 10.1109/CVPR.2016.90.

[17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

[18] Zachary Izzo, Lexing Ying, and James Zou. How to learn when data reacts to your model: performative gradient descent. In International Conference on Machine Learning, pages 4641–4650. PMLR, 2021.

[19] Maurice Jakesch, Advait Bhat, Daniel Buschek, Lior Zalmanson, and Mor Naaman. Co-writing with opinionated language models affects users’ views. In Proceedings ofthe 2023 CHI conference on human factors in computing systems, pages 1–15, 2023.

[20] Zhuangzhuang Jia, Yijie Wang, Roy Dong, and Grani A Hanasusanto. Distributionally robust performative optimization. arXiv preprint arXiv:2407.01344, 2024.

[21] Kun Jin, Xueru Zhang, Mohammad Mahdi Khalili, Parinaz Naghizadeh, and Mingyan Liu. Incentive mech anisms for strategic classification and regression problems. In Proceedings of the 23rd ACM Conference on Economics and Computation, pages 760–790, 2022.

[22] Kun Jin, Tian Xie, Yang Liu, and Xueru Zhang. Addressing polarization and unfairness in performative prediction. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 22408– 22416, 2026.

[23] Michael P Kim and Juan C Perdomo. Making decisions under outcome performativity. arXiv preprint arXiv:2210.01745, 2022.

[24] Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009. URL https://api. semanticscholar.org/CorpusID:18268744.

[25] Qiang Li and Hoi-To Wai. State dependent performative prediction with stochastic approximation. In International Conference on Artificial Intelligence and Statistics, pages 3164–3186. PMLR, 2022.

[26] Qiang Li and Hoi-To Wai. Stochastic optimization schemes for performative prediction with nonconvex loss. Advances in Neural Information Processing Systems, 37:8673–8697, 2024.

[27] Qiang Li, Michal Yemini, and Hoi-To Wai. Clipped sgd algorithms for performative prediction: Tight bounds for clipping bias and remedies. arXiv preprint arXiv:2404.10995, 2024.

[28] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

[29] H. Liu, Qiang Li, and Hoi-To Wai. Two-timescale derivative free optimization for performative prediction with markovian data. arXiv preprint arXiv:2310.05792, 2023.

[30] R.D. Luce. Individual Choice Behavior: A Theoretical Analysis. Wiley, 1959. URL https://books. google.com/books?id=a80DAQAAIAAJ.

[31] Celestine Mendler-Dünner, Juan Perdomo, Tijana Zrnic, and Moritz Hardt. Stochastic optimization for performative prediction. Advances in Neural Information Processing Systems, 33:4929–4939, 2020.

[32] Celestine Mendler-Dünner, Frances Ding, and Yixin Wang. Anticipating performativity by predicting from predictions. Advances in neural information processing systems, 35:31171–31185, 2022.

[33] John P Miller, Juan C Perdomo, and Tijana Zrnic. Outside the echo chamber: Optimizing the performative risk. In International Conference on Machine Learning, pages 7710–7720. PMLR, 2021.

[34] Mehrnaz Mofakhami, Ioannis Mitliagkas, and Gauthier Gidel. Performative prediction with neural networks. In International Conference on Artificial Intelligence and Statistics, pages 11079–11093. PMLR, 2023.

[35] Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models, 2021. URL https://arxiv.org/abs/2102.09672.

[36] Juan Perdomo, Tijana Zrnic, Celestine Mendler-Dünner, and Moritz Hardt. Performative prediction. In International Conference on Machine Learning, pages 7599–7609. PMLR, 2020.

[37] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

[38] Mitas Ray, Lillian J Ratliff, Dmitriy Drusvyatskiy, and Maryam Fazel. Decision-dependent risk minimization in geometrically decaying dynamic environments. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 8081–8088, 2022.

[39] Mohamed El Amine Seddik, Suei-Wen Chen, Soufiane Hayou, Pierre Youssef, and Merouane Abdelkader DEBBAH. How bad is training on synthetic data? A statistical analysis of language model collapse. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=t3z6UlV09o.

[40] Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. AI models collapse when trained on recursively generated data. Nature, 631(8022):755–759, 2024.

[41] Ananda Theertha Suresh, Andrew Thangaraj, and Aditya Nanda Kishore Khandavally. Rate of model collapse in recursive training, 2024. URL https://arxiv.org/abs/2412.17646.

[42] Rohan Taori and Tatsunori B. Hashimoto. Data feedback loops: Model-driven amplification of dataset biases. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

[43] Alexander Tong, Kilian FATRAS, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=CD9Snc73AW. Expert Certification.

[44] Yaxuan Wang, Zhongteng Cai, Yujia Bao, Xueru Zhang, and Yang Liu. Observations and remedies for large language model bias in self-consuming performative loop. arXiv preprint arXiv:2601.05184, 2026.

[45] Xiukun Wei and Xueru Zhang. Self-consuming generative models with adversarially curated data, 2025. URL https://arxiv.org/abs/2505.09768.

[46] Xiukun Wei, Min Shi, and Xueru Zhang. Market games for generative models: Equilibria, welfare, and strategic entry. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=T7c5LXuebY.

[47] Xiukun Wei, Yang Zhang, and Xueru Zhang. Stability and diversity of networked self-consuming generative ecosystems. In The Fortieth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/forum?id=Mq8Ixa9IkT.

[48] Tian Xie and Xueru Zhang. Automating data annotation under strategic human agents: Risks and potential solutions. Advances in Neural Information Processing Systems, 37:127436–127482, 2024.

[49] Tian Xie and Xueru Zhang. Non-linear welfare-aware strategic learning. In Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, volume 7, pages 1660–1671, 2024.

[50] Tian Xie, Zhiqun Zuo, Mohammad Mahdi Khalili, and Xueru Zhang. Learning under imitative strategic behavior with unforeseeable outcomes. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=82bNZGMNZa.

[51] Tian Xie, Ding Zhu, Jia Liu, Mahdi Khalili, and Xueru Zhang. Sprint: Stochastic performative prediction with variance reduction. arXiv preprint arXiv:2509.17304, 2025.

[52] Songkai Xue and Yuekai Sun. Distributionally robust performative prediction. Advances in Neural Information Processing Systems, 37:55030–55052, 2024.

[53] Bingji Yi, Qiyuan Liu, Yuwei Cheng, and Haifeng Xu. Escaping model collapse via synthetic data verification: Near-term improvements and long-term convergence. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=yfk6c39omW.

[54] Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification, 2016. URL https://arxiv.org/abs/1509.01626.

[55] Xueru Zhang, Mohammadmahdi Khaliligarekani, Cem Tekin, et al. Group retention when using machine learning in sequential decision making: the interplay between user dynamics and fairness. Advances in neural information processing systems, 32, 2019.

[56] Xueru Zhang, Ruibo Tu, Yang Liu, Mingyan Liu, Hedvig Kjellstrom, Kun Zhang, and Cheng Zhang. How do fair decisions fare in long-term qualification? Advances in neural information processing systems, 33: 18457–18469, 2020.

[57] Xueru Zhang, Mohammad Mahdi Khalili, Kun Jin, Parinaz Naghizadeh, and Mingyan Liu. Fairness interventions as (dis) incentives for strategic manipulation. In International Conference on Machine Learning, pages 26239–26264. PMLR, 2022.

[58] Yang Zhang, Xiukun Wei, and Xueru Zhang. When and how human curation backfires: Preference alignment under multi-model self-consuming loop, 2026. URL https://arxiv.org/abs/2605.29267.

[59] Hongru Zhao, Jinwen Fu, and Tuan Pham. Convergence and stability analysis of self-consuming generative models with heterogeneous human curation, 2025. URL https://arxiv.org/abs/2511.09002.

[60] Yulai Zhao. Optimizing the performative risk under weak convexity assumptions. arXiv preprint arXiv:2209.00771, 2022.

[61] Xue Zheng, Tian Xie, Xuwei Tan, Aylin Yener, and Xueru Zhang. Profl: Performative robust optimal federated learning. arXiv preprint arXiv:2410.18075, 2024.

[62] Zihan Zhu, Ethan Fang, and Zhuoran Yang. Online performative gradient descent for learning nash equilibria in decision-dependent games. Advances in Neural Information Processing Systems, 36:47902– 47913, 2023.

A Related Work 16   
B Complete Instantiation Examples 17   
B.1 Discrete Examples . 17   
B.2 Continuous Examples 18   
C Proofs 19   
C.1 Proof of Explicit Retraining Update . 19   
C.2 Proof of Propostion 3.1 20   
C.3 Proof of Theorem 3.2 23   
C.4 Proof of Theorem 3.3 24   
C.5 Proof of Corollary 3.5 25   
C.6 Proof of Proposition 3.6 26   
C.7 Proofs of Lemma 3.7 and Proposition 3.8 . 29   
C.8 Proof of Theorem 3.9 30   
C.9 Proof of Proposition 4.1 31   
D Extension to Continuous Instance Spaces 33   
D.1 Retraining with Purely Synthetic User-Curated Data 33   
D.2 Retraining on Mixed Synthetic and Reference Data . 36   
E Additional Experimental Results 38   
E.1 Additional CFM Experiments on CIFAR-10 39   
E.2 DDPM Experiments on CIFAR-10 41   
E.3 GPT-2 Experiments on AG News 43   
F Discussion and Extensions 44   
F.1 Nonlinear Preference Responses . 44   
F.2 Stochastic Preference Adaptation 45   
F.3 Heterogeneous User Populations 46   
G Broader impacts 46

## A Related Work

Self-consuming When generative models are repeatedly trained on their own outputs, the training distribution can drift from the original data distribution. Early empirical studies demonstrated that this loop leads to progressive loss of distributional coverage and tail information [2, 14, 40], with Taori and Hashimoto [42], Xie and Zhang [48] showing that feedback loops can amplify existing dataset biases or even introduce complex feature distribution shifts. Subsequent theoretical work characterized the conditions under which collapse occurs and the rate at which errors accumulate [3, 9, 12, 39, 41]. Several mitigation strategies have been proposed, including retaining real data in the training loop [2, 3, 14], filtering generated samples via verification or latent-space methods [6, 53], and incorporating self-correction during generation [15]. Recent work ha further broadened the study of generative feedback from isolated single-model loops to settings with multiple interacting models, networked generative ecosystems, and strategic interactions among generative platforms [46, 47, 58]. Our work builds on this line by studying a setting where the training loop is further shaped by preference-guided curation and where the preference signal itself co-evolves with the model.

Self-consuming with curated data A growing body of work has studied self-consuming retraining with human curation. Ferbach et al. [11] show that, under a fixed reward signal, human curation acts as an implicit preference-optimization mechanism. Wei and Zhang [45] further show that when the same curation process is noisy or adversarially manipulated, the retraining loop can become systematically misaligned from genuine user preferences. Work on multi-model self-consuming systems shows that even benign human curation need not improve alignment once models interact through recycled data: its effect can be amplified, dampened, or even reversed by cross-model coupling [58]. More recently, Zhao et al. [59] studied self-consuming generative models with heterogeneous human curation, proving convergence and stability results across pure synthetic and reference-mixed retraining regimes. Related iterative alignment Gauthier et al. [13] studies the co-evolution of a policy and a learned reward-model estimate around a fixed underlying human preference. Their Stackelberg decomposition characterizes how gradient-based policy optimization can corrupt the reward-model estimate, and their FPO intervention regularizes the policy update to account for this influence. However, these formulations still treat the underlying human preference as exogenous. In many deployed settings, model outputs can reshape what users read, compare, and upload, thereby altering the preferences that guide future curation. Our work studies this endogenous feedback loop by allowing model retraining and user preferences to co-evolve. Tbl. 1 summarizes the key differences.

Performative prediction. Performative prediction (PP) studies learning problems where the deployed model changes the data distribution it will later face [36]. This feedback effect appears in many human-facing ML systems: applicants may strategically modify their features, users may change their participation or engagement behavior, and adversaries may adapt to deployed classifiers. A central difficulty in PP is that the learner is no longer optimizing over a fixed distribution. Previous works mainly focused on the optimization in discriminative PP settings. Several works develop algorithms and convergence guarantees for finding performatively stable solutions. Perdomo et al. [36] introduced repeated risk minimization and repeated gradient descent under sensitivity and convexity assumptions. Subsequent work studied stochastic and online variants, including greedy and lazy deployment schemes [31, 32]. State-dependent performative prediction further generalizes the framework by allowing the next distribution to depend not only on the deployed model but also on the current population state [5, 25]. More recent work has moved beyond the strongly convex setting. For example, Mofakhami et al. [34] considered neural-network-based PP under structural assumptions, and Li and Wai [26] established convergence to stationary performatively stable points for smooth non-convex losses, though with variance-dependent error terms. Another line of work directly optimizes the performative risk or seeks performatively optimal solutions under additional assumptions on the distribution map [18, 29, 33, 38, 61]. Others study PP under weaker convexity assumptions, distributional robustness, privacy, or welfare-oriented objectives [20, 22, 23, 27, 51, 52, 60, 62]. These works show that the choice of solution concept is not only technical: it determines whether the learner aims for equilibrium-like stability, direct performative optimality, or robustness to uncertainty in the distributional response.

A related perspective on endogenous user responses comes from preference dynamics under personalized recommendations [7, 55] and strategic behavior [21, 49, 50, 56, 57]. Dean and Morgenstern [7] study how users preferences evolve through interaction with a recommendation platform. Their setting is closely related on the preference side, but the platform selects from a fixed catalog without generative retraining on user-curated outputs. Our framework additionally closes the self-consuming training loop: model outputs shape user preferences, the evolving preferences guide curation, and the retained outputs are used to retrain the generator. The resulting dynamics involve both the generative distribution and the underlying preference, bringing curation-induced generator collapse and reference-data mixing into the analysis of preference evolution.

## B Complete Instantiation Examples

This appendix provides concrete examples that instantiate every symbol in our framework across several domains.   
We give two discrete examples (Appx. B.1) and one continuous examples (Appx. B.2).

## B.1 Discrete Examples

Example A: Image Generation with Object Classes Consider an image generation platform with $n = 1 0$ object categories.

• Sample space: $\mathcal { X } = \{ x _ { 1 } , \ldots , x _ { 1 0 } \}$ , representing the types: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

• Distribution space: $\mathcal { P } ( \mathcal { X } )$ is the probability simplex over 10 types.

• Feature map: $\varphi : \mathcal { X } \to \mathbb { R } ^ { 1 0 }$ is the softmax output of a pretrained classifier, normalized to $\| \varphi ( x _ { i } ) \| = 1$

Concrete feature vectors. After normalization, the feature vectors for three representative types are:

$$
\begin{array} { r l } & { \varphi ( x _ { \mathrm { a i r p l a n e } } ) = \frac { \left( 0 . 9 5 , \ 0 . 0 2 , \ 0 . 0 1 , \ 0 . 0 0 , \ 0 . 0 0 , \ 0 . 0 0 , \ 0 . 0 0 , \ 0 . 0 1 , \ 0 . 0 1 , \ 0 . 0 0 \right) } { \left\| \cdot \right\| } } \\ & { \approx ( 0 . 9 9 9 , \ 0 . 0 2 1 , \ 0 . 0 1 1 , \ 0 . 0 0 0 , \ 0 . 0 0 0 , \ 0 . 0 0 0 , \ 0 . 0 0 0 , \ 0 . 0 1 1 , \ 0 . 0 1 1 , \ 0 . 0 0 0 ) , } \end{array}
$$

$$
\begin{array} { r l } & { \varphi ( x _ { \mathrm { c a t } } ) = \frac { \left( 0 . 0 1 , \ 0 . 0 1 , \ 0 . 0 3 , \ 0 . 9 0 , \ 0 . 0 1 , \ 0 . 0 2 , \ 0 . 0 0 , \ 0 . 0 1 , \ 0 . 0 0 , \ 0 . 0 1 \right) } { \left\| \cdot \right\| } } \\ & { \approx \left( 0 . 0 1 1 , \ 0 . 0 1 1 , \ 0 . 0 3 3 , \ 0 . 9 9 8 , \ 0 . 0 1 1 , \ 0 . 0 2 2 , \ 0 . 0 0 0 , \ 0 . 0 1 1 , \ 0 . 0 0 0 , \ 0 . 0 1 1 \right) , } \end{array}
$$

$$
\begin{array} { r l } & { \varphi ( x _ { \mathrm { f r o g } } ) = \frac { \left( 0 . 0 0 , \ 0 . 0 0 , \ 0 . 0 2 , \ 0 . 0 1 , \ 0 . 0 2 , \ 0 . 0 1 , \ 0 . 9 2 , \ 0 . 0 1 , \ 0 . 0 0 , \ 0 . 0 1 \right) } { \left\| \cdot \right\| } } \\ & { \qquad \approx \mathrm { ( 0 . 0 0 0 , \ 0 . 0 0 0 , \ 0 . 0 2 2 , \ 0 . 0 1 1 , \ 0 . 0 2 2 , \ 0 . 0 1 1 , \ 0 . 9 9 9 , \ 0 . 0 1 1 , \ 0 . 0 0 0 , \ 0 . 0 1 1 ) } . } \end{array}
$$

Each vector is dominated by one coordinate, reflecting high classifier confidence.

Initial distribution. $p _ { 0 } = ( 0 . 1 0 , 0 . 1 0 , \dots , 0 . 1 0 )$ (uniform across all types).

User preference. Suppose the user prefers airplane and ship content. The initial preference vector is

$$
w _ { 0 } = \frac { ( 0 . 6 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 . 4 , \ 0 ) } { | | ( 0 . 6 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 . 4 , \ 0 ) | | } \approx ( 0 . 8 3 2 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 , \ 0 . 5 5 5 , \ 0 ) ,
$$

with $\| w _ { 0 } \| = 1$

Reward computation. The reward $r _ { 0 } ( x ) = \langle w _ { 0 } , \varphi ( x ) \rangle$ at round $t = 0 \colon$

$$
{ \begin{array} { r } { r _ { 0 } ( x _ { \mathrm { a i r p l a n e } } ) = 0 . 8 3 2 \times 0 . 9 9 9 + 0 . 5 5 5 \times 0 . 0 1 1 + \cdot \cdot \cdot \approx 0 . 8 3 7 , } \\ { r _ { 0 } ( x _ { \mathrm { c a t } } ) = 0 . 8 3 2 \times 0 . 0 1 1 + 0 . 5 5 5 \times 0 . 0 0 0 + \cdot \cdot \cdot \approx 0 . 0 0 9 , } \\ { r _ { 0 } ( x _ { \mathrm { f r o g } } ) = 0 . 8 3 2 \times 0 . 0 0 0 + 0 . 5 5 5 \times 0 . 0 0 0 + \cdot \cdot \cdot \approx 0 . 0 0 0 . } \end{array} }
$$

Airplane receives reward ≈ 0.84, while cat and frog receive near-zero reward. With curation strength $\tau > 0 ,$ the K-way Luce rule strongly favors airplane. Over successive rounds, p<sub>t</sub> concentrates on airplane, w<sub>t</sub> drifts further toward the airplane direction, and frog and horse vanish (Thm. 3.3).

Protection. To prevent frog and horse from disappearing, the designer sets $L = 2 \cdot$

$$
\begin{array} { r l } & { v _ { 1 } = ( \underbrace { 0 } _ { 0 } , \underbrace { 0 } _ { 1 } , \underbrace { 0 } _ { 2 } , \underbrace { 0 } _ { 2 } , \underbrace { 0 } _ { 4 3 } , \underbrace { 0 } _ { 4 } , \underbrace { 0 } _ { 5 } , \underbrace { 1 } _ { 6 } , \underbrace { 1 } _ { 7 } , \underbrace { 0 } _ { 6 } , \underbrace { 0 } _ { 7 } , \underbrace { 0 } _ { 8 } , \underbrace { 0 } _ { 9 } ) , \quad \theta _ { 1 } = 0 . 0 5 \quad ( \mathrm { f r o g \ge 5 \% } ) , } \\ & { v _ { 2 } = ( \underbrace { 0 } _ { 0 } , \underbrace { 0 } _ { 1 } , \underbrace { 0 } _ { 2 } , \underbrace { 0 } _ { 6 } , \underbrace { 0 } _ { 4 } , \underbrace { 0 } _ { 5 } , \underbrace { 0 } _ { 6 } , \underbrace { 0 } _ { 7 } , \underbrace { 0 } _ { 6 } , \underbrace { 1 } _ { 7 } , \underbrace { 0 } _ { 8 } , \underbrace { 0 } _ { 9 } ) , \quad \theta _ { 2 } = 0 . 0 5 \quad ( \mathrm { h o r s e \ge 5 \% } ) . } \end{array}
$$

The constraint $\langle v _ { 1 } , \bar { \varphi } ( p ^ { \star } ) \rangle \geq 0 . 0 5$ ensures that frog-type images constitute at least 5% of the equilibrium output in the feature-weighted sense.

Cost. If real-world images are equally easy to collect across all types, we set $c _ { i } = 1$ uniformly. In practice, some types are naturally rarer. For instance, high-quality frog images may be scarce compared to airplane or cat images, making them more expensive to collect. Setting $c _ { \mathrm { f r o g } } = 5$ reflects this scarcity.

Example B: Text Generation with Writing Style Consider a text generation platform where outputs are categorized by writing style.

• Sample space: ${ \mathcal { X } } = \{ x _ { 1 } , \ldots , x _ { 5 } \}$ , representing five style types: formal, casual, poetic, technical, humorous.

• Distribution space: $\mathcal { P } ( \mathcal { X } ) = \Delta _ { 5 }$

• Feature map: $\varphi : \mathcal { X } \to \mathbb { R } ^ { 5 }$ is the softmax output of a fine-tuned style classifier, normalized to $\| \varphi ( x _ { i } ) \| = 1$

Concrete feature vectors.

$$
\varphi ( x _ { \mathrm { c a s u a l } } ) \approx ( 0 . 0 3 3 , 0 . 9 9 7 , 0 . 0 3 3 , 0 . 0 2 2 , 0 . 0 5 5 ) / \| \cdot \| ,
$$

$$
\varphi ( x _ { \mathrm { p o e t i c } } ) \approx ( 0 . 0 5 5 , \ 0 . 0 3 3 , \ 0 . 9 9 3 , \ 0 . 0 2 2 , \ 0 . 0 5 5 ) / \| \cdot \| ,
$$

$$
\varphi ( x _ { \mathrm { t e c h n i c a l } } ) \approx ( 0 . 0 4 4 , 0 . 0 2 2 , 0 . 0 1 1 , 0 . 9 9 5 , 0 . 0 1 1 ) / \| \cdot \| .
$$

User preference and reward. Most users prefer casual content:

$$
w _ { 0 } = \frac { ( 0 . 1 , \ 0 . 8 , \ 0 . 0 , \ 0 . 0 , \ 0 . 1 ) } { \lVert ( 0 . 1 , \ 0 . 8 , \ 0 . 0 , \ 0 . 0 , \ 0 . 1 ) \rVert } \approx ( 0 . 1 2 4 , \ 0 . 9 9 0 , \ 0 . 0 0 0 , \ 0 . 0 0 0 , \ 0 . 1 2 4 ) .
$$

Then:

$$
\begin{array} { r l } & { r _ { 0 } ( x _ { \mathrm { c a s u a l } } ) = \left. w _ { 0 } , \varphi ( x _ { \mathrm { c a s u a l } } ) \right. \approx 0 . 9 9 , } \\ & { r _ { 0 } ( x _ { \mathrm { p o e t i c } } ) = \left. w _ { 0 } , \varphi ( x _ { \mathrm { p o e t i c } } ) \right. \approx 0 . 0 4 , } \\ & { r _ { 0 } ( x _ { \mathrm { t e c h n i c a l } } ) = \left. w _ { 0 } , \varphi ( x _ { \mathrm { t e c h n i c a l } } ) \right. \approx 0 . 0 2 . } \end{array}
$$

Casual text dominates curation. Over rounds, poetic and technical writing vanish.

Protection. Preserve poetic and technical writing:

$$
\begin{array} { r l } & { v _ { 1 } = ( 0 , \ 0 , \ 1 , \ 0 , \ 0 ) , \quad \theta _ { 1 } = 0 . 0 8 \quad \mathrm { ( p o e t i c \ge 8 \% ) } , } \\ & { v _ { 2 } = ( 0 , \ 0 , \ 0 , \ 1 , \ 0 ) , \quad \theta _ { 2 } = 0 . 0 5 \quad \mathrm { ( t e c h n i c a l \ge 5 \% ) } . } \end{array}
$$

Cost. $c _ { \mathrm { p o e t i c } } = 3$ (high-quality poetic text requires expert curation), $c _ { \mathrm { c a s u a l } } = 1$ (abundant online).

## B.2 Continuous Examples

Example C: Image Generation with CLIP Embedding Consider an image generation platform where outputs live in a continuous space.

• Sample space: $\mathcal { X } \subset \mathbb { R } ^ { m }$ is a compact subset of image space.

• Distribution space: $\mathcal { P } ( \mathcal { X } )$ is the set of all Borel probability measures on X.

• Feature map: $\varphi : \mathcal { X } \to \mathbb { R } ^ { 5 1 2 }$ is a pretrained CLIP image encoder, with $\| \varphi ( x ) \| = 1$ . Feature dimension $d = 5 1 2$ . Semantically similar images have nearby embeddings.

Concrete feature vectors. Three representative images and their (truncated) CLIP embeddings:

$$
\begin{array} { r l } & { \varphi ( x _ { \mathrm { s u n s e t } } ) = ( 0 . 1 2 , - 0 . 0 8 , 0 . 4 3 , \dots ) _ { 5 1 2 } , \quad \mathrm { ( v i b r a n t , w a r m ~ t o n e s ) } } \\ & { \varphi ( x _ { \mathrm { p o r t r a i t } } ) = ( - 0 . 0 5 , 0 . 3 1 , 0 . 1 5 , \dots ) _ { 5 1 2 } , \quad \mathrm { ( h u m a n ~ s u b j e c t , n e u t r a l ) } } \\ & { \varphi ( x _ { \mathrm { m o n o } } ) = ( - 0 . 2 2 , - 0 . 1 4 , - 0 . 3 5 , \dots ) _ { 5 1 2 } , \quad \mathrm { ( d a r k , m o n o c h r o m e ) } } \end{array}
$$

each normalized to $\| \varphi \| = 1$

User preference and reward. Users consistently select vibrant, colorful images. After 5 rounds, the preference vector has drifted to

$$
w _ { 5 } = ( 0 . 1 5 , ~ - 0 . 0 6 , ~ 0 . 3 8 , ~ \ldots ) _ { 5 1 2 } , \quad \| w _ { 5 } \| = 0 . 7 3 .
$$

The reward for each image:

$$
\begin{array} { r } { r _ { 5 } ( x _ { \mathrm { s u n s e t } } ) = \langle w _ { 5 } , \varphi ( x _ { \mathrm { s u n s e t } } ) \rangle \approx 0 . 7 1 , } \\ { r _ { 5 } ( x _ { \mathrm { p o r t r a i t } } ) = \langle w _ { 5 } , \varphi ( x _ { \mathrm { p o r t r a i t } } ) \rangle \approx 0 . 1 8 , } \\ { r _ { 5 } ( x _ { \mathrm { m o n o } } ) = \langle w _ { 5 } , \varphi ( x _ { \mathrm { m o n o } } ) \rangle \approx - 0 . 4 1 . } \end{array}
$$

Sunset images receive high reward; monochrome images receive negative reward. Curation suppresses monochrome content, $p _ { t }$ concentrates on vibrant images, w drifts further from the monochrome direction.

Mean feature vector. The equilibrium mean feature vector is

$$
\bar { \varphi } ( p ^ { \star } ) = \int _ { \mathcal { X } } \varphi ( x ) p ^ { \star } ( d x ) \in \mathbb { R } ^ { 5 1 2 } .
$$

As $p _ { t }$ concentrates on vibrant images, $, \bar { \varphi } ( \boldsymbol { p } _ { t } )$ moves toward the vibrant cluster in CLIP space and away from the monochrome direction.

Protection. To preserve monochrome imagery, the designer collects 500 reference monochrome images and computes their mean CLIP embedding:

$$
v _ { 1 } = \frac { \frac { 1 } { 5 0 0 } \sum _ { j = 1 } ^ { 5 0 0 } \varphi ( x _ { j } ^ { \mathrm { m o n o } } ) } { \left\| \frac { 1 } { 5 0 0 } \sum _ { j = 1 } ^ { 5 0 0 } \varphi ( x _ { j } ^ { \mathrm { m o n o } } ) \right\| } \in \mathbb { R } ^ { 5 1 2 } , \qquad \theta _ { 1 } = 0 . 0 5 .
$$

The constraint $\langle v _ { 1 } , \bar { \varphi } ( p ^ { \star } ) \rangle \geq 0 . 0 5$ ensures the equilibrium retains at least 0.05 projection onto the monochrome direction in CLIP space.

Cost. $c ( x )$ varies across CLIP space: reference images in the monochrome region are expensive to source (niche artistic photography), while vibrant natural images are cheap (abundant on the web).

## C Proofs

## C.1 Proof of Explicit Retraining Update

Lemma C.1 (Explicit retraining update). Let $p _ { t + 1 }$ be defined by (2). Then,

$$
p _ { t + 1 } ( x ) = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p _ { t } ( x ) H _ { p _ { t } , w _ { t } } ^ { K } ( x ) ,\tag{9}
$$

where

$$
H _ { p _ { t } , w _ { t } } ^ { K } ( x ) : = \mathbb { E } _ { x _ { 1 } , \ldots , x _ { K - 1 } \sim p _ { t } } \left[ \frac { K e ^ { \tau \cdot r _ { w _ { t } } ( x ) } } { e ^ { \tau \cdot r _ { w _ { t } } ( x ) } + \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau \cdot r _ { w _ { t } } ( x _ { i } ) } } \right] .
$$

Moreover, for every fixed $x \in { \mathcal { X } } ,$

$$
H _ { p t , w _ { t } } ^ { K } ( x ) \xrightarrow { K  \infty } \frac { e ^ { \tau r _ { t } ( x ) } } { \mathbb { E } _ { z \sim p _ { t } } [ e ^ { \tau r _ { t } ( z ) } ] } .
$$

Proof. First, by minimization of the cross-entropy, we know that for any distribution $q ~ \in ~ \mathcal { P } ( \mathcal { X } )$ arg $\begin{array} { r } { \operatorname* { m a x } _ { p \in \mathcal { P } ( \mathcal { X } ) } \mathbb { E } _ { x \sim q } [ \log p ( x ) ] = q } \end{array}$

Case 1: $\eta = 1$ . When $\eta = 1$ , the objective in (2) reduces to

$$
p _ { t + 1 } = \arg \operatorname* { m a x } _ { p \in \mathcal { P } ( \mathcal { X } ) } \mathbb { E } _ { \hat { x } \sim \mathrm { L u c e } _ { K } ( x _ { 1 } , \ldots , x _ { K } ) } [ \log p ( \hat { x } ) ]
$$

Let $p _ { t } ^ { \mathrm { c u r } }$ denote the distribution of the retained sample xˆ. By the population MLE argument above, $p _ { t + 1 } = p _ { t } ^ { \mathrm { c u r } }$ To characterize $p _ { t } ^ { \mathrm { c u r } }$ , fix round t and sample $\begin{array} { r } { X _ { 1 } , \dotsc , X _ { K } \stackrel { \mathrm { i . i . d . } } { \sim } p _ { t } } \end{array}$ . Let $i _ { K } \in [ K ]$ be selected according to

$$
\operatorname* { P r } ( i _ { K } = i \mid x _ { 1 } , \dots , x _ { K } ) = { \frac { e ^ { \tau r _ { t } ( x _ { i } ) } } { \sum _ { j = 1 } ^ { K } e ^ { \tau r _ { t } ( x _ { j } ) } } } .
$$

Define the retained sample by $\hat { X } : = X _ { i _ { K } }$ . Then $p _ { t } ^ { \mathrm { c u r } }$ is the law of ${ \hat { X } } .$ By symmetry of the K candidates,

$$
{ \begin{array} { r l } & { p _ { t + 1 } ( x ) = p _ { t } ^ { \mathrm { c u r } } ( x ) = \displaystyle \sum _ { i = 1 } ^ { K } \int _ { y _ { j } , ~ j \neq i } { p _ { t } ( y _ { 1 } , \dots , y _ { i - 1 } , x , y _ { i + 1 } , \dots , y _ { K } ) \operatorname* { P r } ( i _ { K } = i \mid x , y _ { j } , ~ j \neq i ) { \prod _ { j \neq i } } } } \\ & { \qquad = K \int _ { y _ { 1 } , \dots , y _ { K - 1 } } p _ { t } ( y _ { 1 } , \dots , y _ { K - 1 } , x ) \operatorname* { P r } ( i _ { K } = K \mid y _ { 1 } , \dots , y _ { K - 1 } , x ) d y _ { 1 } \cdots d y _ { K - 1 } } \\ & { \qquad = p _ { t } ( x ) K \int _ { y _ { 1 } , \dots , y _ { K - 1 } } { \frac { e ^ { \tau r _ { t } ( x ) } } { e ^ { \tau r _ { t } ( x ) } + \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau r _ { t } ( y _ { i } ) } } } \prod _ { i = 1 } ^ { K - 1 } p _ { t } ( y _ { i } ) d y _ { 1 } \cdots d y _ { K - 1 } . } \end{array} }
$$

Therefore,

$$
p _ { t + 1 } ( x ) = p _ { t } ( x ) H _ { p _ { t } , w _ { t } } ^ { K } ( x ) ,
$$

where

$$
\begin{array} { l } { { \displaystyle H _ { p _ { t } , w _ { t } } ^ { K } ( x ) : = K \int _ { y _ { 1 } , \dots , y _ { K - 1 } } \frac { e ^ { \tau r _ { t } ( x ) } } { e ^ { \tau r _ { t } ( x ) } + \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau r _ { t } ( y _ { i } ) } } \prod _ { i = 1 } ^ { K - 1 } p _ { t } ( y _ { i } ) d y _ { 1 } \cdots d y _ { K - 1 } } } \\ { { \displaystyle = \mathbb { E } _ { { \cal X } _ { 1 } , \dots , { \cal X } _ { K - 1 } } } \hat { \tilde { \tilde { \mathbf { \alpha } } } } _ { \sim } ^ { \mathrm { i . i . d . } } p _ { t } \left[ \frac { K e ^ { \tau r _ { t } ( x ) } } { e ^ { \tau r _ { t } ( x ) } + \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau r _ { t } ( X _ { i } ) } } \right] . } \end{array}
$$

Now consider the large-K limit. Since $r _ { t } ( x ) \in [ - 1 , 1 ]$ , we have

$$
e ^ { - \tau } \leq e ^ { \tau r _ { t } ( x ) } \leq e ^ { \tau } \qquad { \mathrm { f o r ~ a l l ~ } } x \in { \mathcal { X } } .
$$

Rewrite the denominator as

$$
\frac { e ^ { \tau r _ { t } ( x ) } } { K } + \frac { K - 1 } { K } \cdot \frac { 1 } { K - 1 } \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau r _ { t } ( X _ { i } ) } .
$$

By the strong law of large numbers,

$$
\frac { 1 } { K - 1 } \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau r _ { t } ( X _ { i } ) } \xrightarrow { \mathrm { a . s . } } \mathbb { E } _ { z \sim p _ { t } } \bigl [ e ^ { \tau r _ { t } ( z ) } \bigr ] .
$$

Hence, for every fixed $x \in \mathcal { X }$

$$
\frac { e ^ { \tau r _ { t } ( x ) } } { \frac { e ^ { \tau r _ { t } ( x ) } } { K } + \frac { K - 1 } { K } \cdot \frac { 1 } { K - 1 } \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau r _ { t } ( X _ { i } ) } } \xrightarrow [ ] { \mathrm { a . s . } } \frac { e ^ { \tau r _ { t } ( x ) } } { \mathbb { E } _ { z \sim p _ { t } } \left[ e ^ { \tau r _ { t } ( z ) } \right] } .
$$

Therefore,

$$
H _ { p _ { t } , w _ { t } } ^ { K } ( x ) \xrightarrow { K  \infty } \frac { e ^ { \tau r _ { t } ( x ) } } { \mathbb { E } _ { z \sim p _ { t } } [ e ^ { \tau r _ { t } ( z ) } ] } .
$$

Consequently, in the large-K limit,

$$
p _ { t + 1 } ( x ) \xrightarrow { K  \infty } p _ { t } ( x ) \frac { e ^ { \tau r _ { t } ( x ) } } { \mathbb { E } _ { z \sim p _ { t } } [ e ^ { \tau r _ { t } ( z ) } ] }
$$

Moreover, for every $K \geq 2 ,$

$$
0 \leq H _ { p _ { t } , w _ { t } } ^ { K } ( x ) \leq \frac { K e ^ { \tau } } { e ^ { - \tau } + ( K - 1 ) e ^ { - \tau } } \leq 2 e ^ { 2 \tau } .
$$

Hence the family $\{ H _ { p _ { t } , w _ { t } } ^ { K } ( x ) \} _ { K \geq 2 }$ is uniformly bounded. By dominated convergence,

$$
\int _ { \mathcal { X } } \bigg | H _ { p _ { t } , w _ { t } } ^ { K } ( x ) - \frac { e ^ { \tau r _ { t } ( x ) } } { \mathbb { E } _ { z \sim p _ { t } } \left[ e ^ { \tau r _ { t } ( z ) } \right] } \bigg | p _ { t } ( d x ) \to 0 .
$$

This proves the total-variation convergence of the curated distribution to its large-K exponential-reweighting limit.

Case 2: $\eta < 1$ . When $\eta < 1$ , the objective in (2) can be written as

$$
( 1 - \eta ) \mathbb { E } _ { x \sim p _ { \mathrm { r e f } } } [ \log p ( x ) ] + \eta \mathbb { E } _ { x \sim p _ { t } ^ { \mathrm { c u r } } } [ \log p ( x ) ] = \mathbb { E } _ { x \sim q _ { t } } [ \log p ( x ) ] ,
$$

where

$$
q _ { t } : = ( 1 - \eta ) p _ { \mathrm { r e f } } + \eta p _ { t } ^ { \mathrm { c u r } } .
$$

Applying the same population MLE argument yields

$$
p _ { t + 1 } = q _ { t } = ( 1 - \eta ) p _ { \mathrm { r e f } } + \eta p _ { t } ^ { \mathrm { c u r } } .
$$

For finite K, since

$$
p _ { t } ^ { \mathrm { c u r } } ( x ) = p _ { t } ( x ) H _ { p _ { t } , w _ { t } } ^ { K } ( x ) ,
$$

we obtain

$$
\begin{array} { r l } & { p _ { t + 1 } ( x ) = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p _ { t } ^ { \mathrm { c u r } } ( x ) } \\ & { \qquad = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p _ { t } ( x ) H _ { p _ { t } , w _ { t } } ^ { K } ( x ) } \\ & { \qquad = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p _ { t } ( x ) \mathbb { E } _ { X _ { 1 } , \ldots , X _ { K - 1 } \overset { \mathrm { i . i . d . } } { \sim } p _ { t } } \left[ \frac { K e ^ { \tau r _ { t } ( x ) } } { e ^ { \tau r _ { t } ( x ) } + \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau r _ { t } ( X _ { i } ) } } \right] . } \end{array}
$$

As $K  \infty ,$

$$
H _ { p _ { t } , w _ { t } } ^ { K } ( x ) \to \frac { e ^ { \tau r _ { t } ( x ) } } { \mathbb { E } _ { z \sim p _ { t } } [ e ^ { \tau r _ { t } ( z ) } ] } ,
$$

and therefore

$$
p _ { t + 1 } ( x ) \xrightarrow { K  \infty } ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p _ { t } ( x ) \frac { e ^ { \tau r _ { t } ( x ) } } { \mathbb { E } _ { z \sim p _ { t } } [ e ^ { \tau r _ { t } ( z ) } ] } .
$$

## C.2 Proof of Propostion 3.1

Proposition 3.1 (Basic properties of curation operator). For any $p \in { \mathcal { P } } ( { \mathcal { X } } )$ , w $\in { \mathcal { W } } ,$ , and $K \geq 2 ,$ the curation operator $H _ { p , w } ^ { K }$ satisfies the following properties:

$$
\textit { l . S t r i c t m o n o t o n i c i t y . F o r a n y x \boldsymbol { i } } , x _ { j } \in \mathcal { X } , r _ { w } ( x _ { i } ) > r _ { w } ( x _ { j } ) \Longleftrightarrow H _ { p , w } ^ { K } ( x _ { i } ) > H _ { p , w } ^ { K } ( x _ { j } )
$$

2. Normalization. $\begin{array} { r } { \sum _ { i = 1 } ^ { n } p ( x _ { i } ) H _ { p , w } ^ { K } ( x _ { i } ) = 1 } \end{array}$ .

3. Odds contraction. $\begin{array} { r } { \bar { I f } \bar { r } _ { w } ( x _ { i } ) - \bar { r } _ { w } ( x _ { j } ) \geq r _ { \Delta } > 0 , } \end{array}$ then

$$
\frac { H _ { p , w } ^ { K } ( x _ { j } ) } { H _ { p , w } ^ { K } ( x _ { i } ) } \leq \lambda _ { K } ( r _ { \Delta } ) : = 1 - \frac { ( 1 - e ^ { - \tau \cdot r _ { \Delta } } ) ( K - 1 ) } { e ^ { 2 \tau } + K - 1 } ,
$$

with $\lambda _ { K } ( r _ { \Delta } ) \in ( 0 , 1 )$ and $\lambda _ { K } ( \boldsymbol { r } _ { \Delta } ) \xrightarrow { K  \infty } e ^ { - \tau \cdot \boldsymbol { r } _ { \Delta } }$

Proof. Write

$$
\begin{array} { r } { a ( x ) : = e ^ { \tau r _ { w } ( x ) } } \end{array}
$$

For a fixed $X _ { 1 } , \ldots , X _ { K - 1 } \overset { \mathrm { i i d } } { \sim } p ,$ let

$$
S : = \sum _ { l = 1 } ^ { K - 1 } a ( X _ { l } ) \geq 0
$$

Since $a ( X _ { \ell } ) > 0$ and $K \geq 2 .$ , we have $S > 0$

So

$$
H _ { p , w } ^ { K } ( x ) = \mathbb { E } _ { \boldsymbol { X } _ { 1 } , \ldots , \boldsymbol { X } _ { K - 1 } \cdot \stackrel { \mathrm { i } \cdot \mathrm { i } \cdot \mathrm { d } \cdot } { \sim } p } \left[ \frac { K e ^ { \tau r _ { w } ( x ) } } { e ^ { \tau r _ { w } ( x ) } + \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau r _ { w } ( \boldsymbol { X } _ { i } ) } } \right] = \mathbb { E } _ { \boldsymbol { X } _ { 1 } , \ldots , \boldsymbol { X } _ { K - 1 } \cdot \stackrel { \mathrm { i } \cdot \mathrm { i } \cdot \mathrm { d } \cdot } { \sim } p } \frac { K a ( \boldsymbol { x } ) } { a ( \boldsymbol { x } ) + S }
$$

We prove the three claims in order.

Part 1: strict monotonicity. Define

$$
f _ { S } ( a ) : = \frac { K a } { a + S }
$$

Then

$$
f _ { S } ^ { \prime } ( a ) = \frac { K ( a + S ) - K a } { ( a + S ) ^ { 2 } } = \frac { K S } { ( a + S ) ^ { 2 } } > 0
$$

Hence $f _ { S } ( a )$ is strictly increasing in $a \in ( 0 , \infty )$ .

On the other hand,

$$
a ^ { \prime } ( x ) = \frac { d } { d r _ { w } ( x ) } e ^ { \tau r _ { w } ( x ) } = \tau e ^ { \tau r _ { w } ( x ) } > 0
$$

so $a ( x )$ is strictly increasing in $r _ { w } ( x )$

Therefore, for every fixed sample path,

$$
r _ { w } ( x _ { i } ) > r _ { w } ( x _ { j } ) \quad \Longrightarrow \quad a ( x _ { i } ) > a ( x _ { j } ) \quad \Longrightarrow \quad f _ { S } ( a ( x _ { i } ) ) > f _ { S } ( a ( x _ { j } ) )
$$

That is,

$$
r _ { w } ( x _ { i } ) > r _ { w } ( x _ { j } ) \quad \Longrightarrow \quad \frac { K e ^ { \tau r _ { w } ( x _ { i } ) } } { e ^ { \tau r _ { w } ( x _ { i } ) } + S } > \frac { K e ^ { \tau r _ { w } ( x _ { j } ) } } { e ^ { \tau r _ { w } ( x _ { j } ) } + S }
$$

Taking expectation over $X _ { 1 } , \ldots , X _ { K - 1 }$ yields

$$
r _ { w } ( x _ { i } ) > r _ { w } ( x _ { j } ) \quad \Longrightarrow \quad H _ { p , w } ^ { K } ( x _ { i } ) > H _ { p , w } ^ { K } ( x _ { j } )
$$

Conversely, since $H _ { p , w } ^ { K } ( x )$ is a strictly increasing function of $r _ { w } ( x )$ , it also reflects strict order, and hence

$$
r _ { w } ( x _ { i } ) > r _ { w } ( x _ { j } ) \quad \Longleftrightarrow \quad H _ { p , w } ^ { K } ( x _ { i } ) > H _ { p , w } ^ { K } ( x _ { j } )
$$

For $K = \infty ,$ we have

$$
H _ { p , w } ^ { \infty } ( x ) = \frac { e ^ { \tau r _ { w } ( x ) } } { \mathbb { E } _ { z \sim p } [ e ^ { \tau r _ { w } ( z ) } ] }
$$

where $\mathbb { E } _ { z \sim p } [ e ^ { \tau r _ { w } ( z ) } ]$ is independent of $x .$ Thus strict monotonicity is immediate.

Part 2: normalization. For sample $\begin{array} { r } { X _ { 1 } , \dotsc , X _ { K } \stackrel { \mathrm { i . i . d . } } { \sim } p _ { t } } \end{array}$ , let $i _ { K } \in [ K ]$ be selected according to

$$
\operatorname* { P r } ( i _ { K } = i \mid x _ { 1 } , \dots , x _ { K } ) = { \frac { e ^ { \tau r _ { t } ( x _ { i } ) } } { \sum _ { j = 1 } ^ { K } e ^ { \tau r _ { t } ( x _ { j } ) } } }
$$

Define the retained sample by $\hat { X } : = X _ { i _ { K } } . \mathbf { B } \mathbf { y }$ the derivation in Lemma C.1, the law of $\hat { X }$ is precisely

$$
\mathrm { P r } ( \hat { X } = x _ { i } ) = p ( x _ { i } ) H _ { p , w } ^ { K } ( x _ { i } )
$$

Therefore,

$$
\sum _ { i = 1 } ^ { n } p ( x _ { i } ) H _ { p , w } ^ { K } ( x _ { i } ) = \sum _ { i = 1 } ^ { n } \operatorname* { P r } ( \hat { X } = x _ { i } ) = 1
$$

For $K = \infty ,$ , we have

$$
H _ { p , w } ^ { \infty } ( x ) = \frac { e ^ { \tau r _ { w } ( x ) } } { \mathbb { E } _ { z \sim p } [ e ^ { \tau r _ { w } ( z ) } ] }
$$

so

$$
\sum _ { i = 1 } ^ { n } p ( x _ { i } ) H _ { p , w } ^ { \infty } ( x _ { i } ) = \frac { \sum _ { i = 1 } ^ { n } p ( x _ { i } ) e ^ { \tau r _ { w } ( x _ { i } ) } ) } { \mathbb { E } _ { z \sim p } \left[ e ^ { \tau r _ { w } ( z ) } \right] } = 1
$$

## Part 3: explicit odds contraction. Write

$$
a ( x _ { i } ) : = e ^ { \tau r _ { w } ( x _ { i } ) } \qquad a ( x _ { j } ) : = e ^ { \tau r _ { w } ( x _ { j } ) }
$$

If $r _ { w } ( x _ { i } ) - r _ { w } ( x _ { j } ) \geq r _ { \Delta } > 0 ,$ , then

$$
a ( x _ { j } ) \leq e ^ { - \tau r _ { \Delta } } a ( x _ { i } )
$$

and therefore

$$
a ( x _ { i } ) - a ( x _ { j } ) \geq ( 1 - e ^ { - \tau r _ { \Delta } } ) a ( x _ { i } )
$$

Let

$$
S : = \sum _ { l = 1 } ^ { K - 1 } a ( X _ { l } ) \geq 0
$$

For any $S \geq 0 ,$

$$
{ \frac { a ( x _ { j } ) / ( a ( x _ { j } ) + S ) } { a ( x _ { i } ) / ( a ( x _ { i } ) + S ) } } = { \frac { a ( x _ { j } ) ( a ( x _ { i } ) + S ) } { a ( x _ { i } ) ( a ( x _ { j } ) + S ) } } = 1 - { \frac { S \cdot ( a ( x _ { i } ) - a ( x _ { j } ) ) } { a ( x _ { i } ) ( a ( x _ { j } ) + S ) } }
$$

So,

$$
\begin{array} { r } { \frac { H _ { p , w } ^ { K } ( x _ { j } ) } { H _ { p , w } ^ { K } ( x _ { i } ) } = \frac { \mathbb { E } _ { X _ { 1 } , \ldots , X _ { K - 1 } \overset { \mathrm { i . i . d . } } { \sim } p } \left[ \frac { K e ^ { \tau r w ( x _ { j } ) } } { e ^ { \tau r w ( x _ { j } ) } + \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau r w ( X _ { i } ) } } \right] } { \mathbb { E } _ { X _ { 1 } , \ldots , X _ { K - 1 } \overset { \mathrm { i . i . d . } } { \sim } p } \left[ \frac { K e ^ { \tau r w ( x _ { i } ) } } { e ^ { \tau r w ( x _ { i } ) } + \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau r w ( X _ { i } ) } } \right] } } \\ { = 1 - \mathbb { E } _ { X _ { 1 } , \ldots , X _ { K - 1 } \overset { \mathrm { i . i . d . } } { \sim } p } \frac { S \cdot \left( a ( x _ { i } ) - a ( x _ { j } ) \right) } { a ( x _ { i } ) ( a ( x _ { j } ) + S ) } } \end{array}\tag{10}
$$

Since $r _ { w } ( x ) \in [ - 1 , 1 ]$ , each $a ( x ) \in [ e ^ { - \tau } , e ^ { \tau } ]$

Using $a ( x _ { i } ) - a ( x _ { j } ) \geq ( 1 - e ^ { - \tau r _ { \Delta } } ) a ( x _ { i } )$ and $a ( x _ { j } ) \leq e ^ { \tau }$ , we obtain

$$
\frac { S \cdot ( a ( x _ { i } ) - a ( x _ { j } ) ) } { a ( x _ { i } ) ( a ( x _ { j } ) + S ) } \geq ( 1 - e ^ { - \tau r _ { \Delta } } ) \frac { S } { e ^ { \tau } + S }
$$

Since $a ( X _ { \ell } ) \geq e ^ { - \tau }$ so

$$
S \geq ( K - 1 ) e ^ { - \tau }
$$

Because the map $\begin{array} { r } { s \mapsto \frac { s } { ( e ^ { \tau } + s ) } } \end{array}$ is increasing,

$$
\mathbb { E } _ { X _ { 1 } , \ldots , X _ { K - 1 } \stackrel { \mathrm { i . i . d . } } { \sim } p \ e ^ { \tau } + S } \geq \frac { ( K - 1 ) e ^ { - \tau } } { e ^ { \tau } + ( K - 1 ) e ^ { - \tau } } = \frac { K - 1 } { e ^ { 2 \tau } + K - 1 }
$$

Substituting this into (10) yields

$$
\frac { H _ { p , w } ^ { K } ( x _ { j } ) } { H _ { p , w } ^ { K } ( x _ { i } ) } \leq 1 - ( 1 - e ^ { - \tau r _ { \Delta } } ) \frac { K - 1 } { e ^ { 2 \tau } + K - 1 } = : \lambda _ { K } ( r _ { \Delta } )
$$

For $K = \infty$ , the claim is immediate:

$$
\frac { H _ { p , w } ^ { \infty } ( x _ { j } ) } { H _ { p , w } ^ { \infty } ( x _ { i } ) } = \frac { \frac { e ^ { \tau r _ { w } ( x _ { j } ) } } { \mathbb { E } _ { z \sim p } [ e ^ { \tau r _ { w } ( z ) } ] } } { \frac { e ^ { \tau r _ { w } ( x _ { i } ) } } { \mathbb { E } _ { z \sim p } [ e ^ { \tau r _ { w } ( z ) } ] } } = \frac { a ( x _ { j } ) } { a ( x _ { i } ) } \leq e ^ { - \tau r _ { \Delta } }
$$

Finally,

$$
\lambda _ { K } ( r _ { \Delta } ) \xrightarrow { K  \infty } e ^ { - \tau r _ { \Delta } }
$$

And for $K \ge 2 , r _ { \Delta } > 0 ,$

$$
0 < \lambda _ { K } ( r _ { \Delta } ) < 1 .
$$

This completes the proof.

## C.3 Proof of Theorem 3.2

Theorem 3.2 (Equilibrium characterization under purely synthetic setting). When $\eta = 1$ , a distribution $p \in { \mathcal { P } } ( { \mathcal { X } } )$ is a fixed point of the retraining step under preference w if and only if all instances in its support share the same reward:

$$
p ( x ) > 0 , p ( x ^ { \prime } ) > 0 \implies r _ { w } ( x ) = r _ { w } ( x ^ { \prime } ) .
$$

Consequently, for every instance $x _ { i } \in { \mathcal { X } } ,$ the pair $( p ^ { \star } = \delta _ { x _ { i } } , w ^ { \star } = \varphi ( x _ { i } ) )$ is an equilibrium: the model has collapsed to a point mass on $x _ { i }$ while user preferences have locked onto its features.

Proof. We prove the two claims in turn.

Part 1: Data fixed-point characterization. (⇒) Suppose p is a fixed point of the data update at preference w, i.e., for all $x \in { \mathcal { X } } ,$

$$
p ( x ) H _ { p , w } ^ { K } ( x ) = p ( x )
$$

Then for every x with $p ( x ) > 0 .$

$$
H _ { p , w } ^ { K } ( x ) = 1
$$

Now let $x , x ^ { \prime } \in \mathcal { X }$ satisfy $p ( x ) > 0$ and $p ( x ^ { \prime } ) > 0 .$ . If

$$
r _ { w } ( x ) \neq r _ { w } ( x ^ { \prime } )
$$

then by Proposition 3.1 (strict monotonicity),

$$
r _ { w } ( x ) > r _ { w } ( x ^ { \prime } ) \quad \Longrightarrow \quad H _ { p , w } ^ { K } ( x ) > H _ { p , w } ^ { K } ( x ^ { \prime } )
$$

or vice versa.

But both x and $x ^ { \prime }$ lie in the support of $p ,$ so we already showed

$$
H _ { p , w } ^ { K } ( x ) = H _ { p , w } ^ { K } ( x ^ { \prime } ) = 1
$$

a contradiction. Therefore,

$$
p ( x ) > 0 , p ( x ^ { \prime } ) > 0 \quad \Longrightarrow \quad r _ { w } ( x ) = r _ { w } ( x ^ { \prime } )
$$

(⇐) Conversely, suppose that for all $x , x ^ { \prime } \in { \mathcal { X } }$

$$
p ( x ) > 0 , p ( x ^ { \prime } ) > 0 \quad \Longrightarrow \quad r _ { w } ( x ) = r _ { w } ( x ^ { \prime } )
$$

Let c denote the common reward value on the support of $p , { \mathrm { i . e . } }$ , for every x with $p ( x ) > 0$

$$
r _ { w } ( x ) = c
$$

Then for every such $x ,$

$$
e ^ { \tau r _ { w } ( x ) } = e ^ { \tau c }
$$

Hence, for any x in the support of $p ,$

$$
H _ { p , w } ^ { K } ( x ) = \mathbb { E } \left[ \frac { K e ^ { \tau c } } { e ^ { \tau c } + \sum _ { \ell = 1 } ^ { K - 1 } e ^ { \tau c } } \right] = 1
$$

Therefore,

$$
p ( x ) H _ { p , w } ^ { K } ( x ) = p ( x )
$$

so $p$ is a fixed point of the data update.

For $K = \infty ,$ , the same conclusion holds since

$$
H _ { p , w } ^ { \infty } ( x ) = \frac { e ^ { \tau c } } { \mathbb { E } _ { z \sim p } [ e ^ { \tau r _ { w } ( z ) } ] } = 1
$$

for every x with $p ( x ) > 0$

Part 2: Singleton equilibria. Fix $i \in \{ 1 , \ldots , n \}$ , and define $p ^ { \star }$ by

$$
p ^ { \star } ( x _ { i } ) = 1 , \qquad p ^ { \star } ( x _ { j } ) = 0 { \mathrm { ~ f o r ~ a l l ~ } } j \neq i ,
$$

and let

$$
\boldsymbol { w } ^ { \star } = \varphi ( \boldsymbol { x } _ { i } )
$$

Since $p ^ { \star }$ is supported on a single point, the condition in Part 1 is trivially satisfied, so $p ^ { \star }$ is a fixed point of the data update.

Moreover,

$$
\bar { \varphi } ( p ^ { \star } ) = \varphi ( x _ { i } ) = w ^ { \star }
$$

Therefore,

$$
\operatorname { P r o j } _ { \mathcal { W } } ( ( 1 - \beta ) w ^ { \star } + \beta \bar { \varphi } ( p ^ { \star } ) ) = \operatorname { P r o j } _ { \mathcal { W } } ( \varphi ( x _ { i } ) ) = \varphi ( x _ { i } ) = w ^ { \star }
$$

where we used that $\| \varphi ( x _ { i } ) \| = 1$ and W is the closed unit ball.

Hence $( p ^ { \star } , w ^ { \star } )$ satisfies both parts of (5), and is therefore an equilibrium of the pure coupled dynamics.

## C.4 Proof of Theorem 3.3

Theorem 3.3 (Local convergence to singleton equilibria). Fix $x _ { i } ~ \in ~ { \mathcal { X } }$ and suppose $\varphi ( x _ { j } ) \neq \varphi ( x _ { i } )$ $f o r a l l j \neq i .$ Define the reward gap $\begin{array} { r } { \gamma _ { i } = \operatorname* { m i n } _ { j \neq i } \langle \varphi ( x _ { i } ) , \varphi ( x _ { i } ) - \varphi ( x _ { j } ) \rangle > 0 } \end{array}$ and let $\lambda _ { K } ( \cdot )$ be the odds-contraction factor from Prop. 3.1. If the initial state satisfies

$$
1 - p _ { 0 } ( x _ { i } ) \leq \varepsilon _ { i } , \qquad \| w _ { 0 } - \varphi ( x _ { i } ) \| \leq \frac { \gamma _ { i } } { 4 }
$$

for some $\varepsilon _ { i } \in ( 0 ,$ min $\left\{ { \textstyle { \frac { 1 } { 2 } } , 1 - \lambda _ { K } \bigl ( { \frac { \gamma _ { i } } { 2 } } \bigr ) , \frac { \gamma _ { i } } { 8 } } \right\} \mathrm { ) }$ , then for all $t \geq 0 ,$

$$
1 - p _ { t } ( x _ { i } ) \leq \left( \frac { \lambda _ { K } \big ( \frac { \gamma _ { i } } { 2 } \big ) } { 1 - \varepsilon _ { i } } \right) ^ { t } \big ( 1 - p _ { 0 } ( x _ { i } ) \big ) .
$$

Consequently, $p _ { t } ( x _ { i } ) \xrightarrow { t  \infty } 1$ and $w _ { t } \xrightarrow { t  \infty } \varphi ( x _ { i } )$ .

Proof. Let

$$
q _ { t } : = 1 - p _ { t } ( x _ { i } ) = \sum _ { j \neq i } p _ { t } ( x _ { j } )
$$

denote the off-winner mass.

Since $\| \varphi ( x ) \| = 1$ for all x and $\varphi ( x _ { j } ) \neq \varphi ( x _ { i } )$ for every $j \neq i ,$ we have

$$
\gamma _ { i } = \underset { j \neq i } { \operatorname* { m i n } } \langle \varphi ( x _ { i } ) , \varphi ( x _ { i } ) - \varphi ( x _ { j } ) \rangle = \underset { j \neq i } { \operatorname* { m i n } } \big ( 1 - \langle \varphi ( x _ { i } ) , \varphi ( x _ { j } ) \rangle \big ) > 0
$$

Fix K, $\lambda _ { K } \big ( \frac { \gamma _ { i } } { 2 } \big ) \in ( 0 , 1 )$ , where $\lambda _ { K } ( \cdot )$ is the odds-contraction factor from Proposition 3.1.

Choose

$$
\varepsilon _ { i } \in \left( 0 , \operatorname* { m i n } \left\{ \frac { 1 } { 2 } , 1 - \lambda _ { K } ( \frac { \gamma _ { i } } { 2 } ) , \frac { \gamma _ { i } } { 8 } \right\} \right) ,
$$

and define

$$
U _ { i } : = \left\{ ( p , w ) : 1 - p ( x _ { i } ) \leq \varepsilon _ { i } , \| w - \varphi ( x _ { i } ) \| \leq \frac { \gamma _ { i } } { 4 } \right\} .
$$

We show that $U _ { i }$ is forward invariant and that $q _ { t }$ decays geometrically.

Uniform reward gap inside $U _ { i }$ . Suppose $( p _ { t } , w _ { t } ) \in U _ { i }$ . Using $r _ { w _ { t } } ( x ) = \langle w _ { t } , \varphi ( x ) \rangle$ ⟩, we have

$$
r _ { w _ { t } } ( x _ { i } ) - r _ { w _ { t } } ( x _ { j } ) = \langle w _ { t } , \varphi ( x _ { i } ) - \varphi ( x _ { j } ) \rangle
$$

Writing

$$
w _ { t } = \varphi ( x _ { i } ) + \big ( w _ { t } - \varphi ( x _ { i } ) \big )
$$

we obtain

$$
r _ { w _ { t } } ( x _ { i } ) - r _ { w _ { t } } ( x _ { j } ) = \langle \varphi ( x _ { i } ) , \varphi ( x _ { i } ) - \varphi ( x _ { j } ) \rangle + \langle w _ { t } - \varphi ( x _ { i } ) , \varphi ( x _ { i } ) - \varphi ( x _ { j } ) \rangle
$$

for every $j \neq i .$

Hence

$$
r _ { w _ { t } } ( x _ { i } ) - r _ { w _ { t } } ( x _ { j } ) \geq \gamma _ { i } - \| w _ { t } - \varphi ( x _ { i } ) \| \| \varphi ( x _ { i } ) - \varphi ( x _ { j } ) \|
$$

Since $\| \varphi ( x _ { i } ) - \varphi ( x _ { j } ) \| \leq 2$ and $\| w _ { t } - \varphi ( x _ { i } ) \| \leq \gamma _ { i } / 4$ , we obtain

$$
r _ { w _ { t } } ( x _ { i } ) - r _ { w _ { t } } ( x _ { j } ) \geq \gamma _ { i } - 2 \cdot \frac { \gamma _ { i } } { 4 } = \frac { \gamma _ { i } } { 2 }
$$

Odds contraction. By Proposition 3.1, for every $j \neq i$

$$
\frac { H _ { p _ { t } , w _ { t } } ^ { K } ( x _ { j } ) } { H _ { p _ { t } , w _ { t } } ^ { K } ( x _ { i } ) } \leq \lambda _ { K } ( \frac { \gamma _ { i } } 2 )
$$

Since the pure data update is

$$
p _ { t + 1 } ( x ) = p _ { t } ( x ) H _ { p _ { t } , w _ { t } } ^ { K } ( x ) ,
$$

we have

$$
\frac { p _ { t + 1 } ( x _ { j } ) } { p _ { t + 1 } ( x _ { i } ) } = \frac { p _ { t } ( x _ { j } ) } { p _ { t } ( x _ { i } ) } \cdot \frac { H _ { p _ { t } , w _ { t } } ^ { K } ( x _ { j } ) } { H _ { p _ { t } , w _ { t } } ^ { K } ( x _ { i } ) } \leq \lambda _ { K } ( \frac { \gamma _ { i } } { 2 } ) \frac { p _ { t } ( x _ { j } ) } { p _ { t } ( x _ { i } ) }
$$

Summing over all $j \neq i$ gives

$$
\sum _ { j \neq i } { \frac { p _ { t + 1 } ( x _ { j } ) } { p _ { t + 1 } ( x _ { i } ) } } \leq \sum _ { j \neq i } \lambda _ { K } ( { \frac { \gamma _ { i } } { 2 } } ) { \frac { p _ { t } ( x _ { j } ) } { p _ { t } ( x _ { i } ) } }
$$

that is,

$$
\frac { q _ { t + 1 } } { 1 - q _ { t + 1 } } \leq \lambda _ { K } ( \frac { \gamma _ { i } } { 2 } ) \frac { q _ { t } } { 1 - q _ { t } }
$$

Since $q _ { t } \leq \varepsilon _ { i }$ , this implies

$$
q _ { t + 1 } \leq \frac { \lambda _ { K } \bigl ( \frac { \gamma _ { i } } { 2 } \bigr ) } { 1 - q _ { t } } q _ { t } \leq \frac { \lambda _ { K } \bigl ( \frac { \gamma _ { i } } { 2 } \bigr ) } { 1 - \varepsilon _ { i } } q _ { t }
$$

Define

$$
\bar { \lambda } _ { K } : = \frac { \lambda _ { K } ( \frac { \gamma _ { i } } { 2 } ) } { 1 - \varepsilon _ { i } }
$$

Because $\begin{array} { r } { \varepsilon _ { i } \leq 1 - \lambda _ { K } \big ( \frac { \gamma _ { i } } { 2 } \big ) } \end{array}$ , we have $\bar { \lambda } _ { K } \in ( 0 , 1 ] .$ . Thus

$$
q _ { t + 1 } \leq \bar { \lambda } _ { K } q _ { t }
$$

Iterating yields

$$
q _ { t } \le \bar { \lambda } _ { K } ^ { t } q _ { 0 }
$$

that is,

$$
1 - p _ { t } ( x _ { i } ) \leq \bar { \lambda } _ { K } ^ { t } \big ( 1 - p _ { 0 } ( x _ { i } ) \big )
$$

Forward invariance of $U _ { i }$ . From the geometric bound,

$$
q _ { t + 1 } \leq \bar { \lambda } _ { K } q _ { t } \leq \bar { \lambda } _ { K } \varepsilon _ { i } \leq \varepsilon _ { i }
$$

So the first defining condition of $U _ { i }$ is preserved.

Next, since

$$
\bar { \varphi } ( p _ { t + 1 } ) - \varphi ( x _ { i } ) = \sum _ { j \neq i } p _ { t + 1 } ( x _ { j } ) \big ( \varphi ( x _ { j } ) - \varphi ( x _ { i } ) \big ) ,
$$

we obtain

$$
\| \bar { \varphi } ( p _ { t + 1 } ) - \varphi ( x _ { i } ) \| \leq \sum _ { j \neq i } p _ { t + 1 } ( x _ { j } ) \| \varphi ( x _ { j } ) - \varphi ( x _ { i } ) \| \leq 2 q _ { t + 1 }
$$

Using the preference update,

$$
w _ { t + 1 } = \mathrm { P r o j } _ { \mathcal { W } } ( ( 1 - \beta ) w _ { t } + \beta \bar { \varphi } ( p _ { t + 1 } ) )
$$

and

$$
\| w _ { t + 1 } - \varphi ( x _ { i } ) \| \leq ( 1 - \beta ) \| w _ { t } - \varphi ( x _ { i } ) \| + \beta \| { \bar { \varphi } } ( p _ { t + 1 } ) - \varphi ( x _ { i } ) \|
$$

Hence

$$
{ \| { w } _ { t + 1 } } - { \varphi } ( { x } _ { i } ) \| \le ( 1 - \beta ) \frac { \gamma _ { i } } 4 + 2 \beta q _ { t + 1 } \le ( 1 - \beta ) \frac { \gamma _ { i } } 4 + 2 \beta { \varepsilon } _ { i }
$$

Since $\varepsilon _ { i } \leq \gamma _ { i } / 8 ,$ , we get

$$
\| w _ { t + 1 } - \varphi ( x _ { i } ) \| \leq ( 1 - \beta ) \frac { \gamma _ { i } } { 4 } + \beta \frac { \gamma _ { i } } { 4 } = \frac { \gamma _ { i } } { 4 }
$$

So the second defining condition of $U _ { i }$ is also preserved. Therefore $U _ { i }$ is forward invariant.

Convergence of $w _ { t }$ . We already proved $q _ { t } \to 0$ . Using again

$$
\lVert \bar { \varphi } ( p _ { t + 1 } ) - \varphi ( x _ { i } ) \rVert \leq 2 q _ { t + 1 } ,
$$

the preference recursion implies

$$
\| w _ { t + 1 } - \varphi ( x _ { i } ) \| \leq ( 1 - \beta ) \| w _ { t } - \varphi ( x _ { i } ) \| + 2 \beta q _ { t + 1 }
$$

Unrolling this recursion gives

$$
\| w _ { t } - \varphi ( x _ { i } ) \| \leq ( 1 - \beta ) ^ { t } \| w _ { 0 } - \varphi ( x _ { i } ) \| + 2 \beta \sum _ { s = 0 } ^ { t - 1 } ( 1 - \beta ) ^ { t - 1 - s } q _ { s + 1 }
$$

Since $q _ { s }  0$ geometrically, the right-hand side tends to 0. Hence

$$
w _ { t } \to \varphi ( x _ { i } )
$$

Together with $q _ { t } \to 0 ,$ , this implies

$$
p _ { t } ( x _ { i } ) \xrightarrow { t \to \infty } 1 , \qquad w _ { t } \xrightarrow { t \to \infty } \varphi ( x _ { i } ) .
$$

## C.5 Proof of Corollary 3.5

Corollary 3.5 (Failure of local stability). Fix $x _ { i } \in { \mathcal { X } } .$ . Ifthere exists $x _ { j }$ with $j \neq i$ such that $\varphi ( x _ { j } ) = \varphi ( x _ { i } )$ then the singleton equilibrium $\delta _ { x _ { i } }$ is not locally asymptotically stable.

Proof. Assume there exists $j \neq i$ such that

$$
\varphi ( x _ { j } ) = \varphi ( x _ { i } )
$$

For any $\varepsilon \in ( 0 , 1 )$ , define

$$
p ^ { \varepsilon } ( x _ { \ell } ) = { \left\{ \begin{array} { l l } { 1 - \varepsilon , } & { \ell = i , } \\ { \varepsilon , } & { \ell = j , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. } \quad w ^ { \varepsilon } : = \varphi ( x _ { i } )
$$

Since $\varphi ( x _ { j } ) = \varphi ( x _ { i } )$ , we have

$$
r _ { w ^ { \varepsilon } } ( x _ { j } ) = r _ { w ^ { \varepsilon } } ( x _ { i } ) ,
$$

so all points in the support of $\cdot _ { p ^ { \varepsilon } }$ have the same reward.

By Theorem $3 . 2 , p ^ { \varepsilon }$ is a fixed point of the data update.

Moreover,

$$
\bar { \varphi } ( p ^ { \varepsilon } ) = ( 1 - \varepsilon ) \varphi ( x _ { i } ) + \varepsilon \varphi ( x _ { j } ) = \varphi ( x _ { i } ) = w ^ { \varepsilon } ,
$$

so the preference update also fixes w<sup>ε</sup>.

Hence $( p ^ { \varepsilon } , w ^ { \varepsilon } )$ is an equilibrium for every $\varepsilon \in ( 0 , 1 )$

As $\varepsilon  0 ,$ , the equilibria $( p ^ { \varepsilon } , w ^ { \varepsilon } )$ converge to the singleton equilibrium at $x _ { i }$

Therefore this singleton equilibrium is not isolated, and hence cannot be locally asymptotically stable. □

## C.6 Proof of Proposition 3.6

Lemma 3.6 (Sensitivity of the retraining step). Let $\begin{array} { r } { d _ { \mathrm { T V } } ( p , q ) : = \frac { 1 } { 2 } \sum _ { x \in \mathcal { X } } | p ( x ) - q ( x ) | } \end{array}$ denote total variation distance. For any $p , q \in \mathcal { P } ( \mathcal { X } ) , w , v \in \mathcal { W } ,$ , and $K \geq 2 ,$ , thefollowing hold:

$$
\begin{array} { r } { d _ { \mathrm { T V } } \left( p H _ { p , w } ^ { K } , q H _ { q , w } ^ { K } \right) \leq L _ { p } d _ { \mathrm { T V } } ( p , q ) ; d _ { \mathrm { T V } } \left( p H _ { p , w } ^ { K } , p H _ { p , v } ^ { K } \right) \leq L _ { w } \| w - v \| ; \| \bar { \varphi } ( p ) - \bar { \varphi } ( q ) \| \leq 2 d _ { \mathrm { T V } } ( p , q ) } \end{array}
$$

where $L _ { p } : = 2 e ^ { 2 \tau } ( 2 + e ^ { 2 \tau } )$ and $L _ { w } : = 2 \tau e ^ { 4 \tau }$ .

Proof. Recall that

$$
\begin{array} { r } { a ( x ) : = e ^ { \tau r _ { w } ( x ) } } \end{array}
$$

Since $r _ { w } ( x ) \in [ - 1 , 1 ] ,$ , we have

$$
e ^ { - \tau } \leq a ( x ) \leq e ^ { \tau }
$$

For finite K,

$$
H _ { p , w } ^ { K } ( x ) = \mathbb { E } _ { { X _ { 1 } , \ldots , X _ { K - 1 } } \sim p } { \left[ \frac { K a ( x ) } { a ( x ) + \sum _ { \ell = 1 } ^ { K - 1 } a ( X _ { \ell } ) } \right] }
$$

For $K = \infty ,$

$$
H _ { p , w } ^ { \infty } ( x ) = \frac { a _ { ( } x ) } { \mathbb { E } _ { z \sim p } [ a ( z ) ] }
$$

We prove the three bounds in turn.

Part 1: Lipschitz continuity in $p .$ . Fix $w \in \mathcal W$

$$
2 d _ { \mathrm { T V } } ( p H _ { p , w } ^ { K } , \ q H _ { q , w } ^ { K } ) = \sum _ { x \in \mathcal { X } } \left| p ( x ) H _ { p , w } ^ { K } ( x ) - q ( x ) H _ { q , w } ^ { K } ( x ) \right|
$$

For each $x \in { \mathcal { X } } .$

$$
p ( x ) H _ { p , w } ^ { K } ( x ) - q ( x ) H _ { q , w } ^ { K } ( x ) = ( p ( x ) - q ( x ) ) H _ { p , w } ^ { K } ( x ) + q ( x ) \big ( H _ { p , w } ^ { K } ( x ) - H _ { q , w } ^ { K } ( x ) \big )
$$

Hence

$$
2 d _ { \mathrm { T V } } ( p H _ { p , w } ^ { K } , q H _ { q , w } ^ { K } ) \leq I + I I ,
$$

where

$$
I : = \sum _ { x \in \mathcal { X } } | p ( x ) - q ( x ) | H _ { p , w } ^ { K } ( x ) , \qquad I I : = \sum _ { x \in \mathcal { X } } q ( x ) \big | H _ { p , w } ^ { K } ( x ) - H _ { q , w } ^ { K } ( x ) \big | .
$$

We first bound I. Since for all $x , p , w , K$

$$
0 \leq H _ { p , w } ^ { K } ( x ) \leq 2 e ^ { 2 \tau }
$$

we have

$$
I \leq 2 e ^ { 2 \tau } \sum _ { x \in \mathcal { X } } | p ( x ) - q ( x ) | = 4 e ^ { 2 \tau } d _ { \mathrm { T V } } ( p , q )
$$

Next we bound II. Fix $x \in \mathcal { X }$ . For finite K, define

$$
g _ { x } ( y _ { 1 } , \dots , y _ { K - 1 } ) : = \frac { K a _ { w } ( x ) } { a _ { w } ( x ) + \sum _ { \ell = 1 } ^ { K - 1 } a _ { w } ( y _ { \ell } ) }
$$

Then

$$
H _ { p , w } ^ { K } ( \boldsymbol { x } ) = \mathbb { E } _ { p ^ { K - 1 } } [ g _ { x } ] , \qquad H _ { q , w } ^ { K } ( \boldsymbol { x } ) = \mathbb { E } _ { q ^ { K - 1 } } [ g _ { x } ]
$$

We compare these two expectations by replacing the coordinate distributions one at a time. Define, for $\ell = 0 , 1 , \hat { \mathbf { \Omega } } , \dots , K - 1$

$$
M _ { \ell } : = \mathbb { E } { \bigl [ } g _ { x } ( Y _ { 1 } , \ldots , Y _ { K - 1 } ) { \bigr ] } , \qquad Y _ { j } \sim \left\{ { \begin{array} { l l } { q } & { { \mathrm { i f ~ } } j \leq \ell , } \\ { p } & { { \mathrm { i f ~ } } j > \ell . } \end{array} } \right.
$$

Then $M _ { 0 } = H _ { p , w } ^ { K } ( x ) , M _ { K - 1 } = H _ { q , w } ^ { K } ( x )$ , and

$$
H _ { p , w } ^ { K } ( x ) - H _ { q , w } ^ { K } ( x ) = \sum _ { \ell = 1 } ^ { K - 1 } ( M _ { \ell - 1 } - M _ { \ell } )
$$

The difference $M _ { \ell - 1 } - M _ { \ell }$ isolates the effect of switching coordinate ℓ from p to $q ,$ with all other coordinates unchanged.

In $M _ { \ell - 1 } - M _ { \ell } ,$ , condition on all coordinates except $y \ell$ . Write $\begin{array} { r } { S _ { - \ell } : = \sum _ { m \neq \ell } a ( y _ { m } ) } \end{array}$ for their total As a function of $y \ell$ alone, the integrand becomes

$$
h ( y \varepsilon ) : = \frac { K a ( x ) } { a ( x ) + a ( y \varepsilon ) + S _ { - \ell } } .
$$

Since h is decreasing in $a ( y _ { \ell } )$ and $a ( y _ { \ell } ) \in [ e ^ { - \tau } , e ^ { \tau } ] ;$

$$
\operatorname* { s u p } _ { y } h ( y ) - \operatorname* { i n f } _ { y } h ( y ) = K a ( x ) \cdot { \frac { e ^ { \tau } - e ^ { - \tau } } { ( a ( x ) + e ^ { - \tau } + S _ { - \ell } ) ( a ( x ) + e ^ { \tau } + S _ { - \ell } ) } }
$$

We bound the numerator and denominator separately:

Numerator: Using $\boldsymbol { a } ( \boldsymbol { x } ) \le e ^ { \tau }$ and $e ^ { \tau } - e ^ { - \tau } \leq 2 e ^ { \tau }$

$$
K a ( x ) ( e ^ { \tau } - e ^ { - \tau } ) \le K e ^ { \tau } \cdot 2 e ^ { \tau } = 2 K e ^ { 2 \tau }
$$

Denominator: each factor satisfies

$$
a ( x ) + e ^ { \pm \tau } + S _ { - \ell } \geq e ^ { - \tau } + e ^ { - \tau } + ( K - 2 ) e ^ { - \tau } = K e ^ { - \tau }
$$

so the product is $\geq K ^ { 2 } e ^ { - 2 \tau }$

Therefore

$$
\operatorname* { s u p } _ { y } h ( y ) - \operatorname* { i n f } _ { y } h ( y ) \leq \frac { 2 K e ^ { 2 \tau } } { K ^ { 2 } e ^ { - 2 \tau } } = \frac { 2 e ^ { 4 \tau } } { K } .
$$

This bound holds for every realization of the conditioned coordinates, since it uses only the worst-case $S _ { - \ell } \geq$ $\left( K - 2 \right) e ^ { - \tau }$

Now, for any bounded function h and any two distributions $p , q ,$

$$
\left| H _ { p , w } ^ { K } ( x ) - H _ { q , w } ^ { K } ( x ) \right| \le ( \operatorname* { s u p } _ { y } h ( y ) - \operatorname* { i n f } _ { y } h ( y ) ) \cdot d _ { \mathrm { T V } } ( p , q ) .
$$

(This follows from the total-variation duality: center h so that $\begin{array} { r } { \| h - \operatorname { c o n s t } \| _ { \infty } \leq ( \operatorname* { s u p } _ { y } h ( y ) - \operatorname* { i n f } _ { y } h ( y ) ) / 2 , } \end{array}$ then apply $| \mathbb { E } _ { p } [ f ] - \mathbb { E } _ { q } [ f ] | \leq 2 \| f \| _ { \infty } d _ { \mathrm { T V } } ( p , q ) . )$

So:

$$
\big | H _ { p , w } ^ { K } ( x ) - H _ { q , w } ^ { K } ( x ) \big | \leq \sum _ { \ell = 1 } ^ { K - 1 } \frac { 2 e ^ { 4 \tau } } { K } d _ { \mathrm { T V } } ( p , q ) = \frac { ( K - 1 ) \cdot 2 e ^ { 4 \tau } } { K } d _ { \mathrm { T V } } ( p , q ) < 2 e ^ { 4 \tau } d _ { \mathrm { T V } } ( p , q )
$$

This is a pointwise bound on x, uniform in $K$

Hence

$$
I I \leq 2 e ^ { 4 \tau } d _ { \mathrm { T V } } ( p , q ) .
$$

Combining the bounds on I and II, we get

$$
2 d _ { \mathrm { T V } } ( p H _ { p , w } ^ { K } , q H _ { q , w } ^ { K } ) \leq \bigl ( 4 e ^ { 2 \tau } + 2 e ^ { 4 \tau } \bigr ) d _ { \mathrm { T V } } ( p , q ) .
$$

Equivalently,

$$
d _ { \mathrm { T V } } ( p H _ { p , w } ^ { K } , q H _ { q , w } ^ { K } ) \leq L _ { p } d _ { \mathrm { T V } } ( p , q ) ,
$$

where

$$
L _ { p } : = 2 e ^ { 2 \tau } ( 2 + e ^ { 2 \tau } )
$$

For $K = \infty$ , the same inequality remains valid. Indeed,

$$
H _ { p , w } ^ { \infty } ( x ) = \frac { a _ { ( } x ) } { \mathbb { E } _ { z \sim p } [ a ( z ) ] }
$$

and the dependence on p is simpler.

Part 2: Lipschitz continuity in w. Fix $p \in { \mathcal { P } } ( { \mathcal { X } } )$ . For finite $K .$ , use the same samples $X _ { 1 } , \ldots , X _ { K - 1 }$ <sup>iid</sup>∼ $p ,$ and write

$$
S _ { w } : = \sum _ { \ell = 1 } ^ { K - 1 } a _ { w } ( X _ { \ell } ) , \qquad S _ { v } : = \sum _ { \ell = 1 } ^ { K - 1 } a _ { v } ( X _ { \ell } ) .
$$

Then for any $x \in \mathcal { X }$

$$
\begin{array} { r } { \left| \cfrac { K a _ { w } ( x ) } { a _ { w } ( x ) + S _ { w } } - \cfrac { K a _ { v } ( x ) } { a _ { v } ( x ) + S _ { v } } \right| \leq \left| \cfrac { K a _ { w } ( x ) } { a _ { w } ( x ) + S _ { w } } - \cfrac { K a _ { v } ( x ) } { a _ { v } ( x ) + S _ { w } } \right| } \\ { + \left| \cfrac { K a _ { v } ( x ) } { a _ { v } ( x ) + S _ { w } } - \cfrac { K a _ { v } ( x ) } { a _ { v } ( x ) + S _ { v } } \right| } \end{array}
$$

For the first term, since

$$
| a _ { w } ( x ) - a _ { v } ( x ) | \leq \tau e ^ { \tau } \| w - v \|
$$

we have

$$
\left| \frac { K a _ { w } ( x ) } { a _ { w } ( x ) + S _ { w } } - \frac { K a _ { v } ( x ) } { a _ { v } ( x ) + S _ { w } } \right| = \frac { K S _ { w } | a _ { w } ( x ) - a _ { v } ( x ) | } { ( a _ { w } ( x ) + S _ { w } ) ( a _ { v } ( x ) + S _ { w } ) } .
$$

Using

$$
S w \le ( K - 1 ) e ^ { \tau } , \qquad a _ { w } ( x ) + S _ { w } \ge K e ^ { - \tau } , \qquad a _ { v } ( x ) + S _ { w } \ge K e ^ { - \tau } ,
$$

we obtain

$$
\left| \frac { K a _ { w } ( x ) } { a _ { w } ( x ) + S _ { w } } - \frac { K a _ { v } ( x ) } { a _ { v } ( x ) + S _ { w } } \right| \leq \tau e ^ { 4 \tau } \| w - v \| .
$$

For the second term,

$$
| S _ { v } - S _ { w } | \leq \sum _ { \ell = 1 } ^ { K - 1 } | a _ { v } ( X _ { \ell } ) - a _ { w } ( X _ { \ell } ) | \leq ( K - 1 ) \tau e ^ { \tau } \| w - v \| .
$$

Therefore

$$
\left| \frac { K a _ { v } ( x ) } { a _ { v } ( x ) + S _ { w } } - \frac { K a _ { v } ( x ) } { a _ { v } ( x ) + S _ { v } } \right| = \frac { K a _ { v } ( x ) | S _ { v } - S _ { w } | } { ( a _ { v } ( x ) + S _ { w } ) ( a _ { v } ( x ) + S _ { v } ) } \leq \tau e ^ { 4 \tau } \| w - v \| .
$$

Combining the two bounds and taking expectation gives

$$
| H _ { p , w } ^ { K } ( x ) - H _ { p _ { v } } ^ { K } ( x ) | \leq 2 \tau e ^ { 4 \tau } \| w - v \| .
$$

Hence

$$
d _ { \mathrm { T V } } ( p H _ { p , w } ^ { K } , p H _ { p _ { v } } ^ { K } ) \leq L _ { w } \Vert w - v \Vert ,
$$

where

$$
L _ { w } : = 2 \tau e ^ { 4 \tau } .
$$

For $K = \infty ,$ , the same bound follows directly from

$$
H _ { p , w } ^ { \infty } ( x ) = \frac { a _ { ( } x ) } { \mathbb { E } _ { z \sim p } [ a ( z ) ] }
$$

Part 3: Lipschitz continuity of ${ \bar { \varphi } } .$ . Let $u \in \mathbb { R } ^ { d }$ satisfy $\lVert u \rVert = 1$ . Since $\| \varphi ( x ) \| = 1$

$$
| \langle u , \varphi ( x ) \rangle | \leq 1
$$

Therefore

$$
| \langle u , \bar { \varphi } ( p ) - \bar { \varphi } ( q ) \rangle | = \left| \sum _ { x \in \mathcal { X } } \langle u , \varphi ( x ) \rangle \big ( p ( x ) - q ( x ) \big ) \right| \leq 2 d _ { \mathrm { T V } } ( p , q ) .
$$

Taking the supremum over all unit vectors u yields

$$
\Vert \bar { \varphi } ( p ) - \bar { \varphi } ( q ) \Vert \leq 2 d _ { \mathrm { T V } } ( p , q ) .
$$

This completes the proof.

## C.7 Proofs of Lemma 3.7 and Proposition 3.8

Lemma 3.7 (Unique equilibrium under frozen-w). Fix w $\in { \mathcal { W } } ,$ and consider the the frozen-w model update $p _ { t + 1 } \dot { ( x ) } \stackrel { \cdot } { = } ( 1 \stackrel { \cdot } { - } \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p ( x ) H _ { p , w } ^ { K } \dot { ( x ) }$ $I f \eta L _ { p } < 1$ , the update is a strict contraction on $( \mathcal { P } ( \mathcal { X } ) , d _ { \mathrm { T V } } )$ . Consequently, there exists a unique equilibrium $p _ { w } ^ { K , \eta }$ and every trajectory converges to it geometrically:

$$
d _ { \mathrm { T V } } ( p _ { t } , p _ { w } ^ { K , \eta } ) \leq ( \eta L _ { p } ) ^ { t } d _ { \mathrm { T V } } ( p _ { 0 } , p _ { w } ^ { K , \eta } ) .
$$

Proof. Fix $w \in \mathcal { W } .$ . For any $p , q \in { \mathcal { P } } ( { \mathcal { X } } )$ , define

$$
T _ { w } ^ { K } ( p ) ( x ) : = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p ( x ) H _ { p , w } ^ { K } ( x )
$$

We show that $T _ { w } ^ { K }$ is a contraction on $( \mathcal { P } ( \mathcal { X } ) , d _ { \mathrm { T V } } )$ .

Let $p , q \in { \mathcal { P } } ( { \mathcal { X } } )$ . Then

$$
\begin{array} { r l } & { d _ { \mathrm { T V } } ( T _ { w } ^ { K } ( p ) , T _ { w } ^ { K } ( q ) ) = d _ { \mathrm { T V } } \left( ( 1 - \eta ) p _ { \mathrm { r e f } } + \eta p H _ { p , w } ^ { K } , ( 1 - \eta ) p _ { \mathrm { r e f } } + \eta q H _ { q , w } ^ { K } \right) } \\ & { \quad \quad \quad = \eta d _ { \mathrm { T V } } ( p H _ { p , w } ^ { K } , q H _ { q , w } ^ { K } ) . } \end{array}
$$

By Proposition 3.6,

$$
d _ { \mathrm { T V } } ( p H _ { p , w } ^ { K } , q H _ { q , w } ^ { K } ) \leq L _ { p } d _ { \mathrm { T V } } ( p , q ) .
$$

Therefore,

$$
d _ { \mathrm { T V } } ( T _ { w } ^ { K } ( p ) , T _ { w } ^ { K } ( q ) ) \leq \eta L _ { p } d _ { \mathrm { T V } } ( p , q ) .
$$

If $\eta L _ { p } < 1$ , then $T _ { w } ^ { K }$ is a strict contraction.

Since $\mathcal { P } ( \mathcal { X } )$ is complete under $d _ { \mathrm { T V } }$ in the finite-state setting, the Banach fixed-point theorem implies that $T _ { w } ^ { K }$ admits a unique fixed point, denoted $p _ { w } ^ { K , \eta }$ , satisfying

$$
p _ { w } ^ { K , \eta } ( x ) = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p _ { w } ^ { K , \eta } ( x ) H _ { ( p _ { w } ^ { K , \eta } ) _ { w } } ^ { K } ( x )
$$

Moreover, for any initialization $p _ { 0 } \in \mathcal { P } ( \mathcal { X }$ , the iterates satisfy

$$
d _ { \mathrm { T V } } ( p _ { t } , p _ { w } ^ { K , \eta } ) \leq ( \eta L _ { p } ) ^ { t } d _ { \mathrm { T V } } ( p _ { 0 } , p _ { w } ^ { K , \eta } ) ,
$$

which proves geometric convergence.

Proposition 3.8 (Closed-form equilibrium). As $K  \infty ,$ , the frozen-w equilibrium takes the form

$$
p _ { w } ^ { \infty , \eta } ( x ) = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) \frac { Z _ { w } ^ { \eta } } { Z _ { w } ^ { \eta } - \eta e ^ { \tau \cdot r _ { w } ( x ) } } .
$$

where $Z _ { w } ^ { \eta } : = \mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } } [ e ^ { \tau \cdot r _ { w } ( x ) } ]$ is the unique scalar satisfying

$$
\sum _ { i = 1 } ^ { n } ( 1 - \eta ) p _ { \mathrm { r e f } } ( x _ { i } ) \frac { e ^ { \tau \cdot r _ { w } ( x _ { i } ) } } { Z _ { w } ^ { \eta } - \eta e ^ { \tau \cdot r _ { w } ( x _ { i } ) } } = 1 , ~ Z _ { w } ^ { \eta } > \eta \operatorname* { m a x } _ { i } e ^ { \tau \cdot r _ { w } ( x _ { i } ) } .
$$

Proof. Recall that

$$
a _ { w } ( x ) : = e ^ { \tau r _ { w } ( x ) }
$$

$$
\begin{array} { r } { H _ { p , w } ^ { \infty } ( x ) = \frac { a _ { w } ( x ) } { \mathbb { E } _ { z \sim p _ { w } } \left[ a _ { w } ( z ) \right] } } \end{array}
$$

For fixed w, the frozen-w update therefore becomes

$$
p _ { t + 1 } ( x _ { i } ) = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x _ { i } ) + \eta p _ { t } ( x _ { i } ) \frac { a _ { w } ( x _ { i } ) } { \mathbb { E } _ { z \sim p _ { t } } \left[ a _ { w } ( z ) \right] } ,
$$

Let $p _ { w } ^ { \infty , \eta }$ be a frozen-w fixed point, then the fixed-point equation is

$$
p _ { w } ^ { \infty , \eta } = ( 1 - \eta ) p _ { \mathrm { r e f } } + \eta p _ { w } ^ { \infty , \eta } \frac { a _ { w } ( x ) } { \mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } } \left[ a _ { w } ( z ) \right] }
$$

Rearranging,

$$
p _ { w } ^ { \infty , \eta } \left( 1 - \eta \frac { a _ { w } ( x ) } { \mathbb { E } _ { p _ { w } ^ { \infty , \eta } } \left[ a _ { w } ( z ) \right] } \right) = ( 1 - \eta ) p _ { \mathrm { r e f } } ,
$$

so

$$
p _ { w } ^ { \infty , \eta } = \frac { ( 1 - \eta ) p _ { \mathrm { r e f } } } { 1 - \eta \frac { a _ { w } ( x ) } { \mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } } \left[ a _ { w } ( z ) \right] } } = ( 1 - \eta ) p _ { \mathrm { r e f } } \frac { \mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } p } [ a _ { w } ( z ) ] } { \mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } } \left[ a _ { w } ( z ) \right] - \eta a _ { w } ( x ) } .
$$

That is

$$
p _ { w } ^ { \infty , \eta } = ( 1 - \eta ) p _ { \mathrm { r e f } } \frac { \mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } } [ e ^ { \tau r _ { w } ( x ) } ] } { \mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } } [ e ^ { \tau r _ { w } ( x ) } ) ] - \eta e ^ { \tau r _ { w } ( x ) } } .
$$

It remains to determine $\mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } } [ a _ { w } ( z ) ]$ . Substituting the formula above, we obtain

$$
\mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } } [ a _ { w } ( z ) ] = \sum _ { i = 1 } ^ { n } ( 1 - \eta ) p _ { \mathrm { r e f } } \frac { \mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } p } [ a _ { w } ( z ) ] } { \mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } } [ a _ { w } ( z ) ] - \eta a _ { w } ( x _ { i } ) } a _ { w } ( x _ { i } )
$$

Since $\mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } } [ a _ { w } ( z ) ]$ ], dividing both sides by $\mathbb { E } _ { z \sim p _ { w } ^ { \infty , \eta } } [ a _ { w } ( z ) ]$ yields

$$
\sum _ { i = 1 } ^ { n } ( 1 - \eta ) p _ { \mathrm { r e f } } \frac { a _ { w } ( x _ { i } ) } { \mathbb { E } _ { z \sim p _ { w } ^ { \infty } } , \eta \left[ a _ { w } ( z ) \right] - \eta a _ { w } ( x _ { i } ) } = 1
$$

We now show that this equation has a unique solution on

$$
( \eta \operatorname* { m a x } _ { i } a _ { w } ( x _ { i } ) , \infty ) .
$$

Define

$$
F ( Z _ { w } ^ { \eta } ) : = \sum _ { i = 1 } ^ { n } ( 1 - \eta ) p _ { \mathrm { r e f } } ( x _ { i } ) \frac { a _ { w } ( x _ { i } ) } { Z _ { w } ^ { \eta } - \eta a _ { w } ( x _ { i } ) } .
$$

For $Z _ { w } ^ { \eta } > \eta$ max<sub>i</sub> $a _ { w } ( x _ { i } )$ , all denominators are positive, and

$$
F ^ { \prime } ( Z _ { w } ^ { \eta } ) = - \sum _ { i = 1 } ^ { n } ( 1 - \eta ) p _ { \mathrm { r e f } } ( x _ { i } ) \frac { a _ { w } ( x _ { i } ) } { ( Z _ { w } ^ { \eta } - \eta a _ { w } ( x _ { i } ) ) ^ { 2 } } < 0 .
$$

Thus F is strictly decreasing on $( \eta \operatorname* { m a x } _ { i } a _ { w } ( x _ { i } ) , \infty )$ . Moreover, $F ( Z _ { w } ^ { \eta } ) \to \infty$ as $Z _ { w } ^ { \eta } \downarrow$ η max<sub>i</sub> $a _ { w } ( x _ { i } )$ , while

$$
F ( Z _ { w } ^ { \eta } ) \to 0 \qquad \mathrm { a s } Z _ { w } ^ { \eta } \to \infty .
$$

Therefore the equation $F ( Z _ { w } ^ { \eta } ) = 1$ has a unique solution when $Z _ { w } ^ { \eta } > \eta \operatorname* { m a x } _ { i } a _ { w } ( x _ { i } )$

Substituting this unique $Z _ { w } ^ { \eta }$ into the formula yields the unique frozen-w fixed point.

## C.8 Proof of Theorem 3.9

Theorem 3.9 (Convergence of coupled dynamics). Consider the coupled dynamics $( 3 ) \AA - \displaystyle ( 4 )$ . If $\eta ( L _ { p } +$ $2 L _ { w } ) < 1$ , then there exists $\begin{array} { r } { \xi \in \left( \frac { \eta L _ { w } } { \beta ( 1 - 2 \eta L _ { w } ) } , \frac { 1 - \eta L _ { p } } { 2 \beta \eta L _ { p } } \right) } \end{array}$ such that the coupled update is a strict contraction on $( \mathcal { P } ( \mathcal { X } ) \times \mathcal { W } , d _ { \xi } )$ , where $d _ { \xi } \big ( ( p , w ) , ( q , v ) \big ) : = d _ { \mathrm { T V } } ( p , q ) + \xi \lVert w - v \rVert$ . Consequently, there exists a unique equilibrium $( p _ { K } ^ { \star } , w _ { K } ^ { \star } )$ satisfying (5), and every trajectory converges to it geometrically.

Proof. Let $( p , w )$ and $( q , v )$ be two states, then:

$$
p _ { t + 1 } ( x ) = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p _ { t } ( x ) H _ { p _ { t } , w } ^ { K } ( x ) , \qquad q _ { t + 1 } = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta q _ { t } ( x ) H _ { q _ { t } , v } ^ { K } ( x ) ,
$$

$$
w _ { t + 1 } = \mathrm { P r o j } _ { \mathcal { W } } ( ( 1 - \beta ) w _ { t } + \beta \bar { \varphi } ( p _ { t + 1 } ) ) , \qquad v _ { t + 1 } = \mathrm { P r o j } _ { \mathcal { W } } ( ( 1 - \beta ) v _ { t } + \beta \bar { \varphi } ( q _ { t + 1 } ) )
$$

We first bound the data coordinate. By Proposition 3.6,

$$
\begin{array} { r l } & { d _ { \mathrm { T V } } ( p _ { t + 1 } , p _ { t + 1 } ) = d _ { \mathrm { T V } } \Big ( ( 1 - \eta ) p _ { \mathrm { r e f } } + \eta p _ { t } H _ { p _ { t } , w } ^ { K } , ( 1 - \eta ) p _ { \mathrm { r e f } } + \eta q _ { t } H _ { q _ { t } , v } ^ { K } , \Big ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad = \eta d _ { \mathrm { T V } } ( p _ { t } H _ { p _ { t } , w } ^ { K } , \ q _ { t } H _ { q _ { t } , v } ^ { K } ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \ \end{array}
$$

Next we bound the preference coordinate. Using the nonexpansiveness of Euclidean projection,

$$
\begin{array} { r } { \| w _ { t + 1 } - v _ { t + 1 } \| \leq \| ( 1 - \beta ) ( w _ { t } - v _ { t } ) + \beta \big ( \bar { \varphi } ( p _ { t + 1 } ) - \bar { \varphi } ( q _ { t + 1 } ) \big | \big | } \\ { \leq ( 1 - \beta ) \| w _ { t } - v _ { t } \| + \beta \| \bar { \varphi } ( p _ { t + 1 } ) - \bar { \varphi } ( q _ { t + 1 } ) \| . } \end{array}
$$

Applying Proposition 3.6 again,

$$
\lVert \bar { \varphi } ( p _ { t + 1 } ) - \bar { \varphi } ( q _ { t + 1 } ) \rVert \leq 2 d _ { \mathrm { T V } } ( p _ { t + 1 } , q _ { t + 1 } ) ,
$$

and hence

$$
\begin{array} { r } { \| w _ { t + 1 } - v _ { t + 1 } \| \leq ( 1 - \beta ) \| w _ { t } - v _ { t } \| + 2 \beta d _ { \mathrm { T V } } \big ( p _ { t + 1 } , q _ { t + 1 } \big ) . } \end{array}
$$

Substituting the bound on $d _ { \mathrm { T V } } ( p _ { t + 1 } , q _ { t + 1 } )$ , we obtain

$$
\begin{array} { r } { \left\| w _ { t + 1 } - v _ { t + 1 } \right\| \leq 2 \beta \eta L _ { p } d _ { \mathrm { T V } } ( p _ { t } , q _ { t } ) + \left( 1 - \beta + 2 \beta \eta L _ { w } \right) \left\| w _ { t } - v _ { t } \right\| . } \end{array}
$$

Now multiply this inequality by $\xi$ and add it to the data-coordinate bound. This gives

$$
\begin{array} { r l } { d _ { \xi } \big ( ( p _ { t + 1 } , w _ { t + 1 } ) , ( q _ { t + 1 } , v _ { t + 1 } ) \big ) = d _ { \mathrm { T V } } ( p _ { t + 1 } , q _ { t + 1 } ) + \xi \| w _ { t + 1 } - v _ { t + 1 } \| } & { } \\ & { \qquad \le \eta L _ { p } d _ { \mathrm { T V } } ( p _ { t } , q _ { t } ) + \eta L _ { w } \| w _ { t } - v _ { t } \| + \xi \Big ( ( 1 - \beta ) \| w _ { t } - v _ { t } \| + 2 \beta d _ { \mathrm { T V } } ( p _ { t + 1 } , q _ { t + 1 } ) \Big ) } \\ & { \qquad \le \Big ( \eta L _ { p } + 2 \beta \eta L _ { p } \xi \Big ) d _ { \mathrm { T V } } ( p , q ) + \xi \Big ( \frac { \eta L _ { w } } { \xi } + ( 1 - \beta + 2 \beta \eta L _ { w } ) \Big ) \| w - v \| . } \end{array}
$$

Therefore the map is a contraction under $d _ { \xi }$ as soon as both coefficients are strictly smaller than the corresponding weights in $d _ { \xi } \mathbf { : }$

$$
\left\{ \begin{array} { r } { \eta L _ { p } + 2 \beta \eta L _ { p } \xi < 1 , \ } \\ { \displaystyle \frac { \eta L _ { w } } { \xi } + ( 1 - \beta + 2 \beta \eta L _ { w } ) < 1 . } \end{array} \right.
$$

so,

$$
\left\{ \begin{array} { r } { \displaystyle \xi < \frac { 1 - \eta L _ { p } } { 2 \beta \eta L _ { p } } } \\ { \displaystyle \xi > \frac { \eta L _ { w } } { \beta ( 1 - 2 \eta L _ { w } ) } } \end{array} \right.
$$

Thus any $\xi$ in the window

$$
\frac { \eta L _ { w } } { \beta ( 1 - 2 \eta L _ { w } ) } < \xi < \frac { 1 - \eta L _ { p } } { 2 \beta \eta L _ { p } }
$$

makes the coupled update a strict contraction.

It remains to check that this window is nonempty. Since all quantities are positive,

$$
\begin{array} { r c l } { \frac { \eta L _ { w } } { \beta ( 1 - 2 \eta L _ { w } ) } < \frac { 1 - \eta L _ { p } } { 2 \beta \eta L _ { p } } } & { \Longleftrightarrow } & { 2 \eta ^ { 2 } L _ { p } L _ { w } < ( 1 - 2 \eta L _ { w } ) ( 1 - \eta L _ { p } ) } \\ & & { \Longleftrightarrow } & { 0 < 1 - \eta L _ { p } - 2 \eta L _ { w } } \\ & & { \Longleftrightarrow } & { \eta ( L _ { p } + 2 L _ { w } ) < 1 . } \end{array}
$$

So under the stated condition the interval is nonempty, and we may choose such a $\xi .$

Finally, $\mathcal { P } ( \mathcal { X } ) \times \mathcal { W }$ is complete under $d _ { \xi }$ in the finite-state setting, since $\mathcal { P } ( \mathcal { X } )$ is complete under $d _ { \mathrm { T V } }$ and W is closed in finite-dimensional Euclidean space.

By the Banach fixed-point theorem, the coupled reinjected dynamics admit a unique equilibrium $( p _ { K } ^ { \star } , w _ { K } ^ { \star } )$ , and every trajectory converges to it geometrically. □

## C.9 Proof of Proposition 4.1

Proposition 4.1 (Sufficient condition for preservation). Fix $\begin{array} { r } { \eta \in \big ( 0 , \frac { 1 } { ( L _ { p } + 2 L _ { w } ) } \big ) . I f \forall \ell \in \{ 1 , \dots , L \} } \end{array}$

$$
\sum _ { i = 1 } ^ { n } p _ { \mathrm { r e f } , i } \left. v _ { \ell } , \varphi ( x _ { i } ) \right. \geq { \frac { \theta _ { \ell } - \eta \cdot \operatorname* { m i n } _ { i = 1 , \cdots , n } \left. v _ { \ell } , \varphi ( x _ { i } ) \right. } { 1 - \eta } } ,
$$

then the equilibrium satisfies the preservation constraint $\left. v _ { \ell } , \bar { \varphi } ( p _ { K } ^ { \star } ( p _ { \mathrm { r e f } } , \eta ) ) \right. \geq \theta _ { \ell } , \forall \ell .$

Proof. Fix $\eta \in ( 0 , \bar { \eta } )$ , and let

$$
( p _ { K } ^ { \star } , w _ { K } ^ { \star } ) : = ( p _ { K } ^ { \star } ( \mu , \eta ) , w _ { K } ^ { \star } ( \mu , \eta ) )
$$

denote the unique equilibrium from Theorem 3.9.

For notational simplicity, write

$$
p ^ { \star } : = p _ { K } ^ { \star } ( \mu , \eta ) .
$$

At equilibrium, the data-update equation gives

$$
p ^ { \star } ( x ) = ( 1 - \eta ) \mu ( x ) + \eta p ^ { \star } ( x ) H _ { ( p ^ { \star } ) _ { w _ { K } ^ { \star } } } ^ { K } ( x ) .
$$

Define the curated equilibrium distribution

$$
q ^ { \star } ( x ) : = p ^ { \star } ( x ) H _ { ( p ^ { \star } ) _ { w _ { K } ^ { \star } } } ^ { K } ( x )
$$

By Proposition 3.1(ii), $q ^ { \star }$ is a probability distribution. Hence the equilibrium identity can be written as

$$
p ^ { \star } = ( 1 - \eta ) \mu + \eta q ^ { \star } .
$$

Applying $\bar { \varphi } ( \cdot )$ to both sides and using linearity,

$$
\bar { \varphi } ( p ^ { \star } ) = ( 1 - \eta ) \bar { \varphi } ( \mu ) + \eta \bar { \varphi } ( q ^ { \star } ) .
$$

Fix any protected direction $v _ { \ell } .$ Taking inner products with $v _ { \ell } ,$ , we obtain

$$
\big \langle v _ { \ell } , \bar { \varphi } ( p ^ { \star } ) \big \rangle = ( 1 - \eta ) \big \langle v _ { \ell } , \bar { \varphi } ( \mu ) \big \rangle + \eta \big \langle v _ { \ell } , \bar { \varphi } ( q ^ { \star } ) \big \rangle .
$$

Now

$$
\left. v _ { \ell } , \bar { \varphi } ( \mu ) \right. = \sum _ { i = 1 } ^ { n } \mu _ { i } \left. v _ { \ell } , \varphi _ { i } \right. .
$$

Define

$$
m _ { \ell } : = \operatorname* { m i n } _ { i = 1 , \ldots , n } \langle v _ { \ell } , \varphi _ { i } \rangle .
$$

Since $q ^ { \star } \in \mathcal { P } ( \mathcal { X } )$ , we have

$$
\left. v _ { \ell } , \bar { \varphi } ( q ^ { \star } ) \right. = \sum _ { i = 1 } ^ { n } q ^ { \star } ( x _ { i } ) \langle v _ { \ell } , \varphi _ { i } \rangle \geq \sum _ { i = 1 } ^ { n } q ^ { \star } ( x _ { i } ) m _ { \ell } = m _ { \ell } .
$$

Therefore,

$$
\left. v _ { \ell } , \bar { \varphi } ( p ^ { \star } ) \right. \geq ( 1 - \eta ) \sum _ { i = 1 } ^ { n } \mu _ { i } \langle v _ { \ell } , \varphi _ { i } \rangle + \eta m _ { \ell } .
$$

Now assume

$$
\sum _ { i = 1 } ^ { n } \mu _ { i } \langle v _ { \ell } , \varphi _ { i } \rangle \geq \frac { \theta _ { \ell } - \eta m _ { \ell } } { 1 - \eta } .
$$

Then

$$
\left. v _ { \ell } , \bar { \varphi } ( p ^ { \star } ) \right. \geq ( 1 - \eta ) \cdot \frac { \theta _ { \ell } - \eta m _ { \ell } } { 1 - \eta } + \eta m _ { \ell } = \theta _ { \ell } .
$$

Since this holds for every $\ell = 1 , \ldots , L ,$ , the equilibrium satisfies

$$
\left. v _ { \ell } , \bar { \varphi } ( p _ { K } ^ { \star } ( \mu , \eta ) ) \right. \geq \theta _ { \ell } , \qquad \ell = 1 , \ldots , L .
$$

This proves the claim.

## D Extension to Continuous Instance Spaces

In the main text, we present the model and results in the discrete setting with finite ${ \mathcal { X } } = \{ x _ { 1 } , \ldots , x _ { n } \}$ . This appendix extends every result to the case where $\mathcal { X } \subset \mathbb { R } ^ { d }$ is compact instance space.

We denote by $\mathcal { P } ( \mathcal { X } )$ the set of Borel probability measures on X. The reward $r ( x ; w ) = \langle w , \varphi ( x ) \rangle$ with $\| \varphi ( x ) \| = 1$ for every $x \in \mathcal { X }$ , feature map, preference state w $\in \mathcal { W } : = \{ w \in \mathbb { R } ^ { d } : \| w \| \leq 1 \}$ , and mean feature operator $\bar { \varphi } ( p ) : = \textstyle \int _ { \mathcal { X } } \varphi ( x ) p ( d x )$ carry over unchanged. The coupled dynamics are

$$
p _ { t + 1 } = ( 1 - \eta ) p _ { \mathrm { r e f } } + \eta p _ { t } H _ { p _ { t } , w _ { t } } ^ { K } \qquad w _ { t + 1 } = \mathrm { P r o j } _ { \mathcal { W } } \bigl ( ( 1 - \beta ) w _ { t } + \beta \bar { \varphi } ( p _ { t + 1 } ) \bigr )\tag{11}
$$

where $p _ { \mathrm { r e f } } \in { \mathcal { P } } ( { \mathcal { X } } )$ is a fixed reference measure, $\eta \in [ 0 , 1 ]$ , and the curation operator $H _ { p , w } ^ { K }$ is defined as in Def. 2.1:

$$
H _ { p , w } ^ { K } ( x ) = \mathbb { E } _ { { \cal X } _ { 1 } , \ldots , { \cal X } _ { K - 1 } \overset { \mathrm { i i d } } { \sim } p } \Big [ \frac { K e ^ { \tau r _ { w } ( x ) } } { e ^ { \tau r _ { w } ( x ) } + \sum _ { i = 1 } ^ { K - 1 } e ^ { \tau r _ { w } ( X _ { i } ) } } \Big ] .
$$

All results below hold for every $K \geq 2$ as well as in the limit $K  \infty$

## D.1 Retraining with Purely Synthetic User-Curated Data

This appendix extends the results to a compact instance space $X \subset \mathbb { R } ^ { m }$ . Throughout Appendix $\mathbf { C } , { \mathcal { P } } ( X )$ denotes the set of Borel probability measures on X. We assume that the feature map $\phi : X  \mathbb { R } ^ { d }$ is continuous and satisfies $\lVert \phi ( x ) \rVert = \hat { 1 }$ for all $x \in X$ . Hence $r _ { w } ( x ) = \langle w , \phi ( x ) \rangle$ and ${ a } _ { w } ( x ) : = \exp ( \tau r _ { w } ( x ) )$ are continuous and satisfy ${ e ^ { - \tau } \leq a _ { w } ( x ) \leq e ^ { \tau } }$

Proposition D.1 (Basic properties of the curation operator on compact spaces). For any $p \in \mathcal { P } ( \mathcal { X } ) , w \in \mathcal { W }$ and $K \geq 2 ,$ , the curation operator satisfies:

1. Strict monotonicity. For any $x , y \in \mathcal { X } , r _ { w } ( x ) > r _ { w } ( y ) \iff H _ { p , w } ^ { K } ( x ) > H _ { p , w } ^ { K } ( y )$

2. Normalization. $\begin{array} { r } { \int _ { \mathcal { X } } H _ { p , w } ^ { K } ( x ) p ( d x ) = 1 . } \end{array}$

3. Odds contraction. $I f r _ { w } ( x ) - r _ { w } ( y ) \geq r _ { \Delta } > 0 ,$ , then $\begin{array} { r } { \frac { H _ { p , w } ^ { K } ( y ) } { H _ { p , w } ^ { K } ( x ) } \leq \lambda _ { K } { \left( r _ { \Delta } \right) } : = 1 - \frac { ( 1 - e ^ { - \tau r _ { \Delta } } ) ( K - 1 ) } { e ^ { 2 \tau } + K - 1 } \in ( 0 , 1 ) , } \end{array}$ and $\lambda _ { K } ( r _ { \Delta } ) \xrightarrow { K  \infty } e ^ { - \tau r _ { \Delta } }$

Proof. Each property is proved by arguments that are entirely pointwise and do not rely on finiteness of $x ;$ they use only the bound $e ^ { - \tau } \leq e ^ { \tau r _ { w } ( x ) } \leq e ^ { \tau }$

Strict monotonicity. For every fixed realization $\begin{array} { r } { S : = \sum _ { \ell = 1 } ^ { K - 1 } e ^ { \tau r _ { w } ( X _ { \ell } ) } > 0 } \end{array}$ , the map $a \mapsto K a / ( a + S )$ is strictly increasing on $( 0 , \infty )$ , and $\begin{array} { r } { a ( x ) : = e ^ { \tau r _ { w } ( x ) } } \end{array}$ is strictly increasing in $r _ { w } ( x )$ . Hence, for every sample path,

$$
r _ { w } ( x ) > r _ { w } ( y ) \implies \frac { K a ( x ) } { a ( x ) + S } > \frac { K a ( y ) } { a ( y ) + S } .
$$

Taking expectations preserves the strict inequality (both sides are continuous functions of the sample path, and the inequality is strict on a set of full measure).

Normalization. Let $X _ { 1 } , \ldots , X _ { K } \stackrel { \mathrm { i i d } } { \sim } p .$ By Fubini’s theorem,

$$
\int _ { \mathcal { X } } H _ { p , w } ^ { K } ( x ) p ( d x ) = \mathbb { E } \left[ \sum _ { j = 1 } ^ { K } \frac { e ^ { \tau r _ { w } ( X _ { j } ) } } { \sum _ { i = 1 } ^ { K } e ^ { \tau r _ { w } ( X _ { i } ) } } \right] = \mathbb { E } [ 1 ] = 1 .
$$

The first equality uses the symmetry of the K i.i.d. draws: each of the K positions contributes identically, so integrating $H _ { p , w } ^ { K ^ { \bullet } } ( x )$ against p(dx) (which accounts for one position) and multiplying by the factor K inside $H _ { p , w } ^ { K }$ reconstructs the full sum over all K positions.

The second equality holds because the Luce probabilities sum to 1 for every realization $( X _ { 1 } , \ldots , X _ { K } )$

Odds contraction. Write $\begin{array} { r } { \iota ( x ) = e ^ { \tau r _ { w } ( x ) } , a ( y ) = e ^ { \tau r _ { w } ( y ) } , S = \sum _ { \ell = 1 } ^ { K - 1 } a ( X _ { \ell } ) . } \end{array}$

Since $r _ { w } ( x ) - r _ { w } ( y ) \ge r _ { \Delta }$ , we have $a ( x ) - a ( y ) \geq ( 1 - e ^ { - \tau r \Delta } ) a ( x )$ . For any fixed $S \geq 0$

$$
\frac { a ( y ) / ( a ( y ) + S ) } { a ( x ) / ( a ( x ) + S ) } = 1 - \frac { S \left( a ( x ) - a ( y ) \right) } { a ( x ) \left( a ( y ) + S \right) } .
$$

Using $a ( x ) - a ( y ) \geq ( 1 - e ^ { - \tau r _ { \Delta } } ) a ( x ) , a ( y ) \leq e ^ { \tau }$ , and $S \geq ( K - 1 ) e ^ { - \tau }$ , the ratio $S / ( e ^ { \tau } + S )$ is bounded below by $( K - 1 ) / ( e ^ { 2 \dot { \tau } } + K - 1 )$ . Taking expectations yields the stated bound. □

Theorem D.2 (Equilibrium characterization on compact spaces). When $\eta = 1$ , a distribution $p \in { \mathcal { P } } ( { \mathcal { X } } )$ is a fixed point ofthe retraining step under preference w $\bar { i } f$ and only ifthe reward $r _ { w }$ is constant on $\operatorname { s u p p } ( p ) .$

$$
x , x ^ { \prime } \in \operatorname { s u p p } ( p ) \quad \Longrightarrow \quad r _ { w } ( x ) = r _ { w } ( x ^ { \prime } ) .
$$

Consequently, for every $x ^ { \star } \in \mathcal { X } ,$ , the pair $( p ^ { \star } = \delta _ { x ^ { \star } } , w ^ { \star } = \varphi ( x ^ { \star } ) )$ is an equilibrium.

Proof. Fixed-point characterization. Let $\mu _ { p } ^ { K } ( d x ) : = H _ { p , w } ^ { K } ( x ) p ( d x )$ . The fixed-point equation is $p = \mu _ { p } ^ { K }$ equivalently

$$
\int _ { A } ( H _ { p , w } ^ { K } ( x ) - 1 ) p ( d x ) = 0 , \qquad \forall A \in \mathcal { B } ( X ) .
$$

Thus $H _ { p , w } ^ { K } ( x ) = 1$ for $p \mathrm { - a . e . ~ } x .$ Since $\phi$ is continuous, $H _ { p , w } ^ { K }$ is continuous in x by dominated convergence.   
Therefore $H _ { p , w } ^ { K } ( x ) = 1$ on supp(p).

If $x , y \in \operatorname { s u p p } ( p )$ and $r _ { w } ( x ) \neq r _ { w } ( y )$ , strict monotonicity from Proposition C.1 gives $H _ { p , w } ^ { K } ( x ) \neq H _ { p , w } ^ { K } ( y )$ contradicting the fact that both equal 1. Hence $r _ { w }$ is constant on supp(p). Conversely, if r<sub>w</sub> is constant on supp(p), then p-a.s. all sampled points have the same reward as every $x \in \operatorname { s u p p } ( p )$ , so directly $H _ { p , w } ^ { K } ( x ) = 1$ on supp(p) and hence $p H _ { p , w } ^ { K } = p$

Conversely, if all support points share a common reward $c ,$ then $e ^ { \tau r _ { w } ( x ) } = e ^ { \tau c }$ for every $x \in \operatorname { s u p p } ( p )$ , and by direct computation $H _ { p , w } ^ { K } ( x ) = 1$ for every such x.

Singleton equilibria. For any $x ^ { \star } \in \mathcal { X } , \delta _ { x ^ { \star } }$ is trivially supported on a single point, so the equal-reward condition holds. Moreover, $\bar { \varphi } ( \delta _ { x ^ { \star } } ) = \varphi ( x ^ { \star } )$ , and $\mathrm { P r o j } _ { \mathcal { W } } \big ( \big ( 1 - \beta \big ) \overline { { \varphi } } ( x ^ { \star } ) + \beta \varphi ( \overline { { x ^ { \star } } } ) \big ) = \varphi ( x ^ { \star } )$ since $\| \varphi ( x ^ { \star } ) \| = 1 \leq 1$ Hence $( \delta _ { x ^ { \star } } , \varphi ( x ^ { \star } ) )$ is an equilibrium. □

Assumption D.3 (Multiscale reward gap). Fix $x ^ { \star } \in \mathcal { X }$ . There exist a radius $r _ { 0 } > 0$ and sequences $\left( a _ { k } \right) _ { k \geq 0 } .$ $( b _ { k } ) _ { k \geq 0 } , ( \gamma _ { k } ) _ { k \geq 0 }$ with $a _ { k } \downarrow 0 , b _ { k } \downarrow 0 , \gamma _ { k } > 0$ , such that the nested balls $U _ { k } : = B _ { 2 ^ { - k } r \mathrm { { o } } } ( x ^ { \star } ) , k = 0 , 1 , 2 , \bar { . . . } .$ (a sequence of shrinking neighborhoods of $x ^ { \star }$ with radii halving at each scale) satisfy, for every $k \geq 0 \mathrm { : }$

1. Feature continuity: $\operatorname* { s u p } _ { x \in U _ { k } } \| \varphi ( x ) - \varphi ( x ^ { \star } ) \| \leq a _ { k }$

2. Reward gap: $\| w - \varphi ( x ^ { \star } ) \| \leq b _ { k } \implies \operatorname* { i n f } _ { y \in U _ { k + 1 } } \langle w , \varphi ( y ) \rangle - \operatorname* { s u p } _ { x \notin U _ { k } } \langle w , \varphi ( x ) \rangle \geq \gamma _ { k } .$

3. Scale compatibility: $a _ { k } < b _ { k + 1 }$ for all $k \geq 0$ , and $a _ { 0 } < b _ { 0 }$

Remark D.4 (Interpretation). Assumption D.3 is the continuous analogue of the feature-separation condition $\varphi ( x _ { j } ) \neq \varphi ( x _ { i } )$ for all $j \neq i$ in Theorem 3.3. In the discrete case, the finite set of competitors admits a single uniform reward gap $\begin{array} { r } { \gamma _ { i } = \operatorname* { m i n } _ { j \neq i } \langle \varphi ( x _ { i } ) , \varphi ( x _ { i } ) - \varphi ( x _ { j } ) \rangle > 0 } \end{array}$ , and the multiscale conditions are automatically satisfied with $a _ { k } = 0 , b _ { k } = \gamma _ { i } / 4 , \gamma _ { k } = \gamma _ { i } / 2$ . In the continuous case, competitors can be arbitrarily close $\mathrm { t o } \ x ^ { \star }$ with arbitrarily small reward disadvantage, so the conditions must hold at every scale separately. Specifically: (i) controls how well the feature mean approximates $\varphi ( x ^ { \star } )$ after mass concentrates at scale $k ; ( \mathrm { i i } )$ provides the reward advantage that drives mass from $\bar { U } _ { k } ^ { c }$ into $U _ { k + 1 } ;$ (iii) ensures that the feature accuracy gained at scale k is sufficient to activate the reward gap at scale $k + 1$ . These conditions are not implied by ordinary feature separation alone. They hold, for example, whenever the chosen neighborhoods $( U _ { k } )$ satisfy a uniform annular reward separation condition

$$
\operatorname* { i n f } _ { y \in U _ { k + 1 } } \langle \phi ( x ^ { \star } ) , \phi ( y ) \rangle > \operatorname* { s u p } _ { x \notin U _ { k } } \langle \phi ( x ^ { \star } ) , \phi ( x ) \rangle
$$

with margins large enough to dominate perturbations $\| w - \phi ( x ^ { \star } ) \| \leq b _ { k }$

Theorem D.5 (Local convergence to singleton equilibria on compact spaces). Let $x ^ { \star } \in \mathcal { X }$ satisfy Assumption D.3. Then there exists $\delta ^ { \star } > 0$ such that $i f$

$$
x ^ { \star } \in \mathrm { s u p p } ( p _ { 0 } ) , \qquad \| w _ { 0 } - \varphi ( x ^ { \star } ) \| \leq \delta ^ { \star } , \qquad p _ { 0 } ( U _ { 1 } ) \geq 1 - \delta ^ { \star } ,
$$

then $p _ { t } ( B _ { \varepsilon } ( x ^ { \star } ) ) \to 1$ for every $\varepsilon > 0$ and $w _ { t } \to \varphi ( x ^ { \star } )$

Proof. Since $a _ { 0 } < b _ { 0 }$ , choose $\delta ^ { \star } > 0$ with

$$
0 < \delta ^ { \star } \leq \operatorname* { m i n } \bigl \{ { \textstyle { \frac { 1 } { 2 } } } , \ b _ { 0 } \bigr \} , \qquad a _ { 0 } + \frac { 2 \delta ^ { \star } } { 1 - \delta ^ { \star } } \leq b _ { 0 }\tag{12}
$$

Define $q _ { k , t } : = p _ { t } ( U _ { k } ^ { c } )$ (the mass outside scale k) and $s _ { k , t } : = p _ { t } ( U _ { k + 1 } )$ (the mass inside scale $k + 1 )$

Assume $x ^ { \star } \in$ supp(p<sub>0</sub>), ∥w<sub>0</sub> − φ(x<sup>⋆</sup>)∥ ≤ δ<sup>⋆</sup>, and $p _ { 0 } ( U _ { 1 } ) \geq 1 - \delta ^ { \star }$

Step 1: The trajectory stays inside the initial preference neighborhood.

We prove $\| w _ { t } - \varphi ( x ^ { \star } ) \| \le b _ { 0 }$ for all t by induction.

The base case is $\| w _ { 0 } - \varphi ( x ^ { \star } ) \| \le \delta ^ { \star } \le b _ { 0 }$

Suppose $\| w _ { t } - \varphi ( x ^ { \star } ) \| \le b _ { 0 }$ . Then reward gap with $k = 0$ gives

$$
\operatorname* { i n f } _ { y \in U _ { 1 } } r ( y ; w _ { t } ) - \operatorname* { s u p } _ { x \notin U _ { 0 } } r ( x ; w _ { t } ) \geq \gamma _ { 0 } .
$$

When $\eta = 1$ , the retraining step gives $\begin{array} { r } { p _ { t + 1 } ( A ) = \int _ { A } H _ { p _ { t } , w _ { t } } ^ { K } ( x ) p _ { t } ( d x ) } \end{array}$

For every $y \in U _ { 0 } ^ { c }$ and $x \in U _ { 1 }$ ,it yields $r _ { w _ { t } } ( x ) - r _ { w _ { t } } ( y ) \geq \gamma _ { 0 } ,$ so Proposition D.1 (odds contraction) gives

$$
H _ { p t , w _ { t } } ^ { K } ( y ) \leq \lambda _ { K } ( \gamma _ { 0 } ) H _ { p t , w _ { t } } ^ { K } ( x ) .
$$

Since this holds for every $x \in { U } _ { 1 } , H _ { p _ { t } , w _ { t } } ^ { K } ( y ) \leq \lambda _ { K } ( \gamma _ { 0 } ) \ \operatorname* { i n f } _ { x \in { U } _ { 1 } } H _ { p _ { t } , w _ { t } } ^ { K } ( x )$ for all $y \in U _ { 0 } ^ { c }$

Therefore

$$
q _ { 0 , t + 1 } = \int _ { U _ { 0 } ^ { c } } H _ { p _ { t } , w _ { t } } ^ { K } ( y ) p _ { t } ( d y ) \leq \lambda _ { K } ( \gamma _ { 0 } ) \operatorname* { i n f } _ { x \in U _ { 1 } } H _ { p _ { t } , w _ { t } } ^ { K } ( x ) \cdot q _ { 0 , t }
$$

and

$$
s _ { 0 , t + 1 } = \int _ { U _ { 1 } } H _ { p _ { t } , w _ { t } } ^ { K } ( x ) p _ { t } ( d x ) \geq \operatorname* { i n f } _ { x \in U _ { 1 } } H _ { p _ { t } , w _ { t } } ^ { K } ( x ) \cdot s _ { 0 , t } ,
$$

whence

$$
\frac { q _ { 0 , t + 1 } } { s _ { 0 , t + 1 } } \leq \lambda _ { K } ( \gamma _ { 0 } ) \ \frac { q _ { 0 , t } } { s _ { 0 , t } } .\tag{13}
$$

Iterating,

$$
\frac { q _ { 0 , t } } { s _ { 0 , t } } \leq \lambda _ { K } ( \gamma _ { 0 } ) ^ { t } \cdot \frac { \delta ^ { \star } } { ( 1 - \delta ^ { \star } ) }
$$

and in particular $\begin{array} { r } { q _ { 0 , t } \le \frac { \delta ^ { \star } } { ( 1 - \delta ^ { \star } ) } } \end{array}$ for all t.

By feature continuity and $\| \varphi ( x ) \| = 1$

$$
\lVert \bar { \varphi } ( p _ { t + 1 } ) - \varphi ^ { \star } \rVert \leq a _ { 0 } + 2 q _ { 0 , t + 1 } .
$$

Since projection is nonexpansive and $\varphi ^ { \star } \in { \mathcal { W } }$

$$
\left\| w _ { t + 1 } - \varphi ( x ^ { \star } ) \right\| \leq \left( 1 - \beta \right) \left\| w _ { t } - \varphi ( x ^ { \star } ) \right\| + \beta \left( a _ { 0 } + 2 q _ { 0 , t + 1 } \right) \leq \left( 1 - \beta \right) b _ { 0 } + \beta \Big ( a _ { 0 } + \frac { 2 \delta ^ { \star } } { 1 - \delta ^ { \star } } \Big ) \leq b _ { 0 } .
$$

## Step 2: First-scale outer leakage vanishes.

Since $\| w _ { t } - \varphi ( x ^ { \star } ) \| \le b _ { 0 }$ for all t, inequality (13) holds for all t, so $\begin{array} { r } { \frac { q _ { 0 , t } } { s _ { 0 , t } } \to 0 } \end{array}$ geometrically.

Since $q _ { 0 , t } \leq { \frac { q _ { 0 , t } } { s _ { 0 , t } } }$ , we get $q _ { 0 , t } \to 0$

Step 3: Scale-by-scale bootstrap.

We prove by induction: for every $k \geq 0$ there exists $T _ { k } < \infty$ with $\| w _ { t } - \varphi ( x ^ { \star } ) \| \le b _ { k }$ for all $t \geq T _ { k }$

The case $k = 0$ is Step 1. Suppose the claim holds for k.

Supportpreservation. Since $H _ { p , w } ^ { K } ( x ) > 0$ for all $x \left( \mathrm { a s } e ^ { \tau r _ { w } \left( x \right) } > 1 \right.$ 0 and the denominator in Definition 2.1 is finite and positive), the pure update preserves the support: s $\mathrm { u p p } ( p _ { t } ) = \mathrm { s u p p } ( p _ { 0 } )$ for all t. Because $x ^ { \star } \in \mathrm { s u p p } ( p _ { 0 } )$ and $U _ { k + 1 }$ is open, $s _ { k , t } = p _ { t } ( U _ { k + 1 } ) > 0$ for all t.

Mass concentration. For $t \geq T _ { k }$ , reward gap gives at scale k.

Exactly as in Step 1, Proposition D.1 (odds contraction) yields

$$
\frac { q _ { k , t + 1 } } { s _ { k , t + 1 } } \leq \lambda _ { K } ( \gamma _ { k } ) \ \frac { q _ { k , t } } { s _ { k , t } } ,
$$

and iterating gives $q _ { k , t } \to 0$

Preference bootstrap. $\begin{array} { r } { \left\| \bar { \varphi } ( p _ { t } ) - \varphi ^ { \star } \right\| \leq a _ { k } + 2 q _ { k , t } , \operatorname { s o } \left\| w _ { t + 1 } - \varphi ( x ^ { \star } ) \right\| \leq ( 1 - \beta ) \left\| w _ { t } - \varphi ( x ^ { \star } ) \right\| + \beta \left( a _ { k } + \varphi ( x ^ { \star } ) \right) \left\| w _ { t } - \varphi ( x ^ { \star } ) \right\| . } \end{array}$ $^ { 2 q _ { k , t + 1 } ) }$

Since $q _ { k , t } \to 0$ , lim sup $\| w _ { t } - \varphi ( x ^ { \star } ) \| \le a _ { k } < b _ { k + 1 }$ , giving $T _ { k + 1 }$ with $\| w _ { t } - \varphi ( x ^ { \star } ) \| \leq b _ { k + 1 }$ for $t \geq T _ { k + 1 }$ Step 4: Convergence. For every $k , p _ { t } ( U _ { k } ^ { c } ) \to 0$

Given $\varepsilon > 0 .$ , choose k with $r _ { k } < \varepsilon ;$ thenp<sub>t</sub> $( B _ { \varepsilon } ( x ^ { \star } ) ^ { c } ) \leq p _ { t } ( U _ { k } ^ { c } ) \to 0$

The induction gives lim su $\mathsf { p } _ { t } d _ { t } \leq a _ { k } \to 0 , \mathsf { s o } w _ { t } \to \varphi ^ { \star }$

Corollary D.6 (Failure of local stability on compact spaces). Fix $x ^ { \star } \in \mathcal { X } .$ . If there exists $x ^ { \prime } \neq x ^ { \star }$ with $\varphi ( x ^ { \prime } ) = \varphi ( x ^ { \star } )$ , then the singleton equilibrium $( \bar { \delta _ { x ^ { \star } } } , \varphi ( \bar { x ^ { \star } } ) )$ is not locally asymptotically stable.

Proof. For any $\varepsilon \in ( 0 , 1 )$ , define $p _ { \varepsilon } : = \left( 1 - \varepsilon \right) \delta _ { x ^ { \star } } + \varepsilon \delta _ { x ^ { \prime } }$ and $w _ { \varepsilon } : = \varphi ( x ^ { \star } )$

Since $\varphi ( x ^ { \prime } ) = \varphi ( x ^ { \star } )$ , the two support points share the same reward under $w _ { \varepsilon } .$ , so $H _ { p _ { \varepsilon } , w _ { \varepsilon } } ^ { K } ( x ^ { \star } ) = H _ { p _ { \varepsilon } , w _ { \varepsilon } } ^ { K } ( x ^ { \prime } ) =$ 1 by the equal-reward characterization in Theorem D.2.

Moreover, $\bar { \varphi } ( p _ { \varepsilon } ) = \varphi ( x ^ { \star } ) = w _ { \varepsilon }$

Hence $( p _ { \varepsilon } , w _ { \varepsilon } )$ is an equilibrium for every $\varepsilon > 0$

Since $( p _ { \varepsilon } , w _ { \varepsilon } ) \to ( \delta _ { x ^ { \star } } , \varphi ( x ^ { \star } ) ) \mathrm { a s } \varepsilon \to 0$ , the singleton equilibrium is not isolated and therefore not locally asymptotically stable. □

## D.2 Retraining on Mixed Synthetic and Reference Data

We now consider $\eta \in ( 0 , 1 )$ , paralleling Section 3.2.

Lemma D.7 (Sensitivity of the retraining step on compact spaces). For $p \in { \mathcal { P } } ( { \mathcal { X } } )$ , define the probability measure $p H _ { p , w } ^ { K } b y$

$$
( p H _ { p , w } ^ { K } ) ( A ) : = \int _ { A } H _ { p , w } ^ { K } ( x ) p ( d x ) , \qquad A \in { \mathcal { B } } ( { \mathcal { X } } ) .
$$

Then for any p, $q \in { \mathcal { P } } ( { \mathcal { X } } )$ , w, $v \in \mathcal W ,$ , and $K \geq 2 ,$

$$
d _ { \mathrm { T V } } \left( p H _ { p , w } ^ { K } , q H _ { q , w } ^ { K } \right) \leq L _ { p } d _ { \mathrm { T V } } ( p , q ) ,
$$

$$
d _ { \mathrm { T V } } \big ( p H _ { p , w } ^ { K } , p H _ { p , v } ^ { K } \big ) \leq L _ { w } \| w - v \| ,
$$

and

$$
\begin{array} { r } { \| \bar { \phi } ( p ) - \bar { \phi } ( q ) \| \le 2 d _ { \mathrm { T V } } ( p , q ) , } \end{array}
$$

where $L _ { p } : = 2 e ^ { 2 \tau } ( 2 + e ^ { 2 \tau } )$ and $L _ { w } : = 2 \tau e ^ { 4 \tau }$

Proof. The proof of Lemma 3.6 (Sec. B.6) relies on three ingredients:

1. the uniform bounds $e ^ { - \tau } \leq e ^ { \tau r _ { w } ( x ) } \leq e ^ { \tau } ;$

2. the variational formula $| \int f d ( p - q ) | \leq 2 \| f \| _ { \infty } d _ { \mathrm { T V } } ( p , q )$ , valid on any measurable space;

3. the coordinate-swapping telescoping argument bounding $| H _ { p , w } ^ { K } ( x ) - H _ { q , w } ^ { K } ( x ) |$ | (Part 1) and $| H _ { p , w } ^ { K } ( x ) -$ $H _ { p , v } ^ { K } ( x ) | ( \mathrm { P a r t } 2 )$

None of these steps use finiteness of $x \colon$ the sums in the discrete proof become integrals, and all bounds hold for arbitrary Borel probability measures.

Part 1: p-Lipschitz bound.

For any measurable A,

$$
( p H _ { p , w } ^ { K } - q H _ { q , w } ^ { K } ) ( A ) = \int _ { A } H _ { p , w } ^ { K } d ( p - q ) + \int _ { A } ( H _ { p , w } ^ { K } - H _ { q , w } ^ { K } ) d q .
$$

Taking the supremum over measurable A gives

$$
2 d _ { \mathrm { T V } } ( p H _ { p , w } ^ { K } , q H _ { q , w } ^ { K } ) \leq 2 \| H _ { p , w } ^ { K } \| _ { \infty } d _ { \mathrm { T V } } ( p , q ) + \| H _ { p , w } ^ { K } - H _ { q , w } ^ { K } \| _ { \infty } .
$$

The first term contributes at most $4 e ^ { 2 \tau } d _ { \mathrm { T V } } ( p , q )$ since $0 \leq H _ { p , w } ^ { K } ( x ) \leq 2 e ^ { 2 \tau }$ . The second term is bounded via the coordinate-swapping argument: the effect of replacing each of the $K - 1$ sample coordinates from p to q contributes at most $\hat { 2 e } ^ { 4 \bar { \tau } } / \dot { K }$ in the sup-norm of the integrand, yielding a total of at most $2 e ^ { 4 \tau } d _ { \mathrm { T V } } ( p , q )$

Combining: $d _ { \mathrm { T V } } ( p H _ { p , w } ^ { K } , q H _ { q , w } ^ { K } ) \leq ( 2 e ^ { 2 \tau } + e ^ { 4 \tau } ) d _ { \mathrm { T V } } ( p , q ) = L _ { p } d _ { \mathrm { T V } } ( p , q ) .$

Part 2: w-Lipschitz bound.

Using $\left. e ^ { \tau r _ { w } \left( x \right) } - e ^ { \tau r _ { v } \left( x \right) } \right. \leq \tau e ^ { \tau } \left. w - v \right.$ and the two-term decomposition of $H _ { p , w } ^ { K } ( x ) - H _ { p , v } ^ { K } ( x )$ (see Sec. B.6, Part 2), each term is bounded by τe<sup>4τ</sup> ∥w − v∥, giving $d _ { \mathrm { T V } } ( p H _ { p , w } ^ { K } , p H _ { p , v } ^ { K } ) \leq 2 \tau e ^ { 4 \tau } \left. w - v \right. = L _ { w } \left. w - v \right.$ Part 3: Mean-feature bound. For any unit vector u, $| \langle u , \varphi ( x ) \rangle | \leq 1 , \operatorname { s o } | \langle u , \bar { \varphi } ( p ) - \bar { \varphi } ( q ) \rangle | \leq 2 d _ { \mathrm { T V } } ( p , q )$

Taking the supremum over u gives $\| \bar { \varphi } ( p ) - \bar { \varphi } ( q ) \| \leq 2 d \mathrm { r v } ( p , q )$

Theorem D.8 (Unique equilibrium under frozen-w on compact spaces). Fix $w \in \mathcal W$ and $K \ge 2 o r K = \infty$ Define $T _ { w } ^ { K } : \mathcal { P } ( \mathcal { X } ) \overset { \cdot } {  } \mathcal { P } ( \mathcal { X } )$ by

$$
T _ { w } ^ { K } ( p ) ( A ) : = ( 1 - \eta ) p _ { \mathrm { r e f } } ( A ) + \eta \int _ { A } H _ { p , w } ^ { K } ( x ) p ( d x ) , \qquad A \in \mathcal { B } ( \mathcal { X } ) .
$$

$I f \eta L _ { p } < 1$ , then $T _ { w } ^ { K }$ is a strict contraction on $( \mathcal { P } ( \mathcal { X } ) , d _ { \mathrm { T V } } )$ with rate $\eta L _ { p } .$ . Consequently, there exists a unique equilibrium $p _ { w } ^ { K , \eta }$ and every trajectory converges geometrically:

$$
d _ { \mathrm { T V } } ( p _ { t } , p _ { w } ^ { K , \eta } ) \leq ( \eta L _ { p } ) ^ { t } d _ { \mathrm { T V } } ( p _ { 0 } , p _ { w } ^ { K , \eta } ) .
$$

Proof. For any $p , q \in { \mathcal { P } } ( { \mathcal { X } } )$ and measurable A,

$$
T _ { w } ^ { K } ( p ) ( A ) - T _ { w } ^ { K } ( q ) ( A ) = \eta \big [ ( p H _ { p , w } ^ { K } ) ( A ) - ( q H _ { q , w } ^ { K } ) ( A ) \big ] .
$$

Therefore, by Lemma D.7,

$$
d _ { \mathrm { T V } } ( T _ { w } ^ { K } ( p ) , T _ { w } ^ { K } ( q ) ) = \eta d _ { \mathrm { T V } } ( p H _ { p , w } ^ { K } , q H _ { q , w } ^ { K } ) \leq \eta L _ { p } d _ { \mathrm { T V } } ( p , q ) .
$$

If $\eta L _ { p } < 1$ , this is a strict contraction. Since $( \mathcal { P } ( \mathcal { X } ) , d _ { \mathrm { T V } } )$ is complete, Banach’s fixed-point theorem gives a unique fixed point $p _ { w } ^ { K , \eta }$ and geometric convergence. □

Proposition D.9 (Closed-form equilibrium on compact spaces). Assume the condition of Theorem D.8, in particular $\eta L _ { p } < 1$ . For the limiting curation operator $K = \infty ,$ , define

$$
a _ { w } ( x ) : = e ^ { \tau r _ { w } ( x ) } , \qquad M _ { w } : = \operatorname* { s u p } _ { x \in \mathcal { X } } a _ { w } ( x ) .
$$

Then the frozen-w equilibrium is given, for every measurable $A \subseteq { \mathcal { X } } , b _ { \mathrm { } }$ y

$$
{ p } _ { w } ^ { \infty , \eta } ( A ) = \int _ { A } \frac { \left( 1 - \eta \right) Z _ { w } ^ { \eta } } { Z _ { w } ^ { \eta } - \eta a _ { w } ( x ) } p _ { \mathrm { r e f } } ( d x ) ,
$$

where $Z _ { w } ^ { \eta } > \eta M _ { w }$ is the unique scalar solving

$$
\int _ { \mathcal { X } } \frac { ( 1 - \eta ) a _ { w } ( x ) } { Z _ { w } ^ { \eta } - \eta a _ { w } ( x ) } p _ { \mathrm { r e f } } ( d x ) = 1 .
$$

Proof. For $K = \infty$ , the curation operator is

$$
H _ { p , w } ^ { \infty } ( x ) = \frac { a _ { w } ( x ) } { \int _ { \mathcal { X } } a _ { w } ( z ) p ( d z ) } .
$$

Let p be a frozen-w fixed point and set

$$
Z : = \int _ { \mathcal X } a _ { w } ( z ) p ( d z ) .
$$

Since ${ e ^ { - \tau } \leq a _ { w } ( x ) \leq e ^ { \tau } }$ , we have $Z \ge e ^ { - \tau }$ . Moreover, $\eta L _ { p } < 1$ implies $\eta < e ^ { - 2 \tau }$ , and hence

$$
\eta M _ { w } \leq \eta e ^ { \tau } < e ^ { - \tau } \leq Z .
$$

Therefore $Z - \eta a _ { w } ( x ) > 0$ for every $x \in \mathcal { X }$

The fixed-point equation

$$
p = ( 1 - \eta ) p _ { \mathrm { r e f } } + \eta p H _ { p , w } ^ { \infty }
$$

is equivalently

$$
p ( d x ) \left( 1 - \frac { \eta a _ { w } ( x ) } { Z } \right) = ( 1 - \eta ) p _ { \mathrm { r e f } } ( d x ) .
$$

Thus

$$
p ( d x ) = \frac { ( 1 - \eta ) Z } { Z - \eta a _ { w } ( x ) } p _ { \mathrm { r e f } } ( d x ) .
$$

Substituting this expression into the definition of Z and dividing by $Z > 0$ gives the scalar equation

$$
\int _ { \mathcal { X } } \frac { ( 1 - \eta ) a _ { w } ( x ) } { Z - \eta a _ { w } ( x ) } p _ { \mathrm { r e f } } ( d x ) = 1 .
$$

It remains to show that this equation has a unique solution on $( \eta M _ { w } , \infty )$ . Define

$$
F ( Z ) : = \int _ { \mathcal { X } } \frac { ( 1 - \eta ) a _ { w } ( x ) } { Z - \eta a _ { w } ( x ) } p _ { \mathrm { r e f } } ( d x ) , \qquad Z > \eta M _ { w } .
$$

The function F is continuous and strictly decreasing on $( \eta M _ { w } , \infty )$ , and $F ( Z ) \to 0$ as $Z \to \infty$

We now check that F is larger than 1 near the left endpoint. Since $a _ { w } ( x ) \in [ e ^ { - \tau } , M _ { w } ]$ , the map $a \mapsto a / ( Z - \eta a )$ is increasing for $Z > \eta M _ { w }$ . Hence

$$
F ( Z ) \geq \frac { ( 1 - \eta ) e ^ { - \tau } } { Z - \eta e ^ { - \tau } } .
$$

Taking $Z \downarrow \eta M _ { w }$ gives

$$
\operatorname* { l i m } _ { Z \downarrow \eta M _ { w } } F ( Z ) \ge \frac { ( 1 - \eta ) e ^ { - \tau } } { \eta ( M _ { w } - e ^ { - \tau } ) } ,
$$

with the right-hand side interpreted as $+ \infty \mathrm { i f } M _ { w } = e ^ { - \tau }$ . Since $M _ { w } \le e ^ { \tau }$ and $\eta < e ^ { - 2 \tau }$ , we have $\eta M _ { w } < e ^ { - \tau }$ which implies

$$
\frac { ( 1 - \eta ) e ^ { - \tau } } { \eta ( M _ { w } - e ^ { - \tau } ) } > 1
$$

whenever $M _ { w } > e ^ { - \tau }$ . Therefore $F ( Z ) > 1$ for $Z$ sufficiently close to $\eta M _ { w } ,$ , while $F ( Z ) \to 0$ as $Z \to \infty$ . By the intermediate value theorem, there exists a solution $Z _ { w } ^ { \eta } > \eta M _ { w }$ , and strict monotonicity gives uniqueness.

Finally, for this unique $Z _ { w } ^ { \eta }$ , define $p _ { w } ^ { \infty , \eta }$ by the displayed formula. The scalar equation implies

$$
\int _ { \mathcal { X } } a _ { w } ( x ) p _ { w } ^ { \infty , \eta } ( d x ) = Z _ { w } ^ { \eta } ,
$$

and also

$$
p _ { w } ^ { \infty , \eta } ( \mathcal { X } ) = ( 1 - \eta ) + \eta \int _ { \mathcal { X } } \frac { ( 1 - \eta ) a _ { w } ( x ) } { Z _ { w } ^ { \eta } - \eta a _ { w } ( x ) } p _ { \mathrm { r e f } } ( d x ) = 1 .
$$

Substituting the formula back into the frozen-w update verifies that $p _ { w } ^ { \infty , \eta }$ is a fixed point. Uniqueness follows from the uniqueness of $Z _ { w } ^ { \eta }$ , equivalently from the contraction in Theorem D.8. □

Theorem D.10 (Convergence of coupled dynamics on compact spaces). ${ \mathit { I f } } \eta ( L _ { p } + 2 L _ { w } ) < 1$ , then there exists $\begin{array} { r } { \xi \in \big ( \frac { \eta L _ { w } } { \beta ( 1 - 2 \eta L _ { w } ) } , \frac { 1 - \eta L _ { p } } { 2 \beta \eta L _ { p } } \big ) } \end{array}$ such that the coupled update is a strict contraction on $( \mathcal { P } ( \mathcal { X } ) \times \mathcal { W } , \ d _ { \xi } )$ with $d _ { \xi } \big ( ( p , w ) , ( q , v ) \big ) : = d _ { \mathrm { T V } } ( p , q ) + \xi \lVert w - v \rVert$ . Consequently, there exists a unique equilibrium $( p _ { K } ^ { \star } , w _ { K } ^ { \star } )$ and every trajectory converges geometrically.

Proof. Let $( p ^ { + } , w ^ { + } ) : = F ( p , w )$ and $( q ^ { + } , v ^ { + } ) : = F ( q , v )$ , where $F$ denotes the joint update. Data coordinate. By Lemma D.7,

$$
\begin{array} { r } { d _ { \mathrm { T V } } ( p ^ { + } , q ^ { + } ) \leq \eta L _ { p } d _ { \mathrm { T V } } ( p , q ) + \eta L _ { w } \left. w - v \right. . } \end{array}
$$

Preference coordinate. Using nonexpansiveness of projection and $\| \bar { \varphi } ( p ^ { + } ) - \bar { \varphi } ( q ^ { + } ) \| \leq 2 d _ { \mathrm { T V } } ( p ^ { + } , q ^ { + } )$

$$
\begin{array} { r l } & { \| w ^ { + } - v ^ { + } \| \leq ( 1 - \beta ) \| w - v \| + 2 \beta d _ { \mathrm { T V } } ( p ^ { + } , q ^ { + } ) } \\ & { \qquad \leq 2 \beta \eta L _ { p } d _ { \mathrm { T V } } ( p , q ) + ( 1 - \beta + 2 \beta \eta L _ { w } ) \| w - v \| . } \end{array}
$$

Combined metric. Multiplying the preference bound by ξ and adding,

$$
\begin{array} { r } { d _ { \xi } ( F ( p , w ) , F ( q , v ) ) \le A ( \xi ) d _ { \mathrm { T V } } ( p , q ) + \xi B ( \xi ) \| w - v \| , } \end{array}
$$

where $A ( \xi ) : = \eta L _ { p } ( 1 + 2 \xi \beta )$ and $B ( \xi ) : = 1 - \beta + 2 \beta \eta L _ { w } + \eta L _ { w } / \xi$ . The map is a contraction when $A ( \xi ) < 1$ and $B ( \xi ) < 1$ :

$$
A ( \xi ) < 1 \Longleftrightarrow \xi < \frac { 1 - \eta L _ { p } } { 2 \beta \eta L _ { p } } ,
$$

$$
B ( \xi ) < 1 \Longleftrightarrow \xi > \frac { \eta L _ { w } } { \beta ( 1 - 2 \eta L _ { w } ) } .
$$

Both requirements are implied by $\eta ( L _ { p } + 2 L _ { w } ) < 1$ , and under this condition the window is nonempty:

$$
\frac { \eta L _ { w } } { \beta ( 1 - 2 \eta L _ { w } ) } < \frac { 1 - \eta L _ { p } } { 2 \beta \eta L _ { p } } \iff \eta ( L _ { p } + 2 L _ { w } ) < 1 .
$$

Completeness. $\mathcal { P } ( \mathcal { X } )$ is complete under $d _ { \mathrm { T V } }$ when $\mathcal { X }$ is compact, and $\mathcal { W }$ is closed in $\mathbb { R } ^ { d } , \operatorname { s o } \left( \mathcal { P } ( \mathcal { X } ) \times \mathcal { W } , d _ { \xi } \right)$ is complete. The Banach fixed-point theorem gives the unique equilibrium and geometric convergence. □

## E Additional Experimental Results

We conduct all the experiments on a server which has two Intel Xeon 6326 CPU and 6 Nvidia A6000 GPU.

![](images/6f8a1972453f41d743f195514445d71b4caefb3df3942b0a7ae13a6ce20ae1e5.jpg)

![](images/76144cec7c72172302d4758c1a96dae7d35c376221fa609e957d224432fdc0ca.jpg)

![](images/14f67dd04f6d5f6e98b759129b017358000193fec27ba60b49f4baccf66e8669.jpg)

![](images/aebaa5a82ab1aceb155435c4e8a216c08c566fc411724d3c2b133f77a518c106.jpg)

Figure 4: Effect of mixing ratio η $( \tau = 5 . 0 , K = \infty , \beta = 0 . 0 1 )$ .Top: per-class proportions. Bottom: preference vector $w _ { t }$ . Dashed line marks $\theta = 0 . 0 8$  
![](images/c5a7c26bf5800a7473601212609eec6b349f11d9685c8e1432bc363cafc0c47f.jpg)

![](images/dbc9217c36ae1b7dfa5ac511147b8826119cf3b8771b536d70cb33a624ddce99.jpg)

![](images/a30678f865a7957a54f2b27bc940676add9f9b3cfb2bbe4a2f1e4f7194140dbc.jpg)

Figure 5: Effect of preference change rate $\beta \left( \eta = 1 . 0 , \tau = 5 . 0 , K = \infty \right)$  
![](images/25021e5f110bf03a4df800db8892d752002ec8d0342e11b22bd789c5d52a91e9.jpg)

![](images/1c281396a28a4ba4b11e6c8496e851fbf333e90f38cb9256d226990c88ef5a21.jpg)

![](images/8329c60115dcf91d2b3260c9ebc595a01c0a85f7f4ee42249f1ef19e5eccaa0d.jpg)

Figure 6: Effect of curation temperature τ (η = 1.0, K = ∞, β = 0.01).  
![](images/103f8c16e05ab05d29526825c18a743847eea0e5e60a5861c652cc04d360887c.jpg)

![](images/2acc72d5cab22423c13c3f0ac94ebf025a9a50cbb6ea5f7d71189dfe0d22e9f9.jpg)

![](images/68825733c776698fe7448c20e3595c940b41bfb83bb7fcea1cd1248230195765.jpg)  
Figure 7: Effect of selection group size $K \left( \eta = 1 . 0 , \tau = 5 . 0 , \beta = 0 . 0 1 \right)$

## E.1 Additional CFM Experiments on CIFAR-10

## E.1.1 Effect of Key Parameters

We present the effect of key parameters for OT-CFM on CIFAR-10. The default configuration is $\eta = 1 . 0$ $\tau = 5 . 0 , K = \infty , \beta = 0 . 0 1 , \bar { w } _ { 0 } = [ 1 , 0 , 0 , 0 , . . . ] .$ We vary one parameter at a time.

Mixing Ratio η. Fig. 4 shows the effect of η. $\mathrm { A t } \eta = 1 . 0$ (pure synthetic), airplane reaches ${ > } 9 5 \%$ by round 20. As η decreases, reference data increasingly stabilizes the distribution: at $\eta = 0 . 8 ,$ , the concentration is slightly mitigated; $\mathrm { a t } \eta = 0 . 5$ , airplane stabilizes around 50%; at $\eta = 0 . 2$ , it stays below 50% with other classes remaining above $\theta = 0 . 0 8 .$ .The preference vector $w _ { t }$ evolves similarly across all settings because $\beta = 0 . 0 1$ is small, confirming that the mode shift is driven by η rather than by changes in $w _ { t }$

Preference change rate $\beta .$ Fig. 5 shows the effect of β. $\operatorname { A t } \beta = 0 . 0 1 , w _ { t }$ changes slowly and the curation direction remains coherent, producing the strongest concentration $( > 9 5 \%$ airplane). $\mathrm { A t } \ \beta = 0 . 3$ , w<sub>t</sub> tracks the current distribution too closely, diluting the preference toward a uniform direction and weakening the curation signal. However, as mode concentration progresses, $\bar { \varphi } ( \boldsymbol { p } _ { t } )$ itself becomes dominated by the airplane component, which in turn pulls w<sub>t</sub> back toward $[ 1 , 0 , \dots ]$ . This restores the curation signal and drives further concentration, although it takes effect more slowly compared to small $\beta .$

Curation Temperature τ. Fig. 6 shows the effect of τ. ${ \bf A t } \ : \tau = 5 . 0 ,$ curation concentrates sharply: even small reward differences are amplified exponentially, driving airplane above 95% by round 20. $\mathbf { A t } \tau = 2 . 5 ,$ the concentration is slower. $\mathrm { { A t } } \ \tau = 1 . 0$ , the softmax weights are nearly uniform and the class distribution remains close to balanced. The speed of mode concentration increases monotonically with τ, consistent with the Lipschitz constants $L _ { p }$ growing rapidly in τ.

Selection Group Size K. Fig. 7 shows the effect of K. At $K = \infty$ and $K = 1 0 .$ , the concentration trajectories are nearly identical, with airplane exceeding 90% by round 20. At $K = 2 ,$ , each Luce group contains only two candidates, limiting the selector’s ability to distinguish high-reward samples. This is consistent with the discussion of the role of K.

## E.1.2 Reference data design

Setting A: Uniform thresholds. As described in Section 5, all four protected classes share the same threshold $\theta = 0 . 0 8$ with c = (1, 1, 11, 12, 13, 16, 30, 28, 27, 25). The feasibility boundary is at $\eta ^ { * } = 0 . 6 7$ . Since the full table is too long, we present Table 3 here to show the partial LP search results.

Table 3: Partial results of searching over 50 evenly spaced points in $\eta \in [ 0 . 2 , 0 . 9 ]$ . (Setting A: protect frog/horse/ship/truck, $\begin{array} { r } { \theta _ { \ell } = 0 . 0 8 . } \end{array}$ , c = (1, 1, 11, 12, 13, 16, 30, 28, 27, 25))
<table><tr><td>η</td><td>air</td><td>auto</td><td>bird</td><td>cat</td><td>deer</td><td>dog</td><td>frog</td><td>horse</td><td>ship</td><td>truck</td><td> $C ^ { * }$ </td><td>J</td><td>status</td></tr><tr><td>0.20</td><td>.000</td><td>.600</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.100</td><td>.100</td><td>.100</td><td>.100</td><td>11.59</td><td>46,378</td><td>√</td></tr><tr><td>0.30</td><td>.000</td><td>.543</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.114</td><td>.114</td><td>.114</td><td>.114</td><td>13.11</td><td>45,883</td><td>√</td></tr><tr><td>0.40</td><td>.000</td><td>.467</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.133</td><td>.133</td><td>.133</td><td>.133</td><td>15.13</td><td>45,388</td><td>√</td></tr><tr><td>0.50</td><td>.000</td><td>.360</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.160</td><td>.160</td><td>.160</td><td>.160</td><td>17.96</td><td>44,892</td><td>√</td></tr><tr><td>0.60</td><td>.000</td><td>.200</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.200</td><td>.200</td><td>.200</td><td>.200</td><td>22.20</td><td>44,397</td><td>1</td></tr><tr><td>0.67</td><td>.000</td><td>.026</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.243</td><td>.243</td><td>.243</td><td>.243</td><td>26.81</td><td>44,043</td><td> $\eta ^ { * }$ </td></tr><tr><td>0.69</td><td colspan="10"></td><td></td><td></td><td>x</td></tr><tr><td>0.90</td><td colspan="10">infeasible infeasible</td><td>一</td><td>一</td><td>x</td></tr></table>

Setting B: Heterogeneous thresholds. The four protected classes have different thresholds reflecting varying preservation priorities: $\theta _ { \mathrm { f r o g } } ~ = ~ 0 . 0 5$ $\theta _ { \mathrm { h o r s e } } ~ = ~ 0 . 0 8$ $\theta _ { \mathrm { s h i p } } ~ = ~ 0 . 1 0$ $\theta _ { \mathrm { t r u c k } } ~ = ~ 0 . 1 5$ . And the c = (1, 1, 11, 12, 13, 16, 30, 28, 27, 25) keep same.

Table 4: Partial results of searching over 50 evenly spaced points in $\eta \in [ 0 . 2 , 0 . 9 ]$ (Setting B: protect frog/horse/ship/truck ${ \ , } \theta = ( 0 . 0 5 ,$ , 0.08, 0.10, 0.15), c = (1, 1, 11, 12, 13, 16, 30, 28, 27, 25)).
<table><tr><td>η</td><td>air</td><td>auto</td><td>bird</td><td>cat</td><td>deer</td><td>dog</td><td>frog</td><td>horse</td><td>ship</td><td>truck</td><td>C*</td><td>J</td><td>status</td></tr><tr><td>0.20</td><td>.000</td><td>.525</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.062</td><td>.100</td><td>.125</td><td>.187</td><td>13.26</td><td>53,030</td><td>√</td></tr><tr><td>0.30</td><td>.000</td><td>.457</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.071</td><td>.114</td><td>.143</td><td>.214</td><td>15.01</td><td>52,534</td><td>√</td></tr><tr><td>0.40</td><td>.000</td><td>.367</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.083</td><td>.133</td><td>.167</td><td>.250</td><td>17.35</td><td>52,039</td><td>√</td></tr><tr><td>0.50</td><td>.000</td><td>.240</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.100</td><td>.160</td><td>.200</td><td>.300</td><td>20.62</td><td>51,544</td><td>√</td></tr><tr><td>0.60</td><td>.000</td><td>.050</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.125</td><td>.200</td><td>.250</td><td>.375</td><td>25.52</td><td>51,049</td><td>√</td></tr><tr><td>0.61</td><td>.000</td><td>.015</td><td>.000</td><td>.000</td><td>.000</td><td>.000</td><td>.130</td><td>.207</td><td>.259</td><td>.389</td><td>26.43</td><td>50,978</td><td> $\eta ^ { * }$ </td></tr><tr><td>0.63</td><td></td><td></td><td></td><td></td><td>infeasible infeasible</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>x</td></tr><tr><td>0.90</td><td colspan="9"></td><td>一 一</td><td></td><td>x</td></tr></table>

Table 4 shows the partial LP search results. The feasibility boundary is at $\eta ^ { * } = 0 . 6 1$ . The heterogeneous thresholds shift the feasibility boundary and change the optimal allocation: the LP assigns more reference mass to classes with higher thresholds (ship, truck) and less to those with lower thresholds (frog, horse). Table 5 compares our designed reference with uniform reference and the actual performance of the reference data is shown in the Fig. 8.

## E.1.3 Generated Samples

To complement the quantitative results, we provide generated samples at selected rounds in Fig. 9 and Fig. 10, offering a visual illustration of how the model’s output evolves under different reference data strategies.

Table 5: Comparison of reference data strategies. $C ^ { * }$ : per-sample cost; J: total cost $( 1 - \eta ) \cdot N \cdot C ^ { * }$ . Last four columns: proportion of each protected class at round 20.
<table><tr><td> $p _ { \mathrm { r e f } }$ </td><td>η</td><td> $C ^ { * }$ </td><td>J</td><td>frog</td><td>horse</td><td>ship</td><td>truck</td></tr><tr><td>Uniform</td><td>0.50</td><td>16.4</td><td>41,000</td><td>.067</td><td>.077</td><td>.059</td><td>.067</td></tr><tr><td>Uniform</td><td>0.20</td><td>16.4</td><td>65,600</td><td>.112</td><td>.113</td><td>.097</td><td>.105</td></tr><tr><td>Ours</td><td>0.61</td><td>26.43</td><td> $\mathbf { 5 0 , 9 7 8 }$ </td><td>.059</td><td>.101</td><td>.116</td><td>.168</td></tr></table>

![](images/f09db578aa59b2aff47c71b1608be478ab2ed2a476f98afbe209dd4d1661bc1a.jpg)  
Figure 8: Class proportions under designed reference.

![](images/18b131b58016cdc50e14e33c55fa7d89763e0eac8a3e3f6fe2990d34e4702795.jpg)  
Figure 9: Generated samples from OT-CFM under $w _ { 0 } = [ 1 , 0 , 0 , 0 , . . . ]$ (airplane preference), $\tau = 5 . 0 , \beta = 0 . 0 1$ . Each row corresponds to a different reference data strategy; each column shows samples at the indicated round. Under pure synthetic training $( \eta = 1 . 0 )$ , the model collapses to airplane images. Mixing with uniform reference data increasingly preserves diversity. The LPdesigned reference with $\bar { \theta } _ { \ell } = 0 . 0 8$ for last four classes (bottom row) maintains diversity at $\eta = 0 . 6 7$

## E.2 DDPM Experiments on CIFAR-10

Setup. The model is DDPM [17, 35] with the VLB objective (50M parameter UNet), initialized from the released pretrained checkpoint on CIFAR-10. The feature map $\varphi ( \dot { \boldsymbol { x } } ) \in \dot { \mathbb { R } } ^ { 1 0 }$ is the normalized softmax output of a pretrained ResNet-56 classifier (accuracy $\geq 9 3 \% )$ , where $\varphi ( x )$ represents the predicted probability of class c. Each round generates 20,000 candidate images, curates 10,000 via K-way Luce selection, mixes with real CIFAR-10 at ratio η to form a training set of size 10,000, and fine-tunes for 1,000 steps. After training, 10,000 fresh samples are generated to compute per-class proportions and update $w _ { t }$ . We run each experiment for 10 rounds with $\tau = 5 . 0$ $K = \infty ,$ , and $\bar { \beta } = \bar { 0 } . 0 1$

![](images/3674462a93bb809f60c677553d32d568a99bffbc0286f6576ffb766d4e20d26c.jpg)  
Figure 10: Generated samples from OT-CFM under $w _ { 0 } = [ 0 , 0 , 0 , 1 , . . . ]$ (cat preference), $\tau = 5 . 0 .$ $\beta = 0 . 0 1$ . Each row corresponds to a different reference data strategy; each column shows samples at the indicated round. Under pure synthetic training $( \eta = 1 . 0 )$ , the model collapses to airplane images. Mixing with uniform reference data increasingly preserves diversity.

Co-evolution dynamics. Fig. 11 shows results under two initializations and two mixing ratios. Under $\eta = 1 . 0 ,$ DDPM exhibits the same initialization-dependent lock-in as CFM: $w _ { 0 } = [ 1 , 0 , 0 , 0 , . . . ]$ drives concentration toward airplane and $\boldsymbol { w } _ { 0 } = [ 0 , 0 , 0 , 1 , 0 ]$ toward cats. Under $\eta = 0 . 1$ , both initializations converge to a similar near-uniform equilibrium, consistent with the uniqueness result. Compared to CFM, DDPM shows slower concentration under $\eta = 1 . 0$ , reaching 60% by round 10. This is expected: DDPM suffers from greater image quality degradation over successive retraining rounds, causing the classifier to assign less confident predictions. More broadly, the theoretical model assumes the retrained distribution exactly matches the training mixture, while in practice finite capacity and training error introduce deviations. Nevertheless, the qualitative predictions of the theory hold: lock-in occurs under pure synthetic training and stabilization occurs under mixture with reference data.

![](images/0196f68ff6f452d020c241d046569ec5e1748c2e8a8a895d340c1f19782f56b5.jpg)  
Figure 11: Co-evolution dynamics under different initializations and mixing ratios for DDPM model. Top: per-class generation proportions. Bottom: preference vector $w _ { t }$

Reproducibility across seeds. To verify that the observed dynamics are not artifacts of a particular random seed, we repeat the experiment $( \eta = 1 . 0 , w _ { 0 } = [ 1 , 0 , 0 , 0 , . . . ] )$ with three independent seeds. Fig. 12 shows that all three runs produce consistent trajectories: airplane proportion reaches 58.29%-59.02% by round 10 and the preference vector follows a similar path. The small variation across seeds confirms that the co-evolution dynamics are driven by the systematic interaction between curation and retraining, rather than by stochastic fluctuations.

Generated Samples. Fig. 13 shows the corresponding generated samples for DDPM. Compared to OT-CFM, DDPM exhibits noticeable image quality degradation over successive rounds: textures become blurry and fine details are lost, reflecting the accumulation of training error across generations. This degradation is most pronounced under $\eta = 1 . 0 \ : \AA$ , where the model retrains entirely on its own imperfect outputs. Nevertheless, the same structural patterns persist: concentration toward a single class under pure synthetic training, and preserved diversity under reference data mixing.

![](images/98bf7ad0eba48187e94687281ef80b05f7d9e6cde574da015532074e473b4a19.jpg)

![](images/90af2fa7734f1fe5a7a1b24fe460d629d0249c5cc3bb09bd0764517f7e485137.jpg)

![](images/f14d1c47b329b679abc6904b0569f6a37ac67fa1b53a5de22bd4eb83428ea38d.jpg)  
Figure 12: Reproducibility of DDPM co-evolution across three independent seeds $( \eta = 1 . 0 , w _ { 0 } =$ $[ 1 , 0 , 0 , 0 , . . . ] , \tau = 5 . 0 , \dot { K } = \infty , \beta = 0 . 0 1 )$ ).

![](images/5845a63a552406f268256460fd79a2b0b0cf0ad7684956a115f2ecaa46d2caa2.jpg)  
Figure 13: Generated samples from DDPM under $w _ { 0 } = [ 1 , 0 , 0 , 0 , . . . ]$ (airplane preference), $\tau = 5 . 0 .$ $\beta = 0 . 0 1$ . Each row corresponds to a different reference data strategy; each column shows samples at the indicated round. Under pure synthetic training $( \eta = 1 . 0 )$ , the model collapses to airplane images. Mixing with uniform reference data increasingly preserves diversity.

## E.3 GPT-2 Experiments on AG News

We replicate the co-evolution experiments on text generation.

Setup. The model used is GPT-2-small (124M parameters) [37], initialized from the HuggingFace pretrained checkpoint. The feature map $\varphi ( x ) \in { \mathsf { \bar { \mathbb { R } } } } ^ { 4 }$ is the normalized softmax output of a pretrained BERT [8] classifier (accuracy $\ge ~ 9 4 \% ) .$ , where $\varphi ( x )$ represents the predicted probability of class $c \in$ {world, sports, business, sciTech}.

Iterative retraining. Each round generates 5,000 candidate texts, curates 2,500 via K-way Luce selection, mixes with real AG News at ratio η to form a training set of size 2,500, and fine-tunes for 1,000 gradient steps. After training, 2,000 fresh samples are generated to compute per-class proportions and update w . We run each experiment for 10 rounds with $\tau = 5 . 0 , K = \infty ,$ and $\beta = 0 . 0 1$

Co-evolution dynamics. Fig. 14 shows results with $\displaystyle w _ { 0 } ~ = ~ [ 0 , 1 , 0 , 0 ]$ (Sports) under four mixing ratios. $\mathrm { A t } \eta = 1 . 0$ , the Sports proportion exceeds 90% by round 10, mirroring the lock-in observed in the image experiments. As η decreases, reference data stabilizes the distribution. Fig.15 shows the same sweep with $\boldsymbol { w } _ { 0 } = [ 0 , 0 , 0 , 1 ]$ (SciTech). The dynamics are qualitatively identical: the system locks into SciTech under $\eta = 1 .$ .0 and stabilizes under small η. This confirms that the initialization-dependent lock-in and the stabilizing effect of reference data hold across both modalities and class structures.

Unlike the image models, GPT-2 deviates more noticeably from the idealized assumptions of our theory. Even when trained entirely on uniformly distributed reference data $( \eta = 0 . 0 )$ , the generated class proportions are not uniform, indicating that GPT-2 is far from a perfect learner: the retrained distribution does not exactly match the training mixture. In addition, the class proportions fluctuate across rounds rather than converging smoothly, likely due to the instability of fine-tuning a pretrained language model on small datasets. Despite these deviations, the qualitative theoretical predictions remain: lock-in under pure synthetic training and stabilization under reference data mixing.

![](images/81999ad48bec244871a255055be54e6c9a78f7a8b2f83809a619e49690b58115.jpg)

![](images/2fbd6a6632fc042f31cf88507f84cedf31981abaeda49fe3939e4b8e9c1037f5.jpg)

![](images/0c3dc53ed05d563525bd11c8c1d3156f82a3f08b43bc3a53ee68166981537429.jpg)

![](images/d427272e0271ad2e0bba9f16720bdb631573d398033b108a54eba1a73659eb44.jpg)

Figure 14: GPT-2 on AG News with $\boldsymbol { w } _ { 0 } = [ 0 , 1 , 0 , 0 ]$ (Sports). Top: per-class proportions. Bottom: preference vector w<sub>t</sub>.  
![](images/62e1e7b68ed64b3c94e0c9c5f8f62ce733531188ac47c463210531dc7d9a8bf6.jpg)

![](images/9663c4ceb976493afd478eaea2bd99a9e778ef26de6eab8dc55efb9cd7bca953.jpg)

![](images/ff59fa63bdf6573c8967563b1658b6c38c06ddcb4f8db6448fb05f5dc6686c04.jpg)

![](images/93026588d005a39d847ec04038007976da835c48c3a2550d3edd752a82af505d.jpg)  
Figure 15: GPT-2 on AG News with $w _ { 0 } = [ 0 , 0 , 0 , 1 ]$ (SciTech). Top: per-class proportions. Bottom: preference vector $w _ { t }$

Reference data design. We apply Alg. 1 to the AG News setting. We protect the class world with threshold $\theta = 0 . 1 5$ . To account for the gap between the theoretical model and GPT-2’s imperfect learning, we calibrate the threshold used in the LP: since GPT-2 achieves roughly 37.5% of the theoretically predicted level under uniform reference data, we scale the LP input threshold to $\begin{array} { r } { \dot { \theta } _ { L P } \approx \frac { 0 . 1 5 } { 0 \ 3 7 5 } = 0 . 4 } \end{array}$ , so that the actual achieved proportion approximates the desired $\theta = 0 . 1 5$ . The costs are $c = ( 2 0 , 5 , 5 , 5 )$ , reflecting that World news requires more effort to source than the other categories.

Table. 6 compares our designed reference with uniform reference and the actual performance of the reference data is shown in the Fig. 16.

Table 6: Comparison of reference data strategies. $C ^ { * }$ : per-sample cost; J: total cost $( 1 - \eta ) \cdot N \cdot C ^ { * }$ . Last columns: proportion of the protected class at round 20.
<table><tr><td>pref</td><td>η</td><td> $C ^ { * }$ </td><td>J</td><td>world</td></tr><tr><td>Uniform</td><td>0.50</td><td>8.75</td><td>10,937.50</td><td>.1211</td></tr><tr><td>Uniform</td><td>0.10</td><td>8.75</td><td>19,687.50</td><td>0.12</td></tr><tr><td>Ours</td><td>0.10</td><td>11.67</td><td>26,250.00</td><td>.1404</td></tr></table>

![](images/e0682f52a2cfa53bf9e3788fc219415e45e7dd3a938e33eb8444244331c47842.jpg)  
Figure 16: Class proportions under designed reference.

## F Discussion and Extensions

Our framework focuses on a basic co-evolutionary setting in which a single aggregate preference state interacts with a self-consuming generative model through preference-guided curation. Building on this framework, several natural extensions arise by enriching the preference dynamics and population structure while retaining the same closed-loop interaction between user feedback and generative retraining. We discuss three such directions below.

## F.1 Nonlinear Preference Responses

The preference update in Eq. 4 moves the current preference toward the mean feature vector of the updated model through linear averaging. This form is related to first-order models of preference and opinion formation. More generally, users may respond nonlinearly to the discrepancy between their current preference and the content to which they are exposed. We consider such responses while retaining the reward $\hat { r _ { w } } ( x ) = \langle w , \varphi ( x ) \rangle$ ⟩, the curation operator, and the model retraining step in Eq. 3.

A gap-responseformulation. Let $\Delta _ { t } : = \bar { \varphi } ( p _ { t + 1 } ) - w _ { t }$ denote the mean feature vector after retraining and its discrepancy from the current preference. For a response function $F : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , define

$$
w _ { t + 1 } = G _ { F } ( w _ { t } , \bar { \varphi } ( p _ { t + 1 } ) ) : = \mathrm { P r o j } _ { W } [ w _ { t } + \beta F ( \bar { \varphi } ( p _ { t + 1 } ) - w _ { t } ) ] .\tag{14}
$$

The linear update is recovered by $F ( \Delta ) = \Delta$ . We additionally consider a saturating response and a statedependent scalar gate.

Saturating response. For $F ( \Delta ) = \operatorname { t a n h } ( g \Delta )$ with $g > 0$ and tanh applied componentwise, the update is

$$
w _ { t + 1 } = \operatorname { P r o j } _ { W } \left[ w _ { t } + \beta \operatorname { t a n h } \left( g \left( { \bar { \varphi } } ( p _ { t + 1 } ) - w _ { t } \right) \right) \right]\tag{15}
$$

This response is smooth and saturates as the feature–preference gap grows, limiting the magnitude of the response in each coordinate.

Scalar-gated response. Let $\gamma _ { t } : = \gamma \big ( w _ { t } , \bar { \varphi } ( p _ { t + 1 } ) \big ) \ \in \ ( 0 , 1 ]$ be a state-dependent response weight. Using $F _ { t } ( \Delta _ { t } ) = \gamma _ { t } \Delta _ { i }$ in the same update template gives

$$
w _ { t + 1 } = \mathrm { P r o j } _ { W } [ ( 1 - \beta \gamma _ { t } ) w _ { t } + \beta \gamma _ { t } \bar { \varphi } ( p _ { t + 1 } ) ] .\tag{16}
$$

The gate acts as a similarity or confidence weight. It changes the adaptation rate according to the current interaction between preferences and model outputs. In the discussion below, $G _ { F }$ denotes the resulting preference update map, including this state-dependent gated form.

Components of the analysis that remain unchanged. Changing the preference response leaves the modelupdate branch unchanged. Consequently, the curation properties in Prop. 3.1, the retraining sensitivity constants $L _ { p }$ and $L _ { w }$ in Lem $\mathrm { D . 7 }$ , and the equal-reward characterization of fixed points of the retraining step at a given preference in Thm. 3.2 continue to apply. These components depend on the reward and feature representation, rather than on the rule used to update preferences.

The singleton equilibria also remain fixed whenever the response vanishes at a zero gap. Indeed, at $p ^ { \star } = \delta _ { x _ { i } } ,$ $\boldsymbol { w } ^ { \star } = \varphi ( \boldsymbol { x } _ { i } )$ , we have $\bar { \varphi } ( p ^ { \star } ) - w ^ { \star } = 0$ . Thus $F ( 0 ) = 0$ implies that the preference update leaves $w ^ { \star }$ unchanged, while the model update is already fixed in the purely synthetic regime. All three response families above vanish at a zero gap.

Response-dependent sensitivity and coupled dynamics. The response function enters the joint analysis through the preference coordinate. Suppose that the resulting map satisfies

$$
\left\| G _ { F } ( w , m ) - G _ { F } ( v , \widetilde { m } ) \right\| \le a _ { F } \| w - v \| + b _ { F } \| m - \widetilde { m } \|
$$

for admissible preference vectors and mean feature vectors, with $a _ { F } , b _ { F } \ge 0$ . For two trajectories $( p _ { t } , w _ { t } )$ and $\left( q _ { t } , v _ { t } \right)$ , the preference-coordinate estimate becomes

$$
\| w _ { t + 1 } - v _ { t + 1 } \| \leq a _ { F } \| w _ { t } - v _ { t } \| + b _ { F } \Big \| \bar { \varphi } ( p _ { t + 1 } ) - \bar { \varphi } ( q _ { t + 1 } ) \Big \|
$$

For the linear response, nonexpansiveness of projection gives $( a _ { F } , b _ { F } ) = ( 1 - \beta , \beta )$ , recovering the estimate used in the proof of Thm. 3.9. For another response, the same proof strategy replaces $( 1 - \beta , \beta )$ with $( a _ { F } , b _ { F } )$ and combines the corresponding preference-coordinate estimate with the unchanged retraining bounds. The admissible weight in the product metric, the contraction condition, and the convergence rate consequently depend on the response. Quantitative lock-in neighborhoods likewise require response-specific control of the preference update. This formulation identifies the part of the analysis that must be adapted, while preserving the curation and retraining components. Deriving explicit bounds for particular nonlinear responses is a direction for further analysis.

## F.2 Stochastic Preference Adaptation

Another natural extension is to model preference changes at the individual level as stochastic rather than deterministic. For example, suppose a user updates toward the feature vector of a randomly observed model output, $X _ { t + 1 } \sim p _ { t + 1 } , w _ { t + 1 } = \bar { \mathrm { P r o j } } _ { W } [ ( 1 - \bar { \beta } ) w _ { t } + \beta \varphi ( X _ { t + 1 } ) ]$ . Under the assumptions $\lvert | w _ { t } \rvert | \leq 1 , \lvert | \varphi ( x ) \rvert | \stackrel {  } { = } 1$ and $\beta \in [ 0 , 1 ]$ , the point before projection already lies in W. Therefore,

$$
\mathbb { E } \left[ w _ { t + 1 } ~ \middle | ~ p _ { t + 1 } , w _ { t } \right] = ( 1 - \beta ) w _ { t } + \beta \bar { \varphi } ( p _ { t + 1 } ) ,
$$

which recovers Eq. 4 as the conditional-mean dynamics. This provides a mean-field interpretation of the deterministic preference update studied in the paper.

The stochastic formulation raises additional questions that do not arise at the mean level. In particular, one may ask whether stochastic trajectories concentrate around the deterministic dynamics, how random exposure affects the probability of entering different lock-in basins under pure self-consumption, and whether reference mixing induces stochastic stability or a concentrated stationary distribution around the deterministic equilibrium. More general noise models, including exogenous perturbations to the preference state, would further require quantifying how random fluctuations propagate through the coupled model–preference feedback loop.

## F.3 Heterogeneous User Populations

The current framework represents the population by a single aggregate preference vector. A natural extension is to consider G user groups with group-specific preferences $\bar { \{ w _ { g , t } \} } _ { g = 1 } ^ { G }$ and population weights $\{ \alpha _ { g } \} _ { g = 1 } ^ { G } ,$ where $\alpha _ { g } \geq 0$ and $\textstyle \sum _ { g = 1 } ^ { G } \alpha _ { g } = 1$ . One possible model replaces the single curated distribution by a mixture of group-specific curation responses,

$$
p _ { t + 1 } ( x ) = ( 1 - \eta ) p _ { \mathrm { r e f } } ( x ) + \eta p _ { t } ( x ) \sum _ { g = 1 } ^ { G } \alpha _ { g } H _ { p _ { t } , w _ { g , t } } ^ { K } ( x ) ,
$$

while each group updates according to

$$
w _ { g , t + 1 } = \mathrm { P r o j } _ { W } \left[ ( 1 - \beta _ { g } ) w _ { g , t } + \beta _ { g } \bar { \varphi } ( p _ { t + 1 } ) \right] .
$$

This extension introduces qualitatively new interactions because all groups affect, and are simultaneously affected by, the same evolving generative distribution. It therefore raises questions about whether group preferences converge or remain separated, how the feedback of one group indirectly changes the preferences of others through the shared model, and how reference-data design should be adapted when different groups impose distinct preservation objectives. Characterizing the equilibria and stability of such multi-group dynamics, as well as designing reference distributions that balance group-specific objectives, is a promising direction for future work.

## G Broader impacts

This work studies feedback loops in self-consuming generative systems, where model outputs are curated by users and reused for future training, while user preferences also evolve over time. This work can help diagnose and mitigate long-term instability, preference drift, and diversity collapse by our theoretical and empirical tools. However, the same steering mechanism is value-neutral with respect to the choice of preserved attributes, and could in principle be used to filter out specific content. Thus, transparent selection of preserved attributes is important in any deployment.