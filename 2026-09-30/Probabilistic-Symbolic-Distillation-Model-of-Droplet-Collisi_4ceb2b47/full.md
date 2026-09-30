# Probabilistic Symbolic-Distillation Model of Droplet Collision for Spray Simulation at High Ambient Pressures

Weiming Xu<sup>1,2#</sup>, Tao Yang<sup>1,2#</sup>, and Peng Zhang<sup>1,2</sup>

1. Department of Mechanical Engineering, City University of Hong Kong, Kowloon Tong, Kowloon,   
999077, Hong Kong   
2. Shenzhen Research Institute, City University of Hong Kong, Shenzhen, 518057, P. R. China

## Abstract

Droplet collision governs droplet population dynamics in many chemical engineering processes, such as spray drying, spray cooling, agricultural spraying, and combustion. Existing analytical models impose deterministic, pairwise boundaries between collision outcomes, whereas machine-learning classifiers lack the explicit functional form required of analytical collision submodels. In this study, we develop a probabilistic symbolic-distillation model using nearly forty thousand experimental events spanning eight regimes and five dimensionless parameters, including over five thousand data for ambient pressure up to 50 atm. A machine-learning teacher learns the joint outcome-probability landscape from these data, and symbolic regression subsequently distils it into eight class-specific expressions that jointly define a coupled analytical model. The resulting analytical field replaces abrupt regime switching with finite-width fuzzy boundaries. It outperforms the evaluated conventional analytical boundary models and reveals that their main limitation is the inability of zero-width boundaries to represent gradual probability transitions. The “biased-dice” sampling scheme provides a statistically consistent and practically convenient model implementation for Eulerian-Lagrangian spray simulation.

Keywords: Droplet collision, Spray simulation, Data-driven learning, Probabilistic modelling, Symbolic regression

## 1. Introduction

Sprays are central to a wide range of chemical and process engineering operations [1, 2], such as drying, cooling, agricultural spraying, and engine combustion. The droplet size distribution determines the process performance [2, 3], but the distribution is not fixed at atomisation. In dense spray regions, high droplet number density causes continuous collisions that redistribute mass and momentum among the droplets. Coalescence, separation after temporal coalescence, and splashing are the collision outcomes that affect droplet size distribution [2]. In the Eulerian-Lagrangian spray simulation, droplets are treated as discrete particles, and their collision partners are identified either deterministically or by stochastic sampling. The outcome of each identified collision is then set by a submodel [4-8]. Previous studies show that the choice of this outcome submodel alters the predicted spray evolution [3, 5, 9].

Binary droplet collisions are conventionally described by four dimensionless groups: Weber number (��), impact parameter (�), size ratio (�), and Ohnesorge number (�ℎ). Outcomes are typically mapped on a �� − � nomogram containing four regimes: bouncing, coalescence, reflexive separation, and stretching separation, with the boundaries between them shifting with � and �ℎ [10-13]. This four-regime division reflects the conditions accessible to early experiments. As experimental measurements extended in Weber number, size ratio, and the range of liquids examined, further outcomes were resolved: splashing and intense fragmentation at high �� [14, 15], rotational separation following temporary coalescence [16], and finger separation [11, 14].

Ambient gas pressure acts as another important parameter. It governs the drainage of the interstitial gas film between the droplets, which dictates whether their surfaces make contact. In general, higher pressure slows film drainage and promotes droplet bouncing, raising the critical Weber number required for coalescence [12, 17]. A recent experiment extending gauge pressure to 50 atm shows that the critical Weber number rises nonlinearly with pressure and then levels off above approximately 20 atm [18]. Therefore, existing collision models cannot predict the collision outcomes at high pressures and are applicable only within the parameter range covered by the data on which they were built.

In Eulerian-Lagrangian spray simulation, the treatment of droplet collision separates into three successive operations [2, 19-21]: detection of collision partners within the Lagrangian tracking loop, determination of the collision outcome from the local values of the governing parameters, and updating of the resulting droplet population in number, size, and velocity. The present work focuses on the second of these. The collision outcome model has conventionally been formulated as a composite of analytical criteria [20, 22]. Each criterion is a closed-form expression delimiting one pair of neighbouring outcomes in the �� − � plane: the lower bouncing boundary (LB), the reflexive separation–coalescence boundary (RS-C), and the stretching separation–coalescence boundary (SS-C), as shown in Fig. 1(a). The transition between soft coalescence and bouncing at very low Weber number is often ignored [21].

A composite model is assembled by selecting a formula for each transition boundary. By applying these formulas sequentially, every collision event is mapped to a specific region on the graph. For example, the construction of Munnannur and Reitz [23], combining the criteria of Estrade et al. [24], Ashgriz and Poo [10], and Brazier-Smith et al. [25], was implemented in KIVA [26]. In this way, evaluating this composite model based on a single aggregate accuracy can be misleading for two reasons. First, the transition criteria built for the model are independent. The overall performance is inherently constrained by its least accurate criterion. Second, an aggregate score averages all criteria together, weighted by how many data points fall in each regime. The same model tested on differently distributed data would score differently.

Some analytical criteria originate in energy analyses of the collision process. Specific boundaries were established by comparing kinetic and surface energies [10], rotational and surface energies [25], or by assessing the kinetic energy required to expel the interstitial gas film [24]. While physically intuitive, these derivations are inherently restricted, as they rely on highly idealized assumptions (e.g., inviscid flow, prescribed geometries) and were calibrated against narrow, atmospheric-pressure datasets. As broader measurements accumulated, subsequent models evolved primarily by empirically relaxing these initial restrictions. Researchers progressively incorporated viscous effects (�ℎ) and size-ratio (∆) dependencies into the classical boundaries for bouncing, reflexive, and stretching separation (e.g., Qian and Law [12], Gotaas et al. [27], Sommerfeld and Kuschel [13], and Sui et al. [28]). However, new data were fitted to extend the existing criteria rather than to revise them. Consequently, the validated parameter space grows incrementally, while ambient pressure and the non-classical outcomes have not entered these criteria, despite being well observed experimentally. Published evaluations demonstrate this predictive shortfall. For example, Agarwal et al. [29] tested the Munnannur and Reitz model, finding that its accuracy fell from 64% on early datasets to 43% on more recent ones [13]. These results indicate that empirical models are reliable only when the test conditions resemble the data they were calibrated on.

Alternatively, data-driven modeling infers collision outcomes directly from measurements. Agarwal [30] showed that machine-learning classifiers trained on 7898 experimental points exceeded 90% accuracy, significantly outperforming traditional physics-based models. Very recently, Yu and Chang [21] extended the training set to 30,809 points and coupled the machine-learning model with analytical criteria to revise conflicting data labels. While this approach achieved 94% training accuracy, performance fell to 82.2% on untrained data, which they attributed to measurement uncertainty in the collected experimental data.

Previous models, both analytical and data-driven, share a deterministic structure: each outputs a single regime label. Xu et al. [31] broke from this convention by treating collision outcomes as a probability distribution. Using a database of 33540 points, they trained a LightGBM (Light Gradient-Boosting Machine) [32] classifier across eight regimes using five parameters, including ambient pressure (up to 9 atm). Consequently, collision conditions are characterized by the relative likelihood of each outcome, allowing transitional regions to appear as continuous probability gradients rather than rigid boundaries. Because the tree-based classifier cannot be inspected or used outside the trained model, the probability field was projected onto a multinomial logistic regression in a second-order polynomial basis of the five parameters to recover an explicit form. A multinomial draw over the predicted probabilities then supplies the single definite outcome that Lagrangian tracking requires at each event. Performance was reported as aggregate accuracy, recall, and specificity over the full database.

For the droplet collision modelling, analytical criteria are limited by the data they were built on, while data-driven models are limited by their lack of interpretability. First, an analytical model is bound by its original assumptions and calibration range. Outside that range, it still gives a result, with no indication that its assumptions no longer hold. It works only in the regimes it was built to distinguish, and any other outcome is misclassified. It always represents boundaries as deterministic lines, while experiments show a probabilistic band. Second, current data-driven approaches capture these probabilistic transitions and provide usable formulas, but not interpretable ones. Their functional form is chosen based on simplified polynomials and only the coefficients are learned from the data, so the expression predicts well but reveals nothing about how the parameters act.

These two gaps converge on a single issue: how these models are evaluated. A composite model is made of separate criteria, one for each boundary. A new model should therefore be compared with it at each boundary separately, not by one overall score. Current data-driven models, however, report only a single accuracy over the whole database. Most of the data lie far from any boundary, so a model can score well overall while still getting the boundaries wrong. So far, no work has compared analytical criteria and data-driven models boundary by boundary, so which performs better there is still unknown. To address these gaps, we proposed a probabilistic symbolic-distillation model. The machine-learning model is not treated as the final predictor but as an intermediate representation from which analytical criteria are recovered. This separates two things that have so far been coupled: the model that fits the data need not be the model that is used, and the form of the criterion need not be assumed in advance.

A LightGBM teacher is trained on 38,762 experimental collision events spanning eight regimes and five parameters, with ambient pressure extending to 50 atm, a new record made very recently [18], and returns a probability distribution over all candidate outcomes rather than a single label, so that transitional bands are preserved as continuous variations rather than resolved into a boundary. Symbolic regression then distils this field into compact closed-form probability functions whose structure is found rather than assumed. Setting two of them equal yields an explicit transition boundary that can be written down and compared with the corresponding energy-based criterion, which is what separates a distilled criterion from a statistical surrogate, while a multinomial draw over the same expressions supplies the single definite outcome that Lagrangian tracking requires.

## 2. Machine-Learning Methodology

## 2.1 Experimental Database

The current database expands Yu and Chang [21] and our recent work [31] to 38,762 collision events by including the high-pressure measurements of Zhang et al. [18]. Each event is classified into one of eight collision regimes and characterized by five dimensionless parameters spanning broad ranges: $W e \ = \ 0 { \sim } 2 0 0 0 , \ B \ = \ 0 { \sim } 1$ 2 $\varDelta \ = \ 1 { \sim } 5$ ， ${ O h } = 9 . 5 \times 1 0 ^ { - 4 } { \sim } 5 . 5 \times 1 0 ^ { - 1 }$ , and $P \ = p / p _ { 0 } = \ 0 . 6 - 5 0$ , where $p _ { 0 }$ is the atmospheric pressure. The addition of over 5,000 high-pressure events is physically significant. Unlike earlier databases mostly restricted to ambient conditions, our previous database extended only to 9 atm but captured only the monotonic portion of the bouncing–coalescence transition. Consequently, models trained on such lowpressure data inherently over-extrapolate this trend, overestimating the critical Weber number at elevated pressures. The newly added data specifically span the pressure plateau [18].

Figure 1(a) displays the eight regimes in the ��–� plane with the LB, SS-C, and RS-C boundaries, while Fig. 1(b) shows their distributions in the present dataset: I, soft coalescence (274 points); II, bouncing (11,122 points); III, hard coalescence (14,715 points); IV, reflexive separation (2,879 points); V, stretching separation (9,175 points); VI, rotational separation (259 points); VII, finger separation (83 points); and VIII, splashing (255 points). Notably, the regimes are strongly imbalanced. Regimes II, III, and V comprise most of the data (up to 90%), while VI, VII, and VIII total under 600 events. This imbalance results from the measurement limits, instead of sampling bias. This inherent imbalance directly necessitates the specific learning algorithms and evaluation metrics, as adopted below.

![](images/622b7e01f0e4b7e1c2f5e9fdfb58772989df5f9ae994617fe81abb160c0611dd.jpg)  
Fig. 1. The eight collision regimes in the $W e - B$ plane. (a) Schematic nomogram showing the regimes and the three classical boundaries: the lower bouncing boundary (LB), the stretching separation–coalescence boundary (SS-C), and the reflexive separation–coalescence boundary (RS-C). (b) Distribution of the experimental data for each regime: soft coalescence after minor deformation (Regime I), bouncing (Regime II), hard coalescence after substantial deformation (Regime III), reflexive separation (Regime IV), stretching separation (Regime V), rotational separation (Regime VI), finger separation (Regime VII), and splashing (Regime VIII). The data point number in each regime is shown.

Figure 2 shows the pairwise distribution of the five parameters. Three features of the coverage bear on the modelling. First, the distribution of $W e$ is multimodal, each mode corresponding to the range over which a group of regimes is observed. The parameters are therefore normalised before training. Second, the data of �ℎ, �, and � fall into discrete bands rather than filling the space, each band being one liquid, one size ratio, or one chamber pressure from a single study. The parameter space is thus sampled at points rather than continuously, and any model must interpolate between the bands. Third, � is concentrated near unity, as most reported experiments use equal or near-equal droplets. The collision events with $\varDelta > 2$ are rare throughout. The elevated-pressure data are confined to regimes I–V, so the behaviour of regimes VI– VIII at $P > 1$ rests entirely on extrapolation.

![](images/19f63d5d3398fd5a5a5bdd71e7ffa7eb9a88f007f46cbca472950d9ef122bf26.jpg)  
Fig. 2. Pairwise distribution of the five dimensionless parameters across the eight collision regimes. These off-diagonal panels show the joint distribution of each pair, while the diagonal panels show the kernel density estimate for each parameter (histogram for �). Colours denote the different regimes. $W e .$ , �ℎ, and � are plotted on logarithmic axes and all parameters are scaled to [0, 1].

These dataset characteristics dictate our modeling choices. First, because regime populations differ by up to three orders of magnitude, the learning algorithm must remain sensitive to rare outcomes represented by mere dozens of events. Consequently, model performance cannot be reliably evaluated via global aggregate accuracy. Second, since the parameter space is sampled in discrete bands rather than continuously, the model must handle sparse and irregular coverage robustly. Third, because nominally identical experiments within transitional bands often yield different outcomes, the learning target must be a probability distribution over candidate regimes rather than a single deterministic label.

The analytical distillation stage faces additional constraints. Specifically, the pressure dependence of the bouncing transition features a nonlinear saturation, and �� spans seven orders of magnitude. A pre-fixed polynomial basis cannot capture this saturation, and assuming a mathematical structure in advance risks enforcing behaviors unsupported by the data. Therefore, the optimal functional form must be discovered directly from the data.

## 2.2 Overview of Probability Symbolic-distillation Method

The probabilistic symbolic-distillation framework is summarized in Figure 3. Each collision event in the database shown in Fig. 3(a) is described by the input vector $x =$ $( P , W e , B , \varDelta , O h )$ and its observed regime. These data are used to train the ensemble LightGBM model in Fig. 3(b), which estimates the normalized eight-regime probability vector $\pmb { p } ^ { T } ( \pmb { x } )$ . The resulting probability fields assign comparable probabilities to neighbouring outcomes within model-inferred transition regions, while identifying a dominant outcome in well-separated regime interiors. These continuous probability fields subsequently serve as the targets for the symbolic-regression model in Fig. 3(c), using the regime-specific fitting strategies described in Section 2.4. The resulting expressions produce eight bounded class scores, which are aligned through classspecific affine transformations and jointly normalised by a softmax function to obtain the analytical probability vector $\pmb { p } _ { k } ^ { S R } ( \pmb { x } )$ . The expressions therefore operate as a coupled multiclass model rather than as independent binary transition criteria. Finally, when a simulation requires one outcome for an individual collision, the realised regime is drawn from the categorical distribution $Y { \sim } C a t [ p _ { k } ^ { S R } ( { \pmb x } ) ]$ . This operation returns a single regime for each event while preserving the predicted probabilities in expectation. Across an ensemble of collisions, the resulting regime counts follow a multinomial distribution. The method thus connects the learned probability landscape, its explicit analytical representation, and the event-level outcome realization required for Eulerian-Lagrangian spray simulations.

![](images/5c95c9e30493f16070e9083c0b6a47b42dd20aa101729ca56c716e2f1a8ff1e9.jpg)  
Fig. 3. Schematic of the probabilistic symbolic-distillation framework for droplet collision prediction, including (a) database input, (b) LightGBM machine learning model, (c) symbolic regression model, and (d) explicit analytical probability model.

## 2.3 LightGBM Teacher Model for Eight-regime Probability Learning

A reliable probabilistic teacher model is first constructed to approximate the regime distribution in parameter space with high accuracy and stability. In this framework, LightGBM is adopted, and droplet-collision prediction is formulated as an eight-regime probabilistic learning problem [32]. Each collision event is represented by the descriptor vector $\pmb { x _ { i } } = ( P _ { i } , W e _ { i } , B _ { i } , \varDelta _ { i } , O h _ { i } )$ , and the observed outcome $y _ { i } \in$ $\left\{ I , I I , \ldots , V I I I \right\}$ provides the training label. The teacher is thus trained to return an eight-component probability vector, thereby enabling a continuous representation of regime transitions rather than a single discrete assignment.

This algorithm is a histogram‑based gradient‑boosted decision‑tree method well‑suited for capturing nonlinear and conditional interactions among the collision descriptors. It was constructed as an ensemble of independently trained LightGBM models to reduce its sensitivity to the particular training sample. Each model was fitted to a separately resampled subset $\mathcal { D } _ { m }$ , containing 80% of the training events. For the �-th LightGBM estimator, the score assigned to regime � after � boosting iterations is written as

$$
\pmb { z } _ { i , k } ^ { ( m ) } = \pmb { z } _ { i , k } ^ { ( m , 0 ) } + \eta \sum _ { t = 1 } ^ { T } \pmb { h } _ { k , t } ^ { ( m ) } ( \pmb { x } _ { i } )\tag{1}
$$

where $\pmb { z } _ { i , k } ^ { ( m , 0 ) }$ is the initial score for regime �, $\pmb { h } _ { k , t } ^ { ( m ) }$ is the contribution of the tree added at iteration $t ,$ and � is the learning rate. At every iteration, candidate splits are evaluated from the first- and second-order derivatives of the multiclass loss. The continuous predictors are binned into histograms, and the leaf with the largest loss reduction is expanded. This procedure permits the model to learn localized, conditional changes in regime occurrence without imposing a prescribed analytical switching boundary.

The accumulated class scores $\pmb { z } _ { i , k } ^ { ( m ) }$ quantify the relative support for the eight collision regimes rather than their individual occurrence probabilities. Because the regimes constitute mutually exclusive outcomes, each score must be evaluated relative to those of the competing regimes. The scores are therefore jointly normalised through the softmax function

$$
\pmb { q } _ { i , k } ^ { ( m ) } = \frac { e ^ { \pmb { z } _ { i , k } ^ { ( m ) } } } { \sum _ { j = 1 } ^ { 8 } e ^ { \pmb { z } _ { i , k } ^ { ( m ) } } } , k = \mathrm { ~ I ~ } , \cdots , \mathbb { M } , \sum _ { k = 1 } ^ { 8 } \pmb { q } _ { i , k } ^ { ( m ) } = 1\tag{2}
$$

where $0 < \pmb { q } _ { i , k } ^ { ( m ) } < 1$ is the probability assigned to Regime k.

For each ensemble member, the tree structures and leaf values were determined by minimizing the class-weighted multiclass cross-entropy loss

$$
\mathcal { L } ^ { ( m ) } = - \frac { 1 } { N _ { m } } \sum _ { i \in \mathcal { D } _ { m } } w _ { y _ { i } } \mathrm { l o g } \pmb { q } _ { i , y _ { i } } ^ { ( m ) }\tag{3}
$$

where $N _ { m }$ is the number of collision events. The weight assigned to Regime � is $\begin{array} { r } { w _ { k } = \frac { N } { K n _ { k } } } \end{array}$ , where $n _ { k }$ is the number of events belonging to that regime in the development database. This weighting prevents the cross-entropy loss from being dominated by bouncing, hard coalescence, and stretching separation, while retaining the sparsely represented outcomes during teacher learning. The teacher consisted of $M = 1 0$ LightGBM estimators fitted to resampled subsets containing 80% of the training data. Each estimator used 200 boosting iterations, 31 leaves, a learning rate of 0.2, and a maximum tree depth of 10. The ensemble probability was obtained by averaging the outputs of the ten estimators,

$$
p _ { i , k } ^ { L G } = \frac { 1 } { M } \sum _ { m = 1 } ^ { 1 0 } \pmb { q } _ { i , k } ^ { ( m ) }\tag{4}
$$

Each $\pmb q _ { i } ^ { ( m ) }$ is softmax-normalised, so the ensemble average also satisfies ${ \mathfrak { d } } <$ $\pmb { p } _ { i , k } ^ { L G } ( \pmb { x } ) < 1$ $\begin{array} { r } { \sum _ { k = 1 } ^ { 8 } p _ { i , k } ^ { L G } = 1 } \end{array}$ . A deterministic teacher prediction may be recovered through $\hat { y } ^ { L G } = \arg \operatorname* { m a x } _ { k } { p _ { i , k } ^ { L G } }$ . In the present framework, however, the complete probability vector $\pmb { p } _ { i } ^ { L G } = ( \pmb { p } _ { i , 1 } ^ { L G } , \cdots , \pmb { p } _ { i , k } ^ { L G } )$ is retained. A sharply concentrated vector identifies a condition for which the teacher assigns one dominant outcome, whereas comparable probabilities indicate reduced separability among neighbouring regimes. These continuous class-probability fields, rather than the discrete argmax labels, constitute the targets for the symbolic-distillation procedure described in Section 2.4.

## 2.4 Symbolic Regression Distillation of Probability Functions

Although the LightGBM teacher provides a full probability vector for each collision condition, deploying its tree ensemble directly in spray simulation is impractical due to computational cost and its black-box nature. We therefore employ symbolic regression to distill each component of the learned probability distribution into an explicit, closed-form expression composed of elementary mathematical functions [33]. The LightGBM ensemble provides, for each collision event, a teacher probability vector $\pmb { p } _ { i } ^ { L G } = ( \pmb { p } _ { i , 1 } ^ { L G } , \cdots , \pmb { p } _ { i , k } ^ { L G } )$ . Symbolic regression (SR) was used to distil these continuous outputs rather than the discrete class labels, thereby retaining information on the relative likelihoods of competing outcomes.

A separate analytical representation was constructed for each regime using the five dimensionless descriptors $\pmb { x _ { i } } = ( P _ { i } , W e _ { i } , B _ { i } , \varDelta _ { i } , O h _ { i } )$ . Because the teacher probabilities exhibit markedly different distributions among the eight regimes, a uniform regression strategy would be dominated by the numerous near-zero values and would inadequately resolve rare or localised outcomes.

Candidate expressions were evolved from the input variables using the basic arithmetic ( + , − , × , and ÷ ), trigonometric ( ��� and ��� ), exponential ( ��� ), logarithmic (��), and square root $( { \surd } )$ operators. The symbolic regression was applied independently to each regime, so the fitting target was adapted to the distribution of the teacher probabilities.

Each teacher probability field $\pmb { p } _ { i } ^ { L G }$ was distilled into an analytical representation by symbolic regression. Because the eight fields differ substantially in both dynamic range and spatial support, fitting the raw probabilities uniformly would be inappropriate. In particular, for globally sparse regimes, the abundant near-zero values would dominate the regression objective and obscure the small but relevant variations that distinguish active from inactive regions. The regression target was therefore defined on a regime-specific basis, and the resulting expression was subsequently mapped to an intermediate regime score $r _ { k }$ through a fixed post-processing operation.

Three fitting strategies were adopted. For Regimes II–V, $\pmb { p } _ { i } ^ { L G }$ are broadly distributed across the unit interval within their active regions, providing sufficiently resolved targets for direct fitting. For Regimes I, VI, VII, and VIII, by contrast, the probability is concentrated near zero for most samples. For Regimes I and VIII, sparsity arises mainly from broadly distributed low probabilities rather than localised support. Symbolic regression was therefore applied to the log-odds of ln ${ \frac { p _ { k } } { 1 - p _ { k } } } .$ For Regimes VI and VII, appreciable probabilities are spatially localised. Each field was therefore factorised into a localising component $f _ { k } ^ { g }$ , fitted over all samples to identify the active region, and a conditional-magnitude component $f _ { k } ^ { m }$ , fitted to the log-odds of $\pmb { p } _ { i } ^ { L G }$ using only the active samples.

In each case, the symbolic search minimised the weighted mean-square error between the expression output and its corresponding teacher-derived target, with fitting error and expression complexity treated as competing objectives. The final expression for each regime was selected from the Pareto front [34] based on fidelity to the teacher output, structural compactness, and numerical stability. The selected expressions were converted into bounded class scores according to

$$
\begin{array} { r } { r _ { k } ( x ) = \left\{ \begin{array} { l l } { \sigma [ f _ { k } ( x ) ] , } & { k \in \{ \mathrm { ~ I ~ } , \forall \Vert \} } \\ { \Pi _ { [ 0 , 1 ] } [ f _ { k } ( x ) ] , } & { k \in \{ \mathrm { I I } , \Pi \mathrm { I I } , \mathrm { I V } , \mathrm { V } \} } \\ { \Pi _ { [ 0 , 1 ] } \big [ f _ { k } ^ { g } ( x ) \big ] \sigma [ f _ { k } ^ { m } ( x ) ] , } & { k \in \{ \mathrm { V I } , \forall \Vert \} } \end{array} \right. } \end{array}\tag{5}
$$

where $\Pi _ { [ 0 , 1 ] } ( u ) = \operatorname* { m i n } ( \operatorname* { m a x } ( u , 0 ) , 1 )$ denotes clipping to [0,1]. Accordingly, the raw-probability fits for Regimes II–V are clipped directly. For Regimes I and VIII, the expression lies on the log-odds scale and is mapped back to a probability by the sigmoid �. Regimes VI and VII combine both mappings: the localising factor $f _ { k } ^ { g }$ is clipped to [0,1] to provide a continuous multiplicative gate, whereas the magnitude factor $f _ { k } ^ { m }$ fitted on the log-odds scale, is transformed by the sigmoid to recover the probability magnitude within the active region.

These independently fitted outputs are bounded class scores, but they are not yet mutually normalised probabilities and do not generally satisfy $\textstyle \sum _ { k } r _ { k } = 1$ . To place the eight scores on a common scale, each $r _ { k }$ was subjected to a class-specific affine transformation, followed by a joint softmax

$$
p _ { k } ^ { S R } ( x ) = \frac { e ^ { [ \alpha _ { k } r _ { k } ( x ) + \beta _ { k } ] } } { \sum _ { j = 1 } ^ { K } e ^ { [ \alpha _ { j } r _ { j } ( x ) + \beta _ { j } ] } } , k = \mathrm { ~ I ~ } , \cdots , \mathbb { M } , \sum _ { k = 1 } ^ { k } p _ { k } ^ { S R } ( x ) = 1\tag{6}
$$

The coupling coefficients $\{ \alpha _ { k } , \beta _ { k } \}$ were determined jointly on a stratified calibration subset of the collected database by minimising the class-weighted multiclass crossentropy against the observed collision labels.

## 2.5 Stochastic Realization of Symbolic Probability Outputs for E-L Simulation

Eulerian-Lagrangian spray simulations require each detected collision to be assigned a single outcome so that the corresponding regime-specific post-collision model can be activated [35]. By contrast, the symbolic model returns the complete probability vector $\pmb { p } _ { k } ^ { S R } ( \pmb { x } )$ . This vector is therefore not directly usable in such simulations. To resolve this, this probability distribution is converted into an eventlevel outcome through categorical sampling (“biased-dice” sampling [31]), as illustrated in Fig. 4. For a collision characterised by �, a uniformly distributed random number $U \in [ 0 , 1 )$ is generated, and the realised regime is determined from the cumulative probabilities

$$
\begin{array} { r } { \hat { Y } ( \pmb { x } ) = \operatorname* { m i n } \bigr \{ k { : } U < \sum _ { j = 1 } ^ { k } p _ { j } ^ { s R } ( \pmb { x } ) \bigr \} , ~ U \sim \mathcal { U } ( 0 , 1 ) } \end{array}\tag{7}
$$

This construction ensures that the conditional probability of selecting regime � is exactly $\pmb { p } _ { k } ^ { S R } ( \pmb { x } )$ . For a collection of collision events with different local conditions, the expected number of occurrences of regime � is therefore $\sum _ { i } { \pmb { p } } _ { i , k } ^ { S R }$ . Event-level categorical draws consequently produce multinomial regime counts at the ensemble level, preserving the outcome distribution represented by the symbolic probability field. This procedure corresponds to the "biased random sampling", where the probability weights determine the likelihood of each face of the conceptual dice [36].

The stochastic realization has a clear limiting behaviour. When one probability approaches unity, the sampled outcome becomes effectively deterministic and coincides almost invariably with the argmax prediction. Within model-inferred transition regions, several neighbouring regimes retain appreciable probabilities and remain accessible under similar collision conditions. Sampling therefore retains this competition instead of collapsing the probability vector prematurely to a single deterministic boundary. This interpretation does not require every sampled event to reproduce its experimentally observed label. Its purpose in forward simulation is to preserve the regime frequencies implied by the probabilistic model. The argmax rule is retained separately for evaluating deterministic classification performance.

![](images/9c900c8a1b81b4ece66f07e6a96f637f5b348e638d2df42717466937db1633b0.jpg)  
Fig. 4. Schematic of the stochastic regime selection from eight-class symbolic regression probability outcomes using multinomial distribution sampling.

In a spray simulation, the procedure is applied after a collision pair has been identified: the five dimensionless descriptors are evaluated from the pre-collision states, the eight analytical probabilities are computed, and one regime is sampled to activate the associated post-collision model. It therefore modifies only the outcome-selection stage and does not alter the collision-frequency or collision-partner models. Because the probability evaluation involves only closed-form expressions followed by a single random draw, the scheme can be incorporated without retaining the original LightGBM model or introducing an iterative inference procedure. Its distributional consistency, convergence with repeated realizations, and relation to deterministic argmax selection are examined in Section 3.3.

## 3. Results and Discussion

3.1 Closed-form Symbolic Probability Functions and Predictive Performance

Symbolic distillation constitutes the central step through which the learned collision-regime probability landscape is converted into an explicit analytical model. As listed in Table 1, the expressions therefore do not act as eight binary transition criteria. When evaluated together, these expressions provide the class scores underlying the coupled probability model. This section examines the predictive structure encoded by these expressions, their consistency with established physical behaviour, their performance over the collected database, and the efficiency of their analytical representation.

Table 1 shows that the eight collision outcomes retain distinct combinations of governing descriptors. The bouncing expression involves �� and � with �ℎ and �, whereas hard coalescence combines ��, �, ∆, and �ℎ. Reflexive separation is associated with sufficient inertia and near-head-on impact, while stretching separation is governed more strongly by the interaction between �� and �. Ambient pressure appears selectively rather than uniformly across the expressions.

The pressure dependence of the bouncing–coalescence transition provides an illustrative example of the physical trends recovered within the probabilistic representation. At fixed $\Delta$ and �ℎ, Fig. 5 extracts, for each � and �, the critical Weber number $W e _ { c r }$ at the midpoint of the fuzzy transition and compares the resulting contour with the experimental data. Rather than being fitted separately as a deterministic boundary, this contour emerges directly from the joint probability field; while its midpoint enables comparison with conventional criteria, the underlying model retains the finite-width redistribution of probability among competing outcomes across the transition. The predicted $W e _ { c r }$ initially increases nonlinearly with pressure and subsequently approaches a plateau, in agreement with the experimental trend reported by Zhang et al. (2026) [18]. The coupled symbolic model can represent this behaviour because pressure is retained explicitly in the relevant class expressions, whereas conventional pressure-independent criteria based only on �� and � necessarily predict an invariant boundary.

![](images/40c407e78ac924eaa50ca70040fc8ea38b54336e641da92f956757489dc911f1.jpg)  
Fig. 5. Critical Weber number $W e _ { c r }$ at the midpoint of the bouncing–coalescence transition predicted by the SR model as a function of the pressure ratio $P = p / p _ { 0 }$ for different impact parameters B, compared with experimental data (Qian and Law, 1997 [12]; Zhang et al., 2026 [18]). $\Delta = 1$ , and $O h = 0 . 0 8$

Three representative expressions illustrate this physical consistency. For soft coalescence, the symbolic score decreases monotonically with $W e _ { ; }$ , and the associated probability changes from nearly unity below $W e = 1$ to negligible values above $W e = 5$ . Consistently, all 274 soft-coalescence events occur below $W e = 5$ . The splashing expression contains only $W e$ and $O h$ , consistent with the variable combination appearing in the classical Mundo–Cossali-type splashing parameter, $K \propto$ $W e ^ { 0 . 6 2 5 } O h ^ { - 0 . 2 5 }$ [37], although not its exact threshold. This compact expression achieves a balanced accuracy of 0.998 and an AUROC of 1 in Figs. 6(a) and 6(b). For reflexive separation, the normalised sensitivity analysis assigns � a dominant contribution of 0.905, while the exponential dependence on � suppresses the score for increasingly oblique collisions and renders it negligible for $B > 0 . 5$ . This agrees with the established confinement of reflexive separation to predominantly near-head-on collisions [10], where temporary coalescence is followed by strong axial deformation and recoil. Taken together, these examples show that the distilled expressions retain physically recognisable limits and variable combinations without imposing them beforehand.

Table 1. Symbolic regression analytical expressions for each collision regime, together with the softmax coupling coefficients $\{ \alpha _ { k } , \beta _ { k } \} .$
<table><tr><td>Regime</td><td>Formulas</td><td></td><td> $\begin{array} { r } { ( { p } _ { k } ^ { S R } ( x ) = \frac { e ^ { \left[ \alpha _ { k } r _ { k } ( x ) + \beta _ { k } \right] } } { \sum _ { j = 1 } ^ { K } e ^ { \left[ \alpha _ { j } r _ { j } ( x ) + \beta _ { j } \right] } } , x _ { 0 } ; P , x _ { 1 } ; W e , \ x _ { 2 } { : } B , x _ { 3 } { : } \Delta , \mathrm { a n d } x _ { 4 } { : } O h ) } \end{array}$ </td></tr><tr><td>I</td><td colspan="3"> $\begin{array} { r } { \boldsymbol { r } _ { 0 } = \sigma ( f _ { 0 } ) = \frac { 1 } { 1 + e ^ { - f _ { 0 } } } , \ \boldsymbol { \alpha } _ { k } = 1 0 . 5 9 , \mathrm { a n d } \ \boldsymbol { \beta } _ { k } = 7 . 0 5 ; } \end{array}$ </td></tr><tr><td>(Soft coalescence)</td><td colspan="3"> $\begin{array} { r } { f _ { 0 } = 4 . 2 9 \ \ln [ \exp ( 2 . 4 9 - ( \cos ( x _ { 1 } ) + x _ { 1 } ) - \sin [ 2 . 1 8 \sin ( u ) ] ) ) + 0 . 1 1 8 ] , \mathrm { { w h e r e } } u = ( - 1 4 . 5 \sqrt { x _ { 4 } } + x _ { 3 } ) \cdot \cos \left( \frac { x _ { 2 } } { - 0 . 1 3 7 } \right) + 2 x _ { 2 } . } \end{array}$ </td></tr><tr><td>Ⅱ</td><td colspan="3"> $r _ { 1 } = c l i p ( f _ { 1 } , \mathbf { 0 } , \mathbf { 1 } ) , \ \alpha _ { k } = 6 . 6 3 , \mathrm { a n d } \ \beta _ { k } = 6 . 8 7 ;$ </td></tr><tr><td>(Bouncing)</td><td colspan="3"> $\begin{array} { r } { f _ { 1 } = \sin { \left[ \exp { \left( x _ { 1 } \cdot ( 1 - x _ { 2 } ) ^ { 2 } \cdot \frac { - 0 . 0 0 1 3 1 x _ { 1 } \cdot x _ { 4 } ^ { - 0 . 4 } } { x _ { 0 } } \right) } \right] } . } \end{array}$ </td></tr><tr><td>ⅢII</td><td colspan="3"> $r _ { 2 } = c l i p ( f _ { 2 } , \mathbf { 0 } , \mathbf { 1 } ) , \alpha _ { k } = 5 . 4 9 , \mathrm { a n d } \beta _ { k } = 7 . 0 4 ;$ </td></tr><tr><td>(Hard coalescence)</td><td colspan="3"> $\begin{array} { r } { f _ { 2 } = c o s \left[ s i n ( x _ { 2 } + \left[ \frac { c o s ( 0 . 0 0 7 4 2 x _ { 1 } + ( x _ { 2 } / \sqrt { x _ { 3 } } ) / 0 . 3 9 1 ) } { - 0 . 6 7 5 } \right] - x _ { 4 } ) + 0 . 5 5 0 \right] \cdot s i n ( \sqrt { x _ { 3 } } ) . } \end{array}$ </td></tr><tr><td>IV (Reflexive separation)</td><td colspan="3"> $r _ { 3 } = c l i p ( f _ { 3 } , \mathbf { 0 } , \mathbf { 1 } ) , \ \alpha _ { k } = 8 . 5 4 , \mathrm { a n d } \ \beta _ { k } = 7 . 0 0 ; \ f _ { 3 } = s i n \left[ \exp ( 1 . 1 0 - \frac { ( 2 . 1 9 / x _ { 1 } + x _ { 4 } ) \cdot 9 . 0 4 } { e x p ( x _ { 2 } / ( - 0 . 1 5 8 ) ) } \right] .$ </td></tr><tr><td>V (Stretching separation)</td><td colspan="3"> $r _ { 4 } = c l i p ( f _ { 4 } , \mathbf { 0 } , \mathbf { 1 } ) , \ \alpha _ { k } = 6 . 0 3 , \mathrm { a n d } \ \beta _ { k } = 7 . 0 0 ; \ f _ { 4 } = s i n [ e x p ( s i n [ s i n ( - 2 . 7 8 x _ { 2 } ) \cdot l n ( x _ { 1 } ) ] ) \cdot x _ { 2 } ] .$ </td></tr><tr><td>VI</td><td colspan="3"> $r _ { 5 } = c l i p ( f _ { 5 } ^ { g } , \mathbf { 0 } , \mathbf { 1 } ) \times \sigma ( f _ { 5 } ^ { m } ) , \alpha _ { k } = 6 . 6 2 , \mathrm { a n d } \beta _ { k } = 8 . 2 8 ;$ </td></tr><tr><td>(Rotational separation)</td><td> $\begin{array} { r } { f _ { 5 } ^ { m } = 2 . 1 9 \cdot \sin { [ \frac { 1 } { 0 . 5 1 5 } \sin ( \sin \frac { \exp ( s i n ( \frac { 1 . 4 9 } { \ln ( c o s ( x _ { 4 } ) ) } ) \cdot x _ { 2 } ) } { 0 . 3 8 3 / \sin ( \frac { \ln x _ { 1 } } { - 0 . 8 1 6 } ) } ) - 0 . 3 3 4 ] } . } \end{array}$ </td><td> $\begin{array} { r } { f _ { 5 } ^ { g } = c o s [ c o s ( c o s ( { c o s ( \frac { { l n ( x _ { 1 } ) \cdot x _ { 2 } } } { 0 . 4 3 8 } ) \cdot ( \frac { { c o s ( x _ { 2 } \cdot 8 . 7 4 ) - 0 . 4 4 7 } ) } { ( x _ { 4 } - x _ { 0 } \cdot ( - 0 . 4 3 9 ) ) \cdot x _ { 3 } } ) } - 0 . 7 6 7 ] \cdot ( - 9 . 3 6 ) , } \end{array}$ </td><td></td></tr><tr><td>VII</td><td> $\begin{array} { r } { r _ { 6 } = c l i p ( f _ { 6 } ^ { g } , \mathbf { 0 } , \mathbf { 1 } ) \times \pmb { \sigma } ( f _ { 6 } ^ { m } ) , \alpha _ { k } = 1 6 . 9 , \mathrm { a n d } \beta _ { k } = 6 . 3 1 ; } \end{array}$ </td><td></td><td></td></tr><tr><td>(Finger separation)</td><td colspan="3"> $\begin{array} { r } { f _ { 6 } ^ { g } = s i n [ e ^ { x _ { 2 } } \cdot \frac { 0 . 4 0 4 } { \cos ( ( c o s ( l n ( x _ { 1 } + 1 2 . 3 ) - 0 . 6 7 4 ) - 0 . 8 7 8 ) \cdot 2 . 3 5 ) - 0 . 7 6 2 } - 0 . 9 7 4 ] \cdot 5 . 8 1 - 3 . 4 5 , } \end{array}$   $\begin{array} { r } { f _ { 6 } ^ { m } = s i n [ \frac { x _ { 0 } + x _ { 2 } } { c o s ( x _ { 2 } ) + x _ { 2 } \cdot c o s ( x _ { 1 } ) } \cdot \frac { 1 } { s i n ( { x _ { 1 } } ^ { 2 } \cdot 1 . 3 8 \cdot 1 0 ^ { - 6 } ) } ] + 0 . 8 9 0 . } \end{array}$ </td></tr><tr><td>VIII</td><td> $\begin{array} { r } { \pmb { r } _ { 7 } = \pmb { \sigma } ( f _ { 7 } ) = \frac { 1 } { 1 + e ^ { - f _ { 7 } } } , \ \alpha _ { k } = 2 6 . 2 , \mathrm { a n d } \ \beta _ { k } = 5 . 0 2 ; } \end{array}$ </td><td></td><td></td></tr><tr><td></td><td colspan="3"> $\begin{array} { r } { f _ { 7 } = \frac { \sin \left[ \ln \left( 2 . 0 1 + \exp \left[ \frac { { \sin ( c o s \left( \frac { - 0 . 3 3 5 } { x _ { 4 } } \right) \cdot 0 . 0 0 2 1 8 \cdot x _ { 1 } + 1 . 4 7 } ) } { 0 . 2 0 5 } \right] + \exp \left[ 2 . 9 9 + x _ { 1 } \cdot s i n ( \frac { 0 . 4 1 9 } { x _ { 4 } } ) \cdot 0 . 0 0 2 2 6 } \righ \right\right]t])  } { 0 \ 1 7 1 } - 3 . 5 8 .  \end{array}$ </td></tr><tr><td>(Splashing)</td><td></td><td></td><td></td></tr></table>

On the collected eight-regime database, the symbolic model achieves a macroaveraged balanced accuracy of 0.918 under the argmax decision rule and a macro AUROC of 0.953, as shown in Figs. 6(a) and 6(b). The class-wise results show that predictive performance is closely related to regime separability. Soft coalescence (I) and finger separation (VII), which occupy comparatively distinct regions of the collision-parameter space, achieve balanced accuracies of 0.979 and 0.999, respectively. By contrast, bouncing (II) and hard coalescence (III) attain lower values of 0.834 and 0.775 because their predicted probabilities are often comparable within the bouncing– coalescence transition band. The representative $W e - B$ map in Fig. 6(c) at $P = 1$ ， $\Delta = 1$ , and $O h = 0 . 0 1$ , connects these performance differences to the spatial probability structure. A single outcome dominates within well-separated regime interiors, while neighbouring outcomes remain simultaneously probable across modelinferred finite-width transition regions. The symbolic model therefore retains strong probability-ranking capability while representing the reduced separability of competing outcomes near their transitions.

Figure 7(a) provides a complementary check of the class-wise results: the confusion matrix is largely diagonal, with the largest off-diagonal entries occurring between Regimes II and III, consistent with their comparatively lower balanced accuracies. Figure 7(b) shows that the symbolic model achieves a favourable accuracy– complexity balance. Increasing the polynomial order of the fixed-basis multinomial logistic-regression models substantially increases expression complexity, whereas the corresponding improvement in balanced accuracy is marginal. The symbolic model achieves higher accuracy than the higher-order polynomial models with a more compact expression because it assigns a separate nonlinear structure to each collision regime instead of expanding a common basis across all regimes.

(a)  
![](images/642988b4413239369c0a29f8f3846d39765efa1f4f782de26425b446630d3c8c.jpg)

(b)  
![](images/64fe594e70deaaa7afde3ca6c7d726ac55aa30272b9a231f006862cf5f3fa257.jpg)

(c)  
![](images/d1bb8a0e13b793728ffd76175c4e1dd59263af249617919c3a9e9c47508d886b.jpg)  
Fig. 6. Predictive performance and fuzzy regime structure of the symbolic-regression model. (a) Balanced accuracy for individual classes and macro-average. (b) ROC curves and corresponding AUROC performance. (c) Fuzzy regime mapping in the We-B plane, obtained by fixing the remaining inputs at $P = 1$ 2 $\Delta = 1$ , and $O h = 0 . 0 1$

(a)  
![](images/87173a804e6721f0936f840f4ed24b36bd260268567d7fede7cba2b1accda28c.jpg)

(b)  
![](images/a946fe8f4afe39b03e63b8aa7208fcfc57d776905cf9dd1cc0043e84a749e892.jpg)  
Fig. 7. Classification errors and model-complexity comparison for the symbolicregression (SR) model. (a) Confusion matrix for classification results. (b) Accuracy– complexity tradeoff between multinomial logistic regression and symbolic regression.

## 3.2 Comparison with Deterministic Analytical Models

The central distinction between the proposed and conventional analytical models lies in their representation of regime transitions. Classical models partition a lowdimensional collision map using prescribed curves and assign each condition to one outcome according to its position relative to the corresponding boundary. The symbolic model instead evaluates a joint probability distribution over all eight outcomes, allowing neighbouring regimes to remain simultaneously plausible where their distributions overlap. As shown in Fig. 8, the classical LB, RS-C, and SS-C criteria pass through the principal transition regions and therefore retain physical value in locating where the dominant outcome changes. The symbolic probability field additionally represents the finite width of these regions and the changing relative likelihoods within them. The distinction is therefore not simply between different transition locations, but between a zero-width deterministic partition and a continuous probability transition.

![](images/9436c3b331a22fc6786eb8044fa1c23f7af7bb7baef4db61e5b750e440174c2c.jpg)  
Fig. 8. Symbolic probability field in the $W e - B$ plane $( P = 1 , \ \Delta = 1$ , and $O h =$ 0.01), overlaid with classical analytical boundary models for bouncing-coalescence (LB), reflexive separation-coalescence (RS-C), and stretching separation-coalescence (SS-C) [10, 11, 13, 24, 25, 28, 38, 39].

The finite width of the model-inferred transition regions has a measurable consequence for prediction confidence and regime separability. When regime interiors are defined by $( p _ { k } ^ { S R } ) _ { m a x } > 0 . 9$ and transition bands by $( p _ { k } ^ { S R } ) _ { m a x } < 0 . 7$ , the mean top probability decreases from 0.949 to 0.575, while the mean balanced accuracy over regimes II–V decreases from 0.917 to 0.701. This reduced separability does not imply that the classical criteria entirely misplace the transitions. For example, 92% of the reflexive-separation samples in the RS–C band lie within a factor of two of the Ashgriz– Poo boundary [10]. The classical criterion therefore identifies the approximate region in which the outcomes compete, but its zero-width partition cannot represent their overlap within that region.

This distinction is reflected quantitatively in the boundary-resolved comparison in Fig. 9. For LB and SS–C, the symbolic model provides relatively modest improvements over the best analytical criteria, increasing balanced accuracy from 0.786 [28] to 0.834 and from 0.849 [25] to 0.878, respectively. A substantially larger improvement is obtained for RS–C, from 0.620 [13] to 0.933, identifying this strongly overlapping transition as the principal limitation of the deterministic formulation. The ROC curves show the same ordering: the symbolic-model AUROC ranges from 0.914 to 0.981 across the three transitions and reaches 0.981 for RS–C. The agreement between the balanced-accuracy and ROC results shows that the improvement reflects stronger discrimination across thresholds rather than the selection of a favourable operating point.

Figure 10 extends the comparison from isolated transitions to simultaneous prediction of the four canonical regimes. Relative to the multi-boundary Analytical Combined Model (ACM) [21], the balanced-accuracy gains of the symbolic model range from 0.0378 to 0.0835, with the largest improvement obtained for bouncing. By evaluating all candidate outcomes within a common probability space, the symbolic model maintains stronger discrimination when the transitions are considered simultaneously.

Together, Figs. 8–10 distinguish between recalibrating individual transition curves and changing the mathematical representation of collision outcomes. Classical criteria remain informative for locating physically meaningful transitions, but their deterministic structure collapses the finite width and internal competition of these transitions into single switching curves. The symbolic framework retains an explicit analytical form while improving both boundary-resolved and simultaneous multiregime prediction, with its largest advantages arising where neighbouring outcomes are least separable. The improvement therefore arises from the unified probability representation rather than from constructing another set of recalibrated deterministic boundaries.

![](images/2ff049a92dfad24a9a12a3bd41fb18dd97710aa12840fa8c3f417583c7654c74.jpg)

![](images/0c0b25542d6db97c2654d32c9251382033457f9435e1a38b0a9bb201e42ca383.jpg)

![](images/997a3627f1a86d88afbdd17da0b0632110a1d1ec0f365427b6fb3cd305a199c8.jpg)

![](images/9b2dd3088cf30f3ad54759c521af95d4b8ce61dd5dc8db2658a2cf2977488b3e.jpg)

![](images/62eacd44710d64d3a04e1bed2cfa6d871c8e20239d83ef8a9d8b5edfd1f3d771.jpg)

![](images/c0081db0ebbc7d9cbd1d3f80d6b290a158a4437090b1e1db5e3b2e6419a05309.jpg)  
Fig. 9. Boundary-resolved balanced-accuracy and ROC comparison among the classical analytical criteria and the proposed symbolic model for (a) bouncingcoalescence (LB), (b) reflexive-separation-coalescence (RS-C), and (c) stretchingseparation-coalescence (SS-C) [10, 11, 13, 24, 25, 28, 38, 39]. The LightGBM teacher is included as a reference.

![](images/f40ebf00dd6f4720b5dc0479e6e81e4d6f2ce676f7ccf182bb80a69618b968a4.jpg)  
Fig. 10. Balanced accuracy comparison between Analytical Combined Model (ACM) and the proposed symbolic model. The LightGBM teacher is included as a reference.

## 3.3 Stochastic Realization of Collision Outcomes by Multinomial Sampling

Eulerian-Lagrangian spray simulations require a discrete outcome for each collision, whereas the symbolic model returns a probability vector over the eight regimes. For collision � , multinomial sampling assigns one categorical outcome $Y _ { i } { \sim } C a t ( \pmb { p } _ { i } ^ { S R } )$ , such that $\mathbb { E } [ 1 ( Y _ { i } = k ) ] = { \pmb { p } } _ { i , k } ^ { S R }$ . It therefore preserves the predicted probability mass in expectation, whereas argmax assignment collapses every probability vector to its largest component. This distinction is evident in Fig. 11(a): argmax assignment overrepresents bouncing by 2.83 percentage points and underrepresents rotational separation by 1.70 percentage points relative to the SRimplied probability mass. By contrast, the mean regime fractions over 200 independent realizations reproduce the predicted masses with a maximum absolute deviation of $9 \times 1 0 ^ { - 5 }$

The repeated realizations in Fig. 11(b) quantify the Monte Carlo convergence of this stochastic interface. As the number of draws per event, �, increases from 1 to 200, the aggregate sampling error decreases from 0.00493 to 0.000350 and follows the expected $M ^ { - 1 / 2 }$ scaling. This convergence is a numerical verification that the sampling implementation is unbiased and statistically consistent with the symbolic probability field. The practical difference between stochastic and deterministic realization is governed by prediction uncertainty. Agreement between multinomial sampling and argmax assignment decreases from 0.987 in the lowest-entropy interval to below 0.5 when the entropy exceeds approximately 1.23, as shown in Fig. 11(c). Sampling therefore approaches deterministic selection when one outcome dominates but retains alternative outcomes in high-entropy transition regions.

(a)  
![](images/184aea227e6c6b8c226b35e71a7f2fbe49feaeed06e2a06388a235ab3ca9fbaf.jpg)

(b)  
![](images/73b19ef40f45090f273f49027024930e2d38e9159d1a707e2205bdd91c3c6e43.jpg)  
Monte Carlo sample size per event (M)

(c)  
![](images/10d372c90e1156fe401063c2da5078ab60908f3c7957628a313a9c43d96e8776.jpg)  
Fig. 11. Statistical behaviour of multinomial collision-outcome sampling: (a) Comparison of the regime fractions obtained from SR probability mass, argmax assignment, and multinomial sampling. (b) Convergence of the aggregate sampling error with the number of realizations �. (c) Agreement between multinomial sampling and argmax selection as a function of prediction entropy. Error bars and shaded regions indicate 95% intervals.

Across repeated realizations, the sampled outcomes attain a macro-averaged balanced accuracy of 0.884, with a 95% confidence interval of [0.880,0.887], as shown in Fig. 12. Multinomial sampling thus provides a distribution-preserving and computationally direct interface between the analytical probability model and the discrete outcomes required by Eulerian-Lagrangian simulation. Its influence on sprayscale droplet statistics remains to be assessed in fully coupled simulations.

![](images/096ca7d7b75c7fcd8c1fa804540a2567a7ce9f9360cdafe714083b89ab73886e.jpg)  
Fig. 12. Balanced accuracy of the sampled outcomes of multinomial sampling: Error bars and shaded regions indicate 95% intervals.

## 3.4 Regime-specific Feature Dependence and Implications for Model Reduction

The finite-width transition regions represented by the symbolic model could arise either from the LightGBM probability landscape or from errors introduced during symbolic compression. Figure 13 distinguishes between these possibilities. The teacher model exhibits reduced maximum probabilities and concentrated misclassifications in the same regions where neighbouring outcomes overlap, while its class-wise performance decreases primarily for the principal transitional regimes (Ⅱ-Ⅴ). The spatial correspondence between the teacher and symbolic probability fields therefore shows that symbolic distillation preserves, rather than creates, the transition structure learned by the teacher. This comparison identifies the origin of the transition bands within the modelling pipeline. It does not by itself establish that their finite width represents an intrinsic physical stochasticity of droplet collisions. The regime-resolved SHAP analysis in Fig. 13 and Table 2 shows that no universal feature hierarchy applies to all eight outcomes. Averaged across classes, � and �� have the largest mean absolute SHAP values of 2.45 and 2.12, respectively, followed by �ℎ at 1.44. Their relative importance nevertheless changes substantially among regimes. Within the teacher–student framework, this heterogeneity suggests that different regimeprobability functions may benefit from regime-specific reduced input sets rather than a universal descriptor set shared across all outcomes.

Table 2. Mean absolute values of SHAP of each input feature across eight regimes.
<table><tr><td>Feature</td><td>Regime I</td><td>Regime ⅡI</td><td>Regime ⅢⅢ</td><td>Regime IV</td><td>Regime V</td><td>Regime VI</td><td>Regime VII</td><td>Regime VIII</td><td>Average</td></tr><tr><td>P</td><td>0.279</td><td>0.786</td><td>0.206</td><td>0.299</td><td>0.248</td><td>0.165</td><td>0.0314</td><td>0.0433</td><td>0.257</td></tr><tr><td>We</td><td>0.799</td><td>4.31</td><td>1.37</td><td>4.54</td><td>4.40</td><td>0.746</td><td>0.484</td><td>0.344</td><td>2.12</td></tr><tr><td>B</td><td>0.709</td><td>3.45</td><td>3.85</td><td>3.95</td><td>4.08</td><td>1.18</td><td>0.958</td><td>1.39</td><td>2.45</td></tr><tr><td>Δ</td><td>0.590</td><td>0.309</td><td>0.743</td><td>0.592</td><td>0.25</td><td>0.108</td><td>0.0103</td><td>0.0368</td><td>0.330</td></tr><tr><td>0h</td><td>1.11</td><td>2.03</td><td>0.741</td><td>4.08</td><td>1.35</td><td>1.10</td><td>0.453</td><td>0.649</td><td>1.44</td></tr></table>

(a)  
![](images/b1315bc36eac5545062f76e5b3294ef6c4253bd26319f0a27d1236a2af988c3d.jpg)

(b)  
![](images/c3f4be1e50070267492ea80009554d79aba22e3222605f646cd4e43a6faec676.jpg)

(c)  
![](images/c8c44466687bce69a98876954d516aad1e415b14eecf3037a2ea5652105cf0e1.jpg)

(d)  
![](images/24639909c4b027c29e6e919d61fbc9b567f2d4f5b18ec5fa23e2c84b84c57188.jpg)

(e)  
![](images/7ee93c84f26bc3fe1f99c6266d375f33f1398b36872de8725fee8a4dc5339362.jpg)

![](images/148d3ceafadacf24e777f5a24f637750ff341004d992a7362caed89db86e540c.jpg)

![](images/a777daca068a77205f5eb3c88da44a0173fb929d1b5dd2bf71ea9e3838d5e4a6.jpg)

![](images/668ff91c959e7034ba4c1e84a77a9a48d451cb1f4d79085d18b7bac90b8bb5bb.jpg)  
Fig. 13. SHAP analysis revealing feature importance and regime-resolved feature contributions of the LightGBM model. (a) soft coalescence, (b) bouncing, (c) hard coalescence, (d) reflexive separation, (e) stretching separation, (f) rotational separation, (g) finger separation, and (h) splashing.

The regime-dependent SHAP patterns provide a possible route towards reducing the symbolic model as the number of descriptors and collision outcomes increases. The present study retains all five inputs to preserve a common teacher–student representation and does not infer reduced feature sets from SHAP values alone. Future reduction should combine regime-specific feature screening with controlled ablation and additional validation, while imposing physical constraints where a descriptor is known to be essential to a collision mechanism. Such a procedure could limit the rapid growth of the symbolic search space without sacrificing regime discrimination or removing physically necessary dependencies.

## 4. Concluding remarks

This study developed a probabilistic symbolic-distillation model for binary droplet-collision in gaseous environments up to 50 atm. A LightGBM teacher was trained on 38,762 experimental events spanning eight collision regimes and five dimensionless parameters ( $W e \in [ 0 , 2 0 0 0 ]$ 2 ${ O h } \in [ 9 . 5 \times 1 0 ^ { - 4 } , 5 . 5 \times 1 0 ^ { - 1 } ]$ 2 $B \in$ [0,1], $\Delta \in [ 1 , 5 ]$ , and $P \in [ 0 . 6 , 5 0 ] )$ , and its probability landscape was distilled into eight class-specific closed-form expressions. Unlike conventional models that prescribe a separate deterministic curve for each transition between regimes, the present model

evaluates the relative likelihoods of all collision outcomes within a single normalised probability space, retaining the flexibility of data-driven learning in an explicit form suitable for Eulerian–Lagrangian spray simulations.

The coupled probability field changes how collision transitions are represented. Instead of imposing zero-width switches, it identifies finite-width fuzzy regions, in which neighbouring outcomes remain simultaneously plausible. The closed-form expressions also recover regime-specific behaviour consistent with established collision physics, including the low-�� limit of soft coalescence, the �� − �ℎ combination appearing in the classical splashing parameter, and the low-�, near-headon constraint of reflexive separation, suggesting that the distilled expressions capture physically relevant trends from the data-derived teacher probabilities.

The symbolic-distillation model retains good predictive discrimination across the conditions examined. On the collected eight-regime database, the distilled model achieves a macro-averaged balanced accuracy of 0.918 and a macro AUROC of 0.953, consistently outperforming the conventional deterministic analytical models evaluated in this study. The largest improvement occurs for the strongly overlapping RS-C transition, for which balanced accuracy increases from 0.620 to 0.933. In the simultaneous prediction of the four canonical regimes, the class-wise balancedaccuracy gains over the multi-boundary ACM range from 0.0378 to 0.0835, indicating that individually plausible pairwise boundaries do not necessarily yield a consistent multiclass partition. Finally, multinomial sampling converts the analytical probability vector into a discrete outcome for each collision while preserving the predicted regime distribution at the ensemble level. It therefore provides a distribution-preserving stochastic interface compatible with Eulerian-Lagrangian simulation, avoiding the systematic compositional shifts introduced by deterministic argmax assignment.

The present work establishes and validates the event-level probability model, which achieves a more favourable accuracy and complexity balance than fixed-basis multinomial logistic regression. All five descriptors are retained here to preserve a common teacher–student input space. As the model is extended to higher-dimensional descriptors and finer outcome taxonomies, regime-specific feature screening may also

help control the increasing symbolic-search cost. Such reduction should, however, be supported by controlled ablation, physical constraints, and additional validation.

Acknowledgements. P.Z. acknowledges support from the National Natural Science Foundation of China (No. 52176134) and partially from the APRC-CityU New Research Initiatives/Infrastructure Support from Central of City University of Hong Kong (No. 9610601). T.Y. thanks Dr. Chenwei Zhang and Dr. Zhenyu Zhang for providing the original experimental data. The authors are grateful to the National Supercomputer Center in Guangzhou (Tianhe-2) for supporting the GPU computing.

Declaration of interests. The authors report no conflict of interest.

Data availability. The data that support the findings of this study are available from the corresponding author upon reasonable request.

Supplementary material. Supporting information of the present model is available.

## References

[1] N. Ashgriz, Handbook of atomization and sprays: theory and applications, Springer Science & Business Media2011.

[2] M. Sommerfeld, L. Pasternak, Advances in modelling of binary droplet collision outcomes in sprays: a review of available knowledge, International Journal of Multiphase Flow 117 (2019) 182-205.

[3] F. Hussain, M. Jaskulski, M. Piatkowski, E. Tsotsas, CFD simulation of agglomeration and coalescence in spray dryer, Chemical Engineering Science 247 (2022) 117064.

[4] S. Pawar, J. Padding, N. Deen, A. Jongsma, F. Innings, J.H. Kuipers, Numerical and experimental investigation of induced flow and droplet–droplet interactions in a liquid spray, Chemical Engineering Science 138 (2015) 17-30.

[5] S. Lain, M. Sommerfeld, Influence of droplet collision modelling in Euler/Lagrange calculations of spray evolution, International Journal of Multiphase Flow 132 (2020) 103392.

[6] N.E. Shlegel, P.P. Tkachenko, P.A. Strizhak, Influence of viscosity, surface and interfacial tensions on the liquid droplet collisions, Chemical Engineering Science 220 (2020) 115639.

[7] J. Restrepo-Cano, F.E. Hernández-Pérez, H.G. Im, Fluid dynamic characterization of binary droplet collisions via the pseudopotential lattice-Boltzmann method, Chemical Engineering Science 310 (2025) 121502.

[8] Y. Pan, K. Suga, Numerical simulation of binary liquid droplet collision, Physics of Fluids 17 (2005) 082105.

[9] G. Finotello, J.T. Padding, K.A. Buist, A. Jongsma, F. Innings, J.A.M. Kuipers, Droplet collisions of water and milk in a spray with Langevin turbulence dispersion, International Journal of Multiphase Flow 114 (2019) 154-167.

[10] N. Ashgriz, J. Poo, Coalescence and separation in binary collisions of liquid drops, Journal of Fluid Mechanics 221 (1990) 183-204.

[11] Y. Jiang, A. Umemura, C.K. Law, An experimental investigation on the collision behaviour of hydrocarbon droplets, Journal of fluid mechanics 234 (1992) 171-190.

[12] J. Qian, C.K. Law, Regimes of coalescence and separation in droplet collision, Journal of fluid mechanics 331 (1997) 59-80.

[13] M. Sommerfeld, M. Kuschel, Modelling droplet collision outcomes for different substances and viscosities, Experiments in Fluids 57 (2016) 187.

[14] K.-L. Pan, P.-C. Chou, Y.-J. Tseng, Binary droplet collision at high Weber number, Physical Review E—Statistical, Nonlinear, and Soft Matter Physics 80 (2009) 036301.

[15] D. Zhou, X. Liu, S. Yang, Y. Hou, X. Zhong, Intense deformation and fragmentation of two droplet collision at high Weber numbers, Colloids and Surfaces A: Physicochemical and Engineering Aspects 655 (2022) 130171.

[16] K.-L. Pan, K.-L. Huang, W.-T. Hsieh, C.-R. Lu, Rotational separation after temporary coalescence in binary droplet collisions, Physical Review Fluids 4 (2019) 123602.

[17] I. Roisman, Experimental and computational investigation of binary drop collisions under elevated pressure, (2017).

[18] C. Zhang, Z. Zhang, P. Zhang, J. Zhou, C. Zhao, Leveling-off of droplet bouncing dynamics under high ambient gas pressures, International Journal of Multiphase Flow 194 (2026) 105468.

[19] P.J. O'ROURKE, Collective drop effects on vaporizing liquid sprays, Princeton University, 1981.

[20] D.P. Schmidt, C.J. Rutland, A new droplet collision algorithm, Journal of Computational Physics 164 (2000) 62-80.

[21] W. Yu, S. Chang, A machine learning-based approach to predict the outcome of binary droplet collision, Chemical Engineering Science 319 (2026) 122349.

[22] A.A. Amsden, KIVA-3V: A block-structured KIVA program for engines with vertical or canted valves, Los Alamos National Lab., NM (United States), 1997.

[23] A. Munnannur, R.D. Reitz, A new predictive model for fragmenting and non-fragmenting binary droplet collisions, International journal of multiphase flow 33 (2007) 873-896.

[24] J.-P. Estrade, H. Carentz, G. Lavergne, Y. Biscos, Experimental investigation of dynamic binary collision of ethanol droplets–a model for droplet coalescence and bouncing, International Journal of Heat and Fluid Flow 20 (1999) 486-491.

[25] P. Brazier-Smith, S. Jennings, J. Latham, The interaction of falling water drops: coalescence, Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, (1972) 393-408.

[26] A.A. Amsden, KIVA-3V, release 2, improvements to KIVA-3V, Los Alamos National Laboratory, Los Alamos, NM, Report No. LA-UR-99-915 10 (1999) 9452.

[27] C. Gotaas, P. Havelka, H.A. Jakobsen, H.F. Svendsen, M. Hase, N. Roth, B. Weigand, Effect of viscosity on droplet-droplet collision outcome: Experimental study and numerical comparison, Physics of fluids 19 (2007).

[28] M. Sui, M. Sommerfeld, L. Pasternak, Modelling the occurrence of bouncing in droplet collision for different liquids, Proc. of the ILASS2019-29th European Conf. on Liquid Atomization and Spray Systems, Paris, France (2019).

[29] A. Agarwal, Y. Wang, L. Liang, C. Naik, E. Meeks, The computational cost and accuracy of spray droplet collision models, WCX SAE World Congress Experience 237383 (2019).

[30] A. Agarwal, Machine learning models for prediction of droplet collision outcomes, arXiv preprint arXiv:2110.00167, (2021).

[31] W. Xu, T. Yang, P. Zhang, Data-driven Learning of Probabilistic Model of Binary Droplet Collision for Spray Simulation, Atomization and Sprays 36 (2026).

[32] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu, Lightgbm: A highly efficient gradient boosting decision tree, Advances in neural information processing systems 30 (2017).

[33] A. Tonda, Review of PySR: High-performance symbolic regression in Python and Julia, Springer, 2025.

[34] G.F. Smits, M. Kotanchek, Pareto-Front Exploitation in Symbolic Regression, in: U.-M.

O’Reilly, T. Yu, R. Riolo, B. Worzel (Eds.), Genetic Programming Theory and Practice II, Springer US, Boston, MA, 2005, pp. 283-299.

[35] S. Kim, D.J. Lee, C.S. Lee, Modeling of binary droplet collisions for application to interimpingement sprays, International Journal of Multiphase Flow 35 (2009) 533-549.

[36] S. Panzeri, C. Magri, L. Carraro, Sampling bias, Scholarpedia 3 (2008) 4258.

[37] C. Mundo, M. Sommerfeld, C. Tropea, Droplet-wall collisions: experimental studies of the deformation and breakup process, International journal of multiphase flow 21 (1995) 151- 173.

[38] C. Hu, S. Xia, C. Li, G. Wu, Three-dimensional numerical investigation and modeling of binary alumina droplet collisions, International Journal of Heat and Mass Transfer 113 (2017) 569-588.

[39] S. Suo, M. Jia, Correction and improvement of a widely used droplet–droplet collision outcome model, Physics of Fluids 32 (2020).