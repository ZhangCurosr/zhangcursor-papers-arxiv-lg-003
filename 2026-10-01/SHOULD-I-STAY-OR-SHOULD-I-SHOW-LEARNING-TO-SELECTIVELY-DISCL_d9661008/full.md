# SHOULD I STAY OR SHOULD I SHOW?LEARNING TO SELECTIVELY DISCLOSE INFORMATION

Carlotta Giacchetta<sup>1∗</sup> Alessando Bogani<sup>1</sup> Cesare Barbera<sup>2,1</sup> Giovanni De Toni<sup>3†</sup> Michele Caprio<sup>4</sup> Andrea Pugnana<sup>1,‡</sup> Andrea Passerini<sup>1,‡</sup>

<sup>1</sup>University of Trento, Trento, Italy <sup>2</sup>University of Pisa, Pisa, Italy

<sup>3</sup>ETH AI Center & ETH, Zurich, Switzerland ¨ <sup>4</sup>University of Warwick, Coventry, UK

name.surname @unitn.it, cesare.barbera@phd.unipi.it, giovanni.detoni@ai.ethz.ch, michele.caprio@warwick.ac.uk

## ABSTRACT

In many high-stakes settings, human decision-makers can acquire support information before making a decision. However, acquiring information is costly, and disclosure may fail to improve human decisions or may even impair them. We tackle this problem by studying selective disclosure, i.e., the problem of learning when to reveal support information to a human decision-maker under a budget constraint. We first show that the optimal policy is a threshold rule on the Value of Information (VoI), i.e., the expected reduction in human decision risk induced by disclosure. Since VoI is unknown in practice, we estimate the regime-specific human risks and bound the possible degradation of the resulting plug-in policy relative to lack of disclosure, as well as its regret relative to the optimal policy. Experiments on benchmark datasets show that selective disclosure outperforms both no disclosure and full disclosure, regardless of whether the support information is beneficial or harmful. Two user studies show that human-AI team performance can improve when disclosure is led by our learned policy and not human-selected, although this advantage varies across tasks. A counterfactual benchmark, which replaces participants’ predictions with a machine-learning prediction when disclosure occurs, suggests that these differences might depend on lower adherence to advice when the information is automatically provided rather than self-requested.

## 1 INTRODUCTION

In many decision-making tasks, humans predict uncertain outcomes with the aid of support information. Financial analysts may conduct costly research to inform investment decisions (Xiong & Yang, 2023), physicians may order invasive diagnostic tests (Kasivisvanathan et al., 2018), and zoologists may collect biological samples to identify endangered species (Li et al., 2017). Such information can improve decisions, but acquiring it is often costly (Saar-Tsechansky et al., 2009). Moreover, support information may also be redundant or even harmful when it introduces noise, confusion, or over-reliance on imperfect external feedback (Romeo & Conti, 2026). Information disclosure must therefore balance the cost of providing information against its potential effect on decision quality.

To tackle this challenge, we introduce Learning to Selectively Disclose (LSD), a decision-theoretic framework that learns when to show additional support information to a human decision-maker within a limited disclosure budget (Fig. 1). Rather than asking whether information is informative in general, LSD asks a more decision-relevant question: will revealing information improve thefinal decision? We formalize this benefit through the conditional Value of Information (VoI), defined as the reduction in human decision risk caused by disclosure.

More precisely, we show that the optimal policy prescribes disclosure for cases with the largest positive VoI. Because VoI is not observable, we cast its estimation as a causal inference problem:

![](images/c4c08636dc824568f0cd2d490b5bfb3ebd6a0a83380a51ac0d5e3bccebffb654.jpg)  
Figure 1: LSD pipeline. For each case $\mathbf { x } _ { i } ,$ a disclosure policy $d : \mathcal { X }  \{ 0 , 1 \}$ decides when to disclose support information S to a human expert H. If no support information is disclosed $( d ( \mathbf { x } _ { i } ) =$ 0), a human expert makes a decision using only baseline features. When support information is disclosed $( d ( \mathbf { x } _ { i } ) = 1 )$ , a human expert uses both baseline and support information.

we map VoI to the negative conditional average treatment effect of information disclosure on human decision loss and we establish conditions for its identification. Then, we derive guarantees for plugin policies relative to no disclosure and the optimal policy, and introduce a class-wise VoI estimator for classification under the 0 1 loss. Experiments on both synthetic and real-world data show that the learned policy identifies instances in which disclosure improves human decisions. We further evaluate our approach with two user studies on two different tasks, showing that our learned policy either outperforms or performs on par with human-selected disclosure. A counterfactual benchmark, which replaces human predictions with Machine Learning (ML) predictions for disclosed items, suggests that lower adherence rates to the ML predictions may partly account for these results.

## Our Contributions. Our main contributions are:

1. We present selective disclosure as a two-regime decision problem and show that the optimal budget-constrained policy discloses information for instances with the largest positive conditional Value of Information (VoI);

2. We formulate selective disclosure as a causal intervention on the decision-maker’s information, characterizing the conditional VoI as the causal reduction in human decision risk induced by disclosure. Then, we establish the assumptions under which VoI is identifiable;

3. We derive performance guarantees for plug-in disclosure policies in terms of their VoI estimation error, both relative to the no-disclosure baseline and relative to the optimal policy under the same budget constraint;

4. For classification under the 0-1 loss, we introduce a class-wise estimator of the regimespecific risks and demonstrate its effectiveness on synthetic and real-world datasets;

5. We run two user studies to compare human-AI team performance in classification tasks under policy-selected and human-selected disclosure.

## 2 PRELIMINARIES

We study a decision-making setting in which a human observes fixed baseline information in order to perform an action. Before selecting the action, the human may additionally receive support in formation to help its decision. More formally, let $\mathcal { X } \subseteq \mathbb { R } ^ { m }$ be the baseline information space and let $\mathcal { V } = \{ 1 , \dots , | \mathcal { V } | \}$ be a finite target space. A case consists of baseline information $\mathbf { X } \in { \mathcal { X } } .$ , a target $Y \in \mathcal { V }$ , and support information $S \in S$ , where S may, e.g., be an ML model prediction. Although the realization of S may differ across cases, we treat it as either disclosed in full or withheld. A binary variable $D \in \{ 0 , 1 \}$ denotes the disclosure regime. We define the action space and, throughout the paper, we set $\mathcal { A } = \mathcal { V }$ , so that the human action is a prediction of the target label. Let denote a population of human decision-makers and let π be a distribution over . For a decisionmaker $H \sim \pi$ , we denote by $A _ { H } ( 0 ) \in { \mathcal { A } }$ and $A _ { H } ( 1 ) \in { \mathcal { A } }$ the actions that H would take under no disclosure and disclosure, respectively. These actions may vary across decision-makers even for the same case. We further assume access to a dataset $\mathcal { T } = \bar { \{ } ( \mathbf { x } _ { i } , \mathcal { Y } _ { i } , a _ { H i } ( 0 ) , a _ { H i } ( 1 ) ) \} _ { i = 1 } ^ { n }$ , sampled from some unknown $P$ over $\mathcal { X } \times \mathcal { Y } \times \mathcal { A } \times \mathcal { A } .$ where $a _ { H i } ( 0 )$ and $a _ { H i } ( 1 )$ are decisions collected for case i under the two disclosure regimes. In particular, each case is assigned to (at least) two distinct decision-makers $H _ { 0 } , H _ { 1 } \sim \pi _ { ☉ }$ , with $H _ { 0 }$ operating under $D = 0$ and $\mathsf { \bar { H } } _ { 1 }$ operating under $D = 1$ Notably, the dataset contains observations from both regimes for each case, but it does not contain both potential actions for the same human-case episode. Thus, our objective is to learn a disclosure policy for a decision-maker drawn from the population π, rather than a policy for a fixed human.

## 3 LEARNING TO SELECTIVELY DISCLOSE

The objective of LSD is to determine when a decision-maker should be given access to support information under a budget constraint. We frame the problem from a decision-theoretic perspective. Given a non-negative loss function $\ell : \mathcal { A } \times \mathcal { y }  \mathbb { R } ^ { + }$ we define the regime-specific conditional risks:

$$
r _ { D } ( { \bf x } ) = \mathbb { E } \left[ \ell \big ( A _ { H } ( D ) , Y \big ) \ | \ { \bf X } = { \bf x } \right] , \qquad D \in { 0 , 1 , }\tag{1}
$$

where the expectation is taken over the target Y, the decision-maker $H \sim \pi$ , and any randomness in the human response, conditional on ${ \textbf { X } } = { \textbf { x } }$ Thus, $r _ { D } ( { \bf x } )$ measures the expected loss under disclosure regime $D$ for an average decision-maker drawn from $\pi ,$ rather than on any particular individual. Smaller values correspond to more reliable decisions, while larger values indicate greater expected error. We define the conditional Value of Information (VoI) as the expected reduction in decision risk induced by disclosure, namely:

$$
\mathrm { V o I } ( \mathbf { x } ) = r _ { 0 } ( \mathbf { x } ) - r _ { 1 } ( \mathbf { x } )\tag{2}
$$

with positive values indicating that revealing S improves decision quality.

Let be the set of measurable disclosure policies $d : \mathcal { X }  \{ 0 , 1 \}$ , where $d ( \mathbf { x } ) = 1$ indicates that the support information is disclosed. We denote the population disclosure risk with $R ( d ) =$ $\mathbb { E } _ { \mathbf { x } } [ ( 1 - d ( \hat { \mathbf { x } } ) ) r _ { 0 } ( \mathbf { x } ) + d ( \mathbf { x } ) r _ { 1 } ( \mathbf { x } ) ]$ ], i.e. the expected loss incurred when each instance is judged under the regime that d selects for it; in particular, $\bar { \boldsymbol { R } } ( 0 )$ and $R ( 1 )$ are the no- and full-disclosure baselines, obtained by never and always revealing S. Typically, access to more information is costly, thus we assume there is a budget level $B \in [ 0 , 1 ]$ that limits the expected disclosure rate. The Learning to Selectively Disclose problem can be formulated as follows:

$$
\operatorname* { m i n } _ { d \in { \mathcal { D } } } \quad R ( d ) \qquad \mathrm { s . t . } \quad { \mathbb { E } } _ { \mathbf { x } } [ d ( \mathbf { x } ) ] \leq B\tag{3}
$$

In the next subsections, we characterize the optimal disclosure policy for the LSD problem, formalize the causal estimation of VoI, provide theoretical guarantees for the corresponding plug-in approaches and then showcase how to estimate them when ℓ is the 0-1 loss, a common measure in human-AI collaboration literature (Ruggieri & Pugnana, 2025).

## 3.1 OPTIMAL DISCLOSURE POLICY

A first question to address is what an optimal policy for LSD looks like. We show that the optimal policy consists of a threshold rule over the VoI, as stated in the following Theorem:

Theorem 1 (Optimal Disclosure Policy). Assume that the cumulative distribution of $\operatorname { V o I } ( \mathbf { x } )$ is continuous. $L e t q _ { 1 - B }$ denote the $( 1 - B )$ )-quantile of $\operatorname { v o I } ( \mathbf { x } )$ and let $\lambda ^ { * } = \operatorname* { m a x } \{ 0 , q _ { 1 - B } \}$ . For any budget $B \in [ 0 , 1 ]$ , the optimal disclosure policy is the threshold rule:

$$
d ^ { * } ( \mathbf { x } ) = \left\{ \begin{array} { l l } { 1 } & { i f \quad \mathrm { V o I } ( \mathbf { x } ) \geq \lambda ^ { * } , } \\ { 0 } & { o t h e r w i s e . } \end{array} \right.\tag{4}
$$

Proof. We provide the proof in Appendix A.1.

Theorem 1 shows that selective disclosure admits an intuitive interpretation: support information is revealed only when it is expected to benefit the decision-maker, i.e., when the expected risk reduction is positive and exceeds a budget-dependent threshold. However, $\operatorname { V o I } ( \mathbf { x } )$ is not directly observable, since a decision-maker either receives $S$ or does not. As a result, the two risks defining $\operatorname { V o I } ( \mathbf { x } )$ cannot be jointly observed for the same decision instance. Estimating $\operatorname { V o I } ( \mathbf { x } )$ is therefore naturally a causal inference problem. In the following section, we formalize the corresponding causal framework and state the assumptions under which $\operatorname { V o I } ( \mathbf { x } )$ is identifiable from observed data.

## 3.2 A CAUSAL INFERENCE INTERPRETATION

Since $\operatorname { V o I } ( \mathbf { x } )$ is never observed for the same human decision-maker, a natural way to handle its estimation is resorting to the Neyman-Rubin potential outcomes framework (Rubin, 1974). However, our framework differs from the canonical treatment-allocation setting in an important aspect. In standard policy learning (Kitagawa & Tetenov, 2018), a treatment is assigned to a unit and directly affects that unit’s outcome. In LSD, instead, the intervention acts on a decision-maker: disclosure changes the information available to a human who must make a decision about a separate case. The case outcome $Y$ is not affected by disclosure; what may change is the human action $A _ { H } ( D )$ and, consequently, the decision loss $\ell ( \dot { \boldsymbol { A } } _ { H } ( \boldsymbol { D } ) , \boldsymbol { Y } )$ . Thus, $\ell ( \bar { A } _ { H } ( 0 ) , \bar { Y } )$ and $\ell ( A _ { H } ( 1 ) , Y )$ are the two potential outcomes of interest, and the causal effect of our interest is not the effect of an intervention on the case itself, but the effect of disclosing support information on the quality of a human decision. More precisely, for a fixed decision-maker $h ,$ , we define the human-specific conditional effect of disclosure on decision loss as $\tau _ { h } ( { \bf x } ) = \mathbb { E } [ \ell ( A _ { h } ( 1 ) , Y ) - \ell ( A _ { h } ( 0 ) , Y ) ^ { \top } | X = { \bf x } , H = h ]$ . Because our objective is to learn a policy for a future decision-maker drawn from a target population π, we consider the population-average effect $\tau _ { \pi } ( \mathbf { x } ) = \mathbb { E } _ { H \sim \pi } [ \tau _ { H } ( \mathbf { x } ) ]$ and note that $- \tau _ { \pi } ( \mathbf { x } ) = \mathrm { V o I } ( \mathbf { x } )$ Hence, $\operatorname { V o I } ( \mathbf { x } )$ measures the causal effect of disclosing information on the quality of a decision made about a case with covariates x, averaged over decision-makers from $\pi .$

Following causal inference practice (Nogueira et al., 2022), we now investigate under which assumptions $\operatorname { V o I } ( \mathbf { x } )$ can be retrieved from empirical data. We assume<sup>1</sup>: (A1) Unconfoundedness, $i . e .$ , conditional on baseline covariates X, the disclosure decision is as good as random with respect to the potential regime-specific risks (Rosenbaum $\&$ Rubin, 1983); (A2) Positivity, $i . e .$ , for all covariate profiles in the population, we observe both disclosure regimes with non-zero probability (Rubin, 1978); (A3) Consistency, $i . e . ,$ if a decision episode is observed under a specific regime, then the observed loss is equal to the corresponding potential loss (Cole & Frangakis, 2009); (A4) Stable Unit Treatment Value Assumption (SUTVA), $i . e . ,$ , for each decision episode, the potential action of the human decision-maker depends only on the information disclosed to that decisionmaker in that episode (Rubin, 1981) and $( A 5 )$ Exchangeability of Human Decision-Makers, $i . e . ,$ the decision-makers are drawn independently of the covariates and of the regime. Under assumptions $A I { \cdot } A 5 ,$ , we can estimate our causal estimand from data:

Proposition 1 (Identifiability of the Value of Information). Under Assumptions $A l { - } A 5 ,$ , the regimespecific potential risks are identifiable from observed data. Let $\ell ^ { \mathrm { o b } \dot { \mathrm { s } } } \bigl ( A _ { H _ { D } } ( D ) , Y \bigr ) \ = \ \mathsf { \Gamma } ( 1 \ -$ $D ) \ell ( A _ { H _ { 0 } } ( 0 ) , Y ) + D \ell ( A _ { H _ { 1 } } ( 1 ) , Y )$ denote the loss observed in a decision episode, where $H _ { D }$ is the decision-maker assigned under regime $D .$ Then, for $d \in \{ 0 , 1 \}$ , the Value of Information

$$
\operatorname { V o I } ( \mathbf { x } ) = r _ { 0 } ( \mathbf { x } ) - r _ { 1 } ( \mathbf { x } ) = \mathbb { E } { \bigl [ } \ell { \bigl ( } A _ { H } ( 0 ) , Y { \bigr ) } - \ell { \bigl ( } A _ { H } ( 1 ) , Y { \bigr ) } \mid \mathbf { X } = \mathbf { x } { \bigr ] }
$$

is identifiable as

$$
\operatorname { V o I } ( \mathbf { x } ) = \mathbb { E } \left[ \ell ^ { \mathrm { o b s } } \big ( A _ { H } ( D ) , Y \big ) \mid \mathbf { X } = \mathbf { x } , D = 0 \right] - \mathbb { E } \left[ \ell ^ { \mathrm { o b s } } \big ( A _ { H } ( D ) , Y \big ) \mid \mathbf { X } = \mathbf { x } , D = 1 \right] .
$$

## Proof. We provide the proof in Appendix A.2

Proposition 1 reduces the $\operatorname { V o I } ( \mathbf { x } )$ to two identifiable, regime-specific conditional risks. Estimating it becomes a standard CATE estimation problem, for which any estimator from the causal inference literature can be used. The simplest one (T-learner Kunzel et al. (2019)) fits a risk model per regime,¨ $\hat { r } _ { 0 }$ and $\hat { r } _ { 1 }$ , and sets $\widehat { \mathrm { V o I } } ( \mathbf { x } ) = \hat { r } _ { 0 } ( \mathbf { x } ) - \hat { r } _ { 1 } ( \mathbf { x } )$ . Still, accurate prediction of $\operatorname { V o I } ( \mathbf { x } )$ , is not sufficient for effective disclosure: the policy depends on the sign (Frauen et al., 2025) and ranking (Arno et al., 2026) of ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ , and estimation errors can cause the policy to disclose information where it is harmful. In the next subsection, we quantify how such errors affect the learned policy’s performance.

## 3.3 GUARANTEES FOR THE PLUG-IN POLICY

Since $\operatorname { V o I } ( \mathbf { x } )$ is identifiable from data collected under the two regimes (Proposition 1), one can threshold an estimate ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ in its place, obtaining what we call a plug-in policy. Let us denote with ${ \hat { d } } ( \mathbf { x } ) = \mathbb { 1 } \{ { \widehat { \mathrm { V o I } } } ( \mathbf { x } ) \geq { \hat { \lambda } } \}$ the plug-in disclosure policy, with threshold $\hat { \lambda } \ge 0$ for a given budget

B, with $\hat { B } = \mathbb { E } _ { \mathbf { x } } [ \hat { d } ( \mathbf { x } ) ]$ its realized disclosure rate and with $\varepsilon ( \mathbf { x } ) = \widehat { \mathrm { V o I } } ( \mathbf { x } ) - \mathrm { V o I } ( \mathbf { x } )$ the estimation error of ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ . Thresholding ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ raises two questions: whether the resulting policy can be worse than no-disclosure and how far it can fall short of the optimal policy.

First, we address whether deploying the plug-in policy can achieve higher risk than simply never disclosing. We stress that this is possible in principle because disclosure can be harmful on average $( \mathbb { E } _ { \mathbf { x } } [ \mathrm { V o I } ( \mathbf { x } ) ] < 0 )$ and a policy that mis-ranks instances can spend its budget where disclosure is harmful. In the following, we quantify how large this degradation can be:

Theorem 2 (No-disclosure degradation). Given a non-negative lossfunction $\ell : \mathcal { A } \times \mathcal { y }  \mathbb { R } ^ { + }$ , for any estimator ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ such that $\| \varepsilon \| _ { L ^ { 2 } ( p ) } < \infty$ and any $\hat { \lambda } \geq 0 ,$ , thefollowing holds:

$$
R ( \hat { d } ) - R ( 0 ) \leq \sqrt { \hat { B } } \| \varepsilon \| _ { L ^ { 2 } ( p ) } - \hat { \lambda } \hat { B } .\tag{5}
$$

Proof. We provide the proof in Appendix A.3.

□

The bound is governed by the VOI(x) estimation error and by the region on which the policy intervenes: for bounded $\| \varepsilon \| _ { L ^ { 2 } ( p ) }$ , it converges to zero as the realized disclosure rate $\hat { B }$ tends to zero but it is positive whenever $\| \varepsilon \| _ { L ^ { 2 } ( p ) } > \hat { \lambda } \sqrt { \hat { B } }$ . Hence, the plug-in policy can perform worse than never disclosing, but only by a bounded amount (see Appendix $\bar { \mathbf { A } } . 3 )$ .

We now present our Oracle regret bound. This addresses how much is lost, relative to the optimal policy, by thresholding an estimate rather than the true $\operatorname { V o I } ( \mathbf { x } )$ . Here we additionally require the plug-in rule to use the threshold that Theorem 1 prescribes for ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ , i.e., the rule that is optimal according to its own estimate of the risks:

Theorem 3 (Oracle regret). Given a non-negative loss function $\ell : \mathcal { A } \times \mathcal { y }  \mathbb { R } ^ { + }$ , let $d ^ { * }$ be the optimal policy of Theorem 1 and $B ^ { * } = \mathbb { E } _ { \mathbf { x } } \mathbf { \bar { [ } } d ^ { * } ( \mathbf { x } ) \mathbf { ] }$ ]. Assume that the cumulative distribution of ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ is continuous, let $\hat { q } _ { 1 - B }$ be its $( 1 - B )$ -quantile and $\hat { \lambda } = \operatorname* { m a x } \{ 0 , \hat { q } _ { 1 - B } \}$ . Then, it holds that

$$
R ( \hat { d } ) - R ( d ^ { * } ) \leq \sqrt { B ^ { * } + \hat { B } } \| \varepsilon \| _ { L ^ { 2 } ( p ) } .\tag{6}
$$

Proof. We provide the proof in Appendix A.4.

The Oracle-regret bound of Eq. (6) shows that the plug-in rule is $O ( \| \varepsilon \| _ { L ^ { 2 } ( p ) } )$ -competitive with an oracle knowing the true risks. Such a bound scales with an upper bound on the mass of the disagreement region, $\begin{array} { r } { \mathbb { E } _ { \mathbf { x } } [ | d ^ { * } ( \mathbf { x } ) - \hat { d } ( \mathbf { x } ) | ] \leq B ^ { * } + \hat { B } } \end{array}$ . The bound can therefore be loose when the two policies agree away from the threshold: estimation errors on instances that both policies disclose, or both withhold, do not affect their relative performance.

## 3.4 ESTIMATING THE VALUE OF INFORMATION FOR 0-1 LOSS

As shown in Theorem 1, learning disclosure policies requires estimating the regime-specific risks $r _ { 0 } ( \mathbf { x } )$ and $r _ { 1 } ( \mathbf { x } )$ and then thresholding their difference. In what follows, we focus on the 0-1 loss, which is widely used in human-AI collaboration to model the error incurred by the human decision maker (Madras et al., 2018; Ruggieri & Pugnana, 2025). The 0 1 loss is defined as:

$$
\ell ( A ( D ) , Y ) = \mathbb { 1 } \{ A ( D ) \neq Y \} ,\tag{7}
$$

Under this loss choice, the expected risk becomes:

$$
r _ { D } ( \mathbf { x } ) = \mathbb { E } [ \mathbb { 1 } \{ A ( D ) \neq Y \} ] , \qquad D \in \{ 0 , 1 \}\tag{8}
$$

Thus, estimating Eq. (8) reduces to a standard classification task, where the goal is to predict whether the human makes mistakes, as a T-learner would do with a single scalar risk model per regime (Kunzel et al., 2019). However, humans do not err uniformly across classes: some classes¨ are systematically confused with others, and disclosure alters these error patterns differently for each class. A scalar risk model condenses all these patterns into a single error probability, blurring the class-specific effects on which the VoI depends (see Appendix E.1). We therefore decompose the regime-specific risk class-wise, relying on the following factorization of the 0 1 risk:

Proposition 2 (Class-wise decomposition). Let us consider $\ell ( A ( D ) , Y ) = \mathbb { 1 } \{ A ( D ) \neq Y \}$ . Then $r _ { D } ( { \bf x } )$ can be written as:

$$
r _ { D } ( \mathbf { x } ) = \sum _ { y \in \mathcal { Y } } \underbrace { { \mathbb { P } } ( Y = y \mid \mathbf { X } = \mathbf { x } ) } _ { p _ { y } } \underbrace { { \mathbb { P } } ( A ( D ) \neq y \mid \mathbf { X } = \mathbf { x } , Y = y ) } _ { q _ { D , y } } .\tag{9}
$$

Proof. We provide the proof in Appendix A.5.

This decomposition suggests a separate estimation strategy for both $p _ { y }$ and $q _ { D , y }$ . More precisely, we approximate $p _ { y }$ using a probabilistic classifier $f : \mathcal { X } \to \Delta ^ { | \mathcal { V } | }$ , and $q _ { D , y }$ via a multi-head architecture with heads and a shared backbone, where any y-th head is a binary classifier $g _ { D , y } : \mathcal { X }  [ 0 , 1 ]$ The resulting risk estimator is then $\begin{array} { r } { \widehat { r } _ { D } ( \mathbf { x } ) = \sum _ { y \in \mathcal { Y } } f _ { y } ( \mathbf { x } ) \cdot g _ { D , y } ( \mathbf { x } ) } \end{array}$ , where $f _ { y }$ refers to the $y - \mathrm { t h }$ output of the probabilistic classifier. The estimated VoI is obtained by plugging $\widehat { r } _ { D } ( { \bf x } )$ into Eq. (2). We refer to this estimator as ClassWise, which specializes the T-learner to a structured class-wise risk model.

## 4 EXPERIMENTAL EVALUATION

In this section<sup>2</sup>, we address the following research questions:

Q1 Does ClassWise learn effective policies?

Q2 Does the ClassWise policy improve human-AI team performance?

Q3 How do participants interact with the ClassWise policy?

## 4.1 EXPERIMENTAL SETTINGS

Datasets. We consider both synthetic and real data. We generate 10000 synthetic data samples using the standard make classification function of sklearn library, considering both a binary $( | \mathscr { V } | = 2 )$ and a multiclass task $( | \mathcal { V } | = 4 )$ . We emulate selective disclosure by treating a subset of features as baseline information and the remaining ones as support information. Then, we simulate human decisions in the two regimes with logistic regression models. We also employ two different real datasets: Email (Rebeka Toth, 2025) and ImageNet-16H (Steyvers et al., 2022). (i) Email contains 1000 emails that can be classified as either legitimate or fraudulent. For this dataset, we take human predictions from (Bogani et al., 2026), who provide for each email a human prediction made either without any assistance $( D = 0 )$ or with the support of an ML model $( D = 1 )$ . (ii) ImageNet-16H contains 1200 noisy images that must be assigned to one of 16 categories. Human decisions for this dataset come from two behavioural experiments: the ones with no assistance come from (Steyvers et al., 2022) $( D = 0 )$ , while predictions with support information come from (Straitouri & Rodriguez, 2024) $( D = 1 )$ , where participants make their decision after observing a set of conformal predictions $( \alpha = . 0 5 ) ^ { 3 }$ . We provide further details in Appendix C.

Methods. For Q1, we evaluate our approach (ClassWise) against the following baselines: (i) an uncertainty-based policy (Confidence) that thresholds $u ( \mathbf { x } ) = 1 - \operatorname* { m a x } _ { y } f ( \mathbf { x } ) _ { y }$ instead of ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ , at the same $1 - B$ quantile (Pugnana et al., 2024); (ii) a policy that randomly assigns disclosure for each instance (Random); (iii) a full disclosure baseline (FullDisc); (iv) and a no-disclosure baseline (NoDisc). For Q2 and Q3, we compare ClassWise against human participants who decide by themselves, item by item, whether to request the support information.

Metrics and Evaluation. For both Q1 and Q2 we evaluate human decision performance using testset accuracy (Acc). For Q3 we also measure participants’ advice adherence (Adh), i.e., the proportion of trials in which support information is available and the participant’s classification matches such information (this amounts to selecting the ML-suggested label for Email and any class included in the prediction set for ImageNet-16H). We evaluate Adherence only on policy-selected items, ensuring that comparisons between MP and HP are based on the same set of items. User studies data are analyzed with logistic mixed-effects models with random intercepts for participants and items (see Appendix E.2 and Appendix E.3 for full analyses details).

Q1 setup. For Q1, (i) we perform a $7 0 / 1 0 / 2 0$ split of the data into training, calibration, and test sets; (ii) we carve out a validation split from the training data (10%) and select hyperparameters by grid search<sup>4</sup>; (iii) we calibrate the disclosure threshold as the 1 B quantile of $\hat { \mathrm { V o I } } ( { \bf x } )$ on the calibration set for each $B \in \{ . 1 , . 2 , . 3 , . 4 , . 5 , . 6 , . 7 , . 8 , . 9 \}$ and (iv) we report the human accuracy on the test set, averaged over five seeds that vary model initialization<sup>5</sup>.

User studies. To answer Q2 and Q3, we run two user studies (545 total participants) in which each participant classifies 20 items sampled from the test set of one of the two real datasets, i.e., Email or ImageNet-16H. We manipulate two between-subjects variables, with two conditions each: (i) Support is either Human-Policy (HP), where participants decide themselves when to request the support information, (i.e., the ML’s model prediction, and specifically the predicted label for emails and the conformal prediction set at $\alpha = . 0 5$ for images), or Machine-Policy (MP), where our ClassWise policy decides when to provide it; (ii) Budget is either Low (LB) or High (HB), making the support information available for at most 6 or 14 of the 20 items (30% and 70%). Participants in HP are told they need not exhaust their budget and are incentivized to request information only when needed, so that their policy is broadly comparable to the learned one. The 20-item samples are stratified so that $5 / 9$ emails and 6/14 images come from the items the policy would select for disclosure under LB/HB, matching its realized disclosure rates in the original test sets, and all four conditions draw from the same pool<sup>6</sup>. Similarly to Scenario 1 in (Palomba et al., 2025), the ML model’s prediction is retrospectively available for every case. We therefore report, as an exploratory analysis, a counterfactual benchmark (MPCFT), obtained by (i) replacing MP participants’ responses with the underlying ML model prediction whenever information is disclosed and (ii) retaining their observed responses otherwise. Thus, MPCFT represents the accuracy that would have been achieved had the ML prediction been enforced as the final decision whenever it was disclosed. See Appendix D for further details on the user studies.

## 4.2 EXPERIMENTAL RESULTS

Q1: ClassWise is effective on all datasets and outperforms simple baselines. Fig. 2 reports the results for our experiments on all four datasets. For Synth we can see that ClassWise dominates the Confidence baseline and Random at every budget level. Interestingly, ClassWise already exceeds FullDisc accuracy at $B = . 3 0$ in the SynthBin $( A c c \approx 0 . 8 3 )$ and at $B = . 5 0$ in the SynthMulti (Acc 0.63), suggesting that disclosing information for only a small fraction of instances suffices to recover the performance of full disclosure.

For Email, disclosure is harmful on average: NoDisc outperforms FullDisc (Acc .79 vs Acc .76). ClassWise is the best-performing policy for all B, peaking at $B = . 3 0 \ : ( A c c \approx . 8 1 )$ Beyond this point, the policy increasingly discloses on instances where it wrongly estimates support to be beneficial, so accuracy decreases and plateaus slightly below NoDisc, which however remains within its confidence band. As anticipated by Theorem 2, the plug-in policy can thus perform worse than never disclosing, although only by a bounded amount. Unlike ClassWise, the Confidence baseline never abstains: its accuracy decreases monotonically with the budget, never exceeds NoDisc, and collapses into FullDisc at B = 1. The reason is that uncertainty measures how hard the case is, not how much disclosure helps: there is no mechanism to abstain when disclosure hurts.

For ImageNet-16H, we observe that on average disclosing support information is helpful: FullDisc achieves Acc .85 while NoDisc Acc .77. Interestingly, ClassWise surpasses the full disclosure baseline already at $B = . 7 0$ and achieves the highest accuracy of .86 at $B = . 9 0$ . Moreover, we observe that ClassWise is always different from Random, suggesting that the policy can learn when support information is useful. Confidence is a stronger baseline here, but it only matches ClassWise at $B \geq . 8 0$ , where nearly every instance is disclosed and little is left to choose. Whenever the budget is binding, ClassWise is always ahead.

![](images/c4cdea9c85fe70921aab2433ac0b28014faaee8ad6f3ecf6ca0d10801e9ca6f9.jpg)

![](images/079ad881c7df8455c7978ae8b47046a536063188e437f7b1258bab7ef79be620.jpg)

![](images/00a5d14782be366b614d77b7c5ec9e55ed6e3e4165e1b08d131c550aea69fef1.jpg)  
Figure 2: Budget-accuracy curves for the four considered datasets. We report average mean values and 95% confidence intervals over five seeds.

![](images/f5bdb8efd6d258291c7b2c002055215777bdb689a1eba422dbdc39181299f167.jpg)  
(a) Accuracy

![](images/3515c66ff1b06c0da28b4260abb1ebf5329f3b678ae68bf63a4b04dc6013db52.jpg)  
(b) Adherence  
Figure 3: User studies results (Q2 and Q3). (a) Participants’ accuracy (Acc) and (b) adherence $( A d h )$ on the AI advice, by Budget (LB/HB) and Support (HP/MP/MPCFT) condition, on Email (left) and ImageNet-16H (right). Error bars are 95% confidence intervals across participants.

Moreover, we ablate the risk estimator considering two variants that do not exploit the classwise decomposition of Proposition 2, i.e., (a) an approach that directly estimates both $r _ { 0 }$ and $r _ { 1 }$ separately (T-Learner) and (b) a joint estimator for both $r _ { 0 }$ and r<sub>1</sub> (S-Learner). ClassWise outperforms both variants at every budget on Synth data, and attains the highest peak accuracy on the real datasets. The gap is largest on Email, where disclosure is harmful: S-Learner shrinks ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ towards zero, estimates a positive VoI almost everywhere and thus never abstains, collapsing to $\mathtt { F u l l 1 D i s c }$ at high budgets, whereas ClassWise withholds support information on a larger fraction of the instances than T-Learner. Full ablation and budget analyses are in Appendix E.1.

Q2: The learned policy matches or outperforms self-selection. We report the two user-study results on human accuracy (Acc) in Fig. 3a. When considering Support, in the ImageNet-16H task accuracy is significantly higher in MP than in HP ( : 0.80 0.11 vs : $0 . 7 7 \pm 0 . 1 3 , p =$ .025). Instead, in the Email task MP and HP are not significantly different ( : $0 . 8 2 \pm 0 . 1 2$ vs : $0 . 8 3 \pm 0 . 1 3 , p = . 7 6 2 )$ . When considering Budget, in the ImageNet-16H task accuracy is significantly higher in HB than in LB (0.83 0.10 vs 0.74 0.13, p < .001), whereas the two conditions do not significantly differ in the Email task (0.84 0.12 vs 0.81 0.12, $p = . 0 5 6 )$ The advice-adherence results reported next may help account for why MP outperforms HP in the ImageNet-16H but not in the Email task. Appendix E.2 reports full results, plus analyses of advice correctness, human confidence, and alignment between ClassWise and human disclosure.

Q3: The learned policy is more beneficial whenever humans follow the advice. We report results on advice-adherence (Adh) in Fig. 3b. When considering Support, in the ImageNet-16H task adherence is not significantly different between MP and HP ( : 0.91 0.11 vs : $0 . 9 3 \pm 0 . 1 8 .$ $p = . 0 8 4 )$ , while in the Email task it is significantly lower in MP than in HP ( : $0 . 8 5 \pm 0 . 1 6$ vs $\mathfrak { Q } \mathrm { : ~ } 0 . 9 6 \pm 0 . 1 5 , p < . 0 0 1 )$ . When considering Budget, in both studies adherence does not differ significantly between HB and LB (ImageNet-16H: $p = . 2 6 0 \mathrm { : }$ Email: $p = . 6 3 6 )$ , not even in interaction with Support (ImageNet-16H: $p = . 3 9 2 ;$ Email: $p = . 1 2 6 )$ .

Accuracy in MPCFT (Fig. 3a) suggests that lower advice-adherence may attenuate the benefits of selective disclosure. In the ImageNet-16H task, where adherence under MP is already high, accuracy in MPCFT is descriptively but not significantly higher than in MP $( \odot \colon 0 . 8 2 \pm 0 . 1 0 ~ \mathrm { v s }$ . : $0 . 8 0 \pm 0 . 1 1 ; p = . 0 5 8 )$ , while it is significantly higher than in HP ( : $0 . 7 7 \pm 0 . 1 3 ; p < . 0 0 1 )$ . In Email, where adherence under MP is lower, MPCFT accuracy is significantly higher than in MP $( \odot ; 0 . 8 5 \pm 0 . 1 0 \mathrm { v s . } \ \Sigma _ { \sharp } ^ { 3 } ; \ 0 . 8 2 \pm 0 . 1 2 ; p = . 0 0 4 )$ and descriptively, but not significantly, higher than in $\mathrm { H P } \left( \mathfrak { B } ; 0 . 8 3 \pm 0 . 1 3 ; p = . 1 9 6 \right)$ . This cross-task pattern suggests that lower adherence may partly contribute to the absence of an MP advantage in the Email task, as would be expected given that the disclosed ML advice is correct on most trials. Additional results are reported in Appendix E.3.

Overall, these results suggest that deploying selective disclosure requires attending not only to which information is valuable, but also to how decision-makers receive information they did not request. Indeed, the pattern we observe is consistent with psychological evidence suggesting that self-produced outcomes are evaluated more favorably than comparable outcomes produced by ex ternal sources (Botti et al., 2023; Enisman et al., 2021).

## 5 RELATED WORK

Our paper builds upon related work on human-AI decision-making and policy learning under budget constraints. We provide an extended overview of these and related topics in Appendix F.

Human-AI Decision-Making. Learning to Defer (Madras et al., 2018) and follow-up work (Mozannar & Sontag, 2020; Verma & Nalisnick, 2022) allocate each instance to the model or to the human, and causal reasoning has so far been used to evaluate such systems (Palomba et al., 2025); in LSD the human always decides, and the policy allocates information. Other work designs the form of support, e.g., conformal prediction sets (De Toni et al., 2024) or explanations (Schemmer et al., 2023), while Noorani et al. (2026) let the AI refine a human expert’s prediction set, recovering labels the human missed without degrading correct human judgments. Noti & Chen (2023) and Ma et al. (2023) are closely related to our work, as they show support when the AI is predicted to outperform the human. Similarly, Bhatt et al. (2025) learn personalized disclosure policies online. We differ by casting selective disclosure as a causal inference problem: disclosure is an intervention on the decision-maker’s information, and the VoI is never observed under both regimes. To the best of our knowledge, LSD is the first to establish when VoI is identifiable from data collected under the two regimes (Section 3.2) and to show that VoI is the required estimand to build budget-constrained disclosure policies with guarantees (Theorems 2 and 3).

Policy Learning. Theorem 1 is the disclosure analogue of optimal policy learning (Kitagawa & Tetenov, 2018) under a capacity constraint, where the optimal rule thresholds the conditional average treatment effect at the larger of zero and a budget-dependent quantile (Bhattacharya & Dupas, 2012; Cerulli, 2026). The difference is where the treatment acts: disclosure is applied to the decisionmaker, while the outcome is the correctness of a decision about a case, so SUTVA is stated over decision episodes and identification requires decision-makers to be exchangeable across regimes.

## 6 CONCLUSION

In this work, we introduced Learning to Selectively Disclose (LSD), a decision-theoretic framework for deciding when to provide support information to a human decision-maker under a budget con straint. We showed that the optimal policy thresholds the Value of Information, revealing informa tion only where the expected benefit is positive and exceeds a budget-dependent threshold. Since the VoI is not directly observable, we gave conditions under which it is identifiable from data collected under the two regimes. We then studied plug-in approaches, for which we bounded the estimation error relative to no disclosure and the regret relative to the optimal policy. For classification tasks under the 0 1 loss, we further decomposed the human risk class-wise and used this decomposition to build a VoI estimator, which proved effective on both synthetic and real benchmarks. Two user studies further indicate that ClassWise-selected disclosure can improve human-AI team accuracy relative to human-selected disclosure, although its effectiveness varies across tasks. This variability may partly reflect how participants respond to advice: in some contexts, adherence may be lower for automatically provided than for comparable self-requested information, and a counterfactual benchmark replacing participants’ predictions with ML ones when disclosed suggests that greater adherence would have yielded higher overall accuracy under ClassWise-selected disclosure.

Limitations and Future Works. Our framework treats the support information S as an indivisible block: the policy decides whether to disclose, never what to disclose. Studying which parts of the information to reveal is left for future work. Also, our user studies do not identify which task characteristics reduce participants’ advice-adherence nor establish a causal relationship between adherence and policy effectiveness, as they were not designed to address these questions. Future work will investigate these aspects with dedicated user studies.

## AI USE STATEMENT

In this work, we used generative AI tools for code support and polishing the writing. Regarding the former, the code was manually reviewed by two authors. Regarding the latter, we have used generative AI tools to provide feedback on the presentation’s quality, with the goal of avoiding concerns that are due solely to inaccuracies in our writing. ChatGPT Astra Pro was used to assist with proofreading and to improve the clarity and presentation of the manuscript. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

The research protocol of our user studies was determined as posing no risk to participants’ wellbeing or rights and therefore as not requiring full ethical review by the Research Ethics Committee of [institution omitted to preserve anonymity] ([protocol number omitted to preserve anonymity]).

## REPRODUCIBILITY STATEMENT

The main text describes the problem formulation, the assumptions for identifying the VoI, the ClassWise estimator, the baselines, the datasets, the evaluation protocol (data splits, threshold calibration, budget grid, and seeds), and the design and analysis of the user studies. Complete proofs are provided in the appendix, together with additional experimental results and details on the user studies design. The implementation and experiment scripts used to produce our results are made available through an anonymized repository, which also contains the data collected in the user studies and the corresponding statistical analysis scripts.

## ACKNOWLEDGMENTS

The authors acknowledge the CINECA award under the ISCRA initiative for the availability of highperformance computing resources and support. This work was funded by the European Union. The views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union, the European Health and Digital Executive Agency (HaDEA) or the European Research Executive Agency. Neither the European Union nor the granting authority can be held responsible for them. Grant Agreement no. 101120763 - TANGO. Michele Caprio gratefully acknowledges support from the Prob AI Hub and the London Mathematical Society.

## REFERENCES

Henri Arno, Dennis Frauen, Emil Javurek, Thomas Demeester, and Stefan Feuerriegel. Ranklearner: Orthogonal ranking of treatment effects. CoRR, abs/2602.03517, 2026.

Susan Athey and Stefan Wager. Policy learning with observational data. Econometrica, 89(1): 133–161, 2021. doi: https://doi.org/10.3982/ECTA15732. URL https://onlinelibrary. wiley.com/doi/abs/10.3982/ECTA15732.

Umang Bhatt, Valerie Chen, Katherine M. Collins, Parameswaran Kamalaruban, Emma Kallina, Adrian Weller, and Ameet Talwalkar. Learning personalized decision support policies. In AAAI, pp. 14203–14211. AAAI Press, 2025.

Debopam Bhattacharya and Pascaline Dupas. Inferring welfare maximizing treatment assignment under budget constraints. Journal of Econometrics, 167(1):168–196, 2012. ISSN

0304-4076. doi: https://doi.org/10.1016/j.jeconom.2011.11.007. URL https://www. sciencedirect.com/science/article/pii/S0304407611002697.

Alessandro Bogani, Nicola Debole, Emanuele Marconato, Andrea Pugnana, Katya Tentori, and Andrea Passerini. Are concept bottleneck models effective as decision-support systems?, 2026. URL https://arxiv.org/abs/2608.25581.

Elizabeth Bondi, Raphael Koster, Hannah Sheahan, Martin J. Chadwick, Yoram Bachrach, A. Taylan Cemgil, Ulrich Paquet, and Krishnamurthy Dvijotham. Role of human-ai interaction in selective prediction. In AAAI, pp. 5286–5294. AAAI Press, 2022.

Simona Botti, Sheena S Iyengar, and Ann L McGill. Choice freedom. Journal of Consumer Psychology, 33(1):143–166, 2023.

Violet A Brown. An introduction to linear mixed-effects modeling in r. Advances in Methods and Practices in Psychological Science, 4(1):2515245920960351, 2021.

Zana Buc¸inca, Siddharth Swaroop, Amanda E. Paluch, Susan A. Murphy, and Krzysztof Z. Gajos. Towards optimizing human-centric objectives in ai-assisted decision-making with offline reinforcement learning. CoRR, abs/2403.05911, 2024.

Yuzhou Cao, Hussein Mozannar, Lei Feng, Hongxin Wei, and Bo An. In defense of softmax parametrization for calibrated and consistent learning to defer. In NeurIPS, 2023.

Aldo Gael Carranza and Susan Athey. Robust offline policy learning with observational data from multiple sources. In AISTATS, volume 258 of Proceedings of Machine Learning Research, pp. 4897–4905. PMLR, 2025.

Giovanni Cerulli. Optimal policy learning with observational data in multi-action scenarios: estimation, risk preference, and potential failures. Int. J. Data Sci. Anal., 22(1):164, 2026.

Mohammad-Amin Charusaie, Amirmehdi Jafari Fesharaki, and Samira Samadi. Defer-and-fusion: Optimal predictors that incorporate human decisions. In 5th Workshop on practical ML for limited/low resource settings, 2024. URL https://openreview.net/forum?id= Pg51u5YboV.

Stephen R Cole and Constantine E Frangakis. The consistency statement in causal inference: a definition or an assumption? Epidemiology, 20(1):3–5, 2009.

Jesse C. Cresswell, Yi Sui, Bhargava Kumar, and Noel Vouitsis. Conformal prediction sets improve ¨ human decision making. In ICML, volume 235 of Proceedings of Machine Learning Research, pp. 9439–9457. PMLR / OpenReview.net, 2024.

Giovanni De Toni, Nastaran Okati, Suhas Thejaswi, Eleni Straitouri, and Manuel Gomez Rodriguez. Towards human-ai complementarity with prediction sets. In NeurIPS, 2024.

Maya Enisman, Hila Shpitzer, and Tali Kleiman. Choice changes preferences, not merely reflects them: A meta-analysis of the artifact-free free-choice paradigm. Journal ofPersonality and Social Psychology, 120(1):16, 2021.

Yizirui Fang and Eric T. Nalisnick. Learning to defer with an uncertain rejector via conformal prediction. Trans. Mach. Learn. Res., 2026, 2026.

Carlos Fernandez-Lor´ ´ıa and Foster Provost. Causal decision making and causal effect estimation are not the same. . . and why it matters. INFORMS Journal on Data Science, 1(1):4–16, 2022. doi: 10.1287/ijds.2021.0006. URL https://doi.org/10.1287/ijds.2021.0006.

Dennis Frauen, Valentyn Melnychuk, Jonas Schweisthal, Mihaela van der Schaar, and Stefan Feuer riegel. Treatment effect estimation for optimal decision-making. In NeurIPS, 2025.

Ruijiang Gao and Mingzhang Yin. Confounding-robust deferral policy learning. In AAAI, pp. 14238–14246. AAAI Press, 2025.

Wenbo Gong, Sebastian Tschiatschek, Sebastian Nowozin, Richard E. Turner, Jose Miguel´ Hernandez-Lobato, and Cheng Zhang. Icebreaker: Element-wise efficient information acquisition´ with a bayesian deep latent gaussian model. In NeurIPS, pp. 14791–14802, 2019.

Peter Green and Catriona J MacLeod. Simr: An r package for power analysis of generalized linear mixed models by simulation. Methods in Ecology and Evolution, 7(4):493–498, 2016.

Patrick Hemmer, Monika Westphal, Max Schemmer, Sebastian Vetter, Michael Vossing, and Ger-¨ hard Satzger. Human-ai collaboration: The effect of AI delegation on human task performance and task satisfaction. In IUI, pp. 453–463. ACM, 2023.

Patrick Hemmer, Max Schemmer, Niklas Kuhl, Michael V¨ ossing, and Gerhard Satzger. Comple-¨ mentarity in human-ai collaboration: concept, sources, and evidence. Eur. J. Inf. Syst., 34(6): 979–1002, 2025.

Kilian Hendrickx, Wannes Meert, Bram Cornelis, and Jesse Davis. Know your limits: Machine learning with rejection for vehicle engineering. In ADMA, volume 13087 of Lecture Notes in Computer Science, pp. 273–288. Springer, 2021.

Robert R. Hoffman, Shane T. Mueller, Gary Klein, and Jordan Litman. Measures for explainable AI: explanation goodness, user satisfaction, mental models, curiosity, trust, and human ai performance. Frontiers Comput. Sci., 5, 2023. doi: 10.3389/FCOMP.2023.1096257. URL https://doi.org/10.3389/fcomp.2023.1096257.

Ronald A. Howard. Information value theory. IEEE Transactions on Systems Science and Cybernetics, 2(1):22–26, 1966. doi: 10.1109/TSSC.1966.300074.

Jarom´ır Janisch, Tomas Pevn´ y, and Viliam Lis´ y. Classification with costly features as a sequential´ decision-making problem. Mach. Learn., 109(8):1587–1615, 2020.

Shihao Ji and Lawrence Carin. Cost-sensitive feature acquisition and classification. Pattern Recognition, 40(5):1474–1485, 2007.

Veeru Kasivisvanathan, Antti S Rannikko, Marcelo Borghi, Valeria Panebianco, Lance A Mynderse, Markku H Vaarala, Alberto Briganti, Lars Budaus, Giles Hellawell, Richard G Hindley,¨ et al. MRI-Targeted or Standard Biopsy for Prostate-Cancer Diagnosis. New England Journal of Medicine, 378(19):1767–1777, 2018.

Edward H. Kennedy. Towards optimal doubly robust estimation of heterogeneous causal effects. Electronic Journal of Statistics, 17(2):3008 – 3049, 2023. doi: 10.1214/23-EJS2157. URL https://doi.org/10.1214/23-EJS2157.

Toru Kitagawa and Aleksey Tetenov. Who should be treated? empirical welfare maximization methods for treatment choice. Econometrica, 86(2):591–616, 2018.

Benjamin Kompa, Jasper Snoek, and Andrew L. Beam. Second opinion needed: communicating uncertainty in medical machine learning. npj Digit. Medicine, 4, 2021.

Levi Kumle, Melissa L-H Vo, and Dejan Draschkow. Estimating power in (generalized) linear˜ mixed models: An open introduction and tutorial in r. Behavior research methods, 53(6):2528– 2543, 2021.

Soren R K ¨ unzel, Jasjeet S Sekhon, Peter J Bickel, and Bin Yu. Metalearners for estimating heteroge- ¨ neous treatment effects using machine learning. Proceedings of the national academy of sciences, 116(10):4156–4165, 2019.

James Law. Review of ”algorithmic learning in a random world by vovk, gammerman and shafer”, springer, 2005, ISBN: 0-387-00152-2. SIGACT News, 37(4):38–40, 2006.

Jing Li, Yaoyao Cui, Juan Jiang, Jianqiu Yu, Lili Niu, Jiabo Deng, Fujun Shen, Liang Zhang, Bisong Yue, and Jing Li. Applying dna barcoding to conservation practice: a case study of endangered birds and large mammals in china. Biodiversity and Conservation, 26(3):653–668, 2017.

Shuqi Liu, Yuzhou Cao, Qiaozhen Zhang, Lei Feng, and Bo An. Mitigating underfitting in learning to defer with consistent losses. In AISTATS, volume 238 of Proceedings of Machine Learning Research, pp. 4816–4824. PMLR, 2024.

Shuqi Liu, Yuzhou Cao, Lei Feng, Bo An, and Luke Ong. When more experts hurt: Underfitting in multi-expert learning to defer. CoRR, abs/2602.17144, 2026.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. CoRR, abs/2201.03545, 2022.

Chao Ma, Sebastian Tschiatschek, Konstantina Palla, Jose Miguel Hern´ andez-Lobato, Sebastian´ Nowozin, and Cheng Zhang. EDDI: efficient dynamic discovery of high-value information with partial VAE. In ICML, volume 97 of Proceedings of Machine Learning Research, pp. 4234–4243. PMLR, 2019.

Shuai Ma, Ying Lei, Xinru Wang, Chengbo Zheng, Chuhan Shi, Ming Yin, and Xiaojuan Ma. Who should i trust: Ai or myself? leveraging human and ai correctness likelihood to promote appropriate trust in ai-assisted decision-making. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, CHI ’23, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9781450394215. doi: 10.1145/3544548.3581058. URL https: //doi.org/10.1145/3544548.3581058.

David Madras, Toniann Pitassi, and Richard S. Zemel. Predict responsibly: Improving fairness and accuracy by learning to defer. In NeurIPS, pp. 6150–6160, 2018.

Charles F. Manski. Statistical treatment rules for heterogeneous populations. Econometrica, 72 (4):1221–1246, 2004. doi: https://doi.org/10.1111/j.1468-0262.2004.00530.x. URL https:// onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0262.2004.00530.x.

Anqi Mao, Christopher Mohri, Mehryar Mohri, and Yutao Zhong. Two-stage learning to defer with multiple experts. In NeurIPS, 2023.

Anqi Mao, Mehryar Mohri, and Yutao Zhong. Mastering multiple-expert routing: Realizable hconsistency and strong guarantees for learning to defer. In ICML, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025.

Yannis Montreuil, Axel Carlier, Lai Xing Ng, and Wei Tsang Ooi. Adversarial robustness in twostage learning-to-defer: Algorithms and guarantees. In ICML, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net, 2025a.

Yannis Montreuil, Axel Carlier, Lai Xing Ng, and Wei Tsang Ooi. Why ask one when you can ask k? two-stage learning-to-defer to the top-k experts. CoRR, abs/2504.12988, 2025b.

Yannis Montreuil, Yeo Shu Heng, Axel Carlier, Lai Xing Ng, and Wei Tsang Ooi. A two-stage learning-to-defer approach for multi-task learning. In ICML, volume 267 of Proceedings ofMachine Learning Research. PMLR / OpenReview.net, 2025c.

Yannis Montreuil, Shu Heng Yeo, Axel Carlier, Lai Xing Ng, and Wei Tsang Ooi. Optimal query allocation in extractive qa with llms: A learning-to-defer framework with theoretical guarantees, 2026. URL https://arxiv.org/abs/2410.15761.

Hussein Mozannar and David A. Sontag. Consistent estimators for learning to defer to an expert. In ICML, Proceedings of Machine Learning Research, pp. 7076–7087. PMLR, 2020.

Hussein Mozannar, Hunter Lang, Dennis Wei, Prasanna Sattigeri, Subhro Das, and David A. Sontag. Who should predict? exact algorithms for learning to defer to humans. In AISTATS, volume 206 of Proceedings ofMachine Learning Research, pp. 10520–10545. PMLR, 2023.

X Nie and S Wager. Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, 108 (2):299–319, 06 2021. ISSN 0006-3444. doi: 10.1093/biomet/asaa076. URL https://doi. org/10.1093/biomet/asaa076.

Ana Rita Nogueira, Andrea Pugnana, Salvatore Ruggieri, Dino Pedreschi, and Joao Gama. Methods˜ and tools for causal discovery and causal inference. Wiley interdisciplinary reviews: data mining and knowledge discovery, 12(2):e1449, 2022.

Sima Noorani, Shayan Kiyani, George J. Pappas, and Hamed Hassani. Human-AI collaborative uncertainty quantification. In Trustworthy AI for Good (AI4GOOD) Workshop @ ICML 2026, 2026. URL https://openreview.net/forum?id=AmZOI1AY1t.

Gali Noti and Yiling Chen. Learning when to advise human decision makers. In IJCAI, pp. 3038– 3048. ijcai.org, 2023.

Nastaran Okati, Abir De, and Manuel Gomez-Rodriguez. Differentiable learning under triage. In NeurIPS, pp. 9140–9151, 2021.

Filippo Palomba, Andrea Pugnana, Jose M.´ Alvarez, and Salvatore Ruggieri. A causal framework<sup>´</sup> for evaluating deferring systems. In AISTATS, volume 258 of Proceedings of Machine Learning Research, pp. 2143–2151. PMLR, 2025.

Dario Pesenti, Alessandro Bogani, Stefano Teso, and Andrea Pugnana. Too much of the same: From algorithmic to human bias in learning to defer, 2026. URL https://arxiv.org/ abs/2608.28050.

Andrea Pugnana, Lorenzo Perini, Jesse Davis, and Salvatore Ruggieri. Deep neural network benchmarks for selective classification. J. Data-centric Mach. Learn. Res., 1:(17):1–58, 2024.

Andrea Pugnana, Giovanni De Toni, Cesare Barbera, Roberto Pellungrini, Bruno Lepri, and Andrea Passerini. To ask or not to ask: Learning to require human feedback. CoRR, abs/2510.08314, 2025.

Clara Punzi, Roberto Pellungrini, Mattia Setzu, Fosca Giannotti, and Dino Pedreschi. Ai, meet human: Learning paradigms for hybrid decision making systems. CoRR, abs/2402.06287, 2024.

Arman Rahbar, Linus Aronsson, and Morteza Haghir Chehreghani. A survey on active feature acquisition strategies. CoRR, abs/2502.11067, 2025.

Tamas Bisztray Rebeka Toth, Nils Gruschka. Constructing and benchmarking: a labeled email dataset for text-based phishing and spam detection framework. CoRR, abs/2511.21448, 2025. Withdrawn.

Giuseppe Romeo and Daniela Conti. Exploring automation bias in human-ai collaboration: a review and implications for explainable AI. AI Soc., 41(1):259–278, 2026.

Paul R Rosenbaum and Donald B Rubin. The central role of the propensity score in observational studies for causal effects. Biometrika, 70(1):41–55, 1983.

Donald B Rubin. Estimating causal effects of treatments in randomized and nonrandomized studies. Journal ofeducational Psychology, 66(5):688, 1974.

Donald B. Rubin. Bayesian Inference for Causal Effects: The Role of Randomization. The Annals of Statistics, 6(1):34 – 58, 1978. doi: 10.1214/aos/1176344064. URL https://doi.org/ 10.1214/aos/1176344064.

Donald B Rubin. Estimation in parallel randomized experiments. Journal ofEducational Statistics, 6(4):377–401, 1981.

Salvatore Ruggieri and Andrea Pugnana. Things machine learning models know that they don’t know. In AAAI, pp. 28684–28693. AAAI Press, 2025.

Maytal Saar-Tsechansky, Prem Melville, and Foster Provost. Active feature-value acquisition. Management Science, 55(4):664–684, 2009.

Mauricio Sadinle, Jing Lei, and Larry A. Wasserman. Least ambiguous set-valued classifiers with bounded error levels. CoRR, abs/1609.00451, 2016.

Max Schemmer, Niklas Kuhl, Carina Benz, Andrea Bartos, and Gerhard Satzger. Appropriate re-¨ liance on AI advice: Conceptualization and the effect of explanations. In IUI, pp. 410–422. ACM, 2023.

Uri Shalit, Fredrik D. Johansson, and David A. Sontag. Estimating individual treatment effect: generalization bounds and algorithms. In ICML, volume 70 of Proceedings ofMachine Learning Research, pp. 3076–3085. PMLR, 2017.

Claudia Shi, David M. Blei, and Victor Veitch. Adapting neural networks for the estimation of treatment effects. In NeurIPS, pp. 2503–2513, 2019.

Mark Steyvers, Heliodoro Tejeda, Gavin Kerrigan, and Padhraic Smyth. Bayesian modeling of human–ai complementarity. Proceedings of the National Academy of Sciences, 119(11): e2111547119, 2022.

Luca Stradiotti, Dario Pesenti, Stefano Teso, and Jesse Davis. Learning to reject low-quality explanations via user feedback. CoRR, abs/2507.12900, 2025.

Eleni Straitouri and Manuel Gomez Rodriguez. Designing decision support systems using counterfactual prediction sets. In ICML, Proceedings of Machine Learning Research, pp. 46722–46744. PMLR / OpenReview.net, 2024.

Joshua Strong, Qianhui Men, and J. Alison Noble. Trustworthy and practical AI for healthcare: A guided deferral system with large language models. In AAAI, pp. 28413–28421. AAAI Press, 2025.

Rajeev Verma and Eric T. Nalisnick. Calibrated learning to defer with one-vs-all classifiers. In ICML, volume 162 of Proceedings of Machine Learning Research, pp. 22184–22202. PMLR, 2022.

Rajeev Verma, Daniel Barrejon, and Eric T. Nalisnick. Learning to defer to multiple experts: Con- ´ sistent surrogate losses, confidence calibration, and conformal ensembles. In AISTATS, volume 206 of Proceedings of Machine Learning Research, pp. 11415–11434. PMLR, 2023.

Stefan Wager and Susan Athey. Estimation and inference of heterogeneous treatment effects using random forests. Journal of the American Statistical Association, 113(523):1228–1242, 2018. doi: 10.1080/01621459.2017.1319839. URL https://doi.org/10.1080/01621459. 2017.1319839.

Bryan Wilder, Eric Horvitz, and Ece Kamar. Learning to complement humans. In IJCAI, pp. 1526– 1533, 2020.

Yan Xiong and Liyan Yang. Secret and overt information acquisition in financial markets. The Review ofFinancial Studies, 36(9):3643–3692, 2023.

Zheng Zhang, Cuong C. Nguyen, Kevin Wells, Thanh-Toan Do, David Rosewarne, and Gustavo Carneiro. Coverage-constrained human-ai cooperation with multiple experts. In AAAI, pp. 18055– 18062. AAAI Press, 2026.

## A PROOFS

## A.1 PROOF OF OPTIMAL DISCLOSURE POLICY

For completeness, we restate here Theorem 1 from Section 3.1, and then provide its proof.

Theorem 1 (Optimal Disclosure Policy). Assume that the cumulative distribution of $\operatorname { V o I } ( \mathbf { x } )$ is continuous. $L e t q _ { 1 - B }$ denote the $( 1 - B )$ -quantile of $\operatorname { V o I } ( \mathbf { x } )$ and let $\lambda ^ { * } = \operatorname* { m a x } \{ 0 , q _ { 1 - B } \}$ . For any budget $B \in [ 0 , 1 ]$ , the optimal disclosure policy is the threshold rule:

$$
d ^ { * } ( \mathbf { x } ) = \left\{ \begin{array} { l l } { 1 } & { i f \quad \mathrm { V o I } ( \mathbf { x } ) \geq \lambda ^ { * } , } \\ { 0 } & { o t h e r w i s e . } \end{array} \right.\tag{10}
$$

Proof. We begin by first expanding the objective,

$$
R ( d ) = \mathbb { E } \big [ ( 1 - d ( \mathbf { x } ) ) r _ { 0 } ( \mathbf { x } ) + d ( \mathbf { x } ) r _ { 1 } ( \mathbf { x } ) \big ] = \mathbb { E } [ r _ { 0 } ( \mathbf { x } ) ] - \mathbb { E } \big [ d ( \mathbf { x } ) \operatorname { V o I } ( \mathbf { x } ) \big ] ,
$$

and since $\mathbb { E } [ r _ { 0 } ( \mathbf { x } ) ]$ does not depend on $d ,$ the original minimization problem is equivalent to

$$
\operatorname* { m a x } _ { d \colon \mathcal { X } \to [ 0 , 1 ] } \mathbb { E } _ { \mathbf { x } } \big [ d ( \mathbf { x } ) \mathrm { V o I } ( \mathbf { x } ) \big ] \quad \mathrm { s . t . } \quad \mathbb { E } _ { \mathbf { x } } [ d ( \mathbf { x } ) ] \leq B .\tag{11}
$$

Introducing the multiplier $\lambda \geq 0$ for the budget constraint, the Lagrangian is

$$
{ \mathcal { L } } ( d , \lambda ) = \mathbb { E } { \big [ } d ( \mathbf { x } ) \operatorname { V o I } ( \mathbf { x } ) { \big ] } - \lambda { \big ( } \mathbb { E } [ d ( \mathbf { x } ) ] - B { \big ) } = \lambda B + \mathbb { E } { \big [ } d ( \mathbf { x } ) \left( \operatorname { V o I } ( \mathbf { x } ) - \lambda \right) { \big ] } .
$$

For any fixed $\lambda \geq 0$ , the maximisation is point-wise in x and the integrand is linear in $d ( \mathbf { x } ) \in [ 0 , 1 ]$ Therefore, any maximiser satisfies

$$
d _ { \boldsymbol { \lambda } } ( \mathbf { x } ) = \left\{ \begin{array} { l l } { 1 , } & { \mathbf { V o I } ( \mathbf { x } ) > \boldsymbol { \lambda } , } \\ { 0 , } & { \mathbf { V o I } ( \mathbf { x } ) < \boldsymbol { \lambda } , } \end{array} \right. \quad \quad d _ { \boldsymbol { \lambda } } ( \mathbf { x } ) \in [ 0 , 1 ] \mathrm { a r b i t r a r y o n } \{ \mathrm { V o I } ( \mathbf { x } ) = \boldsymbol { \lambda } \} .\tag{12}
$$

It remains to select $\lambda ^ { * }$ satisfying complementary slackness, $\lambda ^ { * } \big ( \mathbb { E } [ d _ { \lambda ^ { * } } ( \mathbf { x } ) ] - B \big ) = 0$ . This allows for both binding and non-binding budget scenarios, leading to two mutually exclusive cases.

(i) Non-binding budget. Suppose $\mathbb { P } ( \mathbf { V } \mathrm { O I } ( \mathbf { x } ) > 0 ) < B$ , so that $q _ { 1 - B } \leq 0$ and hence $\lambda ^ { * } = 0$ Then $\lambda ^ { * } = 0$ is admissible: the policy $d ^ { * } ( \mathbf { x } ) = \mathbf { 1 } \{ \mathrm { V o I } ( \mathbf { x } ) > 0 \}$ satisfies $\mathbb { E } [ d ^ { * } ( \mathbf { x } ) ] =$ $\mathbb { P } ( \mathbf { V } \mathrm { O I } ( \mathbf { x } ) > 0 ) < B$ , so the constraint is slack and complementary slackness holds with $\lambda ^ { * } = 0$ . By the continuity assumption at $\lambda ^ { * } = 0$ we have $\mathbb { P } ( \mathbf { V } \mathrm { O I } ( \dot { \mathbf { x } } ) = 0 ) = 0$ , so $d ^ { * } ( \mathbf { x } )$ agrees p-almost everywhere with $\bar { \mathbf { 1 } } \{ \mathrm { V O I } ( \mathbf { x } ) \geq 0 \}$ and is in particular feasible. Note that in this case no feasible policy can spend the full budget profitably: increasing $\mathbb { E } [ d ( \mathbf { x } ) ]$ ] beyond $\mathbb { P } ( \mathbf { V } \mathbf { O I } ( \mathbf { x } ) > 0 )$ requires disclosing on $\left\{ \mathrm { V o I } ( \mathbf { x } ) \leq 0 \right\}$ , which cannot increase the objective. In particular the equation $\mathbb { E } [ d _ { \lambda } ] = \bar { B }$ admits no solution.

(ii) Binding budget. Suppose $\mathbb { P } ( \mathbf { V } \mathrm { O I } ( \mathbf { x } ) ~ > ~ 0 ) ~ \geq ~ B ;$ ; then $F _ { \mathrm { V o I } ( \mathbf { x } ) } ( 0 ) ~ \le ~ 1 - ~ B$ , so $q _ { 1 - B } \geq 0$ and $\lambda ^ { * } = q _ { 1 - B } . \mathrm { \bf ~ B y }$ the continuity assumption at $\lambda ^ { * }$ , P ${ \bar { \bf \Delta } } ( { \bar { \bf V } } { \bar { \bf O } } { \bf I } ( { \bf x } ) = \lambda ^ { * } ) = 0$ and $F _ { \mathrm { V o I ( } \mathbf { x } \in \mathrm { ) } } ( \lambda ^ { * } ) = 1 - B$ , whence $\mathbb { P } ( \mathrm { V o I } ( \mathbf { x } ) > \mathbf { \bar { \mu } } ^ { * } ) = \bar { B }$ . The policy $d _ { \lambda ^ { * } } = \mathbf { 1 } \{ \mathrm { V o I } ( \mathbf { x } ) \geq \lambda ^ { * } \}$ therefore satisfies $\mathbb { E } [ d _ { \lambda ^ { * } } ] = B$ , and complementary slackness holds with $\lambda ^ { * } \geq 0$

Combining the two cases, the optimal multiplier is

$$
\lambda ^ { * } = \operatorname* { m a x } \big \{ 0 , { \cal F } _ { \mathrm { V o I } ( { \bf x } ) } ^ { - 1 } ( 1 - B ) \big \} ,\tag{13}
$$

and, since $\mathbb { P } ( \mathrm { V o I } ( \mathbf { x } ) = \lambda ^ { * } ) = 0$ by the continuity assumption, any maximiser of Eq. (11) agrees p-almost everywhere with

$$
d ^ { * } ( \mathbf { x } ) = \mathbf { 1 } \{ \mathrm { V o I } ( \mathbf { x } ) \geq \lambda ^ { * } \} ,\tag{14}
$$

which is the statement of Theorem 1.

## A.2 PROOF OF IDENTIFIABILITY OF THE VALUE OF INFORMATION

For completeness, we restate here Proposition 1 from Section 3.2, and then provide its proof.

Proposition 1 (Identifiability of the Value of Information). Let $H \sim \pi$ denote a decision-maker drawn from the pool , let $A _ { H } ( 0 )$ and $A _ { H } ( 1 )$ denote the potential actions and $\ell \big ( A _ { H } ( D ) , Y \big )$ the corresponding potential losses for $D \in \{ 0 , 1 \}$ Under assumptions $A l { - } A 5 ,$ the regime-specific potential risks are identifiablefrom observed data. In particular, the Value ofInformation

$$
\operatorname { V o I } ( \mathbf { x } ) = r _ { 0 } ( \mathbf { x } ) - r _ { 1 } ( \mathbf { x } ) = \mathbb { E } { \bigl [ } \ell { \bigl ( } A _ { H } ( 0 ) , Y { \bigr ) } - \ell { \bigl ( } A _ { H } ( 1 ) , Y { \bigr ) } \mid \mathbf { X } = \mathbf { x } { \bigr ] }
$$

is identifiable from observed data as

$$
\operatorname { V o I } ( \mathbf { x } ) = \mathbb { E } \left[ \ell ^ { \mathrm { o b s } } \big ( A _ { H } ( D ) , Y \big ) \mid \mathbf { X } = \mathbf { x } , D = 0 \right] - \mathbb { E } \left[ \ell ^ { \mathrm { o b s } } \big ( A _ { H } ( D ) , Y \big ) \mid \mathbf { X } = \mathbf { x } , D = 1 \right]
$$

Proof. Fix $\mathbf { x } \in \mathcal { X }$ and a regime $d \in \{ 0 , 1 \}$ . To distinguish the random regime D from the value it takes, throughout this proof we write $r _ { d } ( \mathbf { x } )$ for the risk $r _ { D } ( { \bf x } )$ of Eq. (1) evaluated at $D = d .$ Recall from Section 2 that, under regime d, the decision on a case is taken by the decision-maker $H _ { d } \sim \pi$ assigned to that regime. By SUTVA (A4), the action of $H _ { d }$ depends only on the information disclosed to it in that episode, so its potential action $A _ { H _ { d } } ( d )$ is well defined. By consistency (A3), on the event $\{ D = d \}$ the observed loss equals the corresponding potential loss, $\ell ^ { \mathrm { o b s } } \big ( A _ { H } ( D ) , Y \big ) =$ $\ell ( A _ { H _ { d } } ( d ) , Y )$ . By overlap (A2), $\mathbb { P } ( D = d \mid \mathbf { X } = \mathbf { x } ) > 0 ,$ , so conditioning on $\{ { \mathbf X } = { \mathbf x } , D = d \}$ is well defined and

$$
\begin{array} { r } { \mathbb { E } \left[ \ell ^ { \mathrm { o b s } } \big ( A _ { H } ( D ) , Y \big ) \mid \mathbf { X } = \mathbf { x } , D = d \right] = \mathbb { E } \left[ \ell \big ( A _ { H _ { d } } ( d ) , Y \big ) \mid \mathbf { X } = \mathbf { x } , D = d \right] . } \end{array}
$$

By exchangeability of decision-makers (A5), $H _ { d }$ is drawn from π independently of the covariates, and of the regime. The decision-maker observed under regime d is therefore distributed as a generic $H \sim \pi .$ , and

$$
\begin{array} { r } { \mathbb { E } \left[ \ell \big ( A _ { H _ { d } } ( d ) , Y \big ) \mid \mathbf { X } = \mathbf { x } , D = d \right] = \mathbb { E } _ { H \sim \pi } \left[ \ell \big ( A _ { H } ( d ) , Y \big ) \mid \mathbf { X } = \mathbf { x } , D = d \right] . } \end{array}
$$

By unconfoundedness (A1), the potential losses are independent of the regime given X, so

$$
\begin{array} { r } { \mathbb { E } _ { H \sim \pi } \left[ \ell \left( A _ { H } ( d ) , Y \right) \mid \mathbf { X } = \mathbf { x } , D = d \right] = \mathbb { E } _ { H \sim \pi } \left[ \ell \left( A _ { H } ( d ) , Y \right) \mid \mathbf { X } = \mathbf { x } \right] = r _ { d } ( \mathbf { x } ) , } \end{array}
$$

where the last equality is the definition of the regime-specific risk in Eq. (1). Hence both $r _ { 0 } ( \mathbf { x } )$ and $r _ { 1 } ( \mathbf { x } )$ are identified by the observed conditional expectations, and their difference identifies $\mathrm { V o I } ( \mathbf { x } ) \stackrel { \prime } { = } r _ { 0 } ( \mathbf { x } ) - r _ { 1 } ( \mathbf { x } )$ □

## A.3 PROOF OF THE NO-DISCLOSURE DEGRADATION BOUND

For completeness, we restate here Theorem 2 from Section 3.3, and then provide its proof. Recall that ${ \hat { d } } ( \mathbf { x } ) = \mathbb { 1 } \{ \widehat { \mathrm { V o I } } ( \mathbf { x } ) \geq { \hat { \lambda } } \}$ is the plug-in policy, $\hat { B } = \mathbb { E } _ { \mathbf { x } } [ \hat { d } ( \mathbf { x } ) ]$ its realized disclosure rate, and $\varepsilon ( \mathbf { x } ) = \widehat { \mathrm { V o I } } ( \mathbf { x } ) - \mathrm { V o I } ( \mathbf { x } )$ the estimation error of the VoI.

Theorem 2. No-disclosure degradation] Given a non-negative loss function $\ell : \mathcal { A } \times \mathcal { y }  \mathbb { R ^ { + } }$ , for any estimator ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ such that $\| \varepsilon \| _ { L ^ { 2 } ( p ) } < \infty$ and any $\hat { \lambda } \geq 0 ;$ , the following holds:

$$
R ( \hat { d } ) - R ( 0 ) \leq \sqrt { \hat { B } } \| \varepsilon \| _ { L ^ { 2 } ( p ) } - \hat { \lambda } \hat { B } .
$$

Proof. Recall from the proof of Theorem 1 that

$$
R ( d ) = \mathbb { E } _ { \mathbf { x } } \big [ ( 1 - d ( \mathbf { x } ) ) r _ { 0 } ( \mathbf { x } ) + d ( \mathbf { x } ) r _ { 1 } ( \mathbf { x } ) \big ] = \mathbb { E } _ { \mathbf { x } } [ r _ { 0 } ( \mathbf { x } ) ] - \mathbb { E } _ { \mathbf { x } } \big [ d ( \mathbf { x } ) \mathrm { V o I } ( \mathbf { x } ) \big ] .\tag{15}
$$

Since $R ( 0 ) \ = \ \mathbb { E } _ { \mathbf { x } } [ r _ { 0 } ( \mathbf { x } ) ]$ ], we have $R ( { \hat { d } } ) - R ( 0 ) = - \mathbb { E } _ { \mathbf { x } } [ { \hat { d } } ( \mathbf { x } ) \mathbf { V o I ( x ) } ]$ . Writing ${ \bf V o I } ( { \bf x } ) { \bf \beta } = { \bf \beta }$ $\widehat { \mathrm { V o I } } ( \mathbf { x } ) - \varepsilon ( \mathbf { x } )$

$$
R ( { \hat { d } } ) - R ( 0 ) = \underbrace { \mathbb { E } _ { \mathbf { x } } [ { \hat { d } } ( \mathbf { x } ) \varepsilon ( \mathbf { x } ) ] } _ { \mathrm { ( I ) } } - \underbrace { \mathbb { E } _ { \mathbf { x } } [ { \hat { d } } ( \mathbf { x } ) \widehat { \mathrm { V o I } } ( \mathbf { x } ) ] } _ { \mathrm { ( I I ) } } .\tag{16}
$$

For (I), we apply Cauchy–Schwarz to $\hat { d }$ and $\hat { d } \varepsilon ,$ using ${ \hat { d } } ( \mathbf { x } ) ^ { 2 } = { \hat { d } } ( \mathbf { x } )$ since $\hat { d }$ is an indicator:

$$
\begin{array} { r } { \mathbb { E } _ { \mathbf { x } } [ \hat { d } ( \mathbf { x } ) \varepsilon ( \mathbf { x } ) ] \leq \big ( \mathbb { E } _ { \mathbf { x } } [ \hat { d } ( \mathbf { x } ) ] \big ) ^ { 1 / 2 } \big ( \mathbb { E } _ { \mathbf { x } } [ \hat { d } ( \mathbf { x } ) \varepsilon ( \mathbf { x } ) ^ { 2 } ] \big ) ^ { 1 / 2 } \leq \sqrt { \hat { B } } \| \varepsilon \| _ { L ^ { 2 } ( p ) } . } \end{array}
$$

For (II), on $\{ \hat { d } ( \mathbf { x } ) = 1 \}$ we have $\widehat { \mathrm { V o I } } ( \mathbf { x } ) \geq \hat { \lambda } \geq 0 , \mathbf { s o } \hat { d } ( \mathbf { x } ) \widehat { \mathrm { V o I } } ( \mathbf { x } ) \geq \hat { \lambda } \hat { d } ( \mathbf { x } )$ for every x, whence $( \mathrm { I I } ) \geq \hat { \lambda } \hat { B } .$ . Substituting both bounds into Eq. (16) gives the claim. □

The two terms have opposite signs and opposite meanings. Term (I) is the price of estimation error, and it is paid only where the policy discloses: the factor $\hat { d } ( \mathbf { x } )$ zeroes out the contribution of every instance the policy leaves untouched, so that only the error on the disclosed instances, $\mathbb { E } _ { \mathbf { x } } [ \hat { d } ( \mathbf { x } ) \varepsilon ( \mathbf { x } ) ^ { 2 } ]$ , enters the bound, which therefore vanishes as the disclosure rate B<sup>ˆ</sup> shrinks. Term (II) is instead the gain the policy believes it is making, and enters with a negative sign: since disclosure is triggered only above the threshold, every disclosed instance contributes an estimated improvement of at least λ<sup>ˆ</sup>, which offsets part of the estimation error.

When can the plug-in policy degrade? The bound certifies no degradation only when $\left\| \varepsilon \right\| _ { L ^ { 2 } ( p ) } \leq$ ${ \hat { \lambda } } { \sqrt { \hat { B } } } , { \mathrm { i . e . } }$ , when the estimation error is small relative to the margin by which disclosed instances clear the threshold; otherwise, its right-hand side is positive and no-degradation is not guaranteed. When it does occur, degradation has a precise source. By Eq. (15), $R ( { \hat { d } } ) - R ( 0 ) \ =$ $- \mathbb { E } _ { \mathbf { x } } [ \hat { d } ( \mathbf { x } ) \widehat { \mathrm { V o I } } ( \mathbf { x } ) ]$ , so the plug-in policy can be worse than never disclosing only if it discloses on instances where $\mathrm { V o I } ( \mathbf { x } ) < 0 .$ . On these instances ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } ) \geq { \hat { \lambda } } \geq 0$ , hence $\varepsilon ( \mathbf { x } ) > \hat { \lambda } \colon$ degradation requires the estimator to overestimate the VoI by more than the threshold on part of the disclosed region. This is most likely when disclosure is harmful on a large fraction of instances and the budget is not binding. In this case $\hat { \lambda } = 0 ,$ , term (II) provides no margin, and any instance with $\mathrm { V o I } ( \mathbf { x } ) < 0$ but ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } ) \geq 0$ is disclosed. This is the regime we observe on Email, where ClassWise discloses on at most $\approx 5 3 \%$ of the instances (Fig. 7): for $B \geq . 7 0$ the threshold is zero and accuracy falls slightly below $\mathrm { N o D i s c } \left( \mathrm { F i g . } 2 , \mathrm { F i g . } 6 \right)$

## A.4 PROOF OF THE ORACLE REGRET BOUND

For completeness, we restate here Theorem 3 from Section 3.3, and then provide its proof. Recall that ${ \hat { d } } ( \mathbf { x } ) = \mathbb { 1 } \{ \widehat { \mathrm { V O I } } ( \mathbf { x } ) \geq \hat { \lambda } \}$ is the plug-in policy, $\hat { B } = \mathbb { E } _ { \mathbf { x } } [ \hat { d } ( \mathbf { x } ) ]$ its realized disclosure rate, and $\varepsilon ( \mathbf { x } ) = \widehat { \mathrm { V o I } } ( \mathbf { x } ) - \mathrm { V o I } ( \mathbf { x } )$ the estimation error of the VoI.

Theorem 3 (Oracle regret). Given a non-negative loss function $\ell : \mathcal { A } \times \mathcal { y }  \mathbb { R } ^ { + }$ , let $d ^ { * }$ be the optimal policy of Theorem 1 and $B ^ { * } = \mathbb { E } _ { \mathbf { x } } \mathbf { \bar { [ } } d ^ { * } ( \mathbf { x } ) \mathbf { ] }$ ]. Assume that the cumulative distribution of $\hat { \mathrm { V o I } } ( { \bf x } )$ is continuous, let $\hat { q } _ { 1 - B }$ denote its $( 1 - B )$ -quantile and let $\hat { \lambda } = \operatorname* { m a x } \{ 0 , \hat { q } _ { 1 - B } \}$ . Then

$$
R ( \hat { d } ) - R ( d ^ { * } ) \leq \sqrt { B ^ { * } + \hat { B } } \| \varepsilon \| _ { L ^ { 2 } ( p ) } .
$$

Proof. By Eq. (15), and writing $\operatorname { V o I } ( \mathbf { x } ) = \widehat { \operatorname { V o I } } ( \mathbf { x } ) - \varepsilon ( \mathbf { x } )$

$$
\begin{array} { r l } & { R ( \hat { d } ) - R ( d ^ { * } ) = \mathbb { E } _ { \mathbf { x } } \big [ ( d ^ { * } ( \mathbf { x } ) - \hat { d } ( \mathbf { x } ) ) \mathrm { V o I } ( \mathbf { x } ) \big ] } \\ & { \qquad = \underbrace { \mathbb { E } _ { \mathbf { x } } \big [ \big ( d ^ { * } ( \mathbf { x } ) - \hat { d } ( \mathbf { x } ) \big ) \widehat { \mathrm { V o I } } ( \mathbf { x } ) \big ] } _ { \mathrm { ( I I I ) } } - \mathbb { E } _ { \mathbf { x } } \big [ ( d ^ { * } ( \mathbf { x } ) - \hat { d } ( \mathbf { x } ) ) \varepsilon ( \mathbf { x } ) \big ] . } \end{array}\tag{17}
$$

Term (III) is non-positive. The proof of Theorem 1, applied with ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ in place of $\operatorname { V o I } ( \mathbf { x } )$ , shows that the threshold rule $\mathbb { 1 } \{ \widehat { \mathbf { V o I } } ( \mathbf { x } ) \geq \operatorname* { m a x } \{ 0 , \widehat { q } _ { 1 - B } \} \}$ maximizes ${ \mathbb E } _ { { \bf x } } [ d ( { \bf x } ) \widehat { \mathrm { V o I } } ( { \bf x } ) ]$ over all policies with $\mathbb { E } _ { \mathbf { x } } [ d ( \mathbf { x } ) ] \leq B ;$ by assumption, <sup>ˆ</sup>d is that rule. Since $d ^ { * }$ also satisfies the budget constraint, as $B ^ { * } \leq B$ by Theorem 1, we get $\mathbb { E } _ { \mathbf { x } } [ d ^ { * } ( \mathbf { x } ) \widehat { \mathrm { V o I } } ( \mathbf { x } ) ] \leq \mathbb { E } _ { \mathbf { x } } [ \hat { d } ( \mathbf { x } ) \widehat { \mathrm { V o I } } ( \mathbf { x } ) ]$

Let $\Delta ( { \bf x } ) = | d ^ { * } ( { \bf x } ) - \hat { d } ( { \bf x } ) |$ , the indicator of the region where the two policies disagree. Since $d ^ { * } ( \mathbf { x } ) - \hat { d } ( \mathbf { x } ) \in \{ - 1 , 0 , 1 \}$ , we have $\Delta ( { \bf x } ) ^ { 2 } = \Delta ( { \bf x } )$ , and Cauchy–Schwarz applied to $\Delta$ and $\Delta \left. \varepsilon \right.$ gives

$$
\begin{array} { r l } & { { \cal R } ( \hat { d } ) - { \cal R } ( d ^ { * } ) \leq \mathbb { E } _ { \mathbf { x } } \big [ \Delta ( \mathbf { x } ) | \varepsilon ( \mathbf { x } ) | \big ] \leq \big ( \mathbb { E } _ { \mathbf { x } } [ \Delta ( \mathbf { x } ) ] \big ) ^ { 1 / 2 } \big ( \mathbb { E } _ { \mathbf { x } } [ \Delta ( \mathbf { x } ) \varepsilon ( \mathbf { x } ) ^ { 2 } ] \big ) ^ { 1 / 2 } } \\ & { \qquad \leq \big ( \mathbb { E } _ { \mathbf { x } } [ \Delta ( \mathbf { x } ) ] \big ) ^ { 1 / 2 } \| \varepsilon \| _ { L ^ { 2 } ( p ) } . } \end{array}
$$

Finally, since $d ^ { * }$ and $\hat { d }$ take values in $\{ 0 , 1 \} , \Delta ( \mathbf { x } ) \leq d ^ { * } ( \mathbf { x } ) + { \hat { d } } ( \mathbf { x } )$ , so that $\mathbb { E } _ { \mathbf { x } } [ \Delta ( \mathbf { x } ) ] \le B ^ { * } + \hat { B } .$ which gives the claim. □

The intermediate bound shows that the regret depends only on the region where the two policies disagree: both its mass, $\mathbb { E } _ { \mathbf { x } } [ \Delta ( \mathbf { x } ) ]$ , and the estimation error on $\mathrm { i t } , \mathbb { E } _ { \mathbf { x } } [ \bar { \Delta ( \mathbf { x } ) } \varepsilon ( \mathbf { x } ) ^ { 2 } ]$ , enter the bound, whereas errors on instances that both policies disclose, or both withhold, do not contribute. The final bound replaces the mass of the disagreement region with the upper bound $B ^ { * } + { \hat { B } }$ , which is loose when the two policies largely agree.

## A.5 PROOF OF CLASS-WISE DECOMPOSITION

For completeness, we restate here Proposition 2 from Section 3.4, and then provide its proof.

Proposition 2 (Class-wise decomposition). Let us consider $\ell ( A ( D ) , Y ) = \mathbb { 1 } \{ A ( D ) \neq Y \}$ . Then $r _ { D } ( { \bf x } )$ can be written as:

$$
r _ { D } ( \mathbf { x } ) = \sum _ { y \in \mathcal { Y } } \underbrace { { \mathbb { P } } ( Y = y \mid \mathbf { X } = \mathbf { x } ) } _ { p _ { y } } \underbrace { { \mathbb { P } } ( A ( D ) \neq y \mid \mathbf { X } = \mathbf { x } , Y = y ) } _ { q _ { D , y } } .\tag{18}
$$

Proof. Since the expectation of an indicator is the probability of the corresponding event, the risk under the 0-1 loss is

$$
r _ { D } ( \mathbf { x } ) = \mathbb { E } \big [ \mathbb { 1 } \{ A ( D ) \neq Y \} \mid \mathbf { X } = \mathbf { x } \big ] = \mathbb { P } ( A ( D ) \neq Y \mid \mathbf { X } = \mathbf { x } ) .
$$

By the law of total probability over the values of Y ,

$$
\mathbb { P } ( A ( D ) \neq Y \mid \mathbf { X } = \mathbf { x } ) = \sum _ { y \in \mathcal { Y } } \mathbb { P } ( A ( D ) \neq y , Y = y \mid \mathbf { X } = \mathbf { x } ) ,
$$

where on the event $\{ Y = y \}$ the condition $A ( D ) \neq Y$ becomes $A ( D ) \neq y .$ . Finally, by the chain rule of probability, each term factorizes as

$$
\begin{array} { r } { \mathbb { P } ( A ( D ) \neq y , Y = y | \mathbf { X } = \mathbf { x } ) = \mathbb { P } ( Y = y | \mathbf { X } = \mathbf { x } ) \mathbb { P } ( A ( D ) \neq y | \mathbf { X } = \mathbf { x } , Y = y ) , } \end{array}
$$

which gives the claim.

## B A CAUSAL INFERENCE INTERPRETATION

Here we detail the causal interpretation provided in Section 3.2. Throughout, we use D for the random disclosure regime and $\bar { d } \in \{ 0 , 1 \}$ for its values, and $H \sim \pi$ for a random decision-maker and $h \in \mathcal H$ for a fixed one. The disclosure variable D plays the role of a binary treatment on the decision-maker: for a decision-maker $H ,$ we consider the potential actions $A _ { H } ( 0 )$ and $A _ { H } ( 1 )$ , corresponding to the decisions under the two regimes, and the associated potential losses $\ell ( A _ { H } ( 0 ) , Y )$ and $\ell ( A _ { H } ( 1 ) , Y )$ , which are the potential outcomes of interest.

For a fixed decision-maker $h \in \mathcal H$ , we define the human-specific risks

$$
r _ { d } ( \mathbf { x } , h ) = \mathbb { E } \big [ \ell \big ( A _ { h } ( d ) , Y \big ) \mid \mathbf { X } = \mathbf { x } , H = h \big ] , \qquad d \in \{ 0 , 1 \} ,
$$

so that the human-specific effect of disclosure introduced in Section 3.2 is $\tau _ { h } ( { \bf x } ) = r _ { 1 } ( { \bf x } , h ) -$ $r _ { 0 } ( \mathbf { x } , h )$ . This is the effect of disclosing S on the decisions of $h ,$ rather than a treatment effect on the case itself. Under the exchangeability of decision-makers (A5 below), the decision-maker is drawn independently of the case, and the regime-specific risks of Eq. (1) are the population averages $r _ { d } ( \mathbf { x } ) = \mathbb { E } _ { H \sim \pi } [ r _ { d } ( \mathbf { x } , H ) ]$ ]. Hence $\tau _ { \pi } ( \mathbf { x } ) = \mathbb { E } _ { H \sim \pi } [ \tau _ { H } ( \mathbf { x } ) ] = r _ { 1 } ( \mathbf { x } ) - r _ { 0 } ( \mathbf { x } )$ , and $\mathrm { V o I } ( \mathbf { x } ) = - \tau _ { \pi } ( \mathbf { x } )$

We now state the assumptions under which this identification is valid.

A1 Unconfoundedness. Conditional on the baseline covariates X, the disclosure regime is as good as random with respect to the potential losses (Rosenbaum & Rubin, 1983):

$$
\left\{ \ell ( A _ { H } ( 0 ) , Y ) , \ell ( A _ { H } ( 1 ) , Y ) \right\} \perp \perp D \mid \mathbf { X } .
$$

Intuitively, all possible sources of self-selection into disclosure (and non-disclosure) are captured by the observable covariates.

A2 Overlap (positivity). For all covariate profiles in the population, both disclosure regimes are observed with non-zero probability (Rubin, 1978). Let $e ( \mathbf { x } ) = \mathbb { P } ( D = 1 \mid \mathbf { X } = \mathbf { x } )$ denote the disclosure propensity. We assume there exists some $\eta > 0$ such that

$$
\eta < e ( \mathbf { x } ) < 1 - \eta \quad \forall \mathbf { x } \in \mathcal { X } .
$$

This guarantees that we can compare the risks under disclosure and non-disclosure at each covariate level and then aggregate these comparisons.

A3 Consistency. If a decision episode is observed under regime $D = d ,$ the observed loss equals the corresponding potential loss (Cole & Frangakis, 2009):

$$
\ell ^ { \mathrm { o b s } } \big ( A _ { H } ( D ) , Y \big ) = \ell \big ( A _ { H } ( d ) , Y \big ) \quad \mathrm { o n } \ \{ D = d \} .
$$

A4 Stable Unit Treatment Value Assumption (SUTVA). For each decision episode, the potential action of the decision-maker under regime $d \in \{ 0 , 1 \}$ depends only on the information disclosed to that decision-maker in that episode (Rubin, 1981).

A5 Exchangeability of decision-makers. For each disclosure regime $d \in \{ 0 , 1 \}$ , the decisionmaker assigned to a decision episode is drawn from the same target population $\pi ,$ independently of the case covariates and the disclosure regime. Formally,

$$
H _ { d } \mid \mathbf { X } = \mathbf { x } , D = d \sim \pi , \qquad { \mathrm { f o r ~ a l l } } \quad \mathbf { x } \in { \mathcal { X } } \quad { \mathrm { a n d } } \quad d \in \{ 0 , 1 \} .
$$

Accordingly, the regime-specific conditional risk is the expected loss of a decision-maker drawn from π:

$$
r _ { d } ( \mathbf { x } ) = \mathbb { E } _ { H \sim \pi } \left[ \ell \big ( A _ { H } ( d ) , Y \big ) \mid \mathbf { X } = \mathbf { x } \right] .
$$

Therefore,

$$
\mathrm { V o I } ( \mathbf { x } ) = r _ { 0 } ( \mathbf { x } ) - r _ { 1 } ( \mathbf { x } )
$$

measures the expected reduction in decision loss induced by disclosure for a decisionmaker drawn from the target population π, rather than for a particular individual. This assumption does not require observing the same decision-maker under both regimes. It requires that the decision-makers observed under the two regimes be representative of the same target population.

## C IMPLEMENTATION DETAILS

## C.1 SYNTHETIC EXPERIMENTS.

Dataset generation. For our Synth experiments, we consider both a binary classification setting (SynthBin: $| y | = 2 )$ and a multiclass setting $( \mathrm { S y n t h M u 1 t i } : | \mathcal { V } | = 4 )$ . In each case, we generate data using sklearn.make classification with n features = 20 (all informative), n sample $\begin{array} { c c l } { \mathrm { ~ : ~ s ~ } } & { = } & { \mathrm { ~ 1 0 0 0 0 0 ~ } } \end{array}$ , and cl $\mathsf { a s s \mathrm { _ { - } s e p } } = 1$ . To emulate selective disclosure, we treat a subset of $| \mathcal { X } | = 9$ features as baseline information and the remaining $| S | = 1 1$ as hidden (i.e., information that is not available in the no-disclosure regime). Since no real human annotations are available in this setting, we simulate the two decision-makers with logistic regression models: the $D = 0$ decisor is trained on the 9 baseline features only, while the $D = 1$ decisor is trained on all 20 features. For each instance, the corresponding action $a _ { i } ( d )$ is the label predicted by the respective model, so that the $D = 0$ decisor is systematically less accurate, and errors concentrate on the instances whose label depends on the hidden features.

Models. For the ClassWise estimator, we first train a multiclass label model $f \colon \mathcal { X }  \Delta ^ { | \mathcal { V } | }$ implemented as a two-layer MLP with ReLU activations and a final linear layer followed by a softmax over classes, to approximate $\mathbb { P } ( Y = y \mid X = \mathbf { x } )$ . In parallel, for each regime D we train a multi-head error model consisting of a shared backbone (a two-layer MLP with GELU activations) and binary heads, implemented as a single linear layer of size $| { \bf \bar { y } } | ;$ the y-th head takes the shared representation and outputs a logit whose sigmoid corresponds to $\mathbb { P } ( A ( D ) \neq y \mid X = \mathbf { x } , Y = y )$ Only the head of the observed class contributes to the loss, which is a pos weight-weighted binary cross-entropy with the same logit correction at prediction time.

For the T-Learner estimator, we train two separate scalar risk networks, one per regime $D \in$ $\{ 0 , 1 \}$ , each modeling $r _ { D } ( \mathbf { x } ) = \mathbb { P } ( A ( D ) \neq Y \mid \bar { X } = \mathbf { x } )$ . Concretely, each network is a two-layer MLP with GELU activations and dropout, optimized with a weighted binary cross-entropy loss to predict the error indicator $\mathbb { 1 } \{ A ( D ) \neq Y \}$ from the baseline features .

The S-Learner estimator replaces the two separate risk networks of the T-Learner with a single network $z \colon \mathcal { X } \times \{ 0 , 1 \} \ \mathrm { ~ \bar { ~ } \to ~ [ 0 , 1 ] ~ }$ ] that takes the regime as an additional input feature: the binary indicator D is appended to x, and the network is a two-layer $\bf M L P$ with GELU activations and dropout after each hidden layer, followed by a scalar output. It is trained on the pooled data of both regimes, so that every instance contributes two examples, $( \mathbf { x } , 0 )$ with target $\mathbb { 1 } \{ A ( 0 ) \neq Y \}$ and $( \mathbf { x } , 1 )$ with target $\mathbb { 1 } \{ A ( 1 ) { \stackrel { } { = } } Y \}$ , under a single pos weight-weighted binary cross-entropy.

Finally, the Confidence baseline does not model the human error at all: it only trains the label model $g$ of the ClassWise estimator and ranks instances by its predictive uncertainty, setting ${ \hat { r } } _ { 0 } ( \mathbf { x } ) = u ( g ( \mathbf { x } ) )$ and ${ \hat { r } } _ { 1 } ( \mathbf { x } ) = 0$ , so that ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } ) = u ( g ( \mathbf { x } ) )$ . We use $u ( \cdot ) = 1 - \operatorname* { m a x } _ { y } g ( \mathbf { x } ) _ { y }$ in our experiments. This baseline encodes the intuition that information should be disclosed where the instance is intrinsically hard, regardless of whether the support information helps there; note that, since $u \geq 0 ,$ , its estimated ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ is never negative and the policy therefore always spends the full budget, unlike the VoI-based estimators.

For both the SynthBin and SynthMulti settings, all models are trained with Adam for 10 epochs, with weight decay $1 0 ^ { - 4 }$ and without early stopping, using the following hyperparameters:

• Label model f (also used by Confidence): hidden dimension 128, batch size 32, learning rate $5 \times \mathrm { 1 0 ^ { - 4 } }$ , no dropout.

• Error model $g _ { D } \colon$ backbone hidden dimension 256, batch size 32, learning rate $1 0 ^ { - 4 }$ , no dropout.

• T-Learner and S-Learner risk networks: hidden dimension 128, batch size 128, learning rate $5 \times 1 0 ^ { - 4 }$ , dropout 0.05.

## C.2 REAL DATA EXPERIMENTS - EMAIL.

Email Dataset description. We used a dataset comprising 1000 real emails with ground-truth labels indicating whether each email was fraudulent (e.g., a phishing attempt) or legitimate. The emails were selected from the corpus introduced by (Rebeka Toth, 2025), which provides only the messages; the human annotations are instead collected in a dedicated behavioural study (see Bogani et al. 2026 for details on the email selection procedure). In that study, 300 participants were recruited online through Prolific and asked to provide binary judgments for 20 emails randomly sampled from the dataset, indicating whether they considered each email to be fraudulent or legitimate. Participants were randomly assigned to one of two disclosure regimes. In the no-disclosure condition $( D ^ { - } = 0 )$ , participants were presented only with the email to be classified. In the disclosure condition $( D = 1 )$ , they were additionally shown the prediction of a machine-learning model.

Notably, in this dataset, exchangeability of decision-makers is supported by the random assignment of participants to the two disclosure regimes. We process these data in two steps:

(i) We first aggregate the raw annotations into a per-email summary: each row corresponds to a single email and includes the ground-truth label and the empirical human response distribution in each regime. Concretely, for each label $y \in \{ f r a u d u l e n t , l e g i t i m a t e \}$ we compute human 0y as the fraction of $D = 0$ participants who assigned the email to class y, and analogously human 1y for the $D = 1$ participants. This yields, for each email and regime, an estimated human probability distribution over the two outcomes, from which we derive the human error indicators $e r r _ { 0 }$ and err (whether a label sampled from the corresponding distribution differs from the ground truth) used in our experiments.

(ii) Then, we augment the train dataset to increase diversity and robustness. We leverage text-based augmentations provided by (Rebeka Toth, 2025) <sup>7</sup> and complement them with back-translation. For back-translation, we translate each email into seven languages (fr, de, es, it, pt, nl, ru) and then translate it back to English, obtaining multiple paraphrased variants per original message. In total, this procedure yields up to 14 augmented versions per email and results in 15000 samples used in our experiments.

Email models. For the Email dataset, all models operate on precomputed text embeddings rather than on raw text. Each email is encoded with a publicly available transformer-based phishingdetection model <sup>8</sup>, fine-tuned for binary phishing classification: we truncate the email body to 512 tokens and take the final-layer [CLS] representation, yielding a 768-dimensional embedding per (original or augmented) email. On top of these embeddings, every network uses the same MLP trunk: a LayerNorm on the input followed by two GELU hidden layers with dropout after each layer (halved on the second), where the second hidden layer has half the width of the first. The label model f (also used by Confidence) and the T-Learner risk networks use hidden width $7 6 8 \to 1 2 8 \to 6 4$ , the error model $g _ { D }$ of ClassWise uses $7 6 8 \to 2 5 6 \to 1 2 8$ , and the S-Learner network, whose input includes the regime indicator, uses $7 6 9  1 2 8  6 4$ . The label and error models are trained with batch size 64, the T-Learner and S-Learner networks with batch size 128 on inputs standardized with training-split statistics.

Table 1: Selected hyperparameters for the Email risk models.
<table><tr><td>Model</td><td>Epochs</td><td>Optimizer</td><td>LR</td><td>Dropout</td><td>Weight decay</td></tr><tr><td>ClassWise</td><td>50</td><td>ADAM</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td>0.3</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td> $\scriptstyle \mathrm { T - L e a r n e r }$ </td><td>25</td><td>ADAMW</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td>0.1</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>S-Learner</td><td>25</td><td>ADAM</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.1</td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td> $\mathtt { C o n f i d e n c e }$ </td><td>100</td><td>ADAMW</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td>0.3</td><td> $1 0 ^ { - 3 }$ </td></tr></table>

Model selection is performed by grid search over the number of training epochs 25, 50, 100 , the optimizer (ADAM, ADAMW), the learning rate $\{ 1 \times 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } \}$ , the dropout rate 0.1, 0.3 and the weight decay $\lbrace 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \rbrace$ , retaining for each model the configuration with the highest binary AUROC on the validation split. The resulting hyperparameters are reported in Table 1.

## C.3 REAL DATA EXPERIMENTS - IMAGENET-16H.

ImageNet-16H Dataset description. We used the IMAGENET-16H dataset introduced by (Steyvers et al., 2022), as processed for the human study of (De Toni et al., 2024). The dataset comprises 1200 natural images drawn from 16 categories airplane, bear, bicycle, bird, boat, bottle, car, cat, chair, clock, dog, elephant, keyboard, knife, oven, truck , each corrupted with phase noise to make the classification task challenging for both humans and models; we use the highest available noise level (110), which yields the hardest regime. Each image carries a ground-truth category label. A VGG-19 classifier fine-tuned on the noisy images provides softmax scores over the 16 classes, from which (De Toni et al., 2024) construct conformal prediction sets using the standard split-conformal calibration procedure.

Concretely, given a target miscoverage level α, the conformal score of an image is defined as $s = 1 -$ ${ \hat { p } } ( y \mid x )$ , where ${ \hat { p } } ( y \mid x )$ is the model’s softmax probability of the ground-truth class y (Law, 2006). The threshold q is set to the $( 1 - \alpha ) ( n + 1 ) / n$ empirical quantile of these scores on a calibration set, and the prediction set for an image is the collection of classes whose softmax probability exceeds q. This construction enjoys the usual marginal coverage guarantee of $1 - \alpha \colon$ smaller values of α enforce higher coverage and therefore yield larger prediction sets, whereas larger values of α produce smaller, more informative sets (Sadinle et al., 2016). Since the human study recorded, for each image, the responses of participants who were shown a prediction set of a given size, the choice of α determines both the set displayed in the disclosure condition and the corresponding empirical human response distribution. In our experiments we consider three miscoverage levels, $\alpha \in \{ 0 . 0 1 , 0 . 0 2 , 0 . 0 \bar { 5 } \}$ . We will discuss in Appendix E.4 about the choice of α.

Human judgments were collected online in two disclosure regimes, drawn from two separate behavioural studies over the same 1200 images. In the no-disclosure condition $( D ~ = ~ 0 )$ , taken from (Steyvers et al., 2022), participants were shown only the (noisy) image and asked to select one of the 16 categories, yielding 7261 classifications from 145 participants. In the disclosure condition $( D = 1 )$ , taken from (Straitouri & Rodriguez, 2024) <sup>9</sup>, participants were additionally shown the model’s conformal prediction set for the image before selecting a label. The set was presented in a lenient regime: it served as a suggestion rather than a constraint, so participants were free to choose any of the 16 categories, including labels outside the displayed set 10

Notably, exchangeability of decision-makers for this dataset is plausible because: $( i )$ the two studies use the same task and item population and (ii) recruit participants from comparable populations under similar experimental protocols, with the principal difference being the availability of support information. We process the human study data in two steps:

(i) We first aggregate the raw annotations into a per-image summary: each row corresponds to a single image and includes the ground-truth label, the model softmax scores, and the human responses collected under each regime. For the no-disclosure regime, for each category y we compute human 0y as the fraction of $D = 0$ participants who assigned the image to class $y ,$ and a human label is then obtained by sampling from this distribution. For the disclosure regime, the response we use depends on the chosen miscoverage level $\alpha .$ For each image, the experiment of (Straitouri & Rodriguez, 2024) collected disclosure responses from 16 distinct $D = 1$ participants, each of whom was shown a prediction set of a different size, ranging from 1 to 16 labels. Since the size of the conformal prediction set for a given image is determined by $\alpha ,$ each of these responses corresponds to the decision a human would make when assisted at a particular miscoverage level. Fixing α therefore determines, for each image, the size of the set the policy would actually disclose, and we retain only the response of the participant who was shown a set of exactly that size. This yields, for each image and regime, a human label (sampled from the empirical distribution in the no-disclosure regime, or the human response selected via α in the disclosure regime) from which we derive the human error indicators $e r r _ { 0 }$ and $e r r _ { 1 }$ used in our experiments.

(ii) Then, we augment the train dataset to increase diversity and robustness. At training time each image is transformed by composing 3 randomly sampled geometric augmentations drawn from a fixed set of 14 operations (identity, horizontal/vertical flips, rotations of ${ \bar { 1 } } 5 ^ { \circ } / { 3 0 ^ { \circ } } / { 4 5 ^ { \circ } }$ , horizontal/vertical translations, and x/y shears), yielding multiple perturbed variants per original image while preserving its category label and associated human response distributions. Figure 4 illustrates the resulting variants for a representative image.

![](images/4692c61365dcd1a3f265aca119ac8ebd763b07c91351c80f6b30667d993b29d6.jpg)  
Figure 4: Geometric augmentations applied to the ImageNet-16H images at training time, shown here for one image of the cat category at the highest phase-noise level (110). Each panel corresponds to one of the 14 operations in the augmentation pool: the identity (original), horizontal and vertical flips, rotations of $1 5 ^ { \circ }$ $3 0 ^ { \circ }$ and $4 5 ^ { \circ }$ , translations along the four directions, and positive/negative shears along the x and y axes.

Table 2: Selected hyperparameters for the ImageNet-16H risk models, per conformal miscoverage level α.
<table><tr><td>α</td><td>Model</td><td>Epochs</td><td>Optimizer</td><td>LR</td><td>Dropout</td><td>Weight decay</td></tr><tr><td rowspan="4">0.01</td><td>T-Learner</td><td>25</td><td>ADAMW</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td>0.1</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>ClassWise</td><td>25</td><td>ADAMW</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.3</td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>S-Learner</td><td>25</td><td>ADAMW</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td>0.1</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Confidence</td><td>25</td><td>ADAM</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.3</td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td rowspan="4">0.02</td><td>T-Learner</td><td>25</td><td>ADAMW</td><td> $2 \times 1 0 ^ { - 4 }$ </td><td>0.1</td><td>10⁻3</td></tr><tr><td>ClassWise</td><td>25</td><td>ADAMW</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.1</td><td>10-3</td></tr><tr><td>S-Learner</td><td>25</td><td>ADAM</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.3</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Confidence</td><td>25</td><td>ADAM</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.1</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td rowspan="4">0.05</td><td>T-Learner</td><td>25</td><td>ADAMW</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.3</td><td>10⁻3</td></tr><tr><td>ClassWise</td><td>100</td><td>ADAMW</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.3</td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>S-Learner</td><td>100</td><td>ADAM</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.3</td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>Confidence</td><td>25</td><td>ADAM</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>0.1</td><td> $1 0 ^ { - 3 }$ </td></tr></table>

Exchangeability across studies. Unlike Email, where participants were randomly assigned to regimes within a single study, the two ImageNet-16H regimes come from different studies. Both use the same 1200 images at noise level 110, but the protocols slightly differ beyond the availability of support: D = 0 participants classified 200 images each, with a confidence rating on every trial, from a pool spanning four noise levels (Steyvers et al., 2022), whereas D = 1 participants were recruited on Prolific and answered multiple-choice questionnaires at noise level 110 only (Straitouri & Rodriguez, 2024).

We assess the plausibility of the exchangeability assumption using a feature of the design by (Straitouri & Rodriguez, 2024). Their study includes trials in which the displayed prediction set contains the entire label space . Because such a set provides no information about which label is more likely, performance on these trials provides a useful proxy for no-disclosure performance. Restricting the comparison to the same images, accuracy on full-set trials is 0.7713, compared with 0.7647 under $D = 0$ The similarity of these accuracies provides descriptive evidence that the decision-maker populations in the two studies have comparable baseline performance.

We acknowledge that this comparison does not establish exchangeability, which cannot be determined from the observed data alone. Moreover, displaying a nondiscriminative prediction set is not identical to withholding support. Nevertheless, together with the shared task and image set, this evidence supports the plausibility of treating participants in the two studies as draws from a common target population π.

ImageNet-16H Models. For the ImageNet-16H dataset, all models operate directly on the (noisy) images. Every network uses the same feature extractor, an ImageNet-pretrained CONVNEXT-TINY backbone (Liu et al., 2022) that maps each image to a 768-dimensional embedding and is kept frozen during training, followed by an MLP head with an input LayerNorm, two GELU hidden layers and dropout after each (halved on the second). The hidden widths are 128 256 for the label model $f$ (also used by Confidence, and trained with label smoothing 0.05), 256 256 for the error model $g _ { D }$ of ClassWise, and 128 256 for the T-Learner and S-Learner risk networks, the latter taking the regime indicator as an additional input. Losses and logit corrections are as in the synthetic setting. The label and error models are trained with batch size 64, the T-Learner and S-Learner networks with batch size 128.

For all four models, model selection is performed by grid search over the same hyperparameter space described above for Email (epochs, optimizer, learning rate, dropout rate, and weight decay), tuning each model separately for every miscoverage level $\alpha \in \{ 0 . 0 1 , 0 . 0 2 , 0 . 0 5 \}$ and retaining the configuration with the highest binary AUROC on the validation split. The resulting hyperparameters are reported in Table 2.

## D USER STUDIES

Procedure. For both experiments, we recruit participants through Prolific and randomly assign them to one of the four experimental conditions (Human Policy/Low Budget, HP-LB; Human Policy/High Budget, HP-HB; Machine Policy/Low Budget, MP-LB; Machine Policy/High Budget, MP-HB). <sup>11</sup> First, we provide them with task instructions and have them complete two practice trials to familiarize themselves with the interface (see examples of the interfaces used in both experiments in Fig. 5). Both the instructions and the practice trials are tailored to the experimental condition to which participants are assigned. To incentivize attentive responding, we inform participants that the 5 most accurate participants would receive a bonus of £5.00. We also tell participants in the HP conditions that, in the case of a tie, participants who used fewer requests for assistance would be favored to receive the bonus. This is done to prompt participants to request assistance only when they deem it necessary, similarly to what occurs when the provision of assistance is determined by the ClassWise policy in the MP conditions, which not necessarily exhaust the available budget.

Participants then proceed to the actual experiment, which consists of a total of 20 trials. In each trial, participants have to classify an item (an email in the Email task or an image in the ImageNet-16H one). <sup>12</sup> If they are assigned to one of the HP conditions, they can request assistance from the AI system (provided that they have not exhausted the available requests); if they are assigned to one of the MP conditions, depending on the ClassWise policy output, they are either provided with assistance from the AI system or informed that no assistance would be provided for that item. <sup>13</sup> After classifying the item, we also ask participants to indicate how confident they are that their answer is correct on a 7-point Likert scale ranging from 1 (“Not confident at all (guessing)”) to 7 (“Extremely confident”). No feedback is provided to participants on the accuracy of their responses. This both reflects realistic classification settings, in which the correctness of a classification may not be immediately verifiable, and limits learning across trials. Throughout the 20 trials, we include two attention checks and record the number of times participants switch away from the experiment browser tab.

The 20 items presented to each participant are randomly sampled from each dataset’s test set using stratified sampling. In both the HP and MP conditions, sampling preserves, as closely as possible, the proportions observed in the full test set along two characteristics: whether the selective disclosure method would provide support information for an item and whether such information is useful (i.e., whether, in the human-annotation studies used to construct the original datasets, participants classified that item more accurately with support information than without it). This procedure ensures that the samples presented to participants are as representative as possible of the original test set with respect to these characteristics. In the Email task, we further stratified the sampling according to the ground truth of the emails to be classified, in order to avoid item samples in which the emails were predominantly fraudulent or legitimate. Indeed, presenting too many items of the same class to participants during classification tasks may cause them to respond incorrectly simply because a prolonged sequence of identical answers is perceived as unnatural, leading them to change an otherwise correct answer (Pesenti et al., 2026). This risk is particularly relevant in binary classification tasks such as Email, but substantially less so when multiple classes are present, as in ImageNet-16H; we therefore do not apply this additional stratification to the latter. Finally, we specify that, within each experiment, we sample items from the same pool across all four experimental conditions; what differs between the LB and HB conditions is whether the method provides assistance for a given item, depending on the budget level. See Table 3 for details on the item-sampling stratification. 14

Table 3: Stratification of the 20-item samples used in Experiments 1 and 2.
<table><tr><td colspan="4"></td><td colspan="2">Budget</td></tr><tr><td>support information</td><td>Usefulness of assistance</td><td>Ground truth</td><td>Low (30%)</td><td></td><td>High (70%)</td></tr><tr><td colspan="6">ImageNet-16H</td></tr><tr><td colspan="6">20-item sample characteristics</td></tr><tr><td>Not provided</td><td>Not useful</td><td></td><td>12 (62.2%)</td><td></td><td>6 (29.0%)</td></tr><tr><td>Provided</td><td>Not useful</td><td></td><td></td><td>4 (24.9%)</td><td>12 (58.1%)</td></tr><tr><td>Not provided</td><td>Useful</td><td></td><td></td><td>2 (4.6%)</td><td>0 (0.4%)</td></tr><tr><td>Provided</td><td>Useful</td><td></td><td></td><td>2 (8.3%)</td><td>2 (12.4%)</td></tr><tr><td colspan="4">Items with support information in 20-item sample</td><td>6/20 (30%)</td><td>14/20 (70%)</td></tr><tr><td colspan="6">Original test-set characteristics</td></tr><tr><td colspan="4">Test-set size Policy provides assistance</td><td colspan="2">241 80/241 (33%) 170/241 (71%)</td></tr><tr><td colspan="6">Email</td></tr><tr><td colspan="6">20-item sample characteristics</td></tr><tr><td colspan="6"></td></tr><tr><td>Not provided</td><td>Not useful</td><td>Legitimate</td><td>6 (30.5%)</td><td></td><td>4 (20.0%)</td></tr><tr><td>Provided</td><td>Not useful</td><td>Legitimate</td><td>2 (8.0%)</td><td></td><td>4 (18.5%)</td></tr><tr><td>Not provided</td><td>Useful</td><td>Legitimate</td><td></td><td>2 (13.0%)</td><td>1 (9.0%)</td></tr><tr><td>Provided</td><td>Useful</td><td>Legitimate</td><td></td><td>1 (5.5%)</td><td>2 (9.5%)</td></tr><tr><td>Not provided</td><td>Not useful</td><td>Fraudulent</td><td></td><td>5 (23.0%)</td><td>4 (18.5%)</td></tr><tr><td>Provided</td><td>Not useful</td><td>Fraudulent</td><td></td><td>1 (6.5%)</td><td>2 (11.0%)</td></tr><tr><td>Not provided</td><td>Useful</td><td>Fraudulent</td><td></td><td>2 (9.5%)</td><td>2 (6.5%)</td></tr><tr><td>Provided</td><td>Useful</td><td>Fraudulent</td><td></td><td>1 (4.0%)</td><td>1 (7.0%)</td></tr><tr><td colspan="6">Items with support information in 20-item sample</td></tr><tr><td colspan="4">Legitimate emails in 20-item sample</td><td>11/20 (55%)</td><td>11/20 (55%)</td></tr><tr><td colspan="6">Original test-set characteristics</td></tr><tr><td colspan="6">Test-set size</td></tr><tr><td colspan="4">Policy provides assistance</td><td colspan="2">48/200 (24%) 92/200 (46%) 114/200 (57%) 114/200 (57%)</td></tr><tr><td colspan="6">Legitimate emails</td></tr><tr><td colspan="4"></td><td></td><td></td></tr></table>

Note. For the 20-item sample characteristics, values outside parentheses indicate the number of items included in the sample, while percentages in parentheses indicate the corresponding proportion of items in the original test set.

After having completed all 20 trials, we ask participants to complete a trust scale from (Hoffman et al., 2023), consisting of eight 5-point Likert scale items (extremes: I strongly disagree; I strongly agree). The items, presented in randomized order, were the following:

• I am confident in the system. I feel that it works well.

• The outputs of the system are very predictable.

• The system is very reliable. I can count on it to be correct all the time.

• I feel safe that when I rely on the system I will get the right answers.

• The system is efficient in that it works very quickly.

• I am wary of the system.

• The system can perform the task better than a novice human user.

• I like using the system for decision making.

Finally, we ask participants to report their familiarity with AI systems by selecting one of the following options:

• Option 1: I have little or no experience with AI systems and limited or no understanding of how they work.

• Option 2: I use AI systems occasionally but have not a clear understanding of how they function.

• Option 3: I use AI systems and have studied how they work (e.g., through courses, online classes, or self-study).

• Option 4: I develop or build AI systems as part of my work or personal projects.

All datasets generated from the human-subject experiments and analysis scripts can be found in the anomymous repo.

Participants recruitment and samples characteristics We conducted an a priori power analysis using a simulation-based approach (Green & MacLeod, 2016; Kumle et al., 2021) to estimate the sample size to collect. The analysis indicated that a total sample of 268 participants would have provided 82% statistical power to detect an effect of the two-way interaction between support and budget level conditions as small as OR = 0.43. <sup>15</sup> Accordingly, participants were recruited in batches through Prolific. After each batch, we assessed only whether participants met predetermined exclusion criteria: having passed both attention checks and not having left the browser tab in three or more trials. No further analyses of the outcome variables or effect sizes were performed during data collection. Recruitment continued until the final sample comprised a total of at least 268 participants. Inclusion criteria required participants to be native English speakers from the UK and to have a Prolific approval rate above 98%. Participants received compensation equal to £1.30.

For the Email and ImageNet-16H studies we recruited, respectively, a total of 288 and 278 participants, of which 272 and 273 were included in the final sample. In both studies, participants samples were balanced in terms of age, sex, and previous experience with AI across all experimental conditions (see Table 4).

Table 4: Participant characteristics by experimental condition.
<table><tr><td></td><td></td><td></td><td></td><td colspan="4">Past experience with AI</td></tr><tr><td>Experimental condition</td><td>n participants</td><td>Age</td><td>% Female</td><td>Option 1</td><td>Option 2</td><td>Option 3</td><td>Option 4</td></tr><tr><td>ImageNet-16H</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Human Policy/Low Budget</td><td>66</td><td>43.21 ± 14.05</td><td>52%</td><td>5%</td><td>67%</td><td>29%</td><td>0%</td></tr><tr><td>Human Policy/High Budget</td><td>69</td><td>42.06 ± 12.39</td><td>48%</td><td>3%</td><td>46%</td><td>49%</td><td>1%</td></tr><tr><td>Machine Policy/Low Budget</td><td>69</td><td>42.30 ± 12.23</td><td>48%</td><td>6%</td><td>54%</td><td>41%</td><td>0%</td></tr><tr><td>Machine Policy/High Budget</td><td>69</td><td>41.71 ± 13.52</td><td>42%</td><td>1%</td><td>65%</td><td>28%</td><td>6%</td></tr><tr><td>Email</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Human Policy/Low Budget</td><td>72</td><td>44.58 ± 15.63</td><td>51%</td><td>4%</td><td>53%</td><td>37%</td><td>6%</td></tr><tr><td>Human Policy/High Budget</td><td>64</td><td>39.33 ± 11.72</td><td>48%</td><td>0%</td><td>64%</td><td>34%</td><td>2%</td></tr><tr><td>Machine Policy/Low Budget</td><td>71</td><td>41.87 ± 13.15</td><td>55%</td><td>4%</td><td>62%</td><td>31%</td><td>3%</td></tr><tr><td>Machine Policy/High Budget</td><td>65</td><td>41.98 ± 13.07</td><td>55%</td><td>3%</td><td>49%</td><td>48%</td><td>0%</td></tr></table>

## E EXTENDED RESULTS

In this section, we detail the results obtained for both the analyses presented in the main text and additional ones we present here. We organize the analyses around the three research questions of the main text, together with a fourth one that we address here:

Q1 Does our approach learn effective policies?

Q2 Does our learned policy improve human-AI team performance?

Q3 How do participants interact with our policy?

![](images/66809b265ce404717ebdd9e47892b909a43a95affe4658cf5acf613ccb71ef59.jpg)  
Figure 5: (a) and (b): Interfaces presented to participants in the HP conditions in the Email and ImageNet-16H tasks, respectively; (c): Messages shown when information was requested (HP) or provided by the ClassWise policy (MP); (d): Messages shown when, in MP, the ClassWise policy determined information should not be provided for that item.

Q4 How does changing the information content affect our policy?

For questions related to the user studies (Q2 and Q3), we report the full statistical results. Many of the analyses we performed for such questions involved fitting logistic mixed-effects regression models. These models extend standard regression by including random effects that account for the non-independence of repeated observations (e.g., multiple responses from the same participant or to the same item), thereby yielding valid inferences despite correlated observations (for an overview, see Brown, 2021). For all mixed-effects models reported, we include random intercepts for participants and items.

In these regressions, categorical predictors are deviation-coded as $+ 0 . 5 / - 0 . 5$ (the level coded as +0.5 is reported in brackets in the relevant results tables). This coding allows coefficients to be interpreted as comparisons averaged across the levels of the other categorical predictors. When we perform pairwise post-hoc comparisons, we apply Bonferroni corrections to the p values to control the family-wise error rate at a significance level of .05 (we report corrected p values throughout).

## E.1 Q1: DOES OUR APPROACH LEARN EFFECTIVE POLICIES?

Take-home message: ClassWise ranks instances better and abstains more selectively than standard meta-learners, and its ability to abstain matters most when disclosure is harmful on a substantial fraction of the instances.

In the main paper, ClassWise outperforms the baselines across all datasets (Fig. 2). Here we investigate why, through two complementary analyses. First, we isolate the contribution of the classwise decomposition by replacing it with two standard meta-learners while keeping the disclosure rule fixed: any difference in accuracy can then only come from the quality of the VoI estimates. Second, we look at how the policies spend their budget: since the policy never discloses on instances with non-positive estimated VoI, the fraction of instances on which it actually requests help reveals whether an estimator can identify where disclosure does not help.

Ablation on the risk estimator. We compare our ClassWise estimator against two standard meta-learners for conditional treatment effects, the T-Learner and the S-Learner (Kunzel¨ et al., 2019). We do not include doubly robust learners (Kennedy, 2023) or causal forests (Wager & Athey, 2018). DR learners rely on cross-fitted nuisance models, which would further split our small real datasets, while their correction matters little here since disclosure is assigned independently of the covariates. Causal forests, on the other hand, struggle with the high-dimensional representations of the unstructured inputs (text and images) of our real datasets. ClassWise estimates VOI(x) by first predicting the label and then the class-conditional probability of a human error (Proposition 2). Here we ask how much of its performance is due to this decomposition, as opposed to the disclosure rule itself. We therefore keep the policy, the calibration procedure and the budget grid fixed, and vary only how the two regime-specific risks are estimated. All estimators are tuned over the same hyperparameter grid.

The T-Learner fits one scalar risk network per regime directly on the error indicator, ${ \hat { r } } _ { d } ( \mathbf { x } )$ ≈ $\mathbb { P } ( A ( d ) \neq Y \mid X = \mathbf { x } )$ , trained on the targets $\mathbb { 1 } \{ A ( d ) \neq Y \}$ , and sets $\widehat { \mathrm { V o I } } ( \mathbf { x } ) = \hat { r } _ { 0 } ( \mathbf { x } ) - \hat { r } _ { 1 } ( \mathbf { x } )$ ClassWise is itself a T-learner; the T-Learner considered here differs only in modelling each risk with a single scalar output rather than through the class-wise decomposition. Since the two regimes get independent functions, the risk curves are free to take different shapes and ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ is not systematically biased towards zero, at the cost of a higher variance, as it is the difference of two independently estimated quantities.

The S-Learner fits a single network that takes the regime as an additional input feature, $\hat { r } ( \mathbf { x } , d )$ ≈ $\mathbb { P } ( A ( d ) \neq Y \mid X = \mathbf { x } )$ , trained on the pooled data of both regimes, and sets $\widehat { \mathrm { V o I } } ( \mathbf { x } ) = \widehat { r } ( \mathbf { x } , 0 ) -$ $\hat { r } ( \mathbf { x } , 1 )$ . All parameters are shared across regimes, which reduces variance but tends to shrink the effect of d, and hence ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ , towards zero. Fig. 6 reports the budget–accuracy curves.

On ${ \mathrm { S y n t h } }$ data the ordering is stable across budgets: $\mathtt { C l a s s W i s e } > \mathtt { T \mathrm { - } L e a r n e r } > \mathtt { S \mathrm { - } L e a r n e r }$ at every B, in both the SynthBin and the SynthMulti setting. The gap is substantial and does not close with the budget: in the SynthBin case ClassWise plateaus at .85 against .83 for T-Learner and .81 for $\mathrm { { S - L e a r n e r ; } }$ in the $\mathtt { S y n t h M u l t i }$ case both meta-learners converge to full-disclosure accuracy $\left( \approx . 6 2 \right)$ without ever exceeding it, whereas $\mathrm { C } \perp \mathsf { a s s } \mathtt { W } \dot { 1 }$ se exceeds it from B .4. Since the disclosure rule is identical, the difference is entirely attributable to the quality of the $\operatorname { V o I } ( \mathbf { x } )$ estimate, both its ranking and its sign. The $\mathtt { S y n t h M u l t i }$ panel is where the class-wise decomposition pays off most: disclosure affects the  classes heterogeneously, and while a scalar risk model can in principle represent this heterogeneity, the decomposition makes it explicit, so that each head only has to learn the error pattern of a single class.

![](images/d1a00bda61fddb01e81fd8767ce41827d380d505d3412173d5659c53d0a800a1.jpg)  
Figure 6: Budget-accuracy curves for VoI-based disclosure policies on Synth (SynthBin, SynthMulti) and real (Email, ImageNet-16H) data. We report average mean values for 5 seeds and 95% confidence intervals.

On Email, where disclosure is harmful on average, ClassWise attains the highest accuracy overall ( .81 at $B = . 3 0 ) \mathrm { b u t T - L e a r n e r i }$ s better at large budgets ( .79 against .78 for $B \geq . 7 0 )$ S-Learner, instead, degrades monotonically after $B = . 1 0$ and coincides with full disclosure from $B = . 8 0$ onwards. This is the shrinkage of the S-Learner made visible: once ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ is compressed towards zero it loses the information about its sign and ends up almost uniformly positive, so the non-negativity constraint of Theorem 1 stops binding and the policy loses the ability to abstain precisely in the regime where abstention is the whole point.

On ImageNet-16H, where disclosure helps almost everywhere, the picture reverses: abstention is nearly irrelevant, all three estimators improve steadily, and S-Learner is no longer penalised, tracking ClassWise closely up to $B = . { \dot { 6 } } 0$ . Here T-Learner leads at low budgets (e.g. .82 against  .81 at $B = . 2 0 )$ while $\mathtt { C l a s s w i s e }$ takes over from $B \approx . 6 0$ and peaks at  .86 at $B = . 9 0 ,$ , above every other curve and above full disclosure ( .85).

Help-request frequency and quality of the risk estimates. A useful diagnostic of a disclosure policy is not only how much accuracy it attains, but how much of the budget it actually spends to attain it. By Theorem 1, the policy never discloses on instances with non-positive estimated VoI. Once the budget exceeds the fraction of instances with $\widehat { \mathrm { V o I } } ( \mathbf { x } ) > 0$ , additional budget triggers no further disclosures, and the fraction of instances on which the policy requests help plateaus below the nominal budget. The height of this plateau measures the size of the positive-VoI set identified by each estimator, i.e., the fraction of instances it deems worth disclosing. Fig. 7 reports this fraction as a function of the budget for all datasets.

The S-Learner stays on the diagonal spent = budget in every setting: ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } ) \geq 0$ on essentially every instance, so the sign constraint of Theorem 1 never binds and the policy never abstains. By sharing all parameters across regimes, the S-Learner compresses ${ \widehat { \mathrm { V o I } } } ( \mathbf { x } )$ towards a small, almost uniformly positive value and loses the information about its sign. The comparison of interest is therefore between T-Learner and ClassWise, which both abstain, but to different extents depending on the dataset.

On SynthBin and SynthMulti, ClassWise abstains earlier and more often than T-Learner. ClassWise plateau is at about 54% of the instances in SynthBin and 71% in SynthMulti, against 69% and 83% for T-Learner. Since ClassWise also attains the highest accuracy at every budget (Fig. 2, Fig. 6), its smaller positive-VoI set does not come from missing useful disclosures, but from correctly withholding the support information where it would not help.

On Email, disclosure is harmful on average, and both estimators abstain on a large fraction of instances: T-Learner plateau is  66%, ClassWise is  53%. The more parsimonious policy of ClassWise attains the highest accuracy overall, at intermediate budgets (Fig. 2, Fig. 6). At large budgets, the additional disclosures of T-Learner yield a slightly higher accuracy, suggesting that ClassWise withholds a small set of instances on which disclosure would still have helped.

![](images/27263d37b5355d43843a81e7af7b7bacba56eb4625ed3e295821c2481e9cc8db.jpg)  
Figure 7: Fraction of instances on which the policy requests help, as a function of the nominal budget, for S-Learner, T-Learner and ClassWise (mean over 5 seeds, with 95% confidence intervals). The dashed line is spent = budget. A curve plateaus below the diagonal when the estimated positive-VoI set is smaller than the budget, and the height of the plateau is the fraction of instances the estimator deems worth disclosing.

On ImageNet-16H disclosure is beneficial for almost every instance, and both estimators disclose almost up to the full budget: at B = 1, ClassWise discloses to 98% of the instances and T-Learner to nearly all of them. With few instances to withhold, the two estimators differ only in the ordering of the instances, not in the size of the positive-VoI set.

## E.2 Q2: DOES OUR LEARNED POLICY IMPROVE HUMAN-AI TEAM PERFORMANCE?

Take-home message: The learned policy matches or outperforms self-selection and is associated with greater confidence discrimination between correct and incorrect classifications. Furthermore, we note that human-selected disclosure only partially align with ClassWise-selected ones.

In the main paper, we assess differences in overall accuracy across the experimental conditions. Here, in addition to reporting the full results of that analysis, we examine participants’ accuracy as a function of the correctness of the ML advice provided as auxiliary information, providing a more detailed picture of how differences in adherence may contribute to differences in overall ac curacy, particularly in the Email task. Furthermore, we assess the effects of Support and Budget on participants’ confidence in their own classifications, exploring the potential benefits of a selective disclosure approach beyond classification accuracy alone. Finally, to provide a more complete comparison between ClassWise-selected and human-selected disclosure, we examine alignment between ClassWise-selected and human-selected disclosure.

Overall accuracy. We assess whether overall classification accuracy differs across the four experimental conditions by fitting a logistic mixed-effects model predicting the correctness of participants classifications from Support, Budget, and their interaction. This analysis was presented in the main text and will not be further discussed, but see Table 5 for the full results.

Classification accuracy by correctness of support information. We explore how participants accuracy varies as a function of the correctness of the support information provided. In the Email task, correctness was determined by whether the suggested class corresponded to the ground-truth label; in the ImageNet-16H task, it was determined by whether the ground-truth label was included in the prediction set. We focus on trials in which support information was available during classification and, to improve comparability between the MP and HP conditions, consider only items for which the policy would provide support information. Because the ML model prediction constituting the support information was generally accurate, trials in which it was incorrect were relatively rare (75/1, 227 observations in the Email study and 86/1, 827 in the ImageNet-16H study). The subgroups pertaining to wrong-advice instances thus present few observations (as evident from the large error bars characterizing relative to such subgroups in Fig. 8), making inferential statistics potentially unreliable; we therefore limit our discussion to the descriptive patterns.

As shown in Fig. 8, the two studies exhibit different patterns. In the ImageNet-16H study, accuracy in MP relative to HP tends to be higher when the support information is correct and approximately comparable when it is incorrect. In the Email study, the opposite descriptive pattern emerges: accuracy in MP is, on average, lower than in HP when the support information is correct and higher when it is incorrect. These patterns broadly correspond to the differences in adherence between MP and HP observed across the two studies (discussed in the main text and in the following section), with the relative adherence on policy-provided information being higher in the ImageNet-16H study than in the Email study.

<table><tr><td colspan="4">Descriptive statistics</td></tr><tr><td></td><td></td><td>ImageNet-16H</td><td>Email</td></tr><tr><td>Support condition Budget condition</td><td></td><td>Accuracy</td><td>Accuracy</td></tr><tr><td>Human Policy</td><td>Low Budget</td><td> $0 . 7 3 \pm 0 . 1 5$ </td><td> $0 . 8 2 \pm 0 . 1 3$ </td></tr><tr><td>Human Policy</td><td>High Budget</td><td> $0 . 8 1 \pm 0 . 1 1$ </td><td> $0 . 8 4 \pm 0 . 1 3$ </td></tr><tr><td>Machine Policy</td><td>Low Budget</td><td> $0 . 7 5 \pm 0 . 1 1$ </td><td> $0 . 8 1 \pm 0 . 1 2$ </td></tr><tr><td>Machine Policy</td><td>High Budget</td><td> $0 . 8 4 \pm 0 . 0 9$ </td><td> $0 . 8 4 \pm 0 . 1 2$ </td></tr></table>

<table><tr><td colspan="10">Regression coefficients</td></tr><tr><td></td><td colspan="4">ImageNet-16H</td><td colspan="5">Email</td></tr><tr><td>Fixed effect</td><td>OR</td><td>95% CI</td><td>Z</td><td>p</td><td>OR</td><td>95% CI</td><td></td><td>Z</td><td>p</td></tr><tr><td>Intercept</td><td>10.99</td><td>[7.72, 15.64]</td><td>13.31</td><td>&lt; .001</td><td>7.66</td><td>[6.24, 9.39]</td><td></td><td>19.50</td><td>&lt; .001</td></tr><tr><td>Support (HP)</td><td>0.76</td><td>[0.60, 0.97]</td><td>-2.25</td><td>.025</td><td>1.04</td><td></td><td>[0.82, 1.31]</td><td>0.30</td><td>.762</td></tr><tr><td>Budget (LB)</td><td>0.51</td><td>[0.40, 0.65]</td><td>-5.45</td><td>&lt; .001</td><td>0.79</td><td>[0.63, 1.01]</td><td></td><td>-1.91</td><td>.056</td></tr><tr><td>Support × Budget</td><td>1.15</td><td>[0.71, 1.87]</td><td>0.57</td><td>.566</td><td>1.04</td><td>[0.65, 1.66]</td><td></td><td>0.15</td><td>.880</td></tr></table>

Table 5: Descriptive statistics (means and standard deviations) and results of the logistic mixedeffects regressions predicting classification accuracy from Support, Budget, and their interaction (OR: odds ratio).

![](images/b3c15605105d7144292157cc8027a2f336e842473719a37e7aea78a05722f9e8.jpg)  
Figure 8: Participants’ average accuracy by experimental condition and correctness of support information. Ratios indicate the number of correct participant responses out of the total number of observations in each subgroup.

Furthermore, this pattern may help explain why a significant difference in accuracy between MP and HP was observed in the ImageNet-16H study but not in the Email study. Because the support information was correct on most trials, even a modest reduction in participants’ tendency to follow it could result in an appreciable decrease in overall accuracy.

Human confidence. We explore participants’ confidence in the correctness of their final classifications using a linear mixed-effects model in which confidence on each trial is predicted by Support, Budget, the presence of support information, participants’ classification accuracy, and all their interactions (see Table 6 for the full results). This analysis allows us to examine not only whether confi dence differs across experimental conditions, but also whether such differences depend on whether participants’ classifications are correct. In particular, greater confidence for correct than incorrect classifications can be interpreted as greater confidence discrimination.

In the ImageNet-16H study, we observe a significant interaction between Support, presence of support information, and participants’ accuracy $( p = . 0 3 1 )$ . When support information was not received, the difference in confidence between correctly and incorrectly classified items was significantly greater in MP than in HP $( \mathbf { M P } ; 2 . 9 9 \pm 1 . 5 7 ; \mathbf { H P } ; 1 . 9 1 \pm 1 . 2 9 ; p < . 0 0 1 )$ . When support information was received, the corresponding difference was descriptively greater in MP than in HP, although this contrast did not reach significance after correction for multiple comparisons (MP: $2 . 0 6 \pm 1 . 4 1 ; \mathrm { H P : 1 . 5 0 \pm 1 . 2 3 ; } p = . 0 8 8 )$ .

We also observe a significant interaction between Budget, presence of support information, and participants’ accuracy $( p = . 0 0 3 )$ . When support information was received, the difference in confidence between correct and incorrect classifications was significantly greater in HB than in LB (HB: $1 . 9 5 \pm 1 . 3 7 ; \mathrm { L B } \mathrm { : } 1 . 6 1 \pm 1 . 3 2 ; p = . 0 4 3 )$ . In contrast, when support information was not received, the corresponding difference did not significantly differ between HB and LB $( \mathrm { H B } \colon 2 . 0 9 \pm 1 . 6 2 ;$ $\mathbf { L B } \colon 2 . 6 2 \pm 1 . 4 2 ; p = . 1 1 0 )$ . One possible explanation is that, under the higher budget, support information was provided for a broader range of items, potentially including easier ones.

Instead, in the Email study, the only significant interaction involving participants’ accuracy (and thus directly relevant to confidence discrimination) was the interaction with Support $( p = . 0 0 5 )$ Specifically, the difference in confidence between correctly and incorrectly classified items was significantly greater in MP than in HP $( \mathrm { M P } ; 0 . 7 1 \pm 0 . 8 6 ; \mathrm { H P } ; 0 . 4 3 \pm 0 . 8 9 ; \textit { p } = . 0 0 5 )$

Overall, across the two studies, participants in MP tended to show greater confidence discrimination than those in HP, assigning relatively higher confidence to correct than to incorrect classifications. Notably, this pattern was also observed in the Email study, in which the selective disclosure method did not significantly increase overall classification accuracy.

Alignment between ClassWise-selected and human-selected disclosure. We explore alignment between ClassWise-selected and human-selected information requests from two perspectives. First, we analyze differences between the frequency of information disclosure by the policy and humans using one-sample t-tests. Participants in the HP conditions request auxiliary information on fewer trials than the method would provide it in both the ImageNet-16H and Email tasks (see Table 7).

Second, focusing on trials in which HP participants requested information, we descriptively examine how often these requests correspond to items for which ClassWise would also have provided information. In the $\mathtt { I m a g e N e t - 1 6 H }$ task, the proportion of human-requested trials that were also selected by $\mathtt { C l a s s W i s e }$ was $0 . 4 0 \pm 0 . 3 0$ in LB and $0 . 8 7 \pm 0 . 1 3$ in HB. In the Email task, the corresponding proportions were $0 . 3 1 { \pm } 0 . 2 6$ and 0.56 0.20, respectively. These proportions indicate that human requests only partially overlap with policy-selected disclosure, although the extent of this overlap differs considerably across tasks and budget conditions.

Taken together, these analyses indicate that humans and ClassWise allocate disclosure differently. Participants request information on fewer trials than the policy provides it, which may limit the potential benefits of human-selected disclosure when useful assistance remains unrequested. Determining why participants request assistance relatively infrequently (e.g., because of overconfidence in their own classification ability or uncertainty about the value of assistance) requires further study, as the present user studies are designed to compare ClassWise-selected and human-selected disclosure at the level of the complete strategies rather than identify the mechanisms underlying human request behavior. In addition, the descriptive overlap analysis indicates that the two strategies do not necessarily select the same trials for disclosure. These exploratory results therefore suggest that differences between policy-selected and human-selected disclosure concern both how frequently information is requested and which items receive it, motivating future work on the consequences of these differences for human-AI team performance.

## E.3 Q3: HOW DO PARTICIPANTS INTERACT WITH OUR POLICY?

Take-home message: The learned policy is more beneficial whenever humans follow the advice, and self-reported trust in the AI system is broadly consistent with advice-adherence trends.

In addition to the analyses of advice adherence discussed in the main text, for which we report full details here, we analyze participants’ responses to the trust scale, providing a complementary, self-reported perspective on their trust in the system providing auxiliary information.

<table><tr><td colspan="7">Descriptive statistics</td></tr><tr><td colspan="3"></td><td colspan="2">ImageNet-16H</td><td colspan="2">Email</td></tr><tr><td>Support</td><td>Budget</td><td>Information</td><td>Incorrect</td><td>Correct</td><td>Incorrect</td><td>Correct</td></tr><tr><td>Human Policy</td><td></td><td>Low Budget Not received</td><td> $4 . 2 2 \pm 1 . 4 5$ </td><td> $6 . 3 0 \pm 0 . 6 7$ </td><td> $5 . 2 0 \pm 1 . 2 9$ </td><td> $5 . 5 6 \pm 0 . 9 9$ </td></tr><tr><td>Human Policy</td><td>Low Budget</td><td>Received</td><td> $3 . 2 4 \pm 1 . 4 9$ </td><td> $4 . 5 8 \pm 1 . 4 5$ </td><td> $5 . 3 2 \pm 1 . 3 0$ </td><td> $5 . 6 7 \pm 0 . 9 8$ </td></tr><tr><td>Human Policy</td><td></td><td>High Budget Not received</td><td> $4 . 7 4 \pm 1 . 6 0$ </td><td> $6 . 4 5 \pm 0 . 5 0$ </td><td> $5 . 2 4 \pm 1 . 2 8$ </td><td> $5 . 6 9 \pm 0 . 8 9$ </td></tr><tr><td>Human Policy</td><td>High Budget</td><td>Received</td><td> $3 . 0 8 \pm 1 . 4 6$ </td><td> $4 . 7 5 \pm 1 . 4 8$ </td><td> $5 . 0 0 \pm 1 . 3 1$ </td><td> $5 . 4 2 \pm 1 . 1 9$ </td></tr><tr><td></td><td></td><td>Machine Policy Low Budget Not received</td><td> $3 . 1 1 \pm 1 . 4 5$ </td><td> $6 . 2 2 \pm 0 . 6 7$ </td><td> $4 . 8 2 \pm 1 . 2 8$ </td><td> $5 . 5 9 \pm 0 . 8 7$ </td></tr><tr><td></td><td>Machine Policy Low Budget</td><td>Received</td><td> $3 . 6 8 \pm 1 . 6 8$ </td><td> $5 . 5 8 \pm 1 . 0 8$ </td><td> $5 . 1 1 \pm 1 . 2 6$ </td><td> $5 . 5 2 \pm 0 . 9 3$ </td></tr><tr><td>Machine Policy High Budget Not received</td><td></td><td></td><td> $3 . 8 4 \pm 1 . 9 6$ </td><td> $6 . 6 3 \pm 0 . 4 2$ </td><td> $4 . 7 4 \pm 1 . 1 6$ </td><td> $5 . 5 2 \pm 0 . 7 8$ </td></tr><tr><td>Machine Policy High Budget</td><td></td><td>Received</td><td> $3 . 6 1 \pm 1 . 5 9$ </td><td> $5 . 7 9 \pm 0 . 8 7$ </td><td> $4 . 7 7 \pm 1 . 3 6$ </td><td> $5 . 6 2 \pm 0 . 9 5$ </td></tr></table>

Regression coefficients
<table><tr><td></td><td colspan="4">ImageNet-16H</td><td colspan="4">Email</td></tr><tr><td>Fixed effect</td><td>b</td><td>SE</td><td>t</td><td>p</td><td>b</td><td>SE</td><td>t</td><td>p</td></tr><tr><td>Intercept</td><td>4.504</td><td>.096</td><td>46.79</td><td>&lt; .001</td><td>5.168</td><td>.069</td><td>75.19</td><td>&lt; .001</td></tr><tr><td>Support (HP)</td><td>.732</td><td>.150</td><td>4.89</td><td>&lt; .001</td><td></td><td>.440.134</td><td>3.30</td><td>.001</td></tr><tr><td>Budget (LB)</td><td></td><td>-.383.150</td><td>-2.56</td><td>.011</td><td></td><td>.064.133</td><td>.48</td><td>.632</td></tr><tr><td>Information received</td><td></td><td>-.062.093</td><td>-.67</td><td>.503</td><td>-.099.082</td><td></td><td>-1.22</td><td>.223</td></tr><tr><td>Accuracy</td><td></td><td>1.480.075</td><td>19.81</td><td>&lt; .001</td><td></td><td>.410.047</td><td>8.66</td><td>&lt; .001</td></tr><tr><td>Support × Budget</td><td></td><td>.192.297</td><td>.65</td><td>.518</td><td>-.134.266</td><td></td><td>-.50</td><td>.615</td></tr><tr><td>Support × Information</td><td></td><td>-.989.178</td><td>-5.56</td><td>&lt; .001</td><td>-.403.165</td><td></td><td>-2.44</td><td>.015</td></tr><tr><td>Budget × Information</td><td></td><td>.544.175</td><td>3.10</td><td>.002</td><td></td><td>.138.161</td><td>.86</td><td>.391</td></tr><tr><td>Support × Accuracy</td><td>-.716.136</td><td></td><td>-5.27</td><td>&lt; .001</td><td>-.362.093</td><td></td><td>-3.91</td><td>&lt; .001</td></tr><tr><td>Budget × Accuracy</td><td></td><td>.261.136</td><td>1.92</td><td>.055</td><td>-.087.092</td><td></td><td>-.95</td><td>.344</td></tr><tr><td>Information × Accuracy</td><td></td><td>-.438.103</td><td>-4.24</td><td>&lt; .001</td><td></td><td>.083.089</td><td>.94</td><td>.349</td></tr><tr><td>Support × Budget × Information</td><td></td><td>-.122.348</td><td>-.35</td><td>.727</td><td></td><td>.449.320</td><td>1.40</td><td>.160</td></tr><tr><td>Support × Budget × Accuracy</td><td></td><td>-.049.270</td><td>-.18</td><td>.856</td><td>-.066.184</td><td></td><td>-.36</td><td>.721</td></tr><tr><td>Support × Information × Accuracy</td><td></td><td>.428.199</td><td>2.15</td><td>.031</td><td></td><td>.225.178</td><td>1.26</td><td>.207</td></tr><tr><td>Budget × Information × Accuracy</td><td></td><td>-.587.197</td><td>-2.98</td><td>.003</td><td>-.065.176</td><td></td><td>-.37</td><td>.710</td></tr><tr><td>Support × Budget × Information × Accuracy</td><td>-.180.390</td><td></td><td>-.46</td><td>.646</td><td>-.032.350</td><td></td><td>-.09</td><td>.928</td></tr></table>

<table><tr><td colspan="8">Post-hoc contrasts</td></tr><tr><td></td><td colspan="4">ImageNet-16H</td><td colspan="4">Email</td></tr><tr><td>Contrast</td><td>Estimate SE</td><td></td><td>z</td><td>p</td><td>Estimate</td><td>SE</td><td>z p</td></tr><tr><td>HP vs. MP, information not received</td><td></td><td></td><td>-.716.136-5.27</td><td>&lt; .001</td><td></td><td></td><td></td></tr><tr><td>HP vs. MP, information received</td><td></td><td></td><td>-.288.143-2.02</td><td>.088</td><td></td><td></td><td></td></tr><tr><td>LB vs. HB, information not received</td><td></td><td></td><td>.261.136 1.92</td><td>.110</td><td></td><td></td><td></td></tr><tr><td>LB vs. HB, information received</td><td></td><td></td><td>-.326.142-2.30</td><td>.043</td><td></td><td></td><td></td></tr><tr><td>HP vs. MP</td><td></td><td></td><td></td><td></td><td>-.249.089-2.79.005</td><td></td><td></td></tr></table>

Table 6: Descriptive statistics and results of the linear mixed-effects models predicting confidence ratings from Support, Budget, presence of information, classification accuracy, and their interactions. Descriptive statistics represent participant-level means standard deviations. Post-hoc contrasts compare confidence discrimination, defined as the difference in confidence between correct and incorrect classifications (post-hoc contrasts are reported only for significant effects involving classification accuracy; dashes indicate that no post-hoc contrast was conducted because the corresponding effect was not significant)

Advice adherence. We assess participants’ adherence to the AI advice provided as support information by fitting a logistic mixed-effects model predicting whether participants’ classifications are consistent with the AI advice from Support, Budget, and their interaction. This analysis was presented in the main text and will not be further discussed, but see Table 8 for the full results. <sup>16</sup>

<table><tr><td>Study</td><td></td><td>Budget Requested assistance MP assistance</td><td></td><td>95% CI</td><td>t</td><td>df</td><td>p</td></tr><tr><td>ImageNet-16H</td><td>Low</td><td> $3 . 7 3 \pm 2 . 1 2$ </td><td>6</td><td>[3.21, 4.25]</td><td>-8.70</td><td></td><td> $6 5 \_ < . 0 0 1$ </td></tr><tr><td>ImageNet-16H</td><td>High</td><td> $5 . 9 9 \pm 3 . 2 0 $ </td><td>14</td><td>[5.22, 6.75]</td><td>-20.82</td><td></td><td> $6 8 \_ < . 0 0 1$ </td></tr><tr><td>Email</td><td>Low</td><td> $3 . 9 3 \pm 1 . 9 7$ </td><td>5</td><td>[3.47, 4.39]</td><td>-4.60</td><td></td><td> $7 1 \ < . 0 0 1$ </td></tr><tr><td>Email</td><td>High</td><td> $6 . 4 1 \pm 4 . 6 3$ </td><td>9</td><td>[5.25, 7.56]</td><td>-4.48</td><td></td><td> $6 3 \_ < . 0 0 1$ </td></tr></table>

Table 7: Number of assistance requests made by participants in the Human-Policy condition compared with the number of trials on which assistance was provided in the corresponding Machine-Policy condition. Requested assistance is reported as mean standard deviation. Confidence intervals and t tests refer to one-sample tests comparing the observed number of requests with the number of assistance opportunities provided in the corresponding MP condition.
<table><tr><td colspan="7">Descriptive statistics</td></tr><tr><td></td><td></td><td colspan="2">ImageNet-16H</td><td colspan="3">Email</td></tr><tr><td>Support condition Budget condition with support information</td><td></td><td>Proportion of agreement</td><td></td><td>Proportion of agreement with support information</td><td></td><td></td></tr><tr><td>Human Policy Human Policy</td><td colspan="2">Low Budget</td><td colspan="2"> $0 . 9 2 \pm 0 . 2 4$ </td><td colspan="3"> $0 . 9 6 \pm 0 . 1 7$ </td></tr><tr><td></td><td colspan="2">High Budget</td><td colspan="2"> $0 . 9 4 \pm 0 . 1 3$ </td><td colspan="3"> $0 . 9 5 \pm 0 . 1 2$ </td></tr><tr><td>Machine Policy</td><td colspan="2">Low Budget</td><td colspan="2"> $0 . 9 1 \pm 0 . 1 2$ </td><td colspan="3"> $0 . 8 1 \pm 0 . 1 7$ </td></tr><tr><td>Machine Policy</td><td colspan="2">High Budget</td><td colspan="2"> $0 . 9 2 \pm 0 . 1 0$ </td><td colspan="3"> $0 . 8 9 \pm 0 . 1 4$ </td></tr><tr><td colspan="9">Regression coefficients</td></tr><tr><td colspan="9"></td></tr><tr><td rowspan="2">Fixed effect</td><td>OR</td><td>ImageNet-16H 95% CI</td><td>Z</td><td>p</td><td>OR</td><td>95% CI</td><td>Z</td><td>p</td></tr><tr><td>48.02</td><td></td><td>11.85</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Intercept</td><td>[25.31, 91.13]</td><td>1.73</td><td>&lt; .001 .084</td><td>25.08</td><td>[13.88, 45.30]</td><td></td><td> $1 0 . 6 8 \ \mathrm { ~ < ~ } . 0 0 1$ </td></tr><tr><td colspan="2">Support condition (HP)</td><td>1.72</td><td>[0.93, 3.18]</td><td></td><td>4.63</td><td>[2.14, 10.04]</td><td>3.89</td><td>&lt; .001</td></tr><tr><td colspan="2">Budget condition (LB)</td><td>0.70 0.59</td><td>[0.37, 1.31]</td><td>-1.13 .260</td><td>0.83</td><td>[0.38, 1.81]</td><td>-0.47</td><td>.636</td></tr><tr><td colspan="2">Support × Budget</td><td>[0.18, 1.97]</td><td>-0.86</td><td>.392</td><td>3.31</td><td>[0.72, 15.29]</td><td>1.53</td><td>.126</td></tr></table>

Table 8: Descriptive statistics and results of the logistic mixed-effects regressions predicting participants’ agreement with the support information from Support, Budget, and their interaction (OR: odds ratio).

Trust in the AI system. We analyze the trust index, computed by averaging participants’ ratings across the eight items of the trust scale, using linear models in which trust index values are predicted by Support condition, Budget condition, and their interaction (see Table 9 for the full results).

In the ImageNet-16H study, trust index values do not significantly differ between MP and HP $( \mathrm { M P } ; 3 . 0 6 \pm 0 . 7 3 ; \mathrm { H P } ; 2 . 9 0 \pm 0 . 7 5 ; p = . 0 6 4 )$ ). while they are significantly higher in HB than in LB $( \mathrm { H B } ; 3 . 1 3 \pm 0 . 7 0 ; \mathrm { L B } ; 2 . 8 3 \pm 0 . 7 5 ; p < . 0 0 1 )$ , while the interaction between Support and Budget is not significant (p = .096). In contrast, in the Email study, trust index values are significantly lower in MP than in HP (MP: 3.04 0.82; HP: 3.32 0.70; p = .003). HB and LB do not significantly differ (p = .861), whereas the interaction between Support and Budget is significant $( p = . 0 4 6 )$ Follow-up comparisons indicate that the difference between MP and HP is significant under LB (MP-LB: 2.95  0.81; HP-LB: 3.40  0.70; p = .002), but not under HB (MP-HB: 3.15  0.82; HP-HB: 3.23  0.70; p = 1).

This pattern is broadly consistent with the results on adherence on support information: participants in MP reported similar trust in the AI system to those in HP in the ImageNet-16H study, whereas in the Email study, particularly under the lower budget, participants in MP reported lower values than those in HP.

Counterfactual benchmark. We report here the results of the counterfactual analysis, in which we assess the performance that MP participants would have achieved had their predictions been overruled with the ML advice when provided (for the ImageNet-16H task, we convert each prediction set into a single prediction by retaining its highest-scoring class, which is treated as the counterfactual response). This analysis was presented in the main text and will not be further discussed, but see Table 10 for the full results. 17

<table><tr><td colspan="4">Descriptive statistics</td></tr><tr><td></td><td></td><td>ImageNet-16H</td><td>Email</td></tr><tr><td>Support condition Budget condition</td><td></td><td>Trust index</td><td>Trust index</td></tr><tr><td>Human Policy</td><td>Low Budget</td><td> $2 . 6 7 \pm 0 . 7 2$ </td><td> $3 . 4 0 \pm 0 . 7 0$ </td></tr><tr><td>Human Policy</td><td>High Budget</td><td> $3 . 1 2 \pm 0 . 7 1$ </td><td> $3 . 2 3 \pm 0 . 7 0$ </td></tr><tr><td>Machine Policy</td><td>Low Budget</td><td> $2 . 9 8 \pm 0 . 7 5$ </td><td> $2 . 9 5 \pm 0 . 8 1$ </td></tr><tr><td>Machine Policy</td><td>High Budget</td><td> $3 . 1 4 \pm 0 . 6 9$ </td><td> $3 . 1 5 \pm 0 . 8 2$ </td></tr></table>

<table><tr><td colspan="10">Regression coefficients</td></tr><tr><td rowspan="2">Fixed effect</td><td colspan="4">ImageNet-16H</td><td colspan="4">Email</td></tr><tr><td>b</td><td>SE</td><td>t</td><td>p</td><td>b</td><td>SE</td><td>t</td><td>p</td></tr><tr><td>Intercept</td><td>2.978</td><td>0.044</td><td>68.37</td><td>&lt; .001</td><td>3.182</td><td>0.046</td><td>69.12</td><td>&lt; .001</td></tr><tr><td>Support (HP)</td><td>-0.162</td><td>0.087</td><td>-1.86</td><td>.064</td><td>0.273</td><td>0.092</td><td>2.96</td><td>.003</td></tr><tr><td>Budget (LB)</td><td>-0.303</td><td>0.087</td><td>-3.48</td><td>&lt; .001</td><td>-0.016</td><td>0.092</td><td>-0.18</td><td>.861</td></tr><tr><td>Support × Budget</td><td>-0.291</td><td>0.174</td><td>-1.67</td><td>.096</td><td>0.369</td><td>0.184</td><td>2.00</td><td>.046</td></tr></table>

<table><tr><td colspan="4">Post-hoc comparisons: Email</td></tr><tr><td>Contrast</td><td>Estimate</td><td>SE</td><td>t p</td></tr><tr><td>HP-LB vs. MP-LB</td><td>0.457</td><td>0.127 3.61</td><td>.002</td></tr><tr><td>HP-LB vs. HP-HB</td><td>0.168 0.130</td><td>1.29</td><td>1</td></tr><tr><td>HP-LB vs. MP-HB</td><td>0.257 0.130</td><td>1.98</td><td>.294</td></tr><tr><td>MP-LB vs. HP-HB</td><td>-0.289 0.131</td><td>-2.21</td><td>.167</td></tr><tr><td>MP-LB vs. MP-HB</td><td>-0.201 0.130</td><td>-1.54</td><td>.745</td></tr><tr><td>HP-HB vs. MP-HB</td><td>0.088 0.134</td><td>0.66</td><td>1</td></tr></table>

Table 9: Descriptive statistics and results of the linear models predicting trust index values from Support, Budget, and their interaction. Descriptive statistics represent means standard deviations. Post-hoc comparisons for the Email study are Bonferroni-corrected for six comparisons.

## E.4 Q4: HOW DOES CHANGING THE INFORMATION CONTENT AFFECT OUR POLICY?

Take-home message: The benefit of selective disclosure depends on how informative the support information is. At $\alpha = 0 . 0 5$ , where prediction sets are small and informative, ClassWise prioritizes them at every budget; as α decreases and uninformative full sets become frequent, the gain of any disclosure policy shrinks, and at $\alpha = 0 . 0 1$ no policy separates from full disclosure.

In the ImageNet-16H setting the support information S is the conformal prediction set $C ( \mathbf { x } )$ whose size is governed by the miscoverage level α: smaller α enforces higher coverage and therefore yields larger sets. The size of $C ( \mathbf { x } )$ directly controls how informative disclosure can be. In particular, a set that coincides with the full label space, $| C ( \mathbf { x } ) | = | \mathcal { V } | = 1 6$ , is compatible with every class and thus carries no discriminative signal: it cannot help the decision-maker refine their judgement, so its Value of Information is non-positive and a well-behaved policy should never spend budget disclosing it. The prevalence of such full sets consequently upper-bounds the benefit attainable at a given α.

We examine the role of α along three axes: (i) we analyse how the distribution of set sizes varies with α, motivating the choice of $\alpha = 0 . 0 5$ in the main paper; (ii) we report the budget–accuracy curves for the remaining levels $\alpha \in \{ 0 . 0 1 , 0 . 0 2 \}$ and comment on how the achievable benefit shrinks as sets grow; and (iii) we inspect which sets our policy actually chooses to disclose as a function of the budget, showing that at $\alpha = 0 . 0 5$ it essentially never wastes budget on uninformative full sets.

<table><tr><td colspan="3">Descriptive statistics</td></tr><tr><td></td><td>ImageNet-16H</td><td>Email</td></tr><tr><td>Support condition</td><td>Accuracy</td><td>Accuracy</td></tr><tr><td>Human Policy</td><td> $0 . 7 7 \pm 0 . 1 3$ </td><td> $0 . 8 3 \pm 0 . 1 3$ </td></tr><tr><td>Machine Policy</td><td> $0 . 8 0 \pm 0 . 1 1$ </td><td> $0 . 8 2 \pm 0 . 1 2$ </td></tr><tr><td>Machine Policy – Counterfactual benchmark</td><td> $0 . 8 2 \pm 0 . 1 1$ </td><td> $0 . 8 5 \pm 0 . 1 0$ </td></tr></table>

<table><tr><td colspan="9">Regression coefficients</td></tr><tr><td rowspan="2">Fixed effect</td><td colspan="4">ImageNet-16H</td><td colspan="4">Email</td></tr><tr><td>OR</td><td>95% CI</td><td>Z</td><td>p</td><td>OR</td><td>95% CI</td><td>Z</td><td>p</td></tr><tr><td>Intercept</td><td>14.40</td><td>[10.01, 20.71]</td><td>14.38</td><td>&lt; .001</td><td>9.18</td><td>[7.35, 11.48]</td><td>19.49</td><td>&lt; .001</td></tr><tr><td>Support - HP</td><td>0.59</td><td>[0.44, 0.80]</td><td>-3.40</td><td>&lt; .001</td><td>0.87</td><td>[0.63, 1.19]</td><td>-0.87</td><td>.385</td></tr><tr><td>Support - MP</td><td>1.05</td><td>[0.84, 1.32]</td><td>0.43</td><td>.668</td><td>0.82</td><td>[0.66, 1.03]</td><td>-1.71</td><td>.087</td></tr><tr><td>Budget (LB)</td><td>0.44</td><td>[0.35, 0.56]</td><td>-6.90</td><td>&lt; .001</td><td>0.76</td><td>[0.59, 0.97]</td><td>-2.23</td><td>.026</td></tr><tr><td>Support - HP × Budget</td><td>1.41</td><td>[0.77, 2.57]</td><td>1.12</td><td>.264</td><td>1.12</td><td>[0.59, 2.12]</td><td>0.35</td><td>.727</td></tr><tr><td>Support - MP × Budget</td><td>1.07</td><td>[0.68, 1.70]</td><td>0.31</td><td>.758</td><td>1.00</td><td>[0.64, 1.57]</td><td>0.02</td><td>.983</td></tr></table>

<table><tr><td colspan="9">Post-hoc comparisons</td></tr><tr><td></td><td colspan="4">ImageNet-16H</td><td colspan="4">Email</td></tr><tr><td>Contrast</td><td>OR</td><td>SE</td><td>Z</td><td>p</td><td>OR</td><td>SE</td><td>Z</td><td>p</td></tr><tr><td>HP vs. MP</td><td>0.75</td><td>0.09</td><td>-2.33</td><td>.059</td><td>1.03</td><td>0.13</td><td>0.21</td><td>1</td></tr><tr><td>HP vs. MPCFT</td><td>0.61</td><td>0.08</td><td>-4.00</td><td>&lt; .001</td><td>0.79</td><td>0.10</td><td>-1.84</td><td>.196</td></tr><tr><td>MP vs. MPCFT</td><td>0.81</td><td>0.07</td><td>-2.34</td><td>.058</td><td>0.77</td><td>0.06</td><td>-3.25</td><td>.004</td></tr></table>

Table 10: Descriptive statistics and results of the logistic mixed-effects regressions comparing observed accuracy in HP and MP with counterfactual accuracy under full advice adherence on automatically disclosed support information (MPCFT). Descriptive statistics represent participant-level means standard deviations. Post-hoc comparisons are averaged across budget conditions and Bonferroni-corrected for three comparisons (OR: odds ratio).

Table 11: Distribution of conformal prediction set sizes on the ImageNet-16H dataset (n = 1200 images, = 16 classes) for the three miscoverage levels α considered in our experiments. Smaller α enforces higher coverage and thus yields larger sets; note in particular the mass on the full set $( | C ( \mathbf { x } ) | = 1 6 )$ , which carries no information for the decision-maker. Dashes denote empty bins.
<table><tr><td></td><td colspan="10">Set size  $| C ( \mathbf { x } ) |$ </td><td colspan="5"></td><td colspan="2">Summary</td></tr><tr><td>α</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>Mean</td><td>Sing.%</td></tr><tr><td>0.01</td><td>664</td><td>92</td><td>32</td><td>9</td><td>10</td><td>3</td><td>1</td><td>1</td><td>一</td><td>1</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td><td>387</td><td>6.05</td><td>55.3</td></tr><tr><td>0.02</td><td>735</td><td>148</td><td>54</td><td>31</td><td>16</td><td>13</td><td>9</td><td>8</td><td>4</td><td>4</td><td>2</td><td>1</td><td>一</td><td>5</td><td>1</td><td>169</td><td>3.75</td><td>61.3</td></tr><tr><td>0.05</td><td>804</td><td>200</td><td>99</td><td>55</td><td>24</td><td>13</td><td>3</td><td>一</td><td>一</td><td>一</td><td>_</td><td></td><td>一</td><td></td><td>一</td><td>2</td><td>1.64</td><td>67.0</td></tr></table>

(i) Set-size distribution and the choice of α. Table 11 reports the distribution of set sizes on the ImageNet-16H dataset for the three miscoverage levels considered in our experiments. As expected, the smaller α, the larger the prediction sets: the mean set size shrinks from 6.05 at α = 0.01 to 1.64 at $\alpha = 0 . 0 5$ . The focus for our purposes is not the exact size but whether the disclosed set is informative: a large set provides little guidance to the decision-maker, and in the limit the full set $( | C ( \mathbf { x } ) | = 1 6 )$ carries no signal at all. From this perspective the three levels differ sharply.

At α = 0.05 the support information is almost always useful: essentially all sets have size below 7, and only 2 out of 1200 images fall on large, uninformative sets $( | C ( \mathbf { x } ) | > 1 0 )$ $\mathrm { A t } \alpha = 0 . 0 1$ and $\alpha = 0 . 0 2 .$ , instead, a substantial fraction of the sets is large and uninformative (the full set alone accounts for 387 and 169 images respectively) so a non-negligible share of any disclosed feedback would be effectively useless. This is precisely why we adopt $\alpha = 0 . 0 5$ in the main paper: it is the level at which disclosure is almost always informative, so that revealing a set genuinely helps the human decision rather than wasting budget on redundant, uninformative feedback.

![](images/ef2e008bebbc1f16e2b9744a5fd1027181b283d1dcfd1f53b0e592577373bb32.jpg)  
Figure 9: Budget–accuracy curves on the ImageNet-16H dataset for the two additional miscoverage levels $\alpha \in \{ 0 . 0 1 , 0 . 0 2 \}$ . As α decreases the prediction sets grow larger and less informative, lowering the average benefit of full disclosure (FullDisc 80.5% against 85.5% at $\alpha = 0 . 0 5 )$

(ii) Disclosure accuracy across miscoverage levels. Fig. 9 reports the budget–accuracy curves for $\alpha \in \{ 0 . 0 5 , 0 . 0 2 , 0 . 0 1 \}$ , complementing the results in the main paper (Fig. 2). A first effect of lowering α is visible already at the endpoints: as the prediction sets grow larger and less discriminative, the average benefit of full disclosure shrinks. Full disclosure $( \mathtt { F u l l D i s c } )$ attains about 80.5% at both $\alpha = 0 . 0 1$ and $\alpha = 0 . 0 2$ , i.e. only about 3.3 points above the no-disclosure accuracy (NoDisc 77.2%), against the 8.3-point gap observed at $\alpha = 0 . 0 5$ . Larger sets thus carry less usable signal, and the ceiling that any disclosure policy can reach is correspondingly lower.

At $\alpha \ = \ 0 . 0 2$ , selective disclosure remains effective: ClassWise reaches the accuracy of full disclosure already at $B = 0 . 5$ , whereas Confidence needs $B = 0 . 8$ and Random the full budget. Both policies then slightly exceed FullDisc at higher budgets $( \mathbb { C } 1 \mathsf { a s s w i s e } \approx 8 1 . 0 \%$ at $B =$ $0 . 7 , \mathsf { C o n f } \colon$ idence 81.3% at $B = 0 . 9 )$ , although these gains are within the confidence bands. At $\alpha = 0 . 0 1$ , instead, no policy separates from full disclosure: ClassWise, Confidence and even Random track FullDisc within their confidence bands, and ClassWise falls slightly below it at high budgets ( 79.3% at $B = 0 . 9 $ , about three images).

We attribute this behaviour to the set-size distribution at $\alpha = 0 . 0 1$ (Table 11). When α is very small, the prediction sets are large: the mean set size is 6.05, and about one third of the images (387/1200) receive the full, completely uninformative set. First, fewer instances receive an informative set, so the mass of instances with positive VoI is smaller and a selective policy has less signal to exploit. Second, estimating the VoI becomes harder: the risk models must tell apart many large, overlapping sets whose effect on the human decision is weak and noisy. Ranking errors then cause the policy to withhold informative sets in favour of uninformative ones, which explains why it can fall slightly below full disclosure at high budgets. Overall, these results indicate that the benefit of selective disclosure depends on how informative the support information is: when it carries little signal, as at $\alpha = 0 . 0 1$ , there is little for any policy to gain over simply disclosing everything.

(iii) Does the policy disclose informative sets? Since a large prediction set carries little signal, a good disclosure policy should spend its budget on small, informative sets and avoid the uninformative full set $( | C ( \mathbf { x } ) | = 1 6 )$ whenever possible. Here we ask whether the ClassWise policy does so, and how this behaviour changes with the budget. Fig. 10 shows, for each budget B, the fraction of test instances of each set size to which the policy discloses the support information: a policy indifferent to set size would disclose a fraction B of the instances of every size.

At $\alpha = 0 . 0 5$ , the policy clearly favours small sets at every budget. Full sets are essentially absent, since only 2 exist in the whole dataset. $\mathrm { A t } \alpha = 0 . 0 2$ and $\alpha = 0 . 0 1$ , the picture becomes budgetdependent. At small budgets the policy still prioritizes small sets, showing that the estimated VoI ranks informative sets first. As the budget grows, however, the policy increasingly discloses full sets as well, reaching on average 77 of the 85 full sets in the test set (91%) at $B = 1$ for $\alpha = 0 . 0 1$ , and all 38 of the 38 (100%) full sets for $\alpha = 0 . 0 2$

![](images/50a484263127ddd60678b759cf8c0bc08c41eb294467c1c332a3d0330055a8b2.jpg)

![](images/e91cd3e1dadb4c414ec63efc95bf2be48e2694b265f41720b9cb34244808775e.jpg)

![](images/8a8a157095564c80cba85622facd49594015e770a821be705f29ce366428f794.jpg)  
Figure 10: Prediction-set sizes disclosed by the ClassWise policy on ImageNet-16H, at α $\{ 0 . 0 5 , 0 . 0 2 , 0 . 0 1 \}$ . For each set size, the curves show the share of the test set that the policy discloses at budgets $B \in \{ \mathrm { 0 . 1 , 0 . 4 , 0 . 7 } \}$ (mean over 5 seeds, with 95% confidence intervals). The dashed line (FullDisc) is the share of the test set with that set size, i.e., what full disclosure would reveal, so the gap between a curve and the dashed line gives the instances of that size the policy withholds.

These results confirm $\alpha = 0 . 0 5$ as the most favourable operating point: the regime in which the policy can disclose informative feedback at every budget level. These motivate its use in the main paper.

## F EXTENDED RELATED WORK

Learning to defer and its extensions. (Madras et al., 2018) introduce Learning to Defer (LtD), casting human-AI collaboration as a mutually exclusive choice: for each instance, either the model predicts or the case is deferred to a human expert (for a broader overview of rejection and deferral, see Ruggieri & Pugnana (2025)). The framework has since been developed along several axes. On the theoretical side, (Mozannar & Sontag, 2020) derive consistent surrogate losses for the joint model-plus-deferral objective, (Verma & Nalisnick, 2022) propose a one-vs-all formulation with improved calibration of the deferral decision, and (Mozannar et al., 2023) study exact algorithms for learning who should predict. For further work developing LtD methods with theoretical guarantees, see Okati et al. (2021); Charusaie et al. (2024); Cao et al. (2023); Liu et al. (2024); Gao & Yin (2025); Li et al. (2017); Montreuil et al. (2025a; 2026); Fang & Nalisnick (2026). Other theoretical extensions of the LtD approach involved considering multiple-expert settings (Verma et al., 2023; Mao et al., 2023; Montreuil et al., 2025b; Mao et al., 2025; Zhang et al., 2026; Liu et al., 2026), multi-task settings (Pugnana et al., 2025; Montreuil et al., 2025c), and aspects such as rejection of unexplainable decisions (Stradiotti et al., 2025).

On the empirical side, (Hemmer et al., 2023) examine how delegation affects task performance and satisfaction with real participants, (Palomba et al., 2025) evaluate deferring systems through a causal lens, and (Bondi et al., 2022) investigate how communicating a deferral affects human accuracy. Other works have examined the impact class distribution on rejected instances (Pugnana et al., 2024), with Pesenti et al. (2026) examining how class imbalances may affect users’ interactions with LtD systems. Deferral and rejection have also been studied in healthcare (Strong et al., 2025; Kompa et al., 2021), vehicle engineering (Hendrickx et al., 2021), and question answering (Montreuil et al., 2026). The defining feature of LtD is that authority over the final decision is allocated: on deferred instances the model abstains, and on the remaining ones the human is bypassed entirely. Our setting is structurally different. The human is always the decision-maker, and what is allocated is not authority but information: the policy decides what the human sees, never what the human decides. Consequently our risk is defined over human actions in two information regimes (Eq. (1)), rather than over a model prediction and a human prediction.

AI-assisted decision-making. A second line of work keeps the human as the final decision-maker and asks what form the support should take. Prediction sets are a prominent example: (Straitouri & Rodriguez, 2024) design decision-support systems based on counterfactual prediction sets, (De Toni et al., 2024) construct prediction sets that target the expert’s accuracy rather than coverage, and (Cresswell et al., 2024) show, in a pre-registered randomized controlled trial, that conformal prediction sets improve human accuracy over fixed-size (top-k) sets with the same coverage. (Schemmer et al., 2023) instead conceptualize appropriate reliance on AI advice and study how explanations affect it. Broader syntheses are provided by (Punzi et al., 2024), who survey learning paradigms for hybrid decision-making systems, and by (Hemmer et al., 2025), who conceptualize human-AI complementarity and review its sources and the empirical evidence for it.

Closer to our setting, a smaller body of work decides, instance by instance, whether (and which) support to show. (Noti & Chen, 2023) study a regression task (pretrial risk assessment) and learn from past human predictions a policy that, given the case, the algorithmic risk score and the human’s initial estimate, reveals the score only when it is predicted to be more accurate than that estimate; in a large-scale experiment, this improves human predictions over always showing the score. (Ma et al., 2023) do not learn the disclosure rule itself: they estimate the decision-maker’s correctness likelihood on each instance by applying an approximation of their individual decision rules to similar labelled cases, and compare it with the AI’s calibrated confidence; whenever the human is predicted to be more likely correct, the AI recommendation is either withheld (only its explanation is shown) or revealed only after an independent human judgement. Other works choose among several forms of support. (Bhatt et al., 2025) cast this choice as a stochastic contextual bandit and learn online, separately for each new decision-maker, a policy that selects, for each input, the form of support (e.g., none, a model or LLM prediction, expert consensus) expected to minimize that individual’s error. (Buc¸inca et al., 2024) instead apply offline reinforcement learning to previously collected interaction data to choose among no assistance, explanation only, recommendation with explanation, and on-demand advice, optimizing for immediate accuracy, for the decision-maker’s learning, or for both; their policies condition on a discrete state that includes the decision-maker’s need for cognition and task knowledge and, as a proxy for AI uncertainty, the ground-truth correctness of the AI recommendation. In the symmetric direction, (Pugnana et al., 2025) propose Learning to Ask, where a model decides under a budget when to query a human for enriched feedback, and characterize the optimal querying rule as a threshold on the risk difference between a standard and an enriched predictor; relatedly, (Wilder et al., 2020) and (Charusaie et al., 2024) train predictors that complement, or directly incorporate, human decisions.

Compared with the approaches that adapt the support shown to the human, LSD differs in three respects. First, disclosure is subject to a budget, which makes the optimal policy a non-negative, budget-dependent threshold on VoI; without a budget and with only two actions, the optimal policy of (Bhatt et al., 2025) reduces to disclosing whenever an individual-level VoI is positive, i.e., the unconstrained (B = 1) case of our threshold rule. Second, VoI is the causal reduction in human decision risk induced by disclosure, and therefore depends on how decision-makers actually use the disclosed information, whereas (Noti & Chen, 2023) and (Ma et al., 2023) condition disclosure on whether the AI is predicted to be more accurate than the human alone. Third, we establish when VoI is identifiable from data collected under the two regimes, and bound both the degradation of the resulting plug-in policies relative to no disclosure and their regret relative to the optimal policy.

Active feature acquisition. In Active Feature Acquisition (AFA), an agent decides which missing feature values to acquire, at a cost, in order to improve a downstream predictive model (Janisch et al., 2020; Rahbar et al., 2025; Saar-Tsechansky et al., 2009; Ji & Carin, 2007). (Saar-Tsechansky et al., 2009) acquire feature values at training time to improve model induction, whereas at test time (Ji & Carin, 2007) formulate cost-sensitive acquisition and classification jointly and (Janisch et al., 2020) cast classification with costly features as a sequential decision problem solved by reinforcement learning. (Rahbar et al., 2025) provide a unified view, organizing existing methods into embedded cost-aware predictors, model-based approaches, model-free policies, and hybrid strategies. Related in spirit are (Ma et al., 2019) and (Gong et al., 2019), who score candidate acquisitions with information-theoretic criteria. Our VoI instead follows the classical decision-theoretic notion of value of information as the expected reduction in decision loss (Howard, 1966), with one twist: for a Bayesian decision-maker this quantity is never negative, whereas ours is measured on actual human decisions and can be. Two differences matter. First, the acquisition target is a model’s prediction rather than a human’s decision, so the relevant risk is a model risk and the counterfactual “what would the predictor do without this feature” is directly computable, whereas the human counterfactual is not; this is precisely what forces the causal treatment of Appendix B. Second, AFA chooses which features to acquire, often sequentially and under a per-instance budget, whereas LSD makes a single binary decision on an indivisible block of support information, under a budget on the expected disclosure rate across cases.

Policy learning under budget constraints. The causal reading of our framework connects it to optimal policy learning (OPL), which studies how to learn, from data, treatment assignment rules that maximize a welfare objective (Manski, 2004; Athey & Wager, 2021; Cerulli, 2026). Without constraints, the first-best rule treats every unit whose conditional average treatment effect (CATE) is positive; (Kitagawa & Tetenov, 2018) propose empirical welfare maximization over constrained classes of treatment rules, (Bhattacharya & Dupas, 2012) show that when a budget caps the frac tion of treated units the optimal rule treats those whose CATE exceeds a quantile threshold, and (Carranza & Athey, 2025) extend policy learning to observational data from multiple sources using doubly robust estimators. Our Theorem 1 recovers a threshold rule of the same shape, with the VoI in the role of the CATE and the budget B in the role of the capacity constraint. As in budgetconstrained OPL, the threshold is non-negative, so disclosure is withheld whenever it is expected to harm the decision, even if budget remains; in our setting this case is far from marginal, since support information can mislead the decision-maker, e.g., by inducing over-reliance. The substantive difference is where the treatment acts. In standard OPL the treatment is applied to the unit whose outcome is measured. Here the treatment (disclosure) is applied to the decision-maker, while the outcome is the correctness of a decision about a case. This has two consequences we make explicit in Appendix B: the missing-potential-outcome problem arises at the level of the decision episode rather than the case, and is addressed by assigning different decision-makers to the two regimes, which requires those observed under each regime to be exchangeable; and SUTVA must be stated over episodes, since what must not interfere is the information disclosed to a given decision-maker in a given episode.

Heterogeneous treatment effect estimation. Since the VoI is the negative conditional average treatment effect (CATE) of disclosure on decision loss (Section 3.2), estimating it connects our work to the literature on heterogeneous treatment effects (Nogueira et al., 2022). Meta-learners reduce CATE estimation to standard supervised learning, either by fitting one outcome model per treatment arm (T-learner) or a single model with the treatment as input (S-learner) (Kunzel et al., 2019); more¨ refined estimators include the R-learner (Nie & Wager, 2021), doubly robust learners (Kennedy, 2023), causal forests (Wager & Athey, 2018) and neural architectures sharing representations across arms (Shalit et al., 2017; Shi et al., 2019). In our setting, doubly robust corrections are less critical than in observational studies, since each case is observed under both regimes and disclosure is assigned independently of the case covariates. Moreover, accurate effect estimation is neither nec essary nor sufficient for good causal decisions (Fernandez-Lor´ ´ıa & Provost, 2022): (Frauen et al., 2025) show that thresholding CATE estimators trained for estimation accuracy can yield suboptimal policies, since accuracy away from the decision boundary is irrelevant to the decision, and (Arno et al., 2026) show that, when individuals must be prioritized, recovering the ranking of treatment effects is easier than estimating their magnitude, and propose to learn it directly. Our setting requires both: under a binding budget, the optimal policy of Theorem 1 depends only on the ranking of the VoI, while the non-negative threshold depends on its sign. These are precisely the properties on which our plug-in guarantees (Section 3.3) and our ablation (Appendix E.1) focus.