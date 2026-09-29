# SIMPLEX DIFFUSION MODELS

Justin Deschenaux Google DeepMind EPFL

Alexandre Galashov Google DeepMind UCL Gatsby

Andrew Campbell Google DeepMind

Li Kevin Wenliang Google DeepMind

James Thornton Google DeepMind

Arnaud Doucet Google DeepMind

Valentin De Bortoli Google DeepMind

## ABSTRACT

Diffusion models have revolutionized generative modeling for continuous data through the gradual refinement of a belief state. This iterative refinement has not yet carried over to discrete diffusion models, which discard uncertainty at intermediate steps through categorical sampling (information collapse). We propose Simplex Diffusion Models (SDMs), a framework that lifts the diffusion process to the probability simplex to represent beliefs over categories. SDMs admit probability paths with closed-form reverse transitions and can be trained with a simple cross-entropy loss. Contrary to earlier proposals such as Dirichlet Flow Matching which requires integrating an ordinary differential equation, we introduce a DDIM-like sampler with a tunable level of stochasticity. Because SDMs operate on samples on the simplex, they can carry uncertainty across denoising steps, which mitigates information collapse. On OpenWebText, SDMs are competitive with strong Discrete Diffusion baselines, achieving 17.0 GenPPL at 5.46 unigram entropy in 64 sampling steps, close to real validation data. Even without Self-Conditioning (SC), SDMs outperform masked and uniform diffusion (with SC or predictor-corrector sampling) on code generation (TinyGSM, T “ 0.1; 49.0% vs. 45.8%). Distilled down to 8 steps, SDMs solve 32.1% of GSM8K problems, more than distilled Discrete Diffusion models with 128 steps (21.4%).

## 1 INTRODUCTION

Denoising diffusion models (Ho et al., 2020; Song et al., 2021b) are state-of-the-art generative models in continuous domains (Ho et al., 2022; Rombach et al., 2022; Ramesh et al., 2022; Watson et al., 2023). However, a vast portion of real-world data—including text, biological sequences, and graph structures—is discrete, and extending diffusion to those modalities remains challenging (Austin et al., 2021; Hoogeboom et al., 2021; Campbell et al., 2022).

Existing diffusion models for categorical data generally fall into three categories. First, categorical diffusion models such as masked or uniform diffusion models (Sahoo et al., 2024; Shi et al., 2024; Ou et al., 2025) define the corruption process directly on sequences of discrete tokens. While theoretically sound, they suffer from information collapse. Indeed, as intermediate states are discrete, the denoiser’s uncertainty is discarded at each update. Hence, these models rely on Self-Conditioning (Chen et al., 2022) or loopholing (Jo et al., 2026) to carry beliefs across sampling steps.

Second, continuous relaxations diffuse one-hot vectors (Lee et al., 2026; Roos et al., 2026; Potaptchik et al., 2026), learned embeddings (Dieleman et al., 2022; Gulrajani & Hashimoto, 2023; Deschenaux & Gulcehre, 2026; Chemseddine et al., 2026; Yang et al., 2026), or frozen embeddings (Hu et al., 2026; Shen et al., 2026). However, the identity of the clean category tends to be destroyed abruptly, within a short window during the forward process, and this worsens as the dimension of the encoding grows (Pynadath et al., 2026a; Shabalin et al., 2026).

Finally, a third line of work diffuses directly on the probability simplex, whose elements are distributions over categories (Richemond et al., 2022; Cheng et al., 2024; Davis et al., 2024; Stark et al., 2024; Cheng et al., 2025; Williams et al., 2026; Boget & Kalousis, 2026; Chandra et al., 2026).

![](images/c19a9200fbe63d992d66635ac6aa01c9d14bf72d1c6781df0169da4c81680e56.jpg)

![](images/bafb550d2a4a4571fba5fd432e32f11ce066b845f13d951c2a3a1397beaf31fe.jpg)  
SDM (Ours; Argmax + SC, = 1) SDM (Ours; Expectation, = 1) MDM + SC MDM UDM + PC UDM NFE vs TinyGSM Accuracy

Figure 1: SDMs outperform Discrete Diffusion on TinyGSM. Left: Accuracy as a function of the number of function evaluation (NFE) at low temperature $( T = 0 . 1 )$ . Right: Accuracy against NFEs at $T = 1$ . Without Self-Conditioning (SC), SDMs with the expected embedding outperform masked diffusion with SC (MDM + SC) and uniform diffusion with predictor-corrector sampling (UDM + PC). At 512 NFE, SDMs reach 49.0% at T “ 0.1, vs. 45.8% for $\mathrm { \mathbf { M D M ^ { \ddag } } + S C }$ , and 45.8% at $T = 1 ,$ vs. 36.8% for $\mathrm { U D M ^ { \dagger } + P C }$ . With SC, SDM (Argmax + SC) is the best diffusion variant at $T = 0 . 1$ from 32 NFE onwards, and solves 57% of problems with 8k NFEs. Autoregressive models with greedy decoding is best (62.6%). All SDMs use churn κ “ 1 and the adaptive sampling time grid (Section J.1), and we compare with the best baseline configurations (Section 5.2). <sup>:</sup>/<sup>;</sup>: trained with the uniform/adaptive time sampler. The accuracy drop for UDM at 16k is consistent with typical floating-point rounding errors, which can accumulate at very small step sizes (Karras et al., 2022).

However, they typically sample by numerically integrating an ODE or SDE: Dirichlet Flow Matching (DFM; Stark et al., 2024), the closest to our work, scales poorly to large vocabularies (Table 3). While the concurrent Simplax model (Sakurai et al., 2026) does not rely on numerical integration, their sampling procedure operates on categorical samples rather than the continuous representation itself.

Contributions. We propose Simplex Diffusion Models (SDMs), whose intermediate states are distributions over tokens.

1. Similar to DFM, SDMs use a Dirichlet forward process, but parameterized so that its mean and concentration are set separately. This path admits closed-form reverse transitions, inducing a DDIM-style sampler (Song et al., 2021a), with a churn parameter that controls the sampling stochasticity. SDMs are trained with a simple cross-entropy objective. Because SDMs operate on samples on the simplex, they can carry uncertainty across denoising steps, mitigating information collapse.

2. We show that SDMs unify discrete and continuous diffusions. A category sampled from the diffused simplex state has the same marginal distribution as if it were corrupted by Discrete Diffusion (Proposition 3.1). Viewing the inverse Dirichlet concentration as a temperature, SDMs reduce to Discrete Diffusion at high temperature, and follow a deterministic path with Gaussian fluctuations at low temperature (Proposition 4.1 and Section E.7).

3. Empirically, even without Self-Conditioning (SC), SDMs outperform masked and uniform diffusion (using either SC or predictor-corrector sampling) on TinyGSM $( T = 0 . 1 ; 4 9 . 0 \%$ vs. 45.8%; Figure 1). Distilled down to 8 steps, SDMs solve 32.1% of the GSM8K problems, more than distilled Discrete Diffusion models with 128 steps (21.4%; Table 5). On Sudoku, SDMs with SC match the best Discrete Diffusion model (99.1% vs. 99.3%). On OpenWebText, SDMs are competitive with UDM and MDMs with all those methods achieving a GenPPL/Entropy frontier that matches real data after logit shaping interventions. This result calls into question the validity of OpenWebText in assessing the unconditional text generation capabilities of small scale models. We also validate the flexibility of our approach by achieving competitive performance on quality and diversity metrics for molecule generation, see Section K.7.

## 2 BACKGROUND

Notation. Let $\mathcal { X } = \{ 1 , \ldots , N \}$ be the finite categorical vocabulary. Let $\Delta _ { N }$ denote the probability simplex on $\mathbb { R } ^ { N }$ , and $\\bar { \mathbf { 1 } } = ( 1 , \ldots , 1 ) ^ { \top }$ be the all-ones vector. For $i \in \mathcal { X }$ , let $e _ { i } \in \Delta _ { N }$ be the standard basis (one-hot) vector corresponding to state $i ,$ and let $\odot$ denote the Hadamard product. Let $t \mapsto \alpha _ { t }$ be non-increasing on r0, 1s with $\alpha _ { 0 } = 1 , \alpha _ { 1 } = 0$ , and $0 < \alpha _ { t } < 1$ , for $t \in ( 0 , 1 )$ . We denote by $\boldsymbol { \pi } ~ = ~ ( \pi _ { 1 } , \ldots , \pi _ { N } ) \in ~ \Delta _ { N }$ a reference prior distribution. We denote by Dir, Beta, Gamma, Ber and Cat the Dirichlet, Beta, Gamma, Bernoulli and Categorical distributions respectively. We recall basic facts about these distributions in Section A.

Discrete Diffusion Models. Discrete Diffusion models (DDMs; Sohl-Dickstein et al., 2015; Austin et al., 2021; Campbell et al., 2022; Sahoo et al., 2024; Shi et al., 2024; Ou et al., 2025) are generative models over discrete spaces. DDMs define a conditional corruption process

$$
p _ { t | 0 } ( x _ { t } | x _ { 0 } ) = \mathrm { C a t } \left( x _ { t } ; \alpha _ { t } e _ { x _ { 0 } } + ( 1 - \alpha _ { t } ) \pi \right) ,\tag{1}
$$

that induces marginal distributions $\left( p _ { t } \right) _ { t \in [ 0 , 1 ] } ,$ , given by $\begin{array} { r } { p _ { t } ( x _ { t } ) = \sum _ { x _ { 0 } } p _ { t | 0 } ( x _ { t } | x _ { 0 } ) p _ { 0 } ( x _ { 0 } ) } \end{array}$ , bridging $p _ { 0 } = p _ { \mathrm { d a t a } } \tan { p _ { 1 } = \pi }$ . The generative process is defined in terms of transitions $p _ { s \mid t }$ with $s \leqslant t$ . The transition $p _ { s \mid t }$ is compatible if $\begin{array} { r } { p _ { s } ( x _ { s } ) = \sum _ { x _ { t } } p _ { s | t } ( x _ { s } | x _ { t } ) p _ { t } ( x _ { t } ) } \end{array}$ . Thus, one can generate $x _ { 0 } \sim$ p<sub>data</sub> by applying $p _ { s \mid t }$ for n iterations, starting from $x _ { 1 } \sim p _ { 1 }$ . For a time grid $0 = t _ { 0 } < t _ { 1 } < . . . < t _ { n } = 1$

$$
\begin{array} { r } { p _ { 0 } ( x _ { 0 } ) = \sum _ { \left( x _ { t _ { 1 } } , \ldots , x _ { t _ { n - 1 } } , x _ { 1 } \right) } p _ { 1 } ( x _ { 1 } ) \prod _ { i = 1 } ^ { n } p _ { t _ { i - 1 } | t _ { i } } ( x _ { t _ { i - 1 } } | x _ { t _ { i } } ) . } \end{array}
$$

While using marginal transitions $\begin{array} { r } { p _ { s | t } ( x _ { s } | x _ { t } ) = \sum _ { x _ { 0 } } p _ { s | 0 , t } ( x _ { s } | x _ { 0 } , x _ { t } ) p _ { 0 | t } ( x _ { 0 } | x _ { t } ) } \end{array}$ is possible, recent work instead defines bridge (approximate) transitions ${ \hat { p } _ { s | t } ^ { \theta } ( x _ { s } | x _ { t } ) : = p _ { s | 0 , t } ( \cdot | \mathbf { x } _ { \theta } ( t , x _ { t } ) , x _ { t } ) }$ , where ${ \bf x } _ { \theta } ( t , x _ { t } ) : [ 0 , 1 ] \times \mathcal { X }  \Delta _ { N }$ is learned (Gourevitch et al., 2026). One obtains θ by maximizing an expected Evidence Lower Bound (ELBO).

Dirichlet Flow Matching. Dirichlet llow matching (DFM; Stark et al., 2024) extends Flow Matching (FM; Lipman et al., 2023) to the simplex $\Delta _ { N }$ to model categorical data. The data distribution $p _ { \mathrm { d a t a } } = ( p _ { \mathrm { d a t a , 1 } } , \cdot \cdot \cdot , p _ { \mathrm { d a t a , } N } )$ on X is first lifted to an atomic measure on the simplex which we also denote as $p _ { \mathrm { d a t a } }$ in a slight abuse of notation for $P \in \Delta _ { N }$

$$
\begin{array} { r } { p _ { \mathrm { d a t a } } \left( P \right) = \sum _ { i = 1 } ^ { N } p _ { \mathrm { d a t a } , i } \delta _ { e _ { i } } ( P ) . } \end{array}\tag{2}
$$

Stark et al. (2024) define a probability path $( p _ { t } ) _ { t \in [ 0 , 1 ] }$ between the distribution $p _ { 0 } = p _ { \mathrm { d a t a } }$ (at time $t = 0 )$ and the uniform Dirichlet prior $p _ { 1 } = { \mathrm { D i r } } ( \cdot ; { \bf \bar { 1 } } ) { \mathrm { ~ } } ( { \mathrm { a t ~ } } t = 1 )$ using

$$
p _ { t } ( P _ { t } ) = \sum _ { P _ { 0 } } p _ { t | 0 } ( P _ { t } \mid P _ { 0 } ) p _ { \mathrm { d a t a } } ( P _ { 0 } ) \mathrm { w h e r e } p _ { t | 0 } ( P _ { t } | P _ { 0 } ) = \mathrm { D i r } \big ( P _ { t } ; \mathbf { 1 } + h _ { t } P _ { 0 } \big ) ,
$$

with $t \mapsto h _ { t }$ a decreasing function with $h _ { 1 } ~ = ~ 0$ and $\mathrm { l i m } _ { t  0 } h _ { t } = \infty$ . A velocity field $u _ { t } ( P _ { t } )$ ensuring (approximately) that $\mathrm { d } P _ { t } = u _ { t } ( P _ { t } ) \mathrm { d } t$ satisfies $P _ { t } \sim p _ { t }$ for $P _ { 1 } \sim p _ { 1 }$ , hence $P _ { 0 } \sim p _ { \mathrm { d a t a } } .$ , is then learned through cross-entropy.

## 3 SIMPLEX DIFFUSION

## 3.1 PROBABILITY PATH AND BRIDGE

As in Dirichlet Flow Matching (Stark et al., 2024), we build a probability path on the simplex by considering $P _ { 0 } \sim p _ { \mathrm { d a t a } }$ , see (2), and Dirichlet distributions to define $p _ { t | 0 } ( P _ { t } | P _ { 0 } )$ .

Probability Path. Let $t \mapsto c _ { t }$ be a positive function and $t \mapsto \alpha _ { t }$ a non-increasing function such that $\alpha _ { 0 } = 1$ and $\alpha _ { 1 } = 0$ . We consider the corruption mechanism defined by

$$
p _ { t | 0 } ( P _ { t } | P _ { 0 } ) = \mathrm { D i r } ( P _ { t } ; \beta _ { t } ( P _ { 0 } , \pi ) ) , \qquad \beta _ { t } ( P _ { 0 } , \pi ) = c _ { t } ( \alpha _ { t } P _ { 0 } + ( 1 - \alpha _ { t } ) \pi ) ,\tag{3}
$$

where we recall that $\pi \in \Delta _ { N }$ is a reference prior distribution on X chosen by the user<sup>1</sup>. In particular $p _ { 1 | 0 } ( P _ { 1 } | P _ { 0 } ) = \operatorname { D i r } ( P _ { 1 } ; c _ { 1 } \pi )$ is independent of $P _ { 0 }$ . We have $\mathbb { E } [ P _ { t } | P _ { 0 } ] = ( { \dot { 1 } } - \alpha _ { t } ) \pi + \alpha _ { t } { \dot { P } } _ { 0 } : = { \bar { P } } _ { t }$ and $\mathrm { C o v } [ P _ { t } | P _ { 0 } ] = ( \mathrm { d i a g } ( \bar { P } _ { t } ) - \bar { P } _ { t } \bar { P } _ { t } ^ { \top } ) / ( c _ { t } + 1 )$ . We refer to $c _ { t }$ as the concentration parameter.

This forward model on the simplex induces a forward model on the discrete state-space commonly used in Discrete Diffusions/Flow Matching (Campbell et al., 2024; Sahoo et al., 2024; Shi et al., 2024).

Proposition 3.1 (Induced forward): $L e t t \in [ 0 , 1 ]$ and $P _ { t }$ satisfying (3). In addition, let $x _ { t }$ be such that $p _ { t | 0 } ( x _ { t } | P _ { t } , P _ { 0 } ) = P _ { t , x _ { t } }$ . Then, recalling that $\begin{array} { r } { P _ { 0 } = e _ { x _ { 0 } } , } \end{array}$ , we have that

$$
p _ { t | 0 } ( x _ { t } | P _ { 0 } ) = \alpha _ { t } \delta _ { x _ { 0 } } ( x _ { t } ) + ( 1 - \alpha _ { t } ) \pi _ { x _ { t } } .
$$

We now propose an alternative representation of (3) enabling us to sample easily from $p _ { t | 0 } ( P _ { t } \mid P _ { 0 } )$

Proposition 3.2 (Interpolation path): Let $t \in [ 0 , 1 ]$ . Let $W _ { t } \sim$ Beta $( c _ { t } \alpha _ { t } , c _ { t } ( 1 - \alpha _ { t } ) )$ and let $\bar { V } _ { t } \sim \operatorname * { D i r } ( c _ { t } ( 1 - \alpha _ { t } \bar { ) } \pi )$ be independent $o f W _ { t }$ . Define

$$
P _ { t } = W _ { t } P _ { 0 } + ( 1 - W _ { t } ) V _ { t } .\tag{4}
$$

Then the conditional density of $P _ { t }$ given $P _ { 0 }$ is given by $p _ { t | 0 } ( P _ { t } \mid P _ { 0 } ) = \operatorname { D i r } \left( P _ { t } ; \beta _ { t } ( P _ { 0 } , \pi ) \right)$

(4) shows that, for $t \in ( 0 , 1 )$ , $P _ { t }$ is a random convex combination (with interpolation weight $W _ { t }$ P $( 0 , 1 ) )$ of a data sample $P _ { 0 }$ and a noise sample $V _ { t }$

The concentration parameter $c _ { t }$ acts inversely to temperature: larger values concentrate probability mass around the mean, whereas smaller values push it toward the simplex vertices. Specializing momentarily to $c _ { t } = \varepsilon / ( 1 - \alpha _ { t } )$ for $t > 0$ , we interpret ε as an inverse temperature. In Section $^ { 4 , }$ we formally prove that as $\varepsilon \to 0$ (high temperature), the conditional distribution of $P _ { t } \mid P _ { 0 }$ concentrates entirely on the vertices. Hence Discrete Diffusion models emerge as a limiting case of Simplex Diffusion Models. Conversely, as $\varepsilon  \infty$ , the distribution concentrates on the mean, $\alpha _ { t } \dot { P } _ { 0 } + \bar  ( 1 -$ $\alpha _ { t } ) \pi$ . Figure 6 illustrates these regimes across varying $\varepsilon > 0$ and $t \in [ 0 , 1 ]$

Backward Bridges. We now return to a general concentration schedule $\left( c _ { t } \right) _ { t \in \left[ 0 , 1 \right] }$ . In order to obtain a generative model, in the spirit of DDIM (Song et al., 2021a), we need to identify a conditional distribution $p _ { s | 0 , t }$ satisfying the following backward compatibility condition for any $s , t \in [ 0 , 1 ]$ with $s < t$ and $P _ { s } \in \Delta _ { N }$

$$
p _ { s | 0 } ( P _ { s } | P _ { 0 } ) = \int _ { \Delta _ { N } } p _ { s | 0 , t } ( P _ { s } | P _ { 0 } , P _ { t } ) p _ { t | 0 } ( P _ { t } | P _ { 0 } ) \mathrm { d } P _ { t } .\tag{5}
$$

Leveraging classical properties of Dirichlet distributions, we propose such a backward bridge model. For $t \in ( 0 , 1 ]$ , write $\beta _ { t } ( P _ { 0 } , \pi ) = a _ { t } P _ { 0 } + b _ { t } \pi$ so that $a _ { t } = c _ { t } \alpha _ { t }$ and $b _ { t } = c _ { t } ( 1 - \alpha _ { t } )$ . The quantity $b _ { t }$ can be thought of as the concentration associated to π at time t. Let $r _ { s , t } = \operatorname* { m i n } \left\{ 1 , b _ { s } / b _ { t } \right\}$

Proposition 3.3 (Simplex Transition): Let $\kappa \in [ 0 , 1 )$ and $s , t \in ( 0 , 1 ]$ with $s < t .$ Denote $\rho _ { s , t } ^ { \kappa } = ( 1 - \kappa ) r _ { s , t }$ . For any $P _ { 0 } , P _ { s } , P _ { t } \in \Delta _ { N }$ , consider $P _ { s , t } ^ { \kappa } = ( P _ { s , t , 1 } ^ { \kappa } , \bar { \cdot } \cdot \cdot , P _ { s , t , N } ^ { \kappa } )$ , where

$$
P _ { s , t , i } ^ { \kappa } = \frac { B _ { i } P _ { t , i } } { \sum _ { j = 1 } ^ { N } B _ { j } P _ { t , j } }\tag{6}
$$

with independent $B _ { i } \sim$ Beta $\begin{array} { r } { \big ( \rho _ { s , t } ^ { \kappa } \beta _ { t } ( P _ { 0 } , \pi ) _ { i } , ( 1 - \rho _ { s , t } ^ { \kappa } ) \beta _ { t } ( P _ { 0 } , \pi ) _ { i } \big ) , f o r \ i = 1 , \cdot \cdot \cdot , N } \end{array}$ . Moreover, let $\bar { W } _ { s , t } ^ { \kappa }$ and $V _ { s , t } ^ { \kappa }$ be given by

$$
\begin{array} { r l r } { W _ { s , t } ^ { \kappa } \sim \mathrm { B e t a } \left( \rho _ { s , t } ^ { \kappa } c _ { t } , \ c _ { s } - \rho _ { s , t } ^ { \kappa } c _ { t } \right) , } & { } & { V _ { s , t } ^ { \kappa } \sim \mathrm { D i r } \left( \beta _ { s } ( P _ { 0 } , \pi ) - \rho _ { s , t } ^ { \kappa } \beta _ { t } ( P _ { 0 } , \pi ) \right) , } \end{array}\tag{7}
$$

the random variables $( B _ { i } , W _ { s , t } ^ { \kappa } , V _ { s , t } ^ { \kappa } )$ being all independent. Finally let

$$
P _ { s } = W _ { s , t } ^ { \kappa } P _ { s , t } ^ { \kappa } + ( 1 - W _ { s , t } ^ { \kappa } ) V _ { s , t } ^ { \kappa } .\tag{8}
$$

Then the induced transition kernel denoted $p _ { s \vert 0 , t }$ satisfies the compatibility condition (5). For the limiting case $\kappa = 1$ , we define $p _ { s | 0 , t } ( P _ { s } | P _ { 0 } , P _ { t } ) = p _ { s | 0 } ( P _ { s } | P _ { 0 } )$

![](images/6283426f61419bebe8b03da67aa4cde5a73b85a66ecc0c76ddb737557ff8a958.jpg)  
Figure 2: Left to right: different stages of a backward step during inference. We assume that at the current step, $P _ { 0 } = e _ { x _ { 0 } }$ where $x _ { 0 } \sim \hat { P } _ { \theta } ( t , P _ { t } )$ is such that $P _ { 0 } = e _ { 1 }$ . The top row corresponds to a low churn $\kappa = 0 . 1 5$ and the bottom row corresponds to a high churn of $\kappa = 0 . 6 5$ . We assume an interpolation schedule $\alpha _ { t } = 1 - t$ and concentration schedule $c _ { t } = \varepsilon / ( 1 - \alpha _ { t } )$ with $\varepsilon = 4 . 0$ Additionally, $s = 0 . 4 0$ and $t = 0 . 7 0$ . The dots with colors yellow, orange, red correspond to three different samples from the current step. In the first step, using the churn $\kappa \in [ 0 , 1 ]$ , we sample $P _ { s , t } ^ { \kappa }$ This corresponds to a multiplicative noising step around $P _ { t }$ , with a higher churn corresponding to a higher noise. In the second step, we sample the innovation variable $V _ { s , t } ^ { \breve { \kappa } }$ , a Dirichlet random variable concentrated around the prediction $P _ { 0 }$ . In the third step we sample from a Beta random variable $W _ { s , t } ^ { \kappa }$ corresponding to the mixing weight in the interpolation between the thinned variable $P _ { s , t } ^ { \kappa }$ and the innovation $V _ { s , t } ^ { \kappa }$ . For high churns, the mixing weight $W _ { s , t } ^ { \kappa }$ is close to $0 ,$ , meaning that we favor the innovation random variable $V _ { s , t } ^ { \kappa }$ . Finally in the fourth step, we visualize the interpolation between the innovation $V _ { s , t } ^ { \kappa }$ and the thinned variable $P _ { s , t } ^ { \kappa }$ using the sampled Beta mixing weight $W _ { s , t } ^ { \kappa }$

The simplex transition outputs a new state $P _ { s }$ in (8) which is a random convex combination of an innovation term $V _ { s , t } ^ { \kappa }$ and a thinned version $P _ { s , t } ^ { \kappa }$ of the current state $P _ { t }$ , with interpolation weight $W _ { s , t } ^ { \kappa }$ . Indeed for $\bar { P } _ { t } \sim \mathrm { D i r } ( \beta _ { t } ( P _ { 0 } , \pi ) )$ , one can show that $P _ { s , t } ^ { \kappa } \sim \mathrm { D i r } ( \rho _ { s , t } ^ { \kappa } \beta _ { t } ( P _ { 0 } , \pi ) )$ . The churn parameter $\kappa \in [ 0 , 1 \bar { \bf \Phi } ]$ controls how much additional randomness is introduced in the reverse transition at inference time. We present those different steps in Figure 2.

When $\kappa = 0 .$ , we retain the largest fraction of the current state that is compatible with the prescribed concentration schedule. In this case $\rho _ { s , t } ^ { 0 } ~ = ~ r _ { s , t } , ~ W _ { s , t } ^ { 0 } ~ \sim$ Beta $( r _ { s , t } c _ { t } , c _ { s } - r _ { s , t } c _ { t } )$ and $V _ { s , t } ^ { 0 } \ \sim$ Dir $( \beta _ { s } ( P _ { 0 } , \pi ) - r _ { s , t } \beta _ { t } ( P _ { 0 } , \pi ) )$ . The corresponding update is

$$
P _ { s } = W _ { s , t } ^ { 0 } P _ { s , t } ^ { 0 } + ( 1 - W _ { s , t } ^ { 0 } ) V _ { s , t } ^ { 0 } .\tag{9}
$$

For the concentration $c _ { t } = \varepsilon / ( 1 - \alpha _ { t } )$ , we have $b _ { t } = c _ { t } ( 1 - \alpha _ { t } ) = \varepsilon$ , hence $r _ { s , t } = 1$ . In that case $P _ { s , t } ^ { 0 } = P _ { t }$ and $V _ { s , t } ^ { 0 } = e _ { x _ { 0 } }$ , so (9) reduces to $P _ { s } = W _ { s , t } ^ { 0 } P _ { t } + ( 1 - W _ { s , t } ^ { 0 } ) e _ { x _ { 0 } } . \mathrm { I f } \varepsilon \to + \infty , ( 9 )$ becomes

$$
P _ { s } = \frac { 1 - \alpha _ { s } } { 1 - \alpha _ { t } } P _ { t } + \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } e _ { x _ { 0 } } ,
$$

which is the exact deterministic DDIM rule. We refer to Proposition 4.1 for a theoretical analysis of those temperature limits. Equipped with a compatible backward transition mechanism, we can now define our training and inference algorithms.

## 3.2 TRAINING AND INFERENCE

Define $0 = t _ { 0 } < \cdots < t _ { M } = 1$ . At inference time, we will follow a DDIM approach (Song et al., 2021a): ideally we would generate data by starting from $P _ { t _ { M } } \sim p _ { 1 } ( \cdot )$ and $\begin{array} { r } { P _ { t _ { k } } \sim p _ { t _ { k } | t _ { k + 1 } } ( \cdot | ^ { - } P _ { t _ { k + 1 } } ) } \end{array}$

for $k = M - 1 , . . . , 0$ , where for $0 \leqslant s < t \leqslant 1$

$$
p _ { s | t } ( P _ { s } | P _ { t } ) = \int p _ { s | 0 , t } ( P _ { s } | P _ { 0 } , P _ { t } ) p _ { 0 | t } ( P _ { 0 } | P _ { t } ) \mathrm { d } P _ { 0 } ,\tag{10}
$$

with $p _ { 0 | t } ( P _ { 0 } | P _ { t } )$ the posterior distribution of $P _ { 0 }$ given $P _ { t }$ . Here $\begin{array} { r c l } { { p _ { 1 } ( P _ { 1 } ) } } & { { = } } & { { \int p _ { 1 | 0 } ( P _ { 1 } \quad } } \end{array}$ $P _ { 0 } ) p ( P _ { 0 } ) \mathrm { d } P _ { 0 } = \mathrm { D i r } ( P _ { 1 } ; c _ { 1 } \pi )$ . If we had access to the true backward transitions (10), then this procedure would return samples from $P _ { 0 }$ , hence from the data distribution. Since $P _ { 0 }$ is supported on the vertices of the simplex, then $p _ { 0 | t } ( P _ { 0 } \mid P _ { t } )$ is a categorical distribution on those vertices which we approximate by a denoiser $\hat { P } _ { \theta } ( t , P _ { t } )$ such that ${ \hat { P } } _ { \theta , i } ( t , P _ { t } ) \approx \mathbb { P } ( P _ { 0 } = e _ { i } \mid P _ { t } ) = \mathbb { P } ( x _ { 0 } = i \mid P _ { t } )$ . For each time step $k = M { - } 1 , \ldots , 0 .$ , we then sample $P _ { t _ { k } } \sim p _ { t _ { k } | 0 , t _ { k + 1 } } ( \cdot \mid P _ { 0 } , P _ { t _ { k + 1 } } )$ where $P _ { 0 } = e _ { \tilde { x } _ { \mathrm { c } } }$ for $\tilde { x } _ { 0 } \sim \hat { P } _ { \theta } ( t _ { k + 1 } , P _ { t _ { k + 1 } } ) ;$ i.e., we approximate $p _ { 0 | t } ( P _ { 0 } | P _ { t } )$ by $\begin{array} { r } { p _ { 0 | t } ^ { \theta } ( P _ { 0 } | P _ { t } ) = \sum _ { i = 1 } ^ { N } \hat { P } _ { \theta , i } ( t , P _ { t } ) \delta _ { e _ { i } } ( P _ { 0 } ) } \end{array}$ To learn the denoiser $\hat { P } _ { \theta }$ from $P _ { t }$ , we minimize the cross entropy loss

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = - \mathbb { E } _ { t \sim w ( t ) , x _ { 0 } \sim p _ { \mathrm { d a t a } } , P _ { 0 } = e _ { x _ { 0 } } , P _ { t } \sim \mathrm { D i r } \left( \beta _ { t } ( P _ { 0 } , \pi ) \right) } [ \log \left( \hat { P } _ { \theta } ( t , P _ { t } ) _ { x _ { 0 } } \right) ] , } \end{array}\tag{11}
$$

as $\begin{array} { r } { \sum _ { i = 1 } ^ { N } P _ { 0 , i } \log \left( \hat { P } _ { \theta } ( t , P _ { t } ) _ { i } \right) = \log \left( \hat { P } _ { \theta } ( t , P _ { t } ) _ { x _ { 0 } } \right) } \end{array}$ . In Section F, we show that this loss corresponds to a negative evidence lower bound (ELBO) if wptq is selected as the uniform distribution on the set $\left\{ t _ { 1 } , . . . , t _ { M } \right\}$ . Given a token embedding matrix $E \in \mathbb { R } ^ { N \times d }$ , the denoiser $\hat { P } _ { \theta } ( t , P _ { t } )$ embeds the simplex state $P _ { t }$ either via its expected embedding $P _ { t } ^ { \top } E \mathrm { o r } ,$ , to avoid this dense matrix product on large vocabularies, via the most likely token embedding $E _ { \mathrm { a r g m a x } \ : P _ { t } ; }$ ; see Section J.2 for details. We summarize the training of Simplex Diffusion Models in Algorithm 1 and inference in Algorithm 2. These algorithms are described for a single token for the sake of simplicity. The extension to multiple tokens is described in Section C.1.

## 4 CONCENTRATION PROPERTIES

## 4.1 TEMPERATURE LIMITS

In Section 3, we briefly discussed how the distribution of $P _ { t }$ behaves in both the high and low temperature settings $c _ { t } = \varepsilon / ( 1 - \alpha _ { t } )$ . We show it here formally.

Proposition 4.1 (Temperature limits): Let $c _ { t } = \varepsilon / ( 1 - \alpha _ { t } )$ and assume that $P _ { 0 } = e _ { x _ { 0 } }$

(i) High-temperature limit. Fix $t \in ( 0 , 1 )$ . As $\varepsilon \to 0 ,$

$$
P _ { t } \stackrel { d } { \longrightarrow } B _ { t } P _ { 0 } + ( 1 - B _ { t } ) V ,
$$

where $B _ { t } \ \sim \ \mathrm { B e r } ( \alpha _ { t } )$ and $\begin{array} { r } { V \sim \sum _ { i = 1 } ^ { N } \pi _ { i } \delta _ { e _ { i } } } \end{array}$ , independently. Thus, $P _ { t }$ is supported on the vertices of $\Delta _ { N }$ . Moreover, $\mathit { f u x } \ : 0 < s < \bar { t } < 1$ and consider the $\kappa = 0$ backward bridge (9). For every fixed $P _ { t } \in \Delta _ { N } , a s \varepsilon \to 0 ,$

$$
P _ { s } \stackrel { d } { \longrightarrow } W _ { s , t } ^ { 0 } P _ { t } + \left( 1 - W _ { s , t } ^ { 0 } \right) e _ { x _ { 0 } } , \qquad W _ { s , t } ^ { 0 } \sim \mathrm { B e r } \left( \frac { 1 - \alpha _ { s } } { 1 - \alpha _ { t } } \right) .
$$

Thus, the limiting transition remains supported on simplex vertices.

(ii) Low-temperature limit. $F i x t \in ( 0 , 1 ) . A s \varepsilon \to \infty ,$

$$
P _ { t } \stackrel { \mathbb { P } } { \longrightarrow } \bar { P } _ { t } : = \alpha _ { t } P _ { 0 } + ( 1 - \alpha _ { t } ) \pi .\tag{12}
$$

Proposition 4.1 shows that the inverse temperature interpolates between two very different regimes. At high temperature, the simplex-valued state collapses onto the vertices, and with $\kappa = 0$ it reduces the reverse bridge to a transition between the current and clean vertices. Thus the discrete-diffusion interpretation is recovered for both the probability path and the backward bridge. At low temperature, by contrast, the randomness disappears and the forward state interpolates deterministically between $P _ { 0 }$ and π. However, the deterministic limit in (12) only describes the first-order behavior as $\varepsilon \to \infty$ . In Section E.7 we identify a Gaussian fluctuation limit result, drawing connections between the limit $\varepsilon  + \infty$ and Gaussian diffusion models.

## 4.2 CONCENTRATION SCHEDULE VIA NORMALIZED VARIANCE

While Proposition 4.1 establishes that SDMs interpolate between Discrete Diffusion (high temperature) and the mean interpolation (low temperature), it remains unclear how to select $t \mapsto c _ { t }$ in practice. Here we focus on the setting where $\pi = { \bf 1 } / N , { \mathrm { i . e . } }$ , the uniform prior. Rather than tuning an unbounded $c _ { t } \in ( 0 , \infty )$ , we define $c _ { t }$ in terms of the fraction $t \mapsto \nu _ { t } \in ( 0 , 1 )$ of the maximal total variance (TVar), defined by

$$
\operatorname { T V a r } ( t ) : = \operatorname { T r } \left[ \operatorname { C o v } \left( P _ { t } \mid P _ { 0 } \right) \right] = { \frac { 1 } { c _ { t } + 1 } } \left[ 1 - \| \alpha _ { t } P _ { 0 } + ( 1 - \alpha _ { t } ) \pi \| _ { 2 } ^ { 2 } \right] = { \frac { N - 1 } { N } } { \frac { 1 - \alpha _ { t } ^ { 2 } } { c _ { t } + 1 } } .
$$

Because $c _ { t } \in ( 0 , \infty )$ , the total variance is bounded above by $\frac { N - 1 } { N } ( 1 - \alpha _ { t } ^ { 2 } )$ . Setting $\textstyle \nu _ { t } : = { \frac { 1 } { c _ { t } + 1 } }$ (equivalently $\begin{array} { r } { c _ { t } = \frac { 1 } { \nu _ { t } } - 1 ) } \end{array}$ , the total variance is exactly the fraction $\nu _ { t }$ of this bound. In practice, we experiment with two variants (one constant, one piecewise linear):

$$
\begin{array} { r } { \nu _ { t } ^ { \mathrm { c s t . } } = \nu _ { 0 } , \qquad \nu _ { t } ^ { \mathrm { c s t . - l i n } } = \left\{ \begin{array} { l l } { \nu _ { 0 } , } & { \mathrm { i f ~ } t < \ell , } \\ { \nu _ { 0 } + \frac { t - \ell } { 1 - \ell } ( \nu _ { 1 } - \nu _ { 0 } ) , } & { \mathrm { i f ~ } t \geqslant \ell . } \end{array} \right. } \end{array}
$$

This parameterization is more interpretable than the raw value of $c _ { t } . \ \mathrm { A t } \ t \ = \ 1$ , setting $c _ { 1 } \ = \ N$ induces a uniform distribution on the simplex, and a larger $c _ { 1 }$ concentrates around π. However, for $t < 1$ , the mean $\alpha _ { t } P _ { 0 } + ( 1 - \alpha _ { t } ) \pi$ moves towards the vertex $P _ { 0 }$ , and a large $c _ { t }$ collapses $P _ { t }$ onto the mean, which reveals the clean token almost deterministically. In contrast, TVar operates on a bounded interval with clear extrema: it approaches its upper bound when $P _ { t }$ collapses onto the vertices, and vanishes when $P _ { t }$ collapses onto its mean $( c _ { t } \to \infty )$ .

## 5 EXPERIMENTS

Table 1: Accuracy (%) on Sudoku in 180 steps, varying the sampling schedule. <sup>:</sup>trained with uniform time sampler, <sup>;</sup>trained with adaptive time sampler. For each column, we underline the best and bold the highest accuracy for continuous methods. We show SDM with normalized variance $\nu _ { t } ^ { \mathrm { { c s t . - l i n . } } }$ , with $\nu _ { 0 } = 0 . 4 ,$ , ν<sub>1</sub> “ 0.75, ℓ “ 0.8. Values are mean ˘ std over 5 seeds.

We compare SDMs with autoregressive models (AR), masked and uniform diffusion models (MDMs, UDMs), and Flow Language Models (FLM) and Hyperspherical Flows (S-FLM). We evaluate on Sudoku (Alp, 2024; Ben-Hamu et al., 2025; Kim et al., 2026b), code generation on TinyGSM (Liu et al., 2023a; Kim et al., 2026a), language modeling on OpenWebText (OWT) (Gokaslan & Cohen, 2019), language understanding; Section J.7, and molecule generation following GenMol (Lee et al., 2025; Section J.8). Prior work mostly compares models by Generative Perplexity (Gen. PPL) on OWT, which does not always correlate with downstream performance such as functional correctness in code (Deschenaux & Gulcehre, 2024; Feng et al., 2025; Franca & Tong, 2026; Velickoviˇ c et al.´ , 2026). Thus, we primarily compare models on TinyGSM.

<table><tr><td>Model</td><td>Linear</td><td>Cosine</td></tr><tr><td>Discrete</td><td></td><td></td></tr><tr><td>AR (Greedy)</td><td> $3 . 1 _ { + 0 . 5 }$ </td><td></td></tr><tr><td>MDM</td><td> $6 9 . 4 \substack { + 4 . 8 }$ </td><td> $6 9 . 7 _ { + 4 . 0 }$ </td></tr><tr><td> $\mathbf { M D M ^ { \ddag } \left( + S C \right) }$ </td><td> $9 4 . 7 _ { + 0 . 8 }$ </td><td> $9 9 . 3 _ { + 0 . 2 }$ </td></tr><tr><td> $\mathrm { U D M ^ { \ddag } }$ </td><td> $8 2 . 1 _ { + 1 . 7 }$ </td><td> $8 2 . 4 _ { + 1 . 7 }$ </td></tr><tr><td> $\mathrm { U D M } ^ { \dag } \left( + \mathrm { S C } \right)$ </td><td> $9 7 . 8 _ { + 1 . 7 }$ </td><td> $9 8 . 1 _ { + 1 . 2 }$ </td></tr><tr><td>Continuous</td><td></td><td></td></tr><tr><td> $\mathrm { F L M ^ { \ddag } }$ </td><td> $7 5 . 3 _ { + 3 . 4 }$ </td><td> $7 4 . 9 _ { + 3 . 5 }$ </td></tr><tr><td> $\mathbb { S } – \mathrm { F L M } ^ { \ddagger }$ </td><td> $8 7 . 3 _ { + 1 . 5 }$ </td><td> $8 7 . 3 _ { + 1 . 7 }$ </td></tr><tr><td> $\mathrm { S D M } ^ { \dag } \left( \kappa = 1 . 0 \right)$ </td><td> $8 8 . 4 _ { + 2 . 2 }$ </td><td> $8 8 . 9 _ { + 2 . 4 }$ </td></tr><tr><td>SDM†  $( + \mathrm { { S C } ; \kappa = 1 . 0 ) }$ </td><td> $\underline { { 9 9 . 1 } } _ { + 0 . 2 }$ </td><td> $\mathbf { 9 8 . 7 _ { + 0 . 3 } }$ </td></tr></table>

## 5.1 REASONING ON SUDOKU

Experimental Setup. We train on 200k grids with 30/81 clues revealed and evaluate on 5k unseen examples. We train a modified Diffusion Transformer (DiT) (Peebles & Xie, 2023) as in Lou et al. (2024) with embedding dimension 512, 8 layers and 8 attention heads (28.6M parameters) for 50k steps. We use 180 sampling steps for the diffusion variants. See Section J.4 for further details.

Results. SDMs are the strongest continuous method on Sudoku and match the Discrete Diffusion models (Table 1). Without Self-Conditioning (SC), SDMs outperform MDMs and UDMs with ancestral sampling and FLMs, and match S-FLMs. With SC, SDMs reach 99.1%, outperforming the continuous baselines and within 0.2 points of the best result overall. SDMs also outperform

Dirichlet Flow Matching (DFM) by a wide margin (76.7%; Table 2). Setting $\kappa = 1$ works best. Without SC, increasing κ from 0 to 1 improves the accuracy from 73.9% to 88.4% (Tables 21 to 23). With SC, SDMs perform similarly across churns. Training with the adaptive time sampler (Section J.1) generally improves MDMs, UDMs, and SDMs at low churn.

## 5.2 CODE GENERATION ON TINYGSM

Experimental Setup. We train on TinyGSM, a dataset of 11.8M synthetic math word problems with executable Python solutions, and evaluate on the GSM8K test set by executing one generated solution per problem. We tokenize with the SmolLM tokenizer, and use a context length of 512. We train a 12-layer DiT with hidden dimension 768 and 12 attention heads (167.9M parameters) for 250k steps (« 77% for SC variants, to match the training FLOPs; Section I.3) with a batch size of 512. See Section J.5 for further details. With ODE sampling, FLM, S-FLM and DFM perform worse than the discrete baselines on TinyGSM, so we defer them to Section K.

Results. We compare SDMs against MDMs and UDMs with the ancestral sampler, with a Predictor-Corrector (PC) sampler (Section I.2) and with Self-Conditioning (SC; Section I.3). Since the expected embedding is costly with a large vocabulary, we also train SDMs that embed only the token argmax $P _ { t } .$ optionally combined with SC (Argmax + SC; Section J.2). With the expected embedding and without SC, SDMs outperform all diffusion baselines at 512 steps (Figure 1), reaching 45.8% at $T = 1$ (vs. 36.8% for UDM + PC) and 49.0% at $T = 0 . 1$ (vs. 45.8% for MDM + SC). With Argmax + SC, SDMs perform best at $T = 0 . 1$ from 32 function evaluations (NFE) onward and reach 57.0% at 8k NFE (50.2% at 512 NFE). Chemseddine et al. (2026) train Spherical Flow (SF) with the same setup and find that PC sampling and SC are essential. With 512 NFE $( T = 1 )$ , the accuracy increases from 6.4% with ODE sampling to 41.7% with PC and SC. In this setting, SDMs without SC outperform SFs without SC (45.8% vs. 32.4%) and perform similarly to SFs with SC (Table 4).

Time Schedule, Churn and Blockwise Generation. We train with an adaptive time sampler that upweights noise levels where the model learns most, using a density fitted to the training loss (Section J.1), and derive the sampling time grid from the same density (Section J.1, Dieleman et al., 2022; Raya et al., 2026). With the expected embedding, this adaptive schedule improves SDMs $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 , \kappa = 1 )$ over the linear grid by 9–11 points at 512 steps and 18–19 points at 64 steps; it does not help the Argmax inputs (e.g. Argmax + SC at $T = 1$ and 512 steps: 40.3% adaptive vs. 43.7% linear for $\nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ (Tables 29 to 32). The adaptive time sampler can improve MDM + SC but not $\mathrm { U D M } + \mathrm { P C } ,$ so we report MDM + SC trained with the adaptive time sampler and UDM + PC trained with the uniform time sampler (Section K.2). As on Sudoku, SDMs perform best with κ “ 1 (45.8% vs. 12.6% at κ “ 0, T “ 1, 512 steps). SFs likewise needs stochastic sampling. Finally, full-sequence SDMs outperform a block-autoregressive variant (Han et al., 2023; Arriola et al., 2025) with block size 32 at matched NFE, despite its stronger left-to-right bias (Figure 12).

Distillation and Auto-Guidance. We distill the SDM with the expected embedding (no SC; $\nu _ { t } ^ { \mathrm { c s t . - l i n . } }$ with $\nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ into a few-step generator (Section G). With 8 NFEs, the distilled SDM solves 32.1% of the problems at T “ 1, more than IDLM (Li et al., 2026) with 128 NFEs (21.4%; Table 5) and the undistilled SDM with 8 NFEs (« 6%; Figure 13). The accuracy improves up to « 40% at 128 NFEs and then plateaus. Distillation reduces the diversity, which we measure using the Abstract Syntax Trees (AST) of the generated programs (Section J.5). The AST diversity score decreases from 35.3 for the undistilled SDM with 512 NFEs (40.0 with 8 NFEs) to 17.5 for the distilled SDM with 8 NFEs at $T = 1$ (higher is more diverse; Table 33). See Section K.3 for more details. Finally, guiding SDMs away from a weaker checkpoint (Auto-Guidance; Karras et al., 2024) improves accuracy in all settings we tested, mostly in the low churn and high temperature regime (Section K.4).

## 5.3 LANGUAGE MODELING ON OPENWEBTEXT (OWT)

Experimental Setup. We train on OWT (Gokaslan & Cohen, 2019) for unconditional language modeling, and evaluate the generation quality via Generative Perplexity (GenPPL) under a pretrained GPT-2-large model (Radford et al., 2019) alongside unigram entropy $( H _ { 1 } )$ . We tokenize with the GPT-2 tokenizer, and use a context length of 1024. We train a 12-layer DiT with hidden dimension 768 and 12 attention heads (167.9M parameters) for 1M steps with a batch size of 512.

![](images/006a0b36284dccef0fd12750e851881c67e9c22578573ee9427ab76f7451c316.jpg)  
Unigram Entropy H (nats) vs. Gen PPL

![](images/710532fcc10cbfed3bcd305dc91111503b25663a6dca076b4b14fad714cbc695.jpg)  
Figure 3: OWT Pareto Frontiers of Entropy $H _ { 1 }$ vs. GPT-2 GenPPL. Dotted crosshairs and (‹) denote measured OpenWebText validation data $( H _ { 1 } = 5 . 4 6$ nats, Gen $\mathrm { P P L } = 1 5 . 2 )$ . Left: Headto-head Pareto frontiers comparing 64-step SDM, MDM, and UDM against 1024-step AR after sampler interventions. Right: Progressive ablation on 64-step SDM showing gains from Stage A (nucleus top-p) Ñ Stage B (power-law logit temperature annealing) Ñ Stage C (sequence frequency penalty) Ñ Stage D (local frequency penalty).

Results. With a few sampling interventions that improve the diffusion variants (MDMs, UDMs and SDMs), SDMs reach the same Pareto frontier as MDMs and UDMs. Similar interventions also improve AR models, whose frontier reaches the real OWT validation data. These results highlight that heuristic optimization can surpass the empirical validation point of OWT through simple logit shaping, thereby illustrating the limitation of generative frontiers (Pynadath et al., 2026b). We briefly describe the four logit shaping interventions we introduce. First, we consider top-p sampling as in Holtzman et al. (2019). Second we introduce a power-law logit temperature annealing during sampling similarly to DiffusionGemma Team et al. (2026); Chang et al. (2022). The third intervention which is by far the most impactful is a sequence frequency penalty, see (42) and (43) for more details. Note that this intervention benefits all experimental setups including MDM, UDM, SDM and AR. Finally, SDM is the only model to benefit from a localfrequency penalty which bridges the gap between SDMs and state-of-the-art MDM and UDM for text generation, achieving a GenPPL/Entropy trade-off that is close to real validation data (GenPPL/Entropy: 17.0/5.46). More details on the interventions and additional results can be found in Section K.5. Our main results including the Pareto frontiers for all models we consider are presented in Figure 3. Non-cherry picked samples are shown in Section L. Finally, in Section K.6, we present additional non-generative evaluation result on language understanding by evaluating our models on ARC-easy (Clark et al., 2018), PIQA (Bisk et al., 2020) and HellaSwag (Zellers et al., 2019).

## 6 CONCLUSION

This paper introduces Simplex Diffusion Models (SDMs) for discrete data. Unlike standard Discrete Diffusion models, SDMs can maintain a continuous distributional belief state throughout the generative process, circumventing information collapse even without Self-Conditioning, with which SDMs remain complementary. Our proposed generative model introduces a flexible DDIM-type sampler that does not require discretizing an ODE or SDE.

When compared to strong baselines, SDMs are competitive on unconditional text generation and molecular synthesis, and state-of-the-art on coding. Nevertheless, our experiments remain limited to small models, and whether these conclusions remain valid at scale is an open question. Further experimentation is needed to assess the viability of the approach on large-scale language generation tasks. More broadly, SDMs suggest that there is a false dichotomy between discrete and continuous diffusion language models. Discrete Diffusion Models emerge at the high temperature limit, and with the Argmax input (Section J.2), the denoiser processes a discrete sequence, while the generative process itself remains continuous on the simplex.

## AI USE STATEMENT

We acknowledge the use of AI models during the preparation of this manuscript. AI tools were used to assist in the writing of the manuscript, the derivation of the proofs and the writing of the code. Experiments were run with the use of agentic tools. The main ideas were all introduced without relying on AI tools. Interpretation of the results and first writing of the manuscript were entirely done with no LLM assistance. The TinyGSM dataset is fully generated by GPT-3.5 (as explained by Liu et al., 2023a). The authors take full responsibility for all content, scientific claims, citations, and conclusions presented in this work.

## REPRODUCIBILITY STATEMENT

To ensure the full reproducibility of our theoretical and empirical results, we provide comprehensive documentation across the paper and its supplementary material:

• Theoretical Claims and Proofs: Complete mathematical proofs for all propositions (Propositions 3.1 to 3.3 and 4.1) and extended results, including Gaussian fluctuation limits (Section E.7), the variational lower bound (Section F), and Simplex DMD distillation (Section G), are provided in Section E, Section F, Section G, with all mathematical assumptions explicitly stated.

• Algorithms and Pseudocode: Self-contained algorithmic pseudocode is detailed in Algorithm 1 (Training), Algorithm 2 (Inference), Algorithm 3 (Accelerated Marsaglia–Tsang Gamma Sampling; see also Section D), and Algorithm 4 (Simplex Distribution Matching Distillation), along with multi-token and classifier-free guidance extensions in Sections C.1 and C.2.

• Model Architectures and Hyperparameters: Detailed neural network architectures (Diffusion Transformer configurations, layer counts, embedding dimensions, attention heads), optimizer parameters (Adam/AdamW learning rates, warmup, EMA decay, gradient clipping), adaptive time schedules (Section J.1), denoiser input parameterizations (Section J.2), and variance schedules (ν<sub>t</sub>; Section 4.2) are specified in Sections J.1, J.2, J.4, J.5, J.7 and J.8.

• Experimental Details and Full Sweeps: Full experimental ablations and hyperparameter sweeps are documented in Section K and in the full result tables (Tables 21 to 23 and 29 to 32), including exact mean and standard deviation values across random seeds.

## REFERENCES

Sarah Alamdari, Nitya Thakkar, Rianne van den Berg, Alex Lu, Nicolo Fusi, Ava Amini, and Kevin Yang. Protein generation with evolutionary diffusion: Sequence is all you need. In NeurIPS Generative AI and Biology (GenBio) Workshop, 2023.

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Mart´ın Blazquez, Guilherme Penedo,´ Lewis Tunstall, Andres Marafioti, Hynek Kydl´ ´ıcek, Agustˇ ´ın Piqueres Lajar´ın, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clementine Fourrier, Ben Burtenshaw, Hugo´ Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, and Thomas Wolf. Smollm2: When smol goes big – data-centric training of a small language model. arXiv preprint arXiv:2502.02737, 2025.

Ali Alp. Sudoku puzzle generator. https://github.com/alicommit-malp/sudoku, 2024.

Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. In International Conference on Learning Representations, 2025.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, 2021.

Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, and Jian Zhou. Dirichlet diffusion score model for biological sequence generation. In International Conference on Machine Learning, 2023.

Jack Baker, Paul Fearnhead, Emily Fox, and Christopher Nemeth. Large-scale stochastic sampling from the probability simplex. Advances in Neural Information Processing Systems, 31, 2018.

Georgios Batzolis, Mark Girolami, and Luca Ambrogioni. CoBit: Language modeling with bitstream diffusion. arXiv preprint arXiv:2605.07013, 2026.

Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas Nolte, and Brian Karrer. Accelerated sampling from masked diffusion models via entropy bounded unmasking. In Advances in Neural Information Processing Systems, 2025.

Joe Benton, Yuyang Shi, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. From denoising diffusions to denoising Markov models. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 86(2):286–301, 2024.

G Richard Bickerton, Gaia V Paolini, Jer´ emy Besnard, Sorel Muresan, and Andrew L Hopkins.´ Quantifying the chemical beauty of drugs. Nature chemistry, 4(2):90–98, 2012.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 7432–7439, 2020.

Yoann Boget and Alexandros Kalousis. Unrestrained simplex denoising for discrete data: A non-Markovian approach applied to graph generation. arXiv preprint arXiv:2603.28572, 2026.

Bastian Boll, Daniel Gonzalez-Alvarado, and Christoph Schnorr. Generative modeling of dis-¨ crete joint distributions by e-geodesic flow matching on assignment manifolds. arXiv preprint arXiv:2402.07846, 2024.

Bastian Boll, Daniel Gonzalez-Alvarado, Stefania Petra, and Christoph Schnorr. Generative assign-¨ ment flows for representing and learning joint distributions of discrete data. Journal of Mathematical Imaging and Vision, 67(3):34, 2025.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http: //github.com/google/jax.

Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems, 2022.

Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. In International Conference on Machine Learning, 2024.

Nuria Alina Chandra, Yucen Lily Li, Alan N Amin, Alex Ali, Joshua Rollins, Sebastian W Ober, Aniruddh Raghu, and Andrew Gordon Wilson. A unification of discrete, Gaussian, and simplicial diffusion. In International Conference on Learning Representations, 2026.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. MaskGIT: Masked generative image transformer. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Jannis Chemseddine, Gregor Kornhardt, and Gabriele Steidl. Spherical flows for sampling categorical data. arXiv preprint arXiv:2605.05629, 2026.

Ricky TQ Chen and Yaron Lipman. Flow matching on general geometries. In International Conference on Learning Representations, 2024.

Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In Advances in Neural Information Processing Systems, 2018.

Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using dif fusion models with self-conditioning. In International Conference on Learning Representations, 2022.

Yuxin Chen, Chumeng Liang, Hangke Sui, Ruihan Guo, Chaoran Cheng, Jiaxuan You, and Ge Liu. LangFlow: Continuous diffusion rivals discrete in language modeling. arXiv preprint arXiv:2604.11748, 2026.

Chaoran Cheng, Jiahan Li, Jian Peng, and Ge Liu. Categorical flow matching on statistical manifolds. In Advances in Neural Information Processing Systems, 2024.

Chaoran Cheng, Jiahan Li, Jiajun Fan, and Ge Liu. α-flow: A unified framework for continuousstate discrete flow matching models. arXiv preprint arXiv:2504.10283, 2025.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Maurice G Cox. The numerical evaluation of B-splines. IMA Journal ofApplied Mathematics, 10 (2):134–149, 1972.

Oscar Davis, Samuel Kessler, Mircea Petrache, <sup>˙</sup>Ismail <sup>˙</sup>I Ceylan, Michael Bronstein, and Avishek J Bose. Fisher flow matching for generative modeling over discrete data. In Advances in Neural Information Processing Systems, 2024.

Carl de Boor. On calculating with b-splines. Journal of Approximation Theory, 6(1):50–62, 1972.

Valentin De Bortoli, Emile Mathieu, Michael Hutchinson, James Thornton, Yee Whye Teh, and Arnaud Doucet. Riemannian score-based generative modelling. In Advances in Neural Information Processing Systems, 2022.

Justin Deschenaux and Caglar Gulcehre. Promises, outlooks and challenges of diffusion language modeling. arXiv preprint arXiv:2406.11473, 2024.

Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast LLMs via self-distillation through time. In International Conference on Learning Representations, volume 2025, pp. 5007– 5045, 2025.

Justin Deschenaux and Caglar Gulcehre. Language modeling with hyperspherical flows. arXiv preprint arXiv:2605.11125, 2026.

Justin Deschenaux, Caglar Gulcehre, and Subham Sekhar Sahoo. The diffusion duality, chapter ii: ψ-samplers. In International Conference on Learning Representations, 2026a.

Justin Deschenaux, Lan Tran, and Caglar Gulcehre. Partition generative modeling: Masked model ing without masks. In International Conference on Learning Representations, volume 2026, pp. 132963–132988, 2026b.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, 2019.

Luc Devroye. Random variate generation in one line of code. In Proceedings ofthe 28th Conference on Winter Simulation, pp. 265–272, 1996.

Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, Curtis Hawthorne, Remi´ Leblond, Will Grathwohl, and Jonas Adler. Continuous diffusion for categorical data. arXiv preprint arXiv:2211.15089, 2022.

DiffusionGemma Team, Adrien Ali Ta¨ıga, James Assiene, Daniele Calandriello, Rahma Chaabouni, Joao Gante, Tamara von Glehn, Nate Keating, Chris Knutsen, Martin Kukla, et al. Diffu-˜ sionGemma technical report. arXiv preprint arXiv:2608.00146, 2026.

Ian Dunn and David Ryan Koes. Mixed continuous and categorical flow matching for 3D de novo molecule generation. arXiv preprint arXiv:2404.19739, 2024.

Peter Ertl and Ansgar Schuffenhauer. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal of Cheminfor matics, 1:1–11, 2009.

Nima Fathi, Torsten Scholak, and Pierre-Andre Noel. Unifying autoregressive and diffusion-based sequence generation. In Conference on Language Modeling, 2025.

Guhao Feng, Yihan Geng, Jian Guan, Wei Wu, Liwei Wang, and Di He. Theoretical benefit and limitation of diffusion language model. In Advances in Neural Information Processing Systems, 2025.

Nic Fishman, Leo Klarner, Emile Mathieu, Michael Hutchinson, and Valentin De Bortoli. Metropolis sampling for constrained diffusion models. In Advances in Neural Information Processing Systems, 2023.

Nic Fishman, Leo Klarner, Valentin De Bortoli, Emile Mathieu, and Michael Hutchinson. Diffusion models for constrained domains. Transactions on Machine Learning Research, 2024.

Griffin Floto, Thorsteinn Jonsson, Mihai Nica, Scott Sanner, and Eric Zhengyu Zhu. Diffusion on the probability simplex. In International Conference on Machine Learning Worshop on Sampling and Optimization in Discrete Space, 2023.

Antonio Franca and Alexander Tong. Hacking generative perplexity: Why unconditional text evaluation needs distributional metrics. arXiv preprint arXiv:2606.08417, 2026.

Aaron Gokaslan and Vanya Cohen. Openwebtext corpus. http://Skylion007.github.io/ OpenWebTextCorpus, 2019.

Samson Gourevitch, Yazid Janati, Dario Shariatian, Umut Simsekli, Eric Moulines, Eric P. Xing, and Alain Durmus. Uniform diffusion models revisited: Leave-one-out denoiser and absorbing state reformulation. arXiv preprint arXiv:2605.22765, 2026.

Will Grathwohl, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. Ffjord: Free-form continuous dynamics for scalable reversible generative models. In International Con ference on Learning Representations, 2018.

Will Grathwohl, Kevin Swersky, Milad Hashemi, David Duvenaud, and Chris Maddison. Oops i took a gradient: Scalable sampling for discrete distributions. In International Conference on Machine Learning, 2021.

Dylan Greaves. Extended one-liners for the Beta, Gamma, and Dirichlet distributions with shape parameters below one. arXiv preprint arXiv:2604.11199, 2026.

Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. In Advances in Neural Information Processing Systems, 2023.

Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. Ssd-LM: Semi-autoregressive simplex-based diffusion language model for text generation and modular control. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11575–11596, 2023.

Ulrich G Haussmann and Etienne Pardoux. Time reversal of diffusions. The Annals of Probability, 14(3):1188–1205, 1986.

Doron Haviv, Aram-Alexandre Pooladian, Dana Pe’er, and Brandon Amos. Wasserstein flow matching: Generative modeling over families of distributions. In International Conference on Machine Learning, 2025.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, 2020.

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. arXiv preprint arXiv:1904.09751, 2019.

Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, and Luke Zettlemoyer. Surface form competition: Why the highest probability answer isn’t always right. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 7038–7051, 2021.

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forre, and Max Welling. Argmax flows ´ and multinomial diffusion: Learning categorical distributions. In Advances in Neural Information Processing Systems, volume 34, pp. 12454–12465, 2021.

Emiel Hoogeboom, David Ruhe, Jonathan Heek, Thomas Mensink, and Tim Salimans. Beyond single tokens: Distilling discrete diffusion models via discrete MMD. arXiv preprint arXiv:2603.20155, 2026.

Keya Hu, Linlu Qiu, Yiyang Lu, Hanhong Zhao, Tianhong Li, Yoon Kim, Jacob Andreas, and Kaiming He. ELF: Embedded language flows. arXiv preprint arXiv:2605.10938, 2026.

Chin-Wei Huang, Milad Aghajohari, Joey Bose, Prakash Panangaden, and Aaron C Courville. Riemannian diffusion models. In Advances in Neural Information Processing Systems, 2022.

Jaehyeong Jo and Sung Ju Hwang. Continuous diffusion model for language modeling. In Advances in Neural Information Processing Systems, 2025.

Mingyu Jo, Jaesik Yoon, Justin Deschenaux, Caglar Gulcehre, and Sungjin Ahn. Loopholing discrete diffusion: Deterministic bypass of the sampling wall. In International Conference on Learning Representations, 2026.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. In Advances in Neural Information Processing Systems, 2022.

Tero Karras, Miika Aittala, Tuomas Kynka¨anniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine.¨ Guiding a diffusion model with a bad version of itself. In Advances in Neural Information Processing Systems, 2024.

Jaeyeon Kim, Jonathan Geuter, David Alvarez-Melis, Sham M. Kakade, and Sitan Chen. Stop training for the worst: Progressive unmasking accelerates masked diffusion training. In International Conference on Machine Learning, 2026a.

Jaeyeon Kim, Seunggeun Kim, Taekyun Lee, David Z. Pan, Hyeji Kim, Sham Kakade, and Sitan Chen. Fine-tuning masked diffusion for provable self-correction. In International Conference on Machine Learning, 2026b.

Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the ELBO with simple data augmentation. In Advances in Neural Information Processing Systems, 2023.

Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In Advances in Neural Information Processing Systems, 2021.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Nikita Kornilov, David Li, Tikhon Mavrin, Aleksei Leonov, Nikita Gushchin, Evgeny Burnaev, Iaroslav Koshelev, and Alexander Korotin. Universal inverse distillation for matching models with real-data supervision (no gans). In International Conference on Learning Representations, 2026.

Chanhyuk Lee, Jaehoon Yoo, Manan Agarwal, Sheel Shah, Jerry Huang, Aditi Raghunathan, Seunghoon Hong, Nicholas M Boffi, and Jinwoo Kim. One-step language modeling via continuous denoising. arXiv preprint arXiv:2602.16813, 2026.

Seul Lee, Karsten Kreis, Srimukh Prasad Veccham, Meng Liu, Danny Reidenbach, Saee Paliwal, Weili Nie, and Arash Vahdat. Genmol: A drug discovery generalist with discrete diffusion. In International Conference on Machine Learning, 2025.

Jose Lezama, Tim Salimans, Lu Jiang, Huiwen Chang, Jonathan Ho, and Irfan Essa. Discrete predictor-corrector diffusion models for image synthesis. In International Conference on Learning Representations, 2023.

David Li, Nikita Gushchin, Dmitry Abulkhanov, Eric Moulines, Ivan Oseledets, Maxim Panov, and Alexander Korotin. IDLM: Inverse-distilled diffusion language models. In International Conference on Machine Learning, 2026.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, and Yi Zhang. TinyGSM: achieving ą 80% on GSM8k with small language models. arXiv preprint arXiv:2312.09241, 2023a.

Guan-Horng Liu, Tianrong Chen, Evangelos Theodorou, and Molei Tao. Mirror diffusion models for constrained and watermarked generation. In Advances in Neural Information Processing Systems, 2023b.

Sulin Liu, Juno Nam, Andrew Campbell, Hannes Stark, Yilun Xu, Tommi Jaakkola, and Rafael¨ Gomez-Bombarelli. Think while you generate: Discrete diffusion with planned denoising. In´ International Conference on Learning Representations, 2025.

Yue Liu, Yuzhong Zhao, Zheyong Xie, Qixiang Ye, Jianbin Jiao, Yao Hu, Shaosheng Cao, and Yunfan Liu. Balancing understanding and generation in discrete diffusion models. In International Conference on Machine Learning, 2026.

Aaron Lou and Stefano Ermon. Reflected diffusion models. In International Conference on Machine Learning, 2023.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion language modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning, 2024.

Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diffinstruct: A universal approach for transferring knowledge from pre-trained diffusion models. In Advances in Neural Information Processing Systems, 2023.

Rabeeh Karimi Mahabadi, Hamish Ivison, Jaesung Tae, James Henderson, Iz Beltagy, Matthew E Peters, and Arman Cohan. Tess: Text-to-text self-conditioned simplex diffusion. In Proceedings ofthe 18th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 2347–2361, 2024.

George Marsaglia and Wai Wan Tsang. A simple method for generating Gamma variables. ACM Transactions on Mathematical Software, 26(3):363–372, 2000.

Kai Wang Ng, Guo-Liang Tian, and Man-Lai Tang. Dirichlet and Related Distributions: Theory, Methods and Applications. John Wiley & Sons, 2011.

Emmanuel Noutahi, Cristian Gabellini, Michael Craig, Jonathan S. C. Lim, and Prudencio Tossou. Gotta be safe: a new framework for molecular design. Digital Discovery, 3(4):796–804, 04 2024.

Andrey Okhotin, Dmitry Molchanov, Vladimir Arkhipkin, Grigory Bartosh, Viktor Ohanesian, Aibek Alanov, and Dmitry P Vetrov. Star-shaped denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, 2023.

Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. In International Conference on Learning Representations, 2025.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

Peter Potaptchik, Jason Yim, Adhi Saravanan, Peter Holderrieth, Eric Vanden-Eijnden, and Michael S. Albergo. Discrete flow maps. arXiv preprint arXiv:2604.09784, 2026.

Patrick Pynadath, Jiaxin Shi, and Ruqi Zhang. Candi: Hybrid discrete-continuous diffusion models. In International Conference on Machine Learning, 2026a.

Patrick Pynadath, Jiaxin Shi, and Ruqi Zhang. Generative frontiers: Why evaluation matters for diffusion language models. arXiv preprint arXiv:2604.02718, 2026b.

Yiming Qin, Manuel Madeira, Dorina Thanou, and Pascal Frossard. Defog: Discrete flow matching for graph generation. In International Conference on Machine Learning, 2025.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019. URL https://api. semanticscholar.org/CorpusID:160025533.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

Gabriel Raya, Bac Nguyen, Georgios Batzolis, Yuhta Takida, Dejan Stancevic, Naoki Murata, Chieh-Hsin Lai, Yuki Mitsufuji, and Luca Ambrogioni. Noise scheduling as information-guided allocation in diffusion training. arXiv preprint arXiv:2602.18647, 2026.

Pierre H Richemond, Sander Dieleman, and Arnaud Doucet. Categorical SDEs with simplex diffusion. arXiv preprint arXiv:2210.14784, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨ resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Daan Roos, Oscar Davis, Floor Eijkelboom, Michael M. Bronstein, Max Welling, <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan, Luca Ambrogioni, and Jan-Willem van de Meent. Categorical flow maps. In International Conference on Machine Learning, 2026.

Subham S Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In Advances in Neural Information Processing Systems, 2024.

Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, and Volodymyr Kuleshov. The diffusion duality. In International Conference on Machine Learning, 2025.

Subham Sekhar Sahoo, Jean-Marie Lemercier, Zhihan Yang, Justin Deschenaux, Jingyu Liu, John Thickstun, and Ante Jukic. Scaling beyond masked diffusion language models. In International Conference on Machine Learning, 2026.

Jinya Sakurai, Patrick Pynadath, Satoshi Hayakawa, Jaehong Yoon, Xulei Yang, Nancy F Chen, and Xun Xu. Simplex relaxation for discrete diffusion. arXiv preprint arXiv:2608.10615, 2026.

Tim Salimans, Thomas Mensink, Jonathan Heek, and Emiel Hoogeboom. Multistep distillation of diffusion models via moment matching. In Advances in Neural Information Processing Systems, 2024.

Yair Schiff, Subham Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dalla-Torre, Bernardo Almeida, Alexander Rush, Thomas Pierrot, and Volodymyr Kuleshov. Simple guidance mechanisms for discrete diffusion models. In International Conference on Learning Representations, 2025.

Alexander Shabalin, Simon Elistratov, Viacheslav Meshchaninov, Ildus Sadrtdinov, and Dmitry Vetrov. Why Gaussian diffusion models fail on discrete data? In Third Conference on Language Modeling, 2026.

Junzhe Shen, Jieru Zhao, Ziwei He, and Zhouhan Lin. CoDAR: Continuous diffusion language models are more powerful than you think. arXiv preprint arXiv:2603.02547, 2026.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and general ized masked diffusion for discrete data. In Advances in Neural Information Processing Systems, 2024.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021b.

Yuxuan Song, Zhe Zhang, Yu Pei, Jingjing Gong, Qiying Yu, Zheng Zhang, Mingxuan Wang, Hao Zhou, Jingjing Liu, and Wei-Ying Ma. Shortlisting model: A streamlined simplex diffusion for discrete variable generation. In Advances in Neural Information Processing Systems, 2025.

Hannes Stark, Bowen Jing, Chenyu Wang, Gabriele Corso, Bonnie Berger, Regina Barzilay, and Tommi Jaakkola. Dirichlet flow matching with applications to DNA sequence design. In International Conference on Machine Learning, 2024.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568(C), 2024.

Haoran Sun, Hanjun Dai, Bo Dai, Haomin Zhou, and Dale Schuurmans. Discrete Langevin samplers via Wasserstein gradient flow. In International Conference on Artificial Intelligence and Statistics, 2023.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee´ Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and\` efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Petar Velickoviˇ c, Federico Barbero, Christos Perivolaropoulos, Simon Osindero, and Razvan Pas-´ canu. Perplexity cannot always tell right from wrong. arXiv preprint arXiv:2601.22950, 2026.

Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal´ Frossard. DiGress: Discrete denoising diffusion for graph generation. In International Conference on Learning Representations, 2023.

Dimitri von Rutte, Janis Fluri, Yuhui Ding, Antonio Orvieto, Bernhard Sch¨ olkopf, and Thomas¨ Hofmann. Generalized interpolating discrete diffusion. In International Conference on Machine Learning, 2025.

Dimitri von Rutte, Janis Fluri, Omead Pooladzandi, Bernhard Sch¨ olkopf, Thomas Hofmann, and¨ Antonio Orvieto. Scaling behavior of discrete diffusion language models. In International Conference on Learning Representations, 2026.

Guanghan Wang, Yair Schiff, Subham Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling. In Advances in Neural Information Processing Systems, 2025.

Linxuan Wang, Ziyi Wang, Yikun Bai, Wei Deng, Guang Lin, and Qifan Song. Generalized discrete diffusion with self-correction. In International Conference on Machine Learning, 2026.

Joseph L Watson, David Juergens, Nathaniel R Bennett, Brian L Trippe, Jason Yim, Helen E Eisenach, Woody Ahern, Andrew J Borst, Robert J Ragotte, Lukas F Milles, et al. De novo design of protein structure and function with RFdiffusion. Nature, 620(7976):1089–1100, 2023.

Bernardo Williams, Victor M Yeom-Song, Marcelo Hartmann, and Arto Klami. Simplex-to-Euclidean bijections for categorical flow matching. In Artificial Intelligence and Statistics, 2026.

Jin Xu, Xiaojiang Liu, Jianhao Yan, Deng Cai, Huayang Li, and Jian Li. Learning to break the loop: Analyzing and mitigating repetitions for neural text generation, 2022. arXiv preprint arXiv:2206.02369, 2022.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

Zhihan Yang, Wei Guo, Shuibai Zhang, Subham Sekhar Sahoo, Yongxin Chen, Arash Vahdat, Morteza Mardani, and John Thickstun. Continuous diffusion scales competitively with discrete diffusion for language. arXiv preprint arXiv:2605.18530, 2026.

Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman,¨ and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4791–4800, 2019.

Jinwei Zhang, Dimitri von Rutte, Yuhui Ding, and Thomas Hofmann. Denoising is not the end: Dis-¨ crete diffusion language models with self-correction. In Workshop on Latent & Implicit Thinking – Going Beyond CoT Reasoning, 2026.

Kaizhong Zhang and Dennis Shasha. Simple fast algorithms for the editing distance between trees and related problems. SIAM Journal on Computing, 18(6):1245–1262, 1989.

Yixiu Zhao, Jiaxin Shi, Feng Chen, Shaul Druckmann, Lester Mackey, and Scott Linderman. Informed correctors for discrete diffusion models. In Advances in Neural Information Processing Systems, 2025.

## ORGANIZATION OF THE SUPPLEMENTARY

## Part I: Algorithms and Implementation Details

A Beta, Dirichlet and Gamma Distributions 21   
A.1 Basic Definitions 21   
A.2 Some Useful Properties of Dirichlet Distributions 21   
B Training and Inference Algorithms 22   
B.1 Training 22   
B.2 Inference 22   
C Several Extensions 24   
C.1 Extension to Multiple Tokens 24   
C.2 Classifier-Free Guidance 24   
D Fast Sampling of Dirichlet Random Variables 25   
Part II: Theory and Generalizations   
E Proofs and Additional Results 27   
E.1 Proof of Proposition 3.1 . 27   
E.2 Proof of Proposition 3.2 . 27   
E.3 Proof of Proposition 3.3 . 27   
E.4 A Representation of $P _ { t }$ 28   
E.5 Illustration and Temperature Limits for Proposition E.1 29   
E.6 Proof of Proposition 4.1 . 31   
E.7 Gaussian Fluctuations . 33   
F Variational Lower Bound 35   
G Distillation 36   
G.1 Simplex Distribution Matching Distillation 36   
G.2 Link with Inverse Distilled Language Models 38   
Part III: Connection with the Literature   
H Extended Related Work 40   
H.1 General overview . 40   
H.2 Simplex Relaxation for Discrete Diffusion . 41   
Extended Background 43   
I.1 Discrete Diffusion Models 43   
I.2 Predictor-Corrector Sampler 44   
I.3 Self-Conditioning and Loopholing 44   
I.4 Dirichlet Flow Matching 45   
Part IV: Experimental Setup, Ablations and Results   
J Extended Experimental Details 47   
J.1 Time Samplers and Sampling Schedules . 47   
J.2 Denoiser Input for SDMs . 47   
J.3 Dirichlet Flow Matching Baseline 48   
J.4 Sudoku 50   
J.5 TinyGSM 50   
J.6 OpenWebText . 51   
J.7 Language Understanding 51   
J.8 Unconditional molecular generation 52   
K Additional Experimental Results 54   
K.1 Sudoku 54   
K.2 TinyGSM 54   
K.3 Distillation on TinyGSM . 58   
K.4 Auto-Guidance on TinyGSM . 58   
K.5 OpenWebText . 60   
K.6 Language Understanding 62   
K.7 Unconditional molecular generation 63   
L Additional OWT Samples 69   
M Full Result Tables 76   
M.1 Sudoku 76   
M.2 TinyGSM . 80

## Part I: Algorithms and Implementation Details

## A BETA, DIRICHLET AND GAMMA DISTRIBUTIONS

## A.1 BASIC DEFINITIONS

For sake of completeness, we recall elementary facts about the Beta, Dirichlet and Gamma distributions. The Beta distribution is a distribution on r0, 1s. Given $a , b > 0$ , the associated Beta density $\mathrm { B e t a } ( x ; a , b )$ is given for any $x \in [ 0 , 1 ]$ by

$$
\operatorname { B e t a } ( x ; a , b ) = { \frac { \Gamma ( a + b ) } { \Gamma ( a ) \Gamma ( b ) } } x ^ { a - 1 } ( 1 - x ) ^ { b - 1 } ,
$$

where $\Gamma ( \cdot )$ is the Gamma function. The Dirichlet distribution is a distribution on the simplex $\Delta _ { N }$ Given $\beta = ( \beta _ { 1 } , \dots , \beta _ { N } )$ , the associated Dirichlet density $\operatorname { D i r } ( P ; \beta )$ is given for any $P \in { \bar { \Delta } } _ { N }$ by

$$
\mathrm { D i r } ( P ; \beta ) = \frac { \Gamma ( \sum _ { i = 1 } ^ { N } \beta _ { i } ) } { \prod _ { i = 1 } ^ { N } \Gamma ( \beta _ { i } ) } \prod _ { i = 1 } ^ { N } p _ { i } ^ { \beta _ { i } - 1 } ,
$$

We have

$$
\mathbb { E } [ P ] = { \bar { P } } = \left( { \frac { \beta _ { 1 } } { C } } , . . . , { \frac { \beta _ { N } } { C } } \right) , \quad \operatorname { C o v } [ P ] = { \frac { \operatorname { d i a g } ( { \bar { P } } ) - { \bar { P } } { \bar { P } } ^ { \top } } { C + 1 } } ,\tag{13}
$$

where $\begin{array} { r } { C = \sum _ { i = 1 } ^ { N } \beta _ { i } } \end{array}$ the concentration parameter of the Dirichlet distribution. The Gamma distribution is a distribution on r0, 8q defined for parameters $\alpha , \beta > 0$ by

$$
{ \mathrm { G a m m a } } ( x ; \alpha , \beta ) = { \frac { \beta ^ { \alpha } } { \Gamma ( \alpha ) } } x ^ { \alpha - 1 } \exp ( - \beta x ) .
$$

We will use extensively in practice the following fact. For N independently distributed Gamma random variables $X _ { 1 } \sim$ Gamma $( \alpha _ { 1 } , 1 ) , \cdot \cdot \cdot , X _ { N } \sim$ Gamma $( \alpha _ { N } , 1 )$ , we have

$$
P = \left( { \frac { X _ { 1 } } { \sum _ { i = 1 } ^ { N } X _ { i } } } , \cdots , { \frac { X _ { N } } { \sum _ { i = 1 } ^ { N } X _ { i } } } \right) \sim \mathrm { D i r } ( \alpha _ { 1 } , \cdot \cdot \cdot , \alpha _ { N } ) .
$$

We will use the following convention throughout. Whenever a Gamma/Beta/Dirichlet parameter lies on the boundary (for instance, some Dirichlet coordinates are zero, or a formula is obtained as the limit $t \downarrow 0 )$ , the corresponding distribution is understood in the natural degenerate/weak-limit sense. In particular, Dirichlet distributions with some zero coordinates are supported on the corresponding face of the simplex, Beta distributions with a zero parameter are degenerate at 0 or 1, and formulas involving $t = 0$ are interpreted as limits of the same formulas for $t > 0$

## A.2 SOME USEFUL PROPERTIES OF DIRICHLET DISTRIBUTIONS

We now recall two simple results for Dirichlet distributions. To keep the paper self-contained, we provide proofs below without any claim of originality.

Proposition A.1 (Sum of Dirichlet random variables): Let $\alpha , \beta \in \mathbb { R } ^ { N }$ with $\beta _ { i } \geqslant \alpha _ { i } \geqslant 0$ $f o r \bar { a } n y i \in \{ 1 , \ldots , N \}$ and assume that $X \ \sim \ \operatorname { D i r } ( \alpha )$ Denote $\gamma ~ = ~ \beta - \alpha$ and let $W \sim$ Beta $\textstyle ( \sum _ { i = 1 } ^ { N } \alpha _ { i } , \sum _ { i = 1 } ^ { N } \gamma _ { i } )$ independent of X. Let $V \sim \operatorname { D i r } ( \gamma )$ , independent of X and W. Finally, let Y be given by

$$
Y = W X + ( 1 - W ) V .
$$

Then, we have that $Y \sim \operatorname { D i r } ( \beta )$

Proof. Using the Gamma characterization of Dirichlet random variables, there exist independent random variables $( Z _ { i } ) _ { i = } ^ { N }$ such that $Z _ { i } ~ \sim ~ \mathrm { G a m m a } ( \alpha _ { i } , 1 )$ and $\begin{array} { r } { X ~ = ~ \left( Z _ { i } / \sum _ { j = 1 } ^ { N } Z _ { j } \right) _ { i = 1 } ^ { N } . } \end{array}$ Similarly, there exist independent random variables $( Z _ { i } ^ { \prime } ) _ { i = 1 } ^ { N }$ such that $Z _ { i } ^ { \prime } \sim \mathrm { G a m m a } ( \gamma _ { i } , 1 )$ and $\begin{array} { r } { V = \left( Z _ { i } ^ { \prime } / \sum _ { j = 1 } ^ { N } Z _ { j } ^ { \prime } \right) _ { i = 1 } ^ { N } } \end{array}$ . We denote $\begin{array} { r } { \bar { Z } = \sum _ { i = 1 } ^ { N } Z _ { i } , \bar { Z } ^ { \prime } = \sum _ { i = 1 } ^ { N } Z _ { i } ^ { \prime } , } \end{array}$ , and $W = \bar { Z } / ( \bar { Z } + \bar { Z } ^ { \prime } )$ . Note that $\bar { Z }$ and $\bar { Z } ^ { \prime }$ are independent of each other and independent of V and X. The random variable W is independent of $\dot { X }$ and $V$ by Lukacs’s Proportion-Sum Independence Theorem, see Ng et al. (2011). In addition, $\begin{array} { r } { \bar { Z } \sim \mathrm { G a m m a } ( \sum _ { i = 1 } ^ { N } \alpha _ { i } , 1 ) } \end{array}$ and $\begin{array} { r } { \bar { Z } ^ { \prime } \sim \mathrm { G a m m a } ( \sum _ { i = 1 } ^ { N } \gamma _ { i } , 1 ) } \end{array}$ using the summation property of independent Γ distributions. Now using the characterization of Beta distributions with Gamma distributions, we get that $W \sim$ Beta $\textstyle ( \sum _ { i = 1 } ^ { N ^ { - } } \alpha _ { i } , \sum _ { i = 1 } ^ { N } \gamma _ { i } )$ . Finally, we have that

$$
Y = W X + ( 1 - W ) V = \left( \frac { Z _ { i } + Z _ { i } ^ { \prime } } { \sum _ { j = 1 } ^ { N } ( Z _ { j } + Z _ { j } ^ { \prime } ) } \right) _ { i = 1 } ^ { N } .
$$

We conclude the proof using the summation property of independent Γ random variables and the Γ characterization of Dirichlet random variables. □

Proposition A.2 (Dirichlet thinning): Let $X \sim \operatorname { D i r } ( \alpha )$ with $\boldsymbol { \alpha } \in \mathbb { R } ^ { N }$ with $\alpha _ { i } \ \geqslant \ 0$ for any $i \in \{ 1 , \ldots , N \}$ . The construction has to be understood on the supporting face $I = \{ i : \alpha _ { i } >$ $0 \} ,$ ; coordinates outside I are fixed at zero and omittedfrom the normalization. For $\dot { \rho } \in ( 0 , 1 ]$ and $i \in I ,$ let $B _ { i } \sim \mathrm { B e t a } ( \rho \dot { \alpha } _ { i } , ( 1 - \rho ) \alpha _ { i } )$ be mutually independent random variables, also independent of X, and define Y by

$$
Y _ { i } = { \frac { B _ { i } X _ { i } } { \sum _ { j \in I } B _ { j } X _ { j } } } \quad f o r i \in I , \qquad Y _ { i } = 0 \quad f o r i \not \in I .
$$

Then $Y \sim \operatorname { D i r } ( \rho \alpha ) .$

Proof. By restricting to the supporting face and relabelling its coordinates, it is enough to consider $\alpha _ { i } > 0$ for every $i \bar { \in } \{ 1 , \ldots , \bar { N } \}$ . We sample mutually independent variables $Z _ { i } \sim \bar { \Gamma } ( \alpha _ { i } , 1 )$ and $B _ { i } \sim \mathrm { B e t a } ( \rho \alpha _ { i } , ( 1 - \rho ) \alpha _ { i } )$ , and define $\begin{array} { r } { X = ( Z _ { i } / \sum _ { j = 1 } ^ { N } Z _ { j } ) _ { i = 1 } ^ { N } } \end{array}$ . This defines the joint distribution of $( X , ( B _ { i } ) _ { i = 1 } ^ { N } )$ . For any $i \in \{ 1 , \ldots , N \}$ , let $Z _ { i } ^ { \prime } = Z _ { i } B _ { i }$ and we have that

$$
Y = \left( \frac { B _ { i } X _ { i } } { \sum _ { j = 1 } ^ { N } B _ { j } X _ { j } } \right) _ { i = 1 } ^ { N } = \left( \frac { B _ { i } Z _ { i } } { \sum _ { j = 1 } ^ { N } B _ { j } Z _ { j } } \right) _ { i = 1 } ^ { N } = \left( \frac { Z _ { i } ^ { \prime } } { \sum _ { j = 1 } ^ { N } Z _ { j } ^ { \prime } } \right) _ { i = 1 } ^ { N } .
$$

We now use the fact that for independent random variables $A \ \sim \ \mathrm { B e t a } ( b , a \ - \ b )$ and $B \ \sim$ ${ \mathrm { G a m m a } } ( a , 1 )$ then $A B \ \sim \ \mathrm { G a m m a } ( b , 1 )$ , using Lukacs’s Proportion-Sum Independence Theorem, see Ng et al. (2011). So for any $i \in \{ 1 , \ldots , N \}$ , with $A \ = \ B _ { i } , \ B \ = \ Z _ { i } , \ a \ = \ \alpha _ { i }$ and $b ~ = ~ \rho \alpha _ { i }$ . It follows that that $( Z _ { i } ^ { \prime } ) _ { i = 1 } ^ { N }$ is a collection of independent random variables such that $Z _ { i } ^ { \prime } \sim \mathrm { G a m m a } ( \rho \alpha _ { i } , 1 )$ . Therefore, using once again the Gamma representation of Dirichlet random variables, $Y \sim \mathrm { D i r } ( \rho \dot { \alpha } )$ q. □

## B TRAINING AND INFERENCE ALGORITHMS

## B.1 TRAINING

The training algorithm is detailed in Algorithm 1.

## B.2 INFERENCE

The inference algorithm is described in Algorithm 2. For readability, this is presented in the case $0 < \rho _ { s , t } ^ { \kappa } < 1$ with non-degenerate Beta and Dirichlet parameters (boundary cases are understood by continuity).

Algorithm 1 Training of Simplex Diffusion Model   
Require: Data distribution $p _ { \mathrm { d a t a } } .$ Reference distribution $\pi ~ \in ~ \Delta _ { N }$ (e.g., Uniform), Schedule   
$( \alpha _ { t } ) _ { t \in [ 0 , 1 ] }$ , Concentration $\left( c _ { t } \right) _ { t \in \left( 0 , 1 \right] }$ , time-sampling density $w ( t )$ , and Model $\hat { P } _ { \theta } ~ : ~ \Delta _ { N } ~ \times$   
$[ 0 , 1 ] \stackrel { \cdot } {  } \bar { \Delta } _ { N }$   
1: while not converged do   
2: 1. Sample Data and Time   
3: Sample $x _ { 0 } \sim p _ { \mathrm { d a t a } }$ $\vartriangleright x _ { 0 } \in \{ 1 , \dots , N \}$   
4: Sample time $t \sim w ( t )$ Ź Uniform by default   
5: 2. Compute Corruption Parameters   
6: $P _ { 0 }  e _ { x _ { 0 } }$ $\mathsf { \Gamma } \simeq P _ { 0 } \in \{ 0 , 1 \} ^ { N } ; \mathrm { i . e . , } P _ { 0 } = \mathrm { O n e H o t } ( x _ { 0 , \ast } )$   
7: $\beta _ { t } \gets c _ { t } \left( \alpha _ { t } P _ { 0 } + ( 1 - \alpha _ { t } ) \pi \right)$ Ź Dirichlet parameters $\beta _ { t } \in \mathbb { R } _ { + } ^ { N }$   
8: 3. Sample Noisy Simplex State $P _ { t }$   
9: Sample $\dot { P } _ { t } \sim \operatorname * { D i r } ( \beta _ { t } )$   
10: 4. Predict and Optimize   
11: $\hat { P } _ { 0 } \gets \hat { P } _ { \theta } ( t , P _ { t } )$ Ź Model prediction (Softmax output)   
12: $\mathcal { L } \gets \mathrm { C r o s s } \underline { { \mathrm { E n t r o p y } } } ( x _ { 0 } , \hat { P } _ { 0 } ) = - \log ( \hat { P } _ { 0 , x _ { 0 } } )$   
13: $\theta \gets \theta - \eta \nabla _ { \theta } \mathcal { L }$ Ź Gradient descent step   
14: end while

Algorithm 2 Inference for Simplex Diffusion Model   
Require: Trained Model ${ \hat { P } } _ { \theta } ,$ Prior π, Schedule $( \alpha _ { t } ) _ { t \in [ 0 , 1 ] } ,$ , Concentration $( c _ { t } ) _ { t \in [ 0 , 1 ] }$ , Churn $\kappa \in$   
$[ 0 , 1 ]$ , Number of steps M, Time sequence $0 = t _ { 0 } < \cdots < t _ { M } = 1 .$   
1: 1. Initialize   
2: Sample $P _ { 1 } \sim \operatorname* { D i r } ( c _ { 1 } \pi )$ Ź Sample from scaled prior   
3: 2. Reverse Generative Loop   
4: for $k = M , M - 1 , \ldots , 2$ do   
5: $t  t _ { k } , \qquad s  t _ { k - 1 }$   
6: a. Sample a Clean Vertex   
7: Sample $\widetilde { x } _ { 0 } \sim \mathrm { C a t } ( \hat { P } _ { \theta } ( t , P _ { t _ { k } } ) )$   
8: $P _ { 0 }  e _ { \tilde { x } _ { 0 } }$   
9: b. Compute $p _ { s | 0 , t } ( P _ { s } \mid P _ { 0 } , P _ { t } )$ Parameters   
10: $r _ { s , t } \gets \operatorname* { m i n } \left\{ 1 , \frac { c _ { s } ( 1 - \alpha _ { s } ) } { c _ { t } ( 1 - \alpha _ { t } ) } \right\}$   
11: $\rho \gets ( 1 - \kappa ) \dot { r } _ { s , t }$   
12: $a _ { W } \gets \rho c _ { t }$   
13: $b _ { W } \gets c _ { s } - \rho c _ { t }$   
14: $\beta _ { V } \gets \beta _ { s } ( \dot { P _ { 0 } } , \pi ) - \rho \beta _ { t } ( P _ { 0 } , \pi )$   
15: c. Sample Variables   
16: Sample $W \sim \mathrm { B e t a } ( a _ { W } , b _ { W } )$ and $V \sim \operatorname { D i r } ( \beta _ { V } )$   
17: d. Compute $P _ { t _ { k } } ^ { \kappa }$   
18: for $i \in \{ 1 , \ldots , \ddot { N } \}$ do   
19: Sample $B _ { i } \sim \stackrel { \prime } { \mathrm { B e t a } } ( \rho \beta _ { t } ( P _ { 0 } , \pi ) _ { i } , ( 1 - \rho ) \beta _ { t } ( P _ { 0 } , \pi ) _ { i } )$   
20: end for   
$B \odot P _ { t _ { k } }$   
21: $P _ { t _ { k } } ^ { \kappa } \gets \frac { \smile \smile \llangle \iota _ { k } } { \sum _ { i = 1 } ^ { N } B _ { i } P _ { t _ { k } , i } }$ Ź Element-wise mult. & normalize   
22: e. Update State   
23: $\hat { P _ { t _ { k - 1 } } } \dot {  } W P _ { t _ { k } } ^ { \kappa } + ( 1 - W ) V$   
24: end for   
25: 3. Endpoint   
26: Sample $x _ { \mathrm { d a t a } } \sim \mathrm { C a t } ( \hat { P } _ { \theta } ( t _ { 1 } , P _ { t _ { 1 } } ) )$   
27: $P _ { t _ { 0 } } \gets e _ { x _ { \mathrm { d a t a } } }$   
28: return $x _ { \mathrm { d a t a } }$

## C SEVERAL EXTENSIONS

## C.1 EXTENSION TO MULTIPLE TOKENS

So far, we have presented Simplex Diffusion Models for a single token taking values in $\mathcal { X } =$ $\{ 1 , \ldots , N \}$ . We now discuss how to extend the construction to sequences $x _ { 0 } = ( \overline { { x _ { 0 } ^ { 1 } } } , \ldots , x _ { 0 } ^ { L } ) \in \mathcal { X } ^ { L }$ To obtain a scalable solution, we consider a product of simplices and represent the state at time t by

$$
P _ { t } = ( P _ { t } ^ { 1 } , \dots , P _ { t } ^ { L } ) \in ( \Delta _ { N } ) ^ { L } , \qquad P _ { 0 } ^ { \ell } = e _ { x _ { 0 } ^ { \ell } } , \quad \ell \in \{ 1 , \dots , L \} .
$$

We then define the forward process independently across positions:

$$
p _ { t | 0 } ( P _ { t } \mid P _ { 0 } ) = \prod _ { \ell = 1 } ^ { L } \mathrm { D i r } \big ( P _ { t } ^ { \ell } ; \beta _ { t } ( P _ { 0 } ^ { \ell } , \pi ) \big ) , \qquad \beta _ { t } ( P _ { 0 } ^ { \ell } , \pi ) = c _ { t } \big ( \alpha _ { t } P _ { 0 } ^ { \ell } + ( 1 - \alpha _ { t } ) \pi \big ) .
$$

The denoiser is now a joint sequence model

$$
\hat { P } _ { \theta } : ( \Delta _ { N } ) ^ { L } \times [ 0 , 1 ] \to ( \Delta _ { N } ) ^ { L } , \qquad \hat { P } _ { \theta } ( t , P _ { t } ) = ( \hat { P } _ { 0 } ^ { 1 } , \ldots , \hat { P } _ { 0 } ^ { L } ) ,
$$

where each $\hat { P } _ { 0 } ^ { \ell } \in \Delta _ { N }$ is the predicted clean distribution at position ℓ. Importantly, although the forward and reverse kernels factorize across positions, each $\hat { P } _ { 0 } ^ { \ell }$ can depend on the full noisy sequence $P _ { t }$

Training is performed with the sequence-level extension of the cross-entropy objective. At inference time, Proposition 3.3 is applied independently at each position after sampling a clean token from each predicted categorical distribution. More precisely, for any $0 < s < t \leqslant 1$ , we first draw, conditionally independently across positions,

$$
\widetilde { x } _ { 0 } ^ { \ell } \sim { \mathrm { C a t } } ( \hat { P } _ { 0 } ^ { \ell } ) , \qquad \widetilde { P } _ { 0 } ^ { \ell } = e _ { \widetilde { x } _ { 0 } ^ { \ell } } , \qquad \ell \in \{ 1 , \dots , L \} .
$$

We then sample independently across positions

$$
\begin{array} { r } { W _ { s , t , \ell } ^ { \kappa } \sim \mathrm { B e t a } \left( \rho _ { s , t } ^ { \kappa } c _ { t } , c _ { s } - \rho _ { s , t } ^ { \kappa } c _ { t } \right) , \qquad V _ { s , t , \ell } ^ { \kappa } \sim \mathrm { D i r } \left( \beta _ { s } ( \widetilde { P } _ { 0 } ^ { \ell } , \pi ) - \rho _ { s , t } ^ { \kappa } \beta _ { t } ( \widetilde { P } _ { 0 } ^ { \ell } , \pi ) \right) , } \end{array}
$$

where

$$
\rho _ { s , t } ^ { \kappa } = ( 1 - \kappa ) r _ { s , t } , \qquad r _ { s , t } = \operatorname* { m i n } \left\{ 1 , \frac { c _ { s } ( 1 - \alpha _ { s } ) } { c _ { t } ( 1 - \alpha _ { t } ) } \right\} .
$$

For each $i \in \{ 1 , \ldots , N \}$ , we also sample

$$
B _ { i } ^ { \ell } \sim \mathrm { B e t a } \left( \rho _ { s , t } ^ { \kappa } \beta _ { t } ( \widetilde { P } _ { 0 } ^ { \ell } , \pi ) _ { i } , ( 1 - \rho _ { s , t } ^ { \kappa } ) \beta _ { t } ( \widetilde { P } _ { 0 } ^ { \ell } , \pi ) _ { i } \right) .
$$

We then define

$$
( P _ { s , t } ^ { \ell } ) _ { i } ^ { \kappa } = \frac { B _ { i } ^ { \ell } P _ { t , i } ^ { \ell } } { \sum _ { j = 1 } ^ { N } B _ { j } ^ { \ell } P _ { t , j } ^ { \ell } } , \qquad P _ { s } ^ { \ell } = W _ { s , t , \ell } ^ { \kappa } ( P _ { s , t } ^ { \ell } ) ^ { \kappa } + ( 1 - W _ { s , t , \ell } ^ { \kappa } ) V _ { s , t , \ell } ^ { \kappa } .
$$

When $\rho _ { s , t } ^ { \kappa } = 1$ , we set $( P _ { s , t } ^ { \ell } ) ^ { \kappa } = P _ { t } ^ { \ell }$ directly and omit the $B _ { i } ^ { \ell }$ variables.

This yields the factorized mixture reverse transition

$$
p _ { s | t } ^ { \theta } ( P _ { s } \mid P _ { t } ) = \prod _ { \ell = 1 } ^ { L } \left\{ \sum _ { j = 1 } ^ { N } \hat { P } _ { 0 , j } ^ { \ell } p _ { s | 0 , t } ( P _ { s } ^ { \ell } \mid P _ { 0 } ^ { \ell } = e _ { j } , P _ { t } ^ { \ell } ) \right\} .
$$

After the final positive-time step, we compute $( \hat { P } _ { 0 } ^ { 1 } , \dots , \hat { P } _ { 0 } ^ { L } ) = \hat { P } _ { \theta } ( t _ { 1 } , P _ { t _ { 1 } } )$ and sample $x _ { \mathrm { d a t a } } ^ { \ell } \sim$ $\mathrm { C a t } ( \hat { P } _ { 0 } ^ { \ell } )$ independently across positions conditional on the prediction.

This factorized extension preserves the simplicity of the single-token construction while allowing the neural predictor $\hat { P } _ { \theta }$ to exploit full sequence context.

## C.2 CLASSIFIER-FREE GUIDANCE

We now describe how to incorporate classifier-free guidance into the generative reverse process for the single-token case. The extension to multiple tokens is straightforward. Let $c \in { \mathcal { C } }$ denote an external condition $( \mathrm { e . g . }$ , class label, text prompt, side information). We replace the predictor $\hat { P } _ { \theta } ( t , P _ { t } )$ by a conditional predictor $\hat { P } _ { \theta } ( t , c , P _ { t } )$

Training. During training, we sample a dropped condition c˜ according to

$$
\tilde { c } = \left\{ \begin{array} { l l } { c , } & { \mathrm { w i t h } \mathrm { p r o b a b i l i t y } 1 - p _ { \mathrm { d r o p } } , } \\ { \emptyset , } & { \mathrm { w i t h } \mathrm { p r o b a b i l i t y } p _ { \mathrm { d r o p } } , } \end{array} \right.
$$

where ∅ denotes the null condition. We then optimize the same objective as before, replacing c by c˜. This gives

$$
L _ { \mathrm { C F G } } ( \theta ) = \mathbb { E } _ { t , x _ { 0 } , P _ { t } , \tilde { c } } \left[ - \log \hat { P } _ { \theta } ( t , \tilde { c } , P _ { t } ) _ { x _ { 0 } } \right] .
$$

Guided Plug-In Prediction. At inference time, given $P _ { t }$ and a target condition $c ,$ we compute the unconditional and conditional predictions

$$
\hat { P } _ { 0 } ^ { \mathrm { u } } = \hat { P } _ { \theta } ( t , \infty , P _ { t } ) , \qquad \hat { P } _ { 0 } ^ { \mathrm { c } } = \hat { P } _ { \theta } ( t , c , P _ { t } ) .
$$

For a guidance scale $w \geqslant 0$ , we define the guided predictor $\tilde { P } _ { 0 } \in \Delta _ { N }$ by

$$
\tilde { P } _ { 0 , i } = \frac { ( \hat { P } _ { 0 , i } ^ { \mathrm { u } } ) ^ { 1 - w } ( \hat { P } _ { 0 , i } ^ { \mathrm { c } } ) ^ { w } } { \sum _ { j = 1 } ^ { N } ( \hat { P } _ { 0 , j } ^ { \mathrm { u } } ) ^ { 1 - w } ( \hat { P } _ { 0 , j } ^ { \mathrm { c } } ) ^ { w } } , \qquad i \in \{ 1 , . . . , N \} .
$$

The cases $w = 0 , w = 1$ , and $w > 1$ correspond respectively to unconditional sampling, conditional sampling, and classifier-free guidance.

Guided Reverse Transition. Classifier-free guidance is implemented by replacing the sample $P _ { 0 }$ from $\hat { P } _ { 0 }$ in Proposition 3.3 and Algorithm 2 by one from its guided version $\tilde { P } _ { 0 }$

## D FAST SAMPLING OF DIRICHLET RANDOM VARIABLES

In order to sample from Dirichlet distribution efficiently one leverage the representation of Dirichlet distributions in terms of Gamma distributions. More precisely, for $\bar { Y ^ { \mathrm { ~ \scriptsize ~ \sim ~ } } } \operatorname { D i r } ( \alpha )$ with $\alpha =$ $( \alpha _ { 1 } , \dots , \alpha _ { N } )$ can be efficiently obtained by defining

$$
Y = \left( \frac { X _ { 1 } } { \sum _ { j = 1 } ^ { N } X _ { j } } , \dots , \frac { X _ { N } } { \sum _ { j = 1 } ^ { N } X _ { j } } \right) ,
$$

with $X _ { i } \sim \mathrm { G a m m a } ( \alpha _ { i } , 1 )$ . Hence in order to sample efficiently from a Dirichlet distribution one must efficiently sample from a Gamma distribution. In order to do so, we leverage the Marsaglia– Tsang algorithm, Marsaglia & Tsang (2000); see also Algorithm 3. However, the current implementation of the Gamma sampler in JAX is either slow or approximate Bradbury et al. (2018), see $\mathtt { h t t p s : / / g }$ ithub.com/jax-ml/jax/issues/38141.

In order to generate a Gamma random variable of shape $( d _ { 1 } , \ldots , d _ { n } )$ , the current official implementation splits an original key $\textstyle d = \prod _ { i = 1 } ^ { n } d _ { i }$ times. Then it proceeds in using vmap on those d dimensions. The body of this parallelised function also contains key splitting. This incurs large memory overhead which makes the whole function slow. Instead, we propose an exact implementation which does not rely on vmap and therefore do not require creating many keys which significantly reduce the memory overhead. We reproduce a minimal example of this overhead as well as our solution below and provide some performance benchmark on CPU hardware, see Figure 4. All benchmarks were conducted on an AMD EPYC 7B13 processor (32 physical cores, 2.45GHz base clock, 128MB L3 cache) with 117 GB of DDR4 RAM using JAX and the XLA CPU backend in single precision (float32).

Finally, we highlight that a recent approach by Greaves (2026) show that Gamma distributions can be generated as extended one-liner, thereby resolving a conjecture of Devroye (1996). In practice, the approach leverages a Generalized Acceptance-Complement procedure with a Gaussian proposal and a triangular symmetric complement. In particular, the proposed sampler only requires three draws of independent random variables. As a consequence, this new sampler i) decreases drastically the number of calls to pseudo random number generators ii) bypasses the need of a while loop to sample from the distribution. Early benchmarks of this approach forecast a sampling speed-up of 1.5x over our fast implementation of the Marsaglia–Tsang algorithm.

Algorithm 3 Marsaglia & Tsang Gamma Sampler   
Require: PRNG Key, Shape parameter α ą 0   
Ensure: Sample x „ Gammapα, 1q   
1: 1. Boosting for Small Alpha   
2: if α ă 1 then   
3: $\alpha ^ { \prime } \gets \alpha + 1$ Ź Boost variance for stability   
4: is small Ð True   
5: else   
6: $\alpha ^ { \prime }  \alpha$   
7: is small Ð False   
8: end if   
9: 2. Compute Constants   
10: $d \gets \alpha ^ { \prime } - \frac { 1 } { 3 }$   
11: c Ð 1   
?<sub>9d</sub>   
12: 3. Rejection Sampling Loop   
13: while sample not accepted do   
14: a. Generate Candidates   
15: Sample $z \sim \mathcal { N } ( 0 , 1 )$ Ź Standard Normal   
16: Sample $u \sim \mathcal { U } ( 0 , 1 )$ Ź Uniform   
17: v Ð 1 \` c ¨ z   
18: if v ď 0 then   
19: continue Ź Reject negative support   
20: end if   
21: $v  v ^ { 3 }$   
22: $x _ { \mathrm { p r o p } }  d \cdot v$ Ź Proposed sample   
23: b. Acceptance Checks   
24: Ź Squeeze Test (Fast Check)   
25: if $u < 1 - 0 . 0 3 3 1 \cdot z ^ { 4 }$ then   
26: break   
27: end if   
28: Ź Log-Likelihood Test (Exact Check)   
29: $\mathbf { i f } \log ( u ) \leq 0 . 5 z ^ { 2 } + d ( 1 - v + \log ( v ) )$ then   
30: break   
31: end if   
32: end while   
33: 4. Final Correction   
34: if is small is True then   
35: Sample $u _ { \mathrm { b o o s t } } \sim \mathcal { U } ( 0 , 1 )$   
36: $x  x _ { \mathrm { p r o p } } \cdot u _ { \mathrm { b o o s t } } ^ { 1 / \alpha }$ Ź Apply boosting correction   
37: else   
38: x Ð x<sub>prop</sub>   
39: end if   
40: return x

```python
(a) JAX Native Pattern (vmap over scalar (b) Vectorized Pattern (Counter-based
PRNG) PRNG)
@partial(jax.jit, static_argnames=("N", "d")) @partial(jax.jit, static_argnames=("N", "d"))
2 def jax_native_pattern(key, N: int, d: int): 2 def vectorized_pattern(key, N: int, d: int):
3 keys = jax.random.split(key, N) 3 k1, k2 = jax.random.split(key, 2)
4 def _body(k): 4 z = jax.random.normal(k1, (N, d))
5 k1, k2 = jax.random.split(k, 2) 5 u = jax.random.uniform(k2, (N, d))
6 z = jax.random.normal(k1, (d,)) 6 return z, u
7 u = jax.random.uniform(k2, (d,))
8 return z, u
9 return jax.vmap(_body)(keys)
```  
Figure 4: Comparison of PRNG generation patterns in JAX. Pattern (a) derives per-element keys inside vmap, creating 120 MB of DRAM traffic across Threefry rounds. Pattern (b) splits a single scalar key in CPU registers and generates $N \times d$ values via counter indexing, achieving a 3.8ˆ speedup with $N = 1 0 ^ { \overline { { 6 } } }$ and d “ 1.

## Part II: Theory and Generalizations

## E PROOFS AND ADDITIONAL RESULTS

## E.1 PROOF OF PROPOSITION 3.1

We have that for any $t \in [ 0 , 1 ]$

$$
\begin{array} { r l } & { p _ { t | 0 } ( x _ { t } | P _ { 0 } ) = \displaystyle \int _ { \Delta _ { N } } p _ { t | 0 } ( x _ { t } | P _ { t } , P _ { 0 } ) p _ { t | 0 } ( P _ { t } | P _ { 0 } ) \mathrm { d } { P _ { t } } } \\ & { \qquad \quad = \displaystyle \int _ { \Delta _ { N } } P _ { t , x _ { t } } p _ { t | 0 } ( P _ { t } | P _ { 0 } ) \mathrm { d } { P _ { t } } = \mathbb { E } _ { P _ { t } | P _ { 0 } } [ P _ { t , x _ { t } } ] . } \end{array}
$$

From (13), the mean of the Dirichlet distribution is given for any $\boldsymbol { x } _ { t } \in \mathcal { X }$ by

$$
\mathbb { E } _ { P _ { t } | P _ { 0 } } [ P _ { t , x _ { t } } ] = \alpha _ { t } \delta _ { x _ { 0 } } ( x _ { t } ) + ( 1 - \alpha _ { t } ) \pi _ { x _ { t } } ,
$$

which concludes the proof.

## E.2 PROOF OF PROPOSITION 3.2

Fix $t \in ( 0 , 1 )$ . Recall that $\begin{array} { r } { P _ { 0 } = e _ { x _ { 0 } } , } \end{array}$ , which, under our boundary convention, may be viewed as the degenerate Dirichlet random variable $P _ { 0 } \sim \mathrm { D i r } ( c _ { t } \alpha _ { t } e _ { x _ { 0 } } )$ . Let $W _ { t } \sim$ Beta $( c _ { t } \alpha _ { t } , c _ { t } ( 1 - \alpha _ { t } ) )$ and $\bar { V _ { t } } \tilde { \sim } \operatorname* { D i r } \left( c _ { t } ( 1 - \alpha _ { t } ) \pi \right)$ independently, and define

$$
P _ { t } = W _ { t } P _ { 0 } + ( 1 - W _ { t } ) V _ { t } .
$$

The result is then an immediate application of Proposition A.1 with

$$
\alpha  c _ { t } \alpha _ { t } e _ { x _ { 0 } } , \qquad \beta  c _ { t } ( \alpha _ { t } e _ { x _ { 0 } } + ( 1 - \alpha _ { t } ) \pi ) , \qquad \gamma = \beta - \alpha  c _ { t } ( 1 - \alpha _ { t } ) \pi .
$$

Hence

$$
P _ { t } \sim \mathrm { D i r } \left( c _ { t } \left( \alpha _ { t } P _ { 0 } + ( 1 - \alpha _ { t } ) \pi \right) \right) = \mathrm { D i r } \left( \beta _ { t } ( P _ { 0 } , \pi ) \right) ,
$$

which concludes the proof.

## E.3 PROOF OF PROPOSITION 3.3

Since $\alpha _ { s } \geqslant \alpha _ { t } ,$ the definition of $\boldsymbol { r } _ { s , t }$ ensures that

$$
\gamma _ { s , t } ^ { \kappa } ( P _ { 0 } , \pi ) = \beta _ { s } ( P _ { 0 } , \pi ) - \rho _ { s , t } ^ { \kappa } \beta _ { t } ( P _ { 0 } , \pi ) \geqslant 0
$$

coordinatewise. We write $\rho = \rho _ { s , t } ^ { \kappa }$

We first apply Proposition A.2. Since $\begin{array} { r l r } { P _ { t } } & { { } | } & { P _ { 0 } \mathrm { ~ \ \sim ~ \ } \operatorname { D i r } \big ( \beta _ { t } ( P _ { 0 } , \pi ) \big ) } \end{array}$ and $\begin{array} { r l } { B _ { i } } & { { } \sim } \end{array}$ Beta $( \rho \beta _ { t } ( P _ { 0 } , \pi ) _ { i } , ( 1 - \rho ) \beta _ { t } ( P _ { 0 } , \pi ) _ { i } )$ independently, the vector defined in (6) satisfies

$$
P _ { s , t } ^ { \kappa } \mid P _ { 0 } \sim \mathrm { D i r } \left( \rho \beta _ { t } ( P _ { 0 } , \pi ) \right) .
$$

By construction, $\begin{array} { r l r } { V _ { s , t } ^ { \kappa } } & { { } \sim } & { \mathrm { D i r } \left( \gamma _ { s , t } ^ { \kappa } ( P _ { 0 } , \pi ) \right) } \end{array}$ independently of $P _ { s , t } ^ { \kappa } ,$ and $W _ { s , t } ^ { \kappa } \mathrm { ~ \sim ~ }$ Beta $\begin{array} { r l } { \left. \left( \sum _ { i } \rho \beta _ { t } ( P _ { 0 } , \pi ) _ { i } , \sum _ { i } \gamma _ { s , t } ^ { \kappa } ( P _ { 0 } , \pi ) _ { i } \right) \right. } & { { } } \end{array}$ Since $\begin{array} { r c l } { \sum _ { i } \rho \beta _ { t } ( P _ { 0 } , \pi ) _ { i } } & { = } & { \rho c _ { t } } \end{array}$ and $\begin{array} { r l } { \sum _ { i } \gamma _ { s , t } ^ { \kappa } ( P _ { 0 } , \pi ) _ { i } } & { { } = } \end{array}$ $c _ { s } - \rho c _ { t }$ , this is exactly the distribution of $W _ { s , t } ^ { \kappa } \mathrm { i n } ( 7 )$

We can therefore apply Proposition A.1 with $X \gets P _ { s , t } ^ { \kappa } , \alpha \gets \rho \beta _ { t } ( P _ { 0 } , \pi )$ and $\beta \gets \beta _ { s } ( P _ { 0 } , \pi )$ which gives

$$
P _ { s } = W _ { s , t } ^ { \kappa } P _ { s , t } ^ { \kappa } + ( 1 - W _ { s , t } ^ { \kappa } ) V _ { s , t } ^ { \kappa } \sim \mathrm { D i r } \big ( \beta _ { s } ( P _ { 0 } , \pi ) \big ) .
$$

This is exactly the compatibility condition (5).

## E.4 A REPRESENTATION OF $P _ { t }$

We provide here an explicit representation of $P _ { t }$ under exact DDIM sampling.

Proposition $\mathbf { E . l } ( P _ { t }$ as a random convex combination): Consider $c _ { t } ~ = ~ \varepsilon / ( 1 - \alpha _ { t } )$ Let $\kappa = 0$ and $( t _ { i } ) _ { i = 0 } ^ { M }$ with $M \in \mathbb { N }$ such that $0 = t _ { 0 } < \cdots < t _ { M } = 1$ . Let $P _ { t _ { M } } \sim p _ { t _ { M } }$ For $i = M - 1 , \ldots , 0 ,$ conditionally on $P _ { t _ { i + 1 } }$ , sample $P _ { 0 } ^ { i } \sim p _ { 0 | t _ { i + 1 } } ( \cdot \mid P _ { t _ { i + 1 } } )$ , write $P _ { 0 } ^ { i } = e _ { x _ { 0 } ^ { i } }$ , and then sample P<sub>t</sub> from $p _ { t _ { i } | 0 , t _ { i + 1 } } ( \cdot \mid P _ { 0 } ^ { i } , P _ { t _ { i + 1 } } )$ , equivalently using (9). For any $i \in \{ 0 , \ldots , M \}$ we have $P _ { t _ { i } } \sim p _ { t _ { i } }$ and

$$
P _ { t _ { i } } = \sum _ { j = i } ^ { M - 1 } L _ { i , j } e _ { x _ { 0 } ^ { j } } + L _ { i , M } P _ { t _ { M } } ,\tag{14}
$$

$$
w i t h L _ { i } = ( L _ { i , j } ) _ { j = i } ^ { M } a n d L _ { i } \sim \mathrm { D i r } \bigl ( c _ { t _ { i } } - c _ { t _ { i + 1 } } , \dots , c _ { t _ { M - 1 } } - c _ { t _ { M } } , c _ { t _ { M } } \bigr ) .
$$

Equation (14) shows that $P _ { t _ { i } }$ is a random convex combination of the successive posterior cleanstate draws $( x _ { 0 } ^ { j } ) _ { i = i } ^ { M - 1 }$ and the initial noise $P _ { t _ { M } }$ . For the plug-in transitions, this exact representation need not hold, but it motivates the interpretation of $P _ { t }$ as a belief-memory state. In Section E.5, we investigate the high and low temperature limits of this representation.

Proof. We first verify the marginal claim. It holds at $i \ = \ M$ by assumption. If $P _ { t _ { i + 1 } } \sim p _ { t _ { i + 1 } }$ and $P _ { 0 } ^ { i } \sim p _ { 0 | t _ { i + 1 } } ( \cdot \mid P _ { t _ { i + 1 } } )$ , then $( P _ { 0 } ^ { i } , P _ { t _ { i + 1 } } )$ has joint law $p ( P _ { 0 } ) p _ { t _ { i + 1 } | 0 } ( P _ { t _ { i + 1 } } \mid P _ { 0 } )$ . Integrating the exact kernel $p _ { t _ { i } | 0 , t _ { i + 1 } }$ and using the compatibility condition (5) gives $P _ { t _ { i } } \sim p _ { t _ { i } }$ . Backward induction therefore proves $P _ { t _ { i } } \sim p _ { t _ { i } }$ for every i.

The proof relies on the stick-breaking property of the Dirichlet distribution (a direct consequence of Proposition A.1). If a random vector $X \ \stackrel { \cdot } { \sim } \ \operatorname { D i r } ( \alpha _ { 1 } , \ldots , \alpha _ { K } )$ and a random variable $\hat { W } \sim$ $\begin{array} { r } { \mathrm { B e t a } ( \sum _ { k = 1 } ^ { K } \alpha _ { k } , \alpha _ { 0 } ) } \end{array}$ are independent, then the augmented vector $( 1 - W , W X _ { 1 } , \ldots , W X _ { K } )$ is distributed as Dir $( \alpha _ { 0 } , \alpha _ { 1 } , \ldots , \alpha _ { K } )$ . We proceed by backward induction on i, from $i = M - 1$ down to 0. To simplify notation, we define the parameter sequence $\begin{array} { r } { \gamma _ { i } ~ = ~ \frac { \varepsilon } { 1 - \alpha _ { t _ { i } } } - \frac { \varepsilon } { 1 - \alpha _ { t _ { i + 1 } } } } \end{array}$ for $i \in \left\{ 0 , \ldots , M - 1 \right\}$ , and $\begin{array} { r } { \gamma _ { M } = \frac { \varepsilon } { 1 - \alpha _ { t _ { M } } } } \end{array}$ . We start with $i = M - 1 . \mathrm { { B y } } \left( 9 \right)$ , we have:

$$
P _ { t _ { M - 1 } } = W _ { t _ { M - 1 } , t _ { M } } ^ { 0 } P _ { t _ { M } } + ( 1 - W _ { t _ { M - 1 } , t _ { M } } ^ { 0 } ) e _ { x _ { 0 } ^ { M - 1 } } ,
$$

where $W _ { t _ { M - 1 } , t _ { M } } ^ { 0 } \sim \mathrm { B e t a } \left( \gamma _ { M } , \gamma _ { M - 1 } \right)$ . The vector $( 1 - W _ { t _ { M - 1 } , t _ { M } } ^ { 0 } , W _ { t _ { M - 1 } , t _ { M } } ^ { 0 } )$ follows a Dirichlet distribution with parameters $\left( \gamma _ { M - 1 } , \gamma _ { M } \right)$ . Setting $L _ { M - 1 , M - 1 } = 1 - W _ { t _ { M - 1 } , t _ { M } } ^ { 0 }$ and $L _ { M - 1 , M } =$ $W _ { t _ { M - 1 } , t _ { M } } ^ { 0 }$ satisfies the proposition for the base case. Now assume the proposition holds for some $i + 1 \leqslant M - 1 ; { \mathrm { i . e . } }$

$$
P _ { t _ { i + 1 } } = \sum _ { j = i + 1 } ^ { M - 1 } L _ { i + 1 , j } e _ { x _ { 0 } ^ { j } } + L _ { i + 1 , M } P _ { t _ { M } } ,
$$

where $L _ { i + 1 } ^ { \varepsilon } \sim \operatorname { D i r } ( \gamma _ { i + 1 } , \dots , \gamma _ { M } )$ . Note that the sum of these concentration parameters is exactly $\begin{array} { r } { \sum _ { k = i + 1 } ^ { M } \gamma _ { k } = \frac { \varepsilon } { 1 - \alpha _ { t _ { i + 1 } } } } \end{array}$ . So using Proposition 3.3 and applying the backward transition to step i, we

![](images/3c51c5f57df11f7ae2da770810d8c5b60965434ac30dbf981cecaa891b818d7d.jpg)  
Figure 5: Visualization of the Self-Conditioning weights $L _ { i , j }$ from Proposition E.1. Each panel shows the exact mean profile $\mathbb { E } [ L _ { i , j } ]$ (solid line) together with a $\pm 1$ standard-deviation band for the Dirichlet law in Proposition E.1, for $\varepsilon \in \{ 0 . 0 1 , 1 , 1 0 0 \}$ and $i \in \{ 0 , 5 , 1 0 , 1 5 \}$ . The mean curve is independent of ε.

have:

$$
P _ { t _ { i } } = W _ { t _ { i } , t _ { i + 1 } } ^ { 0 } P _ { t _ { i + 1 } } + ( 1 - W _ { t _ { i } , t _ { i + 1 } } ^ { 0 } ) e _ { x _ { 0 } ^ { i } } ,
$$

where $\begin{array} { r } { W _ { t _ { i } , t _ { i + 1 } } ^ { 0 } \sim \mathrm { B e t a } \left( \frac { \varepsilon } { 1 - \alpha _ { t _ { i + 1 } } } , \gamma _ { i } \right) = \mathrm { B e t a } \left( \sum _ { k = i + 1 } ^ { M } \gamma _ { k } , \gamma _ { i } \right) } \end{array}$ . Therefore, we get

$$
P _ { t _ { i } } = ( 1 - W _ { t _ { i } , t _ { i + 1 } } ^ { 0 } ) e _ { x _ { 0 } ^ { i } } + \sum _ { j = i + 1 } ^ { M - 1 } \left( W _ { t _ { i } , t _ { i + 1 } } ^ { 0 } L _ { i + 1 , j } \right) e _ { x _ { 0 } ^ { j } } + \left( W _ { t _ { i } , t _ { i + 1 } } ^ { 0 } L _ { i + 1 , M } \right) P _ { t _ { M } } .
$$

We identify the new weights $L _ { i }$ as follows: $L _ { i , i } = 1 - W _ { t _ { i } , t _ { i + 1 } } ^ { 0 }$ and $L _ { i , j } = W _ { t _ { i } , t _ { i + 1 } } ^ { 0 } L _ { i + 1 , j }$ for $j \in$ $\{ i + 1 , \ldots , M \}$ . Because $\begin{array} { r } { L _ { i + 1 } \ \sim \ \mathrm { D i r } ( \gamma _ { i + 1 } , \dots , \gamma _ { M } ) } \end{array}$ and $W _ { t _ { i } , t _ { i + 1 } } ^ { 0 } \ \sim$ Beta $\scriptstyle \left( \sum _ { k = i + 1 } ^ { M } \gamma _ { k } , \gamma _ { i } \right)$ applying the stick-breaking property guarantees that the augmented vector $L _ { i }$ is distributed as $\operatorname { D i r } ( \gamma _ { i } , \gamma _ { i + 1 } , \dots , \gamma _ { M } )$ . This concludes the inductive step and the proof. □

## E.5 ILLUSTRATION AND TEMPERATURE LIMITS FOR PROPOSITION E.1

We first present in Figure 5 the Self-Conditioning weights $L _ { i , j } ^ { \varepsilon }$ appearing in Proposition E.1 for various ε and index i.

Let us now investigate the high and low temperature limits of Proposition E.1.

Corollary E.1 (Limiting behavior of intrinsic Self-Conditioning): Let ${ \cal L } _ { i } \ : = \ : ( L _ { i , j } ) _ { j = i } ^ { M }$ be the Dirichlet-distributed weight vector defined in Proposition $E . l .$ We define the unscaled parameters $\begin{array} { r } { c _ { j } = \frac { 1 } { 1 - \alpha _ { t _ { j } } } - \frac { \smile } { 1 - \alpha _ { t _ { j + 1 } } } f o r { j ^ { ' } } \in \{ i , \dots , \overset {  } { M } - 1 \} } \end{array}$ and $\begin{array} { r } { { \displaystyle c _ { M } = \frac { 1 } { 1 - \alpha _ { t _ { M } } } } } \end{array}$ Let $C _ { i } \ =$ $\begin{array} { r } { \sum _ { j = i } ^ { M } c _ { j } = \frac { 1 } { 1 - \alpha _ { t _ { i } } } } \end{array}$ . We have the following limiting cases.

1. Low-temperature limit (Deterministicflow): $A s \varepsilon \to \infty , L _ { i }$ converges in distribution to a deterministic vector:

$$
L _ { i } \stackrel { d } {  } \bar { L } _ { i } , ~ w h e r e ~ \bar { L } _ { i , j } = \frac { c _ { j } } { C _ { i } } .
$$

2. High-temperature limit (Discretejumps): As $\varepsilon \to 0 , L _ { i }$ converges in distribution to a categorical distribution over the canonical basis $( u _ { j } ) _ { j = \ l } ^ { M }$ ofthe simplex over indices $j \in \{ i , . . . , M \}$

$$
L _ { i } \stackrel { d } { \to } u _ { J _ { i } } , \qquad w h e r e \quad J _ { i } \sim \mathrm { C a t } \left( \left( \frac { c _ { j } } { C _ { i } } \right) _ { j = i } ^ { M } \right) .
$$

Therefore, we recover the Discrete Diffusion model behavior when $\varepsilon \to 0$ as in that case only one component $j ^ { \star } \in \{ i , \ldots , M \}$ is non-zero. This means that all contributions at steps other than $t _ { j } ,$ are neglected. This is not the case when $\varepsilon > 0$ and in the limit $\varepsilon  + \infty$ , the mixing of the clean-state draws is deterministic.

Proof. Recall that Proposition E.1 assumes the schedule $c _ { t } = \varepsilon / ( 1 - \alpha _ { t } )$ , and gives

$$
L _ { i } \sim \mathrm { D i r } \left( c _ { t _ { i } } - c _ { t _ { i + 1 } } , \ldots , c _ { t _ { M - 1 } } - c _ { t _ { M } } , c _ { t _ { M } } \right) .
$$

In terms of the unscaled parameters $\tilde { c } _ { j }$ of the statement, namely $\begin{array} { r } { \tilde { c } _ { j } = \frac { 1 } { 1 - \alpha _ { t _ { j } } } - \frac { 1 } { 1 - \alpha _ { t _ { j + 1 } } } } \end{array}$ for $j \in$ $\{ i , \ldots , M - 1 \}$ and $\begin{array} { r } { \tilde { c } _ { M } = \frac { 1 } { 1 - \alpha _ { t _ { M } } } } \end{array}$ , we have

$$
c _ { t _ { j } } - c _ { t _ { j + 1 } } = \varepsilon \tilde { c } _ { j } \quad ( j \leqslant M - 1 ) , \qquad c _ { t _ { M } } = \varepsilon \tilde { c } _ { M } ,
$$

so that $L _ { i } \sim \mathrm { D i r } ( \varepsilon \tilde { c } _ { i } , \dots , \varepsilon \tilde { c } _ { M } )$ . Since $u \mapsto \alpha _ { u }$ is non-increasing, each $\tilde { c } _ { j } \ \geqslant \ 0 ,$ , and the total concentration telescopes:

$$
{ \sum } _ { j = i } ^ { M } \varepsilon \tilde { c } _ { j } = \varepsilon C _ { i } = \frac { \varepsilon } { 1 - \alpha _ { t _ { i } } } = c _ { t _ { i } } \in ( 0 , \infty ) \qquad \mathrm { f o r ~ } i \geqslant 1 .
$$

Writing $\bar { L } _ { i } = \left( \tilde { c } _ { j } / C _ { i } \right) _ { i = i } ^ { M } \in \Delta _ { M - i + 1 }$ , which does not depend on ε, this reads

$$
L _ { i } \sim \mathrm { D i r } \left( \left( \varepsilon C _ { i } \right) \bar { L } _ { i } \right) .
$$

So the following results follow.

(1) Low-temperature limit. $\mathbf { A s } \ \varepsilon \to \infty$ we have $\varepsilon C _ { i } \to \infty$ , so $L _ { i } \to \bar { L } _ { i }$ in probability, hence in distribution. Explicitly, $\mathbb { E } [ L _ { i , j } ] = \tilde { c } _ { j } / C _ { i }$ for every ε, while

$$
\operatorname { V a r } [ L _ { i , j } ] = \frac { ( \tilde { c } _ { j } / C _ { i } ) ( 1 - \tilde { c } _ { j } / C _ { i } ) } { \varepsilon C _ { i } + 1 } \xrightarrow [ \varepsilon  \infty ] { } 0 .
$$

(2) High-temperature limit. As $\varepsilon \to 0$ we have $\varepsilon C _ { i } \to 0$ , so we obtain

$$
\begin{array} { r } { L _ { i } \stackrel { d } { \longrightarrow } \sum _ { j = i } ^ { M } \frac { \tilde { c } _ { j } } { C _ { i } } \delta _ { u _ { j } } , } \end{array}
$$

where $( u _ { j } ) _ { j = i } ^ { M }$ denotes the canonical basis of $\mathbb { R } ^ { M - i + 1 }$ . Equivalently, $L _ { i } ~ \stackrel { d } { \to } ~ u _ { J _ { i } }$ with $J _ { i } \ \sim$ Cat $\left( ( \tilde { c } _ { j } / C _ { i } ) _ { j = i } ^ { M } \right)$ . Indices $j$ with $\tilde { c } _ { j } = 0 ,$ , which occur when α is constant on $[ t _ { j } , t _ { j + 1 } ]$ , satisfy $L _ { i , j } \doteq 0$ almost surely and receive zero mass in the limit, consistently with the statement. □

![](images/e9f181173ea243dee71935f870e84fd5ef8cac09175408bc2b1c71b78989dcff.jpg)  
t = 0.05  
t = 0.25  
t = 0.5  
t = 0.75  
t = 0.99  
Figure 6: Each figure corresponds to the density of $p _ { t | 0 } ( P _ { t } | P _ { 0 } ) = \mathrm { D i r } ( c _ { t } ( \alpha _ { t } P _ { 0 } + ( 1 - \alpha _ { t } ) \pi ) )$ , with π the uniform distribution and $c _ { t } = \varepsilon / ( 1 - \alpha _ { t } )$ for different values of $t \in [ 0 , 1 ]$ and $\varepsilon > 0$ . We also display the mean (red star) of $p _ { t | 0 } , \mathrm { i . e . , \mathbb { E } } [ P _ { t } | P _ { 0 } ] = \alpha _ { t } P _ { 0 } + ( 1 - \alpha _ { t } )$ π and observe that this quantity is independent of ε. Finally, we also display 5 samples (white dots) of $p _ { t | 0 }$ . Note that as $\varepsilon \to 0$ the samples concentrate on the vertices of the simplex, thereby illustrating the convergence of SDMs to their Discrete Diffusion counterpart.

## E.6 PROOF OF PROPOSITION 4.1

In this section, we investigate the temperature limits of SDMs. The complete design space is summarized in Figure 7. We recall that we set $c _ { t } = \varepsilon / ( 1 - \alpha _ { t } )$

We use the representation from Proposition 3.2,

$$
P _ { t } = W _ { t } P _ { 0 } + \left( 1 - W _ { t } \right) V _ { t } ,
$$

where

$$
W _ { t } \sim \mathrm { B e t a } ( \varepsilon h _ { t } , \varepsilon ) , \qquad V _ { t } \sim \mathrm { D i r } ( \varepsilon \pi ) , \qquad h _ { t } = \frac { \alpha _ { t } } { 1 - \alpha _ { t } } ,
$$

and $W _ { t }$ and $V _ { t }$ are independent.

High-Temperature Limit. For fixed $t \in ( 0 , 1 )$ , standard results for the Beta and Dirichlet distributions give

$$
W _ { t } \stackrel { d } { \longrightarrow } B _ { t } , \qquad B _ { t } \sim \mathrm { B e r } ( \alpha _ { t } ) ,
$$

![](images/319b764f4277e66fcca24b4ca6013343ec9c8b7fc1eeef3c9a49c82a11d6f013.jpg)  
Figure 7: The design space of Simplex Diffusion Models (SDMs). The forward process is controlled by the inverse temperature ε: as $\varepsilon  0 .$ , the simplex mass concentrates on vertices and recovers discrete-diffusion behavior; as $\varepsilon  \infty ,$ the forward process concentrates around the deterministic interpolation $\bar { P } _ { t } = \alpha _ { t } P _ { 0 } + ( 1 - \alpha _ { t } ) \tau$ π. The reverse process is controlled by the churn parameter κ: $\kappa = 0$ fully trusts the current belief $P _ { t }$ , while $\kappa = 1$ yields the independent bridge $p _ { s | 0 , t } =$ $p _ { s | 0 }$ . SDMs operate in the intermediate regime, maintaining a continuous belief state throughout sampling.

and

$$
V _ { t } \stackrel { d } { \longrightarrow } V , \qquad V \sim \sum _ { i = 1 } ^ { N } \pi _ { i } \delta _ { e _ { i } } .
$$

The limiting variables are independent. Hence, by the continuous mapping theorem,

$$
P _ { t } \xrightarrow { d } B _ { t } P _ { 0 } + ( 1 - B _ { t } ) V .
$$

Since $P _ { 0 } = e _ { x _ { 0 } }$ and $V$ is supported on the canonical basis vectors, the limiting law is supported on the vertices of $\Delta _ { N }$

For $\kappa = 0$ , the backward transition is

$$
P _ { s } = W _ { s , t } ^ { 0 } P _ { t } + \left( 1 - W _ { s , t } ^ { 0 } \right) e _ { x _ { 0 } } ,
$$

with

$$
W _ { s , t } ^ { 0 } \sim \mathrm { B e t a } \Bigg ( \frac { \varepsilon } { 1 - \alpha _ { t } } , \frac { \varepsilon } { 1 - \alpha _ { s } } - \frac { \varepsilon } { 1 - \alpha _ { t } } \Bigg ) .
$$

Applying again the small-concentration Beta limit gives

$$
W _ { s , t } ^ { 0 } \stackrel { d } { \longrightarrow } \mathrm { B e r } \left( \frac { 1 - \alpha _ { s } } { 1 - \alpha _ { t } } \right) .
$$

Therefore, for every fixed $P _ { t } \in \Delta _ { N }$ , we have in the limit

$$
P _ { s } \stackrel { d } { \longrightarrow } W _ { s , t } ^ { 0 } P _ { t } + \left( 1 - W _ { s , t } ^ { 0 } \right) e _ { x _ { 0 } } , \qquad W _ { s , t } ^ { 0 } \sim \mathrm { B e r } \left( \frac { 1 - \alpha _ { s } } { 1 - \alpha _ { t } } \right) .
$$

In particular, when $P _ { t }$ is a simplex vertex, the limiting transition is supported on simplex vertices.

Low-Temperature Limit. For fixed $t \in ( 0 , 1 )$ , standard results give

$$
W _ { t } \stackrel { \mathbb { P } } { \longrightarrow } \frac { h _ { t } } { 1 + h _ { t } } = \alpha _ { t } ,
$$

and

$$
V _ { t } \xrightarrow { \mathbb { P } } \pi .
$$

Using once again the representation of $P _ { t }$ and the continuous mapping theorem,

$$
P _ { t } \stackrel { \mathbb { P } } { \longrightarrow } \alpha _ { t } P _ { 0 } + ( 1 - \alpha _ { t } ) \pi ,
$$

which concludes the proof.

## E.7 GAUSSIAN FLUCTUATIONS

The next result identifies the fluctuations around the deterministic path at scale $\varepsilon ^ { - 1 / 2 }$ . For this result, we introduce the tangent space of the simplex $\mathrm { T } \Delta _ { N } = \{ x \in \mathbb { R } ^ { N } , \mathbf { \dot { \Omega } } ^ { \top } x = 0 \}$

Proposition E.2 (Gaussian fluctuation limit): Let $U \in \mathbb { R } ^ { N \times ( N - 1 ) }$ be an orthonormal basis for the tangent space $\mathrm { T } \Delta _ { N }$ . Let $t \in ( 0 , 1 ]$ and let $P _ { t } \sim \mathrm { D i r } ( \beta _ { t } ( P _ { 0 } , \pi ) )$ with $\pi _ { i } > 0$ for all i. Let $X _ { 0 } = \overset { \vartriangle } { \sqrt { \varepsilon } } U ^ { \dag } ( P _ { 0 } - \pi )$ and $X _ { t } = \dot { \sqrt { \varepsilon } } \bar { U } ^ { \top } ( P _ { t } - \pi )$ . We have

$$
X _ { t } = \alpha _ { t } X _ { 0 } + ( 1 - \alpha _ { t } ) \xi _ { t } ^ { \varepsilon } ,
$$

where $\begin{array} { r l r } { \xi _ { t } ^ { \varepsilon } } & { { } \quad } & { = \quad \quad \frac { \sqrt { \varepsilon } } { 1 - \alpha _ { t } } U ^ { \top } ( P _ { t } \quad - \quad \bar { P } _ { t } ) . } \end{array}$

$$
U ^ { \top } [ \mathrm { d i a g } ( \pi ) - \pi \pi ^ { \top } + \alpha _ { t } ( P _ { 0 } - \pi ) ( P _ { 0 } - \pi ) ^ { \top } ] U . \ A s \varepsilon  \infty ,
$$

$$
\xi _ { t } ^ { \varepsilon } \overset { d } { \to } \mathcal { N } \left( 0 , \Sigma _ { t } \right) .
$$

In addition, the spectrum $o f \Sigma _ { t }$ is contained in rmin<sub>i</sub> $\pi _ { i } , 2 ]$ for all $t \in ( 0 , 1 ] .$

To connect with standard continuous Gaussian diffusion models, we project the scaled displacement of the forward state from the prior π onto an orthonormal basis of the simplex tangent space.

Proposition E.3 (Asymptotic normality and covariance structure): Assume that π has strictly positive components. Let $t \in \mathsf { \Gamma } ( 0 , 1 ]$ and let $P _ { t _ { - } } \sim \mathrm { D i r } ( \beta _ { t } ( P _ { 0 } , \pi ) )$ . Define $\begin{array} { r l } { \bar { P } _ { t } } & { { } = } \end{array}$ $\alpha _ { t } P _ { 0 } + ( 1 - \alpha _ { t } ) \cdot$ π and $V _ { t } = ( 1 - \alpha _ { t } ) ^ { - 1 } \big ( \mathrm { d i a g } \big ( \bar { P } _ { t } \big ) - \bar { P } _ { t } \bar { P } _ { t } ^ { \top } \big ) . A s \varepsilon  \infty ,$ , we have

$$
\sqrt { \varepsilon } ( P _ { t } - \bar { P } _ { t } ) \stackrel { d } { \to } { \mathcal { N } } \left( 0 , ( 1 - \alpha _ { t } ) ^ { 2 } V _ { t } \right) .
$$

In addition, letting $V _ { \pi } = \mathrm { d i a g } ( \pi ) - \pi \pi ^ { \top }$ , we have

$$
V _ { t } = V _ { \pi } + \alpha _ { t } ( P _ { 0 } - \pi ) ( P _ { 0 } - \pi ) ^ { \top } .
$$

Proof. For the rest of the proof, we define $\begin{array} { r } { \tilde { V } _ { t } = \mathrm { d i a g } ( \bar { P } _ { t } ) - \bar { P } _ { t } \bar { P } _ { t } ^ { \top } } \end{array}$ . Hence, we have that $\tilde { V } _ { t } = ( 1 -$ $\alpha _ { t } ) V _ { t }$ . We leverage the Gamma representation of the Dirichlet distribution. Let $\mu _ { t } = \pi + h _ { t } P _ { 0 }$ with $\begin{array} { r } { h _ { t } = \frac { \alpha _ { t } } { 1 - \alpha _ { t } } } \end{array}$ , such that the concentration parameters are $\beta _ { t } ^ { \varepsilon } = \varepsilon \mu _ { t }$ . We can write $P _ { t } \stackrel { d } { = } Y ^ { \varepsilon } / ( \mathbf { 1 } ^ { \top } Y ^ { \varepsilon } )$ where $Y ^ { \varepsilon }$ is a vector of independent random variables $Y _ { i } ^ { \varepsilon } \sim \mathrm { G a m m a } ( \varepsilon \mu _ { t , i } , 1 )$

Using the Central Limit Theorem, as $\varepsilon \to \infty$ , we have

$$
\sqrt { \varepsilon } \left( \frac { Y ^ { \varepsilon } } { \varepsilon } - \mu _ { t } \right) \overset { d } { \to } \mathcal { N } \left( 0 , \mathrm { d i a g } ( \mu _ { t } ) \right) .
$$

We apply the multivariate Delta method with $\begin{array} { r } { g ( x ) = \frac { x } { \mathbf { 1 } ^ { \top } x } } \end{array}$ . Let $\begin{array} { r } { c _ { t } = \mathbf { 1 } ^ { \top } \boldsymbol { \mu } _ { t } = 1 + h _ { t } = \frac { 1 } { 1 - \alpha _ { t } } } \end{array}$ . Notice that $\begin{array} { r } { g ( \mu _ { t } ) = \frac { \mu _ { t } } { c _ { t } } = \bar { P } _ { t } } \end{array}$ . The Jacobian of $g$ at $\mu _ { t }$ is given by

$$
\nabla g ( \mu _ { t } ) = { \frac { 1 } { { \bf 1 } ^ { \top } \mu _ { t } } } \mathrm { I d } - { \frac { \mu _ { t } { \bf 1 } ^ { \top } } { ( { \bf 1 } ^ { \top } \mu _ { t } ) ^ { 2 } } } = { \frac { 1 } { c _ { t } } } \left( \mathrm { I d } - { \bar { P } } _ { t } { \bf 1 } ^ { \top } \right) .
$$

The asymptotic covariance is $\nabla g ( \mu _ { t } ) \mathrm { d i a g } ( \mu _ { t } ) \nabla g ( \mu _ { t } ) ^ { \top }$ which can be simplified in

$$
\begin{array} { r l } & { \frac { 1 } { c _ { t } ^ { 2 } } \left( I - \bar { P } _ { t } \mathbf { 1 } ^ { \top } \right) \mathrm { d i a g } ( \mu _ { t } ) \left( I - \mathbf { 1 } \bar { P } _ { t } ^ { \top } \right) = \displaystyle \frac { 1 } { c _ { t } } \left( I - \bar { P } _ { t } \mathbf { 1 } ^ { \top } \right) \mathrm { d i a g } ( \bar { P } _ { t } ) \left( I - \mathbf { 1 } \bar { P } _ { t } ^ { \top } \right) } \\ & { \phantom { \frac { 1 } { c _ { t } ^ { 2 } } } = \displaystyle \frac { 1 } { c _ { t } } \left( \mathrm { d i a g } ( \bar { P } _ { t } ) - \bar { P } _ { t } \bar { P } _ { t } ^ { \top } - \bar { P } _ { t } \bar { P } _ { t } ^ { \top } + \bar { P } _ { t } ( \mathbf { 1 } ^ { \top } \bar { P } _ { t } ) \bar { P } _ { t } ^ { \top } \right) } \\ & { \phantom { \frac { 1 } { c _ { t } ^ { 2 } } } = \displaystyle \frac { 1 } { c _ { t } } ( \mathrm { d i a g } ( \bar { P } _ { t } ) - \bar { P } _ { t } \bar { P } _ { t } ^ { \top } ) } \\ & { = ( 1 - \alpha _ { t } ) \tilde { V } _ { t } = ( 1 - \alpha _ { t } ) ^ { 2 } V _ { t } . } \end{array}
$$

To obtain the second part of the proposition, we expand $\tilde { V } _ { t } = \mathrm { d i a g } ( \bar { P } _ { t } ) - \bar { P } _ { t } \bar { P } _ { t } ^ { \top }$ using $\bar { P } _ { t } = \alpha _ { t } P _ { 0 } +$ $( 1 - \alpha _ { t } ) \pi$ . Since diag $( P _ { 0 } ) = \bar { P _ { 0 } } P _ { 0 } ^ { \bar { 7 } }$ , because $\begin{array} { r } { P _ { 0 } = e _ { x _ { 0 } } , } \end{array}$ , we have

$$
\begin{array} { r l } & { \tilde { V } _ { t } = \alpha _ { t } \mathrm { d i a g } ( P _ { 0 } ) + ( 1 - \alpha _ { t } ) \mathrm { d i a g } ( \pi ) - \big [ \alpha _ { t } ^ { 2 } P _ { 0 } P _ { 0 } ^ { \top } + \alpha _ { t } ( 1 - \alpha _ { t } ) ( P _ { 0 } \pi ^ { \top } + \pi P _ { 0 } ^ { \top } ) + ( 1 - \alpha _ { t } ) ^ { 2 } \pi \pi ^ { \top } \big ] } \\ & { \quad = ( \alpha _ { t } - \alpha _ { t } ^ { 2 } ) P _ { 0 } P _ { 0 } ^ { \top } + ( 1 - \alpha _ { t } ) \mathrm { d i a g } ( \pi ) - \alpha _ { t } ( 1 - \alpha _ { t } ) \big ( P _ { 0 } \pi ^ { \top } + \pi P _ { 0 } ^ { \top } \big ) - ( 1 - \alpha _ { t } ) ^ { 2 } \pi \pi ^ { \top } } \\ & { \quad = ( 1 - \alpha _ { t } ) \big [ \mathrm { d i a g } ( \pi ) - \pi \pi ^ { \top } + \alpha _ { t } P _ { 0 } P _ { 0 } ^ { \top } - \alpha _ { t } ( P _ { 0 } \pi ^ { \top } + \pi P _ { 0 } ^ { \top } ) + \alpha _ { t } \pi \pi ^ { \top } \big ] } \\ & { \quad = ( 1 - \alpha _ { t } ) \big [ V _ { \pi } + \alpha _ { t } ( P _ { 0 } - \pi ) ( P _ { 0 } - \pi ) ^ { \top } \big ] , } \end{array}
$$

which concludes the proof since $\tilde { V } _ { t } = ( 1 - \alpha _ { t } ) V _ { t }$

Proposition E.4 (Spectral stability of the projected covariance): Let $U \in \mathbb { R } ^ { N \times ( N - 1 ) }$ be an orthonormal basis for the tangent space $\mathrm { T } \Delta _ { N } . ~ H \pi$ has strictly positive components, the spectrum of $U ^ { \top } V _ { t } U$ is uniformly bounded for all $t \in ( 0 , 1 ]$

$$
0 < \operatorname* { m i n } _ { i } \pi _ { i } \leqslant \lambda _ { \operatorname* { m i n } } ( U ^ { \top } V _ { t } U ) \leqslant \lambda _ { \operatorname* { m a x } } ( U ^ { \top } V _ { t } U ) \leqslant 2 .
$$

Proof. Let $v \in \mathbb { R } ^ { N - 1 }$ such that $\| \boldsymbol { v } \| _ { 2 } ~ = ~ 1$ and $u \ : = \ : U \ i$ . It follows that $\| u \| _ { 2 } ~ = ~ \| v \| _ { 2 } ~ = ~ 1$ and $u ^ { \top } { \bf 1 } = 0$ . Using the decomposition of $V _ { t }$ from Proposition E.3, we get

$$
\boldsymbol { v } ^ { \top } ( \boldsymbol { U } ^ { \top } V _ { t } \boldsymbol { U } ) \boldsymbol { v } = \boldsymbol { u } ^ { \top } V _ { \pi } \boldsymbol { u } + \alpha _ { t } ( \boldsymbol { u } ^ { \top } ( P _ { 0 } - \pi ) ) ^ { 2 } .\tag{15}
$$

We analyze the two terms in (15) separately. First, we have

$$
\begin{array} { l } { { \displaystyle u ^ { \top } V _ { \pi } u = \sum _ { i = 1 } ^ { N } \pi _ { i } u _ { i } ^ { 2 } - \left( \sum _ { i } \pi _ { i } u _ { i } \right) ^ { 2 } } } \\ { { \displaystyle \qquad = \sum _ { i = 1 } ^ { N } \pi _ { i } ( u _ { i } - \pi ^ { \top } u ) ^ { 2 } } } \\ { { \displaystyle \qquad \geqslant \operatorname* { m i n } _ { i } \pi _ { i } \left( 1 + N ( \pi ^ { \top } u ) ^ { 2 } \right) \geqslant \operatorname* { m i n } _ { i } \pi _ { i } } , } \end{array}
$$

where we have used that $u ^ { \top } u = 1$ and $u ^ { \top } { \bf 1 } = 0$ . Combining this result and (15), we get that $\begin{array} { r } { \lambda _ { \operatorname* { m i n } } ( U ^ { \top } V _ { t } U ) \gtrsim \operatorname* { m i n } _ { i } \pi _ { i } > 0 . } \end{array}$

For the upper-bound, we use that

$$
\lambda _ { \operatorname* { m a x } } ( U ^ { \top } V _ { t } U ) \leqslant \operatorname { T r } ( U ^ { \top } V _ { t } U ) \leqslant \operatorname { T r } ( V _ { t } ) .\tag{16}
$$

Using that $P _ { 0 } = e _ { x _ { 0 } }$ , we have that

$$
\mathrm { T r } ( V _ { \pi } ) = 1 - \Vert \pi \Vert ^ { 2 } , \qquad \mathrm { T r } ( ( P _ { 0 } - \pi ) ( P _ { 0 } - \pi ) ^ { \top } ) = \Vert P _ { 0 } - \pi \Vert ^ { 2 } = 1 - 2 \pi ^ { \top } P _ { 0 } + \Vert \pi \Vert ^ { 2 } .
$$

Therefore, combining this result, $\alpha _ { t } \leqslant 1$ and (16) we get that

$$
\begin{array} { r } { \lambda _ { \operatorname* { m a x } } ( U ^ { \top } V _ { t } U ) \leqslant 1 - \| \pi \| ^ { 2 } + 1 - 2 \pi ^ { \top } P _ { 0 } + \| \pi \| ^ { 2 } \leqslant 2 , } \end{array}
$$

which concludes the proof.

Finally, we conclude this section with the proof of Proposition E.2.

Proof. Scaling the perturbation $P _ { t } - \pi = \alpha _ { t } ( P _ { 0 } - \pi ) + ( P _ { t } - \bar { P } _ { t } )$ by $\sqrt { \varepsilon } U ^ { \top }$ yields the linear form. By Proposition E.3, $\sqrt { \varepsilon } ( P _ { t } - \bar { P } _ { t } ) \stackrel { d } { \to } \mathcal { N } \left( 0 , ( 1 - \alpha _ { t } ) ^ { 2 } V _ { t } \right)$ . Applying the linear transformation $\frac { 1 } { \left( 1 - \alpha _ { t } \right) } U ^ { \top }$ directly yields

$$
\xi _ { t } ^ { \varepsilon } \overset { d } {  } \mathcal { N } ( 0 , \frac { 1 } { ( 1 - \alpha _ { t } ) ^ { 2 } } U ^ { \top } ( 1 - \alpha _ { t } ) ^ { 2 } V _ { t } U ) = \mathcal { N } ( 0 , U ^ { \top } V _ { t } U ) ,
$$

which concludes the proof. The second part of the proof is a direct consequence of Proposition E.4.

## F VARIATIONAL LOWER BOUND

We derive a variational lower bound for the inference procedure described in Section 3.2.

Fix a grid $0 = t _ { 0 } < t _ { 1 } < \cdots < t _ { M } = 1$ and write $P _ { k } = P _ { t _ { k } }$ . For $j \in \mathcal { X }$ and $k = 1 , \ldots , M - 1$ define

$$
R _ { k } ^ { j } ( P _ { k } \mid P _ { k + 1 } ) = p _ { t _ { k } | 0 , t _ { k + 1 } } ( P _ { k } \mid e _ { j } , P _ { k + 1 } ) .
$$

The learned reverse kernel is

$$
K _ { k } ^ { \theta } ( P _ { k } \mid P _ { k + 1 } ) = \sum _ { j = 1 } ^ { N } \hat { P } _ { \theta , j } ( t _ { k + 1 } , P _ { k + 1 } ) R _ { k } ^ { j } ( P _ { k } \mid P _ { k + 1 } ) ,\tag{17}
$$

and the endpoint decoder is $p _ { \theta } ( x \mid P _ { 1 } ) = \hat { P } _ { \theta } ( t _ { 1 } , P _ { 1 } ) _ { x }$

For any $x \in \mathcal { X }$ , define the bridge path law

$$
Q _ { x } ( P _ { 1 : M } ) : = \mathrm { D i r } ( P _ { M } ; c _ { 1 } \pi ) \prod _ { k = 1 } ^ { M - 1 } R _ { k } ^ { x } ( P _ { k } \mid P _ { k + 1 } ) .
$$

Because $\alpha _ { 1 } = 0$ , the terminal marginal is $\mathrm { D i r } ( c _ { 1 } \pi ) = \mathrm { D i r } ( \beta _ { t _ { M } } ( e _ { x } , \pi ) )$ . One can easily check by backward induction that

$$
P _ { k } \sim \operatorname { D i r } ( \beta _ { t _ { k } } ( e _ { x } , \pi ) ) \quad { \mathrm { u n d e r } } Q _ { x } , \qquad k = 1 , \dots , M .\tag{18}
$$

Proposition F.1 (Cross-Entropy ELBO): Let $p _ { \theta } ( x )$ denote the marginal likelihood of the fixed-grid sampler with terminal law $\operatorname { D i r } ( c _ { 1 } \pi )$ , reverse kernels (17), and endpoint decoder $\hat { P } _ { \theta } ( t _ { 1 } , P _ { 1 } )$ . For every $x _ { 0 } \in \mathcal { X }$

$$
\begin{array} { r } { \log p _ { \theta } ( x _ { 0 } ) \geqslant \mathcal { L } _ { \mathrm { E L B O } } ( \theta ; x _ { 0 } ) , } \end{array}
$$

where

$$
\mathcal { L } _ { \mathrm { E L B O } } ( \theta ; x _ { 0 } ) = \sum _ { k = 1 } ^ { M } \mathbb { E } _ { P _ { k } \sim \mathrm { D i r } ( \beta _ { t _ { k } } ( e _ { x _ { 0 } } , \pi ) ) } \left[ \log \hat { P } _ { \theta } ( t _ { k } , P _ { k } ) _ { x _ { 0 } } \right] .\tag{19}
$$

Proof. Since (17) is a nonnegative mixture, for every $P _ { k + 1 }$

$$
K _ { k } ^ { \theta } ( P _ { k } \mid P _ { k + 1 } ) \geqslant { \hat { P } } _ { \theta } ( t _ { k + 1 } , P _ { k + 1 } ) _ { x _ { 0 } } R _ { k } ^ { x _ { 0 } } ( P _ { k } \mid P _ { k + 1 } ) .
$$

Retaining this single component at every reverse step gives

$$
p _ { \theta } ( x _ { 0 } ) \geqslant \mathbb { E } _ { Q _ { x _ { 0 } } } \left[ \prod _ { k = 1 } ^ { M } \hat { P } _ { \theta } ( t _ { k } , P _ { k } ) _ { x _ { 0 } } \right] .
$$

Taking logarithms and applying Jensen’s inequality,

$$
\begin{array} { r l r } { \log p _ { \theta } ( x _ { 0 } ) } & { } & { \log \mathbb { E } _ { Q _ { x _ { 0 } } } \left[ \prod _ { k = 1 } ^ { M } \hat { P } _ { \theta } ( t _ { k } , P _ { k } ) _ { x _ { 0 } } \right] } \\ & { } & { \geqslant \displaystyle \sum _ { k = 1 } ^ { M } \mathbb { E } _ { Q _ { x _ { 0 } } } \left[ \log \hat { P } _ { \theta } ( t _ { k } , P _ { k } ) _ { x _ { 0 } } \right] . } \end{array}
$$

Using the marginals (18) proves (19).

Thus, sampling k uniformly from the inference grid and then sampling $P _ { k } \sim \mathrm { D i r } ( \beta _ { t _ { k } } ( e _ { x _ { 0 } } , \pi ) )$ gives an unbiased estimator of $- \mathcal { L } _ { \mathrm { E L B O } } ( \theta ; x _ { 0 } ) / M$

Alternative bound. Keeping the full mixture before applying Jensen gives a tighter but generally intractable mixture-KL bound, which we do not consider further.

## G DISTILLATION

## G.1 SIMPLEX DISTRIBUTION MATCHING DISTILLATION

Throughout this section, we restrict attention to time pairs for which the Dirichlet parameters below are strictly positive.

Distribution Matching Distillation. We consider a multistep and simplicial version of Distribution Matching Distillation (DMD) (Luo et al., 2023; Yin et al., 2024). We introduce a generator $G : \mathcal { P } \times \lceil 0 , 1 \rceil \times \Delta _ { N } \times \mathcal { U } \to \Delta _ { N }$ , where $\mathcal { U }$ is a noise space and $\mathcal { P }$ is a real vector space which represents the parameter space of the parametric function $G _ { \eta } .$ . The main goal of the distillation method we describe below is to find $G _ { \eta }$ such that, for some random variable $\mathbf { \nabla } U \sim \mathbb { P } _ { U }$ taking values in $u ,$ we have $G _ { \eta } ( t , P _ { t } , U ) \sim \hat { p } _ { 0 | t } ( \cdot | \dot { P } _ { t } )$ , where $p _ { 0 \mid t }$ is given by Bayes’s rule, (3) and

$$
\int _ { \Delta _ { N } } p _ { t } ( P _ { t } ) p _ { 0 | t } ( P _ { 0 } | P _ { t } ) \mathrm { d } P _ { t } = \int _ { \Delta _ { N } } p _ { t } ( P _ { t } ) \hat { p } _ { 0 | t } ( P _ { 0 } | P _ { t } ) \mathrm { d } P _ { t } .
$$

In particular, $G _ { \eta } ( t , P _ { t } , U )$ should take values on the vertices of $\Delta _ { N } ,$ since its prior distribution $\begin{array} { r } { p _ { \mathrm { d a t a } } ( P ) = \sum _ { i = 1 } ^ { N } p _ { \mathrm { d a t a } , i } \delta _ { e _ { i } } ( P ) } \end{array}$ (see $( 2 ) )$ only has mass on the vertices. In what follows, we make no such assumption on the form of $G _ { \eta }$ and only ensure that $G _ { \eta }$ takes values in $\Delta _ { N }$

We denote $\mathrm { d } G _ { \eta } ( t , P _ { t } , U ) : \mathcal { P } \to \mathbb { R } ^ { N }$ the derivative of $G _ { \eta }$ with respect to η, evaluated at $( \eta , t , P _ { t } , U )$ Similarly, we denote $\mathrm { D } \dot { G } _ { \eta } ( t , P _ { t } , U )$ such that for any $h \in \mathcal { P }$

$$
\mathrm { d } G _ { \eta } ( t , P _ { t } , U ) ( h ) = \mathrm { D } G _ { \eta } ( t , P _ { t } , U ) h .
$$

We recall that for every $t \in [ 0 , 1 ]$ , we have that $P _ { t }$ is given by (3). For notational simplicity, we define $\mathbb { P } _ { t }$ the distribution of $\bar { P _ { t } }$ with $P _ { t } \sim p _ { t | 0 } ( \cdot | P _ { 0 } )$ and $P _ { 0 } = e _ { x _ { 0 } }$ with $x _ { 0 } \sim p _ { \mathrm { d a t a } }$ . More precisely for any test function $f \in \mathrm { C } ^ { \infty } ( \Delta _ { N } , \mathbb { R } )$ we have that

$$
\int _ { \Delta _ { N } } f ( P ) \mathrm { d } \mathbb { P } _ { t } ( P ) = \int _ { \Delta _ { N } } \int _ { \Delta _ { N } } f ( P _ { t } ) \mathrm { d } p _ { t | 0 } ( P _ { t } \mid P _ { 0 } ) \mathrm { d } p _ { \mathrm { d a t a } } ( P _ { 0 } ) .
$$

Similarly, we denote $\mathbb { P } _ { t } ^ { \eta , s }$ the distribution of $P _ { t }$ with $P _ { t } \sim p _ { t | 0 } ( \cdot | P _ { 0 } ^ { s } )$ where $\begin{array} { r } { P _ { 0 } ^ { s } = G _ { \eta } ( s , P _ { s } , U ) } \end{array}$ where $P _ { s } \sim p _ { s | 0 } ( \cdot \mid P _ { 0 } )$ and $P _ { 0 } \sim p _ { \mathrm { d a t a } } \left( \cdot \right)$ . More precisely for any test function $f \in \mathrm { C } ^ { \infty } ( \Delta _ { N } , \mathbb { R } )$ we have that

$$
\begin{array} { r l r } {  { \int _ { \Delta _ { N } } f ( P ) \mathrm { d } \mathbb { P } _ { t } ^ { \eta , s } ( P ) } } \\ & { } & { = \int _ { \Delta _ { N } } \int _ { \Delta _ { N } } \int _ { U } \int _ { \Delta _ { N } } \int _ { \Delta _ { N } } f ( P _ { t } ) \mathrm { d } p _ { t \mathrm { l } 0 } ( P _ { t } \mid P _ { 0 } ^ { s } ) \mathrm { d } \delta _ { G _ { \eta } ( s , P _ { s } , U ) } ( P _ { 0 } ^ { s } ) \mathrm { d } \mathbb { P } _ { U } ( U ) \mathrm { d } p _ { s \mid 0 } ( P _ { s } \mid P _ { 0 } ) \mathrm { d } p _ { \mathrm { d a n a } } ( P _ { 0 } ) . } \end{array}
$$

We denote by $\mathbb { P } _ { 0 \mid t } ^ { \eta , s }$ the conditional distribution of $P _ { 0 } ^ { s }$ given $P _ { t }$ under this construction, and by $\mathbb { P } _ { 0 \mid t }$ the conditional distribution of $P _ { 0 }$ given $P _ { t }$ under the data-forward joint distribution.

We consider the following distribution matching loss

$$
\mathcal { L } ( \eta ) = \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { 1 } \mathrm { K L } ( \mathbb { P } _ { t } ^ { \eta , s } | \mathbb { P } _ { t } ) \mathrm { d } \mathbb { Q } ( s , t ) .\tag{20}
$$

Note that in the original setting of DMD (Luo et al., 2023; Yin et al., 2024), we have that $\mathrm { d } \mathbb { Q } ( s , t ) =$ $\delta _ { 1 } ( s ) \mathrm { d } \mathbb { Q } ( t )$ . In the multistep regime however, we consider a general distribution $\mathbb { Q } .$ In what follows,

we are going to first derive the gradient of the loss function and then give an equivalent expression for (20) which can be readily implemented.

First, we recall the forward process

$$
p _ { t | 0 } ( P _ { t } | P _ { 0 } ) = \mathrm { D i r } ( P _ { t } ; \beta _ { t } ( P _ { 0 } , \pi ) ) , \qquad \beta _ { t } ( P _ { 0 } , \pi ) = a _ { t } P _ { 0 } + b _ { t } \pi = c _ { t } ( \alpha _ { t } P _ { 0 } + ( 1 - \alpha _ { t } ) \pi ) ,
$$

where $a _ { t } = c _ { t } \alpha _ { t }$ and $b _ { t } = c _ { t } ( 1 - \alpha _ { t } )$ , as in Section 3. In particular, using the fact that a Dirichlet random variable can be generated using Gamma random variables (see Section $\mathbf { A } )$ , then for any $t \in [ 0 , 1 ]$ there exists $F _ { t }$ such that $F _ { t } ( P _ { 0 } , \bar { V } )$ has distribution $p _ { t | 0 } ( \cdot | P _ { 0 } )$ , with $V = \{ V _ { i } \} _ { i = 1 } ^ { N }$ which represents N independent uniform random variables used to generate the N Gamma random variates. In addition, $F _ { t }$ is differentiable with respect to $P _ { 0 }$ . We denote $\mathrm { d } F _ { t } ( P _ { 0 } , V ) : \mathbb { R } ^ { N } \to \mathbb { R } ^ { N }$ the differential of $F _ { t }$ with respect to $P _ { 0 }$ evaluated at $( P _ { 0 } , V )$ . Finally, we introduce $\mathrm { ~ \mathrm { D } ~ } F _ { t } ( P _ { 0 } , V ) \in \mathbb { R } ^ { N \times N }$ such that for any $h \in \mathbb { R } ^ { N }$ we have

$$
\mathrm { d } F _ { t } ( P _ { 0 } , V ) ( h ) = \mathrm { D } F _ { t } ( P _ { 0 } , V ) h .
$$

Surrogate Loss. While (20) is a valid loss, it is not tractable. However, leveraging stop-gradient techniques and an equivalent of Tweedie’s identity in the case of Dirichlet distributions, we will be able to derive a surrogate tractable loss with identical gradients in Proposition G.2. We start with the following lemma.

Lemma G.1 (Chain rule): Let $s , t \in [ 0 , 1 ] , P _ { s } \in \Delta _ { N } , U \sim \mathbb { P } _ { U }$ and $V = \{ V _ { i } \} _ { i = 1 } ^ { N } , \ : N$ independent uniform random variables. Let $f : \Delta _ { N } \to \mathbb { R }$ be a differentiable function. Define $g : \mathcal { P } \to \mathbb { R }$ such that $g ( \eta ) = f ( F _ { t } ( G _ { \eta } ( s , P _ { s } , U ) , V ) )$ q. We have

$$
\nabla _ { \boldsymbol { \eta } } g ( \boldsymbol { \eta } ) = \mathrm { D } G _ { \boldsymbol { \eta } } ( s , P _ { s } , U ) ^ { \top } \mathrm { D } F _ { t } ( G _ { \boldsymbol { \eta } } ( s , P _ { s } , U ) , V ) ^ { \top } \nabla f ( F _ { t } ( G _ { \boldsymbol { \eta } } ( s , P _ { s } , U ) , V ) ) .
$$

Next, we consider the following lemma which shows that Dirichlet distribution also enjoys some form of Tweedie’s identity.

Lemma G.2 (Tweedie meets Dirichlet): $L e t p _ { X }$ be a distribution over $\Delta _ { N }$ and $p _ { Y \mid X }$ a Dirichlet distribution with parameter $\alpha ( X ) \in ( 0 , + \infty ) ^ { N }$ . Then, we have

$$
\nabla \log p _ { Y } ( y ) = \left\{ \frac { \mathbb { E } [ \alpha _ { i } ( X ) | Y ] - 1 } { y _ { i } } \right\} _ { i = 1 } ^ { N } .
$$

Proof. We have that $p _ { X }$ is a distribution over $\Delta _ { N }$ and $p _ { Y \mid X }$ is a Dirichlet distribution with parameter $\alpha ( X ) \in ( 0 , + \infty ) ^ { N }$ . We have that for any $y \in \Delta _ { N }$

$$
p _ { Y | X } ( y | X ) = \frac { \Gamma ( \sum _ { i = 1 } ^ { N } \alpha _ { i } ( X ) ) } { \prod _ { i = 1 } ^ { N } \Gamma ( \alpha _ { i } ( X ) ) } \prod _ { i = 1 } ^ { N } y _ { i } ^ { \alpha _ { i } ( X ) - 1 } .
$$

Then, we have that

$$
\nabla _ { y } \log p _ { Y | X } ( y | X ) = \{ ( \alpha _ { i } ( X ) - 1 ) / y _ { i } \} _ { i = 1 } ^ { N } .
$$

In addition, we have that

$$
\nabla \log p _ { Y } ( y ) = \int \nabla _ { y } \log p _ { Y | X } ( y | x ) p _ { X | Y } ( x | y ) \mathrm { d } x .
$$

Finally, we get that

$$
\nabla \log p _ { Y } ( y ) = \left\{ \frac { \mathbb { E } [ \alpha _ { i } ( X ) | Y ] - 1 } { y _ { i } } \right\} _ { i = 1 } ^ { N } .
$$

Combining the standard pathwise gradient identity for the KL divergence with Lemma G.1 and Lemma G.2, we obtain the following proposition.

Proposition G.1: Let $\mathcal { L } ( \eta )$ be given by (20). Then, we have that

$$
\begin{array} { r l } & { \nabla _ { \eta } \mathcal { L } ( \eta ) = \displaystyle \int \boldsymbol { a } _ { t } \mathrm { D } G _ { \eta } ( s , P _ { s } , U ) ^ { \top } \mathrm { D } F _ { t } ( G _ { \eta } ( s , P _ { s } , U ) , V ) ^ { \top } } \\ & { \qquad \times \frac { \mathbb { E } _ { \mathbb { P } _ { 0 | t } ^ { \eta , s } } \left[ P _ { 0 } ^ { s } \left| F _ { t } ( G _ { \eta } ( s , P _ { s } , U ) , V ) \right. \right] - \mathbb { E } _ { \mathbb { P } _ { 0 | t } } \left[ P _ { 0 } \left| F _ { t } ( G _ { \eta } ( s , P _ { s } , U ) , V ) \right. \right] } { F _ { t } ( G _ { \eta } ( s , P _ { s } , U ) , V ) } } \\ & { \qquad \times \mathrm { d } \mathbb { Q } ( s , t ) \mathrm { d } p _ { \mathrm { d a t a } } ( P _ { 0 } ) \mathrm { d } p _ { s } | 0 ( P _ { s } | P _ { 0 } ) \mathrm { d } \mathbb { P } _ { U } ( U ) \mathrm { d } V , } \end{array}
$$

where the division is coordinate-wise.

We can therefore find an equivalent loss to (20) which is tractable and yields the same gradients.

Proposition G.2 (Surrogate loss): Let $\hat { \mathcal { L } } ( \eta )$ be given by

$$
\begin{array} { r l } & { \hat { \mathcal { L } } ( \eta ) = \displaystyle \int a _ { t } \log ( F _ { t } ( G _ { \eta } ( s , P _ { s } , U ) , V ) ) ^ { \top } } \\ & { \quad \quad \quad \quad \quad \quad \quad \mathrm { s g } \left( \mathbb { E } _ { \mathbb { P } _ { 0 | t } ^ { \eta , s } } [ P _ { 0 } ^ { s } | F _ { t } ( G _ { \eta } ( s , P _ { s } , U ) , V ) ] - \mathbb { E } _ { \mathbb { P } _ { 0 | t } } [ P _ { 0 } | F _ { t } ( G _ { \eta } ( s , P _ { s } , U ) , V ) ] \right) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \times \mathrm { d } \mathbb { Q } ( s , t ) \mathrm { d } p _ { \mathrm { d a t a } } ( P _ { 0 } ) \mathrm { d } p _ { s | 0 } ( P _ { s } | P _ { 0 } ) \mathrm { d } \mathbb { P } _ { U } ( U ) \mathrm { d } V . } \end{array}
$$

Then $\hat { \mathcal { L } }$ and L have the same gradients.

Algorithm and Links with Literature. Leveraging the loss given in Proposition G.2, we consider an algorithm to learn a distilled model. The full algorithm is given in Algorithm 4. With exact conditional means, i.e., the denoisers are exact, and a valid pathwise derivative of the corruption map, Algorithm 4 is valid. In practice, learned denoisers approximate these conditional means, yielding an approximate gradient. To differentiate through the forward process, we here differentiate through the fast Gamma implementation of Section D. Note that one might be able to leverage the implementation of Greaves (2026) to obtain numerical gradients which are more precise. We leave this exploration for future work. It is apparent that Algorithm 4 follows the same pattern as many existing distillation algorithms such as DMD (Yin et al., 2024) but also Multistep Moment matching Distillation (MMD) (Salimans et al., 2024). In fact, our algorithm can be interpreted as a simplicial extension of the MMD algorithm. In both cases, we train a generator network $G _ { \eta }$ as well as an auxiliary denoiser approximating $\mathbb { E } _ { \mathbb { P } _ { 0 | t } ^ { \eta , s } } [ P _ { 0 } ^ { s } | P _ { t } ]$ . Another line of related work is Universal Distillation Matching (UDM) and its discrete counterpart Inverse Distillation Language Model (IDLM) (Kornilov et al., 2026; Li et al., 2026). While closely related to our method, we highlight the key differences between these methods and our contribution in Section G.2.

## G.2 LINK WITH INVERSE DISTILLED LANGUAGE MODELS

In this section, we draw connections between the Simplex DMD framework and the Inverse Distilled Language Model (IDLM) approach (Kornilov et al., 2026; Li et al., 2026). We show that both methods share a common structure but differ in how the gradient of the generative loss is handled, with the Simplex DMD framework resolving a fundamental difficulty encountered in IDLM.

IDLM Generative Loss. Adapting the formulation of Kornilov et al. (2026); Li et al. (2026) to the notation of the present paper, the IDLM generative loss can be written as

$$
\begin{array} { r } { \mathcal { L } ^ { \mathbb { G } } ( \eta ) = \displaystyle \int G _ { \eta } ( s , P _ { s } , U ) ^ { \top } \left( \log D _ { \phi } ( F _ { t } ( G _ { \eta } ( s , P _ { s } , U ) , V ) , s , t ) - \log D _ { \mathrm { t e a c h } } ( F _ { t } ( G _ { \eta } ( s , P _ { s } , U ) , V ) , t ) \right) } \\ { \times \mathrm { d } \mathbb { Q } ( s , t ) \mathrm { d } p _ { \mathrm { d a t a } } ( P _ { 0 } ) \mathrm { d } p _ { s | 0 } ( P _ { s } | P _ { 0 } ) \mathrm { d } \mathbb { P } _ { U } ( U ) \mathrm { d } V , ~ } \end{array}
$$

where $D _ { \phi }$ plays the role of the auxiliary model and $D _ { \mathrm { t e a c h } }$ the teacher model from Li et al. (2026). The generator $G _ { \eta }$ appears twice in this expression: as the denoised sample $P _ { 0 } ^ { s } = G _ { \eta } ( s , P _ { s } , \dot { U } )$ and as the input to the forward reparameterization $P _ { t } = F _ { t } ( G _ { \eta } ( s , P _ { s } , U ) , \dot { V } )$ . As a result, the gradient

```latex
Algorithm 4 Simplex DMD Training
Require: Teacher denoiser $D _ { \operatorname { t e a c h } } ( P _ { t } , t ) \approx \mathbb { E } _ { \mathbb { P } _ { 0 | t } } [ P _ { 0 } \mid P _ { t } ]$
Require: Generator $G _ { \eta } ( s , P _ { s } , U )$ , auxiliary denoiser $D _ { \phi } ( P _ { t } , s , t ) \approx \mathbb { E } _ { \mathbb { P } _ { 0 \mid t } ^ { \eta , s } } [ P _ { 0 } ^ { s } \ | \ P _ { t } ]$
Require: Data distribution $p _ { \mathrm { d a t a } } ,$ noise distribution $\mathbb { P } _ { U }$ , time distribution $\mathbb { Q }$
Require: Forward process parameters $( \alpha _ { t } ) _ { t \in [ 0 , 1 ] } , ( c _ { t } ) _ { t \in [ 0 , 1 ] } .$ , and $\pi ,$ with $a _ { t } = c _ { t } \alpha _ { t }$
Require: Learning rates $\mathrm { l r } _ { \eta } , \mathrm { l r } _ { \phi }$
1: repeat
2: 1. Sample data and times
3: $x _ { 0 } \sim p _ { \mathrm { d a t a } } , \quad P _ { 0 } \gets e _ { x _ { 0 } }$
4: $( s , t ) \sim \mathbb { Q }$
5: 2. Teacher forward to time s Ź No gradient through this step
6: for $i = 1 , \ldots , N$ do
7: $G _ { i } ^ { ( s ) } \gets \mathrm { G A M M A S A M P L E } \big ( \beta _ { s } ( P _ { 0 } , \pi ) _ { i } \big )$ Ź Algorithm 3
8: end for
9: $\begin{array} { r } { P _ { s }  ( G _ { 1 } ^ { ( s ) } , \ldots , G _ { N } ^ { ( s ) } ) / { \sum _ { j } G _ { j } ^ { ( s ) } } } \end{array}$
10: 3. Generator (one denoising step)
11: $U \sim \mathbb { P } _ { U }$
12: $P _ { 0 } ^ { s } \gets G _ { \eta } ( s , P _ { s } , U )$
13: 4. Student forward to time t Ź Differentiable through $P _ { 0 } ^ { s }$
14: for $i = 1 , \ldots , N$ do
15: $G _ { i } ^ { ( t ) } \gets \mathrm { G A M M A S A M P L E } ( \beta _ { t } ( P _ { 0 } ^ { s } , \pi ) _ { i } )$ Ź Algorithm 3
16: end for
17: $\textstyle P _ { t } \gets ( G _ { 1 } ^ { ( t ) } , \ldots , G _ { N } ^ { ( t ) } ) / { \sum _ { j } G _ { j } ^ { ( t ) } }$
18: 5. Score difference (stopped gradients)
19: $\Delta \gets \mathrm { s g } \big ( D _ { \phi } ( P _ { t } , s , t ) \big ) \ - \mathrm { \tilde { s g } } \big ( \bar { D _ { \mathrm { t e a c h } } } ( P _ { t } , t ) \big )$
20: 6. Generator update
N
21: $\hat { \mathcal { L } } _ { \eta } \gets a _ { t } \sum ^ { i \ v { v } } \log ( P _ { t , k } ) \cdot \Delta _ { k }$ Ź Gradient flows only through log $P _ { t }$
k“1
22: $\boldsymbol { \eta } \gets \boldsymbol { \eta } - \mathrm { l r } _ { \eta } \cdot \nabla _ { \eta } \hat { \mathcal { L } } _ { \eta }$
23: 7. Auxiliary denoiser update
24: $\begin{array} { r } { \hat { \mathcal { L } } _ { \phi } \gets - \sum _ { k = 1 } ^ { N } \log ( D _ { \phi } ( \mathrm { s g } ( P _ { t } ) , s , t ) ) _ { k } \cdot \mathrm { s g } ( P _ { 0 } ^ { s } ) _ { k } } \end{array}$
25: $\phi  \phi - \mathrm { l r } _ { \phi } \cdot \nabla _ { \phi } \hat { \mathcal { L } } _ { \phi }$
26: until converged
```

of $\mathcal { L } ^ { \mathrm { G } }$ with respect to η splits into two terms:

$$
\begin{array} { r l r } & { } & { \nabla _ { \eta } \mathcal { L } ^ { \mathrm { G } } ( \eta ) = \displaystyle \int \mathrm { D } G _ { \eta } ^ { \top } \underbrace { \left( \log D _ { \phi } ( P _ { t } , s , t ) - \log D _ { \mathrm { t e a c h } } ( P _ { t } , t ) \right) } _ { \mathrm { d i r e c t ~ g r a d i e n t } } \mathrm { d } ( \cdots ) } \\ & { } & { \qquad + \displaystyle \int \mathrm { D } G _ { \eta } ^ { \top } \mathrm { D } F _ { t } ^ { \top } \underbrace { \left( \frac { \partial \log D _ { \phi } } { \partial P _ { t } } - \frac { \partial \log D _ { \mathrm { t e a c h } } } { \partial P _ { t } } \right) ^ { \top } G _ { \eta } } _ { \mathrm { i n d i r e c t ~ g r a d i e n t } } \mathrm { d } ( \cdots ) . } \end{array}\tag{21}
$$

(22)

As discussed in Li et al. (2026); Hoogeboom et al. (2026), the indirect gradient (22) is empirically high-variance, and current implementations discard it entirely. The resulting update, however, is not generally the gradient of any well-defined objective.

Comparison with Simplex DMD. By contrast, the surrogate loss $\hat { \mathcal { L } } ( \eta )$ of Proposition G.2 yields a single gradient term

$$
\nabla _ { \eta } \hat { \mathcal { L } } ( \eta ) = \int a _ { t } \mathrm { D } G _ { \eta } ^ { \top } \mathrm { D } F _ { t } ^ { \top } \frac { \mathrm { s g } \big ( D _ { \phi } ( P _ { t } , s , t ) - D _ { \mathrm { t e a c h } } ( P _ { t } , t ) \big ) } { P _ { t } } \mathrm { d } ( \cdot \cdot \cdot ) ,\tag{23}
$$

which is the exact gradient of the KL divergence $\mathcal { L } ( \eta )$ defined in (20). We highlight several structural differences with the IDLM gradient (21)–(22).

1. Single gradient path. In (23), the gradient flows exclusively through the forward map $F _ { t }$ There is no direct term because the DMD loss is derived from $\mathrm { K L } ( \mathbb { P } _ { t } ^ { \eta , s } \| \mathbb { P } _ { t } )$ via the Stein score identity for pushforward measures, which produces a gradient that is entirely of the indirect type.

2. Exact gradient with stop-gradient. The sg operator on $\Delta$ in (23) is not a heuristic approximation: Proposition G.2 guarantees $\nabla _ { \eta } \hat { \mathcal { L } } \ = \ \nabla _ { \eta } \mathcal { L }$ . In contrast, discarding the indirect gradient in IDLM is an approximation without such a guarantee.

3. Auxiliary model. The IDLM auxiliary loss $- \int P _ { 0 } ^ { s , \top } \log D _ { \phi } ( P _ { t } , s , t ) \mathrm { d } ( \cdot \cdot \cdot )$ is a crossentropy between the generator output and the auxiliary model. This is identical in structure to the auxiliary denoiser loss in Step 7 of Algorithm 4, confirming the correspondence between the auxiliary model of Li et al. (2026) and the auxiliary denoiser $D _ { \phi }$

4. Differentiability. In the simplicial framework, $F _ { t }$ is differentiable via the Γ- reparameterization (Algorithm 3), so the Jacobian D $F _ { t }$ in (23) can be computed exactly by automatic differentiation. In discrete models (Li et al., 2026), the forward corruption is non-differentiable, necessitating biased approximations. This is one of the key advantages of the simplicial formulation.

## Part III: Connection with the Literature

## H EXTENDED RELATED WORK

## H.1 GENERAL OVERVIEW

In this section, we present an overview of simplex diffusion models and their applications to discrete data generation. For Discrete Diffusions, we refer the reader to the original paper of Austin et al. (2021) and more recent extensions (Campbell et al., 2022; Benton et al., 2024; Lou et al., 2024; Sahoo et al., 2024; Shi et al., 2024).

Several simplex diffusion models are obtained by defining a forward stochastic process directly on the simplex. Then, relying on tools from time-reversal (Haussmann & Pardoux, 1986), similar to

Song et al. (2021b) in Euclidean state spaces, one can define a generative process. In Richemond et al. (2022), inspired by Baker et al. (2018), the forward process corresponds to a set of N Cox– Ingersoll–Ross processes that converge to a Gamma distribution and to a Dirichlet distribution after renormalization. Floto et al. (2023) consider another forward process based on a softmax transformation of an Ornstein–Uhlenbeck process. One of the main advantages of Floto et al. (2023) is that for any $t \in [ 0 , 1 ]$ the distribution $p _ { t | 0 }$ is a logistic normal distribution which is easy to sample. Dirichlet Diffusion Score Models (DDSMs) (Avdeyev et al., 2023) consider instead a Jacobi diffusion process as a forward process that converges to a Beta distribution. By then leveraging a stick-breaking construction, the authors obtain a forward process converging to a Dirichlet distribu tion. Benton et al. (2024) instead use the Wright-Fisher diffusion as a forward process, a diffusion on the simplex originating from genetics. Recently, in Chandra et al. (2026), it was shown that discrete, continuous and Wright-Fisher diffusions could arise from a similar particle perspective, where the Wright-Fisher diffusion is obtained as a limit where the number of particles goes to infinity and we consider reproduction within the particles’ evolution.

Yet another approach is to define diffusion models taking into account the special geometry of the simplex. Doing so, it is then possible to define diffusion models on an associated statistical manifold using general techniques from Flow Matching and Riemannian diffusion models (De Bortoli et al., 2022; Huang et al., 2022; Chen & Lipman, 2024). While in Cheng et al. (2024); Davis et al. (2024); Williams et al. (2026), the authors derive the metrics on the simplex using the Fisher–Rao connection, Boll et al. (2024; 2025) consider e-connections, see (Davis et al., 2024, Appendix E.2) for a discussion of the choice of metrics. In Han et al. (2023); Mahabadi et al. (2024); Jo & Hwang (2025), the authors consider a Gaussian diffusion in the space of logits embeddings. Those different approaches, along with the one ignoring the geometry of the simplex and simply performing a linear diffusion in that space (Dunn & Koes, 2024), can be unified using the concept of α-divergence to define the statistical manifold (see Cheng et al., 2025).

Another approach incorporating the geometry of the simplex into the generation process is to consider a constrained forward process using either projection or reflection of the original Stochastic Differential Equation (Liu et al., 2023b; Fishman et al., 2024; 2023; Lou & Ermon, 2023).

The closest work to ours is the Dirichlet Flow Matching approach introduced by Stark et al. (2024). In this work, the authors introduced a forward process akin to ours. The main difference between the two approaches arises from inference. While Stark et al. (2024) leverage a flow perspective, our approach is akin to DDIM (Song et al., 2021a). We also provide a different temperature-parameterized process. Note that very recently Boget & Kalousis (2026) have proposed the same forward process as Stark et al. (2024). They introduce an inference procedure which, like ours, allows for stochasticity but assumes $p _ { s | 0 , t } ( P _ { s } | P _ { 0 } , P _ { t } ) = p _ { s | 0 } ( P _ { s } | P _ { 0 } )$ as in star-shaped diffusion models (Okhotin et al., 2023) (corresponding to $\kappa = 1$ in our case). Another concurrent work (Simplax) (Sakurai et al., 2026) also uses Dirichlet simplex states, but their denoiser and sampling procedure operate on categorical samples rather than the continuous simplex state itself.

We conclude by mentioning that Haviv et al. (2025) introduce Wasserstein Flow Matching, i.e., define Flow Matching on the space of distributions, similar to the simplicial approach lifting distributions on a given state space to the space of distributions of distributions. Finally, we note that the recent work of Song et al. (2025) introducing Shortlisting Models (SLMs) also claims a simplex approach as they “aim to preserve the core principle of simplex-based methods, gradual information growth”. To do so, they operate on the space of probability vectors. Starting from a single full probability vector, they progressively eliminate categories until the final denoised state is one-hot.

## H.2 SIMPLEX RELAXATION FOR DISCRETE DIFFUSION

In this section, we describe the approach of Sakurai et al. (2026) and compare it with our framework. We first outline how they construct their forward process alongside their simplex relaxation. Next, we discuss their chosen training loss. Finally, we investigate their sampling procedure and show that it can be simplified to bypass the simplex relaxation, thereby highlighting a fundamental difference from Simplex Diffusion Models.

Forward Process. Sakurai et al. (2026) first consider a forward process on the categorical space given for any $t \in [ 0 , 1 ]$ and $x _ { t } , x _ { 0 } \in \{ 1 , \ldots , N \}$ by

$$
p _ { t | 0 } ( x _ { t } | x _ { 0 } ) = \mathrm { C a t } ( x _ { t } ; \alpha _ { t } x _ { 0 } + ( 1 - \alpha _ { t } ) \pi ) ,
$$

where $\pi \in \Delta _ { N }$ is a probability distribution. Note that, as emphasized by Sakurai et al. (2026), one can define a compatible backward bridge transition for this forward rule by defining for any $s , t \in [ 0 , 1 ]$ and $s \leqslant t$ and $x _ { 0 } , x _ { s } , x _ { t } \in \{ 1 , \ldots , N \}$ by

$$
p _ { s | 0 , t } ( x _ { s } | x _ { 0 } , x _ { t } ) = \mathrm { C a t } \left( x _ { s } ; \frac { \left[ \frac { \alpha _ { t } } { \alpha _ { s } } x _ { t } + \left( 1 - \frac { \alpha _ { t } } { \alpha _ { s } } \right) \langle x _ { t } , \pi \rangle \mathbf { 1 } \right] \odot ( \alpha _ { s } x _ { 0 } + ( 1 - \alpha _ { s } ) \pi ) } { \langle x _ { t } , \alpha _ { t } x _ { 0 } + ( 1 - \alpha _ { t } ) \pi \rangle } \right) .
$$

We denote $r _ { s \vert 0 , t }$ the mean of $p _ { s | 0 , t } .$ . In particular, we have that for any $x _ { 0 } , x _ { s } \in \{ 1 , \ldots , N \}$

$$
p _ { s | 0 } ( x _ { s } | x _ { 0 } ) = \sum _ { x _ { t } = 1 } ^ { N } p _ { s | 0 , t } ( x _ { s } | x _ { 0 } , x _ { t } ) p _ { t | 0 } ( x _ { t } | x _ { 0 } ) .
$$

One of the main innovations of Sakurai et al. (2026) is to introduce the simplex-valued variable $w _ { t } \in \Delta _ { N }$ for any $t \in [ 0 , 1 ]$ with

$$
p ( w _ { t } | x _ { t } , x _ { 0 } ) = \mathrm { D i r } ( w _ { t } ; \eta _ { t } ( \alpha _ { t } x _ { 0 } + ( 1 - \alpha _ { t } ) \pi ) + x _ { t } ) .
$$

Training Loss. The training loss they consider is given for a given $s , t \in [ 0 , 1 ]$ with $s \ \leqslant \ t .$ $x _ { 0 } \in \{ 1 , \ldots , N \}$ and $w _ { t } \in \Delta _ { N }$ 1

$$
\mathcal { L } _ { s , t } = \sum q ( \tilde { x } _ { t } | w _ { t } ) \mathrm { K L } ( p _ { s | 0 , t } ( x _ { s } | x _ { 0 } , \tilde { x } _ { t } ) | p ( x _ { s } | \hat { x } _ { \theta } ( x _ { t } ) , \tilde { x } _ { t } ) ) .
$$

This can be simplified (see (Sakurai et al., 2026, Proposition 5)) in

$$
\begin{array} { r l } & { \mathcal { L } _ { s , t } = \langle w _ { t } , \log ( \alpha _ { t } \hat { x } _ { \theta } ( x _ { t } ) + ( 1 - \alpha _ { t } ) \pi ) - \log ( \alpha _ { t } x _ { 0 } + ( 1 - \alpha _ { t } ) \pi ) \rangle } \\ & { \qquad + \left. \rho _ { s | 0 , t } , \log ( \alpha _ { s } x _ { 0 } + ( 1 - \alpha _ { s } ) \pi ) - \log ( \alpha _ { s } \hat { x } _ { \theta } ( x _ { t } ) + ( 1 - \alpha _ { s } ) \pi ) \right. } \end{array}
$$

where $\rho _ { s | 0 , t }$ is defined by (Sakurai et al., 2026, Equation 11)

$$
\odot \left[ \frac { \alpha _ { t } } { \alpha _ { s } } ( w _ { t } \oslash ( \alpha _ { t } x _ { 0 } + ( 1 - \alpha _ { t } ) \pi ) ) + \left( 1 - \frac { \alpha _ { t } } { \alpha _ { s } } \right) \langle w _ { t } , \pi \oslash ( \alpha _ { t } x _ { 0 } + ( 1 - \alpha _ { t } ) \pi ) \rangle { \bf 1 } \right] .
$$

In contrast, we use a simple cross-entropy loss (11), and discuss a true ELBO in Section F.

Sampling. Once the model is trained, they sample from the model as follows. Let $s , t \in [ 0 , 1 ]$ with $s \leqslant t .$ Assume that we have access to a pair $( x _ { t } , w _ { t } )$ . Then, they let the denoiser predict $x _ { \theta } ( t , x _ { t } )$ Then, they sample $x _ { s } \sim \mathrm { C a t } ( \rho _ { s | 0 , t } )$ and $w _ { s } \sim \mathrm { D i r } ( \eta _ { s } ( \alpha _ { s } x _ { \theta } ( t , x _ { t } ) + ( 1 - \alpha _ { s } ) \pi ) + x _ { s } )$ . In (Sakurai et al., 2026, Proposition 1), it is shown that $x _ { s } \sim p _ { s | 0 , t } ( x _ { s } | x _ { \theta } ( t , x _ { t } ) , w _ { t } )$ . Therefore, we have that for any test function $f : \{ 1 , \dots , N \} \to \mathbb { R }$

$$
\mathbb { E } \big [ f ( x _ { s } ) | x _ { t } \big ] = \sum f ( x _ { s } ) \mathbb { E } \big [ p _ { s | 0 , t } ( x _ { s } | x _ { 0 } , w _ { t } ) | x _ { t } \big ] p _ { 0 | t } ( x _ { 0 } | x _ { t } ) ,
$$

where the expectation is w.r.t. $w _ { t }$ . In addition, we have that

$$
\mathbb { E } \big [ p _ { s | 0 , t } ( x _ { s } | x _ { 0 } , w _ { t } ) | x _ { t } \big ] = \big ( \alpha _ { s } x _ { 0 } + ( 1 - \alpha _ { s } ) \pi \big )
$$

$$
\odot \left[ \frac { \alpha _ { t } } { \alpha _ { s } } ( \mathbb { E } [ w _ { t } | x _ { t } ] \odot ( \alpha _ { t } x _ { 0 } + ( 1 - \alpha _ { t } ) \pi ) ) + \left( 1 - \frac { \alpha _ { t } } { \alpha _ { s } } \right) \langle \mathbb { E } [ w _ { t } | x _ { t } ] , \pi \oslash ( \alpha _ { t } x _ { 0 } + ( 1 - \alpha _ { t } ) \pi ) \rangle { \bf 1 } \right] .
$$

We have that

$$
\mathbb { E } \big [ w _ { t } | x _ { t } \big ] = \frac { \eta _ { t } } { 1 + \eta _ { t } } \big ( \alpha _ { t } x _ { 0 } + ( 1 - \alpha _ { t } ) \pi \big ) + \frac { 1 } { 1 + \eta _ { t } } x _ { t } .
$$

Therefore, we have that

$$
\mathbb { E } \big [ p _ { s | 0 , t } ( x _ { s } | x _ { 0 } , w _ { t } ) | x _ { t } \big ] = \frac { \eta _ { t } } { 1 + \eta _ { t } } ( \alpha _ { s } x _ { 0 } + ( 1 - \alpha _ { s } ) \pi ) + \frac { 1 } { 1 + \eta _ { t } } r _ { s | 0 , t } .
$$

Hence, we get that

$$
\mathbb { E } [ f ( x _ { s } ) | x _ { t } ] = \sum f ( x _ { s } ) \sum \left( \frac { \eta _ { t } } { 1 + \eta _ { t } } p _ { s | 0 } ( x _ { s } | x _ { 0 } ) + \frac { 1 } { 1 + \eta _ { t } } p _ { s | 0 , t } ( x _ { s } | x _ { 0 } , x _ { t } ) \right) p _ { 0 | t } ( x _ { 0 } | x _ { t } ) .
$$

Therefore, we can interpret the transition proposed in Sakurai et al. (2026) as a pure discrete backward transition with remasking with remasking levels controlled by $\eta _ { t }$

In contrast, in our framework, we do not maintain a discrete state during the generation and instead only track a simplex state and only sample from the terminal simplex state.

## I EXTENDED BACKGROUND

## I.1 DISCRETE DIFFUSION MODELS

In this section, we outline the training and sampling procedures for Discrete Diffusion models. For simplicity, we describe processes over scalar variables, and the extension to sequences is similar to Section C.1. Refer to Austin et al. (2021); Campbell et al. (2022); Sahoo et al. (2024); Shi et al. (2024); Schiff et al. (2025); von Rutte et al.¨ (2025); Gourevitch et al. (2026) for the derivations. Discrete Diffusion Models define a corruption process (1) in terms of a prior π $\in \Delta _ { N }$ . Prior work mainly focuses on the absorbing (or masked) prior $\pi ^ { \mathrm { { m a s k } } } = m$ , where m is the one-hot embedding of a special [MASK] token, and the uniform prior $\pi ^ { \mathrm { u n i f } } = { \bf 1 } / N$ . Several works study mixtures of $\pi ^ { \mathrm { m a s k } }$ and $\pi ^ { \mathrm { u n i f } }$ (Fathi et al., 2025; von Rutte et al. ¨ , 2025; Liu et al., 2026; Wang et al., 2026; Zhang et al., 2026), or data-dependent priors (Alamdari et al., 2023; Vignac et al., 2023; Qin et al., 2025) but it is not clear whether elaborate priors are necessary at scale for language modeling (Sahoo et al., 2026; von Rutte et al.¨ , 2026).

Sampling. Let us refer to $p _ { s | 0 , t }$ as the bridge (Gourevitch et al., 2026):

$$
\begin{array} { r l } & { p _ { s | 0 , t } ( x _ { s } \mid x _ { 0 } , x _ { t } ) = \frac { p _ { t | s } ( x _ { t } \mid x _ { s } ) p _ { s | 0 } ( x _ { s } \mid x _ { 0 } ) } { p _ { t | 0 } ( x _ { t } \mid x _ { 0 } ) } } \\ & { \qquad = \mathrm { C a t } \left( x _ { s } ; \frac { \left[ \alpha _ { t | s } e _ { x t } + ( 1 - \alpha _ { t | s } ) ( e _ { x _ { t } } ^ { \top } \pi ) \mathbf { 1 } \right] \odot \left[ \alpha _ { s } e _ { x _ { 0 } } + ( 1 - \alpha _ { s } ) \pi \right] } { \alpha _ { t } ( e _ { x _ { t } } ^ { \top } e _ { x _ { 0 } } ) + ( 1 - \alpha _ { t } ) ( e _ { x _ { t } } ^ { \top } \pi ) } \right) } \end{array}\tag{24}
$$

where $\alpha _ { t | s } = \alpha _ { t } / \alpha _ { s }$ . While the bridge (24) is defined for discrete variables $x _ { 0 } , x _ { s } , x _ { t }$ , it is possible to define transitions $\hat { p } _ { s | t } ^ { \theta } ( x _ { s } | x _ { t } )$ by replacing $e _ { x _ { 0 } }$ with the predictions of a denoiser $\mathbf { x } _ { \theta } ( t , x _ { t } )$ . Thus, with a slight abuse of notation, one can write $\hat { p } _ { s | t } ^ { \theta } ( x _ { s } | x _ { t } ) = p _ { s | 0 , t } ( x _ { s } | \mathbf { x } _ { \theta } ( t , x _ { t } ) , x _ { t } )$ . After expanding (24) with the absorbing and uniform priors, we find that

$$
p _ { s | 0 , t } ^ { \mathrm { m a s k } } ( x _ { s } \mid x _ { 0 } , x _ { t } ) = \left\{ \begin{array} { l l } { \mathrm { C a t } ( x _ { s } ; e _ { x _ { 0 } } ) } & { \mathrm { i f ~ } x _ { t } = x _ { 0 } } \\ { \mathrm { C a t } \left( x _ { s } ; \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } e _ { x _ { 0 } } + \frac { 1 - \alpha _ { s } } { 1 - \alpha _ { t } } m \right) } & { \mathrm { i f ~ } x _ { t } = m , } \end{array} \right.
$$

and

$$
p _ { s | 0 , t } ^ { \mathrm { u i f } } ( x _ { s } \mid x _ { 0 } , x _ { t } ) = \mathrm { C a t } \left( x _ { s } ; \frac { N \alpha _ { t } ( e _ { x _ { t } } ^ { \top } e _ { x _ { 0 } } ) e _ { x _ { t } } + ( \alpha _ { t | s } - \alpha _ { t } ) e _ { x _ { t } } + ( \alpha _ { s } - \alpha _ { t } ) e _ { x _ { 0 } } + D _ { s , t } \mathbf { 1 } / N } { N \alpha _ { t } ( e _ { x _ { t } } ^ { \top } e _ { x _ { 0 } } ) + 1 - \alpha _ { t } } \right) = \frac { N \alpha _ { t } } { N \alpha _ { t } } \frac { 1 } { N \alpha _ { t } } \frac { \mathcal { M } ( E _ { x _ { t } } ^ { \top } ( x _ { t } ) ) } { N \alpha _ { t } } .
$$

where $D _ { s , t } : = ( 1 - \alpha _ { t | s } ) ( 1 - \alpha _ { s } )$ . For a time grid $0 = t _ { 0 } < t _ { 1 } < . . . < t _ { n } = 1$ , the standard ancestral sampler applies the transition $\hat { p } _ { s | t } ^ { \theta } ( x _ { s } | x _ { t } )$ n times to obtain $x _ { 0 } \colon$

$$
x _ { 1 } \sim \mathrm { C a t } ( \pi ) , \qquad x _ { t _ { i - 1 } } \sim \hat { p } _ { t _ { i - 1 } | t _ { i } } ^ { \theta } ( \cdot \mid x _ { t _ { i } } ) \quad \mathrm { f o r } i = n , \ldots , 1 .
$$

Alternatively, Predictor-Corrector samplers (Grathwohl et al., 2021; Campbell et al., 2022; Lezama et al., 2023; Sun et al., 2023; Campbell et al., 2024; Kim et al., 2026b; Wang et al., 2025; Liu et al., 2025; Zhao et ${ \mathrm { a l . , } }$ 2025; Deschenaux et al., 2026a; Gourevitch et al., 2026) also exist. We describe the variant used in our experiments in Section I.2.

Temperature. All samplers can scale the logits $\zeta$ of the denoiser by a temperature $T > 0 .$ , i.e., they use the prediction

$$
\operatorname { s o f t m a x } ( \zeta / T )\tag{25}
$$

instead of softmaxpζq. $T = 1$ recovers the denoiser prediction, and $T  0$ approaches greedy decoding. For SDMs, the scaled prediction replaces $\hat { P } _ { \theta } ( t , P _ { t } )$ in the transition of Proposition 3.3.

Training. As for Variational (continuous) Diffusion Models (Sohl-Dickstein et al., 2015; Ho et al., 2020; Kingma et al., 2021; Kingma & Gao, 2023), Discrete Diffusion Models can be trained by minimizing an expected Negative Evidence Lower Bound (NELBO). Specifically, one can first derive an expression for the standard discrete-time expected NELBO with n noise levels:

$$
L _ { n } ( x _ { 0 } ; \theta ) = \mathbb { E } \left[ - \log \widehat { p } _ { 0 | t _ { 1 } } ^ { \theta } ( x _ { 0 } \mid x _ { t _ { 1 } } ) + \sum _ { i = 2 } ^ { n } \operatorname { K L } \left( p _ { t _ { i - 1 } | 0 , t _ { i } } ( \cdot \mid x _ { 0 } , x _ { t _ { i } } ) \parallel \widehat { p } _ { t _ { i - 1 } | t _ { i } } ^ { \theta } ( \cdot \mid x _ { t _ { i } } ) \right) \right] ,\tag{26}
$$

and find the limit of (26) as $n  \infty$ to obtain a continuous-time objective. For the absorbing corruption processes, the limit of (26) converges to (Ou et al., 2025; Sahoo et al., 2024; Shi et al., 2024):

$$
L _ { \infty } ^ { \mathrm { m a s k } } ( x _ { 0 } ; \theta ) = - \int _ { 0 } ^ { 1 } \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } \mathbb { E } _ { p _ { t | 0 } } \left[ e _ { x _ { 0 } } ^ { \top } \log \mathbf { x } _ { \theta } ( t , x _ { t } ) \right] \mathrm { d } t .
$$

With uniform corruption, (26) converges to (Schiff et al., 2025; Sahoo et al., 2025):

$$
\begin{array} { r l r } {  { L _ { \infty } ^ { \mathrm { m i f } } ( x _ { 0 } ; \theta ) = - \int _ { 0 } ^ { 1 } \frac { \alpha _ { t } ^ { \prime } } { N \alpha _ { t } } \mathbb { E } _ { p _ { t \mathrm { i } } \mathrm { [ m } } [ \frac { N } { \bar { x } _ { x _ { t } } } - \frac { N } { e _ { x _ { t } } ^ { \top } { \bf x } _ { \theta } ( t , x _ { t } ) }  } } \\ & { } & {  - ( \kappa _ { t } \mathbb { I } _ { x _ { t } = x _ { 0 } } + \mathbb { I } _ { x _ { t } \neq x _ { 0 } } ) ( N \log ( e _ { x _ { t } } ^ { \top } { \bf x } _ { \theta } ( t , x _ { t } ) ) - { \bf 1 } ^ { \top } \log { \bf x } _ { \theta } ( t , x _ { t } ) )  } \\ & { } & {  - N \frac { \alpha _ { t } } { 1 - \alpha _ { t } } ( \log ( e _ { x _ { t } } ^ { \top } { \bf x } _ { \theta } ( t , x _ { t } ) ) - \log ( e _ { x _ { 0 } } ^ { \top } { \bf x } _ { \theta } ( t , x _ { t } ) ) ) \mathbb { I } _ { x _ { t } \neq x _ { 0 } }  } \\ & { } & {  - ( ( N - 1 ) \kappa _ { t } \mathbb { I } _ { x _ { t } = x _ { 0 } } - \frac { 1 } { \kappa _ { t } } \mathbb { I } _ { x _ { t } \neq x _ { 0 } } ) \log { \kappa _ { t } } ] \mathrm { d } t , } \end{array}
$$

where $\begin{array} { r } { \kappa _ { t } : = \frac { 1 - \alpha _ { t } } { N \alpha _ { t } + 1 - \alpha _ { t } } } \end{array}$ and I is the indicator function.

## I.2 PREDICTOR-CORRECTOR SAMPLER

Our PC baselines for MDMs and UDMs use a pure predict-and-renoise sampler. Instead of the ancestral transition $\hat { p } _ { s \vert t } ^ { \theta } .$ , each step (1) samples $\scriptstyle { \hat { x } } _ { 0 }$ from the plug-in posterior at target time 0, i.e., the bridge (24) with $s \ = \ 0$ and $e _ { x _ { 0 } }$ replaced by the denoiser prediction, and (2) re-applies the forward process (1) to $\scriptstyle { \hat { x } } _ { 0 }$ at the next time. Let $( t _ { i } ) _ { i = 0 } ^ { n }$ be the sampling time grid (Section J.1), and let $\mathbf { x } _ { \theta } ( t , x _ { t } )$ denote the denoiser, with logits scaled by the temperature $T$ of (25). Starting from $x _ { t _ { n } } \sim \operatorname { C a t } ( \pi )$ , we repeat for $i = n , \ldots , 1 ;$

(Predict)

(Re-noise)

$$
\begin{array} { r l } & { \hat { x } _ { 0 } \sim \mathrm { C a t } \big ( { \mathbf { x } _ { \theta } } ( t _ { i } , x _ { t _ { i } } ) \big ) , } \\ & { x _ { t _ { i - 1 } } \sim p _ { t _ { i - 1 } | 0 } ( \cdot \mid \hat { x } _ { 0 } ) = \mathrm { C a t } \big ( \alpha _ { t _ { i - 1 } } e _ { \hat { x } _ { 0 } } + ( 1 - \alpha _ { t _ { i - 1 } } ) \pi \big ) , } \end{array}
$$

independently for each position, and we return $\scriptstyle { \hat { x } } _ { 0 }$ at the last step $( t _ { 0 } = 0 , \alpha _ { t _ { 0 } } = 1 )$ . The sampler therefore uses the bridge $p _ { s | 0 , t } ( x _ { s } \mid x _ { 0 } , x _ { t } ) = p _ { s | 0 } ( x _ { s } \mid x _ { 0 } )$ with the plug-in prediction, as in starshaped diffusion (Okhotin et al., 2023). Every step discards $\boldsymbol { x } _ { t _ { i } }$ except through the prediction $\scriptstyle { \hat { x } } _ { 0 }$ For $\mathbf { M D M s } ,$ this means that tokens unmasked at earlier steps can be masked again, which lets the sampler revise earlier choices.

## I.3 SELF-CONDITIONING AND LOOPHOLING

Because Discrete Diffusion models operate directly on discrete state spaces, the rich categorical distribution predicted by the denoiser collapses into a single discrete token value at each sampling step. Consequently, sampling discards the uncertainty of the denoiser. Therefore, it is common to resort to Self-Conditioning $( \mathrm { S C } ;$ Chen et al., 2022) or Loopholing (Jo et al., 2026) to propagate continuous information across sampling steps. We implement SC following Jo et al. (2026).

Architecture. We decompose the denoiser into three parts. An embedding layer $e ( \cdot )$ maps the input state $x _ { t }$ to one vector per position. A backbone $f _ { \theta }$ maps the embedded input and the time t to the last hidden representation h, which we call the latent. An output head $g$ (a linear projection followed by a softmax) maps h to the predicted distribution over clean tokens. Without SC, the denoiser computes

$$
h = f _ { \theta } \big ( e ( x _ { t } ) , t \big ) , \qquad \mathbf { x } _ { \theta } ( t , x _ { t } ) = g ( h ) .
$$

With ${ \mathrm { S C } } ,$ the denoiser additionally receives a latent $h ^ { \mathrm { p r e v } }$ and adds it to the input embedding after a LayerNorm $\operatorname { L N } ( \cdot )$

$$
h = f _ { \theta } \big ( e ( x _ { t } ) + \mathrm { L N } ( h ^ { \mathrm { p r e v } } ) , t \big ) , \qquad \mathbf { x } _ { \theta } ( t , x _ { t } , h ^ { \mathrm { p r e v } } ) = g ( h ) .\tag{27}
$$

Setting $h ^ { \mathrm { p r e v } } = 0$ recovers a denoiser without context. Following Jo et al. (2026), we initialize the scale and shift of LN to zero, so that SC initially leaves the input unchanged.

Training. Let $\operatorname { s g } ( \cdot )$ denote the stop-gradient operator, which acts as the identity in the forward pass and blocks gradients in the backward pass. With probability $p _ { \mathrm { S C } } = 0 . 9$ , we apply SC: a first forward pass with $h ^ { \mathrm { p r e v } } = 0$ and without gradient produces a latent $h ^ { 0 }$ , and a second forward pass (27) with $h ^ { \mathrm { p r e v } } = \operatorname { s g } ( h ^ { 0 } )$ produces the prediction on which we compute the loss. With probability $1 - p _ { \mathsf { S C } }$ , we use a single forward pass with $h ^ { \mathrm { p r e v } } = 0$ . This avoids unrolling the sampling trajectory during training.

Matching the Training FLOPs. We count the cost of a forward pass as 1 and that of a forward and backward pass as 3. A training step with SC then costs on average $3 + p _ { \mathrm { S C } } \cdot 1 = 3 + 0 . 9 = 3 . 9$ forward-equivalents, against 3 without SC. To match the training FLOPs of models trained without ${ \mathrm { S C } } ,$ we train SC models for a fraction $3 / 3 . 9 \approx 0 . 7 7$ of the training steps.

Sampling. At the first sampling step, $h ^ { \mathrm { p r e v } } = 0$ . The latent h computed at step $t _ { i }$ is then passed as $h ^ { \mathrm { p r e v } }$ to step $t _ { i - 1 }$ . Next to the sampled discrete state, each step thus carries a deterministic continuous state. This adds one LayerNorm and one addition per step, and no extra forward pass.

## I.4 DIRICHLET FLOW MATCHING

This section contains additional background on Euclidean and Dirichlet Flow Matching. To ensure consistency with the Discrete Diffusion literature, we denote the noise distribution by $p _ { 1 }$ and the empirical data distribution by $p _ { 0 }$

Continuous Normalizing Flows. Continuous Normalizing Flows (CNFs; Chen et al., 2018; Grathwohl et al., 2018) are generative models on $\mathbb { R } ^ { d }$ that transport samples from a tractable prior $p _ { 1 } ~ = ~ p _ { \mathrm { n o i s e } }$ to an unknown data distribution $p _ { 0 } ~ = ~ p _ { \mathrm { d a t a } }$ . To match the rest of the paper, we reverse the usual CNF time convention, in which $t = 0$ is noise. The time-dependent velocity field $u _ { t } ^ { \theta } \colon  { \mathbb { R } } ^ { d } \to  { \mathbb { R } } ^ { d }$ induces a flow map $\phi _ { t } \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ via the Ordinary Differential Equation (ODE):

$$
\frac { \mathrm { d } x _ { t } } { \mathrm { d } t } = u _ { t } ^ { \theta } ( x _ { t } ) , \quad x _ { 1 } \sim p _ { 1 } ,\tag{28}
$$

where $\theta$ parameterizes a neural network. Integrating (28) from $t = 1 \mathrm { t o } t = 0$ maps approximately noise $x _ { 1 }$ to data $x _ { 0 } = \phi _ { 0 } ( x _ { 1 } )$ q, defining intermediate densities $p _ { t } = [ \phi _ { t } ] _ { \sharp } p _ { 1 }$ along the probability path $\{ p _ { t } \} _ { t \in [ 0 , 1 ] }$

Flow Matching. Rather than leaving the velocity field $u _ { t } ^ { \theta }$ unconstrained and optimizing θ through expensive ODE simulation, Flow Matching (FM) constructs a generative process using conditional probability paths $p _ { t \mid 0 } ( x _ { t } \mid x _ { 0 } )$ and conditional velocity fields $u _ { t \mid 0 } ( x _ { t } \mid x _ { 0 } )$ , conditioned on clean data samples $x _ { 0 } \sim p _ { 0 }$ . From this conditional pair, we define

$$
p _ { t } ( x _ { t } ) = \int p _ { t | 0 } ( x _ { t } \mid x _ { 0 } ) p _ { 0 } ( x _ { 0 } ) \mathrm { d } x _ { 0 } .
$$

and

$$
u _ { t } ( x _ { t } ) = \int u _ { t } ( x _ { t } \mid x _ { 0 } ) p _ { 0 \mid t } ( x _ { 0 } \mid x _ { t } ) \mathrm { d } x _ { 0 } = \int u _ { t } ( x _ { t } \mid x _ { 0 } ) { \frac { p _ { t \mid 0 } ( x _ { t } \mid x _ { 0 } ) p _ { 0 } ( x _ { 0 } ) } { p _ { t } ( x _ { t } ) } } d x _ { 0 } .\tag{29}
$$

It can then be shown that these quantities (see e.g., Lipman et al. (2023) ) satisfy

$$
\frac { \partial p _ { t } ( x ) } { \partial t } + \nabla \cdot \big ( p _ { t } ( x ) u _ { t } ( x ) \big ) = 0 .\tag{30}
$$

Satisfying (30) ensures that integrating an ODE of drift $u _ { t }$ from t “ 1 to $t = t ^ { \prime }$ produces a sample $x _ { t ^ { \prime } } \sim p _ { t ^ { \prime } }$

Extension to the Simplex. Dirichlet Flow Matching (DFM; Stark et al., 2024) extends FM to the simplex $\Delta _ { N }$ to model categorical data. In particular, DFM tackles the issue of contracting support on $\Delta _ { N }$ by defining conditional probability paths with full support:

$$
p _ { t | 0 } ( P _ { t } | P _ { 0 } ) = \operatorname { D i r } \big ( P _ { t } ; { \mathbf { 1 } } + h _ { t } P _ { 0 } \big ) .
$$

Stark et al. (2024) show that when paired with the following conditional velocity field $u _ { t \vert 0 }$ $( p _ { t | 0 } , u _ { t | 0 } )$ satisfy the continuity equation. For $P _ { 0 }$ the one-hot embedding $e _ { i }$ of category $i ,$

$$
\begin{array} { l } { { u _ { t \vert 0 } ( P _ { t } \vert P _ { 0 } ) = C ( P _ { t , i } , t ) ( e _ { i } - P _ { t } ) , } } \\ { { \displaystyle C ( P _ { t , i } , t ) = - \tilde { I } _ { P _ { t , i } } ( t + 1 , N - 1 ) \frac { \mathcal { B } ( t + 1 , N - 1 ) } { ( 1 - P _ { t , i } ) ^ { N - 1 } P _ { t , i } ^ { t } } , } } \end{array}\tag{31}
$$

where B denotes the beta function, $\begin{array} { r } { \tilde { I } _ { x } ( a , b ) = \frac { \hat { \sigma } } { \hat { \sigma } a } I _ { x } ( a , b ) } \end{array}$ the partial derivative of the regularized incomplete beta function, and N the number of categories. Like us, Stark et al. (2024) train a denoiser $p _ { 0 | t } ^ { \theta } ( x _ { 0 } | x _ { t } ) : ( \Delta _ { N } ) ^ { L } \mapsto ( \Delta _ { N } ) ^ { L }$ with Cross-Entropy. However, their sampling dynamics differs, as they marginalize the conditional as in (29), replacing the true posterior by the learned one. Therefore Stark et al. (2024) propose a deterministic sampler, while ours is not, even in the case κ “ 0 (Section 3).

## Part IV: Experimental Setup, Ablations and Results

## J EXTENDED EXPERIMENTAL DETAILS

## J.1 TIME SAMPLERS AND SAMPLING SCHEDULES

We distinguish two choices. The time sampler is the distribution of t during training: uniform $( ^ { \dag } )$ or adaptive $( \mp )$ . The sampling schedule is the time grid used at inference: linear, cosine or adaptive. The two can be combined freely, except that the adaptive schedule requires the adaptive time sampler, since it reuses the loss profile fitted during training.

Motivation. By default, recent Discrete Diffusion draws the time t uniformly from $t \in [ 0 , 1 ]$ during training. However, recent continuous diffusion language models require adaptive time samplers (Dieleman et al., 2022; Pynadath et al., 2026a; Batzolis et al., 2026; Chemseddine et al., 2026; Chen et al., 2026; Deschenaux & Gulcehre, 2026; Lee et al., 2026; Potaptchik et al., 2026; Raya et al., 2026; Roos et al., 2026; Yang et al., 2026) to perform well. During training, we use the derivative of the loss profile $t \mapsto { \frac { \mathrm { d } } { \mathrm { d } t } } L _ { t }$ as a proxy for where the network learns the most. Assuming that the true loss increases monotonically with t, the derivative $\begin{array} { r } { \frac { \mathrm { d } L _ { t } } { \mathrm { d } t } \geqslant 0 } \end{array}$ directly defines an importance density $q ( t ) \propto \frac { \mathrm { d } L _ { t } } { \mathrm { d } t }$

High-Level Algorithm. We maintain a ring buffer to store recent $( t , L _ { t } )$ pairs. During the first 1k training steps, we start with $t \sim \mathcal { U } ( 0 , 1 )$ to fill the buffer. Afterwards, every 50 training steps, we approximate the loss profile using the content of the ring buffer, with either B-splines (Cox, 1972; de Boor, 1972) or with the mean per bucket, and differentiate via finite differences to obtain the empirical density $\begin{array} { r } { \hat { q } ( t ) \propto \frac { \mathrm { d } L _ { t } } { \mathrm { d } t } } \end{array}$ . We track an $\mathrm { E M A } \ \hat { q } _ { \mathrm { E M A } }$ of the successive densities $\hat { q }$ with momentum 0.9 for stability. Let $\hat { F } _ { \mathrm { E M A } }$ denote the CDF obtained by numerically integrating qˆ<sub>EMA</sub>. Finally, we evaluate $\hat { F } _ { \mathrm { E M A } }$ on a fine grid of 1k values and store $( \hat { F } _ { \mathrm { E M A } } ( t ) , t )$ . We evaluate $t = \hat { F } _ { \mathrm { E M A } } ^ { - 1 } ( u )$ with linear interpolation for $u \sim \mathcal { U } ( 0 , 1 )$ during training (inverse transform sampling).

Estimating the Loss Profile. Deschenaux & Gulcehre (2026) estimates $L _ { t }$ by fitting a global cubic B-spline with ridge regression. In preliminary experiments, we found that a simpler bucketed piecewise-linear estimator led to stronger performance. Therefore, we partition r0, 1s into $B = 5 0$ uniform bins, compute the mean loss within each bin independently, and linearly interpolate, placing the estimated means at the center of each bin. We compute the density as $\begin{array} { r } { \hat { q } ( t ) = \mathbf { \dot { m } } \mathbf { a x } \left( 0 , \frac { \mathrm { d } L } { \mathrm { d } t } \right) } \end{array}$ . While the true $\textstyle { \frac { \mathrm { d } L } { \mathrm { d } t } }$ should be non-negative in principle, max removes numerical artifacts that would make the density negative.

Sampling Time Schedules. We define the sampling time grid as $0 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { n } = 1$ and compare three schedules. The first is the linear grid $\begin{array} { r } { t _ { i } ~ = ~ \frac { i } { n } . } \end{array}$ . The second is the cosine grid $\begin{array} { r } { t _ { i } = \cos \left( \frac { \pi } { 2 } \left( 1 - \frac { i } { n } \right) \right) } \end{array}$ (Chang et al., 2022; Shi et al., 2024). The third is an adaptive grid that reuses the loss profile fitted during training. Specifically, we take the final CDF $\hat { F } _ { \mathrm { E M A } }$ of the adaptive time sampler, stored as a lookup table of 1k points, and place the grid at uniform quantiles:

$$
\begin{array} { r } { t _ { i } = \hat { F } _ { \mathrm { E M A } } ^ { - 1 } \left( \frac { i } { n } \right) , \qquad i = 0 , \dots , n , } \end{array}
$$

evaluated by linear interpolation, exactly as during training. Since $\begin{array} { r } { \hat { q } _ { \mathrm { E M A } } \propto \operatorname* { m a x } ( 0 , \frac { \mathrm { d } L _ { t } } { \mathrm { d } t } ) } \end{array}$ , the grid spends more denoising steps on the noise levels where the loss changes fastest. These are the same noise levels at which the adaptive sampler concentrates training. The grid needs no extra computation or tuning at inference.

## J.2 DENOISER INPUT FOR SDMS

The SDM denoiser receives a sequence of simplex states $P _ { t } ~ = ~ ( P _ { t } ^ { 1 } , \ldots , P _ { t } ^ { L } ) \in ( \Delta _ { N } ) ^ { L }$ (Section C.1). As in Section I.3, an embedding layer e maps the input to one vector in $\mathbb { R } ^ { d }$ per position, a backbone $f _ { \theta }$ maps the embedded sequence and the time t to the last hidden representation h, and an output head g maps h to the predicted distribution $\hat { P } _ { \theta } ( t , P _ { t } )$ . Let $E \in \mathbb { R } ^ { N \times d }$ be the input embedding matrix, whose i-th row $E _ { i }$ embeds token $i \in \mathcal { X }$ . We consider two embeddings of $\dot { P _ { t } } .$ , each with or without Self-Conditioning.

Expectation. We feed the expected embedding

$$
\begin{array} { r } { e ( P _ { t } ) ^ { \ell } = ( P _ { t } ^ { \ell } ) ^ { \top } E = \sum _ { i = 1 } ^ { N } P _ { t , i } ^ { \ell } E _ { i } = \mathbb { E } _ { X \sim \mathrm { C a t } ( P _ { t } ^ { \ell } ) } \big [ E _ { X } \big ] . } \end{array}\tag{32}
$$

This input uses the full belief state: two states with the same most likely token but different uncertainty can map to different inputs. Note that we are not ensured that we do not necessarily have that $e ( P ) { \overset { } { = } } e ( Q )$ implies $P = { \dot { Q } } .$ , i.e. the embedding is not necessarily injective. Since $P _ { t } ^ { \ell }$ has strictly positive entries almost surely, (32) requires a dense product with $E ,$ which costs 2N d FLOPs per token, as much as the output projection. On TinyGSM (N « 49k, $d = 7 6 8 )$ , this adds « 29% to the FLOPs of a forward pass of our 12-layer DiT.

Argmax. To avoid this product, we embed only the most likely token, both during training and at sampling:

$$
\begin{array} { r } { e ( P _ { t } ) ^ { \ell } = E _ { \hat { x } _ { t } ^ { \ell } } , \qquad \hat { x } _ { t } ^ { \ell } = \mathrm { a r g m a x } _ { i } P _ { t , i } ^ { \ell } , } \end{array}
$$

which is a table lookup, as for MDMs and UDMs. The denoiser then sees only $\hat { x } _ { t } ^ { \ell } .$ , but the sampler is unchanged: the transition of Proposition 3.3 still acts on the full state $P _ { t }$ , so the belief state is still carried from step to step.

Self-Conditioning. We also combine both inputs with SC, as described in Section I.3, with the simplex state $P _ { t }$ in place of $x _ { t } \colon$ the backbone receives $e ( P _ { t } ) + \mathrm { L N } ( h ^ { \mathrm { p r e v } } )$ . SC gives the denoiser a rich summary of its previous prediction, which complements the argmax input in particular (Argmax + SC).

Which Input Is Used Where. On Sudoku, SDMs use the expectation input, with and without SC (Table 1). On TinyGSM, we report the expectation, Argmax and Argmax $+ \ S C$ inputs (Tables 29 to 32).

## J.3 DIRICHLET FLOW MATCHING BASELINE

To understand the benefits of our simplex based method, we also compared to the Dirichlet Flow Matching (DFM) method, as described in Section I.4. The original implementation provided by the authors at https: $/ / { \mathfrak { g } } \mathrm { : }$ ithub.com/HannesStark/dirichlet-flow-matching does not include the ability to run on any of the datasets we experimented on. Furthermore, the original implementation and experiments deal with small vocab sizes (only up to 160 categories) whereas our text experiments use up to 50k tokens. We therefore re-implemented and extended the DFM method to provide a fair comparison to our approach. We made two key changes to DFM. The first was to use the adaptive time sampler during training which we found helped performance on TinyGSM for our method. In the original presentation in Stark et al. (2024), eq (14) presents the corruption distribution as (in the authors’ original notation)

$$
p _ { t } ( \mathbf { x } | \mathbf { x } _ { 1 } = \mathbf { e } _ { i } ) = \mathrm { D i r } ( \mathbf { x } ; \alpha = \mathbf { 1 } + t \cdot \mathbf { e } _ { i } ) .
$$

In the original implementation during training, t is sampled as an exponential random variable with scale parameter $\begin{array} { r } { \dot { \alpha _ { \mathrm { s c a l e } } } , t \sim \mathrm { E x p } ( \frac { 1 } { \alpha _ { \mathrm { s c a l e } } } ) } \end{array}$ . We instead sample t with the adaptive time sampler described in Section J.1.

Secondly, we improved the sampling implementation to handle much larger numbers of categories than the original implementation. Sampling in DFM requires integrating the conditional velocity field (31) towards target vertices $e _ { i }$ on the simplex $\Delta _ { N }$ . Writing $b = P _ { t , i } \in [ 0 , 1 ]$ for the coordinate along category i, the velocity is $u _ { t \mid 0 } ( P _ { t } \mid P _ { 0 } = e _ { i } ) = C ( b , t ) ( e _ { i } - P _ { t } )$ with

$$
C ( b , t ) = - \tilde { I } _ { b } ( t + 1 , N - 1 ) \frac { { \cal B } ( t + 1 , N - 1 ) } { ( 1 - b ) ^ { N - 1 } b ^ { t } } ,\tag{33}
$$

where B and $\tilde { I }$ are defined in Section I.4. Since the first argument is $t + 1$ , we have $\tilde { I } _ { b } ( t { + } 1 , N { - } 1 ) =$ $\begin{array} { r } { \frac { \partial } { \partial t } J _ { b } ( t + 1 , N - 1 ) } \end{array}$

In the original implementation of Stark et al. (2024), $\tilde { I } _ { b } ( t { + } 1 , N { - } 1 )$ is calculated through computing $I _ { b } ( t + 1 , N - 1 )$ at linearly spaced intervals of $b \in [ 0 , 1 ]$ and $t \in [ t _ { \operatorname* { m i n } } , t _ { \operatorname* { m a x } } ]$ and using numerical differences to approximate the gradient. This breaks down for large vocab sizes because if we consider a sample from the uniform prior Dirp1q then the coordinate b is distributed according to Beta $1 , N - 1 )$ which has mean $1 / \dot { N } \approx 2 \times \mathrm { \dot { 1 0 } ^ { - 5 } }$ Therefore we need more precision in this b range than a linear spacing of b would provide. The original implementation used $\Delta b \ = \ 1 0 ^ { - 3 }$ which immediately skips over the high probability region at $2 \times 1 0 ^ { - 5 }$ for large vocabulary sizes. We instead use a non-uniform geometric discretization grid that is denser around the prior mean $1 / N$

$$
b _ { k } = b _ { \mathrm { m i n } } \left( \frac { b _ { \mathrm { m a x } } } { b _ { \mathrm { m i n } } } \right) ^ { \frac { k } { M - 1 } }
$$

with $M = 1 0 0 0 , b _ { \mathrm { m i n } } = 1 0 ^ { - 7 }$ and $b _ { \mathrm { m a x } } = 0 . 9 9 9$

Furthermore, the original implementation uses scipy.special.betainc to provide the values of $I _ { b } ( t + 1 , N - 1 )$ . However, for $I _ { b }$ close to $1 . 0 ,$ values quickly round to exactly 1.00 when represented as floats thus giving 0 numerical gradient. We avoid this by using the complementary Beta CDF function $\mathtt { s c i p y }$ .special.betaincc to compute gradients in the regime where $I _ { b } >$ 0.5 because this keeps numerical values closer to 0.0 where they have more precision.

Finally, as b gets larger in (33) $\tilde { I } _ { b }  0$ while $\frac { 1 } { ( 1 - b ) ^ { N - 1 } } \to \infty$ . These two effects should approxi mately cancel leaving a well-conditioned $C ( b , { \dot { t } } )$ however when the terms are represented numerically, underflow and overflow can result in attempting to compute $0 \times \infty$ . To avoid this situation, we can reformulate $C ( b , t )$ into an exact integral where $\zeta 1 - b ) ^ { N - 1 }$ is canceled analytically

$$
\begin{array} { r l } { C ( b , t ) = } & { \frac { B ( t + 1 , N - 1 ) } { \left( 1 - b \right) ^ { N - 1 / 2 } } \frac { \partial } { \partial t } \left[ \frac { 1 } { B ( t + 1 , N - 1 ) } \int _ { \mathbb { R } } ^ { 1 } s ^ { \prime } ( 1 - s ) ^ { N - 2 } d s \right] } \\ & { - \frac { 1 } { \left( 1 - b \right) ^ { N - 1 / 2 } } \left[ \int _ { \mathbb { R } } ^ { 1 } s ^ { \prime } ( 1 - s ) ^ { \mathrm { i } \cdot \nabla - 2 \mathrm { i } \mathrm { i } \cdot \tilde { N } _ { i } ( s ) } d s } \\ & { - \frac { \frac { 1 } { \delta } } { B ( t + 1 , N - 1 ) } \int _ { 0 } ^ { 1 } s ^ { \prime } ( 1 - s ) ^ { N - 2 } d s \right] } \\ & { - \frac { 1 } { \left( 1 - b \right) ^ { N - 1 / 2 } } \int _ { 0 } ^ { 1 } s ^ { \prime } ( 1 - s ) ^ { N - 2 } \left| \psi ( t + N ) - \psi ( t + 1 ) + \mathrm { i } \mathrm { i } \mathrm { i } ( s ( s ) ) \right| d s } \\ & { - \frac { 1 } { \left( 1 - b \right) ^ { N - 1 / 2 } } \int _ { 0 } ^ { 1 } \left( \left( 1 - b \right) \psi ^ { ( 1 } \left( 1 - s \right) ^ { N - 1 } \left( 1 - s \right) ^ { N - 2 } \right. } \\ & { \left. - \frac { 1 } { \left( 1 - b \right) ^ { N - 1 / 2 } } \int _ { 0 } ^ { 1 } \left( s \right) \left( 1 - \left. \mathrm { e } \right) \psi ^ { ( 1 } \left( 1 - s \right) ^ { N - 1 } \left( 1 - s \right) ^ { N - 2 } \right. \right. } \\ & { \left. \left. \int _ { 0 } ^ { 1 } \left( 1 + \frac { 1 } { b } \right) ^ { N - 1 } \psi ^ { ( 1 ) } \left( 1 - s \right) ^ { N / 2 } + \ln ( b + \left( 1 - s \right) \right) \mathrm { d } u \right. } \end{array}\tag{34}
$$

(35)

(36)

where $\begin{array} { r } { \psi ( z ) = \frac { \mathrm { d } } { \mathrm { d } z } \ln \Gamma ( z ) = \frac { \Gamma ^ { \prime } ( z ) } { \Gamma ( z ) } } \end{array}$ is the digamma function. In the above derivation, we have used in (34) that $\frac { \partial } { \partial t }$ ln $\boldsymbol { B } ( t + 1 , N - 1 ) = \boldsymbol { \psi } ( t + 1 ) - \boldsymbol { \psi } ( t + N )$ , and in (35) the substitution $s = b + ( 1 - b ) u$ allowing $( 1 - b ) ^ { N - 1 }$ to cancel in (36). For $b \gtrsim 0 . 0 1$ , we numerically integrate this integral using 48-node Gauss-Legendre quadrature instead of the numerical gradient approach. For numerical integration, we make the substitution $( 1 - u ) ^ { N - 2 } = e ^ { - w } \implies u = 1 - e ^ { \frac { - w } { N - 2 } }$ and integrate over the range $w \in [ 0 , 5 0 ]$ . This prevents standard quadrature on $[ 0 , 1 ]$ from stepping over the narrow region near zero where $\stackrel { \bullet } { ( 1 - u ) } ^ { N - 2 }$ is concentrated before decaying to zero.

These three improvements to the DFM sampling algorithm allow us to safely sample at the 50k vocabulary size scale.

To verify our implementation, we first re-ran the toy experiment from Stark et al. (2024) where the model is tasked to reproduce a synthetic categorical distribution. We train our re-implemented model with 40 categories, sample it and compute the KL-divergence to the ground truth. We obtain a KL-divergence of 0.03 approximately matching the value from Stark et al. (2024, Figure 4).

We integrate the marginal velocity field with the explicit Euler method in α from $\alpha _ { \mathrm { m i n } } = 1 \tan \alpha _ { \mathrm { m a x } } =$ $4 \alpha _ { \mathrm { s c a l e } } .$ , project onto the simplex after every step, and return the argmax of the final prediction. We use the same number of steps as for the other methods (180 on Sudoku; 64 or 512 on TinyGSM), so one step is one NFE.

## J.4 SUDOKU

Data Generation. We generate 9x9 Sudoku puzzles with 30 visible cells using the backtracking generator of Alp (2024). We produce partial grids by iteratively removing cells and verifying that the solution remains unique. We ensure that the training and validation sets share no full grids. We use 200k training and 5k validation puzzles (instead of the 48k/2k split in Deschenaux & Gulcehre (2026); Kim et al. (2026b)) since we observed mild overfitting in preliminary experiments, especially when using the adaptive time sampler. We report the exact-match accuracy on the validation set.

Tokenization. We represent each puzzle as a sequence of 180 tokens drawn from a vocabulary of size 14. We map digits $\{ 1 , \ldots , 9 \}$ directly to their numerical values, empty cells to ID 0, row separators | to ID 10, sequence boundaries [BOS] to ID 11, (unused) padding to ID 12, and the [MASK] to ID 13. To accommodate both autoregressive and diffusion algorithms, we concatenate the unsolved puzzle and its complete solution into a sequence of 180 tokens:

$$
\Big ( [ \mathrm { B o s } ] , 5 , 3 , \ldots , 2 , \ | , \ldots , | , \ , \ldots , 9 , \ [ \mathrm { B o s } ] , 5 , 3 , 4 , \ldots , 2 , \ | , \ldots , | , 3 , \ldots , 9 \Big ) .
$$

Prompt / Unsolved Puzzle p90 tokensq

Target / Complete Solution p90 tokensq

During training, we only corrupt the target tokens and keep the first half clean as conditioning.   
During inference, we start from a clean prompt and denoise only the second half.

Architecture and Hyperparameters. We train an 8-layer Diffusion Transformer (DiT) (Peebles & Xie, 2023) with hidden dimension 512, 8 attention heads, an MLP expansion ratio of 4, 1D Rotary Positional Embeddings (Su et al., 2024), and untied input/output embeddings (28.6M total parameters). We apply a 0.1 dropout rate to the output projection (Sahoo et al., 2024). We train with Adam $( \beta _ { 1 } = 0 . 9 , \bar { \beta } _ { 2 } = 0 . 9 9 9 , \bar { \epsilon } = 1 0 ^ { - 8 } )$ , gradient clipping at maximum norm 1.0, and no weight decay. The learning rate warms up linearly to $3 \times 1 0 ^ { - 4 }$ over 2.5k steps and remains constant afterwards. We maintain an Exponential Moving Average (EMA) of the parameters with decay 0.9999 for evaluation for all methods. We implement time conditioning via Adaptive LayerNorm (AdaLN) (Peebles & Xie, 2023). The time-independent variants (AR and absorbing diffusion) receive a constant (zero) vector in place of time-conditioning, to share the exact same architecture with the other approaches. We train for 50k steps with a global batch size of 256 in full 32-bit precision (« 77% of the steps, i.e. 38,462, for SC variants, to match the training FLOPs; Section I.3).

## J.5 TINYGSM

Tokenization. We train our models on TinyGSM (Liu et al., 2023a), a dataset containing approximately 11.8M synthetic grade-school math word problems associated with executable Python programs producing the correct answer. We evaluate the models zero-shot on the 1319 test problems of GSM8K (Cobbe et al., 2021) by executing a generated program and verifying its numerical output. Following Deschenaux & Gulcehre (2026), we tokenize TinyGSM with the SmolLM tokenizer (Allal et al., 2025) (49k tokens). Unlike GPT-2, SmolLM pretrains on code, and thus its tokenizer compresses Python programs better. Kim et al. (2026a) originally used the Qwen2 tokenizer (Yang et al., 2024), but its 151k tokens induce huge embedding tables, and therefore we chose SmolLM as a compact alternative.

Architecture and Hyperparameters. We train a 12-layer Diffusion Transformer (DiT) (Peebles & Xie, 2023) with hidden dimension 768, 12 attention heads, an MLP expansion ratio of 4, 1D Rotary Positional Embeddings (Su et al., 2024), and untied input and output embeddings (167.9M total parameters). Following Sahoo et al. (2024), we apply a 0.1 dropout rate to the output projection. We optimize all models using Adam (Kingma & Ba, 2014) $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \epsilon = 1 0 ^ { - 8 } )$ without weight decay, clipping gradient norms at 1.0. We linearly warm up the learning rate to $3 \times 1 0 ^ { - 4 }$ over 2,500 steps and hold it constant thereafter. We maintain an Exponential Moving Average (EMA) of the parameters with decay 0.9999 and evaluate the EMA weights across all methods, including the autoregressive baseline. We implement time conditioning via Adaptive LayerNorm (AdaLN) (Peebles & Xie, 2023) with a conditioning dimension of 128. The time-independent models (autoregressive and absorbing Discrete Diffusion) receive a constant zero vector as time conditioning, to keep the architecture the same across experiments. We train each model for 250k steps with a global batch size of 512 in full 32-bit floating-point precision $( \approx ~ 7 7 \%$ of the steps for SC variants, to match the training FLOPs; Section I.3). We train all SDMs with the adaptive time sampler (Section J.1). During evaluation, we run 512 sampling steps for all diffusion variants unless stated otherwise.

AST Diversity Score. We measure how structurally different the K “ 5 programs sampled for each GSM8K problem are by comparing their abstract syntax trees (ASTs). For each sample, we keep the text from the first function definition onward (or the content of a Markdown code block, if present), parse it with Python’s ast module, and discard samples that do not parse. We then normalize each tree: we remove docstrings, rename user-defined variables, functions and arguments to canonical identifiers within each scope (VAR0, VAR1, . . . ) while keeping built-in names such as sum or range, and replace keyword-argument names and import aliases by fixed placeholders. Each node is labeled by its node type; names also carry their canonical identifier, constants their type and value, and arithmetic operators their type.

For two normalized trees $T _ { a }$ and $T _ { b }$ with $\left| T _ { a } \right|$ and $\left| T _ { b } \right|$ nodes, we compute the tree edit distance $\mathrm { T E D } ( T _ { a } , T _ { b } )$ (Zhang & Shasha, 1989) with unit costs for insertion, deletion and relabeling, and define the similarity

$$
\sin ( T _ { a } , T _ { b } ) = \operatorname* { m a x } \Bigl ( 0 , 1 - \frac { \mathrm { T E D } ( T _ { a } , T _ { b } ) } { \operatorname* { m a x } ( | T _ { a } | , | T _ { b } | ) } \Bigr ) \in [ 0 , 1 ] .
$$

For a problem i whose set of parsable samples $\nu _ { i }$ has at least two elements, the diversity is one minus the mean pairwise similarity,

$$
D _ { i } = 1 - \left( \begin{array} { c } { | \mathcal { V } _ { i } | } \\ { 2 } \end{array} \right) ^ { - 1 } \sum _ { a < b \in \mathcal { V } _ { i } } \sin ( T _ { a } , T _ { b } ) .
$$

We report $\begin{array} { r } { 1 0 0 \cdot \frac { 1 } { | \mathcal { T } | } \sum _ { i \in \mathcal { T } } D _ { i } } \end{array}$ , where I is the set of problems with at least two parsable samples; higher is more diverse. AST Div. (Correct) is the same score computed only on the samples whose execution returns the reference answer, averaged over the problems with at least two such samples. It measures whether a model finds structurally different correct solutions, rather than rewarding diversity that comes from incorrect programs.

## J.6 OPENWEBTEXT

Tokenization. We evaluate unconditional language modeling on OpenWebText (OWT) (Gokaslan & Cohen, 2019). Following prior work, we tokenize OWT using the GPT-2 tokenizer (Radford et al., 2019) (50257 tokens) and pack documents into sequences of 1024 tokens. Unlike Sudoku and TinyGSM, which require conditional prefix completion, we corrupt all 1024 positions during training and denoise sequences from scratch during inference. We measure the sample quality with the Generative Perplexity (Gen. PPL) under a pretrained GPT-2-large model and the unigram entropy. Specifically, we produce Pareto curves, following Pynadath et al. (2026a).

Architecture and Hyperparameters. We use the same 12-layer DiT architecture as in TinyGSM. We use the exact same optimizer, learning rate schedule, AdaLN time conditioning, EMA decay rate, and global batch size as well. To follow the academic literature, we train each model for 1M steps. We sample the diffusion models with 64 steps, and the AR model token by token (1024 steps).

## J.7 LANGUAGE UNDERSTANDING

We evaluate our model on language understanding benchmarks such as ARC-easy (Clark et al., 2018), PIQA (Bisk et al., 2020) and HellaSwag (Zellers et al., 2019). The goal of this benchmark is to evaluate the accuracy of a language model to choose the correct continuation when faced with multiple choices for a given context prompt. This is a significant departure from the sample quality metrics that we report for Sudoku, TinyGSM and OWT.

We compare our method to GPT-2 (Radford et al., 2019), the retrained LLaMA baseline (Touvron et al., 2023) of von Rutte et al.¨ (2025), Generalized Interpolating Discrete Diffusion (GIDDs) (von Rutte et al.¨ , 2025), Masked Diffusion Models (MDMs) as reported in Deschenaux & Gulcehre (2025) and Partition Generative Models (PGMs) (Deschenaux et al., 2026b).

In each scenario we are given a context s of length $L _ { s }$ and possible continuations $c ^ { ( 1 ) } , \ldots , c ^ { ( n ) }$ $( n = 2$ in the case of $\mathrm { P I Q A } .$ , and $n = 4$ in the case of ARC-easy and HellaSwag). For a continuation $\boldsymbol { c } = ( c ^ { 1 } , \dots , c ^ { L _ { c } } )$ of length $L _ { c } ,$ we denote by $x = ( s , c ) \in \mathcal { X } ^ { L }$ the full sequence, with $L = L _ { s } + L _ { c } .$ For AR models, the score of a continuation is given by

$$
S _ { \mathrm { A R } } ( s , c ) = \sum _ { \ell = 1 } ^ { L _ { c } } \log p _ { \theta } ( c ^ { \ell } \mid s , c ^ { 1 } , \ldots , c ^ { \ell - 1 } ) .
$$

In the case of MDMs, the score is given by

$$
S _ { \mathrm { M D M } } ( s , c ) = - \sum _ { \ell = 1 } ^ { L } \int _ { 0 } ^ { 1 } \frac { \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } \langle x ^ { \ell } , \log \mathbf { x } _ { \theta } ( t , x _ { t } ) ^ { \ell } \rangle \mathrm { d } t .\tag{37}
$$

In von Rutte et al.¨ (2025), the integral is discretized on a uniform grid with 128 NFE, whereas in Deschenaux & Gulcehre (2025); Deschenaux et al. (2026b) the authors sample uniformly 1024 times in the interval r0, 1s.

In our case, the noising process retains significant information about the initial sample contrary to MDMs. Therefore, we consider an additional regularization term emphasizing that we care about the continuation and not only the whole text plausibility. In practice, we consider

$$
S _ { \mathrm { S D M } } ( s , c ) = \sum _ { \ell = L _ { s } + 1 } ^ { L } \int _ { 0 } ^ { 1 } q _ { \ell } \langle x ^ { \ell } , \log x _ { \theta } ( t , x _ { t } ) ^ { \ell } \rangle \mathrm { d } t - w \sum _ { \ell = L _ { s } + 1 } ^ { L } \int _ { 0 } ^ { 1 } q _ { t } \langle \tilde { x } ^ { \ell } , \log x _ { \theta } ( t , \tilde { x } _ { t } ) ^ { \ell } \rangle \mathrm { d } t ,\tag{38}
$$

where x˜ is the same as x except that the context is replaced by "Answer". Given a score $S ,$ we define the accuracy as follows. Given a context s and possible continuations $c ^ { ( 1 ) } , \ldots , c ^ { ( n ) }$ for this context with ground-truth $c ^ { ( 1 ) }$ , we define the accuracy as $\csc = \delta _ { 1 } \big ( \mathrm { a r g m a x } _ { j \in \{ 1 , \dots , n \} } S ( s , c ^ { ( j ) } ) \big )$ , where we normalize each score by the byte length of its continuation. For each task we report the average accuracy over the test set.

The quantity $q _ { t }$ is the density we consider to reweight the time during training with the ring buffer, see Section J.1 for details. This practice is consistent with the recommendation of Holtzman et al. (2021) who introduced Domain Conditional Pointwise Mutual Information. For completeness, we sweep over $w \in [ 0 , 1 ]$ and remark that $w = 0$ corresponds to disregarding the domain regularization.

We first evaluate the results for a checkpoint obtained while training with the hyper-parameters of Section J.6. However, we note that the training procedure on OWT noises the whole sequence of 1024 tokens. Hence, when we noise only the continuation while keeping the context clean in (38), the model is severely out of distribution. We therefore also train a short adaptation run, which changes how the sequence is corrupted on OWT. To train the adapted model, we initialize the training run with the baseline weights and train the model for 20,000 iterations with global batch size 256 and EMA decay rate 0.999. We consider a learning rate of $1 \times 1 0 ^ { - 4 }$ with cosine decay to 0 after 20,000 iterations. Finally, during the adaptation run, we train on OWT sequences of length 128 and corrupt them as at evaluation: each sequence is split into a clean context and a noised continuation, with lengths $( \boldsymbol { L } _ { s } , \boldsymbol { L } _ { c } )$ drawn from the 1,024 pairs measured on the tokenized evaluation prompts.

## J.8 UNCONDITIONAL MOLECULAR GENERATION

Experimental setup. We evaluate unconditional de novo molecular generation following the Gen-Mol benchmark protocol (Lee et al., 2025) using the SAFE (Sequential Attachment-based Fragment Embedding) molecular representation (Noutahi et al., 2024). We train on the SAFE-GPT dataset (Noutahi et al., 2024) (v2 version) using the SAFE tokenizer, which has a vocabulary size of |V| “ 1,880. All sequences are padded or truncated to a fixed length of $L = 2 5 6$ . In all diffusion experiments, we treat padding tokens as ordinary tokens during forward corruption, reverse denoising, and in the loss, whereas in the autoregressive (AR) baseline, padding tokens are ignored in the cross-entropy loss.

For all experiments, we use a 12-layer BERT (Devlin et al., 2019) transformer $( d _ { \mathrm { m o d e l } } = 7 6 8 .$ , 12 attention heads, MLP dimension 3,072) similar to the 87M-parameter BERT-Base architecture of GenMol (Lee et al., 2025). We use Post-LayerNorm layers as in MaskGIT (Chang et al., 2022), but replace its MLM prediction head with a single zero-initialized linear output layer that is not tied to the input embeddings. For masked diffusion, similar to Sahoo et al. (2024), we do not condition the network on the diffusion time, while for uniform and Simplex diffusion, we study both variants – with and without time conditioning. Time-conditioned variants use Sinusoidal time embedding with SiLU activation and embedding dimension 128, the time conditioning is provided via adaptive layer normalization (AdaLN). Diffusion models use bidirectional attention, while the AR model uses causal attention. We disable dropout on the attention probabilities and apply a hidden dropout rate of 0.1 to the embedding output and to every attention and MLP sublayer output. For Simplex Diffusion, the denoiser receives the expectation embedding (32) of each simplex state. We do not use Self-Conditioning in any of the diffusion models. SDM and masked diffusion are trained with cross-entropy objectives, while uniform diffusion is trained with the ELBO objective.

All models are trained for up to 400k iterations with a global batch size of 2,048 using AdamW $( \mathrm { l r } ~ = ~ 3 ~ \times ~ 1 0 ^ { - 4 }$ with a 2,500-step linear warmup followed by a constant schedule, weight decay $0 . 0 , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \epsilon = \mathrm { \hat { 1 } } 0 ^ { - 8 } )$ , global gradient-norm clipping at 1.0, and an EMA of the parameters with decay 0.9999; all evaluations use the EMA parameters. During training, diffusion times are drawn from $[ 1 0 ^ { - 4 } , 1 ] \cdot$ : by stratified uniform sampling for MDM and UDM, and by the adaptive time sampler of Section J.1 for SDM.

During training, we monitored the evaluation metrics (see below) and noticed that some runs (including AR ones) exhibited collapse (all the evaluation metrics went to zero, while the loss went up). We therefore used early stopping. For each training run, we select the checkpoint with the highest Quality (see below) metric, computed on 1,000 generated molecules. For this evaluation, every method used the standard sampler (see below) with temperature $T ~ = ~ 1$ and 256 sampling steps, and we used $\kappa = 0$ for Simplex diffusion. The checkpoints were saved every 500 steps. In addition, for Simplex Diffusion, we swept over the concentration schedules: Constantlinear $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 8 )$ , Constant-linear $\cdot ( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ , Constantlinear $( \nu _ { 0 } ~ = ~ 0 . 4 , \nu _ { 1 } ~ = ~ 0 . 7 5 , \ell ~ = ~ 0 . 2 )$ q, Constant $( \nu ~ = ~ 0 . 5 )$ , Constant $( \nu ~ = ~ 0 . 2 5 )$ , where the Constant-linear $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ schedule achieved the best Quality metric and is used for all reported Simplex Diffusion evaluations.

Evaluation Metrics. Following Lee et al. (2025), we generate $M = 1 { , } 0 0 0$ molecules from scratch for each of 3 sampling seeds, and report the mean ˘ standard deviation over seeds. Generated SAFE strings are decoded using the post-processing from (Lee et al., 2025). We report four metrics: Validity (percentage of the M generated sequences that decode to chemically valid molecules), Uniqueness (percentage of unique canonical SMILES among valid molecules), Diversity (one minus the mean pairwise Tanimoto similarity of Morgan fingerprints with radius 2 and 2,048 bits, over the unique valid molecules), and Quality (percentage of generated samples that are simultaneously valid, unique and drug-like. Drug-like molecules are defined as those satisfying quantitative estimate of drug-likeness $\mathrm { Q E D } \geqslant 0 . 6$ (Bickerton et al., 2012) and synthetic accessibility $\mathrm { S A } \leqslant 4$ (Ertl & Schuffenhauer, 2009), respectively, following Lee et al. (2025).).

Sampling and Hyperparameter Sweeps. For the autoregressive (AR) baseline, we generate sequences token-by-token using temperature-scaled softmax sampling with temperature T. For all diffusion models, we evaluate sampling budgets of t32, 64, 256, 1024u steps and three sampling schedules: linear, cosine, and adaptive (Section J.1). The detailed results in Tables 14 and 15 use 256 steps.

We use two diffusion sampling strategies: (i) standard (ordinary) sampling (Table 14), where all sequence positions at step t Ñ s are updated solely via the reverse bridge transition kernel using temperature-scaled clean predictions sampled from $\hat { P } _ { 0 } ^ { \ell } \stackrel { } { = } \mathrm { S o f t m a x } ( z _ { \theta } ^ { \ell } ( P _ { t } ) / T )$ (with churn $\kappa \in \ \bar { [ 0 , 1 ] }$ in Proposition 3.3 for SDM); (ii) confidence-based (conf.) sampling (Table 15), where each position $\ell \in \{ 1 , \ldots , L \}$ is scored by its maximum predicted probability $\begin{array} { r } { c _ { \ell } = \operatorname* { m a x } _ { v \in \mathcal { X } } \hat { P } _ { 0 , v } ^ { \ell } . } \end{array}$ At step $t  s ,$ we commit the union of the $\lfloor \alpha _ { s } ^ { p } L \rfloor$ most confident positions (top-fraction schedule with power p) directly to their most likely token arg max<sub>vPX</sub> $\hat { P } _ { 0 , v } ^ { \ell }$ (a simplex vertex for SDM). The remaining positions follow the standard update (i). Here, $\alpha _ { s }$ is defined in (1) and is evaluated at the target time s of the transition from t to s. Note that the end at the step going from $t  s$ we have a mixed of clean and noisy tokens in the sample. In the case of UDM and MDM those samples are in distribution while in the case of SDM they are severely out of distribution. Nevertheless we observe benefits from using this confidence based sampler even without retraining. Additional training of the denoiser with such strategy (e.g. training with a fraction $\alpha _ { t } ^ { p }$ of positions replaced by clean vertices) could, in principle, further improve Quality and remains an interesting avenue for future work. For MDM, unmasked tokens are never re-masked, and when the top-fraction schedule is active, positions outside the committed set stay masked. For SDM and UDM, the committed set is recomputed at every step.

## K ADDITIONAL EXPERIMENTAL RESULTS

## K.1 SUDOKU

We report the full Sudoku results for SDMs with $\kappa \in \{ 0 , 0 . 2 , 1 \}$ in Tables 21 to 23, for the baselines in Table 24, and for our re-implementation of DFM in Table 2. The full tables are in Section M.1.

Churn and Concentration Schedule. SDMs perform best with $\kappa = 1$ . Without ${ \mathrm { S C } } ,$ every concentration schedule reaches 86–92% at $\kappa = 1$ , whereas no schedule exceeds 79.4% at $\kappa = 0 .$ . The best result (91.6%) comes from $\nu _ { t } ^ { \mathrm { { c s t . - l i n . } } }$ with $\nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2$ . With SC, SDMs are robust to the churn: all three values reach 98.6–99.2%.

Comparison with the Baselines. We compare against SDMs with $\nu _ { t } ^ { \mathrm { { c s t . - l i n . } } }$ with $\nu _ { 0 } ~ = ~ 0 . 4 , ~ \nu _ { 1 } ~ =$ $0 . 7 5 , \ \ell \ = \ 0 . 8$ and $\kappa ~ = ~ 1$ , as in Table 1. Without SC, SDMs (88.4%) outperform DFM (76.7%) and FLMs (75.3%), and match S-FLMs (87.3%). Among discrete models without SC, only UDMs with predictor-corrector sampling (95.8%) exceed them. With SC, SDMs reach 99.1% $( \kappa ~ = ~ 1 )$ , on par with the best discrete model (MDM with ${ \mathrm { S C } } .$ 99.3%).

Table 2: Accuracy (%) on Sudoku in 180 steps comparing Dirichlet Flow Matching and our SDM method. SDMs use $\nu _ { t } ^ { \mathrm { c s t . - l i \bar { n } . } }$ with $\nu _ { 0 } ~ = ~ 0 . 4$ $\nu _ { 1 } = 0 . 7 5$ $\ell \ = \ 0 . 8$ , as in Table 1.
<table><tr><td>Model</td><td>Linear</td></tr><tr><td>Continuous</td><td></td></tr><tr><td>DFM (reimpl.)</td><td> $7 6 . 7 _ { + 0 . 7 5 }$ </td></tr><tr><td>SDM  $( \kappa = 1 . 0 )$ </td><td> $8 8 . 4 _ { + 2 . 2 }$ </td></tr><tr><td>SDM  $( + \mathrm { { S C } } ; \kappa = 1 . 0 )$ </td><td> $\underline { { 9 9 . 1 } } _ { + 0 . 2 }$ </td></tr></table>

Adaptive Time Sampler. Without SC, the adaptive time sampler improves ancestral sampling across models: MDMs $( 6 0 . 1 \%  6 9 . 4 \% )$ , UDMs $( 7 3 . 7 \%  8 2 . 1 \% )$ , FLMs $( 7 3 . 4 \%  7 5 . 3 \% )$ S-FLMs (83.5% Ñ 87.3%) and SDMs at $\kappa = 0 ( 7 3 . 9 \%  7 7 . 1 \% )$ . At $\kappa = 1$ , SDMs perform similarly with both samplers (within 3 points). With SC, the adaptive sampler increases the variance across seeds for both SDMs and UDMs, and it degrades predictor-corrector sampling for the discrete baselines. We therefore train SDMs with the uniform time sampler in the main text.

Comparison with DFM. With the improvements of Section J.3, DFM reaches 76.7% on Sudoku, well below SDMs with $\nu _ { t } ^ { \mathrm { { c s t . - l i n . } } }$ with $\nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8$ and κ “ 1 (88.4%, and 99.1% with SC; Table 2).

## K.2 TINYGSM

Full tables are in Section M.2.

Effect of the Churn κ. Tables 29 to 32 report SDM accuracy for $\kappa \in \{ 0 , 0 . 2 , 1 \}$ . The tables cover three sampling grids and several concentration schedules (Section 4.2), as well as the three input variants (expected embedding, Argmax, $\mathrm { A r g m a x } + \mathrm { S C } )$ , two temperatures and two step budgets. Figures 8 to 11 sweep $\kappa \in \{ 0 , 0 . 1 , \ldots , 1 \}$ for two concentration schedules. In all configurations of the tables, $\kappa = 1$ gives the highest accuracy. The gain is largest for the expected-embedding input with the adaptive grid. For $\nu ^ { \mathrm { { c s t . - l i n . } } }$ with $\nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2$ , accuracy rises from 12.6% to 45.8% at $T = 1$ and from 19.3% to 49.0% at $T = 0 . 1$ (512 steps). The trend is not always monotone. In some configurations, $\kappa = 0 . 2$ is below $\kappa = 0$ , mostly with the Argmax inputs at 64 steps (e.g., 15.6% Ñ 10.0% Ñ 22.5% for Argmax, $\nu = 0 . 5 ,$ cosine grid, $T = 1 )$ .

![](images/d695b7edba64071e0ff5480077bb0930b2fa53e3c6a856aa0ea655cb1930fbf0.jpg)  
Churn vs TinyGSM Accuracy (T=1, NFE=512)

Figure 8: Churn κ vs. TinyGSM accuracy at $T = 1 . 0$ with 512 steps for the Linear (left), $C o \mathrm { - }$ sine (center) and Adaptive (right) sampling schedules. We compare two concentration schedules, Const.-Lin. $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2$ , orange) and Constant $( \nu = 0 . 5$ , indigo), and three input variants: Argmax + SC (‹), Argmax (■) and Expectation $P _ { t } \left( \bullet \right) .$ . Horizontal lines show AR decoding (greedy: $6 2 . 6 \% ;$ sampling: 52.6%) and UDM with the Predictor-Corrector (PC) sampler. Accuracy generally increases with κ. With the adaptive schedule, Const.-Lin. with the expectation input rises from 12.6% at $\kappa = 0$ to 45.8% at $\kappa = 1 , 9 . 0$ points above the best UDM with PC (36.8%). Exact values for $\kappa \in \{ 0 , 0 . 2 , 1 \}$ are in Table 29.  
![](images/bda9ba500f65948c1bbb36748a8a80eedaf97c54564d7227d9e1825a0266a90a.jpg)

![](images/fe1fbf738d814589e40d91cebdb1488047ec9464b304bfd2105f8b7c6e32316b.jpg)

![](images/3d08959e94222f3608cce21a39caa0157e3cb8cdc87769505d9570a2b3a5813d.jpg)  
Churn vs TinyGSM Accuracy (T=0.1, NFE=512)  
Figure 9: Churn κ vs. TinyGSM accuracy at $T ~ = ~ 0 . 1$ with 512 steps for the Linear (left), Cosine (center) and Adaptive (right) sampling schedules. We compare two concentration schedules, Const.-Lin. $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2$ , orange) and Constant $( \nu = 0 . 5 ,$ , indigo), and three input variants: Argmax + SC (‹), Argmax (■) and Expectation $P _ { t } \left( \bullet \right)$ . Horizontal lines show AR decoding (greedy: 62.6%; sampling: 52.6%) and UDM with the Predictor-Corrector (PC) sampler. Accuracy generally increases with $\kappa ,$ and $A r g m a x + S C$ is the strongest input under the linear and cosine schedules. $\mathbf { A } \mathbf { t } \kappa = 1$ , Constant with $A r g m a x + S C$ reaches 50.9% (cosine), above the best UDM with PC (45.5%). With the adaptive schedule, Const.-Lin. with the expectation input rises from 19.3% at $\kappa = 0$ to 49.0% at $\kappa = 1$ . Exact values for $\kappa \in \{ 0 , 0 . 2 , 1 \}$ are in Table 31.

Blockwise Generation. We compare full-sequence SDMs with a block-autoregressive variant (Han et al., 2023; Arriola et al., 2025) that generates blocks of 32 tokens from left to right, with the linear, cosine or adaptive grid (Figure 12). Both use $\kappa = 1$ , with the expected-embedding and $\mathrm { \ A r g m a x + \mathrm { S C } }$ inputs. Full-sequence SDMs are more accurate at every NFE budget, despite the stronger left-to-right bias of block generation.

![](images/b169f5c0b57710ff8a86ec0572fe3f78d9f909fe7fc76eaf6e85be8c17f724f1.jpg)

![](images/aeb301e9ab6ddb4232c4b1508f7e708fb9edf94ce97a655b9771d546397c647b.jpg)

![](images/ab545499e9f70de8a47677ce6c540c2946b1c81c41f7e522fd6a2ed269cc81b1.jpg)  
Churn vs TinyGSM Accuracy (T=1, NFE=64)

Figure 10: Churn $\kappa \mathrm { ~ } \mathbf { V } \mathbf { S } .$ TinyGSM accuracy at $T ~ = ~ 1 . 0$ with 64 steps for the Linear (left), Cosine (center) and Adaptive (right) sampling schedules. We compare two concentration schedules, Const.-Lin. $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2$ , orange) and Constant $( \nu = 0 . 5 ,$ , indigo), and three input variants: Argmax + SC (‹), Argmax (■) and Expectation $P _ { t } \left( \bullet \right)$ . Horizontal lines show AR decoding (greedy: 62.6%; sampling: 52.6%) and UDM with the Predictor-Corrector (PC) sampler. With few steps and high temperature, the expectation input combined with the adaptive schedule and $\kappa = 1$ performs best: $\mathtt { C o n s t . - L i n }$ . rises from 10.6% at $\kappa = 0$ to $\mathbf { 3 2 . 7 \% }$ at $\kappa = 1$ and Constant from 11.2% to 31.8%, compared with 14.7% and 13.8% under the linear schedule and 23.7% for the best UDM with PC. Exact values for $\kappa \in \{ 0 , 0 . 2 , 1 \}$ are in Table 30.

![](images/defcff46956f6a1472da17c21ed2867065ed1d4b003d27042c758ff50f61d717.jpg)

![](images/b60581aae867727ddee2ab3e276251f2625dd9d839cfe42df0e00196af7bdd9e.jpg)

![](images/1d55cab38206219e6a0237a456be608227b7c91df1821d3460e46755b31dc242.jpg)  
Churn vs TinyGSM Accuracy (T=0.1, NFE=64)  
Figure 11: Churn κ vs. TinyGSM accuracy at $T ~ = ~ 0 . 1$ with 64 steps for the Linear (left), Cosine (center) and Adaptive (right) sampling schedules. We compare two concentration schedules, Const.-Lin. $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2$ , orange) and Constant $( \nu = 0 . 5$ , indigo), and three input variants: Argmax + SC (‹), Argmax (■) and Expectation P<sub>t</sub> (‚). Horizontal lines show AR decoding (greedy: 62.6%; sampling: 52.6%) and UDM with the Predictor-Corrector (PC) sampler. At κ “ 1, Argmax + SC reaches up to 44.5% (Constant, adaptive), above the best UDM with PC (38.3%). With the adaptive schedule, the expectation input also improves strongly with churn, from 21.8% to 40.1% (Constant) and from 18.9% to 39.5% (Const.-Lin.). Exact values for $\kappa \in \{ 0 , 0 . 2 , 1 \}$ are in Table 32.

![](images/93063489ba46bb94f19eefc8846dda5a9f3c5fbd3bfd9ee023fa6df7267eab0b.jpg)  
Total NFE vs TinyGSM Accuracy Full-Sequence SDM vs Block-by-Block  
Figure 12: Full-sequence vs. block-wise SDM generation on TinyGSM. Accuracy vs. total NFE at $T = 0 . 1 ( \mathrm { l e f t } )$ and $T = 1$ (right) for block-wise generation with block size 32 under the linear, cosine and adaptive grids, compared with full-sequence SDMs (expected-embedding and Argmax + SC inputs, κ “ 1). Block-wise generation is less accurate at every budget.

Adaptive Grid. The adaptive grid (Section J.1) gives the largest gains for SDMs with the expected-embedding input. These gains are largest at small step budgets: \`18.0 points at 64 steps and T “ 1 (Table 30).

Training Sampler and Grid for the Baselines. Tables 25 to 28 report every baseline trained with the uniform (<sup>:</sup>) and the adaptive (<sup>;</sup>) time sampler, and sampled with the linear, cosine and adaptive grids. The adaptive training sampler improves the SC variants in 11 of 12 settings (e.g., UDM + SC from 31.7% to 39.7% at $T = 0 . 1$ with 512 steps) and FLMs in all four settings. It lowers UDM + PC in all four settings (e.g., from 45.5% to 40.1%), consistent with our Sudoku results. For discrete baselines, the adaptive grid changes accuracy by between ´5.8 and \`5.8 points relative to the linear grid. For every baseline, we sweep the training time sampler (uniform or adaptive), the loss (CE or ELBO for MDMs), the sampler (ancestral, PC, SC) and the sampling grid (linear, cosine, and adaptive when trained with the adaptive sampler), at the same temperatures and step budgets as SDMs, and compare

Table 3: Accuracy (%) on TinyGSM (Adaptive schedule) comparing Dirichlet Flow Matching and our SDM method across step budgets and logit temperature T. For SDM, we use the expectation input, $\nu _ { t } ^ { \mathrm { c s t . - l i n . } } ~ ( \nu _ { 0 } ~ =$ $0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ and κ “ 1 (Tables 29 to 32).
<table><tr><td>Model</td><td>64</td><td>512</td></tr><tr><td>T = 1.0</td><td></td><td></td></tr><tr><td>DFM (reimpl.)</td><td>5.4</td><td>6.1</td></tr><tr><td>SDM (κ = 1.0)</td><td>32.7</td><td>45.8</td></tr><tr><td>T = 0.1</td><td></td><td></td></tr><tr><td>DFM (reimpl.)</td><td>12.1</td><td>13.3</td></tr><tr><td>SDM (κ = 1.0)</td><td>39.5</td><td>49.0</td></tr></table>

SDMs against the best of these configurations. SDMs have a larger search space, since we additionally sweep the concentration schedule (4–5 per input) and the churn $\kappa \in \{ 0 , 0 . 2 , 1 \} ;$ however, a single configuration $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 , \kappa = 1$ , adaptive grid) is the best Expectation SDM in three of the four settings.

Comparison with DFM. We trained our re-implemented DFM model (Section J.3) on TinyGSM, using the exact same network architecture and training setup as our method. We again find that DFM performs worse than our method (Table 3).

Comparison with Spherical Flow. Chemseddine et al. (2026) train Spherical Flow (SF) on TinyGSM with the same architecture, tokenizer, context length and training budget as ours, and evaluate it at $T = 1$ . Table 4 compares their reported numbers with SDMs at matched NFE. Both SDMs and SFs benefit from increased stochasticity during generation. With the least stochasticity (ODE for Spherical Flow, κ “ 0 for SDMs), neither exceeds 13%. Without SC, SDMs outperform Spherical Flow with predictor-corrector (PC) sampling with 64 (32.7% vs. 26.9%) and 512 (45.8% vs. 32.4%) NFEs. SDMs without SC perform similarly to Spherical Flow with PC and SC (32.7% vs. 35.6% at 64 NFE, 45.8% vs. 41.7% at 512 NFE).

Table 5: Accuracy (%) of distilled models on TinyGSM across sampling steps $\mathit { \Pi } ( T \ = \ 1 )$ SDMs use the concentration schedule defined in terms of the total variance with $\nu _ { t } ^ { \mathrm { c s t . - l i n . } }$ $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$
<table><tr><td>Model</td><td>8</td><td>32</td><td>64</td><td>128</td></tr><tr><td>IDLM-MDLM (Li et al., 2026)</td><td></td><td>12.8</td><td>14.9</td><td>19.9</td></tr><tr><td>IDLM-Duo (Li et al., 2026)</td><td></td><td>15.4</td><td>19.0</td><td>21.4</td></tr><tr><td>SDM (undistilled)</td><td>6.3</td><td>25.4</td><td>32.7</td><td>35.5</td></tr><tr><td>SDM (distilled)</td><td>32.1</td><td>37.6</td><td>37.8</td><td>39.4</td></tr></table>

## K.3 DISTILLATION ON TINYGSM

Teacher. We distill the SDM trained with the cross-entropy loss (11), the expected-embedding input (no $\mathbf { S } \mathbf { C } ) , \alpha _ { t } \ = \ 1 - \ t ,$ a uniform prior π, the schedule $\nu _ { t } ^ { \mathrm { { c s t . - l i n . } } }$ with $\nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2$ (Section 4.2), and the adaptive time sampler (Section J.1). This is the SDM (Expectation) of Figure 1. We sample distilled models with κ “ 1, the adaptive grid and T “ 1.

Hyperparameters. We sweep 36 configurations (Table 6) and the sampling churn $\kappa \in \{ 0 , 0 . 2 , 1 \}$ We do not hold out a validation split: configurations are evaluated, and the reported one is selected, on the GSM8K test set, so the 32.1% in the main text is a best-of-sweep number. The conclusion does not depend on this selection: with $\kappa = 1$ , all 36 configurations reach between 27.4% and 32.1% pass@1 with 8 steps (median 28.9%), above IDLM with 128 steps (21.4%). The generator loss weight $\lambda _ { \mathrm { g e n } }$ has little to no effect.

Table 4: Accuracy (%) of SDMs and Spherical Flow on TinyGSM at $T = 1$ Spherical Flow numbers come from Chemseddine et al. (2026) (Table 20), who train with the same architecture, tokenizer, context length and training budget as ours. Each PC entry uses their best sampler configuration for that budget. For SDMs, we always use $\nu _ { t } ^ { \mathrm { c s t . - l i n . } } ( \nu _ { 0 } \stackrel { - } { = } 0 . 2 , \nu _ { 1 } = 0 . 5$ ℓ “ 0.2) and the adaptive grid, as in Figure 1.

<table><tr><td>Model</td><td>64NFE</td><td>512 NFE</td></tr><tr><td>Spherical Flow</td><td></td><td></td></tr><tr><td>ODE</td><td>6.1</td><td>6.4</td></tr><tr><td>PC</td><td>26.9</td><td>32.4</td></tr><tr><td>PC + SC</td><td>35.6</td><td>41.7</td></tr><tr><td>SDMs</td><td></td><td></td></tr><tr><td>Expectation (κ = 0)</td><td>10.6</td><td>12.6</td></tr><tr><td>Expectation (κ = 1)</td><td>32.7</td><td>45.8</td></tr><tr><td>Argmax + SC (κ = 1)</td><td>22.4</td><td>36.9</td></tr></table>

Comparison with IDLM. We take the IDLM results from Li et al. (2026) without re-training. They use the same tokenizer, metric (pass@1) and temperature $( T = 1 )$ as us (Table 5). The remaining differences are the distillation objective, the diffusion process (masked vs. simplex) and the teacher.

Accuracy vs. Steps and Diversity. Figure 13 compares distilled and undistilled SDMs across sampling budgets. Table 33 reports the multi-sample accuracy and AST diversity of the baselines. Distillation trades diversity for speed: with 8 NFEs at $T = \dot { 1 }$ , the distilled SDM reaches an AST diversity of 17.5, vs. 35.3 for the undistilled SDM with 512 NFEs and 36.7 for AR.

## K.4 AUTO-GUIDANCE ON TINYGSM

While Classifier-Free Guidance (CFG; Ho & Salimans, 2022) is a standard mechanism for improving sample fidelity in diffusion models, it requires training with conditioning dropout and cannot be applied directly to unconditional generation or pre-trained models without dropout tokens. To sharpen generation quality at inference time without modifying the training objective, we adapt Auto-Guidance (Karras et al., 2024) to Simplex Diffusion Models (SDMs). Specifically, we guide the fully converged model $\theta _ { \mathrm { f i n a l } }$ away from an earlier, unconverged checkpoint $\theta _ { K }$ saved during training. More precisely, we modify the logits and probabilities of the Simplex Diffusion Model as

Table 6: Distilled SDMs on TinyGSM (8 steps, $\kappa = 1$ , adaptive grid, $T = 1 , K = 5$ samples per problem). We sweep the learning rate $\in \ \dot { \{ 3 }  \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 3 \times \mathbf { \bar { 1 0 ^ { - 5 } } } \}$ , the Adam parameters $( \beta _ { 1 } , \beta _ { 2 } ) \in \{ 0 , 0 . 9 \} \times \{ 0 . 9 5 , 0 . 9 9 9 \}$ and the generator loss weight $\lambda _ { \mathrm { g e n } } \in \{ 1 , 2 , 5 \}$ (36 configurations), and report the 25 configurations with highest pass@1.
<table><tr><td>lr</td><td> $\beta _ { 1 }$ </td><td> $\beta _ { 2 }$ </td><td> $\lambda _ { \mathrm { g c n } }$ </td><td>pass@1 (%) ↑</td><td>pass@2 (%) ↑</td><td>pass@5 (%) ↑</td><td>AST Div. ↑</td><td>AST Div. (Correct) ↑</td></tr><tr><td>1e-05</td><td>0.0</td><td>0.95</td><td>1.0</td><td>31.9</td><td>40.0</td><td>48.7</td><td>17.5</td><td>7.5</td></tr><tr><td>1e-05</td><td>0.0</td><td>0.95</td><td>2.0</td><td>31.9</td><td>40.0</td><td>48.7</td><td>17.5</td><td>7.5</td></tr><tr><td>1e-05</td><td>0.0</td><td>0.95</td><td>5.0</td><td>31.9</td><td>40.0</td><td>48.7</td><td>17.5</td><td>7.5</td></tr><tr><td>1e-05</td><td>0.9</td><td>0.999</td><td>1.0</td><td>31.8</td><td>40.0</td><td>49.6</td><td>17.4</td><td>7.6</td></tr><tr><td>1e-05</td><td>0.9</td><td>0.999</td><td>2.0</td><td>31.8</td><td>40.0</td><td>49.6</td><td>17.4</td><td>7.6</td></tr><tr><td>1e-05</td><td>0.9</td><td>0.999</td><td>5.0</td><td>31.7</td><td>40.0</td><td>49.6</td><td>17.4</td><td>7.6</td></tr><tr><td>3e-06</td><td>0.0</td><td>0.999</td><td>2.0</td><td>30.1</td><td>39.6</td><td>49.9</td><td>19.3</td><td>8.7</td></tr><tr><td>3e-06</td><td>0.0</td><td>0.999</td><td>1.0</td><td>29.9</td><td>39.3</td><td>49.8</td><td>19.3</td><td>8.7</td></tr><tr><td>3e-06</td><td>0.0</td><td>0.999</td><td>5.0</td><td>29.9</td><td>39.3</td><td>49.8</td><td>19.3</td><td>8.7</td></tr><tr><td>1e-05</td><td>0.0</td><td>0.999</td><td>1.0</td><td>29.5</td><td>38.1</td><td>48.5</td><td>18.2</td><td>8.3</td></tr><tr><td>1e-05</td><td>0.0</td><td>0.999</td><td>2.0</td><td>29.5</td><td>38.1</td><td>48.5</td><td>18.2</td><td>8.3</td></tr><tr><td>1e-05</td><td>0.0</td><td>0.999</td><td>5.0</td><td>29.5</td><td>38.1</td><td>48.5</td><td>18.2</td><td>8.3</td></tr><tr><td>3e-05</td><td>0.9</td><td>0.95</td><td>1.0</td><td>29.2</td><td>37.1</td><td>46.6</td><td>20.5</td><td>8.1</td></tr><tr><td>3e-05</td><td>0.9</td><td>0.95</td><td>2.0</td><td>29.2</td><td>37.1</td><td>46.6</td><td>20.5</td><td>8.1</td></tr><tr><td>3e-05</td><td>0.9</td><td>0.95</td><td>5.0</td><td>29.2</td><td>37.1</td><td>46.6</td><td>20.5</td><td>8.1</td></tr><tr><td>3e-06</td><td>0.9</td><td>0.999</td><td>1.0</td><td>28.9</td><td>38.3</td><td>48.7</td><td>17.5</td><td>8.4</td></tr><tr><td>3e-06</td><td>0.9</td><td>0.999</td><td>2.0</td><td>28.9</td><td>38.2</td><td>48.7</td><td>17.6</td><td>8.3</td></tr><tr><td>3e-06</td><td>0.9</td><td>0.999</td><td>5.0</td><td>28.9</td><td>38.3</td><td>48.7</td><td>17.5</td><td>8.4</td></tr><tr><td>3e-05</td><td>0.0</td><td>0.95</td><td>1.0</td><td>28.8</td><td>37.0</td><td>44.9</td><td>19.6</td><td>8.4</td></tr><tr><td>3e-05</td><td>0.0</td><td>0.95</td><td>2.0</td><td>28.8</td><td>36.9</td><td>44.9</td><td>19.6</td><td>8.2</td></tr><tr><td>3e-05</td><td>0.0</td><td>0.95</td><td>5.0</td><td>28.8</td><td>36.9</td><td>44.9</td><td>19.6</td><td>8.2</td></tr><tr><td>3e-05</td><td>0.0</td><td>0.999</td><td>1.0</td><td>28.6</td><td>35.8</td><td>45.3</td><td>19.6</td><td>8.4</td></tr><tr><td>3e-05</td><td>0.0</td><td>0.999</td><td>2.0</td><td>28.6</td><td>35.8</td><td>45.3</td><td>19.6</td><td>8.5</td></tr><tr><td>3e-05</td><td>0.0</td><td>0.999</td><td>5.0</td><td>28.6</td><td>35.8</td><td>45.3</td><td>19.6</td><td>8.4</td></tr><tr><td>3e-05</td><td>0.9</td><td>0.999</td><td>1.0</td><td>28.6</td><td>36.6</td><td>44.7</td><td>18.4</td><td>8.0</td></tr></table>

![](images/be62f24a8f129ed4095646bb49051552ef22874389add9674874e23b18f4c39e.jpg)

![](images/a2c8ea19c09fc94ceadda93e7382172e5fea2beda2e4d6b1ae94691ffa105a5d.jpg)  
SDM (Distilled; Adaptive, = 1) SDM (Argmax + SC, = 1) SDM (Expectation, = 1)  
NFE vs TinyGSM Accuracy Distilled SDM vs Non-Distilled SDM  
Figure 13: Distilled vs. undistilled SDMs on TinyGSM across sampling steps $( \mathrm { N F E } \in \ [ 1 , 2 ^ { 1 4 } ]$ $\kappa = 1$ , adaptive grid) at $T = 0 . 1$ (left) and $T = 1$ (right). $\mathbf { A } \mathbf { t } T = 1$ , the distilled SDM solves 32.1% of the problems with 8 NFEs, against 6.3% for the undistilled SDM with the expected embedding, and 39.4% with 128 NFEs. Beyond 128 NFEs it plateaus at 40–41%, whereas the undistilled SDM keeps improving (45.8% at 512 and 49.7% at 16k NFEs).

follows

$$
\tilde { \zeta } ( x _ { t } , t ) = \zeta _ { \theta _ { \mathrm { f i n a l } } } ( x _ { t } , t ) + w \big ( \zeta _ { \theta _ { \mathrm { f i n a l } } } ( x _ { t } , t ) - \zeta _ { \theta _ { K } } ( x _ { t } , t ) \big ) , \qquad \hat { p } _ { 0 } ( x _ { t } , t ) = \mathrm { s o f t m a x } \Bigg ( \frac { \tilde { \zeta } ( x _ { t } , t ) } { T } \Bigg ) ,
$$

where w $\geqslant 0$ controls the guidance strength (w “ 0 recovers the un-guided baseline).

We apply Auto-Guidance to the TinyGSM SDM with the expected embedding (no ${ \mathrm { S C } } ;$ $\nu _ { 0 } ~ = ~ 0 . 2 , ~ \nu _ { 1 } ~ = ~ 0 . 5 , ~ \ell ~ = ~ 0 . { \overset { . } { 2 } } )$ , sampled with 64 steps. The guided model is the EMA checkpoint at 250k steps, and the guiding model uses the raw (non-EMA) parameters at step $\begin{array} { r l r } { K } & { { } \in } & { \left\{ 0 , 1 0 \mathbf { k } , 2 0 \mathbf { k } , 3 0 \mathbf { k } , 5 0 \mathbf { k } , 1 0 0 \mathbf { k } , 1 5 0 \mathbf { k } , 2 0 0 \mathbf { k } , 2 4 9 . 5 \mathbf { k } \right\} } \end{array}$ We sweep w P $\{ 0 . 0 5 , 0 . 1 , 0 . 1 5 , 0 . 2 , 0 . 2 5 , 0 . 3 5 , 0 . 5 , 0 . 7 5 , 1 . 0 , 1 . 5 \}$ and logit temperatures $T \in \{ 0 . 0 1 , 0 . 1 , 1 . 0 \}$ for each churn κ P $\{ 0 . 0 , 0 . 2 , 1 . 0 \}$ , with 3 evaluation seeds. Figure 14 shows TinyGSM pass@1 accuracy for $w \in \{ 0 . 1 , 0 . 2 5 , 0 . 5 , 1 . 0 \}$ . We observe that guiding against an early-to-intermediate checkpoint $( K \in [ 2 0 \mathbf { k } , 5 0 \mathbf { k } ] )$ maximizes the accuracy for all churn levels $\kappa ,$ while for checkpoints close to convergence, the accuracy gets closer to the unguided baseline.

Figure 15 summarizes the best auto-guided accuracy $( w ^ { \star } )$ alongside the accuracy gain $( \Delta \% )$ over the un-guided baseline $( w = 0 )$ across logit temperatures $T \in \{ 0 . 0 1 , 0 . 1 , 1 . 0 \}$ . We observe improvements over the baseline in all frameworks we investigate. In particular, the Auto-Guidance has the strongest effect with low churn and high temperature.  
![](images/8393c8c8ac84e81ab15890e0cdb45fce5e35537ca98cd841d43e184b37fd41bc.jpg)

![](images/16f78bb0c1ec8d2c59803508d3ae25f5142cf76dba89cc043299b0ec87b27e7a.jpg)

![](images/99083d407f7a5ac3b1eabc3b0ea4abfd7c6150bb96df06b9be61e905faa419ad.jpg)  
Guidance w = 0:10 Guidance w = 0:25 Guidance w = 0:50 Guidance w = 1:00 Un-Guided Baseline (w = 0) Logit Temperature T= 0:01 ( 1 SD) Logit Temperature T= 0:1 ( 1 SD) Logit Temperature T= 1:0 ( 1 SD)

Figure 14: TinyGSM accuracy (%) vs. Auto-Guidance checkpoint step K for different churn regimes (κ $\in \ \{ 0 . 0 , 0 . 2 , 1 . 0 \} )$ , with 64 sampling steps. We compare logit temperatures $T ~ = ~ 0 . 0 1$ (green), $T ~ = ~ 0 . 1$ (orange) and $T ~ = ~ 1 . 0$ (blue) for different guidance weights $w \in \{ 0 . 1 , 0 . 2 5 , 0 . 5 , 1 . 0 \}$ against the corresponding un-guided baselines $( w = 0 ;$ , dashed horizontal lines). Error bars denote a variation of one standard deviation across 3 evaluation seeds. Early-tointermediate checkpoints $( K \in [ 2 0 \mathbf { k } , 5 0 \mathbf { k } ] )$ consistently achieve peak accuracy.  
![](images/f8718b5c16f1cf4ce43c64b9b7e04acdc9382cd113ffe44d00e056b3db99ecf5.jpg)

![](images/f23d88ad158fc190a8966b7dc9e20182b4ddb03bb3b04d7a52e744f3b3c20374.jpg)  
Best Auto-Guided (w\*, ±1 SD) Un-Guided Baseline (w = 0, ±1 SD) Auto-Guidance Boost ( %, ±1 SD) Simplicial Churn = 0.0 Simplicial Churn = 0.2 Simplicial Churn = 1.0  
Logit Temperature vs TinyGSM Accuracy & Auto-Guidance Gain (Logit T <= 1, Mean ± 1 SD)

Figure 15: Logit temperature sensitivity and Auto-Guidance gain on TinyGSM $( T \in$ $\{ 0 . 0 1 , 0 . 1 , 1 . 0 \} )$ . Left: Validation accuracy (%) of the best auto-guided model $( w ^ { \star }$ , solid lines with stars) vs. the un-guided baseline $( w = 0$ , dashed lines with circles) for $\kappa \in \{ 0 . 0 , 0 . 2 , 1 . 0 \}$ Right: Accuracy gain $( \Delta \% )$ from Auto-Guidance. Error bars denote a variation of one standard deviation across 3 evaluation seeds.

## K.5 OPENWEBTEXT

Inference Interventions. In the following paragraph, we show how different interventions can push the Pareto frontier in terms of Generative Perplexity and unigram entropy, see Pynadath et al.

(2026b) for a discussion on those evaluations. We explore those interventions for the four different classes of models we investigate. Namely, we propose interventions for Autoregressive (AR) models, Masked Diffusion Models (MDMs), Uniform Diffusion Models (UDMs) and Simplex Diffusion Models (SDMs). Most of those techniques can be interpreted as being part of a logit shaping pipeline. The model was trained according to Section J.6.

First, we consider some nucleus sampling techniques. For completeness, we recall the process of nucleus sampling adapted from Holtzman et al. (2019). In the case of one token we denote $\left\{ \zeta _ { 1 } , \dots , \zeta _ { N } \right\}$ , the proposed logits, i.e., we have that log $\hat { P } _ { 0 | t } ( \cdot | P _ { t } ) ~ = ~ \{ \zeta _ { 1 } , \ldots , \zeta _ { N } \}$ in the case of SDMs for instance. Next, we denote $\big \{ \zeta _ { \varphi ( 1 ) } , \dots , \zeta _ { \varphi ( N ) } \big \}$ the sorted set of logits (in descending order, i.e., $\zeta _ { \varphi ( j ) } \geqslant \zeta _ { \varphi ( j + 1 ) }$ for any $j \in \{ 1 , \ldots , N - 1 \} )$ . Next, we denote $k \in \{ 1 , \ldots , N \}$ the smallest integer such that $\begin{array} { r } { \sum _ { j = 1 } ^ { k } p _ { \varphi ( j ) } \geqslant p . } \end{array}$ , where p is a hyperparameter and $\left\{ p _ { 1 } , \ldots , p _ { N } \right\} =$ softmax $\mathbf { \nabla } : \left( \left\{ \zeta _ { 1 } , \dots , \zeta _ { N } \right\} \right)$ . We denote ϕ the inverse of $\varphi , \mathrm { i . e . }$ . for any $i \in \{ 1 , \ldots , N \} , \varphi ( \phi ( i ) ) = i .$ For any $j \in \{ 1 , \ldots , N \}$ , we denote $\hat { \zeta } _ { j } = \zeta _ { j } \mathrm { i f } \phi ( j ) \leqslant k$ and $\hat { \zeta } _ { j } = - \infty$ otherwise. Next, we consider TOP such that

$$
\operatorname { T O P } ( p , \{ \zeta _ { 1 } , \dots , \zeta _ { N } \} ) = \{ \hat { \zeta } _ { 1 } , \dots , \hat { \zeta } _ { N } \} .\tag{39}
$$

Another intervention we consider is temperature scaling of the logits. More precisely, we consider

$$
\operatorname { T E M P } ( T , \{ \zeta _ { 1 } , \dots , \zeta _ { N } \} ) = \{ \zeta _ { 1 } / T , \dots , \zeta _ { N } / T \} .\tag{40}
$$

For the first intervention we consider $T$ in (40) to be in the following set

$$
\{ 0 . 2 0 , 0 . 3 0 , 0 . 4 0 , 0 . 5 0 , 0 . 6 0 , 0 . 6 5 , 0 . 7 0 , 0 . 7 5 , 0 . 8 0 , 0 . 8 5 , 0 . 9 0 , 0 . 9 5 , 1 . 0 0 , 1 . 0 5 , 1 . 1 0 , 1 . 1 8 , 1 . 2 8 \} .\tag{41}
$$

and $p \in \{ 0 . 9 2 , 0 . 9 6 \}$ in (39). This yields 34 sweeps for each family UDM, MDM, SDM and AR after this intervention.

For the second intervention, which only applies to diffusion methods, we consider temperature annealing. Let $t \in [ 0 , 1 ]$ be the diffusion time of the forward process, we consider a similar sweep as before but instead consider a temperature annealing procedure where

$$
T = T _ { \mathrm { s t a r t } } + ( T _ { \mathrm { e n d } } - T _ { \mathrm { s t a r t } } ) t ^ { 1 . 5 } ,
$$

with $T _ { \mathrm { e n d } }$ given by (41) and $T _ { \mathrm { s t a r t } } = 0 . 8 T _ { \mathrm { e n d } }$ . We do not claim that the power relationship with coefficient 1.5 and $\dot { T } _ { \mathrm { s t a r t } } = 0 . 8 T _ { \mathrm { e n d } }$ are optimal but we found those values to give good results in early experiments. Note that a linear temperature annealing was already considered in DiffusionGemma Team et al. (2026) with $T _ { \mathrm { s t a r t } } = 0 . 4$ and $T _ { \mathrm { e n d } } = 0 . 8$ . Similar power law temperature annealing schedule were also identified in Chang et al. (2022). Since AR has no corruption time, we instead anneal its temperature over token positions. We always consider TEMP and then follow it by TOP.

For the third intervention, which is by far the most influential one, we consider afrequency penalty. The frequency penalty is a sequence based penalty. In what follows, in the case of UDM and MDM, $x _ { t } \in \{ 1 , \ldots , \mathbf { \bar { N } } \} ^ { L }$ where $L$ is the sequence length and N is the vocabulary size (in the case of MDM we assume that one of the token is the masked one denoted rMASKs ). We denote the count variable $\{ c _ { t , v } \} _ { v = 1 } ^ { N } \in \mathbb { N } ^ { N }$ which is defined for any $v \in \{ 1 , \ldots , N \}$ with $v \not = \left\lceil \mathbf { M A S K } \right\rceil$

$$
c _ { t , v } = \sum _ { j = 1 } ^ { L } \delta _ { v } ( x _ { t , j } ) .\tag{42}
$$

In addition, in the case $v = \left[ \mathrm { M A S K } \right] , c _ { t , v } = 0$ . In other words, $\boldsymbol { c } _ { t , v }$ is the count of tokens which have values v in the sequence $x _ { t }$ , except for potentially the mask token. In the case of the simplex diffusion model, we slightly modify the definition of the count in (42) and we define for any $v \in$ $\{ 1 , \ldots , N \}$ with $v \not = \left[ \mathrm { M A S K } \right]$

$$
c _ { t , v } = \sum _ { j = 1 } ^ { L } \delta _ { v } ( \operatorname { a r g m a x } P _ { t , j } ) .\tag{43}
$$

Let $\{ c _ { t , v } \} _ { v = 1 } ^ { N } \in \mathbb { N } ^ { N }$ be defined either with (42) or (43). We introduce the frequency penalty FREQ given by

$$
\mathrm { F R E Q } ( \lambda , \{ \zeta _ { 1 } , \ldots , \zeta _ { N } \} ) = \{ \zeta _ { 1 } - \lambda \log ( 1 + c _ { t , 1 } ) , \ldots , \zeta _ { N } - \lambda \log ( 1 + c _ { t , N } ) \} .
$$

![](images/1657a2a230fb36316106f8117ab8d9f834f82e274da489e6257c83746c31bc24.jpg)  
Unigram Entropy H (nats) vs. Gen PPL

![](images/1dcd6c7b0cb9da0e5c53fae7a2f579f0d3c965642d346936057a64c7e3769eff.jpg)  
Figure 16: OpenWebText $( L = 1 0 2 4 )$ Pareto frontiers across generative families. Best cumulative Pareto frontiers of Generative Perplexity (scored by GPT-2 Large; top is lower/better) versus sequence token diversity (right is higher/better) for Simplex Diffusion Models (SDM, orange), Masked Diffusion (MDM, blue), Uniform Discrete Diffusion (UDM, green), and the Autoregressive baseline (AR, black). Left: Unigram token entropy $H _ { 1 }$ (nats). Right: Bigram token entropy $H _ { 2 } ( \mathrm { n a t s } )$ . Dotted reference lines and the gold star (‹) denote the measured OpenWebText validation distribution $( H _ { 1 } = 5 . 4 6$ nats, $H _ { 2 } = 6 . 5 7$ nats, Gen $\mathrm { P P L } = 1 5 . 1 9 )$

Repetition penalties were also considered in Xu et al. (2022). We always consider TEMP and then follow it by FREQ and then by TOP. In the case of AR one can define a similar intervention on the tokens already generated. We consider a regularization penalty $\lambda \in \{ 1 . 5 , 3 . 0 , 4 . 5 , 6 . 0 , 8 . 0 \}$ for all the sweeps.

For the fourth and final intervention, we consider another token penalty but at the local level, contrary to the global level of the frequency penalty of the third intervention. In particular, we consider $\{ \bar { c } _ { t , j , v } \} \in \mathbb { R } ^ { \breve { L } \times N }$ given for any $j \in \left\{ { 1 , \ldots , L } \right\}$ and $v \in \{ 1 , \ldots , N \}$ with $v \not = \left[ \mathrm { M A S K } \right]$ by

$$
\hat { c } _ { t , j , v } = \log \left( 1 + \frac { \alpha _ { t } N } { 1 - \alpha _ { t } } \right) \delta _ { v } ( x _ { t , j } ) .\tag{44}
$$

We set $\hat { c } _ { t , j , v } = 0$ for $v = \left[ \mathrm { M A S K } \right]$ . Finally, similarly to (43), we can define the simplicial counterpart of (44) as

$$
\hat { c } _ { t , j , v } = \log \left( 1 + \frac { \alpha _ { t } N } { 1 - \alpha _ { t } } \right) \delta _ { v } ( \operatorname { a r g m a x } P _ { t , j } ) .
$$

We are now ready to define the local frequency intervention FREQLOC given by

$$
\mathrm { F R E Q L O C } ( \gamma , \{ \zeta _ { 1 } , \ldots , \zeta _ { N } \} ) = \{ \zeta _ { 1 } - \gamma \hat { c } _ { t , j , 1 } , \ldots , \zeta _ { N } - \gamma \hat { c } _ { t , j , N } \} ,
$$

where here we have assumed that $\left\{ \zeta _ { 1 } , \dots , \zeta _ { N } \right\}$ are the logits at position $j \in \left\{ 1 , \ldots , L \right\}$ . We consider $\gamma \in \{ 2 . 0 , 2 . 5 \}$ for SDM only. We do not consider the intervention for AR as it already yields models which are scoring better than real data on the Generative Perplexity and Entropy Pareto frontier. In the case of UDM the intervention did not improve the Pareto frontier and is therefore not reported. In the case of MDM the local frequency constraint does not change the prediction since we do not consider remasking. Therefore, we only report SDMs results.

In Figure 16, we illustrate the obtained Pareto frontiers after all those interventions. Note that the AR Pareto frontier scores better than the real data on OpenWebText, thereby putting into question the validity of generative frontiers on OWT as a meaningful benchmark for language generation. In Figure 17 and Figure 18, we show the effect of each intervention on the Pareto frontier for all four models that we are investigating. The best hyperparameters for each model at matched entropy are reported in Table 7 and at matched Generative Perplexity are reported in Table 8.

For each evaluation point, we generate 128 unconditional sequences of length 1024 (evaluated in batches of 2 over 64 batches) using 64 sampling steps for all diffusion models and 1024 steps (tokenby-token) for the autoregressive baseline.

## K.6 LANGUAGE UNDERSTANDING

We describe the setup in Section J.7.

![](images/31ad510bd0e971449e7eadd2d946f1ffc269f6ea969c2d3a32002eade891c425.jpg)  
SDM — Unigram Entropy H<sub>1</sub> (nats) vs. Gen PPL

![](images/19f16d7ae5e9dd2a547c7a801cee000e19e1ef1c5ace5b085cdf392a62d101b4.jpg)

![](images/26ab1bb037bac1b7812fbde2dfb4049d99cd233ed74df95bbfc3a89038b5ea62.jpg)  
UDM — Unigram Entropy H (nats) vs. Gen PPL

![](images/1042c9dee8dbc2a7e84e3b0261a085ac05d45714a8aa7935276177cfb217979a.jpg)  
Figure 17: Progressive intervention ablations on Unigram Entropy $H _ { 1 }$ (nats) vs. Generative Perplexity. Cumulative Pareto frontiers per family across the intervention stack: Stage A (nucleus truncation $p ~ \in ~ \{ 0 . 9 2 , 0 . 9 6 \}$ + initial temperature $T _ { \mathrm { s t a r t } } )$ , Stage B (\` power-law temperature annealing $\bar { T _ { \mathrm { s t a r t } } } ~ = ~ 0 . 8 0 T _ { \mathrm { e n d } } ,$ exponent 1.5), Stage C (\` sequence-level frequency penalty $\lambda \in \{ 1 . 5 , 3 . 0 , 4 . 5 , 6 . 0 , 8 . 0 \}$ ), and Stage D (\` local frequency penalty γ, SDM only).

Table 7: Vertical Cuts on Unigram Entropy $( H _ { 1 } ~ \geqslant ~ \tau ) \colon$ Minimum Generative Perplexity (GenPPL Ó) on OpenWebText. AR is our model with the same architecture as the diffusion mod els, trained on OWT (1024 steps). The best diffusion method is bold, second best is underlined. The associated hyperparameters $( p , T _ { \mathrm { s t a r t } } , \lambda , \gamma )$ are shown in grey below each entry. The real OWT validation data (‹) has $H _ { 1 } = 5 . 4 6$ and $\mathrm { G e n P P L } = 1 5 . 2$
<table><tr><td>Threshold  $( H _ { 1 } \geqslant \tau )$ </td><td>SDM (Simplex)</td><td>MDM (Masked)</td><td>UDM (Uniform)</td><td>AR (1024 steps)</td></tr><tr><td> $H _ { 1 } \geqslant 5 . 2 0$ </td><td>14.0</td><td>14.7</td><td>13.5</td><td>8.9</td></tr><tr><td> $H y p e r p a r a m e t e r s \left( p , T _ { s t a r t } , \lambda , \gamma \right)$ </td><td>(.96, .40, 3.0, 2.0)</td><td>(.96, .70, 1.5, 0.0)</td><td>(.92, .60, 1.5, 0.0)</td><td>(.96, .65, 3.0, 0.0)</td></tr><tr><td> $H _ { 1 } \geqslant 5 . 3 5$ </td><td>15.8</td><td>16.3</td><td>15.2</td><td>8.9</td></tr><tr><td> $H y p e r p a r a m e t e r s \left( p , T _ { s t a r t } , \lambda , \gamma \right)$ </td><td>(.92, .30, 6.0, 2.5)</td><td>(.92, .60, 3.0, 0.0)</td><td>(.96, .60, 1.5, 0.0)</td><td>(.96, .65, 3.0, 0.0)</td></tr><tr><td> $\mathbf { H _ { 1 } } \geqslant 5 . 4 6 ( \star \mathbf { O W T } = 1 5 . 2 )$ </td><td>17.0</td><td>16.9</td><td>17.3</td><td>9.0</td></tr><tr><td> $H y p e r p a r a m e t e r s \left( p , T _ { s t a r t } , \lambda , \gamma \right)$ </td><td>(.92, .30, 6.0, 2.5)</td><td>(.92, .50, 4.5, 0.0)</td><td>(.92, .70, 1.5, 0.0)</td><td>(.96, .65, 3.0, 0.0)</td></tr><tr><td> $H _ { 1 } \geqslant 5 . 6 5$ </td><td>20.6</td><td>24.0</td><td>22.5</td><td>9.0</td></tr><tr><td> $H y p e r p a r a m e t e r s \left( p , T _ { s t a r t } , \lambda , \gamma \right)$ </td><td>(.96, .30, 8.0, 2.0)</td><td>(.92, .70, 3.0, 0.0)</td><td>(.96, .60, 3.0, 0.0)</td><td>(.96, .65, 3.0, 0.0)</td></tr><tr><td> $H _ { 1 } \geqslant 5 . 8 5$ </td><td>26.2</td><td>31.2</td><td>26.7</td><td>9.3</td></tr><tr><td> $H y p e r p a r a m e t e r s \left( p , T _ { s t a r t } , \lambda , \gamma \right)$ </td><td>(.92, .40, 6.0, 2.0)</td><td>(.92, .65, 4.5, 0.0)</td><td>(.96, .70, 3.0, 0.0)</td><td>(.92, .70, 3.0, 0.0)</td></tr></table>

The results sweeping on the weight w in (38) and comparing the baseline model with the adapted model are reported in Figure 19. The adapted model outperforms the baseline model, which confirms that the baseline is out of distribution when only the continuation is noised. Finally, our main results and comparison with benchmarks are presented in Table 9.

## K.7 UNCONDITIONAL MOLECULAR GENERATION

The experimental setup is described in Section J.8.

![](images/5eeb585a836506ed75eac3a94a01e9c184261365d59fa813e132c1330a5b4876.jpg)

![](images/ba1c3fa2b5c70688680cf442b0b62c273d11c5bc16062c2788563b59f4a875e7.jpg)

![](images/f3bb318cd0bb10139a93869b8f30d0874909a20a8733ec37b12d99e1ded5a73d.jpg)

![](images/80ebacf9ac64ad13254b069d4ed9ee8200040bba61558e4028e81ed8f380b9d3.jpg)  
Figure 18: Progressive intervention ablations on Bigram Entropy $H _ { 2 }$ (nats) vs. Generative Perplexity. Order-sensitive bigram diversity control $\left( H _ { 2 } \right)$ across the same nested intervention stages (Stage A Ñ Stage D) for SDM, MDM, UDM, and AR. Both the sequence frequency penalty (Stage C) and the local frequency penalty (Stage D) preserve gains along the bigram entropy $H _ { 2 }$ axis.

Table 8: Horizontal Cuts on Generative Perplexity $\begin{array} { r } { ( \mathbf { G e n P P L } \leqslant \tau _ { \mathbf { P P L } } ) \colon } \end{array}$ Maximum Unigram Entropy $( H _ { 1 } \uparrow )$ on OpenWebText. AR is our model with the same architecture as the diffusion models, trained on OWT (1024 steps). The best diffusion method is bold, second best is underlined. The associated hyperparameters $( p , T _ { \mathrm { s t a r t } } , \lambda , \gamma )$ are shown in grey below each entry. The real OWT validation data (‹) has $\mathrm { G e n P P L } = 1 5 . 2$ and $H _ { 1 } = 5 . 4 6$
<table><tr><td>Budget (GenPPL ≤ τPPL)</td><td>SDM (Simplex)</td><td>MDM (Masked)</td><td>UDM (Uniform)</td><td>AR (1024 steps)</td></tr><tr><td>GenPPL ≤ 12.5</td><td>5.09</td><td>5.17</td><td>5.13</td><td>6.18</td></tr><tr><td>Hyperparameters  $( p , T _ { s t a r t } , \lambda , \gamma )$ </td><td>(.92, .30, 4.5, 2.5)</td><td>(.96, .50, 3.0, 0.0)</td><td>(.96, .50, 1.5, 0.0)</td><td>(.96, .40, 8.0, 0.0)</td></tr><tr><td>GenPPL ≤ 14.0</td><td>5.20</td><td>5.18</td><td>5.23</td><td>6.20</td></tr><tr><td>Hyperparameters  $( p , T _ { s t a r t } , \lambda , \gamma )$ </td><td>(.92, .40, 3.0, 2.0)</td><td>(.96, .40, 4.5, 0.0)</td><td>(.96, .50, 1.5, 0.0)</td><td>(.96, .40, 8.0, 0.0)</td></tr><tr><td>GenPPL ≤ 15.2 (* OWT H1 = 5.46)</td><td>5.29</td><td>5.22</td><td>5.32</td><td>6.21</td></tr><tr><td>Hyperparameters  $( p , T _ { s t a r t } , \lambda , \gamma )$ </td><td>(.96, .40, 3.0, 2.0)</td><td>(.92, .70, 1.5, 0.0)</td><td>(.96, .50, 1.5, 0.0)</td><td>(.92, .65, 4.5, 0.0)</td></tr><tr><td>GenPPL ≤ 18.0</td><td>5.54</td><td>5.52</td><td>5.48</td><td>6.29</td></tr><tr><td> $H y p e r p a r a m e t e r s \left( p , T _ { s t a r t } , \lambda , \gamma \right)$ </td><td>(.96, .30, 6.0, 2.5)</td><td>(.96, .50, 4.5, 0.0)</td><td>(.96, .65, 1.5, 0.0)</td><td>(.96, .65, 4.5, 0.0)</td></tr><tr><td>GenPPL ≤ 22.0</td><td>5.72</td><td>5.59</td><td>5.62</td><td>6.43</td></tr><tr><td> $H y p e r p a r a m e t e r s \left( p , T _ { s t a r t } , \lambda , \gamma \right)$ </td><td>(.92, .40, 4.5, 2.0)</td><td>(.96, .65, 3.0, 0.0)</td><td>(.96, .50, 3.0, 0.0)</td><td>(.96, .50, 8.0, 0.0)</td></tr></table>

Sampling hyperparameters selection. To select the optimal inference hyperparameters for each method, sampler, and schedule, we swept over:

• Autoregressive (AR): softmax temperature $T \in \{ 0 . 0 , 0 . 5 , 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 , 1 . 0 \}$

• Discrete Diffusion (MDM & UDM): logit temperature $T \in \{ 0 . 0 0 1 , 0 . 0 0 5 , 0 . 0 1 , 0 . 0 3 , 0 . 0 5 \}$ 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0u, top-fraction schedule power p P t0.5, 1.0, 1.5, 2.0, 3.0u.

• Simplex Diffusion (SDM): logit temperature $T \ \in \ \{ 0 . 0 0 1 , 0 . 0 0 5 , 0 . 0 1 , 0 . 0 3 , 0 . 0 5 , 0 . 0 8 , 0 . 1 ,$ 0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0u, churn κ P t0.0, 0.2, 1.0u, top-fraction schedule power p P t0.5, 1.0, 1.5, 2.0, 3.0u.

![](images/c4f8b271f824b476705ee222ed34f621d4b490023b99c39536273da27c24c119.jpg)  
Figure 19: We present a sweep on the Pointwise Mutual Information (PMI) weight w in (38). For each of the three tasks that we investigate HellaSwag (Zellers et al., 2019), ARC-easy (Clark et al., 2018) and PIQA (Bisk et al., 2020), we report the influence of w on the final results. Note that $w = 0$ corresponds to the setup which is most comparable to the MDM scoring (37) (even though MDM noise is applied on the whole sequence while ours focus on the continuation). We denote $w ^ { \star }$ the best weighting for each task. The optimal weight differs across tasks $( w ^ { \star } = 0 . 1$ for PIQA, 0.6 for ARC-Easy and 0.55 for HellaSwag) and is selected on the evaluation split. The results are averaged over 5 different seeds. In red we report the results for the baseline out of distribution OWT checkpoint. In blue we report the results after the short adaptation run.

Table 9: Zero-shot accuracy (%) on language understanding benchmarks. We evaluate SDMs by sampling 5 times with a different random seed, and report the $\mathrm { \ m e a n _ { \pm s t d } }$ . The best score per column is bolded and the second best is underlined. The LLaMA baseline is taken from von Rutte et al.¨ (2025).
<table><tr><td>Model</td><td>PIQA</td><td>ARC-Easy</td><td>HellaSwag</td></tr><tr><td colspan="4">Autoregressive baselines</td></tr><tr><td>LLaMA</td><td>62.7</td><td>40.5</td><td>33.1</td></tr><tr><td>GPT-2</td><td>62.9</td><td>43.8</td><td>28.9</td></tr><tr><td colspan="4">Discrete Diffusion models</td></tr><tr><td>MDM</td><td>54.1</td><td>31.0</td><td>31.1</td></tr><tr><td>PGM</td><td>58.9</td><td>40.4</td><td>33.2</td></tr><tr><td>GIDD</td><td>56.4</td><td>31.0</td><td>31.9</td></tr><tr><td colspan="4">Ours (Simplex Diffusion)</td></tr><tr><td> $w = 0$ </td><td> $5 4 . 3 _ { \pm 0 . 4 }$ </td><td> $3 6 . 3 _ { \pm 0 . 4 }$ </td><td> $3 5 . 6 _ { \pm 0 . 2 }$ </td></tr><tr><td> $w = w ^ { \star }$ </td><td> $5 4 . 6 _ { \pm 0 . 5 }$ </td><td> $\underline { { 4 3 . 2 } } \underline { { + 0 . 3 } }$ </td><td> $\mathbf { 3 7 . 9 _ { \pm 0 . 2 } }$ </td></tr></table>

The optimal sampling hyperparameters corresponding to Figure 20 are reported in Table 10. The optimal sampling hyperparameters corresponding to Figure 21 are reported in Table 11, while those for Figure 22 are reported in Table 12. The optimal sampling hyperparameters corresponding to each row of Tables 14 and 15 with 256 sampling steps, are reported in Table 13 and are selected based on the quality metric.

Sampling budgets and sampling time schedules. We start from an ablation over sampling time schedules for Simplex Diffusion for molecular generation. In Figure 20, we report performance of Simplex Diffusion (with time conditioning) as a function of sampling steps using either standard or confidence-based sampler. We see that depending on the sampling budget, the type of a sampler and the evaluation metric chosen, the choice of a sampling time schedule may differ. For the next ablation, we select adaptive time schedule for Simplex diffusion since it offers overall a good trade off between different metrics and different sampling budgets.

Comparison across model variants. Next, we compare performance of Simplex Diffusion to masked and uniform diffusion. For these methods we use cosine time schedule since it led to the best Quality. We report results in Figure 21 for the standard sampler and in Figure 22 for the confidence-based one. We see that both masked and uniform diffusion lead to higher Quality than Simplex Diffusion, while achieving lower Diversity. In case of confidence-based sampler the finding remains except for the fact that Simplex diffusion performs on-par with Uniform diffusion without time conditioning. These results demonstrate that different diffusion methods provide a trade-off between these two metrics.

Table 10: Selected hyperparameters for Simplex Diffusion (with time conditioning, SDM TE) across noise schedules and sampling steps (molecular generation; N P t32, 64, 256, 1024u) corresponding to Figure 20. T: logit sampling temperature; κ: Simplex churn parameter; p: top-fraction power for α<sup>p</sup> to select the most confident subset.
<table><tr><td colspan="2"></td><td colspan="2">Standard Sampler</td><td colspan="3">Confidence-Based Sampler</td></tr><tr><td>Schedule</td><td>Steps (N)</td><td>Temperature (T)</td><td>Churn (κ)</td><td>Temperature (T)</td><td>Churn (κ)</td><td>Power (p)</td></tr><tr><td rowspan="4">Linear</td><td>32</td><td>0.01</td><td>1.0</td><td>0.01</td><td>1.0</td><td>1.0</td></tr><tr><td>64</td><td>0.005</td><td>1.0</td><td>0.05</td><td>1.0</td><td>1.0</td></tr><tr><td>256</td><td>0.05</td><td>0.2</td><td>0.05</td><td>1.0</td><td>1.0</td></tr><tr><td>1024</td><td>0.001</td><td>0.2</td><td>0.05</td><td>1.0</td><td>1.0</td></tr><tr><td rowspan="4">Cosine</td><td>32</td><td>0.03</td><td>1.0</td><td>0.03</td><td>1.0</td><td>1.0</td></tr><tr><td>64</td><td>0.05</td><td>1.0</td><td>0.05</td><td>1.0</td><td>1.0</td></tr><tr><td>256</td><td>0.10</td><td>0.0</td><td>0.05</td><td>1.0</td><td>1.0</td></tr><tr><td>1024</td><td>0.15</td><td>1.0</td><td>0.08</td><td>1.0</td><td>1.0</td></tr><tr><td rowspan="4">Adaptive</td><td>32</td><td>0.005</td><td>0.0</td><td>0.05</td><td>1.0</td><td>1.0</td></tr><tr><td>64</td><td>0.001</td><td>0.0</td><td>0.05</td><td>1.0</td><td>1.0</td></tr><tr><td>256</td><td>0.005</td><td>0.2</td><td>0.03</td><td>1.0</td><td>1.0</td></tr><tr><td>1024</td><td>0.03</td><td>1.0</td><td>0.005</td><td>1.0</td><td>1.0</td></tr></table>

Table 11: Selected hyperparameters for diffusion methods under standard sampling across steps (N P t32, 64, 256, 1024u) corresponding to Figure 21. Simplex methods use the adaptive schedule; Masked and Uniform use the cosine schedule. T: logit sampling temperature; κ: Simplex churn parameter.
<table><tr><td>Method</td><td>Schedule</td><td>Steps (N)</td><td>Temperature (T)</td><td>Churn (κ)</td></tr><tr><td rowspan="4">Masked Diffusion (MDM)</td><td>Cosine</td><td>32</td><td>0.05</td><td></td></tr><tr><td></td><td>64</td><td>0.01</td><td></td></tr><tr><td></td><td>256</td><td>0.005</td><td></td></tr><tr><td></td><td>1024</td><td>0.03</td><td></td></tr><tr><td rowspan="4">Uniform Diffusion (UDM)</td><td>Cosine</td><td>32</td><td>0.05</td><td></td></tr><tr><td></td><td>64</td><td>0.001</td><td></td></tr><tr><td></td><td>256</td><td>0.01</td><td></td></tr><tr><td></td><td>1024</td><td>0.01</td><td></td></tr><tr><td rowspan="4">Uniform (with time conditioning)</td><td>Cosine</td><td>32</td><td>0.005</td><td></td></tr><tr><td></td><td>64</td><td>0.001</td><td></td></tr><tr><td></td><td>256</td><td>0.05</td><td></td></tr><tr><td></td><td>1024</td><td>0.05</td><td></td></tr><tr><td rowspan="4">Simplex Diffusion (SDM)</td><td>Adaptive</td><td>32</td><td>0.08</td><td>0.2</td></tr><tr><td></td><td>64</td><td>0.10</td><td>0.2</td></tr><tr><td></td><td>256</td><td>0.10</td><td>1.0</td></tr><tr><td></td><td>1024</td><td>0.05</td><td>1.0</td></tr><tr><td rowspan="4">Simplex (with time conditioning)</td><td>Adaptive</td><td>32</td><td>0.005</td><td>0.0</td></tr><tr><td></td><td>64</td><td>0.001</td><td>0.0</td></tr><tr><td></td><td>256</td><td>0.005</td><td>0.2</td></tr><tr><td></td><td>1024</td><td>0.03</td><td>1.0</td></tr></table>

Quality–diversity trade-offs. We further highlight this point by plotting a Pareto frontier for different methods with confidence-based sampler in Figure 23, where different points represent different sampling seeds, different number of sampling steps, different sampling temperatures, different churns and different top fraction power, see Section J.8 for details. We see that Simplex Diffusion Pareto frontier is situated more towards right and top, though it does not achieve the highest Quality, compared to Uniform and Masked diffusion. Understanding how to push this frontier further to the top right is a promising future research direction.

Table 12: Selected hyperparameters for diffusion methods under confidence-based sampling across steps (molecular generation; N P t32, 64, 256, 1024u) corresponding to Figure 22. Simplex methods use the adaptive schedule; Masked and Uniform diffusion use the cosine schedule. $T \colon$ logit sampling temperature; κ: Simplex churn parameter; $p \mathrm { : }$ top-fraction power for $\alpha _ { s } ^ { p }$ to select the most confident subset.
<table><tr><td>Method</td><td>Schedule</td><td>Steps (N)</td><td>Temperature (T)</td><td>Churn (κ)</td><td>Power  $( p )$ </td></tr><tr><td rowspan="5">Masked Diffusion (MDM)</td><td>Cosine</td><td>32</td><td>0.01</td><td></td><td>1.5</td></tr><tr><td></td><td>64</td><td>0.01</td><td></td><td>1.5</td></tr><tr><td></td><td>256</td><td>0.01</td><td></td><td>1.0</td></tr><tr><td></td><td>1024</td><td>0.01</td><td></td><td>2.0</td></tr><tr><td>Cosine</td><td>32</td><td>0.10</td><td></td><td>2.0</td></tr><tr><td rowspan="4"></td><td></td><td>64</td><td>0.03</td><td></td><td>1.5</td></tr><tr><td></td><td>256</td><td>0.03</td><td></td><td>1.0</td></tr><tr><td>Cosine</td><td>1024</td><td>0.01</td><td></td><td>0.5</td></tr><tr><td></td><td>32</td><td>0.01</td><td></td><td>1.0</td></tr><tr><td rowspan="4">Simplex Diffusion (SDM)</td><td></td><td>64</td><td>0.01</td><td></td><td>1.0</td></tr><tr><td></td><td>256</td><td>0.01</td><td></td><td>1.0</td></tr><tr><td></td><td>1024</td><td>0.01</td><td></td><td>1.0</td></tr><tr><td>Adaptive</td><td>32</td><td>0.05</td><td>1.0</td><td>1.0</td></tr><tr><td rowspan="4">Simplex (with time conditioning)</td><td></td><td>64</td><td>0.01</td><td>1.0</td><td>1.0</td></tr><tr><td></td><td>256</td><td>0.05</td><td>1.0</td><td>1.0</td></tr><tr><td></td><td>1024</td><td>0.08</td><td>1.0</td><td>1.0</td></tr><tr><td>Adaptive</td><td>32</td><td>0.05</td><td>1.0</td><td>1.0</td></tr><tr><td rowspan="4"></td><td></td><td>64</td><td>0.05</td><td>1.0</td><td>1.0</td></tr><tr><td></td><td>256</td><td>0.03</td><td>1.0</td><td>1.0</td></tr><tr><td></td><td>1024</td><td>0.005</td><td>1.0</td><td>1.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 13: Optimal sampling hyperparameters for each method and schedule (molecular generation) in Table 14 (standard sampling) and Table 15 (confidence-based sampling, conf.) using 256 sampling steps. T: logit temperature; κ: Simplex churn (Proposition 3.3); $p \mathrm { : }$ top-fraction power for $\alpha _ { s } ^ { p }$ to select the most confident subset.
<table><tr><td>Method</td><td>Schedule</td><td>Standard Sampling Hypers</td><td>Confidence (conf.) Sampling Hypers</td></tr><tr><td>Autoregressive</td><td></td><td> $T = 0 . 7$ </td><td></td></tr><tr><td>Simplex</td><td>Linear</td><td> $T = 0 . 0 3 , \ \kappa = 1 . 0$ </td><td> $T = 0 . 0 5 , \kappa = 1 . 0 , p = 1 . 0$ </td></tr><tr><td>Simplex</td><td>Cosine</td><td> $T = 0 . 0 1 , \ \kappa = 1 . 0$ </td><td> $T = 0 . 0 5 , \ \kappa = 1 . 0 , \ \bar { p } = 1 . 0$ </td></tr><tr><td>Simplex</td><td>Adaptive</td><td> $T = 0 . 1 0 , \ \kappa = 1 . 0$ </td><td> $T = 0 . 0 5 , \kappa = 1 . 0 , p = 1 . 0$ </td></tr><tr><td>Simplex (time cond.)</td><td>Linear</td><td> $T = 0 . 0 5 , \ \kappa = 0 . 2$ </td><td> $T = 0 . 0 5 , \kappa = 1 . 0 , p = 1 . 0$ </td></tr><tr><td>Simplex (time cond.)</td><td>Cosine</td><td> $T = 0 . 1 0 , \ \kappa = 0 . 0$ </td><td> $T = 0 . 0 5 , \ \kappa = 1 . 0 , \ \bar { p } = 1 . 0$ </td></tr><tr><td>Simplex (time cond.)</td><td>Adaptive</td><td> $T = 0 . 0 0 5 , \ \kappa = 0 . 2$ </td><td> $T = 0 . 0 3 , \kappa = 1 . 0 , p = 1 . 0$ </td></tr><tr><td>Uniform</td><td>Linear</td><td> $T = 0 . 0 1$ </td><td> $T = 0 . 0 3 , p = 1 . 5$ </td></tr><tr><td>Uniform</td><td>Cosine</td><td> $T = 0 . 0 1$ </td><td> $T = 0 . 0 3 , p = 1 . 0$ </td></tr><tr><td>Uniform</td><td>Adaptive</td><td> $T = 0 . 1 0$ </td><td> $T = 0 . 0 3 , \ { \\overline { { p } } } = 1 . 0$ </td></tr><tr><td>Uniform (time cond.)</td><td>Linear</td><td> $T = 0 . 0 1$ </td><td> $T = 0 . 0 1 , p = 1 . 0$ </td></tr><tr><td>Uniform (time cond.)</td><td>Cosine</td><td> $T = 0 . 0 5$ </td><td> $T = 0 . 0 1 , p = 1 . 0$ </td></tr><tr><td>Uniform (time cond.)</td><td>Adaptive</td><td> $T = 0 . 0 8$ </td><td> $T = 0 . 0 1 , p = 1 . 0$ </td></tr><tr><td>Masked</td><td>Linear</td><td> $T = 0 . 0 1$ </td><td> $T = 0 . 0 1 , p = 1 . 5$ </td></tr><tr><td>Masked</td><td>Cosine</td><td> $T = 0 . 0 0 5$ </td><td> $T = 0 . 0 1 , p = 1 . 0$ </td></tr><tr><td>Masked</td><td>Adaptive</td><td> $T = 0 . 0 0 1$ </td><td> $T = 0 . 0 1 , \ { \overset { \cdot } { p } } = 1 . 0$ </td></tr></table>

Detailed results at 256 steps. Tables 14 and 15 report all four evaluation metrics under standard and confidence-based sampling. With confidence-based sampling and an adaptive grid, timeconditioned SDM achieves 87.23% Quality and 0.857 Diversity. For comparison, time-conditioned UDM with a cosine grid achieves 91.97% and 0.835, while MDM with a cosine grid achieves

![](images/418e9d2651adc2997656242e36fb840a10471dd6e76e8e80885dc72a1c85aff1.jpg)

![](images/d1fbc3c3cb73052b6d5159cab68f12bddb3c6dadbb82737274b90e2762a10233.jpg)  
Linear Cosine Adaptive Standard sampler Confidence-based sampler

Figure 20: Sampling-budget and sampling time schedule comparison for time-conditioned SDM. Quality and Diversity (molecular generation) are shown for standard sampling (solid) and confidence-based sampling (dashed), using linear (blue), cosine (orange), and adaptive (green) in ference grids. Increasing the sampling budget improves Quality under confidence-based sampling while reducing Diversity. Standard sampling retains higher Diversity, with smaller and nonmono tonic changes in Quality. Error bars show one standard deviation across three sampling seeds.  
![](images/6007e59fe6967fe34f31ccfb426a4f664ec971371728ac83ea54b24f244bcd47.jpg)

![](images/abe8f842ab42bbe64bc2a8ac689ff8372042f04a9e9148997664c9b037c7f2e9.jpg)  
Masked Diffusion (MDM) Uniform Diffusion (UDM) Uniform (with time conditioning) Simplex Diffusion (SDM) Simplex (with time conditioning)

Figure 21: Quality and Diversity versus sampling steps under standard sampling (molecular generation). Comparison across all five diffusion methods, using adaptive time schedule for Simplex diffusion and cosine for the rest. Error bars show one standard deviation across three sampling seeds.  
![](images/b4bbcd223e96300bece95b627e83920d2995d69fc4a3691605b18086e69ce4c6.jpg)

![](images/b9dec60f48827c315d01a2f96a8e815fc9ffc44d8e63a65e64283288ebd05df9.jpg)  
Masked Diffusion (MDM) Uniform Diffusion (UDM) Uniform (with time conditioning) Simplex Diffusion (SDM) Simplex (with time conditioning)  
Figure 22: Quality and Diversity versus sampling steps under confidence-based sampling (molecular generation). Comparison across all five diffusion methods, using adaptive time schedule for Simplex diffusion and cosine for the rest. Error bars show one standard deviation across three sampling seeds.

92.80% and 0.819. Time conditioning increases the selected confidence-based Quality scores from   
85.87% to 87.23% for SDM with an adaptive grid and from 86.03% to 91.97% for UDM with

![](images/a1554c4d48df0e47dcb57e2ffb29e390c733160b9194758d9d4455e2f9fd2f5e.jpg)  
Figure 23: Empirical Quality–Diversity Pareto frontiers across diffusion methods with confidence based sampling and hyperparameter sweeps. Molecule Quality (%), Ò, versus Diversity (Ò) across all evaluated sampling hyperparameter configurations. All markers denote individual exploration runs sweeping sampling temperatures, number of sampling steps, top-fraction power for selecting the most confident subset and churn parameters, see Section J.8 for more details. Solid curves trace the empirical Pareto frontier of non-dominated configurations for each diffusion family, where we use adaptive time schedule for Simplex diffusion and cosine for the rest. Distinct stars mark external and autoregressive baselines: GenMol (gold star), the autoregressive AR model (black star), and SAFE-GPT (gray star). Simplex Diffusion Pareto frontier leans more towards top right, with many points located in high diversity regime.

a cosine grid, accompanied by reductions in Diversity. These results highlight the same findings observed in the figures above.

All reported diffusion configurations exceed our AR baseline in Quality. Time-conditioned SDM also achieves higher reported Quality and Diversity than the published GenMol reference, although the training and sampling protocols differ. Overall, the results demonstrate the applicability of SDMs to molecular generation and their competitive performance in higher-diversity regimes, while identifying a remaining gap in maximum Quality relative to masked diffusion. Finding ways to bridge the gap and to push the Pareto frontier for Simplex Diffusion is left for future work.

## L ADDITIONAL OWT SAMPLES

Tables 16 to 20 show non-cherry-picked unconditional 1024-token samples from SDMs trained on OWT.

Table 14: Unconditional de novo molecular generation on SAFE-GPT with standard sampling following the GenMol evaluation protocol (Lee et al., 2025) (1,000 molecules/seed, mean ˘ std over 3 seeds). Bold: best; underline: second best among our runs. All our runs use 256 sampling steps.
<table><tr><td>Method</td><td></td><td>Schedule Quality (%) ↑</td><td>Validity (%)↑</td><td>Uniqueness (%)↑</td><td>Diversity ↑</td></tr><tr><td colspan="6">Published Reference (Lee et al., 2025)</td></tr><tr><td>SAFE-GPT</td><td></td><td> $5 4 . 7 \pm 0 . 3$ </td><td> $9 4 . 0 \pm 0 . 4$ </td><td> $1 0 0 . 0 \pm 0$ </td><td> $0 . 8 7 9 \pm 0 . 0 0 1$ </td></tr><tr><td>GenMol</td><td></td><td> $8 4 . 6 \pm 0 . 8$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $9 9 . 7 \pm 0 . 1 $ </td><td> $0 . 8 1 8 \pm 0 . 0 0 1$ </td></tr><tr><td colspan="6">Our Run Comparison</td></tr><tr><td>Autoregressive (AR)</td><td></td><td> $6 3 . 9 7 \pm 1 . 6 7$ </td><td> $9 6 . 3 7 \pm 0 . 2 1$ </td><td> $9 9 . 7 6 \pm 0 . 0 6$ </td><td> $\underline { { 0 . 8 8 3 } } \pm 0 . 0 0 2$ </td></tr><tr><td>Simplex</td><td>Linear</td><td> $7 3 . 5 7 \pm 1 . 8 6$ </td><td> $9 9 . 6 3 \pm 0 . 1 2$ </td><td> $9 9 . 8 3 \pm 0 . 1 2$ </td><td> $0 . 8 7 9 \pm 0 . 0 0 1$ </td></tr><tr><td>Simplex</td><td>Cosine</td><td> $7 2 . 3 3 \pm 0 . 1 5$ </td><td> $9 9 . 4 0 \pm 0 . 4 0 $ </td><td> ${ \bf 9 9 . 9 7 \pm 0 . 0 6 }$ </td><td> $0 . 8 8 1 \pm 0 . 0 0 0$ </td></tr><tr><td>Simplex</td><td>Adaptive</td><td> $7 2 . 4 7 \pm 2 . 2 1$ </td><td> $9 9 . 6 3 \pm 0 . 1 5$ </td><td> $9 9 . 8 0 \pm 0 . 1 7$ </td><td> $\mathbf { 0 . 8 8 5 \pm 0 . 0 0 2 }$ </td></tr><tr><td>Simplex (time cond.)</td><td>Linear</td><td> $7 2 . 5 3 \pm 0 . 7 6$ </td><td> $9 9 . 0 0 \pm 0 . 1 0 $ </td><td> $9 9 . 7 0 \pm 0 . 3 0$ </td><td> $0 . 8 7 9 \pm 0 . 0 0 2$ </td></tr><tr><td>Simplex (time cond.)</td><td>Cosine</td><td> $7 4 . 7 0 \pm 0 . 3 6$ </td><td> $9 8 . 9 0 \pm 0 . 4 4$ </td><td> $9 7 . 6 4 \pm 0 . 7 6$ </td><td> $0 . 8 7 1 \pm 0 . 0 0 1$ </td></tr><tr><td>Simplex (time cond.)</td><td>Adaptive</td><td> $7 2 . 6 0 \pm 1 . 1 0$ </td><td> $9 8 . 8 3 \pm 0 . 0 6$ </td><td> $9 9 . 7 3 \pm 0 . 2 3 $ </td><td> $\underline { { 0 . 8 8 3 } } \pm 0 . 0 0 1$ </td></tr><tr><td>Uniform</td><td>Linear</td><td> $8 0 . 6 0 \pm 1 . 4 1$ </td><td> $9 9 . 2 7 \pm 0 . 2 1 $ </td><td> $9 8 . 8 9 \pm 0 . 4 4$ </td><td> $0 . 8 6 5 \pm 0 . 0 0 1$ </td></tr><tr><td>Uniform</td><td>Cosine</td><td> $8 0 . 2 7 \pm 0 . 3 2$ </td><td> $9 9 . 3 0 \pm 0 . 4 6$ </td><td> $9 8 . 4 2 \pm 0 . 7 2$ </td><td> $0 . 8 6 7 \pm 0 . 0 0 1$ </td></tr><tr><td>Uniform</td><td>Adaptive</td><td> $7 5 . 7 0 \pm 1 . 8 7$ </td><td> $9 8 . 4 3 \pm 0 . 4 2 $ </td><td> $9 8 . 5 8 \pm 0 . 2 7$ </td><td> $0 . 8 7 3 \pm 0 . 0 0 3$ </td></tr><tr><td>Uniform (time cond.)</td><td>Linear</td><td> $8 3 . 0 3 \pm 1 . 5 0 $ </td><td> ${ \bf 9 9 . 8 0 \pm 0 . 0 0 }$ </td><td> $9 9 . 7 3 \pm 0 . 2 1$ </td><td> $0 . 8 5 9 \pm 0 . 0 0 3$ </td></tr><tr><td>Uniform (time cond.)</td><td>Cosine</td><td> $8 2 . 8 0 \pm 0 . 3 6$ </td><td> $9 9 . 5 7 \pm 0 . 0 6$ </td><td> $9 9 . 8 7 \pm 0 . 1 2$ </td><td> $0 . 8 6 0 \pm 0 . 0 0 1$ </td></tr><tr><td>Uniform (time cond.)</td><td>Adaptive</td><td> $7 3 . 8 0 \pm 0 . 5 3$ </td><td> $9 8 . 3 7 \pm 0 . 1 2$ </td><td> $9 8 . 6 8 \pm 0 . 5 4$ </td><td> $0 . 8 7 3 \pm 0 . 0 0 2$ </td></tr><tr><td>Masked</td><td>Linear</td><td> $\underline { { 8 4 . 8 3 \pm 0 . 6 8 } }$ </td><td> $9 9 . 6 7 \pm 0 . 1 2$ </td><td> $9 9 . 9 0 \pm 0 . 1 0 $ </td><td> $0 . 8 5 5 \pm 0 . 0 0 2$ </td></tr><tr><td>Masked</td><td>Cosine</td><td> $\mathbf { 8 4 . 9 3 \pm 0 . 7 8 }$ </td><td> $9 9 . 7 7 \pm 0 . 3 2 $ </td><td> $9 9 . 7 7 \pm 0 . 0 6$ </td><td> $0 . 8 5 4 \pm 0 . 0 0 1$ </td></tr><tr><td>Masked</td><td>Adaptive</td><td> $8 1 . 1 7 \pm 1 . 6 5$ </td><td> $\overline { { 9 9 . 2 0 \pm 0 . 0 0 } }$ </td><td> $9 9 . 8 3 \pm 0 . 1 5$ </td><td> $0 . 8 6 3 \pm 0 . 0 0 1$ </td></tr></table>

Table 15: Unconditional de novo molecular generation on SAFE-GPT with confidencebased (conf.) sampling following the GenMol evaluation protocol (Lee et al., 2025) (1,000 molecules/seed, mean ˘ std over 3 seeds). Bold: best; underline: second best among our runs. All our runs use 256 sampling steps.
<table><tr><td>Method</td><td>Schedule</td><td>Quality (%) ↑</td><td>Validity (%)↑</td><td>Uniqueness (%)↑</td><td>Diversity ↑</td></tr><tr><td colspan="6">Published Reference (Lee et al., 2025)</td></tr><tr><td>SAFE-GPT</td><td></td><td> $5 4 . 7 \pm 0 . 3$ </td><td> $9 4 . 0 \pm 0 . 4$ </td><td> $1 0 0 . 0 \pm 0$ </td><td> $0 . 8 7 9 \pm 0 . 0 0 1$ </td></tr><tr><td>GenMol</td><td></td><td> $8 4 . 6 \pm 0 . 8$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $9 9 . 7 \pm 0 . 1 $ </td><td> $0 . 8 1 8 \pm 0 . 0 0 1$ </td></tr><tr><td colspan="6">Our Run Comparison</td></tr><tr><td>Autoregressive (AR)</td><td></td><td> $6 3 . 9 7 \pm 1 . 6 7$ </td><td> $9 6 . 3 7 \pm 0 . 2 1$ </td><td> $9 9 . 7 6 \pm 0 . 0 6$ </td><td> $\mathbf { 0 . 8 8 3 \pm 0 . 0 0 2 }$ </td></tr><tr><td>Simplex</td><td>Linear</td><td> $8 3 . 8 7 \pm 1 . 3 6$ </td><td> $9 9 . 8 0 \pm 0 . 1 0$ </td><td> $9 9 . 8 7 \pm 0 . 1 2$ </td><td> $0 . 8 5 9 \pm 0 . 0 0 0$ </td></tr><tr><td>Simplex</td><td>Cosine</td><td> $8 3 . 5 7 \pm 1 . 5 9$ </td><td> $9 9 . 7 7 \pm 0 . 1 5$ </td><td> $9 9 . 8 7 \pm 0 . 1 5$ </td><td> $0 . 8 6 2 \pm 0 . 0 0 1$ </td></tr><tr><td>Simplex</td><td>Adaptive</td><td> $8 5 . 8 7 \pm 0 . 7 0$ </td><td> $9 9 . 6 3 \pm 0 . 3 1 $ </td><td> $\pm \mathbf { 0 . 9 3 } \pm \mathbf { 0 . 0 6 }$ </td><td> $0 . 8 5 9 \pm 0 . 0 0 1$ </td></tr><tr><td>Simplex (time cond.)</td><td>Linear</td><td> $8 5 . 8 7 \pm 0 . 6 7$ </td><td> $9 9 . 6 3 \pm 0 . 3 5$ </td><td> $9 9 . 8 0 \pm 0 . 1 0$ </td><td> $0 . 8 5 8 \pm 0 . 0 0 1$ </td></tr><tr><td>Simplex (time cond.)</td><td>Cosine</td><td> $8 5 . 6 0 \pm 0 . 8 5$ </td><td> $9 9 . 6 7 \pm 0 . 0 6$ </td><td> $9 9 . 7 0 \pm 0 . 2 0$ </td><td> $0 . 8 5 8 \pm 0 . 0 0 0$ </td></tr><tr><td>Simplex (time cond.)</td><td>Adaptive</td><td> $8 7 . 2 3 \pm 0 . 7 1$ </td><td> $9 9 . 8 7 \pm 0 . 1 2$ </td><td> $9 9 . 7 7 \pm 0 . 0 6$ </td><td> $0 . 8 5 7 \pm 0 . 0 0 1$ </td></tr><tr><td>Uniform</td><td>Linear</td><td> $8 5 . 2 3 \pm 0 . 9 3$ </td><td> $9 9 . 3 7 \pm 0 . 2 3 $ </td><td> $9 8 . 5 6 \pm 0 . 5 9$ </td><td> $0 . 8 5 8 \pm 0 . 0 0 3$ </td></tr><tr><td>Uniform</td><td>Cosine</td><td> $8 6 . 0 3 \pm 1 . 1 8$ </td><td> $9 9 . 6 3 \pm 0 . 0 6$ </td><td> $9 7 . 6 9 \pm 0 . 2 7$ </td><td> $0 . 8 5 7 \pm 0 . 0 0 1$ </td></tr><tr><td>Uniform</td><td>Adaptive</td><td> $8 3 . 6 0 \pm 0 . 2 6$ </td><td> $9 9 . 2 7 \pm 0 . 3 1 $ </td><td> $9 7 . 0 1 \pm 0 . 5 1 $ </td><td> $\underline { { 0 . 8 6 3 } } \pm 0 . 0 0 1$ </td></tr><tr><td>Uniform (time cond.)</td><td>Linear</td><td> $8 9 . 6 0 \pm 0 . 3 6$ </td><td> ${ \bf 9 9 . 9 3 \pm 0 . 1 2 }$ </td><td> $9 9 . 2 0 \pm 0 . 4 0$ </td><td> $0 . 8 4 6 \pm 0 . 0 0 2$ </td></tr><tr><td>Uniform (time cond.)</td><td>Cosine</td><td> $9 1 . 9 7 \pm 0 . 7 0$ </td><td> $9 9 . 9 0 \pm 0 . 1 0 $ </td><td> $9 9 . 3 0 \pm 0 . 2 0 $ </td><td> $0 . 8 3 5 \pm 0 . 0 0 1$ </td></tr><tr><td>Uniform (time cond.)</td><td>Adaptive</td><td> $9 1 . 5 0 \pm 1 . 1 1 $ </td><td> $\overline { { 9 9 . 8 7 \pm 0 . 0 6 } }$ </td><td> $9 9 . 3 7 \pm 0 . 1 5$ </td><td> $0 . 8 3 9 \pm 0 . 0 0 2$ </td></tr><tr><td>Masked</td><td>Linear</td><td> ${ \bf 9 2 . 8 0 \pm 0 . 2 6 }$ </td><td> $9 9 . 9 0 \pm 0 . 1 0 $ </td><td> $9 9 . 5 3 \pm 0 . 3 8 $ </td><td> $0 . 8 2 0 \pm 0 . 0 0 0$ </td></tr><tr><td>Masked</td><td>Cosine</td><td> ${ \bf 9 2 . 8 0 \pm 0 . 6 1 }$ </td><td> $\overline { { 9 9 . 7 7 \pm 0 . 0 6 } }$ </td><td> $9 9 . 8 3 \pm 0 . 1 2$ </td><td> $0 . 8 1 9 \pm 0 . 0 0 1$ </td></tr><tr><td>Masked</td><td>Adaptive</td><td> $9 2 . 0 0 \pm 1 . 0 5$ </td><td> $9 9 . 7 7 \pm 0 . 1 2$ </td><td> $9 9 . 8 3 \pm 0 . 0 6$ </td><td> $0 . 8 2 3 \pm 0 . 0 0 1$ </td></tr></table>

Table 16: Unconditional 1024-token generation from Simplex Diffusion Model trained on OWT (Sample 1).
<table><tr><td>Sample 1 (1024 tokens)</td><td>GenPPL (GPT-2 Large): 14.15 — Unigram Entropy: 5.27 — Distinct-2: 0.724</td></tr><tr><td colspan="2">bring peace and stability Middle East,&quot; he told reporters. &quot;We&#x27;re all going to be hopeful if we&#x27;re part of an agreement that is retroactively . .. And I think the Democratic Party will be up for grabs if she takes it seriously.&quot;</td></tr><tr><td colspan="2">Republican strategist Steve Bannon also criticized Trump&#x27;s comments earlier this week, saying that he had a &quot;waste for American diplomacy&quot; by &quot;trying to burn out sanctions relief with Iran.</td></tr><tr><td colspan="2">New York Times columnist Hugh Hewitt, also a former State Department official and a member of the Republican National Committee, says Trump is trying to forge a deal with Iran without ever breaching international sanctions.</td></tr><tr><td colspan="2">“I have a lot of miscalculations because I think they&#x27;ve succeeded in a bad engagement&quot; Hewitt told The New York Times, adding that the administration needs to understand the terms of the deal.</td></tr><tr><td colspan="2">&quot;I think I don&#x27;t think this deal will be bad, but it&#x27;s hard — which is very hard, very hard — to fix it,&quot; he added. &quot;What can we do? Can we depolarize the rest of the Middle East?&quot;</td></tr><tr><td colspan="2">10 p.m.</td></tr><tr><td colspan="2">North Korea&#x27;s government says it backs Republican presidential candidate Donald Trump for setting a new tone with his criticism of North Korea&#x27;s state-controlled media, in a move at odds with Washington rhetoric over its pursuit of nuclear weapons.</td></tr><tr><td colspan="2">South Korean Foreign Minister Kim Kyung-seo responded in a statement calling Trump &quot;reckless and reckless.&quot; He criticizes Trump on North Korea, calling him a &quot;highly dangerous man for the President of the United States.&quot;</td></tr><tr><td colspan="2">He also said he would not talk to North Korea without doubting its nuclear weapons abroad. &quot;Without such a strong leadership, any threat from the North Korean regime would be rejected,&quot; he said. South Korea has said it banned Pyongyang&#x27;s nuclear programs in the past because it would stop North Koreans from leaving the country without their weapons.</td></tr><tr><td colspan="2">9:30 p.m. North Korean military leader Jang Song-thaek says Trump &quot;totally impoundately&quot; the U.S. nuclear weapons program because it was &quot;totally</td></tr><tr><td colspan="2">fragile&quot; after the United States dropped an H-bomb on the city of Nagasaki, according to an Air Force One video released Tuesday. He also said he hoped that the U.S. would make a good return solely to the negotiating table.</td></tr><tr><td colspan="2">He also called for North Korea to get serious about halting its program, although the U.S. has not sanctioned or formally approved its nuclear- weapons program since August 1950. He also urged the United States to learn a lesson from history.</td></tr><tr><td colspan="2">Trump said on Tuesday that the U.S. must getting ready to restart its nuclear program. He said restarting its work with North Korea could mean</td></tr><tr><td colspan="2">more years to begin in the coming months &quot;It will be very gradual,&quot; he said. “And I think that will not be abrupt — and it will will be gradual — until the United States can get back on</td></tr><tr><td colspan="2">the table.&quot;</td></tr><tr><td colspan="2">8:30 p.m. Yuri Yuri, South Korea&#x27;s former deputy prime minister, says Donald Trump believes the U.S. should continue to unravel its nuclear program</td></tr><tr><td colspan="2">but is a “preaching and grave mistake.&quot; His remarks followed a recent high-level meeting between Japan and South Korea at an annual summit in Seoul.</td></tr><tr><td colspan="2">Yuri, who is South Korea&#x27;s first president, has criticized the U.S. should apologize for dropping two atomic bombs on Hiroshima during World</td></tr><tr><td colspan="2">War II. He said the U.S. should take steps necessary to try to halt its nuclear program. 8:45 p.m.</td></tr><tr><td colspan="2">Former Republican presidential candidate Donald Trump said on Tuesday that his country&#x27;s nuclear and missile programs should hold a meeting to discuss “a peaceful solution.&quot;</td></tr><tr><td colspan="2">Trump made similar comments about his country&#x27;s relationship with the United States, which has long labeled its nuclear weapons and other NATO allies are &quot;obsolete.&quot;</td></tr><tr><td colspan="2">North Korean leader, Kim Il Jong-un, had earlier this month accused China of cutting off off the country&#x27;s nuclear program during talks with other parties for peace talks.</td></tr><tr><td colspan="2">Trump&#x27;s comments have been seen as provocative by some over his tough stance toward China and his controversial unification policy. But South Korean Foreign Minister Jang Song-thaek issued a statement Tuesday saying the issue was non-negotiable and adding that nuclear</td></tr></table>

Table 17: Unconditional 1024-token generation from Simplex Diffusion Model trained on OWT (Sample 2).
<table><tr><td>Sample 2 (1024 tokens)</td><td>GenPPL (GPT-2 Large): 19.52 — Unigram Entropy: 5.59 —</td></tr><tr><td>after it was advertised as a “gender-only&quot; service run by a non-binary woman. The ad came less than two hours after a video featuring a YouTube user called Avoid Open Door Wicked, which featured 12 women, 14 men, 11 men and 10 women was posted online on its website. &quot;I have been harassed 400+ times because I am surrounded by a hostile environment directed at me,&quot; the 23-year-old woman wrote. &quot;I can&#x27;t believe it,&quot; read one of the ads, which read &quot;Divorce is in my veil!&quot; Another added: &quot;You are not a misogynist, and you cannot use your own operating system to escape your own travails.&quot;</td><td></td></tr><tr><td>&quot;If you were a non-binary woman would you be captive for your own sexual desires? If you&#x27;d you were a woman woman would you be captive for your desires?&quot; asked ACLU attorney Jennifer Partridge, who filed the case through the Justice Department&#x27;s nonprofit Philanthropy Law Project.</td><td></td></tr><tr><td>&quot;It&#x27;s reflective of what we kind of social network is about and whether it is a real issue,&quot; Partridge added</td><td>Courtney ACLU attorneys argued that the company&#x27;s discrimination against harassment and gender-based bias could violate her legal rights.</td></tr><tr><td>&quot;We do not substantially suggest that gender discrimination is not a real issue,&quot; she wrote her brief. “This ad does not fit the context of our social Facebook or Google ads.&quot; Partridge noted that the company&#x27;s policy — which requires customers to purchase or sell their products</td><td></td></tr><tr><td>Partridge said in an email. &quot;This is not an abstract issue.&quot;</td><td>online — suggests more for women than it does for men. The company also sells products in the same retail department stores. &quot;One could argue that this product based on gender-based disrespect is far more sanctimonious what people might buy from a store online,&quot;</td></tr><tr><td>said. “I don&#x27;t think another person should be discriminated against.&quot;</td><td>The company&#x27;s justification is that its site must engage users to &#x27;disconnect&#x27; amongst psychological issues. “I don&#x27;t think it should,&quot; Partridge</td></tr><tr><td>potential trespassing. Story continues below advertisement — Hillary Clinton has maintained her commanding lead in the presidential race</td><td>The company has also said that its decision to remove names or corporate logos from all ads on the site because it doesn&#x27;t want to deter any</td></tr><tr><td>over Republican presidential nominee Donald Trump. Clinton says Trump leads her by 39%, according to a new NBC News/Wall Street Journal Journal poll that shows Republican presidential nominee Donald Trump with the largest-ever lead in any presidential election.</td><td></td></tr><tr><td></td><td>The poll, conducted by Public Opinion Strategies, a long-time polling firm, surveyed 1,000 likely voters. It has a margin of error at 3.0 points</td></tr><tr><td>with the total sample of 1,000. Clinton gets 43 percent of the vote behind Green Green Party candidate Jill Stein (38 percent), while Trump gets 41 (36 percent) and Jill Stein</td><td></td></tr><tr><td>(35 percent). Under the case for both candidates, Stein would get 33.9 percent of the vote.</td><td>The poll predicts Clinton would win over Trump (34 percent), while Stein would get 9.7 percent ahead of Stein (8 percent) Stein/Garrabee (6</td></tr><tr><td>percent). candidate. In November, when she joined Barack Obama&#x27;s national security team, she said she would need to cast her first female vote in the</td><td>Despite last week&#x27;s debate announcing her bid for the presidency, Clinton has implied that she would support any future Republican presidential</td></tr><tr><td>U.S. Senate to serve as president. The poll also shows Trump leads all other Republican candidates with 34% support, while Texas Sen. Ted Cruz leads the presumptive GOP</td><td></td></tr><tr><td>nominee with just 15%. Trump has previously abandoned former Democratic Secretary of State Hillary Clinton after launching an unsuccessful bid to revive his</td><td></td></tr><tr><td>also suggested that he might be tempted to endorse Clinton if his party didn&#x27;t back him up. Story continues below advertisement.</td><td>campaign. He himself, however, has endorsed the presumptive GOP nominee, admitting that he still had no intention of securing support. He</td></tr><tr><td>between Oct. 19 through Nov. 20, 2012, with Barack Obama and Mitt Romney registered among likely voters. The margin of error is plus or</td><td>The polls are conducted by landline and automated landline telephone interviews with 1,000 likely voters nationwide and were conducted</td></tr><tr><td>minus 3.5 points. Also on HuffPost: Two new skyscraper plans will bring some of Manhattan&#x27;s tallest skyline to the rest of the world, after Madison Square Park</td><td></td></tr></table>

Table 18: Unconditional 1024-token generation from Simplex Diffusion Model trained on OWT (Sample 3).
<table><tr><td>Sample 3 (1024 tokens)</td><td>GenPPL (GPT-2 Large): 19.53 Unigram Entropy: 5.23 Distinct-2: 0.645</td></tr><tr><td>Section 494 Miscellaneous Related Articles Section 494 Miscellaneous. Section 493 Miscellaneous Articles Section 494 Miscellaneous Ar- ticles Sections 495 Miscellaneous Sections 487 Miscellaneous Sections 488 Miscellaneous Sections 489 24/24 Sections 4811 Miscellaneous Sections 4812 24/24 4813 Miscellaneous Sections 4812 24/24 4813 Miscellaneous Section 4812.</td><td>Related Articles Section 491. Related Articles Section 492 Miscellaneous Related Articles Section 493 Miscellaneous. Section 493. Related</td></tr><tr><td>(a) It shall be unlawful as a citizen of the United States or a foreign territory of the United States; (a) may conduct, including but not not limited to as a citizen of the United States; (b) as an ex-concitizen of the State of Hawaii or a foreign territory; or (b) as a non-citizen of Hawaii or a former citizen of the United States; (ii) perform other activities as defined in this title. If such an individual does not wish to sign up for</td><td rowspan="2">annual free membership or holiday free walks in an area outside Hawaii, it shall be unlawful to use internet services, advertisements newspaper</td></tr><tr><td>articles, or witherbecoming to advertise such activities. Any person who knowingly giving birth to another person under this same title (or any other person under this title or regulations) shall be penalized.</td></tr><tr><td>he become a resident of a foreign territory, or a Territory of another state or vice versa, there is no penalty for violation of this above law. If it also is found that a person who denies giving birth occurs at a site of birth will he is a resident of a foreign territory, or Foreign Territory of the State.</td><td></td></tr><tr><td>that such person knowingly violates the above law. The incurring any violation of the above law or regulations may be taken pursuant to the provisions of the Rawled &quot;Hawaii&quot; Virgin Islands Act of 2018. Hawaiian and native Hawai&#x27;i people: Hawaiian is a Hawaiian language which is a native Hawaiian language that has been traditionally</td><td>To prohibit any person visiting a foreign territory or from becoming a resident of any territory outside the United States, I am hereby directing</td></tr><tr><td>people are descendants of Native Hawaiian people who are descended from Native Hawaiian tribes, and since then they have been interceded by tribes from other Native Hawaiian tribes. A permanent Hawaiian resident or permanent Hawaiian resident who resides in Hawaii becomes a permanent Hawaiian resident shall continue</td><td>associated with the Hawai&#x27;i people. It is nevertheless traditional Hawaiian language spoken by native Hawai&#x27;i people. The Native Hawaiian</td></tr><tr><td>resident order obtain permission from his residence in Hawaii for a specified time period. Permanent Hawaiian residents shall be required to reside before a Hawai court. A permanent Hawaiian resident shall continue to become legal citizens of Poly Hawaiians because they are native Hawai&#x27;i residents who are non-institutionalized employment and who operate their own agricultural enterprises.</td><td>permanent Hawaiian resident shall be required surrender reside in Hawaii court for specified time period, and if granted permanent Hawaiian</td></tr><tr><td>amended by sections VII, II and Article XII, the Constitution of the Republic on January 1, 2016. Article XV01 U.S. Constitution of Samoa — The Pacific Ocean Rights Act of Samoa Act: This act was performed on January 1, 2016 while</td><td>Article XV01 of the Civil Rights Act of Samoa: I am proposing to amend section article XV01 of the Civil Rights Act of Samoa and shall be</td></tr><tr><td>amended by section VI. The American Samoa Act: This Act shall be amended in section Article XIII, the Pacific Ocean Rights Act as amended by section XXVIII.</td><td>facilitating the Civil Rights Act of Samoa. This Act shall be amended in section I, II, sections XVI, and XVI, the Constitution of Samoa as</td></tr><tr><td>This Act shall be amended in section Article XIV, the Constitution of the United States as amended by section VI. Section 18017 U.S. Constitution of the United States of America: Nothing shall be made construed unlawful for any county, state, political subdivision of any country or any other State whatsoever to employ any person in the United States of America. Section 18017 U.S. Constitution</td><td></td></tr><tr><td>Indigenous Indian Americans, immigrants, citizens of the United States. Corrections / Proions: Section 17:20 Sec. 2 — the Constitution 1/3 dated January 1, 2017. Section 17:20 Sec. 3 — the text of the Constitution 1/3, dated January 1, 2017. (Proclamation). Section 17:20 Sec. 3</td><td></td></tr></table>

Table 19: Unconditional 1024-token generation from Simplex Diffusion Model trained on OWT (Sample 4).
<table><tr><td>Sample 4 (1024 tokens)</td><td>GenPPL (GPT-2 Large): 20.85 — Unigram Entropy: 5.60 一 Distinct-2:0.848</td></tr><tr><td>Commissions are come under a special agreement, which allows member states to be analysed and then sell all of the emissions emitted from</td><td>efficient price for carbon emitted from renewable energy rests with the European Commission, spokesman Richard Ritter said.</td></tr><tr><td>other EU states.</td><td>Under this agreement member states would buy emission-free allowances. The member states could then decide to sell them only if they</td></tr><tr><td>couldn&#x27;t afford to buy additional items.</td><td>For emissions-free under the new agreement, those allowances from Germany would have to give up their market share, said Ritter.</td></tr><tr><td></td><td>“The Commission will henceforth refuse to use emission-free allowances as an exercise for Germany or for Jean Juniet,&quot; Ritter said. “There</td></tr><tr><td>are other alternatives I would advise that we should be able to bring them into the domestic market, but not at all.&quot; Ritter added that the EU would need to remove emission-free allowances from its domestic market before sending them back to emission-free</td><td></td></tr><tr><td>markets. He added that such plans were approved by the German parliament. German MEP Bart Schoto told reporters on Sunday that there is no need for a new deal. &quot;It&#x27;s just a simple legislative process,&quot; he said. He</td><td></td></tr><tr><td>suggested emission-free allowances should be sold. &quot;I don&#x27;t think so much but but leaving is not really a problem,&quot; he said. &quot;If we think we will be able to keep up the cost of our growing economy</td><td></td></tr><tr><td>then I think this pathway will be very difficult.&quot; Other measures must be taken through the new emissions agreement, Ritter said. The measures could include building incentives to build</td><td></td></tr><tr><td>costs. &quot;We will have to expand our capacity if we can&#x27;t get our policies back into place,&quot; he said. &quot;It&#x27;s going to be a very tough decision and it will</td><td>nuclear power plants, which could cost about $1 billion a year plus subsidies for nuclear power plants in South Florida, which could raise fuel</td></tr><tr><td>be taken out as soon as possible,&quot; he said Merkel&#x27;s action &#x27;unemptive&#x27;: Juncker, who was arrested in Brussels in connection with his meeting between the two leaders and German</td><td></td></tr><tr><td>Reinforce Eisenberg at a press conference in Berlin last week.</td><td>Chancellor Angela Merkel, met privately with former Chancellor Gerhard Schroeder, the former Bavarian prime minister and medical doctor</td></tr><tr><td>rule imposed by the European Union. On Friday, Germany&#x27;s state energy company (SDF) and its state utility EDF — which has been the main target for calls for divesting coal, said</td><td>The commission&#x27;s decision has been fraught with political tensions. However, Germany has appealed to Brussels against a form of one-party</td></tr><tr><td>it would not move forward. Eisenberg said it was highly unlikely that Merkel would make any effort to change her position.</td><td></td></tr><tr><td>with Der Spiegel.</td><td>&quot;This is a step in a right direction,&quot; said Marguerte Schweiko, the head of the German daily newspaper Vorb Mauschaftung in an interview</td></tr><tr><td>friends, who control the world&#x27;s most valuable fossil fuel reserves.&quot;</td><td>&quot;The chancellor has clearly changed position... She has already shown herself capable of exerting power without the popular support of her</td></tr><tr><td>“&quot;Merkel is an anti-nuclear campaigner... She has made it clear that she did not actually participate in any concrete action.&quot;</td><td>Schweiko also said Merkel her actions are “unemptive&quot;, adding that “the German government played a principal role in cultivation of tools for</td></tr><tr><td>climate change&quot;. She added that Merkel had pressure on many Germans. “Mrs Merkel seems to have lost her credibility in many ways she cannot stand trusted,&quot;she said.</td><td></td></tr><tr><td>inconceivable that all parties can agree on a legally binding treaty—even within the framework of the parliament—but it documents highly polarised,&quot; the statement read.</td><td>Earlier this week, the German Green Party, Christian Saar, issued a statement condemning “polarly polarized&quot; climate negotiations. “It is</td></tr><tr><td>that are realistic about climate change, as well as other countries that have warned against taking climate action.</td><td>The EU has started undertaking its own voluntary emissions trading scheme, making matters more pressing among some European countries</td></tr><tr><td>an opportunity to kick-start talks in Paris on climate change.</td><td>Germany may have already signed up to reduce carbon emissions, meaning that still does not have the resources to do so. But it may also have</td></tr><tr><td>Later this year, Europe and the major United States are expected to host talks on agriculture and climate change. Their involvement The Paris talks is scheduled to begin at the end of 2012, months after the Kyoto Protocol took place in Copenhagen in December 2009.</td><td></td></tr></table>

Table 20: Unconditional 1024-token generation from Simplex Diffusion Model trained on OWT (Sample 5).
<table><tr><td>Sample 5 (1024 tokens)</td><td>GenPPL (GPT-2 Large): 4. 04 Unigram Entropy: 3.56</td></tr><tr><td colspan="2">Commandments. 300. 325 325. The Son of God, His Son, His Father and the Commandments. 300. 325 325. The Son of God, The Son of the Holy and Father and the Commandments.</td></tr><tr><td colspan="2">300. 325 325 Elijah Elijah is not commanded by The Lord of Christ, His Father and the Commandments.</td></tr><tr><td colspan="2">300. 325 325 Elijah is not commanded by Jesus Christ, His Father, the Commandments.</td></tr><tr><td colspan="2">300. 325 325. Of The Lord of Christ, His Father and the Commandments. 300. 325. The Lord of Christ, His Son and the Commandments.</td></tr><tr><td colspan="2">300. 325 325. The Lord of Christ, His Father and the Commandments. 301. 325 The Lord of Christ, His Father and the Commandments. 301. 325. The Lord of Christ, His Father, the Commandments.</td></tr><tr><td colspan="2">325 325. Of The Lord&#x27;s son, His Father, the Commandments. 301. the Lord Elijah is not commanded by The Lord of son, His Father, the Commandments. 301. the Lord Elijah is not commanded by The</td></tr><tr><td colspan="2">Lord&#x27;s son, Father, the Commandments. 301. the Lord Elijah is not being commanded by the Lord&#x27;s Christ, His Son, His Father and the Commandments. 301. the Lord Elijah is not</td></tr><tr><td colspan="2">being commanded by The Lord&#x27;s son, His Father and the Commandments.</td></tr><tr><td colspan="2">302. 304 308 308. The Son of God to His Father and the Commandment 351. 303. 308 308 The Son of God, Maker Commandments. 304. 328 309. Jesus God to His Father and the Commandment 351. 303. 328 309. Jesus Lord to be Father, Maker Commandments.</td></tr><tr><td colspan="2">328 309. Jesus Lord to be Stronger. 305. 328 309. The Son of God. Maker Commandment. 306. 328 309 309. The Son of Christ, His Father, the Commandments.</td></tr><tr><td colspan="2">309. The Lord to be Strong, Maker Commandments.</td></tr><tr><td colspan="2">307. 328. The Son of Christ, His Father, the Commandments. 308. 328 328. The Son of Christ, His Father, the Commandments. 328 328. The Lord to be Stronger.</td></tr><tr><td colspan="2">322. 329 330. The Son of Christ, His Father, the Commandments. 323. 329 330. Jesus The Son of God, Maker Command. Jesus Lord to be Strong. 323. 329 330. The Son of God, Maker Commandments.</td></tr><tr><td colspan="2">324. 329 331 Lose! Of The Son&#x27;s God, His Father and the Commandments. 326. 329 331. The Son of Christ, His Father and the Commandments.</td></tr><tr><td colspan="2">325. 320 315 315. Jesus The Son of God, His Exalted Father and the Commandments. 326. 320 320. Jesus the Son of Christ, His Father and the Commandments.</td></tr><tr><td colspan="2">325 320 320. Jesus Christ to the Father, the Commandment 351. 325 320 320. The Son of Christ to the Father and the Commandment 351. 327. 320 320. The Son of Christ to the Father and the Commandments.</td></tr><tr><td colspan="2">349. 321 321. The Son of God, Maker Command. 350. 321 321. Jesus Christ to He Father, Maker Commandment. 351. 321 321. Jesus the Son of God to His Father, the Commandment 351.</td></tr><tr><td colspan="2">352 321 321. The Son of Christ to His Father, the Commandment 351. 352. 321 321. The Son&#x27;s God, His Exalted Father and the</td></tr><tr><td colspan="2">Commandment 351. 353. 321 321. The Son of Christ to His Father, the Commandment 351. 352. 412 355. The Son of Christ to His Father, the Commandment 351. 352. 412 355. The Son of God, Maker Commandment.</td></tr><tr><td colspan="2">355. The Lord to be Strongerful. 354. 412 355. The Lord of his Father, Maker Commandments.</td></tr><tr><td colspan="2">355. 412 412. Jesus The Son of Christ, the Father and the Commandments. 355. 412 412. The Son of Christ, the Father and the Commandments. 356. 412 412. Jesus Christ to the Father, Maker Commandments.</td></tr><tr><td colspan="2">357. 412 410. The Son of Christ, His God, the Commandment 351. 356 412 412 410. Jesus Christ to the Father, Maker Commandments.</td></tr><tr><td colspan="2">357. 412 410. The Son of God, the Father and the Commandments.</td></tr><tr><td colspan="2">358. 412 412 410. The Lord&#x27;s God, the Father and Commanders. 358. 412 410. The Son of God, Maker Commandments. Jesus Christ to the Father, the Commandments. 357. 412 414. The Son of Christ, the Father and the Commandments. 358. 412 415. It</td></tr></table>

## M FULL RESULT TABLES

## M.1 SUDOKU

Table 21: Accuracy (%) on Sudoku in 180 steps for Simplex Diffusion Models with Churn $\kappa = 0 . 0$ across sampling schedules. Values are reported as $\mathrm { { \ m e a n } _ { \pm \mathrm { { s t d } } } }$ over 5 seeds. Column-wise top scores are bolded, and overall best for $\kappa = 0 . 0$ is underlined
<table><tr><td>Concentration Schedule</td><td>Linear</td><td>Cosine</td><td>Adaptive</td></tr><tr><td>Uniform time sampler</td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td> $6 2 . 8 _ { + 5 . 0 }$ </td><td> $6 2 . 4 _ { + 4 . 7 }$ </td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td> $7 4 . 0 _ { + 4 . 4 }$ </td><td> $7 3 . 7 _ { + 4 . 3 }$ </td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td> $7 3 . 9 _ { + 4 . 7 }$ </td><td> $7 4 . 0 _ { + 4 . 7 }$ </td><td></td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td> $6 4 . 9 _ { + 3 . 7 }$ </td><td> $6 4 . 6 \substack { + 3 . 1 }$ </td><td></td></tr><tr><td>Constant  $( \nu = 0 . 2 5 )$ </td><td> $5 6 . 0 _ { + 6 . 2 }$ </td><td> $5 5 . 6 _ { + 5 . 8 }$ </td><td></td></tr><tr><td>Adaptive time sampler</td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td> $7 8 . 5 _ { + 2 . 1 }$ </td><td> $7 8 . 1 _ { + 2 . 3 }$ </td><td> $7 5 . 3 _ { + 2 . 5 }$ </td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td> $7 9 . 4 \substack { + 2 . 6 }$ </td><td> $7 9 . 2 _ { + 2 . 7 }$ </td><td> $7 7 . 2 \substack { + 3 . 1 }$ </td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td> $7 7 . 1 _ { + 2 . 5 }$ </td><td> $7 6 . 9 _ { + 2 . 4 }$ </td><td> $7 5 . 1 _ { + 2 . 8 }$ </td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td> $7 8 . 0 _ { + 2 . 0 }$ </td><td> $7 7 . 9 _ { + 2 . 1 }$ </td><td> $7 4 . 2 _ { + 2 . 0 }$ </td></tr><tr><td>Constant (ν = 0.25)</td><td> $7 6 . 0 _ { + 3 . 5 }$ </td><td> $7 5 . 5 \substack { + 3 . 6 }$ </td><td> $7 2 . 4 \substack { + 4 . 2 }$ </td></tr><tr><td>Self-Conditioning + Uniform time sampler</td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td> $9 7 . 4 _ { + 2 . 3 }$ </td><td> $9 4 . 6 _ { + 6 . 6 }$ </td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td> $9 6 . 9 _ { + 3 . 1 }$ </td><td> $9 4 . 0 _ { + 9 . 1 }$ </td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td> $\underline { { 9 9 . 2 } } _ { + 0 . 2 }$ </td><td> $\mathbf { 9 9 . 1 _ { + 0 . 2 } }$ </td><td></td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td> $9 8 . 7 _ { + 0 . 6 }$ </td><td> $9 8 . 1 _ { + 1 . 3 }$ </td><td></td></tr><tr><td>Constant  $( \nu = 0 . 2 5 )$ </td><td> $9 7 . 6 _ { + 1 . 7 }$ </td><td> $9 6 . 5 _ { + 3 . 4 }$ </td><td></td></tr><tr><td>Self-Conditioning + adaptive time sampler</td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td> $9 1 . 6 _ { + 1 7 . 3 }$ </td><td> $8 8 . 5 _ { + 2 4 . 3 }$ </td><td> $8 9 . 7 _ { + 2 1 . 7 }$ </td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td> $9 4 . 0 _ { + 1 0 . 1 }$ </td><td> $8 5 . 4 _ { + 2 4 . 8 }$ </td><td> $8 9 . 5 _ { + 1 7 . 9 }$ </td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td> $9 7 . 3 _ { + 4 . 5 }$ </td><td> $9 5 . 2 _ { + 9 . 1 }$ </td><td> $9 6 . 6 _ { + 4 . 9 }$ </td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td> $9 9 . 1 _ { + 0 . 2 }$ </td><td> $9 8 . 9 _ { + 0 . 5 }$ </td><td> $\mathbf { 9 8 . 9 } _ { + 0 . 3 }$ </td></tr><tr><td>Constant  $( \nu = 0 . 2 5 )$ </td><td> $9 8 . 1 _ { + 2 . 9 }$ </td><td> $9 7 . 1 _ { + 5 . 0 }$ </td><td> $9 7 . 5 _ { + 4 . 1 }$ </td></tr></table>

Table 22: Accuracy (%) on Sudoku in 180 steps for Simplex Diffusion Models with Churn $\kappa = 0 . 2$ across sampling schedules. Values are reported as mean<sub>˘std</sub> over 5 seeds. Column-wise top scores are bolded, and overall best for $\kappa = 0 . 2$ is underlined
<table><tr><td>Concentration Schedule</td><td>Linear</td><td>Cosine</td><td>Adaptive</td></tr><tr><td>Uniform time sampler</td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td> $8 1 . 1 _ { + 3 . 2 }$ </td><td> $8 1 . 4 _ { + 3 . 3 }$ </td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td> $8 5 . 3 _ { + 3 . 2 }$ </td><td> $8 5 . 4 _ { + 3 . 3 }$ </td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td> $8 0 . 4 _ { + 3 . 7 }$ </td><td> $7 9 . 1 _ { + 3 . 5 }$ </td><td></td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td> $8 3 . 5 _ { + 2 . 0 }$ </td><td> $8 0 . 0 _ { + 1 . 8 }$ </td><td></td></tr><tr><td>Constant (ν = 0.25)</td><td> $7 6 . 6 _ { + 3 . 5 }$ </td><td> $7 7 . 6 _ { + 3 . 5 }$ </td><td></td></tr><tr><td>Adaptive time sampler</td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td> $8 8 . 3 _ { + 1 . 4 }$ </td><td> $8 8 . 0 _ { + 1 . 3 }$ </td><td> $8 5 . 7 _ { + 1 . 6 }$ </td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td> $8 7 . 9 _ { + 1 . 6 }$ </td><td> $8 7 . 6 _ { + 1 . 7 }$ </td><td> $8 5 . 1 _ { + 1 . 9 }$ </td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td> $8 5 . 2 _ { + 1 . 8 }$ </td><td> $8 2 . 9 _ { + 1 . 8 }$ </td><td> $8 2 . 1 _ { + 2 . 1 }$ </td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td> $8 7 . 5 _ { + 2 . 0 }$ </td><td> $8 3 . 6 _ { + 1 . 6 }$ </td><td> $8 4 . 0 _ { + 3 . 2 }$ </td></tr><tr><td>Constant  $( \nu = 0 . 2 5 )$ </td><td> $8 6 . 3 _ { + 2 . 2 }$ </td><td> $8 6 . 1 _ { + 1 . 8 }$ </td><td> $8 4 . 4 _ { + 2 . 4 }$ </td></tr><tr><td>Self-Conditioning + Uniform time sampler</td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td> $9 8 . 3 _ { + 1 . 4 }$ </td><td> $9 5 . 8 _ { + 4 . 5 }$ </td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td> $9 7 . 3 _ { + 2 . 4 }$ </td><td> $9 5 . 8 _ { + 4 . 8 }$ </td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td> $\underline { { 9 8 . 6 } } _ { + 0 . 2 }$ </td><td> $9 5 . 8 _ { + 0 . 6 }$ </td><td></td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td> $9 7 . 8 _ { + 0 . 5 }$ </td><td> $9 2 . 7 _ { + 1 . 1 }$ </td><td></td></tr><tr><td>Constant (ν = 0.25)</td><td> $9 8 . 5 _ { + 0 . 6 }$ </td><td> $\mathbf { 9 7 . 5 _ { \div 1 . 4 } }$ </td><td></td></tr><tr><td>Self-Conditioning + adaptive time sampler</td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td> $8 9 . 8 _ { + 2 1 . 6 }$ </td><td> $8 6 . 6 _ { + 2 8 . 2 }$ </td><td> $8 6 . 9 _ { + 2 8 . 1 }$ </td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td> $9 2 . 7 _ { + 1 3 . 8 }$ </td><td> $8 7 . 7 _ { + 2 2 . 3 }$ </td><td> $8 8 . 5 _ { + 2 1 . 9 }$ </td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td> $9 6 . 7 _ { + 4 . 3 }$ </td><td> $9 2 . 0 _ { + 8 . 3 }$ </td><td> $9 5 . 0 _ { + 5 . 9 }$ </td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td> $9 8 . 2 _ { + 0 . 2 }$ </td><td> $9 3 . 4 _ { + 0 . 3 }$ </td><td> $9 7 . 1 _ { + 2 . 0 }$ </td></tr><tr><td>Constant  $( \nu = 0 . 2 5 )$ </td><td> $9 8 . 4 _ { + 2 . 1 }$ </td><td> $9 7 . 0 _ { + 3 . 6 }$ </td><td> $9 7 . 2 _ { + 4 . 3 }$ </td></tr></table>

Table 23: Accuracy (%) on Sudoku in 180 steps for Simplex Diffusion Models with Churn $\kappa = 1 . 0$ across sampling schedules. Values are reported as $\mathrm { { \ m e a n } _ { \pm \mathrm { { s t d } } } }$ over 5 seeds. Column-wise top scores are bolded, and overall best for $\kappa = 1 . 0$ is underlined
<table><tr><td>Concentration Schedule</td><td>Linear</td><td>Cosine</td><td>Adaptive</td></tr><tr><td>Uniform time sampler</td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td> $8 9 . 3 _ { + 2 . 3 }$ </td><td> $8 9 . 7 _ { + 2 . 3 }$ </td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td> $9 1 . 5 _ { + 1 . 9 }$ </td><td> $9 1 . 6 _ { + 1 . 6 }$ </td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td> $8 8 . 4 _ { + 2 . 2 }$ </td><td> $8 8 . 9 _ { + 2 . 4 }$ </td><td></td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td> $9 0 . 5 _ { + 1 . 1 }$ </td><td> $9 0 . 6 _ { + 1 . 2 }$ </td><td></td></tr><tr><td>Constant (ν = 0.25)</td><td> $8 6 . 9 _ { + 2 . 1 }$ </td><td> $8 7 . 8 _ { + 2 . 5 }$ </td><td></td></tr><tr><td>Adaptive time sampler</td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td> $9 0 . 8 _ { + 0 . 9 } $ </td><td> $9 0 . 8 _ { + 1 . 0 }$ </td><td> $8 9 . 4 _ { + 1 . 0 }$ </td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td> $9 0 . 5 _ { + 1 . 1 }$ </td><td> $9 0 . 3 _ { + 1 . 3 }$ </td><td> $8 8 . 8 _ { + 1 . 5 }$ </td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td> $8 8 . 3 _ { + 1 . 3 }$ </td><td> $8 8 . 5 _ { + 1 . 2 }$ </td><td> $8 6 . 5 _ { + 1 . 4 }$ </td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td> $9 0 . 6 _ { + 1 . 5 }$ </td><td> $9 0 . 4 _ { + 1 . 5 }$ </td><td> $8 8 . 8 _ { + 1 . 8 }$ </td></tr><tr><td>Constant  $( \nu = 0 . 2 5 )$ </td><td> $8 9 . 6 _ { + 1 . 8 }$ </td><td> $8 9 . 8 _ { + 1 . 8 }$ </td><td> $8 8 . 3 _ { + 1 . 9 }$ </td></tr><tr><td>Self-Conditioning + Uniform time sampler</td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td> $9 8 . 9 _ { + 0 . 8 }$ </td><td> $9 7 . 2 \substack { + 2 . 8 }$ </td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td> $9 8 . 3 _ { + 1 . 1 }$ </td><td> $9 6 . 9 _ { + 3 . 4 }$ </td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td> $\underline { { 9 9 . 1 } } _ { + 0 . 2 }$ </td><td> $\mathbf { 9 8 . 7 _ { \pm 0 . 3 } }$ </td><td></td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td> $9 8 . 5 \substack { + 0 . 6 }$ </td><td> $9 7 . 5 \substack { + 1 . 1 }$ </td><td></td></tr><tr><td>Constant (ν = 0.25)</td><td> $\underline { { 9 9 . 1 } } _ { + 0 . 3 }$ </td><td> $9 8 . 5 \substack { + 0 . 8 }$ </td><td></td></tr><tr><td>Self-Conditioning + adaptive time sampler</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { C o n s t . - L i n . ~ } ( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td> $9 1 . 2 _ { + 1 8 . 8 }$ </td><td> $8 7 . 5 \substack { + 2 6 . 6 }$ </td><td> $8 8 . 6 _ { + 2 4 . 4 }$ </td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td> $9 3 . 9 _ { + 1 1 . 9 }$ </td><td> $8 9 . 1 _ { + 2 0 . 7 }$ </td><td> $9 0 . 3 _ { + 1 8 . 7 }$ </td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td> $9 7 . 5 _ { + 3 . 9 }$ </td><td> $9 4 . 8 _ { + 8 . 9 }$ </td><td> $9 6 . 3 _ { + 5 . 4 }$ </td></tr><tr><td>Constant (ν = 0.5)</td><td> $9 8 . 8 _ { + 0 . 2 }$ </td><td> $9 8 . 2 _ { + 0 . 2 }$ </td><td> $\mathbf { 9 8 . 3 _ { + 0 . 2 } }$ </td></tr><tr><td>Constant (ν = 0.25)</td><td> $9 8 . 8 _ { + 1 . 5 }$ </td><td> $9 7 . 8 _ { + 3 . 3 }$ </td><td> $9 8 . 1 _ { + 2 . 9 }$ </td></tr></table>

Table 24: Baseline Models Accuracy (%) on Sudoku in 180 steps across sampling schedules for the Appendix. We evaluate Autoregressive (AR), Discrete Absorbing (MDM with CE and ELBO losses), Discrete Uniform (UDM), Predictor-Corrector (PC), Self-Conditioning (SC / loopholing), Flow Matching over one-hot encoding (FLM with uniform, Gauss-Hermite LUT, and adaptive time samplers), and Hyperspherical Flows (S-FLM). <sup>:</sup>trained with uniform time sampler, <sup>;</sup>trained with adaptive time sampler. Values are reported as mean $\pm \mathrm { s t d }$ over 5 seeds. We bold the best result and underline the second best in each column.
<table><tr><td>Model</td><td>Linear</td><td>Cosine</td><td>Adaptive</td></tr><tr><td>Autoregressive</td><td></td><td></td><td></td></tr><tr><td>AR (Sample)</td><td> $2 . 9 _ { + 0 . 6 }$ </td><td></td><td></td></tr><tr><td>AR (Greedy)</td><td> $3 . 1 _ { + 0 . 5 }$ </td><td></td><td></td></tr><tr><td>Discrete (Absorbing)</td><td></td><td></td><td></td></tr><tr><td>Ancestral† (CE)</td><td> $6 0 . 1 _ { + 3 . 8 }$ </td><td> $6 1 . 1 \substack { + 4 . 5 }$ </td><td></td></tr><tr><td> $\mathrm { \ A n c e s t r a l ^ { \ddag } \left( C E \right) }$ </td><td> $6 9 . 4 _ { + 4 . 8 }$ </td><td> $6 9 . 7 _ { + 4 . 0 }$ </td><td> $7 0 . 0 _ { + 4 . 4 }$ </td></tr><tr><td> $\mathrm { A n c e s t r a l ^ { \dag } \ ( E L B O ) }$ </td><td> $5 6 . 0 _ { + 5 . 4 }$ </td><td> $5 5 . 8 _ { + 5 . 8 }$ </td><td></td></tr><tr><td> $\mathrm { \ A n c e s t r a l ^ { \ddag } \ ( E L B O ) }$ </td><td> $7 1 . 5 \substack { + 2 . 2 }$ </td><td> $7 1 . 7 \substack { + 1 . 6 }$ </td><td> $7 1 . 8 _ { + 2 . 4 }$ </td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r } ^ { \dag }$ </td><td> $8 5 . 3 _ { + 1 . 2 }$ </td><td> $8 4 . 6 _ { + 1 . 0 }$ </td><td></td></tr><tr><td> $\mathrm { P r e d i c t o r  – C o r r e c t o r ^ { \ddag } }$ </td><td> $3 3 . 8 _ { + 4 1 . 4 }$ </td><td> $3 5 . 6 _ { + 4 0 . 7 }$ </td><td> $6 9 . 6 _ { + 3 1 . 6 }$ </td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C } ^ { \dag } \left( \mathrm { C E } \right)$ </td><td> $9 1 . 2 _ { + 0 . 7 }$ </td><td> $9 8 . 3 _ { + 0 . 5 }$ </td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C ^ { \ddagger } } \left( \mathrm { C E } \right)$ </td><td> $9 4 . 7 \substack { + 0 . 8 }$ </td><td> $\mathbf { 9 9 . 3 _ { \div 0 . 2 } }$ </td><td> $9 1 . 6 _ { + 1 . 0 }$ </td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C } ^ { \dagger } \mathrm { ( E L B O ) }$ </td><td> $9 0 . 5 _ { + 0 . 8 }$ </td><td> $9 8 . 1 _ { + 0 . 8 }$ </td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C ^ { \ddagger } } \left( \mathrm { E L B O } \right)$ </td><td> $9 4 . 3 _ { + 0 . 7 }$ </td><td> $9 9 . 2 _ { + 0 . 2 }$ </td><td> $9 0 . 9 _ { + 1 . 0 }$ </td></tr><tr><td>Discrete (Uniform)</td><td></td><td></td><td></td></tr><tr><td>Ancestral† (ELBO)</td><td> $7 3 . 7 _ { + 3 . 4 }$ </td><td> $7 3 . 6 _ { + 3 . 2 }$ </td><td></td></tr><tr><td> $\mathrm { \ A n c e s t r a l ^ { \ddag } \ ( E L B O ) }$ </td><td> $8 2 . 1 _ { + 1 . 7 }$ </td><td> $8 2 . 4 \substack { + 1 . 7 }$ </td><td> $8 1 . 5 \substack { + 1 . 6 }$ </td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r } ^ { \dag }$ </td><td> $\underline { { 9 5 . 8 } } _ { + 1 . 0 }$ </td><td> $9 5 . 2 _ { + 1 . 3 }$ </td><td></td></tr><tr><td> $\mathrm { P r e d i c t o r  – C o r r e c t o r ^ { \ddag } }$ </td><td> $9 2 . 9 _ { + 0 . 9 }$ </td><td> $9 2 . 3 _ { + 0 . 9 }$ </td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C } ^ { \dagger } \mathrm { ( E L B O ) }$ </td><td> $\mathbf { 9 7 . 8 _ { \div 1 . 7 } }$ </td><td> $9 8 . 1 _ { + 1 . 2 }$ </td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C ^ { \ddagger } } \left( \mathrm { E L B O } \right)$ </td><td> $9 4 . 9 _ { + 5 . 1 }$ </td><td> $9 5 . 5 _ { + 4 . 8 }$ </td><td> $\mathbf { 9 3 . 9 . } _ { + 5 . 5 }$ </td></tr><tr><td>Euclidean / Spherical Flows</td><td></td><td></td><td></td></tr><tr><td> $\mathbf { F L M } ^ { \dagger } \mathbf { \Psi } ( \mathbf { U n i f o r m } )$ </td><td> $7 3 . 4 \substack { + 5 . 2 }$ </td><td> $7 3 . 0 _ { + 5 . 1 }$ </td><td></td></tr><tr><td>FLM (Gauss-Hermite LUT)</td><td> $6 7 . 7 _ { + 3 . 4 }$ </td><td> $6 7 . 4 \substack { + 3 . 2 }$ </td><td></td></tr><tr><td> $\mathrm { F L M ^ { \ddag } \left( A d a p t i v e \right) }$ </td><td> $7 5 . 3 _ { + 3 . 4 }$ </td><td> $7 4 . 9 _ { + 3 . 5 }$ </td><td> $6 9 . 2 _ { + 4 . 2 }$ </td></tr><tr><td> $\mathbb { S } – \mathrm { F L M } ^ { \dagger }$ </td><td> $8 3 . 5 _ { + 2 . 2 }$ </td><td> $8 4 . 0 _ { + 1 . 8 }$ </td><td></td></tr><tr><td> $\mathbb { S } – \mathrm { F L M } ^ { \ddagger }$ </td><td> $8 7 . 3 _ { + 1 . 5 }$ </td><td> $8 7 . 3 _ { + 1 . 7 }$ </td><td> $8 4 . 5 _ { + 1 . 9 }$ </td></tr></table>

## M.2 TINYGSM

Table 25: Baseline Models Accuracy (%) on TinyGSM $( T = 1 . 0 , 5 1 2$ steps) across sampling schedules. We evaluate Autoregressive (AR), Discrete Absorbing (CE and ELBO losses), Discrete Uniform (ELBO loss), Predictor-Corrector, Self-Conditioning (Ancestral + SC), FLM, and S-FLM. <sup>:</sup>trained with the uniform time sampler, <sup>;</sup>trained with the adaptive time sampler. We bold the best result in each column and underline the best result within each section (both if achieving both).
<table><tr><td>Model</td><td>Linear</td><td>Cosine</td><td>Adaptive</td></tr><tr><td>Autoregressive</td><td></td><td></td><td></td></tr><tr><td>AR (Sample)</td><td>52.6</td><td>一</td><td></td></tr><tr><td>AR (Greedy)</td><td>62.6</td><td>一</td><td></td></tr><tr><td>Discrete (Absorbing)</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { \ A n c e s t r a l ^ { \dag } \left( C E \right) }$ </td><td>12.9</td><td>14.8</td><td></td></tr><tr><td> $\mathrm { \ A n c e s t r a l ^ { \ddag } \left( C E \right) }$ </td><td>13.2</td><td>13.4</td><td>13.1</td></tr><tr><td> $\mathbf { A n c e s t r a l } ^ { \dagger } \mathbf { \Phi } ( \mathbf { E L B O } )$ </td><td>15.8</td><td>14.2</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l ^ { \ddagger } \left( E L B O \right) }$ </td><td>14.0</td><td>13.9</td><td>13.9</td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r } ^ { \dag }$ </td><td>13.3</td><td>12.1</td><td></td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r ^ { \ddag } }$ </td><td>10.8</td><td>11.4</td><td>10.9</td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C } ^ { \dag } \left( \mathrm { C E } \right)$ </td><td>23.0</td><td>26.4</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l + S C ^ { \ddagger } \left( C E \right) }$ </td><td>22.1</td><td>23.2</td><td>21.9</td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C } ^ { \dagger } \mathrm { ( E L B O ) }$ </td><td>20.5</td><td>22.2</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C ^ { \ddagger } \left( E L B O \right) }$ </td><td>23.5</td><td>22.2</td><td>21.1</td></tr><tr><td>Discrete (Uniform)</td><td></td><td></td><td></td></tr><tr><td> $\mathbf { A n c e s t r a l } ^ { \dagger } \mathbf { \Phi } ( \mathbf { E L B O } )$ </td><td>17.7</td><td>17.0</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l ^ { \ddagger } \left( E L B O \right) }$ </td><td>16.7</td><td>16.2</td><td>15.6</td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r } ^ { \dag }$ </td><td>36.8</td><td>36.5</td><td></td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r ^ { \ddag } }$ </td><td>32.4</td><td>30.9</td><td>26.6</td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C ^ { \dag } \left( E L B O \right) }$ </td><td>15.6</td><td>15.2</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C ^ { \ddagger } \left( E L B O \right) }$ </td><td>19.2</td><td>19.0</td><td>18.6</td></tr><tr><td> $E u c l i d e a n / S p h e r i c a l F l o w s$ </td><td></td><td></td><td></td></tr><tr><td>FLM (Gauss-Hermite LUT)</td><td>0.6</td><td>0.5</td><td></td></tr><tr><td>FLM†</td><td>3.0</td><td>2.9</td><td>一</td></tr><tr><td> $\mathrm { F L M ^ { \ddag } }$ </td><td>4.0</td><td>3.9</td><td>3.6</td></tr><tr><td> $\mathbb { S } – \mathrm { F L M } ^ { \ddagger }$ </td><td>10.7</td><td>11.5</td><td>10.3</td></tr><tr><td> $\mathbb { S } – \mathrm { F L M } ^ { \ddagger }$  (argmax velocity)</td><td>15.0</td><td>16.5</td><td>13.2</td></tr></table>

Table 26: Baseline Models Accuracy (%) on TinyGSM $( T ~ = ~ 1 . 0 ,$ 64 steps) across sampling schedules. We evaluate Autoregressive (AR), Discrete Absorbing (CE and ELBO losses), Discrete Uniform (ELBO loss), Predictor-Corrector, Self-Conditioning (Ancestral + SC), FLM, and S-FLM. <sup>:</sup>trained with the uniform time sampler, <sup>;</sup>trained with the adaptive time sampler. We bold the best result in each column and underline the best result within each section (both if achieving both).
<table><tr><td>Model</td><td>Linear</td><td>Cosine</td><td>Adaptive</td></tr><tr><td>Discrete (Absorbing)</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { \ A n c e s t r a l ^ { \dag } \left( C E \right) }$ </td><td>9.7</td><td>11.1</td><td></td></tr><tr><td> $\mathrm { \ A n c e s t r a l ^ { \ddag } \left( C E \right) }$ </td><td>8.3</td><td>11.7</td><td>11.2</td></tr><tr><td> $\mathbf { A n c e s t r a l } ^ { \dagger } \mathbf { \Phi } ( \mathbf { E L B O } )$ </td><td>10.0</td><td>13.3</td><td></td></tr><tr><td> $\mathrm { \ A n c e s t r a l ^ { \ddag } \ ( E L B O ) }$ </td><td>8.5</td><td>11.8</td><td>12.6</td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r } ^ { \dag }$ </td><td>10.7</td><td>10.2</td><td></td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r ^ { \ddag } }$ </td><td>9.8</td><td>11.4</td><td>7.2</td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C } ^ { \dag } \left( \mathrm { C E } \right)$ </td><td>12.2</td><td>19.9</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l + S C ^ { \ddagger } \left( C E \right) }$ </td><td>12.6</td><td>19.6</td><td>18.0</td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C } ^ { \dagger } \mathrm { ( E L B O ) }$ </td><td>12.6</td><td>17.2</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C ^ { \ddagger } \left( E L B O \right) }$ </td><td>12.9</td><td>19.3</td><td>18.7</td></tr><tr><td>Discrete (Uniform)</td><td></td><td></td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l } ^ { \dagger } \ ( \mathrm { E L B O } )$ </td><td>14.4</td><td>14.5</td><td></td></tr><tr><td> $\mathrm { \ A n c e s t r a l ^ { \ddag } \ ( E L B O ) }$ </td><td>14.9</td><td>15.6</td><td>14.6</td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r } ^ { \dag }$ </td><td>23.3</td><td>23.7</td><td></td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r ^ { \ddag } }$ </td><td>20.5</td><td>20.1</td><td>16.6</td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C } ^ { \dagger } \mathrm { ( E L B O ) }$ </td><td>12.0</td><td>14.7</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C ^ { \ddagger } \left( E L B O \right) }$ </td><td>17.0</td><td>18.2</td><td>16.7</td></tr><tr><td>Euclidean / Spherical Flows</td><td></td><td></td><td></td></tr><tr><td>FLM (Gauss-Hermite LUT)</td><td>0.5</td><td>0.6</td><td>一</td></tr><tr><td> $\mathrm { F L M } ^ { \dagger }$ </td><td>2.5</td><td>2.5</td><td>一</td></tr><tr><td> $\mathrm { F L M ^ { \ddag } }$ </td><td>3.3</td><td>3.5</td><td>4.0</td></tr><tr><td> $\mathbb { S } – \mathrm { F L M } ^ { \ddagger }$ </td><td>4.5</td><td>9.7</td><td>9.5</td></tr><tr><td>S-FLM‡ (argmax velocity)</td><td>9.1</td><td>14.2</td><td>12.9</td></tr></table>

Table 27: Baseline Models Accuracy (%) on TinyGSM (T “ 0.1, 512 steps) across sampling schedules. We evaluate Autoregressive (AR), Discrete Absorbing (CE and ELBO losses), Discrete Uniform (ELBO loss), Predictor-Corrector, Self-Conditioning (Ancestral + SC), FLM, and S-FLM. <sup>:</sup>trained with the uniform time sampler, <sup>;</sup>trained with the adaptive time sampler. We bold the best result in each column and underline the best result within each section (both if achieving both). With argmax velocity, the S-FLM sampler is deterministic given the initial noise and does not depend on the temperature, hence identical rows at $T = 1$ and $T = 0 . 1$
<table><tr><td>Model</td><td>Linear</td><td>Cosine</td><td>Adaptive</td></tr><tr><td>Autoregressive</td><td></td><td></td><td></td></tr><tr><td>AR (Sample)</td><td>52.6</td><td></td><td></td></tr><tr><td>AR (Greedy)</td><td>62.6</td><td></td><td></td></tr><tr><td>Discrete (Absorbing)</td><td></td><td></td><td></td></tr><tr><td>Ancestral† (CE)</td><td>31.0</td><td>31.1</td><td></td></tr><tr><td>Ancestral (CE)</td><td>30.4</td><td>30.0</td><td>30.3</td></tr><tr><td>Ancestral† (ELBO)</td><td>32.0</td><td>32.0</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l ^ { \ddagger } \left( E L B O \right) }$ </td><td>32.3</td><td>33.5</td><td>32.0</td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r } ^ { \dag }$ </td><td>39.6</td><td>37.7</td><td></td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r ^ { \ddag } }$ </td><td>39.8</td><td>39.2</td><td>38.9</td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C } ^ { \dag } \left( \mathrm { C E } \right)$ </td><td>44.0</td><td>42.7</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l + S C ^ { \ddagger } \left( C E \right) }$ </td><td>45.8</td><td>45.5</td><td>42.7</td></tr><tr><td>Ancestral + SC† (ELBO)</td><td>39.0</td><td>38.0</td><td></td></tr><tr><td>Ancestral + SC‡ (ELBO)</td><td>43.2</td><td>43.5</td><td>42.6</td></tr><tr><td>Discrete (Uniform)</td><td></td><td></td><td></td></tr><tr><td>Ancestral† (ELBO)</td><td>33.0</td><td>32.6</td><td></td></tr><tr><td> $\mathrm { \ A n c e s t r a l ^ { \ddag } \ ( E L B O ) }$ </td><td>34.0</td><td>31.2</td><td>34.6</td></tr><tr><td>Predictor-Corrector†</td><td>45.5</td><td>45.5</td><td></td></tr><tr><td>Predictor-Corrector</td><td>40.1</td><td>39.8</td><td>40.1</td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C ^ { \dag } \left( E L B O \right) }$ </td><td>31.7</td><td>30.3</td><td></td></tr><tr><td>Ancestral + SC‡ (ELBO)</td><td>39.7</td><td>38.3</td><td>38.8</td></tr><tr><td>Euclidean / Spherical Flows</td><td></td><td></td><td></td></tr><tr><td>FLM (Gauss-Hermite LUT)</td><td>0.8</td><td>0.8</td><td></td></tr><tr><td> $\mathrm { F L M } ^ { \dagger }$ </td><td>6.4</td><td>6.4</td><td></td></tr><tr><td> $\mathrm { F L M ^ { \ddag } }$ </td><td>8.3</td><td>8.3</td><td>8.3</td></tr><tr><td> $\mathbb { S } – \mathrm { F L M } ^ { \ddagger }$ </td><td>15.4</td><td>15.4</td><td>14.7</td></tr><tr><td>S-FLM (argmax velocity)</td><td>15.0</td><td>16.5</td><td>13.2</td></tr></table>

Table 28: Baseline Models Accuracy (%) on TinyGSM $( T ~ = ~ 0 . 1$ , 64 steps) across sampling schedules. We evaluate Autoregressive (AR), Discrete Absorbing (CE and ELBO losses), Discrete Uniform (ELBO loss), Predictor-Corrector, Self-Conditioning (Ancestral + SC), FLM, and S-FLM. <sup>:</sup>trained with the uniform time sampler, <sup>;</sup>trained with the adaptive time sampler. We bold the best result in each column and underline the best result within each section (both if achieving both). With argmax velocity, the S-FLM sampler is deterministic given the initial noise and does not depend on the temperature, hence identical rows at $T = 1$ and $T = 0 . 1$
<table><tr><td>Model</td><td>Linear</td><td>Cosine</td><td>Adaptive</td></tr><tr><td>Discrete (Absorbing)</td><td></td><td></td><td></td></tr><tr><td>Ancestral† (CE)</td><td>25.0</td><td>28.2</td><td></td></tr><tr><td> $\mathrm { \ A n c e s t r a l ^ { \ddag } \left( C E \right) }$ </td><td>24.9</td><td>29.3</td><td>27.3</td></tr><tr><td> $\mathbf { A n c e s t r a l } ^ { \dagger } \mathbf { \Phi } ( \mathbf { E L B O } )$ </td><td>27.4</td><td>29.5</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l ^ { \ddagger } \left( E L B O \right) }$ </td><td>29.6</td><td>30.7</td><td>29.9</td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r } ^ { \dag }$ </td><td>31.1</td><td>31.8</td><td></td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r ^ { \ddag } }$ </td><td>33.0</td><td>35.2</td><td>32.9</td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C } ^ { \dag } \left( \mathrm { C E } \right)$ </td><td>31.5</td><td>38.7</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l + S C ^ { \ddagger } \left( C E \right) }$ </td><td>35.9</td><td>40.3</td><td>41.4</td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C } ^ { \dagger } \mathrm { ( E L B O ) }$ </td><td>29.8</td><td>34.5</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C ^ { \ddagger } \left( E L B O \right) }$ </td><td>35.6</td><td>39.8</td><td>41.1</td></tr><tr><td>Discrete (Uniform)</td><td></td><td></td><td></td></tr><tr><td>Ancestral† (ELBO)</td><td>30.7</td><td>31.5</td><td></td></tr><tr><td>Ancestral‡ (ELBO)</td><td>32.3</td><td>32.4</td><td>31.6</td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r } ^ { \dag }$ </td><td>38.3</td><td>37.0</td><td></td></tr><tr><td> $\mathrm { P r e d i c t o r - C o r r e c t o r ^ { \ddag } }$ </td><td>35.7</td><td>35.2</td><td>33.5</td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C ^ { \dag } \left( E L B O \right) }$ </td><td>27.8</td><td>30.7</td><td></td></tr><tr><td> $\mathrm { A n c e s t r a l } + \mathrm { S C ^ { \ddagger } \left( E L B O \right) }$ </td><td>34.7</td><td>37.5</td><td>37.1</td></tr><tr><td>Euclidean / Spherical Flows</td><td></td><td></td><td></td></tr><tr><td>FLM (Gauss-Hermite LUT)</td><td>0.7</td><td>0.8</td><td></td></tr><tr><td>FLM†</td><td>6.6</td><td>6.4</td><td></td></tr><tr><td>FLM</td><td>8.1</td><td>8.3</td><td>7.9</td></tr><tr><td>S-FLM</td><td>9.4</td><td>14.1</td><td>13.4</td></tr><tr><td>S-FLM‡ (argmax velocity)</td><td>9.1</td><td>14.2</td><td>12.9</td></tr></table>

Table 29: TinyGSM Simplex Diffusion (SDM) Accuracy (%) $\begin{array} { r l r } { ( T } & { { } = } & { 1 . 0 , } \end{array}$ 512 steps) across noise concentration schedules, sampling schedules (Linear, Cosine, Adaptive), and churn $( \kappa \in \{ 0 . 0 , 0 . 2 , 1 . 0 \} )$ . We compare Expectation $( P _ { t } )$ , Argmax, and Argmax with Self-Conditioning (+ SC). All SDMs are trained with the adaptive time sampler. We bold the best result in each column and underline the best result within each section (both if achieving both).
<table><tr><td rowspan="2">Concentration Schedule</td><td colspan="3">Linear</td><td colspan="3">Cosine</td><td colspan="3">Adaptive</td></tr><tr><td>κ = 0</td><td>κ = 0.2</td><td>κ = 1.0</td><td>κ = 0</td><td>κ = 0.2</td><td>κ = 1.0</td><td>κ = 0</td><td>κ = 0.2</td><td> $\kappa = 1 . 0$ </td></tr><tr><td>Expectation Input Processing (Pt)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td>15.4</td><td>25.1</td><td>34.7</td><td>19.8</td><td>29.1</td><td>38.8</td><td>12.6</td><td>35.7</td><td>45.8</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td>15.1</td><td>24.5</td><td>31.7</td><td>17.9</td><td>28.4</td><td>36.8</td><td>11.5</td><td>29.2</td><td>38.8</td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td>15.7</td><td>23.1</td><td>33.6</td><td>17.5</td><td>24.4</td><td>38.2</td><td>15.0</td><td>29.1</td><td>41.7</td></tr><tr><td>Constant (ν = 0.25)</td><td>12.1</td><td>19.4</td><td>26.6</td><td>14.2</td><td>25.8</td><td>32.5</td><td>10.2</td><td>33.9</td><td>43.3</td></tr><tr><td>Constant (ν = 0.1)</td><td>4.7</td><td>9.6</td><td>12.8</td><td>6.5</td><td>15.6</td><td>20.1</td><td>3.5</td><td>17.0</td><td>24.5</td></tr><tr><td>Argmax Input Processing</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td>15.5</td><td>26.7</td><td>35.7</td><td>17.4</td><td>26.9</td><td>33.3</td><td>14.9</td><td>21.9</td><td>30.4</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td>18.3</td><td>26.3</td><td>34.2</td><td>18.4</td><td>26.0</td><td>37.4</td><td>16.0</td><td>20.2</td><td>30.6</td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td>17.1</td><td>24.8</td><td>33.9</td><td>16.6</td><td>22.4</td><td>33.0</td><td>16.7</td><td>16.4</td><td>29.5</td></tr><tr><td>Constant (ν = 0.25)</td><td>19.3</td><td>29.3</td><td>31.8</td><td>17.8</td><td>27.3</td><td>35.0</td><td>17.8</td><td>23.0</td><td>30.3</td></tr><tr><td>Argmax + Self-Conditioning (SC)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td>17.1</td><td>30.2</td><td>38.1</td><td>16.9</td><td>31.0</td><td>40.1</td><td>17.0</td><td>26.6</td><td>36.9</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td>16.9</td><td>32.5</td><td>43.7</td><td>18.9</td><td>32.8</td><td>42.7</td><td>16.9</td><td>30.8</td><td>40.3</td></tr><tr><td> $\mathrm { C o n s t . - L i n . } ( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td>13.7</td><td>25.6</td><td>34.7</td><td>17.4</td><td>24.1</td><td>33.5</td><td>11.2</td><td>20.2</td><td>32.5</td></tr><tr><td>Constant (ν = 0.5)</td><td>21.1</td><td>29.3</td><td>40.8</td><td>19.5</td><td>27.6</td><td>41.4</td><td>20.5</td><td>25.6</td><td>40.6</td></tr><tr><td>Constant (ν = 0.25)</td><td>17.6</td><td>31.1</td><td>37.0</td><td>18.2</td><td>29.0</td><td>38.9</td><td>17.6</td><td>29.3</td><td>37.8</td></tr></table>

Table 30: TinyGSM Simplex Diffusion (SDM) Accuracy (%) $\begin{array} { r l r } { ( T } & { { } = } & { 1 . 0 , } \end{array}$ 64 steps) across noise concentration schedules, sampling schedules (Linear, Cosine, Adaptive), and churn $( \kappa \in \{ 0 . 0 , 0 . 2 , 1 . 0 \} )$ . We compare Expectation $( P _ { t } ) _ { \ l }$ , Argmax, and Argmax with Self-Conditioning (+ SC). All SDMs are trained with the adaptive time sampler. We bold the best result in each column and underline the best result within each section (both if achieving both).
<table><tr><td rowspan="2">Concentration Schedule</td><td colspan="3">Linear</td><td colspan="3">Cosine</td><td colspan="3">Adaptive</td></tr><tr><td>κ = 0</td><td>κ = 0.2</td><td>κ = 1.0</td><td>κ = 0</td><td>κ = 0.2</td><td>κ = 1.0</td><td>κ = 0</td><td>κ = 0.2</td><td>κ = 1.0</td></tr><tr><td>Expectation Input Processing (Pt)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td>6.7</td><td>9.8</td><td>14.7</td><td>14.3</td><td>16.8</td><td>22.8</td><td>10.6</td><td>23.1</td><td>32.7</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td>4.9</td><td>12.0</td><td>16.1</td><td>14.1</td><td>16.1</td><td>20.4</td><td>10.3</td><td>15.3</td><td>25.5</td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td>7.4</td><td>9.4</td><td>13.8</td><td>15.0</td><td>14.2</td><td>24.0</td><td>11.2</td><td>21.1</td><td>31.8</td></tr><tr><td>Constant (ν = 0.25)</td><td>4.2</td><td>5.9</td><td>8.6</td><td>11.4</td><td>14.6</td><td>18.9</td><td>8.0</td><td>18.2</td><td>30.1</td></tr><tr><td>Constant  $( \nu = 0 . 1 )$ </td><td>0.8</td><td>1.5</td><td>1.9</td><td>5.0</td><td>6.7</td><td>9.3</td><td>2.7</td><td>8.9</td><td>13.4</td></tr><tr><td>Argmax Input Processing</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td>13.1</td><td>13.1</td><td>20.9</td><td>16.2</td><td>14.0</td><td>21.4</td><td>14.4</td><td>10.4</td><td>17.3</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td>9.7</td><td>13.5</td><td>22.8</td><td>16.1</td><td>14.7</td><td>22.4</td><td>14.1</td><td>11.0</td><td>17.1</td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td>14.4</td><td>13.2</td><td>22.0</td><td>15.6</td><td>10.0</td><td>22.5</td><td>13.7</td><td>9.2</td><td>19.3</td></tr><tr><td>Constant  $( \nu = 0 . 2 5 )$ </td><td>12.6</td><td>13.3</td><td>21.7</td><td>16.8</td><td>13.5</td><td>21.6</td><td>15.3</td><td>11.9</td><td>17.4</td></tr><tr><td>Argmax + Self-Conditioning (SC)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td>11.9</td><td>16.4</td><td>24.1</td><td>17.6</td><td>15.8</td><td>25.0</td><td>13.2</td><td>14.6</td><td>22.4</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td>10.0</td><td>16.4</td><td>25.7</td><td>15.1</td><td>17.5</td><td>28.1</td><td>14.6</td><td>17.0</td><td>26.4</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td>6.2</td><td>15.1</td><td>22.5</td><td>13.4</td><td>12.9</td><td>23.1</td><td>9.4</td><td>11.6</td><td>18.2</td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td>13.6</td><td>16.3</td><td>25.5</td><td>18.1</td><td>13.6</td><td>29.9</td><td>15.8</td><td>17.0</td><td>25.6</td></tr><tr><td>Constant  $( \nu = 0 . 2 5 )$ </td><td>13.5</td><td>15.5</td><td>22.0</td><td>14.8</td><td>16.4</td><td>24.7</td><td>13.5</td><td>16.0</td><td>22.1</td></tr></table>

Table 31: TinyGSM Simplex Diffusion (SDM) Accuracy (%) $( T \ = \ 0 . 1 , \ 5 1 2$ steps) across noise concentration schedules, sampling schedules (Linear, Cosine, Adaptive), and churn $( \kappa \in \{ 0 . 0 , 0 . 2 , 1 . 0 \} )$ . We compare Expectation $( P _ { t } )$ , Argmax, and Argmax with Self-Conditioning (+ SC). All SDMs are trained with the adaptive time sampler. We bold the best result in each column and underline the best result within each section (both if achieving both).
<table><tr><td rowspan="2">Concentration Schedule</td><td colspan="3">Linear</td><td colspan="3">Cosine</td><td colspan="3">Adaptive</td></tr><tr><td>κ = 0</td><td>κ = 0.2</td><td>κ = 1.0</td><td>κ = 0</td><td>κ = 0.2</td><td>κ = 1.0</td><td>κ = 0</td><td>κ = 0.2</td><td>κ = 1.0</td></tr><tr><td>Expectation Input  $P r o c e s s i n g \left( P _ { t } \right)$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td>27.3</td><td>36.0</td><td>40.4</td><td>30.8</td><td>38.5</td><td>42.4</td><td>19.3</td><td>43.9</td><td>49.0</td></tr><tr><td>Const.-Lin  $. ( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td>27.7</td><td>36.4</td><td>38.2</td><td>30.8</td><td>39.3</td><td>42.5</td><td>19.8</td><td>40.8</td><td>44.3</td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td>27.0</td><td>30.5</td><td>41.7</td><td>29.5</td><td>33.3</td><td>43.3</td><td>22.2</td><td>35.0</td><td>44.8</td></tr><tr><td>Constant  $( \nu = 0 . 2 5 )$ </td><td>18.3</td><td>25.0</td><td>30.1</td><td>22.5</td><td>31.2</td><td>38.5</td><td>11.3</td><td>38.3</td><td>42.5</td></tr><tr><td>Constant  $( \nu = 0 . 1 )$ </td><td>7.7</td><td>10.2</td><td>13.8</td><td>9.3</td><td>17.0</td><td>22.1</td><td>2.9</td><td>17.5</td><td>21.0</td></tr><tr><td>Argmax Input Processing</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td>33.0</td><td>40.8</td><td>43.8</td><td>34.7</td><td>39.8</td><td>45.5</td><td>31.5</td><td>39.7</td><td>42.9</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td>31.6</td><td>42.2</td><td>46.4</td><td>33.6</td><td>40.8</td><td>45.5</td><td>30.7</td><td>38.8</td><td>45.4</td></tr><tr><td>Constant (ν = 0.5)</td><td>33.6</td><td>37.8</td><td>44.5</td><td>34.4</td><td>33.6</td><td>46.1</td><td>34.5</td><td>32.4</td><td>43.4</td></tr><tr><td>Constant (ν = 0.25)</td><td>34.2</td><td>42.7</td><td>44.9</td><td>33.2</td><td>41.1</td><td>45.7</td><td>33.4</td><td>38.2</td><td>43.7</td></tr><tr><td>Argmax + Self-Conditioning (SC)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td>33.3</td><td>45.5</td><td>50.0</td><td>33.9</td><td>43.7</td><td>48.9</td><td>31.4</td><td>43.8</td><td>50.2</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td>31.6</td><td>46.1</td><td>49.6</td><td>33.4</td><td>45.1</td><td>50.5</td><td>31.0</td><td>46.2</td><td>50.3</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td>27.5</td><td>38.8</td><td>46.1</td><td>31.8</td><td>36.6</td><td>46.4</td><td>19.9</td><td>35.9</td><td>43.2</td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td>34.0</td><td>42.2</td><td>50.6</td><td>34.7</td><td>40.8</td><td>50.9</td><td>33.1</td><td>39.2</td><td>50.6</td></tr><tr><td>Constant (ν = 0.25)</td><td>31.7</td><td>41.8</td><td>48.7</td><td>32.6</td><td>43.6</td><td>49.4</td><td>33.7</td><td>43.4</td><td>50.6</td></tr></table>

Table 32: TinyGSM Simplex Diffusion (SDM) Accuracy (%) $\begin{array} { r l r } { ( T } & { { } = } & { 0 . 1 } \end{array}$ 64 steps) across noise concentration schedules, sampling schedules (Linear, Cosine, Adaptive), and churn $( \kappa \in \{ 0 . 0 , 0 . 2 , 1 . 0 \} )$ . We compare Expectation $( P _ { t } )$ , Argmax, and Argmax with Self-Conditioning $( + \thinspace \thinspace \mathrm { { S C } ) }$ . All SDMs are trained with the adaptive time sampler. We bold the best result in each column and underline the best result within each section (both if achieving both).
<table><tr><td rowspan="2">Concentration Schedule</td><td colspan="3">Linear</td><td colspan="3">Cosine</td><td colspan="3">Adaptive</td></tr><tr><td> $\kappa = 0$ </td><td>κ = 0.2</td><td> $\kappa = 1 . 0$ </td><td>κ = 0</td><td> $\kappa = 0 . 2$ </td><td> $\kappa = 1 . 0$ </td><td>κ = 0</td><td> $\kappa = 0 . 2$ </td><td> $\kappa = 1 . 0$ </td></tr><tr><td>Expectation Input Processing (Pt)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td>11.3</td><td>16.7</td><td>20.5</td><td>25.7</td><td>28.7</td><td>32.0</td><td>18.9</td><td>33.0</td><td>39.5</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td>11.7</td><td>22.5</td><td>28.8</td><td>25.8</td><td>29.2</td><td>33.6</td><td>19.0</td><td>32.1</td><td>38.8</td></tr><tr><td>Constant (ν = 0.5)</td><td>15.3</td><td>16.2</td><td>20.0</td><td>26.8</td><td>21.1</td><td>33.3</td><td>21.8</td><td>31.9</td><td>40.1</td></tr><tr><td>Constant  $( \nu = 0 . 2 5 )$ </td><td>6.2</td><td>7.1</td><td>9.5</td><td>18.0</td><td>19.9</td><td>25.1</td><td>12.1</td><td>24.0</td><td>31.0</td></tr><tr><td>Constant  $( \nu = 0 . 1 )$ </td><td>0.8</td><td>0.6</td><td>1.5</td><td>7.7</td><td>9.4</td><td>12.4</td><td>2.6</td><td>7.9</td><td>12.4</td></tr><tr><td>Argmax Input Processing</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td>27.2</td><td>27.9</td><td>38.8</td><td>30.7</td><td>29.6</td><td>39.1</td><td>31.1</td><td>27.9</td><td>37.4</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td>23.4</td><td>30.1</td><td>38.5</td><td>32.4</td><td>30.9</td><td>39.3</td><td>27.9</td><td>29.2</td><td>37.4</td></tr><tr><td>Constant  $( \nu = 0 . 5 )$ </td><td>30.7</td><td>28.6</td><td>37.4</td><td>32.1</td><td>22.1</td><td>36.7</td><td>32.4</td><td>25.7</td><td>37.8</td></tr><tr><td>Constant  $( \nu = 0 . 2 5 )$ </td><td>29.7</td><td>30.8</td><td>35.4</td><td>31.9</td><td>28.6</td><td>37.8</td><td>32.1</td><td>27.5</td><td>37.4</td></tr><tr><td>Argmax + Self-Conditioning (SC)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ </td><td>26.6</td><td>30.9</td><td>40.6</td><td>31.2</td><td>33.9</td><td>42.0</td><td>31.1</td><td>33.6</td><td>42.0</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 2 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 2 )$ </td><td>20.8</td><td>31.4</td><td>42.0</td><td>30.4</td><td>34.3</td><td>43.4</td><td>28.5</td><td>36.1</td><td>46.0</td></tr><tr><td>Const.-Lin.  $( \nu _ { 0 } = 0 . 4 , \nu _ { 1 } = 0 . 7 5 , \ell = 0 . 8 )$ </td><td>13.3</td><td>29.2</td><td>36.9</td><td>26.7</td><td>25.4</td><td>38.8</td><td>18.6</td><td>28.7</td><td>37.9</td></tr><tr><td>Constant (ν = 0.5)</td><td>29.6</td><td>33.0</td><td>41.0</td><td>32.7</td><td>27.3</td><td>44.2</td><td>32.2</td><td>36.6</td><td>44.5</td></tr><tr><td>Constant (ν = 0.25)</td><td>27.5</td><td>29.9</td><td>39.4</td><td>30.1</td><td>32.5</td><td>40.8</td><td>30.7</td><td>33.0</td><td>42.5</td></tr></table>

Table 33: Multi-Sample Accuracy and AST Structural Diversity (K “ 5 samples per problem) on TinyGSM across sampling steps NFE P t8, 64, 512u and temperatures $T \in \{ 0 . 1 , \bar { 1 } . 0 \}$ . <sup>:</sup>trained with uniform time sampler (evaluated with linear sampling schedule), <sup>;</sup>trained with adaptive time sampler (evaluated with adaptive sampling schedule). $\bar { \cdot } \mathrm { E L B O ^ { , } }$ denotes the evidence lower bound objective, $^ { \circ \circ } \mathrm { S C } ^ { \prime \prime }$ denotes Self-Conditioning, and $\mathbf { \ddot { \mu } } ^ { 6 6 } \mathbf { P } \mathbf { C } ^ { \mathbf { \ " } }$ denotes the Predictor-Corrector sampler. Arrows (Ò / Ó) indicate whether higher or lower values are better. For each column, we bold the overall best result and underline the best result within each section. For 64 and 512 steps, pass@1 is the single-sample accuracy from Tables 25 to 28 (baselines), Tables 29 to 32 (SDMs, ν<sub>0</sub> “ 0.2, $\nu _ { 1 } = 0 . 5 , \ell = 0 . 2 )$ and Figure 13 (distilled SDM); for 8 steps it is the accuracy of the first of the K samples. For AR, this is greedy decoding in the $T = 0 . 1$ row and sampling in the T “ 1.0 row.
<table><tr><td>Model</td><td>Temp. (T)</td><td>Steps (NFE)</td><td>pass@1 (%) ↑</td><td>pass@2 (%) ↑</td><td>pass@5 (%) ↑</td><td>AST Div. ↑</td><td>AST Div. (Correct) ↑</td></tr><tr><td colspan="8">Autoregressive</td></tr><tr><td></td><td>0.1</td><td>512</td><td>62.6</td><td>66.3</td><td>69.6</td><td>9.6</td><td>6.2</td></tr><tr><td></td><td>1.0</td><td>512</td><td>52.6</td><td>66.2</td><td>77.6</td><td>36.7</td><td>29.5</td></tr><tr><td colspan="8">Absorbing Diffusion (MDM†; ELBO; Linear Schedule)</td></tr><tr><td></td><td>0.1</td><td>8</td><td>4.5</td><td>7.7</td><td>13.3</td><td>40.3</td><td>8.0</td></tr><tr><td></td><td>0.1</td><td>64</td><td>27.4</td><td>37.7</td><td>52.3</td><td>33.4</td><td>19.2</td></tr><tr><td></td><td>0.1</td><td>512</td><td>32.0</td><td>43.2</td><td>56.9</td><td>31.6</td><td>19.3</td></tr><tr><td></td><td>1.0</td><td>8</td><td>0.5</td><td>0.8</td><td>1.4</td><td>51.1</td><td>14.3</td></tr><tr><td></td><td>1.0</td><td>64</td><td>10.0</td><td>15.6</td><td>28.6</td><td>47.7</td><td>22.6</td></tr><tr><td>1.0</td><td></td><td>512</td><td>15.8</td><td>24.1</td><td>38.6</td><td>44.8</td><td>23.4</td></tr><tr><td colspan="8">Absorbing Diffusion (MDM‡; ELBO; SC; Adaptive Schedule)</td></tr><tr><td></td><td>0.1</td><td>8</td><td>19.6</td><td>29.7</td><td>43.8</td><td>30.4</td><td>14.6</td></tr><tr><td></td><td>0.1</td><td>64</td><td>41.1</td><td>51.6</td><td>62.0</td><td>29.1</td><td>19.1</td></tr><tr><td></td><td>0.1</td><td>512</td><td>42.6</td><td>54.9</td><td>65.0</td><td>28.3</td><td>18.9</td></tr><tr><td></td><td>1.0</td><td>8</td><td>4.5</td><td>8.3</td><td>17.7</td><td>38.9</td><td>14.6</td></tr><tr><td></td><td>1.0</td><td>64</td><td>18.7</td><td>28.9</td><td>43.6</td><td>41.7</td><td>21.7</td></tr><tr><td></td><td>1.0</td><td>512</td><td>21.1</td><td>36.2</td><td>49.7</td><td>41.3</td><td>24.1</td></tr><tr><td colspan="8">Uniform Diffusion (UDM†; ELBO; Linear Schedule)</td></tr><tr><td></td><td>0.1</td><td>8</td><td>11.2</td><td>17.3</td><td>28.7</td><td>35.9</td><td>15.4</td></tr><tr><td></td><td>0.1</td><td>64</td><td>30.7</td><td>42.5</td><td>56.7</td><td>32.0</td><td>19.6</td></tr><tr><td></td><td>0.1</td><td>512</td><td>33.0</td><td>46.1</td><td>59.6</td><td>31.9</td><td>19.7</td></tr><tr><td></td><td>1.0</td><td>8</td><td>3.2</td><td>5.3</td><td>10.3</td><td>46.5</td><td>15.4</td></tr><tr><td></td><td>1.0</td><td>64</td><td>14.4</td><td>22.3</td><td>36.1</td><td>44.1</td><td>23.6</td></tr><tr><td></td><td>1.0</td><td>512</td><td>17.7</td><td>25.6</td><td>40.5</td><td>42.9</td><td>23.1</td></tr><tr><td colspan="8">Uniform Diffusion (UDM†; PC; Linear Schedule)</td></tr><tr><td></td><td>0.1</td><td>8</td><td>11.5</td><td>18.1</td><td>31.0</td><td>39.0</td><td>17.9</td></tr><tr><td></td><td>0.1</td><td>64</td><td>38.3</td><td>49.0</td><td>61.5</td><td>33.4</td><td>22.2</td></tr><tr><td></td><td>0.1</td><td>512</td><td>45.5</td><td>56.8</td><td>67.1</td><td>30.6</td><td>21.3</td></tr><tr><td></td><td>1.0</td><td>8</td><td>4.4</td><td>7.6</td><td>13.8</td><td>49.9</td><td>19.0</td></tr><tr><td></td><td>1.0</td><td>64</td><td>23.3</td><td>34.9</td><td>51.1</td><td>44.3</td><td>26.8</td></tr><tr><td>1.0</td><td></td><td>512</td><td>36.8</td><td>49.3</td><td>62.2</td><td>40.6</td><td>27.6</td></tr><tr><td colspan="8">Simplex Diffusion (SDM‡; Expectation; κ = 1; Adaptive Schedule)</td></tr><tr><td colspan="8"></td></tr><tr><td></td><td>0.1</td><td>8</td><td>17.1</td><td>27.7</td><td>42.8</td><td>36.5</td><td>21.2</td></tr><tr><td></td><td>0.1</td><td>64</td><td>39.5</td><td>52.4</td><td>64.8</td><td>32.3</td><td>22.5</td></tr><tr><td></td><td>0.1</td><td>512</td><td>49.0</td><td>58.0</td><td>68.2</td><td>30.4</td><td>21.6</td></tr><tr><td></td><td>1.0</td><td>8</td><td>5.4</td><td>9.7</td><td>17.9</td><td>40.0</td><td>17.3</td></tr><tr><td></td><td>1.0 1.0</td><td>64</td><td>32.7</td><td>44.0</td><td>57.7</td><td>38.7</td><td>25.9</td></tr><tr><td></td><td></td><td>512</td><td>45.8</td><td>55.1</td><td>66.6</td><td>35.3</td><td>24.8</td></tr><tr><td colspan="8">Simplex Diffusion (SDM‡; Argmax + SC; κ = 1; Adaptive Schedule)</td></tr><tr><td colspan="8"></td></tr><tr><td></td><td>0.1</td><td>8</td><td>19.9</td><td>29.3</td><td>41.4</td><td>29.2 26.8</td><td>13.9 17.5</td></tr><tr><td></td><td>0.1</td><td>64</td><td>42.0</td><td>55.7</td><td>64.5 67.5</td><td>23.2</td><td>16.2</td></tr><tr><td></td><td>0.1</td><td>512</td><td>50.2 3.9</td><td>60.0 6.2</td><td>12.2</td><td>25.0</td><td>7.8</td></tr><tr><td></td><td>1.0</td><td>8 64</td><td>22.4</td><td>32.6</td><td>48.0</td><td>38.3</td><td>19.6</td></tr><tr><td></td><td>1.0 1.0</td><td>512</td><td>36.9</td><td>50.1</td><td>62.6</td><td>35.3</td><td>22.3</td></tr><tr><td colspan="8"></td></tr><tr><td colspan="8">Distilled Simplex Diffusion (SDM‡; Expectation; κ = 1; Adaptive Schedule)</td></tr><tr><td></td><td>0.1</td><td>8</td><td>31.9</td><td>39.2</td><td>48.1</td><td>17.4</td><td>7.9</td></tr><tr><td></td><td>0.1</td><td>64</td><td>37.5</td><td>45.7</td><td>53.1</td><td>18.8</td><td>9.9</td></tr><tr><td></td><td>0.1</td><td>512</td><td>40.6</td><td>46.3</td><td>53.7</td><td>18.0</td><td>10.8</td></tr><tr><td></td><td>1.0</td><td>8</td><td>31.9</td><td>40.0</td><td>48.7</td><td>17.5</td><td>7.5</td></tr><tr><td></td><td>1.0</td><td>64 512</td><td>37.8 39.6</td><td>45.8 46.5</td><td>53.6 53.9</td><td>19.0 18.1</td><td>10.7 10.4</td></tr><tr><td>1.0</td></table>