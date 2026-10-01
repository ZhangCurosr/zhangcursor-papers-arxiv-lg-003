# STEEPEST GUIDANCE: A PRACTICAL AND PRINCI-PLED APPROACH TO INFERENCE-TIME ALIGNMENT OF FLOW AND DIFFUSION-BASED MODELS

Shokichi Takakura<sup>1</sup>, Akifumi Wachi<sup>1</sup>, Rei Higuchi<sup>2,3</sup>, Kohei Miyaguchi<sup>1</sup>, Taiji Suzuki<sup>2,3</sup> <sup>1</sup>LY Corporation, <sup>2</sup>The University of Tokyo, <sup>3</sup>RIKEN AIP

{stakakur,akifumi.wachi}@lycorp.co.jp, higuchi-rei714@g.ecc.u-tokyo.ac.jp kmiyaguc@lycorp.co.jp, taiji@mist.i.u-tokyo.ac.jp

## ABSTRACT

Inference-time alignment of flow and diffusion-based models is critical for achieving flexible generative modeling. Theoretically, Doob’s h-transform provides an elegant solution to this problem, and most existing methods are based on this principle. However, in practice, estimating the optimal guidance derived from Doob’s h-transform at inference time is challenging. To deal with this issue, we regard inference-time alignment as a sequential optimization problem in the space of probability measures and propose a novel framework called Steepest Guidance, based on the principle of maximizing local improvement in the objective. We provide a theoretical analysis of the proposed method and demonstrate its effectiveness through extensive experiments.

## 1 INTRODUCTION

Flow and diffusion-based generative models have emerged as powerful tools for modeling complex data distributions across various domains, including image generation (Ho et al., 2020; Lipman et al., 2022), natural language processing (Li et al., 2022), and molecular design (Hoogeboom et al., 2022). In many applications, we often aim to optimize some criteria over the generated samples. For instance, in image generation, we may want to generate images that not only look realistic but also satisfy certain aesthetic criteria or exhibit diversity. This can be formalized as an optimization problem on a reward functional $R [ \mu ]$ over the distribution of generated samples µ. Typically, the reward functional is defined as the expected value of a reward function $r : \mathbb { R } ^ { d } $ R over the distribution µ, $\begin{array} { r } { \mathrm { i . e . , } R [ \mu ] = \int r ( y ) \mu ( d y ) } \end{array}$ . However, in some applications, we may want to consider more complex rewards such as diversity-promoting objectives (Corso et al., 2023; Vinograd et al., 2026; Santi et al., 2025) or risk-sensitive objectives (Zhang et al., 2020; Santi et al., 2025; Wang et al., 2026), which cannot be expressed as the expected value of a reward function since they are non-linear in µ.

Reward-guided generation is often formulated as an optimal control problem (Domingo-Enrich et al., 2024), where the generative model is treated as a stochastic process, and the reward serves as a control objective. Theoretically, the optimal solution to this problem is given by Doob’s htransform (Rogers & Williams, 2000), which provides a principled way to modify the generative process to maximize the expected reward, while ensuring that the generated samples remain close to the original data distribution. Several methods (Domingo-Enrich et al., 2024; Uehara et al., 2024) have been proposed to address this problem by learning a guidance model that steers the generative process towards high-reward regions of the sample space. In addition, Marion et al. (2024); Kawata et al. (2025); Santi et al. (2025) have developed a general framework for fine-tuning diffusion and flow models to maximize arbitrary reward functionals including non-linear ones.

On the other hand, a line of work has developed training-free approaches, where the guidance is estimated during the inference phase without additional training. The plug-in estimator (Dandapanthula & Boffi, 2026) computes the optimal guidance via backpropagation through the generative process, which requires the reward and sampling process to be differentiable. Recently, to deal with nondifferentiable rewards, Zhu et al. (2026) have proposed REINFORCE (Williams, 1992) estimators for the optimal guidance.

In spite of the intensive research on training-free inference-time alignment of flow and diffusion models, there are still several challenges that remain to be addressed. First, most existing methods can only handle linear reward functionals, and cannot be applied to non-linear reward functionals. Second, even if the reward functional is linear, the optimal guidance is difficult to estimate in practice. Recently, Dandapanthula & Boffi (2026) have revealed that the plug-in estimator with finite Monte Carlo samples is biased, leading to suboptimal performance in reward-guided generation. In addition, as we show later, REINFORCE estimators also suffer from large bias, which causes the performance of inference-time alignment of flow models to degrade significantly.

In this paper, we regard the inference-time alignment of flow and diffusion models as a sequential optimization problem in the space of probability measures, and propose a novel method called Steepest Guidance based on the local improvement of the reward functional. For linear reward functionals, we show that it can be estimated in an unbiased manner, leading to improved performance in inference-time alignment. In addition, steepest guidance can be naturally applied to non-linear reward functionals through a particle approximation. Our contributions are summarized as follows:

• We regard the inference-time alignment of flow and diffusion models as a sequential optimization problem in the space of probability measures, and propose Steepest Guidance, which guides the generative process towards the steepest ascent direction of the reward functional. Furthermore, we generalize it to an entropy-regularized objective and develop Regularized Steepest Guidance.

• Accounting for the evolution of the objective, we establish provable reward improvement and global convergence of our proposed method under suitable assumptions. This can be seen as a functional extension of the existing analysis of Wasserstein gradient flows (Bakry et al., 2014; Chizat, 2022; Nitanda et al., 2022) from fixed-objective optimization to the optimization of objectives that evolve along the generative process.

• Through extensive experiments, we demonstrate that the proposed method consistently outperforms existing approaches and successfully handles non-linear reward functionals.

## 1.1 RELATED WORK

Here, we briefly review the related work and refer to Appendix A for a more detailed discussion.

Inference-time Alignment of Flow and Diffusion Models. Inference-time alignment of flow and diffusion models can be categorized into two main approaches: selection-based methods and guidance-based methods. Selection-based methods include sequential Monte Carlo (Wu et al., 2023), SVDD (Li et al., 2024), and Best-of-N (Nakano et al., 2021), which utilize multiple particles and select or resample them to obtain better samples. Guidance-based methods include DOIT (Zhu et al., 2026) and Gradient Guidance (Guo et al., 2024), which are often efficient since they can utilize gradient information of the reward function. However, they require computationally expensive backpropagation or suffer from large bias.

Optimization in the Space of Probability Measures. Several works (Marion et al., 2024; Kawata et al., 2025; Santi et al., 2025) have regarded reward-guided generation as an optimization problem in the space of probability measures and developed fine-tuning methods for general reward functionals, which require additional training. On the other hand, (Mean-field) Langevin dynamics (Welling & Teh, 2011; Chizat, 2022; Nitanda et al., 2022) is based on a Wasserstein gradient flow and optimizes a functional over the space of probability measures through gradient-based particle updates. However, the distribution of the real-world data is often complex and multimodal, which hinders the convergence of such methods. Recently, Slowly Annealed Langevin Dynamics (SALD) (Nitanda et al., 2026) has been applied to inference-time alignment of flow and diffusion models, but i cannot be applied to non-linear reward functionals.

## 2 PRELIMINARIES

In this section, we introduce the flow and diffusion-based generative models and formulate the inference-time alignment as an optimal control problem. For $d \in \mathbb { N } .$ , let $\mathcal { P }$ be the space of probability measures on $\mathbb { R } ^ { d }$ which have density functions, and finite entropy and second moment. We denote a data distribution by $\pi _ { 1 } \in \mathcal { P }$

## 2.1 FLOW AND DIFFUSION-BASED MODELS

A flow matching model is formulated as the following ordinary differential equation (ODE):

$$
\mathrm { d } Y _ { t } = v _ { t } ( Y _ { t } ) \mathrm { d } t , \quad Y _ { 0 } \sim { \mathcal { N } } ( 0 , I ) ,
$$

where $v _ { t } ( Y _ { t } )$ is a time-dependent vector field, which is approximated by a neural network. Typically, $v _ { t } ( x )$ is defined as $\begin{array} { r } { \mathbb { E } \left[ \frac { \mathrm { d } \bar { Y _ { t } } } { \mathrm { d } t } \mid Y _ { t } = x \right] } \end{array}$ , where $Y _ { t } : = ( 1 - t ) Y _ { 0 } + t Y _ { 1 } , Y _ { 0 } \sim \mathcal { N } ( 0 , I )$ , and $Y _ { 1 } \sim \pi _ { 1 }$ Furthermore, for any $\sigma _ { t } \in \mathbb { R }$ , the solution of the SDE

$$
\mathrm { d } Y _ { t } = \left( v _ { t } ( Y _ { t } ) + \frac { \sigma _ { t } ^ { 2 } } { 2 } \nabla \log \pi _ { t } ( Y _ { t } ) \right) \mathrm { d } t + \sigma _ { t } \mathrm { d } W _ { t } , \quad Y _ { 0 } \sim \mathcal { N } ( 0 , I )\tag{1}
$$

has the same marginal distribution as the solution of the ODE, where $\pi _ { t }$ is the marginal distribution of $Y _ { t }$ . In this paper, we consider a memoryless noise schedule $\sigma _ { t } ^ { 2 } = 2 ( 1 - t ) / t$ following Domingo-Enrich et al. (2024); Bergmeister et al. (2026).

On the other hand, for diffusion models, we first define the following (forward) SDE (Jiao et al., 2025):

$$
\mathrm { d } X _ { t } = - \frac { 1 } { 2 ( 1 - t ) } X _ { t } \mathrm { d } t + \frac { 1 } { \sqrt { 1 - t } } \mathrm { d } W _ { t } , \quad X _ { 0 } \sim \pi _ { 1 } .
$$

Then, the reverse-time SDE is given by

$$
\mathrm { d } Y _ { t } = \left( \frac { 1 } { 2 } Y _ { t } + \nabla \log \pi _ { t } ( Y _ { t } ) \right) \frac { \mathrm { d } t } { t } + \frac { 1 } { \sqrt { t } } \mathrm { d } W _ { t } , \quad Y _ { 0 } \sim \mathcal { N } ( 0 , I ) ,
$$

where $Y _ { t }$ has the same marginal distribution as $X _ { 1 - t }$ . Here, ∇ log $\pi _ { t } ( y )$ is the score function of $\pi _ { t }$ and is approximated by a neural network.

In this paper, we consider the following SDE as a general form of flow and diffusion models:

$$
\mathrm { d } Y _ { t } = b _ { t } ( Y _ { t } ) \mathrm { d } t + \sigma _ { t } \mathrm { d } W _ { t } , \quad Y _ { \varepsilon } \sim \pi _ { \varepsilon } , \quad t \in [ \varepsilon , 1 ] .
$$

Here, we introduce a small constant $\varepsilon \geq 0$ and assume that $\varepsilon > 0$ for theoretical analysis to avoid blow-up of $\sigma _ { t } \mathrm { a t } t = 0$ . We denote its path measure by $\mathbb { P } _ { \pi }$ and the marginal distribution of $Y _ { t }$ by $\pi _ { t } .$ In the following, we assume that the second moment of $\pi _ { t }$ is finite.

## 2.2 INFERENCE-TIME ALIGNMENT OF FLOW MODELS

In this paper, we aim to maximize a reward functional $R : \mathcal P \to \mathbb { R }$ over the generated samples by adding a guidance term $g _ { t } ( y )$ to the generative process:

$$
\mathrm { d } Y _ { t } = \left( b _ { t } ( Y _ { t } ) + g _ { t } ( Y _ { t } ) \right) \mathrm { d } t + \sigma _ { t } \mathrm { d } W _ { t } , \quad Y _ { \varepsilon } \sim \pi _ { \varepsilon } , \quad t \in [ \varepsilon , 1 ] .
$$

This guidance based formula has been widely employed by several existing works such as classifier/classifier-free guidance (Dhariwal & Nichol, 2021; Ho & Salimans, 2022). We call a guidance $g _ { t } ( y )$ admissible if it satisfies $\begin{array} { r } { \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \left\| g _ { t } ( Y _ { t } ) \right\| ^ { 2 } / \sigma _ { t } ^ { 2 } \right] \mathrm { d } t < \infty } \end{array}$ . We denote its path measure by $\mathbb { P } _ { \pi ^ { g } }$ and the marginal distribution of $Y _ { t }$ by $\pi _ { t } ^ { g }$ . Then, we consider the following optimization problem:

$$
\operatorname* { m a x } _ { g } R [ \pi _ { 1 } ^ { g } ] - { \frac { 1 } { \lambda } } \mathrm { K L } ( \mathbb { P } _ { \pi ^ { g } } \mid \mathbb { P } _ { \pi } ) ,\tag{2}
$$

where KL is the KL divergence and $\lambda > 0$ is a regularization parameter. Typically, the reward functional is defined as the expected value of a reward function $\bar { r } : \mathbb { R } ^ { d } \to$ R over the distribution $\mu ,$ , i.e., $\begin{array} { r } { R [ \mu ] = \int r ( y ) \mu ( d y ) } \end{array}$ . On the other hand, in some applications, we may want to consider more complex rewards such as risk-averse objectives like CVaR or diversity-promoting objectives like Rao’s quadratic entropy:

$$
R _ { \mathrm { C V a R } } [ \mu ] = \mathbb { E } _ { \mu } \left[ r ( Y ) \mid r ( Y ) \leq q _ { \alpha } \right] , \quad R _ { \mathrm { R a o } } [ \mu ] = - \frac { 1 } { 2 } \int \int k ( y , y ^ { \prime } ) \mu ( d y ) \mu ( d y ^ { \prime } ) ,
$$

where $q _ { \alpha }$ is the α-quantile of $r ( Y )$ and $k : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }  \mathbb { R }$ is a positive definite kernel. See Table 3 for other examples of (non-linear) reward functionals.

In this paper, we assume that the reward functional R has a functional derivative $\begin{array} { r } { \frac { \delta R } { \delta \mu } [ \mu ] ( y ) \quad } \end{array}$ with $\begin{array} { r } { \| \nabla \frac { \delta R } { \delta \mu } [ \mu ] ( y ) \| , | \frac { \delta R } { \delta \mu } [ \mu ] ( y ) | \leq C _ { V } \mathrm { a n d } \mu _ { 1 } ^ { * } : = \mathrm { a r g m a x } _ { \mu _ { 1 } \in \mathcal { P } } R [ \mu _ { 1 } ] - \frac { 1 } { \lambda } \mathrm { K L } ( \mu _ { 1 } | \pi _ { 1 } ) } \end{array}$ exists.

Definition 2.1 (Functional Derivative). The functional derivative of a reward functional $R : \mathcal P $ R with respect to a measure $\mu \in \mathcal P$ is a function $\begin{array} { r }  \frac { \delta R } { \delta \mu } [ \mu ] ( y ) \end{array}$ such that for any $\nu \in \mathcal { P }$ and the mixture path $\mu _ { \epsilon } : = ( 1 - \epsilon ) \mu + \epsilon \nu .$ , it holds that $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } \epsilon } R [ \mu _ { \epsilon } ] \Big | _ { \epsilon = 0 } = \int \frac { \delta R } { \delta \mu } [ \mu ] ( y ) ( \nu - \mu ) ( \mathrm { d } y ) } \end{array}$

## 2.3 OPTIMAL CONTROL VIA DOOB’S h-TRANSFORM

As a theoretical principle to determine the guidance for the reward maximization, Doob’s $h -$ transform has been utilized by several existing works (Uehara et al., 2024; Domingo-Enrich et al., 2024; Kawata et al., 2025; Bergmeister et al., 2026; Zhu et al., 2026) to guide the generative process. Formally, Doob’s h-function is defined as $\begin{array} { r } { h _ { t } ( y ) = \operatorname { \mathbb { E } } \left| \exp ( \lambda \frac { \delta R } { \delta \mu } [ \mu _ { 1 } ^ { * } ] ( Y _ { 1 } ) ) \mid Y _ { t } = y \right| } \end{array}$ , and also it can be defined through the following SDE in the sense that the density ratio between the marginal distributions of $Y _ { t }$ for the original process and the following process is proportional to $h _ { t } \colon$

$$
\mathrm { d } Y _ { t } = \left( b _ { t } ( Y _ { t } ) + g _ { t } ^ { * } ( Y _ { t } ) \right) \mathrm { d } t + \sigma _ { t } \mathrm { d } W _ { t } , \quad Y _ { 0 } \sim \mathcal { N } ( 0 , I ) ,
$$

where $g _ { t } ^ { * } ( y ) = \sigma _ { t } ^ { 2 } \nabla$ log $h _ { t } ( y )$ . As discussed in Domingo-Enrich et al. (2024), under the memoryless noise schedule this Doob transform solves the control problem (2) with $\varepsilon = 0$

Unfortunately, Doob’s h-transform does not admit a closed-form expression and requires numerical estimation, presenting a significant computational challenge. To tackle this problem, some recent works have proposed to estimate the optimal guidance $g _ { t } ^ { * } ( y )$ at inference time using Monte Carlo samples from the conditional distribution $\pi _ { 1 } ( \cdot \ \overline { { | } } \ Y _ { t } = y )$ . From the chain rule, the optimal guidance $g _ { t } ^ { * } ( y )$ can be expressed as

$$
g _ { t } ^ { * } ( y ) = \sigma _ { t } ^ { 2 } \frac { \nabla _ { y } \mathbb { E } _ { \pi _ { 1 } ( \cdot \vert Y _ { t } = y ) } \left[ \exp ( \lambda \frac { \delta R } { \delta \mu } [ \mu _ { 1 } ^ { * } ] ( Y _ { 1 } ) ) \right] } { \mathbb { E } _ { \pi _ { 1 } ( \cdot \vert Y _ { t } = y ) } \left[ \exp ( \lambda \frac { \delta R } { \delta \mu } [ \mu _ { 1 } ^ { * } ] ( Y _ { 1 } ) ) \right] } .
$$

Let $z _ { 1 } , \ldots , z _ { k }$ be i.i.d. samples from the conditional distribution $\pi _ { 1 } ( Z \mid Y _ { t } \ = \ y )$ . Then, the denominator can be estimated as $\begin{array} { r } { \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \exp ( \lambda \frac { \delta R } { \delta \mu } [ \mu _ { 1 } ^ { * } ] ( z _ { i } ) ) } \end{array}$ . To estimate the numerator, there are two main approaches: Plug-in and REINFORCE estimators.

Plug-in Estimator. Assume that the sampling process can be expressed as $z _ { i } = f ( y , \epsilon _ { i } )$ for some deterministic function f and random variable $\epsilon _ { i } .$ . Then, the plug-in estimator of $g _ { t } ^ { * } ( y )$ is given by

$$
\nabla _ { y } \mathbb { E } _ { \pi _ { 1 } ( \cdot | Y _ { t } = y ) } \left[ \exp \left( \lambda \frac { \delta R } { \delta \mu } [ \mu _ { 1 } ^ { * } ] ( Y _ { 1 } ) \right) \right] \simeq \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \nabla _ { y } \exp \left( \lambda \frac { \delta R } { \delta \mu } [ \mu _ { 1 } ^ { * } ] ( f ( y , \epsilon _ { i } ) ) \right) .
$$

Note that the plug-in estimator requires the reward and sampling process to be differentiable. Furthermore, even if the reward and sampling process are differentiable, computing the gradient is computationally expensive since it requires backpropagation through the generative process.

REINFORCE Estimator. Following Zhu et al. (2026), let us consider the following identity (Williams, 1992):

$$
\nabla _ { y } \mathbb { E } _ { \pi _ { 1 } ( \cdot | Y _ { t } = y ) } \left[ f ( Y _ { 1 } ) \right] = \mathbb { E } _ { \pi _ { 1 } ( \cdot | Y _ { t } = y ) } \left[ f ( Y _ { 1 } ) \nabla _ { y } \log { \pi _ { 1 } ( Y _ { 1 } \mid Y _ { t } = y ) } \right] .
$$

Then, the REINFORCE estimator of $g _ { t } ^ { * } ( y )$ is given by

$$
\nabla _ { y } \mathbb { E } _ { \pi _ { 1 } ( \cdot | Y _ { t } = y ) } \left[ \exp \left( \lambda \frac { \delta R } { \delta \mu } [ \mu _ { 1 } ^ { * } ] ( Y _ { 1 } ) \right) \right] \simeq \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \exp \left( \lambda \frac { \delta R } { \delta \mu } [ \mu _ { 1 } ^ { * } ] ( z _ { i } ) \right) \nabla _ { y } \log \pi _ { 1 } ( z _ { i } \mid Y _ { t } = y ) .
$$

Instead of the differentiability of the reward and sampling process, the REINFORCE estimator only requires the differentiability of the log-likelihood of the conditional distribution $\pi _ { 1 } ( \cdot \ | \ Y _ { t } = y )$ Therefore, the REINFORCE estimator can be applied to non-differentiable reward functions. As shown in the following lemma, we can compute the gradient of the log-likelihood of the conditional distribution $\pi _ { 1 } ( \cdot \mid Y _ { t } = y )$ from the score function of the marginal distribution $\pi _ { t }$

Lemma 2.2. For any $t \in [ \varepsilon , 1 )$ , we have

$$
\nabla _ { y } \log \pi _ { 1 } ( z \mid Y _ { t } = y ) = { \left\{ \begin{array} { l l } { - { \frac { 1 } { ( 1 - t ) ^ { 2 } } } ( y - t z ) - { \frac { t v _ { t } ( y ) - y } { 1 - t } } } & { ( F l o w \ M a t c h i n g \ M o d e l ) , } \\ { - { \frac { 1 } { 1 - t } } \cdot ( y - { \sqrt { t } } z ) - \nabla \log \pi _ { t } ( y ) } & { ( D i f f u s i o n \ M o d e l ) . } \end{array} \right. }
$$

See Appendix E for the proof.

Challenges in Estimating the Optimal Guidance. While Doob’s h-transform provides a principled way to compute the optimal guidance, there are two main challenges in computing the optimal guidance $g _ { t } ^ { * } ( y )$

Non-linearity ofthe rewardfunctional: First, while $\begin{array} { r } { \frac { \delta R } { \delta \mu } [ \mu _ { 1 } ^ { * } ] ( y ) = r ( y ) } \end{array}$ for linear reward functionals, we cannot compute $\frac { \delta R } { \delta \mu } [ \mu _ { 1 } ^ { * } ] ( y )$ for general non-linear reward functionals because it requires the knowledge of the optimal distribution $\mu _ { 1 } ^ { * }$ , which is exactly what we are looking for.

Non-linearity of the guidance: Second, even if we can compute $\textstyle { \frac { \delta R } { \delta \mu } } [ \mu _ { 1 } ^ { * } ]$ , we cannot obtain unbiased estimates of $g _ { t } ^ { * } ( y )$ since a non-linear transformation (logarithm) is applied to the expectation. Therefore, if we use a finite number of samples, the estimator of $g _ { t } ^ { * } ( y )$ is biased. For the plug-in estimator, Dandapanthula & Boffi (2026) have shown that the bias of the plug-in estimator leads to suboptimal performance in reward-guided generation. Here, we show that the REINFORCE estimator may be biased even in the case where the plug-in estimator is unbiased. Specifically, we consider a simple toy problem, where the terminal distribution is the one-dimensional Gaussian and the reward function is $r ( y ) = y .$ In this case, the plug-in estimator is unbiased but the REINFORCE estimator suffers from large bias as shown in Fig. 1. See Appendix L for detailed experimental settings.

![](images/8f780fc33a078c508342c8d9504c222e5df082fbccdbea7b67616c2697f7bef6.jpg)  
Figure 1: Relative bias of the REINFORCE estimators for the toy Gaussian problem.

## 3 PROPOSED METHOD: STEEPEST GUIDANCE

As shown in the previous section, the optimal guidance $g _ { t } ^ { * } ( y )$ is difficult to estimate in practice due to the non-linearity of the reward functional and the guidance. Instead of estimating the globally optimal guidance, we propose to consider the local optimality of the guidance and design a steepest guidance that improves the reward functional in a local manner. The following proposition is fundamental to analyzing the improvement of the reward functional with respect to the guidance.

Proposition 3.1. For any admissible guidance $g _ { t } ( y )$ , we have

$$
R [ \pi _ { 1 } ^ { g } ] - R [ \pi _ { 1 } ] = \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ g _ { t } ( Y _ { t } ) \cdot \nabla \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( Y _ { t } ) \right] \mathrm { d } t , \mathrm { K L } ( \mathbb { P } _ { \pi ^ { g } } \mid \mathbb { P } _ { \pi } ) = \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \frac { \Vert g _ { t } ( Y _ { t } ) \Vert ^ { 2 } } { 2 \sigma _ { t } ^ { 2 } } \right] \mathrm { d } t ,
$$

where $V ( t , \mu ) : = R [ K _ { t } \mu ]$ and $K _ { t }$ is the transition kernelfrom t to 1.

See Appendix F for the proof. We observe that this proposition provides the steepest direction to improve the objective at each time $t ,$ and that the functional derivative $\frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( Y _ { t } )$ directly appears in the expectation without any non-linear transformation, in contrast to Doob’s h-transform. Similar results can be found for linear functionals in Jiao et al. (2025) but we extend them to general reward functionals through a generator-based argument that extends to non-linear functionals.

## 3.1 LOCAL IMPROVEMENT OF THE REWARD FUNCTIONAL

Here, we explain the intuitive idea behind our proposed method. Let $\mu _ { \tau }$ be the marginal distribution of $Y _ { \tau }$ at time $\tau \in \ [ \varepsilon , 1 )$ . Then, we consider the following SDE, in which a time-independent guidance is applied over a small interval $[ \tau , \tau + \Delta \tau ]$ with $\tau + \Delta \tau \leq 1$

$$
\begin{array} { r l } { \mathrm { d } Y _ { t } = ( b _ { t } ( Y _ { t } ) + g ( Y _ { t } ) ) \cdot \mathrm { d } t + \sigma _ { t } \mathrm { d } W _ { t } } & { ( t \in [ \tau , \tau + \Delta \tau ] ) , } \\ { \mathrm { d } Y _ { t } = b _ { t } ( Y _ { t } ) \cdot \mathrm { d } t + \sigma _ { t } \mathrm { d } W _ { t } } & { ( t \in [ \tau + \Delta \tau , 1 ] ) . } \end{array}
$$

Intuitively, Proposition 3.1 implies that for a small time interval $\Delta \tau .$ , we have

$$
\begin{array} { r } { R [ \mu _ { 1 } ^ { g } ] - R [ \mu _ { 1 } ] - \frac { 1 } { \lambda } \mathrm { K L } ( \mathbb { P } _ { \mu ^ { g } } \mid \mathbb { P } _ { \mu } ) \simeq \Delta \tau \cdot \mathbb { E } _ { \mu _ { \tau } } \left[ g ( Y _ { \tau } ) \cdot \nabla _ { y } \frac { \delta V ( \tau , \mu _ { \tau } ) } { \delta \mu } ( Y _ { \tau } ) - \frac { \Vert g ( Y _ { \tau } ) \Vert ^ { 2 } } { 2 \lambda \sigma _ { \tau } ^ { 2 } } \right] , } \end{array}
$$

which is a quadratic function of $g$ and can be maximized by setting $\begin{array} { r } { g ( y ) = \lambda \sigma _ { \tau } ^ { 2 } \nabla _ { y } \frac { \delta V ( \tau , \mu _ { \tau } ) } { \delta \mu } ( y ) } \end{array}$

Based on the above observation, we propose a steepest guidance defined as

$$
g _ { t } ( y ) = \lambda \sigma _ { t } ^ { 2 } \nabla _ { y } \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) ,\tag{3}
$$

which is characterized by the local optimality of the reward improvement. The following theorem shows that the steepest guidance provably improves the reward functional while controlling the KL divergence between the original and guided processes.

Theorem 3.2. Assume that the steepest guidance defined in $E q . \ ( 3 )$ is admissible. Then, we have

$$
\begin{array} { r } { R [ \pi _ { 1 } ^ { g } ] - R [ \pi _ { 1 } ] = \lambda \Gamma _ { 1 } , \quad \mathrm { K L } ( \mathbb { P } _ { \pi ^ { g } } \mid \mathbb { P } _ { \pi } ) = \frac { \lambda ^ { 2 } \Gamma _ { 1 } } { 2 } , } \end{array}
$$

$$
\begin{array} { r } { w h e r e ~ \Gamma _ { t } : = \int _ { \varepsilon } ^ { t } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \sigma _ { t } ^ { 2 } \left\| \nabla _ { y } \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( Y _ { t } ) \right\| ^ { 2 } \right] d t f o r t \in [ \varepsilon , 1 ] . } \end{array}
$$

See Appendix G for the proof. We may interpret this theorem as a functional version of the descent lemma, analogous to the standard analysis of finite-dimensional gradient descent. It also suggests that a larger λ is expected to yield a greater improvement in the reward functional. Indeed, in Section 4.1, we establish global convergence for a regularized variant of steepest guidance by taking sufficiently large λ, under suitable assumptions.

## 3.2 PRACTICAL IMPLEMENTATION OF STEEPEST GUIDANCE

To implement the steepest guidance in practice, we need to discretize the process and estimate $g _ { t } ( y )$ from finite Monte Carlo samples. First, we focus on the case where R is linear in $\mu ,$ i.e., $\begin{array} { r } { \dot { R } [ \mu ] = \int r ( y ) \mu ( d y ) } \end{array}$ for some reward function $r : \mathbb { R } ^ { d }  \mathbb { R }$ . In this case, we have $V ( t , \mu ) =$ $\textstyle \int r ( y _ { 1 } ) K _ { t } \mu ( d y _ { 1 } )$ and $\begin{array} { r } { \frac { \delta V ( t , \mu ) } { \delta \mu } ( y ) = E [ r ( Y _ { 1 } ) \mid Y _ { t } = y ] } \end{array}$ . Therefore, the steepest guidance can be estimated as follows:

$$
\hat { g } _ { t } ^ { \mathrm { s t e e p e s t } } ( y ) = \lambda \sigma _ { t } ^ { 2 } \frac { 1 } { k } \sum _ { i = 1 } ^ { k } r ( y _ { i } ) \nabla _ { y } \log \pi _ { 1 } ( y _ { i } \mid Y _ { t } = y ) ,\tag{4}
$$

where $y _ { i }$ are i.i.d. samples from the conditional distribution $\pi _ { 1 } ( \cdot \mid Y _ { t } = y )$ . An important property of the above estimator is that it is unbiased, i.e., E $\left[ \hat { g } _ { t } ^ { \mathrm { s t e e p e s t } } ( y ) \right] = g _ { t } ( y )$ , which is in contrast to the case of the optimal guidance.

Remark 3.3. In practice, it is often computationally expensive to sample from the conditional distribution $\pi _ { 1 } ( \cdot \ | \ \overline { { Y } } _ { t } = y )$ . Thus, previous works have proposed to use the approximate sampling processes such as GLASS flow (Holderrieth et al., 2025) and Diamond Map (Holderrieth et al., 2026). Under the approximate sampling process, the estimator in Eq. (4) is biased but can still be effective in practice, as shown in our experiments.

In the case where R is non-linear in $\mu ,$ the population first variation $\textstyle { \frac { \delta R } { \delta \mu } } [ \mu _ { 1 , t } ] ( y )$ depends on the unknown terminal distribution $\mu _ { 1 , t } : = K _ { t } \pi _ { t } ^ { g }$ . Following previous work (Takakura et al., 2026), we approximate this first variation using lookahead particles. Let $\{ y ^ { ( j ) } \} _ { j = 1 } ^ { N }$ be samples in a batch. Then, for each $j = 1 , \ldots , N$ , we generate k i.i.d. samples $\{ y _ { i } ^ { ( j ) } \} _ { i = 1 } ^ { k }$ from the conditional distribution $\pi _ { 1 } ( \cdot \mid Y _ { t } = y ^ { ( j ) } )$ and define the empirical measure $\begin{array} { r } { \hat { \mu } _ { 1 } = \frac { 1 } { N k } \sum _ { i = 1 } ^ { k } \sum _ { j = 1 } ^ { N } \delta _ { y _ { i } ^ { ( j ) } } } \end{array}$ <sub>)</sub> . With a slight abuse of notation, we write $\frac { \delta R } { \delta \mu } [ \hat { \mu } _ { 1 } ] ( y )$ for the sample-based plug-in approximation obtained by replacing the population quantities in $\textstyle { \frac { \delta R } { \delta \mu } } [ \mu _ { 1 , t } ] ( y )$ with their empirical counterparts. We then estimate the steepest guidance by

$$
\hat { g } _ { t } ^ { \mathrm { s t e e p e s t } } ( y ^ { ( j ) } ) = \lambda \sigma _ { t } ^ { 2 } \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \frac { \delta R } { \delta \mu } [ \hat { \mu } _ { 1 } ] ( y _ { i } ^ { ( j ) } ) \nabla _ { y } \log \pi _ { 1 } ( y _ { i } ^ { ( j ) } \mid Y _ { t } = y ^ { ( j ) } ) .\tag{5}
$$

Unlike the linear-reward estimator in Eq. (4), this sample-based estimator is generally biased for finite N because the estimated first variation depends non-linearly on the empirical samples. We show the detailed algorithm of steepest guidance in Algorithm 1.

## 4 REGULARIZED STEEPEST GUIDANCE

We are often interested in optimizing the terminal KL-regularized objective instead of the original reward functional R to ensure that the generated samples are close to the original distribution:

$$
\operatorname* { m a x } _ { g } J _ { \eta } ( \pi _ { 1 } ^ { g } ) : = \operatorname* { m a x } _ { g } R [ \pi _ { 1 } ^ { g } ] - \frac { 1 } { \eta } \mathrm { K L } ( \pi _ { 1 } ^ { g } \mid \pi _ { 1 } ) ,\tag{6}
$$

for $\eta > 0$ . From the data-processing inequality, we have KL $_ { \iota } ( \pi _ { 1 } ^ { g } \mid \pi _ { 1 } ) \le \mathrm { K L } ( \mathbb { P } _ { \pi ^ { g } } \mid \mathbb { P } _ { \pi } )$ . Therefore, we can obtain the following results:

Corollary 4.1. Applying steepest guidance $\begin{array} { r } { g _ { t } ( y ) = \lambda \sigma _ { t } ^ { 2 } \nabla _ { y } \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) } \end{array}$ with $0 < \lambda \leq 2 \eta$ , we have

$$
J _ { \eta } ( \pi _ { 1 } ^ { g } ) - J _ { \eta } ( \pi _ { 1 } ) \ge \lambda \left( 1 - \frac { \lambda } { 2 \eta } \right) \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \sigma _ { t } ^ { 2 } \left\| \nabla _ { y } \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( Y _ { t } ) \right\| ^ { 2 } \right] \mathrm { d } t \ge 0 .
$$

See Appendix H for the proof. This corollary shows that the steepest guidance improves the KL regularized objective for $0 < \lambda \leq 2 \eta$ . However, this guarantee constrains the guidance strength relative to the KL regularization parameter, and thus λ cannot be chosen independently of η.

Instead of controlling the terminal KL divergence via pathwise KL divergence, we can consider the terminal KL-regularized objective $J _ { \eta } ( K _ { t } \mu )$ directly thanks to our general formulation. However, computing the first-order variation of $J _ { \eta } ( K _ { t } \mu )$ requires the density ratio $\frac { K _ { t } \mu ( y ) } { K _ { t } \pi _ { t } ( y ) }$ , which is generally intractable. To deal with this issue, let us consider the quantity $\begin{array} { r } { \mathcal { V } ( t , \mu ) : = V ( t , \mu ) - \frac { 1 } { \eta } \mathrm { K L } ( \mu \mid \pi _ { t } ) } \end{array}$ From the data-processing inequality, we can show that $\mathcal { V } ( t , \mu )$ is a lower bound on $J _ { \eta } ( K _ { t } \mu )$ . Based on this observation, we propose to use $\mathcal { V } ( t , \mu )$ to construct the following guided SDE:

$$
\mathrm { d } Y _ { t } ^ { g } = \big ( b _ { t } \big ( Y _ { t } ^ { g } \big ) + g _ { t } \big ( Y _ { t } ^ { g } \big ) \big ) \cdot \mathrm { d } t + \sigma _ { t } \mathrm { d } W _ { t } ,\tag{7}
$$

where

$$
g _ { t } ( y ) = \lambda \sigma _ { t } ^ { 2 } \nabla _ { y } \frac { \delta \mathcal { V } ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) = \lambda \sigma _ { t } ^ { 2 } \nabla _ { y } \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) - \frac { \lambda } { \eta } \sigma _ { t } ^ { 2 } \nabla _ { y } \log \frac { \pi _ { t } ^ { g } ( y ) } { \pi _ { t } ( y ) } .
$$

Even in this case, the steepest guidance requires the density ratio but we can construct an SDE that has the same marginal distribution as $\operatorname { E q . } \left( 7 \right)$ without computing it.

Proposition 4.2. The following SDE has the same marginal distribution as $E q . \ ( 7 )$

$$
\begin{array} { r l } & { \mathrm { d } Y _ { t } ^ { g } = ( b _ { t } ( Y _ { t } ^ { g } ) + g _ { t } ( Y _ { t } ^ { g } ) ) \cdot \mathrm { d } t + \sqrt { 1 + \frac { 2 \lambda } { \eta } \cdot \sigma _ { t } \cdot \mathrm { d } W _ { t } } , } \\ & { } \\ & { w h e r e \enspace g _ { t } ( y ) : = \lambda \sigma _ { t } ^ { 2 } \nabla _ { y } \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) + \frac { \lambda } { \eta } \sigma _ { t } ^ { 2 } \nabla _ { y } \log \pi _ { t } ( y ) . } \end{array}
$$

See Appendix I for the proof. We call the above method regularized steepest guidance. As shown in the following theorem, regularized steepest guidance improves $J _ { \eta }$

Theorem 4.3. For the regularized steepest guidance, we have

$$
J _ { \eta } ( \pi _ { 1 } ^ { g } ) - J _ { \eta } ( \pi _ { 1 } ) \ge \lambda \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \sigma _ { t } ^ { 2 } \left\| \nabla \frac { \delta \mathcal { V } ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( Y _ { t } ) \right\| ^ { 2 } \right] \mathrm { d } t .
$$

See Appendix J for the proof. In contrast to Corollary 4.1, the above theorem ensures the improvement of the KL-regularized objective even when $\lambda > 2 \eta$

## 4.1 GLOBAL CONVERGENCE FOR CONCAVE REWARD FUNCTIONALS

If the reward functional R is concave, i.e., for any $\mu , \nu \in \mathcal P$ and $\theta \in [ 0 , 1 ]$ , we have $R ( \theta \mu + ( 1 -$ $\theta ) \nu ) \geq \theta R ( \mu ) + ( 1 - \theta ) R ( \nu )$ , then we can prove the global convergence of the regularized steepest guidance under some structural assumptions. For each $t \in [ \varepsilon , 1 ] ,$ , let $\pi _ { t } ^ { * } \in \mathrm { { a r g m a x } } _ { \mu \in \mathcal { P } } \mathcal { V } ( t , \mu )$ Following the literature (Nitanda et al., 2022; Chizat, 2022), we assume the uniform log-Sobolev inequality and additional regularity conditions:

Assumption 4.4. (Uniform LSI) For any $t \in [ \varepsilon , 1 ]$ , there exists a constant $\alpha _ { t } > 0$ such that for any $\mu \in \mathcal P$ and smooth function $g : \mathbb { R } ^ { d }  \mathbb { R } .$ , we have

$$
\mathbb { E } _ { \nu _ { t } ^ { \mu } } \left[ g ^ { 2 } ( Y _ { t } ) \log g ^ { 2 } ( Y _ { t } ) \right] - \mathbb { E } _ { \nu _ { t } ^ { \mu } } \left[ g ^ { 2 } ( Y _ { t } ) \right] \log E _ { \nu _ { t } ^ { \mu } } [ g ^ { 2 } ( Y _ { t } ) ] \leq \frac { 2 } { \alpha _ { t } } \mathbb { E } _ { \nu _ { t } ^ { \mu } } \left[ \| \nabla _ { y } g ( Y _ { t } ) \| ^ { 2 } \right] ,
$$

where $\begin{array} { r } { \nu _ { t } ^ { \mu } ( y ) \propto \exp ( \eta \frac { \delta V ( t , \mu ) } { \delta \mu } ( y ) ) \cdot \pi _ { t } ( y ) } \end{array}$ . (Regularity condition) Furthermore, defining

$$
s _ { t } ( y ) : = \nabla _ { y } \log \frac { \pi _ { t } ^ { * } ( y ) } { \pi _ { t } ( y ) } , \Phi _ { t } ( y ) : = \left. s _ { t } ( y ) \right. ^ { 2 } , \Psi _ { t } ( y ) : = \operatorname { d i v } s _ { t } ( y ) + s _ { t } ( y ) \cdot \nabla _ { y } \log \pi _ { t } ^ { * } ( y ) ,
$$

there exists $C _ { t } < \infty$ such that s $\begin{array} { r } { \mathsf { u p } _ { y } \Phi _ { t } ( y ) \le C _ { t } , \mathsf { s u p } _ { y } | \Psi _ { t } ( y ) | \le C _ { t } } \end{array}$

![](images/365cfa818f36e61f7815825308fa97a8f53f2021199b4be220a01fb37f0141c5.jpg)  
Figure 2: Toy experiment comparing Doob’s h-transform (left), Steepest Guidance (middle), and Regularized Steepest Guidance (right).

Table 1: Text-to-image reward summary for a prompt “A portrait photo of a golden-yellow lion”. Results are reported as mean ± standard deviation over 32 generated images for each model, reward function, and method.
<table><tr><td>Model</td><td>Reward</td><td></td><td>Unguided Steepest (Ours)</td><td>DOIT</td><td>SVDD</td></tr><tr><td>SD</td><td>Blueness</td><td> $- 0 . 7 2 \pm 0 . 0 6$ </td><td> ${ \bf - 0 . 1 2 \pm 0 . 2 1 }$ </td><td> $- 0 . 6 9 \pm 0 . 0 6$ </td><td> $- 0 . 6 7 \pm 0 . 0 6$ </td></tr><tr><td></td><td>ImageReward</td><td> $- 0 . 3 9 \pm 0 . 4 9$ </td><td> ${ \bf 0 . 8 6 \pm 0 . 5 3 }$ </td><td> $0 . 1 0 \pm 0 . 4 5$ </td><td> $0 . 6 8 \pm 0 . 4 4$ </td></tr><tr><td></td><td>PickScore</td><td> $2 0 . 9 7 \pm 0 . 4 8$ </td><td> ${ \bf 2 1 . 9 0 \pm 0 . 4 8 }$ </td><td> $2 1 . 3 8 \pm 0 . 4 5$ </td><td> $2 1 . 8 4 \pm 0 . 4 4$ </td></tr><tr><td></td><td>Compressibility</td><td> $- 1 4 . 3 5 \pm 1 . 8 0$ </td><td> $- 5 . 2 5 \pm 1 . 6 5$ </td><td> $- 1 4 . 1 7 \pm 1 . 7 4$ </td><td> $- 1 3 . 5 4 \pm 1 . 7 7$ </td></tr><tr><td>FLUX</td><td>Blueness</td><td> $- 0 . 3 3 \pm 0 . 0 5$ </td><td> $\mathbf { - 0 . 1 3 \pm 0 . 0 6 }$ </td><td> $- 0 . 2 9 \pm 0 . 0 4$ </td><td> $- 0 . 2 3 \pm 0 . 0 4$ </td></tr><tr><td></td><td>ImageReward</td><td> $0 . 0 5 \pm 0 . 2 8$ </td><td> ${ \bf 1 . 6 6 \pm 0 . 2 4 }$ </td><td> $0 . 4 5 \pm 0 . 3 0$ </td><td> $0 . 9 2 \pm 0 . 2 9$ </td></tr><tr><td></td><td>PickScore</td><td> $2 1 . 4 5 \pm 0 . 5 3$ </td><td> ${ \bf 2 2 . 6 2 \pm 0 . 3 1 }$ </td><td> $2 2 . 1 0 \pm 0 . 3 8$ </td><td> $2 2 . 4 2 \pm 0 . 3 0$ </td></tr><tr><td></td><td>Compressibility</td><td> $- 8 . 2 6 \pm 0 . 7 7$ </td><td> $- 4 . 6 8 \pm 0 . 5 6$ </td><td> $- 7 . 6 1 \pm 0 . 6 7$ </td><td> $- 6 . 5 9 \pm 0 . 4 9$ </td></tr></table>

The LSI condition can be established through several technical tools (Chewi et al., 2022); for example, if $\pi _ { t }$ satisfies the LSI and the oscillation of $\eta \frac { \delta V ( t , \mu ) } { \delta \mu } ( \cdot )$ is bounded, then the Bakry-Emery and Holley-Stroock arguments yield the LSI condition of $\nu _ { t } ^ { \mu }$ (Bakry & Emery, 1985; Holley & Stroock,<sup>´</sup> 1987). Since $\pi _ { t } \ ( t < 1 )$ is smoothed by a Gaussian, we expect the LSI constant $\alpha _ { t }$ to be larger than that of the target distribution. Under the above assumption, we have the following result.

Theorem 4.5. Assume that R is concave and Assumption 4.4 holds. Then, we have

$$
J _ { \eta } ( \pi _ { 1 } ^ { * } ) - J _ { \eta } ( \pi _ { 1 } ^ { g } ) = O \left( \varepsilon + \frac { \eta } { \lambda ^ { 2 } } \int _ { \varepsilon } ^ { 1 } \frac { C _ { t } ^ { 2 } } { \alpha _ { t } ^ { 2 } } w _ { \lambda } ( t ) \mathrm { d } t \right) .
$$

where $\begin{array} { r } { w _ { \lambda } ( t ) : = \lambda \sigma _ { t } ^ { 2 } \alpha _ { t } / \eta \cdot \exp ( - \frac { \lambda } { \eta } \int _ { t } ^ { 1 } \sigma _ { s } ^ { 2 } \alpha _ { s } \mathrm { d } s ) } \end{array}$ is a weightfunction that satisfies $\begin{array} { r } { \int _ { \varepsilon } ^ { 1 } w _ { \lambda } ( t ) \mathrm { d } t \leq 1 } \end{array}$

See Appendix K for the proof. Thus, if $\textstyle \int _ { \varepsilon } ^ { 1 } C _ { t } ^ { 2 } / \alpha _ { t } ^ { 2 } \cdot w _ { \lambda } ( t )$ dt is finite, by setting λ sufficiently large and ε sufficiently small, we can ensure that the generated distribution $\pi _ { 1 } ^ { g }$ is close to the optimal distribution $\pi _ { 1 } ^ { * }$ . Note that while the assumptions are similar, the proof techniques are different from those of mean-field Langevin dynamics (Nitanda et al., 2022; Chizat, 2022) since the objective functional $\mathcal { V } ( t , \mu )$ is time-dependent and we carefully handle this time-dependency in the proof. In that sense, this convergence theorem can be viewed as a functional generalization of the wellknown convergence analysis for Langevin dynamics (Bakry et al., 2014); specifically, it extends to distribution-dependent (mean-field) and time-dependent objective.

## 5 NUMERICAL EXPERIMENTS

In this section, we evaluate the performance of the proposed steepest guidance method on synthetic and image generation tasks.

## 5.1 TOY EXPERIMENTS

First, we consider a toy experiment to illustrate the differences between Steepest Guidance and Doob’s h-transform with finite lookahead samples. We consider a simple 1D Gaussian mixture model $\textstyle \pi _ { 1 } = { \frac { 1 } { 2 } } { \mathcal { N } } ( x \mid - 3 . 0 , 1 . 0 ) + { \frac { 1 } { 2 } } { \mathcal { N } } ( x \mid 3 . 0 , 1 . 0 )$ as the target distribution and a step reward $r ( y ) = 1 0 \cdot { \bf 1 } \{ y \geq 0 \}$ . That is, the objective is to select the right mode in the positive region.

![](images/45b278109d6ea66a94cddec125a3fad4f151ae92c53ee9e7d5aa7d3f7d8f7769.jpg)  
Figure 3: Trade-off between blueness and image reward for (Regularized) Steepest Guidance.

![](images/8e4135ae6876577d8b9b72f1479e14adcc7432a81badc00160aba4930b17334d.jpg)  
Figure 4: Comparison between Steepest Guidance with and without CVaR. Horizontal lines indicate the lowertail CVaR with α = 0.5.

Fig. 2 illustrates the distributions obtained by each method and the optimal distribution derived analytically. We see that 1) empirical Doob’s guidance sometimes fails to select the right mode while ideally it matches the optimal distribution, 2) steepest guidance successfully selects the right mode but the distribution fails to match the optimal distribution, and 3) regularized steepest guidance successfully approximates the optimal distribution. The results match our theoretical analysis.

## 5.2 IMAGE GENERATION

Next, we evaluate the proposed methods on image generation tasks using Stable Diffusion v1.5 (SD) (Rombach et al., 2022) and FLUX.1 [dev] (FLUX) (Black Forest Labs, 2024). As reward functions, we consider Blueness (how blue an image is), PickScore (Kirstain et al., 2023), ImageReward (Xu et al., 2023), and Compressibility (scaled negative compression size). For SD experiments using ImageReward or PickScore, we set $k = 8 ;$ for all other experiments, we use k = 4. See Appendix L for details of the experimental setup and additional results including results for other prompts, generated images and ablation studies on k.

Steepest Guidance consistently outperforms baselines: Here, we consider expected reward maximization and evaluate the performance of the proposed steepest guidance against baselines including empirical Doob’s transform and SVDD (Li et al., 2024), which is a state-of-the-art selection-based approach. We tune the hyperparameter λ separately for steepest guidance and Doob’s h-transform for each model and reward, using random seeds distinct from those used for final evaluation. Table 1 shows that the steepest guidance outperforms baselines across reward functions and models.

Regularized Steepest Guidance improves the reward–quality trade-off: We compare the performance of regularized steepest guidance and (vanilla) steepest guidance. Since the terminal KL divergence is not directly measurable, we use ImageReward as a proxy to assess the extent to which the guided distribution deviates from the original distribution in terms of prompt alignment. Fig. 3 shows that regularized steepest guidance (orange and purple curves) achieves better trade-offs between Blueness and ImageReward than vanilla steepest guidance (gray dotted curve) by controlling the strength of the regularization parameter η.

Steepest Guidance can handle non-linear rewards: To demonstrate the effectiveness of steepest guidance for non-linear reward functionals, we consider CVaR as a reward functional. We set α = 0.5. Fig. 4 shows the distribution of the rewards obtained by maximizing the expectation (left) and CVaR (right). We see that the proposed method achieves better CVaR (horizontal line) compared to maximizing the expectation.

## 6 CONCLUSION

We proposed steepest guidance, a training-free approach to aligning flow and diffusion models. Different from existing approaches based on Doob’s h-transform, steepest guidance directly optimizes the reward functional in a local manner. As a result, our approach simplifies guidance estimation and accommodates general reward functionals. Theoretically, we established provable improvement guarantees for the ideal dynamics and global convergence under suitable assumptions. The experiments demonstrated the effectiveness of our proposed method in text-to-image generation tasks.

## ACKNOWLEDGMENTS

RH was partially supported by JSPS KAKENHI (24K02905) and JST BOOST (JPMJBS2418). TS was partially supported by JST CREST (PMJCR2015) and JST ERATO (JPMJER2601). This research is supported by the National Research Foundation, Singapore and the Ministry of Digital Development and Information under the AI Visiting Professorship Programme (award number AIVP-2024-004). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the views of National Research Foundation, Singapore and the Ministry of Digital Development and Information.

## REFERENCES

Michael Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. Journal ofMachine Learning Research, 26(209):1–80, 2025.

D. Bakry and M. Emery. Diffusions hypercontractives. In Jacques Az <sup>´</sup> ema and Marc Yor (eds.), ´ Seminaire de Probabilit´ es XIX 1983/84´ , pp. 177–206, Berlin, Heidelberg, 1985. Springer Berlin Heidelberg. ISBN 978-3-540-39397-9.

Dominique Bakry, Ivan Gentil, and Michel Ledoux. Analysis and geometry of Markov diffusion operators. Springer, 2014.

Andreas Bergmeister, Stefanie Jegelka, Nikolas Nusken, Carles Domingo-Enrich, and Jakiw Pid-¨ strigach. Reinforce adjoint matching: Scaling RL post-training of diffusion and flow-matching models. arXiv preprint arXiv:2605.10759, 2026.

Black Forest Labs. FLUX. https://github.com/black-forest-labs/flux, 2024.

Zander W Blasingame and Chen Liu. Greed is good: A unifying perspective on guided generation. arXiv preprint arXiv:2502.08006, 2025.

Sinho Chewi, Murat A Erdogdu, Mufan Li, Ruoqi Shen, and Shunshi Zhang. Analysis of Langevin monte carlo from Poincare to Log-Sobolev. In Po-Ling Loh and Maxim Raginsky (eds.), Proceedings ofThirty Fifth Conference on Learning Theory, volume 178 of Proceedings ofMachine Learning Research, pp. 1–2. PMLR, 02–05 Jul 2022.

Lena´ ¨ıc Chizat. Mean-field langevin dynamics : Exponential convergence and annealing. Transactions on Machine Learning Research, 2022. ISSN 2835-8856.

G Corso, Y Xu, VD Bortoli, R Barzilay, and TS Jaakkola. Particle guidance: non-IID diverse sampling with diffusion models. arXiv preprint arXiv:2310.13102, 2023.

Sanjit Dandapanthula and Nicholas M Boffi. Are we really tilting? the mechanics of reward guid ance in flow and diffusion models. arXiv preprint arXiv:2606.02884, 2026.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat GANs on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Carles Domingo-Enrich, Michal Drozdzal, Brian Karrer, and Ricky TQ Chen. Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. arXiv preprint arXiv:2409.08861, 2024.

Yingqing Guo, Hui Yuan, Yukang Yang, Minshuo Chen, and Mengdi Wang. Gradient guidance for diffusion models: An optimization perspective. Advances in Neural Information Processing Systems, 37:90736–90770, 2024.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Peter Holderrieth, Uriel Singer, Tommi Jaakkola, Ricky TQ Chen, Yaron Lipman, and Brian Karrer. GLASS flows: Transition sampling for alignment of flow and diffusion models. arXiv preprint arXiv:2509.25170, 2025.

Peter Holderrieth, Douglas Chen, Luca Eyring, Ishin Shah, Giri Anantharaman, Yutong He, Zeynep Akata, Tommi Jaakkola, Nicholas Matthew Boffi, and Max Simchowitz. Diamond maps: Efficient reward alignment via stochastic flow maps. arXiv preprint arXiv:2602.05993, 2026.

Richard Holley and Daniel Stroock. Logarithmic sobolev inequalities and stochastic ising models. Journal of statistical physics, 46(5-6):1159–1194, 1987.

Emiel Hoogeboom, Vıctor Garcia Satorras, Clement Vignac, and Max Welling. Equivariant dif-´ fusion for molecule generation in 3D. In International conference on machine learning, pp. 8867–8887. PMLR, 2022.

Yuchen Jiao, Yuxin Chen, and Gen Li. Towards a unified framework for guided diffusion models. arXiv preprint arXiv:2512.04985, 2025.

Ryotaro Kawata, Kazusato Oko, Atsushi Nitanda, and Taiji Suzuki. Direct distributional optimization for provable alignment of diffusion models. In International Conference on Learning Representations, volume 2025, pp. 72792–72840, 2025.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-Pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652–36663, 2023.

Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-LM improves controllable text generation. Advances in neural information processing systems, 35:4328–4343, 2022.

Xiner Li, Yulai Zhao, Chenyu Wang, Gabriele Scalia, Gokcen Eraslan, Surag Nair, Tommaso Biancalani, Shuiwang Ji, Aviv Regev, Sergey Levine, et al. Derivative-free guidance in continuous and discrete diffusion models with soft value-based decoding. arXiv preprint arXiv:2408.08252, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.

Pierre Marion, Anna Korba, Peter Bartlett, Mathieu Blondel, Valentin De Bortoli, Arnaud Doucet, Felipe Llinares-Lopez, Courtney Paquette, and Quentin Berthet. Implicit diffusion: Efficient´ optimization through stochastic sampling. arXiv preprint arXiv:2402.05468, 2024.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. WebGPT: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

Atsushi Nitanda, Denny Wu, and Taiji Suzuki. Convex analysis of the mean field langevin dynamics. In International Conference on Artificial Intelligence and Statistics, pp. 9741–9757. PMLR, 2022.

Atsushi Nitanda, Dake Bu, Yueming Lyu, and Tanya Veeravalli. Slowly annealed langevin dynamics: Theory and applications to training-free guided generation. arXiv preprint arXiv:2605.07950, 2026.

L Chris G Rogers and David Williams. Diffusions, Markov processes, and martingales, volume 2. Cambridge university press, 2000.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695, June 2022.

Riccardo De Santi, Marin Vlastelica, Ya-Ping Hsieh, Zebang Shen, Niao He, and Andreas Krause. Flow density control: Generative optimization beyond entropy-regularized fine-tuning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Shokichi Takakura, Akifumi Wachi, Rei Higuchi, Kohei Miyaguchi, and Taiji Suzuki. Inferenceaware meta-alignment of LLMs via non-linear GRPO. arXiv preprint arXiv:2602.01603, 2026.

Masatoshi Uehara, Yulai Zhao, Kevin Black, Ehsan Hajiramezanali, Gabriele Scalia, Nathaniel Lee Diamant, Alex M Tseng, Tommaso Biancalani, and Sergey Levine. Fine-tuning of continuoustime diffusion models as entropy-regularized control. arXiv preprint arXiv:2402.15194, 2024.

Gal Vinograd, Idan Achituve, and Ethan Fetaya. Diverse sampling in diffusion models with marginal preserving particle guidance. arXiv preprint arXiv:2605.06553, 2026.

Zifan Wang, Riccardo De Santi, Xiaoyu Mo, Michael M Zavlanos, Andreas Krause, and Karl H Johansson. Efficient tail-aware generative optimization via flow model fine-tuning. arXiv preprint arXiv:2602.16796, 2026.

Max Welling and Yee W Teh. Bayesian learning via stochastic gradient langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), pp. 681–688, 2011.

Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3):229–256, 1992.

Luhuan Wu, Brian Trippe, Christian Naesseth, David Blei, and John P Cunningham. Practical and asymptotically exact conditional sampling in diffusion models. Advances in Neural Information Processing Systems, 36:31372–31403, 2023.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. ImageReward: Learning and evaluating human preferences for text-to-image generation, 2023.

Junyu Zhang, Alec Koppel, Amrit Singh Bedi, Csaba Szepesvari, and Mengdi Wang. Variational policy gradient method for reinforcement learning with general utilities. Advances in Neural Information Processing Systems, 33:4572–4583, 2020.

Qijie Zhu, Zeqi Ye, Han Liu, Zhaoran Wang, and Minshuo Chen. Training-free adaptation of diffusion models via Doob’s h-transform. arXiv preprint arXiv:2602.16198, 2026.

Nicolas Zilberstein, Morteza Mardani, and Santiago Segarra. Repulsive latent score distillation for solving inverse problems. arXiv preprint arXiv:2406.16683, 2024.

## A RELATED WORK

In this section, we provide a detailed discussion on related work and compare our method with previous works on reward-guided generation.

## A.1 SYSTEMATIC COMPARISON WITH PREVIOUS WORKS

Here, we show in Table 2 a systematic comparison of our method with previous works on rewardguided generation from the perspective of three key properties: training-free, derivative-free, and the ability to handle non-linear reward functionals.

## A.2 OTHER RELATED WORKS

Diversity-seeking Generation A line of work (Corso et al., 2023; Vinograd et al., 2026; Zilberstein et al., 2024) proposes diversity-seeking generation methods utilizing repulsive forces between particles. Such diversity-seeking generation methods can be interpreted as an optimization of nonlinear reward functionals but they are not applicable to general reward functionals.

Optimization of Non-linear Reward Functionals The optimization of general functionals over probability measures is studied in various contexts, including training of two-layer neural networks (Nitanda et al., 2022; Chizat, 2022), reinforcement learning (Zhang et al., 2020), and inference-aware training of LLMs (Takakura et al., 2026).

Table 2: Systematic comparison with previous works on reward-guided generation. (1) Trainingfree: The method does not require additional training of models. (2) Derivative-free: The method does not require gradients of the reward functional or the generative process. (3) Non-linear reward functional: The method can handle non-linear reward functionals.
<table><tr><td>Method</td><td>(1)</td><td>(2)</td><td>(3)</td></tr><tr><td>Steepest Guidance (Ours)</td><td>√</td><td>√</td><td>√</td></tr><tr><td>DOIT (Zhu et al., 2026)</td><td>√</td><td>V</td><td>X</td></tr><tr><td>FDC (Santi et al., 2025)</td><td>X</td><td>X</td><td>√</td></tr><tr><td>Adjoint Matching (Domingo-Enrich et al., 2024)</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Reinforce Adjoint Matching (Bergmeister et al., 2026)</td><td>×</td><td>√</td><td>×</td></tr><tr><td>ŠALD (Nitanda et al., 2026)</td><td>√</td><td>√</td><td>X</td></tr><tr><td>Greedy Guidance (Blasingame &amp; Liu, 2025)</td><td>√</td><td>X</td><td>X</td></tr><tr><td>Gradient Guidance (Guo et al., 2024)</td><td>√</td><td>×</td><td>X</td></tr></table>

Relation to SALD SALD (Nitanda et al., 2026) employs dynamics similar to ours. Indeed, for linear reward functionals, greedy guidance can be regarded as a special case of SALD since SALD considers general guidance which satisfies $g _ { 1 } ( y ) = \nabla r ( y )$ . However, our approach is completely different in its design principles. First of all, SALD is designed for sampling from tilted distributions, which is the optimal solution to KL-regularized expected reward maximization. Therefore, it cannot be applied to general non-linear reward functionals. On the other hand, greedy guidance is designed from the perspective of optimization of probability measures and can handle general reward functionals. Furthermore, SALD’s guidance does not generally align with a direction that improves the expected reward. Consequently, its improvement guarantee relies on sufficiently slow annealing. In practice, such slowdown can significantly affect the efficiency of generation.

## B EXAMPLES OF REWARD FUNCTIONALS

In this section, we provide representative examples of (non-linear) reward functionals for various applications. Table 3 summarizes representative reward functionals. We mainly follow Santi et al. (2025) and have added several reward functionals that are not included in their work.

For entropy maximization, we cannot directly compute the first-order variation of the entropy functional since it requires knowledge of the density $\mu ( x )$ . Thus, we propose to use the following relation instead:

$$
\operatorname { E n t } ( \mu ) = - \mathrm { K L } ( \mu \mid \pi _ { 1 } ) - \mathbb { E } _ { x \sim \mu } \left[ \log \pi _ { 1 } ( x ) \right] .
$$

Utilizing the above relation, entropy maximization can be interpreted as KL-regularized expected reward maximization. In general, we cannot compute log $\pi _ { 1 } ( x )$ but we can use the score function $\nabla \log \pi _ { 1 } ( x )$ . Thus, if the sampling process is differentiable, we can estimate the guidance using a plug-in estimator and the chain rule.

## B.1 FIRST-ORDER VARIATION OF REWARD FUNCTIONALS

Here, we provide the first-order variation of CVaR and Rao’s quadratic entropy, which are used in our experiments.

Lemma B.1 (First-order variation of CVaR). Let $R _ { \mathrm { C V a R } } [ \mu ] = \mathbb { E } _ { \mu } \left[ r ( Y ) \mid r ( Y ) \leq q _ { \alpha } \right]$ be the CVaR functional, where $q _ { \alpha }$ is the α-quantile of $r ( Y )$ . Assume that µ has a strictly positive density on R<sup>d</sup> and that the law of $r ( Y )$ under µ has a continuous density $f _ { \mu }$ in a neighborhood of $q _ { \alpha }$ , with $f _ { \mu } ( q _ { \alpha } ) > 0$ . Then, thefirst-order variation of $R _ { \mathrm { C V a R } }$ is given by

$$
{ \frac { \delta R _ { \mathrm { C V a R } } } { \delta \mu } } [ \mu ] ( y ) = { \frac { 1 } { \alpha } } \cdot \operatorname* { m i n } \{ r ( y ) - q _ { \alpha } , 0 \} .
$$

Proof. For $\nu \in \mathcal { P }$ , let $\mu _ { \epsilon } = ( 1 - \epsilon ) \mu + \epsilon \nu ,$ and denote the reward distribution function under a measure ρ by $F _ { \rho } .$ . Write $q = q _ { \alpha }$ and define $q _ { \epsilon } = \operatorname* { i n f } \{ z : F _ { \mu _ { \epsilon } } ( z ) \geq \alpha \}$

Table 3: Representative reward functionals for various applications.
<table><tr><td>Application</td><td>Functional  $R [ \mu ]$ </td></tr><tr><td>Expected reward</td><td> $\mathbb { E } _ { x \sim \mu } \left[ r ( x ) \right]$ </td></tr><tr><td>KL divergence</td><td> $\begin{array} { r } { D _ { \mathrm { K L } } ( \mu \| \pi _ { 1 } ) : = \int \mu ( x ) \log \frac { \mu ( x ) } { \pi _ { 1 } ( x ) } \ \mathrm { d } x } \end{array}$ </td></tr><tr><td>Conditional value-at-risk</td><td> $\mathbb { E } _ { x \sim \mu } \left[ r ( x ) \mid r ( x ) \le q _ { \alpha } \right]$ </td></tr><tr><td>Variance</td><td> $\mathbb { E } _ { \boldsymbol { x } \sim \boldsymbol { \mu } } \left[ ( r ( \boldsymbol { x } ) - \mathbb { E } _ { \boldsymbol { x } \sim \boldsymbol { \mu } } \left[ r ( \boldsymbol { x } ) \right] ) ^ { 2 } \right]$ </td></tr><tr><td>Entropy</td><td> $\mathrm { E n t } [ \mu ] : = - \mathbb { E } _ { x \sim \mu } \left[ \log \mu ( x ) \right]$ </td></tr><tr><td>Optimal experimental design</td><td> $\mathsf { s } \bigl ( \mathbb { E } _ { \boldsymbol { x } \sim \mu } \left[ \Phi ( \boldsymbol { x } ) \Phi ( \boldsymbol { x } ) ^ { \top } - \lambda I \right] \bigr )$ </td></tr><tr><td>Log-barrier</td><td> $- \beta \log ( \mathbb { E } _ { x \sim \mu } \left[ c ( x ) \right] - C )$ </td></tr><tr><td>Maximum mean discrepancy</td><td> $\mathrm { M M D } _ { k } ( \mu \| \pi _ { 1 } ) : = \| m _ { \mu } - m _ { \pi _ { 1 } } \| , \ m _ { \mu } : = \mathbb { E } _ { x \sim \mu } \left[ k ( x , \cdot ) \right]$ </td></tr><tr><td>Best- or worst-case reward</td><td> $\mathbb { E } _ { x _ { 1 } , \ldots , x _ { N } \sim \mu ^ { N } } \left[ \operatorname* { m a x } _ { i } r ( x _ { i } ) \right] , \mathbb { E } _ { x _ { 1 } \ldots , x _ { N } \sim \mu ^ { N } } \left[ \operatorname* { m i n } _ { i } r ( x _ { i } ) \right]$ </td></tr></table>

First, we prove the differentiability of $q _ { \epsilon }$ with respect to ϵ. Since

$$
F _ { \mu _ { \epsilon } } ( q _ { \epsilon } ) = ( 1 - \epsilon ) F _ { \mu } ( q _ { \epsilon } ) + \epsilon F _ { \nu } ( q _ { \epsilon } ) = \alpha ,\tag{8}
$$

we have $| F _ { \mu } ( q _ { \epsilon } ) - \alpha | \le \epsilon .$ For sufficiently small $\epsilon ,$ we have

$$
\frac { f _ { \mu } ( q ) } { 2 } | q _ { \epsilon } - q | \le | F _ { \mu } ( q _ { \epsilon } ) - F _ { \mu } ( q ) | \le \epsilon ,
$$

where the last inequality follows from $| F _ { \mu } ( q _ { \epsilon } ) - \alpha | \le \epsilon .$ Since $f _ { \mu } ( q ) > 0$ , we obtain $q _ { \epsilon } - q = O ( \epsilon )$ which establishes the differentiability of $q _ { \epsilon }$ with respect to ϵ at $\epsilon = 0 .$ . By differentiating Eq. (8) with respect to $\epsilon ,$ we obtain

$$
f _ { \mu _ { \epsilon } } ( q _ { \epsilon } ) q _ { \epsilon } ^ { \prime } = F _ { \mu } ( q _ { \epsilon } ) - F _ { \nu } ( q _ { \epsilon } ) ,
$$

where $\begin{array} { r } { q _ { \epsilon } ^ { \prime } = \frac { \mathrm { d } q _ { \epsilon } } { \mathrm { d } \epsilon } } \end{array}$

Let

$$
H _ { \rho } ( a ) : = \int r ( y ) { \bf 1 } \{ r ( y ) \leq a \} \rho ( \mathrm { d } y ) .
$$

Note that $R _ { \mathrm { C V a R } } [ \rho ] = \alpha ^ { - 1 } H _ { \rho } ( q _ { \alpha } )$ . Then, we have

$$
\begin{array} { l } { { \displaystyle R _ { \mathrm { C V a R } } [ \mu _ { \epsilon } ] - R _ { \mathrm { C V a R } } [ \mu ] = \frac { 1 } { \alpha } \left( H _ { \mu _ { \epsilon } } ( q _ { \epsilon } ) - H _ { \mu } ( q ) \right) } } \\ { { \displaystyle \qquad = \frac { 1 } { \alpha } \left( \underbrace { H _ { \mu _ { \epsilon } } ( q _ { \epsilon } ) - H _ { \mu } ( q _ { \epsilon } ) } _ { = : D _ { 1 } } + \underbrace { H _ { \mu } ( q _ { \epsilon } ) - H _ { \mu } ( q ) } _ { = : D _ { 2 } } \right) . } } \end{array}
$$

For $D _ { 1 }$ , we have

$$
D _ { 1 } = H _ { \mu _ { \epsilon } } ( q _ { \epsilon } ) - H _ { \mu } ( q _ { \epsilon } ) = \epsilon \int r ( y ) { \bf 1 } \{ r ( y ) \leq q _ { \epsilon } \} ( \nu - \mu ) ( { \bf d } y )
$$

and

$$
\operatorname* { l i m } _ { \epsilon \to 0 } \frac { D _ { 1 } } { \epsilon } = \int r ( y ) { \bf 1 } \{ r ( y ) \leq q \} ( \nu - \mu ) ( \mathrm { d } y ) .
$$

For $D _ { 2 } .$ , we have

$$
D _ { 2 } = H _ { \mu } ( q _ { \epsilon } ) - H _ { \mu } ( q ) = q f _ { \mu } ( q ) ( q _ { \epsilon } - q ) + o ( | q _ { \epsilon } - q | ) ,
$$

and

$$
\operatorname* { l i m } _ { \epsilon \to 0 } \frac { D _ { 2 } } { \epsilon } = q f _ { \mu } ( q ) q _ { 0 } ^ { \prime } = q ( F _ { \mu } ( q ) - F _ { \nu } ( q ) ) = \int q { \bf 1 } \{ r ( y ) \leq q \} ( \mu - \nu ) ( { \bf d } y ) .
$$

Combining the above results, we obtain

$$
\begin{array} { l } { \displaystyle \operatorname* { l i m } _ { \epsilon \to 0 } \frac { R _ { \mathrm { C V a R } } [ \mu _ { \epsilon } ] - R _ { \mathrm { C V a R } } [ \mu ] } { \epsilon } = \frac { 1 } { \alpha } \left( \operatorname* { l i m } _ { \epsilon \to 0 } \frac { D _ { 1 } } { \epsilon } + \operatorname* { l i m } _ { \epsilon \to 0 } \frac { D _ { 2 } } { \epsilon } \right) } \\ { \displaystyle \quad \quad = \frac { 1 } { \alpha } \int ( r ( y ) - q ) \mathbf { 1 } \{ r ( y ) \le q \} ( \nu - \mu ) ( \mathrm { d } y ) . } \end{array}
$$

This implies that the first-order variation of $R _ { \mathrm { C V a R } }$ with respect to $\mu$ is

$$
\frac { \delta R _ { \mathrm { C V a R } } } { \delta \mu } [ \mu ] ( y ) = \frac { 1 } { \alpha } \operatorname* { m i n } \{ r ( y ) - q , 0 \} .
$$

Lemma B.2 (First-order variation of Rao’s quadratic entropy). $\begin{array} { r l r l } { L e t \_ }  & { { } R _ { \mathrm { R a o } } [ \mu ] } & { } & { { } = } \end{array}$ $\begin{array} { r } { - \frac { 1 } { 2 } \int \int k ( y , y ^ { \prime } ) \dot { \mu ( \mathrm { d } y ) } \mu ( \mathrm { d } y ^ { \prime } ) } \end{array}$ be Rao’s quadratic entropy functional, where $k : \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } \overset { \vartriangle } { \vline } $ R is a symmetric positive definite kernel. Then, the first-order variation of $R _ { \mathrm { R a o } }$ is given by

$$
\frac { \delta R _ { \mathrm { R a o } } } { \delta \mu } [ \mu ] ( y ) = - \int k ( y , y ^ { \prime } ) \mu ( \mathrm { d } y ^ { \prime } ) .
$$

Proof. For any $\nu \in \mathcal { P } ( \mathbb { R } ^ { d } )$ , let $h = \nu - \mu .$ . By the symmetry of k, we have

$$
R _ { \mathrm { R a o } } [ \nu ] - R _ { \mathrm { R a o } } [ \mu ] = - \int \int k ( y , y ^ { \prime } ) h ( \mathrm { d } y ) \mu ( \mathrm { d } y ^ { \prime } ) - \frac { 1 } { 2 } \int \int k ( y , y ^ { \prime } ) h ( \mathrm { d } y ) h ( \mathrm { d } y ^ { \prime } ) .
$$

The first term is linear in h, while the second term is quadratic. Therefore,

$$
\frac { \delta R _ { \mathrm { R a o } } } { \delta \mu } [ \mu ] ( y ) = - \int k ( y , y ^ { \prime } ) \mu ( \mathrm { d } y ^ { \prime } ) .
$$

## C DETAILED ALGORITHM OF STEEPEST GUIDANCE

```latex
Algorithm 1 Steepest Guidance for Reward-Guided Generation
1: Choose a time grid $\varepsilon = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { L } = 1$ and set $\Delta t _ { l } = t _ { l + 1 } - t _ { l }$
2: Sample M particles $\{ y _ { 0 } ^ { ( j ) } \} _ { j = 1 } ^ { M }$ from $\pi _ { \varepsilon }$ by evolving the unguided base process from its native
prior to time ε.
3: for $l = 0 , \ldots , L - 1$ do
4: Sample $y _ { l , i } ^ { ( j ) }$ from $\pi _ { 1 } ( \cdot \mid Y _ { t _ { l } } = y _ { l } ^ { ( j ) } )$ for $j = 1 , \dots , M$ and $i = 1 , \ldots , k .$
5: Construct the empirical terminal measure $\begin{array} { r } { \hat { \mu } _ { 1 , l } = \frac { 1 } { M k } \sum _ { j = 1 } ^ { M } \sum _ { i = 1 } ^ { k } \delta _ { y _ { l , i } ^ { ( j ) } } } \end{array}$
6: Compute the plug-in approximations $\begin{array} { r } { \frac { \delta R } { \delta \mu } [ \hat { \mu } _ { 1 , l } ] ( y _ { l , i } ^ { ( j ) } ) } \end{array}$ for all i and j, using the abuse of notation
introduced in the main text.
7: Estimate the steepest guidance for each $j = 1 , \dots , M \colon$
$\hat { g } _ { t _ { l } } ^ { \mathrm { s t e e p e s t } } ( y _ { l } ^ { ( j ) } ) = \lambda \sigma _ { t _ { l } } ^ { 2 } \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \frac { \delta R } { \delta \mu } [ \hat { \mu } _ { 1 , l } ] ( y _ { l , i } ^ { ( j ) } ) \nabla _ { y _ { l } ^ { ( j ) } } \log \pi _ { 1 } ( y _ { l , i } ^ { ( j ) } \mid Y _ { t _ { l } } = y _ { l } ^ { ( j ) } ) .$
8: Update the particles:
$y _ { l + 1 } ^ { ( j ) } = y _ { l } ^ { ( j ) } + \left( b _ { t _ { l } } ( y _ { l } ^ { ( j ) } ) + \hat { g } _ { t _ { l } } ^ { \mathrm { s t e e p e s t } } ( y _ { l } ^ { ( j ) } ) \right) \Delta t _ { l } + \sigma _ { t _ { l } } \sqrt { \Delta t _ { l } } \xi _ { l } ^ { ( j ) } , \quad \xi _ { l } ^ { ( j ) } \sim \mathcal { N } ( 0 , I ) .$
9: end for
10: Return: $\{ y _ { L } ^ { ( j ) } \} _ { j = 1 } ^ { M }$ as the generated samples.
```

## D AUXILIARY LEMMAS

Lemma D.1 (Corollary 18 in (Albergo et al., 2025)). Let $Y _ { t }$ be the solution of the SDE

$$
\mathrm { d } Y _ { t } = b _ { t } ( Y _ { t } ) \mathrm { d } t + \sigma _ { t } \mathrm { d } W _ { t } ,
$$

and let $\tilde { Y } _ { t }$ be the solution ofthe SDE

$$
\mathrm { d } \tilde { Y } _ { t } = \big ( b _ { t } \big ( \tilde { Y } _ { t } \big ) + g _ { t } \big ( \tilde { Y } _ { t } \big ) \big ) \mathrm { d } t + \sqrt { 1 + 2 \lambda } \cdot \sigma _ { t } \mathrm { d } W _ { t } ,
$$

where $g _ { t } ( y ) ~ = ~ \lambda \sigma _ { t } ^ { 2 } \nabla _ { y } \log \pi _ { t } ( y )$ . Then the marginal distribution $\tilde { \pi } _ { t }$ of $\tilde { Y } _ { t }$ coincides with the marginal distribution π<sub>t</sub> ofY<sub>t</sub> for all $t \in [ 0 , 1 ] , i f \tilde { \pi } _ { 0 } = \pi _ { 0 }$

Lemma D.2 (Score Function of Flow Models). For a flow model, the score function ∇ log $\pi _ { t } ( y )$ can be expressed as

$$
\nabla \log \pi _ { t } ( y ) = \frac { t v _ { t } ( y ) - y } { 1 - t }
$$

where $v _ { t } ( y )$ is the velocityfield oftheflow model.

Proof. From the definition of the flow model, we have $Y _ { t } = t Y _ { 1 } + ( 1 - t ) Y _ { 0 }$ , where $Y _ { 0 } \sim { \mathcal { N } } ( 0 , I )$ and $Y _ { 1 } \sim \pi _ { 1 }$ . Therefore, we have $\begin{array} { r } { \frac { Y _ { t } - t Y _ { 1 } } { 1 - t } = Y _ { 0 } \sim \mathcal { N } ( 0 , I ) } \end{array}$ and

$$
\begin{array} { l } { \displaystyle \pi _ { t } ( y ) = \int \pi _ { t } ( y \mid Y _ { 1 } = z ) \pi _ { 1 } ( z ) \mathrm { d } z } \\ { \displaystyle \qquad = \int C \cdot \exp \left( - \frac { 1 } { 2 } \bigg \| \frac { y - t z } { 1 - t } \bigg \| ^ { 2 } \right) \pi _ { 1 } ( z ) \mathrm { d } z , } \end{array}
$$

where $C$ is a normalization constant. Differentiating $\pi _ { t } ( y )$ with respect to $y ,$ we have

$$
\begin{array} { l } { \displaystyle \nabla \pi _ { t } ( y ) = \int C \cdot \exp \left( - \frac { 1 } { 2 } \Bigg \| \frac { y - t z } { 1 - t } \Bigg \| ^ { 2 } \right) \cdot \frac { t z - y } { ( 1 - t ) ^ { 2 } } \pi _ { 1 } ( z ) \mathrm { d } z } \\ { \displaystyle \qquad = \int \pi _ { t } ( y \mid Y _ { 1 } = z ) \cdot \frac { t z - y } { ( 1 - t ) ^ { 2 } } \pi _ { 1 } ( z ) \mathrm { d } z } \\ { \displaystyle \qquad = \int \pi _ { 1 } ( z \mid Y _ { t } = y ) \cdot \frac { t z - y } { ( 1 - t ) ^ { 2 } } \pi _ { t } ( y ) \mathrm { d } z } \\ { \displaystyle \qquad = \pi _ { t } ( y ) \cdot \mathbb { E } \left[ \frac { t Y _ { 1 } - Y _ { t } } { ( 1 - t ) ^ { 2 } } \mid Y _ { t } = y \right] . } \end{array}
$$

Thus, the score function is given by

$$
\begin{array} { r l } { \nabla \log \pi _ { 4 } ( y ) = \frac { \nabla \pi _ { 4 } ( y ) } { \pi _ { 4 } ( y ) } } \\ & { = \mathbb { E } \left[ \frac { l ( Y _ { 1 } - Y _ { 2 } ) } { ( 1 - l ) ^ { 2 } } | Y _ { i } - y | \right] } \\ & { = \mathbb { E } \left[ \frac { l ( 1 - l ) ( Y _ { 1 } - Y _ { 2 } ) + \frac { \sqrt { \pi _ { 4 } } } { ( 1 - l ) ^ { 2 } \sqrt { \pi _ { 4 } } } | Y _ { 0 } | - Y _ { 1 } } { ( 1 - l ) ^ { 2 } } | Y _ { i } - y | \right] } \\ & { = \mathbb { E } \left[ \frac { l ( 1 - l ) ( Y _ { 1 } - Y _ { 0 } ) - ( 1 - l ) Y _ { 2 } } { ( 1 - l ) ^ { 2 } } | Y _ { i } - y | \right] } \\ & { = \mathbb { E } \left[ \frac { l ( 1 - l ) ( Y _ { 1 } - Y _ { 0 } ) - ( 1 - l ) Y _ { 2 } } { ( 1 - l ) ^ { 2 } } | Y _ { i } - y | \right] } \\ & { = \mathbb { E } \left[ \frac { l ( Y _ { 1 } - Y _ { 0 } ) - Y _ { 1 } } { ( 1 - l ) } | Y _ { i } - y | \right] } \\ & { = \frac { \mathbb { E } \left[ l ( Y _ { 1 } , ( y ) - y ) \right] } { ( 1 - l ) } , } \end{array}
$$

where the last equality follows from the definition of the velocity field $v _ { t } ( y ) = \mathbb { E } [ Y _ { 1 } - Y _ { 0 } \mid Y _ { t } = y ] .$

Lemma D.3 (Conditional Distribution of Memoryless Flow). Assume that the flow model uses the memoryless noise schedule

$$
\sigma _ { t } ^ { 2 } = \frac { 2 ( 1 - t ) } { t } .
$$

Then, for any $t \in [ \varepsilon , 1 )$ ,

$$
\begin{array} { r } { Y _ { t } \mid Y _ { 1 } = z \sim \mathcal { N } \big ( t z , ( 1 - t ) ^ { 2 } I \big ) . } \end{array}
$$

Proof. The time reversal of Eq. (1) is given by

$$
\mathrm { d } X _ { t } = \left( - v _ { 1 - t } ( X _ { t } ) + \frac { \sigma _ { 1 - t } ^ { 2 } } { 2 } \nabla \log \pi _ { 1 - t } ( X _ { t } ) \right) \mathrm { d } t + \sigma _ { 1 - t } \mathrm { d } W _ { t } , \qquad X _ { 0 } \sim \pi _ { 1 } .
$$

Since $\sigma _ { 1 - t } ^ { 2 } = 2 t / ( 1 - t )$ , Lemma D.2 yields

$$
\begin{array} { l } { \displaystyle \mathrm { d } X _ { t } = \left( - v _ { 1 - t } ( X _ { t } ) + \frac { t } { 1 - t } \cdot \frac { ( 1 - t ) v _ { 1 - t } ( X _ { t } ) - X _ { t } } { t } \right) \mathrm { d } t + \sqrt { \frac { 2 t } { 1 - t } } \mathrm { d } W _ { t } } \\ { \displaystyle \quad = - \frac { X _ { t } } { 1 - t } \mathrm { d } t + \sqrt { \frac { 2 t } { 1 - t } } \mathrm { d } W _ { t } . } \end{array}
$$

To solve this linear SDE, define $Z _ { s } : = X _ { s } / ( 1 - s )$ . Then, we have

$$
\begin{array} { l } { \displaystyle \mathrm { d } Z _ { s } = \frac { 1 } { 1 - s } \mathrm { d } X _ { s } + \frac { X _ { s } } { ( 1 - s ) ^ { 2 } } \mathrm { d } s } \\ { = \displaystyle \frac { 1 } { 1 - s } \left( - \frac { X _ { s } } { 1 - s } \mathrm { d } s + \sqrt { \frac { 2 s } { 1 - s } } \mathrm { d } W _ { s } \right) + \frac { X _ { s } } { ( 1 - s ) ^ { 2 } } \mathrm { d } s } \\ { = \displaystyle \sqrt { \frac { 2 s } { ( 1 - s ) ^ { 3 } } } \mathrm { d } W _ { s } . } \end{array}
$$

Conditional on $X _ { 0 } = z ,$ , we have $Z _ { 0 } = z$ . Integrating the above equation from 0 to s and multiplying by 1 − s gives

$$
X _ { s } = ( 1 - s ) z + ( 1 - s ) \int _ { 0 } ^ { s } \sqrt { \frac { 2 r } { ( 1 - r ) ^ { 3 } } } \mathrm { d } W _ { r } .
$$

The stochastic integral is centered Gaussian with covariance

$$
( 1 - s ) ^ { 2 } \int _ { 0 } ^ { s } \frac { 2 r } { ( 1 - r ) ^ { 3 } } \mathrm { d } r \cdot I = s ^ { 2 } I .
$$

Hence,

$$
\begin{array} { r } { X _ { s } \mid X _ { 0 } = z \sim \mathcal { N } \big ( ( 1 - s ) z , s ^ { 2 } I \big ) . } \end{array}
$$

Under the time-reversal coupling, $X _ { s } = Y _ { 1 - s }$ and $X _ { 0 } = Y _ { 1 }$ . Taking $s = 1 - t$ proves

$$
\begin{array} { r } { Y _ { t } \mid Y _ { 1 } = z \sim \mathcal { N } \big ( t z , ( 1 - t ) ^ { 2 } I \big ) . } \end{array}
$$

Lemma D.4 (Kolmogorov Backward Equation). Let $K _ { t }$ be the transition kernel of the base SDE from time t to time 1, and define $V ( t , \mu ) : = R [ K _ { t } \mu ]$ . Then, for any $t \in [ \varepsilon , 1 )$ and $\mu \in \mathcal P ( \mathcal V )$ ),

$$
\partial _ { t } V ( t , \mu ) + \mathfrak { L } _ { t } V ( t , \mu ) = 0 ,
$$

where

$$
\mathfrak { L } _ { t } V ( t , \mu ) : = \mathbb { E } _ { \mu } \left[ \mathcal { L } _ { t } \frac { \delta V ( t , \mu ) } { \delta \mu } ( y ) \right] ,
$$

$$
\mathcal { L } _ { t } f ( y ) : = b _ { t } ( y ) \cdot \nabla f ( y ) + \frac { \sigma _ { t } ^ { 2 } } { 2 } \Delta f ( y ) .
$$

Proof. The adjoint of $\mathcal { L } _ { t }$ is defined as

$$
\mathcal { L } _ { t } ^ { \ast } \rho ( y ) : = - \operatorname { d i v } ( b _ { t } ( y ) \rho ( y ) ) + \frac { \sigma _ { t } ^ { 2 } } { 2 } \Delta \rho ( y ) .
$$

Fix $t \in [ \varepsilon , 1 )$ and $\mu \in \mathcal P ( \mathcal V )$ , and let $( \mu _ { s } ) _ { s \in [ t , 1 ] }$ be the marginals of the base SDE with $\mu _ { t } ~ =$ $\mu . ~ \mathrm { B y }$ the Chapman–Kolmogorov property, $\dot { V ( t , \mu ) } = R [ \mu _ { 1 } ] = V ( s , \mu _ { s } )$ for every $s \in \ \lvert t , 1 \rvert$ Differentiating this identity with respect to s at $s = t ,$ and using the Fokker–Planck equation $\partial _ { s } \mu _ { s } =$ $\mathcal { L } _ { s } ^ { * } \mu _ { s }$ , gives

$$
\begin{array} { l } { 0 = \partial _ { t } V ( t , \mu ) + \displaystyle \int \frac { \delta V ( t , \mu ) } { \delta \mu } ( y ) \mathcal { L } _ { t } ^ { \ast } \mu ( \mathrm { d } y ) } \\ { = \partial _ { t } V ( t , \mu ) + \mathbb { E } _ { \mu } \left[ \mathcal { L } _ { t } \frac { \delta V ( t , \mu ) } { \delta \mu } ( y ) \right] } \\ { = \partial _ { t } V ( t , \mu ) + \mathfrak { L } _ { t } V ( t , \mu ) , } \end{array}
$$

where the second equality follows from integration by parts. This proves the claim.

Lemma D.5 (Relative Entropy Dissipation Formula). Let $\mu _ { \tau }$ and $\pi _ { \tau }$ be the marginal distributions $o f Y _ { \tau }$ at time $\tau \in [ \varepsilon , 1 ]$ . Assume that $\mu _ { t }$ and $\pi _ { t } f o r t \in [ \tau , 1 ]$ are defined by the marginal distributions of the SDE:

$$
\mathrm { d } Y _ { t } = b _ { t } ( Y _ { t } ) \mathrm { d } t + \sigma _ { t } \mathrm { d } W _ { t } ,
$$

with initial distributions $\mu _ { \tau }$ and $\pi _ { \tau }$ , respectively. Then, for almost every $t \in ( \tau , 1 )$ , we have

$$
\frac { \mathrm { d K L } ( \mu _ { t } \mid \pi _ { t } ) } { \mathrm { d } t } = - \frac { \sigma _ { t } ^ { 2 } } { 2 } \mathbb { E } _ { \mu _ { t } } \left[ \left\| \nabla _ { y } \log \frac { \mathrm { d } \mu _ { t } } { \mathrm { d } \pi _ { t } } ( Y _ { t } ) \right\| ^ { 2 } \right] .
$$

Proof. Let $f _ { t } : = \mathrm { d } \mu _ { t } / \mathrm { d } \pi _ { t }$ . Both $\mu _ { t }$ and $\pi _ { t }$ satisfy the Fokker–Planck equation $\partial _ { t } \rho _ { t } = \mathcal { L } _ { t } ^ { * } \rho _ { t }$ . Hence, differentiating the relative entropy and using integration by parts gives

$$
\begin{array} { r l r } {  { \frac { \mathrm { d K L } ( \mu _ { t } \mid \pi _ { t } ) } { \mathrm { d } t } = \int \log f _ { t } ( y ) \mathcal { L } _ { t } ^ { * } \mu _ { t } ( \mathrm { d } y ) - \int f _ { t } ( y ) \mathcal { L } _ { t } ^ { * } \pi _ { t } ( \mathrm { d } y ) } } \\ & { } & { = \mathbb { E } _ { \mu _ { t } } [ \mathcal { L } _ { t } \log f _ { t } ( Y _ { t } ) ] - \mathbb { E } _ { \pi _ { t } } [ \mathcal { L } _ { t } f _ { t } ( Y _ { t } ) ] . } \end{array}
$$

Since

$$
\mathcal { L } _ { t } \log { f _ { t } } = \frac { \mathcal { L } _ { t } f _ { t } } { f _ { t } } - \frac { \sigma _ { t } ^ { 2 } } { 2 } \| \nabla _ { y } \log { f _ { t } } \| ^ { 2 } ,
$$

the preceding display becomes

$$
\frac { \mathrm { d K L } ( \mu _ { t } \mid \pi _ { t } ) } { \mathrm { d } t } = - \frac { \sigma _ { t } ^ { 2 } } { 2 } \mathbb { E } _ { \mu _ { t } } \left[ \left\| \nabla _ { y } \log f _ { t } ( Y _ { t } ) \right\| ^ { 2 } \right] ,
$$

which is the desired result.

Lemma D.6 (Optimality Gap Bounds). Fix $t \in [ \varepsilon , 1 ]$ , and assume that $V ( t , \cdot )$ is concave. Then,for every $\mu \in \mathcal { P } , \pi _ { t } ^ { * } \in \mathop { \operatorname { a r g m a x } } _ { \mu \in \mathcal { P } } \mathcal { V } ( t , \mu )$ satisfies

$$
\mathrm { K L } ( \mu \mid \pi _ { t } ^ { * } ) \leq \eta \left( \mathcal { V } ( t , \pi _ { t } ^ { * } ) - \mathcal { V } ( t , \mu ) \right) \leq \mathrm { K L } ( \mu \mid \nu _ { t } ^ { \mu } ) .
$$

Proof. See Proposition 1 of Nitanda et al. (2022).

Lemma D.7 (Initial optimality gap at the cutoff). Let

$$
\phi ( y ) : = \frac { \delta R } { \delta \mu } [ \pi _ { 1 } ] ( y ) ,
$$

$$
\begin{array} { r } { L _ { \phi } : = \displaystyle \exp \operatorname* { s s u p } _ { \pi _ { 1 } } \phi - \displaystyle \exp \operatorname* { i n f } _ { \pi _ { 1 } } \phi , } \end{array}
$$

$$
S _ { 2 } : = \mathbb { E } _ { \pi _ { 1 } } \left[ \| Y _ { 1 } - \mathbb { E } _ { \pi _ { 1 } } \left[ Y _ { 1 } \right] \| ^ { 2 } \right] .
$$

Under the boundedfirst-order-variation assumption, $L _ { \phi } \leq 2 C _ { V }$ . Suppose that R is concave, $L _ { \phi } <$ $\infty ,$ and $S _ { 2 } < \infty$ . Let $\pi _ { \varepsilon } ^ { * } \in \mathrm { a r g m a x } _ { \mu \in \mathcal { P } } \mathcal { V } ( \varepsilon , \mu )$ and define

$$
\Delta _ { \varepsilon } : = \mathcal { V } ( \varepsilon , \pi _ { \varepsilon } ^ { * } ) - \mathcal { V } ( \varepsilon , \pi _ { \varepsilon } ) .
$$

Then, for the linear flow interpolation with independent endpoints,

$$
\Delta _ { \varepsilon } \leq \frac { S _ { 2 } } { 4 \eta } \left( e ^ { \eta L _ { \phi } } - 1 - \eta L _ { \phi } \right) \frac { \varepsilon ^ { 2 } } { ( 1 - \varepsilon ) ^ { 2 } } = O ( \varepsilon ^ { 2 } ) .
$$

For the diffusion interpolation,

$$
\Delta _ { \varepsilon } \leq \frac { S _ { 2 } } { 4 \eta } \left( e ^ { \eta L _ { \phi } } - 1 - \eta L _ { \phi } \right) \frac { \varepsilon } { 1 - \varepsilon } = O ( \varepsilon ) .
$$

Proof. Write

$$
Y _ { \varepsilon } = a _ { \varepsilon } Y _ { 1 } + b _ { \varepsilon } Z , \qquad Z \sim { \mathcal { N } } ( 0 , I ) , \quad Z \perp Y _ { 1 } ,
$$

where $( a _ { \varepsilon } , b _ { \varepsilon } ) = ( \varepsilon , 1 - \varepsilon )$ for the linear flow interpolation and $( a _ { \varepsilon } , b _ { \varepsilon } ) = ( \sqrt { \varepsilon } , \sqrt { 1 - \varepsilon } )$ for the diffusion interpolation. Since $K _ { \varepsilon } \pi _ { \varepsilon } = \pi _ { 1 }$ , the chain rule gives

$$
f _ { \varepsilon } ( y ) : = \frac { \delta V ( \varepsilon , \pi _ { \varepsilon } ) } { \delta \mu } ( y ) = \mathbb { E } \left[ \phi ( Y _ { 1 } ) \mid Y _ { \varepsilon } = y \right] .
$$

Since $K _ { \varepsilon }$ is linear, concavity of R implies concavity of $V ( \varepsilon , \cdot )$ . Lemma D.6 with $\mu = \pi _ { \varepsilon }$ and the definition of $\nu _ { \varepsilon } ^ { \pi _ { \varepsilon } }$ imply

$$
\begin{array} { r l } & { \Delta _ { \varepsilon } \leq \displaystyle \frac { 1 } { \eta } \mathrm { K L } ( \pi _ { \varepsilon } \mid \nu _ { \varepsilon } ^ { \pi _ { \varepsilon } } ) } \\ & { \quad = \displaystyle \frac { 1 } { \eta } \log \mathbb { E } _ { \pi _ { \varepsilon } } \left[ \exp \left( \eta \left\{ f _ { \varepsilon } ( Y _ { \varepsilon } ) - \mathbb { E } _ { \pi _ { \varepsilon } } \left[ f _ { \varepsilon } ( Y _ { \varepsilon } ) \right] \right\} \right) \right] . } \end{array}\tag{9}
$$

Let $P _ { y }$ denote the conditional law of $Y _ { 1 }$ given $Y _ { \varepsilon } = y$ . Since ϕ has essential range of length $L _ { \phi } ,$ Pinsker’s inequality gives

$$
\begin{array} { r } { | f _ { \varepsilon } ( y ) - \mathbb { E } _ { \pi _ { 1 } } \left[ \phi ( Y _ { 1 } ) \right] | ^ { 2 } \leq L _ { \phi } ^ { 2 } \mathrm { T V } ( P _ { y } , \pi _ { 1 } ) ^ { 2 } } \\ { \leq \frac { L _ { \phi } ^ { 2 } } { 2 } \mathrm { K L } ( P _ { y } \mid \pi _ { 1 } ) . } \end{array}
$$

Averaging over $Y _ { \varepsilon }$ yields

$$
\operatorname { V a r } { ( f _ { \varepsilon } ( Y _ { \varepsilon } ) ) } \leq \frac { L _ { \phi } ^ { 2 } } { 2 } \operatorname { M I } ( Y _ { 1 } ; Y _ { \varepsilon } ) .\tag{10}
$$

Let $Q _ { \varepsilon } : = \mathcal { N } ( a _ { \varepsilon } \mathbb { E } _ { \pi _ { 1 } } \left[ Y _ { 1 } \right] , b _ { \varepsilon } ^ { 2 } I )$ . Then,

$$
\begin{array} { r l } & { \mathrm { M I } ( Y _ { 1 } ; Y _ { \varepsilon } ) \leq \mathbb { E } _ { \pi _ { 1 } } \left[ \mathrm { K L } \left( \mathcal { N } ( a _ { \varepsilon } Y _ { 1 } , b _ { \varepsilon } ^ { 2 } I ) \mid Q _ { \varepsilon } \right) \right] } \\ & { \quad \quad \quad = \frac { a _ { \varepsilon } ^ { 2 } } { 2 b _ { \varepsilon } ^ { 2 } } S _ { 2 } . } \end{array}
$$

Combining this with (10) gives

$$
\mathrm { V a r } \left( f _ { \varepsilon } ( Y _ { \varepsilon } ) \right) \leq \frac { L _ { \phi } ^ { 2 } S _ { 2 } } { 4 } \frac { a _ { \varepsilon } ^ { 2 } } { b _ { \varepsilon } ^ { 2 } } .\tag{11}
$$

Set $U _ { \varepsilon } : = f _ { \varepsilon } ( Y _ { \varepsilon } ) - \mathbb { E } _ { \pi _ { \varepsilon } } \left[ f _ { \varepsilon } ( Y _ { \varepsilon } ) \right]$ . Then $\mathbb { E } \left[ U _ { \varepsilon } \right] = 0$ and $U _ { \varepsilon } \leq L _ { \phi } . \mathrm { f f } L _ { \phi } = 0$ , then $U _ { \varepsilon } = 0$ almost surely and (9) gives $\Delta _ { \varepsilon } = 0$ . Suppose that $L _ { \phi } > 0$ . For any centered random variable $U \leq L ,$ Bennett’s moment-generating-function inequality gives

$$
\log \mathbb { E } \left[ e ^ { \eta U } \right] \leq \frac { \mathrm { V a r } ( U ) } { L ^ { 2 } } \left( e ^ { \eta L } - 1 - \eta L \right) .
$$

Applying this inequality to (9) and using (11) gives

$$
\Delta _ { \varepsilon } \leq \frac { S _ { 2 } } { 4 \eta } \left( e ^ { \eta L _ { \phi } } - 1 - \eta L _ { \phi } \right) \frac { a _ { \varepsilon } ^ { 2 } } { b _ { \varepsilon } ^ { 2 } } .
$$

Substituting the two choices of $( a _ { \varepsilon } , b _ { \varepsilon } )$ proves the claim.

Lemma D.8 (Fisher Information Comparison via Pinsker’s Inequality). Define

$$
\begin{array} { r l } & { s _ { t } ( y ) : = \nabla _ { y } \log \frac { \pi _ { t } ^ { * } ( y ) } { \pi _ { t } ( y ) } , } \\ & { \Phi _ { t } ( y ) : = \| s _ { t } ( y ) \| ^ { 2 } , } \\ & { \Psi _ { t } ( y ) : = \mathrm { d i v } s _ { t } ( y ) + s _ { t } ( y ) \cdot \nabla _ { y } \log \pi _ { t } ^ { * } ( y ) , } \end{array}
$$

and suppose that there exists $C _ { t } < \infty$ such that

$$
\operatorname* { s u p } _ { y } \bigl \| s _ { t } ( y ) \bigr \| ^ { 2 } \leq C _ { t } , \qquad \operatorname* { s u p } _ { y } \bigl | \Psi _ { t } ( y ) \bigr | \leq C _ { t } .
$$

Then, for every $\mu \in \mathcal P$

$$
\begin{array} { r l } & { I ( \pi _ { t } ^ { * } \mid \pi _ { t } ) - I ( \mu \mid \pi _ { t } ) \leq \frac { 5 C _ { t } } { \sqrt { 2 } } \sqrt { \mathrm { K L } ( \mu \mid \pi _ { t } ^ { * } ) } } \\ & { \qquad - I ( \mu \mid \pi _ { t } ^ { * } ) } \\ & { \qquad \leq \frac { 5 C _ { t } } { \sqrt { 2 } } \sqrt { \mathrm { K L } ( \mu \mid \pi _ { t } ^ { * } ) } . } \end{array}
$$

Proof. We have

$$
\nabla _ { y } \log \frac { \mu ( y ) } { \pi _ { t } ( y ) } = \nabla _ { y } \log \frac { \mu ( y ) } { \pi _ { t } ^ { * } ( y ) } + s _ { t } ( y ) .
$$

Expanding the square yields

$$
I ( \mu \mid \pi _ { t } ) = I ( \mu \mid \pi _ { t } ^ { * } ) + 2 \int \mu ( y ) \nabla _ { y } \log \frac { \mu ( y ) } { \pi _ { t } ^ { * } ( y ) } \cdot s _ { t } ( y ) \mathrm { d } y + \mathbb { E } _ { \mu } \left[ \Phi _ { t } \right] .
$$

By integration by parts,

$$
\int \mu ( y ) \nabla _ { y } \log \frac { \mu ( y ) } { \pi _ { t } ^ { * } ( y ) } \cdot s _ { t } ( y ) \mathrm { d } y = - \mathbb { E } _ { \mu } \left[ \Psi _ { t } \right] .
$$

Furthermore,

$$
\mathbb { E } _ { \pi _ { t } ^ { * } } \left[ \Psi _ { t } \right] = 0 , \qquad I ( \pi _ { t } ^ { * } \mid \pi _ { t } ) = \mathbb { E } _ { \pi _ { t } ^ { * } } \left[ \Phi _ { t } \right] .
$$

Consequently,

$$
\begin{array} { r l } & { I ( \pi _ { t } ^ { * } \mid \pi _ { t } ) - I ( \mu \mid \pi _ { t } ) = \big ( \mathbb { E } _ { \pi _ { t } ^ { * } } \left[ \Phi _ { t } \right] - \mathbb { E } _ { \mu } \left[ \Phi _ { t } \right] \big ) } \\ & { \phantom { I ( \pi _ { t } ^ { * } | \pi _ { t } ) } + 2 \left( \mathbb { E } _ { \mu } \left[ \Psi _ { t } \right] - \mathbb { E } _ { \pi _ { t } ^ { * } } \left[ \Psi _ { t } \right] \right) - I ( \mu \mid \pi _ { t } ^ { * } ) . } \end{array}
$$

Since $0 \leq \Phi _ { t } \leq C _ { t }$ and $| \Psi _ { t } | \leq C _ { t }$ , we have

$$
\begin{array} { r l } & { \mathbb { E } _ { \pi _ { t } ^ { * } } \left[ \Phi _ { t } \right] - \mathbb { E } _ { \mu } \left[ \Phi _ { t } \right] \leq C _ { t } \mathrm { T V } ( \mu , \pi _ { t } ^ { * } ) , } \\ & { \mathbb { E } _ { \mu } \left[ \Psi _ { t } \right] - \mathbb { E } _ { \pi _ { t } ^ { * } } \left[ \Psi _ { t } \right] \leq 2 C _ { t } \mathrm { T V } ( \mu , \pi _ { t } ^ { * } ) . } \end{array}
$$

Pinsker’s inequality gives

$$
\mathrm { T V } ( \mu , \pi _ { t } ^ { * } ) \leq \sqrt { \frac { 1 } { 2 } \mathrm { K L } ( \mu \mid \pi _ { t } ^ { * } ) } .
$$

Combining these inequalities gives the first inequality in the statement. The second follows from $I ( \mu \mid \pi _ { t } ^ { * } ) \geq 0$ □

## E PROOF OF LEMMA 2.2

By Bayes’ rule, we have

$$
\log \pi _ { 1 } ( z \mid Y _ { t } = y ) = \log \pi _ { t } ( y \mid Y _ { 1 } = z ) + \log \pi _ { 1 } ( z ) - \log \pi _ { t } ( y ) ,
$$

and differentiating both sides with respect to y gives

$$
\nabla _ { y } \log \pi _ { 1 } ( z \mid Y _ { t } = y ) = \nabla _ { y } \log \pi _ { t } ( y \mid Y _ { 1 } = z ) - \nabla _ { y } \log \pi _ { t } ( y ) .
$$

The second term $\nabla \log \pi _ { t } ( y )$ is the score function of $\pi _ { t } .$ , which is estimated by a neural network for diffusion models and can be computed from the velocity field for flow models using Lemma D.2. Therefore, we only need to compute the first term $\nabla$ log $\overline { { \pi } } _ { t } ( y \mid Y _ { 1 } = z )$

Flow Models. Under the memoryless noise schedule, Lemma D.3 gives

$$
\pi _ { t } ( y \mid Y _ { 1 } = z ) = C \cdot \exp { \left( - { \frac { 1 } { 2 ( 1 - t ) ^ { 2 } } } { \left\| y - t z \right\| } ^ { 2 } \right) } ,
$$

where $C$ is a normalization constant. Thus,

$$
\nabla _ { y } \log \pi _ { t } ( y \mid Y _ { 1 } = z ) = - \frac { 1 } { ( 1 - t ) ^ { 2 } } ( y - t z ) .
$$

Diffusion Models. For $t \in [ \varepsilon , 1 )$ , let $s = 1 - t \in ( 0 , 1 - \varepsilon ]$ . The solution of the forward SDE satisfies

$$
\begin{array} { r } { X _ { s } \mid X _ { 0 } = z \sim \mathcal { N } ( \sqrt { 1 - s } z , s I ) . } \end{array}
$$

The reverse process is the time reversal of the forward process, so $Y _ { t } = X _ { 1 - t }$ and $Y _ { 1 } = X _ { 0 }$ under the time-reversal coupling. Therefore,

$$
\boldsymbol { Y _ { t } } \mid \boldsymbol { Y _ { 1 } } = z \sim \mathcal { N } ( \sqrt { t } z , ( 1 - t ) I )
$$

and we have

$$
\nabla _ { y } \log \pi _ { t } ( y \mid Y _ { 1 } = z ) = - { \frac { 1 } { 1 - t } } ( y - { \sqrt { t } } z ) .
$$

## F PROOF OF PROPOSITION 3.1

The marginal distribution $\pi _ { t } ^ { g }$ of the guided SDE satisfies the Fokker–Planck equation

$$
\begin{array} { r } { \partial _ { t } \pi _ { t } ^ { g } = \mathcal { L } _ { t } ^ { * } \pi _ { t } ^ { g } - \mathrm { d i v } ( g _ { t } \pi _ { t } ^ { g } ) . } \end{array}
$$

Therefore, the chain rule for functions of measures and integration by parts give

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } V ( t , \pi _ { t } ^ { g } ) } { \mathrm { d } t } = \partial _ { t } V ( t , \pi _ { t } ^ { g } ) + \int \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) \partial _ { t } \pi _ { t } ^ { g } ( \mathrm { d } y ) } \\ { \displaystyle \qquad = \partial _ { t } V ( t , \pi _ { t } ^ { g } ) + \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \mathcal { L } _ { t } \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) \right] + \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ g _ { t } ( y ) \cdot \nabla \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) \right] } \\ { \displaystyle \qquad = \partial _ { t } V ( t , \pi _ { t } ^ { g } ) + \mathfrak { L } _ { t } ^ { g } V ( t , \pi _ { t } ^ { g } ) , } \end{array}
$$

where

$$
\mathfrak { L } _ { t } ^ { g } V ( t , \mu ) : = \mathfrak { L } _ { t } V ( t , \mu ) + \mathbb { E } _ { \mu } \left[ g _ { t } ( y ) \cdot \nabla \frac { \delta V ( t , \mu ) } { \delta \mu } ( y ) \right] .
$$

Applying Lemma D.4, we have

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } V ( t , \pi _ { t } ^ { g } ) } { \mathrm { d } t } = \partial _ { t } V ( t , \pi _ { t } ^ { g } ) + \mathfrak { L } _ { t } V ( t , \pi _ { t } ^ { g } ) + \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ g _ { t } ( y ) \cdot \nabla \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) \right] } \\ { \displaystyle \qquad = \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ g _ { t } ( y ) \cdot \nabla \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) \right] . } \end{array}
$$

Integrating both sides from $t = \varepsilon \tan t = 1$ and using $K _ { \varepsilon } \pi _ { \varepsilon } = \pi _ { 1 }$ , we have

$$
\begin{array} { l } { { \displaystyle V ( 1 , \pi _ { 1 } ^ { g } ) - V ( \varepsilon , \pi _ { \varepsilon } ) = \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ g _ { t } ( y ) \cdot \nabla \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) \right] \mathrm { d } t } } \\ { { \displaystyle \qquad = R [ \pi _ { 1 } ^ { g } ] - R [ \pi _ { 1 } ] } . } \end{array}
$$

This completes the proof of the reward improvement.

Since the base and guided processes share the initial marginal $\pi _ { \varepsilon } ,$ , the KL identity follows from Girsanov’s theorem and the admissibility conditions:

$$
\mathrm { K L } ( \mathbb { P } _ { \pi ^ { g } } \mid \mathbb { P } _ { \pi } ) = \frac { 1 } { 2 } \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \frac { \mid \mid g _ { t } ( Y _ { t } ) \mid \mid ^ { 2 } } { \sigma _ { t } ^ { 2 } } \right] \mathrm { d } t .
$$

## G PROOF OF THEOREM 3.2

Applying Proposition 3.1 to the steepest guidance $\begin{array} { r } { g _ { t } ( y ) = \lambda \sigma _ { t } ^ { 2 } \nabla _ { y } \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) } \end{array}$ , we have

$$
R [ \pi _ { 1 } ^ { g } ] - R [ \pi _ { 1 } ] = \lambda \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \sigma _ { t } ^ { 2 } \left\| \nabla _ { y } \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) \right\| ^ { 2 } \right] \mathrm { d } t ,
$$

$$
\mathrm { K L } ( \mathbb { P } _ { \pi ^ { g } } \mid \mathbb { P } _ { \pi } ) = \frac { \lambda ^ { 2 } } { 2 } \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \sigma _ { t } ^ { 2 } \left\| \nabla _ { y } \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) \right\| ^ { 2 } \right] \mathrm { d } t .
$$

Combining the above two equations, we have

$$
R [ \pi _ { 1 } ^ { g } ] - R [ \pi _ { 1 } ] - \frac { 1 } { \lambda } \mathrm { K L } ( \mathbb { P } _ { \pi ^ { g } } \mid \mathbb { P } _ { \pi } ) = \frac { \lambda } { 2 } \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \sigma _ { t } ^ { 2 } \left\| \nabla _ { y } \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) \right\| ^ { 2 } \right] \mathrm { d } t
$$

This completes the proof.

## H PROOF OF COROLLARY 4.1

Define

$$
A : = \int _ { \varepsilon } ^ { 1 } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \sigma _ { t } ^ { 2 } \left\| \nabla _ { y } \frac { \delta V ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( Y _ { t } ) \right\| ^ { 2 } \right] \mathrm { d } t .
$$

From Theorem 3.2, we have

$$
R [ \pi _ { 1 } ^ { g } ] - R [ \pi _ { 1 } ] = \lambda A , \qquad \mathrm { K L } ( \mathbb { P } _ { \pi ^ { g } } \ | \ \mathbb { P } _ { \pi } ) = \frac { \lambda ^ { 2 } } { 2 } A .
$$

From the information processing inequality, we have $\mathrm { K L } ( \pi _ { 1 } ^ { g } \mid \pi _ { 1 } ) \le \mathrm { K L } ( \mathbb { P } _ { \pi ^ { g } } \mid \mathbb { P } _ { \pi } )$ . Therefore,

$$
\begin{array} { l } { { J _ { \eta } ( \pi _ { 1 } ^ { g } ) - J _ { \eta } ( \pi _ { 1 } ) = R [ \pi _ { 1 } ^ { g } ] - { \displaystyle \frac { 1 } { \eta } } \mathrm { K L } ( \pi _ { 1 } ^ { g } \mid \pi _ { 1 } ) - R [ \pi _ { 1 } ] } } \\ { ~ } \\ { { \displaystyle \geq \lambda A - \frac { \lambda ^ { 2 } } { 2 \eta } A } } \\ { { \displaystyle ~ = \lambda \left( 1 - \frac { \lambda } { 2 \eta } \right) A } } \\ { { ~ \displaystyle \geq 0 , } } \end{array}
$$

where the last inequality follows from $0 < \lambda \leq 2 \eta$ . This completes the proof.

## I PROOF OF PROPOSITION 4.2

The result follows from Lemma D.1.

## J PROOF OF THEOREM 4.3

By Proposition 4.2, it suffices to analyze the guided SDE with $\begin{array} { r } { g _ { t } ( y ) = \lambda \sigma _ { t } ^ { 2 } \nabla _ { y } \frac { \delta \mathcal { V } ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) } \end{array}$ . Since $V ( t , \mu ) = R [ K _ { t } \mu ]$ satisfies the Kolmogorov backward equation, Lemmas D.4 and D.5 give

$$
\partial _ { t } \mathcal { V } ( t , \mu ) + \mathfrak { L } _ { t } \mathcal { V } ( t , \mu ) = \frac { \sigma _ { t } ^ { 2 } } { 2 \eta } I ( \mu \mid \pi _ { t } ) \ge 0 .
$$

The chain-rule calculation for the guided SDE, together with $\begin{array} { r } { g _ { t } ( y ) = \lambda \sigma _ { t } ^ { 2 } \nabla _ { y } \frac { \delta \mathcal { V } ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( y ) } \end{array}$ , then yields

$$
\begin{array} { r l } & { \displaystyle \frac { \mathrm { d } \mathcal { V } ( t , \pi _ { t } ^ { g } ) } { \mathrm { d } t } = \partial _ { t } \mathcal { V } ( t , \pi _ { t } ^ { g } ) + \mathfrak { L } _ { t } \mathcal { V } ( t , \pi _ { t } ^ { g } ) + \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ g _ { t } ( Y _ { t } ) \cdot \nabla _ { y } \frac { \delta \mathcal { V } ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( Y _ { t } ) \right] } \\ & { \quad \quad \quad \quad \quad = \frac { \sigma _ { t } ^ { 2 } } { 2 \eta } I ( \pi _ { t } ^ { g } \mid \pi _ { t } ) + \lambda \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \sigma _ { t } ^ { 2 } \left. \nabla _ { y } \frac { \delta \mathcal { V } ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( Y _ { t } ) \right. ^ { 2 } \right] } \\ & { \quad \quad \quad \quad \geq \lambda \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \sigma _ { t } ^ { 2 } \left. \nabla _ { y } \frac { \delta \mathcal { V } ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( Y _ { t } ) \right. ^ { 2 } \right] . } \end{array}
$$

Integrating over $t \in [ \varepsilon , 1 ]$ and using

$$
\mathcal { V } ( \varepsilon , \pi _ { \varepsilon } ) = R [ K _ { \varepsilon } \pi _ { \varepsilon } ] - \eta ^ { - 1 } \mathrm { K L } ( \pi _ { \varepsilon } \mid \pi _ { \varepsilon } ) = R [ \pi _ { 1 } ] = J _ { \eta } ( \pi _ { 1 } )
$$

and $\mathcal { V } ( 1 , \pi _ { 1 } ^ { g } ) = J _ { \eta } ( \pi _ { 1 } ^ { g } )$ proves the claim.

## K PROOF OF THEOREM 4.5

For $t \in [ \varepsilon , 1 ]$ , let $\Delta _ { t } = \mathcal { V } ( t , \pi _ { t } ^ { * } ) - \mathcal { V } ( t , \pi _ { t } ^ { g } )$ . Since $\pi _ { \varepsilon } ^ { g } = \pi _ { \varepsilon }$ , the initial gap is

$$
\Delta _ { \varepsilon } = \mathcal { V } ( \varepsilon , \pi _ { \varepsilon } ^ { * } ) - \mathcal { V } ( \varepsilon , \pi _ { \varepsilon } ) \geq 0 .
$$

Since we assume that $\pi _ { 1 }$ has a finite second moment, and the first-order variation of R is bounded, it follows from Lemma D.7 that $\Delta _ { \varepsilon } = { O } ( \varepsilon ^ { 2 } )$ for the linear flow interpolation and $\Delta _ { \varepsilon } = O ( \varepsilon )$ for the diffusion interpolation. For simplicity, write $\nu _ { t } ^ { g } : = \nu _ { t } ^ { \pi _ { t } ^ { g } }$ . By the definition of V,

$$
\nabla _ { y } \frac { \delta \mathcal { V } ( t , \mu ) } { \delta \mu } ( y ) = - \frac { 1 } { \eta } \nabla _ { y } \log \frac { \mu ( y ) } { \nu _ { t } ^ { \mu } ( y ) } .
$$

From the first-order optimality condition, $\pi _ { t } ^ { * } ~ = ~ \nu _ { t } ^ { \pi _ { t } ^ { * } }$ . Hence, $\frac { \delta \mathcal { V } ( t , \pi _ { t } ^ { * } ) } { \delta \mu }$ is constant in y and $\mathfrak { L } _ { t } \mathcal { V } ( t , \pi _ { t } ^ { * } ) = 0$ . Moreover, Lemmas D.4 and D.5 imply

$$
\partial _ { t } \mathcal { V } ( t , \mu ) + \mathfrak { L } _ { t } \mathcal { V } ( t , \mu ) = \frac { \sigma _ { t } ^ { 2 } } { 2 \eta } I ( \mu \mid \pi _ { t } ) .
$$

In particular, since $\mathfrak { L } _ { t } \mathcal { V } ( t , \pi _ { t } ^ { * } ) = 0$ , we have

$$
\partial _ { t } \mathcal { V } ( t , \pi _ { t } ^ { * } ) = \frac { \sigma _ { t } ^ { 2 } } { 2 \eta } I ( \pi _ { t } ^ { * } \mid \pi _ { t } ) ,
$$

whereas

$$
\partial _ { t } \mathcal { V } ( t , \pi _ { t } ^ { g } ) + \mathfrak { L } _ { t } \mathcal { V } ( t , \pi _ { t } ^ { g } ) = \frac { \sigma _ { t } ^ { 2 } } { 2 \eta } I ( \pi _ { t } ^ { g } \mid \pi _ { t } ) .
$$

By Proposition 4.2, we may use the Fokker–Planck equation of the marginally equivalent ideal SDE:

$$
\partial _ { t } \pi _ { t } ^ { g } = \mathcal { L } _ { t } ^ { * } \pi _ { t } ^ { g } - \lambda \sigma _ { t } ^ { 2 } \operatorname { d i v } \left( \pi _ { t } ^ { g } \nabla _ { y } \frac { \delta \mathcal { V } ( t , \pi _ { t } ^ { g } ) } { \delta \mu } \right) .
$$

The envelope theorem and integration by parts now yield

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } \Delta _ { t } = \partial _ { t } \mathcal { V } ( t , \pi _ { t } ^ { * } ) - ( \partial _ { t } \mathcal { V } ( t , \pi _ { t } ^ { g } ) + \mathfrak { L } _ { t } \mathcal { V } ( t , \pi _ { t } ^ { g } ) ) } \\ { \displaystyle - \lambda \sigma _ { t } ^ { 2 } \mathbb { E } _ { \pi _ { t } ^ { g } } \left[ \left\| \nabla _ { y } \frac { \delta \mathcal { V } ( t , \pi _ { t } ^ { g } ) } { \delta \mu } ( Y _ { t } ) \right\| ^ { 2 } \right] } \\ { = \displaystyle - \frac { \lambda \sigma _ { t } ^ { 2 } } { \eta ^ { 2 } } I ( \pi _ { t } ^ { g } \mid \nu _ { t } ^ { g } ) + \frac { \sigma _ { t } ^ { 2 } } { 2 \eta } \left( I ( \pi _ { t } ^ { * } \mid \pi _ { t } ) - I ( \pi _ { t } ^ { g } \mid \pi _ { t } ) \right) . } \end{array}
$$

Since $K _ { t }$ is linear, concavity of R implies concavity of $V ( t , \cdot )$ . Applying Lemma D.6 with $\mu = \pi _ { t } ^ { g }$ gives

$$
\mathrm { K L } ( \pi _ { t } ^ { g } \mid \pi _ { t } ^ { * } ) \leq \eta \Delta _ { t } \leq \mathrm { K L } ( \pi _ { t } ^ { g } \mid \nu _ { t } ^ { g } ) .
$$

Thus, $\operatorname { L S I } ( \alpha _ { t } )$ yields

$$
I ( \pi _ { t } ^ { g } \mid \nu _ { t } ^ { g } ) \ge 2 \eta \alpha _ { t } \Delta _ { t } .
$$

Applying Lemma D.8 with $\mu = \pi _ { t } ^ { g }$ gives

$$
\begin{array} { r l } & { I ( \pi _ { t } ^ { * } \mid \pi _ { t } ) - I ( \pi _ { t } ^ { g } \mid \pi _ { t } ) \leq \frac { 5 C _ { t } } { \sqrt { 2 } } \sqrt { \mathrm { K L } ( \pi _ { t } ^ { g } \mid \pi _ { t } ^ { * } ) } } \\ & { \qquad \leq 5 C _ { t } \sqrt { \frac { \eta } { 2 } } \sqrt { \Delta _ { t } } . } \end{array}
$$

Consequently,

$$
\frac { \mathrm { d } } { \mathrm { d } t } \Delta _ { t } \leq - 2 a _ { t } \Delta _ { t } + 2 a _ { t } h _ { t } \sqrt { \Delta _ { t } } ,
$$

where

$$
\begin{array} { c c } { \displaystyle { a _ { t } : = \frac { \lambda \sigma _ { t } ^ { 2 } \alpha _ { t } } { \eta } , \qquad } } & { \displaystyle { h _ { t } : = \frac { 5 \sqrt { \eta } C _ { t } } { 4 \sqrt { 2 } \lambda \alpha _ { t } } , } } \\ { \displaystyle { A _ { \lambda , \varepsilon } : = \int _ { \varepsilon } ^ { 1 } a _ { s } \mathrm { d } s , \qquad } } & { \displaystyle { w _ { \lambda } ( t ) : = a _ { t } \exp \left( - \int _ { t } ^ { 1 } a _ { s } \mathrm { d } s \right) . } } \end{array}
$$

For $\delta > 0 .$ , let $u _ { \delta , t } : = \sqrt { \Delta _ { t } + \delta }$ . Then,

$$
\frac { \mathrm { d } } { \mathrm { d } t } u _ { \delta , t } \leq - a _ { t } u _ { \delta , t } + a _ { t } h _ { t } + a _ { t } \sqrt { \delta } .
$$

Applying Gronwall’s inequality and sending¨ $\delta \downarrow 0$ implies

$$
\sqrt { \Delta _ { 1 } } \leq e ^ { - A _ { \lambda , \varepsilon } } \sqrt { \Delta _ { \varepsilon } } + \int _ { \varepsilon } ^ { 1 } h _ { t } w _ { \lambda } ( t ) \mathrm { d } t .
$$

Moreover,

$$
\int _ { \varepsilon } ^ { 1 } w _ { \lambda } ( t ) \mathrm { d } t = 1 - e ^ { - A _ { \lambda , \varepsilon } } .
$$

Hence, $e ^ { - A _ { \lambda , \varepsilon } } \delta _ { \varepsilon } + w _ { \lambda } ( t )$ dt is a probability measure on $\{ \varepsilon \} \cup [ \varepsilon , 1 ]$ . Jensen’s inequality gives

$$
\begin{array} { r l } & { \displaystyle \Delta _ { 1 } \leq e ^ { - A _ { \lambda , \varepsilon } } \Delta _ { \varepsilon } + \int _ { \varepsilon } ^ { 1 } h _ { t } ^ { 2 } w _ { \lambda } ( t ) \mathrm { d } t } \\ & { \quad \quad = e ^ { - A _ { \lambda , \varepsilon } } \Delta _ { \varepsilon } + \frac { 2 5 \eta } { 3 2 \lambda ^ { 2 } } \int _ { \varepsilon } ^ { 1 } \frac { C _ { t } ^ { 2 } } { \alpha _ { t } ^ { 2 } } w _ { \lambda } ( t ) \mathrm { d } t } \\ & { \quad \quad \leq O ( \varepsilon ) + \frac { 2 5 \eta } { 3 2 \lambda ^ { 2 } } \int _ { \varepsilon } ^ { 1 } \frac { C _ { t } ^ { 2 } } { \alpha _ { t } ^ { 2 } } w _ { \lambda } ( t ) \mathrm { d } t , } \end{array}
$$

where the initialization term is in fact $O ( \varepsilon ^ { 2 } )$ for the linear flow interpolation. Finally, since $K _ { 1 }$ is the identity kernel, $\mathcal { V } ( 1 , \mu ) = J _ { \eta } ( \mu )$ and therefore

$$
\Delta _ { 1 } = \operatorname* { s u p } _ { \mu \in \mathcal { P } } J _ { \eta } ( \mu ) - J _ { \eta } ( \pi _ { 1 } ^ { g } ) ,
$$

which proves the claim.

## L EXPERIMENTAL DETAILS AND ADDITIONAL RESULTS

## L.1 VARIANCE REDUCTION

Equations (4) and (5) state the uncentered estimators used in the theoretical discussion. In all experiments, we center the coefficients multiplying the conditional score to reduce variance and remove the finite-sample dependence on the additive constant of the first-order variation.

For Steepest Guidance, let

$$
\psi _ { i } ^ { ( j ) } : = \frac { \delta R } { \delta \mu } [ \hat { \mu } _ { 1 , l } ] ( y _ { l , i } ^ { ( j ) } ) , \qquad b _ { - i } ^ { ( j ) } : = \frac { 1 } { k - 1 } \sum _ { i ^ { \prime } \neq i } \psi _ { i ^ { \prime } } ^ { ( j ) } ,
$$

where $k \geq 2$ and $b _ { - i } ^ { ( j ) }$ is the leave-one-out baseline. The estimator used in the experiments is

$$
\hat { g } _ { t _ { l } } ^ { \mathrm { s t e e p e s t , L O O } } ( y _ { l } ^ { ( j ) } ) = \lambda \sigma _ { t _ { l } } ^ { 2 } \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \left( \psi _ { i } ^ { ( j ) } - b _ { - i } ^ { ( j ) } \right) \nabla _ { y _ { l } ^ { ( j ) } } \log \pi _ { 1 } ( y _ { l , i } ^ { ( j ) } \mid Y _ { t _ { l } } = y _ { l } ^ { ( j ) } ) .
$$

This estimator is exactly invariant to replacing every $\psi _ { i } ^ { ( j ) }$ by $\psi _ { i } ^ { ( j ) } + c$ for an arbitrary constant c. For a linear reward functional, the leave-one-out baseline is independent of the i-th lookahead sample conditional on $Y _ { t _ { l } } = y _ { l } ^ { ( j ) }$ and therefore preserves the unbiasedness of the estimator because the conditional score has zero expectation. For non-linear reward functionals, the finite-particle estimator remains generally biased as discussed in the main text; the leave-one-out centering is used for invariance and variance reduction.

For the softmax estimator used by DOIT, the coefficients have empirical mean $1 / k$ by construction, where k is the number of lookahead samples. We therefore subtract $1 / k$ from each softmax coefficient before multiplying it by the conditional score. Since the conditional score has zero expectation, this centering leaves the target expectation unchanged while removing the constant component of the coefficient and reducing variance.

## L.2 EXPERIMENTAL SETUP FOR BIAS EVALUATION.

Here, we describe the experimental setup of the bias evaluation. We evaluate the finite-sample bias in Fig. 1 on a one-dimensional Gaussian toy problem. The terminal distribution is $X _ { 1 } \sim \bar { \mathcal { N } } ( 0 , 1 )$ and we evaluate the guidance at $t = 0 . 5$ . We set the linear reward $r ( x ) = x$ and $\lambda = 5$ . In this toy problem, we can sample the conditional distribution exactly.

For each of 128 independently sampled states and each $k \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 \}$ , we draw 512 independent REINFORCE estimates. Let $g ^ { * }$ denote the analytic Doob’s guidance, $\bar { g } _ { k }$ the empirical one, and $s _ { k } ^ { 2 }$ the unbiased empirical variance across repeats. We report the relative bias

$$
\frac { \sqrt { \mathbb { E } _ { y } [ ( \bar { g } _ { k } - g ^ { * } ) ^ { 2 } - s _ { k } ^ { 2 } / 5 1 2 ] } } { \sqrt { \mathbb { E } _ { y } [ ( g ^ { * } ) ^ { 2 } ] } } ,
$$

where the expectations are empirical averages across states. The subtraction in the numerator removes the finite-repeat variance contribution to the squared error of $\bar { g } _ { k }$ . Error bars show the standard error across states.

Finally, we show that the plug-in estimator has no bias in this Gaussian linear-reward setting. The conditional sample can be expressed as follows:

$$
X _ { i } ( y ) = a _ { t } y + b _ { t } \epsilon _ { i } , \qquad \epsilon _ { i } \sim { \mathcal { N } } ( 0 , 1 ) ,
$$

where $a _ { t }$ and $b _ { t }$ are functions of t but not of $y .$ . Then, the plug-in estimator is

$$
\begin{array} { r l r } {  { \hat { g } _ { k } ^ { \mathrm { p l u g } } ( y ) = \sigma _ { t } ^ { 2 } \nabla _ { y } \log ( \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \exp \bigl ( \lambda X _ { i } ( y ) \bigr ) ) } } \\ & { } & { = \sigma _ { t } ^ { 2 } \nabla _ { y } [ \lambda a y + \log ( \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \exp ( \lambda b \epsilon _ { i } ) ) ] = \sigma _ { t } ^ { 2 } \lambda a . } \end{array}
$$

On the other hand, the analytic Doob’s guidance is

$$
\begin{array} { r } { g ^ { * } ( y ) = \sigma _ { t } ^ { 2 } \nabla _ { y } \log \mathbb { E } \left[ \exp \left( \lambda ( a y + b \epsilon ) \right) \right] } \\ { = \sigma _ { t } ^ { 2 } \nabla _ { y } \left( \lambda a y + \frac { \lambda ^ { 2 } b ^ { 2 } } { 2 } \right) = \sigma _ { t } ^ { 2 } \lambda a . } \end{array}
$$

Thus, $\hat { g } _ { k } ^ { \mathrm { p l u g } } ( y ) = g ^ { * } ( y )$ , which implies that the plug-in estimator has no bias in this setting.

## L.3 REWARD FUNCTIONS.

ImageReward and PickScore. ImageReward (Xu et al., 2023) and PickScore (Kirstain et al., 2023) are reward models that predict the score of an image based on human preferences.

Blueness. Following (Dandapanthula & Boffi, 2026), we define the blueness reward function as

$$
r _ { \mathrm { b l u e } } ( x ) = \frac { 1 } { H W } \sum _ { i = 1 } ^ { H } \sum _ { j = 1 } ^ { W } ( B _ { i , j } - R _ { i , j } - G _ { i , j } ) ,
$$

where H and W are the height and width of the image, respectively, and $R _ { i , j } , G _ { i , j }$ , and $B _ { i , j }$ are the red, green, and blue pixel values at position (i, j), respectively.

Compressibility. Compressibility is defined as

$$
r _ { \mathrm { c o m p r e s s } } ( x ) = - \frac { \mathrm { J P E G S i z e } _ { q = 9 5 } ( x ) } { 1 0 0 0 0 } ,
$$

where $\mathrm { J P E G S i z e } _ { q = 9 5 } ( x )$ is the size of the JPEG-compressed image x with quality factor $q = 9 5$ This is an example of a non-differentiable reward function.

## L.4 DETAILED SETUP FOR TEXT-TO-IMAGE EXPERIMENTS.

Hyperparameter Settings. We set L = 50 for FLUX and L = 160 for SD and sample 32 images with batch size M = 8 for each method in the main experiments. We summarize the grid for hyperparameter search and optimal λ in Table 4.

Guidance Schedule. For ImageReward and PickScore, we apply the guidance from l = 10 to l = 50 for FLUX and from l = 120 to l = 160 for SD, where l is the diffusion step index. For Blueness and Compressibility, we apply the guidance from l = 0 to l = 10 for FLUX and l = 0 to l = 50 for SD since the rewards are based on global structure acquired early in the process. For SD, we apply the guidance every 5 steps to reduce the computational cost.

Approximate Posterior Sampling. Following Dandapanthula & Boffi (2026), we use the Diamond Map (Holderrieth et al., 2026) for approximate posterior sampling of FLUX. Pretrained weights are available at https://huggingface.co/gabeguofanclub/ flux-1-dev-flowmap-lsd. For SD, we cannot find a pretrained Diamond Map, so we utilize Tweedie’s formula instead for 1-step denoising. The noise ratio is set to 5.0 for both models.

## L.5 ADDITIONAL RESULTS ON TEXT-TO-IMAGE GENERATION.

In addition to the main results reported in Table 1, we provide results for two additional prompts in Tables 5 and 6. We observe that the proposed method achieves better or comparable performance across the additional prompts, demonstrating its robustness.

## L.6 EXAMPLES OF GENERATED IMAGES.

Here, we provide examples of images generated with SD and FLUX for the Blueness, ImageReward, PickScore, and Compressibility reward functions.

![](images/a01dc5dc7420c116c501f542efd9ea85133284a9166f583540e6940a5448ddb5.jpg)  
Figure 5: Selected images generated with SD for the Blueness, ImageReward, PickScore, and Com pressibility reward functions (top to bottom).

![](images/ba7ccc60f4e77611a85d2d88478f4fd9c13b92e427b29d88b196279d3e16d91e.jpg)  
Figure 6: Selected images generated with FLUX for the Blueness, ImageReward, PickScore, and Compressibility reward functions (top to bottom).

Table 4: A summary of the grid search for the guidance scale λ in the text-to-image experiments. The table shows the search grid and selected optimal λ for each prompt, reward function, model, and method. Lion, Tokyo, and Lake denote the prompts in Tables 1, 5, and 6, respectively.
<table><tr><td rowspan="2">Reward</td><td rowspan="2">Grid</td><td colspan="2">SD</td><td colspan="2">FLUX</td></tr><tr><td>Steepest</td><td>Doob</td><td>Steepest</td><td>Doob</td></tr><tr><td colspan="6">Lion</td></tr><tr><td>Blueness</td><td>[50, 100, 200, 300, 400, 500]</td><td>500</td><td>300</td><td>500</td><td>400</td></tr><tr><td>ImageReward</td><td>[20, 30, 40, 50, 60, 70]</td><td>50</td><td>30</td><td>40</td><td>30</td></tr><tr><td>PickScore</td><td>[20, 30, 40, 50, 60, 70]</td><td>40</td><td>40</td><td>30</td><td>40</td></tr><tr><td>Compressibility</td><td>[10, 20, 40, 60, 80, 100]</td><td>100</td><td>20</td><td>100</td><td>100</td></tr><tr><td colspan="6">Tokyo</td></tr><tr><td>Blueness</td><td>[50, 100, 200, 300, 400, 500]</td><td>500</td><td>400</td><td>400</td><td>200</td></tr><tr><td>ImageReward</td><td>[20, 30, 40, 50, 60, 70]</td><td>70</td><td>70</td><td>40</td><td>60</td></tr><tr><td>PickScore</td><td>[20, 30, 40, 50, 60, 70]</td><td>30</td><td>60</td><td>20</td><td>50</td></tr><tr><td>Compressibility</td><td>[10, 20, 40, 60, 80, 100]</td><td>100</td><td>10</td><td>40</td><td>10</td></tr><tr><td colspan="6">Lake</td></tr><tr><td>Blueness</td><td>[50, 100, 200, 300, 400, 500]</td><td>500</td><td>400</td><td>500</td><td>500</td></tr><tr><td>ImageReward</td><td>[20, 30, 40, 50, 60, 70]</td><td>70</td><td>60</td><td>50</td><td>70</td></tr><tr><td>PickScore</td><td>[20, 30, 40, 50, 60, 70]</td><td>40</td><td>50</td><td>20</td><td>30</td></tr><tr><td>Compressibility</td><td>[10, 20, 40, 60, 80, 100]</td><td>100</td><td>80</td><td>40</td><td>100</td></tr></table>

Table 5: Text-to-image reward summary for the prompt “A cinematic night photograph of a rainy Tokyo street with neon reflections.” Results are reported as mean ± standard deviation over $n = 3 2$ generated samples.
<table><tr><td>Model</td><td>Reward</td><td></td><td>Unguided Steepest (Ours)</td><td>DOIT</td><td>SVDD</td></tr><tr><td>SD</td><td>Blueness</td><td> $- 0 . 3 4 \pm 0 . 0 7$ </td><td> ${ \bf 0 . 1 5 \pm 0 . 2 1 }$ </td><td> $- 0 . 3 1 \pm 0 . 0 7$ </td><td> $- 0 . 2 8 \pm 0 . 0 7$ </td></tr><tr><td></td><td>ImageReward</td><td> $1 . 3 2 \pm 0 . 2 9$ </td><td> ${ \bf 1 . 7 8 \pm 0 . 1 4 }$ </td><td> $1 . 5 5 \pm 0 . 2 4$ </td><td> $1 . 7 5 \pm 0 . 1 5$ </td></tr><tr><td></td><td>PickScore</td><td> $2 2 . 1 2 \pm 0 . 4 7$ </td><td> $2 2 . 9 0 \pm 0 . 4 8$ </td><td> $2 2 . 4 6 \pm 0 . 5 0$ </td><td> ${ \bf 2 2 . 9 5 \pm 0 . 5 1 }$ </td></tr><tr><td></td><td>Compressibility</td><td> $- 1 6 . 3 5 \pm 1 . 7 3$ </td><td> $- 4 . 6 1 \pm 1 . 9 9$ </td><td> $- 1 5 . 9 8 \pm 1 . 7 7$ </td><td> $- 1 5 . 6 0 \pm 1 . 6 7$ </td></tr><tr><td>FLUX</td><td>Blueness</td><td> $- 0 . 2 2 \pm 0 . 0 4$ </td><td> ${ \bf - 0 . 0 3 \pm 0 . 0 7 }$ </td><td> $- 0 . 1 7 \pm 0 . 0 4$ </td><td> $- 0 . 1 1 \pm 0 . 0 3$ </td></tr><tr><td></td><td>ImageReward</td><td> $0 . 9 3 \pm 0 . 5 2$ </td><td> ${ \bf 1 . 8 4 \pm 0 . 0 7 }$ </td><td> $1 . 4 1 \pm 0 . 4 8$ </td><td> $1 . 5 6 \pm 0 . 3 4$ </td></tr><tr><td></td><td>PickScore</td><td> $2 2 . 2 3 \pm 0 . 4 7$ </td><td> ${ \bf 2 3 . 5 3 \pm 0 . 3 2 }$ </td><td> $2 2 . 9 0 \pm 0 . 4 9$ </td><td> $2 3 . 3 0 \pm 0 . 3 4$ </td></tr><tr><td></td><td>Compressibility</td><td> $- 8 . 1 2 \pm 1 . 1 2$ </td><td> $\mathbf { - 5 . 1 3 \pm 0 . 2 7 }$ </td><td> $- 7 . 1 5 \pm 0 . 7 6$ </td><td> $- 6 . 3 5 \pm 0 . 5 7$ </td></tr></table>

## L.7 ADDITIONAL RESULTS ON NON-LINEAR REWARD FUNCTIONALS.

Diversity Seeking Generation. To enhance diversity in generated images, we compare entropy and Rao’s quadratic entropy as reward functionals. For Rao’s quadratic entropy, we use DINOv2 to extract features and an RBF kernel to compute the entropy. Note that entropy regularization does not require an additional model for feature extraction. The bandwidth of the RBF kernel is adaptively set to the median of pairwise squared distances between features divided by max{log M, 1}. We set the guidance scale λ to 1, 000 and Rao’s quadratic entropy reward weight to 1. For entropy maximization, we use the KL-regularized formulation described in Section B and set its reward and KL weights to $2 \times 1 0 ^ { - 4 }$ . Figure 7 shows images generated for the prompt “VAN GOGH CAFE TERASSE” using Rao’s quadratic entropy, entropy regularization, and unguided sampling. While unguided sampling produces images with similar scene composition (e.g., the 1st, 3rd, 4th, and 7th images in the bottom row), regularized sampling produces visually varied images. We also report in-batch diversity scores in Table 7. The score is computed as the mean pairwise cosine distance between the normalized DINOv2 image embeddings within the batch. As we discuss in Section A, several methods (Corso et al., 2023; Vinograd et al., 2026; Zilberstein et al., 2024) have been proposed to enhance diversity in generated images but they are based on heuristics or require first- and second-order derivatives of the kernel.

Table 6: Text-to-image reward summary for the prompt “A dreamy watercolor painting of a mountain lake at sunrise.” Results are reported as mean ± standard deviation over $n = 3 2$ generated samples.
<table><tr><td>Model</td><td>Reward</td><td></td><td>Unguided Steepest (Ours)</td><td>DOIT</td><td>SVDD</td></tr><tr><td>SD</td><td>Blueness</td><td> $- 0 . 4 7 \pm 0 . 1 0$ </td><td> ${ \bf 0 . 3 2 \pm 0 . 2 0 }$ </td><td> $- 0 . 4 4 \pm 0 . 1 0$ </td><td> $- 0 . 4 0 \pm 0 . 0 9$ </td></tr><tr><td></td><td>ImageReward</td><td> $0 . 0 3 \pm 0 . 4 3$ </td><td> $\mathbf { 0 . 6 4 \pm 0 . 1 6 }$ </td><td> $0 . 3 3 \pm 0 . 2 6$ </td><td> $0 . 5 8 \pm 0 . 1 8$ </td></tr><tr><td></td><td>PickScore</td><td> $2 1 . 8 4 \pm 0 . 5 4$ </td><td> $2 2 . 5 4 \pm 0 . 5 2$ </td><td> $2 2 . 1 2 \pm 0 . 5 3$ </td><td> ${ \bf 2 2 . 5 7 \pm 0 . 5 2 }$ </td></tr><tr><td></td><td>Compressibility</td><td> $- 1 1 . 7 6 \pm 1 . 4 2$ </td><td> $- 4 . 2 1 \pm 0 . 9 6$ </td><td> $- 1 1 . 4 1 \pm 1 . 4 4$ </td><td> $- 1 0 . 9 8 \pm 1 . 5 9$ </td></tr><tr><td>FLUX</td><td>Blueness</td><td> $- 0 . 5 8 \pm 0 . 0 7$ </td><td> ${ \bf 0 . 0 8 \pm 0 . 0 9 }$ </td><td> $- 0 . 4 8 \pm 0 . 0 7$ </td><td> $- 0 . 3 8 \pm 0 . 0 7$ </td></tr><tr><td></td><td>ImageReward</td><td> $0 . 9 7 \pm 0 . 2 1$ </td><td> ${ \bf 1 . 4 2 \pm 0 . 1 2 }$ </td><td> $1 . 1 7 \pm 0 . 1 8$ </td><td> $1 . 3 4 \pm 0 . 1 7$ </td></tr><tr><td></td><td>PickScore</td><td> $2 3 . 4 4 \pm 0 . 4 9$ </td><td> ${ \bf 2 4 . 3 7 \pm 0 . 3 6 }$ </td><td> $2 4 . 0 2 \pm 0 . 3 8$ </td><td> $2 4 . 3 2 \pm 0 . 4 5$ </td></tr><tr><td></td><td>Compressibility</td><td> $- 7 . 7 8 \pm 1 . 1 2$ </td><td> $- 4 . 6 6 \pm 0 . 2 9$ </td><td> $- 6 . 8 3 \pm 1 . 0 3$ </td><td> $- 5 . 5 4 \pm 0 . 4 8$ </td></tr></table>

Table 7: Within-batch diversity for the eight images in Fig. 7, measured as mean pairwise cosine distance between normalized DINOv2 image embeddings. Higher is more diverse.
<table><tr><td>Method</td><td>DINO diversity ↑</td></tr><tr><td>Rao&#x27;s Quadratic Entropy</td><td>0.555</td></tr><tr><td>Entropy</td><td>0.504</td></tr><tr><td>Unguided</td><td>0.428</td></tr></table>

## L.8 EFFECT OF GUIDANCE SCALING

Some prior works (Zhu et al., 2026; Dandapanthula & Boffi, 2026) introduce the heuristic of using a guidance scaling factor γ to control the strength of the guidance. That is, the guidance is scaled by the factor $\gamma \colon \gamma g _ { t } ( y )$ . To see the effect of guidance scaling, we vary γ for empirical Doob’s htransform and observe how it influences the resulting reward. Here, we fix $\lambda = 3 0$ for ImageReward and $\lambda = 1 0 0$ for Compressibility, with $k = 4$ . We use $\gamma \in \{ 2 , 4 , 6 , 8 , 1 0 , 1 2 \}$ for ImageReward and $\gamma \in \{ 5 , 1 0 , 2 0 , 4 0 , 6 0 , 8 0 \}$ for Compressibility. Fig. 8 shows the results of varying the guidance scaling factor γ and compares them to our steepest guidance. We see that applying a guidance scaling improves the performance to some extent, but it does not outperform steepest guidance. One possible explanation is that guidance scaling may amplify a bias in the guidance, leading to suboptimal performance. This suggests that the observed performance gap in the main experiments cannot be explained solely by insufficient guidance strength

## L.9 COMPARISON WITH SALD

Here, we compare the performance of the proposed steepest guidance method with SALD (Nitanda et al., 2026). We do not apply slowdown here to reduce the computational cost. Following the original paper, we consider define f in SALD as the Gaussian-smoothed reward function, i.e., $f _ { t _ { l } } ( y ) = \mathbb { E } _ { Z \sim \mathcal { N } ( 0 , I ) } [ r ( y + \bar { \sigma } _ { t _ { l } } Z ) ]$ for $\bar { \sigma } _ { t _ { l } } = \sqrt { \Delta _ { t _ { l } } } \sigma _ { t }$ . and gradients are estimated using $\nabla f _ { t _ { l } } ( y ) \simeq$ $\begin{array} { r } { \frac { 1 } { k \bar { \sigma } _ { t _ { l } } } \sum _ { i = 1 } ^ { k } r ( y + \bar { \sigma } _ { t _ { l } } Z _ { i } ) Z _ { i } } \end{array}$ , where $Z _ { i } \sim \mathcal { N } ( 0 , I )$ independently. Then, the guidance in SALD is defined as $\lambda \sigma _ { t } ^ { 2 } \hat { \nabla } f _ { t _ { l } } ( y )$ . For a fair comparison, we subtract the leave-one-out baseline from the reward. Fig. 9 shows the results of varying the hyperparameter λ in SALD and compares them to our steepest guidance. We see that SALD cannot achieve the same level of performance as our steepest guidance method. In particular, the difference is large for ImageReward experiments. This may be because ImageReward is sensitive to fine-grained features that are lost through Gaussian smoothing.

## L.10 ABLATION STUDY ON k.

We conduct an ablation study on the number of lookahead particles k. We vary $k \in \{ 4 , 8 , 1 6 , 3 2 \}$ while keeping λ fixed. We use the same λ values as in the main experiments. Fig. 10 shows that

Rao's Quadratic Entropy

![](images/29475c5c5640ea4d2f90bebe8456d25741fac8b9a6ee2a6838db8783c6aac264.jpg)  
Figure 7: Comparison of diversity-seeking generation using SD. From top to bottom, the rows show Steepest Guidance with Rao’s quadratic entropy, entropy regularization, and the unguided baseline.

![](images/8e238a39e4db0c3e845012bd6a3ad992046bc4aa1288fce1fddebab2933e0f03.jpg)

![](images/82f242a5f0e4f1a8c7c015351ba36bc147c63f35f7bba5d0b988e6a16a19639b.jpg)  
Figure 8: Ablation study on the guidance scaling factor γ for FLUX with ImageReward and Compressibility. Average performance is shown as horizontal lines.

the performance improves as k increases, but the proposed method consistently outperforms the baselines for all k values.

![](images/8db40b5d3d654546fab3dbc5282795a59a6c52a1b2d228427b80053384e48751.jpg)

![](images/f8fd02f12f080ecae6adefaa677cbb848252edf43c22c4f80c9fecb381d6cffb.jpg)  
Figure 9: Comparison of the proposed steepest guidance method with SALD (Nitanda et al., 2026) for FLUX with ImageReward and Compressibility. Average performance is shown as horizontal lines.

![](images/cc55e7f2a1be8c9bfc31ab8e829bd2dbdbfb6d8f5a22a283ac20f82820786d9a.jpg)

![](images/35b6591f4df019fefa982ac23b8e674b957f02bbcaa0d209e66d66dba71d18ef.jpg)  
Figure 10: Ablation study on the number of lookahead particles k for FLUX with ImageReward and Compressibility. Average performance is shown as horizontal lines.