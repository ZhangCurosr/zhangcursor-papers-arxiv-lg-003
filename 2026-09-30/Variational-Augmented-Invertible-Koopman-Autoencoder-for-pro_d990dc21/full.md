# Variational Augmented Invertible Koopman Autoencoder for probabilistic time series forecasting

Anthony Frion anthony.frion@hereon.de   
Institute of coastal modelling   
Helmholtz-Zentrum Hereon   
Lucas Drumetz lucas.drumetz@imt-atlantique.fr   
IMT Atlantique   
Lab-STICC, UMR CNRS 6285, Brest, France

Guillaume Tochon guillaume.tochon@lrde.epita.fr LRE EPITA, Le Kremlin-Bicêtre, France

Mauro Dalla Mura mauro.dalla-mura@gipsa-lab.grenoble-inp.fr   
Université Grenobles Alpes   
Grenoble INP   
GIPSA-lab, Grenoble, France   
Institut Universitaire de France   
Ali Can Bekar ali.bekar@hereon.de   
Institute of coastal modelling   
Helmholtz-Zentrum Hereon   
Abdeldjalil Aïssa El Bey abdeldjalil.aissaelbey@imt-atlantique.fr   
IMT Atlantique   
Lab-STICC, UMR CNRS 6285, Brest, France

## Abstract

Neural Koopman autoencoder models have been shown to successfully build a latent embedding with linear dynamics for arbitrary dynamical systems, enabling strong performance in long-term time series forecasting. However, these models usually work in a deterministic setting, which does not allow the quantification of the uncertainty of their predictions. Thus, we propose the new Variational Augmented Invertible Koopman AutoEncoder (VAIKAE), in which the latent embedding follows a Gaussian distribution instead of being deterministic. A key property of the VAIKAE architecture is that it leverages normalizing flow models, enabling the use of likelihood computations in the state space of dynamical systems for training a model. We further propose new strategies for uncertainty-aware latent data assimilation with a trained VAIKAE model. The efectiveness of our methods is demonstrated in a series of experiments on long-term time series forecasting benchmarks.

## 1 Introduction

Eficient dynamical models are necessary to represent systems whose governing equations are either unknown (Ljung, 2010) or too costly to integrate numerically at the required resolution (Kochkov et al., 2021). Neural networks (Legaard et al., 2023) and the Koopman operator theory (Brunton et al., 2022) have both recently gained popularity for modeling dynamical systems and are often used jointly (Lusch et al., 2018), notably under the framework of Koopman autoencoders (KAEs, Nayak et al. (2025)). KAEs seek a projection from the state space using an autoencoder such that the latent dynamics are approximately linear. Established linear theory, including eficient long-horizon predictions, makes KAEs attractive for modeling dynamical systems. However, most of the existing KAE models are deterministic, and we argue that modeling dynamical systems in the presence of observation and process noise calls for a stochastic formulation.

We consider a discrete autonomous dynamical system with an unknown one-time-step evolution function $\mathcal { M } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ . One can seek to characterize such a dynamical system by using only some available observations of its state over time. Although many other configurations are possible, we will assume that observations $\mathbf { y } _ { t } \in \mathbb { R } ^ { n }$ are full but noisy measurements of the state $\mathbf { x } _ { t } \in \mathbb { R } ^ { n }$ , yielding

$$
\forall t \in \mathbb { N } , \quad \mathbf { x } _ { t + 1 } = \mathcal { M } ( \mathbf { x } _ { t } ) + \epsilon _ { t } , \quad \epsilon _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \Sigma _ { \epsilon } ) ,\tag{1}
$$

$$
\forall t \in [ t _ { 0 } , . . . , t _ { T } ] , \quad \mathbf { y } _ { t } = \mathbf { x } _ { t } + \eta _ { t } , \quad \eta _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \Sigma _ { \eta } ) ,\tag{2}
$$

where $\epsilon _ { t }$ and $\pmb { \eta } _ { t }$ are, respectively, process and observation noise vectors, which are generally assumed to follow zero-mean Gaussian distributions with time-independent covariance matrices $\Sigma _ { \epsilon }$ and $\Sigma _ { \eta }$ . We assume that observations are not available for every time index but only for a specific set $[ t _ { 0 } , . . . , t _ { T } ]$

Building an approximation $\hat { \mathcal { M } }$ of the unknown state dynamics $\mathcal { M }$ using an observation set $\left( \mathbf { y } _ { t _ { 0 } } , . . . , \mathbf { y } _ { t _ { T } } \right)$ is a ubiquitous problem in machine learning (Raissi et al., 2019; Li et al., 2021) and Koopman operator applications (Brunton et al., 2022), known as system identification. It enables prediction of the evolution of the state $\mathbf { x } _ { t }$ from any given initial condition. Having access to a surrogate M<sup>ˆ</sup> also facilitates data assimilation (Carrassi et al., 2018), which consists of combining a dynamical model with an observation set to estimate the posterior distribution of the state over time, i.e. $P ( \mathbf x _ { t } | \mathbf y _ { t _ { 0 } } , . . . , \mathbf y _ { t _ { T } } )$ . In many cases, the observations $\mathbf { y } _ { t }$ are assumed to be noiseless, i.e. $\eta _ { t } = 0$ in equation 2. Many existing system identification methods seek a deterministic approximation of equation 1, yet this problem has two sources of uncertainty: first, the state dynamics is inherently noisy (unless $\epsilon _ { t } = 0 )$ , and thus even a perfect deterministic estimate of the dynamics $\hat { \mathcal { M } } = \mathcal { M }$ does not fully characterize the evolution of the state $\mathbf { x } _ { t }$ . This corresponds to the aleatoric uncertainty. In addition, the performance of system identification can be limited by other factors such as the quality and quantity of the available observations as well as the capacity of the model used to compute $\hat { \mathcal { M } }$ . This corresponds to the epistemic uncertainty. A thorough discussion of the sources of uncertainty can be found in Haynes et al. (2023). In particular, we have used the original mathematical definition of aleatoric and epistemic uncertainties, but the definitions may vary in the machine learning community. Stochastic models can estimate uncertainties and address additional tasks such as anomaly detection (Pang et al., 2021) and change point detection (Truong et al., 2020). $\mathrm { B y }$ fully approximating equation 1, they also facilitate uncertainty-aware data assimilation. Ideally, these models should be well calibrated, meaning their estimated uncertainties should match their error statistics, as can be assessed synthetically with the spread-skill ratio or more comprehensively with a spread-skill plot (Haynes et al., 2023).

The rest of this manuscript is organized as follows: in section 2, we review related work in Koopman operator theory, probabilistic time series forecasting and data assimilation. In section 3, we present VAIKAE: a new stochastic KAE architecture which, to our knowledge, is the first to enable explicit state likelihood computations. We then outline strategies for training a model and using it for uncertainty-aware data assimilation. In section 4, we present the results of VAIKAE on a time series forecasting benchmark where the input is a complete historical time series. In section 5, we experiment on a satellite image time series benchmark where the input observations are irregularly sampled in time, and showcase the efectiveness and good calibration of our data assimilation methods in this setting. Section 6 concludes our work.

## 2 Background

## 2.1 Koopman operator theory

The Koopman operator theory, first described by Koopman (1931), has been extensively used in the last few decades for data-driven analysis of nonlinear dynamical systems, following the work of Mezić (2005). It states that any nonlinear dynamical system can be described by a linear operator acting on its measurement functions, which is infinite-dimensional in the general case. Concretely, we consider a discrete dynamical operator $\mathcal { M } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ following equation 1 with no process noise, i.e. $\epsilon _ { t } = 0$ . The Koopman operator K of $\mathcal { M }$ is such that, for any measurement function $g : \mathbb { R } ^ { n }  \mathbb { R }$ and for any state $\mathbf { x } _ { t } \in \mathbb { R } ^ { n }$ at an arbitrary time $t ,$

$$
\begin{array} { r } { \mathcal { K } g ( \mathbf { x } _ { t } ) \triangleq g \circ \mathcal { M } ( \mathbf { x } _ { t } ) = g ( \mathbf { x } _ { t + 1 } ) . } \end{array}\tag{3}
$$

Since this is true for any input state $\mathbf { x } _ { t } ,$ one can simply write $\mathcal { K } g = g \circ \mathcal { M }$ . The Koopman operator is linear but dificult to define for nonlinear dynamics M due to the infinite dimensionality of its input function space. Many recent methods seek to find finite-dimensional approximations of the Koopman operator K to model $\mathcal { M }$ . In a general framework, such an approximation can be characterized by three components: a square matrix $\mathbf { K } \in \mathbb { R } ^ { d \times d }$ , an embedding function $\Phi : \mathbb { R } ^ { n }  \mathbb { R } ^ { d }$ and a decoding function $\psi : \mathbb { R } ^ { d }  \mathbb { R } ^ { n }$ . Φ embeds a state $\mathbf { x } _ { t } \in \mathbb { R } ^ { n }$ expressed in the natural basis of the dynamical system using a set of d measurement functions, yielding a vector $\mathbf { z } _ { t } = \Phi ( \mathbf { x } _ { t } ) \in \mathbb { R } ^ { d }$ . This vector is multiplied by K in order to get the latent embedding $\mathbf { z } _ { t + 1 }$ which can be decoded back to the state space using ψ. This process is summarized by

$$
\mathbf { x } _ { t + \tau } \approx \hat { \mathbf { x } } _ { t + \tau } = \psi ( \mathbf { K } ^ { \tau } \Phi ( \mathbf { x } _ { t } ) )\tag{4}
$$

for any chosen prediction time $\tau > 0$ . Formally, K approximates the restriction of the infinite-dimensional Koopman operator $\kappa$ on the set of d measurement functions represented by Φ. Thus, a fundamental assumption of this approach is that Φ (approximately) defines a Koopman invariant subspace (Brunton et al., 2016), i.e. a set of measurement functions that is stable by application of the Koopman operator.

Many practical approaches have been proposed for obtaining Φ, as reviewed by Brunton et al. (2022). An early and popular method is dynamic mode decomposition (DMD, Schmid (2010)), which defines Φ and ψ as identity functions. This means assuming that the set of natural measurement functions consisting of projections of $\mathbf { x } \in \mathbb { R } ^ { n }$ to its n variables is approximately Koopman invariant, i.e. that the dynamical system under study is approximately linear. The matrix K is then estimated from a dataset of consecutive system states. Extended dynamic mode decomposition (eDMD, Williams et al. (2015)) generalizes DMD by using a hand-designed Φ that includes the natural measurement functions. $\psi$ is then obtained by projecting the latent embedding onto its n first variables. eDMD converges to the Koopman operator as its latent embedding size d grows to infinity (Korda & Mezić, 2018), yet obtaining good practical performance often requires physical insight on the studied dynamical system and a large latent dimension $d .$ Thus, many subsequent works have used neural autoencoders for learning Φ and $\psi$ as encoding and decoding functions.

The Koopman autoencoder (KAE) models generally define Φ, ψ and K as three learnable components. We distinguish two desirable properties for such models:

• The learned embedding Φ should be as close to Koopman invariant as possible, i.e. $\mathbf { K } \Phi ( \mathbf { x } _ { t } )$ ≈ $\Phi ( \mathbf { x } _ { t + 1 } )$ . This criterion is linked to the expressivity of Φ and to the size d of the latent embedding.

• The learned embedding should be as close to invertible as possible, i.e. $\psi \circ \Phi ( \mathbf { x } _ { t } ) \approx \mathbf { x } _ { t }$ . Ideally, this reconstruction should be analytically exact.

A model that perfectly respects these two properties would perfectly represent the state dynamics. Early KAE models (Lusch et al., 2018; Otto & Rowley, 2019; Li et al., 2020; Azencot et al., 2020) train two neural networks for Φ and $\psi$ with several loss terms including a reconstruction loss so that $\psi \circ \Phi$ is close to the identity function. This should satisfy the first desired criterion if the model for Φ is expressive enough and uses a large latent size $d .$ But it inevitably leads to a reconstruction error with the composition $\psi \circ \Phi$ Thus, more recent works (Meng et al., 2024; Jin et al., 2024; Hou et al., 2024) use analytically invertible neural architectures, such as normalizing flows (Kobyzev et al., 2020), so that $\psi = \Phi ^ { - 1 }$ , ensuring an exact reconstruction of the input state from its embedding. A drawback of this choice is that the learned embedding must match the state dimension, i.e. $d = n$ , whereas a larger embedding is beneficial for the first criterion of finding a Koopman invariant subspace. To address this issue, Meng et al. (2024); Jin et al. (2024) pad the embedding with zeros, yet this approach still lacks expressivity. Instead, Frion et al. (2025); Lupascu et al. (2026) augment the learned invertible embedding with a second encoder that has no invertibility constraint. This enables greater representational power to learn a Koopman invariant subspace while still guaranteeing an exact reconstruction of the input state. However, all methods mentioned so far are restricted to deterministic prediction, and thus we now turn our attention to probabilistic methods.

## 2.2 Stochastic time series forecasting

Stochastic rather than deterministic models can capture complex conditional or unconditional probability distributions (Sengar et al., 2025) or quantify predictive uncertainty (Haynes et al., 2023). Ensemble averages also improve deterministic metrics such as root mean squared error over a single prediction (Milinski et al., 2020). In practice, ensembling extended the horizon of skillful predictions for chaotic systems such as the global atmosphere (Nathaniel et al., 2024), which lead to a large shift towards stochastic neural models (Oskarsson et al., 2024; Price et al., 2025; Alet et al., 2025; Lang et al., 2026; Agarwal et al., 2026). Simple and successful techniques to obtain stochasticity with generic neural network architectures include Monte Carlo dropout (Gal & Ghahramani, 2016), stochastic weight averaging (Izmailov et al., 2018), Bayesian neural networks (Jospin et al., 2022) and model ensembling (Lakshminarayanan et al., 2017; Frion et al., 2024b).

Difusion models (Yang et al., 2023) are gaining increasing popularity for probabilistic time series forecasting (Liao et al., 2026). TimeGrad (Rasul et al., 2021) leverages a difusion model that estimates the distribution of the state at each time step based on the hidden state of a recurrent neural network. CSDI (Tashiro et al., 2021) uses a score-based difusion model conditioned on observed data, and can solve multiple tasks including time series imputation and long-term forecasting. TimeDif (Shen & Kwok, 2023) builds a conditioning signal that combines future mixup (inspired by the mixup from Zhang et al. (2018)) and autoregressive initialization. TMDM (Li et al., 2024) builds on the NSFormer architecture (Liu et al., 2022) and minimizes an evidence lower bound (ELBO) loss to estimate the posterior distribution of the time series given an input history. The authors of D<sup>3</sup>U (Li et al., 2025) first train a deterministic model to learn the conditional mean, and then train a DDPM (Ho et al., 2020) to model the probabilistic part of the prediction.

Another line of work relies on the Koopman operator theory. The authors of Pan & Duraisamy (2020) implement the trainable components of a KAE architecture as Bayesian neural networks (Jospin et al., 2022), enabling probabilistic outputs. DeSKO (Han et al., 2022) uses an encoder that outputs the mean and variance of a latent diagonal Gaussian distribution for uncertainty-aware model predictive control. KoVAE (Naiman et al., 2024) is a variational autoencoder model that projects each latent sequence to its best linear fit using DMD. Deep Probabilistic Koopman (Mallen et al., 2024) models probability distributions from which the parameters follow a quasi-periodic evolution in time. KooNPro (Zheng et al., 2025), inspired by Lusch et al. (2018), designs a KAE with an auxiliary model that estimates a Gaussian probability distribution for the spectrum of the latent dynamics, and derives an associated ELBO criterion (Garnelo et al., 2018).

## 2.3 Data assimilation with automatic diferentiation and neural networks

Data assimilation (Carrassi et al., 2018) is a Bayesian framework that leverages both a dynamical model M and a set of imperfect observations $\mathbf { y } _ { t }$ to reconstruct the full state of a system $\mathbf { x } _ { t }$ over time. Its joint use with machine learning is a rich and ongoing field of study (Bocquet, 2023; Cheng et al., 2023). In particular, variational data assimilation, consisting of gradient descent on a Bayesian maximum a posteriori cost, classically requires the complex hand-derivation of an adjoint model of the dynamics. This derivation can be avoided by implementing the dynamics in an automatic diferentiation framework such as PyTorch (Paszke et al., 2017) or JAX (Bradbury et al., 2018), as discussed by e.g. Gelbrecht et al. (2023); Sapienza et al. (2024); Frion et al. (2026). Alternatively, one can use a neural emulator that approximates the true dynamics and is diferentiable by design (Nonnenmacher & Greenberg, 2021; Hatfield et al., 2021).

Following this second approach, latent data assimilation consists of solving a data assimilation problem in the latent space of a trained neural emulator. It can rely on various classical data assimilation methods, e.g. ensemble-based (Peyron et al., 2021) or variational (Melinc & Zaplotnik, 2024) methods. The motivations for resorting to latent data assimilation include working in a lower-dimensional space than the physical state space to reduce the computational cost (Peyron et al., 2021; Melinc & Zaplotnik, 2024) and building a non-Gaussian prior distribution (Pasmans et al., 2026; Fan et al., 2026). Most related to the present work, some methods (Frion et al., 2024a; Shoji et al., 2025; Frion et al., 2025; Tong et al., 2026) perform data assimilation in the latent space of a KAE model to benefit from linear latent dynamics.

![](images/86ecb656c8d0f5cab25f1316d4c2b8fc5ecd72cf2fcd190b63d03a8e1955ada3.jpg)  
Figure 1: Graphical representation of the VAIKAE architecture. The two dashed lines represent sampling from a diagonal Gaussian distribution, while the full lines represent deterministic operations.

## 3 Proposed methods

## 3.1 A new Koopman autoencoder model architecture: VAIKAE

We base our new KAE model on the recently proposed Augmented Invertible Koopman AutoEncoder (AIKAE, Frion et al. (2025)). A visual representation of the AIKAE architecture is shown in appendix A. It comprises 3 learnable components: an invertible encoder $\phi : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ (implemented as a normalizing flow), a Koopman matrix $\mathbf { K } \in \mathbb { R } ^ { d \times d }$ and an augmentation encoder $\chi : \mathbb { R } ^ { n }  \mathbb { R } ^ { p }$ . In this model, the latent Koopman embedding $\Phi ( { \bf x } _ { t } )$ of $\mathbf { x } _ { t } \in \mathbb { R } ^ { n }$ is defined as the concatenation of $\mathbf { z } _ { t } ^ { i } = \phi ( \mathbf { x } _ { t } )$ and $\mathbf { z } _ { t } ^ { a } = \chi ( \mathbf { x } _ { t } )$ , so that the model has an unrestricted latent dimension $d = n + p$ while still exactly reconstructing the input state using the analytical inverse $\phi ^ { - 1 }$ of $\phi$ . We respectively use superscripts ·<sup>i</sup> and ·<sup>a</sup> to represent the invertible (first n components) and augmentation (last p components) parts of a vector in $\mathbb { R } ^ { d }$

Here, we extend the AIKAE framework to a Variational Augmented Invertible Koopman AutoEncoder (VAIKAE), which substitutes the deterministic latent embedding of the AIKAE with a probabilistic one. Concretely, the embedding of a state $\mathbf { x } _ { t } \in \mathbb { R } ^ { n }$ by the VAIKAE is a diagonal Gaussian distribution<sup>1</sup> defined by its mean $\mu _ { t } \in \mathbb { R } ^ { d }$ and (diagonal) covariance $\pmb { \sigma } _ { t } \in \mathbb { R } ^ { d }$ . These mean and covariance are obtained by repurposing the augmentation encoder $\chi$ so that it additionally outputs the variance coeficients, leading to:

$$
\chi ( \mathbf { x } _ { t } ) = \binom { \mu _ { t } ^ { a } } { \pmb { \sigma } _ { t } } .\tag{5}
$$

Thus, the output of $\chi : \mathbb { R } ^ { n }  \mathbb { R } ^ { p + d }$ is decomposed into 2 parts: the first p components represent the mean $\pmb { \mu } _ { t } ^ { a }$ of the augmentation part of the encoding and the last d components represent the diagonal covariance of the global latent embedding. The invertible encoder $\phi$ keeps an analogous role as in AIKAE, here outputting $\mu _ { t } ^ { i } = \phi ( { \mathbf x } _ { t } )$ . The VAIKAE architecture is summarized in figure 1.

Another design choice would have been to define a third encoder to estimate the latent diagonal variance coeficients $\sigma _ { t }$ while $\phi$ and χ produce the means of the invertible and augmentation parts of the latent embedding. However, we instead let $\chi$ additionally learn the variance coeficients, thus sharing weights with the mean $\pmb { \mu } _ { t } ^ { a }$ of the augmentation encoding. One advantage of this is that VAIKAE requires only few additional parameters (all in the output layer of $\chi )$ relative to its deterministic AIKAE counterpart.

## 3.2 Training a VAIKAE

In this section, we show that the VAIKAE framework enables us to explicitly evaluate the predicted probability density of a state value given an earlier observed value. This mathematical derivation is then used to design a training criterion for obtaining calibrated stochastic predictions with the model.

We work with the assumptions of equations 1 and 2, where $\mathbf { y } _ { t } \in \mathbb { R } ^ { n }$ is a noisy observation of the true state $\mathbf { x } _ { t } \in \mathbb { R } ^ { n }$ at time t. Thus, assuming here that we only have access to $\mathbf { y } _ { 0 }$ at time 0, our objective is to characterize the posterior probability distribution $P _ { \mathbf { x } _ { t } } ( \cdot | \mathbf { y } _ { 0 } )$ , for any prediction time $t \geq 0$ . We start from the initial latent embedding’s probability distribution $P _ { \mathbf { z } _ { 0 } } ( \cdot | \mathbf { y } _ { 0 } ) \sim \mathcal { N } ( \pmb { \mu } _ { 0 } , \pmb { \Sigma } _ { 0 } = \mathrm { d i a g } ( \pmb { \sigma } _ { 0 } ) )$ , where the conditional mean $\pmb { \mu } _ { \mathrm { 0 } }$ and variance coeficients $\pmb { \sigma } _ { 0 }$ are respectively obtained with

$$
{ \displaystyle { \pmb \mu } _ { 0 } = \left( \begin{array} { c } { { \phi ( { \bf y } _ { 0 } ) } } \\ { { \chi ( { \bf y } _ { 0 } ) _ { 1 : p } } } \end{array} \right) } ,\tag{6}
$$

$$
\pmb { \sigma } _ { 0 } = \chi ( \mathbf { y } _ { 0 } ) _ { p + 1 : p + d } .\tag{7}
$$

From here on, all subsequent latent embeddings $\mathbf { z } _ { t }$ are linearly related to $\mathbf { z } _ { 0 }$ through $\mathbf { z } _ { t } = \mathbf { K } ^ { t } \mathbf { z } _ { 0 }$ . Thus, when conditioned on $\mathbf { y } _ { 0 } ,$ , they also follow Gaussian distributions as:

$$
P _ { \mathbf { z } _ { t } } ( \cdot | \mathbf { y } _ { 0 } ) \sim \mathcal { N } ( \mu _ { t } , \Sigma _ { t } )\tag{8}
$$

with mean $\pmb { \mu } _ { t } = \mathbf { K } ^ { t } \pmb { \mu } _ { 0 }$ , covariance $\begin{array} { r } { \Sigma _ { t } = \mathbf { K } ^ { t } \Sigma _ { 0 } ( \mathbf { K } ^ { t } ) ^ { \intercal } } \end{array}$ , and ·<sup>⊺</sup> denoting matrix transposition. Importantly, while $\Sigma _ { 0 }$ is a diagonal matrix, $\Sigma _ { t }$ is likely to be a full covariance matrix when K is not diagonal.

From this point, one can use the properties of the normalizing flow $\phi$ to evaluate the probability density function of $\mathbf { x } _ { t }$ given the distribution of $\mathbf { z } _ { t } = \phi ( \mathbf { x } _ { t } )$ , and ultimately evaluate the likelihood given the observed value $\mathbf { y } _ { 0 }$ . We apply the following change of variable formula (discussed in $\mathrm { e . g }$ . Dinh et al. (2014)):

$$
P _ { { \bf x } _ { t } } ( { \bf x } ) = P _ { { \bf z } _ { t } ^ { i } } ( \phi ( { \bf x } ) ) \left| \operatorname* { d e t } \frac { \partial \phi ( { \bf x } ) } { \partial { \bf x } } \right| .\tag{9}
$$

Concerning the first factor $P _ { \mathbf { z } _ { t } ^ { i } } ( \phi ( \mathbf { x } ) )$ , it should be noted that $ { \mathbf { z } } _ { t } ^ { i }$ is a marginal distribution of $\mathbf { z } _ { t } .$ , and thus $ { \mathbf { z } } _ { t } ^ { i }$ is also Gaussian when $\mathbf { z } _ { t }$ is Gaussian (Bishop & Nasrabadi, 2006). Its moments can be obtained by taking the first n components $\mu _ { t } ^ { i }$ of the mean and the upper-left $n \times n$ block $\Sigma _ { t } ^ { i }$ of the covariance of $\mathbf { z } _ { t }$ . Besides, the determinant of the Jacobian matrix $\frac { \partial \phi ( \mathbf { x } ) } { \partial \mathbf { x } }$ is not easy to obtain in general, yet $\phi$ is here a normalizing flow model, and is therefore specifically designed to have a tractable and easily computable Jacobian. When conditioning equation 9 on an observed value of $\mathbf { y } _ { 0 }$ and injecting equation $^ { 8 , }$ we obtain:

$$
P _ { \mathbf { x } _ { t } } ( \mathbf { x } | \mathbf { y } _ { 0 } ) = P _ { \mathbf { z } _ { t } ^ { i } } ( \phi ( \mathbf { x } ) | \mathbf { y } _ { 0 } ) \left| \operatorname* { d e t } { \frac { \partial \phi ( \mathbf { x } ) } { \partial \mathbf { x } } } \right| = { \mathcal { N } } ( \phi ( \mathbf { x } ) ; \mu _ { t } ^ { i } , \Sigma _ { t } ^ { i } ) \left| \operatorname* { d e t } { \frac { \partial \phi ( \mathbf { x } ) } { \partial \mathbf { x } } } \right| .\tag{10}
$$

As a practical training criterion, one can use this equation to maximize the likelihood $P _ { \mathbf { x } _ { t } } ( \mathbf { y } _ { t } | \mathbf { y } _ { 0 } )$ of subsequent observations $\mathbf { y } _ { t }$ when predicting from an input observation $\mathbf { y } _ { 0 }$ . Importantly, this strategy only accounts for the marginal distributions on the state variables at each time step, and thus does not ensure coherent predicted trajectories. $\mathrm { Y e t }$ , we can still obtain temporally coherent trajectories by drawing samples from $P _ { \mathbf { z } _ { 0 } } ( \cdot | \mathbf { y } _ { 0 } )$ and then deterministically propagating them to generate one trajectory from each of these samples, rather than independently drawing samples from the marginal distributions at each time step.

We now describe a complete strategy for training the VAIKAE model. Let us assume, for simplicity, that the training dataset is composed of $N$ sets of observations $\mathbf { Y } _ { 1 } , . . . , \mathbf { Y } _ { N }$ such that, for any $1 \leq i \leq N$ $\mathbf Y _ { i } = ( \mathbf y _ { i , t _ { i , 0 } } , . . . , \mathbf y _ { i , t _ { i , T _ { i } } } )$ with $0 = t _ { i , 0 } < . . . < t _ { i , T _ { i } }$ representing the indices where observations are available. The individual sets of observations are usually overlapping slices of longer sets of observations. We define θ as the concatenation of the coeficients of K and the trainable parameters of $\phi$ and $\chi$ . We take inspiration from the loss function of the deterministic AIKAE model to design three loss function terms:

• A prediction loss $\begin{array} { r } { L _ { p r e d } ( \theta ) = \sum _ { i = 1 } ^ { N } \sum _ { \tau = 0 } ^ { T _ { i } } \mathbb { E } _ { \mathbf { x } \sim P _ { \mathbf { x } _ { i , t _ { i } , \mathbf { \tau } } } ( \cdot | \mathbf { y } _ { i , 0 } ) } | | \mathbf { x } - \mathbf { y } _ { i , t _ { i , \tau } } | | _ { 2 } ^ { 2 } } \end{array}$ consisting in the mean squared error between sampled predictions obtained from equation 4 and the corresponding true states.

• A linearity loss $\begin{array} { r } { L _ { l i n } ( \theta ) = \sum _ { i = 1 } ^ { N } \sum _ { \tau = 0 } ^ { T _ { i } } \mathbb { E } _ { \mathbf { z } \sim P _ { \mathbf { z } _ { i , t _ { i } } } } { ( \cdot | \mathbf { y } _ { i , 0 } ) } | | \mathbf { z } - \phi ( \mathbf { y } _ { i , t _ { i , \tau } } ) | | _ { 2 } ^ { 2 } } \end{array}$ , which is meant to ensure that the latent embeddings of observations of the same state over time are truly linearly related.

• An orthogonality loss $L _ { o r t h } ( \theta ) = | | \mathbf { K } \mathbf { K } ^ { \intercal } - \mathbf { I } _ { d } | | _ { 2 } ^ { 2 }$ , which ensures that the eigenvalues of the learned K remain close to the unit circle to obtain stable dynamics.

These loss terms are studied in a deterministic setting in Frion et al. (2024a). While $L _ { o r t h }$ was shown to promote long-term stability, K could also be constrainted to be unitary by construction (Zhang et al., 2024).

In practice, the expected values in $L _ { p r e d }$ and $L _ { l i n }$ are approximated by sampling from the latent Gaussian distribution $P _ { \mathbf { z } _ { i , 0 } } ( \cdot | \mathbf { y } _ { i , 0 } )$ and then advancing these samples deterministically to obtain samples from all relevant variables. However, using a loss function based only on these terms would result in a collapse of the learned variance vectors $\pmb { \sigma } _ { 0 }$ to 0, reducing to deterministic predictions. To force some stochasticity into the model, we add the new likelihood loss function $L _ { l k l }$ , which is directly derived from equation 10 as:

$$
L _ { l k l } ( \theta ) = \sum _ { i = 1 } ^ { N } \sum _ { \tau = 0 } ^ { T _ { i } } - \log ( P _ { \mathbf { x } _ { i , t _ { i , \tau } } } ( \mathbf { y } _ { i , t _ { i , \tau } } | \mathbf { y } _ { i , 0 } ) ) .\tag{11}
$$

Since the negative log-likelihood is a proper scoring rule (see appendix section C.2), it should favor wellcalibrated predictions. Finally, one can now construct the global loss function for VAIKAE:

$$
L ( \theta ) = L _ { p r e d } ( \theta ) + \alpha L _ { l i n } ( \theta ) + \beta L _ { o r t h } ( \theta ) + \gamma L _ { l k l } ( \theta ) ,\tag{12}
$$

where $\alpha , \beta , \gamma$ are relative weights. Depending on the application, one may set $\alpha = 0$ and/or $\beta = 0 ,$ yet $L _ { p r e d }$ is always present as the main loss term, while $L _ { l k l }$ is necessary to obtain stochastic predictions in practice.

## 3.3 Performing data assimilation in the latent space of a trained VAIKAE

Here, we consider the problem of using multiple observations $\mathbf { y } _ { t }$ to make stochastic predictions, in the data assimilation context from section 2.3. Given an observations set $\left( \mathbf { y } _ { t _ { 0 } } , . . . , \mathbf { y } _ { t _ { T } } \right)$ , Frion et al. (2025) solves a strong-constraint 4D-Var problem in the latent space of an AIKAE model, which can be written as

$$
\mathbf { z } _ { * } = \underset { \mathbf { z } _ { 0 } \in \mathbb { R } ^ { d } } { \arg \operatorname* { m i n } } \sum _ { \tau = 0 } ^ { T } | | \boldsymbol { \phi } ^ { - 1 } ( \mathbf { K } ^ { t _ { \tau } } \mathbf { z } _ { 0 } ) - \mathbf { y } _ { t _ { \tau } } | | ^ { 2 } ,\tag{13}
$$

z<sub>∗</sub> can then be used to produce predictions at any time $t \geq t _ { 0 }$ . Equation 13 can be conveniently solved with automatic diferentiation. Crucially, one can use $\Phi ( \mathbf { y } _ { t _ { 0 } } )$ as the initial guess that solves equation 13. Due to the general non-convex nature of the problem, this initialization both reduces the number of gradient steps needed for convergence and improves the quality of the end result.

In a VAIKAE, Φ predicts not only a single value of $\mathbf { z } _ { 0 }$ but the mean and variance of a latent diagonal Gaussian distribution. Equation 13 can thus be adapted for stochastic predictions by solving 4D-Var on the mean and then decoding this mean to estimate an associated uncertainty with $\chi .$ . This can be written as:

$$
\pmb { \mu } _ { * } = \underset { \pmb { \mu } _ { 0 } \in \mathbb { R } ^ { d } } { \arg \operatorname* { m i n } } \sum _ { \tau = 0 } ^ { T } | | \phi ^ { - 1 } ( \mathbf { K } ^ { t _ { \tau } } \pmb { \mu } _ { 0 } ) - \mathbf { y } _ { t _ { \tau } } | | ^ { 2 } ,\tag{14}
$$

$$
\pmb { \sigma } _ { \ast } = \chi ( \phi ^ { - 1 } ( \pmb { \mu } _ { \ast } ^ { i } ) ) _ { p + 1 : p + d } .\tag{15}
$$

As a reminder, $\mu _ { * } ^ { i }$ is the invertible part of $\pmb { \mu } _ { \ast }$ , i.e. its first n components. This method can provide uncertainty quantification, yet it will likely be under-confident since using multiple observations should significantly improve the skill of the prediction with regard to predictions from a single observation, while retaining the same spread. Alternatively, one can reformulate the original formulation of 4D-Var to solve for the parameters of a Gaussian distribution instead of a point estimate. We propose to minimize the continuous ranked probability score (CRPS), which is a proper scoring rule and performs strongly on fitting forecasting ensembles: see e.g. Gneiting & Raftery (2007) and appendix C.2. Concretely, we solve:

$$
\mu _ { * } , \pmb { \sigma } _ { * } = \arg \operatorname* { m i n } _ { \pmb { \mu } _ { 0 } , \pmb { \sigma } _ { 0 } } \sum _ { \tau = 0 } ^ { T } \mathrm { C R P S } ( \mathbf { x } _ { t _ { \tau } } , \mathbf { y } _ { t _ { \tau } } ) .\tag{16}
$$

Table 1: Summary of probabilistic long-term time series forecasting results. For each dataset and metric, the best result is in bold and the second best result is underlined.
<table><tr><td rowspan=1 colspan=2>Model</td><td rowspan=1 colspan=1>VAIKAE</td><td rowspan=1 colspan=1> $\overline { { \mathbf { D } ^ { 3 } \mathbf { U } } }$ </td><td rowspan=1 colspan=1>TMDM</td><td rowspan=1 colspan=1>TimeDiff</td><td rowspan=1 colspan=1>CSDI</td><td rowspan=1 colspan=1>TimeGrad</td></tr><tr><td rowspan=3 colspan=1>ETTTm1</td><td rowspan=3 colspan=1>MSEMAECRPS</td><td rowspan=3 colspan=1>0.3700.3850.299</td><td rowspan=1 colspan=1>0.363</td><td rowspan=1 colspan=1>0.607</td><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>0.867</td><td rowspan=3 colspan=1>1.7161.0570.665</td></tr><tr><td rowspan=1 colspan=1>0.386</td><td rowspan=2 colspan=1>0.5580.429</td><td rowspan=2 colspan=1>0.5770.454</td><td rowspan=2 colspan=1>0.6900.773</td></tr><tr><td rowspan=1 colspan=1>0.285</td></tr><tr><td rowspan=1 colspan=1>7TTT2</td><td rowspan=1 colspan=1>MSEMAECRPS</td><td rowspan=1 colspan=1>0.2530.3190.247</td><td rowspan=1 colspan=1>0.2410.3020.243</td><td rowspan=1 colspan=1>0.5240.4930.380</td><td rowspan=1 colspan=1>0.2840.3420.316</td><td rowspan=1 colspan=1>1.2910.5760.625</td><td rowspan=1 colspan=1>1.3850.7320.785</td></tr><tr><td rowspan=2 colspan=1>Weaatkr</td><td rowspan=2 colspan=1>MSEMAECRPS</td><td rowspan=2 colspan=1>0.2080.2540.199</td><td rowspan=2 colspan=1>0.2220.2640.207</td><td rowspan=2 colspan=1>0.2440.2860.226</td><td rowspan=1 colspan=1>0.277</td><td rowspan=2 colspan=1>0.8420.5230.508</td><td rowspan=2 colspan=1>0.8850.5510.482</td></tr><tr><td rowspan=1 colspan=1>0.3310.293</td></tr><tr><td rowspan=1 colspan=1>So0ar</td><td rowspan=1 colspan=1>MSEMAECRPS</td><td rowspan=1 colspan=1>0.2330.2880.235</td><td rowspan=1 colspan=1>0.2370.2700.186</td><td rowspan=1 colspan=1>0.2950.3170.375</td><td rowspan=1 colspan=1>1.1690.9360.900</td><td rowspan=1 colspan=1>0.8480.8180.649</td><td rowspan=1 colspan=1>1.2111.0040.783</td></tr><tr><td rowspan=1 colspan=1>ETL</td><td rowspan=1 colspan=1>MSEMAECRPS</td><td rowspan=1 colspan=1>0.1720.2640.199</td><td rowspan=1 colspan=1>0.1790.2670.202</td><td rowspan=1 colspan=1>0.2220.3290.446</td><td rowspan=1 colspan=1>0.7300.6900.475</td><td rowspan=1 colspan=1>0.5530.7950.465</td><td rowspan=1 colspan=1>0.6450.7230.503</td></tr><tr><td rowspan=2 colspan=1>Trrac</td><td rowspan=2 colspan=1>MSEMAECRPS</td><td rowspan=2 colspan=1>0.4520.3050.243</td><td rowspan=2 colspan=1>0.4680.2990.232</td><td rowspan=2 colspan=1>0.7210.4110.552</td><td rowspan=1 colspan=1>1.465</td><td rowspan=1 colspan=1>0.921</td><td rowspan=2 colspan=1>0.9320.8070.657</td></tr><tr><td rowspan=1 colspan=1>0.8510.671</td><td rowspan=1 colspan=1>0.6780.612</td></tr></table>

Note that, in practice, we perform each gradient descent step by drawing multiple Monte Carlo samples from ${ \bf z } _ { 0 } \sim \mathcal { N } ( \pmb { \mu } _ { 0 } , \mathrm { d i a g } ( \pmb { \sigma } _ { 0 } ) )$ with the reparameterization trick from Kingma & Welling (2013) and advancing each of them deterministically to obtain samples from $\big ( \mathbf { x } _ { t _ { 0 } } , . . . , \mathbf { x } _ { t _ { T } } \big )$ conditioned on $\mathbf { z } _ { 0 }$ . We then use the sample-based computation of the CRPS with equation 20. As detailed in appendix C.2, we average the CRPS obtained for all individual variables of the state. Analogously to the AIKAE-based assimilation in equation 13, one can use the pre-trained VAIKAE to obtain initial guesses for both the initial mean and variance in equation 16, which enables a faster convergence and a better end result. While minimizing the cost of equation 16 means that we no longer benefit from the rigorous Bayesian maximum a posteriori formalism of 4D-Var, we will show in section 5.2 that it still enables us to obtain well-calibrated posterior distribution estimates.

## 4 Probabilistic long-term time series forecasting

Here, we test the performance of our VAIKAE model on the popular "Informer benchmark", named after a method that popularized it (Zhou et al., 2021). This benchmark consists of a set of time series datasets, from which one usually extracts a set of L consecutive time steps and uses it as an input to predict the state over the following T time steps. While many methods leverage the joint information of all variables in the input time series (Zhou et al., 2021; 2022; Zhang & Yan, 2023; Liu et al., 2024), some recent methods have obtained strong performance by considering the variables of the time series independently from each other (Zeng et al., 2023; Nie et al., 2023; Frion et al., 2025), and we adopt this second approach.

Although multiple methods (e.g. Zhou et al. (2022); Zhang & Yan (2023); Zeng et al. (2023); Nie et al. (2023); Liu et al. (2024); Frion et al. (2025)) have been proposed to perform long-term forecasting on this benchmark in a deterministic setup, evaluated solely through their mean squared error (MSE) and mean average error (MAE), we here focus on probabilistic models, which are evaluated with additional metrics such as the CRPS. A general presentation of our metrics can be found in appendix C.

We use the setup of Li et al. (2025), where the input length is $L = 9 6$ and the prediction length is $T = 1 9 2$ Similar to Frion et al. (2025), we use a delay embedding of the input time series, which means that the input size is $n = L = 9 6$ . The size of the augmentation encoding is set to $p = 3 2$ , so that the global latent space of the model is of size $d = n + p = 1 2 8$ . Further implementation details can be found in appendix D.1. Our baselines are five recent and competitive methods based on difusion models: TimeGrad (Rasul et al., 2021), CSDI (Tashiro et al., 2021), TimeDif (Shen & Kwok, 2023), TMDM (Li et al., 2024) and $\mathrm { D ^ { 3 } U }$ (Li et al., 2025). These methods are discussed in section 2.2. Their results are directly taken from Li et al. (2025).

![](images/a1786bcf4269870d47f048b8f04da607606e7cdbfdee85de3bfc22e9a517acce.jpg)  
Figure 2: Representative examples from the test sets of ETTm1 (top left), ECL (top right), Weather (bottom left) and Solar (bottom right). For each example, we show the 96-step input context, the subsequent 192-step groundtruth and the median VAIKAE prediction with corresponding 50% and 90% confidence intervals.

The MSE, MAE and CRPS of all models, summarized in table 1, exhibit strong performance of VAIKAE, which obtains either the best or second best result on all dataset-metric combinations. We additionally plot representative examples from the test sets of diferent datasets in figure 2. In most of these examples, the 90% confidence interval of the predictions includes the true state during most of the predicted intervals, which corresponds to the expected behavior, and suggests that the uncertainties are well calibrated. An exception is the example from the Solar dataset, where the groundtruth lies outside of the 90% confidence interval for a large portion of the predicted time steps, which shows that our prediction is overconfident for this dataset. This observation is consistent with the relatively poor CRPS obtained by VAIKAE for Solar in table 1 compared to the strong associated MSE result.

## 5 Data assimilation on satellite image time series

Here, we test our VAIKAE model on a satellite image time series forecasting benchmark, which was previously considered by e.g. Frion et al. (2024a; 2025). This benchmark consists of multispectral images with 10 separate spectral bands from the visible and infrared spectral domains. The images are obtained from the Sentinel-2 satellite constellation with a time step of 5 days and a spatial resolution of 10 meters. They are irregularly sampled due to the meteorological conditions that prevent obtaining high-quality images of the ground on most dates, which corresponds to the observation setup of equation 2. The benchmark considers two areas, of $5 \times 5$ km (i.e. 500 × 500 pixels) each, which are respectively named "Fontainebleau" and "Orléans", and both mostly covered by forest, although with difering properties.

Following Frion et al. (2024a; 2025), we train our VAIKAE model to predict pixelwise dynamics on a restricted time domain of the data from the Fontainebleau area, and then test its performance on both the held out time domain of the Fontainebleau area (i.e. temporal extrapolation) and the full time domain of the Orléans area (i.e. spatial extrapolation). In contrast to Frion et al. (2025), we here train our model on a larger spatial subdomain, of size 300 × 300 pixels (i.e. 9 square kilometers), as shown in figure 8, and we directly use irregularly-sampled data rather than a time-interpolated time series. The considered time domains for training and testing are consistent with prior works, and are respectively composed of $T _ { t r a i n } = 2 4 2$ time steps (roughly three and a half years) and the following $T _ { t e s t } = 1 0 0$ time steps (roughly one and a half years).

Table 2: Forecasting CRPS for diferent methods and areas
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>CRPS on the Fontainebleau area</td><td rowspan=1 colspan=1>CRPS on the Orléans area</td></tr><tr><td rowspan=1 colspan=1>VAIKAE</td><td rowspan=1 colspan=1>0.0241</td><td rowspan=1 colspan=1>0.0473</td></tr><tr><td rowspan=1 colspan=1>VIKAE</td><td rowspan=1 colspan=1>0.0304</td><td rowspan=1 colspan=1>0.0567</td></tr><tr><td rowspan=1 colspan=1>VKAE</td><td rowspan=1 colspan=1>0.0276</td><td rowspan=1 colspan=1>0.0540</td></tr><tr><td rowspan=1 colspan=1>KAE ensemble</td><td rowspan=1 colspan=1>0.0258</td><td rowspan=1 colspan=1>0.0530</td></tr></table>

## 5.1 Comparison of VAIKAE with other Koopman-based stochastic models

We consider several KAE methods for probabilistic forecasting of the pixelwise reflectance vectors:

• Our VAIKAE model, trained with the loss function of equation 11.

• An ablated version of VAIKAE that does not include an augmentation part in its latent embedding, but only an invertible encoding, thus facing the restriction $d = n$ . We refer to this model as VIKAE.

• Another ablation, named VKAE, where the latent embedding is obtained by a regular autoencoder, similar to Frion et al. (2024a) yet also with a Gaussian distribution for its latent embedding.

• An ensemble of 16 deterministic KAE models, jointly trained with a variance-promoting loss term, following Frion et al. (2024b). We call this approach KAE ensemble.

These models all have similar parameter counts except for the KAE ensemble, as each of its 16 members has about as many parameters as non-ensembled models. The training dataset contains slices of the full time series, spanning at most 100 time steps. Further implementation details can be found in appendix D.2. Our metric is the CRPS, computed on time steps $T _ { t r a i n }$ to $T _ { t r a i n } + T _ { t e s t }$ (for Fontainebleau) or 1 to $T _ { t r a i n } + T _ { t e s t }$ (for Orléans) with regard to the available observations in this range, using a single observation $\mathbf { y } _ { 0 }$ at time 0 as input. As summarized by table 2, VAIKAE performs best on both spatial areas. Consistently with the results of Frion et al. (2025) in a deterministic setting, VIKAE performs worst. This can be interpreted as a consequence of the restricted latent dimension $d = n$ of an invertible model, which prevents it from finding an accurate Koopman invariant subspace for this system. VKAE, which does not face this restriction, performs better than VIKAE, yet VAIKAE is the only ensemble-free model that outperforms the KAE ensemble.

## 5.2 Latent data assimilation with a trained model

Having established the superiority of VAIKAE to other KAE architectures for probabilistic long-term forecasting from a single observed reflectance vector, we now study how such a trained model can be used to leverage multiple observations with data assimilation, using the methods presented in section 3.3. We consider 3 diferent procedures for predicting the state of the system from time $T _ { t r a i n }$ to $T _ { t r a i n } + T _ { t e s t } \mathrm { : }$

• Predicting without assimilation, using only $\mathbf { y } _ { 0 }$ as an input, corresponding to the evaluation criterion in section 5.1. We refer to this method as single-obs.

• Assimilating on all available observations $\mathbf { y } _ { t }$ such that t is between 0 and $T _ { t r a i n } { - 1 }$ , using equations 14 and 15. We refer to this method as 4DVar.

• Assimilating the same set of observations with equation 16, thus jointly optimizing on the initial latent mean and variance. We refer to this method as 4DVar-CRPS.

![](images/e1e5e3286fc7c5d6b7bd35f7c61ec27850d08dbbd201f96cda23ea61a9f0d7d1.jpg)

![](images/3c24e9dfe75cbc2bc2e92ab390d7f11d8353b51e5bae4843b6e866c389cb8ab8.jpg)

![](images/b269a76c8a94e57962f645a4c024887997543901ea1a817c9c9d0f4903b9546e.jpg)  
Figure 3: Probabilistic performance of three methods, the first of which leverages only the initial observation while the two others assimilate all the blue-colored datapoints. We show only the B7 band, from the infrared domain, which is the most energetic in the dataset, yet we remind that our methods jointly manipulate the 10 spectral bands of pixelwise reflectance vectors. The dark shadings represent 50% confidence intervals while the light shadings represent 90% confidence intervals.

![](images/8e32d1e9778fc66ce6b4b3e7f8525960b07c2900fdf462568989af63621c6cc5.jpg)

![](images/26e78109f827e59f2c01520e939accdc0181c624a059fca751b90da29804bc69.jpg)

![](images/711f676b46428074ec7be8ef0bbc091bdffc95ca53f4c3778e4df2a062fa8e5e.jpg)

![](images/595f1e307a995964a417ea0bb6a27045ede0cc88e3efab7c27ae7d256e199a86.jpg)  
Figure 4: Summary of 4DVar-CRPS extrapolation results on the Orléans area. From left to right: 1) RGB composition of a true image 2) Mean squared error on the B7 spectral band, averaged over the extrapolation time range 3) Corresponding variance 4) Corresponding spread-skill ratio. Since the ideal value is 1, white marks a good calibration while blue and red respectively indicate overconfidence and underconfidence.

Table 3: Forecasting CRPS obtained when predicting from a single observation or assimilating on multiple observations with two diferent latent data assimilation methods.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>CRPS on the Fontainebleau area</td><td rowspan=1 colspan=1>CRPS on the Orléans area</td></tr><tr><td rowspan=1 colspan=1>single-obs</td><td rowspan=1 colspan=1>0.0241</td><td rowspan=1 colspan=1>0.0457</td></tr><tr><td rowspan=1 colspan=1>4DVar</td><td rowspan=1 colspan=1>0.0153</td><td rowspan=1 colspan=1>0.0273</td></tr><tr><td rowspan=1 colspan=1>4DVar-CRPS</td><td rowspan=1 colspan=1>0.0140</td><td rowspan=1 colspan=1>0.0244</td></tr></table>

![](images/b37705be0375d81d7c2204b8f63b9b94186b778779e0b4d21df5934b3b40f0f9.jpg)

![](images/9a1a3da3569f76901eef5b4a4d794317f24efd2c027e068100228ff3d5bdee46.jpg)

![](images/1b0183c772a71ffdf74d4d9ed0c522ee58f53750e1422acc18834335a3ebec37.jpg)

![](images/c514516ddb1b2741c80b13802715d8bfa9a162c05f362615164cf8db7348e29f.jpg)  
Figure 5: 4DVar-CRPS results on four randomly selected pixels from the test Orléans area. We consider the B7 spectral band, which is the most energetic in the dataset, although our method jointly processes the 10 spectral bands of the pixels. For each subfigure, the dark and light shading respectively correspond to the 50% and 90% confidence intervals of the predicted probability distributions.

We compute the CRPS obtained when estimating the reflectance vectors on time steps $T _ { t r a i n }$ to $T _ { t r a i n } { + } T _ { t e s t } { . }$ which are located beyond the assimilation window. The results of all 3 methods, summarized in Table 3, show that the CRPS in both areas is significantly reduced when multiple observations are included instead of a single one, marking a large improvement from single-obs to 4DVar. When using 4DVar-CRPS to fit both the mean and variance of the initial latent Gaussian distribution, a smaller additional gain is obtained. Interestingly, the performance gap between the training Fontainebleau area and the test Orléans area gets reduced, both in absolute and relative value, with stronger assimilation techniques. This shows that data assimilation can alleviate the distribution shift of the data, even without re-training or fine-tuning a model.

To bring more visual insight into the behavior of the three methods, we show their respective predictions on a randomly selected pixel of the Fontainebleau area in figure 3. From this example, one can see that single-obs, despite having some skill, does not accurately fit the local maxima of the reflectance dynamics, either in the assimilation or extrapolation domain. In contrast, 4DVar’s central prediction better captures the true dynamics, including in the extrapolation range, yet it is clearly underconfident. 4DVar-CRPS corrects this flaw by tightening the uncertainties around a similarly-behaving central

![](images/377747b89871998cdf3f42b87ed0f13914392c3269d02b6e36927e475bc3a749.jpg)

![](images/5154e60a9d0b806e9f81acaa8b13e0024bcde65ce6516ece2dce7266de2d0ae7.jpg)  
Figure 6: Spread-skill plots for time series extrapolation with 4DVar CRPS on the Fontainebleau (train) and Orléans (test) areas.

prediction. On figure 4, we display temporally aggregated statistics of our 4DVar-CRPS predictions over the test Orléans area, again focusing on the most energetic spectral band of the dataset. One can see that the spread-skill ratio (SSR) remains close to its ideal value of 1 in a large part of this spatial domain, but that some areas have largely overconfident predictions with a SSR below 1 while a smaller portion of the domain has underconfident predictions with a SSR above 1. We additionally display some 4DVar-CRPS confidence intervals on the test Orléans area in figure 5, qualitatively showing well calibrated predictions. These results are further discussed and illustrated with corresponding sampled state trajectories in appendix E.

Finally, figure 6 shows spread-skill plots (Haynes et al., 2023) of the 4DVar-CRPS predictions on both spatial areas in the extrapolation time domain. These plots are obtained by binning all predictions (mixing time steps, spatial locations and spectral bands) in a histogram according to their spread (i.e. standard deviation) estimated over many samples, and plotting the average skill (i.e. root mean squared error) for each bin. The inset spread frequencies plot indicates the relative sizes of the spread bins. A perfect spread-skill plo would match the 1:1 line, as the spread should match the skill on average. On the figure, one can see that the Fontainebleau predictions are nearly perfectly calibrated for low spread values but underconfident for large spread values, which however correspond to much fewer datapoints, and thus have a limited impact on the global SSR, which has a nearly perfect value of 1.04. The predictions on Orléans are overall slightly overconfident, leading to a global SSR of 0.83.

## 6 Conclusion

In this paper, we have introduced VAIKAE, a Koopman autoencoder model that enables probabilistic time series forecasting with a linearly evolving Gaussian distribution as its latent dynamics. We showed that this model enables analytical likelihood computation for the state at subsequent times given an observed initial state, thus inspiring a likelihood-based training criterion. We then discussed how to leverage a pre-trained VAIKAE model for uncertainty-aware latent data assimilation, enabling long-term time series forecasting using multiple irregularly-sampled state observations. We finally demonstrated the strong performance of our methods, with well-calibrated predictions on two long-term time series forecasting benchmarks, one of which contains irregularly-sampled time series.

Interesting directions for future work might include training our model with a CRPS-based loss criterion, as is becoming increasingly popular for global atmosphere forecasting models. This approach has an important computational cost compared to deterministic forecasting, but scales better than our likelihood-based approach for large state dimensions. Besides, our new data assimilation method for optimizing jointly on the mean and variance of the initial latent state using a CRPS-based variational cost should be explored in more general contexts than for VAIKAE only.

## References

Siddhant Agarwal, Ali Bekar, Anthony Frion, Eduardo Zorita-Calvo, and David Greenberg. Skillful seasonal forecasts with a freely evolving AI weather model. 2026.

Ferran Alet, Ilan Price, Andrew El-Kadi, Dominic Masters, Stratis Markou, Tom R Andersson, Jacklynn Stott, Remi Lam, Matthew Willson, Alvaro Sanchez-Gonzalez, et al. Skillful joint probabilistic weather forecasting from marginals. arXiv preprint arXiv:2506.10772, 2025.

Omri Azencot, N Benjamin Erichson, Vanessa Lin, and Michael Mahoney. Forecasting sequential data using consistent Koopman autoencoders. In International conference on machine learning, pp. 475–485. PMLR, 2020.

Christopher M Bishop and Nasser M Nasrabadi. Pattern recognition and machine learning, volume 4. Springer, 2006.

Marc Bocquet. Surrogate modeling for the climate sciences dynamics with machine learning and data assimilation. Frontiers in Applied Mathematics and Statistics, 9:1133226, 2023.

James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Yash Katariya, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github. com/jax-ml/jax.

Steven L Brunton, Bingni W Brunton, Joshua L Proctor, and J Nathan Kutz. Koopman invariant subspaces and finite linear representations of nonlinear dynamical systems for control. PloS one, 11(2):e0150171, 2016.

Steven L. Brunton, Marko Budišić, Eurika Kaiser, and J. Nathan Kutz. Modern Koopman theory for dynamical systems. SIAM Review, 64(2):229–340, 2022. doi: 10.1137/21M1401243. URL https://doi. org/10.1137/21M1401243.

Alberto Carrassi, Marc Bocquet, Laurent Bertino, and Geir Evensen. Data assimilation in the geosciences: An overview of methods, issues, and perspectives. Wiley Interdisciplinary Reviews: Climate Change, 9(5): e535, 2018.

Sibo Cheng, César Quilodrán-Casas, Said Ouala, Alban Farchi, Che Liu, Pierre Tandeo, Ronan Fablet, Didier Lucor, Bertrand Iooss, Julien Brajard, et al. Machine learning with data assimilation and uncertainty quantification for dynamical systems: a review. IEEE/CAA Journal of Automatica Sinica, 10(6):1361– 1387, 2023.

Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.

Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real NVP. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=HkpbnH9lx.

Hang Fan, Lei Bai, Ben Fei, Yi Xiao, Kun Chen, Yubao Liu, Yongquan Qu, Fenghua Ling, and Pierre Gentine. Physically consistent global atmospheric data assimilation with machine learning in latent space. Science Advances, 12(1):eaea4248, 2026.

Anthony Frion, Lucas Drumetz, Mauro Dalla Mura, Guillaume Tochon, and Abdeldjalil Aissa El Bey. Neural Koopman prior for data assimilation. IEEE Transactions on Signal Processing, 72:4191–4206, 2024a.

Anthony Frion, Lucas Drumetz, Guillaume Tochon, Mauro Dalla Mura, and Abdeldjalil Aissa El Bey. Koopman ensembles for probabilistic time series forecasting. In 2024 32nd European Signal Processing Conference (EUSIPCO), pp. 2542–2546. IEEE, 2024b.

Anthony Frion, Lucas Drumetz, Mauro Dalla Mura, Guillaume Tochon, and Abdeldjalil AISSA EL BEY. Augmented invertible Koopman autoencoder for long-term time series forecasting. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=o6ukhJLzMQ.

Anthony Frion, Vien Minh Nguyen-Thanh, Ali Can Bekar, Pauleo R Nimtz, Vadim Zinchenko, and David S Greenberg. ADDA: a modular framework for representing, simulating and assimilating dynamics with end-to-end diferentiability. arXiv preprint arXiv:2608.23297, 2026.

Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pp. 1050–1059. PMLR, 2016.

Marta Garnelo, Dan Rosenbaum, Christopher Maddison, Tiago Ramalho, David Saxton, Murray Shanahan, Yee Whye Teh, Danilo Rezende, and S. M. Ali Eslami. Conditional neural processes. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 1704–1713. PMLR, 10–15 Jul 2018. URL https: //proceedings.mlr.press/v80/garnelo18a.html.

Maximilian Gelbrecht, Alistair White, Sebastian Bathiany, and Niklas Boers. Diferentiable programming for Earth system modeling. Geoscientific Model Development, 16(11):3123–3135, 2023.

Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association, 102(477):359–378, 2007.

Minghao Han, Jacob Euler-Rolle, and Robert K. Katzschmann. DeSKO: Stability-assured robust contro with a deep stochastic Koopman operator. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=hniLRD\_XCA.

Sam Hatfield, Matthew Chantry, Peter Dueben, Philippe Lopez, Alan Geer, and Tim Palmer. Building tangent-linear and adjoint models for data assimilation with neural networks. Journal of Advances in Modeling Earth Systems, 13(9):e2021MS002521, 2021.

Katherine Haynes, Ryan Lagerquist, Marie McGraw, Kate Musgrave, and Imme Ebert-Uphof. Creating and evaluating uncertainty estimates with neural networks for environmental-science applications. Artificial Intelligence for the Earth Systems, 2(2):220061, 2023. doi: 10.1175/AIES-D-22-0061.1. URL https: //journals.ametsoc.org/view/journals/aies/2/2/AIES-D-22-0061.1.xml.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Xiao Hou, Jin Zhang, and Le Fang. Invertible neural network combined with dynamic mode decomposition applied to flow field feature extraction and prediction. Physics of Fluids, 36(9), 2024.

Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. arXiv preprint arXiv:1803.05407, 2018.

Yuhong Jin, Lei Hou, and Shun Zhong. Extended dynamic mode decomposition with invertible dictionary learning. Neural networks, 173:106177, 2024.

Laurent Valentin Jospin, Hamid Laga, Farid Boussaid, Wray Buntine, and Mohammed Bennamoun. Handson Bayesian neural networks—a tutorial for deep learning users. IEEE Computational Intelligence Magazine, 17(2):29–48, 2022.

Alex Kendall and Yarin Gal. What uncertainties do we need in Bayesian deep learning for computer vision? Advances in neural information processing systems, 30, 2017.

Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International conference on learning representations, 2021.

Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114, 2013.

Ivan Kobyzev, Simon JD Prince, and Marcus A Brubaker. Normalizing flows: An introduction and review of current methods. IEEE transactions on pattern analysis and machine intelligence, 43(11):3964–3979, 2020.

Dmitrii Kochkov, Jamie A Smith, Ayya Alieva, Qing Wang, Michael P Brenner, and Stephan Hoyer. Machine learning–accelerated computational fluid dynamics. Proceedings of the National Academy of Sciences, 118 (21):e2101784118, 2021.

Bernard O Koopman. Hamiltonian systems and transformation in hilbert space. Proceedings of the National Academy of Sciences, 17(5):315–318, 1931.

Milan Korda and Igor Mezić. On convergence of extended dynamic mode decomposition to the Koopman operator. Journal of Nonlinear Science, 28(2):687–710, 2018.

Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In The 41st international ACM SIGIR conference on research & development in information retrieval, pp. 95–104, 2018.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017.

Simon Lang, Mihai Alexe, Mariana CA Clare, Christopher Roberts, Rilwan Adewoyin, Zied Ben Bouallègue, Matthew Chantry, Jesper Dramsch, Peter D Dueben, Sara Hahner, et al. AIFS-CRPS: ensemble forecasting using a model trained with a loss function based on the continuous ranked probability score. npj Artificial Intelligence, 2(1):18, 2026.

Christian Legaard, Thomas Schranz, Gerald Schweiger, Ján Drgoňa, Basak Falay, Cláudio Gomes, Alexandros Iosifidis, Mahdi Abkar, and Peter Larsen. Constructing neural network based models for simulating dynamical systems. ACM Computing Surveys, 55(11):1–34, 2023.

Qi Li, Zhenyu Zhang, Lei Yao, Zhaoxia Li, Tianyi Zhong, and Yong Zhang. Difusion-based decoupled deterministic and uncertain framework for probabilistic multivariate time series forecasting. In The Thirteenth International Conference on Learning Representations, 2025.

Yunzhu Li, Hao He, Jiajun Wu, Dina Katabi, and Antonio Torralba. Learning compositional Koopman operators for model-based control. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=H1ldzA4tPr.

Yuxin Li, Wenchao Chen, Xinyue Hu, Bo Chen, Mingyuan Zhou, et al. Transformer-modulated difusion models for probabilistic multivariate time series forecasting. In International Conference on Learning Representations, volume 2024, pp. 18604–18622, 2024.

Zhe Li, Shiyi Qi, Yiduo Li, and Zenglin Xu. Revisiting long-term time series forecasting: An investigation on linear mapping. arXiv preprint arXiv:2305.10721, 2023.

Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial diferential equations. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id=c8P9NQVtmnO.

Kaiyuan Liao, Xiwei Xuan, and Kwan-Liu Ma. Deep learning for time series forecasting: a survey of recent advances. Frontiers of Computer Science, 20(11):2011359, 2026.

Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary transformers: Exploring the stationarity in time series forecasting. Advances in neural information processing systems, 35:9881–9893, 2022.

Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. iTransformer: Inverted transformers are efective for time series forecasting. In International conference on learning representations, volume 2024, pp. 11116–11140, 2024.

Lennart Ljung. Perspectives on system identification. Annual Reviews in Control, 34(1):1–12, 2010.

Eric Lupascu, Xiao Li, and Benjamin Schäfer. Predicting power grid frequency dynamics with invertible Koopman-based architectures. In 2026 Open Source Modelling and Simulation of Energy Systems (OSM-SES), pp. 1–6. IEEE, 2026.

Bethany Lusch, J Nathan Kutz, and Steven L Brunton. Deep learning for universal linear embeddings of nonlinear dynamics. Nature communications, 9(1):4950, 2018.

Andrew L Maas, Awni Y Hannun, Andrew Y Ng, et al. Rectifier nonlinearities improve neural network acoustic models. In Proc. icml, volume 30, pp. 3. Atlanta, GA, 2013.

Alex T Mallen, Henning Lange, and J Nathan Kutz. Deep probabilistic Koopman: long-term time-series forecasting under periodic uncertainties. International Journal of Forecasting, 40(3):859–868, 2024.

Boštjan Melinc and Žiga Zaplotnik. 3d-var data assimilation using a variational autoencoder. Quarterly Journal of the Royal Meteorological Society, 150(761):2273–2295, 2024.

Yuhuang Meng, Jianguo Huang, and Yue Qiu. Koopman operator learning using invertible neural networks. Journal of Computational Physics, 501:112795, 2024.

Igor Mezić. Spectral properties of dynamical systems, model reduction and decompositions. Nonlinear Dynamics, 41:309–325, 2005.

Sebastian Milinski, Nicola Maher, and Dirk Olonscheck. How large does a large ensemble need to be? Earth System Dynamics, 11(4):885–901, 2020.

Ilan Naiman, N. Benjamin Erichson, Pu Ren, Michael W. Mahoney, and Omri Azencot. Generative modeling of regular and irregular time series data via Koopman VAEs. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=eY7sLb0dVF.

Juan Nathaniel, Yongquan Qu, Tung Nguyen, Sungduk Yu, Julius Busecke, Aditya Grover, and Pierre Gentine. Chaosbench: A multi-channel, physics-based benchmark for subseasonal-to-seasonal climate prediction. Advances in Neural Information Processing Systems, 37:43715–43729, 2024.

Indranil Nayak, Ananda Chakrabarti, Mrinal Kumar, Fernando L Teixeira, and Debdipta Goswami. Temporally consistent koopman autoencoders for forecasting dynamical systems. Scientific Reports, 15(1):22127, 2025.

Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=Jbdc0vTOcol.

Marcel Nonnenmacher and David S Greenberg. Deep emulators for diferentiation, forecasting, and parametrization in Earth science simulators. Journal of Advances in Modeling Earth Systems, 13(7): e2021MS002554, 2021.

Joel Oskarsson, Tomas Landelius, Marc Peter Deisenroth, and Fredrik Lindsten. Probabilistic weather forecasting with hierarchical graph neural networks. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=wTIzpqX121.

Samuel E Otto and Clarence W Rowley. Linearly recurrent autoencoder networks for learning dynamics. SIAM Journal on Applied Dynamical Systems, 18(1):558–593, 2019.

Shaowu Pan and Karthik Duraisamy. Physics-informed probabilistic learning of linear embeddings of nonlinear dynamics with guaranteed stability. SIAM Journal on Applied Dynamical Systems, 19(1):480–509, 2020.

Guansong Pang, Chunhua Shen, Longbing Cao, and Anton Van Den Hengel. Deep learning for anomaly detection: A review. ACM computing surveys (CSUR), 54(2):1–38, 2021.

Ivo Pasmans, Yumeng Chen, Tobias Sebastian Finn, Marc Bocquet, and Alberto Carrassi. Ensemble Kalman filter in latent space using a variational autoencoder pair. Quarterly Journal of the Royal Meteorological Society, 152(775):e70070, 2026.

Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, Adam Lerer, et al. Automatic diferentiation in PyTorch. 2017.

Mathis Peyron, Anthony Fillion, Selime Gürol, Victor Marchais, Serge Gratton, Pierre Boudier, and Gael Goret. Latent space data assimilation by using deep learning. Quarterly Journal of the Royal Meteorological Society, 147(740):3759–3777, 2021.

Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R Andersson, Andrew El-Kadi, Dominic Masters, Timo Ewalds, Jacklynn Stott, Shakir Mohamed, Peter Battaglia, et al. Probabilistic weather forecasting with machine learning. Nature, 637(8044):84–90, 2025.

Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learn ing framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational physics, 378:686–707, 2019.

Kashif Rasul, Calvin Seward, Ingmar Schuster, and Roland Vollgraf. Autoregressive denoising difusion models for multivariate probabilistic time series forecasting. In International conference on machine learning, pp. 8857–8868. PMLR, 2021.

Facundo Sapienza, Jordi Bolibar, Frank Schäfer, Brian Groenke, Avik Pal, Victor Boussange, Patrick Heimbach, Giles Hooker, Fernando Pérez, Per-Olof Persson, et al. Diferentiable programming for diferential equations: A review. arXiv preprint arXiv:2406.09699, 2024.

Peter J Schmid. Dynamic mode decomposition of numerical and experimental data. Journal of fluid mechanics, 656:5–28, 2010.

Maximilian Seitzer, Arash Tavakoli, Dimitrije Antic, and Georg Martius. On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=aPOpXlnV1T.

Sandeep Singh Sengar, Afan Bin Hasan, Sanjay Kumar, and Fiona Carroll. Generative artificial intelligence: a systematic review and applications. Multimedia Tools and Applications, 84(21):23661–23700, 2025.

Lifeng Shen and James Kwok. Non-autoregressive conditional difusion models for time series prediction. In International Conference on Machine Learning, pp. 31016–31029. PMLR, 2023.

Yutaka Shoji, Sohei Arisaka, Eikichi Ono, and Kuniaki Mihara. Data assimilation for HVAC simulations in Koopman-invariant subspace using Kalman filter. In Proceedings of the 12th ACM International Conference on Systems for Energy-Eficient Buildings, Cities, and Transportation, pp. 306–307, 2025.

Nicki Skafte, Martin Jørgensen, and Søren Hauberg. Reliable training and estimation of variance networks. Advances in Neural Information Processing Systems, 32, 2019.

Yusuke Tashiro, Jiaming Song, Yang Song, and Stefano Ermon. CSDI: Conditional score-based difusion models for probabilistic time series imputation. Advances in neural information processing systems, 34: 24804–24816, 2021.

Xin T Tong, Yanyan Wang, and Liang Yan. Latent autoencoder ensemble Kalman filter for nonlinear data assimilation. arXiv preprint arXiv:2603.06752, 2026.

Charles Truong, Laurent Oudre, and Nicolas Vayatis. Selective review of ofline change point detection methods. Signal processing, 167:107299, 2020.

Matthew O Williams, Ioannis G Kevrekidis, and Clarence W Rowley. A data–driven approximation of the Koopman operator: Extending dynamic mode decomposition. Journal of Nonlinear Science, 25:1307–1346, 2015.

Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in neural information processing systems, 34: 22419–22430, 2021.

Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Difusion models: A comprehensive survey of methods and applications. ACM computing surveys, 56(4):1–39, 2023.

Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers efective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 11121–11128, 2023.

Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In International Conference on Learning Representations, 2018. URL https: //openreview.net/forum?id=r1Ddp1-Rb.

Jingdong Zhang, Qunxi Zhu, and Wei Lin. Learning hamiltonian neural koopman operator and simultaneously sustaining and discovering conservation laws. Physical Review Research, 6(1):L012031, 2024.

Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In The eleventh international conference on learning representations, 2023.

Ronghua Zheng, Hanru Bai, and Weiyang Ding. Koonpro: A variance-aware Koopman probabilistic mode enhanced by neural process for time series forecasting. In The Thirteenth International Conference on Learning Representations, 2025.

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond eficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pp. 11106–11115, 2021.

Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International conference on machine learning, pp. 27268–27286. PMLR, 2022.

![](images/b6621b68d340d2d61a59185f8f39baa5f857b9a52af9775e9684a652ce3092cb.jpg)  
Figure 7: Visual representation of the AIKAE architecture, copied with permission from Frion et al. (2025).

## A Representation of the AIKAE architecture

The VAIKAE architecture, which is presented in section 3.1, is based upon the AIKAE architecture introduced by Frion et al. (2025). We show a visual representation of the AIKAE in figure 7. The main diference between the new VAIKAE architecture shown in figure 1 and the AIKAE is that the size of the output of χ is increased so that it produces a latent diagonal variance vector $\sigma _ { t }$ for the whole latent distribution in addition to the mean $\pmb { \mu } _ { t } ^ { a }$ of the augmentation part of the latent space.

## B Description of the datasets

We consider 6 standard datasets for long-term time series forecasting, with varying sizes and properties. The ETTm1 and ETTm2 datasets (Zhou et al., 2021) correspond to measurements of 7 factors from electricity transformers, such as load and oil temperature, recorded every 15 minutes from July 2016 to July 2018. Weather (Wu et al., 2021) contains 21 weather indicators recorded every 10 minutes by the Max Planck Biogeochemistry Institute in 2020. Solar (Lai et al. (2018), also called Solar-Energy) contains measurements from the solar power production of 137 photovoltaic plants, recorded every 10 minutes in 2006. ECL (Wu et al. (2021), also called Electricity) reports the hourly electricity consumption of 321 clients from 2012 to 2014. Finally, Trafic (Wu et al., 2021) records the hourly occupancy rates of 862 roads in the San Francisco Bay Area, from January 2015 to December 2016.

For all of these datasets, we adopt the same approach as Li et al. (2025) and many other previous works by using the earlier time steps of the time series for training, an intermediary time range for validation and the latest part for testing. The length ratios of the train-validation-test splits are 6:2:2 for the ETTm1 and ETTm2 datasets, and 7:1:2 for all other datasets.

## C Description of the evaluation metrics

## C.1 Deterministic metrics

Mean squared error (MSE) is commonly used as a training criterion and as an evaluation metric for neural networks on regression problems. It is defined as the averaged squared error between a pointwise prediction (or, alternatively, the mean of a predicted distribution) and the corresponding groundtruth values. For a time series $( \mathbf { x } _ { t } ^ { i } ) _ { 1 \leq t \leq T , 1 \leq i \leq n }$ with an n-dimensional state $\mathbf { x } _ { t } \in \mathbb { R } ^ { n }$ and a corresponding prediction $( \hat { \mathbf { x } } _ { t } ^ { i } ) _ { 1 \leq t \leq T , 1 \leq i \leq n }$ the MSE is computed as:

$$
\mathrm { M S E } = \frac { 1 } { n T } \sum _ { i = 1 } ^ { n } \sum _ { t = 1 } ^ { T } ( \mathbf { x } _ { t } ^ { i } - \hat { \mathbf { x } } _ { t } ^ { i } ) ^ { 2 } .\tag{17}
$$

Mean average error (MAE) corresponds to the average absolute diference between the predicted and true state, over diferent variables and time steps of a time series. With similar notation as above, it can be expressed as:

$$
\mathrm { M A E } = \frac { 1 } { n T } \sum _ { i = 1 } ^ { n } \sum _ { t = 1 } ^ { T } | \mathbf x _ { t } ^ { i } - \hat { \mathbf x } _ { t } ^ { i } | .\tag{18}
$$

## C.2 Proper scoring rules for probabilistic forecasts

Proper scoring rules (Gneiting & Raftery, 2007) are central to the training and evaluation of data-driven stochastic models. In short, proper scoring rules are used to compare a predicted probability distribution to a single true datapoint that is supposedly sampled from a groundtruth probability distribution. In the limit of an infinite number of datapoints, (strictly) proper scoring rules are minimized (only) when the predicted probability distribution is the same as the groundtruth distribution. A popular proper scoring rule for both training and evaluating probabilistic neural networks is the negative log-likelihood, also called the logarithmic score (see e.g. Lakshminarayanan et al. (2017); Kendall & Gal (2017)). However, some pitfalls of negative log-likehood as a training criterion have been identified by e.g. Skafte et al. (2019); Seitzer et al. (2022). This partly explains the recent gain in popularity of the CRPS, which we present next.

Continuous ranked probability score (CRPS) is a strictly proper scoring rule, defined for a univariate probability distribution, described by its cumulative distribution function F. When the true observed value is $x _ { t r u e } \in \mathbb { R }$ , the CRPS can be written as:

$$
\mathrm { C R P S } = \int _ { \mathbb { R } } ( F ( x ) - \mathbb { 1 } _ { x \geq x _ { t r u e } } ) ^ { 2 } d x ,\tag{19}
$$

where $\mathbb { 1 } _ { x \geq x _ { t r u e } }$ denotes the indicator function, with value 1 when $x \geq x _ { t r u e }$ and 0 otherwise. Importantly, for multivariate quantities, one usually considers the average of the CRPS over all variables of the state, although this means evaluating the marginals of this multivariate distribution rather than the joint distribution (Alet et al., 2025).

In practice, as detailed by e.g. Gneiting & Raftery (2007), when the predicted probability distribution is defined (or approximated) by a finite ensemble of M equiprobable values $x _ { 1 } , . . . , x _ { M }$ , the CRPS can be written as

$$
\mathrm { C R P S } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } | x _ { t r u e } - x _ { i } | - \frac { 1 } { 2 } \frac { 1 } { M ^ { 2 } } \sum _ { j = 1 } ^ { M } \sum _ { k = 1 } ^ { M } | x _ { j } - x _ { k } | .\tag{20}
$$

This expression is easy to compute and diferentiate in practice, and enables estimation of the CRPS of a complex probability distribution by drawing M samples. It has thus been used to train multiple probabilistic forecasting models, notably for recent data-driven models of the atmosphere (Price et al., 2025; Alet et al., 2025; Lang et al., 2026; Agarwal et al., 2026). In this work, we do not use this strategy for training our models, although this would be a possibility. We however use the expression from equation 20 in order to solve a gradient descent on the latent variational data assimilation cost of equation 16.

## D Implementation details

## D.1 Informer benchmark

For all datasets in this benchmark, the invertible encoder ϕ is implemented using a NICE (Dinh et al., 2014) normalizing flow model with l coupling layers, each using the additive coupling law with a simple multi-layer perceptron (MLP) comprising a single layer with width 256 and a leaky rectified linear unit nonlinearity (Maas et al., 2013). The second encoder $\chi$ is simply an MLP network with 3 layers of respective widths [256, 128, 160]. Note that, as outlined in section 3.1, the output layer is of size $1 6 0 = 1 2 8 + 3 2 =$ $p + d = n + 2 d .$ , as the input to the model is of size $n = 9 6$ and we have fixed the size of the augmentation encoding to $d = 3 2$

![](images/90be7a82e59467a261e1a0e40c7fa04f143eebc1d4e2fb2db0d0a18face8c9d2.jpg)

![](images/e3dc188d9feebd7fa9563b2cfebe0891e3b6008eb0a5d555a7c7a384c22b2b7f.jpg)  
Figure 8: Left: an image from the forest of Fontainebleau. Right: an image from the forest of Orléans. The date for both images is $2 0 / 0 6 / 2 0 1 8$ . Those are RGB compositions with saturated colors. The red square is the $3 0 0 \times 3 0 0$ pixel training area and the blue square is the $3 0 0 \times 3 0 0$ pixel test area.

Like several strong variable-independent methods (Li et al., 2023; Frion et al., 2025), we use a reversible instance normalization (RevIN, Kim et al. (2021)) of the input. We use the training loss of equation 11 with relative weights $\alpha = \beta = 0 , \gamma = 1 0 ^ { - 2 }$ . The models are trained with the Adam optimizer, using a learning rate r and default momentum parameters $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ . The batch size is set to 128. A hyperparameter search was performed for each dataset on the following values: $l \in \{ 4 , 6 \} , r \in \{ 1 0 ^ { - 3 } , 2 \cdot 1 0 ^ { - 3 } \}$

In our experiments on this benchmark, we observed a trade-of between accurate prediction of the conditional mean and uncertainty calibration, controlled by the weight used for $\gamma$ in the loss function. $\gamma = 0$ reduces to the AIKAE (Frion et al., 2025) in practice, which is a strong model for deterministic predictions but does not provide uncertainties. As γ increases, the uncertainties increase (i.e. become better calibrated) yet the deterministic MSE and MAE metrics tend to simultaneously worsen. We solve this dilemma by choosing the relatively low $\gamma = 1 0 ^ { - 2 }$ , which leads to a highly overconfident model, and then performing a recalibration of the uncertainties as a post-processing method. Concretely, for each variable and prediction time step, we compute the average spread-skill ratio over the validation dataset. Afterwards, when evaluating the model’s predictions, we inflate the variable-and-time-step-wise standard deviation of the predictions with the inverse of these observed spread-skill ratios.

## D.2 Satellite image time series benchmark

In contrast with the other set of experiments, our VAIKAE model now uses the Real-NVP normalizing flow architecture (Dinh et al., 2017), which is significantly more expressive than the previously used NICE architecture (Dinh et al., 2014) but more prone to instabilities. We stack 6 Real-NVP coupling layers, which each internally uses a simple MLP network with a single hidden layer of width 256. The augmentation encoder $\chi$ is a MLP network with 2 hidden layers of respective sizes 512 and 256. The augmentation part $\mathbf { z } _ { t } ^ { a }$ of the latent embedding is of size $p = 1 6$ . Since the state is a vector of $n = 1 0$ reflectance values, the full latent size of the model is $d = n + p = 2 6$

On figure 8, we show RGB compositions of images from the two considered spatial areas, along with the subdomains used for training and testing our models. As mentioned in the main text, the whole training domain is of size $2 4 2 \times 3 0 0 \times 3 0 0 \times 1 0$ , with 242 time steps, $3 0 0 \times 3 0 0$ pixels and $n = 1 0$ spectral bands, yet the observations are sparse since only about 1 in 5 time steps are actually observed, although all pixels and spectral bands are always observed at the same time. We create slices of observations starting from each observed pixel before time $T = 1 4 2$ and then including all available observations for the subsequent 100 time steps. These slices of observations are randomly separated into batches of size 2048 and used for training with the loss of equation 11. For VAIKAE and VIKAE, we use $\alpha = 1 0 ^ { - 3 } , \beta = 1 0 ^ { - 2 } , \gamma = 1 0 ^ { 5 }$ . We use the Adam optimizer with a learning rate of $1 0 ^ { - 4 }$ . For VKAE, since the reconstruction is not guaranteed to be exact by design, we additionally use a reconstruction loss term as in Frion et al. (2024a), and the likelihood loss term is adapted so that the intractable determinant of the Jacobian of the latent embedding is simply substituted by 1, which amounts to ignoring the deformation of the latent space. Our validation criterion is the mean squared error of predictions from time 0 to 242 on a subset of the spatial training domain. Despite the relatively low value of the weight λ for the likelihood loss terms, the uncertainties are relatively well calibrated at the end of the training, and a post-processing recalibration is not required.

![](images/4bbacfc50e09e6f408ffb8660e638f75336630f3ea9bfde0e1accb5782b4a6bf.jpg)

![](images/60e2afbd7c60afe34ec6d1f2f0502dd7cce024307b071f51f96b3580e59ccafb.jpg)

![](images/34762a34309a1d894e17b19cb2d722f398f49b3408cff3bf5b515335eaad42eb.jpg)

![](images/463b83cddf5cd2f48fa588051547fceb79782ace594f8fc28836d35a667bdb0e.jpg)  
Figure 9: Each subfigure corresponds to a randomly selected pixel from the Orléans area, for which we assimilate the blue-colored observations and sample 100 state trajectories from the inferred distribution.

## E Additional satellite image time series results

Here, we plot some results obtained with the "4DVar-CRPS" assimilation method, described in equation 16, which jointly optimizes on the mean and variance of the initial latent Gaussian embedding of the state. We randomly sample 4 pixels from the Orléans area, from which no data was observed while training the VAIKAE model. On figure 9, we show for each of these pixels 100 sampled trajectories from the assimilated initial distribution, including the "central prediction" which corresponds to the mean of the initial latent Gaussian distribution but not necessarily to the mean or median of the trajectories in the state space, due to the deformation between the latent and state spaces. These predictions are performed over the same data as for the confidence intervals of figure 5, and in fact illustrate how such confidence intervals are obtained: by sampling a large number of trajectories and computing associated quantiles for each variable and time step. From figure $5 ,$ one can see that most of the observations are captured by the 90% confidence interval, although some clear outliers, either in the assimilation or extrapolation range, remain outside of it, which does correspond to the expected behavior of the probabilistic forecasting model. This visually relevant identification of seemingly anomalous values in the time series hints at how a trained VAIKAE model could be used alongside with 4DVar-CRPS for anomaly detection.