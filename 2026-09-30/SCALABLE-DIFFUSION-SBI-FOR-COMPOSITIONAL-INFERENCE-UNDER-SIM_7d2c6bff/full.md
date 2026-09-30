# SCALABLE DIFFUSION SBI FOR COMPOSITIONAL INFERENCE UNDER SIMULATOR MISSPECIFICATION

Vincent D. Zaballa & Elliot E. Hui

University of California, Irvine

vzaballa@uci.edu

## ABSTRACT

Simulation-based inference is challenging when many heterogeneous observations must be composed, hierarchical latent structure must be preserved, and the simulator is misspecified relative to observed data. We develop sampling and fine-tuning methods for diffusion-based inference in design-conditional settings, where the same simulator is queried across different experimental conditions ξ. We extend compositional score-based inference with a continuous-time diffusion coefficient that accounts for the number of observations, avoiding Jacobian and auxiliary-covariance corrections. We introduce Hierarchical Blockwise Diffusion Sampling (HBDS), which infers shared parameters and group-specific latent states using a single pretrained model, with the hierarchy specified only at sampling time. Together, these methods support variable observation sets and groupings without retraining. To address misspecification, we introduce path-regularized finetuning that adapts the learned likelihood to observations and transfers corrections to posterior inference. Using Girsanov’s theorem, we quantify path divergence between pretrained and fine-tuned models across experimental designs and interpret it alongside predictive errors to distinguish candidate misspecification correction from unnecessary adaptation. We evaluate compositional sampling on exact-score Gaussian and Simple Likelihood, Complex Posterior benchmarks, HBDS with analytic and learned scores on a controlled hierarchical model, and fine-tuning and localization on a separate analytic model with known design-dependent discrepancy. Finally, we apply the framework to 940 measurements across four cell lines in a mechanistic Bone Morphogenetic Protein signaling model, where fine-tuning improves posterior-predictive accuracy relative to the pretrained model and shifts posterior marginals toward the least-squares reference while retaining spread.

## 1 INTRODUCTION

Mechanistic simulators compress scientific knowledge into predictive models whose latent parameters carry physical or biological meaning. Simulation-based inference (SBI) (Cranmer et al., 2020) can amortize Bayesian inference in these models even when their likelihoods cannot be evaluated. In reality, however, inference rarely consists of conditioning on a single observation from a perfectly specified simulator. Data may contain many heterogeneous observations, shared and group-specific latent variables, and systematic discrepancies between the simulator and the observed system.

These challenges have motivated several complementary directions in SBI. Compositional methods aggregate multiple observations in a score-based model without retraining a posterior estimator, beginning with Factorized Neural Posterior Score Estimation (F-NPSE) (Geffner et al., 2023) and followed by covariance-aware diffusion samplers for tall-data SBI (Linhart et al., 2026). Meanwhile, recent hierarchical approaches explicitly model shared and local latent structure through hierarchical estimators or training constructions (Arruda et al., 2026; Charles et al., 2026; Heinrich et al., 2024). Separately, simulator misspecification has motivated a growing literature on detection and mitigation (Boelts, 2026; Cannon et al., 2022), including explicit discrepancy models (Ward et al., 2022), misspecification-aware representations and summary statistics (Huang et al., 2023b; Schmitt et al., 2024), and distributional alignment or calibration methods (Wehenkel et al., 2025). These lines of work address composition, hierarchy, and misspecification, but typically as distinct problems.

![](images/4139e74c69ae34b6359cc9ceffebff2c2e3990c107e2d01ff708479dc8ddcf7c.jpg)  
Figure 1: Schematic overview of the framework. (Left) Compositional sampling combines observations using the derived diffusion coefficient $g _ { n } ( t )$ and optional Langevin correctors, without retraining. (Middle) HBDS alternates updates of shared parameters θ and group-specific latent states $R _ { k }$ conditioned on each group’s data $\mathcal { D } _ { k }$ . (Right) Path divergence $\bar { \mathcal { A } } ( \bar { \theta } ^ { \mathrm { r e f } } , \bar { \xi } )$ measures the magnitude of likelihood adaptation across experimental designs. Comparing this field with predictive improvement helps distinguish candidate misspecification correction from over-adaptation.

We study how these requirements can be combined within a single diffusion-based SBI framework (Sharrock et al., 2024). Our central idea is to treat a pretrained amortized conditional model as a reusable inference engine: observation sets can be composed, hierarchical structure introduced at sampling time, and the simulator-trained surrogate adapted to observed data while retaining posterior inference in the original mechanistic parameterization.

Achieving the last of these requires likelihood and posterior score representations to remain coupled as the simulator-trained surrogate is adapted to observed data. We use the Simformer of Gloeckler et al. (2024) as the diffusion backbone and introduce a new mechanism for transfer learning (Pan & Yang, 2010) through shared embeddings, allowing likelihood fine-tuning to update representations used for posterior inference (§ G.7). The framework is summarized in Figure 1. Building on this coupled surrogate, our contributions are:

• Composition-aware diffusion sampling. We derive an n-dependent continuous-time diffusion coefficient for compositional SBI, extending F-NPSE to efficient SDE sampling while avoiding the Jacobian and auxiliary-covariance costs of JAC and GAUSS.

• Hierarchical inference at sampling time. We introduce Hierarchical Blockwise Diffusion Sampling (HBDS), reusing a single joint diffusion model to infer shared and group-specific latents with hierarchy introduced only at sampling time.

• Path-space adaptation and localization. We fine-tune the surrogate likelihood to observations while transferring the correction to posterior inference, and use Girsanov’s theorem to isolate where model misspecification may occur.

• Large-scale systems biology inference. We apply the full framework to a Bone Morphogenetic Protein (BMP) signaling simulator with 60 shared parameters, five receptor quantities per cell line, and 940 observations, showing how composition, hierarchy, and misspecification-aware adaptation jointly make SBI practical on a nontrivial scientific model.

## 2 BACKGROUND

## 2.1 DESIGN-DEPENDENT SIMULATION-BASED INFERENCE

SBI considers a simulator from which observations can be sampled conditional on latent parameters and, when relevant, experimental conditions. In that setting we write $y \sim p _ { \mathrm { e v a l } } ( y \mid \theta )$ , where $p _ { \mathrm { e v a l } } ( y , \theta ) = p _ { \mathrm { e v a l } } ( y \mid \theta ) p ( \theta )$ is an expert simulator, distinct from the unknown true data-generating process $\boldsymbol { p } ^ { * } ( y , \theta )$ We adopt the $p _ { \mathrm { e v a l } } ( y \mid \theta )$ notation of Smith et al. (2025) and use the designdependent setting where an experimental design ξ alters the joint model and resulting posterior, yielding $p _ { \mathrm { e v a l } } ( y , \bar { \theta } \mid \xi )$ and related $\boldsymbol { p } ^ { * } ( y , \theta \mid \boldsymbol { \xi } )$ . We represent a simulator as a function

$$
f : \Xi \times \Theta \longrightarrow \mathcal { V } , \quad \Xi \subset \mathbb { R } ^ { D } , \Theta \subset \mathbb { R } ^ { M } , \mathcal { V } \subset \mathbb { R } ^ { B } ,
$$

where $\Xi , \Theta$ , and $\mathcal { V }$ denote the spaces of experimental designs, latent model parameters, and observations, respectively. The simulator generates samples from the prior predictive distribution, $\theta \sim p ( \theta )$ and $y \sim p _ { \mathrm { e v a l } } ( y \mid \theta , \xi )$ , which provide training data for amortized density, ratio, or score estimators targeting quantities such as the posterior $p ( \theta \mid \bar { y } , \xi )$ or surrogate likelihood $p ( y \mid \theta , \xi )$

## 2.2 MASK-CONDITIONED JOINT DIFFUSION MODELS

Diffusion-based SBI often trains an estimator for a particular inference task, such as posterior or likelihood estimation (Sharrock et al., 2024). Mask-conditioned joint diffusion models instead support different conditioning patterns within a single network (Gloeckler et al., 2024). Variables such as $y , \xi ,$ , and θ are represented by tokens encoding their identity, value, and whether they are latent or conditioned. For d modeled variables, a condition mask $\bar { M _ { C } } \in \{ 0 , 1 \} ^ { d }$ identifies those held fixed, with $M _ { C } = ( 0 , \dots , 0 )$ giving an unconditional model. An attention mask $M _ { E }$ incorporates the expert model’s graphical structure into the transformer (Weilbach et al., 2023).

Writing the tokenized variables collectively as $y _ { 0 } .$ , a prescribed forward perturbation kernel $p _ { t } ( y _ { t } \mid y _ { 0 } )$ provides noisy versions $y _ { t }$ . The masked input $y _ { t } ^ { M c }$ retains clean values for conditioned variables and noisy values for latent variables. The network learns the conditional score, the gradient of the log density with respect to latent variables, through masked denoising score matching (Hyvärinen, 2005; Vincent, 2011). We define

$$
\begin{array} { r } { \ell ( \phi , M _ { C } , t , y _ { 0 } , y _ { t } ) = \zeta ( t ) ( 1 - M _ { C } ) \Big ( s _ { \phi } ^ { M _ { E } } \big ( y _ { t } ^ { M _ { C } } , t \big ) - \nabla _ { y _ { t } } \log p _ { t } \big ( y _ { t } | y _ { 0 } \big ) \Big ) , } \end{array}
$$

where $\zeta ( t )$ weights the noise levels and $( 1 - M _ { C } )$ selects latent coordinates. We minimize the expected squared residual,

$$
\mathcal { L } ( \phi ) = \mathbb { E } _ { M _ { C } , t , y _ { 0 } , y _ { t } } \Big [ \big | \big | \ell \big ( \phi , M _ { C } , t , y _ { 0 } , y _ { t } \big ) \big | \big | _ { 2 } ^ { 2 } \Big ] ,
$$

yielding the pretrained (PT) model $p _ { \phi _ { 0 } } ( y , \theta \mid \xi )$ , where $\phi _ { 0 }$ denotes the PT network parameters. The choice of condition masks and their frequency during training influences conditional sampling performance (§ F).

## 2.3 SCORE-BASED DIFFUSION SAMPLING

The learned conditional score defines a sampler that reverses the noising used during pretraining. We use a variance-preserving SDE (VP-SDE), the continuous-time limit of the forward Gaussian noising chain in denoising diffusion probabilistic models (DDPMs) (Ho et al., 2020; Song et al., 2021). At fixed conditioning, let $y _ { t }$ denote the variables being generated and suppress the conditioning arguments. The forward process transforms $p _ { 0 } ( y _ { 0 } )$ toward a tractable Gaussian $p _ { T } ( y _ { T } )$ , and the learned score $s _ { \phi } ( y _ { t } , t ) \approx \nabla _ { y _ { t } } \log p _ { t } ( y _ { t } )$ gives the reverse dynamics,

$$
d y _ { t } = \left\{ \begin{array} { l l } { f ( y _ { t } , t ) d t + g ( t ) d w , } & { \mathrm { ( f o r w a r d ) } } \\ { [ f ( y _ { t } , t ) - g ( t ) ^ { 2 } s _ { \phi } ( y _ { t } , t ) ] d t + g ( t ) d \overleftarrow { w } , } & { \mathrm { ( r e v e r s e ) } , } \end{array} \right.
$$

where $w$ and $\stackrel {  } { w }$ are forward- and reverse-time Wiener processes, and $f$ and $g$ are the drift and diffusion coefficients (Anderson, 1982; Song et al., 2021). For the VP-SDE, $\begin{array} { r } { f ( y , \dot { t } ) = - \frac { 1 } { 2 } \beta ( t ) y } \end{array}$ and $g ( t ) = \sqrt { \beta ( t ) }$ for a nonnegative noise rate $\beta ( t )$ . Sampling integrates the reverse SDE from $y _ { T } \sim p _ { T }$ at $t = T \tan t = 0$ , holding conditioned variables fixed.

## 3 METHODS

## 3.1 FINE-TUNING TO ADDRESS MODEL MISSPECIFICATION

The PT joint surrogate provides a conditional likelihood $p _ { \phi _ { 0 } } ( y \mid \theta , \xi ) \approx p _ { \mathrm { e v a l } } ( y \mid \theta , \xi )$ , which inherits simulator misspecification relative to $\boldsymbol { p } ^ { * } ( y \mid \boldsymbol { \theta } , \boldsymbol { \xi } )$ . Following Cannon et al. (2022), we conceptualize misspecification via an unknown transformation that maps simulator outputs to observations. In our setting this is naturally a map between different sample spaces, since simulator outputs must be post-processed (e.g. normalized) to match the observation representation:

$$
y \sim p _ { \mathrm { e v a l } } ( \cdot \mid \theta , \xi ) , \qquad y _ { o } = T ( y ) , \qquad T : \mathcal { V } _ { \mathrm { s i m } } \to \mathcal { V } _ { \mathrm { o b s } } .
$$

![](images/e97795665f5dea31ff937b3cc2bcb656de5dd47e1043327f519a4f59710f2458.jpg)  
Figure 2: Likelihood fine-tuning of the joint diffusion surrogate. At fixed reference parameters and design, generated responses are evaluated against observations with a path-space KL penalty relative to the $\bar { \mathrm { P T } }$ model. Gradients through sampling update the token embeddings, FiLM layers, and score network, which also support posterior queries.

Under this representation, $\boldsymbol { p } ^ { * } ( \cdot \mid \boldsymbol { \theta } , \xi )$ is the pushforward $p ^ { * } ( \cdot \mid \theta , \xi ) = T _ { \# } p _ { \mathrm { e v a l } } ( \cdot \mid \theta , \xi )$ . Rather than specifying $T$ explicitly, we seek to learn an implicit misspecification correction that absorbs this transformation into the joint model’s parameters. We draw on diffusion fine-tuning to address simulator misspecification, adapting the surrogate likelihood to observed data. Specifically, we use the implicit diffusion method of Marion et al. (2025), which allows us to optimize agreement with observations by differentiating through likelihood sampling. We discuss alternative diffusion fine-tuning approaches and their suitability for SBI in § G.6.

In SBI, observations $y _ { o } .$ , or design–observation pairs $( y _ { o } , \xi )$ , provide a target for fine-tuning the surrogate likelihood. The learned correction must also inform the posterior over θ conditioned on those observations. We enable this transfer through token-aware Feature-wise Linear Modulation (FiLM) (Perez et al., 2018), which modulates token embeddings according to their identity and whether they are latent or conditioned. These embeddings are shared across likelihood and posterior queries in the masked transformer, allowing likelihood fine-tuning to update representations used for posterior inference; Figure 2 illustrates this mechanism, with embedding details in § G.7.

How can we fine-tune the likelihood $p _ { \phi } ( y \mid \theta , \xi )$ to observed data when the corresponding mechanistic parameters θ are unknown? For observed design–response pairs $( y _ { i } ^ { o } , \xi _ { i } )$ , we condition the surrogate likelihood on a fixed, simulator-aligned reference parameter $\theta ^ { \mathrm { r e f } }$ . In supervised settings, $\theta ^ { \mathrm { r e f } }$ may be provided by a labeled real-world dataset $\mathcal { D } _ { \mathrm { c a l } } = \{ ( \theta ^ { i } , y _ { o } ^ { i } ) \} _ { i = 1 } ^ { N _ { \mathrm { c a l } } }$ , where each observation $y _ { o } ^ { i }$ is paired with its ground-truth parameter $\theta ^ { i }$ , as in Wehenkel et al. (2025). When it is unknown, it may instead be an externally obtained best-fit estimate. In the BMP application, we use the domain-standard least-squares estimate $\theta ^ { \mathrm { L S R } } \in$ arg min<sub>θ</sub> $\textstyle \sum _ { i } \ell ( f ( \theta , \xi _ { i } ) , y _ { i } ^ { o } )$ , which provides a parameter configuration explicitly grounded in the expert simulator. We do not interpret $\theta ^ { \mathrm { L S R } }$ as a true parameter vector; it serves only as a fixed reference for adapting the likelihood to observed data.

For fixed $( \theta ^ { \mathrm { r e f } } , \xi )$ , we draw samples $y \sim p _ { \phi } ( y \mid \theta ^ { \mathrm { r e f } } , \xi )$ and evaluate their agreement with observations through a reward $R ( y ; y ^ { o } )$ . The diffusion sampler induces an output distribution $\pi ^ { \star } ( \phi ; \theta ^ { \mathrm { r e f } } , \xi )$ defined implicitly by the network parameters $\phi$ . We formulate fine-tuning as optimization of a functional $\bar { \mathcal F }$ of this sampled distribution, differentiating through the sampler with the adjoint method of Implicit Diffusion (Marion et al., 2025). Suppressing the fixed conditioning arguments, the endpoint-regularized problem is min ${ } _ { \phi } \ F ( \pi ^ { \star } ( \phi ) )$ , with

$$
\begin{array} { r } { \mathcal { F } ( p _ { \phi } ) : = - \mathbb { E } _ { y \sim p _ { \phi } } [ R ( y ; y ^ { o } ) ] + \lambda \operatorname { K L } ( p _ { \phi } \| \pi ^ { \star } ( \phi _ { 0 } ) ) , } \end{array}\tag{1}
$$

where $\lambda \geq 0$ penalizes departure from the PT output law. In our experiments, R is the negative mean squared error (MSE) between generated and observed responses. In implementation, we replace the endpoint KL by a path-space KL penalty, giving the stronger regularization described in §§ 3.2 and G.3. Alternative fine-tuning frameworks are discussed in § G.6. Surprisingly, anchoring likelihood fine-tuning at a fixed $\theta ^ { \mathrm { r e f } }$ does not reduce posterior inference to a point estimate — in the BMP example, the fine-tuned (FT) posterior marginals generally shift toward the reference relative to pretraining while retaining substantial spread (Figure 15).

## 3.2 LOCALIZING SIMULATOR MISSPECIFICATION WITH PATH-SPACE DIVERGENCE

For fixed $( \theta , \xi )$ , the reverse-SDE sampler underlying $\pi ^ { \star } ( \phi ; \theta , \xi )$ induces a path measure $\mathbb { P } _ { \phi } ^ { \theta , \xi }$ over trajectories $Y = ( Y _ { t } ) _ { t \in [ 0 , T ] }$ , whose generated endpoint distribution is $\pi ^ { \star } ( \phi ; \theta , \xi )$ . Fine-tuning from ϕ to ϕ therefore changes not only the endpoint distribution of generated responses but the probability law of the complete sampling trajectory, giving PT and FT path measures $\mathbb { P } _ { \phi _ { 0 } } ^ { \hat { \theta } , \xi }$ and $\mathbb { P } _ { \phi } ^ { \theta , \xi }$ . We quantify this change using path-space $\mathrm { K L , }$ which we use in place of the endpoint penalty in Eq. (1). Since the generated response is a projection of the complete diffusion trajectory, the data-processing inequality implies that path-space KL upper-bounds the corresponding endpoint divergence and can therefore capture adaptation not visible from the final samples alone. Under the regularity and integrability conditions stated in § G.2, Girsanov’s theorem gives

$$
\begin{array} { r } { \mathrm { K L } \Big ( \mathbb { P } _ { \phi } ^ { \theta , \xi } \Big \Vert \mathbb { P } _ { \phi _ { 0 } } ^ { \theta , \xi } \Big ) = \frac { 1 } { 2 } \mathbb { E } _ { \mathbb { P } _ { \phi } ^ { \theta , \xi } } \left[ \int _ { 0 } ^ { T } g ( t ) ^ { 2 } \left. s _ { \phi } ^ { y } ( Y _ { t } , t ; \theta , \xi ) - s _ { \phi _ { 0 } } ^ { y } ( Y _ { t } , t ; \theta , \xi ) \right. _ { 2 } ^ { 2 } d t \right] . } \end{array}\tag{2}
$$

We denote this discrepancy by $\mathcal { A } ( \theta , \xi )$ . Evaluating $\mathcal { A }$ at the same fixed reference parameter $\theta ^ { \mathrm { r e f } }$ used during fine-tuning yields a non-negative scalar field over design space, assigning each design a measure of the trajectory-level displacement between the PT and FT reverse-SDEs.

Stochastic optimal control (SOC) interpretation. Equation (2) admits a SOC interpretation. In our reverse-time convention, a control field $u ( x , t )$ perturbs a reference drift through $d X _ { t } ^ { u ^ { \bullet } } = [ b _ { 0 } ( X _ { t } ^ { u } , t ) +$ $\sigma ( t ) u ( X _ { t } ^ { u } , t ) ] d t + \sigma ( t ) d \overleftarrow { w } _ { t }$ , integrated from $t = T$ to $t = 0$ , inducing a controlled path measure $\mathbb { P } ^ { u }$ Taking the PT reverse-SDE as the reference and $\sigma ( t ) = g ( t ) I$ gives $\begin{array} { r } { \overline { { u } } _ { \phi } = - g ( t ) ( s _ { \phi } ^ { \bar { y } } - s _ { \phi _ { 0 } } ^ { y } ) } \end{array}$ . Hence, $\mathcal { A } ( \theta , \xi )$ is the quadratic control energy associated with transporting the PT path law to the FT one, connecting likelihood adaptation to SOC and KL-constrained path-space transport (Blessing et al., 2025).

Fisher geometry over conditioning variables. Beyond comparing PT and FT path laws at fixed $( \theta , \xi )$ , the same path-space formulation induces a local geometry over the conditioning variables themselves. Let $\bar { z } = ( \bar { \theta } , \xi )$ denote the continuous mechanistic and design coordinates, and fix the learned model ϕ. Assuming the induced path laws vary smoothly with z, nearby path measures satisfy the absolute-continuity conditions required by Girsanov’s theorem, and distinct nearby conditioning values induce distinct path laws, the family

$$
\mathcal { M } _ { \phi } = \left\{ \mathbb { P } _ { \phi } ^ { z } : z \in \Theta \times \Xi \right\} \subset \mathcal { P } \big ( C ( [ 0 , T ] ; \mathbb { R } ^ { d _ { y } } ) \big )
$$

can be treated locally as a statistical manifold (Figure 3). Here, $C ( [ 0 , T ] ; \mathbb { R } ^ { d _ { y } } )$ is the space of continuous trajectories $Y : [ 0 , T ]  \mathbb { R } ^ { d _ { y } }$ , and $\mathcal { P } ( \cdot )$ denotes probability measures on that space. With a shared diffusion schedule and terminal law, the same Girsanov argument gives

$$
\mathcal { T } _ { \phi } ^ { \mathrm { p a t h } } ( z ) : = \mathbb { E } _ { Y \sim \mathbb { P } _ { \phi } ^ { z } } \left[ \int _ { 0 } ^ { T } g ( t ) ^ { 2 } J _ { z } s _ { \phi } ^ { y } ( Y _ { t } , t ; z ) ^ { \top } J _ { z } s _ { \phi } ^ { y } ( Y _ { t } , t ; z ) d t \right] ,\tag{3}
$$

where $J _ { z } s _ { \phi } ^ { y } ~ = ~ \partial s _ { \phi } ^ { y } / \partial z$ Thus, $\mathcal { T } _ { \phi } ^ { \mathrm { p a t h } } ( z )$ is the local Fisher–Rao metric induced by KL divergence on the conditional reverse-SDE path laws (Amari & Nagaoka, 2000). When the conditioning parameterization is locally nonidentifiable, $\mathcal { T } _ { \phi } ^ { \mathrm { p a t h } } ( z )$ may be singular, consistent with the classical connection between local identifiability and nonsingularity of the Fisher information matrix (Rothenberg, $1 9 \hat { 7 } 1 )$ ). This includes the familiar case of mechanistic parameter non-identifiability, where distinct parameter combinations θ induce indistinguishable responses (Raue et al., 2009; Villaverde, 2019). The tangent space $T _ { \mathbb { P } _ { \phi } } \mathcal { M } _ { \phi }$ cap-

![](images/0d7ae50f3732bd36bb1cd6f5259ea3205ab3c0962db72a65eeb7159e71e3ac0e.jpg)  
Figure 3: Conditional path-law geometry under fine-tuning.

tures first-order perturbations of the conditional path law around $\mathbb { P } _ { \phi } ^ { z }$ , with $\mathcal { T } _ { \phi } ^ { \mathrm { p a t h } } ( z )$ defining the local metric on this space. We further discuss the parameter- and design-space structure of this geometry in $\ S \ G . 4$

## 3.3 COMPOSITIONAL SAMPLING

A simformer can represent arbitrary conditionals of a fixed joint distribution, but explicitly representing each design–observation pair in the joint makes the modeled state grow with the number of experiments and ties inference to a fixed number of observation tokens. Instead, for each observation $( y _ { j } , \xi _ { j } )$ we use the posterior score $s _ { \phi } ( \theta _ { t } , t ; y _ { j } , \xi _ { j } ) \approx \nabla _ { \theta _ { t } } \log p _ { t } ( \theta _ { t } \ | \ y _ { j } , \xi _ { j } )$ , and combine these per-observation scores at inference to approximate the posterior conditioned on $\mathcal { D } _ { n } = \{ ( y _ { j } , \xi _ { j } ) \} _ { j = 1 } ^ { n }$ Since the same score model is reused for every $( y _ { j } , \xi _ { j } )$ , the number and subset of observations can vary without changing or retraining the learned joint model. To achieve this, we first review the discrete F-NPSE construction of Geffner et al. (2023), then introduce a continuous-time extension.

Discrete compositional sampling. Under conditional independence given $\theta ,$ the target posterior factorizes as $\begin{array} { r } { \hat { p ( { \boldsymbol { \theta } } \mid { \mathcal D } _ { n } ) } \propto p ( { \boldsymbol { \theta } } ) ^ { \bar { 1 } - \tilde { n _ { \mathbf { \theta } } } } \prod _ { j = 1 } ^ { n } p ( { \boldsymbol { \theta } } \mid y _ { j } , \xi _ { j } ) } \end{array}$ . F-NPSE exploits this factorization by defining intermediate distributions whose scores combine the per-observation posterior scores above with the corresponding prior-score correction. In the DDPM setting (Ho et al., 2020), k indexes diffusion steps and $\alpha _ { k }$ denotes one-step signal retention, with signal scaling $\sqrt { \alpha _ { k } }$ and added noise variance $1 - \alpha _ { k }$ per coordinate. The resulting normalized reverse transition can be written as

$$
\widetilde { p } _ { k } ( \theta _ { k - 1 } \mid \theta _ { k } , \mathcal { D } _ { n } ) = \mathcal { N } \Bigg ( \theta _ { k - 1 } ; \mu _ { k } ^ { \mathrm { G } } , \frac { 1 - \alpha _ { k } } { \kappa _ { k } ^ { ( n ) } } I \Bigg ) , \qquad \kappa _ { k } ^ { ( n ) } = 1 + ( n - 1 ) ( 1 - \alpha _ { k } ) ,\tag{4}
$$

where $\mu _ { k } ^ { \mathrm { G } }$ combines the learned reverse scores for each $( y _ { j } , \xi _ { j } )$ with the prior-score correction implied by the posterior factorization above; its full form and assumptions are given in § A.2. The transition variance in Equation (4) decreases with the number of observations n and approaches $1 / n$ as $\alpha _ { k }  0$ . This transition-level contraction motivates a narrower cumulative noising path than the single-observation VP process, echoing the $n ^ { - 1 }$ covariance contraction of Bernstein–von Mises theory (van der Vaart, 1998, Ch. 10).

From discrete to continuous dynamics. To construct a continuous-time noising process, consider a hypothetical draw $\theta _ { 0 }$ from the desired posterior. Ordinary VP noising attenuates this draw by $\sqrt { \alpha ( t ) }$ and adds Gaussian noise with variance $1 - \alpha ( t )$ per coordinate, where $\alpha ( t )$ denotes cumulative squared signal retention. Motivated by the discrete contraction in Eq. (4), we retain this mean attenuation but prescribe a smaller accumulated noise variance ${ \cal V } _ { n } ( t ) = ( 1 - \alpha ( t ) ) / \kappa ^ { ( n ) } ( t )$ , where $\kappa ^ { ( n ) } ( t ) = 1 + ( n - 1 ) ( 1 - \alpha ( t ) )$ . This introduces observation-count dependence without estimating the posterior covariance. Keeping the ordinary VP-SDE drift $f ( t ) \stackrel {  } { = } \dot { \alpha } ( t ) / ( 2 \alpha ( t ) )$ (Song et al., $2 0 2 1 ) .$ , the variance evolution $\dot { V } _ { n } ( t ) = 2 f ( t ) V _ { n } ( t ) + g _ { n } ( t ) ^ { 2 }$ then determines the diffusion coefficient for this noising process

$$
g _ { n } ( t ) ^ { 2 } = \beta ( t ) \frac { 1 + \left( n - 1 \right) \left( 1 - \alpha ( t ) \right) ^ { 2 } } { [ \kappa ^ { ( n ) } ( t ) ] ^ { 2 } } , \qquad \beta ( t ) = - \frac { \dot { \alpha } ( t ) } { \alpha ( t ) } .\tag{5}
$$

The coefficient satisfies $g _ { n } ( t ) ^ { 2 } \leq \beta ( t )$ , with $g _ { 1 } ( t ) ^ { 2 } = \beta ( t )$ and $g _ { n } ( t ) ^ { 2 } / \beta ( t )  1 / n$ as $\alpha ( t )  0$ We call the resulting sampler F-NPSE-SDE. It uses $g _ { n } ( t )$ in the reverse-SDE predictor, which advances between noise levels, and in L optional unadjusted Langevin corrections, which explore the current compositional marginal at fixed diffusion time. For a concentrated marginal, the score can change rapidly over short distances, so a correction step may overshoot even when its direction points toward higher density. The smaller coefficient moderates score-driven drift and noise, reducing the risk of overshoot in stiff dynamics. Our controlled exact-score Gaussian example shows that these numerical issues arise even in a simple model and that the schedule reduces both step-size sensitivity and sampling error (§ A.5). This yields an inexpensive schedule depending only on diffusion time and observation count, with its derivation in § A.3.

Practical scaling. F-NPSE-SDE requires only score evaluations, avoiding the score Jacobians of JAC and auxiliary covariance estimation of GAUSS (Linhart et al., 2026). These costs become especially important in HBDS (§ 3.4), where compositional sampling is repeated within each block update. Computational complexity and observed runtime and memory limitations on the BMP model are discussed in § A.8.

## 3.4 HIERARCHICAL BLOCKWISE DIFFUSION SAMPLING

Naïve compositional sampling, which we refer to as pooled compositional sampling, treats all experimental observations as if they share a single posterior distribution. This ignores the fact that different groups of experiments may induce distinct posteriors, leading to suboptimal inference. For example, the BMP dataset comprises $K = 4$ cell lines consisting of a parent and three receptorknockdown lines. Each line has its own receptor state $R ^ { k }$ , shared across its ligand-dose experiments, while the biophysical parameters θ are shared across all four lines (Appendix C). We use a hierarchical Bayesian formulation (Gelman et al., 1995) with shared parameters θ and group-specific latent states $R ^ { 1 : K }$ . Given designs $\overset { \cdot } { \xi ^ { 1 : K } }$ and observations $y ^ { 1 : K }$ across K groups, we target the joint posterior $p ( \theta , R ^ { 1 : K } \mid y ^ { 1 : K } , \xi ^ { 1 : K } )$ . To approximately sample this posterior, HBDS initializes a shared parameter state and separate receptor states. At every reverse-diffusion time, each $R ^ { k }$ is updated with its own $( y ^ { k } , \xi ^ { k } )$ and current global θ, then θ is updated using all receptor blocks. Repeating yields samples with one globally consistent parameter vector and $\check { K }$ distinct local latent states without training a separate hierarchical estimator or tokenizing the complete dataset. The HBDS algorithm and analytic example are in §§ B and B.1.

## 4 EXPERIMENTS

We evaluate $g _ { n } ( t )$ on an exact toy Gaussian model and the Simple Likelihood, Complex Posterior (SLCP) Papamakarios et al. (2019) benchmarks (§ 4.1), hierarchical inference on a controlled toy (§ 4.2), and misspecification-aware fine-tuning (§ 4.3). The BMP application (§ 4.4) combines finetuning with HBDS across 940 observations, assessing predictive improvement and its relationship to design-localized path divergence.

## 4.1 EVALUATING F-NPSE-SDE

Exact-score Gaussian benchmark. We compare F-NPSE-SDE with GAUSS, JAC, and Langevin (Linhart et al., 2026), and the original F-NPSE (Geffner et al., 2023). The 10-dimensional Gaussian model (§ A.5) has correlation $\rho = 0 . 8$ and exact compositional scores and posteriors. Here, n counts observations, T predictor steps, and L Langevin corrections per predictor. $\mathbf { A } \mathbf { { t } } n = 1 0 0$ and $T = 4 0 0$ , Table 1 shows that F-NPSE-SDE is competitive with GAUSS and JAC in maximum sliced Wasserstein distance, matches F-NPSE, and improves over Langevin. Full sweep settings and results appear in § A.6.

SLCP benchmark. We next evaluate compositional sampling on SLCP, where the scores must be learned and the posterior is multimodal. Following the protocol of Linhart et al. (2026), we evaluate three independently trained single-observation score networks at $n \in \{ 1 , 1 4 , 3 0 \}$ , with all samplers sharing the same frozen network within each comparison. ${ \mathrm { A t } } n = 3 0 { , } { \mathrm { ~ \ ' G A U S S } }$ and JAC remain the most accurate, while F-NPSE-SDE improves over Langevin and F-NPSE in sliced Wasserstein distance and MMD (Table 1), without JAC’s score Jacobians or GAUSS’s auxiliary DDIM rollouts. Training, stabilization, and results across observation counts and network seeds are detailed in § A.7.

Table 1: Comparison of compositional samplers on the Gaussian benchmark $( n = 1 0 0 , T = 4 0 0 )$ and SLCP (n = 30). Entries are mean standard deviation over five seeds for Gaussian and 25 test instances using one network seed for SLCP.
<table><tr><td rowspan="2">Method</td><td>Gaussian  $( T = 4 0 0 , n = 1 0 0 )$ </td><td colspan="3">SLCP (n = 30)</td></tr><tr><td>max-sW↓</td><td>sW↓</td><td>MMD↓</td><td>C2ST → 0.5</td></tr><tr><td>GAUSS</td><td> $0 . 2 4 \pm 0 . 2 3$ </td><td> $0 . 4 5 \pm 0 . 1 8$ </td><td> $0 . 0 5 \pm 0 . 1 6$ </td><td> $0 . 9 7 \pm 0 . 0 4$ </td></tr><tr><td>JAC</td><td> $0 . 2 6 \pm 0 . 2 2$ </td><td> $0 . 5 3 \pm 0 . 1 9$ </td><td> $0 . 0 5 \pm 0 . 1 4$ </td><td> $0 . 9 5 \pm 0 . 0 5$ </td></tr><tr><td>Langevin</td><td> $0 . 5 2 \pm 0 . 5 0$ </td><td> $0 . 8 1 \pm 0 . 2 8$ </td><td> $0 . 1 3 \pm 0 . 2 2$ </td><td> $0 . 9 7 \pm 0 . 0 3$ </td></tr><tr><td>F-NPSE</td><td> $0 . 2 9 \pm 0 . 2 4$ </td><td> $0 . 8 3 \pm 0 . 2 9$ </td><td> $0 . 1 1 \pm 0 . 1 3$ </td><td> $0 . 9 8 \pm 0 . 0 3$ </td></tr><tr><td>F-NPSE-SDE (L = 1)</td><td> $0 . 2 9 \pm 0 . 2 5$ </td><td> $0 . 7 4 \pm 0 . 3 1$ </td><td> $0 . 1 0 \pm 0 . 1 5$ </td><td> $0 . 9 8 \pm 0 . 0 3$ </td></tr><tr><td>F-NPSE-SDE (L = 5)</td><td> $0 . 3 1 \pm 0 . 2 7$ </td><td> $0 . 7 3 \pm 0 . 2 8$ </td><td> $0 . 1 0 \pm 0 . 1 5$ </td><td> $0 . 9 8 \pm 0 . 0 3$ </td></tr></table>

## 4.2 EVALUATING POOLED COMPOSITIONAL SAMPLING AND HBDS

Hierarchical toy model. We first isolate the effect of hierarchical structure in a controlled model with a shared continuous parameter $\theta \in \mathbb { R }$ and group-specific latent states $z _ { g } \in \{ - 1 , + 1 \}$ ,

$$
\begin{array} { r } { z _ { g } \sim \mathrm { R a d } \left( \frac { 1 } { 2 } \right) , \qquad \theta \sim \mathcal { N } ( 0 , 1 ) , \qquad x _ { g i } \mid \theta , z _ { g } \sim \mathcal { N } ( 5 z _ { g } + \theta , 1 ) . } \end{array}
$$

Here ${ \mathrm { R a d } } ( { \textstyle { \frac { 1 } { 2 } } } )$ assigns equal probability to the two signs. We compare learned-score samplers using $n = 2 0$ observations generated with $\theta \ = \ 1$ , and ten samples from $z ~ = ~ + 1$ and $z = - 1$ . HBDS permits each group to select its own latent mode, whereas pooled compositional sampling ties all groups to a single latent state. Figure 4 shows pooled sampling cannot represent groups requiring different latent modes, while HBDS maintains separate groupspecific states together with a shared θ. The analytic posterior structure and additional scorebased diagnostics are given in § B.1. We further evaluate HBDS on the BMP model in $\ S 4 . 4$

![](images/8392a93d2f5638a312be5a538684c094dd82487d428ffd435d822ba5336c8b36.jpg)  
Figure 4: (Left) Pooled sampling uses one shared latent. $( R i g h t )$ HBDS retains separate group latents with a shared θ. Colors identify the generating groups; plotted latent states are continuous sampler outputs. Dotted lines mark the generating $\theta = 1$

## 4.3 FINE-TUNING UNDER SIMULATOR MISSPECIFICATION

Fine-tuning and path-divergence localization. To study how fine-tuning responds to model misspecification and whether path divergence localizes the resulting adaptation, we use a toy linear-Gaussian simulator $y \mid \theta , \xi \sim \overset { \cdot } { \sim } \mathcal { N } ( \xi ^ { \top } \theta , \overset { \cdot } { \sigma } ^ { 2 } )$ with $\theta \sim \mathcal { N } ( 0 , I _ { 2 } )$ , designs $\bar { \xi } \in [ - 2 , 2 ] ^ { 2 } ,$ , and $\sigma = 0 . 2$ At a fixed generating parameter $\theta ^ { \star }$ , the noise-free observation response is $\bar { y } _ { o } ( \xi ) \overset { \cdot } { = } \xi ^ { \top } \theta ^ { \star } + h ( \xi )$ where h is a localized Gaussian bump omitted from the simulator. We fine-tune on ten observations with Gaussian noise of variance $\sigma ^ { 2 }$ , then assess where the correction helps or hurts across design space. We then estimate the $\mathrm { P T }$ and FT likelihood means from 16 samples per design and measure their absolute errors against $y _ { o } ( \xi )$ , giving the pointwise error maps $e _ { \mathrm { P T } } ( \xi )$ and $e _ { \mathrm { F T } } ( \xi )$ in Figure 5. We measure mean absolute error (MAE) averages over 1,681 grid points. Fine-tuning reduces error near the bump but adds errors elsewhere; however, employing autoguidance (Karras et al., 2024) to blend the PT & FT models lowers MAE to 0.12 (Figure 12). Without autoguidance, mean $\mathcal { A } ( \theta ^ { \mathrm { L S R } } , \xi )$ inside the misspecified region $( h \ge 0 . 1 )$ is 4.73 times that outside and correlates with h (Pearson $r = 0 . 7 5 )$ . Path divergence and error maps show where fine-tuning helps or over-adapts, a trade-off we explore through regularization and autoguidance sweeps in $\ S \ G \mathcal { G }$

SBI misspecification comparison. On BMP (§ 4.4), we compare with Flow Matching Corrected Posterior Estimation (FMCPE), which corrects an NPE using calibration pairs (Ruhlmann et al., 2026), here constructed from LSR fits. FMCPE uses subset-specific posterior models, whereas FT also adapts the likelihood and supports flexible conditioning, composition, and HBDS. Across the FMCPE observation- and LSR-budget sweep, FT achieves the lowest mean median predictive distance, while FMCPE-LSR has lower best-draw RMSE (Eqs. (64) and (65) and Table 7). Because both metrics evaluate predictions through the original simulator, the comparison remains subject to its misspecification. We provide the full evaluation protocol alongside PT comparisons and calibration diagnostics in $\ S \ G . 8$

![](images/ee6257b4c33bbc7c0e8ec2016268fd4ec4443053c4830f38a1d538d990b32e21.jpg)

![](images/29adfa284f3eae7520b78830c9662ed90664b567743cdce60516142836155b06.jpg)

![](images/abc2e3dfc2b83f155294d76236cb44d06cbb37385a0aece9d38ff532274fe7f5.jpg)

![](images/3a574bb663c083048aee24678cbab7ed25a0ca70360b53108cf632ae112ba925.jpg)  
Figure 5: Likelihood adaptation and prediction error in the analytic toy at a fixed LSR reference. (Left to right) Known discrepancy $h ( \xi )$ , PT absolute error $e _ { \mathrm { P T } } ( \xi )$ , FT absolute error $e _ { \mathrm { F T } } ( \xi )$ , and path divergence $A ( \theta ^ { \mathrm { r e f } } , \xi )$

![](images/e2704b28d1e08cbe60c49ed33228b453552d7c4b89e19ae6ecbc3dfe7f72e2a6.jpg)  
Figure 6: BMP predictions and design-localized adaptation. (Left) BMP4/BMP10 competition in NMuMG. (Middle) The same competition in BMPR2 KD. (Right) BMP4 titration in NMuMG. From top to bottom: LSR and FT predictions against observations; low (L), middle (M), and high (H) LSR cost groups; experiment-normalized path divergence ${ \mathcal { A } } ( \theta , \xi )$ ; and reduction in absolute prediction error from LSR to the FT bootstrap median.

## 4.4 BMP APPLICATION

We combine fine-tuning with HBDS to infer shared biophysical parameters and cell-line-specific receptor states from 940 observations across four cell lines, using the model, priors, and training procedures described in $\ S \ S \ C , D ,$ F and G. We first hold parameters at the LSR reference to isolate how fine-tuning changes predictions. The bottom row of Figure 6 reports $\Delta e _ { \mathrm { L S R } } ( \xi )$ , the LSR simulator’s absolute error against $y _ { o } ( \xi )$ minus that of the FT bootstrap median, on a symmetriclogarithmic axis that is linear near zero. Positive values favor FT, with improvements throughout the BMPR2-knockdown competition series and at lower BMP4 doses. In the BMP4 titration, these gains generally coincide with larger path divergence, but adaptation in the NMuMG competition series often increases error, showing why the magnitude of adaptation must be interpreted alongside its predictive benefit. The LSR cost-group analysis adds context to these local differences, showing that globally higher-cost fits can perform better at individual designs (§ G.10). When we use the FT model with $\lambda \stackrel { - } { = } 5 \times 1 0 ^ { - 4 }$ for posterior inference, HBDS yields RMSE 0.34 and median distance 14.94, compared with 0.36 and 66.77 under pooled sampling. We examine the contributions of the sampler, FiLM, and regularization, together with the aggregate LSR comparison, in Tables 3 to 5.

## 5 DISCUSSION

Together, these methods enable flexible diffusion SBI across changing observation sets, hierarchical latent structure, and simulator misspecification, with the BMP model providing a large-scale scientific stress test. Rather than requiring a PT SBI model to anticipate this downstream structure, our methods adapt it to the problem at hand.

Limitations and future work. Compositional sampling and HBDS both rely on approximations to intermediate-time posteriors. More accurate intermediate targets and faster sampling with learned flow maps (Boffi et al., 2025; Song et al., 2023) could improve compositional inference and HBDS block updates. Fine-tuning at a fixed reference parameter $\theta ^ { \mathrm { r e f } }$ can improve misspecified regions while degrading regions that were already well modeled. These results suggest tuning adaptation across experimental designs rather than imposing one global regularization strength, with predictive error and path divergence guiding where to focus updates and where to preserve the PT model. Path-space Fisher geometry over $( \theta , \xi ) \left( \ S \operatorname { G } . 4 \right)$ further suggests design-dependent adaptation, particleor geometry-based refinement of $\dot { \theta } ^ { \mathrm { r e f } }$ , and diffusion-based Bayesian experimental design over ξ (Iollo et al., 2025).

## ACKNOWLEDGEMENTS

We thank the Elowitz Lab for helpful feedback. We also thank UCI’s Research Cyberinfrastructure Center (RCIC) for providing and supporting the HPC3 computing resources used in this work (Research Cyberinfrastructure Center, 2026). This research was partially funded by the National Institute of General Medical Sciences (NIGMS) of the National Institutes of Health (NIH) under award number 1F31GM145188-01.

## REFERENCES

Shun-ichi Amari and Hiroshi Nagaoka. Methods ofinformation geometry, volume 191. American Mathematical Soc., 2000.

Brian D.O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313–326, 1982. ISSN 0304-4149. doi: 10.1016/0304-4149(82)90051-5.

Yaron E Antebi, James M Linton, Heidi Klumpe, Bogdan Bintu, Mengsha Gong, Christina Su, Reed McCardell, and Michael B Elowitz. Combinatorial signal perception in the bmp pathway. Cell, 170(6):1184–1196, 2017.

Jonas Arruda, Niels Bracher, Ullrich Köthe, Jan Hasenauer, and Stefan T. Radev. Diffusion models in simulation-based inference: A tutorial review. arXiv preprint arXiv:2512.20685, 2025.

Jonas Arruda, Vikas Pandey, Catherine Sherry, Margarida Barroso, Xavier Intes, Jan Hasenauer, and Stefan T. Radev. Compositional amortized inference for large-scale hierarchical bayesian models. In International Conference on Learning Representations, 2026.

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th Annual International Conference on Machine Learning, pp. 41–48. Association for Computing Machinery, 2009. doi: 10.1145/1553374.1553380.

Christopher M. Bishop. Mixture density networks. Technical Report NCRG/94/004, Neural Computing Research Group, Aston University, Birmingham, UK, 1994.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, pp. 4965–4987, 2024.

Denis Blessing, Julius Berner, Lorenz Richter, Carles Domingo-Enrich, Yuanqi Du, Arash Vahdat, and Gerhard Neumann. Trust region constrained measure transport in path space for stochastic optimal control and inference. In Advances in Neural Information Processing Systems, volume 38, 2025.

Jan Boelts. Model misspecification in simulation-based inference: Recent advances and open challenges. ICLR Blogposts 2026, April 2026. Published April 27, 2026.

Nicholas M. Boffi, Michael S. Albergo, and Eric Vanden-Eijnden. Flow map matching with stochastic interpolants: A mathematical framework for consistency models. Transactions on Machine Learning Research, 2025.

Patrick Cannon, Daniel Ward, and Sebastian M. Schmon. Investigating the impact of model misspecification in neural simulation-based inference. arXiv preprint arXiv:2209.01845, 2022. doi: 10.48550/arxiv.2209.01845.

Pierre Cesar, Sofya Dymchenko, Abhishek Purandare, and Bruno Raffin. Learning where to simulate: Generative active sampling for online PDE surrogate training. arXiv preprint arXiv:2606.09949, 2026.

Giovanni Charles, Cosmo Santoni, Seth Flaxman, and Elizaveta Semenova. Tokenised flow matching for hierarchical simulation based inference. arXiv preprint arXiv:2604.20723, 2026.

Kevin Clark, Paul Vicol, Kevin Swersky, and David Fleet. Directly fine-tuning diffusion models on differentiable rewards. In International Conference on Learning Representations, pp. 4793–4822, 2024.

Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference. Proceedings ofthe National Academy ofSciences, 117(48):30055–30062, November 2020. ISSN 0027-8424. doi: 10.1073/pnas.1912789117. arXiv: 1911.01429 Publisher: Proceedings of the National Academy of Sciences.

Riccardo De Santi, Marin Vlastelica, Ya-Ping Hsieh, Zebang Shen, Niao He, and Andreas Krause. Flow density control: Generative optimization beyond entropy-regularized fine-tuning. In Advances in Neural Information Processing Systems, volume 38, pp. 11056–11088, 2025.

Daniel Dimitrov, Stefan Schrod, Martin Rohbeck, and Oliver Stegle. Interpretation, extrapolation and perturbation of single cells. Nature Reviews Genetics, pp. 1–22, 2026.

Robert M. Dirks, Justin S. Bois, Joseph M. Schaeffer, Erik Winfree, and Niles A. Pierce. Thermodynamic analysis of interacting nucleic acid strands. SIAM Review, 49(1):64–88, 2007. doi: 10.1137/060651100.

Carles Domingo i Enrich, Michal Drozdzal, Brian Karrer, and Ricky T. Q. Chen. Adjoint matching: Fine-tuning flow and diffusion generative models with memoryless stochastic optimal control. In International Conference on Learning Representations, pp. 53791–53846, 2025.

Conor Durkan, Iain Murray, and George Papamakarios. On contrastive learning for likelihood-free inference. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 2771–2781. PMLR, 2020.

Alain Durmus and Éric Moulines. Nonasymptotic convergence analysis for the unadjusted Langevin algorithm. The Annals ofApplied Probability, 27(3):1551–1587, 2017. doi: 10.1214/16-AAP1238.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK: Reinforcement learning for fine-tuning text-to-image diffusion models. In Advances in Neural Information Processing Systems, volume 36, pp. 79858–79885, 2023.

Richard Gao, Michael Deistler, and Jakob H Macke. Generalized bayesian inference for scientific simulators via amortized cost estimation. In Advances in Neural Information Processing Systems, volume 36, pp. 80191–80219, 2023.

Tomas Geffner, George Papamakarios, and Andriy Mnih. Compositional score modeling for simulation-based inference. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 11098–11116. PMLR, 23–29 Jul 2023.

Andrew Gelman, John B Carlin, Hal S Stern, and Donald B Rubin. Bayesian data analysis. Chapman and Hall/CRC, 1995.

I. V. Girsanov. On transforming a certain class of stochastic processes by absolutely continuous substitution of measures. Theory of Probability & Its Applications, 5(3):285–301, 1960. doi: 10.1137/1105027.

Manuel Gloeckler, Michael Deistler, Christian Dietrich Weilbach, Frank Wood, and Jakob H. Macke. All-in-one simulation-based inference. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 15735–15766. PMLR, 21–27 Jul 2024.

David Greenberg, Marcel Nonnenmacher, and Jakob Macke. Automatic posterior transformation for likelihood-free inference. In International Conference on Machine Learning, pp. 2404–2414. PMLR, 2019.

Daniel Habermann, Marvin Schmitt, Lars Kühmichel, Andreas Bulling, Stefan T. Radev, and Paul-Christian Bürkner. Amortized bayesian multilevel models. Bayesian Analysis, pp. 1–30, 2025. doi: 10.1214/25-BA1570.

Aaron J. Havens, Benjamin Kurt Miller, Bing Yan, Carles Domingo-Enrich, Anuroop Sriram, Daniel S. Levine, Brandon M. Wood, Bin Hu, Brandon Amos, Brian Karrer, Xiang Fu, Guan-Horng Liu, and Ricky T. Q. Chen. Adjoint sampling: Highly scalable diffusion samplers via adjoint matching. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 22204–22237. PMLR, 2025.

Lukas Heinrich, Siddharth Mishra-Sharma, Chris Pollard, and Philipp Windischhofer. Hierarchical neural simulation-based inference over event ensembles. Transactions on Machine Learning Research, 2024, 2024. ISSN 2835-8856.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, 2020.

Daolang Huang, Ayush Bharti, Amauri Souza, Luigi Acerbi, and Samuel Kaski. Learning robust statistics for simulation-based inference under model misspecification. In Advances in Neural Information Processing Systems, volume 36, pp. 7289–7310, 2023a.

Peide Huang, Xilun Zhang, Ziang Cao, Shiqi Liu, Mengdi Xu, Wenhao Ding, Jonathan Francis, Bingqing Chen, and Ding Zhao. What went wrong? closing the sim-to-real gap via differentiable causal discovery. In Proceedings of The 7th Conference on Robot Learning, volume 229 of Proceedings ofMachine Learning Research, pp. 734–760. PMLR, 2023b.

Aapo Hyvärinen. Estimation of non-normalized statistical models by score matching. Journal of Machine Learning Research, 6(24):695–709, 2005.

Jacopo Iollo, Christophe Heinkelé, Pierre Alliez, and Florence Forbes. Bayesian experimental design via contrastive diffusions. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 73125–73151, 2025.

Cristian Perez Jensen, Luca Schaufelberger, Riccardo De Santi, Kjell Jorner, and Andreas Krause. Value matching: Scalable and gradient-free reward-guided flow adaptation. In International Conference on Learning Representations, 2026.

Tero Karras, Miika Aittala, Tuomas Kynkäänniemi, Jaakko Lehtinen, Timo Aila, and Samuli Laine. Guiding a diffusion model with a bad version of itself. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9798331314385.

Heidi E Klumpe, Matthew A Langley, James M Linton, Christina J Su, Yaron E Antebi, and Michael B Elowitz. The context-dependent, combinatorial logic of bmp signaling. Cell systems, 13(5):388–407, 2022.

Julia Linhart, Alexandre Gramfort, and Pedro Rodrigues. ℓ-C2ST: Local diagnostics for posterior approximations in simulation-based inference. In Advances in Neural Information Processing Systems, volume 36, pp. 56384–56410, 2023.

Julia Linhart, Gabriel Victorino Cardoso, Alexandre Gramfort, Sylvain Le Corff, and Pedro L. C. Rodrigues. Diffusion posterior sampling for simulation-based inference in tall data settings. Transactions on Machine Learning Research, 2026.

Jan-Matthis Lueckmann, Pedro J Goncalves, Giacomo Bassetto, Kaan Öcal, Marcel Nonnenmacher, and Jakob H Macke. Flexible statistical inference for mechanistic models of neural dynamics. In Advances in Neural Information Processing Systems, volume 30, 2017.

Jan-Matthis Lueckmann, Giacomo Bassetto, Theofanis Karaletsos, and Jakob H. Macke. Likelihood free inference with emulator networks. In Proceedings of The 1st Symposium on Advances in Approximate Bayesian Inference, volume 96 of Proceedings ofMachine Learning Research, pp. 32–53. PMLR, 2019.

Abbas Mammadov, Jerry Y. Huang, Justin Lin, Partha Kaushik, Sheel Shah, Kartik Nair, Yee Whye Teh, and Nicholas M. Boffi. Wtf?! simulation-free reinforcement learning with wasserstein-tilted flow maps, 2026.

Pierre Marion, Anna Korba, Peter Bartlett, Mathieu Blondel, Valentin De Bortoli, Arnaud Doucet, Felipe Llinares-López, Courtney Paquette, and Quentin Berthet. Implicit diffusion: Efficient optimization through stochastic sampling. In Proceedings ofthe 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings ofMachine Learning Research, pp. 1999–2007. PMLR, 2025.

Benjamin Kurt Miller, Christoph Weniger, and Patrick Forré. Contrastive Neural Ratio Estimation. In Advances in Neural Information Processing Systems, volume 35, 2022.

Aayush Mishra, Daniel Habermann, Marvin Schmitt, Stefan T Radev, and Paul-Christian Bürkner. Ro bust amortized bayesian inference with self-consistency losses on unlabeled data. In International Conference on Learning Representations, 2026.

Sinno Jialin Pan and Qiang Yang. A survey on transfer learning. IEEE Transactions on Knowledge and Data Engineering, 22:1345–1359, 2010.

George Papamakarios and Iain Murray. Fast ϵ-free inference of simulation models with Bayesian conditional density estimation. In Advances in Neural Information Processing Systems, pp. 1036– 1044, 2016.

George Papamakarios, David C. Sterratt, and Iain Murray. Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows. In Proceedings ofthe Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings of Machine Learning Research, pp. 837–848. PMLR, 2019.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 32, 2018. doi: 10.1609/aaai.v32i1.11671.

Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739v2, 2023. Version 2. Subsequently withdrawn and subsumed by arXiv:2407.08737.

Andreas Raue, Clemens Kreutz, Thomas Maiwald, Julie Bachmann, Marcel Schilling, Ursula Klingmüller, and Jens Timmer. Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. Bioinformatics, 25(15):1923–1929, 2009.

Research Cyberinfrastructure Center. Facilities managed by Research Cyberinfrastructure Center (RCIC) at UC Irvine. University of California, Irvine, April 2026.

Gareth O. Roberts and Richard L. Tweedie. Exponential convergence of Langevin distributions and their discrete approximations. Bernoulli, 2(4):341–363, 1996. doi: 10.2307/3318418.

Thomas J Rothenberg. Identification in parametric models. Econometrica: Journal of the Econometric Society, pp. 577–591, 1971.

Pierre-Louis Ruhlmann, Michael Arbel, Florence Forbes, and Pedro L. C. Rodrigues. Flow matching calibration for simulation-based inference under model misspecification. In International Conference on Machine Learning, 2026.

Marvin Schmitt, Paul-Christian Bürkner, Ullrich Köthe, and Stefan T. Radev. Detecting model misspecification in amortized bayesian inference with neural networks. In Pattern Recognition. DAGM GCPR 2023, volume 14264 of Lecture Notes in Computer Science, pp. 541–557. Springer, 2024. doi: 10.1007/978-3-031-54605-1\_35.

Louis Sharrock, Jack Simons, Song Liu, and Mark Beaumont. Sequential neural score estimation: Likelihood-free inference with conditional score based diffusion models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 44565–44602. PMLR, 2024.

Freddie Bickford Smith, Jannik Kossen, Eleanor Trollope, Mark van der Wilk, Adam Foster, and Tom Rainforth. Rethinking aleatoric and epistemic uncertainty. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

Henry D. Smith, Nathaniel L. Diamant, and Brian L. Trippe. Calibrating generative models to distributional constraints. In International Conference on Machine Learning, 2026.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 32211–32252. PMLR, 2023.

Christina J Su, Arvind Murugan, James M Linton, Akshay Yeluri, Justin Bois, Heidi Klumpe, Matthew A Langley, Yaron E Antebi, and Michael B Elowitz. Ligand-receptor promiscuity enables cellular addressing. Cell systems, 13(5):408–425, 2022.

Camille Touron, Gabriel V. Cardoso, Julyan Arbel, and Pedro L. C. Rodrigues. Theoretical guidelines for annealed Langevin dynamics in compositional simulation-based inference. arXiv preprint arXiv:2605.21253, 2026.

Masatoshi Uehara, Yulai Zhao, Tommaso Biancalani, and Sergey Levine. Understanding reinforcement learning-based fine-tuning of diffusion models: A tutorial and review. arXiv preprint arXiv:2407.13734, 2024a.

Masatoshi Uehara, Yulai Zhao, Kevin Black, Ehsan Hajiramezanali, Gabriele Scalia, Nathaniel Lee Diamant, Alex M. Tseng, Tommaso Biancalani, and Sergey Levine. Fine-tuning of continuous-time diffusion models as entropy-regularized control. arXiv preprint arXiv:2402.15194, 2024b.

Masatoshi Uehara, Yulai Zhao, Kevin Black, Ehsan Hajiramezanali, Gabriele Scalia, Nathaniel Lee Diamant, Alex M. Tseng, Sergey Levine, and Tommaso Biancalani. Feedback efficient online fine-tuning of diffusion models. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 48892–48918. PMLR, 2024c.

Masatoshi Uehara, Yulai Zhao, Ehsan Hajiramezanali, Gabriele Scalia, Gokcen Eraslan, Avantika Lal, Sergey Levine, and Tommaso Biancalani. Bridging model-based optimization and generative modeling via conservative fine-tuning of diffusion models. In Advances in Neural Information Processing Systems, volume 37, pp. 127511–127535, 2024d.

A. W. van der Vaart. Asymptotic Statistics. Cambridge University Press, 1998. doi: 10.1017/CBO9 780511802256.

Alejandro F. Villaverde. Observability and structural identifiability of nonlinear biological systems. Complexity, 2019(1):8497093, 2019. doi: https://doi.org/10.1155/2019/8497093.

Pascal Vincent. A connection between score matching and denoising autoencoders. Neural Comput., 23(7):1661–1674, July 2011. ISSN 0899-7667. doi: 10.1162/NECO\_a\_00142.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8228–8238, 2024.

Daniel Ward, Patrick Cannon, Mark Beaumont, Matteo Fasiolo, and Sebastian Schmon. Robust neural posterior estimation and statistical model criticism. In Advances in Neural Information Processing Systems, volume 35, pp. 33845–33859, 2022.

Antoine Wehenkel, Juan L. Gamella, Ozan Sener, Jens Behrmann, Guillermo Sapiro, Joern-Henrik Jacobsen, and Marco Cuturi. Addressing misspecification in simulation-based inference through data-driven calibration. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 65949–65980. PMLR, 2025.

Christian Weilbach, William Harvey, and Frank Wood. Graphically structured diffusion models. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

Christopher Williams, Andrew Campbell, Arnaud Doucet, and Saifuddin Syed. Score-optimal diffusion schedules. In Advances in Neural Information Processing Systems, volume 37, pp. 107960–107983, 2024.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. ImageReward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems, volume 36, pp. 15903–15935, 2023.

## APPENDICES

A Compositional sampling details 17   
A.1 Existence of cumulative transition score 17   
A.2 Derivation of the reverse transition kernel 17   
A.3 Deriving the continuous-time diffusion coefficient 19   
A.4 Langevin refinement and numerical accuracy 21   
A.5 Experimental evaluation 23   
A.6 Exact-score Gaussian benchmark 25   
A.7 SLCP benchmark 25   
A.8 Computational cost of compositional samplers 26   
A.9 Compositional sampling related works 26   
B Hierarchical blockwise diffusion sampling details 27   
B.1 Toy simulator model example 28   
B.2 Hierarchical SBI related works 29   
C BMP signaling pathway mathematical model 30   
D Cell receptor hierarchical modeling 30   
E Additional BMP experimental results 32   
E.1 Calibrated receptor diffusion 32   
E.2 BMP posterior-sampling ablations 33   
F Pretraining details 33   
G Fine-tuning and path-space details 34   
G.1 Fine-tuning objective and optimization 34   
G.2 Path-space formulation and Girsanov’s theorem 35   
G.3 Stochastic optimal control connection 36   
G.4 Fisher–Rao optimization analysis . 37   
G.5 Related work on simulator misspecification in SBI 40   
G.6 Related work on diffusion fine-tuning 40   
G.7 Embedding-based transfer learning 43   
G.8 FMCPE-LSR baseline 43   
G.9 Design-dependent misspecification toy 44   
G.10 BMP misspecification evaluation 46   
H AI use statement 51

## A COMPOSITIONAL SAMPLING DETAILS

## A.1 EXISTENCE OF CUMULATIVE TRANSITION SCORE

We begin with a proposition for the existence of a cumulative transition score.

Proposition A.1 (Existence of a cumulative transition score). Under the reverse–forward consistency assumption in Eq. (6), we construct a cumulative transition $\tilde { p } _ { k } ( \theta _ { k - 1 } | \theta _ { k } )$ whose score combines the per-observation reverse scores with the common forward-kernel correction. This gives the compositional transition used below.

Proof(existence ofa cumulative transition kernel). Fix a discrete step $k \in \{ 1 , \ldots , K \}$ . For each observation $y _ { j }$ let $p _ { k } ^ { j } ( \theta _ { k } )$ denote the marginal at diffusion time $t _ { k }$ and $q _ { k } \mathopen { } \mathclose \bgroup ( \theta _ { k } \mathopen { } \mathclose \bgroup | \theta _ { k - 1 } \aftergroup \egroup )$ the common forward kernel. Assume the (exact) reverse–forward consistency

$$
p _ { k } ^ { j } ( \theta _ { k } ) p _ { k } ^ { j } ( \theta _ { k - 1 } | \theta _ { k } ) \ = \ p _ { k - 1 } ^ { j } ( \theta _ { k - 1 } ) q _ { k } ( \theta _ { k } | \theta _ { k - 1 } ) \qquad { \mathrm { f o r ~ e a c h ~ } } j \in \{ 1 , \dots , n \} .\tag{6}
$$

We seek a transition kernel $\tilde { p } _ { k } ( \theta _ { k - 1 } | \theta _ { k } )$ that satisfies the composition condition (cf. Eq. (24) of Geffner et al. (2023))

$$
\tilde { p } _ { k - 1 } ( \theta _ { k - 1 } ) = \int \tilde { p } _ { k } ( \theta _ { k } ) \tilde { p } _ { k } ( \theta _ { k - 1 } | \theta _ { k } ) d \theta _ { k } .\tag{7}
$$

where $\begin{array} { r } { \tilde { p } _ { k } ( \theta _ { k } ) ~ = ~ Z _ { k } ^ { - 1 } \prod _ { i } ^ { n } p _ { k } ^ { j } ( \theta _ { k } ) } \end{array}$ . Following Geffner et al. (2023), we rewrite Eq. (7) (cf. Eqs. (25)–(26) therein) by using Eq. (6) to express ratios of marginals and reverse kernels:

$$
\tilde { p } _ { k - 1 } ( \theta _ { k - 1 } ) = \int d \theta _ { k } p _ { k } ^ { 1 } ( \theta _ { k } ) \frac { q _ { k } ( \theta _ { k } | \theta _ { k - 1 } ) } { p _ { k } ^ { 2 } ( \theta _ { k - 1 } | \theta _ { k } ) } \cdot \cdot \cdot \frac { q _ { k } ( \theta _ { k } | \theta _ { k - 1 } ) } { p _ { k } ^ { n } ( \theta _ { k - 1 } | \theta _ { k } ) } \frac { Z _ { k } } { Z _ { k - 1 } } \tilde { p } _ { k } ( \theta _ { k - 1 } | \theta _ { k } ) .
$$

One way to satisfy this equality, as noted in Geffner et al. (2023), is to set

$$
\tilde { p } _ { k } ( \theta _ { k - 1 } | \theta _ { k } ) = p _ { k } ^ { 1 } ( \theta _ { k - 1 } | \theta _ { k } ) \frac { Z _ { k } } { Z _ { k - 1 } } \frac { p _ { k } ^ { 2 } ( \theta _ { k - 1 } | \theta _ { k } ) } { q _ { k } ( \theta _ { k } | \theta _ { k - 1 } ) } \cdot \cdot \cdot \frac { p _ { k } ^ { n } ( \theta _ { k - 1 } | \theta _ { k } ) } { q _ { k } ( \theta _ { k } | \theta _ { k - 1 } ) } .\tag{8}
$$

Hence, under exact reversibility, a cumulative transition kernel of the form exists, establishing the claim. □

This derivation follows the compositional construction of Geffner et al. (2023), using the perobservation reverse–forward consistency relation in Eq. (6). We spell out the intermediate algebra to clarify how the individual reverse kernels and common forward kernel enter the cumulative transition, then derive its score under a Gaussian parameterization in § A.2. Our aim is to make the cumulative transition score explicit under these assumptions, rather than to establish an exact reverse kernel for the joint posterior.

## A.2 DERIVATION OF THE REVERSE TRANSITION KERNEL

We next derive the analytic form of the cumulative transition score under a Gaussian diffusion parameterization. Starting from the cumulative reverse transition (cf. Eq. (8))

$$
\tilde { p } _ { k } ( \theta _ { k - 1 } | \theta _ { k } ) = p _ { k } ^ { 1 } ( \theta _ { k - 1 } | \theta _ { k } ) \frac { Z _ { k } } { Z _ { k - 1 } } \prod _ { j = 2 } ^ { n } \frac { p _ { k } ^ { j } ( \theta _ { k - 1 } | \theta _ { k } ) } { q _ { k } ( \theta _ { k } | \theta _ { k - 1 } ) } ,
$$

where $Z _ { k }$ and $Z _ { k - 1 }$ are normalization constants ensuring integration to one. We retain the DDPM notation, with k indexing discrete steps at diffusion times $t _ { k }$ and $\alpha _ { k }$ denoting one-step signal retention, distinct from the continuous-time cumulative factor $\alpha ( t )$ used in § A.3. The Gaussian kernels are parameterized as

$$
p _ { k } ^ { j } ( \theta _ { k - 1 } | \theta _ { k } ) = \mathcal N \big ( \theta _ { k - 1 } ; \mu _ { k } ^ { j } , ( 1 - \alpha _ { k } ) I \big ) ,\tag{9}
$$

$$
q _ { k } ( \theta _ { k } | \theta _ { k - 1 } ) = \mathcal { N } \big ( \theta _ { k } ; \sqrt { \alpha _ { k } } \theta _ { k - 1 } , ( 1 - \alpha _ { k } ) I \big ) ,\tag{10}
$$

and we differentiate log $\tilde { p } _ { k } ( \theta _ { k - 1 } | \theta _ { k } )$ with respect to $\theta _ { k - 1 }$ to get

$$
\nabla _ { \theta _ { k - 1 } } \log \tilde { p } _ { k } ( \theta _ { k - 1 } | \theta _ { k } ) = \sum _ { j = 1 } ^ { n } \nabla _ { \theta _ { k - 1 } } \log p _ { k } ^ { j } ( \theta _ { k - 1 } | \theta _ { k } ) - ( n - 1 ) \nabla _ { \theta _ { k - 1 } } \log q _ { k } ( \theta _ { k } | \theta _ { k - 1 } ) ,\tag{11}
$$

where the normalization constants cancel after differentiation. For the reverse Gaussian kernels, the mean can be written in terms of the learned score function as

$$
\mu _ { k } ^ { j } = \frac { 1 } { \sqrt { \alpha _ { k } } } \theta _ { k } + \frac { 1 - \alpha _ { k } } { \sqrt { \alpha _ { k } } } s _ { \phi } ^ { j } ( \theta _ { k } , t _ { k } ) ,\tag{12}
$$

where $s _ { \phi } ^ { j } ( \theta _ { k } , t _ { k } )$ denotes the estimated score for observation $y _ { j }$ . Before substituting the score of the log of these kernels into Eq. (11), we recall two Gaussian score identities. For a Gaussian density $\mathcal { N } ( x ; \mu , \Sigma )$ , the score with respect to the argument x is

$$
\nabla _ { x } \log \mathcal { N } ( x ; \mu , \Sigma ) = - \Sigma ^ { - 1 } ( x - \mu ) ,
$$

while the score with respect to the mean $\mu$ is

$$
\nabla _ { \mu } \log \mathcal { N } ( x ; \mu , \Sigma ) = \Sigma ^ { - 1 } ( x - \mu ) .
$$

Applying these identities gives

$$
\nabla _ { \theta _ { k - 1 } } \log { p _ { k } ^ { j } ( \theta _ { k - 1 } | \theta _ { k } ) } = \frac { \mu _ { k } ^ { j } - \theta _ { k - 1 } } { 1 - \alpha _ { k } } ,\tag{13}
$$

$$
\nabla _ { \theta _ { k - 1 } } \log q _ { k } ( \theta _ { k } | \theta _ { k - 1 } ) = { \frac { \sqrt { \alpha _ { k } } } { 1 - \alpha _ { k } } } \theta _ { k } - { \frac { \alpha _ { k } } { 1 - \alpha _ { k } } } \theta _ { k - 1 } .\tag{14}
$$

We then substitute these into Eq. (11) and group by $\theta _ { k }$ and $\theta _ { k - 1 }$ terms to form the cumulative transition kernel in terms of the mean values at the transition

$$
\begin{array} { c } { { \nabla _ { \theta _ { k - 1 } } \log \tilde { p } _ { k } ( \theta _ { k - 1 } | \theta _ { k } ) = \displaystyle \sum _ { j = 1 } ^ { n } \frac { \mu _ { k } ^ { j } - \theta _ { k - 1 } } { 1 - \alpha _ { k } } ~ - ~ ( n - 1 ) \left( \frac { \sqrt { \alpha _ { k } } } { 1 - \alpha _ { k } } \theta _ { k } - \frac { \alpha _ { k } } { 1 - \alpha _ { k } } \theta _ { k - 1 } \right) } } \\ { { { } } } \\ { { = \displaystyle \frac { \sum _ { j = 1 } ^ { n } \mu _ { k } ^ { j } - n \theta _ { k - 1 } } { 1 - \alpha _ { k } } - \frac { ( n - 1 ) \sqrt { \alpha _ { k } } } { 1 - \alpha _ { k } } \theta _ { k } + \frac { ( n - 1 ) \alpha _ { k } } { 1 - \alpha _ { k } } \theta _ { k - 1 } } } \\ { { { } } } \\ { { { } = \displaystyle \frac { \sum _ { j = 1 } ^ { n } \mu _ { k } ^ { j } - ( n - 1 ) \sqrt { \alpha _ { k } } \theta _ { k } } { 1 - \alpha _ { k } } - \frac { n - \alpha _ { k } ( n - 1 ) } { 1 - \alpha _ { k } } \theta _ { k - 1 } . } } \end{array}\tag{15}
$$

We can now identify the terms previously used by Geffner et al. (2023) for compositional sampling. Matching the grouped score to the Gaussian identity

$$
\nabla _ { \theta _ { k - 1 } } \log \mathcal { N } ( \theta _ { k - 1 } ; \mu _ { k } , \sigma _ { k } ^ { 2 } ) = \sigma _ { k } ^ { - 2 } \mu _ { k } - \sigma _ { k } ^ { - 2 } \theta _ { k - 1 } ,
$$

we identify the coefficients of $\theta _ { k - 1 }$ and the constant term as

$$
\sigma _ { k } ^ { - 2 } = \frac { n - \alpha _ { k } ( n - 1 ) } { 1 - \alpha _ { k } } , \qquad \sigma _ { k } ^ { - 2 } \mu _ { k } = \frac { 1 } { 1 - \alpha _ { k } } \sum _ { j = 1 } ^ { n } \mu _ { k } ^ { j } - \frac { ( n - 1 ) \sqrt { \alpha _ { k } } } { 1 - \alpha _ { k } } \theta _ { k } .
$$

Therefore,

$$
\mu _ { k } = \frac { \sum _ { j = 1 } ^ { n } \mu _ { k } ^ { j } - ( n - 1 ) \sqrt { \alpha _ { k } } \theta _ { k } } { n - \alpha _ { k } ( n - 1 ) } ,\tag{16}
$$

$$
\sigma _ { k } ^ { 2 } = \frac { 1 - \alpha _ { k } } { n - \alpha _ { k } ( n - 1 ) } ,\tag{17}
$$

as originally used by Geffner et al. (2023). We can now substitute approximate scores for a final cumulative score expression, $\nabla _ { \theta _ { k - 1 } } \log \tilde { p } _ { k } ( \theta _ { k - 1 } | \theta _ { k } )$ . Using the equivalence between diffusion means and scores (12),

$$
\sum _ { j = 1 } ^ { n } \mu _ { k } ^ { j } = \frac { n } { \sqrt { \alpha _ { k } } } \theta _ { k } + \frac { 1 - \alpha _ { k } } { \sqrt { \alpha _ { k } } } \sum _ { j = 1 } ^ { n } s _ { \phi } ^ { j } ( \theta _ { k } , t _ { k } ) ,
$$

Writing the right-hand side of Eq. (15) as RHS, we substitute the score parameterization to obtain

$$
\begin{array} { l } { { \displaystyle \mathrm { R H S } = \frac { 1 } { 1 - \alpha _ { k } } \sum _ { j = 1 } ^ { n } \mu _ { k } ^ { j } - \frac { ( n - 1 ) \sqrt { \alpha _ { k } } } { 1 - \alpha _ { k } } \theta _ { k } + \frac { ( n - 1 ) \alpha _ { k } - n } { 1 - \alpha _ { k } } \theta _ { k - 1 } } } \\ { { \displaystyle ~ = \frac { 1 } { 1 - \alpha _ { k } } \left( \frac { n } { \sqrt { \alpha _ { k } } } \theta _ { k } + \frac { 1 - \alpha _ { k } } { \sqrt { \alpha _ { k } } } \sum _ { j = 1 } ^ { n } s _ { \theta } ^ { j } ( \theta _ { k } , t _ { k } ) \right) - \frac { ( n - 1 ) \sqrt { \alpha _ { k } } } { 1 - \alpha _ { k } } \theta _ { k } + \frac { ( n - 1 ) \alpha _ { k } - n } { 1 - \alpha _ { k } } \theta _ { k - 1 } } } \\ { { \displaystyle ~ = \frac { 1 } { \sqrt { \alpha _ { k } } } \sum _ { j = 1 } ^ { n } s _ { \phi } ^ { j } ( \theta _ { k } , t _ { k } ) + \left[ \frac { n } { ( 1 - \alpha _ { k } ) \sqrt { \alpha _ { k } } } - \frac { ( n - 1 ) \sqrt { \alpha _ { k } } } { 1 - \alpha _ { k } } \right] \theta _ { k } + \frac { ( n - 1 ) \alpha _ { k } - n } { 1 - \alpha _ { k } } \theta _ { k - 1 } } } \\  { \displaystyle ~ = \frac { 1 } { \sqrt { \alpha _ { k } } } \sum _ { j = 1 } ^ { n } s _ { \phi } ^ { j } ( \theta _ { k } , t _ { k } ) + \frac { ( n - ( n - 1 ) \alpha _ { k } ) \theta _ { k } + \sqrt { \alpha _ { k } } \left( ( n - 1 ) \alpha _ { k } - n \right) \theta _ { k - 1 } } { \left( 1 - \alpha _ { k } \right) \sqrt { \alpha _ { k } } } . ~ ( 1 8 \mathrm { ~ } } \end{array}
$$

Finally, we arrive at the cumulative transition score. To apply this insight in practice, we need to define the $\mu _ { k }$ and $\sigma _ { k } ^ { 2 }$ terms that will be used in a Gaussian transition. Grouping the $\theta _ { k }$ and $\theta _ { k - 1 }$ terms as before and simplifying we arrive at the mean transition in terms of different scores

$$
\mu _ { k } = \frac { 1 } { \sqrt { \alpha _ { k } } } \theta _ { k } + \frac { 1 - \alpha _ { k } } { \sqrt { \alpha _ { k } } \big ( n - \alpha _ { k } ( n - 1 ) \big ) } \sum _ { j = 1 } ^ { n } s _ { \phi } ^ { j } ( \theta _ { k } , t _ { k } ) ,\tag{19}
$$

$$
\sigma _ { k } ^ { 2 } = \frac { 1 - \alpha _ { k } } { n - \alpha _ { k } ( n - 1 ) } .\tag{20}
$$

Notice that both the aggregate score contribution and the transition variance contain the common factor $n - \alpha _ { k } ( n - 1 )$ , which depends jointly on the number of composed observations and the noise schedule. We denote this quantity by

$$
\kappa _ { k } ^ { ( n ) } : = n - \alpha _ { k } ( n - 1 ) ,
$$

and refer to it as the compositional precisionfactorfor n composed observations. It controls how composition rescales the aggregate score and the variance of the normalized reverse transition.

The mean and variance of this Gaussian reverse kernel define a DDPM-style predictor,

$$
\theta _ { k - 1 } = \frac { 1 } { \sqrt { \alpha _ { k } } } \theta _ { k } + \frac { 1 - \alpha _ { k } } { \sqrt { \alpha _ { k } } \kappa _ { k } ^ { ( n ) } } \sum _ { j = 1 } ^ { n } s _ { \phi } ^ { j } ( \theta _ { k } , t _ { k } ) + \sqrt { \frac { 1 - \alpha _ { k } } { \kappa _ { k } ^ { ( n ) } } } z _ { k } , z _ { k } \sim \mathcal { N } ( 0 , I ) .\tag{21}
$$

To relate these steps to a continuous VP schedule, take a forward-time grid $0 = t _ { 0 } < \cdots < t _ { K } = 1$ and let $\alpha ( t )$ denote cumulative squared signal retention, with $\alpha ( 0 ) = 1$ . The one-step factors and their cumulative product satisfy

$$
\alpha _ { k } = \frac { \alpha ( t _ { k } ) } { \alpha ( t _ { k - 1 } ) } ,\tag{22}
$$

$$
\prod _ { j = 1 } ^ { k } \alpha _ { j } = \alpha ( t _ { k } ) .\tag{23}
$$

The score network is evaluated at the corresponding continuous time $t _ { k }$ , while $\alpha _ { k }$ controls only the transition between $t _ { k - 1 }$ and $t _ { k }$

## A.3 DERIVING THE CONTINUOUS-TIME DIFFUSION COEFFICIENT

We seek a continuous-time counterpart to the discrete predictor in Eq. (21). Here t denotes continuous diffusion time, $\alpha ( t )$ denotes cumulative squared signal retention from the clean state, and a dot denotes differentiation with respect to t. The one-step DDPM factors remain $\alpha _ { k }$ , with discrete steps

indexed by k. For a forward process with drift $f ( t ) \theta .$ , diffusion coefficient $g ( t )$ , and corresponding marginal $p _ { t } .$ , the reverse SDE admits a predictor-corrector (PC) decomposition (Song et al., 2021),

$$
\begin{array} { r } { \mathrm { d } \Theta _ { t } = \underbrace { \left( f ( t ) \theta _ { t } - \frac { 1 } { 2 } g ( t ) ^ { 2 } \nabla \log p _ { t } ( \theta _ { t } ) \right) \mathrm { d } t } _ { \mathrm { P r o b a b i l i t y ~ F l o w ~ P r e d i c t i o n ~ O D E } } + \underbrace { \left( - \frac { 1 } { 2 } g ( t ) ^ { 2 } \nabla \log p _ { t } ( \theta _ { t } ) \mathrm { d } t + g ( t ) \mathrm { d } \widetilde { w } _ { t } \right) } _ { \mathrm { L a n g e v i n ~ C o r r e c t i o n ~ S D E } } . } \end{array}\tag{24}
$$

Sampling follows this equation toward decreasing t. The probability-flow term advances samples between noise levels, while the Langevin term provides score-driven exploration at the current level. The shared coefficient $g ( t )$ determines the strength of both the score-driven motion and the injected noise. For compositional sampling, we seek an observation-count-dependent choice $g ( t ) = g _ { n } ( t )$ to improve the stability and numerical accuracy of these corrections. We examine its effects and the associated exploration trade-offs in $\ S \ A . 4$

Here we construct $g _ { n } ( t )$ by retaining the ordinary VP mean attenuation and prescribing a Gaussian noising variance that contracts with the observation count $n ,$ motivated by the discrete variance contraction in Eq. (21). Specifying this conditional kernel does not require knowing the clean posterior or assuming that it is Gaussian. With the VP drift fixed, the diffusion coefficient then follows from the chosen variance evolution.

Deriving the diffusion coefficient $g _ { n } ( t )$ . Let $x = x _ { 1 : n }$ collect the observed design–response pairs. Suppose we could draw $\theta _ { 0 } \sim p _ { 0 } ( \theta \mid x )$ from the clean posterior. Ordinary VP noising would perturb this draw according to

$$
\Theta _ { t } \mid \Theta _ { 0 } = \theta _ { 0 } \sim \mathcal { N } \Big ( \sqrt { \alpha ( t ) } \theta _ { 0 } , [ 1 - \alpha ( t ) ] I \Big ) .
$$

Here $\alpha ( t )$ is the cumulative squared signal retention, with $\alpha ( 0 ) = 1$ and $\dot { \alpha } ( t ) = - \beta ( t ) \alpha ( t )$ for the VP noise-rate schedule $\beta ( t ) \geq 0$ . This kernel has accumulated noise variance $1 - \alpha ( t )$ per coordinate, independent of the number of composed observations $n .$

The discrete compositional update in Eq. (21) contracts its one-step Gaussian reverse-transition variance by $\kappa _ { k } ^ { ( n ) } = 1 + ( n - 1 ) ( 1 - \alpha _ { k } )$ . We use the same functional form with cumulative retention $\alpha ( t )$ to prescribe the conditional noising variance, defining a distinct cumulative contraction factor $\kappa ^ { ( n ) } ( t )$

$$
\operatorname { C o v } ( \Theta _ { t } \mid \Theta _ { 0 } = \theta _ { 0 } ) = V _ { n } ( t ) I ,\tag{25}
$$

$$
V _ { n } ( t ) : = \frac { 1 - \alpha ( t ) } { \kappa ^ { ( n ) } ( t ) } ,\tag{26}
$$

$$
\kappa ^ { ( n ) } ( t ) : = 1 + ( n - 1 ) ( 1 - \alpha ( t ) ) .\tag{27}
$$

This cumulative contraction is a modeling prescription, not the infinitesimal limit of the DDPM transition. The factors $\kappa _ { k } ^ { ( n ) }$ and $\kappa ^ { ( n ) } ( t )$ share a functional form but act on one-step and accumulated noise variances, respectively. The prescribed kernel recovers ordinary VP noising at $n = 1$ and reduces the noising variance for $n > 1$

With $V _ { n } ( t )$ specified, the remaining task is to find the diffusion coefficient $g _ { n } ( t )$ that produces it while retaining the VP drift $f _ { n } ( t ) = - \beta ( t ) / 2$ . For the forward SDE $\mathrm { d } \Theta _ { t } = f _ { n } ( t ) \Theta _ { t } \mathrm { d } t + g _ { n } ( t )$ dw<sub>t</sub>, applying Itô’s formula to each coordinate’s squared deviation from its conditional mean and taking expectations gives the variance evolution

$$
\dot { V } _ { n } ( t ) = 2 f _ { n } ( t ) V _ { n } ( t ) + g _ { n } ( t ) ^ { 2 } , \qquad V _ { n } ( 0 ) = 0 .\tag{28}
$$

The first term comes from differentiating the squared deviation under the drift. With the chosen VP drift, $2 f _ { n } ( t ) V _ { n } ( t ) = - \beta ( t ) V _ { n } ( t )$ , so this contribution contracts the existing variance. The second term is Itô’s additional contribution from the noise increment’s variance $g _ { n } ( t ) ^ { \bar { 2 } } \mathrm { d } t$ , which the ordinary chain rule would miss. The remaining stochastic term has zero expectation. Consequently, the required diffusion variance rate is

$$
g _ { n } ( t ) ^ { 2 } = \dot { V } _ { n } ( t ) - 2 f _ { n } ( t ) V _ { n } ( t ) = \dot { V } _ { n } ( t ) + \beta ( t ) V _ { n } ( t ) .\tag{29}
$$

Thus, $g _ { n } ( t ) ^ { 2 }$ must both compensate for contraction under the drift and produce the prescribed increase in cumulative variance; it is not simply $V _ { n } ( t )$ or $\dot { V } _ { n } ( t )$

To evaluate this expression, we first differentiate the prescribed variance with respect to cumulative retention, then apply the chain rule to obtain its time derivative. Holding n fixed and temporarily writing α for $\alpha ( t )$ , the numerator $1 - \alpha$ has derivative 1, while the denominator $n - ( n - 1 ) \alpha$ has derivative $- ( n - 1 )$ ). The quotient rule gives

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } \alpha } \left( \frac { 1 - \alpha } { n - ( n - 1 ) \alpha } \right) = \frac { ( - 1 ) \big [ n - ( n - 1 ) \alpha \big ] - ( 1 - \alpha ) \big [ - ( n - 1 ) \big ] } { [ n - ( n - 1 ) \alpha ] ^ { 2 } } } \\ { = \frac { - n + ( n - 1 ) \alpha + ( n - 1 ) - ( n - 1 ) \alpha } { [ n - ( n - 1 ) \alpha ] ^ { 2 } } } \\ { = - \frac { 1 } { [ n - ( n - 1 ) \alpha ] ^ { 2 } } . } \end{array}
$$

The terms proportional to α cancel, leaving $- n + ( n - 1 ) = - 1$ in the numerator. Restoring the time dependence and applying the chain rule yields

$$
\begin{array} { l } { \displaystyle \dot { V } _ { n } ( t ) = \left. \frac { \mathrm { d } } { \mathrm { d } \alpha } \left( \frac { 1 - \alpha } { n - ( n - 1 ) \alpha } \right) \right. _ { \alpha = \alpha ( t ) } \dot { \alpha } ( t ) } \\ { \displaystyle = - \frac { \dot { \alpha } ( t ) } { [ \kappa ^ { ( n ) } ( t ) ] ^ { 2 } } = \frac { \beta ( t ) \alpha ( t ) } { [ \kappa ^ { ( n ) } ( t ) ] ^ { 2 } } , } \end{array}\tag{30}
$$

where the final equality uses $\dot { \alpha } ( t ) = - \beta ( t ) \alpha ( t )$

Substituting this derivative and the prescribed variance ${ \cal V } _ { n } ( t ) = ( 1 - \alpha ( t ) ) / \kappa ^ { ( n ) } ( t )$ into Eq. (29) gives

$$
\begin{array} { c } { { g _ { n } ( t ) ^ { 2 } = \displaystyle \frac { \beta ( t ) \alpha ( t ) } { [ \kappa ^ { ( n ) } ( t ) ] ^ { 2 } } + \beta ( t ) \frac { 1 - \alpha ( t ) } { \kappa ^ { ( n ) } ( t ) } } } \\ { { = \beta ( t ) \displaystyle \frac { \alpha ( t ) + ( 1 - \alpha ( t ) ) \kappa ^ { ( n ) } ( t ) } { [ \kappa ^ { ( n ) } ( t ) ] ^ { 2 } } . } } \end{array}
$$

To simplify the numerator, substitute $\kappa ^ { ( n ) } ( t ) = 1 + ( n - 1 ) ( 1 - \alpha ( t ) )$

$$
\begin{array} { l } { \alpha ( t ) + ( 1 - \alpha ( t ) ) \kappa ^ { ( n ) } ( t ) = \alpha ( t ) + ( 1 - \alpha ( t ) ) \left[ 1 + ( n - 1 ) ( 1 - \alpha ( t ) ) \right] } \\ { \qquad = \alpha ( t ) + ( 1 - \alpha ( t ) ) + ( n - 1 ) ( 1 - \alpha ( t ) ) ^ { 2 } } \\ { \qquad = 1 + ( n - 1 ) ( 1 - \alpha ( t ) ) ^ { 2 } . } \end{array}
$$

Consequently, the diffusion variance rate is

$$
g _ { n } ( t ) ^ { 2 } = - \frac { \dot { \alpha } ( t ) } { \alpha ( t ) } \frac { 1 + ( n - 1 ) ( 1 - \alpha ( t ) ) ^ { 2 } } { [ \kappa ^ { ( n ) } ( t ) ] ^ { 2 } } .
$$

Equivalently, in terms of the VP noise-rate schedule,

$$
g _ { n } ( t ) ^ { 2 } = \beta ( t ) \frac { 1 + ( n - 1 ) ( 1 - \alpha ( t ) ) ^ { 2 } } { [ 1 + ( n - 1 ) ( 1 - \alpha ( t ) ) ] ^ { 2 } } .\tag{31}
$$

For $n = 1 , \kappa ^ { ( 1 ) } ( t ) = 1$ , so the prescribed cumulative variance reduces to $V _ { 1 } ( t ) = 1 - \alpha ( t )$ and the diffusion coefficient satisfies $\bar { g _ { 1 } } ( t ) ^ { 2 } = \beta ( t )$ , recovering the ordinary VP-SDE. For general $n ,$ these coefficients realize the chosen linear Gaussian forward process with ordinary VP mean attenuation and reduced cumulative noising variance $V _ { n } ( t )$

The coefficient satisfies $g _ { n } ( t ) ^ { 2 } \leq \beta ( t )$ and depends only on diffusion time and observation count. It therefore provides a composition-dependent sampling scale without score Jacobians or auxiliary covariance estimates. This construction specifies a chosen forward noising process, rather than establishing an exact reverse process for the product bridge.

## A.4 LANGEVIN REFINEMENT AND NUMERICAL ACCURACY

Discretizing the Langevin term in Eq. (24) gives an unadjusted Langevin algorithm (ULA) correction between predictor steps. We now examine how the choice $g ( t ) = g _ { n } ( t )$ affects the stability and accuracy of these updates. We assess how efficiently these corrections explore the compositional marginals $\widetilde { p } _ { t }$ introduced in § 3.3.

Mixing at fixed diffusion time. The predictor advances between noise levels, while Langevin corrections help particles explore each intermediate target before moving on. Their effectiveness depends on both the amount of exploration and the numerical accuracy of the updates. Strongly curved or anisotropic marginals can make these corrections sensitive, e.g. rapidly contracting directions may force small steps even when exploration in other directions requires much longer trajectories – a form of numerical stiffness. Starting from the predictor output $\theta _ { t } ^ { ( 0 ) }$ and writing $\widetilde { s } _ { t } = \nabla _ { \theta }$ log $\widetilde { p } _ { t }$ , one ULA correction is

$$
\theta _ { t } ^ { ( \ell + 1 ) } = \theta _ { t } ^ { ( \ell ) } + \underbrace { \frac { \eta g ( t ) ^ { 2 } } { 2 } \widetilde { s } _ { t } ( \theta _ { t } ^ { ( \ell ) } ) } _ { \mathrm { d r i f t ~ u p d a t e } } + \underbrace { \sqrt { \eta } g ( t ) \varepsilon _ { \ell } } _ { \mathrm { d i f f u s i o n ~ n o i s e } } , \qquad \varepsilon _ { \ell } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , I _ { d } ) .\tag{32}
$$

Here L is the number of corrections, $\ell = 0 , \ldots , L - 1$ indexes them, and d is the parameter dimension. The Langevin correction step size $\eta > 0$ controls the size of each update while the diffusion noise level t remains unchanged. This update uses the coefficient $g ( t ) = g _ { n } ( t )$ from Eq. (24). Its scoredriven drift $\frac 1 2 g ( t ) ^ { 2 } \widetilde s _ { t }$ moves toward higher density, while its noise covariance $\bar { \eta g } ( t ) ^ { 2 } I _ { d }$ supplies exploration. The term $f ( t ) \theta$ belongs to the predictor and is not part of this within-level correction. “Unadjusted” means there is no Metropolis accept/reject step. Consequently, for a fixed nonzero step size, a stable ULA chain generally has an invariant distribution that differs from $\widetilde { p } _ { t }$ , introducing discretization bias (Durmus & Moulines, 2017; Roberts & Tweedie, 1996).

Numerical accuracy of Langevin correction. As a one-dimensional Gaussian illustration of this step-size restriction, consider the target $\mathcal { N } ( m , v )$ with mean m and variance v. Substituting its score $\widetilde { s } _ { t } ( \ r _ { \theta } ) = - ( \theta - m ) / v$ into Eq. (32) gives

$$
\theta _ { t } ^ { ( \ell + 1 ) } - m = \left( 1 - \frac { \eta g ( t ) ^ { 2 } } { 2 v } \right) ( \theta _ { t } ^ { ( \ell ) } - m ) + \sqrt { \eta } g ( t ) \varepsilon _ { \ell } , \qquad \varepsilon _ { \ell } \sim \mathcal { N } ( 0 , 1 ) .\tag{33}
$$

For example, if $\eta g ( t ) ^ { 2 } = 6 v$ , the deterministic part takes a displacement δ from the mean to 2δ. It crosses the mean and ends twice as far away. More generally, two trajectories receiving identical noise have their separation multiplied by $1 \dot { - } \eta g ( t ) ^ { 2 } / ( 2 v )$ , which amplifies its magnitude when $\eta g ( t ) ^ { 2 } > 4 v$ . Here the score changes at rate $- 1 / v$ with the parameter, so a narrower target produces a more rapidly varying restoring drift. Stability therefore depends on the diffusion coefficient and correction step size relative to the target variance, not on score magnitude alone.

Numerical stability does not by itselfimply accurate sampling. Even in the stable range $0 < \eta g ( t ) ^ { 2 } <$ 4v, this chain has stationary variance $\dot { v / [ 1 - \eta g ( t ) ^ { 2 } / ( 4 v ) ] }$ , which exceeds the target variance v. Reducing $g ( t ) ^ { 2 }$ or η reduces this discretization bias but also slows exploration, potentially requiring more corrections. Increasing L alone does not remove the bias.

In this illustration, using $g _ { n } ( t ) ^ { 2 } \leq \beta ( t )$ reduces the update scale relative to v and, within the stable range, the stationary-variance inflation at a given η. Because the schedule depends on observation count rather than local curvature, the step size and correction count still need to be chosen appropriately.

Choosing correction step sizes and counts. Touron et al. (2026, §3) give rules for choosing the step size and number of Langevin steps together to meet a desired Wasserstein error tolerance in annealed Langevin sampling. Their guarantees require smooth, strongly log-concave intermediate densities and controlled score error. In their bound, more steps reduce mixing error but do not remove the residual error from the finite step size and score approximation. The required number of steps also depends on how much the target changes between noise levels, so the correction count need not be the same throughout sampling.

Choosing diffusion noise levels. Choosing the noise levels themselves is a separate optimization problem. For example, Williams et al. (2024) construct score-optimal schedules by estimating the cost of moving between successive marginals and redistributing the time points to approximately equalize that cost. This requires an additional cost-estimation and grid-adaptation procedure, distinct from choosing the number of ULA corrections.

## A.5 EXPERIMENTAL EVALUATION

Gaussian benchmark. We used the correlated Gaussian model of Linhart et al. (2026), with parameter dimension $d = 1 0$ and observation-noise correlation $\rho = 0 . 8$ . The prior and conditionally independent observations are

$$
\theta \sim \mathcal { N } ( 0 , I _ { d } ) ,\tag{34}
$$

$$
x _ { j } \mid \theta \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( \theta , \Sigma ) ,\tag{35}
$$

$$
\Sigma = ( 1 - \rho ) I _ { d } + \rho \mathbf { 1 1 } ^ { \top } ,\tag{36}
$$

for $j = 1 , \dots , n ,$ , where $\textbf { 1 } \in \mathbb { R } ^ { d }$ is the all-ones vector. The exact posterior is $p ( \theta ~ \mid ~ x _ { 1 : n } ) ~ =$ $\mathcal { N } ( \theta ; m _ { 0 } , C _ { 0 } )$ , with mean m<sub>0</sub> and covariance $C _ { 0 }$ before diffusion noise is added, given by

$$
{ \cal C } _ { 0 } = ( I _ { d } + n \Sigma ^ { - 1 } ) ^ { - 1 } , \qquad m _ { 0 } = { \cal C } _ { 0 } \Sigma ^ { - 1 } \sum _ { j = 1 } ^ { n } x _ { j } .
$$

Controlled Gaussian setup. We compared the derived diffusion variance rate $g _ { n } ( t ) ^ { 2 }$ with the ordinary VP rate $g _ { \mathrm { V P } } ( t ) ^ { 2 } = \beta ( t )$ on the same prescribed Gaussian path. We examined numerical stability, covariance tracking, and posterior-sampling accuracy as n increases. Covariance tracking compares the sample covariance with its analytic target at each noise level. It checks whether the samples have the intended spread and correlations, which numerical stability alone does not guarantee. Exact scores let us isolate these effects without score-estimation error, and each paired comparison held the terminal particles, random noise draws, time grid, and correction count fixed. For Euler–Maruyama sampling, we used $\beta ( t ) = 0 . 0 5 + 1 9 . 9 5 t , \alpha ( t ) \bar { = } \exp ( - 0 . 0 5 t - 9 . 9 7 5 t ^ { 2 } )$ , and the forward process from Eq. (26). At diffusion time $t ,$ this gives a Gaussian marginal with mean $m _ { t }$ covariance $\bar { C } _ { t } .$ , and exact score $s _ { t } ^ { \mathrm { G } }$ , the gradient of its log density,

$$
m _ { t } = \sqrt { \alpha ( t ) } m _ { 0 } , \qquad C _ { t } = \alpha ( t ) C _ { 0 } + \frac { 1 - \alpha ( t ) } { \kappa ^ { ( n ) } ( t ) } I _ { d } , \qquad s _ { t } ^ { \mathrm { G } } ( \theta ) = - C _ { t } ^ { - 1 } ( \theta - m _ { t } ) .
$$

Both coefficient choices, $c ( t ) = g _ { n } ^ { 2 } ( t )$ and $c ( t ) = \beta ( t )$ , retained this score, the linear reverse drift $\beta ( t ) \theta / 2$ , and the terminal law $\mathcal { N } ( m _ { 1 } , C _ { 1 } )$ . Each run used 1,000 particles and 400 equal reversetime steps from 1 to 0, with step size $h \stackrel { \cdot } { = } 1 / 4 0 0$ and no final Tweedie correction. We replaced the diffusion coefficient in both the predictor and ULA correctors while keeping the score and initialization unchanged. This isolates the effect of using $g _ { n } ( t ) ^ { 2 }$ rather than $\beta ( t )$ for the prescribed compositional noise path.

Stiffness and step-size sensitivity. Here the predictor advances between noise levels using an Euler–Maruyama step of the full reverse SDE in Eq. (24), before any additional ULA corrections. To assess stability, imagine two nearby particles receiving the same Gaussian noise. Their noise increments cancel in the difference, isolating how the drift changes their separation. This depends on the spatial derivative of the drift, its Jacobian, rather than on the drift’s magnitude alone. Let $h > 0$ denote the sampling step size, so each predictor advances from diffusion time t to $t - h .$ . Combining the two score terms in Eq. (24) gives the sampling drift $b _ { t }$ and its Jacobian in this reverse-time direction,

$$
b _ { t } ( \theta ) = \frac { \beta ( t ) } { 2 } \theta + c ( t ) s _ { t } ^ { \mathrm { G } } ( \theta ) = \frac { \beta ( t ) } { 2 } \theta - c ( t ) C _ { t } ^ { - 1 } ( \theta - m _ { t } ) ,\tag{37}
$$

$$
J _ { t } : = \frac { \partial b _ { t } } { \partial \theta } = \frac { \beta ( t ) } { 2 } I _ { d } - c ( t ) C _ { t } ^ { - 1 } .\tag{38}
$$

At a given noise level, $m _ { t }$ and $C _ { t }$ are fixed with respect to $\theta ,$ so differentiating the Gaussian score contributes $- C _ { t } ^ { - 1 }$ . Writing δ for the separation between the two particles, one predictor step gives $\delta _ { \mathrm { n e x t } } = ( I _ { d } + \dot { h } J _ { t } ) \delta$ . To check whether discretization turns contraction into expansion, we examine this update one eigendirection at a time. Here $J _ { t }$ is symmetric, so its eigenvectors form an orthogonal basis and any separation can be decomposed into components along these directions. Along each eigenvector, applying the Jacobian is exactly multiplication by its eigenvalue. This reduces the matrix update to a scalar calculation for each component, without changing the dynamics. For a negative eigenvalue $- r .$ , its magnitude $r > 0$ is the local contraction rate, so the continuous drift brings nearby particles closer along that direction. Taking δ along this eigenvector gives $J _ { t } \delta = - r \delta$ . Euler adds h times this rate of change to the current separation, giving

![](images/a647175bcf9bbad62bbe6889d58f6e5084e623e6ae0ccf1b632a3ed5ce6b601c.jpg)

![](images/8149e3dc1bcd7e97a602b57df38279b66c7dbd00184753fef6e99cfcb17b4967.jpg)

![](images/925f904aec7f89cec740876b00b443efb5b35aca2410c4b70d6f9a21f1a65a12.jpg)

![](images/9cbfdea1e650ea438d0a2d4121379a5528cd89ff8ce035f8312a4feb2b6c0660.jpg)

![](images/8b4485e330010e4e31ef3b8bbc927041e3bca925a9b9d3c26dfc6eac2cc6181c.jpg)

![](images/a06220c43d4ac3a9a26c13271ac2e03a3f0b6627a640639f39dc6a3facac6e7a.jpg)  
Figure 7: Compositional diffusion improves finite-budget Gaussian sampling. (a–d) Diffusion schedules, covariance tracking, and matched particle paths. $\mathrm { ~ A t ~ } n = 1 0 0$ , ordinary-VP covariance errors require a logarithmic scale; path gaps and braces mark off-scale excursions. (e) Endpoint sW reduction at matched cost, with positive values favoring $g _ { n } ^ { 2 } ( t ) ;$ ; bars are pointwise 95% intervals over five paired seeds. (f) Step size relative to the local contraction limit. Ordinary VP crosses 1, indicating numerical amplification, while the derived coefficient stays below it. Accuracy improves even below this threshold.

$$
\delta _ { \mathrm { n e x t } } = \delta + h J _ { t } \delta = \delta - h r \delta = ( 1 - h r ) \delta .
$$

When $h r > 2 .$ , the multiplier $1 - h r$ is less than 1. Its negative sign means that the particles exchange their ordering along this direction, while its absolute value exceeding one means that their distance increases. For example, hr = 3 gives $\delta _ { \mathrm { n e x t } } = - 2 \delta$ , so the particles overshoot each other and end twice as far apart. This is numerical amplification in a direction where the continuous drift contracts. Checking all negative eigenvalues identifies the directions most sensitive to this overshoot.

We can summarize this sensitivity by asking how close each update comes to its local contraction limit. Along a predictor direction with contraction rate r, this limit is $2 / r$ , so the ratio $h / ( 2 / r ) = h r / 2$ reaches one at the boundary between contraction and amplification. The same reasoning extends to a ULA corrector of duration $h ,$ whose contraction limit along a direction with variance v is $4 v / c ( t )$ , as in Eq. (33). The corresponding ratio $h c ( t ) / ( 4 v )$ is largest in the narrowest direction of the Gaussian, where the covariance eigenvalue is smallest. To capture the most restrictive update encountered during sampling, we define $\bar { R ( \boldsymbol { n } ) }$ as the largest of these ratios across both update types, their contracting directions, and all sampling steps for n observations. With one correction per predictor step $( L = 1 )$

on the grid $t _ { k } = k h$ , the predictor uses time $t _ { k }$ and the subsequent corrector uses $t _ { k - 1 }$ , giving

$$
R ( n ) = \operatorname* { m a x } _ { 1 \leq k \leq 4 0 0 } \left\{ \underbrace { \frac { h } { 2 } \operatorname* { m a x } [ - \lambda _ { i } ( J _ { t _ { k } } ) ] } _ { \mathrm { p r e d i c t o r } } + \underbrace { \frac { h c ( t _ { k - 1 } ) } { 4 \lambda _ { \operatorname* { m i n } } ( C _ { t _ { k - 1 } } ) } } _ { \mathrm { c o r r e c t o r } } \right\} ,\tag{39}
$$

where $[ a ] _ { + } = \operatorname* { m a x } ( a , 0 )$ and $\lambda _ { i }$ denotes an eigenvalue. We exclude positive predictor eigenvalues because they describe expansion already present in the continuous drift, rather than amplification introduced by discretization. Panel (f) of Figure 7 reports $R ( n )$ for each coefficient choice, showing how the same sampling step size fares as more observations are composed. Values below one mean that every tested contracting direction remains within its local step-size limit, whereas values above one identify at least one update that amplifies separation where the continuous drift would contract. A smaller $R ( n )$ therefore leaves more room below these limits, although it does not by itself imply faster mixing or better posterior accuracy. It remains a local diagnostic rather than a guarantee of global stability or a measure of accumulated sampling error.

Assessing sampling accuracy. Avoiding numerical amplification does not ensure that the particles accurately represent the target distribution. For a Gaussian target, the mean and covariance together determine the distribution, so samples can be centered correctly yet misrepresent uncertainty by being too concentrated, too dispersed, or incorrectly correlated. We therefore tracked whether the particles reproduce the intended spread and correlations throughout sampling by comparing their empirical covariance with the analytic target $C _ { t }$ , using the relative covariance error $\lVert \widehat { \mathrm { C o v } } ( \theta _ { t } ) - C _ { t } \rVert _ { F } / \lVert C _ { t } \rVert _ { F } .$ where $\| \cdot \| _ { F }$ dis the Frobenius norm. The plotted paths show eight matched particles projected onto the leading posterior principal component, against the exact mean and pointwise 90% band. For panels (e,f), we fixed $L = 1$ and swept $n \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 , 4 0 , 4 8 , 6 4 , 1 0 0 \}$ . Each method used 800 full compositional-score calls per run, each processing all particles. We measured empirical sliced 2-Wasserstein distance (sW) against 1,000 exact-posterior draws using 512 random unit directions, taking the square root of the mean squared difference between sorted projections. At each $n ,$ five paired seeds shared observations, terminal particles, predictor/corrector noise, reference draws, and projection directions across methods. Panel (e) reports the mean paired difference $\mathrm { s W } _ { \beta } - \mathrm { s W } _ { g _ { n } } ,$ with pointwise 95% Student-t intervals using four degrees of freedom.

Effect of the diffusion coefficient. The two coefficients give identical samples at $n = 1$ , while the derived coefficient lowers sW in all 45 paired comparisons with $n > 1 . { \mathrm { A t } } n = 1 0 0$ , mean sW falls from 0.0153 to 0.0074, a reduction of approximately 52%. The absolute improvement peaks near $n = 8$ , rather than increasing monotonically. Ordinary VP crosses $R = 1$ between the tested values $n = 4 0$ and $n = 4 8 ;$ at $n = 1 0 0$ , R is 2.49 for ordinary VP versus 0.07 for the derived coefficient. Finite ordinary-VP endpoints with $L = 1$ can conceal transient relative covariance errors of order $1 0 ^ { 1 4 2 }$ ; these runs remain in the accuracy comparison. The coefficient therefore improves finite-budget accuracy and reduces step-size sensitivity, but the accuracy gains already appear below the local instability threshold. Changing the coefficient while retaining the score and linear drift also changes compatibility with the prescribed continuous-time marginals, so numerical stiffness alone does not explain the difference. These controlled Gaussian results do not establish the same behavior for learned-score SLCP or BMP sampling.

## A.6 EXACT-SCORE GAUSSIAN BENCHMARK

We evaluated the complete observation-count sweep $n \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 , 6 4 , 1 0 0 \}$ using the exact compositional score and posterior of the 10-dimensional correlated Gaussian model defined in $\ S \operatorname { A } . 5$ For every $n ,$ each method used five common data seeds, 1,000 posterior particles, and $T = 4 0 0$ reverse steps. The F-NPSE-SDE comparisons used $L \in \{ 1 , 5 \}$ Langevin corrections per predictor. This paired design separates changes caused by the compositional sampler from variation in observations or reference samples.

## A.7 SLCP BENCHMARK

We evaluated $n \in \{ 1 , 1 4 , 3 0 \}$ across three independently trained single-observation score networks and 25 common SLCP test instances per network seed. Each network was trained on simulatorgenerated SLCP pairs. Within each seed and test instance, all compositional samplers used the same frozen score network and reference posterior, so the comparison isolates the sampling method. Following Linhart et al. (2026), JAC and Langevin bounded standardized iterates coordinatewise to [ 3, 3] for numerical stability. We report sliced Wasserstein distance, MMD, and C2ST using the same definitions as in Table 1.

## A.8 COMPUTATIONAL COST OF COMPOSITIONAL SAMPLERS

Consider sampling $P$ posterior particles, or $P$ simultaneous reverse-diffusion chains, over d unknown parameters conditioned on n observations using $T$ reverse steps. Let $C _ { s }$ denote the architecturedependent cost of one unbatched score-network evaluation for one particle-observation pair at one diffusion time. F-NPSE-SDE evaluates n scores for every particle in each predictor and repeats these evaluations in each of L ULA corrector steps. Its compute cost is therefore $\mathcal { O } ( T ( L + 1 ) P n C _ { s } )$ Because scores can be accumulated over observations and ULA steps run sequentially, its working state requires $\mathcal { O } ( P d )$ memory beyond the score network’s minibatch activations. Thus, increasing L increases runtime but not the asymptotic peak state memory.

JAC forms a full $d \times d$ score Jacobian for every particle-observation pair at every reverse step. Computing these Jacobians requires roughly $\mathcal { O } ( \bar { T } \bar { P } n d C _ { s } )$ work, corresponding to approximately $d$ reverse-mode passes per score, in addition to dense covariance inversions and solves. A fully vectorized implementation stores $\mathcal { O } ( P n d ^ { 2 } )$ Jacobian values, an nd-fold increase over the $\mathcal { O } ( P \bar { d } )$ particle state of $\mathrm { F - N P S E - S D E }$ . For pooled BMP inference, $n = 9 4 0$ and $d = 6 5$ , so this auxiliary state is approximately $6 . 1 \times 1 0 ^ { 4 }$ times larger. Streaming over observations can instead reduce peak Jacobian storage to $\mathcal { O } ( P d ^ { 2 } )$ , but serializes the same computation and retains the additional autodifferentiation and dense linear-algebra costs.

GAUSS avoids online Jacobians but estimates a covariance for each observation from M auxiliary samples generated by S-step DDIM rollouts. For fixed conditioning, its preprocessing cost is $\mathcal { O } ( \bar { n M S } \bar { C } _ { s } )$ . A straightforward batched implementation uses $O ( n M d \bar { + } n d ^ { 2 } )$ storage for auxiliary samples and per-observation covariances. Online covariance accumulation can remove the $n M d$ term, and processing contexts sequentially can further reduce peak storage, but each context still requires the rollout computation and estimation of $\iota d \times d$ covariance. Sampling then requires $\mathcal { O } ( T \bar { P _ { n } } C _ { s } )$ score evaluations and dense precision operations.

These costs are amplified by the conditional block updates in HBDS. Each reverse-time level updates four five-dimensional receptor blocks conditional on the current 60-dimensional global state, then updates the global state conditional on the refined receptors. JAC consequently requires four sets of local Jacobians and a global $6 0 \times 6 0$ Jacobian family. If fully vectorized, the global family alone is approximately $n ( 6 0 ) ^ { 2 } \bar { / } ( 6 0 + 4 \times 5 ) \approx 4 . 2 \times 1 0 ^ { 4 }$ times larger than the complete F-NPSE-SDE HBDS particle state. Streaming reduces peak memory but repeats this Jacobian computation serially within every block update. For GAUSS, the relevant conditional covariances change with the current global and receptor states, so its DDIM rollouts become nested inside the block refinements rather than being amortized once. By contrast, F-NPSE-SDE stores only the HBDS particle state, $\mathcal { O } ( P ( 6 0 + 4 \times 5 ) )$ , beyond common score-network activations. HBDS therefore adds sequential score evaluations for F-NPSE-SDE without requiring these Jacobian or covariance calculations.

BMP runtime and memory. In our BMP experiments, JAC increased HBDS sampling from approximately one hour to more than ten hours, while GAUSS exhausted GPU memory during contextspecific covariance estimation across 940 designs. We therefore omit both from the subsequent BMP evaluation.

## A.9 COMPOSITIONAL SAMPLING RELATED WORKS

Factorized posterior sampling. Our construction builds on F-NPSE (Geffner et al., 2023), which combines single-observation posterior scores with a prior correction for inference from multiple observations. We retain this factorization and use its discrete transition to motivate a continuous-time Gaussian surrogate. The contribution is the resulting variance prescription and compatible diffusion coefficient, rather than score composition itself; Arruda et al. (2025) review this broader family of SBI methods.

Inference-time schedule adjustments. Our scalar approximation is closely related to the samplingtime modifications of Arruda et al. (2026), who stabilize composition through score damping and an adjustable constant shift of the log signal-to-noise ratio (SNR), alongside minibatch score estimation to reduce memory requirements. For our Gaussian surrogate, define $\mathrm { S N R } _ { n } ( t ) = \alpha ( t ) / V _ { n } ( t )$ and $\mathrm { S N R } _ { \mathrm { V P } } ( t ) = \alpha ( \dot { t } ) / ( \bar { 1 } - \alpha ( t ) )$ at times with $0 < \alpha ( t ) < 1$ . The prescribed variance in Eq. (26) gives

$$
\begin{array} { r } { \mathrm { l o g } \mathrm { S N R } _ { n } ( t ) = \log \mathrm { S N R } _ { \mathrm { V P } } ( t ) + \log \kappa ^ { ( n ) } ( t ) . } \end{array}
$$

Thus, $\kappa ^ { ( n ) } ( t )$ induces a time- and observation-count-dependent log-SNR shift. We obtain this adjustment by prescribing a composition-dependent cumulative variance and deriving its compatible diffusion coefficient while retaining the ordinary VP mean attenuation. Since the mean attenuation is unchanged while the variance is reduced, this is not simply a reparameterization of the ordinary VP schedule, nor is it equivalent to multiplying the composed score by a damping factor. The derivation realizes the chosen Gaussian surrogate but does not establish an exact reverse process for the full compositional posterior.

Covariance-aware sampling. The samplers of Linhart et al. (2026) address errors in composing diffusion-time scores through score Jacobians (JAC) or auxiliary posterior covariance estimates (GAUSS). We compare these methods directly on Gaussian and SLCP benchmarks in § 4.1; their additional computations become more expensive when repeated inside HBDS block updates. In our BMP implementation, JAC increased HBDS sampling from approximately one hour to more than ten hours, while GAUSS exhausted GPU memory during covariance estimation across 940 designs (§ A.8). These observations concern the tested implementation, not an impossibility of adapting either method. Our score-only compositional sampler makes the repeated block updates practical in this setting. The hierarchical inference construction is discussed separately in § B.2.

## B HIERARCHICAL BLOCKWISE DIFFUSION SAMPLING DETAILS

Grouped posterior target. Suppose there are K experimental groups, each with design $\xi ^ { k }$ , observation $y ^ { k }$ , and local latent state ${ \dot { R } } ^ { \dot { k } }$ , together with global parameters θ. Assuming condition-specific independent receptor priors, as in our BMP model, the grouped posterior factorizes as

$$
p ( \theta , R ^ { 1 : K } \mid y ^ { 1 : K } , \xi ^ { 1 : K } ) \propto p ( \theta ) \prod _ { k = 1 } ^ { K } p ( R ^ { k } \mid \xi ^ { k } ) p ( y ^ { k } \mid R ^ { k } , \theta , \xi ^ { k } ) .\tag{40}
$$

A pooled model restricts this family by tying $R ^ { 1 } = \cdots = R ^ { K }$ . If two groups require different local states on a set with positive posterior mass, that restriction cannot represent the grouped posterior; § B.1 gives an analytic example.

Blockwise intermediate target. HBDS uses conditional scores to approximate the unavailable noised joint posterior $p _ { t } ( \theta , R ^ { \smash { \sum } _ { j } } \mid y ^ { 1 : K } , \xi ^ { 1 : K } )$ . Over each diffusion interval, it advances and refines the local receptor blocks before updating the shared parameters using all groups. Additional block rounds repeat this interval after re-noising the updated states, rather than adding another reverse transition after the predictor–corrector updates.

Algorithm 1 uses $\mathrm { P C } _ { t  s } ^ { B }$ to denote a predictor–corrector update that advances block B from t to s and applies L Langevin corrections at s. In the continuous implementation, the predictor uses the probability-flow part of Eq. (24) and is deterministic. Each correction follows Eq. (32) with noise covariance $\eta g _ { n _ { B } } ( s ) ^ { 2 } I$ , where $\eta = ( t - s ) / L$ for $L > 0$ . Here $n _ { B }$ is the observation count for that block, namely the size of group k for $R ^ { k }$ and the total count for θ. Setting $L = 0$ skips corrections, which are also omitted on the final diffusion interval. Thus, no additional Gaussian reverse step is needed after these updates. The initialization laws $q _ { \mathrm { i n i t } }$ use Gaussian noise for θ and, in the BMP experiments, the receptor prior in § D.

Algorithm 1: Hierarchical Blockwise Diffusion Sampling (HBDS)   
Input: Groups $\begin{array} { r } { \{ ( \xi ^ { k } , y ^ { k } ) \} _ { k = 1 } ^ { K } ; } \end{array}$ ; score model $s _ { \phi } ;$ sampling schedule $( f , g _ { n } ) ;$ time grid   
$t _ { T } > \cdots > t _ { 0 } = 0 ;$ block rounds $M \geq 1 ;$ corrector steps $L \geq 0 .$   
$\mathrm { P C } _ { t  s }$ combines one predictor step $t \to s$ with L Langevin corrections at $s .$   
Init: $\theta _ { t _ { T } } \sim q _ { \mathrm { i n i t } } ^ { \theta } ; R _ { t _ { T } } ^ { k } \sim q _ { \mathrm { i n i t } } ^ { R ^ { k } }$ for $k = 1 , \ldots , K$   
for $i = T , T - 1 , \ldots ,$ 1 do   
$( t , s ) \gets ( t _ { i } , t _ { i - 1 } )$   
for $m = 1 , \ldots , M$ do   
for $k = 1$ to K do   
L $R _ { s } ^ { k } \gets \mathrm { P C } _ { t \to s } ^ { R } \big ( R _ { t } ^ { k } \ | \ \theta _ { t } , \xi ^ { k } , y ^ { k } \big )$   
$\mathcal { G } _ { s } \gets \{ ( R _ { s } ^ { k } , \xi ^ { k } , y ^ { k } ) \} _ { k = 1 } ^ { K }$   
$\theta _ { s } \gets \mathrm { P C } _ { t  s } ^ { \theta } ( \theta _ { t } \mid \mathcal { G } _ { s } )$   
if m $< M$ then   
Re-noise $( \theta _ { s } , R _ { s } ^ { 1 : K } )$ to obtain $( \theta _ { t } , R _ { t } ^ { 1 : K } )$ at noise level t   
Output: $( \theta _ { 0 } , \{ R _ { 0 } ^ { k } \} _ { k = 1 } ^ { K } )$ conditioned on $\{ ( \xi ^ { k } , y ^ { k } ) \} _ { k = 1 } ^ { K } .$

## B.1 TOY SIMULATOR MODEL EXAMPLE

We consider a hierarchical model with a discrete latent $k \in \{ - 1 , + 1 \}$ , a continuous latent $\theta \in \mathbb { R } .$ and an observation $x \in \mathbb { R }$ . The generative process is

$$
\begin{array} { r l r } & { k \sim \mathrm { R a d } \left( \frac { 1 } { 2 } \right) , } & { { \mathrm { ( R a d e m a c h e r : ~ } } \mathbb { P } [ k = + 1 ] = \mathbb { P } [ k = - 1 ] = \frac { 1 } { 2 } ) , } \\ & { \theta \sim { \mathcal N } ( 0 , 1 ) , } & \\ & { x | \theta , k \sim { \mathcal N } ( 5 k + \theta , 1 ) , } & { \mathrm { i . e . , ~ } x = 5 k + \theta + \varepsilon , \ \varepsilon \sim { \mathcal N } ( 0 , 1 ) \ \mathrm { i n d e p } . } \end{array}
$$

The joint density factorizes as

$$
\begin{array} { r } { p ( \theta , k , x ) = p ( k ) p ( \theta ) p ( x | \theta , k ) = \frac { 1 } { 2 } \mathcal { N } ( \theta ; 0 , 1 ) \mathcal { N } \big ( x ; 5 k + \theta , 1 \big ) . } \end{array}
$$

Thus, the likelihood is $p ( x | \theta , k ) = \mathcal { N } ( x ; 5 k + \theta , 1 )$ , with prior $p ( \theta ) = \mathcal { N } ( 0 , 1 )$ and $\begin{array} { r } { p ( k ) = \frac { 1 } { 2 } } \end{array}$ . The corresponding posterior distributions are analytically tractable. The discrete latent variable has

$$
p ( k = + 1 | x ) = \frac { \mathcal { N } ( x ; 5 , 2 ) } { \mathcal { N } ( x ; 5 , 2 ) + \mathcal { N } ( x ; - 5 , 2 ) } ,\tag{41}
$$

$$
\begin{array} { r } { p ( k = - 1 | x ) = 1 - p ( k = + 1 | x ) , } \end{array}\tag{42}
$$

while the continuous latent variable conditioned on x and k follows

$$
\begin{array} { r } { p ( \theta | x , k ) = \mathcal { N } \big ( \theta ; \frac { 1 } { 2 } ( x - 5 k ) , \frac { 1 } { 2 } \big ) . } \end{array}\tag{43}
$$

Together these define the exact posterior for a single observation,

$$
\begin{array} { r } { p ( \theta , k | x ) = p ( \theta | x , k ) p ( k | x ) , } \end{array}
$$

which serves as the analytic ground truth for single-sample posterior evaluation.

Pooled versus grouped posterior structure. For K experimental groups, we introduce one groupspecific discrete latent $z _ { g } \in \{ - 1 , + 1 \}$ per group while sharing the continuous parameter θ:

$$
\begin{array} { r } { z _ { g } \sim \operatorname { R a d } \left( \frac { 1 } { 2 } \right) , \qquad \theta \sim \mathcal { N } ( 0 , 1 ) , \qquad x _ { g } \mid \theta , z _ { g } \sim \mathcal { N } ( 5 z _ { g } + \theta , 1 ) . } \end{array}
$$

The hierarchical posterior is

$$
p ( \theta , z _ { 1 : K } \mid x _ { 1 : K } ) \propto p ( \theta ) \prod _ { g = 1 } ^ { K } p ( z _ { g } ) p ( x _ { g } \mid \theta , z _ { g } ) ,\tag{44}
$$

![](images/39a532b70e5e96b3480c34be57f34b00dd84fc2d677b6e58702e4293b0937e6b.jpg)  
Figure 8: Joint-state diagnostic for the hierarchical toy with 20 observations across two groups and no ULA correctors. (Left) Pooled sampling with one shared latent. (Right) HBDS with separate group latents. Rows use analytic (top) and learned (bottom) scores; dotted lines mark the generating $\theta = 1$ This oracle diagnostic conditions HBDS latent updates on the true θ and parameter updates on the true group signs; pooled sampling receives no such information.

which allows each group to select its own latent mode. A pooled model instead ties all groups to a single latent z,

$$
p _ { \mathrm { p o o l } } ( \theta , z \mid x _ { 1 : K } ) \propto p ( \theta ) p ( z ) \prod _ { g = 1 } ^ { K } p ( x _ { g } \mid \theta , z ) .\tag{45}
$$

Thus, whenever different groups are better explained by different latent modes, the hierarchical posterior can represent the oracle structure while the pooled posterior cannot. For example, observations near +5 and 5 are naturally explained by different group-specific latents with $\theta \approx 0$ , whereas the pooled model must force both groups to share the same latent explanation.

Learned-score illustration. The main-text comparison in Figure 4 used 20 observations, ten per group, generated with $\theta = 1$ , signs $z _ { 1 } = - 1$ and $z _ { 2 } = + 1$ , and unit observation-noise standard deviation. Both samplers reused a frozen single-observation score network, with 1,000 particles per sampling seed across seeds 0, 1, and 2, 200 predictor intervals, and no ULA correctors. HBDS conditioned its block updates on cumulative-VP Tweedie estimates, without access to the generating parameters or signs. The plot shows all saved paired particles without jitter or latent rounding; continuous latent states are sampler approximations to the discrete target.

## B.2 HIERARCHICAL SBI RELATED WORKS

Hierarchical posterior estimators. Hierarchical SBI explicitly distinguishes shared parameters from local latent variables. Heinrich et al. (2024) learn dataset-wide representations of event ensembles and separate global and local posterior estimators, supporting varying ensemble sizes. Habermann et al. (2025) introduce multilevel neural posterior estimation (ML-NPE), pairing hierarchical summary networks with global and local posterior estimators. Their framework supports covariates, shared parameters, and varying numbers of groups and observations per group within the training range, with validation through simulation-based calibration and comparisons to Stan. Charles et al. (2026) combine group-aware tokenization with flow matching, using a per-site likelihood surrogate to generate synthetic multi-site datasets for training a separate hierarchical posterior estimator. These approaches build the hierarchy into the learned inference model.

Hierarchical score models. Arruda et al. (2026) train global and local score estimators on individual groups, compose the global scores across groups, and then sample local parameters conditional on the sampled globals. This supports hierarchical inference without simulating the full hierarchy for each training example. HBDS instead reuses the conditional queries of one joint score model and alternates local and global updates within the reverse-diffusion process, rather than training separate estimators for the hierarchical posterior factorization. The distinction is conditional-model reuse and blockwise sampling, not hierarchical composition itself. Their compositional sampling modifications are discussed in $\ S \bar { \bf A } . 9$

Nested, design-dependent observations. In BMP, each of four cell lines has one receptor vector shared across 235 distinct ligand-dose observations, while biophysical parameters are shared across all cell lines (§ C). Assigning an independent local latent to every scalar observation would discard this within-cell-line sharing. Conversely, treating a cell line as one group requires representing its panel of design–response pairs within the group-level inference model. ML-NPE already accommodates nested observations and changing dataset sizes (Habermann et al., 2025); HBDS instead performs the aggregation through score composition at sampling time. It retains the single-observation representation and composes scores within receptor blocks and across groups for the shared parameters. Observation subsets can therefore change without training another hierarchical estimator, provided the underlying model has learned the required conditionals. The same sampler can also reuse the joint model adapted through likelihood fine-tuning (§ 3.1).

## C BMP SIGNALING PATHWAY MATHEMATICAL MODEL

The BMP dataset contains $n = 9 4 0$ steady-state cell response measurements from $C = 4$ cell lines, each profiled under 235 distinct ligand-dose conditions. We write $y = ( y _ { 1 } , \dotsc , y _ { n } )$ for these observations. The parent and three receptor-knockdown lines form four groups, each with a distinct receptor state $R ^ { k } \in \mathbb { R } ^ { 5 }$ shared across its ligand-dose experiments. These groups share the simulator’s biochemical affinities and catalytic efficiencies, collected in $\theta \in \mathbb { R } ^ { 6 0 }$ , giving the hierarchical structure used by HBDS in § 3.4. Each single-observation latent state $( \theta , R ^ { k } )$ therefore contains 65 quantities.

BMPs are homodimeric and heterodimeric ligands that transmit information by assembling signaling complexes with two classes of serine/threonine-kinase receptors: type I $( A _ { i } )$ and type II $( B _ { k } )$ . In a parsimonious “one-step” mass–action kinetics model (Su et al., 2022), each ligand $L _ { j }$ binds one receptor of each class in a single equilibrium step

$$
A _ { i } + B _ { k } + L _ { j } \ { \stackrel { K _ { i j k } } { \longleftrightarrow } } \ T _ { i j k } , \varepsilon _ { i j k } T _ { i j k } \ \longrightarrow \ S ,
$$

where $K _ { i j k }$ is an effective affinity for forming the active trimer $T _ { i j k }$ and $\varepsilon _ { i j k }$ is its catalytic efficiency in phosphorylating SMAD1/5/8, producing a transcriptional signal S. Interestingly, despite treating the 10 known ligands and $4 \times 3 = 1 2$ canonical receptors as effectively promiscuous, the model recovers dose–response data and explains how receptor context rewires downstream transcriptional programs during development or oncogenesis (Antebi et al., 2017; Su et al., 2022). Here, we model a subset of the known ligands and canonical receptors where $I = 3 , J = 5$ , and $K = 2$ denote the numbers of type-I receptors, ligands, and type-II receptors, respectively, so that $i = 1 , \ldots , I ,$ $j = 1 , \dots , J ,$ and ${ \bf \bar { \Psi } } k = 1 , \ldots , { \bf K }$ . Table 2 summarizes how the SBI variables map onto this BMP model. The receptor concentrations A and B can also be treated as fixed design variables, as in perturbation-modeling settings where biological covariates are specified interventions (Dimitrov et al., 2026), giving $( L _ { j } , A _ { i } , B _ { k } ) \in \Xi ^ { J + I + K }$ and $( \varepsilon _ { i j k } , K _ { i j k } ) \in \Theta _ { > 0 } ^ { 2 I ^ { \prime } J K }$ , but since receptor measurements are noisy, we treat them as latent variables throughout the experiments.

Table 2: BMP-to-SBI mapping and dimensions.
<table><tr><td>SBI symbol</td><td>BMP meaning</td><td>Dimension</td></tr><tr><td>ξ</td><td>Experimental design: ligand doses  $L _ { \mathrm { 1 : J } }$ </td><td> $J = 5$ </td></tr><tr><td>θ</td><td>Shared affinities  $K _ { i j k }$  and efficiencies  $\varepsilon _ { i j k }$ </td><td> $2 I J K = 6 0$ </td></tr><tr><td> $R ^ { k }$ </td><td>Cell-line receptor state  $A _ { 1 : I } , B _ { 1 : K }$ </td><td> $I + K = 5$ </td></tr><tr><td>y</td><td>Observation: steady-state SMAD response S</td><td>1</td></tr></table>

## D CELL RECEPTOR HIERARCHICAL MODELING

We implement diffusion guidance by constructing a probabilistic prior over receptor expression informed by quantitative polymerase chain reaction (qPCR) measurements and then using the resulting (marginal) log-density as an energy function for guidance during sampling. The goal is to bias samples toward receptor configurations consistent with known KD conditions while explicitly accounting for the substantial uncertainty inherent to qPCR.

Indices and observations. Let $s \in \{ 0 , 1 , 2 , 3 \}$ index the four experimental cell-line conditions (one wild-type and three KD conditions). Let $j \in \{ 0 , 1 , 2 , 3 , 4 \}$ index the five BMP-pathway receptors considered in this work (ACVR1, BMPR1A, ACVR2A, ACVR2B, BMPR2). For each condition s and receptor $j$ we observe a positive relative qPCR readout $x _ { s , j } > 0$ , interpreted as a noisy proxy for receptor abundance (relative mRNA expression). Because qPCR variability is multiplicative, we model observations in log space, $y _ { s , j } \triangleq \log x _ { s , j }$

Knockdown targets. For each condition $s ,$ the KD target index is $k _ { s } \in \{ - 1 , 0 , 1 , 2 , 3 , 4 \}$ , where $k _ { s } = - 1$ denotes no knockdown (wild-type) and $k _ { s } = j$ denotes that receptor j is knocked down in condition s. We define an indicator

$$
\mathbb { I } _ { s , j } \triangleq \mathbb { I } [ k _ { s } = j ] ,
$$

which is 1 iff receptor $j$ is the KD target in condition s and 0 otherwise.

Hierarchical model (baseline receptor distribution). We posit receptor-specific baseline logmeans and log-standard-deviations that are partially pooled across receptors via global hyperparameters. Let $\mu _ { j }$ denote the baseline log-mean for receptor $j ,$ and let $\sigma _ { j }$ denote the baseline log-standard-deviation for receptor $j .$ . We introduce global hyperparameters $( \mu _ { 0 } , \kappa )$ controlling the population distribution of receptor means, and $( \ell _ { 0 } , \eta )$ controlling the population distribution of receptor log-variances. In standard hierarchical form,

$$
\begin{array} { c } { { \mu _ { 0 } \sim { \mathcal N } ( m _ { 0 } , v _ { 0 } ) , } } \\ { { \kappa \sim \mathrm { H a l f N o r m a l } ( \lambda _ { \kappa } ) , } } \\ { { \mu _ { j } \mid \mu _ { 0 } , \kappa \sim { \mathcal N } ( \mu _ { 0 } , \kappa ^ { 2 } ) , ~ j = 0 , \ldots , 4 , } } \\ { { \ell _ { 0 } \sim { \mathcal N } ( m _ { \ell } , v _ { \ell } ) , } } \\ { { \eta \sim \mathrm { H a l f N o r m a l } ( \lambda _ { \eta } ) , } } \\ { { \log \sigma _ { j } \mid \ell _ { 0 } , \eta \sim { \mathcal N } ( \ell _ { 0 } , \eta ^ { 2 } ) , ~ j = 0 , \ldots , 4 . } } \end{array}
$$

This hierarchy induces partial pooling: when data are weak or noisy, $\{ \mu _ { j } , \sigma _ { j } \}$ are shrunk toward the global centers $( \mu _ { 0 } , \ell _ { 0 } )$ , while still allowing receptor-specific deviations.

Condition-level variability. To account for inter-condition variability (e.g., biological differences between cell lines and experimental perturbations not explained by the KD), we add a conditionspecific random effect $\delta _ { s , j }$ for each receptor:

$$
\begin{array} { r } { \tau \sim \mathrm { H a l f N o r m a l } ( \lambda _ { \tau } ) , } \\ { \delta _ { s , j } \mid \tau \sim \mathcal { N } ( 0 , \tau ^ { 2 } ) , \qquad s = 0 , \ldots , 3 , j = 0 , \ldots , 4 . } \end{array}
$$

This term captures the fact that receptor expression can vary appreciably across conditions even in the absence of knockdown.

Knockdown effect model (multiplicative fold-change). Knockdown acts multiplicatively in the original receptor domain, which corresponds to an additive shift in log space. We therefore model a KD fold-change via a log-shift $\nu _ { j } \leq 0$ and additional KD uncertainty $\omega _ { j } \geq 0 \mathrm { : }$

$$
\begin{array} { r l } & { \nu _ { j } \sim \mathcal { N } ( m _ { \nu , j } , v _ { \nu , j } ) \mathrm { w i t h s u p p o r t o n } ( - \infty , 0 ] , } \\ & { \omega _ { j } \sim \mathrm { H a l f N o r m a l } ( \lambda _ { \omega , j } ) , } \end{array}
$$

and incorporate KD by modifying the conditional distribution of $y _ { s , j }$ . Specifically, the KD target receptor $( \mathbb { I } _ { s , j } = 1 )$ receives a negative mean shift and an additional variance term:

$$
y _ { s , j } \mid \mu _ { j } , \sigma _ { j } , \delta _ { s , j } , \nu _ { j } , \omega _ { j } \sim { \mathcal { N } } \Big ( \mu _ { j } + \delta _ { s , j } + \mathbb { I } _ { s , j } \nu _ { j } , \sigma _ { j } ^ { 2 } + \mathbb { I } _ { s , j } \omega _ { j } ^ { 2 } \Big ) .
$$

Non-target receptors $( \mathbb { I } _ { s , j } = 0 )$ follow the baseline distribution.

![](images/da3d5ecf2f37822f666a71241b87bef99afd37b86664868d299d2b2a839e8be8.jpg)  
Figure 9: Comparison between BMP receptor posterior samples (columns) across LSR best-fit $\theta ^ { \mathrm { L S R } }$ (top), applying a truncated lognormal prior to the BMP receptors’ distributions (middle), and applying the marginal hierarchical receptor priors § D (bottom). Model identifiability in the $\theta ^ { \mathrm { L S \acute { R } } }$ parameter set can be seen in the multi-modal distributions of the $\mathcal { \theta } ^ { \mathrm { L S R } }$ fits as well as the case of BMPR2, where $\theta ^ { \mathrm { L S R } }$ only works if the ACVR1 KD and BMPR1A KD conditions are 1.5 the amount of the parent cell line. Imposing a diffusion prior on the receptors in the BMP simformer avoids this clash of parameters.

Observation model in the original receptor domain. Let $r _ { s , j } \triangleq \exp ( y _ { s , j } )$ denote the receptor expression in the original (positive) domain. In our implementation we further impose a physically plausible upper bound $h > 0$ and use a truncated LogNormal likelihood with support $r _ { s , j } \in ( 0 , h ) ;$

$$
r _ { s , j } \mid \mu _ { j } , \sigma _ { j } , \delta _ { s , j } , \nu _ { j } , \omega _ { j } \sim \mathrm { T r u n c L o g N o r m a l } \Big ( \mu _ { j } + \delta _ { s , j } + \mathbb { I } _ { s , j } \nu _ { j } , \sqrt { \sigma _ { j } ^ { 2 } + \mathbb { I } _ { s , j } \omega _ { j } ^ { 2 } } ; ~ h \Big ) .
$$

Equivalently, this is a LogNormal distribution in $\boldsymbol { r } _ { s , j }$ with truncation at $h ,$ which provides numerical stability and reflects the bounded receptor regime used in simulation.

Using the model for guidance. The hierarchical model defines a receptor prior density $p ( r \mid s )$ for each condition s. During diffusion sampling, we compute a denoised receptor estimate and apply guidance by adding the gradient of the (marginal) log-prior,

$$
g ( r , s ) \ \triangleq \ \nabla _ { r } \log p ( r \mid s ) ,
$$

scaled by a time-dependent factor (analogous to classifier-free guidance when used as a prior). In practice, marginalization over latent variables $( \mathbf { e } . \mathbf { g } . , \mu _ { 0 } , \kappa , \ell _ { 0 } , \eta , \tau )$ can be performed either by a tractable approximation (e.g., moment-matching) or by mixture density networks (Bishop, 1994); both yield a differentiable log-density suitable for guidance.

## E ADDITIONAL BMP EXPERIMENTAL RESULTS

## E.1 CALIBRATED RECEPTOR DIFFUSION

The LSR receptor estimates in Figure 9 illustrate potential identifiability issues. Some knockdown (KD) conditions have higher fitted basal receptor levels than the parent cell line, contrary to the intended knockdown. Low predictive error alone therefore does not establish physical plausibility. The receptor priors in § D allow posterior sampling to incorporate these biological constraints.

LSR fit and receptor sensitivity. We compared FT posterior-predictive fit with the historical least-squares regression estimate $\mathbf { \bar { \theta } } ^ { \mathrm { L S R } }$ and a sensitivity analysis in which its ACVR1 estimate was reduced by 10% (Table 3). The FT model in this comparison has three transformer layers and a feed-forward widening factor of two, with $\lambda = 5 \times \bar { 1 0 } ^ { - 4 }$ . Its all-data row used HBDS on 940 observations, whereas each cell-line row used an independent pooled posterior conditioned on that cell’s 235 observations. These rows are not subsets of a single all-data posterior. Each run used 250 draws, 100 diffusion steps, five Langevin corrections per step, and sampling seed 42. The all-data FT RMSE is 0.26. The historical LSR errors were retained without recomputation, and their all-data value is not the root-mean-square of the four cell-wise values, so they should not be treated as a consistently aggregated baseline until the original evaluation protocol is reconciled. The receptor perturbation illustrates local fit sensitivity.

Table 3: RMSE comparison of historical LSR, ACVR1 −10% LSR $( \theta ^ { \mathrm { L S R \dagger } } )$ , and FT predictions. FT uses HBDS for all data and independent pooled inference for each cell line. Lower values are better.
<table><tr><td>Condition</td><td> $\theta ^ { \mathrm { L S R } }$ </td><td> $\theta ^ { \mathrm { L S R \dagger } }$ </td><td>FT</td></tr><tr><td>All data</td><td>0.08</td><td>1.02</td><td>0.26</td></tr><tr><td>NMuMG</td><td>0.08</td><td>0.08</td><td>0.48</td></tr><tr><td>BMPR1A KD</td><td>0.07</td><td>0.06</td><td>0.31</td></tr><tr><td>BMPR2 KD</td><td>0.09</td><td>0.06</td><td>0.13</td></tr><tr><td>ACVR1 KD</td><td>0.06</td><td>2.04</td><td>0.22</td></tr></table>

Table 4: Pooled/HBDS comparison for PT, FT, and separately trained FT without FiLM (¬FiLM), using 250 draws and one sampling seed.
<table><tr><td colspan="4"></td></tr><tr><td>Method</td><td>RMSE↓ Median↓</td><td>θ</td><td>R</td></tr><tr><td>PT Pooled</td><td>0.58</td><td>1,594.55</td><td>0.22</td></tr><tr><td>FT Pooled</td><td>0.36</td><td>66.77 0.18</td><td>0.06 0.02</td></tr><tr><td>¬FiLM Pooled</td><td>12.30</td><td>4,538.82</td><td>0.13 0.03</td></tr><tr><td>PT HBDS</td><td>0.78</td><td>194.01</td><td>0.04</td></tr><tr><td>FT HBDS</td><td>0.34</td><td>14.94</td><td>0.05 0.02 0.07</td></tr><tr><td>¬FiLM HBDS</td><td>0.91</td><td>274.61</td><td>0.09 0.01</td></tr></table>

## E.2 BMP POSTERIOR-SAMPLING ABLATIONS

We compared pooled sampling and HBDS on 940 BMP observations, with 235 observations from each of four cell lines. Pooled sampling infers the 60 shared biophysical parameters and one shared five-dimensional receptor vector, whereas HBDS maintains a separate receptor state for each cell line. Both samplers used the derived diffusion coefficient, 100 diffusion steps, five Langevin corrections per step, 250 posterior draws, and sampling seed 42. HBDS used three block-refinement rounds with Tweedie conditioning for both blocks. Pooled sampling used three resampling passes in one sequential round, which are not hierarchical block updates. These are single-seed results, with no between-seed uncertainty estimates.

Models and predictive metrics. The PT and FT models in Tables 4 and 5 have four transformer layers and a feed-forward widening factor of four. FT started from the matched PT model and used $\lambda { \stackrel { \cdot } { = } } 5 \times 1 0 ^ { - 4 }$ for the sampler comparison. The no-FiLM model is a separately pretrained and fine-tuned architectural ablation that retains mask-state embeddings, not the same model with FiLM switched off at sampling time. It uses the same depth and width as the FiLM model. For posterior draw k, let $d _ { k } = \| \mathbf { y } _ { k } - \mathbf { y } _ { o } \| _ { 2 }$ , where $\mathbf { y } _ { k }$ is evaluated through the original mechanistic simulator over N observations. We report $\mathrm { R M S E } = \mathrm { m i n } _ { k } d _ { k } / \sqrt { N }$ and median distance ${ \mathrm { m e d i a n } } _ { k } d _ { k }$ . Thus, RMSE measures the best predictive draw rather than the posterior mean, while median distance summarizes response-space error across draws.

Calibration. We report the logged local classifier two-sample test (ℓ-C2ST) (Linhart et al., 2023) statistics separately for parameters and receptors. The tests used an MLP with three folds and two ensemble members, classifier seed zero, and 100 permutation-based null trials. These are single-run statistics, not averages across independent sampling seeds.

Fine-tuning improves both predictive-fit metrics over its matched PT model under pooled sampling and HBDS (Table 4). For each FT architecture, HBDS gives lower errors than pooled sampling, and FT with FiLM has the lowest reported values among the completed comparisons. The three-layer results in Table 3 use a different model family and are not a controlled comparison of network size.

## F PRETRAINING DETAILS

Transformer architecture. The simformer uses value, ID, condition, and attention dimensions of 32, with three attention heads of size 32. The model used for the cell-line and LSR comparison in Table 3 has three layers and a feed-forward widening factor of two. The model/sampler and regularization ablations in Tables 4 and 5 instead used four layers and a widening factor of four. Rather than summing token embeddings (ID, condition, and value), we concatenate them prior to projection, which empirically produced more stable training and better downstream posterior quality.

We trained the simformer using 1,000 prior parameter draws evaluated across 940 experimental designs. For each draw $\theta \sim p ( \theta )$ and design ξ, we generated expert-model outputs $y \sim p _ { \mathrm { e v a l } } ( y \mid \theta , \xi )$ to form training examples $( \theta , \xi , y )$ . While preliminary results suggest that increasing the number of prior parameter draws improves downstream posterior quality, we restricted this study to 1,000 draws for computational simplicity.

During training we employed two forms of masking. For edge masking, we applied masks corresponding to marginals, the full joint, and the expert model graph, each with equal probability $\left( { \frac { 1 } { 3 } } \right.$ of the time). For condition masking, we used four different masks: (i) joint masking (noise all variables), (ii) posterior conditioning, (iii) likelihood conditioning, and (iv) random conditioning on 30% of tokens in arbitrary groupings. These were applied with probabilities of 20% for the joint, posterior, and likelihood masks, and 40% for the random conditioning. We found that heavier use of posterior and likelihood masks tended to produce overconfident models, whereas increasing the joint masking improved generalization.

Pretraining pipeline. We trained for 100 epochs with a batch size of 2,048. Evaluating these parameter draws across all 940 designs produced approximately $1 , 0 0 0 \times 9 4 0 = 9 4 0 , 0 0 0$ simulator outputs. While this represents a large training set, the dimensionality of the parameter space $( \theta \in \mathbb { R } ^ { 6 0 } )$ implies a trade-off between the number of prior parameter draws available and the coverage of experimental conditions. In practice, this dataset size was sufficient for demonstrating the model’s capabilities within available compute, but there is room for future work studying scaling laws of the number of prior parameter draws and the number of parameters of the architecture.

Compute resources. All models were trained on NVIDIA A100 GPUs. Sampling and smaller ablation runs were also performed on NVIDIA A30 GPUs when memory constraints allowed.

## G FINE-TUNING AND PATH-SPACE DETAILS

## G.1 FINE-TUNING OBJECTIVE AND OPTIMIZATION

MSE reward. We use the negative-MSE reward $R ( y ; y ^ { o } ) = - \| y - y ^ { o } \| _ { 2 } ^ { 2 }$ from § 3.1 to measure agreement between conditional likelihood samples and observed responses. The fixed conditioning reference $\theta ^ { \mathrm { r e f } }$ combines the biophysical parameters θ and receptor states R in the BMP model. For an observation $y ^ { o }$ under design $\xi$ and n Monte Carlo response samples $\{ y _ { j } \} _ { j = 1 } ^ { n }$ drawn from the diffusion sampler at this fixed condition,

$$
\{ y _ { j } \} _ { j = 1 } ^ { n } \sim \pi ^ { \star } ( \phi ; \theta ^ { \mathrm { r e f } } , \xi ) ,
$$

we estimate the expected reward by

$$
\mathbb { E } _ { \boldsymbol { y } \sim p _ { \phi } ( \cdot | \theta ^ { \mathrm { r e f } } , \xi ) } [ R ( \boldsymbol { y } ; \boldsymbol { y } ^ { o } ) ] \approx - \frac { 1 } { n } \sum _ { j = 1 } ^ { n } \bigl \| y _ { j } - \boldsymbol { y } ^ { o } \bigr \| _ { 2 } ^ { 2 } .
$$

We aggregate these per-condition rewards across a minibatch during fine-tuning. At fixed conditioning, $\mathbb { E } \| Y ^ { \smile } - \check { y ^ { 0 } } \| _ { 2 } ^ { 2 } = \| \dot { \mathbb { E } Y } - y ^ { \mathrm { o } } \| _ { 2 } ^ { 2 } + \operatorname { t r } \mathrm { C o v } ( Y )$ , so the corresponding sample-level loss penalizes both mean prediction error and predictive variance.

Adjoint gradients. The diffusion sampler induces the mapping $\pi ^ { \star } ( \phi ; \theta ^ { \mathrm { r e f } } , \xi )$ implicitly through the reverse-time SDE. At fixed $( \theta ^ { \mathrm { r e f } } , \xi )$ , the likelihood score defines the sampling drift $\mu _ { \phi } ^ { y }$ , with dynamics integrated from $t = T$ to $t = 0$

$$
\begin{array} { r } { \mu _ { \phi } ^ { y } ( y , t ) : = f ( y , t ) - g ( t ) ^ { 2 } s _ { \phi } ^ { y } ( y , t ; \theta ^ { \mathrm { r e f } } , \xi ) , } \end{array}\tag{46}
$$

$$
d Y _ { t } = \mu _ { \phi } ^ { y } ( Y _ { t } , t ) d t + g ( t ) d { \overleftarrow { w } } _ { t } .\tag{47}
$$

Here $f$ and $g$ are the forward-SDE coefficients from $\ S \ 2 . 3$ . The superscript $y$ distinguishes this likelihood drift from the compositional transition mean $\mu _ { k } ^ { \mathrm { G } }$ in Eq. (4). The diffusion coefficient and starting noise law are independent of $\phi .$ We assume the drift is differentiable in $y$ and $\phi ,$ with integrable sensitivities permitting differentiation under the expectation. Holding the sampled noise realization fixed, the adjoint starts from the reward gradient at $Y _ { 0 }$ and follows the same trajectory from $t = 0 \mathrm { t o } t = T$ . The system of Marion et al. (2025, Appendix A.3, Eq. 14), expressed in this clock with column-vector gradients, becomes

Table 5: BMP regularization and Langevin-correction sweep. Each setting used 250 HBDS draws and one sampling seed. L is the number of corrector steps per diffusion step.
<table><tr><td></td><td colspan="2"> $L = 0$ </td><td colspan="2"> $L = 1$ </td><td colspan="2"> $L = 3$ </td><td colspan="2"> $L = 5$ </td></tr><tr><td>λ</td><td>RMSE↓</td><td>Median↓</td><td>RMSE↓</td><td>Median↓</td><td>RMSE↓</td><td>Median ↓</td><td>RMSE↓</td><td>Median↓</td></tr><tr><td>0</td><td>0.29</td><td>12.16</td><td>0.36</td><td>12.96</td><td>0.36</td><td>12.97</td><td>0.36</td><td>12.97</td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>0.30</td><td>28.15</td><td>0.32</td><td>89.55</td><td>0.32</td><td>90.63</td><td>0.32</td><td>90.87</td></tr><tr><td> $5 \times 1 0 ^ { - 4 }$ </td><td>0.28</td><td>14.30</td><td>0.34</td><td>14.89</td><td>0.33</td><td>14.62</td><td>0.34</td><td>14.94</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>0.34</td><td>29.97</td><td>0.36</td><td>41.20</td><td>0.36</td><td>41.36</td><td>0.36</td><td>41.86</td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td>0.33</td><td>45.26</td><td>0.40</td><td>93.51</td><td>0.42</td><td>94.78</td><td>0.41</td><td>99.30</td></tr><tr><td> $1 0 ^ { - 1 }$ </td><td>0.30</td><td>52.17</td><td>0.52</td><td>157.55</td><td>0.53</td><td>153.48</td><td>0.48</td><td>159.86</td></tr><tr><td>1</td><td>0.30</td><td>47.00</td><td>0.40</td><td>130.73</td><td>0.41</td><td>134.90</td><td>0.41</td><td>139.27</td></tr></table>

$$
A _ { 0 } = \nabla _ { y } R ( Y _ { 0 } ; y ^ { o } ) ,
$$

$$
\frac { d A _ { t } } { d t } = - \big [ \partial _ { y } \mu _ { \phi } ^ { y } ( Y _ { t } , t ) \big ] ^ { \top } A _ { t } ,
$$

$$
G _ { 0 } ^ { \phi } = 0 ,
$$

$$
\frac { d G _ { t } ^ { \phi } } { d t } = - \big [ \partial _ { \phi } \mu _ { \phi } ^ { y } ( Y _ { t } , t ) \big ] ^ { \top } A _ { t } .
$$

Here $A _ { t }$ propagates the endpoint reward sensitivity along the sampled path, while $G _ { t } ^ { \phi }$ accumulates the parameter gradient. The minus signs account for expressing the adjoint in the original diffusion clock. Taking the expectation of $G _ { T } ^ { \phi }$ over paths yields the endpoint reward gradient with respect to ϕ, which enters the minimized objective with a minus sign. To include the path penalty, we differentiate the full sampled trajectory cost,

$$
- R ( Y _ { 0 } ; y ^ { o } ) + \frac { \lambda } { 2 } \int _ { 0 } ^ { T } g ( t ) ^ { 2 } \big \| s _ { \phi } ^ { y } ( Y _ { t } , t ) - s _ { \phi _ { 0 } } ^ { y } ( Y _ { t } , t ) \big \| _ { 2 } ^ { 2 } d t .
$$

Averaging this cost over FT paths gives the SOC objective in Eq. (52), with the integrated penalty contributing $\lambda \mathcal { A } ( \theta ^ { \mathrm { r e f } } , \xi )$ from Eq. (2). Beyond the reward-only adjoint above, differentiation includes the integral’s dependence on both the sampled states $Y _ { t }$ and the parameters ϕ. Our implementation uses checkpointed backpropagation through the discretized rollout to compute discrete-adjoint gradients of both the endpoint reward and the integrated penalty (§ G.6).

For BMP, we fixed $\theta ^ { \mathrm { r e f } } = ( \theta ^ { \mathrm { L S R } } , R ^ { \mathrm { L S R } } )$ at a best-fit least-squares regression (LSR) solution. We note that the choice of reference is an important hyperparameter during fine-tuning, similar to the choice of a known pair in Wehenkel et al. (2025). In $\ S \operatorname { G } . 4 ,$ , we propose alternating Fisher–Rao-guided updates of posterior proposal particles with likelihood fine-tuning, allowing the mechanistic reference to evolve alongside the model.

Regularization and corrector sweeps. We compared seven regularization strengths and $L \in$ 0, 1, 3, 5 Langevin corrections per diffusion step using the four-layer FiLM model (Table 5). Each setting used HBDS with 100 diffusion steps, three block-refinement rounds, Tweedie conditioning for both blocks, 250 posterior draws, and sampling seed 42. All settings retained the derived diffusion coefficient, including the predictor-only $L = 0$ case. At $L = 5 , \lambda = 1 0 ^ { - 4 }$ gives the lowest RMSE among positive penalties, while $\lambda = 5 \stackrel { \cdot } { \times } 1 0 ^ { - 4 }$ gives the lowest median distance. The unregularized model has the lowest median distance overall. In this single-seed sweep, omitting corrections yields lower errors than adding them for every $\lambda ,$ showing that more corrector steps need not improve predictive fit at a fixed discretization. This is a correction-count ablation, not a comparison with the ordinary VP coefficient.

## G.2 PATH-SPACE FORMULATION AND GIRSANOV’S THEOREM

We briefly state the change-of-measure result used in §§ 3.1 and 3.2. Let $\Omega = C ( [ 0 , T ] ; \mathbb { R } ^ { d } )$ denote the space of continuous trajectories. Consider two SDEs evolving from $t = 0 \mathrm { t o } t \overset { \cdot } { = } T$ with the same

initial law and scalar diffusion coefficient,

$$
\begin{array} { r l r } & { } & { d X _ { t } = b _ { 0 } ( X _ { t } , t ) d t + g ( t ) d w _ { t } , \qquad X _ { 0 } \sim p _ { 0 } , } \\ & { } & { d X _ { t } = b _ { 1 } ( X _ { t } , t ) d t + g ( t ) d \widetilde { w } _ { t } , \qquad X _ { 0 } \sim p _ { 0 } , } \end{array}
$$

and let $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ denote their induced probability measures on Ω. We assume $g ( t ) > 0$ and that solutions exist, remain finite on $[ 0 , T ]$ , and have uniquely determined path laws. Here w and $\widetilde { w }$ are Wiener processes under $\mathbb { P } _ { 0 }$ and $\mathbb { P } _ { 1 }$ , respectively. It is convenient to express the drift perturbation in diffusion-normalized coordinates,

$$
u _ { t } ( x ) : = \frac { b _ { 1 } ( x , t ) - b _ { 0 } ( x , t ) } { g ( t ) } .
$$

In stochastic optimal control (SOC) this normalized drift perturbation is conventionally interpreted as the control; we return to this connection in § G.3. We further assume that $u _ { t }$ is sufficiently integrable for the associated stochastic exponential to be a true martingale under $\mathbb { P } _ { 0 }$ (Novikov’s condition is a standard sufficient condition). This ensures that the trajectory weights have expectation one under the reference law and define a normalized probability measure. Under these assumptions, Girsanov’s theorem (Girsanov, 1960) expresses the change from drift $b _ { 0 }$ to $b _ { 1 }$ through the following Radon–Nikodym derivative, which reweights reference trajectories to obtain the modified path law,

$$
\frac { d \mathbb { P } _ { 1 } } { d \mathbb { P } _ { 0 } } ( \boldsymbol { X } ) = \exp \left( \int _ { 0 } ^ { T } u _ { t } ( \boldsymbol { X } _ { t } ) ^ { \top } d w _ { t } - \frac { 1 } { 2 } \int _ { 0 } ^ { T } \| u _ { t } ( \boldsymbol { X } _ { t } ) \| _ { 2 } ^ { 2 } d t \right) .\tag{48}
$$

Here the stochastic integral is taken under the reference law $\mathbb { P } _ { 0 } .$ , induced by drift $b _ { 0 }$ . The Radon– Nikodym derivative reweights these trajectories to recover expectations under $\mathbb { P } _ { 1 }$ , the law induced by the changed drift $b _ { 1 } = b _ { 0 } + g u$ with the same diffusion coefficient and initial distribution. Consequently, provided the control energy is integrable under $\mathbb { P } _ { 1 }$

$$
\mathrm { K L } ( \mathbb { P } _ { 1 } \| \mathbb { P } _ { 0 } ) = \frac { 1 } { 2 } \mathbb { E } _ { \mathbb { P } _ { 1 } } \left[ \int _ { 0 } ^ { T } \| u _ { t } ( X _ { t } ) \| _ { 2 } ^ { 2 } d t \right] .\tag{49}
$$

For the PT and FT reverse SDEs in our setting, the shared starting law is the noise distribution at $t = T$ and the drift difference is $\mu _ { \phi } ^ { y } - \mu _ { \phi _ { 0 } } ^ { y } = - \breve { g } ( t ) ^ { 2 } ( s _ { \phi } ^ { y } - s _ { \phi _ { 0 } } ^ { y } )$ . The normalized reverse-drift change is therefore $u _ { t } = - g ( t ) ( s _ { \phi } ^ { y } - s _ { \phi _ { 0 } } ^ { y } )$ . Rewriting the processes in increasing sampling time reverses this sign but leaves the squared control energy unchanged, yielding Eq. (2).

## G.3 STOCHASTIC OPTIMAL CONTROL CONNECTION

Likelihood fine-tuning as drift control. We fix $( \theta ^ { \mathrm { r e f } } , \xi )$ and suppress these conditioning arguments. To connect the Implicit Diffusion notation in $\ S \operatorname { G } .$ 1 to the SOC notation, we write the same drifts as $b _ { \phi } = \mu _ { \phi } ^ { y }$ and $b _ { 0 } = b _ { \phi _ { 0 } } = \mu _ { \phi _ { 0 } } ^ { y }$ , where the subscript 0 denotes the PT reference, not a time. We retain the reverse-SDE convention in which diffusion time t decreases from $T$ to 0, so sampling ends at $t = 0$ . Let $Y _ { t } ^ { \phi _ { 0 } }$ and $Y _ { t } ^ { \phi }$ denote the PT and FT reverse-SDE trajectories, with path laws $\mathbb { P } _ { \phi _ { 0 } }$ and $\mathbb { P } _ { \phi }$ respectively, and dynamic

$$
\begin{array} { l } { { d Y _ { t } ^ { \phi _ { 0 } } = b _ { 0 } ( Y _ { t } ^ { \phi _ { 0 } } , t ) d t + g ( t ) d \overleftarrow { w } _ { t } , } } \\ { { d Y _ { t } ^ { \phi } = \left[ b _ { 0 } ( Y _ { t } ^ { \phi } , t ) + g ( t ) u _ { \phi } ( Y _ { t } ^ { \phi } , t ) \right] d t + g ( t ) d \overleftarrow { \widetilde { w } } _ { t } , } } \end{array}
$$

where

$$
u _ { \phi } ( y , t ) : = g ( t ) ^ { - 1 } \big [ b _ { \phi } ( y , t ) - b _ { 0 } ( y , t ) \big ] = - g ( t ) \big [ s _ { \phi } ^ { y } ( y , t ) - s _ { \phi _ { 0 } } ^ { y } ( y , t ) \big ] .\tag{50}
$$

Thus, fine-tuning changes the reverse drift from $b _ { 0 }$ to $b _ { \phi } = b _ { 0 } + g u _ { \phi }$ , while preserving the diffusion coefficient $g ( t )$ and the common starting noise law at $t = T$ . Let $\pi _ { \phi } = \pi ^ { \star } ( \phi ; \theta ^ { \mathrm { r e f } } , \xi )$ denote the corresponding endpoint distribution. Extracting the generated response $Y _ { 0 }$ is a measurable projection of the full trajectory, so the data-processing inequality and the control-energy identity in Eq. (2) give

$$
\mathrm { K L } ( \pi _ { \phi } \| \pi _ { \phi _ { 0 } } ) \leq \mathrm { K L } ( \mathbb { P } _ { \phi } \| \mathbb { P } _ { \phi _ { 0 } } ) = \frac { 1 } { 2 } \mathbb { E } _ { \mathbb { P } _ { \phi } } \int _ { 0 } ^ { T } \| u _ { \phi } ( Y _ { t } , t ) \| _ { 2 } ^ { 2 } d t ,\tag{51}
$$

under the assumptions in § G.2. Replacing the endpoint penalty in Eq. (1) by this path penalty gives the parameterized SOC cost

$$
\begin{array} { r l } & { \mathcal { T } _ { \lambda } ( \phi ) : = - \mathbb { E } _ { y \sim \pi _ { \phi } } [ R ( y ; y ^ { o } ) ] + \lambda \mathrm { K L } ( \mathbb { P } _ { \phi } \| \mathbb { P } _ { \phi _ { 0 } } ) } \\ & { \quad \quad = \mathbb { E } _ { \mathbb { P } _ { \phi } } \left[ - R ( Y _ { 0 } ; y ^ { o } ) + \displaystyle \frac { \lambda } { 2 } \int _ { 0 } ^ { T } \| u _ { \phi } ( Y _ { t } , t ) \| _ { 2 } ^ { 2 } d t \right] , } \end{array}\tag{52}
$$

This is $\mathrm { K L }$ -regularized SOC within the score-network parameterization, with observation loss $- R$ as the sampling-endpoint cost and λ pricing control effort (Uehara et al., 2024b). Since the expected reward is unchanged and $\lambda \geq 0$ , data processing gives ${ \mathcal { F } } ( \pi ^ { \star } ( \phi ) ) \leq { \mathcal { T } } _ { \lambda } ( \phi )$ The SOC cost is therefore a path-regularized upper bound on the endpoint objective, not an identical rewrite of it. We optimize this cost through the score-network parameters using the checkpointed rollout differentiation described in $\ S \ G . 1$ , without learning a separate control network or value function. Rewriting the sampler in increasing sampling time reverses the drift and control signs but leaves the quadratic cost unchanged.

Path-space trust regions. The same penalty admits a constrained interpretation,

$$
\operatorname* { m i n } _ { \phi } \ - \mathbb { E } _ { \mathbb { P } _ { \phi } } [ R ( Y _ { 0 } ; y ^ { o } ) ] \qquad \mathrm { s u b j e c t ~ t o } \qquad \mathrm { K L } ( \mathbb { P } _ { \phi } | | \mathbb { P } _ { \phi _ { 0 } } ) \leq \rho ,\tag{53}
$$

where $\rho \ge 0$ is a divergence budget. Its Lagrangian is $\mathcal { I } _ { \lambda } ( \phi ) - \lambda \rho$ with multiplier $\lambda \geq 0$ , so fixing λ recovers the penalized optimization over $\phi .$ This motivates choosing the penalty through a divergence budget rather than treating it only as a tuning constant, in the spirit of path-space trust-region methods (Blessing et al., 2025). For multiple observed designs, constraints KL $\mathopen { } \mathclose \bgroup ( \mathbb { P } _ { \phi } ^ { \theta ^ { \mathrm { r e f } } , \xi } \mathopen { } \mathclose \bgroup \| \mathbb { P } _ { \phi _ { 0 } } ^ { \theta ^ { \mathrm { r e f } } , \xi } \aftergroup \egroup ) \leq \rho ( \xi )$ would introduce design-specific multipliers $\lambda _ { \xi }$ . Adjusting these against their budgets could regulate how strongly each condition is adapted, alongside a curriculum that selects which designs to revisit. This is a prospective extension, not an implemented adaptive optimizer or a claim of strong duality for the neural parameterization.

## G.4 FISHER–RAO OPTIMIZATION ANALYSIS

Let $z = ( \theta , \xi )$ collect the mechanistic parameters and experimental design. Under the regularity assumptions in § 3.2, the conditional reverse-SDE path laws $\{ \bar { \mathbb { P } } _ { \phi } ^ { z } : z \in$ $\Theta \times \Xi \}$ can be viewed locally as a statistical manifold. Its Fisher–Rao geometry is Riemannian on neighborhoods where the Fisher metric is nonsingular, and degenerate where it is singular. The path-space Fisher metric in Eq. (3) can be partitioned according to the two types of conditioning variables (Figure 10),

![](images/3d6fd5193366695cf3d88b0ae7e16a39d3515ef1c6f5f408779fc32654aada2b.jpg)  
Figure 10: Joint parameter Fisher geometry.

$$
\mathcal { T } _ { \phi } ^ { \mathrm { p a t h } } ( z ) = \left[ I _ { \theta \theta } ( z ) \quad I _ { \theta \xi } ( z ) \right] ,\tag{54}
$$

For example, the parameter block of Eq. (3) is

$$
I _ { \theta \theta } ( z ) = \mathbb { E } _ { Y \sim \mathbb { P } _ { \phi } ^ { z } } \left[ \int _ { 0 } ^ { T } g ( t ) ^ { 2 } J _ { \theta } s _ { \phi } ^ { y } ( Y _ { t } , t ; z ) ^ { \top } J _ { \theta } s _ { \phi } ^ { y } ( Y _ { t } , t ; z ) d t \right] .
$$

We first consider how this geometry could refine a population of mechanistic reference parameters during fine-tuning. We then examine design sensitivity and parameter–design coupling, before propagating reference uncertainty into path-divergence diagnostics.

Refining the posterior proposal. The block $I _ { \theta \theta }$ measures how perturbations of the mechanistic parameters change the conditional path law while holding the experimental design fixed. As a proposed extension, the fixed reference $\theta ^ { \mathrm { r e f } }$ could be replaced by a posterior proposal distribution $q ( \theta )$ that is updated during fine-tuning. Consider one observed design–response pair $( \bar { \xi } , y ^ { o } )$ and m proposal particles $\theta ^ { ( i ) } \sim q .$ . For each particle, we draw n predictive response samples $y _ { j } ^ { ( i ) } \sim p _ { \phi } ( \cdot \mid \theta ^ { ( i ) } , \xi )$ at this same design. We use the same reward $R ( y ; y ^ { o } )$ as in § G.1 and write $\mathbb { E } _ { \phi } [ \cdot \ : | \ : \theta , \xi ]$ for expectation over responses from this conditional likelihood. Averaging over proposal particles and predictive responses gives

$$
\mathbb { E } _ { \theta \sim q } \mathbb { E } _ { \phi } [ R ( y ; y ^ { o } ) \mid \theta , \xi ] \approx - \frac { 1 } { m n } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { n } \| y _ { j } ^ { ( i ) } - y ^ { o } \| _ { 2 } ^ { 2 } .
$$

At fixed $\phi ,$ the same reward can be used to tilt the posterior proposal toward parameter values whose conditional likelihood better predicts the observation. Specifically, consider

$$
q _ { \phi } ^ { \star } = \arg \operatorname* { m i n } _ { \boldsymbol { q } } \left\{ - \mathbb { E } _ { \boldsymbol { \theta } \sim \boldsymbol { q } } \left[ \mathbb { E } _ { \boldsymbol { \phi } } [ R ( \boldsymbol { y } ; \boldsymbol { y } ^ { o } ) \mid \boldsymbol { \theta } , \boldsymbol { \xi } ] \right] + \tau \mathrm { K L } ( \boldsymbol { q } \| \boldsymbol { p } ) \right\} ,\tag{55}
$$

$$
q _ { \phi } ^ { \star } ( \theta ) \propto p ( \theta ) \exp \left[ \frac { \mathbb { E } _ { \phi } [ R ( y ; y ^ { o } ) \mid \theta , \xi ] } { \tau } \right] ,\tag{56}
$$

where $p ( \theta )$ is the prior and $\tau > 0$ controls the strength of the reward tilt, so predictive agreement reweights the prior toward a posterior proposal concentrated on parameter values that better explain the observation. Recent concurrent work similarly contrasts KL reward tilting with transport-based adaptation, moving samples toward higher reward under a geometry induced by the pretrained generative dynamics (Mammadov et al., 2026). Motivated by a related transport view in mechanistic parameter space, we instead use the Fisher geometry of the conditional diffusion path law to move proposal particles in $\theta .$

The Fisher block $I _ { \theta \theta } ( \theta , \xi )$ provides a local geometry for reward-driven motion of the posterior proposal particles. At iteration $k ,$ , with $\phi _ { k }$ fixed, we propose the particle update

$$
\begin{array} { r l } & { \theta _ { k + 1 } ^ { ( i ) } = \theta _ { k } ^ { ( i ) } + \Delta \theta _ { k } ^ { ( i ) } , } \\ & { \Delta \theta _ { k } ^ { ( i ) } = \eta I _ { \theta \theta } ( \theta _ { k } ^ { ( i ) } , \xi ) ^ { \dagger } \nabla _ { \theta } \mathbb { E } _ { \phi _ { k } } [ R ( y ; y ^ { o } ) \mid \theta , \xi ] | _ { \theta = \theta _ { k } ^ { ( i ) } } } \\ & { \qquad + \eta \tau \nabla _ { \theta } \log p ( \theta _ { k } ^ { ( i ) } ) + \sqrt { 2 \eta \tau } \varepsilon _ { k } ^ { ( i ) } , \qquad \varepsilon _ { k } ^ { ( i ) } \sim \mathcal { N } ( 0 , I ) . } \end{array}\tag{57}
$$

(58)

Here $\eta > 0$ is the particle-update step size, and denotes the Moore–Penrose pseudoinverse. For a positive-semidefinite Fisher matrix, it inverts positive eigenvalues while leaving zero eigenvalues at zero, reducing to the ordinary inverse when the matrix is nonsingular. The first term moves each proposal particle toward improved predictive agreement using the learned path geometry. In Fisher-null directions this term vanishes, while the prior drift and stochastic term remain active, allowing the proposal population to explore mechanistic variation that is not locally resolved by the conditional path law.

The reward-tilted distribution $q _ { \phi _ { k } } ^ { \star }$ expresses an ideal balance between predictive agreement and proximity to the prior. Motivated by this balance, the updates above use predictive sensitivities, prior drift, and stochastic exploration to refine an evolving proposal population, without assuming that its particles are distributed according to $q _ { \phi _ { k } } ^ { \star }$ . After updating all m particles, we represent the next proposal by the empirical measure

$$
q _ { k + 1 } ( \theta ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { \theta _ { k + 1 } ^ { ( i ) } } ( \theta ) .
$$

Holding these proposal particles fixed, we update the likelihood parameters by averaging the finetuning objective in Eq. (1) across them, using the path-space penalty  from Eq. (2) in place of the endpoint KL as in the main text,

$$
\phi _ { k + 1 } \in \arg \operatorname* { m i n } _ { \phi } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \left[ - \mathbb { E } _ { \phi } \left[ R ( y ; y ^ { o } ) \mid \theta _ { k + 1 } ^ { ( i ) } , \xi \right] + \lambda \mathcal { A } \Big ( \theta _ { k + 1 } ^ { ( i ) } , \xi \Big ) \right] ,\tag{59}
$$

Here $\mathcal { A } ( \theta , \xi )$ is evaluated for the candidate model $\phi$ relative to the fixed PT model $\phi _ { 0 } .$ , so each summand is the conditional SOC cost $\mathcal { I } _ { \lambda }$ in Eq. (52). The procedure therefore alternates between updating the posterior proposal under the current likelihood and fine-tuning the likelihood across the resulting proposal particles.

Design sensitivity and active simulation acquisition. The block $I _ { \xi \xi }$ measures how strongly the conditional path law changes under local perturbations of the experimental design. For a small $\delta \xi$

$$
\mathrm { K L } \left( \mathbb { P } _ { \phi } ^ { \theta , \xi + \delta \xi } \Big | \Big | \mathbb { P } _ { \phi } ^ { \theta , \xi } \right) \approx \frac { 1 } { 2 } \delta \xi ^ { \top } I _ { \xi \xi } ( \theta , \xi ) \delta \xi .
$$

PT model. Large $I _ { \xi \xi }$ identifies regions where the learned simulator surrogate changes rapidly with design. Scalar summaries such as tr $I _ { \xi \xi }$ or log det $( I _ { \xi \xi } + \epsilon I )$ could therefore prioritize additional simulator calls where denser coverage is most useful, rather than expanding the pretraining set uniformly. Related work uses diffusion models to acquire simulation parameters from surrogate loss or uncertainty in online PDE-surrogate training (Cesar et al., 2026), whereas our proposed signal is derived from local path-space sensitivity.

FT model. $I _ { \xi \xi }$ instead measures sensitivity of the adapted likelihood across experimental conditions. Designs with large sensitivity may warrant greater weight in a fine-tuning curriculum, particularly when accompanied by high predictive error or large PT–FT path divergence. Sensitivity alone does not imply misspecification, since additional PT simulations improve fidelity to the simulator whereas $\bar { \mathcal { A } } ( \theta , \bar { \xi } )$ measures departure from the simulator toward observations.

Experimental design. The same geometry could also guide the choice of experiments for learning mechanistic parameters. At a candidate design $\xi ,$ large $I _ { \theta \theta } ( \theta , \xi )$ indicates strong local sensitivity of the predictive path law to θ. An objective such as

$$
\log \operatorname* { d e t } \left( I _ { \theta \theta } ( \theta , \xi ) + \epsilon I \right)
$$

could therefore prioritize experimental conditions that are locally informative about the mechanistic parameters.

Parameter–design coupling for proposal refinement. Beyond selecting new designs, the joint geometry could help refine posterior proposals using the designs already observed.

Parameter–design overlap. The cross block $I _ { \theta \xi }$ measures whether perturbations of θ and ξ induce similar first-order changes in the conditional path law. Removing their respective local scales gives the normalized coupling

$$
C ( \theta , \xi ) = I _ { \theta \theta } ^ { \dag / 2 } I _ { \theta \xi } I _ { \xi \xi } ^ { \dag / 2 } ,\tag{60}
$$

with $I ^ { \dag / 2 } = ( I ^ { \dag } ) ^ { 1 / 2 }$ . The singular vectors of C identify parameter and design directions with aligned effects. Larger singular values indicate parameter-induced changes that can be closely reproduced by local design perturbations, while smaller values indicate more distinct effects.

Residual parameter geometry. Removing this design-associated component from the parameter block gives the Schur complement

$$
I _ { \theta | \xi } ^ { \mathrm { e f f } } = I _ { \theta \theta } - I _ { \theta \xi } I _ { \xi \xi } ^ { \dag } I _ { \xi \theta } .\tag{61}
$$

Geometrically, $I _ { \theta | \xi } ^ { \mathrm { e f f } }$ retains the component of parameter-induced path-law variation that cannot be reproduced locally by changing the design.

Proposal refinement across observed designs. For a collection of observed designs, a non-negative weighted average of $I _ { \theta | \xi } ^ { \mathrm { e f f } } ( \theta , \xi )$ could replace $I _ { \theta \theta }$ in the proposal-particle update in Eq. (58). The resulting geometry emphasizes mechanistic directions whose effects remain distinguishable across the fine-tuning curriculum after removing locally design-mimickable variation.

The same effective information could also be used prospectively to favor new designs at which weakly identified parameter directions become distinguishable from nearby design perturbations.

Uncertainty-aware path divergence. A distribution over reference parameters also yields an uncertainty-aware version of the adaptation diagnostic. In place of evaluating the path divergence $\mathcal { A } ( \theta , \xi )$ from Eq. (2) at one $\theta ^ { \mathrm { r e f } }$ , define

$$
\overline { { \mathcal { A } } } _ { q } ( \xi ) = \mathbb { E } _ { \theta \sim q } \left[ \mathcal { A } ( \theta , \xi ) \right] ,\tag{62}
$$

$$
V _ { A , q } ( \boldsymbol { \xi } ) = \operatorname { V a r } _ { \boldsymbol { \theta } \sim q } \left[ A ( \boldsymbol { \theta } , \boldsymbol { \xi } ) \right] .\tag{63}
$$

The first quantity identifies designs at which substantial adaptation is required across plausible mechanistic parameterizations, whereas the second identifies designs whose apparent misspecification depends strongly on parameter uncertainty. Here $V _ { A , q } ( \xi )$ quantifies how sensitive the inferred adaptation is to uncertainty in the mechanistic reference θ. This is distinct from Monte Carlo uncertainty arising from the finite trajectory samples used to estimate each $\mathcal { A } ( \theta , \xi )$

Future work. Although we do not develop these extensions here, they suggest promising ways to improve both PT and FT models in future SBI applications. For example, path-space geometry could guide refinement and particle transport of posterior proposals over mechanistic parameters, while incorporating parameter uncertainty into adaptation diagnostics (Eqs. (62) and (63)). It could also inform simulation acquisition for pretraining, complementing diffusion-based active sampling (Cesar et al., 2026), while supporting design-adaptive fine-tuning curricula and local parameter–design identifiability analysis (Eqs. (60) and (61)).

## G.5 RELATED WORK ON SIMULATOR MISSPECIFICATION IN SBI

Simulation-based inference (SBI) treats the expert model simulator as an implicit, or intractable, likelihood model from which one can sample, $y \sim p ( y | \theta )$ , but whose density cannot be evaluated. Starting with samples from a prior $\theta \sim p ( \theta )$ , synthetic pairs (y, θ) are generated to fit a neural density estimator for the likelihood $p _ { \phi } ( y | \theta )$ (Lueckmann et al., 2019; Papamakarios et al., 2019), posterior $p _ { \phi } ( \theta | y )$ (Greenberg et al., 2019; Lueckmann et al., 2017; Papamakarios & Murray, 2016; Sharrock et al., 2024), or likelihood-to-evidence ratio (Durkan et al., 2020; Miller et al., 2022) using observed data $y _ { o } .$ , enabling parameter inference without explicit likelihoods. Thus, SBI is well-suited for biological simulators that use stochastic differential equations or simulation by convex optimization (Dirks et al., 2007; Su et al., 2022). SBI relies on a simulator that accurately reflects the true, and unknown, data-generating process of the observed data, i.e., $\begin{array} { r } { p ^ { \star } ( y _ { o } ) : = \int p ^ { \star } ( \theta ) p ^ { \star } ( y _ { o } | \theta ) d \theta } \end{array}$ , which is typically unavailable and can be the cause of poor performance of SBI models (Cannon et al., 2022; Schmitt et al., 2024). This has resulted in a variety of techniques to improve robustness of SBI inference to model misspecification (Gao et al., 2023; Huang et al., 2023a; Mishra et al., 2026; Ward et al., 2022; Wehenkel et al., 2025) but there is a gap in methods that transfer improvement in predictive accuracy to posterior inference.

## G.6 RELATED WORK ON DIFFUSION FINE-TUNING

Diffusion fine-tuning is a rapidly growing field with diverse optimization and regularization strategies. We review a selection of approaches relevant to SBI, summarized in Table 6, rather than attempting an exhaustive survey. The comparison is intended to highlight the main tradeoffs among methods, including what feedback they require, how that signal is converted into model updates, whether gradients must pass through sampling, what auxiliary quantities must be learned, and how strongly the adapted model is tied to its pretrained reference. We then motivate our use of Implicit Diffusion for the present SBI setting while highlighting alternative strategies that may be preferable under different feedback, computational, and regularization requirements.

Reward optimization and gradient estimators. Reward-based diffusion fine-tuning encompasses policy-gradient and direct-differentiation methods, with different requirements on reward access (Uehara et al., 2024a). DPOK (Fan et al., 2023) and DDPO (Black et al., 2024) treat denoising as a sequential decision process, allowing optimization without differentiating the reward. DPOK explicitly regularizes deviation from the PT model through a KL penalty and optionally learns a value baseline to reduce gradient variance. When rewards are differentiable, ReFL (Xu et al., 2023) backpropagates through a clean prediction from one selected late denoising step, holding earlier states fixed. AlignProp (Prabhudesai et al., 2023) uses randomly truncated rollout backpropagation, with low-rank adaptation and gradient checkpointing to reduce memory requirements. DRaFT-K (Clark et al., 2024) restricts backpropagation to the last K denoising steps, and DRaFT-LV reduces gradient variance for $K = 1$ These approaches trade access to reward gradients and the cost of differentiating long trajectories against the sampling variance of policy-gradient estimates. Preference-based alternatives such as Diffusion-DPO optimize an ELBO-derived objective from paired comparisons rather than a supplied scalar reward (Wallace et al., 2024). We focus here on methods compatible with observation-based rewards and distributional constraints.

Table 6: Representative approaches to reward- and constraint-based generative-model adaptation, compared by the feedback they require, how model updates are obtained, whether gradients propagate through sample generation, any additional learned objects, and how deviation from a reference model is controlled. In the feedback column, R denotes reward values without sample derivatives, $\nabla _ { y } R$ denotes differentiable reward or loss feedback, $( h , h ^ { \star } )$ denotes constraint statistics and target moments, and $\nabla _ { y } E$ denotes gradients of a differentiable unnormalized target energy $E ( y )$ , with $p ^ { \star } ( y ) \propto \exp [ - \check { E } ( y ) ] ;$ when an energy is instead defined relative to a reference distribution $p _ { 0 }$ the equivalent tilted target is $p ^ { \star } ( y ) \propto p _ { 0 } ( y ) \exp [ - E ( y ) ]$ . In the sampler-gradient column, “Full” denotes differentiation through the complete sampled trajectory and explicit step labels indicate partial differentiation. In the learned-auxiliary column, $\bullet \bullet \underline { { } } \ \bullet \bullet$ means that no additional function or parameter is fitted beyond the adapted generator or control and any supplied reward model; computed adjoint states, trajectory queues, and replay buffers are not counted. PT denotes a pretrained model. For Implicit Diffusion, Full∗ denotes full finite-horizon sampler differentiation. Our implementation uses checkpointed backpropagation through a complete discretized rollout rather than the original queued joint sampling–optimization scheme, so that each gradient update corresponds to a complete rollout under a single selected design.
<table><tr><td>Method</td><td></td><td>Feedback Update mechanism</td><td>Sampler gradients</td><td>Learned auxiliaries</td><td>Reference regularization</td></tr><tr><td>DPOK (Fan et al., 2023)</td><td>R</td><td>Policy gradient</td><td></td><td>Optional value baseline</td><td>Discrete path KL</td></tr><tr><td>DDPO (Black et al., R 2024)</td><td></td><td>Policy gradient</td><td></td><td></td><td>No explicit PT KL</td></tr><tr><td>ReFL (Xu et al., 2023)</td><td> $\nabla _ { y } R$ </td><td>Intermediate clean-prediction gradient</td><td>1 selected step</td><td></td><td>Pretraining denoising loss</td></tr><tr><td>AlignProp (Prabhudesai et al., 2023)</td><td> $\nabla _ { y } R$ </td><td>Randomly truncated rollout backpropagation</td><td>Random trailing steps</td><td></td><td>No explicit PT KL</td></tr><tr><td>DRaFT-K (Clark et al., 2024)</td><td> $\nabla _ { y } R$ </td><td>Truncated rollout backpropagation</td><td>Last K steps</td><td></td><td>LoRA weight decay (implementation)</td></tr><tr><td>ELEGANT (Uehara et al., 2024b)</td><td> $\nabla _ { y } R$ </td><td>SOC objective via rollout gradients</td><td>Full</td><td>Initial-time value function and initial-law sampler</td><td>Drift-control cost and initial-law KL</td></tr><tr><td>Adjoint Matching (Domingo i Enrich et al., 2025)</td><td> $\nabla _ { y } R$ </td><td>Regression on detached adjoint-derived control targets</td><td></td><td></td><td>SDE path KL (SOC)</td></tr><tr><td>TR-SOCM (Blessing et al., 2025)</td><td> $\nabla _ { y } R$ </td><td>Reweighted adjoint regression</td><td></td><td></td><td>Inter-iterate path-KL constraint</td></tr><tr><td>Adjoint Sampling (Havens et al., 2025)</td><td> $\nabla _ { y } E$ </td><td>Adjoint regression with reference bridges</td><td></td><td></td><td>Control cost, energy target</td></tr><tr><td>Implicit Diffusion (Marion et al., 2025)</td><td> $\nabla _ { y } R$ </td><td>Joint sampling- optimization with adjoint gradients</td><td> $\mathrm { F u l l } ^ { * }$ </td><td></td><td>Optional SDE path KL</td></tr><tr><td>CGM-relax / CGM-reward (Smith et al., 2026)</td><td>(h, h*)</td><td>Likelihood-ratio gradients</td><td></td><td>Tilt coefficients (reward only)</td><td>KL to PT (path KL for SDEs)</td></tr><tr><td>Value Matching (Jensen et al., 2026)</td><td>R</td><td>Online value-function learning</td><td></td><td>Value function</td><td>Control-based</td></tr></table>

Stochastic optimal control. SOC treats fine-tuning as learning a change to the PT drift that improves terminal reward while penalizing the accumulated control effort. With a shared initial law and diffusion coefficient, Girsanov’s theorem expresses path-space KL as a quadratic cost of this drift change (§ G.2). This gives a trajectory-level notion of staying close to the PT model, rather than a penalty on network weights. ELEGANT (Uehara et al., 2024b) uses this formulation to optimize both the denoising drift and the initial noise distribution, regularizing both changes. Its implementation estimates the value function at the initial diffusion time and learns an auxiliary process for sampling the adapted initial distribution. Adjoint Matching (Domingo i Enrich et al., 2025) takes a different route, fitting the control to targets obtained from a simplified backward adjoint equation. Trajectories and targets are held fixed during each regression update, avoiding differentiation through the controlled rollout, although reward and base-drift derivatives are still needed to construct the targets. Its reward fine-tuning formulation uses a memoryless noise schedule to recover the intended reward-tilted output distribution without separately learning the initial noise law. TR-SOCM (Blessing et al., 2025) fits controls using reweighted adjoint targets with a path-KL constraint relative to the preceding iterate. A scalar dual multiplier controls the trust region, while buffered trajectories support reuse across updates. Adjoint Sampling (Havens et al., 2025) instead applies adjoint regression to unnormalized energy targets, using reference bridges to reuse terminal samples and energy gradients across multiple parameter updates. It learns an energy-targeting sampler rather than preserving a PT data distribution. These options differ in the auxiliary quantities they learn, their sampling requirements, and their target distributions.

Implicit Diffusion and sampler gradients. Implicit Diffusion (Marion et al., 2025) separates gradient estimation through a parameterized sampler from the algorithm used to interleave sampling and optimization. Its finite-horizon adjoint formulation propagates terminal-loss sensitivity through the sampling dynamics and can incorporate a Girsanov path-KL penalty (Marion et al., 2025, Appendix A.3). Algorithm 3 additionally maintains a queue of trajectories that advance while model parameters change, coupling sampling with parameter updates. Our implementation instead completes reverse-SDE rollouts at fixed parameters and uses checkpointed backpropagation, computing discreteadjoint gradients while trading recomputation for memory. We therefore use the finite-horizon sampler-gradient formulation, not the queued optimization algorithm. This shares the pathwise differentiation principle of DRaFT (Clark et al., 2024), whereas Adjoint Matching holds trajectories and adjoint targets fixed during control regression. Neither checkpointing nor adjoint differentiation itself specifies the regularizer.

Controlling departure from the pretrained model. Fine-tuning methods differ not only in how they obtain parameter updates, but also in how they constrain departure from the pretrained (PT) model. DRaFT uses AdamW weight decay on its LoRA factors, shrinking the learned update toward the PT parameters (Clark et al., 2024, Appendix A.1). This is parameter-space regularization rather than a KL between sampling laws; a separate ablation instead penalizes PT–FT noise-prediction differences at the final denoising step (Clark et al., 2024, Appendix B.8). DPOK directly regularizes the sampling distribution by summing transition KLs over the discrete denoising chain, yielding a path-space KL that also bounds endpoint divergence (Fan et al., 2023). ReFL instead retains a pretraining denoising loss (Xu et al., 2023). For continuous SDEs with shared diffusion and initial laws, the corresponding path KL is the integrated drift-control energy in Eq. (49); this is the regularizer we pair with Implicit Diffusion. ELEGANT additionally regularizes changes to the initial noise law. Adjoint Sampling uses a control cost to reach an energy-defined target, so its regularization serves target transport rather than preservation of a PT data distribution.

Feedback and distributional constraints. Beyond the gradient estimator, the reliability and availability of feedback determine which adaptations are useful. Uehara et al. (2024c) study online fine-tuning with limited reward queries, while BRAID (Uehara et al., 2024d) uses conservative reward estimates to limit overoptimization outside the support of offline data. Calibrating Generative Models (CGM) (Smith et al., 2026) seeks the KL-closest model satisfying moment constraints. CGM-relax penalizes constraint violations, whereas CGM-reward fits exponential-tilt coefficients that define a reward. Both differentiate density ratios with sampled outputs held fixed, avoiding derivatives of the constraint functions and backpropagation through sample generation. For continuous diffusions, these are path-measure ratios evaluated using Girsanov’s theorem, so tractable endpoint likelihoods are not required. A response-mean constraint also differs from our expected sample-wise MSE, which penalizes predictive variance as well as mean error (§ G.1). Flow Density Control (De Santi et al., 2025) extends optimization to broader utilities and divergences through mirror-flow updates, and Value Matching (Jensen et al., 2026) studies gradient-free reward-guided flow adaptation. These works motivate considering both the computational cost of fine-tuning and whether its objective remains informative away from the observed data.

Why this fine-tuning formulation for SBI? The challenge in our SBI setting is to match a finite set of observed design–response pairs $( \xi , y _ { o } )$ without access to the true data-generating process for additional samples or queries at unmeasured designs. We adapt the conditional likelihood at a simulator-aligned $\dot { \theta } ^ { \mathrm { r e f } }$ , using discrepancy from these observations as the learning signal and the PT path law as a regularizing reference. We optimize this path-regularized objective using checkpointed backpropagation through the discretized reverse-SDE sampler. This computes pathwise parameter gradients through a discrete adjoint, following the finite-horizon sampler-gradient formulation of Implicit Diffusion (Marion et al., 2025) and sharing the differentiation principle of direct reward fine tuning (Clark et al., 2024). This lets us differentiate the observation-based loss directly without fitting a separate value function or replacing it with a control-regression objective. The SBI contribution is the adaptation at mechanistically grounded conditioning values, its transfer to posterior queries through shared FiLM-modulated embeddings (§ G.7), and the interpretation of conditional path divergence alongside predictive error, rather than a new gradient estimator. Because the required correction varies with $\xi ,$ choosing which designs to revisit, how to weight their losses, and how strongly to regularize raises a curriculum problem (Bengio et al., 2009). Our toy illustrates how updates that correct a misspecified region can degrade regions that were already accurate (§ G.9). Interpreting $\mathcal { A } ( \theta , \xi )$ alongside predictive error reveals these heterogeneous effects and motivates adaptive curricula and design-specific trust regions, with weights $\lambda _ { \xi }$ adjusted to control departure from the PT path law (Blessing et al., 2025). Such design-dependent penalties remain a proposed extension. Adjoint Matching, TR-SOCM, and Adjoint Sampling provide alternative optimization routes worth investigating for this conditional likelihood setting.

## G.7 EMBEDDING-BASED TRANSFER LEARNING

Fine-tuning a joint simformer through its likelihood requires the learned correction to transfer to posterior queries, but the original architecture of Gloeckler et al. (2024) zeros condition inputs for latent tokens and leaves this transfer largely to the transformer. We instead apply token-aware Feature-wise Linear Modulation (FiLM) (Perez et al., 2018) before the transformer. For token i, let $e _ { i } ^ { \mathrm { v a l } } , e _ { i } ^ { \mathrm { i d } }$ , and $e _ { i } ^ { \mathrm { c o n d } }$ denote its value, identity, and latent-or-conditioned state embeddings, and define $b _ { i } = [ e _ { i } ^ { \mathrm { v a l } } , e _ { i } ^ { \mathrm { i d } } ]$ . A linear projection of $[ e _ { i } ^ { \mathrm { i d } } , e _ { i } ^ { \mathrm { c o n d } } ]$ produces $[ \tilde { \gamma } _ { i } , \tilde { \beta } _ { i } ]$ , which we bound as $\gamma _ { i } =$ $1 + \varepsilon$ tanh $( \tilde { \gamma } _ { i } )$ and $\beta _ { i } = \varepsilon \operatorname { t a n h } ( \tilde { \beta } _ { i } )$ before forming the transformer input $z _ { i } = [ \gamma _ { i } \odot b _ { i } + \beta _ { i } , e _ { i } ^ { \mathrm { { c o n d } } } ]$ Thus, the condition state both enters the transformer explicitly and modulates the value–identity representation, while ε limits how far latent and conditioned embeddings can separate; likelihood fine-tuning can therefore update a representation that posterior queries reuse.

## G.8 FMCPE-LSR BASELINE

Protocol. FMCPE first trains a neural posterior estimator (NPE) on simulator-generated pairs and then uses flow matching to correct its predictions with a smaller calibration set (Ruhlmann et al., 2026). For BMP, we used the authors’ Lampe/Zuko NPE implementation and treated the 4,816 retained LSR fits as pseudo-calibration parameters. They are estimates from the observed BMP dataset, not ground-truth parameters from a high-fidelity data-generating process. The complete sweep contains 48 runs: four nested observation budgets (235, 470, 705, and 940 measurements), four nested LSR budgets (1,204, 2,408, 3,612, and 4,816 fits), and three seeds (33, 43, and 53). Observation subsets were balanced across cell lines. For each condition, we trained a dedicated NPE and correction flows, drew 1,024 posterior samples, and evaluated the first 250 through the original one-step BMP simulator over all 940 measurements.

For the simformer comparison, we evaluated the FT model and the PT model from which it was initialized, each with sampling seeds 42, 43 and 44. Every run used 250 joint draws of the 60 biophysical parameters and 20 cell-line receptor concentrations, 20 predictor steps, five corrector steps, three EM steps, hierarchical receptor sampling, and Tweedie conditioning in both parameter blocks. We computed the predictive metrics using the original BMP simulator over all 940 measurements.

Shared predictive metrics. Across our BMP posterior-inference experiments, we evaluate inferred parameters by running them through the original simulator and comparing its predictions with the observed responses. RMSE and median distance summarize this agreement, capturing the best fit among posterior draws and the typical fit across draws, respectively. For the full-data comparison here, let $\hat { \mathbf { y } } ^ { ( s ) }$ be the simulator’s response vector for posterior draw $s ,$ let $\mathbf { y } ^ { o }$ contain the 940 observed responses, and let $r _ { y } = \operatorname* { m a x } _ { i } y _ { i } ^ { o } - \operatorname* { m i n } _ { i } y _ { i } ^ { o }$ . We report

$$
\mathrm { R M S E } _ { \mathrm { m i n } } = \operatorname* { m i n } _ { s } \frac { 1 } { r _ { y } } \sqrt { \frac { 1 } { 9 4 0 } \sum _ { i = 1 } ^ { 9 4 0 } \left( \hat { y } _ { i } ^ { ( s ) } - y _ { i } ^ { o } \right) ^ { 2 } } ,\tag{64}
$$

$$
d _ { \mathrm { m e d } } = \mathrm { m e d i a n } _ { s } \left. \hat { \mathbf y } ^ { ( s ) } - \mathbf y ^ { o } \right. _ { 2 } .\tag{65}
$$

This same RMSE definition is used for every method, with the FMCPE entries taken from its saved 250-draw prediction banks. Both metrics are in response space.

Results. FMCPE-LSR can achieve lower best-draw RMSE than either simformer, particularly with larger LSR budgets (Table 7). With all 940 observations and 4,816 LSR fits, its RMSE is 0.175, compared with 0.252 for PT and 0.245 for FT. This advantage does not extend to median distance, which is 31.05 for full-budget FMCPE-LSR versus 11.11 for PT and 10.38 for FT. Neither simformer is surpassed on mean median distance by any FMCPE-LSR configuration in the sweep. For each observation subset, our FMCPE-LSR implementation required training a dedicated NPE and two correction flows, whereas the PT and FT joint models reuse their weights across subsets through masking and compositional sampling. The FMCPE networks contained 0.87–1.28 million parameters in total, including the NPE frozen during correction, compared with approximately 0.48 million for each PT/FT simformer. Thus, the lower best-draw error came with subset-specific training and a larger combined model, without a corresponding improvement in typical predictive fit.

## G.9 DESIGN-DEPENDENT MISSPECIFICATION TOY

Analytic model. The simulator and observation model in § 4.3 share the prior $\theta \sim \mathcal { N } ( 0 , I _ { 2 } )$ and noise standard deviation $\sigma = 0 . 2 { \it \Omega } _ { \mathrm { { } i } }$ , but differ by the additive discrepancy

$$
h ( \xi ) = \exp \left( - \frac { \| \xi - c \| _ { 2 } ^ { 2 } } { 2 w ^ { 2 } } \right) , \qquad c = ( 1 , - 0 . 5 ) ^ { \top } , \quad w = 0 . 4 , \quad \xi \in [ - 2 , 2 ] ^ { 2 } .
$$

Table 7: Predictive errors through the original BMP simulator on all 940 measurements, using 250 parameter draws per run. Values are mean $\pm \ \mathrm { S E }$ over three FMCPE training seeds or three sampling seeds of each fixed PT or FT simformer. FMCPE ℓ-C2ST uses separate shared-normalization correction refits and a joint 80-dimensional parameter target.
<table><tr><td>Method</td><td>Observations</td><td>LSR fits</td><td>RMSE↓</td><td>Median distance ↓</td><td>l-C2ST↓</td></tr><tr><td>FMCPE-LSR</td><td>235</td><td>1,204</td><td> $0 . 2 4 2 \pm 0 . 0 0 4$ </td><td> $2 0 3 . 6 6 1 \pm 3 8 . 3 0 7$ </td><td> $0 . 1 9 8 \pm 0 . 0 2 3$ </td></tr><tr><td>FMCPE-LSR</td><td>235</td><td>2,408</td><td> $0 . 2 3 1 \pm 0 . 0 2 6$ </td><td> $2 0 3 . 4 5 1 \pm 7 0 . 0 1 9$ </td><td> $0 . 1 5 0 \pm 0 . 0 3 9$ </td></tr><tr><td>FMCPE-LSR</td><td>235</td><td>3,612</td><td> $0 . 1 8 4 \pm 0 . 0 1 0$ </td><td> $7 7 . 8 8 7 \pm 1 9 . 5 2 1$ </td><td> $0 . 1 9 4 \pm 0 . 0 5 0$ </td></tr><tr><td>FMCPE-LSR</td><td>235</td><td>4,816</td><td> $0 . 1 6 8 \pm 0 . 0 1 1$ </td><td> $1 3 6 . 4 6 0 \pm 7 7 . 8 5 5$ </td><td> $0 . 1 5 7 \pm 0 . 0 4 9$ </td></tr><tr><td>FMCPE-LSR</td><td>470</td><td>1,204</td><td> $0 . 2 5 6 \pm 0 . 0 0 8$ </td><td> $2 8 3 . 9 3 5 \pm 2 6 . 4 8 4$ </td><td> $0 . 1 3 7 \pm 0 . 0 2 0$ </td></tr><tr><td>FMCPE-LSR</td><td>470</td><td>2,408</td><td> $0 . 2 0 2 \pm 0 . 0 0 6$ </td><td> $1 0 8 . 8 0 3 \pm 1 9 . 1 9 3$ </td><td> $0 . 0 8 1 \pm 0 . 0 1 3$ </td></tr><tr><td>FMCPE-LSR</td><td>470</td><td>3,612</td><td> $0 . 1 9 5 \pm 0 . 0 0 3$ </td><td> $1 5 4 . 7 4 8 \pm 4 6 . 5 3 7$ </td><td> $0 . 0 9 5 \pm 0 . 0 1 9$ </td></tr><tr><td>FMCPE-LSR</td><td>470</td><td>4,816</td><td> $0 . 1 7 7 \pm 0 . 0 0 3$ </td><td> $3 1 . 5 0 5 \pm 1 0 . 2 0 6$ </td><td> $0 . 0 6 5 \pm 0 . 0 0 4$ </td></tr><tr><td>FMCPE-LSR</td><td>705</td><td>1,204</td><td> $0 . 2 4 6 \pm 0 . 0 0 3$ </td><td> $2 9 6 . 3 2 2 \pm 8 8 . 5 3 5$ </td><td> $0 . 0 7 4 \pm 0 . 0 2 7$ </td></tr><tr><td>FMCPE-LSR</td><td>705</td><td>2,408</td><td> $0 . 1 9 7 \pm 0 . 0 1 4$ </td><td> $1 6 1 . 3 2 2 \pm 4 1 . 2 3 7$ </td><td> $0 . 0 9 9 \pm 0 . 0 2 2$ </td></tr><tr><td>FMCPE-LSR</td><td>705</td><td>3,612</td><td> $0 . 1 9 3 \pm 0 . 0 0 2$ </td><td> $9 8 . 6 9 7 \pm 1 8 . 0 4 2$ </td><td> $0 . 1 2 4 \pm 0 . 0 4 6$ </td></tr><tr><td>FMCPE-LSR</td><td>705</td><td>4,816</td><td> $0 . 1 6 8 \pm 0 . 0 0 7$ </td><td> $5 0 . 3 5 5 \pm 1 6 . 2 5 6$ </td><td> $0 . 0 7 2 \pm 0 . 0 3 6$ </td></tr><tr><td>FMCPE-LSR</td><td>940</td><td>1,204</td><td> $0 . 2 7 0 \pm 0 . 0 1 2$ </td><td> $2 0 3 . 9 5 0 \pm 2 0 . 4 0 9$ </td><td> $0 . 0 8 3 \pm 0 . 0 1 7$ </td></tr><tr><td>FMCPE-LSR</td><td>940</td><td>2,408</td><td> $0 . 2 1 0 \pm 0 . 0 2 5$ </td><td> $1 8 8 . 1 1 7 \pm 7 3 . 5 8 9$ </td><td> $0 . 1 1 7 \pm 0 . 0 2 9$ </td></tr><tr><td>FMCPE-LSR</td><td>940</td><td>3,612</td><td> $0 . 1 8 6 \pm 0 . 0 2 3$ </td><td> $6 2 . 0 1 1 \pm 2 6 . 7 6 6$ </td><td> $0 . 1 2 9 \pm 0 . 0 1 0$ </td></tr><tr><td>FMCPE-LSR</td><td>940</td><td>4,816</td><td> $\mathbf { 0 . 1 7 5 \pm 0 . 0 0 5 }$ </td><td> $3 1 . 0 5 1 \pm 8 . 4 6 0$ </td><td> $0 . 0 8 6 \pm 0 . 0 1 7$ </td></tr><tr><td>PT BMP Simformer</td><td>940</td><td>0</td><td> $0 . 2 5 2 \pm 0 . 0 0 3$ </td><td> $1 1 . 1 1 1 \pm 0 . 1 1 6$ </td><td> $0 . 1 5 7 \pm 0 . 0 1 0$ </td></tr><tr><td>FT BMP Simformer</td><td>940</td><td>1</td><td> $0 . 2 4 5 \pm 0 . 0 0 3$ </td><td> ${ \bf 1 0 . 3 7 6 \pm 0 . 0 5 8 }$ </td><td> $0 . 1 8 9 \pm 0 . 0 1 9$ </td></tr></table>

Pretraining drew designs uniformly over this square. Each observation panel contains $n = 1 0$ responses generated at $\mathbf { \widehat { \theta } } ^ { \star } = ( 0 . 8 , - 1 . 2 ) ^ { \top }$ ; the localization run used a separate, manually specified design panel. Fine-tuning conditioned on an LSR estimate fitted under the simulator model, not on $\theta ^ { \star }$ . We distinguish the misspecified region $h ( \xi ) \geq 0 . 1$ from its complement.

The true posterior is available analytically. Let X have rows $\xi _ { i } ^ { \top }$ , and let $\mathbf { y } ^ { o }$ and h collect the observations and discrepancies $h ( \xi _ { i } )$ . Then

$$
p ^ { \star } ( \theta \mid \mathbf { y } ^ { o } , X ) = { \mathcal { N } } ( { \boldsymbol { \mu } } _ { \star } , { \boldsymbol { \Sigma } } ) , \qquad { \boldsymbol { \Sigma } } = ( I _ { 2 } + \sigma ^ { - 2 } X ^ { \top } X ) ^ { - 1 } , \qquad { \boldsymbol { \mu } } _ { \star } = \sigma ^ { - 2 } { \boldsymbol { \Sigma } } X ^ { \top } ( \mathbf { y } ^ { o } - \mathbf { h } ) .
$$

The simulator-implied posterior has the same covariance but omits h from its mean. This gives an exact reference for evaluating posterior correction.

The toy exposes a trade-off hidden by posterior metrics alone: adaptation can move the posterior toward the truth while degrading a likelihood that was already accurate away from the discrepancy. More aggressive full-network fine-tuning reduces posterior $W _ { 2 }$ further to 0.03, but raises correctregion likelihood RMSE to 0.32.

Design-localized error and adaptation. Figure 5 uses a separate full-network fine-tuning run with joint two-dimensional RBF design embeddings and $\lambda = 0 . 1$ . We estimated the PT and FT likelihood means using 16 draws per design at the same fixed LSR reference and evaluated their absolute errors against the known noise-free response at $\theta ^ { \star }$ . The displayed $e _ { \mathrm { P T } }$ and $e _ { \mathrm { F T } }$ maps share a color scale, and their annotations average these pointwise errors over all 1,681 design-grid points. This run is distinct from the posterior benchmark above and supplies the $\lambda = 0 . 1$ setting in the regularization sweep below.

MAE decreases from 0.31 to 0.20 inside the misspecified region $( h \ge 0 . 1 )$ , but increases from 0.11 to 0.15 outside it. Path divergence has a Pearson correlation of 0.75 with h, and its mean inside the misspecified region is 4.73 times that outside. Thus, concentrated adaptation can coexist with both local correction and deterioration elsewhere; path divergence alone does not determine whether predictions improve. This analysis used one training seed, one dataset of 10 observations, and one generating parameter vector, so it is a controlled diagnostic rather than a multi-seed benchmark.

Regularization trade-off. Figure 11 varies λ for the joint-RBF model while holding the PT model, observation panel, LSR reference, and remaining training settings fixed. Strong regularization reduces response adjustment and path divergence, retaining more of the PT model, but also weakens correction near the omitted bump. Increasing λ from 1 to 10 reduces RMSE outside the misspecified region by about 3%, while increasing it inside from 0.25 to 0.29. None of the tested FT models improves full-grid RMSE over PT. Here prediction error compares the FT mean at $\theta ^ { \mathrm { L S R } }$ with the noiseless true response at $\theta ^ { \star }$ , so it includes reference-parameter error. This trade-off motivates a design-dependent penalty $\lambda ( \xi )$ that preserves well-modeled regions while permitting correction where needed; testing that extension remains future work.

Sampling-time autoguidance. Inspired by autoguidance (Karras et al., 2024), we tested score blending as a sampling-time alternative to changing the training penalty. Using the frozen PT and FT models from Figure 5, we sampled with $s _ { w } ^ { y } = s _ { \mathrm { P T } } ^ { \overline { { y } } } + w ( s _ { \mathrm { F T } } ^ { y } - s _ { \mathrm { P T } } ^ { y } )$ at the same LSR reference, where $w = 0$ recovers pretraining and $w = 1$ recovers fine-tuning. Here the endpoints differ through misspecification correction, rather than training quality alone. Figure 12 shows that intermediate weights can retain correction near the discrepancy while attenuating some errors elsewhere. With 64 response draws per design, 100 reverse steps, and matched sampling noise across weights, $w = 0 . 5$ gives full-grid MAE 0.12, compared with 0.14 for PT and 0.15 for FT. The full sweep also tested $\bar { w } \in \{ 1 . 2 5 , 1 . 5 , 2 \}$ , which increase full-grid error beyond FT. The lower row estimates PT-relative path divergence along each weight’s own reverse trajectories, using 64 paths per design rather than rescaling the FT map; this discretized estimate excludes the final Tweedie step. The best weight was selected retrospectively in this single-seed comparison, and improvement is not uniform across designs. These results suggest a sampling-time control for over-adaptation.

Table 8: Fine-tuning on the analytic design-misspecification toy. Posterior errors are measured against the exact true posterior; likelihood RMSE is evaluated in correctly specified and misspecified regions of the design space. The improvement factor is the PT error divided by the FT error, so values above 1 favor fine-tuning and values below 1 indicate degradation.
<table><tr><td>Metric</td><td>PT</td><td>FT</td><td>Improvement (PT/FT)</td></tr><tr><td>Posterior  $W _ { 2 } \downarrow$ </td><td>0.09</td><td>0.07</td><td>1.35×</td></tr><tr><td>Posterior mean error ↓</td><td>0.08</td><td>0.03</td><td>2.88×</td></tr><tr><td>Correct-region likelihood RMSE↓</td><td>0.05</td><td>0.26</td><td>0.18×</td></tr><tr><td>Misspecified-region likelihood RMSE↓</td><td>0.45</td><td>0.30</td><td>1.49×</td></tr></table>

Table 9: Interpretation of design-localized fine-tuning. Path divergence $\mathcal { A } ( \theta , \xi )$ measures departure from the PT model at fixed θ, while predictive error change is evaluated against observations at the same design ξ.
<table><tr><td>A(θ, ξ)</td><td>Error improves</td><td>Error does not improve</td></tr><tr><td>Small</td><td>Adequate simulator or mild correction</td><td>Under-correction</td></tr><tr><td>Large</td><td>Corrected misspecification</td><td>Over-correction or forgetting</td></tr></table>

Table 9 relates the magnitude of adaptation to its predictive benefit at each design. Large path divergence with reduced error is consistent with misspecification correction, whereas large divergence without improvement suggests unnecessary adaptation or forgetting. Small divergence with persistent error indicates under-correction, while small divergence where predictions are already accurate suggests that little correction is needed.

## G.10 BMP MISSPECIFICATION EVALUATION

LSR cost groups. The LSR ensemble was generated by fitting the expert simulator to the same BMP observations across four cell lines in 6,000 bounded least-squares runs with different random initializations. Each run minimized squared prediction error and yielded a candidate parameter set, providing alternative fits to the same data. We used the 4,816 fits retained in the original filtered bank provided by Klumpe et al. (2022). Each fit contains 60 biophysical parameters and five receptor multiplicative factors, giving 65 features. We transformed these features to model-normalized coordinates, z-scored each feature across fits, and computed PCA (Figure 13). To group solutions by parameter structure and fit quality, we applied k-means with nine clusters to standardized PC1, PC2, and $\log _ { 1 0 } C ,$ , where C is the total LSR fitting cost, multiplying the standardized cost coordinate by eight. We ordered clusters by their mean log-cost and merged clusters 1–3, 4–6, and 7–9 into low (L), middle (M), and high (H) cost groups, respectively. These labels summarize global fit quality, not performance at every design. For the local group assignments in Figures 6 and 14, we computed MSE across member fits at each design and selected the group with the lowest value. Thus, a globally higher-cost group can still fit a particular experimental condition best, as explicitly shown in Figure 14A.

![](images/d291c946b28daff6ddb6b439e1545782b38e9dfb952f7e3ed960e2f359382a8a.jpg)  
Figure 11: Regularization sweep on the joint-RBF misspecification toy. Columns increase λ from $1 0 ^ { - 4 }$ to 10. (Top) FT absolute prediction error $e _ { \mathrm { F T } } ( \xi )$ , with full-grid MAE annotated. (Bottom) Path divergence $\mathcal A ( \xi )$ . Stars mark observed designs; dashed contours enclose $h ( \xi ) \geq 0 . 1$ . Color scales are shared within rows. The sweep used one training seed and 16 Monte Carlo draws per design to estimate response means.

![](images/e20072fbc6c9e9cc33bd2b3a0b1252227fc4d8509905b9b81a6d3ddb8068faa0.jpg)  
Figure 12: PT–FT score blending on the misspecification toy. Columns vary the sampling weight w from PT (0) to FT (1), without retraining. (Top) Absolute error of the predictive mean against the noise-free response, with full-grid MAE annotated. (Bottom) Estimated path divergence from PT. Stars mark observed designs; color scales are shared within rows. Intermediate weights reduce average error while limiting departure from PT.

Ligand-competition and titration conditions. The three response curves in Figure 6 probe competition for receptors and dose-dependent signaling. The ligand-competition axis transitions from high BMP4 and low BMP10 to high BMP10 and low BMP4. The NMuMG response changes along this axis, whereas the BMPR2-knockdown line remains close to zero. The BMP4 titration in NMuMG provides a dose–response comparison that also includes a decrease in observed signal at the highest concentration. We held the mechanistic parameters at θ<sup>LSR</sup> and sampled the FT likelihood, isolating response correction from changes in inferred parameters. The FT model improves over the simulator at this reference throughout the displayed BMPR2-knockdown series and at lower BMP4 doses, while the NMuMG competition series and higher-dose titration retain regions where LSR predicts better. These conditional-likelihood comparisons are distinct from the aggregate HBDS posterior-predictive benchmark in Table 3.

Design-level adaptation. We relate the design-conditional path divergence in Eq. (2) to the LSR cost groups defined above. Panel A of Figure 14 shows which group has the lowest MSE at each experimental condition. Although the low-cost group dominates, other groups fit better at particular ligand conditions and cell lines. These preferences describe relative fit among groups of parameter estimates, rather than establishing simulator misspecification on their own.

Comparing panels A and B reveals a qualitative correspondence between some departures from the dominant LSR group and elevated path divergence (θ, ξ). Panels C and D then distinguish adaptation from predictive improvement by comparing FT responses with the selected LSR anchor and the PT bootstrap median, respectively. Fine-tuning improves over the PT model across many conditions, but its comparison with LSR remains mixed. Nevertheless, improvements over LSR occur both at conditions favoring alternative parameter groups and at some conditions where the dominant group remains preferred. The green-boxed examples connect these design-level diagnostics to the response curves in Figure 6.

Together, these results provide a proof of concept for design-localized likelihood correction where fine-tuning can improve predictions beyond the selected mechanistic fit on particular conditions, while path divergence identifies where the likelihood changes. The remaining regions where LSR performs better also show that the current fine-tuning procedure does not fully exploit this potential, or, can be improved. This motivates the design-adaptive curricula and divergence-aware regularization discussed in §§ G.3 and G.4, targeting residual error while preserving successful corrections. Establishing whether these extensions yield broader gains requires further evaluation, which we leave to future work.

![](images/e1c346707ddee34c237579dd939916bea36f9fb7421c013143ab2792ed1bbbd0.jpg)

![](images/8bab7f6e166632e87cc2cb6b90b09611520033f24f3cdb5451bfce02653d54c7.jpg)  
Figure 13: LSR solution structure and cost groups across 4,816 retained fits. (Left) PCA coordinates colored by $\log _ { 1 0 }$ total LSR cost. (Right) The same projection colored by low, middle, and high cost groups, containing 1,296, 2,245, and 1,275 fits. PC1 and PC2 explain 8.9% and 5.3% of the standardized feature variance. Grouping also uses fitting cost, so overlap in this two-dimensional projection is expected.

Posterior transfer without collapse. Although likelihood fine-tuning is anchored at a fixed reference $\theta ^ { \mathrm { L S R } }$ , the resulting posterior does not collapse onto this point estimate. Figure 15 shows that FT posterior marginals generally shift toward the reference while retaining substantial spread, indicating that the likelihood-side correction transfers to posterior queries without reducing inference to the fine-tuning anchor.

![](images/a95e011f4cd65a0c20cf3a119f7fe88b6781af1ebde105de0ca9098111d2368e.jpg)  
Figure 14: Design-localized adaptation in the BMP model across four cell lines (rows) and experimental conditions (columns). Thin black lines separate titration series; the thick black line separates ligand-competition (“Rim”) from gradient series (“Gradient”). Green boxes mark the conditions illustrated in Figure 6. (A) Best-fitting LSR parameter group; colors identify groups, not error magnitudes. (B) Path divergence (θ, ξ), with yellow indicating greater adaptation. (C) Fine-tuning distance improvement relative to the selected LSR anchor. (D) Fine-tuning distance improvement relative to the PT bootstrap median. In C and D, blue indicates reduced distance to observations and red indicates increased distance relative to the comparator.

![](images/fd4f7a1f902945b0ccac6cd6a85e74bb2cd36ac35797973d10f717e1fde62c89.jpg)  
Figure 15: BMP posterior alignment with the LSR reference. Purple and green show PT and FT marginals; red lines mark the fitted LSR reference. The upper and lower blocks show 30 binding affinities K and 30 phosphorylation efficiencies ε, with ligand rows and receptor-complex columns. Type-I receptors label the column groups above; type-II receptors label individual columns beneath them, identifying each receptor pair. FT marginals generally move closer to the reference while retaining spread rather than collapsing to it.

## H AI USE STATEMENT

AI tools assisted with grammar, organization, TikZ figures, and experiment management. The authors take responsibility for the final content.