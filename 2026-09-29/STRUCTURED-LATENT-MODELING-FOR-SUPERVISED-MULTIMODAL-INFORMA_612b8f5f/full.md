# STRUCTURED LATENT MODELING FOR SUPERVISED MULTIMODAL INFORMATION DECOMPOSITION

Wanting Huang   
Department of Computer Science University of Iowa   
wanting-huang@uiowa.edu Weiran Wang   
Department of Computer Science University of Iowa   
weiran-wang@uiowa.edu   
Sanvesh Srivastava   
Department of Statistics   
University of Iowa   
sanvesh-srivastava@uiowa.edu

## ABSTRACT

Multimodal prediction relies on diverse forms of evidence: information repeated across modalities, cues specific to a single source, and complex cross-modal dependencies that emerge only when inputs are considered together. While recent methods promote richer interactions, they lack a principled way to isolate these target-relative contributions within learned continuous representations. We introduce a framework that applies contrastive or masked objectives at intermediate layers, coupled with source-wise invertible normalizing flows and a supervised, low-rank latent variable model. This architecture explicitly factorizes the joint distribution into shared task-relevant variation, modality-specific predictive variation, and task-irrelevant dependence. Drawing connections to prior multimodal learning assumptions, our approach evaluates how modalities independently and jointly contribute to the target. Ultimately, this framework unites intermediate representation learning with structured likelihood-based guidance, offering a practical latent-variable lens for characterizing continuous multimodal interactions. Empirically, we demonstrate the effectiveness of our approach across diverse multimodal benchmarks, showing robust improvements in predictive performance.

## 1 INTRODUCTION

Supervised multimodal fusion aims to learn representations that reflect the heterogeneity and interconnections between modalities (Liang et al., 2024a), capturing both overlapping (redundant) and modality-specific (unique) information to maximize predictive performance. Traditionally, multi view learning relies on the redundancy assumption—presuming all task-relevant information is shared across modalities (Tosh et al., 2021; Federici et al., 2020). However, this often fails in realworld supervised settings where modalities carry unique predictive signals. When applied to complex multimodal data, standard late fusion architectures frequently suffer from unimodal bias (also known as modality competition or imbalance (Huang et al., 2022; Zhang et al., 2024b)). During joint training, the fusion network tends to disproportionately rely on a dominant, easily accessible modality, inadvertently suppressing the learning of weaker modalities, and occasionally resulting in worse performance than their single-modality counterparts. A prominent example is visual question answering, where the vision modality is often ignored because the text already correlates strongly with the answer (Goyal et al., 2017; Cadene et al., 2019). By analyzing the learning dynamics of deep fusion networks, recent studies Zhang et al. (2024b); Kontras et al. (2025) suggest that inter modality correlation is the fundamental driver of this unimodal bias.

Recent approaches have attempted to address these challenges by quantifying or isolating multi modal interactions. Unsupervised representation learning methods (Liang et al., 2023b; Dufumier et al., 2025; Wen et al., 2026) rely on handcrafted, label-preserving perturbations and contrastive learning to extract either a single representation per view, or a fused representation that captures both shared and unique predictive information. Meanwhile, supervised approaches use labels to filter out task-irrelevant information and mitigate modality competition through mechanisms such as gradient balancing, prototype-based modality rebalancing, and game-theoretic regularization (Wang et al., 2020; Fan et al., 2023; Kontras et al., 2025). However, they do not learn explicitly disentangled latent variables. These developments motivate the important research question: how can a structured model of the source–target distribution support representation learning and mitigate unimodal bias? Such a model must explicitly disentangle private and shared predictive information, accommodating nonlinear features while distinguishing source–target relationships from other cross-source associations.

To address these limitations, we propose a supervised latent variable model (LVM) framework centered on the Supervised Structured Multimodal Latent Variable Model (S2MLVM). This novel formulation generalizes both unsupervised (Bach & Jordan, 2005; Lock et al., 2013) and existing supervised (Palzer et al., 2022) LVMs. Its key design divides each view’s latent space into distinct factors: a shared predictive component, a unique predictive component, and a shared nuisance (nonpredictive) component. Marginalizing these factors yields a low-rank-plus-diagonal covariance, imposing a compact dependence structure that accommodates shared source variation without requiring all of it to be directly predictive. Explicitly separating shared and private predictive signals from nuisance variations, our framework enables classifiers to optimally weight each component. Just as single-view disentanglement improves interpretability and sample efficiency (Bengio et al., 2013; Locatello et al., 2020), this multimodal factorization directly empowers supervised fusion, boosting generalization and mitigating unimodal bias. In summary, our main contributions are:

• We integrate the structured S2MLVM covariance model into representation learning, using neural encoders as initial feature extractors coupled with dimension-preserving normalizing flows (Dinh et al., 2017; Papamakarios et al., 2021). To accommodate high-dimensional, nonlinear multimodal data and optimize the statistical model jointly with the representations, we develop two instantiations (S2MLVM-MASK and S2MLVM-CONTRAST). These optimize a three-term objective combining a task loss, the joint Flow–S2MLVM density criterion, and an auxiliary representation objective to maintain input information while reducing dimensionality. Both maintain modality-specific encoding without early crossmodal fusion: S2MLVM-MASK reconstructs masked clean features within each source, whereas S2MLVM-CONTRAST contrasts augmented multimodal observations.

• We theoretically analyze our framework’s connection to common assumptions in multimodal learning. By drawing connections to Partial Information Decomposition (PID, Williams & Beer, 2010; Bertschinger et al., 2014), we elucidate the mathematical mechanisms—such as cooperative nuisance suppression and V-structure exploitation—that generate synergistic information under our LVM.

• We demonstrate the efficacy of S2MLVM across synthetic and real-world benchmarks. In controlled settings, it accurately recovers task-relevant subspaces. Across diverse real-world datasets, both instantiations mitigate unimodal bias and outperform strong representation-learning baselines in two- and three-modality scenarios.

## 2 METHOD

We develop S2MLVM, a supervised multimodal learning framework that integrates representation learning with structured probabilistic modeling. Each modality is encoded via a source-specific pathway and transformed by an invertible normalizing flow. The resulting representations are jointly modeled with the target using a structured latent variable model that explicitly disentangles shared predictive, modality-specific predictive, and target-irrelevant variation. This joint model provides likelihood-based generative guidance during training. While the framework naturally extends to an arbitrary number of modalities (by incorporating additional source pathways and block-structured LVM components), we present the formulation for two sources $x ^ { ( 1 ) }$ and $x ^ { ( 2 ) }$ with target y for clarity.

## 2.1 STRUCTURED LATENT VARIABLE MODELING

Consider paired observations $\mathcal { D } = \{ ( x _ { i } ^ { ( 1 ) } , x _ { i } ^ { ( 2 ) } , y _ { i } ) \} _ { i = 1 } ^ { N }$ . Each source $x _ { i } ^ { ( m ) }$ is mapped to a continuous representation through a source-specific encoder $g _ { \theta _ { m } }$ followed by an invertible flow $f _ { \phi _ { m } }$

$$
h ^ { ( m ) } = g _ { \theta _ { m } } ( x ^ { ( m ) } ) \in \mathbb { R } ^ { d _ { m } } , \qquad u ^ { ( m ) } = f _ { \phi _ { m } } ( h ^ { ( m ) } ) \in \mathbb { R } ^ { d _ { m } } , \quad m \in \{ 1 , 2 \} ,\tag{1}
$$

and these representations do not depend on the target $y .$ . We also extract continuous embedding $\tilde { y } \in \mathbb { R } ^ { d _ { y } }$ of the target y if it is categorical.

Our framework, the Supervised Structured Multimodal Latent Variable Model (S2MLVM), defines a joint generative process over the concatenated features $w = [ u ^ { ( 1 ) } ; u ^ { ( 2 ) } ; \tilde { y } ] \in \mathbb { R } ^ { d }$ , where $d =$ $d _ { 1 } + d _ { 2 } + d _ { y }$ . We model w using four independent latent factors $\dot { z } = [ z ^ { 1 2 } ; z ^ { 1 } ; z ^ { 2 } ; z ^ { c } ]$ of dimension $K = k _ { 1 2 } + \check { k } _ { 1 } + k _ { 2 } + k _ { c }$ . The generative model $w = A z + \epsilon$ follows the structured linear mapping:

$$
\begin{array} { r l } { \left[ { \boldsymbol u } ^ { ( 1 ) } \right] = \left[ \begin{array} { c c c c } { \Lambda _ { 1 } ^ { 1 2 } } & { \Lambda _ { 1 } ^ { 1 } } & { 0 } & { \Lambda _ { 1 } ^ { c } } \\ { \Lambda _ { 2 } ^ { 1 2 } } & { 0 } & { \Lambda _ { 2 } ^ { 2 } } & { \Lambda _ { 2 } ^ { c } } \\ { B ^ { 1 2 } } & { B ^ { 1 } } & { B ^ { 2 } } & { 0 } \end{array} \right] \left[ \begin{array} { c } { z ^ { 1 2 } } \\ { z ^ { 1 } } \\ { z ^ { 2 } } \\ { z ^ { c } } \end{array} \right] + \left[ \begin{array} { c } { \epsilon _ { 1 } } \\ { \epsilon _ { 2 } } \\ { \epsilon _ { y } } \end{array} \right] , \quad } & { z \sim \mathcal { N } ( 0 , I _ { K } ) , } \end{array}\tag{2}
$$

where $\Lambda _ { m } ^ { 1 2 } , \Lambda _ { m } ^ { m }$ , and $\Lambda _ { m } ^ { c }$ represent the shared predictive, modality-specific predictive, and targetirrelevant components, respectively, for modality $m$ . Loading matrices $B ^ { 1 2 }$ and $B ^ { m }$ characterize the effects of shared and modality m-specific latent variables on y˜. The observation noise $\epsilon = [ \epsilon _ { 1 } ; \epsilon _ { 2 } ; \epsilon _ { y } ]$ is drawn from $\mathcal { N } ( 0 , \Psi )$ , with a positive diagonal covariance $\mathbf { \Psi } \Psi = \mathrm { d i a g } ( \Psi _ { 1 } , \Psi _ { 2 } , \Psi _ { y } )$

This block-structured loading pattern explicitly disentangles the latent space into distinct semantic components. The factor $z ^ { \mathbf { \breve { 1 } 2 } }$ captures shared predictive information that influences both sources and the target, while $z ^ { 1 }$ and $z ^ { 2 }$ isolate unique predictive information specific to each source. Crucially, the target $\tilde { y }$ does not depend on the shared nuisance factor $z ^ { c }$ . This structure ensures that task-irrelevant cross-modal correlations are absorbed by $z ^ { c }$ rather than confounding the predictive representations. Marginalizing over the latent factors gives

$$
\begin{array} { r } { \boldsymbol { w } \sim \mathcal { N } ( \boldsymbol { 0 } , \boldsymbol { \Omega } ) , \qquad \boldsymbol { \Omega } = \boldsymbol { A } \boldsymbol { A } ^ { \top } + \boldsymbol { \Psi } . } \end{array}\tag{3}
$$

This formulation provides a unified perspective on multi-view latent variable models. In the unsupervised setting (where the target $y$ is absent), it is impossible to distinguish predictive shared variation $( z ^ { 1 2 } )$ from nuisance shared variation $( z ^ { c } ) ,$ , so they collapse into a single shared factor. Under this restriction, the model reduces to probabilistic Canonical Correlation Analysis (CCA) (Bach & Jordan, 2005) if we retain only this shared factor, and to Joint and Individual Variation Explained (JIVE) (Lock et al., 2013) if we also include the unique factors $z ^ { 1 }$ and $z ^ { 2 }$ . Conversely, the full supervised structure generalizes recent predictive LVMs like supervised JIVE (Palzer et al., 2022), which lack the nuisance factor $z ^ { c }$ . By introducing $z ^ { c } .$ , S2MLVM explicitly separates shared predictive variation from task-irrelevant cross-modal correlations.

## 2.2 FLOW-BASED DISTRIBUTION MODELING

To apply the Gaussian S2MLVM to complex multimodal data, we must bridge the gap between the structured Gaussian base distribution and the highly non-Gaussian neural representations $h ^ { ( m ) }$ We achieve this using source-wise normalizing flows $f _ { \phi _ { m } }$ , which provide invertible, dimensionpreserving transformations with tractable Jacobian determinants (Dinh et al., 2017; Papamakarios et al., 2021). Although each flow processes a single modality independently, their outputs $u ^ { ( m ) }$ are modeled jointly with the target $\tilde { y }$ under the S2MLVM base distribution to retain all source– source and source–target dependencies. This unified construction enables a joint maximum likelihood framework: we can perform maximum likelihood estimation of the structured loading matrix $A$ and residual covariance Ψ while simultaneously learning the flow parameters $\phi = \left( \phi _ { 1 } , \phi _ { 2 } \right)$ to map the initial encoder outputs into Gaussian-distributed variables.

For fixed initial encoders, the joint log-density of this feature-space model is given by

$$
\log p _ { \phi , A , \Psi } ( h ^ { ( 1 ) } , h ^ { ( 2 ) } , \tilde { y } ) = \log \mathcal { N } ( w ; 0 , \Omega ) + \sum _ { m = 1 } ^ { 2 } \log \left| \operatorname* { d e t } J _ { f _ { \phi _ { m } } } ( h ^ { ( m ) } ) \right| ,\tag{4}
$$

where $\Omega = A A ^ { \top } + \Psi$ . The corresponding fitting objective for an observation is the negative loglikelihood, $\mathcal { L } _ { \mathrm { d e n s i t y } } = - \log p _ { \phi , A , \Psi } ( h ^ { ( 1 ) } , h ^ { ( 2 ) } , \tilde { y } )$ . The Gaussian term fits the structured joint relationships, while the Jacobian terms account for volume changes induced by the flows.

While finite-capacity flows practically optimize the Kullback-Leibler divergence between the empirical distribution and the Gaussian LVM, normalizing flows theoretically possess universal approximation capabilities for continuous probability distributions (Papamakarios et al., 2021). Because our flows operate independently on each modality, this provides a strong asymptotic guarantee that, given sufficient capacity, they can perfectly transform the source representations to match the Gaussian marginals required by the joint LVM.

## 2.3 AUXILIARY REPRESENTATION LEARNING

Because the initial encoders perform dimension reduction on the raw inputs, an auxiliary objective is necessary to prevent representation degeneracy before the representations enter the structured LVM. Although the subsequent normalizing flows are invertible, this auxiliary loss ensures the encoders extract and retain sufficient information. The objective $\mathcal { L } _ { \mathrm { r e p } }$ is computed directly on the encoder representations $h ^ { ( m ) }$ and is instantiated by either masked reconstruction $( \mathcal { L } _ { \mathrm { r e c } } )$ or contrastive learning $( \mathcal { L } _ { \mathrm { c o n } } )$ . Only one of these alternatives is used in each variant; they are not added together.

Masked reconstruction. Following recent masked representation learning methods (He et al., 2022; Baevski et al., 2022), let $H ^ { ( m ) }$ denote clean features at an intermediate layer, ${ \widehat { H } } ^ { ( m ) }$ the reconstructed features, and $M _ { m }$ the set of masked indices for modality m. We define

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { 2 } \sum _ { m = 1 } ^ { 2 } \ell \Bigl ( \widehat { H } ^ { ( m ) } [ M _ { m } ] , \mathrm { s g } ( H ^ { ( m ) } [ M _ { m } ] ) \Bigr ) ,\tag{5}
$$

where ℓ is a distance metric (e.g., Smooth L1 loss) averaged over the masked entries, and sg stops gradients through the target argument. The objective encourages recovery of source features from partial observations.

Contrastive learning. Let v and $v ^ { + }$ be normalized projections of the representations $h ^ { ( m ) }$ extracted from two independently augmented versions of the same modality. The InfoNCE objective (Oord et al., 2018; Chen et al., 2020) for this observation is:

$$
\mathcal { L } _ { \mathrm { c o n } } = - \log \frac { \exp ( v ^ { \top } v ^ { + } / \tau ) } { \exp ( v ^ { \top } v ^ { + } / \tau ) + \sum _ { v ^ { - } \in \mathcal { N } ( v ) } \exp ( v ^ { \top } v ^ { - } / \tau ) } ,\tag{6}
$$

where $\tau > 0$ is the temperature, and $\mathcal { N } ( v )$ contains negative samples from other observations. In practice, this loss is symmetrized across both anchor directions. Crucially, positive pairs are formed from the same joint observation rather than across modalities, leaving the extraction of cross-modal shared information to the S2MLVM.

## 2.4 TOTAL REPRESENTATION LEARNING OBJECTIVE

Because our framework explicitly isolates predictive components, we construct the final representation by computing and concatenating the posterior means of the predictive latent factors $( z ^ { 1 \bar { 2 } } , z ^ { 1 }$ , and $z ^ { 2 } )$ given $( \bar { u ^ { ( 1 ) } } , \bar { u ^ { ( 2 ) } } )$ ) from the S2MLVM. To provide direct discriminative supervision and accommodate standard downstream evaluations, this representation is passed to a multilayer perceptron (MLP) optimized with a standard task loss $\mathcal { L } _ { \mathrm { M L P } } \ ( \mathrm { e . g . }$ , cross-entropy for classification or mean squared error for regression) against the ground-truth targets.

The overall training objective minimizes a weighted combination of this discriminative task loss, the generative Flow–S2MLVM maximum-likelihood loss $\scriptstyle ( { \mathcal { L } } _ { \mathrm { d e n s i t y } } )$ defined in Section 2.2, and the auxiliary representation loss $( \mathcal { L } _ { \mathrm { r e p } } )$ defined in Section 2.3:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { M L P } } + \lambda _ { \mathrm { d e n s i t y } } \mathcal { L } _ { \mathrm { d e n s i t y } } + \lambda _ { \mathrm { r e p } } \mathcal { L } _ { \mathrm { r e p } } .\tag{7}
$$

Optimization. In practice, the total loss $\mathcal { L }$ is averaged over the training set and minimized using stochastic gradient descent (SGD). All parameters—including source encoders, normalizing flows, structured covariance parameters, and the final MLP classifier—are optimized jointly end-to-end.

## 3 CONNECTIONS TO OTHER MULTIMODAL LEARNING FRAMEWORKS

To understand how our model captures complex multimodal interactions, we analyze it through the lens of Partial Information Decomposition (PID) (Williams & Beer, 2010; Liang et al., 2024b), a framework that decomposes joint predictive information into unique, redundant, and synergistic components. Our S2MLVM connects deeply to this formalism. PID achieves this decomposition by defining a constraint space $\Delta _ { P }$ of adversarial joint distributions Q that preserve the true sourcetarget marginals:

$$
\Delta _ { \cal P } = \{ Q \mid Q ( u ^ { ( 1 ) } , \tilde { y } ) = { \cal P } ( u ^ { ( 1 ) } , \tilde { y } ) , Q ( u ^ { ( 2 ) } , \tilde { y } ) = { \cal P } ( u ^ { ( 2 ) } , \tilde { y } ) \} .\tag{8}
$$

While the latent factors in our model $( z ^ { 1 2 } , z ^ { 1 } , z ^ { 2 } , z ^ { c } )$ possess clear intuitive meanings, they do not map one-to-one onto the information-theoretic atoms of PID. Instead, as we detail in Appendix $\mathbf { B } ,$ the formal PID definition extracts synergy by identifying a worst-case adversary $q _ { M I } ^ { * }$ that minimizes the joint mutual information:

$$
q _ { M I } ^ { * } = \arg \operatorname* { m i n } _ { Q \in \Delta _ { P } } I _ { Q } ( u ^ { ( 1 ) } , u ^ { ( 2 ) } ; \tilde { y } ) .\tag{9}
$$

We show that this adversary is remarkably strong: it correlates the private noise across modalities to align with the combined task signal, maximally reducing the joint predictive information.

Furthermore, we demonstrate that the assumption of conditional independence $( \boldsymbol { u } ^ { ( 1 ) } \perp \perp \boldsymbol { u } ^ { ( 2 ) } \mid \tilde { \boldsymbol { y } } ) -$ a widespread premise in multimodal machine learning (Blum $\&$ Mitchell, 1998; Chaudhuri et al., 2009)—provides a tractable analytic proxy $q _ { C E } ^ { * }$ for the intractable PID optimization. This adversary is defined by maximizing the conditional entropy:

$$
q _ { C E } ^ { * } = \arg \operatorname* { m a x } _ { Q \in \Delta _ { P } } H _ { Q } ( u ^ { ( 1 ) } , u ^ { ( 2 ) } \mid \tilde { y } ) .\tag{10}
$$

Because our generative structure models the latent features as jointly Gaussian, their information quantities are fully determined by the joint covariance matrix. Consequently, the constraint $Q \in$ $\Delta _ { P }$ , which fixes the source-target marginals, restricts the adversary to exclusively manipulating the cross-source covariance block. This means the adversarial covariance matrix $\Sigma ^ { ( \dot { Q } ) }$ is identically the original covariance matrix $\Sigma$ plus an off-diagonal cross-source perturbation block C. $\mathbf { A } s$ detailed in Appendix B, the optimal conditional entropy adversary acts as a decoupling filter by injecting the exact negative conditional cross-covariance $\mathrm { \bar { \it C } } _ { C E } = \bar { - } \Sigma _ { u ^ { ( 1 ) } u ^ { ( 2 ) } | \tilde { y } }$ as this perturbation. Under our S2MLVM parameterization, this injected covariance takes a closed form:

$$
\begin{array} { r } { C _ { C E } = - \Big ( \underbrace { \Lambda _ { 1 } ^ { c } \big ( \Lambda _ { 2 } ^ { c } \big ) ^ { \top } } _ { \mathrm { s h a r e d n u i s a n c e } } + \underbrace { \Lambda _ { 1 } ^ { 1 2 } \big ( \Lambda _ { 2 } ^ { 1 2 } \big ) ^ { \top } } _ { \mathrm { s h a r e d s i g n a l } } - \underbrace { \sum _ { u ^ { ( 1 ) } \tilde { y } } \sum _ { \tilde { y } \tilde { y } } ^ { - 1 } \sum _ { \tilde { y } u ^ { ( 2 ) } } } _ { \mathrm { i n d u c e d V - s t u c t u r e } } \Big ) . } \end{array}\tag{11}
$$

This term-by-term algebraic decomposition reveals that all latent factors provide opportunities for synergistic predictive power. To eliminate synergy and achieve conditional independence, the adversary $C _ { C E }$ must systematically inject noise to cancel three distinct mechanisms: the cooperative suppression of the shared nuisance $z ^ { c } ,$ the noise-averaging of the redundant shared signal $\dot { z } ^ { 1 2 }$ , and the V-structure “explaining-away” correlation induced by the independent private signals $z ^ { 1 }$ and $z ^ { 2 }$

Because $q _ { C E } ^ { * }$ provides a closed-form but sub-optimal minimization of the joint mutual information $( I _ { q _ { M I } ^ { * } } \leq I _ { q _ { C E } ^ { * } } ) ,$ substituting it into the PID equations yields analytic bounds on the true optimal quantities. Specifically, our tractable proxy safely underestimates both Synergy $( S _ { C E } \leq { \bar { S _ { M I } } } )$ and Redundancy $( R _ { C E } \leq R _ { M I } )$ , while overestimating the Unique Information $( \breve { U _ { i } ^ { C E } } \geq \breve { U } _ { i } ^ { \breve { M } I } )$ . Finally, our structured model explicitly accommodates the spectrum of multimodal assumptions. Many classical multi-view frameworks rely on the multi-view redundancy assumption (Federici et al., 2020; Tsai et al., 2021; Tosh et al., 2021), positing that all task-relevant information is shared $( z ^ { 1 } , z ^ { 2 }$ are empty). By explicitly isolating the private predictive factors $( z ^ { 1 } , z ^ { 2 } )$ from the shared redundancy $( z ^ { 1 2 } )$ , S2MLVM provides a generalization that efficiently learns in both highly redundant and highly unique multimodal settings without discarding synergistic interactions.

## 4 RELATED WORK

Unsupervised multimodal representation learning. Multimodal representation learning traditionally relies on correspondence and partial observations to retain information across sources. Contrastive learning and mutual information-based methods (Oord et al., 2018; Belghazi et al., 2018;

Chen et al., 2020; Tian et al., 2020; Radford et al., 2021) learn transferable representations by aligning paired views. The Multi-View Information Bottleneck (Federici et al., 2020) formalizes this by providing an information-theoretic account for retaining shared predictive content under strict multi-view redundancy assumptions. To move beyond purely shared information, recent works explicitly isolate modality-specific or complementary content. Deep generative models (Wang et al., 2016; Lee & Pavlovic, 2021a; Palumbo et al., 2023; Zhang et al., 2026) separate shared and private variations through structured likelihood modeling. Parallel efforts in self-supervised learning, such as FactorCL (Liang et al., 2023b) and DisentangledSSL (Wang et al., 2025), achieve this factorization using complex augmentations and contrastive learning. Other approaches actively encourage complementary interactions: CoMM (Dufumier et al., 2025) models joint spaces to capture beyondredundancy interactions, while InfMasking (Wen et al., 2026) contrasts partially masked features against complete fusions to elicit synergy. While these self-supervised methods demonstrate the necessity of learning from more than just cross-modal agreement, they largely rely on complex ad-hoc regularizations or masking strategies. We generalize this principle of likelihood-based separation to the supervised setting, providing a rigorous statistical formalization for complex source–target dependencies.

Balancing supervised multimodal optimization. In joint multimodal training, models often dis proportionately rely on a single dominant modality (Peng et al., 2022; Huang et al., 2022), a phenomenon formally linked to inter-modality correlations (Zhang et al., 2024b). To mitigate this bias, a prominent line of work directly intervenes in the optimization process. Techniques like OGM (Peng et al., 2022), AGM (Li et al., 2023), and MLB (Kontras et al., 2024) dynamically scale gradient updates based on ongoing modality contributions, while scheduling methods like MMPareto (Wei & Hu, 2024), MLA (Zhang et al., 2024a), and ReconBoost (Hua et al., 2024) explicitly reconcile conflicting unimodal and multimodal objectives. More recently, methods use game-theoretic valuation (Wei et al., 2024) or mutual-information decomposition (Kontras et al., 2025) to formalize modality cooperation. We share the motivation of using task-related dependence to guide learning; however, S2MLVM achieves this balance inherently via its generative objective, bypassing the need for explicit gradient modulation.

Target-relative information decomposition. Partial information decomposition (PID) isolates the unique, redundant, and synergistic information provided by multiple sources about a target (Williams & Beer, 2010; Bertschinger et al., 2014). Recent works have extended these concepts to multimodal learning, developing scalable estimators for quantifying interactions (Liang et al., 2023a) and deriving learning guarantee (Liang et al., 2024a). While these works focus on estimating information-theoretic quantities from datasets or fixed representations, evaluating PID on continuous, high-dimensional neural representations presents significant computational challenges. Existing continuous estimators rely on variational optimization (Pakman et al., 2021), neural estimation (Kleinman et al., 2021), convex approximations for Gaussians (Venkatesh & Schamberg, 2022; Venkatesh et al., 2023), or invertible transformations to latent Gaussian spaces (Zhao et al., 2025). We build on this latent-Gaussian principle rather than proposing a new PID definition. By explicitly disentangling shared and modality-specific predictive factors, our framework naturally yields the covariance structures needed to analyze synergistic information.

## 5 EXPERIMENTS

We adopt the experimental setups, including datasets and base architectures, of recent related works. For all experiments, we tune hyperparameters and perform ablation studies exclusively on the validation set. We select the best-performing model configuration on the validation set and evaluate it task MLP predictions to report the final metrics on the held-out test set. We report accuracy for classification tasks and mean squared error (MSE) for regression tasks. Our repeated-run results use 5 seeds and are summarized by the mean and sample standard deviation. We focus on comparing with the current state-of-the-art unsupervised (e.g., COMM (Dufumier et al., 2025), InfMasking (Wen et al., 2026)) and supervised (MCR (Kontras et al., 2025)) methods using their configurations (data, architecture, and implementation); results taken from these prior works are annotated with <sup>∗</sup>.

Table 1: Recovery of S2MLVM subspaces. Here dim $= d _ { 1 } = d _ { 2 }$ and $k = k _ { 1 2 } = k _ { 1 } = k _ { 2 } = k _ { c } .$
<table><tr><td colspan="2">Setting</td><td>Canonical correlation (CC) ↑</td></tr><tr><td rowspan="4"> $k = 4 , \sigma = 0 . 4$ </td><td> $d i m = 1 2$ </td><td> $0 . 9 9 4 7 9 \pm 0 . 0 0 3 1 5$ </td></tr><tr><td> $d i m = 2 4$ </td><td> $0 . 9 8 8 1 6 \pm 0 . 0 0 3 9 6$ </td></tr><tr><td> $d i m = 4 8$ </td><td> $0 . 9 4 8 6 7 \pm 0 . 0 3 0 7 3$ </td></tr><tr><td> $d i m = 9 6$ </td><td> $0 . 9 3 1 0 4 \pm 0 . 0 3 4 7 7$ </td></tr><tr><td rowspan="4"> $d i m = 2 4 , k = 4$ </td><td> $\overline { { \sigma = 0 . 2 } }$ </td><td> $\overline { { 0 . 9 8 6 2 4 \pm 0 . 0 0 5 2 3 } }$ </td></tr><tr><td> $\sigma = 0 . 4$ </td><td> $0 . 9 8 8 1 6 \pm 0 . 0 0 3 9 6$ </td></tr><tr><td> $\sigma = 0 . 6$ </td><td> $0 . 9 9 1 0 9 \pm 0 . 0 0 2 0 6$ </td></tr><tr><td> $\sigma = 0 . 8$ </td><td> $0 . 9 9 1 3 7 \pm 0 . 0 0 1 5 5$ </td></tr><tr><td rowspan="4"> $\overline { { d i m = 4 8 , \sigma = 0 . 4 } }$ </td><td> $\overline { { k = 4 } }$ </td><td> $\overline { { 0 . 9 4 8 6 7 \pm 0 . 0 3 0 7 3 } }$ </td></tr><tr><td> $k = 8$ </td><td> $0 . 9 6 4 8 8 \pm 0 . 0 1 1 1 8$ </td></tr><tr><td> $k = 1 2$ </td><td> $0 . 9 7 3 3 3 \pm 0 . 0 0 5 5 9$ </td></tr><tr><td> $k = 1 6$ </td><td> $0 . 9 6 5 0 2 \pm 0 . 0 0 8 0 4$ </td></tr></table>

## 5.1 SUBSPACE RECOVERY ON SYNTHETIC DATA

We first verify that maximum likelihood modeling via SGD recovers the true task-relevant latent structure. We generate two observed sources and the response from independent Gaussian latent blocks $( z ^ { 1 2 } , \ { \overline { { z } } } ^ { 1 } , \ z ^ { 2 } .$ , and $z ^ { c } )$ following the S2MLVM loading pattern in equation 2. We use isotropic observation noise with covariance $\Psi \ : = \ : \sigma ^ { 2 } I$ . we then apply fixed random nonlinear flow transformations to both observed sources (without the nonlinear mappings, the recovery would be perfect across settings). For the experiments, we first vary the ambient source dimension over $d _ { 1 } \doteq d _ { 2 } \in \{ 1 2 , 2 4 , 4 8 , 9 6 \}$ while fixing $k = k _ { 1 2 } = k _ { 1 } = k _ { 2 } = k _ { c } = 4$ and $\sigma = 0 . 4$ Second, we vary the observation-noise standard deviation over $\sigma \in \{ 0 . 2 , 0 . 4 , 0 . 6 , 0 . 8 \}$ while fixing $d _ { 1 } = d _ { 2 } = 2 4$ and $k \ = \ 4$ Then, we vary the dimensionality of each latent block over $k \in \{ 4 , 8 , 1 2 , 1 6 \}$ while fixing $d _ { 1 } = d _ { 2 } = 4 8$ and $\sigma = 0 . 4$ . Models are trained on 10,000 samples for 12,000 steps using SGD (momentum 0.9, batch size 512, learning rate $3 \times 1 0 ^ { - 3 } )$ . The checkpoint achieving the lowest negative log-likelihood on a 2,000-sample validation set is evaluated on 4,096 held-out test samples. Results are averaged over five random seeds.

Because individual loading coordinates are only identifiable up to rotations within each latent block, we evaluate recovery at the subspace level using a rotation-invariant metric:

$$
\mathrm { C C } = \frac { 1 } { k _ { 1 2 } + k _ { 1 } + k _ { 2 } } \sum _ { i = 1 } ^ { k _ { 1 2 } + k _ { 1 } + k _ { 2 } } \sigma _ { i } \mathopen { } \mathclose \bgroup \left( Q ^ { \top } \widehat { Q } \aftergroup \egroup \right) .\tag{12}
$$

Here, $Q$ and $\widehat { Q }$ denote orthonormal bases for the true and estimated task-relevant subspaces, respectively, while $\sigma _ { i }$ indicates the i-th singular value. Mean canonical correlation (CC) measures subspace alignment through their principal angles (larger is better), and is bounded in $[ 0 , 1 ]$ . Table 1 demonstrates that S2MLVM consistently recovers the task-relevant subspace under varying conditions. While performance degrades smoothly as the estimation problem becomes more challenging—such as with higher ambient or latent dimensions—the recovered predictive geometry remains robust, maintaining high canonical correlation across all settings.

## 5.2 EXPERIMENTS ON TRIFEATURE

We use TriFeatures (Dufumier et al., 2025) as a controlled diagnostic to examine whether the latent variables learned by our model exhibit the intended functional specialization. By providing explicit control over target labels, TriFeatures enables us to systematically evaluate the extraction of shared, source-specific, and cross-source information. Each input image comprises three categorical attributes—shape, texture, and color—which are used to construct four targeted diagnostic tasks. Redundancy $( R )$ targets the shape attribute shared across both images. The two unique tasks, $U _ { 1 }$ and $U _ { 2 } .$ , correspond to the independent texture attributes of the first and second images, respectively. These three attribute-level tasks are evaluated as 10-way classification problems. Finally, Synergy (S) is a binary classification task evaluating a cross-source relation determined jointly by the texture of the first image and the color of the second. Crucially, neither attribute alone is sufficient to predict the target; the synergistic relation can only be resolved by integrating information contributed by both sources. Following InfMasking (Wen et al., 2026), we train on 10,000 synthetic image pairs and evaluate on 4,096 held-out pairs. For the encoders of both S2MLVM-MASK and S2MLVM-CONTRAST, each view uses an AlexNet frontend and an independent width-512 Transformer with its own CLS token. We train both architectures for 100 epochs. The representation dimension is 512 for $u ^ { ( 1 ) } , u ^ { ( 2 ) }$ , and y˜. The latent dimensions are $k _ { 1 2 } = k _ { 1 } = k _ { 2 } = k _ { c } = 8 .$

Table 2: Classification accuracy (%) on TriFeature, for different task labels $R / U _ { 1 } / U _ { 2 } / S .$
<table><tr><td>Method</td><td>R-ACC ↑</td><td> $U _ { \mathrm { 1 ^ { - } } } \mathrm { A C C \uparrow }$ </td><td> $U _ { \mathrm { 2 ^ { - } } } \mathrm { A C C \uparrow }$ </td><td>U-ACC↑</td><td>S-ACC ↑</td></tr><tr><td>FactorCL</td><td> $9 9 . 8 ^ { * }$ </td><td></td><td></td><td> $6 2 . 5 ^ { * }$ </td><td> $4 6 . 5 ^ { * }$ </td></tr><tr><td>CoMM</td><td> $9 9 . 9 \pm 0 . 1 ^ { \ast }$ </td><td> $8 4 . 4 \pm 2 . 4 ^ { * }$ </td><td> $9 1 . 2 \pm 1 . 0 ^ { * }$ </td><td> $8 6 . 8 \pm 3 . 0 ^ { * }$ </td><td> $7 1 . 4 \pm 3 . 5 ^ { * }$ </td></tr><tr><td>InfMasking</td><td> $9 9 . 9 \pm 0 . 1 ^ { \ast }$ </td><td> $9 0 . 7 \pm 2 . 1 ^ { \ast }$ </td><td> $9 1 . 4 \pm 3 . 0 ^ { * }$ </td><td> $9 0 . 6 \pm 2 . 3 ^ { \ast }$ </td><td> $7 7 . 0 \pm 4 . 2 ^ { * }$ </td></tr><tr><td>MCR</td><td> $9 9 . 9 \pm 0 . 1 $ </td><td> $9 2 . 1 \pm 1 . 4$ </td><td> $9 3 . 2 \pm 0 . 9$ </td><td> $9 2 . 7 \pm 1 . 2$ </td><td> $9 0 . 2 \pm 3 . 9$ </td></tr><tr><td>S2MLVM-MASK</td><td> $9 3 . 4 \pm 1 . 1$ </td><td> $9 1 . 3 \pm 2 . 2$ </td><td> $8 4 . 9 \pm 3 . 5$ </td><td> $8 8 . 1 \pm 2 . 7$ </td><td> $8 5 . 4 \pm 4 . 3$ </td></tr><tr><td>S2MLVM-CONTRAST</td><td> ${ \bf 9 9 . 9 \pm 0 . 1 }$ </td><td> ${ \bf 9 5 . 1 \pm 0 . 7 }$ </td><td> ${ \bf 9 5 . 3 \pm 0 . 8 }$ </td><td> ${ \bf 9 5 . 2 \pm 0 . 5 }$ </td><td> ${ \bf 9 2 . 3 \pm 0 . 9 }$ </td></tr><tr><td> $z ^ { 1 2 }$ </td><td> $9 3 . 6 \pm 3 . 3$ </td><td> $3 4 . 2 \pm 7 . 1$ </td><td> $3 2 . 7 \pm 1 1 . 8$ </td><td> $3 3 . 5 \pm 9 . 5$ </td><td> $5 6 . 2 \pm 2 . 6$ </td></tr><tr><td> $z ^ { 1 }$ </td><td> $7 0 . 4 \pm 4 . 3$ </td><td> $8 3 . 5 \pm 9 . 2$ </td><td> $1 4 . 1 \pm 7 . 5$ </td><td> $4 8 . 8 \pm 8 . 4$ </td><td> $4 8 . 6 \pm 1 . 6$ </td></tr><tr><td> $z ^ { 2 }$ </td><td> $6 4 . 3 \pm 9 . 5$ </td><td> $1 0 . 6 \pm 5 . 2$ </td><td> $7 8 . 4 \pm 1 4 . 0$ </td><td> $4 4 . 5 \pm 9 . 6$ </td><td> $4 9 . 8 \pm 1 . 7$ </td></tr></table>

Table 3: Test results on two modality benchmarks, by unsupervised, supervised, and our methods.
<table><tr><td>Method</td><td>V&amp;T EE↓</td><td>MIMIC ↑</td><td>CREMA-D ↑</td><td>UCF101 ↑</td><td>MOSEI↑</td></tr><tr><td>FactorCL</td><td> $1 0 . 8 \pm 0 . 6 ^ { * }$ </td><td> $6 7 . 3 \pm 0 . 0 ^ { * }$ </td><td> $6 1 . 1 \pm 1 . 3$ </td><td> $5 0 . 3 \pm 2 . 2$ </td><td> $7 8 . 5 \pm 0 . 1$ </td></tr><tr><td>CoMM</td><td> $8 . 0 \pm 2 . 1 ^ { * }$ </td><td> $6 6 . 4 \pm 0 . 4 ^ { * }$ </td><td> $5 9 . 9 \pm 2 . 5$ </td><td> $5 1 . 0 \pm 2 . 1$ </td><td> $7 9 . 7 \pm 0 . 3$ </td></tr><tr><td>InfMasking</td><td> $4 . 2 \pm 0 . 4 ^ { * }$ </td><td> $6 8 . 1 \pm 0 . 4 ^ { * }$ </td><td> $6 3 . 9 \pm 2 . 7$ </td><td> $5 1 . 3 \pm 1 . 9$ </td><td> $8 0 . 4 \pm 0 . 4$ </td></tr><tr><td>Unimodals</td><td> $\overline { { \mathrm { I } : 1 . 9 \pm 0 . 0 } }$ </td><td> $\overline { { \mathrm { S } : 5 6 . 3 \pm 0 . 5 } }$ </td><td> $\overline { { \mathrm { V } : 5 5 . 6 \pm 3 . 4 } }$ </td><td> $\overline { { \mathrm { V } : 4 2 . 7 \pm 1 . 2 } }$ </td><td> $\overline { { \mathrm { V } : 6 5 . 2 \pm 0 . 2 } }$ </td></tr><tr><td>Joint Training</td><td> $\mathrm { F } : 8 7 . 7 \pm 0 . 5$   $1 . 9 \pm 0 . 1$ </td><td> $\mathrm { T } : 6 8 . 0 \pm 0 . 4$   $6 6 . 8 \pm 1 . 4$ </td><td> $\mathrm { A } : 5 6 . 5 \pm 3 . 7$   $6 2 . 6 \pm 5 . 8 ^ { * }$ </td><td> $\mathrm { A } : 2 8 . 8 \pm 1 . 7$   $4 7 . 7 \pm 1 . 5 ^ { * }$ </td><td> $\mathrm { T } : 7 8 . 4 \pm 1 . 3$   $8 0 . 5 \pm 0 . 2 ^ { * }$ </td></tr><tr><td>MCR</td><td> $2 . 0 \pm 0 . 1 8$ </td><td> $6 8 . 6 \pm 0 . 8$ </td><td> $7 6 . 1 \pm 1 . 6 ^ { * }$ </td><td> $5 5 . 2 \pm 1 . 8 ^ { \ast }$ </td><td> $\mathbf { 8 0 . 8 \pm 0 . 4 ^ { * } }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>S2MLVM</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>-MASK</td><td> ${ \bf 1 . 6 \pm 0 . 1 }$ </td><td> ${ \bf 6 9 . 1 \pm 0 . 4 }$ </td><td> $7 4 . 7 \pm 1 . 7$ </td><td> $5 7 . 0 \pm 1 . 8$ </td><td> $8 0 . 5 \pm 0 . 8$ </td></tr><tr><td>-CONTRAST</td><td> $2 . 6 \pm 0 . 2$ </td><td> $6 8 . 7 \pm 0 . 5$ </td><td> ${ \bf 7 6 . 2 \pm 2 . 0 }$ </td><td> ${ \bf 5 7 . 2 \pm 2 . 0 }$ </td><td> $8 0 . 3 \pm 0 . 6$ </td></tr></table>

We provide the classification accuracy on test set of several methods in Table 2. In general, supervised methods (MCR and S2MLVM) outperform unsupervised methods (e.g., COMM and InfMasking), due to joint training of predictors and predictive representations, and contrastive learning turns out to be a better representation learning objective for this dataset. To see that our latent factors well extracts the desired predictive information, after model training (with task loss on $( z ^ { 1 2 } , z ^ { 1 } , z ^ { 2 } ) )$ , we train a small MLP on each extracted latent factor by S2MLVM-CONTRAST and obtain 3 additional accuracies. As expected, For the tasks R, $U _ { 1 }$ and ${ \mathrm { { \bar { U } } _ { 2 } } } ,$ performing classification only on $z ^ { 1 2 } , z ^ { 1 } , z ^ { 2 }$ correspondingly maintains most accuracy. For the synergy task S however, each individual factor is not sufficient to maintain desired predictive information.

## 5.3 REAL-WORLD MULTIMODAL BENCHMARKS

Datasets and tasks. We evaluate two groups of real-world benchmarks. The first follows the InfMasking setup (Wen et al., 2026) on MultiBench (Liang et al., 2021), covering Vision&Touch end-effector (V&T EE) prediction, MIMIC diagnostic-group prediction, MOSI sentiment classification, UR-FUNNY humor recognition, and MUSTARD sarcasm recognition. The second follows the MCR configurations for CREMA-D, UCF101, and MOSEI (Kontras et al., 2025). Complete dataset definitions and evaluation conventions are provided in Appendix A.1; dataset-specific model architetures are summarized in Appendix A.2.

Table 4: Test results on two modality benchmarks MOSI (MO), UR-FUNNY (UR), MUSTARD (MU), by unsupervised, supervised, and our methods.  
Table 5: Architecture ablation on MOSI (MO), UR-FUNNY (UR), MUSTARD (MU) validation sets.
<table><tr><td>Method</td><td>MO↑</td><td>UR↑</td><td>MU↑</td></tr><tr><td>FactorCL</td><td> $5 1 . 2 \pm 1 . 6 ^ { * }$ </td><td> $6 0 . 5 \pm 0 . 8 ^ { * }$ </td><td> $5 5 . 8 \pm 0 . 9 ^ { \ast }$ </td></tr><tr><td>CoMM</td><td> $6 3 . 7 \pm 2 . 5 ^ { * }$ </td><td> $6 3 . 3 \pm { 0 . 5 ^ { * } }$ </td><td> $6 4 . 4 \pm 1 . 1 ^ { * }$ </td></tr><tr><td>InfMasking</td><td> $6 9 . 0 \pm 1 . 2 ^ { \ast }$ </td><td> $6 4 . 3 \pm 0 . 9 ^ { * }$ </td><td> $6 6 . 8 \pm 2 . 5 ^ { \ast }$ </td></tr><tr><td>Unimodals</td><td> $\overline { { \mathrm { V } : 5 5 . 1 \pm 1 . 3 } }$   $\mathrm { T } : 7 2 . 0 \pm 1 . 4$ </td><td> $\overline { { 5 2 . 4 \pm 0 . 5 } }$   $6 1 . 7 \pm 0 . 7$ </td><td> $\overline { { 5 7 . 7 \pm 2 . 2 } }$   $6 3 . 4 \pm 2 . 0$ </td></tr><tr><td>Joint Training</td><td> $7 2 . 9 \pm 1 . 6$ </td><td> $6 1 . 7 \pm 1 . 0$ </td><td> $6 0 . 6 \pm 2 . 0$ </td></tr><tr><td>MCR</td><td> $7 5 . 0 \pm 1 . 2$ </td><td> $5 4 . 4 \pm 2 . 1$ </td><td> $6 0 . 5 \pm 3 . 5$ </td></tr><tr><td>S2MLVM</td><td></td><td></td><td></td></tr><tr><td>-MASK</td><td> ${ \bf 7 5 . 8 \pm 0 . 8 }$ </td><td> $6 3 . 7 \pm 2 . 0$ </td><td> ${ \bf 6 8 . 6 \pm 5 . 5 }$ </td></tr><tr><td>-CONTRAST</td><td> $7 4 . 3 \pm 1 . 9$ </td><td> ${ \bf 6 5 . 0 \pm 1 . 8 }$ </td><td> $6 6 . 0 \pm 2 . 9$ </td></tr></table>

<table><tr><td>S2MLVM</td><td colspan="2">MO↑UR↑MU↑</td></tr><tr><td> $- \mathbf { M A S K }$  w/o Flow</td><td>73.6 68.7</td><td>63.8 63.8 59.6 48.6 57.3</td></tr><tr><td> $\mathbf { w } / \mathbf { o } ~ z ^ { c }$  -CONTRAST</td><td>64.6 76.2</td><td>62.0 63.6 59.4</td></tr><tr><td>w/o Flow</td><td>70.2</td><td>61.4 55.8</td></tr><tr><td> $\mathbf { w } / \mathbf { o } ~ z ^ { c }$ </td><td>72.9</td><td>58.9 52.9</td></tr></table>

Table 6: Test accuracies (%) on three-modality benchmarks.
<table><tr><td>Method</td><td>UR-FUNNY↑</td><td>V&amp;T Contact ↑</td><td>MOSEI↑</td></tr><tr><td>CoMM</td><td> $6 4 . 8 \pm 1 . 1 ^ { \ast }$ </td><td> $9 4 . 1 \pm 0 . 2 ^ { * }$ </td><td> $7 9 . 0 \pm 0 . 3$ </td></tr><tr><td>InfMasking</td><td> $6 5 . 6 \pm 1 . 2 ^ { * }$ </td><td> $9 4 . 1 \pm 0 . 1 ^ { \ast }$ </td><td> $7 8 . 9 \pm 0 . 8$ </td></tr><tr><td>MCR</td><td> $6 6 . 6 \pm 1 . 5$ </td><td> $9 4 . 1 \pm 0 . 3$ </td><td> $8 1 . 1 \pm 0 . 4 ^ { * }$ </td></tr><tr><td>S2MLVM-MASK</td><td> $6 6 . 6 \pm 1 . 8$ </td><td> $9 4 . 0 \pm 0 . 3$ </td><td> $8 0 . 7 \pm 0 . 7$ </td></tr><tr><td> $\mathrm { S 2 M L V M  – C O N T R A S T }$ </td><td> $6 6 . 0 \pm 0 . 9$ </td><td> $9 3 . 6 \pm 0 . 9$ </td><td> $8 1 . 4 \pm 0 . 5$ </td></tr></table>

Experiments with Two Input Modalities. Tables 3 and 4 summarize the performance on the two-modality benchmarks. We observe that the optimal auxiliary objective varies depending on the dataset: S2MLVM-MASK excels on V&T EE, MIMIC, MOSI, and MUSTARD, whereas S2MLVM-CONTRAST achieves the best performance on UR-FUNNY, CREMA-D, and UCF101. Across all tasks, S2MLVM consistently outperforms previous unsupervised and supervised methods, indicating better utilization of all decomposed predictive information. Optimal hyperparameters and sensitivity analysis are discussed in Appendix (cf. Table 7 and Table 8).

Ablation studies. We conduct two sets of ablation studies to examine the contributions of the proposed model components and training objectives. As shown in Table 5, removing the invertible transformations (w/o Flow) or the separate nuisance component $( \mathbf { w } / \mathbf { o } z ^ { c } )$ significantly degrades performance across all tasks. In the appendix, we ablate the loss terms in Table 9. The results confirms that both auxiliary representation objective and likelihood are indispensable for effectively capturing predictive information, validating our structural design.

Experiments with Three Input Modalities. We extend our LVM with one more modality, and consider UR-FUNNY with visual, textual, and acoustic features; V&T contact prediction with im age, force, and proprioceptive observations; and MOSEI with visual, textual, and acoustic features. As shown in Table 6, S2MLVM consistently outperforms baselines, demonstrating that it can scale to multiple modalities while preserving strong predictive performance.

## 6 CONCLUSION

We introduced S2MLVM, a structured supervised latent variable model that tackles modality competition in multimodal fusion. By disentangling representations into shared predictive, modality specific predictive, and task-irrelevant nuisance factors, our framework isolates and better utilizes diverse multimodal evidence. Integrating this structured likelihood model with representation learning—via dimension-preserving flows—bridges the gap between classic statistical models and high-dimensional deep networks. Empirically, S2MLVM mitigates unimodal bias, outperforming strong baselines across diverse real-world benchmarks, and scales seamlessly to more than two modalities. Ultimately, our approach offers a unified latent-variable lens for characterizing complex continuous multimodal interactions. Given the inherent complexity of synergy, cleanly extracting all PID components into completely separate latent variables remains an open question. Future work will explore extending our framework to handle partially missing modalities (Wu & Goodman, 2018; Lee & Pavlovic, 2021b; Lee & van der Schaar, 2021).

## AI USE STATEMENT

Parts of this manuscript were drafted with assistance from AI. All AI-assisted content was subsequently reviewed, verified, and edited by the authors.

## REPRODUCIBILITY STATEMENT

We provide the information needed throughout the main paper and appendix. The main paper describes the proposed model, learning objectives, and common experimental protocol. The appendix provides additional details on dataset preprocessing and evaluation protocols, source-specific architectures, hyperparameter configurations, and experimental settings.

## REFERENCES

Francis R. Bach and Michael I. Jordan. A probabilistic interpretation of canonical correlation analysis. Technical Report 688, Dept. of Statistics, University of California, Berkeley, 2005.

Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. data2vec: A general framework for self-supervised learning in speech, vision and language. In ICML, 2022.

Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajeshwar, Sherjil Ozair, Yoshua Bengio, Aaron Courville, and Devon Hjelm. Mutual information neural estimation. In ICML, 2018.

Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE Trans. Pattern Analysis and Machine Intelligence, 35(8):1798–1828, 2013.

Nils Bertschinger, Johannes Rauh, Eckehard Olbrich, Jurgen Jost, and Nihat Ay. Quantifying unique ¨ information. Entropy, 16(4):2161–2183, 2014.

Avrim Blum and Tom Mitchell. Combining labeled and unlabeled data with co-training. In Proceedings ofthe eleventh annual conference on Computational learning theory, pp. 92–100, 1998.

Remi Cadene, Corentin Dancette, Hedi Ben-younes, Matthieu Cord, and Devi Parikh. RUBi: Reducing unimodal biases for visual question answering. In NeurIPS, 2019.

Kamalika Chaudhuri, Sham M. Kakade, Karen Livescu, and Karthik Sridharan. Multi-view clustering via canonical correlation analysis. In ICML, 2009.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.

Thomas M. Cover and Joy A. Thomas. Elements ofInformation Theory. second edition, 2006.

Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using Real NVP. In ICLR, 2017.

Benoit Dufumier, Javiera Castillo Navarro, Devis Tuia, and Jean-Philippe Thiran. What to align in multimodal contrastive learning? In ICLR, 2025. URL https://openreview.net/ forum?id=Pe3AxLq6Wf.

Yunfeng Fan, Wenchao Xu, Haozhao Wang, Junxiao Wang, and Song Guo. Pmr: Prototypical modal rebalance for multimodal learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2023.

Marco Federici, Anjan Dutta, Patrick Forre, Nate Kushman, and Zeynep Akata. Learning robust´ representations via multi-view information bottleneck. In ICLR, 2020.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll’ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16000–16009, 2022.

Cong Hua, Qianqian Xu, Shilong Bao, Zhiyong Yang, and Qingming Huang. ReconBoost: Boosting can achieve modality reconcilement. In ICML, 2024.

Yu Huang, Junyang Lin, Chang Zhou, Hongxia Yang, and Longbo Huang. Modality competition: What makes joint training of multi-modal network fail in deep learning?(provably). In International conference on machine learning, pp. 9226–9259. PMLR, 2022.

Michael Kleinman, Alessandro Achille, Stefano Soatto, and Jonathan C Kao. Redundant information neural estimation. Entropy, 23(7):922, 2021.

Konstantinos Kontras, Christos Chatzichristos, Matthew Blaschko, and Maarten De Vos. Improving multimodal learning with multi-loss gradient modulation. In BMVC, 2024.

Konstantinos Kontras, Thomas Strypsteen, Christos Chatzichristos, Paul Pu Liang, Matthew B. Blaschko, and Maarten De Vos. Balancing multimodal training through game-theoretic regularization. In NeurIPS, 2025. URL https://openreview.net/forum?id=auiURbhoYx.

Changhee Lee and Mihaela van der Schaar. A variational information bottleneck approach to multiomics data integration. In AISTATS, 2021.

Mihee Lee and Vladimir Pavlovic. Private-shared disentangled multimodal vae for learning of latent representations. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 1692–1700. IEEE, 2021a.

Mihee Lee and Vladimir Pavlovic. Private-shared disentangled multimodal VAE for learning of latent representations. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2021b.

Hong Li, Xingyu Li, Pengbo Hu, Yinuo Lei, Chunxiao Li, and Yi Zhou. Boosting multi-modal model performance with adaptive gradient modulation. In ICCV, 2023.

Paul Pu Liang, Yiwei Lyu, Xiang Fan, Zetian Wu, Yun Cheng, Jason Wu, Leslie Chen, Peter Wu, Michelle A. Lee, Yuke Zhu, Ruslan Salakhutdinov, and Louis-Philippe Morency. Multibench: Multiscale benchmarks for multimodal representation learning, 2021. URL https://arxiv. org/abs/2107.07502.

Paul Pu Liang, Yun Cheng, Xiang Fan, Chun Kai Ling, Suzanne Nie, Richard Chen, Zihao Deng, Nicholas Allen, Randy Auerbach, Faisal Mahmood, Ruslan Salakhutdinov, and Louis-Philippe Morency. Quantifying & modeling multimodal interactions: An information decomposition framework. In NeurIPS, 2023a.

Paul Pu Liang, Zihao Deng, Martin Q. Ma, James Zou, Louis-Philippe Morency, and Russ Salakhutdinov. Factorized contrastive learning: Going beyond multi-view redundancy. In NeurIPS, 2023b.

Paul Pu Liang, Chun Kai Ling, Yun Cheng, Alex Obolenskiy, Yudong Liu, Rohan Pandey, Alex Wilf, Louis-Philippe Morency, and Ruslan Salakhutdinov. Multimodal learning without labeled multimodal data: Guarantees and applications. In ICLR, 2024a.

Paul Pu Liang, Amir Zadeh, and Louis-Philippe Morency. Foundations & trends in multimodal machine learning: Principles, challenges, and open questions. ACM Comput. Surv., 56(10), 2024b.

Francesco Locatello, Ben Poole, Gunnar Ratsch, Bernhard Sch¨ olkopf, Olivier Bachem, and Michael¨ Tschannen. Weakly-supervised disentanglement without compromises. In ICML, 2020.

Eric F. Lock, Katherine A. Hoadley, J. S. Marron, and Andrew B. Nobel. Joint and individual variation explained (jive) for integrated analysis of multiple data types. Annals of Applied Statistics, 7(1):523–542, 2013.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Ari Pakman, Amin Nejatbakhsh, Dar Gilboa, Abdullah Makkeh, Luca Mazzucato, Michael Wibral, and Elad Schneidman. Estimating the unique information of continuous variables. In NeurIPS, 2021.

Emanuele Palumbo, Imant Daunhawer, and Julia E. Vogt. MMVAE+: Enhancing the generative quality of multimodal VAEs without compromises. In ICLR, 2023.

Elise F. Palzer, Christine H. Wendt, Russell P. Bowler, Craig P. Hersh, Sandra E. Safo, and Eric F. Lock. sJIVE: Supervised joint and individual variation explained. Computational Statistics & Data Analysis, 175:107547, 2022.

George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57):1–64, 2021.

Xiaokang Peng, Yake Wei, Andong Deng, Dong Wang, and Di Hu. Balanced multimodal learning via on-the-fly gradient modulation. In CVPR, 2022.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In ECCV, 2020.

Christopher Tosh, Akshay Krishnamurthy, and Daniel Hsu. Contrastive learning, multi-view redundancy, and linear models. In Algorithmic Learning Theory, 2021.

Yao-Hung Hubert Tsai, Yue Wu, Ruslan Salakhutdinov, and Louis-Philippe Morency. Selfsupervised learning from a multi-view perspective. In ICLR, 2021.

Praveen Venkatesh and Gabriel Schamberg. Partial information decomposition via deficiency for multivariate gaussians. In IEEE International Symposium on Information Theory (ISIT), 2022. doi: 10.1109/ISIT50566.2022.9834649.

Praveen Venkatesh, Corbett Bennett, Sam Gale, Tamina K. Ramirez, Greggory Heller, Severine Durand, Shawn Olsen, and Stefan Mihalas. Gaussian partial information decomposition: Bias correction and application to high-dimensional data. In NeurIPS, 2023.

Chenyu Wang, Sharut Gupta, Xinyi Zhang, Sana Tonekaboni, Stefanie Jegelka, Tommi Jaakkola, and Caroline Uhler. An information criterion for controlled disentanglement of multimodal data. In ICLR, 2025.

Weiran Wang, Xinchen Yan, Honglak Lee, and Karen Livescu. Deep variational canonical correlation analysis. arXiv:1610.03454, 2016.

Weiyao Wang, Du Tran, and Matt Feiszli. What makes training multi-modal classification networks hard? In CVPR, 2020.

Yake Wei and Di Hu. MMPareto: Boosting multimodal learning with innocent unimodal assistance. In ICML, 2024.

Yake Wei, Ruoxuan Feng, Zihe Wang, and Di Hu. Enhancing multimodal cooperation via samplelevel modality valuation. In CVPR, 2024.

Liangjian Wen, Qun Dai, Jianzhuang Liu, Jiangtao Zheng, Yong Dai, Dongkai Wang, Zhao Kang, Jun Wang, Zenglin Xu, and Jiang Duan. Infmasking: Unleashing synergistic information by contrastive multimodal interactions. Advances in Neural Information Processing Systems, 38: 15529–15555, 2026.

Paul L. Williams and Randall D. Beer. Nonnegative decomposition of multivariate information. arXiv preprint arXiv:1004.2515, 2010.

Mike Wu and Noah Goodman. Multimodal generative models for scalable weakly-supervised learning. In NeurIPS, 2018.

Xiaohui Zhang, Jaehong Yoon, Mohit Bansal, and Huaxiu Yao. Multimodal representation learning by alternating unimodal adaptation. In CVPR, 2024a.

Yedi Zhang, Peter E. Latham, and Andrew Saxe. Understanding unimodal bias in multimodal deep linear networks. In ICML, 2024b.

Yijie Zhang, Yiyang Shen, and Weiran Wang. Disentanglement of variations with multimodal generative modeling. In ICLR, 2026.

Wenyuan Zhao, Adithya Balachandran, Chao Tian, and Paul Pu Liang. Partial information decomposition via normalizing flows in latent gaussian distributions. In NeurIPS, 2025. URL https://openreview.net/forum?id=X13jOIhnog.

## A APPENDIX

## A.1 DATASETS

We evaluate the proposed framework across affective computing, multimodal language understanding, robotics, and audio–visual recognition benchmarks.

MultiBench. MOSI contains 2,199 opinion segments paired with visual, acoustic, and textual observations. In the InfMasking-style evaluation, we use the two-modality setting and evaluate the representation with binary sentiment classification.

MOSEI extends this setting to approximately 23,000 monologue clips and is evaluated under both the two-modality supervised protocol inherited from MCR and the visual–acoustic–textual threemodality setting.

UR-FUNNY contains 16,514 examples from 1,866 TED-talk videos and evaluates humor recognition. We use both its two-modality configuration and its visual–acoustic–textual configuration for the three-modality experiments.

MUSTARD contains 690 balanced dialogue utterances and evaluates sarcasm recognition (Liang et al., 2021; Wen et al., 2026; Kontras et al., 2025).

Robotics benchmark. Vision&Touch contains 150 robotic manipulation trajectories with 1,000 time steps per trajectory. We consider two tasks. End-effector prediction is a regression task evaluated from the two-modality representation, whereas contact prediction is evaluated as a three-modality classification task using image, force, and proprioceptive observations. We report $\mathrm { M S E } \times 1 0 ^ { - 4 }$ for end-effector prediction and classification accuracy for contact prediction.

Audio–visual benchmarks. CREMA-D evaluates six-way emotion recognition from synchronized face video and speech produced by 91 actors.

UCF101 is used for audio–visual action recognition; following the MCR protocol, we retain the subset of action categories for which both modalities are available. These benchmarks follow the supervised evaluation protocol used by MCR (Kontras et al., 2025).

MIMIC combines a 24-hour sequence of clinical measurements with static patient descriptors and defines a binary diagnostic-group prediction task. It contains 53,423 admissions from 38,597 patients. We retain MIMIC in the comparison where required to reproduce the benchmark coverage and averages reported by prior multimodal interaction methods.

## A.2 ARCHITECTURE

As illustrated in Figure 1,the overall architecture consists of dataset-specific neural encoders, normalizing flows, and the S2MLVM latent variable model. Each source pathway consists of a datasetspecific backbone and input adapter, followed by an independent Transformer with its own learnable CLS token. The clean, unprojected CLS representations are passed separately to their respective flows, without cross-modal attention or target inputs in the source encoders. We evaluate two model variants, S2MLVM-MASK and S2MLVM-CONTRAST, which share this identical architecture and differ only in their auxiliary representation loss. To make downstream predictions, the concatenated posterior means of the predictive latent blocks $( z ^ { 1 2 } , z ^ { 1 }$ , and z<sup>2</sup>) are extracted from the LVM and passed through a task MLP comprising two linear layers with an intervening ReLU activation, where the supervised task loss is applied.

However, specific details are handled differently for each dataset. For MOSI, UR-FUNNY, MUS-TARD, and MOSEI, we use separate temporal Transformers with modality-specific input projections to encode the precomputed visual and textual feature sequences. For MIMIC, an MLP encodes the static patient descriptors, while a GRU encodes the multivariate clinical time series. For Vision&Touch end-effector regression, we use a ResNet-18 image backbone and a temporal force encoder as separate source pathways. The three-modality contact-prediction configuration additionally includes a dedicated proprioception encoder. For CREMA-D and UCF101, we adopt the ResNet-

![](images/5820271d4bcb0591fa30186e3e4609673713dfd26ae88974632bf6af56f079fb.jpg)  
Figure 1: Overview architecture.

Table 7: Selected hyperparameter for each dataset.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">S2MLVM-MASK</td><td colspan="2">S2MLVM-CONTRAST</td><td rowspan="2">dim k</td><td rowspan="2">loss (λdensity, λrep)</td></tr><tr><td>Mask ratio</td><td>Batch size</td><td>Temperature</td><td>Batch size</td></tr><tr><td>MOSI</td><td>0.55</td><td>96</td><td>0.10</td><td>32</td><td>4</td><td>(1,1)</td></tr><tr><td>MIMIC</td><td>0.60</td><td>96</td><td>0.20</td><td>48</td><td>12</td><td>(1,1)</td></tr><tr><td>UR-FUNNY</td><td>0.30</td><td>64</td><td>0.10</td><td>128</td><td>4</td><td>(1,1)</td></tr><tr><td>MUSTARD</td><td>0.45</td><td>64</td><td>0.10</td><td>128</td><td>4</td><td>(1,1)</td></tr><tr><td>CREMA-D</td><td>0.55</td><td>32</td><td>0.15</td><td>8</td><td>16</td><td>(0.2,0.8)</td></tr><tr><td>UCF101</td><td>0.55</td><td>16</td><td>0.05</td><td>16</td><td>12</td><td>(0.2,1)</td></tr><tr><td>MOSEI</td><td>0.85</td><td>64</td><td>0.05</td><td>32</td><td>12</td><td>(0.5,0.8)</td></tr></table>

18 backbone configuration of MCR (Kontras et al., 2025), with independent visual and acoustic encoders operating on video frames and audio spectrograms, respectively.

Besides, the unimodal and joint-training baselines are reproduced following MCR (Kontras et al., 2025). Unimodal training optimizes each modality independently with the supervised task loss, whereas joint training optimizes all modalities together using a single supervised task objective.

## A.3 HYPERPARAMETER TUNING

The final comparison uses validation-selected hyperparameters, summarized in Table 7. For S2MLVM-MASK, these are the mask ratio and batch size; for S2MLVM-CONTRAST, they are the InfoNCE temperature and batch size. Here, mask ratio m ∈ [0, 1] denotes the fraction of input feature tokens randomly masked for the masked-reconstruction objective, temperature τ is the temperature parameter that scales the similarity logits before the softmax in the InfoNCE objective, and batch size B denotes the number of training examples in each minibatch. The validation sensitivity of these hyperparameters is reported in Table 8.

## A.4 LOSS ABLATION

We use a strict one-factor-at-a-time validation procedure. We conduct an ablation study on the loss coefficients to examine the contribution of each loss term, with the results reported in Table 9.

Table 8: Hyperparameter sensitivity on validation sets.
<table><tr><td>Study</td><td>Candidate</td><td>MOSI↑</td><td>MUSTARD ↑</td></tr><tr><td rowspan="7">Mask ratio (S2MLVM-MASK)</td><td> $m = 0 . 4 0$ </td><td>77.6</td><td>65.2</td></tr><tr><td> $m = 0 . 4 5$ </td><td>79.1</td><td>70.3</td></tr><tr><td> $m = 0 . 5 0$ </td><td>77.8</td><td>66.7</td></tr><tr><td> $m = 0 . 5 5$ </td><td>80.0</td><td>70.0</td></tr><tr><td> $m = 0 . 6 0$ </td><td>75.6</td><td>62.7</td></tr><tr><td> $m = 0 . 6 5$ </td><td>77.9</td><td>65.2</td></tr><tr><td> $m = 0 . 7 0$ </td><td>79.7 74.4</td><td>64.4 63.1</td></tr><tr><td rowspan="4">Temperature (S2MLVM-CONTRAST)</td><td> $m = 0 . 7 5$ </td><td></td><td></td></tr><tr><td> $\tau = 0 . 0 5$   $\tau = 0 . 1 0$ </td><td>77.7 79.9</td><td>64.9 69.2</td></tr><tr><td> $\tau = 0 . 1 5$ </td><td>79.0</td><td>68.7</td></tr><tr><td> $\tau = 0 . 2 0$ </td><td>79.2</td><td>67.1</td></tr><tr><td>Batch size  $( \mathrm { S } 2 \mathrm { M L V M - M A S K } )$ </td><td> $B = 3 2$   $B = 4 8$   $B = 6 4$   $B = 9 6$   $B = 1 2 8$ </td><td>75.5 78.9 79.1 79.6 77.0</td><td>68.0 65.8 70.2 64.7 64.6</td></tr><tr><td>Batch size (S2MLVM-CONTRAST)</td><td> $B = 3 2$   $B = 4 8$   $B = 6 4$   $B = 9 6$   $B = 1 2 8$ </td><td>82.7 80.8 79.0 79.0 78.6</td><td>64.9 68.8 67.7 65.9 72.2</td></tr></table>

Table 9: Loss-coefficient ablations on validation datasets.
<table><tr><td>Variant</td><td>MOSI↑</td><td>MUSTARD ↑</td></tr><tr><td rowspan="3">S2ML  $\mathbf { \nabla } \mathbf { \cdot } \mathbf { M } \mathbf { - } \mathbf { M } \mathbf { A } \mathbf { S } \mathbf { K } \left( \lambda _ { \mathrm { d e n s i t y } } = 1 , \lambda _ { \mathrm { r e p } } = 1 \right)$  w/o  $L _ { \mathrm { d e n s i t y } }$ </td><td>79.0</td><td>68.6</td></tr><tr><td>64.6</td><td>63.2</td></tr><tr><td>73.3</td><td>66.5</td></tr><tr><td rowspan="3">S2MLVM-CONTRAST  $( \lambda _ { \mathrm { d e n s i t y } } = 1 , \lambda _ { \mathrm { r e p } } = 1 )$  w/o  $L _ { \mathrm { d e n s i t y } }$ </td><td>79.9</td><td>66.4</td></tr><tr><td>75.4</td><td>64.2</td></tr><tr><td>77.5</td><td>62.8</td></tr></table>

## B CONNECTION TO PARTIAL INFORMATION DECOMPOSITION (PID)

In this section, we analyze the theoretical origins of predictive synergy within our proposed latent variable model. While the previous sections established the practical benefits of explicitly disentangling predictive and nuisance factors for supervised fusion, we now formalize how this structure captures synergistic interactions between modalities. We utilize the Partial Information Decomposition (PID) framework (Williams & Beer, 2010; Bertschinger et al., 2014) to isolate synergistic information, demonstrating how specific latent factors—such as shared nuisance variables and task-relevant signals—interact to provide predictive power that is only accessible when multiple modalities are observed jointly.

Specifically, PID decomposes the joint and marginal mutual informations into four non-negative components: the unique information provided by each individual view $\mathrm { ( U n i q u e _ { 1 } }$ and $\mathrm { U n i q u e } _ { 2 } )$ , the redundant information shared by both views (Redundancy), and the synergistic information that arises only from their combination (Synergy). These components satisfy the classic PID system of

equations (Williams & Beer, 2010):

$$
\begin{array} { r l } & { I _ { P } ( U ^ { 1 } , U ^ { 2 } ; Y ) = \mathrm { U n i q u e } _ { 1 } + \mathrm { U n i q u e } _ { 2 } + \mathrm { R e d u n d a n c y } + \mathrm { S y n e r g y } , } \\ & { \qquad I _ { P } ( U ^ { 1 } ; Y ) = \mathrm { U n i q u e } _ { 1 } + \mathrm { R e d u n d a n c y } , } \\ & { \qquad I _ { P } ( U ^ { 2 } ; Y ) = \mathrm { U n i q u e } _ { 2 } + \mathrm { R e d u n d a n c y } . } \end{array}
$$

Because this system is underdetermined, modern PID formulations define a constraint space $\Delta _ { P }$ of adversarial joint distributions $Q$ that preserve the pairwise source-target marginals of the true distribution $P \colon$

$$
\Delta _ { \cal P } = \left\{ Q ( U ^ { 1 } , U ^ { 2 } , Y ) ~ \big | ~ Q ( U ^ { 1 } , Y ) = { \cal P } ( U ^ { 1 } , Y ) , ~ Q ( U ^ { 2 } , Y ) = { \cal P } ( U ^ { 2 } , Y ) \right\} .
$$

By minimizing the joint information over $Q \in \Delta _ { P }$ , the framework isolates the synergistic interactions from the baseline predictive information. Following Bertschinger et al. (2014) and Liang et al. (2024a), this adversarial optimization yields formal definitions for the unique and synergistic information:

$$
\begin{array} { r l } & { \mathrm { U n i q u e } _ { j } = \displaystyle \operatorname* { m i n } _ { Q \in \Delta _ { P } } I _ { Q } ( U ^ { j } ; Y \mid U ^ { - j } ) , } \\ & { \mathrm { S y n e r g y } = I _ { P } ( U ^ { 1 } , U ^ { 2 } ; Y ) - \displaystyle \operatorname* { m i n } _ { Q \in \Delta _ { P } } I _ { Q } ( U ^ { 1 } , U ^ { 2 } ; Y ) , } \\ & { \mathrm { R e d u n d a n c y } = \displaystyle \operatorname* { m a x } _ { Q \in \Delta _ { P } } I _ { Q } ( U ^ { 1 } ; U ^ { 2 } ; Y ) = I _ { P } ( U ^ { 1 } ; Y ) + I _ { P } ( U ^ { 2 } ; Y ) - \displaystyle \operatorname* { m i n } _ { Q \in \Delta _ { P } } I _ { Q } ( U ^ { 1 } , U ^ { 2 } ; Y ) . } \end{array}
$$

## B.1 THE ADVERSARIAL LATENT FRAMEWORK

To provide a self-contained analysis of synergy, we briefly recall the generative structure of our latent variable model. We assume the multi-view observations $\check { U } ^ { 1 } , \bar { U } ^ { 2 }$ and the target $Y$ are generated from a set of mutually orthogonal Gaussian latent factors:<sup>1</sup>

$$
\begin{array} { r l } & { U ^ { 1 } = \Lambda _ { 1 } ^ { 1 2 } z ^ { 1 2 } + \Lambda _ { 1 } ^ { c } z ^ { c } + \Lambda _ { 1 } ^ { 1 } z ^ { 1 } + \epsilon ^ { 1 } } \\ & { U ^ { 2 } = \Lambda _ { 2 } ^ { 1 2 } z ^ { 1 2 } + \Lambda _ { 2 } ^ { c } z ^ { c } + \Lambda _ { 2 } ^ { 2 } z ^ { 2 } + \epsilon ^ { 2 } } \\ & { Y = B ^ { 1 2 } z ^ { 1 2 } + B ^ { 1 } z ^ { 1 } + B ^ { 2 } z ^ { 2 } + \epsilon ^ { Y } } \end{array}
$$

Here, $z ^ { 1 2 }$ is the task-relevant shared factor, $z ^ { c }$ is the task-irrelevant shared nuisance factor, $z ^ { 1 } , z ^ { 2 }$ are view-specific task-relevant private factors, and $\epsilon ^ { j } \sim \mathcal { N } ( \mathbf { 0 } , \Sigma _ { \epsilon ^ { j } } )$ is the full-rank independent private noise. We assume the nuisance loadings ${ \Lambda } _ { j } ^ { c }$ are orthogonal to the combined task-relevant loadings $\tilde { \Lambda } _ { j } = [ \Lambda _ { j } ^ { 1 2 } , \Lambda _ { j } ^ { j } ] \ ( \mathrm { i . e . , } \ ( \tilde { \Lambda } _ { j } ) ^ { \top } \Lambda _ { j } ^ { c } = { \bf 0 } )$ , and the task-relevant loading matrices $\tilde { \Lambda } _ { j }$ have full column rank.

For our Gaussian model, any adversarial distribution $Q \in \Delta _ { P }$ must preserve the true marginal predictive covariances $( \Sigma _ { U ^ { 1 } U ^ { 1 } } , \Sigma _ { U ^ { 2 } U ^ { 2 } } , \Sigma _ { U ^ { 1 } Y } , \Sigma _ { U ^ { 2 } Y } , \Sigma _ { Y Y } )$ . Because these marginals are fixed, the only parameter the adversary can manipulate is the cross-source covariance $\Sigma _ { U ^ { 1 } U ^ { 2 } }$ . This allows us to frame any valid adversary $q \in \Delta _ { P }$ as a pure noise-injection strategy: the adversary retains the true generative process $p$ (leaving the latent factors and marginal variances untouched), but injects a cross-covariance matrix $C$ directly between the independent private noises $\epsilon ^ { 1 } , \epsilon ^ { 2 }$ . This creates the adversarial joint covariance:

$$
\Sigma _ { U U } ^ { ( Q ) } = \Sigma _ { U U } ^ { ( p ) } + \left[ { \bf 0 } _ { U } \Sigma \right. \left. \Sigma \right] .
$$

Because the injected matrix C alters only the off-diagonal cross-source covariance, any choice of C that keeps the resulting $\Sigma _ { U U } ^ { ( Q ) }$ positive semi-definite guarantees $Q \in \Delta _ { P }$ . The question then becomes: what constitutes the ‘worst-case” cross-covariance C? This depends on the chosen adversarial objective.

## B.2 THE CONDITIONAL INDEPENDENCE ADVERSARY $( q _ { C E } ^ { * } )$

To formally capture the notion of decoupling the views, we first define the least-informative distribution using the Maximum Entropy (MaxEnt) principle applied to the sources. We seek the distribution $q _ { C E } ^ { * }$ that maximizes the conditional entropy of the sources $h _ { Q } ( U \mid Y )$ , which is equivalent to maximizing the determinant $| \Sigma _ { U | Y } ^ { ( Q ) } | ;$

$$
q _ { C E } ^ { * } = \arg \operatorname* { m a x } _ { Q } | \Sigma _ { U | Y } ^ { ( Q ) } | \quad \mathrm { s u b j e c t } \mathrm { t o } \quad \Sigma ^ { ( Q ) } \succeq 0 .
$$

By maximizing the residual uncertainty of the sources, the MaxEnt objective forces the observations to be as unstructured and independent as possible once the target $Y$ is known. This adversary implies that, conditioned on Y , we have independent $U ^ { 1 }$ and $U ^ { 2 }$

Lemma 1 (Least-Informative Coupling via Maximum Conditional Entropy). Suppose the true data distribution p is generated by the model. The true model contains synergy because the shared nuisance $f a c t o r \ z ^ { c }$ allows the views to cooperatively suppress shared noise, yielding a cleaner prediction of Y . Furthermore, because the independent task-relevant factors $( z ^ { { \mathrm { i } } \bar { 2 } } , z ^ { 1 } , z ^ { \bar { 2 } } )$ all independently cause Y, they form a classical V-structure. While the private factors $z ^ { 1 }$ and $z ^ { 2 }$ are unconditionally independent, conditioning on their shared effect $Y$ renders all taskfactors conditionally dependent via the “explaining away” effect. At the observation level, this structure induces a negative conditional correlation between the views, mathematically captured by $- \Sigma _ { U ^ { 1 } Y } \Sigma _ { Y Y } ^ { - 1 } \Sigma _ { Y U ^ { 2 } }$

To achieve conditional independence, the optimal adversary $q _ { C E } ^ { * }$ must cancel both of these mechanisms. Under the noise-injection framework, this is achieved by injecting the negative of the true conditional cross-covariance into the private noise:

$$
\begin{array} { r } { C _ { C E } = - \Sigma _ { U ^ { 1 } U ^ { 2 } | Y } ^ { ( p ) } . } \end{array}
$$

Proof. The objective is to maximize the determinant of the joint conditional covariance matrix $| \Sigma _ { U | Y } ^ { ( Q ) } |$ . By the properties of multivariate Gaussians, $\Sigma _ { U | Y } ^ { ( Q ) }$ is given by the Schur complement of the target variance $\Sigma _ { Y Y }$ . Since $Q$ is formed by injecting $C$ into the off-diagonal of the observations, the conditional covariance under $Q$ is the true conditional covariance plus the injected noise block:

$$
\Sigma _ { U | Y } ^ { ( Q ) } = \Sigma _ { U U } ^ { ( Q ) } - \Sigma _ { U Y } \Sigma _ { Y Y } ^ { - 1 } \Sigma _ { Y U } = \Sigma _ { U | Y } ^ { ( p ) } + \left[ { \bf 0 } \quad { \cal C } \right] = \left[ \Sigma _ { U ^ { 1 } | Y } ^ { ( p ) } \quad \Sigma _ { U ^ { 1 } U ^ { 2 } | Y } ^ { ( p ) } + { \cal C } \right] .
$$

By Hadamard’s inequality (Cover & Thomas, 2006, Theorem $1 7 . 9 . 4 )$ , the determinant of a block matrix with fixed diagonal blocks is maximized when the off-diagonal blocks are zero. Therefore, to maximize $| \Sigma _ { U | Y } ^ { ( Q ) } |$ , the adversary must set $C _ { C E } = - \Sigma _ { U ^ { 1 } U ^ { 2 } | Y } ^ { ( p ) }$ . For jointly Gaussian variables, a zero conditional cross-covariance implies conditional independence, achieving $U ^ { 1 } \perp \perp  { U } ^ { 2 } \mid Y$ Furthermore, because this choice makes $\Sigma _ { U | Y } ^ { ( Q ) }$ a block diagonal matrix whose diagonal blocks are the valid marginals from $p ,$ the conditional covariance $\Sigma _ { U | Y } ^ { ( Q ) }$ is positive semi-definite (PSD). By the properties of the Schur complement, a joint block covariance matrix is PSD if and only if its lower-right block $( \Sigma _ { Y Y } )$ and its Schur complement $( \Sigma _ { U | Y } ^ { ( Q ) } )$ are both PSD. Since the target variance $\Sigma _ { Y Y } \succ 0$ is fixed by the constraint space, and our choice of $C _ { C E }$ ensures $\Sigma _ { U | Y } ^ { ( Q ) } \succeq 0 .$ , the resulting full joint covariance matrix $\Sigma ^ { ( Q ) }$ is guaranteed to be a valid PSD matrix. This confirms that $q _ { C E } ^ { * }$ is a feasible point in the constraint space $\Delta _ { P }$ □

Remark 1 (The Three Mechanisms of Synergy: Suppression, Averaging, and Explaining-Away). This algebraic solution reveals how $q _ { C E } ^ { * }$ removes the sources ofmulti-view synergy. By substituting the generative components, the true conditional cross-covariance decomposes into three terms, each corresponding to a synergistic mechanism:

$$
\Sigma _ { U ^ { 1 } U ^ { 2 } | Y } ^ { ( p ) } = \underbrace { \Lambda _ { 1 } ^ { c } ( \Lambda _ { 2 } ^ { c } ) ^ { \top } } _ { \substack { I . S h a r e d N u i s a n c e } } + \underbrace { \Lambda _ { 1 } ^ { 1 2 } ( \Lambda _ { 2 } ^ { 1 2 } ) ^ { \top } } _ { \substack { 2 . T n e S h a r e d S i g n a l } } - \underbrace { \Sigma _ { U ^ { 1 } Y } \Sigma _ { Y Y } ^ { - 1 } \Sigma _ { Y U ^ { 2 } } } _ { \substack { 3 . I n d u c e d V - S t n c t u r e } } .
$$

By injecting $C _ { C E } = - \Sigma _ { U ^ { 1 } U ^ { 2 } | Y } ^ { ( p ) }$ into the noise, the adversary acts as a filter that cancels all three mechanisms:

1. Suppression (Driven by $z ^ { c } ) { \mathrm { : } }$ : When the views share a nuisance factor $z ^ { c } ,$ , the optimal joint regression linearly cross-references the views to subtract and cancel out the shared interference. Consider a toy system where $U ^ { 1 } = Y + z ^ { c }$ and $U ^ { 2 } = z ^ { c }$ . Marginally, $U ^ { 2 }$ uninformative about $Y$ . However, a joint regression model computing the optimal linear estimator $( W = \Sigma _ { U U } ^ { - 1 } \Sigma _ { U Y } )$ assigns a negative weight to $U ^ { 2 }$ . This allows the model to subtract the shared noise, computing $U ^ { 1 } - U ^ { 2 } = Y$ and recovering the target. $q _ { C E } ^ { * }$ eliminates this mechanism by injecting $- \Lambda _ { 1 } ^ { c } ( \Lambda _ { 2 } ^ { c } ) ^ { \top }$ to decouple $z ^ { c }$ across the views, turning it into independent private noise so it can no longer be subtracted.

2. Averaging (Driven by $z ^ { 1 2 } ) .$ : Because both views observe the true shared signal $z ^ { 1 2 }$ corrupted by independent private noise, they provide redundant measurements. For example, $i f \dot { U } ^ { 1 } = \stackrel { \smile } { Y } + \epsilon ^ { 1 } \sp { \bullet } a n d U ^ { 2 } = Y + \epsilon ^ { 2 } ( w i t h \epsilon \ll \stackrel {  } { \bot } \stackrel {  } {  } \epsilon ^ { 2 } )$ , the optimaljoint estimator applies positive weights to both views to compute their average. This averaging boosts the signal-to-noise ratio by suppressing the uncorrelated private noises, yielding a cleaner prediction of Y than either view could individually. $q _ { C E } ^ { * }$ breaks this capability by injecting $- \Lambda _ { 1 } ^ { 1 2 } ( \Lambda _ { 2 } ^ { 1 \breve { 2 } } ) ^ { \top }$ which cancels the true shared signal correlation across the views, destroying the redundancy that enabled the averaging.

3. V-Structure Exploitation (Driven by $z ^ { 1 2 } , z ^ { 1 } , z ^ { 2 } ) \mathrm { . }$ : Because the independent task-relevant factors all independently cause the target $Y ,$ , they form a classical joint V-structure. $O b \mathrm { . }$ serving Y renders all taskfactors conditionally dependent via the “explaining away” effect, which a joint model exploits. $q _ { C E } ^ { * }$ breaks this cooperative dependence because its injected noise $( + \Sigma _ { U ^ { 1 } Y } \Sigma _ { Y Y } ^ { - 1 } \Sigma _ { Y U ^ { 2 } } )$ cancels the induced negative explaining-away correlation.

Together, these mechanisms dictate that any latent structure—whether shared task-relevant, shared nuisance, or conditionally induced by a V-structure—provides an opportunityfor synergistic predictive power. The baseline distribution $q _ { C E } ^ { * }$ eliminates them all.

## B.3 THE MINIMUM MUTUAL INFORMATION ADVERSARY $( q _ { M I } ^ { * } )$

The original PID framework (Bertschinger et al., 2014) defines its adversary $q _ { M I } ^ { * }$ through an alternative objective: to find the distribution that minimizes the joint mutual information $\bar { I _ { Q } } ( Y ; U )$ Because $H ( Y )$ is fixed across $\Delta _ { P }$ , this is equivalent to maximizing the residual generalized variance $| \Sigma _ { Y | U } ^ { ( Q ) } |$

$$
q _ { M I } ^ { * } = \arg \operatorname* { m a x } _ { Q \in \Delta _ { P } } | \Sigma _ { Y | U } ^ { ( Q ) } | .
$$

The connection between the two adversarial objectives is governed by the determinant identity:

$$
\big | \Sigma _ { Y | U } ^ { ( Q ) } \big | = \big | \Sigma _ { Y Y } \big | \frac { \big | \Sigma _ { U | Y } ^ { ( Q ) } \big | } { \big | \Sigma _ { U U } ^ { ( Q ) } \big | } .
$$

This identity follows directly from factoring the determinant of the joint covariance matrix $\Sigma ^ { ( Q ) }$ in two equivalent ways via its Schur complements: $| \Sigma ^ { ( Q ) } | = | \Sigma _ { Y Y } | | \Sigma _ { U | Y } ^ { ( Q ) } | = | \Sigma _ { U U } ^ { ( Q ) } | | \Sigma _ { Y | U } ^ { ( Q ) } |$ . To minimize the mutual information, $q _ { M I } ^ { * }$ manipulates the cross-covariance matrix $C$ to reduce the denominator $| \Sigma _ { U U } ^ { ( Q ) }$ |. Mathematically, injecting correlation reduces the generalized variance of a joint distribution. Because the marginal blocks $\Sigma _ { U ^ { 1 } U ^ { 1 } }$ and $\Sigma _ { U ^ { 2 } U ^ { 2 } }$ are fixed by $\Delta _ { P }$ , the joint determinant factors via the Schur complement as $| \Sigma _ { U U } ^ { ( Q ) } | = | \Sigma _ { U ^ { 1 } U ^ { 1 } } | | \Sigma _ { U ^ { 2 } U ^ { 2 } } - \Sigma _ { U ^ { 2 } U ^ { 1 } } ^ { ( Q ) } \Sigma _ { U ^ { 1 } U ^ { 1 } } ^ { - 1 } \bar { \Sigma } _ { U ^ { 1 } U ^ { 2 } } ^ { ( Q ) } |$ . Injecting a cross-covariance subtracts a positive semi-definite matrix from the second term, thereby reducing the overall determinant. $q _ { M I } ^ { * }$ exploits this property by injecting positive correlation between the private noises across the views. This compresses the joint noise distribution to align with the signal, maximizing the residual target variance.

Like $q _ { C E } ^ { * } , q _ { M I } ^ { * }$ must first decouple the response-irrelevant nuisance factor $z ^ { c } , \operatorname { I f } z ^ { c }$ remains shared across the views, the optimal estimator will exploit this correlation to subtract the views and cancel the nuisance interference (the suppression mechanism). To prevent this cooperative cancellation and minimize the mutual information $I _ { Q } ( Y ; U ) , q _ { M I } ^ { * }$ adopts the same strategy as $q _ { C E } ^ { * }$ within the nuisance subspace: it injects negative correlation to decouple $z ^ { c }$ , turning it into independent private noise.

However, the two adversaries diverge fundamentally in how they treat the task-relevant subspace. The optimization objective of $q _ { C E } ^ { * }$ yields a clean, term-by-term matrix decomposition that explicitly decouples the task variables to achieve conditional independence $( I _ { q _ { \small C E } ^ { * } } ( \dot { U } ^ { 1 } ; U ^ { 2 } \ | \ Y ) \ \stackrel { \bullet } { = } \ 0 )$ In contrast, the objective of $q _ { M I } ^ { * }$ mathematically couples these variables together through the inverse covariance matrix. Because this coupling prevents a simple discrete factorization for the task subspace, the adversary must instead holistically inject positive correlation to align the joint noise ellipse with the combined signal direction. While this alignment effectively hinders the regression model, it inevitably introduces a conditional dependence between the views. Thus, the optimal mutual information adversary sacrifices conditional independence $( I _ { q _ { M I } ^ { * } } ( U ^ { 1 } ; U ^ { 2 } \mid Y ) > 0 )$ to achieve its objective.

To see this mechanically, consider a simplified scalar system where the private task-relevant factors $z ^ { 1 } , z ^ { 2 }$ are absent. The target is defined purely by the shared factor $Y \stackrel { \bullet } { = } z ^ { 1 2 }$ , with $z ^ { 1 2 } \sim \mathcal { N } ( 0 , 1 )$ The observations are $U = \Lambda z ^ { 1 2 } + n$ , where the stacked vector $\Lambda = \left[ \begin{array} { l } { \lambda _ { 1 } } \\ { \lambda _ { 2 } } \end{array} \right]$ defines the specific

1-dimensional “signal direction”. Here, $n \sim \mathcal N ( \mathbf { 0 } , \Sigma _ { N } ^ { ( p ) } )$ represents the sum of the full-rank private noise and the completely decoupled nuisance factors. Under the noise-injection framework, the adversary injects its cross-covariance c exclusively into this noise block, resulting in the total adversarial noise covariance $\Sigma _ { N } ^ { ( Q ) } = \left[ \begin{array} { c c } { { v _ { 1 } } } & { { c } } \\ { { c } } & { { v _ { 2 } } } \end{array} \right]$

The joint mutual information is minimized when $| \Sigma _ { Y | U } ^ { ( Q ) } |$ is maximized. Under our explicit model, $\Sigma _ { Y Y } = 1 , \Sigma _ { Y U } = \Lambda ^ { \top }$ , and $\Sigma _ { U U } ^ { ( Q ) } = \Lambda \Lambda ^ { \top } + \Sigma _ { N } ^ { ( Q ) }$ . Substituting these into the residual variance yields:

$$
\begin{array} { r } { | \Sigma _ { Y | U } ^ { ( Q ) } | = 1 - \Lambda ^ { \top } \left( \Lambda \Lambda ^ { \top } + \Sigma _ { N } ^ { ( Q ) } \right) ^ { - 1 } \Lambda . } \end{array}
$$

To simplify the inverted term, we apply the Woodbury matrix identity, $( \mathbf { A } + \mathbf { U C V } ) ^ { - 1 } = \mathbf { A } ^ { - 1 } -$ $\mathbf { A } ^ { - 1 } \mathbf { U } ( \mathbf { C } ^ { - 1 } + \mathbf { V } \mathbf { A } ^ { - 1 } \mathbf { U } ) ^ { - 1 } \mathbf { V } \mathbf { A } ^ { - 1 }$ , setting $\mathbf { A } = \Sigma _ { N } ^ { ( Q ) } , \mathbf { U } = \Lambda , \mathbf { V } = \Lambda ^ { \top }$ , and $\mathbf { C } = 1$ . Letting $K ( c ) = \Lambda ^ { \top } ( \Sigma _ { N } ^ { ( Q ) } ) ^ { - 1 } \Lambda$ represent the Signal-to-Noise Ratio (SNR) penalty, the inverse becomes:

$$
\left( \Sigma _ { N } ^ { \left( Q \right) } + \Lambda \Lambda ^ { \top } \right) ^ { - 1 } = ( \Sigma _ { N } ^ { \left( Q \right) } ) ^ { - 1 } - ( \Sigma _ { N } ^ { \left( Q \right) } ) ^ { - 1 } \Lambda \left( 1 + K ( c ) \right) ^ { - 1 } \Lambda ^ { \top } ( \Sigma _ { N } ^ { \left( Q \right) } ) ^ { - 1 } .
$$

Substituting this expanded inverse back into the residual variance equation yields a step-by-step algebraic reduction:

$$
\begin{array} { l } { | \Sigma _ { Y | U } ^ { ( Q ) } | = 1 - \Lambda ^ { \top } \left[ ( \Sigma _ { N } ^ { ( Q ) } ) ^ { - 1 } - ( \Sigma _ { N } ^ { ( Q ) } ) ^ { - 1 } \Lambda \left( 1 + K ( c ) \right) ^ { - 1 } \Lambda ^ { \top } ( \Sigma _ { N } ^ { ( Q ) } ) ^ { - 1 } \right] \Lambda } \\ { = 1 - \underbrace { \Lambda ^ { \top } ( \Sigma _ { N } ^ { ( Q ) } ) ^ { - 1 } \Lambda } _ { K ( c ) } + \underbrace { \Lambda ^ { \top } ( \Sigma _ { N } ^ { ( Q ) } ) ^ { - 1 } \Lambda } _ { K ( c ) } ( 1 + K ( c ) ) ^ { - 1 } \underbrace { \Lambda ^ { \top } ( \Sigma _ { N } ^ { ( Q ) } ) ^ { - 1 } \Lambda } _ { K ( c ) } } \\ { = 1 - K ( c ) + \frac { K ( c ) ^ { 2 } } { 1 + K ( c ) } } \\ { = \frac { 1 } { 1 + K ( c ) } . } \end{array}
$$

Thus, maximizing the residual variance is exactly equivalent to minimizing the SNR penalty term $K ( c )$ :

$$
K ( c ) = [ \lambda _ { 1 } \quad \lambda _ { 2 } ] \left[ { v _ { 1 } } \quad c \right] ^ { - 1 } \left[ \lambda _ { 1 } \right] = \frac { \lambda _ { 1 } ^ { 2 } v _ { 2 } + \lambda _ { 2 } ^ { 2 } v _ { 1 } - 2 c \lambda _ { 1 } \lambda _ { 2 } } { v _ { 1 } v _ { 2 } - c ^ { 2 } } .
$$

By taking the derivative with respect to c and setting it to zero, we find the adversary optimally minimizes the SNR penalty at the roots $c = \lambda _ { 1 } v _ { 2 } / \lambda _ { 2 }$ and $c = { \lambda _ { 2 } v _ { 1 } / \lambda _ { 1 } }$ . The true optimal adversarial correlation $c ^ { * }$ is the unique root that satisfies the positive semi-definite covariance constraint $c ^ { 2 } < v _ { 1 } v _ { 2 }$ . Because $v _ { 1 } , v _ { 2 } > 0$ , both roots strictly match the sign of the signal correlation $\lambda _ { 1 } \lambda _ { 2 }$ ensuring c<sup>∗</sup> always matches this sign as well. This sign-matching mathematically guarantees that the adversarial noise is geometrically aligned with the task signal. For instance, if the views have positively aligned signal loadings $( \lambda _ { 1 } \lambda _ { 2 } > 0 )$ , an optimal estimator would average the views to boost the signal while canceling independent noise. By injecting a positive noise correlation $( c ^ { * } > 0 )$

the adversary ensures that the noise components constructively reinforce each other upon addition. Conversely, if the signal extraction requires subtraction, the adversary injects negative correlation to ensure the subtracted noises similarly reinforce. Consequently, any linear combination that at tempts to boost the signal will simultaneously amplify the geometrically aligned adversarial noise, structurally preventing the model from achieving synergistic noise cancellation.

Remark 2 (The Mechanics of Adversarial Correlation). The noise-injectionframework captures the distinction between the adversaries: while both counteract cooperative nuisance suppression $( i . e . ,$ injecting negative correlation to render the nuisance factor $z ^ { c }$ independent), their treatment of the task-relevant predictive subspace differs.

To achieve conditional independence, $q _ { C E } ^ { * }$ acts as a decoupling filter: it injects the negative conditional cross-covariance $\begin{array} { r } { C _ { C E } = - \Sigma _ { U ^ { 1 } U ^ { 2 } | Y } ^ { ( p ) } } \end{array}$ . In our scalar example, this corresponds to setting $c = 0 ,$ , making the noise conditionally independent so the regression model could average the views to extract a cleaner signal.

In contrast, $q _ { M I } ^ { * }$ goes further to minimize mutual information. Rather than just canceling the conditional correlation, it injects a positive correlation C<sub>MI</sub>. By stretching and rotating the Gaussian noise ellipse $\Sigma _ { N } ^ { ( Q ) }$ so that it mimics the signal direction Λ, the adversary ensures that any linear estimator weight vector w attempting to extract the signal (maximizing the projection $w ^ { \top } \Lambda )$ simultaneously captures adversarial noise variance $( w ^ { \top } \Sigma _ { N } ^ { ( Q ) } w )$ . Mathematically, this minimizes the maximum achievable signal-to-noise ratio $\begin{array} { r } { ( K ( c ) = \operatorname* { m a x } _ { w } \frac { ( w ^ { \top } \Lambda ) ^ { 2 } } { w ^ { \top } \Sigma _ { N } ^ { ( Q ) } w } ) } \end{array}$ . Because the noise aligns with the signal, the estimator cannot project away the noise without also annihilating the signal itself, creating a structural dependency.

## B.4 SHARED PROPERTIES AND ANALYTIC BOUNDS

Despite their differing mathematical objectives, both $q _ { C E } ^ { * }$ and $q _ { M I } ^ { * }$ are constrained within $\Delta _ { P }$ . Because of this shared constraint space, they share structural properties that define the PID decomposition.

Remark 3 (The Marginal Preservation). Because any valid adversary $Q \in \Delta _ { P }$ matches the true predictor-target marginals $( I _ { Q } ( U ^ { j } ; Y ) = I _ { P } ( U ^ { j } ; Y ) )$ , the views maintain their individual marginal predictive relationships with Y even when the shared task factor $z ^ { 1 2 }$ is adversarially corrupted. By the chain rule, $\hat { I _ { Q } ( U ^ { 1 } , U ^ { 2 } ; Y ) } = I _ { P } ( U ^ { - j } ; Y ) + I _ { Q } ( U ^ { j } ; Y \mid U ^ { - j } )$ . Since the marginal term is fixed, minimizing the joint information is identical to minimizing the conditional mutual information. Thus, because $q _ { C E } ^ { * }$ is a sub-optimal minimizer ofjoint information $( I _ { q _ { C E } ^ { * } } \geq I _ { q _ { M I } ^ { * } } ) ,$ , its resulting conditional mutual information is larger, acting as an upper bound on the true unique information.

Remark 4 (Analytic Bounds via the Tractable Proxy). Because $q _ { C E } ^ { * }$ provides a closed-form but sub-optimal minimization ofthe joint information $( I _ { q _ { M I } ^ { * } } \leq I _ { q _ { C E } ^ { * } } ) ,$ substituting it into the PID equations yields analytic bounds on the true optimal quantities. Specifically, it underestimates Synergy $( S _ { C E } \ \le \ S _ { M I } )$ and Redundancy $( R _ { C E } \ \le \ R _ { M I } )$ , while overestimating Unique Information $( \check { U } _ { j } ^ { \check { C } E } \geq \check { U } _ { j } ^ { \check { M } I } ) .$

Finally, this formulation reveals that conditional mutual information is distinct from unique information. By the chain rule, $I _ { P } ( U ^ { 1 } ; Y \mid U ^ { 2 } ) = U n i q u e _ { 1 } + S y n e r g y .$ . By removing the synergy, the adversaries reduce the conditional mutual information to its unique information component.

## B.5 BRIDGE TO MULTI-VIEW LEARNING

This framework provides a theoretical bridge to classic multi-view learning. Classic multi-view algorithms typically invoke two distinct premises that are often bundled together: the redundancy assumption (that the views provide identical predictive information, meaning zero unique information), which is relied upon by modern contrastive and bottleneck methods (Federici et al., 2020; Tosh et al., 2021), and the conditional independence assumption (that $U ^ { 1 } \perp \perp { \boldsymbol { U } } ^ { 2 } \mid { \boldsymbol { Y } } ~ )$ , which is foundational to classic algorithms like Co-training (Blum & Mitchell, 1998) and CCA dimension reduction (Chaudhuri et al., 2009). Our baseline distribution $q _ { C E } ^ { * }$ is the mathematical embodiment of the latter. Conditional independence does not imply redundancy. Because $q _ { C E } ^ { * }$ preserves the true marginals, it allows for amounts of unique predictive information; it merely forbids the views from interacting synergistically. By measuring our tractable proxy $S _ { C E } ,$ , we approximate the true synergy and directly quantify how much the true data violates the conditional independence assumption. When ${ \cal S } _ { C E } \ \gg \ 0 .$ the views cooperate to cancel noise, indicating that algorithms assuming conditional independence will be suboptimal compared to joint models that can exploit synergistic cancellation.