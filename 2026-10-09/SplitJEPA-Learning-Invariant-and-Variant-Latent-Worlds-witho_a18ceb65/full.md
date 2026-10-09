# SplitJEPA: Learning Invariant and Variant Latent Worlds without Reconstruction

Ruijin Hua<sup>1</sup>, Zichuan Liu<sup>2</sup>, Zhuokai Zhao<sup>3</sup>, Yujia Zheng<sup>1</sup>

<sup>1</sup> University of Illinois Urbana-Champaign <sup>2</sup> Carnegie Mellon University <sup>3</sup> University of Chicago

## Abstract

Understanding a dynamical world calls for more than a latent state that summarizes its observations: the state should also be organized into the factors that stay shared across related observations and the factors that vary between them. For example, a robot pushing a cube to a goal should take the same action when the camera shifts or the lights dim, since nothing in the scene has moved. Existing approaches to this decomposition commonly obtain it through reconstruction, so the latent variables must first explain the entire observational world before their organization can be trusted. Joint embedding predictive architectures (JEPAs) model the latent state directly and never reconstruct, yet no existing result recovers the invariant and variant parts of the state they learn. How to learn the invariant-variant structure of the latent world without paying for its reconstruction therefore remains open. To close this gap, we introduce SplitJEPA, a JEPA that jointly recovers the latent state and its invariant and variant organization directly in representation space, without any reconstruction. We prove that, under stationary Gaussian predictive dynamics and a full-rank variation condition, SplitJEPA identifies the invariant and variant subspaces up to independent block-wise isometries, without introducing an observation decoder. Since the guarantee needs no decoder, the result extends reconstruction-free latent recovery to invariant-variant block identification. Experiments on synthetic nonlinear systems and robotic manipulation tasks support the theoretical results and show their practical value for both robustness and eficiency.

## 1 Introduction

Learning a useful representation of a dynamical world requires more than preserving information about its observations. The representation should also reveal how the latent state is structured into factors that remain stable and factors that vary across relevant changes (Scholkopf et al., 2021;¨ K¨ugelgen et al., 2021). Robot manipulation provides an example. Diferent executions of the same task may vary in robot configuration, local motion, viewpoint, illumination, and object appearance. A useful representation should retain the invariant content while separating the sources of variation, allowing downstream computations to use stable information without discarding variation when it is useful. This motivates structured representation learning: recovering not only the global latent state, but also its invariant–variant decomposition.

This objective has inspired invariant representation learning from multiview and temporal robotic data (Sermanet et al., 2018; Nair et al., 2022), as well as content-style approaches that seek to separate shared information from varying factors (Shrestha & Fu, 2025; Timilsina et al., 2026; Wang et al., 2025). These approaches commonly rely on reconstruction or related generative objectives: an encoder maps observations into latent variables, while the decoder requires those variables to keep enough information to reproduce the observations. Although efective, reconstruction introduces additional modeling and computational costs and can direct representation capacity toward noise and observation details that are unnecessary for downstream tasks. More fundamentally, reconstruction encourages the representation to preserve information needed to reproduce the observations, but it does not by itself specify how that information should be partitioned into invariant and variant structure. Indeed, an invertible nonlinear reparameterization may preserve the complete observation distribution while mixing the underlying latent factors (Hyvarinen & Pajunen, 1999; Locatello et al.,¨ 2019). Identifiable generative models address this ambiguity by introducing additional supervision or functional constraints (Hyvarinen et al., 2024; Moran & Aragam, 2026), illustrating that these¨ additional structures determine the latent organization. Our goal is to explore whether suitable structure can identify the desired decomposition directly in representation space without requiring an observation decoder. This raises the core question of our work: can the invariant–variant structure ofa dynamical latent state be identified without reconstructing observations?

![](images/954c7a427871dd624f0d815ff31506147d6e3b9556bc17453d630610210f31bc.jpg)  
Figure 1: Block identifiability without observation reconstruction. Reconstruction-based identifiable models recover structured latent factors through an encoder-decoder pathway together with additional structural assumptions. SplitJEPA combines prediction in latent space with block consistency to recover invariant and variant structure directly in representation space.

Reconstruction-free learning ofers a promising foundation. Joint-Embedding Predictive Architectures (JEPAs) learn informative representations by making predictions directly in representation space without reconstruction (Assran et al., 2023; Bardes et al., 2024; Dawid & LeCun, 2024). Related robotic studies further show that useful invariances can be learned without pixel reconstruction and that separating view-invariant from view-dependent information can improve visual world models (Zhang et al., 2021; Pang et al., 2025). However, these empirical results do not guarantee recovery of the ground-truth latent structure or its invariant-variant decomposition. Contrastive objectives can reduce nonlinear latent ambiguities by exploiting relationships between observations under asymmetry assumptions (Hyvarinen & Morioka, 2016; Hyvarinen & Morioka, 2017; Zimmermann et al., ¨ 2021). At the same time, block identification does not significantly require identifying individual coordinates within each block. LeJEPA provides a complementary result in this direction. Under stationary Gaussian predictive dynamics and appropriate representation normalization, it identifies the complete latent process (Balestriero & LeCun, 2025; Klindt et al., 2026). Nevertheless, the residual linear mixing can still entangle invariant and variant directions and therefore does not identify their decomposition. Consequently, whether the invariant-variant structure can be identified without reconstruction still remains an open question.

We introduce SplitJEPA, a JEPA that provably learns separate invariant and variant representations of a latent world without reconstruction. Our key insight is that prediction in representation space and consistency across invariant pairs resolve distinct levels of latent ambiguity. We achieve prediction in latent space, while block consistency under suficient variation resolves the mixing between invariant and variant factors. We establish conditions under which SplitJEPA identifies both latent blocks up to within-block orthogonal transformations, using only predictive and invariant correspondences without reconstructing observations. We further show that downstream predictions based on the invariant block are insensitive to variant perturbations when other inputs remain fixed. These results guide a framework that combines latent prediction, invariant-block agreement, and representation normalization. Experiments on controlled nonlinear systems and robotic manipulation tasks validate the proposed theory and demonstrate principled block recovery and improved downstream control.

## 2 Problem Background and Formulation

We consider a latent world observed through an unknown nonlinear process,

$$
h = f ( z ) , \qquad z = { \Biggl [ } { \boldsymbol { z } } _ { H } { \Biggr ] } \in \mathbb { R } ^ { d } , \qquad d = d _ { H } + d _ { L } ,\tag{1}
$$

where $z _ { H } \in \mathbb { R } ^ { d _ { H } }$ and $z _ { L } \in \mathbb { R } ^ { d _ { L } }$ denote factors that remain shared and factors that may change across related observations. An encoder $\phi : \mathcal { H } \xrightarrow { } \mathbb { R } ^ { d }$ induces

$$
r = \phi ( h ) = g ( z ) , \qquad g = \phi \circ f , \qquad r = \left[ r _ { H } \right] .\tag{2}
$$

We study whether $r _ { H }$ and $r _ { L }$ recover the corresponding latent blocks without reconstructing ℎ.

Following nonlinear identifiability settings, we take � to be a smooth embedding on the connected latent support and � to be continuously diferentiable (K¨ugelgen et al., 2021; Daunhawer et al., 2023). This preserves latent-state information while leaving its parameterization undetermined, the central ambiguity of nonlinear latent-variable models (Hyvarinen & Pajunen, 1999).¨

Predictive structure. Temporal and contrastive identifiability results constrain latent parameterizations through relations between observations (Hyvarinen & Morioka, 2016; Hyvarinen & Morioka,¨ 2017; Zimmermann et al., 2021). For LeJEPA, this connection yields linear recovery under stationary Gaussian latent dynamics (Klindt et al., 2026; Balestriero & LeCun, 2025). We build on this predictive setting:

$$
z ^ { ( 1 ) } \mid X \sim N ( 0 , I _ { d } ) , \qquad z ^ { ( 2 ) } = \rho _ { X } z ^ { ( 1 ) } + \sqrt { 1 - \rho _ { X } ^ { 2 } } \epsilon ,\tag{3}
$$

where $0 < \rho _ { X } < 1 , \epsilon \mid X \sim { \cal N } ( 0 , I _ { d } )$ , and $\epsilon \perp z ^ { ( 1 ) } \mid X$ . We denote this distribution by $\mathcal { D } _ { \mathrm { p r e d } }$ . The context � controls the predictive correlation, while both states have the same marginal distribution.

Invariant correspondences. Content-style learning supplies a complementary distinction: paired views preserve content while style may change (K¨ugelgen et al., 2021). Multiview and multimodal identifiability results similarly use shared factors to link observations with view-specific variation (Lyu et al., 2022; Daunhawer et al., 2023). Following this principle, let $\mathcal { D } _ { \mathrm { i n v } }$ denote a distribution over pairs

$$
z ^ { ( 1 ) } = ( z _ { H } , z _ { L } ^ { ( 1 ) } ) , \qquad z ^ { ( 2 ) } = ( z _ { H } , z _ { L } ^ { ( 2 ) } ) .\tag{4}
$$

The learner observes only $\big ( f ( z ^ { ( 1 ) } ) , f ( z ^ { ( 2 ) } ) \big )$ . The invariance is specific to this correspondence: �<sub>�</sub> is shared within an invariant pair but may change across predictive pairs. Whereas $\mathcal { D } _ { \mathrm { p r e d } }$ constrains predictive dependence, ${ \mathcal { D } } _ { \mathrm { i n v } }$ gives the partition its meaning; the extent of variation within these correspondences determines whether they sufice to identify it.

Identification target. Prior content-style and multimodal results identify shared factors up to a within-block invertible transformation (K¨ugelgen et al., 2021; Daunhawer et al., 2023). We seek the block-orthogonal form

$$
g _ { H } ( z ) = A _ { 1 1 } z _ { H } , \qquad g _ { L } ( z ) = A _ { 2 2 } z _ { L } ,\tag{5}
$$

where $A _ { 1 1 } ^ { \top } A _ { 1 1 } = I _ { d _ { H } }$ and $A _ { 7 7 } ^ { \top } A _ { 2 2 } = I _ { d _ { L } }$ . This identifies the two subspaces while allowing coordinates to mix within each block. The total latent dimension and both block dimensions are taken as given.

Recovering this organization is stronger than retaining the complete latent state. The following construction illustrates the distinction at the level of observations alone.

Theorem 1 (Nonlinear Latent Ambiguity). Let $h = f ( z _ { H } , z _ { L } )$ . For any smooth nonlinearfunction $\psi : \mathbb { R } ^ { d _ { L } }  \dot { \mathbb { R } } ^ { d _ { H } }$ , define $\tilde { z } _ { H } = z _ { H } + \psi ( z _ { L } ) , \tilde { z } _ { L } = z _ { L }$ , and $\tilde { f } ( \tilde { z } _ { H } , \tilde { z } _ { L } ) = f \bigl ( \tilde { z } _ { H } - \psi ( \tilde { z } _ { L } ) , \tilde { z } _ { L } \bigr )$ . Then $\widetilde { f } ( \widetilde { z } ) = f ( z )$ , while $\begin{array} { r } { \frac { \partial \widetilde { z } _ { H } } { \partial z _ { L } } = J _ { \psi } ( z _ { L } ) } \end{array}$ . Whenever this derivative is nonzero, the alternative first block depends on the variantfactor.

Structure. Letting $\tilde { z } = V _ { \mathrm { l e a k } } ( z )$ , the transformation has Jacobian $J _ { V _ { \mathrm { l e a k } } } ( z ) = \left\lceil \begin{array} { c c } { I _ { d _ { H } } } & { J _ { \psi } ( z _ { L } ) } \\ { 0 } & { I _ { d _ { L } } } \end{array} \right\rceil$ . Its of-diagonal block introduces variant information into the first block, while the explicit inverse preserves the complete latent state.

Example 1. Let $d _ { H } = d _ { L } = 1$ and $\psi ( z _ { L } ) = z _ { L } ^ { 2 }$ The states $( z _ { H } , z _ { L } ) = ( 2 , 1 )$ and (2, 2) share the same invariant $z _ { H } = 2 . \ A f t e r$ the transformation,

![](images/6fbd0b724316bf203188b7acbe6fc07ceeb89a05832308ee38520d3e48e69bb7.jpg)

![](images/a40e133a7596587a74673110bf6e993a53475a6e0065bf1a50a4b0a9462f1528.jpg)

they become $( \tilde { z } _ { H } , \tilde { z } _ { L } ) = ( 3 , 1 )$ and $( 6 , 2 )$ Figure 2: Nonlinear mixing of the latent blocks.. Thus, the transformedfirst coordinate varies with $z _ { L }$ , though the original invariantfactor is unchanged.

Proposition 1 (Ambiguity under the Basic Setup). Under the smooth observation model above, the invariant-variant decomposition is not identifiable from observations alone. The parameterizations $( f , z )$ and $( \tilde { f } , \tilde { z } )$ in Theorem 1 are observationally equivalent but are not related by independent within-block transformations when $J _ { \psi }$ is nonzero somewhere on the latent support. Consequently, information preservation does not imply recovery ofthe block-structured target.

This ambiguity concerns observation equivalence, where the transformed parameterization need not preserve the relational structure specified above for all such admissible transformations. Recent work similarly studies which latent structures remain recoverable when full nonlinear identification is unavailable, using diversity in latent-observation dependencies as an identifying signal (Zheng et al., 2026). Our setting uses a diferent source of structure: prediction first reduces the ambiguity to a global orthogonal transformation, and variation across correspondences identifies its block partition.

## 3 SplitJEPA: Learning Invariant and Variant Latent Worlds without Reconstruction

SplitJEPA resolves two levels of ambiguity: predictive identification recovers the complete latent state, and suficiently diverse correspondences determine its invariant-variant partition. We first characterize the ambiguity remaining after predictive identification (Section 3.1), then show how suficient variation resolves the latent partition (Section 3.2). We subsequently discuss the resulting robustness (Section 3.3) and formulate the estimation conditions (Section 3.4).

## 3.1 Predictive Identification and the Remaining Ambiguity

JEPA learns from statistical relations between observations without reconstructing them (Assran et al., 2023; Bardes et al., 2024). LeJEPA provides a reconstruction-free instance of this principle (Balestriero & LeCun, 2025). Under the predictive setting in Section 2, its identifiability analysis characterizes global latent recovery through a linear representation (Klindt et al., 2026). For centered and whitened population solutions, this takes the form $g ( z ) = Q z$ , where $Q ^ { \top } Q = I _ { d }$ This result excludes the nonlinear mixing in Theorem 1, but a global rotation may still mix the two latent blocks.

Proposition 2 (Block Ambiguity of LeJEPA). Suppose a representation belongs to the globally identified equivalence class, so that $g ( z ) ~ = ~ Q z , Q ^ { \top } Q ~ = ~ I _ { d }$ For any orthogonal matrix $R \in$ $\mathbb { R } ^ { d \times d }$ , the representation $\tilde { g } ( z ) \ : = \ : R g ( z )$ belongs to the same equivalence class. Since this class contains transformations with nonzero cross-block components, predictive identification alone does not determine the invariant-variant partition.

Orthogonal transformations preserve distances, covariance, and the isotropic predictive structure. As a result, global recovery does not select the orientation associated with the correspondence defining $z _ { H }$ . Nevertheless, reducing the original nonlinear ambiguity to an unknown linear transformation substantially simplifies the remaining problem.

Proposition 3 (Recovery up to Orthogonal Transformation). Under global predictive identification in the centered and whitened representation class, the learned partition admits

$$
\boldsymbol { r } = \boldsymbol { Q } \boldsymbol { z } = \left[ \begin{array} { l l } { \boldsymbol { A } _ { 1 1 } } & { \boldsymbol { A } _ { 1 2 } } \\ { \boldsymbol { A } _ { 2 1 } } & { \boldsymbol { A } _ { 2 2 } } \end{array} \right] \left[ \begin{array} { l } { \boldsymbol { z } _ { H } } \\ { \boldsymbol { z } _ { L } } \end{array} \right] , \qquad \boldsymbol { Q } ^ { \top } \boldsymbol { Q } = \boldsymbol { I } _ { d } .\tag{6}
$$

The complete latent state is linearly recoverable as $z = Q ^ { \dagger } r$ . The block-orthogonal target in Equation 5 holds if and only $i f A _ { 1 2 } = 0 a n d A _ { 2 1 } = 0 .$

It locates the remaining ambiguity in $A _ { 1 2 }$ and $A _ { 2 1 }$ . Predictive recovery preserves the complete state, but it does not determine which directions should remain unchanged under the correspondence defining $z _ { H }$ . The question is whether observed changes sufice to determine the desired partition.

## 3.2 Identifying the Partition through Variation

Once the representation is globally linear, agreement across correspondences constrains its dependence on the variant state. The strength of this constraint depends on which directions actually change. We formalize the required coverage below.

a) Matched PushCube Inputs c) Reconstruction-free Objectives d) Block-identifiable Representationb) Latent Factors and Observations

![](images/4e8730e2d9eaa2a26ede360d089020c69137736c768fee5812abb98305f0ae3e.jpg)  
Figure 3: Overview of reconstruction-free block identifiability. (a) Matched PushCube trajec tories share invariant task content while varying in execution and environmental factors. (b) The ground-truth latent state consists of invariant and variant blocks thatjointly generate the observations. (c) Predictive alignment and variance preservation identify the complete latent state, while block consistency determines its partition without an observation decoder. (d) The resulting representation recovers the invariant and variant subspaces. (e) Reconstruction alone permits observationally equivalent latent parameterizations, and prediction alone leaves a global orthogonal ambiguity; additional consistency structure resolves this ambiguity and yields reconstruction-free block identification.

Assumption 1 (Suficient Variation). For $( z ^ { ( 1 ) } , z ^ { ( 2 ) } ) \sim \mathcal { D } _ { \mathrm { i n v } }$ , let $\Delta z _ { L } = z _ { L } ^ { ( 1 ) } - z _ { L } ^ { ( 2 ) }$ . The secondmoment matrix ofthe variant changes satisfies

$$
\Sigma _ { \Delta L } = \mathbb { E } _ { \mathcal { D } _ { \mathrm { i n v } } } \left[ \Delta z _ { L } \Delta z _ { L } ^ { \top } \right] \succ 0 .\tag{7}
$$

Positive definiteness means that $\mathbb { E } [ ( \nu ^ { \top } \Delta z _ { L } ) ^ { 2 } ] > 0$ for every nonzero $\nu \in \mathbb { R } ^ { d _ { L } }$ : no linear direction in the variant block remains unchanged across all available pairs. A single pair need not vary every coordinate, and the coordinates of $\Delta z _ { L }$ need not be independent.

Intuition. Predictive learning first recovers the complete latent state up to a global orthogonal transformation $g ^ { \star } ( z ) = Q z$ . After partitioning � according to the invariant and variant latent blocks, the of-diagonal matrices $A _ { 1 2 }$ and $A _ { 2 1 }$ describe the remaining cross-block mixing. For matched pairs sharing $z _ { H }$ , the invariant-representation diference is $g _ { H } ^ { \star } ( z ^ { ( 1 ) } ) - g _ { H } ^ { \star } ( z ^ { ( 2 ) } ) = { \cal A } _ { 1 2 } \Delta z _ { L }$ . If the variant diferences have a full-rank second moment, agreement of $g _ { H } ^ { \star }$ across the matched pairs forces $A _ { 1 2 } = 0$ . The orthogonality of � then forces $A _ { 2 1 } = 0$ , leaving a block-diagonal transformation.

Theorem 2 (Reconstruction-free Block Identifiability). Under the predictive model in Equation 3 and Assumption 1, let $g ^ { \star }$ be a global population minimizer of $\mathcal { L } _ { \mathrm { p r e d } }$ over the centered and whitened representation class specified in Section 3.4. $I f g _ { H } ^ { \star } ( z ^ { ( 1 ) } ) = g _ { H } ^ { \star } ( \bar { z ^ { ( 2 ) } } )$ almost surely under $\mathcal { D } _ { \mathrm { i n v } } ,$ , then

$$
\begin{array} { r } { g ^ { \star } ( z ) = Q z = \left[ { A _ { 1 1 } } \quad \begin{array} { c c } { 0 } \\ { 0 } \end{array} \right] \left[ \begin{array} { c c } { z _ { H } } \\ { z _ { L } } \end{array} \right] , } \end{array}\tag{8}
$$

where $A _ { 1 1 } ^ { \top } A _ { 1 1 } = I _ { d _ { H } }$ and $A _ { 2 2 } ^ { \top } A _ { 2 2 } = I _ { d _ { L } }$ . Hence, the invariant and variant latent subspaces are identified up to independent within-block orthogonal transformations without reconstruction.

Identifying what can change. An invariant representation ignores every change exposed by the correspondence. However, agreement on a limited set of changes may leave other variant directions undetected. A correspondence is informative not merely because it preserves $z _ { H } .$ , but because its variation covers the directions in $z _ { L }$ . Suficient variation ensures that every nonzero linear dependence on $z _ { L }$ is exposed somewhere in the pair distribution. This condition becomes significant after global predictive recovery: the remaining dependence on $z _ { L }$ is described by a matrix.

Interpretation. Theorem 2 separates recovery of the complete latent state from identification of its organization. Predictive information determines which latent information is retained, while agreement across matched correspondences determines how that information is partitioned into invariant and variant subspaces. The resulting identification is block-wise: invariant factors may remain mixed within $r _ { H }$ , and variant factors may remain mixed within $r _ { L }$ . What is excluded is mixing across the two blocks. Thus, the theorem identifies the invariant-variant decomposition up to independent within-block isometries, without requiring an observation decoder or access to the ground-truth latent variables. The proof is provided in Appendix B.2.

Remark 1 (Stability under approximate invariant consistency). The suficient-variation condition also provides a quantitative stability guarantee when invariant consistency is approximate. Suppose that global recovery gives $g ( z ) = Q z$ with � orthogonal, and that the matched-pair consistency error satisfies $\begin{array} { r } { \mathbb { E } \Big [ \big \| g _ { H } \big ( z ^ { ( 1 ) } \big ) - g _ { H } \big ( z ^ { ( 2 ) } \big ) \big \| _ { 2 } ^ { 2 } \Big ] \leq \varepsilon . \ I f \Sigma _ { \Delta L } = \mathbb { E } \big [ \Delta z _ { L } \Delta z _ { L } ^ { \top } \big ] \leq \mu I } \end{array}$ for some $\mu > 0 ;$ , then $\varepsilon \geq \mathrm { t r } \big ( A _ { 1 2 } \Sigma _ { \Delta L } A _ { 1 2 } ^ { \top } \big ) \geq \mu \| A _ { 1 2 } \| _ { F } ^ { 2 }$ . Consequently, $\begin{array} { r } { \| A _ { 1 2 } \| _ { F } \leq \sqrt { \frac { \varepsilon } { \mu } } . } \end{array}$

Example 2. Let $z _ { H } = c a n d z _ { L } = ( u , \nu ) ^ { \top }$ . Forfixed �, varying (�, �) traces a two-dimensional latent sheet. Onefamily ofcorrespondences changes �, while another changes �, possibly at diferent values of�. Although neither spans the variant plane alone, their combined changes do. Since prediction in latent space constrains � to be an orthogonal transform of �, agreement of�<sub>�</sub> along both directions removes its dependence on $z _ { L } ;$ orthogonality assigns the complementary variant subspace to $r _ { L }$

Learning from separate changes. Example 2 illustrates the role of complementary evidence. A direction that remains unchanged in one collection may change in another. The diferences determine which directions can belong to the invariant representation. A single diference generally rules out only a subset of possible crossblock transformations. Diferent pairs contribute diferent constraints. What matters is whether the collection of diferences spans the variant subspace. This condition is relevant in practical settings. For example, in robot manipulation, if viewpoint and lighting always change in a fixed linear combination, another combination may re-

![](images/86f5ceda190d8d7d6c768911c70197e91585d34f1ce17755d0f948d6495646c0.jpg)  
Figure 4: Complementary changes span the variant subspace.

main unchanged and therefore indistinguishable from invariant content. Thus, the condition concerns the directions revealed by the changes and their aggregate coverage.

Why suficient variation is plausible. Complementary changes can arise from combining repeated measurements of invariant content. In a robotic environment, observations from diferent cameras and lighting conditions expose diferent visual directions. Matched robotic states may additionally difer in arm configuration or approach, provided that the factors assigned to $z _ { H }$ remain fixed. Related to this, diferent policies have been used to recover environmental features in identifiable skill learning (Reizinger et al., 2026). Here, diversity plays a more specific role: individual sources need not be suficient, but their combined changes within correspondences that preserve $z _ { H }$ must span the variant subspace. The relevant diversity is therefore measured within the specified correspondence. Observations must preserve the shared state while revealing diferent directions of variation. The learner need not observe these latent changes directly; they only need to be present in the paired data. This also explains the role of Assumption 1. A variant direction that never changes across the available pairs cannot be distinguished from invariant content. A direction that changes only weakly provides less reliable evidence in finite samples. Suficient variation ensures that every variant direction contributes information for determining the partition.

## 3.3 Robustness under Variant Perturbations

Block identifiability is not only a representation-level guarantee; it also has an operational consequence for downstream prediction. When a downstream computation uses only the invariant representation, changes confined to the variant latent factor have no path through its representation.

Theorem 3 (Gradient Isolation). Under the block identifiability established in Theorem 2, suppose a diferentiable downstream predictor depends solely on the invariant representation $r _ { H } .$ . Then the prediction is locally insensitive to perturbations in $\begin{array} { r } { z _ { L } \colon \ \frac { \partial Y } { \partial z _ { L } } \ = \ 0 . } \end{array}$ . Consequently, local variant perturbations cannot propagate through the invariant representation.

Interpretation. Theorem 3 gives block identifiability an operational meaning: a predictor using $r _ { H }$ can respond to changes in the shared state without being afected by how the variant factors realize that state. Indeed, $g _ { H } ( z ) = A _ { 1 1 } z _ { H }$ implies that all admissible states with the same $z _ { H }$ produce the same invariant representation, even when their observations difer substantially. The zero gradient is the local expression of this invariance. The invariance guarantee is specific to variant changes, which do not require insensitivity to changes in $z _ { H }$ . Once $g _ { H }$ depends only on $z _ { H }$ , a change in $z _ { L }$ has no path into the invariant representation. The result applies to finite variant perturbations, as long as the perturbed state remains within the modeled support. The proof is provided in Appendix B.3.

Example 3. Continuing Example 2, fix a sheet $z _ { H } ~ = ~ c$ and the downstream context, and move locally $\mathit { a l o n g z } ( t ) = ( c , u + t \delta _ { u } , \nu + t \delta _ { \nu } )$ . The observation and $r _ { L }$ may change, but Theorem 2 gives $r _ { H } ( t ) = A _ { 1 1 } c$ along this path. Hence any predictor $Y ( t ) = \Psi ( { \mathrm { C o n t e x t } } , r _ { H } ( { \bar { t } } ) )$ has zero directional derivative at $t = 0 f o r$ every variant direction $( \delta _ { u } , \delta _ { \nu } )$

Task-relative robustness. In robotic manipulation, changes in camera pose or illumination leave predictions based on $r _ { H }$ unchanged when they afect only $z _ { L }$ . If a perturbation also modifies the shared task content, the invariance guarantee no longer applies, and the resulting prediction change depends on the magnitude of the change in the invariant factor and the sensitivity of the downstream map. This guarantee is relative to the specified correspondence: a camera change may be variant for object recognition but relevant for viewpoint estimation. Perturbations that also change $z _ { H }$ fall outside this invariance guarantee. With imperfect block recovery, downstream sensitivity additionally depends on residual cross-block dependence and the predictor’s sensitivity.

## 3.4 Estimation and Normalization

We now translate these theoretical conditions into representation-space objectives. For pairs drawn from the corresponding relational distribution, write $g = \phi \circ f ,$ , with $g { \check { ( } } z ^ { ( i ) } ) \ = \ \phi ( h ^ { ( i ) } )$ for each observed pair. The losses are evaluated using the observed pairs alone.

Representation and consistency losses. We use

$$
\mathcal { L } _ { \mathrm { p r e d } } ( g ) = \frac { 1 } { 2 } \mathbb { B } _ { \mathcal { D } _ { \mathrm { p r e d } } } \left[ \| g ( z ^ { ( 1 ) } ) - g ( z ^ { ( 2 ) } ) \| _ { 2 } ^ { 2 } \right] , \quad \mathcal { L } _ { \mathrm { i n v } } ( g ) = \mathbb { E } _ { \mathcal { D } _ { \mathrm { i n v } } } \left[ \| g _ { H } ( z ^ { ( 1 ) } ) - g _ { H } ( z ^ { ( 2 ) } ) \| _ { 2 } ^ { 2 } \right] .\tag{9}
$$

For estimation, predictive alignment and invariant consistency are combined with a variancepreserving regularizer:

$$
{ \mathcal { L } } _ { \mathrm { r e p r } } ( g ) = { \mathcal { L } } _ { \mathrm { p r e d } } ( g ) + \lambda _ { \mathrm { i n v } } { \mathcal { L } } _ { \mathrm { i n v } } ( g ) + \lambda _ { \mathrm { v a r } } { \mathcal { L } } _ { \mathrm { v a r } } ( g ) , \qquad \lambda _ { \mathrm { i n v } } , \lambda _ { \mathrm { v a r } } > 0 .\tag{10}
$$

The variance regularizer discourages collapse, while $\mathcal { L } _ { \mathrm { i n v } }$ encourages agreement of the designated invariant block across matched pairs. Empirical versions replace expectations with sample averages.

Normalization constraint. The exact analysis additionally restricts the full representation under the predictive marginal to satisfy

$$
\begin{array} { r } { \mathbb { E } [ g ( z ) ] = 0 , \qquad \mathrm { C o v } ( g ( z ) ) = I _ { d } , \qquad z \sim N ( 0 , I _ { d } ) . } \end{array}\tag{11}
$$

Finite-weight variance regularization encourages this normalized geometry but does not by itself impose exact whitening. Full whitening includes zero cross-covariance between the designated blocks. Once predictive optimality yields $g ^ { \star } ( z ) = Q z$ , normalization gives $Q Q ^ { \top } = I _ { d } .$ . Since $Q$ is square, it is orthogonal, providing the geometry used to eliminate both cross-block components.

## 4 Experiments

We test two implications of the theory: whether correspondences resolve the ambiguity left by predictive recovery, and whether the resulting blocks support control in robotic manipulation under varying conditions. Synthetic experiments isolate the identification mechanism; robotic experiments evaluate a practical implementation across manipulation tasks and evaluation settings.

Table 1: Block recovery on the synthetic OU system. All methods reuse the same frozen reconstruction-free encoder and difer only in the information available to the spectral recovery stage. Results are the mean ± standard deviation over five seeds. Global $R ^ { 2 }$ is approximately $0 . 9 9 7 7 \pm 0 . 0 0 1 6$ for all methods.
<table><tr><td>Method</td><td> $\operatorname { I n v a r i a n t } R ^ { 2 }$ </td><td> $\operatorname { V a r i a n t } R ^ { 2 }$ </td><td> $\| A _ { 1 2 } \| _ { F }$ </td><td> $\| A _ { 2 1 } \| _ { F }$ </td><td>Block Purity</td></tr><tr><td>LeJEPA</td><td> $0 . 9 6 1 3 \pm 0 . 0 3 3 4$ </td><td> $0 . 9 6 1 2 \pm 0 . 0 3 2 8$ </td><td> $0 . 3 2 9 2 \pm 0 . 2 1 4 5$ </td><td> $0 . 3 2 9 9 \pm 0 . 2 1 3 7$ </td><td> $0 . 9 6 3 5 \pm 0 . 0 3 1 8$ </td></tr><tr><td>Shuffled Pairs</td><td> $0 . 7 1 3 4 \pm 0 . 0 7 1 4$ </td><td> $0 . 7 1 0 2 \pm 0 . 0 7 3 6$ </td><td> $1 . 0 6 0 0 \pm 0 . 1 4 1 5$ </td><td> $1 . 0 6 0 6 \pm 0 . 1 3 9 4$ </td><td> $0 . 7 1 4 2 \pm 0 . 0 7 3 4$ </td></tr><tr><td>SplitJEPA (Ours)</td><td> $\mathbf { 0 . 9 9 7 9 \pm 0 . 0 0 1 4 }$ </td><td> $\mathbf { 0 . 9 9 7 3 \pm 0 . 0 0 1 9 }$ </td><td> $\mathbf { 0 . 0 0 3 1 \pm 0 . 0 0 1 8 }$ </td><td> $\mathbf { 0 . 0 2 3 6 \pm 0 . 0 0 2 1 }$ </td><td> $\mathbf { 0 . 9 9 9 9 9 3 \pm 0 . 0 0 0 0 1 }$ </td></tr></table>

![](images/29f4546c07a0e96fff02af3546abe05cd0e3d365f669134dee227173cec71925.jpg)

![](images/b23162d6287a26db12481a63c88e82fbea52aa66f69f56def6f34bd70287f057.jpg)

![](images/662c7ccbf09407daa92421a071af87a3df50578540d22023b09a4a6b5544c87b.jpg)

![](images/eb270c1336da0802f20fc4d825cef9e9a04b329e9e5d41bbe5c9dfbd3cf725b1.jpg)  
Figure 5: Ablation of the predictive, invariance, and variance-preserving components. Bars show mean ± standard deviation over five seeds. (a) Removing predictive alignment reduces global linear recovery. (b)–(c) Removing invariant-pair recovery leaves global identification unchanged but degrades block recovery and increases cross-block leakage. (d) Removing variance regularization and ZCA normalization increases the representation covariance error.

## 4.1 Synthetic Evaluation

Simulation Setup. We train a reconstruction-free predictive encoder on a nonlinear Gaussian OU system, and all methods share the encoder. We report Global $R ^ { 2 } .$ , Invariant and Variant $R ^ { 2 }$ cross-block leakage $\| A _ { 1 2 } \| _ { F }$ and $\| A _ { 2 1 } \| _ { F } ,$ , and block purity over five seeds. Ground-truth latents are used only for evaluation. This shared-encoder comparison isolates organization of the recovered information from encoder quality. Data generation and estimation details appear in Appendix C.1.

Main Results. Table 1 illustrates the distinction in Proposition 2: equally informative representations can difer substantially in how they organize latent factors. With the predictive encoder fixed, SplitJEPA achieves near-perfect block recovery with little cross-block leakage, whereas LeJEPA retains mixing between the designated blocks. Shufling the pairs disrupts this separation without changing the globally available information. These results support the SplitJEPA mechanism: meaningful correspondences resolve an ambiguity that global latent recovery alone leaves open.

Ablations and Scaling. Figure 5 distinguishes the roles of prediction, correspondence and normalization. Removing prediction impairs global recovery, while removing invariant-pair recovery primarily disrupts the latent partition. Removing variance regularization and ZCA normalization increases covariance error and makes complementary-block recovery less stable. Scaling illustrated in Figures 7–11 reveals a corresponding distinction: more predictive training improves global recovery, whereas more recovery samples improve block separation even with the encoder fixed. Larger latent spaces make complementary-block estimation more sensitive, highlighting the finite-sample demands of recovering the full partition. Full results appear in Appendix C.

## 4.2 Robotic Manipulation Evaluation

Robotic manipulation Setup. We evaluate PickCube and PushCube in ManiSkill (Tao et al., 2025)<sup>0</sup> using 800 successful trajectories per task. Matched trajectories provide approximate crossexecution correspondences, while same-state base/side-camera views provide viewpoint correspondences. Behavior-cloning policies use either $r _ { H }$ or $r _ { H } + r _ { L } ;$ we compare them with Raw, frozen R3M (Nair et al., 2022), and LeJEPA (Balestriero & LeCun, 2025). We report validation action

![](images/2db68b14d58279e9a489ca6cf4ffc011fed107e188a02740035a15c5de18f571.jpg)

![](images/160a3d1a91829592a237ef5cf763f344cb017a4ca240e177240378c99a214497.jpg)  
Figure 6: Ofline action prediction performance. Validation action MSE during ofline behavior cloning training on PickCube and PushCube.

Table 2: Robotic manipulation performance. Results on PickCube and PushCube under ID and OOD conditions. CONTACT and GRASP are reported for ID; all OOD columns report SR@10cm.
<table><tr><td rowspan="3">Method</td><td colspan="5">PushCube Manipulation</td><td colspan="5">PickCube Manipulation</td></tr><tr><td colspan="2">ID</td><td colspan="3">OOD</td><td colspan="2">ID</td><td colspan="3">OOD</td></tr><tr><td></td><td>CONTACT SR@10cm</td><td>Camera</td><td>Lighting</td><td>Visual</td><td>GRASP</td><td>SR@10cm</td><td>Camera</td><td>Lighting</td><td>Visual</td></tr><tr><td>Raw</td><td> $1 0 0 . 0 { \pm } 0 . 0 \ \qquad $ </td><td> $9 8 . 0 { \pm } 1 . 0 $ </td><td> $1 2 . 8 { \pm } 6 . 8 $ </td><td> $9 8 . 3 { \pm } 1 . 5 $ </td><td> $1 2 . 7 { \pm } 7 . 0 $ </td><td> $2 3 . 0 { \pm } 1 . 6 $ </td><td> $1 4 . 0 { \pm } 1 . 4 $ </td><td> $1 0 . 3 { \pm } 0 . 5 $ </td><td> $1 1 . 7 { \pm } 0 . 9$ </td><td> $1 0 . 7 { \pm } 0 . 5 $ </td></tr><tr><td>R3M</td><td> $1 0 0 . 0 { \pm } 0 . 0 \ \qquad $ </td><td> $9 7 . 7 { \pm } 1 . 2 $ </td><td> $4 8 . 0 { \pm } 7 . 5 $ </td><td> $9 0 . 7 { \pm } 2 . 1 $ </td><td>25.8±1.8</td><td> $7 3 . 0 { \pm } 1 . 6 $ </td><td> $3 2 . 0 { \pm } 1 . 4 $ </td><td> $1 4 . 3 { \pm } 0 . 5 $ </td><td> $1 0 . 0 { \pm } 0 . 0 $ </td><td> $1 0 . 0 { \pm } 0 . 0 $ </td></tr><tr><td>LeJEPA</td><td> $9 0 . 3 { \pm } 1 4 . 2 $ </td><td> $6 7 . 3 { \pm } 2 0 . 1 $ </td><td> $2 5 . 0 { \pm } 2 . 6 $ </td><td> $3 1 . 0 { \pm } 3 1 . 2 $ </td><td> $2 8 . 7 { \pm } 1 7 . 5 $ </td><td> $3 3 . 3 { \pm } 3 . 3 $ </td><td> $1 6 . 3 { \pm } 0 . 9 $ </td><td> $1 0 . 0 { \pm } 0 . 0 $ </td><td> $1 0 . 0 { \pm } 0 . 0 $ </td><td> $1 0 . 0 { \pm } 0 . 0 $ </td></tr><tr><td>Ours-rH</td><td> $1 0 0 . 0 { \pm } 0 . 0 \ \qquad $ </td><td> $9 9 . 3 { \pm } 0 . 6 $ </td><td> $7 1 . 5 { \pm } 1 0 . 4 $ </td><td> $\mathbf { 9 9 . 7 { \pm 0 . 6 } }$ </td><td> $6 4 . 5 { \pm } 1 2 . 7 $ </td><td> ${ \bf 9 7 . 0 { \pm 1 . 4 } }$ </td><td> $4 5 . 0 { \pm } 2 . 4 $ </td><td> $\mathbf { 3 0 . 3 { \pm 6 . 1 } }$ </td><td> $4 2 . 7 { \pm } 4 . 5 $ </td><td> $2 8 . 3 { \pm } 3 . 9 $ </td></tr><tr><td> $\mathbf { O u r s } – r _ { H } + r _ { L }$ </td><td> $1 0 0 . 0 { \pm } 0 . 0 \ \qquad $ </td><td> $9 9 . 3 { \pm } 0 . 6 $ </td><td> $7 5 . 3 { \pm } 8 . 1 $ </td><td> $9 8 . 3 { \pm } 1 . 5 $ </td><td> $\mathbf { 7 0 . 2 \pm 1 2 . 9 }$ </td><td> $9 6 . 7 { \pm } 0 . 5 $ </td><td> ${ \bf 4 6 . 3 \pm 4 . 9 }$ </td><td> $2 7 . 7 { \pm } 0 . 9 $ </td><td> $4 3 . 3 { \pm } 2 . 9$ </td><td> ${ \bf 3 0 . 3 2 1 . 7 }$ </td></tr></table>

MSE and closed-loop SR@10cm under ID, camera, lighting, and combined visual shifts. Full data construction, training and evaluation details appear in Appendix D.1.

Action prediction and control. Both proposed representations improve validation action prediction over the baselines (Figure 6). Their similar errors suggest that $r _ { H }$ retains much of the information needed for this task, despite excluding the designated variant block.

The clearest closed-loop benefits appear under visual shifts (Table 2), where camera and lighting changes alter observations without changing the physical control objective. The gains are especially pronounced on PushCube; PickCube remains more challenging, indicating that separating visual variation does not remove the demands of grasping and transport. The relative performance of $r _ { H }$ and $r _ { H } + r _ { L }$ also depends on the shift: retaining both blocks benefits some conditions.

Ablation and Robustness Experiments. Table 6 tests prediction in latent space, invariance and cross-trajectory diversity. Removing prediction reduces performance under camera and combined shifts, despite relatively small ID changes on PushCube. Removing invariance produces the largest PushCube degradation: Camera SR@10cm falls while ID success remains high. PickCube also loses combined-shift performance, although camera and lighting efects are less uniform. The diversity ablation reveals a task-dependent trade-of: removing the diversity mechanism is neutral or beneficial in several PushCube conditions, but reduces PickCube camera and combined-shift success for both representation inputs. In contrast, the population guarantee requires informative variation among exact invariant correspondences, and practical cross-trajectory pairing trades broader variation coverage against approximate alignment quality. Additional experiments appear in Appendix D.2.

## 5 Conclusion

We present SplitJEPA, a reconstruction-free framework for recovering invariant and variant latent structure. We show that prediction in representation space and invariant-block consistency jointly yield block identifiability under suficient variation, and that the recovered structure isolates computations based on the invariant block from variant perturbations. Experiments on controlled nonlinear systems and robotic manipulation tasks demonstrate improved block recovery, prediction and control. Their scale is the main limitation of this work, and scaling SplitJEPA to large-scale complex real-world scenarios is an important next step. More broadly, our results suggest that representation learning can recover not only the information in a latent world but also how it is organized.

## References

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. arXiv preprint arXiv:2301.08243, 2023.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

Randall Balestriero and Yann LeCun. Lejepa: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544, 2025.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. Transactions on Machine Learning Research, pp. 1–28, 2024.

Imant Daunhawer, Alice Bizeul, Emanuele Palumbo, Alexander Marx, and Julia E Vogt. Identifiability results for multimodal contrastive learning. In ICLR, 2023.

Anna Dawid and Yann LeCun. Introduction to latent variable energy-based models: a path toward autonomous machine intelligence. Journal of Statistical Mechanics: Theory and Experiment, pp. 1–32, 2024.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked´ autoencoders are scalable vision learners. In CVPR, 2022.

Aapo Hyvarinen and Hiroshi Morioka. Nonlinear ica of temporally dependent stationary sources. In AISTATS, 2017.

Aapo Hyvarinen and Petteri Pajunen. Nonlinear independent component analysis: Existence and¨ uniqueness results. Neural Networks, pp. 429–439, 1999.

Aapo Hyvarinen, Ilyes Khemakhem, and Ricardo Monti. Identifiability of latent-variable and¨ structural-equation models: from linear to nonlinear: A. hyvarinen et al.¨ Annals of the Insti tute of Statistical Mathematics, pp. 1–33, 2024.

Aapo Hyvarinen and Hiroshi Morioka. Unsupervised feature extraction by time-contrastive learning¨ and nonlinear ica. In NeurIPS, 2016.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In ICLR, 2014.

David Klindt, Yann LeCun, and Randall Balestriero. When does lejepa learn a world model? arXiv preprint arXiv:2605.26379, 2026.

Julius Von K¨ugelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Scholkopf, Michel¨ Besserve, and Francesco Locatello. Self-supervised learning with data augmentations provably isolates content from style. In NeurIPS, 2021.

Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Ratsch, Sylvain Gelly, Bernhard Sch ¨ olkopf,¨ and Olivier Bachem. Challenging common assumptions in the unsupervised learning of disentangled representations. arXiv preprint arXiv:1811.12359, 2019.

Qi Lyu, Xiao Fu, Weiran Wang, and Songtao Lu. Understanding latent correlation-based multiview learning and self-supervision: An identifiability perspective. In ICLR, 2022.

Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels. arXiv preprint arXiv:2603.19312, 2026.

Gemma Moran and Bryon Aragam. Toward interpretable deep generative models via causal representation learning. Journal ofthe American Statistical Association, pp. 259–275, 2026.

Suraj Nair, Aravind Rajeswaran, Vikash Kumar, Chelsea Finn, and Abhinav Gupta. R3m: A universal visual representation for robot manipulation. In CoRL, 2022.

Jing-Cheng Pang, Nan Tang, Kaiyuan Li, Yuting Tang, Xin-Qiang Cai, Zhen-Yu Zhang, Gang Niu, Masashi Sugiyama, and Yang Yu. Learning view-invariant world models for visual robotic manipulation. In ICLR, 2025.

Patrik Reizinger, Balint Mucs´ anyi, Siyuan Guo, Benjamin Eysenbach, Bernhard Sch´ olkopf, and¨ Wieland Brendel. Skill learning via policy diversity yields identifiable representations for reinforcement learning. In ICLR, 2026.

Bernhard Scholkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner,¨ Anirudh Goyal, and Yoshua Bengio. Towards causal representation learning. arXiv preprint arXiv:2102.11107, 2021.

Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey Levine, and Google Brain. Time-contrastive networks: Self-supervised learning from video. In ICRA, 2018.

Sagar Shrestha and Xiao Fu. Content-style learning from unaligned domains: Identifiability under unknown latent dimensions. In ICLR, 2025.

Stone Tao, Fanbo Xiang, Arth Shukla, Yuzhe Qin, Xander Hinrichsen, Xiaodi Yuan, Chen Bao, Xinsong Lin, Yulin Liu, Tse-Kai Chan, Yuan Gao, Xuanlin Li, Tongzhou Mu, Nan Xiao, Ar nav Gurha, Viswesh N, Yong Woo Choi, Yen-Ru Chen, Zhiao Huang, Roberto Calandra, Rui Chen, Shan Luo, and Hao Su. Maniskill3: GPU parallelized robot simulation and rendering for generalizable embodied AI. In 7th Robot Learning Workshop, 2025.

Subash Timilsina, Sagar Shrestha, and Xiao Fu. Identifiable shared component analysis of unpaired multimodal mixtures. In NeurIPS, 2024.

Subash Timilsina, Hoang-Son Nguyen, Sagar Shrestha, and Xiao Fu. Content-style identification via diferential independence. In ICML, 2026.

Chenyu Wang, Sharut Gupta, Xinyi Zhang, Sana Tonekaboni, Stefanie Jegelka, Tommi Jaakkola, and Caroline Uhler. An information criterion for controlled disentanglement of multimodal data. In ICLR, 2025.

Yuxuan Wu, Ziyu Wang, Bhiksha Raj, and Gus Xia. Unsupervised disentanglement of content and style via variance-invariance constraints. In ICLR, 2025.

Hanqi Yan, Lingjing Kong, Lin Gui, Yuejie Chi, Eric Xing, Yulan He, and Kun Zhang. Counterfactual generation with identifiability guarantees. In NeurIPS, 2023.

Amy Zhang, Rowan Thomas McAllister, Roberto Calandra, Yarin Gal, and Sergey Levine. Learning invariant representations for reinforcement learning without reconstruction. In ICLR, 2021.

Yujia Zheng, Zijian Li, Shunxing Fan, Andrew Gordon Wilson, and Kun Zhang. Diverse dictionary learning. In ICLR, 2026.

Roland S. Zimmermann, Yash Sharma, Stefen Schneider, Matthias Bethge, and Wieland Brendel. Contrastive learning inverts the data generating process. In ICML, 2021.

## A Related Work

Reconstruction-Free Predictive Learning departs from reconstruction-based representation learning by constraining latent representations through predictive relationships between observations rather than by reconstructing the observations themselves (He et al., 2022; Kingma & Welling, 2014). This perspective also appears in nonlinear ICA, where contrastive and predictive objectives can achieve component-wise identifiability by exploiting temporal variation and additional sources of asymmetry, such as non-Gaussianity (Hyvarinen & Morioka, 2016; Hyvarinen & Morioka, 2017).¨ JEPAs (Assran et al., 2023; 2025; Bardes et al., 2024; Dawid & LeCun, 2024; Maes et al., 2026) follow this principle by predicting target representations from contextual observations, allowing the learned state to emphasize information that is predictive across observations. LeJEPA (Balestriero & LeCun, 2025; Klindt et al., 2026) further connects reconstruction-free prediction to identifiability through predictive alignment and an explicit Gaussianity regularizer. Under stationary Gaussian predictive dynamics and appropriate normalization, its population representation recovers the complete latent state up to a global orthogonal transformation. SplitJEPA builds on this global recovery result but considers a finer identification target: the remaining orthogonal transformation can preserve the complete latent state while still mixing its invariant and variant subspaces.

Content-Style Disentanglement seeks to decompose observations into a shared component that remains stable across related observations and a variation-specific component. With paired or aligned observations, discriminative learning can identify shared content, and in some settings can further separate shared and private factors, without reconstructing observations (K¨ugelgen et al., 2021; Lyu et al., 2022; Daunhawer et al., 2023). Recent work also characterizes controlled separation through variance–invariance constraints and information criteria (Wu et al., 2025; Wang et al., 2025). Other approaches study identification from unpaired domains using distributional variation together with additional statistical or structural conditions (Timilsina et al., 2024; Shrestha & Fu, 2025; Yan et al., 2023). In particular, diferential independence constrains the geometry of content- and style induced changes, enabling identification without assuming statistical independence between the two factors (Timilsina et al., 2026). We adopt the paired-correspondence structure established in this literature and use it to resolve the residual mixing left by predictive identification. Our work thereby recovers both invariant and variant subspaces within the complete predictive representation, up to independent block-wise isometries and without an observation decoder.

## B Proof

## B.1 Proof of Theorem 1

Theorem (Nonlinear Latent Ambiguity). Let $h = f ( z _ { H } , z _ { L } )$ . For any smooth nonlinear function $\psi : \mathbb { R } ^ { d _ { L } }  \mathbb { R } ^ { d _ { H } }$ , define $\tilde { z } _ { H } = z _ { H } + \psi ( z _ { L } ) , \tilde { z } _ { L } = z _ { L }$ , and $\tilde { f } ( \tilde { z } _ { H } , \tilde { z } _ { L } ) = f \bigl ( \tilde { z } _ { H } - \psi ( \tilde { z } _ { L } ) , \tilde { z } _ { L } \bigr )$ . Then $\widetilde f ( \widetilde z ) = f ( z )$ , while $\begin{array} { r } { \frac { \partial \widetilde { z } _ { H } } { \partial z _ { L } } = J _ { \psi } ( z _ { L } ) } \end{array}$ . Whenever this derivative is nonzero, the alternative first block depends on the variantfactor.

Proof. Let the continuous random vector be $z = [ z _ { H } ^ { \top } , z _ { L } ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { d }$ , where $d = d _ { H } + d _ { L }$ and $h = f ( z )$ We first construct a transformation that induces explicit variant-factor leakage. Choose an arbitrary smooth nonconstant function $\psi : \mathbb { R } ^ { d _ { L } }  \mathbb { R } ^ { d _ { H } }$ , and define $\widetilde { z } = V _ { \mathrm { l e a k } } ( z )$ by

$$
\widetilde { z } _ { H } = z _ { H } + \psi ( z _ { L } ) , \qquad \widetilde { z } _ { L } = z _ { L } .\tag{12}
$$

The Jacobian matrix of this transformation is

$$
J _ { V _ { \mathrm { l e a k } } } ( z ) = \left[ { \begin{array} { c c } { I _ { d _ { H } } } & { J _ { \psi } ( z _ { L } ) } \\ { 0 } & { I _ { d _ { L } } } \end{array} } \right] .\tag{13}
$$

Since det $J _ { V _ { \mathrm { l e a k } } } ( z ) ~ = ~ 1$ , the mapping preserves volume. Furthermore, $V _ { \mathrm { l e a k } } ^ { - 1 } ( \widetilde { z } _ { H } , \widetilde { z } _ { L } ) ~ = ~ \left( \widetilde { z } _ { H } ~ - ~ \right.$ $\psi ( \widetilde { z } _ { L } ) , \widetilde { z } _ { L } )$ , so $V _ { \mathrm { l e a k } }$ is a global smooth difeomorphism.

Returning to the explicit leakage construction, define the alternative observation-generating map

$$
\widetilde { f } = f \circ V _ { \mathrm { l e a k } } ^ { - 1 } .\tag{14}
$$

Then

$$
h = f ( z ) = f \bigl ( V _ { \mathrm { l e a k } } ^ { - 1 } ( \widetilde { z } ) \bigr ) = \widetilde { f } ( \widetilde { z } ) .\tag{15}
$$

Therefore, if the alternative latent is the pushforward $P _ { \widetilde { z } } = ( V _ { \mathrm { l e a k } } ) _ { \# } P _ { z }$ , the pairs $( f , P _ { z } )$ and $( \widetilde f , P _ { \widetilde { z } } )$ induce the same observed marginal distribution $p ( h )$

Conclusion. The true latent model $( f , P _ { z } )$ and the alternative latent model $( \widetilde f , P _ { \widetilde { z } } )$ generate the same observed marginal distribution $p ( h )$ . Consequently, observations of ℎ alone cannot distinguish the true invariant latent factor $z _ { H }$ from the alternative factor $\widetilde { z } _ { H } = z _ { H } + \psi ( z _ { L } )$ , which contains nonlinear information from the variant latent factor $z _ { L }$ . Accordingly, an encoder $\phi : \mathcal { H } \xrightarrow { } \mathbb { R } ^ { d }$ cannot identify the learned invariant representation $r _ { H } = [ \phi ( h ) ] _ { H }$ with the true $z _ { H }$ from the observation law alone. □

## B.2 Proof of Theorem 2

Theorem (Reconstruction-free Block Identifiability). Under the predictive model in Equation 3 and Assumption $^ { l , }$ let $g ^ { \star }$ be a global population minimizer of $\mathcal { L } _ { \mathrm { p r e d } }$ over the centered and whitened representation class specified in Section 3.4. $I f g _ { H } ^ { \star } ( z ^ { ( 1 ) } ) = g _ { H } ^ { \star } \hat { ( } z ^ { ( 2 ) } )$ almost surely under $\mathcal { D } _ { \mathrm { i n v } } ,$ , then

$$
\begin{array} { r } { g ^ { \star } ( z ) = Q z = \left[ { A _ { 1 1 } } \quad \begin{array} { c c } { 0 } \\ { 0 } \end{array} \right] \left[ \begin{array} { c c } { z _ { H } } \\ { z _ { L } } \end{array} \right] , } \end{array}\tag{16}
$$

where $A _ { 1 1 } ^ { \top } A _ { 1 1 } = I _ { d _ { H } }$ and $A _ { 2 2 } ^ { \top } A _ { 2 2 } = I _ { d _ { L } }$ Hence, the invariant and variant latent subspaces are identified up to independent within-block orthogonal transformations without observation recon struction.

Proof. The proof follows the identifiability decomposition induced by the predictive objective and the shared trajectory structure. We first show that the predictive dynamics restrict the optimal representation to a linear metric-preserving embedding through Hermite spectral analysis. Subsequently, the shared invariance removes the cross-block dependency, while orthogonality removes the remaining residual leakage.

## Step 1. Predictive Dynamics Strictly Isolates the Linear Eigenspace.

We analyze the reconstruction-free predictive objective defined on the positive pairs introduced in Section 2. By the normalization constraint in Equation 11, $\mathrm { C o v } ( g ( z ) ) ~ = ~ I _ { d }$ and $\mathbb { E } [ g ( z ) ] = 0 .$ . Therefore, $\mathbb { E } \Vert g ( z ) \Vert _ { 2 } ^ { 2 } = d .$ The Hermite expansion energy is globally normalized: $\begin{array} { r } { \sum _ { i = 1 } ^ { d } \sum _ { | \alpha | \geq 1 } ( c _ { \alpha } ^ { ( i ) } ) ^ { 2 } = d . } \end{array}$ . Under this constraint, minimizing the predictive distance is equivalent to maximizing $\mathbb { E } [ \langle g ( z ^ { ( 1 ) } ) , g ( z ^ { ( 2 ) } ) \rangle ]$

Following the reconstruction-free predictive learning framework introduced in JEPA-style representation learning (Assran et al., 2023; Bardes et al., 2024), and its theoretical analysis in LeJEPA (Klindt et al., 2026) and the predictive model in Equation 3, we models the context-conditioned transition as

$$
z ^ { ( 2 ) } = \rho _ { X } z ^ { ( 1 ) } + \sqrt { 1 - \rho _ { X } ^ { 2 } } \epsilon , \qquad \epsilon \sim N ( 0 , I _ { d } ) ,\tag{17}
$$

where conditionally on $X , \epsilon$ is independent of $z ^ { ( 1 ) }$ and $0 < \rho _ { X } < 1$ . Stationarity means $z ^ { ( 1 ) } \mid X \sim$ ${ \cal N } ( 0 , I _ { d } )$ and hence also $z ^ { ( 2 ) } \mid X \sim \mathbf { \bar { N } } ( 0 , I _ { d } )$

Because the representation is centered by the normalization constraint in Equation 11, the degree-0 constant term vanishes $( c _ { 0 } ^ { ( i ) } = 0 )$ . The same constraint gives $\begin{array} { r } { \mathrm { V a r } ( g _ { i } ( z ) ) = \sum _ { | \alpha | \geq 1 } \bigl ( c _ { \alpha } ^ { ( i ) } \bigr ) ^ { 2 } = 1 } \end{array}$ Grouping by total degree $k = | \alpha |$ , let $\begin{array} { r } { w _ { k } ^ { ( i ) } = \sum _ { | \alpha | = k } \bigl ( c _ { \alpha } ^ { ( i ) } \bigr ) ^ { 2 } } \end{array}$ , and $\begin{array} { r } { \sum _ { k \geq 1 } w _ { k } ^ { ( i ) } = 1 } \end{array}$

To compute the cross-trajectory correlation exactly, the calculation proceeds in three steps:

(a) The transition attenuates each Hermite degree by $\rho _ { X } ^ { k }$ conditionally on �. Consider the univariate case for the �-th coordinate, $z _ { j } ^ { ( 2 ) } = \rho _ { X } z _ { j } ^ { ( 1 ) } + \sqrt { 1 - \rho _ { X } ^ { 2 } } \epsilon _ { j }$ . We use the exponential generating function for Hermite polynomials:

$$
\sum _ { k = 0 } ^ { \infty } { \frac { s ^ { k } } { k ! } } H e _ { k } ( x ) = \exp \left( s x - { \frac { s ^ { 2 } } { 2 } } \right)
$$

evaluating at $z _ { j } ^ { ( 2 ) }$ , and taking conditional expectation over $\epsilon _ { j }$ gives

$$
\mathbb { E } _ { \epsilon _ { j } } \left[ \exp \left( s z _ { j } ^ { ( 2 ) } - \frac { s ^ { 2 } } { 2 } \right) \bigg | z _ { j } ^ { ( 1 ) } , X \right] = \exp \left( s \rho _ { X } z _ { j } ^ { ( 1 ) } - \frac { ( s \rho _ { X } ) ^ { 2 } } { 2 } \right) .\tag{18}
$$

Matching coeficients gives

$$
\mathbb { E } _ { \epsilon _ { j } } \left[ H e _ { k } \Big ( z _ { j } ^ { ( 2 ) } \Big ) \mid z _ { j } ^ { ( 1 ) } , X \right] = \rho _ { X } ^ { k } H e _ { k } \Big ( z _ { j } ^ { ( 1 ) } \Big ) .\tag{19}
$$

(b) Diferent Hermite components do not interact across views. In the multivariate case, conditional independence across coordinates and iterated expectation give

$$
\begin{array} { r } { \mathbb { E } [ H _ { \alpha } ( z ^ { ( 2 ) } ) H _ { \beta } ( z ^ { ( 1 ) } ) \mid X ] = \rho _ { X } ^ { \mid \alpha \mid } \mathbb { E } [ H _ { \alpha } ( z ^ { ( 1 ) } ) H _ { \beta } ( z ^ { ( 1 ) } ) \mid X ] = \rho _ { X } ^ { \mid \alpha \mid } \delta _ { \alpha \beta } . } \end{array}\tag{20}
$$

We use the normalized multivariate Hermite basis $\begin{array} { r } { H _ { \alpha } ( z ) = \prod _ { j = 1 } ^ { d } \mathrm { H e } _ { \alpha _ { j } } ( z _ { j } ) / \sqrt { \alpha _ { j } ! } } \end{array}$ . After averaging over �, define

$$
\lambda _ { k } = \mathbb { E } _ { X } [ \rho _ { X } ^ { k } ] ,\tag{21}
$$

so that

$$
\mathbb { E } [ H _ { \alpha } ( z ^ { ( 2 ) } ) H _ { \beta } ( z ^ { ( 1 ) } ) ] = \lambda _ { | \alpha | } \delta _ { \alpha \beta } .\tag{22}
$$

(c) Nonlinearity strictly reduces correlation. The total cross-trajectory correlation of coordinate � is

$$
\mathbb { E } [ g _ { i } ( z ^ { ( 2 ) } ) g _ { i } ( z ^ { ( 1 ) } ) ] = \sum _ { k \geq 1 } w _ { k } ^ { ( i ) } \lambda _ { k } .\tag{23}
$$

Since $0 < \rho _ { X } < 1$ , we have $\lambda _ { k } < \lambda _ { 1 }$ for every $k \geq 2$ . Therefore,

$$
\sum _ { k \geq 1 } w _ { k } ^ { ( i ) } \lambda _ { k } \leq \lambda _ { 1 } \sum _ { k \geq 1 } w _ { k } ^ { ( i ) } = \lambda _ { 1 } ,\tag{24}
$$

and equality requires $w _ { k } ^ { ( i ) } = 0$ for all $k \geq 2$ . Thus all variance at a global optimum lies in degree one. Since the degree-1 Hermite basis consists of the linear coordinate functions, $r = g ( z ) = Q z$

The normalization constraint in Equation 11 now gives $I _ { d } = \mathrm { C o v } ( g ( z ) ) = Q Q ^ { \top }$ . Because $g$ has output dimension $d ,$ the matrix $\dot { Q } \in \mathbb { R } ^ { d \times d }$ is square; hence $Q$ is orthogonal and also satisfies $Q ^ { \top } Q = I _ { d }$ . Partition it as

$$
\begin{array} { r } { Q = \left[ { A _ { 1 1 } \quad A _ { 1 2 } } \right] , } \end{array}\tag{25}
$$

where � ∈ R<sup>��</sup> <sup>×��</sup> , $A _ { 1 2 } \in \mathbb { R } ^ { d _ { H } \times d _ { L } } , A _ { 2 1 } \in \mathbb { R } ^ { d _ { L } \times d _ { H } }$ , and $A _ { 2 2 } \in \mathbb { R } ^ { d _ { L } \times d _ { L } }$ . The Hermite argument establishes $g ( z ) = Q z$ almost everywhere under the Gaussian marginal. Since $g$ is continuous and the Gaussian distribution has full support on $\mathbb { R } ^ { d }$ , the equality extends to every point in the latent support, including the endpoints occurring under $\mathcal { D } _ { \mathrm { i n v } }$

## Step 2. Content Invariance Eliminates the Upper-Right Block $\left( A _ { 1 2 } = 0 \right)$ .

Since Step 1 establishes $r = g ( z ) = Q z$ , the learned invariant block is $r _ { H } = g _ { H } ( z ) = A _ { 1 1 } z _ { H } + A _ { 1 2 } z _ { L }$ $\begin{array} { r } { \frac { \partial r _ { H } } { \partial z _ { L } } = A _ { 1 2 } } \end{array}$ , and the elimination of cross-block dependence follows the identifiability arguments developed in recent content-style separation theory (Shrestha & Fu, 2025).

We want to establish that the extracted invariant block does not depend on the variant factor, which is equivalent to proving that the partial derivative strictly vanishes:

$$
\frac { \partial r _ { H } } { \partial z _ { L } } = A _ { 1 2 } = \mathbf { 0 }
$$

Because $g ^ { \star }$ satisfies $\mathcal { L } _ { \mathrm { i n v } } ( g ^ { \star } ) = 0$ , we have $g _ { H } ^ { \star } ( z ^ { ( 1 ) } ) = g _ { H } ^ { \star } ( z ^ { ( 2 ) } )$ ) almost surely for $( z ^ { ( 1 ) } , z ^ { ( 2 ) } ) \sim \mathcal { D } _ { \mathrm { i n v } }$ Since these pairs satisfy $z _ { H } ^ { ( 1 ) } = z _ { H } ^ { ( 2 ) }$ , it follows that $A _ { 1 2 } \left( z _ { L } ^ { ( 1 ) } - z _ { L } ^ { ( 2 ) } \right) = A _ { 1 2 } \Delta z _ { L } = 0$ almost surely. By Assumption 1, $\Sigma _ { \Delta L } = \mathbb { E } [ \Delta z _ { L } \Delta z _ { L } ^ { \top } ] \succ 0$ . Hence

$$
0 = \mathbb { E } \| A _ { 1 2 } \Delta z _ { L } \| _ { 2 } ^ { 2 } = \operatorname { T r } ( A _ { 1 2 } \Sigma _ { \Delta L } A _ { 1 2 } ^ { \top } ) = \| A _ { 1 2 } \Sigma _ { \Delta L } ^ { 1 / 2 } \| _ { F } ^ { 2 }\tag{26}
$$

Since $\Sigma _ { \Delta L } ^ { 1 / 2 }$ is invertible, $A _ { 1 2 } = 0$

Step 3. Metric Preservation Removes the Remaining Cross Block $( A _ { 2 1 } = 0 )$ ).

With $A _ { 1 2 } = 0$ , the upper-left block of $Q Q ^ { \top } = I _ { d }$ is

$$
A _ { 1 1 } A _ { 1 1 } ^ { \top } = I _ { d _ { H } } .\tag{27}
$$

Thus the square matrix $A _ { 1 1 }$ is invertible. The upper-right block of $Q Q ^ { \top } = I _ { d }$ is

$$
A _ { 1 1 } A _ { 2 1 } ^ { \top } + A _ { 1 2 } A _ { 2 2 } ^ { \top } = A _ { 1 1 } A _ { 2 1 } ^ { \top } = 0 .\tag{28}
$$

Multiplying by $A _ { 1 1 } ^ { - 1 }$ gives $A _ { 2 1 } = 0$ . Finally, the lower-right block gives $A _ { 2 2 } A _ { 2 2 } ^ { \top } = I _ { d _ { L } }$ . Because $A _ { 1 1 }$ and $A _ { 2 2 }$ are square, we equivalently have

$$
A _ { 1 1 } ^ { \top } A _ { 1 1 } = I _ { d _ { H } } , \qquad A _ { 2 2 } ^ { \top } A _ { 2 2 } = I _ { d _ { L } } .\tag{29}
$$

Conclusion. Substituting $A _ { 1 2 } = 0$ and $A _ { 2 1 } = 0$ into $Q$ gives

$$
\begin{array} { r } { J _ { g } = Q = \left[ { \begin{array} { c c } { A _ { 1 1 } } & { 0 } \\ { 0 } & { A _ { 2 2 } } \end{array} } \right] . } \end{array}\tag{30}
$$

Therefore, the learned representation $r = g ( z )$ decomposes as

$$
r _ { H } = A _ { 1 1 } z _ { H } , \quad \quad r _ { L } = A _ { 2 2 } z _ { L } ,\tag{31}
$$

where $A _ { 1 1 } ^ { \top } A _ { 1 1 } = I _ { d _ { H } }$ and $A _ { 2 2 } ^ { \top } A _ { 2 2 } = I _ { d _ { L } }$ . Hence, $r _ { H }$ recovers the invariant latent subspace associated with $z _ { H }$ , while $r _ { L }$ recovers the variant latent subspace associated with $z _ { L }$ , in each case up to an independent orthogonal transformation. □

## B.3 Proof of Theorem 3

Theorem (Gradient Isolation). Under the block identifiability established in Theorem $^ { 2 , }$ suppose a diferentiable downstream predictor depends solely on the invariant representation $r _ { H }$ . Then the prediction is locally insensitive to perturbations in $\begin{array} { r } { z _ { L } \colon \ \frac { \partial Y } { \partial z _ { L } } \ = \ 0 } \end{array}$ . Consequently, local variant perturbations cannot propagate through the invariant representation.

Proof. By Theorem 2, the learned representation admits the block-diagonal decomposition

$$
r = g ( z ) = \left[ \begin{array} { c c } { { A _ { 1 1 } } } & { { 0 } } \\ { { 0 } } & { { A _ { 2 2 } } } \end{array} \right] \left[ \begin{array} { c c } { { z _ { H } } } \\ { { z _ { L } } } \end{array} \right]
$$

which implies $\begin{array} { r } { \frac { \partial r _ { H } } { \partial z _ { L } } = 0 } \end{array}$ . Consider a downstream predictor

$$
Y = \Psi ( { \mathrm { C o n t e x t } } , r _ { H } )
$$

whose prediction depends only on the invariant representation. Applying the multivariate chain rule,

$$
\frac { \partial Y } { \partial z _ { L } } = \frac { \partial Y } { \partial r _ { H } } \frac { \partial r _ { H } } { \partial z _ { L } } = 0
$$

Conclusion. Because $r _ { H } = A _ { 1 1 } z _ { H }$ is independent of $z _ { L }$ , its Jacobian satisfies $\begin{array} { r } { \frac { { \partial r _ { H } } } { { \partial z _ { L } } } = 0 } \end{array}$ . Therefore, for any diferentiable downstream predictor $Y = \Psi ( \mathrm { C o n t e x t } , r _ { H } )$ whose context is held fixed, we have $\begin{array} { r } { \frac { { \bf \dot { \sigma } } _ { \partial Y } } { \partial z _ { L } } = 0 } \end{array}$ . Therefore, perturbations confined to the variant latent factor $z _ { L }$ cannot propagate to � through the learned invariant representation $r _ { H }$ □

## C Synthetic Evaluation Additional Details

## C.1 Synthetic Setting Details

We consider an 8-dimensional latent state $z = [ z _ { H } ^ { \top } , z _ { L } ^ { \top } ] ^ { \top }$ , where $z _ { H } , z _ { L } \in \mathbb { R } ^ { 4 }$ are independently distributed as ${ \cal N } ( 0 , I _ { 4 } )$ . Observations are generated as $\overset { \_ } { h } = f ( z )$ using a four-layer smooth nonlinear coupling map with coupling strength 0.5. For predictive training, we generate stationary Gaussian OU pairs with correlation $\rho = 0 . 5$ . The encoder is trained with the reconstruction-free alignment– SIGReg objective for 10,000 updates using batch size 256, learning rate $3 \times 1 0 ^ { - 3 }$ , and SIGReg weight $1 0 ^ { - 6 }$ . The encoder is then frozen for all subsequent experiments. No ground-truth latent variables, decoder, or reconstruction objective is used during training. For each trained encoder, we first fit a centered ZCA whitening transformation using 30,000 unlabeled representations. Let ¯� denote the resulting whitened representation. We then construct $N = 3 0 { , } 0 0 0$ matched invariant pairs and define $\Delta \bar { r } _ { i } = \bar { \bar { r } } _ { i } ^ { ( 1 ) } - \bar { r } _ { i } ^ { ( 2 ) }$ To match the suficient-variation condition in Assumption 1, we estimate the uncentered second-moment matrix $\begin{array} { r } { \widehat { M } _ { \Delta r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Delta \bar { r } _ { i } \Delta \bar { r } _ { i } ^ { \top } } \end{array}$ . Under the exact block model, invariant correspondences satisfy $\Delta z _ { H } = 0$ , so the null space of the population diference moment is the invariant subspace. We therefore recover the invariant block from the bottom $d _ { H }$ eigenspace of $\widehat { M } _ { \Delta r }$ and use its orthogonal complement as the $d _ { L }$ -dimensional variant subspace. The resulting spectral transformation is applied to the frozen whitened representation.

SplitJEPA uses the correctly matched invariant pairs, LeJEPA retains the centered ZCA representation without the spectral rotation, and Shufled Pairs randomly permutes the second member of each pair, destroying the shared-invariant correspondence. We omit No Diversity as a separate numerical baseline because it is operationally identical to LeJEPA in the present implementation. Specifically, setting the paired variant states equal gives $\Delta z _ { L } \ = \ 0$ , so the diference covariance contains no informative direction for spectral recovery. The resulting recovery transformation defaults to the identity head and produces exactly the LeJEPA representation. This is an implementation-specific realization of the boundary case in which the full-rank variation assumption fails; the invariant subspace is not uniquely determined by the pair diferences.

Evaluation uses independent sets of 15,000 probe samples, 15,000 held-out samples, and 15,000 held-out invariant pairs. We report Global $\dot { R } ^ { 2 }$ for linear recovery of the complete latent state, block-level $R ^ { 2 }$ for the invariant and variant subspaces, and the cross-block leakage terms $\| A _ { 1 2 } \| _ { F }$ and $\| A _ { 2 1 } \| _ { F }$ . For comparisons across latent dimensions, we normalize the leakage terms as $\| A _ { 1 2 } \| _ { F } / \sqrt { d _ { H } }$ and $\| A _ { 2 1 } \| _ { F } / \sqrt { d _ { L } }$ . Block purity summarizes separation between the two recovered blocks, defined as Purity $\begin{array} { r } { ( A ) = 1 - \frac { \| A _ { 1 2 } \| _ { F } ^ { 2 } + \| A _ { 2 1 } \| _ { F } ^ { 2 } } { \| A \| _ { F } ^ { 2 } } } \end{array}$ , where a value of one indicates perfect block separation. Main comparisons are aggregated over five seeds (0–4); the single-seed scaling experiments are intended only as finite-sample diagnostics.

## C.2 Loss Ablations

We isolate the three mechanisms associated with the identifiability pipeline. The Full model uses predictive alignment and SIGReg during encoder training, centered ZCA normalization, and the matched-pair spectral head. For w/o Predictive, the encoder is trained using SIGReg alone, with the OU-pair alignment term removed. For w/o Invariance, we retain the full encoder and ZCA normalization but remove the matched-pair spectral rotation. For w/o Variance, we remove both SIGReg during encoder training and ZCA normalization during head fitting, retaining only centering, predictive alignment, and matched-pair recovery, testing the variance-preserving feasible geometry implemented by SIGReg and ZCA. Because this ablation removes two coupled components, it evaluates their combined contribution to variance-preserving geometry.

As shown in Fig. 5, removing predictive alignment decreases Global $R ^ { 2 }$ . The invariant and variant scores remain relatively high, but they are no longer meaningful evidence of Theorem 2 because the preceding global-identification condition has failed. Without matched-pair recovery the complementary failure mode is produced: Global $R ^ { 2 }$ remains $0 . 9 9 7 7 \pm 0 . 0 0 1 6$ , whereas Invariant and Variant $R ^ { \tilde { 2 } }$ fall, and the leakage terms increase to $0 . 3 3 2 0 \pm 0 . 2 1 5 2$ and $0 . 3 3 0 0 \pm 0 . 2 1 4 9$ . Removing the variance preservation leaves the mean block scores near those of the full model but increases the covariance error and increases the variability of $A _ { 2 1 }$ . Thus, in this synthetic setting, predictive and invariance information determine the two recovery stages, while variance preservation mainly improves normalization and finite-sample stability.

![](images/77cc87b7a1c86381092102696271db7e7519b5e8a9ca826faa329f5f1e4f0594.jpg)

![](images/b2c20cfe165da3d3360d429f44cf2f80805a1283fbfba41349fce951c2ebb913.jpg)

![](images/08876608c03f863b3f485929741c05d417141dcc3a4643e5b09dce3cbfd0cf4e.jpg)  
Figure 7: Global linear identification under three scaling regimes. Higher Global $R ^ { 2 }$ indicates more accurate linear identification of the complete latent representation.

## C.3 Scaling Diagnostics

We conduct three scaling studies by varying the latent dimension, the predictive training budget, and the sample budget used for whitening and spectral-head fitting. In each of the scaling figures, panels (a)–(c) correspond to latent-dimension scaling, encoder training-sample scaling, and head-fitting sample scaling, respectively. Every panel compares Ours, LeJEPA, and Shufled Pairs (except for Global $R ^ { 2 } ,$ , which remains consistent because of the experiment setting). All curves use seed 0 to diagnose trends. Together, these diagnostics characterize how predictive training and finite-sample moment estimation approach the population recovery conditions as their respective data budgets increase.

Latent-dimension scaling. We vary the total latent dimension over � ∈ {4, 8, 16, 32, 64}, $d _ { H } =$ $d _ { L } \ = \ d / 2$ , and increase the representation dimension accordingly while keeping the remaining settings fixed. Panel (a) of Figs. 7–11 shows that Global $R ^ { 2 }$ remains high and exceeds 0.9999 for $d \ge 1 6 ,$ . For our method, both block-level $R ^ { 2 }$ values remain above 0.994 and normalized $\| A _ { 1 2 } \| _ { F }$ remains below $1 . 5 \times 1 0 ^ { - 3 }$ . Normalized $\| A _ { 2 1 } \| _ { F }$ , however, increases from 0.0053 at $d = 4$ to 0.0325 at $d = 6 4$ . This asymmetry indicates that increasing dimension has little efect on global recovery but makes estimation of the complementary block more sensitive.

Encoder training-sample scaling. We vary the number of online OU training pairs over $\{ 2 . 5 6 \times$ $1 0 ^ { 5 } , 6 . 4 0 \times 1 0 ^ { 5 } , \tilde { 1 . } 2 8 \times \hat { 1 } 0 ^ { 6 } , 2 . 5 6 \times 1 0 ^ { 6 } , 5 . 1 2 \dot { \times } 1 0 ^ { 6 } \}$ }, while keeping the latent dimension, observation map, optimization configuration, and downstream recovery procedure fixed. As shown in panel (b) of Figs. 7–11, Global $R ^ { 2 ^ { \circ } }$ increases with the training budget, with a clear transition around $1 . 2 8 \times 1 0 ^ { \dot { 6 } }$ pairs and limited improvement beyond $2 . 5 6 \times 1 0 ^ { \overline { { 6 } } }$ The block-level leakage decreases in parallel, indicating that insuficient predictive training primarily manifests as imperfect recovery of the global representation.

Head-fitting sample scaling. With the predictive encoder fixed, we vary the common sample budget for ZCA whitening and matched-pair recovery over {250, 500, 1000, 3000, 10000, 30000}. Panel (c) of Figs. 7–11 shows a strong reduction in normalized $\| A _ { 2 1 } \| _ { F }$ , from 0.156 with 250 samples to 0.012 with 30000 samples, while normalized $\| A _ { 1 2 } \| _ { F }$ remains below 0.005 throughout. The Global and block-level $R ^ { 2 }$ curves remain comparatively stable. This performance indicates that the matched-pair covariance already identifies the invariant nullspace reliably, while more samples are primarily needed to estimate whitening and the complementary block accurately.

## C.4 Efficiency at Matched Recovery Quality

Eficiency Protocol. We compare the end-to-end eficiency of our reconstruction-free implementation with V3 (Wu et al., 2025), a content-style method based on variance-invariance constraints. Both implementations use the same latent dimension, nonlinear observation map, five random seeds and 256 atomic observations per optimization step. Each method receives its native grouping structure: SplitJEPA uses matched pairs with shared content and varying style, whereas V3 uses groups with shared style and varying content. The two protocols share the same marginal latent distribution and total observation budget. We evaluate the content and style branches of V3 against $z _ { H }$ and $z _ { L }$ , respectively. Runtime was measured using a monotonic wall-clock timer. For SplitJEPA, it includes data generation, predictive training, whitening, and matched-pair block recovery. For V3, it includes data generation, encoder-decoder training, vector quantization, and all variance-invariance losses. Model construction and held-out evaluation are excluded for both methods, and no training iterations are discarded as warm-up.

![](images/b56a1326f4f99f8bafb1b00b2d333e7d2251df9c11d219af845f1ad1228fa522.jpg)

![](images/d97b753151e337f7dccc7f03875c4bddbc996412e9fc4cedfef89627977468be.jpg)

![](images/e2e4fdc560ca2a2423ab65d59933b454983f7018a580133aa13a99463687a66f.jpg)  
Figure 8: Recovery of the invariant latent block under three scaling regimes. Invariant $R ^ { 2 }$ measures the linear recoverability of the ground-truth invariant factors from the learned invariant representation.

![](images/655adb0aab26502232f8cc7470477806fcba2d038919c0599f8c511645ab97fd.jpg)

![](images/15c659cca2d05b1fc7ec50438ca610b5a58493b177df248ce5225858725b1964.jpg)

![](images/bc9922f18e5389eaa6e2bc82182e089ce78e53fd4c34f89e3909e9513cb51351.jpg)  
Figure 9: Recovery of the variant latent block under three scaling regimes. Variant $R ^ { 2 }$ measures the linear recoverability of the ground-truth variant factors from the learned variant representation.

We evaluate both methods at 5%, 10%, 20%, 40%, 60%, 80%, and 100% of a 10,000-step training budget. The common recovery target is ${ \frac { 1 } { 2 } } \big ( R _ { \mathrm { I n v a r i a n t } } ^ { 2 } +$ $R _ { \mathrm { V a r i a n t } } ^ { 2 } ) \geq 0 . 9 9$ . Every run completes the full training budget. We report the cumulative runtime and observation count at the first evaluated checkpoint that reaches the target. Runs that never reach it are retained and assigned their full-budget runtime and observation count, yielding capped time-to-target and observation-cost metrics. Checkpoint evaluation time is excluded for both methods.

![](images/4bb42053a97a110c5c47bd1cd63e571dd25e34b452e9b760a2840be6d92adb77.jpg)

Figure 12: Simulation eficiency comparison. Eficiency Results. As shown in Fig. 12, SplitJEPA reaches the recovery target in all five runs, whereas V3 reaches it in three. The capped time-to-target is $3 . 1 6 \pm 1 . 1 4$ seconds for SplitJEPA and $8 . 5 4 \pm 3 . 0 5$ seconds for V3, corresponding to a 2.70× lower capped runtime for SplitJEPA. The diference in observation cost is larger: SplitJEPA uses $0 . 3 0 7 \pm \mathrm { \bar { 0 . 1 1 4 } }$ million observations under the same capped protocol, compared with $1 . 9 4 6 \pm 0 . 6 6 8$ million for V3, a 6.33× reduction. The implementations contain 64 and 272 trainable parameters, respectively. The result demonstrates that SplitJEPA achieves a higher target attainment rate with lower capped runtime, lower observation cost, and fewer trainable parameters.

![](images/39a5d07ca5bded6b269c1e59b4f69ac9d784f6a9f349746e447399baeebe302c.jpg)

![](images/40ab639439e1bb582abc9cc42a74045bab45dcc0a672cfefd48847420f08045d.jpg)

![](images/df1d68c0fad2dc9832fdb791368d00d743e580393b515f6aadc543141ba231d3.jpg)  
Figure 10: Normalized cross-block leakage $A _ { 1 2 }$ under three scaling regimes. The normalized $A _ { 1 2 }$ score quantifies leakage from the variant latent block into the recovered invariant block and is defined as $\bar { \| } A _ { 1 2 } \| _ { F } / \sqrt { d _ { H } }$

![](images/508743d14cad38d87a970095df229cc76291b5cb28a556221153d7a15f45a921.jpg)

![](images/412934c5356d64e61daea8371d32c415b4697d38b1b7686596c76a589767881b.jpg)

![](images/b3170db295c7848b96cdd75e740bd3ea0ef1f540d9fe6621a5071f4c5f9db9dd.jpg)  
Figure 11: Normalized cross-block leakage $A _ { 2 1 }$ under three scaling regimes. The normalized $A _ { 2 1 }$ score quantifies leakage from the invariant latent block into the recovered variant block and is defined as $\bar { | | } { A _ { 2 1 } } { \| _ { F } } / { \sqrt { d _ { L } } }$

## C.5 Additional Robustness

Full-coordinate Nonlinear MLP. To test the block-recovery mechanism beyond the default coupling structure, we concatenate the latent blocks, apply a dense orthogonal afine transformation that mixes the two blocks, and then apply the elementwise map $\eta ( u ) = u + \alpha \operatorname { t a n h } ( u )$ for $\alpha \geq 0$ Because $\eta ^ { \prime } ( u ) = 1 + \alpha \mathrm { s e c h } ^ { 2 } ( u ) \geq 1$ , the elementwise map is smooth and injective. We use the same reconstruction-free training and block-recovery pipeline as in the main synthetic experiment. After training, the encoder is frozen and all block controls are applied to the same frozen encoder for each seed. Table 3 reports the five-seed results. Our recovery achieves the best invariant and variant $R ^ { 2 }$ values, with $\| \bar { A } _ { 1 2 } \| _ { F } = 0 . 0 0 5 6 \pm 0 . 0 0 0 7 , \| A _ { 2 1 } \| _ { F } = 0 . 0 1 8 7 \pm 0 . 0 0 3 7$ . In contrast, the LeJEPA condition retains substantially larger cross-block leakage despite using the same frozen encoder. Shufled Pairs further degrades both block alignment and purity. These results show that the proposed representation-only recovery remains efective under joint nonlinear mixing, provided that the preceding predictive encoder reaches a suficiently informative global representation.

Coupling-strength sensitivity. We further vary the nonlinear coupling strength in the default synthetic configuration while keeping the training and recovery procedure fixed. We consider coupling strengths 0.5, 1.0 and 2.0. Table 4 reports the resulting global identification and block-structure measures. Increasing the coupling strength reduces Global ${ \bf \ddot { \boldsymbol { R } } } ^ { 2 }$ and increases the orthogonality error, while the normalized cross-block terms become 0.0572 and 0.0210. At strength 2.0, Global $R ^ { 2 }$ further decreases to 0.8765, with orthogonality error and cross-block terms further increasing. Theorem 2 conditions on the predictive encoder attaining the globally identifiable linear solution class. This experiment shows how departures from that geometry under finite optimization appear jointly as increased orthogonality error and cross-block leakage.

Table 3: Reconstruction-free block recovery underjoint nonlinear MLP mixing (mean ± standard deviation over five seeds). All methods reuse the same frozen MLP encoder and difer only in the information available during spectral recovery. Global $R ^ { 2 }$ is approximately 0.977 for all methods.
<table><tr><td>Method</td><td>Invariant  $R ^ { 2 }$ </td><td>Variant  $R ^ { 2 }$ </td><td> $\| A _ { 1 2 } \| _ { F }$ </td><td> $\| A _ { 2 1 } \| _ { F }$ </td><td>Block Purity</td></tr><tr><td>LeJEPA</td><td> $0 . 6 7 7 8 \pm 0 . 0 3 6 0$ </td><td> $0 . 6 7 6 9 \pm 0 . 0 3 8 7$ </td><td> $1 . 0 9 4 0 \pm 0 . 0 6 9 6$ </td><td> $1 . 0 9 1 0 \pm 0 . 0 7 1 0$ </td><td> $0 . 6 9 3 6 \pm 0 . 0 3 9 3$ </td></tr><tr><td>Shuffled Pairs</td><td> $0 . 6 5 9 8 \pm 0 . 0 5 5 5$ </td><td> $0 . 6 6 0 6 \pm 0 . 0 5 6 7$ </td><td> $1 . 1 2 4 3 \pm 0 . 0 9 9 7$ </td><td> $1 . 1 2 2 0 \pm 0 . 1 0 1 3$ </td><td> $0 . 6 7 5 0 \pm 0 . 0 5 8 1$ </td></tr><tr><td>SplitJEPA (Ours)</td><td> $\mathbf { 0 . 9 7 7 0 \pm 0 . 0 0 0 3 }$ </td><td> $\mathbf { 0 . 9 7 6 9 \pm 0 . 0 0 0 2 }$ </td><td> $\mathbf { 0 . 0 0 5 6 \pm 0 . 0 0 0 7 }$ </td><td> $\mathbf { 0 . 0 1 8 7 \pm 0 . 0 0 3 7 }$ </td><td> $\mathbf { 0 . 9 9 9 9 5 } \pm \mathbf { 0 . 0 0 0 0 } 2$ </td></tr></table>

Table 4: Sensitivity to nonlinear coupling strength. Seed-0 diagnostic under the default synthetic configuration. Increasing the coupling strength progressively degrades the global identification stage.
<table><tr><td>Coupling Strength</td><td>Global  $R ^ { 2 }$ </td><td> $\| Q ^ { \top } Q - I \| _ { F } / \sqrt { d }$ </td><td> $\| A _ { 1 2 } \| _ { F }$ </td><td> $\| A _ { 2 1 } \| _ { F }$ </td></tr><tr><td>0.5</td><td>0.9979</td><td>0.0199</td><td>0.0026</td><td>0.0155</td></tr><tr><td>1.0</td><td>0.9565</td><td>0.0586</td><td>0.0572</td><td>0.0210</td></tr><tr><td>2.0</td><td>0.8765</td><td>0.1507</td><td>0.1393</td><td>0.0289</td></tr></table>

## D Robotic Manipulation Experiments Additional Details

## D.1 Robotic Manipulation Setting Details

Data Processing. We evaluate PickCube-v1<sup>1</sup> and PushCube-v1<sup>2</sup> in ManiSkill3. PickCube requires the robot to grasp a cube, transport it in 3 dimensions, and place it at a target pose, whereas PushCube requires the robot to establish contact and move the cube into a planar goal region. The visual input is a 128 × 128 RGB image from the third-person base camera. At each control step, the policy receives the four most recent images and predicts an 8-dimensional action. The same manipulation dynamics, object initialization distribution, action space, controller, and episode horizon are used for ID and OOD evaluation; only the specified visual nuisance is changed. For each task, the ofline dataset contains 800 successful trajectories organized into 100 matched groups of eight. Trajectories within a group share object identity, initial object pose, and goal pose, while difering in robot initialization, execution trajectory, approach or contact direction, and timing. We use two forms of correspondence. Cross-view pairs render the same physical state from the fixed base and side cameras and therefore provide state-matched visual variation. Cross-trajectory pairs are weaker correspondences: frames are approximately aligned by normalized execution progress and preserve group-level task content. These pairs broaden the coverage of trajectory variation while relaxing exact state alignment, allowing the diversity ablation to examine the trade-of between variation coverage and correspondence precision. We reserve 20% of the trajectories within each group for validation.

Representation Learning. The robotic implementation follows the same identification decomposition as the synthetic study: prediction in latent space preserves the evolving state, normalization regularizes the geometry of the complete representation, and matched correspondences determine the invariant block. The controlled synthetic setting estimates the final block transform explicitly through ZCA whitening and eigendecomposition. In the visual robotic setting, we optimize counterparts of these components jointly. Both tasks use an ImageNet-initialized ResNet-50 encoder operating on 128 × 128 RGB observations. The encoder produces a partitioned representation $r = \left[ r _ { H } ; r _ { L } \right]$ , with $d _ { H } = d _ { L } = 6 4$ for PickCube and $d _ { H } = d _ { L } =$ 128 for PushCube. For robotic representation learning, an action-conditioned predictor maps $\left( \boldsymbol { r } _ { t } , \boldsymbol { a } _ { t } \right)$ to a stop-gradient target $r _ { t + 1 }$ , while the invariance loss is applied only to $r _ { H }$ . Variance and covariance regularization are used to prevent representation collapse. Each optimization step combines a batch of 32 adjacent temporal transitions with an independently shufled batch of 32 matched pairs. Cross-trajectory and cross-view pairs are pooled for training, so their relative frequency follows the number of available pairs. Random pairs with diferent matched-group identities are excluded from the main training objective and are used only as diagnostic controls and in the shufled-correspondence ablation. Held-out cross-trajectory and cross-view pairs are used for block validation. We optimize with AdamW using an initial learning rate of $3 \times \dot { 1 } 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 }$ , cosine learning-rate decay, and gradient clipping at norm 10. PickCube is trained for 200 epochs and PushCube for 100 epochs. Ours-�<sub>�</sub> exposes only the learned invariant block to the downstream policy, whereas Ours- $\cdot r _ { H } + r _ { L }$ exposes the complete partitioned representation.

![](images/1bb689d4415b81d602519256de5562d623a8061de2d74a47d325605d9501246e.jpg)  
Figure 13: Sample eficiency of learned representations. Performance of diferent representations under varying amounts of training data on PushCube and PickCube, evaluated under ID and OOD settings.

Downstream Policies and Evaluation. All downstream policies are trained by behavior cloning on ID demonstrations. The policy head is a two-layer MLP with hidden dimensions 512 and 256. Raw learns directly from RGB observations, R3M uses a frozen pretrained R3M visual feature, and LeJEPA uses a frozen reconstruction-free representation without the proposed block decomposition. These method-specific visual frontends are retained throughout the comparison, and the demonstration budget, temporal context, action target, and evaluation protocol are held fixed. Each trained policy is evaluated on 100 closed-loop episodes under one ID and three OOD visual conditions. We report SR@10cm as the fraction of episodes in which the object reaches within 10 cm of the goal at any time, using the three-dimensional object-goal distance for PickCube and the planar distance for PushCube. GRASP is the fraction of PickCube episodes in which the environment reports the object as grasped at least once, while CONTACT is the fraction of PushCube episodes in which the TCP comes within 4.5 cm of the object at least once, serving as a consistent proxy for contact. Camera OOD rotates the base-camera azimuth by $1 0 ^ { \circ }$ , and the reported value averages the paired −10<sup>◦</sup> and $+ 1 0 ^ { \circ }$ conditions. Lighting OOD scales the intensity of the oficial scene lighting to 0.6, while Visual OOD combines the camera and lighting interventions. The same physical reset seeds are shared across methods within each task, and no policy is fine-tuned or adapted using OOD observations. Unless stated otherwise, results are the mean ± standard deviation over three downstream training seeds.

## D.2 Robustness Experiments

Sample Eficiency. We examine whether the advantage of the learned representations persists when the amount of downstream supervision is reduced. We vary the fraction of behavior-cloning demonstrations from 10% to 100%. As shown in Figure 13, the proposed representations remain efective across training budgets with the clearest advantage appearing under visual distribution shifts. On PushCube, both Ours- $\cdot r _ { H }$ and Ours- $\cdot r _ { H } + r _ { L }$ already achieve above 90% ID SR@10cm with only 10% of the demonstrations. Their Camera and Visual OOD performance further improves as the training budget increases. In contrast, the baseline representations either require substantially more demonstrations to approach the ID performance of our method or remain sensitive to the visual shifts. PickCube exhibits a stronger dependence on the amount of policy-training data because the task requires grasping, transporting, and accurately approaching the goal. The ID performance of baseline methods increases with data volume, but under OOD conditions most baselines do not improve with more data, instead reaching a low-performance plateau under these visual shifts. Nevertheless, our two representations consistently outperform the baselines across the evaluated budgets. Neither of our representation inputs dominates uniformly: Ours- $r _ { H }$ is preferable in several low-data or camera-shift settings, whereas Ours- $\cdot r _ { H } + r _ { L }$ becomes competitive or stronger when additional execution-specific information is useful. Overall, the results show that the recovered blocks remain usable under limited downstream supervision and that the robustness improvement is not restricted to the full-data condition.

Table 5: Performance retention under representation interventions
<table><tr><td></td><td colspan="3">PushCube</td><td colspan="3">PickCube</td></tr><tr><td>Intervention</td><td> $r _ { H } + r _ { L }$ </td><td> $r _ { H }$ </td><td> $r _ { H }$  advantage</td><td> $r _ { H } + r _ { L }$ </td><td> $r _ { H }$ </td><td> $r _ { H }$  advantage</td></tr><tr><td>Zero rL</td><td> $9 3 . 7 \pm 3 . 4 \%$ </td><td> $9 9 . 9 \pm 0 . 2 \%$ </td><td> $+ 6 . 1 \mathsf { p p }$ </td><td> $6 1 . 4 \pm 5 . 1 \%$ </td><td> $9 9 . 7 \pm 0 . 5 \%$ </td><td> $+ 3 8 . 4 \mathrm { p p }$ </td></tr><tr><td>Noise  $0 . 5 \sigma$ </td><td> $9 9 . 0 \pm 1 . 0 \%$ </td><td> $1 0 0 . 0 \pm 0 . 0 \%$ </td><td> $+ 1 . 0 \mathrm { p p }$ </td><td> $8 7 . 9 \pm 0 . 8 \%$ </td><td> $1 0 0 . 0 \pm 0 . 0 \%$ </td><td> $+ 1 2 . 1 \ \mathsf { p p }$ </td></tr><tr><td>Noise  $1 . 0 \sigma$ </td><td> $9 0 . 2 \pm 0 . 5 \%$ </td><td> $9 9 . 9 \pm 0 . 2 \%$ </td><td> $+ 9 . 7 \mathrm { p p }$ </td><td> $6 2 . 2 \pm 4 . 6 \%$ </td><td> $1 0 0 . 0 \pm 0 . 0 \%$ </td><td> $+ 3 7 . 8 \mathrm { \ p p }$ </td></tr><tr><td>Noise 2.0σ</td><td> $4 4 . 1 \pm 4 . 0 \%$ </td><td> $9 9 . 9 \pm 0 . 2 \%$ </td><td> $+ 5 5 . 8 \mathrm { p p }$ </td><td> $2 9 . 6 \pm 0 . 8 \%$ </td><td> $1 0 0 . 0 \pm 0 . 0 \%$ </td><td> $+ 7 0 . 4 \mathrm { p p }$ </td></tr></table>

![](images/c273aac10d45ab3453d9c23246ccb8cc403cc4c1c97c526f997d77c720bc5f4c.jpg)

![](images/d3b774ccb1194362b33878af8eb67b8c8eb26a840489752851b20fc569f3523b.jpg)  
Figure 14: Empirical validation of invariant-variant block specialization. Representation responses to matched viewpoint and trajectory variations on PickCube and PushCube. The learned invariant block $r _ { H }$ changes substantially less than the variant block $r _ { L }$ under the same variations.

Block Validation. To verify whether our $r _ { H }$ and $r _ { L }$ exhibit the intended diferential sensitivity, we measure the representation distance separately in $r _ { H }$ and $r _ { L }$ normalized by the corresponding distance $\begin{array} { r } { D _ { B } ( x , x ^ { \prime } ) = \frac { 1 } { \sqrt { d _ { B } } } \left\| \frac { r _ { B } ( x ) - r _ { B } ( x ^ { \prime } ) } { \sigma _ { B } } \right\| _ { 2 } } \end{array}$ between randomly paired observations for each pair type, where $\sigma _ { B }$ is the coordinate-wise standard deviation estimated from the held-out pair endpoints. Figure 14 shows a consistent separation between the two blocks. On PickCube, the normalized response to a viewpoint change is 0.18 in $r _ { H }$ and 1.51 in $r _ { L }$ , making the response of the variant block 8.4× larger. Across matched trajectories, the corresponding distances yield a 2.1× diference. The same pattern is more pronounced on PushCube: the viewpoint and the matched-trajectory responses correspond to 14.0× and 3.1× larger changes in $r _ { L }$ , respectively. These results provide a representation validation that is distinct from downstream $\operatorname { s R } .$ . The invariant block remains comparatively stable when viewpoint and execution factors change, whereas the variant block responds more strongly to the same interventions. It is consistent with the intended block structure in which task-level information is retained in $r _ { H }$ while viewpoint and trajectory variation is preferentially expressed through $r _ { L }$

Controlled Intervention. To evaluate the robustness implication of Theorem 3, we compare two representation interfaces: one exposes only the invariant block $r _ { H } .$ , while the other exposes both $r _ { H }$ and the variant block $r _ { L }$ . We hold the environment state and all other policy inputs fixed and intervene only on $r _ { L }$ . Across both tasks, the $r _ { H }$ -only policy retains approximately 100% of its clean performance, whereas the $r _ { H } + r _ { L }$ policy becomes progressively more sensitive as the magnitude of corruption increases. Table 5 shows that routing predictions through $r _ { H }$ provides a robust interface to perturbations confined to $r _ { L }$ , while exposing the variant block introduces an additional pathway through which such perturbations can afect downstream decisions. These results support Theorem 3 that once the invariant block is separated from variation, predictions based only on $r _ { H }$ are substantially more robust to perturbations of the variant component.

Table 6: Robot manipulation representation ablation results. Values report the mean $\pm$ standard deviation over three training seeds. All variants retain the same partitioned architecture and downstream pipeline as the Full model and remove only the indicated component. ID is evaluated using SR@10cm and the task-specific interaction rate (CONTACT for PushCube and GRASP for PickCube); all OOD columns report SR@10cm.
<table><tr><td rowspan="3">Method</td><td colspan="5">PushCube</td><td colspan="5">PickCube</td></tr><tr><td colspan="2">ID</td><td colspan="3">OOD</td><td colspan="2">ID</td><td colspan="3">OOD</td></tr><tr><td>SR@10cm CONTACT</td><td></td><td>Camera</td><td>Lighting</td><td>Visual</td><td>|SR@10cm</td><td>GRASP</td><td>Camera</td><td>Lighting</td><td>Visual</td></tr><tr><td>Full (rH)</td><td> $9 9 . 3 \pm 0 . 6 $ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $7 1 . 5 \pm 1 0 . 4$ </td><td> $9 9 . 7 \pm 0 . 6 $ </td><td> $6 4 . 5 \pm 1 2 . 7$ </td><td> $4 5 . 0 \pm 2 . 4$ </td><td> $9 7 . 0 \pm 1 . 4$ </td><td>30.3 ± 6.1</td><td> $4 2 . 7 \pm 4 . 5$ </td><td> $2 8 . 3 \pm 3 . 9$ </td></tr><tr><td>Full  $( r _ { H } + r _ { L } )$ </td><td> $9 9 . 3 \pm 0 . 6 $ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $7 5 . 3 \pm 8 . 1 $ </td><td> $9 8 . 3 \pm 1 . 5$ </td><td> ${ \bf 7 0 . 2 \pm 1 2 . 9 }$ </td><td> ${ \bf 4 6 . 3 \pm 4 . 9 }$ </td><td> $9 6 . 7 \pm 0 . 5$ </td><td> $2 7 . 7 \pm 0 . 9$ </td><td> $4 3 . 3 \pm 2 . 9$ </td><td> ${ \bf 3 0 . 3 \pm 1 . 7 }$ </td></tr><tr><td>w/o Predictive (rH)</td><td> $9 8 . 7 \pm 0 . 6 $ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $6 5 . 7 \pm 6 . 4$ </td><td> $9 9 . 7 \pm 0 . 6 $ </td><td> $5 4 . 0 \pm 1 4 . 3$ </td><td> $3 9 . 7 \pm 4 . 5$ </td><td> $9 2 . 7 \pm 1 . 2$ </td><td> $2 0 . 7 \pm 0 . 9$ </td><td> $3 5 . 7 \pm 3 . 9$ </td><td> $1 9 . 0 \pm 0 . 8$ </td></tr><tr><td>w/o Predictive  $( r _ { H } + r _ { L } )$ </td><td> $9 8 . 7 \pm 0 . 6 $ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $6 5 . 3 \pm 8 . 8$ </td><td>98.0 ± 1.0</td><td> $5 9 . 8 \pm 1 2 . 4$ </td><td> $3 9 . 3 \pm 2 . 5$ </td><td> $9 4 . 0 \pm 1 . 4$ </td><td>18.3 ± 1.7</td><td> $3 9 . 3 \pm 3 . 4$ </td><td> $1 8 . 7 \pm 1 . 2$ </td></tr><tr><td>w/o Invariance  $\left( r _ { H } \right)$ </td><td> $9 8 . 7 \pm 0 . 6 $ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $1 9 . 7 \pm 9 . 5$ </td><td> $8 8 . 7 \pm 6 . 7 $ </td><td> $1 5 . 7 \pm 1 1 . 0$ </td><td> $4 4 . 3 \pm 1 . 2$ </td><td> $9 2 . 0 \pm 4 . 2 $ </td><td> $2 4 . 0 \pm 3 . 6$ </td><td> $4 5 . 3 \pm 1 . 9$ </td><td> $1 5 . 3 \pm 0 . 9$ </td></tr><tr><td>w/o Invariance  $\left( r _ { H } + r _ { L } \right)$ </td><td> $9 8 . 0 \pm 1 . 0 $ </td><td> $9 9 . 7 \pm 0 . 6 $ </td><td> $1 5 . 7 \pm 1 0 . 1$ </td><td> $9 3 . 3 \pm 1 . 5$ </td><td> $6 . 5 \pm 4 . 5$ </td><td> $4 1 . 0 \pm 3 . 7$ </td><td> $9 2 . 3 \pm 0 . 9$ </td><td> $2 4 . 0 \pm 3 . 3$ </td><td> $4 1 . 7 \pm 3 . 4$ </td><td> $1 3 . 0 \pm 0 . 8$ </td></tr><tr><td>w/o Diversity  $\left( r _ { H } \right)$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $7 8 . 0 \pm 4 . 8$ </td><td> $9 9 . 7 \pm 0 . 6 $ </td><td> $6 7 . 0 \pm 9 . 5$ </td><td> $4 5 . 0 \pm 2 . 8$ </td><td> ${ \bf 9 9 . 0 \pm 0 . 0 }$ </td><td> $2 1 . 7 \pm 1 . 2$ </td><td> $4 4 . 0 \pm 1 . 4$ </td><td> $1 8 . 7 \pm 3 . 1 $ </td></tr><tr><td>w/o Diversity  $( r _ { H } + r _ { L } )$ </td><td> ${ \bf 1 0 0 . 0 \pm 0 . 0 }$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> ${ \bf 7 9 . 3 \pm 6 . 8 }$ </td><td>99.7 ± 0.6</td><td> $6 7 . 8 \pm 1 5 . 3$ </td><td> $4 3 . 3 \pm 0 . 5$ </td><td>98.0 ± 0.8</td><td> $2 1 . 3 \pm 2 . 9$ </td><td> ${ \bf 4 7 . 0 \pm 3 . 3 }$ </td><td> $2 0 . 3 \pm 2 . 1$ </td></tr></table>

![](images/bfaa0c20e56e66193b7b62dc69252f1cccf305eefa052d341b3d981480fa981a.jpg)  
Figure 15: Temporal transferability. Normalized representation changes between state-matched observations from diferent trajectories and timesteps averaged over three seeds.

Temporal Transferability We evaluate whether $r _ { H }$ transfers more consistently than $r _ { L }$ across execution time using state-matched asynchronous pairs. Each pair is drawn from diferent validation trajectories in the same $H .$ -group and at diferent normalized timesteps. Simulator states are used only to construct the pairs, while the encoder receives observations alone. We group the pairs into small, medium and large normalized timestep gaps and measure each block’s representation change relative to its random-pair distance. For each block, we measure its standardized representation distance $D _ { B }$ between the paired observations and normalize it by the mean distance between observations with diferent �: $\begin{array} { r } { S _ { B } ( \nu ) = \frac { \mathbb { E } _ { ( x _ { a } , x _ { b } ) \sim \nu } [ D _ { B } ( x _ { a } , x _ { b } ) ] } { \mathbb { E } _ { ( x _ { a } , x _ { b } ) \sim \mathrm { r a n d o m } } [ D _ { B } ( x _ { a } , x _ { b } ) ] } } \end{array}$ . A smaller $S _ { B }$ indicates that block � transfers more consistently across the tested temporal variation. We use $S _ { H } / S _ { L }$ as the relative stability measure, where a value below one indicates greater temporal stability in $r _ { H } . \mathrm { A l l }$ results are averaged over three independently trained seeds without additional training.

Figure 15 shows a consistent separation between the two representation blocks. Across both tasks and all temporal gaps, $r _ { H }$ changes less than $r _ { L }$ between state-matched. As the temporal gap increases, both representations exhibit larger changes, reflecting the increasing diference between observations collected farther apart in execution, while the representation shift in $r _ { H }$ is always smaller than that in $r _ { L }$ . The asynchronous pairs preserve the shared task condition and current physical state while changing the trajectory-specific temporal context. Accordingly, the relative stability of $r _ { H }$ indicates that it captures the information invariant across these observations, whereas the greater variation in $r _ { L }$ reflects its sensitivity to execution-specific factors. Since a downstream predictor operating on $r _ { H }$ encounters a smaller representation shift across timesteps, the learned invariant block provides a more transferable interface for prediction throughout task execution.