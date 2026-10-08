# Revisiting Explainable AI through Model-Independent Concept Dictionaries

Thomas Schnake<sup>d,e,</sup>∗, Doreen Schöppenthau<sup>c,a</sup>, Alexander Meyer<sup>c,a</sup>, Jacques Corbeil<sup>g,h</sup>, Klaus-Robert Müller<sup>a,b,f,i,</sup>∗, Grégoire Montavon<sup>a,c,</sup>∗

<sup>a</sup>BIFOLD – Berlin Institute for the Foundations of Learning and Data, Berlin, Germany <sup>b</sup>Machine Learning Group, Technische Universität Berlin, Germany <sup>c</sup>Institute for AI in Medicine (IKIM), Charité – Universitätsmedizin Berlin, Germany <sup>d</sup>Department of Chemistry, Chemical Physics Theory Group, University of Toronto, Toronto, Canada <sup>e</sup>Vector Institute for Artificial Intelligence, Toronto, Canada <sup>f</sup>Department of Artificial Intelligence, Korea University, Seoul, Korea <sup>g</sup>Department of molecular medicine, Université Laval, Québec, QC, Canada <sup>h</sup>Mila – Quebec Artificial Intelligence Institute, Canada <sup>i</sup>Max Planck Institute for Informatics, Saarbrücken, Germany

## Abstract

Modern applications of AI rely on increasingly complex models. Explainable AI (XAI) has emerged as a set of techniques aimed at improving model transparency. However, existing XAI methods typically assume input features to be inherently interpretable, or they rely on intermediate internal abstractions that are dificult to characterize and highly architecture-specific, hindering consistent use across models. To address these limitations, we propose DictXAI, a method that defines concepts directly in the input domain via a dictionary—a large, potentially overcomplete set of predefined elements, each carrying an interpretable meaning. Technically, DictXAI first computes a sparse code of the input and then attributes the model’s prediction to the associated dictionary elements. We demonstrate the actionable nature of DictXAI explanations, showing that they can attribute AI malfunctions (e.g., Clever Hans efects) directly to identifiable artifact patterns in the data, while fostering human-AI alignment on intricate biomedical signals. We further demonstrate our method’s ability to operate across a wide variety of dictionaries, including learned image bases, analytically defined waveforms for electrocardiography, and experimentally acquired dictionary elements. Overall, our results show that DictXAI provides more interpretable, actionable, and architecture-agnostic insights than classical XAI or existing concept-based approaches.

Keywords: Explainable AI (XAI), Concept-based explanations, Dictionary learning, Sparse coding, Model-agnostic interpretability

## 1. Introduction

Recent advances in artificial intelligence (AI) have produced models of unprecedented scale and complexity. Explainable AI (XAI) [1, 2] addresses the black-box nature of these models, aiming to reintroduce transparency into their prediction strategies—whether to meet regulatory demands [3] or to guide scientific inquiry [4, 5]. Although XAI methods can scale to such models, rigorously assessing explanation quality remains a multifaceted challenge [6, 7]. Explanations should not only be faithful to the model’s decision strategy, but also be presented in a format that is understandable to the explainee and actionable for relevant downstream tasks.

![](images/bf125a3b7d4492a9ff14efb0967bd701cbe732019d2fa84556401926de21af58.jpg)  
Figure 1: Overview of DictXAI. Left (Classical XAI): Standard workflows perform prediction and subsequently explain model decisions by computing attribution heatmaps directly on raw input pixels. Right (DictXAI, ours): Our approach introduces a predefined dictionary of concepts (here, multi-scale oriented Gabor filters). First, sources are inferred via sparse coding of the input ①. The input is then reconstructed via the decoder and passed to the classifier to predict ②, after which explanations are propagated back through the classifier and the decoder to the dictionary coeficients ③. This produces sparse explanations (where crosses denote zero attribution due to sparsity and red dot sizes indicate attribution magnitude) that are semantically interpretable along explicit axes (e.g., angle and scale) while preserving foundational XAI desiderata such as conservation and continuity.

In pursuit of more human-aligned explanations, concept-based XAI (e.g., [8, 9]) has emerged as a prominent direction. Its objective is to explain model behavior via high-level semantic units (referred to as “concepts”) rather than raw input features. In vision tasks, for example, concepts may include textures or object parts, each abstracting over diverse pixel arrangements. To identify such concepts, dominant paradigms typically probe internal representations by isolating neurons or activation subspaces that respond to specific inputs [9, 10]. Yet this reliance on intermediate layers faces fundamental limitations: neurons are frequently polysemantic [11], and representations are typically not comparable across architectures.

To overcome these limitations, we introduce DictXAI, an explanation framework that represents concepts directly in the input domain by projecting inputs onto a predefined, human-inspectable dictionary (see Figure 1). Depending on the application, dictionary elements may comprise mathematically parameterized primitives (e.g., Gabor filters or temporal wavelets with tunable scale and orientation), learned visual bases [12], and empirical reference spectra derived from laboratory measurements. While concept-based methods in latent space typically enforce orthogonality [13, 14], DictXAI introduces overcompleteness as a core design principle, enabling richly expressive concept extraction directly in the input domain. Furthermore, unlike recent autoencoder-based approaches [10], our framework avoids complex non-linear encoders; explanations are governed strictly by the original model and the explicit dictionary elements, preserving end-to-end transparency. Owing to this architectural simplicity, DictXAI naturally integrates with existing attribution techniques—such as Layer-wise Relevance Propagation (LRP) [15], Shapley values [16], and higher-order extensions [17]. DictXAI thus inherits the formal faithfulness and computational flexibility of these established methods while substantially elevating their interpretability and human alignment.

Through an application-grounded evaluation spanning multiple use cases, we demonstrate Dict-XAI’s advantages over classical XAI methods and existing concept-based extensions. First, we show its utility in purging a model of its reliance on spurious correlations, the so-called Clever Hans efect [18, 19]. By selectively removing the ofending input-space dictionary atoms, Dict-XAI blocks the flawed strategy at its source and prevents it from re-emerging through alternative internal circuits. Furthermore, we demonstrate DictXAI’s ability to operate across a wide variety of application-relevant data modalities and concept spaces. This includes using dictionaries of amplitude-modulated sinusoidal (AMS) waveforms to interpret QRS complex properties (e.g., duration) predicted from electrocardiogram (ECG) signals, as well as utilizing spectral profiles of microbial isolates as an interpretable basis for mass spectrometry analysis. In each case, DictXAI derives human-understandable and actionable insights that go significantly beyond the capabilities of classical pixel-level attribution or existing concept-based methods.

The remainder of this paper is structured as follows. Section 2 reviews related work in latentand input-level concept-based XAI. Section 3 introduces the mathematical framework of DictXAI, detailing its technical properties and the flexibility aforded by diverse dictionary choices. Section 4 presents an application-grounded evaluation of our method. Finally, Section 5 concludes the paper and outlines directions for future research.

## 2. Related Work

DictXAI formulates concept-based explanations directly in the input domain using overcomplete dictionaries. Accordingly, our work connects to two primary bodies of literature: concept-based XAI methods operating on internal representations (Section 2.1) and attribution methods that construct bases directly in the input domain (Section 2.2). For a broader overview of XAI, including classical attribution techniques such as Layer-wise Relevance Propagation (LRP) upon which our explanation framework builds, we refer to comprehensive review papers [1, 2].

## 2.1. Concepts in Latent Activation Layers

The intermediate layers of deep neural networks provide a rich representational space in which human-understandable concepts can emerge [20, 21]. Numerous works in XAI have sought to leverage these internal representations. For instance, Concept Relevance Propagation (CRP) [9] extends the LRP framework by filtering relevance flows through individual internal units (e.g., neurons or convolutional channels), where each unit is associated with a concept post hoc by inspecting the instances that maximally activate it. Moving from single units to layer-wide representations, Kim et al. [8] and Zhou et al. [13] utilize auxiliary concept datasets to identify concept-discriminative directions—termed Concept Activation Vectors (CAVs)—enabling attribution across distributed semantic directions rather than isolated channels.

Subsequent work has generalized concept vectors to multidimensional subspaces, introducing completeness guarantees to the resulting explanations [22, 14]. These frameworks bypass supervised labeling by extracting concept subspaces in an unsupervised manner via clustering or relevancedriven subspace analysis. In a related vein, Fel et al. [23] cast concept discovery as an unsupervised dictionary learning problem, applying non-negative matrix factorization (NMF) to hidden activations to uncover latent concept atoms. More recently, several works have turned to training sparse autoencoders (SAEs) on intermediate representations to isolate highly granular, monosemantic latent concepts [10, 24, 25].

Whereas all of the aforementioned methods are principally designed to operate in hidden layers, DictXAI shifts the explanation strategy entirely to the input domain. Grounding concepts directly in the input space confers three distinct advantages: (i) it avoids inheriting the black-box opacity of intermediate representations or introducing auxiliary autoencoding models fitted on them; (ii) unlike intermediate layers, the input domain provides rich, explicit semantics—such as scale, angle, or physical concentration—that dictionary elements can directly map to; and (iii) input-defined concepts are inherently model-agnostic, enabling direct, unmediated comparisons of decision strategies across difering architectures.

## 2.2. Concepts within the Input Domain

While latent-space approaches extract abstract representations from hidden layers, a parallel line of research decomposes explanations directly within the input domain using structured mathematical bases. For instance, Vielhaben et al. [26] proposed DFT-LRP, which introduces an inspection layer that allows propagation-based explanation methods to express relevance in a Fourier basis, particularly suited for mapping to the frequency domain that characterizes audio applications. A similar Fourier-basis approach, but applied to two-dimensional signals, was used by Kaufmann et al. [19] to explain image-based anomaly detection models; there, frequency-domain explanations were shown to uniquely reveal the model’s sensitivity to intricate technical artifacts, such as the presence or absence of image antialiasing.

Other frameworks leverage localized spatial-frequency representations. Kasmi et al. [27] proposed the Wavelet Attribution Method (WAM), which transforms the input into the wavelet domain to obtain localized spatial-frequency representations, enabling standard gradient-based explanation methods to assign importance scores to individual wavelet functions. Similarly, Kolek et al. [28] proposed CartoonX (later extended in [29]), which transforms inputs into the wavelet domain and learns a sparse perturbation mask over the wavelet coeficients via rate-distortion optimization.

Despite their utility, these existing input-domain frameworks remain fundamentally constrained by fixed, non-redundant or orthogonal basis sets where the number and functional form of basis elements are strictly tied to predefined mathematical transforms. Consequently, they lack the flexibility to incorporate empirical domain knowledge, learned dictionaries, or non-orthogonal primitives. In contrast, DictXAI afords complete freedom in the choice of representation: dictionaries can comprise parameterized functions, data-driven learned atoms, or experimentally acquired physical spectra. Furthermore, by embracing overcompleteness, DictXAI can decouple subtle, highly correlated concept variations—such as continuous scale shifts in physiological waveforms or overlapping spectral signatures in microbiological mixtures—that rigid orthogonal transforms cannot isolate.

## 3. The DictXAI Method

In this section, we introduce DictXAI, a framework that explains machine learning predictions in terms of a predefined, potentially overcomplete dictionary of input-domain concepts. In the context of images, each dictionary element (or atom) can represent, for example, an edge at a specific spatial location and orientation, or more complex pixel arrangements observed in the data. For a given input $\pmb { x } \in \mathbb { R } ^ { d }$ with prediction $y = f ( { \pmb x } )$ , DictXAI operates via three sequential steps: (1) express the input data in terms of the dictionary elements using linear sparse coding, (2) feed the reconstructed input into the ML model to obtain the prediction, and (3) attribute the prediction to dictionary atoms that contribute to it. This pipeline is illustrated at a high level in Figure 1. We detail each step below.

Step 1: Linear Sparse Coding. DictXAI assumes a dictionary of size K, where each dictionary element $\mathbf { d } _ { j } \in \mathbb { R } ^ { d }$ is expressible in input space. Linear sparse coding sets as an objective to reexpress the input data as a sparse combination of the dictionary elements [12, 30]. For a given data point $\pmb { x } \in \mathbb { R } ^ { d }$ , this task can be formulated mathematically as the LASSO objective [31]:

$$
\operatorname* { m i n } _ { \boldsymbol { \alpha } } \Big \{ \frac { 1 } { 2 } \Big \| \sum _ { j = 1 } ^ { K } \alpha _ { j } \mathbf { d } _ { j } - \pmb { x } \Big \| ^ { 2 } + \lambda \sum _ { j = 1 } ^ { K } | \alpha _ { j } | \Big \}\tag{1}
$$

where the variables $( \alpha _ { j } ) _ { j }$ denote the weighting coeficients. The first term in the objective ensures the reconstruction error is minimized. The second term enforces sparsity by driving noncontributing coeficients $\alpha _ { j }$ to zero. We note that, if these coeficients are calculated through an encoding function rather than optimized directly, the approach becomes conceptually equivalent to Sparse Autoencoders, commonly used in mechanistic interpretability [10]. However, directly solving Eq. (1) ofers two clear advantages. Because the coeficients $\alpha _ { j }$ are optimized freely per sample, this optimization is not constrained by the capacity limits of a fixed encoder. Furthermore, the absence of an encoder eliminates structural architectural bias and avoids the risk of out-of-distribution generalization failure.

Step 2: Recovering the Forward Pass. Having extracted a sparse representation of the input data, we must now re-establish the exact input-output relationship to be explained. Standard linear sparse coding produces a lossy reconstruction; that is, the linear combination of dictionary elements $\begin{array} { r } { \hat { \mathbf { x } } = \sum _ { j = 1 } ^ { K } \alpha _ { j } \mathbf { d } _ { j } } \end{array}$ does not exactly match the original input x. Feeding xˆ into the classifier can alter the prediction of the downstream ML model as well as the decision strategy employed to reach it. To prevent this distortion and remain faithful to the original prediction behavior, we reformulate the sparse coding problem above to deliver an explicit, lossless input reconstruction:

$$
\operatorname* { m i n } _ { \alpha , \mathbf { d } _ { 0 } } \Big \{ \frac { 1 } { 2 } \| \mathbf { d } _ { 0 } \| ^ { 2 } + \lambda \sum _ { j = 1 } ^ { K } | \alpha _ { j } | \Big \} \quad \mathrm { s . t . } \quad \pmb { x } = \sum _ { j = 0 } ^ { K } \alpha _ { j } \mathbf { d } _ { j } \quad \mathrm { a n d } \quad \alpha _ { 0 } = 1 .\tag{2}
$$

In this formulation, we have introduced an instance-specific dictionary atom $\mathbf { d } _ { 0 } \in \mathbb { R } ^ { d }$ whose magnitude acts as a slack variable that absorbs the residual reconstruction error. In practice, the solution to this optimization problem is equivalently obtained by solving Eq. (1) for α and setting the residual atom to ${ \bf d } _ { 0 } = { \pmb x } - \hat { { \pmb x } }$

We can now formally link the expressions of each atom in the dictionary to the model’s prediction via the equation:

$$
y = f \Big ( \sum _ { j = 0 } ^ { K } \alpha _ { j } \mathbf { d } _ { j } \Big ) .\tag{3}
$$

In other words, our approach yields a formulation that is functionally equivalent to the original $y = f ( { \boldsymbol { \mathbf { x } } } )$ , but crucially, incorporates the interpretability structure uncovered in Step 1. The notion of improving interpretability while maintaining exact functional equivalence is much in the spirit of advanced interpretability techniques, such as virtual layers [26, 14].

Step 3: Attribution to the Dictionary Atoms. Leveraging this computational structure, we can now focus on the problem of attributing the model prediction to the dictionary atoms. Many baseline frameworks commonly employed for input features can be readily extended to dictionary elements. We sketch the approach below for three cases representative of the major families of attribution methods, namely, occlusion-based, gradient-based, and propagation-based:

$$
\mathrm { O c c l u s i o n } \ [ 3 2 ] \quad R _ { j } = f ( { \pmb x } ) - f ( { \pmb x } - \alpha _ { j } { \bf d } _ { j } ) ,\tag{4}
$$

$$
\mathrm { I n t e g r a t e d ~ G r a d i e n t s ~ ( I G ) ~ } [ 3 3 ] \quad R _ { j } = \int _ { 0 } ^ { 1 } \frac { \partial y } { \partial \alpha _ { j } } \frac { \partial \alpha _ { j } } { \partial t } d t ,\tag{5}
$$

$$
\mathrm { L a y e r - w i s e ~ R e l e v a n c e ~ P r o p a g a t i o n ~ ( L R P ) ~ } [ 1 5 ] \quad R _ { j } = \sum _ { i = 1 } ^ { d } \frac { [ \alpha _ { j } \mathbf { d } _ { j } ] _ { i } } { x _ { i } } R _ { i } .\tag{6}
$$

For occlusion, removing atom $j$ corresponds directly to subtracting its contribution $\alpha _ { j } \mathbf { d } _ { j }$ from the input. This formulation naturally generalizes to Shapley Value Sampling [16], which aggregates these marginal efects across atom coalitions. For IG, the integration path is given by linearly scaling the weighting coeficients from zero to their actual value, i.e., $\ \alpha ( t ) = t \alpha$ . For LRP, inputfeature attributions $( R _ { i } ) _ { i = 1 } ^ { d }$ are propagated backward to the dictionary layer, here using the basic LRP-0 rule [34] with the convention $0 / 0 = 0$ . For improved stability, the dictionary layer and the model’s first linear layer can be merged into a single linear layer, granting access to more robust propagation variants based on LRP-γ [34].

## 3.1. Choice of Dictionary

The choice of dictionary defines the conceptual basis for interpretability and directly governs the expressiveness of the resulting explanations. In practice, dictionaries fall into three primary categories, each presenting distinct trade-ofs between flexibility, parametric transparency, and physical grounding.

Learned Dictionaries. A fully data-driven approach constructs dictionaries using unsupervised learning algorithms such as sparse dictionary learning [35, 12]. Most techniques initialize a random dictionary and alternate between identifying the optimal sparse code and updating the dictionary atoms to minimize reconstruction error; other approaches optimize end-to-end via gradient descent by incorporating a parameterized encoder function. To illustrate this behavior, Figure 2A displays a dictionary of $K = 8 0 0$ atoms trained on the MNIST handwritten digit benchmark (for training details, see Supplementary Note A). The resulting atoms capture refined, domain-specific primitives—here, an expressive variety of localized stroke contours. Because the dictionary atoms and sparse inference are jointly optimized for the target data distribution, learned dictionaries ofer near-optimal trade-ofs between sparsity and reconstruction fidelity. On the other hand, learned atoms lack explicit parametric formulas, which can complicate the qualitative interpretation of downstream explanations. Furthermore, data-driven dictionary learning assumes an abundant sample supply; data scarcity risks producing overfitted dictionary elements that impede the extraction of clear, human-aligned concepts.

C  
A Learned dictionary  
![](images/09658f1660cbd045198de7f692bebd369f8c95c274c0beed60843e9401b0a9b7.jpg)

B Analytically defined dictionary  
![](images/2c3fe3daab657ffea1e13d90fba29291ba38fa3ee4b5f7123801af18e0e53632.jpg)

![](images/3ef14e722fa79c4815c797d223a8a4b44dec3e3fe055d0be7c915afe23c69625.jpg)  
Figure 2: Dictionary designs and concept attribution profiles on MNIST. A Subset of atoms of intermediate $L _ { 2 }$ norm from a dictionary learned on the MNIST training set via sparse dictionary learning $( K = 8 0 0 ; \mathrm { M S E } = 0 . 0 1 7$ on 1 000 test images; $\lambda = 1 0 ^ { - 3 } )$ ). B Parameter sweeps from an analytically defined Gabor dictionary $( K \approx 1 0 0 0 0 ;$ $\mathrm { M S E } = 0 . 0 3 2$ on the same 1 000 test images; $\lambda = 1 0 ^ { - 3 } )$ . Each row isolates variations along a single parameter while holding others fixed: orientation θ (top), envelope scale σ (middle), and vertical center $y _ { 0 }$ (bottom). C Classaggregated profiles for the Gabor dictionary across test digits (classes 5–9, 30 samples each). Panels display Hinton diagrams where marker size denotes magnitude and color denotes sign (red: positive, blue: negative). Top row: mean sparse coding coeficients; bottom row: DictXAI explanation (mean relevance scores). Values are plotted across the flattened $5 \times 5$ spatial grid (y-axis, indices 0–24, row-major) and orientation $\theta \in [ 0 , \pi ]$ (x-axis), with the scale parameter σ averaged out.

Analytically Parameterized Dictionaries. When data is scarce or domain knowledge suggests a natural structure for the relevant concepts, dictionaries can instead be defined in closed form and indexed by interpretable parameters. A prominent example is a family of Gabor filters (Figure 2B; formal description in Supplementary Note A), which captures localized texture and frequency information across specific spatial frequencies, orientations, phase ofsets, aspect ratios, and spatial locations. Discretizing these parameters yields a structured dictionary of analytical atoms. A major advantage of this design is that each atom ${ \bf d } _ { j }$ possesses an explicit coordinate in the underlying parameter space. Consequently, downstream explanations can be mapped directly into the metric space spanned by these parameters, facilitating intuitive visualization. For example, when atoms vary along grid-discretized parameters such as orientation and spatial position, feature attributions can be rendered as a Hinton diagram over that parameter grid (Figure 2C). This enables practitioners to systematically identify which parameterized features the model relies on when making predictions, both locally and across the dataset. However, because analytical dictionaries are not adapted to the data distribution, spanning continuous parameter spaces requires extensive discretization: in our illustrative setting, an overcomplete dictionary comprising over 12 more elements than the learned baseline still achieves almost two times higher reconstruction error (MSE = 0.032 vs. 0.017).

Dictionaries of Elementary Constituents. A third category emerges in physical and biological domains where the observed signal is known to be an additive mixture of basic physical components in unknown proportions. This setting is ubiquitous in applications such as spectrometry and computational cytometry, where an aggregate measurement reflects a superposition of constituent profiles— such as a bulk chemical spectrum composed of pure molecular signatures, or bulk RNA sequencing deconvolved into cell-type-specific references. In these scenarios, the dictionary is neither learned statistically from the target dataset nor generated from an idealized functional form; rather, it comprises an empirical reference library isolated through prior experimental characterization. The principal advantage of such dictionaries is their direct physical grounding: each atom corresponds unambiguously to a concrete, real-world entity. However, their fidelity depends strictly on library completeness; unmodeled chemical contaminants or uncharacterized cell types cannot be captured within a fixed constituent basis.

Importantly, these dictionary archetypes are not mutually exclusive. In principle, one could construct hybrid dictionaries that augment analytical or physical reference bases with learnable atoms. When fixed functional families or reference libraries cannot fully capture all factors of variation in the data, auxiliary learnable elements could absorb the remaining residual variance while preserving the explicit interpretability of the known components.

## 3.2. Properties of DictXAI

In this section, we analyze DictXAI across multiple operational criteria, ranging from formally verifiable axiomatic guarantees to qualitative usability attributes. Together, these properties cover the five foundational explanation desiderata established by Swartout and Moore [6] and adopted in modern XAI [36, 37]: faithfulness (D1), understandability (D2), suficiency (D3), low overhead (D4), and runtime eficiency (D5). Table 1 provides a systematic comparative summary alongside representative concept- and transform-based methods from the literature. To reflect the distinct dictionary paradigms established in Section 3.1, we evaluate DictXAI under two primary operating modes: a curated regime based on pre-defined analytical or physical components, and a fully datadriven learned regime. As summarized in Table 1, while both regimes share identical faithfulness (D1) and computational complexity (D5), they exhibit complementary trade-ofs across usability, steerability, and annotation requirements.

D1: Faithfulness (Conservation and Input Stability) requires an explanation to reflect the model’s true decision mechanism without distorting evidence or behaving erratically under minor noise. DictXAI satisfies this through two foundational guarantees. First, conservation ensures that decomposed relevance scores sum exactly to the model’s output to permit unambiguous interpretation of relative evidence shares. By prepending a single linear dictionary layer to the network and executing standard attribution engines (e.g., LRP or Shapley values) over this extended architecture, Dict-XAI directly inherits their conservation guarantees, satisfying $\begin{array} { r } { \sum _ { j = 0 } ^ { K } R _ { j } = \sum _ { i = 1 } ^ { d } R _ { i } = f ( { \pmb x } ) } \end{array}$ . While transform propagation frameworks (DFT-LRP, WAM) also conserve relevance, methods rooted in directional projections (TCAV) or rate-distortion masking (CartoonX) are non-conservative by design. Second, input stability demands that bounded input perturbations yield bounded variations in concept attributions. DictXAI guarantees this stability by lifting the smoothness of the underlying model directly into the dictionary domain:

Proposition 1 (Lipschitz Continuity of Dictionary-Level Derivatives). Recall from $E q s . \ ( 1 ) \not { – } ( 3 )$ that the model prediction can be expressed as $y = f ( \pmb { x } ) = f ( \hat { \pmb { x } } + \mathbf { d } _ { 0 } )$ , where $\begin{array} { r } { \hat { \pmb x } = \sum _ { i = 1 } ^ { K } \alpha _ { j } \mathbf { d } _ { j } } \end{array}$ and the residual ${ \bf d } _ { 0 } = { \pmb x } - \hat { { \pmb x } }$ is held constant when diferentiating with respect to α at x. $I f \nabla f$ is $L _ { - }$ Lipschitz continuous on a convex domain $\Omega \subseteq \mathbb { R } ^ { d }$ , then for any dictionary atom $\mathbf { d } _ { j } \ ( j = 1 , \ldots , K )$ the derivative map $\pmb { x } \mapsto \partial y / \partial \alpha _ { j }$ is $( L \| \mathbf { d } _ { j } \| _ { 2 } ) { \boldsymbol { \cdot } } L$ ipschitz continuous with respect to x on Ω.

Proof. Evaluating the partial derivative with respect to code coordinate $\alpha _ { j }$ via the chain rule through the intermediate representation $\begin{array} { r } { \hat { \pmb x } = \sum _ { j = 1 } ^ { K } \alpha _ { j } \mathbf { d } _ { j } } \end{array}$ yields the directional derivative along atom ${ \bf d } _ { j }$ :

$$
\frac { \partial y } { \partial \alpha _ { j } } ( { \pmb x } ) = { \bf d } _ { j } ^ { \top } \nabla f ( { \pmb x } ) .
$$

For any two points ${ \pmb x } _ { 1 } , { \pmb x } _ { 2 } \in \Omega$ , applying the Cauchy–Schwarz inequality followed by the L-Lipschitz continuity of $\nabla f$ gives:

$$
\begin{array} { r l r } {  {  \frac { \partial y } { \partial \alpha _ { j } } ( { \pmb x } _ { 1 } ) - \frac { \partial y } { \partial \alpha _ { j } } ( { \pmb x } _ { 2 } )  =  { \bf d } _ { j } ^ { \top } ( \nabla f ( { \pmb x } _ { 1 } ) - \nabla f ( { \pmb x } _ { 2 } ) )  } } \\ & { } & { \leq \| { \bf d } _ { j } \| _ { 2 } \| \nabla f ( { \pmb x } _ { 1 } ) - \nabla f ( { \pmb x } _ { 2 } ) \| _ { 2 } } \\ & { } & { \leq L \| { \bf d } _ { j } \| _ { 2 } \| { \pmb x } _ { 1 } - { \pmb x } _ { 2 } \| _ { 2 } . } \end{array}
$$

Hence, input gradient stability strictly bounds variations in the dictionary domain, scaled by the atom norms ${ \| \mathbf { d } _ { j } \| } _ { 2 }$ . Furthermore, under standard LASSO regularity conditions (such as the dictionary satisfying the general position assumption), the active coordinate support and its sign pattern remain locally invariant. On these support regions, the coordinate mappings $\pmb { x } \mapsto \alpha _ { j } ( \pmb { x } )$ are locally Lipschitz continuous. Because products of bounded Lipschitz maps remain Lipschitz, these stability guarantees extend directly to attribution methods based on coeficient-gradient products (such as Integrated Gradients and Gradient  Input) and stabilized LRP propagation rules.

D2: Understandability (Concept Steerability) requires explanations to align with the language of domain experts, enabling the underlying concept vocabulary to be steered toward task-specific ontologies. DictXAI natively fulfills this property by allowing practitioners to directly inject domain knowledge into the dictionary—whether via expert-defined analytical bases, flexible overcomplete unions of them, or experimentally acquired prototypes. In contrast, rigid transform baselines (DFT-LRP, WAM, CartoonX) remain bound to a single, mathematically immutable transform, while unsupervised latent decompositions (DRSA) cannot be steered, relying on the tenuous assumption that unsupervised data statistics happen to align with the expert’s semantic domain.

D3: Suficiency (Granularity and Discovery) demands that an explanation vocabulary possesses both the resolution and the scope needed to fully characterize model decisions. DictXAI addresses this through two complementary facets. First, concept granularity governs the spatial and semantic resolution at which an explainer isolates evidence. While classical transforms remain constrained by input-space orthogonality, overcomplete dictionaries allow DictXAI to disentangle localized, highly fine-grained primitives directly from the input signal without collapsing distinct features into coarse averages. Second, concept discovery ensures comprehensive coverage across the data manifold. Through learned dictionary elements, DictXAI accounts for hard-to-specify yet humanaligned concepts (e.g., image parts or textures) that are missing from classical analytical families.

$D \llcorner$ Low Overhead (Codebase and Annotation Burden) is essential for practical adoption. DictXAI balances these dimensions across its two operating modes. First, codebase overhead remains minimal: prepending an input-level dictionary avoids intercepting intermediate activations, altering network architectures, or managing layer-specific backward hooks required by latent-space frameworks like CRP or DRSA. When using curated dictionaries, no auxiliary training pipelines are needed (✓); learned dictionaries introduce only a standard unsupervised dictionary learning pre-routine (∼). Second, the annotation overhead is virtually absent. Unlike supervised approaches (e.g., TCAV) that require gathering positive and negative concept exemplars, learned DictXAI requires solely unlabeled data (✓), while curated DictXAI requires only the initial parametric specification or assembly of a reference basis (∼).

<table><tr><td rowspan="2"></td><td rowspan="2">CRP [9]</td><td rowspan="2">DRSA [14]</td><td rowspan="2">TCAV [8]</td><td rowspan="2">DFT-LRP [26]</td><td rowspan="2">WAM [27]</td><td rowspan="2">CartoonX [28]</td><td colspan="2">DictXAI (ours)</td></tr><tr><td>curated learned</td><td></td></tr><tr><td rowspan="2">D1</td><td>Conservation</td><td>~(1)</td><td>J</td><td>X</td><td>√</td><td>J</td><td>X J</td><td>J</td></tr><tr><td>Input stability</td><td>√</td><td>J</td><td>N/A</td><td>V √</td><td>X</td><td>√</td><td>√</td></tr><tr><td>D2</td><td>Concept steerability</td><td>X</td><td>X</td><td>√ X</td><td>X</td><td>X</td><td>√</td><td>X</td></tr><tr><td rowspan="2">D3</td><td>Concept granularity</td><td>√</td><td>√</td><td>√</td><td> $x ^ { ( 2 ) }$ </td><td>x(2) x(2)</td><td>√</td><td>√</td></tr><tr><td>Concept discovery</td><td>√</td><td>√ X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>√</td></tr><tr><td rowspan="2">D4</td><td>Low codebase overhead</td><td>~</td><td>X</td><td>X</td><td>√</td><td>√ 2</td><td>√</td><td>2</td></tr><tr><td>Low annotation overhead</td><td>~</td><td>√</td><td>X</td><td>√</td><td>√ √</td><td>2</td><td>√</td></tr><tr><td rowspan="2">D5</td><td>Compute cost (until concepts)</td><td>f</td><td>f</td><td>f</td><td>f</td><td>f</td><td>Mf</td><td>f + SC</td></tr><tr><td>Compute cost (until pixels)</td><td>Kf</td><td>Kf</td><td>N/A</td><td>f</td><td>f</td><td>Mf</td><td>f + SC</td></tr></table>

Table 1: Systematic comparison of concept-based XAI methods across technical properties, grouped by the five desiderata of [6] (D1: faithfulness, D2: understandability, D3: suficiency, D4: low overhead, D5: runtime eficiency). Symbols $\checkmark , \sim$ , and ✗ denote satisfied, partially satisfied, and unsatisfied properties, respectively, while N/A indicates that a property is not applicable due to the underlying attribution paradigm. DictXAI is evaluated under both domain-curated and data-driven learned dictionary regimes, assuming LRP as the base attribution engine. In D5, f denotes the cost of a single model evaluation (forward and backward pass), K is the number of queried concepts, M is the number of optimization steps in CartoonX, and SC is the cost of solving the linear sparse coding subproblem. <sup>(1)</sup> Inherits conservation from LRP but strong dissipation due to single-channel relevance filtration. <sup>(2)</sup> Fundamentally limited by input space orthogonality.

D5: Runtime Eficiency (Computational Complexity) enables deployment of explanation methods at scale (e.g., on large models or on entire datasets). Letting f denote the cost of a model forward– backward pass, intermediate-layer methods (CRP, DRSA) cost f to attribute relevance to concepts but require K sequential backward sweeps (Kf) to map relevance from each queried concept back to the input, while perturbation techniques like CartoonX incur M iterative optimization steps (Mf). In contrast, DictXAI performs sparse coding once per sample at cost SC and executes a single forward–backward pass on the dictionary-extended model $( f + \mathrm { S C } )$ . This achieves simultaneous concept attribution and pixel-level grounding without scaling linearly with the number of queried concepts K.

## 4. Evaluation

DictXAI attributes predictive evidence over structured dictionary bases rather than raw input dimensions or fixed internal layers. Consequently, comparing our framework to existing XAI methods requires evaluating explanations constructed over fundamentally disparate concept representations. Standard input-space faithfulness metrics, such as Pixel-Flipping [38] or bounding-box IoU [39], cannot readily account for such basis shifts, as they presuppose a shared coordinate frame. We therefore adopt an application-grounded evaluation framework [40], where each benchmark defines a task-specific success criterion: downstream classification accuracy on a decorrelated test set for Clever Hans unlearning (Section 4.1), alignment with an independently verifiable failure mode in electrocardiogram analysis (Section 4.2), and the ability to calibrate a mass spectrometry representation for downstream microbial analysis (Section 4.3).

## 4.1. Task 1: Data-Level Mitigation of the Clever Hans Efect

Machine learning models frequently encounter and exploit spurious correlations within training data. When a model relies on such uninformative artifacts to minimize its empirical risk—a phenomenon known as the Clever Hans efect [18, 19]—its generalization performance degrades substantially when evaluated on test data where these shortcut correlations no longer hold. While detecting shortcut strategies via post-hoc attribution is well established, mitigating reliance on them remains an open challenge, with existing solutions largely confined to altering a model’s internal representations post-hoc [41].

Rather than modeling an ML model’s internal representations—a prerequisite for altering its internals—DictXAI grounds explanations directly in concepts defined in the input domain. This distinct property raises the question of whether DictXAI provides a pathway to address the Clever Hans efect directly at the data level rather than at the model level. Intervening at the data level ofers a fundamental advantage: by sanitizing the dataset itself, it permanently prevents shortcut strategies from re-emerging whenever new models are trained or existing ones updated. To test this capability, we design an intervention pipeline where DictXAI attributions guide the removal of artifact components from the training data prior to retraining. Our experimental protocol consists of three steps:

Step 1 Spuriously Correlated Dataset: We consider the MNIST dataset augmented with three types of artifacts: (i) a fixed-location horizontal stripe, (ii) a dotted line at varying locations, and (iii) a signature glyph in the bottom-right corner (cf. Supplementary Note B). These watermarks are systematically inserted into all training samples of class ‘6’.

Step 2 Initial Training and Attribution: We train five seeded copies of a small CNN on the contaminated data (details in Supplementary Note B) and compute post-hoc explanations over each method’s respective attribution basis—for DictXAI, we use the same dictionary learning algorithm as in Section 3.1 and train it on the contaminated data. For an input sample x, each basis element $j$ with scalar relevance $R _ { j }$ can be mapped to an associated pixel-space attribution heatmap $h _ { j } \in \mathbb { R } ^ { d }$ (cf. Supplementary Note C).

Step 3 Removal of Spurious Features, Retraining, and Evaluation: To standardize feature isolation across methods and emulate human oversight, we employ an oracle spatial mask $\boldsymbol { m } \in \mathbb { R } ^ { d }$ localizing the watermark. Each basis element j is scored by its expected overlap with the artifact region, $S _ { j } = \mathbb { E } [ h _ { j } ^ { \top }$ m]. We then identify and remove the contributions of the k basis elements exhibiting the highest overlap scores $S _ { j }$ from each training sample. A fresh model is then trained from scratch on the cleaned data and evaluated on a decorrelated test set, in which the artifact remains present but is distributed uniformly across all classes.

![](images/78e3c54d52d661e6192aeb36db31a02e37601421d9994a97af0d298afe500020.jpg)  
Figure 3: Horizontal-stripe variant, class “6”. Upper row: the original input, and the same sample after the oracle removes the top-k most artifactual elements encoded by each method. Elements are ranked by their overlap S with the watermark mask over a fixed 500-image training subset. To keep the comparison fair across diferently sized bases, k is fixed at the grid point closest to 5% of each method’s total concept count (between 4.1 and 5.7%). Lower row: pixel-wise LRP-γ heatmaps of the base model and of models retrained on the resulting pruned data, each averaged over five seeds. The color scale for each panel is normalized to that of the base model on the unpruned, contaminated data.

We evaluate DictXAI against several baselines. To remain compatible with data-level intervention in Step 3, we exclude latent-space attribution methods such as TCAV or CRP: they operate within hidden layer representations and lack a direct mechanism to remove flawed concepts from the input data. Instead, we select baselines spanning two categories: (1) Spatial attributions: We consider LRP [15, 34] (γ = 0.1), which produces pixel-level attributions, and Shapley Value Sampling [16], which yields model-agnostic attributions over coarser 4 4 spatial patches, averaged over 10 feature permutations. (2) Transformed input bases: We consider DFT-LRP [26], which attributes relevance in the Fourier domain associated with the input images, alongside two methods operating over a level-3 Daubechies-5 wavelet basis: WAM [27], which scores wavelets via input-gradient attributions (Gradient Input), and CartoonX [28], which identifies relevant wavelets via rate-distortion optimization.

Because feature granularity varies considerably across these methods, we cannot set the removal budget k to a uniform constant. Instead, we sweep through a broad range of values for k and measure the decorrelated test accuracy after retraining, essentially a variation of the Remove-and-Retrain (ROAR) evaluation framework [42]. The resulting accuracy curves are shown in Figure 4. DictXAI achieves the strongest performance across all three artifact variants (cf. Step 1), consistently outperforming the original contaminated model and closely matching the ideal oracle model trained on clean data. Notably, DictXAI reaches peak accuracy after removing only 2.8% (horizontal), 5.8% (dotted), and 1.0% (signature) of its dictionary elements, compared to 7.5%, 9.6%, and 5.7% of pixels for LRP, respectively, demonstrating a more compact and targeted concept granularity than raw pixel representations. In contrast, alternative baselines (Shapley Value Sampling, DFT-LRP, WAM, and CartoonX) struggle substantially, especially on the more distributed horizontal and dotted artifacts, where they fail to outperform the unmitigated base model despite targeted feedback.

![](images/6fdf562c9ee0c1676b396b38a8b612d01f58703dac561dbd3ed967edbb11c805.jpg)  
Figure 4: Remove-and-retrain (ROAR) curves for the three watermark variants: overall accuracy on the decorrelated test set as a function of the fraction of concepts removed before fine-tuning. Lines and shading represent mean and std over five seeds, and stars mark maxima. For each XAI method, extracted concept elements are inspected pixelwise, ranked by their overlap with the watermark $( S _ { j } )$ , and removed in this order. The three horizontal references are fine-tuning on watermark-free data (dashed, upper bound), fine-tuning without any removal (dash-dotted, the k = 0 control), and the original model (dotted).

While these methods falter, DictXAI maintains robust performance across every setting, trailing the clean-trained oracle by only 0.7, 0.0, and 0.2 percentage points on the horizontal, dotted, and signature watermarks, respectively.

On a qualitative level, Figure 3 illustrates which features each method isolates and removes from the contaminated inputs. For DFT-LRP, WAM, and CartoonX, the removal process introduces pronounced visual distortions across the image. For LRP and Shapley Value Sampling, the placeholder values used to mask pixels and patches leave behind sharp boundaries and unnatural gray shading. In contrast, DictXAI produces sanitized images that closely resemble authentic uncorrupted digits, retaining only a faint trace where the watermark originally appeared. This indicates that DictXAI efectively disentangles the watermark from digit semantics, whereas alternative baselines disrupt the underlying data distribution by distorting meaningful signal. These removal artifacts directly impact subsequent model behavior. For DFT-LRP, WAM, and CartoonX, the resulting distortions yield noisy attribution heatmaps, suggesting that the retrained models attend to synthetic masking artifacts rather than true digit features. Furthermore, explanations for Shapley, WAM, and CartoonX reveal that the retrained models still rely heavily on the watermark shortcut; while this reliance is somewhat reduced for LRP and DFT-LRP, it persists nonetheless. Only for DictXAI is shortcut reliance no longer qualitatively visible in the retrained model’s explanations.

## 4.2. Task 2: Mapping an ML Decision Strategy to the Human Expert Domain

A core challenge in XAI is translating model decision strategies into the conceptual vocabulary of domain experts. In time-series and biosignal analysis, standard attribution methods highlight isolated sample points that lack semantic meaning, whereas practitioners reason over multi-scale morphological patterns and local frequencies [1, 26]. This alignment is critical in electrocardiogram (ECG) analysis. Deep learning models applied to continuous ECG recordings have demonstrated remarkable diagnostic capabilities, detecting subtle, clinically silent conditions—such as asymptomatic ventricular dysfunction [43] or occult atrial fibrillation [44]—that rigid, rule-based delineation algorithms often miss. Yet because these end-to-end models bypass predefined physiological features, high aggregate performance alone cannot ensure reliability. As ECG analysis transitions from controlled 12-lead settings to single-lead wearable monitoring where motion artifacts sharply degrade signal quality, clinical safety—and regulatory approval for high-risk diagnostic AI—demands verifying that predictions reflect genuine electrophysiological intervals rather than noise-driven shortcuts.

To evaluate whether DictXAI can bridge this gap and provide meaningful clinical verification, we consider the task of explaining a model trained to predict mean QRS duration from continuous single-lead recordings. QRS duration reflects ventricular conduction, and gradual prolongation or widening independently indicates structural heart disease and elevates the risk of major adverse cardiovascular events [45]. Because duration is governed by the temporal scale of the cardiac complex rather than isolated amplitudes, point-wise XAI methods cannot capture this relational property [46]. Specifically, we build a 1D CNN [47] to classify narrow ( 85 ms) versus prolonged ( 105 ms) QRS complexes, excluding borderline cases (85–105 ms; Figure 5 A). The model is trained on 10-second Lead I traces from MIMIC-IV-ECG<sup>1</sup>, comprising 104 212 samples (70/10/20% train/val/test split). The resulting model achieves 95.03% test accuracy (Supplementary Note D), providing a high-performing black-box predictor whose internal strategy we subject to post-hoc explanation.

In this application setting, DictXAI readily benefits from existing signal dictionaries designed for ECG morphology (e.g., [48]). Specifically, we adopt the amplitude-modulated sinusoidal (AMS) waveforms from [48] to model the QRS complex, while replacing their Hermite functions with Gaussian atoms for the P and T waves. The exact parameterization and construction of the dictionary are detailed in Supplementary Note D. Discretizing the parameters of these constituent waveforms across physiologically motivated grids yields an overcomplete dictionary of 300 000 atoms. Sparse representations are obtained eficiently via Orthogonal Matching Pursuit (OMP), using a budget of 150 nonzero coeficients per 10-second recording. DictXAI relevance scores are then computed for each dictionary element using the method specified in Eq. (6). For comparison against classical point-wise attribution [46], we include feature-level LRP as a baseline.

Figure 5 B shows the explanations produced by the LRP baseline and DictXAI. LRP heatmaps allocate high relevance scores to high-amplitude regions of the signal. While their locations coincide with QRS features, they provide limited insight into the exact nature of the ML calculation and how it discriminates between the two classes. In contrast, DictXAI delivers crucial explanatory feedback: AMS-level attribution (averaged across 200 samples per class) highlights a systematic diference between the narrow and wide QRS complexes defining Classes 1 and 2, showing that positive relevance is concentrated within their respective QRS intervals.

The quantitative relevance profile obtained via DictXAI (Figure 5B) indicates that the model primarily relies on detecting wide QRS complexes (Class 2), defaulting to Class 1 in their absence regardless of whether narrow QRS complexes are explicitly detected. In Figure 5C, we validate this explanation-derived hypothesis by perturbing the ECG waveform with technical noise (here, additive white Gaussian noise) designed to degrade the model’s feature detection capabilities. As noise increases in the moderate regime, the logit diference tilts toward Class 1 despite the purely technical nature of the signal degradation, before reversing under extreme noise levels. This behavior confirms the asymmetric reliance on Class 2 features revealed by DictXAI. Clinically, this reveals a hazardous failure mode: noise causes the model to miss prolonged QRS complexes and silently revert to a normal-conduction baseline. This vulnerability was predicted from the explanation alone—prior to any empirical perturbation—demonstrating that dictionary-based explanations can yield actionable insights into model brittleness and associated risks.

![](images/5764015b731a7c115dbf15e46d96eef53966bfe445d11b56db892c1cb31eb885.jpg)

A ECG Data  
B Model's Explanation  
![](images/42c864d10a719c0438c4f73d89e505c8c618b482ff20b72045534c495006978d.jpg)

![](images/6265894d8b567b57a0e95c0ad103a2f61a6687e9bf39c534915dd55876ce8c6a.jpg)  
Figure 5: Explaining the prediction of QRS segment width in an ECG signal. A: Zoomed-in excerpt of the 10-second Lead I recording for two samples from the MIMIC-IV-ECG dataset, one from each class. B: Explanation of the trained ML model, showing amplitude-driven explanations for the LRP baseline and DictXAI’s significantly more discriminative explanations attributing class evidence to distinct AMS widths. C: Verification that the ML model’s strategy of relying mainly on class 2 features lacks robustness to noise. Adding moderate noise (std 0.3) degrades feature detection, causing the model prediction diference (y<sub>2</sub> y<sub>1</sub>) to revert toward the Class 1 baseline.

## 4.3. Task 3: Guided Design of Data Representations

Designing efective data representations is critical when deploying machine learning to complex physical signals. A prominent domain where this challenge arises is spectral analysis, which encompasses modalities that characterize materials or biospecimens as continuous signal intensity distributions over physical axes (e.g., mass-to-charge ratios, wavelengths, or frequencies). Often reflecting intricate biological or chemical mixtures, these spectra routinely serve as direct inputs to predictive models. In clinical settings, end-to-end approaches operating directly on raw spectra (e.g., MALDI-TOF MS) are increasingly adopted to predict diagnostic endpoints such as antimicrobial resistance [49, 50]. However, defining an optimal representation remains a key bottleneck: practitioners must eliminate pervasive baseline drift and high-dimensional noise without attenuating low-abundance, highly informative peaks.

Because these spectral spaces are high-dimensional and non-intuitive, evaluating whether a given preprocessing pipeline preserves meaningful biological signal is non-trivial. Standard featureattribution methods merely highlight isolated spectral bins or broad peak regions (cf. Fig. 6, top), ofering limited insight unless a domain expert can laboriously map those coordinates back to specific biomolecules. DictXAI bridges this interpretability gap. Because physical spectra often adhere to an additive superposition of constituent components (e.g., individual microbial species within a co-culture), clean reference profiles naturally form a semantic dictionary. Decomposing the input over this dictionary enables explanations grounded directly in known biological entities.

![](images/a7fa564c234616a1d79761a33fba7598e54620852a8474f8847bf8589e97b5b7.jpg)  
Figure 6: Comparison between classical spectrum-based explanations (top) and DictXAI microbe-based explanations (bottom) of spectral similarity, computed on raw spectra (left) versus band-pass filtered spectra (right). Top: Traces depict MALDI-TOF mass spectra for two co-cultures $( { \pmb x } , { \pmb x } ^ { \prime } )$ , with red bars highlighting spectral regions contributing positively to predicted similarity. Bottom: DictXAI bipartite relevance graphs linking dictionary coeficients from the two mixtures. Taxa in black and gray denote species present and absent from the respective mixture; edge thickness denotes the attributed concept-level relevance. Filtering isolates true biological matches (e.g., C. tertium, DH5a-K12) from broad spectral background.

To demonstrate this, we consider an experimental mass spectrometry benchmark involving microbial mixtures. Here, the dictionary is curated by culturing individual microbial taxa on selective media to acquire pure MALDI-TOF reference spectra. Our dictionary encompasses 23 distinct isolates, each represented by 48 technical replicates, yielding K = 1104 prototype elements. To assess what two co-culture mixtures x and $\mathbf { { x } ^ { \prime } }$ share, we model their comparative similarity via the inner product $y = \langle \Phi ( { \pmb x } ) , \Phi ( { \pmb x } ^ { \prime } ) \rangle$ , where Φ denotes candidate preprocessing transformations. Using DictXAI, we decompose each mixture into sparse coeficients over the reference dictionary $\begin{array} { r } { ( \pmb { x } \approx \sum _ { j } \alpha _ { j } \mathbf { d } _ { j } } \end{array}$ and $\begin{array} { r } { \mathbf { x } ^ { \prime } \approx \sum _ { j ^ { \prime } } \alpha _ { j ^ { \prime } } ^ { \prime } \mathbf { d } _ { j ^ { \prime } } ) } \end{array}$ , allowing the similarity score to be expanded as:

$$
y = \Bigl \langle \Phi \Bigl ( \sum _ { j } \alpha _ { j } \mathbf { d } _ { j } \Bigr ) , \ \Phi \Bigl ( \sum _ { j ^ { \prime } } \alpha _ { j ^ { \prime } } ^ { \prime } \mathbf { d } _ { j ^ { \prime } } \Bigr ) \Bigr \rangle .\tag{7}
$$

Following [17], this formulation can be faithfully attributed and visualized as a bipartite relevance graph connecting the activated dictionary concepts across both branches. As shown in Fig. 6 (bottom), inspecting these bipartite explanations across choices of Φ reveals immediately whether a representation emphasizes true shared taxa or spurious baseline artifacts, directly guiding the design of robust preprocessing pipelines.

Equipped with these concept-level bipartite attributions, DictXAI provides an actionable mechanism to systematically compare candidate representations Φ. In particular, results in Fig. 6 reveal that applying a band-pass filter fundamentally alters the attribution topology: under the raw representation (left), similarity is difuse and driven by broad background correlations that activate spurious connections. In contrast, post-filtering, relevance collapses almost entirely onto the shared taxa truly present in both mixtures (e.g., Clostridium tertium and DH5a-K12 ). DictXAI thus functions as an unsupervised evaluation diagnostic for representation engineering, enabling practitioners to validate that a pipeline preserves biologically meaningful structure without relying on downstream supervised labels or ad-hoc task performance. In this respect, our framework aligns with the unsupervised validation paradigm of Kaufmann et al. [19], extending the approach beyond low-level feature attribution to interpretable, dictionary-level concept attribution.

## 5. Conclusion

We introduced DictXAI, a framework for concept-based explanation that defines its concepts directly in the input domain via a dictionary. By re-expressing the input as a sparse code over dictionary atoms, our method provides a new, highly granular basis for attribution that can be readily tailored to specific domain needs through custom dictionary design.

By operating in the input domain, DictXAI overcomes the core limitations of latent-space concept extraction. Concepts are human-inspectable primitives that can be named and verified directly, bypassing latent-layer probing, polysemantic neuron ambiguity, and non-linear autoencoder approximations while enabling direct comparisons across distinct architectures. Crucially, because representations and interventions operate directly at the input level, DictXAI ensures that concepts are transparent by design, mathematically well-defined, and actionable at their source.

We demonstrated these advantages through an application-grounded evaluation across three use cases. (1) In a vision model trained on artifact-contaminated MNIST data, DictXAI precisely isolates the shortcut features and neutralizes them at their root through direct input-level intervention, thereby preventing the model from circumventing the correction in downstream training. (2) On continuous ECG recordings, DictXAI leverages a parametric AMS waveform dictionary to audit a complex black-box neural network, successfully probing the multi-scale temporal decision strategy used to assess QRS complex duration. (3) Finally, in microbial mass spectrometry, DictXAI attributes spectral similarity directly to biological isolates, providing an unsupervised diagnostic to guide preprocessing pipelines in a strongly confounded data modality.

## Limitations and Future Work

Despite these advantages, several limitations of the current framework suggest concrete avenues for future research. First, DictXAI currently relies on fixed-dimensional vector representations, restricting its out-of-the-box applicability to variable-sized images, raw text, or non-Euclidean data structures. Scaling the framework to such unstructured domains could be achieved by integrating convolutional sparse coding or adaptive dictionary learning directly into the explanation pipeline. Second, as input dimensionality grows, overcomplete dictionaries are susceptible to the curse of dimensionality, where the number of required atoms scales exponentially to preserve concept granularity. Addressing this scaling bottleneck calls for factored dictionary designs, hierarchical sparse representations, or deep dictionary formulations that factorize large concept spaces into compact, separable components.

Looking forward, the fine-grained interpretability aforded by input-space dictionaries unlocks novel translational applications. In high-stakes biomedical domains, DictXAI can be extended from isolated diagnosis auditing to comparative, fine-grained concept mapping across clinically similar disease phenotypes. By decomposing shared versus pathology-specific atoms, future work could pinpoint subtle morphological or spectral distinctions that diferentiate ambiguous conditions, ultimately transforming post-hoc attribution into an actionable tool for clinical hypothesis generation and diferential diagnosis.

## Acknowledgements

This work was funded by the German Ministry for Education and Research as BIFOLD - Berlin Institute for the Foundations of Learning and Data (ref. BIFOLD25B). Klaus-Robert Müller was supported in part by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2019-II190079, Artificial Intelligence Graduate School Program, Korea University), a grant funded by the Korea government (MSIT, No. RS-2024-00457882, AI Research Hub Project), the German Research Foundation (DFG), and the Hector Fellow Academy. Grégoire Montavon and Alexander Meyer are supported by the Einstein Center for Early Disease Interception (EC-EDI). Thomas Schnake is a postdoctoral fellow at the University of Toronto in the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship Program, a program of Schmidt Sciences.

## Declaration of Generative AI and AI-assisted technologies in the writing process

During the preparation of this work the author(s) used Gemini 3.8 and Claude 5 in order to improve formulations. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the publication.

## References

[1] W. Samek, G. Montavon, S. Lapuschkin, C. J. Anders, K.-R. Müller, Explaining deep neural networks and beyond: A review of methods and applications, Proceedings of the IEEE 109 (2021) 247–278.

[2] A. B. Arrieta, N. D. Rodríguez, J. D. Ser, A. Bennetot, S. Tabik, A. Barbado, S. García, S. Gil-Lopez, D. Molina, R. Benjamins, R. Chatila, F. Herrera, Explainable artificial intelligence (XAI): concepts, taxonomies, opportunities and challenges toward responsible AI, Information Fusion 58 (2020) 82–115.

[3] C. A. Zhang, S. Cho, M. Vasarhelyi, Explainable artificial intelligence (xai) in auditing, International Journal of Accounting Information Systems 46 (2022) 100572.

[4] F. Wong, E. J. Zheng, J. A. Valeri, N. M. Donghia, M. N. Anahtar, S. Omori, A. Li, A. Cubillos-Ruiz, A. Krishnan, W. Jin, A. L. Manson, J. Friedrichs, R. Helbig, B. Hajian, D. K. Fiejtek, F. F. Wagner, H. H. Soutter, A. M. Earl, J. M. Stokes, L. D. Renner, J. J. Collins, Discovery of a structural class of antibiotics with explainable deep learning, Nature 626 (2024) 177–185.

[5] M. Esders, T. Schnake, J. Lederer, A. Kabylda, G. Montavon, A. Tkatchenko, K.-R. Müller, Analyzing atomic interactions in molecules as learned by neural networks, Journal of Chemical Theory and Computation 21 (2025) 714–729.

[6] W. R. Swartout, J. D. Moore, Explanation in second generation expert systems, in: J.-M. David, J.-P. Krivine, R. Simmons (Eds.), Second Generation Expert Systems, Springer Berlin Heidelberg, Berlin, Heidelberg, 1993, pp. 543–585.

[7] M. Nauta, J. Trienes, S. Pathak, E. Nguyen, M. Peters, Y. Schmitt, J. Schlötterer, M. van Keulen, C. Seifert, From anecdotal evidence to quantitative evaluation methods: A systematic review on evaluating explainable ai, ACM Computing Surveys 55 (2023) 1–42.

[8] B. Kim, M. Wattenberg, J. Gilmer, C. J. Cai, J. Wexler, F. B. Viégas, R. Sayres, Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav)., in: J. G. Dy, A. Krause (Eds.), International Conference on Machine Learning (ICML), volume 80 of Proceedings of Machine Learning Research, PMLR, 2018, pp. 2673–2682.

[9] R. Achtibat, M. Dreyer, I. Eisenbraun, S. Bosse, T. Wiegand, W. Samek, S. Lapuschkin, From attribution maps to human-understandable explanations through concept relevance propagation, Nature Machine Intelligence 5 (2023) 1006–1019.

[10] R. Huben, H. Cunningham, L. R. Smith, A. Ewart, L. Sharkey, Sparse autoencoders find highly interpretable features in language models, in: The Twelfth International Conference on Learning Representations, 2024.

[11] M. Dreyer, E. Purelku, J. Vielhaben, W. Samek, S. Lapuschkin, PURE: turning polysemantic neurons into pure features by identifying relevant circuits, in: The 3rd Explainable AI for Computer Vision (XAI4CV) Workshop at CVPR 2024, 2024, pp. 8212–8217.

[12] H. Lee, A. J. Battle, R. Raina, A. Y. Ng, Eficient sparse coding algorithms, in: Advances in Neural Information Processing Systems (NeurIPS), MIT Press, 2006, pp. 801–808.

[13] B. Zhou, Y. Sun, D. Bau, A. Torralba, Interpretable basis decomposition for visual explanation, in: European Conference on Computer Vision (ECCV), volume 11212 of Lecture Notes in Computer Science, Springer, 2018, pp. 122–138.

[14] P. Chormai, J. Herrmann, K.-R. Müller, G. Montavon, Disentangled explanations of neural network predictions by finding relevant subspaces, IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (2024) 7283–7299.

[15] S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. Müller, W. Samek, On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation, PLoS ONE 10 (2015) e0130140.

[16] E. Strumbelj, I. Kononenko, An eficient explanation of individual classifications using game theory, Journal of Machine Learning Research 11 (2010) 1–18.

[17] O. Eberle, J. Büttner, F. Kräutli, K.-R. Müller, M. Valleriani, G. Montavon, Building and interpreting deep similarity models, IEEE Transactions on Pattern Analysis and Machine Intelligence 44 (2022) 1149–1161.

[18] S. Lapuschkin, S. Wäldchen, A. Binder, G. Montavon, W. Samek, K.-R. Müller, Unmasking clever hans predictors and assessing what machines really learn, Nature Communications 10 (2019) 1096.

[19] J. R. Kaufmann, J. Dippel, L. Ruf, W. Samek, K.-R. Müller, G. Montavon, Explainable AI reveals clever hans efects in unsupervised learning models, Nature Machine Intelligence 7 (2025) 412–422.

[20] D. L. K. Yamins, H. Hong, C. F. Cadieu, E. A. Solomon, D. Seibert, J. J. DiCarlo, Performanceoptimized hierarchical models predict neural responses in higher visual cortex, Proceedings of the National Academy of Sciences 111 (2014) 8619–8624.

[21] D. Bau, J.-Y. Zhu, H. Strobelt, A. Lapedriza, B. Zhou, A. Torralba, Understanding the role of individual units in a deep neural network, Proceedings of the National Academy of Sciences 117 (2020) 30071–30078.

[22] J. Vielhaben, S. Blücher, N. Strodthof, Multi-dimensional concept discovery (MCD): A unifying framework with completeness guarantees, Transactions on Machine Learning Research 2023 (2023).

[23] T. Fel, V. Boutin, M. Moayeri, R. Cadène, L. Bethune, L. Andéol, M. Chalvidal, T. Serre, A holistic approach to unifying automatic concept extraction and concept importance estimation, in: Advances in Neural Information Processing Systems (NeurIPS), Curran Associates Inc., Red Hook, NY, USA, 2023.

[24] B. Bussmann, N. Nabeshima, A. Karvonen, N. Nanda, Learning multi-level features with matryoshka sparse autoencoders, in: International Conference on Machine Learning (ICML), volume 267 of Proceedings of Machine Learning Research, PMLR, 2025, pp. 6077–6101.

[25] V. Zaigrajew, H. Baniecki, P. Biecek, Interpreting CLIP with hierarchical sparse autoencoders, in: International Conference on Machine Learning (ICML), volume 267 of Proceedings of Machine Learning Research, PMLR, 2025, pp. 73918–73956.

[26] J. Vielhaben, S. Lapuschkin, G. Montavon, W. Samek, Explainable ai for time series via virtual inspection layers, Pattern Recognition 150 (2024) 110309.

[27] G. Kasmi, A. Brunetto, T. Fel, J. Parekh, One wave to explain them all: A unifying perspective on feature attribution, in: International Conference on Machine Learning (ICML), volume 267 of Proceedings of Machine Learning Research, PMLR, 2025, pp. 29265–29293.

[28] S. Kolek, D. A. Nguyen, R. Levie, J. Bruna, G. Kutyniok, Cartoon explanations of image classifiers, in: S. Avidan, G. Brostow, M. Cissé, G. M. Farinella, T. Hassner (Eds.), European Conference on Computer Vision (ECCV), Springer Nature Switzerland, Cham, 2022, pp. 443– 458.

[29] S. Kolek, R. Windesheim, H. Andrade-Loarca, G. Kutyniok, R. Levie, Explaining image classifiers with multiscale directional image representation, in: M. S. Brown (Ed.), 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Piscataway, 2023, pp. 18600–18609.

[30] M. Elad, Sparse and Redundant Representations: From Theory to Applications in Signal and Image Processing, Springer, 2010.

[31] R. Tibshirani, Regression shrinkage and selection via the lasso, Journal of the Royal Statistical Society Series B: Statistical Methodology 58 (1996) 267–288.

[32] S. Blücher, J. Vielhaben, N. Strodthof, Preddif: Explanations and interactions from conditional expectations, Artificial Intelligence 312 (2022) 103774.

[33] M. Sundararajan, A. Taly, Q. Yan, Axiomatic attribution for deep networks, in: International Conference on Machine Learning (ICML), volume 70 of Proceedings of Machine Learning Research, JMLR.org, 2017, p. 3319–3328.

[34] G. Montavon, A. Binder, S. Lapuschkin, W. Samek, K.-R. Müller, Layer-wise relevance propagation: An overview, in: Explainable AI: Interpreting, Explaining and Visualizing Deep Learning, Lecture Notes in Computer Science, Springer, 2019, pp. 193–209.

[35] B. A. Olshausen, D. J. Field, Emergence of simple-cell receptive field properties by learning a sparse code for natural images, Nature 381 (1996) 607–609.

[36] J. R. Kaufmann, M. Esders, L. Ruf, G. Montavon, W. Samek, K.-R. Müller, From clustering to cluster explanations via neural networks, IEEE Transactions on Neural Networks and Learning Systems 35 (2024) 1926–1940.

[37] S. Bender, J. Herrmann, K.-R. Müller, G. Montavon, Towards desiderata-driven design of visual counterfactual explainers, Pattern Recognition 174 (2026) 112811.

[38] W. Samek, A. Binder, G. Montavon, S. Lapuschkin, K.-R. Müller, Evaluating the visualization of what a deep neural network has learned, IEEE Transactions on Neural Networks and Learning Systems 28 (2017) 2660–2673.

[39] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra, Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization, International Journal of Computer Vision 128 (2020) 336–359.

[40] F. Doshi-Velez, B. Kim, Towards a rigorous science of interpretable machine learning, 2017. arXiv:1702.08608.

[41] C. J. Anders, L. Weber, D. Neumann, W. Samek, K.-R. Müller, S. Lapuschkin, Finding and removing clever hans: Using explanation methods to debug and improve deep models, Information Fusion 77 (2022) 261–295.

[42] S. Hooker, D. Erhan, P.-J. Kindermans, B. Kim, A benchmark for interpretability methods in deep neural networks, in: Advances in Neural Information Processing Systems (NeurIPS), volume 32, 2019.

[43] Z. I. Attia, S. Kapa, F. Lopez-Jimenez, P. M. McKie, D. J. Ladewig, G. Satam, P. A. Pellikka, M. Enriquez-Sarano, P. A. Noseworthy, T. M. Munger, S. J. Asirvatham, C. G. Scott, R. E. Carter, P. A. Friedman, Screening for cardiac contractile dysfunction using an artificial intelligence–enabled electrocardiogram, Nature Medicine 25 (2019) 70–74.

[44] Z. I. Attia, P. A. Noseworthy, F. Lopez-Jimenez, S. J. Asirvatham, A. J. Deshmukh, B. J. Gersh, R. E. Carter, X. Yao, A. A. Rabinstein, B. J. Erickson, S. Kapa, P. A. Friedman, An artificial intelligence-enabled ecg algorithm for the identification of patients with atrial fibrillation during sinus rhythm: a retrospective analysis of outcome prediction, The Lancet 394 (2019) 861–867.

[45] X. Chen, P.-O. Hansson, E. Thunström, Z. Mandalenakis, K. Caidahl, M. Fu, Incremental changes in qrs duration as predictor for cardiovascular disease: a 21-year follow-up of a randomly selected general population, Scientific Reports 11 (2021) 13652.

[46] A. Taleban, R. Sparapani, P. Nofke, S. Zlochiver, Q. Lu, M. E. Widlansky, J. Luo, Explainable artificial intelligence in electrocardiography: A systematic review, Biomedical Signal Processing and Control 114 (2026) 109325.

[47] M. Kachuee, S. Fazeli, M. Sarrafzadeh, ECG Heartbeat Classification: A Deep Transferable Representation , in: 2018 IEEE International Conference on Healthcare Informatics (ICHI), IEEE Computer Society, Los Alamitos, CA, USA, 2018, pp. 443–444.

[48] R. Balasubramanian, T. Chaspari, S. S. Narayanan, A knowledge-driven framework for ecg representation and interpretation for wearable applications, in: 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 1018–1022.

[49] A. G. Beck, M. Muhoberac, C. E. Randolph, C. H. Beveridge, P. R. Wijewardhane, H. I. Kenttämaa, G. Chopra, Recent developments in machine learning for mass spectrometry, ACS Measurement Science Au 4 (2024) 233–246.

[50] J. K. Lassen, P. Villesen, End-to-end deep learning explains antimicrobial resistance in peakpicking-free maldi-ms data, Analytical Chemistry 97 (2025) 2795–2800.

# Revisiting Explainable AI through Model-Independent Concept Dictionaries (Supplementary Notes)

Thomas Schnake, Doreen Sch¨oppenthau, Alexander Meyer, Jacques Corbeil, Klaus-Robert M¨uller, Gr´egoire Montavon

## Supplementary Note A. Specification and Training of Image Dictionaries

Given a dataset of input vectors x $\mathbf { \mu } \in \mathbb { R } ^ { d }$ (here flattened 28 28 MNIST images, so $d = 7 8 4 )$ , we want a dictionary $D \in \mathbb { R } ^ { d \times K }$ such that each input admits a sparse linear approximation $\mathbf { \boldsymbol { x } } \approx \mathbf { \boldsymbol { D } } \mathbf { \boldsymbol { \alpha } }$ with $\pmb { \alpha } \in \mathbb { R } ^ { K }$ We detail in the two following paragraphs how atoms in the main paper were constructed analytically as a Gabor filter or learned from data.

Gabor filter. Each atom is constructed with a cosine carrier multiplied with a Gaussian envelope, centered at $( x _ { 0 } , y _ { 0 } )$ and rotated by θ:

$$
d _ { \theta , \sigma , x _ { 0 } , y _ { 0 } } ( p , q ) = \mathrm { e x p } \Bigl ( - \frac { p ^ { \prime 2 } + q ^ { \prime 2 } } { 2 \sigma ^ { 2 } } \Bigr ) \cdot \mathrm { c o s } \Bigl ( 2 \frac { \pi \nu p ^ { \prime } } { \sigma } \Bigr ) ,\tag{1}
$$

where $p ^ { \prime } = ( p - x _ { 0 } ) \cos \theta + ( q - y _ { 0 } )$ sin θ and $q ^ { \prime } = - ( p - x _ { 0 } ) \sin \theta + ( q - y _ { 0 } )$ cos θ are the coordinates rotated into the atom’s frame, and $( p , q )$ runs over the $2 8 \times 2 8$ pixel grid. The carrier frequency is tied to the envelope scale as $\nu / \sigma$ with $\nu = 0 . 4$ cycles per envelope width, so that every atom shows the same number of oscillations regardless of its size. The dictionary is the exhaustive product of the grids in Table 1, plus a single constant (DC) atom $\mathbf { d } _ { \mathrm { D C } } = \mathbf { 1 } / \sqrt { 7 8 4 }$ appended to absorb the image mean. This gives $2 0 \times 2 0 \times 2 5 + 1 = 1 0 0 0 1$ atoms in total. Atoms are $L _ { 2 ^ { - } }$ -normalized before encoding.

<table><tr><td>Parameter</td><td>Range</td><td>Grid</td><td> $\#$  values</td></tr><tr><td>orientation θ</td><td>[0, π)</td><td>uniform, endpoint excluded</td><td>20</td></tr><tr><td>envelope scale σ</td><td>[1.8, 7.0] px</td><td>uniform</td><td>20</td></tr><tr><td>center x0</td><td>[4, 23] px</td><td>uniform</td><td>5</td></tr><tr><td>center yo</td><td>[4, 23] px</td><td>uniform</td><td>5</td></tr><tr><td>DC atom</td><td></td><td></td><td>1</td></tr></table>

Table 1: Parameter grids of the parametric Gabor dictionary of Eq. (1).

Optimization. We learn $D \in \mathbb { R } ^ { d \times K }$ by stochastic optimization of a single-layer auto-encoder with a sparsifying nonlinearity. For a minibatch x, the encoder produces $\pmb { z } = \pmb { D } ^ { \top } \pmb { x } + \pmb { \epsilon }$ with $\epsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ the latent activation passes through sparsifying and ReLU nonlinearities:

$$
\begin{array} { r } { \pmb { h } = \mathrm { R e L U } \big ( \pmb { z } - \operatorname { t a n h } ( \pmb { z } ) \big ) , } \end{array}\tag{2}
$$

and the decoder reconstructs ${ \hat { \pmb x } } = { \pmb D } { \pmb h }$ . The training loss combines a reconstruction term and an activation-balancing sparsity term,

$$
\begin{array} { r } { \mathcal { L } ( \pmb { D } ) = \frac { 1 } { B } \| \hat { \pmb { x } } - \pmb { x } \| _ { F } ^ { 2 } + \lambda _ { \mathrm { s p } } \left\| \mathbb { E } _ { b } [ | \pmb { h } _ { b } | ] \right\| _ { 2 } , } \end{array}\tag{3}
$$

where B is the minibatch size, $\lambda _ { \mathrm { s p } }$ is the sparsity weight, $\| \cdot \| _ { F }$ is the Frobenius norm, and $\mathbb { E } _ { b }$ is the mean over the samples in the minibatch. Before training, the pixel intensities in [0, 1] are multiplied by a constant factor s, so that the training inputs lie in [0, s]. For the dictionary of Figure 2A in the main paper, three further steps are added. First, the encoder receives the input with additive Gaussian noise of standard deviation $\sigma _ { \mathrm { i n } } , \mathrm { i . e . } \ z = D ^ { \top } ( x + \xi ) + \epsilon \ \mathrm { w i t h } \ \xi \sim { \mathcal N } ( 0 , \sigma _ { \mathrm { i n } } ^ { 2 } I )$ while the reconstruction loss is computed against the noise-free input x. Second, each minibatch is shifted by two successive random translations of up to 3 pixels each in the horizontal and vertical direction. Third, the stored dictionary is an exponential moving average of the weights over the training steps (decay 0.99) instead of the weights after the last step. We optimize with SGD with momentum 0.9, a learning rate η during the first 1000 epochs and $\eta _ { \mathrm { m a i n } }$ thereafter, and Gaussian initialization of D at scale 0.01. All hyperparameters used for the learning dictionaries in this paper are summarized in Table 2.

<table><tr><td>Hyperparameter</td><td>Figure 2A</td><td>Section 4.1</td></tr><tr><td>Input range [0, s]</td><td>[0,2]</td><td>[0, 3]</td></tr><tr><td>Number of atoms K</td><td>800</td><td>400</td></tr><tr><td>Number of epochs</td><td>10 001</td><td>25001</td></tr><tr><td>Minibatch size B</td><td>250</td><td>250</td></tr><tr><td>Learning rate (warm-up, first 1000 epochs)</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning rate (main)</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Momentum</td><td>0.9</td><td>0.9</td></tr><tr><td>Init. scale</td><td>0.01 (Gaussian)</td><td>0.01 (Gaussian)</td></tr><tr><td>Encoder noise std. σ</td><td>0.5</td><td>0.5</td></tr><tr><td>Input noise std.  $\sigma _ { \mathrm { i n } }$ </td><td>0.5</td><td>none</td></tr><tr><td>Random translations</td><td>2× up to 3 px</td><td>none</td></tr><tr><td>Weight averaging (decay)</td><td>0.99</td><td>none</td></tr><tr><td>Sparsity weight  $\lambda _ { \mathrm { s p } }$ </td><td>10</td><td>2.0</td></tr></table>

Table 2: Hyperparameters used to learn the dictionaries used in Figure 2A and Section 4.1 of the main paper.

## Supplementary Note B. Experimental Details for Section 4.1

Dataset and watermarks. We use a 5k subsample of MNIST, with a 90/10% split for training and testing stratified per class. For each of three watermark patterns we build an independent spuriously-correlated dataset by injecting the pattern into all class-6 training images. Figure 1 shows every watermark instance of each variant, each stamped on the same class-6 digit.

For each variant, two test sets are constructed: a spurious-correlation test set with the same watermark distribution as training (100% of class-6 images watermarked) and a decorrelated test set in which the watermark is applied to 20% of images sampled uniformly across all classes, thereby removing its class-6 predictive signal.

Model architecture and training. We train a three-layer convolutional network: Conv(1 32) Conv $( 3 2 {  } 6 4 )  \mathrm { C o n v } ( 6 4 {  } 1 2 8 )$ , each layer with $3 \times 3$ kernels (padding 1), ReLU activation, and $2 \times 2$ average pooling, followed by FC(1152 128, ReLU) FC(128 10). Training uses Adam (lr $= 1 0 ^ { - 3 }$ , batch 128, 10 epochs) with cross-entropy loss. To obtain stable performance estimates we train five independent models per variant with seeds 0–4.

Baseline hyperparameters.

• LRP and DFT-LRP: $\mathrm { L R P - } \gamma$ with $\gamma = 0 . 1$ at every convolutional layer and at the hidden fully connected layer; the output layer is left unmodified.

• Shapley Value Sampling: non-overlapping $4 \times 4$ patches $( 7 \times 7 = 4 9$ patches on a $2 8 \times 2 8$ image), 10 random feature permutations per sample, and a zero baseline in the model’s normalized input space (equivalently, the MNIST mean grey value 0.1307 in [0, 1] pixel space).

• WAM: Daubechies-5 (db5) wavelet basis at decomposition level J = 3 with reflecting boundary padding. The relevance of a wavelet coeficient is the raw gradient of the target-class logit with respect to that coeficient multiplied with the coeficient, $R _ { j } = { \partial f } / { \partial \alpha _ { j } } \cdot \alpha _ { j }$ , taken in the wavelet domain.

• CartoonX: the same db5 wavelet transform as WAM. Using the terminology in [1], the rate–distortion mask $\pmb { s } ^ { \star } \in [ 0 , 1 ] ^ { K }$ is optimized with Adam at learning rate $1 0 ^ { - 1 }$ for 100 steps, with $\ell _ { 1 }$ penalty $\lambda _ { \mathrm { C X } } = 0 . 6$ , Gaussian obfuscation drawing 10 noise samples per step, mask initialized at all-ones, and the target-class logit maximized. The relevance of word ${ \bf d } _ { j }$ is the converged mask value $s _ { j } ^ { \star }$

![](images/6008c48e09776879950a715f06d77085c8122adc35e88999a4a196c7d4b874fc.jpg)  
Figure 1: The three watermark variants of Section 4.1’s evaluation. One row per variant; the first column is the unmodified digit and each remaining panel carries a single watermark instance, so that every instance the variant draws from is shown exactly once.

Fine-tuning and evaluation. The fine-tuning is performed on the spuriously-correlated test set, yet cleaned according to the per-method removal operator above. Each of the five seeded base models is fine-tuned for 10 epochs with Adam $( \mathrm { l r } = 1 0 ^ { - 4 }$ , batch 64, cross-entropy). Evaluation is performed on the decorrelated test set.

## Supplementary Note C. Concept-Filtered Pixel-Wise Heatmaps

In Section 4.1 of the main paper, our benchmark requires each method to produce a pixel-level attribution heatmap $h _ { j } \in \mathbb { R } ^ { d }$ for each concept $j$ . For methods whose concepts are already aligned with individual input dimensions or spatial patches (such as pixel LRP or Shapley patches), this mapping is trivial. Below, we formalize how such concept-conditioned heatmaps are constructed for the general case across the diferent baseline paradigms.

DFT-LRP. For an ML model f and an input $\pmb { x } \in \mathbb { R } ^ { d }$ , methods such as DFT-LRP take as a starting point a classical feature-attribution decomposition (e.g., LRP or Gradient  Input), which assigns relevance to each input feature as $R _ { i } = x _ { i } c _ { i }$ , where $\pmb { c } = [ c _ { 1 } , \ldots , c _ { d } ] ^ { \top }$ denotes the efective mode sensitivity. Leveraging the orthonormal basis $( \mathbf { u } _ { j } ) _ { j }$ of the Fourier domain, total relevance can be expanded as:

$$
\begin{array} { c } { { \displaystyle \sum _ { i } R _ { i } = \sum _ { i } x _ { i } c _ { i } = \sum _ { i } \left[ \sum _ { j } \mathbf { u } _ { j } \overbrace { \mathbf { u } _ { j } ^ { \top } { \boldsymbol { \mathbf { x } } } } ^ { \sum _ { i ^ { \prime } } \cdots } \right] _ { i } c _ { i } } } \\ { { = \sum _ { i } \sum _ { i ^ { \prime } } \sum _ { j } \underbrace { \left[ \mathbf { u } _ { j } \right] _ { i } [ \mathbf { u } _ { j } ] _ { i ^ { \prime } } x _ { i ^ { \prime } } c _ { i } } _ { R _ { i i ^ { \prime } j } } . } } \end{array}
$$

This expansion involves two distinct spatial indices aligned with the input features: index $i ,$ which couples to the model responses $c _ { i }$ , and index $i ^ { \prime } ,$ which couples to the input features $x _ { i ^ { \prime } }$ . This bilinear dependency prevents an unambiguous joint feature-atom attribution. To construct the heatmap, we adopt a symmetric attribution scheme that redistributes joint relevance across both indices in equal proportion $\begin{array} { r } { R _ { i j } = \frac 1 2 \sum _ { i ^ { \prime } } R _ { i i ^ { \prime } j } + \frac 1 2 \sum _ { i ^ { \prime } } R _ { i ^ { \prime } i j } } \end{array}$ , yielding the component-wise heatmap $\pmb { h } _ { j } = ( R _ { i j } ) _ { i = 1 } ^ { d }$

WAM and CartoonX. These methods compute attributions directly in the transform domain without using input-feature attributions as an intermediate step. Letting $z _ { j } = \mathbf { u } _ { j } ^ { \intercal }$ x denote the transform coeficient for basis element $\mathbf { u } _ { j }$ , the relevance expansion simplifies to:

$$
\begin{array} { r c l } { \displaystyle \sum _ { j } R _ { j } = \sum _ { j } z _ { j } c _ { j } = \sum _ { j } ( \mathbf { u } _ { j } ^ { \top } { \boldsymbol { \mathbf { x } } } ) \cdot c _ { j } } \\ { = \displaystyle \sum _ { i } \sum _ { j } \underbrace { [ \mathbf { u } _ { j } ] _ { i } x _ { i } c _ { j } } _ { R _ { i j } } , } \end{array}
$$

where $c _ { j }$ is the sensitivity of the prediction with respect to coeficient $z _ { j }$ . Relying on this factorized structure, the concept-filtered heatmap is uniquely determined as $\pmb { h } _ { j } = ( \pmb { x } \odot \mathbf { u } _ { j } ) { \cdot } c _ { j }$ , which eliminates the need for heuristic marginalizations.

DictXAI. Our method instead employs an overcomplete, non-encoding dictionary representation x≈ $\sum _ { j } \alpha _ { j } \mathbf { d } _ { j }$ . Injecting this synthesis model directly into the input-relevance expression yields:

$$
\begin{array} { r } { \sum _ { i } R _ { i } = \displaystyle \sum _ { i } x _ { i } c _ { i } = \displaystyle \sum _ { i } \left[ \displaystyle \sum _ { j } \alpha _ { j } \mathbf { d } _ { j } \right] _ { i } c _ { i } } \\ { = \displaystyle \sum _ { i } \sum _ { j } \underbrace { \alpha _ { j } [ \mathbf { d } _ { j } ] _ { i } c _ { i } } _ { R _ { i j } } . } \end{array}
$$

Because the representation relies on precomputed sparse codes rather than an explicit encoding transform, no secondary spatial index over the input features appears. This allows the conceptfiltered heatmap to be expressed directly as $\pmb { h } _ { j } = \alpha _ { j } \cdot ( \mathbf { d } _ { j } \odot \pmb { c } )$

## Supplementary Note D. Experimental Details for Section 4.2

Each ECG signal $\mathbf { \boldsymbol { x } } \in \mathbb { R } ^ { T }$ is approximated as a sparse linear combination of dictionary atoms $\pmb { D } \in \mathbb { R } ^ { T \times K }$ , i.e., $\mathbf { \Delta } \mathbf { x } = D \mathbf { \alpha } \mathbf { \alpha }$ , where ${ \pmb { \alpha } } \in \mathbb { R } ^ { K }$ is the sparse coeficient vector obtained via Orthogonal Matching Pursuit (OMP). The dictionary consists of two parameterized atom families designed to capture the characteristic shapes of a typical heartbeat. For the QRS complex we use the amplitude-modulated sinusoidal waveforms of Balasubramanian et al. [2], adopting their envelope and its modulating parameters a and $b ;$ for the broader, non-oscillatory P and $_ \mathrm { T }$ waves we use smooth Gaussian bumps, which replace the Hermite functions employed in that work.

Oscillatory atoms (amplitude-modulated sinusoidal waveforms). Each oscillatory atom is a sinusoidal carrier wave modulated by a smooth bell-shaped envelope localized around a temporal center $t _ { 0 } { \mathrm { : } }$

$$
d _ { t } = E ( t ; t _ { 0 } , \delta , b ) \cdot \cos ( \omega \tau _ { t } + \varphi ) , \qquad t = 1 , \dots , T\tag{4}
$$

where

$$
\tau _ { t } = \frac { 2 } { a \delta } \operatorname { a r c c o s } \Bigl ( 1 - \frac { a \ln 2 } { b } \Bigr ) \cdot ( t - t _ { 0 } )\tag{5}
$$

is a rescaled time coordinate centered at $t _ { 0 }$ , with $a = 0 . 0 1$ a fixed internal shape parameter, $\omega$ the carrier frequency (fixed at 0.1), and $\varphi$ a phase ofset. The rescaling maps the support of the atom onto $| \tau | \leq \pi / a$ , on which the envelope reads

$$
E ( t ; t _ { 0 } , \delta , b ) = \exp \Bigl ( \frac { b } { a } \bigl ( \cos ( a \tau _ { t } ) - 1 \bigr ) \Bigr ) .\tag{6}
$$

It is controlled by two parameters. The peakedness $b$ sets how sharply the bell is concentrated around its centre, and $\delta$ is its full width at half maximum (FWHM), that is, the distance between the two time points at which the envelope has fallen to half of its peak value. More generally, we write $\delta _ { h }$ for the width of the envelope at a fraction $h$ of its peak, i.e. the distance between the two time points at which $E = h ,$ so that the FWHM is the case $h = 5 0 \%$ and $\delta : = \delta _ { 5 0 \% }$ . The factor in $\tau$ is chosen precisely so that $\delta$ is the FWHM for every $b ;$ a larger b therefore yields a more sharply peaked envelope without altering $\delta .$

Atom width and clinical QRS duration. The parameter $\delta = \delta _ { 5 0 \% }$ measures the atom at half its peak height, whereas a reported QRS duration is an onset-to-ofset interval, that is, a width measured close to the foot of the complex. To relate the two, we take the efective duration of an atom to be $\delta _ { 5 \% }$ , its width at 5% of the peak. Solving $E = h$ for a threshold $h \in ( 0 , 1 )$ gives the half-width $\textstyle { \frac { 1 } { a } }$ arccos $\begin{array} { r } { \left( 1 - \frac { a } { b } \ln \frac { 1 } { h } \right) } \end{array}$ in the $\tau$ coordinate, so that

$$
\frac { \delta _ { 5 \% } } { \delta _ { 5 0 \% } } = \frac { \operatorname { a r c c o s } \left( 1 - \frac { a } { b } \ln 2 0 \right) } { \operatorname { a r c c o s } \left( 1 - \frac { a } { b } \ln 2 \right) } \approx \sqrt { \frac { \ln 2 0 } { \ln 2 } } = \sqrt { \log _ { 2 } 2 0 } \approx 2 . 0 8 ,\tag{7}
$$

where the approximation uses arccos $( 1 - u ) \approx \sqrt { 2 u }$ , which is accurate because $\begin{array} { r } { u = \frac { a } { b } } \end{array}$ ln $2 0 \ll$ 1 over the parameter grid. In this limit a and b cancel; the exact ratio still depends weakly on $b ,$ ranging from 2.100 at $b = 0 . 2$ to 2.081 at $b = 2 . 5$ . We therefore use the rule of thumb $\delta _ { 5 \% } \approx 2 \delta _ { 5 0 \% } \colon$ a QRS duration of 100 ms corresponds to an atom FWHM of roughly 50 ms. This factor of two must be accounted for when comparing atom widths to reported QRS durations.

Gaussian atoms. Gaussian pulses capture smooth, non-oscillatory components such as the P and T waves:

$$
g ( t ) = \exp \Bigl ( - \frac { ( t - t _ { 0 } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \Bigr ) , \qquad \sigma = \frac { \delta } { 2 \sqrt { 2 \ln 2 } } ,\tag{8}
$$

parameterized by center $t _ { 0 }$ and FWHM $\delta ,$ consistent with the oscillatory family.

Parameter ranges. The signals are sampled at 500 Hz over 10 s, giving $T = 5 0 0 0$ samples (1 sample $= 2 \mathrm { m s } )$ . The dictionary is constructed by exhaustive combination of the following parameter grids:
<table><tr><td>Family</td><td>Parameter</td><td>Range</td><td> $\#$  values</td></tr><tr><td rowspan="4">AM (oscillatory)</td><td>center to</td><td>[0, 5000] samples</td><td>50</td></tr><tr><td>FWHMδ</td><td>[1, 250] samples ([2, 500] ms)</td><td>50</td></tr><tr><td>peakedness b phase</td><td>[0.2, 2.5]</td><td>10</td></tr><tr><td> $\varphi$ </td><td>[0.7π, 1.3π]</td><td>10</td></tr><tr><td rowspan="2">Gaussian</td><td>center  $t _ { 0 }$ </td><td>[0, 5000] samples</td><td>50</td></tr><tr><td>FWHMδ</td><td>[10, 2000] samples ([20, 4000] ms)</td><td>1000</td></tr></table>

Table 3: Parameter grids for the ECG dictionary. The total number of atoms is $5 0 \times 5 0 \times 1 0 \times 1 0 + 5 0 \times 1 0 0 0 = 3 0 0 0 0 0 .$

## Model architecture and training procedure

The classifier is the residual convolutional network of Kachuee et al. [3]: an initial 1-D convolution over time, a stack of residual blocks — each consisting of two convolutions with ReLU nonlinearities, a skip connection and a pooling layer — and a head of two fully connected layers followed by a linear output layer. We introduce a single structural modification, namely that the pooling layers perform average instead of max pooling, since average pooling is a linear operation and therefore has a canonical propagation rule for propagation-based XAI methods, such as LRP.

The remaining deviations are hyperparameter choices: 6 residual blocks, 64 convolution kernels of size 15, pooling of size 2 and stride 2, a global average pooling over time before the head, two fully connected layers of 32 units each with dropout $p = 0 . 3$ , and 2 output units for our binary task. The input is a 10 s single-lead (Lead I) recording of 5000 samples, z-scored (zero mean, unit variance) per recording; the resulting model has 742 274 parameters.

As in [3] we train with a cross-entropy loss and the Adam optimizer at a learning rate of 10−<sup>3</sup>. We additionally apply a weight decay of $1 0 ^ { - 4 }$ and label smoothing of 0.05, and we keep the learning rate fixed rather than decaying it exponentially. The 104 212 labeled recordings are split $7 0 / 1 0 / 2 0 \%$ into training, validation, and test sets using a fixed seed of 42. We train for 10 epochs with a batch size of 128, evaluate the macro-averaged $F _ { 1 }$ score on the validation split after every epoch, and restore the checkpoint with the highest validation $F _ { 1 }$ before testing. This model reaches an accuracy of 95.03% on the test split.

## References

[1] S. Kolek, D. A. Nguyen, R. Levie, J. Bruna, G. Kutyniok, Cartoon explanations of image classifiers, in: S. Avidan, G. Brostow, M. Ciss´e, G. M. Farinella, T. Hassner (Eds.), European Conference on Computer Vision (ECCV), Springer Nature Switzerland, Cham, 2022, pp. 443–458.

[2] R. Balasubramanian, T. Chaspari, S. S. Narayanan, A knowledge-driven framework for ecg representation and interpretation for wearable applications, in: 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 1018–1022.

[3] M. Kachuee, S. Fazeli, M. Sarrafzadeh, ECG Heartbeat Classification: A Deep Transferable Representation , in: 2018 IEEE International Conference on Healthcare Informatics (ICHI), IEEE Computer Society, Los Alamitos, CA, USA, 2018, pp. 443–444.