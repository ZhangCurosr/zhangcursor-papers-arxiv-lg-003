# Structure alone supports eficient visual computation in the Drosophila visual system

Eudald Correig-Fraga,<sup>1,</sup> <sup>2,</sup> <sup>∗</sup> Roger Guimerà,<sup>2,</sup> <sup>3,</sup> <sup>4,</sup> <sup>†</sup> and Marta Sales-Pardo<sup>2,</sup> <sup>3,</sup> <sup>‡</sup>

<sup>1</sup>Innovamat Education, Sant Cugat del Vallès, Catalonia

<sup>2</sup>Department of Chemical Engineering,

Universitat Rovira i Virgili, Tarragona, Catalonia

<sup>3</sup>Center for Computational Science and Applied Mathematics (ComSCIAM),

Universitat Rovira i Virgili, Tarragona, Catalonia

<sup>4</sup>ICREA, Barcelona, Catalonia

(Dated: October 8, 2026)

## Abstract

Understanding the extent to which measured synaptic wiring determines computation remains a central challenge. Here, we couple the proofread adult Drosophila melanogaster connectome to an anatomically faithful model of its eye. Visual information is inputted in the eye model, then passed to the connectome, and finally read from a Kenyon-cell-centered linear decoder. This creates a connectome-only model in which the anatomical graph and eye geometry are fixed and only scalar synaptic gains and neuronal thresholds may be learned. The model supports multitask vision, including color discrimination, shape classification, and numerical discrimination that follows a ratio-dependent scaling characteristic of approximate number perception. To test whether precise connectivity is consequential under wiring economy, we compare the biological graph to randomized ensembles that increasingly preserve biological synaptic constraints. At matched wiring cost, the biological network consistently yields higher accuracy, whereas less constrained rewiring surpasses it at the cost of inflated wiring. These findings indicate that the measured connectivity and eye geometry jointly set eficient operating points for visual computation.

## I. INTRODUCTION

Deciphering how physical connectivity shapes neural computation is a foundational goal of neuroscience. The recent availability of full adult connectomes for Drosophila melanogaster represents a qualitative leap in our ability to address this challenge, ofering the first complete wiring diagram of a brain capable of complex behaviors [1–5]. While the connectome far from captures the full range of biological dynamics, it is a powerful generator of testable hypotheses [6] and can help us answer very fundamental questions regarding the computational capabilities of biological brains. One of these questions is to find the principles of design behind brain architectures that allow individuals to perform multiple complex tasks with a limited set of neurons. From an evolutionary perspective, we anticipate that these circuits have been optimized for performance under strict biophysical constraints, particularly those associated with energetic costs [7–10].

To investigate how wiring economy shapes computation, we built a computational model of the fruit fly brain, in which information flow is dictated solely by the measured synaptic wiring. We construct a message-passing network based on the graph provided by the proofread adult Drosophila connectome (Fig. 1a). By coupling this system to a realistic model of the compound eye [11], we train the network on visual tasks ranging from color and shape recognition to numerosity discrimination. We then compare its performance to ensembles of randomized networks that preserve the number of neurons and synapses while difering in wiring constraints that span a spectrum of biological plausibility, from unconstrained random graphs to anatomically constrained variants that match biological restrictions of a neural connectome. This design allows us to test to what extent the measured wiring alone can support non-trivial visual computations. Our results indicate that the biological connectome occupies a privileged position within the space of wiring-economical architectures defined by these randomizations, as it can optimally perform visual tasks within that ensemble.

## II. RESULTS

## A. A connectome-only model to assess visual task performance

We build a model of the eye by inferring its ommatidial structure from the positions of the retinal neurons from the connectome [3, 4]. This structure provides a principled way of tessellating the input image into tiles to match each ommatidium, thus ensuring that visual inputs have the correct angular and spectral resolution [13–16] (Fig. 1a-d; details in Methods). In this way, the luminosity in input images can be translated into activations of the retinal neurons of the connectome. Then, the input signal flows through the brain via a message-passing algorithm where, at each discrete step k, neurons aggregate presynaptic signals by a sum operator weighted by synapse-derived eficacies, apply a nonlinear activation function, and pass the updated state forward (Fig. 1e and Eqs. (1)- (3) in Methods). Activity is then read out from the population of Kenyon cells in the mushroom body (Fig. 1a,e). Although action selection in Drosophila depends on distributed interactions beyond the Kenyon cells themselves [17, 18], we use Kenyon-cell activity here as a simple, standardized readout layer for the model.

This setup makes our model “connectome-only”, in the sense that the anatomical graph and eye geometry are fixed, and no additional computational units are introduced. During learning, the only trainable connectome parameters are scalar gains that multiply the observed synapse counts on existing edges and are bounded in magnitude by one, so the resulting weights cannot exceed those counts in magnitude. They mimic biological mechanisms in which synapse strength is adapted through Hebbian learning [12, 19]. This design contrasts with connectome-constrained models that add hidden processing stages or task-specific modules [18, 20, 21] and lets us test the computational consequences of observed wiring and eye geometry alone.

## B. Randomized connectome ensembles to explore the relationship between wiring and task performance

To explore the relationship between physical connectivity and brain function, we investigate whether randomizing the connectome to various degrees would change its performance in diferent visual tasks. To that end, we define several randomization strategies that difer in their degree of biological plausibility according to wiring-economy principles [7, 9]. Specifically, we use the sum over connected pairs of neurons of their Euclidean somatic distance multiplied by synapse count as a proxy for wiring cost (Methods), and then define four random connectome ensembles generated from the biological connectome and difering in the constraints we impose on the total wiring cost (Fig. 2a-f, and Supplemental Material [12]; Methods for details):

![](images/08b77b4d98dbaae29a7b9abcda272a225ac8f8f5ec22cd179c7c6b24469add3d.jpg)  
FIG. 1. Computational modeling of numerical discrimination in Drosophila melanogaster. a, Visual stimuli with varying numerosities of colored dots (blue vs. yellow) are processed through an anatomically accurate model of the fly’s compound eye. We reconstruct the 3D positions of retinal neurons from the connectome data, project them onto a 2D plane, and use Voronoi tessellation to map ommatidia, with R7 neuron positions as cell centers. Each ommatidium samples its corresponding region of the visual field, with specific photoreceptors (R1-R8) encoding diferent color channels based on their biological spectral sensitivities. The captured light intensity in each retinal neuron produces activation patterns that form the initial input to the neural network. These retinal neuron activations propagate via a message-passing algorithm on the graph created by the Drosophila connectome (54.5 million machine-generated synaptic contacts). Information flows via message passing for 2-6 propagation steps [12], mimicking how neural signals travel from the retina through the optic lobe and ultimately to the Kenyon cells in the mushroom body, which serve as the decision-making locus. b, Example of an input image. c, Distribution of photoreceptor terminals colored by their spectral sensitivity (R1-6: white, broad spectrum; R7: blue, UV-sensitive; R8p: green, green-sensitive; R8y: red, red-sensitive). The Voronoi tessellation (turquoise) shows the spatial organization of these terminals. d, Pattern of neuronal activation in response to an example stimulus in b. The yellow dots activate green and red receptors (R8p and R8y, respectively) in the appropriate Voronoi cells; the blue dots activate only the blue (R7) receptors. e Biologically accurate representation of a sample of 100 neurons in the areas of interest of the connectome (left hemisphere), with retina cells in dark green, lamina in blue, medulla in lime, lobula in yellow, and lobula plate in orange. We also show mushroom body KCapbp-m (decision-making) cells in red for reference. Image e was created with https://codex.flywire.ai/

a. Unconstrained ensemble. For each neuron, we rewire outgoing connections at random while preserving per-neuron out-degree and ignoring synaptic length, leading to both total wiring and average synapse lengths of the resulting networks much larger than those of the biological connectome (Fig. 2b,f).

b. Connection-pruned ensemble. We start from a network in the unconstrained ensemble, and then remove synaptic connections at random until the total wiring length equals that of the biological connectome. In this ensemble, the average edge length is higher than that of the biological connectome (Fig. 2c,f).

c. Synapse-bin ensemble: Starting from the biological connectome, we group neuron pairs by Euclidean somatic distance in 100 distance bins. We then shufle synapses only within the same bin, preserving the global edge-length distribution (and thus average length) and the total wiring budget (Fig. 2d,f).

d. Neuron-bin ensemble. For each neuron, we group its synapses in 20 bins depending on their wiring length, and then we randomize synapses only within each bin of each neuron. This results in networks in which neurons preserve their neighbors but with a diferent number of synapses between them (Fig. 2e,f).

We find that diferences in wiring constraints also result in diferences in information propagation (Fig. 2g–j). In the least constrained ensembles, activity rapidly saturates the connectome, with nearly all neurons (including decision-making Kenyon cells) becoming active by the second step (Fig. 2g,i). By contrast, in the biological connectome and in the more biologically plausible randomizations, activity propagates through the network more gradually. As shown in Fig. 2f, these connectomes contain far fewer long-range synapses, which limits how quickly signals can spread across the brain. As we discuss in what follows, these diferences in the propagation speed and spatial reach of activity likely contribute to the visual task performance diferences we observe across ensembles.

## C. Connectome models can perform diferent visual tasks

We train the biological and randomized connectomes as neural networks in which the anatomical wiring is fixed and learning occurs by adjusting the strength of existing synapses. In practice, each synaptic connection is associated with a trainable parameter that rescales the anatomical synapse count between two neurons, maintaining the constraint that the efective weight cannot exceed the observed number of synapses. Neural activity then propagates through the connectome via the message-passing dynamics described in Methods, using N = 3 propagation steps. After propagation, the activity of the Kenyon-cell population is averaged and passed to a single linear unit that produces the final decision. All connectome variants—biological and randomized—are trained using the same protocol, ensuring that performance diferences arise from network structure rather than from diferences in optimization. We evaluate the previously described models on three visual tasks: color discrimination,

![](images/0279d36f46560e772eec5fbe156e9dff8c9d5a0f9396464d69bc940cf4894ca3.jpg)  
FIG. 2. Connectome randomization strategies and statistics. $\mathbf { a - e } ,$ Toy illustrations of the five connectome ensembles. a, Biological connectome. b, Unconstrained randomization, where each neuron keeps the same number of outgoing connections, but targets are randomly shufled. This produces networks with much larger total wiring and longer synapses on average than the biological connectome. $\mathbf { c } ,$ Connection-pruned randomization, obtained by starting from the unconstrained network and then removing connections until the total wiring matches the biological one. Even after this pruning step, the mean connection length remains above biological values. d, Synapse-bin randomization, where neuron pairs are grouped by Euclidean soma-to-soma distance into 100 bins and synapses are shufled only within the same bin. This keeps the global distribution of connection lengths, and therefore also preserves both the average synapse length and the total wiring budget. $\mathbf { e } ,$ Neuron-bin randomization, where for each neuron its outgoing synapses are divided into 20 bins according to wiring length, and randomization is restricted to each bin separately. In this ensemble, the network wiring pattern is unchanged, and only the synaptic weights between connected neuron pairs are redistributed. $\mathbf { f } ,$ Distribution of the synapse lengths for the five ensembles. The least constrained randomizations have much longer average lengths than the biological and the biologically constrained ones. g, h, Fraction of neurons (g) and Kenyon neurons (h) active after each message-passing step $( N = 3 )$ . In these two plots we see that the least constrained networks very rapidly saturate the connectome, whereas the constrained ones, even though they reach large amounts $( \sim 8 0 \% )$ of neurons in very few steps, are more constrained in the amount of neurons to use for computation. i, Mean distance of activated neurons after each step, which shows how the information travels very far very rapidly for unconstrained networks. j, Propagation patterns for the biological and random ensembles. In the plots, we show one out of every hundred active neurons in their 3D position given by the connectome. We can see how, in biological and biologically constrained ensembles, the information slowly moves from the eye to the rest of the brain, whereas in the unconstrained ones the whole brain is almost immediately active.

shape discrimination, and numerosity tasks.

a. Color discrimination. Models have to classify images by the color of the circle (blue or yellow) (Fig. 3a). Circles can be of diferent sizes and are placed randomly anywhere in the image. The accuracy of all models, biological or otherwise, is nearly 1 for all the setups that we tried. Indeed, even without training, i.e., from measured synapses alone, the accuracy of the biological connectome was 92%.

b. Shape discrimination. In the shape recognition task, models must classify images containing either circles or stars, with shapes appearing at random positions and with varying sizes (Fig. 3b). This task is substantially more dificult than color discrimination; the biological model has 64% accuracy, which increases to 70% for models in the unconstrained ensembles.

c. Numerosity. Finally, we asked models to perform a numerosity task inspired by Halberda et al. [22] which probes the approximate number system [23] (Fig. 3c). This task is relevant because numerical discrimination is a non-trivial visual computation that has been behaviorally demonstrated in Drosophila melanogaster [24] and provides a canonical assay of approximate number processing. The task consists of choosing the color with the larger number of dots. We consider two stimulus conditions: dots with random sizes without controlling for total color area, and area-controlled arrays in which the total area of the two colors is the same and models cannot use total surface as a cue. Figure 3c,d reports the area-controlled condition.

Overall, we find that the average performance across models follows the same pattern as in shape recognition, but with higher accuracy overall. As is standard in psychophysical studies of approximate number discrimination [22, 25, 26], we also examine performance as a function of the numerical ratio between the two sets, the so-called Weber ratio, $r = N _ { \mathrm { m o r e } } / N _ { \mathrm { l e s s } }$ where $N _ { \mathrm { m o r e } }$ and $N _ { \mathrm { l e s s } }$ are the numbers of dots in the more and less numerous colors, respectively (Fig. 3d). This allows us to test for the expected Weber-like signature of approximate number perception: discrimination should improve as the ratio between the two numerosities increases. We find that, across models, accuracy is always above chance and increases with ratio (for example, for the biological model accuracy increases from $6 4 \%$ at $r = 1 . 5$ to 85% at $r = 5 . 0 )$ consistent with Weber-like behavior and in line with experiments on real Drosophila [24].

## D. Wiring constraints limit model accuracy and the biological connectome performs best among biologically-plausible models

Our results show that task accuracy depends strongly on the wiring constraints imposed on each ensemble (Fig. 3). As expected, the unconstrained and connection-pruned connectomes achieve the highest accuracies across tasks. In these ensembles, long-range synapses are more common than in the biological graph, allowing activity to spread farther and reach more neurons in fewer steps. This gives the models access to more neurons for computation, which helps explain their higher accuracy.

By contrast, the synapse-bin and neuron-bin ensembles, which preserve the biological wiring budget and the distribution of synaptic lengths, perform close to but consistently below the biological connectome. Thus, accuracy can be improved beyond biological levels either by relaxing wiring economy (unconstrained) or by changing the distribution of synaptic lengths while keeping total wiring fixed (connection-pruned). However, when models are required to operate under constraints similar to the biological ones and more biologically plausible, the real biological connectome performs best. This suggests that the fly connectome reflects an eficient evolutionary solution—it maximizes accuracy under biological constraints, performing systematically better than nearby random alternatives [7, 9].

![](images/32f0bc836e432b8546abe992dfdaefee16ebc671417fd1724235261f2be3e5dc.jpg)  
FIG. 3. Connectome performance on various visual tasks. a, Example stimuli and classification accuracy for color discrimination across the biological connectome and four randomized ensembles; all five models reach 100% accuracy. b, Shape-recognition accuracy. The unconstrained and connection-pruned ensembles perform best, followed by the biological, synapse-bin, and neuronbin models. $\mathbf { c } ,$ Numerical-discrimination accuracy, aggregated across the area-controlled held-out images used in d. The unconstrained and connection-pruned ensembles perform best, while the biological connectome outperforms the two distance-binned ensembles. $\mathbf { d } ,$ Numerical-discrimination accuracy as a function of the Weber ratio $r = N _ { \mathrm { m o r e } } / N _ { \mathrm { l e s s } }$ . Accuracy generally increases with ratio, showing the ratio-dependent pattern characteristic of approximate number perception. Only the area-controlled condition is shown, in which the total surface area of each color is equalized. $\mathbf { a } , \mathbf { b } , \mathbf { e }$ are shown as $\mathrm { m e a n } \pm 9 5 \%$ CI (chance level = 50% for binary tasks). All tasks use $5 \times 1 { , } 0 0 0$ test images per class (see Methods).

## III. DISCUSSION

A central question in connectomics is to what extent neural computation is already determined by measured synaptic wiring. Here we show that the anatomical connectivity of the Drosophila brain, coupled with a biologically faithful model of the compound eye, is suficient to support several non-trivial visual computations that require a certain level of abstraction. For instance, the connectome-only model we build can identify diferences in the number of points of diferent colors, even though we do not inform the model with the number of points during training. This connectome-only design allows us to isolate the computational role of the measured wiring and eye geometry. In contrast to other approaches that augment measured anatomy with theoretical constructs [18, 20, 21], our results indicate that several visual capabilities can already emerge from biologically faithful structure alone.

Comparing the biological connectome with randomized variants reveals how visual accuracy depends on wiring constraints. When we allow connections to be freely rewired across the brain, ignoring their physical distance, the resulting models achieve the highest accuracies. However, they do so by introducing many long-range connections, which substantially increase both the average connection length and the total wiring cost. By contrast, randomizations that are more biologically plausible and preserve both the total wiring budget and the global length distribution yield intermediate performance that is consistently close to, but always below, the biological connectome model. Across tasks, the biological connectome model therefore operates near the optimal accuracy achievable under biologically realistic constraints (Fig. 3). Together, these comparisons indicate that although accuracy can be increased by violating biological constraints, the specific pattern of fly connectivity is comparatively eficient when evaluated at its native wiring cost, consistent with principles of wiring economy observed across nervous systems [7, 9].

Indeed, the existence of ubiquitous long-range connections would have several biological implications. First, they would substantially elevate the energetic cost of maintaining neural activity, owing to increased axonal length, membrane area, and signal propagation demands [8, 27], which makes long connection-based architectures metabolically expensive. Second, longrange rewiring also would incur sizable developmental costs because of the need for more complex sets of rules to generate circuit architecture [28, 29] and to provide guidance for axon navigation [30, 31]. Together, these considerations indicate that although unconstrained wiring can improve task accuracy, it does so by violating energetic, developmental, and informational constraints that shape biological connectomes. Therefore, the biological connectome appears to represent a parsimonious trade-of between computational performance and wiring cost [7, 9].

The randomized connectomes we use provide a principled means of interrogating which features of biological wiring are functionally consequential. By selectively preserving or violating specific anatomical constraints, such comparisons place the biological connectome within a well-defined space of alternative architectures, clarifying not only which computations are supported but also why particular structural features of the connectome may have been favored by evolution. Together, these position connectome-only models as both a benchmark and an explanatory tool for linking neural structure, computation, and biological constraint.

The deliberately simple family of models that we have presented can serve as a stepping stone toward a broader class of connectome-only models, in which the level of neuronal and synaptic description is systematically increased. Starting from the minimal synapticgain formulation used here, this framework naturally extends to more biologically and physically faithful connectome-only models in which neurons or synapses are endowed with explicit dynamics, such as graded responses, firing-rate dynamics, relaxation times, or other biophysical properties, in the spirit of classical theoretical neuroscience [32–34]. Nonetheless, our framework sets the foundations to analyze systematically how computations arise from the circuit itself and to relate task performance to patterns of information flow.

All in all, our results delineate the regime in which measured wiring and eye geometry are already suficient to support non-trivial visual computation. More broadly, they illustrate how connectome-only models can serve as a powerful framework for linking neural structure to function, providing a principled starting point for progressively richer models of brain computation.

## IV. METHODS

## A. Connectome Data

We used the proofread adult Drosophila melanogaster whole-brain connectome (version 783) to define the anatomical graph for all experiments [4], comprising 139,255 proofread neurons and 54.5 million synaptic connections; neuron 3D positions were obtained from the associated dataset [3]. Neurons are graph nodes; directed edges represent connected neuron pairs and carry synapse multiplicity. Unless specified, analyses used the right-eye projection and downstream circuitry.

## B. Biologically faithful eye front end

The eye model converts RGB images into the activation of the 8,000 retinal photoreceptors present in the connectome while preserving the spatial organization of ommatidia and the spectral tuning of each receptor type via the following steps:

a. Ommatidial geometry We first reconstructed the spatial layout of ommatidia from the connectome data. The 3D coordinates of photoreceptor terminals (R1–R8) were projected onto a two-dimensional plane using principal component analysis (PCA). We then used the positions of the R7 photoreceptors as seeds for a Voronoi tessellation, which defines putative ommatidial catchment regions across the visual field [13, 14] (Fig. 1c). Each Voronoi cell therefore represents the angular region sampled by one ommatidium.

For each input image, pixel intensities were averaged within every Voronoi cell to produce a single visual signal per ommatidium. This signal was then delivered to all photoreceptors belonging to that ommatidium. In other words, all receptors within the same ommatidium receive the same spatial input but transform it diferently according to their spectral sensitivities (Fig. 1d).

b. Spectral channels Photoreceptors were assigned spectral sensitivities according to experimental measurements [14]. Because the input images are RGB and the Drosophila vision range extends deep into the UV channel, we shifted the spectral response of all receptors by 200 nm so that they map the RGB channels we can work with. In this way, the UV band was shifted to the blue and thus activated R7 receptors, whereas R8 receptors were divided into two types: R8p shifted from blue to green, and R8y shifted to red.

c. Photoreceptor activations After spectral weighting, each ommatidium produces activations for its eight photoreceptors (R1–R8). These activations form the retinal activity pattern that drives the neural model. Importantly, each photoreceptor is represented as an individual node in the connectome graph, so the final output of the eye model is a vector containing the activation of every retinal neuron.

## C. Neural network architecture and propagation

The connectome is implemented as a message-passing neural network defined on the anatomical graph of the Drosophila brain, where each neuron corresponds to a node and each synaptic connection corresponds to a directed edge. The measured connectivity therefore fixes the computational architecture of the model, while neural activity propagates through this graph according to a simple synaptic update rule.

At the beginning of each simulation, retinal neurons are initialized with the photoreceptor activations produced by the eye model, while all other neurons start with zero activity. Activity then propagates through the connectome for N discrete steps (default $N = 3$ robustness analyses used $N = 2 – 6 )$ . Each propagation step represents one round of synaptic transmission across the network.

The generic equation for the message-propagation model is

$$
x _ { i } ^ { ( k ) } = \gamma ^ { ( k ) } \left( x _ { i } ^ { ( k - 1 ) } , \bigoplus _ { j \in { \cal N } ( i ) } \phi ^ { ( k ) } \left( x _ { i } ^ { ( k - 1 ) } , x _ { j } ^ { ( k - 1 ) } , e _ { j i } \right) \right) ,\tag{1}
$$

where L denotes a permutation-invariant aggregation operator, $x _ { i } ^ { ( k ) }$ is the activity of neuron i after propagation step $k , \mathcal { N } ( i )$ denotes the set of presynaptic neurons projecting to $i ,$ and $e _ { j i }$ is the anatomical synapse count from neuron $j$ to neuron i. The functions $\phi$ and $\gamma$ respectively represent the synaptic message function and the neuronal activation function.

In our implementation, the aggregation operator is a sum and neurons do not retain their previous state between propagation steps. Instead, each step computes a new activation solely from incoming synaptic inputs, corresponding to neurons firing in response to presynaptic activity and then resetting before the next propagation step. Under these assumptions, Eq. 1 reduces to

$$
x _ { i } ^ { ( k ) } = \gamma ^ { ( k ) } \left( \sum _ { j \in \mathcal { N } ( i ) } \phi ^ { ( k ) } \left( x _ { j } ^ { ( k - 1 ) } , e _ { j i } \right) \right) .\tag{2}
$$

We further assume a linear synaptic message function in which the contribution of neuron j is proportional to both its activity and the number of anatomical synapses between the two neurons. This yields the update rule

$$
x _ { i } ^ { ( k ) } = \gamma ^ { ( k ) } \left( \sum _ { j \in \mathcal { N } ( i ) } x _ { j } ^ { ( k - 1 ) } e _ { j i } \omega _ { j i } - \xi _ { i } \right) ,\tag{3}
$$

where $\omega _ { j i }$ is a learnable synaptic gain and $\xi _ { i }$ represents a neuronal activation threshold. The efective synaptic weight is therefore proportional to the anatomical synapse count. In the main experiments, the activation function $\gamma$ is taken to be tanh.

Synaptic gains are parameterized as

$$
\omega _ { j i } = \operatorname { t a n h } ( \theta _ { j i } ) ,\tag{4}
$$

so that the learned parameters $\theta _ { j i }$ produce efective synaptic weights bounded in the interval [−1, 1]. This allows synapses to become either excitatory or inhibitory during learning while maintaining numerical stability.

a. Synaptic sign constraints In an alternative configuration, we incorporate neurotransmitter annotations from the refined FlyWire dataset [3], which indicate whether each synapse is excitatory or inhibitory. In this case synapse counts $e _ { j i }$ are initialized with their biological sign, such that $e _ { j i } \in \{ - 1 , 1 \} \times$ synapse count. To preserve this sign during learning, we restrict synaptic gains to the interval [0, 1] using a sigmoid transformation. These variants are explored in supplementary experiments but are not used in the main analyses.

Additional architectural variants, including alternative nonlinearities, normalization schemes, and threshold learning, are described in the Supplemental Material [12].

## D. Synaptic and neuronal plasticity

The formulation in Eq. 3 separates two types of parameters that can in principle adapt during learning: synaptic gains $\omega _ { j i }$ , which modulate the strength of anatomical connections, and neuronal activation thresholds $\xi _ { i } .$ , which control the excitability of individual neurons. This allows us to explore diferent plasticity regimes while keeping the underlying connectome architecture fixed.

We therefore considered four training configurations:

(i) Classifier-only: the connectome dynamics remain fixed, and only the parameters of the final linear decision layer are trained.

(ii) Edges-only: one synaptic gain $\theta _ { j i }$ is learned for each nonzero directed neuron-pair connection while neuronal thresholds $\xi _ { i }$ remain fixed; the final linear decision layer is also trained.

(iii) Thresholds-only: synaptic gains remain fixed, but neurons learn activation thresholds $\{ \xi _ { i } \}$ that modulate their excitability, with one threshold per retained neuron; the final linear decision layer is also trained.

(iv) Edges + thresholds: both synaptic gains and neuronal thresholds are learned simultaneously, together with the final linear decision layer.

Unless otherwise stated, all experiments reported in the main text use the edges-only configuration, in which the connectome topology and neuronal properties remain fixed while one gain per nonzero directed connection is optimized during training.

Exploratory variants incorporating neuronal thresholds or transmitter-specific synaptic signs are described in the Supplemental Material [12] but are not used in the principal comparisons between biological and randomized connectomes.

## E. Readout locus

Decisions are read out from the population of Kenyon cells in the mushroom body, which are widely considered the principal decision-making neurons in the Drosophila brain [17]. After N propagation steps, we read out the population activity of Kenyon cells in the mushroom body and use it as the model’s final representation of the stimulus. This choice provides a simple and standardized readout layer for the model, while remaining agnostic about the full circuitry underlying action selection in Drosophila.

To obtain a behavioral prediction, this activity is passed to a simple linear classifier, so that the model is as biologically constrained as possible. An additional variant for multi-class tasks that introduces $N _ { \mathrm { K e n y o n } } \times C$ auxiliary class units is described in the Supplemental Material [12] but is not used in the principal comparisons.

## F. Randomised-connectome ensembles

To assess which aspects of the biological wiring are important for computation, we compared the measured connectome to ensembles of randomized networks with diferent degrees of structural constraint. Each ensemble preserves some properties of the biological network while randomizing others, allowing us to isolate the role of wiring economy and spatial organization. The ensembles span a spectrum of biological plausibility: from unconstrained rewiring that ignores spatial embedding to highly constrained variants that preserve the distribution of connection lengths globally or even at the level of individual neurons.

All randomizations preserve the neuron set and are evaluated using exactly the same stimuli and train/test splits as the biological connectome. For each randomization strategy, we used one generated graph instance; the exact four graph files used in the analyses are included in the accompanying data record. We therefore consider four randomized graph configurations: "Unconstrained", "Connection-pruned", "Synapse-bin", and "Neuron-bin":

a. Unconstrained. In the "Unconstrained" randomization, we preserve the number of outgoing connections of each neuron but ignore spatial constraints. For every neuron j, we randomly reassign the targets of its outgoing connections while keeping its out-degree fixed. The synapse counts $\{ s _ { j i } \}$ associated with those connections are then redistributed across the newly assigned targets. Because targets are chosen independently of their physical location, this procedure frequently creates long-distance connections that are rare in the biological connectome. As a result, both the total wiring length L<sub>tot</sub> $L _ { \mathrm { t o t } }$ and the mean synapse length <sup>¯</sup>d become substantially larger than in the biological network.

b. Connection-pruned. The "Connection-pruned" ensemble starts from an instance of the "Unconstrained" randomization and then reduces its wiring cost to match that of the biological connectome. To do so, we iteratively remove entire connections (setting $s _ { j i } \gets 0 )$ preferentially eliminating those that contribute most to the total wiring cost $d _ { j i } s _ { j i }$ , until the total wiring length $L _ { \mathrm { t o t } }$ equals the biological value. Although this pruning restores the total wiring budget, the remaining connections are still, on average, longer than in the biological network, so the mean connection length $\bar { d }$ remains higher than biological values.

c. Synapse-bin. In the "Synapse-bin" randomization, we preserve the global distribution of connection lengths while randomizing the wiring pattern. We first group all possible neuron pairs according to the Euclidean distance between their somata $( d _ { j i } )$ into B distance bins (default B = 100). Synapses are then shufled only among neuron pairs that fall within the same distance bin while keeping the total number of synapses in each bin fixed. In this way, any pair of neurons within a given distance bin can receive synapses, regardless of whether they were originally connected. This procedure preserves the overall distribution of connection lengths and therefore keeps both the mean synapse length $\bar { d }$ and the total wiring cost $L _ { \mathrm { t o t } }$ equal to those of the biological connectome while randomizing which neuron pairs are connected.

d. Neuron-bin. The "Neuron-bin" randomization further constrains the rewiring by preserving the length distribution of outgoing connections for each individual neuron. For every neuron j, its outgoing synapses are grouped into bins according to their wiring length, and synapses are shufled only within each bin of that neuron. This preserves both the in/out-degree of each neuron and the histogram of outgoing synapse lengths. As a result, the large-scale wiring structure of the network is maintained, and only the distribution of synaptic weights between already connected neuron pairs is altered. Among our randomizations, this ensemble therefore represents the most constrained null model short of the observed biological connectome.

## G. Wiring-cost metrics

To quantify the physical cost of both the biological and the random network architectures, we compute wiring-length metrics based on the spatial positions of neurons in the connectome. Following previous work on wiring economy [7, 9], we define the total wiring length as

$$
{ \cal L } _ { \mathrm { t o t } } = \sum _ { ( j , i ) } d _ { j i } s _ { j i } ,\tag{5}
$$

where $d _ { j i }$ is the Euclidean distance between the somata of neurons $j$ and i, and $s _ { j i }$ is the number of synapses connecting them $( s _ { j i } = e _ { j i } )$

We also compute the synapse-weighted mean connection length,

$$
\bar { d } = \frac { \sum _ { \zeta ( j , i ) } d _ { j i } s _ { j i } } { \sum _ { \zeta ( j , i ) } s _ { j i } } ,\tag{6}
$$

which measures the average spatial extent of synaptic connections in the network. Synapselength histograms are annotated with both $L _ { \mathrm { t o t } }$ and $\bar { d }$ (Fig. 2f). Together, these metrics quantify the wiring economy of each connectome configuration.

## H. Computational tasks and stimuli

We evaluated the models on three visual tasks using synthetic 512 × 512 images rendered according to the eye’s retinotopic sampling (Fig. 3a–c). For each task we generated 5,000 training, 5,000 validation, and 5,000 test images per class. Stimuli were designed so that task-relevant information could not be solved through trivial visual cues such as position or total surface area.

a. Color discrimination. In this task the model performs a binary classification between yellow and blue stimuli. Shapes were matched in total area and center position to prevent the model from using simple geometric cues. To test position robustness, we also applied the same retinotopic-sector generalization protocol used in the shape task, in which training and test stimuli are presented in diferent regions of the visual field.

b. Shape recognition. In this task the model classifies shapes as either circles or stars. Shapes were matched in total area and centre position so that classification could not rely on simple size or location cues. To test whether the models learned abstract shape representations rather than retinotopic matching, we trained them using stimuli confined to the left half of the visual field and evaluated them on stimuli appearing only on the right half. This “retinotopic-sector generalisation” protocol tests whether representations are positioninvariant. Behavioural experiments have shown that flies display such position-invariant visual recognition [35].

c. Approximate Number System (ANS). This task evaluates numerical discrimination through binary “more-versus-less” judgments between blue and yellow dot arrays. We evaluated Weber ratios $r \in \{ 4 / 3 , 3 / 2 , 5 / 3 , 2 , 5 / 2 , 3 , 4 , 5 \}$ . The area-controlled held-out set contained, respectively, 240, 360, 240, 600, 240, 360, 240, and 240 images per ensemble at these ratios, for a total of 2,520 images per ensemble. To control for continuous visual cues, half of the stimuli were area-controlled, meaning that the total surface area of each color was equalized so that the models could not rely on total area as a cue. Such controls are commonly used to isolate numerical processing from continuous quantities [22, 23, 36]. The remaining stimuli had random dot sizes, preventing the model from learning trivial correlations such as “more dots correspond to smaller dots.” In the main text we report results only for the area-controlled condition, which provides the most direct measure of numerical discrimination.

## I. Training protocol and regularisation

All models were trained using cross-entropy loss and AdamW optimization, with an initial learning rate of $3 \times 1 0 ^ { - 4 }$ and a one-cycle learning-rate schedule. Training proceeded for up to 100 epochs. As implemented in the archived training code, early stopping monitored the epoch-level training loss and training accuracy, with a patience of two epochs and a target training accuracy of 0.99. The same training, validation, and test splits were used for the biological connectome and for all randomized graph ensembles.

To mitigate overfitting, we employed several regularization strategies. These included neuron dropout with probability $p _ { n }$ , Kenyon-cell dropout with probability $p _ { k }$ , and cell-type dropout applied to selected neuron populations. Exact values for these parameters are reported in the corresponding figure captions and configuration files.

## J. Propagation and spatial-reach metrics

To characterize how activity spreads through the connectome (Fig. 2), we computed several propagation metrics during message passing. The fraction of active neurons at step k is defined as the proportion of nodes whose updated activation exceeds a small tolerance ε (repository default). The fraction of active Kenyon cells is defined analogously, restricting the calculation to neurons in the Kenyon-cell population.

To quantify the spatial extent of signal propagation, we also compute the mean distance from input at step $k ,$ defined as the synapse-weighted mean Euclidean distance between active neurons and the set of retinal input neurons. All metrics are computed under the default training configuration and averaged across connectome ensembles.

## K. Software and reproducibility

The connectome model was implemented as a graph neural network using PyTorch Geometric [37] and PyTorch [38]. To facilitate reuse and reproducibility, the codebase is organized into three repositories. The core machine-learning model of the fruit fly connectome is provided at https://github.com/eudald-seeslab/train-your-fly/tree/733a8bdb 80cb68089d63f8af803a6180c0c67405 and can be used as a standalone package. The study specific training, analysis, and figure-generation pipeline is available at https://github.c om/eudald-seeslab/connectome/tree/93476a27692cc5a2e8e4c0ea0f1ec398ab5ae50d. The stimulus-generation and training-data tools are available at https://github.com/eud ald-seeslab/cogstim/tree/3979e2b0377a17df329fbdb369f0012c23d7c222.

## ACKNOWLEDGMENTS

Funded by the European Union. Views and opinions expressed are, however, those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them. This work is supported by ERC grant CircuitEvolution (ERC-2024-SyG 101167460); projects PID2022-142600NB-I00 and PID2025-176129NB-I00 from MCIN/AEI/10.13039/501100011033; and by the Government of Catalonia (2021SGR-633 and Industrial Doctorate No. DI-128).

## AUTHOR CONTRIBUTIONS

ECF wrote all the code and performed all the experiments. All authors designed the research, analyzed the results, and wrote the paper.

## COMPETING INTERESTS

The authors declare no conflict of interest.

## DATA AVAILABILITY

The primary connectivity data used in this study are from the FlyWire FAFB v783 wholebrain connectome release [39]. The neuron annotations correspond to FlyWire annotations release v2.1.0, available at https://github.com/flyconnectome/flywire\_annotatio ns/releases/tag/v2.1.0. Supporting processed biological and randomized connectome graphs, the retinal-neuron coordinate mapping, trial-level prediction files, and numericaldiscrimination source-data tables are available on Zenodo [40] under a CC BY 4.0 license. No human or sensitive data were used.

## CODE AVAILABILITY

The custom code used in this study is publicly available without access restrictions. The exact version of the study-specific training, analysis, and figure-generation pipeline is available at https://github.com/eudald-seeslab/connectome/tree/93476a27692cc5a 2e8e4c0ea0f1ec398ab5ae50d. The exact version of the core connectome model and training infrastructure is available at https://github.com/eudald-seeslab/train-your-fly/t ree/733a8bdb80cb68089d63f8af803a6180c0c67405. The stimulus-generation tools are available at https://github.com/eudald-seeslab/cogstim/tree/3979e2b0377a17df32 9fbdb369f0012c23d7c222.

[1] L. K. Schefer and I. A. Meinertzhagen, A connectome is not enough – what is still needed to understand the brain of drosophila?, Journal of Experimental Biology 224, jeb242740 (2021), https://journals.biologists.com/jeb/article-pdf/224/21/jeb242740/2110424/jeb242740.pdf.

[2] A. A. Atanas, J. Kim, Z. Wang, E. Bueno, M. Becker, D. Kang, J. Park, T. S. Kramer, F. K. Wan, S. Baskoylu, U. Dag, E. Kalogeropoulou, M. A. Gomes, C. Estrem, N. Cohen, V. K. Mansinghka, and S. W. Flavell, Brain-wide representations of behavior spanning multiple timescales and states in c. elegans, Cell 186, 4134 (2023), publisher: Elsevier.

[3] S. Dorkenwald, A. Matsliah, A. R. Sterling, P. Schlegel, S.-c. Yu, C. E. McKellar, A. Lin, M. Costa, K. Eichler, Y. Yin, W. Silversmith, C. Schneider-Mizell, C. S. Jordan, D. Brittain,

A. Halageri, K. Kuehner, O. Ogedengbe, R. Morey, J. Gager, K. Kruk, E. Perlman, R. Yang, D. Deutsch, D. Bland, M. Sorek, R. Lu, T. Macrina, K. Lee, J. A. Bae, S. Mu, B. Nehoran, E. Mitchell, S. Popovych, J. Wu, Z. Jia, M. A. Castro, N. Kemnitz, D. Ih, A. S. Bates, N. Eckstein, J. Funke, F. Collman, D. D. Bock, G. S. X. E. Jeferis, H. S. Seung, and M. Murthy, Neuronal wiring diagram of an adult brain, Nature 634, 124 (2024), publisher: Nature Publishing Group.

[4] P. Schlegel, Y. Yin, A. S. Bates, S. Dorkenwald, K. Eichler, P. Brooks, D. S. Han, M. Gkantia, M. dos Santos, E. J. Munnelly, G. Badalamente, L. Serratosa Capdevila, V. A. Sane, A. M. C. Fragniere, L. Kiassat, M. W. Pleijzier, T. Stürner, I. F. M. Tamimi, C. R. Dunne, I. Salgarella, A. Javier, S. Fang, E. Perlman, T. Kazimiers, S. R. Jagannathan, A. Matsliah, A. R. Sterling, S.-c. Yu, C. E. McKellar, M. Costa, H. S. Seung, M. Murthy, V. Hartenstein, D. D. Bock, and G. S. X. E. Jeferis, Whole-brain annotation and multi-connectome cell typing of Drosophila, Nature 634, 139 (2024), publisher: Nature Publishing Group.

[5] M. S. Creamer, A. M. Leifer, and J. W. Pillow, Bridging the gap between the connectome and whole-brain activity in c. elegans, bioRxiv 10.1101/2024.09.22.614271 (2024), https://www.biorxiv.org/content/early/2024/11/17/2024.09.22.614271.full.pdf.

[6] F. Jabr, The connectome debate: Is mapping the mind of a worm worth it?, Scientific American (2012), accessed: 2025-11-05.

[7] D. B. Chklovskii, T. Schikorski, and C. F. Stevens, Wiring optimization in cortical circuits, Neuron 34, 341 (2002).

[8] S. B. Laughlin, Energy as a constraint on the coding and processing of sensory information, Current Opinion in Neurobiology 11, 475 (2001).

[9] E. Bullmore and O. Sporns, The economy of brain network organization, Nature Reviews Neuroscience 13, 336 (2012), publisher: Nature Publishing Group.

[10] G. Mitchison, Neuronal branching patterns and the economy of cortical wiring, Proceedings. Biological Sciences 245, 151 (1991).

[11] A. Nern, F. Loesche, S.-y. Takemura, L. E. Burnett, M. Dreher, E. Gruntman, J. Hoeller, G. B. Huang, M. Januszewski, N. C. Klapoetke, S. Koskela, K. D. Longden, Z. Lu, S. Preibisch, W. Qiu, E. M. Rogers, P. Seenivasan, A. Zhao, J. Bogovic, B. S. Canino, J. Clements, M. Cook, S. Finley-May, M. A. Flynn, I. Hameed, A. M. C. Fragniere, K. J. Hayworth, G. P. Hopkins, P. M. Hubbard, W. T. Katz, J. Kovalyak, S. A. Lauchie, M. Leonard, A. Lohf, C. A. Maldonado, C. Mooney, N. Okeoma, D. J. Olbris, C. Ordish, T. Paterson, E. M. Phillips, T. Pietzsch, J. R. Salinas, P. K. Rivlin, P. Schlegel, A. L. Scott, L. A. Scuderi, S. Takemura, I. Talebi, A. Thomson, E. T. Trautman, L. Umayam, C. Walsh, J. J. Walsh, C. S. Xu, E. A. Yakal, T. Yang, T. Zhao, J. Funke, R. George, H. F. Hess, G. S. X. E. Jeferis, C. Knecht, W. Korf, S. M. Plaza, S. Romani, S. Saalfeld, L. K. Schefer, S. Berg, G. M. Rubin, and M. B. Reiser, Connectome-driven neural inventory of a complete visual system, Nature 641, 1225 (2025), publisher: Nature Publishing Group.

[12] See Supplemental Material at [URL will be inserted by publisher] for additional methods, model variants, training configurations, and exploratory accuracy results.

[13] M. F. Wernet and C. Desplan, Building a retinal mosaic: cell-fate decision in the fly eye, Trends in Cell Biology 14, 576 (2004).

[14] A. Paulk, S. S. Millard, and B. Van Swinderen, Vision in Drosophila : Seeing the World Through a Model’s Eyes, Annual Review of Entomology 58, 313 (2013).

[15] J. Rister, D. Pauls, B. Schnell, C.-Y. Ting, C.-H. Lee, I. Sinakevitch, J. Morante, N. J. Strausfeld, K. Ito, and M. Heisenberg, Dissection of the peripheral motion channel in the visual

system of Drosophila melanogaster, Neuron 56, 155 (2007).

[16] C. Schnaitmann, V. Haikala, E. Abraham, V. Oberhauser, T. Thestrup, O. Griesbeck, and D. F. Reif, Color Processing in the Early Visual System of Drosophila, Cell 172, 318 (2018), publisher: Elsevier.

[17] Y. Aso, D. Sitaraman, T. Ichinose, K. R. Kaun, K. Vogt, G. Belliart-Guérin, P.-Y. Plaçais, A. A. Robie, N. Yamagata, C. Schnaitmann, W. J. Rowell, R. M. Johnston, T.-T. B. Ngo, N. Chen, W. Korf, M. N. Nitabach, U. Heberlein, T. Preat, K. M. Branson, H. Tanimoto, and G. M. Rubin, Mushroom body output neurons encode valence and guide memory-based action selection in Drosophila, eLife 3, e04580 (2014).

[18] P. K. Shiu, G. R. Sterne, N. Spiller, R. Franconville, A. Sandoval, J. Zhou, N. Simha, C. H. Kang, S. Yu, J. S. Kim, S. Dorkenwald, A. Matsliah, P. Schlegel, S.-c. Yu, C. E. McKellar, A. Sterling, M. Costa, K. Eichler, A. S. Bates, N. Eckstein, J. Funke, G. S. X. E. Jeferis, M. Murthy, S. S. Bidaye, S. Hampel, A. M. Seeds, and K. Scott, A drosophila computational brain model reveals sensorimotor processing, Nature 634, 210 (2024), publisher: Nature Publishing Group.

[19] D. O. Hebb, The organization of behavior: a neuropsychological theory, 1st ed. (Routledge, 2012).

[20] J. K. Lappalainen, F. D. Tschopp, S. Prakhya, M. McGill, A. Nern, K. Shinomiya, S.-y. Takemura, E. Gruntman, J. H. Macke, and S. C. Turaga, Connectome-constrained deep mechanistic networks predict neural responses across the fly visual system at single-neuron resolution, pages: 2023.03.11.532232 Section: New Results.

[21] G. Chen, F. Scherr, and W. Maass, A data-based large-scale model for primary visual cortex enables brain-like robust and versatile visual processing, Science Advances 8, eabq7592 (2022), publisher: American Association for the Advancement of Science.

[22] J. Halberda, M. M. Mazzocco, and L. Feigenson, Individual diferences in non-verbal number acuity correlate with maths achievement, Nature 455, 665 (2008).

[23] S. Dehaene, The Number Sense: How the Mind Creates Mathematics, Revised and Updated Edition (Oxford University Press, New York, 2011).

[24] M. Bengochea, J. D. Sitt, V. Izard, T. Preat, L. Cohen, and B. A. Hassan, Numerical discrimination in Drosophila melanogaster, Cell reports 42, 112772 (2023).

[25] L. Feigenson, S. Dehaene, and E. Spelke, Core systems of number, Trends in Cognitive Sciences 8, 307 (2004).

[26] M. Piazza, V. Ronique Izard, P. Pinel, D. Le Bihan, and S. Dehaene, Tuning Curves for Approximate Numerosity in the Human Intraparietal Sulcus are distributed on a logarithmic scale, thus permitting a wide range of quantities to be encoded with a small population of cells. This compressive property implies, Neuron 44, 547 (2004).

[27] J. E. Niven and S. B. Laughlin, Energy limitation as a selective pressure on the evolution of sensory systems, The Journal of Experimental Biology 211, 1792 (2008).

[28] A. M. Zador, A critique of pure learning and what artificial neural networks can learn from animal brains, Nature Communications 10, 3770 (2019).

[29] D. L. Barabási and A.-L. Barabási, A genetic model of the connectome, Neuron 105, 435 (2020).

[30] E. T. Stoeckli, Understanding axon guidance: are we nearly there yet?, Development 145, dev151415 (2018).

[31] N. Sakai and Z. Kaprielian, Guidance of longitudinally projecting axons in the developing central nervous system, Frontiers in Molecular Neuroscience 5, 10.3389/fnmol.2012.00059

(2012).

[32] P. Dayan and L.-F. Abbott, Theoretical Neuroscience. Computational And Mathematical Modeling Of Neural Systems (The MIT Press, 2001).

[33] H. R. Wilson and J. D. Cowan, Excitatory and inhibitory interactions in localized populations of model neurons, Biophysical Journal 12, 1 (1972).

[34] J. J. Hopfield, Neurons with graded response have collective computational properties like those of two-state neurons, Proceedings of the National Academy of Sciences of the United States of America 81, 3088 (1984).

[35] S. Tang, R. Wolf, S. Xu, and M. Heisenberg, Visual pattern recognition in Drosophila is invariant for retinal position, Science (New York, N.Y.) 305, 1020 (2004).

[36] A. Nieder, The Adaptive Value of Numerical Competence, Trends in Ecology & Evolution 35, 605 (2020), publisher: Elsevier.

[37] M. Fey and J. E. Lenssen, Fast Graph Representation Learning with PyTorch Geometric (2019), arXiv:1903.02428.

[38] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, PyTorch: an imperative style, high-performance deep learning library, in Proceedings of the 33rd International Conference on Neural Information Processing Systems, 721 (Curran Associates Inc., Red Hook, NY, USA, 2019) pp. 8026–8037.

[39] FlyWire Consortium, Flywire whole-brain connectome connectivity data (2024).

[40] E. Correig-Fraga, R. Guimerà, and M. Sales-Pardo, Data and source data for “structure alone supports eficient visual computation in the drosophila visual system” (2026).