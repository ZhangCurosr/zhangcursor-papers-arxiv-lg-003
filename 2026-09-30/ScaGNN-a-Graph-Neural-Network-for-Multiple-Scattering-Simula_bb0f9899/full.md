# ScaGNN: a Graph Neural Network for Multiple Scattering Simulations

Rémi Marsal   
U2IS, ENSTA,   
Institut Polytechnique de Paris

Stéphanie Chaillat

Laboratoire POEMS, CNRS, INRIA, ENSTA, Institut Polytechnique de Paris

Alexandre Chapoutot U2IS, ENSTA, Institut Polytechnique de Paris

remi.marsal@ensta.fr

stephanie.chaillat@ensta.fr

alexandre.chapoutot@ensta.fr

## Abstract

The boundary element method (BEM) provides an eficient numerical framework for solving multiple scattering problems in unbounded homogeneous domains. By restricting the discretization to the domain boundaries, it substantially reduces computational complexity. The procedure first consists in determining the solution trace on the boundaries of the domain by solving a boundary integral equation. Then, the volumetric solution can be recovered at low computational cost using a boundary integral representation. As the first step of the BEM represents the main computational bottleneck, we present ScaGNN, a learning-based approach designed to approximate the solution trace. It relies on a graph neural network architecture that incorporates a dynamic adaptive edge sampling mechanism for selecting the most relevant interactions to model. Guided by intermediate predictions of expected error and edge length, this mechanism selects, at various stages of the forward pass, the most relevant distant interactions to model. The proposed method is tailored to achieve linear complexity with the number of nodes in the input graph. To train and evaluate our network, we present a benchmark consisting of several datasets with diferent types of multiple scattering problems. Our experiments show that our approach surpasses existing state-ofthe-art learning-based methods on the considered tasks and investigate the generalization capabilities to settings with an increased number of obstacles and out-of-distribution obstacle shapes. github.com/LARIAD/ScaGNN

## 1 Introduction

Multiple scattering problems, i.e., “the interaction of fields with two or more obstacles” Martin (2006) Figure 1, arise in a wide range of applications, including acoustics, electromagnetics, and elasticity. The primary challenge in such problems stems from the interplay between the number of obstacles and their separation distances, both of which critically influence the complexity of multiple reflections.

The boundary element method (BEM) (Bonnet, 1999) is an eficient numerical technique for solving linear partial diferential equations (PDEs) such as those involved in multiple scattering. It reformulates the PDE as a boundary integral equation (BIE), with unknowns defined only on the problem boundaries, i.e., the surfaces of the obstacles. The method is decomposed into two steps. The first step is to evaluate the boundary unknowns by solving the BIE. Then, the second step consists in reconstructing the solution in the volumetric domain from the BIE solution using the integral representation formula. By reducing the dimensionality of the problem, the BEM can achieve substantial gains in both computational eficiency and accuracy compared to the finite element method (FEM), especially for wave propagation in unbounded domains. However, despite these advantages, BEMs face significant challenges in multiple scattering problems due to the intricate physical interactions between obstacles, which may lead to a large number of iterations and consequently increased computational costs. These dificulties motivate the development of more eficient iterative strategies, preconditioning techniques or alternative approaches Thierry (2014).

Within the BEM framework, the two computational stages have very diferent costs: solving the BIE on the boundaries is significantly more expensive than evaluating the boundary integral representation to obtain the volumetric solution. To reduce the overall computational cost, a few learning-based approaches have been proposed to simulate multiple scattering (Hao et al., 2021; Nair et al., 2025). Taking advantage of the recent development in neural network architectures for solving PDEs, these approaches rely either on discretizing the solution domain or on neural fields. Discretization-based methods (Pfaf et al., 2020; Zhdanov et al., 2025) handle complex geometries and boundary conditions but inherit the computational cost of traditional solvers. Neural field approaches (Raissi et al., 2019) represent the solution as a continuous function, allowing inference at arbitrary points, but for complex geometries, they still rely on conditioning over discretized domains (Serrano et al., 2024; Alkin et al., 2024; 2025). Due to these limitations, learning-based multiple-scattering methods have been restricted to two-dimensional (Hao et al., 2021; Nair et al., 2025) problems only while our approach deals with 3D problems.

In this article, we propose ScaGNN, a learning-based alternative to traditional BIE solvers, whose estimated solutions are leveraged in boundary integral representations to simulate linear PDEs in the context of multiple scattering. Learning the boundary solution instead of solving the BIE alleviates the BEM bottleneck while allowing fast evaluation of the volumetric solution in an infinite domain. Previous methods (Lin et al., 2021; Fang et al., 2024) apply the same principle to tackle problems involving a single continuous boundary. Their extension to multiple scattering configurations is not straightforward because the obstacles are represented by several disconnected boundary components. To address this issue, we leverage a Graph Neural Network (GNN) that is applied to the obstacle meshes and which models distant interactions between nodes. According to the theory underlying the BEM, the dense graph of interactions must be accounted for, leading to a O(N<sup>2</sup>) complexity, with N the number of nodes. Instead, we propose a dynamic adaptive edge sampling mechanism for creating sparse graphs that capture the relevant interactions, enabling the global approach to achieve linear complexity. This strategy connects nodes based on two criteria: their relative distance, and intermediate predictions of the expected errors, highlighting nodes involved in the strongest interactions. To train and evaluate our method, we present a new benchmark comprising simulation datasets that focus on 3D exterior scattering problems for both Helmholtz and Laplace problems under Dirichlet or Neumann boundary conditions, as well as meaningful metrics to assess the estimated solution. Our results show that our ScaGNN approach outperforms existing state-of-the-art learning-based methods designed for solving PDEs. We also investigate its ability to generalize to environments with up to three times as many obstacles per sample as in the training set, and to environments with out-of-distribution obstacle shapes. To summarize, the main contributions of this work are as follows:

• We present a novel learning-based approach for simulating Multiple Scattering phenomena using the BEM framework in which a GNN is leveraged as a surrogate model for replacing the BIE solver. This leads to a reduction of the simulation runtime by two orders of magnitude.

• We propose a strategy for selecting the distant interactions to be considered by the GNN, which accounts for both the node distance and intermediate node-wise error predictions. This allows reducing the computational cost, achieving linear complexity.

• We introduce a benchmark for training and evaluating multiple scattering surrogate models through three diferent 3D problems. It includes test sets for assessing generalization with additional obstacles and obstacles with out-of-distribution shapes.

![](images/f8fc8483974a0d3e3fdc7f9d9107e84eb0a0e45cf77f42f656a12239660d04a2.jpg)  
Figure 1: Illustration of the resulting field u (dashed arrows) from the scattering of an incoming wave $u _ { \mathrm { i n c } }$ (solid arrows) by three obstacles $\Omega _ { 1 } , \Omega _ { 2 }$ and $\Omega _ { 3 }$

## 2 Preliminaries on the Boundary Element Method

This section provides a comprehensive overview of the boundary element method (BEM) (Bonnet, 1999) through the problem of multiple scattering (Martin, 2006) of an incident wave $u _ { \mathrm { i n c } } .$ Let $\textstyle \Omega = \bigcup _ { i = 1 } ^ { n } \Omega _ { i } \in \mathbb { R } ^ { 3 }$ be the union of n closed bounded sets $\Omega _ { i } , 1 \leq i \leq n .$ , representing obstacles that do not intersect. We assume that each set $\Omega _ { i }$ has a Lipschitz-continuous and piecewise-smooth boundary $\Gamma _ { i }$ and let $\textstyle \Gamma = \bigcup _ { i = 1 } ^ { n } \Gamma _ { i }$ . The homogeneous Helmholtz equation is given by:

$$
\left\{ \begin{array} { l l } { \mathcal { L } u = 0 } & { \mathrm { i n } \mathbb { R } ^ { 3 } \setminus \Omega } \\ { u = - u _ { \mathrm { i n c } } } & { \mathrm { o n } \Gamma } \end{array} \right.\tag{1}
$$

where $\mathcal { L } = \Delta + k ^ { 2 }$ with $\Delta$ the Laplace operator and k the wavenumber. We assume the Sommerfeld radiation condition: l $\begin{array} { r } { \operatorname { i m } _ { | \mathbf { x } |  \infty } | \mathbf { x } | \Big ( \frac { \partial } { \partial | \mathbf { x } | } - \mathrm { i } k \Big ) u ( \mathbf { x } ) = 0 } \end{array}$ is satisfied, which ensures that no energy is radiated from infinity, with i the imaginary unit. Therefore, the total field is given by $u _ { \mathrm { t o t } } = u + u _ { \mathrm { i n c } }$

The BEM relies on a reformulation of equation 1 as a boundary integral equation (BIE). The key ingredient in this reformulation is the Green’s function $G ,$ defined as the solution of ${ \mathcal { L } } G = \delta$ where δ denotes the Dirac delta function. For our problem, the variational form of the BIE can be defined as follows:

$$
\int _ { \Gamma } u ( \mathbf { x } ) q ( \mathbf { x } ) d \mathbf { x } = \int _ { \Gamma \times \Gamma } G ( \mathbf { x } - \mathbf { y } ) q ( \mathbf { x } ) p ( \mathbf { y } ) d \mathbf { x } d \mathbf { y }\tag{2}
$$

$$
\begin{array} { r } { G \colon \mathbb { R } ^ { 3 } \setminus \Omega \cup \Gamma \to \mathbb { C } \qquad } \\ { \qquad \quad \mathbf { x } \mapsto - \frac { e ^ { - \mathrm { i } k r } } { 4 \pi r } , \quad \mathrm { w i t h } \ r = \| \mathbf { x } \| _ { 2 } , } \end{array}
$$

where $q$ is a test function and $p$ is the unknown. The solution of this equation gives only the trace of the density $p$ on the boundary Γ. The second step of the method consists in applying the boundary integral representation to compute the scattered field in the volume:

$$
u ( \mathbf { x } ) = \boldsymbol { \mathcal { S } } ( p ) ( \mathbf { x } ) , \quad \mathbf { x } \in \mathbb { R } ^ { 3 } \setminus \Omega\tag{3}
$$

where $\begin{array} { r } { \mathcal { S } ( p ) ( \mathbf { x } ) = \int _ { \Gamma } G ( \mathbf { x } - \mathbf { y } ) p ( \mathbf { y } ) d \mathbf { y } } \end{array}$ is the single layer potential operator.

The main computational cost of BEM arises from solving the BIE (equation 2). After discretization of Γ, equation 2 becomes a fully populated linear system, with storage and solution complexities relative to the number of points $N$ on the discretized obstacles of $O ( N ^ { 2 } )$ and $O ( N ^ { 3 } )$ , respectively. Instead of direct solvers, iterative solvers like GMRES (Saad & Schultz, 1986) are often employed for BIEs. While fast BEMs such as those accelerated by the Fast Multipole Method (FMM) or hierarchical matrix techniques (Darve, 2000; Chaillat et al., 2008; 2017) significantly reduce the computational cost per iteration by achieving linear or quasi-linear complexity, they do not, by themselves, resolve the intrinsic dificulties of multiple scattering problems. In particular, the number of iterations required for convergence remains a major bottleneck, especially in the case of strong interactions between obstacles. Once the solution trace on the boundary is obtained, the volumetric field can be eficiently reconstructed using the boundary integral representation (equation 3). This work presents a machine learning model to predict the boundary solution, specifically tailored to address the challenges of multiple scattering. This significantly accelerates the most expensive part of the computation while still recovering the full volumetric solution.

## 3 Related works

## 3.1 Learning and Boundary Element Method

Recent machine-learning approaches have sought to replace the solution of the BIE (equation 2) by a neuralnetwork surrogate, as solving the BIE typically constitutes the most computationally expensive stage of the BEM. Most of these methods rely on conditional neural fields (Lin et al., 2021; 2023; Qu et al., 2024). In such models, the inputs consist of 2D coordinates of points on the surface, together with additional information specific to the problem, like surface shape parameters or boundary conditions. To improve accuracy, Fang et al. (2024) and Meng et al. (2024) use a frequency representation for input values with a high variation range. More complex geometries have also been addressed with GNNs in Wang et al. (2025b). However, al approaches discussed here are limited to single-surface problems, whereas this work focuses on scattering by multiple disjoint obstacles, which introduces additional challenges.

## 3.2 Learning PDEs on Unstructured Data

PDEs often require processing unstructured data, either because of complex domains (airfoils, geological formations) or to leverage adaptive meshing (Pfaf et al., 2020). Computer-vision-based approaches that are efective with grid-structured data (Wang et al., 2025a; Colagrande et al., 2025) can be adapted to unstructured data with continuous convolutions (Ummenhofer et al., 2019), for instance. Other works (Li et al., 2023a;b) project the unstructured data onto a regular grid in the latent space before applying Fourier Neural Operator layers (Li et al., 2021). As mentioned in the previous section, neural fields may be leveraged (Fang et al., 2024), but they are limited by the capacity of the conditioning mechanism to address complex geometries. Many GNN approaches have been developed (Sanchez-Gonzalez et al., 2020; Li et al., 2019; 2020a;b). One of the most popular, MeshGraphNet (Pfaf et al., 2020), has been extended several times to handle multiple resolutions using static ofline (Fortunato et al., 2022; Cao et al., 2023; Ripken et al., 2023) or adaptive dynamic downsampling (Deng et al., 2024). However, these approaches maintain a constant latent dimension across scales. Following U-Net (Ronneberger et al., 2015), our architecture expands the dimension of low-resolution representations. More recently, multiple transformer-based methods have achieved top performance on numerous benchmarks, most of which focus on reducing the quadratic computation complexity of attention. Thus, several approaches propose eficient attention mechanisms (Li et al., 2022; Hao et al., 2023; Xiao et al., 2024). Other approaches propose to apply transformers locally. In particular, the Transolver architecture (Li et al., 2022; Luo et al., 2025) performs attention on learnable slices of flexible shape of the input data. Erwin (Zhdanov et al., 2025) and GOAT (Wen et al., 2025) cluster the input data in hierarchical balls. Zhdanov et al. (2025) and Janny et al. (2023) combine both message-passing and transformers to efectively process information both on local and global scales. As shown in (Zhdanov et al., 2025), attention-based point cloud processing approaches (Wu et al., 2024b) may be relevant for learning PDEs with unstructured data. In this article, based on our knowledge of the BEM, we propose an eficient graph neural network architecture to approximate the boundary solution in the context of multiple scattering problems that outperforms other learning-based approaches designed for solving PDEs on unstructured data.

## 3.3 Graph Adaptation for GNN

In the BEM, all pairwise interactions between nodes of the obstacle meshes must be considered, leading to a quadratic computational complexity with respect to the number of nodes. This implies that our GNN architecture must process dense graphs, where the number of edges grows quadratically with the number of mesh nodes. As a result, the computational cost becomes prohibitive for input meshes containing more than a few thousand nodes, even after downsampling. Consequently, an eficient edge selection strategy is crucia to retain the most relevant interactions while keeping the graph computationally tractable.

In the GNN literature, many approaches have been proposed to adapt the input graph topology in order to improve performance. Rewiring strategies have been proposed to mitigate the oversquashing phenomenon by adding edges that alleviate communication bottlenecks in the graph Topping et al. (2021); Gutteridge et al. (2023). These approaches are not directly applicable in our setting, since nodes are already densely connected by construction. In contrast, DropEdge Rong et al. (2019) and DropMessage Fang et al. (2023) dynamically modify the graph topology to reduce oversmoothing and overfitting. However, the edge selection is purely random: edges are independently dropped according to a Bernoulli distribution, which may result in highly unbalanced node connectivity when the dropping rate is large. Several GNN methods propose dedicated remeshing techniques tailored to the task at hand. In MeshGraphNet (Pfaf et al., 2020), the authors employ adaptive remeshing strategies based on local remeshing techniques (Narain et al., 2012) for cloth simulation. This approach requires a groundtruth sizing field to guide the remeshing process. To simulate collisions between objects with GNNs, Pfaf et al. (2020); Yu et al. (2024) create contact edges between nodes from distinct objects when their distance is below a prescribed threshold. To adapt edge selection to any task, graph sparsification learning approaches Rathee et al. (2021); Luo et al. (2021); Saha et al. (2023); Qian et al. (2023) benefit from recent advances in diferentiable sampling of discrete random variables Jang et al. (2016); Maddison et al. (2016). These methods learn edge selection probabilities and use them to determine which edges should be retained or removed. However, when applied to dense graphs, they require considering all possible node pairs, resulting in a computational complexity of $\mathcal { O } ( N ^ { 2 } )$ , where N denotes the number of nodes. To address the specific challenges posed by dense graphs in BEMs, we introduce a computationally eficient edge sampling strategy based on the edge length and predictions of the expected error, as it highlights nodes involved in strong interactions.

## 4 ScaGNN Method

In this section, we present our ScaGNN method, illustrated in Figure 2. In the context of multiple scattering, it is essential to represent both the geometry of obstacles and their interaction accurately. For these reasons, we adopt a GNN architecture with edge features following MeshGraphNet (Pfaf et al., 2020), since it allows expressive modeling of interactions between two nodes. The dense interaction graph induced by the BEM is computationally prohibitive to process directly. To reduce the number of interactions to model, we leverage a hierarchical GNN inspired by MuS-GNN (Lino et al., 2022), which aggregates nodes locally, and we introduce a dynamic adaptive edge sampling mechanism to select the most relevant long-range interactions. In the following, we first define the diferent graph structures used to represent obstacles, the mappings between successive resolution levels, and the interactions between distant nodes. We then present the diferent components of the GNN architecture, including the encoders, processor, and decoders. Finally, we present the dynamic adaptive edge sampling strategy together with the loss functions used to train the entire framework.

Graphs definition Let $\mathcal { G } ^ { 0 } = ( V ^ { 0 } , E ^ { 0 } )$ denote the Boundary Graph with nodes $V ^ { 0 }$ and bidirected edges $E ^ { 0 }$ . It is composed of M disconnected components $\mathcal { G } _ { \Gamma _ { m } } ^ { 0 } = ( V _ { \Gamma _ { m } } ^ { 0 } , E _ { \Gamma _ { m } } ^ { 0 } )$ , each corresponding to the mesh of the boundary $\Gamma _ { m }$ of the obstacle $\Omega _ { m } , 1 \leq m \leq M$ . We construct a multiscale hierarchy of L node levels by building an octree for each obstacle. At each level $0 < \ell < L$ , we obtain a node set $V ^ { \ell }$ with $N _ { \ell }$ nodes whose positions form a subset of those at the previous level, i.e., $p ( V ^ { \ell } ) \subset p ( V ^ { \ell - 1 } )$ , where $p ( V )$ denotes the set of node positions associated with V . We also define directed downsampling edges $E ^ { \ell - 1 \to \ell }$ and directed upsampling edges $E ^ { \ell \to \ell - 1 }$ , which connect nodes between consecutive levels $\ell - 1$ and ℓ. The downsampling edges define the Downsampling Graph $\mathcal { G } ^ { \ell - 1 \to \ell } = ( V ^ { \ell - 1 } , E ^ { \ell - 1 \to \ell } )$ . Similarly, the upsampling edges define the Upsampling Graph $\mathcal { G } ^ { \ell  \ell - 1 } = ( V ^ { \ell - \bar { 1 } } , E ^ { \ell  \ell - 1 } )$ . To eficiently model long-range interactions between nodes, we create K Distant Interaction Graph $\mathcal { G } _ { k } ^ { L - 1 } = ( V ^ { L - 1 } , E _ { k } ^ { L - 1 } ) , 1 \le k \le K$ , where $E _ { k } ^ { L - 1 }$ denotes a set of directed edges. In contrast to the predefined graph edges, the edges in $E _ { k } ^ { L - 1 }$ are selected dynamically during the forward pass using the adaptive edge sampling strategy. All graphs are associated with node features and edge features. Examples of these graph structures for two spherical obstacles are shown in Section A.

![](images/3a48717e1b1b3abf2bc9f464d109865f7bbb4f5d38a5bace295153d59820c15b.jpg)  
Figure 2: Illustration of ScaGNN architecture with $L = 3$ hierarchical levels. The node and edge encoders have not been represented for better readability. The graph representation processed by each MP layer is given below the corresponding block. The architecture first consists of a Boundary Block with $N _ { \mathrm { b } }$ MP layers applied to the Boundary Graph $\mathcal { G } ^ { 0 }$ , then two Downsampling Blocks. At the lowest resolution, K Distant Interaction Blocks are applied. They are composed of an intermediate decoder whose outputs are used by our dynamic adaptive edge sampling to select the edges of the Distant Interaction Graph $\mathcal { G } _ { 1 } ^ { 2 } , . . . , \mathcal { G } _ { K } ^ { 2 }$ and $N _ { \mathrm { d } }$ MP layers. Finally, two Upsampling Blocks and a Boundary Block of $N _ { \mathrm { b } }$ MP layers are applied to the Boundary Graph $\mathcal { G } ^ { 0 }$ before the final decoder. Gradient flows through solid arrows, not dashed arrows.

Encoders First, a node encoder initializes the features for all the nodes in $V ^ { 0 }$ . It consists of a two-layer MLP that takes the boundary conditions as input(more details in Section D) and has an output dimension of $d _ { 0 }$ . Then, each edge set $E ^ { 0 } , E ^ { \ell - 1 \to \ell }$ and $E ^ { \ell \to \ell - 1 }$ $0 < \ell < L$ , is associated with a dedicated edge encoder that initializes its edge features. The edge features in each set $E _ { k } ^ { L - 1 } , 1 \le k \le K$ , are initialized successively using a shared edge encoder during the forward pass, after the edge sets have been generated by our dynamic adaptive edge sampling strategy. The edge encoders are two-layer MLPs with output dimension of $d _ { 0 }$ for edges in $E ^ { 0 } , d _ { \ell }$ for edges in $E ^ { \ell - 1 \to \bar { \ell } }$ and $E ^ { \ell - 1 \to \ell }$ , and $d _ { L - 1 }$ for edges in $E _ { k } ^ { L - 1 }$ . They take as inputs a sinusoidal encoding of the edge lengths (Vaswani et al., 2017), the normalized direction of edges and additional features related to the boundary conditions (more details in Section D).

Processor The processor is composed of multiple message-passing (MP) layers that operate on the graphs previously defined (see Section C for more details on MP layers). The first step is a Boundary Block which consists of $N _ { \mathrm { b } }$ MP layers that propagate information through the Boundary Graph $\mathcal { G } ^ { 0 }$ . It is followed by $L - 1$ Downsampling Blocks, each associated with a Downsampling Graph $\mathcal { G } ^ { \ell - \bar { 1 } \to \ell } , \ 0 < \ell < L$ . While MuS-GNN (Lino et al., 2022) maintains a fixed latent dimension across scales, our architecture aligns with the U-Net practice (Ronneberger et al., 2015) of widening the feature space at coarser levels, allowing richer representations at lower resolutions. To this end, the node features of $\mathcal { G } ^ { \ell - 1 \to \ell }$ are first initialized by the Node Feature Expander, a single-layer MLP projecting the node features output by the previous Downsampling Blocks (or by the Boundary Block when $\ell - 1 = 0 )$ from a dimension $d _ { \ell - 1 }$ to a dimension $d _ { \ell } .$ . Then, a MP layer is applied. At the lowest resolution, K Distant Interaction Blocks are successively executed on the corresponding Distant Interaction Graphs $\mathcal { G } _ { k } ^ { L - 1 } , 1 \le k \le K$ . Each Distant Interaction Block is composed of an intermediate decoder, a dynamic adaptive edge sampling mechanism to create the edge set $E _ { k } ^ { L - 1 }$ , and $N _ { \mathrm { d } }$ MP layers. The node features of the k-th Distant Interaction Block are initialized using the output features of the preceding Distant Interaction Blocks (or the final Downsampling Block when $k = 0 )$ . Thereafter, $L - 1$ Upsampling Blocks are applied, each operating on an Upsampling Graph $\mathcal { G } ^ { \ell \to \ell - 1 }$ , and composed of a MP layer that reduces the feature dimension from $d _ { \ell }$ to $d _ { \ell - 1 }$ . The node features of $\mathcal { G } ^ { \ell \to \ell - 1 }$ are initialized with the node features output by the preceding Upsampling Block (or by the last Distant Interaction Block if $\ell = L - 1 )$ for the nodes that are also processed by this block, i.e., for the nodes in $V ^ { \ell }$ . Otherwise, the remaining node features of $\mathcal { G } ^ { \ell \to \ell - 1 }$ , those associated with the nodes $V ^ { \ell - 1 } \setminus V ^ { \ell } ,$ , are initialized with node features of the Downsampling Graph $\mathcal { G } ^ { \ell - 1 \to \ell }$ . Finally, a Boundary Block of $N _ { \mathrm { b } }$ MP layers is applied to the Boundary Graph $\mathcal { G } ^ { 0 }$ . The node features are initialized from the node features produced by the last Upsampling Block.

Decoders Our GNN architecture includes several decoders: K intermediate decoders, one in each Distant Interaction Block, and the final decoder after the last Boundary Block. The intermediate decoders are two-layer MLPs operating on the node features computed by the last MP layer before each Distant Interaction Block. The k-th intermediate decoder outputs two vectors $\hat { \mathbf { y } } _ { L - 1 , k } \in \mathbb { R } ^ { N _ { L - 1 } \times d _ { \mathrm { f } } }$ and $\hat { \mathbf { e } } _ { k } \in \mathbb { R } ^ { N _ { L - 1 } \times d _ { \mathrm { f } } }$ corresponding to the approximate solution of the BIE (equation 2) and the expected error associated with this prediction for each node in $V ^ { L - 1 }$ . The predicted errors are then used to sample the edges of the diferent graphs $\mathcal { G } _ { k } ^ { L - 1 }$ $1 \leq k \leq K$ , while the predictions of the BIE solution are leveraged to generate the supervision signals for the error predictions (see the following sections). The final decoder is a linear layer that takes the node features of the last message-passing layer as input and outputs the vector $\hat { \mathbf { y } } _ { 0 } \in \mathbb { R } ^ { d _ { \mathrm { f } } \times N _ { 0 } }$ corresponding to the approximate solution of the BIE (equation 2) on the boundary discretization of Γ. For Laplace problems, $d _ { \mathrm { f } } = 1$ , and for Helmholtz problems, $d _ { \mathrm { f } } = 2$ for the real and imaginary parts of the BIE solution.

Losses For each intermediate decoder $1 \leq k \leq K$ , the predictions $\hat { \mathbf { y } } _ { L - 1 , k }$ are supervised with the groundtruth BIE solution $\mathbf { y } _ { L - 1 } ^ { * } \in \mathbb { R } ^ { d _ { \mathrm { f } } \times N _ { L - 1 } }$ restricted to the low-resolution nodes $\dot { V } ^ { L - 1 }$ . The predictions $\hat { \mathbf { e } } _ { k }$ are trained to match the groundtruth error $\mathbf { e } _ { k } ^ { \ast } \in \mathbb { R } ^ { d _ { \mathrm { f } } \times N _ { L - 1 } }$ at the intermediate decoder k. It is given by

$$
\mathbf { e } _ { k } ^ { * } = | \hat { \mathbf { y } } _ { L - 1 , k } - \mathbf { y } _ { L - 1 } ^ { * } |\tag{4}
$$

where $| . |$ is the element-wise absolute value. The groundtruth error $\mathbf { e } _ { k } ^ { \ast }$ is treated as a fixed target; therefore, gradients are not backpropagated through it. The predictions $\hat { \mathbf { y } } _ { 0 }$ of the final decoder are supervised by the groundtruth BIE solution on the discretization of the boundary Γ, $\mathbf { y } _ { 0 } ^ { * } \in \mathbb { R } ^ { d _ { \mathrm { f } } \times N _ { 0 } }$ . All predictions (errors and BIE solutions) are compared with their respective groundtruth using the Huber loss $\mathcal { L } _ { \mathrm { h u b e r } }$ (Huber, 1992), and the resulting losses are summed. In the total loss $\mathcal { L } _ { \mathrm { t o t a l } }$ , the contributions of the intermediate decoders are weighted with hyperparameter $0 < \gamma < 1$ so that decoders appearing earlier in the GNN architecture have a smaller impact on the overall loss:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { h u b e r } } ( \hat { \mathbf { y } } _ { 0 } , \mathbf { y } _ { 0 } ^ { * } ) + \sum _ { k = 0 } ^ { K - 1 } \gamma ^ { K - k } \Big ( \mathcal { L } _ { \mathrm { h u b e r } } ( \hat { \mathbf { y } } _ { L - 1 , k } , \mathbf { y } _ { L - 1 } ^ { * } ) + \mathcal { L } _ { \mathrm { h u b e r } } ( \hat { \mathbf { e } } _ { k } , \mathbf { e } _ { k } ^ { * } ) \Big ) .\tag{5}
$$

Dynamic Adaptive Edge Sampling Since each node interacts with every other node, the number of interactions grows quadratically with the number of nodes. Consequently, processing such dense obstacle graphs with a GNN becomes computationally prohibitive, even after reducing the mesh resolution. To make our method tractable, we propose a dynamic adaptive edge sampling. It selects a subset of relevant interactions to create the edge sets $\mathbf { \check { \it E } } _ { k } ^ { L - 1 } , \mathbf { \check { 1 } } \leq k \leq K$ , for each Distant Interaction Graph. By restricting each node in a graph $\mathcal { G } _ { k } ^ { L - 1 }$ to a fixed set of $N _ { \mathrm { e } }$ incoming edges, i.e., each node serves as a destination node only $N _ { \mathrm { e } }$ times, we achieve linear complexity with respect to the number of nodes. The choice of the source nodes is based on two criteria: the distance to the destination node $( { \mathrm { i . e . } }$ , the edge length) and the expected error predicted by the intermediate decoder of the current Distant Interaction Block. Let $n _ { \mathrm { d } }$ be a node in $V ^ { L - 1 }$ with position $p _ { n _ { \mathrm { d } } }$ . To sample each of the $N _ { \mathrm { e } }$ incoming edges of $n _ { \mathrm { d } }$ in $E _ { k } ^ { L - 1 }$ , C candidate source nodes $\{ n _ { c } \} _ { c = 1 } ^ { C }$ are uniformly drawn from $V ^ { L - 1 }$ . The position of $n _ { c }$ is $p _ { n _ { c } }$ and its expected error is $\hat { e } _ { k , n _ { c } } \in \mathbb { R } _ { + } ^ { d _ { \mathrm { f } } }$ (corresponding to the $n _ { c }$ -th index of error prediction $\hat { \mathbf { e } } _ { k } )$ . The candidate node minimizing the score function $f _ { \mathrm { s c o r e } } ^ { \alpha }$ based on these two criteria is selected as the source node $n _ { \mathrm { s } }$ for the corresponding edge:

$$
n _ { \mathrm { s } } = \underset { \left\{ n _ { c } \right\} _ { c = 1 } ^ { C } } { \mathrm { a r g m i n } } f _ { \mathrm { s c o r e } } ^ { \alpha } ( p _ { n _ { \mathrm { d } } } , p _ { n _ { c } } , \hat { e } _ { k , n _ { c } } )\tag{6}
$$

where the score function $f _ { \mathrm { s c o r e } } ^ { \alpha }$ is defined as:

$$
f _ { \mathrm { s c o r e } } ^ { \alpha } ( p _ { n _ { \mathrm { d } } } , p _ { n _ { c } } , \hat { e } _ { k , n _ { c } } ) = \frac { \| p _ { n _ { \mathrm { d } } } - p _ { n _ { c } } \| _ { 2 } } { ( \sum _ { d = 1 } ^ { d _ { \mathrm { f } } } \hat { e } _ { k , n _ { c } , d } ) ^ { \alpha } }\tag{7}
$$

with $\hat { e } _ { k , n _ { c } , d } .$ the d-th dimension of $\hat { e } _ { k , n _ { c } } , 1 \le d \le d _ { \mathrm { f } }$ , and $\alpha \in \mathbb { R }$ is a hyperparameter controlling the relative sensitivity of the score function to the error and the distance between the two nodes. The predicted errors indicate nodes involved in the most complex or dificult to model interactions (see Section H). Consequently, the selection of these nodes as source nodes ensures that the message-passing scheme captures the most significant interactions. However, this criterion remains the same for every destination node $n _ { \mathrm { d } }$ . By adding the edge length, we tailor the criterion to each destination node. The use of the edge length comes from the expression of Green’s functions in the BIE (equation 2). For Laplace and Helmholtz problems, the Green’s function are given by $\begin{array} { r } { G ( \mathbf { x } ) = - \frac { 1 } { 4 \pi r } } \end{array}$ and $\begin{array} { r } { G ( \mathbf { x } ) = - \frac { e ^ { - \mathrm { i } k r } } { 4 \pi r } } \end{array}$ , respectively, where $r = \| \mathbf { x } \| _ { 2 } , \mathbf { x } \in \mathbb { R } ^ { 3 } \setminus \Omega \cup \Gamma$ The physical meaning is that interactions between obstacles are stronger at short distances, but long-range interactions remain important. Importantly, selecting a source node from among C uniformly sampled candidate nodes to generate the edges in the graphs $\mathcal { G } _ { k } ^ { L - 1 } , 1 \le k \le K$ , avoids computing the distance between all pairs of nodes, which significantly reduces the computational cost of our dynamic adaptive edge sampling.

## 5 Experiments

The performance of ScaGNN is evaluated through extensive experiments on a new benchmark specially designed for this purpose. The results of this new method are compared against those of other state-of-the-art learning-based approaches for solving PDEs on unstructured data. This includes GNN-based methods like MeshGraphNet (Pfaf et al., 2020) (hereafter MGN) and MuS-GNN (Lino et al., 2022), transformer-based methods such as Transolver (Wu et al., 2024a) and Transolver++ (Luo et al., 2025), and a hybrid method that leverages both MP layers and transformers: Erwin (Zhdanov et al., 2025). We also include the point cloud processing method Point Transformer v3 (Wu et al., 2024b) (hereafter PTv3).

## 5.1 New ScaGNN Benchmark

A new benchmark to evaluate and compare learning-based approaches for multiple scattering problems in unbounded homogeneous domains is proposed. We consider three exterior problems with diferent levels of complexity: a Laplace Dirichlet problem with standard boundary conditions, a Helmholtz Dirichlet problem with an incident wave emitted from a monopole source and a Helmholtz Neumann problem with an incident plane wave. For each problem, the training set consists of samples containing three ellipsoidal obstacles. Four test sets are additionally considered: three datasets containing three, six, and nine ellipsoidal obstacles per sample, respectively, and one dataset containing three rounded parallelepiped obstacles per sample. The obstacle locations and shape parameters together with the boundary condition parameters (e.g., the source location or the wavenumber when applicable) vary from one data sample to another. For each sample, the groundtruth solution on the obstacle boundary is computed using the BEM. We also introduce metrics dedicated to each problem that measure prediction errors. Further details about our benchmark are provided in Section B.

## 5.2 Implementation Details

Regarding our architecture, we set the number of levels to $L = 3$ and the expansion rate in the Node Feature Expander to 2. This means that, for $0 < \ell < L ,$ , the latent dimension $d _ { \ell }$ at level ℓ is given by $d _ { \ell } = d _ { 0 } \times 2 ^ { \ell }$ The latent dimension at the first level is set to $d _ { 0 } = 6 4$ . The number K of low-resolution graphs and Distant Interaction Block is set to 3 unless otherwise mentioned. In each graph $\mathcal { G } _ { k } ^ { L - 1 } , 1 \le k \le \dot { K }$ , each node is connected to $N _ { \mathrm { e } } = 2 0$ nodes to create the edge sets $E _ { k } ^ { L - 1 }$ , unless otherwise mentioned. For modeling distant interactions, the number C of candidate edges sampled per required edge is set to 2 and the exponent α in the score function $f _ { \mathrm { s c o r e } } ^ { \alpha }$ is set to 1.0, unless otherwise specified. In the loss function $\mathcal { L } _ { \mathrm { t o t a l } }$ , we set γ to 0.4 and the parameter δ of all Huber losses is set to 1.0. The architecture details of the methods used for comparison are given in Section E. Their hyperparameters have been tuned both (i) to maximize performance and (ii) to ensure the computational cost and number of parameters is similar or higher than those of our method. The aim is to show that the observed performance gain can only be attributed to its contributions. The models are supervised with the groundtruth BIE solution using a Huber loss with parameter $\delta = 1 . 0$ . Additional implementation details, including the hardware used and the training procedure, are provided in Section E.

## 5.3 ScaGNN Performance

Table 1: Benchmark results on in-distribution test datasets comparing the prediction errors, number of parameters and computational cost (in FLOPs).
<table><tr><td rowspan="2">Architecture Type GNN Transformer</td><td rowspan="2">Method</td><td rowspan="2">Number of Parameters</td><td rowspan="2"> $\mathrm { F L O P s }$ </td><td rowspan="2">Laplace  $\mathrm { E r r } _ { \mathrm { r e l } }$ </td><td colspan="2">Helmholtz Dirichlet</td><td rowspan="2">Helmholtz Neumann  $\mathrm { E r r _ { a m p l } }$   $\mathrm { E r r _ { a n g l e } }$ </td></tr><tr><td> $\mathrm { E r r _ { a m p l } }$ </td><td> $\mathrm { E r r _ { a n g l e } }$ </td></tr><tr><td>√</td><td>PTv3</td><td>19.2 M</td><td>16.8 G</td><td>0.081</td><td>0.148</td><td>0.141</td><td>0.070 0.073</td></tr><tr><td>√</td><td>Transolver</td><td>4.6M</td><td>38.4 G</td><td>0.183</td><td>0.596</td><td>0.454 0.070</td><td>0.073</td></tr><tr><td>√</td><td>Transolver++</td><td>4.2 M</td><td>35.6 G</td><td>0.195</td><td>0.635 0.466</td><td>0.103</td><td>0.122</td></tr><tr><td>√ √</td><td>Erwin</td><td>7.9 M</td><td>42.4 G</td><td>0.119</td><td>0.218</td><td>0.175 0.074</td><td>0.076</td></tr><tr><td>√</td><td>MGN</td><td>5.3M</td><td>160.4 G</td><td>0.173</td><td>0.286</td><td>0.181 0.068</td><td>0.068</td></tr><tr><td>√</td><td>MuS-GNN</td><td>5.6M</td><td>91.5 G</td><td>0.281</td><td>0.282</td><td>0.183</td><td>0.068 0.068</td></tr><tr><td>√</td><td>Ours  $\overline { { ( N _ { \mathrm { e } } = 4 ) } }$ </td><td>5.6M</td><td>17.1 G</td><td>0.070</td><td>0.122</td><td>0.117</td><td>0.055 0.054</td></tr><tr><td>√</td><td>Ours  $( N _ { \mathrm { e } } = 2 0 )$ </td><td>5.6M</td><td>29.6 G</td><td>0.045</td><td>0.087</td><td>0.086</td><td>0.044 0.043</td></tr></table>

In-distribution results In Table 1, we report performance in terms of prediction errors on the test datasets that follow the same data distribution as the training set, i.e., with three ellipsoidal obstacles per sample. To demonstrate that the comparison is fair, we also report the number of parameters and the computational cost in FLOPs (see Section G for more details on the FLOPs measurement). As our approach involves random processes, we report the mean of five evaluations, each with a diferent seed. We measure an average relative standard deviation of 0.18% across all datasets and metrics, demonstrating the stability of our method even with a small number $N _ { \mathrm { e } }$ of edges in the Distant Interaction Graph (see Section F for the full details of relative standard deviation per dataset).

The results show the superiority of ScaGNN: regardless of the problem or the metric, our new method with $N _ { \mathrm { e } } = 4$ and $N _ { \mathrm { e } } = 2 0$ (i.e., small or large number of modeled interactions) outperform other learning-based approaches, whether their architecture is based on transformers, GNNs, or both, even if their number of parameters or computational cost far exceed ours. Thus, our approach with $N _ { \mathrm { e } } = 4$ and $N _ { \mathrm { e } } = 2 0$ improves the performance of the best baseline, PTv3, by 19% and 38%, respectively. The capacities of fully GNN architectures such as MGN and MuS-GNN are limited by design: since they cannot create edges between disconnected parts of the input graph, they are unable to model interactions between obstacles. However, MGN and MuS-GNN architectures can sometimes achieve better performance than transformer-based methods. While Transformers are, in principle, able to capture any interactions between two nodes, the attention mechanism does not allow rich modeling of the interactions between two nodes due to the limited number of attention heads (between 4 and 16). In contrast, our approach leverages MP layers to model interactions. Each one is simulated using a MLP with 64 to 256 latent dimensions. Furthermore, most transformer-based approaches are tailored for PDE benchmarks such as (Pfaf et al., 2020; Janny et al., 2023) which typically operate on nearly regular meshes spanning the entire computational domain. This is not the case for the PTv3 architecture, which is specifically designed for processing point clouds. This may explain why PTv3 achieves the best performance among transformer-based approaches on our benchmark. The irregular distribution of points in our data, consisting of points sampled on the surfaces of distant objects, closely resembles that of point cloud datasets. However, PTv3’s good performance may also be attributed to its larger number of parameters. In Section I, we analyze the correlations between the ScaGNN prediction errors and several characteristics of the data samples, including the GMRES iteration count (a proxy for problem dificulty), the wavenumber and the obstacle dispersion. Finally, qualitative results are given in Section K.

Number of obstacles  
![](images/40e133795bf44b5a3202df08756b956caed34537ad585250c1bab110c3a3d0d0.jpg)

![](images/312845665142b044e2bce1d35013a229def8d532ccf1bdc45d9d7db75cf3e279.jpg)

![](images/515e5c7c0d09dc057bdee7efe33535139639c110156e7b813eaa9f9bdac3bd11.jpg)

![](images/e37d7da45dd735048f2afd02b092be42ddc659ff6efb5effccdeda27a364783b.jpg)

![](images/331a7552813ed32a79ae50b43148ddf77fca0ae08d3c491bbe28e62d734daf84.jpg)  
Figure 3: Estimation errors as a function of the number of obstacles for the Laplace (top), the Helmholtz Dirichlet (middle) and the Helmholtz Neumann (bottom) problems, respectively.

Generalization capabilities with more obstacles In Figure 3, we examine the generalization ability of models trained on datasets containing three obstacles per sample when evaluated on environments containing six and nine obstacles, for the diferent problems of the ScaGNN benchmark. Since the number of distant edges per node is fixed to achieve linear scaling with respect to the size of the input mesh, increasing the number of obstacles at test time results in sparser Distant Interaction Graphs. To continue identifying $N _ { \mathrm { e } }$ relevant interactions per node from a larger set of potential edges, our dynamic adaptive edge sampling must become more selective. The solution is to increase the number C of candidate edges so as to compare more candidate edges when selecting each edge of a Distant Interaction Graph $\mathcal { G } _ { k } ^ { L - 1 } , 1 \le k \le K$ . A thorough study of the impact of C on performance for diferent numbers of obstacles (see Section J.6) suggests a heuristic for adjusting the value of C. When the number of obstacles is multiplied by $x , C$ becomes $C + x - 1$

The results in Figure 3 show that performance decreases for all methods with the rise of the number of obstacles. This trend is expected due to the quadratic growth in the number of pairwise interactions as the number of obstacles increases, which substantially increases the problem complexity. In contrast, all models are trained on simpler problems involving three obstacles and are designed to scale linearly with the input size. Nevertheless, both ScaGNN with $N _ { \mathrm { e } } = 4$ and $N _ { \mathrm { e } } = 2 0$ remain superior to the other baselines. Furthermore, interpolations of the results with linear regressions show that performance reduction remains limited for modest increases in obstacle count. For every new obstacle, ScaGNN prediction errors increase on average by 0.027 and 0.024 when $N _ { \mathrm { e } } = 4$ and $N _ { \mathrm { e } } = 2 0$ , respectively, with a coeficient of determination $R ^ { 2 } > 0 . 9 9 9$ for both settings. The analysis of ScaGNN stability in Section F highlights that even when the number of obstacles increases, the relative standard deviations of the metrics remain very low.

Table 2: Benchmark results on out-of-distribution obstacle shapes test sets.
<table><tr><td rowspan=1 colspan=2>Method</td><td rowspan=1 colspan=1>Laplace $\mathrm { E r r } _ { \mathrm { r e l } }$ </td><td rowspan=1 colspan=1>Helmholtz Dirichlet $\mathrm { E r r _ { a m p l } }$      $\mathrm { E r r } _ { \mathrm { a n g l e } }$ </td><td rowspan=1 colspan=1>Helmholtz Neumann $\mathrm { E r r _ { a m p l } }$      $\mathrm { E r r } _ { \mathrm { a n g l e } }$ </td></tr><tr><td rowspan=1 colspan=2>PTv3</td><td rowspan=1 colspan=1>0.166</td><td rowspan=1 colspan=1>0.203      0.165</td><td rowspan=1 colspan=1>0.071       0.098</td></tr><tr><td rowspan=5 colspan=2>TransolverTransolver++ErwinMGNMuS-GNN</td><td rowspan=1 colspan=1>0.195</td><td rowspan=1 colspan=1>0.437      0.381</td><td rowspan=2 colspan=1>0.076      0.0910.087       0.120</td></tr><tr><td rowspan=1 colspan=1>olver++</td><td rowspan=1 colspan=1>0.206</td><td rowspan=1 colspan=1>0.458      0.390</td></tr><tr><td rowspan=1 colspan=1>rwin</td><td rowspan=1 colspan=1>0.148</td><td rowspan=1 colspan=1>0.211      0.199</td><td rowspan=2 colspan=1>0.088      0.1160.079      0.081</td></tr><tr><td rowspan=1 colspan=1>0.211</td><td rowspan=1 colspan=1>0.214      0.158</td></tr><tr><td rowspan=1 colspan=1>0.334</td><td rowspan=1 colspan=1>0.214      0.160</td><td rowspan=1 colspan=1>0.063      0.073</td></tr><tr><td rowspan=1 colspan=2>Ours $\overline { { ( N _ { \mathrm { e } } = 4 ) } }$ </td><td rowspan=1 colspan=1>0.150</td><td rowspan=1 colspan=1>0.147      0.120</td><td rowspan=2 colspan=1>0.058      0.0690.049      0.062</td></tr><tr><td rowspan=1 colspan=2>Ours $( N _ { \mathrm { e } } = 2 0 )$ </td><td rowspan=1 colspan=1>0.149</td><td rowspan=1 colspan=1>0.144     0.103</td></tr></table>

Generalization capabilities with diferent obstacle shapes In Table 2, we measure performance on out-of-distribution obstacle shapes that are parallelepipeds with rounded edges. As expected, we observe a performance degradation compared with the same settings with ellipsoidal obstacles. Nevertheless, ScaGNN performs on par with Erwin and remains superior to the other state-of-the-art approaches.

![](images/706b56e97b74c9640e8e7b266c2bcb94aa9c1a8a5f1db2f1a87893a777d9314e.jpg)  
Figure 4: Runtime comparison with respect to the number of obstacles for learning-based methods and for the BEM, considering diferent convergence tolerance thresholds rtol for GMRES. The two curves obtained using our method, Ours $( N _ { e } = 4 )$ and Ours $( N _ { e } = 2 0 )$ , overlap.

Runtime analysis In Figure 4, we study the runtime required to generate the solution traces on the boundaries for the Helmholtz Dirichlet problem with respect to the number of obstacles. We compare all approaches on the same GPU, including learning-based methods and the BEM with diferent convergence tolerance thresholds rtol for GMRES. It is worth noting that due to the problem sizes examined in this work (the average number of nodes per data sample with nine obstacles is $N = 1 2 , 0 0 0 )$ , we do not leverage the FMM-accelerated BEM on GPU. This follows Gumerov et al. (2019)’s conclusion, which reports that for $N \lesssim 2 0 , 0 0 0$ , the standard BEM on GPU remains faster than the FMM-accelerated variant. With the exception of Transolver and Transolver++, which achieve the lowest runtime due to the small number of learnable slices and layers that we found to provide the best performance, the runtime of the other learning-based methods is similar. We observe that our method with $N _ { e } = 4$ and $N _ { e } = 2 0$ achieves the same runtime because the same architecture is processed and the extra computational cost when $N _ { e } = 2 0$ is parallelized by the GPU. More importantly, this study shows that the BEM is orders of magnitude slower than the learning-based approaches and the BEM’s runtime increases faster with the number of obstacles than that of learning-based methods. The BEM is also sensitive to the wavelength and to the relative position of obstacles, while the runtime of learning-based methods only depends on the number of nodes. Finally, this study shows that learning-based methods constitute a viable alternative to the BEM in terms of runtime even when the BEM’s convergence tolerance thresholds are very low, as in Figure 4.

Table 3: Ablation
<table><tr><td>Dimension at the finest level</td><td>Dimension expansion factor</td><td>K</td><td> $N _ { \mathrm { d } }$ </td><td>Intermediate Predictions</td><td>Edge Sampling</td><td> $\#$  params</td><td>FLOPs</td><td> $\mathrm { E r r _ { a m p l } }$ </td><td> $\mathrm { E r r } _ { \mathrm { a n g l e } }$ </td></tr><tr><td>184</td><td>1</td><td>1</td><td>6</td><td>x</td><td>Uniform</td><td>5.6M</td><td>79.2 G</td><td>0.109</td><td>0.105</td></tr><tr><td>64</td><td>2</td><td>1</td><td>6</td><td>x</td><td>Uniform</td><td>5.2 M</td><td>29.0 G</td><td>0.106</td><td>0.104</td></tr><tr><td>32</td><td>3</td><td>1</td><td>6</td><td>x</td><td>Uniform</td><td>5.8 M</td><td>26.3 G</td><td>0.128</td><td>0.128</td></tr><tr><td>64</td><td>2</td><td>3</td><td>2</td><td>x</td><td>Uniform</td><td>5.2 M</td><td>29.4 G</td><td>0.100</td><td>0.099</td></tr><tr><td>64</td><td>2</td><td>3</td><td>2</td><td>√</td><td>Uniform</td><td>5.6M</td><td>29.6 G</td><td>0.100</td><td>0.099</td></tr><tr><td>64</td><td>2</td><td>3</td><td>2</td><td>√</td><td>Adaptive</td><td>5.6M</td><td>29.6 G</td><td>0.087</td><td>0.086</td></tr></table>

## 5.4 Ablation

In Table 3, we study the impact of the diferent components of our method for the exterior Helmholtz Dirichlet problem. In the first three rows, we demonstrate that the computational cost can be significantly reduced by increasing the latent dimension at low resolution, while maintaining similar performance and number of parameters. Although it is a common deep learning practice (Ronneberger et al., 2015), it is not applied in GNNs for PDE simulation (Lino et al., 2022; Deng et al., 2024). This justifies the introduction of the Node Feature Expander module in our architecture. However, we observe that the latent dimension at the finest level and the expansion factor must be carefully tuned to maximize performance for a given budget of parameters and computational cost in FLOPs. In our case, we found that a latent dimension of 64 at the finest level and an expansion factor of 2 is optimal. The fourth row shows that $K > 1$ , i.e. using multiple distinct Distant Interaction Graphs $\mathcal { G } _ { k } ^ { L - 1 } , 1 \stackrel { \textstyle \cdot } { \leq } k \leq K$ , for modeling distant interactions within a single forward pass, improves performance. This can be explained by the greater diversity of interactions modeled by our GNN. Moreover, the impact of increasing K on computational cost is minimal; it is only due to the initialization of K times as many edges. It is worth noting that when K increases, the number of MP layers per Distant Interaction Block is adapted so that the total number of MP layers remains identical.

The last row highlights the gain achieved by selecting edges of the Distant Interaction Graph using our dynamic adaptive edge sampling, which relies on the intermediate predictions. The slight increase in the number of parameters and in the computational cost is explained by the intermediate decoders introduced to predict the expected errors. In the penultimate row, the same architecture trained with the intermediate predictions but with uniform sampling, instead of our dynamic adaptive edge sampling, does not improve performance. This demonstrates that the gain with our dynamic adaptive edge sampling is not due to the intermediate predictions; it can only be explained by the selection of more relevant distant edges. More thorough ablations of hyperparameters are provided in the Appendix, focusing on the impact of the number of edges $N _ { \mathrm { e } }$ per Distant Interaction Graph (Section J.1), α in the score function $f _ { \mathrm { s c o r e } } ^ { \alpha }$ (Section J.3), the number K of the Distant Interaction Graph (Section J.2), and the weight γ in the loss $\mathcal { L } _ { \mathrm { t o t a l } }$ (Section J.4).

## 6 Conclusion

In this article, we present ScaGNN, a new learning-based method for simulating multiple scattering problems. Such problems are well suited to BEMs, but can become computationally expensive. Our approach proposes a new GNN architecture to approximate the solution of the BIE. In particular, the key contribution is a dynamic adaptive edge sampling mechanism that identifies the most relevant distant interactions to simulate, and creates edges accordingly. The edge selection is based on two criteria: the edge length and the expected error predicted by intermediate decoders which highlights nodes involved in strong interactions. By design, our approach achieveq linear complexity with respect to the number of nodes in the input obstacle meshes. To evaluate the eficiency of this method, we introduce a novel benchmark of multiple scattering simulations including Laplace and Helmholtz problems. Our results demonstrate that ScaGNN surpasses other state of-the-art learning-based methods for solving PDEs and processing point clouds. Additionally, ScaGNN generalizes better to settings with a larger number of obstacles and to out-of-distribution obstacle shapes than the other baselines we compare against. Our study focuses on problems with a small number of obstacles and with simple shapes due to the high cost to generate the training data. Extending the approach to more challenging problems with a larger number of obstacles and more complex shapes is needed to validate its applicability to practical engineering problems. To reduce the need for groundtruth, exploring self-supervised pretraining or physics-informed approaches could also be a relevant perspective.

## Acknowledgments

This work is supported by Agence de l’Innovation de Défense (AID) via Centre Interdisciplinaire d’Études pour la Défense et la Sécurité (CIEDS), through the APRO project.

## References

Benedikt Alkin, Andreas Fürst, Simon Schmid, Lukas Gruber, Markus Holzleitner, and Johannes Brandstetter. Universal physics transformers: A framework for eficiently scaling neural operators. Advances in Neural Information Processing Systems, 37:25152–25194, 2024. 2

Benedikt Alkin, Maurits Bleeker, Richard Kurle, Tobias Kronlachner, Reinhard Sonnleitner, Matthias Dorfer, and Johannes Brandstetter. Ab-upt: Scaling neural cfd surrogates for high-fidelity automotive aerodynamics simulations via anchored-branched universal physics transformers. arXiv preprint arXiv:2502.09692, 2025. 2

Timo Betcke and Matthew W. Scroggs. Bempp-cl: A fast Python based just-in-time compiling boundary element library. Journal of Open Source Software, 6(59):2879, March 2021. doi: 10.21105/joss.02879. 20

Marc Bonnet. Boundary Integral Equation Methods for Solids and Fluids. John Wiley & Sons, 1999. 1, 3

Yadi Cao, Menglei Chai, Minchen Li, and Chenfanfu Jiang. Eficient learning of mesh-based physical simulation with bi-stride multi-scale graph neural network. In International conference on machine learning, pp. 3541–3558. PMLR, 2023. 4

Stéphanie Chaillat, Marc Bonnet, and Jean-Francois Semblat. A multi-level fast multipole BEM for 3-d elastodynamics in the frequency domain. Computer Methods in Applied Mechanics and Engineering, 197: 4233–4249, 2008. doi: 10.1016/j.cma.2008.04.018. 4

Stéphanie Chaillat, Luca Desiderio, and Patrick J. Ciarlet. Theory and implementation of h-matrix based iterative and direct solvers for helmholtz and elastodynamic oscillatory kernels. Journal of Computational Physics, 351:165–186, 2017. doi: 10.1016/j.jcp.2017.08.021. 4

Alex Colagrande, Paul Caillon, Eva Feillet, and Alexandre Allauzen. Linear attention with global context: A multipole attention mechanism for vision and physics. arXiv preprint arXiv:2507.02748, 2025. 4

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-eficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022. 22

Éric Darve. The fast multipole method: Numerical implementation. Journal of Computational Physics, 160(1):195–240, 2000. doi: 10.1006/jcph.2000.6451. URL https://www.sciencedirect.com/science/ article/pii/S0021999100964519. 4

Huayu Deng, Xiangming Zhu, Yunbo Wang, and Xiaokang Yang. Evomesh: Adaptive physical simulation with hierarchical graph evolutions. arXiv preprint arXiv:2410.03779, 2024. 4, 12

William Falcon and The PyTorch Lightning team. Pytorch lightning, 2019. URL https://lightning.ai/ docs/pytorch/stable/. 24

Taoran Fang, Zhiqing Xiao, Chunping Wang, Jiarong Xu, Xuan Yang, and Yang Yang. Dropmessage: Unifying random dropping for graph neural networks. In Proceedings of the AAAI conference on artificial intelligence, pp. 4267–4275, 2023. 5

Zhiwei Fang, Sifan Wang, and Paris Perdikaris. Learning only on boundaries: A physics-informed neural operator for solving parametric partial diferential equations in complex geometries. Neural computation, 36(3):475–498, 2024. 2, 4

Meire Fortunato, Tobias Pfaf, Peter Wirnsberger, Alexander Pritzel, and Peter Battaglia. Multiscale meshgraphnets. ICML-AI4Science, 2022. 4

Christophe Geuzaine, Jean-Francois Remacle, and P Dular. Gmsh: a three-dimensional finite element mesh generator. International Journal for Numerical Methods in Engineering, 79(11):1309–1331, 2009. 20

Nail A Gumerov, Yulia A Pityuk, Olga A Abramova, and Iskander S Akhatov. Gpu accelerated fast multipole boundary element method for simulation of 3d bubble dynamics in potential flow. arXiv preprint arXiv:1905.01341, 2019. 11

Benjamin Gutteridge, Xiaowen Dong, Michael M Bronstein, and Francesco Di Giovanni. Drew: Dynamically rewired message passing with delay. In International Conference on Machine Learning, pp. 12252–12267. PMLR, 2023. 5

Wenqu Hao, Yongpin P Chen, Pei-Yao Chen, Ming Jiang, Sheng Sun, and Jun Hu. Solving two-dimensional scattering from multiple dielectric cylinders by artificial neural network accelerated numerical green’s function. IEEE Antennas and Wireless Propagation Letters, 20(5):783–787, 2021. 2

Zhongkai Hao, Zhengyi Wang, Hang Su, Chengyang Ying, Yinpeng Dong, Songming Liu, Ze Cheng, Jian Song, and Jun Zhu. Gnot: A general neural operator transformer for operator learning. In Internationa Conference on Machine Learning, pp. 12556–12569. PMLR, 2023. 4

Peter J Huber. Robust estimation of a location parameter. In Breakthroughs in statistics: Methodology and distribution, pp. 492–518. Springer, 1992. 7

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016. 5

Steeven Janny, Aurélien Beneteau, Madiha Nadri, Julie Digne, Nicolas Thome, and Christian Wolf. Eagle: Large-scale learning of turbulent fluid dynamics with mesh transformers. ICLR, 2023. 4, 9

Yunzhu Li, Jiajun Wu, Russ Tedrake, Joshua B Tenenbaum, and Antonio Torralba. Learning particle dynamics for manipulating rigid bodies, deformable objects, and fluids. ICLR, 2019. 4

Zijie Li, Kazem Meidani, and Amir Barati Farimani. Transformer for partial diferential equations’ operator learning. Transactions on Machine Learning Research, 2022. 4

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Graph kernel network for partial diferential equations. ICLR Workshop DeepDifEq, 2020a. 4

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Andrew Stuart, Kaushik Bhattacharya, and Anima Anandkumar. Multipole graph neural operator for parametric partial diferential equations. Advances in Neural Information Processing Systems, 33:6755–6766, 2020b. 4

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial diferential equations. ICLR, 2021. 4

Zongyi Li, Daniel Zhengyu Huang, Burigede Liu, and Anima Anandkumar. Fourier neural operator with learned deformations for pdes on general geometries. Journal of Machine Learning Research, 24(388):1–26, 2023a. 4

Zongyi Li, Nikola Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, et al. Geometry-informed neural operator for large-scale 3d pdes. Advances in Neural Information Processing Systems, 36:35836–35854, 2023b. 4

Guochang Lin, Pipi Hu, Fukai Chen, Xiang Chen, Junqing Chen, Jun Wang, and Zuoqiang Shi. Binet: learning to solve partial diferential equations with boundary integral networks. CSIAM Transactions on Applied Mathematics, 2021. 2, 4

Guochang Lin, Fukai Chen, Pipi Hu, Xiang Chen, Junqing Chen, Jun Wang, and Zuoqiang Shi. Bi-greennet: learning green’s functions by boundary integral network. Communications in Mathematics and Statistics, 11(1):103–129, 2023. 4

Mario Lino, Stathi Fotiadis, Anil A Bharath, and Chris D Cantwell. Multi-scale rotation-equivariant graph neural networks for unsteady eulerian fluid dynamics. Physics of Fluids, 34(8), 2022. 5, 6, 8, 12, 24

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. International Conference on Learning Representations (ICLR), 2017. 22

Dongsheng Luo, Wei Cheng, Wenchao Yu, Bo Zong, Jingchao Ni, Haifeng Chen, and Xiang Zhang. Learning to drop: Robust graph neural network via topological denoising. In Proceedings of the 14th ACM international conference on web search and data mining, pp. 779–787, 2021. 5

Huakun Luo, Haixu Wu, Hang Zhou, Lanxiang Xing, Yichen Di, Jianmin Wang, and Mingsheng Long. Transolver++: An accurate neural solver for pdes on million-scale geometries. ICML, 2025. 4, 8, 24

Chris J Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables. arXiv preprint arXiv:1611.00712, 2016. 5

P. A. Martin. Multiple Scattering: Interaction of Time-Harmonic Waves with N Obstacles, volume 107 of Encyclopedia of Mathematics and its Applications. Cambridge University Press, 2006. 1, 3

Bin Meng, Yutong Lu, and Ying Jiang. Solving partial diferential equations in diferent domains by operator learning method based on boundary integral equations. arXiv preprint arXiv:2406.02298, 2024. 4

Siddharth Nair, Timothy F Walsh, Greg Pickrell, and Fabio Semperlotti. Multiple scattering simulation via physics-informed neural networks. Engineering with Computers, 41(1):31–50, 2025. 2

Rahul Narain, Armin Samii, and James F O’brien. Adaptive anisotropic remeshing for cloth simulation. ACM transactions on graphics (TOG), 31(6):1–10, 2012. 5

Tobias Pfaf, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter Battaglia. Learning mesh-based simulation with graph networks. In International conference on learning representations, 2020. 2, 4, 5, 8, 9, 24

Chendi Qian, Andrei Manolache, Kareem Ahmed, Zhe Zeng, Guy Van den Broeck, Mathias Niepert, and Christopher Morris. Probabilistically rewired message-passing neural networks. arXiv preprint arXiv:2310.02156, 2023. 5

Wenzhen Qu, Yan Gu, Shengdong Zhao, Fajie Wang, and Ji Lin. Boundary integrated neural networks and code for acoustic radiation and scattering. International Journal of Mechanical System Dynamics, 4(2): 131–141, 2024. 4

Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational physics, 2019. 2

Mandeep Rathee, Zijian Zhang, Thorben Funke, Megha Khosla, and Avishek Anand. Learnt sparsification for interpretable graph neural networks. arXiv preprint arXiv:2106.12920, 2021. 5

Winfried Ripken, Lisa Coifard, Felix Pieper, and Sebastian Dziadzio. Multiscale neural operators for solving time-independent pdes. NeurIPS workshop DLDE III, 2023. 4, 22

Yu Rong, Wenbing Huang, Tingyang Xu, and Junzhou Huang. Dropedge: Towards deep graph convolutional networks on node classification. arXiv preprint arXiv:1907.10903, 2019. 5

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pp. 234–241. Springer, 2015. 4, 6, 12

Youcef Saad and Martin H Schultz. Gmres: A generalized minimal residual algorithm for solving nonsymmetric linear systems. SIAM Journal on scientific and statistical computing, 7(3):856–869, 1986. 4, 20

Avishkar Saha, Oscar Mendez, Chris Russell, and Richard Bowden. Learning adaptive neighborhoods for graph neural networks. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 22541–22550, 2023. 5

Alvaro Sanchez-Gonzalez, Jonathan Godwin, Tobias Pfaf, Rex Ying, Jure Leskovec, and Peter Battaglia. Learning to simulate complex physics with graph networks. In International conference on machine learning, pp. 8459–8468. PMLR, 2020. 4

Louis Serrano, Thomas X Wang, Etienne Le Naour, Jean-Noël Vittaut, and Patrick Gallinari. Aroma: Preserving spatial structure for latent pde modeling with local neural fields. Advances in Neural Information Processing Systems, 37:13489–13521, 2024. 2

PyTorch Team. Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation, 2024. URL https://pytorch.org/. 24

Bertrand Thierry. A remark on the single scattering preconditioner applied to boundary integral equations. Journal of Mathematical Analysis and Applications, 413(1):212 – 228, 2014. ISSN 0022-247X. doi: http://dx.doi.org/10.1016/j.jmaa.2013.11.051. 2

Jake Topping, Francesco Di Giovanni, Benjamin Paul Chamberlain, Xiaowen Dong, and Michael M Bronstein. Understanding over-squashing and bottlenecks on graphs via curvature. arXiv preprint arXiv:2111.14522, 2021. 5

Benjamin Ummenhofer, Lukas Prantl, Nils Thuerey, and Vladlen Koltun. Lagrangian fluid simulation with continuous convolutions. In International conference on learning representations, 2019. 4

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 6

Sifan Wang, Jacob H Seidman, Shyam Sankaran, Hanwen Wang, George J Pappas, and Paris Perdikaris. Cvit: Continuous vision transformer for operator learning. ICLR, 2025a. 4

Taiyi A Wang, Ian McBrearty, and Paul Segall. Graph neural network based elastic deformation emulators for magmatic reservoirs of complex geometries. Volcanica, 8(1):95–109, 2025b. 4

Shizheng Wen, Arsh Kumbhat, Levi Lingsch, Sepehr Mousavi, Yizhou Zhao, Praveen Chandrashekar, and Siddhartha Mishra. Geometry aware operator transformer as an eficient and accurate neural surrogate for pdes on arbitrary domains. arXiv preprint arXiv:2505.18781, 2025. 4

Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, and Mingsheng Long. Transolver: A fast transformer solver for pdes on general geometries. ICML, 2024a. 8, 24

Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, and Hengshuang Zhao. Point transformer v3: Simpler faster stronger. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4840–4851, 2024b. 4, 8, 24, 30, 31, 32, 33, 34, 35

Zipeng Xiao, Zhongkai Hao, Bokai Lin, Zhijie Deng, and Hang Su. Improved operator learning by orthogonal attention. ICML, 2024. 4

Youn-Yeol Yu, Jeongwhan Choi, Woojin Cho, Kookjin Lee, Nayong Kim, Kiseok Chang, ChangSeung Woo, Ilho Kim, SeokWoo Lee, Joon Young Yang, et al. Learning flexible body collision dynamics with hierarchical contact mesh transformer. In International Conference on Learning Representations, volume 2024, pp. 23367–23394, 2024. 5

Maksim Zhdanov, Max Welling, and Jan-Willem van de Meent. Erwin: A tree-based hierarchical transformer for large-scale physical systems. ICML, 2025. 2, 4, 8, 24

A Illustrations of Graph Representations 19   
B Benchmark details 19   
B.1 Addressed Problems 19   
B.2 Datasets 20   
B.3 Metrics 21   
B.4 Hardware details 21   
C Message-Passing 22   
D Input details 22   
E Addition Implementation details 22   
F Stability Analysis 23   
G FLOPs measurement 24   
H Repartition of the Expected Error 25   
I Analysis of the model performance 26   
J Additional Ablations 26   
J.1 Impact of the Number of Edge $N _ { \mathrm { e } }$ per Distant Interaction Graph . 26   
J.2 Impact of the Number K of Distant Interaction Graph . 26   
J.3 Impact of the α Parameter in the Score Function f<sup>α</sup> score 28   
J.4 Impact of the weight γ in the loss $\mathcal { L } _ { \mathrm { t o t a l } }$ 28   
J.5 Impact of the Number C of Candidate Edges during Training . 29   
J.6 Scaling the Number C of Candidate Nodes with the Number of Obstacles at Test Time 29   
K Qualitative results 30   
K.1 Boundary solution 30   
K.2 Volumetric solution 35

## A Illustrations of Graph Representations

Figure 5 illustrates of the diferent types of graph representations used in ScaGNN.

![](images/dac583774c51dd80d0679421b4250efc212d5db26054b09aa59a7eaafb63eb32.jpg)  
Figure 5: Illustration of some directed graph representations used in ScaGNN in the case of two obstacles. Graph edges are in dark blue, while the light blue corresponds to the obstacle meshes. ${ \mathcal { G } } ^ { 0 } { \mathrm { : } }$ : the Boundary Graph (arrows have been omitted for readability), $\mathcal { G } ^ { 1  2 }$ : the Downsampling Graph from level 1 to $2 , \mathcal { G } ^ { 2 }$ : a Distant Interaction Graph and $\mathcal { G } ^ { 2  1 }$ : the Upsampling Graph from level 2 to 1. We omit the Downsampling and Upsampling Graphs $\mathcal { G } ^ { 0 \to 1 }$ and $\mathcal { G } ^ { 1  0 }$ between levels 0 and 1 which are similar to graphs $\mathcal { G } ^ { 1  2 }$ and $\bar { g } ^ { 2 \to \bar { 1 } }$ at a lower resolution.

## B Benchmark details

## B.1 Addressed Problems

This section presents our benchmark in detail. We address the following three multiple scattering problems defined by a PDE and boundary conditions:

1. Exterior Laplace Dirichlet problem with standard boundary conditions:

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { \Delta u ( \mathbf { x } ) = 0 , } & { \qquad \mathbf { x } \in \mathbb { R } ^ { 3 } \setminus \Omega \cup \Gamma } \\ { u ( \mathbf { x } ) = - \Phi _ { 0 } - \frac { \Phi _ { 1 } } { \| \mathbf { x } - \mathbf { x } _ { 0 } \| _ { 2 } } - 2 \Phi _ { 2 } \mathbf { v } \cdot \frac { \mathbf { x } - \mathbf { x } _ { 0 } } { \| \mathbf { x } - \mathbf { x } _ { 0 } \| _ { 2 } } , } & { \qquad \mathbf { x } \in \Gamma } \\ { u ( x ) \to 0 , } & { \qquad \mathrm { a s ~ } | x | \to \infty . } \end{array} \right. } \end{array}\tag{8}
$$

where $\Phi _ { 0 } , \Phi _ { 1 }$ and $\Phi _ { 2 }$ are three constants between −1 and 1 so that $| \Phi _ { 0 } | + | \Phi _ { 1 } | + | \Phi _ { 2 } | = 1 , \mathbf { x } _ { 0 } \in \mathbb { R } ^ { 3 } \backslash \Omega \cup \Gamma$ and v is a unit vector in $\mathbb { R } ^ { 3 }$

2. Exterior Helmholtz Dirichlet problem with an incident wave of unit amplitude emitted from a monopole source. The Dirichlet boundary condition is therefore parametrized by the source location

$\mathbf { x } _ { 0 } \in \mathbb { R } ^ { 3 } \setminus \Omega \cup \Gamma$ and the wavenumber k:

$$
\left\{ \begin{array} { l l } { ( \Delta + k ^ { 2 } ) u ( \mathbf x ) = 0 , } & { \mathbf x \in \mathbb R ^ { 3 } \setminus \Omega \cup \Gamma } \\ { u ( \mathbf x ) = - \frac { \mathrm e ^ { \mathrm i \mathbf k \| \mathbf x - \mathbf x _ { 0 } \| _ { 2 } } } { \| \mathbf x - \mathbf x _ { 0 } \| _ { 2 } } , } & { \mathbf x \in \Gamma } \\ { + \mathrm { ~ S o m m e r f e l d ~ r a d i a t i o n ~ c o n d i t i o n } . } \end{array} \right.\tag{9}
$$

3. Exterior Helmholtz Neumann problem with an incident plane wave of unit amplitude. The Neumann boundary condition is parametrized by the incident wave’s direction v and the wavenumber k:

$$
\left\{ \begin{array} { l l } { ( \Delta + k ^ { 2 } ) u ( \mathbf { x } ) = 0 , } & { \mathbf { x } \in \mathbb { R } ^ { 3 } \setminus \Omega \cup \Gamma } \\ { \frac { \partial u } { \partial \mathbf { n } } = - \mathrm { i } k \mathrm { e } ^ { \mathrm { i } k \mathbf { x } \cdot \mathbf { v } } , } & { \mathbf { x } \in \Gamma } \\ { + \mathrm { S o m m e r f e l d ~ r a d i a t i o n ~ c o n d i t i o n } . } \end{array} \right.\tag{10}
$$

where $\begin{array} { r } { \frac { \partial u } { \partial \mathbf { n } } = \nabla u ( \mathbf { x } ) } \end{array}$ · n is the normal derivative and n the normal vector to Γ at x.

## B.2 Datasets

We generated a training dataset and several test datasets for each problem. The training set samples contain three ellipsoidal obstacles, and we created test sets with three, six and nine ellipsoidal obstacles, as well as a test set with three rounded parallelepiped obstacles referred to as the OoD test set. General statistics of the training and test sets are given in Table 4. Each dataset sample consists of the meshes of randomly sized and positioned, non-overlapping obstacles, randomly selected boundary condition parameters depending on the problem, and the corresponding boundary solution trace as labels for each mesh node. The data samples’ characteristics are provided in Table 5. The meshes representing the obstacles and the trace solution were generated using the GMSH library (Geuzaine et al., 2009) and the BEMPP library (Betcke & Scroggs, 2021). The BIE and the representation formulations used for each problem we address are given in Table 6. The BIE solution is computed with GMRES (Saad & Schultz, 1986) with a convergence tolerance of $1 0 ^ { - 5 }$ and double precision. We also recorded the number of GMRES iterations required to converge for each sample to monitor their computational complexity. We provide an illustration of a dataset sample with three ellipsoids and one with three rounded parallelepipeds in Figure 6

Table 4: General dataset statistics that are common for all problems
<table><tr><td rowspan="2">Datasets</td><td rowspan="2">Training Set</td><td colspan="4">Test Sets</td></tr><tr><td>3 obstacles</td><td>6 obsatcles</td><td>9 obstacles</td><td>OoD</td></tr><tr><td rowspan="4">Number of samples Number of obstacles per samples Total number of nodes</td><td>10k</td><td>1k</td><td>1k</td><td>1k</td><td>1k</td></tr><tr><td>3</td><td>3</td><td>6</td><td>9</td><td>3</td></tr><tr><td>41M</td><td>4M</td><td>8M</td><td>12M</td><td>3M</td></tr><tr><td>Ellipsoid</td><td>Ellipsoid</td><td>Ellipsoid</td><td>Ellipsoid</td><td>Rounded parallelepiped</td></tr></table>

Table 5: Main characteristics of our datasets. Lengths are given without units.
<table><tr><td>Environment size</td><td>10 × 10 × 10</td></tr><tr><td>Edge length in obstacle meshes</td><td>0.1</td></tr><tr><td>Minimal distance between two obstacles</td><td>0.1</td></tr><tr><td>Ellipses semi-axes length (min − max)</td><td>0.3 - 1.5</td></tr><tr><td>Wavelength (min − max)</td><td>0.6- 6</td></tr><tr><td>Rounded parallelepiped length, width, height (min – max)</td><td>0.6-3</td></tr><tr><td>Rounded parallelepiped rounded radius length (min – max)</td><td>0.3 – max(parallelepiped length, width, height)</td></tr><tr><td>Laplace boundary condition constants  $\Phi _ { 0 } , \Phi _ { 1 }$  and Φ2 (min − max)</td><td> $\begin{array} { c } { { - 1 - 1 , | \Phi _ { 0 } | + | \Phi _ { 1 } | + | \Phi _ { 2 } | = 1 } } \\ { { 1 0 ^ { - 5 } } } \end{array}$ </td></tr></table>

Table 6: Formulations of the BIE and of the representations used to generate the dataset for each problem where S and N are the single-layer and the hypersingular boundary integral operators, respectively, while $s$ and D are the single-layer and the double-layer potential operators, respectively.
<table><tr><td rowspan=1 colspan=1>Problem</td><td rowspan=1 colspan=1>BIE</td><td rowspan=1 colspan=1>Representation</td></tr><tr><td rowspan=1 colspan=1>Laplace Dirichlet</td><td rowspan=1 colspan=1> $\mathrm { S } p = u$ </td><td rowspan=1 colspan=1> $\overline { { u = S p } }$ </td></tr><tr><td rowspan=1 colspan=1>Helmholtz Dirichlet</td><td rowspan=1 colspan=1> $\mathrm { S } p = u$ </td><td rowspan=1 colspan=1> $\overline { { u = S p } }$ </td></tr><tr><td rowspan=1 colspan=1>Helmholtz Neumann</td><td rowspan=1 colspan=1> $\begin{array} { r } { \underline { { \mathbf { N } p } } = \frac { \partial \boldsymbol { u } } { \partial \mathbf { n } } } \end{array}$ </td><td rowspan=1 colspan=1> $u = \mathcal { D } p$ </td></tr></table>

![](images/84e0e2db0d5daa04b09cabc39855c79175d7c0a943c109cc25ac369fff79583a.jpg)  
(a) Ellipsoids Obstacles.

![](images/964bc06b5791263f6203facc7f76dc008e3d8b8ed002d02509bfcb8a92e032a9.jpg)  
(b) Parallelepipeds Obstacles.  
Figure 6: Illustrations of dataset samples with obstacle meshes in blue and the source in red.

## B.3 Metrics

For evaluation, we assess the performance on the Laplace problem using the relative error of the trace, $\mathrm { E r r } _ { \mathrm { r e l } }$ For the Helmholtz problems, we introduce two metrics: the relative error of the trace amplitude, $\mathrm { E r r _ { a m p l } } ,$ and the absolute error of the trace phase, $\mathrm { E r r _ { a n g l e } }$ . The definitions of these metrics for a single sample are given by:

$$
{ \mathrm { E r r } } _ { \mathrm { r e l } } = { \frac { \sum _ { { \boldsymbol { x } } \in \Gamma } \left| { \hat { \boldsymbol { p } } } ( { \boldsymbol { x } } ) - { \boldsymbol { p } } ^ { * } ( { \boldsymbol { x } } ) \right| } { \sum _ { { \boldsymbol { x } } \in \Gamma } | { \boldsymbol { p } } ^ { * } ( { \boldsymbol { x } } ) | } }\tag{11}
$$

$$
\mathrm { E r r } _ { \mathrm { a m p l } } = \frac { 1 } { \# \Gamma } \sum _ { x \in \Gamma } \left| \frac { | \hat { p } ( x ) | - | p ^ { * } ( x ) | } { | p ^ { * } ( x ) | } \right|\tag{12}
$$

$$
\mathrm { E r r } _ { \mathrm { a n g l e } } = \frac { 1 } { \# \Gamma } \sum _ { x \in \Gamma } \mathrm { a t a n 2 } ( \sin ( \Delta p ) , \cos ( \Delta p ) ) , \quad \Delta p = \angle \hat { p } ( x ) - \angle { p ^ { * } ( x ) }\tag{13}
$$

where $p ^ { * }$ and $\hat { p }$ denote the ground-truth boundary trace obtained from the BEM and the neural network prediction, respectively, # indicates the cardinality of a set, and $\angle$ stands for the angle of a complex number. These metrics are then averaged over all samples in a dataset.

## B.4 Hardware details

The data was generated on a 36 Intel Core i9-10980XE (3.00GHz) CPUs. The time required to generate our training datasets with 10000 samples depends on the problem and the BIE formulation. Generating the Laplace Dirichlet, Helmholtz Dirichlet and Helmholtz Neumann training datasets took 12, 36 and 96 hours, respectively.

## C Message-Passing

MP is applied to a graph $\mathrm { G } { = } ( \mathrm { V } , \mathrm { E } )$ . It consists of a edge feature update equation 14 followed by a node feature update equation 15:

$$
f _ { e _ { k l } } ^ { \prime } = \phi ^ { e } ( f _ { e _ { k l } } , f _ { v _ { k } } , f _ { v _ { l } } )\tag{14}
$$

$$
f _ { v _ { k } } ^ { \prime } = \phi ^ { n } ( f _ { v _ { k } } , \sum _ { k } f _ { e _ { k l } } ^ { \prime } )\tag{15}
$$

where $\phi ^ { e }$ and $\phi ^ { n }$ are MLPs and $f _ { e _ { k l } }$ are the features of edge $e _ { k l } \in E$ connecting node $v _ { k } \in V$ to node $v _ { l } \in V$ of features $f _ { v _ { k } }$ and $f _ { v _ { l } }$ , respectively.

## D Input details

Tables 7 to 9 provide the details of the encoder inputs for the Laplace Dirichlet, the Helmholtz Dirichlet and the Helmholtz Neumann problems, respectively. It is worth noting that transformer-based baselines require the absolute position of each input node in order to locate them relative to one another. For this reason, in the Helmholtz Neumann problems with a plane wave as the incident field, the absolute node positions are given with respect to the average node position. In pure GNN approaches, this information is not mandatory since the relative position between each node is directly encoded through the edge features. However, for the sake of fairness, this input is used with all methods. For the Laplace Dirichlet and the Helmholtz Dirichlet problems, the node absolute coordinates must be provided regardless of the architecture due to $\mathbf { x } _ { \mathrm { 0 } }$ in the boundary conditions.

Table 7: Inputs of the node and edge encoders, and their corresponding dimensionalities for the Laplace Dirichlet problems.
<table><tr><td>Encoder</td><td>Input Feature</td><td>Dimensions</td></tr><tr><td rowspan="5">Node encoder</td><td>Sinusoidal encoding of the distance between the current node and x₀</td><td>128</td></tr><tr><td>Normalized direction of  $\mathbf { x } _ { \mathrm { 0 } }$  relative to the current node</td><td>3</td></tr><tr><td>Normalized direction v</td><td>3</td></tr><tr><td>Each term of the boundary condition computed at the current</td><td>3</td></tr><tr><td>node position</td><td></td></tr><tr><td rowspan="2">Edge encoder</td><td>Sinusoidal encoding of the edge length</td><td>128</td></tr><tr><td>Normalized direction of the edge</td><td>3</td></tr></table>

## E Addition Implementation details

The octree partitioning is performed using the implementation of Ripken et al. (2023). Every two levels are retained to construct the Downsampling and Upsampling Graphs.

All models are trained from scratch for 100 epochs, with AdamW optimizer (Loshchilov & Hutter, 2017), a batch size of 16, a learning rate starting at $1 0 ^ { - 4 }$ decreasing to $1 0 ^ { - 7 }$ with a cosine scheduler, and gradient clipping by norm with a maximum value of 1.0. Data augmentation is employed by applying the same random rotation to both the input mesh node positions and the source location.

All training experiments have been conducted on a single NVIDIA RTX 3090 GPU with 24GB memory. Training times range from 4 hours for Point Transformer v3, benefiting from FlashAttention acceleration Dao et al. (2022), to more than 18 hours for Transolver and Transolver++, due to gradient accumulation, which is required for batch sizes larger than 1. Otherwise, training time is 8 hours for ScaGNN, 11 hours for MuS-GNN, 12 hours for Erwin and 18 hours for MeshGraphNet. Note that FlashAttention cannot be used for training Erwin because its distance-based attention bias position encoding is essential for good performance but is not supported by FlashAttention.

Table 8: Inputs of the node and edge encoders, and their corresponding dimensionalities for the Helmholtz Dirichlet problems.
<table><tr><td>Encoder</td><td>Input Feature</td><td>Dimensions</td></tr><tr><td rowspan="5">Node encoder</td><td>Sinusoidal encoding of the distance between the current node and</td><td>128</td></tr><tr><td>the source  $\mathbf { x } _ { \mathrm { 0 } }$ </td><td></td></tr><tr><td>Normalized direction of the source x0 relative to the current node</td><td>3</td></tr><tr><td>Wavenumber k of the sample</td><td>1</td></tr><tr><td>Sine and cosine of the angle of the incident wave with wavenumber k coming from  $\mathbf { x } _ { \mathrm { 0 } }$ </td><td>2</td></tr><tr><td rowspan="5">Edge encoder</td><td>Sinusoidal encoding of the edge length</td><td>128</td></tr><tr><td>Normalized direction of the edge</td><td>3</td></tr><tr><td>Wavenumber k of the sample</td><td>1</td></tr><tr><td>Sine and cosine of the angle of an incident wave with wavenumber</td><td>2</td></tr><tr><td>k at the destination node coming from the source node</td><td></td></tr></table>

Table 9: Inputs of the node and edge encoders, and their corresponding dimensionalities for the Helmholtz Neumann problems.
<table><tr><td>Encoder</td><td>Input Feature</td><td>Dimensions</td></tr><tr><td rowspan="6">Node encoder</td><td>Normalized direction of v of the incident wave</td><td>3</td></tr><tr><td>Wavenumber k of the sample</td><td>1</td></tr><tr><td>Sine and cosine of the angle of the incident wave with wavenumber k</td><td>2</td></tr><tr><td>Sinusoidal encoding of the distance between the current node and</td><td>128</td></tr><tr><td>the average of node positions</td><td>3</td></tr><tr><td>Normalized direction of the average node position</td><td></td></tr><tr><td rowspan="5">Edge encoder</td><td>Sinusoidal encoding of the edge length</td><td>128</td></tr><tr><td>Normalized direction of the edge</td><td>3</td></tr><tr><td>Wavenumber k of the sample</td><td>1</td></tr><tr><td>Sine and cosine of the angle of an incident wave with wavenumber</td><td>2</td></tr><tr><td>k at the destination node coming from the source node</td><td></td></tr></table>

The implementation details of the diferent state-of-the-art methods we compare against are given in Table 10. The hyperparameters have been tuned to maximize performance on the Helmholtz Dirichlet problem while maintaining similar or higher computational cost and number of parameters compared to our approach. Due to the nature of the diferent architectures considered, one cannot simultaneously match the number of parameters and the computational cost of our method.

## F Stability Analysis

In Table 11, we provide the relative standard deviation over five inferences with diferent seeds to study the stability of our approach. The results show a relative standard deviation always lower than 0.3% regardless of the problem studied, the number of obstacles or the number of connections per node $N _ { \mathrm { e } }$ in the Distant Interaction Graph. This demonstrates the low variability of ScaGNN predictions despite the random processes involved in the dynamic adaptive edge sampling.

Table 10: Implementation details of the diferent methods we compare against in this paper.
<table><tr><td>Model</td><td>Parameter</td><td>Value</td></tr><tr><td>MeshGraphNet (Pfaff et al., 2020)</td><td>Processor depth Latent dimension</td><td>15 128</td></tr><tr><td>Point Transformer V3 (Wu et al., 2024b)</td><td>Grid Size Encoder Latent dimensions Encoder depths Encoder heads Encoder patch size Decoder Latent dimensions Decoder depths Decoder heads</td><td>0.2 (64, 128, 256) (2, 2, 6) (4, 8, 16) 2048 (64, 128) (2, 2) (4, 8)</td></tr><tr><td>Transolver (Wu et al., 2024a)</td><td>Stride Latent dimension Number of layers MPL ratio</td><td>2 256 8 4</td></tr><tr><td>Transolver++ (Luo et al., 2025)</td><td>Latent dimension Number of layers MPL ratio MPNN dim.</td><td>256 8 4</td></tr><tr><td></td><td>Latent dimensions Window sizes Encoder depths Encoder heads Decoder depths Decoder heads Stride</td><td>(64, 128, 256) (512, 512, 512) (2, 2, 6) (4, 8, 16) (2, 2) (4, 8)</td></tr><tr><td>MuS-GNN (Lino et al., 2022)</td><td>Distance-based attention bias MPNN dim. Scale number</td><td>2 Enabled 128 3</td></tr></table>

Table 11: Relative standard deviation results over five inferences.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=7>Number of obstacles</td><td rowspan=1 colspan=1>Laplace $\mathrm { E r r } _ { \mathrm { r e l } }$ </td><td rowspan=1 colspan=1>Helmholtz Dirichlet $\mathrm { E r r _ { a m p l } }$     $\mathrm { E r r _ { a n g l e } }$ </td><td rowspan=1 colspan=1>Helmholtz Neumann $\mathrm { E r r _ { a m p l } }$      $\mathrm { E r r } _ { \mathrm { a n g l e } }$ </td></tr><tr><td rowspan=3 colspan=1>Ours $\overline { { ( N _ { \mathrm { e } } = 4 ) } }$ Ours $( N _ { \mathrm { e } } = 4 )$  $\mathrm { O u r s } \left( N _ { \mathrm { e } } = 4 \right)$ </td><td rowspan=3 colspan=7>369</td><td rowspan=1 colspan=1>0.29%</td><td rowspan=2 colspan=1>0.20%     0.21%0.39%     0.11%</td><td rowspan=3 colspan=1>0.08%     0.12%0.07%     0.08%0.04%     0.03%</td></tr><tr><td rowspan=1 colspan=1>0.14%</td></tr><tr><td rowspan=1 colspan=1>0.18%</td><td rowspan=1 colspan=1>0.20%     0.20%</td></tr><tr><td rowspan=3 colspan=1>Ours $\overline { { ( N _ { \mathrm { e } } = 2 0 ) } }$ Ours $( N _ { \mathrm { e } } = 2 0 )$ Ours $( N _ { \mathrm { e } } = 2 0 )$ </td><td rowspan=3 colspan=7>39</td><td rowspan=1 colspan=1>0.31%</td><td rowspan=2 colspan=1>0.18%     0.21%0.15%     0.15%</td><td rowspan=3 colspan=1>0.10%     0.09%0.13%     0.09%0.09%     0.08%</td></tr><tr><td rowspan=1 colspan=2>6</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>0.24%</td></tr><tr><td rowspan=1 colspan=1>0.03%</td><td rowspan=1 colspan=1>0.16%     0.07%</td></tr></table>

## G FLOPs measurement

FLOPs is measured using Lightning measure\_flops function (Falcon & team, 2019). Point Transformer v3 (Wu et al., 2024b) enhanced conditional position encoding (xCPE) was re-implemented in native $\mathrm { P y }$ Torch (Team, 2024) so its computational cost can be captured by measure\_flops. Our native $\mathrm { P y }$ Torch version of xCPE was only used for measuring Point Transformer v3 FLOPs.

Figure 7 shows the computational cost of ScaGNN when $N _ { \mathrm { e } } = 4$ (in blue) and $N _ { \mathrm { e } } = 2 0$ (in orange) as a function of the number of nodes per sample in the test sets with 3, 6 and 9 obstacles. Linear regression fits, with coeficients of determination close to 1, confirm the linear relation between the size of the input mesh and ScaGNN computational cost.

![](images/cd16efcf9edd8d9d5e8de8bf6f800fc0a8fc7d86bd9c1ecc5eaaf155da12988d.jpg)  
Figure 7: Computational cost of ScaGNN with $N _ { \mathrm { e } } = 4$ (in blue) and $N _ { \mathrm { e } } = 2 0$ (in orange) as a function of the number of nodes per sample in the test sets with 3, 6 and 9 obstacles.

## H Repartition of the Expected Error

In this section, we analyse the repartition of the predicted errors on obstacles for the Helmholtz Dirichlet problem. Motivated by the underlying physics, in Table 12, we report the correlations between the predicted errors for each intermediate decoder $1 \leq k \leq K$ and diferent features that are expected to indicate strong or complex interactions. These features include the distance to the monopole source and the distance to the closest node from another obstacle. The first intermediate predictions $( k = 1 )$ are returned directly after the downsampling part of the GNN architecture when only local interactions have been modeled. In this situation, no node is aware of the other obstacles, so the intermediate predictions mainly highlight the regions closest to the monopole source, where the prediction errors are largest. For the subsequent intermediate predictions, Distant Interaction Blocks have already been applied, so each node on an obstacle is aware of the location of the other obstacles. This information is used to refine error prediction, resulting in higher predicted errors in regions where obstacles are close to each other. This behavior is also illustrated in Figure 8. Both our quantitative and qualitative studies show that areas with higher predicted errors correspond to the nodes that are more likely to be involved in the strongest or most complex interactions. Consequently, selecting edges based on error predictions favors the connection to such nodes.

Table 12: Correlations between the expected error predicted by each Distant Interaction Block and the inverse of the two distances: the source distance and the distance to the closest obstacle.
<table><tr><td>Index of the Distant Interaction Block</td><td>1</td><td>2</td><td>3</td></tr><tr><td>Inverse distance to the source Inverse distance to the closest obstacle</td><td>0.65 0.21</td><td>0.46 0.51</td><td>0.50 0.47</td></tr></table>

![](images/d4d5758c61a10d1041a2c2ffdc9d96ab9bd5525db8c2b37026ee83d3dd5f97b5.jpg)  
Figure 8: Expected error predicted by ScaGNN intermediate decoders for a sample from the Helmholtz Dirichlet test set. It is worth noting that the intermediate decoder predictions on the low-resolution graph have been interpolated to the original mesh.

Table 13: Correlations between the log mean absolute error and several dataset sample features.
<table><tr><td></td><td>Laplace</td><td>Helmholtz Dirichlet</td><td>Helmholtz Neumann</td></tr><tr><td rowspan="2">log number of GMRES iterations</td><td>-0.15</td><td>0.80</td><td>0.76</td></tr><tr><td></td><td>0.81</td><td>0.81</td></tr><tr><td>Log wavenumber Obstacle dispersion</td><td>0.06</td><td>-0.23</td><td>-0.11</td></tr></table>

## I Analysis of the model performance

In Table 13, we examine and analyze, for each problem with three ellipsoidal obstacles, the correlation between the performance and various properties of dataset samples, including the log number of GMRES iterations, the log wavenumber, and the obstacle dispersion. We quantify dispersion as the maximum of the minimum distances between obstacle pairs for each dataset sample. For wave propagation problems, the dificulty mainly depends on the wavenumber and the distance between obstacles, which both generate more complex reflections. Consequently, we anticipate that ScaGNN performance will decrease in these cases. This is indeed what we observe: strong correlations between ScaGNN prediction errors and the wavenumber. Regarding obstacle dispersion, the correlation with ScaGNN prediction errors is lower since most of the error is already explained by the wavenumber, but it is still present. The aforementioned sources of dificulty for wave problems are also reflected by the number of GMRES iterations to generate the groundtruth solution, so we naturally notice important correlations between the number of GMRES iterations and the ScaGNN performance. In comparison, ScaGNN prediction errors are weakly correlated with the number of GMRES iterations or with the obstacle dispersion for Laplace problems, which are not wave problems.

## J Additional Ablations

## J.1 Impact of the Number of Edge $N _ { \mathrm { e } }$ per Distant Interaction Graph

In Figure 9, we show how the error decreases when the number of edges per node in the Distant Interaction Graphs grows. We observe that the error reaches a lower bound when the Distant Interaction Graph is densely connected.

## J.2 Impact of the Number K of Distant Interaction Graph

Figure 10 illustrates how the number K of Distant Interaction Graphs afects performance on the Helmholtz Dirichlet problem, with corresponding parameter counts and FLOPs reported in Table 14. As K varies, we adjust the value of $N _ { \mathrm { d } }$ so that the Distant Interaction Graphs are processed by $K \times N _ { \mathrm { d } } = 6$ message-passing layers, restricting our study to $K \in \{ 1 , 2 , 3 , 6 \}$ . The growth in parameter count and FLOPs with increasing K stems from the initialization of additional edges and from the intermediate predictions. $K = 6$ does not always lead to the best performance since edge features are not carried over from one message-passing layer to the next, as each layer operates on a diferent Distant Interaction Graph. Overall, K = 3 ofers a good trade-of between performance gain and the added cost in parameters and FLOPs, and is therefore the value we adopt for our main results.

![](images/41a2de7d8d3deb8f2e6f1c167c3153268f540ffdb748828b5359e3c481864391.jpg)

![](images/19892df4f3465af34b2d2ab9f870161a877082dc807c20df83ed9c21c43b9e02.jpg)  
Figure 9: Estimation errors as a function of the number of edges $N _ { \mathrm { e } }$ per Distant Interaction Graph.“All” means that the Distant Interaction Graph is densely connected.

![](images/a4b95cad2f4b6cf13d070b1f5cb5989e8d3e9161947e14e879338a66ba174ac1.jpg)

![](images/b0e386b92fb2e8b99c6435c8e6639d195abd90f458814a4c7152cdfb710c8e79.jpg)  
Figure 10: Estimation errors as a function of the number K of distinct Distant Interaction Graphs processed in our GNN to model distant interactions.

<table><tr><td>K</td><td>Number of parameters FLOPs</td></tr><tr><td>1</td><td>5.4M 29.1 G</td></tr><tr><td>2</td><td>5.5 M 29.3 G</td></tr><tr><td>3</td><td>5.6 M 29.6 G</td></tr><tr><td>6</td><td>6.0 M 30.4 G</td></tr></table>

Table 14: Number of parameters and FLOPs for diferent values of K.

## J.3 Impact of the α Parameter in the Score Function $f _ { \mathrm { s c o r e } } ^ { \alpha }$

In Figure 11, we study the impact of the α parameter in the score function $f _ { \mathrm { s c o r e } } ^ { \alpha } ,$ which controls the relative importance given to the error predictions compared to the edge length when sampling distant edges. The results for the Helmholtz Dirichlet problem highlight that a negative $\alpha ,$ i.e., favoring connections from nodes with low predicted error, leads to higher errors. This can be interpreted by the fact that such edges are involved in weak interactions (see Section H), and therefore fail to bring relevant information to the destination node. When the model is evaluated with the same number of obstacles as in the training set, $\alpha = \infty$ provides the best performance, which means that $f _ { \mathrm { s c o r e } } ^ { \alpha }$ accounts only for the error predictions and the edge length is ignored. However, with two or three times as many obstacles as in the training set, a lower value of α improves performance. For the main results of this paper, we chose $\alpha = 1 . 0$ , which strikes a balance between the behaviors mentioned above.

![](images/9c4114070e915703238a6836f66d2ef802b72166f9e62d3a405e5ded05672d84.jpg)

![](images/8bae73da9d834aeb1be1733aa25e69a26f835ccf3be755601da2f11b80ac4d9b.jpg)  
Figure 11: Estimation errors as a function of the α parameter in the score function $f _ { \mathrm { s c o r e } } ^ { \alpha } .$ . When $\alpha = \infty$ , the score function only depends on error predictions and the edge length is ignored

9 obstacles   
6 obstacles   
3 obstacles

## J.4 Impact of the weight $\gamma$ in the loss $\mathcal { L } _ { \mathrm { t o t a l } }$

![](images/b0a3b4c69bdf8a75f12a5a6440f4ada162f82b50d31dc623e8b18915c75f8eb9.jpg)

![](images/30d47638f24d4f21832f2a03f72da519bfe4e6b5ac5d1eff4044d724a2f42d42.jpg)  
Figure 12: Estimation errors with three obstacles as a function of the γ parameter that weights the intermediate predictions in the loss function $\mathcal { L } _ { \mathrm { t o t a l } }$

Out-of-distribution shapes 中 In-distribution shapes

Figure 12 show the impact of the $\gamma$ parameter in the loss function ${ \mathcal { L } } _ { \mathrm { t o t a l } } .$ , which governs the weighting of the intermediate predictions. The results for the Helmholtz Dirichlet problem with three obstacles highlight that $\gamma = 0 . 2$ and $\gamma = 0 . 4$ are optimal when the model is evaluated on in-distribution obstacle shapes. Since we also observe lower error when $\gamma = 0 . 4$ on out-of-distribution shapes, we keep this value for the main results of this work.

![](images/0fc2e78140ef55b9880741ba3c107594c18e9393dafff362214fb93b81fd62c6.jpg)

![](images/9f5de653d60ca4730a123fc39ef7a22842341172e3ed9d048d59fbfb50926d9a.jpg)  
Figure 13: Estimation errors with three obstacles as a function of the number C of candidate edges from which each edge in a Distant Interaction Graph $\mathcal { G } _ { k } ^ { L - 1 } , 1 \le k \le K$ , is selected. More specifically, C varies during both training and testing.

## J.5 Impact of the Number C of Candidate Edges during Training

In Figure 13, we examine the efect of varying the number C of candidate edges from which each edge of a Distant Interaction Graph $\mathcal { G } _ { k } ^ { L - 1 } , 1 \le k \overset { \cdot } { \le } K$ , is selected during training. We obtain optimal results with $C = 2$

## J.6 Scaling the Number C of Candidate Nodes with the Number of Obstacles at Test Time

![](images/88e94c8341fa217bf61276d17608626b35b3040173b36e8854f794f387c20cbb.jpg)

![](images/f9e36f794d08dd89c51a1cd02423442a7f8aeb9d4219b574b50878435317fc03.jpg)  
Figure 14: Estimation errors as a function of the number C of candidate edges for selecting each edge in the Distant Interaction Graph $\mathcal { G } _ { k } ^ { L - 1 } , 1 \le k \le K$ . Here, C varies at test time while the evaluated model has been trained with $C = 2$

As the number of interactions grows quadratically with the number of nodes but the number of edges per node in the Distant Interaction Graph $\mathcal { G } _ { k } ^ { \tilde { L } - 1 } , 1 \leq k \leq K$ , is fixed, edge selection in our adaptive edge sampling must become more stringent. To ensure that the most relevant edges are retained, we increase the number of candidate edges C as the number of obstacles increases. Figure 14 illustrates that when our ScaGNN method is evaluated with more obstacles than seen in the training set, performance can be optimized by increasing the number C of candidate edges at test time while the model has been trained with $C = 2$ . This observation motivates a simple heuristic for scaling C with the number of obstacles: when the obstacle count is multiplied by a factor x, C should be increased by $x - 1$

## K Qualitative results

## K.1 Boundary solution

Figures 15 to 19 show qualitative results of the boundary solution for the Laplace Dirichlet, the Helmholtz Dirichlet and the Helmholtz Neumann problems with our ScaGNN approach and Point Transformer v3 (Wu et al., 2024b), the best baseline.

![](images/ccaffcd163a163b582db49e32479e13ac47c65bb6b4903d25608233b669dce67.jpg)

![](images/a16cfdd0bbf4ae3e40dde7fa93855c8e7598abcbc69c4722c6406b049a842632.jpg)

![](images/77d919a4df9124bc6ea09924167f540d5368bdf935c7ceb60df27d3f62faee79.jpg)

![](images/5fe66cbb364ab2c3f608ddc59b673a6f2278e7a5ca903230fd27f7a7aed5c910.jpg)

![](images/64552420e2d708cc12be978ca258558e35e89029038b2a3dbe2bfc4e7f97df55.jpg)

![](images/afde23dad393895930d5b78b0710df551e8c114e958549f9120acf2469d1b92e.jpg)  
Figure 15: Qualitative results on the Laplace Dirichlet problem with Point Transformer v3 (Wu et al., 2024b) and ScaGNN.

![](images/208d9e92a5f830992ac631f8ba6d4317bc892f3da3174fd5f2c6a86742b49ba8.jpg)

![](images/9c444d0f541aa3af96d475e35f22c68d82ed65fca69d4fadc323760e52839cf1.jpg)

![](images/c976673b50c635b19e69da3f9be1644f75ee5a617a536b71c442cf620a247a6e.jpg)

![](images/08f2b004c1f00be6abc205cf01e655c7200e3aba557222e8fabd4e7f0d9c27a1.jpg)

![](images/b4eafa6ba78d41d2fc5e83b73a537008d06d6fe18850399896984a3f60e0a02d.jpg)

![](images/040d7c3f2795826840eccd3459aafa5dd5238dfe961bf997273acb5eb88e3acd.jpg)  
Figure 16: Qualitative results for amplitude predictions on the Helmholtz Dirichlet problem with Point Transformer v3 (Wu et al., 2024b) and ScaGNN.

![](images/219e8ed79526331c02d5895d4a123ae27c4cb35c685061d62b4bd8c63bf07b82.jpg)  
Figure 17: Qualitative results for angle predictions on the Helmholtz Dirichlet problem with Point Transformer v3 (Wu et al., 2024b) and ScaGNN.

![](images/e1c64a20542b7c2377058b1e4afea06cf356836ccbc3bdc95e72c2f32c4df6cc.jpg)  
log-Amplitude

Relative amplitude error  
![](images/e28fbef01951cc0af9c1de73c23eb019fc7d00a81f944ee776115bc264b1ccd3.jpg)

![](images/f782a5e4c551183d117db62b94259b6b4a1daa0b2b93120862f828eecbfe8530.jpg)

![](images/ac2925620611938b7d9d26b39990874bda17ef299d8839ecfb45134d2fffbfb5.jpg)

![](images/f1d99785777124d41531ff91fe36211ec4597bf53e0d96d007a4bbda6b473a99.jpg)

![](images/3292e6d8eda0f516f9e953b721367d76c457317dc9f02f4ca2d39947a7673c0b.jpg)  
Figure 18: Qualitative results for amplitude predictions on the Helmholtz Neumann problem with Point Transformer v3 (Wu et al., 2024b) and ScaGNN.

![](images/dbbc622c7bc0212f008124473ef55028c52d8e60fd8aea57fcaca2ba72ae6b1a.jpg)

![](images/45c20bd74d699dcdbf4aee98855cb06739188a2c2061a823da40e18b28b4e506.jpg)

![](images/7e7eec7bc011833f610ba39af2f0d0ab5271dfbebb62045ce2718fd60b54fdd7.jpg)

![](images/53b64797d08b97ef44e6f80d9d1d890b1304cd27fbec155234406c85679d45c2.jpg)

![](images/f4f9ada0ecfd23aec9cff896d257c95eaa59bf33e664c6dee5ebd8aa1d631cec.jpg)

![](images/854d95d635a42c5f69c1a98b4ebc19c9802546c780d7cd80daff9d48660bd0b9.jpg)  
Figure 19: Qualitative results for angle predictions on the Helmholtz Neumann problem with Point Transformer v3 (Wu et al., 2024b) and ScaGNN.

## K.2 Volumetric solution

![](images/0efa6f9de452e301f71846f9f0769f86df19d33303e603e3eb93ada42b5a109a.jpg)  
Figure 20: From left to right are the volumetric solutions of the total field obtained with GMRES (the groundtruth), with ScaGNN and with PTv3 for estimating the trace solution on the boundary, respectively, and the corresponding errors relative to the groundtruth with ScaGNN and PTv3, respectively. For each problem, the volumetric solutions of the total field and their associated errors are sampled within a square domain of side length 10 on the plane z = 0, and the obstacles are represented in white.

In Figure 20, we provide qualitative results relative to the volumetric solution of the total field, which is the sum of the incident and the scattered field, for both ScaGNN and the best baseline, Point Transformer v3 (Wu et al., 2024b).