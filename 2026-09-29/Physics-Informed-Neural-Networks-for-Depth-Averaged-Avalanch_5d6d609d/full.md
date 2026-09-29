# Physics-Informed Neural Networks for Depth-Averaged Avalanche Dynamics

Pradyumn Singh Sikarwar Vishal Sharma Gaurav Bhutani<sup>∗</sup>

School of Mechanical and Materials Engineering, Indian Institute of Technology Mandi, India

## Abstract

Accurate prediction of avalanche motion is essential for hazard assessment in mountainous terrain. This study developed and evaluated a physics-informed neural network (PINN) framework for the Savage–Hutter model of depth-averaged granular flow, progressing systematically from one-dimensional analytical verification to two-dimensional experimental validation. In the first stage, three one-dimensional problems of increasing complexity were verified against the analytical solution: height prediction with prescribed velocity, velocity prediction with prescribed height, and coupled prediction of both fields using the conservative formulation. The decoupled tests demonstrated that PINNs accurately reconstructed the spatio-temporal evolution of each field when the other was prescribed. The coupled conservative formulation learned both fields simultaneously without any prescribed data, achieving mean height and velocity RMSEs of $4 . 3 \times 1 0 ^ { - 2 }$ and 7.9 10−<sup>2</sup> in non-dimensional units, respectively, compared with the analytical solution. A systematic hyperparameter sensitivity study evaluated the influence of network depth, width, collocation density, learning rate, and number of epochs on the conservative PINN formulation. In the second stage, the framework was extended to two dimensions and validated against laboratory experiments on the collapse of a cylindrical granular pile on an inclined plane, with <sub>TITAN2D</sub> providing numerical comparisons. Training with purely physicsbased losses led to a collapse to the trivial zero solution; this was resolved by augmenting the loss with 10 sparse training data points, drawn from the final deposit profiles, thereby resulting in a physics-informed, data-assisted hybrid framework. Peak flow depth, depth-averaged velocity, RMSE, and wetted-area IoU were used to evaluate agreement at both global and local levels. The framework achieved global height RMSE values ranging from 2.7 to 6.7 mm across the four experimental cases, with mean wetted-area IoU values ranging from 69 to 81 %, demonstrating consistent performance across variations in pile mass and slope angle.

Keywords: Physics-informed neural networks; Avalanches; Depth-averaged flows; Savage–Hutter model; Conservative formulation; Hyperbolic partial diferential equations

## 1 Introduction

Rapid granular flows driven by gravity, such as snow, rock, and debris avalanches, pose severe hazards to mountain communities and infrastructure worldwide. Since full-scale experiments with such flows are dangerous, expensive, and rarely repeatable, physics-based modelling has become the primary means of understanding their dynamics and assessing the associated risks. The behaviour of these flows is governed by strong gravitational forcing and frictional resistance, giving rise to nonlinear, unsteady dynamics that are dificult to describe analytically and demanding to resolve numerically.

When the flow thickness is much smaller than the downslope length scale, the vertical dependence can be integrated out, reducing the three-dimensional governing equations to a depthaveraged two-dimensional system for the flow thickness and depth-averaged velocity components. The model of Savage and Hutter [1989], arguably the most influential formulation in this class, captures the essential physics of rapid dry granular avalanches, including inertia, basal Coulomb friction, internal shear stress, and lateral earth pressure, within a hyperbolic system of conservation laws. Extensive work has since built on this framework, both in extending the theoretical analysis of the governing equations [Gray and Edwards, 2014] and in developing robust numerical techniques for their solution [Tai et al., 2002, Mangeney et al., 2003]. In particular, Pitman et al. [2003] and Patra et al. [2005] developed <sub>TITAN2D</sub>, a parallel adaptive-mesh Godunov solver for depth-averaged granular flows over natural terrain, which has become a widely used tool for hazard assessment. Despite their accuracy, mesh-based solvers can be dificult to set up and apply in practice. Discretisation choices, mesh generation over complex terrain, and the engineering efort required to develop general-purpose codes all impose non-trivial overhead before any simulation can be run. These considerations motivate the search for alternative solution strategies that reduce implementation complexity while retaining physical fidelity.

In recent years, physics-informed neural networks (PINNs) have emerged as a mesh-free alternative for solving diferential equations [Raissi et al., 2019, Karniadakis et al., 2021]. Rather than discretising the domain, PINNs encode the governing equations, initial conditions, boundary conditions, and, where available, observed data, directly into a neural network loss function, with derivatives computed via automatic diferentiation. This framework has since been applied to advection-dominated flows [Mao et al., 2020, Jagtap et al., 2020] and systems with complex source terms [Karniadakis et al., 2021], with substantial efort directed toward improving PINN trainability through gradient balancing [Wang et al., 2021], spectral-bias analysis [Wang et al., 2022], and domain-decomposition strategies [Jagtap and Karniadakis, 2020].

Within the context of depth-averaged flow equations, PINNs have recently been successfully applied to the shallow-water equations (SWEs) in hydraulic settings. Applications include sphericaldomain geophysical flows [Bihlo and Popovych, 2022], inverse PINNs that infer bed elevation from surface observations [Dazzi, 2024], benchmarking against finite-volume solvers for free-surface flows [Qi et al., 2024], river-model downscaling [Feng et al., 2023], and, more recently, unsteady riverine and complex-terrain domains [Yin et al., 2025, Tian et al., 2025].

Meanwhile, PINNs have also begun to appear in the broader granular-flow literature, though exclusively through continuum-level rheological models rather than depth-averaged formulations. Baldoni et al. [2026] used PINNs to identify the parameters of the µ(I)-rheology from synthetic granular column collapse data, treating it as an inverse problem on the full Navier–Stokes-like equations. Pai et al. [2025] proposed a Lagrangian PINN framework incorporating the µ(I)-rheology for granular flows, demonstrating its ability to capture free-surface evolution and particle trajectories in the column-collapse problem. To the best of the authors’ knowledge, this is the first study that applies PINNs to the depth-averaged granular avalanche equations. Although these equations share the structure of hydraulic shallow-water equations, they difer substantially in their constitutive physics, incorporating Coulomb basal friction and slope-dependent source terms arising from the inclined-plane geometry.

The Savage–Hutter system presents several features that complicate PINN training. First, basal resistance follows a direction-dependent Coulomb friction law rather than the smooth drag formulations typical of hydraulic models, introducing a non-smooth operator into the governing equations. Second, the Mohr–Coulomb earth pressure closure introduces sign-dependent source terms that can obstruct gradient-based optimisation. Third, the solution is compactly supported: the flow thickness vanishes outside a finite, time-dependent wetted region. Near the moving wet– dry interface, the sharp transition in flow depth, together with non-smooth Coulomb friction and velocity-normalisation terms, can produce poorly conditioned residuals at collocation points and destabilise PINN training.

The present study addresses these questions through a systematic two-stage investigation. Section 2 presents the mathematical formulation, including the one-dimensional governing equations, non-dimensionalisation, the analytical similarity solution, and the extension to two-dimensional depth-averaged equations. Section 3 describes the PINN methodology, covering the network architecture, residual formulation, and composite loss function. In the first stage (Section 4), the one-dimensional Savage–Hutter equations are solved and verified against the analytical solution for a parabolic-cap initial profile, through two decoupled single-field tests — height prediction with prescribed velocity, and velocity prediction with prescribed height — followed by the fully coupled prediction of both fields under the conservative formulation. In the second stage (Section 5), the framework is extended to two dimensions, addressing the trivial-solution pathology through sparse experimental observations, and benchmarked against the finite-volume solver <sub>TITAN2D</sub> [Patra et al., 2005]. Results are also validated against the laboratory experiments of Maeno et al. [2013] for the collapse of a cylindrical granular pile on an inclined plane, using root-mean-square error (RMSE), intersection-over-union (IoU) of wet-cell footprints, and time-resolved peak-height and velocity tracking. Section 6 summarises the findings and outlines future directions.

## 2 Mathematical Formulation

Granular avalanches were modelled as a single-phase, incompressible continuum, implying that the bulk density ρ remains constant in both space and time, and that volume changes due to granular dilation are negligible. This assumption has been widely adopted for dense flows where particles remain in contact and packing variations are minimal [Savage and Hutter, 1989]. Under these conditions, the equations of mass and momentum conservation governing the motion of the continuum are expressed as:

$$
\nabla \cdot \mathbf { u } = 0 ,\tag{1}
$$

$$
\rho \left( \frac { \partial \mathbf { u } } { \partial t } + \nabla \cdot ( \mathbf { u } \mathbf { u } ) \right) = - \nabla \cdot \pmb { \sigma } + \rho \mathbf { g } ,\tag{2}
$$

where u is the velocity vector, σ is the negative of the Cauchy shear stress tensor (so that compressive stresses are positive), and g denotes the acceleration due to gravity. In the one-dimensional depth-averaged formulation, the x- and z-directions denote the downslope and bed-normal directions, respectively. The y-direction represents the cross-slope direction, introduced in the two dimensional formulation in Section 2.2.

## 2.1 One-dimensional depth-averaged equations

The depth-averaged approach is valid when the flow thickness is much smaller than the downslope length scale, allowing the flow properties to be averaged over the depth. At the free surface, kinematic and stress-free boundary conditions were applied. At the bed, a no-penetration condition and a Coulomb friction law were imposed, the latter relating basal shear stress to the normal stress and the bed friction angle. In the depth-averaged formulation, vertical velocity and accelerations were neglected, and the vertical normal stress $\sigma _ { z z }$ was taken to be hydrostatic:

$$
\sigma _ { z z } ( z ) = \rho g _ { z } ( h - z ) .\tag{3}
$$

The normal stress in the x-direction was related to the vertical normal stress through the earth pressure coeficient, $k _ { a p } ,$ such that:

$$
\sigma _ { x x } = k _ { a p } \sigma _ { z z } ,\tag{4}
$$

where $k _ { a p }$ represents the ratio of horizontal to vertical normal stress. In this study, $k _ { a p } = 1$ was adopted, which is standard for unconfined granular flows on inclined planes [Mangeney et al., 2005, 2007]. The same value was used in the two-dimensional formulation.

Integrating Equations (1) and (2) from the bed to the free surface, applying Leibniz’s rule, and imposing the boundary conditions yields the one-dimensional depth-averaged continuity and momentum equations:

$$
\frac { \partial h } { \partial t } + \frac { \partial ( h u ) } { \partial x } = 0 ,\tag{5}
$$

and

$$
\frac { \partial ( h u ) } { \partial t } + \frac { \partial } { \partial x } \left( h u ^ { 2 } + \frac { g _ { z } h ^ { 2 } } { 2 } \right) = S _ { x } ,\tag{6}
$$

where h denotes flow depth along the z-direction, u is the depth-averaged velocity in the x-direction, $g _ { z } = g \cos \zeta$ is the bed-normal component of gravitational acceleration, ζ is the basal inclination angle, and $S _ { x }$ is the source term containing the gravitational and basal friction contributions. The $\textstyle { \frac { 1 } { 2 } } g _ { z } h ^ { 2 }$ represents the depth-averaged hydrostatic pressure arising from the horizontal normal stress. The source term is:

$$
S _ { x } = g _ { x } h - h g _ { z } \mathrm { s g n } ( u ) \tan \delta ,\tag{7}
$$

Here, $g _ { x } = g$ sin ζ is the downslope component of gravitational acceleration and δ is the bed friction angle. The sign function ensures that basal friction always opposes the direction of motion; this conservative form is particularly suited to PINN-based solvers, which propagate conservation-law solutions more accurately [Jagtap et al., 2020].

## 2.1.1 Non-dimensionalization

Savage and Hutter [1989] derived their analytical solutions in non-dimensional form by separating the characteristic flow thickness H from the downslope length scale L, with aspect ratio $\epsilon = H / L \ll$ 1. The same non-dimensionalisation was adopted here for the one-dimensional verification cases. The scaling is:

$$
\hat { x } = \frac { x } { L } , \qquad \hat { h } = \frac { h } { H } , \qquad \hat { u } = \frac { u } { \sqrt { g L } } , \qquad \hat { t } = t \sqrt { \frac { g } { L } } .\tag{8}
$$

Substituting into Equations (5)–(6), the non-dimensional continuity equation retains its form:

$$
\frac { \partial \hat { h } } { \partial \hat { t } } + \frac { \partial ( \hat { h } \hat { u } ) } { \partial \hat { x } } = 0 ,\tag{9}
$$

and the non-dimensional momentum equation in conservative form is:

$$
\frac { \partial ( \hat { h } \hat { u } ) } { \partial \hat { t } } + \frac { \partial } { \partial \hat { x } } \left( \hat { h } \hat { u } ^ { 2 } + \frac { \epsilon \cos \zeta } { 2 } \hat { h } ^ { 2 } \right) = \hat { h } \left( \sin \zeta - \mathrm { s g n } ( \hat { u } ) \tan \delta \cos \zeta \right) .\tag{10}
$$

The aspect ratio ϵ appears in the horizontal normal stress flux coeficient $\frac { \epsilon \cos \zeta } { 2 }$ as a direct consequence of the two-scale non-dimensionalisation: h is scaled by $H = \epsilon L$ while x is scaled by L. The non-dimensional system is governed by the slope angle $\zeta ,$ the bed friction angle $\delta ,$ and the aspect ratio ϵ.

## 2.1.2 Analytical similarity solution

A classical self-similar analytical solution of the one-dimensional Savage–Hutter model [Savage and Hutter, 1989] was employed to verify the PINN predictions. This solution describes a granular mass undergoing rigid-body translation combined with self-similar lateral spreading. All quantities are expressed in non-dimensional form introduced in Section 2.1.1.

A similarity coordinate centred on the accelerating pile is:

$$
\hat { \eta } ( \hat { x } , \hat { t } ) = \frac { \hat { x } - \frac { 1 } { 2 } \hat { a } \hat { t } ^ { 2 } } { \hat { g } ( \hat { t } ) } ,\tag{11}
$$

where $\hat { g } ( \hat { t } )$ denotes the non-dimensional half-width of the flowing mass and aˆ = sin $\zeta - \cos \zeta$ tan δ is the non-dimensional efective downslope acceleration (equal to the dimensional acceleration a normalised by g). The analytical flow height takes the parabolic similarity form:

$$
\hat { h } _ { \mathrm { a n a } } ( \hat { x } , \hat { t } ) = \frac { K } { \hat { g } ( \hat { t } ) } ( 1 - \hat { \eta } ^ { 2 } ) ,\tag{12}
$$

which is taken non-zero only for $| \hat { \eta } | \le 1$ , with $K = 2 \epsilon$ cos $\zeta .$ The initial parabolic pile $\hat { h } ( \hat { x } , 0 ) =$ max $( 1 - \hat { x } ^ { 2 } , 0 )$ is shown in Figure 1.

![](images/68af14cc6237d4e8070f273ef7b6a01e4dcb8b99cf89e4744df6bd7b97b3b56a.jpg)  
Figure 1: Geometric configuration of the one-dimensional benchmark problem. The initial pile follows the parabolic non-dimensional profile on an inclined plane of angle $\zeta .$ The front and rear positions are denoted by $\hat { x } _ { F }$ and ${ \hat { x } } _ { R }$

The temporal evolution of $\hat { g } ( \hat { t } )$ follows from the implicit relation:

$$
\sqrt { \hat { g } ( \hat { t } ) \big ( \hat { g } ( \hat { t } ) - 1 \big ) } + \ln \left[ \sqrt { \hat { g } ( \hat { t } ) } + \sqrt { \hat { g } ( \hat { t } ) - 1 } \right] = \sqrt { K } \hat { t } .\tag{13}
$$

Since Equation (13) has no closed-form inversion, $\hat { g } ( \hat { t } )$ was reconstructed numerically using 200 uniformly distributed time instances over $0 \leq \hat { t } \leq 4$ . The quadratic fit approximated the resulting profile well over the full time interval:

$$
\begin{array} { r } { \hat { g } ( \hat { t } ) \approx 0 . 1 1 6 1 3 \hat { t } ^ { 2 } + 0 . 7 4 4 9 7 \hat { t } + 1 , } \end{array}\tag{14}
$$

The analytical downslope velocity is:

$$
\hat { u } _ { \mathrm { a n a } } ( \hat { x } , \hat { t } ) = \sqrt { \frac { 2 K } { \hat { g } ( \hat { t } ) \left( \hat { g } ( \hat { t } ) - 1 \right) } } \hat { \eta } .\tag{15}
$$

## 2.2 Two-dimensional depth-averaged equations

The one-dimensional depth-averaged equations were extended to include a cross-slope direction denoted as $y ,$ following the same derivation approach with additional cross-slope terms in the continuity and momentum equations; full details are given in Sharma and Bhutani [2026].

The depth-averaged continuity equation in two dimensions is given as:

$$
\frac { \partial h } { \partial t } + \frac { \partial ( h u ) } { \partial x } + \frac { \partial ( h v ) } { \partial y } = 0 ,\tag{16}
$$

where $h ( x , y , t )$ denotes the flow depth in the z-direction in a bed-fitted (local) orthogonal coordinate system, with the $x$ aligned downslope and the y in the cross-slope direction. The variables u and v are the depth-averaged velocity components in the x- and y-directions, respectively.

The normal stresses in the $x -$ and y-directions follow the same $k _ { a p } = 1$ relation:

$$
\sigma _ { x x } = k _ { a p } \sigma _ { z z } , \qquad \sigma _ { y y } = k _ { a p } \sigma _ { z z } .\tag{17}
$$

Basal resistance followed a dry Coulomb friction law, $\tau ^ { b } = \sigma _ { z z } ^ { b }$ tan $\delta ,$ with the basal normal stress from Equation (3), given by:

$$
\sigma _ { z z } ^ { b } = \rho h g _ { z } .\tag{18}
$$

The internal shear stress $\bar { \sigma } _ { x y }$ , obtained from the Coulomb yielding and Mohr-circle considerations, is:

$$
\bar { \sigma } _ { x y } = \mathrm { s g n } \bigg ( \frac { \partial u } { \partial y } \bigg ) \ \bar { \sigma } _ { x x } \ \mathrm { s i n } \phi ,\tag{19}
$$

where the overbar denotes the depth-averaging operator.

Substituting the body force, basal friction, and internal stress expressions into the depthaveraged momentum equations yields:

$$
\frac { \partial } { \partial t } ( h u ) + \frac { \partial } { \partial x } \left( h u ^ { 2 } + \frac { g _ { z } h ^ { 2 } } { 2 } \right) + \frac { \partial } { \partial y } ( h u v ) = S _ { x } ,\tag{20}
$$

$$
\frac { \partial } { \partial t } ( h v ) + \frac { \partial } { \partial x } ( h u v ) + \frac { \partial } { \partial y } \left( h v ^ { 2 } + \frac { g _ { z } h ^ { 2 } } { 2 } \right) = S _ { y } ,\tag{21}
$$

where the Mohr–Coulomb source terms are:

$$
S _ { x } = g _ { x } h - \frac { u } { \vert \mathbf { u } \vert } \left( h g _ { z } \mu \right) - h \operatorname { s g n } \left( \frac { \partial u } { \partial y } \right) \sin \phi \frac { \partial ( g _ { z } h ) } { \partial y } ,\tag{22}
$$

$$
S _ { y } = g _ { y } h - \frac { v } { \vert \mathbf { u } \vert } \left( h g _ { z } \mu \right) - h \operatorname { s g n } \left( \frac { \partial v } { \partial x } \right) \sin \phi \frac { \partial ( g _ { z } h ) } { \partial x } ,\tag{23}
$$

where $\mathbf { u } = ( u , v )$ is the depth-averaged velocity field, and $\mu = \tan \delta$ is the basal friction coeficient. The general formulation includes curvature-correction terms involving the basal radii of curvature (see Sharma and Bhutani [2026] for the complete derivation); these vanish for the planar inclinedplane geometry used here, yielding Equations (22)–(23) as written. The same governing equations were solved by both <sub>TITAN2D</sub> and the PINN framework. In the two-dimensional setting, dimensional equations were used throughout, as <sub>TITAN2D</sub> provides its solution in dimensional form [Patra et al., 2005].

## 3 Physics-Informed Neural Network Framework

A PINN framework was constructed and trained to solve the depth-averaged Savage–Hutter equations in both one and two dimensions. The framework consists of a fully connected feed-forward neural network whose parameters were optimised by minimising a composite loss function encoding the governing equations, initial conditions, boundary conditions, and, where available, sparse training data points. All spatial and temporal derivatives required to evaluate the governing equations were computed using automatic diferentiation, thereby eliminating the need for a mesh or finite-diference stencil [Baydin et al., 2018]. A schematic of the framework is shown in Figure 2.

![](images/0731bd6796bf4bcdbe0711484d1102d7257bd3cf1a98cbf5988cad34eb837330.jpg)  
Figure 2: Schematic of the PINN framework used in this study.

The solution fields were approximated by a fully connected feed-forward neural network,

$$
\mathbf { U } _ { \theta } ( \mathbf { x } ) = \mathcal { N } _ { \theta } ( \mathbf { x } ) ,\tag{24}
$$

where $\mathcal { N } _ { \theta }$ denotes the network with trainable parameters $\boldsymbol { \theta } , { \bf x }$ the input coordinates, and $\mathbf { U } _ { \theta }$ the predicted solution fields: $( \hat { x } , \hat { t } ) \mapsto ( \hat { h } , \hat { u } )$ for the one-dimensional cases and $( x , y , t ) \mapsto ( h , u , v )$ for the two-dimensional cases.

All input coordinates were linearly normalised to $[ - 1 , 1 ]$ using the respective domain bounds. A hyperbolic tangent (tanh) activation, a standard for PINNs applied to conservation laws [Raissi et al., 2019, Mao et al., 2020], was used at each hidden layer. In the two-dimensional formulation, a softplus activation $\sigma ( z ) = \ln ( 1 + e ^ { z } )$ was additionally applied to the h output neuron to enforce $h \geq 0 ;$ the velocity outputs u and v were left unconstrained. In the one-dimensional setting, all outputs were left unconstrained; h remained non-negative throughout training without enforcement. All weights were initialised using the Xavier uniform scheme, and biases were initialised to zero.

Substituting the network output $\mathbf { U } _ { \theta }$ into the governing equations yields PDE residuals that are minimised at collocation points distributed over the interior of the spatio-temporal domain. Automatic diferentiation constructs a computational graph through which gradients propagate back to the network parameters, rather than merely evaluating derivatives at given points [Baydin et al., 2018].

For the two-dimensional system, substituting $\mathbf { U } _ { \theta } = ( h , u , v )$ into Equations (16), (20), and (21) yields three residuals:

$$
\mathcal { R } _ { \mathrm { m a s s } } = \frac { \partial h } { \partial t } + \frac { \partial ( h u ) } { \partial x } + \frac { \partial ( h v ) } { \partial y } ,\tag{25}
$$

$$
\mathcal { R } _ { x } = \frac { \partial ( h u ) } { \partial t } + \frac { \partial } { \partial x } \Big ( h u ^ { 2 } + \textstyle \frac { 1 } { 2 } g _ { z } h ^ { 2 } \Big ) + \frac { \partial ( h u v ) } { \partial y } - S _ { x } ,\tag{26}
$$

$$
\mathcal { R } _ { y } = \frac { \partial ( h v ) } { \partial t } + \frac { \partial ( h u v ) } { \partial x } + \frac { \partial } { \partial y } \big ( h v ^ { 2 } + \textstyle { \frac { 1 } { 2 } } g _ { z } h ^ { 2 } \big ) - S _ { y } ,\tag{27}
$$

where $S _ { x }$ and $S _ { y }$ are the Mohr–Coulomb source terms defined in Equations (22)–(23). The corresponding one-dimensional residuals follow directly from Equations (9)–(10) and are not repeated here. If the network exactly satisfies the governing equations, all three residuals vanish pointwise; the training objective was therefore constructed to minimise them at collocation points sampled within the computational domain.

The conservative flux terms hu, hv, $h u ^ { 2 }$ $h v ^ { 2 }$ , and huv were assembled as composite tensor expressions and diferentiated directly through the computational graph rather than expanded analytically via the product rule beforehand, which would dissolve the coupling between h and u (or v) into separate derivative branches before diferentiation. Tian et al. [2025] showed that this approach yields substantially higher accuracy than the product-rule expansion for shallow-water residuals; the same approach was adopted here.

The network was trained by minimising a composite loss function consisting of weighted physicsbased terms. The PDE residual loss enforces the governing conservation laws at $N _ { f }$ collocation points sampled within the interior of the spatio-temporal domain:

$$
\mathcal { L } _ { \mathrm { p d e } } = \frac { 1 } { N _ { f } } \sum _ { i = 1 } ^ { N _ { f } } \left( \vert \mathcal { R } _ { \mathrm { m a s s } } ( x _ { i } , y _ { i } , t _ { i } ) \vert ^ { 2 } + \vert \mathcal { R } _ { x } ( x _ { i } , y _ { i } , t _ { i } ) \vert ^ { 2 } + \vert \mathcal { R } _ { y } ( x _ { i } , y _ { i } , t _ { i } ) \vert ^ { 2 } \right) ,\tag{28}
$$

The initial condition loss enforces agreement with the prescribed initial state at $N _ { \mathrm { i c } }$ points sampled along the initial profile:

$$
\mathcal { L } _ { \mathrm { i c } } = \frac { 1 } { N _ { \mathrm { i c } } } \sum _ { j = 1 } ^ { N _ { \mathrm { i c } } } | \mathbf { U } _ { \boldsymbol \theta } ( x _ { j } , y _ { j } , 0 ) - \mathbf { U } _ { \mathrm { i c } } ( x _ { j } , y _ { j } ) | ^ { 2 } ,\tag{29}
$$

where $\mathbf { U } _ { \mathrm { i c } } ( x _ { j } , y _ { j } )$ denotes the prescribed initial profile evaluated at the j-th point. Boundary conditions were incorporated through

$$
\mathcal { L } _ { \mathrm { b c } } = \frac { 1 } { N _ { \mathrm { b c } } } \sum _ { k = 1 } ^ { N _ { \mathrm { b c } } } \left| \mathbf { U } _ { \boldsymbol \theta } ( x _ { k } , y _ { k } , t _ { k } ) - \mathbf { U } _ { \mathrm { b c } } ( x _ { k } , y _ { k } , t _ { k } ) \right| ^ { 2 } ,\tag{30}
$$

where $\mathbf { U } _ { \mathrm { b c } } ( x _ { k } , y _ { k } , t _ { k } )$ denotes the prescribed Dirichlet boundary value at the k-th point and time $t _ { k }$

In the two-dimensional formulation, the Savage–Hutter system is homogeneous in the depthaveraged fields: $h \equiv 0 , u \equiv 0 , v \equiv 0$ satisfy Equations (16)–(21) identically and therefore minimise $\mathcal { L } _ { \mathrm { p d e } }$ from the outset of training (see Section 5.2). To break this trivial attractor, a sparse data loss was appended:

$$
\mathcal { L } _ { \mathrm { d a t a } } = \frac { 1 } { N _ { d } } \sum _ { m = 1 } ^ { N _ { d } } \left| h _ { \theta } ( x _ { m } , y _ { m } , t _ { m } ) - h _ { m } ^ { \mathrm { o b s } } \right| ^ { 2 } ,\tag{31}
$$

where $N _ { d }$ is the number of observed depth values, $h _ { m } ^ { \mathrm { o b s } }$ is the measured flow depth at location $( x _ { m } , y _ { m } , t _ { m } )$ , and $h _ { \theta }$ is the network’s depth prediction at that point. The complete training objective was therefore

$$
\mathcal { L } _ { \mathrm { t o t a l } } = w _ { \mathrm { p d e } } \mathcal { L } _ { \mathrm { p d e } } + w _ { \mathrm { i c } } \mathcal { L } _ { \mathrm { i c } } + w _ { \mathrm { b c } } \mathcal { L } _ { \mathrm { b c } } + w _ { \mathrm { d a t a } } \mathcal { L } _ { \mathrm { d a t a } } ,\tag{32}
$$

with fixed scalar weights $w _ { \mathrm { p d e } } , \ w _ { \mathrm { i c } } , \ w _ { \mathrm { b c } } .$ , and $w _ { \mathrm { d a t a } }$ . For the one-dimensional verification cases, $w _ { \mathrm { d a t a } } = 0$ and the framework operated as a data-free solver. In the two-dimensional setting $w _ { \mathrm { d a t a } } >$ 0, rendering the framework a physics-informed, data-assisted hybrid.

The optimisation strategy difered between the one- and two-dimensional settings. The onedimensional verification cases were trained using Adam with a fixed learning rate of $\alpha = 1 0 ^ { - 3 }$ , with the number of epochs specified in Table 1. For the two-dimensional cases, training was carried out in two sequential stages: Adam $( \alpha = 1 0 ^ { - 3 } )$ for early-stage optimisation, followed by up to 1000 steps of L-BFGS to refine the solution once the parameters were near a local minimum.

The collocation strategy difered between the two settings. For the one-dimensional cases, collocation points were drawn from a uniform distribution over $\hat { x } \in [ - 8 , 1 0 ] , \hat { t } \in [ 0 , 4 ]$ . For the two-dimensional cases, the temporal coordinates of PDE collocation points were drawn from a Beta(1, 3) distribution to concentrate sampling near $t = 0$ , where depth gradients and the growth of the velocity field are steepest. Collocation points were resampled every 500 epochs during the Adam stage to prevent overfitting, and a <sub>ReduceLROnPlateau</sub> scheduler halved $\alpha$ whenever $\mathcal { L } _ { \mathrm { t o t a l } }$ did not decrease for 5000 consecutive epochs. No scheduler was applied in the one-dimensional setting, where a fixed learning rate produced stable convergence throughout.

## 4 One-Dimensional Analytical Verification

## 4.1 Problem configurations

Three one-dimensional test cases were set up to verify the PINN formulation against the analytical solution of Savage and Hutter [1989]: a decoupled height test, a decoupled velocity test, and a fully coupled test. In all cases, the spatial domain spanned $\hat { x } \in [ - 8 , 1 0 ]$ and the simulation ran over $\hat { t } \in [ 0 , 4 ]$ . These bounds were chosen to keep the granular mass within the incline plane throughout, with the asymmetry in $\hat { x }$ reflecting the net downslope transport. The initial condition in all three cases was a parabolic granular pile at rest (Figure 1),

$$
\hat { h } ( \hat { x } , 0 ) = \operatorname* { m a x } \Bigl ( 1 - \hat { x } ^ { 2 } , 0 \Bigr ) , \qquad \hat { u } ( \hat { x } , 0 ) = 0 ,\tag{33}
$$

with homogeneous Dirichlet conditions imposed at the spatial boundaries,

$$
\hat { h } ( - 8 , \hat { t } ) = \hat { h } ( 1 0 , \hat { t } ) = 0 ,
$$

to ensure the mass remained confined within the domain. The three configurations are summarised in Table 1.

In the decoupled continuity test, the flow depth $\hat { h } ( \hat { x } , \hat { t } )$ was predicted from the continuity equation alone, while the velocity field was prescribed from the analytical solution $\hat { u } _ { \mathrm { a n a } } ( \hat { x } , \hat { t } )$ and substituted directly into the continuity residual during training. In the decoupled momentum test, the velocity field $\hat { u } ( \hat { x } , \hat { t } )$ was learned from the momentum equation, with the analytical height profile $\hat { h } _ { \mathrm { a n a } } ( \hat { x } , \hat { t } )$ substituted into the momentum residual. In the coupled test, both $\hat { h } ( \hat { x } , \hat { t } )$ and $\hat { u } ( \hat { x } , \hat { t } )$ were learned simultaneously from the full conservative system (Equations (9)–(10)), with no field prescribed. The network architectures and sampling parameters for all three cases are given in Table 1; the decoupled and coupled hyperparameter settings were both determined through the sensitivity study presented in Appendix A.

Table 1: Hyperparameter settings for the one-dimensional decoupled and coupled formulations.
<table><tr><td>Hyperparameter</td><td>Decoupled-h</td><td>Decoupled-u</td><td>Coupled</td></tr><tr><td>Network architecture</td><td></td><td></td><td></td></tr><tr><td>Output variables</td><td> $h ( x , t )$ </td><td> $u ( x , t )$ </td><td> $h ( x , t ) , u ( x , t )$ </td></tr><tr><td>Width (Neurons per layer)</td><td>8</td><td>32</td><td>64</td></tr><tr><td>Depth (Hidden layers)</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Activation</td><td>tanh</td><td>tanh</td><td>tanh</td></tr><tr><td>Weight initialisation</td><td>Xavier uniform</td><td>Xavier uniform</td><td>Xavier uniform</td></tr><tr><td>Training</td><td></td><td></td><td></td></tr><tr><td>Optimiser</td><td>Adam  $1 0 ^ { - 3 }$ </td><td>Adam</td><td>Adam</td></tr><tr><td>Learning rate Epochs</td><td></td><td>10-3</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>5,000</td><td>10,000</td><td>20,000</td></tr><tr><td>Sampling</td><td></td><td></td><td></td></tr><tr><td> $N _ { \mathrm { c o l } } ~ \mathrm { ( P D E ~ p o i n t s ) }$ </td><td>2,000</td><td>10,000</td><td>20,000</td></tr><tr><td> $N _ { \mathrm { i c } }$  (IC points)</td><td>1,000</td><td>2,000</td><td>3,000</td></tr><tr><td> $N _ { \mathrm { b c } } \ \mathrm { ( B C \ p o i n t s ) }$ </td><td>1,000</td><td></td><td>3,000</td></tr><tr><td>Domain</td><td></td><td></td><td></td></tr><tr><td>x range</td><td>[-8,10]</td><td>[−8,10]</td><td>[−8,10]</td></tr><tr><td>t range</td><td>[0,4]</td><td>[0,4]</td><td>[0,4]</td></tr><tr><td>Input normalisation</td><td>[−1,1]</td><td>[−1,1]</td><td>[−1,1]</td></tr><tr><td>Loss weights</td><td></td><td></td><td></td></tr><tr><td> $w _ { \mathrm { p d e } }$ </td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td> $w _ { \mathrm { i c } }$ </td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td> $w _ { \mathrm { b c } }$ </td><td>1.0</td><td></td><td>1.0</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

## 4.2 Decoupled verification

In both decoupled tests, the PINN reproduced the analytical solution well across the full simulation window. For the Decoupled-h test (Figure 3), the initial parabolic shape was predicted accurately, and the PINN tracked the analytical profiles closely as the pile spread.

![](images/f1cddb865d9cff4956347b65661f0f61e0aaa2992e150f0bcc597446da17d209.jpg)  
(a) tˆ = 0.0

![](images/ccbd8fd116fe01eb122fe90b05a73f572e60fd143242baee89e7837f1970d52c.jpg)  
(b) tˆ = 0.5

![](images/481a9ed1fde7ae323bf0f61b423a76dedb18ac868287d6a06cf7d507540bbe0e.jpg)  
(c) $\hat { t } = 1 . 0$

![](images/3766dc390e32c32530a688948b9a7a130928f0b6b83f612b3b5e62b4ebdbf687.jpg)  
(d) $\hat { t } = 2 . 0$

![](images/9caeb56c384d57c8096ea19308a789930bd3d8d7156e3b82f00084e85b150790.jpg)  
(e) $\hat { t } = 3 . 0$

![](images/828bdb34228dbddbe78035067fcdd0b075c375884b7b821acff58cfe463fcf2f.jpg)  
(f) $\hat { t } = 4 . 0$  
Figure 3: Decoupled-h: analytical and PINN-predicted height profiles at six time instants.

For the Decoupled-u test, the predicted velocity profiles are shown in Figure 4. At $\hat { t } = 0$ the network returned a near-zero velocity field. As the flow developed, the PINN captured the increasing velocity and tracked the width of the active region throughout the process. Errors were concentrated at the front and rear edges of the pile, where the analytical velocity drops sharply, but the network prediction tapers smoothly beyond the support. This is a known limitation of smooth function approximators.

![](images/7b2b4c4610c840fd21c94d4f23e5391668e6930de19fcba59826141f8edab871.jpg)  
(a) tˆ = 0.0

![](images/073d4dbbec443818138e36efb18797818da86d6ff6c6e5a8ade9ba7a152dedd7.jpg)  
(b) tˆ = 0.5

![](images/1f01f183a101a173b6858ce8eb9cce977a426703311c7dbbb8f249db649c3f07.jpg)  
(c) tˆ = 1.0

![](images/ca9a2da2f0607a6d2e73a83efdd6b9214a1b996519baa8620ec7d9756855cabc.jpg)  
(d) tˆ = 2.0

![](images/d1d8f38f62e848359c26bb92cf2e9ebe022111dbbe3eaae232c5f1c99c57bc6c.jpg)  
(e) tˆ = 3.0

![](images/adf2d6e3e9087e87a54258926ab676ddca628a1f25d864d75799f6112d157424.jpg)  
(f) $\hat { t } = 4 . 0$  
Figure 4: Decoupled-u: analytical and PINN-predicted velocity profiles at six time instants.

To quantify agreement in both tests, the time-resolved RMSE was computed as

$$
\mathrm { R M S E } _ { \phi } ( \hat { t } ) = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \hat { \phi } _ { \mathrm { P I N N } } ( \hat { x } _ { i } , \hat { t } ) - \hat { \phi } _ { \mathrm { a n a } } ( \hat { x } _ { i } , \hat { t } ) \right) ^ { 2 } } ,\tag{34}
$$

where $\hat { \phi } \in \{ \hat { h } , \hat { u } \}$ . The temporal evolution of both errors is shown in Figure 5. Both curves share the same qualitative shape: a sharp rise during the early transient, and a steady decline thereafter. The early peak has two causes: the initial profile contains a sharp discontinuity at the pile edges that a smooth tanh network cannot represent exactly, and the flux divergence is largest and most localised near the compact front at early times. Once the pile flattens and spreads, both dificulties ease.

The diference in magnitude between the two curves is notable. The height RMSE (Figure 5a) peaked at roughly 0.020 and averaged $7 . 2 \times 1 0 ^ { - 3 }$ over the full interval, while the velocity RMSE (Figure 5b) peaked near 0.20 and settled at a higher level throughout. This gap reflects the inherent diference in dificulty between the two equations: the continuity equation is linear in $\hat { h }$ when uˆ is prescribed, whereas the momentum equation requires balancing nonlinear flux gradients against Coulomb source terms. Consistently, the Decoupled-u test required a wider network (32 vs. 8 neurons) and five times as many collocation points.

![](images/febe79ecac7317b95f1180e288386565144213030f5ef67d41508cfc0bff5339.jpg)  
(a) Height RMSE

![](images/a4a602a3bf34226817eff6ab9ed30dc7fb226e014aa48d2df63dc4f5dd4137cb.jpg)  
(b) Velocity RMSE  
Figure 5: Decoupled verification: temporal evolution of RMSE for the height field (a) and velocity field (b).

## 4.3 Coupled verification

In the coupled formulation, a single neural network maps $( \hat { x } , \hat { t } ) \mapsto ( \hat { h } , \hat { u } )$ and learns both fields simultaneously from the conservative Equations (9)–(10). The decoupled tests established that the momentum equation is the harder component; this dificulty is compounded here by the nonlinear coupling between the two fields, so a wider, more densely sampled network was required than in either decoupled test.

The predicted height and velocity profiles are compared against the analytical solution in Figures 6 and 7, and the training diagnostics are shown in Figure 8. In both fields, the PINN reproduced the analytical solution accurately across the full simulation window, with errors concentrated at the pile peaks during the early transient phase. The height profiles (Figure 6) show that the initial parabolic pile was recovered accurately at $\hat { t } = 0$ , with both the peak height and support width correctly reproduced. As the pile spread and flattened, the PINN closely tracked the analytical shape. At earlier times $( \hat { t } = 0 . 5 \substack { - 1 . 0 } )$ , small deviations appeared near the pile peak.

![](images/278399d59f0e069fee70b1b9a5c6d38e4e3c5ea732286dec4d2e051d4d9aef81.jpg)

![](images/247d4a0f2fec42789ae8cfd1d164ac08c01d72b058594d2566a354bfe8c68bcb.jpg)

![](images/279212f4f0ee243381c0c5cc6b1dbabaf41c1c070c2d07d5b077de8bd993bc7a.jpg)

![](images/a5b7cb32706314aa5d8dab18a97a87c2879b2facee6ea5b12270019b5a62654c.jpg)

![](images/d555a981a93577227c6ebcecb8b3a19b67fd4d105278cd8db37eb571ce818024.jpg)

![](images/944c6e95b90c059954977e2bb243bb704db6e1cf77d0d670a844a8489c405bb1.jpg)  
Figure 6: Coupled formulation: analytical and PINN-predicted height profiles at $\hat { t } = 0 , 0 . 5 , 1 . 0 .$ 2.0, 3.0, and 4.0.

The velocity profiles (Figure 7) showed the same overall pattern. $\operatorname { A t } { \hat { t } } = 0$ , the network returned the near-zero velocity field, consistent with the initial condition. As the flow developed, the PINN captured the increasing velocity and reproduced the linear velocity structure within the active region, with both slope and magnitude in good agreement with the analytical solution throughout. Discrepancies were confined to the front and rear edges of the pile, where the prediction tapered smoothly beyond the sharp cutof of the analytical solution.

![](images/83cb6d608e1086ab237fe864671edb9d56d0d5ff54cae7b7fd0b8ba9f59e80ff.jpg)  
Figure 7: Coupled formulation: analytical and PINN-predicted velocity profiles at $\hat { t } = 0 , 0 . 5 , 1 . 0 .$ 2.0, 3.0, and 4.0. The shaded band marks the active region.

The training diagnostics (Figure 8) confirmed that all three loss components decreased steadily over 20,000 epochs, with <sub>PDE</sub> and <sub>IC</sub> reaching $\mathcal { O } ( 1 0 ^ { - 5 } \ – 1 0 ^ { - 6 } )$ by the end of training. The finalepoch weight histogram was Gaussian $( \mu = 0 . 0 0 0$ and $\sigma = 0 . 1 4 9 )$ , with no sign of exploding or vanishing weights. The RMSE for both fields peaked near $\hat { t } \approx 0 . 3 – 0 . 5$ and then decayed, with the height error settling below 0.025 for the remainder of the simulation. Averaged over the full simulation window, the mean RMSE was $4 . 3 \times 1 0 ^ { - 2 }$ for height and $7 . 9 \times 1 0 ^ { - 2 }$ for velocity.

![](images/d2d3bb40e1a4f3baddbdc35fe9c8a1b5d6847e44541986daeaeb6a9659c75759.jpg)  
(a) Training loss components

![](images/4ea53271bb7701d44c857f82671bf0e18c9282534ece5b987e5b35e9a9e47c34.jpg)  
(b) Network weight histogram at final epoch

![](images/3c622c2210f9e8cd403701db36d44465f690cae0755a8e805ca867fad55337f3.jpg)  
(c) RMSE of height and velocity fields vs. time  
Figure 8: Coupled PINN training diagnostics: (a) component losses, (b) final-epoch weight histogram $( \mu = 0 . 0 0 0 , \sigma = 0 . 1 4 9 )$ , and (c) RMSE of height and velocity fields over time.

## 5 Two-Dimensional Validation: Experimental Granular Avalanche

## 5.1 Experimental configuration

The framework was next validated against the laboratory experiments of Maeno et al. [2013], in which a hollow cylinder filled with dry sand was rapidly lifted from an inclined surface, allowing the pile to collapse freely under gravity. Four experimental cases were considered, spanning two pile heights and two slope angles, as summarised in Table 2.

Table 2: Experimental cases from Maeno et al. [2013] were used to validate the two-dimensional PINN. All cases share $R _ { 0 } = 0 . 0 5 \mathrm { m }$ $\rho = 1 4 8 1$ kg $\mathrm { m ^ { - 3 } }$ $\phi = 2 4 ^ { \circ }$ , and $\delta = 2 8 ^ { \circ }$
<table><tr><td>Case</td><td>M (kg)  $\zeta ~ ( ^ { \circ } )$ </td><td> $H _ { 0 }$  (m)</td><td> $t _ { \mathrm { m a x } }$  (s)</td></tr><tr><td>C1</td><td>1.0 10</td><td>0.085</td><td>0.70</td></tr><tr><td>C2</td><td>1.0 15</td><td>0.085</td><td>0.70</td></tr><tr><td>C3</td><td>2.5 10</td><td>0.215</td><td>0.95</td></tr><tr><td>C4</td><td>2.5 15</td><td>0.215</td><td>0.95</td></tr></table>

The domain for all cases spanned $x ~ \in ~ [ - 0 . 4 , 0 . 8 ]$ m in the downslope direction and $y \in$ $[ - 0 . 4 , 0 . 4 ]$ m in the cross-slope direction, with the pile centroid initially at the origin. These bounds were chosen so the spreading mass remained within the domain throughout. The geometry is illustrated in Figure 9.

![](images/988e521166862749cd4051bdfa6e91887e186cfee8bbba8b22d1e048edea8665.jpg)  
(a)

![](images/11444ad916ce7b5319a82bf80a016ced377a86c8958e4a5d5c7a6aff02206545.jpg)  
(b)  
Figure 9: Three-dimensional schematic of the experimental configuration on a plane inclined at angle ζ to the horizontal, with x directed down-slope, y cross-slope, and z normal to the inclined surface. (a) Initial condition: a cylindrical granular pile of radius $R _ { 0 }$ and height $H _ { 0 }$ centred at the origin. (b) Final deposit: the collapsed pile spreads into an elongated elliptical deposit with greater runout in the downslope direction.

The primary analysis focuses on Case C4 $( M = 2 . 5 \mathrm { k g } , \zeta = 1 5 ^ { \circ } )$ , which combines the highest aspect ratio with the steepest slope and therefore represents the most demanding test of the framework. Results for Cases C1–C3 are also presented.

## 5.2 Trivial-solution pathology in two-dimensional PINNs

A fundamental dificulty arises when the Savage–Hutter system is trained using only PDE, initialcondition and boundary-condition losses: the network converges to the trivial zero solution $h \equiv 0$ $u \equiv 0 , v \equiv 0$ , rather than the physically meaningful flow. Two mechanisms drive this collapse.

First, the Savage–Hutter system is homogeneous in the depth-averaged fields, so $h = u = v = 0$ satisfies all governing equations exactly and yields a zero PDE residual, making the trivial solution a valid minimiser of the PDE loss from the start of training. Second, the physical deposit occupies only a small fraction of the domain; the optimiser is continuously pushed toward zero by a large proportion of collocation points lying outside the wetted region. The contribution of the second mechanism was investigated separately through a pseudo-2D strip-flow experiment, in which the cylindrical pile was replaced by a full-width strip to increase the wetted fraction of the domain; the results are presented in Appendix B.

Several strategies were explored to prevent this collapse. These included hard enforcement of the initial condition, causal weighting to prioritise the early-time solution, learning-rate scheduling, an additional mass-conservation loss, and non-dimensionalisation of the governing equations. Hard enforcement of the initial condition prevented the solution from collapsing to zero, but the network instead reproduced an almost stationary pile with little spreading or downslope motion. Causal weighting improved the temporal evolution and allowed the pile to spread, but the predicted height still decreased excessively at later times. Learning-rate scheduling slowed the progression toward the trivial state but did not prevent the eventual collapse. Similarly, the mass-conservation loss reduced the rate of mass loss but was not suficient to maintain a physically meaningful solution throughout the simulation. Non-dimensionalisation improved the overall flow behaviour and produced the most physically consistent result among the purely physics-based formulations, but the predicted height still collapsed at later times. Therefore, although these modifications improved various aspects of the training behaviour, none was suficient on its own to yield a stable and physically meaningful solution over the entire simulation period.

Together, these two mechanisms create a loss landscape in which the trivial solution sits in a deep, wide basin that is dificult to escape through gradient-based optimisation alone. To break this, a sparse data loss was appended to the training objective, incorporating a small number of experimental observations that provide a signal strong enough to pull the network away from zero and anchor it toward a physically meaningful solution.

## 5.3 Minimum-data requirement study

To determine the minimum number of data points required to escape the trivial solution identified in Section 5.2, a systematic study of data requirements was conducted, in which the number of data points per profile, N, was varied from 1 to 20. At each value of N, points were sampled along the centerline profile $( y = 0 )$ and the cross-slope profile $( x = 0 )$ of the deposit at the final time, yielding 2N data points in total. Three independent training runs were performed at each N to assess repeatability; details are provided in the Appendix.

The global height RMSE decreased sharply up to $N = 3 .$ , showed only a small further improvement by $N = 5$ , and changed little thereafter. On this basis, $N = 5$ points per profile was adopted for all subsequent cases, giving 10 training points per case in total.

![](images/29a723b4a40b4d39f32bbe8ce541b65f83faa8b94b12b6c52bcf87136a4c63c2.jpg)  
Figure 10: Global height RMSE as a function of the number of observations per profile, N, for C4 (2.5 kg, 15◦), compared with <sub>TITAN2D</sub> (orange circles) and experimental measurements (blue squares).

The ten training data points were selected to cover the key features of the deposit geometry in both directions: the five centerline points sampled the upstream face, the deposit peak, and the downstream tail, while the five cross-slope points spanned the full lateral extent of the deposit about the centerline. Exact point coordinates are provided in Appendix D.4.

## 5.4 Network architecture and two-stage training

The two-dimensional PINN took the input triple $( x , y , t )$ and produced the output triple $( h , u , v )$ via a fully connected network. All inputs were linearly scaled to [ 1, 1] using the domain bounds before entering the network. A softplus activation was applied to the h output to enforce non-negativity of the flow depth, while u and v were left unconstrained. The architecture is summarised in Table 3.

Table 3: 2D PINN architecture and training configuration for the Maeno et al. [2013] cases.
<table><tr><td>Property Input dimension</td><td>Value</td></tr><tr><td>Hidden layers Neurons per layer Activation Output activation on h Weight initialisation</td><td> $\textrm { 3 } ( x , y , t )$  5 64 tanh softplus Xavier uniform</td></tr><tr><td>Total trainable parameters Optimisers Adam epochs Initial learning rate α L-BFGS steps PDE collocation points  $N _ { f }$  Initial-condition points  $N _ { \mathrm { i c } }$  Boundary-condition points  $N _ { \mathrm { b c } }$ </td><td>17,091 Adam, L-BFGS 500,000  $1 0 ^ { - 3 }$  1,000 50,000 5,000 4,000 1</td></tr></table>

Training followed the two-stage protocol of Section 3, using the configuration in Table 3: 500,000 Adam epochs with a scheduler and collocation resampling, followed by up to 1,000 L-BFGS steps. The physics loss weights were set to $w _ { \mathrm { p d e } } = w _ { \mathrm { i c } } = w _ { \mathrm { b c } } = 1$ , while the data weight was elevated to $w _ { \mathrm { d a t a } } = 5 0$ . This elevation was necessary to prevent data loss from vanishing before the PDE and IC losses drove the network away from the trivial zero solution. The value $w _ { \mathrm { d a t a } } = 5 0$ was selected empirically to consistently prevent collapse across repeated initialisations.

All two-dimensional experiments were run on an NVIDIA RTX A5000 GPU, requiring approximately 6.5 h per case, compared with 3–12 min for the corresponding <sub>TITAN2D</sub> simulations (Table 4). At present, the PINN framework is computationally expensive to train than the conventional solver. This cost, however, corresponds to training each case independently from random initialisation. To investigate whether this overhead could be reduced, a preliminary transfer-learning study was conducted in which the network trained for Case C4 was used to initialise Case C2 rather than being reinitialised from scratch. Under this scheme, the training time was reduced by approximately 94 % relative to training from scratch, while the predicted deposit remained in close agreement with the corresponding <sub>TITAN2D</sub> reference. Although demonstrated for only a single case pair, this result suggests that a pre-trained network can serve as an efective initialisation for related flow conditions, substantially reducing the computational cost of training new cases. Beyond computationa performance, PINNs ofer an additional practical advantage over conventional numerical methods. Once the governing equations are incorporated into the loss function, the same computational framework can be applied to diferent classes of partial diferential equations without developing new discretisation schemes or implementing specialised numerical treatments for additional terms. Consequently, the efort required to derive, code, and validate discretisations for modified governing equations can be significantly reduced.

Table 4: Wall-clock training and simulation times for all four experimental cases. PINN training was performed on an NVIDIA RTX A5000 GPU; <sub>TITAN2D</sub> simulations were run on 8 CPU cores. The PINN time is identical across cases because training duration depends solely on the machinelearning parameters, which are held fixed throughout.
<table><tr><td></td><td></td><td>Case Configuration PINN training time (h)</td><td>TITAN2D simulation time (min)</td></tr><tr><td>C1</td><td> $1 . 0 \mathrm { k g } , 1 0 ^ { \circ }$ </td><td>≈6.5</td><td>3</td></tr><tr><td>C2</td><td> $1 . 0 \mathrm { k g } , 1 5 ^ { \circ }$ </td><td>≈6.5</td><td>7</td></tr><tr><td>C3</td><td> $2 . 5 \mathrm { k g } , 1 0 ^ { \circ }$ </td><td>≈6.5</td><td>6</td></tr><tr><td>C4</td><td> $2 . 5 \mathrm { k g } , 1 5 ^ { \circ }$ </td><td>≈6.5</td><td>12</td></tr></table>

## 5.5 Results: primary case C4

The results for C4 are presented in three parts: training convergence, spatial and temporal evolution of the deposit, and quantitative error metrics. The repeatability of these results across diferent random initialisations is assessed in Appendix D.

Training convergence is shown in Figure 11. By the end of the Adam stage, the total loss has decreased to $\mathcal { L } _ { \mathrm { t o t a l } } \approx 6 . 0 \times 1 0 ^ { - 4 }$ , with $\mathcal { L } _ { \mathrm { p d e } } \approx 4 . 6 \times 1 0 ^ { - 4 }$ and $\mathcal { L } _ { \mathrm { i c } } \approx 1 . 4 \times 1 0 ^ { - 4 }$ . The data loss fell to near zero by mid-training, indicating the network had fitted the ten training points and was extrapolating via the PDE and IC losses. The subsequent L-BFGS stage converged within approximately 50 steps, reducing the total loss to $\mathcal { L } _ { \mathrm { t o t a l } } \approx 4 . 5 \times 1 0 ^ { - 4 }$

![](images/ed350c709c7c68d1acceaebc2c226424f02a86b04f8620267d8759817f8f7235.jpg)  
(a) Adam stage (500 000 epochs)

![](images/47d810b4b59cd51f87af25b060d25669b33342cc6c16b080aab835f007e53baa.jpg)  
(b) L-BFGS stage (1000 steps)  
Figure 11: Training loss history for C4. (a) Adam stage: all loss components decreased monotonically; the data loss dominated early training and guided the network away from the trivial solution. (b) L-BFGS stage: rapid convergence within $\approx 5 0$ steps, reducing $\mathcal { L } _ { \mathrm { t o t a l } }$ from $6 . 0 \times 1 0 ^ { - 4 }$ to $4 . 5 \times 1 0 ^ { - 4 }$

Figure 12 presents the predicted centerline height profile across six training checkpoints. $\mathrm { { A t 1 0 ^ { 4 } } }$ Adam epochs, the network had not yet recovered the initial condition or any later-time profiles. The initial condition was captured by $1 0 ^ { 5 }$ epochs, and the bulk of the dynamic information was captured between $1 0 ^ { 5 }$ and $3 \times 1 0 ^ { 5 }$ epochs, as the profiles progressively aligned with <sub>TITAN2D</sub>. Beyond $3 \times 1 0 ^ { 5 }$ epochs, additional Adam iterations yielded only marginal changes, whereas L-BFGS sharpened the wavefront and removed edge artefacts.

![](images/3b59be9c171d192325ad69d78808c8100bb4f73f05573285574c83238114ba4b.jpg)  
(a) Epoch 10<sup>4</sup>

![](images/ce77be030e6b9dddef74aa0aa21659360171beffc2a8e336de9ca4e89c52c3d2.jpg)  
(b) Epoch 10<sup>5</sup>

![](images/16bd8654207ffb4421e69df3286beaf2e79b67c64d04a4ba164d399ec4cf00ec.jpg)  
(c) Epoch 2  10<sup>5</sup>

![](images/0b4ed12229616b448afb425a5e4e7e5e2a0bcab0b8d09c1346ef770915b71e07.jpg)  
<sup>(d)</sup> <sup>Epoch</sup> <sup>3</sup> × <sup>105</sup>

![](images/26f9a3c806441b5ee9cb1a5568fe3df667a86e057714e713e3a5bf56ba39f7af.jpg)  
(e) Epoch 5  10<sup>5</sup>

![](images/fb9d0f9cc7c336f179d699d025d1c217ff52cb2db1075b144b31ea73c5831a63.jpg)  
(f) L-BFGS (final)  
Figure 12: Centerline profile h(x, y=0, t) for C4 at six training checkpoints. PINN (solid), <sub>TITAN2D</sub> (dashed) at t = 0, 0.32, 0.63, 0.95 s.

The centerline and cross-slope height profiles at the four time snapshots are shown in Figure 13. The cylindrical initial condition was recovered accurately at $t = 0 \mathrm { s }$ in both directions, and the collapse and lateral spreading were closely tracked across all intermediate snapshots. The front position and runout length agreed well with <sub>TITAN2D</sub> along the centerline, and the deposit width and peak height were well reproduced in the cross-slope direction throughout.

![](images/d13aa0bbc8db0acf502b6544784092685578c57eac86e4a39dd14524bc28cdcd.jpg)

![](images/e5e064af9ce8d84a69cecacce6e4ee01778ab7cb4315317171e1d221ca5f99da.jpg)

![](images/44e112c73052033fa754018ccbe848428c8dcd6bb8e1b0af5da8d939d8408268.jpg)

![](images/353320401f3776091f7d80440cc50cc99775e3208a4781aab2c0c082eb048711.jpg)

![](images/dc014a66716a5a5e6fa8a5a0544b238ab526022e0eaf2e21734afc499ef69b99.jpg)

![](images/522e922f8a0c8f6f7aa63ffe5c4cb81459be4e2f6c44c45c77eed91cd687187b.jpg)

![](images/045123a71d0f43de4b9d6c76d7f6600f9557753a13cd951261f3f2358f02551a.jpg)  
Downslope position x (m)

![](images/6b97ad95f26691c400146530e99002535187fbfb3b95d4907d0f832927d94cb3.jpg)  
Figure 13: Centerline (left) and cross-slope (right) height profiles at four time snapshots for C4. PINN (solid blue) and <sub>TITAN2D</sub> (dashed orange).

The final deposit profiles at $t = 0 . 9 5 \mathrm { s }$ are compared with the experimental measurements of Maeno et al. [2013] in Figure 14, with the ten training data points indicated by filled red circles. Along the centerline, the PINN accurately reproduced the peak height and downstream tail out to the full runout, whereas <sub>TITAN2D</sub> slightly underpredicted both. Along the cross-slope direction, both solvers accurately reproduced the deposit width, though <sub>TITAN2D</sub> produced a sharper lateral cutof, whereas the PINN yielded a smoother transition to the dry region. This closer agreement suggests that the combination of the physics constraint and sparse training data points steered the solution toward the measured deposit geometry.

![](images/151289971730ecab6ae26d01876190445333a485ba55d856b59946c494354746.jpg)  
x (m)

![](images/a59166bb86cc360a5a6ac6be136cc9931414697e1efbbf98fdda63a69052bebd.jpg)  
y (m)  
Figure 14: Final deposit profiles at $t = 0 . 9 5 \mathrm { s }$ for C4. Left: centerline $( y = 0 )$ ; right: cross-slope $( x = 0 )$ . PINN (solid blue), <sub>TITAN2D</sub> (dashed red), and Maeno et al. [2013] experimental data (circles). Filled red circles indicate the training data points.

The depth contours at four time instants are compared in Figure 15. $\mathrm { A t      } ~ t = 0 \mathrm { s }$ , both solutions reproduced the cylindrical initial pile with peak depths near 21.5 cm and a compact circular footprint. As the avalanche developed, the deposit spread preferentially downslope, with PINN contour shapes and levels remaining in close agreement with <sub>TITAN2D</sub>. For visualisation, a depth cutof of 0.05 cm was used to display the low-depth extent of the flow and as the lowest contour level in each panel. This visualisation cutof is distinct from the threshold $h _ { \mathrm { t h r } } = H _ { 0 } / 5 0$ used for the quantitative IoU and wetted-area calculations. By $t = 0 . 9 5 \mathrm { s }$ , both solutions yielded a thin, elongated deposit with peak depths below 2 cm, but the PINN extended visibly farther downslope than <sub>TITAN2D</sub>.

![](images/b4114c873841073b81841a5bb48f31daad7c3617ea8a88ad86bbaf28936b4497.jpg)

![](images/b2f50259c42662155fada3631d00434626b59673ebdbc320b9d21db8229a22ad.jpg)

![](images/c6f5b16f7e045e5d87346f23b5ddfb71bccc9fc38ce4cfdd78ba8d395ae7593d.jpg)

![](images/19f7a3b9c2118e36a7bddfe24d57a67d1f695e3759ced0a9d17c673678a8b646.jpg)

![](images/336f101786a2e27bdc59a1941a5ba0e32e562f8a0a9ffe8db4d0be2f93dc132f.jpg)

![](images/5d6e27aafa414bc949aa85efe9e46c16be85849bf7294d99ed29d636af0f7bfc.jpg)

![](images/372633df7af84e619f749b5d827b1cd92383eff119756fda536959289283db76.jpg)

![](images/77566be5d290853c1f36f65d8271df47911bb9e22f012cd6c0c0a65e1d3f4151.jpg)  
Figure 15: Depth contours for C4 at $t = 0 . 0 0 , 0 . 2 3 , 0 . 4 7$ , and 0.95 s (columns, left to right). Top row: PINN; bottom row: <sub>TITAN2D</sub>. Contour levels are identical across columns. The grey region marks the wetted footprint, defined as the area where the flow depth exceeds the minimum threshold.

The integral flow diagnostics are presented in Figure 16. The maximum height $h _ { \mathrm { m a x } }$ decayed monotonically from 21.5 cm as material redistributed downslope, and the PINN tracked this decay closely throughout. The mean velocity V increased sharply over the first 0.1 s, peaked near $0 . 7 \mathrm { m } / \mathrm { s } ,$ and then decreased steadily as basal friction dissipated momentum; both phases were well reproduced, with only a small ofset near the peak. The same quantities, plotted against runout distance, show the height traces remaining closely aligned throughout, while the velocity trace shows a larger gap in the middle of the runout range before converging again near the end.

![](images/da9cc512bbd6f6f87aff0261b56f238ff689e8bde8459f2a593f23a449f95e1d.jpg)

![](images/fc555d44f50c15df77945bfffe6e6215c8cb71ab30f95077e9e5f5895c545fcd.jpg)

![](images/78c8b37bfea64f010569a0ca804a3c469d9fc414435e683ac5df5afc28390985.jpg)

![](images/b96a0d16023662c330bf94abce058dcd239dbfcf68a01559a91a7f6b4952219a.jpg)  
Figure 16: Integral flow diagnostics for C4: $h _ { \mathrm { m a x } }$ and V versus time (top row) and runout distance (bottom row). <sub>TITAN2D</sub> (solid red); PINN (dashed blue).

The time-resolved RMSE and wetted-area IoU are shown in Figure 17. The RMSE was largest during the early collapse phase and decreased steadily as the deposit thinned; the global value of 6.7 mm, averaged over the full simulation window, represents $3 \%$ of the initial pile height. All quantitative wetted-region metrics reported in this section, including the IoU and wetted-area calculations, were evaluated using the relative depth threshold $h _ { \mathrm { t h r } } = H _ { 0 } / 5 0$ . For Case C4, this corresponds to $h _ { \mathrm { t h r } } = 4 . 3 \mathrm { m m }$

![](images/f4eed44ac800f188ad277c77ca749342bef8405132ad6b5bdce3a47c234f0a48.jpg)

![](images/4ecade4acd72cdb432aaccf855d19f8856e1e6ada94de14543d73c807ff29905.jpg)  
Figure 17: Time-resolved error metrics for C4. Left: RMSE of the depth field h; the global value averaged over all time steps is $6 . 7 \times 1 0 ^ { - 3 } \mathrm { m }$ . Right: Intersection over Union (IoU) of the wetted area; the time-averaged value is 69.0 %.

The final wetted footprints are compared in Figure 18. The PINN wetted area of $\mathrm { 0 . 2 0 9 1 m ^ { 2 } }$ exceeded the <sub>TITAN2D</sub> reference of $\mathrm { 0 . 1 5 7 9 m ^ { 2 } }$ , with the excess confined to the deposit periphery.

![](images/4e5838e43708370848a7f67cbbb758455acfe8f296a7c8ddfaa880842960a39a.jpg)  
Figure 18: Final wetted footprints at $t = 0 . 9 5 \mathrm { s }$ for C4. Blue: PINN $( A _ { \mathrm { { P I N N } } } = 0 . 2 0 9 1 \mathrm { { m ^ { 2 } } ) }$ ; red: TITAN2D $( A _ { \mathrm { T i t a n } } = 0 . 1 5 7 9 \mathrm { m ^ { 2 } } )$ ; purple: overlap region. The black circle marks the initial pile boundary. Wet cells were classified using the threshold $h _ { \mathrm { t h r } } = H _ { 0 } / 5 0 = 4 . 3 \mathrm { m m }$

## 5.6 Multi-case validation

To assess whether the performance observed for C4 was maintained across diferent flow conditions, the framework was applied to three additional cases, C1 (1 kg at 10◦), C2 (1 kg at 15◦), and C3 (2.5 kg at 10◦), using the identical network architecture, training procedure, and data points throughout. Each was evaluated from a single training run, as was C4. The repeatability associated with this choice is characterised in Appendix D.

The final deposit profiles for Cases C1–C3 are shown in Figure 19. In all three cases, the PINN reproduced the experimental peak height and deposit width closely in both the centerline and cross-slope directions. Along the centerline, the PINN also tracked the downstream deposit tail more closely to the experiment than <sub>TITAN2D</sub>, which truncated this region earlier in all three cases.

![](images/76a4e5bafb60a2f8acb8c56d15c8825b68759783fa80d781a25fbb1be3e34d9e.jpg)  
x (m)

![](images/f1d31ee158eda94015feedc88aca5dc24603fc9008dc268b082ff32b15c6794a.jpg)  
y (m)

![](images/11b0533ee0b81728c1221817729a1b5fafdd50d15bdbbe362cda668d11b49275.jpg)  
x (m)

![](images/6bf27c7d45d03f6c913207e97cf1036cbf5f435293554926b045d6373c105a28.jpg)  
y (m)

![](images/efcc5929d54f41e8d274059598a1a85f9addf049a8859f92ea735b681594ce8f.jpg)  
x (m)

![](images/9eb198cca90a952b310c26ab715236d8d6f1a71d2e2d439ece534216525f38a0.jpg)  
y (m)  
Figure 19: Final deposit profiles for Cases C1–C3. Left column: centerline $( y = 0 )$ ; right column: cross-slope $( x = 0 )$ . Rows from top to bottom: C1 (1 kg 10◦), C2 (1 kg 15◦), and C3 (2.5 kg 10◦). PINN (solid blue), <sub>TITAN2D</sub> (dashed red), and experimental data from Maeno et al. [2013] (circles). Filled red circles indicate training observations.

The depth contours at $t = 0 . 2 0 , 0 . 4 0$ , and 0.70 s for Cases C1–C3 are presented in Figure 20. In all three cases, the footprint grew, and the peak depth decreased over time, with the initially compact boundary developing local asymmetries and lobed extensions that persisted from $t = 0 . 4 0$ to 0.70 s.

![](images/2be008c2b28c154ad85360cd1f82582a084a8144bc79c19a3b9397bc7a29110c.jpg)

![](images/660716422b644a023697b735feb7cd3171a38c78d843e997653a541b6fa1acda.jpg)

![](images/96f5216058eb1b0f06327b5d3f47a545de81927e9bcb220dc7f66cdcfabb67b3.jpg)

![](images/47d61d8095cd655097fe339bd97c9c240d5ce7e1087f3c8c2440784c4747f677.jpg)

![](images/6de575f5680a8d2a23cdd4c1f438975094356d7e29ef8cca93f9a269bc2ae27f.jpg)

![](images/41634e1ff7804437f8cce409f64ca2495aefa0ac4fa5b4465e2c0ecc0cddf10c.jpg)

![](images/74380f1a2fd7508485ffdc1fdbeda83f53028bc07fb33de76365a1d4c1a6d758.jpg)

![](images/df2e0f458185b815d328af0b9c16244e1d6b9de63ab150bc45a8c7bc9610478d.jpg)

![](images/32098cab84a72d2477a0168eb7a84fff6c83818b94b7d70199a2d13b35ef4521.jpg)  
Figure 20: PINN depth contours at $t = 0 . 2 0 , 0 . 4 0 .$ , and 0.70 s (columns, left to right) for Cases C1– C3 (rows, top to bottom): 1 kg 10◦, 1 kg 15◦, and 2.5 kg 10◦. The grey region marks the wetted footprint, defined using the same height cutof of 0.05 cm used for C4.

The global RMSE and mean IoU for all four cases are collected in Table 5. RMSE increased with pile mass (1 kg: 2.7–3.1 mm; 2.5 kg: 4.7–6.7 mm), reflecting the steeper depth gradients and more energetic early collapse of larger piles under a fixed collocation budget. Expressed relative to the initial pile height, the global RMSE ranged from 2.2 % (C3) to 3.6 % (C1) across the four cases. The IoU was highest for Case C2 (80.7 %) and lowest for C4 (69.0 %). In all cases, RMSE peaked during the early collapse phase and decreased as the deposit settled, consistent with the trend observed for C4.

Table 5: Global RMSE and mean IoU for all four experimental cases.
<table><tr><td>Case</td><td>Configuration</td><td>Global RMSE (m)</td><td>Mean IoU (%)</td></tr><tr><td>C1</td><td>1.0 kg, 10°</td><td> $\overline { { 3 . 1 \times 1 0 ^ { - 3 } } }$ </td><td>70.9</td></tr><tr><td>C2</td><td>1.0 kg, 15°</td><td> $2 . 7 \times 1 0 ^ { - 3 }$ </td><td>80.7</td></tr><tr><td>C3</td><td>2.5 kg, 10°</td><td> $4 . 7 \times 1 0 ^ { - 3 }$ </td><td>73.1</td></tr><tr><td>C4</td><td>2.5 kg, 15°</td><td> $6 . 7 \times 1 0 ^ { - 3 }$ </td><td>69.0</td></tr></table>

The final wetted footprints for Cases C1–C3 are shown in Figure 21, and the corresponding wetted areas for all four cases are given in Table 6. The PINN wetted area exceeded the <sub>TITAN2D</sub> reference in every case, with the excess ranging from 12.1 % in Case C2 to 32.4 % in Case C4. The excess increased in proportion to mass at a fixed slope angle. In all cases, the overlap region covered the bulk of the <sub>TITAN2D</sub> footprint, indicating that the deposit interior was well reproduced and that the discrepancies were confined to the wet–dry boundary.

![](images/47641f367bbf607cb86aa522d836b6915b118afd80154d3c481e12330a2f8478.jpg)

![](images/99346f9bd57f8fd14be340cca63993c39ec1529463f097f368109982ff2b24b2.jpg)

![](images/b21b25b30ad3726eca79a2585fecfa0c804611b485d700eb4a3d6647084b40ec.jpg)  
Figure 21: Final wetted footprints for Cases C1–C3 (left to right): 1 kg $1 0 ^ { \circ }$ , 1 kg $1 5 ^ { \circ }$ , and 2.5 kg $1 0 ^ { \circ }$ PINN (blue), <sub>TITAN2D</sub> (red), and overlap (purple). The black circle marks the initial pile boundary. Wetted areas are annotated on each panel. Wet cells were classified using $h _ { \mathrm { t h r } } = H _ { 0 } / 5 0$ , giving 1.7 mm for C1–C2 and 4.3 mm for C3.

Table 6: Final wetted areas for all four experimental cases.
<table><tr><td>Case</td><td>Configuration</td><td> $A _ { \mathrm { P I N N } }$   $( \mathrm { m } ^ { 2 } )$ </td><td> $A _ { \mathrm { T I T A N 2 D } }$   $( \mathrm { m } ^ { 2 } )$ </td><td>Excess  $( \% )$ </td></tr><tr><td>C1</td><td> $1 . 0 \mathrm { k g } , 1 0 ^ { \circ }$ </td><td>0.1191</td><td>0.0965</td><td>23.4</td></tr><tr><td>C2</td><td> $1 . 0 \mathrm { k g } , 1 5 ^ { \circ }$ </td><td>0.1190</td><td>0.1061</td><td>12.1</td></tr><tr><td>C3</td><td> $2 . 5 \mathrm { k g } , 1 0 ^ { \circ }$ </td><td>0.1967</td><td>0.1538</td><td>27.9</td></tr><tr><td>C4</td><td> $2 . 5 \mathrm { k g } , 1 5 ^ { \circ }$ </td><td>0.2091</td><td>0.1579</td><td>32.4</td></tr></table>

## 6 Conclusions

This study demonstrated, for the first time, that physics-informed neural networks can solve the depth-averaged Savage–Hutter granular avalanche equations, progressing systematically from onedimensional analytical verification to two-dimensional experimental validation against the laboratory measurements of Maeno et al. [2013].

In the one-dimensional setting, decoupled single-field tests showed that the continuity equation was readily satisfied by a compact network, whereas the momentum equation required greater architectural capacity and collocation density, reflecting the nonlinear coupling. The fully coupled conservative formulation resolved both fields simultaneously without any prescribed data, achieving mean height and velocity RMSEs of $4 . 3 \times 1 0 ^ { - 2 }$ and $7 . 9 \times 1 0 ^ { - 2 }$ in non-dimensional units, respectively, compared with the analytical parabolic-cap solution.

In the two-dimensional setting, training with purely physics-based losses led to collapse onto the trivial zero solution. Augmenting the loss with ten sparse training data points per case, drawn from the final deposit profiles along the centerline and cross-slope directions, provided suficient signal to anchor the optimisation away from the zero attractor while preserving the physical consistency enforced by the PDE constraint. In post-event forensic scenarios, final deposit measurements are often the only data available at an avalanche site; the present results indicate that such measurements, combined with the governing equations, are suficient to reconstruct the full spatio-temporal flow history with accuracy consistent across variations in pile mass and slope angle, as reported above.

Three limitations apply to the present results. The smooth output of a fully connected network cannot represent the sharp wet–dry interface characteristic of granular avalanche fronts, leading to a systematic overestimation of the wetted area. The data-assistance requirement means the framework is not a fully data-free solver, and its applicability depends on the availability of at least sparse final-deposit measurements. The training time of approximately 6.5 h per case substantially exceeds the wall-clock time required for a single <sub>TITAN2D</sub> simulation.

Several directions for future work follow from these limitations. Extension to irregular terrain, via bed-fitted coordinate transformations or terrain-aware architectures, is required for real geophysical hazard applications. The integration of physics-informed approaches with operator learning architectures, such as neural operators, holds promise for producing generalisable solvers that retain physical consistency across diverse terrain and flow configurations. Building on these directions, future work will focus on making the framework more robust, more generalisable and more computationally eficient, broadening its applicability to a wider range of avalanche scenarios.

## Acknowledgments

P.S.S. gratefully acknowledges the Ministry of Education (MoE), Government of India, for funding his MTech (Research) studies. The authors thank the Data Science (DS) Laboratory at the Indian Institute of Technology Mandi for providing the computational resources used in this work. The authors would also like to acknowledge the developers of the open-source <sub>TITAN2D</sub> code.

## Computer Code Availability

Hardware requirements: GPU. Programming language: Python. The source code is available for download at <sub>https:</sub>//<sub>github.com</sub>/<sub>Prasik3182</sub>/<sub>PINNs\_Avalanche.git</sub>. The TITAN2D reference data used for benchmarking is archived on Zenodo at <sub>https:</sub>//<sub>doi.org</sub>/<sub>10.5281</sub>/<sub>zenodo.</sub> <sub>21393644</sub>.

## References

S. B. Savage and K. Hutter. The motion of a finite mass of granular material down a rough incline. Journal of Fluid Mechanics, 199:177–215, 1989. doi: 10.1017/S0022112089000340.

J. M. N. T. Gray and A. N. Edwards. A depth-averaged µ(I)-rheology for shallow granular freesurface flows. Journal of Fluid Mechanics, 755:503–534, 2014. doi: 10.1017/jfm.2014.450.

Y.-C. Tai, S. Noelle, J. M. N. T. Gray, and K. Hutter. Shock-capturing and front-tracking methods for granular avalanches. Journal of Computational Physics, 175(1):269–301, 2002. doi: 10.1006/ jcph.2001.6946.

Anne Mangeney, J-P Vilotte, Marie-Odile Bristeau, Benoît Perthame, François Bouchut, Chiara Simeoni, and Sudhakar Yerneni. Numerical modeling of avalanches based on Saint Venant equations using a kinetic scheme. Journal of Geophysical Research: Solid Earth, 108(B11), 2003. doi: 10.1029/2002JB002024.

E. Bruce Pitman, Crina C. Nichita, Abani Patra, Andrew C. Bauer, Michael F. Sheridan, and Marcus Bursik. Computing granular avalanches and landslides. Physics of Fluids, 15(12):3638– 3646, 2003. doi: 10.1063/1.1614253.

Abani K Patra, Andrew C Bauer, C. C. Nichita, E. Bruce Pitman, Michael F. Sheridan, M. Bursik, Byron Rupp, A. Webber, A. J. Stinton, L. M. Namikawa, and C. S. Renschler. Parallel

adaptive numerical simulation of dry avalanches over natural terrain. Journal of Volcanology and Geothermal Research, 139(1–2):1–21, 2005. doi: 10.1016/j.jvolgeores.2004.06.014.

Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partia diferential equations. Journal of Computational Physics, 378:686–707, 2019. doi: 10.1016/j.jcp. 2018.10.045.

George Em Karniadakis, Ioannis G Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 3(6):422–440, 2021. doi: 10.1038/s42254-021-00314-5.

Zhiping Mao, Ameya D Jagtap, and George Em Karniadakis. Physics-informed neural networks for high-speed flows. Computer Methods in Applied Mechanics and Engineering, 360:112789, 2020. doi: 10.1016/j.cma.2019.112789.

Ameya D. Jagtap, Ehsan Kharazmi, and George Em Karniadakis. Conservative physics-informed neural networks on discrete domains for conservation laws: Applications to forward and inverse problems. Computer Methods in Applied Mechanics and Engineering, 365:113028, 2020. doi: 10.1016/j.cma.2020.113028.

Sifan Wang, Yujun Teng, and Paris Perdikaris. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing, 43(5):A3055– A3081, 2021. doi: 10.1137/20M1318043.

Sifan Wang, Xinling Yu, and Paris Perdikaris. When and why PINNs fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449:110768, 2022. doi: 10.1016/j.jcp.2021. 110768.

Ameya D. Jagtap and George Em Karniadakis. Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear partial diferential equations. Communications in Computational Physics, 28(5):2002– 2041, 2020. doi: 10.4208/cicp.OA-2020-0164.

Alex Bihlo and Roman O. Popovych. Physics-informed neural networks for the shallow-water equations on the sphere. Journal of Computational Physics, 456:111024, 2022. ISSN 0021-9991. doi: 10.1016/j.jcp.2022.111024.

Susanna Dazzi. Physics-informed neural networks for the augmented system of shallow water equations with topography. Water Resources Research, 60(10):e2023WR036589, 2024. doi: 10. 1029/2023WR036589.

Xin Qi, Gustavo A. M. de Almeida, and Sergio Maldonado. Physics-informed neural networks for solving flow problems modeled by the 2D shallow water equations without labeled data. Journal of Hydrology, 636:131263, 2024. doi: 10.1016/j.jhydrol.2024.131263.

Dapeng Feng, Zeli Tan, and Qiang He. Physics-informed neural networks of the Saint-Venant equations for downscaling a large-scale river model. Water Resources Research, 59(2):e2022WR033168, 2023. doi: 10.1029/2022WR033168.

Zeda Yin, Jimeng Shi, Linlong Bian, William H. Campbell, Sumit R. Zanje, Beichao Hu, and Arturo S. Leon. Physics-informed neural network approach for solving the one-dimensional unsteady shallow-water equations in riverine systems. Journal of Hydraulic Engineering, 151(1): 04024060, 2025. doi: 10.1061/JHEND8.HYENG-13572.

Yongfu Tian, Shan Ding, Lida Huang, Guofeng Su, and Jianguo Chen. Physics-informed neural networks for solving the two-dimensional shallow water equations with terrain topography and rainfall source terms. Water Resources Research, 61:e2025WR040052, 2025. doi: 10.1029/2025WR040052.

Barbara Baldoni, Mickaël Delcey, Yoann Cheny, Adrien Gans, Mathieu Jenny, and Sébastien Kiesgen de Richter. Rheological parameter identification in granular materials using physicsinformed neural networks. Powder Technology, 475:122342, 2026. ISSN 0032-5910. doi: 10.1016/j.powtec.2026.122342.

P-H Pai, Luca Sarno, Y-C Tai, and H-C Kan. An adaptive refinement neural particle method for granular flows. Physics of Fluids, 37(8):083331, 2025. doi: 10.1063/5.0276235.

Fukashi Maeno, Andrew J Hogg, R Stephen J Sparks, and Gary P Matson. Unconfined slumping of a granular mass on a slope. Physics of Fluids, 25(2):023302, 2013. doi: 10.1063/1.4792707.

A. Mangeney, F. Bouchut, J. P. Vilotte, E. Lajeunesse, A. Aubertin, and M. Pirulli. On the use of Saint Venant equations to simulate the spreading of a granular mass. Journal of Geophysical Research: Solid Earth, 110(B9):B09103, 2005. doi: 10.1029/2004JB003161.

A. Mangeney, F. Bouchut, N. Thomas, J. P. Vilotte, and M. O. Bristeau. Numerical modelling of self-channelling granular flows and of their levee-channel deposits. Journal of Geophysical Research: Earth Surface, 112(F2):F02017, 2007. doi: 10.1029/2006JF000469.

Vishal Sharma and Gaurav Bhutani. Computational modeling of river-blocking snow avalanches: a case study in the Indian Himalayas. Natural Hazards, 122:355, 2026. doi: 10.1007/ s11069-026-08106-9.

Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jefrey Mark Siskind. Automatic diferentiation in machine learning: a survey. Journal of Machine Learning Research, 18(153):1–43, 2018. URL <sub>http:</sub>//<sub>jmlr.org</sub>/<sub>papers</sub>/<sub>v18</sub>/<sub>17-468.html</sub>.

## A Hyperparameter Sensitivity: Decoupled and Coupled Case

The hyperparameter sensitivity results for the decoupled one-dimensional tests are presented in Figure 22. Each hyperparameter was varied independently while all others were held fixed; error bars show mean standard deviation. The selected values, summarised in Table 1, are consistent with the trends observed here.

The height RMSE was insensitive to all five hyperparameters; the velocity RMSE accounted for most of the variation, reflecting the greater dificulty of the momentum equation established in the main text. The learning rate $\alpha = 1 0 ^ { - 3 }$ sat in the flat region before the sharp velocity error increase at $\alpha = 1 0 ^ { - 1 }$ . Depth 5 fell within the flat region of the depth sweep, beyond which no further gain in accuracy was observed, and training cost increased linearly. Collocation density had no systematic efect on accuracy, confirming that the chosen values of 2000 and 10000 for both cases were not limiting factors.

![](images/810e16d8b3b5bfec1e248a5852fca2fe8c2a9c0b94daf8278ef711f9aa897853.jpg)  
Figure 22: Hyperparameter sensitivity of the decoupled one-dimensional PINN: prediction RMSE (left) and wall-clock training time (right). Rows $\mathrm { ( a ) - ( e ) }$ : learning rate $\alpha ,$ width $N _ { w } ,$ depth $N _ { d } ,$ collocation points $N _ { \mathrm { c o l } }$ , and epochs $N _ { e }$ . Error bars show mean standard deviation over three independent runs. Height (blue circles); velocity (orange squares).

The sensitivity study for the coupled formulation is presented in Figure 23. ${ \mathrm { R M S E } _ { h } }$ remained nearly flat across all settings while the velocity error accounted for almost all the variation. Training cost scaled linearly with network size and collocation density. Based on these sweeps, a five-layer network with 64 neurons per layer was selected, as summarised in Table 1.

![](images/072afea0c2a0c9883cc2e41874dfa0f9009dada1788bb170ae45d7b5f85b3211.jpg)

![](images/269f772ae87bc5e23ea6fccde5b33ab6bfbade8ae655c2d4c022df68919630f9.jpg)

![](images/df89907a3f0eb15d6e2df32589f5d976cb90231b4dea08d82cbab14f197a709d.jpg)

![](images/19dbf108bd424c0d6688b8ee4631d796f1bb9246a3d7fd6b75f4c7e2de81266d.jpg)

![](images/ca5b1a2bbff201d5ef82986f5566b99c60572bf4824557c9f7cc8b7283b70245.jpg)

![](images/aeeb977109a1b3dde18c707f10b2fc32331ac3e23d0b2ea2b2ce0c076ba15261.jpg)

![](images/ce407f7fe64ef220e42e2f16b510d0d0ce873f9ff55711eaa388b8ca98677381.jpg)

![](images/31301018358f2f3406111e8e281daefbb95d0372f096bec7c76b5a22c291a2cf.jpg)

![](images/cbc75cadad60e9fc4f05a08bebb655a6bf571b598203af98f24e1c0af0ed2da5.jpg)

![](images/add8019ba911663d06f624e011801c58d6ceef5dbbe72ab01e6fb79c13f393b3.jpg)  
Figure 23: Hyperparameter sensitivity of the coupled one-dimensional PINN: prediction accuracy (left) and wall-clock training time (right). Rows (a)–(e): learning rate $\alpha ,$ width $N _ { w } ,$ depth $N _ { d } .$ collocation points $N _ { c } ,$ , and epochs $N _ { e }$

## B Pseudo-2D Strip Flow: Diagnosing Dry-Region Dominance

Section 5.2 identified two structural causes of the trivial-solution collapse in the two-dimensional setting: the mathematical homogeneity of the Savage–Hutter system, and the dominance of the dry region in the computational domain. The cylindrical initial condition used in Cases C1–C4 occupies approximately 0.82 % of the $1 . 2 \mathrm { m } \times 0 . 8$ m domain, leaving approximately 99.18 % of the spatial domain initially dry. Although the wetted footprint increases as the granular mass spreads, the dry region remains substantially larger than the active flow region. To isolate the contribution of this dry-region dominance, the cylindrical pile was replaced by a full-width parabolic ridge uniform in the cross-slope $( y )$ direction. This configuration is referred to as the pseudo-2D case: the initial condition and the resulting flow are uniform in $y ,$ so the problem is efectively one-dimensional in its dynamics while remaining formally two-dimensional in its spatial inputs and governing equations. The strip occupies a substantially larger fraction of the domain, directly increasing the proportion of collocation points that sample the active flow region.

## B.1 Problem configuration

The initial condition was a parabolic ridge uniform in $y ,$ centred at $x = 0 . 4 0$ m with peak height $H _ { 0 } = 0 . 0 8 5 \mathrm { m }$ The granular material properties matched those of Cases C1–C4: bulk density $\rho = 1 4 8 1 \mathrm { k g } \mathrm { m } ^ { - 3 }$ , internal friction angle $\phi = 2 4 ^ { \circ }$ , and bed friction angle $\delta = 2 8 ^ { \circ }$ . The same five layer, 64-neuron network architecture, loss formulation, and training configuration described in the main text were used, with one deliberate omission: no sparse data augmentation was included. If dry-region dominance were the sole cause of the trivial collapse, the wider wetted footprint should allow the physics loss alone to sustain a physical solution. A volume ratio, defined as the ratio of the predicted total volume at time t to the initial volume $V _ { 0 }$ , was tracked throughout training as the primary diagnostic of mass retention. The experiment was run at slope angles of $1 5 ^ { \circ }$ , 20◦, and $3 0 ^ { \circ }$ , and the simulation window ran to $t _ { \mathrm { m a x } } = 0 . 5 5 \mathrm { s }$ in each case.

## B.2 Results

At all three slope angles, training exhibited a brief collapse toward zero near epochs 12000–14000, followed by recovery to a physical solution. This recovery did not occur in the equivalent cylindricalpile experiments without data augmentation, confirming that the wider wetted footprint provided suficient gradient signal to escape the trivial attractor. The final volume ratios are collected in Table 7: higher slope angles produced higher retained volume, with the $3 0 ^ { \circ }$ case retaining 96.7 % of the initial volume while the $1 5 ^ { \circ }$ case settled near 0.675 and was still slowly increasing at the end of training. The centreline profiles show the flow spreading downslope with increasing runout at steeper angles, and the cross-slope profiles remained uniform in y throughout, which is physically consistent with a strip initial condition in the absence of a cross-slope gravity component. Centreline profiles and depth contours for each slope angle are presented in Sections B.2.1–B.2.3.

Table 7: Final volume ratios for the pseudo-2D experiment. Values are reported at the end of training and averaged over the final 10000 epochs.
<table><tr><td>ζ</td><td>Final volume ratio Mean (last 10000 epochs)</td><td></td></tr><tr><td> $1 5 ^ { \circ }$ </td><td>0.675</td><td>0.645</td></tr><tr><td> $2 0 ^ { \circ }$ </td><td>0.745</td><td>0.723</td></tr><tr><td> $3 0 ^ { \circ }$ </td><td>0.967</td><td>0.969</td></tr></table>

These results indicate that the trivial-solution collapse is driven by the dominance of the dry region within the computational domain: because the physical deposit occupies only a small fraction of the domain, the majority of collocation points lie outside the wetted footprint and continuously pull the PDE loss toward $h = u = v = 0$ , overwhelming the comparatively weak gradient signal from the active flow region. The pseudo-2D experiment tested this mechanism directly by enlarging the wetted footprint through a full-width strip geometry. With a larger fraction of the domain occupied by flow, a greater proportion of collocation points contributed a non-trivial gradient signal, and recovery from the trivial solution became possible. This confirms that the imbalance between wetted and dry collocation points is a primary driver of the collapse, and motivates the data-augmentation strategy adopted in the main text, which supplies a gradient signal independent of the wetted-area fraction.

## B.2.1 $\zeta = 1 5 ^ { \circ }$

At 15◦, the flow decelerates throughout the simulation window under the net frictional resistance. The centreline peak advances progressively downslope and flattens as the deposit spreads, while the wavefront position and peak height remain in close agreement with <sub>TITAN2D</sub> at every snapshot. The depth contours show the expected elongation of the deposit in the downslope direction.

![](images/1008156fd07163035e6e1a7e9c5748de915bee447b4c1c838e85022ef18109b9.jpg)  
(a) t = 0.00 s

![](images/b39af12faf90f740ee207c688df3e8849d4d9db2351ea8382574e804c7ed780d.jpg)

![](images/7c4e2c392823269a36d3b61b713e0745beebd285e97d3e8aa09a0347f086e360.jpg)  
(c) t = 0.20 s

![](images/825dbcb677422bb9ac3cac40878ed3194643a8093fff77a6ee43df56a49001e0.jpg)  
(d) $t = 0 . 3 0 \mathrm { s }$

(b) t = 0.10 s t = 0.40 s  
![](images/0746ae67897083873599fdf21fdc567aa16707d77dcf04852497462a657dc600.jpg)  
(e) t = 0.40 s

![](images/7a3e1d0965e4a4fa73108288967abe4204910f46d24c66a65444d1f88f6c60c5.jpg)  
(f) $t = 0 . 5 5 \mathrm { s }$  
Figure 24: Pseudo-2D case, $\zeta = 1 5 ^ { \circ }$ : centreline height profiles at six time instants. PINN (solid blue) and <sub>TITAN2D</sub> (dashed red).

![](images/8b2f868f553f407cb18d60a1a35cc4bbc836484f90f11a545fd2b1a722dbcfac.jpg)  
(a) t = 0.00 s

![](images/b38db6d64687ce4fcf9dc44135f7a8e97369ae4fb6d8f2bdfa608579bbbbdbe9.jpg)  
(b) $t = 0 . 1 0 \mathrm { s }$

![](images/6d2a894523682a388a21a0e60956f091436a0dff807d7914104a355510855268.jpg)  
(c) t = 0.20 s

![](images/cdc605465402c2ad515fcf36a243e31de90a58a2f6b2e0a3760c6292f2e90a8f.jpg)  
(d) $t = 0 . 3 0 \mathrm { s }$

![](images/35b973e0f047e08d65e8301a2231e7fb875293aad1ea2bc0ad263ecfd7bef5bf.jpg)  
(e) t = 0.40 s

![](images/01a3e97246b6dceb0212ae6b72fd1bcea0257a0a129082ffb2da522e49fdafd8.jpg)  
(f) t = 0.55 s  
Figure 25: Pseudo-2D case, $\zeta = 1 5 ^ { \circ }$ : top-view depth contours at six time instants. Contour levels are identical across all panels.

## B.2.2 $\zeta = 2 0 ^ { \circ }$

At $2 0 ^ { \circ }$ , the reduced frictional resistance produced a faster collapse and greater downslope runout within the same simulation window. The centreline peak decayed more rapidly, and the downstream tail extended further than at $1 5 ^ { \circ }$ , while the wavefront position remained in close agreement with <sub>TITAN2D</sub> throughout. The depth contours show a more elongated deposit in the downslope direction relative to the $1 5 ^ { \circ }$ case.

![](images/2cbd1279bb51331fbc6c09d4c06522a07547e7532692b29fa3a9b7821d57e0d3.jpg)  
(a) t = 0.00 s

![](images/bc22d1ac9d80950e2e7d221ab7bdf2eb3efbeb02c315b03ed9d3d40280d26a84.jpg)  
(b) t = 0.10 s

![](images/d6c4ed102cbdf4cb861aa7a3ae3ccb5c8333e385f4a47f8a577a98d1d1d1725c.jpg)  
(c) $t = 0 . 2 0 \mathrm { s }$

![](images/2a6b87958f02dad504b9fde53e8e008fac1831b1f038f7d6c1ce7fd2d45df653.jpg)  
(d) t = 0.30 s

![](images/a0cee1692fd7be6461e650b45f48f874492c23d6389d8b71fb3b31736e79f8f0.jpg)  
(e) t = 0.40 s

![](images/2ef5e0f473b769f187eb7b6a92bcbfbd7800214abb65d223e6767a0f4ba4c438.jpg)  
(f) t = 0.55 s

Figure 26: Pseudo-2D case, $\zeta = 2 0 ^ { \circ }$ : centreline height profiles at six time instants. PINN (solid blue) and <sub>TITAN2D</sub> (dashed red).  
![](images/81594753ae4ce7c8a45af005f150660b0be835db27fe5e59d7464e01e5374b83.jpg)

![](images/66fe3afac9d6d82c6f60f3829003c9dc9e1bee3772b4a1c4104fc36c98e72ccd.jpg)  
(a) t = 0.00 s

![](images/0b15ed95781135a86c5fdf95d19d8ed46a41f1a9ad444703a2627dddce03a443.jpg)

![](images/502ad2acf9ef208a6b46f0b11a300cbcb689d6113b7643cab6a71827dc3b4b86.jpg)  
(d) t = 0.30 s

(b) t = 0.10 s  
![](images/b0ba9435dbe5323aed631005d18be776c969e09645a94b24b8f468c86552da32.jpg)  
(e) t = 0.40 s

(c) t = 0.20 s  
![](images/eedff4455f0b6c81213b3aea7f1b0455fcd09949dd34e9ab7c3499d89e9493d6.jpg)  
(f) t = 0.55 s  
Figure 27: Pseudo-2D case, $\zeta = 2 0 ^ { \circ }$ : top-view depth contours at six time instants. Contour levels are identical across all panels.

## B.2.3 $\zeta = 3 0 ^ { \circ }$

At 30◦, the slope exceeds the bed friction angle, so the net gravitational driving force is positive, and the flow accelerates throughout the simulation window. The wavefront advanced rapidly, and the downstream tail extended well beyond the initial pile location by $t = 0 . 5 5 \mathrm { s }$ , with only modest decay in peak height. The depth contours show the deposit elongated downslope, with the major axis considerably longer than in either of the decelerating cases, and the PINN remained in close agreement with <sub>TITAN2D</sub> throughout.

![](images/2fa3ad44c2b030a02f047884934969603bfb3e43702bc5725c09ac15749414ca.jpg)  
(a) t = 0.00 s

![](images/c649b84baf1d53023c8a14fac44eef1addc2a99163e3548115be2dd27eb49d41.jpg)  
(b) t = 0.10 s

![](images/c6710331627aef8287423e682196ab04b69b8e7b8dae90a0a32428eb59092236.jpg)  
(c) t = 0.20 s

![](images/f07c017896cd9926580260ddebdcf17a679df03d3fbd20777426094b4ec15d58.jpg)  
(d) $t = 0 . 3 0 \mathrm { s }$

![](images/c52a56a61eabd8df077b12cdbf36b1452ca7861597f79b12e5d850196e961aab.jpg)  
(e) t = 0.40 s

![](images/f310095371329bafc6d828dc4b9ef4965086ee1364c96405aafccfdc5acb3f26.jpg)  
(f) t = 0.55 s  
Figure 28: Pseudo-2D case, $\zeta = 3 0 ^ { \circ } ;$ centreline height profiles at six time instants. PINN (solid blue) and <sub>TITAN2D</sub> (dashed red).

![](images/a30c1ea4f10012e059e46a2b83d8abc197e5f610b74422c253593aeaacac0975.jpg)  
(a) t = 0.00 s

![](images/b9daff6d5d0237a6e3547d157fedf47e1250b14e30fd561985debfe44d78a205.jpg)  
(b) t = 0.10 s

![](images/16e8484abedca625c9172222d3f31e741b30f5e8d8dab3c2d12a867c1b146bdd.jpg)  
(c) t = 0.20 s

![](images/0281e81f77018228913e5dca59b6b8991c5a0ffd30ecef1e461e551875611ccf.jpg)  
(d) t = 0.30 s

![](images/ca55eaa7e2955c93e91d3c4d6cdfe237b9164a0221781abd3bebd848019f9d33.jpg)  
(e) t = 0.40 s

![](images/966e39a3319c435572830e562aaaf2f4201aba8eb275108a91199259590ffce9.jpg)  
(f) t = 0.55 s  
Figure 29: Pseudo-2D case, $\zeta = 3 0 ^ { \circ }$ : top-view depth contours at six time instants. Contour levels are identical across all panels.

## C Extended Results for Secondary Cases C1–C3

Time-resolved profile comparisons and integral diagnostics for Cases C1, C2 and C3 are presented here, complementing the final-deposit profiles in the main text. For each case, the centreline and cross-slope profiles at successive snapshots are followed by integral flow diagnostics and the IoU evolution, mirroring the analysis for C4 in the main text.

## C.1 Case C1: 1 kg, 10◦

At 10◦, the deposit peak was recovered accurately at early times, and agreement with <sub>TITAN2D</sub> was closest at $t = 0 . 1 8$ and 0.35 s. By $t = 0 . 7 0 \mathrm { s }$ , the PINN peak was slightly taller and shifted downslope relative to <sub>TITAN2D</sub>. The centreline and cross-slope profiles are shown in Figure 30.

![](images/14d2384c484bed717be6bea9d65b5c3112d763a0e15885bbd4bd6e412e29c935.jpg)  
Downslope position x (m)

![](images/b7004ed94975e9e5a3972dcb382a0f00a850b396e4a6e0f3780d5004f5865198.jpg)

![](images/85bfd7abcd1f0959b0adef498a8f0ddc14712fe8e63836a9effccdf2863856db.jpg)

![](images/acff3a1c535f5507cd725fcedb9c1147ae0f7240d87060f5cae6856ecf67721e.jpg)

![](images/c01c5fb5214655f491602ce590fd60eca3ab14601bf27661f14e0be9bcf5713f.jpg)

![](images/1c1346d6bb0ef32cc1f6f402990ac98cf1c6d2ba903d21b378f7c14a654ce519.jpg)

![](images/c461f6a0451594e3c714f55545c20eb79366942730bdb1d566bf2d71bfdb387d.jpg)

![](images/7e475c017c93c1738cb0c07a0d363ac19ad8989440d8633fc07579044397491b.jpg)  
Figure 30: Case C1 (1 kg, 10◦): centreline (left) and cross-slope (right) height profiles at $t = 0 . 0 0$ 0.18, 0.35, and 0.70 s. PINN (solid blue) and <sub>TITAN2D</sub> (dashed red).

The integral diagnostics are shown in Figure 31. The maximum height decayed monotonically from $H _ { 0 } = 8 5$ mm, with the PINN closely tracking <sub>TITAN2D</sub> throughout. The mean velocity reached a lower peak than in C4, reflecting the weaker gravitational drive at 10◦.

![](images/254a8c68c5c40d7f3e7302acc5c1bd48dbfdc9c8d260c1c223a37497bec0dcfc.jpg)  
Time t (s)

![](images/0e171dcfecc9256d7c0c5f5d6cfd14269354630e3ffc0127d11967fe9474c960.jpg)

![](images/db05f980711f1d23be8c81c9de99ee6304c1abbe256da5364f01cfde7f4c8e1d.jpg)  
Runout distance (m)

![](images/7fc235d745c1ec8e8675c3988735666bad12b3eaf901d4a4b3f064ffb22be7d6.jpg)  
Runout distance (m)  
Figure 31: Integral flow diagnostics for Case C1 (1 kg, 10◦): $h _ { \mathrm { m a x } }$ and V<sup>¯</sup> versus time (top row); centroid displacement and $\bar { V }$ versus runout distance (bottom row). <sub>TITAN2D</sub> (solid red); PINN (dashed blue).

The IoU started near its maximum, as both solutions shared the same initial cylindrical footprint, then decreased through the spreading phase as the wavefronts diverged slightly, stabilising toward a time-mean of 70.9 % (Figure 32).

![](images/bdf370bc1828583afc4fef6e86177d0b71026c9eef1eb53c4e769f3fe883e020.jpg)  
Figure 32: Time-resolved IoU for Case C1 (1 kg, 10◦). Time-mean value: 70.9 %.

## C.2 Case C2: 1 kg, 15◦

Case C2 shared the pile mass and geometry of Case C1; only the slope angle increased. The steeper incline drove a faster collapse and a more elongated final deposit, and the PINN reproduced both

![](images/8876ce87fd553ebcc9e4b74433796c5214cccef7ab853e8f0324c34090667e41.jpg)

![](images/ee571ba8a8e63ce1976b44aa848e7c0e1b35c90a9c00f7a2054a37ff3c942643.jpg)

![](images/e8494b1afce71115c2b12c82e551c441c4b3b21425aa0b1cfd08c58e6f00aba1.jpg)

![](images/30f10be4bdf8b646fb2bbe665035db098c2564fe718dd2b4354a08064febcc27.jpg)

![](images/91c8443133ada22281cbc820e9401f897e292d25c48c0f883a75a5e4629137f7.jpg)

![](images/2bf40c474cc7a0d5492def601924f5fafec70caf417053ee7bdd6d70585edfa0.jpg)

![](images/e8ea4fd91731e0d01a6fa6ff987a7d66d4b27fb95f83cb253c31a2867f53cda6.jpg)

![](images/1adcd1ea620a10d76e1584cde9969130a10a6bc89e7d97f85c83b27397f1feb1.jpg)  
Figure 33: Case C2 (1 kg, 15◦): centreline (left) and cross-slope (right) height profiles at $t = 0 . 0 0$ 0.18, 0.35, and 0.70 s. PINN (solid blue) and <sub>TITAN2D</sub> (dashed red).

The integral diagnostics are shown in Figure 34. The maximum height decayed monotonically from $H _ { 0 } = 8 5 \mathrm { { m m } }$ , with the PINN closely tracking <sub>TITAN2D</sub> throughout, and the mean velocity

reached a higher peak than in C1, consistent with the stronger gravitational drive at 15◦.

![](images/82ac8d09ec763cf16ee289731b7e4fff004daed6b7601438cb2b7ee94eaff00d.jpg)

![](images/3b746c93217c630640f7ea4365a25a445ffb0f91aeb659e96959c2af15406a9d.jpg)

![](images/a6dd971e9e98c674b69b6fe3f91647589ee17b9274bce405051f3129de527568.jpg)

![](images/543edba8c0d02697b4e47d7580ca6bd06319886bb8eda5bcb2ab505e9bb180e7.jpg)  
Figure 34: Integral flow diagnostics for Case C2 (1 kg, 15◦). Layout as in Figure 31.

The time-mean IoU for Case C2 was 80.7 %, the highest value across all four cases (Figure 35).

![](images/a95221d796d75704002753863aecffaee3b7dc0c634c5d684e01b82650ed7d43.jpg)  
Figure 35: Time-resolved IoU for Case C2 (1 kg, 15◦). Time-mean value: 80.7 %.

## C.3 Case C3: 2.5 kg, 10◦

Case C3 paired the higher pile mass of C4 with the lower slope angle of Case C1. The larger initial volume produced steeper depth gradients during the early collapse and a wider final deposit than either of the 1 kg cases, and agreement with <sub>TITAN2D</sub> was closest at the final snapshot once the deposit had thinned and lateral gradients had relaxed (Figure 36).

![](images/6aaf6142b3d2f41f9c1c314aae462fcea4d62c73741df68fb5fc4d227a89c3ef.jpg)  
Downslope position x (m)

![](images/29ae220a0db9620fd43660d8b70baa717b8ad8002996f935501e96d7987cf0e7.jpg)  
Cross-slope position y (m)

![](images/7360d92ce21a2f5cc68e0c72232c79e8a221aa930e179b8a7fb9ff0be938f97b.jpg)

![](images/1cc929b13c8336192192ad027c71c9cfe3bc5f8badc46cffeb43ca7ea36d2be8.jpg)

![](images/77ada0569ee76e12578e1dfc890fa14a24ef29e4533f51428713c3755e028d82.jpg)

![](images/91b4744a18d81b6e03994601d6e4b18d724257196558ae2ba12aee7682d011e0.jpg)

![](images/dac19ad6f5b4aeb622a953f1d55053d18999030b14d21ab497f95af7b51e1fee.jpg)  
Downslope position x (m)

![](images/8ca4d3006cbb2c44e737adcb8603c95edc6e2f9a393a2783aff16e489e425ea5.jpg)  
Cross-slope position y (m)  
Figure 36: Case C3 (2.5 kg, 10◦): centreline (left) and cross-slope (right) height profiles at t = 0.00, 0.24, 0.48, and 0.95 s. PINN (solid blue) and <sub>TITAN2D</sub> (dashed red).

The integral diagnostics are shown in Figure 37. The maximum height decayed sharply within the first 0.1 s, with the PINN settling slightly above the <sub>TITAN2D</sub> reference at later times. The mean velocity reached a lower, later peak in the PINN prediction than in <sub>TITAN2D</sub>, and the velocity– distance curve shows that the PINN sustains motion over a longer runout distance than <sub>TITAN2D</sub>.

![](images/06af4050c560564ef8096f7ed3553b2cfe1471fe8ee4c524ba79b209db989673.jpg)

![](images/8510b2adac9bfe4265e8245442ad3c199e2ea2949897eaa78af19368aac908cf.jpg)

![](images/52c1ceffb9be0b8f5460f176e2d5478cab11e3c984bd4638c01b2bf70ee812aa.jpg)

![](images/73e472b290200809afc4a9816891a6302671ba6903bccbc348879efa75ad7e99.jpg)  
Figure 37: Integral flow diagnostics for Case C3 (2.5 kg, 10◦). Layout as in Figure 31.

The time-resolved IoU for Case C3 is shown in Figure 38. The metric started near its maximum, dipped to a minimum during the early spreading phase, and then recovered to stabilise near the time-mean of 73.1 % for the remainder of the simulation.

![](images/b81c376dd8ecda98a57f59758f332f681981ccc5114ddeaa8e4129ad5cc0bcd1.jpg)  
Figure 38: Time-resolved IoU for Case C3 (2.5 kg, 10◦). Time-mean value: 73.1 %.

## D Repeatability Analysis

The results for Cases C1–C4 in the main text each correspond to a single training run. To assess how sensitive these results are to the choice of random initialisation, three independent training runs were performed for C4, each initiated from a diferent random seed. All other aspects of the

setup, including the architecture, hyperparameters, collocation strategy, and observation set, were identical across runs, as described in the main text.

## D.1 Profile variability across runs

Figure 39 overlays the centreline and cross-slope height profiles from all three runs at four time instants alongside the <sub>TITAN2D</sub> reference. The inter-run spread was small at every snapshot and was comparable in magnitude to the single-run PINN–<sub>TITAN2D</sub>. The three curves were nearly indistinguishable at t = 0 s and remained closely clustered through the final deposit.

![](images/93d983015f4e1a377fd63a80b39b3eceb0380366af22ad504b6e401dea207acd.jpg)  
Downslope position x (m)

![](images/52fec3a5bed652972786cc04ac0739509f4d1ab46c1ac45fb2ce1e2c277fd186.jpg)

![](images/43e1287148c0c64bbde96b07577d695323616a94a81873cf8275d6b01f6dd254.jpg)

![](images/d31c10e79fd2e8274dedc37ff2a681e7c7b2450ce7b0f5881d927fa6ee81cf14.jpg)

![](images/d730446fddc3ab2411e32d78b7535ce4f0db5cd5f1a0c0f87651f497323f1657.jpg)

![](images/96e612fbfca96d204ce67d4fe067104e98e22cc56b646ce6e9c8b52f0388363a.jpg)

![](images/51364f230f0bdece913d7b5041a45f689c6473c2c7162207945895464b356608.jpg)

![](images/1aff1eafbc10b37546c807bbe11632f46170da3ece572da7fc81a8ca9a2e28f0.jpg)  
Figure 39: Repeatability study for C4 (2.5 kg, 15◦): centerline (left) and cross-slope (right) height profiles at $t = 0 . 0 0 , 0 . 2 3 , 0 . 4 7 .$ , and 0.95 s. Each solid curve is one of three independent PINN runs. <sub>TITAN2D</sub> reference (dashed red).

## D.2 Final deposit comparison

The inter-run spread at the final deposit $( t = 0 . 9 5 \mathrm { s } )$ was substantially smaller than the gap between the PINN band and <sub>TITAN2D</sub> in both the centreline and cross-slope directions (Figure 40), confirming that the discrepancies reported in the main text reflect genuine model behaviour rather than sensitivity to random initialisation.

![](images/44dca486e57140b2fd63e58342ec6cc02aa93d4b531742d0db031793ab37fd06.jpg)  
(a) Centreline (y = 0).

![](images/15ba5b8318c133d8a9023d601588470540a8639f10f4542dfd7bef14388357e7.jpg)  
(b) Cross-slope (x = 0).  
Figure 40: Final deposit profiles for C4 across three independent training runs (solid lines). <sub>TITAN2D</sub> (dashed red) and Maeno et al. (2013) experimental data (open circles) are shown for reference.

## D.3 Error metrics across runs

The time-resolved RMSE and IoU for all three runs are shown in Figure 41. Both metrics followed the same trajectory regardless of seed: RMSE peaked during the early collapse phase and decayed as the deposit thinned. IoU was more variable across runs during the collapse and spreading phase (roughly $t < 0 . 5 \mathrm { s } )$ , with each run showing a dip of difering depth and timing, before converging to a similar range by the final snapshot. The RMSE at the final snapshot lay within a narrow band across all three runs, and the time-mean IoU varied by only a few percentage points, confirming that a single training run is representative of the framework’s predictive capability for this problem.

![](images/ba45876844a204ac1ccd66ff0c9ed717b4f64cb9b3d593bdf7f389604c5101cd.jpg)  
(a) Height RMSE versus time.

![](images/68eb8404d2257eab08fe7862e9e5c837b07062aa0fbd7689eedd215dc1ee5120.jpg)  
(b) IoU versus time.  
Figure 41: Time-resolved error metrics across three independent training runs for C4. Each curve corresponds to one run; the bandwidth quantifies run-to-run variability.

## D.4 Training data point coordinates

Table 8 lists the coordinates of the ten training data points used for Case C4, selected to sample the upstream face, deposit peak, and downstream tail along the centerline, as well as the full lateral extent of the deposit along the cross-slope profile.

Table 8: Coordinates of the ten training data points used for C4. All values are in centimetres.
<table><tr><td>Profile</td><td>Point x (cm)</td><td>y (cm)</td></tr><tr><td rowspan="5">Centerline (y = 0)</td><td>P1</td><td>-10 0</td></tr><tr><td>P2</td><td>-5 0</td></tr><tr><td>P3</td><td>5 0</td></tr><tr><td>P4</td><td>10 0</td></tr><tr><td>P5</td><td>25 0</td></tr><tr><td rowspan="5">Cross-slope (x = 0)</td><td>P6</td><td>0</td><td>-17.0</td></tr><tr><td>P7</td><td>0</td><td>-7.0</td></tr><tr><td>P8</td><td>0</td><td>-1.0</td></tr><tr><td>P9</td><td>0</td><td>9.0</td></tr><tr><td>P10</td><td>0</td><td>13.0</td></tr></table>