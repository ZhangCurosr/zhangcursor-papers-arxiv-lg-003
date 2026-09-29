# On the Numerical Reliability of Differentiable Physics-Based Optimization for Robotic Material Manipulation

Xintong Yang<sup>1</sup>, Minglun Wei<sup>1</sup>, Yu-Kun Lai<sup>2</sup>, and Ze Ji<sup>3,\*</sup>

<sup>1</sup>School of Engineering, Cardiff University, Cardiff, United Kingdom

<sup>2</sup>School of Computer Science and Informatics, Cardiff University, Cardiff, United Kingdom

<sup>3</sup>College of Mechanical and Electrical Engineering, Hohai University, China

Corresponding author: Ze Ji (z.ji@hhu.edu.cn)

Abstract—Differentiable physics is increasingly used in robotic material manipulation for system identification, trajectory or skill optimization, demonstration generation, and robot or end-effector design. These applications depend on gradients propagated through long, contact-rich simulation rollouts. We study the numerical reliability of those gradients using two Material Point Method (MPM) system-identification benchmarks derived from elastoplastic and granular manipulation. The benchmarks provide controlled cases for three effects that also arise in broader differentiable physics-based optimization. GPU many-toone sums whose order depends on thread scheduling changed long-horizon gradients and reversed the sign of one parameter gradient relative to a deterministic reference. Finite-difference checks became less reliable for longer rollouts because repeatedrun loss variation grew much faster than the loss change produced by the tested parameter perturbations. Observation and loss definitions changed optimization behaviour and the solution preferred by an independent metric. These results motivate reproducible accumulation, finite-difference validation that compares perturbation-induced loss changes with repeatedrun variation, and explicit reporting of objective construction when differentiable simulation is used for robotic optimization.

Index Terms—differentiable simulation, material point method, physics-based optimization, system identification, deformable object manipulation, reproducibility

## I. INTRODUCTION

Robotic manipulation of deformable and granular materials appears in food handling, excavation and levelling, laboratory automation, and tool-based interaction. The outcome of these tasks depends on material deformation, friction, flow, and contact with the robot or end effector. Differentiable physics provides gradients through these interactions and has been used to identify material parameters for robotic manipulation [1], [2], generate demonstrations for granular-material learning [3], optimize excavation and levelling skills [4], and optimize robot shape [5]. These applications place numerical properties of the simulator directly inside the robotics optimization loop.

System identification, trajectory optimization, skill optimization, and differentiable end-effector design update different variables, while their computational structure is similar. A task loss is evaluated after simulated interaction, and derivatives are propagated backward through the dynamics to the optimized parameters, controls, skills, or geometry. Changes in accumulation order, numerical precision, rollout horizon, or observation discretization can therefore change the local gradient presented to the optimizer. We study these numerical properties as part of the interface between differentiable simulation and robotic optimization.

![](images/68cfe26bd3b35e7218e0960c348db2541b73642ece230857ad1604889f4369e6.jpg)  
Fig. 1. Robotic material-manipulation settings motivating this study. Top: Real elastoplastic interaction and the corresponding MPM simulation from the DPSI-derived setting [1]. Bottom: Real and simulated granular digging from the DDBot-derived setting [2].

The Material Point Method (MPM) is useful for robotic interaction with materials undergoing large deformation. Material state is carried by Lagrangian particles and momentum is exchanged through a temporary Eulerian background grid [6]. Each substep performs particle-to-grid (P2G) transfer, grid update and contact handling, followed by grid-to-particle (G2P) transfer. On a GPU, these transfers repeatedly combine many thread-local contributions into one shared grid or gradient entry. We refer to this many-to-one summation as a parallel reduction. Atomic additions allow the contributions to arrive in different thread orders, and floating-point rounding can make the final sum depend slightly on that order [7], [8]. Contact and observation losses add further nonlinear and discrete effects. Prior work has also shown that contact formulation and event timing can strongly affect differentiable-simulation gradients [9]–[11].

Our previous studies provide two controlled robotic testbeds for examining these effects. Differentiable Physics-based

System Identification (DPSI) reported multiple parameter solutions and disagreement between Chamfer-distance and Earth-Mover’s-distance objectives, with outcomes depending on the loss and initialization [1]. Differentiable Digging Robot (DDBot) reported exploding and fluctuating gradients and rugged loss landscapes, motivating gradient clipping and line search [2]. We use DPSI-derived elastoplastic and DDBotderived granular cases (Fig. 1) to examine GPU reduction order, finite-difference reliability across rollout length, and the construction of observations and losses. The experiments optimize material parameters. The same numerical operations also occur when gradients are propagated to robot trajectories, skill parameters, or end-effector geometry, which makes the findings relevant to a wider class of differentiable physics-based robotic optimization problems.

## II. METHOD AND STUDY SETUP

## A. Benchmark settings

The DPSI case models elastoplastic material with fixedcorotated elasticity and von Mises plasticity [1]. The DDBotderived case models granular material with a Hencky/Saint Venant-Kirchhoff elastic response and Drucker-Prager plasticity [2]. Both use the same projection-based signed distance field contact formulation. Keeping the contact model fixed allows this study to isolate numerical effects associated with differentiation, reduction order, rollout length, and objective construction. Appendix A gives the MPM equations, constitutive models, contact formulation, and observation definitions.

We evaluate 32-bit and 64-bit floating point (f32 and f64), several geometric objectives, multiple particle densities for the elastoplastic case, two rollout horizons for the granular case, and deterministic versus ordinary GPU parallel reductions. The complete experiment matrix and optimizer settings are given in Appendix C. Particle observations (PRT) compare simulated particles directly with a target. Point-cloud observations (PCD) first extract a surface representation. Chamfer distance (CD), Earth Mover’s distance (EMD), and height-map distance (HMD) then measure geometric discrepancy. Appendix A defines these quantities and the point-selection rules.

## B. Gradient evaluation and reproducible accumulation

Gradients are computed with reverse-mode automatic differentiation (AD) through the MPM rollout and compared with central finite differences (FD) over several perturbation sizes. For each perturbation, we compare the resulting change in loss with the spread obtained by repeating an identical forward simulation. This shows whether the FD numerator is large enough to be distinguished from ordinary numerical variation. Long trajectories use checkpointing to limit stored simulation state. Appendix B gives the reverse-mode equations, checkpointing, SVD derivative in detail.

To study parallel reductions, we compare ordinary GPU atomic addition with a deterministic fixed-point alternative. In the ordinary implementation, many threads add particle or gradient contributions to the same destination, so the order of the floating-point additions can change from run to run. In the deterministic implementation, each bounded contribution is mapped to a common integer scale, the integers are added exactly within the allocated range, and the result is converted back to floating point. Appendix B gives the reduction equations, range calculation, implementation details, and a toy numerical example.

## III. FINDINGS

## A. Parallel reductions can change the reverse gradient

Repeated GPU rollouts first diverge at the P2G parallel reduction, where contributions from many particles are added to the same grid entries. The reverse pass contains two further many-to-one sums: the adjoint of G2P sends contributions back to shared grid entries, and gradients from many particles are combined into shared material-parameter gradients. The differences are initially small, approximately $1 0 ^ { - 7 }$ in f32 and $1 0 ^ { - 1 6 }$ in f64, then propagate through later simulation and reverse-mode operations. On the 311-step DDBot-derived granular case, the GPU ensemble mean for $\partial L / \partial \nu$ was +1.29, while the deterministic single-threaded CPU reference was −7.99 (Fig. 2a). The two signs imply opposite updates of Poisson’s ratio.

We replaced the shared reductions involved in particle-grid transfer and parameter-gradient accumulation with deterministic integer reductions, and grouped repeated Chamfer-gradient contributions so they are summed in a fixed order. Five representative configurations were repeated after these changes, including f32 and f64 cases and both benchmark families. All five produced identical state and loss checksums. Appendix B describes the deterministic reductions, and Appendix D reports the repeat and accuracy measurements.

## B. Finite-difference checks can be unreliablefor longer rollouts

A central finite-difference estimate compares two forward losses evaluated at $\theta + h$ and $\theta - h$ . It is a useful external check for AD when the loss difference caused by this parameter perturbation is clearly larger than the variation obtained by repeating the same forward simulation. In the DPSI-derived elastoplastic case, this ratio was 93–806 at two global steps, meaning that the perturbation-induced loss change was much larger than the measured f32 loss spread. At horizons of ten steps or more, the ratio fell to 2–14 because repeated-run loss variation increased much faster than the loss change caused by the tested perturbations.

The f64 finite-difference estimates agreed closely with AD at short horizons: the worst relative discrepancy was at most 1.5× $1 0 ^ { - 5 }$ at two global steps and remained below $3 \times 1 0 ^ { - 5 }$ at five and ten steps. At 20 and 40 steps, the finite-difference sequence no longer converged for the worst component under the tested relative perturbation sizes, and the worst AD/FD discrepancy increased to $3 . 4 1 \times 1 0 ^ { - 1 }$ and $2 . 7 0 \times 1 0 ^ { - 1 }$ (Fig. 2b). Validation cost also increased from 25 s to 178 s. Thus, increasing rollout length did not provide a stronger finite-difference reference in this experiment. Appendix D reports the horizon measurements, while Appendix B defines the perturbation-induced loss change and repeated-run loss spread used in this comparison.

![](images/e7808ec0b52bab17f81fa173edce8554922f058d42c58948e27c7cf1d58015bf.jpg)

![](images/af4001fb23c213b1904d2c72357808ab3ef52193832f9da77bdd3c7b284707a9.jpg)

![](images/d45a503028f6c22ac8c99ffc7099b858f35824f91de3487614c67b9de0090471.jpg)  
Fig. 2. Three numerical findings measured in the robotic material-manipulation testbeds. (a) Unordered GPU many-to-one summation changed the sign of a DDBot-derived parameter gradient. (b) Finite-difference agreement in the DPSI-derived case depends on rollout horizon. (c) The objective with the most frequent descent did not give the lowest independent height-map error.

## C. Observation and loss design change optimization

DPSI previously reported that CD and EMD can favour different spatial aspects and converge to different parameters [1]. Under the common numerical setup used here, particle Chamfer distance (PRT-CD) reduced its own objective in 78 of 80 attempted epochs (97.5%). The corresponding rates were 67.5% for particle EMD (PRT-EMD), 53.8% for point-cloud CD (PCD-CD), and 57.5% for point-cloud EMD (PCD-EMD).

An independent terminal HMD gives a different ranking: mean HMD was 3738.4 mm for PRT-CD and 3698.6 mm for PRT-EMD (Fig. 2c). The particle-density experiment also shows that simulation resolution and observation resolution need not grow together. Increasing the DPSI-derived sampling from 218 to 876 particles increased the number of occupied surface cells from 89 to 151, a factor of 1.70 rather than four. In one 876-particle case, f32 and f64 selected different surface points and changed PCD-EMD by 0.517%, which is larger than many accepted line-search improvements in the campaign. Appendix D reports the complete results.

These measurements show that the loss supplied to the optimizer includes choices made by the observation pipeline as well as the simulated physics. In system identification, this can change the recovered material parameters. In trajectory, policy, or design optimization, it can change the gradient direction associated with controls, contact sequences, or shape variables.

## IV. CONCLUSION

This study examined numerical reliability in differentiable MPM optimization using elastoplastic and granular systemidentification cases as controlled testbeds. The experiments varied floating-point precision, rollout horizon, parallel reduction strategy, particle sampling, and geometric objective while keeping the underlying contact formulation fixed. The appendices provide the MPM formulation, reverse-mode implementation, deterministic accumulation method, complete experimental protocol, and detailed measurements.

There are three findings. First, unordered parallel reductions can propagate small floating-point differences into materially different long-horizon gradients; one DDBot-derived Poissonratio gradient changed sign relative to the deterministic reference, while deterministic reductions made all five selected repeat cases bit-identical. Second, finite-difference validation became less reliable as the rollout length increased. In the DPSI-derived horizon experiment, the loss change caused by the tested parameter perturbations was 93-806 times larger than repeated-run f32 variation at two steps, and only 2-14 times larger at horizons of ten steps or more. The f64 finite-difference sequence also stopped converging for the worst component at 20 and 40 steps under the tested perturbation sizes. Third, observation and loss construction changed both optimization behaviour and evaluation ranking; the objective that decreased the most did not achieve the lowest independent HMD, and particle density changed the effective surface observation nonlinearly.

These findings motivate three practices for differentiable physics-based optimization. Many-to-one sums that influence gradients should use a reproducible accumulation strategy when long rollouts can amplify small rounding differences. AD/FD checks should test several perturbation sizes and compare the resulting perturbation-induced loss changes with repeated-forward loss variation; a finite-difference estimate is a weak reference when these quantities are of similar magnitude or when the estimate does not converge as the perturbation size is reduced. The observation, objective, and discrete selection procedures should be treated as part of the optimization specification and reported with enough detail to reproduce their derivatives. These recommendations apply to system identification and to other gradient-based uses of differentiable simulation, including trajectory optimization, contact-rich control, and differentiable robot or end-effector shape optimization, because the same numerical operations connect simulated states to the final optimization variables.

This work is limited to TaiChi-based differentiable MPM implementation and observations collected by two motions (pressing a playdough and scooping a box of sand). Future investigation should be extended to other differentiable programming languages and more diverse contact processes.

## REFERENCES

[1] X. Yang, Z. Ji, and Y.-K. Lai, “Differentiable physicsbased system identification for robotic manipulation of elastoplastic materials,” The International Journal of Robotics Research, vol. 44, no. 13, pp. 2126–2155, 2025. DOI: 10.1177/02783649251334661.

[2] X. Yang, M. Wei, Y.-K. Lai, and Z. Ji, “DDBot: Differentiable physics-based digging robot for unknown granular materials,” IEEE Transactions on Robotics, vol. 42, pp. 152–169, 2026. DOI: 10.1109/TRO.2025. 3636815.

[3] M. Wei, X. Yang, Y.-K. Lai, S. A. Tafrishi, and Z. Ji, “A physics-informed demonstration-guided learning framework for granular material manipulation,” IEEE Transactions on Neural Networks and Learning Systems, vol. 37, no. 4, pp. 1590–1604, 2026. DOI: 10.1109/ TNNLS.2025.3622482.

[4] M. Wei, X. Yang, J. Yan, Y.-K. Lai, and Z. Ji, “Celebi’s choice: Causality-guided skill optimisation for granular manipulation via differentiable simulation,” in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025, pp. 15 845–15 852. DOI: 10. 1109/IROS60139.2025.11247432.

[5] X. Ye, X. Gao, K. Wu, Z. Pan, and T. Komura, “SDRS: Shape-differentiable robot simulator,” IEEE Transactions on Robotics, vol. 42, pp. 1309–1329, 2026. DOI: 10. 1109/TRO.2025.3636344.

[6] Y. Hu et al., “A moving least squares material point method with displacement discontinuity and two-way rigid body coupling,” ACM Transactions on Graphics, vol. 37, no. 4, 2018. DOI: 10.1145/3197517.3201293.

[7] J. Demmel and H. D. Nguyen, “Fast reproducible floating-point summation,” in 2013 IEEE 21st Symposium on Computer Arithmetic, 2013, pp. 163–172. DOI: 10.1109/ARITH.2013.9.

[8] J. Demmel and H. D. Nguyen, “Parallel reproducible summation,” IEEE Transactions on Computers, vol. 64, no. 7, pp. 2060–2070, 2015. DOI: 10.1109/TC.2014. 2345391.

[9] K. Werling, D. Omens, J. Lee, I. Exarchos, and C. K. Liu, “Fast and feature-complete differentiable physics engine for articulated rigid bodies with contact constraints,” in Proceedings of Robotics: Science and Systems, Virtual, Jul. 2021. DOI: 10.15607/RSS.2021.XVII.034.

[10] Y. D. Zhong, J. Han, B. Dey, and G. O. Brikis, “Improving gradient computation for differentiable physics simulation with contacts,” in Proceedings of the 5th Annual Learning for Dynamics and Control Conference, ser. Proceedings of Machine Learning Research, vol. 211, PMLR, 2023, pp. 128–141. [Online]. Available: https: //proceedings.mlr.press/v211/zhong23a.html.

[11] M. Liu, G. Yang, S. Luo, and L. Shao, “SoftMAC: Differentiable soft body simulation with forecast-based contact model and two-way coupling with articulated rigid bodies and clothes,” in 2024 IEEE/RSJ International

Conference on Intelligent Robots and Systems (IROS), 2024, pp. 12 008–12 015. DOI: 10.1109/IROS58592. 2024.10801308.

## APPENDIX AMPM AND OBSERVATION BACKGROUND

## A. One MPM substep

The implementation follows the quadratic Moving Least Squares Material Point Method (MLS-MPM) with an Affine Particle-In-Cell (APIC) velocity representation [6]. Particles carry mass $m _ { p } ,$ position $x _ { p } ,$ velocity $v _ { p } ,$ , an affine velocity matrix $C _ { p } ,$ and deformation gradient $F _ { p } .$ . Grid nodes are indexed by i, have positions $x _ { i } ,$ , and receive particle contributions through interpolation weights $w _ { i p }$ . A useful simplified view of particle-to-grid transfer is

$$
m _ { i } = \sum _ { p } w _ { i p } m _ { p } ,\tag{1}
$$

$$
( m v ) _ { i } = \sum _ { p } w _ { i p } m _ { p } [ v _ { p } + C _ { p } ( x _ { i } - x _ { p } ) ] ,\tag{2}
$$

$$
\boldsymbol { f } _ { i } ^ { \mathrm { i n t } } = - \sum _ { p } V _ { p } ^ { 0 } P _ { p } \boldsymbol { F } _ { p } ^ { T } \nabla \boldsymbol { w } _ { i p } ,\tag{3}
$$

where $V _ { p } ^ { 0 }$ is the reference particle volume and $P _ { p }$ is the first Piola-Kirchhoff stress produced by the constitutive model. The grid velocity is updated using internal force, gravity, boundary conditions, and tool contact,

$$
v _ { i } ^ { + } = \frac { ( m v ) _ { i } } { m _ { i } } + \Delta t \left( \frac { f _ { i } ^ { \mathrm { i n t } } } { m _ { i } } + g \right) ,\tag{4}
$$

for active nodes with nonzero mass. Grid-to-particle transfer interpolates the updated grid velocity back to each particle. The APIC affine matrix is reconstructed from the local grid velocity field, then the deformation and position are advanced approximately as

$$
F _ { p } ^ { + } = \left( I + \Delta t C _ { p } ^ { + } \right) F _ { p } , \qquad x _ { p } ^ { + } = x _ { p } + \Delta t v _ { p } ^ { + } .\tag{5}
$$

The actual kernels fuse several algebraic terms for efficiency, while the data flow remains the sequence above. This structure explains why shared accumulation appears in both the forward and backward computations. P2G sends many particle contri butions to the same grid node. In reverse mode, differentiating a grid-to-particle gather sends many adjoint contributions back to shared grid entries.

## B. Material and contact models

The DPSI-derived elastoplastic case uses fixed-corotated elasticity with von Mises plasticity, which describes an elastic response followed by irreversible deformation once the deviatoric stress reaches the yield condition [1]. The DDBot-derived granular case uses a Hencky/Saint Venant-Kirchhoff elastic response with Drucker-Prager plasticity. The Drucker-Prager criterion couples shear resistance to pressure and uses a friction angle to model granular yielding [2].

Both benchmark settings use the same projection-based signed distance field (SDF) contact formulation inherited from the earlier simulators. For a query point x, the SDF ϕ(x) is positive outside the tool and negative inside it. Its spatial gradient gives the outward surface normal $n = \nabla \phi / \| \nabla \phi \|$ When a grid node or particle enters the tool, the relative velocity is decomposed into normal and tangential components. The inward normal component is removed, and the tangential component is modified according to the friction coefficient. Tool motion is included through the relative velocity. Holding this formulation fixed allows the experiments to focus on numerical reliability rather than differences between contact models.

## C. Observation operators and losses

For a simulated particle set $X = \{ x _ { a } \}$ and target point set $Y = \{ y _ { b } \}$ , the symmetric Chamfer distance (CD) used in the benchmark has the form

$$
D _ { \mathrm { C D } } ( X , Y ) = \sum _ { a } \operatorname* { m i n } _ { b } \| x _ { a } - y _ { b } \| _ { 2 } + \sum _ { b } \operatorname* { m i n } _ { a } \| y _ { b } - x _ { a } \| _ { 2 } .\tag{6}
$$

Earth Mover’s distance (EMD) forms a one-to-one pairing between points. We write this pairing as π, where $\pi ( a ) = b$ means that simulated point $x _ { a }$ is paired with target point y<sub>b</sub>:

$$
D _ { \mathrm { E M D } } ( X , Y ) = \operatorname* { m i n } _ { \pi } \sum _ { a } \| x _ { a } - y _ { \pi ( a ) } \| _ { 2 } .\tag{7}
$$

When the two point sets have the same number of points, π is a permutation of the target indices. When their point counts differ, the experiments compare as many pairs as the smaller set contains and record how many target points remain unmatched. This point-count rule matters because changing the number of observed surface points changes both the number of terms in the EMD sum and which target points can participate. Height-map distance (HMD) first projects each point set to a horizontal height grid and sums the per-cell height differences. The DPSI-derived height map covers a $0 . 1 1 \mathrm { m } \times 0 . 1 1$ m region using 32 × 32 cells, following the previous DPSI evaluation [1].

Particle observations (PRT) apply CD or EMD directly to simulated particles. Point-cloud observations (PCD) first apply a surface observer. The observer divides the horizontal region into cells and retains the highest particle in each occupied cell. Its output can change when a cell becomes occupied or empty, or when a different particle becomes the highest one in a cell. Similar discrete choices occur in nearest-neighbour matching when two candidates have equal or nearly equal distances. We record the selected surface winners and point correspondences because changes in these choices can change the loss derivative. These observation details provide context for the stronger variability of PCD objectives reported in Sec. III-C.

## APPENDIX B

## DIFFERENTIATION AND DETERMINISTIC ACCUMULATION

## A. Reverse-mode computation in operational form

Let $s _ { k }$ contain the differentiable simulator state at substep k, and let θ contain the material and contact parameters. One substep and the terminal loss can be written as

$$
s _ { k + 1 } = F _ { k } ( s _ { k } , \theta ) , \qquad L = \ell ( s _ { K } ) .\tag{8}
$$

Reverse-mode AD propagates the adjoint $\lambda _ { k } = \partial L / \partial s _ { k }$ from the terminal state toward the initial state. With $\lambda _ { K } = \partial \ell / \partial s _ { K }$

$$
\lambda _ { k } = \left( \frac { \partial F _ { k } } { \partial s _ { k } } \right) ^ { T } \lambda _ { k + 1 } ,\tag{9}
$$

$$
\nabla _ { \boldsymbol { \theta } } L = \sum _ { k = 0 } ^ { K - 1 } \left( \frac { \partial F _ { k } } { \partial \boldsymbol { \theta } } \right) ^ { T } \lambda _ { k + 1 } .\tag{10}
$$

The implementation evaluates these equations with the following sequence for each checkpoint window:

1) Recompute the forward states in the current window from its saved boundary state.

2) Seed the gradient of the geometric loss at the terminal frame of the window.

3) Execute the gradient kernels in reverse simulation order: advection, G2P, grid update and contact, P2G, and material response.

4) Apply the hand-written SVD vector-Jacobian product at the material stage because the Taichi SVD operation in the tested build has no generated adjoint.

5) Copy the adjoint of the first state in the window to the terminal state of the preceding window and continue the reverse computation.

The checkpoint scheme stores O(W) device frames for a window of W global steps and recomputes earlier primal states when needed. The DPSI-derived tape measurement used 1.86 GiB peak device memory, while the DDBot-derived configuration used 2.47 GiB on an 11 GiB RTX 2080 Ti. Grid fields accounted for 99.4% of the analytic DPSI-derived tape footprint.

For an external gradient check, a scalar parameter component is perturbed by h and evaluated with the central finite difference

$$
g _ { \mathrm { F D } } ( h ) = \frac { L ( \theta + h ) - L ( \theta - h ) } { 2 h } .\tag{11}
$$

Several values of h are tested. For one parameter θ, we define the perturbation-induced loss change as

$$
\Delta L _ { \theta } ( h ) = \left| L ( \theta + h ) - L ( \theta - h ) \right| .\tag{12}
$$

We also repeat an identical forward evaluation R times and measure its loss spread as

$$
V = \operatorname* { m a x } _ { r = 1 , \ldots , R } L _ { r } - \operatorname* { m i n } _ { r = 1 , \ldots , R } L _ { r } .\tag{13}
$$

The ratio $\Delta L _ { \theta } ( h ) / V$ is reported only as an interpretation aid. A large value means that the effect of the parameter perturbation is clearly larger than the measured run-to-run variation, while a value near one means that numerical variation can substantially contaminate the finite-difference numerator. In the DPSI-derived sign diagnostic, a factor of ten was used as a conservative case-specific rule before assigning an f32 finite-difference sign; it is not treated as a universal threshold across datasets.

## B. Parallel reductions in MPM

A parallel reduction combines many values produced by different threads into one shared destination. For a grid or gradient entry j, the operation has the form

$$
S _ { j } = \sum _ { p \in \mathcal { N } ( j ) } c _ { p j } ,\tag{14}
$$

where thread p computes a contribution $c _ { p j }$ and all contributions in $\mathcal { N } ( j )$ must be added to the same $S _ { j }$ . In $\mathrm { P 2 G } , S _ { j }$ can be grid mass or momentum and the contributing threads are particles whose interpolation stencils include node $j .$ Reverse mode introduces the same pattern when the adjoint of G2P sends contributions back to shared grid entries and when per-particle derivatives are combined into material-parameter gradients.

On a GPU, these many-to-one sums are commonly implemented with atomic addition. Atomicity prevents two threads from overwriting one another, while the scheduler is still free to choose which contribution is added first. A reduction containing $a , b ,$ and c may therefore be evaluated as $( a + b ) + c$ in one execution and $a + ( b + c )$ in another. Floating-point addition rounds after each operation and is not exactly associative, so the two orders can differ in their last bits. The first divergence in our isolated MPM test occurred in P2G grid mass and momentum even though all per-particle quantities entering the reduction were bit-identical. The same ordering effect was also present in the reverse G2P scatter and material-parameter reductions.

## C. Fixed-point integer accumulation

The deterministic reduction places all contributions on one common binary scale before adding them. Let F be the number of fractional bits and let $c _ { i }$ be one bounded floating-point contribution. We encode

$$
q _ { i } = \mathrm { r o u n d } \left( c _ { i } 2 ^ { F } \right) , \qquad Q = \sum _ { i } q _ { i } , \qquad \widehat { S } = Q 2 ^ { - F } .\tag{15}
$$

The additions in $Q$ are integer additions, so their result is independent of thread arrival order while the integer range is respected. Quantization occurs once per contribution and conversion back to floating point occurs once after the reduction.

The required integer range can exceed one native field, especially for f64. We therefore represent Q using a high field and a low field,

$$
Q = Q ^ { \mathrm { h i } } 2 ^ { W } + Q ^ { \mathrm { l o } } , \qquad 0 \le Q ^ { \mathrm { l o } } < 2 ^ { W } .\tag{16}
$$

This can be read as one wider integer split into two blocks. The implementation uses binary blocks, with carry from the low field transferred to the high field before the value is converted back to floating point.

A small numerical example illustrates the complete reduction. Suppose three threads contribute

$$
c _ { 1 } = 1 . 2 5 , \qquad c _ { 2 } = 0 . 8 7 5 , \qquad c _ { 3 } = 1 . 5 0 ,\tag{17}
$$

and choose $F = 3 ,$ so the common fixed-point scale is $2 ^ { F } = 8$ Equation (15) gives

$$
q _ { 1 } = 1 0 , \qquad q _ { 2 } = 7 , \qquad q _ { 3 } = 1 2 .\tag{18}
$$

Any arrival order gives the same integer tota $Q = 1 0 + 7 + 1 2 =$ 29, and converting back gives $\widehat { S } = 2 9 / 8 = 3 . 6 2 5$ . To show the high/low storage, take the toy field width $W = 4$ , whose low field stores values from 0 to 15. Then

$$
Q ^ { \mathrm { h i } } = 1 , \qquad Q ^ { \mathrm { l o } } = 1 3 ,\tag{19}
$$

because $1 \times 1 6 + 1 3 = 2 9$ . If an update makes the low field exceed 15, the overflow is carried into the high field. The real implementation uses much larger binary fields and a scale selected from the reduction bounds; this small example only makes the representation and order independence visible.

The bit budget depends on the maximum number of contributions n. With

$$
N = \lceil \log _ { 2 } n \rceil , \qquad W = 5 3 - N , \qquad B = 2 W ,\tag{20}
$$

we require $B \geq p + 8 ,$ , where $p$ is the target floating-point precision (24 significand bits for f32 and 53 for f64). The DDBot-derived P2G reduction with 27,440 particles has $N =$ 15, $W = 3 8 .$ , and $B = 7 6$ , leaving 23 bits beyond f64 precision. The narrowest margin in the complete campaign occurs in the highest-density DPSI-derived PRT-CD reduction, where 31 guard bits remain.

Repeated-source Chamfer gradients need a related change. A generated adjoint may send several pairwise loss terms to the same source particle through floating-point atomic addition. We sort pairs by source particle and store compressed sparse row (CSR) offsets. One source particle then reads and sums its own contiguous block of pair contributions in a fixed order. In the six-launch diagnostic, the generated adjoint produced six distinct gradients, while the CSR gather produced one repeated value.

## APPENDIX C EXPERIMENTAL PROTOCOL

TABLE I  
BENCHMARK CONFIGURATIONS USED IN THE OBJECTIVE MATRIX.
<table><tr><td>Setting</td><td>Configuration</td></tr><tr><td>DPSI-derived elastoplastic</td><td>Fixed-corotated elasticity with von Mises plas- ticity; 218, 451, or 876 particles; 94 global steps with 50 MPM substeps per step; PRT-</td></tr><tr><td>DDBot-derived granular</td><td>CD, PRT-EMD, PCD-CD, and PCD-EMD. Hencky/Saint Venant-Kirchhoff elasticity with Drucker-Prager plasticity; 27,440 particles; 200 or 311 global steps with 20 MPM substeps per step; HMD and PCD-EMD.</td></tr></table>

The experiment matrix contains 64 primary cells. The DPSIderived part contains four objectives, two precisions (f32 and f64), three particle densities, one 94-step horizon, and two initialization seeds, giving 48 cells. The DDBot-derived part contains two objectives, two precisions, one particle density, two horizons, and two seeds, giving 16 cells. Five representative cells were repeated for bit-level reproducibility, giving 69 campaign tasks in total. All use the same projection-based SDF contact formulation.

DPSI-derived optimization uses Adam. DDBot-derived opti mization uses RMSProp followed by a line-search evaluation of candidate step scales. The DDBot-derived line-search set includes both gradient directions so the experiment can record cases in which the computed direction does not locally reduce the objective. Across the base DDBot-derived cells, 14 of 58 accepted f32 steps and 13 of 42 accepted f64 steps selected the opposite direction.

The particle-density experiment applies only to the DPSIderived cases. Its sampling densities correspond to 218, 451, and 876 particles. At the initial state, these produce 89, 105, and 151 occupied PCD surface cells. The observed surface representation therefore grows more slowly than the underlying particle count because many added particles project into cells that are already occupied.

## APPENDIX D DETAILED RESULTS

A. Gradient validation across horizon

TABLE II  
DPSI-DERIVED HORIZON EXPERIMENT. THE F64 COLUMN REPORTS THE WORST RELATIVE AD/FD DISCREPANCY ACROSS THE TESTED MATERIAL COMPONENTS.
<table><tr><td>Global steps</td><td>f64 worst error</td><td>f32 loss spread</td><td>FD time</td></tr><tr><td>2</td><td> $\overline { { 4 . 4 \times 1 0 ^ { - 6 } ~ t o ~ 1 . 5 \times 1 0 ^ { - 5 } } }$ </td><td> $\overline { { 2 . 2 \times 1 0 ^ { - 7 } \ t o \ 1 . 4 \times 1 0 ^ { - 6 } } }$ </td><td>25 s</td></tr><tr><td>5</td><td> $2 . 7 5 \times 1 0 ^ { - 5 }$ </td><td> $2 . 9 \times 1 0 ^ { - 5 }$ </td><td>38 s</td></tr><tr><td>10</td><td> $2 . 9 3 \times 1 0 ^ { - 5 }$ </td><td> $7 . 1 \times 1 0 ^ { - 5 }$ </td><td>59 s</td></tr><tr><td>20</td><td> $3 . 4 1 \times 1 0 ^ { - 1 }$ </td><td> $4 . 7 \times 1 0 ^ { - 5 }$ </td><td>99 s</td></tr><tr><td>40</td><td> $2 . 7 0 \times 1 0 ^ { - 1 }$ </td><td> $2 . 8 \times 1 0 ^ { - 5 }$ </td><td>178 s</td></tr></table>

Table II gives the measurements summarized in Fig. 2b. At two global steps, $\Delta L _ { \theta } ( h ) / V$ ranged from 93 to 806 across the tested material components. At horizons of ten steps or more, it fell to 2–14 because the repeated-run f32 loss spread increased much faster than the loss change caused by the same relative parameter perturbations. This makes the finitedifference numerator increasingly difficult to distinguish from ordinary numerical variation. At 20 and 40 global steps, the fixed relative perturbation sizes also stopped giving a converged f64 finite-difference sequence for the worst component. The 2-step setting therefore has the lowest runtime, the smallest repeated-forward variation, and the most consistent f64 finitedifference reference among the tested horizons.

## B. Objective and particle-density results

TABLE III  
REFERENCE-DENSITY DPSI-DERIVED OPTIMIZATION BEHAVIOUR. DESCENT COUNT IS THE NUMBER OF EPOCHS WHOSE POST-UPDATE OBJECTIVE IS LOWER THAN THE ENTRY VALUE FOR THAT EPOCH, AGGREGATED OVER TWO SEEDS AND BOTH PRECISIONS. HMD IS AN INDEPENDENT TERMINAL EVALUATION.
<table><tr><td>Objective</td><td>Descending epochs</td><td>Mean HMD (mm)</td></tr><tr><td>PRT-CD</td><td>78/80 (97.5%)</td><td>3738.4</td></tr><tr><td>PRT-EMD</td><td>54/80 (67.5%)</td><td>3698.6</td></tr><tr><td>PCD-CD</td><td>43/80 (53.8%)</td><td>3711.2</td></tr><tr><td>PCD-EMD</td><td>46/80 (57.5%)</td><td>3708.6</td></tr></table>

The full particle-density result is objective-dependent. At f32, PRT-EMD improves from a total loss change of $- 0 . 8 7 \% / -$ 0.48% at the reference density (seed 0/1) to −2.60%/ − 1.26% at 2× density and $- 5 . 2 1 \% / - 4 . 0 6 \%$ at 4× density. PRT-CD changes in the other direction, from $- 1 0 . 7 5 \% / - 2 . 8 0 \%$ at the reference density $\mathrm { t o \ - 0 . 6 4 \% / - 0 . 3 0 \% }$ at 2× and $- 0 . 0 7 \% / -$ 0.60% at 4×. These results motivate reporting particle density together with the observation objective.

The PCD surface count grows from 89 to 151 while particle count grows from 218 to 876. Between the first two density levels, 93% of newly added particles land in an already occupied surface cell; between the second and third levels the figure is 89%. The particle-density change therefore modifies both the MPM discretization and the observation sampling, while the $3 2 \times 3 2$ PCD grid remains fixed.

## C. Runtime and reproducibility measurements

The contact timing isolates a large implementation cost that is separate from the numerical questions in the paper. Both contact backends evaluate the same projection-based SDF contact formulation, while the GPU version removes six host/device frame transfers per MPM substep. The measured speedup at 4,000 particles ranges from 35.1× on CPU execution of the device-style kernels to 92.9× on the tested CUDA configuration, depending on precision and device. The 94-step benchmark gives the end-to-end 12.73× value in Table IV.

The deterministic accumulator also improves numerical accuracy against a fixed-order long-double reference in the isolated reduction test. For f32 grid mass, the maximum error was $2 . 0 0 \times 1 0 ^ { - 1 0 }$ with deterministic accumulation and $1 . 4 4 \times 1 0 ^ { - 9 }$ for the best of eight atomic runs. The fixed-point representation therefore provides order-independent repetition while retaining adequate precision for the tested reductions.

## D. Connection to earlier DPSI and DDBot observations

The earlier DPSI study observed that CD and EMD may move in opposite directions during optimization and can converge to different parameter values even when the resulting deformations appear similar [1]. It also reported modelling discrepancies around sharp tool contacts and noted that time integration, step size, and contact handling can contribute to those differences. The objective matrix in this paper adds controlled measurements of observation choice, particle density, precision, and reduction order around the same class of systemidentification problem.

The earlier DDBot study reported exploding and fluctuating gradients in long granular trajectories and used gradient clipping and line search during optimization [2]. The DDBotderived experiments here isolate one numerical source of such variability by comparing unordered GPU reductions with an order-independent reference. This gives a reproducible benchmark for later studies that change the contact formulation while keeping the remaining numerical controls fixed.

TABLE IV  
SELECTED COMPUTATIONAL MEASUREMENTS.
<table><tr><td>Case</td><td>Runtime or outcome</td></tr><tr><td>DPSI 94-step SDF contact, CPU loop vs. GPU kernels</td><td>260.821 s → 20.488 s (12.73×)</td></tr><tr><td>DPSI FD validation, 2 steps vs. 40 steps</td><td>25 s → 178 s</td></tr><tr><td>DDBot HMD epoch, 200 steps</td><td>about 115 s; about 154 s with the extended line-search set</td></tr><tr><td>DDBot PCD-EMD epoch, 311 steps</td><td>about 170 s; about 227 s with the extended line-search set</td></tr><tr><td>DPSI particle-density change</td><td>about +1.5% wall time for denser f32 cases; recorded device memory unchanged</td></tr><tr><td>Deterministic repeat cells</td><td>5/5 bit-identical across selected f32/f64 and DPSI/DDBot-derived cases</td></tr></table>