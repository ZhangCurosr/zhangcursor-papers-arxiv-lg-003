# Transformer-inspired architectures make machine-learned electron densities accurate and transferable enough to bypass self-consistency

Jeheon Woo

Center for AI and Natural Sciences, Korea Institute for Advanced Study, Seoul 02455, Republic of Korea jhwoo1905@kias.re.kr

## Abstract

We introduce machine learning models that accurately predict the electron density, the central variable of density functional theory. They are built on Transformer-inspired architectures, which require little computation per parameter and can therefore be enlarged at low cost. The proposed models, SwinUNETR-V2 and its attention-free counterpart NeXtUNETR, achieve state-of-the-art accuracy on QM9, with normalized mean absolute errors in the density averaging 0.075% and 0.046%, respectively. The predicted densities are accurate enough to bypass the self-consistent iteration: a single Kohn–Sham step reproduces the self-consistent electronic structure. For NeXtUNETR, the median errors are 0.036 meV in the total energy, 0.65 meV Å<sup>−1</sup> in the forces and 1.8 meV in the HOMO–LUMO gap. The forces not only recover the self-consistent minima in geometry optimization but also successfully drive ab initio molecular dynamics. Fine-tuned on small datasets, both networks also transfer accurately to systems containing elements absent from QM9 or larger than its molecules.

Kohn–Sham density functional theory<sup>1,2</sup> is the most widely used electronic-structure method in chemistry and materials science, and the electron density is its central variable: it determines all ground-state properties<sup>1</sup>, and the Kohn–Sham Hamiltonian is built from it. Finding the density is the costly part: the self-consistent-field (SCF) iteration diagonalizes a Hamiltonian built from a trial density and updates the density until the input and output densities agree. Because this iteration is repeated at every step of quantum-chemical simulations such as geometry optimization and molecular dynamics, it dominates their cost. Machine learning ofers two ways around it. Machine-learned interatomic potentials (MLIPs) avoid the electronic problem by mapping atomic positions directly to energies and forces <sup>3–5</sup>, and foundation models now span most of the periodic table<sup>6</sup>, but they return no electronic structure. Learning the density <sup>7,8</sup>, or the density matrix or Hamiltonian<sup>9,10</sup>, retains it: from a predicted density, a single Kohn–Sham step returns energies, forces and orbitals together, and their quality is set by that of the density.

The density at a point is determined mainly by the atoms nearby, a property known as the nearsightedness of electronic matter<sup>11</sup>, so a model can learn it from local environments and apply it to larger systems. A useful model must be accurate, because density errors propagate into every derived property, and transferable, because a model trained on small molecules must also serve larger, charged or diferently composed systems. Most density models describe a molecule by its atoms: kernel methods and graph neural networks, mostly rotationally equivariant, expand the density in atom-centered or floating basis functions or predict it point by point <sup>12–16</sup>. Others work on the real-space grid of the calculation, where the density forms a three-dimensional image, and refine a superposition of atomic densities (SAD), alone <sup>8</sup> or together with grid-projected atomic fingerprints <sup>17</sup>, with convolutional neural networks (CNNs). Li et al. cast this refinement as image super-resolution <sup>8</sup>: their residual network (here SR-ResNet), trained on QM9 (ref. 18), was more accurate than earlier models and transferred to unseen geometries and elements after fine-tuning on small datasets.

If density prediction is an imaging problem, advances in computer vision may carry over to it. The Transformer<sup>19</sup>, developed for natural language processing, was applied to images by the Vision Transformer<sup>20</sup>, and hierarchical variants such as the Swin Transformer <sup>21</sup>, with self-attention in local windows and a hierarchy of resolutions, reach or exceed the accuracy of CNNs such as ResNet <sup>22</sup> on major vision benchmarks. In three dimensions, SwinUNETR and SwinUNETR-V2 (refs. 23,24) are widely used for volumetric medical image segmentation, where the convolutional 3D U-Net <sup>25</sup> had long been the standard. Transformers have also reshaped CNNs: ConvNeXt<sup>26</sup> reached the accuracy of Swin by adopting its macro-design, with large-kernel depthwise convolutions in place of attention, and 3D UX-Net and MedNeXt carried this design to volumetric images <sup>27,28</sup>. We call both kinds of networks Transformer-inspired: they share the hierarchical macro-design and block recipe of the Swin Transformer, with or without attention. To our knowledge, density refinement on real-space grids treated as images has so far used only CNNs <sup>8,17</sup>, and whether Transformerinspired architectures make densities more accurate and transferable has not been tested.

An accurate density can replace the SCF iteration altogether. The Kohn–Sham Hamiltonian built from the predicted density is diagonalized once, and this single Kohn–Sham step gives the orbitals, a new density and the energy. Because the Kohn–Sham energy is stationary at the self-consistent density, the error of this one-step energy is second order in the density error <sup>29,30</sup>. Forces need more care. The predicted density is not exactly self-consistent, so, as for other non-self-consistent energies<sup>31,32</sup>, the exact gradient of the one-step energy retains a term, omitted by standard Kohn– Sham forces, from the response of the density to the nuclear positions. For a learned density, the response must be propagated through the network; we call the added term the non-selfconsistency correction (NSCC). Non-self-consistent properties of learned densities have so far been tested on fixed structures <sup>8,14,17,33,34</sup>, and SCF-free dynamics has used system-specific models of the density<sup>7,35</sup> or density matrix<sup>10,36</sup>. Whether a transferable density model makes one Kohn–Sham step accurate enough for geometry optimization and energy-conserving molecular dynamics has remained open.

Here we show that Transformer-inspired architectures make learned densities accurate and transferable enough to bypass the SCF iteration. We adapt two U-shaped networks to refine the SAD guess: SwinUNETR-V2, with windowed self-attention, and NeXtUNETR, an attention-free counterpart with the same hierarchical design. Trained on QM9, they reach normalized mean absolute errors (NMAEs) in the density averaging 0.075% and 0.046%, respectively, below those of any published model trained on the same data. At equal training data, NeXtUNETR is the most accurate network at every size tested, and in the same U-shaped network, ConvNeXt blocks are markedly more accurate than Swin blocks. One Kohn–Sham step from the NeXtUNETR densities gives median errors of 0.036 meV in the total energy and 0.65 meV Å<sup>−1</sup> in the forces, and both networks give frontier orbital energies within a few meV. With or without the NSCC, the forces relax 200 molecules from nearby starting structures, 97–98.5% of them to within 0.1 Å of the SCF minima. The NSCC, which also lowers the force error for most molecules, makes molecular dynamics energy-conserving for a molecule with more heavy atoms than any training molecule. Both networks also transfer: without fine-tuning, SwinUNETR-V2 is the most accurate model on charged fragments with unseen elements and on water clusters, and fine-tuned on about a hundred structures, both outperform SR-ResNet on these fragments. Architectural advances made for images thus carry over to electronic structure.

c  
![](images/db4d34d8c094d94cf0ce8126c11561b347a15fd8c9d7ccaf65844999dfd267ea.jpg)

![](images/2644d505eedd44e6eba3d043cdbcc27bde070321cc0fc8a7e57fed4e3655d595.jpg)

![](images/c6fcc0412af73f3621c43b3e39a4ec23a3e2f7dcfff004290a78aacc05aff8b7.jpg)

![](images/4abdc1cd3f619d7b47d5cfa06b21115df9499cb0f04573d7f22bca8f370c10a9.jpg)  
Fig. 1 | One-step Kohn–Sham calculation from predicted densities, and the architectures compared. a, Schematic workflow. A U-shaped network refines the superposition of atomic densities (SAD) on the real-space grid (left) into the predicted density $\rho _ { \mathrm { M L } }$ . Its encoder blocks mix features within $7 \times 7 \times 7 \mathrm { - v o x e l }$ regions (enlarged): SwinUNETR-V2 by windowed attention, NeXtUNETR by a depthwise convolution. One diagonalization of the Kohn–Sham Hamiltonian $H [ \rho _ { \mathrm { M L } } ]$ , without a self-consistent-field (SCF) loop, gives the energy, forces and orbital energies (right: the one-step energy surface). b, Density NMAE against parameter count for three families trained on the same 8,000 QM9 molecules (QM9-subset-8k). c, Floating-point operations per molecule against parameter count. d, Density NMAE against network forward time per molecule (one NVIDIA H200 NVL, batch size 1, fp32). b–d, Means over 500 randomly sampled QM9 test molecules; one training run per model; labels give the parameter count; counting and timing in Methods. Extended Data Fig. 1 shows the details of the architectures.

## Results

## Density refinement and the one-step calculation

A SAD guess density is sampled on the real-space grid of the calculation, and a neural network returns the refined density $\rho _ { \mathrm { M L } }$ on the same grid (Fig. 1a). The Kohn–Sham Hamiltonian matrix $H [ \rho _ { \mathrm { M L } } ]$ is built and diagonalized once. Like a converged Kohn–Sham calculation, this single step gives the orbitals and orbital energies, the density matrix and the properties derived from it, and the one-step total energy, whose analytic derivative gives the forces. We report forces with the NSCC (w/ NSCC), which are the exact gradient of the one-step energy, and without it $( \mathrm { w } / \mathrm { o }$ NSCC), the standard Kohn–Sham force.

Both networks are U-shaped encoder–decoders that process the grid at its input resolution and five successively halved ones (Extended Data Fig. 1). SwinUNETR-V2 (ref. 24) mixes information within $7 \times 7 \times 7 \cdot$ -voxel windows by self-attention, whose weights depend on the input; NeXtUNETR mixes a neighborhood of the same size by a depthwise convolution<sup>26</sup>, whose weights are fixed after training. The baseline is the SR-ResNet of Li et al. <sup>8</sup>.

All densities are Perdew–Burke–Ernzerhof (PBE) valence densities (Methods). We measure the error of a density by its NMAE over the grid, in percent of the electron count (Methods). The energy error is $\varepsilon _ { E } = | E _ { \mathrm { 1 - s t e p } } - E _ { \mathrm { S C F } } |$ per molecule, and the force error $\varepsilon _ { F }$ is the absolute error of a Cartesian force component, summarized by its mean or median over all components.

## Architecture comparison at equal training data

We trained ten models of the three families on the same 8,000 QM9 molecules <sup>18</sup> (QM9-subset-8k) and evaluated them on 500 molecules randomly sampled from the QM9 test split (Fig. 1b–d). The U-shaped networks were enlarged through the channel width C, the number of feature channels in the first encoder stage, which sets the widths of all later stages; SR-ResNet was enlarged in width and depth. NeXtUNETR is the most accurate family at every size. Its smallest model, with 17 million parameters and an NMAE of 0.187%, is more accurate than every SwinUNETR-V2 model (best 0.204%, 292 million parameters) and the largest SR-ResNet (0.198%, 70 million). At the same channel width, C = 96, NeXtUNETR-67M reaches 0.133% against 0.204% for SwinUNETR-V2-292M, with a quarter of the parameters and one-eleventh of the floating-point operations.

In every family, the floating-point operations grow in proportion to the parameter count (Fig. 1c), but the operations per parameter difer: 7–9 GFLOPs per million parameters for NeXtUNETR, 19–22 for SwinUNETR-V2 and 285 for SR-ResNet, because the U-shaped networks place most of their parameters on coarse grids, whereas SR-ResNet runs all its residual blocks at half resolution. The U-shaped networks can therefore be enlarged at little computational cost, and the added capacity makes them more accurate: in both families the NMAE falls with model size until it levels of at 150–300 million parameters (Fig. 1b).

In ablations (Extended Data Table 1), the encoder block makes a large diference. With Swin Transformer blocks in place of the ConvNeXt blocks of its encoder, NeXtUNETR’s NMAE rises from 0.133 to 0.250%. The two blocks have the same feed-forward layers and difer chiefly in how they mix space, by windowed attention or by depthwise convolution, and in their normalization (layer or group). Removing the convolutional residual blocks on the downsampled skips and in the bottleneck, which SwinUNETR-V2 inherits from SwinUNETR<sup>23</sup>, nearly halves its parameter count, from 292 to 154 million, but raises the NMAE only from 0.204 to 0.209%.

For the remaining results, SwinUNETR-V2 and NeXtUNETR were trained on the full QM9 training set (123,835 molecules) in the SwinUNETR-V2-292M and NeXtUNETR-67M configurations, which share C = 96. SR-ResNet is Li et al.’s published 9-million-parameter QM9 model.

## Densities, energies and orbitals on QM9

On the 10,000 QM9 test molecules, the three models reduce the mean NMAE of the SAD guess, 13.9%, to 0.138% for SR-ResNet, 0.075% for SwinUNETR-V2 and 0.046% for NeXtUNETR (Table 1). The last two are 46% and 67% below that of SR-ResNet and below the 0.116% reported for BOA, an equivariant basis-function network trained on the same densities and split <sup>37</sup>. The ranking of the three models holds for almost every molecule.

One diagonalization brings the density closer to the self-consistent one, lowering the mean NMAE by a factor of 2.4–3.0 for all three models (Fig. 2a, dashed). The energy error is second order in the density error, so the diferences between models widen (Fig. 2b). The median $\varepsilon _ { E }$ is 0.52 meV for SR-ResNet, 0.14 meV for SwinUNETR-V2 and 0.036 meV for NeXtUNETR; the last two are 2.5–3 orders of magnitude below the 1 kcal $\mathrm { m o l ^ { - 1 } }$ (43 meV) threshold usually called chemical accuracy. NeXtUNETR’s mean errors in the highest occupied and lowest unoccupied molecular orbital (HOMO and LUMO) energies and in the HOMO–LUMO gap are 2.8, 2.2 and 2.9 meV, about a quarter of SR-ResNet’s; its median gap error is 1.8 meV.

Table 1 | Accuracy on the QM9 test set. Density NMAE of the predicted density and of the density after one Kohn–Sham step, and errors of the one-step energies and forces (∆HOMO, ∆LUMO and ∆gap: absolute errors of the highest occupied and lowest unoccupied molecular orbital (HOMO and LUMO) energies and of the gap). For the SAD guess and the three models evaluated here, mean and median over the 10,000 test molecules (ε<sub>F</sub>: over 539,613 force components); in parentheses, the mean without the two molecules with a reference HOMO–LUMO gap below 1 eV<sup>a</sup>. For the other models, mean errors as reported. Best value in bold.
<table><tr><td></td><td></td><td colspan="2">Density NMAE (%)</td><td colspan="4">One-step energies (meV)</td><td colspan="2">One-step forces εF (meV Å−1)</td></tr><tr><td>Model</td><td></td><td>Predicted</td><td>One step</td><td>εE</td><td>ΔHOMO</td><td>∆LUMO</td><td>∆gap</td><td>w/o NSCC</td><td>w/ NSCC</td></tr><tr><td>SAD guess (input)</td><td>mean</td><td>13.9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>median</td><td>14.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Previous work</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>eqDeepDFT (ref. 13)</td><td>mean</td><td> $0 . 2 7 ^ { \mathrm { b } }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ELECTRA (ref. 16)</td><td>mean</td><td>0.176b</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BOA (ref. 37)</td><td>mean</td><td>0.116; 0.134b</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SR-ResNet (ref. 8)c</td><td>mean</td><td>0.138</td><td>0.058</td><td>2.84 (1.03)</td><td>11.7</td><td>9.4</td><td>10.7</td><td>11.87 (11.63)</td><td>17.05 (8.68)</td></tr><tr><td></td><td>median</td><td>0.129</td><td>0.050</td><td>0.52</td><td>9.2</td><td>6.4</td><td>7.1</td><td>5.54</td><td>3.37</td></tr><tr><td>This work</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SwinUNETR-V2</td><td>mean</td><td>0.075</td><td>0.030</td><td>0.90 (0.26)</td><td>6.1</td><td>4.5</td><td>5.7</td><td>6.21 (6.04)</td><td>10.12 (4.36)</td></tr><tr><td></td><td>median</td><td>0.070</td><td>0.026</td><td>0.14</td><td>4.8</td><td>3.0</td><td>3.7</td><td>2.93</td><td>2.37</td></tr><tr><td>NeXtUNETR</td><td>mean</td><td>0.046</td><td>0.015</td><td>0.42 (0.087)</td><td>2.8</td><td>2.2</td><td>2.9</td><td>3.21 (3.09)</td><td>4.46 (1.63)</td></tr><tr><td></td><td>median</td><td>0.043</td><td>0.013</td><td>0.036</td><td>2.1</td><td>1.4</td><td>1.8</td><td>1.45</td><td>0.65</td></tr></table>

![](images/b81344b39c2b48b89e9d398b5c5190dbb860d33fb05e5759ce21695e9669bdc3.jpg)

![](images/3e45eb10d97e7c9c96dc91311b3a64fc2aac8e6cc2b7615f48512eb7d077de2e.jpg)

![](images/565b2d25903732f9493ffb9bf47469ed00c8712ae1e2d0a1ef08f8cfc6712660.jpg)

d  
![](images/59113560c6add46b1114a9695e133a99caf382cc9423b8b3967c6c04ed7c7b52.jpg)

![](images/9de76d572b1127594c7f268420f40c9b7ad0603f85cfac35141da00140ebfe7a.jpg)  
Fig. 2 | Accuracy on QM9 and geometry optimization on the one-step surface. a, Cumulative distribution of the density NMAE over the 10,000 QM9 test molecules; solid, predicted density $\rho _ { \mathrm { M L } } ;$ dashed, density after the one-step diagonalization. b, Same for the one-step energy error $\varepsilon _ { E }$ . c, Same for the force error per Cartesian component; solid, with the non-self-consistency correction (w/ NSCC); dashed, without it $( \mathrm { w } / \mathrm { o }$ NSCC). At most 4 molecules (b) and 110 components (c) per model lie beyond the axes, for SwinUNETR-V2 and NeXtUNETR almost all from the two molecules with a reference HOMO–LUMO gap below 1 eV. d, QM9 85492 $\mathrm { ( C _ { 7 } H _ { 8 } O _ { 2 } ) }$ , drawn at random from the molecules that every model optimizes to within 0.01 Å of the reference optimum (four more in Supplementary Fig. 1). e,f, Its optimization from an MMFF94 starting structure on the SCF and one-step surfaces, $\mathrm { w } / $ and $\mathrm { w / o }$ NSCC: energy relative to each surface’s optimum (e; symmetric logarithmic axis) and maximum force (f; dotted, the convergence criterion). Extended Data Table 2 summarizes all 200 molecules.

The two molecules with reference HOMO–LUMO gaps below 1 eV, 0.38 and 0.03 eV, dominate the mean energy errors and the mean force errors with the NSCC. For them, one diagonalization can occupy the wrong orbitals, giving energy errors of 0.3–18 eV. Table 1 therefore also gives the means without them, and Extended Data Fig. 2 excludes them.

## One-step forces

With the NSCC, the median absolute errors of the force components are 3.37 meV $\textup { \AA } ^ { - 1 }$ for SR-ResNet, 2.37 meV $\textup { \AA } ^ { - 1 }$ for SwinUNETR-V2 and 0.65 meV $\textup { \AA } ^ { - 1 }$ for NeXtUNETR (Fig. 2c and Table 1). The median absolute reference component is 90 meV $\textup { \AA } ^ { - 1 }$ . Without the NSCC, the medians of the same three models are 5.54, 2.93 and 1.45 meV $\textup { \AA } ^ { - 1 }$ . The NSCC lowers the permolecule mean absolute error of the forces for 88% of the molecules with SR-ResNet, 86% with SwinUNETR-V2 and 97% with NeXtUNETR.

The mean over all molecules is nevertheless larger with the NSCC, because the correction amplifies the errors of the two small-gap molecules. Without them, the mean is smaller with the NSCC for every model (Table 1). The NSCC, which makes the force the exact gradient of the one-step energy, thus also brings it closer to the SCF force for most molecules.

## Geometry optimization

To test whether the one-step surfaces place their minima where the SCF surface does, we optimized 200 QM9 test molecules drawn at random (Methods). Each molecule was optimized on the SCF surface and on the one-step surfaces of both networks, w/ and $\mathrm { w / o }$ NSCC, and of the SAD guess as a control (Fig. 2d–f, Extended Data Table 2). All runs used geomeTRIC <sup>39</sup> with the same criteria and started from an MMFF94 (ref. 40) relaxation of the reference optimum.

All optimizations with SwinUNETR-V2 and NeXtUNETR, w/ and $\mathrm { w / o }$ NSCC, converged in a median of 15.5–16 steps, as on the SCF surface (15.5). The final structures lie a median of 0.002– 0.005 Å from the optimum the reference reaches from the same start, and 97–98.5% lie within 0.1 Å, against 42.5% of the starting structures. The runs shown also follow the reference step by step (Fig. 2e,f and Supplementary Fig. 1). QM9 contains only equilibrium structures, but every optimization starts from a distorted one; the agreement along the paths indicates that the predicted densities, and the energies and forces derived from them, remain accurate away from equilibrium.

Without a network, optimization fails: on the SAD surface the runs end, in the median, farther from the reference optimum than they started (w/ NSCC) or do not converge (w/o NSCC) (Extended Data Table 2). The one-step surfaces of the networks thus place their minima where the SCF surface does and are as easy to optimize on. This agreement comes from the accuracy of the predicted density, and near minima it does not depend on the NSCC.

## Molecular dynamics

Molecular dynamics probes the one-step surface farther from its minima. We ran 1 ps of constantenergy (NVE) dynamics of alanine dipeptide from four conformational basins, C7eq, C5, C7ax and αL (Fig. 3), with forces from the SCF reference or from the one-step calculation with the NSCC. With 10 heavy atoms, the dipeptide has more than any QM9 molecule, and near room temperature it samples distorted structures unlike any in QM9.

The models reproduce the energy landscape and the motion on it. On the $( \varphi , \ \psi )$ map, the relative one-step energies deviate from the SCF ones by 1.5 meV for SwinUNETR-V2 and 1.3 meV for NeXtUNETR on average within 1 eV of the minimum (Fig. 3a). As in the reference, the trajectories from C7eq, C5 and C7ax stay in their basins, and those from αL leave it in the same direction; individual trajectories are not expected to coincide, because nearby ones diverge.

a  
![](images/0edfb71c495a914e80f74833906733dd0af1b57d6705b8ed9331abe9ceea12ec.jpg)

![](images/6bc189b4f4c459b954f198339460f383411cbb08b7f0f16166e9b39c78fad470.jpg)  
c

![](images/95d4fc4438335550dcbe4edfb978706c800cfac3cada7afeda662c92fef7c8fc.jpg)

b  
![](images/f39bc7e82063d7ce77feaa0f8b6093cd5f256711f5f8c033be9ea79863f24a55.jpg)

![](images/438dc040f8da96464b27664b4d262c552ad6535f683d82cdaa87431647c50310.jpg)

d  
![](images/d9113f3f9194ed8a606a243eae339c0aa1744ce5044720d2d19e3070e24cdb27.jpg)  
Fig. 3 | Molecular dynamics with one-step forces. 1 ps NVE trajectories of alanine dipeptide from four basins (C7eq, C5, C7ax, αL), with forces from the SCF reference or from one Kohn–Sham step on the SwinUNETR-V2 or NeXtUNETR density, w/ NSCC; all runs from a basin share their initial velocities (Methods). a, Left: backbone dihedrals $\varphi$ (C<sup>′</sup>–N–Cα–C) and ψ (N–Cα–C–N<sup>′</sup>). Middle: the reference trajectories on the SCF $( \varphi , \psi )$ energy map (contours at 0.15, 0.3, 0.5 and 0.75 eV); stars, starting structures. Right: error of each model’s map (model − SCF, each relative to its own minimum; color scale saturates at ±5 meV) with that model’s trajectories and the 0.3 and 0.75 eV SCF contours; hatched, more than 1 eV above the reference minimum. b, Vibrational density of states (VDOS) of the C7eq trajectories (reference in gray). $^ { \mathbf { c } , \mathbf { d } , }$ , Error of the one-step energy (c) and forces (d; mean absolute error over force components) on the frames of the $\mathrm { C 7 e q }$ reference trajectory; light lines, every frame; dark lines, 25 fs running means. Extended Data Fig. 3 shows energy conservation and the dynamics w/o NSCC.

The vibrational density of states (VDOS), which reflects the curvature of the surface along the trajectory, has the same bands for both models as for the reference (Fig. 3b); the amide I and amide II band centroids lie within $9 \mathrm { c m } ^ { - 1 }$ of the reference in all four trajectories, below the $3 3 \mathrm { c m } ^ { - 1 }$ resolution of a 1 ps trajectory.

Along the reference C7eq trajectory, the one-step energies of SwinUNETR-V2 and NeXtUNETR are ofset from the SCF energy by +17.3 and +8.9 meV on average (Fig. 3c), and their force errors, 38.7 and 19.7 meV $\textup { \AA } ^ { - 1 }$ , do not grow with time (Fig. 3d). The errors exceed those on QM9, as expected for distorted structures, and NeXtUNETR’s are about half of SwinUNETR-V2’s in all four trajectories.

With the NSCC, the models conserve energy nearly as well as the reference: the standard deviation of the total energy is 0.06–0.10 meV atom<sup>−1</sup>, against 0.05–0.07 meV atom<sup>−1</sup> (Extended Data Fig. 3). Without the NSCC, the deviation is 20–70 times larger, and the systems heat by 24–97 K. Energy conservation shows that the forces are consistent with the one-step energy, and the small energy errors show that this energy stays close to the SCF energy: one Kohn–Sham step can replace the SCF calculation in molecular dynamics of this molecule outside the training distribution.

## Transfer beyond QM9

We tested transfer on four test sets (Table 2), three from Li et al. <sup>8</sup> and one new: thermally distorted $\mathrm { C _ { 7 } O _ { 2 } H _ { 1 0 } }$ isomers from molecular dynamics; charged biomolecular fragments containing phosphorus and magnesium, both absent from QM9 (the MT set); water clusters of 30 molecules (W1500/30), three times larger than the largest QM9 molecule; and donor–bridge–acceptor (D–B–A) chains, in which an electron donor and an acceptor at the two ends polarize the density along a saturated or conjugated bridge. The test chains have longer bridges than any training chain, so this set tests extrapolation to longer molecules. Following Li et al., each model was applied directly (zero-shot), fine-tuned on a small target set and trained from scratch on that set.

After fine-tuning, SwinUNETR-V2 and NeXtUNETR are more accurate than SR-ResNet on all four test sets, by up to a factor of 2.0 on the D–B–A chains (0.075 against 0.149%). NeXtUNETR is the most accurate on $\mathrm { { C } _ { 7 } \mathrm { { O } _ { 2 } \mathrm { { H } _ { 1 0 } } } }$ and D–B–A, and the two are within 6% of each other on MT and W1500/30. Trained from scratch, SwinUNETR-V2 is the most accurate model on every test set.

Without fine-tuning, the most accurate model depends on the test set: SwinUNETR-V2 on MT and W1500/30, NeXtUNETR on D–B–A and SR-ResNet on $\mathrm { C _ { 7 } O _ { 2 } H _ { 1 0 } }$ . NeXtUNETR, the most accurate model on QM9, is the least accurate zero-shot on MT. Fine-tuning removes most of the diferences between the two networks. Pretrained on QM9, SwinUNETR-V2 and NeXtUNETR adapt to new systems from 110–353 structures better than SR-ResNet and extrapolate to longer D–B–A bridges; trained from scratch on the same sets, every model is far less accurate.

## Size transfer to water clusters

To test zero-shot transfer in size, we built Wmin-PBE/3–100, a set of 53 water clusters of 3–100 molecules (Fig. 4). Up to 30 molecules, they are the putative global minima of the WDBase database<sup>41,42</sup>; larger ones were grown from the 30-molecule minimum one water at a time with the TIP4P model<sup>43</sup>, and all were relaxed with PBE (Methods). The largest has 300 atoms, ten times as many as the largest QM9 molecule. SwinUNETR-V2’s NMAE is nearly independent of size,

Table 2 | Transfer beyond QM9. Density NMAE (%) of the QM9 models applied directly (zero-shot) and after fine-tuning on the target training set, and of the same architectures trained from scratch on that set. One training run per fine-tuned or from-scratch entry (protocols in Methods). Best value per row in bold.
<table><tr><td>Test set</td><td>Training</td><td>SR-ResNet</td><td>SwinUNETR-V2</td><td>NeXtUNETR</td></tr><tr><td> $\mathrm { { C _ { 7 } O _ { 2 } H _ { 1 0 } } ^ { a } }$ </td><td>zero-shot</td><td>0.662</td><td>0.853</td><td>0.809</td></tr><tr><td rowspan="5"> $\mathrm { M T ^ { b } }$ </td><td>fine-tuned</td><td>0.244</td><td>0.228</td><td>0.167</td></tr><tr><td>from scratch</td><td>1.885</td><td>1.064</td><td>1.706</td></tr><tr><td>zero-shot</td><td>2.168</td><td>1.518</td><td>3.501</td></tr><tr><td>fine-tuned</td><td>0.339</td><td>0.214</td><td>0.203</td></tr><tr><td>from scratch</td><td>3.478</td><td>0.993</td><td>1.931</td></tr><tr><td rowspan="3"> $\mathrm { W 1 5 0 0 / 3 0 ^ { c } }$ </td><td>zero-shot</td><td>0.723</td><td>0.432</td><td>0.637</td></tr><tr><td>fine-tuned</td><td>0.127</td><td>0.088</td><td>0.089</td></tr><tr><td>from scratch</td><td>1.283</td><td>0.403</td><td>0.981</td></tr><tr><td rowspan="3"> $\mathrm { D - B { - } A ^ { d } }$ </td><td>zero-shot</td><td>0.299</td><td>0.356</td><td>0.250</td></tr><tr><td>fine-tuned</td><td>0.149</td><td>0.093</td><td>0.075</td></tr><tr><td>from scratch</td><td>0.725</td><td>0.540</td><td>0.860</td></tr></table>

<sup>a</sup> Ab initio molecular dynamics snapshots of 113 isomers; 4,520 test and 226 training structures.  
<sup>b</sup> Charged biomolecular fragments containing P and Mg; 110 test and 110 training structures.  
<sup>c</sup> The three 30-molecule clusters of W1500 (WDBase minima); trained on 3–13-molecule clusters (226 structures).  
<sup>d</sup> Donor–bridge–acceptor molecules; 144 test (C10, C12 bridges) and 353 training structures (C2–C8 bridges).

![](images/3bdcae08a455b54e3c716ce3a9179a407031b1d156d992db89746f34fd65847f.jpg)  
Fig. 4 | Zero-shot transfer to water clusters far larger than any training molecule. Density NMAE on Wmin-PBE/3–100 (log–log axes), 53 PBE-optimized water clusters: one of each size from 3 to 50 molecules and clusters of 60, 70, 80, 90 and 100 molecules, up to 300 atoms (right: the 100-molecule cluster, with hydrogen bonds dotted). None of the models was trained on water. Solid lines, models trained on the full QM9 training set; dashed lines, the same architectures trained on QM9-subset-8k (SwinUNETR-V2-292M, NeXtUNETR-67M and SR-ResNet-9M of Fig. 1).

0.50–0.59% in each size band (3–10, 11–30, 31–50 and 60–100 molecules), whereas SR-ResNet’s rises from 0.72 to 1.10%. NeXtUNETR is the most accurate on the smallest clusters, 0.46% for 3–10 molecules, but degrades beyond about 30 molecules, reaching 2.6% for 60–100 molecules.

This degradation does not appear to be intrinsic to the architecture. Trained on QM9-subset-8k, NeXtUNETR and SwinUNETR-V2 both have nearly flat errors, 0.68 and 0.69% over all clusters, whereas SR-ResNet degrades to 3.4% at 100 molecules (Fig. 4, dashed). During NeXtUNETR’s full-QM9 training, the NMAE on 60–100-molecule clusters fell to 0.68% after 20 epochs and then rose to 2.6% by the end of training, while that on small clusters fell to 0.46% (Extended Data Fig. 4). As its accuracy on QM9 improves, NeXtUNETR thus appears to specialize to the molecular sizes of QM9.

## Discussion

Transformer-inspired hierarchical networks predict electron densities accurately enough that a single Kohn–Sham step reproduces the self-consistent electronic structure, from the total energy and forces to the orbital energies. The one-step forces recover the SCF minima in geometry optimization and drive energy-conserving molecular dynamics of a molecule with more heavy atoms than any training molecule. On QM9, NeXtUNETR outperforms SwinUNETR-V2 at lower cost.

Two features of the design matter. The Transformer-style hierarchy, which starts at half resolution and holds most parameters on coarse grids, keeps the cost per parameter low. The block design determines how well the parameters are used: in the same U-shaped network, with the same feedforward layers, ConvNeXt blocks outperform Swin blocks that mix a $7 \times 7 \times 7$ neighborhood by attention (NMAE 0.133 against 0.250%). In computer vision, too, much of the advantage of Transformers comes from their macro-design and block recipe rather than from self-attention itself<sup>26–28</sup>. Attention may still help away from the training distribution, where SwinUNETR-V2 transfers best without fine-tuning to the charged fragments and the water clusters, but the size transfer of NeXtUNETR changes with its training (Extended Data Fig. 4), so single training runs cannot settle this.

The NSCC makes one-step forces usable for dynamics and lowers their error for most molecules. Minima are found equally well without it, but the uncorrected force is not the gradient of the one-step energy, and trajectories heat up. The NSCC costs one network backward pass per force evaluation.

A one-step evaluation costs far more than an MLIP evaluation, so this route complements $\mathrm { M L I P s ^ { 5 , 6 } }$ rather than replacing them. In exchange, it keeps the Kohn–Sham Hamiltonian: orbital energies, the density and the properties derived from them come with the energy and forces, from one model trained only on densities. Unlike an SCF calculation, it has a fixed cost and is diferentiable end to end. One-step surfaces could generate or check training data for MLIPs, or drive simulations that need electronic properties along the trajectory.

The study has limitations. Each model was trained once per condition, so diferences of a few percent may lie within seed variation, and each molecular dynamics condition was run once, for 1 ps. One diagonalization can fail for near-degenerate frontier orbitals, as in two of the 10,000 QM9 molecules. All calculations used closed-shell PBE valence densities with pseudopotentials for molecules in periodic cells. Training on of-equilibrium structures and applying the approach to extended systems and other functionals are the next steps.

## Methods

## Reference densities

All densities are valence densities from Kohn–Sham density functional theory with the PBE functional<sup>44</sup>, computed with $\mathrm { P y S C F ^ { 4 5 } }$ in the Gaussian-and-plane-wave (GPW) scheme<sup>46</sup>: Goedecker– Teter–Hutter (GTH) pseudopotentials<sup>47</sup> with PBE parameters<sup>48</sup>, GTH-TZV2P basis sets<sup>49</sup> and a plane-wave cutof of 200 Ha (400 Ry) for the density. Valence electron counts are H 1, C 4, N 5, O 6, F 7, P 5 and Mg 10. Each molecule is placed with the centroid of its atoms at the center of an orthorhombic periodic cell whose edges are its extent along each axis plus $4 \textup { \AA } .$ , rounded up to an even number of grid intervals. The Hartree potential is evaluated by fast Fourier transform with the $\mathbf { G } = 0$ term omitted, and no Coulomb truncation is applied. The self-consistent field (SCF) is converged to $1 0 ^ { - 1 1 }$ Ha. The density is stored on the real-space grid of the calculation, whose spacing, π/20 bohr (0.083 Å), resolves plane waves up to the 200 Ha cutof.

QM9 (ref. 18) densities are those of Li et al.<sup>8</sup>, computed with this protocol on the B3LYP geometries of QM9, with their training, validation and test split of 123,835, 50 and 10,000 molecules (the split sizes of DeepDFT<sup>13</sup>). The $\mathrm { { C _ { 7 } O _ { 2 } H _ { 1 0 } } }$ and MT densities are also theirs; the MT labels use Fermi–Dirac smearing of 0.01 Ha and an SCF tolerance of 10<sup>−7</sup> Ha. We computed all other densities with PySCF 2.6.2 and a reimplementation of their generator, with the same settings, cell rule and grid; because PySCF 2.6.2 rounds the number of grid points per axis to an odd value, the grid size is fixed to the even value their version (PySCF 2.0.1) produces.

## Input density

The input to every network is a superposition of atomic densities (SAD) <sup>50</sup> sampled on the grid of the reference density, generated with PySCF 2.6.2 by a reimplementation of Li et al.’s procedure <sup>8</sup>. For each element, a free-atom density matrix is obtained in the minimal GTH-SZV basis from the one-electron Hamiltonian with the pseudopotential and spherically averaged occupations, and projected onto GTH-TZV2P. The atomic blocks are assembled into a block-diagonal density matrix, scaled to N valence electrons, evaluated on the grid and renormalized to N on the grid. For geometry optimization and molecular dynamics, the same density is built as a structure-factor sum, $\begin{array} { r } { \rho _ { \mathrm { S A D } } ( \mathbf { G } ) = \sum _ { A } \rho _ { Z _ { A } } ( \mathbf { G } ) e ^ { - i \mathbf { G } \cdot \mathbf { R } _ { A } } } \end{array}$ , of element densities $\rho _ { Z }$ precomputed in reciprocal space (G is a reciprocal-lattice vector and $Z _ { A }$ the element of atom $\mathrm { A } )$ , which makes it diferentiable with respect to the nuclear positions $R _ { A }$ by automatic diferentiation.

## Network architectures

Output. Both networks predict a correction to the SAD density. The input $\rho _ { \mathrm { S A D } }$ is centered in a zero-padded grid of the next multiple of 32 voxels (at least 64) along each axis, passed through the backbone B and a $1 \times 1 \times 1$ convolution head $h ,$ and added back: $\tilde { \rho } = \mathrm { R e L U } ( \rho _ { \mathrm { S A D } } + h ( B ( \rho _ { \mathrm { S A D } } ) ) )$ ， cropped to the original grid. The output is rescaled to the number of valence electrons, $\rho _ { \mathrm { M L } } =$ $\tilde { \rho } { N } / ( \Delta V \sum _ { \mathbf { r } } \tilde { \rho } ( \mathbf { r } ) )$ , where $\Delta V$ is the voxel volume. The ReLU makes the density non-negative and the rescaling conserves the electron count exactly (Extended Data Fig. 1a).

SwinUNETR-V2. We used the $\mathrm { M O N A I ^ { 5 1 } }$ implementation of SwinUNETR-V2 (ref. 24). A $2 \times 2 \times 2$ patch embedding maps the input to feature size C at half resolution; four encoder stages follow, each opened by a residual block of two $3 \times 3 \times 3$ convolutions (the V2 addition)

and built from Swin Transformer blocks <sup>21</sup> that apply multi-head self-attention within $7 \times 7 \times$ 7-voxel windows, shifted by 3 voxels in alternate blocks. Patch merging after each stage halves the resolution and doubles the channels, giving $C \cdot \{ 1 , 2 , 4 , 8 , 1 6 \}$ channels down to $1 / 3 2$ of the input resolution. The stage depths are (5, 4, 2, 2), with (3, 6, 12, 24) attention heads; the depths were chosen in a preliminary search. As in MONAI’s implementation of SwinUNETR <sup>23,51</sup>, the skip connections carry the input, the patch-embedded input and the outputs of the first three stages, and the stage-4 output feeds the bottleneck; all but the input are first layer-normalized without learned parameters. The input, the patch-embedded input and the outputs of the first two stages then pass through such residual blocks; the stage-3 output, at $1 / 1 6$ resolution, enters the decoder without one, and the stage-4 output passes through one as the bottleneck. The decoder upsamples by transposed convolution, concatenates the skip features and applies a residual block, and a final $1 \times 1 \times 1$ convolution maps its C output channels to the single channel passed to h. Where a residual block changes the channel count, its shortcut is ${ \mathrm { ~ a ~ 1 ~ } } \times { \mathrm { ~ 1 ~ } } \times { \mathrm { ~ 1 ~ } }$ convolution with group normalization. The convolutional blocks use group normalization<sup>52</sup> with $C / 3$ groups instead of MONAI’s default instance normalization.

NeXtUNETR. NeXtUNETR keeps the macro-design of SwinUNETR- $\cdot \mathrm { V 2 } \mathrm { ~ - ~ } \mathrm { a ~ 2 } \times 2 \times 2$ patchify stem to half resolution, four stages of depths (5, 4, 2, 2) with $C \cdot \{ 1 , 2 , 4 , 8 \}$ channels and a 16C bottleneck at $1 / 3 2$ resolution — and replaces each Swin block by a three-dimensional ConvNeXt block<sup>26</sup>: a $7 \times 7 \times 7$ depthwise convolution (zero-padded), group normalization $( C / 3$ groups), a $1 \times 1 \times 1$ convolution expanding the channels fourfold, GELU, a $1 \times 1 \times 1$ projection and a residual connection, without layer scale or stochastic depth. The stem and the downsampling between stages are strided $2 \times 2 \times 2$ convolutions followed by group normalization and GELU; the bottleneck is one ConvNeXt block. NeXtUNETR has none of the $3 \times 3 \times 3$ residual blocks of SwinUNETR-V2: each stage begins directly with its ConvNeXt blocks, and the skip connections take each stage’s output unchanged. Each decoder level applies a transposed convolution, concatenation with the skip features, a $1 \times 1 \times 1$ fusion convolution and one ConvNeXt block, and a final transposed convolution returns to the input resolution before the $1 \times 1 \times 1$ head.

SR-ResNet. The baseline is the super-resolution residual network of Li et al. <sup>8</sup>: 32 residual blocks of 32 channels $( 5 \times 5 \times 5$ convolutions with circular padding) applied to the SAD density sampled at every second grid point, followed by twofold pixel-shufle upsampling to the full grid. It does not add the input density to its output, and it rescales the output to the integral of its subsampled input (times eight) instead of to $N ;$ we evaluate it as published, without further rescaling. Table 1 uses Li et al.’s released QM9 model; for Fig. 1 the same architecture was retrained with their schedule (see QM9-subset-8k below).

## Training

The networks were implemented in PyTorch<sup>53</sup> (version 2.3.1) with $\mathrm { M O N A I ^ { 5 1 } }$ (version 1.3.2). SwinUNETR-V2 and NeXtUNETR were trained with the same settings. The loss is the permolecule normalized mean absolute error (NMAE) of the density on the unpadded grid (defined below, without the factor of 100). Training used $\mathrm { A d a m W ^ { 5 4 } }$ with learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 }$ and moment decay rates $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ , batch size 1, for 50 epochs. The first epoch uses a constant learning rate of $1 0 ^ { - 7 }$ as a warm-up; the learning rate then follows a cosine decay from $1 0 ^ { - 4 }$ toward zero over the remaining 49 epochs, stepped once per epoch. Each training molecule is augmented on the fly, identically in input and label, by a rotation of the grid by multiples of $9 0 ^ { \circ }$ about each axis (drawn independently per axis) and, with probability 0.5, by an integer periodic shift of up to 4 voxels along each axis. During training the output is rescaled to the grid integral of the reference density, which equals N up to the discretization, and the rescaling factor is capped at 100; at inference the factor is not capped (the target equals N to within 10<sup>−8</sup>). Of the end-of-epoch checkpoints, we kept the one with the lowest NMAE on the 50 validation molecules; for both full-QM9 models this is the last epoch.

QM9-subset-8k. The 8,000 molecules were drawn from the QM9 training split, stratified by size (five bins) and composition (molecules with F, and the CHNO, CHO, CHN and CH classes), with proportional allocation and at least five molecules per stratum (fixed random seed). The ten models of Fig. 1 and the ablation models of Extended Data Table 1 were trained on it for 50 epochs with the validation set above. SwinUNETR-V2 has widths C = 24, 48, 96 and 144 (18.5, 73.1, 291.7 and 655.9 million parameters) and NeXtUNETR C = 48, 96, 144 and 192 (17.5, 67.5, 150.1 and 265.4 million). SR-ResNet was trained with Li et al.’s schedule: $\mathrm { A d a m ^ { 5 5 } }$ $( ( \beta _ { 1 } , \ \beta _ { 2 } ) = ( 0 . 9 , \ 0 . 9 9 9 )$ , as in their released code) without weight decay, a first epoch at $1 0 ^ { - 5 }$ of the learning rate, then cosine decay over the remaining epochs, with their NMAE loss and augmentation. The learning rate was 0.1, the value Li et al. used for their QM9 variants (0.01 for their main model), and the best validation checkpoint was kept. SR-ResNet-70M has 64 channels and 64 residual blocks. The 3D U-Nets of Extended Data Table 1 have four levels, from the full grid down to 1/8 resolution, with 96, 192, 384 and 768 channels; each level applies two $3 \times 3 \times 3$ convolutions with circular padding and ReLU, downsampling is by max pooling and upsampling by transposed convolution, and the output is formed as for SwinUNETR-V2 and NeXtUNETR. The plain version has no normalization; the residual version adds a residual connection around each pair of convolutions and group normalization (16 groups). Both used the optimizer, learning rate and loss of SwinUNETR-V2 and NeXtUNETR. The plain ResNet applies the 32 residual blocks of SR-ResNet to the full grid, without subsampling or upsampling, and was trained like SR-ResNet. The 500 test molecules of Fig. 1 were drawn uniformly at random from the 10,000 test molecules (fixed seed).

Full QM9. The models of Table 1 and of all later Results sections were trained on all 123,835 training molecules at C = 96 (SwinUNETR-V2 291.7 and NeXtUNETR 67.5 million parameters; breakdown in Extended Data Fig. 1e).

## Density error, cost and precision

The density error is the normalized mean absolute error on the reference grid, in percent, NMAE = $\begin{array} { r } { 1 0 0 \times \sum _ { \mathbf { r } } \left| \rho _ { \mathrm { M L } } ( \mathbf { r } ) - \rho _ { \mathrm { r e f } } ( \mathbf { r } ) \right| / \sum _ { \mathbf { r } } \rho _ { \mathrm { r e f } } ( \mathbf { r } ) } \end{array}$ . For the one-step density, $\rho _ { \mathrm { M I } }$ is replaced by the density of the one-step density matrix on the same grid points.

All network inference reported in the paper is in fp32 with TF32 tensor-core arithmetic disabled. Networks were trained with PyTorch’s default, which enables TF32 for cuDNN convolutions. Switching it on changes the predictions of SwinUNETR-V2 and NeXtUNETR by at most 0.01% and their mean NMAE by at most 0.001 percentage points (Supplementary Table 1), and the mean NMAE of Li et al.’s published model on the 500 molecules of Fig. 1 from 0.135 to 0.136%; the SR-ResNets trained on QM9-subset-8k are more accurate when evaluated in TF32, the precision they were trained in (0.238 and 0.191% instead of 0.320 and 0.198%; Supplementary Table 1).

For Fig. 1, every model was run on the same 500 QM9 test molecules on one NVIDIA H200 NVL GPU (PyTorch 2.3.1, CUDA 12.1) with batch size 1 and the input already on the GPU.

The forward time is the network alone, without the construction of the SAD density or the Kohn– Sham step; each molecule was timed three times and the median kept, and the reported value is the mean over the 500 molecules. The inputs have 483 distinct grid shapes and cuDNN autotuning was disabled. Floating-point operations were counted with PyTorch’s FlopCounterMode on each input (convolutions, including depthwise and transposed ones, matrix products and attention); the operations not counted (normalization, activations, softmax and residual additions) contribute little.

## One-step Kohn–Sham calculation

From the predicted density $\rho _ { \mathrm { M L } }$ , the Kohn–Sham Hamiltonian matrix $\begin{array} { r } { H [ \rho _ { \mathrm { M L } } ] = T + V _ { \mathrm { n l } } + V _ { \mathrm { l o c } } + } \end{array}$ $V _ { \mathrm { H } } [ \rho _ { \mathrm { M L } } ] + V _ { \mathrm { x c } } [ \rho _ { \mathrm { M L } } ]$ is built with the same GPW integration as the reference: the Hartree potential by fast Fourier transform on the grid, the PBE potential through libxc <sup>56</sup> with density gradients from the same transform, the local pseudopotential on the grid and the kinetic and non-local pseudopotential terms analytically. The generalized eigenvalue problem $H C = S C \varepsilon$ is solved once and the lowest $N / 2$ orbitals are doubly occupied, $D = 2 C _ { \mathrm { o c c } } C _ { \mathrm { o c c } } ^ { \mathsf { T } }$ . The eigenvalues are the one-step orbital energies. The one-step energy is the Kohn–Sham energy functional of this density matrix,

$$
E _ { \mathrm { 1 - s t e p } } = \mathrm { T r } [ D ( T + V _ { \mathrm { n l } } ) ] + \int v _ { \mathrm { l o c } } \rho _ { D } \mathrm { d } \mathbf { r } + E _ { \mathrm { H } } [ \rho _ { D } ] + E _ { \mathrm { x c } } [ \rho _ { D } ] + E _ { \mathrm { N N } } ,
$$

where $\rho _ { D }$ is the density of D and $E _ { \mathrm { N N } }$ the nuclear repulsion, evaluated by Ewald summation. Because D is an idempotent density matrix with the correct number of electrons, $E _ { \mathrm { 1 - s t e p } } \geq E _ { \mathrm { S C F } }$ $( E _ { \mathrm { 1 - s t e p } } - E _ { \mathrm { S C F } } > 0$ for all 10,000 QM9 test molecules and all three models).

Neither PySCF nor GPU4PySCF provides the one-step energy or its gradient. The one-step calculation and all forces described below are our own code, which calls the libraries’ routines for the grid density, the potentials, the integrals and their nuclear derivatives. The QM9 energies, orbital energies and forces (Table 1, Fig. 2a–c) were computed on the CPU with the $\mathrm { P y S C F }$ build distributed by Li et al. (PySCF 2.0.1 with libxc 5.1.7), with which their reference energies are reproduced exactly and their published SR-ResNet errors are recovered (NMAE 0.138% against $0 . 1 4 \% ; \varepsilon _ { E }$ 2.84 against 3.0 meV); with the reference density as input, the one-step energy difers from the SCF energy by 0.001 meV on average. Geometry optimization and molecular dynamics used our GPU implementation built on $\mathrm { G P U 4 P y S C F } ^ { 5 7 , 5 8 }$ (version 1.8.0, with $\mathrm { P y S C F 2 . 1 4 . 0 , C u P y }$ 14.1.1 and CUDA 12). In this implementation, the network and the diagonalization stay on the 200 Ha grid of the training data. $E _ { \mathrm { 1 - s t e p } }$ , the force terms at fixed D and the response amplitudes are evaluated on a finer grid of 400 Ha $( { \sqrt { 2 } }$ times as many points per axis, rounded up to an even number); the derivative of $H [ \rho _ { \mathrm { M L } } ]$ and the vector–Jacobian product stay on the 200 Ha grid. The reference SCF runs entirely at 400 Ha with GPU4PySCF’s multigrid Kohn–Sham solver <sup>59</sup>. The finer grid reduces the egg-box artifact <sup>60</sup> of a grid fixed in space by a factor of 13–16 in energy and 5–6 in force, and the force remains the exact gradient of the reported energy (Supplementary Note 1 and Supplementary Fig. 2). These checks used an earlier CPU counterpart of the GPU implementation, run with PySCF 2.6.2, that evaluated the matrix G of the NSCC (defined below) on the 200 Ha grid; its later version, which evaluates G on the 400 Ha grid as the GPU implementation does, gives the CPU timings of Supplementary Table 2.

Within each study, the reference and the models use the same implementation and grid. Given the same density matrix, the CPU (PySCF 2.0.1) and GPU implementations difer at 200 Ha by 0.2–3.4 meV in energy and 0.3–6.9 meV $\textup { \AA } ^ { - 1 }$ in force (component mean absolute error, MAE;

three structures), mainly through the discretization of the density on the multigrid; at 400 Ha, SCF calculations of alanine dipeptide (C7eq and C5) with the two implementations agree within 0.41 meV in energy and 0.1 meV $\textup { \AA } ^ { - 1 }$ in force (component MAE).

## Forces and the non-self-consistency correction

The one-step energy depends on the nuclear positions R through the basis functions, the pseudopotentials and the nuclear repulsion, and through the input density $\rho _ { \mathrm { M L } } ( R ) = f _ { \theta } ( \rho _ { \mathrm { S A D } } ( R ) )$ which in turn determines D. The force without the non-self-consistency correction (NSCC), denoted $\mathrm { w / o \ N S C C } .$ is minus the standard Kohn–Sham gradient evaluated for the one-step solution: the Hellmann–Feynman, pseudopotential, kinetic and basis-function (Pulay) derivatives contracted with $D ,$ the energy-weighted overlap term −Tr(W ∂S/∂R) with $\begin{array} { r } { W = 2 \sum _ { i \in \mathrm { o c c } } \varepsilon _ { i } c _ { i } c _ { i } ^ { \mathsf { T } } } \end{array}$ , and the nuclear repulsion. It treats D as if it were self-consistent, neglecting the response of D to $R ,$ both explicit and through the input density.

The force with the NSCC $\left( \mathrm { w } \right)$ NSCC) is the exact gradient $- { \mathrm { d } E _ { 1 - \mathrm { s t e p } } } / { \mathrm { d } R } .$ With $\begin{array} { r l } { G } & { { } = } \end{array}$ $\partial E _ { \mathrm { 1 - s t e p } } / \partial D = H [ \rho _ { D } ]$ , the Kohn–Sham Hamiltonian matrix of the output density, $\mathrm { d } E _ { 1 - \mathrm { s t e p } } / \mathrm { d } R =$ $\partial E _ { \mathrm { 1 - s t e p } } / \partial R | _ { D } + \mathrm { T r } [ G \mathrm { d } D / \mathrm { d } R ]$ . Because H is built from $\rho _ { \mathrm { M L } }$ and not from D, dD/dR follows from uncoupled first-order perturbation theory,

$$
\mathrm { T r } \bigg [ G \frac { \mathrm { d } D } { \mathrm { d } R } \bigg ] = 4 \sum _ { i \in \mathrm { o c c } , a \in \mathrm { v i r t } } G _ { a i } \frac { H _ { a i } ^ { R } - \varepsilon _ { i } S _ { a i } ^ { R } } { \varepsilon _ { i } - \varepsilon _ { a } } - 2 \sum _ { i , j \in \mathrm { o c c } } G _ { i j } S _ { i j } ^ { R } ,
$$

where $H ^ { R }$ and $S ^ { R }$ are the total derivatives of $H [ \rho _ { \mathrm { M L } } ]$ and of the overlap matrix $S$ with respect to $R ,$ in the basis of the orbitals $C .$ The force $\mathrm { w / o }$ NSCC replaces this term by the overlap term $2 \textstyle \sum _ { i \in \mathrm { o c c } } \varepsilon _ { i } S _ { i i } ^ { R }$ , its value at self-consistency; the NSCC is the diference. It vanishes when the input density is self-consistent, since then $G = H [ \rho _ { \mathrm { M L } } ] , G _ { a i } = 0$ and $G _ { i j } = \varepsilon _ { i } \delta _ { i j }$ . The explicit part of $H ^ { R }$ (basis functions and pseudopotentials at fixed $\rho _ { \mathrm { M L } } )$ is evaluated analytically by contracting the amplitude matrix $P = C _ { \mathrm { v i r t } } z C _ { \mathrm { o c c } } ^ { \mathsf { T } } , z _ { a i } = 4 G _ { a i } / ( \varepsilon _ { i } - \varepsilon _ { a } )$ , with the integral derivatives. The implicit part, through ρ<sub>ML</sub>, is $\begin{array} { r l } { \int w ( \mathbf { r } ) \partial \rho _ { \mathrm { M L } } ( \mathbf { r } ) / \partial R \mathrm { d } \mathbf { r } } & { { } } \end{array}$ with the cotangent field $w = v _ { \mathrm { H } } [ \rho _ { P } ] + f _ { \mathrm { x c } } [ \rho _ { \mathrm { M L } } ] \rho _ { P }$ (with the corresponding gradient terms of the PBE kernel), where $\rho _ { P }$ is the density of P. This integral is obtained by a vector–Jacobian product (one backward pass) through the rescaling and the network, followed by the derivative of the SAD density.

For the SAD control of Extended Data Table 2, the same expressions are applied with $\rho _ { \mathrm { M L } }$ replaced by $\rho _ { \mathrm { S A D } }$ . For SR-ResNet the vector–Jacobian product follows its own input (every second grid point) and output (no rescaling to N) conventions.

Every term was checked against finite diferences of $E _ { \mathrm { 1 - s t e p } }$ . End to end, analytic and centraldiference forces (step 0.002 Å) agree within 0.03–1.8 meV $\textup { \AA } ^ { - 1 }$ (largest component deviation, at most $0 . 4 \%$ of the component) on QM9 molecules and within 0.36 meV $\textup { \AA } ^ { - 1 }$ (component MAE) for alanine dipeptide; for the SAD control $( \mathrm { s t e p ~ 0 . 0 0 1 ~ \AA } )$ their ratio is 0.9995–1.0001. The QM9 reference forces are the gradient $\mathrm { w / o }$ NSCC of the one-step calculation with the reference density as input, which agrees with $\mathrm { P y S C F ^ { \prime } s }$ own SCF gradient in Li et al.’s build within 0.006–0.2 meV $\textup { \AA } ^ { - 1 }$ (largest component deviation) on five molecules. In the GPU implementation, the reference force is likewise our gradient (the $\mathrm { w / o }$ NSCC terms, exact at self-consistency) applied to the converged SCF density matrix, so that reference and model forces pass through the same code; it agrees with GPU4PySCF’s own SCF gradient within 0.0003 meV $\textup { \AA } ^ { - 1 }$ (component MAE). Evaluating the SR-ResNet and SwinUNETR-V2 force errors on 30 QM9 molecules at 400 instead of 200 Ha changes the per-molecule force MAEs by at most 0.16 meV $\textup { \AA } ^ { - 1 }$ , so the reported errors reflect the predicted density, not the integration grid. QM9 forces are reported as computed, without removing the net force.

## Geometry optimization

The 200 QM9 test molecules are the first, in a fixed random order, for which RDKit <sup>61</sup> assigned the bonds, an MMFF94 (ref. 40) starting structure could be built, the cell converged (below) and no heavy-atom dihedral difered by more than $6 0 ^ { \circ }$ between the starting structure and the reference optimum.

For each molecule, the reference optimum $x _ { \mathrm { r e f } }$ is the SCF minimum in a cell built by the rule of “Reference densities”, obtained from the QM9 geometry by alternating SCF relaxation (with the optimizer and criteria below) and cell rebuilding until the cell no longer changed (at most three passes), and the starting structure is an MMFF94 relaxation of $x _ { \mathrm { r e f } }$ with RDKit, Kabsch-aligned<sup>62</sup> to it. From this start, the structure was optimized on each surface of Extended Data Table 2 with geomeTRIC<sup>39</sup> (version 1.1.1) in translation–rotation internal coordinates, with its default trust radius and convergence criteria (maximum force 23.1 meV $\textup { \AA } ^ { - 1 }$ , RMS force 15.4 meV $\textup { \AA } ^ { - 1 }$ , energy change $1 0 ^ { - 6 }$ Ha, maximum and RMS displacement $1 . 8 \times 1 0 ^ { - 3 }$ and $1 . 2 \times 1 0 ^ { - 3 } \mathrm { ~ \AA ~ }$ , and at most 300 optimization steps. Net translation and rotation were projected out of every force. All runs used the 400 Ha GPU implementation; the reference SCF was converged to 10<sup>−11</sup> Ha from an atomic guess at every step, and the networks ran in fp32. Distances are all-atom Kabsch rootmean-square deviations (RMSDs) from the optimum the reference run reached from the same start (median 0.116 Å for the starting structures).

On each network surface, 7–14 molecules ended more than 0.05 $\textrm { \AA }$ from the reference optimum. For the 11 such molecules with SwinUNETR-V2 w/ NSCC, restarting each optimization from the other surface’s end point showed that the two runs ended in diferent minima along soft torsions, which in all but one case exist on both surfaces.

## Molecular dynamics

Alanine dipeptide (Ace–Ala–NMe, 22 atoms) was started from its four gas-phase basins C7eq, C5, C7ax and αL; each starting structure was built with MMFF94, the backbone dihedrals restrained to literature values, relaxed again without the restraint and then relaxed on the PBE/GTH surface in the cell below (200 Ha grid) to a maximum force of 20 meV $\textup { \AA } ^ { - 1 }$ , which changed $\varphi$ and ψ by at most 3.3<sup>◦</sup>. The cell is cubic, 13.30 Å $( 1 6 0 ^ { 3 } ~ \mathrm { g r i d } )$ , large enough for the molecule to rotate.

Trajectories were integrated with velocity Verlet <sup>63</sup> in the NVE ensemble for 2,000 steps of 0.5 fs (1 ps). Initial velocities were drawn from a Maxwell–Boltzmann distribution at 600 K (one random seed shared by all methods for a basin) with the net momentum removed; by equipartition the trajectories settle at 258–284 K (except the runs $\mathrm { w / o }$ NSCC, which heat up). Forces came from the reference SCF (400 Ha, 10<sup>−11</sup> Ha, started from the previous step’s density matrix) or from the one-step calculation on the SwinUNETR-V2 or NeXtUNETR density, with or without the NSCC. Because the grid is fixed in space, the forces do not sum exactly to zero; rather than projecting out the net force, which would break consistency with the reported energy, we recentered the centroid in the cell and removed the center-of-mass velocity at every step.

Energy conservation is measured by the standard deviation of the total energy per atom over the second half of each trajectory (not detrended). The vibrational density of states is the Fourier transform of the windowed, mass-weighted velocity autocorrelation function (velocities every 2.5 fs; resolution $3 3 ~ \mathrm { c m } ^ { - 1 } )$ ; band positions are intensity-weighted centroids over fixed windows (amide II, 1,250–1,550 cm<sup>−1</sup>; amide I, 1,550–1,800 cm<sup>−1</sup>). The $( \varphi , \psi )$ maps were computed on a $1 5 ^ { \circ }$ grid (576 points) of MMFF94 structures relaxed with the two dihedrals constrained, with the SCF and each model’s one-step energy (with the implementation and cell of the dynamics), each relative to its own minimum; mean absolute map errors are taken over the 541 points within 1 eV of the SCF minimum. The per-frame errors of Fig. 3c,d are the one-step energies and forces (w/ NSCC) evaluated on the 400 frames (every 2.5 fs) of each reference trajectory.

## Single-point timing

Single-point wall times (Supplementary Table 2) were measured on the 200 QM9 molecules of the geometry-optimization set, each at its reference-SCF minimum, on one NVIDIA A40 (48 GB), A100 SXM4 (80 GB) or H100 NVL GPU and on 16 cores of an AMD EPYC 7543 CPU. On each device the reference SCF and the one-step calculation used the same implementation: on the GPUs that of the geometry optimizations, on the CPU the PySCF 2.6.2 implementation above. Both used the settings of the geometry optimizations (SCF converged to 10<sup>−11</sup> Ha from the atomic guess on the 400 Ha grid). Each time is that of a new molecule in a running program; program start, model loading, file input and cell construction are excluded, and performance options that leave the results unchanged (forces within 0.02 meV $\mathrm { ~ \AA ~ } ^ { - 1 } )$ were used. Every calculation was run once per molecule, after a warm-up on a molecule outside the set and with per-molecule caches emptied before each calculation, and the table reports the median over the 200 molecules. Network forward and backward passes are also reported separately.

## Transfer benchmarks

Datasets. $C _ { 7 } O _ { 2 } H _ { 1 0 } { } ^ { 8 } \colon$ 113 isomers × 50 snapshots from ab initio molecular dynamics; following Li et al., the first two snapshots of each trajectory are used for training (226), the third for validation (113) and the last 40 for testing (4,520). $M T ^ { 8 }$ : charged biomolecular fragments from a GTPbound microtubule (total charge −4) containing C, H, N, O, P and Mg; the first 10 structures of each of Li et al.’s runs are used for training (110) and the 11th for validation, and the test set is 10 structures per run drawn from their test range in ten index-quantile bins (110). W1500: water clusters from the WDBase database of water-cluster minima <sup>41,42</sup> (its 1 kcal mol<sup>−1</sup> files), taking the first min(count, $\lfloor 1 5 0 0 / n \rfloor )$ structures of each n-molecule file, which reproduces Li et al.’s counts; training on $n = 3 { - } 1 3 \ ( 2 2 6 )$ , validation on n = 14 (17) and testing on the three $n = 3 0$ clusters (W1500/30). Wmin-PBE/3–100 : for $n = 3 – 3 0$ the putative global minima of the TTM2.1-F potential listed in WDBase (for n = 26–30 from an incomplete search), and for $n = 3 1 \mathrm { - } 5 0 , 6 0 , 7 0$ 80, 90 and 100 clusters grown deterministically from the 30-molecule minimum, one rigid water at a time: among the $\mathrm { T I P 4 P ^ { 4 3 } }$ placements within 0.02 eV of the lowest-energy one, the one giving the smallest cell was kept, with rigid-body reorganization every five additions and a beam of three candidate structures. All Wmin-PBE clusters were relaxed on the PBE/GTH surface (200 Ha) to a maximum force of 50 meV $\textup { \AA } ^ { - 1 }$ ; W1500 uses the WDBase geometries as given. Both were labeled with the protocol above. D–B–A chains (constructed here): six donors (NMe<sub>2</sub>, NH<sub>2</sub>, OMe, OH, Me, F) and six acceptors (NO<sub>2</sub>, CN, CF<sub>3</sub>, CHO, COOH, $\mathrm { C O N H _ { 2 } ) }$ at the ends of saturated (alkane) and conjugated (trans-polyene) bridges of C2, C4, . . . , C12. The parent chains, built with

RDKit (ETKDG version $3 ^ { 6 4 }$ and MMFF94), were relaxed on the PBE/GTH surface (200 Ha) to a maximum force of 50 meV $\mathrm { \AA } ^ { - 1 } ;$ ; donor and acceptor substituents were relaxed with the bridge frozen (to the same threshold, except two $\mathrm { N M e _ { 2 } }$ fragments that stopped at 56 and 84 meV $\textup { \AA } ^ { - 1 }$ after 80 steps) and the doubly substituted molecules assembled from them. All molecules of a series (one bridge type and length) share one cell, centered on the bridge, whose edges are the largest extent in the series plus 4 Å. Models were trained on all C2–C8 molecules (parent, singly and doubly substituted; 353 training, 39 validation drawn at random) and tested on the 144 doubly substituted molecules with C10 and C12 bridges.

Protocols. Zero-shot results apply the QM9 models of Table 1 directly. Fine-tuning starts from them and trains for 50 epochs on the target training set, with fresh optimizer and scheduler states. SwinUNETR-V2 and NeXtUNETR use the loss, optimizer, schedule and augmentation of pretraining (AdamW, learning rate $1 0 ^ { - 4 } )$ , and SR-ResNet Li et al.’s protocol (Adam, learning rate 0.01, a first epoch at 10<sup>−5</sup> of that rate, then cosine decay, with the Adam settings above). Models trained from scratch use the same settings from random initialization. For every model, the end-ofepoch checkpoint with the lowest validation NMAE is evaluated. Wmin-PBE/3–100 was evaluated zero-shot only. On the transfer sets, only the density was evaluated.

## Statistics and reproducibility

Each model and condition was trained once, and each molecular dynamics condition was run once; no seed variation is reported. QM9 results include all 10,000 test molecules; the two molecules with a reference HOMO–LUMO gap below 1 eV are excluded only from the parenthesized means of Table 1 and from Extended Data Fig. 2, by that criterion alone.

## Data availability

The QM9, $\mathrm { C _ { 7 } O _ { 2 } H _ { 1 0 } }$ and MT reference densities are those of Li et al. $^ 8 \cdot$ , deposited in Caltech-DATA<sup>65–67</sup>; their released SR-ResNet models are on Figshare<sup>68</sup>. The densities computed here (the W1500 clusters used, Wmin-PBE/3–100 and the D–B–A chains), the structures and the source data will be made publicly available upon publication.

## Code availability

The model definitions, the training code, the one-step and force code and the analysis scripts will be made publicly available upon publication.

## References

1. Hohenberg, P. & Kohn, W. Inhomogeneous electron gas. Phys. Rev. 136, B864–B871 (1964).

2. Kohn, W. & Sham, L. J. Self-consistent equations including exchange and correlation efects. Phys. Rev. 140, A1133–A1138 (1965).

3. Behler, J. & Parrinello, M. Generalized neural-network representation of high-dimensional potential-energy surfaces. Phys. Rev. Lett. 98, 146401 (2007).

4. Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C. & Csányi, G. MACE: higher order equivariant message passing neural networks for fast and accurate force fields. In Adv. Neural Inf. Process. Syst. 35, 11423–11436 (2022).

5. Unke, O. T. et al. Machine learning force fields. Chem. Rev. 121, 10142–10186 (2021).

6. Batatia, I. et al. A foundation model for atomistic materials chemistry. J. Chem. Phys. 163, 184110 (2025).

7. Brockherde, F. et al. Bypassing the Kohn–Sham equations with machine learning. Nat. Commun. 8, 872 (2017).

8. Li, C., Sharir, O., Yuan, S. & Chan, G. K.-L. Image super-resolution inspired electron density prediction. Nat. Commun. 16, 4811 (2025).

9. Li, H. et al. Deep-learning density functional theory Hamiltonian for eficient ab initio electronic-structure calculation. Nat. Comput. Sci. 2, 367–377 (2022).

10. Shao, X., Paetow, L., Tuckerman, M. E. & Pavanello, M. Machine learning electronic structure methods based on the one-electron reduced density matrix. Nat. Commun. 14, 6281 (2023).

11. Prodan, E. & Kohn, W. Nearsightedness of electronic matter. Proc. Natl Acad. Sci. USA 102, 11635–11638 (2005).

12. Grisafi, A. et al. Transferable machine-learning model of the electron density. ACS Cent. Sci. 5, 57–64 (2019).

13. Jørgensen, P. B. & Bhowmik, A. Equivariant graph neural networks for fast electron density estimation of molecules, liquids, and solids. npj Comput. Mater. 8, 183 (2022).

14. Koker, T., Quigley, K., Taw, E., Tibbetts, K. & Li, L. Higher-order equivariant neural networks for charge density prediction in materials. npj Comput. Mater. 10, 161 (2024).

15. Fu, X. et al. A recipe for charge density prediction. In Adv. Neural Inf. Process. Syst. 37, 9727–9752 (2024).

16. Elsborg, J., Thiede, L., Aspuru-Guzik, A., Vegge, T. & Bhowmik, A. ELECTRA: a Cartesian network for 3D charge density prediction with floating orbitals. In Adv. Neural Inf. Process. Syst. 38, 31897–31926 (2025).

17. Lee, R.-G. & Kim, Y.-H. Convolutional network learning of self-consistent electron density via grid-projected atomic fingerprints. npj Comput. Mater. 10, 248 (2024).

18. Ramakrishnan, R., Dral, P. O., Rupp, M. & von Lilienfeld, O. A. Quantum chemistry structures and properties of 134 kilo molecules. Sci. Data 1, 140022 (2014).

19. Vaswani, A. et al. Attention is all you need. In Adv. Neural Inf. Process. Syst. 30 (2017).

20. Dosovitskiy, A. et al. An image is worth 16x16 words: transformers for image recognition at scale. In Proc. International Conference on Learning Representations (2021).

21. Liu, Z. et al. Swin Transformer: hierarchical vision transformer using shifted windows. In Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 9992–10002 (2021).

22. He, K., Zhang, X., Ren, S. & Sun, J. Deep residual learning for image recognition. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 770–778 (2016).

23. Hatamizadeh, A. et al. Swin UNETR: Swin transformers for semantic segmentation of brain tumors in MRI images. In Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2021), Lecture Notes in Computer Science Vol. 12962, 272–284 (Springer, 2022).

24. He, Y. et al. SwinUNETR-V2: stronger Swin transformers with stagewise convolutions for 3D medical image segmentation. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2023, Lecture Notes in Computer Science Vol. 14223, 416–426 (Springer, 2023).

25. Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T. & Ronneberger, O. 3D U-Net: learning dense volumetric segmentation from sparse annotation. In Medical Image Computing and Computer-Assisted Intervention – MICCAI 2016, Lecture Notes in Computer Science Vol. 9901, 424–432 (Springer, 2016).

26. Liu, Z. et al. A ConvNet for the 2020s. In Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 11966–11976 (2022).

27. Lee, H. H., Bao, S., Huo, Y. & Landman, B. A. 3D UX-Net: a large kernel volumetric ConvNet modernizing hierarchical transformer for medical image segmentation. In Proc. International Conference on Learning Representations (2023).

28. Roy, S. et al. MedNeXt: transformer-driven scaling of ConvNets for medical image segmentation. In Medical Image Computing and Computer Assisted Intervention – MICCAI 2023, Lecture Notes in Computer Science Vol. 14223, 405–415 (Springer, 2023).

29. Harris, J. Simplified method for calculating the energy of weakly interacting fragments. Phys. Rev. B 31, 1770–1779 (1985).

30. Foulkes, W. M. C. & Haydock, R. Tight-binding models and density-functional theory. Phys. Rev. B 39, 12520–12536 (1989).

31. Niklasson, A. M. N. Next generation extended Lagrangian first principles molecular dynamics. J. Chem. Phys. 147, 054103 (2017).

32. Niklasson, A. M. N. & Negre, C. F. A. Shadow energy functionals and potentials in Born– Oppenheimer molecular dynamics. J. Chem. Phys. 158, 154105 (2023).

33. Focassio, B., Domina, M., Patil, U., Fazzio, A. & Sanvito, S. Linear Jacobi-Legendre expansion of the charge density for machine learning-accelerated electronic structure calculations. npj Comput. Mater. 9, 87 (2023).

34. Grisafi, A., Lewis, A. M., Rossi, M. & Ceriotti, M. Electronic-structure properties from atomcentered predictions of the electron density. J. Chem. Theory Comput. 19, 4451–4460 (2023).

35. Hazra, S., Patil, U. & Sanvito, S. A charge-density machine-learning workflow for computing the infrared spectrum of molecules. J. Chem. Phys. 163, 174104 (2025).

36. Hazra, S., Patil, U. & Sanvito, S. Predicting the one-particle density matrix with machine learning. J. Chem. Theory Comput. 20, 4569–4578 (2024).

37. Klockow, M. V., Ickler, M. K., Lippmann, P. & Hamprecht, F. A. A function-centric graph neural network approach for predicting electron densities. In Proc. International Conference on Learning Representations https://openreview.net/forum?id=HDdkFjFEZd (2026).

38. Jørgensen, P. B. & Bhowmik, A. QM9 charge densities and energies calculated with VASP. DTU Data https://doi.org/10.11583/DTU.16794500 (2022).

39. Wang, L.-P. & Song, C. Geometry optimization made simple with translation and rotation coordinates. J. Chem. Phys. 144, 214108 (2016).

40. Halgren, T. A. Merck molecular force field. I. Basis, form, scope, parameterization, and performance of MMFF94. J. Comput. Chem. 17, 490–519 (1996).

41. Rakshit, A., Bandyopadhyay, P., Heindel, J. P. & Xantheas, S. S. Atlas of putative minima and low-lying energy networks of water clusters n = 3–25. J. Chem. Phys. 151, 214307 (2019).

42. Database of water cluster minima (WDBase). University of Washington https://sites.uw. edu/wdbase/database-of-water-clusters/ (2026).

43. Jorgensen, W. L., Chandrasekhar, J., Madura, J. D., Impey, R. W. & Klein, M. L. Comparison of simple potential functions for simulating liquid water. J. Chem. Phys. 79, 926–935 (1983).

44. Perdew, J. P., Burke, K. & Ernzerhof, M. Generalized gradient approximation made simple. Phys. Rev. Lett. 77, 3865–3868 (1996).

45. Sun, Q. et al. Recent developments in the PySCF program package. J. Chem. Phys. 153, 024109 (2020).

46. Lippert, G., Hutter, J. & Parrinello, M. A hybrid Gaussian and plane wave density functional scheme. Mol. Phys. 92, 477–487 (1997).

47. Goedecker, S., Teter, M. & Hutter, J. Separable dual-space Gaussian pseudopotentials. Phys. Rev. B 54, 1703–1710 (1996).

48. Krack, M. Pseudopotentials for H to Kr optimized for gradient-corrected exchange-correlation functionals. Theor. Chem. Acc. 114, 145–152 (2005).

49. VandeVondele, J. et al. Quickstep: fast and accurate density functional calculations using a mixed Gaussian and plane waves approach. Comput. Phys. Commun. 167, 103–128 (2005).

50. Van Lenthe, J. H., Zwaans, R., Van Dam, H. J. J. & Guest, M. F. Starting SCF calculations by superposition of atomic densities. J. Comput. Chem. 27, 926–932 (2006).

51. Cardoso, M. J. et al. MONAI: an open-source framework for deep learning in healthcare. Preprint at https://arxiv.org/abs/2211.02701 (2022).

52. Wu, Y. & He, K. Group normalization. In Computer Vision – ECCV 2018, Lecture Notes in Computer Science Vol. 11217, 3–19 (Springer, 2018).

53. Paszke, A. et al. PyTorch: an imperative style, high-performance deep learning library. In Adv. Neural Inf. Process. Syst. 32 (2019).

54. Loshchilov, I. & Hutter, F. Decoupled weight decay regularization. In Proc. International Conference on Learning Representations (2019).

55. Kingma, D. P. & Ba, J. Adam: a method for stochastic optimization. In Proc. International Conference on Learning Representations (2015).

56. Lehtola, S., Steigemann, C., Oliveira, M. J. T. & Marques, M. A. L. Recent developments in libxc—a comprehensive library of functionals for density functional theory. SoftwareX 7, 1–5 (2018).

57. Wu, X. et al. Enhancing GPU-acceleration in the Python-based simulations of chemistry frameworks. WIREs Comput. Mol. Sci. 15, e70008 (2025).

58. Li, R., Sun, Q., Zhang, X. & Chan, G. K.-L. Introducing GPU acceleration into the Pythonbased simulations of chemistry framework. J. Phys. Chem. A 129, 1459–1468 (2025).

59. Li, R. et al. Implementation of the multigrid Gaussian-plane-wave algorithm with GPU acceleration in PySCF. J. Chem. Theory Comput. 22, 10101–10112 (2026).

60. Soler, J. M. et al. The SIESTA method for ab initio order-N materials simulation. J. Phys. Condens. Matter 14, 2745–2779 (2002).

61. RDKit: open-source cheminformatics, release 2024.03.3. Zenodo https://doi.org/10.5281/ zenodo.11396708 (2024).

62. Kabsch, W. A solution for the best rotation to relate two sets of vectors. Acta Crystallogr. A 32, 922–923 (1976).

63. Swope, W. C., Andersen, H. C., Berens, P. H. & Wilson, K. R. A computer simulation method for the calculation of equilibrium constants for the formation of physical clusters of molecules: application to small water clusters. J. Chem. Phys. 76, 637–649 (1982).

64. Wang, S., Witek, J., Landrum, G. A. & Riniker, S. Improving conformer generation for small rings and macrocycles based on distance geometry and experimental torsional-angle preferences. J. Chem. Inf. Model. 60, 2044–2058 (2020).

65. Li, C., Sharir, O., Yuan, S. & Chan, G. K.-L. QM9 DFT electron density. CaltechDATA https://doi.org/10.22002/7vr2f-0r732 (2024).

66. Li, C., Sharir, O., Yuan, S. & Chan, G. K.-L. C<sub>7</sub>O<sub>2</sub>H<sub>10</sub> isomer DFT electron density. Caltech-DATA https://doi.org/10.22002/04d1z-1d608 (2024).

67. Li, C., Sharir, O., Yuan, S. & Chan, G. K.-L. GTP-in-microtubule DFT electron density. CaltechDATA https://doi.org/10.22002/v5tec-g6p43 (2024).

68. Li, C., Sharir, O., Yuan, S. & Chan, G. K.-L. Image super-resolution inspired electron density prediction. Figshare https://doi.org/10.6084/m9.figshare.25365508 (2024).

## Acknowledgements

This work was supported by a KIAS Individual Grant (AP110201) via the Center for Artificial Intelligence and Natural Sciences at Korea Institute for Advanced Study.

## Author contributions

J.W. conceived the study, carried out all calculations and analyses, and wrote the manuscript.

## Competing interests

The author declares no competing interests.

![](images/dd7748a00282683704e6f337f7956405e0029e5c88481e5de2a3d81513876038.jpg)

![](images/e669942b5171d802a716e24535adee66de8796f6bd89d979319c8f7ac79587f2.jpg)  
Extended Data Fig. 1 | SwinUNETR-V2 and NeXtUNETR. $k ^ { 3 }$ denotes a $k \times k \times k$ kernel or patch. a, Density formulation shared by both models: the zero-padded SAD density is refined by the backbone B and a $1 ^ { 3 }$ convolution head h into a correction that is added back, made non-negative, cropped and rescaled to the number of valence electrons N (Methods). $^ { \mathbf { b } , \mathbf { c } , }$ The backbones of SwinUNETR-V2 (b) and NeXtUNETR (c) on a common grid: rows are resolutions from the input grid to $1 / 3 2$ (channel counts in multiples of $C \ = \ 9 6 )$ ; columns are the encoder, the blocks on the skip connections and the decoder. SwinUNETR-V2 places $3 ^ { 3 }$ residual blocks on the skips (except at $1 / 1 6 )$ , in the bottleneck and in the decoder, layer-normalizes the features leaving the encoder (without learned parameters) and ends in a $1 ^ { 3 }$ convolution from C channels to one; NeXtUNETR uses ConvNeXt blocks in the bottleneck and decoder and passes the skips unchanged. d, Swin block (alternate blocks shift the windows by 3 voxels), $3 ^ { 3 }$ residual block (with $\mathrm { ~ a ~ } 1 ^ { 3 }$ convolution on the shortcut where the channel count changes) and ConvNeXt block, with parameter counts at $C = 9 6$ for the first stage. $\mathbf { e } ,$ Parameters by function. Spatial mixing takes 2.4% (attention) and 3.3% (depthwise $7 ^ { 3 }$ kernels) of the parameters of SwinUNETR-V2 and NeXtUNETR, respectively; SwinUNETR-V2’s size comes from its standard (non-depthwise) $3 ^ { 3 }$ convolutions (84%), with 44% in the bottleneck block alone.

## Extended Data

![](images/bad88094cc2d7dd0e260524588f04ad679eefd36448bd257ad884fbcd7f94065.jpg)

![](images/5caf99a0d61ae3832e0b1e3d1f5b04bece6bf0eab46eb624c39aef8883526064.jpg)

![](images/32d63a0e2e8f6cbe0993fe7fae139d0d4b3922a6ea7107e8ef835b04ef9a22ac.jpg)

![](images/c62180d02d6a366b17f931c2f16d157ab8103495277e8fb8d2218e451a9d3cc7.jpg)

![](images/afb6774bbc22e04de4a6323734ebc6916f7a992b8244ca59dc54a33e269eb31e.jpg)

f  
![](images/a6597e97085dcc77171ec90ca6ea851d6578b7d4c63b93d3ab0e4517c8a1acf8.jpg)  
Extended Data Fig. 2 | One-step energies and forces against the reference SCF, QM9 test set. Total energy per molecule (a,d) and Cartesian force components with (b,e) and without (c,f) the non-self-consistency correction, for SwinUNETR-V2 (top) and NeXtUNETR (bottom). The two molecules with a reference HOMO–LUMO gap below 1 eV (QM9 56343 and 37992; energy errors 0.6–5.3 eV with these two models and 0.3–18 eV including SR-ResNet, force-component errors up to 300 eV Å<sup>−1</sup> w/ NSCC) are excluded; with them, the force-component $\mathrm { R } ^ { 2 } \ \mathrm { w } / $ NSCC becomes negative. $\mathrm { R } ^ { 2 } ~ ( 1 - \mathrm { R } ^ { 2 }$ for energies) and MAE are over the remaining 9,998 molecules (539,508 components); Table 1 reports all 10,000. Dotted: $y = x .$

φ (°)  
a  
![](images/65ac15ac4835d5e135e62e0722c75072c04e2c80f898a78bece008594f869094.jpg)

![](images/c2cc6f8c266e3f4ae2792cf7b699ab2e7eca0fae4a8b2d105f243fd2655cc29e.jpg)

![](images/014854e39a683cefefaaa5a9dac2dce9357ef12113e0cd9ef5c448d19d772e4b.jpg)

![](images/ae1bc410cde6eef431c5ee0a23bf876d98bb8c097f8dba24b0269536f5d55e7b.jpg)

![](images/bea0a13bad1bf69933dab43e0e6488585e71259c74b80866d5ef917300e4415b.jpg)  
Reference (SCF), VDOS SwinUNETR-V2 w/ NSCC NeXtUNETR w/ NSCC Reference (SCF) SwinUNETR-V2 w/o NSCC NeXtUNETR w/o NSCC

b  
![](images/0dbd357c4949da95c42cf19c5147435c29681ea3fc30601d0bff01e0f4e8f213.jpg)  
c

![](images/2a62e158a7cd8a30635087d26f442f0104442e838d5c0d562bd15eece166f9bb.jpg)  
d

![](images/0e7f4c08efea621f1420186bc268b3f681264c6d34be61f4a9505062ca19dc05.jpg)  
e

![](images/79c1c22d3fc29d1483967465cd8c8a7c8aad4f0fc84e6b2933424de8b9a426fb.jpg)  
Extended Data Fig. 3 | Energy conservation requires the non-self-consistency correction. ${ \mathbf { a } } ,$ Backbone dihedrals of the four alanine-dipeptide trajectories of each method on its own $( \varphi , \psi )$ map: the SCF map for the reference, the model’s one-step map for the model $\left( \mathrm { w } \right)$ and $\mathrm { w / o }$ NSCC share it). $\mathbf { b } ,$ Vibrational density of states (VDOS) of the C7eq trajectories $\mathrm { w } / $ and $\mathrm { w / o }$ NSCC (SwinUNETR-V2 above, NeXtUNETR below, ofset vertically), each over the reference (gray). c, Total energy per atom relative to its initial value along the C7eq trajectories. d, Standard deviation of the total energy over the second half of each trajectory, all four trajectories. $\mathbf { e } ,$ Force error (component MAE, 25 fs running mean) on the frames of the C7eq reference trajectory, $\mathrm { w } / $ and $\mathrm { w / o }$ NSCC. Without the NSCC, the s.d. is 1.7–3.7 meV $\mathrm { a t o m } ^ { - 1 }$ (reference 0.05–0.07 meV $\mathrm { a t o m } ^ { - 1 } )$ ) and the systems heat by 24–97 K (not shown). Settings as in Fig. 3; one trajectory per condition.

![](images/877b419375158870ae14cbff47504bbf6a3c53845a485291e548bd5b98b6e4c0.jpg)  
b

![](images/083805f893f6d1fa6958d1a4e7bf371f743963cf897895ceaafe8b987670fc84.jpg)  
Extended Data Fig. 4 | Size extrapolation during training. Density NMAE on Wmin-PBE/3–100 for full-QM9-trained NeXtUNETR (a) and SwinUNETR-V2 (b) after 10, 20, 30 and 40 epochs and at the end of training (50 epochs). The NeXtUNETR error on small clusters falls with training (3–10 molecules: from 0.58% to 0.46%), whereas that on large clusters is lowest after 20 epochs and then rises (60–100 molecules: from 0.68% to 2.57%). The SwinUNETR-V2 error falls in every size range up to 40 epochs and then levels of (60–100 molecules: from 0.80% to 0.59%).

Extended Data Table 1 | Structural ablations on QM9-subset-8k. Density NMAE (%) on the 500 QM9 test molecules of Fig. 1. Each variant is compared with the first row of its block: relative change of the mean (∆ mean) and share of molecules on which the variant is worse. One training run each (50 epochs); fp32 evaluation.
<table><tr><td>Family</td><td>Model</td><td>Params (M)</td><td>NMAE mean / median (%)</td><td>∆ mean</td><td>Worse on</td></tr><tr><td rowspan="2">SwinUNETR-V2</td><td>reference (SwinUNETR-V2-292M of Fig. 1)</td><td>291.7</td><td>0.204 / 0.190</td><td></td><td></td></tr><tr><td>without the 33 residual blocks on the downsam- pled skips and in the bottlenecka</td><td>153.8</td><td>0.209 / 0.196</td><td>+2%</td><td>69%</td></tr><tr><td rowspan="2">NeXtUNETR</td><td>reference (ConvNeXt encoder blocks)</td><td>67.5</td><td>0.133 / 0.124</td><td></td><td></td></tr><tr><td>encoder blocks replaced by Swin blocksb</td><td>73.2</td><td>0.250 / 0.236</td><td>+88%</td><td>99.8%</td></tr><tr><td rowspan="2">SR-ResNet</td><td>reference (half-resolution input, ×2 pixel-shuffle upsampling)c</td><td>9.4</td><td>0.320 / 0.309</td><td></td><td></td></tr><tr><td>plain ResNet on the full grid (no upsampling)</td><td>8.3</td><td>0.301 / 0.276</td><td>-6%</td><td>29%</td></tr><tr><td rowspan="2">3D U-Net</td><td>reference (plain 3D U-Net, base width 96)</td><td>50.4</td><td>0.499 / 0.466</td><td></td><td></td></tr><tr><td>3D residual U-Net with group normalization (base width 96)</td><td>51.2</td><td>0.259 / 0.243</td><td>-48%</td><td>0%</td></tr></table>

<sup>a</sup> Residual blocks on the 1/2, 1/4 and 1/8 skip connections and the bottleneck block (138 M parameters, 127 M of them in the bottleneck block).

Extended Data Table 2 | Geometry optimization on one-step surfaces. 200 QM9 test molecules optimized with geomeTRIC from MMFF94 starting structures. RMSD: Kabsch rootmean-square deviation from the optimum the reference reaches from the same start. SAD: the one-step calculation on the SAD density, without a network. Converged: all criteria met within 300 steps. Steps include the starting structure; 301 means not converged within 300 steps. w/ and w/o NSCC: with and without the non-self-consistency correction. p90: 90th percentile.
<table><tr><td>Surface</td><td>Converged</td><td>Steps, median (max)</td><td>RMSD, median / p90 (Å)</td><td>Within 0.1 Å</td><td>Heavy-atom RMSD, median (Å)</td></tr><tr><td>MMFF94 start</td><td></td><td></td><td>0.116 / 0.335</td><td>42.5%</td><td>0.078</td></tr><tr><td>Reference SCF</td><td>200/200</td><td>15.5 (61)</td><td>0 (by definition)</td><td></td><td></td></tr><tr><td>SwinUNETR-V2 w/ NSCC</td><td>200/200</td><td>16 (45)</td><td>0.003 / 0.034</td><td>98.0%</td><td>0.002</td></tr><tr><td>SwinUNETR-V2 w/o NSCC</td><td>200/200</td><td>16 (63)</td><td>0.005 / 0.038</td><td>97.0%</td><td>0.003</td></tr><tr><td>NeXtUNETR w/ NSCC</td><td>200/200</td><td>15.5 (52)</td><td>0.002 / 0.021</td><td>98.5%</td><td>0.001</td></tr><tr><td>NeXtUNETR w/o NSCC</td><td>200/200</td><td>15.5 (53)</td><td>0.003 / 0.032</td><td>98.0%</td><td>0.002</td></tr><tr><td>SAD w/ NSCC</td><td>199/200</td><td>22 (301)</td><td>0.156 / 0.472</td><td>23.0%</td><td>0.120</td></tr><tr><td>SAD w/o NSCC</td><td>5/200</td><td>301 (301)</td><td>0.245 / 0.416</td><td>2.5%</td><td>0.181</td></tr></table>

## Supplementary Information

## Supplementary Note 1: the 400 Ha evaluation grid

Why. The labels, the SAD input and the network outputs are all defined on the uniform 200 Ha (400 Ry) grid of the reference densities of Li et al. <sup>8</sup>. A grid fixed in space is not translation invariant: moving a molecule rigidly relative to the grid points changes the multigrid energy and forces, with the period of the grid (the egg-box efect). We encountered it in geometry optimization on the 200 Ha grid, where the maximum force stalled at 12–28 meV $\textup { \AA } ^ { - 1 }$ while the energy kept decreasing.

Size of the efect. Supplementary Fig. 2 quantifies it for the relaxed alanine dipeptide (C7eq), translated rigidly by one grid spacing along each axis (0.144 Å along the cell diagonal, 13 points). A rigid translation changes no physical quantity, so every variation is an artifact. These measurements used our CPU implementation (PySCF 2.6.2).

• At 200 Ha, the energy varies by 21.5 meV peak to peak (reference SCF) and 21.3 meV (one-step, SwinUNETR-V2), and returns to within 0.02 meV after one period.

• Per-atom forces (net force removed) vary by 26.8 and 27.3 meV Å<sup>−1</sup> RMS.

• The variation is concentrated on oxygen (up to 200 meV Å<sup>−1</sup>) and nitrogen (90 meV Å<sup>−1</sup>), whose GTH pseudopotential cores are the hardest and whose valence densities are therefore the sharpest relative to the grid spacing. Carbon and hydrogen vary by 22–30 meV $\textup { \AA } ^ { - 1 }$

What the $ \mathit { 4 0 0 }$ Ha grid changes. On a 400 Ha grid $( { \sqrt { 2 } }$ times as many points per axis), the same translation changes the energy by 1.3 meV (SCF) and 1.6 meV (one-step) and per-atom forces by 4.4 and 5.4 meV $\textup { \AA } ^ { - 1 }$ RMS (largest 38 and 45 meV $\textup { \AA } ^ { - 1 }$ , on oxygen). The artifact thus falls 13–16-fold in energy and 5–6-fold in force.

The artifact is common to the reference and the one-step calculation, so it cancels in the comparisons the paper reports:

• the electronic-energy error of the one-step calculation (SwinUNETR-V2, alanine dipeptide) is 2.97 meV on average at 200 Ha and 2.96 meV at 400 Ha, varying by 0.42 meV over the translation on both grids;

• on 30 QM9 molecules the per-molecule force MAE against the reference changes by at most 0.16 meV $\textup { \AA } ^ { - 1 }$ between the two grids.

The 400 Ha grid therefore matters for the energy surface itself, which geometry optimization and molecular dynamics follow, and not for the accuracy figures.

How. The networks are not retrained: the SAD input, the prediction $\rho _ { \mathrm { M L } }$ , the Kohn–Sham Hamiltonian matrix $H [ \rho _ { \mathrm { M L } } ]$ and its diagonalization stay on the 200 Ha grid, where the training data are defined, and only the evaluation of the resulting density matrix D moves to the finer grid.

• Energy. $E _ { \mathrm { 1 - s t e p } } ~ = ~ E [ { \cal D } ]$ is assembled on the 400 Ha grid. The kinetic and nonlocalpseudopotential terms are analytic and do not depend on the grid. The local pseudopotential, Hartree and exchange–correlation terms use the density of D collocated on GPU4PySCF’s multigrid at 400 Ha.

• Grid size. The fine mesh is the label mesh scaled by $\sqrt { 2 }$ per axis and rounded up to an even number of points, since odd meshes break the parity on which the label grid’s density representation relies.

• Force. The force is the exact gradient of this energy.

– The $\mathrm { w / o }$ NSCC term (fixed D) is evaluated on the 400 Ha grid.

– The NSCC uses $G = \partial E _ { \mathrm { 1 - s t e p } } / \partial D$ , the Kohn–Sham Hamiltonian matrix of D on the 400 Ha grid, in the response amplitudes of Methods.

– The derivative of $H [ \rho _ { \mathrm { M L } } ]$ with respect to the nuclear positions, and the vector–Jacobian product through the network and the SAD, are taken on the 200 Ha grid, where $\rho _ { \mathrm { M I } }$ is defined.

• Reference. The reference SCF runs entirely at 400 Ha. Its force is our gradient at the converged density matrix, which agrees with $\mathrm { G P U 4 P y S C F ^ { \prime } s }$ own SCF gradient within 0.0003 meV $\textup { \AA } ^ { - 1 }$ (component MAE).

## Validation and cost.

• For alanine dipeptide, the analytic force agrees with central finite diferences of the 400 Ha energy within 0.36 meV $\textup { \AA } ^ { - 1 }$ (component MAE, SwinUNETR-V2 $\mathrm { w } / $ NSCC, h = 0.002 Å). This is the same residual as that of the 200 Ha path against its own energy.

• Building the response amplitudes from the 200 Ha Kohn–Sham Hamiltonian matrix instead would leave an inconsistency of 1.15 meV $\textup { \AA } ^ { - 1 }$

• Along the translation, the analytic net force reproduces the derivative of the energy within the error of the 13-point diference.

• The finer evaluation adds $4 \%$ to an energy-and-force step (alanine dipeptide, SwinUNETR-$\mathrm { V } 2 ~ \mathrm { w } /$ NSCC, GPU implementation on an A100: 6.72 against 6.44 s).

• The RMS of what remains of the artifact (4.4–5.4 meV Å<sup>−1</sup>; 1.3–1.6 meV in energy) is well below the geometry-optimization maximum-force criterion (23 meV $\textrm { \AA } ^ { - 1 } )$ , although the largest per-atom deviation over a full grid period, on oxygen, is 38–45 meV $\textup { \AA } ^ { - 1 }$

![](images/edd24ea4680979e9da778cd8735b90d6a061bb62e65e3d2262d643a46ed63ca5.jpg)  
Supplementary Fig. 1 | Further geometry optimizations from MMFF94 starting structures. Four molecules drawn at random, together with the example of ${ \mathrm { F i g . } }$ . 2d–f, from the molecules whose model optima all lie within 0.01 Å RMSD of the optimum the reference reaches from the same start: QM9 72909 $\left( \mathrm { C } _ { 7 } \mathrm { H } _ { 1 1 } \mathrm { N O } ; \ \mathbf { a } \right)$ , 39043 $\mathrm { ( C _ { 8 } H _ { 1 0 } O ; }$ b), 40727 $\left( \mathrm { C } _ { 9 } \mathrm { H } _ { 1 4 } ; \ \mathbf { c } \right)$ and 107860 $\left( \mathrm { C } _ { 7 } \mathrm { H } _ { 1 1 } \mathrm { N O } ; { \bf d } \right)$ . Left, reference optimum; right, energy relative to each surface’s own optimum (top) and maximum force (bottom) along the optimization, for the reference SCF and for SwinUNETR-V2 and NeXtUNETR with and without the non-self-consistency correction (blue circles, SwinUNETR-V2; green diamonds, NeXtUNETR; black triangles, reference; solid, w/ NSCC; dashed, w/o NSCC). Energy on a symmetric logarithmic axis. Dotted: maximum-force criterion.

![](images/0f403d7170cfebee4d8d319f48c2c23165c8b875eb1a8137aa0bc12e962e3b59.jpg)

![](images/a574bf7dc69dbf7ab7fef40bfd4eb102e22b48d679e1e50b78d81b930cd2e560.jpg)

![](images/6db83529e332c65202e5ba33525f21ad90f71e9cdd8db58f3e7814a57db1459a.jpg)  
Supplementary Fig. 2 | The egg-box efect of the 200 Ha grid and the 400 Ha evaluation. The relaxed alanine dipeptide (C7eq) translated rigidly by one spacing of the 200 Ha grid along each axis (0.144 Å along the cell diagonal; 13 points). a, Energy relative to the unshifted geometry, for the reference SCF (solid lines, filled circles) and the one-step calculation (SwinUNETR-V2; dashed lines, open squares), evaluated on the 200 Ha grid (orange) and on the 400 Ha grid (blue); the 400 Ha curves are not periodic in the 200 Ha spacing. b, Per-atom force variation: deviation of each force component from its mean over the 13 points, net force removed at each point, RMS over atoms. c, Largest deviation of a per-atom force vector over the translation, by element (filled, SCF; hatched, one-step). One-step forces are w/ NSCC; in this CPU calculation the fixed-D term is evaluated on the 400 Ha grid but the Kohn–Sham matrix of the output density that enters the NSCC on the 200 Ha grid (the GPU implementation evaluates both on the 400 Ha grid; fully consistent, the 400 Ha RMS is 5.3 instead of 5.4 meV Å<sup>−1</sup>).

Supplementary Table 1 | Inference in fp32 and TF32. Density NMAE (%, mean over the 500 QM9 test molecules of Fig. 1) and network forward time per molecule, with TF32 tensor-core arithmetic disabled (fp32, used throughout the paper) and enabled; forward times on one NVIDIA H200 NVL GPU with batch size 1, as in Fig. 1d. “Prediction change” is the mean relative change of the predicted density between the two settings. All networks we trained used TF32 (PyTorch’s default for cuDNN convolutions).
<table><tr><td>Model</td><td>Params (M)</td><td>NMAE fp32</td><td>NMAE TF32</td><td>Forward fp32 (ms)</td><td>Forward TF32 (ms)</td><td>Prediction change (%)</td></tr><tr><td>SwinUNETR-V2-18M</td><td>18.5</td><td>0.369</td><td>0.370</td><td>88</td><td>76</td><td>0.006</td></tr><tr><td>SwinUNETR-V2-73M</td><td>73.1</td><td>0.255</td><td>0.255</td><td>145</td><td>94</td><td>0.004</td></tr><tr><td>SwinUNETR-V2-292M</td><td>291.7</td><td>0.204</td><td>0.204</td><td>278</td><td>116</td><td>0.003</td></tr><tr><td>SwinUNETR-V2-656M</td><td>655.9</td><td>0.215</td><td>0.215</td><td>550</td><td>165</td><td>0.003</td></tr><tr><td>NeXtUNETR-17M</td><td>17.5</td><td>0.187</td><td>0.188</td><td>38</td><td>33</td><td>0.010</td></tr><tr><td>NeXtUNETR-67M</td><td>67.5</td><td>0.133</td><td>0.133</td><td>74</td><td>60</td><td>0.008</td></tr><tr><td>NeXtUNETR-150M</td><td>150.1</td><td>0.116</td><td>0.116</td><td>115</td><td>89</td><td>0.007</td></tr><tr><td>NeXtUNETR-265M</td><td>265.4</td><td>0.120</td><td>0.120</td><td>157</td><td>115</td><td>0.007</td></tr><tr><td>SR-ResNet-9M</td><td>9.4</td><td>0.320</td><td>0.238</td><td>213</td><td>57</td><td>0.222</td></tr><tr><td>SR-ResNet-70M</td><td>70.2</td><td>0.198</td><td>0.191</td><td>1,544</td><td>208</td><td>0.123</td></tr></table>

Supplementary Table 2 | Single-point wall times. Median time per molecule (s; three significant figures) over the 200 QM9 molecules of Extended Data Table 2 (4–25 atoms, median 19), each at its reference-SCF minimum. One calculation per molecule and row.
<table><tr><td></td><td>A40</td><td>A100</td><td>H100</td><td>CPU (16 cores)</td></tr><tr><td>Reference SCF</td><td></td><td></td><td></td><td></td></tr><tr><td>energy</td><td>10.6</td><td>2.29</td><td>2.17</td><td>61.8</td></tr><tr><td>energy + forces</td><td>12.8</td><td>2.69</td><td>2.41</td><td>68.4</td></tr><tr><td>NeXtUNETR</td><td></td><td></td><td></td><td></td></tr><tr><td>network forward</td><td>0.140</td><td>0.113</td><td>0.0686</td><td>1.89</td></tr><tr><td>network backward</td><td>0.159</td><td>0.128</td><td>0.0837</td><td>2.22</td></tr><tr><td>energy</td><td>1.63</td><td>0.543</td><td>0.451</td><td>6.95</td></tr><tr><td>energy + forces w/o NSCC</td><td>2.89</td><td>0.687</td><td>0.573</td><td>12.7</td></tr><tr><td>energy + forces w/ NSCC</td><td>4.39</td><td>0.997</td><td>0.801</td><td>24.2</td></tr><tr><td>SwinUNETR-V2</td><td></td><td></td><td></td><td></td></tr><tr><td>network forward</td><td>0.788</td><td>0.708</td><td>0.268</td><td>8.34</td></tr><tr><td>network backward</td><td>0.818</td><td>0.557</td><td>0.297</td><td>11.8</td></tr><tr><td>energy</td><td>2.32</td><td>1.15</td><td>0.664</td><td>13.1</td></tr><tr><td>energy + forces w/o NSCC</td><td>3.61</td><td>1.32</td><td>0.791</td><td>19.6</td></tr><tr><td>energy + forces w/ NSCC</td><td>5.78</td><td>2.03</td><td>1.23</td><td>39.2</td></tr></table>

Times are for a new molecule in a running program: program start, library and model loading, file input and cell construction are excluded; every row starts from the same emptied caches.

Reference SCF: PBE, converged to 10<sup>−11</sup> Ha from the atomic guess, density and forces on the 400 Ha grid; median 16 SCF cycles (range 9–20). One-step: as described in Methods (network on the 200 Ha grid, energy and forces on the 400 Ha grid).

Networks run in fp32 (TF32 of). Network forward and backward: the network alone, from the SAD input. The backward pass gives the gradient with respect to the network input only (weights frozen); it is the network part of the w/ NSCC force.

Hardware: one NVIDIA A40 (48 GB), A100 SXM4 (80 GB) or H100 NVL GPU (GPU4PySCF 1.8.0 with PySCF 2.14.0, CuPy 14.1.1, PyTorch 2.3.1); CPU: 16 cores of an AMD EPYC 7543 (PySCF 2.6.2, PyTorch 2.3.1).