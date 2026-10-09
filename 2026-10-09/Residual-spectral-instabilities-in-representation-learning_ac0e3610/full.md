# Residual spectral instabilities in representation learning

Zhen Li<sup>∗</sup>

Department of Complexity Science and Engineering, Graduate School of Frontier Sciences, The University of Tokyo, Kashiwa 277-8561, Japan.

Learned representations can lose latent degrees of freedom successively, suggesting a cascade of transitions whose underlying stability principle remains unclear. Here we formulate dimensionwise posterior collapse in variational autoencoder (VAE) as a fluctuation theory around partially collapsed states. Interpreting the negative evidence lower bound as an efective free energy, its quadratic expansion defines a Gaussian theory whose Hessian acts as a mass matrix for latent fluctuations. We show that the collapsed directions form an invariant fluctuation sector and derive its exact mass spectrum in terms of a conditional residual operator. A local reactivation direction lowers the free energy when the decoder variance falls below the residual spectral upper edge, with equality marking marginality. The criterion recovers principal component thresholds in the linear Gaussian VAE limit. Viewed in reverse along continuously connected branches, the reactivation boundary provides a local criterion for successive collapse. Numerical continuation experiments show successive loss of latent dimensions near these spectral marginalities. These results support a spectral cascade interpretation governed by residual information left unexplained by the surviving representation.

Introduction— Learning a representation amounts to retaining and organizing a restricted set of degrees of freedom from a high dimensional data distribution [1– 5]. A fundamental question is therefore how these de grees of freedom are lost as the balance between information preservation and regularization is varied [3]. From a statistical physics perspective, this question can be approached through the stability of representations in which some latent coordinates have already collapsed [6, 7]. A soft mode in such a state identifies a local route by which a dormant coordinate can resume carrying information, analogous to instabilities of soft modes in structural phase transitions [6–10]. Continuous connections between active and collapsed branches then relate this reactivation boundary to the loss of a latent degree of freedom under reverse parameter variation. A broader challenge in representation learning is to understand how the information retained by an existing representation shapes the spectral stability boundaries for subsequent changes in its efective dimensionality.

Variational autoencoders (VAEs) provide a natural setting in which to address this question [11, 12]. A VAE represents high dimensional data x through a lowerdimensional stochastic latent variable z. The encoder $q _ { \phi } ( z | \pmb { x } )$ maps each input to a distribution in latent space, while the decoder $p _ { \theta } ( { \pmb x } | { \pmb z } )$ reconstructs the data from the resulting representation. Learning therefore balances two competing tendencies: preserving suficient information in z to reconstruct the data and regularizing its distribution toward a prescribed prior p(z). This competition is encoded in the evidence lower bound (ELBO), which combines a reconstruction term with a Kullback–Leibler penalty that drives the variational posterior toward the latent prior [11–16]. A suficiently strong relative penalty can render latent variables uninformative, producing posterior collapse [17–19]. This phenomenon has been attributed to optimization dynamics, expressive decoders, inference mismatch, rate, distortion, and intrinsic properties of the variational objective [18, 20–23]. Analytical studies of linear VAEs have further revealed a close connection between collapse and the principal-component spectrum of the data [18, 24–26]. More generally, the negative ELBO can be viewed as an efective free energy functional, whose extrema define competing learned representations [27, 28]. Within this picture, the fully collapsed state is a trivial stationary solution whose local quadratic stability is controlled by the largest eigenvalue of the data covariance matrix [17]. This identifies a local stability boundary intrinsic to the variational landscape.

Complete collapse, however, conceals a richer structure. In practice, latent variables need not become uninformative simultaneously: individual posterior variances can approach their prior values at distinct control parameters, leaving intermediate states in which only a subset of latent directions remains active. Such dimension-wise collapse has been observed in both linear Gaussian VAEs and nonlinear ones, but the covariance criterion for the fully collapsed state does not determine the stability of these partially collapsed representations [17, 18]. This hierarchy motivates an analogy with successive symmetry breaking [6, 7, 29–31]. Complete collapse provides a symmetric reference background, invariant under sign reversals of every latent coordinate, while partially collapsed states retain this invariance in their collapsed sectors. Reactivation through a posterior-mean mode can break the corresponding reflection symmetry by developing coupled encoder and decoder fields. A sequence of such activations suggests a cascade from the fully collapsed background toward states with more active coordinates [6, 7, 31, 32]. The reverse sequence ofers a picture of dimension-wise collapse as successive symmetry restoration. This analogy motivates our analysis of quadratic fluctuations about partially collapsed backgrounds.

The active variables already encode part of the data fluctuations. Residual information left unresolved by this representation can make the reactivation of a collapsed coordinate locally favorable. This raises two related questions: What residual fluctuations can make a collapsed latent coordinate locally favorable to reactivate, and how do the resulting stability boundaries organize successive collapse?

Here we construct a Gaussian fluctuation theory around partially collapsed stationary states of Gaussian VAEs. Fluctuations associated with collapsed latent directions form an invariant sector of the Hessian, whose stability is governed by a conditional residual operator. We determine when the quadratic form in the col lapsed sector acquires a negative direction. Its marginal boundary identifies a possible continuous bifurcation between stationary backgrounds with diferent numbers of active latent fields. This criterion recovers the covariance threshold for complete collapse and the principalcomponent thresholds of linear Gaussian VAEs. This local fluctuation analysis suggests a cascade of spectral marginalities, whose relation to successive collapse is examined numerically in nonlinear VAEs.

Gaussian fluctuations— We consider a Gaussian VAE [33] with data $\pmb { x } \in \mathbb { R } ^ { N }$ and latent variables $z \in \mathbb { R } ^ { n }$ ， $n < N$ . The latent prior, decoder, and diagonal variational posterior are

$$
p ( z ) = \prod _ { j = 1 } ^ { n } \mathcal { N } ( z _ { j } ; 0 , 1 ) ,\tag{1}
$$

$$
p _ { \theta } ( \pmb { x } | \boldsymbol { z } ) = \prod _ { i = 1 } ^ { N } \mathcal { N } ( x _ { i } ; f _ { i } ( \boldsymbol { z } ) , \sigma ^ { \prime 2 } ) ,\tag{2}
$$

$$
q _ { \phi } ( z | \mathbf { x } ) = \prod _ { j = 1 } ^ { n } \mathcal { N } ( z _ { j } ; \mu _ { j } ( \mathbf { x } ) , \sigma _ { j } ^ { 2 } ( \mathbf { x } ) ) ,\tag{3}
$$

respectively. Here $\sigma ^ { \prime 2 }$ is the decoder variance and serves as the control parameter. Up to an additive constant, we write the negative evidence lower bound as an efective free energy functional [19, 34],

$$
{ \mathcal { F } } [ \Phi ] = \frac { 1 } { 2 \sigma ^ { \prime 2 } } \left. \int d z q _ { \phi } ( z | \pmb { x } ) \| \pmb { x } - \pmb { f } ( z ) \| ^ { 2 } \right. _ { \pmb { x } }\tag{4}
$$

$$
+ \frac 1 2 \left. \sum _ { j = 1 } ^ { n } \left[ \mu _ { j } ^ { 2 } ( { \pmb x } ) + \sigma _ { j } ^ { 2 } ( { \pmb x } ) - \ln \sigma _ { j } ^ { 2 } ( { \pmb x } ) \right] \right. _ { { \pmb x } } ,\tag{5}
$$

where $\Phi : = \{ f , \mu , \sigma \}$ denotes the learned fields. Increasing $\sigma ^ { \prime 2 }$ weakens the reconstruction term relative to the latent regularization and therefore favors collapsed representations.

The conventional fully collapsed state corresponds to $f _ { i } = { \langle x _ { i } \rangle } _ { x } , \mu _ { j } = 0$ and $\sigma _ { j } ~ = ~ 1$ for all i and j [17]. Here we instead consider a generic stationary state in which the latent space is partitioned into an active set A and a collapsed set C. Such a state, consistent with the stationary equation [17], satisfies

$$
\bar { f } ( z ) = \bar { f } ^ { A } ( z _ { A } ) ,\tag{6}
$$

and for $c \in C$

$$
\bar { \mu } _ { c } ( { \pmb x } ) = 0 , \quad \bar { \sigma } _ { c } ( { \pmb x } ) = 1 ,\tag{7}
$$

so that

$$
\bar { q } (  { \boldsymbol { z } } |  { \boldsymbol { { x } } } ) = \bar { q } _ { A } (  { \boldsymbol { { z } } } _ { A } |  { \boldsymbol { { x } } } ) \varphi _ { C } (  { \boldsymbol { { z } } } _ { C } ) ,\tag{8}
$$

with $\varphi _ { C }$ the standard Gaussian measure in the collapsed subspace. The active variables therefore retain a nontrivial representation of the data, whereas the collapsed variables carry no information about x.

To analyze local stability around this partially collapsed stationary state, we expand the efective free energy around Φ:<sup>¯</sup>

$$
{ \mathcal { F } } [ { \bar { \Phi } } + \delta \Phi ] = { \mathcal { F } } [ { \bar { \Phi } } ] + { \frac { 1 } { 2 } } \left. \delta \Phi , \boldsymbol { K } \delta \Phi \right. + O ( \delta \Phi ^ { 3 } ) ,\tag{9}
$$

where $K : = \delta ^ { 2 } { \mathcal F } / \delta \Phi ^ { 2 } | _ { \bar { \Phi } }$ is the Hessian. Equation (9) defines a Gaussian fluctuation theory around the learned state, with the eigenvalues of K playing the role of squared masses [35]. A zero eigenvalue identifies a marginal fluctuation mode, whereas a negative eigenvalue indicates a direction of decreasing free energy [8, 36].

The factorization in Eq. (9) imposes a nontrivial structure on these fluctuations. We decompose a decoder fluctuation into the component that is independent of the collapsed coordinates and its orthogonal complement,

$$
\delta { \pmb f } = P _ { A } \delta { \pmb f } + ( 1 - P _ { A } ) \delta { \pmb f } \equiv \delta { \pmb f } ^ { \| } + \delta { \pmb f } ^ { \perp } ,\tag{10}
$$

where $P _ { A }$ denotes averaging over the Gaussian measure $\varphi _ { C }$ . Together with the encoder fluctuations, this defines

$$
\delta \Phi _ { A } : = ( \delta { \pmb f } ^ { \parallel } , \delta { \pmb \mu } _ { A } , \delta { \pmb \sigma } _ { A } ) ,\tag{11}
$$

$$
\delta \Phi _ { C } : = ( \delta { \pmb f } ^ { \perp } , \delta { \pmb \mu } _ { C } , \delta { \pmb \sigma } _ { C } ) ,\tag{12}
$$

and remarkably, the quadratic theory separates exactly,

$$
K = K _ { A } \oplus K _ { C } .\tag{13}
$$

Thus, fluctuations that activate a collapsed latent direction form an invariant sector of the full Hessian. Local stability against reactivation of collapsed directions can therefore be analyzed independently of the fluctuations of active sectors. It’s worth noting that the stability of the full stationary state additionally requires $\kappa _ { A } \succeq 0$

Residual spectrum— To determine the stability of the collapsed sector, we characterize the data fluctuations left unresolved by the active representation.

At a stationary point, variation with respect to the decoder gives the conditional mean $[ 1 7 ] , \bar { f } ^ { A } ( z _ { A } ) = \mathbb { E } [ { \pmb x } | z _ { A } ]$ where the expectation is taken with respect to the joint distribution $p _ { \mathrm { d a t a } } ( { \pmb x } ) \bar { q } _ { A } ( { \pmb z } _ { A } | { \pmb x } )$ . The corresponding residual field,

$$
r _ { i } ( { \pmb x } , z _ { A } ) : = x _ { i } - \bar { f } _ { i } ^ { A } ( z _ { A } ) ,\tag{14}
$$

therefore represents the part of the data fluctuations left unresolved by the active latent representation. To characterize which of these residual fluctuations can be captured by an additional latent degree of freedom, we introduce the conditional residual operator

$$
( \mathcal { R } _ { A } h ) _ { i } ( z _ { A } ) = \mathbb { E } \left[ r _ { i } ( \pmb { x } , z _ { A } ) h ( \pmb { x } ) \ | \ z _ { A } \right] ,\tag{15}
$$

which maps a fluctuation $h ( { \pmb x } )$ in data space to its residual correlation at fixed $z _ { A }$ . Its positive semi-definite Gram operator,

$$
\begin{array} { r } { \mathcal { C } _ { A } : = \mathcal { R } _ { A } ^ { \dagger } \mathcal { R } _ { A } , \quad \Lambda _ { * } ^ { ( A ) } : = \operatorname* { s u p s p e c } \mathcal { C } _ { A } = \| \mathcal { R } _ { A } \| ^ { 2 } , } \end{array}\tag{16}
$$

defines the residual spectrum of the partially collapsed representation.

The spectral upper edge has a useful variational interpretation [37]. Let $\rho _ { A }$ denote the active latent marginal induced by $p _ { \mathrm { d a t a } } \bar { q } _ { A }$ . Since the stationary decoder is the conditional mean, $\mathcal { R } _ { A } h = \operatorname { C o v } ( \pmb { x } , h ( \pmb { x } ) \mid z _ { A } )$ , and hence

$$
\Lambda _ { * } ^ { ( A ) } = \operatorname* { s u p } _ { \langle h ^ { 2 } \rangle _ { \pmb { x } } = 1 } \mathbb { E } _ { \rho _ { A } } \left[ \left\| \operatorname { C o v } ( \pmb { x } , h ( \pmb { x } ) \mid \boldsymbol { z } _ { A } ) \right\| ^ { 2 } \right] .\tag{17}
$$

Thus the softest direction is selected by the strongest conditional correlation between a normalized data fluctuation and the unexplained residual. This optimization extends the principal components to nonlinear data fluctuations and their conditional residual correlations at each active latent state. The resulting spectrum characterizes residual structure conditioned on the surviving representation. Its dependence on the learned joint distribution is essential: data covariance alone does not generally specify these conditional correlations.

The physical significance of $\mathcal { C } _ { A }$ follows from the Gaussian fluctuation theory developed above. Expanding decoder fluctuations in Hermite modes of a collapsed latent coordinate, the first Hermite mode couples to fluctuations of its posterior mean, while the second couples to fluctuations of its posterior variance. All remaining modes are strictly massive. For each residual eigenmode ${ \mathcal C } _ { A } \psi _ { \alpha } = \Lambda _ { \alpha } ^ { ( A ) } \psi _ { \alpha }$ , the coupled fluctuations reduce to independent mass blocks. For example, normalize a positiveeigenvalue mode by $\langle \psi _ { \alpha } ^ { 2 } \rangle _ { \pmb { x } } = 1$ . Its decoder partner is $\chi _ { \alpha } = \mathcal { R } _ { A } \psi _ { \alpha } / \sqrt { \Lambda _ { \alpha } ^ { ( A ) } }$ , with $\mathbb { E } _ { \rho _ { A } } \| \chi _ { \alpha } \| ^ { 2 } = 1$ . For a collapsed coordinate $c ,$ the perturbations $\delta { f ^ { \perp } } = u z _ { c } \chi _ { \alpha }$ and $\delta \mu _ { c } ~ = ~ v \psi _ { \alpha }$ contribute $\bar { \mathsf { \Omega } } _ { 2 } ^ { 1 } ( u , v ) M _ { \mu , \alpha } ( u , v ) ^ { \mathsf { T } }$ to the freeenergy expansion, where

$$
M _ { \mu , \alpha } = \left( { \begin{array} { c c } { { 1 / \sigma ^ { \prime 2 } } } & { { - \sqrt { \Lambda _ { \alpha } ^ { ( A ) } } / \sigma ^ { \prime 2 } } } \\ { { - \sqrt { \Lambda _ { \alpha } ^ { ( A ) } } / \sigma ^ { \prime 2 } } } & { { 1 } } \end{array} } \right) .\tag{18}
$$

The diagonal terms penalize decoder and encoder distortions separately; residual correlations couple them and can make a coordinated fluctuation favorable. In this normalized field metric, the lower squared mass is

$$
m _ { \mu , \alpha , - } ^ { 2 } = \frac { 1 + \sigma ^ { \prime 2 } - \sqrt { ( 1 - \sigma ^ { \prime 2 } ) ^ { 2 } + 4 \Lambda _ { \alpha } ^ { ( A ) } } } { 2 \sigma ^ { \prime 2 } } .\tag{19}
$$

Since det $M _ { \mu , \alpha } = ( \sigma ^ { \prime 2 } - \Lambda _ { \alpha } ^ { ( A ) } ) / \sigma ^ { \prime 4 }$ and its trace is positive, the lower mode changes sign precisely at the residual spectral threshold. The posterior-width channel couples to the second Hermite mode and has the same marginality condition; its mass block and the remaining spectrum are given in the Supplemental Material. Their lowest eigenvalues soften and vanish precisely at $\sigma ^ { \prime 2 } = \Lambda _ { \alpha } ^ { ( A ) }$ Thus, for a nonempty collapsed set $C$ and a bounded residual operator, nonnegative curvature in the collapsed sector is equivalent to

$$
K _ { C } \succeq 0 \quad \Longleftrightarrow \quad \sigma ^ { \prime 2 } \geq \Lambda _ { * } ^ { ( A ) } .\tag{20}
$$

Equivalently, $\operatorname { E q . }$ . (20) locates the onset of negative curvature in the collapsed fluctuation sector:

$$
\exists \delta \Phi _ { C } \ s . t . \ \langle \delta \Phi _ { C } , \pmb { K } _ { C } \delta \Phi _ { C } \rangle < 0 \iff \sigma ^ { \prime 2 } < \Lambda _ { * } ^ { ( A ) } .\tag{21}
$$

The unstable fluctuation is a coupled deformation of the encoder and decoder fields, with their mixing set by resid ual correlations. The quadratic operator contains an identical set of mass blocks for each collapsed coordinate, so a negative direction can be supported in a single coordinate sector. This establishes a local instability toward reactivation of at least one collapsed field. Higherorder terms govern the continuation to finite amplitude, while the evolution dynamics determine which stationary branch is approached.

For $\sigma ^ { \prime 2 } > \Lambda _ { * } ^ { ( A ) }$ , this sector has strictly positive curvature; equality closes its spectral gap and yields a zero mass mode when the spectral supremum is attained by an eigenmode.

The familiar criterion for complete posterior collapse follows as a limiting case [17]. For $A = \emptyset$ , no active representation survives, $\bar { f } ^ { \alpha }$ reduces to the data mean, and the nonzero residual spectrum coincides with the covariance spectrum $\Sigma _ { X }$ of the data [38]. Hence,

$$
\Lambda _ { * } ^ { ( \emptyset ) } = \lambda _ { \mathrm { m a x } } \left( \Sigma _ { X } \right) ,\tag{22}
$$

recovering the full-collapse threshold obtained previously [17]. The covariance criterion is therefore the terminal member of a more general hierarchy: before complete collapse, stability is controlled by the spectrum of fluctuations left unexplained by the current representation rather than by the spectrum of the original data alone.

To relate the reactivation criterion to collapse, consider a coordinate c that is active in A and collapsed in $A ^ { \prime } =$

$A \backslash \{ c \}$ . The relevant quadratic expansion is taken about the stationary background with active set $A ^ { \prime } .$ . Its modes of the collapsed sector describe coupled encoder–decoder fluctuations that reactivate c. For a background with nonnegative curvature for the collapsed sector, a zeromass mode must lie at the residual spectral upper edge [Eqs. (19) and (20)]:

$$
\sigma ^ { \prime 2 } = \Lambda _ { * } ^ { ( A ^ { \prime } ) } ( \sigma ^ { \prime 2 } ) , \qquad A \to A ^ { \prime } .\tag{23}
$$

Interpreting an observed collapse through this condition assumes that the active branch reaches such a marginal collapsed background continuously. Under this assumption, successive losses of active coordinates can be associated with successive marginalities, each evaluated on the corresponding collapsed branch. This provides a possible local explanation of the cascade, whose consistency with numerical continuation we examine below.

Spectral cascade— We examine whether increasing the control parameter $\sigma ^ { \prime 2 }$ produces successive collapse near the marginal points of the corresponding collapsed backgrounds. We use a set of 8 dimension synthetic Gaussian mixture data (see Supplemental Material) with principal components $\lambda _ { i } = \exp [ - 0 . 1 ( i - 1 ) ] \ ( i = 1 , \ldots , 8 )$ , and employ a four-dimensional latent space. The encoder consists of two hidden layers of widths 256 and 128, while the decoder uses the reversed architecture, with ReLU activations throughout. The decoder variance $\sigma ^ { \prime 2 }$ is varied as the control parameter.

Figure 1(a) shows the resulting efective number of active latent dimensions. Starting from $| { \cal A } | = 4$ at small $\sigma ^ { \prime 2 }$ , the representation loses one latent degree of freedom at a time and passes through a sequence from 4 to 0. The individual posterior variances in Fig. 1(b) reveal the microscopic origin of this staircase. Each $\left. \sigma _ { j } ^ { 2 } \right. _ { x }$ approaches its prior value of unity at a distinct value of $\sigma ^ { \prime 2 }$ after which the corresponding latent coordinate remains collapsed. The steps in |A| therefore represent genuine dimension-wise posterior collapse rather than a smooth reduction of information distributed uniformly among all latent variables.

To probe the collapsed-sector curvature in Eq. (21), we estimate the residual spectral upper edge $\hat { \Lambda } _ { * } ^ { ( A ) }$ along the numerical branches with a nonempty collapsed sector. At each frozen checkpoint, we average the decoder over collapsed prior coordinates and approximate the conditional residual operator using posterior weights normalized over sampled data and Monte Carlo samples of the active latent marginal. Power iteration on its empirical Gram operator, restricted to centered data-space fluctuations, yields the leading eigenvalue. To reduce finite-sampling bias, repeated estimates with B data points and L active latent samples are fitted to $\hat { \Lambda } ( B , L ) = \Lambda _ { \infty } + b / B + c / L$ by inverse-variance weighted least squares. The intercept defines $\hat { \Lambda } _ { * } ^ { ( A ) }$ . Detail algorithm can be referred in the Supplemental Material. Figure 1(c) compares this quantity with the spectral marginality condition $\sigma ^ { \prime 2 } = \hat { \Lambda } _ { * } ^ { ( A ) }$ . On intervals with fixed A, the estimated spectral upper edge generally lies below $\sigma ^ { \prime 2 }$ , consistent with positive squared masses in the collapsed sector. The approach to equality near the observed loss of a latent degree of freedom is consistent with a marginal connection between stationary branches. Once the transition occurs, the active representation changes and $\hat { \Lambda } _ { * } ^ { ( A ) }$ shifts onto a new branch.

The quantity $\sigma ^ { \prime 2 } - \hat { \Lambda } _ { * } ^ { ( A ) }$ in Fig. 1(d) controls the sign of the lowest squared masses of the collapsed sector. Its approach to zero signals softening about the corresponding background and is consistent with a continu ous connection to a branch carrying one additional active field. Each segment is evaluated with the residual operator of its own active background. The proximity of the observed collapse events to these marginal points supports the proposed spectral interpretation of the numerical branches. As a direct curvature check, we perturb representative projected backgrounds along the predicted coupled encoder–decoder modes and evaluate the result ing free-energy diferences. Fixed-background scans exhibit a change from negative to positive curvature near the residual spectral threshold, supporting the local fluctuation picture (see Supplemental Material).

Discussion and conclusions— In the linear Gaussian VAE limit, the residual spectrum can be evaluated explicitly. For an active set A aligned with the principal directions of the data, the condition for local stability in the collapsed sector under linear perturbations reduces to (see Supplemental Material):

$$
\sigma ^ { \prime 2 } \geq \operatorname* { m a x } _ { \alpha \not \in A } \lambda _ { \alpha } ( \Sigma _ { X } ) ,\tag{24}
$$

recovering the exact principal component thresholds known for linear VAEs [18, 24, 25]. The residual spectrum therefore generalizes the fixed covariance spectrum of linear models to a representation dependent spectrum for nonlinear learned states.

Our analysis is restricted to the quadratic expansion of the efective free energy and thus describes a free, Gaussian theory of fluctuations. The direct result is a criterion for the onset of negative curvature in the collapsed sector about a specified stationary background. Under a continuous branch connection, the reverse passage through marginality describes the disappearance of an active latent field. The sequence of realized backgrounds, the number of fields involved in each transition, and discontinuous branch changes depend on the global free-energy landscape, nonlinear fluctuation terms, and evolution dynamics. Higher-order terms couple the fluctuation modes and govern the free-energy variation along directions whose quadratic restoring term vanishes at marginality. Extending the expansion along these soft directions would help determine the finite-amplitude branches and the character of the associated bifurca tions [39, 40].

![](images/b30c1b5e1d55a1bcc3b76996b169115a74587638b39f856671eb2740a0aaf111.jpg)  
FIG. 1. Spectral cascade of dimension-wise posterior collapse. (a) Number of active latent dimensions |A| and ${ \bf ( b ) }$ mean posterior variances $\left. \sigma _ { j } ^ { 2 } \right. _ { \pmb { x } }$ versus decoder variance $\scriptstyle { \dot { \sigma } } ^ { \prime 2 }$ showing showing successive loss of individual latent dimensions. (c) Estimated leading residual eigenvalue $\hat { \Lambda } _ { * } ^ { ( A ) }$ , and the dashed line denotes the marginality condition $\sigma ^ { \prime 2 } = \hat { \Lambda } _ { * } ^ { ( A ) }$ . (d) Estimated spectral stability margin $\sigma ^ { \prime 2 } - \hat { \Lambda } _ { \ast } ^ { ( A ) }$ , which controls the sign of the lowest squared masses of the collapsed sector.

A second limitation lies in the background state itself. Although the stability criterion has a simple spectral form, the residual operator depends on the stationary ${ \bar { f } } ^ { A }$ and ${ \bar { q } } _ { A } .$ which obey nonlinear self-consistency equations involving high dimensional integrals. Except in special limits such as the linear Gaussian model, these equations cannot generally be eliminated to express the residual spectrum solely in terms of simple data statistics. Controlled approximations to the background state, and relations between residual spectra across successive stationary branches, therefore remain important open problems.

From the perspective of representation learning, the residual spectrum identifies fluctuations left unresolved by the active latent fields that can destabilize a collapsed sector. It thereby connects the local availability of an additional representational degree of freedom to the information already retained by the learned background. Relating these spectra across stationary branches ofers a route toward understanding how the efective dimension of a representation changes as the balance between reconstruction and regularization is varied. Extending this stability analysis to other latent-variable objectives is a natural next step toward a spectral account of the organization of learned representations [3, 41, 42].

Acknowledgement—This work was supported by JST SPRING, Grant No. JPMJSP2108.

[1] Y. Bengio, A. Courville, and P. Vincent, Representation learning: A review and new perspectives, IEEE Transactions on Pattern Analysis and Machine Intelligence 35, 1798 (2013).

[2] Y. LeCun, Y. Bengio, and G. Hinton, Deep learning, Nature 521, 436 (2015).

[3] D. J. Strouse and D. J. Schwab, The deterministic information bottleneck, Neural Computation 29, 1611 (2017).

[4] G. E. Hinton and R. R. Salakhutdinov, Reducing the dimensionality of data with neural networks, Science 313, 504 (2006).

[5] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, Learning representations by back-propagating errors, Nature 323, 533 (1986).

[6] G. Chechik, A. Globerson, N. Tishby, and Y. Weiss, Information bottleneck for gaussian variables, Advances in Neural Information Processing Systems 16 (2003).

[7] T. Wu and I. Fischer, Phase transitions for the information bottleneck in representation learning, in International Conference on Learning Representations (2020).

[8] W. Cochran, Crystal stability and the theory of ferroelectricity, Advances in Physics 9, 387 (1960).

[9] J. Scott, Soft-mode spectroscopy: Experimental studies of structural phase transitions, Rev. Mod. Phys. 46, 83 (1974).

[10] L. Casella and A. Zaccone, Soft mode theory of ferroelectric phase transitions in the low-temperature phase, Journal of Physics: Condensed Matter 33, 165401 (2021).

[11] D. P. Kingma and M. Welling, Auto-encoding variational bayes, arXiv preprint arXiv:1312.6114 (2013).

[12] D. J. Rezende, S. Mohamed, and D. Wierstra, Stochastic backpropagation and approximate inference in deep generative models, in International Conference on Machine Learning (PMLR, 2014) pp. 1278–1286.

[13] M. D. Hofman, M. J. Johnson, et al., Elbo surgery: yet another way to carve up the variational evidence lower bound, in Workshop in advances in approximate Bayesian inference, NIPS, Vol. 1 (2016).

[14] L. Girin, S. Leglaive, X. Bie, J. Diard, T. Hueber, and X. Alameda-Pineda, Dynamical variational autoencoders: A comprehensive review, Foundations and Trends in Machine Learning 15, 1 (2022).

[15] D. M. Blei, A. Kucukelbir, and J. D. McAulife, Variational inference: A review for statisticians, Journal of the American statistical Association 112, 859 (2017).

[16] P. K. Diederik and W. Max, An introduction to variational autoencoders, Foundations and Trends® in Machine Learning 12, 307 (2019).

[17] Z. Li, F. Zhang, Z. Zhang, and Y. Chen, Posterior collapse as a phase transition in variational autoencoders, Physica A 683, 131228 (2026).

[18] J. Lucas, G. Tucker, R. B. Grosse, and M. Norouzi, Don’t blame the elbo! a linear vae perspective on posterior

collapse, in Advances in Neural Information Processing Systems, Vol. 32 (2019).

[19] B. Dai, Z. Wang, and D. Wipf, The usual suspects? reassessing blame for vae posterior collapse, in International Cnference on Machine Learning (PMLR, 2020) pp. 2313–2322.

[20] J. He, D. Spokoyny, G. Neubig, and T. Berg-Kirkpatrick, Lagging inference networks and posterior collapse in variational autoencoders, in International Conference on Learning Representations (2019).

[21] C.-W. Huang, S. Tan, A. Lacoste, and A. Courville, Improving explorability in variational inference with annealed variational objectives, in Advances in Neural Information Processing Systems, Vol. 31 (2018).

[22] I. Higgins, L. Matthey, A. Pal, C. Burgess, X. Glorot, M. Botvinick, S. Mohamed, and A. Lerchner, beta-VAE: Learning basic visual concepts with a constrained variational framework, in International Conference on Learning Representations (2017).

[23] A. Alemi, B. Poole, I. Fischer, J. Dillon, R. A. Saurous, and K. Murphy, Fixing a broken elbo, in International Conference on Machine Learning (PMLR, 2018) pp. 159– 168.

[24] M. E. Tipping and C. M. Bishop, Probabilistic principal component analysis, Journal of the Royal Statistical Society Series B: Statistical Methodology 61, 611 (1999).

[25] H. Dang, T. Tran, T. Nguyen, and N. Ho, Beyond vanilla variational autoencoders: Detecting posterior collapse in conditional and hierarchical variational autoencoders, in International Conference on Learning Representations (2024).

[26] Z. Wang and L. Ziyin, Posterior collapse of a linear latent variable model, in Advances in Neural Information Processing Systems, Vol. 35 (2022) pp. 37537–37548.

[27] G. Carleo, I. Cirac, K. Cranmer, L. Daudet, M. Schuld, N. Tishby, L. Vogt-Maranto, and L. Zdeborov´a, Machine learning and the physical sciences, Rev. Mod. Phys. 91, 045002 (2019).

[28] R. Iten, T. Metger, H. Wilming, L. Del Rio, and R. Renner, Discovering physical concepts with neural networks, Phys. Rev. Lett. 124, 010508 (2020).

[29] P. M. Chaikin, T. C. Lubensky, and T. A. Witten, Principles of condensed matter physics, Vol. 10 (Cambridge University Press Cambridge, 1995).

[30] L. Landau, The theory of phase transitions, Nature 138, 840 (1936).

[31] K. Rose, E. Gurewitz, and G. C. Fox, Statistical mechanics and phase transitions in clustering, Physical Review Letters 65, 945 (1990).

[32] K. Rose, Deterministic annealing for clustering, compression, classification, regression, and related optimization problems, Proceedings of the IEEE 86, 2210 (1998).

[33] D. J. Rezende and F. Viola, Taming vaes, arXiv preprint arXiv:1810.00597 (2018).

[34] Y. Ichikawa and K. Hukushima, High-dimensional asymptotics of vaes: threshold of posterior collapse and dataset-size dependence of rate-distortion curve, J. Stat. Mech. 2025, 073402 (2025).

[35] G. Mussardo, Statistical field theory: an introduction to exactly solved models in statistical physics (Oxford University Press, Oxford, 2010).

[36] A. Bruce and R. Cowley, Structural Phase Transitions (Taylor & Francis, 1981).

[37] B. Sch¨olkopf, A. Smola, and K.-R. M¨uller, Nonlinear

component analysis as a kernel eigenvalue problem, Neural computation 10, 1299 (1998).

[38] G. Blanchard, O. Bousquet, and L. Zwald, Statistical properties of kernel principal component analysis, Machine Learning 66, 259 (2007).

[39] K. G. Wilson and J. Kogut, The renormalization group and the ϵ expansion, Physics Reports 12, 75 (1974).

[40] J. Zinn-Justin, Quantum field theory and critical phenomena, Vol. 171 (Oxford University Press, Oxford, 2021).

[41] A. Achille and S. Soatto, Emergence of invariance and disentanglement in deep representations, J. Mach. Learn. Res. 19, 1 (2018).

[42] P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, P.-A. Manzagol, and L. Bottou, Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion, J. Mach. Learn. Res. 11 (2010).

# Supplemental material for “Residual spectral instabilities in representation learning”

## Objective, notation, and stationary equations

We use the Gaussian VAE and the efective free energy $\mathcal { F }$ (negative ELBO) introduced in the main text. The data and latent variables are $\pmb { x } \in \mathbb { R } ^ { N }$ and $z \in \mathbb { R } ^ { n }$ , respectively, where $N > n$ . The data distribution is $p _ { \mathrm { d a t a } } ( \pmb { x } )$ , and the fixed decoder variance $\sigma ^ { \prime 2 } > 0$ serves as the control parameter. Throughout, $\sigma _ { j }$ denotes an encoder standard deviation and $\sigma _ { j } ^ { 2 }$ its variance. We use $\lambda _ { \alpha }$ for data covariance eigenvalues and $\Lambda _ { \alpha } ^ { ( A ) }$ for residual eigenvalues; ${ \Lambda } _ { * } ^ { ( A ) }$ denotes the upper edge of the residual spectrum. The latent prior, decoder, and diagonal variational posterior are

$$
p ( z ) : = \prod _ { j = 1 } ^ { n } \mathcal { N } ( z _ { j } ; 0 , 1 ) ,\tag{S1}
$$

$$
p _ { \theta } ( \pmb { x } | \boldsymbol { z } ) : = \prod _ { i = 1 } ^ { N } \mathcal { N } ( x _ { i } ; f _ { i } ( \boldsymbol { z } ) , \sigma ^ { \prime 2 } ) ,\tag{S2}
$$

$$
q _ { \phi } ( z | \mathbf { x } ) : = \prod _ { j = 1 } ^ { n } \mathcal { N } ( z _ { j } ; \mu _ { j } ( \mathbf { x } ) , \sigma _ { j } ^ { 2 } ( \mathbf { x } ) ) .\tag{S3}
$$

Up to an additive constant, the efective free energy reads

$$
\begin{array} { l } { { \displaystyle { \mathcal F } [ \Phi ] = \frac { 1 } { 2 \sigma ^ { \prime 2 } } \int d { \pmb x } d z p _ { \mathrm { d a t a } } ( { \pmb x } ) q _ { \phi } ( z | { \pmb x } ) \sum _ { i = 1 } ^ { N } [ x _ { i } - f _ { i } ( z ) ] ^ { 2 } } } \\ { { \displaystyle ~ + \frac { 1 } { 2 } \int d { \pmb x } p _ { \mathrm { d a t a } } ( { \pmb x } ) \sum _ { j } [ \mu _ { j } ( { \pmb x } ) ^ { 2 } + \sigma _ { j } ( { \pmb x } ) ^ { 2 } - \ln \sigma _ { j } ( { \pmb x } ) ^ { 2 } ] } . } \end{array}\tag{S4}
$$

We vary the learned fields $\Phi : = \{ f , \mu , \sigma \}$ independently. Data averages are denoted by

$$
\langle g \rangle _ { \pmb { x } } : = \int d \pmb { x } p _ { \mathrm { d a t a } } ( \pmb { x } ) g ( \pmb { x } ) .
$$

Varying the decoder mean and the Gaussian encoder parameters directly in Eq. (S4) gives the stationary equations [17]:

$$
0 = \int d { \pmb x } p _ { \mathrm { d a t a } } ( { \pmb x } ) q _ { \phi } ( { \pmb z } | { \pmb x } ) [ x _ { i } - f _ { i } ( { \pmb z } ) ] ,\tag{S5}
$$

$$
\mu _ { j } ( { \pmb x } ) = - \frac { 1 } { 2 \sigma ^ { \prime 2 } } \int d z q _ { \phi } ( z | { \pmb x } ) \sum _ { i } [ x _ { i } - f _ { i } ( z ) ] ^ { 2 } \frac { z _ { j } - \mu _ { j } ( { \pmb x } ) } { \sigma _ { j } ( { \pmb x } ) ^ { 2 } } ,\tag{S6}
$$

$$
\sigma _ { j } ( x ) - \sigma _ { j } ( x ) ^ { - 1 } = - \frac { 1 } { 2 \sigma ^ { \prime 2 } } \int d z q _ { \phi } ( z | x ) \sum _ { i } [ x _ { i } - f _ { i } ( z ) ] ^ { 2 } \left[ \frac { [ z _ { j } - \mu _ { j } ( x ) ] ^ { 2 } } { \sigma _ { j } ( x ) ^ { 3 } } - \frac { 1 } { \sigma _ { j } ( x ) } \right] .\tag{S7}
$$

Specifically,

$$
f _ { i } ( z ) = \frac { \int d \pmb { x } p _ { \mathrm { d a t a } } ( \pmb { x } ) q _ { \phi } ( z | \pmb { x } ) x _ { i } } { \int d \pmb { x } p _ { \mathrm { d a t a } } ( \pmb { x } ) q _ { \phi } ( z | \pmb { x } ) } ,\tag{S8}
$$

is the conditional mean of $x _ { i }$ under the joint distribution $p _ { \mathrm { d a t a } } ( { \pmb x } ) q _ { \phi } ( { \pmb z } | { \pmb x } )$

## Second variation at a general stationary point

Choose a stationary solution $\bar { \Phi } = ( \bar { f } , \bar { \mu } , \bar { \sigma } )$ as the background for a fluctuation expansion, and let $\bar { q }$ be the variational posterior evaluated on this background:

$$
\bar { q } ( \boldsymbol { z } | \mathbf { x } ) = \prod _ { j = 1 } ^ { n } \mathcal { N } ( z _ { j } ; \bar { \mu } _ { j } ( \mathbf { x } ) , \bar { \sigma } _ { j } ( \mathbf { x } ) ^ { 2 } ) ,\tag{S9}
$$

We write the residual field, standardized latent variable, and marginal density on the background as

$$
r _ { i } ( { \pmb x } , z ) = x _ { i } - \bar { f } _ { i } ( z ) ,\tag{S10}
$$

$$
\epsilon _ { j } ( { \pmb x } , z ) = \frac { z _ { j } - \bar { \mu } _ { j } ( { \pmb x } ) } { \bar { \sigma } _ { j } ( { \pmb x } ) } ,\tag{S11}
$$

$$
\rho ( z ) = \int d { \pmb x } p _ { \mathrm { d a t a } } ( { \pmb x } ) \bar { q } ( z | { \pmb x } ) ,\tag{S12}
$$

respectively.

Denote the fluctuations by $\delta f _ { i } ( z ) , \delta \mu _ { j } ( { \pmb x } )$ , and $\delta \sigma _ { j } ( { \pmb x } )$ , and their collection by $\delta \Phi : = ( \delta \pmb { f } ( \pmb { z } ) , \delta \pmb { \mu } ( \pmb { x } ) , \delta \pmb { \sigma } ( \pmb { x } ) )$ . Following the main text, we define the Hessian, or mass operator, of the efective free energy by

$$
K : = \left. \frac { \delta ^ { 2 } { \mathcal F } } { \delta \Phi ^ { 2 } } \right| _ { \bar { \Phi } } .\tag{S13}
$$

Nonnegative free-energy curvature corresponds to $\pmb { K } \succeq 0$ . The Gaussian fluctuation theory is defined by

$$
{ \mathcal { F } } [ { \bar { \Phi } } + \delta \Phi ] = { \mathcal { F } } [ { \bar { \Phi } } ] + { \frac { 1 } { 2 } } \langle \delta \Phi , K \delta \Phi \rangle + O ( \delta \Phi ^ { 3 } ) ,\tag{S14}
$$

We denote the quadratic form by $Q [ \delta \Phi ] : = \langle \delta \Phi , K \delta \Phi \rangle$ . Here the field inner product uses $L ^ { 2 } ( p _ { \mathrm { d a t a } } ( \pmb { x } ) \bar { q } ( \pmb { z } | \pmb { x } ) ; \mathbb { R } ^ { N + 2 n } )$ To display the second variation explicitly, introduce the local field vector $\delta \pmb { v } ( \pmb { x } , z ) : = ( \delta \pmb { f } ( z ) , \delta \pmb { \mu } ( \pmb { x } ) , \delta \pmb { \sigma } ( \pmb { x } ) ) ^ { \top }$ . Then

$$
Q [ \delta \Phi ] = \int d \pmb { x } d z p _ { \mathrm { d a t a } } ( \pmb { x } ) \bar { q } ( z | \pmb { x } ) \delta \pmb { v } ^ { \top } \pmb { k } ( \pmb { x } , z ) \delta \pmb { v } ,\tag{S15}
$$

$$
\begin{array} { r } { \pmb { k } : = \left( \begin{array} { l l l } { \pmb { k } _ { f f } } & { \pmb { k } _ { f \mu } } & { \pmb { k } _ { f \sigma } } \\ { \pmb { k } _ { f \mu } ^ { \top } } & { \pmb { k } _ { \mu \mu } } & { \pmb { k } _ { \mu \sigma } } \\ { \pmb { k } _ { f \sigma } ^ { \top } } & { \pmb { k } _ { \mu \sigma } ^ { \top } } & { \pmb { k } _ { \sigma \sigma } } \end{array} \right) . } \end{array}\tag{S16}
$$

The local coeficients k represent the free-energy quadratic form under this joint integral; the induced operator on the weighted field spaces is K. The coeficients are

$$
k _ { f f } ^ { i i ^ { \prime } } = \frac { 1 } { \sigma ^ { \prime 2 } } \delta _ { i i ^ { \prime } } ,\tag{S17}
$$

$$
k _ { f \mu } ^ { i j } = - { \frac { 1 } { \sigma ^ { \prime 2 } } } r _ { i } ( { \pmb x } , z ) { \frac { \epsilon _ { j } } { \bar { \sigma } _ { j } } } ,\tag{S18}
$$

$$
k _ { f \sigma } ^ { i j } = - \frac { 1 } { \sigma ^ { \prime 2 } } r _ { i } ( { \pmb x } , z ) \frac { \epsilon _ { j } ^ { 2 } - 1 } { \bar { \sigma } _ { j } } ,\tag{S19}
$$

$$
k _ { \mu \mu } ^ { j k } = \delta _ { j k } + \frac { 1 } { 2 \sigma ^ { \prime 2 } } \sum _ { i } r _ { i } ^ { 2 } \frac { \epsilon _ { j } \epsilon _ { k } - \delta _ { j k } } { { \bar { \sigma } _ { j } } { \bar { \sigma } _ { k } } } ,\tag{S20}
$$

$$
k _ { \mu \sigma } ^ { j k } = \frac { 1 } { 2 \sigma ^ { \prime 2 } } \sum _ { i } r _ { i } ^ { 2 } \frac { \epsilon _ { j } ( \epsilon _ { k } ^ { 2 } - 1 ) - 2 \delta _ { j k } \epsilon _ { j } } { \bar { \sigma } _ { j } \bar { \sigma } _ { k } } ,\tag{S21}
$$

$$
k _ { \sigma \sigma } ^ { j k } = ( 1 + \bar { \sigma } _ { j } ^ { - 2 } ) \delta _ { j k } + \frac { 1 } { 2 \sigma ^ { \prime 2 } } \sum _ { i } r _ { i } ^ { 2 } \frac { ( \epsilon _ { j } ^ { 2 } - 1 ) ( \epsilon _ { k } ^ { 2 } - 1 ) + \delta _ { j k } ( 1 - 3 \epsilon _ { j } ^ { 2 } ) } { \bar { \sigma } _ { j } \bar { \sigma } _ { k } } .\tag{S22}
$$

## Partial collapse and the invariant Hessian blocks

Besides the fully collapsed trivial solution, the stationary equations (S5)–(S7) also give partially collapsed background,

$$
\bar { f } _ { i } ( z ) = \bar { f } _ { i } ^ { A } ( z _ { A } ) ,\tag{S23}
$$

$$
\bar { \mu } _ { c } ( { \pmb x } ) = 0 \quad ( c \in { \cal C } ) ,\tag{S24}
$$

$$
\bar { \sigma } _ { c } ( { \pmb x } ) = 1 \quad ( c \in { \cal C } ) ,\tag{S25}
$$

where the active set A and the collapsed set C partition the latent indices $\{ 1 , \ldots , n \}$

$$
z = ( z _ { A } , z _ { C } ) .\tag{S26}
$$

We use $a \in A$ and $c \in C .$ while $i , i ^ { \prime }$ label data components and $j ,$ k label arbitrary latent components.

Moreover, the encoder can be decomposed as

$$
\bar { q } (  { \boldsymbol { z } } |  { \boldsymbol { { x } } } ) = \bar { q } _ { A } (  { \boldsymbol { { z } } } _ { A } |  { \boldsymbol { { x } } } ) \varphi _ { C } (  { \boldsymbol { { z } } } _ { C } ) ,\tag{S27}
$$

where φ<sub>C</sub> is the standard |C|-dimensional Gaussian density, and the residual reduces to the main-text field $r _ { i } ( { \pmb x } , z _ { A } ) : =$ $x _ { i } - \bar { f } _ { i } ^ { A } ( z _ { A } )$ . We may also write the marginal density for the active latent variables on the background as

$$
\rho _ { A } ( z _ { A } ) = \int d { \pmb x } p _ { \mathrm { d a t a } } ( { \pmb x } ) \bar { q } _ { A } ( z _ { A } | { \pmb x } ) .\tag{S28}
$$

It is easy to discover that there is no active-collapse coupling in the Hessian (S16) for perturbations on encoders $\delta \mu$ and $\delta \pmb { \sigma } .$ by splitting them into active set A and collapsed set $C \colon$

$$
\delta \pmb { \mu } = ( \delta \pmb { \mu } _ { A } , \delta \pmb { \mu } _ { C } ) ,\tag{S29}
$$

$$
\delta \pmb { \sigma } \ = ( \delta \pmb { \sigma } _ { A } , \delta \pmb { \sigma } _ { C } ) .\tag{S30}
$$

We can also orthogonal project $\delta f _ { i } ( z )$ in $L ^ { 2 } ( \rho _ { A } \varphi _ { C } )$ to separate the efects for active parts

$$
\delta f _ { i } ^ { \parallel } = P _ { A } \delta f _ { i } \equiv \int d z _ { C } \varphi _ { C } \delta f _ { i } ,\tag{S31}
$$

which only holds $z _ { A }$ dependent, and collapsed parts

$$
\delta f _ { i } ^ { \perp } = ( 1 - P _ { A } ) \delta f _ { i } .\tag{S32}
$$

Here, $P _ { A } \delta f _ { i } ^ { \perp } = 0$ , and

$$
\int d z _ { A } d z _ { C } \rho _ { A } \varphi _ { C } \delta f _ { i } ^ { \parallel } \delta f _ { i } ^ { \perp } = 0 ,\tag{S33}
$$

cancelling the couplings between $\delta { f } ^ { \parallel }$ and $\delta { f ^ { \perp } }$ in the Hessian (S16). Moreover, the kernels with active encoder fields have no $z _ { C }$ dependence except $\delta f _ { i } ^ { \perp }$ , while the kernels with collapsed encoder fields contain $z _ { c } \ \mathrm { o r } \ z _ { c } ^ { 2 } - 1$ whose Gaussian averages vanish. Thus, the couplings of $\delta { \pmb f } ^ { \| } - \delta { \pmb \mu } _ { C } , \delta { \pmb f } ^ { \| } - \delta { \pmb \sigma } _ { C } , \delta { \pmb f } ^ { \perp } - \delta { \pmb \mu } _ { A }$ , and $\delta { \pmb f } ^ { \perp } { - } \delta { \pmb \sigma } _ { A }$ also vanish.

We can rewrite $\delta \Phi$ as

$$
\delta \Phi = ( \delta \Phi _ { A } , \delta \Phi _ { C } ) ,\tag{S34}
$$

where

$$
\delta \Phi _ { A } : = ( \delta { \pmb f } ^ { \parallel } , \delta { \pmb \mu } _ { A } , \delta { \pmb \sigma } _ { A } ) ,
$$

$$
\delta \Phi _ { C } : = ( \delta { \pmb f } ^ { \perp } , \delta { \pmb \mu } _ { C } , \delta { \pmb \sigma } _ { C } ) ,\tag{S35}
$$

(S36)

the Hessian separates into the active and collapsed invariant fluctuation sectors, represented by $\delta \Phi _ { A }$ and $\delta \Phi _ { C } \colon$

$$
K = K _ { A } \oplus K _ { C } , \qquad Q = Q _ { A } + Q _ { C } .\tag{S37}
$$

Substituting the partially collapsed background into the kernels gives:

$$
\begin{array} { l } { { \displaystyle Q _ { C } = \frac { 1 } { \sigma ^ { \prime 2 } } \int d z _ { A } d z _ { C } \rho _ { A } ( z _ { A } ) \varphi _ { C } ( z _ { C } ) \sum _ { i } [ \delta f _ { i } ^ { \perp } ( z _ { A } , z _ { C } ) ] ^ { 2 } } } \\ { { \displaystyle ~ + \sum _ { c \in { \cal C } } \int d x p _ { \mathrm { d a t a } } ( x ) \left[ ( \delta \mu _ { c } ( x ) ) ^ { 2 } + 2 ( \delta \sigma _ { c } ( x ) ) ^ { 2 } \right] } } \\ { { \displaystyle ~ - \frac { 2 } { \sigma ^ { \prime 2 } } \sum _ { c \in { \cal C } } \sum _ { i } \int d x d z _ { A } d z _ { C } p _ { \mathrm { d a t a } } ( x ) \bar { q } _ { A } ( z _ { A } | { \bf x } ) \varphi _ { C } ( z _ { C } ) } } \\ { { \displaystyle ~ \times ~ r _ { i } ( x , z _ { A } ) \delta f _ { i } ^ { \perp } ( z _ { A } , z _ { C } ) \left[ z _ { c } \delta \mu _ { c } ( x ) + ( z _ { c } ^ { 2 } - 1 ) \delta \sigma _ { c } ( x ) \right] } . } \end{array}\tag{S38}
$$

## Conditional residual operator and collapsed-sector stability

We now analyze the collapsed-sector fluctuations $\delta \Phi _ { C }$ through $Q _ { C }$ in Eq. (S38), with the active background $\bar { f } _ { i } ^ { A }$ ${ \bar { q } } _ { A }$ , and $\rho _ { A }$ held fixed. The coupling to $\delta \mu _ { c }$ and $\delta \sigma _ { c }$ involves only $z _ { c }$ and $z _ { c } ^ { 2 } - 1$ . For each $c \in C$ , introduce the normalized Hermite polynomials,

$$
h _ { 1 } ^ { c } = z _ { c } , \qquad h _ { 2 } ^ { c } = { \frac { z _ { c } ^ { 2 } - 1 } { \sqrt 2 } } ,\tag{S39}
$$

which obey

$$
\int d z _ { C } \varphi _ { C } h _ { 1 } ^ { c } h _ { 1 } ^ { d } = \int d z _ { C } \varphi _ { C } h _ { 2 } ^ { c } h _ { 2 } ^ { d } = \delta _ { c d } ,\tag{S40}
$$

and,

$$
\int d z _ { C } \varphi _ { C } h _ { 1 } ^ { c } h _ { 2 } ^ { d } = 0 .\tag{S41}
$$

Using the same projection $P _ { A }$ as in Eq. (S31), we can expand the collapsed-sector decoder fluctuation $\delta f _ { i } ^ { \perp }$ in Hermite modes as

$$
\delta f _ { i } ^ { \perp } = \sum _ { c \in C } \left( h _ { 1 } ^ { c } \delta f _ { i c } ^ { ( 1 ) } + h _ { 2 } ^ { c } \delta f _ { i c } ^ { ( 2 ) } \right) + \delta f _ { i } ^ { \mathrm { r e s t } } ,\tag{S42}
$$

where

$$
\delta f _ { i c } ^ { ( 1 ) } ( z _ { A } ) = \int d z _ { C } \varphi _ { C } h _ { 1 } ^ { c } \delta f _ { i } ^ { \perp } ,\tag{S43}
$$

$$
\delta f _ { i c } ^ { ( 2 ) } ( z _ { A } ) = \int d z _ { C } \varphi _ { C } h _ { 2 } ^ { c } \delta f _ { i } ^ { \perp } ,\tag{S44}
$$

and the remainder $\delta f _ { i } ^ { \mathrm { r e s t } }$ is orthogonal to $1 , h _ { 1 } ^ { c } , h _ { 2 } ^ { c }$ at each $z _ { A }$

The encoder perturbations and the decoder coeficients in Eq. (S42) are functions of diferent variables. For each $c \in C$ , the scalar functions $\delta \mu _ { c } ( { \pmb x } )$ and $\delta \sigma _ { c } ( { \pmb x } )$ belong to the real weighted Hilbert space $\mathcal { H } _ { x } = L ^ { 2 } ( p _ { \mathrm { d a t a } } )$ , while the collections $\{ \delta f _ { i c } ^ { ( 1 ) } ( z _ { A } ) \}$ <sub>i</sub> and $\{ \delta f _ { i c } ^ { ( 2 ) } ( z _ { A } ) \} _ { i }$ belong to $\mathcal { H } _ { A } = L ^ { 2 } ( \rho _ { A } ; \mathbb { R } ^ { N } )$ . For $h \in \mathcal { H } _ { x }$ and $\pmb { \mathscr { g } } = ( g _ { 1 } , \dots , g _ { N } ) \in \mathcal { H } _ { A }$ , the corresponding squared norms are

$$
\| h \| _ { x } ^ { 2 } = \langle h ^ { 2 } \rangle _ { \pmb { x } } = \int d \pmb { x } p _ { \mathrm { d a t a } } ( \pmb { x } ) h ( \pmb { x } ) ^ { 2 } ,\tag{S45}
$$

$$
\| { \pmb g } \| _ { A } ^ { 2 } = \mathbb { E } _ { \rho _ { A } } \| { \pmb g } \| ^ { 2 } = \sum _ { i } \int d z _ { A } \rho _ { A } ( z _ { A } ) g _ { i } ( z _ { A } ) ^ { 2 } .\tag{S46}
$$

The inner products use the same integrals, with each square replaced by the product of the two functions. For an individual decoder coeficient or a remainder, we also use the scalar norms

$$
\| a \| _ { \rho _ { A } } ^ { 2 } = \int d z _ { A } \rho _ { A } ( z _ { A } ) a ( z _ { A } ) ^ { 2 } ,\tag{S47}
$$

$$
\| b \| _ { \rho _ { A } \varphi _ { C } } ^ { 2 } = \int d z _ { A } d z _ { C } \rho _ { A } ( z _ { A } ) \varphi _ { C } ( z _ { C } ) b ( z _ { A } , z _ { C } ) ^ { 2 } .\tag{S48}
$$

The mixed terms in $Q _ { C }$ couple these two spaces through the residual field $r _ { i }$ . To express this coupling, define the conditional residual operator $\mathcal { R } _ { A } : \mathcal { H } _ { x }  \mathcal { H } _ { A }$ by

$$
( \mathcal { R } _ { A } h ) _ { i } ( z _ { A } ) = \frac { \int d \pmb { x } p _ { \mathrm { d a t a } } ( \pmb { x } ) \bar { q } _ { A } ( z _ { A } | \pmb { x } ) r _ { i } ( \pmb { x } , z _ { A } ) h ( \pmb { x } ) } { \rho _ { A } ( z _ { A } ) } .\tag{S49}
$$

Equivalently, $( \mathcal { R } _ { A } h ) _ { i } ( z _ { A } ) = \mathbb { E } [ r _ { i } ( \pmb { x } , z _ { A } ) h ( \pmb { x } ) \ | \ z _ { A } ]$ , with expectation under $p _ { \mathrm { d a t a } } ( { \pmb x } ) \bar { q } _ { A } ( { \pmb z } _ { A } | { \pmb x } )$ , as in the main text. Thus $\mathcal { R } _ { A }$ takes a scalar encoder perturbation, such as $\delta \mu _ { c }$ or $\delta \sigma _ { c } ,$ to an N-component function of $z _ { A }$ as the decoder

coeficients. The definition is understood on the support of $\rho _ { A }$ and for functions for which the displayed integrals are finite. The factor $\rho _ { A } ^ { - 1 }$ converts the coupling integral into the weighted inner product in $\mathcal { H } _ { A }$

The adjoint $\mathcal { R } _ { A } ^ { \dagger } : \mathcal { H } _ { A }  \mathcal { H } _ { x }$ is determined by transferring $\mathcal { R } _ { A }$ from one side of this inner product to the other. Substituting Eq. (S49) and interchanging the integrations gives

$$
\begin{array} { l } { { \displaystyle \sum _ { i } \int d z _ { A } \rho _ { A } ( z _ { A } ) g _ { i } ( z _ { A } ) ( \mathcal { R } _ { A } h ) _ { i } ( z _ { A } ) } } \\ { { \displaystyle ~ = \int d x p _ { \mathrm { d a t a } } ( \boldsymbol { x } ) h ( \boldsymbol { x } ) \left[ \sum _ { i } \int d z _ { A } \bar { q } _ { A } ( z _ { A } | \boldsymbol { x } ) r _ { i } ( \boldsymbol { x } , z _ { A } ) g _ { i } ( z _ { A } ) \right] } } \\ { { \displaystyle ~ = \int d x p _ { \mathrm { d a t a } } ( \boldsymbol { x } ) h ( \boldsymbol { x } ) ( \mathcal { R } _ { A } ^ { \dagger } g ) ( \boldsymbol { x } ) . } } \end{array}\tag{S50}
$$

Comparing the last two lines identifies the adjoint as

$$
( \mathcal { R } _ { A } ^ { \dagger } \pmb { g } ) ( \pmb { x } ) = \sum _ { i } \int d z _ { A } \bar { q } _ { A } ( z _ { A } | \pmb { x } ) r _ { i } ( \pmb { x } , z _ { A } ) g _ { i } ( z _ { A } ) ,\tag{S51}
$$

which maps the decoder coeficients back to a scalar function of ${ \bf \delta } _ { { \bf \alpha } } \mathbf { \delta } _ { { \bf \alpha } }$

Equation (S38) now becomes

$$
\begin{array} { l } { { \displaystyle Q _ { C } = \frac { 1 } { \sigma ^ { \prime 2 } } \sum _ { i } \int d z _ { A } d z _ { C } \rho _ { A } ( z _ { A } ) \varphi _ { C } ( z _ { C } ) [ \delta f _ { i c } ^ { \mathrm { r e s t } } ] ^ { 2 } } } \\ { { \displaystyle ~ + \sum _ { c \in C } \int d x p _ { \mathrm { d a t a } } ( x ) \big [ ( \delta \mu _ { c } ) ^ { 2 } + 2 ( \delta \sigma _ { c } ) ^ { 2 } \big ] } } \\ { { \displaystyle ~ + \frac { 1 } { \sigma ^ { \prime 2 } } \sum _ { c \in C } \sum _ { i } \int d z _ { A } \rho _ { A } ( z _ { A } ) \Big [ ( \delta f _ { i c } ^ { ( 1 ) } ) ^ { 2 } + ( \delta f _ { i c } ^ { ( 2 ) } ) ^ { 2 } } } \\ { { \displaystyle ~ - 2 \delta f _ { i c } ^ { ( 1 ) } ( \mathcal { R } _ { A } \delta \mu _ { c } ) _ { i } - 2 \sqrt { 2 } \delta f _ { i c } ^ { ( 2 ) } ( \mathcal { R } _ { A } \delta \sigma _ { c } ) _ { i } \Big ] . } } \end{array}\tag{S52}
$$

Thus each collapsed coordinate contributes independent posterior-mean and posterior-width mass blocks. We parameterize the latter by the standard-deviation fluctuation $\delta \sigma _ { c } ,$ with $\delta ( \sigma _ { c } ^ { 2 } ) = 2 \delta \sigma _ { c }$ at the collapsed background. These blocks act respectively on $( \{ \delta f _ { i c } ^ { ( 1 ) } \} _ { i } , \delta \mu _ { c } )$ and $( \{ \delta f _ { i c } ^ { ( 2 ) } \} _ { i } , \delta \sigma _ { c } )$ :

$$
{ \pmb { K } } _ { C , \mu } = \left( { \begin{array} { c c } { \sigma ^ { \prime - 2 } I } & { - \sigma ^ { \prime - 2 } { \mathcal { R } } _ { A } } \\ { - \sigma ^ { \prime - 2 } { \mathcal { R } } _ { A } ^ { \dagger } } & { I } \end{array} } \right) ,\tag{S53}
$$

$$
{ \cal K } _ { C , \sigma } = \left( { \ L } { \sigma } _ { - \sqrt 2 \sigma ^ { \prime - 2 } \mathcal { R } _ { A } ^ { \dagger } } ^ { \prime - 2 } \right. \left. - \sqrt 2 \sigma ^ { \prime - 2 } \mathcal { R } _ { A } \right) .\tag{S54}
$$

Here the matrices represent the mass operators induced by $Q _ { C }$ in the weighted function spaces above, after integration over the collapsed Gaussian variables. They are not pointwise matrices at a fixed $( x , z )$ . The remainder $\delta f _ { i } ^ { \mathrm { r e s t } }$ contributes the positive mass block $\sigma ^ { \prime - 2 } I$ to $K _ { C }$ . Thus these remaining modes are strictly massive, and the collapsed sector decomposes as

$$
K _ { C } \simeq \bigoplus _ { c \in C } \left( K _ { C , \mu } \oplus K _ { C , \sigma } \right) \oplus \sigma ^ { \prime - 2 } I _ { \mathrm { r e s t } } .\tag{S55}
$$

Define the positive semi-definite Gram operator

$$
\mathcal { C } _ { A } = \mathcal { R } _ { A } ^ { \dagger } \mathcal { R } _ { A } ,\tag{S56}
$$

whose spectrum is the residual spectrum of the partially collapsed representation. Completing squares, or taking the

Schur complement of the decoder blocks, gives the exact identity

$$
\begin{array} { l } { { Q _ { C } = \displaystyle \frac { 1 } { \sigma ^ { \prime 2 } } \sum _ { i } \int d z _ { A } d z _ { C } \rho _ { A } ( z _ { A } ) \varphi _ { C } ( z _ { C } ) [ \delta f _ { i } ^ { \mathrm { r e s t } } ] ^ { 2 } } } \\ { { \displaystyle \qquad + \frac { 1 } { \sigma ^ { \prime 2 } } \sum _ { c \in C } \sum _ { i } \int d z _ { A } \rho _ { A } ( z _ { A } ) \big [ \delta f _ { i c } ^ { \mathrm { i } ( 1 ) } - ( { \mathcal R } _ { A } \delta \mu _ { c } ) _ { i } \big ] ^ { 2 } } } \\ { { \displaystyle \qquad + \frac { 1 } { \sigma ^ { \prime 2 } } \sum _ { c \in C } \sum _ { i } \int d z _ { A } \rho _ { A } ( z _ { A } ) \big [ \delta f _ { i c } ^ { \mathrm { ( 2 ) } } - \sqrt { 2 } ( { \mathcal R } _ { A } \delta \sigma _ { c } ) _ { i } \big ] ^ { 2 } } } \\ { { \displaystyle \qquad + \sum _ { c \in C } \int d x p _ { \mathrm { d a x } \delta ( { \pmb x } ) } \Big [ \delta \mu _ { c } ( { \pmb x } ) \big [ ( I - C _ { A } / \sigma ^ { \prime 2 } ) \delta \mu _ { c } \big ] ( { \pmb x } ) } } \\ { { \displaystyle \qquad + 2 \delta \sigma _ { c } ( { \pmb x } ) \big [ ( I - C _ { A } / \sigma ^ { \prime 2 } ) \delta \sigma _ { c } \big ] ( { \pmb x } ) \Big ] . } } \end{array}\tag{S57}
$$

Assume $\mathcal { R } _ { A }$ is bounded. The leading residual eigenvalue (more generally, the supremum of the residual spectrum) is

$$
\Lambda _ { * } ^ { ( A ) } = \| \mathcal { R } _ { A } \| ^ { 2 } = \operatorname* { s u p } _ { h \neq 0 } \sum _ { i } \frac { \int d z _ { A } \rho _ { A } ( z _ { A } ) [ ( \mathcal { R } _ { A } h ) _ { i } ( z _ { A } ) ] ^ { 2 } } { \int d x p _ { \mathrm { d a t a } } ( x ) h ( x ) ^ { 2 } } = \operatorname* { s u p } \operatorname { s p e c } \mathcal { C } _ { A } .\tag{S58}
$$

The necessary and suficient condition for nonnegative curvature in the collapsed sector is therefore

$$
\boxed { K _ { C } \succeq 0 \quad \Longleftrightarrow \quad \sigma ^ { \prime 2 } \geq \Lambda _ { * } ^ { ( A ) } . }\tag{S59}
$$

For the complete stationary solution, one must also require ${ \pmb K } _ { A } \succeq 0$ . Since $Q = Q _ { A } + Q _ { C }$ , the collapsed perturbations can be analyzed independently of $\delta \Phi _ { A }$ at quadratic order. Their stability still depends on the active background through $\mathcal { R } _ { A }$

## Mass spectrum and soft modes

For a positive residual eigenvalue, choose ${ \mathcal C } _ { A } \psi _ { \alpha } = \Lambda _ { \alpha } ^ { ( A ) } \psi _ { \alpha }$ with $\langle \psi _ { \alpha } ^ { 2 } \rangle _ { \pmb { x } } = 1$ , and define its decoder partner by

$$
\chi _ { \alpha } = \frac { \mathcal { R } _ { A } \psi _ { \alpha } } { \sqrt { \Lambda _ { \alpha } ^ { ( A ) } } } .\tag{S60}
$$

Then $\mathbb { E } _ { \rho _ { A } } \| \chi _ { \alpha } \| ^ { 2 } = 1$ and $\mathcal { R } _ { A } ^ { \dagger } \chi _ { \alpha } = \sqrt { \Lambda _ { \alpha } ^ { ( A ) } \psi _ { \alpha } }$ . The two-dimensional subspace

$$
\mathcal { V } _ { \alpha } = \operatorname { s p a n } \left\{ ( \chi _ { \alpha } , 0 ) , ( 0 , \psi _ { \alpha } ) \right\}\tag{S61}
$$

is invariant under each mass operator. Its first component is the first Hermite decoder coeficient for the mean channel and the second Hermite coeficient for the width channel. In this orthonormal basis, the restrictions of the operators have the matrix representations

$$
\begin{array} { r l } & { K _ { C , \mu } | _ { \mathcal { V } _ { \alpha } } \hat { = } M _ { \mu , \alpha } : = \left( \begin{array} { c c } { 1 / \sigma ^ { \prime 2 } } & { - \sqrt { \Lambda _ { \alpha } ^ { ( A ) } } / \sigma ^ { \prime 2 } } \\ { - \sqrt { \Lambda _ { \alpha } ^ { ( A ) } } / \sigma ^ { \prime 2 } } & { 1 } \end{array} \right) , } \\ & { K _ { C , \sigma } | _ { \mathcal { V } _ { \alpha } } \hat { = } M _ { \sigma , \alpha } : = \left( \begin{array} { c c } { 1 / \sigma ^ { \prime 2 } } & { - \sqrt { 2 \Lambda _ { \alpha } ^ { ( A ) } } / \sigma ^ { \prime 2 } } \\ { - \sqrt { 2 \Lambda _ { \alpha } ^ { ( A ) } } / \sigma ^ { \prime 2 } } & { 2 } \end{array} \right) . } \end{array}\tag{S62}
$$

Here $\widehat { = }$ denotes a matrix representation on $\gamma _ { \alpha } ; { \cal K } _ { { \cal C } , \mu }$ and $K _ { C , \sigma }$ retain their operator meaning. The matrix $M _ { \mu , \alpha }$ is the mean-channel block used in the main text; $M _ { \sigma , \alpha }$ denotes the corresponding width-channel block. For a single collapsed coordinate, the mode amplitudes $u , v$ in the main text give $\delta { \pmb f } ^ { \perp } = u z _ { c } { \pmb \chi } _ { c }$ and $\delta \mu _ { c } = v \psi _ { \alpha }$ , with quadratic

free-energy contribution $\begin{array} { r } { \frac { 1 } { 2 } ( u , v ) M _ { \mu , \alpha } ( u , v ) ^ { \mathsf { T } } } \end{array}$ . Diagonalization gives

$$
m _ { \mu , \alpha , \pm } ^ { 2 } = \frac { 1 + \sigma ^ { \prime 2 } \pm \sqrt { ( 1 - \sigma ^ { \prime 2 } ) ^ { 2 } + 4 \Lambda _ { \alpha } ^ { ( A ) } } } { 2 \sigma ^ { \prime 2 } } ,\tag{S63}
$$

$$
m _ { \sigma , \alpha , \pm } ^ { 2 } = \frac { 1 + 2 \sigma ^ { \prime 2 } \pm \sqrt { ( 1 - 2 \sigma ^ { \prime 2 } ) ^ { 2 } + 8 \Lambda _ { \alpha } ^ { ( A ) } } } { 2 \sigma ^ { \prime 2 } } .\tag{S64}
$$

Both traces are positive, while the determinants are respectively $( \sigma ^ { \prime 2 } - \Lambda _ { \alpha } ^ { ( A ) } ) / \sigma ^ { \prime 4 }$ and $2 ( \sigma ^ { \prime 2 } - \Lambda _ { \alpha } ^ { ( A ) } ) / \sigma ^ { \prime 4 }$ . Thus both lower branches vanish at $\sigma ^ { \prime 2 } = \Lambda _ { \alpha } ^ { ( A ) }$ and become negative below it. At equality, their zero-mode amplitudes obey

$$
\delta f _ { c } ^ { ( 1 ) } = \mathcal { R } _ { A } \delta \mu _ { c } ,\tag{S65}
$$

$$
\delta { \bf f } _ { c } ^ { ( 2 ) } = \sqrt { 2 } \mathcal { R } _ { A } \delta \sigma _ { c } ,\tag{S66}
$$

where $\delta \mathbf { { f } } _ { c } ^ { ( k ) } : = \{ \delta { f _ { i c } ^ { ( k ) } } \} _ { i = 1 } ^ { N }$ for $k = 1 , 2 ,$ , and the encoder perturbation is proportional to $\psi _ { \alpha }$ . These relations also follow from the completed squares in Eq. (S57). Each mass branch is repeated |C| times, in addition to the multiplicity of the residual eigenvalue.

Encoder modes in ker $\mathcal { R } _ { A }$ have squared masses 1 and 2 in the mean and width channels, respectively. Decoder modes in ker $\mathcal { R } _ { A } ^ { \dagger }$ , together with the remaining Hermite modes, have squared mass $\sigma ^ { \prime - 2 }$ . Hence the stability boundary is controlled by the upper edge of the residual spectrum. If that edge is attained by an eigenmode, marginality produces the zero modes above; otherwise it marks closure of the spectral gap without a normalizable eigenmode at zero.

## Interpretation and the fully collapsed limit

For the active background, Eq. (S5) gives a conditional expectation of $x _ { i }$ under the distribution $p _ { \mathrm { d a t a } } ( { \pmb x } ) \bar { q } _ { A } ( { \pmb z } _ { A } | { \pmb x } )$

$$
\bar { f } _ { i } ^ { A } ( z _ { A } ) = \frac { \int d \pmb { x } p _ { \mathrm { d a t a } } ( \pmb { x } ) \bar { q } _ { A } ( z _ { A } | \pmb { x } ) x _ { i } } { \rho _ { A } ( z _ { A } ) } .\tag{S67}
$$

Consequently, the residual operator can be written as

$$
( \mathcal { R } _ { A } h ) _ { i } ( z _ { A } ) = \frac { \int d \pmb { x } p _ { \mathrm { d a t a } } ( \pmb { x } ) \bar { q } _ { A } ( z _ { A } | \pmb { x } ) x _ { i } h ( \pmb { x } ) } { \rho _ { A } ( z _ { A } ) }\tag{S68}
$$

which is the conditional covariance of $x _ { i }$ and $h ( { \pmb x } )$ at fixed $z _ { A }$ under $p _ { \mathrm { d a t a } } ( { \pmb x } ) \bar { q } _ { A } ( { \pmb z } _ { A } | { \pmb x } )$ . In particular, $\mathcal { R } _ { A } 1 = 0 .$ , and

$$
\Lambda _ { * } ^ { ( A ) } = \operatorname* { s u p } _ { \langle h ^ { 2 } \rangle _ { \infty } = 1 } \int d z _ { A } \rho _ { A } ( z _ { A } ) \sum _ { i } \left[ \frac { \int d x p _ { \mathrm { d a t a } } ( x ) \bar { q } _ { A } ( z _ { A } | x ) [ x _ { i } - \bar { f } _ { i } ^ { A } ( z _ { A } ) ] h ( x ) } { \rho _ { A } ( z _ { A } ) } \right] ^ { 2 } .\tag{S69}
$$

The leading eigenvalue of the data covariance matrix admits the same function-space variational formulation. For finite data covariance $\Sigma _ { X }$

$$
\lambda _ { \operatorname* { m a x } } \bigl ( \Sigma _ { X } \bigr ) = \operatorname* { s u p } _ { \langle h ^ { 2 } \rangle _ { \boldsymbol { \alpha } } = 1 } \sum _ { i } \left[ \int d \boldsymbol { x } p _ { \mathrm { d a t a } } ( \boldsymbol { x } ) \bigl ( x _ { i } - \langle \boldsymbol { x } _ { i } \rangle _ { \boldsymbol { x } } \bigr ) h ( \boldsymbol { x } ) \right] ^ { 2 } ,\tag{S70}
$$

gives the leading covariance eigenvalue [38]. Analogously, the leading residual eigenvalue ${ \Lambda } _ { * } ^ { ( A ) }$ measures the strongest residual correlation accessible to an additional latent degree of freedom given the active representation.

For a fully collapsed stationary state, $A = \emptyset$ , we have

$$
( \mathcal { R } _ { \mathcal { Q } } h ) _ { i } = \int d \pmb { x } p _ { \mathrm { d a t a } } ( \pmb { x } ) ( \boldsymbol { x } _ { i } - \langle \boldsymbol { x } _ { i } \rangle _ { \pmb { x } } ) h ( \pmb { x } ) , \qquad \Lambda _ { * } ^ { ( \mathcal { Q } ) } = \lambda _ { \mathrm { m a x } } ( \pmb { \Sigma } _ { X } ) .\tag{S71}
$$

This recovers the full-collapse threshold in Ref. [17].

## Linear perturbations and the linear VAE

We evaluate the residual spectrum for linear fluctuations directly at a partially collapsed stationary state of the linear Gaussian VAE. Let the data be Gaussian with mean $\langle { \pmb x } \rangle _ { \pmb x }$ and covariance $\Sigma _ { X }$ . Denote its positive eigenvalues and orthonormal eigenvectors by $\begin{array} { r } { \lambda _ { \alpha } = \lambda _ { \alpha } ( \pmb { \Sigma } _ { X } ) } \end{array}$ and $e _ { \alpha }$ , with $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \dots > 0$ , and write

$$
y _ { \alpha } ( { \pmb x } ) : = \sum _ { i } e _ { \alpha i } ( x _ { i } - \langle x _ { i } \rangle _ { \pmb x } ) , \qquad \langle y _ { \alpha } y _ { \beta } \rangle _ { \pmb x } = \lambda _ { \alpha } \delta _ { \alpha \beta } .\tag{S72}
$$

All data directions below are restricted to the support of $\Sigma _ { X }$ . We label each active latent coordinate by the principal direction it represents, so that A also labels the represented principal directions, as in the main text. The notation $\alpha \not \in A$ refers to unrepresented data directions; $c \in C$ continues to label collapsed latent coordinates.

With learned encoder variances, the stationary solution is [25]

$$
\bar { f } _ { i } ^ { A } ( z _ { A } ) = \langle x _ { i } \rangle _ { x } + \sum _ { a \in A } \sqrt { \lambda _ { a } - \sigma ^ { \prime 2 } } e _ { a i } z _ { a } ,\tag{S73}
$$

$$
\bar { \mu } _ { a } ( { \pmb x } ) = \frac { \sqrt { \lambda _ { a } - \sigma ^ { \prime 2 } } } { \lambda _ { a } } y _ { a } ( { \pmb x } ) ,\tag{S74}
$$

$$
\bar { \sigma } _ { a } ^ { 2 } = { \frac { \sigma ^ { \prime 2 } } { \lambda _ { a } } } ,\tag{S75}
$$

where $\lambda _ { a } > \sigma ^ { \prime 2 }$ , while $\bar { \mu } _ { c } = 0$ and $\bar { \sigma } _ { c } = 1$ for $c \in C$ . Under the joint distribution $p _ { \mathrm { d a t a } } ( { \pmb x } ) \bar { q } _ { A } ( { \pmb z } _ { A } | { \pmb x } )$ , the active latent marginal $\rho _ { A }$ is standard Gaussian. Gaussian conditioning gives

$$
\mathbb { E } [ y _ { \alpha } \mid z _ { A } ] = { \left\{ \begin{array} { l l } { { \sqrt { \lambda _ { \alpha } - \sigma ^ { \prime 2 } } } z _ { \alpha } , } & { \alpha \in A , } \\ { 0 , } & { \alpha \not \in A , } \end{array} \right. }\tag{S76}
$$

$$
\operatorname { C o v } ( y _ { \alpha } , y _ { \beta } \mid z _ { A } ) = \delta _ { \alpha \beta } \left\{ { \sigma } ^ { \prime 2 } , \alpha \in A , \right.\tag{S77}
$$

In particular, $\bar { f } _ { i } ^ { A } = \mathbb { E } [ x _ { i } \mid z _ { A } ]$ , so the residual field is the same $r _ { i } = x _ { i } - \bar { f } _ { i } ^ { A }$ used above.

To evaluate the conditional residual operator, choose the normalized linear encoder modes and constant decoder modes

$$
\psi _ { \alpha } ( { \pmb x } ) = \frac { y _ { \alpha } ( { \pmb x } ) } { \sqrt { \lambda _ { \alpha } } } , \qquad \chi _ { \alpha } ( { \pmb z } _ { A } ) = { \pmb e } _ { \alpha } .\tag{S78}
$$

They are orthonormal in $\mathcal { H } _ { x }$ and $\mathcal { H } _ { A }$ , respectively. Substituting Eq. (S77) into the conditional residual operator gives

$$
( \mathcal { R } _ { A } \psi _ { \alpha } ) _ { i } = \frac { \mathbb { E } [ r _ { i } y _ { \alpha } \mid z _ { A } ] } { \sqrt { \lambda _ { \alpha } } } = \sqrt { \Lambda _ { \alpha } ^ { ( A ) } } \chi _ { \alpha i } ,\tag{S79}
$$

$$
\sqrt { \Lambda _ { \alpha } ^ { ( A ) } } = \left\{ \begin{array} { l l } { \sigma ^ { \prime 2 } / \sqrt { \lambda _ { \alpha } } , } & { \alpha \in A , } \\ { \sqrt { \lambda _ { \alpha } } , } & { \alpha \notin A . } \end{array} \right.\tag{S80}
$$

The adjoint in Eq. (S51) similarly gives

$$
\begin{array} { l } { { ( \mathcal { R } _ { A } ^ { \dagger } \mathcal { X } _ { \alpha } ) ( { \pmb x } ) = \displaystyle \sum _ { i } e _ { \alpha i } \int d z _ { A } \bar { q } _ { A } ( z _ { A } | { \pmb x } ) r _ { i } ( { \pmb x } , z _ { A } ) } } \\ { ~ = \left. \begin{array} { l l } { { ( \sigma ^ { \prime 2 } / \lambda _ { \alpha } ) y _ { \alpha } ( { \pmb x } ) , } } & { { \alpha \in A } , } \\ { { y _ { \alpha } ( { \pmb x } ) , } } & { { \alpha \not \in A } } \end{array} \right. }  \\ { { ~ = \sqrt { \Lambda _ { \alpha } ^ { ( A ) } } \psi _ { \alpha } ( { \pmb x } ) . } } \end{array}\tag{S81}
$$

(S82)

Thus these modes form singular pairs of $\mathcal { R } _ { A }$ itself, and ${ \mathcal C } _ { A } \psi _ { \alpha } = \Lambda _ { \alpha } ^ { ( A ) } \psi _ { \alpha }$ with

$$
\Lambda _ { \alpha } ^ { ( A ) } = \left\{ \begin{array} { l l } { \sigma ^ { \prime 4 } / \lambda _ { \alpha } , } & { \alpha \in { \cal A } , } \\ { \lambda _ { \alpha } , } & { \alpha \notin { \cal A } . } \end{array} \right.\tag{S83}
$$

Here α indexes the linear modes in Eq. (S78); this is the residual spectrum relevant to linear fluctuations, not a claim that these modes exhaust the unrestricted functional spectrum.

In the collapsed sector, linear decoder and encoder-mean fluctuations can now be expanded using the same singular pairs:

$$
\delta f _ { i } ^ { \perp } ( z ) = \sum _ { c \in C } \sum _ { \alpha } z _ { c } \delta f _ { \alpha c } ^ { ( 1 ) } \chi _ { \alpha i } ,\tag{S84}
$$

$$
\delta \mu _ { c } ( { \pmb x } ) = \sum _ { \alpha } \delta \mu _ { \alpha c } \psi _ { \alpha } ( { \pmb x } ) .\tag{S85}
$$

The coeficients $\delta f _ { \alpha c } ^ { ( 1 ) }$ and $\delta \mu _ { \alpha c }$ are constants. A constant encoder-mean fluctuation decouples because $\mathcal { R } _ { A } 1 = 0$ and has squared mass one. Since a linear decoder has no second Hermite mode, $\delta f _ { i c } ^ { ( 2 ) } = \delta f _ { i } ^ { \mathrm { r e s t } } = 0$ . The posterior-width fluctuations therefore decouple and contribute $2 \textstyle \sum _ { c \in C } \| \delta \sigma _ { c } \| _ { x } ^ { 2 }$ , even if $\delta \sigma _ { c }$ is allowed to be an arbitrary function of x. Equation (S52) reduces to

$$
\begin{array} { l } { { \displaystyle { \cal Q } _ { \cal C } ^ { \mathrm { l i n } } = \sum _ { c \in { \cal C } } \sum _ { \alpha } \left[ \frac { ( \delta f _ { \alpha c } ^ { ( 1 ) } - \sqrt { \Lambda _ { \alpha } ^ { ( A ) } } \delta \mu _ { \alpha c } ) ^ { 2 } } { \sigma ^ { \prime 2 } } + \left( 1 - \frac { \Lambda _ { \alpha } ^ { ( A ) } } { \sigma ^ { \prime 2 } } \right) ( \delta \mu _ { \alpha c } ) ^ { 2 } \right] } } \\ { { \displaystyle ~ + 2 \sum _ { c \in { \cal C } } \| \delta \sigma _ { c } \| _ { x } ^ { 2 } } . } \end{array}\tag{S86}
$$

The mean-sector matrices are $M _ { \mu , \alpha } ,$ representing the restrictions of $K _ { C , \mu }$ in Eq. (S62), with squared masses given by Eq. (S64) and residual eigenvalues given by Eq. (S83). For a nonempty collapsed sector, nonnegative curvature under these linear fluctuations is equivalent to $\sigma ^ { \prime 2 } \geq \operatorname* { m a x } _ { \alpha } \Lambda _ { \alpha } ^ { ( A ) }$ . Since $\lambda _ { a } > \sigma ^ { \prime 2 }$ on the active stationary branch, represented directions satisfy $\sigma ^ { \prime 4 } / \lambda _ { a } < \bar { \sigma ^ { \prime 2 } }$ . The stability criterion therefore reduces to

$$
\sigma ^ { \prime 2 } \geq \operatorname* { m a x } _ { \alpha \notin A } \lambda _ { \alpha } ,\tag{S87}
$$

where the maximum is taken to be zero if no unrepresented direction remains. This recovers the principal component thresholds of the linear Gaussian VAE discussed in the main text and in Appendix D.2 of Ref. [25]. For $A = \emptyset$ , it reduces to the full-collapse threshold $\lambda _ { \operatorname* { m a x } } ( \pmb { \Sigma } _ { X } )$ .

## Simulation details and additional results

(Codes are available at here)

## Data construction

The experiment uses an equally weighted, covariance-matched mixture of two Gaussian distributions in $N = 8$ dimensions. Let $G \sim { \mathcal { N } } ( 0 , I _ { 8 } )$ and let $\eta \in \{ - 1 , + 1 \}$ take either value with probability $1 / 2$ , independently of G. Define

$$
v = { \frac { 1 } { \sqrt { 8 } } } ( 1 , \ldots , 1 ) ^ { \mathsf { T } } , \qquad a = 0 . 5 ,\tag{S88}
$$

$$
\begin{array} { r } { \pmb { Y } = ( I _ { 8 } - a ^ { 2 } \pmb { v } \pmb { v } ^ { \top } ) ^ { 1 / 2 } \pmb { G } + a \eta \pmb { v } , } \end{array}\tag{S89}
$$

$$
{ \pmb X } = O D ^ { 1 / 2 } { \pmb Y } , \qquad D = \mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { 8 } ) ,\tag{S90}
$$

$$
\lambda _ { i } = \exp [ - 0 . 1 ( i - 1 ) ] , \qquad i = 1 , \ldots , 8 .\tag{S91}
$$

Here O is a fixed orthogonal matrix obtained by QR decomposition of an $8 \times 8$ matrix of independent standard Gaussian entries. Conditioned on $\eta ,$ the two components have means $\pm a O D ^ { 1 / 2 } v$ and common covariance ${ \cal O D } ^ { 1 / 2 } ( I _ { 8 } -$ $a ^ { 2 } { \pmb v } { \pmb v } ^ { \top } ) D ^ { 1 / 2 } O ^ { \top }$ . The mixture therefore satisfies

$$
\mathbb { E } [ X ] = 0 , \quad \quad \mathrm { C o v } ( X ) = O D O ^ { \mathsf { T } } .\tag{S92}
$$

This construction changes higher-order statistics while preserving the population covariance spectrum. In particular, $U = { \pmb v } ^ { \mathsf { T } } { \pmb Y } = \sqrt { 1 - a ^ { 2 } } G _ { \parallel } + a \eta$ has unit variance and excess kurtosis $- 2 a ^ { 4 } = - 0 . 1 2 5 . \mathrm { ~ A t ~ } a = 0$ , the distribution reduces to the Gaussian covariance-matched reference.

We generate 256000 samples using data seed 1234 and subtract their sample mean. The covariance is left at its sampled value, so its empirical eigenvalues difer slightly from the prescribed population values $\lambda _ { i } .$ The same centered dataset is used throughout training and spectral estimation.

## Network and continuation protocol

The latent dimension is $n = 4 \cdot$ . The encoder has hidden widths 256, 128 with ReLU activations and separate linear output heads for $\pmb { \mu }$ and log $\sigma ^ { 2 }$ . The log variances are clamped to [−12, 8]. The decoder has hidden widths 128, 256 with ReLU activations and an 8-dimensional linear output. The prior is standard Gaussian and the isotropic decoder variance $\sigma ^ { \prime 2 }$ is fixed at each continuation point. Training minimizes the empirical negative ELBO in Eq. (S4), using one reparameterized latent sample per input and the analytical Gaussian KL term.

The variance increases from 0.65 to 1.05 in increments of 0.001, giving 401 checkpoints. With training seed 0, the first point is trained for 120 epochs; each subsequent point is initialized from the previous network weights and trained for 30 epochs.

At each checkpoint, we evaluate the per-coordinate mean KL over the full dataset,

$$
\overline { { \mathrm { K L } } } _ { j } = \frac { 1 } { 2 } \left. \mu _ { j } ^ { 2 } + \sigma _ { j } ^ { 2 } - 1 - \log \sigma _ { j } ^ { 2 } \right. _ { \pmb { x } } .\tag{S93}
$$

A coordinate belongs to A when $\overline { { \mathrm { K L } } } _ { j } > 1 0 ^ { - 3 }$ ; the remaining coordinates form C. This finite-threshold classification provides an operational approximation to the exact collapsed background used in the theory. The numerical results here use one training seed; repeated spectral measurements below quantify sampling variability at fixed network weights.

## Measurement of the residual spectral upper edge

Spectral estimation is performed after training, with every checkpoint frozen. We measure checkpoints with $C \neq \varnothing$ including complete collapse. In the following, ${ \bar { q } } _ { A }$ denotes the active encoder extracted from the frozen network.

Sampling and decoder projection. For one estimator repeat, draw B data points $\{ \pmb { x } _ { b } \} _ { b = 1 } ^ { B }$ uniformly without replace ment from the training set. Draw L active latent samples by first selecting a data index uniformly, with replacement, and then sampling its active posterior. Thus the sampling density is

$$
\widehat { \rho } _ { A } ( \boldsymbol { z } _ { A } ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \bar { q } _ { A } ( \boldsymbol { z } _ { A } \mid \mathbf { x } _ { b } ) .\tag{S94}
$$

For each sampled $z _ { A , \ell } ,$ approximate the projected decoder by

$$
\widehat { \pmb { f } } ^ { A } ( \mathsf { z } _ { A , \ell } ) = \frac { 1 } { K _ { \mathrm { M C } } } \sum _ { k = 1 } ^ { K _ { \mathrm { M C } } } \pmb { f } _ { \theta } ( \mathsf { z } _ { A , \ell } , \pmb { \xi } _ { C , k } ) , \quad \pmb { \xi } _ { C , k } \sim \mathcal { N } ( 0 , I _ { | C | } ) , \quad K _ { \mathrm { M C } } = 1 2 8 .\tag{S95}
$$

Within a repeat, the same collapsed-coordinate draws are used at every active latent sample. Decoder outputs are averaged before constructing residuals. For $A = \varnothing$ , a single projected decoder value is shared by all quadrature rows. Projection approximates $P _ { A } f _ { \theta } ;$ ; its identification with the conditional mean relies on the stationary-background assumption.

Discrete residual operator. Normalize posterior weights over the sampled data and define

$$
w _ { \ell b } = \frac { \bar { q } _ { A } ( z _ { A , \ell } \mid \pmb { x } _ { b } ) } { \sum _ { b ^ { \prime } = 1 } ^ { B } \bar { q } _ { A } ( z _ { A , \ell } \mid \pmb { x } _ { b ^ { \prime } } ) } ,\tag{S96}
$$

$$
\boldsymbol { r } _ { \ell i b } = x _ { b i } - \widehat { f } _ { i } ^ { A } ( z _ { A , \ell } ) , \qquad ( T _ { \ell } ) _ { i b } = w _ { \ell b } \boldsymbol { r } _ { \ell i b } .\tag{S97}
$$

The weights are evaluated by a softmax of log posterior densities. For a vector of sampled function values $\textbf { \em h } =$ $( h ( \pmb { x } _ { 1 } ) , \dots , h ( \pmb { x } _ { B } ) ) ^ { \top }$ , T<sub>ℓ</sub>h approximates $( \mathcal { R } _ { A } h ) ( z _ { A , \ell } )$ . Using data norm $B ^ { - 1 } h ^ { \top } h$ and quadrature weight $L ^ { - 1 }$ gives the empirical Gram operator

$$
\widehat { \mathcal { C } } _ { A ; B , L } = \frac { B } { L } \sum _ { \ell = 1 } ^ { L } T _ { \ell } ^ { \mathsf { T } } T _ { \ell } .\tag{S98}
$$

Power iteration is restricted to centered vectors with the projector $P _ { 0 } = I _ { B } - { \bf 1 1 } ^ { \mathsf { T } } / B$ . Starting from a centered, normalized Gaussian random vector, each step applies the Gram operator, subtracts the mean, and normalizes the resulting vector. The estimate is the Rayleigh quotient $\widehat { \Lambda } ( B , L ) \doteq h ^ { \mathsf { T } } \widehat { \mathcal { C } } _ { A ; B , L } h$ for $h ^ { \top } h = 1$ . Iteration stops after at most 100 steps, or when successive quotients difer by at mos $1 0 ^ { - 6 } \operatorname* { m a x } ( 1 , | \widehat { \Lambda } _ { \mathrm { p r e v i o u s } } | )$ . Centering implements the theoretical identity $\mathcal { R } _ { A } 1 = 0$ . The code also records the empirical conditional residual mean $\sum _ { b } w _ { \ell b } r _ { \ell b }$ as a diagnostic of stationarity and sampling error.

Sampling-bias extrapolation and uncertainty. At each checkpoint, we use the nine combinations

$$
B \in \{ 1 0 2 4 , 2 0 4 8 , 4 0 9 6 \} , \qquad L \in \{ 2 5 6 , 5 1 2 , 1 0 2 4 \} ,\tag{S99}
$$

with 16 independently seeded estimator repeats per combination. We fit the cell means to the empirical extrapolation model

$$
\widehat { \Lambda } ( B , L ) = \Lambda _ { \infty } + \frac { b } { B } + \frac { c } { L } ,\tag{S100}
$$

using inverse squared standard errors of the cell means as weights. The fitted intercept defines the reported $\hat { \Lambda } _ { * } ^ { ( A ) }$

(a)  
![](images/2c26c2cf397df88aa870e43392d57ecd63f3b654f82eff724c31a0dfcbc95e28.jpg)

(b)  
![](images/5ca598ca3addef5fe03be0fb0f848cac3bbbec25b26c1d95d3c1f5c160d8f4f8.jpg)  
FIG. S1. Additional results for the covariance-matched Gaussian mixture with $a = 0 . 5$ and training seed 0. (a) Number of active latent coordinates |A| versus decoder variance $\sigma ^ { \prime 2 }$ . (b) Squared mass $m _ { \mu , * , - } ^ { 2 }$ obtained from the estimated leading residual eigenvalue using Eq. (S64). Dots show the checkpoint estimates; solid curves are segmented Gaussian local-linear smooth guides. Vertical dotted lines mark the displayed changes in $| A | ,$ and the horizontal dashed line marks zero mass. Panel (b) is evaluated where the collapsed sector is nonempty.

![](images/8a997f2436584701f1b1e3569844a157bc34bf9d59c357047b41de83dfbbef0f.jpg)

![](images/2251544f9d6d0c1ce3e2ea1331aaad4ddec35c5e295ab9c47714d63ac2bfd3dd.jpg)

![](images/f946a9c378d0e77a8a526aa1dbd31519370d64ba7cc4d8cbd58705cc8ce2c1de.jpg)

![](images/b6139bda264ee96c912b7045ca89c1ecaddeceed40746aaffe7ced2f51ddfb7c.jpg)  
FIG. S2. Direct free-energy curvature checks for fixed projected backgrounds with (a) $| A | = 3 ,$ (b) $| A | = 2 $ , (c) $| { \cal A } | = 1$ , and (d) $| { \cal A } | = 0$ . Colors and marker shapes distinguish the checkpoint variances $\sigma _ { 0 } ^ { \prime 2 }$ given in each legend. Points show $\kappa ( 0 . 0 1 )$ from Eq. (S102); error bars denote one standard error over four measurement repeats. Dashed curves show the predicted lower mass evaluated with $\lambda _ { \mathrm { { m o d e } } }$ . Dotted lines mark zero curvature and the predicted marginal ratio $\sigma ^ { \prime 2 } / \lambda _ { \mathrm { m o d e } } = 1$ . Measured points are displayed without smoothing or a zero constraint. Asterisks identify backgrounds containing repeats that reached the iteration limit before satisfying the eigenvector-residual tolerance; these repeats are retained. The scans vary the decoder variance at fixed background fields.

## Additional results

Figure S1 shows the mean-channel squared mass associated with the estimated residual spectral upper edge along the continuation trajectory. We evaluate $m _ { \mu , * , - } ^ { 2 }$ from $\operatorname { E q . }$ (S64) by substituting $\hat { \Lambda } _ { * } ^ { ( A ) }$ for $\Lambda _ { \alpha } ^ { ( A ) }$ . This quantity expresses the measured residual spectrum in the field-space mass normalization used in the main text. The estimates approach zero near successive decreases of $| A |$ and increase within the intervening branches. The scatter displays the sampling variability of the spectral estimates near marginality.

We additionally test the mean-channel fluctuation prediction by evaluating free-energy changes along explicitly constructed coupled modes. For the mixture with $a = 0 . 5$ and training seed 0, we select backgrounds at $\sigma _ { 0 } ^ { \prime 2 } = 0 . 7 5 0$ $( | A | = 3 ) , 0 . 8 4 0 \ ( | A | = 2 ) , 0 . 9 3 0 \ ( | A | = 1 )$ , and $1 . 0 2 0 \ ( | A | = 0 )$ . The active encoder is frozen, the collapsed posteriors are set to standard normals, and the decoder is projected as in Eq. (S95). These operations define approximate stationary backgrounds; the conditional residual mean provides a diagnostic of their stationarity and finite-sampling error.

Each measurement repeat uses $B = 4 0 9 6$ data points, $L = 1 0 2 4$ active-latent samples, and $K _ { \mathrm { M C } } = 1 2 8$ collapsed coordinate samples. We retain the leading centered empirical eigenfunction $\psi _ { : }$ normalized by $\begin{array} { r } { B ^ { - 1 } \sum _ { b } \psi ( \pmb { x } _ { b } ) ^ { 2 } = 1 } \end{array}$ and its eigenvalue $\lambda _ { \mathrm { { m o d e } } } .$ , and define the decoder partner $\chi = \widehat { \mathcal { R } } _ { A } \psi / \sqrt { \lambda _ { \mathrm { m o d e } } }$ . Here $\lambda _ { \mathrm { m o d e } }$ is the finite-sample spectral estimate paired with the constructed mode, whereas $\hat { \Lambda } _ { * } ^ { ( A ) }$ in the continuation analysis denotes the extrapolated estimate. Power iteration uses two initial vectors and at most 300 steps per initialization, with relative eigenvectorresidual tolerance $1 0 ^ { - 6 }$ ; the mode with the larger Rayleigh quotient is retained.

At each probe variance $\sigma ^ { \prime 2 }$ , let $( u , v )$ be the unit lower-eigenvalue vector of the mean-channel block in $\operatorname { E q } .$ (S62), with $\Lambda _ { \alpha } ^ { ( A ) }$ replaced by $\lambda _ { \mathrm { m o d e } }$ . For one collapsed coordinate $^ { c , }$ we impose

$$
\mu _ { c } ( { \pmb x } ; t ) = t v \psi ( { \pmb x } ) , \qquad { \pmb f } ( { \pmb z } ; t ) = \widehat { { \pmb f } } ^ { A } ( { \pmb z } _ { A } ) + t u { z } _ { c } \chi ( { \pmb z } _ { A } ) , \qquad \sigma _ { c } = 1 .\tag{S101}
$$

All remaining encoder fields are held fixed. An independent set of 16384 joint samples $( { \pmb x } , z _ { A } )$ from the same empirical data distribution evaluates the negative ELBO, using common samples for the perturbed and unperturbed fields. Integration over $z _ { c }$ uses its exact Gaussian first and second moments. The measured curvature is

$$
\kappa ( t ) = \frac { \mathcal { F } ( t ) + \mathcal { F } ( - t ) - 2 \mathcal { F } ( 0 ) } { t ^ { 2 } } .\tag{S102}
$$

The calculation uses double precision and amplitudes $t = 0 . 0 1 , 0 . 0 2 , 0 . 0 4 , 0 . 0 8$ . Figure S2 shows the raw diferences at $t = 0 . 0 1$ , including their finite-amplitude correction of order $t ^ { 2 }$ , for 21 probe ratios $\sigma ^ { \prime 2 } / \lambda _ { \mathrm { m o d e } } = 0 . 9 0 , 0 . 9 1 , \dots , 1 . 1 0$ Four independently seeded measurement repeats determine means and standard errors; probe points within a repeat share samples.

The direct curvatures change sign near the predicted spectral edge for each active dimension, with small ofsets near marginality. This comparison tests the constructed mean-channel direction on a fixed projected background. The background remains fixed as $\sigma ^ { \prime 2 }$ varies, so the scan probes local curvature rather than tracking a sequence of stationary solutions. Finite data and quadrature budgets and approximate background stationarity limit the comparison.