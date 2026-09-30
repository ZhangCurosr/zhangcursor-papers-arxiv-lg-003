# Scene-Consistent Illumination Transfer for Inserted Advertising Graphics

Rameshwar Mishra<sup>∗</sup>, Bishshoy Das<sup>†</sup>, A. V. Subramanyam<sup>∗</sup>, Guan-Ming Su<sup>†</sup>

<sup>∗</sup>Indraprastha Institute of Information Technology, Delhi, India

{rameshwarm, subramanyam}@iiitd.ac.in

<sup>†</sup>Dolby Laboratories, Bangalore, India and Sunnyvale, USA

{bishshoy.das, guanming.su}@dolby.com

Abstract—Replacing a visible advertisement in a broadcast frame is geometrically straightforward but photometrically delicate. A pasted graphic can have the correct perspective and still appear detached when its brightness, shading, or shadow disagrees with the surface beneath it. This paper presents Ad-Relight, an inference-only procedure for transferring scene illumination to a supplied advertising graphic without collecting a banner-specific training set. The procedure first separates slowly varying shade from graphic structure, then probes a pretrained diffusion relighter with two nearly identical backgrounds to isolate the contribution of the target region. A final pass combines this residual with a smoothed luminance field and a soft attenuation mask. Across 560 generated placements, the approach improves structural similarity, perceptual distance, and illumination agreement over geometric compositing and direct relighting baselines. Human judgments and an automated preference study show the clearest gains on floor-mounted graphics with nonuniform lighting. The current study is image based; temporal stabilization remains an open extension.

Index Terms—advertising graphics, image harmonization, illumination transfer, diffusion prior, scene compositing

## I. TASK AND CONTRIBUTIONS

Personalized advertising often begins with a finished graphic and a location in an existing frame. The production system must preserve the graphic’s identity while making it look as though it was present when the scene was captured. This requirement is especially visible on courts, floors, walls, and other broad surfaces: the replacement area may contain a brightness ramp, a soft occlusion, or a color cast that cannot be reproduced by a homography alone. Figure 1 illustrates the resulting gap between geometric placement and scene-aware appearance.

Let I denote a frame, M a mask for the region to be replaced, and L a user-supplied banner. The desired output should satisfy three constraints: its boundary must follow M, the semantic content of L must remain legible, and its low-frequency appearance should agree with the illumination already present in I. These constraints pull in different directions. Strong generative editing can improve realism while changing a logo, whereas a literal paste preserves the logo while exposing the mismatch in lighting.

Our design treats a pretrained relighting network as a source of illumination evidence rather than as a direct banner generator. A pair of controlled queries reveals how the network responds to the original target region; that response is then reused while the supplied graphic remains explicit throughout the pipeline. This separation makes the method usable at inference time and avoids fitting a new relighting model to a small, specialized advertising collection.

The paper makes three contributions:

• a formulation of banner replacement as regionconditioned illumination transfer, with identity preservation treated as a first-class constraint;

• a training-free procedure that combines shade normalization, differential probing, and soft shadow attenuation under the name Ad-Relight;

• a focused evaluation covering 560 placements, component ablations, automated rankings, and human pairwise preferences.

## II. POSITIONING

Classical compositing methods modify image gradients or local appearance to hide a seam. Poisson editing is a representative example: it provides a principled way to blend boundaries, but it does not by itself infer the illumination that should fall across a new planar object [2]. Learned harmonization systems use broader image context to align a foreground with its background [3]; they are effective when the training distribution covers the desired composite, yet a printed floor banner has a distinctive combination of planar perspective, texture, and spatial light variation.

Diffusion models provide a useful alternative because their denoising process contains broad visual priors [1]. Recent illumination-editing work imposes light-transport consistency during training and exposes a strong general-purpose relighting backbone [4]. Directly applying such a backbone to a pasted banner is not sufficient: the network may interpret a horizontal graphic as part of the floor, or may alter the lettering while attempting to improve realism. Ad-Relight therefore uses the network twice for diagnosis and once for synthesis. The central distinction is not a new generative model, but a test-time construction that extracts a local lighting signal before asking the model to edit the banner.

## III. AD-RELIGHT

Figure 2 summarizes the pipeline. All operations are performed independently for one frame. The relighting backbone

![](images/ae846eeaedddd0aaeab2560e53fa4ef18b461c2625b6a8646935a1d98d5b7541.jpg)  
Fig. 1. A supplied graphic can be geometrically correct yet visually detached from its host surface. The examples motivate matching spatial brightness changes, rather than treating the replacement as a uniformly lit texture.

is frozen, and no banner examples are used to update its parameters.

## A. Region-conditioned preparation

We crop the target support using M and denote its original appearance by O. The replacement graphic is first placed in the same planar coordinate system; this geometric step can be obtained from the host application’s mask or a homography. To avoid a perfectly synthetic surface, a mild texture layer T is mixed into the graphic L:

$$
L _ { \mathrm { t e x } } = ( 1 - \alpha ) L + \alpha T , \qquad 0 \leq \alpha \leq 1 .\tag{1}
$$

The texture weight is fixed at $\alpha = 0 . 3$ in our experiments. It is deliberately small: texture should support contact with the scene without competing with the supplied branding.

The main preparation step transfers only slowly varying shade. Let $Y _ { O }$ and $Y _ { L }$ be the luminance channels of O and $L _ { \mathrm { t e x } } .$ , respectively, and let $\mathcal { G } _ { K }$ denote a Gaussian low-pass operator. We form a smooth component and a normalized structural component for each image:

$$
\begin{array} { l l } { { S _ { O } = \mathcal { G } _ { K } ( Y _ { O } ) , } } & { { R _ { O } = \displaystyle \frac { Y _ { O } } { S _ { O } + \delta } , } } \\ { { S _ { L } = \mathcal { G } _ { K } ( Y _ { L } ) , } } & { { R _ { L } = \displaystyle \frac { Y _ { L } } { S _ { L } + \delta } . } } \end{array}\tag{2}
$$

Here δ prevents unstable division in dark pixels. The banner’s luminance is replaced by $Y _ { L } ^ { \star } \ = \ S _ { O } R _ { L }$ . Thus, the graphic retains its internal contrast while inheriting the broad shade pattern measured on the host surface. This operation is intentionally conservative: it does not ask a generator to redraw letters, edges, or colors.

## B. Differential illumination probe

The prepared banner is not yet a reliable conditioning image for a relighter. We therefore query the frozen backbone with two versions of the scene. In the first query, $B _ { O }$ contains the complete frame and the original target region. In the second, $B _ { M }$ has the target region suppressed while the rest of the frame is unchanged. Let $F _ { \phi }$ be the relighting network and let $Q _ { O }$ and $Q _ { M }$ be its corresponding responses. Under the local linear behavior encouraged by consistent-light training, the two responses can be viewed as

$$
\begin{array} { r } { Q _ { O } \approx T _ { \phi } L _ { \phi } , \qquad Q _ { M } \approx T _ { \phi } ( L _ { \phi } - L _ { O } ) , } \end{array}\tag{3}
$$

where $L _ { O }$ is the illumination contribution associated with the target region. Their difference gives a spatially registered residual:

$$
\epsilon = Q _ { O } - Q _ { M } \approx T _ { \phi } L _ { O } .\tag{4}
$$

The residual is not treated as a physically calibrated light map. It is a model-space measurement that retains the direction and relative strength of the target area’s contribution. Subtraction also suppresses scene content shared by both queries, which is useful when the frame contains spectators, court markings, or other high-contrast details.

## C. Guided relighting and contact attenuation

The residual can contain small traces of appearance leakage. We stabilize it with a second guide obtained from the target luminance:

$$
G = { \mathcal G } _ { K ^ { \prime } } ( Y _ { O } ) , \qquad B _ { \epsilon } = \alpha _ { \epsilon } G + ( 1 - \alpha _ { \epsilon } ) \epsilon ,\tag{5}
$$

where $K ^ { \prime } < K$ . The final backbone call uses $B _ { \epsilon }$ as its lighting condition and $L ^ { \star }$ as the image to be edited:

$$
{ \cal O } _ { L } = F _ { \phi } ( B _ { \epsilon } , L ^ { \star } ) .\tag{6}
$$

The smaller blur radius preserves broad gradients while attenuating unstable, high-frequency differences.

Finally, we estimate contact darkening from the smoothed target luminance. Otsu thresholding supplies τ , and the continuous attenuation field is

$$
d ( x , y ) = \operatorname* { m i n } \left( 1 , \frac { Y _ { O } ( x , y ) } { \tau } \right) .\tag{7}
$$

The luminance of the relit banner is mixed with its attenuated version using $\alpha _ { s }$

$$
Y _ { \mathrm { f i n a l } } = \alpha _ { s } Y _ { O _ { L } } + ( 1 - \alpha _ { s } ) Y _ { O _ { L } } d .\tag{8}
$$

This soft weighting avoids the hard contour that a binary shadow mask would produce. We use $K = 9 9 , K ^ { \prime } = 2 1$ $\alpha _ { \epsilon } = 0 . 4$ , and $\alpha _ { s } = 0 . 2$

## IV. EMPIRICAL STUDY

## A. Protocol and measurements

We evaluate the frame-level problem on 80 source frames and seven replacement graphics, producing 560 source–banner cases. The collection varies camera elevation, surface material, banner color, and the strength of the illumination gradient. The difficult subset contains large horizontal placements where the host floor is glossy or unevenly lit. Each method receives the same planar placement and the same source graphic.

We compare against four alternatives: a perspective warp with no appearance correction, a training-free image compositor, direct use of the relighting backbone, and a shadowconditioned relighting pipeline. The last three are intended to separate the value of generic diffusion composition, an unmodified relighter, and explicit shadow guidance. Scores are computed inside the replacement mask. SSIM measures structural agreement, LPIPS measures learned perceptual distance, and ILL-SIM is the cosine similarity between the host and output luminance fields.

![](images/7a9400808af4738d54b9b580c125d9450e1e1fc37869ed9d977d52678380fa70.jpg)  
Fig. 2. Processing graph for Ad-Relight. A low-frequency shade field is transferred to the supplied graphic, two backbone queries provide a target-region residual, and a mixture of that residual with a smoothed luminance guide controls the final relighting pass.

TABLE I  
FRAME-LEVEL COMPARISON ON 560 PLACEMENTS. HIGHER IS PREFERRED FOR SSIM AND ILL-SIM; LOWER IS PREFERRED FOR LPIPS.
<table><tr><td>Method</td><td>SSIM ↑</td><td>LPIPS↓</td><td>ILL-SIM ↑</td></tr><tr><td>Warp-only</td><td>0.89</td><td>0.12</td><td>0.82</td></tr><tr><td>Training-free compositor</td><td>0.45</td><td>0.57</td><td>0.62</td></tr><tr><td>Direct relighting</td><td>0.91</td><td>0.07</td><td>0.77</td></tr><tr><td>Shadow-guided relighting</td><td>0.92</td><td>0.09</td><td>0.79</td></tr><tr><td>Ad-Relight</td><td>0.95</td><td>0.03</td><td>0.92</td></tr></table>

## B. Appearance comparison

The aggregate scores in Table I show that preserving the supplied graphic and matching the host luminance are compatible objectives. The warp-only system keeps the banner recognizable but leaves it visually flat. Direct relighting improves global appearance in ordinary cases, yet it can absorb the banner into the floor texture when the target is horizontal. The differential signal gives Ad-Relight a more localized condition, which explains its improvement in ILL-SIM as well as its lower LPIPS. Figure 3 shows representative sports frames with different camera heights and floor materials.

## C. Component sensitivity

We test two nearby parameter settings and three targeted removals. M1 and M2 perturb the texture blend, blur radii, residual mixture, and shadow weight together. M3 removes the low-frequency guide G, M4 omits the shade-transfer stage, and M5 removes the differential residual ϵ. The numbers in Table II indicate that the residual is particularly important for perceptual similarity, while shade preparation has a strong effect on structural and illumination scores. The full configuration remains the most balanced choice.

TABLE II  
ABLATION ON THE SAME BENCHMARK.
<table><tr><td>Variant</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>ILL-SIM ↑</td></tr><tr><td>M1</td><td>0.92</td><td>0.03</td><td>0.85</td></tr><tr><td>M2</td><td>0.91</td><td>0.05</td><td>0.87</td></tr><tr><td>M3 (−G)</td><td>0.89</td><td>0.07</td><td>0.83</td></tr><tr><td>M4 (—shade)</td><td>0.86</td><td>0.05</td><td>0.80</td></tr><tr><td>M5 (−€)</td><td>0.80</td><td>0.10</td><td>0.85</td></tr><tr><td>Full</td><td>0.95</td><td>0.03</td><td>0.92</td></tr></table>

TABLE III  
PERCENTAGE OF GPT-4O PAIRWISE CHOICES FAVORING AD-RELIGHT.
<table><tr><td>Criterion</td><td>Warp</td><td>Direct</td><td>Shadow-guided</td></tr><tr><td>Gradient fidelity</td><td>100.0</td><td>97.6</td><td>95.2</td></tr><tr><td>Lighting agreement</td><td>97.6</td><td>95.2</td><td>92.8</td></tr><tr><td>Scene realism</td><td>90.4</td><td>92.8</td><td>85.7</td></tr></table>

## D. Preference studies

We use two complementary preference tests. First, GPT-4o is asked to choose between unlabeled outputs using three criteria: gradient fidelity, lighting agreement, and overall realism. The percentage of cases favoring Ad-Relight is shown in Table III. The model favors the proposed output in every comparison, with the largest margins for the geometric baseline.

Second, 25 participants completed 324 two-choice judgments across four questionnaire variants. The options were unlabeled and the task emphasized the same three properties. Figure 5 summarizes the preference split for the direct and warp-only comparisons. Human responses favor the proposed output most consistently when a strong gradient is present; this aligns with the quantitative ILL-SIM improvement rather than merely reflecting a preference for a particular logo color.

## E. Scope and failure modes

The evidence is limited to independent frames. The approach assumes that the backbone’s response changes smoothly enough for subtraction to reveal a useful regional cue; scenes with strong reflections, moving shadows, or severe occlusion can violate that assumption. The pipeline also inherits the mask and homography supplied by the placement system, so an inaccurate target boundary is not corrected by relighting. These limitations suggest that a future video version should estimate a temporally filtered residual and jointly refine geometry and illumination.

![](images/95cb36aad8008fa3042589b1730c74e3a95e0805d24af5d44f6cd37d3fd53f4f.jpg)  
Fig. 3. Representative replacements arranged by scene and method. Geometric pasting leaves the graphic uniformly lit, while direct relighting can borrow texture from the floor. The proposed pipeline preserves the logo and follows the host surface’s broad brightness changes.

![](images/8d2f102fff986e7490b72070df374e72e67399f4c8e1119aedc8b19cfd5ff773.jpg)  
Fig. 4. Ablation examples. Removing a stage changes either the banner’s contact with the floor or the strength of the recovered illumination gradient.

## V. PERSPECTIVE

Ad-Relight reframes banner integration as a measurement problem. Instead of asking a general-purpose generator to invent a plausible appearance for a logo, it first extracts how the host region influences a frozen relighting model and then applies that evidence to the unchanged graphic. The resulting procedure is compact, training-free, and particularly effective for planar surfaces whose lighting varies across the banner. The reported gains do not remove the need for temporal modeling or better placement masks, but they show that a carefully designed test-time probe can turn a general illumination prior into a practical tool for personalized advertising.

![](images/5fb0799cc996f140f16202498da9cab9a83b10f2dc112e6298240a3ca7a16e48.jpg)  
Fig. 5. Human preference rates for two comparisons. Green bars indicate selections of Ad-Relight.

## REFERENCES

[1] J. Ho, A. N. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems, vol. 33, 2020.

[2] P. Perez, M. Gangnet, and A. Blake, “Poisson image editing,” ACM Transactions on Graphics, vol. 22, no. 3, pp. 313–318, 2003.

[3] Y.-H. Tsai, X. Shen, Z. Lin, K. Sunkavalli, X. Lu, and M.-H. Yang, “Deep image harmonization,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 3789–3797, 2017.

[4] L. Zhang, A. Rao, and M. Agrawala, “Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport,” in International Conference on Learning Representations, 2025.