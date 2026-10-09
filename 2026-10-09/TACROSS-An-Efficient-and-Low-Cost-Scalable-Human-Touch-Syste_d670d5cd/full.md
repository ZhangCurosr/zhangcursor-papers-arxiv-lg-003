![](images/ef4204c2c512ab3a39ac7cca0c6adef215f10e0ea990da2472536333b637ce59.jpg)  
Fig. 1. TACROSS overview. (a) Synchronized human demonstrations with egocentric RGB, hand pose, and tactile observations. (b) Contact canonicalization and sensor-specific adaptation align piezoresistive glove and capacitive Revo2 observations. Shared temporal and inter-finger encoding produces tactile latents for robot-supervised ACT learning. Training uses masked contact-semantic targets and contrastive alignment, while deployment uses robot observations only. (c) Vision-IK Ego achieves similar success (91.5% vs. 92.2%) at 3.5× the average collection rate and 95.7% lower equipment cost than Robot-only. (d) Four contact-rich manipulation tasks for closed-loop evaluation.

# TACROSS: An Efficient and Low-Cost Scalable Human Touch System Across Heterogeneous Tactile Sensors for Dexterous Robot Learning

Bo Chen<sup>1,6,∗</sup>, Huanzhang Hu<sup>2,6,∗</sup>, Junyang Ma<sup>5</sup>, Bo Yue<sup>2</sup>, Fangdi Yu<sup>3</sup>, Haijier Chen<sup>4,6</sup>, Xianxin Lai<sup>1,6</sup>, Shuyu Pan<sup>6</sup>, Zhen Yang<sup>1,6</sup>, Xiaoquan Sun<sup>1,6</sup>, Wenze Cui<sup>1</sup>, Zhongliang Jiang<sup>1</sup>, Shaopeng Liu<sup>1,†</sup>, and Jiayu Chen<sup>1,6,†</sup> <sup>1</sup>The University of Hong Kong <sup>2</sup>The Chinese University of Hong Kong (Shenzhen) <sup>3</sup>Ocean University of China <sup>4</sup>Wuhan University <sup>5</sup>Wuhan Textile University <sup>6</sup>INFIFORCE <sup>∗</sup>Equal contribution; <sup>†</sup>Corresponding author

Abstract— Collecting tactile demonstrations on robots is costly and slow, motivating the use of lower-cost human tactile gloves for scalable data collection. However, human capacitive/piezoresistive gloves and robotic tactile sensors differ fundamentally in transduction principle, sensor layout, spatial resolution, and dynamic response, making alignment of raw sensor channels ill-posed. To address this problem, we present TACROSS, a scalable system for learning from human touch and transferring it to robots that bridges this heterogeneity by aligning tactile streams at the level of contact events rather than raw sensor values. The hardware component of TACROSS integrates a piezoresistive glove with five layers and a cost of \$10.86 with 285 sensing points. To align contact semantics, we design canonicalizers and residual adapters that map heterogeneous signals into a shared tactile latent with 256 dimensions via a temporal Transformer with attention across fingers. We further introduce a robot-grounded policy learning scheme in which robot demonstrations provide the sole source of ground-truth action supervision, while human demonstrations support tactile representation learning and provide confidence-weighted auxiliary supervision through valid retargeted hand targets. We evaluate our system on four contact-rich manipulation tasks. Compared to conventional teleoperation, our proposed system achieves a 3.5× efficiency

improvement while reducing demonstration acquisition equipment cost by 95.7%. We will open-source the TACROSS hardware and software system and publicly release a tactile dataset comprising over 150 hours of recordings. Project page: https://tacross-touch-project.github.io/.

## I. INTRODUCTION

Force tactile interaction technology, an advanced human– computer interaction modality that mimics the sense of physical touch, is increasingly serving as a bridge between the digital and physical worlds [1]. Contact-rich dexterous manipulation depends on events, including grasp onset, loading, stability, slip, and release, that RGB observation leaves visually ambiguous [2]. A central question is how to leverage human tactile demonstrations collected efficiently at low cost to reduce robot data collection demands while preserving effective manipulation performance and retaining robot action supervision [3]. Robot tactile demonstrations expose these events directly [4], but every collection minute occupies the target arm, hand, and sensor platform, incurring operation, reset, and calibration costs. Under our accounting, robot teleoperation records only 40 episodes/h on a USD 12,000 system. Scalable tactile gloves capture distributed human contact patterns [5]–[7], while systems that collect visual and tactile data without robots demonstrate a practical route to acquiring contact-rich manipulation data [8]. Human tactile gloves offer a decoupled, faster, and cheaper alternative: our USD 2,150 Ego-R and USD 520 Ego-V systems reach 112 and 140 episodes/h, corresponding to 2.80× and 3.50× the robot rate, together yielding 159.09 h of recorded human trajectories, 23,677 recorded demonstrations, and 21,412 valid ones (Table II).

Yet this larger human corpus is usable only if it can be related to robot touch. UniTouch and AnyTouch learn shared representations across heterogeneous vision-based tactile sensors [9], [10], motivating our study of alignment across different transduction principles. The human glove is piezoresistive, whereas Revo2 sensing is capacitive. The two differ in transduction, taxel layout, spatial support, resolution, numerical scale, noise, drift, hysteresis, and temporal response. Equal channel values therefore need not denote equal contact, and identical physical events can activate different numbers and locations of taxels. Direct matching of individual channels is thus ill-posed: sensor identity and acquisition artifacts can obscure the interaction state required for control. We term this mismatch the Sensing Gap.

EgoMimic combines human and robot demonstrations through alignment across domains and joint training [11], while DexPilot maps observed human hand motion to a dexterous robot hand–arm system [12]. Even if the sensing gap were bridged at the representation level, a second mismatch remains at the action level: human hand motion, however precisely retargeted, is not an action executed by Revo2. Kinematic and morphological differences between the human hand and the robotic end-effector mean that pseudo actions derived from human demonstrations carry retargeting error and cannot be treated as ground truth. Treating these pseudolabels as ground-truth robot actions can introduce retargeting errors into the learned policy. We term this mismatch the Action Gap. We address the Action Gap by grounding policy learning in robot demonstrations.

We present TACROSS, an integrated hardware–software system for human tactile data acquisition and cross-sensor transfer for robot learning. To solve the Sensing Gap, we propose canonicalizers with validity masks together with residual adapters and a shared causal temporal encoder with attention across fingers. The canonicalizers convert raw human and robot signals into a common masked anatomical contact field, capturing active region, contact probability, coarse force, contact area, duration, loading dynamics, phase, instability, and release, with explicit validity masks distin guishing missing measurements from measured zero contact. The residual adapters account for differences in scale, noise, drift, and hysteresis between domains. And the shared encoder represents contact event dynamics and coordination among fingers across embodiments, together yielding a shared tactile latent with 256 dimensions. To address the Action Gap, we ground policy learning in executed robot actions, which provide the sole source of ground-truth action supervision. Retargeted human hand targets serve as pseudolabels for auxiliary supervision, weighted by confidence and masked by validity.

The contributions of this work are:

• Efficient and Low-Cost Data Acquisition System. We present TACROSS, a hardware–software system with low-cost tactile acquisition, and evaluate its collection throughput and downstream robot performance.

• Tactile Alignment across Sensors and Policy Learning from Robot Demonstrations. We align heterogeneous human and robot touch in a shared contactsemantic space with validity masks and ground policy learning in executed robot actions.

• Open-Source Resources and Tactile Dataset. We will open-source the tactile glove designs, robot platform configurations, and software for data logging, processing, and model training, and release 183.95 h of recorded trajectories: 159.09 h of human data and 24.86 h of robot data.

All design files are available at the project website.

## II. RELATED WORK

## A. Human Tactile Demonstration Acquisition

STAG and Flex-Glove enable distributed human tactile capture through inexpensive sensing and personalized fabrication [5], [13]. OSMO uses the same glove on humans and robots, while DexViTac combines multimodal acquisition and kinematics-grounded representation learning with matched tactile hardware [14], [15]. ForceMimic records interaction wrenches and motion for force-centric imitation learning. TACROSS connects a piezoresistive human glove to a capacitive robot hand, addressing sensor heterogeneity with measurement validity and causal synchronization.

## B. Cross-Sensor Tactile Alignment

Domain-adversarial training uses gradient reversal to learn task-discriminative, domain-invariant features [16]. UniTac-Hand aligns heterogeneous touch through masked MANO UV maps and paired interactions with contrastive, reconstruction, and adversarial objectives [17]. TactAlign learns a rectified-flow mapping from interaction-derived pseudo pairs, supporting unpaired demonstrations and dynamic contact [18]. TACROSS represents regional contact state, relative intensity, duration, spatial descriptors, and phase. Sensorspecific adapters feed shared causal temporal and inter-finger encoders. Positive windows match task/object metadata and contact semantics without cross-domain frame synchronization. Field-level validity masks distinguish missing measurements from valid no-contact observations.

## C. Learning from Human and Robot Demonstrations

EgoMimic jointly trains policies on human and robot demonstrations [11]. FreeTacMan combines temporal tactile pretraining with ACT and evaluates optical cross-sensor generalization [8]. MimicTouch learns from multimodal human demonstrations and bridges the embodiment gap through online residual reinforcement learning [19]. TACROSS uses human touch for representation learning and robot actions for ground-truth supervision. HaMeR [20] and calibrated retargeting provide human hand pseudo-targets for confidenceweighted, validity-masked auxiliary supervision. Arm targets require separate supervision. ACT [21] takes the robot tactile latent, RGB, and proprioception, using robot observations at deployment.

## III. TACROSS SYSTEM AND METHOD

## A. Problem Formulation and Overview

TACROSS consists of a human tactile acquisition system, a contact-semantic encoder with human and robot input branches, and an ACT policy. Training uses human and robot episodes, deployment uses the robot branch and policy.

Human and robot episodes are defined as

$$
\mathcal { D } _ { H } = \{ { I } _ { t } ^ { H } , { q } _ { t } ^ { H } , { X } _ { t } ^ { H } , { \tilde { a } } _ { t } ^ { R , H } , { w } _ { t } ^ { H } , { \tau } _ { t } \} _ { t = 1 } ^ { T _ { H } } ,\tag{1}
$$

$$
\mathcal { D } _ { R } = \{ I _ { t } ^ { R } , q _ { t } ^ { R } , X _ { t } ^ { R } , a _ { t } ^ { R } , \tau _ { t } \} _ { t = 1 } ^ { T _ { R } } ,\tag{2}
$$

where $I , ~ q , ~ X$ , and $\tau$ denote RGB images, kinematics/proprioception, tactile observations, and native timestamps. The executed robot action $a _ { t } ^ { R }$ provides the groundtruth action label. The optional retargeted human hand target $\tilde { a } _ { t } ^ { R , H }$ serves as a pseudo-label for auxiliary supervision, with $\hat { w _ { t } ^ { H } }$ encoding confidence and validity.

For each domain, a canonicalizer produces regional contact fields and validity masks. An adapter for each domain converts these fields into tokens, which are processed by shared encoders for temporal dynamics and finger interactions. The complete tactile path is denoted by

$$
z _ { t } ^ { e } = f _ { e } ( X _ { 1 : t } ^ { e } , q _ { 1 : t } ^ { e } ) , \qquad e \in \{ H , R \} ,\tag{3}
$$

The robot policy combines $z _ { t } ^ { R }$ with RGB and proprioception to predict an action chunk:

$$
a _ { t : t + H } ^ { R } = \pi ( I _ { t } ^ { R } , q _ { t } ^ { R } , z _ { t } ^ { R } ) .\tag{4}
$$

Sections B–D describe acquisition, tactile encoding, and policy training, respectively.

## B. Human Tactile Acquisition

Glove Design and Readout. The glove comprises five layers, including two patterned copper–nickel electrodes, a pressure-sensitive layer, and palmar/dorsal fabric encapsulation (Fig. 2). Its $1 2 \times 1 5$ palm and five $3 \times 7$ finger arrays provide 285 physical sensing locations. Matrix multiplexing, a 12-bit ADC, and a wrist controller stream frames of 256 values over serial/BLE at approximately 150 Hz. Physical locations and streamed values are distinct, defective channels retain invalid flags. Table I reports hardware measurements.

Calibration and Causal Synchronization. For channel $i , \ x _ { i , t }$ is the raw reading, $b _ { i }$ its unloaded reference, $g _ { i }$ the calibration gain converting baseline-subtracted readings to the calibrated scale, and $m _ { i } \in \{ 0 , 1 \}$ the channel-validity flag:

$$
\widetilde { x } _ { i , t } = m _ { i } \exp ( g _ { i } ( x _ { i , t } - b _ { i } ) , 0 , x _ { \operatorname* { m a x } } ) .\tag{5}
$$

![](images/30f6af598308e5b672cffa2e976b2c94c2117f0b64d04831406c793f20b0a2ee.jpg)  
Fig. 2. TACROSS glove: sensing coverage,and five-layer structure..

Here, clip $( y , 0 , x _ { \mathrm { m a x } } ) ~ = ~ \operatorname* { m i n } \{ x _ { \mathrm { m a x } } , \operatorname* { m a x } ( 0 , y ) \}$ , where $x _ { \mathrm { m a x } } \ > \ 0$ is the upper bound on that scale. $m _ { i } ~ = ~ 0$ suppresses an invalid channel, not a valid no-contact observation. At time t, each modality selects its latest sample with timestamp $\tau \leq t$ . Episodes store RGB, pose, raw/calibrated touch, timestamps, sample ages, masks, and valid pseudo actions. Figure 3 summarizes the pipeline for acquisition and data conditioning.

TABLE I  
HARDWARE SPECIFICATIONS AND CHARACTERIZATION OF THE TACROSS GLOVE. F.S. DENOTES FULL SCALE.
<table><tr><td>Characteristic</td><td>Reported value</td></tr><tr><td>BOM cost</td><td>USD 10.86/glove (controller 6.58, materials 4.28)</td></tr><tr><td>Derived cost/location</td><td>USD 0.038 (BOM / 285 locations)</td></tr><tr><td>Layout and coverage</td><td>285 locations, ~200 cm2, &gt; 85% coverage</td></tr><tr><td>Frame and rate</td><td>256 values/frame, ~150 Hz</td></tr><tr><td>Serial timing</td><td>1.72 ms/frame (theoretical)</td></tr><tr><td>Force test</td><td>0–20 N, max 83.91/255 ADC, no saturation</td></tr><tr><td>Repeatability</td><td>1.38% F.S. (loaded)</td></tr><tr><td>Nonlinearity</td><td>8.26% F.S., R2 = 0.90</td></tr><tr><td>Hysteresis</td><td>11.65% F.S. (taxel 12, 7)</td></tr><tr><td>Drift</td><td>0.39–0.41% F.S. /  10 min</td></tr><tr><td>Mass and wearability</td><td>Glove &lt; 50 g, system &lt; 85 g, donning &lt; 30 s, wear  $> 2 \mathrm { ~ h ~ }$ </td></tr></table>

## C. Contact-Semantic Alignment

Calibration standardizes part of the numerical preprocessing, but sensor layout, contact distribution, and dynamic response still differ between human and robot hands. TACROSS aligns regional contact semantics (Fig. 4), reducing domain differences irrelevant to control while retaining contact information.

1) Contact Canonicalization across Embodiments: For shared anatomical region $r , \textrm { \textit { g } } _ { \mathrm { { e } } }$ aggregates the available measurements in domain e. The domain-specific configuration $\psi _ { e }$ specifies the channel-to-region correspondence and calibration settings. Correspondence follows anatomy rather than channel number. The canonicalizers are

![](images/a71fd5e8100792eeb9b01313b12f00a7951a57f1f9fd6d9bbbc3dd4a5e81528c.jpg)  
<sup>synchronization</sup> <sup>of</sup> <sup>RGB,</sup> <sup>pose,</sup> <sup>tactile,</sup> <sup>and</sup> <sup>robot</sup> <sup>state.</sup> <sup>(c)</sup> <sup>Multimodal</sup> <sup>episode</sup> <sup>stored</sup> <sup>on</sup> <sup>disk</sup> <sup>and</sup> <sup>used</sup> <sup>as</sup> <sup>input</sup> <sup>to</sup> <sup>train</sup> <sup>ACT.</sup>Fig. 3. TACROSS human demonstration acquisition pipeline. (a) Tactile glove and on-wrist readout integrated with motion capture (Rokoko Ego) or vision-based inverse kinematics (Vision-IK Ego). (b) Sensor calibration and causal synchronization using native timestamps, selecting the latest available sample at or before each reference time. (c) Aligned RGB images, hand poses, and five-finger tactile observations form multimodal episodes for subsequent tactile representation and policy learning.

![](images/eeb29432a5c7a8605283518a53d744a082c031bc64a6c7e4d1402341640ba042.jpg)  
Fig. 4. Contact canonicalization and adaptation, shared spatiotemporal encoding, and ACT learning. Each window contains 80 tokens (16 frames × five fingers), processed by shared Transformers with two layers, four attention heads, and features of dimension 128. H is a learned hand token, F denotes finger features, and repeated labels identify the same tensor. Training uses masked canonical targets in both domains and contrastive attraction/repulsion with phase/force hard negatives. P projects latents from 256 dimensions to tokens with 512 dimensions through Linear, GELU, and LayerNorm layers. The illustrated state has eight dimensions and uses one degree of freedom for hand control, Sec. III-D defines the articulated action interface with 13 dimensions. Deployment uses robot observations.

$$
\begin{array} { r } { ( C _ { t } ^ { e } , M _ { t } ^ { e } ) = g _ { e } ( X _ { 1 : t } ^ { e } , q _ { 1 : t } ^ { e } , \psi _ { e } ) , \quad e \in \{ H , R \} . } \end{array}\tag{6}
$$

Here, $C _ { t } ^ { e }$ stacks regional field vectors $c _ { t , r } ^ { e } ,$ and $M _ { t } ^ { e }$ stacks their componentwise validity masks $m _ { t , r } ^ { e }$ . Both domains use the same field order:

$$
c _ { t , r } ^ { e } = [ p , \widetilde { F } , \Delta \widetilde { F } , d , A , u , v , \rho , \gamma ] _ { t , r } ^ { e } ,\tag{7}
$$

encoding contact probability, relative intensity/change, duration, area, center of pressure, instability proxy, and calibration confidence. Four intensity levels and six causal phases (no contact, onset, loading, stable, instability candidate, release) describe contact evolution. Only available fields are comparable: an unsupported field or region has zero value and mask, whereas valid readings without contact retain valid masks. Here, $\widetilde { F }$ encodes relative tactile intensity rather than calibrated absolute force, and $\rho$ is an instability proxy rather than a direct slip measurement.

2) Adapters for Each Sensor: The scale, response statistics, and temporal behavior of common fields still depend on the sensor. Separate shallow adapters $A _ { H } , A _ { R }$ map them to regional tokens with a common interface:

$$
\begin{array} { r } { u _ { t , r } ^ { e } = A _ { e } \big ( [ c _ { t , r } ^ { e } \odot m _ { t , r } ^ { e } , m _ { t , r } ^ { e } , e ^ { F } , e ^ { P } , x _ { t , r } , e _ { r } ] \big ) . } \end{array}\tag{8}
$$

The inputs are masked fields, masks, intensity/phase embeddings, regional context, and anatomical identity. Domain identity selects the adapter, it is not an additional input to the shared encoder.

3) Shared Spatiotemporal Encoding: A causal window of 16 frames at 15 Hz enters temporal and finger Transformers, each with two layers. Temporal attention models contact evolution, attention across fingers models coordination. Both domains share encoder weights, with padding masks for missing fingers. A learned hand token yields a latent $\boldsymbol { z } _ { t } ^ { e }$ with 256 dimensions, finger tokens feed contact, force, and phase heads. Hand latents are normalized for contrastive learning.

4) Objectives for Alignment across Domains: Positive pairs of human and robot windows are selected for semantic compatibility using task/object, primitive, metadata on active fingers, intensity levels, and phase, without requiring synchronized frames across domains. Multi-positive contrastive learning attracts compatible hand latents and separates negatives. Supervision for contact, force regression, ordinal force, and phase retain predictive contact information, together with contrastive discrimination, they discourage a constant representation. Temporal regularization constrains evolution, while domain-adversarial learning discourages sensor identity [16]. The combined objective is

$$
\begin{array} { r } { \displaystyle \mathcal { L } _ { \mathrm { r e p } } = \sum _ { j = 1 } ^ { 7 } \lambda _ { j } \mathcal { L } _ { j } , } \\ { ( \lambda _ { j } ) _ { j = 1 } ^ { 7 } = ( 1 , 1 , 0 . 5 , 0 . 5 , 1 , 0 . 2 , 0 . 1 ) . } \end{array}\tag{9}
$$

The terms $\mathcal { L } _ { 1 } { - } \mathcal { L } _ { 7 }$ are, respectively, the contrastive loss on hand latents, contact-prediction loss, relative-intensity regression loss, intensity-level loss, phase-classification loss, temporal regularizer, and domain-adversarial loss.

## D. Robot-Grounded Policy Learning

Robot Demonstrations and Retargeted Human Actions. Robot demonstrations provide the sole source of ground-truth action supervision. Human hand targets generated by HaMeR and a calibrated geometric retargeter serve as pseudo-labels for confidence-weighted auxiliary supervision with validity masking. From RGB, HaMeR recovers 21 hand keypoints in three dimensions. We express them in a palm frame centered at the wrist and normalize by palm width. The direction from the index MCP to the little finger MCP defines the lateral axis. The direction from the wrist to the middle finger MCP and cross products complete the orthonormal frame.

For non-thumb finger f, let j ∈ {MCP, PIP, DIP} index its flexion joints. The calibrated angle $\widehat { \theta } _ { f , j }$ measures joint closure relative to the user’s open/fist references. Wholefinger closure is $\begin{array} { r } { c _ { f } = \sum _ { i } \beta _ { f , j } \hat { \theta } _ { f , j } } \end{array}$ , where $\beta _ { f , j } \geq 0$ is the contribution of joint j and $\textstyle \sum _ { j } \beta _ { f , j } \ = \ 1$ . Thumb flexion, abduction, and the relative position of the thumb and index fingertips describe opposition. These features map to Revo2’s six active control channels according to its actuation and coupling geometry. User calibration poses and robot safe endpoints determine channel offsets, scales, and directions. Low-pass filtering, limits on changes per cycle, and joint bounds precede SDK execution. Invalid poses trigger a brief hold, followed by the configured safe state for sustained loss, invalid pseudo targets are excluded from supervision.

Policy Interface and Training. [22]–[24] We use two modes of hand control. In single-DOF mode, the entire hand is treated as one control unit with only open and close commands, $b \in \{ 0 , 1 \}$ . In six-DOF retargeting mode, the geometric mapping above produces targets $\boldsymbol { h } \in \mathbb { R } ^ { 6 }$ for Revo2’s six active joints. With targets for seven arm joints, the corresponding action interfaces are

$$
a ^ { ( 8 ) } = [ a ^ { \mathrm { a r m } } , b ] , \qquad a ^ { ( 1 3 ) } = [ a ^ { \mathrm { a r m } } , h ] .\tag{10}
$$

ACT learns from ground-truth robot actions and receives auxiliary supervision from retargeted human hand pseudolabels, weighted by confidence and masked by validity. Arm targets require separate supervision.

At 15 Hz, ACT takes three $2 4 0 \times 3 2 0$ RGB views, robot proprioception, and a projection of the tactile latent from 256 to 512 dimensions. Robot touch is sampled at 30 Hz and causally synchronized. ACT predicts chunks of 60 actions, each with 8 or 13 dimensions. The tactile encoder is frozen for 20,000 steps, then further trained at a lower learning rate. Deployment uses robot observations and the learned policy. The framework can also be extended to other robot policy architectures.

## IV. EXPERIMENTS

## A. Experimental Setup

Tasks and evaluation. We evaluate four contact-rich tasks on real robots: T1 Fixed-Point Dispensing, T2 Beaker-to-Beaker Liquid Transfer, T3 Precision Pipetting, and T4 Sequential Fruit Pick-and-Place. These tasks emphasize accurate dispensing location, stable liquid transfer, dose control, and stable handling of delicate fruit, respectively. A rollout is successful only when it completes all required task stages within the time limit and satisfies the task constraints. For T4, success requires completing the entire prescribed fruit sequence. Fig. 5 illustrates the four tasks.

For task $k ,$ success rate is $\mathrm { S R } _ { k } = 1 0 0 s _ { k } / n _ { k }$ , where $s _ { k }$ is the number of successful rollouts and $n _ { k }$ is the number of evaluated rollouts. The macro average across the four tasks is $\begin{array} { r } { \overline { { \mathrm { S R } } } = \frac { 1 } { 4 } \sum _ { k = 1 } ^ { 4 } \mathrm { S R } _ { k } } \end{array}$ . We recompute this average from the values for each task before rounding. The separate Sequential Fruit Pick-and-Place experiment on data budgets uses 40 trials per configuration and reports $s _ { k } / n _ { k }$ explicitly.

Robot and learning protocol. All comparisons use the same visual inputs, interface for robot state, ACT backbone, action space, and closed-loop control settings described in Sec. III-D. Deployment uses robot observations only. Robot demonstrations provide ground-truth action targets, while retargeted human hand targets serve as pseudo-labels for confidence-weighted auxiliary supervision with validity masking. The robot and human training counts are specified with the corresponding policy comparisons below. These configurations are analyzed separately from the larger T4 experiment on data budgets.

## B. Data Acquisition Cost and Efficiency

Recording throughput and valid data. Robot-only teleoperation records approximately 40 episodes/h. Rokoko Ego (Ego-R) and Vision-IK Ego (Ego-V) record approximately 112 and 140 episodes/h, corresponding to 2.8× and 3.5× the robot recording rate. Human capture does not continuously occupy the target robot, enabling additional human contact data to be collected independently of robot availability. Table II also reports corpus size and accepted episodes. Validity rates are recomputed as accepted episodes divided by recorded episodes. Across the three collection configurations, the corpus totals 183.95 h of recorded trajectories, 24,913 recorded episodes, and 22,486 valid episodes, with these

T1: Fixed-Point Dispensing  
![](images/c5a9847f864c5e41720cc4f91b20cf7a1e4800b26015408f694b0af917532103.jpg)  
Fig. 5. Representative human and robot demonstrations for T1–T4 (top to bottom). Each task shows human keyframes (left), tactile maps (middle), robot keyframes (right), and index-finger (blue) and thumb (orange) force traces (far right). Timestamps indicate elapsed time in each demonstration.

![](images/acc55c9ff6dfb163e893c534e6e9e8eff1441cbac64e06bba230f8759f206a81.jpg)  
Fig. 6. Evaluation setup on the robot with the arm, Revo2 tactile hand, RGB cameras, task fixture, and receiving tray.

totals distinct from subsets used for policy training.

TABLE II  
DATA COLLECTION EFFICIENCY, SYSTEM COST, AND TASK PERFORMANCE.
<table><tr><td>Metric</td><td>Robot-only</td><td>Ego-R</td><td>Ego-V (Ours)</td></tr><tr><td>Total recording duration (h)</td><td>24.86</td><td>89.49</td><td>69.60</td></tr><tr><td>Acquisition system cost (USD) ↓</td><td>12,000</td><td>2,150</td><td>520</td></tr><tr><td>Recorded demonstrations</td><td>1,236</td><td>12,784</td><td>10,893</td></tr><tr><td>Valid demonstration rate (%)</td><td>86.89</td><td>91.28</td><td>89.44</td></tr><tr><td>Valid demonstrations</td><td>1,074</td><td>11,669</td><td>9,743</td></tr><tr><td>Collection speedup ↑</td><td>1.00×</td><td>2.80×</td><td>3.50×</td></tr><tr><td>Average task success (%)</td><td>92.2</td><td>93.5</td><td>91.5</td></tr></table>

Robot-only: teleoperation; Ego-R: Rokoko mocap + glove; Ego-V (Ours): vision-based IK + glove; both use robot anchors. Robot:Ego is the demonstration ratio (1:0, 1:5 for the Fig. 7 results). Success averages four tasks; speedup is vs. Robot-only; corpus totals differ from training subsets.

Equipment investment. The full-hand glove has a component BOM of USD 10.86, with 285 physical sensing locations and an output rate of approximately 150 Hz. The current Robot-only, Ego-R, and Ego-V acquisition configurations cost approximately USD 12,000, USD 2,150, and USD 520. Relative to Robot-only, the listed Ego-R and Ego-V equipment investments are 82.1% and 95.7% lower. These costs cover demonstration acquisition equipment. Table III compares glove hardware.

## C. Ego Data Gains and Requirements for Robot Demonstrations

Adding Ego data at a fixed robot count. Table IV compares Robot-only with human-augmented training. With the same 30 robot demonstrations, full TACROSS with 150 human demonstrations achieves 91.9% mean success across four tasks, versus 51.3% for Robot-only (+40.6 percentage points, pp). Ego data and the transfer pipeline improve task success without additional robot demonstrations. Section IV-D examines the contributions of tactile input and alignment.

Increasing human data with 15 fixed robot episodes. Table V shows that increasing human demonstrations from 0 to 90 raises four-task mean success from 32.5% to 85.6%, demonstrating improved task performance through Ego-data scaling with robot demonstrations fixed at 15.

Reducing the robot count. Table IV compares the 1:5 and 1:10 robot:human allocations. Changing from 30 robot/150 human to 16 robot/160 human demonstrations reduces the robot count by 46.7%, while mean success decreases from 91.9% to 88.3% (-3.6 pp). Ego augmentation reduces robotdemonstration requirements while maintaining high task success.

TABLE III  
COMPARISON OF FULL-HAND TACTILE GLOVES
<table><tr><td></td><td>FlexGlove <img src="images/dea79cc1c710af80ad5519d496baa0be8f6e41c9b5cef93bbabb803b32427144.jpg"/></td><td>Elastic Tactile Glove</td><td>Tekscan Grip 4256E</td><td>Wuji Glove</td><td>TACROSS glove (Ours) 堂</td></tr><tr><td>Cost/glove (USD)↓</td><td>126.70</td><td>320.70</td><td>10,000-15,000</td><td>3,750</td><td>10.86</td></tr><tr><td>Active taxels</td><td>167</td><td>102</td><td>349</td><td>549</td><td>285 designed</td></tr><tr><td>Coverage</td><td>Palm, thumb,finger pads</td><td>Palm, fingertips,phalanges</td><td>18 relocatable regions</td><td>Full palmar side</td><td>≈ 200 cm²;&gt;85%</td></tr><tr><td>Density / pitch</td><td>No fixed pitch;</td><td>Irregular cells; 2 mm min. pitch</td><td>6.20 cm−2; 4 mm</td><td>≈ 6.25 cm 2; 4 mm</td><td>1.50 / 1.31 cm~²; 8.20 / 8.70 mm</td></tr><tr><td>Frame rate</td><td>60 Hz</td><td></td><td>750 Hz</td><td>120 Hz</td><td>150 Hz</td></tr><tr><td>Force / pressure range</td><td>175 kPa</td><td></td><td>0–345 kPa</td><td>0-20 N</td><td>0–20 N on one taxel</td></tr><tr><td>Cost/taxel (USD) ↓</td><td>0.76</td><td></td><td>28.65–42.98</td><td>6.83</td><td>0.04</td></tr></table>

Comparison of full-hand tactile gloves. Specifications are drawn from publications and product documentation. TACROSS values reflect our hardware design and component BOM. The Tekscan Grip 4256E cost (USD 10,000–15,000) and derived cost/taxel are estimates, not a verified manufacturer quotation Prototype BOMs and commercial system prices have different cost bases.

TABLE IV  
ABLATION STUDY OF HUMAN TACTILE INPUT ANDCONTACT-SEMANTIC ALIGNMENT
<table><tr><td>Characteristic</td><td>A</td><td colspan="2">B</td><td colspan="2">C</td><td colspan="2">Ours</td></tr><tr><td>Robot:Ego ratio</td><td>1:0</td><td>1:5</td><td>1:10</td><td>1:5</td><td>1:10</td><td>1:5</td><td>1:10</td></tr><tr><td>Robot episodes</td><td>30</td><td>30</td><td>16</td><td>30</td><td>16</td><td>30</td><td>16</td></tr><tr><td>Human episodes</td><td>0</td><td>150</td><td>160</td><td>150</td><td>160</td><td>150</td><td>160</td></tr><tr><td>Fixed-Point Dispensing</td><td>52.5</td><td>77.5</td><td>77.5</td><td>47.5</td><td>45.0</td><td>95.0</td><td>92.5</td></tr><tr><td>Beaker-to-Beaker Liquid Transfer</td><td>47.5</td><td>72.5</td><td>70.0</td><td>50.0</td><td>50.0</td><td>92.5</td><td>87.5</td></tr><tr><td>Precision Pipetting</td><td>50.0</td><td>67.5</td><td>65.0</td><td>47.5</td><td>45.0</td><td>92.5</td><td>87.5</td></tr><tr><td>Sequential Fruit Pick-and-Place</td><td>55.0</td><td>75.0</td><td>72.5</td><td>52.5</td><td>47.5</td><td>87.5</td><td>85.0</td></tr><tr><td>Average success</td><td>51.3</td><td>73.1</td><td>71.3</td><td>49.4</td><td>46.9</td><td>91.9</td><td>88.1</td></tr></table>

For B/C/Ours, the left/right subcolumns denote R:H = 1:5/1:10. A: Robot teleoperation only. B: Robot teleoperation + Human Ego without tactile. C: Robot teleoperation + Human Ego without Contact-Semantic Alignment. Ours: Full TACROSS with Contact-Semantic Alignment.

TABLE V  
SCALING HUMAN DATA WITH 15 FIXED ROBOT EPISODES.
<table><tr><td>Characteristic</td><td>A</td><td>B</td><td>C</td><td>D</td><td>E</td><td>F</td></tr><tr><td>Robot episodes</td><td>15</td><td>15</td><td>15</td><td>15</td><td>15</td><td>105</td></tr><tr><td>Human episodes</td><td>0</td><td>20</td><td>40</td><td>60</td><td>90</td><td>0</td></tr><tr><td>Human ratio (%)</td><td>0.0</td><td>57.1</td><td>72.7</td><td>80.0</td><td>85.7</td><td>0.0</td></tr><tr><td>Collection cost (USD)</td><td>2.00</td><td>2.03</td><td>2.06</td><td>2.09</td><td>2.14</td><td>14.00</td></tr><tr><td>T1 success (%)</td><td>40.0</td><td>42.5</td><td>65.0</td><td>80.0</td><td>90.0</td><td>95.0</td></tr><tr><td>T2 success (%)</td><td>45.0</td><td>47.5</td><td>65.0</td><td>77.5</td><td>85.0</td><td>87.5</td></tr><tr><td>T3 success (%)</td><td>27.5</td><td>47.5</td><td>65.0</td><td>75.0</td><td>82.5</td><td>82.5</td></tr><tr><td>T4 success (%)</td><td>17.5</td><td>37.5</td><td>55.0</td><td>72.5</td><td>85.0</td><td>87.5</td></tr><tr><td>T1 time (min)</td><td>7.3</td><td>12.1</td><td>17.0</td><td>22.5</td><td>30.6</td><td>52.2</td></tr><tr><td>T2 time (min)</td><td>15.4</td><td>22.6</td><td>29.3</td><td>37.1</td><td>48.1</td><td>105.4</td></tr><tr><td>T3 time (min)</td><td>21.3</td><td>30.1</td><td>39.9</td><td>49.3</td><td>63.3</td><td>148.8</td></tr><tr><td>T4 time (min)</td><td>28.9</td><td>40.8</td><td>51.5</td><td>62.8</td><td>79.2</td><td>201.3</td></tr></table>

A–E use 0, 20, 40, 60, and 90 human episodes, respectively, with 15 robot episodes fixed in each group. F is a robot-only baseline with 105 real-world robot episodes and no human demonstrations. T1–T4: Fixed-Point Dispensing, Beaker-to-Beaker Liquid Transfer, Precision Pipetting, and Sequential Fruit Pick-and-Place.

Mixing ratios under fixed budgets for collection time. Fig. 7 compares Robot-only, Rokoko-based Ego, and vision-IK-based Ego acquisition under fixed per-task collectiontime budgets. At the nominal 1:5 ratio, the two Ego systems achieve 93.5% and 91.5% macro success, respectively, versus 92.2% for Robot-only, while at 1:10 they achieve 87.5% and 86.5%. The 1:5 mixture offers a better balance between Egodata use and task success than 1:10.

Sequential Fruit Pick-and-Place study of data budgets. The separate T4 study evaluates robot–Ego allocations with 40 rollouts per configuration. Robot-only uses 92 robot demonstrations and achieves 38/40 successes (95.0%). Using 50 robot and 50 Ego demonstrations achieves 37/40 (92.5%), reducing robot demonstrations by 45.7% with a 2.5 pp decrease in observed success. At 75% and 90% Ego, 25 and 10 robot demonstrations yield 36/40 (90.0%) and 34/40 (85.0%), respectively. Ego data can partially replace robot demonstrations on T4, with higher Ego proportions further reducing robot-data requirements but lowering success.

Estimated acquisition time. For the fruit allocations, acquisition time is estimated as

$$
\widehat { T } = N _ { R } \frac { 1 . 5 0 0 } { \alpha _ { R } } + N _ { H } \frac { 6 0 / 1 4 0 } { \alpha _ { V } } \quad \mathrm { m i n } ,\tag{11}
$$

Here, $\alpha _ { R } ~ = ~ 1 , 0 7 4 / 1 , 2 3 6$ and $\alpha _ { V } ~ = ~ 9 , 7 4 3 / 1 0 , 8 9 3$ are the Robot-only and Vision-IK validity rates from Table II. The 50-robot/50-Ego allocation reduces estimated collection time from 158.8 to 110.3 min (30.6%) relative to 92 robot demonstrations, indicating savings in both acquisition time and robot-data requirements.

## D. Ablation Analysis

Ego tactile input and contact-semantic alignment. Table IV compares the full model with variants excluding

(b) Acquisition cost

![](images/93ec8d196ca06278345526f6319d53ebffcaf8bac37c84ba5c1d6f00b782b771.jpg)  
Fig. 7. Success rates and acquisition equipment costs for different robot–Ego data mixtures. (a) Task success under fixed collection time budgets of 50, <sub>Ours</sub> <sub>(1:10)</sub>150, 200, and 250 min for T1–T4, respectively. (b) Equipment cost for each acquisition system. A uses robot-only teleoperation, B combines Rokoko motion capture and a tactile glove, and Ours combines vision-based IK and a tactile glove. Both Ego systems use robot demonstrations as anchors. Ratios denote nominal robot:Ego episode counts.

human tactile input or contact-semantic alignment under two allocations of robot and human training data. With 30 robot and 150 human demonstrations, full TACROSS achieves 91.9% mean success, compared with 73.1% without human tactile input and 49.4% without contact-semantic alignment. With 16 robot and 160 human demonstrations, the corresponding success rates are 88.3%, 71.3%, and 46.9%. Removing human tactile input or contact-semantic alignment reduces success rates in both training settings.

## V. CONCLUSION

We presented TACROSS, an efficient and affordable system for transferring tactile knowledge from humans to robots. It combines tactile acquisition across the hand, alignment of contact semantics across sensors, and policy learning grounded in robot demonstrations with geometric hand retargeting. By combining synchronized human visual, motion, and tactile demonstrations with robot supervision, TACROSS enables faster demonstration collection at lower cost of acquisition equipment while supporting contact-rich dexterous manipulation. This provides a practical way to scale tactile data collection without requiring continuous robot operation.

The limitation of TACROSS is that the transfer still relies on robot demonstrations and retargeting and calibration specific to each embodiment. In the future, we will extend TACROSS to additional robotic hands and tasks, explore tactile representations across embodiments in a shared latent space, and investigate transfer with smaller budgets for robot demonstrations.

## REFERENCES

[1] X. Hu et al., “Research on the development and application of force tactile interaction technology in China,” Intell. Robot., vol. 5, no. 3, pp. 505–519, 2025, doi: 10.20517/ir.2025.26.

[2] W. Liu, J. Wang, Y. Wang, W. Wang, and C. Lu, “ForceMimic: Force-centric imitation learning with force-motion capture system for contact-rich manipulation,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2025.

[3] Z. Wang et al., “HumanEgo: Zero-shot robot learning from minutes of human egocentric videos,” arXiv preprint arXiv:2605.24934, 2026.

[4] N. Gu, K. Kosuge, and M. Hayashibe, “TactileAloha: Learning bimanual manipulation with tactile sensing,” IEEE Robot. Autom. Lett., vol. 10, no. 8, pp. 8348–8355, 2025.

[5] S. Sundaram, P. Kellnhofer, Y. Li, J.-Y. Zhu, A. Torralba, and W. Matusik, “Learning the signatures of the human grasp using a scalable tactile glove,” Nature, vol. 569, no. 7758, pp. 698–702, 2019.

[6] Y. R. Song et al., “OPENTOUCH: Bringing full-hand touch to realworld interaction,” arXiv preprint arXiv:2512.16842, 2025.

[7] Y. Huang et al., “HT-Bench: Benchmarking and learning dexterous full-hand tactile representations with egocentric vision,” arXiv preprint arXiv:2606.19161, 2026.

[8] L. Wu et al., “FreeTacMan: Robot-free visuo-tactile data collection system for contact-rich manipulation,” arXiv preprint arXiv:2506.01941, 2025.

[9] F. Yang et al., “Binding touch to everything: Learning unified multimodal tactile representations,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit., 2024, pp. 26340–26353.

[10] R. Feng et al., “AnyTouch: Learning unified static-dynamic representation across multiple visuo-tactile sensors,” in International Conference on Learning Representations (ICLR), 2025. [Online]. Available: https://openreview.net/forum?id=XToAemis1h

[11] S. Kareer et al., “EgoMimic: Scaling imitation learning via egocentric video,” arXiv preprint arXiv:2410.24221, 2024.

[12] A. Handa et al., “DexPilot: Vision based teleoperation of dexterous robotic hand-arm system,” in Proc. IEEE Int. Conf. Robot. Autom., 2020, pp. 9164–9170.

[13] D. Murphy et al., “Fits like a Flex-Glove: Automatic design of personalized FPCB-based tactile sensing gloves,” arXiv preprint arXiv:2503.06349, 2025.

[14] J. Yin et al., “OSMO: Open-source tactile glove for human-to-robot skill transfer,” arXiv preprint arXiv:2512.08920, 2025.

[15] X. Chen, Y. Pan, M. Li, and X. Ding, “DexViTac: Collecting human visuo-tactile-kinematic demonstrations for contact-rich dexterous manipulation,” arXiv preprint arXiv:2603.17851, 2026.

[16] Y. Ganin et al., “Domain-adversarial training of neural networks,” J. Mach. Learn. Res., vol. 17, no. 59, pp. 1–35, 2016.

[17] C. Zhang, P. Cai, H. Yuan, C. Xu, and Z. Lu, “UniTacHand: Unified spatio-tactile representation for human to robotic hand skill transfer,” arXiv preprint arXiv:2512.21233, 2025.

[18] Y. Wi et al., “TactAlign: Human-to-robot policy transfer via tactile alignment,” arXiv preprint arXiv:2602.13579, 2026.

[19] K. Yu, Y. Han, Q. Wang, V. Saxena, D. Xu, and Y. Zhao, “MimicTouch: Leveraging multi-modal human tactile demonstrations for contact-rich manipulation,” in Proc. 8th Conf. Robot Learning, Proc. Mach. Learn. Res., vol. 270, pp. 4844–4865, 2025.

[20] G. Pavlakos, D. Shan, I. Radosavovic, A. Kanazawa, D. Fouhey, and J. Malik, “Reconstructing hands in 3D with transformers,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), pp. 9826– 9836, 2024.

[21] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” in Robotics: Science and Systems (RSS), 2023.

[22] X. Sun et al., “AtomVLA: Scalable post-training for robotic manipulation via predictive latent world models,” arXiv preprint arXiv:2603.08519, 2026.

[23] W. Song et al., “Rethinking the practicality of vision-language-action model: A comprehensive benchmark and an improved baseline,” arXiv preprint arXiv:2602.22663, 2026.

[24] X. Sun et al., “HiMem-WAM: Hierarchical memory-gated world action models for robotic manipulation,” arXiv preprint arXiv:2606.10363, 2026.