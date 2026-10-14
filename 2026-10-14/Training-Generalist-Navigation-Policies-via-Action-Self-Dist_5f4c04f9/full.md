# Training Generalist Navigation Policies via Action Self-Distillation

Bowen Feng<sup>†</sup> Keqin Wang<sup>†</sup> Yulong Yang<sup>†</sup> Amogh Joshi

Christine Allen-Blanchette Dhruv Shah<sup>‡</sup> Felix Heide<sup>‡</sup>

Princeton University, USA

{bf2192, kw4335, yy0555, aj0699}@princeton.edu {ca15, shahd, fheide}@princeton.edu

asd-nav.github.io

Abstract: Language provides a natural interface for humans to specify and reason about goals, but training language-conditioned robotic policies and reasoning typically requires expensive manual annotations. We propose Action Self-Distillation (ASD), a method for learning language-conditioned reasoning for robotic navigation without human-annotated language or reasoning labels. ASD uses an open-source VLM, which contains broad world knowledge and reasoning capabili ties, as both a teacher network and a student policy. During training, the self-teacher receives privileged trajectory feedback and infers the underlying navigation goal together with goal-oriented reasoning. The student is then trained to condition on the inferred navigation goal, predict future ego-frame trajectory, and provide a description of the scene semantics, plan, and reasoning, from onboard observations alone. Across 13 challenging indoor and outdoor real-world navigation tasks, our method improves success rate over the strongest baseline by 30% in indoor settings, 30% on multi-stage tasks, and achieves 95% success rate on outdoor tasks, validating that privileged trajectory feedback can enable self-distillation for scalable generalist navigation policies without human annotations.

Keywords: Reasoning for Autonomous Navigation, Language-Conditioned Policy Learning, Self-Distillation, Vision-Language-Action Model

## 1 Introduction

Vision-language-action (VLA) models have emerged as a successful approach to robot learning, enabling policies to condition on natural-language commands while grounding them in visual observations [1, 2]. For navigation, this paradigm is especially appealing because language provides a deployable interface for specifying goals in human-centered environments, without requiring privileged map coordinates or goal images [3, 4, 5, 6]. However, training VLMs to end-to-end navigation typically requires datasets with goal descriptions, scene semantics or reasoning traces, followed by supervised fine-tuning [7, 4]. These annotations are expensive to obtain at scale, and many widely used robot navigation datasets contain only sensor observations and trajectories, with no language supervision [8, 9, 4, 10, 11, 12].

Prior approaches address this gap either by relying on human annotators for high-quality language labels [7, 4, 13, 14], or by orchestrating multiple external models to generate goals, semantic descriptions, or reasoning supervision. Departing from these approaches, we investigate whether a single pretrained VLM can supervise itself from trajectory-only navigation data. This motivates our research question: How can we learn language goal-conditioned end-to-end navigation policies from trajectory-only robot datasets, without relying on dataset language annotations or multi-model annotation pipelines?

![](images/77044cbb9212cfe610cfe5720052a2f21eff7839c27df5c95de60dcd642800cd.jpg)  
Figure 1: Action self-distillation. The self-teacher uses privileged ground-truth trajectory feedback to extract scene semantics, infer a language goal, and generate reasoning traces explaining the demonstrated behavior. Blue arrows denote the feedback-conditioned information flow to the self-teacher, while pink arrows denote the student pathway. The student observes only deployable inputs and learns to align its reasoning and trajectory tokens with the self-teacher, removing the need for external goal or reasoning annotations.

In human interaction with frontier VLMs, feedback plays a central role in refining model behavior when initial responses do not match the user’s intent. A key observation is that when a model is conditioned on corrective or informative feedback about its own outputs, it can recover coherent goal-aligned predictions even when it could not do so from the input alone [15, 16, 17]. Inspired by this feedback-driven reasoning capability, we treat trajectory feedback as privileged information during training. Given such feedback, a VLM can infer the underlying goal, extract relevant scene semantics, and produce planning-oriented reasoning during training. To this end, we propose Training Generalist Navigation Policies via Action Self-Distillation (ASD), a framework in which a single model extracts language goals from trajectory-conditioned scene reasoning as self-teacher, and then learns to generate goal-conditioned trajectories together with corresponding reasoning traces.

ASD instantiates this idea through a self-distillation procedure with two passes. In the teacher pass, the VLM receives the current observation together with ground-truth future trajectory, projected onto an egocentric observation image. This ground truth trajectory as privileged feedback information guides the model to produce structured perception-prediction-planning reasoning and to extract a natural language goal. In the student pass, the model is trained to condition on the extracted goal and predict both future actions and structured reasoning from onboard observations alone. The self-teacher is implemented as an EMA-updated copy of the student [18], allowing language-goal and reasoning supervision to be generated within the training loop. Critically, ASD operates directly on raw image-trajectory pairs, requiring no human-provided language goals or reasoning annotations. To our knowledge, this is the first application of feedback-conditioned self-distillation on robot control, converting existing unannotated navigation datasets into goal-conditioned reasoning supervision without external labeling.

We deploy ASD on a FrodoBot rover in zero-shot manner across a range of out-of-distribution natural language goals in both indoor and outdoor environments. The policy is trained on a mixture of RECON [8] and FrodoBot-2K [9], without any language goal or reasoning annotations. Across 9 indoor and 4 outdoor language-conditioned real-world experiments, we compare ASD against competitive VLA navigation baselines, including OmniVLA [7] and NaVILA [19]. ASD demonstrates language goal steerability across directional, color-based, and object-centric commands, and further generalizes to both multi-stage and indoor navigation tasks that are unseen during training. Across all 13 real-world tasks, ASD achieves the highest overall success rate, outperforming the strongest baseline, OmniVLA, by 30% in the indoor setting and reaching a 95% success rate in the outdoor setting. On multi-stage tasks, ASD further improves over the leading baseline, NaVILA, by 30%.

## 2 Related Work

Learning-based Ground Navigation. Early learning-based navigation networks like BADGR [20] showed that ground robots can learn navigation directly from real-world experience, moving away from hand-designed cost functions. ViNG [21] and RECON [8] further demonstrated visual-goal navigation and open-world exploration by combining local navigation with topological information or latent goals. More recent foundation-style navigation policies, including GNM [10], ViNT [11], and NoMaD [12], scale this idea by training goal-conditioned visual navigation policies on heterogeneous robot data, with NoMaD introducing diffusion policies and goal masking to unify goal-directed navigation and exploration. While these methods demonstrate strong large-scale learned navigation capabilities, they remain limited to visual goal conditioning and do not support natural language instructions, which are easier to specify and more practical for real-world deployment.

Natural Language Goals and Reasoning in Navigation. Language has long been studied as a natural interface for robot instruction following and vision-and-language navigation (VLN), where agents map free-form commands to navigation actions from visually grounded observations [22, 23, 24, 25]. Recent work has explored language as a goal modality for robot navigation, recognizing that natural language is the most accessible interface for human control. LM-Nav [3] composes an LLM, CLIPstyle image-language grounding, and a visual navigation model to follow language instructions without language-annotated robot data. NavGPT [5] and NavGPT-2 [6] study explicit LLM reasoning for vision-and-language navigation, while NaVILA [19] and VAMOS [14] extend this toward real robot navigation with embodiment-specific planning modules. More recently, OmniVLA [7] and LeLaN [4] achieve language-conditioned navigation on real robots by relying on external labeling pipelines to supply language annotations at training time. While these works demonstrate that language conditioning can benefit navigation, none address the annotation bottleneck from within the training loop. Our approach instead elicits language goals and contextual descriptions entirely through self-distillation, requiring no external annotation of any kind.

Reasoning Distillation and Self-Distillation. Our framework builds on EMA-based self-supervised learning [18], which showed that models can supervise themselves under richer context without external labels. STaR [15] extended this idea to language by bootstrapping reasoning from selfgenerated rationales, while related methods [16, 26, 27] demonstrated that distilling a privilegedcontext teacher into a deployment-time student is effective for capability acquisition. The closest control-domain analogue is SelfD [28], which pseudo-labels driving videos with action supervision and retrains a student policy, but predicts actions rather than language goals or contextual descriptions. Recent embodied reasoning works [29, 30, 31] similarly use privileged VLM reasoning to improve deployable policies, but rely on separate external teacher models. In contrast, we use the same open-source VLM as both teacher and student, grounding it through annotated trajectory feedback to elicit language goals and contextual descriptions within the training loop. To our knowledge, this is the first application of self-distillation as a grounding mechanism for robot control.

## 3 Training Generalist Navigation Policies via Action Self-Distillation

We propose a novel method for training robotic navigation policies via Action Self-Distillation. Policies trained with ASD can navigate to goals specified by natural language, while also reasoning about their actions. We briefly introduce notation and formulation, and then describe ASD.

## 3.1 Problem Formulation

We consider a ground robot that navigates using ego-centric single-view observations. At time $t ,$ the robot receives a front-camera image $o _ { t }$ , and its state as a two-dimensional birds-eye view location $s _ { t } = [ x , y ] \in \mathbb { R } ^ { 2 }$ . Given a history of $T$ past states $\tau _ { - T : 0 } = \{ s _ { - T } , . . . , s _ { 0 } \}$ , the goal of our policy is to predict a future trajectory $\tau _ { 0 : H } = \{ s _ { 0 } , . . . , s _ { H } \}$ over a planning horizon $H$ . We build on a pretrained vision-language model (VLM) π that autoregressively produces text tokens; trajectories are serialized in a JSON-style text format and tokenized by the language model, enabling trajectory prediction via next-token prediction. We seek a policy that additionally emits a structured reasoning trace R and can be conditioned on a natural-language goal $g \colon$

$$
{ \hat { \tau } } _ { 0 : H } , R \sim \pi ( \cdot \mid o _ { - T : 0 } , \tau _ { - T : 0 } , g ) .\tag{1}
$$

Crucially, neither reasoning annotations nor goal labels exist in any of the navigation datasets we train on [8, 9, 4, 10]; all language supervision is generated by the method itself.

![](images/50d08289004097aa55f232a7ee638556db60af4d4f01a6e2fcffe4a0b3392b41.jpg)  
Figure 2: Action Self-Distillation. A single pre-trained VLM is used as both the self-teacher $\pi _ { \phi }$ and student π<sub>θ</sub>. During training, the self-teacher receives privileged information by projecting the ground-truth future trajectory y onto the front-view observation as $o _ { 0 } ^ { * }$ , enabling it to infer a language goal gˆ and scene-grounded reasoning trace R. The student observes only deployable inputs $o _ { 0 }$ and learns to match the teacher’s reasoning tokens with a KL loss, while predicting future trajectory tokens $\tilde { \tau } _ { 0 : H }$ with a cross-entropy loss against the ground-truth trajectory $y .$ The teacher weights are updated as an EMA of the student.

## 3.2 Training: Language Following and Reasoning via Action Self-Distillation

We use the ground-truth trajectory as a privilegedfeedback information to elicit structured semantics and goal information. This privileged feedback conditioned forward pass constitutes the self-teacher, which generates both a structured reasoning trace and a semantic navigation goal. The student is trained using only robot observations, conditioned on the language goal extracted by the self-teacher, and supervised to generate both the reasoning trace and the future trajectory. At inference time, the privileged trajectory input is removed and the extracted training-time goal is replaced by a free-form language goal provided by a human operator, enabling language-conditioned navigation without requiring human-annotated reasoning or goal labels during training. Both teacher and student are instances of the same pretrained VLM; the self-teacher weights ϕ are maintained as an exponential moving average (EMA) of the student weights θ: $\phi  \alpha \phi + ( 1 - \alpha ) \theta$

Self-Teacher Pass. As listed in Line 5 of Algorithm 1, the self-teacher $\pi _ { \phi }$ receives the current camera view $o _ { 0 }$ {future trajectory: augmented with a polyline overlay of the ground-truth trajectory $y = \tau _ { 0 : H }$ (annotated observation $o _ { 0 } ^ { * } )$ , and produces language goal ${ \hat { g } } ,$ <sup>1,y1,...],...}</sup>, reasoning R, and trajectory $\scriptstyle { \hat { \tau } } _ { 0 : H }$ with prompt $P$ as

$$
{ \hat { g } } , \ R , \ { \hat { \tau } } _ { 0 : H } \ \sim \ \mathrm { s g } ( \pi _ { \phi } ( \cdot \mid P , \ o _ { 0 } ^ { * } , \tau _ { - T : 0 } ) ) .\tag{2}
$$

Student Pass. The student pass is summarized in Line 6 of Algorithm 1. Specifically, the student $\pi _ { \theta }$ receives the image observation $o _ { 0 }$ dent<sup>,</sup> <sup>state</sup> <sup>history</sup> $\tau _ { - T : 0 }$ , and the self-teacher’s extracted goal $\hat { g } .$ , and produces a reasoning trace $\tilde { R }$ Output<sub>and</sub> <sub>trajectory</sub> <sub>with</sub> <sub>the</sub> <sub>mirrored</sub> <sub>prompt</sub> $\tilde { P }$ as

$$
\tilde { R } , \tilde { \tau } _ { 0 : H } \sim \pi _ { \theta } \Big ( \cdot \mid \tilde { P } , o _ { 0 } , \tau _ { - T : 0 } , \hat { g } \Big ) .\tag{3}
$$

Training Objective. The output token sequence is partitioned at the onset of the trajectory block. Scene SemanticTokens in the reasoning span (Perception, Agents, Prediction, Planning) are supervised by a KL divergence loss against the self-teacher’s logits, that is

$$
{ \mathcal { L } } _ { \mathrm { r e a s o n } } = D _ { \mathrm { K L } } \Big ( R \parallel \tilde { R } \Big ) .\tag{4}
$$

Information as Goal<sup>Tokens</sup> <sup>in</sup> <sup>the</sup> <sup>trajectory</sup> <sup>span</sup> <sup>are</sup> <sup>supervised</sup> <sup>by</sup> <sup>cross-entropy</sup> <sup>against</sup> <sup>the</sup> <sup>ground-truth</sup> <sup>trajectory</sup>

$$
\mathcal { L } _ { \mathrm { t r a j } } = \mathrm { C E } ( \tilde { \tau } _ { 0 : H } , \ y ) .\tag{5}
$$

The combined objective is $\mathcal { L } = \mathcal { L } _ { \mathrm { t r a j } } + \mathcal { L } _ { \mathrm { r e a s o n } }$ , summarized in Line 7-9 of Algorithm 1 .

Structured Reasoning Format. The self-teacher prompt enforces a fixed output structure: Task (navigation goal; goal-conditioned mode only), Perception (visible terrain and structures), Agents (moving entities present in the scene, or “none”), Prediction (expected agent motion), and Planning (a discrete action from a fixed vocabulary, justified by named scene features), followed by the executable trajectory. The structured format constrains the teacher to ground its reasoning in observable scene content; the teacher prompt explicitly instructs the model to describe only what is visible in the raw image, so that any claim can be verified against the unannotated observation. Full prompt templates for each dataset are provided in the Section B of supplementary material.

## 3.3 Deployment: Navigating to Language Goals

At deployment, the self-teacher is absent and no ground-truth trajectory is available. The student policy $\pi _ { \theta } ^ { \star }$ plans a trajectory and reasoning trace directly from the current observation and history as

$$
\tilde { R } , \tilde { \tau } _ { 0 : H } \sim \pi _ { \theta } ^ { \star } \Big ( \cdot \mid \tilde { P } , o _ { 0 } , \tau _ { - T : 0 } \Big ) .\tag{6}
$$

In the goal-conditioned setting, a human operator supplies a natural-language goal $g ^ { \circledast }$ in place of the self-teacher’s extracted $\hat { g }$ via

$$
\tilde { R } , \ \tilde { \tau } _ { 0 : H } \sim \pi _ { \theta } ^ { \star } \Big ( \cdot \mid \tilde { P } , o _ { 0 } , \tau _ { - T : 0 } , g ^ { \circledast } \Big ) .\tag{7}
$$

The goal $g ^ { \circledast }$ may take several forms, including a path description (e.g. “navigate along the sidewa $\lfloor \ k ^ { \prime \prime } ) .$ , or an object reference (e.g. “navigate towards the red building”). Because the student is trained on goals $\hat { g }$ produced by the self-teacher from diverse scenes across the full training distribution, this constitutes zero-shot language-conditioned navigation. The model generalizes to operator-supplied instructions at test time without ever seeing human-specified goals during training.

Algorithm 1: Training Policies via Action Self-Distillation   
Input: Trajectory-only dataset D, student $\pi _ { \theta } ,$ self-teacher $\pi _ { \phi } ,$ EMA rate α   
Output: Trained goal-conditioned navigation policy $\pi _ { \theta }$   
1 Initialize θ from a pretrained VLM and set $\phi  \theta ;$   
2 for each training iteration do   
3 Sample $\left( o _ { 0 } , \tau _ { - T : 0 } , y \right)$ from ${ \mathcal { D } } ,$ where $y = \tau _ { 0 : H } ;$   
4 Project y onto $o _ { 0 }$ to obtain privileged observation ${ \tilde { o } } _ { 0 } ;$   
5 $( \hat { g } , \bar { R } , \hat { \tau } _ { 0 : H } ) \sim \pi _ { \phi } ( \cdot \mid P , o _ { 0 } ^ { * } , \tau _ { - T : 0 } ) , \mathrm { E q . } ( 2 ) ;$   
6 $( \tilde { R } , \tilde { \tau } _ { 0 : H } ) \sim \pi _ { \theta } ( \cdot \mid \tilde { P } , o _ { 0 } , \tau _ { - T : 0 } , \hat { g } ) , \mathrm { E q . } ( 3 )$   
7 $\mathcal { L } _ { \mathrm { r e a s o n } }  D _ { \mathrm { K L } } ( R \| \tilde { R } ) , \mathrm { E q . } ( 4 )$   
8 $\mathcal { L } _ { \mathrm { t r a j } }  \mathrm { C E } ( \tilde { \tau } _ { 0 : H } , y )$ , Eq.(5);   
9 $\mathcal { L }  \mathcal { L } _ { \mathrm { t r a j } } + \mathcal { L } _ { \mathrm { r e a s o n } } ;$   
10 Update θ by minimizing $\mathcal { L } ;$   
11 $\phi  \alpha \phi + ( 1 - \alpha ) \theta ;$

## 4 Experiments

In this section, we conduct experiments to assess ASD by addressing the following two questions: Q1: How well does ASD generalize across real-world environments compared with state-of-the-art navigation VLAs? Q2: What capabilities are induced by ASD?

## 4.1 Experimental Setup

ASD uses Qwen2.5-VL-3B-Instruct [32] as the base VLM, hereafter referred to as Qwen2.5-VL, and is trained through the self-distillation pipeline introduced in Section 3, on a mixture of the RECON [8] and FrodoBot2k [9] datasets. The training details are summarized in Section B of the Supplementary Material. We evaluate it in real-world settings across both indoor and outdoor environments. We focus on language-conditioned navigation tasks, where user prompts ask the robot to navigate to a specific goal. All experiments are deployed on a FrodoBot EarthRover Mini equipped with a front-facing camera. At each planning step, ASD predicts receding-horizon trajectory, and converted by the robot’s built-in API into linear and angular velocity commands for execution.

Table 1: Indoor Language-Conditioned Navigation. We evaluate tasks across three categories: directional, object-centric, and multi-stage navigation. Difficulty level increases with the number of possible directions for directional tasks, and the number of objects in the scene for object-centric tasks. In the multi-stage category, Doorway requires the robot to pass through a door, while Hallway further requires it to turn right after the doorway and move along the hallway. We use $- , + ,$ and ∗ to denote easy, medium, and hard tasks. More detailed task descriptions and visualizations are provided in Section C.1 of the supplementary material.
<table><tr><td rowspan="2">Method</td><td colspan="3">Directions</td><td colspan="4">Objects</td><td colspan="2">Multi-stage</td><td rowspan="2">Avg.</td></tr><tr><td>Straight Door (1 direction)</td><td>Left chair+ (2 directions)</td><td>Right chair* (3 directions)</td><td>Blue⁻ chair</td><td>Trash bin</td><td> ${ \mathrm { C h a i r } } ^ { + }$  (2 objects) (3 objects)</td><td>Monitor*</td><td>Doorway+ Hallway*</td><td></td></tr><tr><td>OmniVLA</td><td>0.8</td><td>0.9</td><td>0.2</td><td>0.7</td><td>1.0</td><td>0.4</td><td>0.2</td><td>0.7</td><td>0.0</td><td>0.54</td></tr><tr><td>NaVILA</td><td>0.3</td><td>0.2</td><td>0.1</td><td>0.4</td><td>0.1</td><td>0.3</td><td>0.2</td><td>0.3</td><td>0.5</td><td>0.27</td></tr><tr><td>Qwen2.5-VL-ASD</td><td>0.9</td><td>1.0</td><td>0.8</td><td>1.0</td><td>0.9</td><td>0.9</td><td>0.7</td><td>0.8</td><td>0.6</td><td>0.84</td></tr><tr><td>Qwen3.5-VL-ASD</td><td>1.0</td><td>1.0</td><td>0.9</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.9</td><td>0.9</td><td>0.8</td><td>0.94</td></tr></table>

## 4.2 Indoor Language-Conditioned Navigation

We conduct indoor real-world navigation experiments in a laboratory environment with typical office objects, including chairs, trash bins, boxes, and monitors. We evaluate 9 tasks covering different levels of difficulty, from simple directional and object-identification goals, such as “Navigate to the blue chair”, to multi-stage goals, such as “Navigate through the door, turn right, then follow the hallway”. For each task, we use fixed language prompts and test each method under four different initializations, where the initial position and orientation of the robot and objects are varied. We compare ASD against state-of-the-art language-conditioned VLAs, OmniVLA [7] and NaVILA [19]. See Section C of the Supplementary for detailed tasks information and evaluation.

Indoor language-conditioned navigation results are reported in Table 1. Our ASD achieves consistently high success rates across all tasks and obtains the best average success rate as 84%. Among three hard tasks, including “Navigate to the right chair”, “Navigate to the monitor” and “Navigate through the door, turn right, then follow the hallway”, ASD significantly outperforms leading baseline by 60%, 50% and 10% respectively, demonstrating strong generalization capability in complex tasks.

The task performance of ASD remains relatively consistent compared with competitive baselines. Across all evaluated tasks, ASD achieves success rates above 50%. In contrast, OmniVLA and NaVILA obtain substantially lower success rates on several tasks and even exhibit near-total failure in certain scenarios, such as OmniVLA, which achieves 0% success in Hallway. These results suggest that the baselines have limited robustness to specific navigation scenarios, which we further analyze in Section 4.4.

## 4.3 Outdoor Language-Conditioned Navigation

We conduct outdoor real-world navigation experiments in a campus environment. We evaluate 4 tasks, including navigate along the sidewalk, and navigate to the statue, tree and car. We follow the same experimental setup and compare to the same baselines as in the indoor setting. See Supplementary Section C for tasks and evaluation details. Results are reported in Table 2, where we observe that ASD achieves consistently high success rates across all tasks and obtains the best average suc-

Table 2: Outdoor Language-Conditioned Navigation. We evaluate on 4 different tasks, including navigate along the sidewalk, navigate to the statue/tree/car. More detailed task descriptions and visualizations are provided in Section C.2 of the supplementary material.
<table><tr><td>Method</td><td>Sidewalk</td><td>Statue</td><td>Tree</td><td>Car</td><td>Avg.</td></tr><tr><td>OmniVLA</td><td>1.0</td><td>0.6</td><td>1.0</td><td>1.0</td><td>0.90</td></tr><tr><td>NaVILA</td><td>0.9</td><td>0.1</td><td>0.0</td><td>0.3</td><td>0.33</td></tr><tr><td>Qwen2.5-VL-ASD</td><td>1.0</td><td>0.8</td><td>1.0</td><td>1.0</td><td>0.95</td></tr><tr><td>Qwen3.5-VL-ASD</td><td>1.0</td><td>0.9</td><td>1.0</td><td>1.0</td><td>0.98</td></tr></table>

cess rate of 95%. In contrast, the baselines either show lower success rates on specific tasks, such as OmniVLA achieving 60% on Statue, or frequently fail on several tasks, such as NaVILA achieving 0% on Tree and 10% on Statue. We further discuss the result in Section 4.4.

## 4.4 Behavior Assessment

In this section, we provide a detailed assessment and discussion of ASD’s learned capabilities, qualitative behavior and relative performance gains compared to baselines.

Zero-Shot Navigation to Out-of-Distribution Objects and Goals. Indoor navigation as described in Sec. 4.2 is evaluated in Table 1, highlighting ASD’s ability to generalize to out-of-distribution (OOD)

![](images/f0d7810f8bbc695e99acaf27aa15910f4289340deaec1ac87cca9e595140d684.jpg)

Figure 3: Steerability Demonstration. We demonstrate ASD’s steerability under color (left), direction (middle), and object (right) prompts. Within the same setting, our method can be directed to different targets.  
![](images/efdfe38d24f96808a007be736b6df44961ab908fd044216dde82ad70c0669f81.jpg)  
(a)  
(b)  
(c)  
(d)  
(e)  
Figure 4: Single-Stage Navigation Across Diverse Indoor and Outdoor Environments. We demonstrate ASD’s capability to execute single-stage navigation goals specified in natural language. The examples are ordered from left to right as two indoor tasks (right chair and monitor) and three outdoor tasks (car, statue, and tree). Full videos are provided in the Supplementary Material.

objects, goals, and environments. In particular, our indoor experiments involve laboratory objects and indoor navigation goals that are absent from the training data, since both RECON and FrodoBot-2K are primarily outdoor navigation datasets. Despite this large domain shift, ASD maintains 84% average success, indicating strong generalization to novel object categories, language goals, and indoor environments. We observe a similar trend in the outdoor experiments in Table 2: "navigate to the statue" requires grounding an OOD object category, yet ASD achieves an 80% success, outperforming OmniVLA by 20% and NaVILA by 70%.

Steerability. ASD is steerable with different language goals within the same environment. Fig. 3 shows that ASD can be directed by different Directions, interpreting directional commands, and different Objects, requiring to ground distinct object categories. In the Car example, ASD distinguishes object attributes, such as color. These examples validate the generalization and languagefollowing capabilities. This is notable because our training data follows a one-on-one scene and goal supervision: each observation is paired with a single language goal and a single demonstrated trajectory, without counterfactual trajectory augmentation. In contrast, methods such as OmniVLA rely on broader multimodal training and over 9,500 hours of data collected from 10 platforms. Despite using substantially less data, ASD generalizes to produce distinct trajectories conditioned on different language goals in the same scene, demonstrating more data-efficient steerable behavior.

Multi-Stage Task Execution Capabilities. ASD is capable of executing complex, multi-stage goals, as evaluated in the Multi-stage tasks in Table 1. Doorway is a two-stage task where the robot needs to navigate to the door and go through it. Hallway is a four-stage task where the robot needs to further turn right after passing the door and move along the hallway. Our ASD achieves success rates of 80% and 60% on the shorter and longer tasks, respectively. In contrast, OmniVLA fails on the longer task with a 0% success rate, while NaVILA performs poorly on the shorter task with a 30% success rate. Figure 5 reports additional outdoor and indoor multi-stage navigation tasks.

![](images/d7e8d5f3134e8583996d19724726ff66c7322c0d85616c7c2a6b9aa75d0ba7c4.jpg)  
Figure 5: Multi-Stage Reasoning Capabilities for Outdoor and Indoor Navigation Tasks. We demonstrate ASD’s capability to execute complex, multi-stage navigation goals. The first row contains an indoor navigation demonstration, while the bottom row contains an outdoor navigation demonstration, both with their respective tasks described on the figure. Full videos are provided in the Supplementary Material.

![](images/28d32f814353ded0ba5d40ddd225a218e5925d3ca39abbf291dea6d235dcc097.jpg)  
Figure 6: Reasoning Legibility. We present representative findings from our real-world experiments, showing predicted trajectories alongside the corresponding highlighted reasoning traces. These examples demonstrate that the generated reasoning is interpretable and well aligned with the predicted robot behavior.

Reasoning. Figure 6 shows that ASD’s reasoning aligns with both the goal and the predicted trajectory. In the indoor examples, the model reasons to turn left or right toward the requested chair. In the outdoor examples, it reasons to turn left toward the red car or the statue, relating each planned motion to the target location and the open space ahead.

Foundation Model Evaluations. We perform zero-shot evaluation with vanilla Qwen2.5-VL and Claude Opus 4.7 on the Left chair task. Qwen2.5-VL yields 0% success rates, producing degenerate trajectories. Claude Opus 4.7 generates plausible reasoning and correctly identifies navigation targets, achieving 70% success rate. However, its inference latency reaches several minutes per planning step, leading to an extremely slow average completion time of 846.1 seconds. This renders it impractical for real-world navigation within a reasonable time frame. In contrast, ASD achieves 100% success rate while maintaining a relatively fast completion time. These observations suggest that frontier VLMs possess strong semantic reasoning, but still require a grounding to translate this into actionable, low-latency robot behavior. Our approach provides this grounding mechanism.

Table 3: Task success rates and completion times of foundation models on indoor navigation task. We extend the evaluation of Left chair in Table 1 by including Qwen2.5-VL and Claude Opus 4.7, and report the task completion time for all methods. We omit the completion time for Qwen2.5-VL because its predicted actions are generally physically infeasible.
<table><tr><td>Method</td><td>SR↑</td><td>Time (s) ↓</td></tr><tr><td>OmniVLA NaVILA</td><td>0.9 0.2</td><td>25.3 28.9</td></tr><tr><td>Qwen2.5-VL</td><td>0.0</td><td></td></tr><tr><td>Claude Opus 4.7</td><td>0.7</td><td>846.1</td></tr><tr><td>Qwen2.5-VL-ASD</td><td>1.0</td><td>15.3</td></tr></table>

Importance of Reasoning. We ablate the reasoning traces in Table 4. The performance of Qwen2.5- VL-ASD (SR 80%) decreases when we disable the reasoning traces at training (SR 26%) or inference (SR 14%), demonstrating the importance of reasoning.

Table 4: Frozen-teacher baselines and reasoning ablations. Success rate on representative tasks from Table 1 and Table 2 requiring directional, object-level, multi-stage, and outdoor reasoning. Goal-only supervision trains on self-teacher-inferred goals without reasoning traces. Reasoning off at inference disables reasoning at test time. Labels + SFT baselines fine-tune on offline annotations from a frozen Claude Opus 5 or Qwen2.5-VL teacher. Best results are in bold, second best are underlined
<table><tr><td rowspan="2">Method</td><td>Directions</td><td colspan="2">Objects</td><td>Multi-stage</td><td>Outdoor</td><td rowspan="2">Avg.</td></tr><tr><td>Right chair</td><td>Chair</td><td>Monitor</td><td>Hallway</td><td>Car</td></tr><tr><td>Qwen2.5-VL-ASD (goal-only supervision)</td><td>0.4</td><td>0.2</td><td>0.2</td><td>0.1</td><td>0.4</td><td>0.26</td></tr><tr><td>Qwen2.5-VL-ASD (reasoning off at inference)</td><td>0.1</td><td>0.2</td><td>0.2</td><td>0.0</td><td>0.2</td><td>0.14</td></tr><tr><td>Claude Opus 5 labels + SFT</td><td>0.3</td><td>0.5</td><td>0.3</td><td>0.3</td><td>0.4</td><td>0.36</td></tr><tr><td>Qwen2.5-VL labels + SFT</td><td>0.4</td><td>0.3</td><td>0.3</td><td>0.0</td><td>0.1</td><td>0.22</td></tr><tr><td>NaVILA (RECON + FrodoBots-2K SFT)</td><td>0.0</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.2</td><td>0.16</td></tr><tr><td>Qwen2.5-VL-ASD</td><td>0.8</td><td>0.9</td><td>0.7</td><td>0.6</td><td>1.0</td><td>0.80</td></tr><tr><td>Qwen3.5-VL-ASD</td><td>0.9</td><td>1.0</td><td>0.9</td><td>0.8</td><td>1.0</td><td>0.92</td></tr></table>

Frozen-teacher Baselines & Benefit of EMA Self-Distillation. We ablate the teacher in selfdistillation by comparing against offline-label SFT baselines trained with annotations from Claude Opus 5 or Qwen2.5-VL, as shown in Table 4. We find that Qwen2.5-VL-ASD (SR 80%) outperforms offline labeling with Qwen2.5-VL (SR 22%) and Opus 5 (SR 36%). Using the same training dataset (RECON + FrodoBots-2K) as ASD, we fine-tune NaVILA, which achieves only a 16% success rate. These empirically support the benefit of EMA self-distillation over SFT on teacher pre-annotated pseudo-labels, consistent with findings that teacher-student capacity mismatch degrades distillation performance [33, 34]. Additionally, Qwen3.5-VL-ASD (SR 92%) achieves better performance with a stronger backbone.

## 5 Conclusion

We present Training Generalist Navigation Policies via Action Self-Distillation, a method for learning instruction-following navigation policies from trajectory-only robot data. Using privileged trajectory feedback during training, ASD generates language goals and structured reasoning without human annotations, enabling the student policy to follow complex natural-language instructions and act in the real world. Across 13 real-world zero-shot tasks on a mobile robot, ASD demonstrates strong generalization to OOD directions, colors, objects, indoor laboratory environments, and multistage tasks. In indoor settings, ASD outperforms the strongest baseline, OmniVLA, by 30%; in outdoor settings, it achieves 100% success on in-distribution tasks and 80% success on the OOD commands, outperforming OmniVLA by 20% and NaVILA by 70%. For multi-stage tasks, ASD further improves over the leading baseline, NaVILA, by 30%, despite being trained on significantly less data. Akin to training reasoning-enabled foundation models of language, we hope that ASD serves as a generic recipe for training generalist robot policies with reasoning without requiring expensive human annotations.

Limitations. While ASD removes the need for language annotations, it still relies on demonstration trajectories as privileged feedback; future work could replace these with interactive human feedback for more precise task-level guidance. The quality of self-generated goals and reasoning also depends on the pretrained VLM and may degrade in ambiguous scenes, with small targets, or under finegrained spatial instructions.

## Acknowledgments

This work and Felix Heide were supported by an NSF CAREER Award (2047359), Packard and Sloan Fellowships, and research awards from Samsung, Project X, Amazon Science, and Bosch. Felix Heide is a co-founder of Algolux (now Torc Robotics) and Cephia AI, and Head of AI at Torc Robotics.

## References

[1] A. Brohan, Y. Chebotar, C. Finn, K. Hausman, A. Herzog, D. Ho, J. Ibarz, A. Irpan, D. Kalashnikov, Y. Kuang, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. Conference on Robot Learning (CoRL), 2023.

[2] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[3] D. Shah, B. Osinski, S. Levine, et al. Lm-nav: Robotic navigation with large pre-trained models´ of language, vision, and action. In Conference on robot learning, pages 492–504. pmlr, 2023.

[4] N. Hirose, C. Glossop, A. Sridhar, D. Shah, O. Mees, and S. Levine. Lelan: Learning a languageconditioned navigation policy from in-the-wild video. In Conference on Robot Learning, 2024.

[5] G. Zhou, Y. Hong, and Q. Wu. Navgpt: Explicit reasoning in vision-and-language navigation with large language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 7641–7649, 2024.

[6] G. Zhou, Y. Hong, Z. Wang, X. E. Wang, and Q. Wu. Navgpt-2: Unleashing navigational reasoning capability for large vision-language models. In European Conference on Computer Vision, pages 260–278. Springer, 2024.

[7] N. Hirose, C. Glossop, D. Shah, and S. Levine. Omnivla: An omni-modal vision-languageaction model for robot navigation. arXiv preprint arXiv:2509.19480, 2025.

[8] D. Shah, B. Eysenbach, G. Kahn, N. Rhinehart, and S. Levine. Rapid exploration for open-world navigation with latent goal models. arXiv preprint arXiv:2104.05859, 2021.

[9] N. Hirose, L. Ignatova, K. Stachowicz, C. Glossop, S. Levine, and D. Shah. Learning to drive anywhere with model-based reannotation. IEEE Robotics and Automation Letters, 11(2): 1242–1249, 2025.

[10] D. Shah, A. Sridhar, A. Bhorkar, N. Hirose, and S. Levine. Gnm: A general navigation model to drive any robot. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 7226–7233. IEEE, 2023.

[11] D. Shah, A. Sridhar, N. Dashora, K. Stachowicz, K. Black, N. Hirose, and S. Levine. Vint: A foundation model for visual navigation. arXiv preprint arXiv:2306.14846, 2023.

[12] A. Sridhar, D. Shah, C. Glossop, and S. Levine. Nomad: Goal masked diffusion policies for navigation and exploration. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 63–70. IEEE, 2024.

[13] H.-T. L. Chiang, Z. Xu, Z. Fu, M. G. Jacob, T. Zhang, T.-W. E. Lee, W. Yu, C. Schenck, D. Rendleman, D. Shah, et al. Mobility vla: Multimodal instruction navigation with longcontext vlms and topological graphs. arXiv preprint arXiv:2407.07775, 2024.

[14] M. G. Castro, S. Rajagopal, D. Gorbatov, M. Schmittle, R. Baijal, O. Zhang, R. Scalise, S. Talia, E. Romig, C. de Melo, et al. Vamos: A hierarchical vision-language-action model for capability-modulated and steerable navigation. arXiv preprint arXiv:2510.20818, 2025.

[15] E. Zelikman, Y. Wu, J. Mu, and N. Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

[16] S. Zhao, Z. Xie, M. Liu, J. Huang, G. Pang, F. Chen, and A. Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

[17] C.-Y. Hsieh, C.-L. Li, C.-K. Yeh, H. Nakhost, Y. Fujii, A. Ratner, R. Krishna, C.-Y. Lee, and T. Pfister. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 8003–8017, 2023.

[18] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33: 21271–21284, 2020.

[19] A.-C. Cheng, Y. Ji, Z. Yang, Z. Gongye, X. Zou, J. Kautz, E. Bıyık, H. Yin, S. Liu, and X. Wang. Navila: Legged robot vision-language-action model for navigation. arXiv preprint arXiv:2412.04453, 2024.

[20] G. Kahn, P. Abbeel, and S. Levine. Badgr: An autonomous self-supervised learning-based navigation system. IEEE Robotics and Automation Letters, 6(2):1312–1319, 2021.

[21] D. Shah, B. Eysenbach, G. Kahn, N. Rhinehart, and S. Levine. Ving: Learning open-world navigation with visual goals. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pages 13215–13222. IEEE, 2021.

[22] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sünderhauf, I. Reid, S. Gould, and A. Van Den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 3674–3683, 2018.

[23] J. Gu, E. Stefani, Q. Wu, J. Thomason, and X. Wang. Vision-and-language navigation: A survey of tasks, methods, and future directions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7606–7623, 2022.

[24] A. Ku, P. Anderson, R. Patel, E. Ie, and J. Baldridge. Room-across-room: Multilingual vision-and-language navigation with dense spatiotemporal grounding. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4392–4412, 2020.

[25] B. Lin, Y. Nie, K. Loun Zai, Z. Wei, M. Han, R. Xu, M. Niu, J. Han, L. Lin, C. Lu, et al. Evolvenav: Self-improving embodied reasoning for llm-based vision-language navigation. arXiv e-prints, pages arXiv–2506, 2025.

[26] J. Hübotter, F. Lübeck, L. Behric, A. Baumann, M. Bagatella, D. Marta, I. Hakimi, I. Shenfeld, T. K. Buening, C. Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026.

[27] I. Shenfeld, M. Damani, J. Hübotter, and P. Agrawal. Self-distillation enables continual learning. arXiv preprint arXiv:2601.19897, 2026.

[28] J. Zhang, R. Zhu, and E. Ohn-Bar. Selfd: Self-learning large-scale driving policies from the web. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17316–17326, 2022.

[29] B. Feng, Z. Mei, B. Li, J. Ost, F. Ghilotti, R. Girgis, A. Majumdar, and F. Heide. Verdi: Vlm-embedded reasoning for autonomous driving. arXiv preprint arXiv:2505.15925, 2025.

[30] Y. Xu, Y. Hu, Z. Zhang, G. P. Meyer, S. K. Mustikovela, S. Srinivasa, E. M. Wolff, and X. Huang. Vlm-ad: End-to-end autonomous driving through vision-language model supervision. arXiv preprint arXiv:2412.14446, 2024.

[31] D. Chen, B. Zhou, V. Koltun, and P. Krähenbühl. Learning by cheating. In Conference on robot learning, pages 66–75. PMLR, 2020.

[32] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, et al. Qwen2. 5 technical report. arXiv e-prints, pages arXiv–2412, 2024.

[33] Y. Li, X. Yue, Z. Xu, F. Jiang, L. Niu, B. Y. Lin, B. Ramasubramanian, and R. Poovendran. Small models struggle to learn from strong reasoners. In Findings of the Association for Computational Linguistics: ACL 2025, pages 25366–25394, 2025.

[34] C. Zhang, Q. Li, D. Song, Z. Ye, Y. Gao, and Y. Hu. Towards the law of capacity gap in distilling language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 22504–22528, 2025.

[35] D. A. Reynolds et al. Gaussian mixture models. Encyclopedia of biometrics, 741(659-663):3, 2009.

[36] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, 44(10-11):1684–1704, 2025.

[37] Y. Wang, W. Luo, J. Bai, Y. Cao, T. Che, K. Chen, Y. Chen, J. Diamond, Y. Ding, W. Ding, et al. Alpamayo-r1: Bridging reasoning and action prediction for generalizable autonomous driving in the long tail. arXiv preprint arXiv:2511.00088, 2025.

[38] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Martín-Martín. What matters in learning from offline human demonstrations for robot manipulation. In Conference on Robot Learning (CoRL), 2021.

[39] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

## A Comparing ASD vs. Supervised Finetuning

In this section, we provide further insight into the experimental findings reported in the main paper. We observe that ASD, despite being fine-tuned on substantially less data (∼ 100 hours demonstration data (ASD) v.s. ∼ 9500 hours demonstration data (OmniVLA)) than OmniVLA and NaVILA, exhibits better generalization on indoor out-of-distribution evaluations where ASD has not been explicitly fine-tuned.

In the following, we provide insight into why supervising the reasoning span of our policy with the self-distillation loss $\mathcal { L } _ { \mathrm { r e a s o n } }$ in Eq. (4) leads to better generalization and retention of the pretrained VLM’s capabilities than standard offline imitation.

## A.1 Privileged Soft-label Distillation Objective

During training, the self-teacher $\pi _ { \phi }$ generates a reasoning trace using privileged trajectory feedback

$$
R \sim \mathrm { s g } [ \pi _ { \phi } ( \cdot \mid P , o _ { 0 } ^ { * } , \tau _ { - T : 0 } ) ] ,\tag{8}
$$

where $o _ { 0 } ^ { * } = f ( o _ { 0 } , y )$ is the annotated observation obtained by projecting the ground-truth future trajectory $y = \tau _ { 0 : H }$ onto the front-camera observation $o _ { 0 }$ . The stop-gradient prevents the reasoning loss from updating the self-teacher directly. The student receives only deployable inputs and the self-teacher’s extracted goal $\hat { g }$ as

$$
\tilde { R } \sim \pi _ { \theta } \Big ( \cdot \mid \tilde { P } , o _ { 0 } , \tau _ { - T : 0 } , \hat { g } \Big ) .\tag{9}
$$

The reasoning span is supervised by a forward-KL distillation objective

$$
{ \mathcal { L } } _ { \mathrm { r e a s o n } } = D _ { \mathrm { K L } } \Big ( R \parallel \tilde { R } \Big ) .\tag{10}
$$

$$
{ \mathcal { L } } _ { \mathrm { r e a s o n } } = D _ { \mathrm { K L } } \Big ( \mathrm { s g } [ \pi _ { \phi } ( \cdot \mid P , o _ { 0 } ^ { * } , \tau _ { - T : 0 } ) ] \Big | \Big | \pi _ { \theta } ( \cdot \mid \tilde { P } , o _ { 0 } , \tau _ { - T : 0 } , \hat { g } ) \Big ) ,\tag{11}
$$

where the KL divergence is applied only over the reasoning-token span. Equivalently, for the reasoning tokens $R = ( r _ { 1 } , \ldots , r _ { N } )$ , this can be written autoregressively as

$$
\mathcal { L } _ { \mathrm { r e a s o n } } = \sum _ { i = 1 } ^ { N } \mathbb { E } _ { R _ { < i } \sim \pi _ { \Psi } ( \cdot | P , o _ { 0 } ^ { \star } , \tau - \tau ; 0 ) } [ D _ { \mathrm { K L } } ( \mathrm { s g } [ \pi _ { \phi } ( \cdot \mid P , o _ { 0 } ^ { * } , \tau _ { - T : 0 } , R _ { < i } ) ] \| \pi _ { \theta } ( \cdot \mid \tilde { P } , o _ { 0 } , \tau _ { - T : 0 } , \hat { g } , R _ { < i } ) ) ] .\tag{12}
$$

This form makes explicit that ASD transfers the teacher’s full token distribution at each reasoning step, rather than only a single hard token.

Expanding Eq. (11) over teacher-generated reasoning traces gives

$$
{ \mathcal { L } } _ { \mathrm { r e a s o n } } = \mathbb { E } _ { R \sim \pi _ { \phi } ( \cdot \vert P , o _ { 0 } ^ { * } , \tau _ { - T : 0 } ) } \left[ \log \frac { \pi _ { \phi } ( R \vert P , o _ { 0 } ^ { * } , \tau _ { - T : 0 } ) } { \pi _ { \theta } ( R \vert \tilde { P } , o _ { 0 } , \tau _ { - T : 0 } , \hat { g } ) } \right] .\tag{13}
$$

Since the teacher distribution is passed into a stop-gradient, the first term is constant with respect to $\theta .$ Thus, optimizing Eq. (13) is equivalent to minimizing the teacher-weighted cross entropy

$$
{ \mathcal { L } } _ { \mathrm { r e a s o n } } \equiv - \mathbb { E } _ { R \sim \pi _ { \phi } ( \cdot \vert P , o _ { 0 } ^ { * } , \tau _ { - T : 0 } ) } \left[ \log \pi _ { \theta } ( R \vert \ \tilde { P } , o _ { 0 } , \tau _ { - T : 0 } , \hat { g } ) \right] + \mathrm { c o n s t . }\tag{14}
$$

The resulting gradient is

$$
\begin{array} { r } { \nabla _ { \theta } \mathcal { L } _ { \mathrm { r e a s o n } } = - \mathbb { E } _ { R \sim \pi _ { \phi } ( \cdot | P , o _ { 0 } ^ { * } , \tau - T ; 0 ) } \left[ \nabla _ { \theta } \log \pi _ { \theta } ( R \mid \tilde { P } , o _ { 0 } , \tau _ { - T ; 0 } , \hat { g } ) \right] . } \end{array}\tag{15}
$$

Therefore, the student is trained to place probability mass on reasoning traces that the self-teacher considers likely.

## A.2 Comparison to Standard SFT

Standard SFT can also be viewed as matching a behavior-policy distribution. Let $\pi _ { b } ( R ~ \mid$ $\tilde { P } , o _ { 0 } , \tau _ { - T : 0 } , \hat { g } )$ denote the offline annotation process that produces reasoning traces. Then SFT minimizes

$$
\mathcal { L } _ { \mathrm { S F T } } = - \mathbb { E } _ { R ^ { \star } \sim \pi _ { b } } \left[ \log \pi _ { \theta } ( R ^ { \star } \mid \tilde { P } , o _ { 0 } , \tau _ { - T : 0 } , \hat { g } ) \right] = H ( \pi _ { b } ) + D _ { \mathrm { K L } } ( \pi _ { b } \parallel \pi _ { \theta } ) .\tag{16}
$$

However, in typical offline datasets, $\pi _ { b }$ is observed only through one hard sampled annotation $R ^ { \star }$ per input. Thus, the empirical target distribution collapses to

$$
\hat { \pi } _ { b } ( R \mid \cdot ) = { \bf 1 } [ R = R ^ { \star } ] ,\tag{17}
$$

and SFT reduces to hard-label imitation.

In contrast, ASD constructs a privileged self-teacher distribution

$$
\pi _ { \phi } ( R \mid P , o _ { 0 } ^ { * } , \tau _ { - T : 0 } ) ,\tag{18}
$$

where $o _ { 0 } ^ { * } = f ( o _ { 0 } , y )$ exposes the demonstrated future trajectory to the teacher. Therefore, the target is not merely an offline annotation sampled from a static behavior policy, but a feedback-conditioned reasoning distribution that explains why the demonstrated action is appropriate for the current observation.

This enables ASD to achieve better generalization, because a given scene may admit multiple valid descriptions and reasonings. For example, in the car task, visualized in Figure 7(l), we use the prompt "Navigate to the red car." In this case, there can be multiple valid reasoning traces. The Perception field could describe the scene as "The red car is on the left of the gray car." or equivalently as "The gray car is on the right of the red car." Both descriptions are correct.

However, hard-label imitation in standard SFT, as in Equation 17, forces the model to match one particular description, even when several alternatives are valid. This may limit the model’s ability to represent diverse but correct reasoning patterns. In contrast, our method learns from a distributional target, as in Equation 15, which can capture multiple valid perception modes and therefore provides more flexible supervision.

## B Additional Training Details

## B.1 Datasets

We use a mixture of data from the RECON [8] and FrodoBots-2K [9] datasets for training and validation of ASD. RECON is an outdoor visual navigation dataset collected with a Clearpath Jackal ground robot exploring an unstructured university-campus environment under diverse weather and lighting conditions. Each trajectory pairs a fisheye front-camera image stream with the robot’s commanded actions, and covers scenes ranging from paved sidewalks and grassy fields to parking lots and pedestrian-shared spaces. FrodoBots-2K is a large-scale sidewalk navigation dataset collected with the Earth Rover Zero platform across more than 10 cities, providing thousands of hours of teleoperated driving in pedestrian-shared urban environments under varied lighting, weather, and crowd conditions. Due to the size of the dataset and infeasibility of full training in the project timeline, we choose to evaluate on a subset of the dataset consisting of 25% of the scenes, randomly selected from the available online splits. Compared to RECON’s single-campus Jackal collection, FrodoBots-2K tests generalization on a substantially larger and more visually diverse corpus. The train/validation split is set to 90%:10%.

## B.2 Model and Training Setup

We use Qwen2.5-VL-3B-Instruct [32] as the backbone VLM for both the student and the self-teacher. The student is initialized from the released pretrained checkpoint, and the self-teacher is initialized

with the same weights. During training, the self-teacher is updated as an exponential moving average of the student. We train ASD on a server with 8 A6000 Pro, 256G of RAM, AMD Threadripper 7980X CPU with 64 cores. The key hyperparameters used in our ASD are reported in Table 5.

Table 5: Training hyperparameters. Main hyperparameters used for model training and inference in real-world experiments. Hyperparameters not listed here use the HuggingFace Transformers defaults.
<table><tr><td>Hyperparameter</td><td>Value</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Backbone VLM</td><td>Qwen2.5-VL-3B-Instruct</td><td>Image resolution</td><td>dataset-native</td></tr><tr><td>Max output tokens</td><td>512</td><td>Past waypoints</td><td>8</td></tr><tr><td>Future waypoints</td><td>16</td><td>Prediction horizon</td><td>6.4 s</td></tr><tr><td>Waypoint frequency</td><td>2.5 Hz</td><td>Train/validation split</td><td>9:1</td></tr><tr><td>Optimizer</td><td>adamw_torch_fused</td><td>Learning rate</td><td>1 × 10−5</td></tr><tr><td>Batch size per GPU</td><td>1</td><td>Gradient accumulation</td><td>2</td></tr><tr><td>Training epochs</td><td>2</td><td>LR schedule</td><td>cosine</td></tr><tr><td>Warmup steps</td><td>5</td><td>Random seed</td><td>42</td></tr><tr><td>EMA rate α</td><td>0.999</td><td>Teacher update frequency</td><td>every 10 steps</td></tr></table>

## B.3 Full Prompt Templates for Training

We show the complete prompt templates for the FrodoBots-2K dataset, which produces the paper’s headline numbers. All datasets follow the same structure; only environment-specific nouns (surface type, agent categories, action set) differ. Goal-conditioned variants are marked (GC only).

Template variables. All query templates share the following format keys, populated per sample at training time.

Variable Description   
{hist\_steps} Number of history waypoints shown (default: 8)   
{hist\_hz} Effective history sample rate in Hz (default: 5 Hz)   
{hist\_text} Serialised list of past [x, y] waypoints   
{avg\_speed} Average speed over the history window (m/s)   
{fut\_steps} Number of future waypoints to predict (default: 16)   
{fut\_secs} Future prediction horizon in seconds (default: 6.4 s)   
{fut\_hz} Effective future sample rate in Hz (default: 2.5 Hz)   
{fut\_text} Teacher only: ground-truth future trajectory JSON

Self-Teacher and Student system prompt. Shared by all teacher template variants.

You are an outdoor sidewalk-robot navigation planner.   
You control an Earth Rover Zero navigating sidewalks and outdoor spaces.   
Your sensor is a single forward-facing 1024x576 RGB camera mounted at   
\~0.45 m height.   
Coordinates: x = forward, y = left (meters), origin at current position.

Teacher query template (4-section default). The assistant turn is pre-filled with Perception:.

Past {hist\_steps} waypoints at {hist\_hz} Hz   
(last = [0.0, 0.0], avg speed {avg\_speed} m/s):   
{hist\_text}   
A human operator chose this path for the next {fut\_secs} s   
({fut\_steps} points at {fut\_hz} Hz).   
Justify why this action is sensible using ONLY what is visible in the   
underlying scene. Do not invent agents or obstacles that aren’t actually

visible; many frames on this dataset are empty sidewalks, and "no agents"   
is the correct answer whenever nothing is there.   
Describe only the scene itself, not any markings or overlays on top of it.   
A reader who sees only the raw image must be able to verify your reasoning   
Perception: 1-2 sentences with concrete visual detail about the sidewalk   
and its surroundings -- path surface (concrete, brick, asphalt), edges   
(curbs, grass, walls, parked cars), buildings or foliage, and how much of   
the sidewalk is clear ahead.   
Agents: list each agent actually visible in the image. If the scene is   
empty, write exactly "none".   
Prediction: If you wrote "none" above, write "none" here too. Otherwise   
describe each listed agent’s likely motion in 1-2 sentences.   
Planning: Pick one action from [go straight, turn left, turn right,   
slow down, stop, cross intersection, navigate around] and explain in 1-2   
sentences why this is right, tying the choice to specific features you   
just named in Perception (e.g. "the sidewalk runs straight past a row of   
parked cars", "a driveway cuts across the path at about 6 m").   
Trajectory: {"future\_trajectory": {fut\_text}}

Teacher Task directive (for Goal Conditioning). For goal conditioning, the following directive is inserted immediately before Perception: in the query above. The assistant turn is pre-filled with Task: instead of Perception:. The trainer extracts the emitted Task: sentence and injects it into the student assistant pre-fill; the student never generates it.

Task: one short sentence stating the high-level goal for this scene   
(e.g., "navigate toward the building entrance", "navigate around the   
tree", "navigate past the parked car"). Name a visible target object --   
building, tree, gate, bench, parked car, sign, driveway, lamppost,   
junction. Use positional descriptors ("on the left", "on the right")   
when they help identify a specific object. Do not use ego-motion verbs   
like "turn left" or "go straight" here -- those belong in Planning.

Student query template.

Past {hist\_steps} waypoints at {hist\_hz} Hz   
(last = [0.0, 0.0], avg speed {avg\_speed} m/s):   
{hist\_text}   
Predict the next {fut\_steps} waypoints ({fut\_secs} s at {fut\_hz} Hz).   
Answer in three short sentences, then output the trajectory JSON on a   
single line.   
Perception: sidewalk layout (1 short phrase) and agents from   
[pedestrian, cyclist, vehicle, parked, obstacle, none].   
Prediction: What will each agent do? Write "none" if no agents.   
Planning: What action? Pick one from [go straight, turn left, turn right,   
slow down, stop, cross intersection, navigate around] and give a short   
reason.   
Example: wide sidewalk, no agents; none; go straight, clear walkway.   
Trajectory: {"future\_trajectory": [[x1, y1], [x2, y2], ...]}

Student assistant pre-fill. The student’s assistant turn is pre-filled as follows; loss is computed only over the tokens that follow Perception:. The Task: line (GC only) is prompt context injected by the trainer from the teacher’s output — the student never generates it.

Default (non-GC):

Perception:

Goal-conditioned (GC only), task sentence injected from teacher output:

Task: <extracted task sentence>   
Perception:

KL Calculation. Teacher&student share the same reasoning format, except for a teacher-only inferred goal span, which is excluded when aligning response tokens. We compute Qwen’s standard token-level KL divergence between teacher/student logits before token sampling as forward KL.

## B.4 Training code for reproducibility

For reproducibility, we provide our training code in the Supplementary Material, including data.py, prompt.py, and train.py, together with a README file that describes how to run the training pipeline. Specifically, data.py constructs the image-and-text prompts and batches training samples, prompt.py defines the instruction templates for scene description and path justification, and train.py implements the two-pass self-distillation training loop.

## B.5 Inference Details

We also summarize the goal-conditioned inference procedure in Algorithm 2.

Algorithm 2: ASD Inference (Student Policy)   
Input: New observation $o _ { 0 }$ , history $\tau _ { - T : 0 } .$ , and language goal $g ^ { \circledast }$   
Output: Future trajectory $\tau _ { 0 : H } ^ { \star }$   
1 while the task is not terminated do   
3 $( \tilde { R } , \tilde { \tau } _ { 0 : H } ) \gets \pi _ { \theta } ^ { \star } \big ( o _ { 0 } , \tau _ { - T : 0 } , g ^ { \circledast } \big )$ , Eq. (7) ; // student only   
4 Execute $\tilde { \tau } _ { 0 : K } , K \le H$ , using the robot controller;   
5 Update $o _ { 0 }$ and $\tau _ { - T : 0 }$ from the latest robot state;

Table 6: Prompt for language-conditioned navigation benchmarks.
<table><tr><td></td><td>Task</td><td>Difficulty Prompt</td><td></td></tr><tr><td rowspan="3"></td><td>Door</td><td>★</td><td>&quot;Navigate straight to the door&quot;</td></tr><tr><td>Directions left chair</td><td>*★</td><td>&quot;Navigate to the chair on the left&quot;</td></tr><tr><td>right chair</td><td>***</td><td>&quot;Navigate to the chair on the right&quot;</td></tr><tr><td rowspan="4">Objects</td><td>Blue chair</td><td>★</td><td>&quot;Navigate to the blue chair&quot;</td></tr><tr><td>Trash Bin</td><td>★</td><td>&quot;Navigate to the trash bin&quot;</td></tr><tr><td>Chair</td><td>*★</td><td>&quot;Navigate to the chair&quot;</td></tr><tr><td>Monitor</td><td>**★</td><td>&quot;Navigate to the monitor&quot;</td></tr><tr><td rowspan="2">Multi-stage</td><td>Doorway</td><td>*★</td><td>&quot;Navigate through the door&quot;</td></tr><tr><td>Hallway</td><td>***</td><td>&quot;Navigate through the door, turn right, then follow the hallway&#x27;</td></tr><tr><td rowspan="4">Outdoor</td><td>Sidewalk</td><td>★</td><td>&quot;Navigate along the sidewalk&quot;</td></tr><tr><td>Statue</td><td>*★</td><td>&quot;Navigate to the statue&quot;</td></tr><tr><td>Tree</td><td>*★</td><td>&quot;Navigate to the tree&quot;</td></tr><tr><td>Car</td><td>***</td><td>&quot;Navigate to the red/gray car&quot;</td></tr></table>

## C Language-conditioned Navigation Tasks

Here we provide details of the tasks reported in Section 4. All indoor tasks are evaluated in a laboratory environment containing typical office items such as chairs, trash bins, boxes, and monitors.

![](images/f07fc1936a2e17bf8f09d4ebdd07cd469200c0cf1ca19f00adbc7f37fc8e44c6.jpg)  
Figure 7: Summary of Benchmark Tasks. Panels (a)–(h) show the indoor tasks reported in Table 1, covering directional, object-centric, and multi-stage navigation scenarios. Panels (i)–(l) show the outdoor tasks reported in Table 2. The target object or goal region for each task is highlighted with a cyan bounding box.

All outdoor tasks are evaluated in a campus environment. Figure 7 summarizes all the testing cases, and Table 6 reports the prompt used in each benchmark, which is fixed across ASD, OmniVLA and NaVILA.

## C.1 Indoor Task Setting

## C.1.1 Directions

We evaluate the model’s ability to follow directional instructions when selecting among different target objects. The tasks are ordered from easiest to hardest: door, left chair, and right chair. In the door task (a), the robot is placed in front of a door with no obstacles between them. The user prompt for this task is "Navigate straight to the door", which tests ASD’s capability to go straight. In the left chair task (b), two identical chairs are placed in front of the robot, side by side, and the robot must navigate to the left chair, by following the user prompt: "Navigate to the chair on the left." In the right chair task (c), three identical chairs are placed side by side, and the robot must navigate to the right chair, as following the task prompt: "Navigate to the chair on the right." The door, left chair, and right chair tasks are visualized in Figure 7(a)–(c), respectively. This set of experiments test the directional steering capability, of which the task prompts can be found in Table 6.

## C.1.2 Objects

We evaluate the model’s ability to distinguish different objects. The tasks are ordered from easiest to hardest: blue chair, trash bin, chair and monitor. In the blue chair task, a blue chair is placed within the robot’s field of view, and the robot is prompted to "Navigate to the blue chair". In the trash bin task, a trash bin is placed in a similar setting, and the robot is prompted to "Navigate to the trash bin". In the chair task, a trash bin and a chair are placed in front of the robot, and the robot is prompted to "Navigate to the chair". In the monitor task, a trash bin, a chair, and a monitor are placed in the scene, and the robot is prompted to "Navigate to the monitor". These four tasks are visualized in Figure 7(d)–(g), and the corresponding prompts are summarized in Table 6.

## C.1.3 Multi-stage

We evaluate the model’s ability to complete multi-stage tasks that require executing multiple navigation behaviors in sequence. The tasks are doorway and hallway. In the doorway task, the robot is placed in front of a door and is prompted to "Navigate through the door". In the hallway task, the robot starts from the same physical setup, but must pass through the door, turn right, and then move along the hallway. The corresponding prompt is "Navigate through the door, turn right, then follow the hallway". Both tasks are visualized in Figure 7(h). They share the same physical setup but require different navigation behaviors. The prompts are summarized in Table 6.

## C.2 Outdoor Tasks Setting

We evaluate the model’s ability to navigate to different target objects in outdoor environments, including four tasks: sidewalk, statue, tree, and car. In the sidewalk task (i), the robot navigates along a sidewalk running through the middle of a grassy area, following the prompt "Navigate along the sidewalk". In the statue task (j), the robot navigates to a bronze statue, following the prompt "Navigate to the statue". In the tree task (k), the robot is placed on a sidewalk and needs to navigate to a tree by crossing a grassy area, following the prompt "Navigate to the tree". In the car task (l), the robot starts in a garage and navigates to a car of a specific color, following the prompt "Navigate to the red/grey car". These tasks are visualized in Figure 7(i)–(l), and the corresponding prompts are summarized in Table 6.

## C.3 Evaluation Fairness, Execution and Stopping Criteria

Evaluation Fairness and Role of Reasoning. Across 10 trials per task, evaluators did not provide corrective interventions during execution (e.g. repositioning or issuing recovery commands); so, mean steps to corrective intervention is not applicable. During planning, we evaluate the safety and reasonability of the predicted trajectory, & give all methods three times to replan per episode (same protocol for all models). Episode fails if the robot collided with an obstacle or if the goal is no longer in its FOV.

Execution Behavior and Stopping. Similar to OmniVLA in evaluation/stopping criteria, task completion is judged by method-blinded human evaluator, who stops the episode once the instruction is satisfied. ASD has no learned stopping criterion. Pauses in multi-stage demonstrations are not emergent; they reflect observation and inference latency as ASD replans after executing 8 of 16 predicted actions. OmniVLA predicts only 8 actions, all of which are executed for fairness.

## C.4 Real World Experiments Baselines Implementation

## C.4.1 OmniVLA

We evaluate OmniVLA [7], a vision-based navigation policy that takes a language prompt as the goal input. We use the pretrained omnivla-original checkpoint. During the inference, OmniVLA is queried with the current front-view image and the goal text, and outputs a sequence of 8 future actions. Before execution, the predicted action sequence is checked by human intuitive judgment to determine whether it is safe and reasonable. If the plan is considered unreasonable, OmniVLA is required to replan. We allow at most 3 such heuristic interventions per run; once this limit is exceeded, the run is marked as a failure. For trajectory continuity and smoothness, all eight predicted actions are executed before the next front-view image is acquired. Each run is terminated when the human evaluator judges the task to be successful, or when a collision occurs.

## C.4.2 NaVILA

We also evaluate NaVILA [19], a two-level navigation framework that combines a vision-languageaction model with low-level locomotion skills. We use the pretrained NaVILA’s VLA checkpoint to generate high-level language commands and execute them using the built-in controller of the FrodoBot. At each planning step, the model predicts a future action command, such as “forward 75 cm” or “turn right 45<sup>◦</sup>”. The robot then executes these commands before receiving a new plan. For the multi-stage tasks, Doorway and Hallway, we use the 64-frame checkpoint, which conditions on the previous 64 front-camera frames. For the remaining tasks, we use the 8-frame checkpoint. As in the OmniVLA evaluation, we allow at most three heuristic interventions per run; once this limit is exceeded, the run is marked as a failure. Each run terminates when the human evaluator judges the task to be successful, or when a collision occurs.

## C.4.3 Vanilla Qwen2.5-VL

We perform inference by replacing the ASD checkpoint with the vanilla Qwen2.5-VL-3B-Instruct model, while keeping all other inference settings unchanged.

## C.4.4 Claude Opus 4.7

Claude Opus 4.7 is run in auto mode with low reasoning effort. At each inference step, Claude receives the current front-view image, reasons about the navigation goal, and generates executable motion commands, including the duration for turning and moving forward. The robot then executes the generated command before the next inference step. We choose low reasoning effort because Claude’s reasoning latency is very high, and using low effort reduces the inference time. We use two helper functions, save\_fig.py and send\_command.py, to interact with the robot. The function save\_fig.py saves the current front-view image from the robot to disk, while send\_command.py sends the generated control command to the robot. We then prompt Claude to navigate the robot using the following instruction:

Task: {task-specific prompt listed in Table 6}.   
Please read the current front-view image saved by save\_fig.py. Based on this image   
and the task goal, generate a control command and send it to the robot using   
send\_command.py. After the command is executed, call save\_fig.py again to   
obtain the updated front-view image. Repeat this observe-plan-act loop until   
the task is completed. Stop the process when you judge that the robot has   
reached the target.

## D Additional Real World Qualitative and Open-Loop Quantitative Results

## D.1 Open-Loop Evaluation

We conduct the open-loop experiments using the non-goal-conditioned ASD model, where no language goal is provided to the student. The purpose of this comparison is to evaluate ASD in the open-loop trajectory prediction setting and is the best among competitive supervised finetuning baselines, including Gaussian Policy, Gaussian Mixture model, Diffusion Policy, Qwen2.5-VL + SFT, Alpamayo-1+SFT and OpenVLA+SFT.

## D.2 Open-Loop Experiment Setup

In addition to the real-world language-conditioned navigation tasks reported in Section 4, we conduct an open-loop evaluation on the validation splits of RECON and FrodoBots-2K. Following the inference formulation in Equation (7), the model receives the current front-view image and the past ego-frame trajectory as input, and predicts a future trajectory without closed-loop execution. We train our ASD with 8 past waypoints as history and predict of future waypoints, corresponding to a 6.4-second horizon at 2.5 Hz. The predicted trajectory is compared directly with the ground-truth future trajectory. All baselines are trained and evaluated on the same train/validation split.

## D.3 Open-Loop Baselines

We compare against two groups of baselines. The first group consists of non-VLA policies trained directly on the same data, including Gaussian Policy, Gaussian Mixture Model [35], and Diffusion Policy [36]. The second group consists of VLA backbones fine-tuned with standard SFT, including Qwen2.5-VL [32], Alpamayo [37], and OpenVLA [2]. The training details of each baseline are:

Gaussian Policy. We adopted the robomimic’s built-in BC\_Gaussian algorithm [38]. Robomimic is a framework for robot learning from demonstration, which offers a broad set of offline learning algorithms to learn from datasets. The current front image is encoded by a ResNet18 with spatial softmax pooling, then linearly projected into a flat feature vector. The 8 previous ego-frame waypoints are flattened into a 16 dimensional vector. We concatenate the visual vector and position vector into a fusion vector, then pass it through a shared two-layer MLP, after which two parallel MLPs predict the mean and standard deviation of a single 32-dimensional diagonal Gaussian that jointly parameterized the 16 future waypoints. The training objective is the negative log-likelihood of the ground-truth action under this distribution. All the training hyperparameters are kept at robomimic defaults. At evaluation we deterministically output the predicted Gaussian mean.

Gaussian Mixture Model. We adopted robomimic’s built-in BC\_GMM algorithm. The observation encoder, history fusion, and shared MLP trunk are identical to the Gaussian policy; the only architectural difference is the policy head, which parametrizes a 5-mode mixture of 32-dimensional diagonal Gaussians instead of a single one. Training uses the negative log-likelihood of the ground-truth action under the mixture density. At evaluation we draw one sample from the predicted mixture.

Diffusion Policy. We implement the Diffusion Policy baseline on top of the official codebase [36]. Diffusion Policy was originally proposed for robot manipulation, where a DDPM [39] iteratively denoises Gaussian noise into an end-effector trajectory conditioned on visual observations. The observation encoder and history fusion are identical to the BC\_Gaussian and BC\_GMM baselines (ResNet18 + spatial-softmax pooling, concatenated with the 16-d history vector). As for the policy, we use a conditional U-Net that convolves along the temporal dimension to predict the noise given the fused observation. The network is trained to predict the added noise. At evaluation we sample one noise and denoise it to get the trajectory.

Qwen+SFT. This is implemented as ASD’s student policy training.

Alpamayo-1+SFT. We evaluate Alpamayo-R1 [37], a vision-language trajectory model that pairs a frozen Qwen3-VL-8B visual encoder with a small action-expert denoiser that generates the future trajectory via diffusion-style sampling. We fine-tune the expert denoiser (Stage 2 only, VLM frozen) on RECON and FrodoBots-2K for 5 epochs at batch size 8 and learning rate 1e-4. At inference, the model draws a single trajectory sample per query (num-traj=1) for fair single-prediction comparison against the other baselines.

OpenVLA+SFT. We evaluate OpenVLA [2], a 7B-parameter Prismatic vision-language-action model that emits a single discretized (v\_lin, v\_ang) action token pair per image, which we then integrate through a unicycle model to obtain the predicted trajectory. We add a LoRA adapter of rank 32 and fine-tune on RECON and FrodoBots-2K for 5 epochs at batch size 32 and learning rate 5e-4, deriving action targets from each sample’s first ground-truth future step.

## D.4 Open-Loop Quantitative Evaluation.

Table 7 summarizes 2-second open-loop trajectory prediction results on RECON and FrodoBots-2K, where we report the Absolute Trajectory Error (ATE), Average Displacement Error (ADE), Mean Absolute Error (MAE) and Mean Squared Error (MSE). ASD achieves the lowest error on 7 out of 8 comparisons. ASD has a significant improvement over its direct baseline Qwen+SFT. Since both models share the same Qwen2.5-VL backbone, training data, and optimization, the gap is attributable to the reasoning-token supervision rather than to capacity or training recipe. ASD also uniformly surpasses non-VLA behavior cloning methods, indicating that the gain is not merely closing a VLA-to-specialist gap. The remaining VLA baselines, Alpamayo-1 and OpenVLA, fall well behind under SFT alone, suggesting that the choice of pretrained backbone is itself a strong factor and that navigation behavior does not transfer automatically from any driving VLA. Based on these results, we conclude that the feedback-conditioned reasoning distillation improves short-horizon trajectory prediction over standard SFT, and the gain persists at a larger and more diverse data scale.

Table 7: Open-loop evaluation on RECON and FrodoBots-2K. ASD achieves the best performance across both datasets under the open-loop evaluation protocol. Lower is better for all metrics.
<table><tr><td rowspan="2">Method</td><td colspan="4">RECON</td><td colspan="4">FrodoBots-2K</td></tr><tr><td>ADE↓</td><td>ATE↓</td><td>MAE↓</td><td>MSE↓</td><td>ADE↓</td><td>ATE↓</td><td>MAE↓</td><td>MSE↓</td></tr><tr><td>Gaussian Policy</td><td>0.226</td><td>0.321</td><td>0.144</td><td>0.059</td><td>0.034</td><td>0.046</td><td>0.019</td><td>0.002</td></tr><tr><td>Gaussian Mixture Model</td><td>0.260</td><td>0.364</td><td>0.166</td><td>0.084</td><td>0.036</td><td>0.050</td><td>0.020</td><td>0.002</td></tr><tr><td>Diffusion Policy [36]</td><td>0.180</td><td>0.238</td><td>0.108</td><td>0.045</td><td>0.034</td><td>0.045</td><td>0.020</td><td>0.003</td></tr><tr><td>Qwen2.5-VL+ŠFT [32]</td><td>0.192</td><td>0.247</td><td>0.115</td><td>0.042</td><td>0.064</td><td>0.061</td><td>0.027</td><td>0.007</td></tr><tr><td>Alpamayo-1+SFT [37]</td><td>0.303</td><td>0.362</td><td>0.188</td><td>0.091</td><td>0.093</td><td>0.106</td><td>0.055</td><td>0.041</td></tr><tr><td>OpenVLA+SFT [2]</td><td>0.341</td><td>0.417</td><td>0.208</td><td>0.114</td><td>0.246</td><td>0.269</td><td>0.138</td><td>0.072</td></tr><tr><td>Qwen2.5-VL-ASD</td><td>0.174</td><td>0.227</td><td>0.104</td><td>0.037</td><td>0.031</td><td>0.042</td><td>0.017</td><td>0.004</td></tr></table>

## D.5 Real World Experiments Qualitative Analysis

In this section, we analyze the navigation performance beyond the success rate reported in our Table 1 and Table 2 of the main text. We visualize representative examples from both indoor and outdoor settings and compare them with OmniVLA [7] and NaVILA [19]. We observe consistent performance improvements over OmniVLA and NaVILA across several evaluation settings.

Improved performance in multi-object tasks We observe improved performance of ASD in multi-object tasks, such as Chair and Monitor in the indoor setting. In these tasks, OmniVLA achieves success rates of 40% and 20%, respectively, while our ASD reaches 90% and 70%. This suggests that our method is better able to distinguish the target object from multiple candidate objects in the scene. Figure 8 visualizes representative examples comparing OmniVLA and ASD. OmniVLA navigates the robot toward a distractor object (i.e., net) instead of the target (Figure 8a and Figure 8b), while ASD correctly directs the robot to the specified target (Figure 8c and Figure 8d).

![](images/94a3847e73d249a6e85e7b8997f624d90933b678d81e2124dfd9e5e4829ba94b.jpg)  
(a) Chair (OmniVLA)  
(b) Monitor (OmniVLA)  
(c) Chair (ASD)  
(d) Monitor (ASD)

Figure 8: Evaluation on Multi-object Tasks. We visualize the Chair and Monitor tasks for OmniVLA (left) and ASD (right). (a) OmniVLA navigates the robot toward a distractor object, the net, instead of the target chair. (b) OmniVLA again navigates toward the net instead of the target monitor. (c) ASD correctly directs the robot to the chair. (d) ASD correctly directs the robot to the monitor.

Generalization to multi-stage tasks We observe that ASD generalizes well to multi-stage tasks, such as Doorway and Hallway. None of the baselines succeeds reliably on both tasks: OmniVLA achieves a 70% success rate on the two-stage Doorway task but fails on the four-stage Hallway task (0%), while NaVILA achieves a 50% success rate on Hallway but performs poorly on Doorway (30%). In contrast, our ASD achieves higher success rates on both tasks, reaching 80% and 60%, respectively. Figure 9 visualizes representative comparisons on these tasks. In the Doorway task, NaVILA drives the robot straight forward and collides with the table (Figure 9b), while ASD produces a left-turning trajectory that allows the robot to pass through the door (Figure 9c). In the Hallway task, OmniVLA only navigates the robot through the door and fails to complete the subsequent right turn and hallway-following behavior (Figure 10a). In contrast, ASD successfully completes the full multi-stage instruction (Figure 10c).

![](images/83588fa5a564215fe2005ee6bd859d69c20e01273006a29c0a9577c3606a1126.jpg)  
(a) OmniVLA (Fail)

![](images/dad16e72a15e6b9299abfb27cc3079f749613a74d1c1feded29e38a773494c39.jpg)  
(b) NaVILA (Fail)

![](images/a404e6cab7c483be79d3aec972d6962ab082aa63f2e092d0bb439543236d39d1.jpg)  
(c) ASD (Success)  
Figure 9: Evaluation on Doorway tasks. We visualize ASD’s performance on the Doorway task compared with OmniVLA and NaVILA. (a) OmniVLA drives the robot straight forward and collides with the door frame on the left corner. (b) NaVILA also drives the robot straight forward and collides with the table on the right. (c) ASD produces a left-turning trajectory that allows the robot to pass through the door successfully.

![](images/b0de1c5bfc196821c931610b1a8631f4c8168e6467f08a70ba8c47a31cda15b4.jpg)  
(c) ASD (Success)  
Figure 10: Evaluation on Hallway tasks. We visualize ASD’s performance on the Hallway task compared with OmniVLA and NaVILA. (a) OmniVLA navigates the robot through the door but fails to complete the subsequent right turn and hallway-following behavior. (b) NaVILA successfully completes the full multi-stage instruction. (c) ASD also successfully completes the full multi-stage instruction.

![](images/2ebab34d0e8cf21555e1f33166c032dcb427d6f14c93be98b5b24e31a2b21681.jpg)  
Figure 11: Generalization to Bronze Statue. We evaluate ASD, OmniVLA, and NaVILA on the Statue task and visualize representative examples. (a) ASD successfully reaches the target statue. (b) ASD produces a correct plan but fails to reach the target due to execution error on uneven terrain. (c) OmniVLA fails to generate a suitable trajectory toward the statue. (d) NaVILA moves mostly straight and fails to reach the target, due to a mismatch between the prompt style and the model’s expected instruction format.

Generalization to out-of-distribution objects. We observe stronger performance on the Statue task, where the bronze statue can be regarded as an out-of-distribution object with respect to our training datasets. Despite this OOD challenge, our ASD achieves an 80% success rate, outperforming OmniVLA (60%) and NaVILA (10%). Figure 11 visualizes representative comparisons among ASD, OmniVLA, and NaVILA on the Statue task. OmniVLA fails to generate a suitable trajectory toward the statue (Figure 11c), and NaVILA moves mostly straight and fails to reach the target (Figure 11d). The main failure mode of ASD, illustrated Figure 11b, is related to execution: ASD predicts and executes the next 8 steps at once, so when the terrain is not sufficiently flat, the robot may deviate from the planned trajectory during execution.

Model Collapse and Self-Teacher Reasoning Drift. (1) Self-teacher stays close to the non-finetuned Qwen checkpoint. With α = 0.999 and one EMA update every 10 training steps, approximately 93% of the teacher’s EMA weight remains assigned to the pretrained checkpoint, and 7% stems from subsequent student updates. (2) Even the most-changed tokens remain stable. In a more aggressive stress test (α = 0.99, updated every step), KL(Qwen∥our ckpt) increases smoothly from 0.34 to 0.68 nats/token across 500 scenes, without abrupt changes. The most sensitive tail also changes only modestly: p95 surprisal increases from 4.8 to 6.7, while the maximum just increases from 16.8 to 18.0 nats, indicating proper amount task adaption rather than collapse. (3) Reasoning remains neither repetitive nor random. Collapse may move reasoning-token entropy toward two extremes: zero for repetitive sentences, or the uniform limit (≈ 12.4 nats) for random texts. Our entropy shifts mildly (1.36 → 1.67 nats); it neither suffers from the repetitive reasoning nor random text.

## E Video Narratives

We provide a comprehensive video on project website that describes the core motivation and methodology behind ASD. This video features physical real-world demonstrations along with key qualitative comparisons highlighting tasks where our method significantly outperforms the baselines. In the sideby-side video comparisons, a green border denotes a successful trial, while a red border indicates a navigation failure. Notably, we also detail the inference time efficiency of each approach: while ASD runs entirely at normal speed without requiring any acceleration, the leading baselines—including OmniVLA and NaVILA—require substantial video speed-ups (e.g., 4×, 6×, or 8×) to match our real-time execution.

## E.1 Multi-Stage Task

We first show multi-stage demos: Navigate through the door, turn right and then follow the hallway (Hallway); Navigate right to the sidewalk and navigate along the sidewalk. Those demos are impressive as ASD can generalize decently well to multi-stage task prompts with the training only on the single-stage tasks.

ASD demonstrates strong generalization, achieving a 60% success rate on multi-stage Hallway navigation tasks despite training only on single-stage tasks, outperforming the 50% rate of the NaVILA baseline. While NaVILA is adept at multi-stage goals, it shows limited task generalization, with success rates of only 27% indoors and 33% outdoors on single-task prompts, as detailed in Tables 1 and 2.

We further evaluate ASD on a more complex, outdoor multi-stage task to demonstrate its advanced contextual reasoning capabilities. In this scenario, the agent is presented with a choice between two distinct sidewalks. The multi-stage prompt instructs the agent to: "Navigate right to the sidewalk and navigate along the sidewalk." To succeed, ASD cannot rely on simple forward movement; it must first spatially reason about its environment to differentiate between the sidewalk directly ahead and the one situated to its right. It then must successfully execute the sequence of navigating to and following the correct path. This demo highlights the model’s ability to handle fine-grained spatial ambiguity in unconstrained outdoor environments.

## E.2 Extreme OOD Task

We further demonstrate ASD’s generalization capability by evaluating it on a single-task goal prompt featuring an extreme out-of-distribution (OOD) object. Specifically, the task requires the agent to "Navigate to the statue," a target object entirely absent from the training dataset. ASD achieves an impressive 80% success rate on this task. To ensure the robustness of this evaluation, we vary the initial starting positions, forcing the agent to execute specific left or right steering behavior to reach the target. This variation provides strong evidence that the agent’s success is not a byproduct of a blind directional bias, but rather stems from its deep semantic understanding and precise goal-directed steerability. This task is from main Table 2 called statue, where OmniVLA achieves 60% and NaVILA achieves 10% success rate.

## E.3 Indoor Tasks

We present video demonstrations comparing ASD against leading baselines on medium- and harddifficulty tasks, highlighting instances of successful navigation by our method alongside baseline failures.

In the hard mode’s directional capability task, "Navigate to the right chair," the agent must select the correct target from among three identical chairs placed in front of it. On this task, ASD achieves an 80% success rate, significantly outperforming OmniVLA at 20% and NaVILA at 10%. The video qualitative results show that the strongest baseline, OmniVLA, consistently selects the incorrect direction—steering toward the left chair instead of navigating to the rightmost option. This failure mode shows our method’s better directional semantic understanding during goal-directed navigation.

In the medium-difficulty object selection task, the agent is presented with two objects in front of it: a box and a chair. Given the target prompt, "Navigate to the chair," ASD achieves a 90% success rate, substantially outperforming the leading baseline, OmniVLA, which achieves only 40%. The video qualitative results show that OmniVLA consistently deviates to the black net on the right. In contrast, ASD demonstrates an efficient, direct, and clean trajectory toward the target object.

In the hard-difficulty object selection task, the agent is presented with three distinct objects: a trash can, a monitor, and a chair. When evaluated on the target prompt, "Navigate to the monitor," ASD achieves a 70% success rate, significantly outperforming OmniVLA’s 20% success rate. Consistent with the trends observed in the medium-difficulty task, the qualitative video results show that OmniVLA often deviates toward the black net on the right. In contrast, ASD exhibits a much more direct trajectory toward the correct target.

In the long-horizon single-stage evaluation, the prompt "Navigate through the door" explicitly tests the model’s long-horizon planning and execution capabilities. The qualitative video results demonstrate that ASD outperforms the leading baseline, OmniVLA, by a margin of 10%. This performance gap stems from a distinct failure mode in the baseline: OmniVLA crashes into the door frame, whereas ASD successfully navigates through the narrow opening without any collisions.

In the multi-stage task, "Navigate through the door, turn right, then follow the hallway," the leading baseline, NaVILA, performs comparably to our method, with ASD outperforming it by a margin of 10%. Notably, while the video demonstrations show that both models achieve successful navigation, ASD generates a significantly smoother trajectory during turns. Furthermore, our model achieves this performance running at normal speed, whereas the NaVILA baseline requires a 4.7× speed-up to execute the task efficiently.

## E.4 Outdoor Tasks

We present video comparisons of ASD against leading baselines on the Statue, Tree, and Car tasks. These video qualitative results contrast the typical failure modes of the baseline models with the successful execution of our method.

For instance, in the "Navigate to the Statue" task, OmniVLA correctly identifies the target direction but fails to navigate sufficiently close to the statue itself. This precision error highlights the origin of our model’s 20% success rate advantage over the baseline.

In both the Tree and Car tasks, both ASD and the leading baseline, OmniVLA, achieve a perfect success rate of 100%. However, OmniVLA suffers from extremely slow inference times compared to ASD. Specifically, OmniVLA requires a 8.6× and 6.8× speed-up on the Tree and Car tasks, respectively, to match the execution efficiency of our method.