# VIOLA: LEARNING GENERALIST HUMANOIDCONTROL POLICIES FROM HUMAN DATA

Mert Albaba<sup>∗1,2,3</sup> Jens Beißwenger<sup>∗3,5</sup> Anna Manasyan<sup>∗3,4</sup> Daniel Marta<sup>2</sup> Michael J. Black<sup>3</sup> Wieland Brendel<sup>3,5</sup> Andreas Krause<sup>2</sup> Georg Martius<sup>3,4</sup> Martin Riedmiller<sup>1</sup> <sup>1</sup>Vesoma <sup>2</sup>ETH Zürich <sup>3</sup>MPI-IS <sup>4</sup>University of Tuebingen <sup>5</sup>ELLIS Institute viola.is.tue.mpg.de

## ABSTRACT

Teaching a humanoid to follow instructions with its whole body runs into two obstacles. Its action space is large and tightly coupled: legs, arms, and fingers must move together while the robot keeps its balance, which makes joint-level actions hard to learn. And humanoid demonstrations are scarce, so current humanoid generalist policies do not follow new instructions out of the box and are fine-tuned on teleoperated demonstrations of each task before deployment. Human demonstrations exist in far larger numbers, but a person’s motion is not a robot command. We remove both obstacles by changing what the generalist policy predicts. We introduce VioLA, a generalist humanoid policy that predicts body and hand motion latents instead ofjoint commands. A pretrained body- and hand-controller execute these latents on the robot. Their corresponding motion encoders map human and robot motion into the same latent spaces. A human recording is therefore labeled in the policy’s action space, and the training demonstration pool contains 140.6 million frames, 93.2% of them human. As a result, VioLA follows locomotion instructions on the real robot zero-shot, without task-specific fine-tuning, reaching 100% success where GR00T N1.7 and $\Psi _ { 0 }$ reach 16.7% and 0%, respectively. It also reaches 88.6% manipulation success without task-specific fine-tuning. The same approach works across two VLA and one world-action model backbones. A generalist policy trained on human demonstrations alone performs locomotion tasks on the real robot zero-shot. Code and checkpoints will be released.

![](images/4f1109af446bf41b076ceb14fc982a522ef668299babb704c44a336e6f50bb6f.jpg)  
Figure 1: VioLA. Human and robot motion encoders produce targets in shared body and hand latent spaces. The generalist policy predicts motion latents from vision, language, and proprioception; frozen robot controllers execute its predictions. Annotated human motion therefore supervises the policy in the same action representation as robot motion.

## 1 INTRODUCTION

Consider telling a humanoid robot to walk to a table and close a laptop. The robot must find the table, walk to it, reach for the laptop, and close its screen without losing its balance. Visionlanguage-action models (VLAs) and world-action models (WAMs) learn to follow instructions from demonstrations, with substantial progress on robot arms (Zitkovich et al., 2023; Kim et al., 2025; Black et al., 2025a; Ye et al., 2026). What keeps a similar humanoid generalist from following such instructions? Learning these behaviors on a humanoid presents two difficulties: coordinating the whole body and obtaining enough demonstrations to learn from.

The first obstacle is what the policy must predict. Whole-body control requires the legs, torso, arms, and fingers to move together. Reaching shifts the robot’s weight, and stepping changes where its hands can reach. A generalist policy that predicts joint commands must learn how to coordinate these movements while also learning which movement the instruction requires. Whole-body controllers already solve the coordination half. Given a desired motion, a pretrained whole-body controller computes joint commands to follow the motion while maintaining balance (He et al., 2025a;b; Luo et al., 2026). Recent latent motion controllers accept the desired motion as a compact learned vector, or motion latent, produced by an encoder (Peng et al., 2022; Gehring et al., 2023; Yao et al., 2024; Luo et al., 2026).

The second obstacle is data. The robot demonstrations available for learning diverse tasks are limited. Humanoid teleoperation requires a robot and an operator, making it expensive to record many tasks in many environments. Existing humanoid generalists are fine-tuned to target tasks using additional robot demonstrations (Bjorck et al., 2025; Wei et al., 2026; Ma et al., 2026), so deploying them on a new task requires another round of data collection. However, recordings of people walking, reaching and handling objects could expand the demonstrations available for training a humanoid generalist policy. But these recordings contain human motion, not the robot joint commands that the policy would normally learn to predict. Prior work addresses this either by having a robot mirror the person during recording, which again requires a robot for every demonstration (Fu et al., 2025), or by retargeting the human motion to the robot and using the result to supervise the policy (Kareer et al., 2025; Qiu et al., 2025; Zheng et al., 2026b).

Our observation is that the solution to the first obstacle already contains the solution to the second. Recent latent controllers (Luo et al., 2026) map human motion and robot motion in the same latent space. For example, encoding a person walking produces a latent that commands walking on the robot when decoded by the paired controller. Although hand commands are missing in such latent controllers, a separately pretrained hand module with the same structure can cover the hands: human and robot hand encoders share a latent space, and a hand controller turns hand latents into finger commands. A generalist policy can then predict motion latents instead ofjoint commands, choosing where and when to move, while the frozen controllers determine how the joints must move to execute that choice. Since the generalist policy operates on the common latent space between human and robot motions, a human demonstration can supervise the learning, although the recording contains no robot joint commands.

Built on this observation, our method, VioLA uses these motion latents as the action space of a generalist policy (Figure 1). VioLA predicts chunks of body and hand latents from an egocentric image, an instruction, and proprioception. Frozen human and robot motion encoders convert demonstrations into the latents the policy learns to predict, and frozen controller converts its predictions into joint commands. This choice of action space has two benefits:

• Predicting motions, not joints. The generalist policy selects body and hand motions rather than joint commands. The frozen controller maps the robot’s measured posture and selected motion to joint commands, while a frozen hand controller converts selected hand motions into finger commands.

• Human motion as supervision. The human encoders turn recorded body and hand motion into action targets. Human images, instructions, and encoded motion train the same generalist policy as robot demonstrations, without recording a robot performing each human demonstration.

We use SONIC (Luo et al., 2026) for the body encoders and controller, and a separately pretrained a hand module for the hand encoders and controllers following the same structure. All encoders and controllers remain frozen during generalist-policy training. Human recordings retain their original images and instructions. The training demonstration pool contains 140.6 million frames, 93.2% from humans. Section 3 describes how encoders construct targets and the generalist policy predicts them.

On a real Unitree G1, VioLA achieves 100% locomotion success and 88.6% manipulation success without task-specific fine-tuning for either, among closing a laptop, hanging up a coat, and rotating a chair. The approach is not tied to one model, and same recipe is tested on two VLA and one WAM backbone. In addition, VioLA trained exclusively on human demonstrations shows encoded human motion can zero-shot supervise behaviors executed on the robot.

Our contributions. (1) VioLA, a generalist humanoid policy that predicts body and hand motion latents, while pretrained controllers convert those predictions into coordinated joint commands (Section 3). (2) Using human demonstrations for robot learning, converting human recordings and robot demonstrations into latent targets, creating 781 hours of training data, 93% of which is human data. (3) Zero-shot locomotion and manipulation on a real Unitree G1 in a new environment with different object instances, without task-specific fine-tuning (Sections 4.2 and 4.3), and comparisons of VLA and world-action backbones and of human and robot supervision, including real-robot locomotion learned from human demonstrations alone (Sections 4.6 and 4.5).

## 2 RELATED WORK

Generalist robot policies. Generalist policies learn to perform many tasks from images and language instructions by training on diverse robot demonstrations. RT-2 and OpenVLA adapt pretrained vision-language models to predict actions for robot manipulation (Zitkovich et al., 2023; Kim et al., 2025). $\pi _ { 0 }$ combines a vision-language backbone with flow matching to produce continuous actions across single-arm, dual-arm, and mobile manipulators (Black et al., 2025b); $\pi _ { 0 . 5 }$ extends this approach to generalization across environments through training on robot data and semantic supervision (Black et al., 2025a). World-action models (WAMs) add future-video prediction: DreamZero jointly predicts video and actions, while DiT4DiT uses video-model features to condition action prediction (Ye et al., 2026; Ma et al., 2026).

This generalist approach also extends to humanoids. GR00T and $\Psi _ { 0 }$ train on diverse robot demonstrations and use task-specific teleoperation to adapt to target tasks (Bjorck et al., 2025; Wei et al., 2026). $\Psi _ { 0 }$ also pretrains its vision-language model to predict human wrist and hand actions, while its robot joint-command predictions are trained on robot demonstrations. Humanoid policies differ in how they pass their predictions to low-level control. Helix 02 predicts whole-body joint targets that a controller tracks (Figure AI, 2026). Concurrent MotionWAM instead predicts SONIC body latents alongside direct hand or gripper commands (Zheng et al., 2026a). Human videos train MotionWAM’s video-prediction branch, while robot demonstrations supervise its actions.

VioLA builds on VLA and WAM backbones, with learned motion latents as the policy outputs for both the body and hands. A pretrained body controller and hand decoder turn these predictions into joint commands. Their paired motion encoders provide targets for the same outputs from both human and robot demonstrations.

Motion priors and hierarchical control. Motion imitation provides a way to learn joint coordination before learning a task. DeepMimic trains controllers to track reference movements (Peng et al., 2018); ASE, latent-space priors, and MoConVQ learn motion representations that higher-level policies can use to control characters (Peng et al., 2022; Gehring et al., 2023; Yao et al., 2024). OmniH2O, HOVER, and SONIC bring motion tracking to humanoid hardware (He et al., 2025a;b; Luo et al., 2026). In particular, SONIC pairs motion encoders with a body controller that executes their latents. LeVERB connects vision and language to motion control by first learning a latent space from rendered kinematic demonstrations, then training a controller to execute those latents (Xue et al., 2025). Its hardware experiments replay latent sequences generated in simulation.

VioLA starts with SONIC’s pretrained body encoders and controller, together with separately pretrained hand encoders and their paired robot decoders. These modules remain frozen during generalist-policy training. The generalist policy learns to predict their body and hand motion latents from images and instructions, and uses the robot’s current camera observations to update its predictions during deployment.

![](images/9ce90a3ef3a32767880d39168ee38cdbaca34efd091c130cd6a750704f120729.jpg)

Figure 2: VioLA overview. Left: frozen motion encoders construct body and hand action targets from human or robot demonstrations. Center: the generalist policy predicts motion latents from an egocentric image, an instruction, and proprioception. It is the only component updated during training. Right: a frozen body controller and hand decoder convert the predicted latents into robot joint commands. Human and robot encoders produce targets for these same decoders.

Learning from human demonstrations. Human demonstrations can supply robot training data through physical shadowing, as in HumanPlus (Fu et al., 2025), or through reconstruction and simulation: VideoMimic reconstructs human motion and scene geometry from videos, then trains humanoid controllers to reproduce the movements (Allshire et al., 2025). Other methods use recorded human motion to supervise policies directly. EgoMimic shares pose supervision between humans and robots while retaining a robot joint-action head (Kareer et al., 2025). HAT and EgoVLA predict human-centric wrist and hand motion, which is converted to robot commands at deployment (Qiu et al., 2025; Yang et al., 2025). EgoScale pretrains on human wrist and retargeted finger motion before adapting with human–robot and task-specific robot data (Zheng et al., 2026b).

Another line of work learns action labels for policy pretraining. Being-H0 pretrains a visionlanguage policy to predict tokenized human wrist and finger motion, then adds a robot action head during post-training (Luo et al., 2025). LAPA and WholeBodyVLA infer latent actions from video and use robot demonstrations to learn robot-command prediction (Ye et al., 2025; Jiang et al., 2026). UniT learns shared human–humanoid tokens through visual and action reconstruction (Chen et al., 2026). Predicting these tokens aligns policy features, while a separate prediction head produces robot-specific commands; the shared tokens themselves are not passed to a robot controller for exe cution.

VioLA uses annotated human body and hand motion to label the original human images with the same latent commands that its frozen robot decoders execute. Human demonstrations therefore supervise the generalist policy’s deployed action outputs without a corresponding physical robot demonstration. Because the motion encoders and robot decoders are pretrained together, generalistpolicy training does not need to learn a new mapping from human motion labels to robot joint commands.

## 3 VIOLA

VioLA learns which body and hand motions an instruction requires while pretrained decoders produce the robot’s joint commands. The key choice is to use the shared latent spaces of paired motion encoders and decoders as VioLA’s action space (Figure 2). Human and robot encoders map demonstrated movements into this space, supplying the targets the generalist policy learns to predict. All motion encoders and decoders are pretrained and remain frozen; training updates only the generalist policy.

## 3.1 A SHARED LATENT ACTION SPACE

To close a laptop, the robot must reach for the lid and push it down while remaining balanced. A body motion encoder summarizes a short segment of recorded movement as a vector, called a motion latent. Its paired decoder is a pretrained whole-body controller: given the latent and the robot’s recent state, it produces commands for the legs, torso, and arms to follow the encoded movement while maintaining balance. A separate hand encoder summarizes finger motion, and its paired decoder converts the hand latent into finger commands. VioLA learns to predict these body and hand latents from the scene and instruction. For the laptop task, it predicts the standing, reaching, and hand motions; the pretrained body controller and hand decoder determine the joint commands that execute them.

We use the pretrained body encoders and controller from SONIC (Luo et al., 2026). Its human and robot encoders map movements into the same body latent space. Because the body controller does not command the fingers, we combine it with a hand encoder–decoder module that we pretrained separately (Appendix A.2). Its human and robot hand encoders likewise map motion into a common space. Each encoder summarizes a short motion window with a 64-coordinate latent. We concatenate the body and hand latents to define one action in VioLA’s 128-coordinate action space, $\mathbf z _ { t } = [ \mathbf z _ { t } ^ { \mathrm { b o d y } } ; \mathbf z _ { t } ^ { \mathrm { h a n d } } ]$

VioLA predicts the body and hand latents together so that reaching and finger motion can be timed to one another. Given an egocentric image $I _ { t } ,$ an instruction $\ell ,$ and robot proprioception $\mathbf { s } _ { t }$ , the policy $p _ { \theta }$ predicts a chunk of H successive actions:

$$
\widehat { \mathbf { Z } } _ { t } \sim p _ { \theta } ( \mathbf { Z } _ { t } \mid I _ { t } , \ell , \mathbf { s } _ { t } ) , \qquad \mathbf { Z } _ { t } = [ \mathbf { z } _ { t } , \dots , \mathbf { z } _ { t + H - 1 } ] ^ { \mathsf { T } } .\tag{1}
$$

Each row supplies a body and hand latent for one control step; Appendix A.1 gives the reference windows summarized by each latent. Because the paired encoders accept both human and robot motion, demonstrations from either source can supervise these predictions.

## 3.2 LEARNING FROM HUMAN AND ROBOT DEMONSTRATIONS

An annotated recording of a person closing a laptop contains the person’s images, the instruction, and the body and hand movements used to complete it. Although it contains no robot joint commands, the human motion encoders map those movements into VioLA’s action space. Robot demonstrations enter the same space through the corresponding robot encoders. For demonstrator $e \in$ {human, robot}, the target at time t is

$$
\begin{array} { r } { \mathbf { z } _ { t } = \big [ E _ { \mathrm { b o d y } } ^ { e } ( \mathbf { m } _ { t } ^ { \mathrm { b o d y } } ; \gamma _ { t } ) ; E _ { \mathrm { h a n d } } ^ { e } ( \mathbf { m } _ { t } ^ { \mathrm { h a n d } } ) \big ] . } \end{array}\tag{2}
$$

Here $\mathbf { m } _ { t } ^ { \mathrm { b o d y } }$ and $\mathbf { m } _ { t } ^ { \mathrm { h a n d } }$ are short windows of annotated motion beginning at $t ,$ and $\gamma _ { t }$ is the robot heading used to express the desired body motion. Consecutive targets form the chunk $\mathbf { Z } _ { t }$

Human data still lack one required policy input: robot proprioception. We obtain it by tracking the demonstrated motion with the frozen body controller in simulation. Body targets depend on the executing robot’s current heading, so they are computed as the controller tracks the recording in simulation. The rollout also supplies robot states aligned with the human frames. Body targets for robot demonstrations use the same simulation procedure, but their policy inputs retain the recorded robot proprioception. Hand targets are encoded directly from annotated hand motion. All targets are computed before policy training; Appendix A.1 describes the labeling procedure.

Human and robot demonstrations now have the same form: an image, an instruction, robot proprioception, and a target motion chunk. In the laptop example, the person’s original camera view and instruction, paired with the simulated robot state, teach VioLA to predict the whole-body and hand motion. No corresponding physical robot demonstration is required. Demonstrations without hand annotations supervise only the body predictions.

Because both sources provide targets in the same action space, they train the same prediction head with the same loss. VioLA builds on the GR00T N1.7 (NVIDIA, 2026) backbone, adapting its action expert to predict 128 latent coordinates per step while retaining its flow-matching objective (Lipman et al., 2023). The encoders output quantized codes, but the action expert learns to predict their coordinates continuously.

## 3.3 FROM PREDICTED MOTIONS TO CONTINUOUS EXECUTION

The paired decoders make the learned action space directly usable on the robot. At deployment, VioLA predicts a motion chunk from the robot’s camera image, instruction, and proprioception. At

each control step, the body controller converts the current body latent and recent proprioceptive history $\mathbf { h } _ { t }$ into joint commands, while the hand decoder converts the hand latent into finger commands:

$$
{ \bf u } _ { t } ^ { \mathrm { b o d y } } = D _ { \mathrm { b o d y } } \left( \widehat { \bf z } _ { t } ^ { \mathrm { b o d y } } , { \bf h } _ { t } \right) , \qquad { \bf u } _ { t } ^ { \mathrm { h a n d } } = D _ { \mathrm { h a n d } } ^ { \mathrm { r o b o t } } \left( \widehat { \bf z } _ { t } ^ { \mathrm { h a n d } } \right) .\tag{3}
$$

Body and hand commands are updated at 50 Hz. The body controller reads the robot’s changing state at every update, so joint coordination continues while the generalist policy computes its next chunk. The hand decoder reconstructs finger motion from each predicted hand latent; Appendix A.5 gives frame selection and smoothing details.

Continuous execution also requires consecutive chunks to agree where one replaces the other. While VioLA processes a new observation, the robot is already executing part of the previous prediction. An independently generated replacement can disagree with that ongoing motion and produce a jump at the switch. We therefore apply real-time chunking (RTC) (Black et al., 2025c) in VioLA’s motion latent space: the next prediction is conditioned on the latents scheduled to execute before inference finishes. This gives the new chunk the motion it must continue. Section 4.7 measures the resulting continuity across chunk switches.

## 4 EXPERIMENTS

Can VioLA follow locomotion and manipulation instructions on a real humanoid without taskspecific fine-tuning? We first evaluate task completion, then examine the motion predictions behind these behaviors. The remaining experiments test what human demonstrations contribute, which backbone learns humanoid behavior most effectively, and whether successive predictions produce continuous motion.

## 4.1 SETUP

We evaluate on a Unitree G1 with five-fingered Inspire hands in our laboratory. The six locomotion and posture tasks use instructions from LeVERB; the seven manipulation tasks use instructions from Humanoid Everyday. Both corpora also contribute training demonstrations, so the evaluation tasks overlap the training task families. What is held out is the specific episodes, our laboratory environment, and the object instances; a reserved split of Humanoid Everyday is excluded from training (Appendix A.1). Here, zero-shot means deployment without task-specific fine-tuning. For the main comparison, each policy is tested five times per task. Success requires completing the instruction; progress also credits partial completion. We give each task equal weight and report one standard error over trials. Appendix A.6 describes starting poses, time limits, and scoring.

The training pool combines human recordings from EgoSuite, derived from EgoStandard (Lightwheel AI, 2026), with four robot corpora: Humanoid Everyday (Zhao et al., 2026), PSI (Wei et al., 2026), UnifoLM-WBT (Unitree Robotics, 2026), and LeVERB (Xue et al., 2025). The converted pool contains 140.6 million frames, 93.2% from human demonstrations. VioLA uses GR00T N1.7 as its backbone and trains for 150k steps on human demonstrations, followed by 50k steps with uniform sampling across all five corpora. Appendix A.1 gives the corpus sizes and conversion details.

We compare VioLA with the released GR00T N1.7 (NVIDIA, 2026) and $\Psi _ { 0 }$ (Wei et al., 2026) models to assess instruction following without task-specific adaptation. Human-only and robot-only variants of VioLA examine the contribution of human supervision. We also compare three VioLA variants built on GR00T N1.7, $\pi _ { 0 . 5 }$ (Black et al., 2025a), and DiT4DiT (Ma et al., 2026) to assess how effectively two VLAs and one WAM learn humanoid behavior with the same training data and budget. Appendix A.4 gives the training objectives and treatment of missing annotations.

## 4.2 ZERO-SHOT LOCOMOTION

The first test is whether the policy selects the body movement specified by the instruction. The six tasks span holding a posture, raising an arm, bending, walking, sitting, and standing up from a chair. VioLA predicts the body and hand motions for each instruction. The frozen body controller produces the coordinated body commands, while the hand decoder produces the finger commands.

VioLA succeeds in all 30 trials, completing each task five times (Figure 3, top). The released GR00T N1.7 succeeds only at standing still, reaching 5/30 successes (16.7%); $\Psi _ { 0 }$ completes none (0/30).

![](images/fe28aea44c3d49ec97c008643c85355012b01856349940dbc609a753716cc6de.jpg)  
Figure 3: One policy follows locomotion and manipulation instructions without task-specific fine-tuning. VioLA completes 30/30 locomotion trials (top) and 31/35 manipulation trials (bottom). Released GR00T N1.7 succeeds only at standing still; Ψ succeeds only at chair rotation. Bars show success over five trials per task, with one standard error. Photos illustrate the tasks.

Thus, VioLA follows all six locomotion and posture instructions in our laboratory without taskspecific fine-tuning. The Bend Down sequence in Figure 4 illustrates a full-body change: the robot lowers from standing into a crouched posture and holds it.

## 4.3 ZERO-SHOT MANIPULATION

Manipulation adds a further requirement: the body must bring the hand to the object, and the fingers must act at the right time to move, grasp, or release it. We test whether the same policy can coordinate these movements through task completion, again without task-specific fine-tuning.

VioLA completes 31/35 trials (88.6%; Figure 3, bottom). It succeeds in every trial of Rotate Chair, Close Laptop, Place Dumpling, Close Kettle, and Push Duck, and in 4/5 Hang Coat and 2/5 Store Shampoo trials. Hang Coat requires receiving a coat and folding it over the other arm. These tasks cover pushing and closing objects as well as receiving, transporting, and placing them. Released GR00T N1.7 completes none of the 35 trials. $\Psi _ { 0 }$ completes three chair-rotation trials and no other manipulation task, reaching 3/35 successes (8.6%).

Figure 4 shows how successful manipulation develops over time. To close the laptop, the robot brings its right hand to the lid and pushes it shut. To place the dumpling, it takes the toy from a person’s hand with its left hand, carries it over the desk, and releases it. The accompanying latent traces show the body and hand predictions evolving across successive action chunks.

![](images/d5b5ac73ca08637d28d2ddc2477551eb6c54df9f972d2e2923af470851d667bd.jpg)  
Figure 4: From body motion to object interaction. VioLA bends down, closes a laptop, and places a dumpling toy on the desk (first three rows). In a failed coat execution, VioLA grasps the coat but then drops it (last row; failure marked in red). Black and gold curves show executed body and hand latents projected onto their first principal components (63–91% and 47–87% of their variance). Projections are fitted per recording and scaled separately within each row. Gray bands mark successive action chunks; triangles align image timestamps with the curves.

The four unsuccessful trials show difficulties in acquiring and retaining objects. In the Store Shampoo failures, the robot knocks the bottle over while reaching to grasp it. The last row of Figure 4 shows a failed coat execution: the robot acquires the coat but drops it before completing the folding and return instruction. These examples show why reaching the object or making an initial grasp is insufficient for manipulation success. The separate bottle-carrying task, analyzed in Appendix A.7, remains unsolved: every VioLA variant fails all five trials, at different stages spanning grasp timing, the transition from grasping to walking, and upright release. Including this task lowers VioLA’s manipulation success from 31/35 (88.6%) to 31/40 (77.5%).

## 4.4 PREDICTING MOTIONS, NOT JOINTS

How many distinct motion tokens does VioLA use to perform a task? We examine its body and hand predictions during standing, arm raising, walking, chair rotation, and laptop closing. Nearby latent predictions can produce nearly identical movements, so we group similar predictions and count each group as one distinct motion token. Appendix A.10 explains the grouping procedure and its calibration against decoded motion. Body and hand tokens are counted separately over each complete execution.

Complex movements can be represented with a few motion tokens. A complete chair-rotation execution uses an average of 60 distinct body tokens and 31 hand tokens; laptop closing uses 82 body tokens and 43 hand tokens (Figure 5). These totals cover the entire executions, which last 10.0 and 9.3 s on average. Standing and arm raising require fewer body tokens, about 20 and 36, while walking requires 182. The hand counts also reflect the movement: arm raising uses one hand token, whereas chair rotation and laptop closing require multiple hand motions.

![](images/4bce12a8ed8e31b0c60668e90c759b5388fcc4758c469a288d10c204cec171b8.jpg)  
Figure 5: Complex movements have compact descriptions in motion tokens. Each dot is the number of distinct body (black) or hand (gold) motion tokens over one complete recording, after grouping similar predictions as described in Appendix A.10. Horizontal marks show the mean and vertical whiskers one standard error across five recordings per task. Numbers below give mean counts and durations. Chair rotation and laptop closing require tens of distinct body and hand token over roughly ten seconds of execution.

These counts make the change in the prediction problem concrete. Each body token describes coordinated movement that the frozen controller turns into joint commands using the robot’s changing state; each hand token specifies finger motion through the hand decoder. Complex executions therefore require a small set of distinct motion predictions, while the controllers continue to update the joints at every control step. The generalist policy learns which motions to produce and how to time them, reusing the coordination encoded by the pretrained motion modules.

## 4.5 LEARNING FROM HUMAN DEMONSTRATIONS

VioLA’s motion action space also makes human recordings a source of action supervision. Does that supervision teach behaviors that transfer to a robot, and what does it contribute when robot demonstrations are available? We first train a variant of VioLA using only human demonstrations, then compare the human-then-mixed and robot-only variants. The motion encoders and decoder remain frozen in all variants.

The human-only VioLA variant succeeds in 12/30 locomotion trials (40%): 4/5 bends, 3/5 walks, 2/5 sits, and 3/5 stands from a chair. It does not successfully follow the standing-still or arm-raising instructions. Figure 6 also shows it walking toward a yellow couch and squatting down. These behaviors transfer without any robot demonstration in generalist-policy training. Encoded human motion therefore provides usable action supervision for real-robot behavior, although human-only training does not yet cover all six instructions reliably.

With robot demonstrations available, we compare two VioLA variants after 200k training steps: VioLA uses the human-then-mixed schedule from Section 4.1, while the robot-only policy uses the four robot corpora throughout. VioLA completes 30/30 locomotion trials, compared with 24/30 (80%) for robot-only training. The difference comes from bending, where robot-only training fails all five trials, and one sitting trial in which it leaves the arms extended. Human-then-mixed training thus improves locomotion success by 20 percentage points at the same number of training updates.

The manipulation comparison shows a different pattern. Robot-only training completes 33/35 trials (94.3%), compared with 31/35 (88.6%) for VioLA. Robot-only training succeeds in all five sham poo trials but four kettle trials; VioLA succeeds in two shampoo trials and all five kettle trials. Their success counts match on the other five tasks. Human-then-mixed training therefore improves locomotion by six successful trials but completes two fewer manipulation trials. Its benefit depends on the task: it teaches body movements missing from the robot-only policy, while the robot-only variant is more successful at grasping the shampoo bottle.

"Walk to the yellow couch"  
![](images/ebd2e443427cf5ebcee879951bf97a2525adf20aaf190d758080f51379b04069.jpg)

![](images/c1be5d2e6dce8493cc47fc9c2e7a0f7a18ccd09804492349b1f0a0a608f8f98e.jpg)

Figure 6: Human demonstrations teach robot behavior. Left: the human-only policy follows “walk to the yellow couch” and “squat down.” Right: locomotion success with human-then-mixed training (30/30 trials), robot-only training (24/30), and human-only training (12/30). All three are VioLA variants with the same GR00T backbone and frozen motion modules. Error bars show one standard error over trials. The robot-only and human-then-mixed variants each receive 200k training updates.  
![](images/2066be736c2fd06dd37bd1911babbf6d10af52d9e0832556a3e93d82faa4503a.jpg)  
Figure 7: VioLA with two VLA backbones and one WAM backbone. The variants use the same training data and budget and are evaluated without task-specific fine-tuning. Bars report success and progress over 30 locomotion trials, 35 manipulation trials, or all 65. VioLA reaches 100% locomotion success with either GR00T or DiT4DiT, while GR00T performs best on manipulation. The final panel shows measured prediction delay. Error bars show one standard error over trials for performance and over recorded episodes for delay.

## 4.6 COMPARING POLICY BACKBONES

Which backbone learns humanoid behavior most effectively with VioLA? We compare two VLAs, GR00T N1.7 and $\pi _ { 0 . 5 } .$ , with one WAM, DiT4DiT. All three VioLA variants use the same training data and training budget, predict 64 body and 64 hand coordinates per action, and execute their predictions with the same frozen motion modules. This provides a direct comparison of VLA and WAM backbones for humanoid robot learning under a common training and evaluation protocol. Each backbone retains its learning objective; DiT4DiT also predicts future video. Table 2 gives the state inputs and chunk lengths.

VioLA with either GR00T or DiT4DiT succeeds in all 30 locomotion trials (Figure 7). The $\pi _ { 0 . 5 }$ variant completes 20/30 trials (66.7%): it succeeds at standing, arm raising, sitting, and standing up, but fails walking and full bending. Manipulation separates the integrations more strongly. VioLA completes 31/35 trials (88.6%), compared with 16/35 (45.7%) for $\pi _ { 0 . 5 }$ and 9/35 (25.7%) for DiT4DiT. VioLA therefore learns instruction following with both VLA and WAM backbones. GR00T gives the strongest overall results: it matches DiT4DiT on locomotion and completes substantially more manipulation trials than either alternative.

![](images/25a6e17dda9252ebb2e4dc2ef12be89c26eca620342077d50208222318daf0c5.jpg)  
Figure 8: RTC reduces jumps when a new motion chunk takes over. Each row compares VioLA recordings with RTC on (gold) and off (gray). Curves show the L2 change in the 64-coordinate body latent between consecutive 20 ms control steps. Vertical lines mark chunk switches in the corresponding condition’s color. RTC-off traces show sharp spikes near these switches, whereas RTC-on traces remain close to their between-switch variation. Each recording is aligned to the start of policy execution and shown for 5.1 s; the initial idle-to-policy transition is excluded.

The behavior explains the gap between successful body motion and successful manipulation. All three VioLA variants rotate the chair in every trial. The $\pi _ { 0 . 5 }$ variant also closes the laptop and places the dumpling in every trial, but it drops the coat after grasping it in four trials. DiT4DiT fails to close its thumb around the coat, and a partial dumpling trial ends in a grasp followed by a drop. Such partial completions raise manipulation progress above success, to 51.4% for $\pi _ { 0 . 5 }$ and 29.4% for DiT4DiT. The frozen body controller and hand decoder are shared across the three variants. The differences arise in the generalist policy’s predictions: successful manipulation requires the right reach, grasp, and release sequence through the end of the task.

For VioLA, prediction delay is 160.2 ms with GR00T, 208.6 ms with $\pi _ { 0 . 5 } ,$ and 248.5 ms with DiT4DiT. These measurements span the recorded deployments summarized in Appendix A.8. At 50 Hz, each prediction takes roughly eight to twelve control steps, during which the robot continues to move. This delay makes the transition from the ongoing motion to the next prediction an essentia part of execution.

## 4.7 CONTINUITY ACROSS MOTION CHUNKS

Does replanning preserve the motion already in progress? While a new chunk is being predicted, the robot executes part of the previous chunk. The two predictions can disagree at the point where the new one takes over. We test whether real-time chunking (RTC) (Black et al., 2025c), which conditions the next prediction on the latents scheduled for execution during inference, reduces this disagreement in motion space.

We compare VioLA with RTC enabled and disabled on Stand Still and Raise Arm. The first tests consistency while holding a posture; the second tests continuity while the requested posture changes. Figure 8 measures the L2 change in the body latent between consecutive 20 ms control steps, $\Delta _ { t } =$ $\| \mathbf { z } _ { t } ^ { \mathrm { { b o d y } } } - \mathbf { z } _ { t - 1 } ^ { \mathrm { { b o d y } } } \| _ { 2 }$ . Vertical markers show when execution switches to a new chunk. A spike at a marker indicates an abrupt change in the predicted body latent when the chunk is replaced.

Without RTC, sharp spikes recur near chunk switches, even for Stand Still. With RTC, the changes at these switches stay close to the variation between switches. Over the displayed 5.1 s windows, the largest step change decreases from 0.332 to 0.064 for Stand Still and from 0.455 to 0.070 for Raise Arm. RTC therefore reduces discontinuities in the predicted body latents when consecutive chunks are joined in these recordings. Appendix A.9 gives the measurement and timing details.

## 5 LIMITATIONS AND CONCLUSION

Limitations. VioLA reuses pretrained body and hand modules, so its capabilities depend on the movements those modules can execute. The hand decoder produces finger-position commands without contact-force feedback. The manipulation trials show remaining difficulties in reaching to grasp objects and retaining them through task completion. Incorporating contact feedback into hand control is one direction for improving object interaction.

Learning from human recordings requires human body and hand motion annotations. Extending VioLA to additional video therefore depends on obtaining reliable estimates of that motion. Human supervision also does not improve every behavior under the current training schedule: human-then mixed training improves locomotion, but robot-only training achieves higher manipulation success. How to retain the locomotion benefit while improving object interaction remains an open question.

The main evaluation covers thirteen tasks on one Unitree G1, with five trials per task. Broader testing across tasks, environments, and robot embodiments is needed to assess how reliably VioLA performs beyond this evaluation.

Conclusion. VioLA is a generalist humanoid policy that learns from human and robot demonstrations to predict body and hand movements. Pretrained motion encoders convert the demonstrations into latent targets; VioLA predicts those latents from images, instructions, and proprioception. A frozen body controller turns the body predictions into coordinated joint commands while maintaining balance, and a frozen hand decoder produces the finger commands. Human recordings can therefore teach VioLA which movements an instruction requires without a robot performing each demonstration.

On a real Unitree G1, VioLA achieves 100% locomotion success and 88.6% manipulation success without task-specific fine-tuning. Its complex movements are represented with a few motion tokens. Comparing two VLA backbones and one WAM backbone using the same training data and budget shows that VioLA performs best with GR00T, particularly for manipulation. Training the generalist policy on human demonstrations alone still yields 40% locomotion success on the real robot.

## ACKNOWLEDGMENTS

Mert Albaba conducted this work during his internship at Vesoma. The authors thank our colleagues at Vesoma for their collaboration and for the many thoughtful discussions that shaped this work. Daniel Marta was supported by the Knut and Alice Wallenberg Foundation.

Jens Beißwenger was supported by the Federal Ministry of Research, Technology and Space of Germany (BMFTR, formerly BMBF) under grant no. 01IS24085C (OPENHAFM).

Georg Martius is a member of the Machine Learning Cluster of Excellence, EXC number 2064/1 – Project number 390727645. This work was supported by the German Federal Ministry of Education and Research (BMBF): Tübingen AI Center, FKZ: 01IS18039A. This work was supported by the ERC - 101045454 REAL-RL.

## REFERENCES

Arthur Allshire, Hongsuk Choi, Junyi Zhang, David McAllister, Anthony Zhang, Chung Min Kim, Trevor Darrell, Pieter Abbeel, Jitendra Malik, and Angjoo Kanazawa. Visual imitation enables contextual humanoid control. In Proceedings of the 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pp. 794–815, 2025. URL https://proceedings.mlr.press/v305/allshire25a.html.

Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. GR00T N1: An open foundation model for generalist humanoid robots. arXiv preprint arXiv:2503.14734, 2025. URL https://arxiv.org/abs/2503.14734.

Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π : A vision-language-action model with open-world generalization. In Proceedings ofthe 9th Confer ence on Robot Learning, volume 305 of Proceedings ofMachine Learning Research, pp. 17–40, 2025a. URL https://proceedings.mlr.press/v305/black25a.html.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Robert Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Lucy Xiaoyang Shi, Laura Smith, James Tanner, Quan Vuong, Anna Walling, Haohuan Wang, and Ury Zhilinsky. π : A vision-language-action flow model for general robot control. In Proceedings of Robotics: Science and Systems, 2025b. doi: 10.15607/RSS.2025.XXI.010. URL https://www.roboticsproceedings.org/rss21/p010.html.

Kevin Black, Manuel Y. Galliker, and Sergey Levine. Real-time execution of action chunking flow policies. In Advances in Neural Information Processing Systems, volume 38, 2025c. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html.

Boyu Chen, Yi Chen, Lu Qiu, Jerry Bai, Yuying Ge, and Yixiao Ge. UniT: Toward a unified physical language for human-to-humanoid policy learning and world modeling. arXiv preprint arXiv:2604.19734, 2026. URL https://arxiv.org/abs/2604.19734.

Figure AI. Introducing Helix 02: Full-body autonomy. Project website, January 2026. URL https://www.figure.ai/news/helix-02. Published January 27, 2026.

Zipeng Fu, Qingqing Zhao, Qi Wu, Gordon Wetzstein, and Chelsea Finn. HumanPlus: Humanoid shadowing and imitation from humans. In Proceedings of the 8th Conference on Robot Learning, volume 270 of Proceedings ofMachine Learning Research, pp. 2828–2844, 2025. URL https: //proceedings.mlr.press/v270/fu25a.html.

Jonas Gehring, Deepak Gopinath, Jungdam Won, Andreas Krause, Gabriel Synnaeve, and Nicolas Usunier. Leveraging demonstrations with latent space priors. Transactions on Machine Learning Research, 2023. URL https://openreview.net/forum?id=OzGIu4T4Cz.

Tairan He, Zhengyi Luo, Xialin He, Wenli Xiao, Chong Zhang, Weinan Zhang, Kris M. Kitani, Changliu Liu, and Guanya Shi. OmniH2O: Universal and dexterous human-to-humanoid wholebody teleoperation and learning. In Proceedings ofthe 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pp. 1516–1540, 2025a. URL https:// proceedings.mlr.press/v270/he25b.html.

Tairan He, Wenli Xiao, Toru Lin, Zhengyi Luo, Zhenjia Xu, Zhenyu Jiang, Jan Kautz, Changliu Liu, Guanya Shi, Xiaolong Wang, Linxi Fan, and Yuke Zhu. HOVER: Versatile neural wholebody controller for humanoid robots. In IEEE International Conference on Robotics and Automation, 2025b. URL https://research.nvidia.com/labs/lpr/publication/ he2025hover/.

Haoran Jiang, Jin Chen, Qingwen Bu, Li Chen, Modi Shi, Yanjie Zhang, Delong Li, Chuanzhe Suo, Chuang Wang, Zhihui Peng, and Hongyang Li. WholeBodyVLA: Towards unified latent VLA for whole-body loco-manipulation control. In International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/abs/2512.11047.

Simar Kareer, Dhruv Patel, Ryan Punamiya, Pranay Mathur, Shuo Cheng, Chen Wang, Judy Hoffman, and Danfei Xu. EgoMimic: Scaling imitation learning via egocentric video. In IEEE International Conference on Robotics and Automation, 2025. URL https://egomimic. github.io/.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P. Foster, Pannag R. Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Open-VLA: An open-source vision-language-action model. In Proceedings of the 8th Conference on Robot Learning, volume 270 of Proceedings of Machine Learning Research, pp. 2679–2713, 2025. URL https://proceedings.mlr.press/v270/kim25c.html.

Lightwheel AI. EgoStandard. Dataset, 2026. URL https://huggingface.co/datasets/ LightwheelAI/EgoStandard. Accessed September 10, 2026.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. SMPL: A skinned multi-person linear model. ACM Transactions on Graphics, 34(6):248:1– 248:16, 2015. URL https://smpl.is.tue.mpg.de/.

Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, and Zongqing Lu. Being-H0: Vision-language-action pretraining from largescale human videos. arXiv preprint arXiv:2507.15597, 2025. URL https://arxiv.org/ abs/2507.15597.

Zhengyi Luo, Ye Yuan, Tingwu Wang, Chenran Li, Fernando Castañeda, Sirui Chen, Zi-Ang Cao, Jiefeng Li, David Minor, Qingwei Ben, Jinhyung Park, David Sami, Zi Wang, Xingye Da, Runyu Ding, Cyrus Hogg, Lina Song, Edy Lim, Eugene Jeong, Tairan He, Haoru Xue, Wenli Xiao, Simon Yuen, Jan Kautz, Yan Chang, Umar Iqbal, Linxi Fan, and Yuke Zhu. SONIC: Supersizing motion tracking for natural humanoid whole-body control. Science Robotics, 11(117):eaed4592, 2026. doi: 10.1126/scirobotics.aed4592. URL https://arxiv.org/abs/2511.07820.

Teli Ma, Jia Zheng, Zifan Wang, Chunli Jiang, Andy Cui, Junwei Liang, and Shuo Yang. DiT4DiT: Jointly modeling video dynamics and actions for generalizable robot control. arXiv preprint arXiv:2603.10448, 2026. URL https://arxiv.org/abs/2603.10448.

Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: VQ-VAE made simple. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html.

NVIDIA. GR00T-N1.7-3B. Model card and pretrained checkpoint, 2026. URL https: //huggingface.co/nvidia/GR00T-N1.7-3B. Accessed September 11, 2026.

Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel van de Panne. DeepMimic: Exampleguided deep reinforcement learning of physics-based character skills. ACM Transactions on Graphics, 37(4):143:1–143:14, 2018. doi: 10.1145/3197517.3201311. URL https:// xbpeng.github.io/projects/DeepMimic/index.html.

Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, and Sanja Fidler. ASE: Large-scale reusable adversarial skill embeddings for physically simulated characters. ACM Transactions on Graphics, 41(4), 2022. URL https://xbpeng.github.io/projects/ASE/index. html.

Ri-Zhao Qiu, Shiqi Yang, Xuxin Cheng, Chaitanya Chawla, Jialong Li, Tairan He, Ge Yan, David J. Yoon, Ryan Hoque, Lars Paulsen, Ge Yang, Jian Zhang, Sha Yi, Guanya Shi, and Xiaolong Wang. Humanoid policy ∼ human policy. In Proceedings of the 9th Conference on Robot Learning, volume 305 of Proceedings ofMachine Learning Research, pp. 2888–2906, 2025. URL https: //proceedings.mlr.press/v305/qiu25a.html.

Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: Modeling and capturing hands and bodies together. ACM Transactions on Graphics, 36(6):245:1–245:17, 2017. URL https://mano.is.tue.mpg.de/.

Unitree Robotics. UnifoLM-WBT dataset. Dataset collection, 2026. URL https:// huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset. Accessed September 11, 2026.

Songlin Wei, Hongyi Jing, Boqian Li, Zhenyu Zhao, Jiageng Mao, Zhenhao Ni, Sicheng He, Jie Liu, Xiawei Liu, Kaidi Kang, Sheng Zang, Weiduo Yuan, Marco Pavone, Di Huang, and Yue Wang. $\Psi _ { 0 } { : }$ An open foundation model towards universal humanoid loco-manipulation. arXiv preprint arXiv:2603.12263, 2026. URL https://arxiv.org/abs/2603.12263.

Haoru Xue, Xiaoyu Huang, Dantong Niu, Qiayuan Liao, Thomas Kragerud, Jan Tommy Gravdahl, Xue Bin Peng, Guanya Shi, Trevor Darrell, Koushil Sreenath, and Shankar Sastry. Le-VERB: Humanoid whole-body control with latent vision-language instruction. arXiv preprint arXiv:2506.13751, 2025. URL https://arxiv.org/abs/2506.13751.

Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Xuxin Cheng, Ri-Zhao Qiu, Hongxu Yin, Sifei Liu, Song Han, Yao Lu, and Xiaolong Wang. EgoVLA: Learning vision-language-action models from egocentric human videos. arXiv preprint arXiv:2507.12440, 2025. URL https://arxiv.org/abs/2507.12440.

Heyuan Yao, Zhenhua Song, Yuyang Zhou, Tenglong Ao, Baoquan Chen, and Libin Liu. Mo-ConVQ: Unified physics-based motion control via scalable discrete representations. ACM Transactions on Graphics, 43(4):144:1–144:21, 2024. doi: 10.1145/3658137. URL https: //doi.org/10.1145/3658137.

Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, and Minjoon Seo. Latent action pretraining from videos. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 45d74e190008c7bff2845ffc8e3facd3-Paper-Conference.pdf.

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, Ayaan Malik, Kyungmin Lee, William Liang, Nadun Ranawaka, Jiasheng Gu, Yinzhen Xu, Guanzhi Wang, Fengyuan Hu, Avnish Narayan, Johan Bjorck, Jing Wang, Gwanghyun Kim, Dantong Niu, Ruijie Zheng, Yuqi Xie, Jimmy Wu, Qi Wang, Ryan Julian, Danfei Xu, Yilun Du, Yevgen Chebotar, Scott Reed, Jan Kautz, Yuke Zhu, Linxi Fan, and Joel Jang. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026. URL https://arxiv.org/abs/2602.15922.

Zhenyu Zhao, Hongyi Jing, Xiawei Liu, Jiageng Mao, Abha Jha, Hanwen Yang, Rong Xue, Sergey Zakharov, Vitor Guizilini, and Yue Wang. Humanoid everyday: A comprehensive robotic dataset for open-world humanoid manipulation. In IEEE International Conference on Robotics and Automation, 2026. URL https://arxiv.org/abs/2510.08807.

Jia Zheng, Teli Ma, Yudong Fan, Zifan Wang, Shuo Yang, and Junwei Liang. MotionWAM: Towards foundation world action models for real-time humanoid loco-manipulation. arXiv preprint arXiv:2606.09215, 2026a. URL https://arxiv.org/abs/2606.09215.

Ruijie Zheng, Dantong Niu, Yuqi Xie, Jing Wang, Mengda Xu, Yunfan Jiang, Fernando Castañeda, Fengyuan Hu, You Liang Tan, Letian Fu, Trevor Darrell, Furong Huang, Yuke Zhu, Danfei Xu, and Linxi Fan. EgoScale: Scaling dexterous manipulation with diverse egocentric human data. arXiv preprint arXiv:2602.16710, 2026b. URL https://arxiv.org/abs/2602.16710.

Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, Quan Vuong, Vincent Vanhoucke, Huong Tran, Radu Soricut, Anikait Singh, Jaspiar Singh, Pierre Sermanet, Pannag R. Sanketi, Grecia Salazar, Michael S. Ryoo, Krista Reymann, Kanishka Rao, Karl Pertsch, Igor Mordatch, Henryk Michalewski, Yao Lu, Sergey Levine, Lisa Lee, Tsang-Wei Edward Lee, Isabel Leal, Yuheng Kuang, Dmitry Kalashnikov, Ryan Julian, Nikhil J. Joshi, Alex Irpan, Brian Ichter, Jasmine Hsu, Alexander Herzog, Karol Hausman, Keerthana Gopalakrishnan, Chuyuan Fu, Pete Florence, Chelsea Finn, Kumar Avinava Dubey, Danny Driess, Tianli Ding, Krzysztof Marcin Choromanski, Xi Chen,

Yevgen Chebotar, Justice Carbajal, Noah Brown, Anthony Brohan, Montserrat Gonzalez Arenas, and Kehang Han. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Proceedings of the 7th Conference on Robot Learning, volume 229 of Proceedings ofMachine Learning Research, pp. 2165–2183, 2023. URL https://proceedings.mlr. press/v229/zitkovich23a.html.

## VIOLA: LEARNING GENERALIST HUMANOIDCONTROL POLICIES FROM HUMAN DATA

SUPPLEMENTARY MATERIAL

## A IMPLEMENTATION AND ADDITIONAL ANALYSES

## A.1 MOTION REPRESENTATION AND DATA CONVERSION

Token coordinates and reference windows. Each encoder produces a 64-coordinate motion code, formed by flattening two 32-dimensional finite scalar quantization (FSQ) codes (Mentzer et al., 2024). Coordinates take values in $\{ k / 1 6 : k = - 1 6 , \ldots , 1 5 \}$ . Body and hand codes use this same format, but have separate encoders and decoders. A code summarizes a reference window beginning at the current frame. At 50 Hz, the SONIC v1.1 robot encoder and the hand encoders read ten frames at offsets $0 , 5 , \ldots , 4 5 ;$ the human body encoder reads ten consecutive frames at offsets $0 , 1 , \ldots , 9$ Windows extending beyond the demonstration repeat its final frame. This reference window determines what one code represents; the policy’s chunk length $H$ determines how many successive codes it predicts.

Motion targets. Equation 2 concatenates the body and hand encoder outputs. In this implementation, $\mathbf { z } _ { t } \in \mathcal { G } ^ { 1 2 8 }$ with $\mathcal { G } = \{ k / 1 6 : k = - 1 6 , \ldots , 1 5 \}$ . The human body encoder reads SMPL motion (Loper et al., 2015); human hand motion is obtained by fitting MANO (Romero et al., 2017) to annotated keypoints. Each training example pairs the chunk with an image, instruction, and robot state:

$$
\left( I _ { t } , \ \ell , \ \mathbf { s } _ { t } , \ \mathbf { Z } _ { t } \right) , \qquad \mathbf { Z } _ { t } = \left[ \mathbf { z } _ { t } , \ldots , \mathbf { z } _ { t + H - 1 } \right] ^ { \mathsf { T } } ,\tag{4}
$$

Robot examples use recorded proprioception, while human examples use simulated robot states. The body targets for both sources are constructed through the simulation procedure below. This converts annotated human motion into latent targets without optimizing a separate mapping to robot joints for each recording.

Closed-loop construction of body targets. Body targets for both human and robot demonstrations are generated by tracking the reference motion with the frozen controller in simulation. The body encoder expresses the desired motion relative to the executing robot’s heading, which can differ from the demonstrator’s heading. For example, if the robot turns toward a table more slowly than the person in the recording, the next latent must encode the turn that remains. Using the person’s heading throughout would assume that the robot follows the reference exactly. We therefore compute the targets in closed loop. At each simulation step, the encoder uses the simulated robot’s current heading to compute a code, the controller decodes it into joint commands, and the simulator advances:

$$
\begin{array} { r } { \mathbf { z } _ { t } ^ { \mathrm { b o d y } } = E _ { \mathrm { b o d y } } ^ { e } \left( \mathbf { m } _ { t } ^ { \mathrm { b o d y } } ; \gamma ( \mathbf { s } _ { t } ^ { \mathrm { s i m } } ) \right) , \qquad \mathbf { s } _ { t + 1 } ^ { \mathrm { s i m } } = \mathrm { S i m } \left( \mathbf { s } _ { t } ^ { \mathrm { s i m } } , D _ { \mathrm { b o d y } } ( \mathbf { z } _ { t } ^ { \mathrm { b o d y } } , \mathbf { h } _ { t } ^ { \mathrm { s i m } } ) \right) . } \end{array}\tag{5}
$$

At time $t ,$ the image $I _ { t } ,$ simulated state $\mathbf { s } _ { t } ^ { \mathrm { { s i m } } }$ , and encoded target $\mathbf { z } _ { t } ^ { \mathrm { b o d y } }$ are paired before the simulator advances to $\mathbf { s } _ { t + 1 } ^ { \mathrm { s i m } }$ . Human examples use this simulated state as body proprioception; robot examples retain their recorded proprioception.

Hand-state inputs use the values stored in the data. For robot demonstrations, hand targets come from commanded finger motion where available, while hand proprioception comes from measured motion. This distinction matters during grasping: the commanded fingers can close farther than the object permits them to move. Examples without hand annotations contribute only to the body loss (Equation 7). Motion targets are computed before generalist-policy training; training does not run the simulator or update either motion module.

Body-controller configuration. SONIC v1.1 uses a 1,751-coordinate encoder input containing the reference motion in the simulated robot’s heading frame. The body decoder receives the current 64-coordinate code and ten frames of proprioceptive history: joint offsets, joint velocities, base angular velocity, projected gravity, and previous actions. Together these form 994 input coordinates. The decoder’s 29 outputs are converted to joint-position targets using the controller’s default pose and action scale. Control runs at 50 Hz, with four 200 Hz physics steps per update.

<table><tr><td>Source</td><td>Episodes</td><td>Frames</td><td>Hours</td><td>Targets</td></tr><tr><td>Humanoid Everyday</td><td>4,064</td><td>2,962,764</td><td>16.46</td><td>Body and hands</td></tr><tr><td>PSI</td><td>835</td><td>1,431,454</td><td>7.95</td><td>Body and hands</td></tr><tr><td>UnifoLM-WBT</td><td>2,854</td><td>4,454,240</td><td>24.75</td><td>Body and hands</td></tr><tr><td>LeVERB</td><td>3,555</td><td>713,279</td><td>3.96</td><td>Body only</td></tr><tr><td>EgoSuite</td><td>20,680</td><td>131,010,575</td><td>727.84</td><td>Body and hands</td></tr><tr><td>Total</td><td>31,988</td><td>140,572,312</td><td>780.96</td><td></td></tr></table>

Table 1: Converted demonstration pool. Humanoid Everyday, PSI, and UnifoLM-WBT contain real-G1 demonstrations; LeVERB contains simulated G1 demonstrations, and EgoSuite contains human recordings. All are converted to 50 Hz, with hand targets where available. Hours are frames divided by 50 Hz. Human data are 64.6% of episodes and 93.2% of frames. Sampling weights, not corpus size, determine exposure to each source during training.

Available converted data. The training pool combines human recordings from EgoSuite, derived from EgoStandard (Lightwheel AI, 2026), with four robot corpora: Humanoid Everyday (Zhao et al., 2026), PSI (Wei et al., 2026), UnifoLM-WBT (Unitree Robotics, 2026), and LeVERB (Xue et al., 2025). Table 1 lists the retained demonstrations before the training/evaluation split. The converted human pool contains 20,680 of the 26,382 source episodes; 5,702 are excluded during conversion and filtering. The unified loader reserves 338 Humanoid Everyday episodes for evaluation. It also excludes episodes too short to supply a complete action chunk; at $H = 5 0 .$ , 46 of the 3,555 LeVERB episodes provide no training window. Humanoid Everyday and UnifoLM-WBT supply commanded hand motion, whereas PSI supplies measured hand motion because commanded trajectories are unavailable. Human hand targets are derived from fitted MANO trajectories. LeVERB supplies body targets only.

VioLA trains for 150k steps on human demonstrations, followed by 50k steps with sampling probability 0.20 for each of the five corpora (Figure 9). The robot-only variants use the four robot corpora throughout and are evaluated after 50k and 200k steps. The human-only variant uses human demonstrations throughout. No variant receives task-specific adaptation. With a constant batch size across stages, the VioLA schedule assigns an expected 80% of training examples to human data, distinct from their 93.2% share of the available frames.

## A.2 HAND MOTION INTERFACE

We pretrain a paired hand encoder–decoder module that maps human hand motion and robot hand motion into a shared 64-coordinate space. Its Dex3 and Inspire decoders translate these codes into finger trajectories for the corresponding hardware. All hand encoders and decoders remain frozen during generalist-policy training. The policy learns to predict their codes; it does not learn the hand prior itself.

## A.3 BACKBONE AND ACTION-EXPERT INSTANTIATIONS

The GR00T, $\pi _ { 0 . 5 } ,$ and DiT4DiT variants of VioLA use the same training data and training budget. Each predicts 128 coordinates per action step: 64 for the body and 64 for the hands. VioLA with GR00T predicts H = 40 successive steps; the other variants use H = 50 (Table 2). State inputs differ by backbone. The $\pi _ { 0 . 5 }$ configuration uses 29 body joint offsets, three gravity coordinates, 14 current finger positions expressed in the Dex3 representation, and a 64-coordinate hand code delayed by 50 ticks. The one-second delay prevents the code’s reference window from exposing future targets. During training, this delayed code is zeroed with probability 0.5; it can also be zeroed during deployment. Current finger state is reconstructed from measured hand motion, using frame zero of the hand decoder when conversion is needed.

GR00T N1.7 (NVIDIA, 2026) combines a Cosmos-Reason2 vision-language backbone with a flowmatching action transformer. Its state encoder receives 29 body joint offsets, 14 finger positions, and three gravity coordinates, with state dropout of 0.2. Human and robot examples use the same action expert and embodiment tag. All methods predict native FSQ coordinates without action-target normalization. State inputs are normalized with statistics fitted on training data. $\pi _ { 0 . 5 }$ discretizes the normalized values into 256 bins for its input prompt, whereas GR00T and DiT4DiT project them into learned features.

![](images/47bdb1733619db36c0cbd6eea6936d2393a642145ef3645adabfea0ce2f46155.jpg)  
Figure 9: Dataset size and VioLA training schedule. The first two bars show each source’s share of episodes and frames in the converted pool. The last two show sampling probabilities during 150k steps of human-only training and 50k steps of uniform sampling over all five corpora. Human recordings supply 93.2% of available frames; the training schedule determines how often they are sampled.

DiT4DiT uses Cosmos-Predict2.5-2B, with features from layer 17 conditioning the action transformer. During training, the video model receives supervision from the current frame and four future frames at offsets 12, 24, 36, 48 on the 50 Hz timeline. The objective adds video and motion flowmatching losses, and gradients from the action loss also update the video backbone. At deployment, only the current observation is available; future video features are predicted by the model.

## A.4 FLOW-MATCHING CONVENTIONS AND MASKS

The action expert $v _ { \psi }$ is conditioned on backbone features and robot state, collectively denoted $\mathbf { c } _ { t }$ Flow matching mixes each target with noise and trains the expert to predict the difference. Under the convention below, inference starts from noise at $\tau = 1$ and integrates the learned velocity backward in time toward a clean action chunk at $\tau = 0$ . Encoder codes are treated as continuous coordinates in this regression.

For a target chunk $\mathbf { Z } _ { t } .$ , noise $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ of the same shape, and an interpolation time $\tau ,$

$$
{ \bf X } _ { t , \tau } = ( 1 - \tau ) { \bf Z } _ { t } + \tau \epsilon , \qquad { \bf U } _ { t } = \epsilon - { \bf Z } _ { t } ,\tag{6}
$$

and the action expert $v _ { \psi }$ is trained to output $\mathbf { U } _ { t }$ from $\mathbf { X } _ { t , \tau }$ :

$$
\mathcal { L } _ { \mathrm { m o t i o n } } = \mathbb { E } _ { \epsilon , \tau } \left[ \sum _ { n \in \mathcal { B } } \sum _ { k = 0 } ^ { H - 1 } \sum _ { j = 1 } ^ { 1 2 8 } w _ { n k j } \left( v _ { \psi } ( \mathbf { X } _ { t , \tau } , \tau \mid \mathbf { c } _ { t } ) _ { n , k j } - U _ { n , k j } \right) ^ { 2 } \right] ,\tag{7}
$$

where n indexes the examples in a batch $B ,$ each with its own target chunk, noise, and conditioning, $k$ the step within the chunk, and $j$ the latent coordinate. The weight $w _ { n k j }$ normalizes the loss and is zero for coordinates without a target, such as the hand coordinates of a recording without hand annotations. The reductions below specify each backbone’s normalization. Human and robot examples pass through the same action expert and the same loss; no explicit human–robot source label is supplied to the action expert. The motion encoders and decoders remain frozen.

DiT4DiT is a world-action model: it is trained to predict the future video frames $\mathbf { V } _ { t }$ that follow the current image, as well as the actions. We keep that objective. Its video branch uses the same squared velocity-prediction principle, with its own video representation and loss reduction. The video and motion losses are summed:

$$
{ \mathcal { L } } _ { \mathrm { W A M } } = { \mathcal { L } } _ { \mathrm { v i d e o } } ( \mathbf { V } _ { t } \mid I _ { t } , \ell ) + { \mathcal { L } } _ { \mathrm { m o t i o n } } ( \mathbf { Z } _ { t } \mid I _ { t } , \ell , \mathbf { s } _ { t } ) .\tag{8}
$$

For a human clip, the recorded future frames supervise video prediction, and the encoded body and hand motion supervises action prediction.

The loss excludes invalid frames and missing hand annotations. Let $N$ denote batch size and $m _ { n k j } \in$ {0, 1} indicate whether coordinate $j$ at chunk step $k$ in example n has a valid target. For $\pi _ { 0 . 5 } .$

<table><tr><td>Backbone family</td><td>H</td><td>State conditioning</td><td>Motion-target scaling</td></tr><tr><td>π0.5</td><td>50</td><td>110 values, tokenized</td><td>Identity</td></tr><tr><td>GR00T N1.7</td><td>40</td><td>46 values, state encoder</td><td>Identity</td></tr><tr><td>DiT4DiT</td><td>50</td><td>32 values, state encoder</td><td>Identity</td></tr></table>

Table 2: Backbone-specific interfaces for VioLA. All variants predict the same 128 body and hand coordinates without target normalization. Chunk length and state conditioning follow each backbone.

Equation 7 uses

$$
w _ { n k j } = \frac { m _ { n k j } } { N H \operatorname* { m a x } ( 1 , \sum _ { j ^ { \prime } } m _ { n k j ^ { \prime } } ) } .\tag{9}
$$

Thus each step first averages over its annotated coordinates. GR00T and DiT4DiT instead average over all valid coordinates in the batch:

$$
w _ { n k j } = \frac { m _ { n k j } } { \sum _ { n ^ { \prime } , k ^ { \prime } , j ^ { \prime } } m _ { n ^ { \prime } k ^ { \prime } j ^ { \prime } } } .\tag{10}
$$

GR00T adds $1 0 ^ { - 6 }$ to the denominator, and DiT4DiT requires at least one valid coordinate. For fully annotated batches, the reductions are equivalent up to the numerical offset. With missing hand annotations, the first gives equal weight to action steps, while the second gives equal weight to annotated coordinates.

Equation 6 uses $\tau = 1$ for noise and $\tau = 0$ for clean targets. The $\pi _ { 0 . 5 }$ implementation samples $\tau = 0 . 9 9 9 b + 0 . 0 0 1$ , with b ∼ Beta(1.5, 1). DiT4DiT samples $\tau = b / 0$ .999 with the same beta parameters and discretizes time into 1,000 conditioning buckets. GR00T uses the opposite time direction: its implementation samples $r = 0 . 9 9 9 ( 1 - b )$ , forms $( 1 - r ) \epsilon + r \mathbf { Z } _ { t }$ , and regresses $\mathbf { Z } _ { t } - \epsilon$ Setting $\tau = 1 - r$ and reversing the velocity sign gives the convention of Equation 6. Each backbone retains its own time sampler and numerical solver settings.

## A.5 DECODER EXECUTION

The body controller updates its proprioceptive history and joint commands every 20 ms, while the generalist policy computes its next chunk. The hand decoder reconstructs ten frames from each hand latent. Execution selects frame index five (zero-based), applies a five-tick moving average, and converts the result to the hand’s joint commands. Frame zero is used separately to reconstruct current hand state.

The Unitree G1 uses five-fingered Inspire hands, with six actuated joints per hand. Predicted latents pass through the Inspire decoder to produce 12 finger targets. For policies trained with a 14-value Dex3 hand state, measured Inspire motion is encoded and then reconstructed with the Dex3 decoder. This conversion changes the state representation presented to the policy; the execution targets still use the Inspire decoder.

Although encoder targets lie on the FSQ grid, the action experts predict continuous coordinates. The $\pi _ { 0 . 5 }$ execution path clips predictions to [−1, 1] and passes them directly to the decoders without rounding. The hand decoder is selected for the robot’s hardware before converting its output to finger commands.

Chunk replacement and RTC. While the decoders execute a chunk, the generalist policy uses a new image and robot state to predict its replacement. If the predictions disagree where the new chunk takes over, the commanded motion can change abruptly. RTC operates on motion latents, conditioning the next prediction on the part of the current chunk that will execute before inference finishes. Keeping these latents fixed helps the new chunk continue the current motion. Section 4.7 compares transitions with and without RTC.

## A.6 EVALUATION PROTOCOL

Instructions follow the task descriptions in LeVERB and Humanoid Everyday. All tasks are evaluated in our laboratory with locally available objects rather than the physical objects in the datasets.

Bottle hand-over to the table. VioLA: 0/5 successes

![](images/2e9b7a0f8dd981939ccd43149bd4c397627f4ea4b1264d1ad375c259ffb800fe.jpg)  
Figure 10: Bottle carrying fails at release. One VioLA trial from a third-person camera. The robot takes the bottle from a person, walks to the table, and opens its hand while the bottle is tilted; the bottle then falls off the table. Times are seconds from the start of the clip. The task scores 0/5 and is reported separately from the seven manipulation tasks in Figure 3.

Each policy is tested five times per task. Trials begin from an idle standing pose, except Stand Up, which begins seated on a chair. Locomotion trials allow 5 s, except Bend Down, which allows 10 s. Manipulation trials have no fixed time limit. For VioLA, a frozen SONIC v1.1 controller and a hand decoder execute motion predictions at 50 Hz.

Success requires completing the instruction. Locomotion trials test standing, raising an arm, bending, walking forward, remaining seated, or rising from a chair. Standing up and then walking still counts as success; sitting and immediately standing again does not. Manipulation trials require rotating the chair, folding and returning the coat, closing the laptop or kettle lid, placing the dumpling on the desk, putting the shampoo in the container, or pushing the duck to the desk’s center. Progress assigns intermediate scores to incomplete executions. The robot-only policy receives 0.5 for sitting with its arms still extended instead of lowering them. A DiT4DiT kettle trial receives 0.8 after slowly closing the lid but continuing the closing motion without returning the arm to idle. Thus, the manipulation scores also account for how the robot concludes the action, despite the absence of a fixed timeout. Only a score of one counts as success. We average scores over five trials per task, giving each task equal weight; error bars show one standard error.

## A.7 FAILURES IN GRASPING AND OBJECT TRANSPORT

Bottle carrying. Bottle carrying requires the robot to take a bottle from a person with its left hand, walk to a table, and place it upright. VioLA fails all five trials. It approaches the table but releases the bottle while tilted, causing it to fall. The $\pi _ { 0 . 5 }$ and DiT4DiT variants of VioLA, and the robotonly GR00T variant trained for 200k steps, also fail all five trials. This task is analyzed separately from the seven manipulation tasks in the main figure; including it reduces VioLA’s manipulation success from 31/35 (88.6%) to 31/40 (77.5%).

The VioLA variants fail at different stages. The robot-only variant grasps the bottle but does not start walking toward the table. The $\pi _ { 0 . 5 }$ variant repeatedly reaches toward it but rarely closes its hand. The DiT4DiT variant closes its hand before reaching the bottle. Together, these failures show that grasp timing, the transition from grasping to walking, and stable release remain difficult.

Other manipulation failures. VioLA also fails three of five shampoo trials by knocking the bottle over during the reach, and one of five coat trials. In the failed coat execution shown in Figure 4, the robot acquires the coat but drops it before completing the instruction. These cases distinguish reaching the object from maintaining control of it through placement or handover.

## A.8 EXECUTION TIMING

Timing is measured over 161 episodes of VioLA with GR00T, 83 with $\pi _ { 0 . 5 } ,$ , and 57 with DiT4DiT, including recordings beyond the five scored trials per task. For each episode, prediction delay is the mean time from requesting a chunk to receiving it. The replanning interval is the mean time

<table><tr><td>Backbone</td><td>Episodes</td><td>Chunks</td><td>Delay (ms)</td><td>Replan interval (ms)</td></tr><tr><td>GR0OT</td><td>161</td><td>3,871</td><td> $\overline { { 1 6 0 . 2 \pm 1 . 8 } }$ </td><td> $\overline { { 5 5 9 . 9 \pm 1 . 8 } }$ </td></tr><tr><td>π0.5</td><td>83</td><td>2,554</td><td> $2 0 8 . 6 \pm 0 . 6$ </td><td> $6 0 8 . 4 \pm 0 . 6$ </td></tr><tr><td>DiT4DiT</td><td>57</td><td>1,035</td><td> $2 4 8 . 5 \pm 2 . 5$ </td><td> $6 4 8 . 0 \pm 2 . 4$ </td></tr></table>

Table 3: Measured chunk timing. Mean ± one standard error across recorded episodes. Delay measures the time from requesting to receiving a prediction; the replanning interval measures time between successive chunk arrivals. The episode counts exceed the scored-trial counts because the recording archive includes additional runs.

between consecutive chunk arrivals, measured from their frame indices at 50 Hz. Table 3 averages these episode-level measurements, giving each episode equal weight.

## A.9 MEASURING CHUNK-TRANSITION CONTINUITY

For each task and RTC condition, Figure 8 uses the first recording in chronological order with at least four seconds of policy execution. All four traces are cropped to 5.1 s, the shortest selected duration rounded down to a tenth of a second. This selection does not use latent-change magnitude. The metric is computed directly from the logged, continuous body latents, without grid rounding or projection. Consecutive steps are identified by recording timestamps separated by 20 ms, with a 1 ms tolerance. Differences across timestamp gaps are excluded; the four selected recordings contain no such gaps. Repeated or skipped policy-counter values do not exclude a sample when its recording timestamp remains consecutive. The first policy command is excluded because its preceding command belongs to the idle pose token.

Each chunk-switch marker indicates when execution begins using a new prediction. It is placed at the timestamp of the first recorded sample whose policy counter reaches the new chunk’s logged takeover index. Elapsed time is measured relative to the first policy sample.

The selected recordings run at 50 Hz. Mean prediction delays are 138.9 ms with RTC and 154.1 ms without RTC for Stand Still, and 130.0 ms and 155.3 ms for Raise Arm. Corresponding mean intervals between chunk switches are approximately 537 and 552 ms for Stand Still, and 529 and 553 ms for Raise Arm, measured from the timestamp-aligned takeover markers. Thus the traces compare recorded executions with slightly different timing, rather than imposing identical inference delays. The reported maxima describe these four examples, not averages over repeated trials.

## A.10 MOTION TOKEN COUNTS AND CALIBRATION

Counting distinct motion tokens. We count the distinct representatives needed to describe the policy’s body and hand predictions over a task execution. Each 64-coordinate latent z is mapped to integer grid coordinates solely to measure grouping distance:

$$
\mathbf { q } = \mathrm { c l i p } \big ( \mathrm { r o u n d } ( 1 6 \mathbf { z } ) , - 1 6 , 1 5 \big ) , \qquad d ( \mathbf { q } , \mathbf { q } ^ { \prime } ) = \sum _ { j = 1 } ^ { 6 4 } | q _ { j } - q _ { j } ^ { \prime } | .\tag{11}
$$

We scan each recording in time order, retaining the first prediction. A subsequent prediction is assigned to its nearest retained representative if the distance is at most the threshold $\rho ;$ otherwise, it becomes a new representative. Ties select the earliest representative. Representatives are original predicted vectors, not averages or grid-rounded vectors. Even $\rho = 0$ can group unequal continuous predictions that fall in the same grid cell. Body and hand predictions are grouped separately, with a new set of representatives for each execution. We report counts per execution, with duration for context, and average counts across executions of each task.

Decoded-pose calibration. Calibration uses 84 episodes: 32 chair-rotation and 32 laptop-closing demonstrations, plus 10 policy recordings each of Place Dumpling and Push Duck. The demonstration latents are encoder targets; the policy latents are continuous predictions. A further 21 episodes are held out for validation, with 8, 8, 3, and 2 episodes from the respective tasks. We test body thresholds in {0, 1, 2, 4, 8, 16, 32, 64} and extend the hand grid with {96, 128, 192, 256, 384, 512}. At each threshold, we group the latents and compare the decoded poses of each original latent and its representative at the same robot state. Body decoder outputs are converted to joint-position targets, and forward kinematics gives link positions for the comparison. Body error is the mean root-relative position difference over the 14 links used by SONIC’s MPJPE-L metric. Hand error is the mean position difference over six Dex3 distal-finger link centers of mass, used as fingertip proxies, with the decoder evaluated at frame zero. Each episode’s error is the 95th percentile of its framewise mean point errors. Invalid frames and frames at or after a simulated fall are excluded.

Tracking-error references. We use tracking accuracy to set a physical reference scale for the grouping error. For the body, the reference is SONIC’s reported real-world MPJPE-L of 25.7 mm (2.57 cm) over 14 links (Luo et al., 2026). For the hands, we measure the discrepancy between commanded and observed finger joint poses in 83 Humanoid Everyday episodes (chair rotation, laptop closing, and chair-alt). Forward kinematics gives a mean error of 1.44 cm over six fingertip proxies and 32,516 frames. This hand reference uses the same points as the grouping analysis. These values are tracking-error reference budgets, not task-success tolerances or bounds on the effect of replacing a complete motion sequence.

Threshold selection. For each modality, we find the largest tested threshold for which at least 95% of calibration episodes have their episode error within the tracking-error reference budget. This rule permits thresholds of 16 for the body and 96 for the hands; requiring every calibration episode to pass gives the same values. Episodes receive equal weight, and none are removed as outliers.

We report finer thresholds of 8 for the body and 32 for the hands because averaging over links or fingertips can hide larger deviations at individual points. Reducing the body threshold from 16 to 8 lowers the mean episode-level wrist p95 from 5.09 to 3.03 cm on calibration episodes. Reducing the hand threshold from 96 to 32 lowers the worst-finger p95 from 1.08 to 0.31 cm for chair rotation and from 1.25 to 0.31 cm for laptop closing. At these finer thresholds, both modalities satisfy their mean-error budgets in all 84 calibration episodes, all 21 held-out episodes, and three additional chair-alt transfer episodes. Across the calibration and held-out episodes, the largest episode-level 95th-percentile mean error is 1.234 cm for body links and 0.294 cm for fingertip proxies. Figure 5 reports counts over five recordings per task at these thresholds.

Figures 11 and 12 report the error curves and sensitivity of counts to the grouping threshold. The robot executes the original continuous predictions; the analysis describes them at a resolution referenced to tracking accuracy. It measures local differences in decoded pose, not accumulated execution error. Hand calibration uses Dex3 fingertip proxies rather than the deployed Inspire hand trajectory.

![](images/e2afbdb2264867ecfafac1a8841e275a7dd7385663bab9eb5ff208fbf388a89d.jpg)

Figure 11: Threshold calibration. Each episode is summarized by the 95th percentile of framewise error, averaged over 14 body links or six fingertip proxies. The gray line shows the median of the calibration episodes’ errors; shading spans their minimum and maximum. Gold dots show held-out episodes; triangles mark values above the displayed range. Red horizontal lines mark the trackingerror references (2.57 cm body, 1.44 cm hand). Vertical dashed lines mark the reported thresholds of 8 and 32, chosen finer than the permitted 16 and 96 to reduce wrist and individual-finger deviations.  
![](images/51375d8675476d5fc9f17164066e300d7372b8920ecef3ced6050dc51a3dbe8a.jpg)  
Figure 12: Token counts vs. threshold. Curves show how grouping distance changes the number of distinct body and hand representatives. Each recording’s count is divided by its duration before averaging within each task; the vertical axis is logarithmic. Colored curves identify the five tasks in Figure 5; gray curves show seven additional tasks. Dashed lines mark the reported body and hand thresholds of 8 and 32.