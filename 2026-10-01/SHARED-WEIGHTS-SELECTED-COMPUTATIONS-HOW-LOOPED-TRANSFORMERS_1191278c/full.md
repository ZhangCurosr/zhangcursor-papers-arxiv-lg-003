# SHARED WEIGHTS, SELECTED COMPUTATIONS:HOW LOOPED TRANSFORMERS ROUTEWHAT EACH LOOP DOES

Jiaju Wu<sup>1,2</sup> Yi Hu<sup>1</sup> Muhan Zhang<sup>1</sup>

## ABSTRACT

Looped Transformers repeatedly apply the same set of Transformer layers, giving them a recurrent architecture for latent computation. Their strong performance on iterative reasoning and length-generalization tasks suggests an appealing explanation: recurrence may provide an inductive bias that lets the model reuse a learned algorithm across loops. However, weight sharing alone does not imply that every loop performs the same operation. This raises a basic question: is each loop actually repeating the same computation, and if not, what routes the shared parameters to different operations? We study this question using graph walks as a test case. In the model’s native trajectories, decoded predictions can advance by different numbers of graph steps or remain at a reached target, showing that recurrent progress need not follow a fixed one-loop-one-step pattern. We then show that a frozen loop can be steered toward different transitions by modifying its entering hidden state: a learned linear layer J selects the desired transition without changing the shared Transformer layers. To test how this steering works, we use activation patching and find that attention patterns can recover its effects and switch the selected transition. Across five matched pairs of graph models, changing intermediate supervision during backbone training changes which transitions J can induce. This suggests that J selects computations learned by the backbone rather than creating new algorithms. Together, these results show that the hidden state can control shared computation, with attention routing as a causal pathway.

Code and reproduction materials: https://github.com/wjjpku/howloop

## 1 INTRODUCTION

Recent work on latent reasoning studies how models can carry out multi-step reasoning within their hidden states, without generating every intermediate step as text (Saunshi et al., 2025; Geiping et al., 2025). Recurrent architectures provide a natural way to support this kind of computation: the same parameters can be applied repeatedly while the internal state continues to evolve. Looped Transformers bring this idea to Transformer models (Vaswani et al., 2017) by reusing the same set of Transformer layers over multiple loops (Dehghani et al., 2019; Yang et al., 2024). This recurrent design provides a useful inductive bias for iterative computation. Looped Transformers can learn iterative learning algorithms and multi-step optimization procedures (Yang et al., 2024; Gatmiry et al., 2024), improve length generalization on tasks with iterative solutions (Fan et al., 2025), and use additional loops as a source of latent test-time computation at language-model scale (Geiping et al., 2025; Zhu et al., 2025). These results suggest that recurrent depth can provide a way to reuse learned computation over multiple internal steps.

These results naturally suggest an iterative view of Looped Transformers: the shared layers may repeatedly apply the same learned procedure as computation progresses. For example, in a graph walk, one might expect every loop to move one step along the path. Yet weight sharing does not imply such a fixed correspondence. The parameters are shared across loops, but the hidden state entering those parameters changes as computation proceeds. The same network is therefore applied to different internal states, so weight sharing alone does not specify what operation each loop performs. The training objective can also leave the intermediate trajectory underdetermined. When only the final answer is supervised, the loss does not require the model to follow any particular sequence of intermediate steps. This leads to a basic question: is each loop actually repeating the same computation, and ifnot, what routes the shared parameters to different operations? Answering this question is important for understanding the inductive bias of recurrent depth and the internal process of latent reasoning. It also affects when additional loops provide useful computation and whether recurrent test-time computation can be controlled.

Prior work has gradually moved from asking whether recurrence can implement algorithms to studying how learned computation is organized across loops. Constructive results show that fixedweight Looped Transformers can execute multi-step programs, while learning-based analyses show that looped models can learn iterative optimization procedures (Giannou et al., 2023; Yang et al., 2024; Gatmiry et al., 2024). These results establish that shared parameters can support reusable algorithmic computation, but they do not specify how that computation is distributed across loops after training. Mechanistic work therefore turns to the learned recurrent trajectory itself. Blayney et al. (2026) identify layer-wise fixed points, cyclic latent trajectories, and repeated stages of inference across loops. Probing studies likewise find that intermediate recurrent states do not always form a simple, directly readable latent chain of thought (Lu et al., 2025). More recently, Zhang et al. (2026) study task progress across loops directly. They show that training can select computation frontiers with different speeds, so the amount of progress associated with one loop is learned rather than fixed by weight sharing. Together, these studies characterize how recurrent computation is learned and organized across loops, but leave open whether the state entering a loop can causally select what the same frozen layers compute next, and how this selection is implemented inside the model.

We address these gaps by testing whether the entering hidden state can redirect the next computation of a frozen loop, and by tracing this control to attention routing (Figure 1). We begin with the recurrent trajectories that arise without intervention. In graph-walk tasks, native readouts can advance at different numbers of graph steps or remain at an already reached target. A learned trajectory therefore need not follow a fixed “one loop = one algorithmic step” rule (Section 3). This variation makes the entering hidden state a natural candidate for controlling the next computation. We freeze the shared Transformer layers and learn a linear layer at the loop boundary. Changing only the entering state selects one-hop or two-hop targets for the same frozen loop (Section 4). Having established this control, we ask how it is implemented inside the loop. Patching attention patterns recovers much of the steering effect in the graph case study and the Ouro multi-hop language task. Patching patterns between one-hop and two-hop runs also switches the semantic target, providing causal evidence that routing carries the choice of transition (Section 5). Finally, we test how backbone training and repeated execution shape this control. Across five matched pairs of graph models, intermediate supervision during backbone training changes which computations J can steer. This supports the view that J controls computations learned by the backbone rather than creating new algorithms (Section 6). The controllers can be composed across two successive loops, but accuracy is order-dependent and drops with longer compositions (Section 4.2).

![](images/61b120622d7d3db8ef04861d5e584ea9ccdb85b17cf2f47f819facb7328e0169.jpg)  
Figure 1: Overview of looped computation and state steering. Phenomenon: intermediate readouts reveal different patterns of progress across weight-shared loops. Control: a learned linear layer changes the state entering the next loop to select a different transition while the backbone remains frozen. Mechanism: patching attention patterns from a steered run into an unsteered run transfers routing while preserving the receiving run’s values, recovering steering effects. Flames mark trainable components; snowflakes mark frozen components.

## 2 RELATED WORK

Recurrent computation. Recurrent architectures increase computation depth by repeatedly applying the same parameters (Dehghani et al., 2019). Prior work shows that looped Transformers can execute programs and learn iterative algorithms (Giannou et al., 2023; Yang et al., 2024; Gatmiry et al., 2024), improve length generalization (Fan et al., 2025), and use additional loops to improve reasoning (Saunshi et al., 2025; Geiping et al., 2025; Zhu et al., 2025). These results establish the value of recurrent computation. They do not, however, determine what operation each loop performs in a trained model.

Computation learned across loops. Understanding loop behavior requires distinguishing the algorithms an architecture can express from the computation selected by training. Training budgets can select different rates of task progress within the same shared architecture (Zhang et al., 2026). Studies of looped language models also identify cyclic latent trajectories and repeated stages of inference (Blayney et al., 2026). Intermediate readouts need not directly reveal this internal computation: their interpretation can depend on the layer and decoding method (Lu et al., 2025). These findings motivate going beyond observing an existing trajectory. We ask whether changing the hidden state entering a loop can make the same parameters perform different operations on the input.

Supervision of recurrent states. Supervision provides a way to examine how learned computation affects state control. Training objectives shape both the recurrent trajectory and the information available at intermediate readouts (Fan et al., 2026; Popescu et al., 2026). Supervising outputs does not fully specify the internal computation; even per-loop supervision can leave state variables such as hidden-state scale uncontrolled (Sharma & Vu, 2026). We therefore compare backbones trained under different supervision schemes to test whether steering depends on computational capabilities already learned by the backbone.

State steering and causal interventions. State interventions and activation patching provide complementary tools for testing this account. Steering modifies hidden states to control a frozen model’s behavior (Turner et al., 2023), including through learned affine maps (Singh et al., 2024). Activation patching replaces internal activations to test which components contribute to a behavior (Wang et al., 2023). We combine these tools by using an affine map to change the loop input and select the next target, then patching attention patterns to test whether routing carries this selection. This extends the study of what recurrence can compute to whether learned computation can be selected through the entering state and how that selection is implemented.

## 3 PHENOMENON: TASK PROGRESS VARIES ACROSS LOOPS

An iterative view of looped Transformers suggests that shared layers may repeatedly apply the same learned algorithm across loops. We examine this view using readouts at intermediate loops on the graph walk task to observe how computation progresses as the model runs.

![](images/c328936b3ef282737aa904e2ee7a51ac9b6b5d85f083566176602f27a8f07337.jpg)  
(a) D8L8

![](images/46d067f80dd6b607d3c0f1f1ce1d4e139a8d40620ebc025b210991e3355d0ec1.jpg)  
(b) D8L8

![](images/58327c468be7c84ca23c25ebde8541d0dee8b6516baa2d710bdf6d71044811fb.jpg)  
(c) D8L6

![](images/18ed37418b2c8c5e0ff6eff7351c6d654a497ab352072a2eb82b907460254ecd.jpg)  
(d) D8L6  
Figure 2: Color shows prediction frequency. Red dashed lines mark the training loop $L = 8$ or $L = 6 ;$ white dots mark the unique most frequent prediction at each loop.

Each input consists of a permutation $f _ { G }$ on a graph $G ,$ , a start node $s ,$ and a requested path depth $d = 8$ . Writing $f _ { G } ( v )$ for the successor of v, the target is $f _ { G } ^ { 8 } ( s )$ . We train two models, D8L8 and D8L6, which repeat a two-layer Transformer block for eight and six loops, respectively. Only the final answer is supervised.

For visualization, we take the trained looped block, repeat it, and apply the final readout head after loops 0–16. We intentionally use ten-node cycles for testing so that the iteration length can be clearly identified from the node index. Each plotted block in Figure 2 summarizes predictions across examples for each individual walk. We show selected representative trajectories; Appendix B provides all twelve D8L8 and five D8L6 trajectories.

As shown in Figure 2, the models can reach the same target through different readout trajectories. For D8L8, one trajectory advances mainly one node per loop, whereas another advances mainly two nodes per loop. For D8L6, the model must fit the advancement process into six loops, thus producing trajectories with irregular speeds. But eventually they stabilize at the final answer, even if additional loops are run. This shows that, within a single model, the algorithm can differ across loops. Some loops advance quickly, others slowly, and all eventually stabilize at the end. Sharing weights does not guarantee the same algorithm.

Before selecting models for mechanistic analysis, we must inspect these readouts. These trajectories reveal which mechanism the model primarily uses. We next turn this observed variation into a control test.

## 4 CONTROL: STATE STEERING REDIRECTS THE NEXT TARGET OF A FROZEN LOOP

The different algorithms observed above motivate us to study why they differ. To this end, we try to control the next transition by modifying the hidden state entering the block. We train a linear layer to control the next loop’s target while keeping the backbone weights frozen. We use graph walks as a test case for steering the computation through the state entering the loop. Additional state-control experiments and supporting evidence on Qwen and synthetic knowledge-graph relation composition are provided in Appendix H.

## 4.1 STATE STEERING WITH A FROZEN BACKBONE

Let $F$ denote the shared Transformer block and $h _ { t }$ the sequence of token states after loop t. Our original model computes $h _ { t + 1 } = F ( h _ { t } )$ . To change the next transition, we apply a linear layer J before the next loop, which acts independently on every token:

$$
h _ { t + 1 } = F ( J ( h _ { t } ) ) .\tag{1}
$$

We train J while keeping the backbone frozen. In the graph experiments, J consists of a learned diagonal term, a rank-48 term, and a bias. The token-wise transformation, linear form, and lowrank parameterization constrain the expressive power of J. In particular, J cannot aggregate graph information across token positions on its own; the frozen block $F$ must execute that part of the transition.

We evaluate the fitted map on out-of-distribution graphs that were unseen during both backbone training and map training. This tests whether the map generalizes to unseen graphs and whether modifying the information already represented in each token can actually redirect computation. Appendix C details the graph setup, controller training, and composition evaluations.

We first test whether two maps can select different next targets from the same state. D8L6 seed 6 processes an eight-hop query for six loops to produce h<sub>6</sub>. Let $u = f _ { G } ^ { 8 } ( s )$ be the correct answer to this query. We train $\bar { J _ { \mathrm { o n e } } } \mathrm { s o }$ that one additional loop $F ( J _ { \mathrm { o n e } } ( h _ { 6 } ) )$ predicts $f _ { G } ( u )$ . We train $J _ { \mathrm { t w o } }$ in the same way, with target $f _ { G } ^ { 2 } ( u )$

Both maps select their intended targets on held-out graphs (Figure 3a). Without steering, one-hop accuracy is 48.2%, but with $J _ { \mathrm { o n e } }$ it rises to 99.6%. Two-hop accuracy reaches 98.2% with $J _ { \mathrm { t w o } }$ Thus, changing the entering state selects different next targets under the same frozen weights and one additional loop.

Stay One hop Two hops Others

![](images/d72dfaab210d31991a9b6b62d8fd42319457f71720a7794a8f972865ed1a1a62.jpg)  
(a) from $h _ { 6 }$

![](images/3b8da76e2ac8dedadc320b37ed3609c6a456212c4488e850cf0221cbeda1c2aa.jpg)  
(b) from $F ( J _ { \mathrm { o n e } } ( h _ { 6 } ) )$

![](images/d297c8f3dc47c37823fc6fe9fa7b940028b1f35229286779d022044f31935673.jpg)  
(c) from $F ( J _ { \mathrm { t w o } } ( h _ { 6 } ) )$  
Figure 3: One-loop answer distributions for D8L6. Each row corresponds to a map applied to the labeled starting state before F. In panels (a), (b), and (c), colors are referenced to u, $f _ { G } ( u )$ , and $f _ { G } ^ { 2 } ( u )$ , respectively. Bars show the average over two map fits.

![](images/984149c1e146f515271b5985ecd2423f50f99fd4080a980924a8a54c110bad40.jpg)  
Figure 4: Controller reuse on 512 new held-out graphs with D8L6 seed 6. The horizontal axis counts additional controlled loops after $h _ { 6 } ;$ exact match compares the predicted node with the cumulative target at each call. Curves average two map fits on the same 3,175 examples; mixed averages 32 fixed sequences. Shading shows 95% graph-bootstrap intervals.

## 4.2 COMPOSING CONTROLLER J

Although each map is trained for a single additional loop, it can continue to control the next target when reused on a state produced by a previous controlled loop. We first evaluate all four two-map orders, then test repeated control for up to eight additional calls.

Applying the one-hop map twice reaches the two-hop target in 93.6% of examples. Omitting the second map reduces accuracy to 48.3%. Applying the one-hop map followed by the two-hop map reaches the three-hop target with 65.3% accuracy. Reversing their order reaches the same target with 74.5% accuracy (Figure 3). Thus, the maps can be reused over two loops, but their order affects accuracy.

We separately test longer reuse on 512 new held-out graphs, retaining 3,175 examples with distinct targets through four graph hops. Figure 4 follows repeated one-hop, repeated two-hop, and mixed maps for eight additional controlled loops after $h _ { 6 }$ . The target advances by the cumulative number of requested hops. We score exact match between the predicted node and this cumulative target at each call. At the third call, repeated one-hop and two-hop maps reach 56.3% and 23.6% accuracy. At the eighth, mixed exact match is 10.1%. Thus, control persists beyond the single-loop training setting but degrades under longer reuse. We next intervene inside the frozen block to study how this control works.

## 5 MECHANISM: ATTENTION ROUTING MEDIATES STATE-DEPENDENTCOMPUTATION

State steering changes the next target, but it does not reveal how the frozen block uses the changed state. We first identify how attention patterns route retrieved content. We then test whether attentionpattern patching produces the same steering effect as the linear map J. Finally, we directly switch the selected transition by exchanging patterns between one-hop and two-hop controlled runs.

To make steering cleaner, we need backbones that use both one-hop and two-hop algorithms. We therefore examine the readout at intermediate loops and select backbones whose trajectories advance at different speeds. For each backbone, we average two map fits, and then average the three backbone results with equal weights.

To test whether the mechanism extends to a large language model, we also evaluate Ouro-2.6B on a multi-hop language task. The model reads sentences describing successive letter transfers and answers who holds the letter after a requested number of transfers. We lightly fine-tune Ouro on 1–4-step requests with four loops. Then we freeze the backbone and train a dense linear layer on 1–8-step requests. The layer is applied before loops 2–4.

The three D8L6 interventions use different sets of activations. The pattern-versus-output experiment patches one selected head in the second layer at the answer position (Figure 5b). The steeringtransfer experiment patches the attention patterns of all four second-layer heads at that same position (Figure 5c). The target-switching experiment patches the patterns of all four second-layer heads at every token position (Figure 6). Ouro uses a fixed set of sixteen selected heads at its fourth loop. Head selection and evaluation use separate data. Appendices C and D provide the full setups, intervention protocols, and results for D8L6 and Ouro, respectively.

## 5.1 PATTERN AND OUTPUT PATCHES SELECT DIFFERENT ANSWERS

We patch activations from run 1 into run 2 to separate attention routing from retrieved content. The inputs have aligned token positions. Writing $\alpha _ { i }$ for attention patterns and $V _ { i }$ for values, the original head output in run 2 is $z _ { \mathrm { r a w } } = \alpha _ { 2 } V _ { 2 }$ . We compare:

$$
\begin{array} { r l } & { z _ { \mathrm { r a w } } \xrightarrow [ ] { \mathrm { p a t t e r n p a t c h } } z _ { \mathrm { p a t t e r n } } = \alpha _ { 1 } V _ { 2 } , } \\ & { z _ { \mathrm { r a w } } \xrightarrow [ ] { \mathrm { o u t p u t p a t c h } } z _ { \mathrm { o u t p u t } } = z _ { 1 } = \alpha _ { 1 } V _ { 1 } . } \end{array}\tag{2}
$$

In D8L6, both runs use $J _ { \mathrm { o n e } } .$ . Suppose run 1 starts the controlled transition at C with C→D, while run 2 starts at A with A→B and C→E. Their original answers are D and B. Patching run 1’s pattern into run 2 should select its C record but read $\mathrm { E , }$ changing B→E. Patching the output should instead transfer D, changing B→D.

![](images/caf8a118dff0c956e7a7f14b78915cd359ed95eb1a098000665e05ca28bd1fb7.jpg)

![](images/5e4c84b698a0c99c920e6947ed02f6e23804ea453101fb056d9a0ca56ee158b5.jpg)

![](images/67f8d4b88f9c57750faea5ce4b2785f9113e4210ddc610f247f07eb0041577fd.jpg)  
Figure 5: Attention routing and state steering. (a) Attention computation. (b) Attention-pattern and output patches between graphs. (c) Attention-pattern patches between steered (J) and unsteered (0) runs. D8L6 bars are averaged over backbone means; markers indicate seeds 6 (circle), 10 (diamond), and 13 (triangle). Ouro error bars show 95% Wilson intervals.

For Ouro, the rule sets share seven transfers from $s _ { 1 } \colon s _ { 1 } \stackrel { 7 } { \to } \mathbf { C } \to \mathrm { D }$ in run 1 and $s _ { 1 } \stackrel { 7 } {  } \mathrm { C }  \mathrm { E }$ in run 2. Run 2’s query instead follows $s _ { 2 } \ { \stackrel { 8 } { \to } } \ \mathrm { B }$ . We again expect B→E from pattern patching and B→D from output patching. We require distinct B, D, E; D8L6 also requires correct pre-patch current-node and next-node readouts, while Ouro retains all 256 constructed pairs.

The results follow these predictions (Figure 5b). Pattern patches produce E in 77.8% of D8L6 cases and 71.1% of Ouro cases; output patches produce D in 80.9% and 93.8%, respectively. Thus, attention patterns can steer which content is selected while preserving the patched run’s values.

## 5.2 ATTENTION PATTERNS CARRY STATE-STEERING EFFECTS

We next test whether changing the attention pattern alone can reproduce the effect of J. We patch attention patterns between steered and unsteered runs on the same input. D8L6 uses the one-hop map. Ouro is evaluated on eight-hop requests in the language task. Each run keeps its own values and steering schedule; only the attention pattern changes:

$$
z _ { J  0 } = \alpha _ { J } V _ { 0 } , \qquad z _ { 0  J } = \alpha _ { 0 } V _ { J } .\tag{3}
$$

Here $J \to 0$ patches the steered pattern into the unsteered run, and $0  J$ does the reverse. If the steering effect is carried by the attention pattern, the first patch should improve unsteered accuracy, while the second should reduce steered accuracy.

The mean effects follow this prediction (Figure 5c). In D8L6, patching steered patterns into the unsteered run raises one-hop accuracy from 39.2% to 61.6%. The reverse patch reduces accuracy from 99.7% to 68.1%.

The effect on the unsteered run varies across backbones, but on average the patch still improves accuracy. A single selected head can already transfer part of this effect: for seed 10, patching head 0 raises accuracy from 68.3% to 89.4%, compared with 93.8% when all four heads are patched (means of two map fits). Thus, a steered pattern partially reproduces the steering effect, while other components of the state also affect whether the patch succeeds.

In Ouro, steered patterns produce 248/256 correct answers, close to 251/256 under full steering. Both the unsteered run and the steered run patched with unsteered patterns score zero. Patches using only values, patterns from an unrelated graph, patterns from the wrong loop, or comparison heads also score zero.

## 5.3 ATTENTION ROUTING SWITCHES THE NEXT TARGET

We next ask whether attention patterns determine which of the two controlled targets the model predicts. To test this, we run $F ( { \bar { J } } _ { \mathrm { o n e } } ( h _ { 6 } ) )$ and $F ( J _ { \mathrm { t w o } } ( h _ { 6 } ) )$ on the same input. The two runs share the same state before J and the same frozen block. Let $( \alpha _ { \mathrm { o n e } } , V _ { \mathrm { o n e } } )$ and $( \alpha _ { \mathrm { t w o } } , V _ { \mathrm { t w o } } )$ be their second-layer attention patterns and values. We patch patterns across all heads and token positions:

$$
z _ { \mathrm { t w o \to o n e } } = \alpha _ { \mathrm { t w o } } V _ { \mathrm { o n e } } , \qquad z _ { \mathrm { o n e \to t w o } } = \alpha _ { \mathrm { o n e } } V _ { \mathrm { t w o } } .\tag{4}
$$

![](images/b14c457604b5136ed3ba5376e8bc9c628e519dbe6d35a896ea406961fbc2125c.jpg)  
Figure 6: Pattern patching between one-hop and two-hop runs. Arrows indicate the patch direction. “Raw pattern” uses the run’s own attention pattern, while “Patched pattern” uses the second-layer pattern from the other run. Points show means across backbones, and horizontal bars show their ranges.

Each patched run keeps its own values but takes the attention pattern from the run using the other map.

Patching second-layer attention patterns switches the predicted target in both directions (Figure 6). Patching two-hop patterns into the one-hop run raises the frequency of two-hop answers from 0.1% to 65.9%. The reverse patch raises the frequency of one-hop answers from 0.5% to 78.5%. Both effects appear in all three selected backbones.

Restoring head outputs provides a complementary test. After replacing queries causes errors, restoring selected outputs repairs 74.0% of eligible D8L6 errors and all 249 eligible Ouro errors. Restoring comparison heads repairs 0.5% and none, respectively. Together, these interventions identify attention routing as a causal pathway through which the entering state controls the next target. We next examine how backbone training affects this control.

## 6 STEERING DEPENDS ON THE BACKBONE’S LEARNED ALGORITHMS

The preceding experiments show that attention patterns can switch between controlled targets. The three graph backbones were selected because their readouts showed both one-hop and two-hop progression. We now ask whether J and the frozen F can implement a transition that the backbone has not learned. To study this question, we change the backbone supervision before fitting J.

We train five pairs of ten-node D8L8 models. Within each pair, the models have identical initializations, inputs, architectures, and update budgets. Every query requests eight hops. Final-only training supervises loop 8 and leaves intermediate predictions unconstrained. Stepwise training adds losses for $f _ { G } ^ { t } ( s )$ at loops 1–7 to encourage one-hop progression. Writing $R ( h _ { t } )$ for the readout distribution and $\bar { \ell _ { t } } = \mathrm { C E } ( \bar { R ( } h _ { t } ) , f _ { G } ^ { t } ( s ) )$ ), the objectives are

$$
{ \mathcal { L } } _ { \mathrm { f i n a l } } = \ell _ { 8 } , \qquad { \mathcal { L } } _ { \mathrm { s t e p w i s e } } = \ell _ { 8 } + { \frac { 1 } { 7 } } \sum _ { t = 1 } ^ { 7 } \ell _ { t } .\tag{5}
$$

We then freeze each backbone and fit separate maps for staying at the current node, advancing one hop, and advancing two hops. Each map acts on $h _ { 8 }$ and is followed by one frozen loop. Both supervision regimes use the same map family and fitting protocol.

Supervision changes both unsteered behavior and steering accuracy (Figure 7). Unsteered final-only models retain the answer, whereas stepwise models advance one hop. With J, one-hop accuracy is 100% for stepwise models and 66.5% for final-only models. The ordering reverses for two hops: accuracy is 50.5% for final-only models and 13.2% for stepwise models. Final-only models have higher two-hop accuracy in all five pairs. Thus, the transitions reached by the same map family depend on backbone supervision. Appendix E reports the training protocol and paired comparisons.

We also compare supervision strategies on Ouro antonym cancellation. Each step removes the leftmost adjacent antonym pair. The answer is the pair removed at the requested step. Both backbones are trained for four loops on deletions 1–4. Stepwise training requests four deletions and supervises the current deletion at each loop. Final-only training requests a particular deletion depth and supervises its answer at loop 4.

![](images/195dfb91d22c4f965fed1c327903a6f57d72c42bdd92c57131b2972470051c24.jpg)  
Figure 7: Matched graph supervision. Colors mark answers relative to $u = f _ { G } ^ { 8 } ( s )$ . Bars average two map fits per backbone and five paired backbone seeds.

The two strategies produce different readout sequences. The stepwise model predicts one deletion per loop. The final-only model answers 255/256 requests correctly by loop 3 (Figure 8a,b).

We freeze each backbone and fit a shared rank-128 residual affine map on deletions 1–8 with four loops. The map acts after each loop’s RMSNorm, including before the final output head. Depths 5–8 are included in map training. On held-out sequences at these depths, accuracy is 98.0% for final-only plus J and 7.8% for stepwise plus J. Both unsteered models score zero (Figure 8c). Appendix F provides the Ouro task, training, and evaluation details.

(a) Stepwise  
![](images/a1eebd4e27de2af29757e2259a5c1882ede240af766e700688d75c82fe1cb938.jpg)

(b) Final-only  
![](images/9cec111e52138af96b52185afe4712b3cdd0092346ee71f225d0566f02c19eee.jpg)

(c) Four-loop steering  
![](images/39a3a7fdb2f71520fa59515f68e86569a37bbd5cc061e5c9543dce6c615fdad5.jpg)  
Figure 8: Ouro antonym cancellation. (a,b) Readout accuracy (%) as a function of loop and deletion depth. (c) Accuracy using four loops and a fitted map.

These results show that supervision changes both the unsteered readouts and the targets reached by the fitted maps. We interpret this dependence as evidence that steering relies on the algorithms learned by the backbone.

## 7 CONCLUSION

Looped Transformers reuse the same parameters across recurrent depth, but our results show that they need not reuse them in the same way. The computation performed by a shared block depends on the hidden state with which it is entered. By modifying only this state, we can redirect the same frozen block toward different semantic transitions. Attention interventions further show that this control is implemented through routing: patching attention patterns can recover steering effects and switch the predicted target while preserving the receiving run’s values. Matched supervision experiments show that which transitions are controllable depends on what the backbone has learned, supporting the view that steering redirects computations implemented by the frozen model rather than supplying an arbitrary new computation.

These findings suggest a view of recurrent computation in which the shared Transformer block acts as a reusable executor, while the evolving hidden state determines how that executor is routed across loops. Weight sharing therefore does not imply a fixed “one loop = one algorithmic step” rule. At the same time, this reuse has a finite operating regime: controllers that work over one or two loops degrade under longer composition. Successful long-horizon recurrent reasoning therefore requires not only reusable computation, but also states that remain compatible with repeatedly routing that computation.

## AI USE STATEMENT

In this work, we used ChatGPT and Codex primarily to assist with code implementation and debugging, literature search, and manuscript refinement. During writing, these tools helped improve wording, clarity, and readability; the authors determined the scientific claims and conclusions presented in the manuscript. All AI-assisted work was manually reviewed. We checked code for correctness, verified references, and reviewed the text for accuracy and consistency with the experimental results. The authors take full responsibility for the final content of this work, including all text, claims, and artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

The experiments study recurrent computation using synthetic graph, parity, and rule-based language tasks. They do not involve human participants. The attention interventions are used to analyze model computation; the experiments do not evaluate the safety of deploying steered models in downstream applications.

## REPRODUCIBILITY STATEMENT

Reproduction materials are available in the repository linked below the abstract. Please refer to its README for setup and reproduction instructions.

## REFERENCES

Hugh Blayney, Alvaro Arroyo, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville,<sup>´</sup> Michael M. Bronstein, and Xiaowen Dong. A mechanistic analysis of looped reasoning language models. arXiv preprint arXiv:2604.11791, 2026. URL https://arxiv.org/abs/ 2604.11791.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal Transformers. In International Conference on Learning Representations, 2019. URL https: //arxiv.org/abs/1807.03819.

Ying Fan, Yilun Du, Kannan Ramchandran, and Kangwook Lee. Looped Transformers for length generalization. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 25cc3adf8c85f7c70989cb8a97a691a7-Abstract-Conference.html.

Ying Fan, Anej Svete, and Kangwook Lee. Bridging the gap between latent and explicit reasoning with looped Transformers. arXiv preprint arXiv:2606.31779, 2026. URL https://arxiv. org/abs/2606.31779.

Khashayar Gatmiry, Nikunj Saunshi, Sashank J. Reddi, Stefanie Jegelka, and Sanjiv Kumar. Can looped Transformers learn to implement multi-step gradient descent for in-context learning? In Proceedings ofthe 41st International Conference on Machine Learning, volume 235, pp. 15130– 15152. PMLR, 2024. URL https://proceedings.mlr.press/v235/gatmiry24b. html.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. In Advances in Neural Information Processing Systems, volume 38, pp. 41340–41391, 2025. doi: 10.52202/085713-1380. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 3b01972cf31e6fa0fe29e4b8b5c2a0a1-Abstract-Conference.html.

Angeliki Giannou, Shashank Rajput, Jy-Yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped Transformers as programmable computers. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 11398–11442. PMLR, 2023. URL https://proceedings.mlr.press/ v202/giannou23a.html.

Wenquan Lu, Yuechuan Yang, Kyle Lee, Yanshu Li, and Enqi Liu. Latent chain-of-thought? decoding the depth-recurrent transformer. In First Workshop on the Application ofLLM Explainability to Reasoning and Planning at COLM 2025, 2025. URL https://arxiv.org/abs/2507. 02199.

Andrei Cristian Popescu, Haitz Saez de Oc´ ariz Borde, and Pietro Li´ o. Adaptive depth in\` looped Transformers: Diagnosing learned halting gates and trajectory readouts. arXiv preprint arXiv:2607.20519, 2026. URL https://arxiv.org/abs/2607.20519.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi. Reasoning with latent thoughts: On the power of looped Transformers. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2502.17416.

Rituraj Sharma and Tu Vu. Dense supervision is not enough: The readout blind spot in looped language models. arXiv preprint arXiv:2606.24898, 2026. URL https://arxiv.org/abs/ 2606.24898.

Shashwat Singh, Shauli Ravfogel, Jonathan Herzig, Roee Aharoni, Ryan Cotterell, and Ponnurangam Kumaraguru. Representation surgery: Theory and practice of affine steering. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 45663–45680. PMLR, 2024. URL https://proceedings.mlr. press/v235/singh24d.html.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248, 2023. URL https://arxiv.org/abs/2308.10248.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://arxiv.org/abs/1706.03762.

Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2211. 00593.

Liu Yang, Kangwook Lee, Robert D. Nowak, and Dimitris Papailiopoulos. Looped Transformers are better at learning learning algorithms. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ b8402301e7f06bdc97a31bfaa653dc32-Abstract-Conference.html.

Tong Zhang, Junhao Hu, Yun Peng, and Tao Xie. When does recurrence become an algorithm? convergence selection in weight-tied looped Transformers. arXiv preprint arXiv:2607.20594, 2026. URL https://arxiv.org/abs/2607.20594.

Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, Lu Li, Jiajun Shi, Kaijing Ma, Shanda Li, Taylor Kergan, Andrew Smith, Xingwei Qu, Mude Hui, Bohong Wu, Qiyang Min, Hongzhi Huang, Xun Zhou, Wei Ye, Jiaheng Liu, Jian Yang, Yunfeng Shi, Chenghua Lin, Enduo Zhao, Tianle Cai, Ge Zhang, Wenhao Huang, Yoshua Bengio, and Jason Eshraghian. Scaling latent reasoning via looped language models. arXiv preprint arXiv:2510.25741, 2025. URL https://arxiv.org/abs/2510.25741.

## PART I: ADDITIONAL EXPERIMENTS AND DISCUSSION

This part tests whether maps trained for repeated use remain accurate beyond their training range. We use separate parity and eight-node graph models. These experiments extend the reuse tests in Section 4.2, where each map was trained for one additional loop.

## A DEPTH EXTRAPOLATION

Section 4.2 reuses maps trained for one additional loop. Here we train maps for repeated use and evaluate them beyond the training range. We study parity and a separate set of twelve eight-node D8L8 backbones.

Parity predicts the sum of binary inputs modulo two after t = n loops (Fan et al., 2025). Backbone training uses lengths 1–20, and map training uses lengths 20–40. One fixed map acts before loops 2 through n. In graph continuation, map training supervises successive nodes through loop 16 using states generated by the steered run. The fixed map acts before every loop. We score the prediction at loop t against $f _ { G } ^ { t } { \bar { ( s ) } }$ while keeping the query fixed at depth 8 (Sections A.1 and A.2).

For the parity seed shown, steering improves accuracy over lengths 101–500 from 45.05% to 73.73%.   
At measured lengths 505–1000, accuracy is instead 47.55% without J and 36.40% with J (Figure 9a).   
The improvement therefore does not persist over the full evaluated range.

In graph continuation, all twelve backbones lose accuracy beyond the map-training range. On an independent held-out cycle subset, accuracy falls from 97.42% at loops 9–16 to 35.11% at loops 17–32. Figure 9b shows a separate, broader graph sample. We have not verified that its graph identities are excluded from training. These results show that accurate continuation within the map-training range does not ensure accuracy at greater depth.

![](images/24b63bda658119f01f1c792f488bd5faa830c4c44aa43ca7af913b489802e37b.jpg)

![](images/66d7b2c4f6156192f9a841a032eb888adf59586fb76a6dd255a2ee4075611584.jpg)  
Figure 9: Execution beyond map-training ranges. (a) Parity seed 2 at $t \ = \ n \colon$ dark curves are smoothed; faint points are individual measurements. (b) Graph continuation: lines average twelve backbones and two maps each; shading shows 95% bootstrap intervals over backbone means. Gray regions mark map-training ranges.

## A.1 PARITY

## A.1.1 TASK DEFINITION AND INPUT FORMAT

We use binary parity to compare the requested loop with the loops at which the model predicts the correct answer.

A parity input contains binary tokens, a separator, and optional padding. Token IDs 0 and 1 represent bits, 2 marks the separator, and 3 marks padding. For example, 1 0 1 1 2 has target 1 because it contains three ones. The sequence $ { \mathrm { ~  ~ 1 ~ } }  { \mathrm { ~ 1 ~ } }  { \mathrm { ~ 0 ~ } }  { \mathrm { ~ 0 ~ } }$ 2 has target 0.

We supervise the answer at the separator and any padding tokens that follow it. With padding, the first example becomes $\begin{array} { c c c c c c } { { } } & { { 1 } } & { { 0 } } & { { 1 } } & { { { \overline { { { 1 } } } } } } & { { 2 } } & { { 3 } } & { { 3 } } \end{array}$ , and the target from the separator onward is 1 3 3.

Input-bit positions are excluded from the answer loss. The semantic answer is the parity bit, but the implementation scores the complete answer region, including padding.

## A.1.2 MODEL AND STEERING CONFIGURATION

For an input of length n, we supervise the answer after n loops. Backbone training uses lengths 1–20. We then freeze the backbone and train the map on lengths 20–40. Both stages use cross-entropy on the answer region. The same rank-48 diagonal-plus-low-rank affine map acts before loops 2 through n. No intermediate loop receives a partial-parity target. Evaluation at longer lengths keeps the map fixed and increases both the input length and the requested loop count.

The task and length-dependent loop schedule follow Fan et al. (2025). We use causal attention without positional encodings. Embeddings enter at the first loop only. Each of the three independently trained backbones has one fitted map.

The shared parity block has one layer, hidden dimension 256, 64 attention heads, and MLP dimension 1,024. Backbone training uses 100,001 AdamW updates, batch size 64, learning rate $1 0 ^ { - 4 }$ , weight decay 0.01, and gradient clipping at 1.0. Training is in FP32. The learning rate is held constant through the initial curriculum and then follows cosine decay. Validation at length 20 is evaluated every 1,000 updates.

The map starts as the identity and uses three training stages: lengths 20–24 for 256 updates with batch size 64, lengths 20–32 for 512 updates with batch size 48, and lengths 20–40 for 1,024 updates with batch size 32. The stages total 1,792 updates and 73,728 examples. They use AdamW with zero weight decay, peak learning rate $2 \times 1 0 ^ { - 5 }$ , 1,024 warmup updates, and cosine decay to one tenth of the peak. The diagonal term uses one tenth of the learning rate of the other map parameters. Gradients are clipped at 1.0, and evaluation uses the final map. Lengths 20–40 are therefore all covered, but their total training frequencies are unequal.

## A.1.3 EVALUATION AND READOUT-TIMING ANALYSIS

We evaluate the fitted maps at longer lengths and then measure where accurate readouts occur relative to the requested exit.

Dense evaluation from length 500 to 1000. The extended curve uses the seed-2 backbone at update 99,000 and its map trained on lengths 20–40. We evaluate 101 lengths, 500, 505, . . . , 1000, with 128 binary inputs per length. The unsteered and steered runs use the same inputs, generated with seed 2026092401 + 1009n. We retain all 12,928 sequences. Embeddings enter only at loop 1, and J acts before loops 2 through n.

The original evaluation through length 500 uses 64 examples per length and a 10-length moving mean in Figure 9a. The extended evaluation starts at length 500 and samples every fifth length with 128 paired examples. Its dark curves average adjacent measured points; faint points show individual measurements. Gray shading marks map-training lengths 20–40. Over the 100 sampled lengths strictly above 500, unsteered and steered accuracy average 47.55% and 36.40%, respectively. Each length receives equal weight.

Heatmaps and readout offset. The parity heatmaps use 256 examples for each input length and loop count up to 60. Insets display lengths and loops 54–60 without interpolation; dots mark the unique best loop among $t = n - 1 , n , n + 1$ within the measured grid. The dotted horizontal line marks the longest map-training length, n = 40.

For each seed, the heatmaps, accuracy curves, and timing measurements use the same saved map and intervention schedule. We estimate the positions of the accuracy bands from the period-four Fourier component along the offset between loop number and input length. We unwrap its phase over input length to obtain a continuous estimate of band position. We then fit a line over lengths 41–490.

We compute confidence intervals from 5,000 moving-block bootstrap samples of the fit residuals, with block length 20. This estimate describes the visible accuracy bands; it does not establish an oscillator in the hidden state. We report drift, residual error, and accuracy at the requested loop for all three seeds below.

For the displayed seed 2, the fitted slope of the estimated readout position changes from 0.9904 to 0.9982. Near length 60, steering shifts the best nearby integer readout from $t = n - 1$ to $t = n$ These estimates describe readout timing over the measured range.

## A.1.4 READOUT TIMING ACROSS BACKBONES

We apply the same timing analysis to all three backbones to compare accuracy gains with changes in drift.

Changes in readout timing help explain the initial accuracy improvement for seed 2. Without steering, accurate predictions drift earlier than $t = n$ . Steering moves them closer to the requested loop (Figure 10). Over lengths $4 1 - 4 9 0$ , absolute drift falls from 0.96 to 0.18 loops per 100 tokens. This effect differs across seeds and does not explain the full accuracy curve through length 1000. Section A.1.3 defines the estimator; the table below reports all three backbones.

![](images/999faa47dcb7d11bc44847cec8201e96c0b79042d5bda38005723f9a8c5bbc08.jpg)

![](images/659f9c38b4beb5fd27926f9c9b2aa084ee4c06537578e62363a483afe5c7d267.jpg)

![](images/5af73e7f56de9f32492afb2c47e7c01097c8500f7c7444027cc83617e203497a.jpg)  
Figure 10: Parity readout timing for seed 2. (a,b) Accuracy by input length n and loop t, without and with $J ;$ insets enlarge lengths 54–60. The diagonal marks $t = n ;$ dots mark the best nearby integer loop. (c) Readout offset $t - n$ and fitted trends.

## Long-range accuracy and readout timing.

Table 1: Readout-schedule drift and longer-range accuracy with each fixed steering map. Drift is measured in loops per 100 tokens; MAE is the mean absolute deviation from the fitted phase line. Accuracy averages $t = n$ over lengths 101–500.
<table><tr><td>Seed</td><td>Drift: native → J</td><td> $\mathbf { M A E } \colon \mathrm { n a t i v e }  J$ </td><td>Accuracy: native → J (%)</td></tr><tr><td>0</td><td> $+ 0 . 5 7 3  - 0 . 3 2 7$ </td><td> $0 . 0 6 5  0 . 0 6 9$ </td><td> $3 5 . 2 7  5 5 . 7 1$ </td></tr><tr><td>1</td><td> $+ 1 . 3 1 2  + 1 . 7 3 1$ </td><td> $0 . 1 7 4  0 . 7 4 3$ </td><td> $4 4 . 0 3  4 5 . 2 4$ </td></tr><tr><td>2</td><td> $- 0 . 9 6 1  - 0 . 1 8 3$ </td><td> $0 . 0 6 6  0 . 0 5 2$ </td><td> $4 5 . 0 5  7 3 . 7 3$ </td></tr></table>

Seed 1 shows that higher accuracy at the requested loop need not imply better timing alignment. Its mean accuracy improves slightly with $J ,$ , but both drift and residual error increase. A single linear trend also fits its later readouts poorly.

## A.2 GRAPH CONTINUATION

We next test whether maps trained for repeated use sustain graph walks beyond the training range. These eight-node D8L8 backbones are separate from the ten-node D8L8 models in Section 3 and the D8L6 mechanism models.

## A.2.1 TASK DEFINITION AND INPUT FORMAT

This experiment uses the graph-token format in Appendix C.1, with eight nodes and 29 input tokens. The query continues to specify depth 8 even when execution extends beyond eight loops. At loop t, the continuation target is the node reached after t graph edges from the original start. The input is not rewritten with a new requested depth at each loop.

For example, on the cycle $0 \to 1 \to \cdots \to 7 \to 0$ with start 0, the loop-8 target is 0, the loop-9 target is 1, and the loop-10 target is 2.

## A.2.2 MODEL AND STEERING CONFIGURATION

The continuation study uses twelve eight-node D8L8 backbones with seeds 100–111. Each repeats a shared two-layer block with hidden dimension 256 and four attention heads per layer. Each reaches perfect endpoint accuracy at the supervised eighth loop. We fit two rank-48 diagonal-plus-low-rank maps per frozen backbone, using cross-entropy on the continuation targets.

Backbone training supervises the requested answer at loop 8. Map training instead supervises successive nodes at loops 1–16 and uses states generated by the steered run. The map acts before every loop, including loop 1 and evaluation loops beyond 16. This schedule differs from the single additional loop used for D8L6 target selection.

Backbone training uses 20,000 AdamW updates, batch size 512, learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay 0.3, and 500 warmup updates. We use the final checkpoint. Both the 512-graph selection pool and the 512-graph final test pool are excluded from training.

Each map is trained for 8,000 AdamW updates with batch size 128, learning rate $1 0 ^ { - 4 } ,$ , zero weight decay, and gradient clipping at 1.0. Training excludes both graph pools. Validation runs every 400 updates on the selection pool, and we retain the checkpoint with the lowest mean successor cross-entropy over loops 1–16.

## A.2.3 EVALUATION PROTOCOL

We keep the maps fixed and evaluate them beyond loop 16. The broad evaluation sample contains 8,192 unique graphs drawn uniformly without replacement from all 40,320 eight-node permutations, using seed 2026092402. It includes 993 single cycles, as well as graphs with shorter cycles and fixed points. We retain all eight starting nodes and all predictions. Evaluation uses the twelve original checkpoints and their two saved maps, with steering before every loop from loop 1 onward.

We store counts by graph, loop, backbone, and map. We average over starting nodes and graphs, then over maps and backbones. The plotted 95% interval uses 10,000 bootstrap samples of the twelve backbone means. Gray shading in Figure 9b marks loops 9–16, the displayed part of the map-training range. The map remains active after loop 16.

We have not checked this broad sample against the historical training exclusions. We therefore also report a separate evaluation on graphs excluded from training.

## A.2.4 VARIATION ACROSS BACKBONES AND EVALUATION SETS

Both populations show lower accuracy beyond the fitted loops, although they average over different graph structures.

Across the twelve backbones, steered accuracy ranges from 75.01% to 100.00% over loops 9–16 and from 34.73% to 54.56% over loops 17–32. Each value averages the two fixed maps. Every backbone has lower mean accuracy beyond the map-fitting range.

A separate training-excluded evaluation. The separate evaluation uses 512 graphs excluded from backbone training, map training, and protocol selection. On its 78 single-cycle graphs, steered accuracy is 97.42% over loops 9–16 and 35.11% over loops 17–32. Unsteered accuracy is 12.5% in both ranges. All twelve backbones lose accuracy beyond the map-training range.

We select this subset by graph structure, without filtering on predictions, and score each loop. Its results are separate from the broad-sample curve. On an eight-node cycle, retaining one endpoint produces a correct answer once every eight loops. The resulting 12.5% accuracy does not require the model to continue the graph walk.

## A.3 DISCUSSION

Training a map for repeated use does not ensure that it works at greater depth. Parity steering improves accuracy over an intermediate length range, but its timing effect varies across backbones. The displayed map reduces mean accuracy at measured lengths above 500. In graph continuation, all twelve backbones lose accuracy beyond the training range despite map training on states generated by the steered run.

## PART II: DETAILS AND SUPPORTING RESULTS FOR THE MAIN TEXT

This part provides the settings, evaluation protocols, and supporting results for Sections 3–6. Appendix B gives the complete native readouts. Appendices C and D document state control and attention interventions. Appendices E and F give the supervision comparisons, followed by a consolidated training summary in Appendix G.

## B NATIVE READOUT TRAJECTORIES

The main text shows four native readout trajectories. Here we report all twelve D8L8 seeds (0–11) and five D8L6 seeds (3–7), including the main-text examples. Each model is evaluated on 512 held-out ten-node cycles and all ten starting nodes. We do not filter examples by prediction correctness.

All heatmaps use the visual conventions of Figure 2. Color shows prediction frequency, red dashed lines mark the training loop budget, and white dots mark the unique most frequent prediction. Ties are unmarked. Positions on ten-node cycles measure progress modulo ten. Seed 6 is the original D8L6 mechanism model. Figure 14 shows the additional selected backbones.

![](images/caac529e66a02b8bb182138b38d2ebdbdc7c299c2ec67e432c9c8a42dbcfaa3e.jpg)

![](images/4f62a4819d6b9620823d7d7f7eb066c033d241416f7b1713a8c819febce5ce9a.jpg)

![](images/2c7c731e7058ea3f05d72b6e64f065c27826b0a40938005a234fcbbc3f127c5c.jpg)

![](images/3f7cc73f7762d66cfd6356dfc0f3e5c0a3f3a05d4a398a461879a542e56cf65a.jpg)

![](images/6b7c320f871c8bcd7bb3663fab5c08f60e2e596a9f15d9e7aef56a7b7677ae09.jpg)

![](images/661f330bcaf54cbbd5d5ce8d46222c9a163d5f1840d38d6b47f141df8dd3a126.jpg)  
Figure 11: Native D8L8 readouts for seeds 0–5.

![](images/77bf304999859da5fca42c16b9db3a569ace4ca5f091bc450706045f514fbcab.jpg)

![](images/a9a5c9f33720e2db3b5476acec3514ce77bbd2a79ed24e1c2c47895c2c8fffa3.jpg)

![](images/7ccfe14eeb4f2d76c52df5e53d0493f951291de597725a7654abdf7484500c8a.jpg)

![](images/29058c43515d21a3bfb5bf6ccb632eba9c067d722e9daf24b1ade96c01d5b2df.jpg)

![](images/f6e6577f8483928ed38fa2efea35e49edcca6cca9fddcd6cae7c4aff300bda3d.jpg)  
Figure 12: Native D8L8 readouts for seeds 6–11.

![](images/8bf3afb23605bf7993905f3ad0cf3f9cfd052fe144e07e1c39e938d870cd8e39.jpg)

![](images/308628283ee8694dc545d07823e7e0035fed5581f2e5841bcd863f0e53699103.jpg)

![](images/5d7af60619ea414d61f64cfb0e1b44844b71a64d845c04acbae98422c8c95bfd.jpg)

![](images/66ce109ec8b931db4a360afc2dacd90e1b8a546e737c4eb9b9a790214e51ce1e.jpg)

![](images/525619f2696b771ce7424c98d46b93591d7bc78a2f5d98c8d627e0b517cf3e6f.jpg)

![](images/4bffc6fc4f9fc1c40c239d0a05529ccfb7db849458935b9727436f5d0189b500.jpg)  
Figure 13: Native D8L6 readouts for seeds 3–7.

## B.1 READOUT SUMMARY

We summarize the heatmaps through differences between consecutive model predictions.

We use the saved readout frequencies from 512 held-out ten-node cycles and all ten starts per backbone. Let $\hat { d } _ { t }$ be the unique most frequent decoded path position across these 5,120 examples at loop t. We compute $\Delta _ { t } = ( \hat { d } _ { t } - \hat { d } _ { t - 1 } )$ mod 10 only when both modes are unique. All backbones have a tie at loop 0, so we omit the first increment. D8L8 seed 6 also has a tie at loop 4. We do not break ties arbitrarily.

These increments compare modes across the full evaluation set. They do not describe the distribution of transitions for individual examples. A zero increment, labeled “Hold,” means zero decoded displacement modulo ten. It does not mean that the hidden state performs no computation.

Table 2: The twelve D8L8 backbones and five D8L6 backbones. Model readout increments through the final supervised loop; – marks an undefined increment due to a tie.
<table><tr><td>Configuration</td><td>Seed</td><td>Increments to loops 1 through L</td></tr><tr><td>D8L8</td><td>0</td><td>−, 0, 0, 1, 0, 7, 0,0</td></tr><tr><td>D8L8</td><td>1</td><td>−, 1, 1, 1, 1, 1, 1, 1</td></tr><tr><td>D8L8</td><td>2</td><td>−, 1, 0, 0, 8, 7, 1, 0</td></tr><tr><td>D8L8</td><td>3</td><td>−, 1, 1, 1, 1, 1, 1, 1</td></tr><tr><td>D8L8</td><td>4</td><td>–, 2, 0, 2, 2, 0, 0, 0</td></tr><tr><td>D8L8</td><td>5</td><td>−, 2, 2, 2, 2, 0, 0, 0</td></tr><tr><td>D8L8</td><td>6</td><td>−, 1, 1, −, −, 1, 1, 0</td></tr><tr><td>D8L8</td><td>7</td><td>−, 1, 1, 1, 1, 1, 1, 1</td></tr><tr><td>D8L8</td><td>8</td><td>−, 1, 1, 1, 1, 1, 1, 1</td></tr><tr><td>D8L8</td><td>9</td><td>–, 2, 2, 2, 2, 0, 0, 0</td></tr><tr><td>D8L8</td><td>10</td><td>−, 1, 1, 1, 1, 1, 1, 1</td></tr><tr><td>D8L8</td><td>11</td><td>−, 1, 2, 2, 1, 1, 0, 0</td></tr><tr><td>D8L6</td><td>3</td><td>−, 2, 2, 2, 2, 0</td></tr><tr><td>D8L6</td><td>4</td><td>−, 2,8, 0, 7, 1</td></tr><tr><td>D8L6</td><td>5</td><td>−, 2, 2, 2, 2, 0</td></tr><tr><td>D8L6</td><td>6</td><td>−, 2, 2, 1, 1, 1</td></tr><tr><td>D8L6</td><td>7</td><td>−, 2, 2, 2, 2, 0</td></tr></table>

## C GRAPH WALK

This appendix supports Sections 3–5. It defines the graph task and documents the configurations, evaluation protocols, and detailed results for the control and attention experiments reported in the main text.

## C.1 TASK DEFINITION AND INPUT FORMAT

The task predicts the node reached after a requested number of edges from a specified start. Each node has a discrete token ID. The input consists of a beginning token, one three-token record per directed edge, and a query:

$$
\begin{array} { l l l l } { { \mathtt { B O S } } } & { { [ \mathtt { E D G E } } } & { { \mathtt { s o u r c e \ \ d e s t i n a t i o n } ] } } & { { \cdot \cdot \cdot } } & { { \mathtt { O U E R Y \ s t a r t \ } } } \\ { { \mathtt { D E P T H } [ \mathtt { k } ] } } & { { \mathtt { A N S W E R } } } & { { } } & { { } } \end{array}
$$

The brackets group records for display; they are not input tokens. BOS, EDGE, QUERY, and ANSWER are dedicated tokens, as is each supported depth. The model predicts a node from the hidden state at ANSWER. A ten-node input has 35 tokens. The graph is supplied in the input, rather than stored as a fixed set of edges in the model.

For example, consider the ten-node cycle 0 → 1 → · · · → 9 → 0. Its edge records are EDGE 0 1, EDGE 1 2, and so on through EDGE 9 0. The query QUERY 0 DEPTH[8] ANSWER has target node 8. Changing the start to 3 gives target node 1. These examples illustrate the encoding; they are not evaluation samples.

Backbone training samples requested path lengths 1–8. Cross-entropy supervises the requested node at the final training loop: loop 8 for D8L8 and loop 6 for D8L6. Earlier readouts receive no task loss.

For target selection, the request remains fixed at eight steps. Both maps start from the same D8L6 state after loop 6 and are followed by one frozen loop. The one-hop target is the successor of the eight-step answer; the two-hop target is its second successor. In the cycle example starting at 0, the targets are 9 and 0, respectively. No answer node is appended to the input.

The attention interventions use this same input format and target rule. They replace internal attention quantities in already trained runs; they do not introduce a new training target. Their paired inputs and scoring rules are specified in Appendix C.3.

## C.2 MODEL AND STEERING CONFIGURATION

With the task fixed, we specify the frozen backbone and the maps trained to control its next loop.

Backbone and intervention. Target selection uses the ten-node D8L6 model trained with seed 6, at update 16,000. We evaluate selection-set endpoint accuracy every 1,000 updates and retain the earliest checkpoint with the highest accuracy. The model receives an eight-hop request with answer $u = f _ { G } ^ { 8 } ( s )$ and runs for six loops to produce $h _ { 6 }$

From this state, we compare one unsteered loop, $F ( h _ { 6 } )$ , with $F ( J _ { \mathrm { o n e } } ( h _ { 6 } ) )$ and $F ( J _ { \mathrm { t w o } } ( h _ { 6 } ) )$ ). The input, backbone weights, and number of additional loops remain fixed. Each map acts on every token before the final frozen loop.

The D8L6 backbone has hidden dimension 256, four attention heads, and an MLP dimension of 1,024 in each of its two shared layers. It uses pre-layer normalization, a layer-normalized readout, and no dropout. Backbone training uses 20,000 updates, batch size 512, learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay 0.3, and 500 warmup updates. Requested depths are sampled uniformly from 1–8. Graphs are sampled uniformly from permutations after excluding the locked evaluation and selection graphs.

Map dimensions and training. Both maps have the form $J ( h ) = h ( D + A B ) + b .$ , with hidden dimension 256 and rank 48. The diagonal D has 256 learned entries, $\mathring { A } \in \mathbb { R } ^ { 2 5 6 \times 4 8 } , B \in \mathbb { R } ^ { 4 8 \times 2 5 6 }$ and $b \in \mathbb { R } ^ { 2 5 6 }$ . Each map therefore has 25,088 trainable parameters.

We fit each target map twice, with seeds 1 and 2. Only the map is trained. Cross-entropy supervises $f _ { G } ( u )$ for $J _ { \mathrm { o n e } }$ and $f _ { G } ^ { 2 } ( u )$ for $J _ { \mathrm { t w o } }$ after the frozen loop. Each fit uses 8,000 AdamW updates, batch size 128, learning rate $1 0 ^ { - 4 }$ , zero weight decay, and gradient clipping at 1.0. We validate every 400 updates and retain the earliest checkpoint with the highest validation accuracy. Training excludes the fixed selection and confirmation graphs.

## C.3 EVALUATION AND ATTENTION INTERVENTIONS

The trained maps remain fixed in all evaluations below; the experiments differ in their paired inputs, eligibility rules, and patched components.

Native intermediate readouts. For Figure 2, we request the eight-step answer and evaluate the frozen models at loops 0–16. Loop 0 precedes the first shared block. Each model is evaluated on 512 held-out ten-node cycles with all ten starting nodes, giving 5,120 examples. The figure shows four models from twelve D8L8 runs and the five D8L6 runs. Evaluation extends beyond the six or eight training loops without changing the weights.

Target-selection accuracy. Figure 3a uses the same 4,110 examples across the unsteered and steered conditions. The endpoint, one-hop, and two-hop labels are distinct, and examples are not filtered by prediction correctness. We classify each answer as the endpoint, its first successor, its second successor, or another node, and average the steered results across the two fits per target. The subsequent mechanism tests retain the same backbone and one-hop maps but use their own paired evaluation inputs.

Head selection and paired inputs. The following details describe the original seed-6 evaluation. The three-backbone aggregate and replication cohorts are specified in Appendix C.4.5. The seed-6 tests reuse the frozen checkpoint and two one-hop maps specified above.

We select head H2 in the second layer on an earlier discovery set of 32 graphs. It has the highest mean eligible pattern-patching accuracy across the two map fits, with ties resolved by head index. H3 is the prespecified comparison head. The confirmation keeps this choice fixed and retains results for all four heads.

For confirmation, sampling seed 2026092404 draws 512 graph pairs without replacement from a reserve of 17,519 unused permutation graphs. All 1,024 identities are excluded from backbone training, map training, earlier mechanism evaluations and their generated variants, and the preceding 1,024-graph evaluation of the main D8L6 backbone. The reserve was constructed by single swaps in earlier held-out graphs.

We draw the two graphs in each pair independently; they need not share most edges. Node slots align between inputs. We patch activations from run 1 into run 2. We evaluate all ten current nodes in run 2 and set the current node in run 1 three positions ahead modulo ten. For each graph, we choose the starting node so that eight graph steps reach the designated current node. Each run executes six loops, applies the map to all tokens, and executes a seventh frozen loop.

Scoring populations. These choices produce 5,120 candidate pairs before patching. The patternversus-output experiment requires three distinct answers: the original answer in run 2, the answer in run 1, and the answer obtained by following the patched attention pattern in the graph used for run 2. Both unpatched runs must correctly read their current and next nodes. This leaves 3,966 pairs for fit 1 and 3,963 for fit 2; within each fit, both patch types use exactly the same pairs.

For restoration, the alternative-current run must have correct current and next readouts and a distinct target. Recovery is scored among eligible clean transitions broken by query replacement, giving 4,413 and 4,417 cases. Steering-pattern patching uses 4,044 examples with distinct current, one-hop, and two-hop labels, without conditioning on prediction correctness. The earlier target-selection figure uses a separate cohort of 4,110 examples. Each experiment retains its own denominator.

Uncertainty and implementation checks. For each map fit, we bootstrap 5,000 samples of the 512 graph pairs. All ten starting nodes of a graph remain together. The two fits measure variation between maps on one backbone, rather than between independently trained backbones.

Direct evaluation, reconstructed attention, an independent output hook, and same-run activation replacement agree on predictions when no effective intervention is made.

Table 3: D8L6 evaluation populations. Counts are graph–start instances or paired runs, as appropriate. All map fits within a row use the stated graph population; correctness-based eligibility can differ by fit.
<table><tr><td>Experiment</td><td>Scored count</td><td>Inclusion and intervention</td></tr><tr><td>Native trajectories</td><td>5,120 per backbone</td><td>512 ten-node cycles; all starts; no prediction filtering.</td></tr><tr><td>Target selection</td><td>4,110</td><td>Distinct current, one-hop, and two-hop labels; no prediction filtering; one additional F.</td></tr><tr><td>Pattern vs. output</td><td>3,966 / 3,963</td><td>Distinct semantic answers and correct clean readouts in both runs; selected second-layer head H2.</td></tr><tr><td>Steering-pattern patching, Figure 5c</td><td>4,044</td><td>Distinct current, one-hop, and two-hop labels; no prediction filtering; all L2 heads at the answer</td></tr><tr><td>Output restoration</td><td>4,413 / 4,417</td><td>position. Eligible clean transitions broken by query replacement; restore H2 or control H3.</td></tr><tr><td>Target switching</td><td>4,116</td><td>Separate 512-graph cohort; distinct current, one-hop, and two-hop labels; all second-layer</td></tr><tr><td>Original two- and eight-step composition</td><td>3,200</td><td>heads and token positions. Same 512 graphs as target switching; u through  $f _ { G } ^ { 4 } ( u )$  distinct; no prediction filtering.</td></tr></table>

## C.4 SUPPORTING MECHANISM AND CONTROL RESULTS

Using these evaluation rules, we report output restoration, target switching, and controller reuse.

## C.4.1 RESTORING HEAD OUTPUTS RECOVERS ANSWERS

We test whether restoring selected head outputs repairs errors caused by query replacement.

We run the same graph with two different starting nodes. The original run provides the clean answer and head outputs. The alternative run provides replacement queries. We insert these queries into the original run while keeping its keys and values unchanged. This changes where attention reads without replacing the input’s value vectors. We then replace the selected heads’ disrupted outputs with their clean outputs and let the model continue. For an intervened head, let $Q _ { \mathrm { a l t } }$ denote the replacement queries and $d _ { k }$ the query/key dimension:

$$
z _ { \mathrm { b r o k e n } } = \mathrm { s o f t m a x } \bigg ( \frac { Q _ { \mathrm { a l t } } K _ { \mathrm { c l e a n } } ^ { \top } } { \sqrt { d _ { k } } } \bigg ) V _ { \mathrm { c l e a n } } , \qquad z _ { \mathrm { r e s t o r e d } } \gets z _ { \mathrm { c l e a n } } ,\tag{6}
$$

Here softmax is taken over the visible key positions. If the selected outputs carry information needed for the answer, restoring them should repair the error caused by query replacement. We compare this repair with restoring a control set of heads. Recovery is measured only on examples that were answered correctly before query replacement and incorrectly afterward.

Restoring the selected outputs repairs errors in both models. In the original D8L6 seed-6 evaluation, it repairs 60.1% of disrupted answers, compared with 0.9% for the control. In Ouro, it repairs 249 of 249 answers, while the matched control repairs 0. Restoring all affected outputs repairs every disrupted answer in both models. The selected outputs therefore provide information that later computation uses to produce the answer.

## C.4.2 PATCHING ONE-HOP AND TWO-HOP PATTERNS

We reuse the frozen seed-6 backbone and both pairs of target-specific maps. Sampling seed 2026092501 draws 512 fresh graphs from the training-excluded reserve, excluding previous evaluation graphs and their generated variants. Evaluating all ten starting nodes yields 5,120 examples. We retain the 4,116 examples with distinct current, one-hop, and two-hop labels, without filtering by prediction correctness. Both directions and both map fits use this same population. This cohort is separate from the 4,110-example target-selection figure and the 4,044-example steering-pattern experiment.

The two runs share the input and $h _ { 6 }$ but use different maps before the final frozen loop. We patch post-softmax patterns at all token positions and all four heads of layer 2. Each patched run keeps its own values and continues normally.

Same-run replacements preserve predictions. Direct evaluation and the intervention implementation agree on unpatched predictions; checkpoint hashes remain unchanged. Confidence intervals use 5,000 paired bootstrap samples of graph identities, retaining all starting nodes and both map fits together. The two fits quantify map-fit variation on a fixed backbone.

Table 4: Seed-6 answers matching the target of the run providing the second-layer patterns (percent). Arrows indicate the direction of patching. Confidence intervals describe the two-fit mean, with graphs as bootstrap units.
<table><tr><td>Direction</td><td>Fit 1</td><td>Fit 2</td><td>Mean</td><td>95% CI</td></tr><tr><td> $J _ { \mathrm { t w o } } \to J _ { \mathrm { o n e } }$ </td><td>44.97</td><td>45.82</td><td>45.40</td><td>[44.43, 46.36]</td></tr><tr><td> $J _ { \mathrm { o n e } } \to J _ { \mathrm { t w o } }$ </td><td>59.67</td><td>62.17</td><td>60.92</td><td>[59.84, 61.97]</td></tr></table>

## C.4.3 COMPOSITION OF FROZEN TARGET-SPECIFIC MAPS

We evaluate D8L6 seed 6 and both existing pairs of one-hop and two-hop maps without further training. The first continuation applies a map to $h _ { 6 }$ and executes the seventh loop. The second applies a map to the resulting full token-state sequence and executes the eighth loop. The input remains fixed, and no state reset or intermediate-answer input is used. Each map acts independently on all tokens. We also evaluate all combinations in which either map is omitted, giving nine two-loop conditions per fit.

This exploratory evaluation reuses the 512 training-excluded graphs from the target-switching experiment. All ten starting nodes give 5,120 examples. We require $u , f _ { G } ( u ) , \dots , f _ { G } ^ { 4 } ( u )$ to be pairwise distinct, leaving the same 3,200 examples for every condition. We do not filter on prediction correct ness. Requiring distinct labels through four hops makes this set smaller than those used for a single additional loop.

We define targets relative to u, not the model’s first prediction. Each control that omits the second map is scored against the same cumulative target as the corresponding two-map sequence.

For both map fits, first-continuation predictions agree exactly with the earlier saved evaluations. Model parameters remain unchanged, and model, map, data, and code hashes are recorded. We bootstrap 5,000 paired samples of graph identities, preserving all starting nodes and both map fit within each sampled graph. Intervals quantify graph-sampling uncertainty for the two-fit mean on a fixed backbone.

Table 5: Two-step composition accuracy (%). Confidence intervals are 95% graph-bootstrap intervals for the mean of the two map fits.
<table><tr><td>Backbone seed</td><td>Sequence</td><td>Fit 1</td><td>Fit 2</td><td>Mean</td><td>95% CI</td></tr><tr><td>6</td><td>One-one</td><td>93.88</td><td>93.41</td><td>93.64</td><td>[92.87, 94.36]</td></tr><tr><td>6</td><td>One-two</td><td>63.53</td><td>67.03</td><td>65.28</td><td>[64.02, 66.50]</td></tr><tr><td>6</td><td>Two-one</td><td>73.22</td><td>75.78</td><td>74.50</td><td>[73.42, 75.56]</td></tr><tr><td>6</td><td>Two-two</td><td>68.78</td><td>67.47</td><td>68.12</td><td>[66.57, 69.71]</td></tr></table>

Composition accuracy depends on the sequence of maps. One–one reaches 93.64% and two–two reaches 68.12%. These results establish two-step reuse in seed 6 and motivate testing longer sequences.

## C.4.4 EIGHT-STEP CONTROLLER REUSE

We reuse the same frozen maps over sequences of lengths 1–8 without further training. Starting from native $h _ { 6 } ,$ , we update the state as $h ^ { ( j ) } \stackrel { - } { = } F ( J _ { a _ { i } } ( h ^ { ( j - \bar { 1 } ) } ) )$ . We test repeated one-hop maps, repeated two-hop maps, and 32 distinct nonconstant binary strings of length eight. We sample the strings with a fixed seed before evaluation and use their prefixes for shorter sequences. The target at step j is $f _ { G } ^ { \sum _ { i = 1 } ^ { j } a _ { i } } ( u )$ , where $a _ { i } \in \{ 1 , 2 \}$ . We never reset the state or supply an intermediate answer.

This exploratory evaluation uses the same 512 graphs and starting nodes as the two-step study. The primary set contains the same 3,200 instances with distinct $u , \ldots , f _ { G } ^ { 4 } ( u )$ at every sequence length. We do not filter on predictions, and both map fits exactly reproduce the saved two-step predictions.

A ten-node graph cannot have sixteen distinct successive nodes. Different cumulative targets can therefore coincide in the longer sequences. We report exact match between the predicted node and the cumulative target at each step. These tests measure repeated use on the same graph inputs; they do not test sixteen-hop extrapolation with distinct nodes at every step.

Table 6: Exact match at the eighth additional controlled loop (percent, mean of two fits).
<table><tr><td>Seed</td><td>Repeated one</td><td>Repeated two</td><td>Mixed</td></tr><tr><td>6</td><td>9.66</td><td>10.62</td><td>10.11</td></tr></table>

On the fixed 3,200-instance set, eighth-call exact match is approximately 10% for each sequence family. Successful control over one or two loops therefore does not ensure that the resulting states remain controllable over longer sequences.

Table 7: Accuracy at every controller-sequence length on the fixed 3,200-example population (percent, mean of two fits). Mixed averages the 32 fixed random strings.
<table><tr><td>Seed</td><td>Sequence</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td></tr><tr><td>6</td><td>one</td><td>99.6</td><td>93.6</td><td>57.9</td><td>20.5</td><td>8.3</td><td>6.0</td><td>6.7</td><td>9.7</td></tr><tr><td>6</td><td>two</td><td>98.0</td><td>68.1</td><td>23.4</td><td>10.8</td><td>11.6</td><td>11.1</td><td>10.4</td><td>10.6</td></tr><tr><td>6</td><td>mixed</td><td>99.0</td><td>78.9</td><td>39.7</td><td>12.3</td><td>9.3</td><td>9.1</td><td>9.7</td><td>10.1</td></tr></table>

Independent graph confirmation. We repeat the composition evaluation on 512 additional graphs sampled from the training-excluded locked pool with seed 2026093001. These graphs are disjoint from the 512 graphs used above. All ten starting nodes produce 5,120 examples; the same distinct-$u , \ldots , f _ { G } ^ { 4 } ( u )$ rule retains 3,175 examples, without filtering on predictions. We evaluate five existing D8L6 backbones and two existing map fits per backbone. No parameters are updated. The 32 mixed strings are fixed to those used in the earlier evaluation. Figure 4 shows this independent evaluation for seed 6, with 3,000 graph-bootstrap resamples for its intervals.

Table 8: Independent composition evaluation (percent, mean of two fits). The final column gives the mixed-sequence target accuracy at the eighth additional call.
<table><tr><td>Seed</td><td>One-one</td><td>One-two</td><td>Two-one</td><td>Two-two</td><td>Mixed, call 8</td></tr><tr><td>6</td><td>93.64</td><td>65.04</td><td>75.13</td><td>68.03</td><td>10.09</td></tr><tr><td>3</td><td>22.76</td><td>12.11</td><td>46.96</td><td>97.83</td><td>10.98</td></tr><tr><td>4</td><td>83.15</td><td>41.81</td><td>56.57</td><td>28.79</td><td>9.05</td></tr><tr><td>5</td><td>17.69</td><td>6.68</td><td>29.89</td><td>84.61</td><td>10.64</td></tr><tr><td>7</td><td>19.80</td><td>9.15</td><td>26.61</td><td>91.72</td><td>11.57</td></tr></table>

The independent results reproduce the dependence on map order and backbone. For seed 6, the four two-step accuracies are close to those in the original evaluation. At eight calls, mixed-sequence exact match is between 9.05% and 11.57% across the five backbones.

We also compare continuous control with omitting the current map after the identical controlled prefix, using the map only at the first call, and running the backbone without any map. We record the answer after J but before F and a baseline that simply copies u. The table below reports the mixed strings for seed 6 on the same 3,175 examples. Omitting only the second map reduces accuracy from 78.89% to 32.93%; reading directly after the second map gives 0.66%. At longer lengths, continuous control no longer maintains an advantage. The copy baseline can score above zero when the requested walk returns to $u ;$ the recorded per-example results also support scoring only targets different from u.

Table 9: Independent mixed-sequence controls for seed 6 (percent, mean of two fits). Columns count additional calls after $h _ { 6 }$
<table><tr><td>Condition</td><td>Call 2</td><td>Call 4</td><td>Call 8</td></tr><tr><td>Continuous J</td><td>78.89</td><td>12.62</td><td>10.09</td></tr><tr><td>Omit current J</td><td>32.93</td><td>13.12</td><td>10.00</td></tr><tr><td>J at first call only</td><td>32.93</td><td>7.52</td><td>14.99</td></tr><tr><td>No J</td><td>2.93</td><td>10.47</td><td>16.18</td></tr><tr><td>Read after J, before F</td><td>0.66</td><td>7.72</td><td>10.32</td></tr><tr><td>Copy u</td><td>0.00</td><td>14.25</td><td>17.08</td></tr></table>

## C.4.5 SELECTED BACKBONES AND AGGREGATION

We report mechanism experiments on three selected D8L6 backbones: seeds 6, 10, and 13. Seed 6 is the original case study. We selected seeds 10 and 13 from native readouts before evaluating their mechanism interventions. Both show one-hop and two-hop differences between consecutive model readouts. The mean across these backbones describes the selected cases, not a random sample of seeds.

The heatmaps use 512 held-out ten-node cycles and all ten starting nodes. They follow the visual conventions of Figure 2. They show decoded predictions and do not directly measure internal operations.

Seed 6  
![](images/0c1207d600589450a42f4fe82eef390d9ab696057243b63d2b8407c60f64625c.jpg)

Seed 10  
![](images/47d8d98eb58d7c0cabd2050af9195fbaecaac311404d94a3f89863f2bdc0f331.jpg)

Seed 13  
![](images/7b46b5cf7ba5384d1961625270165d1475e57dfb6fc34f1e0878830998e6866d.jpg)  
Figure 14: Native readouts of the D8L6 backbones used in the attention experiments.

Each backbone has two independently fitted rank-48 maps per target, trained as described in Appendix C.2. We average the fits within each backbone, then average the three backbone accuracies with equal weights. We do not pool examples across backbones. Figure 5b,c shows each backbone mean. Figure 6 shows their range, which is not a confidence interval.

For seeds 10 and 13, we select heads using 32 independent graph pairs. The selected second-layer heads are H0 and H3, respectively; seed 6 retains H2. We evaluate all four heads on confirmation data. The new confirmation set contains 512 graphs in each run. A separate set of 512 graphs is used for second-layer target-switching patches.

All graph identities come from the reserve excluded from backbone and map training. Discovery and confirmation sets are disjoint. The new backbones share these sets, while seed 6 retains its original independent sets. Input alignment, interventions, and inclusion rules remain unchanged. Patches between steered and unsteered runs replace the patterns of all four second-layer heads at the answer position. Target-switching patches cover all heads and positions in layer 2.

The cross-graph eligible counts for fits 1/2 are 3,966/3,963 (seed 6), 4,104/4,104 (seed 10), and 3,822/3,692 (seed 13). Patching between steered and native runs uses 4,044 instances for seed 6 and 4,111 for each new backbone. Target switching uses 4,116 and 4,149, respectively. The latter two experiments exclude label collisions without conditioning on correct predictions.

We also replace queries and restore outputs using the same protocol and selected head for each backbone. The comparison head is the next head modulo four. Averaging within and then across backbones gives 74.0% recovery for the selected head, 0.5% for the comparison head, and 100% for all four heads. Recovery is measured among eligible answers that are correct before query replacement and incorrect afterward.

## D OURO LETTER-WALK

This appendix describes the language-based graph task and attention interventions used in Section 5.

## D.1 TASK DEFINITION AND INPUT FORMAT

Letter-walk expresses a graph walk in text and has a known answer at each requested depth.

Each example defines a cycle over ten people using ten sentences of the form “Alice passes the letter to Bob.” The input also names the starting person and the number of steps to follow. Names are Alice, Bob, Carol, David, Emma, Frank, Grace, Henry, Iris, and Jack. Rules appear in shuffled order, so their textual order does not give the path. Ouro’s tokenizer and chat template encode the instructions, rule sentences, and question as a user message. The target is an assistant message containing only the final person’s name.

For example, take the cycle Alice → Bob → Carol → David → Emma → Frank → Grace → Henry → Iris → Jack → Alice. The prompt includes all ten rules. Starting with Alice, the two-step answer is Carol and the eight-step answer is Iris. For illustration, the question can be phrased as:

The letter starts with Alice. After exactly 02 steps, who has it? Answer directly with only the person’s name.

This example paraphrases the question wording; 02 preserves the template’s two-digit encoding of the requested step count. The rules and question are ordinary text, not one special token per person or per edge. Different prompt templates express the same task.

## D.2 MODEL AND STEERING CONFIGURATION

Supervision and steering. We fine-tune the backbone on 1–4-step requests with four recurrent loops. Answer-token cross-entropy is applied only at loop 4. Prompt tokens are masked from this loss. The target contains the assistant’s answer and the chat-template ending, with no intermediate reasoning trace.

We then freeze the backbone and fit a dense affine map on 1–8-step requests. The same map acts on every token before loops 2–4, and answer supervision remains at loop 4. Both training stages combine answer cross-entropy and general-text cross-entropy with weights 0.8 and 0.2. An eight-step request is therefore beyond the backbone-training depths but within the map-training range. All mechanism evaluations request eight steps. Patching changes internal activations while preserving the requested task and answer rule.

Backbone fine-tuning uses Adafactor with factored FP32 states and no first moment, learning rate $1 0 ^ { - 5 }$ , and 50 warmup updates. Each update contains eight task examples. Map fitting uses AdamW, learning rate $1 0 ^ { - 4 }$ , 50 warmup updates, and sixteen task examples per update, with two examples for each requested depth 1–8. Both stages use FP32 weights with BF16 autocast and gradients through all four loops. The early-exit gate is frozen and unused. The dense map starts as the identity and has 4,196,352 parameters.

Checkpoint and intervention sites. All Ouro mechanism tests use the letter-walk Ouro-2.6B checkpoint saved after 200 backbone updates and its dense affine map saved after 500 map updates. The frozen map is $J ( h ) = h W + b$ with unrestricted W, applied to all token states before loops 2–4. Head outputs are patched before their output projection during prompt processing; generation follows the normal steering schedule of the patched run.

## D.3 EVALUATION AND ATTENTION INTERVENTIONS

We keep the trained backbone and map fixed during head selection and evaluation. Each intervention experiment uses a separate evaluation set.

Head selection. The selected heads are L41.H2/H10/H12/H15, L43.H1/H6/H11/H13/H15, and L47.H0/H2/H3/H4/H7/H13/H15, all at loop 4. Layer and head indices start at zero. We select heads using eight discovery pairs and a separate 64-pair confirmation set for pattern patching. We then fix these sixteen heads. The pattern-versus-output, restoration, and steering-pattern evaluations each use a separate set of 256 pairs.

The pattern-versus-output and restoration experiments use sixteen comparison heads, matched by layer to the selected heads.

Restoring outputs after query replacement. The restoration experiment uses 256 new query pairs generated with seed 2026092642. The graph pairs use the same seven-edge shared-path construction as the pattern-versus-output experiment. However, replacement queries come from a run on the same graph with a different starting person. This evaluation set excludes all earlier evaluation graphs, including the new pattern-versus-output set.

At loop 4, we replace all queries in layers 41, 43, and 47. The patched run keeps its own keys and values. We then restore the clean outputs of the sixteen selected heads, sixteen comparison heads matched by layer, or all 48 affected heads. We measure recovery only for answers that were correct before query replacement and incorrect afterward. Restoration outcomes do not determine which examples are included.

Pairs that separate routing from retrieved content. The pattern-versus-output experiment uses 256 new pairs generated with seed 2026092641. Their 512 distinct graph identities exclude all 1,184 identities from earlier head selection and mechanism evaluations, including the new steering-pattern set. Each graph is a ten-person cycle. From the starting person in run 1, the first seven people on the path are the same in both graphs, but the eighth differs. Run 2 starts at a different person. Names occupy aligned token positions.

The pair construction ensures three distinct answers: the original answer in run 2, the answer in run 1, and the answer in the second graph from the starting person used in run 1. We evaluate the third answer with a reference run that combines that graph and starting person. Pair inclusion does not depend on model accuracy. This construction defines the predicted answers under pattern and output patching without assuming one graph step per loop.

Independent steering-pattern evaluation. We sample 256 new graph pairs with seed 2026092632. Their 512 distinct graph identities exclude all 672 identities from the preceding mechanism evaluations and head selection. Each graph is a ten-person cycle. The paired prompts have aligned token positions, and the existing pair-construction rules are unchanged. No pair is filtered by model correctness. The backbone, map, sixteen heads, and loop-4 intervention sites remain fixed; no training or head reselection is performed.

For the same input, we patch steered patterns into an unsteered run and unsteered patterns into a steered run. The controls patch only the steered values, patterns from an unrelated graph, loop-2 patterns into loop 4, or steered patterns at sixteen disjoint comparison heads matched by layer. These comparison heads are L41.H0/H1/H3/H4, L43.H0/H2/H3/H4/H5, and L47.H1/H5/H6/H8/H9/H10/H11.

All patches affect prompt processing only. During generation, each run keeps its original steering schedule. We use greedy generation with a maximum of sixteen tokens and score both the first answer token and the complete name.

Numerical checks and statistical reporting. Same-run output replacement and restoration of every affected output both yield zero maximum logit error. In the steering-pattern evaluation, same-run output replacement also gives zero maximum logit error. Explicit same-run pattern reconstruction differs from BF16 attention by at most 0.125 in logits and yields no accuracy gain (0/256).

The Ouro figures report Wilson confidence intervals. When full names are generated, we compare their correctness with first-token correctness and count truncated generations as incorrect. No verified graph-exclusion list is available for the backbone. We therefore cannot establish that these graph identities were absent from backbone training.

## D.4 ADDITIONAL RESULTS

Head-output restoration. The table reports both accuracy over all query pairs and recovery among clean answers broken by query replacement.

Table 10: Restoring Ouro head outputs on 256 new query pairs. Recovery conditions on clean answers broken by query replacement.

<table><tr><td>Condition</td><td>Correct / 256</td><td>Recovered / broken</td></tr><tr><td>Clean</td><td>251</td><td></td></tr><tr><td>Replaced queries</td><td>2</td><td>0/249</td></tr><tr><td>Selected 16 heads</td><td>251</td><td>249/249</td></tr><tr><td>Matched 16 neighbors</td><td>2</td><td>0/249</td></tr><tr><td>All 48 affected heads</td><td>251</td><td>249/249</td></tr></table>

The selected outputs recover 100.0% of eligible broken answers (Wilson 95% interval 98.5–100.0%). Complete-name correctness agrees with first-token correctness for the clean, corrupted, selected-head, comparison-head, and all-head conditions. No generation is truncated, and restoring all affected outputs exactly reconstructs the clean logits.

Pattern-versus-output patching. The table reports the answers from each unpatched run and the answer predicted by pattern patching.

Table 11: Ouro pattern-versus-output patching on 256 new pairs. Counts use first-answer-token predictions. Patches use activations from run 1 in run 2. “Rerouted” denotes the answer in the second graph under the patched pattern.

<table><tr><td>Condition</td><td>Run 2</td><td>Run 1</td><td>Rerouted</td><td>Other</td></tr><tr><td>Run 2</td><td>252</td><td>0</td><td>0</td><td>4</td></tr><tr><td>Run 1</td><td>0</td><td>252</td><td>2</td><td>2</td></tr><tr><td>Graph from run 2, start from run 1</td><td>0</td><td>4</td><td>251</td><td>1</td></tr><tr><td>16-head pattern</td><td>37</td><td>0</td><td>182</td><td>37</td></tr><tr><td>16-head output</td><td>5</td><td>240</td><td>7</td><td>4</td></tr><tr><td>16-head values</td><td>221</td><td>0</td><td>24</td><td>11</td></tr><tr><td>Matched-neighbor pattern</td><td>251</td><td>0</td><td>0</td><td>5</td></tr><tr><td>Matched-neighbor output</td><td>252</td><td>0</td><td>0</td><td>4</td></tr><tr><td>Wrong-loop pattern</td><td>137</td><td>3</td><td>0</td><td>116</td></tr></table>

Pattern patching produces the answer predicted by the patched pattern in 182/256 cases (71.1%; Wilson 95% interval 65.3–76.3%). It produces the answer from run 1 in 0/256 cases. Output patching produces these two answers in 7/256 and 240/256 cases, respectively. These results distinguish changing where attention reads from replacing the retrieved output.

Full-name and first-token classifications agree on all 256 pairs in every generated condition. No generation is truncated, so the distinction holds under both scoring rules.

Steering-pattern patching. The earlier 64-pair evaluation gave 61/64 correct answers under the sixteen-head pattern patch, compared with 62/64 under full steering and 0/64 without steering. Table 12 reports the independent 256-pair evaluation used in Figure 5c.

On this new set, the fixed pattern patch reaches 248/256 correct answers (96.9%; Wilson 95% interval 94.0–98.4%). Full steering reaches 251/256. The reverse pattern patch and all four comparison interventions score zero. Full-name and first-token correctness agree for every example in all eight conditions, and no generation is truncated.

Table 12: Independent Ouro steering-pattern evaluation with the fixed sixteen heads. First-token and complete-name scores agree in all conditions.
<table><tr><td>Condition</td><td>Correct / 256</td><td>Accuracy (%)</td></tr><tr><td>Unsteered</td><td>0</td><td>0.0</td></tr><tr><td>Full steering</td><td>251</td><td>98.0</td></tr><tr><td>Steered patterns → unsteered</td><td>248</td><td>96.9</td></tr><tr><td>Unsteered patterns → steered</td><td>0</td><td>0.0</td></tr><tr><td>Steered values only → unsteered</td><td>0</td><td>0.0</td></tr><tr><td>Unrelated-graph patterns → unsteered</td><td>0</td><td>0.0</td></tr><tr><td>Wrong-loop patterns → unsteered</td><td>0</td><td>0.0</td></tr><tr><td>Comparison-head patterns → unsteered</td><td>0</td><td>0.0</td></tr></table>

Full steering succeeds on four examples where the pattern patch fails. The pattern patch succeeds on one example where full steering fails. The paired accuracy gap is 1.17 percentage points (95% bootstrap interval −0.39 to 2.73, using 10,000 resamples of graph pairs). Thus, the fixed pattern patch recovers the steering effect on new evaluation graphs. These results do not establish equivalence to full steering or replication across independently trained Ouro backbones.

## E MATCHED SUPERVISION IN GRAPH WALK

This experiment tests the effect of adding intermediate losses while keeping the query and backbone training stream fixed. It uses ten-node D8L8 models trained separately from the D8L6 control models and the eight-node continuation models.

## E.1 PAIRED BACKBONE TRAINING

Each model repeats a two-layer Transformer block for eight loops. The hidden dimension is 256, with four attention heads, MLP dimension 1,024, and no dropout. We train five pairs with seeds 0–4. Within each pair, both models start from identical parameters and receive the same sequence of training tokens. Every input requests the eight-hop target.

Let $\ell _ { t }$ be cross-entropy on $f _ { G } ^ { t } ( s )$ at loop $t .$ The final-only objective is $\ell _ { 8 }$ . The stepwise objective is

$$
\mathcal { L } _ { \mathrm { s t e p w i s e } } = \ell _ { 8 } + \frac { 1 } { 7 } \sum _ { t = 1 } ^ { 7 } \ell _ { t } .\tag{7}
$$

Thus, the final-loop coefficient remains one. Adding the intermediate losses changes the total loss and gradient scale; these quantities are not normalized to match final-only training.

Both regimes use 20,000 AdamW updates, batch size 512, learning rate $3 \times 1 0 ^ { - 4 }$ , weight decay 0.3, 500 warmup updates, and cosine learning-rate decay. Gradients pass through all eight loops. We use the final checkpoint of every run. Saved hashes verify that each pair shares its initialization and complete training-token stream.

## E.2 MAPS AND EVALUATION

We freeze each backbone and collect $h _ { 8 }$ on fixed depth-eight queries. Three separately fitted maps target $u = f _ { G } ^ { 8 } ( s ) , f _ { G } ( u ) , \mathrm { o r } f _ { G } ^ { 2 } ( u )$ after one additional call to F. Each map has the form

$J ( h ) = h ( D + A B ) + b $ , with a learned diagonal D, rank 48, and 25,088 parameters. It acts on all tokens and starts as the identity.

Each target has two map fits. Each fit uses 8,000 AdamW updates, batch size 128, learning rate $1 0 ^ { - 4 }$ zero weight decay, and gradient clipping at 1.0. We evaluate on the selection set every 400 updates and retain the earliest checkpoint with the highest selection accuracy. The map-training, selection, and test pools contain 2,048, 512, and 512 graphs, respectively. The pools are disjoint and shared across all ten backbones. The test graphs are excluded from backbone training.

The test pool contains 5,120 graph–start instances. We use the same 4,137 instances with distinct $u , f _ { G } ( u )$ , and $f _ { G } ^ { 2 } ( u )$ in every condition. Inclusion depends on graph structure, not predictions. All accuracies below are read after the additional frozen loop. We average the two fits within a backbone before computing the mean and sample standard deviation across the five backbones.

## E.3 RESULTS ACROSS PAIRED SEEDS

Without a map, all five final-only backbones retain u, and all five stepwise backbones predict $f _ { G } ( u )$ on the scoring population. Every fitted stay map reaches 100% accuracy in both regimes. Every stepwise one-hop map also reaches 100%. Table 13 summarizes all conditions, and Table 14 reports the remaining individual fits.

Figure 15 shows the one-hop and two-hop comparisons within each backbone pair.

![](images/7f2ac166651e9ba74918650954dd941d4d9559ae422639b5370bf78612be61cb.jpg)  
Figure 15: Paired backbone comparisons for (a) one-hop and (b) two-hop targets. Lines connect the two supervision regimes for the same backbone seed. Points average two map fits; horizontal offsets separate overlapping points.

Table 13: Matched ten-node D8L8 supervision comparison. Entries are mean target accuracies ± sample standard deviation across five backbone seeds (%). We first average the two map fits within each backbone. Under J, each column uses its own target-specific map.
<table><tr><td>Supervision</td><td>Intervention</td><td>Stay</td><td>One hop</td><td>Two hops</td></tr><tr><td>Final-only</td><td>None</td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td>Stepwise</td><td>None</td><td> $0 . 0 \pm 0 . 0$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td></tr><tr><td>Final-only</td><td>J</td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $6 6 . 5 \pm 1 0 . 0$ </td><td> $5 0 . 5 \pm 1 5 . 2$ </td></tr><tr><td>Stepwise</td><td>J</td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $1 3 . 2 \pm 6 . 8$ </td></tr></table>

Table 14: Matched D8L8 map accuracy (%) on 4,137 common test instances. Each pair of columns reports the two independently fitted maps for that target.
<table><tr><td colspan="4">One hop</td><td colspan="2">Two hops</td></tr><tr><td>Seed</td><td>Supervision</td><td>Fit 1</td><td>Fit 2</td><td>Fit 1</td><td>Fit 2</td></tr><tr><td>0</td><td>Final-only</td><td>78.75</td><td>76.46</td><td>62.05</td><td>56.80</td></tr><tr><td>0</td><td>Stepwise</td><td>100.00</td><td>100.00</td><td>25.57</td><td>15.54</td></tr><tr><td>1</td><td>Final-only</td><td>74.09</td><td>72.71</td><td>66.76</td><td>65.55</td></tr><tr><td>1</td><td>Stepwise</td><td>100.00</td><td>100.00</td><td>11.58</td><td>12.79</td></tr><tr><td>2</td><td>Final-only</td><td>56.42</td><td>55.43</td><td>57.77</td><td>56.90</td></tr><tr><td>2</td><td>Stepwise</td><td>100.00</td><td>100.00</td><td>6.48</td><td>5.00</td></tr><tr><td>3</td><td>Final-only</td><td>68.41</td><td>70.22</td><td>39.21</td><td>40.83</td></tr><tr><td>3</td><td>Stepwise</td><td>100.00</td><td>100.00</td><td>19.17</td><td>20.45</td></tr><tr><td>4</td><td>Final-only</td><td>56.78</td><td>55.60</td><td>30.67</td><td>28.04</td></tr><tr><td>4</td><td>Stepwise</td><td>100.00</td><td>100.00</td><td>7.57</td><td>7.69</td></tr></table>

One-hop accuracy is 66.5 ± 10.0% for final-only models and $1 0 0 . 0 \pm 0 . 0 \%$ for stepwise models. Two-hop accuracy reverses this ordering: 50.5 ± 15.2% for final-only models and $1 \bar { 3 . 2 } \pm 6 . 8 \%$ for stepwise models. Here ± denotes the sample standard deviation across the five backbone means. Final-only two-hop accuracy is higher in every pair. These results describe the transitions reached by the tested map family and fitting procedure. They do not establish the limits of other state interventions.

## F OURO ANTONYM CANCELLATION

This appendix defines the task and the two supervision strategies compared in Section 6.

## F.1 TASK DEFINITION AND INPUT FORMAT

The input contains a word sequence, a deletion rule, and a requested deletion count. At each step, the rule removes the leftmost adjacent antonym pair and joins the remaining words in their original order. The vocabulary consists of twelve pairs: hot/cold, near/far, open/closed, heavy/light, young/old, rich/poor, happy/sad, clean/dirty, full/empty, wet/dry, fast/slow, and alive/dead. Actual examples contain all 24 words, with pair orientations and nesting varied. Ouro encodes the full text using its tokenizer and chat template. The answer contains the two words removed at the requested step, in their left-to-right order.

A short illustrative sequence is hot near far cold open closed. The first deletion removes near far, leaving hot cold open closed. The second removes hot cold, and the third removes open closed. Thus, the answer to deletion 2 is hot cold, not the remaining sequence. This shortened example illustrates the rule; training and evaluation use 24-word inputs. In the final-only format, the question is “Which two words are deleted on deletion number 2? Answer with only those two words in their original left-to-right order.” The stepwise format instead states the requested number of deletions and asks for the pair removed in the current deletion update.

## F.2 MODEL AND STEERING CONFIGURATION

Answer-token supervision. The user prompt is masked from the task loss. Cross-entropy supervises the assistant’s two-word answer and chat-template ending, without a generated deletion trace.

Backbone supervision. We fine-tune the two Ouro backbones separately for 500 updates, with four loops in every run. Final-only training samples requests for deletions 1–4 and supervises the requested pair at loop 4. Stepwise training always requests four deletions. It supervises the pair removed at deletion t after loop t, for $t = { \bar { 1 } } , \dots , { \bar { 4 } }$ . Thus, one model learns to answer a requested depth at the final loop, while the other learns a fixed sequence of intermediate answers.

A shared map for each frozen backbone. Each backbone receives its own map $J ( h ) = h ( I +$ AB) + b, with a rank-128 residual update and 526,336 trainable parameters. It is shared across all four loops and acts on every token after each loop’s RMSNorm, including after loop 4 before the frozen language-model head. Zero initialization of B and b makes the initial map the identity.

Both fits use 500 AdamW updates at learning rate $1 0 ^ { - 4 }$ , with 50 warmup updates. An update contains sixteen examples, two for each requested deletion depth 1–8. The loss weights answer cross-entropy by 0.8 and general-text cross-entropy by 0.2. During both map fitting and evaluation, the prompt specifies the requested deletion count. For both backbones, map training supervises the requested pair at the fixed loop-4 exit. Adaptive halting is disabled so all requests use four loops.

## F.3 EVALUATION PROTOCOL

Native readout heatmaps. Figure 8a,b uses the unsteered backbones at update 500. In the stepwise panel, every prompt requests four deletions. For each of 64 sequences, we compare the parsed pair at each loop with all four true deletion pairs to construct the depth-by-loop matrix.

In the final-only panel, each prompt requests the depth shown on the vertical axis. Each cell contains 64 examples from the saved loop-1/2 and loop-3/4 evaluations. Both panels score the parsed deletion pair, without requiring exact full-text generation or EOS emission.

Paired evaluation at deeper requested deletions. The main evaluation uses prompts that specify a deletion count and ask for the pair removed at that step (template 0 in the implementation). At each depth, the same 64 held-out sequence structures and target pairs are evaluated in all four conditions: each backbone with and without its map. Success requires matching the requested deletion pair. Figure 8c reports the evaluation at map update 500.

At deletion depths 5–8, final-only plus J answers 251/256 requests correctly and stepwise plus J answers 20/256. At depth 8, the counts are 63/64 and 2/64. Both native models score zero on depths 5–8. In the native final-only readout panel, loop 3 answers 255/256 requests correctly, including 63/64 at deletion 4.

We evaluate one backbone and one map per training strategy. Depths 5–8 are beyond backbone training but are included in map training. The low stepwise accuracy shows that the tested map and fitting procedure fail on these requests. It does not rule out other state interventions that could produce the correct answers.

## G TRAINING AND INTERVENTION SUMMARY

Table 15 summarizes the training budgets and intervention schedules for the main-text experiments. The corresponding appendices specify losses, sampling rules, optimizer settings, and checkpoint selection. Repeated map fits optimize separate maps on a frozen backbone; they are not independent backbone runs.

Table 15: Training budgets and intervention sites for the main-text experiments. Backbone updates identify the selected checkpoint or the training budget as specified; map updates give the fitting budget.
<table><tr><td>Experiment</td><td>Backbone</td><td>Map</td><td>Placement</td></tr><tr><td>D8L6 control</td><td>20,000 updates; seed 6 selected at 16,000</td><td>Rank 48; 8,000 updates; two fits per target</td><td>All tokens before the next  $F ;$  maps reused in compositions.</td></tr><tr><td>Matched graph supervision</td><td>20,000 updates; five paired final checkpoints updates; two fits</td><td>Rank 48; 8,000</td><td>At  $h _ { 8 } ,$  followed by one additional  $F .$ </td></tr><tr><td>Ouro letter-walk</td><td>Checkpoint at update 200</td><td>per target Dense affine; checkpoint at</td><td>Before loops 2–4; all tokens.</td></tr><tr><td>Ouro cancellation</td><td>500 updates per training strategy</td><td>update 500 Residual rank 128; 500 updates</td><td>After each loop&#x27;s RMSNorm, including loop</td></tr></table>

## H ADDITIONAL EVIDENCE FOR STATE STEERING

We provide two further demonstrations of state steering: numerical and calendar tasks with a taskadapted Qwen3-8B model, and relation composition in a synthetic knowledge graph. In both settings, a shared affine map modifies hidden states between calls to a frozen backbone and substantially improves the model’s ability to reach the requested target.

## H.1 CONTROL OF CONTINUED COMPUTATION IN QWEN3-8B

Tasks and model. The five task families are integer successor, weekday advancement, repeated doubling, Fibonacci-pair updates, and Collatz iteration. Each prompt specifies an initial state and an iteration count k, and asks for the final answer directly. For Fibonacci, an update maps $( a , b )$ to $( b , a + b )$ and the answer is the second component after k updates. For example, a doubling prompt starting at 8 with $k = 3$ has answer 64.

We form a recurrent model by sharing Qwen3-8B’s full stack of 36 decoder layers across loops. Token embeddings enter once, the native token-position RoPE is retained across loops, and the final RMSNorm and language-model head are applied after the last loop. The backbone is first adapted with four loops and requests $k = 1 , \ldots , 4 ,$ , supervising the final answer. We then freeze the backbone and train one token-wise dense affine map $\bar { J ( h ) } = \bar { h ^ { } } + W h + b ,$ shared across all five tasks and all seven boundaries of an eight-loop execution. The map has 16,781,312 parameters, starts from the identity, and is trained on requests $k = 1 , \ldots , 8$ using final-answer cross-entropy. The evaluated backbone and map are the checkpoints at updates 9,500 and 12,000, respectively.

Paired continuation comparison. The evaluation contains 32 base cases per task and all eight requested counts, giving 1,280 prompts. Every branch preserves the controlled first four loops. From this common boundary $h _ { 4 } .$ , we compare four suffixes: four further $J { - } F$ pairs, four $F$ calls without $^ { J , }$ four applications of J without F, and immediate readout. Thus, the $J + F$ and F-only branches both use eight backbone calls in total. Answers are generated greedily, without a KV cache, with a 12-token limit. Accuracy requires the complete generated answer to match the target after trimming surrounding whitespace and ignoring case. Each branch generates its own continuation from the same prompt.

Table 16: Qwen continuation accuracy (%). All branches share the controlled computation through $h _ { 4 } ;$ column headings specify what happens afterward. Each task row contains 128 long requests $( k = 5  – 8 )$ . The pooled short and long rows each contain 640 prompts.
<table><tr><td>Task or request range</td><td> $J + F$ </td><td>F only</td><td>J only</td><td>Stop</td></tr><tr><td>Successor</td><td>100.00</td><td>75.00</td><td>71.09</td><td>70.31</td></tr><tr><td>Weekday</td><td>100.00</td><td>61.72</td><td>78.12</td><td>52.34</td></tr><tr><td>Doubling</td><td>100.00</td><td>3.91</td><td>4.69</td><td>3.12</td></tr><tr><td>Fibonacci</td><td>100.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Collatz</td><td>48.44</td><td>2.34</td><td>2.34</td><td>2.34</td></tr><tr><td>All long requests  $( k = 5  – 8 )$ </td><td>89.69</td><td>28.59</td><td>31.25</td><td>25.62</td></tr><tr><td>All short requests  $( k = 1 { - } 4 )$ </td><td>99.38</td><td>100.00</td><td>100.00</td><td>100.00</td></tr></table>

Results. Continuing with $J + F$ answers 574/640 long requests correctly, compared with 183/640 for continued $F$ alone, 200/640 for continued J alone, and 164/640 for stopping at $h _ { 4 }$ (Table 16). The full controlled execution reaches 100% on successor, weekday, doubling, and Fibonacci, and 48.44% on Collatz. Removing either the state intervention or the backbone from the suffix sharply reduces long-request accuracy. These results show that a shared state intervention can support effective continued computation across several task families in the same frozen language model.

## H.2 CONTROLLED RELATION COMPOSITION IN A SYNTHETIC KNOWLEDGE GRAPH

Task and frozen backbone. The knowledge graph contains 64 entities and 16 relations. Each relation r is a fixed permutation $f _ { r }$ of the entity set. Given a starting entity $e _ { 0 }$ and a relation sequence $( r _ { 1 } , \ldots , r _ { n } )$ , the target is

$$
e _ { n } = f _ { r _ { n } } \circ \cdot \cdot \cdot \circ f _ { r _ { 1 } } ( e _ { 0 } ) .\tag{8}
$$

Inputs consist of a beginning-of-sequence token, the initial entity identifier, and the relation identifiers. The same relation permutations are used throughout training and evaluation. The backbone has a shared two-layer Transformer block, width 256, eight attention heads, MLP width 1,024, causal attention, and no positional encoding. Embeddings enter only at the first loop. Backbone training uses relation lengths 1–3 with cross-entropy on the aligned intermediate entities; we use the selected checkpoint at update 16,000.

Affine state control. After freezing the backbone, we fit an identity-initialized, token-wise affine map with 65,792 parameters. A length-n query uses one initial F call followed by n − 1 repetitions of J then F. The same map is reused at every boundary. The comparison without J uses the same input, frozen backbone, readout, and n calls to F.

The reported map first follows a length curriculum through every integer length from 4 to $^ { 1 6 , }$ with up to 3,000 updates per stage, batch size 128, and learning rate $1 \dot { 0 } ^ { - 4 }$ . Training supervises the correct intermediate entity after each F call from call 2 onward. Starting from this length-16 checkpoint, a further 10,000 updates mix 5,000 batches with lengths sampled uniformly from 4–16 and 5,000 batches at length 16. This continuation uses AdamW with zero weight decay and a learning rate that rises to $3 \times 1 0 ^ { - 5 }$ over 500 updates, stays constant for 8,500 updates, and decays to zero over the last 1,000. We evaluate the final checkpoint.

![](images/4fbb489cdae85fec458da69d2e990926d72cfde7db20cbf60a134f3a6e4f329e.jpg)  
Figure 16: State steering in synthetic KG relation composition. Each point uses 1,024 queries, paired between the two conditions, with exactly n backbone calls for a length-n query. Gray shading marks backbone training lengths 1–3; blue shading marks map training lengths 4–16. The dashed vertical line marks the end of map training coverage.

Results. We evaluate every length from 1 to 32 on 1,024 sampled queries per length. Across lengths 4–16, the affine map raises target accuracy from 2.53% to 99.92% (Table 17). Accuracy with J remains 99.61% at length 17, 97.75% at length 18, and 85.74% at length 19, before declining with longer compositions (Figure 16). The average is 42.87% over lengths 17–24 and 1.65% over lengths 25–32. The affine intervention therefore enables reliable relation composition well beyond the backbone’s original training lengths, with a clear decline as execution moves farther beyond the map’s training range.

Table 17: KG target accuracy aggregated over equally sized per-length evaluation sets. Every row compares the same queries and the same number of frozen-backbone calls.
<table><tr><td>Relation length</td><td>Queries</td><td>F only (%)</td><td>With affine J (%)</td></tr><tr><td>1-3</td><td>3,072</td><td>99.87</td><td>100.00</td></tr><tr><td>4-16</td><td>13,312</td><td>2.53</td><td>99.92</td></tr><tr><td>17-24</td><td>8,192</td><td>1.54</td><td>42.87</td></tr><tr><td>25-32</td><td>8,192</td><td>1.56</td><td>1.65</td></tr></table>

Together, the Qwen and KG experiments provide additional evidence that shared affine interventions at loop boundaries can substantially improve controlled execution with frozen backbones, extending the state-steering results to a task-adapted language model and a learned symbolic relation executor.