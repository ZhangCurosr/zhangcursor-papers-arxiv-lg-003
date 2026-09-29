# SOLAR: A STATE-DRIVEN ONLINE LEARNING RATE SCHEDULER FOR LLM PRETRAINING

Qiulin Shang<sup>1</sup>, Binyu Wang<sup>2</sup>, Yongqi Qiao<sup>1</sup>, Songde Rao<sup>1</sup>, Zhoutong Wu<sup>1</sup>, Kun Yuan<sup>1</sup>   
<sup>1</sup>Peking University <sup>2</sup>Nanjing University   
qiulin.shang@stu.pku.edu.cn

## ABSTRACT

Learning-rate (LR) scheduling plays a central role in large language model (LLM) pretraining, yet current practice still relies heavily on hand-crafted heuristics such as Warmup-Cosine-Decay and Warmup-Stable-Decay. Because these schedules are fixed in advance, they cannot adapt to evolving optimization dynamics. Online learned scheduling within the Learning to Optimize (L2O) framework offers a dynamic alternative, but remains brittle at LLM scale due to noisy signals, delayed feedback, and the risk of catastrophic divergence. We propose State-driven Online Learning rAte scheduleR (SOLAR), a stabilized framework for reliable online LR adaptation. SOLAR uses a base schedule as a reference and learns bounded, state-dependent residual corrections for individual parameter groups. Each correction re-anchors to the base at every step, allowing the policy to adapt the LR without relearning the warmup–decay profile. A lightweight state representation and progress-aware reward guide online learning, while a Circuit-Breaker restores training after rare unsafe actions. Across autoregressive language-model pretraining, SOLAR improves final perplexity over tuned static schedules and automatic LR tuners for dense models from 60M to 1B, AdamW and Muon, and two MoE settings up to 3B. Matched 130M controls show that adding base anchoring and action bounds improves a global PPO controller from 27.09 to 23.74 final PPL, while group-wise control reaches 22.87 on the same two seeds. A residual policy trained on a 60M proxy can also be frozen and reused at larger dense scales without target PPO updates, remaining effective across a fourfold base-LR range. These results establish SOLAR as a practical learned LR controller for LLM pretraining.

## 1 INTRODUCTION

Despite decades of progress in optimization, the learning-rate schedule for pretraining billionparameter large language models (LLMs) is still set by hand and fixed before training begins. In practice, hand-crafted schedules such as Warmup-Cosine-Decay (Cosine) (Loshchilov & Hutter, 2017) and Warmup-Stable-Decay (WSD) (Hu et al., 2024) remain the default due to their simplicity, reliability, and ease of deployment (Touvron et al., 2023a;b). Yet they impose a fundamental limitation: once training begins, the schedule is entirely predetermined and cannot adapt to the evolving optimization landscape, including shifts in loss dynamics, gradient norms, and parameter magnitudes. Adaptive optimizers do not resolve this limitation: although methods such as Adam (Kingma & Ba, 2015) and Muon (Jordan et al., 2024) rescale updates at the individual parameter level, the global learning rate that governs the overall update magnitude is still specified in advance.

A natural alternative is online learning-rate scheduling, where the learning rate is adjusted on the fly according to current and historical training signals (Baydin et al., 2018; Jin et al., 2021; Daniel et al., 2016). This approach aligns closely with the broader Learning-to-Optimize (L2O) paradigm, as it requires the scheduler to dynamically adapt based on the unfolding optimization state. In large-scale LLM pretraining, however, online scheduling is notoriously difficult to operationalize. The available optimization signals, such as loss trends and gradient norms, are noisy—that is, they are high-variance, non-stationary, and often only weakly informative at the level of individual steps when optimizing a nonconvex objective with stochastic methods (Chowdhery et al., 2023; Huang et al., 2025; McCandlish et al., 2018). A suboptimal scheduling decision may not trigger an immediate failure yet can silently degrade progress over a long horizon; conversely, an overly aggressive adjustment can cause abrupt instability within a few steps (Takase et al., 2025). These delayed and asymmetric failure modes make learned exploration of the learning rate particularly brittle in realistic pretraining runs. As a result, although learned schedulers have shown promise in smaller settings, they have not yet been widely demonstrated as reliable solutions in realistic LLM pretraining settings. This gap motivates the search for an online scheduler that is not only adaptive but also robust enough to operate in such settings.

(a): Optimization Perspective  
![](images/8e490fea4ccd08ca791ad32c7ed315357cec6b40f0ec26e655743f3c3f962e58.jpg)

![](images/f85f314051cba5227cfeab2054ce24a8b26bea537209ed4a28af1b0d9a2a4d9a.jpg)  
Figure 1: Overview of SOLAR. Panel (a) presents the optimization perspective: SOLAR augments a base scheduler with residual learning-rate modulation, applies bounded group-wise LR actions to the LLM pretraining process, and uses an automated Circuit-Breaker to roll back unsafe trajectories when severe loss spikes are detected. Panel (b) illustrates the RL control loop: SOLAR constructs lightweight global and local state features from the current training dynamics, outputs parameter-group-wise residual actions, receives delayed reward feedback after the optimizer update, and improves the scheduler through PPO updates

In this work, we introduce SOLAR (State-driven Online Learning rAte scheduleR), a reinforcement learning (RL)- based L2O framework for online learning-rate scheduling in LLM pretraining. Rather than generating the full LR trajectory, SOLAR uses a base schedule as a reference and learns bounded, state-dependent residual corrections for individual parameter groups. The base supplies the warmup and decay profile, while the policy adapts to the realized optimization state. Each correction re-anchors to the current base LR, preventing exploratory errors from compounding into a new schedule. As illustrated in Figure 1, SOLAR maps lightweight training-state features to parameter-groupwise LR adjustments online. Its stochastic residual actions support fine-grained exploration, and action-scale warmup limits early perturbations. A progress-aware reward combines immediate improvement, longer-term trend, and group stability. A threshold-based Circuit-Breaker detects severe loss spikes and restores the latest safe checkpoint.

![](images/d02d3259df0e1a4afc5c255f06464cd52a2ac1138e551f38df66aaf546942134.jpg)  
Figure 2: Pretraining performance on C4 across model scales. SOLAR consistently achieves the lowest perplexity, outperforming strong baselines in both AdamW and Muon families.

We evaluate SOLAR on autoregressive language modeling over C4 (Raffel et al., 2020) for Llama 2 (Touvron et al., 2023b) models ranging from 60M to 1B parameters, using AdamW (Loshchilov & Hutter, 2019) and Muon. For the 1B model, SOLAR reduces perplexity (PPL) by over 10% relative to AdamW with Cosine and by about 4% relative to Muon with Cosine (Figure 2). Repeated runs preserve the gains at every repeated scale, including two paired 1B AdamW runs. We further validate SOLAR on pretraining a 1B-parameter Qwen2-MoE (Yang et al., 2024) model on The Pile (Gao et al., 2020) under AdamW. SOLAR improves final perplexity from 9.61 to 9.34 while reducing early-stage optimization volatility (Figure 3); a 3B MoE run improves from 10.73 to 10.38 (Appendix C.12). Matched 130M controls clarify why the method works: a recursive global PPO controller ends at 27.09 PPL, base anchoring and action bounds improve it to 23.74, and group-wise control reaches 22.87 on the same two seeds. A residual policy trained exclusively on a 60M proxy can also be frozen and reused at larger dense scales without target PPO updates or SOLAR-specific retuning; it remains effective across 0.5×–2× base LRs.

Overall, our main contributions are:

• Methodological Design. We propose SOLAR, an RL-based framework for online LR scheduling in LLM pretraining. SOLAR keeps the base LR profile outside the learned controller and learns bounded, state-dependent group-wise corrections that re-anchor at every step. A progress-aware reward and Circuit-Breaker support policy learning throughout a full pretraining run.

• Empirical Demonstration. SOLAR improves final PPL over tuned static schedules and automatic LR tuners across dense and MoE models under AdamW and Muon, with repeated gains through 1B. On matched seeds, controls show that base anchoring and action bounds improve a global PPO controller by 3.35 PPL, while group-wise control contributes another 0.87 PPL.

• Practical Scalability. A residual policy acquired through full-length online runs on a 60M proxy can be frozen and reused at larger dense scales without target PPO updates or SOLAR-specific retuning, while remaining effective across a fourfold base-LR range.

## 2 RELATED WORK

Learning Rate Scheduling and Hyperparameter Optimization. Preset schedules encode the open-loop training profile. Cosine (Loshchilov & Hutter, 2017; Touvron et al., 2023a), WSD (Hu et al., 2024), CLR (Smith, 2017), and one-cycle schedules (Smith & Topin, 2019) choose warmup and decay before observing the realized run. Blockwise LR (Wang et al., 2025) assigns different rates to transformer module types but remains preset. Hyperparameter-optimization methods search this profile: Bayesian optimization (Snoek et al., 2012) and population-based training (Jaderberg et al., 2017) use multiple trials, while AutoLRS (Jin et al., 2021) evaluates short candidate segments online. MECHANIC (Cutkosky et al., 2023) instead adapts a global multiplier. SOLAR keeps the base profile outside the learned controller and learns closed-loop corrections within the live target run.

Learned Learning-Rate Controllers. Learned LR control has a substantial lineage. Early work uses reinforcement learning to select optimization hyperparameters or step sizes (Hansen, 2016; Daniel et al., 2016; Xu et al., 2017). Xu et al. (2019) train a PPO controller from past training histories and transfer it across image tasks. GNS (Xiong et al., 2022) encodes layer states with a graph network and learns a global schedule across target episodes. Subramanian et al. (2023) study PPO scheduling on MNIST and CIFAR, while GANNO (Tessera et al., 2023) uses layer-wise agents and absolute LR actions. These studies establish RL scheduling, state conditioning, transfer, and layer-wise control. SOLAR addresses a different operating constraint: a single full autoregressive LLM pretraining trajectory must both train the controller and remain competitive. The base schedule supplies the open-loop profile; the policy learns only bounded residual corrections that re-anchor every step. To our knowledge, SOLAR is the first learned LR scheduler shown to train online throughout the same full LLM pretraining run that it improves. Appendix C.1 compares the learned objects and acquisition protocols, and Appendix C.4 evaluates matched controllers in the LLM setting.

Adaptive and Schedule-Free Optimizers. Adaptive optimizers modify update magnitudes from gradient history. AdaGrad (Duchi et al., 2011), RMSProp (Tieleman & Hinton, 2012), AdamW (Loshchilov & Hutter, 2019), Adafactor (Shazeer & Stern, 2018), Lion (Chen et al., 2023), and Muon (Jordan et al., 2024) differ in their internal update geometry, but still accept an external LR trajectory. D-Adaptation (Defazio & Mishchenko, 2023), Prodigy (Mishchenko & Defazio, 2024), and Schedule-Free optimization (Defazio et al., 2024) reduce that dependence through optimizer-side scalar adaptation or iterate averaging. SOLAR leaves the optimizer update unchanged and controls the LR across parameter groups from observed training states. The AdamW and Muon experiments test whether this scheduler-level control remains useful across distinct optimizer families.

Learning to Optimize. Learning to Optimize (L2O) replaces hand-designed update rules with learned ones (Chen et al., 2022). Learned optimizers map gradients to updates (Andrychowicz et al., 2016; Li & Malik, 2017; Ravi & Larochelle, 2017); hierarchical models (Wichrowska et al., 2017), large-scale meta-training (Metz et al., 2022), and constrained update spaces (Liu et al., 2023; He et al., 2024) improve their reach. SOLAR learns a lower-dimensional object. The base optimizer retains its update rule, and the policy controls a bounded residual around an external schedule. This restriction makes online acquisition feasible inside the target pretraining run.

## 3 PRELIMINARIES

Notation. We consider LLM pretraining as minimizing a loss $\mathcal { L } ( w )$ over parameters $w \in \mathbb { R } ^ { d }$ with stochastic gradients $g _ { t } = \nabla \mathcal L ( w _ { t } )$ . The parameters are partitioned into G controlled groups, each assigned a learning rate $\eta _ { t , g } > 0$ . SOLAR supplies $\{ \eta _ { t , g } \} _ { g = 1 } ^ { G }$ to a standard optimizer (e.g., AdamW, Muon), which internally produces the next iterate $w _ { t + 1 }$ during optimization.

Learning Rate Scheduling as a Sequential Decision Problem. We formulate online LR scheduling as a sequential decision process. At step t, the scheduler observes state $s _ { t } = \{ s _ { t , g } \} _ { g = 1 } ^ { G }$ , outputs the group-wise LR action vector $a _ { t } = \{ a _ { t , g } \} _ { g = 1 } ^ { G }$ , and receives rewards $\boldsymbol { r } _ { t + 1 } = \{ \boldsymbol { r } _ { t + 1 , g } \} _ { g = 1 } ^ { G }$ . The full action vector is applied to one LLM optimizer step, which jointly determines the next training state and the group-wise rewards.

Proximal Policy Optimization (PPO). We train the scheduler with PPO (Schulman et al., 2017) using a parameter-shared independent update across groups. Each controlled group contributes a likelihood ratio paired with its own advantage. The clipped terms are averaged to update the shared actor–critic. Appendix ${ \tt A . 2 }$ gives the exact objective, network architecture, and hyperparameters.

## 4 SOLAR: STATE-DRIVEN ONLINE LEARNING RATE SCHEDULER

SOLAR at a Glance. A base schedule specifies warmup and decay. SOLAR learns the missing closed-loop correction from the current run. It maps global and parameter-group states to bounded residual actions that re-anchor to the base every step. A progress-aware reward trains this policy online, and a Circuit-Breaker restores the last safe state after a severe loss spike.

## 4.1 LIGHTWEIGHT STATE REPRESENTATION

SOLAR’s state design follows a minimal-variance principle: it relies on a compact set of lowvariance optimization signals that are informative yet cheap to compute, avoiding high-dimensional or optimizer-internal statistics. In the dense implementation, each controlled parameter group corresponds to one trainable parameter tensor; the MoE implementations follow the module partition of their respective codebases. SOLAR assigns one learning-rate multiplier per group. To minimize the overhead of online scheduling, SOLAR adopts a lightweight state representation built on a small set of universal optimization signals: loss statistics, gradient norms, and parameter norms.

State representation. For controlled parameter group g, the state is defined as $\begin{array} { r l } { s _ { t , g } } & { { } = } \end{array}$ $[ s _ { t } ^ { \mathrm { g l o b a l } } ; s _ { t , g } ^ { \mathrm { l o c a l } } ]$ . The global state summarizes the macro training phase and loss dynamics:

$$
s _ { t } ^ { \mathrm { g l o b a l } } = \left[ \tau _ { t } , \log L _ { t } , \nu _ { t } , \Delta \mathrm { E M A } _ { t } \right] ,\tag{1}
$$

where $\tau _ { t }$ is the normalized training progress, $L _ { t }$ is the current loss, $\nu _ { t }$ measures short-window loss fluctuation, and $\Delta { \mathrm { E M A } _ { t } }$ captures the discrepancy between short-term and long-term exponential moving averages of the loss. The local state summarizes controlled-parameter-group-level context:

$$
s _ { t , g } ^ { \mathrm { l o c a l } } = \bigl [ \log \eta _ { t , g } ^ { \mathrm { b a s e } } , \log \Vert \nabla _ { t , g } \Vert , a _ { t - 1 , g } , d _ { g } , \log \Vert w _ { t , g } \Vert , \Delta \log \Vert \nabla _ { t , g } \bigr \Vert \bigr ] ,\tag{2}
$$

where $\nabla _ { t , g }$ is the current mini-batch gradient, $a _ { t - 1 , g }$ is the previous scheduling action, $d _ { g }$ is the depth index, $w _ { t , g }$ is the trainable tensor, and $\Delta$ log $\lVert \nabla _ { t , g } \rVert$ measures recent changes in gradient magnitude. This compact representation keeps the scheduler lightweight and decoupled from optimizer-internal statistics. Detailed mathematical formulations for all state features, including moving average decay rates and numerical stability constants, are provided in Appendix A.1.

## 4.2 ACTION SPACE: STOCHASTIC GROUP-WISE RESIDUAL MODULATION

Given the observed state, a conventional action design is to let the policy directly output an absolute learning rate deterministically. However, accurately predicting a single optimal learning rate at each step is inherently difficult in noisy, large-scale, and highly non-stationary LLM pretraining. SOLAR addresses this challenge through two coupled design choices under practical constraints.

Stochastic group-wise residual actions. Instead of predicting a deterministic scalar learning rate, SOLAR samples a joint collection of one-dimensional bounded residual learning-rate actions across controlled parameter groups from a factorized squashed Gaussian policy. The joint policy factorizes across groups, with one squashed Gaussian factor per controlled parameter group. For each controlled parameter group g, the scheduler outputs a mean $\mu _ { t , g } ,$ , while the policy uses a single learnable scalar standard deviation σ shared across all controlled parameter groups. We sample a latent variable, apply the hyperbolic tangent, and clip only the executed action for numerical safety:

$$
u _ { t , g } \sim \mathcal { N } ( \mu _ { t , g } , \sigma ^ { 2 } ) , \qquad \widetilde { u } _ { t , g } = \operatorname { t a n h } ( u _ { t , g } ) , \qquad a _ { t , g } = \operatorname { c l i p } ( \widetilde { a } _ { t , g } , - 1 + 1 0 ^ { - 4 } , 1 - 1 0 ^ { - 4 } ) .
$$

The stochastic policy explores group-specific corrections, while the bounded action limits the damage of any single decision. We use group-specific means $\mu _ { t , g }$ and one learned scalar variance shared across groups; Appendix A.2 gives the implementation.

Residual LR modulation. A base schedule encodes the coarse LR profile. SOLAR keeps that profile outside the learned object and controls a residual:

$$
\eta _ { t , g } = \eta _ { t , g } ^ { \mathrm { b a s e } } \cdot \exp ( \alpha _ { t } a _ { t , g } ) ,\tag{3}
$$

where $\alpha _ { t } \geq 0$ controls the residual range. We linearly warm up $\alpha _ { t }$ from 0 to α during early training. Because each multiplier is applied to the current base LR, the action cannot recursively redefine later LRs. The policy can focus on state-dependent corrections instead of relearning warmup and decay.

## 4.3 REWARD DESIGN FOR SUSTAINED OPTIMIZATION

After action $a _ { t }$ and the optimizer update, the next ordinary forward/backward pass produces $L _ { t + 1 }$ and $\nabla _ { t + 1 }$ . SOLAR reuses these detached quantities and adds no language-model forward or backward pass. A one-step loss reward is too noisy to describe sustained progress, so the reward combines immediate improvement, an EMA trend, and group-wise stability:

$$
r _ { t + 1 , g } = r _ { t + 1 } ^ { \mathrm { p e r f } } + r _ { t + 1 } ^ { \mathrm { t r e n d } } - p _ { t + 1 , g } ^ { \mathrm { s t a b } } ,\tag{4}
$$

where $r _ { t + 1 } ^ { \mathrm { p e r f } }$ rewards step-wise loss reduction, $r _ { t + 1 } ^ { \mathrm { t r e n d } }$ encourages improvements over a longer EMA horizon, and $p _ { t + 1 , g } ^ { \mathrm { s t a b } }$ is a signed, controlled-group-wise stability shaping term (Appendix A.3). It compares each group’s post-update gradient norm with its recent trend, crediting gradients that settle below the trend and penalizing growth above it. The two progress terms remain tied to the global language-modeling objective.

## 4.4 SAFETY AND RECOVERY: CIRCUIT-BREAKER MECHANISM

Rare loss spikes can occur during LLM pretraining (Chowdhery et al., 2023; Zhang et al., 2022; Zeng et al., 2023) and may invalidate online exploration. SOLAR therefore adds a global emergency trigger that terminates unsafe rollouts and restores the latest safe checkpoint. Specifically, we augment the reward with an explicit instability penalty:

$$
r _ { t + 1 , g } ^ { \mathrm { f i n a l } } = r _ { t + 1 , g } - \lambda _ { \mathrm { c b } } \psi _ { t + 1 } ,\tag{5}
$$

where $\lambda _ { \mathrm { c b } }$ is a large penalty coefficient, and $\psi _ { t + 1 }$ is a binary instability indicator defined as

$$
\psi _ { t + 1 } = \mathbb { 1 } \left[ L _ { t + 1 } > \kappa _ { L } \cdot L _ { t + 1 } ^ { \mathrm { e m a } } \right] ,\tag{6}
$$

with $\kappa _ { L }$ denoting a safety threshold, $L _ { t + 1 }$ the post-update training loss, and $L _ { t + 1 } ^ { \mathrm { e m a } }$ its exponential moving average. Unlike the per-group stability shaping term in (4), the Circuit-Breaker acts as a global emergency signal triggered only by severe loss spikes. When $\psi _ { t + 1 } = 1$ , the current rollout is immediately terminated and the scheduler performs a PPO update on the truncated trajectory. The training loop then restores the model and optimizer states from the latest safe checkpoint while retaining the updated PPO state. This prevents post-failure samples from contaminating subsequent policy updates and provides an explicit failure signal that encourages the scheduler to internalize stability constraints.

Algorithm 1: State-Driven Online Learning Rate Scheduler (SOLAR)   
Input: Base scheduler $\{ \eta _ { t , g } ^ { \mathrm { b a s e } } \}$ , optimizer O, policy π , value network V , total steps T   
1 for t = 0, . . . , T − 1 do   
// 1. State Construction   
2 Construct scheduler states $\{ s _ { t , g } \} _ { g = 1 } ^ { G } ;$   
// 2. Per-Group LR Action Sampling   
3 Sample $\{ u _ { t , g } \} _ { g = 1 } ^ { G }$ , form $\{ a _ { t , g } \} _ { g = 1 } ^ { G }$ , and record $\{ \ell _ { t , g } ^ { \mathrm { o l d } } \} _ { g = 1 } ^ { G } ;$   
4 Set learning rates $\eta _ { t , g } = \eta _ { t , g } ^ { \mathrm { b a s e } } \exp ( \alpha _ { t } a _ { t , g } )$ for all $^ { g ; }$   
// 3. Environment Step   
5 Execute one optimizer step with $\{ \eta _ { t , g } \} _ { g = 1 } ^ { G } ;$   
6 Compute group-wise final rewards $\{ r _ { t + 1 , g } ^ { \mathrm { f i n a l } } \} _ { g = 1 } ^ { G }$ and instability indicator ψ<sub>t+1</sub>;   
7 Store $( s _ { t } , u _ { t } , a _ { t } , \ell _ { t } ^ { \mathrm { o l d } } , r _ { t + 1 } ^ { \mathrm { f i n a l } } , s _ { t + 1 } )$ , retaining the group axis, in the rollout buffer;   
// 4. Policy Update & Safety Net   
8 if the PPO update condition is satisfied or $\psi _ { t + 1 } = 1$ then   
9 Update π<sub>θ</sub> and $V _ { \phi }$ using PPO;   
10 if $\cdot \psi _ { t + 1 } = 1$ then   
// Circuit-Breaker Triggered   
11 Emit abort signal;   
12 The outer training loop restores model and optimizer states from the latest safe checkpoint;

## 4.5 IMPLEMENTATION DETAILS AND OVERHEAD

SOLAR uses a two-layer MLP actor–critic trained with PPO; Appendix A.2 gives the architecture.   
Appendix C.14 separates steady-state step time from full-run wall-clock accounting.

## 5 EXPERIMENTS

We ask four questions: Does SOLAR improve final pretraining outcomes? Which control choices make online learning effective? Does the residual policy transfer across scales and base schedules? What computational cost does the controller add?

## 5.1 MAIN PRETRAINING RESULTS

## 5.1.1 DENSE LLAMA 2 PRETRAINING ACROSS OPTIMIZERS

We evaluate on C4 (Raffel et al., 2020) across four Llama 2 (Touvron et al., 2023b) scales (60M to 1B) with AdamW and Muon, following the setup of Zhao et al. (2024). Static schedules, online tuners, and schedule-free baselines share the model, data, optimizer, and training budget. Table 1 reports validation PPL at the final training update. SOLAR-specific hyperparameters are selected once at 60M and then kept fixed. Appendix B.2 gives the complete selection and evaluation protocol.

Both SOLAR modes outperform every non-SOLAR alternative wherever they are evaluated. At 1B, SOLAR-online changes AdamW+Cosine from 16.52 to 14.83 and Muon+Cosine from 14.36 to 13.79. At the larger target scales, SOLAR-frozen further lowers five of the six seed-52 online results by reusing the source-acquired controller. Blockwise LR does not consistently improve the tuned base, while AutoLRS and MECHANIC trail the static schedules in most settings. Schedule-Free AdamW (Defazio et al., 2024) improves slightly over Cosine at all four scales and is the strongest non-SOLAR alternative at 350M, yet remains above SOLAR. The gains persist across the repeated runs summarized in Appendix C.2.

Table 1: Seed-52 final-checkpoint validation perplexity on C4. SOLAR-online acquires its policy within the target run; SOLAR-frozen reuses a policy acquired from full-length online runs at 60M, with no target PPO updates. AvgLR Replay replays SOLAR’s step-wise mean LR. † marks official AdamW-based implementations. Multi-seed results appear in Appendix C.2.
<table><tr><td rowspan="2">Method</td><td colspan="4">Model Scale (Validation Perplexity ↓)</td></tr><tr><td>60M</td><td>130M</td><td>350M</td><td>1B</td></tr><tr><td>Base Optimizer: AdamW</td><td></td><td></td><td></td><td></td></tr><tr><td>Cosine</td><td>30.49</td><td>24.52</td><td>18.31</td><td>16.52</td></tr><tr><td>AvgLR Replay</td><td>30.68</td><td>25.61</td><td>21.69</td><td>20.06</td></tr><tr><td>WSD</td><td>29.80</td><td>23.96</td><td>18.75</td><td>16.29</td></tr><tr><td>CLR</td><td>31.78</td><td>28.14</td><td>21.43</td><td>20.22</td></tr><tr><td>†Blockwise LR</td><td>31.06</td><td>24.38</td><td>18.56</td><td>16.74</td></tr><tr><td>AutoLRS</td><td>31.46</td><td>25.02</td><td>20.07</td><td>19.08</td></tr><tr><td>MECHANIC</td><td>32.14</td><td>26.95</td><td>20.60</td><td>19.25</td></tr><tr><td>†Prodigy</td><td>45.27</td><td>27.51</td><td>22.14</td><td>20.75</td></tr><tr><td>†Schedule-Free</td><td>30.28</td><td>24.49</td><td>18.14</td><td>16.45</td></tr><tr><td>SOLAR (online)</td><td>28.92</td><td>22.79</td><td>17.23</td><td>14.83</td></tr><tr><td>SOLAR (frozen)</td><td></td><td>22.41</td><td>17.37</td><td>14.63</td></tr><tr><td colspan="5">Base Optimizer: Muon</td></tr><tr><td>Cosine</td><td>29.20</td><td>22.55</td><td>16.87</td><td>14.36</td></tr><tr><td>AvgLR Replay</td><td>30.23</td><td>24.41</td><td>18.99</td><td>17.07</td></tr><tr><td>WSD</td><td>29.18</td><td>22.47</td><td>16.62</td><td>14.31</td></tr><tr><td>CLR</td><td>30.79</td><td>26.15</td><td>19.19</td><td>18.62</td></tr><tr><td>AutoLRS</td><td>30.07</td><td>23.63</td><td>18.95</td><td>16.46</td></tr><tr><td>MECHANIC</td><td>31.04</td><td>25.52</td><td>18.88</td><td>16.68</td></tr><tr><td>SOLAR (online)</td><td>28.63</td><td>21.99</td><td>16.35</td><td>13.79</td></tr><tr><td>SOLAR (frozen)</td><td></td><td>21.87</td><td>16.22</td><td>13.54</td></tr></table>

## 5.1.2 MOE PRETRAINING UNDER NON-STATIONARY DYNAMICS

MoE training introduces routing-induced stochasticity and sparse, uneven gradient updates, which often require conservative optimization settings (Fedus et al., 2022; Zoph et al., 2022). We evaluate SOLAR on Qwen2-MoE 1B (Yang et al., 2024) pretrained on The Pile under the setup in Appendix B.4. Figure 3 shows a final PPL improvement from 9.61 to 9.34. SOLAR uses a higher average LR while producing a lower, smoother gradient-norm trace than Cosine in this run. At the module level (Figure 3d), the effective LRs fluctuate around a decaying profile, with distinct actions across groups. Section 5.3 examines these traces. Appendix C.12 reports the separate 3B DeepSeek-V2-style MoE run in Megatron, where final PPL changes from 10.73 to 10.38.

Runtime and Circuit-Breaker usage. No main-table or MoE run triggers rollback. On 1B AdamW, online SOLAR adds 1.23% to full-run wall-clock time and 1.27% to steady-state step time; the frozen mode adds 0.76% and 0.81%, respectively. Appendix C.14 defines both measurements and reports the remaining settings.

## 5.2 FROZEN RESIDUAL-POLICY TRANSFER ACROSS MODEL SCALES

The residual parameterization expresses actions relative to the current base LR. We test whether this dimensionless policy can be acquired on a small proxy and reused at larger dense scales.

Setup. We train SOLAR on five 60M pretraining trajectories of 11K updates. The policy is then frozen and applied to 130M, 350M, and 1B models within the same optimizer family. Each target run combines the frozen policy with its base schedule, performs no PPO updates, and does not retune SOLAR-specific hyperparameters. All source-policy parameters, including the final learned shared log σ, remain fixed, while group-wise actions are recomputed from the current target state at every update.

![](images/07b1471ed4c20a066e2a16e93ce1772b1a74550a54fa94d86023d47deb3425f8.jpg)  
(a) Validation PPL

![](images/ad1a5a78d052419ee1d5dbdc34358f5ca03ecce78d523665849e03db1c1efa11.jpg)  
(b) Global Grad Norm

![](images/e1f582a078d0ec5aa3a98b492f54fa18a760467505bae970acefca11cb5aaf8f.jpg)  
(c) Macro: Average LR

![](images/8cb9a4fe407915bc7e3bd18907c6e6877ba27fbf7b2cd4a56fdeb14af21b18a1.jpg)  
(d) Micro: Attention LR  
Figure 3: Qwen2-MoE 1B pretraining on The Pile. SOLAR improves validation perplexity, suppresses early gradient-norm spikes, and maintains a higher average LR through group-wise micro-modulation. Panel (d) shows one representative attention projection group; the solid line is a 200-step moving average and the shaded region visualizes the corresponding step-wise variation.

Results. SOLAR-frozen improves over the corresponding target-tuned Cosine and WSD schedules at every larger scale in Table 1. It yields the lower PPL in five of six matched seed-52 comparisons with SOLAR-online, while online acquisition yields the lower PPL under the corpus and optimizer shifts in Appendix C.3. Across repeated 130M runs, SOLAR-online and SOLAR-frozen reach mean final PPLs of 22.86 and 22.76 with AdamW, and 22.05 and 22.09 with Muon, respectively. Frozen reuse improves from 25.04 PPL at random initialization to 22.41 after five full-length 60M source runs (Appendix C.3). Appendix C.5 tests the same policy under 0.5×, 1×, and 2× target base LRs: it beats the matched Cosine run in all six settings and completes both 2× runs where Cosine diverges. A separate $\mu \mathrm { P }$ width-transfer experiment removes target-scale base-LR search and retains the improvement, while a C4-to-Pile experiment tests simultaneous scale and corpus shift.

## 5.3 MECHANISTIC EVIDENCE

Unless noted otherwise, we analyze the seed-52 130M Llama 2 run to characterize SOLAR’s structured control. Figure 4 shows that SOLAR acts along three axes: temporal adaptation, modulewise redistribution, and module-specific feedback to gradient statistics.

Control structure. Matched controls identify which design choices separate improvement from degradation. A recursive global PPO controller reaches 27.09 final PPL. Re-anchoring and bounding the same global interface lowers PPL to 23.74; replacing the global action with group-wise control reaches 22.87 (n = 2, seeds 42 and 52). Group-wise hypergradient control reaches 24.26, while an LLM adaptation of the layer-wise GANNO controller reaches 24.95. Untrained, near-deterministic, and state-blind SOLAR policies remain above 25.0. The anchored, state-conditioned global interface reaches 23.74, and group-wise control provides a further 0.87-PPL gain. Appendix C.4 gives the implementations, tuning protocol, and per-seed results.

![](images/2eaedd53a0cadd22c84dbddb9decc62759f794541eeeb10617f38856fed5b575.jpg)  
(a) Average effective LR

![](images/f4791637d982cddd930a263f0e1e270aa1d034e651c1235bf25e6afc84ed9993.jpg)  
(b) Single-group dynamics

![](images/35058361b54f08a69f7feac29082a4b72b2eeca25bbcb5291a1bed8b58f3277d.jpg)  
(c) Module-wise allocation

![](images/73ba5389fbc8f6825023b97794f134ce8056b885086e5a2e6d2897df6b20d10a.jpg)  
(d) LR–GradNorm relation  
Figure 4: Mechanistic analysis of SOLAR on 130M Llama 2. (a)–(d): average effective LR, single-group LR dynamics, module-wise LR allocation, and per-module LR–GradNorm correlation, respectively. In (b), the solid line is a 200-step moving average and the shaded region visualizes the corresponding step-wise variation.

Temporal adaptation. SOLAR maintains a higher average LR than Cosine, while AvgLR Replay ends at 25.61 PPL (Figure 4a). A normalized static group profile reaches 24.19 PPL, compared with 22.79 for SOLAR-online on the same seed. These controls show that neither a scalar temporal schedule nor a static group allocation reproduces SOLAR’s online trajectory. Appendix C.6 gives the protocols and results.

Module-wise redistribution. Figure 4c shows a stable module-level allocation pattern: vocabularyfacing components (token embeddings, LM head) receive larger multipliers, while internal attention and MLP projections are more constrained. This structure is consistent across training and suggests that SOLAR reallocates optimization capacity toward modules that appear more sensitive to learningrate variation. Such structural allocation cannot be reproduced by a scalar global schedule.

Module-specific feedback. SOLAR learns differentiated LR–GradNorm relationships (Figure 4d). Dense transformation matrices exhibit negative correlation, while several post-attention LayerNorm parameters show positive correlation. The mapping therefore changes across modules rather than applying a uniform inverse function of gradient norm.

Appendix C provides state, action, reward, Circuit-Breaker, and base-LR analyses. Appendix C.4 contains the matched learned-controller comparisons.

## 6 CONCLUSION AND LIMITATIONS

We presented SOLAR, an online LR scheduler that uses a base schedule for the coarse LR profile while adapting to the realized optimization state. Its policy learns bounded, state-dependent groupwise corrections that re-anchor to the base at every step, guided by lightweight training features and a progress-aware reward. Across dense and MoE settings, SOLAR improves final PPL over tuned static schedules and automatic LR tuners under both AdamW and Muon, with repeated gains through 1B. On matched seeds, controls show that base anchoring and action bounds improve a global PPO controller from 27.09 to 23.74 PPL, while group-wise control reaches 22.87.

The controls also clarify how the design divides the scheduling problem. Re-anchoring preserves the base scheduler’s warmup–decay profile, bounded residual actions limit the effect of individual policy decisions, and group-wise feedback allows the correction to differ across parameter tensors as the run evolves. This structure supports two complementary operating modes. SOLAR-online learns the controller within the current pretraining run, while SOLAR-frozen reuses an acquired policy without target-side PPO updates or SOLAR-specific retuning. A policy trained on a 60M proxy transfers to larger dense scales and beats matched Cosine runs across the tested fourfold base-LR range. The cross-corpus, reward, and $\mu \mathrm { P }$ results in the appendix extend the evaluation across changes in data, reward composition, and parameterization.

These results provide a practical route to adaptive LR control in LLM pretraining. The primary limitation of SOLAR is that its validation is still limited to dense models up to 1B parameters and a supplementary 3B MoE setting, so its behavior at larger scales remains untested. We leave this investigation to future work.

## REFERENCES

Marcin Andrychowicz et al. Learning to learn by gradient descent by gradient descent. In Advances in Neural Information Processing Systems (NeurIPS), 2016.

Atilim Gunes Baydin, Robert Cornish, David Martinez Rubio, Mark Schmidt, and Frank Wood. Online learning rate adaptation with Hypergradient descent. In International Conference on Learning Representations (ICLR), 2018. URL https://openreview.net/forum?id= BkrsAzWAb.

Tianlong Chen et al. Learning to optimize: A primer and a benchmark. Journal ofMachine Learning Research (JMLR), 2022.

Xiangning Chen et al. Symbolic discovery of optimization algorithms. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Aakanksha Chowdhery et al. PaLM: Scaling language modeling with pathways. Journal ofMachine Learning Research, 24(240):1–113, 2023. URL https://www.jmlr.org/papers/v24/ 22-1144.html.

Ashok Cutkosky, Aaron Defazio, and Harsh Mehta. Mechanic: A learning rate tuner. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Christian Daniel, Jonathan Taylor, and Sebastian Nowozin. Learning step size controllers for robust neural network training. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI), 2016. URL https://ojs.aaai.org/index.php/AAAI/article/view/10187.

DeepSeek-AI et al. DeepSeek-v2: A strong, economical, and efficient Mixture-of-Experts language model, 2024. URL https://arxiv.org/abs/2405.04434.

Aaron Defazio and Konstantin Mishchenko. Learning-rate-free learning by D-Adaptation. In International Conference on Machine Learning (ICML), 2023.

Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, and Ashok Cutkosky. The road less scheduled. In Advances in Neural Information Processing Systems (NeurIPS), 2024. doi: 10.52202/079017-0320. URL https://papers.nips.cc/paper\_files/paper/2024/hash/ 136b9a13861308c8948cd308ccd02658-Abstract-Conference.html.

John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal ofMachine Learning Research (JMLR), 2011. URL https: //jmlr.org/papers/v12/duchi11a.html.

William Fedus, Barret Zoph, and Noam Shazeer. Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal ofMachine Learning Research (JMLR), 2022.

Leo Gao et al. The Pile: An 800gb dataset of diverse text for language modeling, 2020. URL https://arxiv.org/abs/2101.00027.

Samantha Hansen. Using deep q-learning to control optimization hyperparameters, 2016. URL https://arxiv.org/abs/1602.04062.

Yutong He, Qiulin Shang, Xinmeng Huang, Jialin Liu, and Kun Yuan. A mathematics-inspired learning-to-optimize framework for decentralized optimization, 2024. URL https://arxiv. org/abs/2410.01700.

Shengding Hu et al. MiniCPM: Unveiling the potential of small language models with scalable training strategies. In First Conference on Language Modeling (COLM), 2024. URL https: //openreview.net/forum?id=3X2L2TFr0f.

Tianjin Huang, Ziquan Zhu, Gaojie Jin, Lu Liu, Zhangyang Wang, and Shiwei Liu. SPAM: Spike-aware Adam with momentum reset for stable LLM training. In International Conference on Learning Representations (ICLR), 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 7a70ad3d9c704fb9b81b5c69eda722dc-Abstract-Conference.html.

Max Jaderberg et al. Population based training of neural networks, 2017. URL https://arxiv. org/abs/1711.09846.

Yuchen Jin et al. AutoLRS: Automatic learning-rate schedule by Bayesian optimization on the fly. In International Conference on Learning Representations (ICLR), 2021.

Keller Jordan et al. Muon: An optimizer for hidden layers in neural networks. Blog post, 2024. URL https://kellerjordan.github.io/posts/muon/. Accessed: 2026-04-22.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

Ke Li and Jitendra Malik. Learning to optimize. In International Conference on Learning Representations (ICLR), 2017.

Jialin Liu, Xiaohan Chen, Zhangyang Wang, Wotao Yin, and HanQin Cai. Towards constituting mathematical structures for learning to optimize. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 21426–21449, 2023. URL https://proceedings.mlr.press/v202/liu23e.html.

Jingyuan Liu et al. Muon is scalable for LLM training, 2025. URL https://arxiv.org/abs/ 2502.16982.

Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations (ICLR), 2017.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations (ICLR), 2019.

Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model of large-batch training, 2018. URL https://arxiv.org/abs/1812.06162.

Luke Metz, James Harrison, C. Daniel Freeman, Amil Merchant, Lucas Beyer, James Bradbury, Naman Agrawal, Ben Poole, Igor Mordatch, Adam Roberts, and Jascha Sohl-Dickstein. VeLO: Training versatile learned optimizers by scaling up, 2022. URL https://arxiv.org/abs/ 2211.09760.

Konstantin Mishchenko and Aaron Defazio. Prodigy: An expeditiously adaptive parameter-free learner. In International Conference on Machine Learning (ICML), 2024.

Colin Raffel et al. Exploring the limits of transfer learning with a unified text-to-text Transformer. Journal ofMachine Learning Research (JMLR), 2020.

Sachin Ravi and Hugo Larochelle. Optimization as a model for few-shot learning. In International Conference on Learning Representations (ICLR), 2017.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. In International Conference on Learning Representations (ICLR), 2016.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.

Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost. In International Conference on Machine Learning (ICML), 2018.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism, 2019. URL https://arxiv.org/abs/1909.08053.

Leslie N. Smith. Cyclical learning rates for training neural networks. In 2017 IEEE Winter Conference on Applications ofComputer Vision (WACV), 2017.

Leslie N. Smith and Nicholay Topin. Super-Convergence: Very fast training of neural networks using large learning rates. In Artificial Intelligence and Machine Learning for Multi-Domain Operations Applications, volume 11006, pp. 369–386. SPIE, 2019. doi: 10.1117/12.2520589. URL https://doi.org/10.1117/12.2520589.

Jasper Snoek, Hugo Larochelle, and Ryan P. Adams. Practical Bayesian optimization of machine learning algorithms. In Advances in Neural Information Processing Systems (NeurIPS), 2012.

Shreyas Subramanian, Vignesh Ganapathiraman, and Aly El Gamal. Learned learning rate schedules for deep neural network training using reinforcement learning. In Krystal Maughan, Rosanne Liu, and Thomas F. Burns (eds.), The First Tiny Papers Track at ICLR 2023, 2023.

Sho Takase, Shun Kiyono, Sosuke Kobayashi, and Jun Suzuki. Spike no more: Stabilizing the Pre-training of large language models. In Conference on Language Modeling (COLM), 2025. URL https://arxiv.org/abs/2312.16903.

Team OLMo et al. Olmo 3, 2025. URL https://arxiv.org/abs/2512.13961.

Kale-ab Tessera, Callum Rhys Tilbury, Sasha Abramowitz, Ruan de Kock, Omayma Mahjoub, Benjamin Rosman, Sara Hooker, and Arnu Pretorius. Generalisable agents for neural network optimisation, 2023. URL https://arxiv.org/abs/2311.18598.

Tijmen Tieleman and Geoffrey Hinton. Lecture 6.5—RMSProp: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural Networks for Machine Learning, 2012. URL https://www.cs.toronto.edu/\~hinton/coursera/lecture6/lec6.pdf.

Hugo Touvron et al. LLaMA: Open and efficient foundation language models, 2023a. URL https://arxiv.org/abs/2302.13971.

Hugo Touvron et al. Llama 2: Open foundation and fine-tuned chat models, 2023b. URL https: //arxiv.org/abs/2307.09288.

Jinbo Wang, Mingze Wang, Zhanpeng Zhou, Junchi Yan, Weinan E, and Lei Wu. The Sharpness disparity principle in Transformers for accelerating language model Pre-Training. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 64859–64879, 2025. URL https://proceedings.mlr.press/ v267/wang25dl.html.

Olga Wichrowska et al. Learned optimizers that scale and generalize. In International Conference on Machine Learning (ICML), 2017.

Yuanhao Xiong, Li-Cheng Lan, Xiangning Chen, Ruochen Wang, and Cho-Jui Hsieh. Learning to schedule learning rate with graph neural networks. In International Conference on Learning Representations (ICLR), 2022. URL https://openreview.net/forum?id=k7efTb0un9z.

Chang Xu, Tao Qin, Gang Wang, and Tie-Yan Liu. Reinforcement learning for learning rate control, 2017. URL https://arxiv.org/abs/1705.11159.

Zhen Xu, Andrew M. Dai, Jonas Kemp, and Luke Metz. Learning an adaptive learning rate schedule, 2019. URL https://arxiv.org/abs/1909.09712.

An Yang et al. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671.

Greg Yang et al. Tensor programs v: Tuning large neural networks via Zero-Shot hyperparameter transfer. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Aohan Zeng et al. GLM-130b: An open bilingual pre-trained model. In International Conference on Learning Representations (ICLR), 2023. URL https://openreview.net/forum?id= -Aw0rrrPUF.

Susan Zhang et al. OPT: Open pre-trained Transformer language models, 2022. URL https: //arxiv.org/abs/2205.01068.

Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian. GaLore: Memory-efficient LLM training by gradient low-rank projection. In International Conference on Machine Learning (ICML), 2024.

Shuchen Zhu et al. Accelerating LLM Pre-Training through Flat-Direction dynamics enhancement, 2026. URL https://arxiv.org/abs/2602.22681.

Barret Zoph et al. ST-MoE: Designing stable and transferable sparse expert models, 2022. URL https://arxiv.org/abs/2202.08906.

## Appendix

## A SOLAR IMPLEMENTATION DETAILS

In this section, we provide the complete implementation details of the SOLAR scheduler, including state construction, action parameterization, PPO training configurations, and the automated Circuit-Breaker mechanism.

## A.1 STATE DEFINITION AND FEATURE EXTRACTION

This appendix expands the compact state definition in Section 4.1 by specifying normalization constants, EMA coefficients, window sizes, and missing-gradient handling.

State definition. For controlled parameter group $^ { g , }$ the scheduler state is defined as the concatenation of global optimization features and local group-specific features:

$$
s _ { t , g } = \biggl [ s _ { t } ^ { \mathrm { g l o b a l } } ; s _ { t , g } ^ { \mathrm { l o c a l } } \biggr ] .
$$

Global features. The global state captures the overall training dynamics:

$$
s _ { t } ^ { \mathrm { g l o b a l } } = \left[ \tau _ { t } , \log L _ { t } , \nu _ { t } , \Delta \mathrm { E M A } _ { t } \right] .
$$

In implementation, the normalized training progress is $\tau _ { t } = t / T$ , where $T$ is the total number of training steps. The loss term is implemented as:

$$
\log L _ { t } = \log ( L _ { t } + \epsilon ) ,
$$

where $\epsilon = 1 0 ^ { - 8 }$ is the numerical stability constant (used uniformly across all log and EMA denominator computations in this appendix). The short-window loss fluctuation is computed from a sliding window of recent losses:

$$
\nu _ { t } = \frac { \mathrm { S t d } ( \mathcal { H } _ { t } ) } { L _ { t } + \epsilon } ,
$$

where $\mathcal { H } _ { t }$ stores the most recent $w = 2 0$ losses; when fewer than 20 steps have elapsed, we use all available samples. To characterize short-term versus long-term optimization trends, we maintain two exponential moving averages (EMA) of the loss with $\beta _ { s } = 0 . 9$ and $\beta _ { \ell } = 0 . 9 9$ , initialized as EM $\mathrm { [ A _ { 0 } ^ { s h o r t } = E M A _ { 0 } ^ { l o n g } = } L _ { 0 } \mathrm { { : } }$

$$
\mathrm { E M A } _ { t } ^ { \mathrm { s h o r t } } = \beta _ { s } \mathrm { E M A } _ { t - 1 } ^ { \mathrm { s h o r t } } + ( 1 - \beta _ { s } ) L _ { t } ,\tag{7}
$$

$$
\mathrm { E M A } _ { t } ^ { \mathrm { l o n g } } = \beta _ { \ell } \mathrm { E M A } _ { t - 1 } ^ { \mathrm { l o n g } } + ( 1 - \beta _ { \ell } ) L _ { t } .\tag{8}
$$

The trend difference is then defined as:

$$
\Delta \mathrm { E M A } _ { t } = \frac { \mathrm { E M A } _ { t } ^ { \mathrm { s h o r t } } - \mathrm { E M A } _ { t } ^ { \mathrm { l o n g } } } { \mathrm { E M A } _ { t } ^ { \mathrm { l o n g } } + \epsilon } .
$$

Local features. The local state captures tensor-specific statistics:

$$
s _ { t , g } ^ { \mathrm { l o c a l } } = \bigl [ \log \eta _ { t , g } ^ { \mathrm { b a s e } } , \log \Vert \nabla _ { t , g } \Vert , a _ { t - 1 , g } , d _ { g } , \log \Vert w _ { t , g } \Vert , \Delta \log \Vert \nabla _ { t , g } \Vert \bigr ] .
$$

The base learning-rate context is log $\eta _ { t , g } ^ { \mathrm { b a s e } } = \log ( \eta _ { t , g } ^ { \mathrm { b a s e } } + \epsilon )$ and the parameter-scale feature is log $\lVert w _ { t , g } \rVert = \log ( \lVert w _ { t , g } \rVert _ { 2 } + \epsilon )$ , both using the same $\epsilon = 1 0 ^ { - 8 }$ . The gradient-magnitude feature and its recent change are computed respectively as:

$$
\log \| \nabla _ { t , g } \| = \log ( \| \nabla _ { t , g } \| _ { 2 } + \epsilon ) ,\tag{9}
$$

$$
\begin{array} { r } { \Delta \log \| \nabla _ { t , g } \| = \log ( \| \nabla _ { t , g } \| _ { 2 } + \epsilon ) - \log ( \| \nabla _ { t - 1 , g } \| _ { 2 } + \epsilon ) . } \end{array}\tag{10}
$$

The previous action $_ { a _ { t - 1 , g } }$ is the action applied to controlled group g at the previous step. The depth feature $d _ { g }$ records the normalized network depth of the module containing controlled group $g .$ If a controlled parameter group has no gradient at step t, we set gradient-dependent local entries (log $\lVert \nabla _ { t , g } \rVert , \Delta \log \parallel \nabla _ { t , g } \rVert )$ to zero and keep non-gradient entries $( \bar { \log { \eta _ { t , g } ^ { \mathrm { b a s e } } } } , a _ { t - 1 , g } ^ { - } , d _ { g } , \log { \| w _ { t , g } \| } )$ unchanged when available. Each state feature is normalized with EMA running statistics using decay 0.99 before it is passed to the policy.

## A.2 ACTION SPACE, ARCHITECTURE, AND DISTRIBUTED EXECUTION

Action Parameterization. SOLAR controls hyperparameters at the fine-grained level of controlled parameter groups. Each controlled group corresponds to one trainable parameter tensor in the dense implementation; the MoE implementations use the module partition defined by their respective codebases. For every group, the actor samples a pre-Tanh variable, forms a squashed action, and clips only the value sent to the optimizer:

$$
\begin{array} { r l } & { u _ { t , g } \sim \mathcal { N } ( \mu _ { t , g } , \sigma ^ { 2 } ) , } \\ & { \widetilde { a } _ { t , g } = \operatorname { t a n h } ( u _ { t , g } ) , } \\ & { a _ { t , g } = \operatorname { c l i p } ( \widetilde { a } _ { t , g } , - 1 + 1 0 ^ { - 4 } , 1 - 1 0 ^ { - 4 } ) . } \end{array}\tag{11}
$$

The PPO log-probability is evaluated from the stored $u _ { t , g }$ with the Tanh change-of-variables correction, before the numerical clipping that produces $\boldsymbol { a } _ { t , g }$

The scheduler applies multiplicative residuals to the underlying base scheduler. The executed learning rate for controlled parameter group g at step t is:

$$
\eta _ { t , g } = \eta _ { t , g } ^ { \mathrm { b a s e } } \cdot \exp \left( \alpha _ { t } a _ { t , g } \right) ,\tag{12}
$$

where $\eta _ { t , g } ^ { \mathrm { b a s e } }$ is the base learning rate and $\alpha _ { t }$ is the action scale. To ensure stability during the initial phase, we linearly warm up the action scale: $\alpha _ { t } = \alpha \cdot ( t / T _ { \mathrm { w a r m u p } } )$ for $t < T _ { \mathrm { w a r m u p } }$ , and $\alpha _ { t } = \alpha$ afterward, where $T _ { \mathrm { w a r m u p } } = 0 . 1 T$ across all experiments. We set the maximum action scale to $\alpha = 1 . 3$

Architecture and PPO Setup. We instantiate SOLAR with a lightweight actor-critic network. Given the state representation $s _ { t , g }$ for controlled parameter group g, a shared MLP encoder first produces a hidden representation:

$$
h _ { t , g } = f _ { \mathrm { e n c } } ( s _ { t , g } ) ,
$$

where $f _ { \mathrm { e n c } }$ is a two-layer MLP with input dimension 10 (matching the state dimension), hidden dimension 256, and Tanh activations; no layer normalization is used. This representation is then passed to separate actor and critic heads, which are implemented as linear projections:

$$
\mu _ { t , g } = W _ { \mu } h _ { t , g } + b _ { \mu } , \qquad V _ { \phi } ( s _ { t , g } ) = W _ { v } h _ { t , g } + b _ { v } .
$$

The actor policy $\pi _ { \boldsymbol { \theta } } ( \widetilde { a } _ { t , g } \mid s _ { t , g } )$ is a one-dimensional Tanh-Normal distribution with mean $\mu _ { t , g }$ and a globally learnable scalar log-standard-deviation parameter log σ shared across all controlled parameter groups.

The squashed action distribution factorizes across controlled groups,

$$
\pi _ { \boldsymbol { \theta } } ( \widetilde { \boldsymbol { a } } _ { t } \mid \boldsymbol { s } _ { t } ) = \prod _ { g = 1 } ^ { G } \pi _ { \boldsymbol { \theta } } ( \widetilde { \boldsymbol { a } } _ { t , g } \mid \boldsymbol { s } _ { t , g } ) ,\tag{13}
$$

where all factors share the actor parameters $\theta .$ For the stored pre-Tanh sample $u _ { t , g } ,$ , define

$$
\ell _ { t , g } ( \theta ) = \log \mathcal { N } \big ( u _ { t , g } ; \mu _ { \theta } ( s _ { t , g } ) , \sigma ^ { 2 } \big ) - \log \big ( 1 - \operatorname { t a n h } ^ { 2 } ( u _ { t , g } ) \big ) ,\tag{14}
$$

which is the Tanh-Normal log-probability evaluated from the stored $u _ { t , g } . \mathrm { A l l }$ action factors share θ and act on the same LLM transition. Policy learning uses a parameter-shared independent update. Let $\theta _ { \mathrm { o l d } }$ denote the behavior-policy parameters used to collect the rollout. For group $^ { g , }$ the other action factors remain at their behavior-policy samples while its local objective is evaluated under θ:

$$
J _ { g } ^ { \mathrm { i n d } } ( \theta ; \theta _ { \mathrm { o l d } } ) = \mathbb { E } _ { \widetilde { a } _ { \tau , g } \sim \pi _ { \theta } ( \cdot | s _ { \tau , g } ) } \left[ \sum _ { \tau = t } ^ { T - 1 } \gamma ^ { \tau - t } r _ { \tau + 1 , g } ^ { \mathrm { f i n a l } } \right] .\tag{15}
$$

Using group-wise GAE, the independent policy update uses the gradient estimate

$$
\widehat { g } _ { g } : = \widehat { \mathbb { E } } _ { t } \Big [ \widehat { A } _ { t , g } \nabla _ { \theta } \ell _ { t , g } ( \theta ) | _ { \theta = \theta _ { \mathrm { o l d } } } \Big ] .\tag{16}
$$

PPO applies likelihood-ratio clipping to each group objective. The group-specific ratio is

$$
\rho _ { t , g } ( \theta ) = \exp ( \ell _ { t , g } ( \theta ) - \ell _ { t , g } ( \theta _ { \mathrm { o l d } } ) ) .\tag{17}
$$

The per-group log-probabilities are not summed when computing $\rho _ { t , g }$ . Averaging the group surrogates gives

$$
\mathcal { L } _ { \mathrm { P P O } } ( \theta ) = \mathbb { E } _ { t } \left[ \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \operatorname* { m i n } \Bigl ( \rho _ { t , g } ( \theta ) \hat { A } _ { t , g } , \mathrm { c l i p } ( \rho _ { t , g } ( \theta ) , 1 - \epsilon _ { \mathrm { c l i p } } , 1 + \epsilon _ { \mathrm { c l i p } } ) \hat { A } _ { t , g } \Bigr ) \right] ,\tag{18}
$$

where $\hat { A } _ { t , g }$ is the generalized advantage estimate (GAE) (Schulman et al., 2016) for group $^ { g , }$ and $\epsilon _ { \mathrm { c l i p } }$ is the clipping threshold. The minimum and clipping operations are applied to each $( t , g )$ entry before the shared actor receives their average. The value and entropy terms use the same group-wise reduction:

$$
\mathcal { L } _ { V } ( \phi ) = \mathbb { E } _ { t } \left[ \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \left( V _ { \phi } ( s _ { t , g } ) - \hat { R } _ { t , g } \right) ^ { 2 } \right] ,\tag{19}
$$

$$
\mathcal { L } _ { H } ( \theta ) = \mathbb { E } _ { t } \left[ \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \mathcal { H } [ \pi _ { \theta } ( \cdot \mid s _ { t , g } ) ] \right] .\tag{20}
$$

The actor–critic minimizes

$$
\mathcal { I } ( \theta , \phi ) = - \mathcal { L } _ { \mathrm { P P O } } ( \theta ) + \lambda _ { v } \mathcal { L } _ { V } ( \phi ) - \lambda _ { e } \mathcal { L } _ { H } ( \theta ) ,\tag{21}
$$

where $\hat { R } _ { t , g }$ is the target return, $\lambda _ { v }$ is the value-loss coefficient, and $\lambda _ { e }$ is the entropy coefficient.

In our implementation, we optimize ${ \mathcal { I } } ( \theta , \phi )$ using the Adam optimizer with a learning rate of $3 \times 1 0 ^ { - 4 }$ . Standard PPO hyperparameters include a discount factor $\gamma = 0 . 9 9 $ , a clipping threshold $\epsilon _ { \mathrm { c l i p } } = 0 . 2$ , a value loss coefficient $\lambda _ { v } = 0 . 5$ , and an entropy coefficient $\lambda _ { e } = 0 . 0 5$ . During normal training, the RL agent collects trajectories and performs an update every 50 steps, with $K = 4$ optimization epochs per update.

For a rollout of $T _ { r }$ decision steps, states have shape $[ T _ { r } , G , d _ { s } ] ;$ ; pre-Tanh samples, executed actions, log-probabilities, ratios, returns, values, and advantages have shape $[ T _ { r } , \dot { G } ]$ . GAE is computed independently along the temporal axis for each controlled group with $\lambda _ { \mathrm { G A E } } = 0 . 9 5$ . Advantages are normalized once over the complete $[ T _ { r } , G ]$ block as adv $ ( \mathrm { a d v } - \mathrm { m e a n } ( \mathrm { a d v } ) ) / ( \mathrm { s t d } ( \bar { \mathrm { a } } \mathrm { d v } ) +$ $1 0 ^ { - 7 } )$ and reused across PPO epochs. Reward normalization and value-loss clipping are not used.

The actor-critic network weights are initialized as: all feature and hidden linear layers use orthogonal initialization with gain ${ \sqrt { 2 } } .$ , biases set to 0; the actor output layer uses constant initialization of 0.01 on weights with zero bias, so that the initial mean $\mu _ { t , g }$ ≈ 0; the critic output layer uses orthogonal initialization with gain $\sqrt { 2 }$ and zero bias; the log-standard-deviation is initialized to 0 (yielding $\sigma = 1 . 0 )$ and is learned. Because the scheduler only predicts low-dimensional multiplicative LR corrections while leaving the base optimizer’s internal mechanics unchanged, SOLAR remains highly lightweight and seamlessly integrates into standard pretraining pipelines.

Base Scheduler and Transfer Configuration. For the reported dense experiments, the base scheduler uses the Cosine peak LR listed for the corresponding model scale and optimizer in Appendix B.2. The Qwen2-MoE experiment also uses a Cosine base, while the supplementary 3B MoE experiment uses WSD as specified in Appendix C.12. In frozen residual-policy transfer, the target combines its corresponding base schedule with the source policy. The policy performs no target-scale PPO updates; the source policy’s final learned parameters, including the shared log σ, are transferred and kept fixed together with $\alpha ,$ action warmup, reward coefficients, and PPO settings.

Distributed Execution. To minimize synchronization overhead during large-scale pretraining, the RL agent operates exclusively on the rank-0 worker. The rank-0 process evaluates the policy, samples actions, and updates the PPO agent. The sampled actions are then broadcast to all other workers to ensure identical optimizer states across the distributed cluster. In the Distributed Data Parallel (DDP) setting, gradient norms are computed locally on each rank from the already-synchronized gradients (all-reduced during backward()); per-parameter norms are then broadcast from rank-0 rather than all-reduced, since the gradient values are already global and computing norms is a secondary reduction. The full procedure is summarized in Algorithm 2.

## A.3 REWARD DESIGN AND CIRCUIT-BREAKER MECHANISM

Let $L _ { t }$ and $\nabla _ { t }$ denote the loss and synchronized gradient from the forward/backward before the optimizer update at step t. The scheduler observes state $s _ { t } .$ , samples action ${ \boldsymbol { a } } _ { t } ,$ and the optimizer updates parameters. After the update, the next forward/backward produces $L _ { t + 1 }$ and post-update gradient $\nabla _ { t + 1 }$ . The reward credited to $a _ { t }$ is then computed as follows. Since the scheduler outputs controlled-parameter-group-wise learning-rate adjustments, each controlled parameter group receives a reward that combines global progress feedback with a controlled-group-wise stability shaping term:

$$
r _ { t + 1 , g } = r _ { t + 1 } ^ { \mathrm { p e r f } } + r _ { t + 1 } ^ { \mathrm { t r e n d } } - p _ { t + 1 , g } ^ { \mathrm { s t a b } } .\tag{22}
$$

Performance term. We first measure the relative improvement over the previous step:

$$
\chi _ { t + 1 } = \frac { L _ { t } } { L _ { t + 1 } + 1 0 ^ { - 1 0 } } .\tag{23}
$$

The performance term is defined as:

$$
r _ { t + 1 } ^ { \mathrm { p e r f } } = 2 0 \cdot \log ( \chi _ { t + 1 } ) .\tag{24}
$$

Trend term. We incorporate a smoother long-term signal based on the long EMA of the loss:

$$
r _ { t + 1 } ^ { \mathrm { t r e n d } } = 2 \cdot \frac { \mathrm { E M A } _ { t + 1 } ^ { \mathrm { l o n g } } - L _ { t + 1 } } { \mathrm { E M A } _ { t + 1 } ^ { \mathrm { l o n g } } + 1 0 ^ { - 8 } } .\tag{25}
$$

Controlled-group-wise stability shaping term. We compute stability feedback at the same granularity as the scheduling actions. For each controlled parameter group $g ,$ , let $\lVert \nabla _ { t + 1 , g } \rVert _ { 2 }$ denote its gradient norm from the backward after the optimizer update, and let $m _ { t + 1 , g }$ be its exponential moving average (EMA):

$$
m _ { t + 1 , g } = \beta m _ { t , g } + ( 1 - \beta ) \lVert \nabla _ { t + 1 , g } \rVert _ { 2 } ,
$$

where $\beta = 0 . 9 9$ is the smoothing coefficient. The EMA is initialized from the first observed gradient norm, $m _ { 0 , g } = \lVert \nabla _ { 0 , g } \rVert _ { 2 }$ . We define the relative gradient-growth ratio as

$$
q _ { t + 1 , g } = \frac { \lVert \nabla _ { t + 1 , g } \rVert _ { 2 } } { m _ { t + 1 , g } + \epsilon } ,\tag{26}
$$

where $\epsilon = 1 0 ^ { - 8 }$ is the numerical stability constant. The controlled-group-wise stability shaping term is

$$
\begin{array} { r } { p _ { t + 1 , g } ^ { \mathrm { s t a b } } = \left\{ \begin{array} { l l } { q _ { t + 1 , g } - 1 , } & { q _ { t + 1 , g } \leq \tau _ { \mathrm { s e v } } , } \\ { q _ { t + 1 , g } - 1 + \delta _ { \mathrm { s e v } } , } & { q _ { t + 1 , g } > \tau _ { \mathrm { s e v } } , } \end{array} \right. } \end{array}\tag{27}
$$

where $\tau _ { \mathrm { s e v } } = 3 . 0$ and $\delta _ { \mathrm { s e v } } = 2 0$

This signed term compares each group’s post-update gradient norm with its own recent trend. Since it is subtracted from the reward, $q _ { t + 1 , g } < 1$ provides positive shaping, whereas $q _ { t + 1 , g } > 1$ reduces the reward; the severe offset applies when $q _ { t + 1 , g } > \tau _ { \mathrm { s e v } }$ . Because the signal is computed after the update and indexed by controlled group, it provides group-resolved feedback aligned with the group-wise actions. The two progress terms remain shared across groups and tied to the global language-modeling objective. Since the current gradient norm also enters $m _ { t + 1 , g } , 0 \leq q _ { t + 1 , g } \leq 1 / ( 1 - \beta ) = 1 0 0$ and $- 1 \leq p _ { t + 1 , g } ^ { \mathrm { s t a b } } \leq 1 1 9$ no additional clipping is used.

Circuit-Breaker and Restart Procedure. To prevent catastrophic divergence during online exploration and ensure training stability, we implement an automated Circuit-Breaker mechanism. Unlike the controlled-group-wise stability shaping term, which provides dense feedback during normal optimization, the Circuit-Breaker is a global emergency mechanism triggered by severe loss spikes. At each step, we compute the loss spike ratio:

$$
\kappa _ { t + 1 } = \frac { L _ { t + 1 } } { L _ { t + 1 } ^ { \mathrm { e m a } } + 1 0 ^ { - 8 } } .\tag{28}
$$

If $\kappa _ { t + 1 } > \kappa _ { L }$ , the Circuit-Breaker is triggered, where $\kappa _ { L } = 1 . 5$ . The optimizer wrapper executes the following procedure:

1. Punishment: Applies the global Circuit-Breaker penalty $r _ { t + 1 , g } ^ { \mathrm { f i n a l } } = r _ { t + 1 , g } - \lambda _ { \mathrm { c b } }$ for all controlled parameter groups g, with $\lambda _ { \mathrm { c b } } = 1 0 0$

2. Forced Update: Rank 0 immediately performs a PPO update using the transitions accumulated so far, ensuring the agent promptly learns from this catastrophic failure event.

3. Abort Signal: Sets an abort\_training=True flag and broadcasts it to all workers.

The optimizer wrapper does not restore model weights. It returns an abort signal to the outer training loop, which truncates the corrupted trajectory and restores the latest safe model and optimizer state. The forced PPO update is retained rather than replaced by the checkpointed PPO state.

Hyperparameter settings. The numerical constants in (22)–(28) are fixed design choices with distinct meanings. The coefficient 20 in $r _ { t + 1 } ^ { \mathrm { p e r f } }$ scales the stepwise improvement signal, and the coefficient 2 in $r _ { t + 1 } ^ { \mathrm { t r e n d } }$ scales the long-term trend signal. The stability shaping term uses $\tau _ { \mathrm { s e v } } = 3 . 0$ and adds $\delta _ { \mathrm { s e v } } = 2 0$ when gradient growth crosses this threshold. The Circuit-Breaker uses $\kappa _ { L } = 1 . 5$ as the loss-spike trigger and $\lambda _ { \mathrm { c b } } = 1 0 0$ as the emergency penalty. In our main training runs, the Circuit-Breaker is not triggered, so $\kappa _ { L }$ and $\lambda _ { \mathrm { c b } }$ do not affect the nominal optimization trajectory. We selected this configuration once on a small proxy setting and kept it fixed across all experiments.

## A.3.1 CONTROLLER COEFFICIENT CENSUS AND REWARD LEAVE-ONE-OUT ANALYSIS

The calibration covers four reward coefficients and the short-loss EMA decay used in the state. Table 2 lists these coefficients together with the fixed state-normalization decay. Sweeping one coefficient at a time produces final PPL between 22.80 and 22.85, compared with 22.79 at the default point. The largest deviation, 0.06 PPL, is below the observed 130M SOLAR seed standard deviation of 0.08.

Table 2: Controller-coefficient census on 130M/C4/AdamW.
<table><tr><td>Quantity</td><td>Default</td><td>Status</td><td>Values evaluated</td></tr><tr><td>Short-term progress scale</td><td>20</td><td>tunable</td><td>{10, 20, 40} and removal</td></tr><tr><td>EMA trend scale</td><td>2</td><td>tunable</td><td>{1, 2, 4} and removal</td></tr><tr><td>Severe stability offset</td><td>20</td><td>tunable</td><td>{10, 20, 40} and removal</td></tr><tr><td>Severe stability threshold</td><td>3.0</td><td>tunable</td><td>{2.5, 3.0, 4.0}</td></tr><tr><td>State short-loss EMA decay  $\beta _ { s }$ </td><td>0.90</td><td>tunable</td><td>{0.90, 0.95}</td></tr><tr><td>State-normalization EMA decay</td><td>0.99</td><td>fixed</td><td>fixed across experiments</td></tr></table>

The coefficient sweep tests local calibration, while the leave-one-out experiment measures the contribution of each reward component. Removing immediate progress, trend, and stability increases final PPL by 1.94, 0.90, and 0.30, respectively. Removing stability also causes the only Circuit-Breaker activation observed under nominal hyperparameters.

Table 3: Reward leave-one-out on 130M/C4/AdamW using seeds 42 and 52.
<table><tr><td>Variant</td><td>Seed 52</td><td>Seed 42</td><td>Mean PPL</td><td> $\Delta$  vs. full</td><td>CB activations</td></tr><tr><td>Full reward</td><td>22.79</td><td>22.95</td><td>22.87</td><td>0.00</td><td>0/2</td></tr><tr><td>— immediate progress</td><td>24.91</td><td>24.71</td><td>24.81</td><td>+1.94</td><td>0/2</td></tr><tr><td>– EMA trend</td><td>23.71</td><td>23.83</td><td>23.77</td><td>+0.90</td><td>0/2</td></tr><tr><td>— stability shaping term</td><td>23.24</td><td>23.09</td><td>23.17</td><td>+0.30</td><td>1/2</td></tr></table>

The activation occurs in the no-stability seed-42 run at update 12,340. The Circuit-Breaker restores the latest safe state, after which the run completes. Removing stability raises mean PPL by 0.30 and produces the only recovery event observed under the nominal settings.

Algorithm 2: SOLAR optimizer step (PyTorch-style pseudocode)   
Input: Model parameters $\{ w _ { g } \} _ { g = 1 } ^ { G }$ , current loss $L _ { t } ,$ base scheduler $f _ { \mathrm { b a s e } } .$ , policy π<sub>θ</sub>   
// From the forward/backward on the current batch   
1 L ← BROADCASTFROMR $\operatorname { A N K } 0 ( L _ { t } ) ;$   
2 Update loss history and short/long EMAs;   
3 Compute global state $s _ { t } ^ { \mathrm { g l o b a l } }$ ;   
4 for $\overset { \overline { { \mathbf { \alpha } } } } { \boldsymbol { g } } = 1$ to G do   
5 Compute local features from $\nabla _ { t , g } , w _ { t , g } , \eta _ { t , g } ^ { \mathrm { b a s e } } , a _ { t - 1 , g } ,$ and $d _ { g }$ ;   
6 Form state $s _ { t , g } ;$   
7 end   
8 if rank == 0 then   
9 $\{ u _ { t , g } , a _ { t , g } , \ell _ { t , g } ^ { \mathrm { o l d } } \} _ { g = 1 } ^ { G }  \pi _ { \theta } . \mathrm { A C T } ( \{ s _ { t , g } \} _ { g = 1 } ^ { G } ) ;$   
10 else   
11 Allocate placeholder actions;   
12 end   
13 {a<sub>t,g</sub>}<sup>G</sup><sub>g=1</sub> ← BROADCASTFROMRANK0({a<sub>t,g</sub>}<sup>G</sup><sub>g=1</sub>);   
14 Obtain $\{ \eta _ { t , g } ^ { \mathrm { b a s e } } \} _ { g = 1 } ^ { G }$ from $f _ { \mathrm { b a s e } } ;$   
15 for $g = 1$ to G do   
16 Set $\eta _ { t , g }  \eta _ { t , g } ^ { \mathrm { b a s e } } \exp ( \alpha _ { t } a _ { t , g } ) ;$   
17 end   
18 OPTIMIZER.STEP();   
// The next standard training iteration yields $L _ { t + 1 } , \nabla _ { t + 1 } ;$ no additional   
forward/backward is run   
19 Construct next state $s _ { t + 1 }$ from $L _ { t + 1 }$ and $\nabla _ { t + 1 } ;$   
20 Compute controlled-group-wise rewards $\{ r _ { t + 1 , g } \} _ { g = 1 } ^ { G }$ using $L _ { t } , L _ { t + 1 }$ , and $\nabla _ { t + 1 } $   
21 Compute $\psi _ { t + 1 }$ using Eq. (6) and set $r _ { t + 1 , g } ^ { \mathrm { f i n a l } }  r _ { t + 1 , g } - \lambda _ { \mathrm { c b } } \psi _ { t + 1 }$ for every group g;   
22 if rank == 0 then   
23 Store $( s _ { t } , u _ { t } , a _ { t } , \ell _ { t } ^ { \mathrm { o l d } } , r _ { t + 1 } ^ { \mathrm { f i n a l } } , s _ { t + 1 } )$ , retaining the group axis, in the PPO buffer;   
24 if update interval reached or Circuit-Breaker triggered then   
25 π<sub>θ</sub>.UPDATE();   
26 end   
27 end   
28 if Circuit-Breaker triggered $( \psi _ { t + 1 } = 1 )$ then   
29 // Abort Training   
30 Trigger abort signal and broadcast to all workers;   
31 end

## B EXPERIMENTAL DETAILS

## B.1 DENSE MODEL SETUP

We follow the standardized experimental setup introduced by Zhao et al. (2024), with our own scheduler implementations integrated into the same training framework.

For the Llama 2 (Touvron et al., 2023b) models, we use C4 (Raffel et al., 2020) as the pretraining corpus. We fix the batch size to 512 and the maximum sequence length to 256 for all Llama 2 experiments.

We train the following model scales:

• 60M: 11K training steps, corresponding to 1.4B tokens;

• 130M: 20K training steps, corresponding to 2.6B tokens;

• 350M: 60K training steps, corresponding to 7.8B tokens;

• 1B: 100K training steps, corresponding to 13.1B tokens.

Table 5: Selected peak LR for each dense-model scale. All settings use weight decay 0.01 and 10% warmup. SOLAR inherits the corresponding Cosine base and does not run a separate base search.
<table><tr><td>Optimizer</td><td>60M</td><td>130M</td><td>350M</td><td>1B</td></tr><tr><td>AdamW</td><td> $3 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Muon</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 4 summarizes the architectural configurations of all Llama 2 dense models used in our experiments.

Table 4: Architecture of Llama 2 dense models.
<table><tr><td></td><td>60M</td><td>130M</td><td>350M</td><td>1B</td></tr><tr><td> $d _ { \mathrm { m o d e l } }$ </td><td>512</td><td>768</td><td>1024</td><td>2048</td></tr><tr><td> $n _ { \mathrm { l a y e r s } }$ </td><td>8</td><td>12</td><td>24</td><td>24</td></tr><tr><td> $n _ { \mathrm { h e a d s } }$ </td><td>8</td><td>12</td><td>16</td><td>32</td></tr><tr><td> $d _ { \mathrm { f f } }$ </td><td>1376</td><td>2048</td><td>2736</td><td>5461</td></tr><tr><td>Vocab</td><td>32000</td><td>32000</td><td>32000</td><td>32000</td></tr></table>

For hyperparameter tuning, we sweep the peak learning rate over the following search grid:

$$
\{ 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 3 \times 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \} .
$$

Alongside the learning rate, we also perform a grid search for the weight decay parameter over:

$$
\{ 1 0 ^ { - 2 } , 5 \times 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 2 \times 1 0 ^ { - 1 } \} .
$$

## B.2 HYPERPARAMETER SELECTION AND FINAL-PPL REPORTING

We use seed 42 to select scheduler configurations and keep the selected values fixed thereafter. Table 1 reports validation PPL at the final update of the seed-52 runs. We evaluate validation PPL every 1K updates, but earlier checkpoints are not used for reporting. Training ends after 11K, 20K, 60K, and 100K updates for the 60M, 130M, 350M, and 1B models. Table 9 lists the repeated runs separately, including the seed-42 tuning runs.

All dense methods share sequence length 256, global batch size 512, gradient clipping at 1.0, and bf16 arithmetic. AdamW uses $\beta = ( 0 . 9 , 0 . 9 9 9 )$ and $\epsilon = 1 0 ^ { - 8 }$ ; Muon uses momentum 0.95, with $\beta = ( 0 . 9 , 0 . 9 5 )$ for its AdamW-handled parameters. The common base search covers peak LR $\{ 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 3 \times 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ and weight decay {0.01, 0.05, 0.1, 0.2}. Table 5 gives the selected base settings.

Every configuration in the two-stage screen runs to 0.4T, after which the top $k = 2$ configurations continue to $\bar { T } .$ . In a completed 28-point calibration study at 60M, the interim and final rankings have Spearman correlations of $\rho = 0 . 9 6$ for AdamW and 0.93 for Muon. Advancing two configurations retains the final winner in both cases; advancing only the top interim configuration would increase the selected final PPL by 0.14.

The SOLAR search runs once at 60M and is reused under both optimizers and all target scales. The later 130M sweep evaluates sensitivity around the selected α. The one-time 24-configuration search costs 7.46 wall-clock hours (59.7 GPU-hours) on 8×RTX 4090. Five 60M meta-training trajectories cost 1.62 wall-hours (13.0 GPU-hours) for AdamW and 1.99 wall-hours (15.9 GPU-hours) for Muon. Across both optimizers, the pre-submission base searches total 616.91 wall-hours and 4,935.3 GPU-hours. AutoLRS repeats candidate forward/backward segments and costs 1.98–2.00× its matched baseline; at 1B it uses 113.47 wall-hours versus 56.75 for Cosine.

Table 6: Scheduler-specific search at 130M/C4/AdamW. Each method uses the common LR grid unless its update rule determines the LR. No non-scheduler training hyperparameter is retuned.
<table><tr><td>Method</td><td>Selected base LR / WD</td><td>Method-specific search → selected</td></tr><tr><td>Cosine</td><td> $1 0 ^ { - 3 } / 0 . 0 1$ </td><td>decay to 0.1× peak</td></tr><tr><td>WSD</td><td> $1 0 ^ { - 3 } / 0 . 0 1$ </td><td>stable fraction  $\{ 7 0 , 8 0 , 9 0 \} \%  8 0 \% ;$  decay starts at  $0 . 9 T$ </td></tr><tr><td>CLR Blockwise LR Schedule-Free</td><td> $\operatorname* { m a x } { 1 0 ^ { - 3 } } .$  base  $1 0 ^ { - 4 } / 0 . 0 1$   $1 0 ^ { - 3 } / 0 . 0 1$ </td><td>period  $\{ 1 , 2 , 4 \} \mathbf { K }  2 \mathbf { K } ;$  triangular2 segments  $\{ 2 , 3 , 4 \}  3 ;$  decay  $\{ 0 . 5 , 0 . 3 \}  0 . 5$ </td></tr><tr><td>Prodigy MECHANIC</td><td> $1 0 ^ { - 3 } / 0 . 0 1$   $\mathrm { a d a p t i v e } ~ ( d _ { 0 } = 1 0 ^ { - 6 } ) / 0 . 0 1$   $1 0 ^ { - 3 } / 0 . 0 1$ </td><td> $\beta \in \{ 0 . 9 , 0 . 9 5 , 0 . 9 8 \}  0 . 9$   $d _ { \mathrm { c o e f } } \in \{ 0 . 1 , 0 . 5 , 1 . 0 \} \to 1 . 0$  number of scales  $\{ 4 , 6 \}  6$ </td></tr></table>

Table 7: Architecture of the Qwen2-MoE 1B model.
<table><tr><td>Parameter</td><td> $d _ { \mathrm { m o d e l } }$ </td><td> $n _ { \mathrm { l a y e r s } }$ </td><td> $n _ { \mathrm { h e a d s } }$ </td><td> $n _ { \mathrm { k v \_ h e a d s } }$ </td><td> $d _ { \mathrm { f f } }$ </td><td> $d _ { \mathrm { m o e } \_ \mathrm { f f } }$ </td><td> $d _ { \mathrm { s h a r e d \_ f f } }$ </td><td> $n _ { \mathrm { e x p e r t s } }$ </td></tr><tr><td>Value</td><td>768</td><td>15</td><td>12</td><td>12</td><td>3072</td><td>768</td><td>3072</td><td>32</td></tr></table>

## B.3 DATA, SEEDS, GROUPS, AND RECOVERY STATE

The C4 validation stream contains 10M held-out tokens and uses T5TokenizerFast (t5-base, 32K vocabulary). The Pile validation stream contains 1M held-out tokens and uses LlamaTokenizerFast. Loss masks padding tokens. Training seeds are {42, 52, 62} where repeats are available, and the data-shuffle seed is 42. Each trainable tensor forms one controlled group in the dense implementation, giving $G = 9 L + 3 = 7 5 / 1 1 1 / 2 1 9 / 2 1 9$ groups from 60M to 1B.

SOLAR maintains two rolling checkpoints at a 1K-update cadence. A rollback restores model weights, optimizer state, LR-scheduler state, mixed-precision state, per-rank random-number-generator state, and data-loader position. The PPO state is not rolled back, so the forced update triggered by the failure is retained. Stale rollouts are discarded, followed by a 1K-update cooldown. A forced 8× LR spike reproduces the reference trajectory after restoration to within $3 \times 1 0 ^ { - 7 }$ and yields identical sample hashes. Checkpoint maintenance adds 0.59% wall-clock time at 1B and is included in the full-run accounting in Appendix C.14.

## B.4 MOE MODEL SETUP

Qwen2-MoE (Yang et al., 2024) is a strong open-source Mixture-of-Experts (MoE) decoder-only Transformer. Compared with the dense Llama 2 models used in our main experiments, its sparse routing mechanism introduces a more challenging optimization landscape, making it a useful testbed for evaluating online learning-rate scheduling under highly non-stationary training dynamics. For our MoE experiments, we train on The Pile (Gao et al., 2020) and build on the Qwen2-MoE codebase released by LITE (Zhu et al., 2026). To align the MoE setting with the Llama 2 experiments, we disable sliding-window attention and use a maximum sequence length of 256 with batch size 512. The model is configured to activate 4 experts per token via a top-k routing strategy $( k = 4 )$ , with 32 experts in total. To encourage stable expert utilization and mitigate representation collapse, we apply a load-balancing auxiliary loss with coefficient 0.01 and a router z-loss with coefficient 0.001. We train for 100K steps (approximately 13.1B tokens). Table 7 summarizes the full architectural configuration.

Algorithm 3: Warmup-Cosine-Decay baseline   
Input: Base optimizer, peak LR $\eta _ { \mathrm { m a x } } ,$ total steps T, warmup ratio $\rho _ { w } = 0 . 1$   
1 $\bar { T _ { w } } \bar {  } \lfloor \rho _ { w } T \rfloor , \bar { T _ { c } }  \bar { T _ { - } } T _ { w } ;$   
2 for update step $t = 0 , 1 , \dots , T - 1$ do   
3 if $t < T _ { w }$ then   
4 $\begin{array} { r } { \underline { \mathbf { \Lambda } } ( \eta _ { t } \gets \eta _ { \operatorname* { m a x } } \cdot \frac { t + 1 } { T _ { w } } ; } \end{array}$   
5 else   
6 $u \gets t - T _ { w } ;$   
7 $\begin{array} { r } { \eta _ { t }  0 . 1 \eta _ { \operatorname* { m a x } } + \frac { 1 } { 2 } ( \eta _ { \operatorname* { m a x } } - 0 . 1 \eta _ { \operatorname* { m a x } } ) [ 1 + \cos ( \pi \frac { u } { T _ { c } } ) ] ; } \end{array}$   
8 Set the optimizer learning rate to $\eta _ { t }$ and run one update;

## B.5 OPTIMIZER CONFIGURATIONS

Methods that replace or modify optimizer behavior are described here, whereas external LR schedules are described in Appendix B.6.

AdamW. All Adam-family experiments use the default PyTorch AdamW (Loshchilov & Hutter, 2019) implementation with $\dot { \beta } = \left( 0 . 9 , 0 . 9 9 9 \right)$ and $\epsilon = 1 0 ^ { - 8 }$ . The peak learning rate and weight decay are tuned via grid search as described in Appendix B.1.

Muon. Our Muon implementation follows the standard configuration described by Liu et al. (2025). It uses momentum 0.95, Nesterov acceleration, and 5 Newton-Schulz orthogonalization steps. The learning rate for matrix-shaped parameters is scaled by $0 . 2 \sqrt { \operatorname* { m a x } ( A , B ) }$ , where $( A , B )$ are the parameter dimensions. Scalar and embedding parameters are handled by an internal AdamW with $\beta = ( 0 . 9 , 0 . 9 5 )$ and $\epsilon = 1 0 ^ { - 8 }$

Prodigy. We use the official Prodigy (Mishchenko & Defazio, 2024) implementation without an external learning-rate scheduler. We select $d _ { \mathrm { c o e f } }$ using the grid in Table 6 with weight decay fixed at 0.01; the remaining hyperparameters use their defaults. Because the released implementation is AdamW-based, we report Prodigy only under the AdamW group and omit it from Muon.

Schedule-Free AdamW. Schedule-Free AdamW (Defazio et al., 2024) is evaluated using the official implementation. We tune the base learning rate on the common validation grid and search $\beta \in \{ 0 . 9 , \bar { 0 } . 9 5 , 0 . 9 8 \}$ , selecting $\beta = 0 . 9 ;$ weight decay is fixed at 0.01 and the remaining settings follow the official recipe. Because the released implementation is AdamW-based, we report Schedule-Free only under the AdamW group and omit it from Muon.

## B.6 SCHEDULER BASELINES

We compare SOLAR against the scheduler baselines used in the main paper. All baselines are implemented in the same training codebase and evaluated under the same data, model, batch size, sequence length, and optimizer. Unless otherwise specified, all warmup-decay schedules use the same minimum LR ratio 0.1. For methods with additional hyperparameters, we tune them using the same validation setup.

Cosine schedule. Warmup-Cosine-Decay (Cosine) (Loshchilov & Hutter, 2017) uses a linear warmup during the first 10% of training and then applies cosine annealing to decay the learning rate from $\eta _ { \mathrm { m a x } } \tan { 0 . 1 \eta _ { \mathrm { m a x } } }$ over the remaining steps. We tune the peak learning rate and weight decay within the same validation setting as the other baselines.

AvgLR Replay baseline. This diagnostic baseline is placed immediately after Cosine because it replays SOLAR’s scalar average trajectory on the same base scheduling interface, serving as a mechanism test rather than a conventional strong baseline. To construct it, we first record the realized average learning-rate trajectory of a SOLAR run. At each step $t ,$ we compute the unweighted mean across controlled parameter groups:

Algorithm 4: Warmup-Stable-Decay baseline   
Input: Base optimizer, peak LR $\eta _ { \mathrm { m a x } } ,$ total steps T   
1 $\bar { T _ { w } }  \lfloor 0 . 1 T \rfloor , T _ { s }  \lfloor \bar { 0 . 8 T } \rfloor , T _ { d }  T - T _ { w } - \bar { T } _ { s } ;$   
2 for update step $t = 0 , { \bar { 1 } } , \dots , T - 1$ do   
3 if $t < T _ { w }$ then   
4 $\begin{array} { r } { \eta _ { t }  \eta _ { \operatorname* { m a x } } \cdot \frac { t + 1 } { T _ { w } } ; } \end{array}$   
5 else if ${ \mathrm { \Pi } } t < T _ { w } + T _ { s }$ then   
6 $\lfloor \eta _ { t } \gets \eta _ { \mathrm { m a x } } ;$   
7 else   
8 u $ t - ( T _ { w } + T _ { s } ) ;$   
9 $\begin{array} { r } { \eta _ { t }  \eta _ { \operatorname* { m a x } } - ( \eta _ { \operatorname* { m a x } } - 0 . 1 \eta _ { \operatorname* { m a x } } ) \frac { u + 1 } { T _ { d } } } \end{array}$ ;   
10 Set the optimizer learning rate to $\eta _ { t }$ and run one update;

Algorithm 5: CLR   
Input: Base optimizer, peak LR $\eta _ { \mathrm { m a x } } ,$ lower ratio r, warmup steps $T _ { w } ,$ half-cycle length s   
$\eta _ { \mathrm { m i n } }  r \eta _ { \mathrm { m a x } } ;$   
2 for update step $t = 0 , \ldots , T - 1$ do   
3 i $\dot { \cdot } t < T _ { w }$ then   
4 $\begin{array} { r } { \eta _ { t }  \eta _ { \operatorname* { m i n } } + ( \eta _ { \operatorname* { m a x } } - \eta _ { \operatorname* { m i n } } ) \frac { t + 1 } { T _ { w } } } \end{array}$   
5 else   
6 $\begin{array} { r } { u  t - T _ { w } , c  \lfloor \frac { u } { 2 s } \rfloor ; } \end{array}$   
7 $\begin{array} { r } { x  \vert \frac { u } { s } - 2 c - 1 \vert , \overline { { a } }  \operatorname* { m a x } ( 0 , 1 - x ) ; } \end{array}$   
8 $a  a / 2 ^ { c } ;$   
9 $\eta _ { t } \gets \dot { \eta } _ { \mathrm { m i n } } + ( \eta _ { \mathrm { m a x } } - \eta _ { \mathrm { m i n } } ) a ;$   
10 Set the optimizer learning rate to $\eta _ { t }$ and run one update;

$$
\bar { \eta } _ { t } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \eta _ { t , g } ^ { \mathrm { S O L A R } } .
$$

We then replay this scalar trajectory step by step during baseline evaluation and assign the same learning rate to every controlled parameter group:

$$
\eta _ { t , g } = { \bar { \eta } } _ { t } , \quad \forall g .
$$

This baseline uses the SOLAR trajectory directly and does not apply any group-wise weighting.

WSD schedule. Warmup-Stable-Decay (WSD) (Hu et al., 2024) uses linear warmup during the first 10% of training, a stable plateau for the next 80%, and a final linear decay during the last 10%. During the decay phase, the learning rate decreases from $\eta _ { \mathrm { m a x } } \tan 0 . 1 \eta _ { \mathrm { m a x } } .$

CLR schedule. CLR (Smith, 2017) cyclically varies the learning rate between a lower and an upper bound. We implement CLR as a schedule that can be plugged into the base optimizer used in each experiment, so the training loop uses the same optimizer.step() interface as the other baselines. We use a linear warmup from $\eta _ { \mathrm { m i n } }$ to $\eta _ { \mathrm { m a x } }$ , followed by a cyclic schedule. Unless otherwise stated, we use $\eta _ { \mathrm { m i n } } = 0 . 1 \eta _ { \mathrm { m a x } } .$ , the triangu $\tt _ { - a r 2 }$ policy, and a half-cycle length of 1000 optimizer update steps. For hyperparameter tuning, we search the peak learning rate, weight decay, and scheduler-specific settings under the same validation setting as the other baselines.

Blockwise LR. Blockwise LR (Wang et al., 2025) assigns block-type-specific learning rates based on the transformer sharpness disparity principle. We use the official AdamW-based implementation, tune the peak learning rate on the common validation grid, and search the number of segments over $\{ 2 , 3 , 4 \}$ and the decay factor over {0.5, 0.3}, selecting three segments and a decay factor of 0.5; weight decay is fixed at 0.01. Because the released implementation is AdamW-based, we report Blockwise LR only under the AdamW group and omit it from Muon.

Algorithm 6: AutoLRS   
Input: Base optimizer, LR interval $[ \eta _ { \mathrm { m i n } } , \eta _ { \mathrm { m a x } } ] ,$ , number of BO trials K, stage length τ, probe   
length $\tau _ { 0 }$   
1 while training is notfinished do   
2 Buffer the next τ update batches and save model, optimizer, and RNG states;   
3 Reset the Bayesian optimizer for the current stage;   
4 for $i = 1 , \ldots , K$ do   
5 Suggest candidate $\eta _ { i }$ by LCB acquisition with exploration coefficient κ over   
$[ \log \eta _ { \mathrm { m i n } } , \log \eta _ { \mathrm { m a x } } ] ;$   
6 Restore the saved state;   
7 Train for $\tau _ { 0 }$ buffered update steps with global LR $\eta _ { i } ;$   
8 Record the short-horizon loss curve;   
9 Fit the exponential forecaster and predict the stage loss $\hat { \ell } _ { i } ;$   
10 Update the GP with $( \log \eta _ { i } , \hat { \ell } _ { i } ) ;$   
11 Select $i ^ { \star } =$ arg min $\hat { \ell } _ { i } ;$   
12 Restore the saved state and train the real model for τ steps with global LR $\eta _ { i ^ { \star } }$ ;   
13 Increase τ up $\mathrm { t o } \tau _ { \mathrm { m a x } }$ and set $\tau _ { 0 } \gets \operatorname* { m a x } ( 1 , \tau / 1 0 )$ ;

AutoLRS schedule. AutoLRS (Jin et al., 2021) selects a piecewise-constant learning rate online through repeated short-horizon candidate evaluation. At the beginning of each stage, AutoLRS buffers the next τ update batches, checkpoints the model, optimizer, and random-number-generator states, and evaluates K candidate learning rates. Each candidate is trained for only $\tau _ { 0 }$ update steps from the same checkpoint and on the same buffered data. The short-horizon loss curve is then summarized by an exponential forecaster and used as the objective for Bayesian optimization. After the K trials, AutoLRS restores the original checkpoint and trains the real model for the full τ-step stage using the candidate LR with the lowest predicted loss.

We use $K = 1 0 , \kappa = 1 0 0 0$ (the LCB exploration coefficient), $\tau _ { \mathrm { i n i t } } = 1 0 0 0 , \tau _ { 0 , \mathrm { i n i t } } = 1 0 0$ , and $\tau _ { \mathrm { m a x } } = 8 0 0 0$ . The stage length doubles after each stage until it reaches $\tau _ { \mathrm { m a x } } ,$ and $\tau _ { 0 }$ is set to one tenth of the current stage length. During early stages, the objective is the training loss curve; once the stage length reaches $\tau _ { \mathrm { m a x } } ,$ we switch to a fixed validation subset of 4 pre-cached C4 validation batches. AutoLRS assigns the selected LR to all parameter groups.

MECHANIC schedule. MECHANIC (Cutkosky et al., 2023) learns a single model-wide scale that rescales the cumulative update direction produced by the base optimizer. The method maintains a reference copy of the initial parameters, a cumulative base-update direction, and a collection of tuner states that determine the next global scale. At each step, MECHANIC first applies one update of the base optimizer, then computes a scalar feedback signal from the current gradient, the old cumulative update direction, and global norms. This feedback is used to update the tuner states and obtain a new scale $S _ { t + 1 }$ . Finally, the parameters are reparameterized around the reference point as

$$
x _ { t + 1 } = x _ { \mathrm { r e f } } + S _ { t + 1 } \Delta _ { t + 1 } .
$$

We use the official MECHANIC implementation under the same training setup as the corresponding baseline. The base learning rate and weight decay follow the common validation search, and Table 6 reports the number-of-tuners search. The remaining MECHANIC settings follow the official implementation.

## B.7 HARDWARE

All dense and Qwen2-MoE experiments up to 1B parameters run on 8×NVIDIA RTX 4090 GPUs. The DeepSeek-V2-style 3B MoE experiment runs on 32×NVIDIA A800 80GB GPUs. All dense

and 1B MoE wall-clock and overhead measurements use the RTX 4090 system; the 3B MoE timing uses the A800 system.

## C ADDITIONAL ABLATION AND EXPERIMENTAL RESULTS

Unless otherwise specified, the following ablations use the seed-52 run of a 130M-parameter Llama 2 model trained on C4 with AdamW under Appendix B.1. They isolate the state, action, reward, control granularity, safety, and transfer choices used by SOLAR.

## C.1 POSITIONING AMONG LEARNED LR CONTROLLERS

Table 8 distinguishes the object learned by each controller and the runs used to acquire it. Prior work establishes RL-based global scaling, state-conditioned actions, transfer, and layer-wise control. SOLAR targets a different acquisition regime: the controller learns during the same full autoregressive LLM pretraining trajectory that it must improve.

Table 8: Positioning among representative learned LR controllers. “Same live run” means that policy updates occur inside the final run being improved.
<table><tr><td>Method</td><td>Learned object</td><td>Acquisition</td><td>Evaluated setting</td></tr><tr><td>Daniel et al. (2016)</td><td>global step size</td><td>repeated episodes</td><td>small networks</td></tr><tr><td>Xu et al. (2017)</td><td>global absolute LR</td><td>actor-critic with resets</td><td>vision</td></tr><tr><td>Xu et al. (2019)</td><td>global LR profile</td><td>past complete histories</td><td>Fashion-MNIST, CIFAR-10</td></tr><tr><td>GNS (2022)</td><td>global graph policy</td><td>separate target episodes</td><td>vision, GLUE</td></tr><tr><td>Subramanian et al. (2023)</td><td>PPO LR schedule</td><td>separate controller training</td><td>MNIST, CIFAR-100</td></tr><tr><td>GANNO (2023)</td><td>layer-wise absolute LR</td><td>separate environments</td><td>vision</td></tr><tr><td>SOLAR-online</td><td>bounded group residual</td><td>same live run</td><td>autoregressive LLM pretraining</td></tr><tr><td>SOLAR-frozen</td><td>bounded group residual</td><td>source proxy runs</td><td>dense scale/corpus transfer</td></tr></table>

## C.2 REPEATED FINAL-CHECKPOINT RESULTS

Table 9 lists final PPL for every repeated run and reports the mean across available seeds. Hyperparameters remain fixed after selection. The two SOLAR-frozen rows reuse one source-trained policy across all three target seeds.

Table 9: Final-checkpoint PPL by seed. Seed 42 is the selection seed for online and baseline configurations; seeds 52 and 62 use the selected configuration. SOLAR-frozen reuses one source-trained policy across all target seeds.
<table><tr><td>Setting</td><td>Method</td><td>Seed 42</td><td>Seed 52</td><td>Seed 62</td><td>Mean PPL</td></tr><tr><td rowspan="3">60M AdamW</td><td>Cosine</td><td>30.71</td><td>30.49</td><td>30.34</td><td>30.51</td></tr><tr><td>WSD</td><td>29.63</td><td>29.80</td><td>29.91</td><td>29.78</td></tr><tr><td>SOLAR-online</td><td>28.99</td><td>28.92</td><td>29.14</td><td>29.02</td></tr><tr><td rowspan="4">130M AdamW</td><td>Cosine</td><td>24.41</td><td>24.52</td><td>24.57</td><td>24.50</td></tr><tr><td>WSD</td><td>24.07</td><td>23.96</td><td>23.90</td><td>23.98</td></tr><tr><td>SOLAR-online</td><td>22.95</td><td>22.79</td><td>22.84</td><td>22.86</td></tr><tr><td>SOLAR-frozen</td><td>22.96</td><td>22.41</td><td>22.90</td><td>22.76</td></tr><tr><td rowspan="3">130M Muon</td><td>Cosine</td><td>22.61</td><td>22.55</td><td>22.44</td><td>22.53</td></tr><tr><td>SOLAR-online</td><td>22.11</td><td>21.99</td><td>22.06</td><td>22.05</td></tr><tr><td>SOLAR-frozen</td><td>22.24</td><td>21.87</td><td>22.16</td><td>22.09</td></tr><tr><td rowspan="2">350M AdamW</td><td>Cosine</td><td>18.42</td><td>18.31</td><td>18.47</td><td>18.40</td></tr><tr><td>SOLAR-online</td><td>17.19</td><td>17.23</td><td>17.12</td><td>17.18</td></tr><tr><td rowspan="2">1B AdamW</td><td>Cosine</td><td>16.44</td><td>16.52</td><td>一</td><td>16.48</td></tr><tr><td>SOLAR-online</td><td>14.90</td><td>14.83</td><td>一</td><td>14.87</td></tr></table>

SOLAR is lower in every paired 350M and 1B run. The paired gaps are 1.23/1.08/1.35 PPL at 350M and 1.54/1.69 at 1B. At 130M, online/frozen mean PPLs are 22.86/22.76 under AdamW and 22.05/22.09 under Muon.

## C.3 ONLINE ACQUISITION AND FROZEN REUSE

SOLAR supports in-run acquisition and amortized reuse of the same controller design. SOLARonline updates the policy inside the target run. SOLAR-frozen reuses a policy acquired over K full-length 60M online runs and performs no target PPO updates. Freezing fixes all source-policy parameters, including the final learned shared log σ, not the action sequence: the controller continues to recompute group-wise actions from each target state.

Table 10: Source-side acquisition and 130M/C4/AdamW frozen reuse. Every target evaluation uses seed 52. The K = 0 row keeps the policy at its random initialization.
<table><tr><td>Full-length 60M source runs K</td><td>Frozen final PPL</td></tr><tr><td>0</td><td>25.04</td></tr><tr><td>1</td><td>23.26</td></tr><tr><td>2</td><td>22.89</td></tr><tr><td>3</td><td>22.58</td></tr><tr><td>5</td><td>22.41</td></tr></table>

Frozen reuse improves as the policy is acquired over more source runs, moving from 25.04 PPL at K = 0 to 22.41 at K = 5. Each source run is a complete online pretraining trajectory.

Table 11: Online acquisition and frozen reuse across source–target settings. Rows with C4 targets use seed 52; the Pile-target row reports the mean of seeds 42 and 52.
<table><tr><td>Source → target</td><td>Cosine</td><td>SOLAR-frozen</td><td>SOLAR-online</td></tr><tr><td>60M/C4/AdamW → 130M/C4/AdamW</td><td>24.52</td><td>22.41</td><td>22.79</td></tr><tr><td>60M/C4/Muon → 130M/C4/Muon</td><td>22.55</td><td>21.87</td><td>21.99</td></tr><tr><td>60M/C4/AdamW → 130M/Pile/AdamW</td><td>13.96</td><td>13.21</td><td>12.99</td></tr><tr><td>60M/C4/AdamW → 130M/C4/Muon</td><td>22.55</td><td>22.49</td><td>21.99</td></tr></table>

Both SOLAR modes remain below Cosine in the optimizer-matched rows. Frozen reuse reaches the lower PPL on the matched C4 targets, while online acquisition reaches the lower PPL under the corpus and optimizer changes.

## C.4 MATCHED LEARNED-CONTROLLER COMPARISONS

The 130M/C4/AdamW controls in Table 12 share 20K updates, weight decay 0.01, and a tuned peak LR of $1 0 ^ { - 3 }$ where a base schedule is used. N1–N3 are design-axis probes; N4 is an independent LLM adaptation of GANNO with its own controller grid. Each control is tuned on its own search space.

Implementations and tuning protocol. N1 and N2 use the same global, state-conditioned PPO controller with G = 1. N1 applies a bounded multiplier that re-anchors to the shared base schedule at every update; N2 compounds each multiplier on the preceding LR. N3 retains the anchored groupwise interface but replaces PPO with a myopic Baydin-style hypergradient rule. These three controls are trained directly in the 130M target run, and each is searched on its own controller grid. N4 follows the GANNO control structure in a separate implementation: a shared recurrent IPPO actor–critic serves 15 module-level agents, chooses among nine categorical actions on cumulative absolute LRs around $\eta ^ { * } = 1 0 ^ { - 3 }$ every 50 updates, and uses counterfactual difference rewards. Its policy is meta-trained at 60M, frozen, and evaluated at 130M with no target PPO updates; its controller is searched on its own grid. N1–N3 isolate individual design axes, whereas N4 is a complete competing controller adapted to the LLM setting.

Table 12: Matched controls that isolate the SOLAR control structure. The matched SOLAR/N1/N2 comparisons use seeds 42 and 52; each arrow gives the mean. Three-run static-baseline summaries include the sample standard deviation.
<table><tr><td>Method and control structure</td><td>Per-seed PPL → summary</td><td>Isolated axis</td></tr><tr><td>SOLAR-online: anchored group residual, G = 111</td><td>22.95/22.79 → 22.87</td><td>full method</td></tr><tr><td>N1 Global-PPO-Residual: anchored, G = 1</td><td>23.68/23.80 → 23.74</td><td>group granularity</td></tr><tr><td>WSD: tuned static base</td><td>24.07/23.96/23.90 → 23.98 ± 0.09</td><td>learned control</td></tr><tr><td>N3 Group-MHD: anchored, group-wise, non-RL</td><td>24.31/24.21 → 24.26</td><td>learned feedback</td></tr><tr><td>Cosine: tuned static base</td><td>24.41/24.52/24.57 → 24.50 ± 0.08</td><td>learned control</td></tr><tr><td>N4 GANNO-IPPO: layer-wise absolute LR</td><td>24.89/25.01 → 24.95</td><td>residual design</td></tr><tr><td>N2 Global-PPO-Recursive: unanchored, G = 1</td><td>27.30/26.88 → 27.09</td><td>anchoring and bounds</td></tr></table>

N1 and N2 use the same global, state-conditioned PPO interface. On the common seeds 42 and 52, re-anchoring each multiplier to the base and bounding the action improves 27.09 to 23.74, a 3.35-PPL change. Moving from one global action to G = 111 group actions further improves PPL to 22.87; the third SOLAR run gives a three-run mean of 22.86 ± 0.08 in Table 9. N3 shows that a group-wise residual without a learned state-conditioned loop does not recover the same gain. N4 provides a layer-wise RL controller adapted to the same LLM setting. Under 60M-to-130M frozen-policy transfer, N4 reaches 24.95 PPL and SOLAR-frozen reaches a three-run mean of 22.76, a 2.19-PPL gap between the two LLM controller adaptations. Within SOLAR, replacing the residual action with a target-trained absolute-LR action changes 22.79 to 24.74, a 1.95-PPL gap. With the SOLAR architecture held fixed, untrained, near-deterministic, and state-blind policies reach 25.04, 25.03, and 25.21; AvgLR Replay reaches 25.61 at the final checkpoint. Together, these controls isolate the contribution of the anchored residual interface, state-conditioned learning, and group-wise actions.

Failure-mode diagnostics. The N1/N2 pairing exposes a slow failure that a loss-spike safeguard does not detect. In an α = 2.6 stress run (n = 1), N2 produces no loss spike: its recursively updated

LR drifts to a $5 . 7 \times$ mean multiplier and finishes at 79.58 PPL. N3 exposes a second failure mode. Moving its single hypergradient coefficient by one decade changes PPL at the $0 . 4 T$ screen from 31.05 to 47.31, showing sharp sensitivity to its local update scale.

Action-bound sensitivity. The main setting α = 1.3 is inherited from the 60M search; Table 13 evaluates sensitivity around this setting at 130M. Performance remains better than Cosine over a twofold range, and the Circuit-Breaker first activates at $\alpha = 1 . 8$

Table 13: Sensitivity to the residual action bound on 130M/C4/AdamW.
<table><tr><td>α</td><td>0.8</td><td>1.0</td><td>1.3</td><td>1.6</td><td>1.8</td></tr><tr><td>Final PPL</td><td>23.42</td><td>23.05</td><td>22.79</td><td>22.94</td><td>23.61</td></tr><tr><td>CB activation</td><td>no</td><td>no</td><td>no</td><td>no</td><td>yes</td></tr></table>

## C.5 FROZEN-POLICY ROBUSTNESS AND $\mu \mathrm { P }$ COMPATIBILITY

Target base-LR sweep. We freeze a 60M/C4 policy and apply it with no target PPO updates. Table 14 varies the target peak LR over a fourfold window. Each cell is a single seed-52 run; the experiment measures sensitivity to the supplied base LR rather than run-to-run variance.

Table 14: Frozen residual-policy transfer under misspecified target base LRs.
<table><tr><td>Base peak</td><td>130M Cosine</td><td>130M frozen</td><td>350M Cosine</td><td>350M frozen</td></tr><tr><td>0.5×</td><td>25.05</td><td>22.72</td><td>18.79</td><td>17.61</td></tr><tr><td>1.0×</td><td>24.52</td><td>22.41</td><td>18.31</td><td>17.37</td></tr><tr><td>2.0×</td><td>diverged @4.1K</td><td>22.68</td><td>diverged @6.8K</td><td>17.96</td></tr></table>

The frozen policy beats the same-base Cosine run in all six settings and completes both 2× runs. At 130M, the negative-action fraction changes from 0.29 to 0.47 to 0.73 as the base rises from 0.5× to 2×, showing bidirectional state-conditioned correction rather than a fixed positive multiplier.

The main-table protocol searches LR/WD at each target scale, at costs of 14.02, 35.66, and 227.65 wall-clock hours for 130M, 350M, and 1B. Freezing reuses the resulting base schedule and removes target PPO training and target-scale search over SOLAR-specific hyperparameters. The avoided controller-side work is 0.9×, 2.7×, and 2.7× the corresponding LR/WD-search cost.

$\mu \mathbf { P }$ width transfer. We combine SOLAR with maximal-update parameterization (µP) (Yang et al., 2021). A 71M base and 130M target differ only in hidden width; both use 12 layers, head dimension 64, G = 111, sequence length 256, global batch 512, 20K updates, weight decay 0.01, and 10% warmup. A three-point search at 71M selects $\eta ^ { * } = 2 \times 1 0 ^ { - 3 }$ $\mu \mathrm { P }$ transfers per-tensor target LRs from fan-in: $2 . 0 0 0 \times 1 0 ^ { - 3 }$ for embeddings/RMSNorm and approximately $1 . 3 3 \bar { 3 } \times 1 0 ^ { - 3 }$ for attention, MLP, and readout tensors. No LR/WD search occurs at 130M.

Table 15: SOLAR on a directly searched standard-parameterization (SP) base and a $\mu \mathrm { P } \cdot$ -transferred base at 130M/C4/AdamW. For the two-run $\mu \mathrm { P }$ settings, values before the arrow are per-seed final PPL and the arrow gives their mean.
<table><tr><td>Method</td><td>Searched SP base</td><td>μP-transferred base</td></tr><tr><td>Cosine</td><td> $2 4 . 5 0 \pm 0 . 0 8 ( n = 3 )$ </td><td> $2 4 . 4 4 / 2 4 . 5 3  2 4 . 4 9$ </td></tr><tr><td>SOLAR-online</td><td> $2 2 . 8 6 \pm 0 . 0 8 ( n = 3 )$ </td><td> $2 2 . 8 7 / 2 2 . 9 4  2 2 . 9 1$ </td></tr><tr><td>SOLAR-frozen</td><td> $2 2 . 4 1 ( n = 1 )$ </td><td> $2 2 . 6 2 / 2 2 . 7 4 \to 2 2 . 6 8$ </td></tr></table>

µP+Cosine reaches 24.49 PPL, compared with 24.50 for the searched SP base. SOLAR improves each paired $\mu \mathrm { P }$ run by 1.57–1.59 PPL online and 1.79–1.82 frozen. The frozen configuration performs neither LR/WD search nor PPO updates at the 130M target. It inherits $\alpha = 1 . 3 ,$ action warmup $0 . 1 T \left( = 2 \mathrm { K } \right.$ updates), the source policy’s final learned log σ, and all reward coefficients. The online policy’s negative-action fraction is 0.44 under $\mu \mathrm { P }$ and 0.47 under SP.

We validate the $\mu \mathrm { P }$ implementation by checking activation-scale consistency across widths. Across hidden widths $2 5 6 - 1 0 2 4$ , the mean absolute activation at each layer changes by less than 6% under $\mu \mathrm { { P } ; }$ under standard parameterization, the width-1024 value is approximately 2.1× the width-256 value.

Corpus shift. We also train 130M models on The Pile with the same architecture and token budget. The Cosine peak LR is re-searched on The Pile over $\{ 5 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 3 \times 1 0 ^ { - 3 } \} ; 1 0 ^ { - 3 }$ is selected and $3 \times 1 0 ^ { - 3 }$ diverges at 3.4K updates. Table 16 reports the two final PPL values for each method.

Table 16: Cross-corpus evaluation on 130M/The Pile/AdamW.
<table><tr><td>Method</td><td>Seed 52</td><td>Seed 42</td><td>Mean</td><td>Gain vs. Cosine</td></tr><tr><td>Cosine</td><td>13.90</td><td>14.02</td><td>13.96</td><td></td></tr><tr><td>SOLAR-online</td><td>12.94</td><td>13.03</td><td>12.99</td><td>0.97 (6.9%)</td></tr><tr><td>SOLAR-frozen (60M/C4 policy)</td><td>13.18</td><td>13.24</td><td>13.21</td><td>0.75 (5.4%)</td></tr></table>

The online policy preserves its C4 relative gain under the corpus shift. On The Pile, online has a 0.22 lower mean PPL than frozen across seeds 42 and 52; on the seed-52 C4 run, frozen is 0.38 PPL lower than online. Both modes remain below their matched Cosine baselines.

## C.6 STATIC AND OPEN-LOOP REPLAY CONTROLS

We derive replay controls from a completed 130M/C4/AdamW/seed-52 SOLAR-online run. The static group profile averages each of the G = 111 multipliers over 20K updates and normalizes their cross-group mean to one. The time-varying replay applies the recorded multiplier for every group and update. The level-corrected scalar replay rescales SOLAR’s step-wise mean LR before replay. Each replay runs open-loop without a policy, state input, PPO update, or Circuit-Breaker. A five-point global rescaling search for the static profile selects $c = 1 . 0$ and retains its 24.19 final PPL.

Table 17: Replay controls on 130M/C4/AdamW with seed 52. All entries report final PPL.
<table><tr><td>Method</td><td>Group-wise</td><td>Time-varying</td><td>Live state</td><td>Final PPL</td></tr><tr><td>Cosine</td><td></td><td>base only</td><td></td><td>24.52</td></tr><tr><td>AvgLR Replay</td><td></td><td>√</td><td></td><td>25.61</td></tr><tr><td>Level-corrected scalar replay</td><td></td><td>√</td><td></td><td>24.63</td></tr><tr><td>Static normalized group profile</td><td>V</td><td></td><td></td><td>24.19</td></tr><tr><td>SOLAR-online</td><td>√</td><td>V</td><td>√</td><td>22.79</td></tr></table>

AvgLR Replay and the level-corrected scalar replay reach 25.61 and 24.63 PPL. Static group allocation improves Cosine from 24.52 to 24.19 PPL, while SOLAR-online reaches 22.79.

Table 18: SOLAR-derived replays across seeds, base LRs, and corpora. Both replay profiles come from the seed-52 C4 run at the selected base LR. The Pile row uses seed $5 2 ; { \cdots } { } ^ { \mathrm { ~ } , \mathrm { ~ } }$ denotes a setting not evaluated.
<table><tr><td>Setting</td><td>Cosine</td><td>Static group</td><td>Time-varying group</td><td>SOLAR-frozen</td><td>SOLAR-online</td></tr><tr><td>C4, seed 42, 1.0× base</td><td>24.41</td><td>24.29</td><td>23.94</td><td>22.96</td><td>22.95</td></tr><tr><td>C4, seed 52, 0.5× base</td><td>25.05</td><td>24.83</td><td>24.47</td><td>22.72</td><td>一</td></tr><tr><td>C4, seed 52, 2.0× base</td><td>div. @4.1K</td><td>div. @3.8K</td><td>div. @4.4K</td><td>22.68</td><td>一</td></tr><tr><td>Pile, seed 52, 1.0× base</td><td>13.90</td><td>13.71</td><td>13.55</td><td>13.18</td><td>12.94</td></tr></table>

The learned policies remain below both replay controls across the seed, base-LR, and corpus shifts. At the 2.0× base LR, SOLAR-frozen completes training at 22.68 PPL, while Cosine and both open-loop replays diverge.

## C.7 IMPORTANCE OF GLOBAL AND LOCAL STATE FEATURES

To evaluate our state formulation, we compare the Full state (the complete 10-dimensional state described in Appendix A.1) against two variants:

• Global-only state: Omits per-group statistics, relying solely on the 4 shared global features (e.g., training step, global loss).

• Local-only state: Omits macro-phase indicators, relying solely on the 6 per-group local features (e.g., layer-wise gradient norms).

Combining global and local features gives the lowest final PPL in Table 19: 22.79, compared with 23.11 for global-only and 23.57 for local-only.

Table 19: Ablation on state representation for the 130M model.
<table><tr><td>State Variant</td><td>Final Eval PPL (↓)</td></tr><tr><td>Full state (Ours)</td><td>22.79</td></tr><tr><td>Global-only</td><td>23.11</td></tr><tr><td>Local-only</td><td>23.57</td></tr></table>

## C.8 STOCHASTIC EXPLORATION VS. NEAR-DETERMINISTIC SCHEDULING

SOLAR relies on stochastic exploration to gather informative credit assignment signals under noisy, delayed loss feedback. To quantify the contribution of stochastic exploration, we fix the policy standard deviation to a near-zero constant $\sigma _ { \mathrm { d e t } }$ throughout training while retaining the same PPO update and residual action parameterization:

$$
\begin{array} { r l r } & { u _ { t , g } = \mu _ { t , g } + \sigma _ { \mathrm { d e t } } \epsilon _ { t , g } , } & { \epsilon _ { t , g } \sim \mathcal { N } ( 0 , 1 ) , \ } & { \sigma _ { \mathrm { d e t } } \approx 0 , } \\ & { \widetilde { a } _ { t , g } = \operatorname { t a n h } ( u _ { t , g } ) , } & { a _ { t , g } = \mathrm { c l i p } ( \widetilde { a } _ { t , g } , - 1 + 1 0 ^ { - 4 } , 1 - 1 0 ^ { - 4 } ) . } \end{array}
$$

The resulting actions are effectively determined by the policy mean. All other components—the residual LR modulation ((3)), action-scale warmup schedule, PPO surrogate objective, value function, entropy regularization, and network architecture—remain identical to the stochastic policy.

Suppressing stochasticity increases final PPL from 22.79 to 25.03 in Table 20. The result is consistent with stochastic sampling improving exploration under delayed feedback.

Table 20: Comparison of stochastic exploration versus near-deterministic scheduling (130M model).
<table><tr><td>Policy Formulation</td><td>Final Eval PPL (↓)</td></tr><tr><td>Stochastic (Ours)</td><td>22.79</td></tr><tr><td>Near-deterministic</td><td>25.03</td></tr></table>

## C.9 RESIDUAL LR MODULATION VERSUS DIRECT ABSOLUTE LR PREDICTION

A natural baseline for RL-based learning-rate scheduling is to predict the absolute learning rate directly at each step. We instantiate this idea as a Direct-Abs baseline. Specifically, the scheduler first samples an unconstrained latent action,

$$
u _ { t , g } \sim \mathcal { N } ( \mu _ { t , g } , \sigma ^ { 2 } ) , \qquad \widetilde { a } _ { t , g } = \operatorname { t a n h } ( u _ { t , g } ) , \qquad a _ { t , g } = \operatorname { c l i p } ( \widetilde { a } _ { t , g } , - 1 + 1 0 ^ { - 4 } , 1 - 1 0 ^ { - 4 } )
$$

where $\mu _ { t , g }$ is the scheduler output and the scalar σ is a shared learnable parameter controlling action stochasticity. The squashed action $\boldsymbol { a } _ { t , g }$ is then mapped to an absolute learning rate in log-space:

$$
\log \eta _ { t , g } = \log \eta _ { \mathrm { m i n } } + \frac { a _ { t , g } + 1 } { 2 } \left( \log \eta _ { \mathrm { m a x } } - \log \eta _ { \mathrm { m i n } } \right) ,
$$

and the final learning rate is obtained by exponentiation,

$$
\eta _ { t , g } = \exp ( \log \eta _ { t , g } ) .
$$

This parameterization keeps the action bounded and well-defined, while making the scheduler operate directly on the absolute learning-rate scale. We set $\eta _ { \mathrm { m i n } } = 0 . 1 \eta _ { \mathrm { m a x } }$ and select $\eta _ { \mathrm { m a x } }$ from the same seven-point LR grid used for the other schedulers.

Despite this tuning, Direct-Abs remains brittle in practice. Because the scheduler is responsible for both the absolute scale and the temporal evolution of the learning rate from the very beginning of training, early miscalibrated actions can induce overly aggressive updates, destabilize optimization, and even trigger pronounced loss spikes. Once such disruptions occur in the early stage, they are often difficult to recover from, which ultimately degrades final performance.

SOLAR therefore adopts residual LR modulation on top of a base scheduler. Specifically, the scheduler predicts a bounded multiplicative adjustment:

$$
\eta _ { t , g } = \eta _ { t , g } ^ { \mathrm { b a s e } } \cdot \exp ( \alpha _ { t } a _ { t , g } )
$$

where $\boldsymbol { a } _ { t , g }$ is the scheduler action and $\alpha _ { t }$ controls the residual modulation strength. To further stabilize early training, we linearly warm up the residual scale $\alpha _ { t }$ from 0 to α, which limits the magnitude of residual perturbations before the scheduler becomes reliable.

The combined residual-and-warmup design reaches 22.79 final PPL, compared with 24.74 for Direct-Abs under the matched LR search (Table 21).

Table 21: Ablation on residual LR modulation versus direct absolute LR prediction (130M model).
<table><tr><td>Design</td><td>Final Eval PPL (↓)</td></tr><tr><td>SOLAR (Residual + Warmup)</td><td>22.79</td></tr><tr><td>Direct-Abs</td><td>24.74</td></tr></table>

## C.10 ROBUSTNESS TO SUBOPTIMAL BASE LEARNING RATES

Figure 5a complements the matched $0 . 5 { \times } / 1 { \times } / 2 { \times }$ transfer study in Appendix C.5 with a wider sensitivity sweep on the 130M AdamW setting. We vary the peak base LR from $1 0 ^ { - 4 } ~ \mathrm { t o } ~ 1 0 ^ { - 2 }$ for Cosine, WSD, and SOLAR with a Cosine base.

Cosine and WSD deteriorate once the peak LR exceeds their selected operating region and diverge at the largest values. SOLAR remains trainable over a wider observed interval and, at high base

![](images/011cd642faef6e7d9c13404c3e618fb38722332e3a4b6a30289521bc12f8efbd.jpg)  
(a) Final PPL across peak base LRs on the 130M AdamW setting. Missing high-LR points denote diverged runs.

![](images/0826a0d3d7a1289516b4275ab909e7fc9bc6f88c60fd17f31644dbc357d2a0a3.jpg)  
(b) At a peak base LR of 0.005, negative residual actions reduce the effective LR below the supplied base schedule.  
Figure 5: Sensitivity analysis and mechanism illustration for SOLAR.

LRs, uses negative residual actions to reduce the effective LR. Across the evaluated sweep, SOLAR tolerates substantial base-LR misspecification and actively compensates when the supplied LR is too large.

Observed response. Figure 5b records the controller response at a peak base LR of 0.005. As gradient statistics worsen, the policy outputs $a _ { t , g } < 0$ for affected groups. The mapping $\eta _ { t , g } =$ $\eta _ { t , g } ^ { \mathrm { b a s e } } \exp ( \alpha _ { t } a _ { t , g } )$ then lowers their effective LRs without replacing the long-horizon base profile. This trace provides a direct mechanism for the wider stable interval observed in Figure 5a.

## C.11 EFFICACY OF THE AUTOMATED CIRCUIT-BREAKER MECHANISM

The Circuit-Breaker handles rare optimization anomalies by restoring a complete training state. None of the main-table or MoE runs triggers it. We therefore validate recovery separately with a stress test.

Experimental Setup. We stress-test the Circuit-Breaker on the 130M model by increasing the action bound to $\alpha = 1 . 8$ , which makes large LR multipliers more likely.

![](images/139ff1d7566b9d7df0a0cf63f11de040fda4894f50fe8d0cd010e6c2f193a4ac.jpg)  
(a) Training loss trajectories.

![](images/09a574978f6a82fb62e4409ee95733afb0cad7a7f9f3183f2ed41871aaa2df2a.jpg)  
(b) Global gradient norm trajectories.  
Figure 6: Demonstration of the automated Circuit-Breaker mechanism under an aggressive exploration stress test $( \alpha = 1 . 8 )$ . In both panels, the teal line represents the initial training rollout, which experiences a severe spike and triggers the Circuit-Breaker (marked by the red cross). The orange line represents the automatically resumed trajectory, which restarts from the last safe checkpoint (indicated by the vertical dotted line) and successfully completes the training process.

Results and Analysis. As illustrated in Figure 6, the initial training rollout (teal line) proceeds normally until the agent samples an overly aggressive learning rate multiplier, causing a sudden and severe spike in both the training loss and the global gradient norm.

In a conventional pretraining pipeline, such an event would terminate the run, requiring manual inspection, learning-rate reduction, and a restart from a checkpoint. SOLAR handles this autonomously: the Circuit-Breaker detects the anomaly $( \kappa _ { t + 1 } > 1 . 5 )$ , aborts the current step, and applies the global Circuit-Breaker penalty to the RL agent. The scheduler then performs a forced PPO update using this failure experience before resuming the model and optimizer from the last safe checkpoint while retaining the updated PPO state.

After the forced PPO update and rollback, the resumed trajectory passes the failure point and completes training without another severe spike. This stress test verifies the automated recovery path under aggressive exploration.

## C.12 ADDITIONAL VALIDATION ON THE DEEPSEEK-V2 3B MOE MODEL IN MEGATRON

![](images/fafca29fc3bc3bce6cf8493fa4601b9b9e3759aced5285564a3b9b188773a67b.jpg)  
(a) Validation PPL

![](images/ed2e23abfb841be66b6ea65834053b35794412523754427ab368d82af8fa9a9c.jpg)  
(b) Global grad norm  
Figure 7: Additional validation on the DeepSeek-V2 3B MoE model in Megatron. SOLAR is compared against a matched WSD baseline under the same model, data, tokenizer, optimizer, and training setup.

We run a separate 3B-scale MoE experiment following the DeepSeek-V2 (DeepSeek-AI et al., 2024) architecture in Megatron (Shoeybi et al., 2019). This setting changes the model family, corpus, sequence length, distributed stack, and hardware relative to the main dense study.

The model uses sandwich normalization, multi-latent attention, 12 Transformer layers, hidden size 1280, 64 experts, and top-6 routing. The first layer is dense, and the remaining 11 layers use MoE blocks. We train in bf16 with sequence length 4096, global batch size 1024, and 48K updates, or approximately 201B token positions. The corpus is Dolma 3 Mix (Team OLMo et al., 2025). Both methods use the same corpus, tokenizer, model configuration, optimizer, token budget, and 32×A800 80GB hardware. The WSD baseline uses 2K warmup updates, 12K decay updates, a grid-searched peak LR of $8 . 6 \times 1 0 ^ { - 4 }$ , and a minimum LR of $7 \times 1 0 ^ { - 6 }$ . SOLAR uses this same WSD schedule as its base and updates only the residual controller online.

Figure 7 reports validation PPL and global gradient norm. SOLAR reaches a final PPL of 10.38, compared with 10.73 for WSD, and the observed gap peaks near 1.33 around update 38K. Its gradientnorm trace is also lower and smoother over most of the run. We run each method once in this setting. The comparison tests portability to a different MoE stack but does not estimate run-to-run variance.

## C.13 CONTROL EXPERIMENTS

These controls complement the replay analyses in Section 5.3 and Appendix C.6. They compare SOLAR with state-independent stochastic actions and an untrained fixed-policy controller under the same 130M Llama 2 AdamW setup, base schedule, action-scale warmup, and training budget.

## C.13.1 STATE-AGNOSTIC STOCHASTIC SCHEDULER

To evaluate state-conditioned policy learning under the same stochastic residual family, we keep the multiplicative residual mapping but replace the learned state-conditioned action with $a _ { t , g } \sim$ TanhNormal(0, σ). We select σ from {1, 2, 3, 4, 5} on the tuning seed and report that configuration’s final checkpoint. The base schedule, action-scale warmup, and training setup match SOLAR.

The selected state-agnostic scheduler reaches 25.21 PPL, compared with 22.79 for SOLAR (Table 22).

## C.13.2 UNTRAINED FIXED-POLICY CONTROLLER

This control keeps the SOLAR scheduler architecture, state inputs, and residual action parameterization, but fixes the policy at its random initialization and performs no PPO updates. The base schedule, action-scale warmup, and training setup match SOLAR.

The untrained fixed-policy controller reaches 25.04 PPL. Target-run policy updates reduce this value to 22.79 with SOLAR-online.

Table 22: Controls for state-independent stochasticity and untrained policy initialization. All experiments use the 130M Llama 2 model with AdamW on C4.
<table><tr><td>Method</td><td>State- conditioned</td><td>PPO updates</td><td>Stochastic sampling</td><td>Action noise</td><td>Final Eval PPL↓</td></tr><tr><td>SOLAR</td><td>√</td><td>√</td><td>√</td><td>learned</td><td>22.79</td></tr><tr><td>State-agnostic stochastic scheduler</td><td>一</td><td></td><td>√</td><td>selected σ</td><td>25.21</td></tr><tr><td>Untrained fixed-policy controller</td><td>√</td><td></td><td>√</td><td>fixed init</td><td>25.04</td></tr></table>

The state-agnostic and untrained fixed-policy controls finish at 25.21 and 25.04 PPL, while SOLAR finishes at 22.79.

## C.14 RUNTIME ACCOUNTING

Table 23: Steady-state step-time overhead on the 1B AdamW setting with 8×RTX 4090. The post-warmup benchmark window excludes evaluation and checkpointing.
<table><tr><td>Method</td><td>Step Time (s)</td><td>Throughput (tok/s)</td><td>Overhead Increase (%)</td></tr><tr><td>Baseline LRS</td><td>1.976</td><td>66329.01</td><td>0.00</td></tr><tr><td>SOLAR (frozen)</td><td>1.992</td><td>65799.20</td><td>0.81</td></tr><tr><td>SOLAR (online)</td><td>2.001</td><td>65499.35</td><td>1.27</td></tr></table>

Table 24: Full-run wall-clock accounting. These measurements include evaluation, checkpointing, state construction, policy inference, action application, PPO updates for the online mode, and any rollback time. No listed run triggered rollback. Dense and Qwen2-MoE runs use 8×RTX 4090; the 3B MoE runs use 32×A800 80GB.
<table><tr><td>Setting</td><td>Baseline (h)</td><td>SOLAR-online (h)</td><td>Online OH</td><td>Frozen OH</td></tr><tr><td>1B AdamW</td><td>56.75</td><td>57.45</td><td>1.23%</td><td>0.76%</td></tr><tr><td>1B Muon</td><td>65.34</td><td>66.04</td><td>1.07%</td><td>0.69%</td></tr><tr><td>Qwen2-MoE 1B</td><td>127.38</td><td>129.27</td><td>1.48%</td><td></td></tr><tr><td>DeepSeek-V2-style MoE 3B</td><td>163.38</td><td>165.00</td><td>0.99%</td><td></td></tr></table>

SOLAR does not add an LLM forward or backward pass. Its incremental work comprises reductionbased state construction over resident tensors, a small rank-0 policy network, one scalar-action

broadcast per controlled group, and periodic PPO updates in online mode. Freezing the residual policy removes the PPO update while retaining state construction and policy inference.

Policy-side complexity. Let G be the number of controlled groups, D the state dimension, H the hidden width, and A the action dimension. The two-hidden-layer policy costs

$$
\mathcal { O } \left( D H + H ^ { 2 } + H A \right) .
$$

per group and

$$
\mathcal { O } \bigl ( G ( D H + H ^ { 2 } + H A ) \bigr ) .
$$

Here $A = 1$ per group. State extraction uses reductions over existing parameters and gradients; PPO backpropagation is amortized because it occurs periodically rather than at every model update.

Step time and full-run wall-clock time. Table 23 isolates the incremental cost in a steady-state benchmark window. It deliberately excludes evaluation and checkpointing so that the step-level mechanism can be measured without cadence effects. Throughput is computed as

$$
\mathrm { T h r o u g h p u t } = \frac { B _ { \mathrm { g l o b a l } } \cdot L _ { \mathrm { m a x } } } { t _ { \mathrm { s t e p } } } .
$$

On 1B AdamW, the measured step-time increases are 1.27% online and 0.81% frozen.

Table 24 times each complete training job from start to finish. It includes evaluation and checkpointing as shared work, as well as SOLAR’s checkpoint maintenance and any recovery time. The 1B AdamW full-run overhead is 1.23%, slightly below the 1.27% step-time value because common evaluation and checkpoint work enlarge the full-run denominator. Across the listed complete runs, online overhead ranges from 0.99% to 1.48%.

The 3B MoE, 1B dense, and Qwen2-MoE settings use 155, 219, and 593 controlled groups and show 0.99%, 1.23%, and 1.48% full-run overhead, respectively. The dense implementation groups one trainable tensor at a time, while the MoE counts follow each codebase’s module partition.