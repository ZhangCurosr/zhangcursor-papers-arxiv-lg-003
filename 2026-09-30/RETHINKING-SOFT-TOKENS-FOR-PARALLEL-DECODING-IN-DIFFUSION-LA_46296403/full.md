# RETHINKING SOFT TOKENS FOR PARALLEL DECODING IN DIFFUSION LANGUAGE MODELS

Kodai Kawamura Kenji Kawaguchi Anji Liu National University of Singapore

## ABSTRACT

Diffusion language models (DLMs) enable parallel generation by predicting and committing multiple tokens at each denoising step, yet they can generate individually plausible but mutually inconsistent tokens. Recent work shows that soft tokens can mitigate this issue by representing uncertain positions with continuous embeddings built from the model’s predictive distribution at the previous decoding step. However, although soft tokens are commonly understood as preserving predictive uncertainty, how soft-token feedback improves parallel decoding has not been systematically examined. In this paper, we investigate this question in frozen pretrained DLMs to examine soft-token feedback without the effects of additional training. To construct soft-token inputs in a training-free setting, we identify a geometric mismatch between conventional soft-token construction and the pretrained embedding space. Based on this observation, we propose a training-free, geometry-aware construction of soft tokens. Our analysis of soft-token feedback suggests that uncertainty preservation alone does not fully explain how it reshapes subsequent predictions. To better explain how soft-token feedback improves parallel decoding, we provide empirical evidence that it favors coherent token sequences. Across four pretrained DLMs and four math and code benchmarks, our method outperforms standard parallel decoding and a training-free Euclidean soft-token baseline. Code: https: //github.com/kodaikawamura/rethinking-soft-tokens

## 1 INTRODUCTION

Diffusion language models (DLMs) have recently emerged as a promising alternative to autoregressive language models by enabling parallel text generation through iterative denoising (Nie et al., 2025; Ye et al., 2025). By predicting multiple masked positions simultaneously, DLMs naturally support parallel decoding and offer the potential for substantially faster decoding (Wu et al., 2026; Ben-Hamu et al., 2025). However, standard parallel decoding uses a factorized approximation to the joint distribution, ignoring dependencies among simultaneously predicted tokens (Israel et al., 2025). As a result, DLMs can commit multiple tokens that are individually plausible but mutually inconsistent, introducing errors that can propagate through later denoising steps (Liu et al., 2025).

To mitigate this limitation, recent works introduce soft tokens (Hersche et al., 2026; Zhong et al., 2026; Chen et al., 2026). Rather than either replacing a masked position with a discrete prediction or leaving it unresolved as [MASK], soft-token methods represent unresolved positions with contin uous embeddings constructed by interpolating between the [MASK] embedding and a probabilityweighted average of the top-k predicted-token embeddings. These continuous representations are fed back into the model at the next denoising step and have been shown to improve parallel decoding.

Despite their empirical success, a fundamental question remains: how does soft-token feedback improve parallel decoding? Although its benefits are commonly attributed to preserving predictive uncertainty (Hersche et al., 2026; Zhong et al., 2026), this explanation has not been systematically examined and does not specify how soft tokens affect subsequent predictions. Existing methods train or fine-tune models to process soft-token representations, making it difficult to determine whether their gains arise from soft-token feedback itself or from learned adaptation to these representations. To investigate how soft-token feedback reshapes predictions without the effects of additional training, we develop a soft-token construction that can be applied directly to frozen pretrained DLMs.

To enable this investigation, we first address a question: how should soft tokens be constructed for models that have not been trained to process them? We find that directly applying conventional Euclidean soft-token constructions to frozen pretrained DLMs can distort angular information and shrink embedding norms. We therefore propose geometry-aware soft tokens based on spherical interpolation, which can be directly incorporated into pretrained DLMs without additional training. Across four pretrained DLMs and four math and code benchmarks, our method outperforms both standard parallel decoding and training-free Euclidean soft-token baselines.

Using this training-free construction, we then investigate how soft-token feedback affects subsequent predictions. In controlled experiments, we show that uncertainty preservation alone does not fully explain the observed behavior. We further provide empirical evidence that soft-token feedback shifts probability mass toward coherent token sequences and away from inconsistent combinations. These findings offer a possible explanation for how soft-token feedback improves parallel decoding.

## 2 PRELIMINARIES

## 2.1 DIFFUSION LANGUAGE MODELS

Let $\mathbf { x } _ { 0 } = ( x _ { 0 } ^ { 1 } , \hdots , x _ { 0 } ^ { N } ) \in \mathcal { V } ^ { N }$ denote a clean sequence of tokens, where V is a vocabulary of size $V$ that includes a special [MASK] token. In this paper, we focus on masked diffusion language models (Sahoo et al., 2024), in which the [MASK] token serves as an absorbing state during the forward diffusion process.

The model learns the data distribution $p _ { \mathrm { d a t a } } ( \mathbf { x } _ { 0 } )$ by recovering clean sequences from partially masked observations. The forward diffusion process gradually corrupts a clean sequence by replacing its tokens with [MASK]:

$$
\mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { T } ,
$$

where $\mathbf { x } _ { t } = ( x _ { t } ^ { 1 } , \ldots , x _ { t } ^ { N } ) \in \mathcal { V } ^ { N }$ represents the token sequence at diffusion step t. The transition between two consecutive states is modeled as

$$
q ( x _ { t } ^ { i } \mid x _ { t - 1 } ^ { i } ) = Q _ { t } [ x _ { t - 1 } ^ { i } , x _ { t } ^ { i } ] , \qquad q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { t - 1 } ) = \prod _ { i = 1 } ^ { N } q ( x _ { t } ^ { i } \mid x _ { t - 1 } ^ { i } ) ,
$$

where $Q _ { t } \in [ 0 , 1 ] ^ { V \times V }$ is a categorical transition matrix whose entry $Q _ { t } [ a , b ]$ gives the probability of transitioning from token a to token b. In $\mathrm { D L M s } , Q _ { t }$ is constructed such that each token either remains unchanged or transitions to the absorbing [MASK] state, with the probability of being masked increasing over time.

During inference, decoding begins from x<sub>T</sub>, where the prompt tokens remain fixed and all positions to be generated are initialized as [MASK]. At each denoising step, the model maps each input token, including [MASK], to its learned embedding and feeds the embedding sequence into a bidirectional Transformer to obtain a predictive distribution over the vocabulary at each position. A common unmasking strategy selects the positions with the highest prediction confidence and replaces them with their predicted tokens, while the remaining positions stay masked for subsequent iterations. This procedure supports parallel decoding by unmasking multiple positions in a denoising step.

## 2.2 CHALLENGES IN PARALLEL DECODING

Although parallel decoding is possible with diffusion language models, they can generate token predictions that are individually plausible yet mutually inconsistent. Given a partially masked context $\mathbf { x } ,$ let $\mathbf { y } = ( y _ { 1 } , \dots y _ { m } )$ denote the tokens predicted simultaneously. Standard parallel decoding uses the factorized distribution $q _ { \mathrm { p a r a l l e l } } \mathrm { . }$

$$
q _ { \mathrm { p a r a l l e l } } ( \mathbf { y } \mid \mathbf { x } ) = \prod _ { i = 1 } ^ { m } p _ { \theta } ( y _ { i } \mid \mathbf { x } ) ,
$$

where $p _ { \theta }$ is the predictive distribution of the model with parameters θ. This factorization ignores dependencies among simultaneously predicted tokens. Even when every predicted marginal is exact, their product generally differs from the true joint distribution when the tokens are dependent. Thus, fully factorized prediction introduces an approximation error that cannot be eliminated by improving the individual marginals alone (Liu et al., 2025; Li et al., 2026).

Prompt: Answer is either Harry Potter or Star Wars: [MASK] [MASK]

![](images/8732d3c5d1f6f974365b4ce884b98346953e4ba5c24c9747fc243a8978a2d5fb.jpg)  
Figure 1: Soft-token feedback improves sequence-level consistency. (a) Sequential decoding conditions each prediction on previously committed tokens, producing the coherent sequence Harry Potter. (b) Parallel decoding predicts both tokens independently, assigning probability mass to inconsistent combinations and making Harry Wars the most likely sequence. (c) Soft-token feedback using our proposed construction (Section 4) shifts probability mass toward the coherent sequence Harry Potter and away from inconsistent combinations.

## 2.3 SOFT-TOKEN DECODING

Recent work introduces soft tokens and shows improved performance in parallel decoding, suggesting a potential way to mitigate the inconsistencies arising from independent token predictions (Hersche et al., 2026; Zhong et al., 2026; Chen et al., 2026). Instead of retaining the discrete [MASK] token, soft-token decoding constructs a continuous representation from the model’s predictive dis tribution and feeds it into the next denoising iteration. This continuous representation retains information from multiple candidate tokens without committing to a single prediction.

A conventional Euclidean soft token is constructed as (Hersche et al., 2026; Zhong et al., 2026)

$$
\mathbf { e } _ { \mathrm { s o f t } } = ( 1 - \lambda ) \mathbf { e } _ { \mathrm { M A S K } } + \lambda \sum _ { i = 1 } ^ { k } p _ { i } \mathbf { e } _ { i } ,
$$

where $\mathbf { e } _ { i }$ is the embedding of the i-th token among the model’s top-k predicted tokens, $p _ { i }$ is its predictive probability renormalized over these candidate tokens, and $\lambda \in \ [ 0 , 1 ]$ controls the interpolation with the [MASK] embedding ${ \bf e } _ { \mathrm { M A S K } }$ . The resulting soft token replaces the [MASK] embedding in the next denoising iteration. Existing methods either train DLMs from scratch with soft-token inputs or adapt pretrained DLMs to these inputs through additional training.

Figure 1 illustrates how soft-token feedback, implemented in frozen DLMs using our proposed construction (Section 4), can mitigate errors in parallel decoding. We consider a toy experiment in which a prompt admits two coherent sequences: Harry Potter and Star Wars. Sequential decoding allows a DLM to account for the dependency between the two tokens by conditioning the second prediction on the first, but it is slow as the model commits one token per step. When both tokens are predicted independently in parallel, however, the product of their marginals assigns probability mass to inconsistent combinations such as Harry Wars and Star Potter, even when the marginal distribution at each position is exact. Soft-token feedback reshapes these marginals to favor mutually compatible token choices, making Harry Potter the most likely sequence under parallel decoding.

These observations raise a question: how does soft-token feedback reshape subsequent predictions to improve sequence-level consistency? Although prior work attributes performance gains from soft-token feedback to uncertainty preservation (Hersche et al., 2026; Zhong et al., 2026), this explanation has not been systematically examined. We first formalize this prevailing interpretation to provide a reference for examining this behavior.

## 3 AN UNCERTAINTY-PRESERVATION VIEW OF SOFT TOKENS

Soft tokens are commonly understood to preserve uncertainty by carrying information about multiple candidates into subsequent predictions. To formalize this interpretation, we model a soft token as soft evidence (Chan & Darwiche, 2005; Munk et al., 2023) over a hypothetical discrete input state

$$
Z \in \{ \mathrm { M } , v _ { 1 } , \dots , v _ { k } \} ,
$$

where M denotes the unresolved [MASK] state and $v _ { i }$ is the candidate token with the i-th highest predicted probability at the position represented by the soft token.

If a soft token preserves uncertainty in this probabilistic sense, meaning that its role is to represent uncertainty over the values of $Z ,$ , prediction marginalizes over these realizations:

$$
p ( Y = y ~ \vert ~ E ) = \sum _ { z } p ( Z = z ~ \vert ~ E ) p _ { \theta } ( Y = y ~ \vert ~ E , Z = z ) ,
$$

where $E$ is the current decoding context. For interpolation strength λ and candidate probabilities $p _ { i }$ renormalized over the top-k tokens, the distribution over $Z$ is defined by

$$
p ( Z = v _ { i } \mid E ) = \lambda p _ { i } , \qquad p ( Z = \mathrm { \bf { M } } \mid E ) = 1 - \lambda \sum _ { i = 1 } ^ { k } p _ { i } .
$$

Therefore,

$$
p ( Y = y \mid E ) = \left( 1 - \lambda \sum _ { i = 1 } ^ { k } p _ { i } \right) p _ { \theta } { \big ( } Y = y \mid E , Z = \mathrm { M } { \big ) } + \lambda \sum _ { i = 1 } ^ { k } p _ { i } p _ { \theta } { \big ( } Y = y \mid E , Z = v _ { i } { \big ) } .
$$

The same argument extends to multiple soft-token positions. Let S be the set of positions represented by soft tokens, and let $\mathbf { Z } = ( Z _ { j } ) _ { j \in { \mathcal { S } } }$ denote their joint discrete realization. Assuming that the positionwise evidence is realized independently,

$$
p ( \mathbf { Z } = \mathbf { z } \mid E ) = \prod _ { j \in S } p ( Z _ { j } = z _ { j } \mid E ) .
$$

Marginalizing over all joint discrete realizations yields an additive mixture of the predictive distributions they induce, which we denote by $p _ { \mathrm { a d d } } \colon$

$$
p _ { \mathrm { a d d } } ( y ) : = p ( Y = y \mid E ) = \sum _ { \mathbf { z } } p ( \mathbf { Z } = \mathbf { z } \mid E ) p _ { \theta } ( Y = y \mid E , \mathbf { Z } = \mathbf { z } ) .
$$

This additive mixture follows naturally from the probabilistic interpretation above.

We examine whether predictions induced by continuous soft-token inputs agree with this reference. To investigate this question without the effects of additional training, we study soft-token feedback in frozen pretrained DLMs. This requires addressing how soft tokens should be constructed for models that have not been trained to process these representations. We develop such a construction in the following section, then return to the question of how its feedback reshapes subsequent predictions.

## 4 TRAINING-FREE SOFT-TOKEN DECODING

To enable this investigation, we address how to construct soft tokens for a frozen pretrained DLM without additional training. Because the model has not been trained to process mixtures of token embeddings, the placement of these representations within the pretrained embedding space is particularly important. We first identify the geometric mismatch introduced by conventional Euclidean construction and then propose a geometry-aware method for constructing soft tokens.

## 4.1 GEOMETRIC MISMATCH IN EUCLIDEAN SOFT TOKENS

As discussed in Section 2.3, conventional soft-token construction consists of two operations: aggregating the top-k candidate embeddings and interpolating between the aggregated representation and

the [MASK] embedding. To investigate the geometry relevant to these operations, we consider how the resulting input is processed by the frozen model. In all DLMs considered in this work, every self-attention and feed-forward sublayer operates on an RMS-normalized representation:

$$
\mathrm { R M S N o r m } ( \mathbf { h } ) = \left( \sqrt { d } \gamma \right) \odot \frac { \mathbf { h } } { \| \mathbf { h } \| _ { 2 } } .
$$

where d is the hidden dimension, γ is a learned scaling vector, and ⊙ denotes elementwise multiplication (Zhang & Sennrich, 2019). Through this operation, the input representation h is normalized by its magnitude before each Transformer sublayer, reducing sensitivity to the input norm while retaining directional information.

This repeated normalization motivates explicitly controlling the direction of soft-token representations. Our empirical analysis reveals a geometric distinction between the two operations in conventional soft-token construction: top-k candidate embeddings are relatively aligned with one another, whereas their aggregate is nearly orthogonal to the [MASK] embedding across the evaluated models and datasets (Section 6.3).

The relative alignment among candidate embeddings supports retaining Euclidean averaging as a practical choice for top-k aggregation. This choice is also supported by prior work demonstrating the effectiveness of probability-weighted embedding aggregation in frozen autoregressive language models without additional training (Zhang et al., 2025).

Interpolation with [MASK], however, must bridge a much larger angular separation. Under linear interpolation, the resulting direction depends on both the interpolation weight and the relative norms of the two endpoints. Consequently, the same weight can produce different angular movements toward the candidate aggregate, while the embedding norm may also shrink (Appendix B.1). These observations motivate controlling direction and magnitude separately when interpolating between the candidate aggregate and the [MASK] embedding.

## 4.2 GEOMETRY-AWARE SOFT-TOKEN CONSTRUCTION

We construct geometry-aware soft tokens in two stages: candidate aggregation and interpolation with the [MASK] embedding. Following the analysis in Section 4.1, we retain Euclidean top-k aggregation and compute the semantic representation as

$$
\mathbf { m } = \sum _ { i = 1 } ^ { k } p _ { i } \mathbf { e } _ { i } ,
$$

where $p _ { i }$ denotes the probability assigned to the i-th top-k candidate token, whose embedding is $\mathbf { e } _ { i }$

We then interpolate between the [MASK] embedding and the semantic representation by explicitly controlling direction and norm separately. We first interpolate between their directions, then scale the resulting unit vector to match the [MASK] embedding norm, retaining the pretrained scale associated with an unresolved position.

For directional interpolation, we use spherical linear interpolation (SLERP) (Shoemake, 1985). Specifically, we normalize the two endpoints and compute their angular separation:

$$
\widehat { \mathbf { e } } _ { \mathrm { M A S K } } = \frac { \mathbf { e } _ { \mathrm { M A S K } } } { \| \mathbf { e } _ { \mathrm { M A S K } } \| _ { 2 } } , \qquad \widehat { \mathbf { m } } = \frac { \mathbf { m } } { \| \mathbf { m } \| _ { 2 } } , \qquad \theta = \operatorname { a r c c o s } \left( \widehat { \mathbf { e } } _ { \mathrm { M A S K } } ^ { \top } \widehat { \mathbf { m } } \right) .
$$

The resulting soft-token embedding is given by

$$
\mathbf { e } _ { \mathrm { s o f t } } = \| \mathbf { e } _ { \mathrm { M A S K } } \| _ { 2 } \left[ { \frac { \sin ( ( 1 - \lambda ) \theta ) } { \sin \theta } } { \widehat { \mathbf { e } } } _ { \mathrm { M A S K } } + { \frac { \sin ( \lambda \theta ) } { \sin \theta } } { \widehat { \mathbf { m } } } \right] , \qquad \lambda \in [ 0 , 1 ] .
$$

Here, λ controls how far the soft token moves from the [MASK] direction toward the direction of the semantic representation along the spherical arc, while its norm remains fixed at ∥e<sub>MASK</sub>∥<sub>2</sub>.

We use the proposed construction for soft tokens while keeping all pretrained model parameters frozen. At each iteration, we commit discrete tokens at the positions with the highest prediction confidence, while each unresolved position is represented by a soft token recomputed from its updated top-k predictive distribution.

## 5 UNDERSTANDING SOFT-TOKEN FEEDBACK

Having introduced a training-free construction, we now return to the uncertainty-preservation interpretation in Section 3. We first examine whether predictions induced by our construction follow the additive mixture implied by this interpretation. We then investigate how the observed behavior relates to sequence-level consistency in parallel decoding.

## 5.1 LIMITATIONS OF THE UNCERTAINTY-PRESERVATION EXPLANATION

Representing multiple candidate tokens in a continuous input does not by itself determine how their information is combined in subsequent predictions. Under the uncertainty-preservation interpretation in Section 3, this information is propagated by marginalizing over possible discrete input realizations, yielding the additive mixture $p _ { \mathrm { a d d } }$ . To examine whether our soft-token construction follows this interpretation, we compare its predictions with the additive mixture $p _ { \mathrm { a d d } }$ and the following normalized multiplicative mixture:

$$
p _ { \mathrm { m u l t } } ( y ) = \frac { \exp ( \sum _ { \mathbf { z } } p ( \mathbf { Z } = \mathbf { z } \mid E ) \log p _ { \boldsymbol { \theta } } ( Y = y \mid E , \mathbf { Z } = \mathbf { z } ) ) } { \sum _ { y ^ { \prime } } \exp \left( \sum _ { \mathbf { z } } p ( \mathbf { Z } = \mathbf { z } \mid E ) \log p _ { \boldsymbol { \theta } } ( Y = y ^ { \prime } \mid E , \mathbf { Z } = \mathbf { z } ) \right) } .
$$

We compare soft-token predictions with these reference mixtures using Jensen–Shannon (JS) divergence on controlled prompts (Section 6.4). Table 1 shows that predictions induced by our construction are closer to the multiplicative reference than to the additive mixture across all four models. These results suggest that the uncertainty-preservation interpretation alone does not fully account for the predictive behavior induced by our construction. In the following subsection, we examine how this multiplicative behavior may contribute to improved parallel decoding.

Table 1: JS divergence between predictions from our construction and the reference mixtures. Lower values indicate closer agreement; bold indicates the closer reference.

<table><tr><td>Model</td><td>Additive</td><td>Multiplicative</td></tr><tr><td>LLaDA</td><td>0.2099</td><td>0.0410</td></tr><tr><td>LLaDA-1.5</td><td>0.0496</td><td>0.0019</td></tr><tr><td>LLaDA-2.0 mini</td><td>0.2198</td><td>0.0835</td></tr><tr><td>Dream</td><td>0.0566</td><td>0.0119</td></tr></table>

## 5.2 AGREEMENT-SEEKING FEEDBACK AND SEQUENCE-LEVEL CONSISTENCY

Multiplicative aggregation combines predictive distributions through a weighted average of their log probabilities. Predictions assigned very low probability by input realizations with substantial weight can therefore be suppressed, even when other realizations support them. This sensitivity to disagreement across input realizations motivates an agreement-seeking explanation for how softtoken feedback improves parallel decoding.

More specifically, soft-token feedback allows each position to incorporate candidate information from other positions before commitment. We hypothesize that its sensitivity to conflicting candidate evidence helps shift predictions across positions toward mutually compatible choices. Although the resulting joint prediction remains factorized, soft-token feedback can make inconsistent crosscombinations less likely by favoring a coherent combination of tokens across positions.

To examine this sequence-level behavior, we construct a sequential reference from the same frozen model by averaging its joint predictions over all unmasking orders. Each prediction conditions on previously filled tokens, allowing the reference to incorporate dependencies among answer tokens. We evaluate agreement using conditional reverse KL (Appendix D.3), which penalizes probability assigned to combinations poorly supported by the reference. This criterion is relevant to parallel decoding because mode-seeking predictions can yield lower reverse KL by concentrating probability on a coherent alternative and assigning less probability to inconsistent cross-combinations.

The analysis in Section 6.4 shows that our method achieves lower reverse KL than vanilla parallel decoding and Euclidean soft-token baselines, indicating closer agreement with the sequential reference. Together with the mixture analysis, these results support an agreement-seeking interpretation: feedback shifts the marginals toward mutually compatible predictions before commitment.

Table 2: Comparison of accuracy (%) with baseline methods across four diffusion language models under parallel decoding settings. Vanilla denotes standard parallel decoding without soft-token feedback, and Euclidean denotes the training-free soft-token baseline using linear interpolation in embedding space. Tokens/step denotes the number of tokens committed per denoising iteration. The best results for each model, benchmark, and decoding setting are shown in bold.
<table><tr><td rowspan="2">Model</td><td>Method</td><td colspan="3">GSM8K</td><td colspan="3">MATH500</td><td colspan="3">HumanEval</td><td colspan="3">MBPP</td></tr><tr><td>Tokens/step</td><td>2</td><td>4</td><td>8</td><td>2</td><td>4</td><td>8</td><td>2</td><td>4</td><td>8</td><td>2</td><td>4</td><td>8</td></tr><tr><td rowspan="3">LLaDA-2.0 mini</td><td>Vanilla</td><td>89.61</td><td>83.09</td><td>49.81</td><td>40.60</td><td>37.40</td><td>16.20</td><td>70.73</td><td>46.95</td><td>24.39</td><td>55.80</td><td>42.80</td><td>23.80</td></tr><tr><td>Euclidean</td><td>89.76</td><td>85.82</td><td>51.78</td><td>43.60</td><td>37.20</td><td>18.40</td><td>69.51</td><td>50.00</td><td>22.56</td><td>57.60</td><td>43.80</td><td>23.00</td></tr><tr><td>Ours</td><td>90.60</td><td>87.40</td><td>54.20</td><td>44.60</td><td>39.00</td><td>18.80</td><td>71.95</td><td>53.05</td><td>23.17</td><td>57.80</td><td>44.20</td><td>23.40</td></tr><tr><td rowspan="3">LLaDA-1.5</td><td>Vanilla</td><td>72.25</td><td>65.13</td><td>43.82</td><td>20.80</td><td>17.60</td><td>15.00</td><td>38.41</td><td>24.39</td><td>12.80</td><td>34.80</td><td>26.20</td><td>17.20</td></tr><tr><td>Euclidean</td><td>71.40</td><td>66.79</td><td>47.54</td><td>21.00</td><td>18.80</td><td>14.80</td><td>39.02</td><td>29.88</td><td>15.24</td><td>35.00</td><td>27.40</td><td>18.00</td></tr><tr><td>Ours</td><td>71.80</td><td>67.63</td><td>47.99</td><td>24.00</td><td>20.40</td><td>15.00</td><td>43.90</td><td>33.54</td><td>18.90</td><td>35.80</td><td>29.00</td><td>18.40</td></tr><tr><td rowspan="3">LLaDA</td><td>Vanilla</td><td>71.65</td><td>64.44</td><td>36.77</td><td>24.40</td><td>18.20</td><td>12.80</td><td>39.02</td><td>24.39</td><td>15.24</td><td>33.20</td><td>27.00</td><td>16.00</td></tr><tr><td>Euclidean</td><td>72.33</td><td>64.29</td><td>36.69</td><td>22.60</td><td>19.40</td><td>14.80</td><td>41.47</td><td>31.71</td><td>15.24</td><td>34.60</td><td>26.80</td><td>17.80</td></tr><tr><td>Ours</td><td>73.24</td><td>65.58</td><td>36.77</td><td>23.60</td><td>20.00</td><td>14.80</td><td>45.12</td><td>32.32</td><td>19.51</td><td>36.20</td><td>28.60</td><td>19.20</td></tr><tr><td rowspan="3">Dream</td><td>Vanilla</td><td>74.45</td><td>56.94</td><td>16.15</td><td>20.40</td><td>5.20</td><td>0.40</td><td>42.68</td><td>17.68</td><td>3.05</td><td>43.40</td><td>25.20</td><td>14.20</td></tr><tr><td>Euclidean</td><td>77.03</td><td>56.25</td><td>16.22</td><td>21.80</td><td>6.00</td><td>0.80</td><td>46.34</td><td>23.78</td><td>8.54</td><td>45.00</td><td>24.80</td><td>15.00</td></tr><tr><td>Ours</td><td>78.17</td><td>58.61</td><td>18.20</td><td>22.40</td><td>6.40</td><td>0.80</td><td>50.61</td><td>32.32</td><td>14.02</td><td>47.00</td><td>32.80</td><td>17.40</td></tr></table>

## 6 EXPERIMENTS

## 6.1 SETTINGS

Models. We evaluate four instruction-tuned diffusion language models: LLaDA-Instruct-8B (Nie et al., 2025), LLaDA-1.5-Instruct (Zhu et al., 2026), LLaDA-2.0-mini (Bie et al., 2025), and Dream-7B (Ye et al., 2025). All model parameters remain frozen throughout decoding.

Benchmarks. We conduct experiments on four mathematical-reasoning and code-generation benchmarks. GSM8K (Cobbe et al., 2021) contains grade-school mathematical reasoning problems, while MATH500 (Lightman et al., 2024) contains competition-level mathematics problems. HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021b) evaluate Python code generation from natural-language specifications. We report exact-match accuracy for GSM8K and MATH500 and pass@1 for HumanEval and MBPP.

Baselines. We compare our method with vanilla parallel decoding, which commits predictions at the most-confident unresolved positions in each iteration while leaving the remaining positions as [MASK] tokens. To the best of our knowledge, no existing soft-token decoding method can be directly applied to a frozen pretrained DLM without additional training. We therefore construct a training-free Euclidean soft-token baseline. This baseline linearly interpolates between the [MASK] embedding and the aggregated candidate representation, whereas our manifold-aware method uses spherical interpolation. Additional implementation details are provided in Appendix A.

## 6.2 MAIN RESULTS

Table 2 compares our method with standard parallel decoding without soft-token feedback (Vanilla) and the training-free soft-token baseline using linear interpolation in embedding space (Euclidean). Our method outperforms both baselines in most settings without additional training.

On GSM8K with LLaDA-2.0 mini under the same decoding setting, our method achieves 87.40% accuracy, yielding gains of 4.31 and 1.58 percentage points over the respective baselines. Gains extend to other decoding settings. On HumanEval with Dream at four tokens per step, our method achieves 32.32% accuracy, outperforming vanilla parallel decoding and Euclidean soft tokens by 14.64 and 8.54 percentage points, respectively. These results demonstrate that our soft-token construction can improve parallel decoding in frozen pretrained DLMs. Appendix B presents an ablation study, hyperparameter sensitivity analyses, and results showing that our method can improve existing adaptive decoding methods (Wu et al., 2026; Ben-Hamu et al., 2025).

Table 3: Embedding geometry on GSM8K and HumanEval. Candidate–candidate reports the mean pairwise cosine similarity among the top-three predicted-token embeddings. Aggregate–[MASK] reports the mean absolute cosine similarity between the probability-weighted average of these embeddings and the [MASK] embedding.
<table><tr><td>Model</td><td>Dataset</td><td>Candidate-candidate</td><td>Aggregate-[MASK]</td></tr><tr><td rowspan="2">LLaDA</td><td>GSM8K</td><td>0.496</td><td>0.015</td></tr><tr><td>HumanEval</td><td>0.447</td><td>0.011</td></tr><tr><td rowspan="2">LLaDA-1.5</td><td>GSM8K</td><td>0.543</td><td>0.014</td></tr><tr><td>HumanEval</td><td>0.497</td><td>0.011</td></tr><tr><td rowspan="2">LLaDA-2.0 mini</td><td>GSM8K</td><td>0.724</td><td>0.043</td></tr><tr><td>HumanEval</td><td>0.629</td><td>0.074</td></tr><tr><td rowspan="2">Dream</td><td>GSM8K</td><td>0.566</td><td>0.028</td></tr><tr><td>HumanEval</td><td>0.750</td><td>0.070</td></tr></table>

Table 4: Reverse KL divergence from the factorized joint prediction to the dependency-preserving reference. The lowest reverse KL for each model is shown in bold.
<table><tr><td>Model</td><td>Vanilla</td><td>Euclidean</td><td>Ours</td></tr><tr><td>LLaDA</td><td>2.1989</td><td>0.3415</td><td>0.2458</td></tr><tr><td>LLaDA-1.5</td><td>2.0327</td><td>1.9273</td><td>0.5624</td></tr><tr><td>LLaDA-2.0 mini</td><td>2.7338</td><td>1.5610</td><td>1.1739</td></tr><tr><td>Dream</td><td>4.2733</td><td>4.1139</td><td>1.7553</td></tr></table>

## 6.3 GEOMETRIC ANALYSIS OF TOKEN EMBEDDINGS

To examine the geometry motivating our construction in Section 4.1, we analyze four DLMs on GSM8K and HumanEval. For each evaluated unresolved token position in the output sequence, we consider the top-three candidate embeddings and measure two quantities: (i) their average pairwise cosine similarity and (ii) the absolute cosine similarity between their probability-weighted aggregate and the [MASK] embedding.

Table 3 shows that candidate embeddings are more aligned with one another, on average, than their aggregate is with the [MASK] embedding across all eight model–dataset pairs. This finding supports retaining Euclidean averaging as a practical choice for candidate aggregation.

The low absolute cosine similarities between the aggregate and [MASK] indicate a large angular separation. Linear interpolation between such endpoints can distort angular information and reduce the resulting embedding norm. These properties motivate our use of spherical interpolation to control the transition in direction while separately preserving the [MASK] embedding norm.

## 6.4 ANALYZING AGREEMENT-SEEKING BEHAVIOR

Comparison with additive and multiplicative mixtures. Constructing the additive and multiplicative reference mixtures in Sections 3 and 5.1 requires evaluating the model under every joint discrete realization of the soft-token inputs. The number of realizations grows exponentially with the number of soft-token positions, making exhaustive evaluation impractical for long sequences. We therefore use 70 controlled prompts containing two to four masked positions, where individually plausible predictions can form inconsistent combinations. For example, a prompt that admits either New York or San Diego can yield incompatible combinations such as New Diego when the two positions are predicted independently. We measure the Jensen–Shannon (JS) divergence between each reference mixture and the predictive distributions induced by our construction, with results presented in Section 5.1. Details of the prompts and implementation are provided in Appendix D.

Sequence-level consistency. We assess sequence-level consistency using the same toy examples as in the preceding mixture analysis. We obtain per-position predictive distributions from masked inputs for vanilla parallel decoding and from the corresponding soft-token inputs for the two softtoken methods. We compute the joint probability of each complete token combination as the product of its predicted token probabilities across answer positions. As described in Section 5.2, we measure reverse KL from this factorized joint to a sequential reference.

Table 4 shows that our method achieves the lowest reverse KL across all four models. The Euclidean baseline also yields lower divergence than vanilla parallel decoding, but our method consistently provides a larger reduction. These results provide complementary evidence for the proposed agreement-seeking behavior, supporting the interpretation that our soft-token feedback favors coherent alternatives over inconsistent combinations in these controlled settings.

## 7 RELATED WORK

Diffusion language models (DLMs) have emerged as a promising alternative to autoregressive generation. D3PM introduced a framework for diffusion over discrete data using structured corruption processes (Austin et al., 2021a). Subsequent work advanced discrete diffusion through continuoustime formulations (Campbell et al., 2022; Sun et al., 2023), improved learning objectives (Meng et al., 2022; Benton et al., 2024; Lou et al., 2024), and accelerated sampling algorithms (Chen et al., 2024). Recent studies simplify and unify masked diffusion formulations, improving efficiency and scalability for language modeling (Sahoo et al., 2024; Shi et al., 2024). At larger scales, LLaDA demonstrates competitive language modeling and instruction following through training from scratch (Nie et al., 2025), while Dream adapts pretrained autoregressive models to diffusionbased generation (Ye et al., 2025). LLaDA 1.5 improves alignment through variance-reduced preference optimization (Zhu et al., 2026), and LLaDA 2.0 scales diffusion language models to 100B total parameters through progressive conversion of pretrained autoregressive models (Bie et al., 2025).

Soft tokens and continuous feedback have been explored in autoregressive models to support reasoning beyond discrete token sequences. Existing approaches use hidden-state feedback (Hao et al., 2025; Shen et al., 2025), learned soft thought representations (Xu et al., 2025a;b), compressed latent reasoning states (Tan et al., 2025), or probability-weighted embedding mixtures (Zhang et al., 2025). Inspired by the success of continuous feedback in autoregressive models, recent work incorporates continuous feedback into DLMs. Soft-Masking blends the [MASK] embedding with predictedtoken embeddings and trains models to process these inputs (Hersche et al., 2026). EvoToken-DLM introduces progressive soft-token refinement supported by continuous trajectory supervision (Zhong et al., 2026), while DMax combines on-policy uniform training with soft parallel decoding to enable iterative revision in embedding space (Chen et al., 2026). We study soft-token feedback in frozen pretrained DLMs, developing a geometry-aware construction and examining how feedback reshapes predictive distributions and improves sequence-level consistency without additional training.

Concurrent work (Nigam et al., 2026) uses spherical interpolation with iterative Riemannian candidate aggregation, evaluated through continued pretraining of a 169M-parameter model. We develop a closed-form, training-free construction supported by geometric analysis across four pretrained DLMs and investigate its effects on predictive distributions and sequence-level consistency.

## 8 LIMITATIONS AND CONCLUSION

Limitation. Our exact comparison with additive and multiplicative reference mixtures in Section 6.4 is limited to controlled prompts with two to four masked positions rather than sequences from real-world data. However, constructing the additive and multiplicative reference mixtures requires enumerating all joint discrete realizations, whose number grows exponentially with the number of soft-token positions. This makes exhaustive evaluation computationally infeasible for the longer sequences encountered in real-world datasets. The controlled setting enables exact evaluation of these mixtures while making dependencies and inconsistent token combinations explicit.

Conclusion. We introduced an agreement-seeking view of soft-token feedback for parallel decoding. Our probabilistic analysis connects multiplicative-mixture behavior with improved sequencelevel consistency, supporting the interpretation that feedback favors compatible predictions before commitment. To study this behavior without additional training, we developed a geometry-aware soft-token method that outperforms standard parallel decoding and a training-free Euclidean baseline across four frozen pretrained DLMs and four math and code benchmarks. These findings extend the understanding of soft tokens beyond uncertainty preservation. We believe this probabilistic perspective can help design more effective representations for parallel decoding.

## ETHICS STATEMENT

This work studies decoding methods for pretrained diffusion language models using public benchmarks and synthetic prompts. Our method does not address biases, harmful outputs, or potential misuse of the underlying models. These concerns remain relevant when deploying models with the proposed decoding method.

## REPRODUCIBILITY STATEMENT

We describe the evaluated models, benchmarks, and baselines in Section 6.1, with additional evaluation and decoding details in Appendix A. Appendix D documents the controlled experiments and provides the complete toy prompt collection.

## AI USE STATEMENT.

We used generative AI tools to improve the clarity and readability of the manuscript and to generate toy datasets for analyzing our method in Section 6.4. We reviewed the generated examples and AI-assisted revisions and take responsibility for the final content of this work.

## REFERENCES

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 2021a.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021b.

Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas S. Nolte, and Brian Karrer. Accelerated sampling from masked diffusion models via entropy bounded unmasking. In Proceedings ofthe Advances in Neural Information Processing Systems (NeurIPS), 2025.

Joe Benton, Yuyang Shi, Valentin De Bortoli, George Deligiannidis, and Arnaud Doucet. From denoising diffusions to denoising Markov models. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 86(2):286–301, 2024. doi: 10.1093/jrsssb/qkae005. URL https: //academic.oup.com/jrsssb/article/86/2/286/7564909.

Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, Chengxi Li, Chongxuan Li, Jianguo Li, Zehuan Li, Huabin Liu, Lin Liu, Guoshan Lu, Xiaocheng Lu, Yuxin Ma, Jianfeng Tan, Lanning Wei, Ji-Rong Wen, Yipeng Xing, Xiaolu Zhang, Junbo Zhao, Da Zheng, Jun Zhou, Junlin Zhou, Zhanchao Zhou, Liwang Zhu, and Yihong Zhuang. LLaDA2.0: Scaling up diffusion language models to 100B. arXiv preprint arXiv:2512.15745, 2025.

Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/ hash/b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html.

Hei Chan and Adnan Darwiche. On the revision of probabilistic beliefs using uncertain evidence. Artificial Intelligence, 163(1):67–90, 2005.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian,

Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob Mc-Grew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, and Xinchao Wang. Dmax: Aggressive parallel decoding for dllms. arXiv preprint arXiv:2604.08302, 2026.

Zixiang Chen, Huizhuo Yuan, Yongqian Li, Yiwen Kou, Junkai Zhang, and Quanquan Gu. Fast sampling via discrete non-Markov diffusion models with predetermined transition time. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ hash/c153077e44a810cc8728460953af54f1-Abstract-Conference.html.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 2024. URL https://zenodo.org/records/12608602.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. In Proceedings of the Conference on Language Modeling (COLM), 2025.

Michael Hersche, Samuel Moor-Smith, Thomas Hofmann, and Abbas Rahimi. Soft-masked diffusion language models. In Proceedings of the International Conference on Learning Representations (ICLR), 2026.

Daniel Israel, Guy Van den Broeck, and Aditya Grover. Accelerating diffusion LLMs via adaptive parallel decoding. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 2025.

Ian Li, Zilei Shao, Benjie Wang, Rose Yu, Guy Van den Broeck, and Anji Liu. Breaking the factorization barrier in diffusion language models. arXiv preprint arXiv:2603.00045, 2026.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2024.

Anji Liu, Oliver Broadrick, Mathias Niepert, and Guy Van den Broeck. Discrete copula diffusion. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2025.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In Proceedings of the International Conference on Machine Learning (ICML), 2024.

Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/hash/df04a35d907e894d59d4eab1f92bc87b-Abstract-Conference. html.

Andreas Munk, Alexander Mead, and Frank Wood. Uncertain evidence in probabilistic models and stochastic simulators. In Proceedings of the International Conference on Machine Learning (ICML), 2023.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 2025.

Lavanya Nigam, Ishaan Bansal, Aryan Sood, Vidit Aggarwal, and Gaurav Kumar Nayak. Lost in interpolation: Why predictive feedback fails in diffusion language models. arXiv preprint arXiv:2608.06529, 2026.

Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 2024.

Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. CODI: Compressing chain-of-thought into continuous space via self-distillation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 677–693. Association for Computational Linguistics, 2025. URL https://aclanthology.org/2025.emnlp-main. 36/.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 2024.

Ken Shoemake. Animating rotation with quaternion curves. In Proceedings of the 12th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), 1985.

Haoran Sun, Lijun Yu, Bo Dai, Dale Schuurmans, and Hanjun Dai. Score-based continuous-time discrete diffusion models. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=BYWWwSY2G5s.

Wenhui Tan, Jiaze Li, Jianzhong Ju, Zhenbo Luo, Ruihua Song, and Jian Luan. Think silently, think fast: Dynamic latent compression of LLM reasoning chains. In Advances in Neural Information Processing Systems, volume 38, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 0706261aedab63814a2b73c32564b4c4-Abstract-Conference.html.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dLLM: Training-free acceleration of diffusion LLM by enabling KV cache and parallel decoding. In Proceedings ofthe International Conference on Learning Representations (ICLR), 2026.

Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. SoftCoT: Soft chain-of-thought for efficient reasoning with LLMs. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2025a.

Yige Xu, Xu Guo, Zhiwei Zeng, and Chunyan Miao. SoftCoT++: Test-time scaling with soft chainof-thought reasoning. arXiv preprint arXiv:2505.11484, 2025b. URL https://arxiv.org/ abs/2505.11484.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7B: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 2019.

Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, and Xin Wang. Soft thinking: Unlocking the reasoning potential of LLMs in continuous concept space. In Proceedings of the Advances in Neural Information Processing Systems (NeurIPS), 2025.

Linhao Zhong, Linyu Wu, Bozhen Fang, Tianjian Feng, Chenchen Jing, Wen Wang, Jiaheng Zhang, Hao Chen, and Chunhua Shen. Beyond hard masks: Progressive token evolution for diffusion language models. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2026.

Zhanhui Zhou, Lingjie Chen, Hanghang Tong, and Dawn Song. dLLM: Simple diffusion language modeling. arXiv preprint arXiv:2602.22661, 2026.

Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. LLaDA 1.5: Variance-reduced preference optimization for large language diffusion models. In Proceedings ofthe Annual Meeting ofthe Associationfor Computational Linguistics (ACL), 2026.

## APPENDIX

This appendix is organized as follows. Appendix A provides additional evaluation and decoding details. Appendix B presents additional analyses and ablation study. Appendix C reports recovery rates relative to single-token prediction. Appendix D describes the controlled toy experiments and provides the complete prompt collection.

## A EXPERIMENTAL DETAILS

This section supplements the experimental settings in Section 6.1.

## A.1 EVALUATION SETUP

We conduct all experiments using the dLLM framework (Zhou et al., 2026) and follow the task configurations implemented in lm-evaluation-harness (Gao et al., 2024).

For LLaDA-Instruct-8B, LLaDA-1.5-Instruct, and LLaDA-2.0-mini, we use 5-shot evaluation on GSM8K, 4-shot on MATH500, 0-shot on HumanEval, and 3-shot on MBPP. We generate up to 512 tokens on GSM8K, MATH500, and HumanEval, and 256 tokens on MBPP.

For Dream-7B, we follow its default zero-shot configuration on all four benchmarks, with maximum generation lengths of 256, 512, 768, and 1,024 tokens for GSM8K, MATH500, HumanEval, and MBPP, respectively. We apply each model’s chat template and do not use block-diffusion sampling.

## A.2 DECODING CONFIGURATIONS

Decoding protocol. All model parameters remain frozen. We compare vanilla parallel decoding, the Euclidean soft-token baseline, and our geometry-aware construction at 2, 4, and 8 tokens committed per denoising iteration. Vanilla decoding retains discrete [MASK] tokens at unresolved positions, whereas the soft-token methods feed updated continuous representations into the next iteration. We report exact-match accuracy for GSM8K and MATH500 and pass@1 for HumanEval and MBPP.

Hyperparameter settings. Both soft-token methods use the same candidate aggregation, decoding schedule, and hyperparameter search space. We search over $\lambda \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 \}$ and $k \in \{ 2 , 3 , 4 \}$ We configure our method for mathematical reasoning and code generation, keeping the settings fixed across benchmarks and parallel-decoding rates within each domain. For the LLaDA family, we use $\lambda = 0 . 3$ in both domains, with $k = 2$ for mathematical reasoning and $k = 3$ for code generation. For Dream, we use $( k , \lambda ) = ( 3 , 0 . 1 )$ for mathematical reasoning and $( k , \lambda ) = ( 2 , 0 . 3 )$ for code generation. We use spherical interpolation and apply soft-token feedback throughout the denoising process. Appendix B.3 analyzes sensitivity to k and λ.

## B ADDITIONAL ANALYSES AND ABLATIONS

## B.1 GEOMETRIC LIMITATIONS OF LINEAR INTERPOLATION FROM [MASK]

To examine the geometric effects discussed in Section 4.1, we evaluate LLaDA-8B-Instruct on GSM8K and HumanEval. At each masked position, we construct a probability-weighted aggregate m of the top-four predicted token embeddings, with probabilities renormalized over these candi dates, and compute

$$
\mathbf { e } _ { \mathrm { l i n e a r } } = ( 1 - \lambda ) \mathbf { e } _ { \mathrm { M A S K } } + \lambda \mathbf { m } .
$$

We use $\lambda \ = \ 0 . 5 .$ , for which spherical interpolation between the normalized endpoint directions reaches the angular midpoint.

The candidate aggregate and the [MASK] embedding were nearly orthogonal, with a mean angular separation of $8 9 . 4 2 ^ { \circ }$ . Linear interpolation traversed, on average, 59.9% of this separation rather than 50%. Its output norm averaged 0.863 times the [MASK] norm and fell below both endpoint norms at 99.4% of positions.

These measurements show that equal Euclidean interpolation weights do not generally yield the angular midpoint and that interpolation can shrink the representation below both endpoint norms. They support controlling direction and magnitude separately when incorporating candidate information into the [MASK] representation.

## B.2 EFFECTS OF INTERPOLATION AND NORM PRESERVATION

Our construction uses spherical interpolation (SLERP) to determine the soft token’s direction and the [MASK] embedding norm to set its magnitude. We evaluate two variants to examine these choices separately. First, we retain SLERP but replace the output norm ∥e ∥ with ∥m∥ , where m is the aggregated candidate representation defined in Section 4.2. Second, we replace SLERP with linear interpolation while preserving the [MASK] embedding norm:

$$
\mathbf { e } _ { \mathrm { l i n e a r } } = \| \mathbf { e } _ { \mathrm { M A S K } } \| _ { 2 } \frac { ( 1 - \lambda ) \mathbf { e } _ { \mathrm { M A S K } } + \lambda \mathbf { m } } { \| ( 1 - \lambda ) \mathbf { e } _ { \mathrm { M A S K } } + \lambda \mathbf { m } \| _ { 2 } } .
$$

The first variant changes only the output magnitude, while the second changes only the interpolation rule used to determine the direction.

As shown in Table 5, our construction achieves the highest accuracy across all six model–benchmark pairs. The decrease under candidate-norm scaling supports preserving the [MASK] embedding norm, while the gap to norm-matched linear interpolation suggests that norm preservation alone does not account for the observed gains.

Table 5: Effects of interpolation and output norm on accuracy (%). SLERP with the [MASK] norm is our standard construction. The best result for each model–benchmark pair is shown in bold.
<table><tr><td colspan="2"></td><td colspan="2">SLERP</td><td>Linear</td></tr><tr><td>Model</td><td>Benchmark</td><td>[MASK] norm</td><td>Candidate norm</td><td>[MASK] norm</td></tr><tr><td rowspan="2">Dream</td><td>MATH-500</td><td>22.40</td><td>21.20</td><td>16.40</td></tr><tr><td>HumanEval</td><td>50.61</td><td>50.00</td><td>50.00</td></tr><tr><td rowspan="2">LLaDA</td><td>MATH-500</td><td>23.60</td><td>22.60</td><td>21.80</td></tr><tr><td>HumanEval</td><td>45.12</td><td>41.46</td><td>42.07</td></tr><tr><td rowspan="2">LLaDA 1.5</td><td>MATH-500</td><td>24.00</td><td>21.80</td><td>21.20</td></tr><tr><td>HumanEval</td><td>43.90</td><td>39.63</td><td>37.80</td></tr></table>

## B.3 INTERPOLATION STRENGTH AND CANDIDATE COUNT

The interpolation coefficient λ controls how far the soft-token direction moves from [MASK] toward the candidate aggregate, while k determines how many candidate embeddings contribute to that aggregate. Figure 2 examines sensitivity to these two parameters on HumanEval.

Interpolation strength. Smaller values of λ keep the soft-token embedding closer to [MASK], whereas larger values give greater influence to the candidate aggregate. Across the four evaluated models, λ = 0.3 achieves the best results, while larger values reduce accuracy, with the severity of the decline varying across models. We hypothesize that strong interpolation moves the embed ding too far from [MASK]. Our construction is intended to enrich an unresolved position with predictive information while retaining its role as a masked position. Excessive deviation from the pretrained [MASK] representation may make it harder for the frozen model to interpret the position as unresolved, degrading subsequent predictions. These results favor moderate interpolation that incorporates candidate information while maintaining proximity to [MASK].

Candidate count. Accuracy is comparatively stable across the evaluated candidate counts, and increasing k does not consistently improve performance. A plausible explanation is that probability weighting limits the influence of additional, lower-probability candidates on the aggregated representation. This limited sensitivity makes the method less dependent on precise tuning of the candidate count.

![](images/eb4083c07ae88e61382def89814996ff8abab9d25bd2f0b8ec72daedf387ff50.jpg)  
(a) Interpolation coefficient λ.

![](images/997a229597360f8f13fe0aca77e4474a57cb23f9e2ca316982a89d27ca8cf496.jpg)  
(b) Number of candidate tokens k.  
Figure 2: Sensitivity to interpolation strength and candidate count in soft-token decoding.

Table 6: Combining our soft-token feedback with adaptive samplers. We report GSM8K accuracy and HumanEval pass@1 (%; higher is better). Confidence thresholding uses the decoding rule from Fast-dLLM. Bold values indicate improvements over the corresponding sampler without our feedback.

<table><tr><td></td><td colspan="2">LLaDA</td><td colspan="2">Dream</td></tr><tr><td>Sampler</td><td>GSM8K</td><td>HumanEval</td><td>GSM8K</td><td>HumanEval</td></tr><tr><td>Confidence thresholding</td><td>29.80</td><td>45.12</td><td>79.80</td><td>55.49</td></tr><tr><td>+ Ours</td><td>30.20</td><td>48.78</td><td>81.60</td><td>58.54</td></tr><tr><td>EB-Sampler</td><td>24.00</td><td>46.34</td><td>82.60</td><td>56.71</td></tr><tr><td>+ Ours</td><td>26.00</td><td>48.17</td><td>83.60</td><td>58.54</td></tr></table>

## B.4 COMBINING SOFT TOKENS WITH ADAPTIVE SAMPLING

We further investigate whether our soft-token feedback complements adaptive sampling strategies. Specifically, we incorporate our method into confidence-threshold decoding from Fast-dLLM (Wu et al., 2026) and entropy-bounded unmasking in EB-Sampler (Ben-Hamu et al., 2025). We set the confidence threshold to 0.9 and the entropy bound to 0.1. These strategies adaptively determine which tokens to commit at each step, while our method refines the predictive distributions through soft-token feedback. We evaluate each sampler with and without our feedback on GSM8K and HumanEval using LLaDA and Dream.

As shown in Table 6, our method improves both samplers across all evaluated model–dataset combinations, with gains ranging from 0.40 to 3.66 percentage points. These results suggest that our soft-token feedback complements adaptive token selection and that its benefits extend beyond fixedsize parallel decoding.

## C RECOVERY OF SINGLE-TOKEN PREDICTIONS

Beyond aggregate accuracy, we examine whether parallel decoding preserves correctness on examples solved by single-token prediction (STP), which commits one token per denoising iteration. Similar aggregate accuracy does not necessarily imply such preservation. A parallel method can match STP’s accuracy while failing on many examples that STP solves, with these losses offset by successes on other examples. We observe this discrepancy in some settings, motivating a direct assessment of how well parallel decoding retains STP’s successes.

Let $S _ { \mathrm { S T P } }$ and $S _ { \mathrm { p a r a l l e l } }$ denote the sets of examples correctly solved by STP and a given parallel decoding method, respectively. We define recovery rate as

$$
{ \mathrm { R e c o v e r y ~ r a t e } } = 1 0 0 \times { \frac { | S _ { \mathrm { { S T P } } } \cap S _ { \mathrm { { p a r a l l e l } } } | } { | S _ { \mathrm { { S T P } } } | } } .
$$

This metric measures the percentage of examples solved by STP that remain correctly solved under parallel decoding. It gives no credit for examples solved only by the parallel method, and therefore complements overall accuracy.

Table 7 reports recovery rates for mathematical reasoning and code generation. Recovery rates decrease as more tokens are committed per iteration, indicating that greater parallelism makes it harder to retain correctness on examples solved by STP. Our method achieves the highest recovery rate in most settings, suggesting that its accuracy gains are accompanied by better retention of STP-correct examples. For example, on HumanEval at eight tokens per step, our method improves recovery over vanilla parallel decoding from 22.78% to 37.97% for LLaDA-1.5 and from 5.26% to 24.21% for Dream. Nevertheless, recovery remains low in some highly parallel settings, indicating that softtoken feedback mitigates but does not eliminate the degradation associated with parallel decoding.

Table 7: Recovery rate (%) on mathematical reasoning and code generation. Columns labeled 2, 4, and 8 indicate tokens committed per denoising iteration. The best result for each model, benchmark, and decoding setting is in bold.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="3">GSM8K</td><td colspan="3">MATH500</td><td colspan="3">HumanEval</td><td colspan="3">MBPP</td></tr><tr><td>2</td><td>4</td><td>8</td><td>2</td><td>4</td><td>8</td><td>2</td><td>4</td><td></td><td>8</td><td>2</td><td>8</td></tr><tr><td rowspan="3">LLaDA-2.0 mini</td><td>Vanilla</td><td>95.50</td><td>88.33</td><td>53.67</td><td>88.83</td><td>82.04</td><td>37.38</td><td>83.33</td><td>56.82</td><td>28.79</td><td>83.70</td><td>64.89</td><td>34.99</td></tr><tr><td>Euclidean</td><td>95.41</td><td>91.92</td><td>55.24</td><td>92.72</td><td>81.07</td><td>41.26</td><td>81.06</td><td>62.12</td><td>28.03</td><td>85.89</td><td>66.46</td><td>34.80</td></tr><tr><td>Ours</td><td>96.94</td><td>93.23</td><td>58.73</td><td>94.17</td><td>85.92</td><td>42.23</td><td>85.61</td><td>64.39</td><td>28.03</td><td>85.58</td><td>66.46</td><td>35.42</td></tr><tr><td rowspan="3">LLaDA-1.5</td><td>Vanilla</td><td>91.25</td><td>83.35</td><td>56.59</td><td>71.31</td><td>59.84</td><td>40.00</td><td>70.89</td><td>46.84</td><td>22.78</td><td>78.50</td><td>61.00</td><td>38.50</td></tr><tr><td>Euclidean</td><td>89.99</td><td>84.62</td><td>61.01</td><td>68.85</td><td>59.02</td><td>46.72</td><td>69.62</td><td>54.43</td><td>26.58</td><td>78.50</td><td>59.50</td><td>40.50</td></tr><tr><td>Ours</td><td>90.83</td><td>85.67</td><td>61.22</td><td>76.23</td><td>59.84</td><td>47.54</td><td>74.68</td><td>59.49</td><td>37.97</td><td>79.00</td><td>62.00</td><td>40.00</td></tr><tr><td rowspan="3">LLaDA</td><td>Vanilla</td><td>89.24</td><td>78.07</td><td>45.89</td><td>67.19</td><td>51.56</td><td>38.28</td><td>76.32</td><td>48.68</td><td>30.26</td><td>73.00</td><td>61.00</td><td>38.00</td></tr><tr><td>Euclidean</td><td>89.34</td><td>79.19</td><td>45.79</td><td>65.63</td><td>55.47</td><td>40.63</td><td>75.00</td><td>59.21</td><td>28.95</td><td>75.50</td><td>58.00</td><td>38.00</td></tr><tr><td>Ours</td><td>90.05</td><td>80.41</td><td>45.89</td><td>71.88</td><td>55.47</td><td>41.41</td><td>85.53</td><td>60.53</td><td>35.53</td><td>78.50</td><td>64.00</td><td>44.50</td></tr><tr><td rowspan="3">Dream</td><td>Vanilla</td><td>84.74</td><td>65.03</td><td>18.41</td><td>42.37</td><td>11.70</td><td>0.58</td><td>66.32</td><td>29.47</td><td>5.26</td><td>70.45</td><td>39.86</td><td>23.71</td></tr><tr><td>Euclidean</td><td>86.68</td><td>64.20</td><td>18.32</td><td>42.69</td><td>13.45</td><td>0.59</td><td>72.63</td><td>38.95</td><td>13.68</td><td>70.80</td><td>40.50</td><td>24.41</td></tr><tr><td>Ours</td><td>87.60</td><td>66.98</td><td>20.44</td><td>46.20</td><td>13.45</td><td>1.17</td><td>74.74</td><td>54.74</td><td>24.21</td><td>74.23</td><td>53.61</td><td>28.09</td></tr></table>

## D SUPPLEMENTARY ANALYSIS OF SOFT-TOKEN FEEDBACK

This appendix supplements the analysis in Section 6.4. We use controlled prompts to compare softtoken predictions with additive and multiplicative reference mixtures and to evaluate agreement with a sequential reference. We then extend the reverse-KL comparison to model-generated completions. The complete controlled prompt collection is provided in Appendix D.5.

## D.1 PROMPT DESIGN

We use 70 controlled prompts: 30 with two mask placeholders, 20 with three, and 20 with four. Each prompt specifies two alternative answers. For example, a two-mask prompt asks for either New York or San Diego. Independent predictions can mix these alternatives to produce New Diego or San York, violating the stated choice. These prompts therefore expose dependencies between predicted positions and allow us to examine whether soft-token feedback favors mutually consistent predictions.

We use the same prompt collection for reference-mixture comparison and the sequence-level consistency analysis. The prompts were generated with assistance from generative AI and reviewed by the authors, as described in the AI Use Statement. The complete collection appears in Tables 9–11.

## D.2 COMPARISON WITH REFERENCE MIXTURES

We compare the predictive distributions induced by Euclidean soft tokens and our construction with the additive and multiplicative references defined in Section 3. We use the controlled prompts described in Appendix D.1.

Soft-token predictions. For each prompt, we first evaluate the frozen model with all answer positions masked. We use the resulting predictive distributions to select candidate tokens and construct the soft-token embeddings. The continuous-input predictions are obtained from one additional forward pass with these embeddings, without committing discrete tokens or performing further feed back updates. We use $k = 3$ and $\lambda = 0 . 3$ for both soft-token methods. The initial and feedback predictive distributions are computed using the model’s unscaled softmax.

Discrete reference construction. For a queried position $j ,$ let $S _ { j }$ denote the set of input positions represented by soft tokens in the comparison. At each position in $S _ { j }$ , we consider the selected candidate tokens and the unresolved [MASK] state. Let ${ \mathcal { Z } } _ { j }$ denote the Cartesian product of these positionwise state sets. With k candidates per position, this gives up to $( k + 1 ) ^ { | S _ { j } | }$ joint realizations. We enumerate all realizations, keeping the prompt and the remaining input positions fixed.

For each queried answer position $j , s _ { j }$ contains the positions represented by soft tokens in the comparison, including $j$ itself. We enumerate their joint discrete realizations by replacing each soft-token embedding with either the [MASK] embedding or one of its selected candidate-token embeddings. The prompt and all inputs outside $S _ { j }$ remain fixed.

Let $w _ { j } ( \mathbf { z } )$ be the weight of realization $\mathbf { z } \in { \mathcal { Z } } _ { j } .$ , obtained by multiplying the positionwise realization weights defined in Section 3. Let $P _ { j , \mathbf { z } }$ denote the model’s predictive distribution at position j under this discrete input. The two references are

$$
P _ { \mathrm { a d d } , j } ( y ) = \sum _ { \mathbf { z } \in \mathcal { Z } _ { j } } w _ { j } ( \mathbf { z } ) P _ { j , \mathbf { z } } ( y ) ,
$$

and

$$
P _ { \mathrm { m u l t } , j } ( y ) = \frac { \displaystyle \exp \left( \sum _ { \mathbf z \in \mathcal Z _ { j } } w _ { j } ( \mathbf z ) \log P _ { j , \mathbf z } ( y ) \right) } { \displaystyle \sum _ { y ^ { \prime } } \exp \left( \sum _ { \mathbf z \in \mathcal Z _ { j } } w _ { j } ( \mathbf z ) \log P _ { j , \mathbf z } ( y ^ { \prime } ) \right) } .
$$

The candidate probabilities are taken from the initial full-vocabulary distribution and renormalized over the selected top-k candidates. Writing these normalized probabilities as $\bar { p _ { i } }$ , the positionwise realization weights are 1−λ for [MASK] and $\lambda { \bar { p } } _ { i }$ for candidate i. The additive and multiplicative references use identical candidate sets, discrete realizations, and realization weights. They differ only in their pooling operation: a weighted arithmetic mean for the additive reference and a normalized weighted geometric mean for the multiplicative reference. Discrete-input predictive distributions are computed using the model’s softmax without temperature scaling.

Divergence and aggregation. For each method and queried position, we compare the continuousinput prediction with each reference using Jensen–Shannon divergence:

$$
\mathrm { J S } ( P , Q ) = \frac { 1 } { 2 } D _ { \mathrm { K L } } ( P \Vert M ) + \frac { 1 } { 2 } D _ { \mathrm { K L } } ( Q \Vert M ) , \qquad M = \frac { P + Q } { 2 } .
$$

We compute JS divergence over the full output vocabulary using natural logarithms. We first average the divergences over queried positions within each prompt, then average over prompts, giving each prompt equal weight regardless of its number of answer positions.

Table 1 reports the resulting divergences for each model. Lower values indicate closer agreement with the corresponding reference.

## D.3 SEQUENCE-LEVEL CONSISTENCY

Following Section 5.2, we evaluate whether soft-token feedback brings factorized parallel predictions closer to a sequential reference that incorporates dependencies among answer tokens. Using the controlled prompt collection, we enumerate candidate answer sequences and compare their prob abilities under each method and the reference.

Candidate answer support. For a prompt with m masked answer positions, let $\mathbf { a } = ( a _ { 1 } , \dots , a _ { m } )$ and $\mathbf { b } ~ = ~ ( b _ { 1 } , \ldots , b _ { m } )$ be the token sequences corresponding to its two stated alternatives. We tokenize each alternative using the evaluated model’s tokenizer, preserving the whitespace at the answer boundary. This comparison includes only prompts for which both alternatives occupy exactly m answer tokens.

We define the shared support as

$$
S = \left\{ \mathbf { y } \in \mathcal { V } ^ { m } : y _ { j } \in \left\{ a _ { j } , b _ { j } \right\} \mathrm { f o r } \mathrm { e v e r y } j \right\} .
$$

This support contains both stated alternatives and every distinct positionwise recombination of their tokens. Its size is

$$
| S | = \prod _ { j = 1 } ^ { m } | \{ a _ { j } , b _ { j } \} | \leq 2 ^ { m } .
$$

Thus, two-, three-, and four-token examples contain at most four, eight, and sixteen candidate sequences, respectively. Shared tokens between the alternatives reduce these counts. The support is determined by the stated alternatives and is independent of the top-k candidates used to construct soft-token embeddings.

Order-averaged sequential reference. Let x denote the context with all m answer positions masked, and let $\Pi _ { m }$ be the set of all permutations of these positions. For a candidate sequence $\mathbf { y } \in S$ and an order $\pi = ( \pi _ { 1 } , \ldots , \pi _ { m } ) \in \Pi _ { m }$ , we score the sequence by teacher forcing. At step t, we record the probability of $y _ { \pi _ { t } }$ and insert that token before proceeding. Writing $\mathbf { x } _ { < t } ^ { \pi } ( \mathbf { y } )$ for the context after the first $t - 1$ insertions, the joint probability under order π is

$$
p _ { \pi } ( \mathbf { y } \mid \mathbf { x } ) = \prod _ { t = 1 } ^ { m } p _ { \theta } ( y _ { \pi _ { t } } \mid \mathbf { x } _ { < t } ^ { \pi } ( \mathbf { y } ) ) .
$$

The prompt remains fixed, and answer positions not yet filled remain masked. All conditionals use the model’s full-vocabulary softmax without temperature scaling.

We define the sequential reference by uniformly averaging the joint probabilities:

$$
p _ { \mathrm { r e f } } ( \mathbf { y } \mid \mathbf { x } ) = { \frac { 1 } { m ! } } \sum _ { \pi \in \Pi _ { m } } p _ { \pi } ( \mathbf { y } \mid \mathbf { x } ) .
$$

All two, six, or twenty-four orders are enumerated for $m = 2 , 3 ,$ or 4, respectively. The reference uses discrete token and [MASK] inputs without soft-token feedback and is shared across methods.

Factorized parallel predictions. Vanilla parallel decoding uses the predictive distributions from an initial forward pass with all answer positions masked. For Euclidean soft tokens and our construction, these initial predictions are used to construct soft-token embeddings at every answer position. We then obtain updated predictions from one additional forward pass, without committing discrete tokens or performing further feedback updates. Both soft-token methods use $k = 3$ and $\lambda = 0 . 3$ with candidate probabilities renormalized over the top-k tokens, as specified in Appendix D.2.

Let $q _ { j } ^ { ( d ) } ( \cdot \mid \mathbf { x } )$ be the resulting predictive distribution at position $j$ for method $d .$ Each method assigns a factorized probability to a candidate sequence:

$$
q ^ { ( d ) } ( \mathbf { y } \mid \mathbf { x } ) = \prod _ { j = 1 } ^ { m } q _ { j } ^ { ( d ) } ( y _ { j } \mid \mathbf { x } ) .
$$

These probabilities are computed from full-vocabulary softmax distributions without temperature scaling.

Conditional reverse KL. We separately normalize each method’s prediction and the sequential reference over the shared support:

$$
q _ { S } ^ { ( d ) } ( \mathbf { y } \mid \mathbf { x } ) = { \frac { q ^ { ( d ) } ( \mathbf { y } \mid \mathbf { x } ) } { \displaystyle \sum _ { \mathbf { z } \in S } q ^ { ( d ) } ( \mathbf { z } \mid \mathbf { x } ) } } , \qquad p _ { \mathrm { r e f } , S } ( \mathbf { y } \mid \mathbf { x } ) = { \frac { p _ { \mathrm { r e f } } ( \mathbf { y } \mid \mathbf { x } ) } { \displaystyle \sum _ { \mathbf { z } \in S } p _ { \mathrm { r e f } } ( \mathbf { z } \mid \mathbf { x } ) } } .
$$

We then evaluate

$$
D _ { \mathrm { K L } } \Big ( q _ { S } ^ { ( d ) } \lVert p _ { \mathrm { r e f } , S } \Big ) = \sum _ { \mathbf { y } \in S } q _ { S } ^ { ( d ) } ( \mathbf { y } \mid \mathbf { x } ) \log \frac { q _ { S } ^ { ( d ) } ( \mathbf { y } \mid \mathbf { x } ) } { p _ { \mathrm { r e f } , S } ( \mathbf { y } \mid \mathbf { x } ) } .
$$

We use natural logarithms and compute the sum exactly, without sampling candidate sequences or unmasking orders.

Table 4 reports the mean divergence in nats, giving each evaluated prompt equal weight. Lower values indicate closer agreement with the sequential reference within S. Because both distributions are conditioned on this support, the metric does not evaluate their total probability mass on $S$ or discrepancies outside it. The reference is derived from the same model rather than a ground-truth joint distribution, so this comparison serves as a diagnostic of the interpretation in Section 5.2.

## D.4 REVERSE-KL EVALUATION ON MODEL-GENERATED COMPLETIONS

To complement the controlled-prompt analysis, we examine agreement with a sequential reference on spans drawn from model-generated completions. This experiment tests whether the same qualitative pattern extends to mathematical reasoning and code-generation contexts.

Evaluation setup. We evaluate spans containing $m \in \{ 2 , 3 , 4 \}$ consecutive completion tokens. Because the number of possible token sequences grows exponentially with span length, we use the finite-support procedure described below. We use 100 completions for each combination of MATH-500 or HumanEval and Dream-v0-Instruct-7B, LLaDA-1.5, or LLaDA-8B-Instruct. For each completion, we select two approximately evenly spaced evaluation locations. The surrounding context, including tokens to the right of each masked span, remains visible. The evaluation therefore concerns bidirectional infilling with two to four masked tokens.

Compared methods. Discrete MTP predicts all masked positions independently from the initial mask-only forward pass. For the soft-token methods, this pass provides the predictive probabilities, which we renormalize over the top-four candidates to construct

$$
\mathbf { e } _ { \mathrm { t o k } } = \sum _ { v \in \mathrm { T o p 4 } ( p ) } \frac { p ( v ) } { \sum _ { u \in \mathrm { T o p 4 } ( p ) } p ( u ) } \mathbf { e } _ { v } .
$$

Linear and SLERP feedback use

$$
\mathbf { e } _ { \mathrm { l i n e a r } } = ( 1 - \lambda ) \mathbf { e } _ { \mathrm { M A S K } } + \lambda \mathbf { e } _ { \mathrm { t o k } } ,
$$

and

$$
\mathbf { e } _ { \mathrm { S L E R P } } = \Vert \mathbf { e } _ { \mathrm { M A S K } } \Vert _ { 2 } \mathrm { S L E R P } \left( \frac { \mathbf { e } _ { \mathrm { M A S K } } } { \Vert \mathbf { e } _ { \mathrm { M A S K } } \Vert _ { 2 } } , \frac { \mathbf { e } _ { \mathrm { t o k } } } { \Vert \mathbf { e } _ { \mathrm { t o k } } \Vert _ { 2 } } ; \lambda \right) ,
$$

respectively. Both soft-token methods use four candidates, $\lambda = 0 . 3 ,$ and one feedback step. For each method a, the resulting marginals define the factorized probability of a candidate tuple $z =$ $( z _ { 1 } , \ldots , z _ { m } ) \colon$

$$
{ \widetilde { p } } _ { a } ( z ) = \prod _ { j = 1 } ^ { m } p _ { a } ( z _ { j } \mid x ) ,
$$

where x denotes the context with all m span positions masked.

Sequential reference. Unlike the order-averaged reference used for the controlled prompts, this experiment uses confidence-ordered sequential scoring. For each candidate tuple, we repeatedly select the remaining masked position with the highest maximum token probability, score its candidate token, and insert that token before evaluating the next position. Let $x _ { < t } ( z )$ denote the context after inserting the first $t - 1$ selected tokens from $z ,$ with $x _ { < 1 } ( z ) = x ,$ and let $i _ { t } ( z )$ denote the position selected from this context. The sequential reference is

$$
q _ { \mathrm { s e q } } ( z ) = \prod _ { t = 1 } ^ { m } p _ { \theta } \bigl ( z _ { i _ { t } ( z ) } \mid x _ { < t } ( z ) \bigr ) .
$$

The first selected position is shared across candidate tuples, whereas subsequent positions may depend on the previously inserted candidate tokens.

Table 8: Mean conditional reverse KL on the shared support $S _ { 4 } ,$ in nats. Each value averages 100 completion-level measurements. Lower is better; the lowest mean in each row is bold.
<table><tr><td>Model</td><td>Dataset</td><td>SLERP</td><td>Linear</td><td>Discrete MTP</td></tr><tr><td>Dream</td><td>HumanEval</td><td>0.000704</td><td>0.084962</td><td>0.001848</td></tr><tr><td></td><td>MATH-500</td><td>0.000554</td><td>0.003955</td><td>0.001372</td></tr><tr><td>LLaDA-1.5</td><td>HumanEval</td><td>0.035973</td><td>0.037907</td><td>0.038491</td></tr><tr><td></td><td>MATH-500</td><td>0.015964</td><td>0.016934</td><td>0.017006</td></tr><tr><td>LLaDA-8B</td><td>HumanEval</td><td>0.000665</td><td>0.000918</td><td>0.000977</td></tr><tr><td></td><td>MATH-500</td><td>0.053294</td><td>0.053845</td><td>0.065190</td></tr></table>

Finite-support evaluation. Full-vocabulary enumeration requires evaluating $V ^ { m }$ token tuples. We instead construct a shared candidate set $S _ { r }$ by combining the top-r Cartesian supports from all three methods with candidates from confidence-ordered sequential beam search, using branching factor r and beam width $r ^ { 2 }$ . Special tokens are excluded and duplicate tuples are removed. The primary evaluation uses $r = 4$

We normalize both distributions on the shared support:

$$
p _ { a , S _ { r } } ( z ) = \frac { \widetilde { p } _ { a } ( z ) } { \sum _ { z ^ { \prime } \in S _ { r } } \widetilde { p } _ { a } ( z ^ { \prime } ) } , \qquad q _ { S _ { r } } ( z ) = \frac { q _ { \mathrm { s e q } } ( z ) } { \sum _ { z ^ { \prime } \in S _ { r } } q _ { \mathrm { s e q } } ( z ^ { \prime } ) } .
$$

We then compute

$$
D _ { \mathrm { K L } } ( p _ { a , S _ { r } } \parallel q _ { S _ { r } } ) = \sum _ { z \in S _ { r } } p _ { a , S _ { r } } ( z ) \log \frac { p _ { a , S _ { r } } ( z ) } { q _ { S _ { r } } ( z ) } .
$$

This measures reverse KL between distributions conditioned on $S _ { r } ;$ discrepancies outside this support are not evaluated.

Aggregation and results. We first average the span-level divergences within each completion and then average over the 100 completions for each model–dataset pair.

Table 8 reports the mean conditional reverse KL for $r = 4$ . SLERP achieves the lowest observed mean in all six settings. These results are consistent with improved agreement with the sequential reference within the evaluated support for bidirectional infilling with two to four masked tokens. The comparison evaluates the complete feedback constructions, including their different treatments of interpolation and output norm.

## D.5 COMPLETE PROMPT COLLECTION

Tables 9–11 list the complete prompt collection, grouped by the number of mask placeholders. Each table separates a shared prompt template from the variable text, with the caption specifying how to reconstruct each prompt. The two-mask table includes a lead-in column because its introductory wording varies, whereas the three- and four-mask tables use a fixed lead-in.

Table 9: Complete two-mask prompt collection (30 prompts). Each row instantiates the template: {Lead-in} either {Alternative A} or {Alternative B}. The answer is [MASK][MASK].
<table><tr><td>Lead-in</td><td>Alternative A</td><td>Alternative B</td></tr><tr><td>The location is</td><td>New York</td><td>San Diego</td></tr><tr><td>The movie series is</td><td>Star Wars</td><td>Harry Potter</td></tr><tr><td>The framework is</td><td>React Native</td><td>Swift UI</td></tr><tr><td>The dessert is</td><td>ice cream</td><td>apple pie</td></tr><tr><td>The pet is</td><td>black cat</td><td>white dog</td></tr><tr><td>The meal is</td><td>fried rice</td><td>tomato soup</td></tr><tr><td>The sport is</td><td>table tennis</td><td>ice hockey</td></tr><tr><td>The measure is</td><td>heart rate</td><td>blood pressure</td></tr><tr><td>The role is</td><td>math teacher</td><td>history student</td></tr><tr><td>The item is</td><td>train ticket</td><td>hotel room</td></tr><tr><td>The event is</td><td>forest fire</td><td>ocean wave</td></tr><tr><td>The structure is</td><td>steel bridge</td><td>stone tower</td></tr><tr><td>The animal is</td><td>polar bear</td><td>sea turtle</td></tr><tr><td>The role is</td><td>software engineer</td><td>product manager</td></tr><tr><td>The object is</td><td>golden ring</td><td>silver coin</td></tr><tr><td>The instrument is</td><td>electric guitar</td><td>grand piano</td></tr><tr><td>The transport is</td><td>city bus</td><td>cargo train</td></tr><tr><td>The drink is</td><td>orange juice</td><td>green tea</td></tr><tr><td>The clothing is</td><td>winter jacket</td><td>summer dress</td></tr><tr><td>The service is</td><td>cloud storage</td><td>mobile network</td></tr><tr><td>The publication is</td><td>science book</td><td>travel guide</td></tr><tr><td>The landscape is</td><td>rose garden</td><td>pine forest</td></tr><tr><td>The building is</td><td>city library</td><td>village school</td></tr><tr><td>The communication is</td><td>email message</td><td>phone call</td></tr><tr><td>The schedule has</td><td>morning meeting</td><td>evening class</td></tr><tr><td>The artwork is</td><td>oil painting</td><td>pencil drawing</td></tr><tr><td>The finance term is</td><td>bank loan</td><td>credit score</td></tr><tr><td>The condition is</td><td>hand surgery</td><td>knee injury</td></tr><tr><td>The house part is</td><td>front door</td><td>back window</td></tr><tr><td>The media format is</td><td>news article</td><td>radio show</td></tr></table>

Table 10: Complete three-mask prompt collection (20 prompts). Each row instantiates the template: The answer is either {Alternative A} or {Alternative B}: [MASK][MASK][MASK].
<table><tr><td>Alternative A</td><td>Alternative B</td></tr><tr><td>red apple pie</td><td>green bean soup</td></tr><tr><td>silver sports car</td><td>black pickup truck</td></tr><tr><td>summer music festival</td><td>winter sports tournament</td></tr><tr><td>mountain bike trail</td><td>coastal hiking path</td></tr><tr><td>modern art museum</td><td>ancient history archive</td></tr><tr><td>solar energy research</td><td>marine biology laboratory</td></tr><tr><td>classical piano concert</td><td>modern dance performance</td></tr><tr><td>heart surgery team</td><td>cancer research center</td></tr><tr><td>primary school teacher</td><td>college football coach</td></tr><tr><td>mountain train journey</td><td>coastal bus route</td></tr><tr><td>tropical rain forest</td><td>desert sand storm</td></tr><tr><td>global sales report</td><td>local market survey</td></tr><tr><td>wooden kitchen cabinet</td><td>metal office desk</td></tr><tr><td>mobile phone charger</td><td>laptop power adapter</td></tr><tr><td>heavy winter snow</td><td>strong summer rain</td></tr><tr><td>senior software engineer</td><td>junior product designer</td></tr><tr><td>professional tennis player</td><td>college basketball coach</td></tr><tr><td>lunar research station</td><td>solar power satellite</td></tr><tr><td>spicy chicken curry</td><td>sweet apple pastry</td></tr><tr><td>urban public park</td><td>rural community center</td></tr></table>

Table 11: Complete four-mask prompt collection (20 prompts). Each row instantiates the template: The answer is either {Alternative A} or {Alternative B}: [MASK][MASK][MASK][MASK].
<table><tr><td>Alternative A</td><td>Alternative B</td></tr><tr><td>New York subway station black leather office chair</td><td>San Diego beach hotel</td></tr><tr><td>summer music festival ticket mountain bike repair shop</td><td>white wooden dining table winter sports tournament pass</td></tr><tr><td></td><td>coastal hiking guide office</td></tr><tr><td>modern science fiction magazine international airport security checkpoint</td><td>classic detective mystery novel</td></tr><tr><td></td><td>local railway ticket office</td></tr><tr><td>fresh fruit market stall</td><td></td></tr><tr><td>public health research center</td><td>used book store counter</td></tr><tr><td></td><td>private medical training school</td></tr><tr><td>mountain railway ticket office</td><td>coastal airport security gate</td></tr><tr><td>fresh vegetable soup recipe</td><td>warm chocolate cake recipe</td></tr><tr><td>elementary school science teacher</td><td>university history department chair</td></tr><tr><td>national football league game</td><td>local basketball team practice</td></tr><tr><td>tropical rain forest reserve</td><td>desert sand storm warning</td></tr><tr><td>global technology research company</td><td>local community health organization</td></tr><tr><td>heavy winter snow storm</td><td>strong summer rain shower</td></tr><tr><td>senior software engineering manager</td><td>junior product design specialist</td></tr><tr><td>international lunar research station</td><td></td></tr><tr><td>spicy chicken curry recipe</td><td>private orbital tourism company</td></tr><tr><td></td><td>sweet apple pastry recipe</td></tr><tr><td>coastal wildlife protection program</td><td>urban water conservation project</td></tr><tr><td>secure cloud storage service</td><td>public mobile network provider</td></tr></table>