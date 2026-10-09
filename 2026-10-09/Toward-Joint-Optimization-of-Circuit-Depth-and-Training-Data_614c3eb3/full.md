# Toward Joint Optimization of Circuit Depth and Training Data Size in Adaptively Grown Quantum Classifiers

Saeefa Rubaiyet Nowmi   
Old Dominion University   
Norfolk, VA, USA   
snowm001@odu.edu

Md Mahmuduzzaman Kamol Old Dominion University Norfolk, VA, USA mkamo001@odu.edu

Mohammad Saidur Rahman   
University of Texas at El Paso El Paso, TX, USA msrahman3@utep.edu

## Abstract

Building a quantum model involves a tradeoff: how complex the circuit should be, and how much training data it needs. Caro et al. show that models with fewer trainable gates need less training data to generalize well. Q-FLAIR shows that a quantum feature-map circuit can be grown gate-by-gate, stopping once further growth stops improving the training loss. We ask whether these two results combine into a predictable scaling law. Does Q-FLAIR's own stopping rule pick larger or smaller circuits as training data grows? Does the resulting generalization behavior track Caro et al.'s bound?

We reimplement Q-FLAIR's growth mechanism faithfully, including its analytic reconstruction and exact stopping rule. We run it on full-resolution (784-pixel) MNIST 3-vs-5 classification, at five training-set sizes from N = 2000 to 10000. We then fine-tune each resulting circuit, so we can measure Caro et al.'s notion of active gates, K.

We find no predictable relationship between training-set size and the circuit size Q-FLAIR converges to. Circuit size and test accuracy both vary non monotonically with N, and seed-to-seed variance is nearly as large as any trend across N. The empirical generalization gap never exceeds Caro et al.'s bound in 14 of 15 runs, so the bound holds as a valid guarantee. But the gap correlates only weakly with the bound's value $( r = 0 . 1 2 )$ . This shows that K does not explain most of the variation we observe. Why a valid guarantee can coexist with such weak predictive power remains an open question, and answering it may be necessary before circuit depth and training data size can be jointly optimized in practice.

## 1 Introduction

Quantum feature maps embed classical data into quantum states, and this idea underlies most modern quantum machine learning [4, 7]. A common design choice re-encodes data across multiple layers, a technique known as data re-uploading [6]. Building such a model requires two largely independent decisions. The first is how much training data to use. The second is how complex the variational circuit should be.

Caro et al. [2] give a theoretical answer to the second question, given the first. A model with T trainable gates generalizes with error $O ( { \sqrt { T \log T / N } } )$ . This tightens to $O ( \sqrt { K \log ( M T ) / N } )$ when only $K \ll T$ gates change substantially during training. Q-FLAIR [5] answers a complementary question. Given a fixed training set, it grows a feature-map circuit gate-by-gate from empty. It uses an efficient analytic reconstruction to pick each gate's best weight and data feature, and it does this classically. Growth stops once further growth no longer improves the training loss.

Adaptive circuit growth is not unique to Q-FLAIR. ADAPT-VQE grows ansatze gate-by-gate for molecular simulation [3], and VAns grows and prunes circuit structure during training [1]. Caro et al. use VAns in their own unitary compiling experiments. Neither ADAPT-VQE nor VAns, however, is studied as a function of training-set size, and neither work asks the question their combination with a generalization bound raises.

That question is this: if Q-FLAIR's growth algorithm is handed different amounts of training data, does the circuit size it converges to change in a way consistent with Caro et al.'s theory? A related question also follows: is the number of trainable gates required systematically correlated with the size of the training data? Answering these questions requires actually running Q-FLAIR's mechanism, not a fixed circuit family, across a range of N. This is what we do here, on full-resolution (unreduced, 784-pixel) MNIST 3-vs-5 classification. This setting matches Q-FLAIR's own "MNIST $2 8 \times 2 8 ^ { \prime \prime }$ benchmark dimensionality exactly [5].

Contributions. (1) A faithful reimplementation of Q-FLAIR's QNN growth mechanism: the exact gate pool, observable, reconstruction, and stopping rule. We make this tractable at $d = 7 8 4$ by using a bounded scalar optimizer in place of brute-force grid search for the classical feature-selection step. (2) A joint fine-tuning stage that makes Caro et al.'s K measurable for a Q-FLAIR-grown circuit, something neither source method provides on its own. It includes an explicit safety comparison against Q-FLAIR's own pre-fine-tune result.

(3) An empirical finding: on high-dimensional raw-pixel data, circuit size and accuracy vary nonmonotonically and noisily with ${ \bf \check { N } } ,$ with no discernible scaling trend. This indicates that a straightforward joint depth/data scaling relationship does not emerge automatically from combining these two methods, and that achieving one in practice will require further exploration.

## 2 Related Work

Caro et al. [2] validate their generalization bound through two numerical studies: a quantum convolutional neural network applied to phase classification, and a separately-grown circuit (VAns) applied to unitary compiling. In both cases, the circuit architecture is either fixed a priori or grown once and subsequently held constant, with the training-set size N never varied for a given architecture. Neither study therefore examines how an adaptive growth algorithm's own output would respond to changes in the amount of available training data.

Q-FLAIR [5] grows a feature-map circuit gate-by-gate from an empty ansatz. At each iteration, it employs an analytic reconstruction to select the best-performing gate, weight, and data feature entirely through classical computation, terminating once further growth ceases to improve the training loss. Across their reported experiments, N is held fixed for each task under consideration, and the question of how the algorithm's converged output depends on N is not addressed.

The present work addresses this gap directly. We execute Q-FLAIR's growth mechanism, without modification, across a range of values of N, and characterize how its output varies as a result. We further introduce a short post-hoc fine-tuning procedure, which renders Caro et al.'s notion of active gates, K, measurable for the circuits that Q-FLAIR produces.

## 3 Method

Growth (Phase 1). We reimplement Q-FLAIR's gate pool exactly (their Eq. 25): $V =$ $\{ R _ { x } ( \theta ) , R _ { y } ( \theta ) , R _ { x } ( \theta , x _ { k } ) , R _ { y } ( \theta , ^ { ' } x _ { k } ) , R _ { x x } ( \bar { \theta } ) , R _ { y y } ( \theta ) , \bar { R } _ { x x } ( \hat { \theta } , x _ { k } ) , R _ { y y } ( \bar { \theta } , x _ { k } ) , H \}$ We place these gates on a 5-qubit ring topology. This matches Q-FLAIR's own QNN qubit ceiling.

At each iteration, we consider every candidate gate. For each one, we reconstruct its model output as a closed-form sinusoid of its rotation angle. This takes only 2 additional quantum evaluations. We then solve for the loss-minimizing weight classically.

At $d = 7 8 4$ candidate features, a brute-force grid search over feature index is too slow. So we use a bounded scalar minimizer, with convergence tolerance $1 0 ^ { - 3 }$ . This cuts the classical cost by roughly an order of magnitude, solving the same one-dimensional bounded optimization problem that Q-FLAIR specifies. Growth stops the first time the best candidate's training-loss improvement falls below $\Delta L = 1 0 ^ { - 3 }$ . We place no additional cap on circuit size.

Table 1: Per-N summary across 3 seeds, full-resolution (784-pixel) MNIST.
<table><tr><td>N</td><td> $T ^ { * } \left( \mathrm { m e a n } \pm \mathrm { S D } \right)$ </td><td> $\mathrm { t e s t a c c . } ( \mathrm { m e a n } \pm \mathrm { S D } )$ </td><td> $\mathrm { g e n . \ g a p \ ( m e a n \pm S D ) }$ </td></tr><tr><td>2000</td><td> $2 2 . 6 7 \pm 9 . 0 7$ </td><td> $0 . 8 9 2 \pm 0 . 0 2 8$ </td><td> $0 . 0 0 8 9 \pm 0 . 0 0 4 7$ </td></tr><tr><td>4000</td><td> $1 6 . 3 3 \pm 2 . 5 2$ </td><td> $0 . 8 4 1 \pm 0 . 0 0 8$ </td><td> $0 . 0 1 4 1 \pm 0 . 0 1 2 3$ </td></tr><tr><td>6000</td><td> $2 0 . 6 7 \pm 4 . 7 3$ </td><td> $0 . 8 9 2 \pm 0 . 0 2 7$ </td><td> $0 . 0 0 4 5 \pm 0 . 0 0 2 8$ </td></tr><tr><td>8000</td><td> $1 8 . 0 0 \pm 1 . 0 0$ </td><td> $0 . 8 8 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 4 8 \pm 0 . 0 0 2 4$ </td></tr><tr><td>10000</td><td> $2 2 . 6 7 \pm 8 . 9 6$ </td><td> $0 . 8 7 4 \pm 0 . 0 2 2$ </td><td> $0 . 0 1 2 8 \pm 0 . 0 1 0 1$ </td></tr></table>

![](images/6c93debd53cf0dcfde2fd70c846fbe9a58941d7a2231a6507fb44026adb519c5.jpg)  
(a)

![](images/58f08b2f324cc99b0d21ba810686e1cbe261a57414bf5f054b834710a449cf47.jpg)  
(b)

![](images/0ca9cf8a3a984ff61fa14b88cfb10c94b8d31605c6d64142b4f7c317f4fb693a.jpg)  
(c)

![](images/d4b8242c069187f8ea39b593915a220dff380d88873fc5a6aa7f57a423503da3.jpg)  
(d)  
Figure 1: Relationship between training-set size N Vs (a) circuit size $T ^ { * }$ , (b) test accuracy, (c) generalization gap, and (d) Generalization gap versus the full Caro et al. Theorem 3 bound $B _ { K }$ . The shaded region indicates where the bound holds.

Joint fine-tune (Phase 2). Q-FLAIR freezes each gate's weight once found, so Caro et al.'s notion of gates changing during optimization does not apply to Phase 1 alone. To measure it, we warm-start from the finished circuit and run a joint Adam fine-tune over all weights, using mini-batches and early stopping, defining K as the number of gates whose weight moves by more than $\delta = 0 . 0 5$

We record the validation loss at the original Phase-1 weights as a baseline before fine-tuning, and afterward keep whichever weights, original or fine-tuned, give the lower validation loss; Phase 2 can therefore never report a worse model than Phase 1 alone.

We report the full Caro et al. Theorem 3 bound: mi: $\begin{array} { r } { \mathrm { n } _ { K } [ \sqrt { K \log ( M T ) / N } + \sum _ { k > K } M \Delta _ { k } ] } \end{array}$ . When fine-tuning makes no improvement, every $\Delta _ { k }$ is exactly zero and the bound would trivially equal zero, which does not reflect a real generalization guarantee; we flag such cases as not applicable.

## 4 Experimental Setup

We use MNIST digits 3 vs. 5 [5]: raw, unreduced 784-pixel intensities, rescaled to $[ - \pi , \pi ]$ using training-set statistics. We sweep N ∈ {2000, 4000, 6000, 8000, 10000} with 3 seeds per N, yielding 15 runs, each evaluated on the same held-out test set of 1000 examples.

Our loss is the negative log-likelihood, matching both source papers' convention; the decision threshold is set via a classical ROC-style search maximizing balanced accuracy on the training set, as in Q-FLAIR. All circuits are simulated exactly; each run takes minutes to tens of minutes, depending on N and gate count.

We implement circuit construction, statevector simulation, and the Phase-2 Adam fine-tune in PennyLane; Phase 1's classical feature-weight search uses SciPy's bounded scalar minimizer.

## 5 Results

## 5.1 No discernible scaling trend

We might expect more data to support a larger, more accurate circuit. Our results do not support this expectation. Table 1 and Figure 1a-c show neither $T ^ { * }$ nor test accuracy varies monotonically with N.

Circuit size ranges between 16 and 23 gates, with no clear pattern across N. Accuracy ranges between 84% and 89%, also with no clear pattern. The seed-to-seed standard deviation is as large as ±9 gates. This is comparable to the total spread we see across all five values of N.

## 5.2 Consistency with Caro et al.'s bound

One run out of 15 (N=10000) made no improvement during fine-tuning. All its gate weights stayed identical before and after, so K = 0 by construction. For this run, the bound would trivially equal zero. We exclude it from this analysis. That leaves 14 applicable runs.

Across these, the empirical gap never exceeds the full bound (Figure 1d). But the two are only weakly correlated: $r = 0 . 1 2$ . This is not better than using raw T alone $( r = 0 . 0 5 )$ . So the bound holds as valid. But it is not a useful predictor of the actual gap, at this scale and dimensionality.

## 6 Discussion

We attribute the absence of an N-dependent trend to the dimensionality of the candidate feature space: with only ten candidate features, a fixed, absolute training-loss-improvement threshold would interact visibly with N, since a noisy loss estimate at small N would let marginal candidates clear the threshold by chance and inflate circuit size, whereas our actual pool of 784 raw-pixel candidates is large enough that some gate clears the same threshold at almost any N, decoupling the stopping decision from data availability. This suggests that combining an adaptive growth algorithm with a data-efficiency motivation does not, by itself, guarantee a joint scaling relationship; feature-space dimensionality instead acts as a confounding factor that must be addressed explicitly.

A more principled stopping rule might scale its threshold with both N and the candidate pool size, for instance by correcting for the number of comparisons made per iteration, a correction that may be necessary before a genuine joint depth-data scaling law can emerge on high-dimensional inputs.

## 7 Limitations

Our results come from a single dataset and task. We do not know whether these observations generalize to other high-dimensional inputs, or to Q-FLAIR's QSVM variant. We use three seeds per N, and this limits our statistical power. A trend smaller than the observed seed variance could exist, and we would not detect it. Phase 2 is an addition from this work; Q-FLAIR itself does not include a fine-tuning stage. We chose its learning rate and its active-gate threshold δ once, and we did not test other values. Different choices could change the measured value of K, and this could in turn affect our reported correlation with Caro et al.'s bound.

## 8 Conclusion

This work examined whether Q-FLAIR's gate-by-gate growth mechanism, executed at five trainingset sizes on full-resolution MNIST 3-vs-5 classification, yields circuit sizes or generalization behavior consistent with a joint depth-data scaling relationship motivated by Caro et al.'s theory. No such relationship was observed: both circuit size and accuracy vary non-monotonically with N, and although Caro et al.'s bound is never violated in 14 of 15 runs, it correlates only weakly with the observed gap (r = 0.12), an open question in its own right. Joint optimization of circuit depth and training data size cannot, therefore, be assumed to emerge automatically from an adaptive growth algorithm and a generalization bound alone; the dimensionality of the candidate feature pool is a plausible contributing factor warranting further investigation.

## References

[1] M Bilkis, M Cerezo, Guillaume Verdon, Patrick J Coles, and Lukasz Cincio. A semiagnostic ansatz with variable structure for quantum machine learning (2021). arXiv preprint ArXiv:2103.06712.

[2] Matthias C Caro, Hsin-Yuan Huang, Marco Cerezo, Kunal Sharma, Andrew Sornborger, Lukasz Cincio, and Patrick J Coles. Generalization in quantum machine learning from few training data. Nature communications, 13(1):4919, 2022.

[3] Harper R Grimsley, Sophia E Economou, Edwin Barnes, and Nicholas J Mayhall. An adaptive variational algorithm for exact molecular simulations on a quantum computer. Nature communications, 10(1):3007, 2019.

[4] Vojtěch Havlíček, Antonio D Córcoles, Kristan Temme, Aram W Harrow, Abhinav Kandala, Jerry M Chow, and Jay M Gambetta. Supervised learning with quantum-enhanced feature spaces. Nature, 567(7747):209–212, 2019.

[5] Jonas Jäger, Philipp Elsässer, and Elham Torabian. Quantum feature-map learning with reduced resource overhead. Physical Review Research, 8(2):023247, 2026.

[6] Adrián Pérez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and José I Latorre. Data re-uploading for a universal quantum classifier. Quantum, 4:226, 2020.

[7] Maria Schuld and Nathan Killoran. Quantum machine learning in feature hilbert spaces. Physical review letters, 122(4):040504, 2019.