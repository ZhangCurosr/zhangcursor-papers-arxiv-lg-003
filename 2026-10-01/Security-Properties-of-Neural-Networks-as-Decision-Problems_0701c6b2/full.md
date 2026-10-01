# Security Properties of Neural Networks as Decision Problems

Adrian Wurm BTU Cottbus–Senftenberg, Lehrstuhl Theoretische Informatik Platz der Deutschen Einheit 1, 03046 Cottbus, Germany wurm@b-tu.de

25 September 2026

## Abstract

Certifying a deployed neural network raises decision problems that the verification literature has not classified: whether the model carries a backdoor planted in its training data, whether a fault in its stored parameters can drive it into an unsafe state, whether its output leaks a private part of its input. We formalise eight such problems and classify what we can.

The organising observation is a logical one. The function computed by a piecewise linear network, together with all its node values, is definable by a quantifier-free formula of real addition of size linear in the network, so a property of the network is a quantifieralternation sentence, which Sontag’s 1985 theorem places in the polynomial hierarchy at the level of its prefix. Membership results are thus corollaries, and the argument makes plain what they need: that the quantified objects are inputs rather than the network’s own parameters.

Non-interference, monotonicity and counterfactual fairness have exactly the complexity of network equivalence and of interval verification, all co-NP-complete over ReLU. Detection of backdoor triggers from a quantised alphabet is Σ<sup>P</sup><sub>2</sub> -complete, one level above robustness certification, so it does not reduce to polynomially many robustness queries unless the hierarchy collapses. Inversion resistance is co-NP-complete for every $\ell _ { p }$ metric, p a fixed positive integer. Quantifying over parameters instead of inputs - the fault model of bit-flip attacks, radiation upsets and analog accelerators - makes verification ∃R-complete already for networks of identity nodes, for which every previously studied problem is in P, and it stays so when each parameter is confined to a box of inverse-polynomial width; the corresponding safety question is ∀R-complete for ReLU.

## 1 Introduction

Let N be a feedforward neural network whose activations are piecewise linear with rational coeficients. The graph of N - the set of pairs consisting of an input and the induced vector of node values - is then definable by a quantifier-free formula in the language of real addition, of size linear in N: each node contributes a Boolean combination of linear equations and inequalities of constant size, and the network contributes their conjunction. A property of

N of the kind verification asks about is therefore a sentence of real addition whose quantifier prefix is the prefix of the property and whose quantifier-free part names the network. By Sontag’s theorem [34], deciding k-alternation sentences of real addition is complete for $\Sigma _ { k } ^ { \mathrm { P } }$ so such a property lies at the level of the polynomial hierarchy that its prefix dictates, and nowhere else.

This is the organising fact of the present paper, and it has two consequences that pull in opposite directions. On the one hand it makes membership results automatic: the certificate arguments that the verification literature has been reproving for a decade - guess the activation pattern and the violated constraint, then solve a linear program - are the case $k = 1$ of a theorem from 1985, and the higher levels come for free (Lemma 3.2). On the other hand it delimits sharply where the polynomial hierarchy is the right home for such a problem at all. The defining formula is a formula of real addition only because the quantified objects are inputs, which the weights multiply by constants. Quantify instead over the network’s own parameters, and each node contributes a product of two unknowns; the defining formula becomes polynomial, the problem leaves real addition, and its natural home is the existential theory of the reals.

The paper develops both consequences by way of a schema. All the problems we consider, together with all the previously studied ones, are instances of

$$
Q _ { 1 } p \in { \mathcal { P } } \quad Q _ { 2 } x \in A \ : \qquad \Phi \big ( N _ { p } , x \big ) ,
$$

in which x is an input, p is a parameter that perturbs the situation, $Q _ { 1 } , Q _ { 2 }$ are quantifiers, and Φ is a condition expressible by linear constraints on the values of a network at a point. Two independent axes govern the complexity. The first is the quantifier prefix, which places the problem in the polynomial hierarchy. The second, which has not been isolated before, is whether p enters the computation linearly or multiplicatively: input-space parameters enter linearly, parameter-space parameters multiply activations, and that distinction alone decides whether the problem lives in the polynomial hierarchy or in the real hierarchy. Theorem 6.4 shows that the second axis is not a technicality. For networks all of whose nodes compute the identity, every problem previously studied lies in $\mathrm { P } ;$ making the parameters uncertain, within boxes of inverse-polynomial width, makes verification ∃R-complete. Nothing about the activation function is involved, which is exactly why the phenomenon is invisible in a classification indexed by activation functions.

What the schema needs in order to be more than a bookkeeping device is a supply of instances that actually occupy its cells, and the previously studied problems do not: reachability, verification of an interval property, network equivalence and the robustness notions all have a fixed network and a single quantifier, so they sit in one corner of the grid, with network minimisation the lone exception at the second level. Problems with genuinely alternating prefixes, and problems that quantify over the parameters, are supplied instead by the questions that certification of a deployed model turns on. Whether a network carries a backdoor planted by whoever supplied its training data is an ∃∀ question over a trigger. Whether a fault injected into the memory holding the parameters can drive the network into an unsafe state is an ∃∃ question over the parameters. Whether a model reveals the part of its input that was to stay private, whether a stolen copy can be stripped of its watermark, and whether a network that failed certification can be patched without invalidating the rest of the safety case, are further instances, and they fall in four diferent cells. This paper proposes formalisations of eight such properties, classifies what it can, and records in Table 1 where each one lands. Three of them turn out to be old problems in new clothes, which is worth knowing precisely because it tells the reader which of the eight are new.

## 1.1 Contributions

(C1) A uniform membership lemma (Lemma 3.2): for networks whose activations are semilinear with rational coeficients, the graph of the network is definable by a quantifierfree formula of real addition of size linear in the network. Every membership result for such networks in the polynomial hierarchy is therefore a corollary of Sontag’s theorem [34] that k-alternation sentences of real addition are complete for $\Sigma _ { k } ^ { \mathrm { P } }$ . The certificate arguments in the verification literature are its k = 1 case.

(C2) Three security properties turn out to have exactly the complexity of problems that are already classified (Section 4): non-interference that of network equivalence (Proposition 4.2), and monotonicity and counterfactual fairness that of verification of an interval property (Proposition 4.6); all three are co-NP-complete over ReLU networks, and the hardness proofs are given for ReLU alone, with no threshold activation assumed. The equivalence is one of complexity only. The three are diferent specifications, stated over diferent data and naturally implemented by diferent front-ends; what the reductions give is that the classification by activation function transfers to them verbatim, and that an existing equivalence or interval checker decides them without being modified. For monotonicity over ReLU networks the co-NP-completeness is already known from the study of extracted rules [44], with which Remark 4.8 compares.

(C3) Backdoor-trigger detection with a quantised trigger alphabet is Σ<sup>P</sup><sub>2</sub> -complete for ReLU networks (Theorem 5.2). We give the reduction in full, and show in Section 5.2 that the quantisation is not a technical convenience: the continuous relaxation admits a strategy for the adversary - setting a literal and its negation both to zero - that has no Boolean counterpart, so the relaxed problem is a diferent one.

(C4) Inversion resistance is co-NP-complete (Theorem 7.2).

(C5) If the parameters carry the uncertainty, verification is ∃R-complete already for networks of identity nodes (Theorem 6.4), for which every previously studied verification problem lies in P, and it remains so when every parameter is confined to a box of inverse-polynomial width. The corresponding safety question is ∀R-complete for ReLU. The proof is a direct reduction from the range-restricted inversion problem Range-Etr-Inv [4].

(C6) Definitions and initial classifications for universal vulnerability, output indistinguishability, watermark removability and repair, the last of which lands naturally in the second level of the real hierarchy (Section 8).

## 1.2 Related work

On the logical side the paper rests on two classical results and their surrounding literature. Sontag’s theorem on real addition [34] is what places the first-order properties of a piecewise linear network in the polynomial hierarchy, and Lemma 3.2 is little more than the observation that the hypotheses of that theorem are met uniformly. For ∃R as a complexity class, and for the catalogue of problems complete for it, we follow the compendium of Schaefer, Cardinal and Miltzow [28]; for the hierarchy above ∃R we follow Schaefer and Štefankovič [29]. The complete problem we reduce from in Section 6 is the range-restricted inversion problem of Abrahamsen, Miltzow and Seiferth [4], in the streamlined form of [3], whose range promise is what lets us shrink the parameter boxes.

On the verification side, the complexity of reachability for ReLU networks is due to Katz et al. [17] and Sälzer and Lange [27]; the classification across activation functions, and the problems VIP, NE, Min and the robustness family, are from [41, 43, 42], whose notation we follow throughout. Training ReLU networks is ∃R-complete [2]; Theorem 6.4 isolates the reason, which is not the optimisation but the fact that parameters and activations multiply. The complexity of checking rules extracted from a network is studied in [44]. The algorithmic side has matured in parallel: complete verifiers built on SMT solving [17, 18], on mixed integer programming [36] and on branch and bound over bound propagation [40] are compared annually in the international verification competition [8]. The lower bounds below say what those tools cannot be extended to do without leaving the arithmetic they are built on.

Each property we consider has a literature of its own, which is algorithmic or empirical rather than complexity-theoretic. Non-interference originates with Goguen and Meseguer [12] and is surveyed for programming languages by Sabelfeld and Myers [26]; its quantitative form, information leakage, has for neural networks been approached by approximate model counting [7]. Monotonicity is enforced by construction or certified post hoc in [21, 32]; counterfactual fairness is due to Kusner et al. [19], individual fairness to Dwork et al. [9], and its verification is treated in [16]. Backdoor attacks were introduced in [15] and are scanned for by methods such as Neural Cleanse [39]; universal adversarial perturbations are due to Moosavi-Dezfooli et al. [23]. Weight faults are studied as an attack [25], as a reliability problem [20], as a consequence of analog hardware [30], and empirically as a robustness notion [37]. Model inversion [11] and membership inference [31] motivate Section 7, backdoor-based watermarking [5] motivates Problem 7.7, and network repair [35, 13] motivates Section 8. To the best of our knowledge none of these problems has previously been placed in a complexity class, with the qualifications recorded below for backdoors and for parameter perturbation.

Three lines of work stand close enough to ours that the relationship needs stating. First, backdoors have been treated formally from two sides. Pham and Sun [24] give a verification procedure for what is essentially our Problem 5.1, combining statistical sampling with abstract interpretation to certify that no input-agnostic trigger of a given shape achieves a given success rate; Theorem 5.2 supplies the complexity-theoretic floor under that procedure and, through the argument in Section 5.2, explains why its outer search over candidate triggers cannot be replaced by a bounded number of inner verification calls. Goldwasser, Kim, Vaikuntanathan and Zamir [14] prove a sharper-looking statement of an entirely different kind: a backdoor can be planted so that the resulting network is computationally indistinguishable from an honestly trained one, so that no eficient detector succeeds at all under standard cryptographic assumptions. Neither result implies the other. Theirs is an average-case, cryptographic statement about a particular planted construction, and it rules out eficient detection outright; ours is an unconditional worst-case statement about the decision problem itself, and it locates that problem at a specific level of the polynomial hierarchy, which is what licenses the separation from robustness certification drawn in Section 5.2. A Σ<sup>P</sup>-complete problem may still have easy instances, and a cryptographically undetectable backdoor says nothing about where the worst case sits.

Second, the arithmetic in which a network is evaluated is a third axis, orthogonal to the two studied here. Alsmann, Lange and Sälzer [6] classify verification when the network is evaluated in finite-width fixed- or floating-point arithmetic and obtain PSPACEcompleteness for bit-vector specifications. We touch that axis twice, in Section 5.2 and in Section 8, at the points where quantising a continuous search space changes a problem’s class rather than approximating it.

Third, and closest to Section 6, Soltanalian [33] studies exact ReLU verification in a smoothed model in which every weight and bias is independently perturbed by clipped, rounded Gaussian noise, and shows that no sound and complete verifier runs in expected polynomial time under the perturbation unless $\mathrm { N P \subseteq B P P }$ . The question there is diferent from ours in a way worth making explicit: the noise is a preprocessing of the instance, applied once and at random, after which the problem asked is still the input-space question $\forall x \in X$ , and the conclusion is that worst-case hardness survives randomisation of the parameters. Here the parameters are instead quantified inside the decision problem, over an adversarially chosen uncertainty set, and the conclusion is that the problem leaves the polynomial hierarchy altogether. The two results point the same way from opposite directions - randomising the parameters does not make verification easy, and quantifying over them makes it strictly harder - and neither subsumes the other.

## 2 Preliminaries

We follow [42]. A feedforward neural network N is a layered graph computing a function $\mathbb { R } ^ { n } \to \mathbb { R } ^ { m }$ ; node i of layer ℓ computes $\begin{array} { r } { y _ { \ell i } = \sigma _ { \ell i } ( \sum _ { j } c _ { j i } ^ { ( \ell - 1 ) } y _ { ( \ell - 1 ) j } + b _ { \ell i } ) } \end{array}$ with rational weights c and biases b and activation functions $\sigma _ { \ell i }$ drawn from a set $F ;$ such a network is an F-network. An LP specification is a system Ax $\leq b$ of linear inequalities with rational data. We write NNReach(F), VIP(F), NE(F) and Min(F) for reachability, verification of an interval property, network equivalence and network minimisation, as in [42]. The classification of an input is arg ma $\mathrm { x } _ { j } N ( { \boldsymbol { x } } ) _ { j }$ , with ties broken in favour of the smallest index, so that the classification is a total function.

Every node of an F-network applies an activation, the output nodes included. For $F = \{ { \mathrm { R e L U } } \}$ this means that all node values, and in particular all outputs, are nonnegative; linear combinations with negative coeficients are formed in the linear part of the next node, which is where we use them below.

A set is semilinear if it is a Boolean combination of half-spaces; an activation is semilinear if its graph is. ReLU, Heaviside, sign, the identity, leaky ReLU and every rational step function are semilinear with rational coeficients. We write ∃R for the complexity class of problems reducing in polynomial time to the existential theory of the reals Etr, and ∃∀R for the analogous class one alternation up [29].

We use the following results on the two classes involved.

Theorem 2.1 (Sontag [34]). Fix $k \geq 1$ and consider sentences

$$
Q _ { 1 } { \bar { z } } _ { 1 } \ Q _ { 2 } { \bar { z } } _ { 2 } \ \cdot \ \cdot \ \cdot \ Q _ { k } { \bar { z } } _ { k } \ F ( { \bar { z } } )
$$

in which the quantifiers $Q _ { i }$ alternate and F is quantifier-free in the language of real addition with binary-encoded rational coeficients. Deciding such a sentence is log-complete for $\Sigma _ { k } ^ { \mathrm { P } }$ when $Q _ { 1 } = \exists$ , and log-complete for $\Pi _ { k } ^ { \mathrm { P } }$ when $Q _ { 1 } = \forall$

Theorem 2.2 (Abrahamsen, Adamaszek, Miltzow [1]). Etr-Inv is ∃R-complete: given variables $\xi _ { 1 } , \ldots , \xi _ { k }$ confined to $[ \frac { 1 } { 2 } , 2 ]$ and a list of constraints, each of the form $\xi _ { a } + \xi _ { b } = \xi _ { c }$ or $\xi _ { a } \cdot \xi _ { b } = 1$ , decide satisfiability.

Section 6 needs the range-restricted refinement of Etr-Inv, in which each variable is confined not merely to $[ \frac { 1 } { 2 } , 2 ]$ but to a prescribed subinterval of it that may be very short. We follow [4], with the streamlined exposition in [3]; for $\exists \mathbb { R }$ and its complete problems generally we follow the compendium [28].

Definition 2.3 (Range-Etr-Inv). An instance consists of variables $\xi _ { 1 } , \ldots , \xi _ { k } .$ for each i a rational interval $I ( \xi _ { i } ) \subseteq [ \frac { 1 } { 2 } , 2 ]$ , and a list of constraints of the two forms above. The question is whether there is an assignment with $\xi _ { i } \in I ( \xi _ { i } )$ for every i satisfying every constraint. The instance has range parameter δ i $: | I ( \xi _ { i } ) | \leq 2 \delta$ for every $i .$

Theorem 2.4 (Abrahamsen, Miltzow, Seiferth [4, Thm. 3]). Range-Etr-Inv is ∃Rcomplete, and remains so when restricted to instances of range parameter $\delta \ : = \ : O ( k ^ { - c } )$ for any constant $c > 0$ fixed in advance. Theorem 2.2 is the special case in which every $I ( \xi _ { i } ) = [ \textstyle { \frac { 1 } { 2 } } , 2 ]$

## 2.1 A network we shall build repeatedly

Several hardness proofs below run through one ReLU-network encoding of a 3-CNF formula, a variant of the network of [42, Thm. 9(ii)]. The statement records only what the later arguments use, namely how the network behaves on Boolean inputs and the sense in which no real input does better than the Boolean input it rounds to; the construction itself is deferred to the proof.

Write $\rho : \mathbb { R } ^ { n } \to \{ 0 , 1 \} ^ { n }$ for coordinatewise rounding,

$$
\begin{array} { r } { \rho ( z ) _ { a } : = 1 \mathrm { ~ i f ~ } z _ { a } > \frac { 1 } { 2 } , \qquad \rho ( z ) _ { a } : = 0 \mathrm { ~ i f ~ } z _ { a } \le \frac { 1 } { 2 } , } \end{array}
$$

and, for $b \in \{ 0 , 1 \} ^ { n }$ , write $\operatorname { s a t } _ { \varphi } ( b )$ for the number of clauses of $\varphi$ satisfied by the assignment b.

Lemma 2.5 (CNF network). Let $\varphi$ be a 3-CNF formula with n variables and $m \geq 1$ clauses. There is a ReLU-network $M _ { \varphi } : \mathbb { R } ^ { n } $ R with n input nodes, two hidden layers and $4 n + 2 m + 1$ further nodes, computable from $\varphi$ in linear time, such that

(i) $M _ { \varphi } ( b ) = \mathrm { s a t } _ { \varphi } ( b )$ for every $b \in \{ 0 , 1 \} ^ { n }$ ;

(ii) $M _ { \varphi } ( z ) \le \mathrm { s a t } _ { \varphi } ( \rho ( z ) )$ for every $z \in \mathbb { R } ^ { n }$ , so that $0 \leq M _ { \varphi } \leq$ m everywhere;

(iii) $M _ { \varphi } ( \textstyle { \frac { 1 } { 2 } } , \dots , \frac { 1 } { 2 } ) = 0$

Items (i) and (ii) are what the identification of $\mathbb { R } ^ { n }$ with $\{ 0 , 1 \} ^ { n }$ buys. On the Boolean points the network counts satisfied clauses exactly, and every other point is dominated by the Boolean point it rounds to, so nothing is gained by leaving the cube: maximising $M _ { \varphi }$ over $\mathbb { R } ^ { n }$ and over $\{ 0 , 1 \} ^ { n }$ give the same value. Item (iii) is of a diferent kind, and is the one place where a real input does strictly worse than its rounding would suggest; Section 5.2 turns on it.

Proof. For each propositional variable we allocate one input node, for each literal a pair of nodes in the first hidden layer, for each clause a pair of nodes in the second hidden layer, and for the evaluation a single node in the output layer, which follows directly after the second hidden layer. Linear combinations with negative coeficients are formed in the linear part of the following node, as everywhere in this paper.

Literals. For a variable a with input $z _ { a }$ put

$$
\begin{array} { r } { \alpha _ { a } : = 2 \mathrm { R e L U } \left( z _ { a } - \frac { 1 } { 2 } \right) - 2 \mathrm { R e L U } \left( z _ { a } - 1 \right) , \qquad \alpha _ { - a } : = 2 \mathrm { R e L U } \left( \frac { 1 } { 2 } - z _ { a } \right) - 2 \mathrm { R e L U } \left( - z _ { a } \right) , } \end{array}
$$

two ReLU nodes each. So $\alpha _ { a }$ is $2 ( z _ { a } - \textstyle { \frac { 1 } { 2 } } )$ clamped to [0, 1] and $\alpha _ { \neg a }$ is $2 ( \textstyle { \frac { 1 } { 2 } } - z _ { a } )$ clamped to [0, 1]. Three consequences are used below: $\alpha _ { \lambda } \in [ 0 , 1 ]$ for every literal $\lambda ; \alpha _ { a } ( z ) > 0$ exactly when $\begin{array} { r } { z _ { a } > \frac { 1 } { 2 } } \end{array}$ and $\alpha _ { \neg a } ( z ) > 0$ exactly when $\begin{array} { r } { z _ { a } < \frac { 1 } { 2 } } \end{array}$ , so at most one of the two is non-zero and both vanish precisely at $\begin{array} { r } { z _ { a } = \frac { 1 } { 2 } . } \end{array}$ ; and for $b \in \{ 0 , 1 \} ^ { n }$ one has $\alpha _ { \lambda } ( b ) = 1 { \mathrm { i f } } \lambda$ is true under b and $\alpha _ { \lambda } ( b ) = 0$ otherwise.

Clauses. For a clause $C \ = \ ( \lambda _ { 1 } \vee \lambda _ { 2 } \vee \lambda _ { 3 } )$ put $s _ { C } : = \alpha _ { \lambda _ { 1 } } + \alpha _ { \lambda _ { 2 } } + \alpha _ { \lambda _ { 3 } }$ and $\gamma _ { C } : =$ ReL $\mathrm { \Delta } _ { l } \mathrm { U } \mathrm { ( } s _ { C } \mathrm { ) } - \mathrm { R e L U } \mathrm { ( } s _ { C } - 1 \mathrm { ) }$ , again two ReLU nodes, so that $\gamma _ { C }$ is $s _ { C }$ clamped to $[ 0 , 1 ]$

Output. $\begin{array} { r } { M _ { \varphi } : = \sum _ { C } \gamma _ { C } } \end{array}$ , one node, whose argument is non-negative so that the ReLU on it acts as the identity. The node count is 4n $+ 2 m + 1$

(i) For $b \in \{ 0 , 1 \} ^ { n }$ each $\alpha _ { \lambda } ( b )$ is 1 or 0 according as $\lambda$ is true or false under b. Hence $s _ { C } \geq 1$ exactly when $C$ is satisfied by $b ,$ so $\gamma _ { C } = 1$ for the satisfied clauses and $\gamma _ { C } = 0$ for the others, and the output is sat $\mathsf { \Pi } _ { \varphi } ( b )$

(ii) Fix z and let $C$ be a clause false under $\rho ( z )$ . Every literal λ of $C$ is false under $\rho ( z )$ : if $\lambda = a$ this means $\begin{array} { l } { { z _ { a } \leq \frac { 1 } { 2 } } } \end{array}$ , hence $\alpha _ { a } ( z ) = 0 ;$ ; if $\lambda = \neg a$ it means $\begin{array} { r } { z _ { a } > \frac { 1 } { 2 } } \end{array}$ , hence $\alpha _ { \neg a } ( z ) = 0$ . So $s _ { C } = 0$ and $\gamma _ { C } = 0$ . Every remaining clause contributes at most 1, which gives $M _ { \varphi } ( z ) \le \mathrm { s a t } _ { \varphi } ( \rho ( z ) )$ , and the latter is at most $m$

(iii) Taking $\begin{array} { r } { z _ { a } = \frac { 1 } { 2 } } \end{array}$ for every a makes every $\alpha _ { \lambda }$ vanish, hence every $s _ { C } = 0$ and every $\gamma _ { C } = 0$ □

The form in which Section 5 uses the lemma is the following.

Corollary 2.6. Let S be a set of variables of φ and $\sigma \in \{ 0 , 1 \} ^ { S }$ . The supremum of $M _ { \varphi }$ over the inputs whose S-coordinates are fixed to σ equals m if the restriction $\varphi | _ { \sigma }$ is satisfiable, and is at most $m - 1$ otherwise. In the first case it is attained at a Boolean input.

Proof. Rounding fixes Boolean coordinates, so if $z$ has its $S \mathrm { - }$ -coordinates equal to $\sigma$ then so does $\rho ( z )$ . If $\varphi | _ { \sigma }$ is satisfiable, extend $\sigma$ to a satisfying assignment b and apply Lemma $2 . 5 ( \mathrm { i } )$ to get $M _ { \varphi } ( b ) = m$ . If it is not, then every Boolean point agreeing with σ leaves some clause unsatisfied, so $\mathrm { { s a t } } _ { \varphi } ( \rho ( z ) ) \leq m - 1$ for every admissible $z ,$ and Lemma 2.5(ii) bounds $M _ { \varphi }$ by $m - 1$ there. □

The gadget the hardness proofs of Section 4 actually use is the following non-negative indicator, which vanishes identically exactly when $\varphi$ is unsatisfiable.

Definition 2.7 (Violation gadget). For a 3-CNF formula $\varphi$ with m clauses put $\hat { g } _ { \varphi } : =$ $\mathrm { R e L U } \big ( M _ { \varphi } - ( m - 1 ) \big )$ .

Corollary 2.8. $\hat { g } _ { \varphi }$ is a ReLU-network with values in [0, 1], computable from $\varphi$ in linear time, with $\hat { g } _ { \varphi } \equiv 0 \ i f \varphi$ is unsatisfiable, and attaining both the value 1 and the value $0 \ i f \varphi$ is satisfiable.

Proof. Values in [0, 1] because $0 \leq M _ { \varphi } \leq m$ by Lemma 2.5(ii). If $\varphi$ is unsatisfiable then sa $\mathfrak { t } _ { \varphi } ( \rho ( z ) ) \le m - 1$ for every $z ,$ so $M _ { \varphi } \le m - 1$ everywhere by (ii) and the ReLU clips to 0. If $\varphi$ is satisfiable, (i) applied to a satisfying assignment gives an input with $M _ { \varphi } = m$ and hence $\hat { g } _ { \varphi } = 1$ , and (iii) gives an input with $M _ { \varphi } = 0$ and hence $\hat { g } _ { \varphi } = 0$ □

## 3 A schema, and two axes

Definition 3.1 (Schema). A parameterised verification problem is given by a class of perturbation parameters $\mathcal { P }$ , a quantifier prefix $Q _ { 1 } Q _ { 2 } \in \{ \exists , \forall \} ^ { 2 }$ , and a condition Φ expressible by linear constraints on the values of a network at a point. An instance consists of an F-network $N ,$ , a description of ${ \mathcal { P } } _ { : }$ an LP specification A for the inputs, and the data of Φ; the question is whether

$$
\begin{array} { r l r } { Q _ { 1 } p \in \mathcal { P } } & { { } Q _ { 2 } x \in A : } & { \quad \Phi \big ( N _ { p } , x \big ) . } \end{array}
$$

Three kinds of P occur. It may be trivial, the network being fixed; it may be an inputspace set, so that $N _ { p } ( x ) = N ( x \oplus p )$ for some rational afine combination rule ⊕; or it may be a parameter-space set, so that $N _ { p }$ is N with diferent parameters. Table 1 on page 22 places every problem of this paper in the resulting grid.

The two axes are these.

Alternation. The quantifier prefix determines the level of the polynomial hierarchy, and Lemma 3.2 below makes the membership direction automatic for piecewise linear networks. All previously studied problems have P trivial and a single quantifier, hence live at the first level; Min is the exception, and it is at the second.

Arithmetic. Whether p enters the computation linearly or multiplicatively determines whether the resulting constraint system is linear or polynomial, and hence whether the problem lives in the polynomial hierarchy or in the real hierarchy. An input-space parameter is added to an input and enters exactly as an input does. A parameter-space parameter multiplies an activation. Theorem 6.4 shows that this distinction alone accounts for a jump from P to ∃R-complete.

Lemma 3.2 (Uniform membership). Let F be a finite set of semilinear activations with rational coeficients. Then the graph $\{ ( x , \bar { y } ) : \bar { y }$ is the vector of node values of N on x} of an F-network N is defined by a quantifier-free formula of real addition of size linear in the size of N. Consequently, if a property of N is expressible in the schema of Definition 3.1 with k quantifier blocks and a trivial or input-space parameter class, then deciding it lies in $\Sigma _ { k } ^ { \mathrm { P } }$ or $\Pi _ { k } ^ { \mathrm { P } }$ according to the leading quantifier.

Proof. Each activation $\sigma \in F$ has semilinear graph with rational coeficients, so $y = \sigma ( z )$ is equivalent to a Boolean combination $\bigvee _ { r } \big ( y = a _ { r } z + b _ { r } \wedge \alpha _ { r } \leq z \leq \beta _ { r } \big )$ of constant size, the number of pieces depending only on σ and not on N. The value z entering a node is a rational linear form in the values of the previous layer, so each node contributes a constant-size quantifier-free formula, and N contributes their conjunction, of size linear in $N _ { ☉ }$

For the second statement, write the property out. Writing G for the graph formula just constructed, a block $\exists x \in A$ becomes

$$
\exists x \exists y \ \bigl ( G \land A ( x ) \land \cdots \bigr ) ,
$$

and a block $\forall x \in A$ becomes

$$
\forall x \forall \bar { y } \ \big ( G \land A ( x )  \cdot \cdot \cdot \big ) ;
$$

in either case the node variables join the block they belong to without creating an alternation, because they are functionally determined by that block’s variables. An input-space parameter p is a further variable of its own block, and $N _ { p } ( x ) = N ( x \oplus p )$ is again linear. The result is a k-alternation sentence of real addition of size linear in the instance, and Theorem 2.1 applies. □

Remark 3.3. Lemma 3.2 subsumes the certificate arguments used throughout the verification literature - “guess the activation pattern and the violated constraint, then solve a linear program” - which are its $k = 1$ case. It is worth stating because it makes all membership results in this paper immediate, and because it identifies exactly what is needed for them: semilinearity of the activations, and a parameter that enters linearly. Both hypotheses fail in Section 6, and the complexity changes accordingly.

## 4 Three properties that reduce to classified ones

We begin with three properties that a practitioner would reach for first. Each of them turns out to be linear-time interreducible with a problem whose complexity is already known, so that none of them is new as a computational problem, however diferent they are as specifications. We record this because the consequence is useful in both directions: their complexity is classified by activation function already, and a tool that decides equivalence or interval verification decides them once a front-end has been written.

## 4.1 Non-interference

The oldest formal security property there is [12], transplanted from language-based security [26]: the output must not depend on the secret part of the input.

Problem 4.1 (NonInt(F)). Given an F-network N on n inputs and a set $P \subseteq \{ 1 , \ldots , n \}$ of public coordinates, the remaining coordinates being secret, decide whether

$$
\forall x , x ^ { \prime } \in \mathbb { R } ^ { n } : \quad x _ { P } = x _ { P } ^ { \prime } \implies N ( x ) = N ( x ^ { \prime } ) .
$$

Proposition 4.2. NonInt(F) reduces to $N E ( F )$ in linear time for every $F _ { \mathrm { { ; } } }$ , so every upper bound for NE(F) transfers to NonInt(F). Conversely NonInt(ReLU) is $c o { - } N P .$ hard already for $P = \emptyset$ . Hence NonInt(ReLU) is co-NP-complete.

Proof. Let S be the secret coordinates. On the enlarged input $( x _ { P } , x _ { S } , x _ { S } ^ { \prime } ) \in \mathbb { R } ^ { | P | } \times \mathbb { R } ^ { | S | } \times$ R<sup>|S|</sup> define $N _ { 1 } ( x _ { P } , x _ { S } , x _ { S } ^ { \prime } ) : = N ( x _ { P } , x _ { S } )$ and $N _ { 2 } ( x _ { P } , x _ { S } , x _ { S } ^ { \prime } ) : = N ( x _ { P } , x _ { S } ^ { \prime } )$ . Each is a copy of N with some input edges rerouted and some inputs ignored, hence an F-network of the same size; and NonInt(N, P) holds if and only if $N _ { 1 } \equiv N _ { 2 }$ . The construction is computable in linear time.

For hardness take $P = \varnothing .$ , so that the premise $x _ { P } = x _ { P } ^ { \prime }$ is vacuous and non-interference says exactly that N is a constant function. Given a 3-CNF formula $\varphi ,$ , take $N : = \hat { g } _ { \varphi }$ of Definition 2.7. By Corollary 2.8 this network is constant if $\varphi$ is unsatisfiable, and takes both the value 0 and the value 1 if $\varphi$ is satisfiable. So NonInt(N, ∅) holds if and only if $\varphi$ is unsatisfiable, and the construction is linear time. Membership in co-NP for piecewise linear F is Lemma 3.2 with $k = 1$ □

Corollary 4.3 ([42]). NE(ReLU) is co-NP-complete.

Proof. Membership is Lemma 3.2 with $k = 1$ . For hardness compare $\hat { g } _ { \varphi }$ with the network computing the constant 0: by Corollary 2.8 the two are equivalent if and only if $\varphi$ is unsatisfiable. □

The statement is not ours; network equivalence was classified in [42], and Table 1 records it there. We give the argument because it costs two lines once Corollary 2.8 is available, and because keeping it inside the paper is what lets Proposition 4.2 be read without consulting [42].

Remark 4.4. The hardness argument uses only an F-network that is constant precisely when the source instance is negative. For any activation set F admitting such a gadget the same two lines transfer the lower bounds for NNReach(F) to the complement of NonInt(F); in particular the complement of NonInt(F) is then ∃R-hard for every F for which reachability is ∃R-hard [41, 43], so that NonInt(F) is ∀R-hard. We state the ReLU case, which is the one the rest of the paper uses.

Practical reading. Two things follow. First, no new solver is required: a verifier that decides equivalence of two ReLU networks decides non-interference once a front-end has duplicated the network and tied the public inputs. That front-end is real work, and the two specifications look nothing alike on the page; the reduction says that the hard part is shared, not that the problems are the same. Second, and less comfortably, exact non-interference is the wrong specification for a trained model. It is an all-or-nothing condition, and a network trained on data in which the secret correlates with anything at all will fail it, typically by a margin far too small to matter. The specification practitioners actually want is metric - the output should not depend appreciably on the secret - and that is not $\operatorname { N E } ;$ it is Problem 7.5 below, which sits a level higher. The gap between the property that is easy to check and the property one wants is, here, exactly one quantifier alternation. The alternative is to leave the decision setting and measure leakage instead, as approximate counting has been made to do for neural networks [7]; that moves the problem into a counting class rather than up the hierarchy.

## 4.2 Monotonicity and counterfactual fairness

Problem 4.5 (Mono(F), CFair(F)). Mono(F): given N, an LP specification A and a coordinate i, decide whether $N ( x ) \leq N ( x ^ { \prime } )$ componentwise for all $x , x ^ { \prime } \in A$ with $x ^ { \prime } - x \in$ $\mathbb { R } _ { \geq 0 } e _ { i }$ $\operatorname { C F A I R } ( F )$ , after [19]: given N, A and a rational afine involution ι acting on a protected coordinate, decide whether arg max<sub>j</sub> $N ( x ) _ { j } = \arg \operatorname* { m a x } _ { j } N ( \iota x ) _ { j }$ for all $x \in A$ with $\iota x \in A$

Proposition 4.6. For semilinear F with rational coeficients, Mono(F) and CFair(F) are in co-NP. Both are co-NP-hard for $F = \{ { \mathrm { R e L U } } \}$ , hence co-NP-complete.

Proof. Membership is Lemma 3.2 with $k = 1$ : both conditions are single universal blocks over two copies of the network whose inputs are tied by linear equations. (Note that Lemma 3.2 is insensitive to whether the conclusion uses strict or non-strict inequalities, since the language of real addition contains both. This matters because monotonicity is inherently non-strict, whereas VIP is stated in [42] with an open output polyhedron.)

Hardness of Mono(ReLU). Let $\varphi$ be a 3-CNF formula and $\hat { g } : = \hat { g } _ { \varphi }$ its violation gadget, with values in [0, 1]. Build the ReLU-network $M ^ { \prime }$ on inputs $( z , t ) \in \mathbb { R } ^ { n } \times \mathbb { R }$ with the single output

$$
\begin{array} { r } { M ^ { \prime } ( z , t ) : = \mathrm { R e L U } \big ( \hat { g } ( z ) - t \big ) , } \end{array}
$$

which is a legal ReLU-network because $\hat { g } ( z )$ is the value of a node and t an input, so the displayed node applies ReLU to a linear form in earlier values. Take $A ^ { \prime } : = \{ 0 \leq t \leq 1 \}$ and the coordinate t.

Fix $z$ and write $c : = \hat { g } ( z ) \in [ 0 , 1 ]$ . If $c = 0$ then $M ^ { \prime } ( z , t ) = \mathrm { R e L U } ( - t ) = 0$ for every $t \in$ $[ 0 , 1 ]$ , so $M ^ { \prime } ( z , \cdot )$ is constant and in particular non-decreasing. If $c > 0$ then $M ^ { \prime } ( z , 0 ) = c > 0$ while $M ^ { \prime } ( z , c ) = { \mathrm { R e L U } } ( 0 ) = 0$ , and $( z , c ) - ( z , 0 ) \in \mathbb { R } _ { \geq 0 } e _ { t }$ with $( z , c ) \in A ^ { \prime }$ , so monotonicity in t fails. Hence $M ^ { \prime }$ is monotone in t on $A ^ { \prime }$ if and only if $\hat { g } \equiv 0$ , which by Corollary 2.8 holds if and only if $\varphi$ is unsatisfiable.

Hardness of CFair(ReLU). Take the same ${ \hat { g } } ,$ the input specification $A ^ { \prime } : = \{ 0 \leq t \leq 1 \}$ }, which is invariant under the rational afine involution $\iota : ( z , t ) \mapsto ( z , 1 - t )$ , and the ReLUnetwork $M ^ { \prime \prime }$ with the two outputs

$$
\begin{array} { r } { o _ { 1 } ( z , t ) : = \mathrm { R e L U } ( 1 ) = 1 , \qquad o _ { 2 } ( z , t ) : = \mathrm { R e L U } \big ( 2 - 2 u ( z , t ) \big ) , \quad u ( z , t ) : = \mathrm { R e L U } \big ( \hat { g } ( z ) + t - 1 \big ) , } \end{array}
$$

where $o _ { 1 }$ is a node with no incoming edges and bias 1. For $\hat { g } ( z ) , t \in [ 0 , 1 ]$ we have $u \in [ 0 , 1 ]$ and hence $o _ { 2 } = 2 - 2 u \in [ 0 , 2 ]$

If $\varphi$ is unsatisfiable then $\hat { g } \equiv 0 ,$ , so $u ( z , t ) = \mathrm { R e L U } ( t - 1 ) = 0$ on $A ^ { \prime }$ and $o _ { 2 } \equiv 2 > 1 = o _ { 1 }$ the classification is 2 at every point of $A ^ { \prime }$ and counterfactual fairness holds. If $\varphi$ is satisfiable, take $z ^ { * }$ with $\hat { g } ( z ^ { * } ) = 1$ , which exists by Corollary $2 . 8 . { \mathrm { ~ A t ~ } } ( z ^ { * } , 1 )$ we get $u = \mathrm { R e L U } ( 1 ) = 1$ and $o _ { 2 } = 0 < 1 = o _ { 1 }$ , so the classification is 1; at $\iota ( z ^ { * } , 1 ) = ( z ^ { * } , 0 )$ we get $u = \mathrm { R e L U } ( 0 ) = 0$ and $o _ { 2 } = 2 > 1 = o _ { 1 }$ , so the classification is 2, and fairness fails. Both constructions are computable in linear time. □

Remark 4.7. Neither reduction uses anything about ReLU beyond the gadget of Corollary 2.8 - a non-negative F-network, bounded by 1, vanishing identically exactly when the source instance is negative - and the ability to apply an activation to a linear form in earlier node values. For any F admitting such a gadget the same constructions reduce the corresponding reachability problem to Mono(F) and CFair(F); in particular both are ∀Rhard for every F for which reachability is ∃R-hard [41, 43], which includes every non-linear polynomial and the usual sigmoidal activations.

Note also that the earlier version of this argument passed through a Heaviside threshold and produced a network with a negative output. That is not available over $F = \{ { \mathrm { R e L U } } \}$ , since every node of a ReLU-network applies ReLU and all outputs are therefore non-negative; the constructions above make the argument of the final ReLU decrease instead of the value, which is what allows the Heaviside hypothesis to be dropped.

Remark 4.8 (Relation to rule checking). Monotonicity has been studied in this setting before, and for ReLU networks the co-NP-completeness in Proposition 4.6 is not new. Rules extracted from a network are classified in [44] for four rule languages - propositional rules, oblique rules, M-of-N rules and monotonicity rules - under three questions: whether a rule holds of the network, whether a set of rules is consistent, and whether a set of rules is exhaustive. Rule verification is co-NP-complete for ReLU networks in each of the four languages, monotonicity included, and checking a monotonicity rule is shown there to be linear-time reducible to checking an oblique rule. The monotonicity rules of [44] are more general than Problem 4.5: they require that $A x \le A y$ imply that the output does not decrease, for a matrix A, so the order need not be coordinatewise, and Problem 4.5 is the case in which A selects a single coordinate.

Proposition 4.6 adds two things. The hardness reduction is stated for ReLU alone and constructs the network explicitly from the formula, rather than assuming a threshold activation in order to build a Boolean violation indicator; and by Remark 4.7 the same two gadgets carry the classification of ReLU, giving ∀R-hardness wherever reachability is ∃R-hard. The framings are complementary in a further way. In [44] the rule language is the parameter and the interesting phenomena are diferences between languages - consistency and exhaustiveness of monotonicity rules are in P there, while for the other three languages they are co-NP-hard - whereas here monotonicity is one property among several and the parameter of interest is the quantifier prefix. That the ReLU cases agree is a consistency check on both readings.

Remark 4.9. The two gadgets make the reduction concrete: a property whose prefix is a single universal block over inputs to a fixed network is decided by appending a constant number of threshold nodes and asking the interval question. That is a statement about cost and not about meaning. Monotonicity, counterfactual fairness and interval verification are three diferent specifications, written over diferent data and checked by diferent frontends; only their dificulty coincides. Everything genuinely new in the remainder of this paper either alternates quantifiers or perturbs the network itself.

Practical reading. Monotonicity constraints are imposed by regulation in credit scoring, insurance pricing and parts of clinical decision support: a risk score must not fall when a risk factor rises. Proposition 4.6 says that checking monotonicity of an arbitrary trained network costs exactly what robustness certification costs, so the same solvers apply; and, by Remark 4.7, that for sigmoidal networks the problem is ∀R-hard, which is a precise explanation of why monotonicity checkers in the literature [21] restrict themselves to piecewise linear models or else abandon completeness. The design response - architectures that are monotone by construction, or counterexample-guided learning that repairs violations during training $[ 3 2 ] \textrm { - i s } ,$ , in this light, the reasonable one: it replaces a co-NP-complete check by a syntactic invariant or by an incremental search. The same reading applies to fairness: individual fairness in the sense of Dwork et al. [9] is a Lipschitz condition and therefore an instance of the global robustness problem already classified in [42], which is why its verification [16] reuses robustness machinery.

## 5 Alternating input-space properties

We now come to properties whose quantifier prefix alternates. These are genuinely outside the previously classified family, and the first of them is the central security question about a model of unknown provenance.

## 5.1 Backdoors

A backdoored network behaves normally on ordinary inputs but misclassifies any input carrying a trigger : in the standard construction [15] a small patch of fixed pixel values written over a fixed region of the image, which forces a target class whatever the rest of the image is. Deciding whether a network is backdoored is therefore an $\exists \forall$ question: does there exist a patch such that all inputs carrying it are classified as $j ! \mathscr { E }$

Problem 5.1 (Trigger(F)). Given an F-network $N .$ a patch $I \subseteq \{ 1 , \ldots , n \}$ , a finite alphabet $G \subseteq \mathbb { Q }$ , an LP specification A and a target coordinate $j ,$ decide whether

$$
\exists \tau \in G ^ { I } \forall x \in A : \qquad \arg \operatorname* { m a x } _ { j ^ { \prime } } N \big ( x [ I \mapsto \tau ] \big ) _ { j ^ { \prime } } = j ,
$$

where $x [ I \mapsto \tau ]$ denotes x with its I-coordinates overwritten by $\tau .$

Theorem 5.2. Trigger(ReLU) is $\Sigma _ { 2 } ^ { \mathrm { P } }$ -complete.

Proof. Membership. A trigger is an element of the finite grid $G ^ { I }$ and has polynomial bit size, so it may be guessed. What remains is an instance of $\mathrm { V I P } ( \mathrm { R e L U } )$ , which is in co-NP by Lemma 3.2. Hence Trigger $\mathrm { ( R e L U ) } \in \mathrm { N P ^ { c o - N P } = \Sigma _ { 2 } ^ { P } }$

Hardness. We reduce from $\exists \bar { u } \forall \bar { v } \psi ( \bar { u } , \bar { v } )$ with $\psi$ in 3-DNF, which is $\Sigma _ { 2 } ^ { \mathrm { P } }$ -complete. Let $\varphi : = \neg \psi$ , a 3-CNF over the same variables with m clauses, so that

$$
\exists \bar { u } \forall \bar { v } \psi ( \bar { u } , \bar { v } ) \iff \exists \bar { u } : \varphi ( \bar { u } , \cdot ) \mathrm { ~ i s ~ u n s a t i s f i a b l e } .\tag{1}
$$

Let $M : = M _ { \varphi }$ be the ReLU-network of Lemma 2.5, which we use through Corollary 2.6. Now build the Trigger instance. Let N be M with two output nodes, computing $c _ { 1 } : = M$ and $\begin{array} { r } { c _ { 2 } : = m - \frac { 1 } { 2 } } \end{array}$ , the latter by a node with no incoming edges and bias $m - { \textstyle { \frac { 1 } { 2 } } } ;$ ; take $j : = 2 ,$ , the index of $c _ { 2 } ,$ so that, with ties broken towards the smaller index, the target class is achieved exactly when $\begin{array} { r } { M < m - \frac { 1 } { 2 } } \end{array}$ . Let the patch I consist of the input coordinates $z _ { a }$ belonging to the u¯-variables, let $G : = \{ 0 , 1 \}$ , and let $A : = \mathbb { R } ^ { n }$

If u¯ is a Boolean assignment making $\varphi ( \bar { u } , \cdot )$ unsatisfiable, take $\tau \in \{ 0 , 1 \} ^ { I }$ to be u¯. By Corollary 2.6, applied with $S = I$ and $\sigma = \bar { u }$ , we have $M \leq m - 1 < m - \frac { 1 } { 2 }$ for every x, so the target class is achieved for every x and τ witnesses the Trigger instance.

Conversely let $\tau \in \{ 0 , 1 \} ^ { I }$ witness the Trigger instance, and let u¯ be the Boolean assignment it encodes. If $\varphi ( \bar { u } , \cdot )$ were satisfiable, then by Corollary 2.6 there is an input x carrying the patch τ with $\begin{array} { r } { M = m > m - \frac { 1 } { 2 } } \end{array}$ , so $c _ { 1 } > c _ { 2 }$ and the target class fails at that $x _ { i }$ contradicting the choice of τ . So $\varphi ( \bar { u } , \cdot )$ is unsatisfiable, and by (1) the $\Sigma _ { 2 } ^ { \mathrm { P } }$ instance is positive.

The construction is computable in linear time, so Trigger(ReLU) is $\Sigma _ { \mathrm { 2 } } ^ { \mathrm { P } } \mathrm { - h a r d }$

## 5.2 Why the quantisation matters

The restriction of the trigger to a finite alphabet is not a technical convenience. It is exactly what makes the reduction sound, and the reason is worth spelling out, because it has a practical counterpart.

Suppose the trigger were allowed to range over $\mathbb { R } ^ { I }$ . Then the adversary - the ∃ player - acquires a move with no Boolean counterpart. Setting a patch coordinate $z _ { a } : = \frac { 1 } { 2 }$ makes both literal values of the construction in the proof of Lemma 2.5 vanish,

$$
\alpha _ { a } = \alpha _ { \neg a } = 0 :
$$

the literal a and its negation are both false. This is the content of Lemma 2.5(iii), and it is precisely the failure of item (ii) to be tight. Every clause containing only u¯-literals then has clause value 0, whatever the v¯-part of the input does. In the game-theoretic reading, the ∃ player is permitted to abstain on a variable rather than commit to a truth value, and abstention is never worse for him than either commitment, because clause values are monotone in the literal values.

Proposition 5.3. The reduction of Theorem 5.2 is unsound for the relaxed problem: there is a formula for which ∃u¯∀v ψ¯ is false while the relaxed Trigger instance the reduction produces is positive. Moreover Trigger(ReLU) with G replaced by R is in $\Sigma _ { 2 } ^ { \mathrm { P } }$

Proof. Take the 3-CNF formula $\varphi : = ( u \vee { \neg u \vee u } )$ , with the single u¯-variable u, no v¯-variables and $m = 1$ ; equivalently $\psi = \neg \varphi$ , which is unsatisfiable. Since $\varphi ( \bar { u } , \cdot )$ is satisfiable for both values of $\bar { u } ,$ the equivalence (1) shows that ∃u¯∀v ψ¯ is false. The reduction of Theorem 5.2 produces a Trigger instance whose patch is the single coordinate $z _ { u } .$ . The real trigger $\begin{array} { r } { \tau : = \frac { 1 } { 2 } } \end{array}$ fixes the only input coordinate at $\textstyle { \frac { 1 } { 2 } }$ , so $M = 0$ by Lemma 2.5(iii). As $\begin{array} { r } { 0 < m - \frac { 1 } { 2 } } \end{array}$ 2 the target class is achieved at every input, and the relaxed instance is positive.

Membership is Lemma 3.2 with the prefix $\exists \tau \forall x$

Whether the real-valued version is $\Sigma _ { 2 } ^ { \mathrm { P } } .$ -hard is open. We do not claim that it is easier; what Proposition 5.3 establishes is that hardness cannot be transferred along this reduction, so a proof would have to defeat the abstention strategy rather than encode around it, and the two problems are not interchangeable for the purpose of moving hardness between them.

Practical reading. Three consequences, of which the third is the one we would press.

First, backdoor detection is provably harder than robustness certification. A robustness verifier is a co-NP oracle. If Trigger were decidable by polynomially many calls to such an oracle it would lie in $\mathrm { P } ^ { \mathrm { N P } } = \Delta _ { 2 } ^ { \mathrm { P } }$ , and a $\Sigma _ { 2 } ^ { \mathrm { P } }$ -complete problem in $\Delta _ { 2 } ^ { \mathrm { P } }$ collapses the polynomial hierarchy to $\Delta _ { 2 } ^ { \mathrm { P } }$ . So no scheme that reduces backdoor scanning to a bounded number of verification queries can be complete.

Second, this is consistent with the shape of the empirical literature. Detection methods such as Neural Cleanse [39] are search procedures over candidate triggers wrapped around a verification-like inner test - that is, $\Sigma _ { 2 } ^ { \mathrm { P } }$ algorithms with the outer search done heuristically. The theory says the outer search cannot be dispensed with.

Third, and most concretely: gradient-based trigger reconstruction relaxes the trigger to a continuous variable, and Proposition 5.3 says that the relaxation is not a faithful standin for the quantised problem. The relaxed search is permitted to occupy states - a patch value that activates neither a feature nor its complement - that no realisable trigger can occupy, and on the formula exhibited there those states alone make a clean network look backdoored. This is a candidate explanation, at the level of problem structure rather than optimisation, for the well-known phenomenon that such methods reconstruct “triggers” that do not correspond to any planted backdoor. A detection method whose search space is the quantised patch space is solving the stated problem; one whose search space is its convex relaxation is solving a diferent one, and the same caution applies as in the finite-precision setting studied by Alsmann, Lange and Sälzer [6], where the arithmetic in which a network is evaluated likewise changes the problem rather than approximating it.

## 5.3 Universal vulnerability

The dual prefix formalises the statement that a network is nowhere robust. It is the certified form of the observation behind universal adversarial perturbations [23], with the quantifiers the other way round: there the same perturbation fools most inputs, here every input is fooled by some perturbation.

Problem 5.4 $( \mathrm { U N I V V U L N } ( F ) )$ . Given N, an LP specification A, a rational $\varepsilon > 0$ and a metric $d \in \{ d _ { 1 } , d _ { \infty } \}$ , decide whether for every $x \in A$ there is τ with $\| \tau \| \le \varepsilon$ and arg max $_ { j } N ( x + \tau ) _ { j } \neq \arg$ max<sub>j</sub> $N ( x ) _ { j }$ .

Proposition 5.5. $U N I V V U L N ( F ) \in \Pi _ { 2 } ^ { \mathrm { P } }$ for semilinear F with rational coeficients.

Proof. Lemma 3.2 with the prefix ∀x ∃τ.

Hardness is open. We note that the complement asks for the existence of a single ε- robust input, which is the decision version of the question an empirical robustness evaluation is really asking, and that the same relaxation subtlety as in Section 5.2 may arise on the inner quantifier.

Practical reading. UnivVuln is the formal content of the claim “this model is not robust anywhere in the operational domain”, which is what an adversarial evaluation reports when every sampled input admits an attack. The evaluation establishes the property on a finite sample; the decision problem is the certified version, and it sits at the second level of the hierarchy, one above the per-input robustness question CR that verifiers answer. Certifying global non-robustness is therefore harder than certifying local robustness - an asymmetry worth knowing when a safety case tries to argue that a fallback mechanism is always needed.

## 6 Parameter-space uncertainty

Every robustness notion in the verification literature perturbs the input. Several of the failure modes that certification authorities care about perturb the weights: bit-flip and fault-injection attacks on the memory holding the parameters [25], single-event upsets and other transient faults in the accelerators that evaluate the network [20], a concern in avionics and space electronics; the device variation that makes analog and in-memory computing fast but uncertain [30], and the deviation introduced by post-training quantisation. Robustness to weight perturbation has been formalised and studied empirically [37]; what follows is its worst-case complexity.

Two features of these fault models shape the right formalisation. A fault is local, so each stored quantity should carry its own uncertainty set. And a fault corrupts a stored parameter, not an edge of the computation graph: a parameter held once in memory and read on many edges - as in every convolutional layer, and in any implementation backed by a shared weight table - is corrupted on all of them at once. We therefore separate parameters from edges.

Definition 6.1 (Parameterised architecture). A parameterised architecture A consists of a layered graph with edge set $E _ { i }$ , a parameter index set $\{ 1 , \ldots , P \}$ , a parameter map $\pi : E $ $\{ 1 , \ldots , P \}$ , rational biases, and an assignment of activations to nodes. A parameter vector $\theta \in \mathbb { R } ^ { P }$ induces the network $N _ { \theta }$ in which edge e carries the weight $\theta _ { \pi ( e ) }$ . The sharing degree of A is $\operatorname* { m a x } _ { p } | \pi ^ { - 1 } ( p ) |$ ; an architecture of sharing degree 1 is an ordinary network, in which parameters and edges coincide.

Definition 6.2 (Parameter specification). A box parameter specification for A consists of a nominal vector $\mathbf { \theta } ^ { \mathsf { ( 0 } } \in \mathbb { Q } ^ { P }$ and budgets $\delta \in \mathbb { Q } _ { \geq 0 } ^ { P }$ , and denotes

$$
W : = \prod _ { p = 1 } ^ { P } \bigl [ \theta _ { p } ^ { 0 } - \delta _ { p } , \theta _ { p } ^ { 0 } + \delta _ { p } \bigr ] .
$$

A parameter with $\delta _ { p } = 0$ is exact, and the width of W is $\operatorname* { m a x } _ { p } \delta _ { p }$ . More generally a parameter specification is any LP instance over the variables $\theta _ { 1 } , \ldots , \theta _ { P }$

Problem 6.3 (WFault(F), $\mathrm { W S A F E } ( F ) )$ . Given a parameterised F-architecture A, a parameter specification W and LP specifications A, B:

$$
\operatorname { W F A U L T } ( F ) : \quad \exists \theta \in W \ : \ : \ : \exists x \in A : \ : \ : \ : N _ { \theta } ( x ) \in B ,
$$

$$
\operatorname { W S A F E } ( F ) : \forall \in W \forall x \in A : \ N _ { \theta } ( x ) \in B .
$$

WFault is the fault-injection attacker’s question - is there a fault within the model that drives the network into the unsafe region B - and WSafe is the certifier’s.

Theorem 6.4. WFault({id}) is ∃R-complete. Hardness holds already for instances in which

(a) W is a box parameter specification,

(b) the sharing degree is at most $2 ,$

(c) every budget satisfies $\delta _ { p } \leq P ^ { - c }$ , for any constant $c > 0$ fixed in advance, and

(d) B is a conjunction of linear equations on output node values.

The same holds verbatim with ReLU in place of id. Moreover $W S A F E ( \{ \mathrm { R e L U } \} )$ is ∀Rcomplete, already under $( a ) – ( c )$ and with B a single open half-space.

Proof. Membership. Introduce a real variable for every parameter and every node value. Each node contributes $\begin{array} { r } { y _ { v } = \sum _ { u } \theta _ { \pi ( u v ) } y _ { u } + b _ { v } } \end{array}$ , a polynomial equation with rational coeficients, bilinear in the unknowns; $W , A$ and B contribute linear inequalities. The instance is positive if the resulting system has a real solution, so $\mathrm { W F A U L T } ( F ) \in \exists \mathbb { R }$ for every semilinear $F ,$ the semilinear activations contributing Boolean combinations of linear conditions. $\operatorname { W S A F E } ( F ) \in \forall \mathbb { R }$ by complementation.

Hardness of WFault. We reduce from Range-Etr-Inv (Theorem 2.4); the range parameter δ is chosen at the end, when we verify clause (c). Let the instance have k variables $\xi _ { 1 } , \ldots , \xi _ { k }$ with promised intervals $I ( \xi _ { i } ) \subseteq [ { \frac { 1 } { 2 } } , 2 ]$ and constraint list C. The architecture has a single input node $x _ { 1 }$ and the input specification is $A : = \{ x _ { 1 } = 1 \}$ , so that $x _ { 1 }$ carries the value 1. All nodes are identity nodes. Each parameter introduced below is given the box named with it, and every other parameter is exact with nominal value 1.

• Variables. For each $\xi _ { i }$ a parameter $p _ { i }$ with box $I ( \xi _ { i } )$ , and a node $v _ { i }$ with a single incoming edge from $x _ { 1 }$ carrying $p _ { i }$ and bias $0 ,$ so that $v _ { i } = \theta _ { p _ { i } }$ ranges exactly over $I ( \xi _ { i } )$ . The node $v _ { i }$ is the variable $\xi _ { i } .$ , and π is injective on these edges.

• Inversions. For each constraint $\xi _ { a } \cdot \xi _ { b } = 1$ a fresh parameter q with box $I ( \xi _ { b } )$ , read on exactly two edges: an edge from $v _ { a }$ into a new node m, and an edge from $x _ { 1 }$ into a new node $^ { O , }$ both with bias 0. Then $m = \theta _ { q } v _ { a }$ and $o = \theta _ { q } .$ . Add to $B$ the linear equations $o = v _ { b }$ and $m = 1$ . The first forces $\theta _ { q } = \xi _ { b }$ , whereupon the second says precisely $\xi _ { a } \xi _ { b } = 1$ The node $o$ makes the value of a parameter available as the value $o f$ a node, which is what lets the output specification state an equality between two parameters; this is the one place where sharing is used.

• Sums. For each constraint $\xi _ { a } + \xi _ { b } = \xi _ { c }$ add to B the linear equation $v _ { a } + v _ { b } = v _ { c }$ . No new parameter is needed.

Pad with exact edges of nominal weight 1 so that all named nodes lie in the output layer. The architecture has $O ( k + | C | )$ nodes and $P = O ( k + | C | )$ parameters, and sharing degree 2, attained only by the parameters $q .$

For clause (c), note first that we may assume $| C | = O ( k ^ { 3 } )$ : there are at most $k ^ { 3 } +$ $k ^ { 2 }$ distinct constraints over k variables, and duplicates may be deleted without afecting satisfiability. Hence $P = O ( k ^ { 3 } )$ Given the constant $^ { c , }$ apply Theorem 2.4 with range parameter $\delta = O ( k ^ { - ( 3 c + 1 ) } )$ ; then every budget is at most $\delta \leq P ^ { - c }$ for all suficiently large $k ,$ and the finitely many smaller instances may be decided outright. A pair $( \theta , x )$ witnessing WFault assigns to each $v _ { i }$ a value in $I ( \xi _ { i } )$ satisfying all the sum and inversion constraints, that is, a satisfying assignment of the Range-Etr-Inv instance; and conversely. Hence WFault({id}) is ∃R-hard under $\mathrm { ( a ) - ( d ) }$

The ReLU case. Every node value occurring in the construction is positive: $x _ { 1 } = 1$ $\begin{array} { r } { v _ { i } \in I ( \xi _ { i } ) \ \subseteq [ \frac { 1 } { 2 } , 2 ] , o = \theta _ { q } \in [ \frac { 1 } { 2 } , 2 ] , m = \theta _ { q } v _ { a } \in [ \frac { 1 } { 4 } , 4 ] } \end{array}$ , and the padding edges have weight 1; all biases are 0. Declaring every node a ReLU node therefore changes no value, and the same instance witnesses ∃R-hardness of WFault({ReLU}).

Hardness of WSafe({ReLU}). Let $\ell _ { 1 } , \ldots , \ell _ { r }$ be the rational afine forms in the node values $v _ { i } , o ,$ m whose vanishing expresses the conjunction B above - one for each of the equations $o - v _ { b } , m - 1$ and $v _ { a } + v _ { b } - v _ { c } - \mathrm { s o }$ that the Range-Etr-Inv instance is satisfiable if $\exists \theta \in W$ with $\ell _ { 1 } = \cdot \cdot \cdot = \ell _ { r } = 0$ . Extend the network by the nodes $\mathrm { R e L U } ( \ell _ { s } )$ and $\mathrm { R e L U } ( - \ell _ { s } )$ for $s \leq r$ and the single output node

$$
h : = \sum _ { s = 1 } ^ { r } \bigl ( \mathrm { R e L U } ( \ell _ { s } ) + \mathrm { R e L U } ( - \ell _ { s } ) \bigr ) ,
$$

which is non-negative, so the ReLU on the output node is the identity there. Then $h \geq 0$ always, and $h ( \theta ) = 0$ if every $\ell _ { s }$ vanishes. All added edges are exact. Take $B ^ { \prime } : = \{ h > 0 \}$ ， a single open half-space. Since A pins the input,

$$
\begin{array} { l l l } { \operatorname { W S A F E } ( A , W , A , B ^ { \prime } ) \mathrm { ~ h o l d s } } & { \iff } & { \forall \theta \in W : \ h ( \theta ) > 0 } \\ & { \iff } & { \mathrm { t h e ~ R A N G E - E T R - I N V ~ i n s t a n c e ~ i s ~ u n s a t i s f i a b l e } , } \end{array}
$$

and unsatisfiability of Range-Etr-Inv is ∀R-complete by Theorem 2.4. The reduction is computable in linear time and preserves $\mathrm { ( a ) - ( c ) }$ □

Remark 6.5 (What is not claimed). Three restrictions of Theorem 6.4 are worth naming, because each is a natural question we leave open.

Sharing. Hardness uses sharing degree 2: the parameter q is read both on the edge forming the product and on the edge exposing its value as a node value. For an architecture of sharing degree 1 under a box specification there is no mechanism for equating two parameters, and we do not know the complexity of WFault; this is Open Problem (1). Under the general LP parameter specification of Definition 6.2 the tying is expressible directly, so there the theorem holds at sharing degree 1 as well.

The identity case of WSafe. The ∀R-hardness argument aggregates a conjunction of violations into a single half-space using ReLU, which an {id}-network cannot do: an afine network cannot compute a non-negative function vanishing exactly on a prescribed afine subspace. Whether WSafe({id}) is ∀R-hard is open.

Single-parameter faults. If exactly one parameter is inexact, every node value is afine in that parameter and the constraint system is bilinear in one scalar unknown and the inputs. Theorem 6.4 says nothing about this case, and we do not know its complexity.

Remark 6.6 (The point of the theorem). For {id}-networks, every verification problem studied in [42] lies in P: such a network computes an afine map and each question is a linear program [42, Prop. 3]. Theorem 6.4 says that making the parameters uncertain, within boxes whose width shrinks inverse-polynomially in the size of the network, takes the same networks from P to ∃R-complete. Nothing about the activation function is involved, which is precisely why the phenomenon is invisible in a classification indexed by activation functions: the cause is that a parameter multiplies an activation, so uncertainty in the parameters makes the network a polynomial rather than a linear map of the unknowns. Tightening the tolerance is therefore no help, which is clause (c) of the theorem and the reason for reducing from the range-restricted Range-Etr-Inv of [4] rather than from Etr-Inv: the latter confines its variables to the fixed box [ <sup>1</sup>, 2], whose relative width no rescaling of the network can reduce.

The summary is worth stating as a slogan. The activation function decides the complexity of input-space questions; the arithmetic decides the complexity of parameter-space questions. The first axis is classified; the second is not, and on it even the trivial activation is ∃Rcomplete.

Remark 6.7. The result also sharpens the relation to learning. Training ReLU networks is ∃R-complete [2], and the intuition usually ofered is that training searches over weights. Theorem 6.4 isolates that intuition from everything else: it is neither the optimisation nor the activation that produces ∃R-hardness, but the product of a parameter and an activation. Verification under parameter uncertainty is ∃R-complete for the simplest networks there are, and for arbitrarily tight tolerances.

It is worth contrasting this with the smoothed-analysis result of Soltanalian [33], which perturbs every parameter by independent clipped Gaussian noise and shows that exact ReLU verification has no smoothed-polynomial complete algorithm unless NP ⊆ BPP. There the parameters are randomised once, as a preprocessing of the instance, and the problem asked afterwards is still the input-space one; the content is that worst-case hardness survives generic noise. Here the parameters are quantified inside the problem, over an adversarial set, and the content is that the problem leaves the polynomial hierarchy. Read together, the two say that parameter uncertainty neither dissolves the dificulty of verification nor leaves it where it was.

Practical reading. The consequence for tool building is concrete and, we think, underappreciated. Existing complete verifiers encode a ReLU network as a mixed integer linear program [36] or solve it by SMT and branch and bound [18, 40]: the weights are constants, the activations are the binary variables, and the whole question is linear. It is tempting to certify against parameter faults by adding the parameters as variables to the same encoding. Theorem 6.4 says that this cannot work as stated, because the resulting constraints are bilinear; and more than that, it says no reformulation can restore an MILP encoding of polynomial size unless ∃R = NP. Certified parameter robustness is not the input-space problem with a larger box. It is a problem of a diferent arithmetic type, and the methods that apply to it are those of polynomial optimisation - semidefinite relaxations, branchand-bound over parameter intervals, interval arithmetic with refinement - all of which are incomplete in a way the MILP methods are not.

For the fault models themselves the reading is as follows. Analog and in-memory accelerators [30] give every parameter a small non-zero budget simultaneously, which is exactly the setting of Theorem 6.4; and clause (c) says that tightening the device tolerance does not help, since the problem is already ∃R-complete at inverse-polynomial width. A bit flip in a fixed-point weight [25] corrupts one stored parameter, which in a convolutional layer is read on many edges at once; that the hardness construction requires a parameter read on more than one edge is therefore not an artefact but a feature of the fault model. The case in which a single corrupted parameter is read on a single edge is a diferent question, and Remark 6.5 records it as open: there the node values are afine in the unknown, and the reduction above says nothing. What the theorem does establish is that the simultaneoustolerance model, which is the one analog hardware presents, is outside the reach of the current complete-verification stack, and that is consistent with the mitigations in use being redundancy and error-correcting codes on the parameter memory rather than verification.

## 7 Privacy and provenance

We turn to three properties concerning what a model reveals and what it can be made to forget. Each has a clear practical driver, and they occupy three diferent levels of the picture.

## 7.1 Inversion resistance

Model inversion attacks [11] reconstruct an input, or a sensitive attribute of ${ \mathrm { i t } } ,$ from the model’s output. Whether such reconstruction is possible at all is a geometric question: does the output pin the input down?

Problem 7.1 (InvRes(F)). Given an F-network N, LP specifications A and B, a rational $\delta \geq 0$ and a metric $d \in \{ d _ { 1 } , d _ { \infty } \}$ , decide whether

$$
\mathrm { d i a m } \big ( \{ x \in A : N ( x ) \in B \} \big ) \ \leq \ \delta ,
$$

that is, whether all $x , x ^ { \prime } \in A$ with $N ( x ) , N ( x ^ { \prime } ) \in B$ satisfy $d ( x , x ^ { \prime } ) \leq \delta$

A network is inversion resistant for the output region B when the answer is negative for small δ: many admissible inputs produce outputs in B, so observing an output in $B$ reveals little. The decision problem as stated asks the opposite, for convenience of statement.

Theorem 7.2. InvRes(ReLU) is co-NP-complete.

Proof. Membership. The condition is $\forall x \forall x ^ { \prime } ( \cdot \cdot \cdot  d ( x , x ^ { \prime } ) \leq \delta )$ , a single universal block. For $d = d _ { \infty }$ the conclusion max<sub>i</sub> $| x _ { i } - x _ { i } ^ { \prime } | \leq \delta$ is the conjunction of the 2n linear inequalities $x _ { i } - x _ { i } ^ { \prime } \leq \delta$ and $x _ { i } ^ { \prime } - x _ { i } \leq \delta$ , so Lemma 3.2 with $k = 1$ applies.

For $d = d _ { 1 }$ the lemma does not apply directly: the condition $\textstyle \sum _ { i } | x _ { i } - x _ { i } ^ { \prime } | \leq \delta$ is equivalent to the conjunction of $\begin{array} { r } { \sum _ { i } s _ { i } ( x _ { i } - x _ { i } ^ { \prime } ) \leq \delta } \end{array}$ over all sign vectors $s \in \{ - 1 , 1 \} ^ { n }$ , and that quantifier-free formula has exponential size. We argue instead on the complement, which asks for $x , x ^ { \prime } \in A$ with $N ( x ) , N ( x ^ { \prime } ) \ \in \ B$ and $\textstyle \sum _ { i } | x _ { i } - x _ { i } ^ { \prime } | \ > \ \delta$ Guess the activation patterns of the two copies of N and a sign vector $s \in \{ - 1 , 1 \} ^ { n }$ , and verify the single linear inequality $\begin{array} { r } { \sum _ { i } s _ { i } ( x _ { i } - x _ { i } ^ { \prime } ) > \delta } \end{array}$ over the resulting polyhedron by linear programming. This is sound because $\begin{array} { r } { \sum _ { i } s _ { i } t _ { i } \le \sum _ { i } \left| t _ { i } \right| } \end{array}$ for every s, and complete because $s _ { i } = \mathrm { s i g n } ( x _ { i } - x _ { i } ^ { \prime } )$ attains the sum; so the complement is in NP and Inv $\mathrm { R e s } ( \mathrm { R e L U } ) \in \mathrm { c o } { \mathrm { - N P } }$ for $d _ { 1 }$ as well. (Proposition 7.3 below gives a second route, the case $p = 1$ of a uniform argument for all $d _ { p } . )$

Hardness. We reduce the complement of NNReach(ReLU), which is co-NP-hard [27]. Let $( N , A , B )$ be a reachability instance with N on n inputs. Let $N ^ { \prime }$ be the network on $n + 1$ inputs $( x , t )$ defined by $N ^ { \prime } ( x , t ) : = N ( x )$ , obtained from N by adding an input node with no outgoing edges; let $A ^ { \prime } : = A \land 0 \leq t \leq 1$ and $B ^ { \prime } : = B ;$ and let $\delta : = \bar { \frac { 1 } { 2 } }$ with $d = d _ { \infty }$ Then

$$
\{ ( x , t ) \in A ^ { \prime } : N ^ { \prime } ( x , t ) \in B ^ { \prime } \} \ = \ \{ x \in A : N ( x ) \in B \} \times [ 0 , 1 ] .
$$

If the reachability instance is positive this set contains $( x , 0 )$ and $( x , 1 )$ for some $x ,$ so its $d _ { \infty } \mathrm { - d i a m e t e r }$ is at least $1 > \delta$ and the InvRes instance is negative. If it is negative the set is empty and the InvRes condition holds vacuously.<sup>1</sup> The construction is linear time.

Proposition 7.3. For each fixed positive integer $p ,$ InvRes(ReLU) is co-NP-complete when d is the metric $d _ { p } ;$ in particular for the Euclidean metric $d _ { 2 }$ . The same holds when $p$ is part of the input and encoded in unary.

Proof. Hardness is the reduction of Theorem 7.2, which uses only that the appended coordinate t contributes a distance of 1 between $( x , 0 )$ and $( x , 1 )$ ; that holds in every $d _ { p } .$

For membership, consider the complement, which asks for $x , x ^ { \prime } \in A$ with $N ( x ) , N ( x ^ { \prime } ) \in$ $B$ and $d _ { p } ( x , x ^ { \prime } ) > \delta$ . Guess the activation patterns of the two copies of $N$ . What remains is a rational polyhedron $P \subseteq \mathbb { R } ^ { 2 n }$ whose description has size polynomial in the instance, together with the single constraint

$$
f ( x , x ^ { \prime } ) : = \sum _ { i = 1 } ^ { n } | x _ { i } - x _ { i } ^ { \prime } | ^ { p } > \delta ^ { p } ,
$$

both sides of which are rational because $p$ is an integer. The hypothesis that $p$ is fixed, or given in unary, is used here and only here: a $p$ encoded in binary would make $\delta ^ { p }$ and the summands rationals of exponential bit length, and the verifier could not evaluate the certificate in polynomial time. The function $f$ is convex, being the $p \textmd { - }$ -th power of a norm composed with a linear map; convexity needs $p \geq 1$ , which holds as $p$ is a positive integer.

We claim that the supremum of a convex function over a rational polyhedron is either $+ \infty ,$ witnessed by a recession direction, or attained at a vertex of its pointed part. Write $P = Q + L$ with $L$ the lineality space and $Q$ pointed. A convex function that is not constant on a line is unbounded above on it, so either $f$ is constant along $L ,$ in which case the supremum over $P$ equals the supremum over $Q ,$ or the supremum is $+ \infty$ . On $Q = \operatorname { c o n v } ( V ) + \operatorname { c o n e } ( R )$ , a convex function bounded above is non-increasing along every recession direction, because a convex function of one variable that is bounded above on $[ 0 , \infty )$ is non-increasing there; so the supremum over $Q$ equals the supremum over conv $( V )$ which a convex function attains at an extreme point, that is, at a vertex.

Vertices of a rational polyhedron and generators of its recession cone have bit size polynomial in its description, so guessing one of them alongside the activation patterns yields a polynomial-size certificate, and the complement is in $\mathrm { N P } .$ □

Remark 7.4. It is worth saying why this does not leave the polynomial hierarchy, since the constraint $d _ { 2 } ( x , x ^ { \prime } ) ~ > ~ \delta$ is semi-algebraic and not semilinear, and elsewhere in this paper - Section 6 - leaving linear arithmetic costs exactly that. The diference is where the non-linearity sits. In Theorem 6.4 the products are inside the network, one per edge, and they compose: a chain of d nodes realises a monomial of degree $d ,$ which is what allows a network to encode an arbitrary polynomial system. Here there is a single convex polynomial inequality imposed on variables that are otherwise constrained only linearly, and nothing composes. One convex constraint over a polyhedron is not an existential theory of the reals.

Note also that the certificate is not an appeal to the trust region subproblem. Maximising a convex quadratic over a polytope is NP-hard; what gives the polynomial certificate is the location of the maximiser, not the cost of finding it.

For rational $p$ that is not an integer the argument breaks at the last step: $\delta ^ { p }$ and the summands are then algebraic irrationals, and comparing them is a question about sums of radicals, which is not known to be decidable in polynomial time. Whether InvRes is in co-NP for such $p$ is open.

Practical reading. InvRes is the certified form of the question asked before publishing an embedding, a logit vector or a confidence score: does releasing this number identify its source? Because the problem is co-NP-complete, it is decidable by exactly the tools already used for robustness, with a front-end that runs two copies of the network and constrains their outputs to the same region. That is an unusually favourable situation: a privacy property that costs no more than a robustness property. The caveat is that InvRes is a worst-case, geometric notion of leakage and is not a substitute for a statistical one such as diferential privacy [10]; a network can be inversion resistant in this sense and still leak on the input distribution that actually occurs.

## 7.2 Output indistinguishability

Proposition 4.2 observed that exact non-interference is too brittle to be a usable specification. The metric version compares what the model’s outputs look like on two populations.

Problem 7.5 (OutInd(F)). Given N, LP specifications $A _ { 1 } , A _ { 2 }$ , a rational $\delta \geq 0$ and $d \in \{ d _ { 1 } , d _ { \infty } \}$ , decide whether the Hausdorf distance between $N ( A _ { 1 } )$ and $N ( A _ { 2 } )$ is at most $\delta ;$ equivalently, whether

$$
\forall x \in A _ { 1 } \ \exists x ^ { \prime } \in A _ { 2 } : \ d ( N ( x ) , N ( x ^ { \prime } ) ) \leq \delta \qquad \mathrm { a n d \ s y m m e t r i c a l l y } .
$$

Proposition 7.6. $O U T I N D ( F ) \in \Pi _ { 2 } ^ { \mathrm { P } }$ for semilinear F with rational coeficients.

Proof. Lemma 3.2 with the prefix $\forall x \exists x ^ { \prime } ;$ the two symmetric conjuncts share the prefix shape and may be combined. □

We conjecture $\Pi _ { 2 } ^ { \mathrm { P } }$ -completeness. The natural route is the reduction of Theorem 5.2 with the roles of the quantifiers exchanged, and the obstruction is the same relaxation phenomenon: the inner ∃ ranges over a continuum.

Practical reading. OutInd is the certified counterpart of an audit for attribute inference or for membership inference [31]. If the images of two populations - inputs with and without a protected attribute, or members and non-members of the training set - are within $\delta$ in Hausdorf distance, then no adversary observing only the output can separate them by more than $\delta ,$ whatever its computational power. That is a strong guarantee of exactly the kind regulation asks for and empirical audits cannot give. The price is one quantifier alternation:

unlike InvRes, this is not a problem the current verification stack can be pointed at. We regard closing the gap between NonInt (co-NP-complete but unsatisfiable in practice) and OutInd $\mathrm { ( H _ { 2 } ^ { P } }$ but meaningful) as the most useful open problem in this section.

## 7.3 Watermark removability

Model watermarking embeds a signature either in the parameters or, in the variant we formalise, in a set of key inputs with prescribed outputs obtained by deliberate backdooring $[ 5 ] _ { ; }$ , so that the owner of a stolen copy can demonstrate provenance. The security question is whether an attacker can produce a model that is as good as the original on real data but no longer answers the keys correctly. Because watermark keys are deliberately out of distribution, the key inputs lie outside the operational region.

Problem 7.7 (WMRemove(F)). Given an F-network N, an LP specification A, a finite key set $K = \{ ( k _ { 1 } , y _ { 1 } ) , \dots , ( k _ { r } , y _ { r } ) \}$ with $k _ { i } \notin { \cal A }$ and $N ( k _ { i } ) = y _ { i }$ , and a size bound $s \in \mathbb N$ decide whether there is an F-network $N ^ { \prime }$ with at most s nodes such that $N ^ { \prime } ( x ) = N ( x )$ for all $x \in A$ and $N ^ { \prime } ( k _ { i } ) \neq y _ { i }$ for some i.

Proposition 7.8. If the weights of the candidate network $N ^ { \prime }$ are restricted to rationals of bit size polynomial in the instance, then WMRemove $F ) \in \Sigma _ { 2 } ^ { \mathrm { P } }$ for semilinear F with rational coeficients.

Proof. Guess $N ^ { \prime }$ , which is then of polynomial size, and verify $N ^ { \prime } | _ { A } \equiv N | _ { A }$ with a co-NP oracle by Lemma $3 . 2 ;$ the key conditions are evaluated directly. □

The bit-size restriction is the same one implicit in the $\Pi _ { 2 } ^ { \mathrm { P } }$ upper bound for Min in [42, Thm. 11], and removing it is open in both cases. We conjecture $\Sigma _ { 2 } ^ { \mathrm { P } }$ -completeness, by adapting the Σ<sup>P</sup>-hardness proof for minimum equivalent DNF [38] through the reduction chain relating NE and Min in [42].

Practical reading. A watermarking scheme is usually validated empirically, by showing that fine-tuning, pruning and extraction attacks fail to remove the mark [22]. WMRemove is the corresponding worst-case statement, and its shape tells us what such a validation can and cannot establish: since the attacker’s question is $\exists N ^ { \prime } \forall x$ , an equivalence checker alone - which answers the inner ∀ - can never certify that a watermark is irremovable, only that a specific attacker’s candidate failed. Provenance claims built on watermarking therefore rest on a $\Sigma _ { 2 } ^ { \mathrm { P } }$ assumption, and should be stated as such. The corresponding question for the parameter-space attacker, in which $N ^ { \prime }$ ranges over small perturbations of N rather than over all small networks, inherits the arithmetic of Section 6 and is the more realistic threat model.

## 8 Repair

A network that fails certification is not discarded; it is patched, and the patch must be small enough that the rest of the safety case survives.

Problem 8.1 (Repair(F)). Given a parameterised F-architecture A with nominal parameters $\theta ^ { 0 }$ , LP specifications $A , B ,$ , a budget $k \in \mathbb N$ and a parameter specification $W$ , decide whether there is $\theta \in W$ difering from $\theta ^ { 0 }$ in at most k coordinates with $N _ { \theta } ( x ) \in B$ for all $x \in A$

Proposition 8.2. Let F be semilinear with rational coeficients. If W confines the modified parameters to a finite rational grid, then Repair(F) lies in $\Sigma _ { 2 } ^ { \mathrm { P } }$ . If W is an arbitrary LP parameter specification, then Repair(F) lies in ∃∀R.

Proof. In the first case guess the k modified parameters and their grid values, then apply Lemma 3.2 with $k = 1$ to the remaining VIP instance. In the second, the condition is ∃θ ∀x ∀y¯ over a system of polynomial constraints, which is the defining form of ∃∀R [29].

We conjecture completeness in both cases.

Practical reading. Repair tools [35, 13] search over weights continuously, by linear programming or gradient descent with a verification oracle in the loop, and then round. Proposition 8.2 says that the rounding is not a detail of implementation but a change of problem: with a quantised parameter search the problem is in the polynomial hierarchy, and with a continuous one it is in the real hierarchy one level above ∃R. This is the same phenomenon as in Section 5.2, and as in the finite-precision setting of [6], in the parameter space rather than the input space, and it suggests the same methodological conclusion - that a tool should decide, and state, which of the two problems it is solving.

## 9 Summary and open problems

<table><tr><td>problem</td><td>prefix</td><td>parameter</td><td>status</td></tr><tr><td>NNREACH</td><td>∃x</td><td></td><td>NP-complete [27]</td></tr><tr><td>VIP, NE</td><td>∀x</td><td></td><td>co-NP-complete [42]</td></tr><tr><td>NONINT</td><td>∀x</td><td></td><td>co-NP-complete (Prop. 4.2)</td></tr><tr><td>MONO, CFAIR</td><td>∀x</td><td></td><td>co-NP-complete (Prop. 4.6, [44])</td></tr><tr><td>INVRES</td><td> $\forall x \forall x ^ { \prime }$ </td><td></td><td>co-NP-complete (Thm. 7.2)</td></tr><tr><td>MIN</td><td> $\exists N ^ { \prime } \forall x$ </td><td>structure</td><td>in ΠI2 [42]</td></tr><tr><td>TRIGGER (quantised)</td><td> $\exists \tau \forall x$ </td><td>input</td><td> $\Sigma _ { 2 } ^ { \mathrm { P } } .$  -complete (Thm. 5.2)</td></tr><tr><td>TRIGGER (real)</td><td> $\exists \tau \forall x$ </td><td>input</td><td>in  $\mathrm { \dot { \Sigma } _ { 2 } ^ { P } }$  ; hardness open</td></tr><tr><td>UNIVVULN</td><td> $\forall x \exists \tau$ </td><td>input</td><td>in  $\Pi _ { 2 } ^ { \mathrm { P } }$ </td></tr><tr><td>OUTIND</td><td> $\forall x \exists x ^ { \prime }$ </td><td></td><td>in  $\Pi _ { 2 } ^ { \mathbf { \tilde { P } } }$ </td></tr><tr><td>WMREMOVE</td><td> $\exists N ^ { \prime } \forall x$ </td><td>structure</td><td>in  $\Sigma _ { 2 } ^ { \bar { \mathrm { P } } }$  (Prop. 7.8)</td></tr><tr><td>REPAIR (quantised)</td><td> $\exists \theta \forall x$ </td><td>parameters</td><td>in  $\Sigma _ { 2 } ^ { \mathrm { P } }$ </td></tr><tr><td>REPAIR (continuous)</td><td> $\exists \theta \forall x$ </td><td>parameters</td><td>in ∃∀RE</td></tr><tr><td>WFAULT</td><td> $\exists \theta \exists x$ </td><td>parameters</td><td>∃R-complete (Thm. 6.4)</td></tr><tr><td>WSAFE (ReLU)</td><td> $\forall \theta \forall x$ </td><td>parameters</td><td>∀R-complete (Thm. 6.4)</td></tr><tr><td>WSAFE (id)</td><td>A∀x</td><td>parameters</td><td>in ∀R; hardness open</td></tr></table>

Table 1: The problems of this paper in the schema of Section 3. All completeness entries are for ReLU unless stated otherwise. Rows above the Min line are first-level; rows with a parameter-space parameter leave the polynomial hierarchy.

The open problems we would rank highest are the following.

(1) Is WFault still ∃R-hard for architectures of sharing degree 1 - that is, when every parameter is read on a single edge, so that a box specification gives each of them an independent tolerance? The construction of Theorem 6.4 needs one parameter read twice, and without it the node values available are sums of path monomials in which each deeper parameter occurs in a single position, which appears to express interval constraints rather than products. Either a hardness proof or a polynomial-hierarchy upper bound would be interesting.

(2) What is the complexity of the single-corrupted-parameter fault model, in which exactly one budget is non-zero? The node values are then afine in the unknown, so Theorem 6.4 does not apply; NP-hardness looks likely and ∃R-hardness unlikely, but we have neither. Relatedly, is WSafe({id}) ∀R-hard (Remark 6.5)?

(3) Is real-valued Trigger $\Sigma _ { \mathrm { 2 } } ^ { \mathrm { P } } \mathrm { \mathrm { - h a r d ? } }$ By Proposition 5.3 a proof must defeat the abstention strategy, and a proof of the opposite - that the relaxation is strictly easier - would be more interesting still, since it would say that continuous trigger search is solving a tractable shadow of the stated problem.

(4) Is OutInd Π<sup>P</sup>-complete, and is there a specification between NonInt and OutInd that is both meaningful for trained models and decidable at the first level? Is InvRes co-NP-hard under the promise that the preimage is non-empty?

(5) Are WMRemove and Repair complete for their classes, and can the bit-size restriction in Proposition 7.8 - and in the corresponding bound for Min - be removed?

(6) The counting versions. Quantitative information flow, demographic parity and probabilistic certification all ask for a proportion rather than a verdict, and none of the problems above has been studied in its counting form, although approximate counting has been applied to neural networks with security applications in mind [7]. We expect #P-completeness for the first-level problems over ReLU networks, and the sigmoidal case appears to need a counting class over ∃R that does not yet exist.

## Declarations

Competing interests. The author declares that he has no competing interests.

Use of generative artificial intelligence. A large language model (Claude, Anthropic) was used as a research assistant during the preparation of this manuscript. Its use went beyond language editing: it was used to propose candidate formalisations of several of the decision problems studied here, to draft arguments and expository text, and to assemble candidate references. Every definition, problem statement, theorem and proof in this paper has been checked by the author, who takes full responsibility for the correctness of the mathematical content and for the manuscript as a whole. The model is not an author and is not accountable for the work.

## References

[1] Mikkel Abrahamsen, Anna Adamaszek, and Tillmann Miltzow. The art gallery problem is ∃R-complete. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing (STOC), pages 65–73, 2018.

[2] Mikkel Abrahamsen, Linda Kleist, and Tillmann Miltzow. Training neural networks is ∃R-complete. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

[3] Mikkel Abrahamsen and Tillmann Miltzow. Dynamic toolbox for ETRINV, 2019. arXiv:1912.08674.

[4] Mikkel Abrahamsen, Tillmann Miltzow, and Nadja Seiferth. Framework for ∃Rcompleteness of two-dimensional packing problems. In 61st IEEE Annual Symposium on Foundations of Computer Science (FOCS), 2020. Full version: arXiv:2004.07558.

[5] Yossi Adi, Carsten Baum, Moustapha Cisse, Benny Pinkas, and Joseph Keshet. Turning your weakness into a strength: watermarking deep neural networks by backdooring. In 27th USENIX Security Symposium, pages 1615–1631, 2018.

[6] Eric Alsmann, Martin Lange, and Marco Sälzer. The complexity of verifying feedforward neural networks in quantised settings, 2026. arXiv:2605.29537.

[7] Teodora Baluta, Shiqi Shen, Shweta Shine, Kuldeep S. Meel, and Prateek Saxena. Quantitative verification of neural networks and its security applications. In Proceedings of the 2019 ACM SIGSAC Conference on Computer and Communications Security (CCS), pages 1249–1264, 2019.

[8] Christopher Brix, Mark Niklas Müller, Stanley Bak, Taylor T. Johnson, and Changliu Liu. First three years of the international verification of neural networks competition (VNN-COMP). International Journal on Software Tools for Technology Transfer, 25:329–339, 2023.

[9] Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, and Richard Zemel. Fairness through awareness. In Proceedings of the 3rd Innovations in Theoretical Computer Science Conference (ITCS), pages 214–226, 2012.

[10] Cynthia Dwork and Aaron Roth. The algorithmic foundations of diferential privacy. Foundations and Trends in Theoretical Computer Science, 9(3–4):211–407, 2014.

[11] Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security (CCS), pages 1322–1333, 2015.

[12] Joseph A. Goguen and José Meseguer. Security policies and security models. In IEEE Symposium on Security and Privacy, pages 11–20, 1982.

[13] Ben Goldberger, Guy Katz, Yossi Adi, and Joseph Keshet. Minimal modifications of deep neural networks using verification. In 23rd International Conference on Logic for Programming, Artificial Intelligence and Reasoning (LPAR), pages 260–278, 2020.

[14] Shafi Goldwasser, Michael P. Kim, Vinod Vaikuntanathan, and Or Zamir. Planting undetectable backdoors in machine learning models. In 63rd IEEE Annual Symposium on Foundations of Computer Science (FOCS), pages 931–942, 2022.

[15] Tianyu Gu, Brendan Dolan-Gavitt, and Siddharth Garg. BadNets: identifying vulnerabilities in the machine learning model supply chain. arXiv:1708.06733, 2017.

[16] Philips George John, Deepak Vijaykeerthy, and Diptikalyan Saha. Verifying individual fairness in machine learning models. In Conference on Uncertainty in Artificial Intelligence (UAI), 2020.

[17] Guy Katz, Clark Barrett, David L. Dill, Kyle Julian, and Mykel J. Kochenderfer. Reluplex: an eficient SMT solver for verifying deep neural networks. In Computer Aided Verification (CAV), volume 10426 of Lecture Notes in Computer Science, pages 97–117, 2017.

[18] Guy Katz, Derek A. Huang, Duligur Ibeling, Kyle Julian, Christopher Lazarus, Rachel Lim, Parth Shah, Shantanu Thakoor, Haoze Wu, Aleksandar Zeljić, David L. Dill, Mykel J. Kochenderfer, and Clark Barrett. The Marabou framework for verification and analysis of deep neural networks. In Computer Aided Verification (CAV), volume 11561 of Lecture Notes in Computer Science, pages 443–452, 2019.

[19] Matt J. Kusner, Joshua R. Loftus, Chris Russell, and Ricardo Silva. Counterfactual fairness. In Advances in Neural Information Processing Systems (NIPS), 2017.

[20] Guanpeng Li, Siva Kumar Sastry Hari, Michael Sullivan, Timothy Tsai, Karthik Pattabiraman, Joel Emer, and Stephen W. Keckler. Understanding error propagation in deep learning neural network (DNN) accelerators and applications. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC), 2017.

[21] Xingchao Liu, Xing Han, Na Zhang, and Qiang Liu. Certified monotonic neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

[22] Nils Lukas, Edward Jiang, Xinda Zhang, and Florian Kerschbaum. SoK: how robust is image classification deep neural network watermarking? In IEEE Symposium on Security and Privacy, pages 787–804, 2022.

[23] Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi, Omar Fawzi, and Pascal Frossard. Universal adversarial perturbations. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 86–94, 2017.

[24] Long H. Pham and Jun Sun. Verifying neural networks against backdoor attacks. In Computer Aided Verification (CAV), volume 13371 of Lecture Notes in Computer Science, 2022. arXiv:2205.06992.

[25] Adnan Siraj Rakin, Zhezhi He, and Deliang Fan. Bit-flip attack: crushing neural network with progressive bit search. In IEEE/CVF International Conference on Computer Vision (ICCV), pages 1211–1220, 2019.

[26] Andrei Sabelfeld and Andrew C. Myers. Language-based information-flow security. IEEE Journal on Selected Areas in Communications, 21(1):5–19, 2003.

[27] Marco Sälzer and Martin Lange. Reachability is NP-complete even for the simplest neural networks. In Reachability Problems (RP), volume 13035 of Lecture Notes in Computer Science, pages 149–164, 2021.

[28] Marcus Schaefer, Jean Cardinal, and Tillmann Miltzow. The existential theory of the reals as a complexity class: a compendium. In Courses in Discrete and Computational Geometry, volume 31 of Bolyai Society Mathematical Studies, pages 167–313. Springer, Cham, 2026.

[29] Marcus Schaefer and Daniel Štefankovič. Beyond the existential theory of the reals. Theory of Computing Systems, 68:195–220, 2024.

[30] Abu Sebastian, Manuel Le Gallo, Riduan Khaddam-Aljameh, and Evangelos Eleftheriou. Memory devices and applications for in-memory computing. Nature Nanotechnology, 15:529–544, 2020.

[31] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In IEEE Symposium on Security and Privacy, pages 3–18, 2017.

[32] Aishwarya Sivaraman, Golnoosh Farnadi, Todd Millstein, and Guy Van den Broeck. Counterexample-guided learning of monotonic neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

[33] Mojtaba Soltanalian. Random parameter noise does not make exact ReLU verification easy, 2026. arXiv:2607.14375.

[34] Eduardo D. Sontag. Real addition and the polynomial hierarchy. Information Processing Letters, 20(3):115–120, 1985.

[35] Matthew Sotoudeh and Aditya V. Thakur. Provable repair of deep neural networks. In Proceedings of the 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation (PLDI), pages 588–603, 2021.

[36] Vincent Tjeng, Kai Y. Xiao, and Russ Tedrake. Evaluating robustness of neural networks with mixed integer programming. In International Conference on Learning Representations (ICLR), 2019.

[37] Yu-Lin Tsai, Chia-Yi Hsu, Chia-Mu Yu, and Pin-Yu Chen. Formalizing generalization and adversarial robustness of neural networks to weight perturbations. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

[38] Christopher Umans. The minimum equivalent DNF problem and shortest implicants. In Proceedings of the 39th Annual Symposium on Foundations of Computer Science (FOCS), pages 556–563, 1998.

[39] Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng, and Ben Y. Zhao. Neural Cleanse: identifying and mitigating backdoor attacks in neural networks. In IEEE Symposium on Security and Privacy, pages 707–723, 2019.

[40] Shiqi Wang, Huan Zhang, Kaidi Xu, Xue Lin, Suman Jana, Cho-Jui Hsieh, and J. Zico Kolter. Beta-CROWN: eficient bound propagation with per-neuron split constraints for neural network robustness verification. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

[41] Adrian Wurm. Complexity of reachability problems in neural networks. In Reachability Problems (RP), volume 14235 of Lecture Notes in Computer Science, pages 15–27, 2023.

[42] Adrian Wurm. Complexity of reachability and verification problems in neural networks. Preprint, HAL, 2024. hal-04692383.

[43] Adrian Wurm. Robustness verification in neural networks. In Integration of Constraint Programming, Artificial Intelligence, and Operations Research (CPAIOR), volume 14743 of Lecture Notes in Computer Science, pages 263–278. Springer, 2024.

[44] Adrian Wurm. Checking extracted rules in neural networks. In 2025 International Joint Conference on Neural Networks (IJCNN). IEEE, 2025. Preprint arXiv:2509.16547.