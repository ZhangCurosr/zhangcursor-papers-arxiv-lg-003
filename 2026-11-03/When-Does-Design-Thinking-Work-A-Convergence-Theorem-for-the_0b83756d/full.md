# When Does Design Thinking Work? A Convergence Theorem for the Iterate–Test Core

Under review at Journal of Product Innovation Management (JPIM)

Paweł Kuraś<sup>1,∗</sup>, Adrian Michalski<sup>2</sup>, Alicja Gerka<sup>3</sup>, Patryk Organiściak<sup>3</sup>

<sup>1</sup>Institute of Informatics, Kozminski University, ul. Jagiellońska 57/59, 03-301 Warsaw, Poland

<sup>2</sup>Faculty of Mathematics and Applied Sciences, Rzeszów University of Technology,

al. Powstańców Warszawy 12, 35-959 Rzeszów, Poland

<sup>3</sup>Department of Complex Systems, Rzeszów University of Technology,

al. Powstańców Warszawy 12, 35-959 Rzeszów, Poland

<sup>∗</sup>Corresponding author: pkuras@kozminski.edu.pl

November 3, 2026

## Abstract

Academic Summary: Design thinking is widely taught as an iterative loop of ideation, prototyping, and testing, yet conditions under which such a loop must improve innovation outcomes remain unspecified. Taking problem framing as given, we formalize the iterate–test core as search over a finite prototype space whose utility is learned through noisy evaluations. We prove that the prototype a team believes to be best converges almost surely to a local optimum under five conditions: finite framing, local ideation, non-summable discovery rates, unbiased repeated testing, and retention of tested prototypes. Bounded test bias or criteria drift degrades the guarantee by a bounded amount, and multiple stakeholders yield convergence to a locally Pareto-optimal set. Finite-horizon simulations across three landscape classes (additive valuation, NK rugged landscapes, and clustered modular architectures) show that asymptotic limits are rarely reached in practice. Teams face an optimizer’s curse, overestimating their chosen concept by 4–8% and frequently discarding superior prototypes. Testing every prototype once proves to be the worst allocation; optimal performance requires building one to three prototypes per five tests and repeatedly re-testing leaders. Pre-screening concepts yields greater performance gains than tuning testing policies.

Managerial Summary: While design thinking is celebrated for encouraging rapid prototyping and user feedback, product managers lack clear guidelines on how to budget and manage iterative testing. This study models the prototyping cycle to discover how innovation teams can achieve the best possible product under realistic time and budget constraints. Our findings provide four practical rules for innovation leaders: First, never test every prototype just once. Showing many prototypes to users once is the worst use of a development budget; teams should instead build one to three new concepts for every five tests and allocate the remaining tests to verify leading concepts. Second, discount initial enthusiasm: teams systematically overestimate the value of their favorite concept by 4% to 8% due to selection bias. Third, invest in rough pre-screening of ideas before building physical prototypes, which delivers greater performance gains than sophisticated testing algorithms. Fourth,

confirmation bias toward a favorite design causes little harm as long as it remains smaller than test measurement noise. Finally, managers should schedule at least one deliberate return to the problem definition, as iterative testing alone will not signal when a fundamental reframe is needed.

Keywords: design thinking, design process, prototypes, decision making, computational models, stochastic search

## 1 Introduction

Design thinking asks teams to understand the people they are designing for, frame a problem, generate many ideas, build cheap prototypes and test them with users, and to repeat the last three steps until something works (Hasso Plattner Institute of Design, 2010). The method has spread from product design into management, public policy and health care (Liedtka, 2015; Mintrom & Luetjens, 2016; Altman et al., 2018) without a theory of why it should work. Reviews describe a construct that is fragmented and thinly grounded (Johansson-Sköldberg et al., 2013; Micheli et al., 2019), and in health care, where the evidence has been reviewed systematically, it consists mostly of small studies of fair or poor quality (Altman et al., 2018).

This paper asks a narrow question that can be answered exactly. Under what conditions does the iterate–test core of design thinking provably improve what a team builds, and what does the guarantee leave out? We take the framing of the problem as given and study the loop that follows it: propose variants, build one, test it, revise what is believed about every prototype built so far, and repeat. A prototype is a set of features. Its value is a utility that the team can observe only through noisy tests, and the team keeps a belief about each prototype it has tested. A local optimum is a prototype that no nearby variant beats.

The first part of the paper is a convergence result. Suppose the team entertains finitely many features, generates ideas near what it has built, keeps producing improving ideas at a rate that is not summable, tests without bias and re-tests what it has kept, and discards nothing it has tested. Then, with probability one, the prototype the team believes to be best is eventually a fixed one, and it is a local optimum (Theorem 1). Results of this kind are known for search that retains its best solution and for stochastic search under noise. We state the result for the design loop, where each condition names something a team does and can be weakened to see what is lost: a bias of at most B in the tests leaves the team at a prototype that no variant beats by more than 2B, criteria that drift within a band of half-width D behave like a bias of size D, and when stakeholders disagree the set of non-dominated prototypes converges to a locally Pareto-optimal set.

The second part measures what the theorem leaves open, by simulation on three classes of landscape whose local optima are known exactly. The limit turns out to be remote, so what a team has when it stops depends on how fast the loop climbs. A team’s choice among its own prototypes costs it about as much as its failure to find better ones, and its estimate of the prototype it has chosen is too high. Under a fixed budget of tests, never testing anything twice is the worst policy. A bias in favour of the current favourite does no measurable harm until it is about as large as the noise of a test. We also compare the loop with bandit and local-search policies, and teams that reframe with teams that do not. Section 7 sets out what would have to be measured to learn whether design teams satisfy the conditions.

## 2 Background and related work

The five modes of the Stanford d.school (empathise, define, ideate, prototype, test) (Hasso Plattner Institute of Design, 2010) and the Design Council’s Double Diamond (Design Council, 2007) are two widely taught descriptions, and Brown adds the criteria by which an outcome is judged: what people want, what can be built, and what a business can sustain (Brown, 2008, 2009). Experimentation and iteration recur in accounts of the method (Carlgren et al., 2016; Micheli et al., 2019). Liedtka (2015) proposes that it improves outcomes by mitigating biases of judgement, among them the tendency to seek confirmation for a solution one already favours; designers shown an example solution are known to reproduce its features, flaws included (Jansson & Smith, 1991). Novice teams difer in how they combine need-finding, brainstorming and prototyping (Seidel & Fixson, 2013); designers who built several prototypes before receiving feedback produced better designs than those who refined one at a time (Dow et al., 2010); and groups that generate more ideas do not thereby select better ones (Rietzschel et al., 2006).

Formal theories of design begin with Simon, for whom design is search under bounded rationality that ends in a satisfactory alternative (Simon, 1956, 1996). Later accounts describe what design reasoning operates on: function, behaviour and structure (Gero & Kannengiesser, 2004), the joint expansion of concepts and knowledge (Hatchuel & Weil, 2009), the mapping from requirements to parameters (Suh, 1990), and the coupled tasks that force rework (Smith & Eppinger, 1997). Problem and solution are held to evolve together (Maher & Poon, 1996; Dorst & Cross, 2001). Problem setting is central to professional practice (Schön, 1983), framing is an abductive step (Dorst, 2011), and experienced designers routinely reframe the briefs they are given (Paton & Dorst, 2011). McComb et al. (2015) simulate a design team as agents performing simulated annealing and compare its behaviour with human data. None of these accounts states conditions under which an iterative process must improve its outcome. Our model is the simplest representation in which that question can be posed: design as exploration of a space of alternatives (Woodbury & Burrow, 2006), with the frame held fixed.

Outside design, convergence of search to a local optimum is well understood. Adaptive walks on rugged landscapes end at local optima (Kaufman & Levin, 1987), and in the NK model the optima multiply and the walks shorten as interactions among components increase (Kaufman & Weinberger, 1989; Kaufman, 1993). Organisational search of this kind adapts well locally and misses distant peaks (Levinthal, 1997; Levinthal & March, 1993; March, 1991). A canonical genetic algorithm does not converge to the global optimum unless it always retains the best solution found (Rudolph, 1994). In design a prototype’s value is estimated, and search under noisy evaluation has its own literature: simulated annealing converges under noise that is driven down fast enough (Gutjahr & Pflug, 1996), random search over a discrete set converges when samples grow at a suitable rate (Homem-de-Mello, 2003), evolutionary algorithms re-sample candidates (Jin & Branke, 2005), and in ranking and selection an eficient budget gives most replications to the leading alternative and its close competitors (Chen et al., 2000). Imperfect evaluation changes organisational search and can even help it leave a local peak (Knudsen & Levinthal, 2007). The economics of testing in product development covers the choice of test mode (Thomke, 1998), parallel and sequential testing (Loch et al., 2001), the number of concepts to test (Dahan & Mendelson, 2001) and correlated designs (Erat & Kavadias, 2008).

A team that maintains beliefs about alternatives and chooses what to test next is also solving a problem studied in Bayesian optimisation (Jones et al., 1998; Shahriari et al., 2016), with expected improvement and the knowledge gradient (Frazier et al., 2008, 2009) as standard criteria and regret bounds for smooth objectives (Srinivas et al., 2012). In a finite space with independent beliefs the closer analogy is best-arm identification, where the number of tests needed to separate two alternatives grows, up to logarithmic factors, with the inverse square of the diference between them (Even-Dar et al., 2006; Mannor & Tsitsiklis, 2004; Audibert et al., 2010). These results explain why choosing among prototypes is slow and supply the policies with which we compare the loop (Auer et al., 2002; Russo et al., 2018).

![](images/c7a21b43073a6a9cf7717057c6247a6a54de33e22344188e4e796402f3a2c973.jpg)  
Figure 1: The two levels of the model. Empathising and defining produce a frame, a set of features F and a utility U. Within the frame the team repeats a round of ideation, prototyping and testing, and the theorem concerns this loop. A reframe (dashed) returns the team to the outer level and starts a new framing epoch.

## 3 The model

The model has two levels, the choice of a frame and the loop that runs inside it (Figure 1).

## 3.1 Frames, prototypes and the loop

The outer level is the choice of a frame. Empathising gathers observations about users, and defining commits the team to a point of view, a statement of whose problem is being solved and what would count as solving it. In Dorst’s account this step is abductive: starting from the value it wants to create, the team conjectures both the thing to be made and the working principle by which that thing would deliver the value (Dorst, 2011; Kolko, 2010). We do not model how the conjecture is produced. Its result, for our purposes, is a pair $( F , U )$ : a finite set F of features that solutions may combine, and a utility U that says how good each combination is. A reframe replaces $( F , U )$ by another pair, and a framing epoch is the stretch of the project between two reframes.

The inner level is the loop inside an epoch. A prototype is a subset $p \subseteq F$ , so the space of prototypes is $2 ^ { F }$ . In round k the team proposes ideas, each a variant of a prototype it has already built; it chooses one idea and builds it; and it tests the new prototype and some of the older ones, revises its beliefs, and so decides which prototype is currently its favourite. Consider a team designing a travel mug for a commuter, with a frame that admits four features: a folding handle, a one-touch lid, a spill seal and a temperature display. If ideation changes one feature at a time, the mug with a lid and a seal is a local optimum when adding the handle or the display, or removing the lid or the seal, makes it worse, even if a mug three changes away would be better still. Deciding that the commuter is the wrong user is a reframe.

Table 1: Notation.
<table><tr><td></td><td></td></tr><tr><td> $F , 2 ^ { F }$ </td><td>features admitted by the frame; prototype space</td></tr><tr><td> $U ( p ) , U ^ { * }$ </td><td>utility of prototype  $p ,$  in [0, 1]; its maximum</td></tr><tr><td> $\mathcal { N } ( \boldsymbol { p } )$ </td><td>variants of  $p$  that ideation can reach in one step</td></tr><tr><td> $P _ { k } , P _ { \infty }$ </td><td>prototypes built and tested by round  $k ;$  their eventual set</td></tr><tr><td> $\mathcal { F } _ { k }$ </td><td>history up to the end of round k</td></tr><tr><td> $M _ { k } ^ { p }$ </td><td>belief about after round k  $( \widehat { M } _ { k } ^ { p }$   $p$  under biased tests)</td></tr><tr><td> $p _ { k } ^ { * } , X _ { k }$ </td><td>the favourite,  $\arg \operatorname* { m a x } _ { p \in P _ { k } } M _ { k } ^ { p } .$  and the belief about it</td></tr><tr><td> $\delta _ { k }$ </td><td>lower bound on the chance that round k builds a better variant of the favourite</td></tr><tr><td> $\sigma$ </td><td>standard deviation of the noise of  $\mathrm { a }$  test</td></tr><tr><td> $b _ { k } , B , b$ </td><td>bias of the test in round  $k ;$  its bound; the constant bias used in simulations</td></tr></table>

We write $P _ { k } \subseteq 2 ^ { F }$ for the set of prototypes built and tested by the end of round $k ,$ and $\mathcal { F } _ { k }$ for the history up to that point: the ideas proposed, any rough estimates made of them, the prototypes built, and the tests and their results (Table 1).

## 3.2 Utility, tests and beliefs

Following Brown (2009), a solution is judged by its desirability to users, its technical feasibility and its business viability. In our examples the utility is a weighted sum of the three,

$$
U ( p ) = w _ { \mathrm { d } } U _ { \mathrm { d } } ( p ) + w _ { \mathrm { f } } U _ { \mathrm { f } } ( p ) + w _ { \mathrm { v } } U _ { \mathrm { v } } ( p ) ,\tag{1}
$$

with components in [0, 1] and weights summing to one (Keeney & Raifa, 1976). The theory uses only two properties of $U { : }$ it is bounded, and it does not change during an epoch.

The team does not know U. A test of prototype p in round k returns

$$
y = U ( p ) + b _ { k } + \varepsilon _ { k } ,\tag{2}
$$

where $\varepsilon _ { k }$ has mean zero and variance at most $\sigma ^ { 2 }$ given the history and the choice of what to test, and $b _ { k }$ is a systematic error, possibly depending on the prototype and the history, with $| b _ { k } | \le B$ . User interviews, expert critiques and feasibility estimates are all tests in this sense.

From its test results the team forms a belief $M _ { k } ^ { p }$ about each prototype it has built: an estimate of its utility, such as the average of the results or the posterior mean of a Bayesian model (DeGroot, 2004). The favourite $p _ { k } ^ { * }$ is the prototype with the highest belief, ties being broken by any fixed rule, and

$$
X _ { k } = \operatorname* { m a x } _ { p \in P _ { k } } M _ { k } ^ { p }\tag{3}
$$

is how good the team thinks its favourite is. Two other quantities must be kept apart from $X _ { k }$ The true utility $U ( p _ { k } ^ { * } )$ of the favourite is what the team would deliver if it stopped now, the shipped utility. The best true utility among all tested prototypes is what the team has found,

whether or not it knows it.

An idea that has not been built has no test results, so the choice of which idea to build must rest on something cheaper, such as a rough estimate, an analogy or a critique. We call this choice pre-screening. It afects the speed of the loop but not its limit.

## 4 Convergence within a frame

## 4.1 Five assumptions

Assumption 1 (Finite frame). Within an epoch the feature set F is finite, the utility $U : 2 ^ { F } $ [0, 1] does not change, and no two prototypes have exactly the same utility.

Ties are absent for almost every utility measured on a continuous scale. Finiteness is a real restriction, because it rules out the arrival of features nobody had thought of; if additions stop, the results apply from the last one onward.

Assumption 2 (Local ideation). Each prototype p has a set $\mathcal { N } ( p ) \subseteq 2 ^ { F }$ of variants, and every idea, apart from the ideas a team starts with, is a variant of a prototype already in $P _ { k }$ . A prototype is locally optimal if none of its variants has higher utility.

In the simulations variants difer from p in at most two features. If every prototype were a variant of every other, the limit would be the global optimum, at a price in speed that Section 5.5 measures.

Assumption 3 (Persistent generativity). There are numbers $\delta _ { k } \geq 0 .$ , determined by the history $\mathcal { F } _ { k - 1 }$ , such that whenever the favourite $p _ { k - 1 } ^ { * }$ has a better variant that has not yet been built, round k builds and tests such a variant with conditional probability at least $\delta _ { k }$ given $\mathcal { F } _ { k - 1 }$ , and

$$
\sum _ { k \geq 1 } \delta _ { k } = \infty .\tag{4}
$$

Ideas may become harder to find near a peak: a chance of improvement that falls like $1 / k$ satisfies (4), one that falls like $1 / k ^ { 2 }$ does not. The team need not recognise the better variant when it builds it, nor know that one exists.

Assumption 4 (Unbiased, repeated testing). Tests are unbiased $( B = 0 )$ , every prototype in $P _ { k }$ is tested infinitely often, and the belief about a prototype difers from the average of its test results by an amount that vanishes as its tests accumulate.

The last clause holds for the average itself and for conjugate posterior means under any prior. Beliefs then become correct: the noise accumulated over the tests of a prototype is a martingale with increments of bounded variance, so by the strong law for martingales (Durrett, 2019, Sec. 4.5) the average of its results converges to $U ( p )$ , and $M _ { k } ^ { p } \to U ( p )$ almost surely for every retained prototype, in whatever order the tests are chosen.

Assumption 5 (Retention). No tested prototype is discarded: $P _ { k } \subseteq P _ { k + 1 }$

Retention keeps the best prototype found available for comparison, the role that elitism plays in evolutionary search (Rudolph, 1994).

## 4.2 The theorem

Theorem 1. Let Assumptions 1–5 hold. Then almost surely the favourite is eventually a single prototype $p _ { \infty }$ , this prototype is locally optimal, and $X _ { k } \to U ( p _ { \infty } )$

If, in addition, U is drawn from a prior distribution on utilities with values in [0, 1] and the beliefs $M _ { k } ^ { p } = \mathbb { E } [ U ( p ) \mid { \mathcal { F } } _ { k } ]$ are the posterior means under that prior, then $X _ { k }$ is a bounded submartingale with respect to the prior: the team never expects its belief about its favourite to fall.

Proof. By retention, $P _ { k }$ is a non-decreasing sequence of subsets of the finite set $2 ^ { F }$ , so after some random round $k _ { 0 }$ it equals a fixed set $P _ { \infty }$ . Every member of $P _ { \infty }$ is tested infinitely often, so $M _ { k } ^ { p } \to U ( p )$ for each. Their utilities are distinct, so from some round on the favourite is the member $p _ { \infty }$ with the highest utility, and $X _ { k } \to U ( p _ { \infty } )$

Suppose that, with positive probability, $p _ { \infty }$ has a variant q with $U ( q ) > U ( p _ { \infty } )$ . Such a q cannot belong to $P _ { \infty }$ . On this event the favourite has a better unbuilt variant in every late round, so by Assumption 3 the conditional probabilities of building one sum to infinity. By Lévy’s extension of the Borel–Cantelli lemma (Durrett, 2019, Sec. 4.3) one is then built after round $k _ { 0 }$ , contradicting $P _ { k } = P _ { \infty }$ for $k \geq k _ { 0 }$

For the second statement, each $\boldsymbol { M } _ { k } ^ { p }$ is a martingale with values in [0, 1] and the set over which the maximum in (3) is taken does not shrink, so by the conditional Jensen inequality

$$
\mathbb { E } \big [ X _ { k + 1 } ~ \big | ~ \mathcal { F } _ { k } \big ] \ge \mathbb { E } \Big [ \operatorname* { m a x } _ { p \in { \cal P } _ { k } } M _ { k + 1 } ^ { p } \Big | ~ \mathcal { F } _ { k } \Big ] \ge \operatorname* { m a x } _ { p \in { \cal P } _ { k } } \mathbb { E } \big [ M _ { k + 1 } ^ { p } ~ \big | ~ \mathcal { F } _ { k } \big ] = X _ { k } .
$$

Retention and finiteness make the set of prototypes settle, repeated unbiased testing makes beliefs correct, generativity prevents the process from stopping short of a local optimum, and locality determines which optimum is meant. The first statement holds for every utility without ties and whatever the team’s prior, because it uses only the fact that beliefs become correct. The second is the familiar fact that information has non-negative expected value to a Bayesian (DeGroot, 2004), and it holds only for a team whose prior is right and whose beliefs use everything in the history. A team whose prior is much more difuse than the real spread of utilities overestimates its favourite, because the favourite is chosen for having the highest estimate and a difuse prior does nothing to discount the luck in that estimate. This is the optimiser’s curse (Smith & Winkler, 2006), and the simulated team of Section 5 is of this kind.

## 4.3 Biased tests and drifting criteria

People seek and read evidence in ways that favour what they already believe (Nickerson, 1998), and a team with a favourite is no exception. Suppose the bias in (2) is bounded by B and the team, unaware of it, forms its beliefs $\widehat { M } _ { k } ^ { p }$ as if tests were unbiased.

Proposition 1. Let Assumptions $1 , \ 2$ and 5 hold, let every retained prototype be tested infinitely often with $| b _ { k } | \leq B ,$ let beliefs be averages of test results or conjugate normal posterior means, and let Assumption 3 hold with $\delta _ { k } \geq \delta$ for a constant $\delta > 0$ . Then almost surely, from some round on, every favourite is 2B-locally optimal: none of its variants has a utility exceeding its own by more than 2B.

Proof. A belief is a weighted average of the prior mean and the test results, with the weight of the prior tending to zero. Shifting each result by at most B shifts the average by at most

$B ,$ and the unbiased part converges to $U ( p )$ as before, so lim $\begin{array} { r } { \operatorname* { s u p } _ { k } | \widehat { M } _ { k } ^ { p } - U ( p ) | \leq B } \end{array}$ for every $p \in P _ { \infty }$ . Let u be the highest utility in $P _ { \infty }$ . A member with utility below $u - 2 B$ has beliefs eventually below those of the best member, so every late favourite has utility at least $u - 2 B$ The favourite may keep changing among such members, which is why a constant $\delta$ is assumed. Each time the favourite has a better variant not yet built, one is built with probability at least $\delta ;$ if this happened infinitely often, infinitely many prototypes would be built, which is impossible in a finite space. Eventually, then, no favourite has a better unbuilt variant, and every variant that has been built has utility at most u, which exceeds the favourite’s by at most 2B. □

The bound is a worst case in which the bias always works against the better of two close prototypes. A bias that merely flatters the favourite is harmful only when it is large enough to overturn a comparison (Section 5.5).

The same argument covers criteria that move. Suppose that during an epoch the utility is $U _ { k }$ at round k, for instance because the weights in (1) are renegotiated, and that it stays within a band around a reference utility U<sup>¯</sup> without ties.

Corollary 1. $I f | U _ { k } ( p ) - \bar { U } ( p ) | \leq D$ for all k and $p ,$ then under the conditions of Proposition $^ { 1 , }$ with unbiased tests of the current utility and with “better variant” understood with respect to $\bar { U }$ every late favourite is 2D-locally optimal for $\bar { U }$ and 4D-locally optimal for every $U _ { k }$

Proof. A test of $U _ { k } ( p )$ is a test of $\bar { U } ( p )$ with a bias of at most $D ,$ so Proposition 1 applies with $B = D$ . For a late favourite p and a variant $q , U _ { k } ( q ) - U _ { k } ( p ) \leq \bar { U } ( q ) - \bar { U } ( p ) + 2 D \leq 4 D$ □

The model thus tolerates a problem that shifts within bounds while the solution is developed, which is as far as it goes towards the co-evolution of problem and solution (Dorst & Cross, 2001). A discontinuous change of criteria is a reframe.

## 4.4 Several stakeholders

When the weights in (1) have not been agreed, let each stakeholder s in a finite set have a utility $U ^ { ( s ) }$ , say that $q$ dominates p if it is at least as good for every stakeholder and better for one, and let each test return one noisy observation per stakeholder. In place of a favourite the team keeps an archive: the tested prototypes that no other tested prototype dominates according to current beliefs.

Proposition 2. Let Assumptions 1, 2, 4 and 5 hold for every stakeholder, with no ties in any $U ^ { ( s ) }$ . Suppose that whenever a member of the archive has a dominating variant that has not been built, round k builds such a variant with conditional probability at least $\delta _ { k } ,$ , with $\delta _ { k }$ as in Assumption 3. Then almost surely the archive is eventually constant, it is the set of non-dominated members of $P _ { \infty }$ , and none of its members has a dominating variant.

Proof. As before $P _ { k } = P _ { \infty }$ from some round on and all beliefs converge to the true utilities. With no ties, dominance according to beliefs eventually agrees with true dominance on $P _ { \infty } .$ , so the archive settles on its non-dominated members. If a member $p$ had a dominating variant $q ,$ then $q \notin P _ { \infty }$ , the generativity condition would apply in every late round, and a dominating variant would be built, contradicting $P _ { k } = P _ { \infty }$ □

With several stakeholders the loop does not resolve the disagreement: it delivers a set of mutually incomparable prototypes, each locally unimprovable for everyone at once, as archivebased multi-objective methods do (Laumanns et al., 2002).

## 4.5 How fast?

Almost-sure convergence says nothing about time, which has two components. Discovery is governed by δ: if tests were exact, the expected time to a local optimum would be at most the expected number of improvements on the way divided by $\delta ,$ a bound in the spirit of drift analysis (He & Yao, 2001). Recognition is governed by the noise. Telling apart two prototypes whose utilities difer by $\Delta$ takes a number of tests of the order of $\sigma ^ { 2 } / \Delta ^ { 2 }$ on each (Even-Dar et al., 2006; Mannor & Tsitsiklis, 2004). With $\sigma = 0 . 1$ and a diference of one hundredth of a utility unit, one comparison needs hundreds of tests, so the eficient response is to concentrate tests on the leading alternatives (Chen et al., 2000).

## 4.6 Stopping and reframing

Because the limit is local, more iteration cannot repair a poor frame, and a team must decide when to stop and when to reframe. The value of another round within the frame is at most the expected improvement

$$
\operatorname { E I } _ { k } = \operatorname* { m a x } _ { p } \mathbb { E } \big [ ( U ( p ) - X _ { k } ) ^ { + } \mid \mathcal { F } _ { k } \big ] ,\tag{5}
$$

the criterion of Bayesian optimisation (Jones et al., 1998; Frazier et al., 2008), taken over tested prototypes and unbuilt ideas. If a round costs $c _ { \mathrm { i t e r } }$ in the same units, iterating is worthwhile at most while $\mathrm { E I } _ { k } > c _ { \mathrm { i t e r } }$ . A reframe costs $c _ { \mathrm { r e f } }$ and ofers an uncertain gain $G _ { \mathrm { r e f } }$ over what the team holds, and a one-step comparison suggests reframing when

$$
\mathrm { E I } _ { k } - c _ { \mathrm { i t e r } } < G _ { \mathrm { r e f } } - c _ { \mathrm { r e f } } .\tag{6}
$$

The right side contains the one quantity that cannot be derived from inside the frame. It can be estimated as an expected improvement one level up. If what the team comes to hold in a frame is regarded as a normal draw with mean m and standard deviation $s ,$ learned from earlier projects, then

$$
G _ { \mathrm { r e f } } = \left( m - X _ { k } \right) \Phi ( z ) + s \phi ( z ) , \qquad z = ( m - X _ { k } ) / s ,\tag{7}
$$

with $\Phi$ and $\phi$ the standard normal distribution and density functions. Frames are then the arms of a bandit, and (6) is the greedy policy for it. It is a heuristic, and a lopsided one, since it sets the gain of a single round against that of a whole epoch; optimal search over alternatives has the reservation-price structure of Weitzman (1979). Section 5.7 compares it with simpler rules.

## 5 Simulation study

## 5.1 Setup

A prototype is a subset of 14 features, so there are 16,384 prototypes, few enough to compute the utility of every one and to know the optimum $U ^ { * }$ and all local optima exactly. We use three classes of landscape. The main landscape is an instance of (1) with weights $( w _ { \mathrm { d } } , w _ { \mathrm { f } } , w _ { \mathrm { v } } ) =$ (0.45, 0.25, 0.30): desirability rewards the coverage of user needs with diminishing returns, feasibility falls with build and integration costs, and viability rewards business value and penalises scope. Its optimum is 0.763, its mean utility 0.670 and the standard deviation of utility across prototypes 0.046; it has four local optima for one-feature variants and two for two-feature variants. The DVF family, named after the three criteria, consists of 30 landscapes of the same form with independently drawn parameters (one to six local optima for one-feature variants). The NK family consists of 20 landscapes from Kaufman’s NK model (Kaufman, 1993), in which each feature’s contribution depends on 1, 2, 4 or 6 others; these are far more rugged (1 to 185 local optima, median 38).

A run starts from five random ideas. In each round the team picks one of its three leading prototypes, those with the highest beliefs, and proposes five variants that difer from it in one or two features. Among all the ideas it has not yet built, it builds the one that looks best on a rough estimate and tests it once. It then re-tests its three leading prototypes and one further prototype, taking all prototypes in rotation for this last test so that each is tested again and again, as Assumption 4 requires, though at lengthening intervals; after 80 rounds about a third of the prototypes have been tested only once. A round uses five tests. Tests follow (2) with $\sigma = 0 . 1 0$ and no bias. Beliefs are conjugate normal posterior means under an uninformative prior (mean 0.5, variance 0.08), confined to [0, 1]. The rough estimate used for pre-screening is the true utility plus noise of standard deviation 0.10, drawn once per idea; it is as informative as one test and costs nothing, an advantage that Sections 5.5 and 5.6 remove.

Two choices shape what follows. The noise of a test is about twice the spread of utilities across the landscape, so a single test cannot tell an average prototype from a good one; we regard this as a fair picture of an early test with a handful of users, and Section 5.5 halves and doubles it. A round is a substantial unit of work, so we report results from 5 rounds upward and treat 80 rounds, about 400 tests, as a long project. Utilities are reported as a percentage of $U ^ { * }$ , a diference of one point is one percent of $U ^ { * }$ , and intervals are 95% confidence intervals, computed across landscapes for the two families.

## 5.2 The first 80 rounds

On the main landscape (Figure 2) the shipped utility is 90.8% of the optimum after 5 rounds, 92.1% after 10, 93.3% after 20 and 94.8% after 80. At each of these horizons the team has already tested something better than what it would ship: the best prototype found is worth 92.7%, 94.2%, 95.5% and 97.5%. After 80 rounds the team loses 2.5 points to prototypes it has not found and 2.7 points to its own choice among those it has. On the DVF family the two losses are 1.4 and 2.9 points, and on the rugged NK family, where good prototypes are harder to find, 3.6 and 2.7. The cost of choosing is thus comparable to the cost of not finding on every class, and larger on the smoother landscapes.

The team’s belief about its favourite passes 100% of the optimum at round 18 and ends at 102.4%: the team believes it holds something better than the best prototype that exists. The overestimate after 80 rounds is $0 . 0 5 8 \pm 0 . 0 0 4$ in utility units, 7.6% of the optimum; it is 7.3% on the DVF family and 3.8% on the NK family. This is the optimiser’s curse of Section 4. A prior matched to the main landscape cuts the overestimate to $0 . 0 1 4 \pm 0 . 0 0 2$ and leaves the shipped utility where it was (94.6%): calibration corrects what the team believes about its result without improving the result.

## 5.3 How far is the limit?

We ran the loop for up to 2,560 rounds and checked at each horizon whether the favourite is a local optimum, by looking it up in the enumerated set (Figure 3). On the main landscape it is one in 6 of 120 runs after 80 rounds and in 16 of 120 after 2,560 rounds, about 12,800 tests in a space of 16,384 prototypes. With one-feature variants the share after 2,560 rounds is 19%. On the DVF family the share is 2% after 80 rounds and 15% after 1,280. On the NK family it is 26% and 72%, because four of the ten NK landscapes in this experiment have about twenty or more local optima even for two-feature variants, so that one is never far away.

![](images/645999e9f33f1f3863a3c9fde1fb0a96d84c3a95413777f783c6dbdb50b882cb.jpg)  
Figure 2: The baseline loop on the main landscape (250 runs; bands are 95% confidence intervals). The team’s belief about its favourite exceeds the optimum from round 18 on, while the favourite is worth less than the best prototype the team has tested. A prior calibrated to the landscape (dashed) removes most of the overestimate.

A team that stops after a few dozen rounds has in most cases not converged. The obstacle is the ratio of test noise to the diferences between prototypes, far more than the number of peaks. After 2,560 rounds on the main landscape the best prototype tested is a local optimum in 88% of runs and is worth 99.9% of the optimum, while the favourite is worth 95.8%, so the gap between found and shipped has grown from 2.5 to 4.1 points. Late in a run the team holds thousands of prototypes within a few percent of each other, most of them tested once or twice, and the highest of several thousand noisy estimates rarely belongs to the best. Where prototypes difer more, as on the NK family, the gap closes, from 2.8 points after 80 rounds to 0.6 after 1,280.

One remedy is to discount thin evidence. Shipping the prototype with the highest belief minus one posterior standard deviation gains 0.4 points after 80 rounds on the main landscape and 2.4 after 2,560; after 1,280 rounds it gains 1.2 points on the DVF family and 0.2 on the NK family.

## 5.4 New prototypes or re-tests?

We gave every team exactly five tests in each of 60 rounds and varied how many of the five went to new prototypes, from one every other round to all five. The remaining tests were re-tests of the three leading prototypes, so that the arms difer only in the use of tests (Figure 4; 120 runs on the main landscape and 16 on each landscape of the families, with 5 and with 15 ideas per round).

The curves rise and then fall. Giving all five tests to new prototypes, so that nothing is tested twice, is the worst allocation in 23 of the 24 combinations of landscape class, noise level and

(a) Shipped (solid) and best tested (dashed)  
![](images/fdd5a482f024bf596a26768e9734e2674daf89a74400b07d728f766e5a134b79.jpg)

(b) Local optimality, checked exactly  
![](images/cae73f2ea6f96177632d631b27db40a479401e8c8d2631362e79f11f9797d6f3.jpg)  
Figure 3: Short and long runs (main landscape: 120 runs; DVF and NK families: 10 landscapes each, 24 runs per landscape). (a) Utility shipped (solid) and best utility among tested prototypes (dashed). (b) Share of runs in which the favourite (solid) or the best tested prototype (dashed) is a local optimum, determined exactly by enumeration.

(a) Main landscape  
![](images/1f4f0dfe767a73a90fe8fa41b84af48042cdddfe7c13bd49feb215a3bfa99ff3.jpg)

(b) DVF family  
![](images/c08f8706c40eab2db6cf0929c9e73653e7399c4bfbb616cc6bc520d0c8ab040e.jpg)

(c) NK family  
![](images/33f658a173825b65c39a5bd89f0e77f15099f3fa5f7761c3dcf0eaa6c407bc86.jpg)  
new prototypes per round (out of 5 tests; the rest are re-tests of the three leading prototypes)

Figure 4: Shipped utility after 60 rounds of exactly five tests each, as a function of how many of the five tests go to new prototypes, for four levels of test noise (five ideas per round; error bars are 95% confidence intervals; circles mark the best allocation).

number of ideas. Against the best allocation it costs 1.6 to 2.9 points on the main landscape, 1.4 to 6.5 on the DVF family and 1.3 to 11.5 on the NK family, more as the noise grows. Building a new prototype only every second round is never the best allocation either. The best is one new prototype per round in 15 of the 24 combinations, two in five and three in four, and it moves towards re-testing as tests get noisier: with precise tests $( \sigma = 0 . 0 2 5 )$ two or three new prototypes per round are best, and at the baseline noise and above one is best on every class. Tripling the number of ideas per round changes none of this.

A team should therefore build something new in every round and, unless its tests are very precise, spend most of its tests on prototypes it has tested before. As in the experiment of Dow et al. (2010), the gain comes from having alternatives to compare; the simulations add that the comparison is unreliable until the alternatives have been tested more than once.

## 5.5 Relaxing the assumptions

Table 2 changes one element of the baseline at a time.

Table 2: One change at a time: utility shipped after 80 rounds, as a percentage of $U ^ { * }$ , with 95% confidence half-widths (main landscape: 200 runs; families: 16 runs on each landscape). The baseline uses variants within two features, five ideas per round, full retention, one new prototype, three re-tests of leading prototypes and one rotating re-test per round, and $\sigma = 0 . 1 0$
<table><tr><td>Assumption</td><td>Changed element</td><td>Main</td><td></td><td>DVF family NK family</td></tr><tr><td>Baseline</td><td></td><td> $9 4 . 9 \pm 0 . 3$ </td><td> $9 5 . 7 \pm 0 . 5$ </td><td> $9 3 . 7 \pm 0 . 9$ </td></tr><tr><td rowspan="4"></td><td>Local ideation variants within 1 feature</td><td> $9 4 . 5 \pm 0 . 3$ </td><td> $9 5 . 2 \pm 0 . 4$ </td><td> $9 2 . 5 \pm 1 . 1$ </td></tr><tr><td>variants within 3 features</td><td> $9 4 . 7 \pm 0 . 3$ </td><td> $9 5 . 9 \pm 0 . 4$ </td><td> $9 3 . 3 \pm 0 . 8$ </td></tr><tr><td>unbounded variants</td><td> $9 4 . 2 \pm 0 . 3$ </td><td> $9 4 . 9 \pm 0 . 5$ </td><td> $9 2 . 4 \pm 0 . 8$ </td></tr><tr><td>1 idea per round</td><td> $9 4 . 5 \pm 0 . 3$ </td><td> $9 4 . 4 \pm 0 . 5$ </td><td> $8 9 . 6 \pm 1 . 2$ </td></tr><tr><td rowspan="4"></td><td>2 ideas per round</td><td> $9 4 . 8 \pm 0 . 3$ </td><td> $9 5 . 2 \pm 0 . 4$ </td><td> $9 2 . 0 \pm 0 . 9$ </td></tr><tr><td>15 ideas per round</td><td> $9 4 . 8 \pm 0 . 3$ </td><td> $9 5 . 8 \pm 0 . 4$ </td><td> $9 4 . 0 \pm 0 . 9$ </td></tr><tr><td>ideas built without pre-screening</td><td> $9 4 . 0 \pm 0 . 3$ </td><td> $9 4 . 3 \pm 0 . 6$ </td><td> $8 9 . 0 \pm 1 . 4$ </td></tr><tr><td>only the favourite kept</td><td> $9 3 . 7 \pm 0 . 3$ </td><td> $9 3 . 8 \pm 0 . 6$ </td><td> $9 0 . 1 \pm 1 . 0$ </td></tr><tr><td rowspan="4">Testing</td><td>best 3 kept</td><td> $9 4 . 9 \pm 0 . 2 $ </td><td> $9 5 . 7 \pm 0 . 4$ </td><td> $9 4 . 0 \pm 0 . 9$ </td></tr><tr><td>best 10 kept</td><td> $9 4 . 9 \pm 0 . 3$ </td><td> $9 6 . 0 \pm 0 . 4$ </td><td> $9 3 . 4 \pm 0 . 9$ </td></tr><tr><td>all five tests on new prototypes</td><td> $9 2 . 6 \pm 0 . 4$ </td><td> $9 2 . 0 \pm 0 . 7$ </td><td> $8 5 . 4 \pm 1 . 2$ </td></tr><tr><td>no rotating re-test</td><td> $9 4 . 7 \pm 0 . 3$ </td><td> $9 5 . 7 \pm 0 . 4$ </td><td> $9 3 . 0 \pm 0 . 9$ </td></tr><tr><td rowspan="2"></td><td>noise  $\sigma = 0 . 0 5$ </td><td> $9 5 . 9 \pm 0 . 2$ </td><td> $9 7 . 5 \pm 0 . 4$ </td><td> $9 5 . 4 \pm 1 . 0$ </td></tr><tr><td>noise  $\sigma = 0 . 2 0$ </td><td> $9 3 . 5 \pm 0 . 3$ </td><td> $9 3 . 6 \pm 0 . 5$ </td><td> $8 8 . 6 \pm 1 . 3$ </td></tr></table>

The reach of ideation has little efect within limits. One-feature variants cost 0.4 points on the main landscape and the DVF family and 1.2 on the NK family, three-feature variants change nothing measurable, and removing the bound altogether costs 0.7 to 1.3 points: unbounded ideation reaches the global optimum in the limit, but within 80 rounds it spends tests on distant prototypes that are mostly poor. The supply of ideas is important on rugged landscapes and saturates quickly elsewhere. One idea per round costs 0.4 points on the main landscape and 4.0 on the NK family, and fifteen are no better than five. Which idea is built counts for more: building one of the current round’s ideas at random, in place of the idea that looks best among all those not yet built, costs 0.9, 1.4 and 4.7 points on the three classes. A team that remembers only its current favourite loses 1.2 to 3.6 points, and remembering three prototypes recovers all of it, so the favourite needs rivals to be compared with but not a complete archive.

Testing has the largest efects. Spending all five tests of a round on new prototypes costs 2.3, 3.7 and 8.2 points, although such a team finds prototypes as good as the baseline team does; it cannot tell which they are. Dropping only the rotating re-test, which the theorem needs for beliefs to become correct, costs at most 0.6 points within 80 rounds. Halving the noise of a test is worth 1.0 to 1.8 points and doubling it costs 1.4 to 5.1.

Figure 5 adds a confirmation bias: in each round the re-test of the current favourite is inflated by a constant b. At the baseline noise of 0.10 no loss is detectable on any class up to $b = 0 . 1 0 ;$ at $b = 0 . 2 0$ the loss is 1.0 to 3.5 points and at $b = 0 . 3 0$ it is 3.7 to 8.0. The point at which losses begin moves with the noise. With more precise tests $( \sigma = 0 . 0 5 )$ losses are visible from $b = 0 . 0 7 5$ and reach 1.1 to 2.5 points at $b = 0 . 1 0 .$ , and with noisier tests $( \sigma = 0 . 2 0 )$ nothing is lost up to $b = 0 . 2 0$ . A bias changes the outcome only when it is large enough to reverse comparisons that the evidence would otherwise have decided, which here means a bias about as large as the standard deviation of a test. A team with precise tests is the more exposed to its own bias, because less noise is left in which a challenger can be lucky. The guarantee of Proposition 1 weakens in proportion to the bias, whereas the typical loss has a threshold.

![](images/962a185987bfbd5d502b8856ae204afb7f2bfdf79e51c3a4738f89c0252a6ad4.jpg)  
confirmation bias b added to every re-test of the favourite  
Figure 5: Change in shipped utility after 80 rounds when each round’s re-test of the favourite is inflated by b, at three levels of test noise (main landscape: 240 runs per point; families: 16 runs per landscape; error bars are 95% confidence intervals). The dashed line marks $b = \sigma$

Table 3: Utility shipped after 300 tests, as a percentage of $U ^ { * }$ , on landscapes not used for tuning (main landscape: 250 runs; families: 20 landscapes each, 30 runs per landscape; ± is the 95% confidence half-width). “Dif.” is the mean paired diference from the loop across landscapes.
<table><tr><td rowspan="2">Policy</td><td rowspan="2">Main shipped</td><td colspan="2">DVF family</td><td colspan="2">NK family</td></tr><tr><td>shipped</td><td>diff.</td><td>shipped</td><td>diff.</td></tr><tr><td>Random search</td><td> $9 1 . 0 \pm 0 . 4$ </td><td> $8 7 . 9 \pm 0 . 8$ </td><td> $- 7 . 4 \pm 0 . 8$ </td><td> $8 0 . 2 \pm 0 . 9$ </td><td> $- 1 2 . 0 \pm 0 . 9$ </td></tr><tr><td>Hill-climbing</td><td> $9 1 . 0 \pm 0 . 5$ </td><td> $8 8 . 7 \pm 0 . 8$ </td><td> $- 6 . 6 \pm 0 . 8$ </td><td> $8 3 . 8 \pm 1 . 4$ </td><td> $- 8 . 4 \pm 1 . 6$ </td></tr><tr><td>ε-greedy</td><td> $9 4 . 0 \pm 0 . 3$ </td><td> $9 4 . 3 \pm 0 . 6$ </td><td> $- 1 . 1 \pm 0 . 3$ </td><td> $8 9 . 9 \pm 1 . 1$ </td><td> $- 2 . 3 \pm 0 . 6$ </td></tr><tr><td>UCB</td><td> $9 4 . 5 \pm 0 . 3$ </td><td> $9 4 . 7 \pm 0 . 5$ </td><td> $- 0 . 6 \pm 0 . 3$ </td><td> $9 0 . 2 \pm 1 . 1$ </td><td> $- 2 . 1 \pm 0 . 5$ </td></tr><tr><td>Thompson sampling</td><td> $9 3 . 9 \pm 0 . 3$ </td><td> $9 4 . 4 \pm 0 . 7$ </td><td> $- 0 . 9 \pm 0 . 4$ </td><td> $8 9 . 6 \pm 1 . 1$ </td><td> $- 2 . 7 \pm 0 . 6$ </td></tr><tr><td>UCB with pre-screening</td><td> $9 5 . 0 \pm 0 . 2$ </td><td> $9 6 . 1 \pm 0 . 4$ </td><td> $+ 0 . 8 \pm 0 . 2$ </td><td> $9 3 . 6 \pm 1 . 0$ </td><td> $+ 1 . 4 \pm 0 . 5$ </td></tr><tr><td>Loop without pre-screening</td><td> $9 3 . 8 \pm 0 . 3$ </td><td> $9 3 . 9 \pm 0 . 6$ </td><td> $- 1 . 5 \pm 0 . 5$ </td><td> $8 7 . 5 \pm 1 . 2$ </td><td> $- 4 . 7 \pm 0 . 7$ </td></tr><tr><td>Loop</td><td> $9 4 . 7 \pm 0 . 3$ </td><td> $9 5 . 3 \pm 0 . 4$ </td><td></td><td> $9 2 . 2 \pm 0 . 9$ </td><td></td></tr></table>

## 5.6 Comparison with other policies

We compared the loop with five alternatives given 300 tests, the same variants, the same test noise and the same beliefs (Table 3): random search among variants, hill-climbing over one-feature variants with repeated tests, an ε-greedy policy that either tests a new variant of the favourite or re-tests the favourite, an upper-confidence-bound (UCB) policy (Auer et al., 2002) and Thompson sampling (Russo et al., 2018), the last two adding a new variant of the favourite at regular intervals. Because the loop screens its ideas and these policies do not, we also ran the loop without pre-screening and the UCB policy with it. The parameters of each alternative were tuned on ten DVF and eight NK landscapes, and all policies were scored on landscapes and seeds not used for tuning. The loop ran with its baseline settings.

Random search and hill-climbing are 4 to 12 points behind the loop, because a policy that does not concentrate its tests on promising prototypes wastes most of them. Among the policies that do, the diferences are much smaller, and the loop is not the best. Without pre-screening it is level with ε-greedy and Thompson sampling and 0.7 points behind UCB on the main landscape, 0.4 to 0.9 points behind the three bandit policies on the DVF family and 2.1 to 2.7 points behind them on the NK family. With pre-screening on both sides, UCB leads the loop by 0.3 points on the main landscape, which is within noise, and by 0.8 and 1.4 points on the families, where it wins on nine landscapes in ten. Pre-screening is worth 0.9, 1.5 and 4.7 points to the loop and 0.5, 1.4 and 3.5 points to UCB, which on every class exceeds the diference between the two policies. The loop’s fixed routine is thus a good way to spend a testing budget, and the pre-screening of ideas contributes more than the choice of routine.

Table 4: True utility of the prototype shipped after 150 rounds in a world of six frames, by reframing policy and by the cost of a reframe in rounds (240 paired runs; 95% half-widths are at most 0.004). The last column is the mean number of reframes per run at a cost of 10 rounds; the criterion rows difer in the prior used for $G _ { \mathrm { r e f } } .$ Bold marks the best policy at each cost.
<table><tr><td></td><td colspan="4">Cost of a reframe (rounds)</td><td rowspan="2">Reframes</td></tr><tr><td>Policy</td><td>4</td><td>10</td><td>20</td><td>35</td></tr><tr><td>Never</td><td>0.694</td><td>0.694</td><td>0.694</td><td>0.694</td><td>0</td></tr><tr><td>Every 15 rounds</td><td>0.700</td><td>0.697</td><td>0.693</td><td>0.694</td><td>5.0</td></tr><tr><td>Every 25 rounds</td><td>0.703</td><td>0.698</td><td>0.698</td><td>0.698</td><td>4.0</td></tr><tr><td>Every 40 rounds</td><td>0.702</td><td>0.704</td><td>0.703</td><td>0.698</td><td>2.0</td></tr><tr><td>Once, after 75 rounds</td><td>0.701</td><td>0.702</td><td>0.699</td><td>0.699</td><td>1.0</td></tr><tr><td>Criterion, default</td><td>0.703</td><td>0.701</td><td>0.694</td><td>0.699</td><td>1.2</td></tr><tr><td>Criterion, calibrated</td><td>0.702</td><td>0.702</td><td>0.697</td><td>0.699</td><td>1.2</td></tr><tr><td>Criterion, optimistic</td><td>0.703</td><td>0.701</td><td>0.693</td><td>0.695</td><td>1.9</td></tr></table>

## 5.7 Reframing

A world contains six frames, each a diferent DVF landscape. A team has 150 rounds. It works in one frame at a time with the loop of Section 5.1, without the rotating re-test, and a reframe costs a number of rounds and moves it to a frame it has not visited. When the rounds are spent, the team ships whichever of the favourites of the frames it has visited it believes to be best, and the payof is that prototype’s true utility. We compare a team that never reframes, teams that leave each frame after a fixed number of rounds, and teams that follow criterion (6) after a minimum stay of ten rounds. The criterion uses the team’s beliefs only. A round costs 0.002 in utility units and a reframe that many times its length, and $G _ { \mathrm { r e f } }$ follows (7) under three priors for what a frame yields: a default (mean 0.75, standard deviation 0.05), one calibrated to the beliefs teams hold after 75 rounds in these frames (0.74 and 0.04), and an optimistic one (0.79 and 0.04), each updated when the team leaves a frame. Every policy sees the same 240 worlds at each of four costs.

A single reframe at the midpoint is better than none at every cost, clearly so when a reframe is cheap (Table 4). The paired gain is $0 . 0 0 7 \pm 0 . 0 0 4$ and $0 . 0 0 8 \pm 0 . 0 0 4$ in utility at costs of 4 and 10 rounds and $0 . 0 0 4 \pm 0 . 0 0 4$ and $0 . 0 0 5 \pm 0 . 0 0 4$ at costs of 20 and 35, about one percent of what the team ships. Reframing more often adds little: leaving a frame every 40 rounds is better than the single reframe by 0.001 to 0.005 at costs up to 20 rounds, mostly within the confidence intervals, and leaving every 15 rounds is no better than never reframing once a reframe costs 10 rounds or more.

The criterion does about as well as the single scheduled reframe and no better. With the calibrated prior it difers from it by less than the confidence interval at all four costs, and with the default prior at three of them. In most runs it fires at the earliest permitted moment, so it works as an early exit from frames that look poor against the prior and seldom as a signal that iteration has run dry. The expected improvement of a further round does not decline along an epoch: new, thinly tested rivals keep it at an average of 0.011 both in rounds 10 to 20 and in rounds 40 to 75, above the cost of a round in 86% of all rounds. Most of the value of reframing thus lies in doing it once, and criterion (6) adds nothing to a fixed plan as an operating rule.

## 6 Discussion

## 6.1 When design thinking works

In the limit, the iterate–test loop works when the five assumptions hold, and each of them names a practice: keep the brief fixed while iterating, stay within reach of what has been built, keep producing variants, test in ways that can contradict the team and test favourites again, and keep what has been tried. When these hold, improvement follows from the structure of the process, which is consistent with Liedtka’s proposal that the method works by disciplining judgement (Liedtka, 2015). Table 5 sets the assumptions beside what a departure from each cost in the simulations.

Table 5: The five assumptions of Theorem 1: what each means in practice, what the theory says when it fails, and what a departure in its direction cost after 80 rounds (range over the three landscape classes, in points of U<sup>∗</sup>; from Table 2 and Figure 5).
<table><tr><td>Assumption</td><td>In practice</td><td>When it fails</td><td>Simulated departure and its cost</td></tr><tr><td>Finite frame</td><td>lary of features stay fixed optimum is defined during an epoch</td><td>The brief and the vocabu- No fixed space in which an Not simulated</td><td></td></tr><tr><td>Local ideation</td><td>New ideas are variants of what has been built</td><td>With unbounded jumps the limit is global but progress is slower</td><td>Unbounded variants: -0.7 to -1.3</td></tr><tr><td>Persistent gen- erativity</td><td>The team keeps proposing variants, even late</td><td>The process can stop short One idea per round in of a local optimum</td><td>place of five: -0.4 to -4.0</td></tr><tr><td>Unbiased, re- peated testing</td><td>Tests can contradict the team; favourites are tested again</td><td>on top; under bias B the result is only 2B-locally op- 0.2: -1.0 to -3.5</td><td>A lucky prototype can stay No prototype tested twice: e −2.3 to −8.2. Bias b =</td></tr><tr><td>Retention</td><td>Rivals to the favourite are kept</td><td>timal The best prototype found can be lost</td><td>1 Only the favourite kept: -1.2 to -3.6</td></tr></table>

At the horizon of an actual project the answer shifts. Design thinking is usually presented as a discipline of generating: many ideas, many cheap prototypes, an early move from talking to building (Brown, 2008; Hasso Plattner Institute of Design, 2010). In our simulations generation is only half of the dificulty. At every horizon teams have tested something better than what they would ship, their confidence in what they would ship is inflated, and a prototype tested once is a poor basis for a decision. The pattern that an emphasis on building invites, many diferent prototypes each shown to users once, is the worst allocation we found. Experiments on idea selection point the same way: groups that generate more ideas select no better (Rietzschel et al., 2006), and the quality of a group’s best idea depends among other things on its ability to discern quality (Girotra et al., 2010).

Four recommendations follow. A team should screen its ideas before building, since a rough estimate of which idea is most promising was worth more in our comparisons than the diference between the loop’s fixed routine and a tuned bandit policy. It should treat its estimate of its favourite concept as an overestimate; the optimiser’s curse grows with the number of alternatives chosen among (Smith & Winkler, 2006). It should build something new in every round and give most of its remaining tests to its two or three leading concepts, preferring a well-tested concept to one that looked excellent once. And it should worry about confirmation bias most when its tests are precise. On reframing the simulations support a plain rule: plan for at least one return to the problem definition, and do not expect iteration to signal when it is due.

For tools the implication is that evaluation, more than generation, is the scarce resource. A tool that supported it would keep a belief and an evidence count for every concept tested and shrink estimates that rest on little evidence. The same reasoning applies to generative AI as a source of ideas. In a story-writing experiment, ideas from a language model raised the rated creativity of individual stories and made the stories more alike (Doshi & Hauser, 2024). In the model more ideas per round help only on rugged landscapes, and a language model would be more useful if it could rank ideas roughly before any is built.

## 6.2 Limitations

The model is normative: it describes a team that keeps beliefs and revises them consistently, which real teams do informally at best. The frame is taken as given. The model is silent on how a point of view is formed and on what makes one frame better than another, which is where much of the skill in design lies and where problems are wicked in the sense of Rittel & Webber (1973) and Buchanan (1992). Holding the frame fixed during an epoch also sits uneasily with the view that problem and solution evolve together; Corollary 1 accommodates a problem that shifts within bounds and nothing more.

Within the frame, the simplification with the largest consequences concerns what a test returns. In the model a test returns a number. In practice it also returns reasons: a user test shows why a prototype fails, and the reasons direct the next idea. The model therefore understates what a single test of a new prototype is worth, and the allocation result should be read with that in mind. Independent beliefs work in the same direction, since a test of one prototype teaches something about its relatives; correlated beliefs would speed up recognition and call for the correlated knowledge gradient (Frazier et al., 2009). A prototype is, moreover, a set of features drawn from a fixed list, which leaves out changes of architecture or of a continuous parameter.

The simulations use synthetic landscapes with fourteen features. The qualitative results hold across three classes of very diferent ruggedness, but we have not varied the size of the space, and the magnitudes are not estimates for real projects. The threshold for bias was located at three noise levels only, the reframing experiment uses one class of landscape, and pre-screening, which is free in our simulations, would cost something in practice.

## 7 Testing the model on real teams

In the vocabulary of practice a prototype is a concept, and we use that word here. Any account with noisy evaluation predicts that teams sometimes carry forward a concept that is not their best. Three more specific consequences would distinguish the model. Re-testing should reduce the cost of choosing: the shortfall of the concept a team carries forward, relative to the best concept it tested, should be larger in teams that tested each concept once than in teams that returned to their leading concepts, and a team’s own assessment of its chosen concept should exceed an independent assessment by more when it chose among more candidates. Allocation should have an interior optimum: among teams with similar testing efort, outcomes should be best at an intermediate share of tests spent on new concepts and should fall of sharply as that share approaches one. And bias should have a threshold: leading questions and tests confined to the favourite should be associated with worse outcomes only where the bias they introduce is large relative to the variability of test results. The prediction that teams which reframe do better cannot be tested on observational records, because teams reframe when they are in trouble; it requires an experiment in which a reframe is scheduled for some teams and not for others.

The quantities involved can be obtained from recorded design sessions by protocol analysis (Cross et al., 1996). A session is segmented into moves (Goldschmidt, 2014), each labelled as proposing a concept, building one, testing one, or returning to the problem definition, and for each test the coder records the concept, the direction and strength of the evidence, and markers of likely bias, with agreement between coders reported by a chance-corrected measure (Krippendorf, 2018).

A study by Sung & Kelley (2019) shows both that such coding supports a sequential analysis and where its limits lie. They coded nine think-aloud sessions, in which triads of fourth-grade pupils worked on a paper design task, into seven cognitive strategies (699 coded events, with 85% raw agreement on four sessions coded twice) and analysed the 505 transitions by lag-sequential analysis (Bakeman & Quera, 2011). Generating an idea (their code Designing) was followed by anticipating its consequences (Predicting) more often than chance would give, in 33 of the 147 transitions out of generating (z = 3.66), and predicting was followed by a return to generating in 39 of 52 cases (z = 5.17). This cycle takes place in thought and is close to what we call pre-screening. The pupils built and tested nothing and the sessions were pooled, so the study bears on the feasibility of the measurement and not on the loop of building and testing.

In a confirmatory study the unit of analysis is the framing epoch, because the model’s claims hold within a frame. The outcome must be measured independently of the team’s own beliefs, by judges who rate every concept the team tested without knowing which one it chose; the consensual assessment technique gives a procedure and a reliability check for such ratings (Amabile, 1982), and since the ratings are themselves noisy, several judges are needed for each concept. Comparisons should be made within epochs where possible. For a paired comparison with a standardised diference of 0.3, about one hundred epochs give a power of 0.9 at a one-sided level of 5%.

## 8 Conclusion

Within a fixed framing of a problem the loop of ideation, prototyping and testing converges to a locally optimal prototype when the frame is finite, ideation is local, improving ideas keep coming at a non-summable rate, tests are unbiased and repeated, and tested prototypes are retained. The guarantee weakens in proportion to a bounded bias or drift of the criteria, and with several stakeholders it becomes convergence to a locally Pareto-optimal set.

The simulations show that the limit describes the direction of the process more than the position of a team that stops. Except on very rugged landscapes few runs reach a local optimum, and the result depends as much on how well the team can choose among what it has built as on what it has found. A prototype tested once is a poor basis for a decision, the team’s confidence in its favourite is too high, confirmation bias does harm once it exceeds the noise in the tests, and one reframe captures most of what reframing has to ofer.

Design thinking works, in the sense made precise here, when a team keeps its options, generates near them, screens what it builds, and tests its leading prototypes more than once. Whether teams that do these things obtain better outcomes is a question for field data, and Section 7 says which data would settle it.

## Data availability

The code that produces every figure, table and number in this paper, together with the raw outputs, is openly available in a public repository at https://github.com/zxhouse/ dt-formalisation/. The experiments use fixed random seeds and require only numpy and matplotlib.

## Conflict of interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

The authors thank participants of seminars and workshops for helpful comments and discussions on earlier drafts of this work.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## CRediT authorship contribution statement

Paweł Kuraś: Conceptualization, Methodology, Formal analysis, Software, Writing – original draft. Adrian Michalski: Formal analysis, Validation, Writing – review & editing. Alicja Gerka: Conceptualization, Investigation, Writing – review & editing. Patryk Organiściak: Formal analysis, Validation, Writing – review & editing.

## References

Altman, M., Huang, T. T. K., and Breland, J. Y. "Design thinking in health care." Preventing Chronic Disease 15 (2018): E117. doi:10.5888/pcd15.180128

Amabile, T. M. "Social psychology of creativity: A consensual assessment technique." Journal of Personality and Social Psychology 43, no. 5 (1982): 997–1013. doi:10.1037/0022-3514.43. 5.997

Audibert, J.-Y., Bubeck, S., and Munos, R. "Best arm identification in multi-armed bandits." In Proceedings of the 23rd Conference on Learning Theory (COLT 2010), edited by A. T. Kalai and M. Mohri, pp. 41–53. Omnipress, 2010.

Auer, P., Cesa-Bianchi, N., and Fischer, P. "Finite-time analysis of the multiarmed bandit problem." Machine Learning 47, no. 2–3 (2002): 235–256. doi:10.1023/A:1013689704352

Bakeman, R., and Quera, V. Sequential Analysis and Observational Methods for the Behavioral Sciences. Cambridge University Press, 2011. doi:10.1017/CBO9781139017343

Brown, T. "Design thinking." Harvard Business Review 86, no. 6 (2008): 84–92.

Brown, T. Change by Design: How Design Thinking Transforms Organizations and Inspires Innovation. HarperBusiness, 2009.

Buchanan, R. "Wicked problems in design thinking." Design Issues 8, no. 2 (1992): 5–21. doi:10.2307/1511637

Carlgren, L., Rauth, I., and Elmquist, M. "Framing design thinking: The concept in idea and enactment." Creativity and Innovation Management 25, no. 1 (2016): 38–57. doi:10.1111/ caim.12153

Chen, C.-H., Lin, J., Yücesan, E., and Chick, S. E. "Simulation budget allocation for further enhancing the eficiency of ordinal optimization." Discrete Event Dynamic Systems 10, no. 3 (2000): 251–270. doi:10.1023/A:1008349927281

Cross, N., Christiaans, H., and Dorst, K. (Eds.). Analysing Design Activity. Wiley, 1996.

Dahan, E., and Mendelson, H. "An extreme-value model of concept testing." Management Science 47, no. 1 (2001): 102–116. doi:10.1287/mnsc.47.1.102.10666

DeGroot, M. H. Optimal Statistical Decisions. Wiley-Interscience. (Original work published 1970), 2004.

Design Council. Eleven Lessons: Managing Design in Eleven Global Brands. A Study of the Design Process. Design Council, 2007.

Dorst, K., and Cross, N. "Creativity in the design process: Co-evolution of problem–solution." Design Studies 22, no. 5 (2001): 425–437. doi:10.1016/S0142-694X(01)00009-6

Dorst, K. "The core of ‘design thinking’ and its application." Design Studies 32, no. 6 (2011): 521–532. doi:10.1016/j.destud.2011.07.006

Doshi, A. R., and Hauser, O. P. "Generative AI enhances individual creativity but reduces the collective diversity of novel content." Science Advances 10, no. 28 (2024): eadn5290. doi:10.1126/sciadv.adn5290

Dow, S. P., Glassco, A., Kass, J., Schwarz, M., Schwartz, D. L., and Klemmer, S. R. "Parallel prototyping leads to better design results, more divergence, and increased self-eficacy." ACM Transactions on Computer-Human Interaction 17, no. 4 (2010): Article 18. doi:10.1145/ 1879831.1879836

Durrett, R. Probability: Theory and Examples. (5th ed.). Cambridge University Press, 2019. doi:10.1017/9781108591034

Erat, S., and Kavadias, S. "Sequential testing of product designs: Implications for learning." Management Science 54, no. 5 (2008): 956–968. doi:10.1287/mnsc.1070.0784

Even-Dar, E., Mannor, S., and Mansour, Y. "Action elimination and stopping conditions for the multi-armed bandit and reinforcement learning problems." Journal of Machine Learning Research 7 (2006): 1079–1105.

Frazier, P. I., Powell, W. B., and Dayanik, S. "A knowledge-gradient policy for sequential information collection." SIAM Journal on Control and Optimization 47, no. 5 (2008): 2410– 2439. doi:10.1137/070693424

Frazier, P., Powell, W., and Dayanik, S. "The knowledge-gradient policy for correlated normal beliefs." INFORMS Journal on Computing 21, no. 4 (2009): 599–613. doi:10.1287/ijoc. 1080.0314

Gero, J. S., and Kannengiesser, U. "The situated function–behaviour–structure framework." Design Studies 25, no. 4 (2004): 373–391. doi:10.1016/j.destud.2003.10.010

Girotra, K., Terwiesch, C., and Ulrich, K. T. "Idea generation and the quality of the best idea." Management Science 56, no. 4 (2010): 591–605. doi:10.1287/mnsc.1090.1144

Goldschmidt, G. Linkography: Unfolding the Design Process. MIT Press, 2014. doi:10.7551/ mitpress/9455.001.0001

Gutjahr, W. J., and Pflug, G. Ch. "Simulated annealing for noisy cost functions." Journal of Global Optimization 8, no. 1 (1996): 1–13. doi:10.1007/BF00229298

Hasso Plattner Institute of Design at Stanford. An Introduction to Design Thinking: Process Guide. Stanford University, d.school, 2010.

Hatchuel, A., and Weil, B. "C–K design theory: An advanced formulation." Research in Engineering Design 19, no. 4 (2009): 181–192. doi:10.1007/s00163-008-0043-4

He, J., and Yao, X. "Drift analysis and average time complexity of evolutionary algorithms." Artificial Intelligence 127, no. 1 (2001): 57–85. doi:10.1016/S0004-3702(01)00058-3

Homem-de-Mello, T. "Variable-sample methods for stochastic optimization." ACM Transactions on Modeling and Computer Simulation 13, no. 2 (2003): 108–133. doi:10.1145/858481. 858483

Jansson, D. G., and Smith, S. M. "Design fixation." Design Studies 12, no. 1 (1991): 3–11. doi:10.1016/0142-694X(91)90003-F

Jin, Y., and Branke, J. "Evolutionary optimization in uncertain environments—a survey." IEEE Transactions on Evolutionary Computation 9, no. 3 (2005): 303–317. doi:10.1109/TEVC. 2005.846356

Johansson-Sköldberg, U., Woodilla, J., and Çetinkaya, M. "Design thinking: Past, present and possible futures." Creativity and Innovation Management 22, no. 2 (2013): 121–146. doi:10.1111/caim.12023

Jones, D. R., Schonlau, M., and Welch, W. J. "Eficient global optimization of expensive blackbox functions." Journal of Global Optimization 13, no. 4 (1998): 455–492. doi:10.1023/A: 1008306431147

Kaufman, S., and Levin, S. "Towards a general theory of adaptive walks on rugged landscapes." Journal of Theoretical Biology 128, no. 1 (1987): 11–45. doi:10.1016/S0022-5193(87) 80029-2

Kaufman, S. A., and Weinberger, E. D. "The NK model of rugged fitness landscapes and its application to maturation of the immune response." Journal of Theoretical Biology 141, no. 2 (1989): 211–245. doi:10.1016/S0022-5193(89)80019-0

Kaufman, S. A. The Origins of Order: Self-Organization and Selection in Evolution. Oxford University Press, 1993.

Keeney, R. L., and Raifa, H. Decisions with Multiple Objectives: Preferences and Value Tradeofs. Wiley, 1976.

Knudsen, T., and Levinthal, D. A. "Two faces of search: Alternative generation and alternative evaluation." Organization Science 18, no. 1 (2007): 39–54. doi:10.1287/orsc.1060.0216

Kolko, J. "Abductive thinking and sensemaking: The drivers of design synthesis." Design Issues 26, no. 1 (2010): 15–28. doi:10.1162/desi.2010.26.1.15

Krippendorf, K. Content Analysis: An Introduction to Its Methodology. (4th ed.). SAGE, 2018.

Laumanns, M., Thiele, L., Deb, K., and Zitzler, E. "Combining convergence and diversity in evolutionary multiobjective optimization." Evolutionary Computation 10, no. 3 (2002): 263–282. doi:10.1162/106365602760234108

Levinthal, D. A., and March, J. G. "The myopia of learning." Strategic Management Journal 14, no. S2 (1993): 95–112. doi:10.1002/smj.4250141009

Levinthal, D. A. "Adaptation on rugged landscapes." Management Science 43, no. 7 (1997): 934–950. doi:10.1287/mnsc.43.7.934

Liedtka, J. "Perspective: Linking design thinking with innovation outcomes through cognitive bias reduction." Journal of Product Innovation Management 32, no. 6 (2015): 925–938. doi:10.1111/jpim.12163

Loch, C. H., Terwiesch, C., and Thomke, S. "Parallel and sequential testing of design alternatives." Management Science 47, no. 5 (2001): 663–678. doi:10.1287/mnsc.47.5.663.10480

Maher, M. L., and Poon, J. "Modeling design exploration as co-evolution." Microcomputers in Civil Engineering 11, no. 3 (1996): 195–209. doi:10.1111/j.1467-8667.1996.tb00323.x

Mannor, S., and Tsitsiklis, J. N. "The sample complexity of exploration in the multi-armed bandit problem." Journal of Machine Learning Research 5 (2004): 623–648.

March, J. G. "Exploration and exploitation in organizational learning." Organization Science 2, no. 1 (1991): 71–87. doi:10.1287/orsc.2.1.71

McComb, C., Cagan, J., and Kotovsky, K. "Lifting the veil: Drawing insights about design teams from a cognitively-inspired computational model." Design Studies 40 (2015): 119–142. doi:10.1016/j.destud.2015.06.005

Micheli, P., Wilner, S. J. S., Bhatti, S. H., Mura, M., and Beverland, M. B. "Doing design thinking: Conceptual review, synthesis, and research agenda." Journal of Product Innovation Management 36, no. 2 (2019): 124–148. doi:10.1111/jpim.12466

Mintrom, M., and Luetjens, J. "Design thinking in policymaking processes: Opportunities and challenges." Australian Journal of Public Administration 75, no. 3 (2016): 391–402. doi:10.1111/1467-8500.12211

Nickerson, R. S. "Confirmation bias: A ubiquitous phenomenon in many guises." Review of General Psychology 2, no. 2 (1998): 175–220. doi:10.1037/1089-2680.2.2.175

Paton, B., and Dorst, K. "Briefing and reframing: A situated practice." Design Studies 32, no. 6 (2011): 573–587. doi:10.1016/j.destud.2011.07.002

Rietzschel, E. F., Nijstad, B. A., and Stroebe, W. "Productivity is not enough: A comparison of interactive and nominal brainstorming groups on idea generation and selection." Journal of Experimental Social Psychology 42, no. 2 (2006): 244–251. doi:10.1016/j.jesp.2005.04.005

Rittel, H. W. J., and Webber, M. M. "Dilemmas in a general theory of planning." Policy Sciences 4, no. 2 (1973): 155–169. doi:10.1007/BF01405730

Rudolph, G. "Convergence analysis of canonical genetic algorithms." IEEE Transactions on Neural Networks 5, no. 1 (1994): 96–101. doi:10.1109/72.265964

Russo, D. J., Van Roy, B., Kazerouni, A., Osband, I., and Wen, Z. "A tutorial on Thompson sampling." Foundations and Trends in Machine Learning 11, no. 1 (2018): 1–96. doi:10. 1561/2200000070

Schön, D. A. The Reflective Practitioner: How Professionals Think in Action. Basic Books, 1983.

Seidel, V. P., and Fixson, S. K. "Adopting design thinking in novice multidisciplinary teams: The application and limits of design methods and reflexive practices." Journal of Product Innovation Management 30, no. S1 (2013): 19–33. doi:10.1111/jpim.12061

Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., and de Freitas, N. "Taking the human out of the loop: A review of Bayesian optimization." Proceedings of the IEEE 104, no. 1 (2016): 148–175. doi:10.1109/JPROC.2015.2494218

Simon, H. A. "Rational choice and the structure of the environment." Psychological Review 63, no. 2 (1956): 129–138. doi:10.1037/h0042769

Simon, H. A. The Sciences of the Artificial. (3rd ed.). MIT Press, 1996.

Smith, R. P., and Eppinger, S. D. "Identifying controlling features of engineering design iteration." Management Science 43, no. 3 (1997): 276–293. doi:10.1287/mnsc.43.3.276

Smith, J. E., and Winkler, R. L. "The optimizer’s curse: Skepticism and postdecision surprise in decision analysis." Management Science 52, no. 3 (2006): 311–322. doi:10.1287/mnsc.1050. 0451

Srinivas, N., Krause, A., Kakade, S. M., and Seeger, M. W. "Information-theoretic regret bounds for Gaussian process optimization in the bandit setting." IEEE Transactions on Information Theory 58, no. 5 (2012): 3250–3265. doi:10.1109/TIT.2011.2182033

Suh, N. P. The Principles of Design. Oxford University Press, 1990.

Sung, E., and Kelley, T. R. "Identifying design process patterns: A sequential analysis study of design thinking." International Journal of Technology and Design Education 29, no. 2 (2019): 283–302. doi:10.1007/s10798-018-9448-1

Thomke, S. H. "Managing experimentation in the design of new products." Management Science 44, no. 6 (1998): 743–762. doi:10.1287/mnsc.44.6.743

Weitzman, M. L. "Optimal search for the best alternative." Econometrica 47, no. 3 (1979): 641–654. doi:10.2307/1910412

Woodbury, R. F., and Burrow, A. L. "Whither design space?" Artificial Intelligence for Engineering Design, Analysis and Manufacturing 20, no. 2 (2006): 63–82. doi:10.1017/ S0890060406060057