# Polylogarithmic Nash Regret in Matrix Games with Bandit Feedback

Yuheng Zhang University of Illinois Urbana-Champaign yuhengz2@illinois.edu

## Abstract

We study Nash regret minimization in unknown finite matrix games with bandit payof feedback and observed opponent actions. We develop Optimistic Payof Balancing (OPB), which achieves instance-dependent ${ \mathcal { O } } ( \log ^ { 2 } T )$ Nash regret against arbitrary adaptive opponents, including games with nonunique equilibria. This resolves the open problem posed by Maiti et al. (2025), extending their polylogarithmic guarantee under bandit feedback from $2 \times 2$ games to arbitrary finite dimensions. To handle nonunique equilibria, we construct a reference strategy that leaves room for local adjustments. We order independent payof diferences by estimation accuracy and scale these adjustments by uncertainty, allowing the learner to exploit the opponent’s imbalance to ofset estimation costs. Our result thus shows that observing opponent actions sufices for polylogarithmic Nash regret in general finite matrix games.

## 1 Introduction

We study Nash regret minimization in an unknown two-player zero-sum matrix game against an adaptive opponent. The learner controls one player and, after each interaction, observes the opponent’s action and a noisy payof for the sampled action pair. The value $v ( A )$ of the payof matrix A is the expected reward that an equilibrium strategy guarantees against every opponent. When the matrix is unknown, the learner must balance learning its payofs with earning reward against the opponent. Nash regret measures the cumulative reward shortfall relative to the game value (O’Donoghue et al., 2021; Maiti et al., 2025):

$$
R _ { T } = T v ( A ) - \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } r _ { t } \right] ,
$$

where $r _ { t }$ is the observed reward at round t. A small Nash regret therefore guarantees cumulative reward close to the game’s value throughout learning, even when the opponent follows an arbitrary strategy.

Prior work establishes several routes to low regret under diferent assumptions on feedback and opponent behavior. With bandit payof feedback and observed opponent actions, O’Donoghue et al. (2021) analyze an optimistic matrix-game algorithm with $\widetilde { \mathcal { O } } ( \sqrt { n m T } )$ Nash regret for an $n \times m$ game. In self-play, where both players follow prescribed learning algorithms, Ito et al. (2025) obtain improved instance-dependent regret bounds, including logarithmic regret when the Nash equilibrium is unique and pure. Against arbitrary opponents, Ito et al. (2026) obtain logarithmic regret relative to the best worst-case payof of a pure strategy when opponent actions are observed. This benchmark can be smaller than the Nash value, but the two coincide whenever the row player has a pure optimal strategy, including games with a pure Nash equilibrium. Their result therefore also gives logarithmic Nash regret for this class of games.

Table 1: Nash regret guarantees against arbitrary opponents in unknown finite matrix games. Actions means that the opponent’s realized actions are observed; bandit only reveals only the sampled payof. NE denotes a Nash equilibrium. A pure NE is strict if every unilateral deviation to another pure action strictly lowers the deviating player’s payof. The notation ${ \mathcal { O } } _ { A }$ hides constants depending on the fixed game. The bounds of Ito et al. (2026) use the equality of the pure maximin and Nash values in the listed cases. The entries for Maiti et al. (2025) include the extensions to nonunique equilibria discussed in Appendices D.5 and E.4 and report the fixed-horizon rates. Equilibria need not be unique unless strictness is specified.
<table><tr><td>Work</td><td>Feedback</td><td>Games</td><td>Nash regret</td></tr><tr><td>O&#x27;Donoghue et al. (2021)</td><td>Bandit + actions</td><td> $\mathrm { A n y } \ n \times m$ </td><td> $\widetilde { \mathcal { O } } ( \sqrt { n m T } )$ </td></tr><tr><td>Ito et al. (2026)</td><td>Bandit only</td><td> $n \times m ,$  strict pure NE</td><td> ${ \mathcal { O } } _ { A } ( \log T )$ </td></tr><tr><td>Ito et al. (2026)</td><td>Bandit + actions</td><td> $n \times m ,$  pure row optimum</td><td> ${ \mathcal { O } } _ { A } ( \log T )$ </td></tr><tr><td>Maiti et al. (2025)</td><td>Full matrix + actions</td><td> $\mathrm { A n y } \ n \times m$ </td><td> $\mathcal { O } _ { A } ( \log ^ { 2 } T )$ </td></tr><tr><td>Maiti et al. (2025)</td><td> $\mathrm { B a n d i t } + \mathrm { a c t i o n s }$ </td><td>Any  $2 \times 2$ </td><td> $\mathcal { O } _ { A } ( \log ^ { 2 } T )$ </td></tr><tr><td>This work</td><td> $\mathrm { B a n d i t } + \mathrm { a c t i o n s }$ </td><td> $\mathbf { A n y } \ n \times m$ </td><td> $\mathcal { O } _ { A } ( \log ^ { 2 } T )$ </td></tr></table>

Our work is most closely related to Maiti et al. (2025). They show that both playing an empirical equilibrium and using matrix-game UCB can incur $\Omega ( { \sqrt { T } } )$ Nash regret even on fixed $2 \times 2$ instances. They also develop algorithms achieving instance-dependent polylogarithmic Nash regret, allowing nonunique equilibria. With a noisy observation of the entire matrix at each round, their guarantee applies to arbitrary $n \times m$ games. Under bandit feedback, they establish this guarantee only for $2 \times 2$ games, leaving the extension to arbitrary dimensions open. The obstacle is that the learner controls only its own action: the opponent can withhold some columns or reveal them much less often than others, preventing uniform exploration of the matrix. In higher dimensions, these unequal estimation errors must be handled across several interacting strategy adjustments.

The observation of opponent actions is essential to this question. Without it, an $\Omega ( { \sqrt { T } } )$ lower bound holds even over a fixed finite family of $2 \times 2$ games (Ito et al., 2026, Remark 3). This leaves the following question:

Can a learner achieve instance-dependent polylogarithmic Nash regret in arbitrary finite matrix games with bandit payof feedback and observed opponent actions?

Our results. We answer this question afirmatively, resolving the open problem of Maiti et al. (2025). Our algorithm, Optimistic Payof Balancing (OPB; Algorithm 1), satisfies

$$
R _ { T } \leq C ( A ) \log ^ { 2 } ( e T ) \qquad { \mathrm { f o r ~ e v e r y ~ } } T \geq 1
$$

for every fixed finite payof matrix A, where $C ( A )$ is finite and depends only on the matrix, including its dimensions (Theorem 1). The guarantee holds against adaptive opponents and allows nonunique Nash equilibria for both players. Moreover, OPB requires neither the horizon nor prior knowledge of equilibrium supports or game-dependent separation parameters. Thus, bandit payof feedback with observed actions sufices for polylogarithmic Nash regret in arbitrary finite dimensions. Table 1 compares the settings and guarantees.

Technical ideas. We build on the use of opponent actions to adjust a strategy near an estimated equilibrium (Maiti et al., 2025). The challenge in higher dimensions is to handle unequal estimation errors across columns while allowing nonunique equilibria. We address nonuniqueness by maximizing a joint logarithmic objective over row probabilities and column slacks. Once estimates are accurate enough, the resulting reference gives positive probability to every row used by an optimum and positive slack to every column whose payof can exceed the value at an optimum, leaving room for local adjustments.

Our central technique is to select independent payof diferences in order of estimation accuracy and scale each coordinate’s adjustment range by its uncertainty. We also shrink estimated payof coeficients toward zero using confidence bounds, so a truly zero coeficient never triggers an update. The selection order ensures that repeated visits afecting a coordinate either bring its sample counts to the next reconstruction threshold or create a persistent payof imbalance. The local updates exploit this imbalance to earn surplus reward. Its negative regret contribution absorbs the accumulated estimation cost, giving logarithmic regret per epoch and an overall $\mathcal { O } _ { A } ( \log ^ { 2 } T )$ bound.

## 2 Preliminaries

Notation. For a positive integer k, let $[ k ] = \{ 1 , \ldots , k \}$ and $\Delta _ { k } = \{ p \in \mathbb { R } _ { > 0 } ^ { k } : \sum _ { i = 1 } ^ { k } p _ { i } = 1 \}$ . We write $p _ { i }$ for the ith coordinate of a vector p and $A _ { j }$ for the jth column of a matrix A. All logarithms are natural.

Game and feedback model. We consider a two-player zero-sum game with a fixed unknown payof matrix $A \in [ - 1 , 1 ] ^ { n \times m }$ . The row player has n actions and maximizes reward, while the column player has m actions and minimizes it. The entry $A _ { i j }$ is the expected reward of the row player when the players choose actions i and j. For mixed strategies $p \in \Delta _ { n }$ and $q \in \Delta _ { m } .$ the expected payof is $p ^ { \top } A q$ . A pair $( p ^ { \star } , q ^ { \star } ) \in \Delta _ { n } \times \Delta _ { m }$ is a Nash equilibrium (NE) if neither player can improve its payof by changing its own strategy:

$$
p ^ { \top } A q ^ { \star } \leq ( p ^ { \star } ) ^ { \top } A q ^ { \star } \leq ( p ^ { \star } ) ^ { \top } A q \qquad \mathrm { f o r ~ a l l ~ } p \in \Delta _ { n } , ~ q \in \Delta _ { m } .
$$

The value of the game is

$$
v ( A ) = \operatorname* { m a x } _ { p \in \Delta _ { n } } \operatorname* { m i n } _ { q \in \Delta _ { m } } p ^ { \top } A q = ( p ^ { \star } ) ^ { \top } A q ^ { \star } .
$$

An equilibrium row strategy guarantees expected reward at least $v ( A )$ against every opponent strategy. Nash equilibria need not be unique: either player may have multiple equilibrium strategies.

The learner controls the row player and interacts with an adaptive opponent under bandit payof feedback with observed opponent actions. At round $t ,$ the learner chooses a mixed strategy $p _ { t } \in \Delta _ { r }$ using the past observations. The opponent chooses a mixed strategy $q _ { t } \in \Delta _ { m }$ , possibly using the matrix A, the history, and $p _ { t }$ . Conditionally on this history and the chosen strategies, the actions $I _ { t } \sim p _ { t }$ and $J _ { t } \sim q _ { t }$ are sampled independently. The learner then observes $\left( I _ { t } , J _ { t } , r _ { t } \right)$ where $r _ { t } \in [ - 1 , 1 ]$ is a noisy payof with conditional mean $A _ { I _ { t } J _ { t } }$ . Specifically, writing $\mathcal { F } _ { t } ^ { - }$ for the information available after the mixed strategies are chosen and before the actions are drawn, the reward is generated according to

$$
\mathbb { E } [ r _ { t } \mid \mathcal { F } _ { t } ^ { - } , I _ { t } , J _ { t } ] = A _ { I _ { t } J _ { t } } .
$$

This feedback model allows the opponent to respond to the learner’s mixed strategy and allows the reward noise to depend on the history.

Nash regret. We measure performance by the expected Nash regret

$$
R _ { T } = T v ( A ) - \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } r _ { t } \right] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } ( v ( A ) - p _ { t } ^ { \top } A q _ { t } ) \right] .
$$

The expectation includes the randomness of the learner, the opponent, and the rewards.

Under uninformed bandit feedback, where the learner observes the reward but not the opponent’s action, an $\Omega ( { \sqrt { T } } )$ Nash regret lower bound holds even over a fixed finite family of $2 \times 2$ games (Ito et al., 2026, Remark 3). We therefore ask whether observing the opponent’s action, as in the informed feedback model above, allows instance-dependent polylogarithmic Nash regret.

Our goal is a single algorithm that does not require the matrix A or the time horizon and achieves, for every fixed A,

$$
R _ { T } \leq C ( A ) \mathrm { p o l y l o g } ( e T ) \qquad \mathrm { f o r ~ e v e r y ~ i n t e g e r ~ } T \geq 1 ,
$$

where $C ( A )$ is finite and depends only on the payof matrix, including its dimensions.

## 3 Achieving Polylogarithmic Nash Regret

Section 3.1 develops the algorithm step by step, explaining the challenges that motivate each component. Section 3.2 states the regret guarantee, discusses its implications, and gives a proof sketch.

## 3.1 Algorithm Design

Our algorithm, Optimistic Payof Balancing (OPB), combines payof estimation with local strategy updates. It periodically estimates the game and constructs a reference strategy near its optimal set. Between these reconstructions, it uses each observed opponent action to adjust the strategy. These adjustments correct estimation bias and exploit persistent column imbalance to earn surplus over the game value.

The main challenge is that the opponent controls which columns are sampled, so diferent payof estimates can have very diferent accuracies. Holding an estimated equilibrium fixed can then incur the same estimation bias repeatedly. We address this by expressing both payof uncertainty and strategy adjustments in a common set of payof diference coordinates. A further challenge is that equilibria can be nonunique: we therefore construct an interior reference and select independent payof diferences to define the local updates.

We first describe a run with a planned length H. The learner maintains the number of observations $C _ { i j }$ and empirical mean $\widehat { A } _ { i j }$ of each entry. It keeps each local model fixed during an epoch, while updating its strategy after each observed opponent action. Each reconstruction incorporates the new observations and restarts the local update. We now describe how to construct the model, choose its coordinates, and update the strategy, before assembling the complete procedure and removing the need to know the horizon.

Learning entries through optimistic completion. The learner cannot force the opponent to reveal a particular column. We call an entry acquired once it has been observed at least h times. We therefore assign the upper payof bound 1 to entries with fewer than h observations, and use empirical means for the remaining entries. At the start of an epoch, we form the acquisition mask

$M = \{ ( i , j ) : C _ { i j } \geq h \}$ of acquired entries and set

$$
\widehat { A } _ { i j } ^ { M } = \left\{ \begin{array} { l l } { \widehat { A } _ { i j } , } & { ( i , j ) \in M , } \\ { 1 , } & { ( i , j ) \notin M . } \end{array} \right.
$$

This completion makes an unresolved entry attractive until enough samples are collected. Its cost is controlled directly by those samples: each entry is observed at most h times before its empirical mean takes over, so the total expected payof overstatement is at most 2nmh. We can thus analyze learning in the completed game while paying a finite acquisition cost for the original one.

Let ε be the desired accuracy of acquired entries and let $\tau$ distinguish features of the optimal set from estimation error. We use

$$
\ell = \log ( 6 4 n m ( H + 1 ) ^ { 4 } ) , \quad \varepsilon = \ell ^ { - 1 / 2 } , \quad h = \lceil 2 \ell ^ { 2 } \rceil , \quad \tau = \operatorname* { m i n } \{ \sqrt { \varepsilon } , 1 / ( 2 n ) \} .
$$

An entry with c observations has confidence radius ${ \sqrt { 2 \ell / c } } .$ . Since $\sqrt { 2 \ell / h } \le \varepsilon$ , h observations give accuracy at most $\varepsilon ,$ and the acquisition cost is of order $\ell ^ { 2 }$ . The larger threshold τ separates features that vanish with estimation error from fixed positive features of the game. We reconstruct the model initially and whenever an entry count reaches one of $h , 2 h , 4 h , \ldots$ . Between reconstructions, the confidence widths of acquired entries change by at most a constant factor. All quantities defining the model below are frozen within an epoch.

Finding an interior reference when equilibria are nonunique. For the current mask M, let $A ^ { M }$ be the true completed game, with $A _ { i j } ^ { M } = A _ { i j }$ on M and $A _ { i j } ^ { M } = 1$ elsewhere, and write $v _ { M } = v ( A ^ { M } )$ . The optimal set consists of the strategies $x \in \Delta _ { n }$ satisfying $( A ^ { M } ) ^ { \top } x \ge v _ { M } \mathbf { 1 }$ . We call column $j$ binding at a strategy x if its slack $x ^ { \top } A _ { j } ^ { M } - v _ { M }$ is zero. Two optimal row strategies x, y can satisfy

$$
\begin{array} { r } { x _ { i } = 0 < y _ { i } \qquad \mathrm { o r } \qquad x ^ { \top } A _ { j } ^ { M } = v _ { M } < y ^ { \top } A _ { j } ^ { M } . } \end{array}
$$

In the first case, choosing x excludes a row that another optimum uses. In the second, column $j$ is binding at $x$ but has positive slack at y. An empirical equilibrium can lie near either type of boundary. We therefore seek a reference in the optimal set whose row probability is positive whenever $x _ { i } > 0$ is possible at an optimum, and whose column slack is positive whenever $x ^ { \top } A _ { j } ^ { M } - v _ { M } > 0$ is possible. These positive probabilities and slacks provide room for local adjustments.

We write $\widehat { v } = v ( \widehat { A } ^ { M } )$ and define the near-optimal region

$$
Z = \{ x \in \Delta _ { n } : ( { \widehat { A } } ^ { M } ) ^ { \top } x \geq ( { \widehat { v } } - 2 \varepsilon ) \mathbf { 1 } \} .
$$

When the empirical entry errors satisfy the confidence bounds above, $| \widehat { v } - v _ { M } | \leq \varepsilon$ . Every optimal strategy $x$ of $A ^ { M }$ therefore satisfies

$$
x ^ { \top } \widehat { A } _ { j } ^ { M } \geq x ^ { \top } A _ { j } ^ { M } - \varepsilon \geq v _ { M } - \varepsilon \geq \widehat { v } - 2 \varepsilon \qquad ( j \in [ m ] ) ,
$$

so $Z$ contains the entire true optimal set. We define the nonnegative empirical slack $s _ { j } ( x ) =$ $x ^ { \top } \widehat { A } _ { j } ^ { M } - \widehat { v } + 2 \varepsilon$ on $Z$ and choose the joint center

$$
p ^ { c } \in \underset { x \in Z } { \mathrm { a r g m a x } } \left. \sum _ { i = 1 } ^ { n } \log ( \varepsilon + x _ { i } ) + \sum _ { j = 1 } ^ { m } \log ( \varepsilon + s _ { j } ( x ) ) \right. .\tag{1}
$$

The two sums favor positive row probabilities and positive slacks together. The shift by ε accommodates features that are zero throughout the optimal set. We then select the rows and binding columns indicated by the center:

$$
I = \{ i : p _ { i } ^ { c } > \tau \} , \qquad J = \{ j : s _ { j } ( p ^ { c } ) \leq \tau \} , \qquad { \bar { p } } _ { i } = \frac { p _ { i } ^ { c } } { \sum _ { k \in I } p _ { k } ^ { c } } \quad ( i \in I ) .\tag{2}
$$

Since max<sub>i</sub> $p _ { i } ^ { c } \ge 1 / n > \tau$ , the normalization defining $\bar { p }$ is always well defined. For each fixed game, suficiently accurate estimates identify the union of optimal row supports and the columns binding at every optimum. We refer to these columns as the game’s binding columns, and to the remaining columns as nonbinding. The reference $\bar { p }$ then assigns a fixed positive probability to every retained row and has positive surplus against every nonbinding column. Both properties survive small local adjustments. Removing rows outside I is also important: assigning even a small probability to a strictly suboptimal row can accumulate loss throughout the run.

Selecting independent payof equalities. At an optimal strategy $p ,$ every true binding column gives payof $v _ { M }$ . Thus, for any two such columns $j$ and $j _ { 0 }$

$$
p ^ { \top } A _ { j } ^ { M } = p ^ { \top } A _ { j _ { 0 } } ^ { M } = v _ { M } , \qquad \mathrm { s o } \qquad p ^ { \top } ( A _ { j } ^ { M } - A _ { j _ { 0 } } ^ { M } ) = 0 .
$$

Subtracting the reference column removes the unknown value $v _ { M }$ . The algorithm uses the empirical equations $p ^ { \top } ( \widehat { A } _ { j } ^ { M } - \widehat { A } _ { j _ { 0 } } ^ { M } ) = 0$ for columns in $J .$ Some equations can follow from others, so we select a subset whose left-hand sides can be adjusted independently while preserving total probability. The next step uses these quantities to specify how the strategy should change.

The estimates for diferent columns can have very diferent accuracies, and estimation noise can make redundant equations appear independent. We therefore process columns from most to least accurately estimated. We add a column’s equation only when a confidence test certifies that it is independent of the equations already selected, even after allowing for estimation error.

Let $d = | I | , \widehat { B } = \widehat { A } ^ { M } [ I , : ]$ , and $P _ { 0 } = \mathrm { I d } _ { d } - \mathbf { 1 1 } ^ { \top } / d .$ , the projection onto the subspace $\{ u \in \mathbb { R } ^ { d }$ $\mathbf { 1 } ^ { \top } u = 0 \}$ of changes that preserve total probability. We call the entries $( i , j ) \notin$ M artificial entries: both $\widehat { A } ^ { M }$ and $A ^ { M }$ assign them the fixed value 1. They therefore have no estimation error in the completed game. To reflect this, we define efective counts by

$$
\widetilde { C } _ { i j } = \left\{ \begin{array} { l l } { C _ { i j } , } & { ( i , j ) \in M , } \\ { + \infty , } & { ( i , j ) \not \in M , } \end{array} \right. \quad c _ { j } = \displaystyle \operatorname* { m i n } _ { i \in I } \widetilde { C } _ { i j } .
$$

If J is nonempty, we choose a reference column $j _ { 0 } \in \mathrm { a r g m a x } _ { j \in J } c _ { j }$ and process the other columns of J in decreasing count order, breaking ties by index. Starting with an empty list of columns, we tentatively append each candidate $j$ . For the resulting trial list $j _ { 1 } , \dots , j _ { k }$ , we define

$$
\begin{array} { r } { \widehat { D } ^ { \prime } = [ \widehat { B } _ { j _ { 1 } } - \widehat { B } _ { j _ { 0 } } , \ldots , \widehat { B } _ { j _ { k } } - \widehat { B } _ { j _ { 0 } } ] \in \mathbb { R } ^ { d \times k } . } \end{array}
$$

Let $e _ { l } ^ { \prime } = \sqrt { 2 \ell / c _ { j _ { l } } }$ for $l \in [ k ]$ be the corresponding error scales. If $G ^ { \prime } = P _ { 0 } \hat { D } ^ { \prime }$ has full column rank, we compute

$$
R ^ { \prime } = G ^ { \prime } ( ( G ^ { \prime } ) ^ { \top } G ^ { \prime } ) ^ { - 1 } , \qquad \chi ^ { \prime } = 2 \sum _ { l } e _ { l } ^ { \prime } \| R _ { \cdot l } ^ { \prime } \| _ { 1 } ,\tag{3}
$$

and accept the new equation precisely when $\chi ^ { \prime } \leq 1 / 2$ . We reject trials without full column rank. The identities $( \widehat { D } ^ { \prime } ) ^ { \top } R ^ { \prime } = \mathrm { I d } _ { k }$ and $\mathbf { 1 } ^ { \top } R ^ { \prime } = 0$ show how $R ^ { \prime }$ adjusts the selected payof diferences: adding $R ^ { \prime } u$ to a strategy changes these diferences by $u \in \mathbb { R } ^ { k }$ without changing total probability.

The quantity $\chi ^ { \prime }$ measures how much entry uncertainty is amplified by this adjustment. We call the test $\chi ^ { \prime } \leq 1 / 2$ the independence certificate: it ensures that a perturbation within the confidence bounds cannot destroy independence. It rejects directions created entirely by noise and eventually accepts every truly independent equation. A column with $c _ { j } = \infty$ is the all-ones vector on $I ;$ its reference is also all-ones, so its zero diference is rejected.

We denote the accepted columns by $j _ { 1 } , \dots , j _ { b }$ , where $b \leq d - 1$ , and set

$$
\begin{array} { c } { { \widehat { D } _ { l } = \widehat { B } _ { j _ { l } } - \widehat { B } _ { j _ { 0 } } , \qquad \widehat { D } = [ \widehat { D } _ { 1 } , \ldots , \widehat { D } _ { b } ] , } } \\ { { e _ { l } = \sqrt { 2 \ell / c _ { j _ { l } } } , \qquad E = \mathrm { d i a g } ( e _ { 1 } , \ldots , e _ { b } ) , } } \\ { { \widehat { R } = ( P _ { 0 } \widehat { D } ) ( \widehat { D } ^ { \top } P _ { 0 } \widehat { D } ) ^ { - 1 } . } } \end{array}
$$

The selected errors satisfy $0 < e _ { 1 } \leq \cdot \cdot \cdot \leq e _ { b } \leq \varepsilon$ . If J is empty or no equation is accepted, we set $b = 0$ and play $\bar { p }$ throughout the epoch. The following coordinate operations apply when $b > 0$

Using uncertainty to set the adjustment range. Once the optimal row support and binding columns are identified, a true optimum $p ^ { \star }$ supported on $I ,$ viewed as a vector in $\mathbb { R } ^ { I }$ , satisfies $| \widehat { D } _ { l } ^ { \top } p ^ { \star } | \leq 2 e _ { l }$ , since the corresponding true payof diference is zero. We allow residuals $\widehat { D } _ { l } ^ { \top } p = 4 e _ { l } z _ { l }$ with $z _ { l } \in [ - 1 , 1 ]$ , so the optimum’s normalized residual obeys

$$
| z _ { l } ^ { \star } | = \frac { | \widehat { D } _ { l } ^ { \top } p ^ { \star } | } { 4 e _ { l } } \leq \frac { 1 } { 2 } .
$$

The factor four thus leaves at least $1 / 2$ of the coordinate range on either side of $z _ { l } ^ { \star }$ for responding to the opponent’s play.

Under nonuniqueness, these residuals need not determine a unique strategy. We choose the point nearest to the reference $\bar { p }$ with the prescribed residuals and total mass one. Since $\widehat { D } ^ { \top } \widehat { R } = \mathrm { I d } _ { b }$ and $\mathbf { 1 } ^ { \top } \widehat { R } = 0$ , it is

$$
\begin{array} { r } { \widehat { x } = \bar { p } - \widehat { R } \widehat { D } ^ { \top } \bar { p } , \qquad x ( z ) = \widehat { x } + 4 \widehat { R } E z . } \end{array}
$$

The strategy $x ( z )$ has the prescribed payof diferences and total mass one:

$$
\widehat { D } ^ { \top } x ( z ) = 4 E z , \qquad { \bf 1 } ^ { \top } x ( z ) = 1 .
$$

For suficiently accurate estimates, every $z \in [ - 1 , 1 ] ^ { b }$ also gives $x _ { i } ( z ) > 0$ for all $i \in I ,$ and the payof against every nonbinding column remains above $v _ { M }$ . With less accurate estimates, however, $x ( z )$ may have negative coordinates. We therefore play its Euclidean projection onto

$$
\Delta _ { I } = \{ x \in \mathbb { R } _ { \geq 0 } ^ { I } : \sum _ { i \in I } x _ { i } = 1 \} .
$$

Here and below, $\Pi _ { S }$ denotes Euclidean projection onto $S .$

Shrinking payof coeficients using confidence bounds. The empirical payof against column $j$ is linear in these coordinates:

$$
\boldsymbol { x } ( z ) ^ { \top } \widehat { \boldsymbol { B } } _ { j } = \widehat { \boldsymbol { x } } ^ { \top } \widehat { \boldsymbol { B } } _ { j } + 4 \sum _ { l } e _ { l } \widehat { \alpha } _ { l j } z _ { l } , \qquad \widehat { \alpha } _ { j } = \widehat { R } ^ { \top } \widehat { \boldsymbol { B } } _ { j } .
$$

Thus, increasing $z _ { l }$ by an amount $\Delta z _ { l }$ changes the predicted payof against column $j$ by $4 e _ { l } \widehat { \alpha } _ { l j } \Delta z _ { l }$ An error in $\widehat { \alpha } _ { l j }$ can therefore misdirect the strategy update whenever the opponent plays column $j .$ Since the coeficients are fixed within an epoch, the same error can influence many successive

updates. We use the entry confidence bounds to quantify uncertainty in each coeficient and shrink its estimate toward zero before updating.

For $j \in J$ , we set $\delta _ { j } = \sqrt { 2 \ell / c _ { j } }$ , with $\delta _ { j } = 0$ when $c _ { j } = \infty$ , and define

$$
\begin{array} { c } { \displaystyle \nu _ { l } = | | \widehat { R } _ { \cdot } l | | _ { 1 } , \qquad b _ { l j } = 2 \nu _ { l } \left( \delta _ { j } + 2 \sum _ { k } e _ { k } | \widehat { \alpha } _ { k j } | \right) , } \\ { \displaystyle \widetilde { \alpha } _ { l j } = \mathrm { s g n } ( \widehat { \alpha } _ { l j } ) ( | \widehat { \alpha } _ { l j } | - b _ { l j } ) _ { + } . } \end{array}\tag{4}
$$

Here $( a ) _ { + } = \operatorname* { m a x } \{ a , 0 \}$ and $\operatorname { s g n } ( 0 ) = 0$ . The term $\delta _ { j }$ bounds the estimation error in $\widehat { B } _ { j }$ . Each selected diference $\widehat { D } _ { k }$ also has error at most $2 e _ { k }$ in each entry. In the linear combination $\begin{array} { r } { \sum _ { k } \widehat { \alpha } _ { k j } \widehat { D } _ { k } , } \end{array}$ these errors are multiplied by $| \widehat { \alpha } _ { k j } |$ , giving the bound $\begin{array} { r } { 2 \sum _ { k } e _ { k } | \widehat { \alpha } _ { k j } | } \end{array}$ . The factor $\nu _ { l }$ translates these payof errors into uncertainty in the lth coeficient. To state what this interval estimates, we write $B = A ^ { M } [ I , : ] , D _ { k } = B _ { j _ { k } } - B _ { j _ { 0 } }$ , and $D = [ D _ { 1 } , \dots , D _ { b } ]$ . For suficiently accurate estimates, I consists exactly of the rows used by at least one optimal strategy, and $J$ consists exactly of the columns binding at every optimum. The selected diferences $D _ { 1 } , \ldots , D _ { b }$ then express each column $B _ { j } , j \in J$ as the constant payof $v _ { M } \mathbf { 1 }$ plus a linear combination. The coeficients $\alpha _ { j }$ satisfy

$$
B _ { j } = v _ { M } \mathbf { 1 } + D \alpha _ { j } , \qquad | \widehat { \alpha } _ { l j } - \alpha _ { l j } | \leq b _ { l j } .
$$

We choose $\widetilde { \alpha } _ { l j }$ as the point nearest zero in $\begin{array} { r } { [ \widehat { \alpha } _ { l j } - b _ { l j } , \widehat { \alpha } _ { l j } + b _ { l j } ] . \mathrm { ~ I f ~ } | \widehat { \alpha } _ { l j } | \leq b _ { l j } } \end{array}$ , this gives $\widetilde { \alpha } _ { l j } = 0 ;$ otherwise, we keep the estimated sign and reduce its magnitude by $b _ { l j }$ . In particular, $\alpha _ { l j } = 0$ implies $\widetilde { \alpha } _ { l j } = 0$ . For $j \not \in J $ , we set $\widetilde { \alpha } _ { j } = 0 ;$ ; the local neighborhood already preserves positive surplus against these columns.

At the first round $t _ { 0 }$ of each epoch, we initialize $z _ { t _ { 0 } } = 0$ . After observing the opponent’s action $J _ { t } ,$ , we perform projected gradient ascent in the coordinates z:

$$
p _ { t } = \Pi _ { \Delta _ { I } } \bigl ( \widehat { x } + 4 \widehat { R } E z _ { t } \bigr ) , \qquad z _ { t + 1 } = \Pi _ { [ - 1 , 1 ] ^ { b } } \bigl ( z _ { t } + 4 E \widetilde { \alpha } _ { J _ { t } } \bigr ) .\tag{5}
$$

We extend $p _ { t }$ by zero outside I. This is the standard projected gradient update (Zinkevich, 2003), applied to the conservative linear payof model. We use the observed opponent action $J _ { t }$ to choose $\widetilde { \alpha } _ { J _ { t } }$ for the strategy update from $z _ { t }$ to $z _ { t + 1 }$ . We use the reward $r _ { t }$ to update the empirical mean $\dot { A } _ { I _ { t } J _ { t } }$ of the sampled matrix entry. Both the adjustment range and the conservative gradient thus come from the same entry confidence bounds.

Complete procedure. Algorithm 1 assembles the four steps into epochs. At the start of each epoch, we complete the empirical matrix and compute the reference ${ \bar { p } } ,$ select independent payof equalities, and construct the strategy map and corrected coeficients $\widetilde { \alpha } _ { j }$ . We then keep this model fixed: each observed column $J _ { t }$ determines the update of $z ,$ while the reward $r _ { t }$ updates the sampled entry’s count and empirical mean. When a count reaches $h , 2 h , 4 h , . . . ,$ we rebuild the model and reset $z = 0$ before the next round, keeping all accumulated observations.

To remove the need to know the horizon, we apply the doubling trick to the logarithm of the planned run length:

$$
H _ { 0 } = 2 , \qquad H _ { r + 1 } = H _ { r } ^ { 2 } , \qquad \mathrm { s o } \qquad H _ { r } = 2 ^ { 2 ^ { r } } .
$$

After $H _ { r }$ rounds, we start a fresh run with length $H _ { r + 1 }$ , resetting its counts and empirical means.   
Since log $H _ { r + 1 } = 2 \log H _ { r }$ , the polylogarithmic regret bounds of successive runs sum geometrically.

## Algorithm 1. Optimistic Payof Balancing (OPB)

Input: Numbers of actions $n , m$ . For runs $r = 0 , 1 , . . . ,$ reset counts $C _ { i j }$ and means $\widehat { A } _ { i j }$ to zero and set

$$
H = 2 ^ { 2 ^ { r } } , \quad \ell = \log ( 6 4 n m ( H + 1 ) ^ { 4 } ) , \quad \varepsilon = \ell ^ { - 1 / 2 } , \quad h = \lceil 2 \ell ^ { 2 } \rceil , \quad \tau = \operatorname* { m i n } \{ \sqrt { \varepsilon } , 1 / ( 2 n ) \}
$$

Repeat the following epochs until the run has played H rounds.

1. Construct the reference. Set $M = \{ ( i , j ) : C _ { i j } \geq h \}$ and fill $\widehat { A } ^ { M }$ with $\widehat { A } _ { i j }$ on M and 1 elsewhere. Let vb be its game value, $s _ { j } ( x ) = x ^ { \top } \widehat { A } _ { j } ^ { M } - \widehat { v } + 2 \varepsilon$ , and $Z = \{ x \in \Delta _ { n } : s _ { j } ( x ) \geq 0 \forall j \}$ . Compute

$$
p ^ { c } \in \underset { x \in Z } { \mathrm { a r g m a x } } \left. \sum _ { i } \log ( \varepsilon + x _ { i } ) + \sum _ { j } \log ( \varepsilon + s _ { j } ( x ) ) \right. .
$$

Set $\begin{array} { r } { I = \{ i : p _ { i } ^ { c } > \tau \} , J = \{ j : s _ { j } ( p ^ { c } ) \leq \tau \} , \bar { p } _ { i } = p _ { i } ^ { c } / \sum _ { k \in I } p _ { k } ^ { c } } \end{array}$ for $i \in I .$

2. Select independent equations. Set $d = | I | , \widehat { B } = \widehat { A } ^ { M } [ I , : ] , P _ { 0 } = \mathrm { I d } _ { d } - \mathbf { 1 } \mathbf { 1 } ^ { \top } / d _ { \mathrm { \Omega } }$ , and $c _ { j } = \operatorname* { m i n } \{ C _ { i j } : i \in$ $I , \ ( i , j ) \in M \}$ , with min $\mathcal { D } = \infty$ . For nonempty $^ { J , }$ choose $j _ { 0 } \in \arg \operatorname* { m a x } _ { j \in J } c _ { j }$ and scan the remaining columns in decreasing $c _ { j }$ , breaking ties by index. Starting from an empty list, append each candidate tentatively. For the trial list $j _ { 1 } ^ { \prime } , \ldots , j _ { k } ^ { \prime }$ , form

$$
\widehat { D } ^ { \prime } = [ \widehat { B } _ { j _ { l } ^ { \prime } } - \widehat { B } _ { j _ { 0 } } ] _ { l = 1 } ^ { k } , \qquad G ^ { \prime } = P _ { 0 } \widehat { D } ^ { \prime } .
$$

Reject the candidate if ran $\mathfrak { c } ( G ^ { \prime } ) < k ;$ otherwise compute

$$
R ^ { \prime } = G ^ { \prime } ( ( G ^ { \prime } ) ^ { \top } G ^ { \prime } ) ^ { - 1 } , \qquad \chi ^ { \prime } = 2 \sum _ { l = 1 } ^ { k } \sqrt { 2 \ell / c _ { j _ { l } ^ { \prime } } } \| R _ { \cdot l } ^ { \prime } \| _ { 1 } ,
$$

and accept it if $\chi ^ { \prime } \leq 1 / 2$ . Denote the accepted list by $j _ { 1 } , \dots , j _ { b } ;$ set $b = 0$ if $J = \emptyset$

3. Build the local update $( \mathrm { i f } \ b > 0 )$ . Set $\widehat { D } = [ \widehat { B } _ { j \imath } - \widehat { B } _ { j 0 } ] _ { l = 1 } ^ { b }$ and $e _ { l } = { \sqrt { 2 \ell / c _ { j _ { l } } } } , E = \mathrm { d i a g } ( e _ { 1 } , \dots , e _ { b } )$ Compute

$$
\widehat { R } = ( P _ { 0 } \widehat { D } ) ( \widehat { D } ^ { \top } P _ { 0 } \widehat { D } ) ^ { - 1 } , \qquad \widehat { x } = \bar { p } - \widehat { R } \widehat { D } ^ { \top } \bar { p } .
$$

For $j \in J$ and $l \in [ b ]$ , compute

$$
\begin{array} { l } { \widehat { \alpha } _ { j } = \widehat { R } ^ { \top } \widehat { B } _ { j } , \qquad b _ { l j } = 2 \| \widehat { R } . \iota \| _ { 1 } \left( \sqrt { 2 \ell / c _ { j } } + 2 \displaystyle \sum _ { k } e _ { k } | \widehat { \alpha } _ { k j } | \right) , } \\ { \widetilde { \alpha } _ { l j } = \mathrm { s g n } ( \widehat { \alpha } _ { l j } ) ( | \widehat { \alpha } _ { l j } | - b _ { l j } ) _ { + } . } \end{array}
$$

Set $\widetilde { \alpha } _ { j } = 0$ for $j \not \in J$ and initialize $z = 0$

4. Play until the next epoch. Keep the model fixed. If $b = 0$ , play $\bar { p } ;$ otherwise play $p _ { I } = \Pi _ { \Delta _ { I } } ( \widehat { x } + 4 \widehat { R } E z )$ Set $p _ { i } = 0$ outside I, draw $I _ { t } \sim p ,$ , and observe $J _ { t } , r _ { t }$ . For $b > 0 ,$ , update

$$
z \gets \Pi _ { [ - 1 , 1 ] ^ { b } } ( z + 4 E \widetilde { \alpha } _ { J _ { t } } ) .
$$

Increase $C _ { I _ { t } J _ { t } }$ by one and update its empirical mean using $r _ { t }$ . When this count reaches $h , 2 h , 4 h , . . . ,$ start a new epoch on the next round. End the run after H rounds.

Here $\Delta _ { I }$ is the probability simplex on I, Π is Euclidean projection, $( a ) _ { + } = \operatorname* { m a x } \{ a , 0 \}$ , and $\sqrt { 2 \ell / \infty } = 0 .$

## 3.2 Regret Guarantee

In the following theorem, we prove that OPB achieves polylogarithmic Nash regret for every fixed finite matrix game, including games with nonunique equilibria.

Theorem 1 (Polylogarithmic Nash regret). For every $n , m \geq 1$ and every $A \in [ - 1 , 1 ] ^ { n \times m }$ , there is a finite constant $C ( A )$ such that Algorithm 1 satisfies

$$
{ \cal R } _ { T } \leq C ( A ) \log ^ { 2 } ( e T ) \qquad f o r \ e v e r y \ i n t e g e r \ T \geq 1 .
$$

The same constant applies to every opponent and reward process in the feedback model of Section 2.

Theorem 1 resolves the open problem posed by Maiti et al. (2025): whether instance-dependent polylogarithmic Nash regret is achievable in arbitrary $n \times m$ matrix games under bandit payof feedback with observed opponent actions. Their results establish polylogarithmic Nash regret for general matrices with full-matrix feedback, but cover only $2 \times 2$ games under bandit feedback. OPB closes this gap by achieving $\mathcal { O } _ { A } ( \log ^ { 2 } T )$ Nash regret in arbitrary finite dimensions against adaptive opponents.

The guarantee allows nonunique Nash equilibria for both players. Moreover, OPB needs no prior knowledge of the equilibrium supports or game-dependent separation parameters.

Proof sketch. We analyze a run with planned length H through the true completed games $A ^ { M }$ Optimistic completion costs at most 2nmh in expected regret. On a simultaneous confidence event, for suficiently large H, the center and certificate identify the optimal row supports, binding columns, and independent payof diferences. Every local strategy then assigns positive probability bounded away from zero to each selected row and earns surplus against nonbinding columns.

Fix an epoch prefix and write $B = A ^ { M } [ I , : ]$ . For the true diferences $D _ { l } = B _ { j _ { l } } - B _ { j _ { 0 } }$ , each binding column has the representation $B _ { j } = v ( B ) \mathbf { 1 } + D \alpha _ { j }$ . Let $N _ { l }$ count visits with $J _ { t } \in J$ and $\alpha _ { l J _ { t } } \neq 0$ , and set

$$
d _ { l } = \sum _ { t : J _ { t } \in J } \alpha _ { l J _ { t } } , \qquad V = \sum _ { l } e _ { l } ^ { 2 } N _ { l } , \qquad U = \sum _ { l } e _ { l } | d _ { l } | .
$$

Here V measures estimation cost and U measures the available payof advantage. The count ordering ensures that columns with $c _ { j } > q _ { l } : = c _ { j _ { l } }$ share the coeficient $\alpha _ { l j _ { 0 } }$ , while the remaining binding columns receive only $\mathcal { O } _ { A } ( q _ { l } )$ visits before reconstruction. Thus

$$
N _ { l } \leq K _ { A } ( q _ { l } + | d _ { l } | ) , \qquad V \leq K _ { A } \ell + K _ { A } \varepsilon U ,
$$

where $K _ { A }$ depends only on A and may increase between displays.

For the regret bound, we construct a true optimum $x ( z ^ { * } )$ with $\| z ^ { * } \| _ { \infty } \leq 1 / 2$ and adjust its coordinates to obtain a moving comparison point $u ( t )$ that corrects for the empirical payof diferences. We show that $\| u ( t ) \| _ { \infty } \leq 3 / 4$ and that each coordinate has total variation at most $K _ { A } V$ . The implication $\alpha _ { l j } = 0 \Rightarrow \widetilde { \alpha } _ { l j } = 0$ is essential here: coordinate l changes only on its $N _ { l }$ counted visits. We compare projected gradient ascent with $\begin{array} { r } { u _ { l } ( t ) + \frac { 1 } { 4 } \mathrm { s g n } ( d _ { l } ) \in [ - 1 , 1 ] } \end{array}$ . The shift contributes the advantage U, while comparator variation and coeficient errors cost at most $K _ { A } ( \ell + V )$ . Consequently,

$$
\mathcal { R } _ { \mathrm { e p o c h } } : = \sum _ { t } \mathopen { } \mathclose \bgroup \left( v \mathopen { } \mathclose \bgroup \left( B \aftergroup \egroup \right) - p _ { t } ^ { \top } B _ { J _ { t } } \aftergroup \egroup \right) \leq K _ { A } \ell + K _ { A } V - U \leq K _ { A } \ell + ( K _ { A } \varepsilon - 1 ) U ,
$$

where the sum is over the epoch prefix and $p _ { t }$ is restricted to I. For suficiently large H, $K _ { A } \varepsilon \leq 1$ giving $\mathcal { R } _ { \mathrm { e p o c h } } \leq K _ { A } \ell$ . When $b = 0$ , the epoch deficit is already nonpositive.

There are $\mathcal { O } _ { n , m } ( 1 + \log H )$ epochs. Since $\ell = \mathcal { O } _ { n , m } ( \log ( e H ) )$ and $h = \mathcal { O } ( \ell ^ { 2 } )$ , adding acquisition and confidence failure costs gives $\mathcal { O } _ { A } ( \log ^ { 2 } ( e H ) )$ regret for every run prefix. Smaller horizons are absorbed into the instance-dependent constant. Finally, log $H _ { r } = 2 ^ { r }$ log 2 grows geometrically, so summing the conditional run bounds gives $\mathcal { O } _ { A } ( \log ^ { 2 } ( e T ) )$ regret. Appendix A provides the complete proof.

## 4 Related Work

Regret minimization in matrix games. Regret minimization in repeated games has a long history, from approachability and consistent play to adaptive algorithms and regret matching (Blackwell, 1956; Hannan, 1957; Freund and Schapire, 1999; Foster and Vohra, 1999; Hart and

Mas-Colell, 2000). Under bandit feedback, no-regret algorithms compete with the best fixed action using only the rewards of the actions played (Auer et al., 2002; Neu, 2015). A related direction seeks algorithms that adapt to favorable problem structure while preserving worst-case guarantees, leading to instance-dependent bounds in bandits and self-play games (Zimmert and Seldin, 2021; Ito et al., 2025). For learning against arbitrary opponents, recent work exploits the fixed payof matrix and studies regret relative to the Nash value or the pure maximin value (O’Donoghue et al., 2021; Maiti et al., 2025; Ito et al., 2026). These benchmarks distinguish the payof guaranteed by a mixed strategy from that guaranteed by a pure action. Our result addresses the Nash benchmark in arbitrary finite matrix games; Section 1 and Table 1 compare the feedback models and guarantees.

Equilibrium learning with bandit feedback. Another line of work studies equilibrium learning when players observe only their own actions and realized payofs. Early approaches include reinforcement learning and regret testing, with asymptotic guarantees under diferent equilibrium concepts and game assumptions (Hart and Mas-Colell, 2001; Leslie and Collins, 2005; Foster and Young, 2006; Germano and Lugosi, 2007). Subsequent work develops finite-sample guarantees for learning from payof observations in matrix, Markov, polymatrix, and monotone games (Cai et al., 2023; Chen et al., 2024; Faizal et al., 2024; Dong et al., 2025). One direction studies last-iterate convergence, including achievable rates, computational eficiency, and the role of observed opponent actions (Cai et al., 2025; Fiegel et al., 2025; Maiti et al., 2026; Fiegel et al., 2026; Hait et al., 2026). These works study equilibrium learning when all players follow specified algorithms, whereas we study the cumulative payof of one learner facing an arbitrary opponent.

## 5 Conclusion

We study Nash regret minimization in unknown finite matrix games with bandit payof feedback and observed opponent actions. Our algorithm, OPB, achieves instance-dependent $\mathcal { O } _ { A } ( \log ^ { 2 } T )$ Nash regret against arbitrary adaptive opponents, including games with nonunique Nash equilibria. This resolves the open problem of Maiti et al. (2025) on polylogarithmic Nash regret under bandit feedback in arbitrary dimensions. Our construction jointly centers row probabilities and column slacks to leave room for local strategy adjustments even when equilibria are nonunique. By selecting independent payof diferences in order of estimation accuracy and scaling the adjustments by their uncertainty, OPB exploits imbalances in the opponent’s play to earn surplus reward that absorbs the accumulated estimation cost. Together, these ideas establish that observing opponent actions sufices for polylogarithmic Nash regret in general finite matrix games.

## AI use statement

We use GPT-6 Astra to polish the writing, assist with calculations in the proofs, and check their correctness. We review all AI-assisted content and take full responsibility for the final content of this paper.

## References

Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1):48–77, 2002. doi: 10.1137/ S0097539701398375.

David Blackwell. An analog of the minimax theorem for vector payofs. Pacific Journal of Mathematics, 6(1):1–8, 1956. doi: 10.2140/pjm.1956.6.1.

Yang Cai, Haipeng Luo, Chen-Yu Wei, and Weiqiang Zheng. Uncoupled and convergent learning in two-player zero-sum Markov games with bandit feedback. Advances in Neural Information Processing Systems, 36:36364–36406, 2023.

Yang Cai, Haipeng Luo, Chen-Yu Wei, and Weiqiang Zheng. From average-iterate to last-iterate convergence in games: A reduction and its applications. Advances in Neural Information Processing Systems, 38:46937–46967, 2025.

Zaiwei Chen, Kaiqing Zhang, Eric Mazumdar, Asuman Ozdaglar, and Adam Wierman. Decentralized best-response-based learning in two-player zero-sum stochastic games: A finite-sample analysis. arXiv preprint arXiv:2409.01447, 2024.

Jing Dong, Baoxiang Wang, and Yaoliang Yu. Uncoupled and convergent learning in monotone games under bandit feedback. Advances in Neural Information Processing Systems, 38:151665–151683, 2025.

Fathima Zarin Faizal, Asuman Ozdaglar, and Martin J Wainwright. Finite-sample guarantees for learning dynamics in zero-sum polymatrix games. arXiv preprint arXiv:2407.20128, 2024.

Côme Fiegel, Pierre Menard, Tadashi Kozuno, Michal Valko, and Vianney Perchet. The harder path: Last iterate convergence for uncoupled learning in zero-sum games with bandit feedback. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 17131–17152. PMLR, 2025. URL https://proceedings.mlr. press/v267/fiegel25a.html.

Côme Fiegel, Pierre Menard, Tadashi Kozuno, Michal Valko, and Vianney Perchet. Optimal lastiterate convergence in matrix games with bandit feedback using the log-barrier. arXiv preprint arXiv:2604.15242, 2026.

Dean P. Foster and Rakesh V. Vohra. Regret in the on-line decision problem. Games and Economic Behavior, 29(1–2):7–35, 1999. doi: 10.1006/game.1999.0740.

Dean P. Foster and H. Peyton Young. Regret testing: Learning to play Nash equilibrium without knowing you have an opponent. Theoretical Economics, 1(3):341–367, 2006. URL https://www. econtheory.org/ojs/index.php/te/article/viewArticle/20060341/0.

Yoav Freund and Robert E Schapire. Adaptive game playing using multiplicative weights. Games and Economic Behavior, 29(1-2):79–103, 1999.

Fabrizio Germano and Gábor Lugosi. Global Nash convergence of Foster and Young’s regret testing. Games and Economic Behavior, 60(1):135–154, 2007. doi: 10.1016/j.geb.2006.06.001.

Soumita Hait, Ping Li, Haipeng Luo, and Mengxiao Zhang. Near-optimal last-iterate convergence for zero-sum games with bandit feedback and opponent actions. arXiv preprint arXiv:2605.09363, 2026.

James Hannan. Approximation to Bayes risk in repeated play. In Contributions to the Theory of Games III, volume 39 of Annals of Mathematics Studies, pages 97–139. Princeton University Press, 1957.

Sergiu Hart and Andreu Mas-Colell. A simple adaptive procedure leading to correlated equilibrium. Econometrica, 68(5):1127–1150, 2000. doi: 10.1111/1468-0262.00153.

Sergiu Hart and Andreu Mas-Colell. A reinforcement procedure leading to correlated equilibrium. In Gérard Debreu, Wilhelm Neuefeind, and Walter Trockel, editors, Economics Essays: A Festschrift for Werner Hildenbrand, pages 181–200. Springer, 2001.

Alan J. Hofman. On approximate solutions of systems of linear inequalities. Journal of Research of the National Bureau of Standards, 49(4):263–265, 1952. doi: 10.6028/jres.049.027.

Shinji Ito, Haipeng Luo, Taira Tsuchiya, and Yue Wu. Instance-dependent regret bounds for learning two-player zero-sum games with bandit feedback. In Proceedings of Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 2858–2892. PMLR, 2025. URL https://proceedings.mlr.press/v291/ito25a.html.

Shinji Ito, Haipeng Luo, Arnab Maiti, Taira Tsuchiya, and Yue Wu. Adversarial learning in games with bandit feedback: Logarithmic pure-strategy maximin regret. arXiv preprint arXiv:2602.06348, 2026. URL https://arxiv.org/abs/2602.06348v1.

D. S. Leslie and E. J. Collins. Individual Q-learning in normal form games. SIAM Journal on Control and Optimization, 44(2):495–514, 2005. doi: 10.1137/S0363012903437976.

Arnab Maiti, Kevin Jamieson, and Lillian J. Ratlif. On the limitations and possibilities of Nash regret minimization in zero-sum matrix games under noisy feedback. arXiv preprint arXiv:2306.13233, 2025. URL https://arxiv.org/abs/2306.13233v3.

Arnab Maiti, Claire Jie Zhang, Kevin Jamieson, Jamie Heather Morgenstern, Ioannis Panageas, and Lillian J. Ratlif. Eficient uncoupled learning dynamics with O<sup>˜</sup>(T<sup>−1/4</sup>) last-iterate convergence in bilinear saddle-point problems over convex sets under bandit feedback. In Proceedings of The 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pages 2431–2439. PMLR, 2026.

Gergely Neu. Explore no more: Improved high-probability regret bounds for non-stochastic bandits. Advances in Neural Information Processing Systems, 28, 2015.

Brendan O’Donoghue, Tor Lattimore, and Ian Osband. Matrix games with bandit feedback. In Uncertainty in Artificial Intelligence, pages 279–289. PMLR, 2021.

Julian Zimmert and Yevgeny Seldin. Tsallis-INF: An optimal algorithm for stochastic and adversarial bandits. Journal of Machine Learning Research, 22(28):1–49, 2021. URL https://jmlr.org/papers/ v22/19-753.html.

Martin Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. Technical Report CMU-CS-03-110, Carnegie Mellon University, 2003. URL https://www.cs.cmu. edu/\~maz/publications/techconvex.pdf.

## A Proof of Theorem 1

We first analyze a run with planned length H, conditional on any history at its start. Lemma 2 controls the cost of optimistic completion and gives simultaneous sampling bounds. Lemmas 3 and 4 then show that the estimated model recovers the relevant rows and columns and provides valid coeficient intervals. The main regret argument consists of two complementary bounds: Lemma 5 controls the accumulated estimation cost, while Lemma 6 gives a negative payof contribution that absorbs this cost. We combine these lemmas in the final proof of Theorem 1.

Throughout the appendix, t is the round index within a run, and $\ell , \varepsilon , h , \tau$ are its algorithm parameters. The symbol $K _ { A }$ denotes a finite positive constant depending only on A, which may increase between displays. Constants are uniform over the finitely many completed matrices, row sets, reference columns, and ordered lists of selected columns. Every threshold $H _ { A }$ below is finite and depends only on $A ;$ we enlarge it when necessary.

## A.1 Optimistic completion and sampling bounds

For a mask $M \subseteq [ n ] \times [ m ]$ , recall the true completed matrix

$$
A _ { i j } ^ { M } = \left\{ \begin{array} { l l } { A _ { i j } , } & { ( i , j ) \in M , } \\ { 1 , } & { ( i , j ) \notin M . } \end{array} \right.
$$

Let $C _ { i j } ( t )$ count observations through round t and let $\widehat { A } _ { i j } ( c )$ be the empirical mean of the first c observations of entry (i, j). We also define the cumulative sampling probability

$$
Q _ { i j } ( t ) = \sum _ { u = 1 } ^ { t } p _ { u , i } { \bf 1 } \{ J _ { u } = j \} .
$$

The following lemma separates the cost of entries that have not yet been acquired from the regret in the completed games.

Lemma 2 (Completion cost and sampling bounds). For every prefix $s \leq H$ , let

$$
M _ { t } = \{ ( i , j ) : C _ { i j } ( t - 1 ) \geq h \} , \qquad { \mathcal { D } } _ { s } = \sum _ { t = 1 } ^ { s } \bigl ( v ( A ^ { M _ { t } } ) - p _ { t } ^ { \top } A _ { J _ { t } } ^ { M _ { t } } \bigr ) .
$$

The following statements hold.

(i) The expected Nash regret satisfies

$$
s v ( A ) - \mathbb { E } \sum _ { t = 1 } ^ { s } r _ { t } \leq \mathbb { E } \mathcal { D } _ { s } + 2 n m h .\tag{6}
$$

(ii) There is an event $\mathcal { G }$ with $\mathbb { P } ( \mathcal { G } ^ { c } ) \le n m ( 2 H + 1 ) e ^ { - \ell } \le H ^ { - 2 }$ on which

$$
\begin{array} { r } { | \widehat { A } _ { i j } ( c ) - A _ { i j } | \leq \sqrt { 2 \ell / c } , \qquad C _ { i j } ( t ) \geq \frac { 1 } { 2 } Q _ { i j } ( t ) - \ell } \end{array}\tag{7}
$$

simultaneously for every entry, every positive observation count c attained by round H, and every $t \leq H$

(iii) The number of epochs intersecting any prefix is at most

$$
1 + n m \operatorname* { m a x } \{ 0 , 1 + \left\lfloor \log _ { 2 } ( H / h ) \right\rfloor \} = \mathcal { O } _ { n , m } ( 1 + \log H ) .
$$

Proof. Let $\mathcal { H } _ { t }$ contain the information $\mathcal { F } _ { t } ^ { - }$ from the feedback model and the realized opponent action $J _ { t } .$ , before drawing $I _ { t }$ . Conditional independence of the actions implies

$$
\mathbb { P } ( I _ { t } = i \mid \mathcal { H } _ { t } ) = p _ { t , i } , \qquad \mathbb { E } [ r _ { t } \mid \mathcal { H } _ { t } , I _ { t } ] = A _ { I _ { t } J _ { t } } .
$$

Completion cost. The algorithm rebuilds immediately after an entry reaches h observations, so $M _ { t }$ is the mask used at round t. Since $A ^ { M _ { t } } \geq A$ entrywise, we have $v ( A ^ { M _ { t } } ) \geq v ( A )$ and

$$
v ( A ) - p _ { t } ^ { \top } A _ { J _ { t } } \leq v ( A ^ { M _ { t } } ) - p _ { t } ^ { \top } A _ { J _ { t } } ^ { M _ { t } } + p _ { t } ^ { \top } ( A ^ { M _ { t } } - A ) _ { J _ { t } } .
$$

Conditioning on $\mathcal { H } _ { t }$ and then summing gives

$$
\begin{array} { r l } & { \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { s } p _ { t } ^ { \top } \big ( A ^ { M _ { t } } - A \big ) _ { J _ { t } } = \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { s } ( 1 - A _ { I _ { t } J _ { t } } ) \mathbf { 1 } \big \{ C _ { I _ { t } J _ { t } } ( t - 1 ) < h \big \} } \\ & { \qquad \leq 2 n m h . } \end{array}
$$

Each entry contributes at most h observations, each with cost at most two. The tower property also gives $\mathbb { E } r _ { t } = \mathbb { E } [ p _ { t } ^ { \top } A _ { J _ { t } } ]$ , proving (i). No conditioning on $\mathcal { G }$ is used in this argument.

Payof concentration. For a fixed entry, set

$$
X _ { t } = \mathbf { 1 } \{ I _ { t } = i , J _ { t } = j \} ( r _ { t } - A _ { i j } ) , \qquad Y _ { t } = \mathbf { 1 } \{ I _ { t } = i , J _ { t } = j \} .
$$

Conditional Hoefding’s lemma gives

$$
\mathbb { E } \Big [ \exp \{ \lambda X _ { t } - \lambda ^ { 2 } Y _ { t } / 2 \} \Big | \mathcal { H } _ { t } , I _ { t } \Big ] \leq 1 .
$$

Thus $\begin{array} { r } { \exp ( \lambda \sum _ { u < t } X _ { u } - \lambda ^ { 2 } C _ { i j } ( t ) / 2 ) } \end{array}$ is a nonnegative supermartingale starting at one. We stop it at the cth observation of this entry or at H, whichever comes first. On the event that the cth observation occurs, choosing $\lambda = \sqrt { 2 \ell / c }$ bounds the probability of $\widehat { A } _ { i j } ( c ) - A _ { i j } > \sqrt { 2 \ell / c } \mathrm { b y } e ^ { - \ell }$ The negative choice of λ gives the lower tail. A union bound over entries and $c \in [ H ]$ costs at most $2 n m H e ^ { - \ell }$

Count concentration. For the same entry, $\exp ( Q _ { i j } ( t ) / 2 { - } C _ { i j } ( t ) )$ is a nonnegative supermartingale. Indeed, when $J _ { t } = j$ , its conditional expected multiplier is

$$
e ^ { p _ { t , i } / 2 } ( 1 - p _ { t , i } + p _ { t , i } e ^ { - 1 } ) \leq e ^ { p _ { t , i } ( 1 / 2 - ( 1 - e ^ { - 1 } ) ) } \leq 1 ;
$$

when $J _ { t } \ \ne \ j ,$ , the multiplier is one. Ville’s inequality and a union bound over entries imply $Q _ { i j } ( t ) / 2 - C _ { i j } ( t ) \leq \ell$ for all $i , j ,$ t except on an event of probability at most $n m e ^ { - \ell }$ . Together with the payof bounds, this proves (ii), including at random epoch boundaries.

Number of epochs. After the initial epoch, every reconstruction is triggered by an entry count reaching $h , 2 h , 4 h , \ldots$ . Each entry crosses each such threshold at most once, and no count exceeds H. Counting these thresholds proves (iii). □

## A.2 Identifying the optimal support and binding columns

For a matrix G, let $X ( G ) = \{ x \in \Delta _ { n } : G ^ { \top } x \geq v ( G ) \mathbf { 1 } \}$ be its optimal row set. For a completion $A ^ { M }$ write $v _ { M } = v ( A ^ { M } ) , X ^ { M } = X ( A ^ { M } )$ , and

$$
I _ { * } ^ { M } = \{ i : \operatorname* { m a x } _ { x \in X ^ { M } } x _ { i } > 0 \} , \qquad J _ { * } ^ { M } = \{ j : x ^ { \top } A _ { j } ^ { M } = v _ { M } \mathrm { ~ f o r ~ e v e r y ~ } x \in X ^ { M } \} .
$$

The next lemma shows that the empirical construction identifies these sets and leaves room for the local strategy updates.

Lemma 3 (Support identification and feasibility of local strategies). There exist $H _ { A } < \infty$ and $\beta _ { A } > 0$ such that, on $\mathcal { G }$ , every epoch of a run with $H \geq H _ { A }$ has the following properties.

(i) The selected sets satisfy $I = I _ { * } ^ { M }$ and $J = J _ { * } ^ { M }$ . For $B = A ^ { M } [ I , : ]$ , we have $v ( B ) = v _ { M }$ and an optimal row strategy with all coordinates positive. The zero extension of $\bar { p }$ is within $K _ { A } \varepsilon$ of $X ^ { M }$ in $\ell _ { 1 }$ .

(ii) The selected columns form the greedy independent list for the true projected diferences, in the algorithm’s count order. For $b > 0$ , max<sub>l</sub> $\| \widehat { R } _ { \cdot l } \| _ { 1 } \leq K _ { A }$ and

$$
\chi : = 2 \sum _ { l = 1 } ^ { b } e _ { l } \| \widehat { R } _ { \cdot l } \| _ { 1 } \leq K _ { A } \varepsilon .\tag{8}
$$

(iii) For every $z \in [ - 1 , 1 ] ^ { b }$ , the vector $x ( z ) = { \widehat { x } } + 4 { \widehat { R } } E z$ satisfies

$$
{ \bf 1 } ^ { \top } x ( z ) = 1 , \qquad x _ { i } ( z ) \geq \beta _ { A } ~ ( i \in I ) , \qquad x ( z ) ^ { \top } B _ { j } - v _ { M } \geq \beta _ { A } ~ ( j \notin J ) .
$$

For $b = 0$ , the same conclusions hold with $x ( z )$ replaced by ${ \bar { p } } .$ In particular, the simplex projection in (5) is inactive.

Proof. We establish the distance estimate used to identify $I , J ,$ then analyze the selected diferences and the resulting strategies.

Distance to the optimal set. For each fixed $G \in [ - 1 , 1 ] ^ { n \times m }$ , there is a finite $L _ { G }$ such that

$$
\operatorname* { i n f } _ { y \in X ( G ) } \| x - y \| _ { 1 } \leq L _ { G } \big ( v ( G ) - \operatorname* { m i n } _ { j } x ^ { \top } G _ { j } \big ) _ { + } \qquad ( x \in \Delta _ { n } ) .\tag{9}
$$

Here is a direct finite-polytope proof of this form of Hofman’s bound (Hofman, 1952). Consider

$$
{ \mathcal { P } } _ { G } = \{ ( x , a ) : x \in \Delta _ { n } , \ 0 \leq a \leq 2 , \ G ^ { \top } x + a \mathbf { 1 } \geq v ( G ) \mathbf { 1 } \} .
$$

Let $\eta _ { G } > 0$ be the smallest positive last coordinate among its vertices. Such a coordinate exists because the face $a = 2$ is nonempty. In a vertex decomposition of $( x , a ) \in { \mathcal { P } } _ { G }$ , the total weight of vertices with positive last coordinate is at most $a / \eta _ { G }$ . Replacing their row coordinates by any fixed point of $X ( G )$ gives a point of $X ( G )$ within $2 a / \eta _ { G }$ of x in $\ell _ { 1 }$ . Taking $\begin{array} { r } { a = ( v ( G ) - \operatorname* { m i n } _ { j } x ^ { \top } G _ { j } ) _ { + } } \end{array}$ proves (9).

On G, each acquired entry has error at most $\sqrt { 2 \ell / h } \le \varepsilon$ , and each artificial entry agrees exactly with $A ^ { M }$ . The value is Lipschitz in the largest entrywise error, so

$$
\left| \widehat { \boldsymbol { v } } - \boldsymbol { v } _ { M } \right| \leq \varepsilon , \qquad X ^ { M } \subseteq Z \subseteq \left\{ x \in \Delta _ { n } : ( A ^ { M } ) ^ { \top } { \boldsymbol { x } } \geq ( v _ { M } - 4 \varepsilon ) \mathbf { 1 } \right\} .
$$

Equation (9) now places every point in Z within $K _ { A } \varepsilon$ of $X ^ { M }$ .

Identification by the joint center. We use a property of the logarithmic objective. For nonnegative afine functions $f _ { 1 } , \ldots , f _ { a }$ on a compact polytope $P ,$ , a maximizer $x ^ { c }$ of $\begin{array} { r } { \sum _ { r } \log ( \varepsilon + f _ { r } ( x ) ) } \end{array}$ satisfies

$$
\sum _ { r = 1 } ^ { a } { \frac { \varepsilon + f _ { r } ( w ) } { \varepsilon + f _ { r } ( x ^ { c } ) } } \leq a \quad ( w \in P ) , \qquad \varepsilon + f _ { r } ( x ^ { c } ) \geq { \frac { \varepsilon + \operatorname* { m a x } _ { w \in P } f _ { r } ( w ) } { a } } .\tag{10}
$$

The first inequality follows from the nonpositive directional derivative at $x ^ { c }$ toward $w$ . Keeping one positive summand and maximizing its numerator gives the second inequality.

We apply (10) to the $n + m$ features in (1). For every $x \in \Delta _ { n }$ ，

$$
0 \leq s _ { j } ( x ) - ( x ^ { \top } A _ { j } ^ { M } - v _ { M } ) \leq 4 \varepsilon .
$$

If a row probability or true column slack vanishes throughout $X ^ { M }$ , its empirical feature is at most $K _ { A } \varepsilon$ on $Z .$ . Every other feature has a strictly positive maximum on $X ^ { M }$ ; since $X ^ { M } \subseteq Z .$ (10) bounds its value at $p ^ { c }$ below by a positive constant for suficiently small $\varepsilon$ . Because $\varepsilon / \tau \to 0$ and $\tau  0$ as $H \to \infty$ , the tests in (2) identify $I _ { * } ^ { M }$ and $J _ { * } ^ { M }$

Averaging finitely many optimal strategies that witness the positive features yields an optimum with positive probabilities on all rows in $I _ { * } ^ { M }$ and positive slack at every column outside $J _ { * } ^ { M }$ . In particular $J _ { * } ^ { M } \neq \emptyset$ , since otherwise this optimum would beat $v _ { M }$ against every column. The center assigns only ${ \mathcal { O } } _ { A } ( \varepsilon )$ total mass to rows outside $I _ { * } ^ { M }$ . Removing mass a and renormalizing changes a distribution by 2a in $\ell _ { 1 }$ , so $\bar { p }$ remains within $K _ { A } \varepsilon$ of $X ^ { M }$ and keeps positive row probabilities and nonbinding slacks bounded away from zero. Restricting to $I _ { * } ^ { M }$ preserves the value and admits an optimum with all coordinates positive. This proves (i).

Recovery of independent diferences. For the test in (3), let $D ^ { \prime }$ be the true counterpart of the trial matrix ${ \widehat { D } } ^ { \prime }$ . If a trial includes a column with $c _ { j } = \infty ,$ its entries on $I$ are all artificial. The reference also has infinite count, so this candidate’s diference is zero and the trial is rejected. We may therefore restrict attention to positive trial error scales $e _ { l } ^ { \prime } ;$ , with diagonal matrix $E ^ { \prime }$

When $G ^ { \prime } = P _ { 0 } \hat { D } ^ { \prime }$ has full rank, let $F ^ { \prime } = ( D ^ { \prime } ) ^ { \top } R ^ { \prime } - \mathrm { I d }$ . The reference has at least as many efective observations as each candidate, so

$$
| F _ { l k } ^ { \prime } | \le 2 e _ { l } ^ { \prime } \| R _ { \cdot k } ^ { \prime } \| _ { 1 } , \qquad \| ( E ^ { \prime } ) ^ { - 1 } F ^ { \prime } E ^ { \prime } \| _ { \infty } \le \chi ^ { \prime } .
$$

The matrix infinity norm here is the maximum absolute row sum. If the trial is accepted, $\chi ^ { \prime } \leq 1 / 2$ hence $\mathrm { I d } { + } F ^ { \prime }$ is invertible. Since the columns of $R ^ { \prime }$ have zero sum, $( D ^ { \prime } ) ^ { \top } R ^ { \prime } = ( P _ { 0 } D ^ { \prime } ) ^ { \top } R ^ { \prime }$ ; invertibility implies that the true projected diferences are independent.

Conversely, there are only finitely many true trial matrices and column orders. Their independent projected matrices have a positive minimum nonzero singular value. For suficiently small $\varepsilon ,$ each such empirical trial remains independent, its map $R ^ { \prime }$ is bounded, and its certificate is $\mathcal { O } _ { A } ( \varepsilon ) \leq 1 / 2$ Induction through the ordered candidates thus recovers the true greedy list. The same boundedness gives (8), proving (ii).

Feasibility of local strategies. Every selected true diference has zero payof at every optimum. Part (i) and the entry confidence bounds consequently give $\| \widehat { D } ^ { \top } \bar { p } \| _ { \infty } \leq K _ { A } \varepsilon$ . Part (ii) implies

$$
\| \ b { \hat { x } } - \ b { \bar { p } } \| _ { 1 } \leq K _ { A } \varepsilon , \qquad \operatorname* { s u p } _ { z \in [ - 1 , 1 ] ^ { b } } \| 4 \hat { R } E z \| _ { 1 } \leq K _ { A } \varepsilon .
$$

Also $\mathbf { 1 } ^ { \top } \widehat { R } = 0$ , so every $x ( z )$ has total mass one. The positive probabilities and nonbinding slacks of $\bar { p }$ therefore remain bounded below by a common $\beta _ { A } > 0$ throughout this set. The same properties already hold for $\bar { p }$ when $b = 0$ . This proves (iii). Finiteness of the family of completions allows one choice of $H _ { A } , \beta _ { A }$ for all epochs. □

## A.3 Confidence bounds for the true coeficients

We next identify the coeficients estimated by $\widehat { \alpha } _ { j } = \widehat { R } ^ { \intercal } \widehat { B } _ { j }$ and show why shrinking them to the nearest point to zero suppresses updates caused only by estimation error.

Lemma 4 (Coeficient accuracy and zero preservation). On G, for $H \geq H _ { A }$ and every epoch with $b > 0$ , write $B = A ^ { M } [ I , : ]$ and $D = [ D _ { 1 } , \dots , D _ { b } ]$ , where $D _ { l } = B _ { j _ { l } } - B _ { j _ { 0 } }$ . Each binding column has unique coeficients α<sub>j</sub> satisfying

$$
B _ { j } = v ( B ) { \bf 1 } + D \alpha _ { j } \qquad ( j \in J ) .\tag{11}
$$

These coeficients are uniformly bounded by $K _ { A }$ . With $\delta _ { j } = \sqrt { 2 \ell / c _ { j } }$ , taking $\delta _ { j } = 0 \ i f \ : c _ { j } = \infty \quad$ , the confidence intervals satisfy $| \widehat { \alpha } _ { l j } - \alpha _ { l j } | \le b _ { l j }$ . The corrected coeficients have the true sign whenever nonzero and obey

$$
\alpha _ { l j } = 0 \implies \widetilde { \alpha } _ { l j } = 0 , \qquad | \widetilde { \alpha } _ { l j } | \le | \alpha _ { l j } | ,\tag{12}
$$

as well as

$$
| \widetilde { \alpha } _ { l j } - \alpha _ { l j } | \le K _ { A } \left( \delta _ { j } + \sum _ { k } e _ { k } | \alpha _ { k j } | \right) .\tag{13}
$$

If $b = 0$ , every binding column is $v ( B ) \mathbf { 1 }$

Proof. We first establish the exact representation, using Lemma 3. For every $j \in J ,$ , the projected diference $P _ { 0 } ( B _ { j } - B _ { j _ { 0 } } )$ is in the span of $P _ { 0 } D$ . Thus $B _ { j } - B _ { j _ { 0 } } = D \gamma _ { j } + a _ { j } { \bf 1 }$ for some $\gamma _ { j } , a _ { j }$ . Evaluating at a true optimal row strategy gives $a _ { j } = 0$ , because all binding columns have value $v ( B )$ and every column of D has zero payof at that strategy.

For a dual optimum $y ,$ complementary slackness implies $\operatorname { s u p p } ( y ) \subseteq J$ . Moreover, $B y \le v ( B ) \mathbf { 1 }$ and an optimal row strategy with all coordinates positive forces equality in every row. Consequently

$$
v ( B ) { \bf 1 } = B y = B _ { j _ { 0 } } + D \sum _ { j \in J } y _ { j } \gamma _ { j } , \qquad B _ { j } = v ( B ) { \bf 1 } + D \left( \gamma _ { j } - \sum _ { k \in J } y _ { k } \gamma _ { k } \right) .
$$

This proves existence in (11). Independence of $P _ { 0 } D$ gives uniqueness. The true matrices and selected lists form a finite family, so their coeficients have a common finite bound. For $b = 0$ , the same argument gives $B _ { j } = v ( B ) \mathbf { 1 }$

For $b > 0$ , set $F = D ^ { \top } \widehat { R } - \operatorname { I d } _ { b }$ and $\nu _ { l } = \Vert \widehat { R } _ { \cdot l } \Vert _ { 1 }$ . The confidence bounds give

$$
\| D _ { l } - \widehat { D } _ { l } \| _ { \infty } \leq 2 e _ { l } , \qquad | F _ { l k } | \leq 2 e _ { l } \nu _ { k } \leq K _ { A } e _ { l } .\tag{14}
$$

Using (11) and $\mathbf { 1 } ^ { \top } \widehat { R } = 0$ , we obtain the exact identity

$$
\widehat { \alpha } _ { j } - \alpha _ { j } = \boldsymbol { F } ^ { \intercal } \alpha _ { j } + \widehat { R } ^ { \intercal } ( \widehat { B } _ { j } - B _ { j } ) .
$$

Let $\begin{array} { r } { Q _ { j } = \sum _ { k } e _ { k } | \alpha _ { k j } | } \end{array}$ and $\begin{array} { r } { \widehat { Q } _ { j } = \sum _ { k } e _ { k } | \widehat { \alpha } _ { k j } | } \end{array}$ . It follows that

$$
| \widehat { \alpha } _ { l j } - \alpha _ { l j } | \leq \nu _ { l } ( \delta _ { j } + 2 Q _ { j } ) , \qquad Q _ { j } \leq \widehat { Q } _ { j } + \frac { \chi } { 2 } \delta _ { j } + \chi Q _ { j } .
$$

Since $\chi \leq 1 / 2$ , these inequalities imply

$$
| \widehat { \alpha } _ { l j } - \alpha _ { l j } | \leq \frac { \nu _ { l } ( \delta _ { j } + 2 \widehat { Q } _ { j } ) } { 1 - \chi } \leq 2 \nu _ { l } ( \delta _ { j } + 2 \widehat { Q } _ { j } ) = b _ { l j } .
$$

Thus the interval used by the algorithm contains $\alpha _ { l j }$ . The shrinkage in (4) chooses its nearest point to zero, which lies between zero and $\alpha _ { l j }$ , which proves (12) and the sign assertion. Its distance from $\alpha _ { l j }$ is at most $2 b _ { l j }$ . The reverse triangle inequality also gives

$$
\widehat { Q } _ { j } \leq ( 1 + \chi ) Q _ { j } + \frac { \chi } { 2 } \delta _ { j } .
$$

Substituting this into $2 b _ { l j }$ and using $\nu _ { l } \leq K _ { A }$ proves (13). If $c _ { j } = \infty$ , then $B _ { j } = \widehat { B } _ { j } = { \bf 1 }$ ; as this column is binding, $v ( B ) = 1$ and both coeficient vectors are zero. □

## A.4 Column visits and accumulated estimation error

We now fix an epoch and a prefix $\tau$ of its rounds. Let $\begin{array} { r } { N _ { j } = \sum _ { t \in \mathcal { T } } \mathbf { 1 } \{ J _ { t } = j \} } \end{array}$ . For $b > 0$ , define the active columns and their visit counts by

$$
S _ { l } = \{ j \in J : \alpha _ { l j } \neq 0 \} , \qquad N _ { l } = \sum _ { j \in S _ { l } } N _ { j } , \qquad q _ { l } = c _ { j _ { l } } , \qquad d _ { l } = \sum _ { j \in J } N _ { j } \alpha _ { l j } .
$$

The two quantities used in the regret argument are

$$
V = \sum _ { l } e _ { l } ^ { 2 } N _ { l } , \qquad U = \sum _ { l } e _ { l } | d _ { l } | .
$$

Here V measures the accumulated estimation cost, while $U$ measures the payof advantage available from a persistent imbalance in the opponent’s columns. We set $U = V = 0$ for $b = 0$ . For visit bounds, extend $\delta _ { j } = \sqrt { 2 \ell / c _ { j } }$ to all columns, again taking zero for infinite counts.

Lemma 5 (Column visit bounds and accumulated estimation error). On ${ \mathcal { G } } ,$ for $H \geq H _ { A }$ , every epoch prefix satisfies

$$
N _ { j } \leq \frac { 6 c _ { j } } { \beta _ { A } } \quad ( c _ { j } < \infty ) , \qquad \sum _ { j } \delta _ { j } ^ { 2 } N _ { j } \leq \frac { 1 2 m \ell } { \beta _ { A } } ,\tag{15}
$$

where $\beta _ { A }$ is from Lemma 3. For $b > 0$ , the active visit counts obey $N _ { l } \leq K _ { A } ( q _ { l } + | d _ { l } | )$ . Consequently, there is a finite $a _ { A }$ depending only on A such that

$$
V \leq a _ { A } \ell + a _ { A } \varepsilon U .\tag{16}
$$

The bound (16) is stronger than a direct bound by the epoch length: many active visits must either exhaust the available samples or create a large $U _ { ; }$ , which the next lemma uses to reduce regret.

Proof. We use the count thresholds to bound column visits and the order of the selected diferences to control the active counts.

Visits before reconstruction. For a column with finite $c _ { j } .$ , choose an acquired row $i \in I$ attaining its minimum count at the start of the epoch. The next threshold for this entry is at most $2 c _ { j }$ Throughout the epoch, including its final round, its count is therefore at most $2 c _ { j }$ . If u is the last round of the prefix, Lemmas 2 and 3 give

$$
\frac { \beta _ { A } N _ { j } } { 2 } \leq \frac { Q _ { i j } ( u ) } { 2 } \leq C _ { i j } ( u ) + \ell \leq 2 c _ { j } + \ell .
$$

Although $Q _ { i j } ( u )$ includes observations before this epoch, these only increase it. Since $c _ { j } \geq h \geq \ell ;$ we obtain $N _ { j } \leq 6 c _ { j } / \beta _ { A }$ . Multiplying by $\delta _ { j } ^ { 2 } = 2 \ell / c _ { j }$ and summing over finite-count columns proves (15); infinite-count columns contribute zero.

Active visits and signed coeficients. Fix a coordinate $l . ~ \mathrm { I f } ~ j \in J$ has $c _ { j } > q _ { l }$ , it was processed before $j _ { l } ,$ or is the reference column. By Lemma 3, its projected diference is spanned by the previously accepted diferences. The constant-vector remainder again vanishes at a true optimum. Uniqueness in (11) therefore gives

$$
\alpha _ { l j } = \alpha _ { l j _ { 0 } } \qquad ( c _ { j } > q _ { l } , ~ j \in J ) .
$$

Let $\begin{array} { r } { N _ { \mathrm { l o w } } = \sum _ { j \in J : c _ { i } \le q _ { l } } N _ { j } } \end{array}$ . Equation (15) yields $N _ { \mathrm { l o w } } \leq K _ { A } q _ { l } . \ \mathrm { H } \ \alpha _ { l j _ { 0 } } = 0 ,$ , only these columns can be active, so $N _ { l } \le \mathsf { K } _ { A } q _ { l }$

Otherwise, all columns with $c _ { j } \ > \ q _ { l }$ share the same nonzero coeficient. Writing $N _ { \mathrm { h i g h } } =$ $\textstyle \sum _ { j \in J : c _ { j } > q _ { l } } N _ { j }$ , we have

$$
d _ { l } = \alpha _ { l j _ { 0 } } N _ { \mathrm { h i g h } } + \sum _ { j \in J : c _ { j } \le q _ { l } } N _ { j } \alpha _ { l j } .
$$

The coeficients are bounded, and their nonzero values have a positive minimum over the finite family in Lemma 4. It follows that

$$
N _ { \mathrm { h i g h } } \leq K _ { A } ( | d _ { l } | + N _ { \mathrm { l o w } } ) , \qquad N _ { l } \leq N _ { \mathrm { h i g h } } + N _ { \mathrm { l o w } } \leq K _ { A } ( | d _ { l } | + q _ { l } ) .
$$

Finally, $e _ { l } ^ { 2 } q _ { l } = 2 \ell$ and $e _ { l } \leq \varepsilon$ imply

$$
V \leq K _ { A } \sum _ { l } e _ { l } ^ { 2 } q _ { l } + K _ { A } \sum _ { l } e _ { l } ^ { 2 } | d _ { l } | \leq a _ { A } \ell + a _ { A } \varepsilon U
$$

for a suitable $a _ { A } .$ proving (16). For $b = 0$ , the last inequality holds with $U = V = 0$

## A.5 Regret within an epoch

The next lemma isolates the contribution that ofsets V . For an epoch prefix $\tau$ , define its completedgame deficit by

$$
\mathcal { R } _ { \mathrm { e p o c h } } = \sum _ { t \in \mathcal { T } } ( v ( B ) - p _ { t } ^ { \top } B _ { J _ { t } } ) , \qquad B = A ^ { M } [ I , : ] .
$$

In expressions involving B, we identify $p _ { t }$ with its restriction to I.

Lemma 6 (Epoch regret with a payof advantage). There are finite constants $b _ { A } , H _ { A }$ depending only on A such that, on $\mathcal { G }$ , every epoch prefix of a run with $H \geq H _ { A }$ satisfies

$$
\mathcal { R } _ { \mathrm { e p o c h } } \leq b _ { A } \ell + b _ { A } V - U ,\tag{17}
$$

where $U , V$ are the quantities in Lemma 5. $I f b = 0$ , then $\mathcal { R } _ { \mathrm { e p o c h } } \leq 0$

Proof. When $b = 0$ , Lemma 4 makes every binding column equal to $v ( B ) \mathbf { 1 }$ , while Lemma 3 gives positive surplus against every other column. Each round’s deficit is therefore nonpositive. We henceforth suppose $b > 0$

An optimal comparison strategy. Let $F = D ^ { \top } \widehat { R } - \operatorname { I d } _ { b }$ as in the proof of Lemma 4, and define

$$
\boldsymbol { p } ^ { * } = \widehat { \boldsymbol { x } } - \widehat { R } ( \mathrm { I d } _ { b } + \boldsymbol { F } ) ^ { - 1 } \boldsymbol { D } ^ { \top } \widehat { \boldsymbol { x } } .
$$

By (14), $F = { \mathcal { O } } _ { A } ( \varepsilon )$ , so the inverse is bounded for suficiently large H. Since $\widehat { D } ^ { \top } \widehat { x } = 0 , D ^ { \top } \widehat { x } = \mathcal { O } _ { A } ( \varepsilon )$ The correction from xb to $p ^ { * }$ thus has size ${ \mathcal { O } } _ { A } ( \varepsilon )$ and total mass zero. Lemma 3 implies that $p ^ { * }$

remains a probability distribution with positive surplus at every nonbinding column. Also $D ^ { \top } p ^ { * } = 0$ so (11) gives $( p ^ { * } ) ^ { \top } B _ { j } = v ( B )$ for every $j \in J$ . Consequently $p ^ { * }$ is a true optimum.

Because $\boldsymbol { p } ^ { * } - \boldsymbol { \widehat { x } }$ is in the range of $\widehat { R } ,$ we can write

$$
p ^ { * } = \widehat { x } + 4 \widehat { R } E z ^ { * } , \qquad z _ { l } ^ { * } = \frac { \widehat { D } _ { l } ^ { \top } p ^ { * } } { 4 e _ { l } } = \frac { ( \widehat { D } _ { l } - D _ { l } ) ^ { \top } p ^ { * } } { 4 e _ { l } } .
$$

Equation (14) gives $| z _ { l } ^ { * } | \le 1 / 2$ . For the actual iterate $z _ { t } .$ , define

$$
u _ { l } ( t ) = z _ { l } ^ { * } + \sum _ { k } F _ { l k } \frac { e _ { k } } { e _ { l } } ( z _ { k } ^ { * } - z _ { k , t } ) .
$$

This adjustment accounts for using empirical rather than true payof diferences. In fact,

$$
( p ^ { * } - p _ { t } ) ^ { \top } B _ { J _ { t } } = 4 \sum _ { l } e _ { l } \alpha _ { l J _ { t } } ( u _ { l } ( t ) - z _ { l , t } ) \qquad ( J _ { t } \in J ) .\tag{18}
$$

To verify the identity, substitute $p ^ { * } - p _ { t } = 4 \widehat { R } E ( z ^ { * } - z _ { t } )$ and $D ^ { \top } \widehat { R } = \operatorname { I d } _ { b } + F$ into (11).

Feasibility and variation of the comparator. The matrix $K = E ^ { - 1 } F E$ satisfies $| K _ { l k } | \le 2 e _ { k } \nu _ { k }$ so $\| K \| _ { \infty } \leq \chi$ . By (8), we can choose $H _ { A }$ large enough that $\chi \leq 1 / 6$ . Hence

$$
\| u ( t ) \| _ { \infty } \leq \| z ^ { * } \| _ { \infty } + \chi \| z ^ { * } - z _ { t } \| _ { \infty } \leq \frac { 1 } { 2 } + \frac { 3 } { 2 } \chi \leq \frac { 3 } { 4 } .
$$

Let $\mathrm { T V }$ denote the sum of absolute changes between consecutive rounds of the prefix. Lemma 4 bounds all stored coeficients and, by (12), coordinate k changes only when $J _ { t } \in S _ { k }$ . Nonexpansiveness of clipping gives $\mathrm { T V } ( z _ { k } ) \le K _ { A } e _ { k } N _ { k }$ , and therefore

$$
\mathrm { T V } ( u _ { l } ) \leq K _ { A } \sum _ { k } e _ { k } \mathrm { T V } ( z _ { k } ) \leq K _ { A } \sum _ { k } e _ { k } ^ { 2 } N _ { k } = K _ { A } V .\tag{19}
$$

Projected gradient comparison. For scalar updates $z _ { a + 1 } = \Pi _ { [ - 1 , 1 ] } ( z _ { a } + g _ { a } )$ and comparators $c _ { a } \in [ - 1 , 1 ]$ , nonexpansiveness gives

$$
( c _ { a } - z _ { a } ) g _ { a } \leq \frac { 1 } { 2 } \big ( ( z _ { a } - c _ { a } ) ^ { 2 } - ( z _ { a + 1 } - c _ { a } ) ^ { 2 } \big ) + \frac { 1 } { 2 } g _ { a } ^ { 2 } .
$$

Changing the comparator by an amount d changes a squared distance by at most $4 | d |$ . Summing the inequalities thus yields

$$
\sum _ { a } ( c _ { a } - z _ { a } ) g _ { a } \leq 2 + 2 \mathrm { T V } ( c ) + \frac { 1 } { 2 } \sum _ { a } g _ { a } ^ { 2 } .\tag{20}
$$

This is the scalar moving-comparator argument of Zinkevich (2003).

For each l, apply (20) on the subsequence of visits to $S _ { l }$ , using

$$
g _ { t } = 4 e _ { l } \widetilde { \alpha } _ { l J _ { t } } , \qquad c _ { t } = u _ { l } ( t ) + \frac { 1 } { 4 } \mathrm { s g n } ( d _ { l } ) .
$$

The comparator belongs to $[ - 1 , 1 ]$ because $| u _ { l } ( t ) | \le 3 / 4$ . Between these visits the coordinate does not change. Subsequence variation is at most the full variation in (19), and $\begin{array} { r } { \sum _ { t } g _ { t } ^ { 2 } \le K _ { A } e _ { l } ^ { 2 } N _ { l } } \end{array}$ Summing over coordinates, with empty subsequences contributing zero, bounds the total estimated comparison by $K _ { A } ( 1 + V )$

Replacing estimated coeficients by true coeficients. Since $| c _ { t } - z _ { l , t } | \le 2$ , (13) bounds the replacement error by a constant times

$$
\sum _ { l , t \in T : J _ { t } \in S _ { l } } e _ { l } \left( \delta _ { J _ { t } } + \sum _ { k } e _ { k } | \alpha _ { k J _ { t } } | \right) .
$$

For the first term, $2 e _ { l } \delta _ { j } \leq e _ { l } ^ { 2 } + \delta _ { j } ^ { 2 }$ gives

$$
\sum _ { l , t \in \mathcal { T } : J _ { t } \in S _ { l } } e _ { l } \delta _ { J _ { t } } \leq K _ { A } \left( V + \sum _ { j } \delta _ { j } ^ { 2 } N _ { j } \right) \leq K _ { A } ( V + \ell ) ,
$$

where the last inequality is (15). For the second term, let $N _ { l k }$ count visits to $S _ { l } \cap S _ { k }$ . The coeficients are bounded and vanish outside $S _ { k }$ , and

$$
{ e _ { l } e _ { k } N _ { l k } \leq \frac { 1 } { 2 } ( e _ { l } ^ { 2 } N _ { l } + e _ { k } ^ { 2 } N _ { k } ) . }
$$

Summing over l, k therefore bounds this term by $K _ { A } V$ . Combining these estimates with the projected gradient bound gives

$$
\sum _ { t \in T : J _ { t } \in J } 4 \sum _ { l } e _ { l } \alpha _ { l J _ { t } } ( u _ { l } ( t ) - z _ { l , t } ) + \sum _ { l } e _ { l } | d _ { l } | \leq K _ { A } \ell + K _ { A } V .
$$

Here the second term arises from the comparator shift: $\begin{array} { r } { 4 e _ { l } \cdot \frac { 1 } { 4 } \operatorname { s g n } ( d _ { l } ) \sum _ { t \in \mathcal { T } : J _ { t } \in S _ { l } } \alpha _ { l J _ { t } } = e _ { l } | d _ { l } | } \end{array}$ . By (18), the first term is exactly the completed-game deficit on binding rounds. Nonbinding rounds have nonpositive deficit by Lemma 3. This proves (17) with a suitable $b _ { A }$ □

## A.6 Proof of the main theorem

Proof of Theorem 1. We combine the two epoch inequalities, sum over epochs, and then apply the restart schedule.

An epoch contributes at most $\mathcal { O } _ { A } ( \ell )$ . Work on G in a run with $H \geq H _ { A }$ . Lemmas 5 and 6 give

$$
\begin{array} { r l } & { \mathcal { R } _ { \mathrm { e p o c h } } \leq b _ { A } \ell + b _ { A } V - U } \\ & { \qquad \leq b _ { A } ( 1 + a _ { A } ) \ell + ( a _ { A } b _ { A } \varepsilon - 1 ) U . } \end{array}
$$

Since $\varepsilon = \ell ^ { - 1 / 2 } \to 0$ , we can enlarge $H _ { A }$ so that $a _ { A } b _ { A } \varepsilon \leq 1$ . As $U \geq 0 .$ , every epoch prefix then has $\mathcal { R } _ { \mathrm { e p o c h } } \leq K _ { A } \ell .$ This also includes $b = 0$ , where the deficit is nonpositive.

A run prefix contributes at most $\mathcal { O } _ { A } ( \log ^ { 2 } ( e H ) )$ . Lemma 3 gives $v ( B ) = v ( A ^ { M } )$ in every epoch, so the epoch deficits sum to $\mathcal { D } _ { s }$ from Lemma 2. The epoch count in that lemma yields $\mathcal { D } _ { s } \leq K _ { A } \ell ( 1 + \log H )$ on G. On every observation sequence, $\mathcal { D } _ { s } \leq 2 s$ because completed payofs lie in $[ - 1 , 1 ]$ and the played strategies are distributions. By (6) and the failure probability in Lemma 2, we obtain

$$
\begin{array} { r l } { s v ( A ) - \mathbb { E } \displaystyle \sum _ { t = 1 } ^ { s } r _ { t } \leq K _ { A } \ell ( 1 + \log H ) + 2 s \mathbb { P } ( \mathcal { G } ^ { c } ) + 2 n m h } & { } \\ { \leq C ( A ) \log ^ { 2 } ( e H ) . } \end{array}\tag{21}
$$

We used $\ell = \mathcal { O } _ { n , m } ( \log ( e H ) ) , h = \mathcal { O } ( \ell ^ { 2 } )$ , and $s \leq H$ . For $H < H _ { A }$ , the bound $\begin{array} { r } { s v ( A ) - \mathbb { E } \sum _ { t = 1 } ^ { s } r _ { t } \leq } \end{array}$ 2H is covered by increasing the same finite constant $C ( A )$ . Thus (21) holds for every planned length and every prefix, uniformly over permitted opponents and reward processes.

Combining the runs. All preceding bounds hold conditionally on any history at the start of a fresh run. If global time T falls in run $^ { r , }$ we apply (21) to each completed run and the current prefix. Taking expectations gives

$$
R _ { T } \leq C ( A ) \sum _ { k = 0 } ^ { r } \log ^ { 2 } ( e H _ { k } ) \leq K _ { A } \log ^ { 2 } ( e H _ { r } ) , \qquad H _ { k } = 2 ^ { 2 ^ { k } } .
$$

The last inequality follows from $\log ( e H _ { k } ) = 1 + 2 ^ { k }$ log 2 and the geometric sum $\scriptstyle \sum _ { k = 0 } ^ { r } 4 ^ { k }$ . For $r \geq 1$ at least $H _ { r - 1 }$ rounds precede run $^ { r , }$ so

$$
\log H _ { r } = 2 \log H _ { r - 1 } \leq 2 \log T .
$$

The first run has length $H _ { 0 } = 2$ and contributes only a constant. Renaming the constant proves $R _ { T } \leq C ( A ) \log ^ { 2 } ( e T )$ for every integer $T \geq 1$ □