# THE HIDDEN PERCEPTION CONSTRAINT IN TASK-AWARE COMPRESSION

Sahan Liyanaarachchi, Semih Akkoc & Sennur Ulukus   
University of Maryland   
College Park, MD, USA   
sahanl,akkoc,ulukus @umd.edu

Aylin Yener The Ohio State University Columbus, OH, USA yener@ece.osu.edu

## ABSTRACT

With the recent advancements of neural compressors, explicitly incorporating perception constraints into the design of compression schemes has gained significant attention. Traditionally, these perception constraints ensure that the distribution of the reconstruction does not significantly deviate from the distribution of the source, thus attesting to the perceptual quality of the reconstruction. In this work, we uncover several perception constraints that are naturally present in task-aware compression. In particular, we consider a problem where the primary task is reconstruction and the secondary task is classification (i.e., a statistical test). We study this problem at varying levels of domain information available to us and discuss how to utilize the naturally emerging perception constraints to design rateminimal compression schemes that also maximize the utility of our secondary task. We show that in this setting, if the decision boundaries of the classifier are ill-defined (mismatch) for our source distribution, then matching onto a target distribution enhances our classification accuracy.

## 1 INTRODUCTION

Rate distortion theory has been a widely pursued interest among information theorists ever since its first formulation by Shannon (Shannon, 1948). The rate-distortion (RD) problem involves finding the theoretical limit for compression such that the distortion between the reconstruction obtained from the compressed representation and the original source is within a pre-specified level. This constraint is formally known as the distortion constraint and numerous variations of this problem have been studied throughout literature by considering various distortion functions, availability of side information and for different network topologies (Wyner & Ziv, 1976; Yamamoto, 1981; Timo et al., 2011; MolavianJazi & Yener, 2016). With the recent advancements of neural compressors, a new constraint known as the perception constraint was identified in (Blau & Michaeli, 2019) as a necessary feature for modern compression schemes. While the distortion constraint ensures that our reconstruction is close to the original source, the perception constraint ensures that the distribution of the reconstruction is close to the source distribution. This results in more realistic reconstructions, which is a desired feature of modern compressors, but this comes at the expense of extra distortion.

This new problem of finding the optimal rate that simultaneously satisfies both the distortion and perception constraints is known as the rate-distortion-perception (RDP) problem and has gained significant attention recently. We refer to Section 2 for the recent advances on this problem. In this work, we study a practically motivated variant of the above problem. In here, we explore how to compress while simultaneously maximizing the utility of a given task. In particular, we focus on the following problem: Suppose we want to transmit a compressed representation of our data to a remote server which uses the reconstructed data for a certain task. For instance, suppose that our source is a random variable that may originate from several hypotheses, H<sub>i</sub>. We want to compress and transmit our observation of this source random variable X and form a reconstruction X<sup>ˆ</sup> of it at the server (receiver) such that its distortion is within a pre-specified level. This is our primary goal. At the same time, suppose the server uses a generic or ill-defined statistical test (the test may be designed for a different data distribution or there may be a mismatch in prior distributions which was used to design the test) on the reconstructed data to identify the originating hypothesis. We want the error of this statistical test to be low as well. This is our secondary task.

![](images/bb8f30944a5ade79f44616dc84b0ac1f614d513a1ef0d03d98aac15822aaed82.jpg)  
Figure 1: Diagram of the system model.

In the context of machine learning, this can be viewed as using a pretrained classifier at the remote server to classify data coming from a distribution which is significantly different from the training distribution of the classifier. In here, the pretrained classifier can be considered as a third-party tool available at the sever which we use for our task. As a specific example, consider the presence (hypothesis $H _ { 1 } )$ and absence (hypothesis $H _ { 0 } )$ of a particular disease; and that the detector (classifier) has been trained with a data set that is dominated by a certain race; and that the new patient belongs to an underrepresented race. This paper tackles the problem of how to compress the data of the new patient with as few bits as possible, such that, the decompressed data at the detector $( { \hat { X } } )$ has small distortion with the original data (X), and at the same time, the down-stream task of classification of the patient $( H _ { 0 } \mathrm { o r } H _ { 1 } )$ using the decompressed data has small probability of error. Essentially, we are trying to compress the data in a way to leverage a generic classifier to our advantage.

We tackle the above problem by considering varying levels of domain knowledge available to us. In particular, we identify the following three levels of information availability:

• Level 1: We know the distributions of the individual hypotheses of the source random variable and the decision regions of the classifier at the receiver side.

• Level 2: We know the individual distributions for each hypothesis of the source random variable but we do not know the exact decision regions at the receiver side. However, in this scenario, we assume that we do know that the classifier works well for some known distribution (i.e., yields low probability of error on a known distribution). In the computer vision domain, this can be viewed as a generic image classifier which is known to have a good inference accuracy on a common dataset such as ImageNet.

• Level 3: We do not know the individual distributions for each hypothesis of the source random variable and we do not know decision regions of the classifier at the receiver. Again, we assume that the classifier works well for a known distribution.

Under the above three levels of information availability, we find the optimal compression rates that satisfy the primary task while maximizing the performance of the secondary task which is classification. We identify that perception is a hidden constraint within this problem setup and formulate the above problem as an equivalent RDP problem. We show that when the exact decision regions of the classifier is not known, a good strategy is to compress the data so that the reconstruction’s distribution is as close as possible to a target distribution for which we know the test yields good results.

## 2 RELATED WORK

The RDP tradeoff was first formalized in (Blau & Michaeli, 2019), which introduced the RDP function, where the authors show under mild regularity conditions such as convexity of the perception function, that any achievable rate must lie on the epigraph of the RDP function. The achievability of the above RDP function was later established in (Theis & Wagner, 2021) with the use of stochastic coding schemes in the presence of unlimited common randomness. With the use of soft covering lemma, this problem was later studied in (Wagner) with finite common randomness and total variation as the perception function. Under the same setting, the achievable region of rates was fully characterized with the presence of side information in (Hamdi & Gund¨ uz, 2023). The role¨ of side information and the common randomness in the RDP tradeoff was then studied in (Hamdi et al., 2026), which separately treated perception under marginal realism (marginal distribution of reconstruction should match the source distribution) and joint realism (joint distribution of the side information and reconstruction should match the joint distribution of side information and source). The works in (Zhang et al., 2025; Qian et al., 2025) further studied the RDP problem. All of the above works investigate the problem of matching the distribution of the reconstruction on to the distribution of the original source.

In our work, we are interested in matching the reconstruction distribution on to a target distribution that is different from the source distribution. This notion of perception has been studied in several other occasions. In (Akkoc et al.), this notion of perception was termed as deception where they found the optimal compression rates when the reconstruction should exhibit statistical characteristics significantly different from the original distribution. On a separate work in (Arda & Yener, 2025b), the rate distortion framework for text summarization introduced in (Arda & Yener, 2025a) was extended by introducing a perception constraint which measures the perceptual quality between the summaries and a reference distribution instead of the original text distribution.

In the domain of task-aware compression, there exist several works that fit our problem setting. A unified approach for incorporating classification accuracy into lossy image compression was introduced in (Zhang, 2023), where they introduced classification-distortion-perception (CDP) function and the rate-distortion-classification (RDC) function; the former corresponds to minimizing the classification error for a given distortion and perception constraint. whereas the latter corresponds to minimizing the rate given the constraints on the distortion and classification accuracy. The distinction between the RDP, RDC and rate-perception-classification (RPC) functions was highlighted in (Wang et al., 2024) by deriving closed-form expressions of the above functions for Bernoulli and Gaussian source distributions. In (Wang et al., 2025), the authors extended the RDC function by incorporating a perception constraint and multiple classification constraints measured via the entropy of classification labels given the reconstruction. However, here too the perception constraint was measured with respect to the source distribution. In (Nguyen et al., 2025), this entropy-based RDC function was studied in a setting with a universal encoder and multiple task-specific decoders.

The closest to our work is the work in (Nguyen et al., 2026), which considers the problem of minimizing distortion with a rate and perception constraint while using an entropy-based classification constraint. This is an extension of the cross-domain lossy compression framework introduced in (Liu et al., 2022). Here, they formulate the problem as an optimal transport problem where they consider the source distribution as a noisy version of a certain clean distribution. Therefore, perception in this case is measured between this noisy distribution and the clean distribution. They are interested in minimizing the expected distortion given a rate constraint, whereas we are interested in deriving the optimal compression rates for a given distortion in our setting. They solve the optimal transport problem by explicitly incorporating a perception constraint in addition to the entropybased classification constraint, whereas we show that the perception constraint emerges naturally when realizing the constraint on classification accuracy, and that satisfying this naturally emerging perception constraint is sufficient to do well in terms of the classification accuracy.

## 3 PROBLEM FORMULATION

Suppose we have a total of M hypotheses. Let X be a random variable which is generated from hypothesis $H _ { i }$ with probability $\pi _ { i }$ following the distribution<sup>1</sup> $p _ { i } ( x )$ and let be the support of $X$ . The realization of random variable X is then compressed and sent to a remote server which reconstructs it as ${ \hat { X } } \in { \mathcal { X } }$ . At the remote server, this reconstruction will be subjected to a statistical test (classification) which was designed for a random variable $Y$ whose support is also $x ,$ which originates from $H _ { i }$ with probability $\sigma _ { i }$ following the distribution $q _ { i } ( x )$ . In other words we will be using a statistical test designed with incorrect domain knowledge to classify our reconstructed data.

Suppose that $X$ came from the hypothesis $H .$ . Then, based on the reconstruction $\hat { X }$ , we will be generating an estimate $\hat { H }$ of the original hypothesis $H ,$ , using the available test at the server. In particular, let $R _ { i }$ be the decision regions of the predefined test at the server with $R _ { i } \cap R _ { j } = \emptyset$ for $i \neq j$ and $\textstyle \bigcup _ { i = 0 } ^ { M - 1 } R _ { i } = \mathcal { X }$ . Then, $\hat { H } = i \mathrm { i f } \hat { X } \in R _ { i }$ . Let $p _ { e } ( \hat { X } ) = \mathbb { E } [ \mathbb { 1 } \{ H \neq \hat { H } \} ]$ be the probability of error of our test and let $d ( X , { \hat { X } } )$ denote an integrable distortion function which measures how close our reconstruction is to our original random variable. Our goal is to compress the random $X$ such that reconstruction $\hat { X }$ satisfies a constraint on the average distortion and the probability of error. In particular, we want our reconstruction to satisfy $\mathbb { E } [ d ( X , \hat { X } ) ] \le D$ and $p _ { e } ( { \hat { X } } ) \leq \alpha .$ where $D \in \mathbb { R } _ { 0 } ^ { + }$ and $\alpha \in [ 0 , 1 ]$ (see Fig. 1). Here, the distortion constraint ensures the semantic similarity between the original random variable and its reconstruction, and the constraint on the error probability ensures that we are simultaneously performing well at our task of interest which is to identify correctly the underlying hypothesis from which the random variable originated.

We will tackle the above problem at varying levels of domain knowledge that is available to us. To formally define the problem, we will start with a definition for achievability in this problem setting. In here, we will use ${ \bar { X } } ^ { n }$ to represent a collection n i.i.d. realization of X and we use $X _ { n , i }$ to represent the ith element of this collection.

Definition 1 The tuple $( R , D , \alpha )$ is achievable $i f \forall \epsilon > 0 , \exists n \in \mathbb { N } ,$ , an encoding function $f :$ $X ^ { n } \times U  \mathbb { N }$ , and a decoding function g : $\mathbb { N } \times U  { \hat { X } } ^ { n }$ , such that $K = f ( X ^ { n } , U ) , \hat { X } ^ { n } = g ( K , U )$ and satisfies,

$$
\frac { H ( K | U ) } { n } < R + \epsilon , \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } [ d ( X _ { n , i } , \hat { X } _ { n , i } ) ] \leq D , \frac { 1 } { n } \sum _ { i = 1 } ^ { n } p _ { e } ( \hat { X } _ { n , i } ) \leq \alpha ,\tag{1}
$$

where U is the common randomness independent ofX available to both the encoder and decoder.

Next, to characterize the achievable region for a given distortion constraint $D$ and an error bound $\alpha ,$ we will define the rate-distortion-error (RDE) function $R ( D , \alpha )$ as

$$
R ( D , \alpha ) = \operatorname* { i n f } _ { p _ { \hat { X } | X } } I ( X ; \hat { X } ) \quad \mathrm { s u c h t h a t } \quad \mathbb { E } [ d ( X , \hat { X } ) ] \leq D , \quad p _ { e } ( \hat { X } ) \leq \alpha .\tag{2}
$$

Note that this function is not well-defined for every $( D , \alpha )$ pair. This is because the decision regions at the server are suboptimal. For instance, suppose $D = 0$ and we use the mean squared error (MSE) as the distortion metric, then even with infinite rate, we may not be able to achieve the desired error probability since $R _ { i } \mathbf { s }$ are not optimal decision regions for $\dot { X }$ . Therefore, we will define this function only if $S _ { D , \alpha } \neq \emptyset .$ , where $S _ { D , \alpha } = \{ p _ { \hat { X } | X } : \ \mathbb { E } [ d ( X , \hat { X } ) ] \leq D , p _ { e } ( \hat { X } ) \leq \alpha \}$ . However, Proposition 1 shows whenever a tuple $( R , D , \alpha )$ is achievable, the function $R ( D , \alpha )$ is well-defined.

Proposition 1 The tuple $( R , D , \alpha )$ is achievable iff $\mathbf { \nabla } \cdot S _ { D , \alpha } \neq \varnothing$ and $R \geq R ( D , \alpha )$

The proof of Proposition 1 is given in Appendix A. Proposition 1 simply states that $R ( D , \alpha )$ defines the minimum achievable compression rate satisfying a given distortion constraint D and an error bound $\alpha .$ . Therefore, finding the minimum compression rate is equivalent to solving the RDE function over the set of stochastic mappings $p _ { \hat { X } | X }$ from $X$ to $\hat { X }$ that are feasible under the given constraints. Now, given a stochastic mapping $p _ { \hat { X } | X }$ , the probability of error is

$$
p _ { e } ( \hat { X } ) = \sum _ { i = 0 } ^ { M - 1 } \int _ { \mathcal { X } } \pi _ { i } p _ { i } ( x ) \int _ { \bar { R } _ { i } } p _ { \hat { x } | x } \mathrm { d } \hat { x } \mathrm { d } x = 1 - \sum _ { i = 0 } ^ { M - 1 } \int _ { \mathcal { X } } \pi _ { i } p _ { i } ( x ) \int _ { R _ { i } } p _ { \hat { x } | x } \mathrm { d } \hat { x } \mathrm { d } x ,\tag{3}
$$

where ${ \bar { R } } _ { i } = { \mathcal { X } } \backslash R _ { i }$

Let $R _ { i } ^ { * }$ be the decision regions of the optimal hypothesis test constructed for X where $R _ { i } ^ { * } \cap R _ { j } ^ { * } = \varnothing$ for $i \neq j$ and $\begin{array} { r } { \bigcup _ { i = 0 } ^ { M - 1 } R _ { i } ^ { * } = \mathcal { X } } \end{array}$ . Let $p _ { e } ^ { * }$ be the error probability of the optimal hypothesis test. Then, in Lemma 1, whose proof is given in Appendix B, we show that $p _ { e } ( \hat { X } ) \geq p _ { e } ^ { * }$ . Hence, the above problem is feasible only if $\alpha \geq p _ { e } ^ { * }$

Lemma 1 For any stochastic mapping $p _ { \hat { X } | X } , p _ { e } ( \hat { X } ) \geq p _ { e } ^ { * }$ with equality iff almost everywhere,

$$
\int _ { R _ { i } } p _ { \hat { x } | x } d \hat { x } = 1 , i f x \in R _ { i } ^ { * } .\tag{4}
$$

Therefore, we will only consider values of $\alpha \geq p _ { e } ^ { * }$ . In the next section, we will focus on the setting where $\alpha = p _ { e } ^ { * }$ , and show that the above problem can be also formulated as an RDP problem.

## 4 COMPRESSION WITH OPTIMAL ERROR PROBABILITY

In this section, we will study the above problem under the special case where $\alpha = p _ { e } ^ { * }$ . We will call this setup the rate-distortion-minimum-error (RDME) problem. Therefore, our stochastic mappings must satisfy (4). Let us denote a stochastic map that satisfies these conditions by $p _ { \hat { X } | X } ^ { * } .$ Let J be a discrete random variable which assumes $J = i \mathrm { i f } X \in R _ { i } ^ { * }$ . Let $\begin{array} { r } { p ( x ) = \sum _ { i = 0 } ^ { M - 1 } { \pi } _ { i } p _ { i } ( x ) } \end{array}$ and $\begin{array} { r } { P _ { i } = \int _ { R _ { i } ^ { * } } p ( x ) d x } \end{array}$ . Then, $J = i$ with probability $P _ { i }$ . Here, J will denote the estimate of H if we carryout the optimal hypothesis test on X. Note that, X can be also represented using J as $X = X _ { J }$ where $\begin{array} { r } { X _ { i } \sim \frac { 1 } { P _ { i } } p ( x ) \mathbb { 1 } \{ x \in R _ { i } ^ { * } \} } \end{array}$ . Let ${ \hat { X } } _ { i }$ be the reconstruction generated when $J = i .$ . Now, under $p _ { \hat { X } | X } ^ { * }$ , we have that J is fully deterministic given either $\hat { X }$ or X. Therefore,

$$
I ( X ; \hat { X } ) = I ( X ; \hat { X } ) + I ( X ; J | \hat { X } ) = I ( X ; J ) + I ( X ; \hat { X } | J ) = H ( J ) + \sum _ { i = 0 } ^ { M - 1 } P _ { i } I ( X _ { i } ; \hat { X } _ { i } ) .\tag{5}
$$

Let $S _ { D } = \{ D : S _ { D , p _ { c } ^ { * } } \neq \varnothing \}$ . Then, for $D \in S _ { D }$ , the optimal compression rate $R ( D )$ will be given by $R ( D ) = H ( J ) + { \bf \ddot { \cal R } ^ { \prime } } ( D )$ , where

$$
R ^ { \prime } ( D ) = \operatorname* { i n f } _ { p _ { \hat { X } | X } ^ { * } } \sum _ { i = 0 } ^ { M - 1 } P _ { i } I ( X _ { i } ; \hat { X } _ { i } ) \quad \mathrm { s u c h ~ t h a t } \quad \sum _ { i = 0 } ^ { M - 1 } P _ { i } \mathbb { E } [ d ( X _ { i } , \hat { X } _ { i } ) ] \leq D .\tag{6}
$$

Therefore, in order to achieve the optimal error probability, our compression scheme must always send at least the bits necessary to represent our own estimate of the hypothesis $( \mathrm { i . e . , } J )$ done at the transmitter side on X.

Next, note that, we can also represent the above optimization problem as an RDP problem as follows,

$$
R _ { P } ^ { \prime } ( D ) = \operatorname* { i n f } _ { p _ { \hat { x } | x } } \sum _ { i = 0 } ^ { M - 1 } P _ { i } I ( X _ { i } ; \hat { X } _ { i } ) \mathrm { ~ s u c h ~ t h a t ~ } \sum _ { i = 0 } ^ { M - 1 } P _ { i } \mathbb { E } [ d ( X _ { i } , \hat { X } _ { i } ) ] \leq D , \sum _ { i = 0 } ^ { M - 1 } P _ { i } D ( P _ { \hat { X } _ { i } } | | P _ { Y _ { i } } ) \leq P ,\tag{7}
$$

where $P _ { \hat { X } _ { i } }$ is the distribution of ${ \hat { X } } _ { i }$ and $P _ { Y _ { i } } ( x ) = \tilde { q } _ { i } ( x ) = \textstyle { \frac { 1 } { Q _ { i } } } q ( x ) \mathbb { 1 } \{ x \in R _ { i } \}$ with $q ( x ) \ =$ $\begin{array} { r } { \sum _ { i = 0 } ^ { M - 1 } \sigma _ { i } q _ { i } ( x ) , Q _ { i } = \int _ { R _ { i } } q ( x ) d x . } \end{array}$ and $P < \infty$ being our perception constraint. Note that, we have changed $p _ { \hat { X } | X } ^ { * } \to p _ { \hat { X } | X }$ since the perception constraint ensures that any feasible stochastic mapping for the above problem must satisfy (4). Also, note that, as $P \to \infty$ , the above optimization problem will converge to the original optimization problem for the RDME problem. Moreover, in this scenario, we have that $R _ { P } ( \hat { D _ { ) } } \ge \dot { R _ { ( D ) } }$ , where $\dot { R } _ { P } ( D ) = H ( J ) + R _ { P } ^ { \prime } ( \dot { D } )$

## 5 COMPRESSION VIA DISTRIBUTION MATCHING

In Section 3, we formally defined the RDE function and in the previous section, we noticed how the RDME problem can be formulated as an RDP problem. However, both of these problem formulations required knowledge of the decision regions at the receiver (Level 1). These decision regions depend both on the exact distributions it was designed for and the particular test that is used $( \mathrm { e . g . }$ Neyman-Pearson or MAP). In a more general setting, if the task at hand is something more complex, such as image classification, these decision regions depend on both the structure of the classifier at the receiver as well as its training distribution. In this section, we discuss compression schemes designed without explicit knowledge about the decision regions of the test at the receiver. However, in this scenario, we assume that we know the test works well (low error probability) for some known distribution (Level 2). Let this test distribution be defined as follows: the input x to the statistical test is generated from $H _ { i }$ with probability $\gamma _ { i }$ following the distribution $r _ { i } ( x )$ . Now, Theorem 1 gives an important relationship between $p _ { e } ( \tilde { X } )$ and the distributions $r _ { i } ( x )$

Theorem 1 $L e t r _ { e }$ denote the probability oferrorfor a known test distribution $r _ { i } ( x )$ when using the statistical test at the receiver. Then, the following error bound is satisfied

$$
p _ { e } ^ { * } \leq p _ { e } ( \hat { X } ) \leq p _ { e } ^ { * } + C r _ { e } + \sum _ { i = 0 } ^ { M - 1 } P _ { i } \| P _ { \hat { X } _ { i } } - r _ { i } \| _ { T V }\tag{8}
$$

where $\begin{array} { r } { C = \operatorname* { m a x } _ { i } \frac { P _ { i } } { \gamma _ { i } } } \end{array}$ and $\Vert . \Vert _ { T V }$ is the total variation distance.

Corollary 1 Thefollowing error bound is satisfied,

$$
p _ { e } ^ { * } \le p _ { e } ( \hat { X } ) \le p _ { e } ^ { * } + C r _ { e } + \sqrt { \frac { \hat { P } } { 2 } }\tag{9}
$$

where $\begin{array} { r } { \hat { P } = \sum _ { i = 0 } ^ { M - 1 } P _ { i } D ( P _ { \hat { X } _ { i } } | | r _ { i } ) } \end{array}$

The proofs of Theorem 1 and Corollary 1 are given in Appendix C. In order to explicitly represent the probability of error, we needed to know the decision regions of the test at the receiver. Corollary 1 presents us with an alternative way to formulate the constraint on the probability of error. In particular, we can represent the optimal compression problem in (2) as,

$$
R ^ { ( 2 ) } ( D , \alpha ) = \operatorname* { i n f } _ { p _ { \hat { X } | X } } I ( X ; \hat { X } ) \quad \mathrm { s u c h t h a t } \quad \mathbb { E } [ d ( X , \hat { X } ) ] \leq D , \quad \sum _ { i = 0 } ^ { M - 1 } P _ { i } D ( P _ { \hat { X } _ { i } } | | r _ { i } ) \leq P _ { \alpha } ,\tag{10}
$$

where $P _ { \alpha }$ is a function of $\alpha , { \mathrm e . g . } , P _ { \alpha } = 2 ( \alpha - p _ { e } ^ { * } - C r _ { e } ) ^ { 2 } { \mathrm { i f } } \alpha \geq p _ { e } ^ { * } + C r _ { e }$ . However, since the perception constraint only forms an upper bound on the probability of error, we have that $R ( D , \alpha ) \leq$ $R ^ { ( 2 ) } ( D , \alpha )$ even when both of the problems are feasible. We further note that, if we are able to send J to the receiver side, then the optimal compression problem becomes $R _ { P } ^ { \prime } ( D )$ by replacing $P _ { Y _ { i } }$ with $r _ { i }$ . The above formulations represents Level 2 of our information availability.

Next, we look at the more general setting where we do not have any information about the individual distributions of the source random variable (Level 3). Here, we will only have access to the mixture distribution $( p ( x )$ of X). In such a setting, one may try to formulate the optimal compression problem as follows,

$$
R ^ { ( 3 ) } ( D , \alpha ) = \operatorname* { i n f } _ { p _ { \hat { X } | X } } I ( X ; \hat { X } ) \quad \mathrm { s u c h t h a t } \quad { \mathbb { E } } [ d ( X , \hat { X } ) ] \leq D , \quad D ( P _ { \hat { X } } | | P _ { Z } ) \leq P ,\tag{11}
$$

where $\begin{array} { r } { P _ { Z } ( x ) = \sum _ { i = 0 } ^ { M - 1 } \gamma _ { i } r _ { i } ( x ) } \end{array}$ . Here, we would want P to be as small as possible so that we are closer to a sample from the well-known distribution. Our experimental results suggest that the viability of this approach mainly depends on how informative the distortion measure is with respect to the source observation and how informative the mixture $r ( x )$ is with respect to the decision regions of the classifier.

## 6 NUMERICAL RESULTS

In this section, we present numerical results on discrete alphabets for a binary hypothesis test $( M =$ $2 )$ . The exact derivations of our rate function for Bernoulli sources can be found in Appendix D. In here, we consider a quaternary alphabet with a binary hypothesis test and evaluate how our rate and error probability vary at different distortion levels measured using the Hamming distortion. For these experiments, we set $\mathcal { X } = \{ 0 , 1 , 2 , 3 \}$ and the prior probabilities to $\pi _ { 0 } = \sigma _ { 0 } = 0 . 5$ . The optimal decision regions $R _ { i } ^ { * }$ at the transmitter side is obtained through MAP rule and we consider that the test at the receiver is also MAP but its decision regions $R _ { i }$ are obtained following the distributions of $q _ { i } ( x ) \mathbf { s }$ . The distributions $p _ { i } ( x )$ and $q _ { i } ( x )$ for $i = 0 ,$ 1 along with the corresponding decision regions are presented analytically in Appendix E.

In the first experiment, we consider information Level 1 and look at the RDME problem. Here, we consider how the rate varies with distortion at different perception levels enforced in (7). As illustrated in Fig. 2, as we increase the perception constraint ${ \hat { P } } ,$ the rate curves tend to converge and finally settle on the rate curve obtained by directly solving (6) as claimed. In the next experiment we operate at Level 2 and consider the following two scenarios: (i) We transmit the side information J losslessly and try to minimize the additional bits required for the specific reconstruction. (ii) We do not transmit J explicitly, but instead try to minimize the overall rate. For these two scenarios, we examine how the rate varies with distortion for different perception levels and how probability of error varies with perception at different distortion levels. Here, for simplicity, we use $r _ { i } ( x ) = q _ { i } ( x )$ and $\gamma _ { i } = \sigma _ { i }$ for $i = 0 , 1$ . This essentially means that we are matching to the training distribution of the test at the receiver. However, this does not mean that we know the decision regions at the receiver since the test could be different (e.g., Neyman-Pearson test instead of MAP). As depicted in Fig. 2a, transmitting J losslessly induces higher rates for a given perception and distortion level. However, as illustrated in Fig. 2b, transmitting J lowers the error probability as well. Moreover, it can be observed that for a given distortion level, as we lower the perception constraint, the probability of error approaches the limit of our bound (small $\hat { P } )$ in Corollary 1. This shows that for a given distortion level, it is better to operate at the tightest possible perception level that is feasible to minimize the classification error.

Next, we evaluate how rate and probability of error vary when we operate at Level 1 and Level 2. Here, the relationship between $P _ { \alpha }$ and α is defined using $P _ { \alpha } = 2 \bar { ( } \alpha - p _ { e } ^ { * } - C r _ { e } ) ^ { 2 }$ . As shown in Fig. 3a and Fig. 3b, when we operate at Level 1, we obtain lower rates compared to Level 2. However, at Level 1, we observe that this lower rate is mostly achieved when the constraint on the probability of error is satisfied with equality thus resulting in higher error whereas in Level 2, we always observe that the obtained error probability is much smaller compared to that of Level 1. This is because at Level $^ { 2 , }$ we are using an upper bound on error probability to minimize the rate. Though this limits the feasible distortion levels, this is a viable approach when the exact decision regions are unknown. Additionally, it is interesting to see that the best error rate obtained in Level 2 is attained when the distortion constraint is not too tight so as to make the problem infeasible and when it is not too relaxed so as not to convey any useful information.

## 7 SIMULATIONS ON NEURAL CLASSIFIERS

In this section, we move from discrete alphabets to a setting where the test at the receiver is a neural classifier and the decision regions $R _ { i }$ are not available in closed-form. This is precisely the situation described in Section 3 for which Level 2 was introduced: We do not know $R _ { i }$ , but we know a distribution on which the test performs well.

We look at the digit classification problem and consider $M = 1 0$ hypotheses corresponding to the ten digit classes. The source $X$ is drawn from SVHN dataset (Netzer et al., 2011) and the distribution $r _ { i } ( x )$ for which the receiver’s test was designed is the distribution of MNIST images (LeCun et al., 1998) belonging to class $i ,$ both rendered as $3 2 \times 3 2$ RGB images so that the two share common support . We set $\pi _ { i } = \sigma _ { i } = \gamma _ { i } = 1 / M$ for all i, so that the domain mismatch is entirely a mismatch of the distributions within each class and not of the priors. The test at the receiver is a ResNet-9 classifier trained only on MNIST, its error on its own domain is $r _ { e } = 0 . 0 0 7 7$ . Since the transmitter has no access to this classifier, the decision regions $R _ { i }$ are unknown to the compression scheme, and neither the classifier nor its gradients are ever used while designing the encoder and decoder. The estimate $J$ of the hypothesis is produced at the transmitter by a second ResNet-9 classifier trained only on SVHN, whose accuracy is 0.921. This classifier plays the role of the optimal test at the transmitter and yields $p _ { e } ^ { * } = 0 . 0 7 9 0$ and $C = 1 . 0 8 6 7$ . Details of both classifiers and of the compression architecture are given in Appendix F.

Two features of this setting are worth emphasizing before looking at the results. First, the mismatch is severe: Applying the receiver’s test directly to X yields an error probability of 0.8930, which is close to $1 - 1 / M = 0 . 9$ . The decision regions of the receiver are therefore bad for the source, which is the regime in which Theorem 1 predicts that matching onto $r _ { i }$ should help. Second, unlike the discrete examples of Section 3, here we cannot solve (10) directly. We therefore adopt the standard approach of (Blau & Michaeli, 2019) and trace the achievable region rather than impose a constraint: We fix the rate by construction and sweep the Lagrangian

![](images/cce8558156a8d8e96ecc1032bb5162981a6d4b65a72de5681eb99f7b828ded9c.jpg)

![](images/8eb35414bc2ff345fe3a20be576e7b9985b8e1d2b5a8eddc933a2e7203746a24.jpg)  
Figure 2: Variation of $R _ { P } ( D )$ vs distortion with varying perception levels $P$ for the RDME problem.  
(a) rate vs distortion

![](images/3e6e0bb3826ba80f46881386f23a391966810a228a94bc1bf9d0823c591d157a.jpg)  
(b) $p _ { e } ( \hat { X } )$ vs perception

Figure 3: Variation of rate and $p _ { e } ( { \hat { X } } )$ when transmitting side information J separately versus not transmitting J.  
![](images/769fd3fbb0424988c2c19ff8ff5d84e81b8108ae3ba4aaebde7829c6c955e9ae.jpg)  
(a) rate

![](images/f8caf39afd170c5ee81d5517f0073cfe77b74bdd06be39b565900272b10786c5.jpg)  
(b) p<sub>e</sub>(X<sup>ˆ</sup> )  
Figure 4: Variation of rate and $p _ { e } ( \hat { X } )$ vs distortion at information Levels 1 and 2.

$$
\operatorname* { m i n } _ { \mathrm { e n c , d e c } } \quad \sum _ { i = 0 } ^ { M - 1 } P _ { i } \mathbb { E } [ d ( X _ { i } , \hat { X } _ { i } ) ] + \lambda \sum _ { i = 0 } ^ { M - 1 } P _ { i } \delta \Big ( P _ { \hat { X } _ { i } } , P _ { Y _ { i } } \Big ) ,\tag{12}
$$

where $\delta ( \cdot , \cdot )$ is a divergence estimated adversarially and $\lambda \geq 0$ selects an operating point on the distortion-perception tradeoff. The rate is fixed rather than penalized because the bottleneck is a vector quantizer with L tokens over a codebook of K codewords, which gives exactly $R = L \log _ { 2 } K$ bits per image irrespective of the weights. The side information J costs $H ( J ) = \log _ { 2 } M$ bits when it is transmitted. Sweeping λ at each $( L , K )$ therefore produces a family of achievable $( R , D , P , p _ { e } )$ tuples rather than a single constrained solution, and every point in the plots below is an operating point that was actually realized by a trained encoder-decoder pair. In this section, we opt to use the Wasserstein 1-distance $( W _ { 1 } )$ ) instead of the total variation (TV) distance to quantify perception. This is because the $W _ { 1 }$ distance can be expressed as a functional optimization problem using the Kantorovich-Rubenstein duality leading to the WGAN architecture which is well-known to work well for image generation tasks. Even though TV distance can be equivalently expressed as a func tional optimization problem in the space of bounded functions, we see that the smoothness constraint enforced in the Kantrovich duality of the $W _ { 1 }$ distance is necessary to generate good reconstructions from the target distribution.

To evaluate the interplay between distortion, perception and classification error, we consider the two configurations for Level 2 described in Section 3, where in the first configuration we send J losslessly and in the second we do not. As illustrated in Fig. 5 and Fig. 6, when the rate increases our rate distortion curves tend to move towards the low distortion and high perception regime for both configurations. Further, for both configurations, we see that the classification error decreases significantly when the distortion constraint is not too tight and when the perception constraint is binding. This aligns with our theoretical insights and further validates our claim that significant gains in terms of classification accuracy can be obtained when matching on to a target distribution rather than the source distribution in this setting. Moreover, we also see that transmitting J lossessly further enhances our classifications accuracy at the cost of only $H ( J ) \approx 3 . 3 2$ bits.

![](images/7e17309b5bf28d907e9d416b700579f6975b882433584323073796339db0552f.jpg)  
(a) perception vs distortion

![](images/3891239f2c2eb1ac304f4248f0eff22a993500e6ee2d5fd0e5499bf65562c8b3.jpg)  
(b) error vs distortion

![](images/0befcd0e7bec459f4586e0be3b6e3864043638d5a7601acb3c636568fbbfb93d.jpg)  
(c) error vs perception

Figure 5: Variation of distortion, perception and classification error at different rates (true rate R = $R ^ { \prime } + H ( J ) )$ for Level 2 with lossless transmission of J.  
![](images/dd6136257b2309f69af6e4fdb69e36779d2090a990772378618e969943b878f6.jpg)  
(a) perception vs distortion

![](images/283736ea8360500ed73266ff8455828b47f3cc07787f71200207d5cd18bdf3b0.jpg)  
(b) error vs distortion

![](images/c1dc8e821864388c834ed5b06612e8ad2f72e8d290e5e624c2f72fa3b1ae7ca7.jpg)  
(c) error vs perception

Figure 6: Variation of distortion, perception and classification error at different rates for Level 2 without explicitly transmitting J.  
![](images/571720a72b8e143350bbe10bfea1cbd56485dedc0426444b75d6278931c76264.jpg)  
(a) perception vs distortion

![](images/075074175af17b401c4d2b0ba6521ad2069ca5e7b3a8ba2eaa178fc15cf606b3.jpg)  
(b) error vs distortion  
Figure 7: Variation of distortion, perception and classification error at different rates for Level 3.

Next, we consider the setting described in Level 3 where we try to match on to the mixture distribution. For this scenario, we see an interesting phenomenon: In Fig. 7, when the distortion constraint is not too tight, we see that lower perception values lead to lower classification error. However, when the distortion constraint becomes too relaxed, lower perception yields higher classification errors. This is due to the fact that when the distortion constraint is very relaxed, it does not convey any useful information. Therefore, the reconstruction is generated from the target distribution independent of the source image. The reconstructions for Levels 2 and 3 are presented in Appendix G.

## 8 CONCLUSIONS

In this work, we investigated rate-minimal compression schemes that simultaneously enable us to leverage an ill-trained classifier to our advantage. We showed that perception constraints naturally emerge out of the classification constraints defined in the problem. Through theory and extensive simulations, we showed that under this setting, matching on to a target distribution enhances classification accuracy as opposed to reconstructing similar to the source distribution.

## AI USE STATEMENT

LLMs were used for literature review and as a programming assistant for implementing experimental code according to specifications and methodological choices provided by the authors. All AI-assisted outputs were reviewed and verified by the authors. LLMs were not used to derive or verify the proofs, generate synthetic datasets, or determine the scientific conclusions of the work. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

Code, configurations and scripts that generate figures are available at https://anonymous.   
4open.science/r/Task-Aware-Compression.

Theory. All assumptions and the complete proofs which are in Appendixes are self-contained and no result relies on computation.

Discrete experiments (Section 6). The RDP functions are obtained by solving convex programs directly on the quaternary and binary alphabets, with scipy.optimize over a grid of $( \bar { D } , P )$ pairs. The three problem builders differ only in the constraints. The source and test distributions and both sets of decision regions are given in Table 1, the priors are $\pi _ { 0 } = \sigma _ { 0 } = 0 . 5$ , and the distortion is the Hamming distance. We release the solver together with the Streamlit interface used to produce the figures, which shows the distributions, the distortion, the $( D , P )$ grid and the perception levels as controls and plots the curves.

Image experiments (Section 7). Both datasets are the standard torchvision distributions of SVHN and MNIST. Each run writes the achieved operating point of every trained config as a CSV rate, distortion, perception, $p _ { e } ( \hat { X } )$ , the two components of the distortion, and the constraints Section 7 is reproduced by:

python experiments.py --mode rate-sweep \   
--family digits\_rgb --tx svhn --rx mnist \   
--resolution 32 --uniform-priors \   
--arch resnet9 --clf-epochs 100 \   
--clf-batch 512 --clf-lr 0.002 \   
--perception gan --distortion mse+feature \   
--feature-weight 1.0 \   
--feature-source transmitter\_classifier \   
--feature-layer deep \   
--lk-pairs "(8,32),(8,64),(16,16),(16,64),(32,16)" \   
--lambdas "0,1,5,10,20,35,50,75,90,100,200,300" \   
--lambda-scaling balanced \   
--latent-dim 128 --n-down 2 --epochs-per-point 300 \   
--batch 800 --eval-batch 800 --eval-critic-steps 300\   
--decoder-noise 0.5 --seeds 4 \   
--configs sendJ\_cond\_PYj,sendJ\_cond\_PXj,noJ\_cond\_PYj,noJ\_cond\_PXj \   
--save-models --out results/

with the Level 3 configuration noJ uncond PY run separately. Every point is checkpointed as it completes, so the sweep resumes rather than restarts, and the two classifiers are trained once and reused across all configurations from a cache keyed by their configuration hash. The runs reported here used NVIDIA A100 GPU and took approximately 80 GPU-hours in total.

What is and is not deterministic. The data splits, the class sampling and all weight initializations are seeded $( { \tt s p l i t . s e e d = 0 }$ , four compressor seeds per operating point), and the classifier cache makes g and g same across configurations. First, GPU output being non-deterministic means a rerun on different hardware will not reproduce the numbers exactly. We report the mean over four seeds and the figures show the mean points. Second, the perception term is an adversarial estimate and is therefore non-stationary. It depends on how well the critic has been trained, which is why the critic is trained at every step including λ = 0 and refit for a further 300 steps with the frozen weights before the value is recorded.

## REFERENCES

S. Akkoc, S. Liyanaarachchi, S. Ulukus, and A. Yener. The rate-distortion-deception tradeoff. Available online at arXiv:2607.25997.

E. Arda and A. Yener. A rate-distortion framework for summarization. In IEEE ISIT, June 2025a.

E. Arda and A. Yener. Rate-distortion-perception trade-off in summarization. In Allerton Conference, September 2025b.

Y. Blau and T. Michaeli. Rethinking lossy compression: The rate-distortion-perception tradeoff. In ICML, June 2019.

I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville. Improved training of wasserstein gans. In NIPS, December 2017.

Y. Hamdi and D. Gund¨ uz. The rate-distortion-perception trade-off with side information. In¨ IEEE ISIT, June 2023.

Y. Hamdi, A. B. Wagner, and D. Gund ¨ uz. Rate-distortion-perception trade-off with strong realism¨ constraints: Role of side information and common randomness. IEEE Transactions on Information Theory, 72(9):6401–6421, September 2026.

Y. LeCun, C. Cortes, and C. J. C. Burges. The mnist database of handwritten digits. Website, 1998. URL http://yann.lecun.com/exdb/mnist/.

C. T. Li and A. E. Gamal. Strong functional representation lemma and applications to coding theorems. IEEE Transactions on Information Theory, 64(11):6967–6978, November 2018.

H. Liu, G. Zhang, J. Chen, and A. Khisti. Cross-domain lossy compression as entropy constrained optimal transport. IEEE Journal on Selected Areas in Information Theory, 3(3):513–527, September 2022.

E. MolavianJazi and A. Yener. Two-way lossy compression via a relay with self source. In IEEE ISIT, July 2016.

Y. Netzer, T. Wang, A. Coates, A. Bissacco, B. Wu, and A. Y. Ng. Reading digits in natural images with unsupervised feature learning. In NIPS Workshop on Deep Learning and Unsupervised Feature Learning, December 2011. URL http://ufldl.stanford.edu/housenumbers/.

N. Nguyen, T. Nguyen, T. Nguyen, and B. Bose. Universal rate-distortion-classification representations for lossy compression. In IEEE ITW, September 2025.

N. Nguyen, T. Nguyen, and B. Bose. Cross-domain lossy compression via rate- and classificationconstrained optimal transport. In ICLR, January 2026.

J. Qian, S. Salehkalaibar, J. Chen, A. Khisti, W. Yu, W. Shi, Y. Ge, and W. Tong. Rate-distortionperception tradeoff for gaussian vector sources. IEEE Journal on Selected Areas in Information Theory, 6:1–17, November 2025.

C. E. Shannon. A mathematical theory of communication. The Bell System Technical Journal, 27 (3):379–423, July 1948.

L. Theis and A. B. Wagner. A coding theorem for the rate-distortion-perception function. In ICLR Neural Compression Workshop, April 2021.

R. Timo, T. Chan, and A. Grant. Rate distortion with side-information at many decoders. IEEE Transactions on Information Theory, 57(8):5240–5257, August 2011.

A. B. Wagner. The rate-distortion-perception tradeoff: The role of common randomness. Available online at arXiv:2202.04147.

Y. Wang, Y. Wu, S. Ma, and Y-J. A. Zhang. Lossy compression with data, perception, and classification constraints. In IEEE ITW, November 2024.

Y. Wang, Y. Wu, S. Ma, and Y. A. Zhang. Task-oriented lossy compression with data, perception, and classification constraints. IEEE Journal on Selected Areas in Communications, 43(7):2635– 2650, July 2025.

A. Wyner and J. Ziv. The rate-distortion function for source coding with side information at the decoder. IEEE Transactions on Information Theory, 22(1):1–10, January 1976.

H. Yamamoto. Source coding theory for cascade and branching communication systems. IEEE Transactions on Information Theory, 27(3):299–308, May 1981.

G. Zhang, J. Qian, J. Chen, and A. Khisti. Universal rate-distortion-perception representations for lossy compression. IEEE Transactions on Information Theory, 71(11):8633–8653, November 2025.

Y. Zhang. A Rate-Distortion-Classification approach for lossy image compression. Digital Signal Processing, 141, July 2023.

## A PROOF OF PROPOSITION 1

The converse and achievability proofs follow similar to that of (Akkoc et al.). We have used common randomness in this setup to make the achievability proof simpler and to facilitate the use of stochastic encoders and decoders when perception constraints are involved.

## A.1 CONVERSE

From (3), we have that the $p _ { e } ( { \hat { X } } )$ is linear with respect $p _ { \hat { X } | X }$ . Further, we have that $I ( { \hat { X } } ; X )$ is convex and $\mathbb { E } [ d ( X , { \hat { X } } ) ]$ is linear with respect to $p _ { \hat { X } | X }$ . Hence, the set $S = \{ ( D , \alpha ) : S _ { D , \alpha } \neq \emptyset \}$ is convex and $R ( D , \alpha )$ is convex on S. Additionally, the monotonicity of $R ( D , \alpha )$ follows from the fact $S _ { D , \alpha } \subseteq S _ { D ^ { \prime } , \alpha ^ { \prime } }$ for $D \leq D ^ { \prime }$ and $\alpha \leq \alpha ^ { \prime }$ . Now, suppose $( R , D , \alpha )$ is achievable and fix $\epsilon > 0$ Therefore, n and functions $f ( \cdot )$ and $g ( \cdot )$ such that,

$$
n ( R + \epsilon ) > H ( K | U ) \ge I ( X ^ { n } ; K | U ) = I ( X ^ { n } ; K , U )\tag{13}
$$

$$
\ge I ( X ^ { n } ; \hat { X } ^ { n } ) \ge \sum _ { i = 1 } ^ { n } I ( X _ { n , i } ; \hat { X } ^ { n } | X ^ { i - 1 } )\tag{14}
$$

$$
= \sum _ { i = 1 } ^ { n } I ( X _ { n , i } ; { \hat { X } } ^ { n } , X ^ { i - 1 } ) \geq \sum _ { i = 1 } ^ { n } I ( X _ { n , i } ; { \hat { X } } _ { n , i } )\tag{15}
$$

We can construct single-letter stochastic mappings that achieve the rate $I ( X _ { n , i } ; { \hat { X } } _ { n , i } )$ with distortion $D _ { i } = \mathbb { E } [ d ( X _ { n , i } , \hat { X } _ { n , i } ) ]$ and probability of error $p _ { e } ^ { ( i ) } = p _ { e } ( \hat { X } _ { n , i } )$ , by isolating only the ith term of the above encoder-decoder scheme defined using $f ( \cdot )$ and $g ( \cdot )$ . Therefore, based on the convexity of S, we have that $( \tilde { D } , \tilde { \alpha } ) \in S$ , where $\begin{array} { r } { \tilde { D } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } D _ { i } } \end{array}$ and $\begin{array} { r } { \tilde { \alpha } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } p _ { e } ^ { ( i ) } } \end{array}$ . Hence, $S _ { \tilde { D } , \tilde { \alpha } } \neq \emptyset$ Further, since $\tilde { D } \le D$ and $\tilde { \alpha } \leq \alpha$ , it follows that $S _ { \tilde { D } , \tilde { \alpha } } \subseteq S _ { D , { \alpha } }$ and thereby $S _ { D , \alpha } \neq \emptyset$ . Now, from the convexity of $R ( D , \alpha )$ , we have that

$$
n ( R + \epsilon ) > \sum _ { i = 1 } ^ { n } I ( X _ { n , i } ; \hat { X _ { n , i } } ) \geq n \sum _ { i = 1 } ^ { n } \frac { 1 } { n } R ( D _ { i } , p _ { e } ^ { ( i ) } ) \geq n R ( \tilde { D } , \tilde { \alpha } ) \geq n R ( D , \alpha ) .\tag{16}
$$

Since this is true for any $\epsilon > 0$ , we have $R \geq R ( D , \alpha )$

## A.2 ACHIEVABILITY

The achievability follows from the strong function representation lemma (SFRL) (Li & Gamal, 2018). Suppose $R ( D , \alpha )$ is well-defined and let $\epsilon > 0$ . Then, there exists a mapping $p _ { \hat { X } | X }$ and thereby a $\hat { X }$ such that the distortion and probability of error constraints are satisfied with $I ( X ; { \hat { X } } ) <$

$\begin{array} { r } { R ( D , \alpha ) + \frac { \epsilon } { 2 } } \end{array}$ . Now, from SFRL there exists a random variable U independent of $X ^ { n }$ and functions $f ( \cdot )$ and $g ( \cdot )$ such that $K = f ( X ^ { n } , U ) , \hat { X } ^ { n } = g ( K , U )$ and

$$
H ( K | U ) < I ( X ^ { n } ; { \hat { X } } ^ { n } ) + \log \Bigl ( I ( X ^ { n } ; { \hat { X } } ^ { n } ) + 1 \Bigr ) + 5\tag{17}
$$

$$
= n I ( X ; { \hat { X } } ) + \log \Bigl ( n I ( X ; { \hat { X } } ) + 1 \Bigr ) + 5 .\tag{18}
$$

Therefore, by selecting n large enough such that $\frac { \log \left( n I ( X ; \hat { X } ) + 1 \right) + 5 } { n } \leq \frac { \epsilon } { 2 }$ , we have that

$$
\frac { H ( K | U ) } { n } < I ( X ; \hat { X } ) + \frac { \epsilon } { 2 } < R ( D , \alpha ) + \epsilon .\tag{19}
$$

Therefore, any $( R , D , \alpha )$ is achievable for any $R \geq R ( D , \alpha )$

## B PROOF OF LEMMA 1

Using (3), we have that

$$
p _ { e } ( \hat { X } ) = 1 - \sum _ { i = 0 } ^ { M - 1 } \int _ { \mathcal { X } } \pi _ { i } p _ { i } ( x ) \int _ { R _ { i } } p _ { \hat { x } | x } \mathrm { d } \hat { x } \mathrm { d } x\tag{20}
$$

$$
= 1 - \sum _ { i = 0 } ^ { M - 1 } \sum _ { j = 0 } ^ { M - 1 } \int _ { R _ { j } ^ { * } } \pi _ { i } p _ { i } ( x ) \int _ { R _ { i } } p _ { \hat { x } | x } \mathrm { d } \hat { x } \mathrm { d } x\tag{21}
$$

$$
= 1 - \sum _ { j = 0 } ^ { M - 1 } \int _ { R _ { j } ^ { * } } \sum _ { i = 0 } ^ { M - 1 } \pi _ { i } p _ { i } ( x ) \int _ { R _ { i } } p _ { \hat { x } | x } \mathrm { d } \hat { x } \mathrm { d } x\tag{22}
$$

$$
= 1 - \sum _ { j = 0 } ^ { M - 1 } \int _ { R _ { j } ^ { * } } \left( \pi _ { j } p _ { j } ( x ) + \sum _ { i = 1 , i \neq j } l _ { j , i } ( x ) \int _ { R _ { i } } p _ { \hat { x } | x } \mathrm { d } \hat { x } \right) \mathrm { d } x ,\tag{23}
$$

where $l _ { j , i } ( x ) = \pi _ { i } p _ { i } ( x ) - \pi _ { j } p _ { j } ( x )$ . Now, note that in the region $R _ { i } ^ { * }$ we have $l _ { j , i } ( x ) < 0$ . Therefore, to minimize the error probability, our mapping must satisfy (4). Now, substituting these relations back on to (21), we obtain

$$
p _ { e } ( \hat { X } ) = 1 - \sum _ { j = 0 } ^ { M - 1 } \int _ { R _ { j } ^ { * } } \pi _ { j } p _ { j } ( x ) \mathrm { d } x = p _ { e } ^ { * } .\tag{24}
$$

## C PROOFS OF THEOREM 1 AND COROLLARY 1

First, note that we can represent $r _ { e }$ as

$$
r _ { e } = \sum _ { i = 0 } ^ { M - 1 } \int _ { \bar { R } _ { i } } \gamma _ { i } r _ { i } ( x ) , \mathrm { d } x\tag{25}
$$

Further, note that

$$
P _ { \hat { X } _ { j } } ( \hat { x } ) = \frac { 1 } { P _ { j } } \int _ { R _ { j } ^ { * } } p _ { \hat { x } | x } \sum _ { i = 0 } ^ { M - 1 } \pi _ { i } p _ { i } ( x ) \mathrm { d } x .\tag{26}
$$

Now, error of the statistical test satisfies

$$
\begin{array} { l } { { \displaystyle p _ { e } \big ( \hat { X } \big ) = \sum _ { i = 0 } ^ { M - 1 } \int _ { \hat { X } } \pi _ { i } p _ { i } ( x ) \int _ { \bar { R } _ { i } } p _ { \hat { x } \mid x } \mathrm { d } \hat { x } \mathrm { d } x } } \\ { { \displaystyle \quad \quad = \sum _ { i = 0 } ^ { M - 1 } \int _ { \bar { R } _ { i } ^ { * } } \pi _ { i } p _ { i } ( x ) \int _ { \bar { R } _ { i } } p _ { \hat { x } \mid x } \mathrm { d } \hat { x } \mathrm { d } x + \sum _ { i = 0 } ^ { M - 1 } \int _ { R _ { i } ^ { * } } \pi _ { i } p _ { i } ( x ) \int _ { \bar { R } _ { i } } p _ { \hat { x } \mid x } \mathrm { d } \hat { x } \mathrm { d } x } } \end{array}\tag{27}
$$

(28)

$$
\leq \sum _ { i = 0 } ^ { M - 1 } \int _ { \bar { R } _ { i } ^ { * } } \pi _ { i } p _ { i } ( x ) \mathrm { d } x + \sum _ { i = 0 } ^ { M - 1 } \int _ { \bar { R } _ { i } } \int _ { R _ { i } ^ { * } } \pi _ { i } p _ { i } ( x ) p _ { \hat { x } | x } \mathrm { d } \hat { x } \mathrm { d } x\tag{29}
$$

$$
\leq p _ { e } ^ { * } + \sum _ { i = 0 } ^ { M - 1 } P _ { i } \int _ { \hat { R } _ { i } } P _ { \hat { X } _ { i } } ( \hat { x } ) \mathrm { d } \hat { x }\tag{30}
$$

$$
= p _ { e } ^ { * } + \sum _ { i = 0 } ^ { M - 1 } P _ { i } \int _ { \bar { R } _ { i } } r _ { i } ( \hat { x } ) \mathrm { d } \hat { x } + \sum _ { i = 0 } ^ { M - 1 } P _ { i } \left( \int _ { \bar { R } _ { i } } P _ { \hat { X } _ { i } } ( \hat { x } ) \mathrm { d } \hat { x } - \int _ { \bar { R } _ { i } } r _ { i } ( \hat { x } ) \mathrm { d } \hat { x } \right)\tag{31}
$$

$$
\leq p _ { e } ^ { * } + C \sum _ { i = 0 } ^ { M - 1 } \gamma _ { i } \int _ { \bar { R } _ { i } } r _ { i } ( \hat { x } ) \mathrm { d } \hat { x } + \sum _ { i = 0 } ^ { M - 1 } P _ { i } \| P _ { \hat { X } _ { i } } - r _ { i } \| _ { T V }\tag{32}
$$

$$
\leq p _ { e } ^ { * } + C r _ { e } + \sum _ { i = 0 } ^ { M - 1 } P _ { i } \sqrt { \frac { 1 } { 2 } D ( P _ { \hat { X } _ { i } } | | r _ { i } ) }\tag{33}
$$

$$
\leq p _ { e } ^ { * } + C r _ { e } + \sum _ { i = 0 } ^ { M - 1 } \sqrt { P _ { i } } \sqrt { \frac { P _ { i } } { 2 } D ( P _ { \hat { X } _ { i } } | | r _ { i } ) }\tag{34}
$$

$$
\leq p _ { e } ^ { * } + C r _ { e } + \frac { 1 } { \sqrt 2 } \left( \sum _ { i = 0 } ^ { M - 1 } P _ { i } D ( P _ { \hat { X } _ { i } } | | r _ { i } ) \right) ^ { \frac 1 2 } ,\tag{35}
$$

where (33) follows from Pinsker’s inequality and (35) follows from Cauchy-Schwarz inequality.

## D RDE FUNCTION FOR A BERNOULLI SOURCE

We solve the RDE function

$$
\begin{array} { r l } { R ( D , \alpha ) = \underset { p _ { \hat { X } | X } } { \mathrm { m i n } } } & { I ( X ; \hat { X } ) } \\ { \mathrm { s . t . } } & { \mathbb { E } [ d ( X , \hat { X } ) ] \leq D } \\ & { p _ { e } ( \hat { X } ) \leq \alpha , } \end{array}\tag{36}
$$

for a binary source. We assume

$$
\begin{array} { r } { H \in \{ 0 , 1 \} , \quad P ( H = 0 ) = \pi _ { 0 } , \quad P ( H = 1 ) = \pi _ { 1 } , } \end{array}\tag{37}
$$

where $\pi _ { 0 } + \pi _ { 1 } = 1$ . Conditioned on the hypothesis, let

$$
X | H = 0 \sim \operatorname { B e r n } ( a _ { 0 } ) , \quad X | H = 1 \sim \operatorname { B e r n } ( a _ { 1 } )\tag{38}
$$

Therefore, the marginal distribution of X is also Bernoulli, $X \sim \operatorname { B e r n } ( a )$ where $a = \pi _ { 0 } a _ { 0 } + \pi _ { 1 } a _ { 1 }$ We assume that $\hat { X } \in \{ 0 , 1 \}$ , and that the fixed classifier at the receiver decides ${ \hat { H } } = 0 \operatorname { i f } { \hat { X } } = 0$ and ${ \hat { H } } = 1 \operatorname { i f } { \hat { X } } = 1$ . For a binary reconstruction alphabet,

$$
u \triangleq P ( \hat { X } = 1 | X = 0 ) , \quad v \triangleq P ( \hat { X } = 1 | X = 1 ) ,\tag{39}
$$

where $0 \leq u , v \leq 1$ . The probability that the reconstruction equals one is $b \triangleq P ( { \hat { X } } = 1 ) =$ $( 1 - a ) u + a v$ . The mutual information is

$$
I ( X ; \hat { X } ) = H _ { b } ( b ) - ( 1 - a ) H _ { b } ( u ) - a H _ { b } ( v ) .\tag{40}
$$

where $H _ { b } ( \cdot )$ is the binary entropy function. Thus, the rate can be written directly as a function of u and v. We use Hamming distortion, $d ( x , { \hat { x } } ) = \mathbf { 1 } \{ x \neq { \hat { x } } \}$ . The average distortion is therefore

$$
\begin{array} { r l } & { d ( u , v ) = E [ d ( X , \hat { X } ) ] = P ( X \neq \hat { X } ) } \\ & { \qquad = P ( X = 0 ) P ( \hat { X } = 1 | X = 0 ) + P ( X = 1 ) P ( \hat { X } = 0 | X = 1 ) } \\ & { \qquad = ( 1 - a ) u + a ( 1 - v ) . } \end{array}\tag{41}
$$

(42)

(43)

Hence, the distortion constraint becomes $( 1 - a ) u + a ( 1 - v ) \leq D .$

An error occurs under $H = 0$ when $\hat { X } = 1$ , and under $H = 1$ when $\hat { X } = 0$ . Hence, $p _ { e } = \pi _ { 0 } P ( \hat { X } =$ $1 | H = 0 ) + \pi _ { 1 } P ( \hat { X } = 0 | H = 1 )$ . For $H = 0 , P ( \hat { X } = 1 | H = 0 ) = ( 1 - a _ { 0 } ) u + a _ { 0 } v$ . Similarly, $P ( \hat { X } = 0 | H = 1 ) = ( 1 - a _ { 1 } ) ( 1 - u ) + a _ { 1 } ( 1 - v )$ . Therefore,

$$
p _ { e } ( u , v ) = \pi _ { 0 } [ ( 1 - a _ { 0 } ) u + a _ { 0 } v ] + \pi _ { 1 } [ ( 1 - a _ { 1 } ) ( 1 - u ) + a _ { 1 } ( 1 - v ) ] .\tag{44}
$$

We write this as $p _ { e } ( u , v ) = \pi _ { 1 } + c _ { 0 } u + c _ { 1 } v$ , where $c _ { 0 } = \pi _ { 0 } ( 1 - a _ { 0 } ) - \pi _ { 1 } ( 1 - a _ { 1 } )$ and $c _ { 1 } = \pi _ { 0 } a _ { 0 } - \pi _ { 1 } a _ { 1 }$ The probability of error constraint is therefore simply $\pi _ { 1 } + c _ { 0 } u + c _ { 1 } v \leq \alpha$ . Notice that both the distortion constraint and the probability of error constraint are linear in u and v. Substituting the previous expressions, the original optimization problem becomes

$$
\begin{array} { r l } { R ( D , \alpha ) = \underset { 0 \leq u , v \leq 1 } { \operatorname* { m i n } } } & { H _ { b } \big ( ( 1 - a ) u + a v \big ) - ( 1 - a ) H _ { b } ( u ) - a H _ { b } ( v ) } \\ { \mathrm { s . t . } } & { ( 1 - a ) u + a ( 1 - v ) \leq D } \\ & { \pi _ { 1 } + c _ { 0 } u + c _ { 1 } v \leq \alpha . } \end{array}\tag{45}
$$

The mutual information is convex in the conditional distribution $p _ { \hat { X } | X }$ when the source distribution is fixed. Also, both constraints are linear. Therefore, this is a convex optimization problem. Let $\lambda \geq 0$ be the multiplier associated with the distortion constraint and let $\beta \geq 0$ be the multiplier associated with the probability of error constraint. The Lagrangian $L ( u , v , \lambda , \beta )$ is,

$$
\begin{array} { r l } & { L ( u , v , \lambda , \beta ) = H _ { b } \big ( ( 1 - a ) u + a v \big ) - ( 1 - a ) H _ { b } ( u ) - a H _ { b } ( v ) } \\ & { \qquad + \lambda \left[ ( 1 - a ) u + a ( 1 - v ) - D \right] + \beta \left[ \pi _ { 1 } + c _ { 0 } u + c _ { 1 } v - \alpha \right] . } \end{array}\tag{46}
$$

For the derivative calculation, we use natural logarithms, which only changes the scaling of λ and $\beta ,$ and does not change the optimal u and v. Define $\begin{array} { r } { \mathrm { l o g i t } ( x ) = \mathbf { \bar { \log { \frac { x } { 1 - x } } } } } \end{array}$ and $\begin{array} { r } { \sigma ( x ) \ = \ \frac { 1 } { 1 + e ^ { - x } } } \end{array}$ Let $b ~ = ~ ( 1 - a ) u + a v$ . For $0 ~ < ~ u , v ~ < ~ 1$ , differentiating the Lagrangian with respect to u gives l $\begin{array} { r } { \operatorname { o g i t } ( u ) \ = \ \operatorname { l o g i t } ( b ) - \lambda - \beta \frac { c _ { 0 } } { 1 - a } ; } \end{array}$ and differentiation with respect to v gives $\mathrm { l o g i t } ( v ) =$ $\begin{array} { r } { \mathrm { l o g i t } ( b ) + \lambda - \beta \frac { c _ { 1 } } { a } } \end{array}$ . Thus,

$$
u = \sigma \left( \mathrm { l o g i t } ( b ) - \lambda - \beta \frac { c _ { 0 } } { 1 - a } \right) ,\tag{47}
$$

$$
v = \sigma \left( \mathrm { l o g i t } ( b ) + \lambda - \beta \frac { c _ { 1 } } { a } \right) .\tag{48}
$$

The multipliers also satisfy $\lambda \left[ ( 1 - a ) u + a ( 1 - v ) - D \right] = 0 { \mathrm { ~ a n d ~ } } \beta \left[ \pi _ { 1 } + c _ { 0 } u + c _ { 1 } v - \alpha \right] = 0 .$

## ZERO RATE REGION

A possibility is $R ( D , \alpha ) = 0$ . The mutual information vanishes if and only if X and $\hat { X }$ are independent. For the binary channel considered here, this means $u = v = q , 0 \leq q \leq 1$ . For such a channel, $b = q$ and $I ( X ; { \hat { X } } ) = 0$ . The distortion becomes $d ( q ) = ( 1 - a ) q + a ( 1 - q ) = a + ( 1 - 2 a ) q .$ The classification error becomes $p _ { e } ( q ) = \pi _ { 1 } + ( c _ { 0 } + c _ { 1 } ) q .$ Since $c _ { 0 } + c _ { 1 } = \pi _ { 0 } - \pi _ { 1 }$ we obtain $p _ { e } ( q ) = \pi _ { 1 } + ( \pi _ { 0 } - \pi _ { 1 } ) q = \pi _ { 0 } q + \pi _ { 1 } ( 1 - q )$ . Therefore, $R ( D , \alpha ) = 0$ if and only if there exists a $q \in [ 0 , 1 ]$ such that $a + ( 1 - 2 a ) q \leq D$ and $\pi _ { 1 } + ( \pi _ { 0 } - \pi _ { 1 } ) q \leq \alpha$

In the absence of a distortion constraint, a zero-rate classifier can always choose a constant reconstruction corresponding to the more probable hypothesis. Hence, a zero-rate reconstruction satisfying only the error constraint exists whenever $\alpha \geq \operatorname* { m i n } \{ \pi _ { 0 } , \pi _ { 1 } \}$

## CASE 1: ONLY DISTORTION CONSTRAINT IS ACTIVE

Suppose $\lambda > 0 , \beta = 0$ . The problem then reduces to the Bernoulli RD problem. For $0 \le D <$ min $\left\{ a , 1 - a \right\}$ the optimal marginal is $\begin{array} { r } { b = \frac { a - D } { 1 - 2 D } } \end{array}$ . The optimal $u _ { D }$ and $v _ { D }$ are

$$
u _ { D } = \frac { D ( a - D ) } { ( 1 - a ) ( 1 - 2 D ) } , \quad v _ { D } = \frac { ( 1 - D ) ( a - D ) } { a ( 1 - 2 D ) } .\tag{49}
$$

They satisfy $( 1 - a ) u _ { D } + a ( 1 - v _ { D } ) = D$ . The corresponding rate is

$$
R _ { D } ( D ) = H _ { b } ( a ) - H _ { b } ( D ) , \quad 0 \leq D < \operatorname* { m i n } \{ a , 1 - a \} .\tag{50}
$$

This channel is also optimal for the RDE problem whenever it satisfies the classification-error constraint,

$$
\pi _ { 1 } + c _ { 0 } u _ { D } + c _ { 1 } v _ { D } \leq \alpha .\tag{51}
$$

Thus, whenever (51) holds, $R ( D , \alpha ) = H _ { b } ( a ) - H _ { b } ( D )$ . For $D \geq \operatorname* { m i n } \{ a , 1 - a \}$ , the Bernoulli RD function is zero.

## CASE 2: ONLY PROBABILITY OF ERROR CONSTRAINT IS ACTIVE

Now, suppose $\lambda = 0 , \beta > 0$ . Then, (48) become

$$
\mathrm { l o g i t } ( u _ { E } ) = \mathrm { l o g i t } ( b ) - \beta \frac { c _ { 0 } } { 1 - a } ,\tag{52}
$$

$$
\mathrm { l o g i t } ( v _ { E } ) = \mathrm { l o g i t } ( b ) - \beta \frac { c _ { 1 } } { a } .\tag{53}
$$

Define $\begin{array} { r } { t _ { 0 } \triangleq \exp \left( - \beta \frac { c _ { 0 } } { 1 - a } \right) } \end{array}$ and $t _ { 1 } \triangleq \exp \left( - \beta \frac { c _ { 1 } } { a } \right)$ . Then,

$$
u _ { E } = { \frac { b t _ { 0 } } { 1 - b + b t _ { 0 } } } , \quad v _ { E } = { \frac { b t _ { 1 } } { 1 - b + b t _ { 1 } } } .\tag{54}
$$

Using the relation $b = ( 1 - a ) u _ { E } + a v _ { E }$ , we have

$$
b = \frac { 1 - ( 1 - a ) t _ { 0 } - a t _ { 1 } } { ( 1 - t _ { 0 } ) ( 1 - t _ { 1 } ) } .\tag{55}
$$

This expression applies whenever the denominator is nonzero and is in (0, 1). In some cases where (55) is not found, b is obtained directly from the fixed-point equation $b = ( 1 - a ) u _ { E } + a v _ { E }$ . The Lagrange multiplier $\beta$ is selected so that the active error constraint satisfies $\pi _ { 1 } + c _ { 0 } u _ { E } + c _ { 1 } v _ { E } = \alpha$ Thus, this case reduces to a one-dimensional equation in $\beta .$ . The resulting solution is valid provided its distortion satisfies $( 1 - a ) u _ { E } + a ( 1 - v _ { E } ) \bar { \leq } D$ . Then, the rate is

$$
R ( D , \alpha ) = H _ { b } ( b ) - ( 1 - a ) H _ { b } ( u _ { E } ) - a H _ { b } ( v _ { E } ) .\tag{56}
$$

As $\beta$ increases, the optimal probability of error is non-increasing. Hence, the appropriate value of $\beta$ can be determined numerically using a one-dimensional monotone search.

## CASE 3: BOTH CONSTRAINTS ARE ACTIVE

Suppose $\lambda > 0 , \beta > 0$ . Then the complementary slackness condition gives $( 1 - a ) u + a ( 1 - v ) = D$ and $\pi _ { 1 } + c _ { 0 } u + c _ { 1 } v = \alpha$ . Equivalently $( 1 - a ) u - a v = D - a$ and $c _ { 0 } u + c _ { 1 } v = \alpha - \pi _ { 1 }$ . Then, let $\Delta \triangleq ( 1 - a ) c _ { 1 } + a c _ { 0 } , { \mathrm { i f ~ } } \Delta \neq 0 ,$ , (48) has a unique solution

$$
u ^ { \star } = \frac { c _ { 1 } ( D - a ) + a ( \alpha - \pi _ { 1 } ) } { \Delta } , \quad v ^ { \star } = \frac { ( 1 - a ) ( \alpha - \pi _ { 1 } ) - c _ { 0 } ( D - a ) } { \Delta } .\tag{57}
$$

The corresponding reconstruction probability is $b ^ { \star } = ( 1 - a ) u ^ { \star } + a v ^ { \star }$ , and therefore

$$
R ( D , \alpha ) = H _ { b } ( b ^ { \star } ) - ( 1 - a ) H _ { b } ( u ^ { \star } ) - a H _ { b } ( v ^ { \star } )\tag{58}
$$

Solution in (57) is admissible only if $0 \leq u ^ { \star } \leq 1$ and $0 \leq v ^ { \star } \leq 1$ . For an interior solution, the multipliers must also satisfy $\lambda > 0$ and $\beta > 0$ . Define $A \triangleq \operatorname { l o g i t } ( b ^ { \star } ) - \operatorname { l o g i t } ( u ^ { \star } )$ and $B \stackrel { \triangle } { = }$ $\mathrm { l o g i t } ( v ^ { \star } ) ^ { \star } - \mathrm { l o g i t } ( b ^ { \star } )$ . Then, $\begin{array} { r } { A = \lambda + \beta \frac { c _ { 0 } } { 1 - a } } \end{array}$ and $\begin{array} { r } { B = \lambda - \beta \frac { c _ { 1 } } { a } } \end{array}$ . Thus,

$$
\lambda = \log \mathrm { i t } ( b ^ { \star } ) - \log \mathrm { i t } ( u ^ { \star } ) - \beta \frac { c _ { 0 } } { 1 - a } , \beta = \frac { a ( 1 - a ) } { \Delta } \big [ 2 \log \mathrm { i t } ( b ^ { \star } ) - \log \mathrm { i t } ( u ^ { \star } ) - \log \mathrm { i t } ( v ^ { \star } ) \big ] .\tag{59}
$$

Therefore, the optimum solution is at the intersection of the constraint lines defined by $( 1 - a ) u -$ $a v \ : = \ : D \ : - \ : a$ and $c _ { 0 } u + c _ { 1 } v = \alpha - \pi _ { 1 }$ , as long as that intersecting point is inside the allowed $_ \mathrm { ~ 0 ~ t o ~ 1 ~ }$ range and the multipliers are nonnegative. If $\Delta = 0 , \mathrm { i . e . , ~ } \bar { \Delta } = ( 1 - a ) c _ { 1 } + a c _ { 0 } = 0 \nonumber$ the above constraint lines are parallel. If the equations $( 1 - a ) u - a v = D - a$ and $c _ { 0 } u + c _ { 1 } v =$ $\alpha - \pi _ { 1 }$ are inconsistent, the two constraints cannot be active simultaneously. If the two equations are proportional and consistent, the problem then reduces to minimizing the convex mutual information function along a single line segment. For example using the $( 1 - a ) u - a v = D - a$ , we get $\begin{array} { r } { v = \frac { ( 1 - a ) u + a - D } { a } } \end{array}$ . Then, we minimize

![](images/e2bbafc79e4b3bed7403ec3bc6bed536ccc81b80a408d82e429fa2d6f3f01272.jpg)  
Figure 8: Rate–distortion curves for different classification error constraints $\alpha \in \{ 0 . 2 , 0 . 3 , 0 . 4 \}$ with $X | H _ { 0 } \sim$ Bern(0.1), X $H _ { 1 } \sim \mathrm { B e r n } ( 0 . 8 )$ $\pi _ { 0 } = 0 . 6$ and $\pi _ { 1 } = 0 . 4$

$$
\begin{array} { r } { I _ { D } ( u ) \triangleq H _ { b } \Big ( ( 1 - a ) u + a \frac { ( 1 - a ) u + a - D } { a } \Big ) - ( 1 - a ) H _ { b } ( u ) - a H _ { b } \Big ( \frac { ( 1 - a ) u + a - D } { a } \Big ) } \end{array}\tag{60}
$$

over the interval $0 \leq u \leq 1$ and $\begin{array} { r } { 0 \leq \frac { ( 1 - a ) u + a - D } { a } \leq 1 } \end{array}$ . Since this is one dimensional convex optimization problem its minimum occurs either at unique stationary point or at an endpoint.

## FINAL CHARACTERIZATION

For arbitrary $\pi _ { 0 } , \pi _ { 1 } \geq 0$ where $\pi _ { 0 } + \pi _ { 1 } = 1$ and $1 \geq a _ { 0 } , a _ { 1 } \geq 0$ where $a = \pi _ { 0 } a _ { 0 } + \pi _ { 1 } a _ { 1 }$ . The RDE function can be determined as follows. If D and α are not in the feasible set, the $R ( D , \alpha )$ function is not defined. If there exists $q \in [ 0 , 1 ]$ such that $a + ( 1 - 2 a ) q \leq D$ and $\pi _ { 1 } + ( \pi _ { 0 } - \pi _ { 1 } ) q \leq \quad$ α then $R ( D , \alpha ) = 0$ . Otherwise $R ( D , \alpha ) > 0$ and the optimizer is obtained via

$$
( u ^ { * } , v ^ { * } ) = \left\{ \begin{array} { l l } { ( u _ { D } , v _ { D } ) , } & { \mathrm { i f } \lambda > 0 , \beta = 0 , } \\ { ( u _ { E } , v _ { E } ) , } & { \mathrm { i f } \lambda = 0 , \beta > 0 , } \\ { \left( \frac { c _ { 1 } ( D - a ) + a ( \alpha - \pi _ { 1 } ) } { \Delta } , \frac { ( 1 - a ) ( \alpha - \pi _ { 1 } ) - c _ { 0 } ( D - a ) } { \Delta } \right) , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{61}
$$

In all cases, once the optimal pair $( u ^ { * } , v ^ { * } )$ is found, the rate is

$$
R ( D , \alpha ) = H _ { b } ( ( 1 - a ) u ^ { * } + a v ^ { * } ) - ( 1 - a ) H _ { b } ( u ^ { * } ) - a H _ { b } ( v ^ { * } ) .\tag{62}
$$

Thus, for arbitrary $\pi _ { 0 } , \pi _ { 1 } , a _ { 0 } , a _ { 1 }$ , the RDE function is characterized by a finite set of convex optimization branches. The distortion only and both active branches admit closed-form solutions, while the error-only branch reduces to a one-dimensional equation for the multiplier $\beta .$

## E EXPERIMENTAL SETUP FOR SECTION 6

In this section, we present the distributions $p _ { i } ( x )$ and $q _ { i } ( x )$ used for the binary hypothesis experiments discussed in Section 6. We consider a quaternary alphabet on $\{ 0 , 1 , 2 , 3 \}$ and consider $p _ { 0 } ( x )$ to be a skewed distribution concentrated on 0, 1 whereas $p _ { 1 } ( x )$ is a skewed distribution concentrated on 2, 3 . MAP rule is applied on them to find the optimal decision regions $R _ { i } ^ { * }$ . These optimal decision regions yield that $p _ { e } ^ { * } = 0 . 1 2$ . We consider that the statistical test used at the receiver is also MAP. However, it was designed for using distributions $q _ { i } ( x )$ defined in Table 1. These decisions yield $r _ { e } = 0 . 0 3$ . As seen in Table 1, there is a clear mismatch between the optimal decision regions at the transmitter with that used at the receiver.

<table><tr><td rowspan=1 colspan=1>X</td><td rowspan=1 colspan=1> $\overline { { p _ { 0 } ( x ) } }$ </td><td rowspan=1 colspan=1> $p _ { 1 } ( x )$ </td><td rowspan=1 colspan=1> $\overline { { q _ { 0 } ( x ) } }$ </td><td rowspan=1 colspan=1> $\overline { { q _ { 1 } ( x ) } }$ </td><td rowspan=1 colspan=1>Tx DR</td><td rowspan=1 colspan=1>RxDR</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.3</td><td rowspan=1 colspan=1> $\overline { { R _ { 0 } ^ { * } } }$ </td><td rowspan=1 colspan=1> $\overline { { R _ { 1 } } }$ </td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1> $\overline { { R _ { 0 } ^ { * } } }$ </td><td rowspan=1 colspan=1> $\overline { { R _ { 0 } } }$ </td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.2</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1> $\overline { { R _ { 1 } ^ { * } } }$ </td><td rowspan=1 colspan=1> $\overline { { R _ { 1 } } }$ </td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.29</td><td rowspan=1 colspan=1> $\overline { { R _ { 1 } ^ { * } } }$ </td><td rowspan=1 colspan=1> $\overline { { R _ { 1 } } }$ </td></tr></table>

Table 1: Distributions and decision regions (DR).

## F ARCHITECTURE AND TRAINING PARAMETERS FOR THE EXPERIMENTS IN SECTION 7

## F.1 DATASETS AND CLASSIFIERS

Both domains are rendered as $3 2 \times 3 2$ RGB images: SVHN (Netzer et al., 2011) in its native resolution and MNIST (LeCun et al., 1998) replicated across the three channels and resized, so that X and Y share the support required by the problem formulation. We use 8000 training, 1500 validation and 3000 test images per domain, resampled so that the realized class frequencies match the uniform priors $\pi _ { i } = \sigma _ { i } = 1 / M$

The receiver’s test $g _ { Y }$ and the transmitter’s estimator $g _ { X }$ are both $\mathrm { R e s N e t - 9 }$ networks with base width 64 (6.57M parameters), trained with AdamW at a learning rate of $2 \times 1 0 ^ { - 3 }$ , weight decay $5 \times 1 0 ^ { - 4 }$ batch size 512, cosine schedule with warmup, label smoothing 0.05, dropout 0.05, random-crop and flip augmentation, for at most 100 epochs with early stopping on validation accuracy (patience 8). Each is trained on its own domain only: $g _ { Y }$ on MNIST under $\sigma _ { i } , g _ { X }$ on SVHN under $\pi _ { i } .$ . Their accuracies are reported in Table 2. Neither network is ever placed in the computational graph of the encoder or decoder, $g _ { Y }$ is used only to evaluate $p _ { e } ( \hat { X } )$ after training, and $g _ { X }$ is used only to produce J and the feature space of the distortion measure described below.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>train</td><td rowspan=1 colspan=1>validation</td><td rowspan=1 colspan=1>test</td></tr><tr><td rowspan=1 colspan=1>gx (SVHN)</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>0.938</td><td rowspan=1 colspan=1>0.921</td></tr><tr><td rowspan=1 colspan=1>gY (MNIST)</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>0.993</td><td rowspan=1 colspan=1>0.992</td></tr></table>

Table 2: Accuracy of the transmitter’s estimator and the receiver’s test, each on its own domain.

## F.2 ENCODER, DECODER AND RATE

The encoder and decoder are convolutional and are given in Table 3. The encoder maps a $3 2 \times 3 2 \times 3$ image to a 128-dimensional latent vector, which is split into L tokens of dimension 128/L and quantized against a learned codebook of $K$ entries by nearest-neighbor assignment, with straightthrough gradients and a commitment penalty of weight 0.25. The rate is therefore fixed by the geometry of the quantizer as $R = L \bar { \log _ { 2 } K }$ bits per image and does not have to be estimated or penalized. When J is transmitted, it is charged $H ( J ) = \log _ { 2 } M \approx 3 . 3 2$ bits, which is added to the rates reported in Section 7. The four rate points used are listed in Table 4. The transmitted J enters the network as a learned 32-dimensional embedding which modulates both the encoder and the decoder through FiLM layers whose output projection is initialized to zero, so that conditioning begins as the identity map. Following (Blau & Michaeli, 2019), uniform noise of half the code spacing is added to the quantized representation during training so that the decoder is stochastic, which is necessary for good perceptual quality at low rates.

## F.3 DISTORTION MEASURE

We use the combination of squared error and a deep feature distance of (Blau & Michaeli, 2019, Eqn. (11)):

$$
\Delta ( x , \hat { x } ) = \| x - \hat { x } \| ^ { 2 } + \alpha \| \Psi ( x ) - \Psi ( \hat { x } ) \| ^ { 2 } ,\tag{63}
$$

with $\alpha = 1$ . As in (Blau & Michaeli, 2019), the feature map $\Psi$ is an intermediate layer of a trained classifier rather than a generic network, because the term is only meaningful if the feature space carries semantic information. We take Ψ to be the next to last representation of the transmitter’s estimator $g _ { X }$ . We have two main reasons for doing this. First, g<sub>X</sub> is trained on the transmitter domain data that the transmitter owns, therefore, using it discloses nothing about the receiver’s test and keeps the scheme within Level 2 and 3. Second, the depth of the layer matters more than the accuracy of the classifier since we observed that measuring the ratio of the mean squared feature distance between images of different classes to that between images of the same class. A feature space is useful in (63) only if the distance reflects class differences to choose the layer we measured, for each candidate Ψ,

<table><tr><td>Encoder</td><td>Decoder</td></tr><tr><td> $\overline { { 3 2 \times 3 2 \times 3 } }$  Conv 4 × 4, stride 2: 16 × 16 × 32</td><td> ${ \overline { { L \log _ { 2 } K \ b i t s } } }$   $\mathrm { F C } \colon 1 2 8  2 5 6$ </td></tr><tr><td>Conv 4 × 4, stride  $2 \colon 8 \times 8 \times 6 4$ </td><td>FiLM(J)</td></tr><tr><td> $\operatorname { A v g P o o l : } 4 \times 4 \times 6 4$ </td><td> $\mathrm { F C } \colon 2 5 6 \to 8 \times 8 \times 6 4$ </td></tr><tr><td> $\bar { \mathrm { F C } } \colon 1 0 2 4  2 5 6$ </td><td> ${ \mathrm { C o n v T 4 \times 4 , s t r i d e 2 : 1 6 \times 1 6 \times 3 2 } }$ </td></tr><tr><td>FiLM(J) FC: 256 → 128</td><td> ${ \mathrm { C o n v T 4 \times 4 , s t r i d e 2 : 3 2 \times 3 2 \times 3 2 } }$  Conv  $3 \times 3 \colon 3 2 \times 3 2 \times 3 ,$  sigmoid</td></tr></table>

Table 3: Encoder and decoder architecture.
<table><tr><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>token dim.</td><td rowspan=1 colspan=1>R [bits]</td><td rowspan=1 colspan=1> $\overline { { R + H ( J ) } }$ [bits]</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>27.32</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>35.32</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>83.32</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>64</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>192</td><td rowspan=1 colspan=1>195.32</td></tr></table>

Table 4: Rate points. The rate is exact by construction, $R = L \log _ { 2 } K $

$$
\mathrm { s e p } ( \Psi ) = \frac { \mathbb { E } \big [ \| \Psi ( X ) - \Psi ( X ^ { \prime } ) \| ^ { 2 } \big ] } { \mathbb { E } \big [ \| \Psi ( X ) - \Psi ( X ^ { \prime \prime } ) \| ^ { 2 } \big ] } ,\tag{64}
$$

where X and $X ^ { \prime \prime }$ are drawn from the same class and X and $X ^ { \prime }$ from different classes. The ratio is not interpretable on its own, since raw pixel distance already separates classes to some extent. The identity function $\Psi ( x ) = x$ is therefore measured on the same images as baseline, and what matters is the ratio relative to it. Table 5 shows the results for $g _ { X }$ at 0.921 test accuracy. An early layer scores 1.003 against 0.099 for raw pixels, adding almost nothing over pixel distance, whereas the last layer representation scores 1.543. We therefore use the deepest convolutional representation.

<table><tr><td rowspan=1 colspan=1>feature map Ψ</td><td rowspan=1 colspan=1>dimension</td><td rowspan=1 colspan=1>sep(Ψ)</td><td rowspan=1 colspan=1>relative to pixels</td></tr><tr><td rowspan=1 colspan=1>raw pixels (identity)</td><td rowspan=1 colspan=1>3072</td><td rowspan=1 colspan=1>0.990</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>gx, early</td><td rowspan=1 colspan=1>65536</td><td rowspan=1 colspan=1>1.003</td><td rowspan=1 colspan=1>1.013</td></tr><tr><td rowspan=1 colspan=1> ${ \underline { { g x } } } .$ middle</td><td rowspan=1 colspan=1>32768</td><td rowspan=1 colspan=1>1.041</td><td rowspan=1 colspan=1>1.051</td></tr><tr><td rowspan=1 colspan=1>gx, penultimate (used)</td><td rowspan=1 colspan=1>8192</td><td rowspan=1 colspan=1>1.528</td><td rowspan=1 colspan=1>1.543</td></tr></table>

Table 5: Class separation (64) of each candidate feature space.

## F.4 PERCEPTION MEASURE

The divergence $\delta ( \cdot , \cdot )$ in (12) is the Wassertein-1 distance, estimated as in (Blau & Michaeli, 2019) by a critic trained with gradient penalty (Gulrajani et al., 2017)

$$
\delta ( P _ { \hat { X _ { i } } } , P _ { Y _ { i } } ) = \operatorname* { m a x } _ { h \in \mathcal { F } } \mathbb { E } [ h ( Y _ { i } ) ] - \mathbb { E } [ h ( \hat { X } _ { i } ) ] ,\tag{65}
$$

where is realized by the network in Table 6. A single critic serves all M branches, the class embedding is projected onto the image features and added to the scalar output, in the manner of a projection discriminator, and the critic is trained on the pooled batch with the branch labels. We use 5 critic updates per generator update, a gradient penalty weight of 10, and Adam with learning rate

![](images/d69a7f9a6e7774318fc508d3145c1f2ae4eac68b286853834b387bd5ab00ee56.jpg)  
Figure 9: System architecture diagram.

$1 0 ^ { - 4 }$ and $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 5 , 0 . 9 )$ . The critic is trained at every step, including at λ = 0 where it does not influence the objective, and is refit for a further 300 steps with the encoder and decoder frozen before the perception index is reported, so that the values compared across λ are measured by critics at comparable convergence.

<table><tr><td>Layer</td><td>Output</td></tr><tr><td>Input Conv 4 × 4, stride 2, LeakyReLU(0.2) Conv 4 × 4, stride 2, LeakyReLU(0.2) AvgPool, Flatten</td><td>32 × 32 × 3 16 × 16 × 64 8 × 8 × 128 2048</td></tr></table>

Table 6: Critic used to estimate the perception index.

## F.5 TRAINING OF THE COMPRESSION SCHEME

Each operating point is a separate encoder-decoder pair trained for 300 epochs with Adam at a learning rate of $\bar { 1 } 0 ^ { - 3 }$ and a batch size of 800, on the 8000 transmitter domain training images. The reference batches for the perception term are drawn from the receiver domain training split for the $P _ { Y _ { i } }$ configuration and from the transmitter domain split for the $P _ { X _ { i } }$ control; the test split is never used during training.

Following the protocol of (Blau & Michaeli, 2019), for each rate, we first train the distortion-only model (λ = 0) and every $\lambda > 0$ model is initialized from it and trained for the same number of further epochs. Asking an untrained codec to match a distribution while it is still learning to reconstruct rarely converges within a practical budget, and this warm start also removes the initialization as a source of variation between the points of a sweep.

Finally, we note that the numerical values of λ are not comparable across divergences unless the two terms of (12) are on a common scale. The distortion above is a per-pixel mean of order $1 0 ^ { - 2 }$ , whereas the critic estimate is of order 10, thus a literal λ would suppress the distortion term entirely. We therefore measure the magnitude of both terms over a short warmup and apply λ to their ratio, so that λ = 1 weights the two terms equally, the λ grid used is 0, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10, 20, 35, 50, 75, 90, 100, 300 . Reported distortion and perception values are the achieved quantities and are unaffected by this convention.

## G ADDITIONAL RESULTS

## G.1 RDP SURFACE

Fig. 10a illustrates the tradeoff between rate, distortion and perception for Level 2 with lossless transmission of J. This surface was obtained by sweeping the Lagrangian described in Section 7. As illustrated, the classification error decreases towards the low perception and high distortion regime similar to our results with discrete alphabets. This trend is further highlighted in the error, distortion and perception curve depicted in Fig. 10b.

![](images/7a0491b015ae480e3b8ce23fe3a76b6fce8003a4bcea792447f483e2a473cdb4.jpg)  
(a) rate

![](images/cf654eb4b726a0c2c2402d409705234bac5d2cabad5157e5186704a54690fb40.jpg)  
(b) p<sub>e</sub>(X<sup>ˆ</sup> )  
Figure 10: Variation of rate and error with distortion and perception (Level 2 Tx J).

## G.2 RECONSTRUCTION RESULTS

Fig. 11 and Fig. 12 depict sample reconstructions obtained at Level 2 and Level 3, respectively. We see that by matching on to the MNIST data distribution, the decoder generated images resembling samples from the MNIST dataset that is close to our source sample from SVHN dataset. We see that in Level 2, the reconstructions that are obtained at higher λ values (i.e., tighter perception constraint) resembles samples that are closer to MNIST dataset and hence the decrease in error. However, for Level 3, we observe that the reconstructions at moderate λ values generate the best reconstructions from the MNIST dataset. This is because, as λ increases, our distortion constraint also increases, thus allowing the decoder to generate reconstructions independently of the actual source sample. Even though the basic features of the MNIST dataset is present in these reconstructions, they no longer contain any useful information from the source sample. Hence, the decrease in the accuracy in Level 3 at high λ values.

We also test how the classification accuracies would vary when we match on to the source distribution following the Level 2 configuration where we transmit J losslessly. Here, the divergence term in (12) was replaced with $\begin{array} { r } { \sum _ { i = 0 } ^ { M - \bar { 1 } } P _ { i } \delta ( P _ { \hat { X } _ { i } } , P _ { X _ { i } } ) } \end{array}$ and the same training process described in Section 7 was carried out. Fig. 13 illustrates the reconstructions obtained using the above process. As seen, the reconstructions do resemble samples from the SVHN dataset. However, since the classifier was trained on the MNIST dataset, the error constantly remains around 0.90 regardless of the λ value that is used.

R<sup>′</sup> = 24 bits  
R<sup>′</sup> = 64 bits  
R<sup>′</sup> = 96 bits  
![](images/e4a91791f63014aa685adb067f4a2915fcee2adc98d84a7017c8185ec2cfcdfc.jpg)  
R<sup>′</sup> = 128 bits  
(a) Level 2 Tx J.

![](images/83506dc92357bc3c39150f0bf49bcd779ae01dc0ef895c7938a7b7223934d0b5.jpg)  
(b) Level 2 No J.  
Figure 11: Reconstruction results for Level 2.

![](images/6980b68ad8d78ca238ee49d901399c99428e8357575634516a138599838c3cca.jpg)  
Figure 12: Reconstructions for Level 3.

![](images/68ffb5c6eaa7f2acd2f08e0038d8f80c779d1bae536ee3cd69131d2b9ba13d66.jpg)  
Figure 13: Level 2, Tx J, match $P _ { X }$