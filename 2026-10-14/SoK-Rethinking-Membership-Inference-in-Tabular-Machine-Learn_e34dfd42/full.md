# SoK: Rethinking Membership Inference in Tabular Machine Learning

Cristina Pêra<sup>3,4</sup>, Tânia Carvalho<sup>2</sup>, Maxime Cordy<sup>2</sup>, Luís Antunes<sup>1,3</sup>

<sup>1</sup>Department of Computer Science, Faculty of Sciences of the University of Porto, Portugal

<sup>2</sup>SnT, University of Luxembourg, Luxembourg

<sup>3</sup>TekPrivacy, Portugal

<sup>4</sup>Corresponding author: cpera@tekprivacy.com

## Abstract

Membership Inference Attacks (MIAs) are currently the dominant approach to evaluating privacy in machine learning applications. Despite their significance, several concerns remain unexplored, particularly with regard to tabular data. Existing evaluations focus predominantly on high-dimensional datasets, leaving the privacy behavior of small-to-medium datasets unexplored. In this paper, we present a new taxonomy for MIAs and conduct a comprehensive study of the factors influencing vulnerability. We evaluate four distinct state-ofthe-art attacks across diverse tabular datasets and model ar chitectures. Our results show that datasets with higher class granularity and majority class percentage are more susceptible to MIAs. We further demonstrate that advanced MIAs are highly dependent on adversarial knowledge of the target model, while reference model-free attacks often achieve statistically comparable results. Most importantly, we show that records with unique signatures (single-outs) remain highly vulnerable, even when common defenses are applied. These findings indicate that current defenses reduce average leakage, but fail to adequately protect the most sensitive individuals.

## 1 Introduction

In the past decade, Artificial Intelligence (AI) has reached astonishing progress. Currently, nearly every aspect of daily life involves at least one AI-driven system, for instance, personalized recommendations, virtual assistants, and autonomous vehicles. Although such technologies automate processes and improve convenience and efficiency, their rapid development has also raised societal concerns, particularly regarding privacy, as personal information is increasingly stored and reused. This underscores the importance of developing tools and methodologies that can effectively support both developers and existing systems in evaluating their compliance.

A quantitative privacy measure that is currently on the spotlight is Membership Inference Attack (MIA) [56,64,83]. This is, so far, the most widely used and easiest to interpret approach. Put simply, MIA determines whether a specific record (i.e. an individual) is part of the training dataset. For example, if a model was trained exclusively on patients from a mental health clinic, successfully inferring that a person’s record was part of the training set reveals that the individual received psychological treatment. Such a disclosure exposes highly sensitive personal information while also compromising the confidentiality and reputation of the clinic. Therefore, MIAs are valuable tools for auditing data privacy and regulatory compliance of machine learning (ML) models. For instance, under the legal requirement of the right to beforgotten (Art. 17, GDPR), MIAs can verify if the model does not include the specific data point following a deletion request.

This type of attack has been explored on a wide variety of data types and learning tasks. Initially, MIAs were studied in the context of classification problems [3, 41, 54, 60, 64, 85, 88]. Since then, research has been extended to other domains such as regression [21], recommendation systems [8, 22, 89], graph learning [79], natural language processing [62], and even wireless signal analysis [63] and location data [56]. Recently, attention has shifted towards MIAs in Large Language Models (LLMs) [23, 47, 78, 81] and Tabular Foundation Models [5].

Although multiple data domains have been explored for MIAs, tabular data remain overlooked. Structured data are the predominant data format in real-world applications, such as healthcare, finance, demographics, and government statistics. The nature of the attributes and the presence of sensitive information in tabular data make privacy analysis essential. Yet most existing MIA evaluations neglect this setting, particularly simple tabular datasets.

In this work, we focus on MIAs in supervised classification, one of the most widely adopted ML paradigms in practice. We identify three key research gaps. First, models trained on simple tabular data are rarely evaluated, making MIAs less reliable in such data settings. In particular, the relationship between essential dataset characteristics and MIA effectiveness is largely unexplored. Figure 1 illustrates this challenge: although larger training datasets are generally expected to improve privacy by reducing overfitting, increasing class granularity consistently leads to higher MIA performance. Second, while model-based attacks are often considered the most effective approach, they require surrogate models, substantial computational resources, and assumptions about the target model that may not be applicable in practice. It remains unclear whether these additional costs consistently translate into better attack performance than simpler reference model-free alternatives. Third, the privacy risks faced by single-outs, namely individuals with highly unique records [68], have received no attention in the MIA literature, despite being the individuals most likely to suffer severe privacy breaches.

![](images/de5a5470c8620cbe54049ada2a5f92be6de55322ba9d87dcd33813556b0e2d17.jpg)  
Figure 1: Membership Inference Attacks effectiveness for different Purchases variants. Results are reported for five runs.

## Overall, we make the following contributions.

i) We conduct thorough investigations into existing research on MIAs and their respective defenses. Based on our finding, we suggest a new taxonomy for MIAs according to the type of adversary access instead of knowledge. The systematization leads to several key insights: i) most existing MIAs incur high computational costs due to the need to train additional ML models; ii) the majority are designed for deep learning, with relatively few applicable to tabular data; iii) many attacks are not evaluated against defense mechanisms, potentially overestimating their effectiveness and limiting their practical assessment; iv) some defenses (e.g., MemGuard [33], Nirvana [80]) are evaluated against only a single MIA, leaving their robustness across different attack instances and methods uncertain; and v) several proposed attacks do not provide publicly available code, hindering reproducibility (see Table 1).

ii) We comprehensively evaluate four state-of-the-art MIAs across ten small-to-medium tabular datasets spanning diverse characteristics. To reflect the nature of these datasets, we evaluate a variety of ML models , including simpler architectures which are frequently employed in this scenario, yet often overlooked in MIA studies. Our findings provide practical guidance for identifying privacy risks early in the ML pipeline, enabling informed preprocessing decisions, such as balancing target entropy and majority class percentage, or reducing highly granu lar class structures, as well as selecting models that are less susceptible to MIAs.

iii) We demonstrate that effective privacy auditing can be performed with simpler and more resource-efficient methods without compromising attack effectiveness. In particular, model-free MIAs achieve statistically comparable performance to model-based attacks, despite requiring fewer assumptions about the target model, which better reflects realistic black-box scenarios where little or no knowledge of the target model is available.

iv) Finally, we investigate the vulnerability of single-outs to MIAs. Due to their distinctive attribute combinations, such records are more likely to be retained by ML models, increasing the adversary’s confidence. Our results show that single-outs are consistently more vulnerable than non-single-outs and remain highly susceptible even when defense mechanisms are applied. To mitigate this risk, we evaluate prior pseudonymization of the training data and demonstrate that it can reduce attack effectiveness while preserving predictive utility.

Tabular MIAs as a SoK. Several surveys have reviewed MIAs [28, 47, 51, 52]. Our SoK complements these works by offering a practical overview for researchers and practitioners. We introduce a taxonomy based on the minimum adversarial access required, rather than the traditional white/black-box distinction, avoiding overlapping assumptions and enabling clearer comparisons between attack families.

Furthermore, current MIA research mainly focuses on image datasets and complex tabular benchmarks with hundreds of classes, where deep learning models are typically employed (e.g., [3, 72, 85, 86, 88]). Consequently, MIAs on simple-tomedium tabular classification tasks remains poorly understood. We address this gap through a comprehensive empirical evaluation across diverse tabular datasets.

Systematization Scope and Roadmap. We conducted a Google Scholar search using combinations of keywords related to MIAs, privacy-preserving ML, defense for MIAs and tabular data. We considered works published since 2017, excluding prompt-based LLM classification approaches. Among the retrieved literature, we identified 33 representative MIA methods published in recognized peer-reviewed venues that are sufficiently specified through algorithmic descriptions and/or publicly available implementations. The remaining of the paper is organized as follows. Section 2 introduces the preliminaries of the MIA game. Section 3 reviews and discusses existing MIA literature, followed by defenses in Section 4 and evaluation metrics in Section 5. Sections 6 and 7 evaluate MIA effectiveness and single-out vulnerability, respectively. Section 8 discusses challenges and future directions, and Section 9 concludes the paper.

## 2 Preliminaries

Consider a classification problem (supervised learning), where a model f with parameters θ maps inputs $x \in \chi$ to an n-class probability distribution, $f _ { \Theta } : X \to [ 0 , 1 ] ^ { n }$ . The probability assigned to y is denoted as $f ( x ) _ { \chi }$ <sub>y</sub>. Let D be a training dataset, sampled from some underlying distribution D, where N is the number of data instances. We denote $f _ { \boldsymbol { \Theta } } \gets \mathcal { T } ( D )$ as a f trained model with parameters θ that is learned by running the training algorithm $\mathcal { T }$ on the training set D.

Membership Inference Attacks. Successful MIA occurs when an adversary correctly determines if a record belongs to the training set of $f _ { \boldsymbol { \theta } }$ or not. In the former case, the record is said to be a member, and in the latter case the record is a non-member. We define membership inference inspired in a standard security game [3, 32, 86, 88].

Definition 1 (Membership Inference Game). The game proceeds between a challenger and an adversary. Let D be the data distribution and T the training algorithm.

1. The challenger samples a training dataset $S \gets \mathbb { D }$ and trains a model $f _ { \boldsymbol { \Theta } } \gets \mathcal { T } ( S )$ on S.

2. The challengerflips a bit b. $i f b = 1 ,$ , it randomly samples a data pointfrom the training set $( x , y )  S .$ . Otherwise, it selects a point from the distribution $( x , y ) \gets \mathbb { D }$ such that $( x , y ) \notin S .$

3. The challenger sends the target model f<sub>θ</sub> and target data point $( x , y )$ to the adversary.

4. With access to the distribution D, the adversary computes Scor $\mathcal { \prime } _ { M I A } ( ( x , y ) ; f _ { \theta } )$ , and outputs a membership prediction bit $\hat { b }  M I A ( ( x , y ) ; f _ { \boldsymbol { \theta } } )$

5. The output of the game is 1 $i f { \hat { b } } = b ,$ and 0 otherwise.

The prior knowledge of the adversary with respect to the target model, as well as the attack approach, are important factors that can greatly influence the success of MIAs. Concerning the first factor, the adversary can have white-box, grey-box, or black-box knowledge.

In a white-box attack, adversaries have complete knowledge about a target model (architecture and parameters), such as gradients and outputs of the hidden layers, which facilitates the membership inference. Although white-box attacks are more common, black-box attacks are more realistic for ML systems as essential model information, such as model implementation details, is often confidential and protected from normal users interacting with the system.

Black-box attacks assume that the adversary has no access to the target model’s internal parameters or training process and can interact with it only through queries. The literature distinguishes three levels of adversarial knowledge: i) access only to the predicted label, ii) access to confidence scores or probabilities, and iii) additional knowledge of the target model family, training procedure, or data distribution. Although the latter is often classified as black-box, it can also be regarded as a grey-box setting due to the availability of auxiliary information beyond query access.

Let us focus on the most predominant setting: score-based attacks relying on the model’s output probabilities. Once the training process is completed, the target model $f _ { \Theta }$ outputs, for each input x, a probability vector $p \in \mathbb { R } ^ { k }$ of the form $p = ( p _ { 1 } , p _ { 2 } , \dotsc , p _ { k } )$ , where $p _ { i } \in [ 0 , 1 ] \forall i$ and $\textstyle \sum _ { i = 1 } ^ { k } p _ { i } = 1$ . The predicted class label for a feature vector x is the class with the highest probability value in $p ,$ thus $\hat { y } = a r g m a x _ { i \in \{ 1 , . . . , k \} } f _ { \boldsymbol { \Theta } } ( x ) _ { i }$ <sub>i</sub>. The output distribution $p = f _ { \mathsf { \theta } } ( x )$ is often informative about whether a point $( x , y )$ was included in the training dataset $D .$ During the attack, a membership score $S c o r e _ { \mathrm { M I A } } ( ( x , y ) , f _ { \Theta } )$ is assigned to each pair. A binary membership decision is then obtained by comparing this score against a threshold $\beta$ which distinguishes members from non-members.

$$
\mathrm { M I A } ( x ; \theta ) = \mathbb { 1 } [ \mathrm { S c o r e } _ { \mathrm { M I A } } ( x ; \theta ) \geq \beta ] ,\tag{1}
$$

where $\mathbb { 1 } ( \cdot )$ is an indicator function that results in 1 if the event occurs, otherwise 0.

MIAs in overfitted vs well-generalized target models. A successful MIA is typically associated with a high level of overfitting of the target model [15, 60, 64]. Overfitting occurs when a model fits too closely to its training data, resulting in high train accuracy results but fails to make accurate predictions in unseen data. This phenomenon is also associated to memorization, in which larger models are prone to memorizing the training data, thus making them more vulnerable to MIAs [3, 42]. Consider a simple loss function $L ( f _ { \boldsymbol { \Theta } } ( x ) _ { y } )$ where $\begin{array} { r } { R ( \Theta ) = 1 / n \sum _ { i = 1 } ^ { n } L ( f _ { \Theta } ( x ) _ { y } ) } \end{array}$ is the average loss on the training data and $\hat { R } ( \boldsymbol { \theta } )$ the expected loss on unseen data. If the margin of error between $\hat { R } ( \boldsymbol { \theta } )$ and $R ( { \boldsymbol { \theta } } )$ is large and positive, it is said that the model is overfitting. Nevertheless, it has been shown that although overfitting is sufficient it is not necessary for the success of such attacks [37, 59, 86]; well-generalized models can leak membership information, when there are vulnerable data points that cause unique influences on the learning target model, such as outliers [46].

## 3 Review of Membership Inference Attacks

MIAs are commonly divided w.r.t. the type of adversary’s knowledge, namely, black-box or white-box [15,52,57]. However, a white-box adversary can perform any black-box attack. Therefore, we propose a new categorization based on the information accessible to the adversary. Figure 2 presents the new taxonomy for MIAs. We divide it into two distinct target model access scenarios. Output Access, in which an adversary only has access to predicted labels, confidence scores, or probability vectors, and Internal Access, in which an adversary has access to information such as model architecture, parameters, gradients, activations, or training losses.

![](images/b86f6903e9b19670c30efb44540a63c285ba4615b472c6ef962f426d004ffea2.jpg)  
Figure 2: Taxonomy for Membership Inference Attacks.

Given the vast state-of-the-art in this topic, we provide in Table 1 the proposed MIAs. For approaches that were not explicitly named in the original publications, we assign descriptive labels based on their key characteristics to facilitate comparison and discussion.

## 3.1 Output Access

Model outputs include probabilities, logits and predicted labels. Since logits and probabilities reveal similar levels of confidence, we treat them as a single category, distinguishing only between probabilities and labels access.

Probabilities. One of the pioneers MIAs in ML [64] perform the attack by training multiple shadow models that mimic the behavior of the target model. In a black-box setting with access to output probabilities and partial architectural knowledge, multiple shadow models are trained on datasets sampled from the same distribution as the target model. The known membership status of the shadow data is then used to construct an attack dataset of the form {shadow model’s top probabilities, ground-truth label, x}, where $x \in \{ 0 , 1 \}$ indicates whether a sample is a member. The attack is then formulated as a binary classification problem to infer membership. Numerous studies have demonstrated its effectiveness [41, 52, 75].

However, this approach presents some challenges. First, the success of the attack depends on the knowledge of the data distribution and target model’s architecture. Second, it requires multiple shadow models, which is computationally expensive, especially when the model under attack is large. Therefore, subsequent works show improvements in MIAs by restricting knowledge assumptions and using limited shadow models – in some cases, just one [60, 86], as we also demonstrate in Section 6.2. In addition, MIAs have been shown to be effective to some extent through data transfer attacks [60] or model transferability [75].

A major limitation is its poor performance at low falsepositive rates, as it relies only on the mean loss values. Consequently, the attack becomes suboptimal when individual samples have markedly different loss scales and variances [3]. To address this issue, a principled parametric approach models the loss distributions using well-calibrated Gaussian likelihood estimates. The attack proceeds by training multiple reference models, half containing the target example (x,y) (IN models) and half excluding it (OUT models). Each reference model $f _ { \theta } ^ { \prime }$ is queried to obtain the correct class logits, $f _ { \boldsymbol { \Theta } } ^ { \prime } ( x ) y$ . Separate Gaussian distributions, $\mathcal { N } ( \mu _ { I N } , \sigma _ { I N } ^ { 2 } )$ (similar for the OUT models), are then fitted to the IN and OUT logits. Finally, the likelihood ratio between the two distributions is computed and used as the membership score.

Reference models are similar to shadow models but serve a different purpose. While shadow-based attacks mimic the target model, reference-based attacks estimate output likelihoods under the IN and OUT hypotheses, often achieving better performance [13, 54, 67, 72, 85, 87, 88]. Furthermore, a simple modification of the score, called difficulty calibration, can drastically improve the reliability of the attack [24, 77]. A low loss sample is not necessarily a member; it could also be due to its low difficulty. Difficulty calibration assumes that outputs from reference models can effectively represent the difficulty of target samples, thus improving existing MIAs.

Despite such improvements, previous studies are still computationally costly. A lightweight alternative uses a mixture of experts architecture [1]. It works by replacing independent shadow models with shared models derived from a single mixture of experts training process. This can reduce costs by approximately 75% compared to the most applied attack (RMIA [88]) with an equivalent attack efficiency. Nevertheless, it requires a sufficient amount of auxiliary data, which may not be a realistic scenario in many cases.

A different approach leverages distillation, where a "teacher" model transfers knowledge to a "student" model. Reference-based attacks have been extended to this setting [17, 31, 85]. In particular, self-distillation, where both models share the same training data, can amplify overfitting and improve attack performance [31]. This occurs because memorized samples lead to overly confident predictions, increasing the separation between members and non-members.

Knowledge distillation also serves as a proxy for reconstructing training dynamics by tracking how membership signals evolve across training epochs. This strategy captures this evolution as a loss trajectory [39, 45]. Naturally, distillationbased attacks must train and distil one or more local models over many epochs, which incurs in high computational cost. Moreover, it requires access to an auxiliary dataset, which is rarely identical to the target training data, and therefore, the distribution gap degrades signal quality.

To overcome the computational cost of training multiple shadow or reference models, a model-agnostic approach was proposed to eliminate the need for such models [2]. Instead, it trains a quantile regression model using only member and non-member samples to perform membership inference. Nevertheless, the quantile regression quality is sensitive to the

<table><tr><td>Year</td><td>Approach Name</td><td>Attack Classification</td><td>Attack Model-based</td><td>Learning Type</td><td>Adversary Knowledge</td><td>Defenses</td><td>Data Type</td><td>Open Source</td></tr><tr><td>2017</td><td>MIA [64]</td><td>Shadow</td><td>√</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>X</td></tr><tr><td>2018</td><td>LOSS [86]</td><td>Metric-based</td><td>X</td><td>DL &amp; TL</td><td></td><td>X</td><td>T&amp;I</td><td>X</td></tr><tr><td>2019</td><td>MALT [59]</td><td>Metric-based</td><td>X</td><td>DL &amp; TL</td><td></td><td>X</td><td>I</td><td>X</td></tr><tr><td>2019</td><td>MAST [59]</td><td>Shadow</td><td>√</td><td>DL &amp; TL</td><td></td><td>X</td><td>I</td><td>X</td></tr><tr><td>2019</td><td>MATT [59]</td><td>Influence functions</td><td>X</td><td>TL</td><td></td><td>X</td><td>I</td><td>X</td></tr><tr><td>2020</td><td>Prediction uncertainty [50]</td><td>Gradient statistics</td><td>√</td><td>DL</td><td></td><td>X</td><td>T&amp;I</td><td>X</td></tr><tr><td>2020</td><td>Idiosyncratic MIA [37]</td><td>Gradient attribution</td><td>√</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>X</td></tr><tr><td>2021</td><td>Merlin &amp; Morgan [32]</td><td>Metric-based</td><td>X</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>0</td></tr><tr><td>2021</td><td>Privacy risk score [65]</td><td>Metric-based</td><td>X</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>0</td></tr><tr><td>2021</td><td>Data augmentation attack [9]</td><td>Direct-query</td><td>X</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>0</td></tr><tr><td>2021</td><td>Boundary [43]</td><td>Direct-query</td><td>X</td><td>DL</td><td></td><td>√</td><td>I</td><td>C</td></tr><tr><td>2022</td><td>Attack R [85]</td><td>Reference</td><td>√</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>0</td></tr><tr><td>2022</td><td>Attack D [85]</td><td>Distillation</td><td>√</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>0</td></tr><tr><td>2022</td><td>Attack P [85]</td><td>Population</td><td>X</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>0</td></tr><tr><td>2022</td><td>LiRA [3]</td><td>Reference</td><td>√</td><td>DL</td><td></td><td>√</td><td>I</td><td>0</td></tr><tr><td>2022</td><td>Difficulty Calibration [77]</td><td>Reference &amp; Direct-query</td><td>t</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>0</td></tr><tr><td>2022</td><td>ASTER [44]</td><td>Metric-based</td><td>X</td><td>DL &amp; TL</td><td></td><td>X</td><td>T&amp;I</td><td>0</td></tr><tr><td>2022</td><td>SIF [10]</td><td>Influence functions</td><td>X</td><td>DL</td><td></td><td>√</td><td>I</td><td>0</td></tr><tr><td>2022</td><td>TrajectoryMIA [45]</td><td>Distillation</td><td>√</td><td>DL</td><td></td><td>X</td><td>I</td><td>0</td></tr><tr><td>2023</td><td>QMIA [2]</td><td>Regression</td><td>√</td><td>DL</td><td></td><td>X</td><td>T&amp;I</td><td>0</td></tr><tr><td>2023</td><td>GapMIA [80]</td><td>Generalization gap</td><td>X</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>X</td></tr><tr><td>2024</td><td>RMIA [88]</td><td>Reference</td><td>√</td><td>DL &amp; TL</td><td></td><td>X</td><td>T&amp;I</td><td>0</td></tr><tr><td>2024</td><td>IHA [67]</td><td>Influence functions</td><td>X</td><td>DL &amp; TL</td><td></td><td>X</td><td>T&amp;I</td><td>0</td></tr><tr><td>2024</td><td>LiRA-L [67]</td><td>Reference</td><td>√</td><td>DL &amp; TL</td><td></td><td>X</td><td>T&amp;I</td><td>0</td></tr><tr><td>2024</td><td>GLiRA [17]</td><td>Distillation</td><td>√</td><td>DL</td><td></td><td>X</td><td>I</td><td>X</td></tr><tr><td>2024</td><td>RAPID [24]</td><td>Reference</td><td>√</td><td>DL</td><td></td><td>V</td><td>I</td><td>0</td></tr><tr><td>2024</td><td>SeqMIA [39]</td><td>Distillation</td><td>√</td><td>DL</td><td></td><td>V</td><td>I</td><td>0</td></tr><tr><td>2024</td><td>YOQO [82]</td><td>Estimated-query</td><td>√</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>0</td></tr><tr><td>2024</td><td>OSLO [53]</td><td>Estimated-query</td><td>√</td><td>DL</td><td></td><td>√</td><td>I</td><td>0</td></tr><tr><td>2025</td><td>RaMIA [72]</td><td>Reference</td><td>√</td><td>DL</td><td></td><td>X</td><td>T&amp;I</td><td>0</td></tr><tr><td>2025</td><td>DHAttack [40]</td><td>Estimated-query</td><td>√</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>0</td></tr><tr><td>2026</td><td>SHAPOOL [1]</td><td>Reference</td><td>√</td><td>DL</td><td></td><td>X</td><td>I</td><td>0</td></tr><tr><td>2026</td><td>IMIA [13]</td><td>Reference</td><td>√</td><td>DL</td><td></td><td>√</td><td>T&amp;I</td><td>0</td></tr></table>

represents the black-box setting, the white-box and means that proposed attack is tested on both.

Table 1: Overview of membership inference attack proposals and their main characteristics. For each attack, we indicate whether it employs auxiliary models (e.g., shadow models, reference models or/and attack classifiers/regressors), the target model type (deep learning (DL) or traditional learning (TL)), the data types considered (tabular (T) and/or image (I)), and whether the attack was evaluated against defense mechanisms. "Open Source" contains links to repositories.

volume of the data, i.e. on small datasets, the regression model may not learn accurate per-example conditional quantiles.

In contrast, metric-based approaches perform membership inference directly from statistics derived from the target model outputs, avoiding attack model training and, in many cases, the need for auxiliary data [27]. For example, MIA based on the prediction correctness metric requires only knowledge of the predicted labels. The attack follows this principle: if the target model correctly predicts a record, then the adversary infers it as a member, otherwise as a non-member [86]. Several modelfree MIAs infer membership by comparing a metric derived from predictions against a threshold. One approach uses the prediction loss, assuming that training members exhibit lower cross-entropy loss than non-members [86]. Others rely on the maximum prediction probability or prediction entropy [59, 60]. The former exploits the tendency of models to assign higher confidence to training members, whereas the latter leverages the higher uncertainty of non-members, reflected by larger entropy values. However, prediction entropy ignore the ground-truth label, which can lead to misclassification of members and non-members. The modified prediction entropy metric overcomes this limitation by incorporating the groundtruth label into its computation [65].

Although effective, these scalar metrics only consider the model output at the original input and ignore how predictions behave under perturbations. An extension to loss-based attacks analyzes the variation of loss after small input perturbations. Training members are more likely to lie near local minima due to overfitting, causing perturbations to increase their loss. Further improvement combines the original loss with the perturbation-based score using multiple thresholds [32].

Subsequent work introduced more informative metrics by comparing samples against population distributions [85]. Membership is inferred by comparing the target sample against the output distribution estimated from this population. Unlike reference-based attacks, uncertainty arises only from the queried sample, not from retraining different models.

Finally, a few works exploit the local geometry of the model. In particular, Jacobian matrix-based attacks quantify how sensitive the prediction is to small perturbations of the input, providing an additional membership signal. This matrix reflects the relationship between each feature’s perturbation and the corresponding prediction’s change [44, 70]. Samples are clustered based on sensitivity, with the lowest-sensitivity cluster identified as members. However, selecting an appropriate ε for Jacobian approximation can be time-consuming.

Label. A big concern of the majority MIAs is that the confidence (or loss) scores of the model are exposed, which rely on continuous-valued predictions. For a higher level of protection, prior work has considered obfuscating the model’s output by returning only the top label or modifying predicted values for a higher level of protection. Given a target model, the adversary queries an input record and obtains only the predicted label. Despite this limited information, direct-query label attacks can still identify whether a record was in the original training data. The formulation is: if a sample is a member, it tends to be harder to perturb across the decision boundary than a non-member, thus the attack measures how the label changes under perturbations. Membership can be inferred by exploiting data augmentation and decision boundary to evaluate robustness of the target records to these input perturbations [9,43]. Besides, these strategies can also benefit from difficulty calibration [77], making them more effective.

However, direct-query label attacks typically require many queries to estimate prediction robustness, making them computationally expensive and potentially impractical. In contrast, estimated-query attacks replace this iterative process with offline estimation, enabling MIAs from a single query to the target model [53, 82]. These strategies employ IN/OUT surrogate models during an offline phase to estimate the information required for single query inference. While these approaches need only one query on the target model, it requires crafting a query sample across all IN and OUT models. Increasing a little the query budget, a shadow-based proposal aims to approximate the non-member state where its boundary distances are then modeled as Gaussian distributions [40]. Although there is additional computational overhead, this approach presents a higher attack effectiveness. Similarly to this shadow training, distillation has also been tested for labelbased attacks [45]. However, it requires much higher query budget. Despite recent progress, label-based attacks are underexplored and do not work well for non-images data due to the method of measuring the boundary distance.

In summary, attacks that rely on the output access of the target model are the most practical and widely applicable. Probability-based attacks generally offer the best performance by exploiting confidence scores. Nonetheless, labelbased attacks remain applicable even when only predicted labels are exposed. Despite this, their effectiveness is limited by the amount of information available through the model outputs, which restricts the membership signal that can be exploited. Yet, they can achieve a favorable trade-off between practicality and effectiveness.

## 3.2 Internal Access

Although previous attacks are carried out with limited knowledge, several researchers have shown how to improve the attack success in deep learning models [10, 37, 50, 59, 67, 80]. They assume the adversary has access to the target model’s internals, corresponding to a white-box scenario.

Gradients. One of the first proposals demonstrates how to exploit the privacy vulnerabilities of the stochastic gradient descent (SGD) algorithm [50]. Since SGD updates the model parameters by minimizing the loss over small subsets of the training set, the per-sample gradients encode information about whether a record was used for training. The attack leverages these gradient statistics to distinguish members from non-members.

Moreover, exploiting the influence of individual samples in the learned parameters can drive the membership signal [59]. The attack measures the alignment between the parameter displacement and the loss gradient to estimate whether a sample contributed to training. However, it has limited applicability to neural networks because the required assumption of a unique global minimum for the Laplace approximation does not hold; it has been evaluated only on logistic regression. More recent work estimates sample influence through Hessian-weighted self-influence, measuring how strongly a sample affects its own prediction [10]. By introducing parameter-gradient align ment, this approach bridge the gap between loss-based attacks [86] and influence-function-based approaches [59], providing a more principled estimation of sample-specific influence through Hessian-weighted self-influence. To further improve discrimination, self-influence is combined with an influence alignment term that measures the agreement between the target sample’s influence and the average influence of the remaining training data [67]. This unified score yields stronger membership signals but requires access to the full training set to compute the average influence.

Additionally, some records in the target model’s training set may contain specific features that identify them uniquely. For instance, if a target model is trained to recognise a particular individual among multiple faces and only one contains a pink background, then this feature could serve as evidence of membership. This demonstrates that privacy issues resulting from overfitting do not necessarily manifest in the output behavior, but rather in the way the model uses features. In other words, the idiosyncratic use of features by a model can provide evidence for membership [37]. To uncover such memorized features, the attack employs gradient attribution techniques that quantify the contribution of individual input features to the model’s prediction. A Bayes-optimal threat model is used to derive a set of parameters that profile idiosyncratic features, which are then used to construct a logistic attack model.

Hidden layers. Other attacks leverage internal layer representations as the membership signal [38, 80]. One such approach estimates the KL divergence between the hiddenlayer feature distributions of training and test samples, treating this quantity as an internal generalisation gap [80]. This gap is more pronounced at intermediate layers than at the output layer, with early layers showing a sharp increase that gradually narrows deeper in the network. But selecting the most influential subset can improve both the accuracy and efficiency of MIA [38]. Since only a fraction of neurons substantially impacts the target classification task, KL divergence and random forest importance scores are combined to identify the most influential neurons across all layers. SHAP values (explainable AI technique) are then used to quantify each feature’s contribution within those neurons and capture their directional influence on the model output. Membership is inferred from the disparity in activation patterns between the neuron sets characteristic of training and unseen samples. However, estimating KL divergence over high-dimensional feature spaces is susceptible to the curse of dimensionality, limiting scalability to deeper or wider architectures.

In summary, the richer information obtained from internal access often results in higher attack accuracy. However, these methods are generally limited to white-box settings and have been developed primarily for deep learning models. Moreover, these attacks typically incur substantial computational overhead due to the need to compute gradients, Hessian approximations, or intermediate activations.

## 4 Defenses to Membership Inference Attacks

Given the wide range of proposed MIAs, there is a growing need to develop mitigation mechanisms that effectively reduce their success. Defenses against MIAs can be applied during the training phase or the prediction phase [75]. They are typically classified as either model hardening or API hardening methods. The former aims to apply techniques that reduce the target model’s level of overfitting. The latter aims to limit the information accessible in the prediction vector.

## 4.1 Model Hardening Defenses

This category covers regularization techniques, hyperparameter tuning, differential privacy, model stacking, knowledge distillation, outlier detection, and k-anonymity.

Regularization techniques help to smooth model confidence distributions which may weaken MIAs. For instance, dropout reduces overfitting by reducing variance [34, 60, 64]. It randomly sets some nodes to zero during the training process. This effectively creates smaller networks that each need to learn to solve the network’s task. Although this mitigation is generally efficient, it is limited to neural networks. Similarly, l2-regularization [9, 43, 64, 75] penalizes large weights to improve generalization and reduce the confidence gap between members and non-members, although excessive regularization may degrade model performance. More recently, entropy-based regularization has been proposed to prevent overconfident predictions by combining high-entropy soft labels with an entropy regularizer during training [7].

Unlike these methods, adversarial regularization aims to reduce prediction loss by optimizing the target model against membership inference. A common approach is Min-Max privacy game in which the target model is trained using an adversarial process [41, 49]. Specifically, the gain of the MIA is used as a regularizer for the target model that controls the privacy-utility trade-off. Related approaches include adversarial training [33, 42] and data augmentation [10, 37, 41, 42]. While both introduce additional training examples to improve robustness, adversarial training may inadvertently increase memorization and privacy leakage, whereas data augmentation alone has shown limited effectiveness against MIAs [42].

Multi-sample convex combination [80] leverages the generalization gap of the internal layers of the target model. It creates synthetic feature representations in hidden layers by linearly mixing the outputs of hidden units from different data points, resulting in new blended feature representations. This discourages the model from having highly specific memorized patterns in the hidden layers, thus obscuring whether a specific data point was part of the training set or not. This was designed to protect models in a white-box scenario.

Label smoothing or early stopping can also act as regularization techniques [34]. However, label smoothing can inadvertently increase vulnerability to MIAs. By reducing prediction confidence, it can obscure overfitting signals yet amplify membership cues. Conversely, early stopping consistently mitigates MIAs by limiting overfitting: the earlier training is stopped, the lower the attack success.

Hyper-parameter tuning of ML models must be carefully defined. Certain configurations can lead to a higher risk of MIAs. Some models, namely tree-based, have already been tested with multiple combinations of hyper-parameters [54] to guide in the parameters selection. In addition, attention should be given to the choice of target model. For example, the Naive Bayes classifier has been shown to be more resilient to these attacks as it usually does not overfit [75].

Differential privacy (DP) is one of the most widely adopted defenses against MIAs [3, 9, 10, 24, 37, 43, 77]. By injecting calibrated noise during training, typically to gradients in SGD [66], DP limits the influence of any individual record on the learned model. However, several studies have highlighted the difficulty of achieving a satisfactory privacy-utility trade-off. In fact, the privacy budget ε, which represents the degree of privacy protection, when small, it decreases the effectiveness of the adversary, but also reduces the utility of the target model. To mitigate this trade-off, several variants have been proposed, including combining DP with transfer learning in which the knowledge learned from a task is reused [9] or applying DP only to the top layers of a model [37]. Compared with standard regularization techniques such as dropout, DP-based methods generally provide stronger protection [37].

Model stacking is an ensemble learning that combines multiple base ML models to construct a final model [41, 60]. The base models obtain the outputs and concatenate them to apply the result to the final model, which predicts the final output. All base models are trained on disjoint sets of data. However, this strategy is easily circumvented in a white-box setting as the base models are available to the adversary [37].

Knowledge distillation is commonly used for model compression but it also protects against MIAs by thwarting the access of the resulting models to the private training data [31, 34, 61, 71]. The intuition is that since the student model is not directly optimized over the private set, their membership may be protected. However, this method can only be applied when a public dataset is available for training.

Outlier identification estimates whether a given instance is an outlier. An outlier can be more likely to be a vulnerable target record when it is in the target model’s training set, even if the model is already well-generalized [46]. After identifying the outliers, the data points that do not significantly contribute to the utility of the model are removed.

k-anonymity can be applied to the training data before model training, ensuring that each record is indistinguishable from at least k−1 others [75]. This reduces the influence of individual records and can mitigate membership leakage. Since records with unique attribute combinations are generally more susceptible to re-identification, they are also expected to be more vulnerable to MIAs. However, this relationship has not yet been systematically investigated.

## 4.2 API Hardening Defenses

This category aims to modify the model’s external outputs, which includes restriction of prediction vector information and adding noise to the prediction vector.

Restriction of prediction vector information can be achieved using several strategies, for example by: i) restricting the prediction vector to the top k classes by adding a filter to the last layer of the target model [75]; ii) decreasing the precision of the prediction vector by rounding the classification probabilities down to d floating-point digits; and, iii) increasing the entropy of the prediction vector by adding or modifying the softmax layer and increasing its normalizing temperature to above zero [64]. However, confidence masking alone has been shown to provide limited protection [9].

An improved strategy works by replacing confidence scores rather than simply reducing their precision or entropy. It uses low-confidence prediction vectors obtained from randomly generated non-member samples to mask the original scores, while preserving the ordering of predicted classes [7].

Adding noise to the prediction vector means that crafted noise is carefully added to each confidence score vector of the target classifier [33, 75]. Since the adversary may know the defense mechanism, but the defender does not know the attack classifier, it is essential that noise is added randomly;

otherwise, the adversary could learn it and adapt their MIA technique. However, the distance between the noisy and true prediction vectors should remain minimal to preserve the trade-off. Although stated as effective in black-box scenarios, this defense does not modify the generalization gap in the hidden layers of the target model, and thus does not protect against adversaries with access to the model’s internals [80].

Despite the notable differences in defensive approaches between model and API hardening, some can be combined to enhance output protection. For example, generalization and perturbation can be applied together [46]. Nevertheless, API hardening defenses are generally ineffective against whitebox adversaries, who can access the original model outputs before post-processing [37].

## 5 Membership Inference Evaluation

Regardless of the underlying attack strategy, MIA evaluation is formulated as a binary classification problem, where the objective is to distinguish between members and non-members. All attack evaluations are compared to a clueless adversary. This adversary makes random guesses about the membership and is expected to get an accuracy of 0.5.

MIAs were initially evaluated using accuracy to capture overall correctness. For a more comprehensive assessment, complementary measures have been used, such as precision, which quantifies how many truly members belong to the training set, and recall that measures the attack coverage, i.e. the amount of actual members that the adversary successfully detects [37, 64, 75, 86]. Other similar evaluation metrics include F1-score [20, 38] and mean average precision (mAP) [59].

These evaluation metrics have been applied when the attack dataset is balanced, which means that the portions of members and non-members are similar. Membership advantage, obtained through the difference between true-positive rate (TPR) andfalse-positive rate (FPR) [86] quantifies how much better an adversary is than random guessing at distinguishing members from non-members, in which a difference of zero corresponds to the random guess. Unlike precision/recall, the analysis of TPR/FPR is independent of the prevalence of members in the population, and thus may be more appropriate when this prevalence is unknown.

However, such metrics are average-case and assign equal cost to false positives and false negatives. Thus, Receiver Operating Characteristic (ROC) curve [3] and its Area Under the Curve (AUC) [60] were suggested to MIAs assessment since it emphasizes positive predictions over non-membership predictions. It compares the attack’s TPR and FPR for all possible decision threshold. Nevertheless, AUC can be misleading because it averages performance over the entire FPR range, including high false-positive rates that are of limited practical relevance. Consequently, recent work recommends reporting the TPR at a fixed low FPR (e.g., 0.1% or 0.001%) [3].

Additionally, for attacks requiring multiple trained models (e.g. [3, 88]), the uncertainty of the estimated TPR should be quantified, using Clopper-Pearson confidence intervals, to account for variability arising from finite sample sizes [74].

A concern remains with the previous MIAs evaluation metrics: they mostly focus on positive membership. But detecting non-membership is still relevant. In particular, FPR or False Alarm Rate (FAR) shows how non-members are often misclassified as members [58]. It should be noted, however, that achieving reliable MIA with high accuracy and low FAR is difficult. Then a difficulty calibration was proposed to balance accuracy and FPR with the aim of achieving a better separation between member and non-member scores [77]. Similarly, instead of relying on a fixed confidence threshold of 50%, a dynamic threshold is set based on a predefined false-positive tolerance α, which is set to the $\alpha ^ { t h }$ -percentile of confidence scores from a sample. This ensures a controlled FPR level with differing calibration or confidence distributions [37].

## 6 Effectiveness of MIAs on tabular data

In this section, we present the experimental evaluation of MIAs on tabular data. Specifically, we investigate the impact of dataset characteristics on privacy leakage and assess the effectiveness of reference-based attack strategies. To guide our analysis, we address the following research questions:

RQ1: Which characteristics of tabular datasets contribute more to increased susceptibility to MIAs?

RQ2: How effective are surrogate reference models in improving MIAs over reference-free approaches?

Threat model. We consider MIAs against a trained tabular classifier f<sub>θ</sub>. Given a candidate labelled record $( x , y )$ , the adversary’s goal is to infer whether $( x , y )$ was included in the target model’s training set. The attack output is a membership score, where larger values indicate higher confidence that the sample is a member.

We assume a black-box, confidence-score adversary. The adversary can query the target model on candidate records and observe the model confidence assigned to the true label, $p _ { \Theta } ( y \mid x )$ , but does not access model parameters, gradients, or training internals. The adversary knows the model architecture family and has access to auxiliary data drawn from the same distribution as the target training data. This auxiliary data is used to train or reuse reference/shadow models and, when required, to estimate population-level confidence distributions. Candidate membership labels are used only for evaluation, not by the adversary at test time.

Experimental Setup. We evaluate our methods on ten tabular classification datasets spanning a diverse range of dataset sizes, feature dimensionalities, and class cardinalities (Table 4, Appendix A.1). They cover several data domains as well. We evaluate six classifiers including Multilayer Perceptron (MLP), Random Forest (RF), Decision Trees (DT), XG-Boost (XGB), Logistic Regression (LR), Naive Bayes (NB). Details on model architectures are present in Appendix A.2. Although tabular foundation models such as TabPFN [25] have recently attracted considerable attention due to their strong performance on small- to medium-sized datasets, often outperforming classical ML approaches [16], recent work suggests that they may be substantially more vulnerable to MIAs. In particular, successful attacks have been demonstrated with only a limited number of reference models, and in some cases without requiring a separately trained attack model [5]. Furthermore, the in-context learning paradigm employed by these models introduces a distinct attack surface. Thus, we restrict our analysis to classical ML models, which still remain widely used in real-world and industry applications.

We evaluate four state-of-the-art MIAs: RMIA [88], LiRA [3], Attack-P [85] and QMIA [2]. We focus on these attacks, as they are the most applied in the relevant literature. In all cases, the attacks are evaluated on the same audit candidate set and under the same target-model access assumptions, differing only in how they estimate the non-member/reference confidence distribution used to score membership.

All experiments are repeated across five independent random seeds. Each seed induces a new random permutation of the candidate pool, resulting in different training contexts and audit splits. Reported results correspond to the mean and standard deviation across seeds.

The attack effectiveness is then reported using the area under the ROC curve (AUC), which provides a thresholdindependent measure of separability between members and non-members (aggregate discrimination). Additionally, we report TPR at low FPR corresponding to practical attack scenarios in which the adversary acts only on highly confident membership predictions.

## 6.1 Data Factors Affecting MIAs

As previously stated, MIA is not defined solely by the generalisation gap. As such, we next evaluate data-level characteristics that most influence the attacks. Therefore, we present in Table 2 a Spearman correlation between a dataset/target characteristic and the attack’s AUC, computed across datasets. Granularity measures how fine-grained the target label space is. Class imbalance describes the overall unequal distribution of samples across classes. Majority class describes how strongly the largest class dominates the dataset, measured by its percentage of all samples. Rarity measures how poorly represented the rare/minority classes are. Target entropy measures the Shannon entropy of the target-class distribution. Mean |Corr| w/ Target means the average absolute correlation between each input feature and the target variable. Features w/ Outliers is the number or proportion of features containing

detected outlier observations.
<table><tr><td>Attack</td><td>Granularity</td><td>Imbalance</td><td>Majority Class</td><td>Rarity</td><td>Target Entropy</td><td>Mean |Corrl w/ Target</td><td>Features w/ Outliers</td></tr><tr><td>RMIA</td><td>0.831*</td><td>0.079</td><td>-0.624</td><td>-0.677*</td><td>0.915*</td><td>-0.333</td><td>-0.267</td></tr><tr><td>LiRA</td><td>0.774*</td><td>-0.055</td><td>-0.758*</td><td>-0.360*</td><td>0.927*</td><td>-0.491</td><td>-0.377</td></tr><tr><td>Attack-P</td><td>0.604</td><td>-0.261</td><td>-0.818*</td><td>0.006</td><td>0.830*</td><td>-0.442</td><td>-0.359</td></tr><tr><td>QMIA</td><td>0.699*</td><td>-0.236</td><td>-0.794*</td><td>-0.194</td><td>0.915*</td><td>-0.479</td><td>-0.267</td></tr></table>

Table 2: Spearman rank correlation between dataset/class properties and MIA success. \* denotes $p < 0 . 0 5$

The results show that Target Entropy is the dominant data factor contributing to MIA’s susceptibility. It is the only significant property across all four attacks and has the highest correlation magnitudes. Follows class Granularity and Majority Class which contribute to the success of three attacks, but the last is consistently negative, meaning that datasets dominated by one class are harder to attack. Additionally, Rarity tends to show increased success for reference-based attacks.

A possible solution to reduce the impact of Target Entropy and Granularity on the success of the attack is to merge rare fine-grained classes into broader ones if the task allows. Otherwise, class-conditional regularization, such as stronger dropout or weight decay on low-support classes, may mitigate to a certain extent. However, these are specific to gradienttrained parametric models with a differentiable loss, which not all of our models support. The Rarity factor is discussed further in Section 7.

Resampling. Many resampling strategies have been proposed to tackle imbalance and majority class dominance, in which SMOTE [6] is the most well known. However, previous work shows that SMOTE is non-private [18]. As such, recent work has incorporated DP into the generation process of SMOTE. We therefore experiment with PP-SMOTE [19] and DP-SMOTE [90]. Table 3 presents the results for PP-SMOTE since both methods produced very similar conclusions. We use $\mathfrak { E } \in \{ 0 . 0 1 , 0 . 5 , 1 \}$ and resample all classes but the majority class. Due to space limitation, we show only results for the three most vulnerable models.

<table><tr><td rowspan="2">Attack</td><td rowspan="2">Model</td><td colspan="3">COMPAS</td><td colspan="3">Surgery</td><td colspan="3">MIC</td></tr><tr><td>0.01</td><td>0.5</td><td>1</td><td>0.01</td><td>0.5</td><td>1</td><td>0.01</td><td>0.5</td><td>1</td></tr><tr><td rowspan="3">RMIA</td><td>MLP</td><td>-0.4</td><td>7.9</td><td>0.5</td><td>-8.9</td><td>6.3</td><td>11.1</td><td>-4.8</td><td>39.3</td><td>23.3</td></tr><tr><td>RF</td><td>-1.3</td><td>-1.4</td><td>-1.6</td><td>-14.6</td><td>5.6</td><td>9.9</td><td>23.6</td><td>28.0</td><td>32.7</td></tr><tr><td>LR</td><td>-1.2</td><td>-0.0</td><td>0.1</td><td>-1.3</td><td>-0.2</td><td>-0.7</td><td>-9.0</td><td>-8.3</td><td>-7.6</td></tr><tr><td rowspan="3">LiRA</td><td>MLP</td><td>3.2</td><td>7.7</td><td>3.9</td><td>2.4</td><td>12.4</td><td>18.6</td><td>-2.9</td><td>24.9</td><td>15.2</td></tr><tr><td>RF</td><td>-0.9</td><td>-1.2</td><td>-1.6</td><td>-3.8</td><td>9.4</td><td>11.6</td><td>14.0</td><td>17.2</td><td>19.3</td></tr><tr><td>LR</td><td>-1.5</td><td>0.2</td><td>-0.1</td><td>-0.9</td><td>0.0</td><td>0.0</td><td>0.2</td><td>1.1</td><td>1.0</td></tr><tr><td rowspan="3">Attack-P</td><td>MLP</td><td>5.3</td><td>14.2</td><td>7.0</td><td>8.8</td><td>22.8</td><td>27.4</td><td>9.6</td><td>53.1</td><td>28.5</td></tr><tr><td>RF</td><td>-1.2</td><td>-0.9</td><td>-1.0</td><td>6.3</td><td>23.7</td><td>29.0</td><td>9.3</td><td>17.3</td><td>19.0</td></tr><tr><td>LR</td><td>0.4</td><td>-0.1</td><td>-0.1</td><td>2.3</td><td>0.5</td><td>0.4</td><td>5.8</td><td>1.3</td><td>-0.3</td></tr><tr><td rowspan="3">QMIA</td><td>MLP</td><td>5.1</td><td>14.0</td><td>7.1</td><td>1.8</td><td>17.1</td><td>21.9</td><td>6.6</td><td>53.9</td><td>27.0</td></tr><tr><td>RF</td><td>-2.6</td><td>-2.1</td><td>-1.8</td><td>-9.7</td><td>5.8</td><td>12.2</td><td>5.2</td><td>10.3</td><td>10.6</td></tr><tr><td>LR</td><td>-0.2</td><td>-0.6</td><td>-0.7</td><td>2.3</td><td>-0.3</td><td>-0.5</td><td>4.8</td><td>0.1</td><td>-1.7</td></tr></table>

Table 3: Attack AUC change before and after applying PP-SMOTE across three privacy budgets.

In general, the oversampling in these scenarios amplifies the influence of the few original points rather than hiding them. This is especially notable for the MIC dataset, which contains 8 predictive classes, with approximately 85% of the samples belonging to the majority class. Such a highly skewed class distribution makes balancing particularly challenging. Nevertheless, our results suggest that, under certain configurations, privacy-aware synthetic data generation can still offer a degree of protection, highlighting the need for careful tuning and empirical evaluation. Importantly, these findings highlight the urgent need for the continued development of robust resampling strategies against such attacks. The evaluated approaches show that the practical privacy protection offered by DP is not always sufficient in this context.

Model analysis. Overfitting is typically the cleanest modellevel proxy already proved [86], where MIA advantage is theoretically bounded by the generalization gap. Our results (Table 6, Appendix B) show that while DT is the model that overfits the most, RF and XGB present a greater leakage than their generalization gap suggests. This stress the importance of analysis of other factors.

Another important factor is the training paradigm. For instance, RF, DT and XGB are based on tree ensembles that typically memorize via leaf partitioning but get stochastic regularization from bagging/feature subsampling. On the other hand, MLP, gradient-based nets can directly overfit to individual points’ losses, which is reflected in our results.

Finally, the capacity relative to dataset size, i.e., the parameter count vs the number of samples. In our study, we use a fixed architecture for MLP, the parameters do not adapt to n. The capacity/n should gradually decrease as the size of the dataset increases. It is expected that MLP becomes less vulnerable to MIA due to this effect. Future work includes such analysis. For tree-based models the size and depth affecting parameters (e.g. max\_depth) are searched per dataset, so their effective capacity already partially adapts to n.

Insight 1 Target entropy and class granularity are the most robust, attack-agnostic leakage predictors.

## 6.2 On the Use of Reference Models

Beyond identifying which data-level characteristics make MIAs more effective, we next examine how much adversarial effort is required to exploit that vulnerability, specifically, whether costly surrogate reference models are necessary or whether simpler attacks suffice. To test practical equivalence, we apply the Two One-Sided Tests (TOST) procedure. Given a pre-specified equivalence margin ∆ = 0.05, TOST evaluates two one-sided hypotheses simultaneously: $H _ { 1 } : \mu _ { d i f f } \le - \Delta$ and $H _ { 2 } : \mu _ { d i f f } \geq + \Delta$ . Rejecting both, i.e., both one-sided pvalues falling below α, allows the conclusion that the true mean AUC difference lies within $( - \Delta , + \Delta )$ , and is therefore statistically indistinguishable from practically zero. Following standard practice, we report $p _ { - } T O S T = m a x ( p _ { 1 } , p _ { 2 } )$ , which requires $p \_ T O S T < \alpha = 0 . 0 5$ for an equivalence claim.

Figure 3 applies TOST to the surrogate RMIA, instantiated with reference models drawn from five architectures with MLP fixed as the target model. This reflects a more realistic black-box scenario. For demonstration purposes, we focus solely on one target model, specifically the most vulnerable. We compare all architectures against the two reference-free baselines, Attack-P and QMIA, across three datasets.

Under the margin ∆, every gap across all three datasets and all reference model budgets falls within $( - \Delta , + \Delta )$ . Even the COMPAS’s AUC gap (0.032) which lead for surrogate RMIA, remains inside the equivalence bound. Both one-sided tests reject non-equivalence for every dataset and reference-model budget pair, formally confirming that surrogate RMIA and the reference-free baselines are statistically practical equivalent. Appendix D extends this analysis for all datasets.

Additionally, surrogate RMIA shows consistent diminishing returns in which nearly all AUC gain from increasing the reference model budget is realized by the fifth trained model, with budgets of 10 and 30 producing no further meaningful movement in any dataset, $\Delta \le 0 . 0 0 2$ relative to the cost of 5 models. This suggests that any advantage from additional surrogate reference models saturates quickly rather than scaling with attacker compute.

Insight 2 Reference-free MIAs are a cost-effective substitutefor surrogate reference models, reaching statistically equivalent detection power.

## 7 Vulnerability of Single-Outs to MIAs

In this section, we investigate the privacy risks faced by single-out records under MIAs. This record-level analysis deserves a dedicated section, given its singularity. We first evaluate whether records identified as single-outs exhibit increased susceptibility to membership inference compared to the general population. Subsequently, we assess the effectiveness of privacy-preserving defenses in mitigating the leakage associated with these high-risk individuals. To guide our analysis, we address the following research questions:

RQ3: To what extent are single-out records more vulnerable to MIAs than non-single-out records?

RQ4: How effective are existing defense mechanisms in protecting single-out records against MIAs?

The following results concern seven datasets as they include quasi-identifiers that are easily identifiable. The set of such attributes is listed in Table 5 (Appendix A.1). Since singling out is defined with respect to training records, we evaluate susceptibility only among members. All single outs are flagged according to the definition of k-anonymity [69]. Although k = 1 is the standard setting, as it corresponds to a 100% re-identification risk, we also consider $k = 2 .$ , which still implies a 50% risk. Importantly, these single-out records are not used to evaluate re-identification; rather, they define groups of records with different levels of uniqueness, which we then use to assess their susceptibility to MIAs. All experimental settings remain unchanged from the previous section.

## 7.1 Experimental Results

For each dataset, model, and attack, we evaluate the difference in attack performance between SO and non-SO members. For TPR-based evaluation, we first calibrate the attack threshold on non-member records to achieve a fixed FPR, and then compare the TPR of SO and non-SO members at that same threshold. This isolates whether being more identifiable in the underlying data translates into increased MIA vulnerability.

We quantify this difference using the SO susceptibility gap:

$$
\Delta _ { \mathrm { S O } } = \mathrm { T P R } _ { \mathrm { S O } } - \mathrm { T P R } _ { \mathrm { n o n } - \mathrm { S O } } .
$$

Positive values indicate a greater vulnerability among SO records. Figure 4 shows the SO–non-SO performance gap for two TPR@low FPR at k = 1 and $k = 2$ , also including AUC. Each point represents the mean gap pooled across models and attacks, with 95% bootstrap confidence intervals (CI) over model–seed replicates. When CI overlaps zero indicates that the SO effect is not statistically significant at the level of 5%.

The SO vulnerability effect is robust across k and metric. The effect is stronger at $k = 1$ than at k = 2, with the gap decreasing significantly as k is relaxed. This matches the intuition where $k = 1$ defines a stricter, smaller equivalence class than $k = 2$ , thus SO records are more extreme records and therefore more susceptible to membership inference, whereas relaxing k dilutes this distinction. The effect weakens as k increases for both AUC and TPR, with the TPR gap reducing by a comparable, albeit slightly larger, proportion. $\mathrm { A t } k = 1$ TPR@0.1% FPR reaches significance in more datasets than AUC (6/7 vs 5/7), while at $k = 2$ this reverses (4/7 vs 5/7), as Diabetes and Student Performance lose significance for TPR but not AUC. TPR@0.1%FPR at k = 1 shows a strong, robust signal, comparable in strength to looser FPR thresholds (0.5–1% FPR gave marginally higher pooled effect sizes) and weakening only at the strictest 0.01% FPR (5/7 datasets significant at k = 1, dropping to 3/7 at k = 2). Half Million remains non-significant for TPR across every FPR threshold we tested (0.01%–1%) and both values of k, despite showing a small but statistically detectable AUC gap.

Additionally, we observe differences in the strength of the SO-vulnerability effect across attacks in Figure 9 (Appendix E). RMIA and LiRA show the clearest and most consistent separation, while Attack-P exhibits a weaker effect at k = 1 that becomes more pronounced at k = 2. QMIA shows the weakest evidence for a SO-vulnerability gap.

Overall, singling out increases the adversary’s edge over chance by 24.9% at k = 1 (95% CI [16.1%, 35.2%]) and

![](images/175a1fa6ea4e0789dc48cb69ed2713495954fa4a056aa79826a6aed36d45882f.jpg)  
Figure 3: Comparison of Attack-P/QMIA versus surrogate RMIA performance at each reference model budget. The RMIA AUC line represents the mean ± standard deviation across all reference models architectures. The dashed lines show the mean Attack-P and QMIA AUC, with the respective shaded bands depicting the ± Standard Error of the Mean across seeds.

![](images/952e2749e2e8f4942e57df9ff6f1c7380e721fe245e03d8c7589f75a89d517ac.jpg)  
Figure 4: Difference in membership inference vulnerability between singled-out (SO) and non-SO records per dataset, measured by AUC and both TPR@0.1%FPR (solid) and TPR@0.01%FPR (dashed) for anonymity k = 1 and k = 2. Points show the estimated SO–non-SO difference, with horizontal bars denoting confidence intervals (CIs).

16.5% at k = 2 (95% CI [10.2%, 24.1%]).

Insight 3: SO records tend to exhibit greater membership inference vulnerability than non-SO records. This effect is stronger even at the lowest FPR.

## 7.2 Impact of defenses

To understand the effectiveness of defense mechanisms in reducing single-outs’ vulnerability to MIAs, we apply two state-of-the-art defenses: DP [14] [26] and HAMP [7]. The reason these two defenses are chosen is to include both training-time and inference-time defenses. Besides, we do not include methods such as MemGuard, based on adversarial examples, as it has demonstrated similar results to the undefended model [7, 53]. In addition, regularization defenses, such as early stopping, l2, and dropout, were also discarded since they are model dependent and less efficient than DP [37].

Figure 5 reports defense privacy-utility trade-off by dataset for k = 1, considering the baseline as the attack without any defense. The plots demonstrate that the attack AUC change is not directly proportional to that of single-out catch.

As evident, HAMP clusters near zero. It costs no accuracy but also buys almost no AUC reduction. The unchanged target model’s utility is because this attack is applied during inference time, operating only at the output of the model. Consequently, the reduction in single-out catch is usually small, especially for MIC. In contrast, DP defenses show a greater reduction in the target model’s utility, especially for smaller ε values. Despite this, we can find that some DP cases achieve a larger drop in single-out catch than HAMP, while still preserving utility, this is emphasized on MIC and Surgery. Notably, DP ε = 0.01 is the costliest but not proportionally the most protective. On MIC in particular, accuracy is reduced to -0.6 and to -0.8 (near-destroying the model) while AUC tops out around 0.2. Table 10, Appendix F extends the analysis to the remaining attacks, which shows that reference-free attacks can capture higher number of single-outs.

Given such results, it is important to apply specific measures to tackle these vulnerable records.

K-anonymous datasets. We then report in Figure 6 the application of k-anonymity. We use $k \in \{ 2 , 3 , 5 \}$ , and both generalization and suppression to the same set of quasi-identifiers.

Overall, we observe a reduction of MIA AUC. Although this is strongly evident for Surgery, COMPAS and MIC show that anonymization can, for certain settings, increase attack success (up to +4.8 pp). Its quasi-identifiers present lowsignal attributes, thus applying anonymization methods barely changes the attack’s signal. Besides reducing AUC, it also can provide better predictive performance. These results correlate with the Rarity property mentioned in Section 6.1 since rare records are easier to attack because the model memorizes them. By applying k-anonymity’s suppression, any record whose quasi-identifier values occur fewer than k times gets those values replaced with a shared placeholder, weakening the attack when those quasi-identifiers are tied to the label. Future work remains to study the impact of k-anonymity on different sets of quasi-identifiers and test on other attacks.

![](images/8645c89493c99a60cd1b1e02bf46dfaf1b93510d1638cd00db4b440c5c088729.jpg)  
Figure 5: Privacy–utility trade-off of defenses on singled-out (k = 1) records, by dataset, considering RMIA. Each point is one model. Color shows how much less often single-outs are correctly flagged as members after defense.

<table><tr><td>ΔACC</td><td>-0.1 -0.1</td><td>COMPAS</td><td>-0.3</td><td>+0.6</td><td>MIC +0.6</td><td>+0.5</td><td>+16.6</td><td>Surgery +17.0</td><td>+16.7</td><td></td></tr><tr><td>MLP</td><td>-5.0</td><td>-5.0</td><td>-7.2</td><td>-1.2</td><td>+1.9</td><td>+0.0</td><td>-14.4</td><td>-9.1</td><td>-14.3</td><td></td></tr><tr><td>NB</td><td>-0.1</td><td>-0.1</td><td>+0.2</td><td>+1.5</td><td>+1.4</td><td>+1.3</td><td>+0.1</td><td>+0.4</td><td></td><td>-0.0</td></tr><tr><td>DT</td><td>-0.4</td><td>-0.4</td><td>+1.2</td><td>-0.4</td><td>+0.5</td><td>+0.5</td><td>-10.9</td><td>-10.9</td><td></td><td>-11.0</td></tr><tr><td>RF</td><td>+4.8</td><td>+4.8</td><td>-4.0</td><td>-2.4</td><td>-3.5</td><td>-2.6</td><td>-24.7</td><td>-23.5</td><td></td><td>-21.5</td></tr><tr><td>XGB</td><td>-0.2</td><td>-0.2</td><td>+1.1</td><td>+3.0</td><td>+3.0</td><td>+3.2</td><td>-12.4</td><td>-12.5</td><td></td><td>-12.0</td></tr><tr><td>LR</td><td>-0.1</td><td>-0.1</td><td>+0.1</td><td>+0.9</td><td>+1.0</td><td>+0.5</td><td>-1.4</td><td>-0.7</td><td>-1.0</td><td></td></tr><tr><td></td><td>k=2</td><td>k=3</td><td>k=5</td><td>k=2</td><td>k=3</td><td>k=5</td><td>k=2</td><td>k=3</td><td>k=5</td><td></td></tr></table>

Figure 6: k-anonymity application over datasets and its percentage points difference against the baseline.

Insight 4 State-of-the-art defenses can lower average vul nerability but singled-out individuals still amplifies MIA.

## 8 Open Challenges and Research Directions

Privacy-aware data pre-processing. Our findings indicate that privacy should be considered before model training rather than only during or after deployment. It is therefore important to extend the evaluation of privacy-aware preprocessing strategies across a broader range of datasets and application domains to better understand their impact on the privacyutility trade-off. Developing such pipelines may provide a first line of defense against MIAs.

Beyond supervised classification. While classification is the dominant benchmark for MIAs, important learning paradigms remain underexplored, particularly regression models. In regression settings, membership signals are often weaker and harder to calibrate. As models typically exhibit lower overfitting, may have residuals that look similar for members and non-members, making the membership signal noisier and harder to detect [21]. Future research should also investigate privacy risks in other emerging scenarios, including unsupervised learning, tabular foundation models, generative models for synthetic tabular data, and large language models adapted for tabular applications. These settings may exhibit different memorization behavior and require dedicated attack and defense mechanisms.

Enhanced attacks. Current MIAs generally assume that the adversary can query the target model using complete input records. However, this assumption may not be applicable in practice, where only partial information about an individual may be available. Recent studies demonstrate that membership can still be inferred from incomplete feature vectors or feature regions by exploiting the model’s latent representations and memorization patterns [72, 76]. These findings motivate the development of MIAs under more realistic threat models, where adversaries operate with limited or imperfect knowledge of the target records.

## 9 Conclusion

In this paper, we present a systematization of knowledge for tabular MIAs. We suggest a new taxonomy around the adversary’s minimum required access and use it to review a substantial set of attacks and their respective defenses. Our results reveal dataset-level properties such as target entropy, granularity, and majority class are strong predictors of a model’s overall vulnerability. Most importantly, we show that singledout records can remain disproportionately exposed even when defenses appear effective, and that k-anonymity can amplify rather than reduce this exposure, highlighting the need for record-level and worst-case privacy auditing. Likewise, our evidence that reference-free attacks can match reference-based ones challenges the assumption that stronger adversarial capabilities are necessary for meaningful auditing, suggesting that privacy evaluation can be made substantially cheaper without sacrificing effectiveness.

## Ethical Considerations

Stakeholder-Based Analysis. We identify three key stakeholders impacted by this research:

• Machine Learning Practitioners. This work equips ML practitioners training models on tabular data with a systematic understanding of membership inference risk, including which data properties (e.g., singling-out records, class granularity) drive vulnerability, and which common defenses fail to adequately protect against it. This enables more informed decisions about data handling and defense deployment prior to release.

• Data Subjects. MIAs risk is heightened for singled-out records and rare classes, which we show are disproportionately vulnerable. We mitigate this risk by conducting all experiments exclusively on public benchmark tabular datasets, without targeting real individuals or proprietary models. By identifying which records and defenses are insufficiently protective, our work ultimately helps practitioners to better safeguard data subjects going forward.

• Adversaries. We acknowledge that our findings, could potentially be misused by adversaries to more effectively target real-world models. We take this risk seriously and discuss concrete mitigations below.

Mitigations. We take the following concrete steps to mitigate potential harms:

• Exclusive Use of Public Benchmark Data. All experiments are conducted exclusively on public tabular datasets, used in accordance with their licenses. We do not target specific individuals or production systems at any point, ensuring that no one’s real privacy is compromised in the course of this research.

• Systematic, Reproducible Evaluation. We conduct a systematic evaluation of established MIAs using standard, well-documented methodology. This transparency allows practitioners and the defense research community to reproduce, verify, and build directly on our findings.

• Preprocessing Defense. We suggest and evaluate class imbalance and k-anonymization as preprocessing strategies to reduce membership leakage. By releasing findings alongside a practical and deployable mitigation, we reduce the risk introduced by this work and provide practitioners with actionable guidance to protect their data.

Justification for Research and Publication. Prior work has established that classical MIAs can be effective against tabular models trained on real-world data. However, little is known about which records are most vulnerable, why existing defenses may fail to protect them, and which data properties predict this vulnerability. Our study addresses these gaps, helping practitioners identify privacy vulnerabilities and strengthen protections for vulnerable records, particularly those that are singled out or rare. We further contribute a classification of the MIA literature and a practical tool table summarizing available attacks and their requirements, lowering the barrier for practitioners to audit their own models and adopt these techniques for defensive. We emphasize that these findings should support privacy research and responsible deployment, not the targeting of individuals.

## Open Science

Consistent with open science principles and to support transparency, reproducibility, and further research, we make all the artifacts publicly available at https://anonymous.4open. science/r/sokMIA-4439. The README.md file provides a detailed, experiment-by-experiment mapping between reported results and the corresponding code required to reproduce them, including environment setup instructions. Each attack is implemented as a standalone script at the repository root, with the respective run\_\*\_batch.py files to run for all datasets and models at once.

## References

[1] Li Bai, Qingqing Ye, Xinwei Zhang, Sen Zhang, Zi Liang, Jianliang Xu, and Haibo Hu. Toward efficient inference attacks: Shadow model sharing via mixtureof-experts. Advances in Neural Information Processing Systems, 38:138757–138785, 2026.

[2] Martin Bertran, Shuai Tang, Aaron Roth, Michael Kearns, Jamie H Morgenstern, and Steven Z Wu. Scalable membership inference attacks via quantile regression. Advances in Neural Information Processing Systems, 36:314–330, 2023.

[3] Nicholas Carlini, Steve Chien, Milad Nasr, Shang Song, Andreas Terzis, and Florian Tramèr. Membership inference attacks from first principles. In 2022 IEEE symposium on security and privacy (SP), pages 1897–1914, 2022.

[4] Tânia Carvalho, Luís Antunes, Cristina Costa Santos, and Nuno Moniz. Empowering open data sharing for social good: a privacy-aware approach. Scientific Data, 12(1):248, 2025.

[5] Tânia Carvalho and Maxime Cordy. Privacy vulnerabilities of attention layers in tabular foundation models and protection of high-risk queries. arXiv preprint arXiv:2606.26021, 2026.

[6] Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and W Philip Kegelmeyer. Smote: synthetic minority

over-sampling technique. Journal of artificial intelligence research, 16:321–357, 2002.

[7] Zitao Chen and Karthik Pattabiraman. Overconfidence is a dangerous thing: Mitigating membership inference attacks by enforcing less confident prediction. In Proceedings ofthe Network and Distributed System Security Symposium, NDSS 2024. The Internet Society, 2024.

[8] Xiaoxiao Chi, Xuyun Zhang, Yan Wang, Lianyong Qi, Amin Beheshti, Xiaolong Xu, Kim-Kwang Raymond Choo, Shuo Wang, and Hongsheng Hu. Shadow-free membership inference attacks: recommender systems are more vulnerable than you thought. arXiv preprint arXiv:2405.07018, 2024.

[9] Christopher A. Choquette-Choo, Florian Tramèr, Nicholas Carlini, and Nicolas Papernot. Label-only membership inference attacks. In International conference on machine learning, pages 1964–1974. PMLR, 2021.

[10] Gilad Cohen and Raja Giryes. Membership inference attack using self influence functions. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 4892–4901, 2024.

[11] Paulo Cortez. Student Performance. UCI Machine Learning Repository, 2008. Accessed Sep 2025.

[12] Will Cukierski. Acquire valued shoppers challenge. https://www.kaggle.com/c/acquire-valuedshoppers-challenge/data. Accessed Sep 2025.

[13] Yuntao Du, Yuetian Chen, Hanshen Xiao, Bruno Ribeiro, and Ninghui Li. Imitative membership inference attack. In 35th USENIX Security Symposium (USENIX Security 2026), 2026.

[14] Cynthia Dwork. Differential privacy. In International colloquium on automata, languages, and programming, pages 1–12. Springer, 2006.

[15] Antreas Dyonysiou and Elias Athanasopoulos. Sok: Membership inference is harder than previously thought. Proceedings on Privacy Enhancing Technologies, 2023.

[16] Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmüller, Prateek Desai, David Salinas, and Frank Hutter. Tabarena: A living benchmark for machine learning on tabular data. Advances in Neural Information Processing Systems, 38, 2026.

[17] Andrey V Galichin, Mikhail Pautov, Alexey Zhavoronkin, Oleg Y Rogov, and Ivan Oseledets. Glira: Black-box membership inference attack via knowledge distillation. IEEE Transactions on Information Forensics and Security, 2025.

[18] Georgi Ganev, MohammadReza Nazari, Rees Davison, Amirhassan Dizche, Xinmin Wu, Ralph Abbey, Jorge Silva, and Emiliano De Cristofaro. Smote and mirrors: exposing privacy leakage from synthetic minority oversampling. In International Conference on Learning Representations, volume 2026, pages 142401–142417, 2026.

[19] Subhasish Ghosh, Amit Kr Mandal, and Agostino Cortesi. Enhancing deep learning model privacy against membership inference attacks using privacy-preserving oversampling. SN Computer Science, 6(4):316, 2025.

[20] Yuhao Gu, Yuebin Bai, and Shubin Xu. Cs-mia: Membership inference attack based on prediction confidence series in federated learning. Journal of Information Security and Applications, 67:103201, 2022.

[21] Umang Gupta, Dimitris Stripelis, Pradeep K Lam, Paul Thompson, Jose Luis Ambite, and Greg Ver Steeg. Membership inference attacks on deep regression models for neuroimaging. In Medical Imaging with Deep Learning, pages 228–251. PMLR, 2021.

[22] Jiajie He, Yuechun Gu, Keke Chen, and Xintong Chen. Membership inference attacks on recommender system: A survey. arXiv preprint arXiv:2509.11080, 2025.

[23] Yu He, Boheng Li, Liu Liu, Zhongjie Ba, Wei Dong, Yiming Li, Zhan Qin, Kui Ren, and Chun Chen. Towards label-only membership inference attack against pre-trained large language models. In USENIX Security, 2025.

[24] Yu He, Boheng Li, Yao Wang, Mengda Yang, Juan Wang, Hongxin Hu, and Xingyu Zhao. Is difficulty calibration all we need? towards more practical membership inference attacks. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security, pages 1226–1240, 2024.

[25] Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, and Frank Hutter. TabPFN: Accurate predictions on small data with a tabular foundation model. Nature, 637(8045):319–326, January 2025.

[26] Naoise Holohan, Stefano Braghin, Pól Mac Aonghusa, and Killian Levacher. Diffprivlib: The IBM differential privacy library, 2019. PyPI: https://pypi.org/ project/diffprivlib/. URL: https://github. com/IBM/differential-privacy-library, arXiv: 1907.02444.

[27] Hongsheng Hu, Zoran Salcic, Lichao Sun, Gillian Dobbie, Philip S. Yu, and Xuyun Zhang. Membership inference attacks on machine learning: A survey. ACM Computing Surveys (CSUR), 54(11s):1–37, 2022.

[28] Li Hu, Anli Yan, Hongyang Yan, Jin Li, Teng Huang, Yingying Zhang, Changyu Dong, and Chunsheng Yang. Defenses to membership inference attacks: A survey. ACM Computing Surveys, 56(4):1–34, 2023.

[29] imodels. Huggingface. https://huggingface.co/ datasets/imodels/compas-recidivism. Accessed May 2026.

[30] imodels. Huggingface. https://huggingface.co/ datasets/imodels/diabetes-readmission. Accessed May 2026.

[31] Matthew Jagielski, Milad Nasr, Katherine Lee, Christopher A Choquette-Choo, Nicholas Carlini, and Florian Tramer. Students parrot their teachers: Membership inference on model distillation. Advances in Neural Information Processing Systems, 36:44382–44397, 2023.

[32] Bargav Jayaraman, Lingxiao Wang, Katherine Knipmeyer, Quanquan Gu, and David Evans. Revisiting membership inference under realistic assumptions. In Proceedings on Privacy Enhancing Technologies (PoPETs), volume 2021, pages 33–53, 2021.

[33] Jinyuan Jia, Ahmed Salem, Michael Backes, Yang Zhang, and Neil Zhenqiang Gong. Memguard: Defending against black-box membership inference attacks via adversarial examples. In Proceedings ofthe 2019 ACM SIGSAC conference on computer and communications security, pages 259–274, nov 2019.

[34] Yigitcan Kaya, Sanghyun Hong, and Tudor Dumitras. On the effectiveness of regularization against membership inference attacks. arXiv preprint arXiv:2006.05336, 2020.

[35] Tobias Kuehn. Openml. https://www.openml.org/ search?type=data&status=active&id=310&sort= runs. Accessed June 2026.

[36] Privacy Trust Lab. Privacy trust lab datasets. https: //github.com/privacytrustlab/datasets. Accessed Sep 2025.

[37] Klas Leino and Matt Fredrikson. Stolen memories: Leveraging model memorization for calibrated {White-Box} membership inference. In 29th USENIX security symposium (USENIX Security 20), pages 1605–1622, 2020.

[38] Chenxi Li, Abhinav Kumar, Zhen Guo, Jie Hou, and Reza Tourani. Unveiling the unseen: Exploring whitebox membership inference through the lens of explain ability. arXiv preprint arXiv:2407.01306, jul 2024.

[39] Hao Li, Zheng Li, Siyuan Wu, Chengrui Hu, Yutong Ye, Min Zhang, Dengguo Feng, and Yang Zhang. Seqmia:

Sequential-metric based membership inference attack. In Proceedings ofthe 2024 on ACM SIGSAC Conference on Computer and Communications Security, pages 3496– 3510, 2024.

[40] Hao Li, Zheng Li, Siyuan Wu, Yutong Ye, Min Zhang, Dengguo Feng, and Yang Zhang. Enhanced {Label-Only} membership inference attacks with fewer queries. In 34th USENIX Security Symposium (USENIX Security 25), pages 5465–5483, 2025.

[41] Jiacheng Li, Ninghui Li, and Bruno Ribeiro. Membership inference attacks and defenses in classification models. In Proceedings ofthe Eleventh ACM Conference on Data and Application Security and Privacy, pages 5–16, 2021.

[42] Xiao Li, Qiongxiu Li, Zhanhao Hu, and Xiaolin Hu. On the privacy effect of data enhancement via the lens of memorization. IEEE Transactions on Information Forensics and Security, 19:4686–4699, 2024.

[43] Zheng Li and Yang Zhang. Membership leakage in label-only exposures. In Proceedings ofthe 2021 ACM SIGSAC Conference on Computer and Communications Security, pages 880–895, 2021.

[44] Lan Liu, Yi Wang, Gaoyang Liu, Kai Peng, and Chen Wang. Membership inference attacks against machine learning models via prediction sensitivity. IEEE Transactions on Dependable and Secure Computing, 20(3):2341–2347, 2022.

[45] Yiyong Liu, Zhengyu Zhao, Michael Backes, and Yang Zhang. Membership inference attacks by exploiting loss trajectory. In Proceedings of the 2022 ACM SIGSAC conference on computer and communications security, pages 2085–2098, 2022.

[46] Yunhui Long, Vincent Bindchaedler, Lei Wang, Diyue Bu, Xiaofeng Wang, Haixu Tang, Carl A. Gunter, and Kai Chen. Understanding membership inferences on well-generalized learning models. arXiv preprint arXiv:1802.04889, feb 2018.

[47] Matthieu Meeus, Igor Shilov, Shubham Jain, Manuel Faysse, Marek Rei, and Yves-Alexandre de Montjoye. Sok: Membership inference attacks on llms are rushing nowhere (and how to fix it). In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pages 385–401. IEEE, 2025.

[48] Aahan Mehra. Kaggle. https://www.kaggle.com/ datasets/aahanmehra/surgery. Accessed May 2026.

[49] Milad Nasr, Reza Shokri, and Amir Houmansadr. Machine learning with membership privacy using adversarial regularization. In Proceedings of the 2018 ACM SIGSAC conference on computer and communications security, pages 634–646, 2018.

[50] Milad Nasr, Reza Shokri, and Amir Houmansadr. Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning. In 2019 IEEE symposium on security and privacy (SP), pages 739–753. IEEE, 2019.

[51] Jun Niu, Peng Liu, Xiaoyan Zhu, Kuo Shen, Yuecong Wang, Haotian Chi, Yulong Shen, Xiaohong Jiang, Jianfeng Ma, and Yuqing Zhang. A survey on membership inference attacks and defenses in machine learning. Journal ofInformation and Intelligence, 2(5):404–454, 2024.

[52] Jun Niu, Xiaoyan Zhu, Moxuan Zeng, Ge Zhang, Qingyang Zhao, Chunhui Huang, Yangming Zhang, Suyu An, Yangzhong Wang, Xinghui Yue, et al. Comparing different membership inference attacks with a comprehensive benchmark. IEEE Transactions on Information Forensics and Security, 2025.

[53] Yuefeng Peng, Jaechul Roh, Subhransu Maji, and Amir Houmansadr. Oslo: one-shot label-only membership inference attacks. Advances in Neural Information Processing Systems, 37:62310–62333, 2024.

[54] Richard J. Preen and Jim Smith. A hierarchical approach for assessing the vulnerability of tree-based classification models to membership inference attack. arXiv preprint arXiv:2502.09396, feb 2025.

[55] Lennart Purucker. Openml. id=46943; Accessed May 2026.

[56] Apostolos Pyrgelis, Carmela Troncoso, and Emiliano De Cristofaro. Knock knock, who’s there? membership inference on aggregate location data. arXiv preprint arXiv:1708.06145, 2017.

[57] Bosen Rao, Jiale Zhang, Di Wu, Chengcheng Zhu, Xiaobing Sun, and Bing Chen. Privacy inference attack and defense in centralized and federated learning: A comprehensive survey. IEEE Transactions on Artificial Intelligence, 2024.

[58] Shahbaz Rezaei and Xin Liu. On the difficulty of membership inference attacks. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7888–7896, 2021.

[59] Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, Yann Ollivier, and Hervé Jégou. White-box vs black-box: Bayes optimal strategies for membership inference. In International Conference on Machine Learning, pages 5558–5567. PMLR, 2019.

[60] Ahmed Salem, Yang Zhang, Mathias Humbert, Pascal Berrang, Mario Fritz, and Michael Backes. ML-Leaks: model and data independent membership inference attacks and defenses on machine learning models. In 26th Annual Network and Distributed System Security Symposium, NDSS 2019, San Diego, California, USA, February 24-27, 2019. The Internet Society, 2019.

[61] Virat Shejwalkar and Amir Houmansadr. Membership privacy for machine learning models through knowledge transfer. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 9549–9557, 2021.

[62] Virat Shejwalkar, Huseyin A Inan, Amir Houmansadr, and Robert Sim. Membership inference attacks against nlp classification models. In NeurIPS 2021 Workshop Privacy in Machine Learning, 2021.

[63] Yi Shi and Yalin E Sagduyu. Membership inference attack and defense for wireless signal classifiers with deep learning. IEEE Transactions on Mobile Computing, 22(7):4032–4043, 2022.

[64] Reza Shokri, Marco Stronati, Congzheng, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In 2017 IEEE symposium on security and privacy (SP), pages 3–18. IEEE, 2017.

[65] Liwei Song and Prateek Mittal. Systematic evaluation of privacy risks of machine learning models. In 30th USENIX security symposium (USENIX security 21), pages 2615–2632, 2021.

[66] Shuang Song, Kamalika Chaudhuri, and Anand D Sarwate. Stochastic gradient descent with differentially private updates. In 2013 IEEE global conference on sig nal and information processing, pages 245–248. IEEE, 2013.

[67] Anshuman Suri, Xiao Zhang, and David Evans. Do parameters reveal more than loss for membership inference? In Proceedings ofthe 41st International Conference on Machine Learning (ICML), volume 235 of PMLR, 2024.

[68] Latanya Sweeney. k-anonimity: A model for protecting privacy. International journal ofuncertainty,fuzziness and knowledge-based systems, 10(05):557–570, 2002.

[69] Latanya Sweeney. k-anonymity: A model for protecting privacy. International journal of uncertainty, fuzziness and knowledge-based systems, 10(05):557–570, 2002.

[70] Ali Abbasi Tadi, Saroj Dayal, Dima Alhadidi, and Noman Mohammed. Comparative analysis of membership inference attacks in federated and centralized learning. Information, 14(11), 2023.

[71] Xinyu Tang, Saeed Mahloujifar, Liwei Song, Virat Shejwalkar, Milad Nasr, Amir Houmansadr, and Prateek Mittal. Mitigating membership inference attacks by {Self-Distillation} through a novel ensemble architecture. In 31st USENIX security symposium (USENIX security 22), pages 1433–1450, 2022.

[72] Jiashu Tao and Reza Shokri. Range membership inference attacks. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pages 346– 361. IEEE, 2025.

[73] Anthony Therrien. Half a million lifestyle. https://www.kaggle.com/ds/4094599, 2023. Accessed Sep 2025.

[74] Marlon Tobaben, Hibiki Ito, Joonas Jälkö, Yuan He, and Antti Honkela. Impact of dataset properties on membership inference vulnerability of deep transfer learning. Advances in Neural Information Processing Systems, 38:67486–67537, 2026.

[75] Stacey Truex, Ling Liu, Mehmet Emre Gursoy, Lei Yu, and Wenqi Wei. Demystifying membership inference attacks in machine learning as a service. IEEE transactions on services computing, 14(6):2073–2089, 2019.

[76] Xurun Wang, Guangrui Liu, Xinjie Li, Haoyu He, Lin Yao, Zhongyun Hua, and Weizhe Zhang. Membership inference attack with partial features, 2025. arXiv: 2508.06244.

[77] Lauren Watson, Chuan Guo, Graham Cormode, and Alexandre Sablayrolles. On the importance of difficulty calibration in membership inference attacks. volume abs/2111.08440, 2021.

[78] Rui Wen, Zheng Li, Michael Backes, and Yang Zhang. Membership inference attacks against in-context learning. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security, pages 3481–3495, 2024.

[79] Bang Wu, Xiangwen Yang, Shirui Pan, and Xingliang Yuan. Adapting membership inference attacks to gnn for graph classification: Approaches and implications. In 2021 IEEE International Conference on Data Mining (ICDM), pages 1421–1426. IEEE, 2021.

[80] Di Wu, Saiyu Qi, Yong Qi, Qian Li, Bowen Cai, Qi Guo, and Jingxian Chen. Understanding and defending against white-box membership inference attack in deep learning. Knowledge-Based Systems, 259:110014, 2023.

[81] Hengyu Wu and Yang Cao. Membership inference attacks on large-scale models: A survey. arXiv preprint arXiv:2503.19338, 2025.

[82] Yutong Wu, Han Qiu, Shangwei Guo, Jiwei Li, and Tianwei Zhang. You only query once: An efficient label-only membership inference attack. In The Twelfth International Conference on Learning Representations, 2024.

[83] Hongyang Yan, Shuhao Li, Yajie Wang, Yaoyuan Zhang, Kashif Sharif, Haibo Hu, and Yuanzhang Li. Membership inference attacks against deep learning models via logits distribution. IEEE Transactions on Dependable and Secure Computing, 20(5):3799–3808, 2022.

[84] Dingqi YANG. Foursquare global-scale check-in dataset with user social networks. https://sites.google.com/site/yangdingqi/home/foursquaredataset. Accessed Sep 2025.

[85] Jiayuan Ye, Aadyaa Maddi, Sasi Kumar Murakonda, Vincent Bindschaedler, and Reza Shokri. Enhanced membership inference attacks against machine learning models. In Proceedings of the 2022 ACM SIGSAC conference on computer and communications security, pages 3093–3106, 2022.

[86] Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. Privacy risk in machine learning: Analyzing the connection to overfitting. In 2018 IEEE 31st computer securityfoundations symposium (CSF), pages 268–282. IEEE, 2018.

[87] Shi Yichuan, Olivera Kotevska, Viktor Reshniak, and Amir Sadovnik. Assessing membership inference attacks under distribution shifts. In 2024 IEEE International Conference on Big Data (BigData), pages 4127– 4131. IEEE, 2024.

[88] Sajjad Zarifzadeh, Philippe Liu, and Reza Shokri. Lowcost high-power membership inference attacks. arXiv, 2024.

[89] Minxing Zhang, Zhaochun Ren, Zihan Wang, Pengjie Ren, Zhunmin Chen, Pengfei Hu, and Yang Zhang. Membership inference attacks against recommender systems. In Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security, pages 864–879, 2021.

[90] Yan Zhou, Bradley Malin, and Murat Kantarcioglu. Smote-dp: Improving privacy-utility tradeoff with synthetic data. arXiv preprint arXiv:2506.01907, 2025.

## A Experimental Details

This section describes the full experimental setup of the implemented MIAs. All experiments were conducted on Ubuntu 24.04 LTS. The system is an NVIDIA GB10 (Grace Blackwell) ARM-based unified-memory node, with a 20-core SoC (10 Cortex-X925 + 10 Cortex-A725) and around 121 GiB of unified CPU/GPU memory shared with a single onboard NVIDIA GB10 GPU. The experiments were run with CUDA 13.0. This setup was used for model inference and attack execution.

## A.1 Data

The characteristics of the nine datasets used to evaluate the MIAs are presented in Table 4. We use datasets that span a range of sample sizes, features, and labels. The domains include crime, healthcare, finance, education, locations and purchases. Table 5 presents the quasi-identifiers of seven datasets alongside the range of the number of single-outs across the five seeds.

<table><tr><td>Dataset</td><td>Domain</td><td>n</td><td>d</td><td>c</td><td>r</td></tr><tr><td>COMPAS Recidivism [29]</td><td>Crime</td><td>4937</td><td>20</td><td>2</td><td>1.05</td></tr><tr><td>Covid [4]</td><td>Healthcare</td><td>2671</td><td>29</td><td>3</td><td>6.3</td></tr><tr><td>Diabetes Readmission [30]</td><td>Healthcare</td><td>81410</td><td>150</td><td>2</td><td>1.17</td></tr><tr><td>Half Million [73]</td><td>Finance</td><td>15855</td><td>28</td><td>12</td><td>1.08</td></tr><tr><td>Locations [36,84]</td><td>Locations</td><td>5010</td><td>446</td><td>10</td><td>3.18</td></tr><tr><td>Mammography [35]</td><td>Healthcare</td><td>11183</td><td>24</td><td>2</td><td>42.01</td></tr><tr><td>MIC (Myocardial infarction) [55]</td><td>Healthcare</td><td>1699</td><td>122</td><td>8</td><td>119</td></tr><tr><td>Purchases-10 [12,36]</td><td>Purchases</td><td>19766</td><td>600</td><td>10</td><td>5.28</td></tr><tr><td>Student Performance [11]</td><td>Education</td><td>648</td><td>30</td><td>3</td><td>3.55</td></tr><tr><td>Surgery [48]</td><td>Healthcare</td><td>14635</td><td>24</td><td>2</td><td>2.97</td></tr></table>

Table 4: Specifications of the evaluation datasets. n denotes the number of samples, d the number of features, c the number of classes, r the imbalance ratio.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Selected quasi-identifiers</td><td colspan="2"># Single-outs</td></tr><tr><td>k= 1</td><td>k=2</td></tr><tr><td>COMPAS</td><td>Age, Race:Native_American, Race:Other</td><td>16-23</td><td>34-51</td></tr><tr><td>Covid</td><td>Age, Cerebro Vascular, Diabetes</td><td>5-10</td><td>16-25</td></tr><tr><td>Diabetes</td><td>time_in_hospital, race:Other, age:{[}50-70), admission_type_id:Trauma Center, discharge_disposition_id:Discharged to Home, discharge_disposition_id:Other</td><td>11-18</td><td>23-43</td></tr><tr><td>Half Million</td><td>Gender, State, Country</td><td>180-216</td><td>592-620</td></tr><tr><td>MIC</td><td>Age, Obesity, Chronic Pneumonia</td><td>14-19</td><td>24-31</td></tr><tr><td>Student Perf.</td><td>School, Address, MJob, School Sup</td><td>4-12</td><td>13-22</td></tr><tr><td>Surgery</td><td>gender, mort30, mortality_rsi</td><td>267-313</td><td>628-669</td></tr></table>

Table 5: Selected quasi-identifiers for each dataset and corresponding number of single-outs range across the 5 seeds.

## A.2 Model architectures

All models are tuned with Optuna (30 trials, 3-fold CV) to reflect realistic model development practices and obtain competitive configurations. The MLP includes three hidden layers (256, 256, 128 units), each followed by a linear Batch Normalization layer and a ReLU activation layer, ending with a linear classification layer. It uses an SGD optimizer with learning rate 0.01, weight decay $1 0 ^ { - 4 }$ , batch size 256 and 100 epochs. We use learning rate over $[ 1 0 ^ { - 4 } , 1 0 ^ { - 1 } ]$ , weight decay over $[ 1 0 ^ { - 6 } , 1 0 ^ { - 2 } ] .$ , batch size over $\{ 1 2 8 , 2 5 6 , 5 1 2 \}$ , and the optimizer over {SGD,Adam,AdamW}.

For RF, Optuna searches over n\_estimators ∈ $\{ 5 0 , 5 0 0 \}$ max $\_ d e p t h \in$ {None,3,5,8,10,15,20}, min\_samples\_split $\in \quad \{ 2 , 2 0 \}$ , min\_samples\_lea f ∈ $\{ 1 , \operatorname* { m a x } ( 1 0 , \lfloor n / 2 0 0 \rfloor ) \}$ and max\_features over a datasetdependent continuous interval. In the DT model, the search spans over max\_depth ∈ {4,6,10} and min\_samples\_ $\ b { l e a f } \in$ {1,3,5,7}. Moreover, for XGBoost, the gamma $\in \{ 0 , 0 . 5 , 1 \}$ max\_depth ∈ {5,10,15,20}, learning\_rate $\in \{ 0 . 1 , 0 . 0 1 \}$ min\_child\_weight $\in \ \{ 1 , 3 , 5 , 7 \}$ , colsample\_bytree and subsample $\in \ \{ 0 . 8 , 1 \}$ parameters are optimized. With NB, the adjustment is made to the var\_smoothing ∈ $\{ 1 0 ^ { k } : k \in l i n s p a c e ( - 9 , 0 , 1 0 0 ) \}$ parameter. Finally, LR searches over $C \ \in \ \{ 0 . 0 0 0 1 , 0 . 0 0 1 , 0 . 0 1 , 0 . 1 , 1 , 1 0 , 1 0 0 \}$ solver ∈ {lb f gs,newton-cg,saga,sag,newton-cholesky} and penalty set to l2.

## B Performance of the target models

Table 6 shows that DT is the model that overfits the most overall, while LR and NB are the safest models. RF and XGB present a greater leakage than their generalization gap suggests. Meanwhile, MLP’s generalization gap and MIA AUC generally track together. Even though the tree-based models leak more they also show the highest predictive power in average (from 73.1% to 81.6%), similarly to MLP’s utility with 73.9% on average. Notably, LR’s lower leakage pairs with its high predictive power (76.4% average target accuracy), which is just 2.8 pp above the average performance. NB shows to be the model with the lowest predictive power, with only 55.4% accuracy (-18.1 pp lower than the average).

<table><tr><td rowspan="2">Model</td><td colspan="2">COMPAS</td><td colspan="2">MIC</td><td colspan="2">Surgery</td></tr><tr><td>Gen. gap (%)</td><td>MIA AUC (%)</td><td>Gen. gap (%)</td><td>MIA AUC (%)</td><td>Gen. gap (%)</td><td>MIA AUC (%)</td></tr><tr><td>RF</td><td>3.1 ± 2.9</td><td>59.0 ± 4.2</td><td>1.3 ± 1.1</td><td>70.6 ± 5.2</td><td>6.7 ± 0.7</td><td> $7 7 . 0 \pm 2 . 5$ </td></tr><tr><td>MLP</td><td>10.5 ± 8.8</td><td> $6 0 . 0 \pm 5 . 8$ </td><td> $7 . 0 \pm 4 . 9$ </td><td> $5 9 . 1 \pm 4 . 0$ </td><td> $1 7 . 7 \pm 7 . 9$ </td><td> $6 4 . 3 \pm 5 . 3$ </td></tr><tr><td>NB</td><td> $0 . 8 \pm 1 . 7$ </td><td> $5 0 . 5 \pm 0 . 8$ </td><td> $7 . 9 \pm 2 . 0$ </td><td> $5 6 . 0 \pm 1 . 4$ </td><td> $0 . 1 \pm 0 . 7$ </td><td> $5 0 . 2 \pm 0 . 7$ </td></tr><tr><td>DT</td><td>42.3 ± 1.2</td><td>74.2 ± 0.6</td><td>18.0 ± 1.8</td><td>62.8 ± 1.2</td><td> $1 8 . 9 \pm 0 . 4$ </td><td> $6 2 . 3 \pm { 0 . 4 }$ </td></tr><tr><td>XGB</td><td>16.3 ± 1.3</td><td> $6 8 . 3 \pm 0 . 9$ </td><td> $1 0 . 6 \pm 0 . 5$ </td><td> $7 0 . 9 \pm 2 . 2$ </td><td>6.2 ± 0.3</td><td> $6 4 . 3 \pm 0 . 7$ </td></tr><tr><td>LR</td><td> $0 . 6 \pm 1 . 3$ </td><td> $5 1 . 5 \pm 0 . 7$ </td><td> $1 0 . 8 \pm 2 . 3$ </td><td> $6 2 . 6 \pm 1 . 2$ </td><td> $0 . 2 \pm 0 . 6$ </td><td> $5 1 . 5 \pm 0 . 5$ </td></tr></table>

Table 6: Comparison of the generalization gap of the target model with the RMIA AUC.

## C Attack performance on ten datasets

General attacks’ performance is present in Figure 7. We observe that RMIA dominates all the remaining attacks. We

also teste online version of LiRA and RMIA showing small improvements, expect for LiRA in RF and XGB models.

![](images/9504754855d70197fdfa722a1998babc5acf293ef4f53bf0ffe1cc8aa0b361b9.jpg)  
Figure 7: MIA AUC results for all attacks and models, averaged across all datasets.

## D Reference models on RMIA

Figure 8 extends the comparison of attack performance between surrogate RMIA and Attack-P/QMIA for all datasets, via TOST equivalence test. Here, we provide the results for 1 and 5 reference models since there was little improvement in comparing with more models, as stated in Section 6.2. Six datasets stay within the equivalence margin band, highlighting the equivalence of RMIA versus reference-model free attacks. In contrast, this does not fully happen for Student Performance, especially in the 5 reference-model case, likely because it is a small and overfit-prone dataset.

![](images/d66bf758a55300000265a5eab4c18aa21e797bb45167ded96e9b5098b3744b20.jpg)  
Figure 8: Reference model equivalence gap for surrogate RMIA (1 and 5 reference models) versus Attack-P/QMIA. Each dataset point shows the mean MIA AUC gap, with the error gaps representing the Standard Error Mean across surrogate models. The shaded band denotes the 0.05 margin used in the TOST equivalence test.

## E Comparison of single-out MIAs

To assess how each model identifies SO records, Figure 9 compares mean AUCs for SO and non-SO records, with each point representing a dataset-model pair averaged. The box shows the percentage of pairs where SO AUC exceeds non-SO AUC and the one-sided paired Wilcoxon signed-rank p-value testing whether SO AUC is systematically higher.

![](images/01b925761f9167479286dfd7decd15b9be5956ed98c57b9a29ae1bf6a5551643.jpg)  
Figure 9: SO vs. non-SO attack AUC, faceted by attack and anonymity threshold k.

Across panels, most points lie above the diagonal, indicating that SO records are generally more vulnerable to MIAs than non-SO records. The strength of this effect also depends on the attack. RMIA and LiRA form tight bands along the diagonal, with SO AUC consistently slightly higher than non-SO AUC across the full range of attack performance, indicating a uniform and reproducible SO vulnerability effect across dataset/model combinations. In contrast, Attack-P and QMIA show substantially greater dispersion, particularly at lower AUCs (0.5–0.65). QMIA shows the weakest and least reliable SO effect, which further weakens at k = 2, where it no longer reaches significance and its win rate approaches chance.

## F Effectiveness of defenses over all datasets

Figure 10 shows the impact of defenses on all attacks. RMIA and LiRA present a greater drop in single-out catch, following the MIA AUC reduction. For the reference-model free approaches, the drop in attack success does not translate to reduced vulnerability of single-outs.

![](images/e07bcf7a5f60f6a23c5e11a1f9ef778f07497a71faf686baa264788de1ec8068.jpg)  
Figure 10: Resulting application of defenses on the catch of single-outs for k=1. Each point represents a dataset-defense combination, averaging across the 6 models.