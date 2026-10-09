# What is the goal of unsupervised machine learning?

Aapo Hyv¨arinen

Department of Computer Science, University of Helsinki, Pietari Kalmin katu 5, Helsinki, 00560, Finland.

Corresponding author(s). E-mail(s): aapo.hyvarinen@helsinki.fi;

## Abstract

Unsupervised learning is one of the main branches of machine learning. Here I argue that unlike the other branches of machine learning (supervised and reinforcement learning), unsupervised learning is a rather heterogenous field that can serve several diferent goals. It seems futile to try to define one single goal for unsupervised learning. I identify four diferent goals for unsupervised learning: 1) Estimating the distribution, 2) Generating new data points, 3) Extracting features for downstream tasks, and 4) Understanding the data.

Keywords: Unsupervised learning, Deep learning, Probabilistic models

## 1 Introduction

Machine learning is classically divided into supervised learning, unsupervised learning, and reinforcement learning. Roughly speaking, in supervised learning, some kind of prediction accuracy is maximized, and in reinforcement learning, the total reward is maximized. In those cases, we have relatively well-defined goals on which there is a large consensus.

But what is maximized in unsupervised learning, or what is its goal? The starting point in this paper is that the answer to that question is complicated, even controversial. It does not seem possible to point out one single goal even on a conceptual level, let alone a single objective function that would be maximized.

Instead, my claim here is that unsupervised learning has several diferent goals which are quite diferent from each other. Some of them are orthogonal in the sense that trying to reach one goal does not in any way move towards another goal. In fact, the goals may even be a bit contradictory with each other.

Note that the discussion in this paper also applies to self-supervised learning (SSL). While there are diferent viewpoints in the literature, I consider SSL as an algorithmic technique that uses supervised algorithms to accomplish the goals of unsupervised learning. Thus, the goals themselves are the same for SSL and unsupervised learning.

I tentatively specify four diferent goals for unsupervised learning. There is no claim that this list is complete, and other goals could probably be identified. My list includes the following goals:

1. Estimating the distribution

2. Generating new data points

3. Extracting features for downstream tasks

4. Understanding the data

I will try to justify why these can be seen as separate goals in unsupervised learning, and how they relate to each other. Some of the basic methods for each goal are pointed out, but this is not an attempt to make a comprehensive review of the topic. Particular emphasis is here laid on how such learning methods could be validated in view of each of the goals. Related analysis has been presented by [1].

## Definitions

Assume we have an observed data set $\mathbf { X } = \{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { N } \}$ , where each data point $\mathbf { x } _ { i }$ is in a real space $\mathbb { R } ^ { n }$ . In the case of supervised learning, it is further assumed that to each $\mathbf { x } _ { i }$ is associated a label or target $\mathbf { y } _ { i }$ but this is not available in the basic unsupervised case. We further take a probabilistic viewpoint, and make some further assumptions: a) the $\mathbf { x } _ { i }$ are observations of a random variable x that follows the pdf $p _ { \mathbf { x } ; \mathrm { ~ b ~ } }$ the observations are independent of each other, in other words $\mathbf { x } _ { i }$ form an i.i.d. sample. The latter assumption could be relaxed, but it presents the most fundamental case.

## 2 Proposed list of goals for unsupervised learning

## Goal #1: Estimating the distribution

## Problem statement

Perhaps the most fundamental goal of unsupervised learning is learning the probability distribution of the data. This typically takes of the form of learning the probability density function $p _ { \mathbf { x } }$ , or its logarithm, from the data set X. Alternatively, in some recent developments, the score function $\nabla _ { \mathbf { x } } \log p _ { \mathbf { x } }$ is the target of learning [2–4].

From an extreme probabilistic viewpoint, that is all that is ever needed. In theory, even supervised learning can be solved if the relevant densities are known. For example, consider classification with two classes, with pdf’s $p _ { 1 }$ and $p _ { 2 }$ . After learning the densities separately for the two classes, the classification problem can be solved by assigning any data point x to class 1 $\mathrm { i f } p _ { 1 } ( \mathbf { x } ) > p _ { 2 } ( \mathbf { x } )$ and to class 4 otherwise. This is a special case of using the density ratio $p _ { 1 } / p _ { 2 }$ which has many other utilities as well [5]. In fact, even basic logistic regression converges to compute exactly this ratio [6].

Furthermore, a more general form of the regression problem can be formulated by considering the conditional pdf $p ( \mathbf { y } \vert \mathbf { x } )$ . This can be accomplished, for example, by learning the pdf of a data vector where the regressors and regressands are concatenated: $\mathbf { z } _ { i } = ( \mathbf { x } _ { i } , \mathbf { y } _ { i } )$ . Now for any new (conditioning) data point $\mathbf { x } _ { k }$ , we only need to consider the function $g ( \pmb { \eta } ) = p _ { \mathbf { z } } ( \mathbf { x } _ { k } , \pmb { \eta } )$ which is equal to the conditional pdf $p ( \mathbf { y } \vert \mathbf { x } )$ up to a constant (which is $1 / p _ { \mathbf { x } } ( \mathbf { x } ) )$ . Below, it will be seen how such conditioning is also useful in data generation.

The downside with merely performing density estimation is that it may not enable us to understand the structure of the data, since the density approximator is often an uninterpretable black box given by a neural network [3, 7]. Nor is it clear if it is a practical method for solving any supervised tasks; the connections shown above are of a more theoretical nature. These problems will serve as motivations for the other goals below.

## Methods

The classic statistical method for general (“non-parametric”) density estimation is the kernel density estimator [8]; unfortunately, such methods do not work well in high dimensions. Methods based on reproducing kernel Hilbert spaces are a related option [9, 10].

Deep learning research focuses on Energy-Based Modelling and Score-Based Modelling, using methods stemming from score matching [2] or noise-contrastive estimation [6]. Such methods tend to be state-of-the-art (or close) in high-dimensional spaces, although they each pose their specific technical problems, as discussed by [7].

Normalizing flows [11] are another approach. In general, it is very dificult to use using maximum likelihood to estimate the pdf approximated by a a neural network. Normalizing flows make this possible by imposing suitable restrictions on the neural network. However, the quality of the density approximation may sufer from such restrictions.

## Validation

Now we discuss the problem of validating a density estimation method after it has been learned, i.e. showing that it works, or not; or how it compares to other methods. As in the rest of this paper, the discussion focuses on real-valued vector data.

In some cases, the density estimation methods can be validated by computing the objective function on a held-out test data set. Suppose we have two neural networks defined by diferent architectures, both trained with one of the estimation methods above (say, score matching) to approximate the log-pdf. For those methods that work by minimizing an objective function, one can compute the objective (e.g. the score matching objective) on the test data set, which enables comparison of the two neural networks.

However, the objective function values usually have no clear interpretation and it is therefore dificult to say if the value of the objective is small or large, i.e. if the model fit is good or not. The same problem actually occurs with ordinary maximum likelihood: It is usually not possible to say if a given value of likelihood is high or low in some intuitive sense. This is in contrast to classification, where it is clear that, say, 99% classification accuracy is quite high in most applications. Thus, this validation method can only perform a comparison between two architectures or similar cases.

Another widely-used validation method is to use the learned distribution to generate new data by sampling, which will be discussed below. However, this is merely redefining the problem as the validation of data generation, which turns out to be very dificult, as will also be discussed below.

## Goal #2: Generating new data points

## Problem statement

A rather recent development in machine learning is generative AI, which is typically formalized as sampling a data point from the pdf of the data $p _ { \mathbf { x } } .$ . In practice, the distribution is conditioned by a prompt or similar input u, so the sample is actually generated from $p _ { \mathbf { x } | \mathbf { u } }$ . For simplicity, though, we talk about sampling from $p _ { \mathbf { x } }$ in what follows.

While such sampling is a classic problem in statistics, modern generative AI has emphasized the case where $p _ { \mathbf { x } }$ is not given as an analytical formula, as in classical theory, but the sampling system is learned from the observed data set $\mathbf { x } _ { i }$ alone. Such a sampling framework is, in particular, the basis for image generation methods. (Text generation uses rather diferent frameworks and is not considered here.)

## Methods

Recent real-space data generation methods in AI typically use a two-step procedure, where an estimate of the pdf $p _ { \mathbf { x } }$ is first obtained, perhaps in the form of the score function $\nabla _ { \mathbf { x } } \log p _ { \mathbf { x } } \ [ 2 , \ 3 , \ 7 ]$ . Thus, this first step is utilizing the methods described above in relation to Goal $\# 1$ above. As the second step, any method for sampling from a given pdf (or score function) is used. Typical sampling methods include Langevin methods [4, 12] and (generative) difusion models [13, 14]. Such sampling can also happen in a feature space found by another unsupervised learning system [15].

But sampling can also be performed by special methods that do nothing else than sampling. One of the triggers of the generative AI boom was the method of Generative Adversarial Networks or GANs [16], which directly learns to generate new data points from $p _ { \mathbf { x } }$ without performing any density estimation or similar. In fact, many latent variables models, such a variational autoencoders and nonlinear independent component analysis [17, 18] can also generate new data points; however, the results may not be competitive with methods that are specifically created or optimized for data generation.

## Validation

Validation of such data generation is quite challenging. There is no simple obvious measure that would tell how closely a generated data point, or even a set of generated points, follows the distribution of the original data set X. Another caveat is that it is important to make sure the sampling system is not just copying the training data points.

In the case of image generation, visual inspection of the generated images is an obvious choice, but this is clearly not a rigorous, quantitative measure. Researchers have developed some standard measures based on benchmark data sets, so-called “inception” scores or distances [19, 20]. These typically measure how the generated images fit with the classes of the benchmark dataset such as ImageNet. However, they are restricted to domains where a labelled benchmark dataset is available, such as images.

A more general possibility would be to check if a deep neural network can learn to discriminate the generated data points from the original data points. However, since this measure is in fact optimized in the GAN learning process, using it as a new measure does not give any new information if GAN is used as the generator. Moreover, this measure depends on the architecture and training procedure of the neural network that is trained in calculating the measure. Thus, a general solution to this validation problem is hard to find.

## Goal #3: Extracting features for downstream tasks

## Problem statement

Feature extraction means that we learn a function $\mathbf { f } : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ , called the feature extractor, such that the features in $\mathbf { z } = \mathbf { f } ( \mathbf { x } )$ are somehow useful. This is a special case of representation learning [21]. The utility of such features can take many forms, but we consider here the particular case where the features are used for further learning in another task, called the “downstream” task. The downstream task is typically supervised, but it could also be reinforcement learning, and it could even be a further unsupervised task, based on the two goals above. Here we focus on a supervised downstream task, in which case this is closely related to what is called semi-supervised learning.

The motivation of semi-supervised learning is that it is often easy to obtain a large number of data points $\mathbf { x } _ { i }$ that have no labels or targets $\mathbf { y } _ { i }$ , while any labels or targets are more dificult or expensive to obtain. Thus, we can have a large dataset without labels, and only a small data set that has proper labels. For example, one can find, on the Internet, huge amounts of images and text that have no specific information on their contents (e.g. does the image depict a cat or or a dog); these form the unlabelled data set $\mathbf { x } _ { i } .$ Labels can be created, for example, by human experts but this is expensive and can only be done for a small subset. Let us call the subset with labels $\tilde { \mathbf { x } } _ { j }$ with labels $\tilde { \mathbf { y } } _ { j }$

Now, the goal of unsupervised learning in this case is to first learn to transform the data into features $\mathbf { z } = \mathbf { f } ( \mathbf { x } )$ based on the unlabelled data alone. The features z should be such that it is much easier to perform the “downstream” supervised learning task of predicting y from x by using the features z to predict y instead of the original x. Thus, learning the downsteam task becomes possible even using the small labelled dataset consisting of ˜x and $\tilde { \mathbf { y } }$ and the features ˜z.

## Methods

A large number of methods fall into this category. In fact, almost any neural network trained in an unsupervised manner can be hoped to give useful features in its hidden layers. Recent research has further produces a large number of methods based on selfsupervised learning (SSL) principles. Among SSL we can distinguish two main strands: autoencoders and contrastive learning.

Autoencoders [22] are in fact one of the oldest methods for unsupervised learning, having their roots in the early neural networks research in the 1980’s. The basic principle is to predict x by a neural network that takes the same x as input, but has a bottleneck such as a hidden layer with a small number of units. Autoencoders have been recently revived in the form of variational autoencoders or VAEs [17] which provide a principled probabilistic framework. However, it is often not clear why such a probabilistic VAE should be preferred to a very basic autoencoder. Recent research has also emphasized methods where instead of a bottleneck in the neural network, the input is corrupted, for example by adding noise [23].

Contrastive learning is a term used in quite diferent meanings by diferent authors. The main thing most of these methods have in common is that they transform the unsupervised learning into a classification problem (as opposed to ordinary regression as in autoencoders). A fundamental probabilistic framework is provided by noisecontrastive learning [6], while a rather diferent approach is provided by SimCLR [24]. Contrastive learning approaches have also been used in the context of nonlinear ICA [18].

For downstream tasks related to reinforcement learning, the seminal methods include the successor representation [25] and proto-value functions [26]. While these methods may not usually be considered unsupervised learning methods, they explore the action space without referring to the rewards, and are in that sense unsupervised.

## Validation

Validation of whether this goal has been achieved is straightforward in the framework of semi-supervised learning: We only need to measure the performance on the downstream task by some conventional measure such as classification accuracy. Likewise, comparing several unsupervised feature extraction methods is straightforward using the same measure. Thus, the semi-supervised learning framework is interesting also in the sense that its performance can be easily measured; it provides one objective way of comparing diferent unsupervised learning algorithms. However, our emphasis here is that this is only one possible goal of unsupervised learning, and it would be ill-advised to evaluate or compare any unsupervised learning methods by this metric.

## Goal #4: Understanding the data

## Problem statement

Especially in scientific data analysis, the main goal of machine learning is to further the understanding of the structure of the data and the phenomena being measured. It is actually quite commonplace to train a classifier on scientific data, where the classes correspond to something like experimental conditions. But even if such a classifier is trained, the real goal may not be to predict anything, but to understand how the prediction is performed by the system. If a linear regressor or a linear classifier is trained on the data, interpretation of the regression coeficients is straightforward; if a neural network is used, understanding how it functions is notoriously dificult.

Regarding unsupervised learning, a common approach is to find components or factors as a basis for understanding the data. These features may not be very diferent from the features considered in Goal #3 in semi-supervised learning. Thus, the same components may sometimes be useful for both the current goal and Goal #3. In fact, understanding and interpretation by a human investigator could be seen as another downstream task, at least on a very abstract level. But there is actually a clear diference between the goals: the features in Goal #3 may not facilitate understanding the data if the feature extraction is performed by a complex neural network whose computation is too dificult to understand by humans; or simply, if the number and size of the features is overwhelmingly large. Thus, it is justified to consider understanding the data a distinct goal of learning.

Identifiability is a key concept with this goal [18, 27]. It means that the features to be learned are unique and well-defined. Many widely used feature extraction methods only give the features up to a linear transformation, which makes any attempt of interpretation futile. It is important to make sure that a method with identifiable features is used, if the goal is to interpret them. This is another diference to Goal #3, where identifiability may be much less important (but see [28, 29] who propose that identifiability may be important for learning features that transfer to new tasks).

## Methods

Latent variable models provide a widely used framework. Each latent variable or component can hopefully be interpreted as corresponding to some phenomenon or mechanism underlying the data. For example, in neuroscience, brain activity can often be decomposed into a number of linear components [30]. On the other hand, if the dimension of the data is reduced so that the number of components is only two or three, the distribution of the data points can also be visualized in a scatterplot or similar.

To facilitate interpretation, simple unsupervised models based on linear components are often preferred. While the classic linear factor analysis and principal component analysis sufer from lack of identifiability, Independent Component Analysis provides an identifiable alternative based on non-Gaussianity [31–33]. Related methods achieve identifiability by temporal dependencies [33, 34] or nonnegativity [35].

In the nonlinear case, methods related to manifold learning, such as autoencoders and VAE are also widely used. However, such models typically have no identifiability guarantees, and thus the interpretation of the features is highly questionable. For example, in a typical VAE, any orthogonal transformation of the features is equally good, and thus the features are only defined up to an orthogonal transformation; actually there are even stronger indeterminacies, as discussed by [18, 36].

Nonlinear component methods are in fact particularly dificult to identify [37]. Recent advances show, however, that they become identifiable with suitable temporal dependecies, or related departures from independent sampling [18, 27]. Sometimes even such deep learning feature extraction methods may give insight to the structure of the data, if suitable methods for visualization of the features are found; some examples are shown by [38, 39].

In addition to component models, other frameworks can be used. For example, causal discovery attempts to learn the causal connections and directions between the variables [40–42]. This is clearly interesting from the viewpoint of understanding the variables in terms of a system with interactions, and it is paramount from the viewpoint of designing interventions. Data points can also be divided into clusters, and each cluster can be given an interpretation in terms of a subpopulation of individuals, subtypes of diseases, spatial regions, or similar.

Arguably, this goal strongly overlaps with classic statistical parameter estimation. Estimating a statistical model with a small number of parameters that have some meaning is a well-known method for understanding data. As a trivial example, simply estimating the mean and variance will give some information about the data which is easy to interpret. The discussion just given actually largely considered esti mation of statistical models, but focusing on general-purpose models estimated on high-dimensional big data, which is the focus of machine learning.

## Validation

Unfortunately, validation of such results is dificult, almost by definition. The “understanding” that is the goal here should happen in the brains of humans who receive the results of the unsupervised learning as input. It is thus dificult to validate if any understanding was obtained. Asking domain experts is often the only proper way of validating these results.

In the case of causal discovery, the situation is a bit diferent since there is often a ground truth that one tries to recover, i.e. the true causal directions between the variables. In principle, that direction can often be measured by experiments, but that maybe very dificult; the dificulty of performing such experiments is the very motivation of developing causal discovery methods. Likewise, in some cases, ground truth clusters or even components may be known in some special cases.

## 3 Discussion

## 3.1 Connections between the goals

As already pointed out, some of the goals are related. An approximation of the pdf learned in Goal #1 can be directly used for sampling in Goal #2, although sampling can also be achieved in other ways. Likewise, learning features helps both in downstream tasks in Goal #3 and in understanding the underlying structure of the data as in Goal #4, although the optimal features may be diferent in the two cases.

Some of the goals are clearly orthogonal in the sense that one goal can be achieved without advancing towards another. In particular, Goals #1 and #2 need a nonlinear function approximator, but the approximator can be a total black box whose inner structure is not available. In such a case, the approximator would not enable understanding the data as in Goal #4, and might not even yield any features as in Goal #3. In this sense, Goals #1 and #2 are orthogonal to the Goals #3 and #4. In practice, though, if the density is approximated and the sampling is enabled by a neural network, its hidden layers would usually be available as features. Still, such a neural network might not enable any understanding of the data because of the dificulties in interpreting neural networks.

There might even be some contradiction between the goals in the sense that achieving one goal makes achieving another goal more dificult. I think this would in particular be the case regarding the goal of understanding the data (Goal #4), which is best achieved by using simple, even linear functions, while such linear functions would not always be very useful for the other goals.

## 3.2 Criticism of this list and possible further goals

I make no claim that the list of the four goals above is exhaustive or definitive. Further goals can certainly be found, depending on what methods are actually considered to belong to unsupervised machine learning, and some of the goals here may also be found to be ill-formulated.

One objection that could be made is that these four goals are not comparable in their practical relevance. More precisely, it could be argued that density estimation (Goal #1) is not a real goal of unsupervised learning, but only an intermediate goal for something else that needs to be specified. This is in contrast to Goals #2–#4, which all give outputs to the user that solve a real problem — at least after the downstream task in Goal #3 is also performed. Perhaps the main justification for including Goal #1 in this list is that it is so fundamental from the viewpoint of theory and it has so many diferent applications, that it makes sense to consider it a separate goal.

Furthermore, causal learning has many diferent aspects some of which may not fit the four goals given here. While above I considered it from the viewpoint of understanding the data, it is also useful, for example, in reinforcement learning, or in general action selection, where it is important to intervene on the causes instead of the efects [43, 44]. It could thus be argued that causal discovery is a separate goal of unsupervised learning.

Data compression could also be seen as a separate goal. This would not be about simply doing PCA in order to better classify data, but performing compression, typically in some feature space, for facilitating data transmission or storage. It takes many forms, but one specific one would be vector quantization, in the particular sense of coding n-dimensional real-valued vectors corresponding to some features as binary strings. However, this may not be a topic of great interest in the machine learning community at the moment. Another candidate goal is data restoration or cleaning, including denoising, restoring missing values, and similar tasks.

## 3.3 Related work

The goals in the paper are closely related to the work by [1]. However, they consider the problem of evaluating a given statistical generative model and show how diferent evaluation methods (similar to what we discussed above) lead to diferent rankings of such models. They don’t seem to consider the idea that the learning could have completely diferent goals. Nevertheless, they essentially propose something like my Goals #1, #2, and #3 from the viewpoint of evaluating probabilistic generative models; I add Goal #4 (Understanding the data).

## 4 Conclusion

As an answer to the question in the title, I proposed that unsupervised learning does not have a single goal. Instead, it has diferent goals which can be quite diferent and

orthogonal to each other. I gave a list of four goals, but this list can be extended in future work. It would also be interesting to see if a more rigorous taxonomy of the goals could be developed.

## Conflict of Interest Statement

The Author has no conflicts of interest to declare.

## References

[1] Theis, L., Oord, A.v.d., Bethge, M.: A note on the evaluation of generative models. In: Proc. Int. Conf. on Learning Representations (ICLR2016) (2016)

[2] Hyv¨arinen, A.: Estimation of non-normalized statistical models using score matching. J. of Machine Learning Research 6, 695–709 (2005)

[3] Vincent, P.: A connection between score matching and denoising autoencoders. Neural computation 23(7), 1661–1674 (2011)

[4] Song, Y., Ermon, S.: Generative modeling by estimating gradients of the data distribution. In: Advances in Neural Information Processing Systems, pp. 11895– 11907 (2019)

[5] Sugiyama, M., Suzuki, T., Kanamori, T.: Density Ratio Estimation in Machine Learning. Cambridge University Press (2012)

[6] Gutmann, M.U., Hyv¨arinen, A.: Noise-contrastive estimation of unnormalized statistical models, with applications to natural image statistics. J. of Machine Learning Research 13, 307–361 (2012)

[7] Song, Y., Kingma, D.P.: How to train your energy-based models. arXiv preprint arXiv:2101.03288 (2021)

[8] Chen, Y.-C.: A tutorial on kernel density estimation and recent advances. Biostatistics & Epidemiology 1(1), 161–187 (2017)

[9] Sriperumbudur, B., Fukumizu, K., Gretton, A., Hyv¨arinen, A., Kumar, R.: Density estimation in infinite dimensional exponential families. J. of Machine Learning Research 18, 1–59 (2017)

[10] Wenliang, L., Sutherland, D.J., Strathmann, H., Gretton, A.: Learning deep kernels for exponential family densities. In: Proceedings of the 36th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 97, pp. 6737–6746 (2019)

[11] Papamakarios, G., Nalisnick, E., Rezende, D.J., Mohamed, S., Lakshminarayanan, B.: Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research 22(57), 1–64 (2021)

[12] Hyv¨arinen, A.: A noise-corrected Langevin algorithm and sampling by halfdenoising. Transactions on Machine Learning Research (2025)

[13] Croitoru, F.-A., Hondru, V., Ionescu, R.T., Shah, M.: Difusion models in vision: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence (2023)

[14] Yang, L., Zhang, Z., Song, Y., Hong, S., Xu, R., Zhao, Y., Zhang, W., Cui, B., Yang, M.-H.: Difusion models: A comprehensive survey of methods and applications. ACM Computing Surveys 56(4), 1–39 (2023)

[15] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695 (2022)

[16] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. In: Advances in Neural Information Processing Systems, pp. 2672–2680 (2014)

[17] Kingma, D.P., Welling, M.: Auto-encoding variational Bayes. In: Proc. Int. Conf. on Learning Representations (ICLR2014), Banf, Canada (2014)

[18] Hyv¨arinen, A.., Khemakhem, I., Morioka, H.: Nonlinear independent component analysis for principled disentanglement in unsupervised deep learning. Patterns 4(10), 100844 (2023)

[19] Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., Chen, X.: Improved techniques for training gans. Advances in neural information processing systems 29 (2016)

[20] Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)

[21] Bengio, Y., Courville, A., Vincent, P.: Representation learning: A review and new perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence 35(8), 1798–1828 (2013)

[22] Bank, D., Koenigstein, N., Giryes, R.: Autoencoders. Machine learning for data science handbook: data mining and knowledge discovery handbook, 353–374 (2023)

[23] Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., Manzagol, P.-A.: Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. J. Mach. Learn. Res. 11, 3371–3408 (2010)

[24] Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A simple framework for contrastive learning of visual representations. In: International Conference on Machine Learning, pp. 1597–1607 (2020). PMLR

[25] Dayan, P.: Improving generalisation for temporal diference learning: The successor representation. Neural Computation 5, 613–624 (1993)

[26] Mahadevan, S., Maggioni, M.: Proto-value functions: A Laplacian framework for learning representation and control in Markov decision processes. J. of Machine Learning Research 8, 2169–2231 (2007)

[27] Hyv¨arinen, A.., Khemakhem, I., Monti, R.P.: Identifiability of latent-variable and structural-equation models: from linear to nonlinear. Annals of the Institute of Statistical Mathematics 76, 1–33 (2024)

[28] Khemakhem, I., Monti, R.P., Kingma, D.P., Hyv¨arinen, A.: ICE-BeeM: Identifiable conditional energy-based deep models based on nonlinear ICA. In: Advances in Neural Information Processing Systems (NeurIPS2020), Virtual (2020)

[29] Lachapelle, S., Deleu, T., Mahajan, D., Mitliagkas, I., Bengio, Y., Lacoste-Julien, S., Bertrand, Q.: Synergies between disentanglement and sparsity: Generalization and identifiability in multi-task learning. In: International Conference on Machine Learning, pp. 18171–18206 (2023). PMLR

[30] Smith, S.M., Vidaurre, D., Beckmann, C.F., Glasser, M.F., Jenkinson, M., Miller, K.L., Nichols, T.E., Robinson, E.C., Salimi-Khorshidi, G., Woolrich, M.W., et al.: Functional connectomics from resting-state fmri. Trends in cognitive sciences 17(12), 666–682 (2013)

[31] Comon, P.: Independent component analysis—a new concept? Signal Processing 36, 287–314 (1994)

[32] Amari, S.-I., Cichocki, A.: Adaptive blind signal processing—neural network approaches. Proceedings of the IEEE 86(10), 2026–2048 (1998)

[33] Hyv¨arinen, A., Karhunen, J., Oja, E.: Independent Component Analysis. Wiley Interscience, New York (2001)

[34] Cichocki, A., Amari, S.-I.: Adaptive Blind Signal and Image Processing: Learning Algorithms. Wiley, New York (2002)

[35] Cichocki, A., Zdunek, R., Phan, A.-H., Amari, S.-I.: Nonnegative Matrix and Tensor Factorizations: Applications to Exploratory Multi-way Data Analysis. Wiley, Chichester, UK (2009)

[36] Khemakhem, I., Kingma, D.P., Monti, R.P., Hyv¨arinen, A.: Variational autoencoders and nonlinear ICA: A unifying framework. In: Proc. Artificial Intelligence

[37] Hyv¨arinen, A., Pajunen, P.: Nonlinear independent component analysis: Existence and uniqueness results. Neural Networks 12(3), 429–439 (1999)

[38] Zhu, Y., Parviainen, T., Heinil¨a, E., Parkkonen, L., Hyv¨arinen, A.: Unsupervised representation learning of spontaneous MEG data with nonlinear ICA. NeuroImage 274(120142) (2023)

[39] Zhou, D., Wei, X.-X.: Learning identifiable and interpretable latent models of high-dimensional neural activity using pi-VAE. Advances in Neural Information Processing Systems 33, 7234–7247 (2020)

[40] Shimizu, S., Hoyer, P.O., Hyv¨arinen, A., Kerminen, A.: A linear non-Gaussian acyclic model for causal discovery. J. of Machine Learning Research 7, 2003–2030 (2006)

[41] Hoyer, P.O., Janzing, D., Mooij, J., Peters, J., Sch¨olkopf, B.: Nonlinear causal discovery with additive noise models. In: Advances in Neural Information Processing Systems vol. 21, pp. 689–696 (2009)

[42] Peters, J., Janzing, D., Sch¨olkopf, B.: Elements of Causal Inference: Foundations and Learning Algorithms. MIT press, Cambridge, Massachusetts (2017)

[43] Seitzer, M., Sch¨olkopf, B., Martius, G.: Causal influence detection for improving eficiency in reinforcement learning. Advances in Neural Information Processing Systems 34, 22905–22918 (2021)

[44] Dillies, E., Delfosse, Q., Bl¨uml, J., Emunds, R., Busch, F.P., Kersting, K.: Better decisions through the right causal world model. arXiv preprint arXiv:2504.07257 (2025)