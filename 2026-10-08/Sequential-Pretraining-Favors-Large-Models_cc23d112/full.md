# Sequential Pretraining Favors Large Models

Mohnish Harwani Purdue University mharwan@purdue.edu

Yujia Zheng University of Illinois Urbana–Champaign yujiaz@illinois.edu

## Abstract

Large neural networks often acquire capabilities that small models fail to learn. Does this stem from large models learning more representative features, or from being more robust to unaccounted-for adverse effects introduced during training? We define and quantify one such adverse effect, primacy bias, as the extent to which exposure to early data distributions impairs later learning. We show that small models can allocate learning capacity inefficiently toward early distributions, whereas sufficiently overparameterized models are robust to this effect. This inefficiency is particularly consequential in pretraining, where foundation models often encounter heterogeneous data distributions sequentially rather than jointly. As a result, small foundation models can struggle to learn distributions encountered late in training, which is particularly harmful when later data emphasizes desirable capabilities such as code, mathematics, and reasoning. Motivated by these findings, we introduce Exposure Therapy (ET), a simple regularization that promotes more efficient allocation of learning capacity during sequential pretraining. We demonstrate that ET improves foundation models’ performance on late data distributions as well as overall capability in models up to the billion-parameter scale. Overall, our results suggest that some benefits of large foundation models may arise from greater robustness to adverse training effects, rather than from learning more representative features, and that improved training algorithms can recover some of these advantages in smaller models.

## 1 Introduction

Large machine learning models often learn behaviors that smaller models fail to acquire, motivating the continued scaling of models despite substantial training and inference costs [Wei et al., 2022, Huang et al., 2026]. However, scale may provide more than additional representational power: it can also make learning more robust such that it can tolerate a broader set of training algorithms through which data are presented, converging to good solutions more predictably [Lourie et al., 2026, Wen et al., 2026]. One such source of variation in training algorithms is data order. While existing work in curriculum learning has demonstrated how data order can alter generalization [Elman, 1993, Bengio et al., 2009, Hacohen and Weinshall, 2019, Wu et al., 2021, Jia et al., 2026, Finzi et al., 2026], it treats data order as a design choice, whereas pretraining algorithms often enforce it as a constraint.

Loss of plasticity provides an explanation for the variation in performance that data order can introduce. Several lines of work have demonstrated that neural networks gradually lose plasticity as training progresses, motivating representative data to be seen early [Dohare et al., 2024, Lyle et al., 2023]. Other results suggest that capacity may influence this loss of plasticity. Neural networks can preferentially learn simple or high-utility structure early, and gradients associated with dominant features can suppress competing signals [Arpit et al., 2017, Nakkiran et al., 2019, Refinetti et al., 2023, Pezeshki et al., 2021, Shah et al., 2020]. Overparameterization can improve tolerance to noise and reduce interference or forgetting across tasks [Nakkiran et al., 2020, Mirzadeh et al., 2022, Ramasesh et al., 2022, Mohri et al., 2026, Marek et al., 2026]. Most recently, Huang et al. [2026] demonstrated that large model capacity can provide resistance to overwriting useful learned features. While nearly all of these works study data order in terms of retaining previously learned information, they do not address the combined relationship between data order and model capacity in learning new information.

![](images/061e615b3a69453bee83af15e6ed76f6aea94d2534a0abf3251bf185ddd3f97f.jpg)  
Figure 1: Conceptual illustration of primacy bias and Exposure Therapy. Small models may become constrained by parameters recruited for early data, making later distributions harder to fit. Additional capacity can allow optimization to lead to alternative solutions, while Exposure Therapy encourages small models toward parameters that remain compatible with later learning.

In this work, we define and quantify primacy bias: the extent to which early data distributions can impair a model’s ability to fit distributions encountered later in training. We study the role of scaling in primacy bias, and demonstrate that overparameterized models are far more robust to primacy bias while small models are not. These results suggest that some benefits of large models may arise from greater robustness to biases introduced in training, as opposed to learning more representative solutions. In foundation model pretraining, the effects of primacy bias are further amplified. While training foundation models often utilizes several heterogeneous data sources [Xie et al., 2023, Feng et al., 2024, Ibrahim et al., 2024], pretraining often constrains this by organizing the data into a temporal sequence, where foundation models encounter heterogeneous data distributions sequentially rather than jointly. Models may be exposed to different subsets of the training distribution in distinct stages, where early training may emphasize broad natural-language data, and later stages increasingly incorporate more specialized domains such as code, mathematics, tool use, and agentic behavior. We find that this sequential training constraint favors large and overparameterized foundation models in terms of learning in late-stage data. Specifically, we observe that when small models lack the capacity to fit all data they encounter during training, learning new distributions increasingly requires repurposing neurons specialized to earlier data. This process of repurposing neurons is hindered by loss of plasticity and results in small foundation models fitting late-stage data less effectively.

To address these failures, we introduce Exposure Therapy (ET), which sparsely replaces training updates with a sample of downstream data distributions that models would otherwise encounter later in training. An important distinction is that Exposure Therapy does not remove this data from later training where it might otherwise be seen, and simply re-uses existing data. ET encourages models to learn structure specific to downstream data distributions during its critical learning period, reserving capacity for later acquisition. Unlike replay methods [Lopez-Paz and Ranzato, 2017, Chaudhry et al., 2019, 2021], ET focuses on improving performance related to newly encountered distributions, as opposed to preserving previously acquired capabilities, and closing capacity-dependent performance gaps in pretraining. We show that Exposure Therapy improves learning new data distributions as well as overall performance in foundation models, allowing models to recover part of the robustness otherwise obtained through additional parameters, with negligible extra computational cost. Figure 1 visualizes these effects. We demonstrate that Exposure Therapy improves pretraining in foundation models up to the billion-parameter scale.

## 2 Preliminaries

We begin with our problem setup. Our goal is to quantify the extent to which early training distributions can impair a model’s fit on later training distributions, and understand how that impairment changes across model scale. We consider two training conditions. In target-first training, a model is trained on the target distribution during Phase 1; in OOD-first training, it is instead trained on an out-of-distribution (OOD) distribution, which we use to refer to any distribution distinct from the target distribution being evaluated. After Phase 1, both conditions follow the same target-dominated Phase 2 training trajectory, without resetting model parameters, optimizer state, or the learning-rate schedule. For a model with N parameters, let $\theta _ { \mathrm { t g t } } ^ { ( \bar { N } ) } ( t )$ and $\theta _ { \mathrm { o o d } } ^ { ( N ) } ( t )$ denote its parameters after t training steps under target-first and OOD-first training, respectively. Given target loss ℓ, we define primacy sensitivity as the persistent target-performance gap

$$
\Delta _ { \mathrm { p r i m } } ( N ) : = \operatorname* { l i m } _ { t \to \infty } \left[ \ell \Bigl ( \theta _ { \mathrm { o o d } } ^ { ( N ) } ( t ) \Bigr ) - \ell \Bigl ( \theta _ { \mathrm { t g t } } ^ { ( N ) } ( t ) \Bigr ) \right] .\tag{1}
$$

A positive $\Delta _ { \mathrm { p r i m } } ( N )$ therefore indicates that exposure to the OOD distribution during early training persistently impairs the model’s fit on the target distribution. For classification experiments, ℓ denotes classification error, while for language modeling it denotes held-out cross-entropy. Empirically, we approximate the limit in Equation (1) by training the shared Phase 2 trajectory until target performance has converged.

Our controlled experiments in Section 3 illustrate how primacy sensitivity changes across scale by approximating this behavior in MLPs. Section 4 extends these results to pretraining settings, while Section 5 introduces Exposure Therapy in order to provide robustness to pretraining without increasing the number of parameters.

## 3 The Data Order-Capacity Relationship

Primacy bias is not specific to the scale of foundation models or pretraining. More generally, it arises when early exposure to one distribution changes how a model can fit another distribution. We first isolate this effect in vision tasks over MLPs, illustrating how primacy bias is a general property of neural networks, and how circumventing capacity used to fit early distributions allows large models to become more robust to this effect. These task setups are meant to make primacy bias easier to analyze mechanistically and understand why it can be scale dependent, rather than replicate the dynamics of pretraining. Section 4 then extends these basic effects to foundation models, where primacy bias can arise from sequential data order constraints in pretraining and result in poor allocation of learning capacity over pretraining data.

## 3.1 Experimental Setup: Isolating Training-History Sensitivity

We study this relationship in controlled vision experiments on MNIST, Fashion-MNIST, and KMNIST [LeCun et al., 1998, Xiao et al., 2017, Clanuwat et al., 2018], varying MLP width while holding the remaining training procedure fixed. For these experiments, the OOD distribution consists of real inputs paired with fixed random labels, creating substantial early learning with little structure transferable to the target task. During Phase 2, every fourth update is drawn from a separately sampled OOD set and the remaining updates from the target distribution, providing a weak persistent signal for the earlier structure. We train Phase 2 substantially longer than Phase 1 and evaluate models after target performance has converged. Full experimental details are provided in Appendix A.

## 3.2 Primacy bias decreases with model scale

The results of these experiments demonstrate how primacy bias falls sharply with model scale (Figure 2). Small OOD-first models perform substantially worse in terms of accuracy on later distributions compared to target-first models of the same size. As model capacity increases, this gap sharply decreases. At sufficient levels of overparameterization, final performance becomes nearly insensitive to which distribution was encountered first.

Interestingly, in these settings, small models have enough representational capacity to learn late distributions well. However, they fail to adapt to a new distribution of data while sufficiently overparameterized models can. Overparameterized models are more robust to OOD data seen in early training, and are able to learn new distributions significantly better.

![](images/0a20079d58baa85351e792aed1b3256ab4740805d5d3e92755cc94b476ae972c.jpg)

![](images/0bc853e6dd01fcb34ae39e7d2e0ea118976523ee627b8b38aaaab530cee91fe0.jpg)

![](images/cdcfedc05e85bfbbe42931d3ae756808dd32a41e3f7adb5b1a25561d9fbe9dba.jpg)  
Figure 2: Sensitivity to harmful early data decreases with model scale. Primacy sensitivity $\Delta _ { \mathrm { p r i m } }$ compares final target performance under target-first and OOD-first training. Positive values indicate that the early OOD phase impaired later target learning. Columns indicate different datasets; curves within each panel correspond to different seeds. Small models exhibit significant primacy sensitivity, while the gap attenuates sharply with increasing capacity.

Importantly, this difference is not explained by a difference in plasticity in large models. We find that models of all scales tend to exhibit low accuracy on OOD data seen early (Appendix B), suggesting the existence of a latent effect on subsequent learning in small models, even after the behavior learned during that phase is no longer expressed.

This result illustrates a benefit of overparameterization beyond representational capacity alone. A smaller model may contain enough parameters to fit the target distribution in isolation, yet still fail to reach the same solution when part of its training capacity was strongly recruited by an earlier distribution. Larger models are substantially more robust to this dependence on training history.

## 3.3 Representational overlap decreases with model scale

To understand how overparameterized models can become robust to primacy bias, we analyze hiddenunit reuse in MLPs. Specifically, at different thresholds, we rank units by post-activation magnitude on the OOD distribution at the end of Phase-1 and on the target distribution at the end of Phase-2, then measure overlap among the top 1%, 5%, and 10% most active units. This allows us to understand how models are forced to reuse existing parameters to map new distributions, and where any latent effects might lie. Full definitions and experimental details are provided in Appendix C.

Representational overlap in post-activation neurons generally decreases with primacy bias changes across model scale (Figure 3). Small models exhibit both the strongest downstream impairment and the greatest reuse of units recruited during Phase-1. Both quantities decline as scale increases. This pattern is qualitatively consistent across activation thresholds.

More broadly, these results show that the acquisition of representative features in late distributions doesn’t just depend on whether the model has sufficient capacity in principle, but also on how much of that capacity is adaptable after early learning. Overparameterized models can largely circumvent capacity that fit early training distributions, and use other capacity they have available. Small models must often repurpose existing capacity to fit a later distribution, which can be difficult in late training due to factors like loss of plasticity.

## 4 Primacy Bias in Foundation Model Pretraining

Section 3 establishes a general sensitivity of primacy bias across model scale, demonstrating how capacity-constrained models are more vulnerable to this effect. Under foundation model pretraining, we demonstrate how this becomes particularly consequential. Pretraining corpora often contain several heterogeneous domains that are trained on sequentially, rather than jointly. For example, while early learning might emphasize natural language, later learning may emphasize desirable qualities such as code, mathematics, tool-use, and reasoning. A distribution introduced late must therefore be learned after earlier data have already shaped the model during its most plastic period. We show that, from 100M to 1B parameters, substantial early training on another useful distribution can impair how capacity-constrained models later fit another distribution.

![](images/c3c89bfd2b125da838c1423885d71f5286704b2df03aa50086bf32482aa3aac2.jpg)  
Figure 3: Large models recruit a less-overlapping route for downstream learning. Solid curves with circular markers indicate primacy sensitivity $\Delta _ { \mathrm { p r i m } }$ , using the left axis; dashed curves with square markers show cross-phase representational overlap on the right axis. The reduction in sensitivity roughly aligns with the reduction in overlap between the units recruited during early OOD training and those recruited after subsequent target training. This correspondence supports the hypothesis that additional capacity allows larger models to route downstream learning through a less-overlapping set of units, rather than repurposing the units most strongly shaped by the early OOD distribution. At high activation thresholds, this signal becomes less accurate and rises in larger models.

## 4.1 Sequential-Domain Pretraining Protocol

We extend the two-phase setting of Section 3 to causal language-model pretraining at 100M, 500M, and 1B parameters. For each model scale, the first half of training consists either of English data (English-first) or one of four OOD corpora (OOD-first): code, mathematics, German, or Finnish. All models then train on the same majority English distribution for the second half of training. Each phase of training sees an equal training budget which we observe is largely sufficient for convergence. We evaluate how well models fit an English data distribution under both of these conditions where only the early data distribution varies.

Furthermore, we increase both the amount of training data and the optimization budget with model size. Thus, the 1B model is not simply a more overparameterized model trained on the same problem as the 100M model; it is trained on a correspondingly larger corpus. These experiments test whether models that remain capacity-constrained relative to their training data continue to exhibit sensitivity to the distributions encountered early in pretraining. Full experimental details are located in Appendix D.

Figure 4 shows how training on OOD data first in pretraining consistently impairs the acquisition of the English distribution encountered later. At all three parameter scales, models that spend the first half of training on another distribution exhibit substantially higher final English loss. The magnitude of this impairment varies across OOD corpora. Overall, these results demonstrate how primacy sensitivity in neural networks can introduce a practical challenge when paired with temporal pretraining constraints in training less overparameterized foundation models. Useful early data can make capacity-constrained models substantially worse at fitting distributions encountered later, which is especially harmful as later distributions often better match tasks that have real-world applications. Because removing useful early distributions is not a viable solution, an effective pretraining algorithm should instead allow models to learn them while preserving capacity for distributions that will receive greater emphasis later, motivating Exposure Therapy.

![](images/1383cfc3c55317cdaf34e3f5b1f0958fc282801120148fc25844616758c5adcd.jpg)  
Figure 4: Capacity-constrained foundation models remain sensitive to data encountered early in pretraining. The English primacy sensitivity gap reports the percentage increase in final English loss relative to the mean English-first loss at the same model scale. During the first half of training, models see either English or an OOD corpus consisting of code, mathematics, German, or Finnish; all models then train on a majority English distribution during the second half. Points denote individual runs, colors indicate the OOD corpus, and horizontal bars denote group means. Across 100M, 500M, and 1B parameter models, OOD-first pretraining produces substantially higher final English loss. Training data and optimization budget increase with model size, keeping models capacity-constrained relative to their respective training problems.

## 5 Exposure Therapy

The preceding sections demonstrate how primacy bias in neural networks can impair less overparameterized foundation models fit in late training distributions in favor of earlier ones. This phenomenon motivates a direct intervention: expose the model to a representative sample of a later training distribution while its representation is still highly plastic, allowing some capacity to become reserved for that distribution before early learning dominates the representation that a model ultimately develops.

We introduce Exposure Therapy (ET), a data-level regularization that intermittently replaces ordinary training updates with examples drawn from a distribution that would otherwise receive substantial training later. At each optimization step, the ordinary training batch is replaced by a batch from the exposure set with probability a, which we call the exposure rate. Exposure updates replace rather than supplement ordinary updates, so ET does not increase the number of optimization steps. The data used in the exposure set remain in the late data corpus to be encountered again during training, allowing ET to influence optimization without requiring additional training data.

Although ET is motivated with the goal of learning better allocations of structure across data distributions during the early critical learning period, we apply exposure throughout the training trajectory. While pretraining can make phase boundaries between different data boundaries clear, the precise duration of high-plasticity periods during which capacity allocation is most sensitive is still generally unknown. Maintaining sparse exposure avoids requiring an explicit estimate of when this period ends. Full pseudocode and analysis of these controls are provided in Appendix E.

Figure 5 demonstrates the relationship between model capacity and the amount of ET required on simple language tasks. Models with substantial capacity relative to the target distribution either recover from harmful early learning without ET or require only minimal exposure. As capacity decreases, the exposure required to recover the same fraction of performance increases, reaching 20%–30% in the most capacity-constrained settings. This inverse relationship is consistent across all three language tasks. Full experimental details are in Appendix F.

## 5.1 Exposure Therapy in Foundation Model Pretraining

We apply ET to the sequential-domain pretraining setting of Section 4.1. The exposure set consists of English data that would ordinarily be encountered during the later phase of training. We fix the exposure set to 10% of the target data training corpus and the exposure rate of 30%. Exposure updates remain active throughout both phases. We evaluate ET at the billion-parameter scale across code, mathematics, German, and Finnish OOD corpora and the FineWeb and TinyStories English target distributions. We further extend these results and evaluate this phenomenon at the 100M and 500M parameter scales in Appendix G.

![](images/443845124b24f8f0b74a40961cd6689f4cd628e44d531a8b388fa9ce5f262b1e.jpg)

![](images/aa866d4d4b35e5eb3e7c8866d8906ffcb9d1c1aefd2bd5d3623ef0c8389944b8.jpg)

![](images/ed9de9d5cd3a3f80375ec735a05867ec693414b9bb344b365a2a0bbc0af18fb8.jpg)  
Figure 5: Less overparameterized models require stronger Exposure Therapy. Each point pairs a model’s parameter count with the smallest tested exposure dose required to recover 90% of the corresponding target-first model’s performance. We evaluate Transformer classifiers on Yahoo Answers, Yelp, and Emotion. Dashed lines denote dataset-specific linear fits. Across all three datasets, the required exposure dose increases as model capacity decreases.

![](images/81893aed6191d80ccc1cc5cd5db7f1dbb6f551c955f2ff6eeb7be92a966c705a.jpg)

![](images/61dce651d44181063899c3eaa0b6fdfa67bdd1a78d16aa7751d796898b37de05.jpg)  
Figure 6: Exposure Therapy improves both later-distribution and aggregate performance in 1B parameter models. Top: held-out English loss after sequential pretraining on FineWeb and TinyStories; lower is better. Bottom: aggregate loss, computed across the Phase-1 OOD corpus and the English target distribution; lower is better. Within each OOD-corpus group, bars show English-first, OOD-first, and OOD-first training with a 30% Exposure Therapy rate. Black points denote individual runs. OOD-first training consistently increases subsequent English loss, while ET reduces this impairment across every tested OOD and target distribution. At the same time, OOD-first training generally benefits aggregate performance by learning both useful distributions, and ET further reduces aggregate loss across all tested conditions. ET therefore improves acquisition of the later distribution without sacrificing the useful structure learned earlier.

Figure 6 evaluates both subsequent English learning and aggregate performance at the 1B parameter scale. Across every OOD corpus and both English target distributions, OOD-first training produces higher English loss than the corresponding English-first reference, while ET consistently reduces this impairment. Importantly, these gains do not come from simply shifting capacity away from the initial OOD distribution. Exposure Therapy also results in better overall test performance over all training distributions. Rather than trading performance between the two stages, ET generally produces a more efficient allocation of learning capacity that improves fit over late data as well as overall test performance in pretraining. These benefits come at negligible extra computational cost and require no additional data.

## 6 Limitations

Our mechanistic evidence demonstrates that parameters that fit early distributions are often avoided in sufficiently overparameterized models, providing a possible explanation for why large models are more robust to primacy bias. However, these results are limited in scope to MLPs where feature allocation is better understood. Furthermore, understanding the precise effect as to why these parameters are avoided or difficult to repurpose remains future work. Our pretraining experiments also use only two data distributions, whereas realistic pretraining may involve many sequential stages over many more distributions. Due to computational cost, we evaluate Exposure Therapy with fixed hyperparameters in foundation models and do not rigorously identify how to tune hyperparameters in pretraining. Instead, we use toy language models to study how suitable exposure strength varies with scale, and while exact hyperparameters might not transfer, we predict that their qualitative dependence on model scale will persist. Our pretraining setup further assumes that Phase 2 retains a small fraction of OOD data or otherwise some weak training signal to retain earlier structure, and that sampling is shuffled within rather than across distributions, and that useful OOD data precedes English, whereas frontier pretraining often uses the opposite ordering. We perform ablation experiments without these assumptions to more closely match practical pretraining, demonstrating that ET improves pretraining in these settings as well, although in some cases to a reduced degree. While this still might not extend to all forms of pretraining, it demonstrates the broader applicability of this work under other pretraining pipelines.

## 7 Conclusion

Large models often acquire capabilities that smaller models fail to learn. While this advantage is often assumed to stem from large models representing more structure within the training data, our results identify a complementary benefit: scale can also make learning more robust to adverse effects during training. We define and examine one such effect, primacy bias, as the persistent impairment in fitting late data due to exposure to early data distributions. Under mechanistic analysis of MLPs, we find that overparameterized models are able to circumvent parameters heavily recruited to fit early data distributions, which may be difficult to repurpose due to loss of plasticity.

Primacy bias becomes particularly consequential under sequential data order constraints in pretraining. In these settings, foundation models often encounter heterogeneous data distributions sequentially rather than jointly, which can favor fitting early distributions over late ones. When late data represents desirable qualities such as skills in coding, tool-use, and reasoning, this can be especially harmful. In sequential pretraining up to the billion-parameter scale, we find that primacy bias reliably impairs models over several types of data distributions. To help allocate learning capacity more efficiently across all data distributions, we introduce Exposure Therapy, a regularization that introduces a subset of late data early in pretraining where models may exhibit heightened plasticity. While adding negligible extra computational cost, this provides a cheaper alternative to scaling models, and demonstrates how some benefits of model scale can be recovered algorithmically.

## Reproducibility Statement

We provide the code to toy vision experiments along with their mechanistic analysis, foundation model experiments, and Exposure Therapy, and additionally outlining hyperparameters and most experimental details for pretraining experiments within the Appendix. All datasets used are publicly available, and we include scripts to retrieve them.

## References

Devansh Arpit, Stanisław Jastrz˛ebski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, and Simon Lacoste-Julien. A closer look at memorization in deep networks. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, 2017. URL https://proceedings.mlr.press/v70/arpit17a.html.

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th International Conference on Machine Learning, 2009. doi: 10.1145/ 1553374.1553380.

Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K. Dokania, Philip H. S. Torr, and Marc’Aurelio Ranzato. On tiny episodic memories in continual learning. arXiv preprint arXiv:1902.10486, 2019. doi: 10.48550/arXiv.1902.10486.

Arslan Chaudhry, Albert Gordo, Puneet Dokania, Philip H. S. Torr, and David Lopez-Paz. Using hindsight to anchor past knowledge in continual learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 35, 2021. doi: 10.1609/aaai.v35i8.16861.

Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, and David Ha. Deep learning for classical Japanese literature. arXiv preprint arXiv:1812.01718, 2018. doi: 10.48550/arXiv.1812.01718.

Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, and Richard S. Sutton. Loss of plasticity in deep continual learning. Nature, 632, 2024. doi: 10.1038/s41586-024-07711-7.

Jeffrey L. Elman. Learning and development in neural networks: The importance of starting small. Cognition, 48(1), 1993. doi: 10.1016/0010-0277(93)90058-4.

Steven Feng, Shrimai Prabhumoye, Kezhi Kong, Dan Su, Mostofa Patwary, Mohammad Shoeybi, and Bryan Catanzaro. Maximize your data’s potential: Enhancing LLM accuracy with two-phase pretraining. arXiv preprint arXiv:2412.15285, 2024. doi: 10.48550/arXiv.2412.15285.

Marc Finzi, Shikai Qiu, Yiding Jiang, Pavel Izmailov, J. Zico Kolter, and Andrew Gordon Wilson. From entropy to epiplexity: Rethinking information for computationally bounded intelligence. arXiv preprint arXiv:2601.03220, 2026. doi: 10.48550/arXiv.2601.03220.

Guy Hacohen and Daphna Weinshall. On the power of curriculum learning in training deep networks. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, 2019. URL https://proceedings.mlr. press/v97/hacohen19a.html.

Jing Huang, Daniel Wurgaft, Rachit Bansal, Laura Ruis, Naomi Saphra, David Alvarez-Melis, Andrew Kyle Lampinen, Christopher Potts, and Ekdeep Singh Lubana. Why larger models learn more: Effects of capacity, interference, and rare-task retention. arXiv preprint arXiv:2605.29548, 2026. doi: 10.48550/arXiv.2605.29548.

Adam Ibrahim, Benjamin Thérien, Kshitij Gupta, Mats L. Richter, Quentin Anthony, Timothée Lesort, Eugene Belilovsky, and Irina Rish. Simple and scalable strategies to continually pretrain large language models. Transactions on Machine Learning Research, 2024. URL https: //openreview.net/forum?id=DimPeeCxKO.

Yaning Jia, Chunhui Zhang, Xingjian Diao, Xiangchi Yuan, Zhongyu Ouyang, Chiyu Ma, and Soroush Vosoughi. What makes a good curriculum? Disentangling the effects of data ordering on LLM mathematical reasoning. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 2026. doi: 10.18653/v1/2026.acl-long.1591.

Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings ofthe IEEE, 86(11), 1998. doi: 10.1109/5.726791.

David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ f87522788a2be2d171666752f97ddebb-Abstract.html.

Nicholas Lourie, Kyunghyun Cho, Karen Ullrich, and Sanae Lotfi. Small-scale experiments: Are we there yet? arXiv preprint arXiv:2608.11859, 2026. doi: 10.48550/arXiv.2608.11859.

Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, and Will Dabney. Understanding plasticity in neural networks. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, 2023. URL https://proceedings.mlr.press/v202/lyle23b.html.

Martin Marek, Dongkyu Cho, Shikai Qiu, Rumi Chunara, Pavel Izmailov, and Andrew Gordon Wilson. Forgetting in language models: Capacity, optimization, and self-generated replay. arXiv preprint arXiv:2605.26097, 2026. doi: 10.48550/arXiv.2605.26097.

Seyed Iman Mirzadeh, Arslan Chaudhry, Dong Yin, Huiyi Hu, Razvan Pascanu, Dilan Gorur, and Mehrdad Farajtabar. Wide neural networks forget less catastrophically. In Proceedings ofthe 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, 2022. URL https://proceedings.mlr.press/v162/mirzadeh22a. html.

Christopher Mohri, John Duchi, and Tatsunori Hashimoto. A bitter lesson for data filtering. arXiv preprint arXiv:2605.19407, 2026. doi: 10.48550/arXiv.2605.19407.

Preetum Nakkiran, Gal Kaplun, Dimitris Kalimeris, Tristan Yang, Benjamin L. Edelman, Fred Zhang, and Boaz Barak. SGD on neural networks learns functions of increasing complexity. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://proceedings.neurips.cc/paper/2019/hash/ b432f34c5a997c8e7c806a895ecc5e25-Abstract.html.

Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, and Ilya Sutskever. Deep double descent: Where bigger models and more data hurt. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=B1g5sA4twr.

Mohammad Pezeshki, Sékou-Oumar Kaba, Yoshua Bengio, Aaron Courville, Doina Precup, and Guillaume Lajoie. Gradient starvation: A learning proclivity in neural networks. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 0987b8b338d6c90bbedd8631bc499221-Abstract.html.

Vinay V. Ramasesh, Aitor Lewkowycz, and Ethan Dyer. Effect of model and pretraining scale on catastrophic forgetting in neural networks. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=GhVS8\_yPeEa.

Maria Refinetti, Alessandro Ingrosso, and Sebastian Goldt. Neural networks trained with SGD learn distributions of increasing complexity. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, 2023. URL https://proceedings.mlr.press/v202/refinetti23a.html.

Harshay Shah, Kaustav Tamuly, Aditi Raghunathan, Prateek Jain, and Praneeth Netrapalli. The pitfalls of simplicity bias in neural networks. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://proceedings.neurips.cc/paper/2020/ hash/6cfe0e6127fa25df2a0ef2ae1067d915-Abstract.html.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022. URL https://openreview.net/forum? id=yzkSU5zdwD.

Kaiyue Wen, David Hall, Tengyu Ma, and Percy Liang. Fantastic pretraining optimizers and where to find them. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=2J51qUZ0iG.

Xiaoxia Wu, Ethan Dyer, and Behnam Neyshabur. When do curricula work? In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id=tW4QEInpni.

Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017. doi: 10.48550/arXiv.1708.07747.

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, and Adams Wei Yu. DoReMi: Optimizing data mixtures speeds up language model pretraining. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html.

## A Primacy Sensitivity in Toy Experiments

We provide additional implementation details for the toy experiments in Section 3. All experiments compare paired target-first and OOD-first runs at the same model scale and random seed. The two runs use the same architecture, initialization distribution, optimizer, and total number of updates, and differ only in the objective optimized during Phase 1. No parameters, optimizer state, or learning-rate state are reset at the phase boundary.

Shared two-phase protocol. For the toy experiments, the OOD objective is implemented using real inputs from the corresponding training distribution paired with fixed random targets. Thus, the corruption is in the input–target mapping rather than in the input marginal.

For each run, we construct a fixed Phase-1 corruption set A. In the OOD-first condition, all Phase-1 updates are sampled from A; in the target-first condition, all Phase-1 updates are sampled from the target task. During Phase 2, both conditions use the same target-dominated schedule: every fourth optimization step is an OOD update and the remaining three are target updates. Phase-2 OOD updates use a separately sampled fixed corruption set B with newly generated random targets. Set A itself is therefore never re-presented during Phase 2. Training examples are sampled with replacement within each source.

Vision tasks. We evaluate two-hidden-layer MLPs on MNIST, Fashion-MNIST, and KMNIST. Vision models are trained for 30,000 Phase-1 updates followed by 120,000 Phase-2 updates, which we find is appropriate for convergence. We use separate output heads for the target and OOD objectives while sharing the hidden representation, so that any cross-phase interference reflects structure learned in the shared features rather than incidental alignment or misalignment between target class identities.

Optimization and evaluation. All toy models are optimized with Adam and cross-entropy loss using a constant learning rate.

## B OOD-Set Accuracy Across the Full Parameter Ladder

We demonstrate that, while training on OOD data early on can impair the ability to fit different training distributions afterward, this doesn’t necessarily mean that models exhibit high accuracy on early OOD data. Figure 7 demonstrates this trend.

![](images/4bbe6da73689bf8037adc49836743c185a3d9f1f2324c464f2f14ee23afed08f.jpg)

Figure 7: The early OOD mapping is no longer behaviorally expressed after target training across the full parameter ladder. Each panel shows one dataset, with model parameter count on the horizontal axis and accuracy on the original Phase-1 OOD-set on the vertical axis. Pale bars report accuracy at the end of Phase-1, when the OOD-first models have been trained on the fixed random mapping; dark bars report accuracy on the same examples after completing the target-dominated Phase-2. Small models fit the OOD data to high accuracy during Phase-1, and drop much lower after Phase-2. Models demonstrate the primacy bias effect without necessarily expressing OOD data seen initially.

## C Overlap Mechanism

To understand and quantify how early learning can limit the capacity of models on downstream distributions, we compare overlap in usable neurons before and after Phase-2, where they are exposed to target distributions.

We record activation snapshots at the end of each phase using probe data from the OOD and target distributions. For a hidden unit u, we define its activation score at phase t as its mean absolute post-activation response:

$$
s _ { t } ( u ) = \frac { 1 } { | \mathcal { P } _ { t } | } \sum _ { x \in \mathcal { P } _ { t } } \left| a _ { t } ( x ; u ) \right| ,\tag{2}
$$

where $\mathcal { P } _ { t }$ is the corresponding probe set and $a _ { t } ( x ; u )$ is the activation of unit u. These units are post-ReLU hidden activations in MLPs.

For each phase, we select the top fraction f of units according to Equation (2). Let $A _ { \mathrm { c o r } } ^ { ( f ) }$ denote the units most strongly recruited at the end of the OOD phase and $A _ { \mathrm { t g t } } ^ { ( f ) }$ the units most strongly recruited after target training. We measure cross-phase representational overlap using the Jaccard index:

$$
\mathrm { R } ^ { ( f ) } = \frac { \left| A _ { \mathrm { c o r } } ^ { ( f ) } \cap A _ { \mathrm { t g t } } ^ { ( f ) } \right| } { \left| A _ { \mathrm { c o r } } ^ { ( f ) } \cup A _ { \mathrm { t g t } } ^ { ( f ) } \right| } .\tag{3}
$$

In MLPs, the top-f selection and Jaccard overlap in Equation (3) are computed independently within each hidden layer. We report $\operatorname { R } ^ { ( f ) }$ as the mean of these layer-wise Jaccard overlaps across hidden layers. Figure 3 plots how this overlap changes across scales.

## D Pretraining Experimental Details

We provide the full experimental details for the sequential-domain pretraining experiments in Section 4. These experiments use byte-level causal Transformers trained at approximately 100M, 500M, and 1B parameters. For each model scale, we evaluate English-first and OOD-first training across two English corpora and four OOD corpora.

Model architecture. All models operate directly on bytes with a vocabulary size of 256 and a sequence length of 512. We use 7-layer causal Transformers, pre-LayerNorm blocks, and a final LayerNorm before the output head. The feed-forward activation is GeLU, and there is no dropout. Model width is scaled to obtain the three parameter counts used in our experiments:

<table><tr><td>Model scale</td><td>Hidden width</td><td>Parameters</td></tr><tr><td>100M</td><td>1088</td><td>100,650,048</td></tr><tr><td>500M</td><td>2432</td><td>499,545,216</td></tr><tr><td>1B</td><td>3456</td><td>1,007,151,232</td></tr></table>

Training schedule. Training is divided into two equal-length phases. The total number of optimization steps increases with model scale:
<table><tr><td>Model scale</td><td>Phase-1 steps</td><td>Phase-2 steps</td><td>Total steps</td></tr><tr><td>100M</td><td>9,000</td><td>9,000</td><td>18,000</td></tr><tr><td>500M</td><td>27,000</td><td>27,000</td><td>54,000</td></tr><tr><td>1B</td><td>43,000</td><td>43,000</td><td>86,000</td></tr></table>

For each English–OOD corpus pair, the training conditions differ in the data used during Phase 1. English-first models train on the corresponding English corpus during Phase 1, whereas OOD-first models train on the corresponding OOD corpus. Phase 2 is identical across the two conditions and consists of a fixed 75%/25% mixture of English and OOD updates. We implement this mixture deterministically, with every fourth optimization step drawn from the OOD corpus and the remaining three steps drawn from English. Thus, the distinction between English-first and OOD-first training is restricted to the first half of training.

The phase boundary does not trigger any training reset. Model parameters, optimizer state, and the learning-rate schedule continue uninterrupted from Phase 1 into Phase 2.

Optimization. We optimize all models with Adam using $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$ , with no weight decay. The learning rate is $1 . 5 \times 1 0 ^ { - 4 }$ at all model scales. We linearly warm up the learning rate for the first 1,000 optimization steps and then hold it constant for the remainder of training. Gradients are clipped to a maximum norm of 0.5.

All experiments use a batch size of 32 sequences, with 512 bytes per sequence, for a total of 16,384 byte tokens per optimization step.

Pretraining corpora. We use CodeSearchNet for code, OpenWebMath for mathematics, and Wikipedia for German and Finnish. The English target distributions are FineWeb and TinyStories. Each OOD corpus is paired separately with each English corpus, yielding the code, mathematics, German, and Finnish conditions reported in Section 4.

Evaluation. Final evaluation uses 100k bytes of random byte-sequences of withheld data, each 512 bytes in length, at a 50/50 split between each distribution seen. We report English loss using only the held-out English data. We additionally report aggregate loss over the combined held-out English and OOD evaluation sets, thereby measuring performance across both distributions encountered during training.

## E Exposure Therapy Details

Algorithm 1: Training with Exposure Therapy   
Input : model $f _ { \boldsymbol { \theta } } ;$ ordinary training dataset $\{ \mathcal { D } _ { t } \} _ { t = 1 } ^ { T } ;$ target-data pool of late training data $\mathcal { D } _ { \mathrm { g } } ;$   
exposure rate $a \in [ 0 , 1 ] ;$ exposure set size $s ;$ number of training steps $T ;$ batch size $B ;$   
optimizer $\mathrm { O p t }$   
Output :parameters $\theta _ { T }$   
Fix an exposure set   
$\mathcal { A } \subset \mathcal { D } _ { \mathrm { g } } , \qquad | \mathcal { A } | = s ,$   
which remains part of the ordinary training pool: its examples are still drawn by ordinary   
updates, so Exposure Therapy adds no data and withholds none   
for $t \gets 1$ to $T$ do   
$b _ { t } \sim$ Bernoulli(a)   
if $b _ { t } = 1$ then   
draw B from $\{ 1 , \ldots , s \}$ and let $Z _ { t }$ be the corresponding examples of A   
$/ /$ exposure update   
else   
$Z _ { t } \sim$ Batch $( \mathcal { D } _ { t } , B )$ $/ /$ ordinary update   
$g _ { t } \gets \nabla _ { \boldsymbol { \theta } } \left[ \frac { 1 } { | Z _ { t } | } \sum _ { z \in Z _ { t } } \ell ( f _ { \boldsymbol { \theta } } ; z ) \right]$   
$\theta _ { t } \gets \mathrm { O p t } \bar { ( } \dot { \theta } _ { t - 1 } , g _ { t } )$

Algorithm 1 illustrates a simple training loop with Exposure Therapy. Here, $\mathcal { D } _ { t }$ denotes the dataset from which the unmodified training procedure would draw its batch at optimization step t. The ordinary training distribution can change over the training trajectory, as in sequential pretraining. Exposure Therapy fixes an exposure set ${ \mathcal { A } } \subset { \mathcal { D } } _ { \mathrm { g } }$ containing s distinct examples from late training data. The exposure set is not held out: it remains part of the ordinary training pool and its examples are still drawn by ordinary updates, so Exposure Therapy introduces no additional data. At each step, $b _ { t } \sim$ Bernoulli(a) determines whether the ordinary update is replaced by an exposure update. $\begin{array} { r } { \operatorname { I f } b _ { t } = 0 . } \end{array}$ , the model receives a batch from the scheduled ordinary dataset $\mathcal { D } _ { t } \mathbf { . \nabla } \operatorname { I f } b _ { t } \dot { = } 1$ , the model instead receives a batch of $B$ examples drawn from ${ \mathcal { A } } .$ Exposure updates therefore sample the exposure set independently at every step rather than traversing it in a fixed order, and because they replace ordinary updates rather than supplement them, the total number of optimization steps is unchanged.

There are two hyperparameters introduced in ET: exposure rate and exposure set size. Conceptually, exposure set size can be seen as a heuristic to estimate the capacity that should remain availablefor late training, while exposure rate reinforces how strongly training is constrained to preserve that capacity. A larger exposure set size may help models find a more accurate estimate of how capacity should be reserved for this distribution later on. We find that smaller models that are more constrained in terms of learning capacity relative to their datasets require both a more accurate estimate of the target distribution and more frequent reinforcement of those updates, to oppose pressure that may stem from small model scale.

## F Exposure Therapy and Model Scale Experimental Details

Section 5 demonstrates how exposure dosage decreases as models become more overparameterized in toy language models. We list those experimental details here.

For a model with N parameters, let $\operatorname { A c c } _ { \mathrm { t g t } } ( N )$ denote the final target accuracy under target-first training, and $\mathrm { A c c } _ { \mathrm { E T } } ( \bar { a } , c ; N )$ the accuracy obtained with Exposure Therapy. We define the relative target-first performance as

$$
M _ { \mathrm { E T } } ( a , c ; N ) = \frac { \mathrm { A c c } _ { \mathrm { E T } } ( a , c ; N ) } { \mathrm { A c c } _ { \mathrm { t g t } } ( N ) } .\tag{4}
$$

Thus, $M _ { \mathrm { E T } } = 0 . 7 5$ means that the ET model reaches 75% of the target-first model’s accuracy, $M _ { \mathrm { E T } } = 1$ means that it matches the target-first model, and $M _ { \mathrm { E T } } > 1$ means that it outperforms the target-first reference.

We evaluate the minimal ET dose required to reach 90% of the target-first model’s performance.

$$
\begin{array} { r } { { a } _ { 0 . 9 } ^ { \star } ( c ; N ) = \operatorname* { m i n } \left\{ a \in \mathcal { G } _ { a } : \mathrm { A c c } _ { \mathrm { E T } } ( a , c ; N ) \geq 0 . 9 \mathrm { ~ A c c } _ { \mathrm { t g t } } ( N ) \right\} , } \end{array}\tag{5}
$$

where ${ \mathcal { G } } _ { a }$ denotes the tested grid of exposure dosage. Equivalently, $a _ { 0 . 9 } ^ { \star } ( c ; N )$ is the smallest tested dosage satisfying $M _ { \mathrm { E T } } ( a , c ; \mathbf { \bar { \it N } } ) \geq 0 . 9$

Figure 5 plots how this minimum required dose changes across model scales.

## G Exposure Therapy Pretraining at 100M and 500M Scales

We extend the Exposure Therapy pretraining experiments of Section 5.1 to the 100M and 500M parameter scales, and include the 1B results for comparison. We use the same sequential-domain pretraining setup across scales: models are trained either English-first, OOD-first, or OOD-first with a 30% Exposure Therapy rate and 10% exposure set size, with code, mathematics, German, and Finnish as the OOD distributions. We evaluate these on both English-specific test loss and aggregate test loss on all data, using the same metrics as Figure 6. Figure 8 demonstrates that this effect is consistent across all model scales.

## H Ablation Experiments Under Varying Pretraining Conditions

Our pretraining experiments in Section 4 and Section 5 assume several aspects of pretraining that may not extend to all pretraining setups. To indicate the broader applicability of Exposure Therapy and this work, we discuss those design choices in this section and perform ablation experiments without these assumptions.

## H.1 Reverse Order Pretraining

Our pretraining experiments measure the impairment to English learning by seeing other distributions such as code, math, or other languages early in training. While this demonstrates the central claim of this work (early OOD data can impair later learning), most pretraining is done in the reverse order. Broad English natural language data is typically seen first, with distributions like code, math, and other languages seen second. We perform ablation experiments across the 100M, 500M, and 1B parameter models to demonstrate that this effect is persistent, and that Exposure Therapy improves both late learning on specialized distributions and overall loss across all distributions in these settings. Figure 9 demonstrates this effect.

![](images/46e5a51461e35c227d75ebf80f2f1c4e439f1a519db6a7494371ddd088d4d2b9.jpg)  
Figure 8: Exposure Therapy improves sequential pretraining across model scales. Columns show 100M, 500M, and 1B parameter models. The first two rows report, respectively, held-out English loss and aggregate loss when FineWeb is the English target distribution; the bottom two rows report the corresponding quantities for TinyStories. Within each OOD-corpus group, gray bars denote English-first training, solid colored bars denote OOD-first training, and dotted colored bars denote OOD-first training with a 30% Exposure Therapy rate and 10% exposure set size. Black points denote individual runs. Lower values are better. Across all three model scales and both English target distributions, OOD-first training increases subsequent English loss relative to the English-first reference, while Exposure Therapy consistently reduces this impairment. Exposure Therapy also reduces aggregate loss relative to unregularized OOD-first training across the tested conditions.

## H.2 Persistent Effects to Retain Early Structure

Frontier foundation models can extend beyond their training distribution in ways that other models cannot. Typically, this is associated with being able to reconcile task and data-specific qualities during training, converging to a more optimal solution across all training data rather than one solution or the other. For this reason, most pretraining pipelines train foundation models (and LLMs) over a broad range of natural language data before they specialize into more specific distributions of data, because general semantic and grammatical rules are maintained and even central to learning later distributions and behavior, like reasoning.

![](images/f3fc4bef8f5b3d50455fa63a64c9ca12451b0dca7b3a482fb617d1c7f1a3c807.jpg)  
Figure 9: Exposure Therapy remains effective under reverse-order pretraining. Columns show 100M, 500M, and 1B parameter models. The top row reports held-out code/math/German/Finnish target loss, while the bottom row reports aggregate loss over FineWeb and the paired code, mathematics, German, or Finnish distribution; lower is better. Within each distribution, gray bars denote target-first training, solid colored bars denote English-first training, and dotted bars denote English-first training with Exposure Therapy. Black points denote individual runs. Across model scales, ET generally improves late-distribution learning relative to unregularized English-first training and consistently improves aggregate performance, demonstrating that its benefits persist when broad English data is encountered before more specialized distributions.

Because our experiments don’t train foundation models within a comparable margin to frontier models (or even to a conversational level), training naively over two distributions (even at the billion-parameter scale) can be insufficient to retain early learned structures when trained on new distributions, due to a poor capability to reconcile data-specific differences across distributions in small experiments. To approximate this effect, we mix a minority of OOD data into late learning in our experiments through a deterministic mixture, with every fourth optimization step drawn from the OOD corpus and the remaining three steps drawn from the target distribution. This provides a weak training signal to maintain some degree of earlier structure, and is sufficient to cause models to fail to learn late data even when they have sufficient capacity and it is in the majority.

We relax this assumption through two ablation experiments in reverse order pretraining, which can better match practical methods. First, we train using random shuffling over the late learning (Phase-2) data between the target and OOD data, rather than training it deterministically, which can more closely mimic how signals to retain earlier structure are more inconsistent in pretraining. Figure 10 (left) demonstrates that these results are consistent with the deterministic mixture.

Next, we examine how primacy bias changes with no mixture of OOD data in late learning at all. Figure 10 (right) demonstrates these results. Without the presence of OOD data in late learning, dis tributions like code and math that are similar to English often see negligible increases in performance with Exposure Therapy. On distributions that have far less reusable structure (other languages, like German and Finnish), Exposure Therapy still improves late learning. However, aggregate loss over all distributions improves more consistently over all data types. Without the presence of OOD data in late learning, some models can often catastrophically forget the first distribution to better fit later distributions. Interestingly, Exposure Therapy’s encouragement to reconcile data-specific differences seems to improve how previously learned structure is retained, reducing catastrophic forgetting in these settings, in addition to improving late learning. Because this work focuses on improving late learning specifically, we leave further analysis of how ET can reduce catastrophic forgetting to future

![](images/d178999070a2399ccd32b6612b18151af76131db0d63c4694ce7e85cb0c82395.jpg)  
(a) Shuffled Phase-2 mixture

![](images/be0d0780219592b5f85c080351d1c86d551055272a729e22963b784d8036ef56.jpg)  
(b) No Phase-2 mixture

Figure 10: Exposure Therapy under alternative Phase-2 training conditions. Results are shown for 100M-parameter models across code, mathematics, German, and Finnish target distributions. Gray bars denote target-first training, solid colored bars denote English-first training, and dotted bars denote English-first training with Exposure Therapy. Black points show individual runs. Top panels report held-out target loss and bottom panels report aggregate loss over the target and English distributions; lower is better. (a) Phase 2 uses the same target-dominated mixture as the main experiments, but English and target updates are randomly shuffled rather than deterministically interleaved. Exposure Therapy reduces both target and aggregate loss across the tested distributions. (b) Phase 2 contains only the target distribution, removing continued English updates. Exposure Therapy produces little change in target loss for code and mathematics, but improves target loss for German and Finnish and reduces aggregate loss across all four distributions.

work, and include these results solely to demonstrate broader robustness of Exposure Therapy under different pretraining conditions.