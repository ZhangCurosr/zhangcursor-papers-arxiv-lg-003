# SafeLLM4SE: Statistical Evaluation and Reporting for LLM-based Software Engineering Systems

FRANCISCO ORTIN, University of Oviedo, Spain and Munster Technological University, Ireland

Large language models (LLMs) are increasingly used for software engineering tasks, yet their stochastic behavior challenges the validity, reproducibility, and comparability of their evaluations. Conventional practices such as reporting a single output, an average score, best-of-�, or pass@� performance can obscure variability and estimation uncertainty, potentially leading to misleading conclusions about system reliability. This article presents SafeLLM4SE, a practical methodology and reporting standard for statistically principled evaluation of LLM-based software engineering systems. Rather than treating generated outputs as deterministic artifacts, SafeLLM4SE treats them as realizations of a stochastic process and distinguishes quality, stability, and estimation uncertainty. It combines adaptive sampling with confidence intervals, distribution-aware statistical comparisons, efect sizes, and a minimum reporting standard covering model configuration, reproducibility, evaluation procedures, and resource usage. SafeLLM4SE is also provided as an open-source software package available on PyPI, enabling researchers and practitioners to reproduce and extend the methodology. We illustrate its application by comparing two LLMs on HumanEval, a benchmark of programming problems assessed through functional tests.

CCS Concepts: • Software and its engineering → Empirical software validation; • Mathematics of computing → Hypothesis testing and confidence interval computation; • Computing methodologies → Machine learning; • General and reference → Experimentation.

Additional Key Words and Phrases: Large language models, software engineering, stochastic evaluation, adaptive sampling, statistical inference, reproducibility

## 1 Introduction

Large language models (LLMs) are rapidly becoming part of the software engineering (SE) toolchain. They are now used across a broad range of SE activities, including code generation, test generation, program repair, refactoring, documentation, code review, and software maintenance [12]. LLMs increasingly support developers in automating a diverse set of SE tasks and workflows, including tasks of growing complexity [5].

However, LLM-based systems difer from many traditional SE tools in an important respect: their outputs are stochastic. Given the same prompt and nominal model configuration, repeated invocations may produce diferent outputs and, consequently, diferent SE outcomes. For example, an LLM may generate a program that passes all tests in one invocation and an implementation that fails the tests in another invocation for exactly the same task. Thus, the outcome of an evaluation is not necessarily a fixed property of the system, but rather an observation of an underlying stochastic process.

Ouyang et al. [18] investigated non-determinism in ChatGPT code generation across 829 problems. They found substantial variation across repeated generations and, importantly, showed that setting the temperature to zero does not guarantee deterministic code generation. Therefore, LLM non-determinism can afect both software correctness and scientific conclusions, making it a methodological concern rather than merely an implementation detail [18]. This problem is especially acute in agentic SE systems, whose multi-step trajectories compound sources of randomness, yet are commonly evaluated using a single run per task, thereby obscuring execution-to-execution variability [16].

The stochastic nature of LLMs also creates challenges for reproducibility. The same prompts may not produce the same outputs even under the same nominal LLM configuration [18], which complicates reproducibility in contexts such as scientific research and safety-critical applications. Moreover, LLM services may change over time because of model updates or modifications to their serving infrastructure, even when the same model identifier and nominal configuration are used [4, 13]. Reproducible evaluation therefore requires reporting not only the experimental setup, but also how stochastic outputs were sampled and evaluated [19].

Despite this non-deterministic behavior, many researchers and practitioners continue to characterize LLM performance using a single output, the average performance across multiple outputs, best-of-� (i.e., the best output among � executions), or pass@� (i.e., an estimate of the probability that at least one of � generated outputs is correct). These metrics capture diferent aspects of model performance, but they may provide an incomplete picture of the underlying stochastic behavior.

To illustrate the practical consequences, Table 1 presents hypothetical HumanEval results in which diferent models are evaluated 100 times on the same set of code-generation problems, with functional correctness assessed through testing. For each evaluation, the success rate is computed as the proportion of problems for which the generated solution passes all functional tests.

A single-output evaluation identifies A as the best-performing model, whereas its average performance is lower than that of the other two models. Table 1 also illustrates how average performance estimates may depend on the number of samples. For example, C achieves an average success rate of 0.36 at � = 10 and 0.54 at � = 100. Likewise, B and C show comparable average performance for $N = 1 0 0$ , but the variability of C (SD = 0.344) is substantially greater than that of B (SD = 0.175). Consequently, average performance alone does not fully characterize the behavior of either model. In addition, although C’s average success rate is higher than B’s, its 95% confidence interval (CI) is substantially wider. This indicates that the estimated performance of C is less precise, making comparisons and decisions based on the point estimate alone potentially less reliable.

Best-of-� reports the best observed result within a sampling budget, whereas pass@� estimates the probability of obtaining at least one correct result among � samples. Neither alone characterizes the distribution of performance across complete executions. In the example, model C achieves noticeably higher best-of-10 and pass@5 results than the other models, even though its average behavior does not show a comparable advantage. Therefore, these metrics should not be used as standalone measures when the goal is to characterize the overall stochastic behavior of LLM-based SE systems.

These limitations motivate a shift from point-based to distribution-based evaluation of LLMs for SE. The output of an LLM should be regarded as a sample from a stochastic process rather than as a deterministic artifact. Accordingly, the object of evaluation is not a single output, but the underlying distribution of possible outputs and their associated performance. This perspective makes it possible to distinguish three complementary properties of an LLM-based SE system: quality (expected performance of the system), stability (how much the system’s performance varies across repeated executions), and estimation uncertainty (how precisely those quantities have been estimated from a finite number of observations).

To address these issues, we propose SafeLLM4SE, a practical methodology for the statistically principled evaluation and reporting of stochastic LLM-based SE systems. SafeLLM4SE is not intended to introduce a new statistical test. Rather, it defines a minimum standard for evaluation and reporting by integrating established statistical techniques into a workflow designed for SE researchers and practitioners. The framework provides default recommendations supported by an open-source software application, while allowing justified alternatives when the characteristics of a particular experiment warrant them.

Table 1. Hypothetical HumanEval results illustrating how sample size, variability, and uncertainty afect assessments of stochastic LLM-based systems.
<table><tr><td>Model</td><td>Single Output</td><td>Average (N=10)</td><td>Average (N=30)</td><td>Average (N=100)</td><td>SD (N=100)</td><td>95% CI (N=100)</td><td>Pass@5</td><td>Best- of-10</td></tr><tr><td>A</td><td>0.658</td><td>0.277</td><td>0.268</td><td>0.328</td><td>0.152</td><td>[0.298,0.358]</td><td>0.864</td><td>0.931</td></tr><tr><td>B</td><td>0.424</td><td>0.509</td><td>0.542</td><td>0.529</td><td>0.175</td><td>[0.495, 0.563]</td><td>0.903</td><td>0.962</td></tr><tr><td>C</td><td>0.425</td><td>0.360</td><td>0.438</td><td>0.540</td><td>0.344</td><td>[0.473,0.607]</td><td>0.948</td><td>0.981</td></tr></table>

SafeLLM4SE is organized around five principles:

(1) Outputs are samples, not artifacts. For a fixed prompt, model, and configuration, each invocation is treated as an observation from a stochastic process. Evaluation therefore begins by identifying the performance metric induced by the generated artifact (e.g., pass/fail, BLEU score, or code quality).

(2) Distributions matter more than points. Repeated executions are used to characterize the distribution of performance rather than merely to obtain a point estimate such as an average. The framework distinguishes expected quality from variability in system performance.

(3) Inference should replace informal aggregation. SafeLLM4SE recommends quantifying estimation uncertainty using CIs, applying statistical tests appropriate to the experimental design, and reporting efect sizes rather than relying solely on diferences in means or �-values.

(4) Quality, stability, and uncertainty are complementary. A reliable evaluation should report not only how well a system performs on average (quality), but also how consistently it performs across executions (stability) and how precisely these quantities have been estimated (uncertainty).

(5) Statistical rigor should be balanced against evaluation cost. Repeated LLM invocations incur API costs, latency, and computational resources. SafeLLM4SE therefore recommends adaptive sampling: an experiment begins with an initial number of executions and continues sampling until the estimate reaches a predefined precision target or the available budget is exhausted. This replaces arbitrary rules such as “always run the benchmark 100 times” with an explicit relationship between statistical precision and evaluation cost.

Finally, we demonstrate the application of SafeLLM4SE by using its Python implementation to evaluate and compare two LLMs on the HumanEval benchmark.

The remainder of this article is organized as follows. Section 2 reviews prior work on stochastic evaluation, statistical analysis, and reproducible reporting, and situates SafeLLM4SE within these research directions. Section 3 presents the methodology, from probabilistic modeling and adaptive sampling to statistical characterization, reporting, comparison, and evidence-based decision making. Section 4 illustrates the methodology through a HumanEval case study, describing the experimental setup and showing how quality, stability, and estimation uncertainty inform the comparison of two LLMs. Finally, Section 5 summarizes the main lessons and their implications for the evaluation of LLM-based SE systems.

## 2 Related Work

A growing body of empirical research documents non-determinism in code generation by LLM-based systems. Beyond the study by Ouyang et al. [18] discussed in Section 1, HumanEval itself was introduced together with the pass@� estimator to summarize functional correctness over repeated samples [5]; however, pass@� summarizes the probability of obtaining at least one correct solution among � samples per problem. Although pass@1 estimates the mean success probability for a single generation, these point estimates alone do not characterize variability across complete benchmark executions or quantify estimation uncertainty, both of which are central to SafeLLM4SE. More broadly, Liang et al. [15] proposed HELM, a large-scale, multi-metric evaluation framework for language models that emphasizes standardization and transparency across scenarios. However, HELM does not prescribe how repeated stochastic generations should be sampled, aggregated, or statistically compared.

Concerns about treating machine learning benchmark results as fixed numbers rather than as outcomes of a stochastic process predate LLMs. Bouthillier et al. [3] showed that multiple sources of randomness, beyond the commonly reported seed, can substantially afect benchmark outcomes and proposed variance-aware evaluation procedures. In deep reinforcement learning, Colas et al. [7] applied power analysis to determine adequate numbers of runs, and Agarwal et al. [1] advocated reporting interval estimates and robust aggregate statistics instead of isolated point estimates across runs. SafeLLM4SE builds on these ideas but adapts them to the specific setting of LLM-based SE systems, where evaluation cost is dominated by API usage and token consumption rather than training time, motivating its adaptive, budget-aware sampling procedure.

Within empirical SE, Arcuri and Briand [2] established influential guidelines for statistically analyzing randomized algorithms, recommending non-parametric tests, efect sizes, and a minimum number of repeated runs; SafeLLM4SE follows a similar spirit but targets LLM-based systems specifically, adding confidence-interval-driven adaptive sampling and a dedicated reporting standard. In the natural language processing community, Dror et al. [9] and Dodge et al. [8] similarly criticized the reliance on single-run comparisons and proposed protocols for significance testing and computation-aware reporting, respectively, though neither targets the SE-specific artifacts and cost trade-ofs (e.g., functional correctness, token budgets) that SafeLLM4SE addresses.

Finally, reporting checklists for reproducible machine learning research, such as the recommendations of Pineau et al. [19], share SafeLLM4SE’s goal of making empirical results more transparent and comparable. SafeLLM4SE complements these general checklists with an SE-specific methodology that integrates adaptive sampling, distribution-aware statistical comparison, and a minimum reporting standard tailored to stochastic LLM-based SE evaluation. This methodology is made available through an open-source software tool to lower the barrier to adoption [17].

## 3 SafeLLM4SE

SafeLLM4SE defines a minimum evaluation protocol and reporting standard for experimental evaluations of LLM-based SE systems under stochastic generation. Rather than treating an LLM output as a

deterministic artifact, SafeLLM4SE models it as a realization of a stochastic process and specifies the statistical evidence that should be reported to support reliable, reproducible, and comparable experimental conclusions.

## 3.1 Probabilistic Modeling

Given a prompt �, model �, and LLM configuration �, each execution of an LLM produces one realization of a random variable $Y _ { p , m , c } .$ Repeated executions under the same experimental conditions are treated as realizations from the underlying stochastic process governing the system’s outputs.

Depending on the task and the chosen unit of analysis, each output or execution is evaluated using a metric �(�). Examples include functional correctness of generated code (pass or fail), the number of tests passed (integer-valued), execution time, or code quality (real-valued). Let $X _ { p , m , c } = M ( Y _ { p , m , c } )$ denote the resulting random variable. The evaluation is then based on samples of $X _ { p , m , c }$ and aims to characterize its underlying performance distribution $F _ { p , m , c }$

SafeLLM4SE considers a target property � of $F _ { p , m , c }$ , such as its mean or a success probability, and estimates it from repeated observations. The choice of� depends on the metric and unit of analysis used in the evaluation. This distribution captures the expected performance of the system and its variability across repeated executions, whereas uncertainty arises from estimating properties of this distribution from a finite sample. Consequently, a single observation or point estimate such as the sample mean provides only a partial characterization of system behavior. SafeLLM4SE therefore treats repeated executions and the statistical characterization of the resulting distribution as fundamental components of LLM-based SE evaluation.

## 3.2 Adaptive Sampling

Instead of fixing an arbitrary number of executions, SafeLLM4SE uses adaptive sampling to determine when a suficient number of executions has been collected to estimate �, a target property of the performance distribution $F _ { p , m , c } ,$ , with the desired precision. This avoids both undersampling, which may lead to unreliable estimates, and unnecessary executions once additional samples provide little improvement in precision. Consequently, adaptive sampling can reduce API and computational costs, as well as the associated energy consumption and carbon footprint, while maintaining a predefined level of statistical precision.

For a given prompt �, model �, and configuration �, the procedure takes three parameters: a minimum sample size $N _ { \mathrm { m i n } } ,$ a target confidence-interval precision $w _ { \mathrm { m a x } } .$ , and a token-budget threshold � expressed as a number of tokens. The observations collected during sampling are, as mentioned, realizations of $X _ { p , m , c } = M ( Y _ { p , m , c } )$ . Algorithm 1 summarizes the procedure.

The confidence interval �� is computed according to the type of metric being estimated. For continuous metrics, SafeLLM4SE uses a standard �-based CI when the sampled values can reasonably be assumed to follow an approximately normal distribution [11]. When this assumption is not appropriate, SafeLLM4SE uses a bootstrap CI instead [10]. This provides a consistent procedure for metrics whose distributions may be skewed or otherwise depart substantially from normality. For binary metrics, such as whether generated code passes the functional tests, SafeLLM4SE uses the Wilson score interval [20]. This is preferred to the standard Wald interval, whose coverage can be poor when the estimated proportion is close to 0 or 1.

Algorithm 1 Adaptive sampling   
Require: Prompt ${ \boldsymbol { p } } ,$ , model �, configuration �, minimum sample size $N _ { \mathrm { m i n } } ,$ , target precision $w _ { \mathrm { m a x } } ,$   
execution budget $B$ (in tokens)   
Ensure: Sample $S ,$ estimate $\hat { \theta }$ and tokens $T$   
1: Execute $( p , m , c ) \ N _ { \mathrm { m i n } }$ times, collecting the corresponding observations of $X _ { p , m , c }$ in �   
2: $T$ ←total number of tokens consumed so far   
3: $n \gets | S |$   
4: Estimate $\theta$ from � and obtain $\hat { \theta }$   
5: Compute a confidence interval $C I ( S )$ for $\theta$   
6: while $T < B$ and width $( C I ( S ) ) > w _ { \mathrm { m a x } }$ do   
7: Execute $\left( { p , m , c } \right)$ once and add the resulting observation of $X _ { p , m , c }$ to $S$   
8: $T \gets T +$ tokens consumed by this call   
9: $n \gets n + 1$   
10: Estimate $\theta$ from � and obtain $\hat { \theta }$   
11: Compute a confidence interval $C I ( S )$ for $\theta$   
12: end while   
13: return $S , { \hat { \theta } } , T$

The target precision $w _ { \mathrm { m a x } }$ is defined as the maximum total width of the CI for the target property �. For example, a target width of 0.10 corresponds to a margin of error $\mathrm { o f } \pm 0 . 0 5 $ . Thus, the required number of executions is determined by the observed variability of $X _ { p , m , c }$ rather than by an arbitrary fixed value: stable evaluation conditions may reach the target precision with relatively few executions, whereas highly variable conditions require additional samples. This allows SafeLLM4SE to adapt the number of executions to the observed variability of each evaluation instance while targeting a predefined CI width.

## 3.3 Statistical Characterization

SafeLLM4SE requires each evaluation to characterize the three dimensions of quality, stability, and estimation uncertainty, while allowing the specific indicators used for quality and stability to be selected according to the evaluation context. Together, these dimensions provide a more complete description of the performance distribution $F _ { p , m , c }$ than any single point estimate.

Quality describes the central performance of the technique under the evaluation conditions. For a target property � of $F _ { \substack { p , m , c } } ,$ SafeLLM4SE recommends reporting the arithmetic mean as the default measure of central tendency. When the distribution is clearly skewed or contains strong outliers, the median may be reported as a robust measure of typical performance. For binary outcomes, such as whether generated code compiles or whether all functional tests pass, quality is characterized by the probability of success, estimated as the proportion of successful executions among all executions. For proportion-valued outcomes, such as the benchmark success rate obtained in a complete benchmark execution, quality is characterized by the mean of that performance measure across repeated executions. The direction of the metric should also be made explicit (i.e., whether higher or lower values indicate better performance).

Stability describes the variability of $X _ { p , m , c }$ across repeated executions under the same evaluation conditions. SafeLLM4SE recommends reporting the standard deviation as the default measure of dispersion. The coeficient of variation may additionally be reported when comparing techniques with diferent mean performance, provided that the metric has a meaningful ratio scale and a non-zero mean. For highly skewed or heavy-tailed distributions, the interquartile range provides a more robust alternative. Additional distributional descriptors, such as percentiles, may be reported when they provide useful information about the behavior of the system.

Estimation uncertainty arises because $F _ { p , m , c }$ is inferred from a finite sample of executions. Consequently, the estimated property $\hat { \theta }$ is subject to sampling uncertainty. SafeLLM4SE therefore recommends reporting a CI for all primary performance indicators, using the procedure described in Section 3.2, with a 95% confidence level by default. This uncertainty should be clearly distinguished from the variability of the LLM: stability describes how much $X _ { p , m , c }$ varies across executions, whereas the CI describes the precision with which the target property � of $F _ { p , m , c }$ has been estimated from the available sample.

## 3.4 Reporting And Visualization

SafeLLM4SE defines the minimum information that an evaluation should report to allow readers to assess the statistical validity, reproducibility, and practical reliability of its results. The reporting requirements cover the three dimensions mentioned—quality, stability, and estimation uncertainty—as well as the experimental protocol used to generate and evaluate the observations.

Table 2 summarizes the SafeLLM4SE reporting standard. For each evaluated technique, researchers should report the selected quality and stability indicators and their corresponding estimates. CIs and the confidence level used to construct them are required for all primary performance indicators.

For each estimated quality indicator ${ \hat { \theta } } ,$ the reported estimate should be accompanied by its CI. Reporting only a point estimate, such as the mean success rate, is therefore insuficient because it does not indicate the precision of the estimate. Likewise, reporting a CI without describing the number of executions and the sampling procedure makes it dificult to assess how the estimate was obtained.

The reporting of model and execution information is particularly important for evaluations using commercial LLM APIs. Model behavior may change over time because of model updates or changes to the serving infrastructure, even when the same model identifier is used. Researchers should therefore report the exact model identifier or version, the execution date and time, and any available snapshot or deployment information. Random seeds should also be reported when the provider exposes them, although a seed should not be assumed to guarantee reproducibility unless the underlying service provides such a guarantee.

Whenever possible, numerical summaries should be complemented with visualizations of the observed values of $X _ { p , m , c }$ . Boxplots, violin plots, raincloud plots, or empirical cumulative distribution functions (ECDFs) can reveal diferences in variability, skewness, heavy tails, and other distributional characteristics that may be obscured by summary statistics alone. The HumanEval case study illustrates this approach (Figure 1).

Collectively, these requirements constitute the SafeLLM4SE reporting standard. They provide reviewers and practitioners with the information needed to distinguish the observed performance of an LLM-based SE system from its variability and estimation uncertainty, while providing suficient experimental detail to support meaningful replication and comparison.

Table 2. SafeLLM4SE minimum reporting standard for evaluations of LLM-based SE techniques.
<table><tr><td>Category</td><td>Information to report</td><td>Purpose</td></tr><tr><td>Quality</td><td>Selected quality indicator(s) and estimated value .</td><td>Characterize the performance of the technique.</td></tr><tr><td>Stability</td><td>Selected stability indicator(s), such as standard deviation, coefficient of variation, or interquar- tile range.</td><td>Characterize variability across repeated executions.</td></tr><tr><td>Estimation uncertainty</td><td>CI for each primary performance indicator and confidence level.</td><td>Quantify the precision of the reported estimates.</td></tr><tr><td>Sampling</td><td>Number of executions, sampling procedure, minimum sample size, stopping criterion, and execution budget, if applicable.</td><td>Assess the adequacy and effi- ciency of the sampling proce- dure.</td></tr><tr><td>Model</td><td>Model name, model version or identifier, and provider.</td><td>Identify the system being evaluated and facilitate repli- cation.</td></tr><tr><td>Inference configuration</td><td>Temperature and other inference parameters that may affect generation.</td><td>Specify the conditions under which outputs were gener- ated.</td></tr><tr><td></td><td>Random seed, when supported; exact prompt(s); Reproducibility execution date and time; and, when applicable, API or model snapshot information.</td><td>Account for sources of varia- tion and support reproducibil- ity.</td></tr><tr><td>Evaluation</td><td>Benchmark or task version, evaluation proce- dure, test suite or other evaluation artifacts, and software/tool versions.</td><td>Make the evaluation proce- dure reproducible.</td></tr><tr><td>Resources</td><td>Number of generated tokens or equivalent us- age information and, when available, execution cost.</td><td>Quantify the computational and economic resources re- quired.</td></tr></table>

## 3.5 Statistical Comparison

Comparisons between LLM-based SE techniques should be performed over the observed values of $X _ { p , m , c }$ and the corresponding performance distributions $F _ { \substack { p , m , c } } ,$ rather than over isolated point estimates. An essential distinction is whether the observations are independent or paired. Samples are independent when there is no one-to-one correspondence between observations from the two techniques, such as when two techniques are evaluated using diferent sets of users or independently selected software projects. Samples are paired when both techniques are evaluated under the same experimental conditions, creating a natural correspondence between observations. For example, when two models are evaluated on the same HumanEval problems, the results for each problem form a pair. Pairing applies to problemlevel outcomes; complete benchmark executions are paired only if the experimental design links individual runs across techniques.

Table 3. SafeLLM4SE statistical comparison protocol.
<table><tr><td>Design</td><td>Significance test</td><td>Confidence interval (CI)</td><td>Effect size</td></tr><tr><td></td><td>Independent Mann-Whitney U</td><td>Bootstrap CI for the difference</td><td>Cliff&#x27;s δ</td></tr><tr><td>Paired</td><td>Wilcoxon signed- rank</td><td>Paired bootstrap CI</td><td>Matched-pairs rank-biserial correlation</td></tr></table>

SafeLLM4SE defines a default statistical procedure for each experimental design, summarized in Table 3. These procedures constitute the recommended minimum protocol rather than a collection of interchangeable alternatives.

For independent samples, SafeLLM4SE specifies the Mann–Whitney U test because it provides a nonparametric comparison of two distributions without requiring normally distributed observations [11]. The test is complemented by Clif’s �, which describes how often values from one technique exceed those from the other [6]. The diference in quality, $\theta _ { A } - \theta _ { B }$ , is estimated by $\hat { \theta } _ { A } - \hat { \theta } _ { B } . \operatorname { A }$ bootstrap CI for $\theta _ { A } - \theta _ { B }$ quantifies uncertainty in this estimate without requiring a parametric distributional assumption.

For paired samples with suitably symmetric within-pair diferences, SafeLLM4SE recommends the Wilcoxon signed-rank test on the per-instance diferences $\hat { \theta } _ { p , A , c } - \hat { \theta } _ { p , B , c } \left[ 1 1 \right]$ . For markedly asymmetric diferences or binary paired outcomes, the test should be chosen according to the data. The matchedpairs rank-biserial correlation is reported as the efect size because it quantifies the balance between favorable and unfavorable signed ranks and provides a directional measure of the magnitude of the paired efect [14]. A paired bootstrap CI for $\theta _ { A } - \theta _ { B }$ is computed by resampling complete pairs, thereby preserving the dependence structure between the two techniques.

The three reported quantities provide complementary information. The statistical test indicates whether the observed data provide evidence of a diference according to the selected test statistic, the CI quantifies the precision of the estimated diference in the target property, and the efect size describes the magnitude of the distributional or paired efect. Statistical significance should therefore not be interpreted as evidence of practical importance on its own: with suficiently large samples, even very small diferences may become statistically significant.

SafeLLM4SE also recommends reporting the magnitude of non-parametric efect sizes using established interpretation guidelines. For independent samples, Clif’s � ranges from −1 to 1 and measures the diference between the probability that an observation from one technique exceeds an observation from the other and the reverse probability [6]. Values close to zero indicate that these probabilities are approximately balanced, but do not necessarily imply similar distributions. Larger absolute values indicate a stronger tendency for one technique to yield higher values than the other. The conventional thresholds are approximately |� | = 0.147, 0.33, and 0.474 for small, medium, and large efects, respectively. For paired comparisons, the matched-pairs rank-biserial correlation has the same [−1, 1] range and provides an analogous interpretation of efect direction and magnitude [14]. These thresholds should be regarded as general guidelines rather than universal definitions of practical importance; the substantive relevance of an efect ultimately depends on the task, metric, and application context.

Alternative statistical procedures may be used when the experimental design or structure of the data makes the default protocol inappropriate or violates its underlying assumptions. Such deviations should be explicitly justified and reported. Regardless of the procedure used, a statistical comparison should report the estimated diference $\hat { \theta } _ { A } - \hat { \theta } _ { B } ,$ , its CI, the statistical test used, the resulting �-value, the chosen significance level, and the efect size. Together, these quantities allow readers to assess whether a diference is statistically detectable, how precisely it has been estimated, and whether its magnitude is practically meaningful.

## 3.6 Decision Making

SafeLLM4SE has two complementary objectives: (1) to provide a statistically principled methodology for analyzing stochastic LLM-based SE systems, and (2) to establish a minimum reporting standard that facilitates reproducibility, comparability, and evidence-based decision making across LLM4SE studies.

The evidence collected through SafeLLM4SE should support, rather than replace, experimental judgment. Let $\theta _ { A }$ and $\theta _ { B }$ denote the target properties of the performance distributions $F _ { A }$ and $F _ { B }$ for two competing techniques under the same evaluation conditions, and let $\hat { \theta } _ { A }$ and ${ \hat { \theta } } _ { B }$ be their estimates. A higher observed value ${ \hat { \theta } } _ { A }$ alone is therefore not suficient to conclude that technique � is superior to technique �. Such a conclusion should consider the estimated diference, its uncertainty, the magnitude of the efect, and the stability of the two techniques.

A claim that one technique is superior to another should be supported by evidence that considers:

(1) Quality. The technique achieves a better value of the target property $\theta ,$ taking into account the direction of the metric. For example, a higher success probability indicates better quality for binary correctness outcomes, whereas a lower mean execution time indicates better performance for latency measurements.

(2) Estimation uncertainty. The observed diference between $\hat { \theta } _ { A }$ and ${ \hat { \theta } } _ { B }$ should be interpreted together with its CI. A diference should not be considered conclusive when the available data provide insuficient precision to distinguish the two techniques reliably (i.e., when the CI for the estimated diference is compatible with both a meaningful advantage and no meaningful diference).

(3) Statistical evidence. The observed diference should be supported by the statistical analysis described in Section 3.5, rather than being plausibly attributable to sampling variation alone.

(4) Practical relevance. The estimated efect should be suficiently large to be meaningful for the task and application, rather than being statistically significant but practically negligible.

(5) Stability. The variability of $X _ { p , m , c }$ should be considered alongside quality and estimation uncertainty. Lower variability generally indicates more stable performance, but greater variability may be an acceptable trade-of when it is accompanied by a suficiently meaningful improvement in quality.

These criteria should be considered jointly. A technique may, for example, achieve a higher $\hat { \theta }$ while also exhibiting greater variability or substantial estimation uncertainty. In such cases, declaring a clear overall winner may be inappropriate, and the corresponding trade-ofs should be reported explicitly. Similarly, when the estimated diference is small, imprecisely estimated, or associated with a negligible efect size, researchers should avoid definitive superiority claims.

SafeLLM4SE therefore does not prescribe a universal ranking rule or a single decision criterion. Instead, it provides the statistical evidence needed to distinguish clear improvements from trade-ofs and from diferences that cannot be reliably established. This supports transparent and evidence-based decisions about which LLM-based SE technique is preferable for a given task and application context.

## 4 Case Study: Applying SafeLLM4SE To HumanEval

This section illustrates the application of SafeLLM4SE to the evaluation and comparison of two LLMs. We consider gemini-3.1-flash-lite, accessed through the Gemini API, and qwen2.5-coder:7b, accessed through Ollama. Both models are evaluated on the 164 problems of HumanEval using a temperature of 2.0. The experiment deliberately uses a relatively high temperature to amplify stochastic variation and make its efects more apparent in the evaluation results.

## 4.1 Experimental Setup

We treat HumanEval as a single benchmark evaluation comprising its 164 programming problems. Each execution of a model consists of generating one solution for each of the 164 problems and evaluating the generated solutions using the HumanEval functional tests. Thus, each complete execution produces one observation of the model’s stochastic performance on the benchmark.

For each model, we apply the adaptive sampling proposed by SafeLLM4SE. We use $N _ { \mathrm { m i n } } = 3 0$ , an unlimited execution budget, and require the 95% bootstrap CI for the mean benchmark success rate to achieve a maximum total width of $w _ { \mathrm { m a x } } = 0 . 0 1$ (i.e., a maximum margin of error of ±0.005). Since no budget limit is imposed in this case study, sampling stops once at least $N _ { \mathrm { m i n } }$ complete benchmark executions have been collected and the observed CI width is at or below $w _ { \mathrm { m a x } } .$

The outcome for each generated solution is binary: $X _ { \substack { p , m , c , i } } = 1$ when the solution generated for problem $\boldsymbol { p }$ in execution � passes all HumanEval tests, and $X _ { p , m , c , i } = 0$ otherwise. Thus, the performance of execution � is summarized as the proportion of correctly solved HumanEval problems,

$$
X _ { m , c , i } = \frac { 1 } { 1 6 4 } \sum _ { p = 1 } ^ { 1 6 4 } X _ { p , m , c , i } .\tag{1}
$$

Thus, $X _ { m , c , i }$ is a proportion-valued performance measure in the interval [0, 1], with higher values indicating better quality. The target property for quality, $\theta _ { m , c } ,$ , is the expected benchmark success rate under the evaluated conditions. We estimate this quantity using the sample mean of the success rates observed across � complete benchmark executions:

$$
\hat { \theta } _ { m , c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } X _ { m , c , i } .\tag{2}
$$

Quality is therefore estimated from the distribution of complete benchmark executions rather than from separate problem-level success probabilities.

For each model, stability is assessed from the variability of $X _ { m , c , i }$ across repeated executions. We report the standard deviation of the benchmark success rate as the primary stability measure.

The CI for the mean benchmark success rate is computed from the � execution-level observations using bootstrap resampling. Each bootstrap sample consists of � complete benchmark executions sampled with replacement, and the mean success rate is computed for each sample. The 95% percentile bootstrap CI is defined by the 2.5th and 97.5th percentiles of the bootstrap distribution of the mean benchmark success rate. This interval quantifies uncertainty in the estimated mean performance across repeated executions of the fixed set of 164 HumanEval problems.

Table 4. HumanEval results reported according to the SafeLLM4SE reporting standard.
<table><tr><td>Category</td><td>Information</td><td>Gemini</td><td>Qwen-Coder</td></tr><tr><td>Quality</td><td>Mean benchmark success rate  $( \hat { \theta } _ { m , c } )$ </td><td>0.9335</td><td>0.8238</td></tr><tr><td>Stability</td><td>Standard deviation of bench- mark success rate</td><td>0.0110</td><td>0.0204</td></tr><tr><td>Estimation uncertainty</td><td>95% CI for mean benchmark success rate</td><td>[0.9294, 0.9376]</td><td>[0.8188, 0.8287]</td></tr><tr><td>Sampling</td><td> $N _ { \mathrm { m i n } } ;$  target CI; budget; N; total generated solutions</td><td>30; width of  $9 5 \% \mathrm { C I } \leq 0 . 0 1 ;$  unlimited; 30; 4,920</td><td>30; width of  $9 5 \% \mathrm { C I } \leq 0 . 0 1 ;$  unlimited; 67; 10,988</td></tr><tr><td>Model</td><td>Model name, identifier, and provider</td><td>Gemini, gemini-3.1-flash- lite; Gemini API v1beta</td><td>Qwen-Coder, qwen2.5-co- der: 7b; Ollama v0.34.4</td></tr><tr><td>Inference configuration</td><td>Temperature</td><td>2.0 Sep. 24, 2026, 18:46; Hu-</td><td>2.0</td></tr><tr><td>Reproducibility</td><td>Execution date; Prompts; API/model information; seed</td><td>manEval v0.1.10 prompts as available on that date; on- line Gemini API as available on that date; seed unavailable</td><td>Sep. 24, 2026, 18:46; Hu- manEval v0.1.10 prompts as available on that date; Ollam v0.34.4; seed unavailable</td></tr><tr><td>Evaluation</td><td>Benchmark and evaluation procedure</td><td>HumanEval (164 problems); functional tests</td><td>HumanEval (164 problems); functional tests</td></tr><tr><td>Resources</td><td>Number of generated tokens and benchmark executions</td><td>2,542,131; 30</td><td>2,635,240; 67</td></tr></table>

## 4.2 SafeLLM4SE Reporting

Table 4 reports the experimental results according to the SafeLLM4SE reporting standard defined in Table 2. The results show that Gemini achieves a higher mean benchmark success rate, ${ \hat { \theta } } _ { m , c } ,$ of 0.9335 compared with 0.8238 for Qwen-Coder. Gemini also exhibits a lower observed standard deviation across complete HumanEval executions (0.0110 vs. 0.0204), indicating lower observed variability under the evaluated conditions.

Figure 1 shows the distribution ofbenchmark success rates across complete HumanEval executions for both models. Each observation corresponds to one complete execution of the 164 problems, providing a direct view of stochastic variation in benchmark-level performance. Gemini’s success rates are concentrated at higher values than those of Qwen-Coder and exhibit lower dispersion across repeated executions, indicating greater observed stability.

![](images/7c7641a41758a79433c5b8b0d8509633da36080388ab59396a2cf3e2b29241d9.jpg)  
Fig. 1. Distribution of benchmark success rates of Gemini and Qwen-Coder across repeated HumanEval executions obtained with SafeLLM4SE adaptive sampling. Each observation represents one complete execution of the 164 HumanEval problems. Figure generated with the SafeLLM4SE Python tool.

## 4.3 Statistical Comparison

The two models are evaluated through repeated complete executions ofthe same HumanEval benchmark, with each execution comprising the same 164 benchmark problems. The execution-level benchmark success rate $X _ { m , c , i }$ is used to characterize the quality and stability of each model across repeated executions. For the statistical comparison between models, the two samples of execution-level success rates are treated as independent: each execution is an unlinked realization of the model’s own stochastic process, with no run-level correspondence between models.

Following the SafeLLM4SE comparison protocol for independent observations, we compare the two samples of execution-level benchmark success rates using the Mann–Whitney U test. Clif’s � is reported as the efect size. In addition, we obtain a bootstrap CI for the diference in mean benchmark success rate by independently resampling the execution-level observations of each model with replacement, without assuming any correspondence between individual executions of the two models. The results are shown in Table 5.

## 4.4 Decision Making

The results provide consistent evidence in favor of Gemini’s gemini-3.1-flash-lite under the evaluated conditions. Its mean benchmark success rate is 10.97 percentage points higher than that of Qwen-Coder, with a 95% CI for the diference entirely above zero. Gemini also exhibits lower observed variability across repeated HumanEval executions. The Mann–Whitney U test indicates a diference between the observed distributions, and Clif’s $\delta = 1$ indicates complete stochastic separation in the observed samples: every observed Gemini benchmark score exceeds every observed Qwen-Coder score. Moreover, Gemini used fewer total tokens to reach the target CI width in this experiment. Thus, Gemini satisfies the SafeLLM4SE criteria for concluding that it provides superior performance under the evaluated conditions and exhibits lower observed variability across repeated benchmark executions.

Table 5. SafeLLM4SE comparison of the two models.
<table><tr><td>Comparison</td><td>Result</td></tr><tr><td>Mean difference</td><td>+0.1097</td></tr><tr><td>95% bootstrap CI for the difference in mean success rate</td><td> $[ 0 . 1 0 3 3 , 0 . 1 1 6 0 ]$ </td></tr><tr><td>Mann-Whitney U test</td><td> $U = 2 0 1 0 , \ p < 0 . 0 0 1$ </td></tr><tr><td> $\mathrm { C l i f f } ^ { \prime } s \delta$ </td><td>1.0</td></tr></table>

This case study was conducted entirely using the SafeLLM4SE application, developed as part of this work and used to generate all the data and Figure 1. The application is freely available for download [17] and is published on PyPI under the name SafeLLM4SE.

## 5 Conclusions

The main lesson from this work is that trustworthy evaluation of LLM-based SE systems requires treating their outputs as observations of a stochastic process rather than as deterministic artifacts. A single generation or point estimate can conceal substantial variability and may therefore provide an insuficient basis for engineering decisions. Trustworthy evaluation should instead characterize three complementary dimensions: quality, stability, and estimation uncertainty. This perspective shifts the focus from asking which system produced the highest observed score in a particular evaluation to asking what performance can be expected across repeated executions and how precisely that performance has been estimated.

A second lesson is that statistical rigor and practical evaluation constraints should be considered together. Repeated executions are necessary to characterize stochastic behavior, but fixed sample sizes can either provide insuficient evidence or incur unnecessary cost. Adaptive sampling relates the number of executions to a predefined CI-width target, concentrating evaluation efort where additional observations are needed. Its stopping rule and interval coverage should be reported and assessed together. Similarly, comparisons should consider uncertainty and efect magnitude, rather than relying on statistical significance or diferences in point estimates alone. These principles make the resulting evidence more useful for informed engineering decisions and reduce the risk of drawing strong conclusions from inherently variable systems.

Ultimately, trustworthy AI-enabled SE depends not only on the capabilities of the underlying models, but also on the quality and transparency of the evidence used to evaluate them. SafeLLM4SE provides a practical basis for producing such evidence and for making evaluations more reproducible, comparable, and accountable.

The software developed in this work, together with its source code and the data used in the case study, is freely available [17].

## Acknowledgments

This work was supported by the Ministry of Science, Innovation and Universities and co-funded by the European Regional Development Fund (ERDF) under project PID2024-155586OB-I00 (MCIU/AEI/10.13039 /501100011033). Additional support was provided by the Government of the Principality of Asturias through grant GRU-GIC-24-070.

## References

[1] Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G. Bellemare. 2021. Deep Reinforce ment Learning at the Edge of the Statistical Precipice. arXiv preprint arXiv:2108.13264 (2021).

[2] Andrea Arcuri and Lionel Briand. 2014. A Hitchhiker’s Guide to Statistical Tests for Assessing Randomized Algorithms in Software Engineering. Software Testing, Verification and Reliability 24, 3 (2014), 219–250.

[3] Xavier Bouthillier, Pierre Delaunay, Mirko Bronzi, Assya Trofimov, Brennan Nichyporuk, Justin Szeto, Nazanin Moham madi Sepahvand, Edward Raf, Kanika Madan, Vikram Voleti, et al. 2021. Accounting for Variance in Machine Learning Benchmarks. In Proceedings ofMachine Learning and Systems, Vol. 3. 747–769.

[4] Timothée Chauvin, Erwan Le Merrer, François Taïani, and Gilles Tredan. 2026. Log Probability Tracking of LLM APIs. In Proceedings of the International Conference on Learning Representations.

[5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374 (2021).

[6] Norman Clif. 1993. Dominance Statistics: Ordinal Analyses to Answer Ordinal Questions. Psychological Bulletin 114, 3 (1993), 494–509. doi:10.1037/0033-2909.114.3.494

[7] Cédric Colas, Olivier Sigaud, and Pierre-Yves Oudeyer. 2018. How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments. arXiv preprint arXiv:1806.08295 (2018). arXiv:1806.08295 [cs.LG] https://arxiv.org/abs/1806.08295

[8] Jesse Dodge, Suchin Gururangan, Dallas Card, Roy Schwartz, and Noah A. Smith. 2019. Show Your Work: Improved Reporting of Experimental Results. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2185–2194.

[9] Rotem Dror, Gili Baumer, Segev Shlomov, and Roi Reichart. 2018. The Hitchhiker’s Guide to Testing Statistical Significance in Natural Language Processing. In Proceedings ofthe 56th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 1383–1392.

[10] Bradley Efron and Robert J. Tibshirani. 1993. An Introduction to the Bootstrap. Monographs on Statistics and Applied Probability, Vol. 57. Chapman & Hall/CRC, Boca Raton, FL, USA.

[11] Myles Hollander, Douglas A. Wolfe, and Eric Chicken. 2013. Nonparametric Statistical Methods (3rd ed.). John Wiley & Sons, Hoboken, NJ.

[12] Xinyi Hou, Yanjie Zhao, Yue Liu, Zhou Yang, Kailong Wang, Li Li, Xiapu Luo, David Lo, John Grundy, and Haoyu Wang. 2024. Large Language Models for Software Engineering: A Systematic Literature Review. ACM Transactions on Software Engineering and Methodology 33, 8 (2024), 1–79. doi:10.1145/3695988

[13] Daniel Kang et al. 2026. Behavioral Fingerprints for LLM Endpoint Stability and Identity. In Proceedings ofthe ACM Conference on AI and Agentic Systems. 1327–1331. doi:10.1145/3786335.3813194

[14] Dave S. Kerby. 2014. The Simple Diference Formula: An Approach to Teaching Nonparametric Correlation. Comprehensive Psychology 3 (2014), 1. doi:10.2466/11.IT.3.1

[15] Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove,

Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao,Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. 2023. Holistic Evaluation of Language Models. Transactions on Machine Learning Research (2023).

[16] Zairah Mustahsan, Abel Lim, Megna Anand, Saahil Jain, and Bryan McCann. 2025. Stochasticity in Agentic Evaluations: Quantifying Inconsistency with Intraclass Correlation. arXiv:2512.06710 [cs.AI] https://arxiv.org/abs/2512.06710

[17] Francisco Ortin. 2026. SafeLLM4SE. https://github.com/francisco-ortin/safellm4se GitHub repository.

[18] Shuyin Ouyang, Jie M. Zhang, Mark Harman, and Meng Wang. 2025. An Empirical Study of the Non-Determinism of ChatGPT in Code Generation. ACM Transactions on Software Engineering and Methodology 34, 2 (2025), 1–28. doi:10.1145/3697010

[19] Joelle Pineau, Philippe Vincent-Lamarre, Koustuv Sinha, Vincent Larivière, Alina Beygelzimer, Florence d’Alché Buc, Emily Fox, and Hugo Larochelle. 2021. Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program). Journal ofMachine Learning Research 22, 164 (2021), 1–20.

[20] Edwin B. Wilson. 1927. Probable Inference, the Law of Randomness, and Single Samples. J. Amer. Statist. Assoc. 22, 158 (1927), 209–212. doi:10.1080/01621459.1927.10502953