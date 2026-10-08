# VM-ARRAYDPS: VIRTUAL MICROPHONE AUGMENTED DIFFUSION POSTERIORSAMPLING FOR UNSUPERVISED BLIND SPEECH SEPARATION

Jingqi Sun, Haozhan Tang, Shulin He, and Zhong-Qiu Wang

Department of Computer Science and Engineering Southern University of Science and Technology, Shenzhen, China jingqi.sun@outlook.com, wang.zhongqiu41@gmail.com

## ABSTRACT

Blind source separation (BSS) aims to separate multiple source signals from their mixtures without prior knowledge of the sources or mixing process. ArrayDPS addresses this task using a pretrained speech diffusion model, with multi-channel consistency (MC) constraints derived from physical microphone observations guiding posterior sampling. However, these constraints become less effective when only a few physical microphones are available and their recordings have low signal-to-noise ratios (SNRs). To address these limitations, we propose VM-ArrayDPS, which uses independent vector analysis (IVA)-based spatial demixing to construct higher-SNR virtual microphone observations from the physical recordings. These virtual observations are incorporated alongside the physical microphone signals through a weighted MC objective, allowing the IVA outputs to guide source recovery throughout posterior sampling. Evaluation results on 2- and 3-speaker separation in reverberant conditions demonstrate consistent improvements over ArrayDPS.

Index Terms— Blind speech separation, diffusion models, multi-channel consistency

## 1. INTRODUCTION

Separating overlapping speech recorded by microphone arrays is challenging when only a few microphones are available and the recordings contain noise and reverberation. When neither the source signals nor the mixing process is known, this task is referred to as blind speech separation (BSS) [1, 2]. Classical approaches such as independent vector analysis (IVA) [3–8] address this task by exploiting statistical independence across sources and dependencies across frequency bins. Although these filters enhance individual sources, the outputs may still contain residual interference and noise.

Beyond the statistical source models used in classical methods, neural speech separation methods learn speech structure directly from data [9]. Supervised methods typically learn a direct mapping from mixtures to clean sources, requiring paired mixture–source training data, and their performance may degrade when the test conditions differ from those encountered during training [10–21]. Multi-channel speech separation further exploits spatial information from microphone arrays to distinguish and recover individual sources [22–26]. Unsupervised approaches such as UNSSOR remove the need for clean source references by exploiting mixture consistency during training, but still learn a task-specific separation model from the training mixture distribution [27, 28]. An alternative is to use a pretrained speech diffusion model to infer the sources at test time, without training a task-specific separator [29, 30]. To separate a particular mixture, the sampling process must be guided by the recorded signals. Diffusion posterior sampling (DPS) provides such guidance through the gradient of a measurement loglikelihood, which evaluates how well the source estimates account for the observations under a given measurement model [31]. In blind multi-channel separation, the acoustic filters defining this model are unknown. ArrayDPS estimates these filters from the current source estimates and microphone recordings, and constructs a multi-channel consistency (MC) objective from the resulting reconstruction error to guide posterior sampling [32].

This observation-based guidance, however, is constrained by the quality and number of available microphone observations. Array-DPS uses forward convolutive prediction (FCP) [33, 34] to estimate acoustic filters for reconstruction. With few microphones, the resulting MC objective provides limited spatial constraints on source recovery. Low-SNR recordings can further degrade acoustic filter estimation, making the reconstruction error less reliable for guiding posterior sampling. Together, these limitations weaken the observationbased guidance needed for accurate separation.

To address these limitations, we propose VM-ArrayDPS, which augments physical microphone observations with source-enhanced virtual microphones to strengthen the guidance for posterior sampling. These virtual observations are generated through IVA-based spatial demixing, which suppresses interfering sources and produces signals with higher SNR [4]. While ArrayDPS uses IVA outputs only for warm initialization [32], VM-ArrayDPS incorporates them as virtual microphone observations throughout posterior sampling. FCP estimates convolutional mappings from the source estimates to these virtual observations, and the corresponding reconstruction errors contribute additional terms to the MC objective. By introducing source-enhanced observations, VM-ArrayDPS supplements the limited physical observations and reduces reliance on their noisy reconstruction constraints.

Based on this observation, we propose VM-ArrayDPS, which augments ArrayDPS with higher-SNR virtual microphone observations without additional training or clean separation targets. The main contributions are:

• We introduce virtual microphone augmentation into ArrayDPS by incorporating IVA-derived observations as additional measurements for posterior sampling.

• We formulate a virtual-microphone MC objective and introduce a weighting strategy to control the contribution of physical and virtual observations.

Experiments on 2- and 3-speaker SMS-WSJ [35] demonstrate consistent improvements over ArrayDPS [32], with ablations analyzing the effects of the number and weight of virtual microphones.

## 2. REVIEW OF ARRAYDPS

Consider a mixture of C reverberant speech sources observed by M microphones:

$$
y _ { m } ( t ) = \sum _ { c = 1 } ^ { C } h _ { m , c } ( t ) * s _ { c } ( t ) + n _ { m } ( t ) ,\tag{1}
$$

where $s _ { c } ( t )$ is the c-th source signal, $h _ { m , c } ( t )$ is its room impulse response (RIR) to microphone m, ∗ denotes convolution, and $n _ { m } ( t )$ denotes additive noise. In blind separation, neither the sources nor the acoustic filters are assumed known.

ArrayDPS [32] formulates separation as sampling from the posterior distribution

$$
p ( \boldsymbol { s } _ { 1 : C } \mid \boldsymbol { y } _ { 1 : M } ) ,\tag{2}
$$

where $s _ { 1 : C }$ and $y 1 { : } M$ denote all source and microphone signals, respectively. DPS [31] combines a learned prior with a measurement likelihood. In the score-based formulation used by ArrayDPS, the posterior score can be decomposed by Bayes’ rule as

$$
\nabla _ { \mathbf { s } ^ { \tau } } \log p ( \boldsymbol { s } ^ { \tau } \mid \boldsymbol { y } ) = \nabla _ { \boldsymbol { s } ^ { \tau } } \log p ( \boldsymbol { s } ^ { \tau } ) + \nabla _ { \boldsymbol { s } ^ { \tau } } \log p ( \boldsymbol { y } \mid \boldsymbol { s } ^ { \tau } ) ,\tag{3}
$$

where τ is the diffusion noise level. The first term is the prior score. A pretrained speech diffusion model supplies a denoised estimate $\hat { s } ^ { \tau }$ from which the score can be estimated using Tweedie’s formula [36]:

$$
\nabla _ { s ^ { \tau } } \log { p ( s ^ { \tau } ) } = \frac { \hat { s } ^ { \tau } - s ^ { \tau } } { \sigma ( \tau ) ^ { 2 } } .\tag{4}
$$

This prior encourages the recovered signals to remain on the manifold of plausible speech. To connect the source prior to the actual observations, ArrayDPS constructs an estimate of each microphone signal using estimated acoustic transfer functions:

$$
\hat { y } _ { m } ( t ) = \sum _ { c = 1 } ^ { C } ( \hat { h } _ { m , c } * \hat { s } _ { c } ^ { \tau } ) ( t ) .\tag{5}
$$

Assuming additive Gaussian observation noise, the negative loglikelihood is proportional to a reconstruction error,

$$
- \log p ( \boldsymbol { y } \mid \boldsymbol { s } ^ { \tau } ) \propto \sum _ { m = 1 } ^ { M } \left\| \boldsymbol { y } _ { m } - \hat { \boldsymbol { y } } _ { m } \big ( \boldsymbol { s } ^ { \tau } \big ) \right\| _ { 2 } ^ { 2 } .\tag{6}
$$

Consequently, the likelihood score is implemented as the gradient of the MC reconstruction loss:

$$
\nabla _ { s ^ { \tau } } \log p ( \boldsymbol { y } \mid s ^ { \tau } ) \approx - \xi ( \tau ) \nabla _ { s ^ { \tau } } \sum _ { m = 1 } ^ { M } \| y _ { m } - \hat { y } _ { m } ( s ^ { \tau } ) \| _ { 2 } ^ { 2 } ,\tag{7}
$$

where $\xi ( \tau )$ controls the likelihood contribution at the current diffusion level.

ArrayDPS estimates the acoustic transfer functions using FCP [33, 34]. In the short-time Fourier transform (STFT) domain, let $Y _ { m } ( \ell , f )$ denote the observation at microphone m, and $\hat { S } _ { c } ( \ell , f )$ denote the direct-path estimate of source c at time frame ℓ and frequency f. A K-tap frequency-dependent FCP filter $\hat { h } _ { m , c } ( f )$ is estimated by

$$
\hat { h } _ { m , c } ( f ) = \arg \operatorname* { m i n } _ { h _ { m , c } ( f ) } \sum _ { \ell } \frac { | Y _ { m } ( \ell , f ) - h _ { m , c } ( f ) ^ { \sf H } \tilde { \hat { S } } _ { c } ( \ell , f ) | ^ { 2 } } { \hat { \lambda } _ { c } ( \ell , f ) } ,\tag{8}
$$

![](images/32e94ee8c85ea9d46063741474ec1889a42e2e84b42f7fb4d15550da48c23390.jpg)  
Fig. 1: Gradient computation of VM-ArrayDPS at a diffusion step. Given the current source estimate $\mathbf { s } ^ { \tau }$ , the physical and virtual microphone observations are reconstructed and used to compute the corresponding likelihood gradients.

where $\tilde { \hat { S } }$ stacks the required delayed frames and $\hat { \lambda }$ is an observationdependent weighting term. The estimated filter is then used as a frequency-dependent approximation of the RTF for reconstructing the microphone observations.

Finally, ArrayDPS performs the sampling process with the EDM sampler [37]. The IVA output is used as a warm initialization, and the prior and likelihood scores are repeatedly combined to update the source estimates. This design allows a single-speaker speech diffusion model to be applied to the blind multi-speaker separation problem without supervised separation training [32].

## 3. VM-ARRAYDPS

VM-ArrayDPS strengthens DPS by incorporating source-enhanced virtual microphone observations. Generated through IVA-based spatial demixing, these observations supplement the physical microphone signals and provide additional consistency constraints for source recovery. Both physical and virtual observations are incorporated into a unified reconstruction objective, with a weighting factor controlling the contribution of the virtual observations. Fig. 1 illustrates how this objective guides source estimation at each diffusion step. The following subsections describe virtual microphone construction and its integration into posterior sampling.

## 3.1. Virtual Microphone Construction

Although the physical microphone observations provide spatial information, only a small number of microphones may be available in practice. This limits the number of observation constraints imposed on the source estimates, especially when multiple sources need to be recovered from the same set of mixtures. We therefore construct additional observations from the available microphone array, with the goal of obtaining constraints that are more source-oriented while remaining compatible with the forward acoustic model.

To this end, we apply IVA to the physical microphone signals and obtain separated signals through frequency-dependent spatial demixing filters [4]. Compared with the original microphone mixtures, these outputs are more source-oriented and typically exhibit a higher target-to-interference ratio. We use these separated signals as the basis for constructing virtual microphone observations.

We refer to the IVA outputs as virtual microphones because each output can be viewed as a transformed observation of the same underlying acoustic sources. The term “virtual” emphasizes that these observations do not correspond to additional physical microphone positions. Instead, they are obtained by applying linear spatial transformations to the available physical microphone signals, and can therefore be incorporated into the same forward acoustic model. For an IVA output $v _ { j } ( t )$ , we estimate a corresponding filter $\hat { g } _ { j , c }$ <sub>c</sub> and reconstruct it from the current source estimates:

$$
\hat { v } _ { j } ( t ) = \sum _ { c = 1 } ^ { C } ( \hat { g } _ { j , c } \ast s _ { c } ^ { \tau } ) ( t ) .\tag{9}
$$

This construction allows the source-oriented information contained in the IVA outputs to impose additional constraints on the current source estimates without introducing additional physical microphones or clean training data.

## 3.2. Virtual-Microphone Consistency

The virtual microphones are incorporated into the observation likelihood by requiring the current source estimates to explain both the physical microphone signals and the virtual microphone signals. Let $v _ { 1 : Q }$ denote $Q$ virtual microphone signals. Assuming conditional independence between physical and virtual observations given the source estimates, the likelihood score becomes

$$
\begin{array} { r } { \nabla _ { s ^ { \tau } } \log p ( \boldsymbol { y } , \boldsymbol { v } \mid \boldsymbol { s } ^ { \tau } ) \approx - \boldsymbol { \xi } ( \tau ) \nabla _ { s ^ { \tau } } \displaystyle \sum _ { m = 1 } ^ { M } \| \boldsymbol { y } _ { m } - \boldsymbol { \hat { y } } _ { m } \| _ { 2 } ^ { 2 } } \\ { - \boldsymbol { \zeta } ( \tau ) \nabla _ { s ^ { \tau } } \displaystyle \sum _ { j = 1 } ^ { Q } \| \boldsymbol { v } _ { j } - \boldsymbol { \hat { v } } _ { j } \| _ { 2 } ^ { 2 } . } \end{array}\tag{10}
$$

The physical and virtual observations provide complementary constraints: the former retain the original spatial mixtures, while the latter provide source-oriented projections derived from the same microphone array. However, the reliability of a virtual observation is not necessarily identical to that of a physical microphone. An IVA estimate can contain residual interference or artifacts, and forcing exact consistency with it may therefore be harmful. We therefore introduce a relative weight α and use

$$
\begin{array} { r } { \nabla _ { s ^ { \tau } } \log p ( \boldsymbol { y } , \boldsymbol { v } \mid s ^ { \tau } ) \approx - \xi ( \tau ) \nabla _ { s ^ { \tau } } \displaystyle \sum _ { m = 1 } ^ { M } \| { y } _ { m } - \hat { y } _ { m } \| _ { 2 } ^ { 2 } } \\ { - \alpha \cdot \xi ( \tau ) \nabla _ { s ^ { \tau } } \displaystyle \sum _ { j = 1 } ^ { Q } \| { v } _ { j } - \hat { v } _ { j } \| _ { 2 } ^ { 2 } . } \end{array}\tag{11}
$$

The original ArrayDPS likelihood is recovered when $Q = 0$ . In our experiments, α is tuned on the validation setting. Given M physical microphones and C sources, the construction can generate up to M · C virtual observations. In practice, we use only a small number of virtual microphones to improve the observation constraints while limiting the additional computational cost.

## 4. EXPERIMENTAL SETUP

## 4.1. Datasets and Implementation

We evaluate VM-ArrayDPS on the SMS-WSJ corpus [35], a spatialized multi-speaker speech corpus designed for multi-channel speech separation and recognition. The 2-speaker test set contains 1, 332 mixtures sampled at 8 kHz. The mixtures are simulated using a sixmicrophone circular array with a diameter of 20 cm. The sourceto-array distance is uniformly sampled from 1.0 to 2.0 m, the reverberation time $T _ { 6 0 }$ ranges from 0.2 to 0.5 s, and weak additive white noise is introduced with an SNR between 20 and 30 dB. Rather than using the complete six-channel array, we select a subset of physical microphones as the observed inputs, following the setting of Array-DPS and simulating practical scenarios where only a limited number of microphones are available. Specifically, three physical microphones are used in the 2-speaker experiments and four physical microphones are used in the 3-speaker experiments. We additionally evaluate VM-ArrayDPS on a 3-speaker extension generated following the SMS-WSJ simulation procedure.

The pretrained speech diffusion prior and EDM sampling configuration are inherited from the public ArrayDPS implementation [32]. No additional supervised training is performed for VM-ArrayDPS. Given the physical microphone observations, IVA is first applied to obtain source-oriented estimates. These estimates are then used to construct virtual microphone observations, which are incorporated into the posterior objective together with the physical microphone observations. The contribution of the virtual-microphone constraint is controlled by a weighting factor α.

For each utterance, we run sampler five times. Following Array-DPS, we consider three sampling-selection strategies. $\mathrm { \ddot { \Omega } A V \bar { G } ^ { \mathrm { 3 } } }$ averages the five generated outputs, “Max” selects the output with the highest SI-SDR, and “ML” selects the output with the highest likelihood score. We evaluate all three strategies in both the 2- and 3- speaker experiments to examine whether the effect of virtual microphones is consistent across different sample-selection procedures.

We report SDR [38], SI-SDR [39], PESQ [40], and eSTOI [41]. PESQ is an objective speech-quality measure standardized in ITU-T P.862, while eSTOI is an extended short-time objective intelligibility measure designed for intelligibility evaluation under challenging distortions [41]. Since the diffusion prior is not explicitly optimized to remove reverberation, the evaluation targets are the corresponding reverberant speaker images. Therefore, the reported results measure separation quality with respect to the reverberant targets rather than dereverberation performance.

## 5. EVALUATION RESULTS

## 5.1. Comparison with Existing Methods

Table 1 presents the results on the 3-channel 2-speaker SMS-WSJ test set. Results marked with † are obtained from previous work or existing reported results and are not re-evaluated in this work. For a controlled comparison, all VM-ArrayDPS configurations in the table use two virtual microphones with $\alpha = 0 . 5$ . The effects of the number of virtual microphones and α are investigated separately in Section 5.2.

VM-ArrayDPS consistently improves over the corresponding ArrayDPS configuration under all three sampling-selection strategies. With AVG selection, VM-ArrayDPS improves SDR from 15.8 to 16.9 dB and SI-SDR from 15.0 to 16.1 dB. PESQ increases from 3.37 to 3.45, while eSTOI increases from 0.864 to 0.881. With Max selection, SDR improves from 17.2 to 17.9 dB and SI-SDR from 16.5 to 17.2 dB. The corresponding PESQ and eSTOI improvements are from 3.52 to 3.56 and from 0.888 to 0.898, respectively. Under ML selection, VM-ArrayDPS improves SDR from 16.9 to 17.7 dB and SI-SDR from 16.2 to 17.0 dB, while PESQ increases from 3.49 to 3.53 and eSTOI from 0.883 to 0.894.

Table 1: Results on 3-channel 2-speaker SMS-WSJ test set. Results marked with † are obtained from existing studies. All VM-ArrayDPS results use two virtual microphones with $\alpha = 0 . 5$
<table><tr><td>Method</td><td>SDR (dB)↑</td><td>SI-SDR (dB)↑</td><td>PESQ↑</td><td>eSTOI↑</td></tr><tr><td>Mixture</td><td>0.1</td><td>0.0</td><td>1.87</td><td>0.603</td></tr><tr><td>Spatial Clustering†</td><td>9.5</td><td>8.5</td><td>2.52</td><td>0.759</td></tr><tr><td>IVA-Laplace†</td><td>12.0</td><td>10.7</td><td>2.67</td><td>0.802</td></tr><tr><td>IVA-Gaussian†</td><td>13.4</td><td>12.2</td><td>2.82</td><td>0.834</td></tr><tr><td>UNSSOR†</td><td>15.4</td><td>14.4</td><td>3.20</td><td>0.875</td></tr><tr><td>ArrayDPS-AVG</td><td>15.8</td><td>15.0</td><td>3.37</td><td>0.864</td></tr><tr><td>VM-ArrayDPS-AVG</td><td>16.9</td><td>16.1</td><td>3.45</td><td>0.881</td></tr><tr><td>ArrayDPS-Max</td><td>17.2</td><td>16.5</td><td>3.52</td><td>0.888</td></tr><tr><td>VM-ArrayDPS-Max</td><td>17.9</td><td>17.2</td><td>3.56</td><td>0.898</td></tr><tr><td>ArrayDPS-ML</td><td>16.9</td><td>16.2</td><td>3.49</td><td>0.883</td></tr><tr><td>VM-ArrayDPS-ML</td><td>17.7</td><td>17.0</td><td>3.53</td><td>0.894</td></tr></table>

Table 2: Results on 4-channel 3-speaker SMS-WSJ test set.
<table><tr><td>Method</td><td>SDR (dB)↑</td><td>SI-SDR (dB)↑</td><td>PESQ↑</td><td>eSTOI↑</td></tr><tr><td>Mixture</td><td>-3.0</td><td>-3.2</td><td>1.57</td><td>0.458</td></tr><tr><td>ArrayDPS-AVG</td><td>11.5</td><td>10.5</td><td>2.82</td><td>0.763</td></tr><tr><td>VM-ArrayDPS-AVG</td><td>13.2</td><td>12.1</td><td>2.95</td><td>0.798</td></tr><tr><td>ArrayDPS-Max</td><td>13.1</td><td>12.2</td><td>3.00</td><td>0.798</td></tr><tr><td>VM-ArrayDPS-Max</td><td>14.1</td><td>13.2</td><td>3.07</td><td>0.822</td></tr><tr><td>ArrayDPS-ML</td><td>12.8</td><td>11.9</td><td>3.00</td><td>0.792</td></tr><tr><td>VM-ArrayDPS-ML</td><td>13.9</td><td>13.0</td><td>3.05</td><td>0.817</td></tr></table>

The consistent gains across AVG, Max, and ML indicate that the benefit of virtual microphones is not specific to a particular sampleselection strategy. Instead, the additional source-oriented observations provide complementary constraints to the DPS process. Among the three configurations, the Max configuration achieves 17.9 dB SDR and 17.2 dB SI-SDR.

For reference, the conventional spatial clustering and IVA baselines provide lower separation performance than the diffusion-based methods. IVA with a Gaussian prior reaches 12.2 dB SI-SDR, while the reported UNSSOR result reaches 14.4 dB SI-SDR. These comparisons provide reference points from both classical spatial separation and unsupervised neural separation, while the direct Array-DPS comparison isolates the contribution of the proposed virtualmicrophone constraints.

We further evaluate VM-ArrayDPS in the 4-channel 3-speaker setting, where the separation problem is more under-determined than the 2-speaker case. Table 2 compares ArrayDPS and VM-ArrayDPS under the same three sampling-selection strategies. With AVG selection, VM-ArrayDPS improves SDR from 11.5 to 13.2 dB and SI-SDR from 10.5 to 12.1 dB, corresponding to gains of 1.7 and 1.6 dB, respectively. PESQ increases from 2.82 to 2.95, while eSTOI increases from 0.763 to 0.798. With Max selection, SDR increases from 13.1 to 14.1 dB and SI-SDR from 12.2 to 13.2 dB. PESQ and eSTOI also improve from 3.00 to 3.07 and from 0.798 to 0.822, respectively. The same trend holds for ML selection. VM-ArrayDPS improves SDR from 12.8 to 13.9 dB and SI-SDR from 11.9 to 13.0 dB, while PESQ increases from 3.00 to 3.05 and eSTOI from 0.792 to 0.817. Thus, the proposed virtual-microphone constraint provides consistent improvements across all three sampling-selection strategies even when the number of sources increases.

Table 3: Ablation study of number of virtual microphones and virtualmicrophone loss weight α on 3-channel 2-speaker SMS-WSJ test set. All results use AVG sample aggregation.
<table><tr><td>#virtual microphones</td><td>α</td><td>SDR (dB)↑</td><td>SI-SDR (dB)↑</td><td>PESQ↑</td><td>eSTOI↑</td></tr><tr><td>0</td><td>-</td><td>15.8</td><td>15.0</td><td>3.37</td><td>0.864</td></tr><tr><td>2</td><td>1.0</td><td>17.0</td><td>16.1</td><td>3.44</td><td>0.883</td></tr><tr><td>2</td><td>0.5</td><td>16.9</td><td>16.1</td><td>3.45</td><td>0.881</td></tr><tr><td>6</td><td>1.0</td><td>17.0</td><td>16.1</td><td>3.43</td><td>0.882</td></tr><tr><td>6</td><td>0.5</td><td>17.0</td><td>16.2</td><td>3.45</td><td>0.881</td></tr><tr><td>6</td><td>0.1</td><td>16.3</td><td>15.5</td><td>3.41</td><td>0.868</td></tr></table>

## 5.2. Ablation Results

We investigate the effects of the number of virtual microphones and the virtual-microphone loss weight α. We use the 3-channel 2- speaker setting with AVG sampling to isolate these factors. The results are summarized in Table 3.

We first examine the number of virtual microphones. Adding 2 virtual microphones substantially improves the ArrayDPS baseline. With $\alpha = 1 . 0 \mathrm { { ; } }$ , SDR increases from 15.8 to 17.0 dB and SI-SDR from 15.0 to 16.1 dB. Increasing the number of virtual microphones from 2 to 6 provides no further improvement: both configurations obtain 17.0 dB SDR and 16.1 dB SI-SDR. A similar trend is observed with $\alpha \ : = \ : 0 . 5$ , where SI-SDR changes only from 16.1 to 16.2 dB. This suggests that a small number of virtual microphones is sufficient, as additional channels derived from the same physical observations may provide increasingly redundant information.

We next examine the effect of α. With two virtual microphones, reducing α from 1.0 to 0.5 keeps SI-SDR at 16.1 dB and slightly improves PESQ from 3.44 to 3.45. With six virtual microphones, α = 0.5 gives a slightly higher SI-SDR of 16.2 dB than 16.1 dB at $\alpha = 1 . 0 $ , with a small improvement in PESQ. Further reducing α to 0.1 causes clear degradation, with SI-SDR dropping to 15.5 dB and eSTOI to 0.868. These results suggest that α = 0.5 provides sufficient weight for the virtual-microphone constraint, while further reducing α weakens its contribution and degrades performance.

Based on these results, we use 2 virtual microphones with α = 0.5 as the default configuration in the main experiments. The three sampling-selection strategies are then evaluated independently, as reported in Table 1 and 2.

## 6. CONCLUSIONS

We have proposed VM-ArrayDPS, a virtual-microphone augmentation method for unsupervised BSS with diffusion posterior sampling. By reusing higher-SNR IVA outputs as virtual microphone observations, the proposed method introduces additional multi-channel consistency constraints and provides cleaner observations for FCPbased acoustic estimation. On both 2- and 3-speaker SMS-WSJ, VM-ArrayDPS consistently improves over ArrayDPS across multiple sampling-selection strategies. Ablation results further shows that a small number of virtual microphones is sufficient and that assigning them a moderate consistency weight is preferable to treating them as perfectly reliable measurements. These results indicate that virtual observations provide a simple and effective way to strengthen the measurement side of diffusion-based BSS without additional supervised training.

## 7. REFERENCES

[1] X.-R. Cao and R.-w. Liu, “General approach to blind source separation,” IEEE Trans. Signal Process., vol. 44, pp. 562–571, 1996.

[2] H. Sawada, N. Ono, H. Kameoka, D. Kitamura et al., “A review of blind source separation methods: Two converging routes to ILRMA originating from ICA and NMF,” Proc. APSIPA, vol. 8, p. e12, 2019.

[3] T. Kim, H. T. Attias, S.-Y. Lee, and T.-W. Lee, “Blind source separation exploiting higher-order frequency dependencies,” IEEE Trans. Audio, Speech, Lang. Process., vol. 15, pp. 70–79, 2007.

[4] T. Kim, T. Eltoft, and T.-W. Lee, “Independent vector analysis: An extension of ICA to multivariate components,” in International conference on independent component analysis and signal separation, 2006, pp. 165–172.

[5] I. Lee, T. Kim, and T.-W. Lee, “Fast fixed-point independent vector analysis algorithms for convolutive blind source separation,” Signal Processing, vol. 87, pp. 1859–1871, 2007.

[6] N. Ono, “Stable and fast update rules for independent vector analysis based on auxiliary function technique,” in Proc. WASPAA, 2011, pp. 189–192.

[7] D. Kitamura, N. Ono, H. Sawada, H. Kameoka et al., “Determined blind source separation unifying independent vector analysis and nonnegative matrix factorization,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 24, pp. 1626–1641, 2016.

[8] Z. Luo, “Independent vector analysis: Model, applications, challenges,” Pattern Recognition, vol. 138, p. 109376, 2023.

[9] D. Wang and J. Chen, “Supervised speech separation based on deep learning: An overview,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 26, pp. 1702–1726, 2018.

[10] Y. Luo and N. Mesgarani, “Conv-TasNet: Surpassing ideal time– frequency magnitude masking for speech separation,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 27, pp. 1256–1266, 2019.

[11] Y. Luo, Z. Chen, and T. Yoshioka, “Dual-path rnn: Efficient long sequence modeling for time-domain single-channel speech separation,” in Proc. ICASSP, 2020, pp. 46–50.

[12] Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee et al., “TF-GridNet: Integrating full- and sub-band modeling for speech separation,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 31, pp. 3221–3236, 2023.

[13] J. R. Hershey, Z. Chen, J. Le Roux, and S. Watanabe, “Deep clustering: Discriminative embeddings for segmentation and separation,” in Proc. ICASSP, 2016, pp. 31–35.

[14] M. Kolbæk, D. Yu, Z.-H. Tan, and J. Jensen, “Multitalker speech separation with utterance-level permutation invariant training of deep recurrent neural networks,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 25, pp. 1901–1913, 2017.

[15] K. Zmol<sup>ˇ</sup> ´ıkova, M. Delcroix, K. Kinoshita, T. Ochiai´ et al., “Speakerbeam: Speaker aware neural network for target speaker extraction in speech mixtures,” IEEE Journal of Selected Topics in Signal Processing, vol. 13, pp. 800–814, 2019.

[16] Z.-Q. Wang, K. Tan, and D. Wang, “Deep learning based phase reconstruction for speaker separation: A trigonometric perspective,” in Proc. ICASSP, 2019, pp. 71–75.

[17] Y. Liu and D. Wang, “Divide and conquer: A deep CASA approach to talker-independent monaural speaker separation,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 27, pp. 2092–2102, 2019.

[18] Z.-Q. Wang and D. Wang, “Combining spectral and spatial features for deep learning based blind speaker separation,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 27, pp. 457–468, 2018.

[19] Y. Luo, Z. Chen, and N. Mesgarani, “Speaker-independent speech separation with deep attractor network,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 26, pp. 787–796, 2018.

[20] D. Yu, M. Kolbæk, Z.-H. Tan, and J. Jensen, “Permutation invariant

training of deep models for speaker-independent multi-talker speech separation,” in Proc. ICASSP, 2017, pp. 241–245.

[21] Z. Chen, Y. Luo, and N. Mesgarani, “Deep attractor network for singlemicrophone speaker separation,” in Proc. ICASSP, 2017, pp. 246–250.

[22] Y. Luo, C. Han, N. Mesgarani, E. Ceolini et al., “FaSNet: Lowlatency adaptive beamforming for multi-microphone audio processing,” in Proc. ASRU, 2019, pp. 260–267.

[23] Z.-Q. Wang, J. Le Roux, and J. R. Hershey, “Multi-channel deep clustering: Discriminative spectral and spatial embeddings for speakerindependent speech separation,” in Proc. ICASSP, 2018, pp. 1–5.

[24] K. Tesch and T. Gerkmann, “Multi-channel speech separation using spatially selective deep non-linear filters,” IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 32, pp. 542–553, 2023.

[25] Y. Luo, Z. Chen, N. Mesgarani, and T. Yoshioka, “End-to-end microphone permutation and number invariant multi-channel speech separation,” in Proc. ICASSP, 2020, pp. 6394–6398.

[26] H. Chen, Y. Yi, D. Feng, and P. Zhang, “Beam-guided TasNet: An iterative speech separation framework with multi-channel output,” arXiv preprint arXiv:2102.02998, 2021.

[27] Z.-Q. Wang and S. Watanabe, “UNSSOR: Unsupervised neural speech separation by leveraging over-determined training mixtures,” Advances in Neural Information Processing Systems, vol. 36, pp. 34 021–34 042, 2023.

[28] S. Wisdom, E. Tzinis, H. Erdogan, R. J. Weiss et al., “Unsupervised speech separation using mixtures of mixtures,” in ICML 2020 Workshop on Self-supervision in Audio and Speech, 2020.

[29] R. Scheibler, Y. Ji, S.-W. Chung, J. Byun et al., “Diffusion-based generative speech source separation,” in Proc. ICASSP, 2023, pp. 1–5.

[30] S. Lutati, E. Nachmani, and L. Wolf, “Separate and diffuse: Using a pretrained diffusion model for improving source separation,” arXiv preprint arXiv:2301.10752, 2023.

[31] H. Chung, J. Kim, M. T. Mccann, M. L. Klasky et al., “Diffusion posterior sampling for general noisy inverse problems,” arXiv preprint arXiv:2209.14687, 2022.

[32] Z. Xu, X. Fan, Z.-Q. Wang, X. Jiang et al., “ArrayDPS: Unsupervised blind speech separation with a diffusion prior,” pp. 69 160–69 188, 2025.

[33] Z.-Q. Wang, G. Wichern, and J. Le Roux, “Convolutive prediction for reverberant speech separation,” in Proc. WASPAA, 2021, pp. 56–60.

[34] ——, “Convolutive prediction for reverberant speech separation,” in Proc. WASPAA, 2021, pp. 56–60.

[35] L. Drude, J. Heitkaemper, C. Boeddeker, and R. Haeb-Umbach, “SMS-WSJ: Database, performance measures, and baseline recipe for multi-channel source separation and recognition,” arXiv preprint arXiv:1910.13934, 2019.

[36] B. Efron, “Tweedie’s formula and selection bias,” Journal of the American Statistical Association, vol. 106, pp. 1602–1614, 2011.

[37] T. Karras, M. Aittala, T. Aila, and S. Laine, “Elucidating the design space of diffusion-based generative models,” Advances in Neural Information Processing Systems, vol. 35, pp. 26 565–26 577, 2022.

[38] E. Vincent, R. Gribonval, and C. Fevotte, “Performance measurement´ in blind audio source separation,” IEEE transactions on audio, speech, and language processing, vol. 14, pp. 1462–1469, 2006.

[39] J. Le Roux, S. Wisdom, H. Erdogan, and J. R. Hershey, “SDR–halfbaked or well done?” in Proc. ICASSP. IEEE, 2019, pp. 626–630.

[40] A. W. Rix, J. G. Beerends, M. P. Hollier, and A. P. Hekstra, “Perceptual evaluation of speech quality (PESQ)-a new method for speech quality assessment of telephone networks and codecs,” in Proc. ICASSP, vol. 2, 2001, pp. 749–752.

[41] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “An algorithm for intelligibility prediction of time–frequency weighted noisy speech,” IEEE Trans. Audio, Speech, Lang. Process., vol. 19, pp. 2125–2136, 2011.