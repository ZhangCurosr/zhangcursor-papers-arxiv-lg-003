# Sub-Model Short-Term Memory Convolutions for Keyword Spotting Systems on Device

Paweł Warlewski<sup>1</sup>, Artur Czeczko<sup>1</sup>, Artur Szumaczuk<sup>1</sup>, Grzegorz Stefanski ´ <sup>2</sup>, Szymon Klimaszewski<sup>1</sup>

<sup>1</sup> Samsung R&D Institute Poland

<sup>2</sup> Samsung AI Center Warsaw

{p.warlewski2, a.czeczko, a.szumaczuk, g.stefanski, s.klimaszews}@samsung.com

## Abstract

Keyword Spotting (KWS) is becoming increasingly important as voice-controlled devices grow more widespread. While voice interaction with smartphones and smart TVs is already common, deploying KWS on heavily resource-constrained edge devices such as wearables remains challenging. These systems must meet high accuracy requirements while operating under strict constraints on computational power, memory footprint, and real-time latency. In this work, we present an application of the STMC (Short-Term Memory Convolutions) framework to adapt a modular CNN model for online, LSTM-like inference. Our approach reduces power consumption and redundant computations while maintaining the stability and simplicity of training CNNs. We achieve up to 82% and 46% MCPS reduction compared to equivalently frequent standard CNN execution and vanilla STMC, respectively. The best configuration achieves 93.8% accuracy on the 11-class Google Speech Commands task and 97.1% on the same task with zero-padded data. Index Terms: keyword spotting, audio classification, convolutional neural network, edge device

## 1. Introduction

Keyword Spotting (KWS) is a well-defined task of detecting a phrase or a set of phrases within the incoming audio stream. The task is becoming increasingly popular thanks to the advent of voice-controlled devices, e.g., TVs, smartphones, etc. The problem becomes progressively more difficult when available resources such as computational power and memory become scarce, as in the case of wearable devices. To tackle the heavily resource-constrained KWS task, the researchers have widely adopted CNNs, which are known for their small memory footprint and proven performance across various tasks [1]. A variety of architectures have been proposed, including ResNetbased models [2], custom strided CNNs [3], MatchboxNet [4], depthwise separable CNNs [5, 6], or even small MLP networks [7]. Recently, the KWS task has also been tackled using transformers [8, 9], but high computational complexity and memory footprints make them inadequate for resource-constrained environments, such as wireless earbuds or smartwatches. Recent TinyML efforts, such as MCUNet [10], demonstrate the feasibility of deep learning on microcontrollers under strict memory constraints.

A practical KWS system must perform inference with high temporal granularity on a continuous audio stream since the target phrases can be very short and may occur at any time in the incoming signal. Ensuring a small memory footprint using a vanilla CNN requires running the model in a sliding-window fashion with a small hop for high granularity. This setting introduces a lot of redundant calculations and may violate the computational power restriction. Recently, the Short-Term Memory Convolutions (STMC) framework [11] was proposed to tackle the issue by buffering intermediate results and reusing them when necessary. A slight increase in the memory footprint allowed the reduction of computational overhead and the transformation of the CNN into an LSTM-like model. Thanks to additional memory, the STMC CNNs can process input in chunks much smaller than the receptive field, which reduces the size of the required input buffer to balance the memory footprint. Since fully sequential input processing is not required for KWS, further optimizations and footprint reductions are possible.

The contributions of this paper are as follows: (1) Propose a small-footprint CNN architecture that achieves good performance on well-established data. (2) Adopt the CNN architecture to the online keyword spotting task via extending and optimizing the STMC scheme to further reduce model footprint while preserving accuracy.

## 2. Method

This section introduces SM-STMC (Sub-Model Short-Term Memory Convolutions), an optimization of STMC that reduces redundant computations in online convolutional networks.

## 2.1. State Redundancy in the Classical Approach

STMC introduces temporal memory buffers for each convolutional and pooling layer. This mechanism caches intermediate activations from previous time steps and reuses them in subsequent computations, enabling online CNN inference with negligible computational overhead. The original motivation for this design was frame-synchronous tasks such as speech enhancement, where producing an output for every incoming frame is both necessary and desirable.

However, when STMC is applied to an architecture containing pooling layers, the buffering strategy leads to additional temporal states whose usefulness depends on the target task. In particular, a max-pooling operation with a stride greater than one increases the number of internal temporal states by a factor equal to the stride. STMC is explicitly designed so that each incoming input frame yields a valid network output, and all states produced by the buffering mechanism therefore correspond to well-defined CNN computations.

For classification-style problems, or other tasks where an output update is not required at every frame, many of these intermediate temporal states become unnecessary. Although they contain valid information and correctly propagate pooled activations over time, they do not provide any additional benefit when predictions are only consumed at a lower temporal resolution. In such settings, maintaining all pooled states introduces redundancy from the perspective of the downstream task.

This behavior is illustrated in Figure 1, where a pooling layer with a stride of 2 produces two alternating temporal states. Both states are required to preserve the correct temporal evolution of features, but only one may be relevant for producing a classification decision at a given evaluation point.

![](images/4027c009dbc0cf41b5f847128a1e59d96a6880bf7d7d799ed17e26d18d4181f2.jpg)  
Figure 1: New states after pooling layer with the stride of2.

More generally, for a network with N pooling layers of stride S, STMC cycles through $S ^ { N }$ internal temporal configurations. All of these states correspond to valid network evaluations and are necessary to support frame-level output generation. Nevertheless, in tasks where outputs are only required sporadically (e.g., sequence-level classification), many of these states do not contribute any additional useful information to the final decision and can be viewed as redundant from a taskdriven perspective.

## 2.2. Sub-model Decomposition and State Scheduling

SM-STMC reduces state redundancy by decomposing the convolutional backbone into a sequence of sub-models, each ending with either a pooling layer or an original output layer. Each sub-model corresponds to an adequate convolutional block and is executed independently.

These sub-models do not share layers or weights. Instead, they are connected through memory buffers, as in the original formulation. This decomposition preserves the original data flow and, if the sub-models are executed consecutively, the original network structure.

Given the architecture of the model, we can schedule the order in which the sub-models are executed. Let N denote the number of pooling layers plus the output layer. At time step $t ,$ SM-STMC selects the deepest sub-model that should be executed at the current iteration. Let $n \in \{ 0 , 1 , \ldots , N - 1 \}$ . The index of the final sub-model executed at iteration t is given by:

$$
n ^ { * } = \operatorname* { m a x } \Bigl \{ n \mid t \bmod 2 ^ { n } = 2 ^ { n } - 1 \Bigr \}\tag{1}
$$

This approach is conceptually related to early-exit methods (e.g., BranchyNet [12]), in which the execution path is dictated by the current internal state.

The right side of Figure 2 illustrates this process for a network with three convolutional blocks.

![](images/66343819c50dab4db15d3c00639e0f99772cc19d955d94bb7c594369d283659f.jpg)  
Figure 2: Difference between STMC (left) and SM-STMC (right).

In standard STMC (left side of Figure 2), all blocks are active at every time step. In SM-STMC (right side), the execution schedule becomes: (1) block 1, (2) block 1 → block 2, (3) block 1, (4) block 1 → block 2 → block 3.

## 2.3. Memory Management and Computational Complexity

SM-STMC uses the same memory management principles as traditional STMC, but excludes memory associated with redundant states. Execution of the sub-model alters only its corresponding memory buffers. With this scheme, memory is allocated and accessed as if only a single temporal state were maintained.

This approach utilizes the whole network only once every $2 ^ { N }$ steps. Intermediate steps execute fractions of the total com putation, substantially reducing the workload. As a result, the average number of executed layers per time step is reduced, leading to a proportional reduction in operations. This reduction scales with network depth, while the latency and alignment remain unchanged.

## 2.4. Training and Deployment

STMC and SM-STMC do not require re-training. The convolutional backbone is trained independently using standard methods. During evaluation, STMC is applied as a deterministic temporal extension of the trained network. This approach does not incorporate any additional weights or parameters.

SM-STMC is implemented using multiple independent sub-models rather than dynamic conditional execution within a single model. This design avoids runtime control flow, which is unsupported by common deployment frameworks such as TensorFlow Lite. As a result, the method ensures compatibility with standard conversion pipelines and enables efficient deployment, provided that memory management and scheduling are implemented on the target device.

## 3. Experiments

This section evaluates the proposed SM-STMC method in terms of recognition performance and computational efficiency. We compare four approaches: (1) a standard VGG-based (Visual Geometry Group) [13] acoustic embedding with an MLP classifier, (2) the same VGG model evaluated eight times per second using a sliding-window scheme, (3) standard STMC with the same MLP classifier, and (4) the proposed SM-STMC with the same classifier.

In all cases, except the first, we evaluated the classifier eight times per second. With the STMC-based approach, we can invoke the classifier after every time bin, but this is not required. Instead, we did it every eight frames, resulting in 7.8125 inferences per second.

All evaluations were conducted on two different models with different numbers of states. The results were also compared with a conventional LSTM network.

## 3.1. Dataset and Experimental Setup

Training and evaluation were conducted on the Google Speech Commands (GSC) [14] dataset, with 10 target classes $( ^ { \mathrm { { s } } \mathrm { { \bar { y } } \mathrm { { e s } } ^ { \mathrm { { \prime } } } } }$ “No”, “Up”, “Down”, “Left”, “Right”, “On”, “Off’, “Stop”, and “Go”), the rest of the classes served as negative class for the contrast. Two evaluation sets were used: (1) the standard GSC test dataset, where each sample was 1 second long, and (2) a modified dataset where 0.5 seconds of silence was added to both the beginning and the end of each sample, resulting in 2 seconds of audio. The purpose of the second dataset is to showcase the limitations of the offline approach. In such cases, the target word may be only partially contained within the input window, resulting in incorrect negative prediction.

The first VGG backbone model has a receptive field of 41 temporal frames, and the second has a receptive field of 38. Input audio is preprocessed into a Mel spectrogram using a window size of 1024 samples and a hop size of 256 samples, resulting in 62.5 temporal bins per second. Frames corresponding to one second of audio are then fed as an input to the model. In two seconds, this would result in two full input buffers.

The VGG backbone was trained offline using a standard supervised training scheme. STMC and SM-STMC are inferencetime techniques that require no additional retraining, so the original set of weights was applied. All approaches use the same MLP classifier as well.

To demonstrate the effectiveness of this method in an online setting, we evaluate the STMC classifiers eight times per second. To match the setup for the vanilla VGG model, we evaluated it using a sliding-window scheme at the same temporal rate.

Since STMC and SM-STMC achieve identical recognition performance, we report results for only one of them.

The on-device experiments were run on the ARM Cortex-M55 CPU at 196 MHz. Its crucial feature is M-Profile Vector Extension (MVE), also called ARM Helium. MVE uses 128- bit registers, allowing up to 16 parallel operations in int8 data, the type our models are quantized to using standard integeronly inference techniques [15]. To further reduce memory consumption, TensorFlow Lite Micro [16] has been chosen as the deployment framework, optimizing MCU inferences using libraries such as CMSIS-NN [17].

## 3.2. Parameters

Table 1 summarizes the number of parameters for each model configuration. Embedding consists of multiple convolutional blocks, and the parameters for each individual block are reported. STMC-based methods use additional memory buffers with a size depending on the kernel, filters, and mels. In the standard STMC, these values are also multiplied by the number of states. The size of these temporal memory buffers is reported in Table 2. STMC<sup>1</sup> has a total of 4 states, while $\mathrm { S T M C } ^ { 2 }$ has a total of 8 states. Superscripts 1 and 2 denote the first and second backbone configurations, respectively.

Table 1: Parameters ofthe backbone’s embedding and classifier.
<table><tr><td>Model</td><td>Emb</td><td>Cls</td><td>Total</td></tr><tr><td>SM-STMC1</td><td> $\overline { { 5 1 3 + 1 7 7 6 + 6 4 9 6 } }$ </td><td>18443</td><td>27228</td></tr><tr><td>SM-STMC²</td><td> $1 0 5 + 8 1 6 + 1 4 7 2 + 1 2 0 6 4$ </td><td>18443</td><td>32900</td></tr><tr><td>LSTM</td><td>20992</td><td>6155</td><td>27147</td></tr></table>

Table 2: Buffer size (int8).
<table><tr><td>Model</td><td>Total buffer size</td></tr><tr><td>STMC1</td><td>17184</td></tr><tr><td> $\mathbf { S M - S T M C } ^ { 1 }$ </td><td>9280</td></tr><tr><td>STMC2</td><td>21632</td></tr><tr><td> $\mathbf { S M - S T M C ^ { 2 } }$ </td><td>7520</td></tr></table>

## 3.3. Evaluation Metrics

We report weighted average precision, recall, and F1-score. In the single-label multi-class setting considered here, the weighted average recall is equivalent to classification accuracy. Due to space constraints, only average metrics are reported. In all of the streaming models, we classify embeddings every 8th frame, which corresponds to the minimum sequence of frames for $\mathbf { S M - S T M C } ^ { 2 }$ . For consistency, the classifiers of the remaining models are evaluated at the same temporal intervals, resulting in 7.8125 evaluations per second.

Table 3: Weighted average performance comparison on the standard 1 s GSC dataset.
<table><tr><td>Model</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td> $\overline { { \mathrm { V G G } ^ { \mathrm { I } } \left( 1 \times 1 \mathrm { s } \right) } }$ </td><td>0.9362</td><td>0.9278</td><td>0.9292</td></tr><tr><td>SM-STMC1</td><td>0.9494</td><td>0.9382</td><td>0.9403</td></tr><tr><td> $\overline { { \mathbf { V } \mathbf { G } \mathbf { G } ^ { 2 } \left( 1 \times 1 \mathrm { ~ s } \right) } }$ </td><td>0.9159</td><td>0.9035</td><td>0.9055</td></tr><tr><td> $\mathbf { S } \mathbf { M } { - } \mathbf { S } \mathbf { T } \mathbf { M } \mathbf { C } ^ { 2 }$ </td><td>0.9322</td><td>0.9162</td><td>0.9188</td></tr><tr><td>LSTM</td><td>0.9569</td><td>0.9511</td><td>0.9521</td></tr></table>

Table 4: Weighted average performance comparison on the extended 2 s GSC dataset with silence added.
<table><tr><td>Model</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td> $\overline { { \mathrm { V G G } ^ { \mathrm { I } } \left( 1 \times 1 \mathrm { s } \right) } }$ </td><td>0.8353</td><td>0.4981</td><td>0.5798</td></tr><tr><td> $\nabla \mathrm { G G } ^ { 1 } \left( 8 \times 1 \mathrm { s } \right)$ </td><td>0.9733</td><td>0.9708</td><td>0.9712</td></tr><tr><td>SM-STMC1</td><td>0.9738</td><td>0.9710</td><td>0.9715</td></tr><tr><td> $\overline { { \mathrm { V G G } ^ { 2 } \left( 1 \times 1 \mathrm { s } \right) } }$ </td><td>0.7790</td><td>0.6182</td><td>0.6386</td></tr><tr><td> $\nabla \mathrm { G G } ^ { 2 } \left( 8 \times 1 \mathrm { s } \right)$ </td><td>0.9586</td><td>0.9524</td><td>0.9532</td></tr><tr><td> $\mathbf { S } \mathbf { M } { - } \mathbf { S } \mathbf { T } \mathbf { M } \mathbf { C } ^ { 2 }$ </td><td>0.9611</td><td>0.9552</td><td>0.9561</td></tr><tr><td>LSTM</td><td>0.9396</td><td>0.9196</td><td>0.9235</td></tr></table>

The VGG models achieve accuracy comparable to the STMC-based approaches. However, to obtain identical performance, they would need to be evaluated at every time bin.

## 3.4. Results on Standard GSC

Results on the standard GSC dataset are presented in Table 3. The standard VGG model achieves overall lower performance than STMC, with a drop of 1.04%<sup>1</sup> in accuracy. This result is expected, since the classifier is invoked more frequently in the online approach.

## 3.5. Results on Extended Inputs

Table 4 reports results on the 2-second dataset with silence padding. This evaluation highlights the limitations of offline inference. The standard VGG model failed to correctly classify more than half of the samples. This occurs when a keyword spans the boundary between two input audio windows (e.g. “do-” in the first window and “-wn” in the second), leading to misclassification.

Figure 3 illustrates this boundary effect using a zero-padded example evaluated under different temporal shifts. When the keyword is centered within the input frame (temporal shift of 0.0 s), both architectures produce correct predictions. However, when the keyword lies across two frames (shift -0.5 s or 0.5 s), the VGG model’s accuracy drops substantially, whereas the STMC-based model remains stable.

The VGG model (evaluated eight times per second) and the SM-STMC-based models achieve significantly better results, with an accuracy improvement of $3 . 2 8 \% ^ { 1 }$ compared to the 1- second setting. Improvement comes from the fact that the keyword is captured across multiple overlapping windows, increasing the likelihood that at least one window contains a more favorable representation for classification.

![](images/1393d19ac3107e5958b9bb0724ecb2ad9534d5eb013e9c9c6ed00c1755b9d83b.jpg)  
Figure 3: Influence of temporal shift on accuracy. VGG and STMC accuracy on a 2-second zero-padded $" s t o p "$ sample.

## 3.6. Computational Cost and Latency

Computational complexity is measured in million cycles per second (MCPS) and is reported separately for embedding computation, classifier evaluation, memory management, and total cost, reported in Table 5. For SM-STMC, the embedding stage consists of multiple independent sub-models corresponding to different convolutional blocks.

In the case of VGG, evaluating the model eight times per second consumes proportionally more MCPS. Despite the fact that this approach achieved approximately the same results as STMC on 2 seconds of audio, it required almost four times more computational cycles for the first backbone model and three times more for the second.

Compared to the standard VGG baseline, STMC increases computational cost by a factor of 2.27<sup>1</sup> for the first model, while providing fully online inference. SM-STMC reduces embedding computation and memory management overhead by removing redundant states, decreasing this factor to only $1 . 5 3 ^ { 1 }$ For the second model, this factor is respectively $2 . 7 8 ^ { 2 }$ and 1.47<sup>2</sup>.

Table 5: Computational cost comparison ofembedding, classifier, and memory management in MCPS.
<table><tr><td>Model</td><td>Emb</td><td>Cls</td><td>Memory</td><td>Total</td></tr><tr><td> $\overline { { \mathrm { V G G } ^ { \mathrm { I } } \left( 1 \times 1 \mathrm { s } \right) } }$   $\nabla \mathrm { G G } ^ { 1 } \left( 8 \times 1 \mathrm { s } \right)$  STMC¹</td><td>7.40 59.20 16.12</td><td>0.02 0.16 0.16</td><td>一 0.56</td><td>7.42 59.36 16.84</td></tr><tr><td> $\mathbf { S M - S T M C } ^ { 1 }$   $\overline { { \mathrm { V G G } ^ { 2 } \left( 1 \times 1 \mathrm { s } \right) } }$   $\nabla \mathrm { G G } ^ { 2 } \left( 8 \times 1 \mathrm { s } \right)$ </td><td>10.82 3.66 29.28</td><td>0.16 0.02 0.16</td><td>0.39 一</td><td>11.37 3.68 29.44</td></tr><tr><td> $\mathrm { S T M C } ^ { 2 }$   $\mathbf { S } \mathbf { M } { - } \mathbf { S } \mathbf { T } \mathbf { M } \mathbf { C } ^ { 2 }$  LSTM</td><td>9.42 5.12 10.29</td><td>0.16 0.16 0.05</td><td>0.44 0.16 0.01</td><td>10.02 5.44 10.35</td></tr></table>

Total detection latency consists of computational delay and the temporal alignment of the keyword within the input buffer. In a streaming context, predictions are generated every 16 ms. To account for the stochastic nature of keyword onset, we report the average latency as the midpoint between the best-case (keyword ends exactly at a buffer boundary) and worst-case (keyword ends just after a boundary) scenarios, plus the specific computational overhead of the model. Latency is measured according to each model’s native classification rate, e.g., LSTM every frame, while SM-STMC<sup>1</sup> every 4th frame.

For the baseline VGG models, classification occurs once per second, leading to a high average latency of approximately 550–600 ms. In contrast, STMC and LSTM models provide frame-synchronous predictions (16 ms granularity), resulting in early detection with ”negative” reported latency relative to the keyword offset. This is a consequence of the models capturing the discriminative features of the keyword before the entire keyword has been observed.

Table 6: Computational latency and average latency of the models measuredfrom the end ofthe uttered keyword to itsfirst corresponding prediction.
<table><tr><td>Model</td><td>Comp. Latency [ms]</td><td>Avg Latency [ms]</td></tr><tr><td>VGG{</td><td>101</td><td> $\overline { { 6 0 1 \pm 5 0 0 } }$ </td></tr><tr><td>STMC1</td><td>4</td><td> $- 1 5 6 \pm 8$ </td></tr><tr><td>SM-STMC1</td><td>10</td><td> $- 1 4 6 \pm 3 2$ </td></tr><tr><td> $\overline { { \mathrm { ~ V G G } ^ { 2 } } }$ </td><td>50</td><td> $\overline { { 5 5 0 \pm 5 0 0 } }$ </td></tr><tr><td> $\mathrm { S T M C } ^ { 2 }$ </td><td>2</td><td> $- 1 5 8 \pm 8$ </td></tr><tr><td> $\mathbf { S M - S T M C ^ { 2 } }$ </td><td>9</td><td> $- 1 1 6 \pm 6 4$ </td></tr><tr><td>LSTM</td><td>5</td><td> $- 1 6 7 \pm 8$ </td></tr></table>

The SM-STMC approach introduces a controlled tradeoff: by classifying every 4th frame (SM-STMC<sup>1</sup>) or every 8th frame (SM- $\cdot \mathrm { { \cal S T M C } } ^ { 2 } ) .$ , the average latency increases compared to vanilla STMC able to classify data every frame, but remains within the margin of classification time. Furthermore, when comparing the time required to process an equivalent duration of audio (computationally-wise), $\mathbf { S } \mathbf { M } \mathbf { - } \mathbf { S } \mathbf { T } \mathbf { M } \mathbf { C } ^ { \hat { 2 } }$ is 1.78 and 4.44 times more efficient than STMC and LSTM, respectively.

## 4. Conclusion

In this work, we proposed SM-STMC, a sub-model decomposition and scheduling strategy that reduces redundant temporal states in online convolutional networks. The method extends the original STMC framework by exploiting the task-level requirement of sparse output generation, thereby eliminating unnecessary internal states by using model decomposition and executing only the required network depth with static scheduling.

Experimental results on the Google Speech Commands dataset demonstrate that SM-STMC preserves the high recognition performance of CNNs while achieving substantial efficiency gains. Specifically, we observed an MCPS reduction of up to 82% compared to the standard sliding-window baseline and a memory footprint reduction of 65% compared to the original STMC approach. While the reduced classification frequency introduces a marginal increase in latency, the resulting values remain well within the requirements for real-time human-computer interaction.

Importantly, SM-STMC requires no retraining and introduces no additional parameters, making it fully compatible with existing pretrained models. The use of static scheduling and independent sub-models ensures that the system can be deployed directly via standard frameworks like TensorFlow Lite on resource-constrained edge devices, such as wearables and hearables.

## 5. Generative AI Use Disclosure

LLM (Large Language Model) has been used to polish and to find grammar errors in this manuscript. However, it was not used to directly write it or to generate code used in this work.

## 6. References

[1] Y. Zhang, N. Suda, L. Lai, and V. Chandra, “Hello edge: Keyword spotting on microcontrollers,” 2018. [Online]. Available: https://arxiv.org/abs/1711.07128

[2] R. Tang and J. J. Lin, “Deep residual learning for small-footprint keyword spotting,” in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, p. 5484–5488.

[3] T. N. Sainath and C. Parada, “Convolutional neural networks for small-footprint keyword spotting,” in Proc. Interspeech, 2015, pp. 1478–1482.

[4] S. Majumdar and B. Ginsburg, “Matchboxnet: 1d time-channel separable convolutional neural network architecture for speech commands recognition,” p. 3356–3360, Oct. 2020. [Online]. Available: http://dx.doi.org/10.21437/Interspeech.2020-1058

[5] M. Xu and X.-L. Zhang, “Depthwise separable convolutional resnet with squeeze-and-excitation blocks for small-footprint keyword spotting,” p. 2547–2551, Oct. 2020. [Online]. Available: http://dx.doi.org/10.21437/Interspeech.2020-1045

[6] P. Bartoli, T. Bondini, C. Veronesi, A. Giudici, N. Antonello, and F. Zappa, “End-to-end efficiency in keyword spotting: A system-level approach for embedded microcontrollers,” p. 1–4, Oct. 2025. [Online]. Available: http://dx.doi.org/10.1109/ SENSORS59705.2025.11330257

[7] C. P. G. Chen and G. Heigold, “Small-footprint keyword spotting using deep neural networks,” in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2014, pp. 4087–4091.

[8] Y.-A. C. Y. Gong and J. Glass, “Ast: Audio spectrogram transformer,” in Interspeech, 2021, p. 571–575.

[9] M. O. A. Berg and M. T. Cruz, “Keyword transformer: A self attention model for keyword spotting,” in Interspeech, 2021, p. 4249–4253.

[10] J. Lin, W.-M. Chen, Y. Lin, J. Cohn, C. Gan, and S. Han, “Mcunet: Tiny deep learning on iot devices,” 2020. [Online]. Available: https://arxiv.org/abs/2007.10319

[11] G. Stefanski, K. Arendt, P. Daniluk, B. Jasik, and A. Szumaczuk,´ “Short-term memory convolutions,” 2023. [Online]. Available: https://arxiv.org/abs/2302.04331

[12] S. Teerapittayanon, B. McDanel, and H. T. Kung, “Branchynet: Fast inference via early exiting from deep neural networks,” 2017. [Online]. Available: https://arxiv.org/abs/1709.01686

[13] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” 2015. [Online]. Available: https://arxiv.org/abs/1409.1556

[14] P. Warden, “Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition,” ArXiv e-prints, Apr. 2018. [Online]. Available: https://arxiv.org/abs/1804.03209

[15] B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko, “Quantization and training of neural networks for efficient integer-arithmetic-only inference,” 2017. [Online]. Available: https://arxiv.org/abs/1712.05877

[16] R. David, J. Duke, A. Jain, V. J. Reddi, N. Jeffries, J. Li, N. Kreeger, I. Nappier, M. Natraj, S. Regev, R. Rhodes, T. Wang, and P. Warden, “Tensorflow lite micro: Embedded machine learning on tinyml systems,” 2021. [Online]. Available: https://arxiv.org/abs/2010.08678

[17] L. Lai, N. Suda, and V. Chandra, “Cmsis-nn: Efficient neural network kernels for arm cortex-m cpus,” 2018. [Online]. Available: https://arxiv.org/abs/1801.06601