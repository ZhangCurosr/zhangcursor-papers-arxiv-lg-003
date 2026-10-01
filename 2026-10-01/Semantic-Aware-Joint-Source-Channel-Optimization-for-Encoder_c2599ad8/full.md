# Semantic-Aware Joint Source-Channel Optimization for Encoder-Agnostic Digital Video Communication

Xiangben Zhu, Graduate Student Member, IEEE, Caili Guo, Senior Member, IEEE, Yang Yang, Senior Member, IEEE, Chuanhong Liu, and Meiyi Zhu

Abstract—Video semantic communication has attracted increasing attention as a promising approach to improving video transmission efficiency. However, most existing approaches rely on computationally intensive deep learning-based video encoders and decoders, which hinders their deployment in resourceconstrained scenarios. To address this issue, we propose a lightweight semantic-aware joint source-channel optimization (SAJSCO) scheme that can be integrated into existing digital video communication systems as a plug-in module. Specifically, we develop a video communication system model in which the transmitter jointly optimizes source and channel coding parameters based on the inter-frame semantic importance of the input video and estimated channel state information. On this basis, we formulate an optimization problem that maximizes semantic importance weighted video reconstruction quality under a maximum bitrate constraint. To solve it, we first quantify inter-frame semantic importance using a cosine similarity-based metric with a shifted window mechanism. We then develop a multi-actor proximal policy optimization (MPPO) algorithm to solve the formulated problem by jointly adapting the source compression rate and channel coding rate. The learned policy can be directly applied to different video encoders without encoderspecific retraining or fine-tuning. SAJSCO achieves Bjøntegaard Delta rate reductions of 34.86% and 18.01% when integrated with H.265, a conventional video encoder, and DCVC-RT, a deep learning-based video encoder. Over-the-air experiments on a hardware testbed further demonstrate a PSNR gain of up to 1.448 dB with H.265 and an LPIPS reduction of up to 0.033 with DCVC-RT compared with the respective best-performing fixed-parameter baselines.

Index Terms—Video semantic communication, temporal video semantics, deep reinforcement learning, prototype validation.

## I. INTRODUCTION

The deep integration of artificial intelligence (AI) and 6G communication is becoming an inevitable trend. The proliferation of multimedia devices has led to a surge in video traffic, while deep neural networks for video encoding require substantial computational resources, posing significant challenges to 6G wireless communication systems [2]. However, conventional video communication systems based on bit-level transmission and codecs such as H.265 [3] still suffer from limited compression efficiency and are vulnerable to channel impairments, where errors in critical control bits may cause decoding failures [4]. Video semantic communication (VSC), leveraging artificial intelligence technologies, enhances coding efficiency and improves communication robustness, thereby making it a key enabling technology for 6G networks [5].

Existing video semantic communication methods have demonstrated considerable success in reducing transmission bandwidth while maintaining perceptual quality. A representative line of research leverages deep neural networks to jointly optimize source and channel coding in an endto-end manner [6]–[8]. However, such approaches impose heavy computational burdens on communication systems and often lack explicit adaptation to time-varying channel conditions, making practical deployment challenging. Another line of work incorporates semantic awareness into conventional codec-based pipelines [9], offering greater compatibility with existing infrastructures, yet most of these methods focus on spatial semantics such as regions of interest [10], [11]. The temporal semantics inherent in video content, which capture motion patterns and inter-frame dependencies are pivotal for inter-frame prediction and redundancy elimination [12], [13], remain largely underexplored. Motivated by these observations, this work proposes a low-complexity video semantic communication scheme that exploits temporal semantic importance to guide the joint optimization of source and channel coding, while maintaining full compatibility with existing digital communication systems.

## A. Related Work

Existing approaches fall into two broad categories, namely deep neural network-based methods and conventional codecbased methods, which are detailed in the following.

1) Deep video semantic communication: Deep video semantic communication employs deep neural networks as semantic encoders and decoders, encoding videos into compact semantic features to reduce transmission bandwidth [6]. The use of deep learning-based encoders naturally offers advantages in security, as demonstrated in [14], where adversarial residual networks are employed to enhance the privacy of semantic communication. To achieve intelligent task-oriented communication, the authors in [15], [16] developed semantic communication systems that focus on transmitting only task-relevant semantic information, thereby reducing bandwidth consumption. Furthermore, the authors in [17] introduced a lightweight task-oriented semantic communication framework empowered by large-scale AI models, where knowledge distillation is leveraged to reduce model complexity and computational latency. The authors in [18] utilized artificial intelligence generated content (AIGC) to transmit multi-modal semantics and generate videos at the receiver. However, most deep learningbased video encoders introduce considerable computational overhead, rendering real-time encoding of 1080p video at 25 fps unattainable even with state-of-the-art GPU acceleration, posing a fundamental barrier to practical deployment [19].

2) Semantic-driven video communication: Semantic-driven video communication regards semantic information as prior knowledge to optimize existing video communication frameworks. The authors in [20] allocated communication resources based on semantic importance, thereby improving task performance. The authors in [10] separated dynamic and static contents of videos, and encoded their semantics and locations. Spatial semantic importance has been incorporated into ROIbased video coding and resource allocation to prioritize highquality transmission of semantically important regions [9], [11]. However, existing ROI-based semantic communication methods mainly focus on intra-frame semantic importance while ignoring inter-frame semantics, leading to residual temporal redundancy. Deep reinforcement learning has been explored for video communication by modeling adaptive coding and streaming optimization as sequential decisionmaking problems, enabling dynamic parameter adjustment under varying video content and network conditions [21]–[23]. Temporal redundancy across video frames can be exploited to reduce redundant bits and improve bandwidth efficiency, while temporally important semantic content should be assigned more transmission resources for reliable reconstruction [13]. Therefore, the exploitation of temporal semantics for video reconstruction remains insufficiently explored.

## B. Challenges and Contributions

Deep-learning based semantic encoders impose substantial computational overhead. Recently, a growing number of studies [24], [25] have investigated the use of generative large models for video semantic communication, in which only textlevel prompts and key frames are transmitted. However, this paradigm entails substantial decoding complexity and imposes considerable computational demands at the receiver. Although considerable efforts have been devoted to lightweight model design [26], their AI computing requirements remain high, making them difficult to deploy in existing communication systems. Fortunately, this issue can be alleviated by integrating semantic communication with existing communication systems, where the video is first semantically understood and then encoded using conventional video encoders.

In addition, video is a form of spatiotemporal data characterized by temporal evolution and two-dimensional spatial pixels, and its semantic importance is inherently nonuniform. Such semantic importance can provide valuable guidance for video encoding. The study in [5], [9] investigated the semantic importance of video in the spatial dimension and allocated communication resources to different regions according to their semantic importance, thereby improving coding efficiency. To more effectively compress redundant video information and ensure robust transmission, it is essential to investigate the semantic importance of video in the temporal dimension. Moreover, due to the fluctuations of the signal-to-noise ratio (SNR) in real-world video transmission scenarios, it is necessary to jointly consider video semantics and channel state information to design a robust and efficient video transmission mechanism. The work in [27] exploits instantaneous channel state information (CSI) to adaptively rescale image semantic features, thereby enabling implicit joint source–channel rate adaptation. In the study of performance optimization for semantic-aware video transmission, several challenges must be addressed:

Challenge 1: How to enable real-time video semantic communication under the computational constraints of existing communication infrastructure?

Challenge 2: Under computational constraints, how to effectively measure the temporal, or inter-frame, semantic importance of video?

Challenge 3: How to leverage the measured temporal semantic importance together with real-time CSI to achieve effective joint source-channel optimization?

In this paper, we first develop a semantic-aware joint sourcechannel optimization (SAJSCO) scheme for video transmission. We then propose a novel method for measuring semantic importance and devise a deep reinforcement learning-based algorithm to determine the optimal source coding and channel coding parameters. A preliminary version of the work on video source coding optimization was introduced in our conference paper. To the best of our knowledge, this is the first work to investigate video coding parameter optimization from the perspective of inter-frame semantic importance. The main contributions of this paper are summarized as follows:

1) We propose a joint source–channel optimization framework that employs lightweight video semantic extraction and analysis. This framework has low computational requirements and can seamlessly integrate with existing digital video communication systems, offering good robustness against fluctuating channel conditions. Considering the varying inter-frame semantic importance of the video, we formulate an optimization problem for video rate-distortion, with the objective of optimizing the semantic importance weighted video reconstruction quality under the constraint of maximum bitrate. This addresses the aforementioned Challenge 1.

2) To solve the problem, We propose an inter-frame semantic importance evaluation method that models temporal correlations based on cosine similarity in the semantic feature domain. By incorporating a shifted window mechanism, the method captures local inter-GOP dependencies and aggregates them progressively, yielding a finegrained importance representation that effectively guides parameter selection. This addresses the aforementioned Challenge 2.

3) Then, we propose a deep reinforcement learning-based parameter selection algorithm. Specifically, the algorithm comprehensively considers both the inter-frame semantic importance of the video and real-time CSI feedback, achieving a balance between efficient video compression and reliable transmission. Furthermore, a multi-actor reinforcement learning algorithm is designed to separately select the compression rate for source encoding and the coding rate for channel encoding. The multi-actor approach enables the network to learn the characteristics and strategies for both source and channel encoding parameters, thereby enhancing overall performance optimization. This effectively addresses the aforementioned Challenge 3.

![](images/8bde40fffe74b58146e3a221103569f6d408b4e15745ce31c0395119c9f7c90e.jpg)  
Fig. 1. System model of semantic-aware joint source–channel optimization for video communication

4) Experimental results demonstrate that the proposed deep reinforcement learning-based scheme outperforms both traditional and AI-based video communication methods. Additionally, the proposed parameter selection algorithm yields substantial improvements in video quality, achieving better results under both fixed Signal-to-Noise Ratio (SNR) and actual fluctuating SNR conditions.

The remainder of this paper is organized as follows. Sec. II introduces the system model and problem formulation. Sec. III introduces the proposed video inter-frame semantic importance measurement method. The proposed deep reinforcement learning algorithm is detailed in Sec. IV. Sec. V presents the analysis of simulation results. Sec. VI presents the hardware prototype implementation and experimental validation conducted on a physical platform to verify the practical feasibility of the proposed system. Finally, Sec. VII summarizes the key conclusions of this study.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

## A. System model

As shown in Fig. 1, we consider an end-to-end (E2E) video semantic communication system, where the video encoder can be either a conventional codec or an AI-based encoder. At the transmitter, a source video is processed in four steps: (i) a feature extraction module extracts the semantic features of the video; (ii) a semantic importance module evaluates the inter-frame semantic importance of different groups of pictures (GOPs) from these features; (iii) based on the semantic importance and the channel state information (CSI), a reinforcement learning algorithm selects the source and channel coding parameters for each GOP; and (iv) the video encoder and the channel encoder encode the video with the selected parameters. The encoded bitstream is then transmitted over a wireless channel. At the receiver, a channel decoder recovers the bitstream, and a video decoder matched to the video encoder reconstructs the video.

At the semantic transmitter, the input video frames are $\pmb { X } \in \mathbb { R } ^ { L \times C \times D _ { \mathbf { w } } \times D _ { \mathbf { h } } }$ , where L is the length of the video, C is the number of channels, and $D _ { \mathrm { w } }$ and $D _ { \mathrm { h } }$ are the width and height of the video frames. The video is divided into N GOPs, where $L = N \times K$ , and $\pmb { X } = ( X _ { 1 } , X _ { 2 } , \ldots , X _ { N } )$ where each GOP $X _ { i }$ contains K video frames. A pre-trained feature extraction network $f _ { \omega } ( \cdot )$ extracts the semantic features $F = ( F _ { 1 } , F _ { 2 } , \dots , F _ { N } )$ of the video. Specifically, the i-th feature vector ${ \mathbf { } } F _ { i }$ is obtained as

$$
F _ { i } = f _ { \omega } ( X _ { i } ) ,\tag{1}
$$

where $\omega$ denotes the parameters of the pre-trained network. Taking the semantic features $\pmb { F }$ as input, a semantic importance model $S ( \cdot )$ produces the semantic importance of all GOPs as

$$
W = S ( { \boldsymbol { F } } ) ,\tag{2}
$$

where $\pmb { W } = ( W _ { 1 } , W _ { 2 } , \dots , W _ { N } )$ and $W _ { i }$ denotes the semantic importance of GOP $X _ { i }$ . The model $S ( \cdot )$ measures the interframe semantic importance of the video and is detailed in Sec. III. The semantic importance and the channel condition jointly determine how each GOP should be coded: A more important GOP or a worse channel calls for stronger protection, whereas a less important GOP or a better channel allows for a lower bit expenditure. Accordingly, each GOP $X _ { i }$ is assigned a pair of coding parameters $P _ { i } = ( P _ { i } ^ { S } , P _ { i } ^ { C } )$ , where the source coding parameter $P _ { i } ^ { S }$ controls the compression level and the channel coding parameter $P _ { i } ^ { C }$ controls the degree of error protection. The parameters of all GOPs form the set ${ \cal P } = \bar { \{ P _ { i } \} } _ { i = 1 } ^ { N } ,$ which is determined by a reinforcement learning algorithm $G _ { \pi } ( \cdot )$ with policy π from the semantic importance W and the CSI C as

$$
P = G _ { \pi } ( W , C ) ,\tag{3}
$$

where $C = ( C _ { 1 } , C _ { 2 } , \dots , C _ { N } )$ and $C _ { i }$ denotes the channel state information (CSI) of the i-th timeslot, obtained by transmitting pilot signals at the transmitter and estimating them at the receiver. The design of $G _ { \pi } ( \cdot )$ is detailed in Sec. IV.

With the selected parameters, the GOPs are encoded one by one. Specifically, the video encoder $E _ { S }$ first encodes GOP $X _ { i }$ into a source-coded stream as

$$
X _ { s , i } = E _ { S } ( X _ { i } , P _ { i } ^ { S } ) ,\tag{4}
$$

where the video encoder can be instantiated as a traditional H.26X codec or a deep neural network-based video encoder. This encoder-agnostic design allows the proposed method to be applied to different video encoders as a plug-in module. The channel encoder $E _ { C }$ then encodes the source-coded stream as

$$
\hat { X } _ { i } = E _ { C } ( X _ { s , i } , P _ { i } ^ { C } ) ,\tag{5}
$$

The encoded bitstream $X _ { i }$ is sent over the wireless channel, which is modeled as

$$
Y _ { i } = H \hat { X _ { i } } + n ,\tag{6}
$$

where H represents the wireless channel response, and n denotes the independent and identically distributed complex Gaussian noise following $\mathcal { C N } ( 0 , \sigma ^ { 2 } )$

To obtain real-time CSI, pilot symbols $\mathbf { \boldsymbol { x } } _ { p }$ known to both the transmitter and receiver are transmitted, as commonly adopted in pilot-assisted channel estimation [28]. For SNR estimation, the wireless channel H is represented by an equivalent complex channel coefficient h. The received pilot signal $\pmb { y } _ { p }$ can be expressed as

$$
\begin{array} { r } { { \pmb y } _ { p } = h { \pmb x } _ { p } + { \pmb n } . } \end{array}\tag{7}
$$

Based on the known pilot symbols, the equivalent channel coefficient is estimated using the least-squares (LS) method as

$$
\hat { h } = \frac { { \pmb x } _ { p } ^ { H } { \pmb y } _ { p } } { { \pmb x } _ { p } ^ { H } { \pmb x } _ { p } } .\tag{8}
$$

Based on the estimated channel coefficient, the residual noise is obtained as

$$
\hat { \pmb { n } } = \pmb { y } _ { p } - \hat { h } \pmb { x } _ { p } .\tag{9}
$$

Then, the SNR is estimated as

$$
\widehat { \mathrm { S N R } } = \frac { \mathbb { E } \left[ \left. \hat { h } \pmb { x } _ { p } \right. ^ { 2 } \right] } { \mathbb { E } \left[ \left. \pmb { y } _ { p } - \hat { h } \pmb { x } _ { p } \right. ^ { 2 } \right] } .\tag{10}
$$

CSI estimation is performed once for each GOP. Accordingly, the estimated SNR is used to obtain $C _ { i }$ , which represents the CSI associated with the i-th timeslot.

Next, the received bitstream is processed by the channel decoder and the video decoder to reconstruct the video, which can be expressed as

$$
X _ { i } ^ { \prime } = D _ { S } \left( D _ { C } ( Y _ { i } ) \right) ,\tag{11}
$$

where $D _ { C } ( \cdot )$ and $D _ { S } ( \cdot )$ denote the channel decoder and the video decoder, respectively. Then we evaluate quality of the reconstructed video. For the i-th GOP, the mean square error (MSE) is defined as

$$
\mathrm { M S E } _ { i } = \frac { 1 } { K D _ { \mathrm { h } } D _ { \mathrm { w } } } \sum _ { k = 1 } ^ { K } \sum _ { u = 1 } ^ { D _ { \mathrm { h } } } \sum _ { v = 1 } ^ { D _ { \mathrm { w } } } \left( x _ { i , k } ( u , v ) - x _ { i , k } ^ { \prime } ( u , v ) \right) ^ { 2 } ,\tag{12}
$$

where $x _ { i , k } ( m , n )$ and $x _ { i , k } ^ { \prime } ( m , n )$ denote the pixel values at location $( m , n )$ in the k-th frame of $X _ { i }$ and $\mathbf { } X _ { i } ^ { \prime } ,$ , respectively. Accordingly, the Peak Signal-to-Noise Ratio (PSNR) of the i-th GOP is given by

$$
Q _ { i } = 1 0 \log _ { 1 0 } \left( \frac { V _ { \mathrm { m a x } } ^ { 2 } } { \mathrm { M S E } _ { i } } \right) ,\tag{13}
$$

where $V _ { \mathrm { m a x } }$ denotes the maximum possible pixel value.

Different GOPs contribute unequally to the overall semantic perception of the video, whereas conventional distortion metrics such as PSNR and SSIM treat all GOPs equally and therefore cannot adequately characterize their semantic significance. Inspired by [5], [9], [29], we therefore adopt the semantic importance weighted peak signal-to-noise ratio (WPSNR) as the semantic-aware video reconstruction quality metric. Specifically, we assign a semantic weight to each GOP and define the weighted quality metric as

$$
Q _ { i } ^ { w } = W _ { i } \cdot Q _ { i } ,\tag{14}
$$

where $W _ { i }$ denotes the semantic weight of the i-th GOP, $Q _ { i }$ denotes the PSNR of the i-th GOP, and $Q _ { i } ^ { w }$ denotes the corresponding weighted PSNR. Accordingly, WPSNR provides a semantic-aware measure of reconstructed video quality by emphasizing GOPs with higher semantic importance.

## B. Problem formulation

Based on the above semantic-aware video quality metric, we formulate a joint source-channel optimization problem to maximize the overall weighted reconstruction quality under a bitrate constraint. The optimization problem is expressed as

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \boldsymbol { P } } } & { \displaystyle \sum _ { i = 1 } ^ { N } Q _ { i } ^ { w } } \\ { \mathrm { s . t . } } & { R ( \boldsymbol { P } ) \leq R _ { \operatorname* { m a x } } , } \\ & { P _ { i } ^ { S } \in \mathcal { P } ^ { S } , \quad \forall i = 1 , \ldots , N , } \\ & { P _ { i } ^ { C } \in \mathcal { P } ^ { C } , \quad \forall i = 1 , \ldots , N , } \end{array}\tag{15}
$$

where $R ( P )$ denotes the total bitrate corresponding to $P ,$ $R _ { \mathrm { m a x } }$ is the maximum allowable bitrate, and $\mathcal { P } ^ { S }$ and ${ \mathcal { P } } ^ { C }$ represent the feasible sets of the source encoding parameter and channel encoding parameter, respectively.

The formulated optimization problem is difficult to solve directly due to its non-convex nature. Specifically, the joint selection of source and channel coding parameters involves a complex rate-distortion tradeoff that depends on both video semantic importance and time-varying channel conditions. Moreover, the mapping from coding parameters to reconstructed video quality is highly nonlinear and cannot be explicitly characterized by a closed-form mathematical expression.

![](images/b1f14875e6856d9ff99775fb81c30cf93b2e8cd4e958a3682cda180023e0b7d9.jpg)  
Fig. 2. Illustration of the proposed inter-GOP semantic importance calculation method.

To address this issue, we adopt a heuristic learning-based approach and employ deep reinforcement learning (DRL) to learn an effective joint source-channel optimization policy under varying video semantics and channel states.

## III. INTER-FRAME SEMANTIC IMPORTANCE EVALUATION

The temporal semantic importance of each GOP is evaluated via an inter-frame approach. We first extract semantic features from video GOPs and subsequently model the cosine similarity at feature level between adjacent GOPs.

## A. Shifted-Window Feature Construction

The extracted features effectively preserve both the spatial structure and temporal dynamics of the video, thereby providing rich semantic information for subsequent modeling. To capture the inter-frame semantic relevance among adjacent GOPs, we adopt a shifted-window strategy, inspired by the shifted window mechanism in Swin Transformer [30], to perform local dependency analysis in the temporal domain. As illustrated in Fig. 2, the proposed method calculates the semantic importance of each GOP by measuring its semantic similarity with neighboring GOPs within the local window. Specifically, for each target GOP $X _ { i } ,$ , the local feature window is constructed to include the semantic features of M consecutive GOPs centered around the target GOP as much as possible. The corresponding index set is denoted by M(i) and defined as

$$
\mathcal { M } ( i ) = \left\{ j \in \mathbb { Z } \left| b _ { \mathrm { L } } ^ { ( i ) } \leq j \leq b _ { \mathrm { R } } ^ { ( i ) } \right. \right\} ,\tag{16}
$$

where $b _ { \mathrm { L } } ^ { ( i ) }$ and $b _ { r }$ denote the left and right boundaries of the window, respectively, subject to the constraints

$$
0 \leq b _ { \mathrm { L } } ^ { ( i ) } \leq i \leq b _ { \mathrm { R } } ^ { ( i ) } \leq N - 1 , \quad b _ { \mathrm { R } } ^ { ( i ) } - b _ { \mathrm { L } } ^ { ( i ) } + 1 = M .\tag{17}
$$

In this way, the target GOP is positioned as close to the center of the window as possible, while the window size M remains constant. By operating within local shifted windows, this strategy captures the temporal semantic dependencies among adjacent GOPs, thereby circumventing redundant global computations.

## B. Cosine-Similarity-Based Importance Metric

To quantify the temporal semantic importance of video content, we propose a cosine similarity-based inter-frame semantic importance measurement method. For two GOPs semantic features $F _ { i }$ and $F _ { j }$ , the cosine similarity is defined as

$$
\mathrm { s i m } ( F _ { i } , F _ { j } ) = \frac { \langle F _ { i } , F _ { j } \rangle } { \| F _ { i } \| \| F _ { j } \| } ,\tag{18}
$$

where $\langle \cdot , \cdot \rangle$ denotes the inner product. Since the semantic features extracted by the neural network are non-negative, and zero feature vectors do not arise in practice for natural video content, the cosine similarity is guaranteed to be strictly positive, i.e., sim $( F _ { i } , F _ { j } ) \in ( 0 , 1 ]$ . Based on cosine similarity, the semantic importance of the i-th GOP is defined as the inverse of the average cosine similarity between the target GOP and the other GOPs within the window,

$$
W _ { i } = \frac { 1 } { \frac { 1 } { M - 1 } \sum _ { { j \in \mathcal { M } ( i ) } } \sin ( F _ { i } , F _ { j } ) } .\tag{19}
$$

This definition implies that a GOP with lower semantic similarity to its neighboring GOPs is assigned a higher semantic importance, since it contains more distinct semantic information in the temporal sequence.

By sliding the local window over the entire video, the inter-GOP semantic importance vector of the video sequence can be obtained as

$$
\pmb { W } = \left( W _ { 1 } , W _ { 2 } , \dots , W _ { N } \right) .\tag{20}
$$

The resulting semantic importance vector is then used to guide subsequent semantic-aware source-channel parameter optimization. In our previous work [1], semantic importance was used to adaptively adjust the keyframe interval.

## IV. MPPO ALGORITHM FOR JOINT SOURCE–CHANNEL OPTIMIZATION

In this section, we propose a novel DRL algorithm called multi-actor proximal policy optimization (MPPO), which incorporates multiple actor networks and action spaces, to ad-

dress the joint source-channel optimization problem based on inter-frame semantic importance.

## A. DRL-based Parameter Selection Algorithm

In this subsection, we begin by discussing the key components of the MPPO algorithm and the step-by-step process of utilizing it to optimize the video joint source-channel optimization strategy. Subsequently, we delve into the complexity analysis of the proposed MPPO algorithm. We first present the main components of the proposed MPPO algorithm and detail the procedure for employing it to optimize the joint sourcechannel optimization strategy for video transmission. We then provide a computational complexity analysis of the proposed MPPO algorithm.

1) Components of the MPPO Algorithm: In this part, we provide a comprehensive description of the key components of the proposed MPPO algorithm. Once the video semantic features have been extracted and the corresponding inter-frame semantic importance has been computed, the joint sourcechannel optimization process is formulated as a Markov decision process (MDP). In this process, each $\mathrm { G O P }$ is sequentially encoded from $i = 1 \ \mathrm { t o } \ i = N$ according to the selected source and channel coding parameters. The MPPO algorithm consists of four essential components: a) action space, b) state space, c) reward function, and d) DRL agent. These components are outlined as follows:

a) Action space: The agent action consists of two parts, namely the source coding action and the channel coding action. Since source coding and channel coding focus on coding efficiency and transmission reliability, respectively, their optimization objectives are inherently different. Therefore, the proposed MPPO framework employs two actor networks to learn the corresponding policies. The source coding action is selected from the action space $\mathcal { A } _ { S } = \{ C R _ { \operatorname* { m i n } } , \ldots , C R _ { \operatorname* { m a x } } \}$ where CR denotes the compression rate. Similarly, the channel coding action is selected from the action space $A _ { C } =$ $\{ r _ { \operatorname* { m i n } } , \dots , r _ { \operatorname* { m a x } } \}$ , where r denotes the channel coding rate, and $r _ { \mathrm { m i n } }$ and $r _ { \mathrm { m a x } }$ represent the minimum and maximum channel coding rates, respectively. Here, $a _ { S } ^ { ( i ) } \in \mathcal { A } _ { S }$ and $a _ { C } ^ { ( i ) } \in$ $\boldsymbol { \mathcal { A } } _ { C }$ denote the specific source coding action and channel coding action selected at the i-th decision step, respectively. At each step, the two actor networks output the source and channel coding actions, after which source coding is performed first, followed by channel coding.

b) State space: The state of the i-th GOP is defined as $s ^ { ( i ) } = [ W _ { i } , \bar { W } _ { i } , f _ { i } ^ { W } , \bar { \gamma } _ { i } , f _ { i } ^ { \gamma } ]$ , where $W _ { i }$ denotes the semantic weight of the current GOP, $\bar { W } _ { i }$ denotes its normalized value in the range $[ 0 , 1 ] , f _ { i } ^ { W } \in \{ - 1 , 1 \}$ is a binary semantic-weight flag, which is determined by whether the semantic weight of the i-th GOP is larger than the median value of all semantic weights, $\bar { \gamma } _ { i }$ denotes the normalized channel state derived from the SNR, and $f _ { i } ^ { \gamma } \in \{ - 1 , 1 \}$ is a binary channel-state flag dynamically determined according to the SNR, where the threshold is set to the integer SNR point at which the decodingfailure probability is approximately $1 / 2$ . In this way, all state variables are scaled to a comparable range, which facilitates the observation and learning process of the agent.

c) Reward function: The reward function is defined as

$$
R _ { i } = c _ { 1 } \cdot Q _ { i } ^ { w } - c _ { 2 } \cdot d _ { i } + c _ { 3 } \cdot R _ { i } ^ { \mathrm { s n r } } + c _ { 4 } \cdot R _ { i } ^ { \mathrm { w e i g h t } } - \rho \cdot \delta _ { i } ,\tag{21}
$$

where $d _ { i }$ denotes the amount of transmitted data, $R _ { i } ^ { \mathrm { s n r } }$ and $R _ { i } ^ { \mathrm { w e i g h t } }$ denote the reward terms associated with the SNR flag and the semantic-weight flag, respectively. This reward function encourages the agent to maximize the reconstruction quality while controlling the transmission cost. Specifically, $R _ { i } ^ { \mathrm { w e i g h t } }$ is defined as $R _ { i } ^ { \mathrm { { \scriptsize { w e i g h t } } } } = f _ { i } ^ { W } \left( a _ { S } ^ { ( i ) } - \tilde { a } _ { S } \right)$ , and $R _ { i } ^ { \mathrm { s n r } }$ is defined as $R _ { i } ^ { \mathrm { s n r } } = f _ { i } ^ { \gamma } \left( a _ { C } ^ { ( i ) } - \tilde { a } _ { C } \right)$ , where $\tilde { a } _ { S }$ and $\tilde { a } _ { C }$ denote the median values of the source coding action set $\mathcal { A } _ { S }$ and the channel coding action set $\boldsymbol { \mathcal { A } } _ { C }$ , respectively. Moreover, $\delta _ { i } = 1$ if decoding failure occurs at step i and $\delta _ { i } = 0$ otherwise, while $\rho > 0$ is a predefined penalty constant. The term $- c _ { 2 } d _ { i }$ serves as a penalty on the amount of transmitted data. Since this penalty is accumulated over each training episode, selecting actions that incur higher transmission costs without yielding commensurate improvements in reconstruction quality results in a lower cumulative reward. Consequently, the agent learns to avoid such inefficient actions and select source–channel coding parameters that maintain the total bitrate within the constraint $R ( P ) \ \leq \ R _ { \mathrm { m a x } }$ . The coefficients $c _ { 1 } , c _ { 2 } , c _ { 3 } , c _ { 4 }$ are hyperparameters that balance the respective reward terms and are found insensitive to the choice of dataset in practice. By aligning the step-wise reward with the objective in (15), the proposed MPPO algorithm solves the formulated optimization problem through cumulative reward maximization.

d) DRL Agent: The agent is the decision-making entity that interacts with the environment, and is implemented in practice by a set of neural networks. Specifically, it maintains two separate policies, $\pi _ { S } ( a _ { S } ^ { ( i ) } \mid s ^ { ( i ) } )$ and $\pi _ { \cal C } ( a _ { \cal C } ^ { \bar { ( i ) } } \mid s ^ { ( i ) } )$ , governing the selection of source coding and channel coding actions, respectively. At each step, the agent observes the current state $s ^ { ( i ) }$ , samples actions from the two policies, and receives the reward $R ^ { ( i ) }$ from the environment. By iteratively updating both policies through reward feedback and state transitions, the agent gradually learns the optimal source encoding and channel coding strategy. In the proposed MPPO algorithm, the agent is implemented with two actor networks and one shared critic network, corresponding to the two decoupled action spaces.

## 2) Network Architecture:

As illustrated in Fig. 3, the architecture of the proposed MPPO mainly consists of two actor networks and one critic network. The two actor networks have the same structure, but are used to generate different actions, namely the source coding action $a _ { S } ^ { ( i ) }$ and the channel coding action $\dot { a } _ { C } ^ { ( i ) }$ , respectively. Each actor network takes the input state $s ^ { ( i ) }$ and consists of three consecutive fully connected (FC) layers. The first two FC layers are followed by the Tanh activation function, while the final layer employs a Softmax function to generate the predicted probability distribution over the candidate actions. Based on the output distributions of the two actor networks, the source coding action and channel coding action are selected for the i-th GOP.

The critic network is used to evaluate the current decision and also consists of three FC layers. The first two layers are followed by the Tanh activation function, while the final layer outputs the state value, which serves as an estimate of the quality of the selected actions at the current state. In this way, the two actor networks and the critic network work cooperatively to determine the action selection process and evaluate the corresponding decision quality, thereby optimizing the joint source-channel optimization policy.

![](images/76f309f45b9bd0796eb783d947f7af118a93469f3970a5c736b58ed561ffbb7a.jpg)  
Fig. 3. Architecture of semantic-aware joint source–channel optimization for video communication

3) Training Procedure: We next introduce the training procedure of the proposed MPPO algorithm. First, the current state $s ^ { ( i ) }$ is fed into the MPPO agent, which learns the source coding policy $\pi _ { S }$ and the channel coding policy $\pi _ { C }$ and accordingly outputs actions $a _ { S } ^ { ( i ) }$ and $a _ { C } ^ { ( i ) }$ for the source encoder and channel encoder, respectively. Based on the selected coding parameters, the video is successively processed by source coding and channel coding to generate the encoded bitstream, which is then transmitted over the wireless channel. After transmission, channel decoding and source decoding are performed sequentially to reconstruct the video. Then, the WPSNR of the reconstructed video is calculated to evaluate the video quality. Based on the transmission result and reconstruction quality, the reward is obtained and used to guide the update of the policy networks.

During the parameter updating phase, the proposed MPPO framework updates one critic network with parameters ψ and two actor networks with parameters θ, following the multiactor reinforcement learning architecture in [31]. For each actor network, the parameter updating procedure is performed as follows. We first compute the discounted cumulative reward at timestep t, denoted as $R ^ { t }$ , which can be expressed as

$$
R ^ { t } = \sum _ { k = 1 } ^ { t } \eta ^ { t - k } r ^ { k } ,\tag{22}
$$

where η denotes the discount factor and $r ^ { k }$ is the immediate reward at timestep k. Then, the estimator of the advantage function at timestep t, denoted as $\hat { \phi } ^ { t }$ , is calculated by

$$
\hat { \phi } ^ { t } = R ^ { t } - e ^ { t } ,\tag{23}
$$

where $e ^ { t }$ is the state value estimated by the critic network. The estimator $\hat { \phi } ^ { t }$ reflects the advantage of the selected action

over the expected return at the current state.

Next, the probability ratio between the new policy and the old policy is computed as

$$
p ^ { t } ( \theta ) = \frac { \pi _ { \theta } ( a ^ { t } | s ^ { t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( a ^ { t } | s ^ { t } ) } ,\tag{24}
$$

where $\theta _ { \mathrm { o l d } }$ denotes the vector of policy parameters before the update. In line with [32], the main objective of the actor network is

$$
L _ { 1 } ( \theta ) = \mathbb { E } \left[ \operatorname* { m i n } \left( p ^ { t } ( \theta ) \hat { \phi } ^ { t } , \operatorname { c l i p } \left( p ^ { t } ( \theta ) , 1 - \epsilon , 1 + \epsilon \right) \hat { \phi } ^ { t } \right) \right] _ { \mathcal { A } }\tag{25}
$$

where ϵ is clip parameter and E is the expectation operator. The clipping function clip(·) is used to constrain the probability ratio within a predefined range, so as to avoid excessively large policy updates and improve the stability of training.

Following the work in [32], the overall objective further incorporates a value function loss term and an entropy bonus term, which can be written as

$$
\begin{array} { r } { L ( \theta ) = \hat { \mathbb { E } } _ { t } \left[ L _ { 1 } ( \theta ) - c _ { 5 } L _ { t } ^ { V F } ( \theta ) + c _ { 6 } S [ \pi _ { \theta } ] ( s ^ { t } ) \right] , } \end{array}\tag{26}
$$

where $c _ { 5 }$ and $c _ { 6 }$ are coefficients, S denotes the entropy bonus, and $L _ { t } ^ { V F } ( \theta )$ is the value function loss. In our implementation, the value function loss is defined as $L _ { t } ^ { V F } ( \theta ) \stackrel { - } { = } \left( e ^ { t } - R ^ { t } \right) ^ { 2 }$ The second term improves the accuracy of state-value estimation, while the third term encourages policy exploration.

Finally, the policy parameters are updated by gradient descent on the sampled trajectories according to

$$
\theta ^ { ( k ) } \gets \theta ^ { ( k - 1 ) } - \delta \nabla _ { \theta } O ( \theta ) ,\tag{27}
$$

where $\theta ^ { ( k ) }$ denotes the policy parameters at the k-th iteration and δ is the learning rate.

By iteratively updating the policy until convergence, the proposed MPPO algorithm can learn the mapping from source encoding and channel coding parameters to reconstructed video quality. In this way, it is able to determine an effective policy for achieving the optimal rate-distortion tradeoff. The detailed training procedure of the proposed MPPO algorithm is summarized in Algorithm 1.

<table><tr><td colspan="2">Algorithm 1 Training Process of the Proposed MPPO  $\mathrm { \sf A l g o - }$  rithm</td></tr><tr><td>1: Input: Video dataset  $\delta ,$  discount factor</td><td> ${ \mathcal { D } } ,$  training epoch  $E _ { p } ,$  learning rate  $\eta ,$  hyper-parameters  $\epsilon , c _ { 5 } ,$  and  $c _ { 6 } ,$  pre- trained semantic feature extractor, source encoder, and</td></tr><tr><td></td><td>channel encoder. 2: Output: Policy network parameters θ and critic network</td></tr><tr><td>3: repeat</td><td>parameters  $\psi$ </td></tr><tr><td>4:</td><td>Sample a batch of video sequences from dataset D.</td></tr><tr><td>5:</td><td>Extract semantic features and compute GOP semantic</td></tr><tr><td></td><td>importance by (19). Collect trajectories using the old policies.</td></tr><tr><td>6: 7:</td><td>Compute the cumulative reward by (22) and the loss</td></tr><tr><td></td><td>function by (26).</td></tr><tr><td>8:</td><td>Update the network parameters by (27).</td></tr><tr><td></td><td>9: until the loss function in (26) converges.</td></tr></table>

## B. Complexity Analysis

The computational complexity of the proposed MPPO framework arises from three components, namely semantic feature extraction, inter-frame semantic importance evaluation, and the MPPO algorithm.

For the first part, semantic features are extracted by pretrained deep neural network, the complexity of feature extraction for one GOP can be approximated as $\mathcal { O } ( D _ { \mathrm { h } } D _ { \mathrm { w } } )$ Therefore, for a video sequence containing N GOPs, the overall complexity of semantic feature extraction is

$$
\mathcal { O } ( N D _ { \mathrm { h } } D _ { \mathrm { w } } ) .\tag{28}
$$

For the second part, inter-frame semantic importance is computed based on cosine similarity. Let $d _ { f }$ denote the dimension of the semantic feature and M denote the semantic window size. Since each GOP needs to compute cosine similarity with the other M − 1 GOPs in the window, the complexity for one GOP is $O ( ( M - 1 ) d _ { f } )$ . Hence, for all $N$ GOPs, the complexity of semantic importance evaluation is

$$
\mathcal { O } ( N ( M - 1 ) d _ { f } ) .\tag{29}
$$

For the third part, let $d _ { s }$ and $d _ { a }$ denote the dimensions of the state space and action space, respectively. Let $H _ { l }$ denote the number of neurons in the l-th layer of the neural network, and let L denote the number of layers. The computational complexity of the MPPO-based decision process for one GOP can be approximated as [20]

$$
{ \mathcal O } \left( d _ { s } d _ { a } \prod _ { l = 1 } ^ { L } H _ { l } \right) .\tag{30}
$$

Accordingly, for a video sequence with N GOPs, the complexity of this part is

$$
\mathcal { O } \left( N d _ { s } d _ { a } \prod _ { l = 1 } ^ { L } H _ { l } \right) .\tag{31}
$$

Therefore, the total computational complexity of the proposed framework can be expressed as

$$
\mathcal { O } \left( N D _ { \mathrm { h } } D _ { \mathrm { w } } + N ( M - 1 ) d _ { f } + N d _ { s } d _ { a } \prod _ { l = 1 } ^ { L } H _ { l } \right) .\tag{32}
$$

## V. SIMULATION RESULTS AND ANALYSIS

In this section, we perform a comprehensive series of simulations aimed at validating the efficacy of both the proposed SAJSCO scheme and the MPPO algorithm.

## A. Simulation Setup

1) Video Dataset: We use the ActivityNet dataset as the training dataset for the proposed MPPO algorithm. ActivityNet is a large-scale benchmark for human activity understanding, consisting of untrimmed videos collected from YouTube and covering 203 activity categories. Since the videos are collected from real-world web sources, their spatial resolutions are not fixed and vary across samples [33]. In our experiments, the training and test subsets of ActivityNet are divided with a ratio of 5:1. For testing, we also use the HEVC test dataset, where the selected sequences belong to Class D with a resolution of 416 × 240 according to the HEVC common test conditions.

2) Channel Environment: The channel dataset used in this work is RadioML2016.10a [34], which is a widely used benchmark dataset for wireless signal modulation analysis under different channel conditions. Specifically, we select one SNR sequence from the RadioML2016.10a dataset and use its first 64 SNR values to model the time-varying channel condition. The selected SNR values are then normalized and remapped to the range of 0–20 dB according to the considered simulation setting. In addition, LDPC coding is employed for channel encoding and decoding to enhance transmission reliability over the wireless channel.

3) Semantic Extraction Module: The video semantic extractor is built upon the MobileNetV2 architecture [35], which employs inverted residual blocks, linear bottlenecks, and depthwise separable convolutions to reduce computational complexity. It efficiently extracts discriminative frame-level semantic features.

4) Baselines: The baselines are categorized as follows. (i) Non-semantic baselines: H.265 and DCVC-RT [19] with fixed coding rate configurations are included as representatives of non-semantic video coding. Both encoders exploit intraframe and inter-frame redundancy for compression, yet their fixed-parameter configurations allocate resources uniformly without considering the semantic importance of video content. The proposed SAJSCO employs a single MPPO policy, which is trained once and subsequently integrated with both H.265 and DCVC-RT as a plug-in module without encoder-specific retraining or fine-tuning, thereby enabling temporal semanticaware adaptive parameter selection across heterogeneous video encoders. (ii) Spatial-semantic baseline: SwinJSCC [27], a representative analog joint source-channel coding (JSCC) method, is included for comparison. SwinJSCC considers only spatial semantics at the intra-frame level and does not capture temporal semantic dependencies across frames. For a fair comparison, the first frame of each GOP is selected as the key frame and fed into SwinJSCC as an individual image, with the channel bandwidth ratio kept consistent with that of the proposed method.

In the simulations, the source coding action space is instantiated by five CR levels, i.e., $a _ { S } ^ { ( i ) } \in \{ 3 5 , 3 7 , 4 0 , 4 3 , 4 5 \}$ which correspond to different source compression rates. The channel coding action space is instantiated by three LDPC coding rates, i.e., $a _ { C } ^ { ( i ) } \ \in \ \{ 1 / 3 , 1 / 2 , 2 / 3 \}$ . The modulation scheme employed is 16QAM. The quality of the reconstructed videos is evaluated by using PSNR, WPSNR, LPIPS [36], and semantic importance weighted LPIPS(WLPIPS). Similar to the computation of WPSNR in (14), WLPIPS is obtained by weighting the LPIPS of each GOP according to its semantic importance. PSNR and WPSNR reflects the distortion at the pixel level, whereas LPIPS and WLPIPS measures perceptual similarity from the perspective of deep feature representations. The simulations and experiments are performed by the computer with Ubuntu20.04 + CUDA12.9, and the selected deep learning framework is Pytorch. The hardware configuration includes a single NVIDIA Tesla V100 GPU and an Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz. Other simulation parameters are summarized in Table I.

TABLE I  
SIMULATION PARAMETERS
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Number of GOPs, N</td><td>64</td><td>Shifted-window size, M</td><td>8</td></tr><tr><td>Training episodes</td><td>500</td><td>PPO update epochs,  $E _ { p }$ </td><td>8</td></tr><tr><td>Discount factor, η</td><td>0.999</td><td>Clip parameter, €</td><td>0.2</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Penalty constant,  $\rho$ </td><td>50</td></tr><tr><td>Actor learning rate,  $\delta _ { a }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td><td>Critic learning rate,  $\delta _ { c }$ </td><td> $3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Hyper-parameter, c1</td><td>10</td><td>Hyper-parameter, c2</td><td>0.3</td></tr><tr><td>Hyper-parameter, c3</td><td>200</td><td>Hyper-parameter, c4</td><td>10</td></tr><tr><td>Hyper-parameter, c5</td><td>0.5</td><td>Hyper-parameter,  $c _ { 6 }$ </td><td>0.5</td></tr></table>

![](images/d4f2460abc2bc1b5126c45fc03c7598fc39ad1230806f6f223106ebd03e8fe10.jpg)  
Fig. 4. Ablation study of the proposed MPPO algorithm.

## B. Training Convergence and Ablation Study

To evaluate the effectiveness of both the state space design and the multi-actor policy architecture, we conduct an ablation study during the training stage. Specifically, two state-space variants are constructed by separately removing the semanticweight flag $f _ { i } ^ { W }$ and the channel-state flag $f _ { i } ^ { \gamma }$ . In addition, the proposed MPPO is compared with a single-actor PPO (SPPO), in which a single actor jointly selects the source and channel coding actions. All variants are evaluated under the same training settings to ensure a fair comparison. The convergence process over the first 300 training episodes is illustrated in Fig. 4. It can be observed that the proposed MPPO converges faster and achieves a higher and more stable reward than all ablated variants. Removing either $f _ { i } ^ { W }$ or $f _ { i } ^ { \gamma }$ degrades the training performance. Moreover, MPPO consistently outperforms SPPO, which can be attributed to the distinct functional roles of source and channel coding: Source coding governs compression efficiency, whereas channel coding is primarily responsible for ensuring transmission reliability. Unlike a single actor that may struggle to disentangle the heterogeneous roles of the two actions, the multi-actor architecture allows each actor to specialize in its respective decision while optimizing a shared reward.

## C. Video Transmission Under Fixed CSI

We employ the H.265 video encoder on Activitynet dataset and compare against methods with fixed source-channel coding parameters. Specifically, three baselines are constructed using the same three LDPC code rates as the action space, and the CR is adjusted to match the bitrate of the proposed method. The PSNR and LPIPS results versus SNR are shown in Fig. 5. The reconstructed video quality exhibits an evident cliff effect as the SNR varies. This is mainly because the conventional H.265 bitstream is highly sensitive to channel errors. Once critical bits are corrupted during transmission, the decoder may fail, resulting in a sharp performance degradation. The proposed SAJSCO method consistently achieves performance close to the best fixed coding strategy over a wide SNR range, demonstrating strong adaptability to varying channel conditions. More specifically, under low-SNR conditions, the proposed method tends to allocate more bandwidth to improve transmission robustness. As the SNR increases, it gradually reduces the bandwidth while maintaining competitive reconstruction quality, therefore a slight performance decline can also be observed in the high-SNR region.

## D. Video Transmission Under Time-Varying CSI

In this experiment, the previously introduced RadioML channel dataset is adopted to more realistically simulate the time-varying wireless channel during video transmission. Specifically, one SNR value is assigned to each timestep, and after N transmission steps, the average PSNR, WPSNR, LPIPS, and WLPIPS of the entire video transmission process are calculated. For each encoder, three fixed channel coding rates are selected, and the rate set is identical to the candidate channel coding rates in the action space of the proposed MPPO algorithm, so as to ensure a fair comparison.

Fig. 6 shows the PSNR and LPIPS results versus bitrate on the ActivityNet dataset, where H.265 is adopted as the video encoder and RadioML is used to model the wireless channel. For PSNR, SAJSCO achieves BD-rate reductions of 22.74%, 28.92%, and 52.87% compared with the three non-semantic baselines, respectively. For LPIPS, the proposed method reduces the BD-rate by 17.69%, 15.73%, and 36.09% compared with the three non-semantic baselines, respectively. Moreover, SAJSCO also improves the average PSNR over the overlapping bitrate range while reducing LPIPS, indicating better perceptual quality. These results confirm that the proposed method can effectively adapt the source and channel coding parameters according to the channel condition.

![](images/2d6b55b708a269a8fb774ef3c71b8336579ed2e5ebbd88ac8a3a076290c314e0.jpg)  
(a) PSNR versus SNR

![](images/8b9e88768c68e0b0144f7dadce818238d051323cf3a439e600c4fda336fd7734.jpg)  
(b) LPIPS versus SNR

Fig. 5. Performance comparison under different SNR conditions on the ActivityNet dataset with the H.265 encoder.  
![](images/43e2765e3d730afb87ccca70e441c46ea366befc0c74849eaf71e779b45c1601.jpg)  
(a) PSNR versus bitrate

![](images/f7e9994105b86c00322eb2c918b2bf2d74208f4351afb7cc80473e1c095d5c85.jpg)  
(b) LPIPS versus bitrate  
Fig. 6. PSNR and LPIPS performance versus bitrate on the ActivityNet dataset with the H.265 encoder under the RadioML channel setting.

We conduct simulations on HEVC test dataset class D. As shown in Fig. 7, nine curves are plotted in each subfigure, including the proposed SAJSCO, the non-semantic baselines and spatial-semantic baseline SwinJSCC. It can be observed that the proposed SAJSCO achieves better performance with both H.265 and DCVC-RT, while significantly outperforming the spatial-semantic baseline SwinJSCC. The proposed method achieves consistent BD-rate gains over the nonsemantic baselines for both H.265 and DCVC-RT. Compared with the best-performing non-semantic baseline, the proposed SAJSCO scheme still achieves BD-rate reductions of 34.86%, 37.58%, 4.12% and 9.17% under H.265-based transmission in terms of PSNR, WPSNR, LPIPS, and WLPIPS, respectively. When DCVC-RT is adopted as the video encoder, SAJSCO further reduces the BD-rate by 18.01%, 21.77%, 10.24%, and 16.41%, respectively, compared with the strongest nonsemantic baseline. These gains mainly come from the ability of the proposed MPPO framework to adaptively adjust the source and channel coding parameters according to the timevarying CSI and the semantic importance of video content, whereas the non-semantic baselines use fixed coding parameters throughout the transmission process.

TABLE II  
COMPARISON OF COMPLEXITY
<table><tr><td>Method</td><td>MACs</td><td>Params</td><td>Speed</td></tr><tr><td>H.265</td><td>none</td><td>none</td><td>Enc. 42.41 fps, Dec. 25 fps</td></tr><tr><td>SAJSCO+H.265</td><td>318.97M</td><td>2.23M</td><td>Enc. 35.73 fps, Dec. 25 fps</td></tr><tr><td>DCVC-RT</td><td>385G</td><td>20.7M</td><td>Enc. 109 fps, Dec. 90 fps</td></tr><tr><td>SAJSCO+DCVC-RT</td><td>385.32G</td><td>22.93M</td><td>Enc. 66.73 fps, Dec. 90 fps</td></tr></table>

## E. Complexity and Visualization

Fig. 8 presents the visualization results of the 61st–64th frames of the HEVC testset classD BasketballPass video sequence, which belong to the 8th GOP under the timevarying channel condition. The corresponding SNR is 13.7 dB. Four transmission schemes are compared. Under the current bitrate setting, H.265 is combined with the best-performing LDPC coding rate of 1/3, while DCVC-RT is combined with the best-performing LDPC coding rate of 2/3. Due to the different channel coding rates, the two baselines exhibit different levels of error resilience. As shown in the figure, the reconstructed frames are noticeably degraded by channel noise in the current GOP. In particular, DCVC-RT suffers from more severe visual distortion than H.265 under this channel condition. By contrast, after introducing the proposed SAJSCO strategy, the reconstructed frames of both coding schemes exhibit greater robustness against channel noise, with more visual details preserved and better overall quality achieved. This further demonstrates the effectiveness and superiority of the proposed method in robust video transmission over timevarying wireless channels.

We tested H.265, H.265+SAJSCO, DCVC-RT and DCVC-RT+SAJSCO in terms of MACs, parameters, and encoding/decoding speed for 1080p videos, as summarized in TA-BLE II. It should be noted that H.265 is implemented with

![](images/b5a6debbd470edb11f8dc9df0962aac9db1118de762874662f388c97c268b734.jpg)  
(a) PSNR versus bitrate

![](images/acedd6a2e3d71add644f559a048a9ed29999b3355a1590a77c8d4cad51d90a6f.jpg)  
(b) WPSNR versus bitrate

![](images/14091116c631f7fc558d665d099a336676b9fd14429f0214d404f74b3498a22a.jpg)  
(c) LPIPS versus bitrate

![](images/ea9539c77bd29caf65c68410f62912a050c9afb113481b178705d507521b4c4c.jpg)  
(d) WLPIPS versus bitrate

Fig. 7. Performance comparison under the RadioML CSI setting on the HEVC testset.  <sup>E</sup>  
![](images/1b849eee1a4bd5bc37d57f5389c8ce7994c344c49deda4b1a1d0a6702c55c7a2.jpg)  
Fig. 8. Visualization comparison of reconstructed video frames under different transmission schemes.

CPU-based encoding and decoding, whereas DCVC-RT is accelerated using GPU-based encoding and decoding. Compared to non-semantic baselines, the SAJSCO scheme incurs only 4.36 ms of extra latency per frame while maintaining real-time encoding capability, with a negligible increase in parameters.

and signal transmission and reception.To further illustrate the decision mechanism of the proposed MPPO algorithm, Fig. 9 visualizes the state space, <sup>Experimental</sup> <sup>Results</sup> <sup>and</sup> <sup>Analysis</sup>including the CSI represented by SNR and the inter-frame The experiments are conducted over a line-of-sight (LoS)semantic import nce, as well as the corresponding action nsmission link at a carrier frequency of 2.45 GHz. Toselection, including the source coding parameter CR and the channel coding rate r. As shown in Fig. 9a, both the channel condition and the semantic importance vary over time, leading to a dynamically changing transmission environment. Correspondingly, Fig. 9b shows that the proposed algorithm adjusts the source and channel coding parameters adaptively. Specifically, SAJSCO adaptively selects more robust coding parameters for semantically important GOPs or unfavorable channel conditions, while adopting more efficient compression under lower semantic importance or favorable channel conditions to reduce transmission cost. This behavior demonstrates that the proposed MPPO algorithm is able to jointly exploit CSI and semantic importance information to adaptively select appropriate source-channel coding parameters, thereby achieving a better tradeoff between transmission reliability and coding efficiency.

![](images/446ed61771c84955b80259231a24fca1dfb5a85d2836335753afa5b176a06428.jpg)  
(a) Visualization of the state space.

![](images/c3cb9587c487f1023a0b68b7d380a47a54db0b01940fb3dab6b58d62eb060749.jpg)  
(b) Visualization of the selected actions.  
Fig. 9. Visualization of the state space and the corresponding action selection of the proposed MPPO algorithm.

## VI. PROTOTYPE VALIDATION

To further evaluate the practical performance of the proposed algorithm beyond simulation, we implement and validate it on a real-world communication testbed. The testbed is constructed using two Universal Software Radio Peripheral (USRP) devices, which serve as the transmitter and receiver, respectively. The remainder of this section is organized as follows. We first describe the overall hardware architecture of the prototype platform, followed by a presentation and analysis of the experimental results obtained from the physical testbed.

## A. Testbed Architecture

![](images/d68295a0ed4195f3ce6cd62b5bdae6ad285984a98c9a8516367915bdb9bd1a76.jpg)  
Fig. 10. Hardware architecture of the proposed USRP-based communication testbed.

As depicted in Fig. 10, the prototype testbed consists of two laptops and two USRP B210 software-defined radio (SDR)

TABLE III  
EXPERIMENTAL PARAMETERS
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Carrier Frequency</td><td>5.5 GHz</td><td>LDPC Block Length</td><td>1800 bits</td></tr><tr><td>Sampling Rate</td><td>1MSps</td><td>TX Gain (Seg. 1 &amp; 4)</td><td>50 dB</td></tr><tr><td>Symbol Rate</td><td>62.5 kSps</td><td>TX Gain (Seg. 2 &amp; 3)</td><td>60 dB</td></tr><tr><td>Bandwidth</td><td>125 kHz</td><td>RX Gain</td><td>56 dB</td></tr><tr><td>Modulation</td><td>BPSK</td><td>Antenna Model</td><td>VERT2450</td></tr><tr><td>Samples per Symbol</td><td>16</td><td>Antenna Placement</td><td>Vertical</td></tr><tr><td>Preamble Length</td><td>16384 bits</td><td>Link Distance</td><td>160 cm</td></tr></table>

devices, where each laptop is connected to and controls one USRP unit via USB. The two USRPs function as the transmitter and receiver. The detailed radio frequency and hardware configuration parameters are summarized in Table III. On the software side, the USRP devices are controlled via Python scripts using the USRP Hardware Driver (UHD), which provides low-level access to the radio front-end for baseband signal transmission and reception.

## B. Experimental Results and Analysis

The experiments employs a packet-based single-carrier BPSK waveform at a carrier frequency of 5.5 GHz over a line-of-sight (LoS) transmission link. The transmit gain is varied across four equal-duration segments throughout the transmission: The first and last segments operate at 50 dB, while the two intermediate segments operate at 60 dB. The receive gain is maintained at a constant 56 dB throughout all segments. This dynamic gain profile is designed to simulate the channel variations commonly encountered in practical wireless communication scenarios.

As shown in Table IV, experiments are conducted with two video encoders, namely H.265 and DCVC-RT. In both cases, SAJSCO achieves higher reconstruction quality and decoding success rate than the corresponding non-semantic baselines. Specifically, H.265+SAJSCO achieves a best PSNR of 28.905 dB and a decoding success rate of 93.75%, surpassing the best fixed-parameter H.265 baseline (r = 1/2) by 1.448 dB in PSNR and 4.695% in decoding success rate. Similarly, DCVC+SAJSCO achieves a best LPIPS of 0.306, outperforming the best fixed-parameter DCVC baseline (r = 1/3) by 0.033. H.265-based methods yield superior PSNR, WPSNR, and decoding success rate, while DCVC-RT-encoded bitstreams are more susceptible to decoding failure due to the absence of a fixed packetization structure, yet achieve more favorable perceptual quality as reflected by lower LPIPS and WLPIPS scores. These results confirm that the performance gains observed in simulation consistently transfer to real-world wireless transmission, validating the practical deployability of the proposed scheme on commodity SDR hardware.

## VII. CONCLUSION

This paper proposed SAJSCO, a lightweight, encoderagnostic plug-in framework for semantic-aware joint source– channel optimization in video communication. Within this framework, we constructed a video encoding parameter optimization model by jointly considering inter-frame semantic importance and channel state information. By quantifying inter-frame semantic importance through shifted-window cosine similarity and adaptively selecting source–channel parameters via a multi-actor PPO algorithm, the proposed scheme balances reconstruction quality against transmission cost under time-varying channels. Extensive simulation results showed that the proposed scheme achieves significant performance gains over both traditional and deep video encoders, with 34.86% and 18.01% BD-rate reductions, respectively. Furthermore, the proposed scheme is implemented and validated on a real-world USRP-based communication testbed, where experimental results consistently confirm the performance improvements observed in simulation.

TABLE IV  
PERFORMANCE COMPARISON ON THE USRP TESTBED
<table><tr><td>Method</td><td>PSNR</td><td>WPSNR</td><td>LPIPS</td><td>WLPIPS</td><td>Dec. Rate</td></tr><tr><td>H.265+SAJSCO</td><td>28.905</td><td>28.922</td><td>0.358</td><td>0.351</td><td>93.75%</td></tr><tr><td> $\mathrm { H } . 2 6 5 \mathrm { ~ } ( r = 1 / 3 )$ </td><td>25.358</td><td>25.578</td><td>0.518</td><td>0.508</td><td>84.38%</td></tr><tr><td> $\mathrm { H } . 2 6 5 \mathrm { ~ } ( r = \mathrm { 1 } ^ { ' } / 2 )$ </td><td>27.457</td><td>27.514</td><td>0.400</td><td>0.403</td><td>89.06%</td></tr><tr><td>H.265 (r = 2/3)</td><td>27.262</td><td>27.425</td><td>0.527</td><td>0.544</td><td>92.19%</td></tr><tr><td> $\mathrm { D C V C + S A J S C O }$ </td><td>27.741</td><td>27.585</td><td>0.306</td><td>0.311</td><td>75.00%</td></tr><tr><td> $\mathrm { D C V C } ~ ( r = 1 / 3 )$ </td><td>26.455</td><td>26.084</td><td>0.339</td><td>0.351</td><td>70.31%</td></tr><tr><td> $\mathrm { D C V C } ~ ( r = 1 / 2 )$ </td><td>23.917</td><td>23.753</td><td>0.394</td><td>0.400</td><td>46.88%</td></tr><tr><td> $\mathrm { D C V C } \ ( r = 2 / 3 )$ </td><td>26.074</td><td>26.266</td><td>0.350</td><td>0.347</td><td>60.94%</td></tr></table>

## REFERENCES

[1] X. Zhu, C. Guo, Y. Yang, C. Liu, and K. Ding, “Semantic-aware video communication: Enhancing traditional and deep video encoders,” in 2026 IEEE Wireless Communications and Networking Conference (WCNC). IEEE, 2026, pp. 1–6.

[2] Y. Sanjalawe, S. Fraihat, S. Al-E’Mari, M. Abualhaj, S. Makhadmeh, and E. Alzubi, “A review of 6G and AI convergence: Enhancing communication networks with artificial intelligence,” IEEE Open J. Commun. Soc., vol. 6, pp. 2308–2355, 2025.

[3] G. J. Sullivan, J.-R. Ohm, W.-J. Han, and T. Wiegand, “Overview of the high efficiency video coding (HEVC) standard,” IEEE Trans. Circuits Syst. Video Technol., vol. 22, no. 12, pp. 1649–1668, Dec. 2012.

[4] N. Li, Y. Deng, and D. Niyato, “Goal-oriented semantic communication for wireless video transmission via generative ai,” IEEE Trans. Wireless Commun., 2026.

[5] H. Gao, M. Sun, X. Xu, B. Xu, S. Han, B. Wang, S. Jiang, C. Dong, and P. Zhang, “Cross-layer encrypted semantic communication framework for panoramic video transmission,” IEEE Internet Things J., 2025.

[6] T. Y. Tung and D. Gund ¨ uz, “DeepWiVe: Deep-learning-aided wireless¨ video transmission,” IEEE J. Sel. Areas Commun., vol. 40, no. 9, pp. 2570–2583, 2022.

[7] S. Wang, J. Dai, Z. Liang et al., “Wireless deep video semantic transmission,” IEEE J. Sel. Areas Commun., vol. 41, no. 1, pp. 214– 229, 2023.

[8] C. Liang, H. Du, Y. Sun, D. Niyato, J. Kang, D. Zhao, and M. A. Imran, “Generative ai-driven semantic communication networks: Architecture, technologies, and applications,” IEEE Trans. Cogn. Commun. Netw., vol. 11, no. 1, pp. 27–47, 2025.

[9] Y. Wang, P. H. Chan, and V. Donzella, “Semantic-aware video compression for automotive cameras,” IEEE Trans. Intell. Vehicles, vol. 8, no. 6, pp. 3712–3722, 2023.

[10] C. Liang, X. Deng, Y. Sun et al., “VISTA: Video transmission over a semantic communication approach,” in Proc. IEEE Int. Conf. Commun Workshops (ICC Workshops). IEEE, 2023, pp. 1777–1782.

[11] A. Aliouat, N. Kouadria, M. Maimour, S. Harize, and N. Doghmane, “Region-of-interest based video coding strategy for rate/energyconstrained smart surveillance systems using WMSNs,” Ad Hoc Networks, vol. 140, p. 103076, 2023.

[12] Y. Tian, G. Lu, G. Zhai, and Z. Gao, “Non-semantics suppressed mask learning for unsupervised video semantic compression,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 13 610–13 622.

[13] Y. Zhang, G. Lu, Y. Chen, S. Wang, Y. Shi, J. Wang, and L. Song, “Neural rate control for learned video compression,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2023.

[14] B. He, F. Wang, and T. Q. S. Quek, “Secure semantic communication via paired adversarial residual networks,” IEEE Wireless Commun. Lett., 2024.

[15] Y. Yang, C. Guo, F. Liu, L. Sun, C. Liu, and Q. Sun, “Semantic communications with artificial intelligence tasks: Reducing bandwidth requirements and improving artificial intelligence task performance,” IEEE Ind. Electron. Mag., vol. 17, no. 3, pp. 4–13, 2023.

[16] C. Liu, C. Guo, Y. Yang, and N. Jiang, “Adaptable semantic compression and resource allocation for task-oriented communications,” IEEE Trans. Cogn. Commun. Netw., vol. 10, no. 3, pp. 769–782, 2024.

[17] C. Liu, C. Guo, Y. Yang, M. Chen, and T. Q. S. Quek, “Lightweight taskoriented semantic communication empowered by large-scale AI models,” IEEE Trans. Veh. Technol., 2025.

[18] Y. Chen, H. Wang, C. Liu et al., “Generative multi-modal mutual enhancement video semantic communications,” CMES-Comput. Model. Eng. Sci., vol. 139, no. 3, 2024.

[19] Z. Jia, B. Li, J. Li, W. Xie, L. Qi, H. Li, and Y. Lu, “Towards practical real-time neural video compression,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 12 543–12 552.

[20] C. Liu, C. Guo, Y. Yang et al., “OFDM-based digital semantic communication with importance awareness,” IEEE Trans. Commun., 2024.

[21] Z. Huang, J. Sun, X. Guo, and M. Shang, “Adaptive deep reinforcement learning-based in-loop filter for VVC,” IEEE Trans. Image Process., vol. 30, pp. 5439–5451, 2021.

[22] J. Li, H. Wang, Z. Liu, P. Zhou, X. Chen, Q. Li, and R. Hong, “Toward optimal real-time volumetric video streaming: A rolling optimization and deep reinforcement learning based approach,” IEEE Trans. Circuits Syst. Video Technol., vol. 33, no. 12, pp. 7870–7883, 2023.

[23] M. Zhou, X. Wei, S. Kwong, W. Jia, and B. Fang, “Rate control method based on deep reinforcement learning for dynamic video sequences in HEVC,” IEEE Trans. Multimedia, vol. 23, pp. 1106–1121, 2021.

[24] H. Yin, L. Qiao, Y. Ma, S. Sun, K. Li, Z. Gao, and D. Niyato, “Generative video semantic communication via multimodal semantic fusion with large model,” IEEE Trans. Veh. Technol., 2025.

[25] L. Qiao, M. B. Mashhadi, Z. Gao, R. Tafazolli, M. Bennis, and D. Niyato, “Token communications: A large model-driven framework for cross-modal context-aware semantic communications,” IEEE Wireless Commun., vol. 32, no. 5, pp. 80–88, 2025.

[26] Y. Peng, L. Xiang, K. Yang, K. Wang, and M. Debbah, “Semantic communications with computer vision sensing for edge video transmission,” IEEE Trans. Mobile Comput., 2025.

[27] K. Yang, S. Wang, J. Dai, X. Qin, K. Niu, and P. Zhang, “SwinJSCC: Taming swin transformer for deep joint source-channel coding,” IEEE Trans. Cogn. Commun. Netw., vol. 11, no. 1, pp. 90–104, 2025.

[28] S. Coleri, M. Ergen, A. Puri, and A. Bahai, “Channel estimation techniques based on pilot arrangement in OFDM systems,” IEEE Trans. Broadcast., vol. 48, no. 3, pp. 223–229, 2002.

[29] J. Erfurt, C. R. Helmrich, S. Bosse, H. Schwarz, D. Marpe, and T. Wiegand, “A study of the perceptually weighted peak signal-to-noise ratio (WPSNR) for image compression,” in Proc. IEEE Int. Conf. Image Process. (ICIP). IEEE, 2019, pp. 2339–2343.

[30] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 10 012–10 022.

[31] H. Yang, H. Tian, F. Guo, and R. Deng, “Safety-guaranteed energy management in networked multienergy microgrids: A multi-actor singlecritic deep reinforcement learning approach,” IEEE Trans. Ind. Informat., 2026.

[32] J. Schulman, F. Wolski, P. Dhariwal et al., “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[33] F. F.C. Heilbron, V. Escorcia, B. Ghanem, and J. J.C. Niebles, “ActivityNet: A large-scale video benchmark for human activity understanding,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2015, pp. 961–970.

[34] T. J. O’Shea and N. West, “Radio machine learning dataset generation with GNU radio,” in Proc. GNU Radio Conf., vol. 1, no. 1, 2016.

[35] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted residuals and linear bottlenecks,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 4510–4520.

[36] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 586–595.