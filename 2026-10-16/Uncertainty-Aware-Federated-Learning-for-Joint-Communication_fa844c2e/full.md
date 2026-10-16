# Uncertainty-Aware Federated Learning for Joint Communication and Computation in XR Systems

Robert-Jeron Reifert , Graduate Student Member, IEEE, Hayssam Dahrouj , Senior Member, IEEE, Alaa Alameer Ahmad , Haris Gacanin , Fellow, IEEE, and Aydin Sezgin , Senior Member, IEEE

Abstract—The convergence of communication and computation, empowered by artificial intelligence (AI) enhanced capabilities, emerges as a key enabler for realizing the promising functionalities of the sixth-generation of communication systems (6G). To this end, this paper investigates an AI-native, robust network, where multiple base stations (BSs) simultaneously serve several extended reality (XR) devices. The paper then presents an innovative distributed deep learning-based approach to simultaneously manage the transmit and computing powers, and computation allocation, amidst uncertainties in both channel and computing states information. Under such an interference network scenario, the paper proposes a robust approach for minimizing the worst-case delay across the served XR devices, subject to computation and power constraints. The paper leverages a deep neural network (DNN) that can be trained either centrally or in a federated manner to map estimated channels and computation requirements to optimized resource allocations per BS. During training, uncertainty samples, reflecting both channel and computing estimation errors, are injected after the DNN output to account for such joint uncertainties. Our results validate the enhanced robust delay performance and scalability of joint uncertainty injection as compared to uncertainty-agnostic methods and state-of-the-art benchmarks, with federated learningbased solutions achieving near-centralized performance while enhancing efficiency, privacy, and adaptability.

Index Terms—Uncertainty injection, federated learning, joint communication and computation, extended reality, robustness.

## I. INTRODUCTION

As the number of consumers combining smartphones and extended reality (XR) devices is set to double within five years, with three-quarters seeking XR experiences beyond home settings, the demand for cellular XR is rapidly accelerating [2]. Delivering gigabit-per-second data rates, millisecond-level latency, and high computational efficiency, XR is expected to drive the sixth-generation of communication systems (6G) toward major architectural and operational advancements [3]. This evolution requires substantial amounts of remote data connectivity and processing, underscoring the need for an integrated communication and computation platform [4]. Additionally, 6G is envisioned to comprise widespread ultradense networks with numerous base stations (BSs) and devices [5]. Efficient coordination within these networks is essential to prevent communication backbone overload while mitigating large-scale interference in real-time. Consequently, automated joint management of communication and computing resources is critical for future decentralized systems.

In response, the International Telecommunication Union (ITU) steers future networks (6G and beyond) toward native integration of machine learning (ML) and artificial intelligence (AI), enabling AI-native air interfaces and ML-driven resource management [6]. Deep learning, in particular, offers efficient and near-optimal wireless resource management [7], but often assumes perfect knowledge of channel conditions and computing requirements [8]. In practice, robust resource allocation strategies must be uncertainty-aware, i.e., explicitly account for uncertainties in both communication (e.g., missing or outdated channel state information [9]) and computation (e.g., fluctuating processing demands [10]). While prior works have addressed either communication or computation uncertainty separately, only a few have considered their joint effects [11]. From an AI/ML perspective, such uncertainties further propagate into learning-based decision-making. Since training data is often imperfect, ML-based architectures, as explored in [7], must operate under uncertainty in estimated channel states. However, a systematic approach to jointly handling communication and computation uncertainties in AI-driven op timization remains largely unexplored. To bridge this gap, this paper addresses such issues via a joint uncertainty injection scheme in a multi-BS, interference-limited network scenario, generalizing the single BS scenario considered in [1].

As 6G networks tend to embrace data processing decentralization, e.g., by pushing intelligence toward the edge, the paper proposed solutions are poised to provide a valuable datadriven, robust framework for managing multi-BS deployments [12]. In such decentralized networks, relying on traditional centralized learning would require transmitting raw data to a central entity, which is impractical due to privacy issues, excessive bandwidth consumption, and scalability limitations. Federated learning (FL) addresses these challenges allowing individual BSs to train local models using their own data and share only model updates with a central server [13]–[15]. This decentralized paradigm significantly reduces communication overhead while enhancing both scalability and security. Although FL aggregates model parameters globally, the resulting allocations remain heterogeneous across BSs, since each BS performs inference on distinct local data, e.g., local channel estimations. Thus, FL harmonizes training without enforcing identical allocation policies [16].

Along such lines, this paper develops a robust ML-based approach for joint communication and computation resource allocation in delay-sensitive networks with a particular focus on XR systems, accounting for uncertainties in both channel and computing states. More specifically, the paper goal is to minimize the worst-case delay across all XR devices. The paper then formulates both centralized and FL-based optimization frameworks, incorporating uncertainty injection to enhance robustness under both channel and computation estimation errors. The resulting problems are solved using unsupervised learning by means of adopting customized deep neural networks (DNN) trained either centrally or in a federated manner.

## A. Related Works

Within the International Mobile Telecommunications-2030 (IMT-2030), XR is recognized as a key driver of 6G, bringing forth challenges such as mobility, blockages, and AI integration [3]. To address mobility and blockages, [8] proposes a cooperation scheme where strong devices assist weaker ones to improve communication efficiency. Building on this, [13] combines decentralized online learning with caching to reduce the XR delay. Minimizing the XR service latency is explored via cloud-based [17] and edge-based offloading [18]. A key challenge introduced by XR, however, is the need for joint management of communication and computation resources, as explored in recent works [8]–[11], [19]. While [8] assumes perfect knowledge of channels and computing requirements, which is unrealistic in practice, robust approaches have addressed these limitations individually. For communication, references [9], [19] tackle imperfect channel state information. In contrast, [10] introduces a framework to meet task deadlines under processing uncertainty. While [9], [10], [19] address either communication or computation uncertainty, [11] broadens the scope by adopting robust edge processing of video streams before being streamed to the devices.

Despite their numerical merits, [8]–[11], [19] adopt classical optimization techniques. More recently, the integration of AI/ML with next-generation networks is shown to offer scalable and adaptive alternatives, particularly suited for complex tasks like advanced radio resource management [20]. For instance, [21] investigates the use of neural networks in wireless settings, while [22] applies FL for the joint optimization of communication and computation resources. Given that training data is inherently imperfect, [7] proposes an uncertainty injection framework in deep learning to enhance the robustness under channel estimation errors. Yet, the joint impact of communication and computation uncertainties remains largely unexplored, with their coupled effects within the delay objective and shared resource constraints being technically challenging. Based on our conference version [1], we address such gap by jointly considering both channel and processing uncertainty in a multi-BS, interference-limited setting.

In fact, the above-discussed works, including [1], rely on traditional centralized architectures, which come with limiting factors such as front- and backhaul bottlenecks, cloud latency, and privacy concerns. Future networks, however, are poised to spearhead decentralized network architectures, offering advantages such as flexibility, resilience, and lower latency [23]. In particular, a promising approach is FL, which enables efficient knowledge aggregation across devices without requiring continuous interaction or exploration-based updates [24]. The work [25] provides a survey of recent advances in optimization methods for wireless communications, eventually encompassing distributed learning. FL enhances data privacy by ensuring that original training data remains within its source domain [14], while also enabling secure aggregation [15]. For instance, [26] applies FL-based resource allocation for ultrareliable low-latency communication, with significantly reduced data exchange compared to centralized benchmarks.

Recently, some works jointly address the challenges of integrated communication and computation under uncertainty using ML/AI-aided approaches. This includes optimizing communication and computation resources using FL for energy efficiency [22] and employing quantization to minimize the convergence time [27]. Furthermore, [28] accounts for packet errors, while [29] includes channel uncertainty for joint client scheduling. Lastly, [13] investigates personalized FL for cellular VR, integrating online learning and field-of-view caching. While some works (e.g., [22], [27], [28]) focus on FL within wireless networks, others (e.g., [13], [24], [26], [29]) explore FL as a tool for optimizing network operations. In this paper, we adopt the latter perspective, leveraging FL for distributed decision-making and resource allocation.

## B. Contribution

Building upon the aforementioned references, this paper proposes a robust federated deep learning framework for joint communication and computation resource management under channel and computing uncertainties. To this end, the paper considers a multi-BS empowered XR network, where the XR devices rely on strong edge computing capabilities (at the BSs) and high-quality communication links. Unlike our conference version [1], which considers only a single BS, this paper extends the framework to a multi-BS cellular XR network. Our objective is to minimize the worst-case delay across all devices by optimizing transmit power, computing power, and computation allocations. However, this problem is inherently challenging due to the stochastic constraints from channel and computing uncertainties, the communication and computation resource interdependency, and the network’s distributed nature. Compared to [1], our extended model introduces additional operational challenges, such as multi-BS interference, scalable network-wide optimization, and the unique constraints of XR applications, all of which are considered in the current paper.

More specifically, to enable real-time decision-making, we employ an ML-based approach. Our proposed DNN takes estimated channels and computing requirements as inputs and outputs optimized power and computation allocations. The

DNN can then be trained in an unsupervised manner using either centralized learning or an FL approach, where local BSs compute gradient updates that are aggregated at a central server to form a global model. A key novelty is the use of joint uncertainty injection, where sampled channel and computing errors are injected directly into the DNN’s output.

Extensive numerical evaluations validate the effectiveness of uncertainty injection in enhancing the network-wide robust delay performance across various uncertainty levels and network densities while maintaining scalability. The proposed FLbased solutions perform close to the centralized upper bound while significantly outperforming non-ML approaches, and offer practical benefits in terms of complexity and data exchange. The results demonstrate that joint uncertainty injection generalizes beyond single-uncertainty state-of-the-art schemes. These findings underscore the feasibility of the proposed AIdriven, multi-purpose resource management framework for real-world cellular XR applications. The key contributions are summarized as follows:

We consider a cellular XR-based network with multiple distributed, edge-enabled BSs serving delay-sensitive XR devices for both computation (e.g., scene modeling, rendering) and communication (data transmission), under channel and computing uncertainties.

We formulate uncertainty-aware worst-case delay minimization problems under both centralized and distributed settings so as to determine the optimal transmit power, computing power, and computation allocations.

To overcome the complexity of conventional optimization, we develop robust centralized and federated deep learning frameworks, incorporating joint uncertainty injection (injecting channel and computing estimation errors after the DNN outputs) during training.

Our numerical results show that joint uncertainty injection improves the network-wide worst-case delay among XR devices. The proposed FL scheme approaches the centralized upper bound, outperforming non-ML baselines, while being computationally efficient, reducing data exchange, and refining the DNN in early training epochs. This highlights the proposed scheme’s potential in provisioning joint communication and computing for robust XR networks.

## C. Notation and Organization

We use boldface lowercase for vectors (e.g., h), boldface uppercase for matrices (e.g., H), and calligraphic letters for sets (e.g., ). <sub>Θ</sub> denotes a DNN with parameter set Θ. Further, Hermitian transpose, transpose, absolute value, and $\ell _ { p } .$ -norm are denoted as $( \cdot ) ^ { \dag } , \left( \cdot \right) ^ { T } , \left| \cdot \right|$ , and $\| \cdot \| _ { p } ,$ respectively.

The paper is organized as follows: Section II presents the system components, XR operations, and joint communication and computing model, along with the uncertainty-aware worstcase delay minimization problem. Section III introduces the centralized DNN and the joint uncertainty injection mechanism. Section IV extends the discussion to the decentralized FL framework. Simulation results and analysis are shown in Section V, and the paper is concluded in Section VI.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

In this work, we consider an XR network comprising B BSs equipped with embedded computing capabilities, collectively serving a total of K single-antenna XR devices. Each BS is equipped with $L _ { b }$ antennas and hosts a local ML model server to facilitate local training, parameter storage, and inference. We denote the set of BSs by $\boldsymbol { B } = \{ 1 , \cdots , \boldsymbol { B } \}$ and the set of XR devices by $\mathcal { K } = \{ 1 , \cdots , K \}$ . Furthermore, for BS $b ,$ the set of antennas is denoted by $\mathcal { L } _ { b } = \{ i _ { 1 , b } , \cdots , i _ { L _ { b } , b } \}$ , with $i _ { j , b }$ being the $j \cdot$ -th antenna of the b-th BS, with L being the combined number of all BSs’ antennas. The network further comprises of a central processor (CP) with a global ML model server and processing capabilities. Depending on the CP’s integration level and functionality, the network architecture may be categorized as multi-cell, coordinated (Section III), or distributed (Section IV) radio access network (RAN) [30].

We assume disjoint XR device-clusters in this work, where each XR device k is associated with exactly one BS. The set of devices served by BS b is denoted by $\textstyle \mathcal { K } _ { b } \subseteq \mathcal { K }$ , with $\mathcal { K } _ { b } \cap \mathcal { K } _ { b ^ { \prime } } = \emptyset , \forall b \neq b ^ { \prime }$ . Here, $K _ { b }$ denotes the number of devices served by BS b. Fig. 1 illustrates the system model with 3 BSs and 8 XR devices, depicting communication, computation, and inter-BS interference links. In the centralized approach, the CP coordinates all BSs and hosts the network-wide DNN. In the FL-based approach, however, each BS trains a local model for its associated users and the CP maintains the global model, while each user is still primarily served by a single BS.

## A. XR Device Operation

In many key XR applications, such as image and video processing or digital twinning, devices increasingly rely on robust remote processing capabilities provided by the network edge [17]. For example, [11] introduces UAVideo, a framework in which aerial vehicles generate video data, process it locally or at the edge, and then forward it to devices. Minimizing the total delay in such frameworks is critical to ensuring smooth and responsive user experiences. Each device’s total delay comprises two primary components, namely the communication delay, representing the time incurred on the physical communication medium, and the computation delay, denoting the time required to complete the device’s computational tasks.

In our setup, devices such as head-mounted displays, tablets, smartphones, and smartwatches require advanced computation services provided by the network infrastructure. For instance, a head-mounted display may receive a 360-degree video stream pre-rendered and encoded at the BS. Concurrently, control inputs from the XR device, such as movement data or fieldof-view changes, influence the rendering model. This creates a strong interdependence between modeling, pre-rendering, and encoding processes at the BS [17]. Importantly, the computational term in our model captures these rendering and encoding processes, which are known to dominate the processing complexity in XR systems [31]. In contrast, control signaling involves significantly smaller data rates and latency contributions compared to the video stream [8], [18], [31], and is, therefore, negligible relative to the dominant computation and communication delays. Similarly, cache access delays are typically much smaller than rendering and transmission delays in edge-assisted XR systems and are thus not explicitly modeled [18]. Overall, this abstraction focuses on the dominant latency components, highlighting the interplay between communication and computing, and emphasizing the need for efficient resource allocation to meet the stringent latency and reliability requirements of XR applications.

![](images/14de37fc0c599d0f3cae85cc7ee9fe2f2ad4b730b938dd1d58f2003cec3667c4.jpg)  
Fig. 1: Network setup with one central server and global DNN model, 3 multi antenna BSs each serving a number of XR devices accommodating a local DNN model. Devices’ communication, computation, and inter-BS interference links are shown.

## B. Channel and Communication Delay Models

Let $h _ { b , l , k } \in \mathbb { C }$ be the channel coefficient from the b-th BS’s antenna l to device $k ,$ where $\begin{array} { r } { \pmb { h } _ { b , k } = [ h _ { b , i _ { 1 , b } , k } , \cdot \cdot \cdot , h _ { b , i _ { L _ { h , b } , k } } ] ^ { T } } \end{array}$ is the aggregate channel vector between BS b and device k. Device k’s full aggregate channel vector including all BS links is denoted by $\bar { \mathbf { \Xi } } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi } \mathbf { \Xi \Xi } \mathbf { \Xi } \mathbf \mathbf { \Xi } \mathbf { \Xi } \mathbf \Xi \mathbf { \Xi } \mathbf \Xi \Xi \mathbf { } \mathbf \Xi \Xi \mathbf { } \Xi \mathbf \Xi \Xi \mathbf { } \mathbf \Xi \Xi \mathbf \Xi \Xi \mathbf \Xi \Xi \mathbf { } \mathbf \Xi \mathbf \Xi \Xi \mathbf \Xi \mathbf \Xi \Xi \mathbf \Xi \Xi \mathbf \Xi \mathbf \Xi \Xi \mathbf \Xi \mathbf \Xi \mathbf \Xi \Xi \mathbf \Xi \mathbf \mathbf \Xi \Xi \mathbf \mathbf \Xi \mathbf \Xi \mathbf \Xi \mathbf \Xi \mathbf \mathbf \mathbf \Xi \mathbf \Xi \mathbf \mathbf \mathbf \Xi \mathbf \mathbf \mathbf \mathbf \mathbf \Xi \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \Xi \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf \mathbf $ . We assume that the BSs only have access to an estimate of the channels. That is, the estimated channel from BS b’s antenna l to device k is denoted by $\hat { h } _ { b , l , k } \ \in \ \mathbb { C } ,$ with the aggregate estimated channel vectors denoted by $\hat { h } _ { b , k } = [ \hat { h } _ { b , i _ { 1 , b } , k } , \cdot \cdot \cdot , \hat { h } _ { b , i _ { L _ { b } , b } , k } ] ^ { T }$ and $\hat { \pmb { h } } _ { k } \ = \ [ \hat { \pmb { h } } _ { 1 , k } ^ { T } , \cdot \cdot \cdot , \hat { \pmb { h } } _ { B , k } ^ { T } ] ^ { T }$ . The estimated channels approximate the real channels only up to some extend, with some uncertainty consideration. The related literature provides a multitude of different uncertainty models, e.g., including additive or multiplicative uncertainties [32], assuming specific uncertainty bounds [9], or opting for statistical uncertainties [7]. In practice, the channel uncertainty distribution is not known [9]. Yet, as channel estimations can be sampled in large amounts, a data-driven approach is able to learn the channel uncertainty implicitly. Similar to the existing literature [7], we model the channel coefficients in our paper as follows:

$$
h _ { b , l , k } = \hat { h } _ { b , l , k } + h _ { b , l , k } ^ { \mathrm { e r r } } ,\tag{1}
$$

where $h _ { b , l , k } ^ { \mathrm { e r r } } \sim \mathcal { C N } ( 0 , \sigma _ { h } ^ { 2 } )$ is the channel estimation error. Note that the uncertainty-aware framework adopted in (1) does not require explicit knowledge of the underlying distribution of $h _ { b , l , k } ^ { \mathrm { e r r } }$ . In practice, this distribution may vary depending on the channel estimation process (e.g., Gaussian, Rician, or mixed fading statistics). The proposed algorithm operates purely on sampled realizations of the uncertainty and is, therefore, agnostic to the specific parametric form of the distribution. For the sake of reproducibility and consistency with prior work, the simulations in this paper adopt the Gaussian assumption, which is widely used in the literature, e.g., [7] and references therein. In Sec. V-F, we further extend the evaluation to non-Gaussian uncertainty models to validate the generalization capability of the proposed framework.

The beamforming vector for device k, transmitted by BS b, is denoted by $\pmb { v } _ { b , k } \in \mathbb { C } ^ { L _ { b } }$ and defined as $\begin{array} { r l } { { \pmb v } _ { b , k } } & { { } = } \end{array}$ $[ v _ { b , i _ { 1 , b } , k } , \cdot \cdot \cdot , v _ { b , i _ { L _ { b } , b } , k } ] ^ { T }$ , where $\| \pmb { v } _ { b , k } \| _ { 2 } = 1$ . The aggregate beamforming vector for device k is represented as $\begin{array} { r l } { { v _ { k } } } & { { } = } \end{array}$ $[ \pmb { v } _ { 1 , k } ^ { T } , \cdots , \pmb { v } _ { B , k } ^ { T } ] ^ { T }$ . Since each device is associated with only one BS, ${ \pmb v } _ { k }$ is group-sparse by design. This sparsity is enforced by setting ${ \pmb v } _ { b , k } \ = \ { \pmb 0 } ^ { L _ { b } }$ for all $k \ \in \ \mathcal { K } \backslash \mathcal { K } _ { b }$ , at each BS b. Furthermore, let $p _ { k } ^ { \mathrm { t x } }$ denote the transmit power allocated by the serving BS to device k. The transmit power vector is then represented as $\pmb { p } ^ { \mathrm { t x } } = [ p _ { 1 } ^ { \mathrm { t x } } , \cdots , p _ { K } ^ { \mathrm { t x } } ] ^ { T }$

It is important to note that, due to the different time scales between transmit power or computational resource allocation and the device-to-transmitter association, a fixed association provides a reasonable modeling abstraction within the scope of this paper. In practice, association and handover decisions occur at slower time scales and can be treated as outerloop processes [33]. Moreover, any valid association satisfying the aforementioned constraints can be incorporated without fundamentally changing the proposed framework. For moderate variations in the active device set, this can be achieved using standard techniques such as zero-padding, masking, or maximum-size system design [34]. Consequently, the presented results are indicative of performance under different underlying associations.

To clarify the time scales discrepancy among the considered variables, we consider three different categories, namely, radio scheduling/inference (fast), channel/workload variations (medium), and model training (slow). Communication time slots are typically in the order of sub-ms [35] and XR rendering budgets are around 10 ms per frame, with DNN inference running once per BS per scheduling interval. In contrast, channel statistics and XR workload profiles evolve in the higher ms-range (mobility) up to seconds (XR scene changes). Lastly, DNN training and FL aggregation occur offline or in the background at a slower time scale (minuteshours) and are not part of the real-time loop; only inference is latency-critical [36]. This decoupling ensures that only the lightweight inference operates on the sub-ms/ms budget, while (re)training can be triggered when long-term statistics, such as user density or mobility patterns, change [37]. Extensions toward fully dynamic association and variable-size architectures (e.g., permutation-invariant or graph-based models) are left for future work.

The transmitted messages for all devices are encoded as $\xi _ { k }$ $\forall k \in \mathcal { K }$ , where $\xi _ { k } \sim \mathcal { C N } ( 0 , 1 )$ are independent and identically distributed, circularly symmetric Gaussian random variables. Each device k receives its own desired signal, interference from other signals, and noise. Device k’s received signal is

$$
y _ { k } = { h _ { k } ^ { H } } { v _ { k } } \sqrt { p _ { k } ^ { \mathrm { t x } } } { \xi _ { k } } + \sum _ { j \in K \setminus \{ k \} } { h _ { k } ^ { H } } { v _ { j } } \sqrt { p _ { j } ^ { \mathrm { t x } } } { \xi _ { j } } + n _ { k } ,\tag{2}
$$

where $n _ { k }$ represents the additive white Gaussian noise with zero mean and variance $\sigma ^ { 2 }$ . The achievable rate of device k is then given by

$$
r _ { k } = W \log _ { 2 } \left( 1 + \frac { | h _ { k } ^ { H } \pmb { v } _ { k } | ^ { 2 } p _ { k } ^ { \mathrm { t x } } } { \varphi _ { k } + \sigma ^ { 2 } } \right) ,\tag{3}
$$

with system bandwidth W and total interference

$$
\varphi _ { k } = \sum _ { \substack { j \in { \cal K } _ { b } \backslash \{ k \} } } | h _ { k } ^ { H } { \pmb v } _ { j } | ^ { 2 } p _ { j } ^ { \mathrm { t x } } + \sum _ { \substack { j \in { \cal K } \backslash \{ { \cal K } _ { b } \} } } | h _ { k } ^ { H } { \pmb v } _ { j } | ^ { 2 } p _ { j } ^ { \mathrm { t x } } ,\tag{4}
$$

where b denotes the BS serving device $k ,$ while $\varphi _ { k } ^ { \mathrm { i n t r a } }$ and $\varphi _ { k } ^ { \mathrm { i n t e r } }$ represent the intra-BS and inter-BS interference levels, respectively. These terms are detailed as follows:

Intra-BS interference: This interference arises from devices served by the same BS as device k. This interference can be managed by the BS through power allocation.

Inter-BS interference: This interference originates from devices served by other BSs and is beyond the control of device k’s serving BS.

## C. Computing Framework and Computation Delay

From the computation perspective, each XR device requires the execution of specific computational tasks, such as fieldof-view pre-rendering, object tracking, or activity detection [18]. In this work, device k requests to process a task, which consists of an input data size $D _ { k } ^ { \mathrm { i n } }$ , and a computation intensity $\omega _ { k }$ , i.e., the required computation cycles per bit of input data is denoted by $\omega _ { k }$ , for a total of $\omega _ { k } D _ { k } ^ { i n }$ computation cycles for device $k ' \mathrm { s }$ task. However, such a computational requirement $\omega _ { k }$ may not be perfectly known at the corresponding BS, as only an estimation $\hat { \omega } _ { k }$ is available. Specifically, computing environments are highly dynamic, i.e., they vary depending on system load and concurrent processes, and systems dependency, i.e., the inter-relation of input data characteristics [38]. Various statistical models for $\omega _ { k }$ exist in the literature, including uniform, normal, and Gamma distributions [39]. This paper handles computation uncertainty in a data-driven manner by training on a large set of sampled uncertainty realizations. The computation intensity is modeled as

$$
\omega _ { k } = \hat { \omega } _ { k } + \omega _ { k } ^ { \mathrm { e r r } } ,\tag{5}
$$

where $\omega _ { k } ^ { \mathrm { e r r } } \sim \mathcal { N } ( 0 , \sigma _ { \omega } ^ { 2 } )$ is the computing estimation error. Similar to the Gaussian assumption remark made earlier about the channel estimation error, we also note here that the computation estimation error is assumed to be Gaussian for simulations purposes. The framework of our paper is, however, agnostic to the underlying error distribution, since the algorithm operates solely on sampled error realizations.

With an estimation of the required computation cycles for device $k \mathrm { { : } }$ task, i.e., $\hat { \omega } _ { k } D _ { k } ^ { \mathrm { i n } }$ , the corresponding BS allocates $f _ { k } ^ { \mathrm { c o } }$ in cycles/s to compute $k ' s$ task. The overall vector of computation allocations is denoted by $\pmb { f } ^ { \mathrm { c o } } = [ f _ { 1 } ^ { \mathrm { c o } } , \cdot \cdot \cdot , f _ { K } ^ { \mathrm { c o } } ] ^ { T }$ Due to the finite computation resources at the BSs, the maximum computation capacity constraint is given by

$$
\sum _ { k \in \mathcal { K } _ { b } } f _ { k } ^ { \mathrm { c o } } \le F _ { b } ^ { \mathrm { m a x } } , \qquad \forall b \in \mathcal { B } ,\tag{6}
$$

where $F _ { b } ^ { \mathrm { m a x } }$ is BS b’s maximum computation capacity. Due to the intertwined nature of communication and computation resources, especially in terms of the allocated power levels, we consider $p _ { b } ^ { \mathrm { c o } }$ as the total power allocated by BS b for the computation of all tasks. The overall vector of all BSs computation powers is denoted by $\pmb { p } ^ { \mathrm { c o } } = [ p _ { 1 } ^ { \mathrm { c o } } , \cdot \cdot \cdot , p _ { B } ^ { \mathrm { c o } } ] ^ { T }$ . The relation between the computation power $p _ { b } ^ { \mathrm { c o } }$ and computation allocation $f _ { k }$ is given by

$$
p _ { b } ^ { \mathrm { c o } } = \tau \left( \sum _ { k \in \mathcal { K } _ { b } } f _ { k } ^ { \mathrm { c o } } \right) ^ { \mu } ,\tag{7}
$$

where $\tau$ and $\mu$ are constants of the CPU model [8].

## D. Joint Communication and Computation Considerations

By jointly considering the communication and computation delay, device k experiences a total delay $t _ { k }$ as follows

$$
t _ { k } = \frac { \omega _ { k } D _ { k } ^ { \mathrm { i n } } } { f _ { k } ^ { \mathrm { c o } } } + \frac { D _ { k } ^ { \mathrm { o u t } } } { r _ { k } } ,\tag{8}
$$

where $D _ { k } ^ { \mathrm { o u t } }$ denotes the output data size of device $k ' \mathrm { s }$ task computation<sup>1</sup>. While $\omega _ { k }$ in (8) directly reflects computation uncertainty (see (5)), the rate $r _ { k }$ depends on the channel conditions (see (3)), thus capturing the impact of channel uncertainty. In order to achieve fair service among all devices, the worst-case delay is computed as $t _ { \mathrm { m a x } } = \operatorname* { m a x } _ { k \in \mathcal { K } } \{ t _ { k } \}$

In this work, we consider the coupling of communication and computation power consumption $( \mathrm { i } . \mathrm { e } . , p ^ { \mathrm { t x } }$ and $p _ { b } ^ { \mathrm { c o } }$ , respectively) at the BSs, which leads to the overall power constraint

$$
\sum _ { k \in { \mathcal K } _ { b } } p _ { k } ^ { \mathrm { t x } } + p _ { b } ^ { \mathrm { c o } } \le P _ { b } ^ { \operatorname* { m a x } } , \qquad \forall b \in { \mathcal B } ,\tag{9}
$$

where $P _ { b } ^ { \mathrm { m a x } }$ is the maximum allowable power per BS [40]. Eq. (9) emphasizes the intertwined communication and computation powers, especially underlining the need for a careful selection of joint resource allocation under limited resources.

## E. Robust Delay Utility

Based on (8), one can infer that the overall delay $t _ { k }$ of device k is a function of the channel vectors $h _ { k }$ and the computation intensities $\omega _ { k }$ , where both $h _ { k }$ and $\omega _ { k }$ follow some unknown probability distributions. Specifically, each BS only has access to the estimated $\hat { h } _ { k }$ and $\hat { \omega } _ { k } , \forall k \in \mathcal { K } _ { b }$

To best account for the channel and computing uncertainties, the paper utilizes a DNN with inputs $\hat { \boldsymbol { h } } _ { k } , \hat { \omega } _ { k } , \forall k \in \mathcal { K } .$ , and inject samples of uncertainty realizations of both channels and computing requirements into the training phase just after the DNN output [7]. That is, for a given resource allocation, we inject estimation errors to impact the objective value, and afterwards train the network according to the updated objective. Such uncertainty injection supplies a multitude of potential worst-case delays $t _ { \mathrm { m a x } }$ for different realizations. This paper then aims at minimizing the γ-th quantile of the worstcase delay distribution, denoted by $t _ { \mathrm { m a x } } ^ { \gamma }$ , where

$$
\begin{array} { r } { \operatorname* { P r } \Big \lceil t _ { \mathrm { m a x } } > t _ { \mathrm { m a x } } ^ { \gamma } \Big | \hat { h } _ { k } , \hat { \omega } _ { k } , \forall k \in \mathcal { K } \Big \rceil \leq \gamma , } \end{array}\tag{10}
$$

where $t _ { \mathrm { m a x } } ^ { \gamma }$ can be interpreted as the value for which only a fraction γ of all realizations have higher worst-case delays.

While we adopt the robust delay utility in (8) to capture the joint impact of channel and computation uncertainties, other robustness approaches can be found in the existing literature. For instance, common ML-based techniques include data augmentation [41] and noise injection [42], where uncertainty is artificially added to the training data or feature space. While effective in some domains, these methods do not directly capture the system-level uncertainties of communication and computation considered in the current paper. From an optimization perspective, robust designs based on sample average approximation (SAA) combined with weighted minimum mean square error (WMMSE) algorithms have been proposed (e.g., [43]) to address channel uncertainty. Such methods, however, are computationally demanding and less suitable for real-time operation. In contrast, our uncertainty injection scheme integrates robustness directly into the DNN training process, enabling the model to implicitly learn the uncertainty distribution [7].

## F. Problem Formulation

The paper aims at minimizing the robust worst-case delay $t _ { \mathrm { m a x } } ^ { \gamma }$ among all devices. The corresponding optimization problem can then be mathematically formulated as follows:

$$
\begin{array} { r l } { \underset { p ^ { \mathrm { t x } } , p ^ { \mathrm { c o } } , f ^ { \mathrm { c o } } } { \operatorname* { m i n } } } & { t _ { \mathrm { m a x } } ^ { \gamma } } \\ { \mathrm { s . t . } \quad } & { ( 1 ) , ( 5 ) , ( 6 ) , ( 9 ) , ( 1 0 ) , } \\ & { t _ { \mathrm { m a x } } \geq t _ { k } , } \end{array}\tag{11}
$$

$$
\forall k \in { \cal K } ,\tag{11a}
$$

where the computation capacity constraint is given in (6), and the power constraint is given in (9). Constraint (10) ensures the solution’s robustness in terms of the γ-th quantile. Constraint (11a) expresses the worst delay among all devices as $t ^ { \mathrm { m a x } }$ . The intricacy of problem (11) stems from the mathematical coupling between the communication and computation variables and constraints, as well as from the unknown distribution of the worst-case delay $t _ { \mathrm { m a x } }$ given the channel and computing states uncertainties. Additionally, another challenge of problem (11) lies in its centralized nature. Specifically, solving problem (11) requires global knowledge of all channel vectors and computation requirements, $\hat { h } _ { k }$ and $\hat { \omega } _ { k } , \forall k \in \mathcal { K } _ { b } , \forall b \in \mathcal { B }$

The paper next proposes a data-driven solution approach based on a deep learning framework with joint uncertainty injection in Section III. Afterwards, Section IV presents the distributed federated learning-based implementation of the proposed joint uncertainty-based deep learning protocol.

## III. UNCERTAINTY-AWARE CENTRALIZED LEARNING

To tackle the intricate optimization problem (11), this section harnesses the promising abilities of deep learning in terms of automated learning, feature representation, and versatility. Specifically, we utilize a DNN to directly map the problem parameters to a resource allocation solution, which is then further processed by injecting uncertainties for optimizing a robust objective. In a typical DNN, neurons are organized into layers, including input, hidden, and output layers. These layers are densely interconnected, with each connection between neurons having an associated weight that is continuously adjusted during the training process. Additionally, nonlinear activation functions are applied to the neurons within these layers. In general, depth (number of layers) and width (number of neurons in each layer) of the DNN are key factors that influence its capacity to learn and represent data [21].

In a centralized manner, all BSs are connected to and coordinated by a CP with a global ML model server and processing capabilities. We propose a DNN architecture, which takes the estimated channels and estimated computing requirements as inputs, i.e., $\hat { h } _ { k }$ and $\hat { \omega } _ { k } , \forall k \in \mathcal { K }$ . The input layer consists of $2 L K + K$ neurons, which represent the 2LK real and imaginary channel coefficients and the K distinct devices’ tasks. After the input layer, we attach $\eta _ { h }$ fully-connected hidden layers with $n _ { h }$ neurons each, and Rectified Linear Unit (ReLU) activation functions. The output layer consists of $2 K + B$ neurons, where the first $K + B$ entries represent the K devices allocated power and the BSs’ computation powers, and the remaining K entries represent the computation capacities. The DNN then returns the output vector $\pmb { x } = [ x _ { 1 } , \cdot \cdot \cdot , x _ { 2 K + B } ] ^ { T }$ representing the optimized resource allocation $ { p ^ { \mathrm { t x } } } ,  { p ^ { \mathrm { c o } } }$ , and $f ^ { \mathrm { c o } }$ , after some mathematical manipulations. We let $\mathcal { F } _ { \Theta }$ denote the proposed DNN, such that problem (11) is solved by

$$
\begin{array} { r } { \pmb { x } = \mathcal { F } _ { \Theta } ( \hat { h } , \hat { \omega } ) , } \end{array}\tag{12}
$$

where $\hat { \pmb { h } } = [ \hat { \pmb { h } } _ { 1 } ^ { T } , \cdots , \hat { \pmb { h } } _ { K } ^ { T } ] ^ { T }$ and $\hat { \omega } = [ \hat { \omega } _ { 1 } , \cdot \cdot \cdot , \hat { \omega } _ { K } ] ^ { T }$ . Note that Θ refers to the DNN parameters, e.g., weights and biases.<sup>2</sup> Fig. 2 illustrates a representation of the proposed DNN-based solution.

In the context of minimizing the robust worst-case delay among devices, we next explore how to map the DNN output to the resource allocation, how to ensure the robustness of the solution, i.e., in terms of the γ-th quantile guarantee, and how to train the DNN according to the robust objective.

## A. DNN-Driven Resource Allocation Strategy

A major complication of the joint communication and computation resource management problem (11) is the interdependent computation capacity (6) and power (9) constraints.

![](images/9044b76575399e18e5f5d87146acfd4fa0daaa758fbbbce9603145ac559613be.jpg)  
Fig. 2: Proposed DNN architecture with estimated input, resource allocation output, and uncertainty injection. Modifications for the distributed DNN are highlighted in green.

We, thus, first use the softmax function<sup>3</sup> on parts of the DNN output to compute each BS’s transmit and computation powers for all $b \in B$ by computing

$$
[ ( p _ { b } ^ { \mathrm { t x } } ) ^ { T } , p _ { b } ^ { \mathrm { c o } } ] ^ { T } = P _ { b } ^ { \mathrm { m a x } } \cdot \mathrm { s o f t m a x } \left( [ x _ { j _ { b } ^ { 1 } } , \cdot \cdot \cdot , x _ { j _ { b } ^ { K _ { b } + 1 } } ] ^ { T } \right) ,\tag{13}
$$

where the indices $j _ { b } ^ { 1 }$ and $j _ { b } ^ { K _ { b } + 1 }$ denote the first and last output associated with BS $b ,$ respectively. We can calculate them as $j _ { b } ^ { 1 } = ( b - 1 ) ( K _ { b } + 1 ) + 1$ and $j _ { b } ^ { K _ { b } + 1 } = b ( K _ { b } + 1 )$ $\mathrm { e . g . }$ for BS 1, $j _ { 1 } ^ { 1 } = 1$ and $j _ { 1 } ^ { K _ { b } + 1 } = K _ { b } + 1$ , respectively<sup>4</sup>. By using the softmax function, we ensure the full utilization of the complete power budget at each BS. Next, we transform the computation power into cycles using (7), so as to obtain the maximum number of available computation cycles limited by the power allocation, denoted by $F _ { b } ^ { \mathrm { p o w } }$ . Based on $( 7 ) , F _ { b } ^ { \mathrm { p o w } }$ can be written as:

$$
F _ { b } ^ { \mathrm { p o w } } = \sqrt [ \mu ] { p _ { b } ^ { \mathrm { c o } } \tau ^ { - 1 } } .\tag{14}
$$

At this point, there are two limiting factors for the computation allocation $( \mathrm { i . e . , } f ^ { \mathrm { c o } } )$ , namely, the BSs’ maximum computation capacities $F _ { b } ^ { \mathrm { m a x } }$ in (6), and the above power-limited available computation cycles $F _ { b } ^ { \mathrm { p o w } } , \forall b \in B$ . We then use the smallest among the two limiting factors, i.e., min $\left( F _ { b } ^ { \operatorname* { m a x } } , F _ { b } ^ { \operatorname* { p o w } } \right)$ . To determine the computation allocation $\pmb { f } _ { b } ^ { \mathrm { c o } }$ , we use the softmax function on remaining parts of the output layer $\forall b \in B$ by

$$
\begin{array} { r } { \pmb { f } _ { b } ^ { \mathrm { c o } } = \operatorname* { m i n } \bigr ( \boldsymbol { F } _ { b } ^ { \mathrm { m a x } } , \boldsymbol { F } _ { b } ^ { \mathrm { p o w } } \bigr ) \cdot \mathrm { s o f t m a x } \bigr ( [ \boldsymbol { x } _ { i _ { b } ^ { 1 } } , \cdot \cdot \cdot , \boldsymbol { x } _ { i _ { b } ^ { K _ { b } } } ] ^ { T } \bigr ) , } \end{array}\tag{15}
$$

which utilizes either the capacity- or power-limited computation cycles. Here, the indices $i _ { b } ^ { 1 }$ and $i _ { h } ^ { \dot { K } _ { b } }$ can be calculated as $i _ { b } ^ { 1 } = K + B + ( b - 1 ) K _ { b } + 1$ and $i _ { b } ^ { K _ { b } ^ { \smile } } = K + B + b K _ { b }$

## B. Joint Uncertainty Injection Scheme

Given the resource allocation $ { p ^ { \mathrm { t x } } } ,  { p ^ { \mathrm { c o } } }$ , and $\pmb { f } ^ { \mathrm { c o } }$ found via (13) and (15), one can readily compute a worst-case delay for one given channel and computing estimation. However, problem (11) aims at providing a robust $t _ { \mathrm { m a x } } ^ { \gamma }$ accounting for the uncertainties $\boldsymbol { h } ^ { \mathrm { e r r } } = [ h _ { 1 , 1 , 1 } ^ { \mathrm { e r r } } , \cdot \cdot \cdot , h _ { B , L _ { B } , K } ^ { \mathrm { e r r } } ] ^ { T }$ and $\omega ^ { \mathrm { e r r } } =$ $[ \omega _ { 1 } ^ { \mathrm { e r r } } , \cdot \cdot \cdot , \omega _ { K } ^ { \mathrm { e r r } } ] ^ { T }$ , given in (1) and (5), respectively. Thus, we next explain the uncertainty injection approach as first proposed by [7], and set forth the joint uncertainty injection scheme adopted in the joint computation and communication context considered in the current paper.

The basic idea of uncertainty injection is to estimate the statistical objective numerically, using a large number of realizations or samples of the uncertainty. Thereafter, such realizations are injected after the DNN output to compute a large number of objective values. These objectives are then used to obtain an empirical estimate of the robust objective. Such estimate is then used to derive gradients and update the DNN parameters via backpropagation [44].

In this paper, however, the problem is subject to both channel and computing state information uncertainties, as reflected in equations (1) and (5). The adopted objective $t _ { \mathrm { m a x } } ^ { \gamma }$ in problem (11) is, however, a function of the γ-th quantile of the worst-case delay distribution, as defined in (10). The network only has access to the estimated channel states, $\hat { h } ,$ and computing requirements, $\hat { \omega } ,$ , while the true worst-case delay distribution depends on the unknown channel and computing error distributions, $\pmb { h } ^ { \mathrm { e r f } }$ and $\omega ^ { \mathrm { e r f } }$ , respectively. To address this uncertainty, we adopt a sampling-based approach.

As shown in Fig. 2, the DNN operates on estimated channel and computing requirement inputs, $\hat { h }$ and ωˆ. After propagating through multiple hidden layers, the output vector is obtained as $\begin{array} { r } { \textbf { \em x } = \mathcal { F } _ { \Theta } ( \hat { h } , \hat { \omega } ) } \end{array}$ . Using equations (13) and (15), the corresponding resource allocations are derived. At this stage, the BSs sample N realizations of the channel and computing uncertainties, denoted by $\boldsymbol { h } _ { n } ^ { \mathrm { e r r } }$ and $\omega _ { n } ^ { \mathrm { e r r } }$ , for $n \in { 1 , \cdots , N } .$ These samples allow the network to compute N worst-case delays, given by $t _ { \operatorname* { m a x } , n } , \forall n \in { 1 , \cdots , N }$ as follows

$$
t _ { \mathrm { m a x } , n } = \operatorname* { m a x } _ { k \in \mathcal K } \Bigg \{ \frac { \big ( \hat { \omega } _ { k } + \omega _ { k , n } ^ { \mathrm { e r r } } \big ) D _ { k } ^ { \mathrm { i n } } } { f _ { k } ^ { \mathrm { c o } } } + \frac { D _ { k } ^ { \mathrm { o u t } } } { r _ { k , n } } \Bigg \} ,\tag{16}
$$

where $r _ { k , n }$ represents the data rate of the k-th XR device as a function of channel uncertainty, computed by incorporating (1) into (3) as

$$
r _ { k , n } = W \log _ { 2 } \left( 1 + \frac { | \left( \hat { h } _ { k } + h _ { k , n } ^ { \mathrm { e r r } } \right) ^ { H } v _ { k } | ^ { 2 } p _ { k } ^ { \mathrm { t x } } } { \varphi _ { k } + \sigma ^ { 2 } } \right) .\tag{17}
$$

This process, referred to as joint uncertainty injection in Fig. 2, occurs immediately after the DNN output. The computed delay values are then sorted, and the $( \gamma \cdot N )$ -th highest value is selected as an empirical estimate of the γ-th quantile of the worst-case delay, denoted as $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma } { } ^ { 5 }$

It is worth noting herein that the above uncertainty injection procedure is applied only during training, enabling the DNN to implicitly learn the underlying uncertainty distributions. Once trained, the DNN can robustly allocate resources based on estimated inputs without requiring further uncertainty sampling. As depicted in Fig. 2, training-related components are marked red, while real-time operation and testing are highlighted blue.

## C. Unsupervised Learning via Backpropagation

An integral part of the proposed solution is the DNN training procedure, especially in accounting for the uncertainties to ensure a robust solution to problem (11). More specifically, by taking the $( \gamma \cdot N )$ -th highest value $( \mathrm { i . e . , } \ \hat { t } _ { \mathrm { m a x } } ^ { \gamma } )$ of all computed worst-case delays $t _ { \operatorname* { m a x } , n } , \ \forall n \ \in \ \{ 1 , \cdots , N \}$ , gradients are derived from $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ to update the DNN parameters. To be more specific, we use the gradient $\partial \hat { t } _ { \mathrm { m a x } } ^ { \gamma } / \partial \Theta$ , which is computed using Pytorch [45]. In that way, the DNN is trained in an unsupervised fashion, implicitly learning the distribution of channel and computing uncertainties. The numerical merit of our proposed method is further discussed in the simulations section of the paper, as shown in Section V.

For completeness, the deep learning training procedure incorporating joint uncertainty injection is detailed in Algorithm 1. In summary, for each epoch, Algorithm 1 partitions the training process into minibatches, iterating M times with newly sampled realizations in each iteration. After processing all minibatches, the DNN’s performance is validated, and model parameters are saved if they yield the lowest observed robust delay, i.e., $\hat { t } _ { \mathrm { m a x , g l o b a l } } ^ { \gamma }$ . Within each minibatch (and during validation), the CP obtains channel and computing requirement estimates, performs DNN inference, determines resource allocations for each network realization, and updates the DNN parameters based on the computed gradients. Note that the last step is applied only during training. This iterative process continues until either a predefined convergence criterion is met or when the maximum number of training epochs is reached.

Algorithm 1 Proposed Centralized Training Procedure   
1: repeat   
2: for $j = 1 , \cdots , M$ do   
3: Obtain batch hˆ and $\hat { \omega }$   
4: Compute DNN output vector x via (12)   
5: Determine $ { p ^ { \mathrm { t x } } } ,  { p ^ { \mathrm { c o } } }$ , and $f ^ { \mathrm { c o } }$ via (13) and (15)   
6: Compute $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ as described in Section III-B   
7: DNN model update using gradient $\partial \hat { t } _ { \mathrm { m a x } } ^ { \gamma } / \partial \Theta$   
8: end for   
9: Obtain validation set $\hat { h }$ and $\hat { \omega }$   
10: Compute DNN output vector x via (12)   
11: Determine $ { p ^ { \mathrm { t x } } } ,  { p ^ { \mathrm { c o } } }$ , and $f ^ { \mathrm { c o } }$ via (13) and (15)   
12: Compute $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ as described in Section III-B   
13: if $\hat { t } _ { \mathrm { m a x } } ^ { \gamma } < \hat { t } _ { \mathrm { m a x , g l o b a l } } ^ { \gamma }$ then   
14: Save DNN parameters Θ and set $\hat { t } _ { \mathrm { m a x , g l o b a l } } ^ { \gamma } = \hat { t } _ { \mathrm { m a x } } ^ { \gamma }$   
15: end if   
16: until Convergence

Departing from the single-CP optimization problem (11) and its centralized solution via Algorithm 1, we next explore a practical distributed network architecture based on FL.

## IV. UNCERTAINTY-AWARE FEDERATED LEARNING

In this section, we consider the practically feasible and scalable distributed network architecture, as illustrated in Fig. 1. At the foundation of the decentralized network design, each BS independently performs model training, inference, and resource allocation. Under such realm, the CP serves solely as a global parameter server, aggregating and updating ML models without access to global network state information, such as channels, computation requirements, or interference levels. To this end, the next part of the paper formulates a decentralized optimization problem that is better suited to the practical constraints and scalability requirements of future large-scale systems, and proposes a FL-approach to solve the jointly robust optimization problem using a joint uncertainty injection mechanism.

![](images/164e3a74a8b7eb4b077fa5889241627bd1ebf343180de05cc49d2fb54ba01a35.jpg)  
Fig. 3: Channel knowledge representation for 2 BSs and 5 devices illustrating known and unknown channels.

## A. Distributed Network Architecture

In the centralized setup, the CP collects data from the BSs, trains a global ML model discussed in Section III, and determines the resource allocation for each BS. The following proposed distributed network architecture, however, redefines the CP’s role as a global parameter server responsible for ML model aggregation, updates, and transmission. Training and inference are, nevertheless, performed locally at each BS, as illustrated in Fig. 1. This results in a hybrid RAN architecture that exhibits both distributed and coordinated characteristics [30]. Specifically, resource allocation decisions are made locally at each BS, while the CP facilitates networkwide training and the exchange of interference information.

The distributed version of problem (11) for BS b becomes

$$
\begin{array} { r l } { \mathrm { s . t . } \quad } & { { } ( 1 ) , ( 5 ) , ( 6 ) , ( 9 ) , } \end{array}\tag{18}
$$

$$
\mathrm { P r } \left[ t _ { b , \operatorname* { m a x } } > t _ { b , \operatorname* { m a x } } ^ { \gamma } \vert \hat { h } _ { k } , \hat { \omega } _ { k } , \forall k \in \mathcal { K } _ { b } \right] \leq \gamma ,\tag{18a}
$$

$$
{ \varphi } _ { k } ^ { \mathrm { i n t e r } } = \sum _ { j \in { \mathcal { K } } \backslash \{ { \mathcal { K } } _ { b } \} } | { \pmb { h } } _ { k } ^ { H } { \pmb { v } } _ { j } | ^ { 2 } p _ { j } ^ { \mathrm { t x } } , \forall k \in { \mathcal { K } } _ { b } ,
$$

$$
t _ { b , \operatorname* { m a x } } \geq t _ { k } ,\tag{18b}
$$

$$
\forall k \in K _ { b } .\tag{18c}
$$

Here, the optimization variables $p _ { b } ^ { \mathrm { t x } } , p _ { b } ^ { \mathrm { c o } }$ , and $\pmb { f } _ { b } ^ { \mathrm { c o } }$ correspond to the transmit power, computation power, and computation cycles allocated to devices associated with BS b. The transmit power vector $\mathbf { \Delta } _ { p _ { b } ^ { \mathrm { t x } } } ^ { p _ { b } ^ { \mathrm { t x } } }$ and computation allocation vector $\pmb { f } _ { b } ^ { \mathrm { c o } }$ are subsets of $\pmb { p } ^ { \mathrm { t x } }$ and $\scriptstyle f ^ { \mathrm { c o } }$ , respectively, specific to the devices in $\displaystyle { \mathcal { K } } _ { b }$ . The objective is to minimize the worst-case delay among the devices served by BS b, expressed as $t _ { b , \operatorname* { m a x } } =$ $\operatorname* { m a x } _ { k \in \mathcal { K } _ { b } } \left\{ t _ { k } \right\}$ , with $t _ { b , \operatorname* { m a x } } ^ { \gamma }$ denoting the γ-th quantile of the worst-case delay distribution. The constraint set of problem (18) is particularly augmented with constraint (18b), which accounts for the inter-BS interference terms. Specifically, the decentralized network architecture requires careful management of inter-BS interference $\varphi _ { k } ^ { \mathrm { i n t e r } }$ , as discussed next.

## B. Channel Knowledge and Inter-BS Interference

Each device experiences both intra-BS and inter-BS interference, which influence the achievable rate in equation (3) and, consequently, the communication delay as a component of $t _ { k }$ (see equation (8)). While intra-BS interference $\varphi _ { k } ^ { \mathrm { i n t r a } }$ is directly controllable by the device’s serving BS, the inter-BS interference $\varphi _ { k } ^ { \mathrm { i n t e r } }$ originates from external sources and cannot be directly managed by the local BS. For illustration purposes, Fig. 3 shows an example of a network that includes two BSs, five XR devices, and BS-device links. These links are categorized as communication links (solid lines) or inter-BS interference links (dashed lines). For instance, based on Fig. 3, we observe the perspective of BS 1 performing resource allocation for its associated devices. From BS 1’s perspective, known channels are depicted in green, while unknown channels are shown in red. For example, BS 1 estimates its associated intra-BS channels $\hat { h } _ { 1 , 1 } , ~ \hat { h } _ { 1 , 2 } .$ , and $\hat { h } _ { 1 , 3 } ,$ as well as the inter-BS interfering channels $\hat { h } _ { 1 , 4 }$ and $\hat { h } _ { 1 , 5 }$ (i.e., the channels connecting BS 1 to XR devices 4 and 5), through uplink pilot measurements in time-division duplex mode or via user feedback in frequency-division duplex mode [46]. BS 1 is, therefore, able to compute $\varphi _ { 4 } ^ { \mathrm { i n t e r } }$ and $\varphi _ { 5 } ^ { \mathrm { i n t e r } }$ via (18b). Similarly, BS 2 computes $\varphi _ { 1 } ^ { \mathrm { i n t e r } } , \varphi _ { 2 } ^ { \mathrm { i n t e r } }$ , and $\varphi _ { 3 } ^ { \mathrm { i n t e r } }$ . Thus, ideally, each BS b informs all other BS of its induced inter-BS interference $\varphi _ { k } ^ { \mathrm { i n t e r } } , \forall k \in \mathcal { K } \backslash \mathcal { K } _ { b }$ , based on its locally estimated interfering channels. In networks with more than two BSs, each BS estimates its own set of cross-BS interference channel estimates, and then computes and transmits only its own portion of $\varphi _ { k } ^ { \mathrm { i n t e r } }$ . The CP, acting as a central parameter server, is then responsible for collecting all inter-BS interference terms, aggregating them, and distributing the aggregated results among all BSs.

A data-driven approach is adopted to address the inter-BS interference management. Exchanging inter-BS interference information for every channel estimation and each uncertainty sample would be computationally expensive and practically infeasible. Instead, our paper makes use of the mean inter-BS interference, referred to as $\bar { \varphi } _ { k }$ , which is computed and exchanged, aligning with the modular learning framework. This mean interference value can be shared at the end of each learning epoch during training. During inference, the mean inter-BS interference is exchanged initially and requires updates only when channel statistics exhibit significant changes.

## C. Decentralized DNN Architecture

To best empower the distributed FL implementation of the solution framework of problem (18), each BS hosts a local DNN to determine its associated devices’ resource allocation based on local channel estimates, computation requirement estimates, and mean inter-BS interference levels, as depicted in Fig. 2. At every BS $b ,$ the input layer consists of $2 L _ { b } K _ { b } + 2 K _ { b }$ and the output layer of $2 K _ { b } + 1$ neurons. The DNN then returns $\pmb { x } _ { b } = [ x _ { 1 } , \cdot \cdot \cdot , x _ { 2 K _ { b } + 1 } ] ^ { T }$ . Compared to the centralized architecture in Fig. 2, the distributed DNN (with modifications highlighted in green) considers only $K _ { b }$ users and the input layer is extended by additional $K _ { b }$ neurons that represent the mean inter-BS interference $\bar { \varphi } _ { k }$ . We let $\mathcal { F } _ { b , \Theta _ { b } }$ denote the proposed DNN for BS $b ,$ such that problem (18) is solved by

$$
\pmb { x } _ { b } = \mathcal { F } _ { b , \Theta _ { b } } ( \hat { \pmb { h } } _ { b } , \hat { \pmb { \omega } } _ { b } , \bar { \varphi } ) ,\tag{19}
$$

where $\hat { \pmb { h } } _ { b } = [ \hat { \pmb { h } } _ { b , 1 } ^ { T } , \cdot \cdot \cdot , \hat { \pmb { h } } _ { b , K } ^ { T } ] ^ { T }$ and $\hat { \pmb { \omega } } _ { b } = [ \hat { \omega } _ { 1 } , \cdot \cdot \cdot , \hat { \omega } _ { K _ { B } } ] ^ { T }$ Note that $\Theta _ { b }$ refers to the b’th BS’s DNN model parameters. We map the DNN output $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { b } }$ to $\mathbf { \Delta } _ { p _ { b } ^ { \mathrm { t x } } } ^ { p _ { b } ^ { \mathrm { t x } } }$ and $p _ { b } ^ { \mathrm { c o } }$ using equation (13) and to $f _ { b } ^ { \mathrm { p o w } }$ using equation (15). Afterwards, we inject both channel and computing state information error samples to compute the empirical γ-th quantile worst-case delay $\hat { t } _ { b , \operatorname* { m a x } } ^ { \bar { \gamma } }$ similar to the centralized solution.

## D. Federated Deep Learning for Robust Delay Optimization

Fundamentally, distributed optimization is best enabled in a network of multiple entities, e.g., BSs, that solve a global optimization problem, such as the one in (18), by using localized computations collectively [25]. Specifically, FL describes a distributed optimization methodology for collaborative AI model training. In the context of our paper, we consider the B BSs as clients, each hosting a local AI model for performing the communication and computation resource allocation for their associated XR devices. The BSs are connected to the CP, which hosts a global parameter server responsible for model aggregation, as can be seen in Fig. 1.

Our paper, therefore, solves problem (18) using a federated deep learning solution, which partially builds upon the steps of FedAvg-Algorithm $[ 4 7 ] ^ { 6 }$ . More precisely, initially, each BS performs local model updates using the gradient $\partial \dot { t } _ { b , \operatorname* { m a x } } ^ { \gamma } / \partial \Theta _ { b } ,$ based on their local channel and computing requirement estimations. Further, each BS computes its portion of the inter-BS interference $\varphi _ { k } ^ { \mathrm { i n t e r } } , \forall k \in \mathcal { K } \backslash \mathcal { K } _ { b }$ using (4). Then all BSs upload their local model parameters $\Theta _ { b }$ and inter-BS interference $\varphi _ { k } ^ { \mathrm { i n t e r } } , \forall k \in \mathcal { K } \backslash \mathcal { K } _ { b }$ to the CP. The CP aggregates the models to form the global model $\Theta ^ { \mathrm { g l o b a l } }$ by taking the average of $\Theta _ { b }$ $\forall b \in B$ as follows

$$
\Theta ^ { \mathrm { g l o b a l } } = \frac { 1 } { B } \sum _ { b \in B } \Theta _ { b } .\tag{20}
$$

Afterwards, the CP combines all inter-BS interference terms to obtain $\varphi _ { k } ^ { \mathrm { i n t e r } } , \forall k \in \mathcal { K }$ . In the last step, the CP broadcasts the global model parameters $\Theta ^ { \mathrm { g l o b a l } }$ and the inter-BS interference $\varphi _ { k } ^ { \mathrm { i n t e r } } , \forall k \in \mathcal { K }$ to all BSs. The whole process is repeated until a convergence criterion is met or the predefined number of training epochs is reached. The above steps are summarized in the description of Algorithm 2 below.

Algorithm 2 Proposed FL-based Training Procedure   
1: repeat   
2: for all BSs $b \in B$ do   
3: Local model update using gradient $\partial \hat { t } _ { b , \operatorname* { m a x } } ^ { \gamma } / \partial \Theta _ { b }$   
4: Compute portion of inter-BS interference $\varphi _ { k } ^ { \mathrm { i n t e r } }$ , k   
$\kappa \backslash \kappa _ { b }$ using (18b)   
5: Upload local model parameters $\Theta _ { b }$ and inter-BS   
interference $\varphi _ { k } ^ { \mathrm { i n t e r } } , \forall k \in \mathcal { K } \backslash \mathcal { K } _ { b }$ to CP   
6: end for   
7: Model aggregation at CP to form global model Θ<sup>global</sup>   
by taking average of $\Theta _ { b } , \forall b \in B ,$ via (20)   
8: CP adds inter-BS interferences to obtain $\varphi _ { k } ^ { \mathrm { i n t e r } } , \forall k \in \mathcal { K }$   
9: Broadcasting global model $\Theta ^ { \mathrm { g l o b a l } }$ and inter-BS inter  
ference $\varphi _ { k } ^ { \mathrm { i n t e r } } , \forall k \in { \cal K } .$ , to all BSs   
10: until Convergence

Note that the FedAvg algorithm, the steps of which are partially used in Algorithm 2, may not guarantee convergence under highly heterogeneous data distributions [48]. While a formal convergence analysis is beyond the scope of this paper, the simulations in Sec. V-G empirically demonstrate convergence within a practical number of epochs under uncertainty injection.

<sup>6</sup>Note that the original steps according to the FedAvg-Algorithm are marked in italic font in Algorithm 2 description.

## E. Data Exchange

This subsection analyzes the data exchange requirements across the system nodes, both for the proposed centralized (i.e., Algorithm 1) and for the proposed FL (i.e., Algorithm 2) approaches. In the centralized learning setup, a CP manages all BSs. In contrast, the FL setup relies on a common parameter server, while the BSs perform local operations independently.

The centralized learning approach requires all BSs $b \in B$ to upload each devices’ estimated channel coefficients $\hat { h } _ { b , l , k }$ $\forall ( l , k ) \ \in \ ( \mathcal { L } _ { b } , K )$ and estimated computing requirements $\hat { \omega } _ { b , k } , \ \forall k \in \mathcal { K }$ to the CP. In practical operations, each BS sends the channel coefficients and computation requirements for each new acquired channel or computing requirement estimation. Further, the CP broadcasts $ { p ^ { \mathrm { t x } } } ,  { p ^ { \mathrm { c o } } }$ , and $\pmb { f } ^ { \mathrm { c o } }$ . The learning procedure requires a number of E epoches, with M minibatches each, with a size of $\nu _ { m } .$ , and a validation size of $\nu _ { v } .$ . Additionally, only during training, for each channel and computing estimation, the BSs send N sampled channel and computing uncertainties to the CP. The channel coefficients are transmitted as complex floats with size $D _ { \mathrm { c f l o a t } } = 6 4$ bits, while the computing requirements, power loadings, and computing allocations are transmitted as floats with size $D _ { \mathrm { f l o a t } } = 3 2$ bits. The overall data exchange for centralized training becomes

$$
\Omega ^ { \mathsf { c e n } } = E N ( M \nu _ { m } + \nu _ { v } ) \left( ( 3 K + B ) D _ { \mathsf { f l o a t } } + L K D _ { \mathsf { c f l o a t } } \right)\tag{21}
$$

The formulation (21) captures the total data exchange required for centralized learning across multiple epochs and minibatches, factoring in parameter data sizes, the number of BSs and antennas, and the number of devices.

The FL approach facilitates local model updates at the BSs and the exchanged data originates from the i) broadcasting and iii) aggregation step. In each communication round, the CP broadcasts the most recent global model $\mathcal { F } _ { \Theta } ( \cdot )$ , with parameters Θ, and the average per-device inter-BS interference $\bar { \varphi } _ { k } , \forall k \in \mathcal { K }$ . In the aggregation step, each BS uploads its local DNN $\mathcal { F } _ { \Theta _ { b } } ^ { \mathrm { l o c a l } } ( \cdot )$ , with parameters $\Theta _ { b } .$ , and the locally-computed average inter-BS interference $\bar { \varphi } _ { k ^ { \prime } }$ to all $k ^ { \prime } \in \{ \mathcal { K } \backslash \mathcal { K } _ { b } \}$ . The DNN has a total of P parameters, consisting of $\eta _ { h }$ hidden layers, with $n _ { h }$ neurons each, $n _ { \mathrm { i n } }$ input layer neurons, and $n _ { \mathrm { o u t } }$ output layer neurons. The DNN parameter size consisting of weights and biases can thus be calculated as

$$
P = n _ { \mathrm { i n } } n _ { h } + ( \eta _ { h } - 1 ) n _ { h } ^ { 2 } + n _ { \mathrm { o u t } } n _ { h } + \eta _ { h } n _ { h } + n _ { \mathrm { o u t } } ,\tag{22}
$$

where the input and output layers consist of $n _ { \mathrm { i n } } = 2 K _ { b } ^ { 2 } +$ $2 K _ { b }$ and $n _ { \mathrm { o u t } } = 2 K _ { b } + 1$ neurons, respectively. Using P, the overall data exchange in FL, including both broadcasting and aggregation phases, becomes

$$
\begin{array} { r l } & { \Omega ^ { \mathrm { { f e d } } } = E \big ( ( P D _ { \mathrm { { f l o a t } } } + K D _ { \mathrm { { f l o a t } } } ) } \\ & { ~ + B ( P D _ { \mathrm { { f l o a t } } } + ( B - 1 ) K _ { b } D _ { \mathrm { { f l o a t } } } ) \big ) . } \end{array}\tag{23}
$$

A numerical comparison of data exchange requirements for both learning paradigms (i.e., Algorithm 1 and Algorithm 2) is further presented in Section V-I.

We note that uplink communication between devices and BSs (e.g., for channel estimation) is an inherent part of the underlying XR network. The communication overhead analysis above, instead, focuses on the additional overhead introduced by the learning framework, namely model parameter exchange and inter-BS information sharing between BSs and CP, which represent the dominant incremental communication cost.

<table><tr><td>General</td><td>Communication</td><td>Computation</td></tr><tr><td> $\overline { { P ^ { \mathrm { m a x } } = 3 4 ~ \mathrm { d B m } } }$ </td><td> $\overline { { W = 1 0 ~ \mathrm { M H z } } }$ </td><td> $\overline { { F ^ { \mathrm { m a x } } = 4 . 6 \cdot 1 0 ^ { 9 } ~ \mathrm { c y c l e s } / \mathrm { s } } }$ </td></tr><tr><td> $N = 1 0 0 0$  samples</td><td> $D _ { k } ^ { \mathrm { o u t } } = 7 . 5 { \cdot } 1 0 ^ { 4 }$  bits</td><td> $D _ { k } ^ { \mathrm { i n } } = 5 \cdot 1 0 ^ { 4 }$  bits</td></tr><tr><td> $\gamma = 0 . 0 5$ </td><td> $\alpha = 0 . 2 ~ \mathrm { R Z F }$  factor  $\sigma ^ { 2 } = \mathrm { - 1 7 5 ~ } \mathrm { d B m / H z }$ </td><td> $\tau = 1 0 ^ { - 2 8 } , \mu = 3$   $\omega _ { k } \sim \Gamma ( 2 , 2 0 0 ) , \forall k \in \mathcal { K }$ </td></tr></table>

Tab. I: Simulation parameters (unless otherwise mentioned).

## V. SIMULATION RESULTS AND DISCUSSION

In the following, we numerically evaluate the performance of the proposed FL-based solution with joint uncertainty injection. The network consists of $B \ = \ 3$ BS with $L _ { b } ~ = ~ 6$ antennas, $\forall b \in B .$ Unless otherwise mentioned, we place the three BS on a two dimensional plane at ((0.25, 0.25), (1.0, 0.25), (0.625, 0.625)) given in km with a height of 0.1 km. The devices are placed randomly around each BS with a radius of 0.25 km. The default number of XR devices is $K _ { b } = 5 , \forall b \in B$ . Other parameters are summarized in Tab. I. We model the channel coefficients as

$$
h _ { b , l , k } = c ( 4 \pi d _ { b , k } f _ { c } ) ^ { - 1 } F _ { b , l , k } ,\tag{24}
$$

with c denoting the speed of light, $d _ { b , k }$ denoting the distance between BS b and device k in meter, $f _ { c } = 3$ GHz denoting the carrier frequency, and $F _ { b , l , k }$ denoting the fading distribution [49]. We assume Rayleigh fading $F _ { b , l , k } \sim \mathcal { C N } ( 0 , 1 )$ $\forall ( b , l , k ) \in ( B , \mathcal { L } , \mathcal { K } )$ , utilize minimum mean-square estimation for the estimated channels $\hat { h } ,$ and set the beamforming vectors $\begin{array} { r } { { \boldsymbol { v } } _ { k } , \forall k \in { \mathcal { K } } } \end{array}$ , according to a regularized zero-forcing (RZF) beamforming strategy [7], fixing the beamforming vectors ${ \pmb v } _ { k }$ using: $\bar { \cal V } = \bar { \cal H } ( \hat { H } \hat { H } ^ { H } + \bar { \alpha } { \cal I } ) ^ { - 1 }$ , where $V =$ $[ \pmb { v } _ { 1 } , \cdots , \pmb { v } _ { K } ] , \hat { \pmb { H } } = [ \hat { \pmb { h } } _ { 1 } , \cdots , \hat { \pmb { h } } _ { K } ] .$ , α is the RZF factor, and I is a $K \times K$ identity matrix. Each ${ \pmb v } _ { k }$ is normalized to unit norm. As the beamforming vectors are determined a priori, we modify each DNN’s input layer, and only input effective channels, i.e., $\hat { h } _ { i , j } ^ { \mathrm { e f f } } = \hat { h } _ { i } ^ { H } \pmb { v } _ { j } ^ { \mathrm { - } } , \forall i , \bar { j } \in \mathcal { K }$ , thereby reducing the input layer to $2 K ^ { 2 } { + } K$ nodes for Algorithm 1 and $2 K _ { b } ^ { 2 } + 2 K _ { b }$ nodes for Algorithm 2. The ground-truth computation intensities are drawn according to a Gamma distribution [10]. For the DNN parameters and training procedure, we use $\eta _ { h } = 1 0$ fully-connected hidden layers with $n _ { h } ~ = ~ 4 0 0$ neurons each for Algorithm 1, and $\eta _ { h } = 7$ with $n _ { h } = 2 0 0$ for Algorithm 2. We apply ReLU activation and train for $E \ : = \ : 5 0 0$ epochs using $M \ = \ 5 0$ minibatches, each containing $\nu _ { m } ~ = ~ 1 0 0 0$ independent network realizations. Validation and test sizes are set to $\nu _ { v } ~ = ~ 2 0 0 0$ each, where each realization experiences uncertainty injection to obtain the robust objectives. We note that the adopted training procedure is implemented in an offline fashion. Such a procedure is in essence based on the channel and computing statistics, which are relatively slowvarying parameters as compared to the overall algorithmic training time. Such a characteristic of the proposed approach lessens the need for frequent training operations.

1) Benchmark Schemes: To benchmark the proposed robust FL-based scheme (referred to as Robust FL), i.e., Algorithm 2, we compare it against a regular deep learning scheme (Nonrobust FL), which is trained without uncertainty injection. Based on Algorithm 1, we also consider centralized worstcase delay minimization schemes as upper bounds to the decentralized learning approach, namely, a robust (Robust Cen) and a non-robust (Non-robust Cen) variant.

Additionally, we adopt three optimization-based schemes as benchmarks to best assess the performance of the proposed AI-native (data-based) algorithms. More specifically, the nonrobust version of problem (11), assuming perfect channel and computation knowledge, is solved using fractional programming (FP) by the benchmark scheme named Non-robust $F P$ [50]. The Uniform $F P$ scheme equally distributes transmit power and computing resources across devices, optimizing only the ratio of total computing to transmit power via FP [50]. Finally, the SAA/WMMSE scheme combines SAA and WMMSE to solve an adapted version of problem (11), where the quantile constraint (10) is replaced by sample averages [43].

In Section V-B, we benchmark joint uncertainty injection against two modified schemes: Comm UI FL, which accounts only for channel uncertainty, and Comp UI FL, which accounts only for computing uncertainty. Similarly, in Section V-D, we analyze the performance under heterogeneous uncertainty scenarios and benchmark our proposed schemes against $F e d \mathrm { - }$ Prox [51] and a local learning heuristic (hereafter denoted by Local).

For the sake of completeness, the detailed description of the above benchmarking optimization problems can be found in the appendix of the paper.

2) Practical Channel Estimation Error: Based on the model in [52], we assume block fading channels, with the XR devices carrying out channel estimation by averaging over N noisy pilot symbols. Furthermore, the power allocated for pilot transmission is not the same as for the actual transmission. The relation of both power allocations can be written as $p _ { s } = \beta p ^ { ( 1 - \gamma ) }$ , with $p _ { s }$ being the pilot symbol transmit power, where $\gamma$ and $\beta$ are system parameters. As can be seen in [53], the channel estimation error variance $\sigma _ { h } ^ { 2 }$ is a function of such system parameters given by

$$
\sigma _ { h } ^ { 2 } = p ^ { - 4 ( \gamma + 1 ) } \bigl ( 2 \beta N p ^ { 2 ( \gamma + 1 ) } + 1 \bigr ) ( \beta ^ { 2 } N ^ { 2 } ) ^ { - 1 } .\tag{25}
$$

With $\gamma = 0 . 2 , \beta = 0 . 4 .$ , and $N = L$ , the resulting channel estimation error variance $\sigma _ { h } ^ { 2 }$ spans the range shown in Fig. 4 by moderate variations of the underlying system parameters (e.g., pilot length, transmit power, and noise level). This shows how the considered values of $\sigma _ { h } ^ { 2 }$ correspond to realistic operating regimes rather than isolated parameter choices.

3) Practical Computing Estimation Error: The computation uncertainty parameter $\sigma _ { \omega } ^ { 2 }$ models variability in processing requirements due to dynamic XR workloads, such as field-ofview rendering and encoding. Since precise workload statistics depend on the application and system implementation [39], we consider a range of $\sigma _ { \omega } ^ { 2 }$ values in the simulations (see Fig. 5). The value $\sigma _ { \omega } ^ { 2 } = 3 \widetilde { 2 0 0 }$ is selected as a representative intermediate point, while the overall conclusions are drawn based on trends observed across different uncertainty levels.

## A. Impact of Communication and Computation Uncertainty

In the first set of simulations, we analyze the impact of channel and computing requirement uncertainties on the robust delay, $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ . To this end, Fig. 4 illustrates the robust delay as a function of the channel estimation error variance $\sigma _ { h } ^ { 2 } .$

![](images/06b4ffde1384eba29a521b0d0ff8be12fc27f97d2d6e93cac452b18bb8750290.jpg)  
Fig. 4: Robust delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ over channel uncertainty, i.e., $\sigma _ { h } ^ { 2 }$

while keeping the computation estimation error variance fixed at $\sigma _ { \omega } ^ { 2 } ~ = ~ 3 2 0 0$ . As a first observation, ML-based schemes (i.e., Non-robust $F L ,$ Non-robust Cen, Robust $F L ,$ and Robust Cen) outperform the baselines across all tested values, achieving delay reductions of 6 to 56 ms. Higher values of the channel uncertainty (e.g., in dynamic environments) even widens the performance gap between all ML-optimized schemes and the baselines. Fig. 4 shows that up to $\sigma _ { h } ^ { 2 } = 0 . 0 1 6$ SAA/WMMSE is the best-performing optimization baseline, achieving slightly lower $\hat { t } _ { \operatorname* { m a x } } ^ { \bar { \gamma } }$ than Non-robust FL. However, since SAA/WMMSE, as also illustrated in [43], relies on SAA, it provides robustness only in terms of the average performance across samples, which explains its inability to match the robust ML-based schemes. Among the illustrated ML-based methods is the fully centralized approach (Robust Cen) that consistently achieves the lowest robust delay. While Robust FL performs similarly at low $\sigma _ { h } ^ { 2 } .$ , the gap between the two schemes increases as channel uncertainty grows. Despite not matching Algorithm 1’s performance, the Robust FL (i.e., the scheme associated with Algorithm 2) offers a practical alternative with robust delay values close to the centralized scheme. Importantly, Fig. 4 highlights the advantage of robust (uncertainty-aware) learning: Both robust centralized and FL schemes significantly outperform their non-robust counterparts (i.e., Non-robust FL and Non-robust Cen). For instance, at $\sigma _ { h } ^ { 2 } = 0 . 0 0 8$ , Robust FL and Robust Cen achieve 18 ms and 12 ms lower robust delay, respectively, than their non-robust counterparts. Interestingly, uncertainty injection has a more pronounced impact on FL-based schemes, emphasizing its importance in distributed architectures. These findings underscore three significant insights: Uncertainty injection is crucial for minimizing robust delays in general, the proposed FLbased scheme performs closely to its centralized counterpart, and the joint uncertainty injection scheme is effective across varying uncertainty scenarios.

Fig. 5 reaffirms the advantages of Algorithm 2 and uncertainty injection by depicting the robust delay as a function of the computation estimation error variance $\sigma _ { \omega } ^ { 2 }$ with fixed $\sigma _ { h } ^ { 2 } \ = \ 0 . 0 2$ . Fig. 5 readily depicts how Uniform $F P$ consistently performs worse than all ML-based schemes at low $\sigma _ { \omega } ^ { 2 }$ . However, as computation uncertainty increases, non-robust ML-based schemes (Non-robust FL and Nonrobust Cen) become increasingly sensitive, ultimately losing their advantage over the uniform power allocation scheme at $\sigma _ { \omega } ^ { 2 } = 6 4 0 0$ for Non-robust FL and $\sigma _ { \omega } ^ { 2 } = 1 2 8 0 0$ for Non-robust

![](images/5effdd48a969b8c63ae55640144c309112658ea98dc4f99824d16169bf263603.jpg)  
Fig. 5: Robust delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ over computing uncertainty, i.e., $\sigma _ { \omega } ^ { 2 }$

![](images/ca323ab7b94e3fdb35b41c9c50b1bb136c2149c9d089475273bc77574beb88cc.jpg)  
Fig. 6: Robust delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ over computing uncertainty, i.e., $\sigma _ { \omega } ^ { 2 }$

Cen. Similarly, Non-robust FL also loses its advantage over SAA/WMMSE at $\sigma _ { \omega } ^ { 2 } = 1 2 8 0 0$ . This highlights that ML-based optimization without explicit uncertainty awareness does not guarantee robust performance. A deeper comparison with Fig. 4 reveals an interesting remark. While Robust FL is more severely affected by communication uncertainty, Fig. 5 shows that computation uncertainty has a more moderate impact. Robust FL maintains a consistent 14 ms gap from Robust Cen, demonstrating a relatively smaller degradation in highcomputation-uncertainty regimes. Such an observation is in fact reasonable, as distributed algorithms like Algorithm 2 are inherently more sensitive to communication uncertainties, affecting the interference levels but not the computation process.

Overall, both robust uncertainty injection schemes (Robust FL and Robust Cen) improve the network-wide robust delay performance, albeit depicting varying effectiveness across uncertainty conditions. Algorithm 2, particularly, maintains a stable performance gap toward the optimal centralized solution, outperforms non-robust baselines, and demonstrates strong practicality for real-world implementation.

## B. Advantages of Joint Uncertainty Injection

In the next set of simulations, we analyze the impact of two different uncertainty injection sub-schemes: Comm UI FL, which injects only channel uncertainty during training, and Comp UI FL, which injects only computing uncertainty. Fig. 6 depicts the robust delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ as a function of the computation estimation error variance $\sigma _ { \omega } ^ { 2 }$ , with $\sigma _ { h } ^ { 2 } = 0 . 0 1$ . It is first worthy to note that all uncertainty injection schemes outperform the non-robust baseline implementation (Non-robust $F L ) .$ , achieving delay reductions between 1 and 7 ms. Notably, Algorithm 2 (Robust FL), which leverages joint uncertainty injection, consistently achieves the lowest robust delay across all levels of uncertainty. Fig. 6 further highlights the adaptability of Algo rithm 2 across different uncertainty scenarios. For lower values of $\sigma _ { \omega } ^ { 2 }$ , where communication uncertainty dominates, Comm UI FL performs comparably to Robust FL, whereas Comp UI FL is less effective. Conversely, at higher computing uncertainties (e.g., $\sigma _ { \omega } ^ { 2 } = 8 0 0 )$ , Comp UI FL and Robust FL yield similar performances, while Comm UI FL suffers from significant delay increases. These findings underscore the effectiveness of joint uncertainty injection in improving network-wide robust delay performance, with varying effectiveness depending on the uncertainty type. Notably, Algorithm 2 not only generalizes both Comp UI FL and Comm UI FL but also consistently achieves the lowest robust delays, demonstrating its flexibility and superiority across diverse uncertainty conditions.

![](images/d11dcb7554e77018b8dd9c3cac1c4324275300baf1a8af763316dafc67e929c0.jpg)  
Fig. 7: Robust worst-case delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ vs. number of devices K.

## C. Scalability and Density

In this simulation set, we evaluate the algorithmic scalability by fixing the channel and computation estimation error variances to $\begin{array} { r l r } { \sigma _ { h } ^ { 2 } } & { { } = } & { 0 . 0 1 6 } \end{array}$ and $\begin{array} { r l r } { \sigma _ { \omega } ^ { 2 } } & { { } = } & { 3 2 0 0 , } \end{array}$ respectively. The scenario considers an expanded network with $\begin{array} { r l r } { B } & { { } = } & { 5 \quad \mathrm { B S s } . } \end{array}$ , positioned at ((0.25, 0.25), (1, 0.25), (0.625, 0.625), (0.25, 1.25), (1, 1.25)) in km, each equipped with $L _ { b } = 4$ antennas. The number of devices K is varied from 5 to 25, allowing us to analyze the scalability of the proposed learning-based approaches in dense networks. Fig. 7 illustrates the robust delay as a function of K, comparing the proposed FL-based and centralized uncertainty injection schemes against the uniform baseline. The results reveal that, as the number of devices increases, the proposed ML-based methods become increasingly more effective as compared to the uniform allocation scheme. For instance, at $K \ = \ 2 0$ , Robust FL achieves a robust delay reduction of 69 ms relative to the uniform power allocation (Uniform), while the robust centralized scheme (Robust Cen) reduces the delay by 88 ms. However, at high device densities, centralized solutions gain a larger advantage over FL-based approaches. The gap between Robust FL and Robust Cen widens with growing K, as the impact of the emerging interference in multi-device networks becomes more pronounced. At $K \ : = \ : 1 5$ , the gap between the two schemes is 6 ms, which increases to 19 ms at $K \ = \ 2 0$ . However, the computational complexity of Algorithm 1 grows significantly, to the extent that for $K = 2 5$ , the centralized solution, i.e., Robust Cen, becomes unable to determine a reliable solution within a reasonable time. This computational infeasibility region is marked in a shaded area in Fig. 7. When comparing uncertainty injection against non-robust ML-based resource allocation, Robust FL consistently outperforms Non-robust $F L ,$ , with the performance gap widening as the number of devices increases. While both schemes perform similarly at $K \ = \ 5 .$ , the delay gap increases to 7 ms at $K = 1 5$ and further to 13 ms at $K = 2 5 .$ These results underscore three key findings. First, ML-based resource allocation offers substantial performance gains in XR networks, especially at large numbers of XR devices. Second, while FL-based methods provide decentralized and scalable solutions, centralized learning remains superior in minimizing delay under high device densities. Finally, joint uncertainty-aware training plays a crucial role in maintaining robust performance, particularly in dynamic and interference-prone network environments.

![](images/f07c31f9df0d4796fc295b52613e3a70bb8a7e253094be5aa8b2e8079cd860c8.jpg)  
Fig. 8: Robust delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ over network density.

<table><tr><td rowspan="2">Network density Unit in ms</td><td colspan="2">sparse</td><td colspan="2">moderate</td><td colspan="2">dense</td></tr><tr><td> $t _ { \mathrm { i n f } } ^ { \mathrm { d n n } }$ </td><td> $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ </td><td> $t _ { \mathrm { i n f } } ^ { \mathrm { d n n } }$ </td><td> $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ </td><td> $t _ { \mathrm { i n f } } ^ { \mathrm { d n n } }$ </td><td> $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ </td></tr><tr><td>Non-robust FL</td><td>4.498</td><td>43.717</td><td>4.301</td><td>91.918</td><td>4.525</td><td>144.651</td></tr><tr><td>Robust FL</td><td>4.408</td><td>31.392</td><td>4.199</td><td>65.398</td><td>4.288</td><td>112.939</td></tr><tr><td>Non-robust FP</td><td> $8 . 5 { \cdot } 1 0 ^ { 3 }$ </td><td>59.847</td><td> $1 2 . 2 { \cdot } 1 0 ^ { 3 }$ </td><td>129.592</td><td> $2 0 . 2 { \cdot } 1 0 ^ { 3 }$ </td><td>212.580</td></tr><tr><td>SAA/WMMSE</td><td> $9 . 6 { \cdot } 1 0 ^ { 3 }$ </td><td>42.520</td><td> $3 8 . 6 { \cdot } 1 0 ^ { 3 }$ </td><td>96.609</td><td> $1 0 4 . 6 { \cdot } 1 0 ^ { 3 }$ </td><td>176.476</td></tr><tr><td>Uniform FP</td><td> $6 . 4 \cdot 1 0 ^ { 3 }$ </td><td>42.582</td><td> $9 . 9 { \cdot } 1 0 ^ { 3 }$ </td><td>100.187</td><td> $1 6 . 9 { \cdot } 1 0 ^ { 3 }$ </td><td>178.833</td></tr></table>

Tab. II: Mean inference/optimization time $t _ { \mathrm { i n f } } ^ { \mathrm { d n n } }$ for ML-based schemes and optimization baselines.

Along the above discussion, we analyze the behavior of Algorithm 2 in both its robust and non-robust implementations within networks of varying density. To this end, we increase the number of antennas per BS to $L _ { b } ~ = ~ 1 2$ while fixing the channel and computation estimation error variances at $\sigma _ { h } ^ { 2 } \ = \ 0 . 0 3$ and $\sigma _ { \omega } ^ { 2 } ~ = ~ 1 6 0 0$ , respectively. Fig. 8 illustrates the robust delay as a function of network density, where the x-axis represents three density levels, i.e., sparse, moderate, and dense, corresponding to $K _ { b } = 4 , K _ { b } = 8$ , and $K _ { b } = 1 2$ devices per BS. The results indicate that increasing network density significantly degrades the robust delay performance. For instance, in a sparse network, non-robust FL and robust FL achieve delays of 44 ms and 32 ms, respectively. In contrast, in a dense network, the robust delay increases to 145 ms for non-robust FL and 113 ms for robust FL. Tab. II additionally shows the mean inference/optimization time (denoted by $t _ { \mathrm { i n f } } ^ { \mathrm { d n n } } )$ ) alongside the robust delay for both FL-based schemes and optimization baselines for sparse, moderate, and dense networks. Tab. II particularly shows how the inference times of Robust FL and Non-robust FL remain in the order of 4-

![](images/b85fac9b29a0546522c3cdbe94331e775ba902601dddfa6029e152f4f370f849.jpg)  
Heterogenity of Estimation Errors $( \Delta _ { h } , \Delta _ { \omega } )$  
Fig. 9: Robust worst-case delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ vs. data heterogeneity for different FL implementations.

5 ms, whereas optimization-based schemes require several seconds per channel and computation estimation. Moreover, the inference time of the DNN-based schemes scales sublinearly, since only the input and output layers are affected by the number of devices. Interestingly, the gap between nonrobust FL and robust FL increases as the network density grows, rising from 12 ms in sparse networks to 32 ms in dense networks. This highlights the increasing importance of joint uncertainty injection, particularly in high-density scenarios, where interference and computational variations are more pronounced.

These results emphasize the growing importance of distributed uncertainty injection in XR-enabled 6G networks, particularly in dense deployments. As the network density increases, the computational complexity of centralized training algorithms becomes prohibitive, necessitating the use of our proposed distributed FL-based mechanisms to ensure scalability and resilience.

## D. Heterogeneous Uncertainties

To model non-IID local data across BSs, each BS is assigned a fixed estimation error variance, while the spread across BSs controls the heterogeneity level. Increasing this spread yields more diverse local uncertainty conditions while preserving the same average uncertainty. We retain the Gaussian error realizations but make their variances BS-dependent, i.e., $h _ { b , l , k } ^ { \mathrm { e r r } } \sim \mathcal { C N } ( 0 , \sigma _ { h , b } ^ { 2 } )$ and $\omega _ { k } ^ { \mathrm { e r r } } \sim \mathcal N ( 0 , \sigma _ { \omega , b } ^ { 2 } )$ , where $\sigma _ { h , b } ^ { 2 } \in$ $[ \sigma _ { h } ^ { 2 } - \Delta _ { h } / 2 , \sigma _ { h } ^ { 2 } + \Delta _ { h } / 2 ]$ and $\sigma _ { \omega , b } ^ { 2 } \in [ \sigma _ { \omega } ^ { 2 } - \Delta _ { \omega } / 2 , \sigma _ { \omega } ^ { 2 } + \Delta _ { \omega } / 2 ]$ For $B = 3 ,$ we use evenly spaced values within these intervals. The remaining simulation parameters are set to $\sigma _ { h } ^ { 2 } = 0 . 0 1 6 .$ $\sigma _ { \omega } ^ { 2 } = 3 2 0 0 $ , and $E = 1 0 0 0$ training epochs.

Fig. 9 shows the robust delay versus $( \Delta _ { h } , \Delta _ { \omega } )$ . As heterogeneity increases, all schemes experience performance degradation. Under homogeneous uncertainties, i.e., $( \Delta _ { h } , \Delta _ { \omega } ) =$ (0, 0), Robust Cen performs best, followed by Robust FL, while Robust Local and Robust FedProx perform slightly worse. The figure also shows that all non-robust schemes, particularly their federated variants, perform significantly worse. At high heterogeneity, i.e., $( \Delta _ { h } , \Delta _ { \omega } ) = ( 0 . 0 3 , 2 4 0 0 )$ , Robust Local outperforms both Robust FL and Robust FedProx by approximately 2 ms, while Robust Cen still achieves the overall lowest delay. These results indicate that Robust Cen remains effective under strong uncertainty heterogeneity with only minor performance loss. When the uncertainty levels across BSs are similar, Robust FL outperforms both Robust

![](images/b4a4ed318764d0e968d4e03668338dd30fd0bfc0fd145d1c499b6e3bfd8dcca9.jpg)  
Fig. 10: Robust worst-case delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ vs. parameter $\gamma ,$ showing impact of hard latency deadline guarantees.

FedProx and Robust Local, whereas under highly heterogeneous conditions, Robust Local becomes advantageous due to its fully independent model adaptation. While Robust FedProx does not achieve the best performance in this setup, tuning its proximal parameter $\delta$ allows interpolation between FedAvg $( \delta ~ = ~ 0 )$ and more constrained local updates, which may improve performance under specific heterogeneity regimes. Overall, the results show that highly heterogeneous uncertainty scenarios favor either centralized processing or, among distributed approaches, more localized model adaptation.

## E. Impact of Reliability Parameter γ

We investigate the impact of the reliability parameter $\gamma$ in the proposed γ-th quantile formulation, which directly controls the allowable probability of delay violations. The simulation setup uses $\sigma _ { h } ^ { \bar { 2 } } = 0 . 0 1 \bar { 6 } , \sigma _ { \omega } ^ { 2 } = \bar { 3 2 0 0 }$ , with training performed over 750 epochs. Fig. 10 illustrates the robust worst-case delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ as a function of $\gamma$ for the considered schemes. The results show that increasing $\gamma$ leads to lower delay for all methods, while smaller $\gamma$ values $( \mathrm { i . e . }$ , stricter reliability requirements) result in higher delay. Furthermore, the performance gap between robust and non-robust schemes becomes more pronounced $\mathrm { ~ a s ~ } \gamma$ decreases, highlighting the importance of uncertainty-aware designs under stringent latency constraints. It is also observed that Robust Cen is less sensitive to $\gamma$ and maintains more stable performance, whereas Robust $F L$ exhibits a stronger dependence on $\gamma ,$ leading to a larger performance gap at lower values. Overall, the results demonstrate a fundamental trade-off between delay and reliability, with γ serving as a tunable parameter to balance latency performance and deadline violation probability.

## F. Impact of Non-Gaussian Uncertainty Assumptions

To further validate the distribution-agnostic nature of the proposed framework, we consider Laplacian uncertainty, which exhibits heavier tails than the Gaussian distribution and captures rare but significant deviations (e.g., abrupt channel variations). Specifically, we model the error as $x \sim$ $\mathrm { L a p l a c e } ( 0 , \sigma _ { \mathrm { L a p l a c e } } )$ with variance matching the Gaussian case, i.e., $\sigma _ { \mathrm { L a p l a c e } } \in \dot { \{ \sqrt { \sigma _ { h } ^ { 2 } / 2 } }  , \sqrt { \sigma _ { \omega } ^ { 2 } / 2 } \}$ , for either channel or computing uncertainty. Fig. 11 shows the robust worst-case delay for different uncertainty levels. As expected, increasing uncertainty leads to higher delay. Compared to Gaussian uncertainty, the Laplacian model results in slightly higher delay due to its heavier tails. Importantly, when evaluating a Gaussiantrained model under Laplacian uncertainty, only a minor performance degradation of 0.5 ms is observed, indicating strong cross-distribution generalization. These results support the distribution-agnostic design of the proposed uncertainty injection framework.

![](images/56f8fd7172e8044208c1dee16f465e9d75c267fd28fa218cc2d07446c1986114.jpg)

Fig. 11: Robust worst-case delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ vs. three uncertainty levels ( low, medium, high corresponding to $( \sigma _ { h } ^ { 2 } , \sigma _ { \omega } ^ { 2 } ) ~ \in$ (0.004, 800), (0.008, 1600), (0.016, 3200) , respectively).  
![](images/b4d8211f31d2dcdfaa1a7eb725fd20882e05ab3d8f468866de3fc79bf1625efe.jpg)  
Fig. 12: Convergence behavior of centralized learning and FL.

## G. Convergence Behavior

Next, we analyze the convergence behavior of the proposed schemes, as shown in Fig. 12, by setting $\sigma _ { h } ^ { 2 } = 0 . 0 0 8 ,$ $\sigma _ { \omega } ^ { 2 } = 3 2 0 0 , B = 3 \ \mathrm { B S s }$ $L _ { b } ~ = ~ 6$ antennas per BS, and $K _ { b } = 5$ devices per BS over 500 training epochs. The figure presents the convergence of the robust centralized learning scheme Robust Cen and three curves representing the robust FL scheme Robust $F L ,$ where each curve corresponds to one BS. All schemes demonstrate convergence within the observed epochs. The key distinction lies in the comparison between Robust Cen and Robust FL. The centralized scheme, Robust Cen, optimizes for centralized uncertainty and the overall worst-case delay across the network, leading to a higher final objective value (i.e., larger delay). In contrast, FL optimizes local objectives at individual BSs while estimating inter-BS interference, resulting in lower final objective values per BS, but not necessarily for the global robust worst-case delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ . Additionally, Robust FL exhibits a sharp decline in the objective value between epochs 10 and 40, followed by another noticeable drop between epochs 200 and 400. This behavior suggests that Algorithm 2 efficiently refines the model in early stages and benefits from further adjustments in later epochs.

![](images/c219eaacc96c86305869fb5f4b752195d554aa7c749c0f0764717aebc70edd3f.jpg)

Fig. 13: Robust worst-case delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ vs. training epochs showing the impact of uncertainty samples N.
<table><tr><td>Training Epochs E</td><td>200</td><td>300</td><td>400</td><td>500</td></tr><tr><td> $\overline { { N = 5 0 0 } }$ </td><td>4429</td><td>6657</td><td>8885</td><td>11110</td></tr><tr><td> $N = 1 0 0 0$ </td><td>8593</td><td>12906</td><td>17222</td><td>21533</td></tr></table>

Tab. III: Training time in seconds for Robust FL.

## H. Impact of Uncertainty Samples

We investigate the impact of the number of uncertainty samples N used in the joint uncertainty injection scheme. The simulation parameters are set to $\sigma _ { h } ^ { 2 } = 0 . 0 1 6$ and $\sigma _ { \omega } ^ { 2 } = 3 2 0 0$ Fig. 13 shows the robust worst-case delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ as a function of the training epochs E for different values of N. We consider $N \in \{ 5 0 0 , 1 0 0 0 \}$ , where $N = 1 0 0 0$ corresponds to the default configuration used throughout the paper. The results indicate that increasing N improves both convergence stability and delay performance, as a larger number of samples provides a more accurate approximation of the underlying uncertainty distribution. In particular, choosing $N = 1 0 0 0$ consistently achieves a lower delay than $N \ = \ 5 0 0$ across all epochs. However, the performance gap remains relatively small, suggesting that a moderate number of samples already yields a sufficiently accurate approximation of the robust objective. To further quantify the computational overhead, Tab. III reports the corresponding training times. As expected, increasing the number of uncertainty samples from $N = 5 0 0$ to $N = 1 0 0 0$ approximately doubles the training time, since the number of processed uncertainty realizations scales linearly with N. These findings highlight a trade-off between robustness and computational complexity, while also confirming that the proposed framework is not overly sensitive to the exact choice of N, i.e., it can operate efficiently with moderate sample sizes.

## I. Training and Inference Data Exchange

The proposed centralized and FL-based schemes rely on the exchange of data through limited-capacity backhaul network, which connects BSs and CP. The total data exchange required for ML training is depicted in Fig. 14, as a function of the number of BSs for different values of devices per BS. The y-axis of Fig. 14 represents the training data exchange (in megabytes (MB)) for both centralized (i.e., Algorithm 1) and federated (i.e., Algorithm 2) implementations. While the data exchange required for training Algorithm 2 remains within the range of 800-4000 MB, the centralized approach requires up to 495000000 MB $\left( \mathrm { i . e . , 4 9 5 \ T B } \right)$ when the network consists of 8 BSs with $K _ { b } = 6$ . Furthermore, the required data exchange in the centralized implementation grows significantly faster than in the FL-based schemes. Algorithm 1 does not require exchanging the DNN model with the CP but necessitates frequent transmissions of local channel and computation estimations for each training/testing realization. Algorithm 2 only transmits DNN model parameters and inter-BS interference levels, keeping excessive estimations local. Moreover, Fig. 14 highlights the impact of increasing $K _ { b }$ , which further amplifies data exchange requirements in both schemes.

![](images/3f1a8b6e4141e30fede579491c7dd945f8286d5a9e452fc0b6daca1470932314.jpg)

Fig. 14: Exchanged data during training in MegaByte (MB) versus the number of BSs for different device numbers.  
![](images/9817bc421967d5164cc2528d6beff9999acee0e16b1562f6145ff7d0b1b96834.jpg)  
Fig. 15: Exchanged data for inference in kiloByte (kB) versus the number of BSs for different device numbers.

The limited-capacity BS-CP backhaul network may become a major bottleneck in FL training. This overhead can be reduced through communication-efficient FL techniques such as probabilistic device selection, parameter quantization, model compression, or sparse/partial model updates [54], [55]. Such approaches reduce the number of transmitted bits per aggregation round, at the cost of introducing approximation errors into the local or aggregated model updates. Hence, overly aggressive compression may slow convergence or degrade the learned resource-allocation policy, potentially increasing the resulting robust delay. Investigating this communicationrobustness trade-off is an interesting direction for future work.

In practical implementations, the training phase constitutes only a fraction of the overall operational data exchange. Fig. 15 illustrates the inference-phase data exchange requirements (in kilobytes (kB)) versus the number of BSs for different values of $K _ { b } .$ . Both centralized and FL-based schemes initially exhibit low inference data exchange requirements. However, as the number of BSs increases, the centralized scheme’s data exchange grows exponentially, while the federated data exchange exhibits a much slower increase. Specifically, at $B = 8$ and $K _ { b } = 6 .$ Fig. 15 shows that the centralized approach requires 19 kB, whereas the distributed one requires only 1.5 kB.

![](images/eaf0e9a6deb911623b75bd0384e0a1d5e49a1e770b4454d0e8a037c3403a9e37.jpg)

![](images/500b060d96eff28532d5c52d5eba712ae753228bf3cc7843bf8e5410ef590574.jpg)  
Number of Neurons per Hidden Layer n<sub>h</sub>  
Fig. 16: Robust worst-case delay $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ vs. neurons per hidden layer $n _ { h }$ (left) and number of hidden layers $\eta _ { h }$ (right).

Additionally, the impact of higher $K _ { b }$ is more pronounced in the centralized scheme. These findings emphasize that the proposed FL-based robust delay minimization approach requires significantly less data exchange in both training and inference phases, making it a more practical and scalable solution for future data-driven, distributed XR systems.

## J. Ablation Study

As a last simulation set, we investigate the impact of the DNN architecture on the achieved robustness performance. The simulation parameters are set to $\sigma _ { h } ^ { 2 } = 0 . 0 1 6$ and $\sigma _ { \omega } ^ { 2 } =$ 3200. Fig. 16 illustrates the robust worst-case delay $\hat { t } _ { \mathrm { m a x } } ^ { \gamma }$ as a function of the number of neurons per hidden layer $n _ { h }$ (left) and the number of hidden layers $\eta _ { h }$ (right). For the lefthand side of Fig. 16, the number of hidden layers is fixed to $\eta _ { h } ~ = ~ 1 0$ for Robust Cen and $\eta _ { h } ~ = ~ 7$ for Robust $F L$ The results of Fig. 16 show that increasing the number of neurons generally improves the delay performance for both schemes. However, saturation effects can be observed for larger network sizes, where increasing $n _ { h }$ beyond a certain point provides only marginal gains. Furthermore, Robust Cen benefits more strongly from larger DNN sizes, reflecting the increased complexity of the centralized optimization task. For the right-hand side of Fig. 16, the number of neurons per hidden layer is fixed to $n _ { h } ~ = ~ 2 0 0$ , while the number of hidden layers $\eta _ { h }$ is varied for the Robust FL scheme. The results show that deeper architectures may negatively affect performance when the training duration is insufficient, whereas for longer training durations, the impact of $\eta _ { h }$ becomes less significant. Overall, the results highlight a trade-off between model complexity, training effort, and achieved robustness performance.

## K. Edge Deployment Feasibility

To further assess the practical deployment feasibility of the proposed DNN-based resource-allocation schemes, we evaluate their inference latency on edge hardware. The measurements are conducted on an NVIDIA Jetson Orin Nano Developer Kit Super operated in 25 W power mode, running JetPack 7.2.1, CUDA 13.2, Python 3.12.3, and Py-Torch 2.12.0+cu132. The platform is equipped with a 1024- core NVIDIA Ampere GPU, 32 Tensor Cores, and 8 GB of unified LPDDR5 memory. Inference is performed with batch size “one” to reflect online resource allocation for a single instantaneous network state. After 50 warm-up runs, the average inference latency of Robust Cen is 9.578 ms with a standard deviation of 0.12 ms, while Robust FL achieves an average inference latency of 4.587 ms with a standard deviation of 0.06 ms. These results demonstrate that both architectures support millisecond-scale inference on resource-constrained edge hardware. In particular, the lower latency of Robust FL reflects the reduced input dimension and smaller local DNN architecture compared with the centralized scheme. Hence, once trained offline, the proposed models can be deployed for online resource-allocation decisions without requiring computationally expensive optimization during operation.

## VI. CONCLUSION

Automating the decision-making via FL and distributed intelligence emerges as a key enabler toward seamless future XR-based networks. This paper presents a novel deep learningbased approach to jointly optimize the communication and computation resources in both centralized and federated settings while accounting for uncertainties in channel estimation and computing requirements. The proposed robust framework manages transmit power, computing power, and computation allocation through a DNN trained with uncertainty injection. By jointly incorporating uncertainty samples during training and optimizing resource allocations based on the robust utility, our approach effectively minimizes the worst-case delay across multiple devices. Our findings highlight the benefits of joint uncertainty injection, demonstrating that FL-based solutions achieve near-centralized performance while significantly enhancing scalability, efficiency, and robustness. These results particularly highlight the potential of AI-driven, multi-purpose, distributed resource management for real-world XR networks. The results of the current paper further trigger a handful of future research directions in the field, e.g., optimizing other objectives of practical interest within the joint communication and computation interplay, considering hybrid and online learning protocols, addressing different system models of emerging interest within the context of non-terrestrial networks, the impact of stale interference information (among others).

## APPENDIX

For the sake of completeness, this appendix presents a handful of data-driven as well as numerical optimization baselines, which are used as benchmarking schemes in the simulations section of the paper. The methods, hereafter called Non-robust Cen, Non-robust FL, Non-robust FP, Uniform FP, SAA/WMMSE, FedProx, and Local, rely on specific formulations of the optimization problem, as discussed next.

## A. Non-Robust Centralized Learning

The non-robust counterpart of problem (11) rather neglects the channel and computing uncertainties in (1) and (5), and minimizes the nominal worst-case delay $t _ { \mathrm { m a x } }$ among all devices:

$$
\begin{array} { r l } { \underset { { p ^ { \mathrm { t x } } , p ^ { \mathrm { c o } } , f ^ { \mathrm { c o } } } } { \mathrm { m i n } } } & { \underset { { k \in \mathcal { K } } } { \mathrm { m a x } } \{ t _ { k } \} } \\ { \mathrm { s . t . } } & { ( 6 ) , ( 9 ) , } \end{array}\tag{26}
$$

with computation capacity (6) and power constraint (9). Unlike the robust formulation (11), the γ-th quantile constraint (10) is omitted here. The Non-robust Cen benchmark scheme can be

derived from problem (26) by applying Algorithm 1 without uncertainty injection, i.e., by replacing $\hat { t } _ { \operatorname* { m a x } } ^ { \gamma }$ with $\operatorname* { m a x } _ { k \in \mathcal { K } } \{ t _ { k } \}$

## B. Non-Robust Federated Learning

The non-robust version of the distributed problem (18) neglects channel and computing uncertainties, and minimizes the nominal worst-case delay $t _ { b , \operatorname* { m a x } }$ among all devices associated with BS b:

$$
\begin{array} { r l } { \underset { { p _ { b } ^ { \mathrm { t x } } , p _ { b } ^ { \mathrm { c o } } , f _ { b } ^ { \mathrm { c o } } } } { \mathrm { m i n } } } & { \underset { { k \in \mathcal { K } _ { b } } } { \mathrm { m a x } } \left. t _ { k } \right. } \\ { \mathrm { s . t . } } & { ( 6 ) , ( 9 ) , ( 1 8 b ) } \end{array}\tag{27}
$$

with computation capacity (6) constraint, power constraint (9), and inter-BS interference constraint (18b). The Non-robust FL benchmark scheme can be derived from problem (27) by applying Algorithm 2 without uncertainty injection.

## C. Non-Robust Numerical Optimization

To compare the data-driven ML-based schemes to a numerical optimization approach, Non-robust FP utilizes FP [50] to solve problem (26) using standard solvers (e.g., CVX [56]).

## D. Uniform Numerical Optimization

A modified optimization formulation based on problem (26) with additional constraints enforcing equal transmit power and computation cycles is formulated as:

$$
\begin{array} { r l r } { \underset { p ^ { \mathrm { t x } } , p ^ { \mathrm { c o } } , f ^ { \mathrm { c o } } } { \operatorname* { m i n } } } & { \underset { k \in \mathcal { K } } { \operatorname* { m a x } } \{ t _ { k } \} } & \\ { \mathrm { s . t . } } & { ( 6 ) , ( 9 ) , } & \\ & { } & { p _ { k } ^ { \mathrm { t x } } = p _ { k ^ { \prime } } ^ { \mathrm { t x } } , ~ f _ { k } ^ { \mathrm { c o } } = f _ { k ^ { \prime } } ^ { \mathrm { c o } } , ~ \forall k \not = k ^ { \prime } \in \mathcal { K } . } \end{array}\tag{28}
$$

(28a)

Constraint (28a) enforces all devices to be served with the same transmit power $p _ { k } ^ { \mathrm { t x } }$ and computing allocation $f _ { k } ^ { \mathrm { c o } } , \forall k \in$ . Uniform FP then solves (28) using FP [50] optimizing solely the power split between transmission and computation.

## E. Robust Numerical Optimization

Finally, for the SAA/WMMSE scheme, similar to the uncertainty injection scheme proposed in Sec. III-B, N realizations of channel and computing errors $\boldsymbol { h } _ { n } ^ { \mathrm { e r r } }$ and $\omega _ { n } ^ { \mathrm { e r r } }$ $n \in \{ 1 , \ldots , N \}$ , are sampled, yielding N worst-case delays $t _ { \mathrm { m a x } , n }$ as in (16). The resulting SAA-based problem is then formulated as follows:

$$
\begin{array} { r l } { \underset { p ^ { \mathrm { t x } } , p ^ { \mathrm { c o } } , f ^ { \mathrm { c o } } } { \operatorname* { m i n } } } & { t _ { \mathrm { m a x } } ^ { \mathrm { S A A } } } \\ { \mathrm { s . t . } } & { ( 1 ) , ( 5 ) , ( 6 ) , ( 9 ) , } \\ & { t _ { \mathrm { m a x } } ^ { \mathrm { S A A } } = \displaystyle \frac { 1 } { N } \sum _ { n = 1 } ^ { N } t _ { \mathrm { m a x } , n } , } \end{array}\tag{29}
$$

(29a)

where the γ-th quantile constraint (10) is replaced by the SAA constraint (29a). Following [43], problem (29) can then be solved via an SAA and WMMSE-based method.

## F. Robust FedProx-based Learning

The work [51] proposes the Federated Proximal (FedProx) algorithm to address optimization in heterogeneous networks. The key idea is that, in each communication round, all agents receive the global model, use it to initialize their local models, and perform local training with an additional proximal penalty term that discourages large deviations from the global model.

To adopt the FedProx algorithm for solving problem (18), we modify the procedure of Algorithm 2 as follows. In particular, the local model updates in Step 3 of Algorithm 2 are augmented by a modified gradient $\partial \mathcal { L } _ { b } / \partial \Theta _ { b }$ , based on the following loss function:

$$
\mathcal { L } _ { b } = \hat { t } _ { b , \operatorname* { m a x } } ^ { \gamma } + \frac { \delta } { 2 } \| \Theta _ { b } - \Theta ^ { \mathrm { g l o b a l } } \| ^ { 2 } .\tag{30}
$$

Here, δ is a regularization parameter that controls how strongly the local model is pulled toward the global model, i.e., it penalizes local model drift. During inference, all BSs use the final aggregated global model, ensuring consistent and generalizable performance across the network. In our implementation, we set $\delta \ : = \ : 1 0 ^ { - 3 }$ , which provides a trade-off between the standard FedAvg behavior (small δ) and a strongly constrained local update (large δ).

Extensions toward heterogeneous model architectures, e.g., HeteroFL [57], may further improve scalability by adapting model complexity to BS capabilities, but are beyond the scope of this work.

## G. Fully Local Learning

As an additional benchmark, particularly for heterogeneous network scenarios, we consider a fully local learning approach. The key idea is to omit the global model aggregation step and maintain separate models at each BS. The Local learning scheme is obtained as a modified version of Algorithm 2, where the global aggregation in Step 7 is omitted. Consequently, Steps 5 and 9 of Algorithm 2 are performed without the use of a global model Θ<sup>global</sup>, i.e., each BS updates and maintains its own local model independently. Note that this local learning scheme still relies on computing, collecting, and exchanging inter-BS interference information.

## REFERENCES

[1] R.-J. Reifert, H. Dahrouj, A. A. Ahmad, H. Gacanin, and A. Sezgin, “Robust communication and computation using deep learning via joint uncertainty injection,” in Proc. ISWCS, 2024, pp. 1–6.

[2] “Exploring consumer expectations and needs in an evolving XR landscape,” Ericsson ConsumerLab, Tech. Rep., 2024. [Online]. Available: https://www.ericsson.com/4a679a/assets/local/reports-papers/ consumerlab/reports/2024/augmented-tomorrow.pdf

[3] I. F. Akyildiz and H. Guo, “Wireless communication research challenges for extended reality (XR),” ITU Journal on Future and Evolving Technologies, vol. 3, no. 1, pp. 1–15, 2022.

[4] A. Cedergren, F. de Laval, S. Falahati, L. Falk, D. H. Kang, R. S. Karlsson, Y. Lyazidi, A. Mirzaei, J. Olsson, J. L. Pradas, P. Schliwa-Bertling, N. Shi, B. Singh, R. Tano, and A. M. Voicu, “5G networks evolution for extended reality,” IIEEE Commun. Stand. Mag., vol. 8, no. 3, pp. 54–59, 2024.

[5] F. Guo, F. R. Yu, H. Zhang, X. Li, H. Ji, and V. C. M. Leung, “Enabling massive IoT toward 6G: A comprehensive survey,” IEEE Internet Things J., vol. 8, no. 15, pp. 11 891–11 915, 2021.

[6] IMT-2030 Framework: Recommendation ITU-R M.2160-0, International Telecommunication Union (ITU), ITU-R Working Party 5D, Nov 2023.

[7] W. Cui and W. Yu, “Uncertainty injection: A deep learning method for robust optimization,” IEEE Trans. Wirel. Commun., vol. 22, no. 11, pp. 7201–7213, 2023.

[8] R.-J. Reifert, H. Dahrouj, and A. Sezgin, “Extended reality via cooperative NOMA in hybrid cloud/mobile-edge computing networks,” IEEE Internet Things J., pp. 1–1, 2023.

[9] A. J. Anandkumar, A. Anandkumar, S. Lambotharan, and J. A. Chambers, “Robust rate maximization game under bounded channel uncertainty,” IEEE Trans. Veh. Technol., vol. 60, no. 9, pp. 4471–4486, 2011.

[10] S. Li, C. Li, Y. Huang, B. A. Jalaian, Y. T. Hou, and W. Lou, “Enhancing resilience in mobile edge computing under processing uncertainty,” IEEE J. Sel. Areas Commun., vol. 41, no. 3, pp. 659–674, 2023.

[11] Z. Liu, C. Zhan, Y. Cui, C. Wu, and H. Hu, “Robust edge computing in UAV systems via scalable computing and cooperative computing,” IEEE Wirel. Commun., vol. 28, no. 5, pp. 36–42, 2021.

[12] R.-J. Reifert, A. A. Ahmad, H. Dahrouj, A. Chaaban, A. Sezgin, T. Y. Al-Naffouri, and M.-S. Alouini, “Distributed resource management in downlink cache-enabled multi-cloud radio access networks,” IEEE Trans. Veh. Technol., pp. 1–16, 2022.

[13] K. S. Tharakan, H. Dahrouj, N. Kouzayha, H. Elsawy, and T. Y. Al-Naffouri, “Personalized federated learning for cellular VR: Online learning and dynamic caching,” IEEE Trans. Commun., pp. 1–1, 2025.

[14] Y. Liu, X. Zhu, J. Wang, and J. Xiao, “A quantitative metric for privacy leakage in federated learning,” in Proc. IEEE ICASSP, 2021, pp. 3065– 3069.

[15] A. R. Elkordy, J. Zhang, Y. H. Ezzeldin, K. Psounis, and S. Avestimehr, “How much privacy does federated learning with secure aggregation guarantee?” arXiv preprint arXiv:2208.02304, 2022.

[16] P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings et al., “Advances and open problems in federated learning,” Found. Trends Mach. Learn., vol. 14, no. 1–2, pp. 1–210, 2021.

[17] M. Liubogoshchev, K. Ragimova, A. Lyakhov, S. Tang, and E. Khorov, “Adaptive cloud-based extended reality: Modeling and optimization,” IEEE Access, vol. 9, pp. 35 287–35 299, 2021.

[18] N. Q. Hieu, D. N. Nguyen, D. T. Hoang, and E. Dutkiewicz, “When virtual reality meets rate splitting multiple access: A joint communication and computation approach,” IEEE J. Sel. Areas Commun., vol. 41, no. 5, pp. 1536–1548, 2023.

[19] M. Hasan, E. Hossain, and D. I. Kim, “Resource allocation under channel uncertainties for relay-aided device-to-device communication underlaying LTE-A cellular networks,” IEEE Trans. Wirel. Commun., vol. 13, no. 4, pp. 2322–2338, 2014.

[20] S. Tamoor-ul Hassan, S. Samarakoon, M. Bennis, and M. Latva-aho, “Latency-aware radio resource optimization in learning-based cloudaided small cell wireless networks,” IEEE Trans. Green Commun. Netw., vol. 8, no. 1, pp. 542–558, 2024.

[21] M. Chen, U. Challita, W. Saad, C. Yin, and M. Debbah, “Artificial neural networks-based machine learning for wireless networks: A tutorial,” IEEE Commun. Surv. Tutor., vol. 21, no. 4, pp. 3039–3071, 2019.

[22] X. Mo and J. Xu, “Energy-efficient federated edge learning with joint communication and computation design,” J. Commun. Inf. Netw., vol. 6, no. 2, pp. 110–124, 2021.

[23] S. Jere, Y. Song, Y. Yi, and L. Liu, “Distributed learning meets 6G: A communication and computing perspective,” IEEE Wirel. Commun., vol. 30, no. 1, pp. 112–117, 2023.

[24] S. Niknam, H. S. Dhillon, and J. H. Reed, “Federated learning for wireless communications: Motivation, opportunities, and challenges,” IEEE Commun. Mag., vol. 58, no. 6, pp. 46–51, 2020.

[25] Y.-F. Liu, T.-H. Chang, M. Hong, Z. Wu, A. Man-Cho So, E. A. Jorswieck, and W. Yu, “A survey of recent advances in optimization methods for wireless communications,” IEEE J. Sel. Areas Commun., vol. 42, no. 11, pp. 2992–3031, 2024.

[26] S. Samarakoon, M. Bennis, W. Saad, and M. Debbah, “Distributed federated learning for ultra-reliable low-latency vehicular communications,” IEEE Trans. Commun., vol. 68, no. 2, pp. 1146–1159, 2020.

[27] P. S. Bouzinis, P. D. Diamantoulakis, and G. K. Karagiannidis, “Wireless quantized federated learning: A joint computation and communication design,” IEEE Trans. Commun., vol. 71, no. 5, pp. 2756–2770, 2023.

[28] M. Chen, Z. Yang, W. Saad, C. Yin, H. V. Poor, and S. Cui, “A joint learning and communications framework for federated learning over wireless networks,” IEEE Trans. Wirel. Commun., vol. 20, no. 1, pp. 269–283, 2021.

[29] M. M. Wadu, S. Samarakoon, and M. Bennis, “Federated learning under channel uncertainty: Joint client scheduling and resource allocation,” in Proc. IEEE WCNC, 2020, pp. 1–6.

[30] E. A. Jorswieck, “Next-generation multiple access: From basic principles to modern architectures,” Proc. IEEE, vol. 112, no. 9, pp. 1149–1178, 2024.

[31] S. Mangiante, G. Klas, A. Navon, Z. GuanHua, J. Ran, and M. D. Silva, “VR is on the edge: How to deliver 360 videos in mobile networks,” in Proceedings of the Workshop on Virtual Reality and Augmented Reality Network, 2017, pp. 30–35.

[32] S. Loyka and C. D. Charalambous, “On the compound capacity of a class of MIMO channels subject to normed uncertainty,” IEEE Trans. Inf. Theory, vol. 58, no. 4, pp. 2048–2063, 2012.

[33] T. Yoo and A. Goldsmith, “On the optimality of multiantenna broadcast scheduling using zero-forcing beamforming,” IEEE Journal on Selected Areas in Communications, vol. 24, no. 3, pp. 528–541, 2006.

[34] H. Ye, G. Y. Li, and B.-H. F. Juang, “Deep reinforcement learning based resource allocation for V2V communications,” IEEE Transactions on Vehicular Technology, vol. 68, no. 4, pp. 3163–3173, 2019.

[35] 5G; NR; Physical channels and modulation (3GPP TS 38.211 version 16.2.0 Release 16), ETSI, 7 2020, version 16.2.0 Release 16.

[36] H. Dahrouj, S. Liu, and M.-S. Alouini, “Machine learning-based user scheduling in integrated satellite-HAPS-ground networks,” IEEE Network, vol. 37, no. 2, pp. 102–109, 2023.

[37] H. Sun, X. Chen, Q. Shi, M. Hong, X. Fu, and N. D. Sidiropoulos, “Learning to optimize: Training deep neural networks for wireless resource management,” in 2017 IEEE 18th International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), 2017, pp. 1–6.

[38] W. Yuan and K. Nahrstedt, “Energy-efficient soft real-time CPU scheduling for mobile multimedia systems,” ACM SIGOPS Operating Systems Review, vol. 37, no. 5, pp. 149–163, 2003.

[39] J. R. Lorch and A. J. Smith, “Improving dynamic voltage scaling algorithms with PACE,” ACM SIGMETRICS Performance Evaluation Review, vol. 29, no. 1, pp. 50–61, 2001.

[40] X. Ge, Y. Sun, H. Gharavi, and J. Thompson, “Joint optimization of computation and communication power in multi-user massive MIMO systems,” IEEE Trans. Wirel. Commun., vol. 17, no. 6, pp. 4051–4063, 2018.

[41] L. Huang, W. Pan, Y. Zhang, L. Qian, N. Gao, and Y. Wu, “Data augmentation for deep learning-based radio modulation classification,” IEEE Access, vol. 8, pp. 1498–1506, 2020.

[42] U. Shaham, Y. Yamada, and S. Negahban, “Understanding adversarial training: Increasing local stability of supervised models through robust optimization,” Neurocomputing, vol. 307, pp. 195–204, 2018.

[43] R.-J. Reifert, A. Alameer Ahmad, Y. Mao, A. Sezgin, and B. Clerckx, “Rate-splitting multiple access in cache-aided cloud-radio access networks,” Front. Commun. Netw., vol. 2, p. 716620, 2021.

[44] I. Goodfellow, Y. Bengio, A. Courville, and Y. Bengio, Deep learning. MIT press Cambridge, 2016, vol. 1, no. 2.

[45] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al., “Pytorch: An imperative style, high-performance deep learning library,” Advances in neural information processing systems, vol. 32, 2019.

[46] J. Flordelis, F. Rusek, F. Tufvesson, E. G. Larsson, and O. Edfors, “Massive MIMO performance—TDD versus FDD: What do measurements say?” IEEE Trans. Wirel. Commun., vol. 17, no. 4, pp. 2247–2261, 2018.

[47] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Artificial intelligence and statistics. PMLR, 2017, pp. 1273– 1282.

[48] X. Li, K. Huang, W. Yang, S. Wang, and Z. Zhang, “On the convergence of fedavg on non-iid data,” arXiv preprint arXiv:1907.02189, 2019.

[49] A. Alsharoa and M.-S. Alouini, “Improvement of the global connectivity using integrated satellite-airborne-terrestrial networks with resource optimization,” IEEE Trans. Wirel. Commun., vol. 19, no. 8, pp. 5088–5100, 2020.

[50] K. Shen and W. Yu, “Fractional programming for communication systems—part I: Power control and beamforming,” IEEE Trans. Signal Process., vol. 66, no. 10, pp. 2616–2630, 2018.

[51] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” in Proceedings of Machine Learning and Systems, I. Dhillon, D. Papailiopoulos, and V. Sze, Eds., vol. 2, 2020, pp. 429–450.

[52] D. S. Michalopoulos, N. D. Chatzidiamantis, R. Schober, and G. K. Karagiannidis, “The diversity potential of relay selection with practical channel estimation,” IEEE Trans. Wirel. Commun., vol. 12, no. 2, pp. 481–493, 2013.

[53] M. A. Hasabelnaby and A. Chaaban, “Energy-efficient multi-pair computation for intra cell-free massive MIMO communications,” IEEE Wirel. Commun. Lett., vol. 13, no. 3, pp. 776–780, 2024.

[54] M. Chen, N. Shlezinger, H. V. Poor, Y. C. Eldar, and S. Cui, “Communication-efficient federated learning,” Proceedings of the National Academy of Sciences, vol. 118, no. 17, p. e2024789118, 2021. [Online]. Available: https://www.pnas.org/doi/abs/10.1073/pnas. 2024789118

[55] M. Kim, W. Saad, M. Debbah, and C. S. Hong, “SpaFL: Communication-efficient federated learning with sparse models and low computational overhead,” Advances in Neural Information Processing Systems, vol. 37, pp. 86 500–86 527, 2024.

[56] M. Grant and S. Boyd, “CVX: Matlab software for disciplined convex programming, version 2.1,” 2014. [Online]. Available: http: //cvxr.com/cvx

[57] E. Diao, J. Ding, and V. Tarokh, “HeteroFL: Computation and communication efficient federated learning for heterogeneous clients,” arXiv preprint arXiv:2010.01264, 2020.