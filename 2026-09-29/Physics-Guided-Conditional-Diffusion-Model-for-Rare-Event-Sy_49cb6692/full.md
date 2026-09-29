# Physics-Guided Conditional Diffusion Model for Rare Event Synthesis and Diagnosis for the Water-Gas Shift Reaction

Md Abrar Rafid Siddique<sup>†</sup>, Bibek Aryal<sup>†</sup>, and Qiugang Lu<sup>†</sup> \*

<sup>†</sup>Department of Chemical Engineering, Texas Tech University, Lubbock, TX 79409, USA

## Abstract

As the world moves towards sustainable energy sources, hydrogen (H ) can be treated as an eco-friendly alternative to fossil fuels due to its high energy density and zero carbon emission. The water-gas shift (WGS) reaction is a widely used industrial process for hydrogen production by converting carbon monoxide and steam into hydrogen and carbon dioxide. However, occurrences like severe fouling, catalyst deterioration, and thermal runaway can hamper the reaction kinetics/process safety and decrease the yield of H<sub>2</sub>. These incidents are rare, and gathering process data under such abnormal conditions is challenging. In this work, we propose a physics-guided conditional diffusion model to generate realistic rare-event trajectories for the WGS reaction. The proposed model integrates a conditional denoising diffusion probabilistic model (CDDPM) with governing laws of the reaction to generate physically consistent process trajectories. The conditioning features allow the model to produce high-quality synthetic profiles for rare-event domains that are typically beyond the training regimes. The generated rare-event trajectories then augment the raw dataset for a balanced distribution between normal and abnormal conditions. We further propose a hazard score to assess the risk severity of the operating condition based on the operating trajectory. Deep learning models are trained with the augmented dataset to diagnose the health status of the reaction. Simulation results show that the proposed physics-guided diffusion model outperforms data-driven models in terms of the quality of synthetic data and diagnosis performance for rare events.

Keywords: Water-gas shift reaction; Physics-guided conditional diffusion model; Rare-event data synthesis; Hazard score index; Process fault diagnosis.

## 1 Introduction

Excessive use of fossil fuels has increased global warming in recent years [1, 2]. During combustion, fossil fuels emit large amounts of greenhouse gases into the atmosphere. Greenhouse gases $\mathrm { ( C H _ { 4 } , C O _ { 2 } , N _ { 2 } O ) }$ absorb infrared radiation and re-emit it into the earth’s atmosphere, causing global warming [3, 4]. Unlike fossil fuels, hydrogen (H<sub>2</sub>) produces only water vapour $( \mathrm { H } _ { 2 } \mathrm { O } )$ during combustion and does not emit greenhouse gases. Therefore, hydrogen can be considered as a clean and sustainable energy source alternative to fossil fuels [5,6]. Among various hydrogen production technologies, the water-gas shift (WGS) reaction is widely used as a benchmark process. In this reaction, carbon monoxide (CO) reacts with water vapour $( \mathrm { H } _ { 2 } \mathrm { O } )$ to produce hydrogen (H<sub>2</sub>) and carbon dioxide $( \mathrm { C O } _ { 2 } )$ with the aid of catalysts. As a result, the hydrogen content in syngas (a mixture of CO and $\mathrm { H } _ { 2 } )$ increases while the carbon monoxide content decreases [5, 7, 8].

WGS reactions are normally operated within safe limits. However, under severe operating conditions, catalyst deactivation may occur due to thermal sintering, sulphur poisoning, and chloride poisoning [9, 10]. In addition, severe fouling over time can reduce heat transfer efficiency, increasing the risk of thermal runaway because the WGS reaction is moderately exothermic [11]. Since these abnormal conditions occur rarely, only highly limited operating data under these conditions, if any, are available. This creates significant challenges for reliable monitoring, diagnosis, and mitigation of abnormal operations for WGS reactions to prevent disasters from occurring.

To overcome this issue, various data augmentation methods have been developed by creating high-fidelity synthetic profiles under rare events. A common strategy is the numerical simulation of reaction models by leveraging explicit governing laws, such as mass, energy balance, and reaction kinetics to generate numerical data for rare scenarios. For instance, Chen et al. [12] simulated the WGS reaction via discretization of such governing laws with finite-volume method followed by semi-explicit algorithms. Bac et al. [11] modeled the WGS process with 2D Navier-Stokes equations alongside reactive heat and mass transport. They incorporated heat exchange functions to minimize the impact of thermodynamic limitations to increase the overall conversion of CO. However, these methods require full knowledge of the underlying governing laws that may not be available for many processes and operating conditions.

On the other hand, generative models, such as generative adversarial networks (GANs) and diffusion models, have emerged for data synthesis to overcome the scarcity issue [13, 14]. Specifically, GANs involve an adversarial training between a discriminator and a generator, and thus often suffer from training instability and mode collapse [15, 16]. Unlike GANs, diffusion models start with random noise and iteratively denoise it until it forms a realistic sample that resembles the data that they were trained on [14,17]. The absence of adversarial training allows the diffusion model to avoid issues like mode collapse [15]. Thus, diffusion models have been widely adopted in the synthesis of various data forms such as images, audio, and time series [15, 16, 18–20]. For chemical processes, where operating data are predominantly time series, diffusion models have been reported to generate synthetic data for reactions such as the ozone–nitric oxide reaction [21], the prediction of gas-dispersion field distributions [22], and material design and drug discovery [23–25].

Although diffusion models are effective for time-series data synthesis, they often overlook the system governing laws from which time-series data are collected. Thus, how to enable the physical plausibility of synthetic data is of priority for time-series data synthesis. To this end, various physics-informed diffusion models have been proposed. For instance, Yuan et al. [26] proposed a physics-guided motion diffusion model, PhysDiff, and incorporated physical constraints to iteratively pull the motion toward a physically plausible space. Wang et al. [27] proposed PhyDA, a physics-guided diffusion model for data assimilation in atmospheric systems. They incorporated a partial differential equation (PDE)-based physics loss during the training to generate physically consistent atmospheric data. For chemical processes such as WGS-based H<sub>2</sub> production, the compliance with physical laws is critical, since otherwise the synthetic data may show unrealistic behaviors such as negative flow rates, unreasonable temperatures, among others. Moreover, the inclusion of physics can assist the generative model to overcome data scarcity and extrapolate to regimes beyond the training data domain [28]. This is crucial for data synthesis of rare-event cases to enable effective monitoring and prevention of such scenarios. However, to our best knowledge, there has been no reports on developing physics-guided diffusion models for rare-event data synthesis and diagnosis for the WGS reaction, which motivates this work.

![](images/cd04f47bba9334fb4b5b75f01cce95dbb220d17632ff021ea60a66371b676795.jpg)  
Figure 1: The overall framework of the proposed Pg-CDDPM-based rare-event synthesis and diagnosis method.

In this study, we propose a physics-guided conditional denoising diffusion probabilistic model (Pg-CDDPM) for rare-event data synthesis and fault diagnosis of the WGS process. The proposed model incorporates physics (may not be exact) loss into the conventional noise prediction loss to ensure that the generated trajectories satisfy the governing equations. This physics-informed nature together with the conditioning feature allow for the extrapolation to generate rare-event synthetic data. The overall framework of our method is shown in Fig. 1, where limited trajectory data are used by Pg-CDDPM to generate additional synthetic trajectories for rare-event conditions, followed by training a monitoring model (gated recurrent unit, GRU) to diagnose the severity level of abnormal operation in the test data. The main contributions are summarized as follows:

• The WGS reaction and reactor dynamics are modeled using the Eley–Rideal reaction mechanism and a continuous stirred-tank reactor (CSTR) model. Critical parameters, e.g., heat transfer coefficient and activation energy factor, are selected and adjusted to create different operating conditions from normal to abnormal and ultimately disaster scenarios.

• A Pg-CDDPM with a unique backbone 1D U-Net architecture is proposed that incorporates governing equations and is conditioned on heat transfer efficiency and activation energy factors to generate physically consistent WGS operating profiles under normal and rareevent operating conditions, including both interpolation and extrapolation regimes.

• A composite health indicator is developed to assess the health status of the reaction. Different levels of severity are defined based on the health indicator, including normal, slightly abnormal, extreme, and disaster. The generated synthetic data for rare-event scenarios (extreme and disaster) are leveraged to train the health monitoring model for future diagnosis.

• Extensive validations based on the WGS simulator are conducted to evaluate the fidelity of synthetic data, as well as the effectiveness of augmenting raw data with synthetic data in improving the diagnosis performance for different severity classes.

## 2 Preliminaries

## 2.1 Water-Gas Shift Reaction

The WGS reaction is widely used in the industry for hydrogen production and ammonia synthesis. In this reversible reaction, carbon monoxide (CO) reacts with water vapor $\mathrm { ( H _ { 2 } O ) }$ to produce hydrogen (H<sub>2</sub>) and carbon dioxide $\left( \mathrm { C O _ { 2 } } \right)$ :

$$
\mathrm { C O } + \mathrm { H _ { 2 } O }  \mathrm { C O _ { 2 } } + \mathrm { H _ { 2 } } , \qquad \Delta H = - 4 1 . 1 \mathrm { k J / m o l } .\tag{1}
$$

The reaction is moderately exothermic, and its equilibrium conversion decreases with increasing temperature. The reaction is favored thermodynamically at lower temperatures and kinetically at elevated temperatures. Industrially, the reaction is conducted in two stages: a high temperature stage and a low temperature stage. In a high temperature shift reaction, the reactor operates between $3 0 0 ^ { \circ } \mathrm { C }$ and $5 5 0 ^ { \circ } \mathrm { C }$ and uses an iron-based catalyst $( { \mathrm { F e / C r } } )$ . In contrast, in a low temperature stage, the reactor operates between $1 5 0 ^ { \circ } \mathrm { C }$ and $2 3 0 ^ { \circ } \mathrm { C }$ and uses a copper–zinc catalyst supported over alumina $\left( \mathrm { C u O } / \mathrm { Z n O } / \mathrm { A l _ { 2 } O _ { 3 } } \right)$ [5, 9, 29]. As the equilibrium CO conversion decreases at very high temperatures, an intermediate cooling system is included to maintain the temperature within the desired operating range to increase the overall CO conversion.

## 2.2 Denoising Diffusion Probabilistic Models

Diffusion models are a class of probabilistic generative models. They learn the underlying data distribution through a two-stage process. First, they destroy the original data by progressively adding noises. In the reverse process, they learn through a denoising process to generate new samples from pure noise [14, 15, 17]. Among various diffusion models, the DDPM proposed by Ho et al. [14] is a common type due to its ability to produce high-quality samples. As in Fig. 2, the DDPM consists of two Markov chains: a forward diffusion process and a reverse diffusion process. In the forward process, Gaussian noise is gradually added to the original data until it approaches pure noise. In the reverse process, a neural network learns to estimate the added noise and progressively removes it, allowing new data samples to be generated from pure noise. Denote the original data sample with true distribution as $\mathrm { x } _ { 0 } \sim q ( \mathrm { x } _ { 0 } )$ . The diffusion process consists of $T$ time steps, with each diffusion step $t \in \{ 1 , 2 , \dots , T \}$ . The forward process gradually adds Gaussian noise to the original data according to a predefined variance schedule $\{ \beta _ { t } \} _ { t = 1 } ^ { T } ,$ expressed as [14]

$$
q ( \mathrm { x } _ { 1 : T } \mid \mathrm { x } _ { 0 } ) = \prod _ { t = 1 } ^ { T } q ( \mathrm { x } _ { t } \mid \mathrm { x } _ { t - 1 } ) ,\tag{2}
$$

$$
q ( \mathbf { x } _ { t } \mid \mathbf { x } _ { t - 1 } ) = { \mathcal { N } } \left( \mathbf { x } _ { t } ; { \sqrt { 1 - \beta _ { t } } } \mathbf { x } _ { t - 1 } , \beta _ { t } \mathbf { I } \right) ,\tag{3}
$$

where $\mathrm { x } _ { t }$ is the noisy sample at diffusion step $t , \beta _ { t }$ is a predefined variance schedule typically increasing over $t ,$ and I is the identity matrix. Define

$$
\alpha _ { t } = 1 - \beta _ { t } , \qquad \bar { \alpha } _ { t } = \prod _ { i = 1 } ^ { t } \alpha _ { i } .\tag{4}
$$

Then the distribution of $\mathrm { x } _ { t }$ conditioned directly on the original sample $\mathrm { x } _ { \mathrm { 0 } }$ can be written as

$$
q ( \mathrm { x } _ { t } \mid \mathrm { x } _ { 0 } ) = \mathcal { N } \left( \mathrm { x } _ { t } ; \sqrt { \bar { \alpha } _ { t } } \mathrm { x } _ { 0 } , ( 1 - \bar { \alpha } _ { t } ) \mathrm { I } \right) .\tag{5}
$$

Consequently, the noisy sample at any diffusion time step t becomes

$$
\begin{array} { r } { \mathrm { x } _ { t } = \sqrt { \bar { \alpha } _ { t } } \mathrm { x } _ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , \mathrm { I } ) , } \end{array}\tag{6}
$$

where ϵ is the standard Gaussian noise. This formulation enables the noisy sample at any arbitrary t to be generated directly from the original data without repeatedly applying the forward diffusion process. The reverse process starts from a pure Gaussian noise sample $\mathrm { x } _ { T } \sim \mathcal { N } ( 0 , \mathrm { I } )$ . The objective is to gradually remove the added noise and construct new samples that follow the original data distribution. It is defined as a learnable Markov chain via a parameterized neural network

$$
p _ { \boldsymbol { \theta } } \bigl ( \mathbf { x } _ { 0 : T } \bigr ) = p ( \mathbf { x } _ { T } ) \prod _ { t = 1 } ^ { T } p _ { \boldsymbol { \theta } } \bigl ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } \bigr ) ,\tag{7}
$$

where $p ( \mathbf { x } _ { T } ) = \mathcal { N } ( \mathbf { x } _ { T } ; 0 , \mathrm { I } )$ . Each reverse transition is modeled as

$$
\begin{array} { r } { p _ { \theta } \left( \mathrm { x } _ { t - 1 } \mid \mathrm { x } _ { t } \right) = \mathcal { N } \left( \mathrm { x } _ { t - 1 } ; \mu _ { \theta } ( \mathrm { x } _ { t } , t ) , \Sigma _ { \theta } ( \mathrm { x } _ { t } , t ) \right) . } \end{array}\tag{8}
$$

The network learns to predict the noise $\epsilon _ { \theta } ( x _ { t } , t )$ added during the forward process, and then

$$
\mathrm { x } _ { t - 1 } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( \mathrm { x } _ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta } ( \mathrm { x } _ { t } , t ) \right) + \sqrt { \beta _ { t } } z , \qquad z \sim \mathcal { N } ( 0 , I ) ,\tag{9}
$$

where $z$ is standard Gaussian noise. Starting from a pure noise sample $\mathrm { x } _ { T } ,$ , Eq. (9) is repeatedly applied from $t = T$ to $t = 1$ to progressively remove the added noise and generate a sample that follows the original data distribution [14].

![](images/e0b0c9527d355357e685ccea6bb1b876b7672b5aebbdffc9bd87324f333927e9.jpg)  
Figure 2: Overview of the forward and reverse processes of a diffusion model.

## 3 Methodology

## 3.1 WGS Simulator

A dynamic simulator of the WGS reaction Eq. (1) is developed to generate operating trajectories of process variables for training and evaluating the proposed Pg-CDDPM. The simulator captures transient behaviors of the reaction by solving governing equations under varying heat transfer efficiency and activation energy factors. A schematic diagram of the CSTR is shown in Fig. 3. We assume that the reactor has perfect mixing, constant volume, and negligible pressure drop. Specifically, the reaction rate is described by the Eley–Rideal kinetic model with temperaturedependent reaction rate constants obtained from the Arrhenius equation. Reaction dynamics are governed by species mole balances for $\mathrm { C O , H _ { 2 } O , C O _ { 2 } }$ , and $\mathrm { H _ { 2 } } .$ , alongside the energy balance that accounts for the heat released by the reaction and heat exchange with the cooling system. In particular, the mole balance equations are:

$$
\frac { d C _ { \mathrm { C O } } } { d t } = \frac { F } { V } \left( C _ { \mathrm { C O } , f } - C _ { \mathrm { C O } } \right) - r ,\tag{10}
$$

$$
\frac { d C _ { \mathrm { H _ { 2 } O } } } { d t } = \frac { F } { V } \left( C _ { \mathrm { H _ { 2 } O , } f } - C _ { \mathrm { H _ { 2 } O } } \right) - r ,\tag{11}
$$

$$
\frac { d C _ { \mathrm { C O _ { 2 } } } } { d t } = \frac { F } { V } \left( C _ { \mathrm { C O _ { 2 } } , f } - C _ { \mathrm { C O _ { 2 } } } \right) + r ,\tag{12}
$$

$$
\frac { d C _ { \mathrm { H _ { 2 } } } } { d t } = \frac { F } { V } \left( C _ { \mathrm { H _ { 2 } } , f } - C _ { \mathrm { H _ { 2 } } } \right) + r ,\tag{13}
$$

where $C _ { i }$ and $C _ { f , i }$ denote the reactor and feed concentrations of species $i ,$ respectively, F is the volumetric flow rate, $V$ is the reactor volume, and $r$ is the reaction rate (to be defined in Section 3.2). With a cooling water system, the energy balance equation can be expressed as:

$$
\frac { d T _ { r } } { d t } = \frac { F } { V } \left( T _ { f } - T _ { r } \right) - \frac { \Delta H _ { r } } { \rho C _ { p } } r + \frac { U A } { V \rho C _ { p } } \left( T _ { c } - T _ { r } \right) ,\tag{14}
$$

where $T _ { r }$ and $T _ { f }$ are the reactor and feed temperatures, respectively, $T _ { c }$ is the coolant temperature, $\Delta H _ { r }$ is the heat of reaction, $\rho$ is the fluid density, $C _ { p }$ is the heat capacity, and UA is the overall heat transfer coefficient. For this reaction, the input variables include the coolant temperature $T _ { c }$ and feed flow rate $F ,$ whereas the output variables include reactor temperature $T _ { r }$ and concentration of species $C _ { i }$

![](images/8260ec43fb8463ae00a6fa629bf96ee2ed417fd0d0b2a35c7cf8dfdb0cff7167.jpg)  
Figure 3: Schematic diagram of the WGS reaction in a continuous stirred-tank reactor.

## 3.2 Reaction Mechanism

An iron-based catalyst is selected to develop the reaction rate expression. The catalyst operates in a temperature range of $3 0 0 { - } 5 3 0 ^ { \circ } \mathrm { C } .$ . Based on the literature, the WGS reaction on iron-based catalysts follows the Rideal–Eley mechanism, in which steam reacts with an active catalytic site to form hydrogen and an adsorbed oxygen species. The adsorbed oxygen then reacts with carbon monoxide to produce carbon dioxide while regenerating the active sites. The elementary reaction steps are given by [30]

$$
\mathrm { H } _ { 2 } \mathrm { O } + \ast  \mathrm { H } _ { 2 } + \mathrm { O } ^ { \ast } , \quad \mathrm { C O } + \mathrm { O } ^ { \ast }  \mathrm { C O } _ { 2 } + \ast ,
$$

where ∗ denotes an active catalytic site and $O ^ { * }$ represents an oxygen species adsorbed on the catalyst surface. The Eley–Rideal rate expression is derived based on the following assumptions: (i) the adsorption step is at quasi-equilibrium (steady state); (ii) the surface reaction is the ratedetermining step; and (iii) the catalyst active sites are regenerated after product desorption. Accordingly, the reaction-rate expression adopted for the reactor model is

$$
r = \frac { k _ { 2 } K _ { 1 } C _ { \mathrm { C O } } C _ { \mathrm { H _ { 2 } O } } - k _ { 2 , \mathrm { r e v } } C _ { \mathrm { C O _ { 2 } } } C _ { \mathrm { H _ { 2 } } } } { C _ { \mathrm { H _ { 2 } } } + K _ { 1 } C _ { \mathrm { H _ { 2 } O } } } , k _ { 2 } = k _ { 0 } \exp \left( - \frac { E _ { a } } { R T _ { r } } \right) .\tag{15}
$$

For the forward reaction rate constant $k _ { 2 } ,$ , k<sub>0</sub> is the pre-exponential factor, $E _ { a }$ is the activation energy, R is the gas constant, and $T _ { r }$ is the reactor temperature. The reverse reaction rate constant is assumed to be proportional to the forward rate constant, $k _ { 2 , \mathrm { r e v } } = 0 . 5 k _ { 2 }$ , with the adsorption equilibrium constant taken as $K _ { 1 } = 0 . 5$

In our study, we use effective heat transfer coefficient $U A _ { e f f }$ and effective activation energy $E _ { a , e f f }$ to replace those in (14) and (15), respectively, so as to better represent the actual heat transfer efficiency and catalyst activity observed during practical operations:

$$
U A _ { e f f } = f \cdot U A , \quad E _ { a , e f f } = f _ { E _ { a } } \cdot E _ { a } ,\tag{16}
$$

where $f$ and $f _ { E _ { a } }$ stand for heat transfer efficiency factor and activation energy factor. These two factors will be are carefully varied to create different operating conditions and generate diverse process variable trajectories.

![](images/55a8ab28643c6b452edd493a24b004de5b6ffd8b3a767969318621476c3829de.jpg)  
Figure 4: Overview of the proposed $\mathrm { P g } .$ -CDDPM framework.

## 3.3 Proposed Physics-Guided Conditional DDPM

In our study, we will develop a novel Pg-CDDPM to generate synthetic data for the above WGS reaction, illustrated in Fig. 4. First, the original multivariate trajectory $\mathrm { x } _ { 1 : L } ^ { 0 }$ is transformed into pure noise $\mathrm { x } _ { 1 : L } ^ { T }$ through the forward diffusion process, with its reconstruction through the reverse denoising process. During the reverse process, the governing equations are incorporated as physics guidance to improve the physical consistency of generated trajectories. As in Section 2.2, the forward process is defined as

$$
q \left( \mathbf { x } _ { 1 : L } ^ { 1 : T } \mid \mathbf { x } _ { 1 : L } ^ { 0 } \right) = \prod _ { t = 1 } ^ { T } q \left( \mathbf { x } _ { 1 : L } ^ { t } \mid \mathbf { x } _ { 1 : L } ^ { t - 1 } \right) , q \left( \mathbf { x } _ { 1 : L } ^ { t } \mid \mathbf { x } _ { 1 : L } ^ { t - 1 } \right) = \mathcal { N } \left( \mathbf { x } _ { 1 : L } ^ { t } ; \sqrt { 1 - \beta _ { t } } \mathbf { x } _ { 1 : L } ^ { t - 1 } , \beta _ { t } \mathbf { I } \right) ,\tag{17}
$$

where T is the total number of diffusion time steps, L is the total trajectory length, $\mathrm { x } _ { 1 : L } ^ { 0 }$ is the clean (ground-truth) raw trajectory, and $\mathrm { x } _ { 1 : L } ^ { t }$ is the corresponding noisy trajectory at diffusion step t. With Eq. (4), the noisy trajectory at diffusion step t can be sampled directly from the original clean trajectory without iteratively applying the forward process:

$$
q \big ( \mathrm { x } _ { 1 : L } ^ { t } \ | \ \mathrm { x } _ { 1 : L } ^ { 0 } \big ) = \mathcal { N } \left( \mathrm { x } _ { 1 : L } ^ { t } ; \sqrt { \bar { \alpha } _ { t } } \mathrm { x } _ { 1 : L } ^ { 0 } , \left( 1 - \bar { \alpha } _ { t } \right) \mathrm { I } \right) .\tag{18}
$$

The noisy trajectory at any diffusion time step t can be computed as

$$
\begin{array} { r } { \mathrm { x } _ { 1 : L } ^ { t } = \sqrt { \bar { \alpha } _ { t } } \mathrm { x } _ { 1 : L } ^ { 0 } + \sqrt { 1 - \bar { \alpha } _ { t } } \epsilon _ { 1 : L } , \qquad \epsilon _ { 1 : L } \sim \mathcal { N } ( 0 , I ) . } \end{array}\tag{19}
$$

For conditional diffusion models, the reverse process progressively removes the added noise for conditioning feature c:

$$
p _ { \theta } \big ( \mathbf { x } _ { 1 : L } ^ { 0 : T } \mid \mathbf { c } \big ) = p \big ( \mathbf { x } _ { 1 : L } ^ { T } \big ) \prod _ { t = 1 } ^ { T } p _ { \theta } \big ( \mathbf { x } _ { 1 : L } ^ { t - 1 } \mid \mathbf { x } _ { 1 : L } ^ { t } , \mathbf { c } \big ) .\tag{20}
$$

where

$$
\begin{array} { r } { p _ { \theta } \left( \mathbf { x } _ { 1 : L } ^ { t - 1 } \mid \mathbf { x } _ { 1 : L } ^ { t } , \mathbf { c } \right) = \mathcal { N } \left( \mathbf { x } _ { 1 : L } ^ { t - 1 } ; \mu _ { \theta , 1 : L } \left( \mathbf { x } _ { 1 : L } ^ { t } , t , \mathbf { c } \right) , \Sigma _ { \theta , 1 : L } \left( \mathbf { x } _ { 1 : L } ^ { t } , t , \mathbf { c } \right) \right) . } \end{array}\tag{21}
$$

$$
\mathbf { x } _ { 1 : L } ^ { t - 1 } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( \mathbf { x } _ { 1 : L } ^ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \alpha _ { t } } } \epsilon _ { \theta , 1 : L } \big ( \mathbf { x } _ { 1 : L } ^ { t } , t , \mathbf { c } \big ) \right) + \sqrt { \beta _ { t } } z _ { 1 : L } , \qquad z _ { 1 : L } \sim \mathcal { N } ( 0 , I ) .\tag{22}
$$

The model is trained by minimizing the mean-squared error (MSE) loss between the added noise sequence $\epsilon _ { 1 : L } ^ { t }$ at the forward process step t and the predicted noise $\epsilon _ { \theta , 1 : L } ( \cdot )$

$$
\mathcal { L } _ { \mathrm { n o i s e } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left. \epsilon _ { 1 : L } ^ { t } - \epsilon _ { \theta , 1 : L } \left( \mathrm { x } _ { 1 : L } ^ { t } , t , c \right) \right. _ { 2 } ^ { 2 } .\tag{23}
$$

Once the model learns how to remove the noise, it can generate new trajectories from pure noise. In our study, each trajectory consists of seven time-varying variables: the concentrations of $\mathrm { C O } ,$ $\mathrm { C O _ { 2 } , H _ { 2 } O , }$ , and $\mathrm { H _ { 2 } } ,$ the reactor temperature $T _ { r } ,$ the coolant temperature $T _ { c } ,$ , and the feed flow rate F. The clean and noisy trajectories (at diffusion step t) are represented as $\mathbf { x } _ { 1 : L } ^ { 0 } , \mathbf { x } _ { 1 : L } ^ { t } \in \mathbb { R } ^ { m \times L } , m = 7$ for our study. The conditioning variables are the heat transfer efficiency factor $f$ and the activation energy factor $f _ { E _ { a } }$ , represented as $c \in \mathbb { R } ^ { n _ { c } } , n _ { c } = 2 ;$ see Fig. 4.

To incorporate the physics loss into the reverse process, the generated inputs by the diffusion model at diffusion step $t , T _ { c } ^ { t }$ and $F ^ { t }$ , are provided to the WGS reaction simulator, which solves the governing laws Eq. (10)-(16). The simulator outputs physically-driven reactor temperature and species concentration trajectories that the diffusion model aims to produce if the governing laws are fully respected. At difusion step $t ,$ the difference between synthetic outputs $T _ { r , \theta } ^ { t } , C _ { i , \theta } ^ { t }$ and simulator outputs $T _ { r , s i m } ^ { t } , C _ { i , s i m } ^ { t }$ represent the extent at which the diffusion model respects governing laws. This error will serve as a regularization to train the diffusion model

$$
\mathcal { L } _ { \mathrm { p h y s i c s } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \left[ \Vert T _ { r , s i m } ^ { t } - T _ { r , \theta } ^ { t } \Vert _ { 2 } ^ { 2 } + \sum _ { i = 1 } ^ { 4 } \Vert C _ { i , s i m } ^ { t } - C _ { i , \theta } ^ { t } \Vert _ { 2 } ^ { 2 } \right] .\tag{24}
$$

The total loss is defined as a weighted combination of the noise prediction loss (for reproducing the raw data) and the physics loss (for complying with physics):

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { n o i s e } } + \lambda \mathcal { L } _ { \mathrm { p h y s i c s } } ,\tag{25}
$$

where λ is a weight. Note that when simulating $T _ { r , s i m } ^ { t } , C _ { s i m , i } ^ { t } ,$ the inputs to the simulator, although generated by the diffusion network with parameter $\theta ,$ are disconnected from the computational graph (irrelevant to θ).

![](images/0e99adf3de22b612298a3b90b607276837e30ada523b3535acab4aa4fcb78e4d.jpg)  
Figure 5: Architecture of the proposed conditional 1D U-Net used to predict the added noise during the reverse diffusion process.

## 3.3.1 U-Net architecture

The proposed Pg-CDDPM utilizes an 1D U-Net to predict the noise added during the forward process. The 1D U-Net employs 1D convolution operations, which are suitable for time-series applications [31, 32]. The 1D U-Net consists of three main components: an encoder, a bottleneck, and a decoder [33]. The encoder downsamples the noisy input trajectory to extract local features, the bottleneck captures global features from the compressed noisy trajectory, and the decoder upsamples the feature maps back to the original trajectory length to predict the noise.

As shown in Fig. 5, the network takes three inputs: the noisy trajectory $\mathbf { x } _ { 1 : L } ^ { t } \in \mathbb { R } ^ { m \times L }$ , the conditioning features $c \in \mathbb { R } ^ { n _ { c } }$ , and the diffusion time step t. The conditioning features are concatenated with the noisy trajectory along the channel dimension. In the early diffusion steps, only a small amount of noise is added to the input trajectory, whereas the later diffusion steps contain significantly higher noise levels. Thus, the diffusion time step is first converted into a time embedding using a linear layer and then added to the encoder features. This provides the 1D U-Net with information about the current diffusion step, allowing it to understand how much noise has been added to the input trajectory and accurately predict the noise. The encoder consists of two ResBlock 1D layers, each followed by a MaxPool1D layer to extract important features from the input trajectory. The ResBlock 1D structure has been shown in Fig. 6. A bottleneck ResBlock further processes the encoded features before they are passed to the decoder. The decoder consists of two UpConv1D layers and two ResBlock 1D layers that gradually reconstruct the features to the original trajectory length L. Skip connections are used between the encoder and decoder to retain important information from the input trajectory. Finally, a 1 × 1 Conv1D layer maps the decoder features to m output channels, producing the predicted noise $\epsilon _ { \theta , 1 : L } ( \mathrm { x } _ { 1 : L } ^ { t } , t , c ) \in \mathbb { R } ^ { m \times L }$

Our task-specific contributions to the 1D U-Net architecture are summarized as follows: (1) we incorporate ResNet blocks into the 1D U-Net instead of simple 1D convolutional operations. The ResNet blocks learn the residual changes, making them effective for predicting the residual noise representations in the trajectories, while improving gradient propagation during training; and (2) within each ResNet block, the SiLU activation function is employed because it is effective in capturing both positive and negative noise components. The overall architecture of the proposed 1D U-Net is shown in Table 1.

![](images/b276c90ac70277035fab0cff1390b59063ff489f57b786d00edb40679cf8ae5a.jpg)  
Figure 6: Architecture of the ResBlock 1D used as building blocks of the diffusion model.

Table 1: Summary of the proposed 1D U-Net architecture.
<table><tr><td>Layer</td><td>Operation</td><td>Kernel Stride</td><td></td><td>Channels (In → Out)</td><td>Trajectory (In → Out)</td></tr><tr><td>Input</td><td>Concatenate inputs</td><td>一</td><td>一</td><td>7 + 2 → 9</td><td>L → L</td></tr><tr><td>Resnet Block 1</td><td>ResBlock 1D</td><td>3</td><td>1</td><td>9 → base_ch</td><td>L → L</td></tr><tr><td>Maxpooling 1</td><td>MaxPool1D</td><td>2</td><td>2</td><td>base_ch → base_ch</td><td>L → L一2</td></tr><tr><td>Resnet Block 2</td><td>ResBlock 1D</td><td>3</td><td>1</td><td>base_ch → 2 × base_ch</td><td># → L一2</td></tr><tr><td>Maxpooling 2</td><td>MaxPool1D</td><td>2</td><td>2</td><td>2 × base_ch → 2 × base_ch</td><td>L2 → L4</td></tr><tr><td>Bottleneck</td><td>ResBlock 1D</td><td>3</td><td>1</td><td>2 × base_ch → 4 × base_ch</td><td>L4 → L4</td></tr><tr><td>Upsampling 1</td><td>UpConv1D</td><td>2</td><td>2</td><td>4 × base_ch → 2 × base_ch</td><td>L4 → L2</td></tr><tr><td>Resnet Block 3</td><td>ResBlock 1D</td><td>3</td><td>1</td><td>4 × base_ch → 2 × base_ch</td><td>L2 → L一2</td></tr><tr><td>Upsampling 2</td><td>UpConv1D</td><td>2</td><td>2</td><td>2 × base_ch → base_ch</td><td>L2 → L</td></tr><tr><td>Resnet Block 4</td><td>ResBlock 1D</td><td>3</td><td>1</td><td>2 × base_ch → base_ch</td><td>L → L</td></tr><tr><td>Output</td><td>Conv1D</td><td>1</td><td>1</td><td>base_ch → 7</td><td>L → L</td></tr></table>

## 3.3.2 Training and sampling algorithms

As noted before, the inputs generated by the diffusion model are detached from the computational graph when being passed to the WGS reaction simulator. The physics loss is computed by comparing simulated and predicted reactor output variables, and backpropagation is performed only through the reactor output variables predicted by the diffusion model. Also, the generated coolant temperature $T _ { c }$ and the feed flow rate F are clipped to proper ranges: 350–500 K and 70– 150 L min<sup>−1</sup>. This ensures that the diffusion-generated inputs remain within feasible operating limits before being passed to the simulator. Specifically, the selected coolant temperature limits can maintain the reactor temperature at approximately $3 0 0 { - } 4 5 0 ^ { \circ } \mathrm { C }$ (573–723 K), which is suitable for high-temperature shift operation and prevents unrealistic operating conditions. Similarly, constraining the feed flow rate ensures a realistic reactor space time $\tau = V / F \in [ 0 . 6 7$ min, 1.43 min] where V = 100 L. These constraints can assist the diffusion model to produce plausible reactor dynamics and generating stable trajectories.

Algorithm 1 Pg-CDDPM Training   
1: Standardize each trajectory using training-data mean, standard deviation: $\textstyle \mathrm { x } _ { 0 } = { \frac { \mathrm { x } _ { 0 } - \mu } { \sigma } }$   
2: repeat   
3: Sample a mini-batch $( \mathbf { x } _ { 0 } , \mathbf { c } ) .$ , where $\mathbf { x } _ { 0 } \in \mathbb { R } ^ { N \times m \times L }$ is the standardized trajectory tensor, $\mathbf { c \in }$   
$\mathbb { R } ^ { N \times n _ { c } }$ is the conditioning tensor, and N is the number of trajectories in the mini-batch.   
4: Sample one diffusion step out of $[ t _ { 1 } , t _ { 2 } , \ldots , t _ { N } ] , t _ { i } \sim U \{ 1 , \ldots , T \}$ for each trajectory in the   
mini-batch, and store sampled time steps into $\mathbf { t } \in \mathbb { R } ^ { N }$   
5: Sample a Gaussian noise tensor $\epsilon \sim \mathcal { N } ( 0 , \mathrm { I } ) , \epsilon \in \mathbb { R } ^ { N \times m \times L } .$   
6: Generate the noisy trajectories $\mathbf { x ^ { t } } = \sqrt { \bar { \alpha } _ { \mathbf { t } } } \otimes \mathbf { x } ^ { 0 } + \sqrt { 1 - \bar { \alpha } _ { \mathbf { t } } } \otimes \epsilon ,$ where ${ \bar { \alpha } } _ { \mathbf { t } } : = [ { \bar { \alpha } } _ { t _ { 1 } } , \dots , { \bar { \alpha } } _ { t _ { N } } ] ^ { \top } \in \mathbb { R } ^ { N }$   
stacks the $\bar { a } _ { t }$ as in (4) for each sample in the batch, and $\otimes$ denotes sample-wise product.   
7: Predict the noise tensor $\epsilon _ { \theta } \left( \mathbf { x } ^ { \mathbf { t } } , \mathbf { t } , \mathbf { c } \right) \in \mathbb { R } ^ { N \times m \times L }$ using the 1D U-Net.   
8: Compute the diffusion noise loss $\mathcal { L } _ { \mathrm { n o i s e } }$ by applying (23) across all samples in the mini-batch.   
9: Estimate the clean trajectory using the predicted noise $\begin{array} { r } { \hat { \mathbf { x } } ^ { 0 } = \frac { \mathbf { x } ^ { \mathbf { t } } - \sqrt { 1 - \bar { \alpha } _ { \mathbf { t } } } \mathrm { \tilde { \otimes } } \epsilon _ { \theta } \left( \mathbf { x } ^ { \mathbf { t } } , \mathbf { t } , \mathbf { c } \right) } { \sqrt { \bar { \alpha } _ { \mathbf { t } } } } } \end{array}$   
α¯<sub>t</sub>   
10: Split the predicted clean trajectory into outputs $\hat { \mathbf { s } } _ { \theta } ^ { s d } = [ \hat { T } _ { r , \theta } , \hat { C } _ { i , \theta } ]$ and control inputs $\hat { \mathbf { u } } _ { \theta } ^ { s d } = \begin{array} { r l } \end{array}$   
$[ \hat { T } _ { c , \theta } , \hat { F } _ { \theta } ] .$ , where the superscript $^ { \prime \prime } \mathrm { s d } ^ { \prime \prime }$ means standardized.   
11: Inverse-standardize the predicted control inputs to recover their physical values $\hat { \mathbf { u } } _ { \theta } .$   
12: Detach the generated control inputs (gradient free): uˆ $: = [ \hat { T } _ { c } , \hat { F } ] \gets \mathrm { d e t a c h } ( \hat { \mathbf { u } } _ { \theta } )$ from the   
computation graph.   
13: Clip the control inputs $\hat { T } _ { c }$ and $\hat { F }$ to the ranges of 350–500 K and $7 0 { - } 1 5 0 \operatorname { L } \operatorname* { m i n } ^ { - 1 } .$   
14: Randomly initialize the reactor using initial and feed conditions as in Table 5.   
15: Simulate the CSTR using $\hat { T } _ { c } , \hat { F }$ and above initial conditions to obtain simulated outputs,   
followed by standardization to obtain $[ \hat { T } _ { r , s i m } ^ { s d } , \hat { C } _ { i , s i m } ^ { s d } ] .$   
16: Compute the physics loss $\mathcal { L } _ { \mathrm { p h y s i c s } }$ as in (24) and total loss $\mathcal { L } _ { \mathrm { t o t a l } }$ as in (25).   
17: Update the network parameters θ via backpropogation of $\mathcal { L } _ { \mathrm { t o t a l } }$ using the Adam optimizer.   
18: until θ converges to $\theta ^ { * }$

Algorithm 1 summarizes the training procedure of the proposed Pg-CDDPM. It describes the forward process, noise prediction, reconstruction of clean trajectories, computation of physics loss using the CSTR simulator, and optimization of model parameters. Algorithm 2 presents the conditional sampling procedure, where the trained model gradually denoises an initial noise sample through the reverse diffusion process to generate high-quality synthetic reaction data under specified operating conditions.

Algorithm 2 Conditional Sampling Procedure   
1: Sample the initial Gaussian noise $\mathrm { x } _ { 1 : L } ^ { T } \sim \mathcal { N } ( 0 , \mathrm { I } )$   
2: for $t = T , T - 1 , \dots , 1$ do   
3: Predict the Gaussian noise $\epsilon _ { \theta ^ { * } , 1 : L } ( \mathrm { x } _ { 1 : L } ^ { t } , t , c )$   
4: Compute the reverse-diffusion mean using the predicted noise   
$\begin{array} { r } { \mu _ { \theta ^ { * } , 1 : L } = \frac { 1 } { \sqrt { \alpha _ { t } } } \left( \mathrm { x } _ { 1 : L } ^ { t } - \frac { \beta _ { t } } { \sqrt { 1 - \bar { \alpha } _ { t } } } \epsilon _ { \theta ^ { * } , 1 : L } \left( \mathrm { x } _ { 1 : L } ^ { t } , t , c \right) \right) } \end{array}$   
5: if $t > 1$ then   
6: Sample $z _ { 1 : L } \sim \mathcal { N } ( 0 , \mathrm { I } )$   
7: Sample the previous trajectory $\mathbf { x } _ { 1 : L } ^ { t - 1 } = \mu _ { \theta ^ { * } , 1 : L } + \sqrt { \beta _ { t } } z _ { 1 : L }$   
8: end if   
9: Set $\mathrm { x } _ { 1 : L } ^ { 0 } = \mu _ { \theta ^ { * } , 1 : L }$ when $t = 1$   
10: end for   
11: return $\mathrm { x } _ { 1 : L } ^ { 0 }$   
12: Inverse-standardize $\mathrm { x } _ { 1 : L } ^ { 0 }$ using training data mean and standard deviation to nominal ranges.

![](images/39d3a491d471165d7df81d26af296de39780518808d193abfa818df912706730.jpg)  
Figure 7: The flow chart for training the proposed Pg-CDDPM to generate high-fidelity synthetic profiles for the WGS reaction.

The detailed training workflow of the proposed Pg-CDDPM is shown in Fig. 7. First, the diffusion model learns the distribution of training trajectories by predicting the added noise using the U-Net. The predicted noise is then adopted to reconstruct the clean trajectories. Next, the reconstructed trajectories are separated into generated input and output variables. The generated inputs are detached from the gradient computation and passed to the CSTR simulator to obtain corresponding physics-based trajectories. The simulator outputs are then compared with the diffusion-generated state trajectories to calculate the physics loss. Finally, the noise and physics losses are combined to obtain the total training loss to train the diffusion model.

## 3.4 Hazard Score Index

The established physics-guided diffusion model above can assist to generate synthetic operating profiles that are physically consistent. This is particularly useful for the monitoring and diagnosis of rare-event scenarios where data scarcity is a predominant issue. For effective fault diagnosis, these operating profiles must be categorized according to the risk severity of the reactor condition. This severity classification converts complex multivariate trajectories into clear hazard levels that can be used to train and evaluate diagnostic models. To quantitatively reflect the operation status of the WGS reaction, we present a comprehensive hazard score as a health indicator for this process. Specifically, this hazard score combines the effects of reactor temperature, hydrogen concentration, heat transfer degradation, and catalyst degradation:

$$
\begin{array} { l } { { \mathrm { H a z a r d ~ S c o r e } = 0 . 4 \displaystyle \frac { 1 } { 1 + \exp \left[ 0 . 5 \left( \bar { C } _ { \mathrm { H _ { 2 } } } - C _ { \mathrm { H _ { 2 } , \mathrm { c r i t } } } \right) \right] } \qquad } } \\ { { \qquad + \ 0 . 2 \left[ 1 - \exp \left( - \displaystyle \frac { \left( \bar { T } _ { r } - T _ { 0 } \right) ^ { 2 } } { 2 \sigma _ { T } ^ { 2 } } \right) \right] + 0 . 2 ( 1 - f ) + 0 . 2 ( f _ { E _ { a } } - 1 ) , } } \end{array}\tag{26}
$$

where $\bar { C } _ { \mathrm { H _ { 2 } } }$ and $\mathrm { C _ { H _ { 2 } , c r i t } }$ denote the average and critical hydrogen concentrations, while $\hat { T } _ { r } , T _ { 0 . }$ and $\sigma _ { T }$ denote the average reactor temperature, optimal operating temperature, and temperature spread around $T _ { 0 } ,$ respectively. In this study, we consider $C _ { \mathrm { H _ { 2 } , c r i t } } = 1 \mathrm { m o l / L } , T _ { 0 } = 6 7 5 \mathrm { K ( 4 0 2 ^ { \circ } C ) }$ and $\sigma _ { T } = 1 5 0 \mathrm { K } ( \mathrm { i . e . , } a$ temperature spread of $\mathrm { 1 5 0 ^ { \circ } C ) }$ . First, the reactor temperature and hydrogen concentration are averaged over the trajectory length L. The hazard score is formulated such that both extremely low and high temperatures increase the hazard level. Low temperatures result in slow reaction kinetics, whereas high temperatures may lead to thermal runaway. The parameter $\sigma _ { T }$ controls the width of the acceptable temperature range, beyond which the hazard score increases rapidly. Similarly, the hazard score rises sharply when $\bar { C } _ { \mathrm { H _ { 2 } } }$ falls below $C _ { \mathrm { H _ { 2 } , c r i t } }$ . The dependence of the hazard score on $\bar { T }$ and $\bar { C } _ { \mathrm { H _ { 2 } } }$ is shown in Figs. 8(c) and 8(d), respectively. The factors f and $f _ { E _ { a } }$ account for fouling and catalyst degradation. Thus, higher scores represent more severe degradation and hazardous operating conditions. Each trajectory will be assigned to one of the four hazard classes based on its corresponding hazard score index shown in Fig. 8(b). Fig. 8(a) shows the increase in hazard score from normal to disaster conditions.

## 3.5 Hyperparameter Selection

The proposed Pg-CDDPM is implemented using the PyTorch framework and trained using the Adam optimizer. Table 2 summarizes the network configuration, training hyperparameters, and computational hardware used in this study. For the trajectory severity classification, we used the GRU classifier. Table 3 summarizes the hyperparameters used in the GRU classifier.

![](images/c590a1a20b9404e57bd4672eb7b211f321ed1970c018f34cc76c538c4bb8397e.jpg)

<table><tr><td>Hazard Score</td><td>Class</td><td>Operating Condition</td></tr><tr><td> $< 0 . 3 0$ </td><td>0</td><td>Normal</td></tr><tr><td> $0 . 3 0 \leq \mathrm { H S } < 0 . 5 0$ </td><td>1</td><td>Slightly Abnormal</td></tr><tr><td> $0 . 5 0 \leq \mathrm { H S } < 0 . 6 8$ </td><td>2</td><td>Extreme</td></tr><tr><td> $\ge 0 . 6 8$ </td><td>3</td><td>Disaster</td></tr></table>

(a) Hazard score index  
![](images/34549e40238e68af95cf27dda526e2fa9287df4a36ea7506fd76892520ad40a4.jpg)  
(c) Temperature-related hazard score

(b) Hazard classification  
![](images/d0fe2b341845d45f6b20522f680bfde071cf5730514bbcdded3de259601207ca.jpg)  
(d) Hydrogen-related hazard score  
Figure 8: Hazard-based classification of generated reactor trajectories, classification thresholds, and temperature- and hydrogen-related contributions to the hazard score.

## 4 Results and Discussion

This section will evaluate: (i) the performance of the proposed Pg-CDDPM method on synthesizing operating trajectories for the WGS reaction; and (ii) the critical role that synthetic data plays in assisting the monitoring and diagnosis of rare-event cases. First, the proposed diffusion model is trained using ground-truth trajectory data generated by the reactor simulator. After training, the model is used to produce synthetic profiles for different operating conditions as in Tables $5 \mathrm { - } 7 ,$ particularly the disaster case where ground-truth training data are completely absent (an extrapolation problem). Physical consistency and closeness to ground-truth data are the main criteria to assess the quality of synthetic data. Further, the synthetic data are integrated with the raw data to address the data scarcity issue, especially for rare-event scenarios. The mixed datasets will then be leveraged to diagnose different hazard classes, by training a classifier that reliably classify these classes for any future operating data. The classification performance is used to assess the effectiveness of the entire synthesis-and-diagnosis framework.

Table 2: Hyperparameters used for training the proposed $\mathrm { P g \mathrm { - } }$ CDDPM.
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Deep-learning framework</td><td>PyTorch</td><td>Beta schedule</td><td>Linear</td></tr><tr><td>Optimizer</td><td>Adam</td><td> $\beta _ { \mathrm { s t a r t } }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 4 }$ </td><td> $\beta _ { \mathrm { e n d } }$ </td><td> $1 0 ^ { - 2 }$ </td></tr><tr><td>Loss function</td><td>MSE</td><td>Input channels</td><td>9</td></tr><tr><td>Batch size</td><td>5</td><td>Output channels</td><td>7</td></tr><tr><td>Number of epochs</td><td>6000</td><td>Base channels</td><td>64</td></tr><tr><td>Random seed</td><td>42</td><td>Activation function</td><td>SiLU</td></tr><tr><td>Diffusion time steps (T)</td><td>1000</td><td>Physics-loss weight (λ)</td><td>0.3</td></tr><tr><td colspan="2"></td><td>GPU</td><td>NVIDIA RTX 4000</td></tr></table>

Table 3: Training hyperparameters and architecture of the GRU classifier.
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Deep-learning framework</td><td>PyTorch</td><td>Fully connected activation</td><td>ReLU</td></tr><tr><td>Sequence length</td><td>100 time steps</td><td>Number of output classes</td><td>4</td></tr><tr><td>Number of input features</td><td>7</td><td>Loss function</td><td>Cross-entropy</td></tr><tr><td>Input features</td><td> $\mathrm { C O } , \mathrm { H } _ { 2 } \mathrm { O } , \mathrm { C O } _ { 2 } , \mathrm { H } _ { 2 } , T , T _ { c } , q$ </td><td>Optimizer</td><td>Adam</td></tr><tr><td>GRU hidden size</td><td>64</td><td>Learning rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Number of GRU layers</td><td>8</td><td>Batch size</td><td>50</td></tr><tr><td>GRU dropout rate</td><td>0.2</td><td>Number of epochs</td><td>300</td></tr><tr><td>Fully connected architecture</td><td> $6 4  6 4  4$ </td><td>Random seed</td><td>42</td></tr></table>

## 4.1 Data Synthesis of the Proposed Pg-CDDPM

## 4.1.1 Training trajectory generation

To construct the training dataset for the Pg-CDDPM, we used the nonlinear CSTR model described in Section 3.1 to generate dynamic trajectories. Each trajectory consists of 100 time steps. The initial reactor conditions were randomly selected within the ranges listed in Table 5 to produce diverse training trajectories. The inputs were varied within the ranges given in Table 4 using randomly generated piecewise-constant input profiles with an input holding time of 10 simulation steps. The feed conditions used to generate each trajectory are listed in Table 5 . Conditioning features were varied within the ranges listed in Table 6 to create different operating conditions. The ranges of f and $f _ { E _ { a } }$ used to generate the training trajectories are summarized in Table 7.

## 4.1.2 Ablation studies for interpolation and extrapolation

We have conducted ablation studies to evaluate the contribution of the physics-guided loss to the trajectory generation performance of the proposed model. The CDDPM was trained and evaluated both without and with the physics-guided loss while keeping the remaining model architecture and training settings unchanged.

Figs. 9 and 10 present a representative interpolation case $( f = 0 . 7 2 , f _ { E _ { a } } = 1 . 3 0 )$ without and with the physics-guided loss, respectively. In both cases, the generated trajectories closely agree with the simulated reactor temperature and the concentrations of $\mathrm { C O , H _ { 2 } O , }$ , and $\mathrm { H _ { 2 } }$ throughout the entire time horizon. Without the physics-guided loss, the temperature RMSE is approximately

Table 4: Ranges of input variables used for generating the input profiles.
<table><tr><td>Input Variable</td><td>Symbol Range</td><td></td></tr><tr><td>Coolant temperature</td><td> $T _ { c }$ </td><td> $3 5 0 { - } 5 0 0 \mathrm { K }$ </td></tr><tr><td>Feed flow rate</td><td> $F$ </td><td> $7 0 - 1 5 0 \mathrm { L m i n ^ { - 1 } }$ </td></tr><tr><td>Input holding time</td><td> $N _ { \mathrm { h o l d } }$ </td><td>10 time steps</td></tr></table>

Table 5: Initial reactor and feed conditions used for generating the training trajectories. Concentrations are in mol L<sup>−1</sup> and temperatures are in K.
<table><tr><td colspan="4">Initial reactor conditions</td><td colspan="3">Feed conditions</td></tr><tr><td>Parameter</td><td>Symbol</td><td>Value</td><td>Parameter</td><td></td><td>Symbol</td><td>Value</td></tr><tr><td>Initial CO concentration</td><td> $C _ { \mathrm { C O , 0 } }$ </td><td>1.5-2.0</td><td>Feed CO concentration</td><td></td><td> $C _ { f , \mathrm { C O } }$ </td><td>1.5</td></tr><tr><td>Initial  $_ \mathrm { H _ { 2 } O }$  concentration</td><td> $C _ { \mathrm { H _ { 2 } O , 0 } }$ </td><td>3.0-3.5</td><td>Feed  $_ \mathrm { H _ { 2 } O }$ </td><td>concentration</td><td> $C _ { f , \mathrm { H _ { 2 } O } }$ </td><td>3.0</td></tr><tr><td>Initial  $\mathrm { C O _ { 2 } }$  concentration</td><td> $C _ { \mathrm { C O _ { 2 } , 0 } }$ </td><td>0.2</td><td>Feed  $\mathrm { C O _ { 2 } }$ </td><td>concentration</td><td> $C _ { f , \mathrm { C O _ { 2 } } }$ </td><td>0.0</td></tr><tr><td>Initial  $\mathrm { H _ { 2 } }$  concentration</td><td> $C _ { \mathrm { H _ { 2 } , 0 } }$ </td><td>0.2</td><td>Feed  $\mathrm { H _ { 2 } }$ </td><td>concentration</td><td> $C _ { f , \mathrm { H _ { 2 } } }$ </td><td>0.0</td></tr><tr><td>Initial reactor temperature</td><td> $T _ { 0 }$ </td><td>500-550</td><td>Feed temperature</td><td></td><td> $T _ { f }$ </td><td>550</td></tr></table>

Table 6: Process parameters used in the WGS reactor model.
<table><tr><td>Parameter</td><td>Symbol Value</td><td></td><td>Parameter</td><td>Symbol Value</td><td></td></tr><tr><td>Reactor volume</td><td>V</td><td>100 L</td><td>Base overall heat transfer coefficient UA</td><td></td><td> $5 . 0 \cdot 1 0 ^ { 4 }$ </td></tr><tr><td>Fluid density</td><td> $\rho$ </td><td> $1 0 0 0 \mathrm { k g } \mathrm { m } ^ { - 3 }$ </td><td>Heat transfer efficiency factor</td><td>f</td><td> $0 . 6 { - } 1 . 0$ </td></tr><tr><td>Heat capacity</td><td> $C _ { p }$ </td><td> $0 . 2 3 9 \mathrm { k J } \mathrm { k g } ^ { - 1 } \mathrm { K } ^ { - 1 }$ </td><td>Effective heat transfer coefficient</td><td> $U A _ { \mathrm { e f f } }$ </td><td> $U A \cdot f$ </td></tr><tr><td>Pre-exponential factor</td><td> $k _ { 0 }$ </td><td> $1 0 ^ { 1 0 }$ </td><td>Base activation energy</td><td> $E _ { a }$ </td><td> $7 . 2 \cdot 1 0 ^ { 4 } ~ \mathrm { J m o l ^ { - 1 } }$ </td></tr><tr><td>Universal gas constant</td><td> $R$ </td><td> $8 . 3 1 4 ~ \mathrm { J } \mathrm { m o l } ^ { - 1 } \mathrm { K } ^ { - 1 }$ </td><td>Activation energy factor</td><td> $f _ { E _ { a } }$ </td><td>1.0-1.5</td></tr><tr><td>Heat of reaction</td><td> $\Delta H _ { r }$ </td><td> $. 4 . 1 \cdot 1 0 ^ { 4 } ~ \mathrm { J m o l ^ { - 1 } }$ </td><td>Effective activation energy</td><td> $E _ { a , \mathrm { e f f } }$ </td><td> $E _ { a } \cdot f _ { E _ { a } }$ </td></tr></table>

Table 7: Ranges of degradation factors used to generate the training trajectories.
<table><tr><td>Trajectory No.</td><td>Heat Transfer Efficiency Factor f</td><td>Activation Energy Factor  $f _ { E _ { a } }$ </td></tr><tr><td>1-30</td><td>0.6-1.0</td><td>1.0-1.2</td></tr><tr><td>31-60</td><td>0.6-1.0</td><td>1.2-1.3</td></tr><tr><td>61-90</td><td>0.6-1.0</td><td>1.3-1.4</td></tr><tr><td>91-120</td><td>0.6-1.0</td><td>1.4-1.45</td></tr><tr><td>121-130</td><td>0.6-1.0</td><td>1.45-1.5</td></tr></table>

5 K, while the concentration RMSEs are 0.0368, 0.0377, and 0.0348 mol $\mathrm { L } ^ { - 1 }$ for $\mathrm { C O , H _ { 2 } O , }$ and $\mathrm { H _ { 2 } } ,$ respectively. After incorporating the physics-guided loss, these values slightly decrease to 4 K and 0.0283, 0.0264, and 0.0267 mol $\mathrm { L } ^ { - 1 }$ , respectively. Table 8 summarizes the RMSE comparison for this interpolation case. These results indicate that the proposed model accurately captures the nonlinear reactor dynamics within the interpolation region.

The effect of physics guidance is more pronounced in the extrapolation region. At $f = 0 . 5 5$ and $f _ { E _ { a } } = 1 . 5 5$ , the model without the physics-guided loss yields a temperature RMSE of 26.6 K and concentration RMSEs of 0.2643, 0.2763, and 0.2573 mol $\mathrm { L ^ { - 1 } f o r C O , H _ { 2 } O , }$ and $\mathrm { H _ { 2 } } ,$ , respectively, as shown in Fig. 11. After incorporating the physics-guided loss, these values decrease substantially to 3.67 K and 0.0262, 0.0297, and 0.0227 mol $\mathrm { L } ^ { - 1 }$ , respectively, as shown in Fig. 12. Table 9 summarizes the RMSE comparison for this extrapolation case. The substantial improvement shows that the physics-guided loss helps the model generate more accurate and physically consistent trajectories under unseen degradation conditions.

![](images/238a62a5b44260960556531c33f4a4187001a4fce9031f271c23dffdbcc1ffd8.jpg)  
Figure 9: Comparison between synthetic and ground-truth reaction profiles (interpolation) for heat transfer efficiency $f = 0 . 7 2$ and activation energy $f _ { E _ { a } } = 1 . 3 0$ without the physics-guided loss.

![](images/da1e503bcfbe2d4d35d463f90d7ae6f20bbfc8c2a6d9d9d1404149d33ec15331.jpg)  
Figure 10: Comparison between synthetic and ground-truth reaction profiles (interpolation) for heat transfer efficiency $f = 0 . 7 2$ and activation energy $f _ { E _ { a } } = 1 . 3 0$ with the proposed Pg-CDDPM.

Table 8: RMSE comparison for the diffusion model without and with physics-guided loss for the interpolation problem at $f = 0 . 7 2$ and $f _ { E _ { a } } = 1 . 3 0$
<table><tr><td>Model</td><td>Temperature (K)</td><td>CO</td><td>H2O</td><td>H2</td></tr><tr><td>Without physics-guided loss</td><td>5.0</td><td>0.0368</td><td>0.0377</td><td>0.0348</td></tr><tr><td>With physics-guided loss</td><td>4.0</td><td>0.0283</td><td>0.0264</td><td>0.0267</td></tr></table>

![](images/0a3936cff90691dac5055bf373bf4441f1fd1448ad4aefc57bcc4972d5253f9e.jpg)  
Figure 11: Comparison between synthetic and ground-truth reaction profiles (extrapolation) for heat transfer efficiency $f = 0 . 5 5$ and activation energy $f _ { E _ { a } } = 1 . 5 5$ without the physics-guided loss.

The conventional CDDPM learns only the distribution of the original training trajectories and therefore struggles to generate accurate trajectories in the extrapolation region, where no training data are available. In contrast, the proposed physics-guided loss incorporates the reaction model Eqs. (10)–(16) to learn from both the data and governing equations. Thus it can generate more accurate trajectories outside the training region. This improvement is further demonstrated through an analysis of the RMSE distributions in both interpolation and extrapolation regions.

## 4.1.3 RMSE analysis

For analyzing the RMSE distributions of synthetic profiles, 500 trajectories were generated in both the interpolation and extrapolation regions using conventional data-driven CDDPM and the proposed Pg-CDDPM. Fig. 13 shows the trajectory temperature RMSE distributions under interpolation and extrapolation operating conditions. In the interpolation region, the proposed Pg-CDDPM shows a narrower error distribution than the conventional CDDPM. The improvement becomes more visible in the extrapolation region, where both the median and spread of error distributions are significantly lower for the proposed Pg-CDDPM than for the conventional CDDPM.

![](images/c193cdd282181b252da2fa1e55720ea712dc3830a6f82a70332fb8cbc0ec3c3f.jpg)  
Figure 12: Comparison between synthetic and ground-truth reaction profiles (extrapolation) for heat transfer efficiency $f = 0 . 5 5$ and activation energy $f _ { E _ { a } } = 1 . 5 5$ with the proposed Pg-CDDPM.

Table 9: RMSE comparison without and with the physics-guided loss in the extrapolation region at $f = 0 . 5 5$ and $f _ { E _ { a } } = 1 . 5 5$
<table><tr><td>Model</td><td>Temperature (K)</td><td>CO</td><td> $\mathbf { H } _ { 2 } \mathbf { O }$ </td><td> $\mathbf { H } _ { 2 }$ </td></tr><tr><td>Without physics-guided loss</td><td>26.6</td><td>0.2643</td><td>0.2763</td><td>0.2573</td></tr><tr><td>With physics-guided loss</td><td>3.67</td><td>0.0262</td><td>0.0297</td><td>0.0227</td></tr></table>

Figs. 14 (a) and (b) show the trajectory RMSE distributions for species concentrations in the interpolation and extrapolation regions, respectively. A similar trend is also observed in the RMSE distributions of the species concentrations. The spread of the error distribution is smaller for our Pg-CDDPM than for the conventional CDDPM, and the improvement is more visible in the extrapolation region. This demonstrates the main motivation behind using the $\mathrm { P g } .$ -CDDPM to generate rare-event trajectories. The overall ablation study and RMSE analysis demonstrate the contribution of the physics-guided loss to the trajectory-generation performance of the proposed model. Without including physics, the model struggles to generate accurate reactor trajectories in the extrapolation region.

![](images/a879884be101b555e29ca3edf62b7afc47b608abd2c8b4684d470cdecab1f3e5.jpg)  
Figure 13: Comparison of the trajectory temperature RMSE distributions for the conventional C-DDPM and the proposed Pg-CDDPM under interpolation and extrapolation operating conditions.

## 4.2 Trajectory Clustering, Augmentation, and Classifier Performance

## 4.2.1 Trajectory clustering based on the hazard score index

As mentioned before, each of the generated trajectories is classified into one of four different operating classes based on its corresponding hazard score index. Fig. 15 shows the distribution of the simulator-generated trajectories with respect to $f , f _ { E a } ,$ , and average $\mathrm { H _ { 2 } }$ concentration $H _ { \mathrm { 2 , a v g } } .$ As the activation energy f increases (catalyst degradation) and the heat transfer efficiency $f _ { E a }$ decreases (fouling), the trajectories generally shift toward more severe operating regions, with $H _ { \mathrm { 2 , a v g } }$ decreasing substantially. No disaster-class trajectories are observed within the original training range of $f = 0 . 6 \substack { - 1 . 0 }$ and $f _ { E _ { a } } = 1 . 0 – 1 . 5$ . This represents a real-life challenge associated with the lack of rare-event data under disaster operating conditions and motivates the use of the proposed Pg-CDDPM for rare-event data augmentation and diagnosis.

## 4.2.2 Trajectory augmentation

From Fig. 15, we can observe that the original simulator-generated trajectories cover a range of $f = 0 . 6 \substack { - 1 . 0 }$ and $f _ { E _ { a } } = 1 . 0 – 1 . 5 ,$ , and no disaster-class trajectories are observed within the training range. To increase the amount of rare-event data, the Pg-CDDPM was used to generate additional trajectories beyond the training range. After data augmentation, the operating range was extended to $f = 0 . 4 5  – 1 . 0$ and $f _ { E _ { a } } = 1 . 0 – 1 . 6 ,$ as in Fig. 16. The augmented dataset contains additional trajectories in the severe operating regions, including the disaster class, providing a more balanced distribution of trajectories for training the classifier.

![](images/600d69a41b5e63f3d61efb008e2725b80cc88b3e435c89cd2a5aff037e3ae82b.jpg)

(a) Interpolation conditions  
Extrapolation Performance  
![](images/52d405972bd95b6d2a060646535737adbdf120843d03b0c876b5a838b03e82b7.jpg)  
(b) Extrapolation conditions  
Figure 14: Comparison of the trajectory RMSE distributions for the species concentrations under (a) interpolation and (b) extrapolation operating conditions.

## 4.2.3 Operating region diagnosis

The performance of the operating region classifier, particularly structured via a GRU network, was evaluated under three training cases: the original simulator-generated dataset (Case 1), the dataset augmented using the proposed Pg-CDDPM (Case 2), and the dataset augmented using the conventional CDDPM (Case 3). The classifier uses seven process variables we considered as inputs, and outputs four operating classes: normal, slightly abnormal, extreme, and disaster. As in Table 10, Case 1 contains 415 trajectories for each of the normal, slightly abnormal, and extreme classes, with no disaster-class trajectories. For Cases 2 and 3, data augmentation was performed to obtain 415 trajectories for each of the four operating classes. Consequently, a total of 1,245, 1,660, and 1,660 trajectories were used for training in Cases 1, 2, and 3, respectively. A separate test dataset containing 700 trajectories was used to evaluate the classification performance. The test dataset consists of 146 normal, 185 slightly abnormal, 179 extreme, and 190 disaster trajectories. The same test dataset was used for all three cases to ensure a consistent comparison of the classification performance.

![](images/2bbc1bbd7166ce24b7b070ae924ce117b93d47e0aeed65b422e6cbd121aead35.jpg)  
Figure 15: Clustering of the simulator-generated trajectories based on the hazard score with respect to $f , f _ { E _ { a } } ,$ , and $H _ { \mathrm { 2 , a v g } }$

![](images/c2b9408dd07829f0896045b78b7342af905ddc4882e6c42dd77989f13c589121.jpg)  
Figure 16: Distribution of the augmented trajectories based on the hazard score with respect to the $f , f _ { E _ { a } }$ , and $H _ { \mathrm { 2 , a v g } }$

Table 10: Class distribution of the training datasets for the three cases.
<table><tr><td>Training dataset</td><td>Normal</td><td>Slight</td><td>Extreme</td><td>Disaster</td><td>Total</td></tr><tr><td>Case 1: Simulation</td><td>415</td><td>415</td><td>415</td><td>0</td><td>1245</td></tr><tr><td>Case 2: Simulation + Pg-CDDPM</td><td>415</td><td>415</td><td>415</td><td>415</td><td>1660</td></tr><tr><td>Case 3: Simulation + Conventional CDDPM</td><td>415</td><td>415</td><td>415</td><td>415</td><td>1660</td></tr></table>

The diagnosis performance for three cases is compared using the confusion matrices shown in Fig. 17. For Case 1, the classifier performs well for the normal, slightly abnormal, and extreme classes. However, all 190 disaster trajectories are misclassified as extreme because the classifier was not exposed to disaster-class data during training. For Case 2, after augmenting the training dataset with Pg-CDDPM, the classification of the disaster class improves significantly. Out of 190 disaster trajectories, 175 are correctly classified, while only 15 are misclassified as extreme. For the extreme class, 163 out of 179 trajectories are correctly classified, with 12 misclassified as disaster and 4 grouped as slightly abnormal. For Case 3, data augmentation with CDDPM also improves the identification of the disaster class compared with Case 1. However, 151 trajectories are misclassified as the extreme class. Overall, the Pg-CDDPM augmented dataset provides better classification of the extreme and disaster conditions compared with the conventional CD-DPM augmented dataset. Other classification metrics, including precision, recall, and F1-score, are presented in Fig. 18, with their corresponding values listed in Table 11.

## 5 Conclusion

In this work, we proposed a Pg-CDDPM framework for rare-event reaction profile generation and fault diagnosis for the WGS reaction. The proposed method incorporates the reactor governing laws into the diffusion training process through a physics-guided loss. Furthermore, the model was conditioned on the heat transfer efficiency factor and activation energy factor. These features are critical in determining the operating region and status of the reaction. The presence of governing laws enables reliable extension of the diffusion model to generate out-of-domain profiles for extrapolation to risky conditions where data is highly scarce or even completely absent. Numerical studies show that the generated synthetic profiles from physics-guided diffusion model outperforms those from traditional diffusion model in both interpolation and extrapolation regimes. Further, the generated rare-event trajectories were used to augment the training dataset for the extreme and disaster operating classes. With data augmentation, the operating region classifier can effectively diagnose the unseen rare events.

![](images/2caafa0e7d41f5173286ee45300e3f4625f8338d247938cad6583e4c521649ad.jpg)

![](images/f992b7709e010f5965a13518af9ef984e571d6be0640825cbc8ecaad4e4c51a0.jpg)

![](images/689132013b36b31ca23cceb30ca9afb22ddcba25bbb93c1a1e2c57e5b64f5e51.jpg)

Figure 17: Confusion matrices of the GRU classifier for the three training cases.  
![](images/5d3a0d264428f8b3762f891e23e179b73f48eba3f8f3434d40cb1b8b78eab3b5.jpg)

![](images/782c63a9016c4f8254c7a8697d203a0e51332c1c20293ec59d99c6913f9a8abf.jpg)

![](images/f407e05180b0a0999df39628797e92e012ffc5934c610e0c52cc6e0ef5531987.jpg)  
Figure 18: Comparison of the class-wise precision, recall, and F1-score obtained using the three training datasets.

## 6 Acknowledgment

The authors acknowledge the support from Texas Tech University. Md Abrar Rafid Siddique and Bibek Aryal acknowledge the Distinguished Graduate Student Assistantships (DGSA) from Texas Tech University.

Table 11: Comparison of the classification performance for the three training cases. Case 1: Simulation; Case 2: Simulation + Pg-CDDPM; Case 3: Simulation + C-DDPM.
<table><tr><td>Metric</td><td>Case</td><td>Normal</td><td>Slightly Abnormal</td><td>Extreme</td><td>Disaster</td></tr><tr><td rowspan="3">Precision</td><td>Case 1</td><td>0.93</td><td>0.95</td><td>0.48</td><td>0.00</td></tr><tr><td>Case 2</td><td>0.93</td><td>0.92</td><td>0.91</td><td>0.94</td></tr><tr><td>Case 3</td><td>0.89</td><td>0.93</td><td>0.51</td><td>0.67</td></tr><tr><td rowspan="3">Recall</td><td>Case 1</td><td>0.94</td><td>0.92</td><td>1.00</td><td>0.00</td></tr><tr><td>Case 2</td><td>0.93</td><td>0.93</td><td>0.91</td><td>0.92</td></tr><tr><td>Case 3</td><td>0.93</td><td>0.91</td><td>0.88</td><td>0.21</td></tr><tr><td rowspan="3">F1-score</td><td>Case 1</td><td>0.93</td><td>0.93</td><td>0.65</td><td>0.00</td></tr><tr><td>Case 2</td><td>0.93</td><td>0.93</td><td>0.91</td><td>0.93</td></tr><tr><td>Case 3</td><td>0.91</td><td>0.92</td><td>0.65</td><td>0.31</td></tr></table>

## References

[1] Mehmet Bilgili, Sergen Tumse, and Sude Nar. Comprehensive overview on the present state and evolution of global warming, climate change, greenhouse gasses and renewable energy. Arabian Journal for Science and Engineering, 49(11):14503–14531, 2024.

[2] Seyed Ehsan Hosseini. Fossil fuel crisis and global warming. In Fundamentals of Low Emission Flameless Combustion and its Applications, pages 1–11. Elsevier, 2022.

[3] Sajjad Rezaei, Alejandra Hormaza Mejia, Yanchen Wu, Jeffrey Reed, and Jack Brouwer. Global warming impacts of the transition from fossil fuel conversion and infrastructure to hydrogen. Applied Energy, 397:126363, 2025.

[4] Jiannan Wang and Waseem Azam. Natural resource scarcity, fossil fuel energy consumption, and total greenhouse gas emissions in top emitting countries. Geoscience Frontiers, 15(2):101757, 2024.

[5] Roshni Patel, Prashandan Varatharajan, Qi Zhang, Ze Li, and Sai Gu. Catalysts in the watergas shift reaction: A comparative review of industrial and academic contributions. Carbon Capture Science & Technology, 15:100388, 2025.

[6] Adnan Midilli, Mo Ay, Ibraham Dincer, and Marc A Rosen. On hydrogen and hydrogen energy strategies: I: current status and needs. Renewable and Sustainable Energy Reviews, 9(3):255– 271, 2005.

[7] Leila Dehimi, Oualid Alioui, Yacine Benguerba, Krishna Kumar Yadav, Javed Khan Bhutto, Ahmed M Fallatah, Tanuj Shukla, Maha Awjan Alreshidi, Marco Balsamo, Michael Badawi, et al. Hydrogen production by the water-gas shift reaction: A comprehensive review on catalysts, kinetics, and reaction mechanism. Fuel Processing Technology, 267:108163, 2025.

[8] Ru-Ri Lee, I-Jeong Jeon, Won-Jun Jang, Hyun-Seog Roh, and Jae-Oh Shim. Advances in catalysts for water–gas shift reaction using waste-derived synthesis gas. Catalysts, 13(4):710, 2023.

[9] Erlisa Baraj, Karel Ciahotny, and Tom \` a´s Hlin ˇ cˇ´ık. The water gas shift reaction: Catalysts and reaction mechanism. Fuel, 288:119817, 2021.

[10] Andrew M Beale, Emma K Gibson, Matthew G O’Brien, Simon DM Jacques, Robert J Cernik, Marco Di Michiel, Paul D Cobden, Ozlem Pirgon-Galin, Leon Van De Water, Michael J Wat- <sup>¨</sup> son, et al. Chemical imaging of the sulfur-induced deactivation of Cu/ZnO catalyst bodies. Journal of Catalysis, 314:94–100, 2014.

[11] Selin Bac, Seda Keskin, and Ahmet K Avci. Modeling and simulation of water-gas shift in a heat exchange integrated microchannel converter. International Journal of Hydrogen Energy, 43(2):1094–1104, 2018.

[12] Wei-Hsin Chen, Mu-Rong Lin, Tsung Leo Jiang, and Ming-Hong Chen. Modeling and simulation of hydrogen generation from high-temperature and low-temperature water gas shift reactions. International Journal of Hydrogen Energy, 33(22):6644–6656, 2008.

[13] Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in Neural Information Processing Systems, 27, 2014.

[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020.

[15] Jiaqi Zheng, Xuhua Shi, Feifan Shen, and Lingjian Ye. A filtered time conditional denoising diffusion probabilistic model for virtual sample generation of industrial soft sensors with limited data. Measurement, page 120718, 2026.

[16] Yang Li, Han Meng, Zhenyu Bi, Ingolv T Urnes, and Haipeng Chen. Population aware diffusion for time series generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 18520–18529, 2025.

[17] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171. PMLR, 2021.

[18] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer.¨ High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[19] Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, and Bryan Catanzaro. Diffwave: A versatile diffusion model for audio synthesis. arXiv preprint arXiv:2009.09761, 2020.

[20] Lifeng Shen and James Kwok. Non-autoregressive conditional diffusion models for time series prediction. In International Conference on Machine Learning, pages 31016–31029. PMLR, 2023.

[21] Andrew Millard and Henrik Pedersen. Particle-guided diffusion for gas-phase reaction kinetics. arXiv preprint arXiv:2603.05139, 2026.

[22] Haotian Chen, Guohua Chen, Yimeng Zhao, and Qiming Xu. Spatiotemporal prediction of gas dispersion field in chemical industrial parks based on the improved conditional denoising diffusion probability model. Computers & Chemical Engineering, page 109711, 2026.

[23] Amira Alakhdar, Barnabas Poczos, and Newell Washburn. Diffusion models in de novo drug design. Journal of Chemical Information and Modeling, 64(19):7238–7256, 2024.

[24] Liang Wang, Chao Song, Zhiyuan Liu, Yu Rong, Qiang Liu, and Shu Wu. Diffusion models for molecules: A survey of methods and tasks. arXiv preprint arXiv:2502.09511, 2025.

[25] Zhiye Guo, Jian Liu, Yanli Wang, Mengrui Chen, Duolin Wang, Dong Xu, and Jianlin Cheng. Diffusion models in bioinformatics and computational biology. Nature Reviews Bioengineering, 2(2):136–154, 2024.

[26] Ye Yuan, Jiaming Song, Umar Iqbal, Arash Vahdat, and Jan Kautz. Physdiff: Physics-guided human motion diffusion model. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16010–16021, 2023.

[27] Hao Wang, Jindong Han, Wei Fan, Weijia Zhang, and Hao Liu. Phyda: Physics-guided diffusion models for data assimilation in atmospheric systems. arXiv preprint arXiv:2505.12882, 2025.

[28] Zhongxin Yang, Yuanwei Bin, Xiang IA Yang, and Shiyi Chen. Least-action-guided diffusion for physical extrapolation. arXiv preprint arXiv:2606.11277, 2026.

[29] Lucas Nieto Degliuomini, Sebastian Biset, Patricio Luppi, and Marta S Basualdo. A rigorous computational model for hydrogen production from bio-ethanol to feed a fuel cell stack. International Journal of Hydrogen Energy, 37(4):3108–3129, 2012.

[30] Mark E. Davis and Robert J. Davis. Fundamentals of Chemical Reaction Engineering. McGraw-Hill Higher Education, Boston, 1 edition, 2003.

[31] Qimin Deng, Peirong Lu, Shuyun Zhao, and Naiming Yuan. U-net: A deep-learning method for improving summer precipitation forecasts in china. Atmospheric and Oceanic Science Letters, 16(4):100322, 2023.

[32] Dongyang Kuang. A 1d convolutional network for leaf and time series classification. arXiv preprint arXiv:1907.00069, 2019.

[33] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 234–241. Springer, 2015.