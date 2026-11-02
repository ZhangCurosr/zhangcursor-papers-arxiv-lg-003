---
title: "What-You-Sign-Is-Not-What-You-Run-The-Verifiability-Gap-in-Q"
source: https://arxiv.org/pdf/2610.19498v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:09:37"
field: "可信AI系统与机器学习供应链安全"
keywords: ["LLM quantization", "supply chain verifiability", "reproducible builds", "model signing", "GGUF", "floating-point non-determinism", "independence verification rate"]
innovations: ["提出IVR指标并首次测量GGUF生态独立可验证率为0/198", "定位缺失关键坐标（FP收缩设置与每张量配方）并实现7/8 bit-exact重建", "划定B>0 regime下自然方差与篡改检测的安全边界"]
benchmarks: ["Hugging Face GGUF下载量Top 200", "wikitext-2 perplexity"]
---

# 论文速读：What-You-Sign-Is-Not-What-You-Run-The-Verifiability-Gap-in-Q

## 一句话总结
本文首次对Hugging Face最受欢迎的GGUF量化仓库进行流行度采样测量，提出独立可验证率（IVR）指标，发现当前生态中没有任何仓库能支持第三方bit-exact重建；通过实验定位缺失关键坐标（构建环境的FP收缩设置与每张量量化配方），并提出一个基于精确重推导的验证协议以弥合量化供应链的信任缺口。

## 研究问题与动机
1. **签名与运行物分离**：现有模型签名（如Sigstore Model Signing、NVIDIA NGC）仅对全精度权重W的哈希进行认证，但终端用户实际运行的是量化产物Q；当前规范未建立H(W)与Q之间的可验证密码学绑定。
2. **转换过程非确定性**：量化变换T的输入远超W，还包括量化配方θ、校准输入C（imatrix）、执行环境E（编译器、架构、FP收缩设置）及非权重组件M；在已披露规范下Q不唯一，跨平台重跑可产生不同字节。
3. **现有 attestation 规范不足**：IETF ai-model-lifecycle-attestation草案、CycloneDX ML-BOM等可记录来源与工具链信息，但未强制绑定校准输入C与执行环境E，无法支持独立重建验证。
4. **可重编译构建范式未迁移至ML量化**：传统可重编译构建通过记录环境或使环境无关化实现bit-exact重建，但量化中的非确定性存在于权重数值内容而非仅元数据，现有做法无法直接套用。

## 核心贡献（创新点）
1. **首次流行度采样的GGUF独立可验证率测量**：提出IVR指标并实测为0/198，区分了“源可重建”（148/198）与“已记录”（环境记录0/198）的差距，揭示缺失核心在于执行环境。
2. **构造性重建试验定位失败坐标**：在4个公开输入仓库上的重建试验表明，仅靠模型卡信息无法重推Q4_K_M文件；补充**构建FP收缩设置**与**每张量量化配方**后，x86上7/8目标量化文件bit-exact重建，证明这两个坐标是关键缺失。
3. **精确词汇化转换模糊度与必要条件**：引入非环境输入X̄、环境E、可接受输出集𝒜、模糊度预算B（0/1指示器）等概念，形式化证明完整重建规范必须约束E（Observation 1），为后续测量提供理论支架。
4. **安全特征分析划定检测边界**：在B>0 regime下量化自然方差与篡改分布的差异；均匀 nibble-flip 篡改可在约0.02%–0.2%权重阈值外被统计检测，但**定向嵌入编辑**可规避该统计，凸显精确重建的必要性。
5. **提出基于B=0的验证协议与工程建议**：主张量化方应签署绑定(H(W), θ, C, E, M)的manifest，独立重建器重推后比对哈希；建议llama.cpp/GGUF记录量化器版本与构建标志、合并PR #25353（-ffp-contract=off）。

## 方法详解
1. **输入分解与转换建模**：将量化变换写作Q = T(X̄; E)，其中X̄ = (W, θ, C, M)为非环境输入，E为执行环境。可接受输出集𝒜(X̄) = {T(X̄; E): E ∈ ℰ}，模糊度预算B(X̄) = sup_{Q1,Q2∈𝒜} d(Q1,Q2)，d为离散度量（字节不同则d=1）；|𝒜|=1 ⇔ B=0即输出唯一。
2. **独立可验证率（IVR）定义**：IVR = {bit-exact可重推导出产物数} / {评估产物数}。采用**记录输入协议**要求W、θ、C、E四项全部披露，否则视为不可验证；另统计较弱的**源可重建率**（仅要求W、C可获取）。
3. **抽样与评分仪器**：取Hugging Face API返回下载量前200的GGUF仓库（实际198个），通过HTTP range requests仅拉取代表性文件的GGUF头部（8–16 MB）解析key–value块，评分四输入清单：(i) fp16源W是否可获取；(ii)量化器及版本是否记录；(iii)校准imatrix C是否可获取；(iv)执行环境坐标E是否记录。
4. **重建试验设计**：选取4个bartowski仓库（Qwen2.5-7B-Instruct b3772、Qwen3.5-{0.8B,2B,4B} b9222）与1个Granite仓库（b10603），分别用仅模型卡信息（card）、card+每张量配方（+rec）、card+rec±环境组合进行重建，比对per-tensor sha256。
5. **环境必要性测试（M2）**：在arm64/clang与x86_64/gcc上各编译默认与-ffp-contract=off两种build，量化固定fp16至Q4_K_M、Q5_K_M、IQ4_XS、Q8_0四种类型；验证默认build跨架构3/4类型字节不同，禁用收缩后4/4类型bit-identical，代价仅ΔPPL=+0.06%。
6. **安全特征分析（M3）**：在解量化权重重空间度量Δw/σ分布；构建五对环境样本，测量自然方差与均匀nibble-flip篡改的幅度重叠系数；定义检测底限（AUC≥0.99所需最小篡改比例）；测试均匀缩放扰动、自适应放置、定向嵌入编辑三类变体。
7. **校准输入不可重计算证明**：尝试基于模型卡关联的语料库与fp16重算imatrix，发现entry结构完全匹配但importance值中位数相对误差5.5×10⁻⁵，导致重新量化改变141/335张量，证明C必须作为hash-bound输入而非重计算。

## 实验与结果
- **数据集**：Hugging Face下载量Top 200的GGUF仓库（198个有效，来自61个发布者），抽样时间2026年8月。
- **评估基线**：当前生态实践（GGUF元数据+模型卡披露）；对比工具ggufpacker与llama.cpp PR #25353。
- **主要结果**：
  1. **IVR = 0/198**：无任何仓库满足记录输入协议；源可重建率148/198（74.7%，允许基础模型仓库存在公开全精度权重）或111/198（56.1%，要求仓库自身提供fp16 GGUF）。
  2. **缺失结构**：198个仓库全部缺少环境/版本记录；其中50个还缺少fp16源或imatrix。
  3. **重建试验**：仅凭模型卡信息，4个Qwen仓库的Q4_K_M无一成功；补充每张量配方后，x86上7/8目标×量化文件bit-exact重建，arm64上5/8（残差为零块存储差异）。
  4. **FP收缩必要性**：b10448版本在-ffp-contract=off下四种类型跨双架构均bit-identical；qwen2.5-7B b3772因使用-march=native额外需锁SIMD ISA。
  5. **imatrix不可重计算**：重算重要性值中位数相对误差5.5×10⁻⁵，导致重新量化改变141/335张量。
  6. **质量代价**：-ffp-contract=off构建 vs 默认构建，Qwen2.5-1.5B Q4_K_M在wikitext-2上PPL 10.750 vs 10.744，ΔPPL=+0.06%（配对标准误0.005，统计不显著）。
  7. **安全特征**：Q4_K自然方差触及τ=0.1阈值的权重占比0.018%；均匀单步翻转检测底限0.02%，自适应放置隐藏容量约0.43%权重，定向嵌入编辑可完全规避统计检测。
- **最强结果**：在四Qwen+一Granite仓库上，补充两个卡片未记录的坐标后，x86平台7/8量化文件实现bit-exact重建，证明缺失坐标可被定位与记录。

## 相关工作脉络
1. **可重编译构建运动**（Debian实践、diffoscope、reprotest）：确立从源码到二进制独立验证范式；本文将其迁移至量化步骤，发现数值非确定性无法仅靠元数据规范化消除，必须使环境无关或记录环境。
2. **模型签名工作**（Sigstore Model Signing、NVIDIA NGC、OpenSSF Model Signing规范）：认证发布文件哈希与发布者身份；但未绑定量化输入C与E，无法验证Q对W的忠实性。
3. **生命周期attestation草案**（IETF ai-model-lifecycle-attestation）：记录源权重哈希、量化产物哈希、工具哈希与逐层误差容忍；本文指出其对C与E的绑定不足，且子阈值定向编辑可规避误差检查。
4. **ML-BOM透明清单**（CycloneDX）：枚举模型组件；同样未要求记录量化环境或校准输入，与本文主张的manifest绑定范围存在差距。
5. **已有确定性GGUF重建工具**（ggufpacker、llama.cpp PR #25353）：实现特定条件下的bit-exact重建；本文不声称发明这些工具，而是测量当前披露实践距离这些工具所预设前提还有多远。
6. **对抗性fp16攻击**（Mind the Gap等）：展示量化条件后门攻击；本文将其划出威胁模型（因为量化步骤本身忠实，重推导会复现攻击产物），凸显重推导协议只能检测cheating quantizer而非malicious W。

## 局限性与未来方向
1. **模型与类型覆盖有限**：仅测试Qwen 0.5B–14B与Granite 3B，量化类型限于Q4_K、Q5_K、Q6_K、IQ4_XS及Q8_0控制组；其他架构、更大规模模型、更多k-quant变体未验证。
2. **管道局限**：仅评估llama.cpp CPU量化；GPU imatrix生成、GPU原生管道（GPTQ/AWQ/bitsandbytes）可能引入额外非确定性，需另行研究。
3. **环境空间未完全特征化**：仅测试arm64/clang与x86_64/gcc两个(架构,编译器)对，其他编译器版本、CUDA构建、不同量化器release未涵盖；完整X*刻画留作未来工作。
4. **安全特征仅针对均匀篡改**：检测底限与适应性放置、定向编辑的结果基于单一模型家族与五对已知release环境；更广泛环境范围内的泛化未验证。
5. **对抗性fp16攻击未深入评估**：仅运行Mind the Gap开源代码在0.5B模型上未复现成功攻击，声明该威胁类别在重推导协议边界之外。
6. **未来方向**：完整特征化最小完备规范X*；将验证协议扩展至GPU原生管道与更多量化后端；研究统计检测与精确重建结合的混合方案；推动IETF/CycloneDX规范强制绑定C与E。

## 研究启发与可借鉴点
1. **IVR测量框架可迁移至其他ML制品**：将可重编译构建思想引入量化供应链，提出可按比例放大的 popularity-sampled 测量方法，适用于LoRA适配器、嵌入模型、多模态组件等场景的透明性评估。
2. **环境必要性实验设计**：通过交叉架构×编译标志的矩阵实验（双平台×双build×四类型）直接验证“环境是否影响输出”，可复用至其他确定性敏感管线（如训练数据shuffle顺序、优化器状态序列化）。
3. **可重跑审计工具链**：发布census.py评分脚本、GGUF header-probe、byte-region diff、detector等独立工具，与外部重建引擎解耦；这种“测量仪器”与“重建引擎”分离的设计便于社区独立复现与持续监控IVR趋势。
4. **安全特征分析方法**：在解量化权重重空间度量|Δw|/σ分布、计算重叠系数、定义检测底限（AUC≥0.99阈值）的方法，可推广至其他压缩后产物（如稀疏化、剪枝）的篡改检测研究。
5. **协议分层思维**：明确区分“B=0精确重建”（主协议）与“B>0统计检测/attestation”（残差机制）的适用边界，并量化采样检查的复杂度下界（Θ(N/m)）；为后续研究提供清晰的威胁分类与方案选型依据。

## 关键术语表
**Independent Verifiability Rate (IVR)**：在流行度采样生态中，独立第三方能bit-exact重推导出产物的仓库比例，本文定义为记录输入协议下的成功数/评估总数。
**非环境输入 X̄**：量化变换中与执行环境无关的输入集合，包括全精度权重W、量化配方θ、校准输入C（imatrix）、非权重组件M。
**执行环境 E**：影响量化输出的平台相关坐标，包括量化器release版本、编译器、目标架构、浮点收缩设置（-ffp-contract）、SIMD ISA等。
**模糊度预算 B**：度量给定X̄下输出唯一性的0/1指标；B=0表示所有可接受环境中输出字节相同，是精确哈希比对的前提。
**imatrix**：重要性矩阵，通过对校准语料运行模型得到的per-weight统计量，用于加权scale search；本文证明其跨架构不可重计算，必须hash-bound绑定。
**浮点收缩（FP contraction）**：编译器将a·b+c融合为单次FMA操作的优化；本文证明其在ggml-quants.c的scale-search循环中导致跨架构数值分歧，-ffp-contract=off可消除。
**k-quant**：GGUF中固定大小块的低比特量化格式（如Q4_K、Q5_K），每个块含整数code与浮点scale参数；scale search决定scale值从而影响存储字节。
**抽样检查（Spot-checking）**：仅重推产物子集（如Merkle树的部分superblock）以节省开销；本文证明检测稀疏篡改所需开销随N/m增长，极端情况下退化为全量重建。

## 可复现要素
- **数据集**：Hugging Face下载量Top 200 GGUF仓库列表已冻结并随artifacts发布；模型卡片、GGUF头部、公开imatrix、fp16源均来自公共仓库。
- **代码/权重开源**：所有artifacts开源（https://github.com/Rlonglong/gguf-verifiability-artifact），含census.py重跑脚本、模型卡审计 verdicts、GGUF header-probe、byte-region diff、detector工具及原始数据CSV/JSON。
- **关键超参/版本**：llama.cpp pinned builds b3772、b9222、b10448、b10603；编译标志-ffp-contract=off；评估数据集wikitext-2（584个chunks）；安全分析用环境对5组（含publisher artifact）；种子数5。
- **计算环境**：Apple M4（arm64/clang）、Linux x86_64（gcc）；CPU量化；无GPU实验。
- **重推导基准**：ggufpacker与PR #25353作为对照实现，本文工具独立于二者。
