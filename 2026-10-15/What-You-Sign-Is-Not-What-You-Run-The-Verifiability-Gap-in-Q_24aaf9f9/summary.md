---
title: "What-You-Sign-Is-Not-What-You-Run-The-Verifiability-Gap-in-Q"
source: https://arxiv.org/pdf/2610.19498v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:09:49"
---

# 论文速读：What-You-Sign-Is-Not-What-You-Run-The-Verifiability-Gap-in-Q

## 一句话总结
论文揭示了LLM量化供应链中"签名与执行不一致"的可验证性鸿沟：现有模型签名仅认证全精度权重 $W$ 的哈希，但用户实际运行的是量化产物 $Q$（如GGUF），而 $Q$ 的变换过程依赖的执行环境 $E$ 从未被记录，导致第三方无法独立bit-exact重建验证。作者在Hugging Face最流行198个GGUF仓库上首次测得IVR（Independent Verifiability Rate）为0/198，并定位出两个缺失坐标——FP-contraction设置与per-tensor量化配方——补充后7/8目标文件可在x86上完全重算。

## 研究问题与动机
1. **签名与执行的语义断裂**：Sigstore Model Signing、NVIDIA NGC 等机制只对 $H(W)$ 签名，但用户实际部署的是 $Q=\mathcal{T}(\bar{X};E)$，当前规范不绑定 $Q$ 与 $W$ 之间的变换关系。
2. **GGUF格式的结构性缺陷**：GGUF 元数据字段无构建环境（compiler、architecture、FP-contraction setting）的记录槽位，导致即使输入完全披露，重算也可能产生不同字节。
3. **浮点非确定性导致跨平台字节差异**：llama.cpp k-quant scale-search 循环受 `-ffp-contract` 影响，arm64/clang 默认融合乘加与 x86/gcc 默认行为不同，使相同输入在不同平台上产生不同量化结果。
4. **现有审计规范未覆盖关键输入**：IETF `ai-model-lifecycle-attestation` 草案和 CycloneDX ML-BOM 不要求绑定校准输入 imatrix（$C$）和执行环境（$E$），无法支撑独立重建验证。

## 核心贡献（创新点）
1. **首次提出并测量 IVR 指标**：在 popularity-sampled 198 个 GGUF 仓库上，recorded-inputs 协议下 IVR=0/198，首次量化了量化供应链的可验证性缺失规模。
2. **精确定位重算失败的两个缺失坐标**：card 未记录的 FP-contraction 设置与 hybrid 模型的 per-tensor recipe；补充后 7/8 目标量化文件在 x86 上 bit-exact 重算成功，fp16 转换跳也可重算。
3. **建立变换歧义的形式化框架**：引入 $\bar{X}/E$ 分解、admissible output set $\mathcal{A}(\bar{X})$ 与歧义预算 $B$，形式化证明 $E\in X^*$（环境必须被约束），并以 Observation 1 给出实证必要性。
4. **给出 $B{>}0$ regime 的安全特征化**：测量 benign 浮点方差分布与自然噪声的分离度，给出 per-tensor tail statistic 的检测下界（Q4 K 在已知环境对为 0.02%，五对校准扩展后升至约 0.2%），并证明 targeted embedding edit 可完全绕过统计检测。
5. **提出端到端验证协议**：transforming party 应对 $(H(W),\theta,C,E,M)$ 签名 manifest，独立重建者重算后发布到 append-only transparency log；在 $B{=}0$ 条件下可用单一编译器标志（`-ffp-contract=off`）实现跨平台确定性。

## 方法详解
- **输入分解**：$Q=\mathcal{T}(\bar{X};E)$，其中 $\bar{X}=(W,\theta,C,M)$ 为非环境输入（全精度权重、量化配方、imatrix校准输入、tokenizer/chat template等），$E$ 为执行环境（量化器版本、编译器、架构、FP-contraction 设置）。
- **歧义预算定义**：$\mathcal{A}(\bar{X})=\{\mathcal{T}(\bar{X};E):E\in\mathcal{E}\}$，$B(\bar{X})=\sup_{Q_1,Q_2\in\mathcal{A}}d(Q_1,Q_2)$，$d$ 为离散字节距离；$B{=}0$ 当且仅当输出唯一，是 hash 比较的前提。
- **IVR 计算协议**：四输入检查清单——(i) fp16/W 是否可获取、(ii) 量化器及版本是否记录、(iii) imatrix $C$ 是否可获取、(iv) 任意 $E$ 坐标是否记录；全部满足则计为 bit-exact verifiable。
- **Trial re-derivation**：选取 4 个 bartowski 发布的 Qwen/G ranite 仓库，仅用 model card 披露信息（release、imatrix、stock preset）无法重算任何 $Q4\_K\_M$；补充 per-tensor type map 与 `-ffp-contract=off` 后，x86 上 7/8 文件完全重算成功。
- **FP-contraction 实验**：在 b10448 构建 llama.cpp，arm64/clang 与 x86/gcc 各 default 与 `-ffp-contract=off` 两 variant，对 1.5B f16 量化四种类型：default 下 3/4 类型跨架构 byte-differ，启用 flag 后 4/4 类型 bit-identical；PPL 差异仅 +0.06%。
- **安全特征化**：以 dequantized weight space 的 $|\Delta w|/\sigma$ 度量扰动幅度，测量 benign $\mathcal{R}_{\mathrm{nat}}$ 分布与 uniformly placed nibble-flip 分布的分离度，定义 per-tensor tail fraction（$\tau{=}0.1$ 阈值以上比例）作为检测 statistic；同时评估 uniform scale perturbation、adaptive placement 与 targeted embedding edit 三种攻击变体。
- **Sampling bound**：Proposition 1 证明 partial spot-checking 对 $m$-unit 稀疏篡改需 $\Theta(N/m)$ 次打开，sparsity 越高 cost 越大，支持 full re-derivation 优先于 spot-check。

## 实验与结果
- **生态普查**：Hugging Face top-200 GGUF 仓库（2026年8月下载量），有效样本198个（61个发布者）。IVR(recorded-inputs)=0/198；source-reconstructible rate=148/198（74.7%，允许使用 base-model 公开权重）或111/198（56.1%，要求仓库自身提供fp16 GGUF）。
- **表II分解**：198个仓库全部缺失 env/version记录；其中148仅缺该记录，另有50同时缺少fp16或imatrix。
- **Trial re-derivation（表III）**：card alone → 0/8 $Q4\_K\_M$ 重算成功；+per-tensor recipe → x86 7/8 bit-exact，arm64 5/8（residual为all-zero block round差异）；0.8B的23种量化类型从0/23升至23/23。
- **Imatrix重算**：用同一corpus重算imatrix，0/186 entries bit-identical（median relative diff $5.5\times10^{-5}$），且跨arm64/x86亦不同；必须hash-bound而非重算。
- **FP-contraction（表IV/图2）**：default下arm64 vs x86在Q4_K_M、Q5_K_M、IQ4_XS上byte-differ，Q8_0（无scale search）一致；`-ffp-contract=off`使4/4类型bit-identical；divergent byte fraction随模型尺寸增大（1.5B→7B→14B）。
- **质量评估**：Qwen2.5-1.5B-Instruct，WikiText-2，Q4_K_M；default PPL=10.744，`-ffp-contract=off` PPL=10.750，$\Delta\text{PPL}{=}+0.006$（paired SE=0.005，不显著）。
- **检测下界（图3）**：Q4 K在arm64 contraction pair上per-tensor floor=0.02%权重；扩展至5-pair calibration后升至0.1%；自然tail noise中位数0.018%。Uniform scale perturbation在$\epsilon{=}0.02$时PPL发散且被检出，$\epsilon{=}0.005$隐蔽但仅+0.29% PPL。Adaptive placement可隐藏0.43%权重（≈4.7M）的sub-τ编辑。Targeted embedding edit（Q6_K行，95.5%元素一步步长<$\tau$）使greedy next-token翻转且所有统计值<Evasion>。
- **Pilot重算**：fp16转换跳可bit-exact重算（从provider checkpoint出发），验证失败仅位于量化步骤。

## 相关工作脉络
1. **Reproducible Builds（Debian 96%+）**：本文将其范式迁移至ML量化供应链，指出metadata normalization无法触及量化中的浮点非确定性，环境必须被record或render irrelevant。
2. **Sigstore Model Signing / NVIDIA NGC**：仅对 $H(W)$ 签名，不绑定变换输入；本文协议扩展为对 $(H(W),\theta,C,E,M)$ 签名，使 $Q$ 与 $W$ 的faithfulness可验证。
3. **ggufpacker & llama.cpp PR #25353**：已有工程实现cross-machine bit-exact derivation；本文测量这些工具所预设的完整规范在真实生态中的采纳程度（IVR=0/198），并精确定位卡未记录的缺失坐标。
4. **IETF AI Model Lifecycle Attestation / CycloneDX ML-BOM**：记录source hash、tool hash等，但不要求绑定 $C$ 和 $E$；本文指出其不足以支撑独立重算，并建议profile化扩展。
5. **Mind the Gap / Qu-ANTI-zation / Quantization Backdoors**：adversarial-fp16攻击属于 $Q=\mathcal{T}(\bar{X};E)$ 但 $W$ 被恶意准备的场景，本文明确将其划归re-derivation threat model之外（A1），本文聚焦cheating quantizer场景。
6. **Rounding-guided Backdoor Injection / Bit Flip Attacks**：本文针对此类攻击评估了statistic detector的边界，证明targeted edit可完全绕过统计，从而论证依赖exact re-derivation（$B{=}0$）的必要性。

## 局限性与未来方向
- 模型范围有限：仅覆盖Qwen 0.5B–14B与Granite 3B；量化类型限于Q4_K/Q5_K/Q6_K/IQ4_XS/Q8_0。
- 仅评估llama.cpp CPU量化管道；GPU imatrix生成（$E_C$）与GPU原生管道（GPTQ/AWQ/bitsandbytes）未涉及，需依赖attestation路径。
- 未完整评估adversarial-fp16攻击（仅对Mind the Gap代码作preliminary run，未得到working attack）。
- $X^*$ 的完整刻画仍是open problem（如GPU架构、其他compiler variant引入的额外坐标）。
- 未来方向：推动GGUF格式新增环境字段、合并PR #25353作为默认构建、在IETF/ML-BOM中强制绑定 $C$ 与 $E$。

## 研究启发与可借鉴点
1. **IVR指标可直接迁移**：对于LoRA adapter、RAG pipeline组件、multimodal encoder等"有变换但缺乏可重算规范"的artifact，均可定义类似IVR衡量生态可验证性。
2. **$B{=}0$ 前导条件思维**：任何供应链验证协议应以"输出唯一性"为前置；对无法达 $B{=}0$ 的管道，应明确声明其检测下界与盲区（如本文0.2%重量级盲区）。
3. **四输入检查清单范式**：$(W,\theta,C,E,M)$ 的分解方式清晰、可自动化扫描，可复用为其他ML artifact（如DPO checkpoint、vision backbone）的审计框架。
4. **安全特征化而非防御**：本文以distribution separation给出统计边界而非提出新检测器，这种"characterization-first"立场更诚实且便于下游协议设计，值得借鉴。
5. **与团队方向结合机会**：可扩展至multimodal LLM（ViT encoder量化）、跨框架转换（PyTorch→GGUF的checkpoint conversion hop）的IVR测量；也可将`-ffp-contract=off`思路迁移到ONNX/TensorRT量化流程的确定性审计。

## 关键术语表
- **IVR (Independent Verifiability Rate)**：独立可验证率，指第三方在给定披露输入下能bit-exact重算的artifact占被评估artifact的比例，本文核心测量指标。
- **$\bar{X}$ (non-environment inputs)**：非环境输入集合，包含全精度权重 $W$、量化配方 $\theta$、imatrix校准输入 $C$、非权重组件 $M$。
- **$E$ (execution environment)**：执行环境，涵盖量化器版本、编译器、目标架构、FP-contraction设置等决定重算结果的坐标。
- **$B$ (ambiguity budget)**：歧义预算，衡量相同 $\bar{X}$ 在不同 $E$ 下输出是否唯一的0/1指示器；$B{=}0$ 当且仅当 $|\mathcal{A}(\bar{X})|=1$，是可信hash比较的前置条件。
- **$\mathcal{A}(\bar{X})$ (admissible output set)**：在环境空间 $\mathcal{E}$ 中所有可能产出的集合，$\mathcal{A}(\bar{X}){=}\{\mathcal{T}(\bar{X};E):E{\in}\mathcal{E}\}$。
- **$\mathcal{R}_{\mathrm{nat}}$ (natural-variance distribution)**：由benign环境变化引起的去量化权重扰动分布，用于与tampering分布作分离对比。
- **Per-tensor recipe**：量化文件中各tensor实际使用的量化类型（如SSM保持F32、attention提至Q5_K等），卡上仅标注 $Q4\_K\_M$ 不足以重算。
- **$B{=}0$ regime**：确定性量化状态，此时任何 $Q\neq\mathcal{T}(\bar{X};E)$ 均会触发哈希不匹配，是本文验证协议的基础前提。

## 可复现要素
- **数据集**：Hugging Face公开GGUF仓库（top-200下载量，冻结列表198个有效仓库），全部为公开可访问。
- **代码/工具**：开源于 https://github.com/Rlonglong/gguf-verifiability-artifact，含 `census.py`、GGUF header-probe、byte-region diff、tensordata hash、detector 等；Zenodo deposit 将随正式发表同步。
- **关键超参/配置**：llama.cpp releases b3772、b9222、b10448、b10603；`-ffp-contract=off` 编译标志；tail threshold $\tau{=}0.1$；Perplexity 评估使用 WikiText-2（584 chunks）。
- **量化类型**：Q4_K_M、Q5_K_M、Q6_K、IQ4_XS、Q8_0（以及0.8B仓库的全部23种发布类型）。
- **评估机器**：Apple M4 (arm64/clang)、x86-64 Linux (gcc)，CPU-only llama.cpp build。

<!--META
{"keywords": ["GGUF", "model quantization", "reproducible builds", "supply chain security", "IVR", "floating-point non-determinism"], "field": "LLM供应链安全与可
