# Trident: Unifying Guarded Dispatch and Host Execution for PyTorch Triton Workloads

Jinjie Liu<sup>∗</sup> The Chinese University of Hong Kong Hong Kong, China jj<sup>l</sup>iu26@cse.cu<sup>hk</sup>.edu.<sup>hk</sup>

Wenjia Sun   
Beijing Academy of Artificial   
Intelligence   
Beijing, China   
sunwenjia04@163.com   
Xiaoyan Liu<sup>†</sup>   
Beijing Academy of Artificial   
Intelligence   
Beijing, China   
<sub>xy</sub>li<sub>u</sub>01<sub>@</sub>b<sub>aa</sub>i<sub>.ac.cn</sub>   
Ruilin Yang   
Beijing Academy of Artificial   
Intelligence   
Beijing, China   
<sub>r</sub>l<sub>yang@</sub>b<sub>aa</sub>i<sub>.ac.cn</sub>   
Yonghua Lin   
Beijing Academy of Artificial   
Intelligence   
Beijing, China   
<sub>y</sub>hli<sub>n@</sub>b<sub>aa</sub>i<sub>.ac.cn</sub>   
Shuhan Zhang   
Beijing Academy of Artificial   
Intelligence   
Beijing, China   
lisco<sub>py</sub>e@163.com   
Chunlei Men   
Beijing Academy of Artificial   
Intelligence   
Beijing, China   
<sub>c</sub>l<sub>men@</sub>b<sub>aa</sub>i<sub>.ac.cn</sub>

Shaohua Li The Chinese University of Hong Kong Hong Kong, China <sub>s</sub>h<sub>ao</sub>h<sub>ua</sub>li<sub>@cu</sub>hk<sub>.e</sub>d<sub>u.</sub>hk

## Abstract

User-written Triton kernels enable high-performance GPU computation within PyTorch, but their end-to-end latency can remain dominated by host-side orchestration, especially when device execution is short. Although torch.compile can generate native host wrappers for captured graphs, each invocation still passes through runtime-managed specialization lookup, guard evaluation, and preparation before reaching the wrapper.

We present Trident, a compiler backend that removes this recurring overhead from the specialization cache-hit path. Trident introduces the Specialization Cache Module (SCM), which compiles guarded specialization selection, argument and execution-environment preparation, and host execution for multiple specializations into a single executable module. An invocation enters the SCM once, remains in compiled code when a specialization matches, and returns to Python only when a new specialization must be compiled. Built on Torch-MLIR, Trident lowers guards and host-side orchestration to native code while retaining calls to optimized runtime implementations of supported ATen operators. Our evaluation on two LLMs shows that Trident achieves up to a 1.47× speedup in model-level end-to-end latency over eager execution and up to 1.68× over torch.compile.

<sup>∗</sup>This work was completed during an internship at the Beijing Academy of Artificial Intelligence.   
<sup>†</sup>Corresponding author. ACM Reference Format:   
Jinjie Liu, Xiaoyan Liu, Shuhan Zhang, Wenjia Sun, Ruilin Yang, Chunlei Men, Yonghua Lin, and Shaohua Li. 2026. Trident: Unifying Guarded Dispatch and Host Execution for PyTorch Triton Workloads. In . ACM, New York, NY, USA, 13 pages. htps://doi.org/ 10.1145/nnnnnnn.nnnnnnn

## 1 Introduction

With the widespread adoption of large language models [2, 17, 27, 30], machine learning frameworks [1, 6, 14, 20, 23] have become an indispensable part of modern software infrastructure. Among these frameworks, PyTorch [20] has emerged as a leading platform due to its ease of use, flexible programming model, and strong community support. Building on this eager programming model, PyTorch 2 further introduces TorchDynamo and TorchInductor to enable graph compilation without sacrificing Python flexibility [4]. As a representative eager-mode system, PyTorch has become foundational infrastructure for both model research [29] and development and production serving stacks [16, 32].

Within the PyTorch ecosystem, Triton has emerged as a widely used backend for customized GPU kernels, enabling users to express high-performance kernels that can be called directly from PyTorch. However, optimized device code alone does not guarantee low end-to-end latency. Each kernel invocation still incurs host-side orchestration overhead, such as bufer allocation, kernel-variant selection, and kernel launch. This mismatch is particularly significant for short kernels, which are common and repeatedly executed in large-model workloads [16, 32]. As GPU throughput improves, such kernels become shorter, and host-side kernel launch and orchestration overhead can become comparable to or even exceed device execution [11]. Consequently, reducing hostside overhead is essential to fully realize the performance of customized Triton kernels [4].

One approach to reducing these host-side costs is to manually adopt CUDA Graphs, which capture and replay a fixed sequence of GPU operations. Their use, however, requires the captured execution to satisfy constraints on control flow, memory addresses, and kernel behavior. Meeting these constraints often entails substantial modifications to both hostside calling logic and the kernels themselves [16]. Even when capture is feasible, parameter-copying and replay-related overheads can outweigh its benefits and degrade performance [11]. These limitations have motivated framework support that applies host-side optimizations transparently while preserving the eager programming interface.

PyTorch provides this support using Dynamo for graph capture and Inductor as its default compiler backend [4]. As shown in Figure 1(a), Dynamo executes and traces one path through the user’s host-side Python code, capturing the tensor operations and Triton kernel calls along that path in an FX graph [21]. During graph capture, Dynamo records guards that describe the input properties and runtime assumptions under which the captured path remains valid. The resulting FX graph and its guards form a specialization, which Dynamo caches for subsequent invocations. Once the operations of a specialization are known, Inductor generates a host wrapper that performs tasks such as memory management, launch-argument preparation, and kernel launch. By default, Inductor emits this wrapper as Python code. Although graph capture removes repeated interpretation of the original user code, executing the generated wrapper still incurs Python dispatch and object-handling overhead. To further reduce this overhead, Inductor can instead generate a C++ wrapper, allowing the host operations within the captured graph to execute as compiled native code.

However, the C++ wrapper covers only the host operations derived from one FX graph, while each invocation still incurs host-side work outside the wrapper. As illustrated by Figure 1(b), every user function call enters the Dynamo runtime scheduler, which performs a lookup in the specialization cache and evaluates guards to select matching specialization code. The selected specialization code performs argument preparation and environment preparation before calling the Inductor host wrapper. This preparation is necessary to preserve the execution semantics of the Python function when invoking the selected FX graph, but introduces host-side overhead through additional runtime calls and glue code. This host-side cost can slow down the operator’s end-to-end execution, especially for short Triton kernels.

These costs result from PyTorch’s runtime-managed execution model: the Dynamo runtime scheduler performs specialization-cache lookup and dispatch, while Inductor independently compiles the host operations ofeach selected FX graph. Under this model, every invocation inevitably incurs the overhead of runtime-side specialization management, and even a cache hit must still pass through the runtime calls and glue code that connect Dynamo’s selected specialization to Inductor’s host wrapper. In this paper, we investigate whether this recurring path can be removed by compiling guard-based specialization selection and preparation together with host execution. This requires a compiled program that manages multiple guarded specializations while retaining the ability to return to Python when no specialization matches.

In this paper, we propose Trident, a compiler backend that reduces runtime overhead for programs containing userwritten Triton kernels. Built on Torch-MLIR, Trident generates a Specialization Cache Module (SCM), an executable module in which specialization management and host execu tion are compiled together for multiple guarded specializations (Figure 1(c)). In contrast to PyTorch’s runtime-managed execution model, Trident routes every invocation directly into the SCM. The SCM evaluates the guards of its specializations, executes the corresponding host logic in the same module when a guard succeeds, and returns to Python only when no specialization matches. This organization removes the runtime calls and glue code that connect specialization management to host execution on the cache-hit path. To preserve the performance of existing PyTorch operators, ATen operations remain calls to their registered runtime implementations.

Specifically, this paper makes the following contributions:

• We introduce the Specialization Cache Module (SCM), which compiles guard-based specialization selection, preparation of arguments and execution environment, and host execution for multiple specializations into a single executable module. The SCM eliminates the intervening runtime calls and glue code while retaining Python-side compilation when no specialization matches.

• We design and implement Trident, an end-to-end compiler backend built on Torch-MLIR. Trident analyzes Dynamo-captured FX graphs and guards, lowers the guards and host-side orchestration to C++ code, and exposes the resulting SCM through a TVM FFI entry. During lowering, Trident retains supported ATen operators as runtime calls to reuse their optimized implementations.

• We evaluate Trident on Triton operators and two LLMs, DeepSeek-V2-Lite and Qwen3-8B, across standard datasets. Trident achieves up to 1.73× end-toend speedup over the eager wrapper on kernels and outperforms default torch.compile on most operators. At the model level, it achieves up to 1.47× and 1.68× end-to-end latency speedup over the eager wrapper and torch.compile, respectively, while also improving TPOT and ITL.

![](images/b89dba4c9fb103b7921b1186d2560ce1f738d009527c976053b4c338ee414944.jpg)  
Figure 1. (a) An rms\_norm example (omitted for clarity) with two Triton JIT kernels, oneshot\_kernel and loop\_kernel. Diferent input shapes take diferent host-side paths; Dynamo captures the executed path as a guarded FX-graph specialization. (b) Dynamo runtime-managed execution model, where blue edges mark cross-layer and glue overhead between warm-path steps. (c) Trident SCM-based execution model.

## 2 Background and Motivation

## 2.1 Triton Kernels

Triton is a Python-embedded DSL for writing GPU kernels [26]. Users write device computation in @triton.jit functions and call them from user host-side code. As illustrated in Figure 1(a), this user host-side code performs a series of operations such as bufer allocation, launch-argument preparation, kernel-variant selection, and kernel launch. Different inputs can lead to diferent host-side behavior. For example, the code may compute launch parameters from the input shapes, such as the grid size � and the compile-time constant BLOCK\_SIZE, and may select among kernel variants before issuing the kernel launch.

To avoid repeated compilation overhead, the Triton runtime caches compiled GPU binaries and reuses them across kernel calls [26]. For each call, it computes a cache key from properties of the invocation, such as argument types, tl.constexpr values, and launch options, and looks up this key in a per-device kernel cache. On a cache miss, Triton compiles the kernel to a GPU binary (CUBIN) and stores it under that key. On a hit, it reuses the cached CUBIN and issues the kernel launch. When @triton.autotune is used, the runtime additionally benchmarks a set of candidate configurations, such as the number of warps and the number of pipeline stages. It retains the fastest configuration in an in-memory map keyed by user-defined tuning attributes, and can persist these results to Triton’s on-disk cache so that subsequent processes avoid re-autotuning. Later stages of torch.compile can reuse this cached CUBIN rather than recompiling the Triton kernel from source.

## 2.2 Torch Compile

The torch.compile compiles user host-side code while preserving the eager programming interface [4]. Dynamo captures graphs from Python execution. Inductor is the default compiler backend and generates a host wrapper for each captured FX graph.

2.2.1 Dynamo Graph Capture and Guards. As shown in Figure 1(a), Dynamo executes and traces one path through the user’s host-side Python code under a concrete runtime input [4]. In the rms\_norm example, diferent input shapes take diferent host-side branches. Dynamo extracts the tensor operations and Triton kernel calls along that executed path into an FX graph [21]. During this capture, Dynamo records guards that describe the assumptions of the traced path. These guards include tensor-match checks on properties such as dtype, device, and shape-related metadata, controlflow and shape assumptions such as weight.numel() <=

4096. The resulting FX graph and its guards form a specialization, which Dynamo caches for subsequent invocations.

On a later call, the Dynamo runtime scheduler performs a lookup in the specialization cache. A cache miss causes Dynamo to capture and compile a new specialization. PyTorch provides options that let users reduce this guard-related host overhead. A guard filter installed at compile time can drop selected guard kinds so that fewer checks are stored with each specialization, for example retaining tensor-match guards while discarding others. After warmup, users can further skip most remaining guard evaluation on later calls. Both options are unsafe, and a mismatch can silently produce incorrect results. Neither filtering nor skipping removes specializationcache lookup, where the Dynamo runtime scheduler still walks the specialization cache to select the code to execute.

2.2.2 Inductor Host Wrapper. Inductor compiles each captured FX graph into a host wrapper [4]. At runtime, the selected specialization code performs argument preparation and environment preparation before calling the Inductor host wrapper. Argument preparation flattens the user-visible function arguments into the inputs expected by the selected FX graph and converts tensor arguments for the compiled calling interface. Environment preparation retrieves the current CUDA stream from the active device context. Host wrapper execution then performs memory management, launchargument preparation, and kernel launch on that stream. For user-written Triton kernels, Inductor obtains the corresponding GPU binaries from Triton’s kernel cache rather than recompiling those kernels from source.

By default, Inductor emits the wrapper as Python code. To accelerate host-wrapper execution, Inductor can emit a C++ wrapper that avoids Python interpretation overhead. It further supports capturing eligible GPU work within the wrapper as a CUDA Graph after warmup and replaying that graph on subsequent calls, which can reduce repeated kernellaunch cost within one specialization. CUDA Graph capture in Inductor remains subject to constraints on control flow and memory layout, and capture or replay may fail when these constraints are not satisfied. However, these mechanisms optimize only the host operations derived from a single FX graph. Neither the C++ wrapper nor CUDA Graph replay eliminates Dynamo’s specialization-cache lookup or the argument and environment preparation that precedes wrapper entry.

## 2.3 Motivation: Residual Host Overhead

Figure 2 shows the warm-path host and end-to-end performance of representative torch.compile configurations on an rms\_norm example, normalized to the original Triton baseline with eager wrapper. We run the experiment on an NVIDIA H800 GPU with input x of shape (1024, 128) and weight of shape (128,), warm up for 20 iterations, measure

![](images/f1fa8da3d2b70667568772c5f255e1995f70e2242cf5be01dab4dfe91d428cce.jpg)  
Figure 2. Warm-path host and end-to-end performance of torch.compile configurations on rms\_norm, normalized to the original eager wrapper.

10 repeats over 5 rounds for each configuration, and report the median.

Across these settings, every torch.compile configuration is slower than the eager baseline. The default configuration reaches only 0.39× of the eager baseline on the host and 0.43× end to end. Skipping guards improves this to 0.50× and 0.53× of the eager baseline, respectively, but remains well below eager. The C++ wrapper is even slower than default torch.compile, reaching only 0.36× and 0.38× of the eager baseline. Enabling CUDA Graphs is even slower, because replay-time parameter copies into static placeholders can outweigh the launch savings on short kernels [11]. Note that we measure a single input shape after suficient warmup, so this gap reflects residual host-side overhead on the warm path rather than cold compilation or shape switching.

To further support this observation, we export a Perfetto [25] trace of the warm-path C++ wrapper configuration, as illustrated in Figure 3(a). Due to tracing overhead, these profiles are not accurate for absolute performance, but they show that a call reaches C++ wrapper execution only after many intervening runtime frames, which causes the residual host-side overhead and performance degradation.

To this end, we propose Trident, a compiler backend that generates a Specialization Cache Module (SCM) to eliminate the intervening runtime calls and glue code on the warm path. Figure 2 shows that Trident is the only configuration that improves over the eager baseline, with 1.18× host and 1.11× end-to-end speedup, and that it is 2.99× and 2.60× faster than default torch.compile on host and end-to-end latency, respectively. Figure 3(b) further shows that Trident reaches kernel launch with far fewer intervening runtime frames than the Dynamo path.

![](images/97c37bddfeff00a7c442bed3cd34762405ddb9d953bd9cc726ccaf68be438d6a.jpg)  
Figure 3. Warm-path Perfetto traces of rms\_norm: (a) Dynamo with the Inductor C++ wrapper; (b) Trident.

## 3 Methodology

## 3.1 Overview

Trident requires no changes to the body of a supported Python function or its user-written Triton kernels: users simply apply @trident.jit to the Python entry function. Existing PyTorch operations and Triton kernel launches remain unchanged, and the kernels retain their @triton.jit decorators. From this function, TorchDynamo captures an ATen-level FX graph together with the guards that define when the captured computation is valid. Trident turns this guarded input into a compiled multi-version program that handles the recurring cache-hit path. The output is a callable that preserves the source-level signature while routing each invocation to a compatible compiled specialization. Unlike a conventional per-graph host wrapper, this callable moves specialization selection, graph-input preparation, and host execution into the same compiled boundary.

The central artifact is the Specialization Cache Module (SCM), illustrated in Figure 1(c). An SCM contains an ordered set of versions, where each version pairs a guard predicate with the computation captured under that predicate. It also contains a dispatcher that evaluates these predicates and transfers control to the first matching computation. The Python boundary is deliberately narrow: it prepares source arguments for the compiled calling convention and initiates compilation only when the SCM reports that no version matches. Consequently, a cache hit crosses from Python into the SCM once and remains in compiled host code through version selection and execution. Calls from the SCM to registered ATen implementations are internal runtime calls and do not return control to the Python scheduler.

The design is governed by three requirements:

1. Safe reuse. Every assumption that determines the captured Python path or a selected Triton CUBIN must hold before the operation it protects executes.

2. Stable interface. All versions must expose the sourcelevel calling convention despite capture-specific input flattening and internal signatures.

3. Runtime reuse. Compilation should handle ATen and Triton calls with little or no change to supported Python programs while reusing their existing implementations.

The SCM meets these requirements by pairing each version’s guards with its computation, wrapping all versions behind one source-level interface, and reusing PyTorch’s ATen implementations and Triton’s selected CUBINs.

Compilation proceeds through a series of representation transitions rather than by rewriting the user’s kernels. Torch-Dynamo first captures an ATen-level FX graph and the assumptions of the executed Python path. For each capture, Trident packages the FX graph, its validity checks, and any selected Triton CUBINs into a guarded high-level MLIR submodule, which it adds to the stored versions. It then rebuilds the executable SCM by merging all stored submodules, lowering the combined module to the LLVM dialect, adding an ordered dispatcher, and JIT-compiling the complete module in an in-process execution engine. Section 3.2 explains native guarded dispatch, Section 3.3 describes how a specialization is built, Section 3.4 explains how the SCM is lowered and installed, and Section 3.5 describes runtime execution and specialization growth.

## 3.2 Native Guarded Dispatch

In torch.compile, guard management and graph-module execution remain separate runtime stages. As Figure 1(b) shows, the Dynamo runtime scheduler owns specializationcache lookup and guard evaluation, whereas a backendgenerated host wrapper owns execution of the captured FX GraphModule. Even on a cache hit, glue code must connect these stages by preparing graph inputs and the execution environment before transferring control to the wrapper. Because the stages span several language and runtime boundaries, one invocation can require multiple cross-boundary calls. A C++ wrapper accelerates execution inside the selected graph but does not remove the preceding runtime work or boundary crossings. For short GPU kernels, the fixed overhead of this glue code can lie on the critical path.

```perl
Algorithm 1: Ordered dispatch in an SCM.
Input: Input $x ;$ ordered versions $\{ ( G _ { i } , P _ { i } ) \} _ { i = 1 } ^ { n }$
Output: Program result or Miss
1 for $i \gets 1$ to � do
2 if $G _ { i } ( x )$ fails then
3 continue;
4 $o \gets P _ { i } ( x ) ;$
5 if $o = E R R O R ( e )$ then
6 raise $e ;$
7 return $o ;$
8 return Miss;
```

As Figure 1(c) shows, Trident unifies these stages in the SCM. It places each version’s guards and captured graph computation in one module. During lowering, the private graph computation is inlined into its guarded wrapper, so guard control flow and graph host operations become one native program. A guarded specialization is therefore the unit of both reuse and correctness: once an invocation en ters the SCM through one TVM FFI entry, native control flow evaluates guards, advances across rejected versions, and enters the first matching graph computation without returning to the Python scheduler between these steps. For an input �, let $G _ { i } ( x )$ denote the assumptions of version � and let $P _ { i } ( x )$ denote its captured computation. Algorithm 1 formalizes ordered dispatch. The first version whose assump tions hold executes; if every $G _ { i }$ fails, dispatch returns a specialization miss (lines 2-3, 8). An execution error from the selected $P _ { i }$ terminates dispatch immediately and propagates through TVM FFI’s error channel rather than being treated as a miss (lines 5-6). At the Python callable boundary, TVM FFI raises the corresponding exception to the application, so Trident neither tries another version nor triggers recompilation. This organization removes repeated boundary crossings between specialization management and graph execution on the cache-hit path. Necessary internal runtime calls, such as ATen dispatch through Trident FFI, remain within the matched SCM and do not re-enter Python-side specialization management.

To implement $G _ { i } ( x )$ , Trident translates the supported TorchDynamo guards associated with each captured version into checks over source-level runtime inputs. It combines predicate-derived checks with checks for other captured assumptions, removes redundancies, and orders the remaining checks according to their data dependencies. The resulting short-circuiting control flow enters $P _ { i } ( x )$ only after every required check succeeds; any failed check reports a specialization miss. The lowering preserves symbolic relationships that can remain valid across multiple concrete inputs. Guard eval uation and graph-input preparation resolve values through the same logical parameter mapping, ensuring that each predicate protects the corresponding captured computation. Keeping guards explicit in compiler IR allows them to be optimized and compiled together with the computation they protect. Figure 4 summarizes this translation from captured assumptions to compiled guard control flow.

## 3.3 Building a Specialization

Native guarded dispatch requires every captured computation to remain paired with the assumptions that make it safe to reuse and to expose an interface shared by all versions. Trident therefore packages each guarded specialization as a self-contained MLIR submodule, keeping its validity checks and captured work together while allowing it to be merged with the stored versions when rebuilding the SCM.

Construction begins with graph capture and normalization. TorchDynamo captures the executed path as an ATenlevel FX graph with its guards. Trident preserves Dynamo’s capture assumptions and symbolic values while normalizing the graph into the forms expected by the importer. Torch-MLIR [18] then imports the normalized graph as a private function in the new submodule. Giving every imported computation and wrapper a version-specific symbol keeps captures independent when their submodules are later merged.

The exported graph and the source function intentionally use diferent input representations. The Python boundary sees structured source arguments, whereas the imported graph consumes a flat sequence of graph leaves. Flattening is convenient for graph transformations, but exposing it at the external boundary would require Python to choose a specialization before it knew which flat signature to construct. Trident instead arranges each version wrapper’s parameters in the original function’s parameter order and reconstructs the graph inputs only after that version’s guards succeed.

This reconstruction is driven by the exported input tree and graph signature. For each source parameter, Trident records a typed recipe describing its source-level structure. At wrapper entry, the recipe binds the actual argument to a logical input tree. Guard expressions address values through paths in this tree, while the success block recursively flattens the same tree in exporter order and selects the leaves consumed by the imported computation. Structured values are unpacked only after the relevant guards establish that doing so is valid. This shared reconstruction mechanism aligns a guard, such as a check on the second tensor in a tuple, with the corresponding graph operand.

Source-visible constants that afect specialization but are absent from the flattened graph remain wrapper parameters and can participate in guards. Values embedded during export remain part of the captured computation instead of being recreated at runtime. On return, result normalization restores the Python-facing representation. Thus, the external signature remains stable even when versions difer internally in their flattened operands or result structure.

![](images/499856a250fa0a1db00546b7173e6928c96d4a711df8a073f85f51408fa40ed2.jpg)  
Figure 4. Lowering the assumptions of a captured Python path into guard IR. TorchDynamo supplies the captured assumptions, which Trident converts into dependency-ordered semantic checks. The resulting short-circuiting control-flow graph enters the captured computation only after all checks succeed and otherwise returns an explicit specialization miss.

Each Triton call additionally requires a launch description and the CUBIN selected for the current specialization. During capture, executing the exported graph causes Triton to compile or retrieve the kernel variant determined by the current inputs and compile-time launch configuration. Trident obtains the selected CUBIN from Triton’s device-local cache and embeds it in the submodule under a version-specific symbol. The high-level MLIR submodule pairs this binary with a TorchExt launch operation that refers to its symbol and records the launch description and specialization requirements. This representation reuses Triton’s compiled device code without recompiling the Triton program from source.

During whole-SCM lowering, Trident converts the launch description into GPU IR while retaining the embedded CU-BIN. It materializes the kernel’s specialization checks before the launch and converts the runtime arguments to the kernel’s native calling convention. The resulting GPU launch operation carries the dynamic launch dimensions and refers to the same binary. Lowering and translation to LLVM then produce compiled host code that loads the embedded GPU module, resolves the kernel entry, and launches it on the current stream.

Crucially, this construction removes the Triton Python runtime from the recurring cache-hit path. Later hits validate and invoke the embedded CUBIN from compiled host code, bypassing Triton’s Python specialization and launch path. The cost of Triton runtime specialization is therefore incurred once when each specialization is built, rather than on every invocation.

## 3.4 SCM Lowering and Runtime Integration

Building an individual specialization yields a reusable highlevel submodule, but an executable SCM must combine and lower all stored versions into one program with an ordered dispatcher. Trident therefore constructs the SCM at two levels: the frontend builds one high-level submodule for each specialization, whereas an SCM rebuild lowers all stored versions together. Figure 5 distinguishes these two levels explicitly. Before a rebuild, every stored specialization remains a high-level MLIR submodule that retains its source-level semantics. Trident combines these submodules and applies one whole-SCM lowering pipeline, which inlines each graph and converts the combined host program through TVM FFI IR to the LLVM dialect. The same pipeline lowers Triton launch descriptions and preserves every specialization as a distinct lowered entry.

ATen operators use a compiled calling boundary that reuses PyTorch’s registered implementations. We provide Trident FFI, a schema-driven binding layer that covers the registered ATen operator surface of the installed PyTorch build and generates packed wrappers for it. We implement these wrappers with a generic C++ template instead of handwritten per-operator bindings. At build time, a Python generator enumerates the ATen schemas registered with PyTorch’s dis patcher, maps their argument and result types to template parameters, and emits one corresponding wrapper instantiation for each operator. Shared template specializations translate between TVM FFI values and PyTorch values, while the generic wrapper invokes the registered implementation through PyTorch’s boxed dispatcher and returns its result to the SCM. When the generated library is loaded, Trident FFI registers these wrappers under stable trident.aten.\* names in TVM FFI’s global function table. For a general ATen operator, lowering performs a name-based lookup in this table and emits a packed call to the resolved wrapper. The compiler can therefore lower an ATen operation without embedding PyTorch’s dispatcher data structures into its own IR.

![](images/6ab9189ab0590137a52fab6a8b5faf934448598c3350881e9c485494041f05d1.jpg)  
Figure 5. Whole-SCM compilation centered on the main host-lowering path. Each capture contributes a high-level MLIR submodule to the stored versions. Trident merges these submodules and lowers the combined module along the main host path from Torch IR through TVM FFI IR to the LLVM dialect. It then adds the LLVM dispatcher and JIT-compiles the executable SCM behind one TVM FFI callable.

Trident divides ATen-related computation between two paths:

• Native lowering. Selected inexpensive host operations, such as guard arithmetic and tensor-metadata queries, become native instructions when their semantics and ABI representation are known. These operations commonly occur in guards and launch-grid calculations, where a runtime operator lookup would be disproportionately expensive.

• Runtime dispatch. General tensor operators call their generated Trident FFI entries and reuse the implementations registered with PyTorch. An FFI and dispatcher boundary remains for each such operator, but Python no longer schedules these operations or connects them to the selected specialization.

This split targets orchestration overhead while retaining the coverage and device-specific behavior of the existing ATen runtime.

Triton launches follow a separate lowering path because their computation is already present as a CUBIN. For every launch, the host code first evaluates the CUBIN-specific preconditions described in Section 3.3. It then prepares the runtime arguments and launch configuration before submitting the embedded CUBIN to the current CUDA stream. Symbolic shape and grid arithmetic remains in compiler IR throughout this path, allowing constant fragments to be folded while input-dependent fragments remain native runtime calculations. The selected Triton kernel therefore keeps its original CUBIN while its repeated launch preparation becomes part of the compiled SCM.

The TVM FFI representation provides the common ABI across SCM and runtime boundaries. At compile time, lowering maps Torch-level tensor and scalar types, together with the operations that consume them, to typed semantic TVM FFI IR; a subsequent conversion lowers this representation to the packed TVM FFI ABI. At execution time, TVM FFI automatically converts and packs Python arguments at the callable boundary, representing PyTorch tensors through DL-Pack. Runtime function lookup and invocation are checked explicitly, so a missing registration becomes a real execution error. Normal program values and the specialization-miss object share a general result slot, but their runtime type tags remain distinct. This distinction lets the LLVM dispatcher identify a miss without reserving an error code that could be confused with an operator failure.

Finally, Trident lowers the combined module’s remaining control flow and ABI operations to the LLVM dialect. After this lowering pipeline completes, it adds the ordered LLVM dispatcher under one stable symbol derived from the original function name. The execution engine optimizes and JIT-compiles the complete LLVM module, then exposes its entry symbol as a TVM FFI callable. The callable retains the engine so that all executable state remains live for the lifetime of the callable.

## 3.5 Runtime Execution and Specialization Growth

The installed SCM separates the frequent compiled hit path from the infrequent Python compilation path while distinguishing specialization misses from execution errors. On each call, the outer wrapper binds the source arguments and normalizes them into their FFI representations. It then makes one entry call from Python into the SCM dispatcher. This boundary performs representation normalization but does not inspect Dynamo’s specialization cache or choose a captured graph.

On a cache hit, the dispatcher tries versions in creation order. For each version, its wrapper first evaluates the shortcircuiting Dynamo-derived guards. A failed guard returns the distinguished miss value, causing the dispatcher to advance to the next version without unwinding through Python. After all guards succeed, the wrapper reconstructs the flattened graph operands and enters the inlined host computation; launch-specific preconditions are checked immediately before the corresponding Triton CUBIN is submitted. The result travels through the same FFI boundary and is normalized back into its source-level Python representation. The entire cache-hit path therefore remains on the compiled side of the single Python-to-SCM entry.

On a cache miss, every installed wrapper returns the distinguished miss value. The dispatcher forwards the final miss to the outer wrapper, which invokes Python only at this point to capture a new specialization and rebuild the SCM. An empty SCM follows the same path on its first in vocation. The invocation that triggered compilation returns the result produced during capture, while later compatible invocations use the newly installed compiled version. This policy incurs the cost of whole-module reconstruction when the specialization set grows in exchange for a simpler steady state path with lower overhead. Trident then rebuilds and JIT-compiles a replacement SCM from the complete stored version set under the same public entry point.

Real execution errors form a third path and never trigger specialization growth. If an internal runtime call fails, the generated FFI code propagates its error status to the dispatcher, which terminates rather than attempting another version. This behavior preserves the semantic diference between “the assumptions of this version do not hold” and “the selected program failed to execute.” Together, the three paths give the SCM a precise responsibility boundary: compiled code handles expected input variation among installed versions, whereas Python handles only previously unseen variation that requires a new capture.

## 4 Evaluation

## 4.1 Experiment Setup

Platform – We conduct all experiments on a server with two Intel Xeon Platinum 8468H processors (96 physical cores and 192 hardware threads) and NVIDIA H800 GPUs with 80 GB of HBM. Each benchmark uses a single GPU. The system runs NVIDIA driver 580.167.08 and PyTorch 2.11.0.

Kernel benchmarks – We use operators from FlagGems [24] a widely used open-source Triton operator library in the PyTorch ecosystem. We select 21 representative operators that span the main computational idioms in LLM and deep learning kernels: pointwise arithmetic and activations (e.g., add, silu, pow), comparisons and masking (e.g., lt, masked\_fill), normalization (rms\_norm), embedding lookup (embedding), scans (cumsum), sorting (sort), dense matrix multiplication (mm, addmm), concatenation (cat), and convolution (conv2d, conv\_transpose2d). For each operator, we select five representative shapes derived from common LLM tensor dimensions. We time 30 consecutive invocations per shape. We report the first as cold-start latency and the median of the remain ing invocations as warm latency. We report both end-to-end latency (including host and device execution until GPU completion) and host latency (only until the Python call returns).

Model benchmarks – We evaluate DeepSeek-V2-Lite and Qwen3-8B [30] on MMLU (general knowledge) [12], GSM8K (mathematical reasoning) [9], and HumanEval (code generation) [7]. All experiments use batch size one and generate at most 128 new tokens per request. We compare the eager Python wrapper, torch.compile, torch.compile with the C++ wrapper, and Trident while replacing the same operator whitelist—bmm, linear, and embedding—in each model. Because wrapper behavior on real model inputs is complex, we do not enable additional aggressive acceleration options that would compromise correctness, and CUDA Graphs fail without model-specific adaptation. For every mode and dataset, we first execute ten warmup requests and then measure 64 requests for DeepSeek-V2-Lite and 32 requests for Qwen3- 8B.

## 4.2 Kernel Performance

Figure 6 reports warm-path performance relative to the eager wrapper. For each operator and execution mode, we discard the first timed invocation, remove outliers from the remaining samples using the 1.5-IQR rule, and compute the mean latency; the figure reports the arithmetic mean of the five per-shape ratios. Relative to the eager wrapper, Trident is faster on most operators (13 of 21 in host latency and 11 of 21 end to end), with mean speedups of 1.10× and 1.07×. More importantly, on operators with a complete torch.compile baseline, Trident is faster than default torch.compile on 18 of 20 operators, by up to 3.07× in host latency and 2.61× end to end, and it likewise beats the other evaluated torch.compile configurations on most of the suite.

The largest gains occur when a short device kernel is surrounded by substantial host-side dispatch and launch preparation. For example, Trident reaches up to 1.74× host and 1.73× end-to-end speedup on silu, and 1.67× end-to-end on masked\_fill: by placing specialization dispatch, argument preparation, and host execution in one SCM, it removes repeated wrapper orchestration that still remains on the eager and torch.compile warm paths. Conversely, operators such as cat remain below 1× when memory movement or device work dominates, or when too much wrapper logic stays outside the SCM.

This pattern generalizes the rms\_norm observation in Section 2.3. Compiling an FX graph removes repeated interpretation of the original wrapper, but a warm call still enters the Dynamo scheduler for specialization-cache lookup and guard evaluation, performs argument and environment preparation, and crosses into the Inductor wrapper. A C++ wrapper or CUDA-Graph replay optimizes only work inside the selected graph and does not remove that preceding runtime path, so torch.compile is often slower than the already lean eager wrapper. By compiling guarded dispatch and graph execution within the same SCM, Trident removes this residual overhead and is therefore faster than torch.compile on most operators.

![](images/9b98aa04244d37b9e04913bd4bffad9bad785923dfbc5cc65331be7afa7baf4a.jpg)  
Figure 6. Warm host and end-to-end performance relative to the eager wrapper, averaged over five representative shapes per operator. Higher is better; values below 1× are slower than eager. Missing points correspond to configurations that failed to run.

Cold start is slower. The first timed invocation includes Triton compilation and, for Trident, SCM construction, so its cold host latency is typically higher than torch.compile— about 2× across many short kernels, e.g., 0.84 s versus 0.42 s on neg. This cost is paid during warmup. Once the special ization is cached, repeated warm invocations recover the overhead through the per-call savings above. Although the speedup of a single operator invocation is often modest, model inference repeatedly invokes the same kernels across layers and autoregressive decoding steps, allowing these per-invocation savings to accumulate at model scale.

## 4.3 Model Performance

Figure 7 reports model-level performance relative to the eager Python wrapper. Whether compiling an operator is beneficial is dificult to determine statically. We therefore use ten warmup requests to identify and exclude operator candidates that exhibit negative optimization, and apply this filtering policy uniformly to all compilation-based configurations, including torch.compile, its C++ wrapper, and Trident. Across all six model–dataset combinations, Trident consistently improves every reported median metric: it achieves 1.30–1.47× end-to-end latency speedup, 1.14–1.23× TPOT speedup, and 1.13–1.24× ITL speedup. In contrast, torch.compile reaches only 0.86–1.03× of the eager baseline in end-to-end latency and 0.70–0.90× in TPOT, while the C++ wrapper reaches 0.86–1.13× and 0.71–0.99× of the eager baseline, respectively. Thus, conventional compilation frequently regresses relative to the already optimized eager Python wrapper, whereas Trident turns host compilation into consistent end-to-end gains.

The improvement is consistent across both model architectures and all three workload domains. For torch.compile, Trident achieves 1.33–1.68× median end-to-end latency speedup; relative to the C++ wrapper, it achieves 1.16–1.67×. The corresponding TPOT gains are 1.32–1.68× and 1.15– 1.67× over torch.compile and C++ wrapper, respectively. These results show that compiling only the selected graph wrapper is insuficient to reliably outperform the eager Python wrapper at model scale. By placing specialization dispatch and host execution in the same SCM, Trident preserves the optimized device kernels while reducing the residual runtime overhead around them. On Qwen3-8B, the eager Python wrapper is particularly slow on the first token: prefill issues a long sequence of large linear/bmm calls through the Python host path, so wrapper overhead accumulates before the first output token is produced. We therefore emphasize end-toend latency together with the decode-path metrics TPOT and ITL, which better reflect steady-state serving behavior.

![](images/a32778e6850e2237608ffe6a9658d79d1a271c6027c6f8e30ea762f49203eb6b.jpg)  
Figure 7. Model-level median performance relative to the eager Python wrapper on an NVIDIA H800. Results cover DeepSeek V2-Lite and Qwen3-8B on MMLU, GSM8K, and HumanEval. Higher is better; values below 1× are slower than eager.

## 5 Related Work

Reducing host-side GPU orchestration. Prior systems reduce CPU–GPU orchestration costs by changing how device work is submitted or scheduled. CUDA Graphs capture a fixed workflow and replay it with one graph launch; PyTorch applies this mechanism in its reduce-overhead mode, while GraCE uses compiler transformations, parameter-copy elimination, and cost–benefit analysis to increase the workloads for which graph replay is profitable [4, 11]. Other systems move scheduling further onto the device. Rammer constructs a static spatio-temporal schedule across and within operators [19]. Mega-kernel systems push this direction further by executing multiple operators within one device-resident kernel, thereby avoiding repeated host submissions. MonoNN compiles a static single-GPU neural network into a monolithic kernel, while FlashMoE fuses the computation and communication of a distributed MoE layer into one persistent kernel [3, 35]. MPK automates model-level mega kernelization by lowering multi-GPU inference to an SMlevel task graph executed by an in-kernel runtime [8]; Event Tensor further encodes fine-grained task dependencies while supporting dynamic shapes and data-dependent execution in mega-kernels [15]. These approaches reduce launch gaps by changing the granularity and location of device scheduling. In contrast, Trident preserves existing device-kernel bound aries and compiles the framework-side guards, specialization selection, and host execution that precede and connect their launches.

Host code generation. The closest systems compile host logic around device kernels. TorchInductor generates a host wrapper for each captured FX graph to perform tasks such as tensor-size computation, memory management, and calls to generated or external kernels; its C++ wrapper reduces the interpretation overhead of the default Python wrapper [4]. However, Dynamo still performs specialization-cache lookup and guard evaluation before entering that wrapper. Dynamicshape compilers generate a broader runtime path: DISC compiles shape inference, bufer management, kernel-launch control, and device computation together, while BladeDISC combines multiversion code generation with runtime selec tion of shape-appropriate kernels [33, 34]. TVM FFI provides a compact calling convention and zero-copy tensor interoperability for connecting compiled kernels to framework runtimes [5]. TileLang provides a controllable language and compiler for fused neural kernels [28]. In the DeepSeek-V4 deployment stack, Host Codegen complements these kernels by co-generating a lightweight host launcher that validates tensor contracts and marshals arguments through TVM FFI [10]. This launcher removes Python from the perkernel validation path. These systems generate host logic within one graph execution or around an individual kernel, whereas Trident compiles guard-based selection and host execution across multiple Dynamo specializations into one SCM.

Compiling dynamic eager programs. Work on imperative and dynamic neural programs has primarily focused on determining which computation can be represented and optimized as a graph. JANUS speculatively converts Python control flow, dynamic types, and side efects into a symbolic graph [13]; Nimble represents dynamic control flow, data structures, and tensor shapes with a dynamic type system and a lightweight virtual machine [22]; and MAGPY monitors execution state and reference relationships to capture more complete operator graphs from eager programs [31]. Within PyTorch 2’s torch.compile stack, TorchDynamo symbolically analyzes Python bytecode to extract FX graphs and associated guards, reusing a compiled specialization when its guards hold [4].

## 6 Conclusion

This paper studies host-side overhead in PyTorch programs that invoke user-written Triton kernels. For short kernels, specialization management and host preparation outside compiled execution can dominate end-to-end latency. We propose Trident, a Torch-MLIR-based compiler backend that addresses this cost with the Specialization Cache Module (SCM). The SCM compiles guarded specialization selection, preparation, and host execution into a single executable module, so a cache hit stays in compiled code and returns to Python only when a new specialization must be compiled. Our evaluation shows that Trident substantially reduces end-to-end latency on Triton operators and model workloads, indicating that compiling specialization management together with host execution is an efective way to realize the performance of customized Triton kernels.

## References

[1] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jefrey Dean, Matthieu Devin, Sanjay Ghemawat, Geofrey Irving, Michael Isard, et al. 2016. {TensorFlow}: a system for {Large-Scale} machine learning. In 12th USENIX symposium on operating systems design and implementation (OSDI 16). 265–283.

[2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).

[3] Osayamen Jonathan Aimuyo, Byungsoo Oh, and Rachee Singh. 2025. FlashMoE: Fast Distributed MoE in a Single Kernel. In Advances in Neural Information Processing Systems, Vol. 38.

[4] Jason Ansel, Edward Yang, Horace He, Natalia Gimelshein, Animesh Jain, Michael Voznesensky, Bin Bao, Peter Bell, David Berard, Evgeni Burovski, et al. 2024. Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation. In Proceedings of the 29th ACM international conference on architectural support for programming languages and operating systems, volume 2. 929–947.

[5] Apache TVM FFI Community. 2025. Building an Open ABI and FFI for ML Systems. htps://tvm.apache.org/2025/10/21/tvm-fi

[6] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. 2018. JAX: composable transformations of Python+NumPy programs. htp:// github.com/google/jax. Version 0.3.13.

[7] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen

Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, An drew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374 (2021).

[8] Xinhao Cheng, Zhihao Zhang, Yu Zhou, Jianan Ji, Jinchen Jiang, Zepeng Zhao, Ziruo Xiao, Zihao Ye, Yingyi Huang, Ruihang Lai, Hongyi Jin, Bohan Hou, Mengdi Wu, Yixin Dong, Anthony Yip, Zihao Ye, Songting Wang, Wenqin Yang, Xupeng Miao, Tianqi Chen, and Zhihao Jia. 2026. MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs. In 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26). 1909–1926.

[9] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Rei Nakano, Christopher Hesse, and John Schulman. 2021. Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168 (2021).

[10] DeepSeek-AI. 2026. DeepSeek-V4: Towards Highly Eficient Million-Token Context Intelligence. arXiv preprint arXiv:2606.19348 (2026).

[11] Abhishek Ghosh, Ajay Nayak, Ashish Panwar, and Arkaprava Basu. 2026. {GraCE}: Unlocking {CUDA} Graphs with Compiler Support for {ML} Workloads. In 20th USENIXSymposium on Operating Systems Design and Implementation (OSDI 26). 1927–1947.

[12] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring Massive Multitask Language Understanding. International Conference on Learning Representations (2021).

[13] Eunji Jeong, Sungwoo Cho, Gyeong-In Yu, Joo Seong Jeong, Dong-Jin Shin, and Byung-Gon Chun. 2019. JANUS: Fast and Flexible Deep Learning via Symbolic Graph Execution of Imperative Programs. In 16th USENIX S<sub>y</sub>m<sub>p</sub>osium on Networked S<sub>y</sub>stems Desi<sub>g</sub>n and Im<sub>p</sub>lementation (NSDI 19). 453–468.

[14] Yangqing Jia, Evan Shelhamer, Jef Donahue, Sergey Karayev, Jonathan Long, Ross Girshick, Sergio Guadarrama, and Trevor Darrell. 2014. Cafe: Convolutional architecture for fast feature embedding. In Proceedings of the 22nd ACM international conference on Multimedia. 675– 678.

[15] Hongyi Jin, Bohan Hou, Guanjie Wang, Ruihang Lai, Jinqi Chen, Zihao Ye, Yaxing Cai, Yixin Dong, Xinhao Cheng, Zhihao Zhang, Yilong Zhao, Yingyi Huang, Lijie Yang, Jinchen Jiang, Gabriele Oliaro, Jianan Ji, Xu peng Miao, Vinod Grover, Todd C. Mowry, Zhihao Jia, and Tianqi Chen. 2026. Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel. arXiv preprint arXiv:2604.13327 (2026).

[16] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Eficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings ofthe ACM SIGOPS 29th Symposium on Operating Systems Principles.

[17] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).

[18] LLVM. n.d.. Torch-MLIR. htps://github.com/llvm/torch-mlir

[19] Lingxiao Ma, Zhiqiang Xie, Zhi Yang, Jilong Xue, Youshan Miao, Wei Cui, Wenxiang Hu, Fan Yang, Lintao Zhang, and Lidong Zhou. 2020. Rammer: Enabling Holistic Deep Learning Compiler Optimizations

with rTasks. In 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20). 881–897.

[20] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems 32 (2019).

[21] James Reed, Zachary DeVito, Horace He, Ansley Ussery, and Jason Ansel. 2022. torch. fx: Practical program capture and transformation for deep learning in python. Proceedings ofMachine Learning and Systems 4 (2022), 638–651.

[22] Haichen Shen, Jared Roesch, Zhi Chen, Wei Chen, Yong Wu, Mu Li, Vin Sharma, Zachary Tatlock, and Yida Wang. 2021. Nimble: Eficiently Compiling Dynamic Neural Networks for Model Inference. Proceedings of Machine Learning and Systems 3 (2021).

[23] The Theano Development Team, Rami Al-Rfou, Guillaume Alain, Amjad Almahairi, Christof Angermueller, Dzmitry Bahdanau, Nicolas Ballas, Frédéric Bastien, Justin Bayer, Anatoly Belikov, et al. 2016. Theano: A Python framework for fast computation of mathematical expressions. arXiv preprint arXiv:1605.02688 (2016).

[24] The FlagOS Contributors. 2024. FlagGems: An Operator Library for Large Language Models Implemented in the Triton Language. htps: //github.com/flagos-ai/FlagGems.

[25] The Perfetto Authors. 2019. Perfetto: System profiling, app tracing and trace analysis. htps://perfeto.dev/.

[26] Philippe Tillet, Hsiang-Tsung Kung, and David Cox. 2019. Triton: an intermediate language and compiler for tiled neural network computations. In Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages. 10–19.

[27] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).

[28] Lei Wang, Yu Cheng, Yining Shi, Zhiwen Mo, Zhengju Tang, Wenhao Xie, Tong Wu, Lingxiao Ma, Yuqing Xia, Jilong Xue, Fan Yang, and Zhi Yang. 2026. TileLang: Bridge Programmability and Performance in Modern Neural Kernels. In International Conference on Learning Representations.

[29] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, et al. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations. 38–45.

[30] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025).

[31] Chen Zhang, Rongchao Dong, Haojie Wang, Runxin Zhong, Jike Chen, andJidong Zhai. 2024. MAGPY: Compiling Eager Mode DNN Programs by Monitoring Execution States. In 2024 USENIX Annual Technical Conference (USENIX ATC 24). 683–698.

[32] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jef Huang, Cody H Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. 2024. Sglang: Eficient execution of structured language model programs. Advances in neural information processing systems 37 (2024), 62557–62583.

[33] Zhen Zheng, Zaifeng Pan, Dalin Wang, Kai Zhu, Wenyi Zhao, Tianyou Guo, Xiafei Qiu, Minmin Sun, Junjie Bai, Feng Zhang, Xiaoyong Du, Jidong Zhai, and Wei Lin. 2023. BladeDISC: Optimizing Dynamic Shape Machine Learning Workloads via Compiler Approach. Proceedings ofthe ACM on Management ofData 1, 3, Article 206 (2023), 29 pages. doi:10.1145/3617327

[34] Kai Zhu, Wenyi Zhao, Zhen Zheng, Tianyou Guo, Pengzhan Zhao, Feiwen Zhu, Junjie Bai, Jun Yang, Xiaoyong Liu, Lansong Diao, and Wei Lin. 2021. DISC: A Dynamic Shape Compiler for Machine Learning

Workloads. In Proceedings of the 1st Workshop on Machine Learning and S stems. 89–95. doi:10.1145/3437984.3458838

[35] Donglin Zhuang, Zhen Zheng, Haojun Xia, Xiafei Qiu, Junjie Bai, Wei Lin, and Shuaiwen Leon Song. 2024. MonoNN: Enabling a New Monolithic Optimization Space for Neural Network Inference Tasks on Modern GPU-Centric Architectures. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24). 989–1005.