# CUTLASS Tutorial: From Basics to Advanced

**Target Audience**: CUDA engineers who want to understand CUTLASS internals and APIs
**Prerequisites**: Familiarity with CUDA programming, C++ templates, and GPU architecture basics

This tutorial series provides hands-on examples exploring every concept from the [CUTLASS Deep Dive](../../CUTLASS_DEEP_DIVE.md) and [Hacker's Guide](../../CUTLASS_HACKERS_GUIDE.md).

---

## 📚 Learning Path

### **Level 1: Foundations** (Start Here)

| Example | Concepts | Architecture | Lines |
|---------|----------|--------------|-------|
| [`01_basic_gemm_volta.cu`](01_basic_gemm_volta.cu) | Template instantiation, MMA basics, 2-stage pipeline | SM70 (Volta) | ~300 |
| [`02_ampere_async.cu`](02_ampere_async.cu) | cp.async, multistage pipeline, async barriers | SM80 (Ampere) | ~350 |
| [`03_hopper_tma_gmma.cu`](03_hopper_tma_gmma.cu) | TMA descriptors, GMMA, warp specialization | SM90 (Hopper) | ~400 |

**What You'll Learn:**
- How CUTLASS decomposes GEMM across 4 hierarchical levels
- Evolution of Tensor Core instructions (mma.sync → GMMA)
- Data movement strategies (manual → cp.async → TMA)
- Pipeline depth and latency hiding

### **Level 2: Core Abstractions**

| Example | Concepts | Key Skills |
|---------|----------|------------|
| [`04_cute_layouts.cu`](04_cute_layouts.cu) | CuTe Shape/Stride/Layout/Tensor | Layout algebra, hierarchical indexing |
| [`05_collective_builders.cu`](05_collective_builders.cu) | Collective pattern, auto-selection | Builder pattern, architecture portability |
| [`06_mixed_precision.cu`](06_mixed_precision.cu) | FP16/FP8/INT8 GEMMs, type conversions | Precision management, accumulation |

**What You'll Learn:**
- CuTe's compile-time tensor algebra
- How Collective Builders abstract architecture complexity
- Mixed-precision workflows and automatic conversions

### **Level 3: Optimization**

| Example | Concepts | Performance Impact |
|---------|----------|-------------------|
| [`07_epilogue_fusion.cu`](07_epilogue_fusion.cu) | Custom epilogues, operator fusion | 20-40% speedup (eliminates kernel launches) |
| [`08_performance_tuning.cu`](08_performance_tuning.cu) | Tile sizes, clusters, swizzling, profiling | Up to 2× speedup via tuning |
| [`09_advanced_patterns.cu`](09_advanced_patterns.cu) | Split-K, Stream-K, persistent threads | Better parallelism for skinny problems |

**What You'll Learn:**
- Kernel fusion techniques
- Systematic performance tuning methodology
- Advanced scheduling strategies

### **Level 4: Advanced Topics**

| Example | Concepts | Use Cases |
|---------|----------|-----------|
| [`10_blackwell_blockscaled.cu`](10_blackwell_blockscaled.cu) | NVFP4/MXFP8, block scaling, UMMA | LLM inference, quantization |
| [`11_custom_extensions.cu`](11_custom_extensions.cu) | Custom types, tile iterators, epilogues | Novel architectures, research |
| [`12_real_world_attention.cu`](12_real_world_attention.cu) | Transformer attention patterns | Transformers, BERT, GPT |

**What You'll Learn:**
- Cutting-edge Blackwell features
- Extending CUTLASS for custom needs
- Production ML workloads

---

## 🚀 Quick Start

### Build All Examples

```bash
cd /path/to/cutlass/examples/cutlass_tutorial

# Option 1: Using Make
make -j$(nproc)

# Option 2: Using CMake
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS="70;80;90;100"
make -j$(nproc)
```

### Run Specific Example

```bash
# Volta example (if you have SM70+ GPU)
./01_basic_gemm_volta --m=2048 --n=2048 --k=2048

# Hopper example (requires SM90+ GPU)
./03_hopper_tma_gmma --m=4096 --n=4096 --k=4096 --verify
```

### Run All Tests

```bash
make test
# or
ctest --output-on-failure
```

---

## 📖 Reading Each Example

Every example follows this structure:

```cpp
/******************************************************************************
 * EDUCATIONAL OVERVIEW
 * - What this example demonstrates
 * - Key concepts covered
 * - Performance characteristics
 ******************************************************************************/

/******************************************************************************
 * SECTION 1: Type Definitions
 * [Explanation of data types, layouts, precision choices]
 ******************************************************************************/

/******************************************************************************
 * SECTION 2: Tile and Cluster Configuration
 * [Explanation of tile size selection, cluster usage]
 ******************************************************************************/

/******************************************************************************
 * SECTION 3: Collective Construction
 * [Deep dive into mainloop and epilogue collectives]
 ******************************************************************************/

/******************************************************************************
 * SECTION 4: Kernel Composition
 * [How components assemble into final kernel]
 ******************************************************************************/

/******************************************************************************
 * SECTION 5: Execution and Verification
 * [Host code, argument setup, verification]
 ******************************************************************************/
```

**Pro Tip:** Read the source code top-to-bottom, following the numbered sections. Educational comments explain *why* not just *what*.

---

## 🎯 Concept Coverage Matrix

This matrix shows which examples cover which architectural concepts:

| Concept | Examples |
|---------|----------|
| **Hierarchical Decomposition** | 01, 02, 03 |
| **Template Metaprogramming** | 01, 02, 03, 05, 11 |
| **CuTe Layout Algebra** | 04, 05, 06, 12 |
| **Volta Tensor Cores (mma.sync)** | 01 |
| **Ampere (cp.async, TF32)** | 02, 06 |
| **Hopper (TMA, GMMA, clusters)** | 03, 05, 08, 12 |
| **Blackwell (UMMA, block scaling)** | 10 |
| **Collective Pattern** | 05, 06, 07, 08 |
| **Warp Specialization** | 03, 08 |
| **Pipeline Architecture** | 01, 02, 03, 08 |
| **Mixed Precision** | 06, 10 |
| **Epilogue Fusion** | 07, 12 |
| **Performance Tuning** | 08, 09 |
| **Split-K / Stream-K** | 09 |
| **Custom Extensions** | 11 |

---

## 🔍 Deep Dive into Each Example

### 01_basic_gemm_volta.cu - Your First CUTLASS Kernel

**Learning Objectives:**
- Understand the 4-level CUTLASS hierarchy (Device → CTA → Warp → Thread)
- See how template instantiation generates architecture-specific code
- Learn the 2-stage pipeline pattern on Volta

**Key Code Snippets:**
```cpp
// Volta MMA instruction: 16×8×8 (M×N×K)
using MmaOp = arch::Mma<
    gemm::GemmShape<16, 8, 8>,
    32,                          // Threads per warp
    half_t, RowMajor,           // A operand
    half_t, ColumnMajor,        // B operand
    float, RowMajor             // Accumulator
>;
```

**Architecture Insight:**
Volta's `mma.sync` requires explicit thread synchronization. Each warp performs a 16×8×8 matrix multiply in lock-step. The 2-stage pipeline hides loading of the next tile while computing the current one.

---

### 02_ampere_async.cu - Async Memory Pipeline

**Learning Objectives:**
- Use `cp.async` for overlapped data movement
- Implement multistage pipelines (3-7 stages)
- Understand async barrier patterns

**Key Code Snippets:**
```cpp
// Async copy: global → shared
cute::cp_async(smem_ptr, gmem_ptr);

// Commit async operations
cute::cp_async_commit_group();

// Wait for specific stage
cute::cp_async_wait_group<N-1>();  // Wait for all but last N stages
```

**Architecture Insight:**
Ampere's async copy instructions decouple data movement from computation. While threads wait for old data, the GPU can continue loading new data, dramatically improving pipeline efficiency from 2 stages (Volta) to 3-7 stages (Ampere).

---

### 03_hopper_tma_gmma.cu - Hardware-Accelerated GEMM

**Learning Objectives:**
- Create and use TMA descriptors
- Leverage GMMA warpgroup instructions
- Implement warp specialization (producer/consumer roles)
- Use thread block clusters

**Key Code Snippets:**
```cpp
// TMA descriptor creation (host-side)
CUtensorMap tma_desc_A;
cuTensorMapEncodeTiled(
    &tma_desc_A,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    2,                           // 2D tensor
    global_ptr,
    shape,
    stride,
    box_dim,                     // Tile dimensions
    CU_TENSOR_MAP_SWIZZLE_128B   // Auto swizzle for bank conflicts
);

// Producer warp: TMA load (device-side)
if (warp_id == 0) {
    tma_load_2d(tma_desc_A, smem_A, tile_m, tile_k);
}

// Consumer warps: GMMA compute
if (warp_id > 0) {
    wgmma::mma_async(smem_A, smem_B, accum);
}
```

**Architecture Insight:**
Hopper's TMA is a specialized hardware unit that handles bulk tensor copies. A single thread can initiate a multi-KB transfer, freeing other threads for computation. GMMA operates on entire warpgroups (128 threads), achieving higher throughput than per-warp instructions.

---

### 04_cute_layouts.cu - Tensor Algebra Fundamentals

**Learning Objectives:**
- Master Shape, Stride, Layout, and Tensor abstractions
- Perform layout composition and transformation
- Understand compile-time indexing
- Implement custom memory access patterns

**Key Code Snippets:**
```cpp
// Basic layout
auto layout = make_layout(
    make_shape(128, 64),      // 128 rows × 64 cols
    make_stride(1, 128)       // Column-major: stride 1 in rows, 128 in cols
);

// Hierarchical layout: (4 warps × 32 threads) × 8 elements
auto nested = make_layout(
    make_shape(make_shape(4, 32), 8),
    make_stride(make_stride(32, 1), 128)
);

// Layout composition
auto swizzled = composition(
    Swizzle<2, 3, 3>{},       // XOR swizzle for bank conflicts
    layout
);

// Partitioning across threads
auto my_partition = local_partition(
    global_tensor,
    thread_layout,
    thread_id
);
```

**Architecture Insight:**
CuTe's layouts are entirely compile-time. The `layout(i, j)` operation resolves to a constant offset when indices are known at compile time, resulting in zero runtime overhead. This enables the compiler to generate optimal PTX with direct addressing.

---

### 05_collective_builders.cu - Architecture Abstraction

**Learning Objectives:**
- Use CollectiveBuilder for portable code
- Understand automatic architecture selection
- Compare manual vs. automatic configuration
- Inspect generated collective types

**Key Code Snippets:**
```cpp
// Automatic collective selection
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag,                     // SM70, SM80, SM90, SM100
    OpClass,                     // TensorOp, SparseTensorOp
    ElementA, LayoutA, AlignA,
    ElementB, LayoutB, AlignB,
    ElementAccum,
    TileShape, ClusterShape,
    StageCountAuto,              // Compute optimal stages
    KernelScheduleAuto           // Select best schedule
>::CollectiveOp;

// Inspect what was selected
using SelectedMma = typename CollectiveMainloop::MmaOp;
using SelectedCopy = typename CollectiveMainloop::CopyOp;
```

**Architecture Insight:**
The builder uses SFINAE and tag dispatch to select implementations. For example:
- SM70 → `Sm70MmaTwoStage`
- SM80 → `Sm80MmaMultistage` with `CopyAtomCpAsync`
- SM90 → `Sm90MmaTmaGmmaWarpSpecialized` with `CopyAtomTMA`

This happens entirely at compile time with zero runtime branching.

---

### 06_mixed_precision.cu - Precision Management

**Learning Objectives:**
- Implement FP16×FP16→FP32 GEMM
- Use FP8 (E4M3/E5M2) formats
- Understand INT8 quantized inference
- Handle automatic type conversions in epilogue

**Key Code Snippets:**
```cpp
// FP8 GEMM: E4M3 inputs, FP32 accumulation, FP16 output
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementAccum = float;
using ElementC = cutlass::half_t;

// Tensor Cores automatically convert E4M3→FP32 during MMA
// Epilogue converts FP32→FP16 during store

// INT8 with dequantization
struct DequantizeEpilogue {
    float scale_A, scale_B;

    CUTLASS_DEVICE
    float operator()(int32_t acc, float source) const {
        return float(acc) * scale_A * scale_B + source;
    }
};
```

**Architecture Insight:**
Tensor Cores perform type conversions in hardware:
- FP8→FP32: 1 cycle (Hopper+)
- INT8→INT32: 0 cycles (sign extension)
- FP32→FP16: 1 cycle with rounding

Mixed precision enables 2-8× higher throughput while maintaining accuracy through FP32 accumulation.

---

### 07_epilogue_fusion.cu - Kernel Fusion

**Learning Objectives:**
- Fuse activation functions (ReLU, GELU, Sigmoid)
- Add bias and residual connections
- Implement custom epilogue operators
- Measure fusion speedup

**Key Code Snippets:**
```cpp
// Fused GEMM + Bias + GELU
template<typename T>
struct BiasGELUEpilogue {
    T const* bias;

    CUTLASS_DEVICE
    T operator()(T accumulator, T source, int row, int col) const {
        T result = accumulator + source + bias[col];

        // GELU approximation
        T x3 = result * result * result;
        T tanh_arg = T(0.7978845608) * (result + T(0.044715) * x3);
        return T(0.5) * result * (T(1.0) + tanh(tanh_arg));
    }
};
```

**Architecture Insight:**
Without fusion: 3 kernel launches (GEMM → Bias → GELU) = 3× memory traffic
With fusion: 1 kernel launch = 1× memory traffic

The epilogue runs in registers before writing to memory, eliminating intermediate global memory accesses. Typical speedup: 20-40% for small activations, up to 2× for element-wise operations.

---

### 08_performance_tuning.cu - Systematic Optimization

**Learning Objectives:**
- Tune tile sizes (64×64 vs. 128×128 vs. 256×128)
- Configure cluster shapes (1×1 vs. 2×2 vs. 4×2)
- Select swizzle patterns (AlongM, AlongN, Heuristic)
- Measure and profile performance
- Use CUTLASS profiler for auto-tuning

**Key Code Snippets:**
```cpp
// Tile size sweep
constexpr TileSize[] = {
    Shape<_64, _64, _32>{},
    Shape<_128, _128, _32>{},
    Shape<_256, _128, _64>{}
};

for (auto tile : TileSize) {
    auto gemm = create_gemm(tile);
    float tflops = benchmark(gemm);
    std::cout << "Tile " << tile << ": " << tflops << " TFLOPS\n";
}

// Cluster tuning (Hopper+)
using ClusterShape = Shape<_4, _2, _1>;  // 4×2 = 8 CTAs per cluster

// Swizzle tuning
arguments.tile_scheduler.raster_order = RasterOrderOptions::AlongN;
arguments.tile_scheduler.swizzle_size = 2;
```

**Architecture Insight:**
Performance tuning is a multi-dimensional optimization:

| Parameter | Small Values | Large Values |
|-----------|--------------|--------------|
| **Tile Size** | Lower occupancy, less reuse | Higher reuse, may exhaust smem |
| **Cluster** | More parallelism | Better L2 locality |
| **Swizzle** | Simple traversal | Improved cache behavior |
| **Stages** | Less latency hiding | More latency hiding, higher smem |

Optimal values depend on problem size, data types, and architecture. The CUTLASS profiler can exhaustively search this space.

---

### 09_advanced_patterns.cu - Scheduling Strategies

**Learning Objectives:**
- Implement Split-K for skinny GEMMs
- Use Stream-K for better load balancing
- Employ persistent thread blocks
- Handle grouped GEMMs (multiple different-sized GEMMs in one kernel)

**Key Code Snippets:**
```cpp
// Split-K: Divide K dimension across CTAs
int split_k_slices = 8;
arguments.split_k_mode = SplitKMode::kParallel;
arguments.split_k_slices = split_k_slices;

// Stream-K: Dynamic work distribution
using TileScheduler = cutlass::gemm::kernel::StreamKScheduler;

// Grouped GEMM: Different sizes in one launch
struct GroupedGemmArguments {
    int group_count;
    ProblemSize* problem_sizes;  // Array of {M, N, K} per group
    void** ptr_A;                // Array of A pointers
    void** ptr_B;                // Array of B pointers
    void** ptr_C;                // Array of C pointers
};
```

**Architecture Insight:**
**Split-K** partitions reduction across CTAs:
- Normal: M×N CTAs
- Split-K: M×N×split_k_slices CTAs
- Requires final reduction kernel
- Good for: Small M or N, large K

**Stream-K** dynamically assigns tiles:
- No static CTA-to-tile mapping
- Better load balancing for irregular problems
- Persistent threads grab work from queue
- Good for: Variable-sized batched GEMMs

---

### 10_blackwell_blockscaled.cu - Next-Gen Precision

**Learning Objectives:**
- Use NVFP4 (4-bit floating point) format
- Understand block scaling mechanism
- Leverage MXFP8/MXFP6/MXFP4 (Microsoft/OCP formats)
- Achieve 2-4× memory savings with minimal accuracy loss

**Key Code Snippets:**
```cpp
// NVFP4: 4 bits per element + shared scale
using ElementA = cutlass::nvfp4_t;
using ElementScale = cutlass::bfloat16_t;

// Block size: 32 elements share 1 scale
constexpr int kBlockSize = 32;

// Block-scaled collective
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,        // Blackwell only
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 32,       // 32-element alignment for NVFP4
    ElementB, LayoutB, 32,
    float,                       // Accumulate in FP32
    TileShape, ClusterShape,
    StageCountAuto,
    KernelScheduleAuto,
    MainloopBlockScaled          // Enable block scaling
>::CollectiveOp;

// Arguments include scale tensors
arguments.mainloop_args = {
    A_data.get(), {K, 1},
    B_data.get(), {N, 1},
    {A_scales.get(), B_scales.get()}  // Scale pointers
};
```

**Architecture Insight:**
Block scaling trades precision for memory bandwidth:

| Format | Bits/Element | Effective Bits | Speedup (vs FP16) |
|--------|--------------|----------------|-------------------|
| FP16 | 16 | 16 | 1× (baseline) |
| FP8 (E4M3) | 8 | 8 | 2× |
| MXFP8 | 8 + scale/32 | ~8.5 | 1.9× |
| NVFP4 | 4 + scale/32 | ~4.5 | 3.5× |

UMMA instructions perform implicit dequantization:
```
result[i] = A_data[i] * A_scale[i/32] * B_data[j] * B_scale[j/32]
```

This happens in hardware with no extra instructions!

---

### 11_custom_extensions.cu - Extending CUTLASS

**Learning Objectives:**
- Define custom numeric types
- Implement custom tile iterators
- Write custom MMA operations
- Create domain-specific extensions

**Key Code Snippets:**
```cpp
// Custom 8-bit brain float (hypothetical)
struct bf8_t {
    uint8_t storage;

    CUTLASS_HOST_DEVICE
    bf8_t(float f) : storage(float_to_bf8(f)) {}

    CUTLASS_HOST_DEVICE
    operator float() const { return bf8_to_float(storage); }
};

// Register with CUTLASS
namespace cutlass {
template<>
struct sizeof_bits<bf8_t> {
    static constexpr int value = 8;
};
}

// Custom tile iterator with software prefetching
template<typename Element>
class PrefetchingIterator {
    CUTLASS_DEVICE
    void load_with_prefetch(Fragment& frag, int k_future) {
        // Load current tile
        ldmatrix(frag, smem_ptr_);

        // Prefetch future tile into L2
        if (k_future < k_tiles_) {
            prefetch_l2(gmem_ptr_ + k_future * stride_);
        }
    }
};
```

**Architecture Insight:**
CUTLASS is designed for extensibility at every level:

1. **Type level**: Add new numeric formats
2. **Iterator level**: Custom memory access patterns
3. **MMA level**: Novel computation patterns
4. **Epilogue level**: Domain-specific operations

The template system ensures new types/operations compose seamlessly with existing infrastructure.

---

### 12_real_world_attention.cu - Transformer Patterns

**Learning Objectives:**
- Implement attention mechanism (QK^T, Softmax, V)
- Fuse operations for efficiency
- Handle batched attention (multi-head)
- Optimize for typical sequence lengths (512-4096)

**Key Code Snippets:**
```cpp
// Attention: Softmax(Q × K^T / sqrt(d)) × V

// Step 1: Q × K^T (batched GEMM)
// Shape: [batch, num_heads, seq_len, seq_len]
for (int head = 0; head < num_heads; ++head) {
    gemm_qk(Q[head], K_transpose[head], scores[head]);
}

// Step 2: Fused Scale + Softmax + Mask (custom epilogue)
struct ScaleSoftmaxMask {
    float scale;
    bool* attention_mask;

    CUTLASS_DEVICE
    float operator()(float score, int row, int col) {
        // Scale
        float scaled = score * scale;

        // Mask (set to -inf if masked)
        if (attention_mask && !attention_mask[row * seq_len + col]) {
            scaled = -INFINITY;
        }

        // Softmax (done row-wise in epilogue)
        return exp(scaled);  // Normalization in separate reduction
    }
};

// Step 3: Attention × V (another batched GEMM)
for (int head = 0; head < num_heads; ++head) {
    gemm_av(attention[head], V[head], output[head]);
}
```

**Architecture Insight:**
Naive attention: 5 kernel launches (QK^T, Scale, Softmax, Mask, AV)
Optimized (Flash Attention style):
- Fuse QK^T + Scale + Mask
- Online softmax (avoid materializing scores)
- Single pass through V

Memory reduction: O(seq_len²) → O(1) intermediate storage
Speedup: 2-4× for typical models

---

## 🧪 Testing and Verification

Each example includes:

1. **Correctness Verification**: Comparison against reference implementation
2. **Performance Measurement**: TFLOPS reporting
3. **Architecture Detection**: Auto-skip if GPU doesn't support required SM
4. **Detailed Logging**: Explains what's being tested

```bash
# Run with verification
./01_basic_gemm_volta --verify

# Run with profiling info
./08_performance_tuning --profile --verbose

# Run all tests
make test
```

---

## 📊 Performance Expectations

Approximate performance on NVIDIA GPUs (FP16 GEMM, M=N=K=4096):

| Architecture | GPU | Theoretical Peak | CUTLASS Achieved | % of Peak |
|--------------|-----|------------------|------------------|-----------|
| Volta (SM70) | V100 | 125 TFLOPS | 120 TFLOPS | 96% |
| Ampere (SM80) | A100 | 312 TFLOPS | 305 TFLOPS | 98% |
| Hopper (SM90) | H100 | 989 TFLOPS | 970 TFLOPS | 98% |
| Blackwell (SM100) | B200 | 2.25 PFLOPS | 2.2 PFLOPS | 98% |

These examples should achieve similar results with proper tuning.

---

## 🔗 Additional Resources

- **Official Docs**: [CUTLASS Documentation](https://nvidia.github.io/cutlass)
- **Deep Dive**: [CUTLASS_DEEP_DIVE.md](../../CUTLASS_DEEP_DIVE.md)
- **Hacker's Guide**: [CUTLASS_HACKERS_GUIDE.md](../../CUTLASS_HACKERS_GUIDE.md)
- **CUTLASS Repository**: [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
- **Discussions**: [GitHub Discussions](https://github.com/NVIDIA/cutlass/discussions)

---

## 🤝 Contributing

Found issues or have improvements? Please:
1. Test your changes with `make test`
2. Ensure examples build on multiple architectures
3. Add educational comments explaining *why* not just *what*
4. Update this README if adding new examples

---

## ⚖️ License

This tutorial series is part of CUTLASS and follows the same BSD-3-Clause license.

---

**Happy Learning!** 🚀

Start with `01_basic_gemm_volta.cu` and work your way through. By the end, you'll understand CUTLASS internals better than most GPU engineers!
