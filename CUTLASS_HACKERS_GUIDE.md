# CUTLASS Hacker's Guide: Practical Recipes and Patterns

**Version**: 3.9.0
**Companion to**: [CUTLASS Deep Dive](CUTLASS_DEEP_DIVE.md)
**Last Updated**: January 2026

---

## Table of Contents

1. [Quick Start: Your First GEMM](#quick-start)
2. [Building and Running CUTLASS](#building-and-running)
3. [Common GEMM Patterns](#common-gemm-patterns)
4. [Advanced Techniques](#advanced-techniques)
5. [Profiling and Tuning](#profiling-and-tuning)
6. [Debugging CUTLASS Code](#debugging)
7. [Extending CUTLASS](#extending-cutlass)
8. [Real-World Use Cases](#real-world-use-cases)
9. [Performance Troubleshooting](#performance-troubleshooting)
10. [Tips and Tricks](#tips-and-tricks)

---

## 1. Quick Start: Your First GEMM {#quick-start}

### Hello CUTLASS: FP32 GEMM on Hopper

Let's build a simple FP32 GEMM using CUTLASS 3.x on NVIDIA Hopper (SM90):

```cpp
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

using namespace cute;

// Step 1: Define your data types
using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAccumulator = float;

// Step 2: Define layouts
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

// Step 3: Define tile shapes
using TileShape = Shape<_128, _128, _32>;      // CTA tile: 128×128×32
using ClusterShape = Shape<_2, _1, _1>;        // 2×1 cluster

// Step 4: Build the collective mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,                       // Target architecture
    cutlass::arch::OpClassTensorOp,            // Use Tensor Cores
    ElementA, LayoutA, 4,                      // A: float, row-major, align=4
    ElementB, LayoutB, 4,                      // B: float, column-major, align=4
    ElementAccumulator,                        // Accumulator type
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto, // Auto-tune pipeline stages
    cutlass::gemm::collective::KernelScheduleAuto // Auto-select schedule
>::CollectiveOp;

// Step 5: Build the collective epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, 4,
    ElementC, LayoutC, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Step 6: Create the GEMM kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,                      // Problem shape (runtime)
    CollectiveMainloop,
    CollectiveEpilogue
>;

// Step 7: Wrap in device adapter
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Step 8: Launch the kernel
int main() {
    int M = 4096, N = 4096, K = 4096;

    // Allocate device memory
    cutlass::DeviceAllocation<ElementA> A(M * K);
    cutlass::DeviceAllocation<ElementB> B(K * N);
    cutlass::DeviceAllocation<ElementC> C(M * N);
    cutlass::DeviceAllocation<ElementC> D(M * N);

    // Initialize (fill with random data, etc.)
    // ... initialization code ...

    // Setup GEMM arguments
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},                             // Problem size
        {A.get(), {K, 1},                      // A: ptr + stride
         B.get(), {N, 1},                      // B: ptr + stride
         {}},                                  // Mainloop args
        {{1.0f, 0.0f},                         // alpha, beta
         C.get(), {N, 1},                      // C: ptr + stride
         D.get(), {N, 1}}                      // D: ptr + stride
    };

    // Initialize GEMM
    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM initialization failed\n";
        return -1;
    }

    // Run GEMM
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM execution failed\n";
        return -1;
    }

    cudaDeviceSynchronize();
    std::cout << "GEMM completed successfully!\n";

    return 0;
}
```

`★ Insight ─────────────────────────────────────`
**What's happening here:**
- **Collective builders** automatically select optimal implementations for your architecture
- **TileShape** controls the work each threadblock processes (larger = more reuse)
- **ClusterShape** enables multiple blocks to cooperate (Hopper feature)
- **StageCountAuto** lets CUTLASS compute optimal pipeline depth
`─────────────────────────────────────────────────`

---

## 2. Building and Running CUTLASS {#building-and-running}

### Build Configuration

```bash
# Clone CUTLASS
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# Create build directory
mkdir build && cd build

# Configure for Hopper (SM90)
cmake .. \
    -DCUTLASS_NVCC_ARCHS=90a \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUTLASS_ENABLE_EXAMPLES=ON

# Build specific example
make 48_hopper_warp_specialized_gemm -j$(nproc)

# Run
./examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm \
    --m=4096 --n=4096 --k=4096
```

### CMake for Your Project

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyGemmApp CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Point to CUTLASS
set(CUTLASS_DIR /path/to/cutlass)
include_directories(${CUTLASS_DIR}/include)
include_directories(${CUTLASS_DIR}/tools/util/include)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 90)  # Hopper

add_executable(my_gemm my_gemm.cu)
target_compile_options(my_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
```

### Compile Single File

```bash
nvcc my_gemm.cu -o my_gemm \
    -I/path/to/cutlass/include \
    -I/path/to/cutlass/tools/util/include \
    -arch=sm_90 \
    --expt-relaxed-constexpr \
    -O3 \
    -std=c++17
```

---

## 3. Common GEMM Patterns {#common-gemm-patterns}

### Pattern 1: Mixed-Precision GEMM (FP16 Input, FP32 Output)

```cpp
// FP16×FP16 → FP32 output
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAccumulator = float;  // Accumulate in FP32

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,              // FP16: 8-element alignment
    ElementB, LayoutB, 8,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// Tensor Cores automatically handle FP16→FP32 conversion
```

### Pattern 2: GEMM with ReLU Activation

```cpp
// Standard epilogue does: D = alpha * (A×B) + beta * C
// ReLU epilogue does: D = ReLU(alpha * (A×B) + beta * C)

using EpilogueOp = cutlass::epilogue::thread::LinearCombinationReLU<
    ElementC,              // Output type
    128 / cutlass::sizeof_bits<ElementC>::value, // Elements per access
    ElementAccumulator,    // Accumulator type
    ElementAccumulator     // Compute type
>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, 4,
    ElementC, LayoutC, 4,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    EpilogueOp             // Custom epilogue operation
>::CollectiveOp;
```

### Pattern 3: Batched GEMM

```cpp
// Perform multiple independent GEMMs
int batch_count = 10;
int M = 2048, N = 2048, K = 2048;

// Allocate batched arrays
std::vector<ElementA*> A_batch(batch_count);
std::vector<ElementB*> B_batch(batch_count);
std::vector<ElementC*> C_batch(batch_count);

for (int i = 0; i < batch_count; ++i) {
    cudaMalloc(&A_batch[i], M * K * sizeof(ElementA));
    cudaMalloc(&B_batch[i], K * N * sizeof(ElementB));
    cudaMalloc(&C_batch[i], M * N * sizeof(ElementC));
}

// Run batched GEMM
for (int i = 0; i < batch_count; ++i) {
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        {A_batch[i], {K, 1},
         B_batch[i], {N, 1},
         {}},
        {{1.0f, 0.0f},
         nullptr, {N, 1},        // No C input (beta=0)
         C_batch[i], {N, 1}}
    };

    gemm_op.initialize(arguments, workspace.get());
    gemm_op.run();
}
```

### Pattern 4: Strided Batched GEMM

```cpp
// All matrices in contiguous memory with fixed stride
int batch_count = 10;
int64_t batch_stride_A = M * K;
int64_t batch_stride_B = K * N;
int64_t batch_stride_C = M * N;

typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kBatched,
    {M, N, K, batch_count},        // Include batch count
    {A.get(), {K, 1},
     B.get(), {N, 1},
     {},
     batch_stride_A, batch_stride_B},  // Batch strides
    {{1.0f, 0.0f},
     nullptr, {N, 1},
     C.get(), {N, 1},
     batch_stride_C}
};
```

### Pattern 5: FP8 GEMM (Hopper+)

```cpp
using ElementA = cutlass::float_e4m3_t;  // FP8 E4M3
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;        // Output FP16
using ElementAccumulator = float;        // Accumulate in FP32

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 16,             // FP8: 16-element alignment
    ElementB, LayoutB, 16,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;
```

### Pattern 6: Block-Scaled GEMM (Blackwell)

```cpp
// NVFP4 with block scaling
using ElementA = cutlass::nvfp4_t;       // 4-bit per element
using ElementScale = cutlass::bfloat16_t;  // 16-bit scales
using ElementC = cutlass::bfloat16_t;

// Use Blackwell collective with block scaling
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,              // Blackwell
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 32,             // NVFP4: 32-element alignment
    ElementB, LayoutB, 32,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto,
    cutlass::gemm::collective::MainloopBlockScaled  // Enable block scaling
>::CollectiveOp;

// Provide scale tensors in arguments
typename Gemm::Arguments arguments{
    // ... problem size ...
    {A.get(), {K, 1},
     B.get(), {N, 1},
     {scales_A.get(), scales_B.get()}},  // Scale tensors
    // ... epilogue ...
};
```

---

## 4. Advanced Techniques {#advanced-techniques}

### Technique 1: Custom Epilogue Fusion

Fuse bias addition and GELU activation:

```cpp
template<typename ElementOutput_, typename ElementAccumulator_, typename ElementCompute_>
struct LinearCombinationBiasGELU {
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;

    struct Params {
        ElementCompute alpha;
        ElementCompute beta;
        ElementCompute const* bias_ptr;  // Per-column bias
    };

    Params params;

    CUTLASS_HOST_DEVICE
    LinearCombinationBiasGELU(Params const& params_) : params(params_) {}

    CUTLASS_DEVICE
    ElementOutput operator()(
        ElementAccumulator accumulator,
        ElementOutput source,
        int row, int column
    ) const {
        // Compute: result = alpha * accumulator + beta * source + bias[column]
        ElementCompute result =
            ElementCompute(params.alpha) * ElementCompute(accumulator) +
            ElementCompute(params.beta) * ElementCompute(source) +
            ElementCompute(params.bias_ptr[column]);

        // Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        ElementCompute x = result;
        ElementCompute x_cubed = x * x * x;
        ElementCompute tanh_arg = ElementCompute(0.7978845608) * (x + ElementCompute(0.044715) * x_cubed);
        ElementCompute gelu = ElementCompute(0.5) * x * (ElementCompute(1.0) + tanh(tanh_arg));

        return ElementOutput(gelu);
    }
};
```

### Technique 2: Split-K for Skinny Problems

When M or N is small but K is large:

```cpp
// Problem: M=512, N=512, K=16384
// Normal: only 16×16=256 threadblocks
// Split-K: split K into 8 slices → 256×8=2048 threadblocks

int split_k_slices = 8;

typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {A.get(), {K, 1},
     B.get(), {N, 1},
     {}},
    {{1.0f, 0.0f},
     C.get(), {N, 1},
     D.get(), {N, 1}},
    split_k_slices                    // Enable split-K
};

// Each slice computes partial result
// Final reduction combines partials
```

### Technique 3: Stream-K for Load Balancing

Better work distribution than split-K:

```cpp
using TileScheduler = cutlass::gemm::kernel::StreamKScheduler;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler                     // Use Stream-K scheduler
>;
```

### Technique 4: Ping-Pong Scheduling

Double-buffered producer warps:

```cpp
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, 8,
    ElementB, LayoutB, 8,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    KernelSchedule                    // Pingpong schedule
>::CollectiveOp;
```

### Technique 5: Persistent Thread Blocks

Keep threads alive across multiple tiles:

```cpp
using TileScheduler = cutlass::gemm::kernel::PersistentScheduler;

// Threads persist and grab tiles dynamically
// Better for variable-sized batched GEMMs
```

---

## 5. Profiling and Tuning {#profiling-and-tuning}

### Using CUTLASS Profiler

```bash
# Build profiler
make cutlass_profiler -j

# Profile FP16 GEMM on SM90
./tools/profiler/cutlass_profiler \
    --operation=gemm \
    --A=f16:column --B=f16:column --C=f32:column \
    --m=4096 --n=4096 --k=4096 \
    --alpha=1.0 --beta=0.0 \
    --op_class=tensorop \
    --accum=f32 \
    --providers=cutlass \
    --verification=ON

# Exhaustive search for best tile size
./tools/profiler/cutlass_profiler \
    --operation=gemm \
    --m=4096 --n=4096 --k=4096 \
    --exhaustive_search=1 \
    --top_k=10 \
    --sort_by_gflops=1
```

### Nsight Compute Profiling

```bash
# Profile specific kernel
ncu --set full --target-processes all -o profile.ncu-rep \
    ./my_gemm

# Key metrics to examine:
# - Tensor Core utilization
# - Shared memory bank conflicts
# - Global memory throughput
# - Warp stall reasons
```

### Manual Performance Measurement

```cpp
#include <chrono>

// Warmup
for (int i = 0; i < 10; ++i) {
    gemm_op.run();
}
cudaDeviceSynchronize();

// Timing
auto start = std::chrono::high_resolution_clock::now();
int iterations = 100;

for (int i = 0; i < iterations; ++i) {
    gemm_op.run();
}
cudaDeviceSynchronize();

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
double avg_time_ms = (duration / 1000.0) / iterations;

// Compute TFLOPS
double flops = 2.0 * M * N * K;  // One multiply-add = 2 FLOPs
double tflops = (flops / (avg_time_ms / 1000.0)) / 1e12;

std::cout << "Average time: " << avg_time_ms << " ms\n";
std::cout << "Performance: " << tflops << " TFLOPS\n";
```

### Tuning Checklist

1. **Tile Size**: Try 64×64, 128×128, 256×128
2. **Cluster Shape**: Test 1×1, 2×1, 2×2, 4×2
3. **Stage Count**: Auto vs. fixed (3, 5, 7)
4. **Swizzle Pattern**: AlongM, AlongN, Heuristic
5. **Alignment**: Ensure optimal (8+ for FP16, 16+ for FP8)
6. **Split-K**: For skinny problems
7. **Epilogue Fusion**: Combine operations

---

## 6. Debugging CUTLASS Code {#debugging}

### Common Compilation Errors

**Error: "No matching function for overloaded 'CollectiveBuilder'"**

```cpp
// Problem: Unsupported combination of types/architecture
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm70,               // Volta
    cutlass::arch::OpClassTensorOp,
    cutlass::float_e4m3_t, ...         // FP8 not supported on Volta!
>::CollectiveOp;

// Solution: Check compatibility matrix
// FP8 requires SM80+ (Ampere or newer)
```

**Error: "invalid memory access"**

```cpp
// Problem: Alignment mismatch
constexpr int AlignmentA = 8;          // Requesting 8-element alignment
ElementA* ptr = malloc_misaligned();   // But pointer is misaligned!

// Solution: Use aligned allocation
cudaMalloc(&ptr, size);  // CUDA allocations are 256-byte aligned
// Or check: assert((uintptr_t)ptr % (AlignmentA * sizeof(ElementA)) == 0);
```

### Runtime Debugging

```cpp
// Enable CUTLASS error checking
#define CUTLASS_DEBUG_TRACE_LEVEL 1

cutlass::Status status = gemm_op.run();
if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: " << cutlass::cutlassGetStatusString(status) << "\n";

    // Check arguments
    if (status == cutlass::Status::kErrorInvalidProblem) {
        std::cerr << "Invalid problem size: " << M << "×" << N << "×" << K << "\n";
    }
}
```

### Verification Against Reference

```cpp
#include "cutlass/util/reference/device/gemm.h"

// Run reference GEMM
cutlass::reference::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator, ElementAccumulator
> reference_gemm;

reference_gemm(
    {M, N, K},
    alpha,
    A.get(), {K, 1},
    B.get(), {N, 1},
    beta,
    C_ref.get(), {N, 1},
    D_ref.get(), {N, 1}
);

// Compare results
bool passed = cutlass::reference::device::BlockCompareEqual(
    D.get(), D_ref.get(), M * N, 1e-3f, 1e-3f
);

if (!passed) {
    std::cerr << "Verification failed!\n";
    // Print first mismatches
    // ... debugging output ...
}
```

### CUDA-MEMCHECK

```bash
# Check for memory errors
cuda-memcheck ./my_gemm

# Check for race conditions
cuda-memcheck --tool racecheck ./my_gemm

# Check for shared memory races
cuda-memcheck --tool synccheck ./my_gemm
```

---

## 7. Extending CUTLASS {#extending-cutlass}

### Custom Data Type

```cpp
// Define custom BF8 type (hypothetical 8-bit brain float)
struct bf8_t {
    uint8_t storage;

    CUTLASS_HOST_DEVICE
    bf8_t() : storage(0) {}

    CUTLASS_HOST_DEVICE
    explicit bf8_t(float f) {
        // Custom float→BF8 conversion
        storage = float_to_bf8(f);
    }

    CUTLASS_HOST_DEVICE
    operator float() const {
        return bf8_to_float(storage);
    }

private:
    static uint8_t float_to_bf8(float f) { /* conversion */ }
    static float bf8_to_float(uint8_t bf8) { /* conversion */ }
};

// Register with CUTLASS type system
namespace cutlass {
template<>
struct sizeof_bits<bf8_t> {
    static constexpr int value = 8;
};
}
```

### Custom Tile Iterator

```cpp
template<
    typename Shape_,
    typename Element_,
    typename Layout_
>
class CustomTileIterator {
public:
    using Shape = Shape_;
    using Element = Element_;
    using Layout = Layout_;

    struct Params {
        Layout layout;
        // Additional parameters
    };

    CUTLASS_DEVICE
    CustomTileIterator(
        Params const& params,
        Element* ptr,
        int thread_idx
    ) : params_(params), ptr_(ptr) {
        // Initialize thread-specific pointers
    }

    CUTLASS_DEVICE
    void load(Fragment& frag) {
        // Custom load implementation
        // Can use LDG, LDS, TMA, etc.
    }

private:
    Params params_;
    Element* ptr_;
};
```

### Custom Epilogue Operation

Already shown in Advanced Techniques (Bias + GELU example).

---

## 8. Real-World Use Cases {#real-world-use-cases}

### Use Case 1: Transformer Attention (QK^T)

```cpp
// Q: [batch, seq_len, head_dim]
// K: [batch, seq_len, head_dim]
// Output: [batch, seq_len, seq_len]

// GEMM: Q × K^T
// M = seq_len, N = seq_len, K = head_dim

int batch = 32;
int seq_len = 2048;
int head_dim = 128;

// Batched GEMM for all attention heads
for (int b = 0; b < batch; ++b) {
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {seq_len, seq_len, head_dim},
        {Q_batch[b], {head_dim, 1},
         K_batch[b], {head_dim, 1},  // K is transposed
         {}},
        {{1.0f / sqrt(head_dim), 0.0f},  // Scale by 1/sqrt(d_k)
         nullptr, {seq_len, 1},
         scores_batch[b], {seq_len, 1}}
    };
    gemm_op.run(arguments);
}

// Follow with softmax (separate kernel)
```

### Use Case 2: Convolution via Implicit GEMM

```cpp
// 2D convolution: [N, H, W, C] * [R, S, C, K] → [N, P, Q, K]
// Transform to GEMM: [NPQ, C×R×S] × [C×R×S, K] → [NPQ, K]

#include "cutlass/conv/device/implicit_gemm_convolution.h"

using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<
    ElementA, LayoutA,           // Activation
    ElementB, LayoutB,           // Filter
    ElementC, LayoutC,           // Output
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm90,
    TileShape, WarpShape,
    cutlass::conv::ConvType::kForward
>;

typename Conv2d::Arguments arguments{
    {N, H, W, C},                // Input NHWC
    filter.get(),
    {K, R, S, C},                // Filter KRSC
    output.get(),
    {N, P, Q, K},                // Output NPQK
    {1, 1},                      // Stride
    {1, 1}                       // Dilation
};
```

### Use Case 3: Mixed-Precision Training

```cpp
// Forward pass: FP16
// Backward pass: FP16
// Weight update: FP32

// Forward: activations (FP16) × weights (FP16) → output (FP16)
using ForwardGemm = /* FP16×FP16→FP16 GEMM */;

// Backward gradients: grad_output (FP16) × weights^T (FP16) → grad_input (FP16)
using BackwardGemm = /* FP16×FP16→FP16 GEMM */;

// Weight gradients: activations^T (FP16) × grad_output (FP16) → grad_weights (FP32)
using WeightGradGemm = /* FP16×FP16→FP32 GEMM */;

// Weight update: weights (FP32) -= lr * grad_weights (FP32)
// (done in separate kernel)
```

### Use Case 4: Quantized Inference (INT8)

```cpp
// INT8 activations × INT8 weights → INT32 accumulation → FP32 output

using ElementA = int8_t;
using ElementB = int8_t;
using ElementAccumulator = int32_t;
using ElementC = float;

// Scaling factors for dequantization
float scale_A = 0.01f;
float scale_B = 0.02f;

// Custom epilogue: dequantize and scale
struct DequantizeOp {
    float scale_A, scale_B;

    CUTLASS_DEVICE
    float operator()(int32_t accumulator) const {
        return float(accumulator) * scale_A * scale_B;
    }
};
```

---

## 9. Performance Troubleshooting {#performance-troubleshooting}

### Problem: Low Tensor Core Utilization

**Symptoms:** Nsight Compute shows <80% Tensor Core utilization

**Solutions:**
1. Increase tile size (try 256×128 or 256×256)
2. Check alignment (must be 8+ for FP16, 16+ for FP8)
3. Verify data types are Tensor Core-compatible
4. Ensure sufficient stages for pipeline (try StageCountAuto)

### Problem: High Shared Memory Bank Conflicts

**Symptoms:** Nsight shows many bank conflicts in shared memory accesses

**Solutions:**
1. CUTLASS handles swizzling automatically - but check if using custom layouts
2. Verify using TMA on SM90+ (no manual management needed)
3. For custom code, use CuTe's swizzled layouts

### Problem: Low Memory Bandwidth

**Symptoms:** Much lower than HBM theoretical peak

**Solutions:**
1. Check alignment and coalescing
2. Increase problem size (amortize setup costs)
3. Use batched GEMM to increase parallelism
4. Enable TMA for better bulk transfers (SM90+)

### Problem: Slow Compilation

**Symptoms:** CUTLASS kernels take minutes to compile

**Solutions:**
1. Compile in Release mode (`-O3`)
2. Reduce template instantiations (use fewer tile sizes)
3. Use precompiled CUTLASS library
4. Enable `ccache` for incremental builds
5. Compile specific kernels only (use `CUTLASS_LIBRARY_KERNELS`)

---

## 10. Tips and Tricks {#tips-and-tricks}

### Tip 1: Use Auto Modes

Let CUTLASS choose optimal settings:

```cpp
using StageCountType = cutlass::gemm::collective::StageCountAuto;
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
```

### Tip 2: Profile Before Optimizing

Always measure before assuming:

```bash
./cutlass_profiler --operation=gemm --m=M --n=N --k=K --top_k=3
```

### Tip 3: Check Examples First

CUTLASS has 84+ examples covering most use cases:

```bash
ls examples/
# 00_basic_gemm/              - Start here
# 48_hopper_warp_specialized/ - Hopper basics
# 54_hopper_fp8/              - FP8 precision
# 61_gemm_with_topk_softmax/  - Epilogue fusion
# 77_blackwell_fmha/          - Attention patterns
```

### Tip 4: Alignment Matters

```cpp
// Bad: misaligned
constexpr int AlignmentA = 8;
ElementA* ptr = (ElementA*)malloc(size);  // No alignment guarantee

// Good: aligned
cutlass::DeviceAllocation<ElementA> ptr(count);  // Always aligned
```

### Tip 5: Warm Up Before Timing

```cpp
// First kernel launch includes CUDA overhead
gemm_op.run();  // Warmup
cudaDeviceSynchronize();

// Now measure
auto start = std::chrono::high_resolution_clock::now();
gemm_op.run();
cudaDeviceSynchronize();
auto end = std::chrono::high_resolution_clock::now();
```

### Tip 6: Check CUDA Errors

```cpp
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        exit(1); \
    } \
}

CUDA_CHECK(cudaMalloc(&ptr, size));
CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
```

### Tip 7: Use Reference Implementations

For verification:

```cpp
#include "cutlass/util/reference/host/gemm.h"  // CPU reference
#include "cutlass/util/reference/device/gemm.h"  // GPU reference (cuBLAS-like)
```

### Tip 8: Read the Generated PTX

```bash
# Generate PTX to inspect
nvcc -ptx -arch=sm_90 my_gemm.cu -o my_gemm.ptx

# Look for:
# - wgmma instructions (Hopper)
# - ldmatrix instructions
# - bank conflicts in shared memory accesses
```

### Tip 9: Start Simple, Then Optimize

```cpp
// Step 1: Get correctness working with simple config
TileShape = Shape<_64, _64, _32>;
ClusterShape = Shape<_1, _1, _1>;
StageCount = 3;

// Step 2: Profile and identify bottleneck
// Step 3: Tune one parameter at a time
TileShape = Shape<_128, _128, _32>;  // Try larger tile

// Step 4: Use auto-tuning
TileShape = Shape<_128, _128, _32>;
StageCountType = StageCountAuto;     // Let CUTLASS decide
```

### Tip 10: Join the Community

- [CUTLASS GitHub Discussions](https://github.com/NVIDIA/cutlass/discussions)
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/accelerated-computing/cuda)
- Report bugs with minimal reproducible examples

---

## Conclusion

CUTLASS provides building blocks for high-performance GEMM across GPU architectures. Key takeaways:

1. **Start with examples** - 84+ examples cover most use cases
2. **Use Collective Builders** - Let CUTLASS choose optimal implementations
3. **Profile before optimizing** - Measure, don't guess
4. **Leverage automatic modes** - StageCountAuto, KernelScheduleAuto
5. **Verify correctness** - Use reference implementations
6. **Read the docs** - [CUTLASS Doxygen](https://nvidia.github.io/cutlass)

**Next Steps:**
- Explore [examples/](./examples/) directory
- Read [CUTLASS Deep Dive](CUTLASS_DEEP_DIVE.md) for architecture details
- Join the community for support

Happy hacking! 🚀

---

**References:**
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CUTLASS Documentation](./media/docs/cpp/quickstart.md)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute](https://developer.nvidia.com/nsight-compute)
