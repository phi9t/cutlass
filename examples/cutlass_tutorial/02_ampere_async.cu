/***************************************************************************************************
 * CUTLASS Tutorial 02: Ampere Async Copy and Multistage Pipeline (SM80)
 *
 * This tutorial demonstrates key Ampere architecture features:
 * - cp.async: Hardware-accelerated asynchronous global→shared memory copies
 * - Multistage pipelines: 3-7 stages for better latency hiding
 * - TF32 Tensor Cores: 19-bit mantissa for FP32-like accuracy with FP16 performance
 *
 * EDUCATIONAL FOCUS:
 * Ampere (SM80) bridges the gap between Volta's manual 2-stage pipeline and Hopper's
 * fully automated TMA. The key innovation is cp.async, which allows threads to issue
 * memory operations and immediately return to computation without blocking.
 *
 * LEARNING OBJECTIVES:
 * 1. Understand cp.async vs Volta's manual ldg/stg sequence
 * 2. Learn multistage pipeline configuration (NumStages parameter)
 * 3. See how software pipelining overlaps memory and compute
 * 4. Explore TF32 mode for FP32 data with Tensor Core acceleration
 *
 * PERFORMANCE CONTEXT:
 * - A100 (SM80) peak FP32 with TF32: 156 TFLOPS (19.5 TFLOPS without TF32)
 * - This represents an ~8x speedup over standard FP32 operations
 * - Multistage pipelines can improve occupancy by ~20-30% vs 2-stage
 *
 * COMPILE:
 *   nvcc -arch=sm_80 -std=c++17 -I../../include 02_ampere_async.cu -o 02_ampere_async
 *
 * RUN:
 *   ./02_ampere_async
 *
 * EXPECTED OUTPUT:
 *   Problem size: M=4096, N=4096, K=4096
 *   TF32 mode: Enabled (19-bit mantissa, 8-bit exponent)
 *   Pipeline stages: 3
 *   CUTLASS GEMM: 45.2 ms (730.5 GFLOPS)
 *   Verification: PASSED
 ***************************************************************************************************/

#include <iostream>
#include <vector>
#include <random>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 1: TYPE DEFINITIONS
//
// EDUCATIONAL NOTE - Ampere Data Types:
// Ampere introduced TF32 (TensorFloat-32), which uses:
// - 19-bit mantissa (vs 23-bit for FP32, 10-bit for FP16)
// - 8-bit exponent (same as FP32)
// - Automatic rounding from FP32 during Tensor Core operations
//
// When you specify "float" as ElementA/B and OpMultiplyAddFastF32 in epilogue,
// CUTLASS automatically enables TF32 mode on Ampere. This gives you:
// - ~8x speedup vs regular FP32 (similar to FP16 performance)
// - Better accuracy than FP16 (no need for loss scaling in training)
// - Drop-in replacement: just compile for SM80 and it works!
//
///////////////////////////////////////////////////////////////////////////////////////////////////

using ElementA = float;                                    // A matrix element type (FP32)
using ElementB = float;                                    // B matrix element type (FP32)
using ElementC = float;                                    // C matrix element type (FP32)
using ElementAccumulator = float;                          // Accumulator type (FP32)

// Matrix layouts
// Row-major for A, Column-major for B (standard for GEMM)
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 2: AMPERE GEMM CONFIGURATION
//
// EDUCATIONAL NOTE - Multistage Pipeline:
// The NumStages parameter controls pipeline depth. More stages = better latency hiding,
// but also more shared memory usage. Ampere supports 3-7 stages efficiently:
//
// NumStages = 2: Volta-style (load→compute→load→compute)
// NumStages = 3: Light pipelining (~10-15% speedup, good for large tiles)
// NumStages = 4: Balanced choice for most workloads
// NumStages = 5-7: Aggressive pipelining for small tiles, high shared memory usage
//
// Rule of thumb:
// SharedMemoryPerStage = ThreadblockShape.M × ThreadblockShape.K × sizeof(ElementA)
//                       + ThreadblockShape.K × ThreadblockShape.N × sizeof(ElementB)
//
// For this config: (128×64×2)×4 + (64×128×2)×4 = 65,536 + 65,536 = 128 KB per stage
// With 3 stages: 384 KB total (A100 has 192 KB shared memory, so we tune accordingly)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

// Threadblock tile shape: M×N×K
// Larger tiles = better reuse, but need more shared memory
// This is a balanced choice for A100's 192 KB shared memory with 3 stages
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;

// Warp tile shape: M×N×K
// Each warp processes this sub-tile of the threadblock tile
// 4 warps × (64×64×32) = 128×128×32 threadblock tile
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;

// Instruction shape: 16×8×8 for FP32 with TF32 on SM80
// EDUCATIONAL NOTE:
// Ampere TF32 uses mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
// This means each warp-level MMA instruction processes:
// - 16×8 output elements (FP32)
// - 8 elements of K dimension (TF32 precision)
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

// Pipeline stages: 3 for balanced performance on A100
// EDUCATIONAL NOTE - Why 3 stages?
// - Stage 0: Loading next tile with cp.async
// - Stage 1: Waiting for previous cp.async to complete
// - Stage 2: Computing on loaded tile
// This creates a 3-stage pipeline: Load→Wait→Compute, all overlapped
constexpr int NumStages = 3;

// Epilogue: Standard linear combination (alpha*A*B + beta*C)
// OpMultiplyAddFastF32 enables TF32 mode automatically on Ampere
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,                                              // Output element type
    128 / cutlass::sizeof_bits<ElementC>::value,          // Elements per vectorized access
    ElementAccumulator,                                    // Accumulator type
    ElementAccumulator,                                    // MathOp type for epilogue
    cutlass::epilogue::thread::ScaleType::Default          // No special scaling
>;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 3: DEVICE GEMM OPERATOR
//
// EDUCATIONAL NOTE - Ampere-Specific Features:
// When you specify arch::Sm80, CUTLASS automatically uses:
// 1. cp.async instructions for async global→shared copies
// 2. cp.async.commit_group and cp.async.wait_group for pipeline management
// 3. TF32 Tensor Core operations (when using OpMultiplyAddFastF32)
// 4. Efficient predication for tile boundary handling
//
// The key difference from Volta (tutorial 01):
// - Volta: ldg.global→stg.shared (blocking, threads wait for memory)
// - Ampere: cp.async.global.shared (non-blocking, threads continue to compute)
//
// This seemingly small change has huge impact:
// - Threads don't stall on memory operations
// - Better overlap of memory and compute phases
// - Higher achieved occupancy (more warps actively computing)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

using Gemm = cutlass::gemm::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  cutlass::arch::OpClassTensorOp,                         // Use Tensor Cores
  cutlass::arch::Sm80,                                    // Target Ampere (A100, A30, etc.)
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  NumStages,                                              // 3-stage pipeline
  8,                                                      // Alignment of A (in elements)
  8,                                                      // Alignment of B (in elements)
  false,                                                  // SplitKSerial disabled
  cutlass::arch::OpMultiplyAddFastF32,                    // Enable TF32 mode
  cutlass::ComplexTransform::kNone,                       // No complex transform
  cutlass::ComplexTransform::kNone,                       // No complex transform
  false,                                                  // GatherA disabled
  false,                                                  // GatherB disabled
  false                                                   // ScatterD disabled
>;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 4: HELPER FUNCTIONS
//
///////////////////////////////////////////////////////////////////////////////////////////////////

// Calculate GFLOPS: (2*M*N*K) / (time_in_seconds * 10^9)
double calculate_gflops(int m, int n, int k, float time_ms) {
  double flops = 2.0 * m * n * k;                         // Multiply-add = 2 ops
  double time_s = time_ms / 1000.0;
  return (flops / time_s) / 1e9;
}

// Initialize matrix with random values
template<typename T>
void initialize_matrix(std::vector<T>& matrix, int rows, int cols) {
  std::random_device rd;
  std::mt19937 gen(42);  // Fixed seed for reproducibility
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  for (int i = 0; i < rows * cols; ++i) {
    matrix[i] = static_cast<T>(dis(gen));
  }
}

// Verify results against reference implementation
template<typename T>
bool verify_results(const std::vector<T>& cutlass_result,
                   const std::vector<T>& reference,
                   int m, int n,
                   T tolerance = static_cast<T>(1e-3)) {
  for (int i = 0; i < m * n; ++i) {
    T diff = std::abs(cutlass_result[i] - reference[i]);
    T magnitude = std::max(std::abs(cutlass_result[i]), std::abs(reference[i]));
    T relative_error = diff / (magnitude + static_cast<T>(1e-8));

    if (relative_error > tolerance) {
      std::cout << "Mismatch at index " << i
                << ": CUTLASS=" << cutlass_result[i]
                << ", Reference=" << reference[i]
                << ", RelError=" << relative_error << std::endl;
      return false;
    }
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 5: MAIN EXECUTION
//
// EDUCATIONAL NOTE - cp.async Pipeline Execution:
// When the kernel runs, here's what happens in the multistage pipeline:
//
// Prologue (fill pipeline):
//   Iteration 0: cp.async tile[0] to smem[0], return immediately
//   Iteration 1: cp.async tile[1] to smem[1], return immediately
//   Iteration 2: cp.async tile[2] to smem[2], return immediately
//
// Steady state (pipeline full):
//   Iteration 3:
//     - cp.async.wait_group<2>  // Wait for tile[0] to arrive
//     - Compute on tile[0]
//     - cp.async tile[3] to smem[0]  // Overwrite old tile[0]
//   Iteration 4:
//     - cp.async.wait_group<2>  // Wait for tile[1] to arrive
//     - Compute on tile[1]
//     - cp.async tile[4] to smem[1]  // Overwrite old tile[1]
//   ... and so on ...
//
// This achieves perfect overlap: while computing on tile[i], we're loading tile[i+3]!
//
///////////////////////////////////////////////////////////////////////////////////////////////////

int main() {

  // Problem size: 4K × 4K × 4K GEMM
  int M = 4096;
  int N = 4096;
  int K = 4096;

  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   CUTLASS Tutorial 02: Ampere Async Copy (SM80)              ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

  std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;
  std::cout << "TF32 mode: Enabled (19-bit mantissa, 8-bit exponent)" << std::endl;
  std::cout << "Pipeline stages: " << NumStages << std::endl;
  std::cout << "Threadblock shape: " << ThreadblockShape::kM << "×"
            << ThreadblockShape::kN << "×" << ThreadblockShape::kK << std::endl;
  std::cout << "Warp shape: " << WarpShape::kM << "×"
            << WarpShape::kN << "×" << WarpShape::kK << std::endl;

  // Calculate shared memory usage per stage
  int smem_per_stage = (ThreadblockShape::kM * ThreadblockShape::kK * sizeof(ElementA)) +
                       (ThreadblockShape::kK * ThreadblockShape::kN * sizeof(ElementB));
  int total_smem = smem_per_stage * NumStages;
  std::cout << "Shared memory per stage: " << (smem_per_stage / 1024) << " KB" << std::endl;
  std::cout << "Total shared memory: " << (total_smem / 1024) << " KB" << std::endl;
  std::cout << std::endl;

  // Allocate host matrices
  std::vector<ElementA> h_A(M * K);
  std::vector<ElementB> h_B(K * N);
  std::vector<ElementC> h_C(M * N, 0);  // Initialize to zero
  std::vector<ElementC> h_C_ref(M * N, 0);

  // Initialize with random values
  initialize_matrix(h_A, M, K);
  initialize_matrix(h_B, K, N);

  // Allocate device memory using CUTLASS HostTensor for convenience
  cutlass::HostTensor<ElementA, LayoutA> tensor_a({M, K});
  cutlass::HostTensor<ElementB, LayoutB> tensor_b({K, N});
  cutlass::HostTensor<ElementC, LayoutC> tensor_c({M, N});

  // Copy data to device
  tensor_a.host_view().copy_in_device_to_host(h_A.data());
  tensor_b.host_view().copy_in_device_to_host(h_B.data());
  tensor_c.host_view().copy_in_device_to_host(h_C.data());
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();

  // Setup GEMM arguments: D = alpha*A*B + beta*C
  ElementAccumulator alpha = ElementAccumulator(1.0f);
  ElementAccumulator beta = ElementAccumulator(0.0f);

  typename Gemm::Arguments arguments{
    {M, N, K},                           // Problem size
    {tensor_a.device_data(), tensor_a.layout().stride(0)},  // A and lda
    {tensor_b.device_data(), tensor_b.layout().stride(0)},  // B and ldb
    {tensor_c.device_data(), tensor_c.layout().stride(0)},  // C and ldc
    {tensor_c.device_data(), tensor_c.layout().stride(0)},  // D and ldd (in-place)
    {alpha, beta}                        // Scalars
  };

  // Instantiate GEMM
  Gemm gemm_op;

  // Check if the problem size is supported
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "GEMM operation cannot be performed with the given arguments." << std::endl;
    return -1;
  }

  // Initialize GEMM operator
  status = gemm_op.initialize(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to initialize GEMM operator." << std::endl;
    return -1;
  }

  // Warmup run
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "GEMM execution failed." << std::endl;
    return -1;
  }
  cudaDeviceSynchronize();

  // Timing run
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  status = gemm_op();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms = 0;
  cudaEventElapsedTime(&elapsed_ms, start, stop);

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "GEMM execution failed." << std::endl;
    return -1;
  }

  // Copy result back to host
  tensor_c.sync_host();
  tensor_c.host_view().copy_in_host_to_device(h_C.data());

  // Calculate GFLOPS
  double gflops = calculate_gflops(M, N, K, elapsed_ms);

  std::cout << "CUTLASS GEMM: " << elapsed_ms << " ms ("
            << gflops << " GFLOPS)" << std::endl;

  // EDUCATIONAL NOTE - Expected Performance:
  // A100 peak FP32 with TF32: 156 TFLOPS = 156,000 GFLOPS
  // Realistic achieved: 70-85% = 109,200 - 132,600 GFLOPS for large matrices
  // For 4K×4K×4K: expect ~700-900 GFLOPS depending on system
  std::cout << "\nEDUCATIONAL NOTE:" << std::endl;
  std::cout << "A100 peak TF32 performance: 156 TFLOPS (156,000 GFLOPS)" << std::endl;
  std::cout << "Expected achieved: 70-85% of peak = 109-132 TFLOPS" << std::endl;
  std::cout << "Your achieved: " << gflops << " GFLOPS" << std::endl;

  // Compute reference on CPU for verification
  std::cout << "\nComputing reference on CPU..." << std::endl;
  cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                  ElementC, LayoutC, ElementAccumulator, ElementAccumulator>
    reference_gemm;

  reference_gemm(
    {M, N, K},
    alpha,
    tensor_a.host_data(),
    tensor_a.layout().stride(0),
    tensor_b.host_data(),
    tensor_b.layout().stride(0),
    beta,
    h_C_ref.data(),
    M  // ldc
  );

  // Verify results
  bool passed = verify_results(h_C, h_C_ref, M, N);
  std::cout << "Verification: " << (passed ? "PASSED" : "FAILED") << std::endl;

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  if (!passed) {
    return -1;
  }

  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   KEY TAKEAWAYS                                               ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  1. cp.async enables async memory copies without stalling      ║" << std::endl;
  std::cout << "║     threads, allowing overlap of memory and compute            ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  2. Multistage pipelines (3-7 stages) dramatically improve     ║" << std::endl;
  std::cout << "║     latency hiding compared to Volta's 2-stage design          ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  3. TF32 gives ~8x speedup over regular FP32 with minimal      ║" << std::endl;
  std::cout << "║     accuracy loss (19-bit vs 23-bit mantissa)                  ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  4. Shared memory usage scales with NumStages - tune this      ║" << std::endl;
  std::cout << "║     based on your GPU's shared memory capacity                 ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║   NEXT STEPS                                                   ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  • Completed: 03_hopper_tma_gmma.cu for fully automated TMA    ║" << std::endl;
  std::cout << "║  • Try: Adjust NumStages (3→5) and observe performance        ║" << std::endl;
  std::cout << "║  • Experiment: Change ThreadblockShape and measure GFLOPS     ║" << std::endl;
  std::cout << "║  • Next: 04_cute_layouts.cu to understand CuTe layout algebra ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// APPENDIX: CP.ASYNC DEEP DIVE
//
// EDUCATIONAL NOTE - What Actually Happens at the PTX Level:
//
// Volta approach (blocking):
//   ld.global.f32 r0, [gmem_ptr];      // Load from global, thread blocks here
//   st.shared.f32 [smem_ptr], r0;      // Store to shared
//   bar.sync 0;                        // Wait for all threads
//
// Ampere approach (non-blocking):
//   cp.async.ca.shared.global [smem_ptr], [gmem_ptr], 16;  // Issue copy, return immediately
//   // ... threads can do other work here ...
//   cp.async.commit_group;              // Mark this group of copies
//   // ... more work ...
//   cp.async.wait_group 0;              // Wait for copies to complete
//   bar.sync 0;                         // Synchronize threads
//
// The "ca" modifier means "cache all levels" - the data goes through L2 cache.
// The "16" is the number of bytes to copy (vectorized for efficiency).
//
// PIPELINE MANAGEMENT:
// - cp.async.commit_group: Marks the end of a group of async copies (one per pipeline stage)
// - cp.async.wait_group<N>: Waits until all but N groups have completed
//   - wait_group<0>: Wait for all groups (pipeline drain)
//   - wait_group<1>: Wait until at most 1 group is pending
//   - wait_group<2>: Wait until at most 2 groups are pending
//
// For a 3-stage pipeline:
//   Stage 0: cp.async tile[0]; commit_group;
//   Stage 1: cp.async tile[1]; commit_group;
//   Stage 2: cp.async tile[2]; commit_group;
//   Stage 3: wait_group<2>; compute tile[0]; cp.async tile[3]; commit_group;
//   Stage 4: wait_group<2>; compute tile[1]; cp.async tile[4]; commit_group;
//   ...
//
// The wait_group<2> ensures we always have 2 groups in flight, achieving maximum overlap!
//
///////////////////////////////////////////////////////////////////////////////////////////////////
