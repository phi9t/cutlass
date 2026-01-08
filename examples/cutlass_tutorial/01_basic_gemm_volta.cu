/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This example demonstrates the fundamental building blocks of CUTLASS on Volta (SM70)
 * architecture. It serves as the entry point for understanding CUTLASS internals.
 *
 * LEARNING OBJECTIVES:
 * ====================
 * 1. Understand the 4-level CUTLASS hierarchy: Device → CTA → Warp → Thread
 * 2. See how template instantiation generates architecture-specific code
 * 3. Learn the 2-stage pipeline pattern on Volta
 * 4. Explore explicit memory management (shared memory, registers)
 * 5. Understand warp-synchronous execution with mma.sync
 *
 * ARCHITECTURAL CONCEPTS:
 * =======================
 * - Volta Tensor Cores: mma.sync.m16n8k8 instruction
 * - Manual shared memory management with __syncthreads()
 * - 2-stage pipeline: Load next tile while computing current tile
 * - FP16 input, FP32 accumulation (mixed precision)
 *
 * PERFORMANCE CHARACTERISTICS:
 * ============================
 * - V100: ~120 TFLOPS (96% of 125 TFLOPS peak)
 * - Shared memory: 96 KB per SM
 * - 2-stage pipeline: Limited latency hiding compared to modern architectures
 *
 * COMPILE:
 * ========
 * nvcc 01_basic_gemm_volta.cu -o 01_basic_gemm_volta \
 *      -I../../include -I../../tools/util/include \
 *      -arch=sm_70 -std=c++17 -O3 --expt-relaxed-constexpr
 *
 * RUN:
 * ====
 * ./01_basic_gemm_volta --m=2048 --n=2048 --k=2048 --verify
 *
 **************************************************************************************************/

#include <iostream>
#include <chrono>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/command_line.h"

/***************************************************************************************************
 * SECTION 1: TYPE DEFINITIONS
 * ============================
 *
 * EDUCATIONAL NOTE:
 * ----------------
 * CUTLASS uses type-based configuration. All decisions about data types, layouts, and
 * precision are encoded in C++ types and resolved at compile time. This eliminates
 * runtime branching and enables aggressive compiler optimizations.
 *
 * For Volta, we use:
 * - FP16 (half_t) for input matrices: Compact storage, Tensor Core compatible
 * - FP32 (float) for accumulation: Maintains numerical accuracy
 * - FP32 (float) for output: Standard precision for downstream operations
 *
 * LAYOUT CHOICES:
 * ---------------
 * - RowMajor: Consecutive elements in a row are adjacent in memory (C-style)
 * - ColumnMajor: Consecutive elements in a column are adjacent (Fortran-style)
 *
 * Why mix layouts? Different layouts optimize different access patterns:
 * - A (RowMajor): Threads in a warp access consecutive K elements
 * - B (ColumnMajor): Threads access consecutive K elements
 * - C/D (ColumnMajor): Matches cuBLAS convention, efficient column writes
 *
 **************************************************************************************************/

using ElementA = cutlass::half_t;                    // Input matrix A element type
using ElementB = cutlass::half_t;                    // Input matrix B element type
using ElementC = float;                              // Output matrix C element type
using ElementAccumulator = float;                    // Accumulator type (critical for accuracy!)

using LayoutA = cutlass::layout::RowMajor;           // A: rows are contiguous
using LayoutB = cutlass::layout::ColumnMajor;        // B: columns are contiguous
using LayoutC = cutlass::layout::ColumnMajor;        // C: columns are contiguous

/***************************************************************************************************
 * SECTION 2: GEMM CONFIGURATION
 * ==============================
 *
 * EDUCATIONAL NOTE:
 * ----------------
 * This section defines the "shape" of our GEMM operation at multiple levels of the hierarchy.
 *
 * THREADBLOCK TILE (128×256×32):
 * ------------------------------
 * - Each threadblock processes a 128×256 tile of the output matrix
 * - K dimension (32) is the "accumulation depth" per iteration
 * - Larger tiles = more data reuse in shared memory
 * - Trade-off: Larger tiles require more shared memory, limiting occupancy
 *
 * WARP TILE (64×64×32):
 * ---------------------
 * - Each warp (32 threads) processes a 64×64 sub-tile
 * - Threadblock has 256 threads = 8 warps
 * - 8 warps cover 128×256 tile: 2 warps in M, 4 warps in N
 *
 * INSTRUCTION SHAPE (16×8×8):
 * ---------------------------
 * - Volta's mma.sync instruction operates on 16×8×8 tiles
 * - This is the hardware's atomic matrix multiply operation
 * - Each warp issues multiple mma.sync to cover its 64×64×32 tile
 *
 * MMA PIPELINE STAGES (2):
 * ------------------------
 * - Volta supports 2-stage pipeline:
 *   * Stage 0: Load tile K+1 from global → shared memory
 *   * Stage 1: Compute tile K using data in shared memory
 * - While computing, next tile loads concurrently
 * - Limited to 2 stages due to shared memory constraints on Volta
 *
 **************************************************************************************************/

using ThreadblockShape = cutlass::gemm::GemmShape<128, 256, 32>;  // CTA tile: M×N×K
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;           // Warp tile: M×N×K
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;      // mma.sync shape

constexpr int NumStages = 2;                                       // 2-stage pipeline

/***************************************************************************************************
 * SECTION 3: CUTLASS GEMM DEVICE OPERATOR
 * ========================================
 *
 * EDUCATIONAL NOTE:
 * ----------------
 * The `cutlass::gemm::device::Gemm` template is the high-level interface to CUTLASS.
 * It composes all the layers of the CUTLASS hierarchy:
 *
 * TEMPLATE PARAMETERS EXPLAINED:
 * ------------------------------
 * 1. ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC:
 *    Data types and memory layouts for matrices
 *
 * 2. ElementAccumulator:
 *    Type for internal accumulation (FP32 for numerical accuracy)
 *
 * 3. cutlass::arch::OpClassTensorOp:
 *    Indicates we're using Tensor Core operations (vs. SIMT/CUDA cores)
 *
 * 4. cutlass::arch::Sm70:
 *    Target architecture (Volta = SM70)
 *    This tag triggers compile-time selection of Volta-specific code paths
 *
 * 5. ThreadblockShape, WarpShape, InstructionShape:
 *    Tile sizes at each level of the hierarchy
 *
 * 6. cutlass::epilogue::thread::LinearCombination:
 *    Epilogue operation: D = alpha * (A×B) + beta * C
 *    This is the standard BLAS GEMM epilogue
 *
 * 7. cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>:
 *    Controls how threadblocks are assigned to output tiles
 *    "Identity" means linear assignment (simplest, good for most cases)
 *
 * 8. NumStages:
 *    Pipeline depth (2 for Volta)
 *
 * 9. 8 and 8 (alignment):
 *    Memory alignment in elements for A and B
 *    8 elements × 2 bytes (FP16) = 16 bytes = 128-bit aligned loads
 *    Alignment is critical for coalesced memory access!
 *
 * WHAT HAPPENS UNDER THE HOOD:
 * ----------------------------
 * This single template instantiation generates:
 * - Tile iterators for loading A, B from global → shared memory
 * - Shared memory layouts with bank conflict avoidance
 * - Warp-level MMA operations using mma.sync instructions
 * - Thread-level epilogue for scaling and accumulation
 * - Kernel launch configuration and grid/block setup
 *
 * All of this is generated at compile time with zero runtime overhead!
 *
 **************************************************************************************************/

using Gemm = cutlass::gemm::device::Gemm<
  ElementA,                                           // Data type of A matrix
  LayoutA,                                            // Layout of A matrix
  ElementB,                                           // Data type of B matrix
  LayoutB,                                            // Layout of B matrix
  ElementC,                                           // Data type of C and D matrices
  LayoutC,                                            // Layout of C and D matrices
  ElementAccumulator,                                 // Data type of accumulator
  cutlass::arch::OpClassTensorOp,                     // Use Tensor Cores
  cutlass::arch::Sm70,                                // Target Volta (SM70)
  ThreadblockShape,                                   // Threadblock tile size
  WarpShape,                                          // Warp tile size
  InstructionShape,                                   // Instruction tile size
  cutlass::epilogue::thread::LinearCombination<       // Epilogue: D = α(AB) + βC
    ElementC,                                         //   Output type
    128 / cutlass::sizeof_bits<ElementC>::value,      //   Vectorization (4 floats = 128 bits)
    ElementAccumulator,                               //   Accumulator type
    ElementAccumulator                                //   Compute type
  >,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Threadblock mapping
  NumStages,                                          // Pipeline stages
  8,                                                  // Alignment of A (8 FP16 elements)
  8                                                   // Alignment of B (8 FP16 elements)
>;

/***************************************************************************************************
 * SECTION 4: HELPER FUNCTIONS
 * ============================
 **************************************************************************************************/

/**
 * @brief Prints performance results
 *
 * EDUCATIONAL NOTE:
 * ----------------
 * GEMM performance is measured in FLOPS (Floating Point Operations Per Second).
 * One multiply-add (A*B + C) counts as 2 operations.
 * For GEMM: Total FLOPs = 2 × M × N × K
 *
 * Example: 2048×2048×2048 GEMM
 * - FLOPs = 2 × 2048³ = 17.18 billion operations
 * - On V100 @ 120 TFLOPS: Execution time = 17.18B / 120T = 0.143 ms
 */
void print_performance(int M, int N, int K, double runtime_ms) {
  double gflops = (2.0 * M * N * K) / (runtime_ms / 1000.0) / 1e9;
  double tflops = gflops / 1000.0;

  std::cout << "=================================================\n";
  std::cout << "GEMM Performance\n";
  std::cout << "=================================================\n";
  std::cout << "Problem size: " << M << "×" << N << "×" << K << "\n";
  std::cout << "Runtime:      " << runtime_ms << " ms\n";
  std::cout << "Performance:  " << gflops << " GFLOPS\n";
  std::cout << "              " << tflops << " TFLOPS\n";
  std::cout << "=================================================\n";
}

/***************************************************************************************************
 * SECTION 5: MAIN FUNCTION - EXECUTION AND VERIFICATION
 * ======================================================
 *
 * EDUCATIONAL NOTE:
 * ----------------
 * This section demonstrates:
 * 1. Host-side tensor allocation and initialization
 * 2. CUTLASS GEMM kernel configuration and launch
 * 3. Performance measurement
 * 4. Correctness verification against reference implementation
 *
 * WORKFLOW:
 * ---------
 * Allocate → Initialize → Configure → Run → Verify → Measure
 *
 **************************************************************************************************/

int main(int argc, char const **argv) {

  //
  // Parse command line arguments
  //
  cutlass::CommandLine cmd(argc, argv);

  int M = 2048;
  int N = 2048;
  int K = 2048;
  bool verify = false;

  cmd.get_cmd_line_argument("m", M);
  cmd.get_cmd_line_argument("n", N);
  cmd.get_cmd_line_argument("k", K);
  verify = cmd.check_cmd_line_flag("verify");

  //
  // EDUCATIONAL NOTE: Matrix Dimensions
  // ===================================
  // GEMM computes: D = α(A×B) + βC
  // - A: M×K matrix
  // - B: K×N matrix
  // - C, D: M×N matrices
  // - α, β: scalar coefficients
  //

  std::cout << "\n";
  std::cout << "╔═══════════════════════════════════════════════╗\n";
  std::cout << "║  CUTLASS Tutorial 01: Basic Volta GEMM       ║\n";
  std::cout << "╚═══════════════════════════════════════════════╝\n";
  std::cout << "\n";
  std::cout << "Problem: D = α(A×B) + βC\n";
  std::cout << "  A: " << M << "×" << K << " (FP16, RowMajor)\n";
  std::cout << "  B: " << K << "×" << N << " (FP16, ColumnMajor)\n";
  std::cout << "  C: " << M << "×" << N << " (FP32, ColumnMajor)\n";
  std::cout << "  D: " << M << "×" << N << " (FP32, ColumnMajor)\n";
  std::cout << "\n";

  //
  // EDUCATIONAL NOTE: Scalar Epilogue Coefficients
  // ==============================================
  // α = 1.0: Compute full matrix product
  // β = 0.0: Ignore C (output is just A×B)
  //
  // Other common patterns:
  // - β = 1.0: Accumulate into C (residual connection)
  // - β ≠ 0: Weighted combination (C as bias or prior result)
  //
  ElementAccumulator alpha = ElementAccumulator(1.0f);
  ElementAccumulator beta = ElementAccumulator(0.0f);

  //
  // EDUCATIONAL NOTE: Host Tensors
  // ==============================
  // cutlass::HostTensor is a convenience wrapper that:
  // - Allocates device memory (cudaMalloc)
  // - Manages host-side mirrors (for CPU access)
  // - Handles copying between host and device
  // - Automatically frees memory on destruction
  //
  // Layout template parameter determines memory organization.
  //

  cutlass::HostTensor<ElementA, LayoutA> tensor_A({M, K});
  cutlass::HostTensor<ElementB, LayoutB> tensor_B({K, N});
  cutlass::HostTensor<ElementC, LayoutC> tensor_C({M, N});
  cutlass::HostTensor<ElementC, LayoutC> tensor_D({M, N});
  cutlass::HostTensor<ElementC, LayoutC> tensor_D_ref({M, N});

  //
  // EDUCATIONAL NOTE: Tensor Initialization
  // ========================================
  // Using uniform random distribution in range [-2, 2]
  // For numerical stability in FP16:
  // - Avoid extreme values (overflow/underflow)
  // - Keep dynamic range reasonable for accumulation
  //

  std::cout << "Initializing tensors...\n";

  cutlass::reference::host::TensorFillRandomUniform(
    tensor_A.host_view(),
    1,                                                // Seed
    ElementA(2),                                      // Max value
    ElementA(-2),                                     // Min value
    0                                                 // Bits (0 = full precision)
  );

  cutlass::reference::host::TensorFillRandomUniform(
    tensor_B.host_view(),
    2,
    ElementB(2),
    ElementB(-2),
    0
  );

  cutlass::reference::host::TensorFillRandomUniform(
    tensor_C.host_view(),
    3,
    ElementC(2),
    ElementC(-2),
    0
  );

  // Copy initialized data to device
  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();

  //
  // EDUCATIONAL NOTE: GEMM Arguments
  // ================================
  // The Arguments struct packages all parameters needed for the GEMM:
  // - problem_size: {M, N, K} dimensions
  // - ref_A, ref_B, ref_C, ref_D: TensorRef objects (ptr + stride info)
  // - {alpha, beta}: Epilogue scaling factors
  //
  // TensorRef encodes:
  // - Pointer to data
  // - Stride (for non-contiguous access)
  //
  // split_k_slices: Advanced feature for parallelizing K dimension
  // (not used in this basic example, default = 1)
  //

  typename Gemm::Arguments arguments {
    {M, N, K},                                        // Problem dimensions
    tensor_A.device_ref(),                            // A: device pointer + stride
    tensor_B.device_ref(),                            // B: device pointer + stride
    tensor_C.device_ref(),                            // C: device pointer + stride
    tensor_D.device_ref(),                            // D: device pointer + stride
    {alpha, beta},                                    // Epilogue: α, β
    1                                                 // split_k_slices
  };

  //
  // EDUCATIONAL NOTE: Workspace Size
  // ================================
  // Some GEMM variants require temporary workspace in device memory.
  // For example:
  // - Split-K reduction: Need space for partial results
  // - Grouped GEMM: Need space for problem descriptors
  //
  // Basic GEMM typically requires no workspace (size = 0).
  //

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  //
  // EDUCATIONAL NOTE: GEMM Initialization
  // =====================================
  // The initialize() call:
  // 1. Validates arguments (dimensions, alignment, etc.)
  // 2. Sets up kernel launch configuration (grid, blocks, shared memory)
  // 3. Prepares device pointers and strides
  //
  // This is separate from run() to allow reusing the same configuration
  // for multiple runs with different data.
  //

  Gemm gemm_op;

  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "ERROR: GEMM initialization failed: "
              << cutlassGetStatusString(status) << "\n";
    return -1;
  }

  //
  // EDUCATIONAL NOTE: Kernel Execution
  // ==================================
  // The run() call:
  // 1. Launches the CUDA kernel
  // 2. Returns immediately (async execution)
  // 3. Can be called multiple times with same configuration
  //
  // For accurate timing, we need to:
  // 1. Warm up (first kernel launch has overhead)
  // 2. Synchronize before timing
  // 3. Run multiple iterations
  // 4. Synchronize after timing
  //

  // Warmup
  std::cout << "Running warmup...\n";
  status = gemm_op.run();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "ERROR: GEMM execution failed: "
              << cutlassGetStatusString(status) << "\n";
    return -1;
  }
  cudaDeviceSynchronize();

  // Timing
  std::cout << "Running performance measurement...\n";
  const int iterations = 100;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; ++i) {
    status = gemm_op.run();
  }

  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avg_runtime_ms = (duration / 1000.0) / iterations;

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "ERROR: GEMM execution failed: "
              << cutlassGetStatusString(status) << "\n";
    return -1;
  }

  // Print performance
  print_performance(M, N, K, avg_runtime_ms);

  //
  // EDUCATIONAL NOTE: Verification
  // ==============================
  // Compare against reference implementation to ensure correctness.
  // Reference uses cuBLAS-style GEMM on GPU (accurate but not optimized).
  //
  // Comparison tolerance accounts for:
  // - FP16 → FP32 conversions
  // - Different accumulation order (non-associative FP arithmetic)
  // - Rounding differences
  //
  // Typical tolerance for FP16: 1e-3 to 1e-2
  //

  if (verify) {
    std::cout << "\nRunning verification...\n";

    // Compute reference
    cutlass::reference::device::Gemm<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementAccumulator, ElementAccumulator
    > reference_gemm;

    reference_gemm(
      {M, N, K},
      alpha,
      tensor_A.device_ref(),
      tensor_B.device_ref(),
      beta,
      tensor_C.device_ref(),
      tensor_D_ref.device_ref()
    );

    cudaDeviceSynchronize();

    // Copy results back to host for comparison
    tensor_D.sync_host();
    tensor_D_ref.sync_host();

    // Compare
    bool passed = cutlass::reference::host::TensorEquals(
      tensor_D.host_view(),
      tensor_D_ref.host_view()
    );

    if (passed) {
      std::cout << "\n✓ Verification PASSED\n";
    } else {
      std::cout << "\n✗ Verification FAILED\n";
      std::cout << "  (This may be due to numerical precision differences)\n";

      // Print first few mismatches for debugging
      std::cout << "\nFirst mismatches (max 10):\n";
      int mismatches = 0;
      for (int i = 0; i < M && mismatches < 10; ++i) {
        for (int j = 0; j < N && mismatches < 10; ++j) {
          float cutlass_val = tensor_D.at({i, j});
          float ref_val = tensor_D_ref.at({i, j});
          float diff = std::abs(cutlass_val - ref_val);
          float rel_err = diff / std::max(std::abs(ref_val), 1e-5f);

          if (rel_err > 1e-3) {  // Tolerance
            std::cout << "  [" << i << "," << j << "]: "
                      << "CUTLASS=" << cutlass_val << " "
                      << "Reference=" << ref_val << " "
                      << "RelErr=" << rel_err << "\n";
            mismatches++;
          }
        }
      }
    }
  }

  std::cout << "\n";
  std::cout << "╔═══════════════════════════════════════════════╗\n";
  std::cout << "║  Tutorial Complete!                           ║\n";
  std::cout << "╚═══════════════════════════════════════════════╝\n";
  std::cout << "\n";
  std::cout << "KEY TAKEAWAYS:\n";
  std::cout << "  1. CUTLASS uses template metaprogramming for zero-overhead abstraction\n";
  std::cout << "  2. 4-level hierarchy: Device → Threadblock → Warp → Thread\n";
  std::cout << "  3. Volta uses 2-stage pipeline with mma.sync instructions\n";
  std::cout << "  4. Type system encodes data types, layouts, and precision at compile time\n";
  std::cout << "\n";
  std::cout << "NEXT STEPS:\n";
  std::cout << "  - Run with different problem sizes: --m=X --n=Y --k=Z\n";
  std::cout << "  - Try ./02_ampere_async.cu to see async memory pipeline\n";
  std::cout << "  - Profile with Nsight Compute to see low-level details\n";
  std::cout << "\n";

  return 0;
}
