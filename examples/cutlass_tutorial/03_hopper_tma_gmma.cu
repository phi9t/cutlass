/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This example demonstrates modern Hopper (SM90) architecture features including TMA
 * (Tensor Memory Accelerator) and GMMA (warpgroup matrix multiply-accumulate).
 *
 * LEARNING OBJECTIVES:
 * ====================
 * 1. Understand TMA: Hardware-accelerated bulk tensor copies
 * 2. Learn GMMA: Warpgroup-level matrix operations (4 warps collaborate)
 * 3. Explore warp specialization: Producer vs. Consumer roles
 * 4. See thread block clusters in action
 * 5. Understand automatic pipeline stage calculation
 *
 * ARCHITECTURAL BREAKTHROUGHS:
 * ============================
 * Hopper introduces several game-changing features:
 *
 * TMA (Tensor Memory Accelerator):
 * - Dedicated hardware unit for bulk memory transfers
 * - Single thread can initiate multi-KB copy
 * - Automatic swizzling for bank conflict avoidance
 * - Implicit boundary checking (no thread divergence!)
 * - 2D/3D/4D tensor support with arbitrary strides
 *
 * GMMA (wgmma.mma_async):
 * - Operates on entire warpgroup (128 threads = 4 warps)
 * - Higher throughput than Ampere's mma.sync
 * - Direct shared memory operands (no LDMATRIX needed)
 * - Asynchronous execution with barrier-based completion
 *
 * Warp Specialization:
 * - Producer warps: Only do TMA loads
 * - Consumer warps: Only do GMMA compute
 * - Overlapped execution → ~16% speedup vs. uniform work
 *
 * Thread Block Clusters:
 * - Multiple CTAs can cooperate
 * - Share data via distributed shared memory
 * - Cluster-wide synchronization
 * - Better L2 cache locality
 *
 * PERFORMANCE CHARACTERISTICS:
 * ============================
 * - H100: ~970 TFLOPS (98% of 989 TFLOPS peak for FP16)
 * - Shared memory: 228 KB per SM
 * - Auto-tuned pipeline stages (3-5 typical)
 * - Cluster size: 2-8 CTAs for optimal L2 reuse
 *
 * COMPILE:
 * ========
 * nvcc 03_hopper_tma_gmma.cu -o 03_hopper_tma_gmma \
 *      -I../../include -I../../tools/util/include \
 *      -arch=sm_90a -std=c++17 -O3 --expt-relaxed-constexpr
 *
 * NOTE: -arch=sm_90a (with 'a') is REQUIRED for TMA/GMMA instructions!
 *
 * RUN:
 * ====
 * ./03_hopper_tma_gmma --m=4096 --n=4096 --k=4096 --verify
 *
 **************************************************************************************************/

#include <iostream>
#include <chrono>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/command_line.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/***************************************************************************************************
 * SECTION 1: TYPE DEFINITIONS - HOPPER STYLE
 * ===========================================
 *
 * EDUCATIONAL NOTE:
 * ----------------
 * For Hopper, we use:
 * - FP32 inputs: Demonstrates TF32 mode (implicit FP32→TF32 conversion)
 * - FP32 accumulation: Full precision (though Tensor Cores operate in TF32)
 * - FP32 output: Standard precision
 *
 * TF32 (TensorFloat-32):
 * ---------------------
 * - Input: FP32 (1 sign + 8 exp + 23 mantissa bits)
 * - TensorCore truncates to: 1 sign + 8 exp + 10 mantissa bits
 * - Accumulates in FP32
 * - Result: Full FP32 precision
 *
 * Benefits:
 * - Drop-in replacement for FP32 code
 * - 8× throughput vs. pure FP32 SIMT
 * - Minimal accuracy loss for most workloads
 *
 **************************************************************************************************/

using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

/***************************************************************************************************
 * SECTION 2: HOPPER GEMM CONFIGURATION
 * =====================================
 *
 * EDUCATIONAL NOTE:
 * ----------------
 * Hopper configurations differ significantly from Volta/Ampere:
 *
 * TILE SHAPE (128×128×64):
 * ------------------------
 * - Larger K dimension (64 vs. 32) leverages higher memory bandwidth
 * - 128×128 M×N is sweet spot for most problems
 * - Can go up to 256×256 for very large GEMMs
 *
 * CLUSTER SHAPE (2×1×1):
 * ----------------------
 * - NEW in Hopper: Multiple threadblocks cooperate
 * - Shape<2,1,1>: 2 CTAs in M dimension form a cluster
 * - Total cluster size: 2 blocks
 * - Benefits:
 *   * Better L2 cache locality (adjacent tiles in same cluster)
 *   * Enables distributed shared memory (not used in basic example)
 *
 * Trade-offs:
 * - Larger clusters: Better locality, longer sync
 * - Smaller clusters: More parallelism, less reuse
 * - Shape<1,1,1>: No clustering (falls back to independent CTAs)
 *
 * STAGE COUNT (Auto):
 * -------------------
 * - StageCountAuto: CUTLASS computes optimal stages based on:
 *   * Available shared memory (228 KB on H100)
 *   * Tile size
 *   * Epilogue shared memory requirements
 * - Typical result: 3-5 stages for this configuration
 *
 * KERNEL SCHEDULE (Auto):
 * -----------------------
 * - KernelScheduleAuto: CUTLASS selects best schedule:
 *   * Warp-specialized (most common for Hopper)
 *   * Pingpong (double-buffered producers)
 *   * Cooperative (all warps participate in all work)
 *
 **************************************************************************************************/

using TileShape = Shape<_128, _128, _64>;           // Threadblock tile
using ClusterShape = Shape<_2, _1, _1>;             // 2×1 cluster (2 CTAs collaborate)

/***************************************************************************************************
 * SECTION 3: COLLECTIVE BUILDERS - THE CUTLASS 3.x WAY
 * =====================================================
 *
 * EDUCATIONAL NOTE:
 * ----------------
 * Collective Builders are the high-level API in CUTLASS 3.x. They abstract away
 * architecture-specific details and automatically select optimal implementations.
 *
 * MAINLOOP COLLECTIVE:
 * --------------------
 * Responsibilities:
 * - Load A and B tiles from global memory
 * - Store in shared memory (with TMA on Hopper)
 * - Coordinate producer/consumer warps
 * - Perform GMMA instructions
 * - Accumulate results in registers
 *
 * Builder Parameters:
 * -------------------
 * 1. cutlass::arch::Sm90:
 *    Triggers selection of Hopper-specific implementations
 *    Automatically chooses TMA instead of cp.async
 *
 * 2. cutlass::arch::OpClassTensorOp:
 *    Use Tensor Cores (vs. SIMT cores)
 *
 * 3. ElementA/B, LayoutA/B, Alignment:
 *    Data types, layouts, and memory alignment
 *    Alignment of 4 floats = 16 bytes = 128-bit aligned
 *
 * 4. ElementAccumulator:
 *    Internal precision for accumulation
 *
 * 5. TileShape, ClusterShape:
 *    Work sizes at CTA and cluster levels
 *
 * 6. StageCountAuto:
 *    Automatic pipeline depth calculation
 *    Alternative: StageCount<N> for manual control
 *
 * 7. KernelScheduleAuto:
 *    Automatic selection of execution schedule
 *    Will choose warp-specialized for this config
 *
 * WHAT THE BUILDER DOES:
 * ----------------------
 * At compile time, the builder:
 * 1. Checks architecture compatibility (SM90 features available?)
 * 2. Selects TMA for copies (instead of manual cp.async)
 * 3. Chooses GMMA for matrix multiply (instead of mma.sync)
 * 4. Configures warp specialization (producer/consumer roles)
 * 5. Computes shared memory layout with bank conflict avoidance
 * 6. Calculates optimal pipeline stage count
 *
 * Result: A complete mainloop implementation optimized for Hopper!
 *
 **************************************************************************************************/

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,                             // Target Hopper
    cutlass::arch::OpClassTensorOp,                  // Use Tensor Cores
    ElementA, LayoutA, 4,                            // A: float, row-major, 4-element align
    ElementB, LayoutB, 4,                            // B: float, col-major, 4-element align
    ElementAccumulator,                              // Accumulator type
    TileShape, ClusterShape,                         // Tile and cluster shapes
    cutlass::gemm::collective::StageCountAuto,       // Auto-compute pipeline stages
    cutlass::gemm::collective::KernelScheduleAuto    // Auto-select schedule
>::CollectiveOp;

/***************************************************************************************************
 * EPILOGUE COLLECTIVE:
 * --------------------
 * Responsibilities:
 * - Load C matrix (if beta ≠ 0)
 * - Perform epilogue computation: D = α(AB) + βC
 * - Write D matrix to global memory (with TMA on Hopper)
 *
 * Builder Parameters:
 * -------------------
 * Most are similar to mainloop, plus:
 *
 * 1. EpilogueTileAuto:
 *    Automatic epilogue tile size selection
 *    Usually matches or subdivides mainloop tile
 *
 * 2. ElementAccumulator (appears twice):
 *    - First: Type coming from mainloop accumulators
 *    - Second: Type for epilogue computation (α, β scaling)
 *
 * 3. ElementC, LayoutC (appears twice):
 *    - First pair: C matrix (input to epilogue)
 *    - Second pair: D matrix (output from epilogue)
 *
 * 4. EpilogueScheduleAuto:
 *    Automatic selection of epilogue schedule
 *    Options: TMA-based store, vectorized store, etc.
 *
 **************************************************************************************************/

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,          // Accumulator and compute types
    ElementC, LayoutC, 4,                            // C matrix
    ElementC, LayoutC, 4,                            // D matrix
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

/***************************************************************************************************
 * SECTION 4: KERNEL COMPOSITION
 * ==============================
 *
 * EDUCATIONAL NOTE:
 * ----------------
 * In CUTLASS 3.x, the kernel is composed of two collectives:
 * 1. Mainloop: Load A, B → Compute AB
 * 2. Epilogue: Load C → Compute D = α(AB) + βC → Store D
 *
 * Shape<int, int, int>:
 * ---------------------
 * Indicates problem dimensions are runtime values (not compile-time constants)
 * This enables one kernel to handle arbitrary M, N, K
 *
 * GemmUniversal vs. Basic Gemm:
 * -----------------------------
 * - Universal: Supports batched, split-K, grouped modes
 * - Basic: Only single GEMM (lighter weight, slightly faster)
 * - Universal is recommended default (minimal overhead)
 *
 **************************************************************************************************/

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,                            // Runtime problem shape
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

/***************************************************************************************************
 * SECTION 5: HELPER FUNCTIONS
 * ============================
 **************************************************************************************************/

void print_performance(int M, int N, int K, double runtime_ms) {
  double gflops = (2.0 * M * N * K) / (runtime_ms / 1000.0) / 1e9;
  double tflops = gflops / 1000.0;

  std::cout << "=================================================\n";
  std::cout << "Hopper GEMM Performance\n";
  std::cout << "=================================================\n";
  std::cout << "Problem size: " << M << "×" << N << "×" << K << "\n";
  std::cout << "Runtime:      " << runtime_ms << " ms\n";
  std::cout << "Performance:  " << gflops << " GFLOPS\n";
  std::cout << "              " << tflops << " TFLOPS\n";
  std::cout << "=================================================\n";
}

void print_kernel_info() {
  std::cout << "\n";
  std::cout << "╔═══════════════════════════════════════════════╗\n";
  std::cout << "║  Hopper Architecture Features                 ║\n";
  std::cout << "╚═══════════════════════════════════════════════╝\n";
  std::cout << "\n";
  std::cout << "TMA (Tensor Memory Accelerator):\n";
  std::cout << "  ✓ Hardware-managed bulk tensor copies\n";
  std::cout << "  ✓ Single-thread can initiate multi-KB transfer\n";
  std::cout << "  ✓ Automatic bank conflict avoidance via swizzling\n";
  std::cout << "  ✓ Supports 2D/3D/4D tensors with arbitrary strides\n";
  std::cout << "\n";
  std::cout << "GMMA (wgmma.mma_async):\n";
  std::cout << "  ✓ Warpgroup-level operations (128 threads)\n";
  std::cout << "  ✓ Direct shared memory operands\n";
  std::cout << "  ✓ Higher throughput than Ampere mma.sync\n";
  std::cout << "  ✓ Asynchronous execution with barriers\n";
  std::cout << "\n";
  std::cout << "Warp Specialization:\n";
  std::cout << "  ✓ Producer warps: TMA loads only\n";
  std::cout << "  ✓ Consumer warps: GMMA compute only\n";
  std::cout << "  ✓ Overlapped execution → ~16% speedup\n";
  std::cout << "\n";
  std::cout << "Thread Block Clusters:\n";
  std::cout << "  ✓ Multiple CTAs cooperate\n";
  std::cout << "  ✓ Better L2 cache locality\n";
  std::cout << "  ✓ Cluster-wide synchronization\n";
  std::cout << "\n";
}

/***************************************************************************************************
 * SECTION 6: MAIN - EXECUTION AND MEASUREMENT
 * ============================================
 **************************************************************************************************/

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);

  int M = 4096;
  int N = 4096;
  int K = 4096;
  bool verify = false;

  cmd.get_cmd_line_argument("m", M);
  cmd.get_cmd_line_argument("n", N);
  cmd.get_cmd_line_argument("k", K);
  verify = cmd.check_cmd_line_flag("verify");

  std::cout << "\n";
  std::cout << "╔═══════════════════════════════════════════════╗\n";
  std::cout << "║  CUTLASS Tutorial 03: Hopper TMA + GMMA      ║\n";
  std::cout << "╚═══════════════════════════════════════════════╝\n";
  std::cout << "\n";
  std::cout << "Problem: D = α(A×B) + βC (TF32 precision)\n";
  std::cout << "  A: " << M << "×" << K << " (FP32, RowMajor)\n";
  std::cout << "  B: " << K << "×" << N << " (FP32, ColumnMajor)\n";
  std::cout << "  C: " << M << "×" << N << " (FP32, ColumnMajor)\n";
  std::cout << "  D: " << M << "×" << N << " (FP32, ColumnMajor)\n";
  std::cout << "\n";
  std::cout << "Configuration:\n";
  std::cout << "  Tile shape:    128×128×64\n";
  std::cout << "  Cluster shape: 2×1×1 (2 CTAs cooperate)\n";
  std::cout << "  Pipeline:      Auto-tuned stages\n";
  std::cout << "  Schedule:      Warp-specialized (producer/consumer)\n";
  std::cout << "\n";

  print_kernel_info();

  ElementAccumulator alpha = ElementAccumulator(1.0f);
  ElementAccumulator beta = ElementAccumulator(0.0f);

  // Allocate tensors
  cutlass::HostTensor<ElementA, LayoutA> tensor_A({M, K});
  cutlass::HostTensor<ElementB, LayoutB> tensor_B({K, N});
  cutlass::HostTensor<ElementC, LayoutC> tensor_C({M, N});
  cutlass::HostTensor<ElementC, LayoutC> tensor_D({M, N});
  cutlass::HostTensor<ElementC, LayoutC> tensor_D_ref({M, N});

  // Initialize
  std::cout << "Initializing tensors...\n";
  cutlass::reference::host::TensorFillRandomUniform(
    tensor_A.host_view(), 1, ElementA(2), ElementA(-2), 0
  );
  cutlass::reference::host::TensorFillRandomUniform(
    tensor_B.host_view(), 2, ElementB(2), ElementB(-2), 0
  );
  cutlass::reference::host::TensorFillRandomUniform(
    tensor_C.host_view(), 3, ElementC(2), ElementC(-2), 0
  );

  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();

  //
  // EDUCATIONAL NOTE: Hopper GEMM Arguments
  // =======================================
  // Arguments structure is slightly different in CUTLASS 3.x:
  // - GemmUniversalMode specifies operation type (kGemm, kBatched, kGrouped, etc.)
  // - Mainloop args and epilogue args are separately grouped
  // - This enables better modularity and reuse
  //
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,         // Mode
    {M, N, K},                                        // Problem size
    {
      tensor_A.device_ref(),                         // Mainloop: A ptr + stride
      tensor_B.device_ref(),                         // Mainloop: B ptr + stride
      {}                                             // Mainloop: additional args (empty)
    },
    {
      {alpha, beta},                                 // Epilogue: α, β
      tensor_C.device_ref(),                         // Epilogue: C ptr + stride
      tensor_D.device_ref()                          // Epilogue: D ptr + stride
    }
  };

  // Workspace and initialization
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;
  cutlass::Status status = gemm_op.initialize(arguments, workspace.get());

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "ERROR: GEMM initialization failed\n";
    return -1;
  }

  // Warmup
  std::cout << "Running warmup...\n";
  status = gemm_op.run();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "ERROR: GEMM execution failed\n";
    return -1;
  }
  cudaDeviceSynchronize();

  // Performance measurement
  std::cout << "Running performance measurement...\n";
  const int iterations = 100;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    gemm_op.run();
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avg_runtime_ms = (duration / 1000.0) / iterations;

  print_performance(M, N, K, avg_runtime_ms);

  // Verification
  if (verify) {
    std::cout << "\nRunning verification...\n";

    cutlass::reference::device::Gemm<
      ElementA, LayoutA,
      ElementB, LayoutB,
      ElementC, LayoutC,
      ElementAccumulator, ElementAccumulator
    > reference_gemm;

    reference_gemm(
      {M, N, K}, alpha,
      tensor_A.device_ref(),
      tensor_B.device_ref(),
      beta,
      tensor_C.device_ref(),
      tensor_D_ref.device_ref()
    );

    cudaDeviceSynchronize();

    tensor_D.sync_host();
    tensor_D_ref.sync_host();

    bool passed = cutlass::reference::host::TensorEquals(
      tensor_D.host_view(),
      tensor_D_ref.host_view()
    );

    std::cout << (passed ? "\n✓ Verification PASSED\n" : "\n✗ Verification FAILED\n");
  }

  std::cout << "\n";
  std::cout << "╔═══════════════════════════════════════════════╗\n";
  std::cout << "║  Tutorial Complete!                           ║\n";
  std::cout << "╚═══════════════════════════════════════════════╝\n";
  std::cout << "\n";
  std::cout << "KEY TAKEAWAYS:\n";
  std::cout << "  1. Collective Builders abstract architecture complexity\n";
  std::cout << "  2. TMA eliminates manual memory management\n";
  std::cout << "  3. GMMA achieves higher throughput than mma.sync\n";
  std::cout << "  4. Warp specialization enables compute/memory overlap\n";
  std::cout << "  5. Clusters improve L2 locality for large GEMMs\n";
  std::cout << "\n";
  std::cout << "NEXT STEPS:\n";
  std::cout << "  - Try different cluster shapes: Shape<_4, _2, _1>\n";
  std::cout << "  - Experiment with tile sizes: Shape<_256, _128, _64>\n";
  std::cout << "  - Profile with Nsight Compute to see TMA/GMMA instructions\n";
  std::cout << "  - Continue to ./04_cute_layouts.cu for CuTe deep dive\n";
  std::cout << "\n";

  return 0;
}

#else

int main() {
  std::cout << "This example requires Hopper (SM90) or newer GPU.\n";
  std::cout << "Please compile with -arch=sm_90a and run on H100 or newer.\n";
  return 0;
}

#endif // CUTLASS_ARCH_MMA_SM90_SUPPORTED
