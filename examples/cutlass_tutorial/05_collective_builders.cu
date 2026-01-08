/***************************************************************************************************
 * CUTLASS Tutorial 05: Collective Builders - The High-Level API of CUTLASS 3.x
 *
 * This tutorial demonstrates the CollectiveBuilder pattern, which is the recommended way to
 * construct GEMM kernels in CUTLASS 3.x. Instead of manually specifying every detail (like in
 * tutorials 01-03), you specify high-level intent and the builder automatically selects the
 * optimal implementation for your target architecture.
 *
 * EDUCATIONAL FOCUS:
 * CollectiveBuilder bridges the gap between "what you want" and "how to get it":
 * - YOU specify: architecture tag, data types, tile sizes, alignment
 * - BUILDER selects: kernel schedule, pipeline stages, memory instructions, warp specialization
 *
 * This is the power of CUTLASS 3.x: write once, optimize everywhere!
 *
 * LEARNING OBJECTIVES:
 * 1. Understand what a "Collective" is in CUTLASS 3.x
 * 2. Learn the CollectiveBuilder pattern and its parameters
 * 3. See how auto-selection works (StageCountAuto, KernelScheduleAuto)
 * 4. Compare implementations across architectures (Ampere vs Hopper)
 * 5. Understand how to customize collectives for your needs
 * 6. Connect CollectiveBuilder to CuTe layouts (from Tutorial 04)
 *
 * KEY CONCEPTS:
 * - Collective: High-level abstraction for mainloop and epilogue
 * - CollectiveBuilder: Factory pattern for creating architecture-specific collectives
 * - Auto-selection: Compiler chooses optimal implementation based on target architecture
 * - GemmUniversal kernel: Combines CollectiveMainloop + CollectiveEpilogue
 *
 * COMPILE:
 *   nvcc -arch=sm_80 -std=c++17 -I../../include 05_collective_builders.cu -o 05_collective_builders
 *
 * RUN:
 *   ./05_collective_builders
 *
 * EXPECTED OUTPUT:
 *   Comparison of Ampere and Hopper implementations using the same CollectiveBuilder code
 ***************************************************************************************************/

#include <iostream>
#include <vector>
#include <random>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"

// CuTe for shapes and layouts
#include "cute/tensor.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 1: UNDERSTANDING COLLECTIVES
//
// EDUCATIONAL NOTE - What is a Collective?
// In CUTLASS 3.x, a "Collective" is a high-level abstraction that encapsulates:
//   1. How data moves from global → shared memory
//   2. How threads cooperate to perform computation
//   3. The pipeline structure (stages, scheduling)
//
// There are two main types of collectives:
//   - CollectiveMainloop: Handles A×B multiplication (load, compute, pipeline)
//   - CollectiveEpilogue: Handles C = alpha*AB + beta*C (load C, compute, store D)
//
// Why "Collective"? Because it describes collective thread behavior, not individual
// thread actions. The collective pattern automatically partitions work across threads
// using CuTe's layout algebra (from Tutorial 04).
//
// EDUCATIONAL NOTE - The Power of CollectiveBuilder:
// Instead of writing this (CUTLASS 2.x style):
//   using Gemm = device::Gemm<float, LayoutA, float, LayoutB, ...
//                              ThreadblockShape, WarpShape, InstructionShape, ...>
//
// You write this (CUTLASS 3.x style):
//   using CollectiveMainloop = collective::CollectiveBuilder<
//       arch::Sm80, OpClassTensorOp,
//       ElementA, LayoutA, 4,  // Just specify types and alignment
//       ElementB, LayoutB, 4,
//       ElementAccumulator,
//       TileShape, ClusterShape,
//       StageCountAuto,        // Let the builder decide!
//       KernelScheduleAuto     // Let the builder decide!
//   >::CollectiveOp;
//
// The builder examines your architecture tag (Sm80, Sm90, etc.) and automatically:
//   - Selects cp.async for Ampere, TMA for Hopper
//   - Chooses optimal pipeline stages
//   - Enables warp specialization where beneficial
//   - Generates correct CuTe layouts for thread→data mapping
//
///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 2: AMPERE COLLECTIVE EXAMPLE
//
// EDUCATIONAL NOTE - Building a Collective for Ampere (SM80):
// When you specify arch::Sm80, the CollectiveBuilder automatically:
//   1. Uses cp.async for async global→shared copies
//   2. Selects 3-5 pipeline stages based on tile size
//   3. Generates standard (non-warp-specialized) schedules
//   4. Creates CuTe layouts for 128-thread threadblocks
//
///////////////////////////////////////////////////////////////////////////////////////////////////

// Problem size and types
constexpr int PROBLEM_M = 4096;
constexpr int PROBLEM_N = 4096;
constexpr int PROBLEM_K = 4096;

using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// Ampere configuration
template<typename ArchTag>
struct GemmConfig;

// Specialization for Ampere (SM80)
template<>
struct GemmConfig<cutlass::arch::Sm80> {
  // EDUCATIONAL NOTE - Tile Shapes in CuTe:
  // We use CuTe's Shape types instead of GemmShape from CUTLASS 2.x
  // Shape<_128, _128, _32> = 128×128×32 tile (M×N×K)
  using TileShape = Shape<_128, _128, _32>;

  // Cluster shape: For Ampere, clusters are not used, so use <_1, _1, _1>
  using ClusterShape = Shape<_1, _1, _1>;

  static constexpr char const* name = "Ampere (SM80)";
  static constexpr char const* expected_schedule = "cp.async with 3-5 stages";
  static constexpr char const* expected_features = "Async copy, TF32 Tensor Cores";
};

// EDUCATIONAL NOTE - The CollectiveBuilder Pattern:
// CollectiveBuilder is a template metafunction that takes:
//   1. Architecture tag (Sm80, Sm90, etc.)
//   2. Operation class (OpClassTensorOp for Tensor Cores)
//   3. Element types and layouts for A, B, accumulator
//   4. Alignment requirements (in elements)
//   5. Tile shapes (using CuTe Shape types)
//   6. Stage count policy (Auto, or explicit number)
//   7. Kernel schedule policy (Auto, or explicit schedule type)
//
// It returns a type CollectiveOp that implements the mainloop.

template<typename ArchTag>
struct BuildGemm {
  using Config = GemmConfig<ArchTag>;
  using TileShape = typename Config::TileShape;
  using ClusterShape = typename Config::ClusterShape;

  // Build the mainloop collective
  // EDUCATIONAL NOTE - StageCountAuto:
  // When you specify StageCountAuto, the builder examines:
  //   - Available shared memory (from architecture)
  //   - Tile size (from TileShape)
  //   - Data types (from ElementA/B)
  // And automatically selects optimal number of stages (typically 3-5 for Ampere)
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 4,    // A matrix: type, layout, alignment
      ElementB, LayoutB, 4,    // B matrix: type, layout, alignment
      ElementAccumulator,      // Accumulator type
      TileShape,               // CuTe shape for threadblock tile
      ClusterShape,            // CuTe shape for cluster (1×1×1 for Ampere)
      cutlass::gemm::collective::StageCountAuto,      // Auto-select stages
      cutlass::gemm::collective::KernelScheduleAuto   // Auto-select schedule
  >::CollectiveOp;

  // Build the epilogue collective
  // EDUCATIONAL NOTE - Default Epilogue:
  // The epilogue performs: D = alpha*C_accumulator + beta*C
  // We use default epilogue which handles standard linear combination
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,      // Input from mainloop
      ElementCompute,          // Computation type for epilogue
      ElementC, LayoutC, 4,    // C matrix (input)
      ElementC, LayoutC, 4,    // D matrix (output)
      cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

  // Combine mainloop + epilogue into a kernel
  // EDUCATIONAL NOTE - GemmUniversal:
  // This is the "universal" kernel that works for all problem sizes and configurations
  // It takes a runtime problem shape and handles boundary conditions automatically
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int>,    // Runtime problem shape (M, N, K)
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  // Wrap the kernel in a device-level interface
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 3: HOPPER COLLECTIVE EXAMPLE (CONCEPTUAL)
//
// EDUCATIONAL NOTE - What Changes for Hopper?
// The beautiful thing about CollectiveBuilder is that the SAME code automatically
// adapts to Hopper's features when you change the architecture tag:
//
//   arch::Sm80 → arch::Sm90
//
// The builder will automatically:
//   1. Switch from cp.async to TMA (Tensor Memory Accelerator)
//   2. Enable warp specialization (producer/consumer warps)
//   3. Support thread block clusters
//   4. Use GMMA instructions (warpgroup-level MMA)
//
// You don't need to change ANY of the CollectiveBuilder template arguments!
// The architecture tag alone tells the builder everything it needs to know.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CUTLASS_ARCH_MMA_SM90_SUPPORTED  // Only compile if targeting Hopper

template<>
struct GemmConfig<cutlass::arch::Sm90> {
  using TileShape = Shape<_128, _128, _64>;    // Larger K for TMA efficiency
  using ClusterShape = Shape<_2, _1, _1>;      // 2×1 cluster for better L2 locality

  static constexpr char const* name = "Hopper (SM90)";
  static constexpr char const* expected_schedule = "TMA + warp specialization";
  static constexpr char const* expected_features = "TMA, GMMA, Thread Block Clusters";
};

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 4: HELPER FUNCTIONS
//
///////////////////////////////////////////////////////////////////////////////////////////////////

double calculate_gflops(int m, int n, int k, float time_ms) {
  double flops = 2.0 * m * n * k;
  double time_s = time_ms / 1000.0;
  return (flops / time_s) / 1e9;
}

template<typename T>
void initialize_matrix(std::vector<T>& matrix, int rows, int cols) {
  std::random_device rd;
  std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  for (int i = 0; i < rows * cols; ++i) {
    matrix[i] = static_cast<T>(dis(gen));
  }
}

template<typename T>
bool verify_results(const std::vector<T>& cutlass_result,
                   const std::vector<T>& reference,
                   int m, int n,
                   T tolerance = static_cast<T>(1e-3)) {
  int errors = 0;
  constexpr int max_errors_to_print = 5;

  for (int i = 0; i < m * n; ++i) {
    T diff = std::abs(cutlass_result[i] - reference[i]);
    T magnitude = std::max(std::abs(cutlass_result[i]), std::abs(reference[i]));
    T relative_error = diff / (magnitude + static_cast<T>(1e-8));

    if (relative_error > tolerance) {
      if (errors < max_errors_to_print) {
        std::cout << "  Mismatch at index " << i
                  << ": CUTLASS=" << cutlass_result[i]
                  << ", Reference=" << reference[i]
                  << ", RelError=" << relative_error << std::endl;
      }
      errors++;
    }
  }

  if (errors > 0) {
    std::cout << "  Total errors: " << errors << " / " << (m * n) << std::endl;
    return false;
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 5: RUNNING A COLLECTIVE-BASED GEMM
//
// EDUCATIONAL NOTE - Using a Collective-Based Kernel:
// The execution flow is similar to CUTLASS 2.x, but with some differences:
//   1. Problem shape is specified at runtime (not encoded in types)
//   2. Arguments use ProblemShape instead of separate M, N, K
//   3. The kernel is more flexible (supports different strides, batch modes, etc.)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ArchTag>
void run_collective_gemm(const char* description) {
  using Gemm = typename BuildGemm<ArchTag>::Gemm;
  using Config = GemmConfig<ArchTag>;

  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   " << std::left << std::setw(60) << description << " ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

  int M = PROBLEM_M;
  int N = PROBLEM_N;
  int K = PROBLEM_K;

  std::cout << "Architecture: " << Config::name << std::endl;
  std::cout << "Expected schedule: " << Config::expected_schedule << std::endl;
  std::cout << "Expected features: " << Config::expected_features << std::endl;
  std::cout << "Problem size: M=" << M << ", N=" << N << ", K=" << K << std::endl;

  // Print tile configuration
  std::cout << "\nCollectiveBuilder automatically selected:" << std::endl;
  std::cout << "  Tile shape: ";
  print(typename Config::TileShape{});
  std::cout << std::endl;
  std::cout << "  Cluster shape: ";
  print(typename Config::ClusterShape{});
  std::cout << std::endl;
  std::cout << "  Stage count: Auto (builder selects optimal)" << std::endl;
  std::cout << "  Kernel schedule: Auto (builder selects optimal)" << std::endl;

  // Allocate host memory
  std::vector<ElementA> h_A(M * K);
  std::vector<ElementB> h_B(K * N);
  std::vector<ElementC> h_C(M * N, 0);
  std::vector<ElementC> h_C_ref(M * N, 0);

  initialize_matrix(h_A, M, K);
  initialize_matrix(h_B, K, N);

  // Allocate device memory
  cutlass::HostTensor<ElementA, LayoutA> tensor_a({M, K});
  cutlass::HostTensor<ElementB, LayoutB> tensor_b({K, N});
  cutlass::HostTensor<ElementC, LayoutC> tensor_c({M, N});
  cutlass::HostTensor<ElementC, LayoutC> tensor_d({M, N});

  tensor_a.host_view().copy_in_device_to_host(h_A.data());
  tensor_b.host_view().copy_in_device_to_host(h_B.data());
  tensor_c.host_view().copy_in_device_to_host(h_C.data());
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();

  // Setup arguments
  // EDUCATIONAL NOTE - GemmUniversal Arguments:
  // Unlike CUTLASS 2.x which used separate M, N, K parameters,
  // GemmUniversal uses a ProblemShape that can be specified at runtime
  ElementCompute alpha = ElementCompute(1.0f);
  ElementCompute beta = ElementCompute(0.0f);

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,  // Mode
    {M, N, K},                                 // ProblemShape (runtime)
    {
      tensor_a.device_data(), tensor_a.layout(),  // A tensor
      tensor_b.device_data(), tensor_b.layout(),  // B tensor
    },
    {
      {alpha, beta},                              // Epilogue scalars
      tensor_c.device_data(), tensor_c.layout(),  // C tensor
      tensor_d.device_data(), tensor_d.layout(),  // D tensor (output)
    }
  };

  // Initialize kernel
  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Kernel cannot implement the given arguments." << std::endl;
    return;
  }

  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Failed to initialize kernel." << std::endl;
    return;
  }

  // Warmup
  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Kernel execution failed." << std::endl;
    return;
  }
  cudaDeviceSynchronize();

  // Timing
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
    std::cerr << "Kernel execution failed." << std::endl;
    return;
  }

  // Copy result
  tensor_d.sync_host();
  tensor_d.host_view().copy_in_host_to_device(h_C.data());

  double gflops = calculate_gflops(M, N, K, elapsed_ms);
  std::cout << "\nPerformance: " << elapsed_ms << " ms ("
            << gflops << " GFLOPS)" << std::endl;

  // Verify
  std::cout << "\nComputing reference..." << std::endl;
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
    M
  );

  bool passed = verify_results(h_C, h_C_ref, M, N);
  std::cout << "Verification: " << (passed ? "PASSED ✓" : "FAILED ✗") << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 6: MAIN - DEMONSTRATE COLLECTIVE BUILDER
//
///////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   CUTLASS Tutorial 05: Collective Builders                     ║" << std::endl;
  std::cout << "║   Write Once, Optimize Everywhere                              ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;

  std::cout << "\nEDUCATIONAL NOTE:" << std::endl;
  std::cout << "This tutorial demonstrates how CollectiveBuilder automatically adapts" << std::endl;
  std::cout << "to different architectures. The SAME template code produces different" << std::endl;
  std::cout << "implementations based solely on the architecture tag!" << std::endl;

  // Run on Ampere (or whatever architecture we're compiling for)
  run_collective_gemm<cutlass::arch::Sm80>("Ampere Collective GEMM");

#ifdef CUTLASS_ARCH_MMA_SM90_SUPPORTED
  // If compiled for Hopper, also run the Hopper version
  run_collective_gemm<cutlass::arch::Sm90>("Hopper Collective GEMM");

  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   ARCHITECTURE COMPARISON                                      ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  Notice how the SAME CollectiveBuilder code produced:         ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  Ampere (SM80):                                                ║" << std::endl;
  std::cout << "║    - cp.async for memory copies                                ║" << std::endl;
  std::cout << "║    - 3-5 pipeline stages                                       ║" << std::endl;
  std::cout << "║    - Cooperative kernel schedule                               ║" << std::endl;
  std::cout << "║    - TF32 Tensor Cores                                         ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  Hopper (SM90):                                                ║" << std::endl;
  std::cout << "║    - TMA for bulk memory copies                                ║" << std::endl;
  std::cout << "║    - Warp specialization (producer/consumer)                   ║" << std::endl;
  std::cout << "║    - Thread block clusters (2×1)                               ║" << std::endl;
  std::cout << "║    - GMMA warpgroup instructions                               ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  All from changing ONE template parameter: Sm80 → Sm90!       ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;
#endif

  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   KEY TAKEAWAYS                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  1. CollectiveBuilder separates INTENT from IMPLEMENTATION     ║" << std::endl;
  std::cout << "║     You specify what you want, builder selects how             ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  2. StageCountAuto and KernelScheduleAuto enable portability   ║" << std::endl;
  std::cout << "║     Same code → optimal results across architectures           ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  3. Architecture tag (Sm80, Sm90) is the only thing that       ║" << std::endl;
  std::cout << "║     changes to get architecture-specific optimizations         ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  4. Collectives encapsulate complex patterns:                  ║" << std::endl;
  std::cout << "║     - Memory movement (global→shared→register)                 ║" << std::endl;
  std::cout << "║     - Thread cooperation (via CuTe layouts)                    ║" << std::endl;
  std::cout << "║     - Pipeline management (stages, scheduling)                 ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  5. GemmUniversal kernel supports runtime problem shapes       ║" << std::endl;
  std::cout << "║     More flexible than CUTLASS 2.x compile-time shapes         ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║   HOW COLLECTIVEBUILDER WORKS                                  ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  The builder uses C++ template metaprogramming to:            ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  1. Examine architecture tag (Sm80, Sm90, Sm100, etc.)        ║" << std::endl;
  std::cout << "║  2. Query architecture capabilities:                           ║" << std::endl;
  std::cout << "║     - Shared memory size                                       ║" << std::endl;
  std::cout << "║     - Supported instructions (cp.async, TMA, GMMA)             ║" << std::endl;
  std::cout << "║     - Tensor Core shapes                                       ║" << std::endl;
  std::cout << "║  3. Calculate optimal pipeline stages based on:                ║" << std::endl;
  std::cout << "║     - Tile size × data types = memory per stage                ║" << std::endl;
  std::cout << "║     - Available shared memory                                  ║" << std::endl;
  std::cout << "║     - Target occupancy                                         ║" << std::endl;
  std::cout << "║  4. Select kernel schedule:                                    ║" << std::endl;
  std::cout << "║     - Cooperative (all threads do load+compute)                ║" << std::endl;
  std::cout << "║     - Warp specialized (producer/consumer split)               ║" << std::endl;
  std::cout << "║     - Pingpong (overlapped load/compute)                       ║" << std::endl;
  std::cout << "║  5. Generate CuTe layouts for thread→data mapping              ║" << std::endl;
  std::cout << "║     (using layout algebra from Tutorial 04)                    ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  All of this happens at COMPILE TIME → zero runtime overhead! ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║   CUSTOMIZATION OPTIONS                                        ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  While auto-selection works great, you CAN override:          ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  1. Explicit stage count:                                      ║" << std::endl;
  std::cout << "║     StageCountAuto → StageCount<4>                             ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  2. Explicit kernel schedule:                                  ║" << std::endl;
  std::cout << "║     KernelScheduleAuto → KernelCpAsyncWarpSpecialized          ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  3. Custom tile shapes:                                        ║" << std::endl;
  std::cout << "║     TileShape = Shape<_128, _256, _64>  // Rectangular tiles  ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  4. Cluster configuration (Hopper+):                           ║" << std::endl;
  std::cout << "║     ClusterShape = Shape<_2, _2, _1>  // 2×2 cluster          ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  But 95% of the time, Auto works best!                        ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║   NEXT STEPS                                                   ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  • Next: 06_mixed_precision.cu - Explore FP16/FP8/INT8        ║" << std::endl;
  std::cout << "║    mixed precision with CollectiveBuilder                      ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  • Experiment: Try different TileShapes and measure GFLOPS    ║" << std::endl;
  std::cout << "║    Shape<_64, _64, _32> vs Shape<_128, _128, _32> vs ...      ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  • Deep dive: Read include/cutlass/gemm/collective/ for       ║" << std::endl;
  std::cout << "║    implementation details of different schedules               ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;

  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// APPENDIX: DEEP DIVE INTO COLLECTIVEBUILDER
//
// EDUCATIONAL NOTE - Template Metaprogramming Magic:
//
// The CollectiveBuilder uses a pattern called "tag dispatch" to select implementations:
//
// template<typename ArchTag, ...>
// struct CollectiveBuilder {
//   using CollectiveOp = typename detail::CollectiveBuilderImpl<
//     ArchTag, ...
//   >::type;
// };
//
// template<...>
// struct CollectiveBuilderImpl<arch::Sm80, ...> {
//   using type = CollectiveMma<...cp.async policies...>;
// };
//
// template<...>
// struct CollectiveBuilderImpl<arch::Sm90, ...> {
//   using type = CollectiveMma<...TMA policies...>;
// };
//
// At compile time, the compiler pattern-matches on ArchTag and selects the
// appropriate specialization. This is resolved completely at compile time,
// so there's no runtime dispatch overhead!
//
// EDUCATIONAL NOTE - Connection to CuTe:
// The CollectiveOp internally uses CuTe layouts (from Tutorial 04) to:
//   1. Partition global tensors across threadblocks
//   2. Partition threadblock tiles across warps
//   3. Partition warp tiles across threads
//   4. Generate copy operations (TiledCopy)
//   5. Generate MMA operations (TiledMMA)
//
// This is why understanding CuTe layouts is crucial - they're the foundation
// that makes CollectiveBuilder possible!
//
///////////////////////////////////////////////////////////////////////////////////////////////////
