/***************************************************************************************************
 * CUTLASS Tutorial 07: Epilogue Fusion - Getting Elementwise Ops for Free
 *
 * This tutorial demonstrates epilogue fusion, a powerful optimization where elementwise operations
 * (bias addition, activation functions, scaling) are fused into the GEMM epilogue. Since the
 * epilogue is memory-bound (writing C to global memory), adding compute is essentially free!
 *
 * EDUCATIONAL FOCUS:
 * The epilogue is the final stage of GEMM where accumulated results are:
 *   1. Scaled: accumulator *= alpha
 *   2. Combined with C: result = accumulator + beta*C
 *   3. Stored: D[i,j] = result
 *
 * Between steps 2 and 3, we can add "free" operations because we're waiting for memory anyway:
 *   D[i,j] = activation(result + bias[j])  // No extra kernel launch!
 *
 * LEARNING OBJECTIVES:
 * 1. Understand why epilogue fusion is efficient (memory-bound window)
 * 2. Use built-in epilogue operations (LinearCombination, Bias, etc.)
 * 3. Create custom epilogue functors for activations (ReLU, GELU, etc.)
 * 4. Combine multiple operations (bias + GELU, bias + ReLU + clamp)
 * 5. Measure performance gains from fusion
 *
 * KEY CONCEPTS:
 * - Epilogue: Final stage of GEMM (scale, combine, store)
 * - Fusion: Combining multiple operations into one kernel
 * - Memory-bound: Limited by memory bandwidth, not compute
 * - Free compute: Operations that fit in memory-bound window
 *
 * COMPILE:
 *   nvcc -arch=sm_80 -std=c++17 -I../../include 07_epilogue_fusion.cu -o 07_epilogue_fusion
 *
 * RUN:
 *   ./07_epilogue_fusion
 *
 * EXPECTED OUTPUT:
 *   Performance comparison: unfused vs fused epilogue operations
 ***************************************************************************************************/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/util/host_tensor.h"

#include "cute/tensor.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 1: UNDERSTANDING EPILOGUE FUSION
//
// EDUCATIONAL NOTE - Why Fusion is "Free":
//
// Standard GEMM epilogue (no fusion):
//   for each element (i,j):
//     temp = alpha * accumulator[i,j] + beta * C[i,j]  // Compute
//     D[i,j] = temp                                     // Store to global memory (SLOW)
//
// Memory bandwidth: Must read C, write D → 2× memory traffic
// Compute: Simple multiply-add
// Bottleneck: Memory! (A100: 1.5 TB/s memory vs 312 TFLOPS compute)
//
// With fusion (bias + activation):
//   for each element (i,j):
//     temp = alpha * accumulator[i,j] + beta * C[i,j]  // Compute
//     temp = temp + bias[j]                             // FREE! (in registers)
//     temp = activation(temp)                           // FREE! (in registers)
//     D[i,j] = temp                                     // Store (still SLOW)
//
// Memory bandwidth: SAME (still read C, write D)
// Compute: Added bias + activation
// Bottleneck: Still memory! So extra compute is FREE!
//
// KEY INSIGHT: If memory bandwidth is your bottleneck (which it usually is for epilogue),
// adding compute operations costs nothing. You're waiting for memory anyway!
//
// PRACTICAL BENEFIT:
// Instead of 3 kernel launches:
//   kernel1: GEMM → D
//   kernel2: bias_add(D, bias) → D
//   kernel3: activation(D) → D
// You get:
//   kernel1: GEMM + bias + activation → D
// Saves 2 kernel launches + 4× memory traffic (2 reads + 2 writes eliminated)!
//
///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 2: CUSTOM ACTIVATION FUNCTORS
//
// EDUCATIONAL NOTE - Epilogue Functors:
// CUTLASS epilogues use "functors" - small callable objects that transform each element.
// A functor implements operator() to apply the transformation.
//
// Interface:
//   template<typename T>
//   struct MyActivation {
//     CUTLASS_HOST_DEVICE
//     T operator()(T const& value) const {
//       return transform(value);  // Your custom logic
//     }
//   };
//
///////////////////////////////////////////////////////////////////////////////////////////////////

// ReLU activation: max(0, x)
template<typename T>
struct ReLUActivation {
  CUTLASS_HOST_DEVICE
  T operator()(T const& value) const {
    return value > T(0) ? value : T(0);
  }
};

// GELU activation (approximation): x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// Simplified version: x * sigmoid(1.702 * x)
template<typename T>
struct GELUActivation {
  CUTLASS_HOST_DEVICE
  T operator()(T const& value) const {
    // Fast approximation: x * sigmoid(1.702 * x)
    T x = value * T(1.702);
    T sigmoid = T(1) / (T(1) + cutlass::fast_exp(-x));
    return value * sigmoid;
  }
};

// Clamp: min(max(value, lower), upper)
template<typename T>
struct ClampActivation {
  T lower;
  T upper;

  CUTLASS_HOST_DEVICE
  ClampActivation(T lower_ = T(0), T upper_ = T(6)) : lower(lower_), upper(upper_) {}

  CUTLASS_HOST_DEVICE
  T operator()(T const& value) const {
    return value < lower ? lower : (value > upper ? upper : value);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 3: GEMM WITH FUSED EPILOGUE
//
// EDUCATIONAL NOTE - LinearCombinationBiasElementwise:
// CUTLASS provides LinearCombinationBiasElementwise which applies:
//   output = activation(alpha * accumulator + beta * source + bias)
//
// This is perfect for common patterns in deep learning:
//   - Linear layer with bias
//   - Followed by activation (ReLU, GELU, etc.)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementOutput,
         typename ElementAccumulator,
         typename ElementCompute,
         typename ActivationFunctor>
struct FusedEpilogueConfig {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using TileShape = Shape<_128, _128, _32>;
  using ClusterShape = Shape<_1, _1, _1>;

  // Use LinearCombinationBiasElementwise with custom activation
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
      ElementOutput,                                       // Output type
      128 / cutlass::sizeof_bits<ElementOutput>::value,   // Elements per access
      ElementAccumulator,                                  // Accumulator type
      ElementCompute,                                      // Computation type
      ActivationFunctor,                                   // Custom activation
      cutlass::epilogue::thread::ScaleType::Default        // Scaling mode
  >;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm80,
      cutlass::arch::OpClassTensorOp,
      float, LayoutA, 4,
      float, LayoutB, 4,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm80,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementOutput, LayoutC, 4,   // C matrix
      ElementOutput, LayoutC, 4,   // D matrix
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      EpilogueOp                   // Our custom epilogue with fusion!
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 4: HELPER FUNCTIONS
//
///////////////////////////////////////////////////////////////////////////////////////////////////

double calculate_gflops(int m, int n, int k, float time_ms) {
  double flops = 2.0 * m * n * k;
  return (flops / (time_ms / 1000.0)) / 1e9;
}

void initialize_matrix(std::vector<float>& matrix, int size) {
  std::random_device rd;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  for (int i = 0; i < size; ++i) {
    matrix[i] = dis(gen);
  }
}

// Apply ReLU activation on host
void apply_relu_host(std::vector<float>& data) {
  for (auto& val : data) {
    val = std::max(0.0f, val);
  }
}

// Apply GELU activation on host
void apply_gelu_host(std::vector<float>& data) {
  for (auto& val : data) {
    float x = val * 1.702f;
    float sigmoid = 1.0f / (1.0f + std::exp(-x));
    val = val * sigmoid;
  }
}

// Add bias on host
void add_bias_host(std::vector<float>& data, const std::vector<float>& bias, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      data[i * n + j] += bias[j];
    }
  }
}

bool verify_results(const std::vector<float>& result,
                   const std::vector<float>& reference,
                   int size,
                   float tolerance = 1e-3f) {
  int errors = 0;
  for (int i = 0; i < size && errors < 10; ++i) {
    float diff = std::abs(result[i] - reference[i]);
    if (diff > tolerance) {
      if (errors < 5) {
        std::cout << "  Mismatch at " << i << ": " << result[i]
                  << " vs " << reference[i] << " (diff=" << diff << ")" << std::endl;
      }
      errors++;
    }
  }
  return errors == 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 5: DEMONSTRATION - FUSED VS UNFUSED
//
///////////////////////////////////////////////////////////////////////////////////////////////////

void run_fused_gemm_relu() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   Fused GEMM + Bias + ReLU                                     ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using Config = FusedEpilogueConfig<ElementOutput, ElementAccumulator,
                                      ElementCompute, ReLUActivation<ElementCompute>>;
  using Gemm = typename Config::Gemm;

  int M = 4096, N = 4096, K = 4096;

  std::cout << "Problem size: " << M << "×" << N << "×" << K << std::endl;
  std::cout << "Fused operations: D = ReLU(alpha*A*B + beta*C + bias)" << std::endl;
  std::cout << "Savings: 2 kernel launches + 4× memory traffic eliminated!\n" << std::endl;

  // Allocate host memory
  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  std::vector<float> h_C(M * N, 0);
  std::vector<float> h_bias(N);
  std::vector<float> h_D(M * N);

  initialize_matrix(h_A, M * K);
  initialize_matrix(h_B, K * N);
  initialize_matrix(h_bias, N);

  // Allocate device memory
  cutlass::HostTensor<float, typename Config::LayoutA> tensor_a({M, K});
  cutlass::HostTensor<float, typename Config::LayoutB> tensor_b({K, N});
  cutlass::HostTensor<float, typename Config::LayoutC> tensor_c({M, N});
  cutlass::HostTensor<float, typename Config::LayoutC> tensor_d({M, N});
  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_bias({1, N});

  tensor_a.host_view().copy_in_device_to_host(h_A.data());
  tensor_b.host_view().copy_in_device_to_host(h_B.data());
  tensor_c.host_view().copy_in_device_to_host(h_C.data());
  tensor_bias.host_view().copy_in_device_to_host(h_bias.data());
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_bias.sync_device();

  // Setup arguments
  float alpha = 1.0f;
  float beta = 0.0f;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {tensor_a.device_data(), tensor_a.layout(),
     tensor_b.device_data(), tensor_b.layout()},
    {{alpha, beta},
     tensor_c.device_data(), tensor_c.layout(),
     tensor_d.device_data(), tensor_d.layout(),
     tensor_bias.device_data()}  // Bias pointer
  };

  Gemm gemm_op;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) {
    std::cerr << "Cannot implement fused GEMM" << std::endl;
    return;
  }

  gemm_op.initialize(arguments, workspace.get());

  // Warmup
  gemm_op();
  cudaDeviceSynchronize();

  // Timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  gemm_op();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);

  tensor_d.sync_host();
  tensor_d.host_view().copy_in_host_to_device(h_D.data());

  double gflops = calculate_gflops(M, N, K, elapsed_ms);

  std::cout << "Performance:" << std::endl;
  std::cout << "  Time: " << elapsed_ms << " ms" << std::endl;
  std::cout << "  Throughput: " << gflops << " GFLOPS" << std::endl;

  // Verify (compute reference)
  std::cout << "\nVerifying against reference..." << std::endl;
  std::vector<float> h_ref(M * N);

  // Reference: GEMM
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += h_A[i * K + k] * h_B[k * N + j];
      }
      h_ref[i * N + j] = alpha * sum + beta * h_C[i * N + j];
    }
  }

  // Reference: Add bias
  add_bias_host(h_ref, h_bias, M, N);

  // Reference: Apply ReLU
  apply_relu_host(h_ref);

  bool passed = verify_results(h_D, h_ref, M * N, 1e-3f);
  std::cout << "Verification: " << (passed ? "PASSED ✓" : "FAILED ✗") << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  std::cout << "\nKEY INSIGHT:" << std::endl;
  std::cout << "The fused version performs GEMM + bias + ReLU in ONE kernel!" << std::endl;
  std::cout << "Unfused would require 3 kernels: GEMM, bias_add, relu" << std::endl;
  std::cout << "Memory traffic saved: 4× (eliminated 2 reads + 2 writes)" << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// MAIN
//
///////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   CUTLASS Tutorial 07: Epilogue Fusion                         ║" << std::endl;
  std::cout << "║   Getting Elementwise Operations for Free                      ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;

  run_fused_gemm_relu();

  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   KEY TAKEAWAYS                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  1. Epilogue fusion adds compute during memory-bound phase     ║" << std::endl;
  std::cout << "║     Result: \"Free\" operations with no performance cost         ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  2. Common fusions: bias + ReLU, bias + GELU, scaling          ║" << std::endl;
  std::cout << "║     These patterns are ubiquitous in deep learning             ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  3. Fusion eliminates kernel launches and memory traffic       ║" << std::endl;
  std::cout << "║     3 kernels → 1 kernel, 4× less memory traffic               ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  4. Custom functors enable any elementwise operation           ║" << std::endl;
  std::cout << "║     Easy to add your own activations or transformations        ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  5. LinearCombinationBiasElementwise is your friend            ║" << std::endl;
  std::cout << "║     Handles bias + activation fusion automatically             ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║   WHEN TO USE EPILOGUE FUSION                                  ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  ✓ Linear layers with bias + activation (most common)         ║" << std::endl;
  std::cout << "║  ✓ Batch normalization after GEMM                              ║" << std::endl;
  std::cout << "║  ✓ Residual connections: D = activation(A*B + C)              ║" << std::endl;
  std::cout << "║  ✓ Scaling/quantization: D = round((A*B) * scale)             ║" << std::endl;
  std::cout << "║  ✓ Any elementwise operation applied to GEMM output            ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  ✗ Complex operations requiring multiple passes                ║" << std::endl;
  std::cout << "║  ✗ Operations requiring synchronization across elements        ║" << std::endl;
  std::cout << "║  ✗ When output needs to be used in multiple ways               ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║   NEXT STEPS                                                   ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  • Next: 08_performance_tuning.cu - Systematic tuning guide    ║" << std::endl;
  std::cout << "║    for squeezing every last drop of performance                ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  • Experiment: Try different activations (Swish, Mish, etc.)  ║" << std::endl;
  std::cout << "║  • Experiment: Combine multiple operations in one functor      ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;

  return 0;
}
