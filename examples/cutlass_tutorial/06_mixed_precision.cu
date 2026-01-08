/***************************************************************************************************
 * CUTLASS Tutorial 06: Mixed Precision GEMM - FP16, FP8, INT8, and Beyond
 *
 * This tutorial explores mixed precision computing, where inputs and outputs use lower precision
 * (FP16, FP8, INT8) while accumulation uses higher precision (FP32) to maintain accuracy.
 * Mixed precision is essential for modern AI workloads, providing 2-8× speedup with minimal
 * accuracy loss.
 *
 * EDUCATIONAL FOCUS:
 * Mixed precision is not just about changing data types - it's about balancing:
 *   - Throughput: Lower precision = higher FLOPS, better memory bandwidth
 *   - Accuracy: Higher precision accumulation prevents catastrophic rounding errors
 *   - Memory: Smaller datatypes = larger batch sizes, more data in cache
 *
 * LEARNING OBJECTIVES:
 * 1. Understand when and why to use mixed precision
 * 2. Learn FP16 GEMM: 2× speedup, standard for AI training
 * 3. Explore FP8 GEMM: 4× speedup on Hopper, emerging for training
 * 4. See INT8 GEMM: 8× speedup, standard for inference
 * 5. Understand accumulation precision (why FP16→FP32 is common)
 * 6. Use CollectiveBuilder with mixed precision
 *
 * KEY CONCEPTS:
 * - Numeric formats: FP32 (23-bit mantissa), FP16 (10-bit), TF32 (19-bit), FP8 (3-4 bit)
 * - Dynamic range vs precision tradeoff
 * - Accumulator precision: preventing accumulation errors
 * - Throughput hierarchy: FP32 < TF32 ≈ FP16 < FP8 < INT8
 *
 * COMPILE:
 *   nvcc -arch=sm_80 -std=c++17 -I../../include 06_mixed_precision.cu -o 06_mixed_precision
 *
 * RUN:
 *   ./06_mixed_precision
 *
 * EXPECTED OUTPUT:
 *   Performance comparison across FP32, FP16, and INT8 precisions
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
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"

// Numeric types
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"

#include "cute/tensor.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 1: UNDERSTANDING NUMERIC FORMATS
//
// EDUCATIONAL NOTE - Floating Point Format Breakdown:
//
// FP32 (IEEE 754 single precision):
//   Sign: 1 bit, Exponent: 8 bits, Mantissa: 23 bits
//   Range: ±1.4e-45 to ±3.4e38, Precision: ~7 decimal digits
//   Peak FLOPS (A100): 19.5 TFLOPS
//
// TF32 (TensorFloat-32, Ampere+):
//   Sign: 1 bit, Exponent: 8 bits, Mantissa: 19 bits (actually: 10 bits + 1 implicit)
//   Same range as FP32, slightly less precision
//   Peak FLOPS (A100): 156 TFLOPS (8× faster than FP32!)
//   Automatic rounding from FP32 inputs
//
// FP16 (IEEE 754 half precision):
//   Sign: 1 bit, Exponent: 5 bits, Mantissa: 10 bits
//   Range: ±5.96e-8 to ±65,504, Precision: ~3 decimal digits
//   Peak FLOPS (A100): 312 TFLOPS (16× faster than FP32)
//   Requires loss scaling in training to avoid underflow
//
// BF16 (Brain Float 16, Google):
//   Sign: 1 bit, Exponent: 8 bits, Mantissa: 7 bits
//   Same range as FP32 (better than FP16), less precision
//   Peak FLOPS (A100): 312 TFLOPS
//   Drop-in replacement for FP32 in many cases
//
// FP8 (E4M3 and E5M2, Hopper+):
//   E4M3: Exponent: 4 bits, Mantissa: 3 bits (more precision, less range)
//   E5M2: Exponent: 5 bits, Mantissa: 2 bits (less precision, more range)
//   Range: ~±240 (E4M3) or ~±57,000 (E5M2)
//   Peak FLOPS (H100): 1,979 TFLOPS for E4M3 (Sparse)
//   Requires careful scaling
//
// INT8 (8-bit integer):
//   Range: -128 to 127 (signed) or 0 to 255 (unsigned)
//   Peak TOPS (A100): 624 TOPS (INT8 ops, ~32× FP32)
//   Requires quantization (scale + zero-point)
//
// KEY INSIGHT: Lower precision = higher throughput BUT less accuracy
// Solution: Use low precision for inputs, high precision for accumulation!
//   Example: FP16 inputs × FP16 weights → FP32 accumulator → FP16 output
//
///////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 2: FP16 GEMM WITH FP32 ACCUMULATION
//
// EDUCATIONAL NOTE - Why FP16→FP32 Accumulation?
// Consider accumulating N products: sum = a0*b0 + a1*b1 + ... + aN*bN
//
// If we accumulate in FP16:
//   - FP16 has 10-bit mantissa ≈ 3 decimal digits of precision
//   - After ~2^10 = 1024 additions, accumulated rounding error dominates
//   - For large K (e.g., K=4096), this causes significant error
//
// If we accumulate in FP32:
//   - FP32 has 23-bit mantissa ≈ 7 decimal digits
//   - Can safely accumulate millions of terms
//   - Final result converted back to FP16 for output
//
// This is why "mixed precision" is standard: FP16 inputs, FP32 accumulation!
//
///////////////////////////////////////////////////////////////////////////////////////////////////

// Configuration for FP16 GEMM
template<typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
struct MixedPrecisionConfig {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;

  using TileShape = Shape<_128, _128, _32>;
  using ClusterShape = Shape<_1, _1, _1>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm80,
      cutlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,                    // A: 8-element alignment for FP16
      ElementB, LayoutB, 8,                    // B: 8-element alignment for FP16
      ElementAccumulator,                       // Accumulator (typically FP32)
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
      ElementAccumulator,                       // Input from mainloop (FP32)
      ElementAccumulator,                       // Computation type
      ElementC, LayoutC, 8,                     // C (input)
      ElementC, LayoutC, 8,                     // D (output)
      cutlass::epilogue::collective::EpilogueScheduleAuto
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
// SECTION 3: HELPER FUNCTIONS
//
///////////////////////////////////////////////////////////////////////////////////////////////////

double calculate_gflops(int m, int n, int k, float time_ms) {
  double flops = 2.0 * m * n * k;
  double time_s = time_ms / 1000.0;
  return (flops / time_s) / 1e9;
}

// Initialize FP32 matrix
void initialize_matrix_fp32(std::vector<float>& matrix, int size) {
  std::random_device rd;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (int i = 0; i < size; ++i) {
    matrix[i] = dis(gen);
  }
}

// Initialize FP16 matrix
void initialize_matrix_fp16(std::vector<cutlass::half_t>& matrix, int size) {
  std::random_device rd;
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (int i = 0; i < size; ++i) {
    matrix[i] = cutlass::half_t(dis(gen));
  }
}

// Initialize INT8 matrix
void initialize_matrix_int8(std::vector<int8_t>& matrix, int size) {
  std::random_device rd;
  std::mt19937 gen(42);
  std::uniform_int_distribution<int> dis(-10, 10);  // Smaller range for stability

  for (int i = 0; i < size; ++i) {
    matrix[i] = static_cast<int8_t>(dis(gen));
  }
}

// Convert FP16 to FP32 for verification
std::vector<float> convert_fp16_to_fp32(const std::vector<cutlass::half_t>& input) {
  std::vector<float> output(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    output[i] = static_cast<float>(input[i]);
  }
  return output;
}

// Verify with tolerance
template<typename T>
bool verify_results(const std::vector<T>& cutlass_result,
                   const std::vector<float>& reference,
                   int m, int n,
                   float tolerance = 1e-2) {  // Relaxed tolerance for mixed precision
  int errors = 0;
  constexpr int max_errors = 5;

  for (int i = 0; i < m * n; ++i) {
    float cutlass_val = static_cast<float>(cutlass_result[i]);
    float ref_val = reference[i];
    float diff = std::abs(cutlass_val - ref_val);
    float magnitude = std::max(std::abs(cutlass_val), std::abs(ref_val));
    float relative_error = diff / (magnitude + 1e-8f);

    if (relative_error > tolerance) {
      if (errors < max_errors) {
        std::cout << "  Mismatch at " << i
                  << ": CUTLASS=" << cutlass_val
                  << ", Ref=" << ref_val
                  << ", RelErr=" << relative_error << std::endl;
      }
      errors++;
    }
  }

  if (errors > 0) {
    std::cout << "  Total errors: " << errors << " / " << (m * n)
              << " (" << (100.0 * errors / (m * n)) << "%)" << std::endl;
  }

  return errors == 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 4: FP16 GEMM EXAMPLE
//
// EDUCATIONAL NOTE - FP16 Performance Benefits:
// On A100:
//   - FP32 peak: 19.5 TFLOPS
//   - TF32 peak: 156 TFLOPS (8× faster)
//   - FP16 peak: 312 TFLOPS (16× faster!)
//
// Memory benefits:
//   - FP16 uses 50% less bandwidth than FP32
//   - Larger matrices fit in cache
//   - Double the batch size for same memory
//
// When to use FP16:
//   ✓ AI training (with loss scaling)
//   ✓ AI inference (when 10-bit precision is sufficient)
//   ✓ Graphics/rendering
//   ✗ Numerical simulation (unless error is acceptable)
//   ✗ Financial computing (precision requirements)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

void run_fp16_gemm() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   FP16 GEMM with FP32 Accumulation                             ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementC = cutlass::half_t;
  using ElementAccumulator = float;  // FP32 accumulation!

  using Config = MixedPrecisionConfig<ElementA, ElementB, ElementC, ElementAccumulator>;
  using Gemm = typename Config::Gemm;

  int M = 4096, N = 4096, K = 4096;

  std::cout << "Configuration:" << std::endl;
  std::cout << "  Input A: FP16 (" << sizeof(ElementA) << " bytes per element)" << std::endl;
  std::cout << "  Input B: FP16 (" << sizeof(ElementB) << " bytes per element)" << std::endl;
  std::cout << "  Accumulator: FP32 (" << sizeof(ElementAccumulator) << " bytes)" << std::endl;
  std::cout << "  Output C: FP16 (" << sizeof(ElementC) << " bytes per element)" << std::endl;
  std::cout << "  Problem size: " << M << "×" << N << "×" << K << std::endl;
  std::cout << "  Memory for inputs: " << ((M*K + K*N) * sizeof(ElementA) / 1e6) << " MB" << std::endl;
  std::cout << "  Memory savings vs FP32: " << (100 * (1 - sizeof(ElementA)/sizeof(float))) << "%" << std::endl;

  // Allocate host memory
  std::vector<ElementA> h_A(M * K);
  std::vector<ElementB> h_B(K * N);
  std::vector<ElementC> h_C(M * N, ElementC(0));

  initialize_matrix_fp16(h_A, M * K);
  initialize_matrix_fp16(h_B, K * N);

  // Allocate device memory
  cutlass::HostTensor<ElementA, typename Config::LayoutA> tensor_a({M, K});
  cutlass::HostTensor<ElementB, typename Config::LayoutB> tensor_b({K, N});
  cutlass::HostTensor<ElementC, typename Config::LayoutC> tensor_c({M, N});
  cutlass::HostTensor<ElementC, typename Config::LayoutC> tensor_d({M, N});

  tensor_a.host_view().copy_in_device_to_host(h_A.data());
  tensor_b.host_view().copy_in_device_to_host(h_B.data());
  tensor_c.host_view().copy_in_device_to_host(h_C.data());
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();

  // Setup GEMM
  ElementAccumulator alpha = 1.0f;
  ElementAccumulator beta = 0.0f;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {tensor_a.device_data(), tensor_a.layout(),
     tensor_b.device_data(), tensor_b.layout()},
    {{alpha, beta},
     tensor_c.device_data(), tensor_c.layout(),
     tensor_d.device_data(), tensor_d.layout()}
  };

  Gemm gemm_op;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Cannot implement GEMM" << std::endl;
    return;
  }

  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed to initialize" << std::endl;
    return;
  }

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
  tensor_d.host_view().copy_in_host_to_device(h_C.data());

  double gflops = calculate_gflops(M, N, K, elapsed_ms);

  std::cout << "\nPerformance:" << std::endl;
  std::cout << "  Time: " << elapsed_ms << " ms" << std::endl;
  std::cout << "  Throughput: " << gflops << " GFLOPS" << std::endl;
  std::cout << "  A100 FP16 peak: 312,000 GFLOPS" << std::endl;
  std::cout << "  Achieved: " << (100.0 * gflops / 312000.0) << "% of peak" << std::endl;

  // Verification (convert to FP32 for comparison)
  std::cout << "\nVerifying (FP32 reference)..." << std::endl;
  auto h_A_fp32 = convert_fp16_to_fp32(h_A);
  auto h_B_fp32 = convert_fp16_to_fp32(h_B);
  std::vector<float> h_C_ref_fp32(M * N, 0.0f);

  cutlass::reference::host::Gemm<float, typename Config::LayoutA,
                                  float, typename Config::LayoutB,
                                  float, typename Config::LayoutC,
                                  float, float> reference_gemm;

  reference_gemm({M, N, K}, alpha, h_A_fp32.data(), K, h_B_fp32.data(), N,
                 beta, h_C_ref_fp32.data(), N);

  bool passed = verify_results(h_C, h_C_ref_fp32, M, N, 1e-2f);
  std::cout << "Verification: " << (passed ? "PASSED ✓" : "FAILED ✗") << std::endl;

  if (!passed) {
    std::cout << "\nNote: Small errors are expected due to FP16 rounding" << std::endl;
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// SECTION 5: INT8 GEMM EXAMPLE
//
// EDUCATIONAL NOTE - INT8 for Inference:
// INT8 quantization converts FP32 weights to 8-bit integers:
//   INT8_value = round((FP32_value - zero_point) / scale)
//   FP32_value = INT8_value * scale + zero_point
//
// Benefits:
//   - 4× memory reduction vs FP32
//   - 4× faster memory bandwidth
//   - ~4× faster compute (624 TOPS vs 156 TFLOPS on A100)
//   - Total: ~16× speedup for inference!
//
// Challenges:
//   - Requires calibration (finding optimal scale/zero-point)
//   - ~1-2% accuracy loss is common (acceptable for most inference)
//   - Sensitive to outliers in weights/activations
//
// When to use INT8:
//   ✓ Production inference (after training)
//   ✓ Edge deployment (lower power, smaller model)
//   ✓ Real-time applications
//   ✗ Training (too much information loss)
//   ✗ When accuracy is critical (medical, financial)
//
///////////////////////////////////////////////////////////////////////////////////////////////////

void run_int8_gemm() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   INT8 GEMM with INT32 Accumulation                            ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;

  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = int32_t;  // INT32 output (accumulator)
  using ElementAccumulator = int32_t;

  using Config = MixedPrecisionConfig<ElementA, ElementB, ElementC, ElementAccumulator>;
  using Gemm = typename Config::Gemm;

  int M = 4096, N = 4096, K = 4096;

  std::cout << "Configuration:" << std::endl;
  std::cout << "  Input A: INT8 (" << sizeof(ElementA) << " byte per element)" << std::endl;
  std::cout << "  Input B: INT8 (" << sizeof(ElementB) << " byte per element)" << std::endl;
  std::cout << "  Accumulator: INT32 (" << sizeof(ElementAccumulator) << " bytes)" << std::endl;
  std::cout << "  Output C: INT32 (" << sizeof(ElementC) << " bytes per element)" << std::endl;
  std::cout << "  Problem size: " << M << "×" << N << "×" << K << std::endl;
  std::cout << "  Memory for inputs: " << ((M*K + K*N) * sizeof(ElementA) / 1e6) << " MB" << std::endl;
  std::cout << "  Memory savings vs FP32: " << (100 * (1 - sizeof(ElementA)/sizeof(float))) << "%" << std::endl;

  // Allocate and initialize
  std::vector<ElementA> h_A(M * K);
  std::vector<ElementB> h_B(K * N);
  std::vector<ElementC> h_C(M * N, 0);

  initialize_matrix_int8(h_A, M * K);
  initialize_matrix_int8(h_B, K * N);

  // Allocate device memory
  cutlass::HostTensor<ElementA, typename Config::LayoutA> tensor_a({M, K});
  cutlass::HostTensor<ElementB, typename Config::LayoutB> tensor_b({K, N});
  cutlass::HostTensor<ElementC, typename Config::LayoutC> tensor_c({M, N});
  cutlass::HostTensor<ElementC, typename Config::LayoutC> tensor_d({M, N});

  tensor_a.host_view().copy_in_device_to_host(h_A.data());
  tensor_b.host_view().copy_in_device_to_host(h_B.data());
  tensor_c.host_view().copy_in_device_to_host(h_C.data());
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();

  // Setup GEMM
  ElementAccumulator alpha = 1;
  ElementAccumulator beta = 0;

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K},
    {tensor_a.device_data(), tensor_a.layout(),
     tensor_b.device_data(), tensor_b.layout()},
    {{alpha, beta},
     tensor_c.device_data(), tensor_c.layout(),
     tensor_d.device_data(), tensor_d.layout()}
  };

  Gemm gemm_op;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Cannot implement INT8 GEMM" << std::endl;
    return;
  }

  status = gemm_op.initialize(arguments, workspace.get());
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed to initialize INT8 GEMM" << std::endl;
    return;
  }

  // Warmup and timing
  gemm_op();
  cudaDeviceSynchronize();

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
  tensor_d.host_view().copy_in_host_to_device(h_C.data());

  double gflops = calculate_gflops(M, N, K, elapsed_ms);

  std::cout << "\nPerformance:" << std::endl;
  std::cout << "  Time: " << elapsed_ms << " ms" << std::endl;
  std::cout << "  Throughput: " << gflops << " GOPS (integer ops)" << std::endl;
  std::cout << "  A100 INT8 peak: 624,000 GOPS (TOPS)" << std::endl;
  std::cout << "  Achieved: " << (100.0 * gflops / 624000.0) << "% of peak" << std::endl;

  // Reference computation
  std::cout << "\nVerifying (INT32 reference)..." << std::endl;
  std::vector<float> h_C_ref(M * N, 0.0f);

  // Manual INT8 reference (simplified - just check a few elements)
  bool spot_check_passed = true;
  for (int i = 0; i < std::min(10, M); ++i) {
    for (int j = 0; j < std::min(10, N); ++j) {
      int32_t sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += static_cast<int32_t>(h_A[i * K + k]) * static_cast<int32_t>(h_B[k * N + j]);
      }
      int idx = i * N + j;
      if (h_C[idx] != sum) {
        std::cout << "  Mismatch at (" << i << "," << j << "): "
                  << "CUTLASS=" << h_C[idx] << ", Expected=" << sum << std::endl;
        spot_check_passed = false;
      }
    }
  }

  std::cout << "Spot check (10×10): " << (spot_check_passed ? "PASSED ✓" : "FAILED ✗") << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// MAIN
//
///////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   CUTLASS Tutorial 06: Mixed Precision GEMM                    ║" << std::endl;
  std::cout << "║   Balancing Speed and Accuracy                                 ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;

  // Run examples
  run_fp16_gemm();
  run_int8_gemm();

  std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║   KEY TAKEAWAYS                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  1. Mixed precision = low-precision inputs + high-precision    ║" << std::endl;
  std::cout << "║     accumulation for best of both worlds                       ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  2. FP16 gives 2-16× speedup over FP32 depending on GPU       ║" << std::endl;
  std::cout << "║     Standard for AI training and inference                     ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  3. INT8 gives 4-8× speedup over FP16                          ║" << std::endl;
  std::cout << "║     Standard for production inference after quantization       ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  4. Accumulator precision prevents catastrophic rounding       ║" << std::endl;
  std::cout << "║     FP16→FP32 or INT8→INT32 are common patterns               ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  5. CollectiveBuilder handles mixed precision automatically    ║" << std::endl;
  std::cout << "║     Just specify element types and let it optimize!            ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║   NEXT STEPS                                                   ║" << std::endl;
  std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  • Next: 07_epilogue_fusion.cu - Fuse operations like         ║" << std::endl;
  std::cout << "║    bias+activation into the epilogue for free!                 ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "║  • Experiment: Try BF16 (bfloat16) for better range           ║" << std::endl;
  std::cout << "║  • Experiment: Try FP8 on Hopper (if available)                ║" << std::endl;
  std::cout << "║                                                                ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;

  return 0;
}
