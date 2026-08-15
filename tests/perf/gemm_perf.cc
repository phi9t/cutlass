// Tier D — Performance: GEMM throughput measurement.

#include "gtest/gtest.h"
#include "src/core/event.h"
#include "src/core/stream.h"
#include "src/kernels/gemm.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

using namespace gpt;

TEST(GemmPerfTest, SmokeTiming) {
  // This test just ensures GEMM can be launched and timed.
  // It does NOT enforce throughput targets.
  auto stream_r = CudaStream::Create();
  ASSERT_TRUE(stream_r.ok());
  auto stream = stream_r.take();

  const int64_t M = 512;
  const int64_t K = 512;
  const int64_t N = 512;
  const int iters = 10;

  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  for (int64_t i = 0; i < M * K; ++i) {
    h_A[static_cast<size_t>(i)] = static_cast<float>((i % 17) - 8) * 0.01f;
  }
  for (int64_t i = 0; i < K * N; ++i) {
    h_B[static_cast<size_t>(i)] = static_cast<float>((i % 13) - 6) * 0.01f;
  }

  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  cudaMalloc(&d_A, h_A.size() * sizeof(float));
  cudaMalloc(&d_B, h_B.size() * sizeof(float));
  cudaMalloc(&d_C, M * N * sizeof(float));
  cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(d_C, 0, M * N * sizeof(float));

  Tensor2D<float> A{d_A, {M, K}, {K, 1}};
  Tensor2D<float> B{d_B, {K, N}, {N, 1}};
  Tensor2D<float> C{d_C, {M, N}, {N, 1}};

  ASSERT_TRUE(kernels::gemm_f32(A, B, C, 1.0f, 0.0f, stream).ok());
  ASSERT_TRUE(stream.synchronize().ok());

  auto start_result = CudaEvent::Create();
  ASSERT_TRUE(start_result.ok());
  auto stop_result = CudaEvent::Create();
  ASSERT_TRUE(stop_result.ok());
  CudaEvent start = start_result.take();
  CudaEvent stop = stop_result.take();
  ASSERT_TRUE(start.record(stream).ok());
  for (int i = 0; i < iters; ++i) {
    ASSERT_TRUE(kernels::gemm_f32(A, B, C, 1.0f, 0.0f, stream).ok());
  }
  ASSERT_TRUE(stop.record(stream).ok());
  ASSERT_TRUE(stop.synchronize().ok());
  auto elapsed = CudaEvent::elapsed(start, stop);
  ASSERT_TRUE(elapsed.ok()) << elapsed.status().message();
  const float ms_per_iter = elapsed.value() / static_cast<float>(iters);
  EXPECT_TRUE(std::isfinite(ms_per_iter));
  EXPECT_GT(ms_per_iter, 0.0f);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
