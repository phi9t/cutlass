// Tier D — Performance: attention throughput measurement.

#include "gtest/gtest.h"
#include "src/attention/attention.h"
#include "src/core/event.h"

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

using namespace gpt;
using namespace gpt::attention;

TEST(AttentionPerfTest, SmokeTiming) {
  auto stream_result = CudaStream::Create();
  ASSERT_TRUE(stream_result.ok());
  CudaStream stream = stream_result.take();

  const int64_t B = 2;
  const int64_t T = 128;
  const int64_t D = 128;
  const int64_t H = 4;
  const int64_t Dh = 32;
  const int iters = 5;

  std::vector<float> h_X(B * T * D);
  std::vector<float> h_W_qkv(3 * D * D);
  std::vector<float> h_b_qkv(3 * D);
  std::vector<float> h_W_o(D * D);
  std::vector<float> h_b_o(D);
  for (size_t i = 0; i < h_X.size(); ++i) {
    h_X[i] = static_cast<float>((i % 11) - 5) * 0.01f;
  }
  for (size_t i = 0; i < h_W_qkv.size(); ++i) {
    h_W_qkv[i] = static_cast<float>((i % 13) - 6) * 0.01f;
  }
  for (size_t i = 0; i < h_W_o.size(); ++i) {
    h_W_o[i] = static_cast<float>((i % 17) - 8) * 0.01f;
  }
  for (size_t i = 0; i < h_b_qkv.size(); ++i) {
    h_b_qkv[i] = static_cast<float>((i % 7) - 3) * 0.001f;
  }
  for (size_t i = 0; i < h_b_o.size(); ++i) {
    h_b_o[i] = static_cast<float>((i % 5) - 2) * 0.001f;
  }

  float* d_X = nullptr;
  float* d_out = nullptr;
  float* d_W_qkv = nullptr;
  float* d_b_qkv = nullptr;
  float* d_W_o = nullptr;
  float* d_b_o = nullptr;
  float* d_qkv = nullptr;
  float* d_Q = nullptr;
  float* d_K = nullptr;
  float* d_V = nullptr;
  float* d_scores = nullptr;
  float* d_probs = nullptr;
  float* d_ctx = nullptr;
  float* d_merged = nullptr;
  cudaMalloc(&d_X, h_X.size() * sizeof(float));
  cudaMalloc(&d_out, B * T * D * sizeof(float));
  cudaMalloc(&d_W_qkv, h_W_qkv.size() * sizeof(float));
  cudaMalloc(&d_b_qkv, h_b_qkv.size() * sizeof(float));
  cudaMalloc(&d_W_o, h_W_o.size() * sizeof(float));
  cudaMalloc(&d_b_o, h_b_o.size() * sizeof(float));
  cudaMalloc(&d_qkv, B * T * 3 * D * sizeof(float));
  cudaMalloc(&d_Q, B * H * T * Dh * sizeof(float));
  cudaMalloc(&d_K, B * H * T * Dh * sizeof(float));
  cudaMalloc(&d_V, B * H * T * Dh * sizeof(float));
  cudaMalloc(&d_scores, B * H * T * T * sizeof(float));
  cudaMalloc(&d_probs, B * H * T * T * sizeof(float));
  cudaMalloc(&d_ctx, B * H * T * Dh * sizeof(float));
  cudaMalloc(&d_merged, B * T * D * sizeof(float));
  cudaMemcpy(d_X, h_X.data(), h_X.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_qkv, h_W_qkv.data(), h_W_qkv.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_qkv, h_b_qkv.data(), h_b_qkv.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_o, h_W_o.data(), h_W_o.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_o, h_b_o.data(), h_b_o.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  AttentionConfig config{D, H, Dh, true, true};
  AttentionParams params;
  params.W_qkv = Tensor2D<float>{d_W_qkv, {3 * D, D}, {D, 1}};
  params.b_qkv = Tensor1D<float>{d_b_qkv, {3 * D}, {1}};
  params.W_o = Tensor2D<float>{d_W_o, {D, D}, {D, 1}};
  params.b_o = Tensor1D<float>{d_b_o, {D}, {1}};

  AttentionForwardState state;
  state.qkv = Tensor3D<float>{d_qkv, {B, T, 3 * D}, {T * 3 * D, 3 * D, 1}};
  state.Q = Tensor4D<float>{d_Q, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  state.K = Tensor4D<float>{d_K, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  state.V = Tensor4D<float>{d_V, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  state.scores = Tensor4D<float>{d_scores, {B, H, T, T},
                                 {H * T * T, T * T, T, 1}};
  state.probs = Tensor4D<float>{d_probs, {B, H, T, T},
                                {H * T * T, T * T, T, 1}};
  state.context = Tensor4D<float>{d_ctx, {B, H, T, Dh},
                                  {H * T * Dh, T * Dh, Dh, 1}};
  state.context_merged = Tensor3D<float>{d_merged, {B, T, D}, {T * D, D, 1}};
  Tensor3D<const float> X{d_X, {B, T, D}, {T * D, D, 1}};
  Tensor3D<float> out{d_out, {B, T, D}, {T * D, D, 1}};

  ASSERT_TRUE(attention_forward(X, config, params, out, state, stream).ok());
  ASSERT_TRUE(stream.synchronize().ok());

  auto start_result = CudaEvent::Create();
  ASSERT_TRUE(start_result.ok());
  auto stop_result = CudaEvent::Create();
  ASSERT_TRUE(stop_result.ok());
  CudaEvent start = start_result.take();
  CudaEvent stop = stop_result.take();
  ASSERT_TRUE(start.record(stream).ok());
  for (int i = 0; i < iters; ++i) {
    ASSERT_TRUE(attention_forward(X, config, params, out, state, stream).ok());
  }
  ASSERT_TRUE(stop.record(stream).ok());
  ASSERT_TRUE(stop.synchronize().ok());
  auto elapsed = CudaEvent::elapsed(start, stop);
  ASSERT_TRUE(elapsed.ok()) << elapsed.status().message();
  const float ms_per_iter = elapsed.value() / static_cast<float>(iters);
  EXPECT_TRUE(std::isfinite(ms_per_iter));
  EXPECT_GT(ms_per_iter, 0.0f);

  cudaFree(d_X);
  cudaFree(d_out);
  cudaFree(d_W_qkv);
  cudaFree(d_b_qkv);
  cudaFree(d_W_o);
  cudaFree(d_b_o);
  cudaFree(d_qkv);
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_scores);
  cudaFree(d_probs);
  cudaFree(d_ctx);
  cudaFree(d_merged);
}
