// Tier D — Performance: attention throughput measurement.

#include "gtest/gtest.h"
#include "src/attention/attention.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <cstdio>

using namespace gpt;
using namespace gpt::attention;

TEST(AttentionPerfTest, SmokeTiming) {
  // Scaled-down representative shape for CI.
  const int64_t B = 4, T = 64, D = 128, H = 4, Dh = D / H;

  auto s = CudaStream::Create();
  ASSERT_TRUE(s.ok());
  CudaStream stream = s.take();

  test::SimpleRng rng(42);

  std::vector<float> h_X(B * T * D);
  rng.fill(h_X, 1.0f);
  std::vector<float> h_W_qkv(3 * D * D);
  rng.fill(h_W_qkv, 0.3f);
  std::vector<float> h_b_qkv(3 * D);
  rng.fill(h_b_qkv, 0.1f);
  std::vector<float> h_W_o(D * D);
  rng.fill(h_W_o, 0.3f);
  std::vector<float> h_b_o(D);
  rng.fill(h_b_o, 0.1f);

  // Allocate device memory.
  float *d_X, *d_out;
  float *d_W_qkv, *d_b_qkv, *d_W_o, *d_b_o;
  float *d_qkv, *d_Q, *d_K, *d_V, *d_scores, *d_probs, *d_ctx, *d_merged;

  cudaMalloc(&d_X, B * T * D * sizeof(float));
  cudaMalloc(&d_out, B * T * D * sizeof(float));
  cudaMalloc(&d_W_qkv, 3 * D * D * sizeof(float));
  cudaMalloc(&d_b_qkv, 3 * D * sizeof(float));
  cudaMalloc(&d_W_o, D * D * sizeof(float));
  cudaMalloc(&d_b_o, D * sizeof(float));
  cudaMalloc(&d_qkv, B * T * 3 * D * sizeof(float));
  cudaMalloc(&d_Q, B * H * T * Dh * sizeof(float));
  cudaMalloc(&d_K, B * H * T * Dh * sizeof(float));
  cudaMalloc(&d_V, B * H * T * Dh * sizeof(float));
  cudaMalloc(&d_scores, B * H * T * T * sizeof(float));
  cudaMalloc(&d_probs, B * H * T * T * sizeof(float));
  cudaMalloc(&d_ctx, B * H * T * Dh * sizeof(float));
  cudaMalloc(&d_merged, B * T * D * sizeof(float));

  cudaMemcpy(d_X, h_X.data(), B * T * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_qkv, h_W_qkv.data(), 3 * D * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_qkv, h_b_qkv.data(), 3 * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_o, h_W_o.data(), D * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_o, h_b_o.data(), D * sizeof(float), cudaMemcpyHostToDevice);

  AttentionConfig cfg;
  cfg.d_model = D;
  cfg.n_heads = H;
  cfg.head_dim = Dh;
  cfg.use_bias = true;
  cfg.causal = true;

  AttentionParams params;
  params.W_qkv = {d_W_qkv, {3 * D, D}, {D, 1}};
  params.b_qkv = {d_b_qkv, {3 * D}, {1}};
  params.W_o = {d_W_o, {D, D}, {D, 1}};
  params.b_o = {d_b_o, {D}, {1}};

  AttentionForwardState state;
  state.qkv = {d_qkv, {B, T, 3 * D}, {T * 3 * D, 3 * D, 1}};
  state.Q = {d_Q, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  state.K = {d_K, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  state.V = {d_V, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  state.scores = {d_scores, {B, H, T, T}, {H * T * T, T * T, T, 1}};
  state.probs = {d_probs, {B, H, T, T}, {H * T * T, T * T, T, 1}};
  state.context = {d_ctx, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  state.context_merged = {d_merged, {B, T, D}, {T * D, D, 1}};

  Tensor3D<const float> X{d_X, {B, T, D}, {T * D, D, 1}};
  Tensor3D<float> output{d_out, {B, T, D}, {T * D, D, 1}};

  // Warmup.
  auto status = attention_forward(X, cfg, params, output, state, stream);
  ASSERT_TRUE(status.ok()) << status.message();
  stream.synchronize();

  // Timed run.
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int kIters = 10;
  cudaEventRecord(start, stream.get());
  for (int i = 0; i < kIters; ++i) {
    status = attention_forward(X, cfg, params, output, state, stream);
    ASSERT_TRUE(status.ok());
  }
  cudaEventRecord(stop, stream.get());
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_ms = ms / static_cast<float>(kIters);

  printf("Attention forward (B=%ld, T=%ld, D=%ld, H=%ld): %.3f ms avg over %d iters\n",
         B, T, D, H, avg_ms, kIters);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_X); cudaFree(d_out);
  cudaFree(d_W_qkv); cudaFree(d_b_qkv); cudaFree(d_W_o); cudaFree(d_b_o);
  cudaFree(d_qkv); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
  cudaFree(d_scores); cudaFree(d_probs); cudaFree(d_ctx); cudaFree(d_merged);
}
