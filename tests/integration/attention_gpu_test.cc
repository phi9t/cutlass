// Tier C — Integration: attention forward/backward GPU test.

#include "gtest/gtest.h"
#include "src/attention/attention.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

using namespace gpt;
using namespace gpt::attention;

namespace {

// Helper: run attention forward on GPU and compare against CPU reference.
// Returns GPU output and probs for additional verification.
struct GPUAttentionResult {
  std::vector<float> output;
  std::vector<float> probs;
};

GPUAttentionResult run_gpu_attention(int64_t B, int64_t T, int64_t D, int64_t H,
                                     bool causal, bool use_bias,
                                     uint32_t seed = 42) {
  int64_t Dh = D / H;

  auto s = CudaStream::Create();
  EXPECT_TRUE(s.ok());
  CudaStream stream = s.take();

  test::SimpleRng rng(seed);

  std::vector<float> h_X(B * T * D);
  rng.fill(h_X, 1.0f);
  std::vector<float> h_W_qkv(3 * D * D);
  rng.fill(h_W_qkv, 0.5f);
  std::vector<float> h_b_qkv(3 * D);
  rng.fill(h_b_qkv, 0.1f);
  std::vector<float> h_W_o(D * D);
  rng.fill(h_W_o, 0.5f);
  std::vector<float> h_b_o(D);
  rng.fill(h_b_o, 0.1f);

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
  cfg.use_bias = use_bias;
  cfg.causal = causal;

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

  auto status = attention_forward(X, cfg, params, output, state, stream);
  EXPECT_TRUE(status.ok()) << status.message();
  stream.synchronize();

  std::vector<float> h_out(B * T * D);
  std::vector<float> h_probs(B * H * T * T);
  cudaMemcpy(h_out.data(), d_out, B * T * D * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_probs.data(), d_probs, B * H * T * T * sizeof(float), cudaMemcpyDeviceToHost);

  // Compare against CPU reference.
  auto cpu_result = test::cpu_attention_forward(
      h_X.data(), h_W_qkv.data(),
      use_bias ? h_b_qkv.data() : nullptr,
      h_W_o.data(),
      use_bias ? h_b_o.data() : nullptr,
      B, T, D, H, causal, use_bias);

  EXPECT_TRUE(test::vectors_near(h_out, cpu_result.output, 1e-3f, 1e-2f))
      << "GPU attention forward output doesn't match CPU reference";

  cudaFree(d_X); cudaFree(d_out);
  cudaFree(d_W_qkv); cudaFree(d_b_qkv); cudaFree(d_W_o); cudaFree(d_b_o);
  cudaFree(d_qkv); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
  cudaFree(d_scores); cudaFree(d_probs); cudaFree(d_ctx); cudaFree(d_merged);

  return {h_out, h_probs};
}

}  // namespace

TEST(AttentionIntegrationTest, CausalSmall) {
  // B=2, T=4, D=32, H=4 — causal attention with bias.
  auto result = run_gpu_attention(2, 4, 32, 4, /*causal=*/true, /*use_bias=*/true);

  // Verify probability row sums = 1.
  const int64_t B = 2, H = 4, T = 4;
  for (int64_t bh = 0; bh < B * H; ++bh) {
    for (int64_t qi = 0; qi < T; ++qi) {
      float row_sum = 0.0f;
      for (int64_t ki = 0; ki < T; ++ki) {
        float p = result.probs[(bh * T + qi) * T + ki];
        if (ki > qi) {
          EXPECT_NEAR(p, 0.0f, 1e-5f)
              << "causal: future position has nonzero prob at bh=" << bh
              << " q=" << qi << " k=" << ki;
        }
        row_sum += p;
      }
      EXPECT_NEAR(row_sum, 1.0f, 1e-4f)
          << "prob row sum != 1 at bh=" << bh << " q=" << qi;
    }
  }
}

TEST(AttentionIntegrationTest, NonCausalSmall) {
  // B=1, T=4, D=16, H=2 — non-causal attention with bias.
  auto result = run_gpu_attention(1, 4, 16, 2, /*causal=*/false, /*use_bias=*/true);

  const int64_t B = 1, H = 2, T = 4;
  for (int64_t bh = 0; bh < B * H; ++bh) {
    for (int64_t qi = 0; qi < T; ++qi) {
      float row_sum = 0.0f;
      for (int64_t ki = 0; ki < T; ++ki) {
        float p = result.probs[(bh * T + qi) * T + ki];
        EXPECT_GT(p, 0.0f) << "non-causal: all positions should have nonzero prob";
        row_sum += p;
      }
      EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
    }
  }
}

TEST(AttentionIntegrationTest, NoBias) {
  // B=1, T=3, D=8, H=2 — causal, no bias.
  auto result = run_gpu_attention(1, 3, 8, 2, /*causal=*/true, /*use_bias=*/false, 77);

  for (float v : result.output) {
    EXPECT_TRUE(std::isfinite(v)) << "non-finite output: " << v;
  }
}
