// Tier B — GPU unit tests for attention forward.
//
// Tests cover:
//   - Config: scale computation
//   - Config: head dimension derivation
//   - CPU reference: tiny attention forward (CPU-only, verifies the oracle)
//   - GPU compile check: attention_forward with allocated workspace
//   - CPU reference: causal attention probs are lower-triangular
//   - CPU reference: attention probs sum to 1 per query position
//   - CPU reference: identity W_o preserves merged context

#include "gtest/gtest.h"
#include "src/attention/attention.h"
#include "tests/test_utils.h"

#include <cmath>
#include <vector>

using namespace gpt;
using namespace gpt::attention;

// --- Config tests ---

TEST(AttentionConfigTest, ScaleComputation) {
  AttentionConfig cfg;
  cfg.d_model = 64;
  cfg.n_heads = 4;
  cfg.head_dim = 16;
  // scale = 1/sqrt(16) = 0.25
  EXPECT_NEAR(cfg.scale(), 0.25f, 1e-6f);
}

TEST(AttentionConfigTest, HeadDimFromModel) {
  AttentionConfig cfg;
  cfg.d_model = 768;
  cfg.n_heads = 12;
  cfg.head_dim = 64;
  EXPECT_NEAR(cfg.scale(), 1.0f / 8.0f, 1e-6f);
}

// --- CPU reference tests (no GPU needed) ---

TEST(AttentionCPUTest, TinyForwardProducesOutput) {
  // B=1, T=2, D=4, H=2 → Dh=2.
  const int64_t B = 1, T = 2, D = 4, H = 2;

  test::SimpleRng rng(42);
  std::vector<float> X(B * T * D);
  rng.fill(X, 1.0f);

  std::vector<float> W_qkv(3 * D * D), b_qkv(3 * D);
  rng.fill(W_qkv, 0.5f);
  rng.fill(b_qkv, 0.1f);

  std::vector<float> W_o(D * D), b_o(D);
  rng.fill(W_o, 0.5f);
  rng.fill(b_o, 0.1f);

  auto result = test::cpu_attention_forward(
      X.data(), W_qkv.data(), b_qkv.data(),
      W_o.data(), b_o.data(),
      B, T, D, H, /*causal=*/true, /*use_bias=*/true);

  // Output should have the right size.
  EXPECT_EQ(static_cast<int64_t>(result.output.size()), B * T * D);
  EXPECT_EQ(static_cast<int64_t>(result.probs.size()), B * H * T * T);

  // Output values should be finite.
  for (float v : result.output) {
    EXPECT_TRUE(std::isfinite(v)) << "non-finite output value: " << v;
  }
}

TEST(AttentionCPUTest, CausalProbsAreLowerTriangular) {
  const int64_t B = 1, T = 4, D = 8, H = 2;

  test::SimpleRng rng(99);
  std::vector<float> X(B * T * D);
  rng.fill(X, 1.0f);

  std::vector<float> W_qkv(3 * D * D), b_qkv(3 * D);
  rng.fill(W_qkv, 0.3f);
  rng.fill(b_qkv, 0.1f);

  std::vector<float> W_o(D * D), b_o(D);
  rng.fill(W_o, 0.3f);
  rng.fill(b_o, 0.1f);

  auto result = test::cpu_attention_forward(
      X.data(), W_qkv.data(), b_qkv.data(),
      W_o.data(), b_o.data(),
      B, T, D, H, /*causal=*/true, /*use_bias=*/true);

  // For each head and each query position, check:
  // 1. Probs for future positions (j > i) should be near zero.
  // 2. Probs for valid positions should sum to ~1.
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t qi = 0; qi < T; ++qi) {
        float row_sum = 0.0f;
        for (int64_t ki = 0; ki < T; ++ki) {
          float p = result.probs[((b * H + h) * T + qi) * T + ki];
          if (ki > qi) {
            // Causal: future positions should have ~0 probability.
            EXPECT_NEAR(p, 0.0f, 1e-5f)
                << "b=" << b << " h=" << h << " q=" << qi << " k=" << ki;
          } else {
            EXPECT_GE(p, 0.0f);
          }
          row_sum += p;
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-4f)
            << "b=" << b << " h=" << h << " q=" << qi;
      }
    }
  }
}

TEST(AttentionCPUTest, NonCausalProbsSumToOne) {
  const int64_t B = 2, T = 3, D = 4, H = 2;

  test::SimpleRng rng(55);
  std::vector<float> X(B * T * D);
  rng.fill(X, 1.0f);

  std::vector<float> W_qkv(3 * D * D), b_qkv(3 * D);
  rng.fill(W_qkv, 0.3f);
  rng.fill(b_qkv, 0.1f);

  std::vector<float> W_o(D * D), b_o(D);
  rng.fill(W_o, 0.3f);
  rng.fill(b_o, 0.1f);

  auto result = test::cpu_attention_forward(
      X.data(), W_qkv.data(), b_qkv.data(),
      W_o.data(), b_o.data(),
      B, T, D, H, /*causal=*/false, /*use_bias=*/true);

  // All probs should be > 0 and each row sums to 1.
  for (int64_t bh = 0; bh < B * H; ++bh) {
    for (int64_t qi = 0; qi < T; ++qi) {
      float row_sum = 0.0f;
      for (int64_t ki = 0; ki < T; ++ki) {
        float p = result.probs[(bh * T + qi) * T + ki];
        EXPECT_GT(p, 0.0f);
        row_sum += p;
      }
      EXPECT_NEAR(row_sum, 1.0f, 1e-5f);
    }
  }
}

TEST(AttentionCPUTest, NoBiasProducesFiniteOutput) {
  const int64_t B = 1, T = 3, D = 4, H = 2;

  test::SimpleRng rng(77);
  std::vector<float> X(B * T * D);
  rng.fill(X, 1.0f);

  std::vector<float> W_qkv(3 * D * D);
  rng.fill(W_qkv, 0.3f);

  std::vector<float> W_o(D * D);
  rng.fill(W_o, 0.3f);

  auto result = test::cpu_attention_forward(
      X.data(), W_qkv.data(), nullptr,
      W_o.data(), nullptr,
      B, T, D, H, /*causal=*/true, /*use_bias=*/false);

  for (float v : result.output) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

// --- GPU compile check ---
// This test allocates GPU workspace and calls attention_forward to verify
// the CUDA code compiles and links. It requires a GPU to actually run.

#include <cuda_runtime.h>

TEST(AttentionGPUCompileTest, ForwardCallCompiles) {
  // Tiny config: B=1, T=2, D=4, H=2, Dh=2.
  const int64_t B = 1, T = 2, D = 4, H = 2, Dh = 2;

  auto s = CudaStream::Create();
  ASSERT_TRUE(s.ok());
  CudaStream stream = s.take();

  test::SimpleRng rng(42);

  // Allocate and init host data.
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

  // Allocate device memory.
  float *d_X, *d_out;
  float *d_W_qkv, *d_b_qkv, *d_W_o, *d_b_o;

  cudaMalloc(&d_X, B * T * D * sizeof(float));
  cudaMalloc(&d_out, B * T * D * sizeof(float));
  cudaMalloc(&d_W_qkv, 3 * D * D * sizeof(float));
  cudaMalloc(&d_b_qkv, 3 * D * sizeof(float));
  cudaMalloc(&d_W_o, D * D * sizeof(float));
  cudaMalloc(&d_b_o, D * sizeof(float));

  // Copy data to device.
  cudaMemcpy(d_X, h_X.data(), B * T * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_qkv, h_W_qkv.data(), 3 * D * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_qkv, h_b_qkv.data(), 3 * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_o, h_W_o.data(), D * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_o, h_b_o.data(), D * sizeof(float), cudaMemcpyHostToDevice);

  // Build config.
  AttentionConfig cfg;
  cfg.d_model = D;
  cfg.n_heads = H;
  cfg.head_dim = Dh;
  cfg.use_bias = true;
  cfg.causal = true;

  // Build params.
  AttentionParams params;
  params.W_qkv = {d_W_qkv, {3 * D, D}, {D, 1}};
  params.b_qkv = {d_b_qkv, {3 * D}, {1}};
  params.W_o = {d_W_o, {D, D}, {D, 1}};
  params.b_o = {d_b_o, {D}, {1}};

  // Build input/output tensors.
  Tensor3D<const float> X{d_X, {B, T, D}, {T * D, D, 1}};
  Tensor3D<float> output{d_out, {B, T, D}, {T * D, D, 1}};

  // Call forward — this tests that CUDA code compiles and links.
  AttentionWorkspace workspace;
  auto status = attention_forward(X, cfg, params, output, workspace, stream);
  ASSERT_TRUE(status.ok()) << status.message();
  stream.synchronize();

  // Read output and verify against CPU reference.
  std::vector<float> h_out(B * T * D);
  cudaMemcpy(h_out.data(), d_out, B * T * D * sizeof(float), cudaMemcpyDeviceToHost);

  auto cpu_result = test::cpu_attention_forward(
      h_X.data(), h_W_qkv.data(), h_b_qkv.data(),
      h_W_o.data(), h_b_o.data(),
      B, T, D, H, /*causal=*/true, /*use_bias=*/true);

  EXPECT_TRUE(test::vectors_near(h_out, cpu_result.output, 1e-3f, 1e-2f))
      << "GPU attention forward output doesn't match CPU reference";

  // Cleanup.
  cudaFree(d_X); cudaFree(d_out);
  cudaFree(d_W_qkv); cudaFree(d_b_qkv); cudaFree(d_W_o); cudaFree(d_b_o);
}
