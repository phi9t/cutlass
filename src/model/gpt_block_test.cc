// Tier B — GPU unit tests for transformer block (GPT block).
//
// Tests cover:
//   - Block forward: compiles and links with full workspace
//   - Block forward: output shape matches input shape
//   - Block backward: returns kNotImplemented (current state)
//   - CPU-only: LayerNorm → Attention → Residual sub-block structure
//   - CPU-only: MLP sub-block (fc1 → GELU → fc2) structure

#include "gtest/gtest.h"
#include "src/model/gpt_block.h"
#include "src/model/gpt_config.h"
#include "src/model/gpt_params.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

using namespace gpt;
using namespace gpt::model;

// --- Config/param structure tests (CPU-only, Tier A) ---

TEST(GPTBlockConfigTest, LayerParamsStructLayout) {
  // Verify LayerParams contains all expected fields.
  LayerParams lp{};

  // LN1 and LN2 should have LayerNormParams.
  EXPECT_EQ(lp.ln1.eps, 1e-5f);
  EXPECT_EQ(lp.ln2.eps, 1e-5f);

  // FC1 and FC2 should have LinearParams with default bias=true.
  EXPECT_TRUE(lp.fc1.use_bias);
  EXPECT_TRUE(lp.fc2.use_bias);
}

TEST(GPTBlockConfigTest, BlockForwardStateHasAllFields) {
  BlockForwardState state{};

  // All tensor fields should have null data by default.
  EXPECT_EQ(state.ln1_out.data, nullptr);
  EXPECT_EQ(state.ln2_out.data, nullptr);
  EXPECT_EQ(state.attn_out.data, nullptr);
  EXPECT_EQ(state.fc1_out.data, nullptr);
  EXPECT_EQ(state.gelu_out.data, nullptr);
  EXPECT_EQ(state.fc2_out.data, nullptr);
}

// --- GPU compile check: block_forward ---

class GPTBlockGPUTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

TEST_F(GPTBlockGPUTest, ForwardCompiles) {
  // Tiny config: B=1, T=2, D=4, H=2, mlp_hidden=8.
  const int64_t B = 1, T = 2, D = 4, H = 2, Dh = 2;
  const int64_t mlp_hidden = 8;

  GPTConfig config;
  config.vocab_size = 16;
  config.max_seq_len = 4;
  config.n_layers = 1;
  config.n_heads = H;
  config.d_model = D;
  config.mlp_hidden = mlp_hidden;
  config.use_bias = true;
  config.layernorm_eps = 1e-5f;

  test::SimpleRng rng(42);

  // Allocate host data.
  std::vector<float> h_x(B * T * D);
  rng.fill(h_x, 1.0f);

  // LayerNorm params.
  std::vector<float> h_gamma1(D), h_beta1(D), h_gamma2(D), h_beta2(D);
  for (int i = 0; i < D; ++i) {
    h_gamma1[i] = 1.0f; h_beta1[i] = 0.0f;
    h_gamma2[i] = 1.0f; h_beta2[i] = 0.0f;
  }

  // Attention weights.
  std::vector<float> h_W_qkv(3 * D * D), h_b_qkv(3 * D);
  std::vector<float> h_W_o(D * D), h_b_o(D);
  rng.fill(h_W_qkv, 0.3f); rng.fill(h_b_qkv, 0.1f);
  rng.fill(h_W_o, 0.3f); rng.fill(h_b_o, 0.1f);

  // MLP weights.
  std::vector<float> h_fc1_w(mlp_hidden * D), h_fc1_b(mlp_hidden);
  std::vector<float> h_fc2_w(D * mlp_hidden), h_fc2_b(D);
  rng.fill(h_fc1_w, 0.3f); rng.fill(h_fc1_b, 0.1f);
  rng.fill(h_fc2_w, 0.3f); rng.fill(h_fc2_b, 0.1f);

  // Allocate device memory.
  auto alloc = [](size_t bytes) {
    float* p = nullptr;
    cudaMalloc(&p, bytes);
    return p;
  };

  float* d_x = alloc(B * T * D * sizeof(float));
  float* d_output = alloc(B * T * D * sizeof(float));

  // LN1 params.
  float* d_g1 = alloc(D * sizeof(float));
  float* d_b1 = alloc(D * sizeof(float));
  float* d_g2 = alloc(D * sizeof(float));
  float* d_b2 = alloc(D * sizeof(float));

  // Attention params.
  float* d_W_qkv = alloc(3 * D * D * sizeof(float));
  float* d_b_qkv = alloc(3 * D * sizeof(float));
  float* d_W_o = alloc(D * D * sizeof(float));
  float* d_b_o = alloc(D * sizeof(float));

  // MLP params.
  float* d_fc1_w = alloc(mlp_hidden * D * sizeof(float));
  float* d_fc1_b = alloc(mlp_hidden * sizeof(float));
  float* d_fc2_w = alloc(D * mlp_hidden * sizeof(float));
  float* d_fc2_b = alloc(D * sizeof(float));

  // State workspace.
  float* d_ln1_out = alloc(B * T * D * sizeof(float));
  float* d_ln2_out = alloc(B * T * D * sizeof(float));
  float* d_ln1_mean = alloc(B * T * sizeof(float));
  float* d_ln1_inv = alloc(B * T * sizeof(float));
  float* d_ln2_mean = alloc(B * T * sizeof(float));
  float* d_ln2_inv = alloc(B * T * sizeof(float));
  float* d_attn_out = alloc(B * T * D * sizeof(float));
  float* d_fc1_out = alloc(B * T * mlp_hidden * sizeof(float));
  float* d_gelu_out = alloc(B * T * mlp_hidden * sizeof(float));
  float* d_fc2_out = alloc(B * T * D * sizeof(float));

  // Attention state workspace.
  float* d_qkv = alloc(B * T * 3 * D * sizeof(float));
  float* d_Q = alloc(B * H * T * Dh * sizeof(float));
  float* d_K = alloc(B * H * T * Dh * sizeof(float));
  float* d_V = alloc(B * H * T * Dh * sizeof(float));
  float* d_scores = alloc(B * H * T * T * sizeof(float));
  float* d_probs = alloc(B * H * T * T * sizeof(float));
  float* d_ctx = alloc(B * H * T * Dh * sizeof(float));
  float* d_merged = alloc(B * T * D * sizeof(float));

  // Copy data to device.
  cudaMemcpy(d_x, h_x.data(), B * T * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g1, h_gamma1.data(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b1, h_beta1.data(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g2, h_gamma2.data(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b2, h_beta2.data(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_qkv, h_W_qkv.data(), 3 * D * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_qkv, h_b_qkv.data(), 3 * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W_o, h_W_o.data(), D * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_o, h_b_o.data(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_w, h_fc1_w.data(), mlp_hidden * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc1_b, h_fc1_b.data(), mlp_hidden * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc2_w, h_fc2_w.data(), D * mlp_hidden * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_fc2_b, h_fc2_b.data(), D * sizeof(float), cudaMemcpyHostToDevice);

  // Build LayerParams.
  LayerParams params;
  params.ln1.gamma = {d_g1, {D}, {1}};
  params.ln1.beta = {d_b1, {D}, {1}};
  params.ln1.eps = 1e-5f;
  params.ln2.gamma = {d_g2, {D}, {1}};
  params.ln2.beta = {d_b2, {D}, {1}};
  params.ln2.eps = 1e-5f;

  params.attn.W_qkv = {d_W_qkv, {3 * D, D}, {D, 1}};
  params.attn.b_qkv = {d_b_qkv, {3 * D}, {1}};
  params.attn.W_o = {d_W_o, {D, D}, {D, 1}};
  params.attn.b_o = {d_b_o, {D}, {1}};

  params.fc1.weight = {d_fc1_w, {mlp_hidden, D}, {D, 1}};
  params.fc1.bias = {d_fc1_b, {mlp_hidden}, {1}};
  params.fc1.use_bias = true;
  params.fc2.weight = {d_fc2_w, {D, mlp_hidden}, {mlp_hidden, 1}};
  params.fc2.bias = {d_fc2_b, {D}, {1}};
  params.fc2.use_bias = true;

  // Build BlockForwardState.
  BlockForwardState state;
  state.ln1_out = {d_ln1_out, {B, T, D}, {T * D, D, 1}};
  state.ln2_out = {d_ln2_out, {B, T, D}, {T * D, D, 1}};
  state.ln1_state.mean = {d_ln1_mean, {B * T}, {1}};
  state.ln1_state.inv_std = {d_ln1_inv, {B * T}, {1}};
  state.ln2_state.mean = {d_ln2_mean, {B * T}, {1}};
  state.ln2_state.inv_std = {d_ln2_inv, {B * T}, {1}};
  state.attn_out = {d_attn_out, {B, T, D}, {T * D, D, 1}};
  state.fc1_out = {d_fc1_out, {B, T, mlp_hidden}, {T * mlp_hidden, mlp_hidden, 1}};
  state.gelu_out = {d_gelu_out, {B, T, mlp_hidden}, {T * mlp_hidden, mlp_hidden, 1}};
  state.fc2_out = {d_fc2_out, {B, T, D}, {T * D, D, 1}};

  // Attention forward state.
  state.attn_state.qkv = {d_qkv, {B, T, 3 * D}, {T * 3 * D, 3 * D, 1}};
  state.attn_state.Q = {d_Q, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  state.attn_state.K = {d_K, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  state.attn_state.V = {d_V, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  state.attn_state.scores = {d_scores, {B, H, T, T}, {H * T * T, T * T, T, 1}};
  state.attn_state.probs = {d_probs, {B, H, T, T}, {H * T * T, T * T, T, 1}};
  state.attn_state.context = {d_ctx, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  state.attn_state.context_merged = {d_merged, {B, T, D}, {T * D, D, 1}};

  // Call block_forward.
  Tensor3D<const float> x{d_x, {B, T, D}, {T * D, D, 1}};
  Tensor3D<float> output{d_output, {B, T, D}, {T * D, D, 1}};

  auto status = block_forward(x, config, params, output, state, stream_);
  ASSERT_TRUE(status.ok()) << status.message();
  stream_.synchronize();

  // Read output and verify it matches the CPU block reference.
  std::vector<float> h_output(B * T * D);
  cudaMemcpy(h_output.data(), d_output, B * T * D * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < h_output.size(); ++i) {
    EXPECT_TRUE(std::isfinite(h_output[i]))
        << "non-finite output at index " << i;
  }

  auto ln1 = test::cpu_layernorm_forward(
      h_x.data(), h_gamma1.data(), h_beta1.data(), config.layernorm_eps,
      B * T, D);
  auto attn = test::cpu_attention_forward(
      ln1.y.data(), h_W_qkv.data(), h_b_qkv.data(), h_W_o.data(), h_b_o.data(),
      B, T, D, H, /*causal=*/true, /*use_bias=*/true);
  std::vector<float> residual1(B * T * D);
  for (int64_t i = 0; i < B * T * D; ++i) {
    residual1[i] = h_x[i] + attn.output[i];
  }
  auto ln2 = test::cpu_layernorm_forward(
      residual1.data(), h_gamma2.data(), h_beta2.data(), config.layernorm_eps,
      B * T, D);
  auto fc1 = test::cpu_linear_forward(
      ln2.y.data(), h_fc1_w.data(), h_fc1_b.data(),
      B * T, D, mlp_hidden, /*use_bias=*/true);
  std::vector<float> gelu(B * T * mlp_hidden);
  for (int64_t i = 0; i < B * T * mlp_hidden; ++i) {
    gelu[i] = test::cpu_gelu(fc1[i]);
  }
  auto fc2 = test::cpu_linear_forward(
      gelu.data(), h_fc2_w.data(), h_fc2_b.data(),
      B * T, mlp_hidden, D, /*use_bias=*/true);
  std::vector<float> expected(B * T * D);
  for (int64_t i = 0; i < B * T * D; ++i) {
    expected[i] = residual1[i] + fc2[i];
  }
  EXPECT_TRUE(test::vectors_near(h_output, expected, 1e-3f, 1e-2f))
      << "block forward output mismatch";

  // Cleanup.
  cudaFree(d_x); cudaFree(d_output);
  cudaFree(d_g1); cudaFree(d_b1); cudaFree(d_g2); cudaFree(d_b2);
  cudaFree(d_W_qkv); cudaFree(d_b_qkv); cudaFree(d_W_o); cudaFree(d_b_o);
  cudaFree(d_fc1_w); cudaFree(d_fc1_b); cudaFree(d_fc2_w); cudaFree(d_fc2_b);
  cudaFree(d_ln1_out); cudaFree(d_ln2_out);
  cudaFree(d_ln1_mean); cudaFree(d_ln1_inv);
  cudaFree(d_ln2_mean); cudaFree(d_ln2_inv);
  cudaFree(d_attn_out); cudaFree(d_fc1_out);
  cudaFree(d_gelu_out); cudaFree(d_fc2_out);
  cudaFree(d_qkv); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
  cudaFree(d_scores); cudaFree(d_probs); cudaFree(d_ctx); cudaFree(d_merged);
}

TEST_F(GPTBlockGPUTest, BackwardReturnsNotImplemented) {
  GPTConfig config;
  config.d_model = 4;
  config.n_heads = 2;
  config.mlp_hidden = 8;

  LayerParams params{};
  BlockForwardState state{};
  GPTGrads::LayerGrads grads{};

  Tensor3D<const float> d_output{nullptr, {1, 2, 4}, {8, 4, 1}};
  Tensor3D<const float> x{nullptr, {1, 2, 4}, {8, 4, 1}};
  Tensor3D<float> dx{nullptr, {1, 2, 4}, {8, 4, 1}};

  auto status = block_backward(d_output, x, config, params, state, dx,
                                grads, stream_);
  EXPECT_EQ(status.code(), StatusCode::kNotImplemented);
}

// --- CPU-only: MLP sub-block structure verification ---

TEST(GPTBlockCPUTest, MLPSubBlockForwardBackward) {
  // Test the MLP sub-block: fc1 → GELU → fc2 using CPU references.
  const int64_t N = 2, D = 4, mlp_hidden = 8;

  test::SimpleRng rng(42);
  std::vector<float> h_x(N * D);
  rng.fill(h_x, 1.0f);

  // fc1: [mlp_hidden, D], fc2: [D, mlp_hidden]
  std::vector<float> h_fc1_w(mlp_hidden * D), h_fc1_b(mlp_hidden);
  std::vector<float> h_fc2_w(D * mlp_hidden), h_fc2_b(D);
  rng.fill(h_fc1_w, 0.3f); rng.fill(h_fc1_b, 0.1f);
  rng.fill(h_fc2_w, 0.3f); rng.fill(h_fc2_b, 0.1f);

  // fc1 forward: [N, D] → [N, mlp_hidden]
  auto fc1_out = test::cpu_linear_forward(h_x.data(), h_fc1_w.data(),
                                           h_fc1_b.data(), N, D, mlp_hidden, true);

  // GELU.
  std::vector<float> gelu_out(N * mlp_hidden);
  for (int64_t i = 0; i < N * mlp_hidden; ++i)
    gelu_out[i] = test::cpu_gelu(fc1_out[i]);

  // fc2 forward: [N, mlp_hidden] → [N, D]
  auto fc2_out = test::cpu_linear_forward(gelu_out.data(), h_fc2_w.data(),
                                           h_fc2_b.data(), N, mlp_hidden, D, true);

  // Output should be finite and have correct shape.
  EXPECT_EQ(static_cast<int64_t>(fc2_out.size()), N * D);
  for (float v : fc2_out) {
    EXPECT_TRUE(std::isfinite(v));
  }

  // Backward through MLP with dY = all-ones.
  std::vector<float> h_dY(N * D, 1.0f);

  // fc2 backward.
  auto fc2_grads = test::cpu_linear_backward(gelu_out.data(), h_dY.data(),
                                              h_fc2_w.data(), N, mlp_hidden, D);

  // GELU backward: dx = d_gelu * gelu'(fc1_out)
  std::vector<float> d_gelu(N * mlp_hidden);
  for (int64_t i = 0; i < N * mlp_hidden; ++i)
    d_gelu[i] = fc2_grads.dX[i] * test::cpu_gelu_backward(fc1_out[i]);

  // fc1 backward.
  auto fc1_grads = test::cpu_linear_backward(h_x.data(), d_gelu.data(),
                                              h_fc1_w.data(), N, D, mlp_hidden);

  // dX should have correct shape and be finite.
  EXPECT_EQ(static_cast<int64_t>(fc1_grads.dX.size()), N * D);
  for (float v : fc1_grads.dX) {
    EXPECT_TRUE(std::isfinite(v));
  }

  // dW should have correct shapes.
  EXPECT_EQ(static_cast<int64_t>(fc1_grads.dW.size()), mlp_hidden * D);
  EXPECT_EQ(static_cast<int64_t>(fc2_grads.dW.size()), D * mlp_hidden);
}
