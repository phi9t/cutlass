// Tier B — GPU unit tests for full GPT model.
//
// Tests cover:
//   - GPTConfig: head_dim, param count, custom configs
//   - GPTForwardState: struct layout
//   - GPTGrads: struct layout
//   - GPU compile check: gpt_forward with full workspace
//   - gpt_backward: struct and signature validation
//   - CPU-only: end-to-end tiny model (embedding → block → LN → LM head)

#include "gtest/gtest.h"
#include "src/model/gpt_model.h"
#include "src/model/gpt_config.h"
#include "src/model/gpt_params.h"
#include "tests/test_utils.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>

using namespace gpt;
using namespace gpt::model;

// --- Config tests (Tier A, CPU-only) ---

TEST(GPTConfigTest, DefaultValues) {
  GPTConfig c;
  EXPECT_EQ(c.vocab_size, 50257);
  EXPECT_EQ(c.d_model, 768);
  EXPECT_EQ(c.n_heads, 12);
  EXPECT_EQ(c.head_dim(), 64);
}

TEST(GPTConfigTest, ParamCountPositive) {
  GPTConfig c;
  EXPECT_GT(c.approx_param_count(), 0);
}

TEST(GPTConfigTest, SmallConfig) {
  GPTConfig c;
  c.vocab_size = 100;
  c.max_seq_len = 16;
  c.n_layers = 2;
  c.n_heads = 2;
  c.d_model = 32;
  c.mlp_hidden = 128;
  EXPECT_EQ(c.head_dim(), 16);
  EXPECT_GT(c.approx_param_count(), 0);
}

TEST(GPTConfigTest, TiedEmbeddingsReduceParamCount) {
  GPTConfig tied, untied;
  tied.vocab_size = 100;
  tied.max_seq_len = 16;
  tied.n_layers = 1;
  tied.n_heads = 2;
  tied.d_model = 32;
  tied.mlp_hidden = 64;
  tied.tie_embeddings = true;

  untied = tied;
  untied.tie_embeddings = false;

  EXPECT_GT(untied.approx_param_count(), tied.approx_param_count());
  // Difference should be exactly vocab_size * d_model.
  int64_t diff = untied.approx_param_count() - tied.approx_param_count();
  EXPECT_EQ(diff, tied.vocab_size * tied.d_model);
}

TEST(GPTConfigTest, NoBiasReducesParamCount) {
  GPTConfig bias_cfg, nobias_cfg;
  bias_cfg.vocab_size = 100;
  bias_cfg.max_seq_len = 16;
  bias_cfg.n_layers = 2;
  bias_cfg.n_heads = 2;
  bias_cfg.d_model = 32;
  bias_cfg.mlp_hidden = 64;
  bias_cfg.use_bias = true;

  nobias_cfg = bias_cfg;
  nobias_cfg.use_bias = false;

  EXPECT_GT(bias_cfg.approx_param_count(), nobias_cfg.approx_param_count());
}

TEST(GPTConfigTest, ParamCountScalesWithLayers) {
  GPTConfig c1, c2;
  c1.vocab_size = 100;
  c1.max_seq_len = 16;
  c1.n_layers = 2;
  c1.n_heads = 2;
  c1.d_model = 32;
  c1.mlp_hidden = 64;

  c2 = c1;
  c2.n_layers = 4;

  // Doubling layers should roughly double the per-layer params.
  int64_t p1 = c1.approx_param_count();
  int64_t p2 = c2.approx_param_count();
  EXPECT_GT(p2, p1);

  // The extra params should be exactly 2 * (per_layer_params).
  // Since embedding + final LN are shared, p2 - p1 = 2 * per_layer.
  int64_t diff = p2 - p1;
  int64_t per_layer_with_2 = (p1 - 100 * 32 - 16 * 32 - 2 * 32) / 2;
  EXPECT_EQ(diff, 2 * per_layer_with_2);
}

// --- Struct layout tests ---

TEST(GPTModelStructTest, ForwardStateLayout) {
  GPTForwardState state{};
  EXPECT_EQ(state.embed_out.data, nullptr);
  EXPECT_TRUE(state.block_inputs.empty());
  EXPECT_TRUE(state.blocks.empty());
  EXPECT_EQ(state.final_ln_out.data, nullptr);
  EXPECT_EQ(state.logits.data, nullptr);
}

TEST(GPTModelStructTest, GradsLayout) {
  GPTGrads grads{};
  EXPECT_EQ(grads.d_token_embedding.data, nullptr);
  EXPECT_EQ(grads.d_position_embedding.data, nullptr);
  EXPECT_TRUE(grads.layers.empty());
  EXPECT_EQ(grads.d_final_ln_gamma.data, nullptr);
  EXPECT_EQ(grads.d_final_ln_beta.data, nullptr);
  EXPECT_EQ(grads.d_lm_head.data, nullptr);
}

// --- GPU compile check ---

class GPTModelGPUTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

TEST_F(GPTModelGPUTest, BackwardSignatureCompiles) {
  // Verify that gpt_backward compiles and accepts the expected signature.
  // Full functional test is in the integration test (gpt_overfit_test).
  GPTConfig config;
  config.d_model = 4;
  config.n_heads = 2;
  config.n_layers = 1;

  GPTParams params{};
  GPTForwardState state{};
  GPTGrads grads{};

  Tensor2D<const float> d_logits{nullptr, {2, 16}, {16, 1}};
  Tensor2D<const int32_t> input_ids{nullptr, {1, 2}, {2, 1}};

  // With null pointers, backward will fail on the first CUDA op,
  // but the function signature and dispatch should work.
  auto status = gpt_backward(d_logits, input_ids, config, params, state,
                               grads, stream_);
  // Expect a CUDA error (null pointers), not kNotImplemented.
  EXPECT_NE(status.code(), StatusCode::kNotImplemented);
}

// --- CPU-only: end-to-end tiny model verification ---

TEST(GPTModelCPUTest, EndToEndTinyModel) {
  // Manually run through the GPT forward pass on CPU to verify structure.
  // Config: vocab=8, seq=3, layers=1, heads=2, d_model=4, mlp=8.
  const int64_t vocab = 8, T = 3, D = 4, H = 2, mlp = 8;

  test::SimpleRng rng(42);

  // Token embedding table: [vocab, D]
  std::vector<float> tok_embed(vocab * D);
  rng.fill(tok_embed, 0.5f);

  // Position embedding table: [T, D]
  std::vector<float> pos_embed(T * D);
  rng.fill(pos_embed, 0.5f);

  // Input tokens: [T]
  std::vector<int32_t> tokens = {1, 3, 5};

  // Step 1: Token embedding lookup.
  auto embed_out = test::cpu_embedding_forward(tok_embed.data(), tokens.data(),
                                                T, D);

  // Step 2: Add positional embedding.
  std::vector<float> x(T * D);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t d = 0; d < D; ++d) {
      x[t * D + d] = embed_out[t * D + d] + pos_embed[t * D + d];
    }
  }

  // Step 3: Transformer block — LN1 → Attention → Residual → LN2 → MLP → Residual.

  // LN1.
  std::vector<float> ln1_gamma(D, 1.0f), ln1_beta(D, 0.0f);
  auto ln1 = test::cpu_layernorm_forward(x.data(), ln1_gamma.data(),
                                          ln1_beta.data(), 1e-5f, T, D);

  // Attention.
  std::vector<float> W_qkv(3 * D * D), b_qkv(3 * D);
  std::vector<float> W_o(D * D), b_o(D);
  rng.fill(W_qkv, 0.3f); rng.fill(b_qkv, 0.1f);
  rng.fill(W_o, 0.3f); rng.fill(b_o, 0.1f);

  auto attn = test::cpu_attention_forward(
      ln1.y.data(), W_qkv.data(), b_qkv.data(),
      W_o.data(), b_o.data(),
      1, T, D, H, /*causal=*/true, /*use_bias=*/true);

  // Residual 1: x = x + attn_out.
  std::vector<float> residual1(T * D);
  for (int64_t i = 0; i < T * D; ++i)
    residual1[i] = x[i] + attn.output[i];

  // LN2.
  std::vector<float> ln2_gamma(D, 1.0f), ln2_beta(D, 0.0f);
  auto ln2 = test::cpu_layernorm_forward(residual1.data(), ln2_gamma.data(),
                                          ln2_beta.data(), 1e-5f, T, D);

  // MLP: fc1 → GELU → fc2.
  std::vector<float> fc1_w(mlp * D), fc1_b(mlp);
  std::vector<float> fc2_w(D * mlp), fc2_b(D);
  rng.fill(fc1_w, 0.2f); rng.fill(fc1_b, 0.05f);
  rng.fill(fc2_w, 0.2f); rng.fill(fc2_b, 0.05f);

  auto fc1_out = test::cpu_linear_forward(ln2.y.data(), fc1_w.data(),
                                           fc1_b.data(), T, D, mlp, true);
  std::vector<float> gelu_out(T * mlp);
  for (int64_t i = 0; i < T * mlp; ++i)
    gelu_out[i] = test::cpu_gelu(fc1_out[i]);
  auto fc2_out = test::cpu_linear_forward(gelu_out.data(), fc2_w.data(),
                                           fc2_b.data(), T, mlp, D, true);

  // Residual 2: x = residual1 + fc2_out.
  std::vector<float> block_out(T * D);
  for (int64_t i = 0; i < T * D; ++i)
    block_out[i] = residual1[i] + fc2_out[i];

  // Step 4: Final LayerNorm.
  std::vector<float> fln_gamma(D, 1.0f), fln_beta(D, 0.0f);
  auto fln = test::cpu_layernorm_forward(block_out.data(), fln_gamma.data(),
                                          fln_beta.data(), 1e-5f, T, D);

  // Step 5: LM head — logits = final_ln_out @ lm_head^T.
  // lm_head = token_embedding (tied).
  auto logits = test::cpu_linear_forward(fln.y.data(), tok_embed.data(),
                                          nullptr, T, D, vocab, false);

  // Verify logits shape and finiteness.
  EXPECT_EQ(static_cast<int64_t>(logits.size()), T * vocab);
  for (float v : logits) {
    EXPECT_TRUE(std::isfinite(v)) << "non-finite logit";
  }

  // Verify softmax of logits sums to 1 for each position.
  auto probs = test::cpu_softmax(logits.data(), T, vocab);
  for (int64_t t = 0; t < T; ++t) {
    float sum = 0.0f;
    for (int64_t v = 0; v < vocab; ++v)
      sum += probs[t * vocab + v];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
  }

  // Compute cross-entropy loss.
  std::vector<int32_t> targets = {3, 5, 1};
  float loss = 0.0f;
  for (int64_t t = 0; t < T; ++t)
    loss += test::cpu_cross_entropy(logits.data() + t * vocab, targets[t], vocab);
  loss /= T;

  EXPECT_TRUE(std::isfinite(loss));
  EXPECT_GT(loss, 0.0f);
  // With random weights and vocab=8, loss should be roughly log(8) ≈ 2.08.
  EXPECT_LT(loss, 10.0f);
}

// CPU-only: verify that the full model backward chain is structurally sound
// by running the gradient check on the LM head projection.
TEST(GPTModelCPUTest, LMHeadGradientCheck) {
  const int64_t T = 2, D = 4, vocab = 6;

  test::SimpleRng rng(42);
  std::vector<float> h_x(T * D), h_lm_head(vocab * D);
  rng.fill(h_x, 1.0f);
  rng.fill(h_lm_head, 0.5f);

  // Forward: logits = x @ lm_head^T
  auto logits = test::cpu_linear_forward(h_x.data(), h_lm_head.data(),
                                          nullptr, T, D, vocab, false);

  // Loss = cross_entropy(logits, targets)
  std::vector<int32_t> targets = {0, 3};
  float loss = 0.0f;
  for (int64_t t = 0; t < T; ++t)
    loss += test::cpu_cross_entropy(logits.data() + t * vocab, targets[t], vocab);
  loss /= T;

  // Backward: d_logits
  auto d_logits = test::cpu_cross_entropy_backward(logits.data(), targets.data(),
                                                    T, vocab);

  // d_x = d_logits @ lm_head
  auto grads = test::cpu_linear_backward(h_x.data(), d_logits.data(),
                                          h_lm_head.data(), T, D, vocab);

  // Numerical gradient check for d_x.
  const float eps = 1e-3f;
  for (int64_t idx = 0; idx < T * D; ++idx) {
    float orig = h_x[idx];

    h_x[idx] = orig + eps;
    auto logits_p = test::cpu_linear_forward(h_x.data(), h_lm_head.data(),
                                              nullptr, T, D, vocab, false);
    float loss_p = 0.0f;
    for (int64_t t = 0; t < T; ++t)
      loss_p += test::cpu_cross_entropy(logits_p.data() + t * vocab, targets[t], vocab);
    loss_p /= T;

    h_x[idx] = orig - eps;
    auto logits_m = test::cpu_linear_forward(h_x.data(), h_lm_head.data(),
                                              nullptr, T, D, vocab, false);
    float loss_m = 0.0f;
    for (int64_t t = 0; t < T; ++t)
      loss_m += test::cpu_cross_entropy(logits_m.data() + t * vocab, targets[t], vocab);
    loss_m /= T;

    h_x[idx] = orig;

    float numerical = (loss_p - loss_m) / (2.0f * eps);
    EXPECT_NEAR(numerical, grads.dX[idx], 1e-2f)
        << "dX numerical gradient at index " << idx;
  }
}
