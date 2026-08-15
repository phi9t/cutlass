// Tier B — GPU unit tests for full GPT model.
//
// Tests cover:
//   - GPTConfig: head_dim, param count, custom configs
//   - GPTForwardState: struct layout
//   - GPTGrads: struct layout
//   - GPU compile check: gpt_forward with full workspace
//   - gpt_backward: zero-layer model computes embedding, final LN, and LM grads
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

TEST_F(GPTModelGPUTest, ForwardZeroLayerMatchesCPUReference) {
  const int64_t B = 1, T = 3, D = 4, vocab = 8;

  GPTConfig config;
  config.vocab_size = vocab;
  config.max_seq_len = T;
  config.n_layers = 0;
  config.n_heads = 2;
  config.d_model = D;
  config.mlp_hidden = 8;
  config.use_bias = true;
  config.tie_embeddings = true;
  config.layernorm_eps = 1e-5f;

  test::SimpleRng rng(42);
  std::vector<int32_t> h_ids = {1, 3, 5};
  std::vector<float> h_tok(vocab * D), h_pos(T * D);
  std::vector<float> h_gamma(D, 1.0f), h_beta(D, 0.0f);
  rng.fill(h_tok, 0.5f);
  rng.fill(h_pos, 0.5f);

  int32_t* d_ids = nullptr;
  float *d_tok = nullptr, *d_pos = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
  float *d_embed = nullptr, *d_mean = nullptr, *d_inv = nullptr;
  float *d_final = nullptr, *d_logits = nullptr;
  cudaMalloc(&d_ids, B * T * sizeof(int32_t));
  cudaMalloc(&d_tok, vocab * D * sizeof(float));
  cudaMalloc(&d_pos, T * D * sizeof(float));
  cudaMalloc(&d_gamma, D * sizeof(float));
  cudaMalloc(&d_beta, D * sizeof(float));
  cudaMalloc(&d_embed, B * T * D * sizeof(float));
  cudaMalloc(&d_mean, B * T * sizeof(float));
  cudaMalloc(&d_inv, B * T * sizeof(float));
  cudaMalloc(&d_final, B * T * D * sizeof(float));
  cudaMalloc(&d_logits, B * T * vocab * sizeof(float));
  cudaMemcpy(d_ids, h_ids.data(), B * T * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tok, h_tok.data(), vocab * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos, h_pos.data(), T * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, h_gamma.data(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, h_beta.data(), D * sizeof(float), cudaMemcpyHostToDevice);

  GPTParams params;
  params.token_embedding = {d_tok, {vocab, D}, {D, 1}};
  params.position_embedding = {d_pos, {T, D}, {D, 1}};
  params.final_ln.gamma = {d_gamma, {D}, {1}};
  params.final_ln.beta = {d_beta, {D}, {1}};
  params.final_ln.eps = config.layernorm_eps;
  params.lm_head = params.token_embedding;

  GPTForwardState state;
  state.embed_out = {d_embed, {B, T, D}, {T * D, D, 1}};
  state.final_ln_out = {d_final, {B, T, D}, {T * D, D, 1}};
  state.final_ln_state.mean = {d_mean, {B * T}, {1}};
  state.final_ln_state.inv_std = {d_inv, {B * T}, {1}};
  state.logits = {d_logits, {B * T, vocab}, {vocab, 1}};

  Tensor2D<const int32_t> input_ids{d_ids, {B, T}, {T, 1}};
  ASSERT_TRUE(gpt_forward(input_ids, config, params, state, stream_).ok());
  stream_.synchronize();

  std::vector<float> h_logits(B * T * vocab);
  cudaMemcpy(h_logits.data(), d_logits, B * T * vocab * sizeof(float),
             cudaMemcpyDeviceToHost);

  auto tok = test::cpu_embedding_forward(h_tok.data(), h_ids.data(), B * T, D);
  std::vector<float> embed(B * T * D);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t d = 0; d < D; ++d) {
        int64_t out_idx = (b * T + t) * D + d;
        embed[out_idx] = tok[out_idx] + h_pos[t * D + d];
      }
    }
  }
  auto ln = test::cpu_layernorm_forward(
      embed.data(), h_gamma.data(), h_beta.data(), config.layernorm_eps, B * T, D);
  auto expected = test::cpu_linear_forward(
      ln.y.data(), h_tok.data(), nullptr, B * T, D, vocab, false);
  EXPECT_TRUE(test::vectors_near(h_logits, expected, 1e-3f, 1e-2f))
      << "gpt_forward logits mismatch";

  cudaFree(d_ids);
  cudaFree(d_tok); cudaFree(d_pos); cudaFree(d_gamma); cudaFree(d_beta);
  cudaFree(d_embed); cudaFree(d_mean); cudaFree(d_inv);
  cudaFree(d_final); cudaFree(d_logits);
}

TEST_F(GPTModelGPUTest, BackwardZeroLayerComputesGradients) {
  const int64_t B = 1, T = 3, D = 4, vocab = 8;

  GPTConfig config;
  config.vocab_size = vocab;
  config.max_seq_len = T;
  config.n_layers = 0;
  config.n_heads = 2;
  config.d_model = D;
  config.mlp_hidden = 8;
  config.use_bias = true;
  config.tie_embeddings = false;
  config.layernorm_eps = 1e-5f;

  test::SimpleRng rng(17);
  std::vector<int32_t> h_ids = {1, 3, 5};
  std::vector<float> h_tok(vocab * D), h_pos(T * D), h_lm(vocab * D);
  std::vector<float> h_gamma(D, 1.0f), h_beta(D, 0.0f);
  std::vector<float> h_dlogits(B * T * vocab);
  rng.fill(h_tok, 0.5f);
  rng.fill(h_pos, 0.5f);
  rng.fill(h_lm, 0.5f);
  rng.fill(h_dlogits, 0.2f);

  int32_t* d_ids = nullptr;
  float *d_tok = nullptr, *d_pos = nullptr, *d_lm = nullptr;
  float *d_gamma = nullptr, *d_beta = nullptr;
  float *d_embed = nullptr, *d_mean = nullptr, *d_inv = nullptr;
  float *d_final = nullptr, *d_logits = nullptr, *d_dlogits = nullptr;
  float *d_tok_grad = nullptr, *d_pos_grad = nullptr, *d_lm_grad = nullptr;
  float *d_gamma_grad = nullptr, *d_beta_grad = nullptr;
  cudaMalloc(&d_ids, B * T * sizeof(int32_t));
  cudaMalloc(&d_tok, vocab * D * sizeof(float));
  cudaMalloc(&d_pos, T * D * sizeof(float));
  cudaMalloc(&d_lm, vocab * D * sizeof(float));
  cudaMalloc(&d_gamma, D * sizeof(float));
  cudaMalloc(&d_beta, D * sizeof(float));
  cudaMalloc(&d_embed, B * T * D * sizeof(float));
  cudaMalloc(&d_mean, B * T * sizeof(float));
  cudaMalloc(&d_inv, B * T * sizeof(float));
  cudaMalloc(&d_final, B * T * D * sizeof(float));
  cudaMalloc(&d_logits, B * T * vocab * sizeof(float));
  cudaMalloc(&d_dlogits, B * T * vocab * sizeof(float));
  cudaMalloc(&d_tok_grad, vocab * D * sizeof(float));
  cudaMalloc(&d_pos_grad, T * D * sizeof(float));
  cudaMalloc(&d_lm_grad, vocab * D * sizeof(float));
  cudaMalloc(&d_gamma_grad, D * sizeof(float));
  cudaMalloc(&d_beta_grad, D * sizeof(float));
  cudaMemcpy(d_ids, h_ids.data(), B * T * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_tok, h_tok.data(), vocab * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos, h_pos.data(), T * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lm, h_lm.data(), vocab * D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_gamma, h_gamma.data(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_beta, h_beta.data(), D * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dlogits, h_dlogits.data(), B * T * vocab * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(d_tok_grad, 0, vocab * D * sizeof(float));
  cudaMemset(d_pos_grad, 0, T * D * sizeof(float));
  cudaMemset(d_lm_grad, 0, vocab * D * sizeof(float));
  cudaMemset(d_gamma_grad, 0, D * sizeof(float));
  cudaMemset(d_beta_grad, 0, D * sizeof(float));

  GPTParams params;
  params.token_embedding = {d_tok, {vocab, D}, {D, 1}};
  params.position_embedding = {d_pos, {T, D}, {D, 1}};
  params.final_ln.gamma = {d_gamma, {D}, {1}};
  params.final_ln.beta = {d_beta, {D}, {1}};
  params.final_ln.eps = config.layernorm_eps;
  params.lm_head = {d_lm, {vocab, D}, {D, 1}};

  GPTForwardState state;
  state.embed_out = {d_embed, {B, T, D}, {T * D, D, 1}};
  state.final_ln_out = {d_final, {B, T, D}, {T * D, D, 1}};
  state.final_ln_state.mean = {d_mean, {B * T}, {1}};
  state.final_ln_state.inv_std = {d_inv, {B * T}, {1}};
  state.logits = {d_logits, {B * T, vocab}, {vocab, 1}};

  GPTGrads grads;
  grads.d_token_embedding = {d_tok_grad, {vocab, D}, {D, 1}};
  grads.d_position_embedding = {d_pos_grad, {T, D}, {D, 1}};
  grads.d_final_ln_gamma = {d_gamma_grad, {D}, {1}};
  grads.d_final_ln_beta = {d_beta_grad, {D}, {1}};
  grads.d_lm_head = {d_lm_grad, {vocab, D}, {D, 1}};

  Tensor2D<const int32_t> input_ids{d_ids, {B, T}, {T, 1}};
  Tensor2D<const float> d_logits_view{d_dlogits, {B * T, vocab}, {vocab, 1}};

  ASSERT_TRUE(gpt_forward(input_ids, config, params, state, stream_).ok());
  DeviceScratchArena scratch;
  ASSERT_TRUE(scratch.reserve_bytes(1 << 20).ok());
  auto status = gpt_backward(d_logits_view, input_ids, config, params, state,
                             grads, scratch, stream_);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_TRUE(stream_.synchronize().ok());

  auto tok = test::cpu_embedding_forward(h_tok.data(), h_ids.data(), B * T, D);
  std::vector<float> embed(B * T * D);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t d = 0; d < D; ++d) {
        int64_t idx = (b * T + t) * D + d;
        embed[idx] = tok[idx] + h_pos[t * D + d];
      }
    }
  }
  auto ln = test::cpu_layernorm_forward(
      embed.data(), h_gamma.data(), h_beta.data(), config.layernorm_eps, B * T, D);
  auto lm_grads = test::cpu_linear_backward(
      ln.y.data(), h_dlogits.data(), h_lm.data(), B * T, D, vocab);
  auto ln_grads = test::cpu_layernorm_backward(
      lm_grads.dX.data(), embed.data(), h_gamma.data(), ln.mean.data(),
      ln.inv_std.data(), B * T, D);
  auto tok_grads = test::cpu_embedding_backward(
      ln_grads.dx.data(), h_ids.data(), B * T, D, vocab);
  std::vector<float> pos_grads(T * D, 0.0f);
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t d = 0; d < D; ++d) {
        pos_grads[t * D + d] += ln_grads.dx[(b * T + t) * D + d];
      }
    }
  }

  std::vector<float> h_tok_grad(vocab * D), h_pos_grad(T * D);
  std::vector<float> h_lm_grad(vocab * D), h_gamma_grad(D), h_beta_grad(D);
  cudaMemcpy(h_tok_grad.data(), d_tok_grad, vocab * D * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pos_grad.data(), d_pos_grad, T * D * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_lm_grad.data(), d_lm_grad, vocab * D * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_gamma_grad.data(), d_gamma_grad, D * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_beta_grad.data(), d_beta_grad, D * sizeof(float),
             cudaMemcpyDeviceToHost);

  EXPECT_TRUE(test::vectors_near(h_tok_grad, tok_grads, 1e-3f, 1e-2f))
      << "token embedding grad mismatch";
  EXPECT_TRUE(test::vectors_near(h_pos_grad, pos_grads, 1e-3f, 1e-2f))
      << "position embedding grad mismatch";
  EXPECT_TRUE(test::vectors_near(h_lm_grad, lm_grads.dW, 1e-3f, 1e-2f))
      << "lm head grad mismatch";
  EXPECT_TRUE(test::vectors_near(h_gamma_grad, ln_grads.dgamma, 1e-3f, 1e-2f))
      << "final LN gamma grad mismatch";
  EXPECT_TRUE(test::vectors_near(h_beta_grad, ln_grads.dbeta, 1e-3f, 1e-2f))
      << "final LN beta grad mismatch";

  cudaFree(d_ids);
  cudaFree(d_tok); cudaFree(d_pos); cudaFree(d_lm);
  cudaFree(d_gamma); cudaFree(d_beta);
  cudaFree(d_embed); cudaFree(d_mean); cudaFree(d_inv);
  cudaFree(d_final); cudaFree(d_logits); cudaFree(d_dlogits);
  cudaFree(d_tok_grad); cudaFree(d_pos_grad); cudaFree(d_lm_grad);
  cudaFree(d_gamma_grad); cudaFree(d_beta_grad);
}

TEST_F(GPTModelGPUTest, BackwardOneLayerProducesFiniteGradients) {
  const int64_t B = 1, T = 2, D = 4, H = 2, Dh = 2, vocab = 8, mlp = 8;

  GPTConfig config;
  config.vocab_size = vocab;
  config.max_seq_len = T;
  config.n_layers = 1;
  config.n_heads = H;
  config.d_model = D;
  config.mlp_hidden = mlp;
  config.use_bias = false;
  config.tie_embeddings = false;
  config.layernorm_eps = 1e-5f;

  const int64_t param_count = config.approx_param_count();
  std::vector<int32_t> h_ids = {1, 3};
  std::vector<float> h_params(param_count);
  std::vector<float> h_dlogits(B * T * vocab);
  test::SimpleRng rng(29);
  rng.fill(h_params, 0.2f);
  rng.fill(h_dlogits, 0.1f);

  int32_t* d_ids = nullptr;
  float* d_params = nullptr;
  float* d_grads = nullptr;
  float* d_dlogits = nullptr;
  cudaMalloc(&d_ids, B * T * sizeof(int32_t));
  cudaMalloc(&d_params, param_count * sizeof(float));
  cudaMalloc(&d_grads, param_count * sizeof(float));
  cudaMalloc(&d_dlogits, B * T * vocab * sizeof(float));
  cudaMemcpy(d_ids, h_ids.data(), B * T * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_params, h_params.data(), param_count * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_dlogits, h_dlogits.data(), B * T * vocab * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemset(d_grads, 0, param_count * sizeof(float));

  GPTParams params;
  GPTGrads grads;
  ASSERT_TRUE(init_gpt_params(config, d_params, params).ok());
  ASSERT_TRUE(init_gpt_grads(config, d_grads, grads).ok());

  auto alloc = [](size_t bytes) {
    float* p = nullptr;
    cudaMalloc(&p, bytes);
    return p;
  };

  float* d_embed = alloc(B * T * D * sizeof(float));
  float* d_block_input = alloc(B * T * D * sizeof(float));
  float* d_final = alloc(B * T * D * sizeof(float));
  float* d_logits = alloc(B * T * vocab * sizeof(float));
  float* d_final_mean = alloc(B * T * sizeof(float));
  float* d_final_inv = alloc(B * T * sizeof(float));

  float* d_ln1_out = alloc(B * T * D * sizeof(float));
  float* d_ln2_out = alloc(B * T * D * sizeof(float));
  float* d_ln1_mean = alloc(B * T * sizeof(float));
  float* d_ln1_inv = alloc(B * T * sizeof(float));
  float* d_ln2_mean = alloc(B * T * sizeof(float));
  float* d_ln2_inv = alloc(B * T * sizeof(float));
  float* d_attn_out = alloc(B * T * D * sizeof(float));
  float* d_fc1_out = alloc(B * T * mlp * sizeof(float));
  float* d_gelu_out = alloc(B * T * mlp * sizeof(float));
  float* d_fc2_out = alloc(B * T * D * sizeof(float));
  float* d_qkv = alloc(B * T * 3 * D * sizeof(float));
  float* d_Q = alloc(B * H * T * Dh * sizeof(float));
  float* d_K = alloc(B * H * T * Dh * sizeof(float));
  float* d_V = alloc(B * H * T * Dh * sizeof(float));
  float* d_scores = alloc(B * H * T * T * sizeof(float));
  float* d_probs = alloc(B * H * T * T * sizeof(float));
  float* d_ctx = alloc(B * H * T * Dh * sizeof(float));
  float* d_merged = alloc(B * T * D * sizeof(float));

  GPTForwardState state;
  state.embed_out = {d_embed, {B, T, D}, {T * D, D, 1}};
  state.block_inputs.resize(1);
  state.block_inputs[0] = {d_block_input, {B, T, D}, {T * D, D, 1}};
  state.blocks.resize(1);
  state.final_ln_out = {d_final, {B, T, D}, {T * D, D, 1}};
  state.final_ln_state.mean = {d_final_mean, {B * T}, {1}};
  state.final_ln_state.inv_std = {d_final_inv, {B * T}, {1}};
  state.logits = {d_logits, {B * T, vocab}, {vocab, 1}};

  BlockForwardState& block = state.blocks[0];
  block.ln1_out = {d_ln1_out, {B, T, D}, {T * D, D, 1}};
  block.ln2_out = {d_ln2_out, {B, T, D}, {T * D, D, 1}};
  block.ln1_state.mean = {d_ln1_mean, {B * T}, {1}};
  block.ln1_state.inv_std = {d_ln1_inv, {B * T}, {1}};
  block.ln2_state.mean = {d_ln2_mean, {B * T}, {1}};
  block.ln2_state.inv_std = {d_ln2_inv, {B * T}, {1}};
  block.attn_out = {d_attn_out, {B, T, D}, {T * D, D, 1}};
  block.fc1_out = {d_fc1_out, {B, T, mlp}, {T * mlp, mlp, 1}};
  block.gelu_out = {d_gelu_out, {B, T, mlp}, {T * mlp, mlp, 1}};
  block.fc2_out = {d_fc2_out, {B, T, D}, {T * D, D, 1}};
  block.attn_state.qkv = {d_qkv, {B, T, 3 * D}, {T * 3 * D, 3 * D, 1}};
  block.attn_state.Q = {d_Q, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  block.attn_state.K = {d_K, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  block.attn_state.V = {d_V, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  block.attn_state.scores = {d_scores, {B, H, T, T}, {H * T * T, T * T, T, 1}};
  block.attn_state.probs = {d_probs, {B, H, T, T}, {H * T * T, T * T, T, 1}};
  block.attn_state.context = {d_ctx, {B, H, T, Dh}, {H * T * Dh, T * Dh, Dh, 1}};
  block.attn_state.context_merged = {d_merged, {B, T, D}, {T * D, D, 1}};

  Tensor2D<const int32_t> input_ids{d_ids, {B, T}, {T, 1}};
  Tensor2D<const float> d_logits_view{d_dlogits, {B * T, vocab}, {vocab, 1}};

  ASSERT_TRUE(gpt_forward(input_ids, config, params, state, stream_).ok());
  DeviceScratchArena scratch;
  ASSERT_TRUE(scratch.reserve_bytes(1 << 20).ok());
  auto status = gpt_backward(d_logits_view, input_ids, config, params, state,
                             grads, scratch, stream_);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_TRUE(stream_.synchronize().ok());

  std::vector<float> h_grads(param_count);
  cudaMemcpy(h_grads.data(), d_grads, param_count * sizeof(float),
             cudaMemcpyDeviceToHost);
  float abs_sum = 0.0f;
  for (float value : h_grads) {
    EXPECT_TRUE(std::isfinite(value));
    abs_sum += std::abs(value);
  }
  EXPECT_GT(abs_sum, 0.0f);

  cudaFree(d_ids); cudaFree(d_params); cudaFree(d_grads); cudaFree(d_dlogits);
  cudaFree(d_embed); cudaFree(d_block_input); cudaFree(d_final); cudaFree(d_logits);
  cudaFree(d_final_mean); cudaFree(d_final_inv);
  cudaFree(d_ln1_out); cudaFree(d_ln2_out);
  cudaFree(d_ln1_mean); cudaFree(d_ln1_inv);
  cudaFree(d_ln2_mean); cudaFree(d_ln2_inv);
  cudaFree(d_attn_out); cudaFree(d_fc1_out);
  cudaFree(d_gelu_out); cudaFree(d_fc2_out);
  cudaFree(d_qkv); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
  cudaFree(d_scores); cudaFree(d_probs); cudaFree(d_ctx); cudaFree(d_merged);
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
