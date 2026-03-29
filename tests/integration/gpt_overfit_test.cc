// Tier C — Integration: GPT overfit-one-batch test.
//
// Initialize a tiny GPT model, run N training steps on a fixed batch,
// and verify that the loss decreases reliably.

#include "gtest/gtest.h"

#include "src/core/stream.h"
#include "src/model/gpt_config.h"
#include "src/model/gpt_model.h"
#include "src/model/gpt_params.h"
#include "src/ops/loss.h"
#include "src/train/optimizer.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace gpt;
using namespace gpt::model;

namespace {

// Helper: compute the total number of floats needed for forward state buffers.
struct ForwardStateBufferSizes {
  int64_t embed_out;       // B*T*D
  int64_t per_block_ln1;   // B*T*D
  int64_t per_block_ln2;   // B*T*D
  int64_t per_block_ln1_mean;   // B*T
  int64_t per_block_ln1_inv;    // B*T
  int64_t per_block_ln2_mean;   // B*T
  int64_t per_block_ln2_inv;    // B*T
  int64_t per_block_attn_out;   // B*T*D
  int64_t per_block_fc1;        // B*T*mlp_hidden
  int64_t per_block_gelu;       // B*T*mlp_hidden
  int64_t per_block_fc2;        // B*T*D
  // Attention state per block:
  int64_t per_block_qkv;        // B*T*3*D
  int64_t per_block_Q;           // B*H*T*Dh
  int64_t per_block_K;           // B*H*T*Dh
  int64_t per_block_V;           // B*H*T*Dh
  int64_t per_block_scores;      // B*H*T*T
  int64_t per_block_probs;       // B*H*T*T
  int64_t per_block_context;     // B*H*T*Dh
  int64_t per_block_context_merged; // B*T*D
  int64_t final_ln_out;   // B*T*D
  int64_t final_ln_mean;  // B*T
  int64_t final_ln_inv;   // B*T
  int64_t logits;          // B*T*V
};

// Allocate all forward state buffers and wire them into fwd_state.
// Returns total device allocation. Caller must cudaFree the returned pointer.
float* allocate_forward_state(const GPTConfig& cfg, int64_t B, int64_t T,
                              GPTForwardState& fwd) {
  int64_t D = cfg.d_model;
  int64_t H = cfg.n_heads;
  int64_t Dh = cfg.head_dim();
  int64_t M = cfg.mlp_hidden;
  int64_t V = cfg.vocab_size;
  int64_t L = cfg.n_layers;

  // Calculate total floats needed.
  int64_t per_block = B*T*D       // ln1_out
                    + B*T*D       // ln2_out
                    + B*T         // ln1 mean
                    + B*T         // ln1 inv_std
                    + B*T         // ln2 mean
                    + B*T         // ln2 inv_std
                    + B*T*D       // attn_out
                    + B*T*M       // fc1_out
                    + B*T*M       // gelu_out
                    + B*T*D       // fc2_out
                    + B*T*3*D     // qkv
                    + B*H*T*Dh    // Q
                    + B*H*T*Dh    // K
                    + B*H*T*Dh    // V
                    + B*H*T*T     // scores
                    + B*H*T*T     // probs
                    + B*H*T*Dh    // context
                    + B*T*D;      // context_merged

  int64_t total = B*T*D           // embed_out
                + L * per_block
                + B*T*D           // final_ln_out
                + B*T             // final_ln mean
                + B*T             // final_ln inv_std
                + B*T*V;          // logits

  float* buf = nullptr;
  cudaMalloc(&buf, total * sizeof(float));
  cudaMemset(buf, 0, total * sizeof(float));

  float* ptr = buf;
  auto advance = [&](int64_t n) -> float* {
    float* p = ptr;
    ptr += n;
    return p;
  };

  // embed_out
  fwd.embed_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};

  // Per-block state.
  fwd.blocks.resize(L);
  for (int64_t l = 0; l < L; ++l) {
    auto& blk = fwd.blocks[l];
    blk.ln1_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
    blk.ln2_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
    blk.ln1_state.mean = {advance(B*T), {B*T}, {1}};
    blk.ln1_state.inv_std = {advance(B*T), {B*T}, {1}};
    blk.ln2_state.mean = {advance(B*T), {B*T}, {1}};
    blk.ln2_state.inv_std = {advance(B*T), {B*T}, {1}};
    blk.attn_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
    blk.fc1_out = {advance(B*T*M), {B,T,M}, {T*M, M, 1}};
    blk.gelu_out = {advance(B*T*M), {B,T,M}, {T*M, M, 1}};
    blk.fc2_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
    blk.attn_state.qkv = {advance(B*T*3*D), {B,T,3*D}, {T*3*D, 3*D, 1}};
    blk.attn_state.Q = {advance(B*H*T*Dh), {B,H,T,Dh}, {H*T*Dh, T*Dh, Dh, 1}};
    blk.attn_state.K = {advance(B*H*T*Dh), {B,H,T,Dh}, {H*T*Dh, T*Dh, Dh, 1}};
    blk.attn_state.V = {advance(B*H*T*Dh), {B,H,T,Dh}, {H*T*Dh, T*Dh, Dh, 1}};
    blk.attn_state.scores = {advance(B*H*T*T), {B,H,T,T}, {H*T*T, T*T, T, 1}};
    blk.attn_state.probs = {advance(B*H*T*T), {B,H,T,T}, {H*T*T, T*T, T, 1}};
    blk.attn_state.context = {advance(B*H*T*Dh), {B,H,T,Dh}, {H*T*Dh, T*Dh, Dh, 1}};
    blk.attn_state.context_merged = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
  }

  // final_ln_out
  fwd.final_ln_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
  fwd.final_ln_state.mean = {advance(B*T), {B*T}, {1}};
  fwd.final_ln_state.inv_std = {advance(B*T), {B*T}, {1}};

  // logits
  fwd.logits = {advance(B*T*V), {B*T, V}, {V, 1}};

  return buf;
}

// Manually set up GPTParams from a flat buffer, carving out views.
float* allocate_params(const GPTConfig& cfg, GPTParams& params) {
  int64_t D = cfg.d_model;
  int64_t V = cfg.vocab_size;
  int64_t S = cfg.max_seq_len;
  int64_t M = cfg.mlp_hidden;
  int64_t L = cfg.n_layers;

  int64_t count = cfg.approx_param_count();
  float* buf = nullptr;
  cudaMalloc(&buf, count * sizeof(float));

  float* ptr = buf;
  auto advance = [&](int64_t n) -> float* {
    float* p = ptr;
    ptr += n;
    return p;
  };

  // Embeddings.
  params.token_embedding = {advance(V*D), {V, D}, {D, 1}};
  params.position_embedding = {advance(S*D), {S, D}, {D, 1}};

  // Per-layer.
  params.layers.resize(L);
  for (int64_t l = 0; l < L; ++l) {
    auto& lp = params.layers[l];
    // LN1
    lp.ln1.gamma = {advance(D), {D}, {1}};
    lp.ln1.beta = {advance(D), {D}, {1}};
    lp.ln1.eps = cfg.layernorm_eps;
    // Attention: W_qkv [3D, D], b_qkv [3D], W_o [D, D], b_o [D]
    lp.attn.W_qkv = {advance(3*D*D), {3*D, D}, {D, 1}};
    if (cfg.use_bias) {
      lp.attn.b_qkv = {advance(3*D), {3*D}, {1}};
    } else {
      lp.attn.b_qkv = {nullptr, {3*D}, {1}};
    }
    lp.attn.W_o = {advance(D*D), {D, D}, {D, 1}};
    if (cfg.use_bias) {
      lp.attn.b_o = {advance(D), {D}, {1}};
    } else {
      lp.attn.b_o = {nullptr, {D}, {1}};
    }
    // LN2
    lp.ln2.gamma = {advance(D), {D}, {1}};
    lp.ln2.beta = {advance(D), {D}, {1}};
    lp.ln2.eps = cfg.layernorm_eps;
    // MLP
    lp.fc1.weight = {advance(M*D), {M, D}, {D, 1}};
    if (cfg.use_bias) {
      lp.fc1.bias = {advance(M), {M}, {1}};
    } else {
      lp.fc1.bias = {nullptr, {M}, {1}};
    }
    lp.fc1.use_bias = cfg.use_bias;
    lp.fc2.weight = {advance(D*M), {D, M}, {M, 1}};
    if (cfg.use_bias) {
      lp.fc2.bias = {advance(D), {D}, {1}};
    } else {
      lp.fc2.bias = {nullptr, {D}, {1}};
    }
    lp.fc2.use_bias = cfg.use_bias;
  }

  // Final LN.
  params.final_ln.gamma = {advance(D), {D}, {1}};
  params.final_ln.beta = {advance(D), {D}, {1}};
  params.final_ln.eps = cfg.layernorm_eps;

  // LM head.
  if (cfg.tie_embeddings) {
    params.lm_head = params.token_embedding;
  } else {
    params.lm_head = {advance(V*D), {V, D}, {D, 1}};
  }

  return buf;
}

// Set up gradient views from a flat buffer, same layout as params.
float* allocate_grads(const GPTConfig& cfg, GPTGrads& grads) {
  int64_t D = cfg.d_model;
  int64_t V = cfg.vocab_size;
  int64_t S = cfg.max_seq_len;
  int64_t M = cfg.mlp_hidden;
  int64_t L = cfg.n_layers;

  int64_t count = cfg.approx_param_count();
  float* buf = nullptr;
  cudaMalloc(&buf, count * sizeof(float));
  cudaMemset(buf, 0, count * sizeof(float));

  float* ptr = buf;
  auto advance = [&](int64_t n) -> float* {
    float* p = ptr;
    ptr += n;
    return p;
  };

  grads.d_token_embedding = {advance(V*D), {V, D}, {D, 1}};
  grads.d_position_embedding = {advance(S*D), {S, D}, {D, 1}};

  grads.layers.resize(L);
  for (int64_t l = 0; l < L; ++l) {
    auto& lg = grads.layers[l];
    lg.d_ln1_gamma = {advance(D), {D}, {1}};
    lg.d_ln1_beta = {advance(D), {D}, {1}};
    lg.attn.dW_qkv = {advance(3*D*D), {3*D, D}, {D, 1}};
    if (cfg.use_bias) {
      lg.attn.db_qkv = {advance(3*D), {3*D}, {1}};
    } else {
      lg.attn.db_qkv = {nullptr, {3*D}, {1}};
    }
    lg.attn.dW_o = {advance(D*D), {D, D}, {D, 1}};
    if (cfg.use_bias) {
      lg.attn.db_o = {advance(D), {D}, {1}};
    } else {
      lg.attn.db_o = {nullptr, {D}, {1}};
    }
    lg.d_ln2_gamma = {advance(D), {D}, {1}};
    lg.d_ln2_beta = {advance(D), {D}, {1}};
    lg.fc1.d_weight = {advance(M*D), {M, D}, {D, 1}};
    if (cfg.use_bias) {
      lg.fc1.d_bias = {advance(M), {M}, {1}};
    } else {
      lg.fc1.d_bias = {nullptr, {M}, {1}};
    }
    lg.fc2.d_weight = {advance(D*M), {D, M}, {M, 1}};
    if (cfg.use_bias) {
      lg.fc2.d_bias = {advance(D), {D}, {1}};
    } else {
      lg.fc2.d_bias = {nullptr, {D}, {1}};
    }
  }

  grads.d_final_ln_gamma = {advance(D), {D}, {1}};
  grads.d_final_ln_beta = {advance(D), {D}, {1}};

  if (cfg.tie_embeddings) {
    grads.d_lm_head = grads.d_token_embedding;
  } else {
    grads.d_lm_head = {advance(V*D), {V, D}, {D, 1}};
  }

  return buf;
}

}  // namespace

class GPTOverfitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

TEST_F(GPTOverfitTest, LossDecreases) {
  // Tiny GPT: 1 layer, D=32, V=100, T=8, H=4
  GPTConfig cfg;
  cfg.vocab_size   = 100;
  cfg.max_seq_len  = 16;
  cfg.n_layers     = 1;
  cfg.n_heads      = 4;
  cfg.d_model      = 32;
  cfg.mlp_hidden   = 128;
  cfg.tie_embeddings = false;
  cfg.use_bias     = true;
  cfg.layernorm_eps = 1e-5f;

  const int64_t B = 2, T = 8;

  // Allocate model.
  GPTParams params;
  float* param_buf = allocate_params(cfg, params);
  ASSERT_NE(param_buf, nullptr);

  // Initialize parameters with small random values on host, then copy.
  int64_t param_count = cfg.approx_param_count();
  std::vector<float> h_params(param_count);
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  for (auto& v : h_params) v = dist(rng);

  // Set LN gammas to 1.0 and betas to 0.0 for stable init.
  // We'd need to know the exact offsets... for simplicity, just use the
  // random init — the model should still overfit with small weights.
  cudaMemcpy(param_buf, h_params.data(), param_count * sizeof(float),
             cudaMemcpyHostToDevice);

  // Allocate gradients.
  GPTGrads grads;
  float* grad_buf = allocate_grads(cfg, grads);

  // Allocate forward state.
  GPTForwardState fwd_state;
  float* fwd_buf = allocate_forward_state(cfg, B, T, fwd_state);

  // Create a fixed batch of input/target token IDs.
  std::vector<int32_t> h_ids(B * T), h_targets(B * T);
  std::uniform_int_distribution<int32_t> token_dist(0, cfg.vocab_size - 1);
  for (auto& id : h_ids) id = token_dist(rng);
  for (auto& id : h_targets) id = token_dist(rng);

  int32_t *d_ids, *d_targets;
  cudaMalloc(&d_ids, B * T * sizeof(int32_t));
  cudaMalloc(&d_targets, B * T * sizeof(int32_t));
  cudaMemcpy(d_ids, h_ids.data(), B * T * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, h_targets.data(), B * T * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  Tensor2D<const int32_t> input_ids{d_ids, {B, T}, {T, 1}};
  Tensor1D<const int32_t> targets_flat{d_targets, {B * T}, {1}};

  // Allocate device scalar for loss.
  float* d_loss = nullptr;
  cudaMalloc(&d_loss, sizeof(float));

  // Allocate d_logits buffer for backward.
  float* d_logits_buf = nullptr;
  cudaMalloc(&d_logits_buf, B * T * cfg.vocab_size * sizeof(float));

  // Optimizer.
  train::AdamWConfig opt_cfg;
  opt_cfg.lr = 1e-3f;
  opt_cfg.weight_decay = 0.0f;  // no decay for overfitting test
  train::AdamW optimizer;
  ASSERT_TRUE(optimizer.init(param_count, opt_cfg).ok());

  // Training loop.
  const int N = 50;
  std::vector<float> losses;
  losses.reserve(N);

  for (int step = 0; step < N; ++step) {
    // Zero grads.
    cudaMemset(grad_buf, 0, param_count * sizeof(float));

    // Forward.
    auto fwd_status = gpt_forward(input_ids, cfg, params, fwd_state, stream_);
    ASSERT_TRUE(fwd_status.ok()) << fwd_status.message();

    // Loss.
    Tensor2D<const float> logits_view{fwd_state.logits.data,
                                       fwd_state.logits.shape,
                                       fwd_state.logits.stride};
    auto loss_status = ops::cross_entropy_forward(
        logits_view, targets_flat, d_loss, stream_);
    ASSERT_TRUE(loss_status.ok()) << loss_status.message();

    stream_.synchronize();
    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    losses.push_back(h_loss);

    // d_logits from cross_entropy_backward.
    Tensor2D<const float> logits_for_bwd{fwd_state.logits.data,
                                          fwd_state.logits.shape,
                                          fwd_state.logits.stride};
    Tensor2D<float> d_logits_view{d_logits_buf,
                                   {B * T, cfg.vocab_size},
                                   {cfg.vocab_size, 1}};
    auto ce_bwd = ops::cross_entropy_backward(
        logits_for_bwd, targets_flat, d_logits_view, stream_);
    ASSERT_TRUE(ce_bwd.ok()) << ce_bwd.message();

    // Model backward.
    Tensor2D<const float> d_logits_const{d_logits_buf,
                                          {B * T, cfg.vocab_size},
                                          {cfg.vocab_size, 1}};
    auto bwd_status = gpt_backward(
        d_logits_const, input_ids, cfg, params, fwd_state, grads, stream_);
    ASSERT_TRUE(bwd_status.ok()) << bwd_status.message();

    // Optimizer step.
    auto opt_status = optimizer.step(param_buf, grad_buf, param_count, stream_);
    ASSERT_TRUE(opt_status.ok()) << opt_status.message();
    optimizer.advance_step();

    stream_.synchronize();
  }

  // Verify: loss should decrease from first to last.
  ASSERT_GT(losses.size(), 10u);
  float first_loss = losses[0];
  float last_loss = losses.back();

  // The loss should decrease significantly when overfitting a tiny model on
  // a fixed batch.
  EXPECT_LT(last_loss, first_loss)
      << "Loss did not decrease: first=" << first_loss
      << " last=" << last_loss;

  // More stringently: the final loss should be at most 80% of the initial.
  EXPECT_LT(last_loss, 0.8f * first_loss)
      << "Loss did not decrease enough: first=" << first_loss
      << " last=" << last_loss;

  // Cleanup.
  optimizer.release();
  cudaFree(d_logits_buf);
  cudaFree(d_loss);
  cudaFree(d_targets);
  cudaFree(d_ids);
  cudaFree(fwd_buf);
  cudaFree(grad_buf);
  cudaFree(param_buf);
}
