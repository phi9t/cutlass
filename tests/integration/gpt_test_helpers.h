#pragma once

// Shared GPU allocation helpers for GPT integration tests.
//
// Provides inline helpers to allocate and wire:
//   - GPTParams (parameter views into a flat device buffer)
//   - GPTGrads (gradient views into a flat device buffer)
//   - GPTForwardState (all intermediate buffers for forward/backward)
//
// Each allocator returns a device pointer the caller must cudaFree.

#include "src/model/gpt_config.h"
#include "src/model/gpt_model.h"
#include "src/model/gpt_params.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace gpt {
namespace test {

// Allocate all forward state buffers and wire them into fwd_state.
// Returns device allocation pointer. Caller must cudaFree.
inline float* allocate_forward_state(const model::GPTConfig& cfg,
                                     int64_t B, int64_t T,
                                     model::GPTForwardState& fwd) {
  int64_t D = cfg.d_model;
  int64_t H = cfg.n_heads;
  int64_t Dh = cfg.head_dim();
  int64_t M = cfg.mlp_hidden;
  int64_t V = cfg.vocab_size;
  int64_t L = cfg.n_layers;

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
                + L * B*T*D      // block_inputs (saved per layer)
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

  fwd.embed_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};

  fwd.block_inputs.resize(L);
  for (int64_t l = 0; l < L; ++l) {
    fwd.block_inputs[l] = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
  }

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

  fwd.final_ln_out = {advance(B*T*D), {B,T,D}, {T*D, D, 1}};
  fwd.final_ln_state.mean = {advance(B*T), {B*T}, {1}};
  fwd.final_ln_state.inv_std = {advance(B*T), {B*T}, {1}};

  fwd.logits = {advance(B*T*V), {B*T, V}, {V, 1}};

  return buf;
}

// Allocate parameter views from a flat device buffer.
inline float* allocate_params(const model::GPTConfig& cfg,
                              model::GPTParams& params) {
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

  params.token_embedding = {advance(V*D), {V, D}, {D, 1}};
  params.position_embedding = {advance(S*D), {S, D}, {D, 1}};

  params.layers.resize(L);
  for (int64_t l = 0; l < L; ++l) {
    auto& lp = params.layers[l];
    lp.ln1.gamma = {advance(D), {D}, {1}};
    lp.ln1.beta = {advance(D), {D}, {1}};
    lp.ln1.eps = cfg.layernorm_eps;
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
    lp.ln2.gamma = {advance(D), {D}, {1}};
    lp.ln2.beta = {advance(D), {D}, {1}};
    lp.ln2.eps = cfg.layernorm_eps;
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

  params.final_ln.gamma = {advance(D), {D}, {1}};
  params.final_ln.beta = {advance(D), {D}, {1}};
  params.final_ln.eps = cfg.layernorm_eps;

  if (cfg.tie_embeddings) {
    params.lm_head = params.token_embedding;
  } else {
    params.lm_head = {advance(V*D), {V, D}, {D, 1}};
  }

  return buf;
}

// Allocate gradient views from a flat device buffer (same layout as params).
inline float* allocate_grads(const model::GPTConfig& cfg,
                             model::GPTGrads& grads) {
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

}  // namespace test
}  // namespace gpt
