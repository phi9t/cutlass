#pragma once

// Layer 5 — Model: GPT parameter registry.
//
// Flat parameter storage with typed logical views into named parameter
// groups (embeddings, per-layer weights, LN params, etc.).
//
// All params and grads share a single PersistentArena allocation.

#include <cstdint>
#include <vector>

#include "src/attention/attention.h"
#include "src/model/gpt_config.h"
#include "src/ops/layernorm.h"
#include "src/ops/linear.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace model {

// Per-layer parameter views.
struct LayerParams {
  // Pre-attention LayerNorm.
  ops::LayerNormParams ln1;

  // Self-attention.
  attention::AttentionParams attn;

  // Post-attention LayerNorm.
  ops::LayerNormParams ln2;

  // MLP.
  ops::LinearParams fc1;   // [mlp_hidden, d_model]
  ops::LinearParams fc2;   // [d_model, mlp_hidden]
};

// Complete GPT parameter set.
struct GPTParams {
  // Embedding tables.
  Tensor2D<float> token_embedding;       // [vocab_size, d_model]
  Tensor2D<float> position_embedding;    // [max_seq_len, d_model]

  // Per-layer parameters.
  std::vector<LayerParams> layers;

  // Final LayerNorm.
  ops::LayerNormParams final_ln;

  // LM head (may alias token_embedding if tied).
  Tensor2D<float> lm_head;              // [vocab_size, d_model]
};

// Allocate all parameters in a flat buffer and return typed views.
// `buffer` must be pre-allocated with enough space (see GPTConfig::approx_param_count).
Status init_gpt_params(const GPTConfig& config,
                       float* buffer,
                       GPTParams& params);

// ---------------------------------------------------------------------------
// Gradient views — same structure as params.
// ---------------------------------------------------------------------------

struct GPTGrads {
  Tensor2D<float> d_token_embedding;
  Tensor2D<float> d_position_embedding;

  struct LayerGrads {
    Tensor1D<float> d_ln1_gamma;
    Tensor1D<float> d_ln1_beta;
    attention::AttentionGrads attn;
    Tensor1D<float> d_ln2_gamma;
    Tensor1D<float> d_ln2_beta;
    ops::LinearGrads fc1;
    ops::LinearGrads fc2;
  };
  std::vector<LayerGrads> layers;

  Tensor1D<float> d_final_ln_gamma;
  Tensor1D<float> d_final_ln_beta;
  Tensor2D<float> d_lm_head;
};

Status init_gpt_grads(const GPTConfig& config,
                      float* buffer,
                      GPTGrads& grads);

}  // namespace model
}  // namespace gpt
