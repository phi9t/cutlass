#pragma once

// Layer 5 — Model: GPT configuration.

#include <cstdint>

namespace gpt {
namespace model {

struct GPTConfig {
  int64_t vocab_size   = 50257;
  int64_t max_seq_len  = 1024;
  int64_t n_layers     = 12;
  int64_t n_heads      = 12;
  int64_t d_model      = 768;
  int64_t mlp_hidden   = 3072;   // typically 4 * d_model
  bool tie_embeddings  = true;
  bool use_bias        = true;
  float layernorm_eps  = 1e-5f;

  int64_t head_dim() const { return d_model / n_heads; }

  // Total parameter count (approximate, for allocation sizing).
  int64_t approx_param_count() const {
    int64_t embed = vocab_size * d_model + max_seq_len * d_model;
    int64_t per_layer = 4 * d_model * d_model     // QKV + O projections
                      + 2 * d_model                // LN1 gamma/beta
                      + d_model * mlp_hidden       // fc1
                      + mlp_hidden * d_model        // fc2
                      + 2 * d_model;               // LN2 gamma/beta
    if (use_bias) {
      per_layer += 4 * d_model + mlp_hidden + d_model;  // biases
    }
    int64_t final_ln = 2 * d_model;
    int64_t lm_head = tie_embeddings ? 0 : vocab_size * d_model;
    return embed + n_layers * per_layer + final_ln + lm_head;
  }
};

}  // namespace model
}  // namespace gpt
