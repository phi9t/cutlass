#pragma once

// Layer 5 — Model: full GPT model.
//
// Structure:
//   1. Token embedding + positional embedding
//   2. N transformer blocks
//   3. Final LayerNorm
//   4. LM head (logits projection)

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/model/gpt_block.h"
#include "src/model/gpt_config.h"
#include "src/model/gpt_params.h"
#include "src/tensor/tensor_view.h"

#include <vector>

namespace gpt {
namespace model {

// Complete forward state for the model.
struct GPTForwardState {
  Tensor3D<float> embed_out;              // [B, T, D]
  std::vector<Tensor3D<float>> block_inputs;  // per-layer [B, T, D]
  std::vector<BlockForwardState> blocks;  // per-layer
  Tensor3D<float> final_ln_out;           // [B, T, D]
  ops::LayerNormState final_ln_state;
  Tensor2D<float> logits;                 // [B*T, vocab_size]
};

// Forward: tokens → logits.
//   input_ids: [B, T]  int32 token indices
//   logits:    [B*T, vocab_size]  output logits
Status gpt_forward(Tensor2D<const int32_t> input_ids,
                   const GPTConfig& config,
                   const GPTParams& params,
                   GPTForwardState& state,
                   const CudaStream& stream);

// Backward: d_logits → parameter gradients.
//   d_logits: [B*T, vocab_size]
Status gpt_backward(Tensor2D<const float> d_logits,
                    Tensor2D<const int32_t> input_ids,
                    const GPTConfig& config,
                    const GPTParams& params,
                    const GPTForwardState& state,
                    GPTGrads& grads,
                    DeviceScratchArena& scratch,
                    const CudaStream& stream);

}  // namespace model
}  // namespace gpt
