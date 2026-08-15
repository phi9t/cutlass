#pragma once

// Layer 5 — Model: single transformer block.
//
// Pre-norm GPT-2-like structure:
//   residual = x
//   x = LayerNorm(x)
//   x = CausalSelfAttention(x)
//   x = x + residual
//   residual = x
//   x = LayerNorm(x)
//   x = MLP(x)   // fc1 → GELU → fc2
//   x = x + residual

#include "src/attention/attention.h"
#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/model/gpt_config.h"
#include "src/model/gpt_params.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace model {

// Workspace tensors needed for a single block forward pass.
struct BlockForwardState {
  // LayerNorm outputs.
  Tensor3D<float> ln1_out;       // [B, T, D]
  Tensor3D<float> ln2_out;       // [B, T, D]

  // LayerNorm saved state.
  ops::LayerNormState ln1_state;
  ops::LayerNormState ln2_state;

  // Attention intermediates.
  attention::AttentionForwardState attn_state;
  Tensor3D<float> attn_out;      // [B, T, D]

  // MLP intermediates.
  Tensor3D<float> fc1_out;       // [B, T, mlp_hidden]
  Tensor3D<float> gelu_out;      // [B, T, mlp_hidden]
  Tensor3D<float> fc2_out;       // [B, T, D]
};

// Forward: single transformer block.
//   x:      [B, T, D]  (input, also receives output in-place for residual)
//   output: [B, T, D]
Status block_forward(Tensor3D<const float> x,
                     const GPTConfig& config,
                     const LayerParams& params,
                     Tensor3D<float> output,
                     BlockForwardState& state,
                     const CudaStream& stream);

// Backward: single transformer block.
Status block_backward(Tensor3D<const float> d_output,
                      Tensor3D<const float> x,
                      const GPTConfig& config,
                      const LayerParams& params,
                      const BlockForwardState& state,
                      Tensor3D<float> dx,
                      GPTGrads::LayerGrads& grads,
                      DeviceScratchArena& scratch,
                      const CudaStream& stream);

}  // namespace model
}  // namespace gpt
