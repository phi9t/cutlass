// Layer 5 — Model: transformer block implementation.

#include "src/model/gpt_block.h"

#include "src/attention/attention.h"
#include "src/ops/gelu.h"
#include "src/ops/layernorm.h"
#include "src/ops/linear.h"

namespace gpt {
namespace model {

Status block_forward(Tensor3D<const float> x,
                     const GPTConfig& config,
                     const LayerParams& params,
                     Tensor3D<float> output,
                     BlockForwardState& state,
                     const CudaStream& stream) {
  int64_t B = x.shape[0];
  int64_t T = x.shape[1];
  int64_t D = x.shape[2];

  // ---- Sub-block 1: LayerNorm → Attention → Residual ----

  // LN1: [B*T, D]
  {
    Tensor2D<const float> x_2d{x.data, {B * T, D}, {D, 1}};
    Tensor2D<float> ln1_2d{state.ln1_out.data, {B * T, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(ops::layernorm_forward(
        x_2d, params.ln1, ln1_2d, state.ln1_state, stream));
  }

  // Attention.
  {
    attention::AttentionConfig attn_cfg;
    attn_cfg.d_model = config.d_model;
    attn_cfg.n_heads = config.n_heads;
    attn_cfg.head_dim = config.head_dim();
    attn_cfg.use_bias = config.use_bias;
    attn_cfg.causal = true;

    Tensor3D<const float> ln1_in{state.ln1_out.data,
                                  {B, T, D}, {T * D, D, 1}};
    GPT_RETURN_IF_ERROR(attention::attention_forward(
        ln1_in, attn_cfg, params.attn, state.attn_out,
        state.attn_state, stream));
  }

  // Residual add: output = x + attn_out.
  // TODO: Use vec_add kernel over flat buffer.
  // For now, mark the intent.
  {
    int64_t n = B * T * D;
    // output.data[i] = x.data[i] + state.attn_out.data[i]  for i in [0, n)
    // Placeholder — needs elementwise add kernel call.
  }

  // ---- Sub-block 2: LayerNorm → MLP → Residual ----

  // LN2.
  {
    Tensor2D<const float> res_2d{output.data, {B * T, D}, {D, 1}};
    Tensor2D<float> ln2_2d{state.ln2_out.data, {B * T, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(ops::layernorm_forward(
        res_2d, params.ln2, ln2_2d, state.ln2_state, stream));
  }

  // MLP: fc1 → GELU → fc2.
  {
    Tensor2D<const float> ln2_2d{state.ln2_out.data,
                                  {B * T, D}, {D, 1}};
    Tensor2D<float> fc1_2d{state.fc1_out.data,
                            {B * T, config.mlp_hidden},
                            {config.mlp_hidden, 1}};
    GPT_RETURN_IF_ERROR(
        ops::linear_forward(ln2_2d, params.fc1, fc1_2d, stream));

    // GELU (flat).
    Tensor1D<const float> fc1_flat{state.fc1_out.data,
                                    {B * T * config.mlp_hidden}, {1}};
    Tensor1D<float> gelu_flat{state.gelu_out.data,
                               {B * T * config.mlp_hidden}, {1}};
    GPT_RETURN_IF_ERROR(ops::gelu_forward(fc1_flat, gelu_flat, stream));

    // fc2.
    Tensor2D<const float> gelu_2d{state.gelu_out.data,
                                   {B * T, config.mlp_hidden},
                                   {config.mlp_hidden, 1}};
    Tensor2D<float> fc2_2d{state.fc2_out.data,
                            {B * T, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(
        ops::linear_forward(gelu_2d, params.fc2, fc2_2d, stream));
  }

  // Residual add: output = output + fc2_out.
  // TODO: Use vec_add kernel.

  return Status::Ok();
}

Status block_backward(Tensor3D<const float> d_output,
                      Tensor3D<const float> x,
                      const GPTConfig& config,
                      const LayerParams& params,
                      const BlockForwardState& state,
                      Tensor3D<float> dx,
                      GPTGrads::LayerGrads& grads,
                      const CudaStream& stream) {
  // TODO(m6): Implement block backward.
  // Reverse of forward:
  //   1. Residual split: d_fc2_out = d_output, d_residual2 = d_output
  //   2. fc2 backward
  //   3. GELU backward
  //   4. fc1 backward
  //   5. LN2 backward
  //   6. d_residual2 += d_ln2_input
  //   7. Attention backward
  //   8. LN1 backward
  //   9. dx = d_residual1 + d_ln1_input

  return Status(StatusCode::kNotImplemented,
                "block_backward not yet implemented");
}

}  // namespace model
}  // namespace gpt
