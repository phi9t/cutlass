// Layer 5 — Model: transformer block implementation.

#include "src/model/gpt_block.h"

#include "src/attention/attention.h"
#include "src/kernels/elementwise.h"
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
  {
    int64_t n = B * T * D;
    Tensor1D<const float> a{x.data, {n}, {1}};
    Tensor1D<const float> b{state.attn_out.data, {n}, {1}};
    Tensor1D<float> out{output.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(kernels::vec_add_f32(a, b, out, stream));
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
  {
    int64_t n = B * T * D;
    Tensor1D<const float> a{output.data, {n}, {1}};
    Tensor1D<const float> b{state.fc2_out.data, {n}, {1}};
    Tensor1D<float> out{output.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(kernels::vec_add_f32(a, b, out, stream));
  }

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
  int64_t B = x.shape[0];
  int64_t T = x.shape[1];
  int64_t D = x.shape[2];
  int64_t n = B * T * D;

  // Forward was:
  //   ln1_out = LN1(x)
  //   attn_out = Attention(ln1_out)
  //   residual1 = x + attn_out
  //   ln2_out = LN2(residual1)
  //   fc1_out = fc1(ln2_out)
  //   gelu_out = GELU(fc1_out)
  //   fc2_out = fc2(gelu_out)
  //   output = residual1 + fc2_out
  //
  // Backward (d_output given):
  //   Step 1: d_fc2_out = d_output, d_residual2 = d_output  (residual split)

  // ---- Sub-block 2 backward: residual → fc2 → GELU → fc1 → LN2 ----

  // Step 2: fc2 backward. d_fc2_out = d_output.
  //   d_gelu_out = d_fc2_out @ fc2.W
  //   d_fc2.W, d_fc2.b
  Tensor3D<float> d_gelu;
  d_gelu.data = state.gelu_out.data;  // reuse buffer
  d_gelu.shape = {B, T, config.mlp_hidden};
  d_gelu.stride = {T * config.mlp_hidden, config.mlp_hidden, 1};
  {
    Tensor2D<const float> gelu_2d{state.gelu_out.data,
                                   {B * T, config.mlp_hidden},
                                   {config.mlp_hidden, 1}};
    Tensor2D<const float> d_out_2d{d_output.data, {B * T, D}, {D, 1}};
    Tensor2D<float> d_gelu_2d{d_gelu.data,
                               {B * T, config.mlp_hidden},
                               {config.mlp_hidden, 1}};
    GPT_RETURN_IF_ERROR(
        ops::linear_backward(gelu_2d, d_out_2d, params.fc2,
                             d_gelu_2d, grads.fc2, stream));
  }

  // Step 3: GELU backward. dx_gelu = d_gelu * GELU'(fc1_out).
  Tensor3D<float> d_fc1;
  d_fc1.data = state.fc1_out.data;  // reuse buffer
  d_fc1.shape = {B, T, config.mlp_hidden};
  d_fc1.stride = {T * config.mlp_hidden, config.mlp_hidden, 1};
  {
    int64_t m = B * T * config.mlp_hidden;
    Tensor1D<const float> dy{d_gelu.data, {m}, {1}};
    Tensor1D<const float> fc1_flat{state.fc1_out.data, {m}, {1}};
    Tensor1D<float> d_fc1_flat{d_fc1.data, {m}, {1}};
    GPT_RETURN_IF_ERROR(ops::gelu_backward(dy, fc1_flat, d_fc1_flat, stream));
  }

  // Step 4: fc1 backward.
  //   d_ln2_out = d_fc1 @ fc1.W
  //   d_fc1.W, d_fc1.b
  // We need a buffer for d_ln2_out [B*T, D]. Reuse state.ln2_out.
  Tensor3D<float> d_ln2;
  d_ln2.data = state.ln2_out.data;  // reuse buffer
  d_ln2.shape = {B, T, D};
  d_ln2.stride = {T * D, D, 1};
  {
    Tensor2D<const float> ln2_2d{state.ln2_out.data, {B * T, D}, {D, 1}};
    Tensor2D<const float> d_fc1_2d{d_fc1.data,
                                    {B * T, config.mlp_hidden},
                                    {config.mlp_hidden, 1}};
    Tensor2D<float> d_ln2_2d{d_ln2.data, {B * T, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(
        ops::linear_backward(ln2_2d, d_fc1_2d, params.fc1,
                             d_ln2_2d, grads.fc1, stream));
  }

  // Step 5: LN2 backward.
  //   d_residual1 = LN2_backward(d_ln2_out)
  // The input to LN2 was residual1 (= output of sub-block 1 = x + attn_out).
  // We stored output in the forward pass — the LN2 input is state.ln2's source,
  // which was `output` (the residual1 result). We don't have a separate buffer
  // for residual1, but it equals output.data at the point LN2 was called.
  // In forward, LN2 input was `output` (the first residual result).
  // We need the original residual1 for LN backward. It's not separately saved,
  // but we can reconstruct it: residual1 = x + attn_out.
  // For the LN backward, the input x is needed. We'll use a buffer for d_residual1.
  // Write d_residual1 into dx (it will be accumulated later).
  {
    Tensor2D<const float> d_ln2_2d{d_ln2.data, {B * T, D}, {D, 1}};
    // LN2 input was the first residual output. We need to reconstruct it.
    // residual1 = x + attn_out. We'll compute it into dx as a temp buffer.
    // Actually, we need the original input to LN2, not its gradient.
    // The forward saved ln2_state (mean, inv_std), and LN backward needs
    // the original input x. The original LN2 input was the first residual:
    // output = x + attn_out. We don't have this saved separately.
    // But the output buffer passed to block_forward was reused as the
    // residual1 result. In the caller's loop, block_input is reused each
    // layer. So the original residual1 may be overwritten.
    //
    // However, for LN backward, we need the original input to reconstruct
    // x_hat = (x - mean) * inv_std. Since we saved mean and inv_std in
    // ln2_state, and gamma in params.ln2, the backward can be computed.
    // The layernorm_backward signature takes the original x input.
    // We need to recompute residual1 = x + attn_out.
    // x is the block input (passed in), attn_out is saved in state.
    // Compute residual1 into dx buffer temporarily.
    Tensor1D<const float> x_flat{x.data, {n}, {1}};
    Tensor1D<const float> attn_flat{state.attn_out.data, {n}, {1}};
    Tensor1D<float> res1_flat{dx.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(kernels::vec_add_f32(x_flat, attn_flat, res1_flat, stream));

    Tensor2D<const float> res1_2d{dx.data, {B * T, D}, {D, 1}};
    Tensor2D<float> dx_ln2{dx.data, {B * T, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(ops::layernorm_backward(
        d_ln2_2d, res1_2d, params.ln2, state.ln2_state,
        dx_ln2, grads.d_ln2_gamma, grads.d_ln2_beta, stream));
  }

  // Step 6: Accumulate residual from sub-block 2.
  //   d_residual1 = d_output + d_ln2_backward_output
  // dx currently holds LN2's backward output (d w.r.t. residual1 input).
  // Add d_output (the residual path that bypasses sub-block 2).
  {
    Tensor1D<const float> a{dx.data, {n}, {1}};
    Tensor1D<const float> b{d_output.data, {n}, {1}};
    Tensor1D<float> out{dx.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(kernels::vec_add_f32(a, b, out, stream));
  }

  // ---- Sub-block 1 backward: residual → Attention → LN1 ----

  // Step 7: Attention backward.
  //   d_ln1_out = attention_backward(d_attn_out = d_residual1)
  // The gradient flowing into attention is d_residual1 (= dx currently).
  // Write d_ln1_out into state.attn_out buffer (reuse).
  {
    attention::AttentionConfig attn_cfg;
    attn_cfg.d_model = config.d_model;
    attn_cfg.n_heads = config.n_heads;
    attn_cfg.head_dim = config.head_dim();
    attn_cfg.use_bias = config.use_bias;
    attn_cfg.causal = true;

    Tensor3D<const float> d_attn_in{dx.data, {B, T, D}, {T * D, D, 1}};
    Tensor3D<const float> ln1_in{state.ln1_out.data, {B, T, D}, {T * D, D, 1}};
    Tensor3D<float> d_ln1{state.attn_out.data, {B, T, D}, {T * D, D, 1}};

    GPT_RETURN_IF_ERROR(attention::attention_backward(
        d_attn_in, ln1_in, attn_cfg, params.attn,
        state.attn_state, d_ln1, grads.attn, stream));
  }

  // Step 8: LN1 backward.
  //   d_x_from_ln1 = layernorm_backward(d_ln1_out)
  // LN1 input was x (the block input). Write result into state.ln1_out (reuse).
  {
    Tensor2D<const float> d_ln1_2d{state.attn_out.data, {B * T, D}, {D, 1}};
    Tensor2D<const float> x_2d{x.data, {B * T, D}, {D, 1}};
    // Write LN1 backward output to state.ln1_out (reuse buffer).
    Tensor2D<float> dx_ln1{state.ln1_out.data, {B * T, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(ops::layernorm_backward(
        d_ln1_2d, x_2d, params.ln1, state.ln1_state,
        dx_ln1, grads.d_ln1_gamma, grads.d_ln1_beta, stream));
  }

  // Step 9: dx = d_residual1 + d_ln1_backward_output.
  // d_residual1 is in dx, d_ln1 backward output is in state.ln1_out.
  {
    Tensor1D<const float> a{dx.data, {n}, {1}};
    Tensor1D<const float> b{state.ln1_out.data, {n}, {1}};
    Tensor1D<float> out{dx.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(kernels::vec_add_f32(a, b, out, stream));
  }

  return Status::Ok();
}

}  // namespace model
}  // namespace gpt
