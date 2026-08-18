// Layer 5 — Model: transformer block implementation.

#include "src/model/gpt_block.h"

#include <cuda_runtime.h>

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
    if (state.attn_workspace == nullptr) {
      return Status(StatusCode::kInvalidArgument,
                    "block_forward: missing attention workspace");
    }
    GPT_RETURN_IF_ERROR(attention::attention_forward(
        ln1_in, attn_cfg, params.attn, state.attn_out,
        *state.attn_workspace, stream));
    state.attn_state = state.attn_workspace->state();
  }

  // Residual add: output = x + attn_out.
  {
    int64_t n = B * T * D;
    Tensor1D<const float> x_flat{x.data, {n}, {1}};
    Tensor1D<const float> attn_flat{state.attn_out.data, {n}, {1}};
    Tensor1D<float> output_flat{output.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(
        kernels::vec_add_f32(x_flat, attn_flat, output_flat, stream));
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
    Tensor1D<const float> output_flat{output.data, {n}, {1}};
    Tensor1D<const float> fc2_flat{state.fc2_out.data, {n}, {1}};
    Tensor1D<float> out_flat{output.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(
        kernels::vec_add_f32(output_flat, fc2_flat, out_flat, stream));
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
                      DeviceScratchArena& scratch,
                      const CudaStream& stream) {
  int64_t B = x.shape[0];
  int64_t T = x.shape[1];
  int64_t D = x.shape[2];
  int64_t M = config.mlp_hidden;
  int64_t n = B * T * D;
  int64_t mlp_n = B * T * M;

  auto residual1_ptr = scratch.alloc_f32(n);
  if (!residual1_ptr.ok()) return residual1_ptr.status();
  auto d_residual2_ptr = scratch.alloc_f32(n);
  if (!d_residual2_ptr.ok()) return d_residual2_ptr.status();
  auto d_fc2_ptr = scratch.alloc_f32(n);
  if (!d_fc2_ptr.ok()) return d_fc2_ptr.status();
  auto d_gelu_ptr = scratch.alloc_f32(mlp_n);
  if (!d_gelu_ptr.ok()) return d_gelu_ptr.status();
  auto d_fc1_ptr = scratch.alloc_f32(mlp_n);
  if (!d_fc1_ptr.ok()) return d_fc1_ptr.status();
  auto d_ln2_ptr = scratch.alloc_f32(n);
  if (!d_ln2_ptr.ok()) return d_ln2_ptr.status();
  auto d_residual1_ptr = scratch.alloc_f32(n);
  if (!d_residual1_ptr.ok()) return d_residual1_ptr.status();
  auto d_attn_ptr = scratch.alloc_f32(n);
  if (!d_attn_ptr.ok()) return d_attn_ptr.status();
  auto d_ln1_ptr = scratch.alloc_f32(n);
  if (!d_ln1_ptr.ok()) return d_ln1_ptr.status();

  Tensor3D<float> residual1{
      residual1_ptr.value(), {B, T, D}, {T * D, D, 1}};
  Tensor3D<float> d_residual2{
      d_residual2_ptr.value(), {B, T, D}, {T * D, D, 1}};
  Tensor3D<float> d_residual1{
      d_residual1_ptr.value(), {B, T, D}, {T * D, D, 1}};
  Tensor3D<float> d_attn{
      d_attn_ptr.value(), {B, T, D}, {T * D, D, 1}};
  Tensor3D<float> d_ln1{
      d_ln1_ptr.value(), {B, T, D}, {T * D, D, 1}};

  // Reconstruct the first residual activation consumed by LN2:
  // residual1 = x + attn_out.
  {
    Tensor1D<const float> x_flat{x.data, {n}, {1}};
    Tensor1D<const float> attn_flat{state.attn_out.data, {n}, {1}};
    Tensor1D<float> residual_flat{residual1.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(
        kernels::vec_add_f32(x_flat, attn_flat, residual_flat, stream));
  }

  // The final residual add fans d_output to both fc2_out and residual1.
  {
    Tensor1D<const float> d_out_flat{d_output.data, {n}, {1}};
    Tensor1D<float> d_fc2_flat{d_fc2_ptr.value(), {n}, {1}};
    Tensor1D<float> d_residual2_flat{d_residual2.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(
        kernels::vec_scale_f32(d_out_flat, 1.0f, d_fc2_flat, stream));
    GPT_RETURN_IF_ERROR(
        kernels::vec_scale_f32(d_out_flat, 1.0f, d_residual2_flat, stream));
  }

  // fc2 backward: d_fc2_out -> d_gelu_out.
  {
    Tensor2D<const float> gelu_2d{
        state.gelu_out.data, {B * T, M}, {M, 1}};
    Tensor2D<const float> d_fc2_2d{
        d_fc2_ptr.value(), {B * T, D}, {D, 1}};
    Tensor2D<float> d_gelu_2d{
        d_gelu_ptr.value(), {B * T, M}, {M, 1}};
    GPT_RETURN_IF_ERROR(ops::linear_backward(
        gelu_2d, d_fc2_2d, params.fc2, d_gelu_2d, grads.fc2, stream));
  }

  // GELU backward.
  {
    Tensor1D<const float> d_gelu_flat{d_gelu_ptr.value(), {mlp_n}, {1}};
    Tensor1D<const float> fc1_flat{state.fc1_out.data, {mlp_n}, {1}};
    Tensor1D<float> d_fc1_flat{d_fc1_ptr.value(), {mlp_n}, {1}};
    GPT_RETURN_IF_ERROR(
        ops::gelu_backward(d_gelu_flat, fc1_flat, d_fc1_flat, stream));
  }

  // fc1 backward: d_fc1_out -> d_ln2_out.
  {
    Tensor2D<const float> ln2_2d{
        state.ln2_out.data, {B * T, D}, {D, 1}};
    Tensor2D<const float> d_fc1_2d{
        d_fc1_ptr.value(), {B * T, M}, {M, 1}};
    Tensor2D<float> d_ln2_2d{
        d_ln2_ptr.value(), {B * T, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(ops::linear_backward(
        ln2_2d, d_fc1_2d, params.fc1, d_ln2_2d, grads.fc1, stream));
  }

  // LN2 backward: d_ln2_out -> d_residual1_from_ln2.
  {
    Tensor2D<const float> d_ln2_2d{
        d_ln2_ptr.value(), {B * T, D}, {D, 1}};
    Tensor2D<const float> residual1_2d{
        residual1.data, {B * T, D}, {D, 1}};
    Tensor2D<float> d_residual1_2d{
        d_residual1.data, {B * T, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(ops::layernorm_backward(
        d_ln2_2d, residual1_2d, params.ln2, state.ln2_state,
        d_residual1_2d, grads.d_ln2_gamma, grads.d_ln2_beta, stream));
  }

  // Add direct residual branch from final output: d_residual1 += d_residual2.
  {
    Tensor1D<const float> from_ln2{d_residual1.data, {n}, {1}};
    Tensor1D<const float> direct{d_residual2.data, {n}, {1}};
    Tensor1D<float> total{d_residual1.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(kernels::vec_add_f32(from_ln2, direct, total, stream));
  }

  // The first residual add fans d_residual1 to x and attention output.
  {
    Tensor1D<const float> d_residual1_flat{d_residual1.data, {n}, {1}};
    Tensor1D<float> d_attn_flat{d_attn.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(
        kernels::vec_scale_f32(d_residual1_flat, 1.0f, d_attn_flat, stream));
  }

  attention::AttentionConfig attn_cfg;
  attn_cfg.d_model = config.d_model;
  attn_cfg.n_heads = config.n_heads;
  attn_cfg.head_dim = config.head_dim();
  attn_cfg.use_bias = config.use_bias;
  attn_cfg.causal = true;

  {
    Tensor3D<const float> ln1_in{
        state.ln1_out.data, {B, T, D}, {T * D, D, 1}};
    GPT_RETURN_IF_ERROR(attention::attention_backward(
        d_attn, ln1_in, attn_cfg, params.attn, state.attn_state,
        d_ln1, grads.attn, scratch, stream));
  }

  // LN1 backward: d_ln1_out -> d_x_from_ln1.
  {
    Tensor2D<const float> d_ln1_2d{d_ln1.data, {B * T, D}, {D, 1}};
    Tensor2D<const float> x_2d{x.data, {B * T, D}, {D, 1}};
    Tensor2D<float> dx_2d{dx.data, {B * T, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(ops::layernorm_backward(
        d_ln1_2d, x_2d, params.ln1, state.ln1_state,
        dx_2d, grads.d_ln1_gamma, grads.d_ln1_beta, stream));
  }

  // Add direct residual branch from the first residual add: dx += d_residual1.
  {
    Tensor1D<const float> from_ln1{dx.data, {n}, {1}};
    Tensor1D<const float> direct{d_residual1.data, {n}, {1}};
    Tensor1D<float> total{dx.data, {n}, {1}};
    GPT_RETURN_IF_ERROR(kernels::vec_add_f32(from_ln1, direct, total, stream));
  }

  return Status::Ok();
}

}  // namespace model
}  // namespace gpt
