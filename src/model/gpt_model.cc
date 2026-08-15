// Layer 5 — Model: full GPT model implementation.

#include "src/model/gpt_model.h"

#include <cuda_runtime.h>

#include "src/model/gpt_model_forward_internal.h"
#include "src/ops/embedding.h"
#include "src/ops/layernorm.h"
#include "src/ops/linear.h"

namespace gpt {
namespace model {

Status gpt_forward(Tensor2D<const int32_t> input_ids,
                   const GPTConfig& config,
                   const GPTParams& params,
                   GPTForwardState& state,
                   const CudaStream& stream) {
  int64_t B = input_ids.shape[0];
  int64_t T = input_ids.shape[1];
  int64_t D = config.d_model;

  // Step 1: Token embedding.
  {
    Tensor1D<const int32_t> ids_flat{input_ids.data, {B * T}, {1}};
    Tensor2D<float> embed_2d{state.embed_out.data, {B * T, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(
        ops::embedding_forward(params.token_embedding, ids_flat,
                               embed_2d, stream));
  }

  // Step 2: Add positional embedding.
  GPT_RETURN_IF_ERROR(
      add_position_embedding_f32(state.embed_out, params.position_embedding,
                                 stream));

  // Step 3: Transformer blocks.
  Tensor3D<float> block_input{state.embed_out.data, {B, T, D}, {T * D, D, 1}};

  for (int64_t layer = 0; layer < config.n_layers; ++layer) {
    if (state.block_inputs.size() <= static_cast<size_t>(layer) ||
        state.blocks.size() <= static_cast<size_t>(layer)) {
      return Status(StatusCode::kInvalidArgument,
                    "gpt_forward: missing per-layer forward state");
    }
    Tensor3D<const float> x_in{block_input.data, block_input.shape,
                                block_input.stride};
    GPT_RETURN_IF_ERROR(
        copy_3d_f32(x_in, state.block_inputs[layer], stream));
    GPT_RETURN_IF_ERROR(block_forward(
        x_in, config, params.layers[layer],
        block_input,  // output overwrites input for next layer
        state.blocks[layer], stream));
  }

  // Step 4: Final LayerNorm.
  {
    Tensor2D<const float> x_2d{block_input.data, {B * T, D}, {D, 1}};
    Tensor2D<float> ln_2d{state.final_ln_out.data, {B * T, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(ops::layernorm_forward(
        x_2d, params.final_ln, ln_2d, state.final_ln_state, stream));
  }

  // Step 5: LM head — logits = final_ln_out @ lm_head^T.
  {
    Tensor2D<const float> x_2d{state.final_ln_out.data,
                                {B * T, D}, {D, 1}};
    ops::LinearParams lm;
    lm.weight = params.lm_head;
    lm.bias = Tensor1D<float>{nullptr, {config.vocab_size}, {1}};
    lm.use_bias = false;

    GPT_RETURN_IF_ERROR(
        ops::linear_forward(x_2d, lm, state.logits, stream));
  }

  return Status::Ok();
}

Status gpt_backward(Tensor2D<const float> d_logits,
                    Tensor2D<const int32_t> input_ids,
                    const GPTConfig& config,
                    const GPTParams& params,
                    const GPTForwardState& state,
                    GPTGrads& grads,
                    DeviceScratchArena& scratch,
                    const CudaStream& stream) {
  int64_t B = input_ids.shape[0];
  int64_t T = input_ids.shape[1];
  int64_t D = config.d_model;
  int64_t N = B * T;

  auto d_final_ln_out_ptr = scratch.alloc_f32(N * D);
  if (!d_final_ln_out_ptr.ok()) return d_final_ln_out_ptr.status();
  auto d_embed_ptr = scratch.alloc_f32(N * D);
  if (!d_embed_ptr.ok()) return d_embed_ptr.status();

  Tensor3D<float> d_embed{
      d_embed_ptr.value(), {B, T, D}, {T * D, D, 1}};

  // LM head backward: logits = final_ln_out @ lm_head^T.
  {
    Tensor2D<const float> final_ln_2d{
        state.final_ln_out.data, {N, D}, {D, 1}};
    Tensor2D<float> d_final_ln_2d{
        d_final_ln_out_ptr.value(), {N, D}, {D, 1}};

    ops::LinearParams lm;
    lm.weight = params.lm_head;
    lm.bias = Tensor1D<float>{nullptr, {config.vocab_size}, {1}};
    lm.use_bias = false;

    ops::LinearGrads lm_grads;
    lm_grads.d_weight = grads.d_lm_head;
    lm_grads.d_bias = Tensor1D<float>{nullptr, {config.vocab_size}, {1}};
    GPT_RETURN_IF_ERROR(ops::linear_backward(
        final_ln_2d, d_logits, lm, d_final_ln_2d, lm_grads, stream));
  }

  // Final LayerNorm backward. state.embed_out holds the final block output
  // because gpt_forward applies blocks in place in that buffer.
  {
    Tensor2D<const float> d_final_ln_2d{
        d_final_ln_out_ptr.value(), {N, D}, {D, 1}};
    Tensor2D<const float> final_ln_input_2d{
        state.embed_out.data, {N, D}, {D, 1}};
    Tensor2D<float> d_embed_2d{d_embed.data, {N, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(ops::layernorm_backward(
        d_final_ln_2d, final_ln_input_2d, params.final_ln, state.final_ln_state,
        d_embed_2d, grads.d_final_ln_gamma, grads.d_final_ln_beta, stream));
  }

  if (config.n_layers > 0) {
    if (state.block_inputs.size() < static_cast<size_t>(config.n_layers) ||
        state.blocks.size() < static_cast<size_t>(config.n_layers) ||
        grads.layers.size() < static_cast<size_t>(config.n_layers)) {
      return Status(StatusCode::kInvalidArgument,
                    "gpt_backward: missing per-layer backward state");
    }
    for (int64_t layer = config.n_layers - 1; layer >= 0; --layer) {
      Tensor3D<const float> block_input{
          state.block_inputs[layer].data, {B, T, D}, {T * D, D, 1}};
      Tensor3D<const float> d_block_out{
          d_embed.data, {B, T, D}, {T * D, D, 1}};
      Tensor3D<float> d_block_in{
          d_embed.data, {B, T, D}, {T * D, D, 1}};
      GPT_RETURN_IF_ERROR(block_backward(
          d_block_out, block_input, config, params.layers[layer],
          state.blocks[layer], d_block_in, grads.layers[layer], scratch,
          stream));
    }
  }

  // Position embedding and token embedding both receive d_embed.
  GPT_RETURN_IF_ERROR(
      position_embedding_backward_f32(d_embed, grads.d_position_embedding, stream));

  {
    Tensor1D<const int32_t> ids_flat{input_ids.data, {N}, {1}};
    Tensor2D<const float> d_embed_2d{d_embed.data, {N, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(ops::embedding_backward(
        d_embed_2d, ids_flat, grads.d_token_embedding, stream));
  }

  return Status::Ok();
}

}  // namespace model
}  // namespace gpt
