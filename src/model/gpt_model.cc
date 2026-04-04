// Layer 5 — Model: full GPT model implementation.

#include "src/model/gpt_model.h"

#include "src/kernels/elementwise.h"
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
  // position_embedding is [max_seq_len, D]. Slice [0:T, :] and broadcast-add
  // across the batch dimension.
  {
    // For each batch element b and position t:
    //   embed_out[b, t, :] += position_embedding[t, :]
    // Position embedding slice [0:T, :] has stride {D, 1} in the full table.
    // We treat embed_out as [B*T, D] and tile the [T, D] slice B times via
    // a flat vec_add loop: for each batch, add the same T*D block.
    for (int64_t b = 0; b < B; ++b) {
      Tensor1D<const float> embed_slice{
          state.embed_out.data + b * T * D, {T * D}, {1}};
      Tensor1D<const float> pos_slice{
          params.position_embedding.data, {T * D}, {1}};
      Tensor1D<float> out_slice{
          state.embed_out.data + b * T * D, {T * D}, {1}};
      GPT_RETURN_IF_ERROR(
          kernels::vec_add_f32(embed_slice, pos_slice, out_slice, stream));
    }
  }

  // Step 3: Transformer blocks.
  Tensor3D<float> block_input{state.embed_out.data, {B, T, D}, {T * D, D, 1}};

  for (int64_t layer = 0; layer < config.n_layers; ++layer) {
    // Save block input for backward pass before it gets overwritten.
    size_t block_bytes = static_cast<size_t>(B * T * D) * sizeof(float);
    cudaError_t cp_err = cudaMemcpyAsync(
        state.block_inputs[layer].data, block_input.data,
        block_bytes, cudaMemcpyDeviceToDevice, stream.get());
    if (cp_err != cudaSuccess) {
      return Status(StatusCode::kCudaError, cudaGetErrorString(cp_err));
    }

    Tensor3D<const float> x_in{block_input.data, block_input.shape,
                                block_input.stride};
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
                    const CudaStream& stream) {
  int64_t B = input_ids.shape[0];
  int64_t T = input_ids.shape[1];
  int64_t D = config.d_model;
  int64_t V = config.vocab_size;

  // -----------------------------------------------------------------------
  // Stage 5: LM head backward.
  //   Forward was: logits = final_ln_out @ lm_head^T  (no bias)
  //   Backward:    d_final_ln_out = d_logits @ lm_head
  //                d_lm_head = d_logits^T @ final_ln_out
  // -----------------------------------------------------------------------
  // Write d_final_ln_out into state.final_ln_out buffer (reuse).
  {
    Tensor2D<const float> ln_out_2d{state.final_ln_out.data,
                                     {B * T, D}, {D, 1}};

    ops::LinearParams lm;
    lm.weight = params.lm_head;
    lm.bias = Tensor1D<float>{nullptr, {V}, {1}};
    lm.use_bias = false;

    ops::LinearGrads lg;
    lg.d_weight = grads.d_lm_head;
    lg.d_bias = Tensor1D<float>{nullptr, {V}, {1}};

    Tensor2D<float> d_ln_out{state.final_ln_out.data,
                              {B * T, D}, {D, 1}};

    GPT_RETURN_IF_ERROR(
        ops::linear_backward(ln_out_2d, d_logits, lm, d_ln_out, lg, stream));
  }

  // -----------------------------------------------------------------------
  // Stage 4: Final LayerNorm backward.
  //   Forward was: final_ln_out = LN(block_output)
  //   Backward:    d_block_out = LN_backward(d_final_ln_out)
  // -----------------------------------------------------------------------
  // The LN input was the last block's output. In forward, it was block_input
  // (which points to state.embed_out). We write d_block_out into embed_out.
  {
    Tensor2D<const float> d_ln_2d{state.final_ln_out.data,
                                   {B * T, D}, {D, 1}};
    // The input to final LN was the last transformer block's output,
    // which resided in state.embed_out (the block_input buffer).
    Tensor2D<const float> block_out_2d{state.embed_out.data,
                                        {B * T, D}, {D, 1}};
    Tensor2D<float> d_block_out{state.embed_out.data,
                                 {B * T, D}, {D, 1}};

    GPT_RETURN_IF_ERROR(ops::layernorm_backward(
        d_ln_2d, block_out_2d, params.final_ln, state.final_ln_state,
        d_block_out, grads.d_final_ln_gamma, grads.d_final_ln_beta, stream));
  }

  // -----------------------------------------------------------------------
  // Stage 3: Block backward (layer N-1 to 0).
  //   Each block: d_input = block_backward(d_output, x_saved)
  //   Per-layer inputs were saved during forward in state.block_inputs.
  // -----------------------------------------------------------------------
  {
    Tensor3D<float> d_block{state.embed_out.data,
                             {B, T, D}, {T * D, D, 1}};

    for (int64_t layer = config.n_layers - 1; layer >= 0; --layer) {
      Tensor3D<float> dx{state.final_ln_out.data,
                          {B, T, D}, {T * D, D, 1}};

      Tensor3D<const float> d_block_in{d_block.data,
                                        {B, T, D}, {T * D, D, 1}};
      Tensor3D<const float> x_saved{state.block_inputs[layer].data,
                                     {B, T, D}, {T * D, D, 1}};

      GPT_RETURN_IF_ERROR(block_backward(
          d_block_in, x_saved, config, params.layers[layer],
          state.blocks[layer], dx, grads.layers[layer], stream));

      // Copy dx back to d_block for the next layer's backward.
      size_t bytes = static_cast<size_t>(B * T * D) * sizeof(float);
      cudaError_t err = cudaMemcpyAsync(d_block.data, dx.data, bytes,
                                         cudaMemcpyDeviceToDevice, stream.get());
      if (err != cudaSuccess) {
        return Status(StatusCode::kCudaError, cudaGetErrorString(err));
      }
    }
  }

  // -----------------------------------------------------------------------
  // Stage 2: Position embedding backward.
  //   d_position_embedding[t, :] += sum_b(d_embed_out[b, t, :])
  // -----------------------------------------------------------------------
  {
    // d_embed_out is in state.embed_out. Sum across batch dimension.
    for (int64_t b = 0; b < B; ++b) {
      Tensor2D<const float> d_pos_slice{
          state.embed_out.data + b * T * D, {T, D}, {D, 1}};
      // Accumulate into d_position_embedding[0:T, :].
      // Use vec_add to accumulate.
      Tensor1D<const float> src{d_pos_slice.data, {T * D}, {1}};
      Tensor1D<const float> dst_cur{grads.d_position_embedding.data,
                                     {T * D}, {1}};
      Tensor1D<float> dst{grads.d_position_embedding.data, {T * D}, {1}};
      GPT_RETURN_IF_ERROR(kernels::vec_add_f32(dst_cur, src, dst, stream));
    }
  }

  // -----------------------------------------------------------------------
  // Stage 1: Token embedding backward.
  //   Scatter-add d_embed_out into d_token_embedding using input_ids.
  // -----------------------------------------------------------------------
  {
    Tensor2D<const float> d_embed_2d{state.embed_out.data,
                                      {B * T, D}, {D, 1}};
    Tensor1D<const int32_t> ids_flat{input_ids.data, {B * T}, {1}};
    GPT_RETURN_IF_ERROR(ops::embedding_backward(
        d_embed_2d, ids_flat, grads.d_token_embedding, stream));
  }

  return Status::Ok();
}

}  // namespace model
}  // namespace gpt
