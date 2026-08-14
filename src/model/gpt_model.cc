// Layer 5 — Model: full GPT model implementation.

#include "src/model/gpt_model.h"

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
    // Output of each block becomes input to the next.
    // For v1, alternate between two buffers or write in-place.
    // TODO: Proper buffer management.
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
  // TODO(m6): Implement full GPT backward.
  //
  // Stages (reverse order):
  //   5. LM head backward → d_final_ln_out, d_lm_head
  //   4. Final LN backward → d_block_out, d_final_ln_gamma/beta
  //   3. Block backward (layer N-1 to 0) → d_embed_out, per-layer grads
  //   2. Position embedding backward → d_position_embedding
  //   1. Token embedding backward → d_token_embedding

  return Status(StatusCode::kNotImplemented,
                "gpt_backward not yet implemented");
}

}  // namespace model
}  // namespace gpt
