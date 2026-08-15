// Layer 5 — Model: full GPT model implementation.

#include "src/model/gpt_model.h"

#include <cuda_runtime.h>
#include <vector>

#include "src/model/gpt_model_forward_internal.h"
#include "src/ops/embedding.h"
#include "src/ops/layernorm.h"
#include "src/ops/linear.h"

namespace gpt {
namespace model {

namespace {

class DeviceScratch {
 public:
  DeviceScratch() = default;
  ~DeviceScratch() {
    for (float* ptr : buffers_) cudaFree(ptr);
  }

  Result<float*> allocate(int64_t count) {
    float* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, static_cast<size_t>(count) * sizeof(float));
    if (err != cudaSuccess) {
      return Status(StatusCode::kCudaError, cudaGetErrorString(err));
    }
    buffers_.push_back(ptr);
    return ptr;
  }

  DeviceScratch(const DeviceScratch&) = delete;
  DeviceScratch& operator=(const DeviceScratch&) = delete;

 private:
  std::vector<float*> buffers_;
};

}  // namespace

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
  if (config.n_layers != 0) {
    return Status(StatusCode::kNotImplemented,
                  "gpt_backward for transformer layers needs saved block inputs");
  }

  int64_t B = input_ids.shape[0];
  int64_t T = input_ids.shape[1];
  int64_t D = config.d_model;
  int64_t N = B * T;

  DeviceScratch scratch;
  auto d_final_ln_out_ptr = scratch.allocate(N * D);
  if (!d_final_ln_out_ptr.ok()) return d_final_ln_out_ptr.status();
  auto d_embed_ptr = scratch.allocate(N * D);
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

  // Final LayerNorm backward. With zero transformer layers, embed_out is the
  // saved input to the final LN.
  {
    Tensor2D<const float> d_final_ln_2d{
        d_final_ln_out_ptr.value(), {N, D}, {D, 1}};
    Tensor2D<const float> embed_2d{state.embed_out.data, {N, D}, {D, 1}};
    Tensor2D<float> d_embed_2d{d_embed.data, {N, D}, {D, 1}};
    GPT_RETURN_IF_ERROR(ops::layernorm_backward(
        d_final_ln_2d, embed_2d, params.final_ln, state.final_ln_state,
        d_embed_2d, grads.d_final_ln_gamma, grads.d_final_ln_beta, stream));
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
