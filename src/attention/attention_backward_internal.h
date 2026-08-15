#pragma once

// Internal CUDA helpers for attention backward orchestration.

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace attention {

// d_context_merged [B,T,D] -> d_context [B,H,T,Dh]
Status split_merged_heads_grad_f32(Tensor3D<const float> input,
                                   Tensor4D<float> output,
                                   int64_t n_heads,
                                   const CudaStream& stream);

// Scale d_scores by the forward score scale. Causal masked positions already
// have zero probability, so softmax backward has produced zero gradients there.
Status scale_scores_grad_f32(Tensor4D<float> scores,
                             float scale,
                             const CudaStream& stream);

// Pack head-layout dQ/dK/dV into the flat [B,T,3D] QKV gradient buffer.
Status merge_qkv_grads_f32(Tensor4D<const float> dQ,
                           Tensor4D<const float> dK,
                           Tensor4D<const float> dV,
                           Tensor3D<float> dQKV,
                           const CudaStream& stream);

}  // namespace attention
}  // namespace gpt
