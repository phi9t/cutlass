#pragma once

// Internal CUDA helpers for GPT model forward orchestration.

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace model {

// In-place: embed[b,t,d] += position_embedding[t,d].
Status add_position_embedding_f32(Tensor3D<float> embed,
                                  Tensor2D<const float> position_embedding,
                                  const CudaStream& stream);

}  // namespace model
}  // namespace gpt
