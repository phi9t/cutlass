#pragma once

// Internal CUDA helpers for attention forward orchestration.

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace attention {

// scores[b,h,i,j] = scale*scores[b,h,i,j] for valid causal positions and
// -1e9f where j > i.
Status scale_and_causal_mask_scores_f32(Tensor4D<float> scores,
                                        float scale,
                                        const CudaStream& stream);

}  // namespace attention
}  // namespace gpt
