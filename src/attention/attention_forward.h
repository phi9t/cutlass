#pragma once

// Layer 4 — Attention subsystem: forward-specific kernel declarations.

#include <cstdint>

#include "src/core/status.h"
#include "src/core/stream.h"

namespace gpt {
namespace attention {

// Apply causal mask in-place on pre-computed scores [B*H, T, T].
// Sets scores[bh, i, j] = -1e9 for all j > i (future positions).
Status apply_causal_mask_f32(float* scores,
                             int64_t B, int64_t H, int64_t T,
                             const CudaStream& stream);

}  // namespace attention
}  // namespace gpt
