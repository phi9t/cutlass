// Layer 4 — Attention subsystem: CUDA kernels specific to attention.
//
// Contains utility kernels that are attention-specific
// (e.g., causal mask generation, in-place causal mask application).
// The main forward orchestration lives in attention.cc.

#include "src/attention/attention_forward.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace gpt {
namespace attention {

namespace {

// Apply causal mask in-place: set scores[bh, i, j] = -1e9 where j > i.
// Scores layout: [BH, T, T] row-major (BH = B * H).
__global__ void apply_causal_mask_kernel(float* __restrict__ scores,
                                         int64_t BH,
                                         int64_t T) {
  int64_t bh = blockIdx.z;
  int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (bh < BH && i < T && j < T) {
    if (j > i) {
      scores[bh * T * T + i * T + j] = -1e9f;
    }
  }
}

}  // namespace

Status apply_causal_mask_f32(float* scores,
                             int64_t B, int64_t H, int64_t T,
                             const CudaStream& stream) {
  int64_t BH = B * H;
  constexpr int kTile = 16;
  dim3 block(kTile, kTile);
  dim3 grid(static_cast<unsigned>((T + kTile - 1) / kTile),
            static_cast<unsigned>((T + kTile - 1) / kTile),
            static_cast<unsigned>(BH));
  apply_causal_mask_kernel<<<grid, block, 0, stream.get()>>>(
      scores, BH, T);
  return Status::Ok();
}

}  // namespace attention
}  // namespace gpt
