// Layer 4 — Attention subsystem: CUDA kernels specific to attention.
//
// Contains utility kernels that are attention-specific
// (e.g., causal mask generation, in-place causal mask application).
// The main forward orchestration lives in attention.cc.

#include "src/attention/attention_forward.h"

#include <algorithm>
#include <climits>
#include <cuda_runtime.h>
#include <cstdint>

namespace gpt {
namespace attention {

namespace {

// Apply causal mask in-place: set scores[..., i, j] = -1e9 where j > i.
// Uses a 1D grid over (BH * T * T) to avoid grid.z overflow for large BH.
__global__ void apply_causal_mask_kernel(float* __restrict__ scores,
                                         int64_t BH,
                                         int64_t T) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = BH * T * T;
  if (idx >= total) return;

  int64_t j = idx % T;
  int64_t i = (idx / T) % T;
  if (j > i) {
    scores[idx] = -1e9f;
  }
}

}  // namespace

Status apply_causal_mask_f32(float* scores,
                             int64_t B, int64_t H, int64_t T,
                             const CudaStream& stream) {
  int64_t total = B * H * T * T;
  int block = 256;
  int64_t g = (total + block - 1) / block;
  int grid = static_cast<int>(std::min(g, static_cast<int64_t>(INT_MAX)));
  apply_causal_mask_kernel<<<grid, block, 0, stream.get()>>>(
      scores, B * H, T);
  return Status::Ok();
}

}  // namespace attention
}  // namespace gpt
