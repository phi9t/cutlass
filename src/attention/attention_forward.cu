// Layer 4 — Attention subsystem: CUDA kernels specific to attention.
//
// Currently contains utility kernels that are attention-specific
// (e.g., causal mask generation, fused scale+mask).
// The main forward orchestration lives in attention.cc.

#include <cuda_runtime.h>
#include <cstdint>

namespace gpt {
namespace attention {

namespace {

// Generate a causal mask for sequence length T.
// mask[i, j] = 1 (masked) if j > i, else 0.
// Output: [T, T] int8 mask.
__global__ void generate_causal_mask_kernel(int8_t* __restrict__ mask,
                                            int64_t T) {
  int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < T && j < T) {
    mask[i * T + j] = (j > i) ? 1 : 0;
  }
}

// Fused scale + causal mask fill:
//   scores[b, h, i, j] *= scale
//   scores[b, h, i, j] = -inf  if j > i
__global__ void scale_and_causal_mask_kernel(
    float* __restrict__ scores,
    float scale,
    int64_t BH,   // B * H
    int64_t T) {
  int64_t bh = blockIdx.z;
  int64_t i = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t j = blockIdx.x * blockDim.x + threadIdx.x;

  if (bh < BH && i < T && j < T) {
    int64_t idx = bh * T * T + i * T + j;
    if (j > i) {
      scores[idx] = -1e9f;  // effectively -inf for softmax
    } else {
      scores[idx] *= scale;
    }
  }
}

}  // namespace

// TODO: Expose these kernels through a clean internal API and wire
// them into attention.cc once the causal mask path is enabled.

}  // namespace attention
}  // namespace gpt
