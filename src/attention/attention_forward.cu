// Layer 4 — Attention subsystem: CUDA kernels specific to attention.
//
// Currently contains utility kernels that are attention-specific
// (e.g., causal mask generation, fused scale+mask).
// The main forward orchestration lives in attention.cc.

#include "src/attention/attention_forward_internal.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace gpt {
namespace attention {

namespace {

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

Status scale_and_causal_mask_scores_f32(Tensor4D<float> scores,
                                        float scale,
                                        const CudaStream& stream) {
  if (scores.shape[2] != scores.shape[3]) {
    return Status(StatusCode::kInvalidArgument,
                  "scale_and_causal_mask_scores_f32: scores must be square");
  }
  int64_t BH = scores.shape[0] * scores.shape[1];
  int64_t T = scores.shape[2];
  dim3 block(16, 16);
  dim3 grid((T + block.x - 1) / block.x, (T + block.y - 1) / block.y, BH);
  scale_and_causal_mask_kernel<<<grid, block, 0, stream.get()>>>(
      scores.data, scale, BH, T);
  GPT_CHECK_CUDA(cudaGetLastError());
  return Status::Ok();
}

}  // namespace attention
}  // namespace gpt
