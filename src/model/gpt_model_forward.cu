// Layer 5 — GPT model forward CUDA helpers.

#include "src/model/gpt_model_forward_internal.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace gpt {
namespace model {

namespace {

__global__ void add_position_embedding_kernel(
    float* __restrict__ embed,
    const float* __restrict__ position_embedding,
    int64_t B, int64_t T, int64_t D) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * T * D;
  if (idx >= total) return;

  int64_t d = idx % D;
  int64_t t = (idx / D) % T;
  embed[idx] += position_embedding[t * D + d];
}

}  // namespace

Status add_position_embedding_f32(Tensor3D<float> embed,
                                  Tensor2D<const float> position_embedding,
                                  const CudaStream& stream) {
  int64_t B = embed.shape[0];
  int64_t T = embed.shape[1];
  int64_t D = embed.shape[2];
  if (position_embedding.shape[0] < T || position_embedding.shape[1] != D) {
    return Status(StatusCode::kInvalidArgument,
                  "add_position_embedding_f32: shape mismatch");
  }

  int64_t total = B * T * D;
  int block = 256;
  int grid = static_cast<int>((total + block - 1) / block);
  add_position_embedding_kernel<<<grid, block, 0, stream.get()>>>(
      embed.data, position_embedding.data, B, T, D);
  GPT_CHECK_CUDA(cudaGetLastError());
  return Status::Ok();
}

}  // namespace model
}  // namespace gpt
