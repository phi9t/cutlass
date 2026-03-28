// Layer 3 — Composite neural ops: embedding implementation.

#include "src/ops/embedding.h"

#include "src/kernels/elementwise.h"

#include <cuda_runtime.h>

namespace gpt {
namespace ops {

Status embedding_forward(Tensor2D<const float> table,
                         Tensor1D<const int32_t> indices,
                         Tensor2D<float> output,
                         const CudaStream& stream) {
  return kernels::gather_f32(table, indices, output, stream);
}

// ---------------------------------------------------------------------------
// Scatter-add kernel for embedding backward.
// ---------------------------------------------------------------------------

namespace {

__global__ void scatter_add_kernel(const float* __restrict__ dY,
                                   const int32_t* __restrict__ indices,
                                   float* __restrict__ d_table,
                                   int64_t n, int64_t dim) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  int32_t row = indices[idx];
  const float* src = dY + idx * dim;
  float* dst = d_table + static_cast<int64_t>(row) * dim;

  for (int64_t d = 0; d < dim; ++d) {
    atomicAdd(&dst[d], src[d]);
  }
}

}  // namespace

Status embedding_backward(Tensor2D<const float> dY,
                          Tensor1D<const int32_t> indices,
                          Tensor2D<float> d_table,
                          const CudaStream& stream) {
  int64_t n = indices.shape[0];
  int64_t dim = dY.shape[1];
  if (dY.shape[0] != n) {
    return Status(StatusCode::kInvalidArgument,
                  "embedding_backward: dY/indices length mismatch");
  }
  if (d_table.shape[1] != dim) {
    return Status(StatusCode::kInvalidArgument,
                  "embedding_backward: d_table dim mismatch");
  }

  int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);
  scatter_add_kernel<<<grid, block, 0, stream.get()>>>(
      dY.data, indices.data, d_table.data, n, dim);
  return Status::Ok();
}

}  // namespace ops
}  // namespace gpt
