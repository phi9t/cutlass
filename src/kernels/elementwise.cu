// Layer 2 — Primitive kernels: elementwise CUDA implementations.

#include "src/kernels/elementwise.h"

#include <cuda_runtime.h>
#include <cmath>

namespace gpt {
namespace kernels {

namespace {

static constexpr int kBlockSize = 256;

__global__ void vec_add_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void vec_mul_kernel(const float* __restrict__ a,
                               const float* __restrict__ b,
                               float* __restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * b[i];
}

__global__ void vec_scale_kernel(const float* __restrict__ x, float alpha,
                                 float* __restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = alpha * x[i];
}

__global__ void vec_exp_kernel(const float* __restrict__ x,
                               float* __restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = expf(x[i]);
}

__global__ void masked_fill_kernel(const float* __restrict__ x,
                                   const int8_t* __restrict__ mask,
                                   float fill_value,
                                   float* __restrict__ out, int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = mask[i] ? fill_value : x[i];
}

__global__ void gather_kernel(const float* __restrict__ table,
                              const int32_t* __restrict__ indices,
                              float* __restrict__ out,
                              int64_t n, int64_t dim) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int32_t idx = indices[i];
  const float* src = table + static_cast<int64_t>(idx) * dim;
  float* dst = out + i * dim;
  for (int64_t d = 0; d < dim; ++d) {
    dst[d] = src[d];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

Status vec_add_f32(Tensor1D<const float> a, Tensor1D<const float> b,
                   Tensor1D<float> out, const CudaStream& stream) {
  int64_t n = a.shape[0];
  if (b.shape[0] != n || out.shape[0] != n) {
    return Status(StatusCode::kInvalidArgument, "vec_add shape mismatch");
  }
  int grid = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
  vec_add_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      a.data, b.data, out.data, n);
  return Status::Ok();
}

Status vec_mul_f32(Tensor1D<const float> a, Tensor1D<const float> b,
                   Tensor1D<float> out, const CudaStream& stream) {
  int64_t n = a.shape[0];
  if (b.shape[0] != n || out.shape[0] != n) {
    return Status(StatusCode::kInvalidArgument, "vec_mul shape mismatch");
  }
  int grid = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
  vec_mul_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      a.data, b.data, out.data, n);
  return Status::Ok();
}

Status vec_scale_f32(Tensor1D<const float> x, float alpha,
                     Tensor1D<float> out, const CudaStream& stream) {
  int64_t n = x.shape[0];
  if (out.shape[0] != n) {
    return Status(StatusCode::kInvalidArgument, "vec_scale shape mismatch");
  }
  int grid = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
  vec_scale_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      x.data, alpha, out.data, n);
  return Status::Ok();
}

Status vec_exp_f32(Tensor1D<const float> x, Tensor1D<float> out,
                   const CudaStream& stream) {
  int64_t n = x.shape[0];
  if (out.shape[0] != n) {
    return Status(StatusCode::kInvalidArgument, "vec_exp shape mismatch");
  }
  int grid = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
  vec_exp_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      x.data, out.data, n);
  return Status::Ok();
}

Status masked_fill_f32(Tensor1D<const float> x,
                       Tensor1D<const int8_t> mask,
                       float fill_value,
                       Tensor1D<float> out,
                       const CudaStream& stream) {
  int64_t n = x.shape[0];
  if (mask.shape[0] != n || out.shape[0] != n) {
    return Status(StatusCode::kInvalidArgument, "masked_fill shape mismatch");
  }
  int grid = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
  masked_fill_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      x.data, mask.data, fill_value, out.data, n);
  return Status::Ok();
}

Status gather_f32(Tensor2D<const float> table,
                  Tensor1D<const int32_t> indices,
                  Tensor2D<float> out,
                  const CudaStream& stream) {
  int64_t n = indices.shape[0];
  int64_t dim = table.shape[1];
  if (out.shape[0] != n || out.shape[1] != dim) {
    return Status(StatusCode::kInvalidArgument, "gather output shape mismatch");
  }
  int grid = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
  gather_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      table.data, indices.data, out.data, n, dim);
  return Status::Ok();
}

}  // namespace kernels
}  // namespace gpt
