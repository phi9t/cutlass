// Layer 3 — Composite neural ops: GELU CUDA implementation.

#include "src/ops/gelu.h"

#include <cuda_runtime.h>
#include <cmath>

namespace gpt {
namespace ops {

namespace {

static constexpr float kSqrt2OverPi = 0.7978845608028654f;
static constexpr float kCoeff = 0.044715f;

__device__ __forceinline__ float gelu_fwd(float x) {
  float x3 = x * x * x;
  float inner = kSqrt2OverPi * (x + kCoeff * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float gelu_bwd(float x) {
  float x2 = x * x;
  float x3 = x2 * x;
  float inner = kSqrt2OverPi * (x + kCoeff * x3);
  float tanh_val = tanhf(inner);
  float sech2 = 1.0f - tanh_val * tanh_val;
  float d_inner = kSqrt2OverPi * (1.0f + 3.0f * kCoeff * x2);
  return 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
}

__global__ void gelu_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = gelu_fwd(x[i]);
}

__global__ void gelu_backward_kernel(const float* __restrict__ dy,
                                     const float* __restrict__ x,
                                     float* __restrict__ dx, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dx[i] = dy[i] * gelu_bwd(x[i]);
}

}  // namespace

Status gelu_forward(Tensor1D<const float> x, Tensor1D<float> y,
                    const CudaStream& stream) {
  int64_t n = x.shape[0];
  if (y.shape[0] != n) {
    return Status(StatusCode::kInvalidArgument, "gelu_forward shape mismatch");
  }
  int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);
  gelu_forward_kernel<<<grid, block, 0, stream.get()>>>(x.data, y.data, n);
  return Status::Ok();
}

Status gelu_backward(Tensor1D<const float> dy, Tensor1D<const float> x,
                     Tensor1D<float> dx, const CudaStream& stream) {
  int64_t n = x.shape[0];
  if (dy.shape[0] != n || dx.shape[0] != n) {
    return Status(StatusCode::kInvalidArgument,
                  "gelu_backward shape mismatch");
  }
  int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);
  gelu_backward_kernel<<<grid, block, 0, stream.get()>>>(
      dy.data, x.data, dx.data, n);
  return Status::Ok();
}

}  // namespace ops
}  // namespace gpt
