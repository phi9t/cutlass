// Layer 3 — Composite neural ops: LayerNorm CUDA implementation.

#include "src/ops/layernorm.h"

#include "src/kernels/reductions.h"

#include <cuda_runtime.h>

namespace gpt {
namespace ops {

namespace {

// Forward kernel: normalize + affine.
// One thread-block per row for simplicity in v1.
__global__ void layernorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    float* __restrict__ y,
    int64_t rows, int64_t D) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;

  float m = mean[row];
  float s = inv_std[row];

  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (x[row * D + d] - m) * s;
    y[row * D + d] = gamma[d] * x_hat + beta[d];
  }
}

// Backward kernel: compute dx, accumulate dgamma/dbeta.
// This is the v1 reference path — not optimized for large D.
__global__ void layernorm_backward_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ inv_std,
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int64_t rows, int64_t D) {
  int64_t row = blockIdx.x;
  if (row >= rows) return;

  float m = mean[row];
  float s = inv_std[row];

  // Pass 1: compute dot(dy, x_hat) and dot(dy, gamma) for this row.
  float sum_dy_xhat = 0.0f;
  float sum_dy = 0.0f;
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (x[row * D + d] - m) * s;
    float dy_val = dy[row * D + d];
    sum_dy_xhat += dy_val * gamma[d] * x_hat;
    sum_dy += dy_val * gamma[d];
  }

  // Warp reduction (simplified — full implementation would use shared mem).
  // For v1 reference with blockDim.x=1 this is trivially correct.

  // Pass 2: compute dx.
  float inv_D = 1.0f / static_cast<float>(D);
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (x[row * D + d] - m) * s;
    float dy_val = dy[row * D + d];
    dx[row * D + d] = s * (gamma[d] * dy_val
                           - inv_D * sum_dy
                           - inv_D * x_hat * sum_dy_xhat);
  }

  // Accumulate dgamma, dbeta (atomics — slow but correct for v1).
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (x[row * D + d] - m) * s;
    float dy_val = dy[row * D + d];
    atomicAdd(&dgamma[d], dy_val * x_hat);
    atomicAdd(&dbeta[d], dy_val);
  }
}

}  // namespace

Status layernorm_forward(Tensor2D<const float> x,
                         const LayerNormParams& params,
                         Tensor2D<float> y,
                         LayerNormState& state,
                         const CudaStream& stream) {
  int64_t rows = x.shape[0];
  int64_t D = x.shape[1];

  if (params.gamma.shape[0] != D || params.beta.shape[0] != D) {
    return Status(StatusCode::kInvalidArgument,
                  "layernorm_forward: gamma/beta dim mismatch");
  }

  // Compute mean and inv_std using reduction kernels.
  GPT_RETURN_IF_ERROR(kernels::row_mean_inv_std_f32(
      x, state.mean, state.inv_std, params.eps, stream));

  // Normalize + affine.
  int block = std::min(static_cast<int>(D), 256);
  layernorm_forward_kernel<<<static_cast<int>(rows), block, 0, stream.get()>>>(
      x.data, params.gamma.data, params.beta.data,
      state.mean.data, state.inv_std.data, y.data, rows, D);
  return Status::Ok();
}

Status layernorm_backward(Tensor2D<const float> dy,
                          Tensor2D<const float> x,
                          const LayerNormParams& params,
                          const LayerNormState& state,
                          Tensor2D<float> dx,
                          Tensor1D<float> dgamma,
                          Tensor1D<float> dbeta,
                          const CudaStream& stream) {
  int64_t rows = x.shape[0];
  int64_t D = x.shape[1];

  // v1: one block per row, single thread for correctness.
  // TODO(perf): warp-level or block-level reduction for large D.
  layernorm_backward_kernel<<<static_cast<int>(rows), 1, 0, stream.get()>>>(
      dy.data, x.data, params.gamma.data,
      state.mean.data, state.inv_std.data,
      dx.data, dgamma.data, dbeta.data, rows, D);
  return Status::Ok();
}

}  // namespace ops
}  // namespace gpt
