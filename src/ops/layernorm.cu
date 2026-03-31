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
// Uses warp-level and block-level reductions for parallel summation.
// blockDim.x must be <= 256 (max 8 warps).
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
  // Shared memory for cross-warp reduction (max 8 warps = 256 threads).
  __shared__ float s_dy_xhat[8];
  __shared__ float s_dy_gamma[8];

  int64_t row = blockIdx.x;
  if (row >= rows) return;

  float m = mean[row];
  float s = inv_std[row];

  // Pass 1: compute partial sums of dot(dy*gamma, x_hat) and sum(dy*gamma).
  float sum_dy_xhat = 0.0f;
  float sum_dy = 0.0f;
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (x[row * D + d] - m) * s;
    float dy_val = dy[row * D + d];
    sum_dy_xhat += dy_val * gamma[d] * x_hat;
    sum_dy += dy_val * gamma[d];
  }

  // Warp-level reduction using shuffle.
  // Use __activemask() to handle partial warps (blockDim.x not a multiple of 32).
  unsigned active = __activemask();
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_dy_xhat += __shfl_down_sync(active, sum_dy_xhat, offset);
    sum_dy += __shfl_down_sync(active, sum_dy, offset);
  }

  // Block-level reduction via shared memory (for blocks with multiple warps).
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int num_warps = (blockDim.x + 31) / 32;

  if (lane_id == 0) {
    s_dy_xhat[warp_id] = sum_dy_xhat;
    s_dy_gamma[warp_id] = sum_dy;
  }
  __syncthreads();

  // First warp reduces across warps.
  if (warp_id == 0) {
    sum_dy_xhat = (lane_id < num_warps) ? s_dy_xhat[lane_id] : 0.0f;
    sum_dy = (lane_id < num_warps) ? s_dy_gamma[lane_id] : 0.0f;
    // All lanes in warp 0 participate; inactive values are zero so full mask is safe.
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum_dy_xhat += __shfl_down_sync(active, sum_dy_xhat, offset);
      sum_dy += __shfl_down_sync(active, sum_dy, offset);
    }
    // Broadcast final result back to shared memory.
    if (lane_id == 0) {
      s_dy_xhat[0] = sum_dy_xhat;
      s_dy_gamma[0] = sum_dy;
    }
  }
  __syncthreads();

  sum_dy_xhat = s_dy_xhat[0];
  sum_dy = s_dy_gamma[0];

  // Pass 2: compute dx.
  float inv_D = 1.0f / static_cast<float>(D);
  for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
    float x_hat = (x[row * D + d] - m) * s;
    float dy_val = dy[row * D + d];
    dx[row * D + d] = s * (gamma[d] * dy_val
                           - inv_D * sum_dy
                           - inv_D * x_hat * sum_dy_xhat);
  }

  // Accumulate dgamma, dbeta (atomics across rows).
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

  int bwd_block = std::min(static_cast<int>(D), 256);
  layernorm_backward_kernel<<<static_cast<int>(rows), bwd_block, 0, stream.get()>>>(
      dy.data, x.data, params.gamma.data,
      state.mean.data, state.inv_std.data,
      dx.data, dgamma.data, dbeta.data, rows, D);
  return Status::Ok();
}

}  // namespace ops
}  // namespace gpt
