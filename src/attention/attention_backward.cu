// Layer 4 — Attention subsystem: backward-specific CUDA kernels.
//
#include "src/attention/attention_backward_internal.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace gpt {
namespace attention {

namespace {

static constexpr int kBlockSize = 256;

__global__ void split_merged_heads_grad_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t Dh) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * H * T * Dh;
  if (idx >= total) return;

  int64_t d = idx % Dh;
  int64_t t = (idx / Dh) % T;
  int64_t h = (idx / (Dh * T)) % H;
  int64_t b = idx / (Dh * T * H);
  int64_t D = H * Dh;
  output[idx] = input[(b * T + t) * D + h * Dh + d];
}

__global__ void scale_scores_grad_kernel(
    float* __restrict__ scores,
    float scale,
    int64_t total) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) scores[idx] *= scale;
}

__global__ void merge_qkv_grads_kernel(
    const float* __restrict__ dQ,
    const float* __restrict__ dK,
    const float* __restrict__ dV,
    float* __restrict__ dQKV,
    int64_t B,
    int64_t H,
    int64_t T,
    int64_t Dh) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * H * T * Dh;
  if (idx >= total) return;

  int64_t d = idx % Dh;
  int64_t t = (idx / Dh) % T;
  int64_t h = (idx / (Dh * T)) % H;
  int64_t b = idx / (Dh * T * H);
  int64_t D = H * Dh;
  int64_t out_base = (b * T + t) * (3 * D);
  int64_t feature = h * Dh + d;

  dQKV[out_base + feature] = dQ[idx];
  dQKV[out_base + D + feature] = dK[idx];
  dQKV[out_base + 2 * D + feature] = dV[idx];
}

}  // namespace

Status split_merged_heads_grad_f32(Tensor3D<const float> input,
                                   Tensor4D<float> output,
                                   int64_t n_heads,
                                   const CudaStream& stream) {
  int64_t B = input.shape[0];
  int64_t T = input.shape[1];
  int64_t D = input.shape[2];
  if (D % n_heads != 0) {
    return Status(StatusCode::kInvalidArgument,
                  "split_merged_heads_grad_f32: D not divisible by n_heads");
  }
  int64_t Dh = D / n_heads;
  if (output.shape[0] != B || output.shape[1] != n_heads ||
      output.shape[2] != T || output.shape[3] != Dh) {
    return Status(StatusCode::kInvalidArgument,
                  "split_merged_heads_grad_f32: output shape mismatch");
  }
  int64_t total = B * n_heads * T * Dh;
  int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
  split_merged_heads_grad_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      input.data, output.data, B, T, n_heads, Dh);
  GPT_CHECK_CUDA(cudaGetLastError());
  return Status::Ok();
}

Status scale_scores_grad_f32(Tensor4D<float> scores,
                             float scale,
                             const CudaStream& stream) {
  int64_t total = scores.numel();
  int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
  scale_scores_grad_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      scores.data, scale, total);
  GPT_CHECK_CUDA(cudaGetLastError());
  return Status::Ok();
}

Status merge_qkv_grads_f32(Tensor4D<const float> dQ,
                           Tensor4D<const float> dK,
                           Tensor4D<const float> dV,
                           Tensor3D<float> dQKV,
                           const CudaStream& stream) {
  if (dQ.shape != dK.shape || dQ.shape != dV.shape) {
    return Status(StatusCode::kInvalidArgument,
                  "merge_qkv_grads_f32: Q/K/V shape mismatch");
  }
  int64_t B = dQ.shape[0];
  int64_t H = dQ.shape[1];
  int64_t T = dQ.shape[2];
  int64_t Dh = dQ.shape[3];
  int64_t D = H * Dh;
  if (dQKV.shape[0] != B || dQKV.shape[1] != T ||
      dQKV.shape[2] != 3 * D) {
    return Status(StatusCode::kInvalidArgument,
                  "merge_qkv_grads_f32: output shape mismatch");
  }
  int64_t total = B * H * T * Dh;
  int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
  merge_qkv_grads_kernel<<<grid, kBlockSize, 0, stream.get()>>>(
      dQ.data, dK.data, dV.data, dQKV.data, B, H, T, Dh);
  GPT_CHECK_CUDA(cudaGetLastError());
  return Status::Ok();
}

}  // namespace attention
}  // namespace gpt
