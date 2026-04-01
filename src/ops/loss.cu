// Layer 3 — Composite neural ops: cross-entropy CUDA implementation.

#include "src/ops/loss.h"

#include <cfloat>
#include <climits>
#include <cuda_runtime.h>

namespace gpt {
namespace ops {

namespace {

// Per-row log-sum-exp and target logit gather, then accumulate into loss.
__global__ void cross_entropy_forward_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ targets,
    float* __restrict__ loss,
    int64_t N, int64_t V) {
  // One thread per sample (v1 reference).
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const float* row = logits + i * V;
  int32_t target = targets[i];

  // Row max for numerical stability.
  float max_val = -FLT_MAX;
  for (int64_t j = 0; j < V; ++j) {
    max_val = fmaxf(max_val, row[j]);
  }

  // log-sum-exp.
  float sum_exp = 0.0f;
  for (int64_t j = 0; j < V; ++j) {
    sum_exp += expf(row[j] - max_val);
  }

  float lse = max_val + logf(sum_exp);
  float sample_loss = lse - row[target];

  // Accumulate mean loss via atomicAdd.
  atomicAdd(loss, sample_loss / static_cast<float>(N));
}

// Backward: dlogits[i][j] = (softmax[j] - 1{j==target}) / N
__global__ void cross_entropy_backward_kernel(
    const float* __restrict__ logits,
    const int32_t* __restrict__ targets,
    float* __restrict__ d_logits,
    int64_t N, int64_t V) {
  int64_t i = blockIdx.x;
  if (i >= N) return;

  const float* row = logits + i * V;
  float* d_row = d_logits + i * V;
  int32_t target = targets[i];

  // Row max.
  float max_val = -FLT_MAX;
  for (int64_t j = 0; j < V; ++j) {
    max_val = fmaxf(max_val, row[j]);
  }

  // Softmax + gradient.
  float sum_exp = 0.0f;
  for (int64_t j = 0; j < V; ++j) {
    sum_exp += expf(row[j] - max_val);
  }

  float inv_N = 1.0f / static_cast<float>(N);
  for (int64_t j = 0; j < V; ++j) {
    float prob = expf(row[j] - max_val) / sum_exp;
    float indicator = (j == target) ? 1.0f : 0.0f;
    d_row[j] = (prob - indicator) * inv_N;
  }
}

}  // namespace

Status cross_entropy_forward(Tensor2D<const float> logits,
                             Tensor1D<const int32_t> targets,
                             float* d_loss,
                             const CudaStream& stream) {
  int64_t N = logits.shape[0];
  int64_t V = logits.shape[1];
  if (targets.shape[0] != N) {
    return Status(StatusCode::kInvalidArgument,
                  "cross_entropy_forward: targets length mismatch");
  }

  // Zero the loss accumulator.
  cudaMemsetAsync(d_loss, 0, sizeof(float), stream.get());

  int block = 256;
  int grid = static_cast<int>((N + block - 1) / block);
  cross_entropy_forward_kernel<<<grid, block, 0, stream.get()>>>(
      logits.data, targets.data, d_loss, N, V);
  return Status::Ok();
}

Status cross_entropy_backward(Tensor2D<const float> logits,
                              Tensor1D<const int32_t> targets,
                              Tensor2D<float> d_logits,
                              const CudaStream& stream) {
  int64_t N = logits.shape[0];
  int64_t V = logits.shape[1];

  if (N > INT_MAX) {
    return Status(StatusCode::kInvalidArgument,
                  "cross_entropy_backward: N exceeds CUDA grid limit");
  }
  // One block per sample, single thread (v1 reference).
  cross_entropy_backward_kernel<<<static_cast<int>(N), 1, 0, stream.get()>>>(
      logits.data, targets.data, d_logits.data, N, V);
  return Status::Ok();
}

}  // namespace ops
}  // namespace gpt
