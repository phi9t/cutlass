// Layer 6 — Training runtime: AdamW CUDA implementation.

#include "src/train/optimizer.h"

#include <cuda_runtime.h>
#include <cmath>

namespace gpt {
namespace train {

namespace {

__global__ void adamw_kernel(float* __restrict__ params,
                             const float* __restrict__ grads,
                             float* __restrict__ m,
                             float* __restrict__ v,
                             int64_t count,
                             float lr, float beta1, float beta2,
                             float eps, float weight_decay,
                             float bc1, float bc2) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;

  float g = grads[i];
  float mi = beta1 * m[i] + (1.0f - beta1) * g;
  float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
  m[i] = mi;
  v[i] = vi;

  float m_hat = mi / bc1;
  float v_hat = vi / bc2;

  params[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * params[i]);
}

}  // namespace

Status AdamW::init(int64_t param_count, const AdamWConfig& config) {
  config_ = config;
  count_ = param_count;
  step_ = 0;

  size_t bytes = static_cast<size_t>(param_count) * sizeof(float);
  cudaError_t err;

  err = cudaMalloc(&m_, bytes);
  if (err != cudaSuccess) {
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }
  err = cudaMemset(m_, 0, bytes);
  if (err != cudaSuccess) {
    cudaFree(m_); m_ = nullptr;
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }

  err = cudaMalloc(&v_, bytes);
  if (err != cudaSuccess) {
    cudaFree(m_); m_ = nullptr;
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }
  err = cudaMemset(v_, 0, bytes);
  if (err != cudaSuccess) {
    cudaFree(m_); m_ = nullptr;
    cudaFree(v_); v_ = nullptr;
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }

  return Status::Ok();
}

Status AdamW::step(float* params, const float* grads, int64_t count,
                   const CudaStream& stream) {
  if (!m_ || !v_) {
    return Status(StatusCode::kInvalidArgument,
                  "AdamW not initialized");
  }

  float bc1 = 1.0f - std::pow(config_.beta1, static_cast<float>(step_ + 1));
  float bc2 = 1.0f - std::pow(config_.beta2, static_cast<float>(step_ + 1));

  int block = 256;
  int grid = static_cast<int>((count + block - 1) / block);
  adamw_kernel<<<grid, block, 0, stream.get()>>>(
      params, grads, m_, v_, count,
      config_.lr, config_.beta1, config_.beta2,
      config_.eps, config_.weight_decay,
      bc1, bc2);

  return Status::Ok();
}

void AdamW::release() {
  if (m_) { cudaFree(m_); m_ = nullptr; }
  if (v_) { cudaFree(v_); v_ = nullptr; }
  count_ = 0;
}

AdamW::~AdamW() { release(); }

}  // namespace train
}  // namespace gpt
