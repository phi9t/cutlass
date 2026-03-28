// Layer 7 — Data: batcher implementation.

#include "src/data/batcher.h"

#include <cuda_runtime.h>

namespace gpt {
namespace data {

Status Batcher::init(int64_t batch_size, int64_t seq_len) {
  batch_size_ = batch_size;
  seq_len_ = seq_len;
  size_t bytes = static_cast<size_t>(batch_size * seq_len) * sizeof(int32_t);

  // Pinned host buffers.
  cudaError_t err;
  err = cudaMallocHost(&h_input_ids_, bytes);
  if (err != cudaSuccess)
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  err = cudaMallocHost(&h_targets_, bytes);
  if (err != cudaSuccess)
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));

  // Device buffers.
  err = cudaMalloc(&d_input_ids_, bytes);
  if (err != cudaSuccess)
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  err = cudaMalloc(&d_targets_, bytes);
  if (err != cudaSuccess)
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));

  return Status::Ok();
}

Status Batcher::stage_to_device(const CudaStream& h2d_stream) {
  size_t bytes = static_cast<size_t>(batch_size_ * seq_len_) * sizeof(int32_t);

  cudaError_t err;
  err = cudaMemcpyAsync(d_input_ids_, h_input_ids_, bytes,
                         cudaMemcpyHostToDevice, h2d_stream.get());
  if (err != cudaSuccess)
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));

  err = cudaMemcpyAsync(d_targets_, h_targets_, bytes,
                         cudaMemcpyHostToDevice, h2d_stream.get());
  if (err != cudaSuccess)
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));

  return Status::Ok();
}

void Batcher::release() {
  if (h_input_ids_) { cudaFreeHost(h_input_ids_); h_input_ids_ = nullptr; }
  if (h_targets_) { cudaFreeHost(h_targets_); h_targets_ = nullptr; }
  if (d_input_ids_) { cudaFree(d_input_ids_); d_input_ids_ = nullptr; }
  if (d_targets_) { cudaFree(d_targets_); d_targets_ = nullptr; }
}

Batcher::~Batcher() { release(); }

}  // namespace data
}  // namespace gpt
