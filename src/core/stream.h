#pragma once

// Layer 1 — Core runtime: CUDA stream wrapper.

#include <cuda_runtime.h>

#include "src/core/status.h"

namespace gpt {

class CudaStream {
 public:
  CudaStream() : stream_(nullptr), owned_(false) {}

  static Result<CudaStream> Create() {
    cudaStream_t s;
    cudaError_t err = cudaStreamCreate(&s);
    if (err != cudaSuccess) {
      return Status(StatusCode::kCudaError, cudaGetErrorString(err));
    }
    return CudaStream(s, /*owned=*/true);
  }

  // Wrap an externally owned stream (non-owning).
  static CudaStream Wrap(cudaStream_t s) { return CudaStream(s, false); }

  ~CudaStream() {
    if (owned_ && stream_) {
      cudaStreamDestroy(stream_);
    }
  }

  // Move-only.
  CudaStream(CudaStream&& other) noexcept
      : stream_(other.stream_), owned_(other.owned_) {
    other.stream_ = nullptr;
    other.owned_ = false;
  }
  CudaStream& operator=(CudaStream&& other) noexcept {
    if (this != &other) {
      if (owned_ && stream_) cudaStreamDestroy(stream_);
      stream_ = other.stream_;
      owned_ = other.owned_;
      other.stream_ = nullptr;
      other.owned_ = false;
    }
    return *this;
  }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  [[nodiscard]] cudaStream_t get() const { return stream_; }

  Status synchronize() const {
    cudaError_t err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
      return Status(StatusCode::kCudaError, cudaGetErrorString(err));
    }
    return Status::Ok();
  }

 private:
  CudaStream(cudaStream_t s, bool owned) : stream_(s), owned_(owned) {}

  cudaStream_t stream_;
  bool owned_;
};

}  // namespace gpt
