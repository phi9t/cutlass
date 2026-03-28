#pragma once

// Layer 1 — Core runtime: CUDA event wrapper.

#include <cuda_runtime.h>

#include "src/core/status.h"
#include "src/core/stream.h"

namespace gpt {

class CudaEvent {
 public:
  CudaEvent() : event_(nullptr) {}

  static Result<CudaEvent> Create(unsigned flags = cudaEventDefault) {
    cudaEvent_t e;
    cudaError_t err = cudaEventCreateWithFlags(&e, flags);
    if (err != cudaSuccess) {
      return Status(StatusCode::kCudaError, cudaGetErrorString(err));
    }
    CudaEvent ev;
    ev.event_ = e;
    return ev;
  }

  ~CudaEvent() {
    if (event_) cudaEventDestroy(event_);
  }

  CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }
  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      if (event_) cudaEventDestroy(event_);
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  [[nodiscard]] cudaEvent_t get() const { return event_; }

  Status record(const CudaStream& stream) {
    GPT_CHECK_CUDA(cudaEventRecord(event_, stream.get()));
    return Status::Ok();
  }

  Status synchronize() {
    GPT_CHECK_CUDA(cudaEventSynchronize(event_));
    return Status::Ok();
  }

  // Returns elapsed time in milliseconds between two recorded events.
  static Result<float> elapsed(const CudaEvent& start, const CudaEvent& end) {
    float ms = 0.0f;
    cudaError_t err = cudaEventElapsedTime(&ms, start.event_, end.event_);
    if (err != cudaSuccess) {
      return Status(StatusCode::kCudaError, cudaGetErrorString(err));
    }
    return ms;
  }

 private:
  cudaEvent_t event_;
};

}  // namespace gpt
