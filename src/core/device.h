#pragma once

// Layer 1 — Core runtime: device guard.
//
// RAII helper that sets the current CUDA device for a scope and
// restores the previous device on destruction.

#include <cstdint>

#include "src/core/status.h"

namespace gpt {

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_id) : prev_device_(-1) {
    cudaGetDevice(&prev_device_);
    cudaSetDevice(device_id);
  }

  ~DeviceGuard() {
    if (prev_device_ >= 0) {
      cudaSetDevice(prev_device_);
    }
  }

  DeviceGuard(const DeviceGuard&) = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;

 private:
  int prev_device_;
};

// Return the number of visible CUDA devices.
inline Result<int> device_count() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }
  return count;
}

}  // namespace gpt
