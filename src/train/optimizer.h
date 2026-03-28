#pragma once

// Layer 6 — Training runtime: AdamW optimizer.
//
// Operates on flat parameter and gradient buffers.
// Maintains fp32 first and second moment states.

#include <cstdint>

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/tensor/tensor_view.h"

namespace gpt {
namespace train {

struct AdamWConfig {
  float lr = 3e-4f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.01f;
};

class AdamW {
 public:
  AdamW() = default;

  // Initialize optimizer state (m, v buffers) for `param_count` parameters.
  Status init(int64_t param_count, const AdamWConfig& config);

  // Perform one optimizer step:
  //   params -= lr * (corrected_update + weight_decay * params)
  Status step(float* params, const float* grads, int64_t count,
              const CudaStream& stream);

  // Increment step counter (call after each step).
  void advance_step() { ++step_; }

  [[nodiscard]] int64_t current_step() const { return step_; }

  void release();
  ~AdamW();

  AdamW(const AdamW&) = delete;
  AdamW& operator=(const AdamW&) = delete;

 private:
  AdamWConfig config_;
  float* m_ = nullptr;  // first moment
  float* v_ = nullptr;  // second moment
  int64_t count_ = 0;
  int64_t step_ = 0;
};

}  // namespace train
}  // namespace gpt
