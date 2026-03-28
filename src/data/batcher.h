#pragma once

// Layer 7 — Data: batch staging with pinned host buffers.
//
// Stages (input_ids, targets) batches in pinned memory for async H2D copy.

#include <cstdint>

#include "src/core/status.h"
#include "src/core/stream.h"

namespace gpt {
namespace data {

class Batcher {
 public:
  Batcher() = default;

  // Allocate pinned host buffers and device buffers for one batch.
  Status init(int64_t batch_size, int64_t seq_len);

  // Copy from host staging buffers to device.
  Status stage_to_device(const CudaStream& h2d_stream);

  // Access host staging buffers (fill these from TokenDataset).
  [[nodiscard]] int32_t* host_input_ids() { return h_input_ids_; }
  [[nodiscard]] int32_t* host_targets() { return h_targets_; }

  // Access device buffers (pass these to the model).
  [[nodiscard]] int32_t* device_input_ids() { return d_input_ids_; }
  [[nodiscard]] int32_t* device_targets() { return d_targets_; }

  void release();
  ~Batcher();

  Batcher(const Batcher&) = delete;
  Batcher& operator=(const Batcher&) = delete;

 private:
  int64_t batch_size_ = 0;
  int64_t seq_len_ = 0;
  int32_t* h_input_ids_ = nullptr;   // pinned host
  int32_t* h_targets_ = nullptr;     // pinned host
  int32_t* d_input_ids_ = nullptr;   // device
  int32_t* d_targets_ = nullptr;     // device
};

}  // namespace data
}  // namespace gpt
