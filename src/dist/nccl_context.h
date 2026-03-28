#pragma once

// Layer 6 — Distributed: NCCL communicator context.
//
// Manages NCCL communicator lifecycle for a single-node multi-GPU setup.
// One NcclContext per process (one process per GPU).

#include <cstdint>

#include "src/core/status.h"
#include "src/core/stream.h"

namespace gpt {
namespace dist {

struct NcclConfig {
  int world_size = 1;
  int rank = 0;
  int local_gpu_id = 0;  // CUDA device ordinal for this rank.
};

class NcclContext {
 public:
  NcclContext() = default;
  ~NcclContext();

  // Initialize the communicator.
  // For single-node, uses ncclCommInitRank with a shared unique ID.
  Status init(const NcclConfig& config);

  // Tear down the communicator.
  Status destroy();

  // All-reduce in-place: sum across all ranks.
  //   buffer: device pointer, count elements of float.
  Status all_reduce_sum(float* buffer, int64_t count,
                        const CudaStream& stream);

  [[nodiscard]] int rank() const { return config_.rank; }
  [[nodiscard]] int world_size() const { return config_.world_size; }
  [[nodiscard]] bool is_initialized() const { return initialized_; }

  NcclContext(const NcclContext&) = delete;
  NcclContext& operator=(const NcclContext&) = delete;

 private:
  NcclConfig config_;
  void* comm_ = nullptr;  // ncclComm_t, stored as void* to avoid header leak.
  bool initialized_ = false;
};

}  // namespace dist
}  // namespace gpt
