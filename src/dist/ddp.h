#pragma once

// Layer 6 — Distributed: DDP gradient synchronization.
//
// Bucketed gradient all-reduce.
//   1. Partition flat grad buffer into buckets.
//   2. After backward, reduce each bucket via NCCL all-reduce.
//   3. Scale by 1/world_size.

#include <cstdint>
#include <vector>

#include "src/core/status.h"
#include "src/core/stream.h"
#include "src/dist/nccl_context.h"

namespace gpt {
namespace dist {

struct DDPConfig {
  int64_t bucket_size_bytes = 25 * 1024 * 1024;  // 25 MB default
};

// Gradient bucket descriptor.
struct GradBucket {
  float* data;
  int64_t count;  // number of float elements
};

class DDP {
 public:
  DDP() = default;

  // Set up bucket slicing over a flat gradient buffer.
  Status init(float* grad_buffer, int64_t total_count,
              const DDPConfig& config);

  // All-reduce all gradient buckets.
  Status all_reduce_grads(NcclContext& nccl, const CudaStream& comm_stream);

  [[nodiscard]] const std::vector<GradBucket>& buckets() const {
    return buckets_;
  }

 private:
  std::vector<GradBucket> buckets_;
};

}  // namespace dist
}  // namespace gpt
