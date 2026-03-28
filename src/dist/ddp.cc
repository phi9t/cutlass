// Layer 6 — Distributed: DDP implementation.

#include "src/dist/ddp.h"

namespace gpt {
namespace dist {

Status DDP::init(float* grad_buffer, int64_t total_count,
                 const DDPConfig& config) {
  buckets_.clear();

  int64_t bucket_count = config.bucket_size_bytes / sizeof(float);
  if (bucket_count <= 0) bucket_count = total_count;

  int64_t offset = 0;
  while (offset < total_count) {
    int64_t this_count = std::min(bucket_count, total_count - offset);
    buckets_.push_back({grad_buffer + offset, this_count});
    offset += this_count;
  }

  return Status::Ok();
}

Status DDP::all_reduce_grads(NcclContext& nccl,
                             const CudaStream& comm_stream) {
  for (auto& bucket : buckets_) {
    GPT_RETURN_IF_ERROR(
        nccl.all_reduce_sum(bucket.data, bucket.count, comm_stream));
  }
  // TODO: Scale by 1/world_size after all-reduce.
  return Status::Ok();
}

}  // namespace dist
}  // namespace gpt
