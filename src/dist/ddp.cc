// Layer 6 — Distributed: DDP implementation.

#include "src/dist/ddp.h"

#include <algorithm>

#include "src/kernels/elementwise.h"

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
  const float inv_world_size =
      1.0f / static_cast<float>(nccl.world_size());
  for (auto& bucket : buckets_) {
    GPT_RETURN_IF_ERROR(
        nccl.all_reduce_sum(bucket.data, bucket.count, comm_stream));
    Tensor1D<const float> summed{bucket.data, {bucket.count}, {1}};
    Tensor1D<float> averaged{bucket.data, {bucket.count}, {1}};
    GPT_RETURN_IF_ERROR(kernels::vec_scale_f32(
        summed, inv_world_size, averaged, comm_stream));
  }
  return Status::Ok();
}

}  // namespace dist
}  // namespace gpt
