// Layer 6 — Distributed: NCCL context implementation.

#include "src/dist/nccl_context.h"

// TODO(m7): Include <nccl.h> and implement.
// #include <nccl.h>

namespace gpt {
namespace dist {

NcclContext::~NcclContext() {
  if (initialized_) {
    destroy();
  }
}

Status NcclContext::init(const NcclConfig& config) {
  if (initialized_) {
    return Status(StatusCode::kInvalidArgument,
                  "NcclContext already initialized");
  }
  config_ = config;

  // TODO: ncclGetUniqueId, broadcast via shared memory or file,
  //       ncclCommInitRank.
  // For single-node, all ranks can share via a file or MPI.

  initialized_ = true;
  return Status(StatusCode::kNotImplemented,
                "NCCL init not yet wired");
}

Status NcclContext::destroy() {
  if (!initialized_) return Status::Ok();

  // TODO: ncclCommDestroy(comm_);
  initialized_ = false;
  comm_ = nullptr;
  return Status::Ok();
}

Status NcclContext::all_reduce_sum(float* buffer, int64_t count,
                                  const CudaStream& stream) {
  if (!initialized_) {
    return Status(StatusCode::kInvalidArgument,
                  "NcclContext not initialized");
  }
  // TODO: ncclAllReduce(buffer, buffer, count, ncclFloat, ncclSum,
  //                     comm_, stream.get());
  return Status(StatusCode::kNotImplemented,
                "NCCL all_reduce not yet wired");
}

}  // namespace dist
}  // namespace gpt
