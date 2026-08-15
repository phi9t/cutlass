// Layer 6 — Distributed: NCCL context implementation.

#include "src/dist/nccl_context.h"

#include <cuda_runtime.h>
#include <nccl.h>
#include <cstring>
#include <string>

namespace gpt {
namespace dist {

namespace {

Status nccl_status(ncclResult_t result, const char* context) {
  if (result == ncclSuccess) return Status::Ok();
  return Status(StatusCode::kNcclError,
                std::string(context) + ": " + ncclGetErrorString(result));
}

Status cuda_status(cudaError_t result, const char* context) {
  if (result == cudaSuccess) return Status::Ok();
  return Status(StatusCode::kCudaError,
                std::string(context) + ": " + cudaGetErrorString(result));
}

}  // namespace

Status create_nccl_unique_id(unsigned char unique_id[128]) {
  if (unique_id == nullptr) {
    return Status(StatusCode::kInvalidArgument,
                  "create_nccl_unique_id: null output");
  }
  ncclUniqueId id;
  GPT_RETURN_IF_ERROR(nccl_status(ncclGetUniqueId(&id), "ncclGetUniqueId"));
  static_assert(sizeof(id.internal) == 128, "Unexpected ncclUniqueId size");
  std::memcpy(unique_id, id.internal, sizeof(id.internal));
  return Status::Ok();
}

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

  if (config.world_size <= 0) {
    return Status(StatusCode::kInvalidArgument,
                  "NcclContext world_size must be positive");
  }
  if (config.rank < 0 || config.rank >= config.world_size) {
    return Status(StatusCode::kInvalidArgument,
                  "NcclContext rank out of range");
  }
  GPT_RETURN_IF_ERROR(cuda_status(cudaSetDevice(config.local_gpu_id),
                                  "cudaSetDevice"));
  ncclUniqueId unique_id;
  if (config.world_size == 1 && !config.has_unique_id()) {
    GPT_RETURN_IF_ERROR(nccl_status(ncclGetUniqueId(&unique_id),
                                    "ncclGetUniqueId"));
  } else {
    if (!config.has_unique_id()) {
      return Status(StatusCode::kInvalidArgument,
                    "NcclContext multi-rank requires shared unique_id");
    }
    static_assert(sizeof(unique_id.internal) == 128,
                  "Unexpected ncclUniqueId size");
    std::memcpy(unique_id.internal, config.unique_id,
                sizeof(unique_id.internal));
  }
  ncclComm_t comm = nullptr;
  GPT_RETURN_IF_ERROR(nccl_status(
      ncclCommInitRank(&comm, config.world_size, unique_id, config.rank),
      "ncclCommInitRank"));

  comm_ = comm;
  initialized_ = true;
  return Status::Ok();
}

Status NcclContext::destroy() {
  if (!initialized_) return Status::Ok();

  ncclComm_t comm = reinterpret_cast<ncclComm_t>(comm_);
  Status status = nccl_status(ncclCommDestroy(comm), "ncclCommDestroy");
  initialized_ = false;
  comm_ = nullptr;
  return status;
}

Status NcclContext::all_reduce_sum(float* buffer, int64_t count,
                                  const CudaStream& stream) {
  if (!initialized_) {
    return Status(StatusCode::kInvalidArgument,
                  "NcclContext not initialized");
  }
  if (count < 0) {
    return Status(StatusCode::kInvalidArgument,
                  "NcclContext all_reduce count must be non-negative");
  }
  ncclComm_t comm = reinterpret_cast<ncclComm_t>(comm_);
  return nccl_status(ncclAllReduce(buffer, buffer, static_cast<size_t>(count),
                                   ncclFloat, ncclSum, comm, stream.get()),
                     "ncclAllReduce");
}

}  // namespace dist
}  // namespace gpt
