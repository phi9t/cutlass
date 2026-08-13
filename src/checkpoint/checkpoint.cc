// Layer 7 — Checkpoint: implementation.

#include "src/checkpoint/checkpoint.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

namespace gpt {
namespace checkpoint {

namespace fs = std::filesystem;

static Status save_device_buffer(const std::string& path,
                                  const float* d_buf,
                                  int64_t count) {
  size_t bytes = static_cast<size_t>(count) * sizeof(float);
  std::vector<float> host(count);
  cudaError_t err = cudaMemcpy(host.data(), d_buf, bytes,
                                cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }

  std::ofstream f(path, std::ios::binary);
  if (!f) return Status(StatusCode::kInternalError, "Cannot write: " + path);
  f.write(reinterpret_cast<const char*>(host.data()), bytes);
  return Status::Ok();
}

static Status load_device_buffer(const std::string& path,
                                  float* d_buf,
                                  int64_t count) {
  size_t bytes = static_cast<size_t>(count) * sizeof(float);
  std::vector<float> host(count);

  std::ifstream f(path, std::ios::binary);
  if (!f) return Status(StatusCode::kInvalidArgument, "Cannot read: " + path);
  f.read(reinterpret_cast<char*>(host.data()), bytes);

  cudaError_t err = cudaMemcpy(d_buf, host.data(), bytes,
                                cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return Status(StatusCode::kCudaError, cudaGetErrorString(err));
  }
  return Status::Ok();
}

Status save_checkpoint(const std::string& dir,
                       const float* params,
                       const float* opt_m,
                       const float* opt_v,
                       int64_t param_count,
                       const CheckpointMetadata& meta) {
  fs::create_directories(dir);

  GPT_RETURN_IF_ERROR(save_device_buffer(dir + "/params.bin", params, param_count));
  GPT_RETURN_IF_ERROR(save_device_buffer(dir + "/opt_m.bin", opt_m, param_count));
  GPT_RETURN_IF_ERROR(save_device_buffer(dir + "/opt_v.bin", opt_v, param_count));

  // Save metadata.
  std::ofstream mf(dir + "/meta.json");
  if (!mf) return Status(StatusCode::kInternalError, "Cannot write meta.json");
  mf << "{\n"
     << "  \"step\": " << meta.step << ",\n"
     << "  \"param_count\": " << meta.param_count << ",\n"
     << "  \"dataset_cursor\": " << meta.dataset_cursor << ",\n"
     << "  \"config\": " << meta.config_json << "\n"
     << "}\n";

  return Status::Ok();
}

Status load_checkpoint(const std::string& dir,
                       float* params,
                       float* opt_m,
                       float* opt_v,
                       int64_t param_count,
                       CheckpointMetadata& meta) {
  if (!fs::exists(dir + "/params.bin")) {
    return Status(StatusCode::kInvalidArgument,
                  "Checkpoint not found: " + dir);
  }

  GPT_RETURN_IF_ERROR(load_device_buffer(dir + "/params.bin", params, param_count));
  GPT_RETURN_IF_ERROR(load_device_buffer(dir + "/opt_m.bin", opt_m, param_count));
  GPT_RETURN_IF_ERROR(load_device_buffer(dir + "/opt_v.bin", opt_v, param_count));

  // TODO: Parse meta.json and fill CheckpointMetadata.
  // For v1, just read step from a simple format.

  return Status::Ok();
}

}  // namespace checkpoint
}  // namespace gpt
