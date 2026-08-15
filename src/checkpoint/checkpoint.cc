// Layer 7 — Checkpoint: implementation.

#include "src/checkpoint/checkpoint.h"

#include <cuda_runtime.h>
#include <charconv>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <vector>

namespace gpt {
namespace checkpoint {

namespace fs = std::filesystem;

static std::string quoted_key(const char* key) {
  return "\"" + std::string(key) + "\"";
}

static void skip_json_space(std::string_view text, size_t* pos) {
  while (*pos < text.size() &&
         std::isspace(static_cast<unsigned char>(text[*pos])) != 0) {
    ++(*pos);
  }
}

static Result<int64_t> parse_int_field(std::string_view text, const char* key) {
  const size_t key_pos = text.find(quoted_key(key));
  if (key_pos == std::string_view::npos) {
    return Status(StatusCode::kInvalidArgument,
                  "Missing checkpoint metadata field: " + std::string(key));
  }

  size_t pos = key_pos + quoted_key(key).size();
  skip_json_space(text, &pos);
  if (pos >= text.size() || text[pos] != ':') {
    return Status(StatusCode::kInvalidArgument,
                  "Invalid checkpoint metadata field: " + std::string(key));
  }
  ++pos;
  skip_json_space(text, &pos);

  int64_t value = 0;
  const char* begin = text.data() + pos;
  const char* end = text.data() + text.size();
  auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc() || result.ptr == begin) {
    return Status(StatusCode::kInvalidArgument,
                  "Invalid integer checkpoint metadata field: " + std::string(key));
  }
  return value;
}

static Result<std::string> parse_config_field(std::string_view text) {
  const size_t key_pos = text.find(quoted_key("config"));
  if (key_pos == std::string_view::npos) {
    return Status(StatusCode::kInvalidArgument,
                  "Missing checkpoint metadata field: config");
  }

  size_t pos = key_pos + quoted_key("config").size();
  skip_json_space(text, &pos);
  if (pos >= text.size() || text[pos] != ':') {
    return Status(StatusCode::kInvalidArgument,
                  "Invalid checkpoint metadata field: config");
  }
  ++pos;
  skip_json_space(text, &pos);
  const size_t value_begin = pos;
  if (value_begin >= text.size()) {
    return Status(StatusCode::kInvalidArgument,
                  "Invalid checkpoint metadata field: config");
  }

  char opener = text[pos];
  if (opener == '{' || opener == '[') {
    const char closer = opener == '{' ? '}' : ']';
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (; pos < text.size(); ++pos) {
      const char ch = text[pos];
      if (in_string) {
        if (escaped) {
          escaped = false;
        } else if (ch == '\\') {
          escaped = true;
        } else if (ch == '"') {
          in_string = false;
        }
        continue;
      }

      if (ch == '"') {
        in_string = true;
      } else if (ch == opener) {
        ++depth;
      } else if (ch == closer) {
        --depth;
        if (depth == 0) {
          return std::string(text.substr(value_begin, pos - value_begin + 1));
        }
      }
    }
    return Status(StatusCode::kInvalidArgument,
                  "Invalid JSON checkpoint metadata field: config");
  }

  if (opener == '"') {
    bool escaped = false;
    for (++pos; pos < text.size(); ++pos) {
      const char ch = text[pos];
      if (escaped) {
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '"') {
        return std::string(text.substr(value_begin, pos - value_begin + 1));
      }
    }
    return Status(StatusCode::kInvalidArgument,
                  "Invalid string checkpoint metadata field: config");
  }

  while (pos < text.size() && text[pos] != ',' && text[pos] != '}' &&
         std::isspace(static_cast<unsigned char>(text[pos])) == 0) {
    ++pos;
  }
  if (pos == value_begin) {
    return Status(StatusCode::kInvalidArgument,
                  "Invalid checkpoint metadata field: config");
  }
  return std::string(text.substr(value_begin, pos - value_begin));
}

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

  std::ifstream mf(dir + "/meta.json");
  if (!mf) {
    return Status(StatusCode::kInvalidArgument,
                  "Cannot read checkpoint metadata: " + dir + "/meta.json");
  }
  const std::string meta_json((std::istreambuf_iterator<char>(mf)),
                              std::istreambuf_iterator<char>());

  Result<int64_t> step = parse_int_field(meta_json, "step");
  if (!step.ok()) return step.status();
  Result<int64_t> saved_param_count = parse_int_field(meta_json, "param_count");
  if (!saved_param_count.ok()) return saved_param_count.status();
  Result<int64_t> dataset_cursor = parse_int_field(meta_json, "dataset_cursor");
  if (!dataset_cursor.ok()) return dataset_cursor.status();
  Result<std::string> config = parse_config_field(meta_json);
  if (!config.ok()) return config.status();

  meta.step = step.value();
  meta.param_count = saved_param_count.value();
  meta.dataset_cursor = dataset_cursor.value();
  meta.config_json = config.value();

  return Status::Ok();
}

}  // namespace checkpoint
}  // namespace gpt
