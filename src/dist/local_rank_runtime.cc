// Layer 6 — Distributed: local single-node rank bootstrap implementation.

#include "src/dist/local_rank_runtime.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace gpt {
namespace dist {

namespace {

Result<int> parse_int(char const* text, int fallback, char const* name) {
  if (text == nullptr || text[0] == '\0') {
    return fallback;
  }
  errno = 0;
  char* end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    return Status(StatusCode::kInvalidArgument,
                  std::string("Invalid integer env ") + name);
  }
  return static_cast<int>(value);
}

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return 10 + c - 'a';
  if (c >= 'A' && c <= 'F') return 10 + c - 'A';
  return -1;
}

Result<std::array<unsigned char, 128>> parse_unique_id(char const* text) {
  std::array<unsigned char, 128> id = {};
  if (text == nullptr || text[0] == '\0') {
    return id;
  }
  if (std::strlen(text) != id.size() * 2) {
    return Status(StatusCode::kInvalidArgument,
                  "NCCL unique ID hex must be 256 characters");
  }
  for (size_t i = 0; i < id.size(); ++i) {
    int hi = hex_value(text[2 * i]);
    int lo = hex_value(text[2 * i + 1]);
    if (hi < 0 || lo < 0) {
      return Status(StatusCode::kInvalidArgument,
                    "NCCL unique ID hex contains a non-hex character");
    }
    id[i] = static_cast<unsigned char>((hi << 4) | lo);
  }
  return id;
}

}  // namespace

Result<LocalRankEnv> parse_local_rank_env(char const* rank,
                                          char const* world_size,
                                          char const* local_rank,
                                          char const* unique_id_hex) {
  Result<int> parsed_rank = parse_int(rank, 0, "RANK");
  if (!parsed_rank.ok()) return parsed_rank.status();
  Result<int> parsed_world_size = parse_int(world_size, 1, "WORLD_SIZE");
  if (!parsed_world_size.ok()) return parsed_world_size.status();
  Result<int> parsed_local_rank =
      parse_int(local_rank, parsed_rank.value(), "LOCAL_RANK");
  if (!parsed_local_rank.ok()) return parsed_local_rank.status();
  Result<std::array<unsigned char, 128>> parsed_id =
      parse_unique_id(unique_id_hex);
  if (!parsed_id.ok()) return parsed_id.status();

  LocalRankEnv env;
  env.rank = parsed_rank.value();
  env.world_size = parsed_world_size.value();
  env.local_gpu_id = parsed_local_rank.value();
  env.unique_id = parsed_id.value();
  env.has_unique_id = unique_id_hex != nullptr && unique_id_hex[0] != '\0';
  if (env.world_size <= 0) {
    return Status(StatusCode::kInvalidArgument,
                  "WORLD_SIZE must be positive");
  }
  if (env.rank < 0 || env.rank >= env.world_size) {
    return Status(StatusCode::kInvalidArgument, "RANK out of range");
  }
  if (env.local_gpu_id < 0) {
    return Status(StatusCode::kInvalidArgument,
                  "LOCAL_RANK must be non-negative");
  }
  if (env.world_size > 1 && !env.has_unique_id) {
    return Status(StatusCode::kInvalidArgument,
                  "Multi-rank NCCL requires shared unique ID hex");
  }
  return env;
}

Result<NcclConfig> make_nccl_config(LocalRankEnv const& env) {
  if (env.world_size <= 0) {
    return Status(StatusCode::kInvalidArgument,
                  "make_nccl_config: world_size must be positive");
  }
  if (env.rank < 0 || env.rank >= env.world_size) {
    return Status(StatusCode::kInvalidArgument,
                  "make_nccl_config: rank out of range");
  }
  if (env.world_size > 1 && !env.has_unique_id) {
    return Status(StatusCode::kInvalidArgument,
                  "make_nccl_config: multi-rank requires unique_id");
  }

  NcclConfig config;
  config.world_size = env.world_size;
  config.rank = env.rank;
  config.local_gpu_id = env.local_gpu_id;
  if (env.has_unique_id) {
    std::memcpy(config.unique_id, env.unique_id.data(), env.unique_id.size());
  }
  return config;
}

}  // namespace dist
}  // namespace gpt
