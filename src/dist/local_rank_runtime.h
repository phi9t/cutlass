#pragma once

// Layer 6 — Distributed: local single-node rank bootstrap.

#include <array>

#include "src/core/status.h"
#include "src/dist/nccl_context.h"

namespace gpt {
namespace dist {

struct LocalRankEnv {
  int rank = 0;
  int world_size = 1;
  int local_gpu_id = 0;
  std::array<unsigned char, 128> unique_id = {};
  bool has_unique_id = false;
};

Result<LocalRankEnv> parse_local_rank_env(char const* rank,
                                          char const* world_size,
                                          char const* local_rank,
                                          char const* unique_id_hex);

Result<NcclConfig> make_nccl_config(LocalRankEnv const& env);

}  // namespace dist
}  // namespace gpt
