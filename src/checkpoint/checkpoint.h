#pragma once

// Layer 7 — Checkpoint: save/load training state.
//
// Saves:
//   - model parameters
//   - optimizer state (m, v)
//   - training step
//   - dataset cursor
//   - RNG state (future)
//   - config metadata

#include <cstdint>
#include <string>

#include "src/core/status.h"

namespace gpt {
namespace checkpoint {

struct CheckpointMetadata {
  int64_t step = 0;
  int64_t param_count = 0;
  int64_t dataset_cursor = 0;
  // Config hash or serialized config for validation.
  std::string config_json;
};

// Save a checkpoint to disk.
//   dir:     checkpoint directory (will be created if needed)
//   params:  device pointer to flat parameter buffer
//   opt_m:   device pointer to optimizer first moment
//   opt_v:   device pointer to optimizer second moment
//   meta:    checkpoint metadata
Status save_checkpoint(const std::string& dir,
                       const float* params,
                       const float* opt_m,
                       const float* opt_v,
                       int64_t param_count,
                       const CheckpointMetadata& meta);

// Load a checkpoint from disk.
//   dir:     checkpoint directory
//   params:  device pointer (pre-allocated)
//   opt_m:   device pointer (pre-allocated)
//   opt_v:   device pointer (pre-allocated)
//   meta:    filled with loaded metadata
Status load_checkpoint(const std::string& dir,
                       float* params,
                       float* opt_m,
                       float* opt_v,
                       int64_t param_count,
                       CheckpointMetadata& meta);

}  // namespace checkpoint
}  // namespace gpt
