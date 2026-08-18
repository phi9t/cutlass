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

struct TrainingSnapshot {
  float* params = nullptr;
  float* opt_m = nullptr;
  float* opt_v = nullptr;
  int64_t param_count = 0;
  CheckpointMetadata metadata;
};

Status save_training_snapshot(const std::string& dir,
                              const TrainingSnapshot& snapshot);

Status load_training_snapshot(const std::string& dir,
                              TrainingSnapshot& snapshot);

}  // namespace checkpoint
}  // namespace gpt
