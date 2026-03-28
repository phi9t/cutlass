#pragma once

// Layer 7 — Data: pretokenized binary dataset.
//
// Format: flat binary file of int32 token IDs.
// Each sample is a fixed-length (input, target) window of seq_len+1 tokens.
// Rank-local sharding by offset/stride.

#include <cstdint>
#include <string>
#include <vector>

#include "src/core/status.h"

namespace gpt {
namespace data {

struct DatasetConfig {
  std::string path;           // path to binary token file
  int64_t seq_len = 1024;    // context length
  int64_t batch_size = 8;
  int rank = 0;
  int world_size = 1;
};

class TokenDataset {
 public:
  TokenDataset() = default;

  // Load dataset and compute sample boundaries.
  Status init(const DatasetConfig& config);

  // Return total number of samples available for this rank.
  [[nodiscard]] int64_t num_samples() const { return num_samples_; }

  // Fill input_ids [B, T] and targets [B, T] for the next batch.
  // Returns kInvalidArgument if dataset is exhausted (epoch boundary).
  Status next_batch(int32_t* input_ids, int32_t* targets);

  // Reset cursor to beginning (for next epoch).
  void reset() { cursor_ = 0; }

  // Get current cursor position (for checkpointing).
  [[nodiscard]] int64_t cursor() const { return cursor_; }
  void set_cursor(int64_t c) { cursor_ = c; }

 private:
  DatasetConfig config_;
  std::vector<int32_t> tokens_;  // mmap'd or loaded into host memory
  int64_t num_samples_ = 0;
  int64_t cursor_ = 0;
};

}  // namespace data
}  // namespace gpt
