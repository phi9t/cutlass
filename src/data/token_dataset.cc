// Layer 7 — Data: token dataset implementation.

#include "src/data/token_dataset.h"

#include <cstdio>
#include <cstring>

namespace gpt {
namespace data {

Status TokenDataset::init(const DatasetConfig& config) {
  config_ = config;

  // Load binary token file.
  FILE* f = std::fopen(config.path.c_str(), "rb");
  if (!f) {
    return Status(StatusCode::kInvalidArgument,
                  "Cannot open token file: " + config.path);
  }

  std::fseek(f, 0, SEEK_END);
  long file_size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);

  int64_t total_tokens = file_size / sizeof(int32_t);
  tokens_.resize(total_tokens);
  size_t read = std::fread(tokens_.data(), sizeof(int32_t), total_tokens, f);
  std::fclose(f);

  if (static_cast<int64_t>(read) != total_tokens) {
    return Status(StatusCode::kInternalError, "Incomplete read of token file");
  }

  // Each sample needs seq_len+1 tokens (input + one shifted target).
  int64_t window = config.seq_len + 1;
  int64_t total_samples = (total_tokens - 1) / config.seq_len;

  // Rank-local sharding by stride.
  num_samples_ = total_samples / config.world_size;
  cursor_ = 0;

  return Status::Ok();
}

Status TokenDataset::next_batch(int32_t* input_ids, int32_t* targets) {
  int64_t B = config_.batch_size;
  int64_t T = config_.seq_len;

  if (cursor_ + B > num_samples_) {
    return Status(StatusCode::kInvalidArgument,
                  "Dataset exhausted for this epoch");
  }

  for (int64_t b = 0; b < B; ++b) {
    // Global sample index for this rank.
    int64_t sample_idx = (cursor_ + b) * config_.world_size + config_.rank;
    int64_t token_offset = sample_idx * T;

    if (token_offset + T + 1 > static_cast<int64_t>(tokens_.size())) {
      return Status(StatusCode::kInvalidArgument,
                    "Token offset out of bounds");
    }

    std::memcpy(input_ids + b * T, tokens_.data() + token_offset,
                T * sizeof(int32_t));
    std::memcpy(targets + b * T, tokens_.data() + token_offset + 1,
                T * sizeof(int32_t));
  }

  cursor_ += B;
  return Status::Ok();
}

}  // namespace data
}  // namespace gpt
