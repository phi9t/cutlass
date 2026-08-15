// Tier C — Integration: resume from checkpoint produces equivalent results.

#include "gtest/gtest.h"
#include "src/train/trainer.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace gpt;

namespace {

train::TrainerConfig TinyConfig() {
  train::TrainerConfig config;
  config.model_config.vocab_size = 8;
  config.model_config.max_seq_len = 4;
  config.model_config.n_layers = 0;
  config.model_config.n_heads = 2;
  config.model_config.d_model = 8;
  config.model_config.mlp_hidden = 16;
  config.model_config.tie_embeddings = false;
  config.model_config.use_bias = false;
  config.optimizer_config.lr = 0.02f;
  config.optimizer_config.weight_decay = 0.0f;
  config.nccl_config.world_size = 1;
  return config;
}

struct DeviceBatch {
  int32_t* input = nullptr;
  int32_t* targets = nullptr;
  int64_t B = 2;
  int64_t T = 4;

  void allocate() {
    std::vector<int32_t> h_input = {
        0, 1, 2, 3,
        3, 2, 1, 0,
    };
    std::vector<int32_t> h_targets = {
        1, 2, 3, 4,
        2, 1, 0, 7,
    };
    cudaMalloc(&input, B * T * sizeof(int32_t));
    cudaMalloc(&targets, B * T * sizeof(int32_t));
    cudaMemcpy(input, h_input.data(), B * T * sizeof(int32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(targets, h_targets.data(), B * T * sizeof(int32_t),
               cudaMemcpyHostToDevice);
  }

  Tensor2D<const int32_t> input_view() const {
    return Tensor2D<const int32_t>{input, {B, T}, {T, 1}};
  }

  Tensor2D<const int32_t> targets_view() const {
    return Tensor2D<const int32_t>{targets, {B, T}, {T, 1}};
  }

  void release() {
    cudaFree(input);
    cudaFree(targets);
    input = nullptr;
    targets = nullptr;
  }
};

std::vector<float> RunSteps(train::Trainer& trainer, const DeviceBatch& batch,
                            int steps) {
  std::vector<float> losses;
  losses.reserve(static_cast<size_t>(steps));
  for (int i = 0; i < steps; ++i) {
    train::StepMetrics metrics;
    auto status =
        trainer.train_step(batch.input_view(), batch.targets_view(), metrics);
    EXPECT_TRUE(status.ok()) << status.message();
    EXPECT_TRUE(std::isfinite(metrics.loss));
    losses.push_back(metrics.loss);
  }
  return losses;
}

}  // namespace

TEST(ResumeEquivalenceTest, CheckpointResumeMatchesUninterruptedLosses) {
  const std::string dir = "/tmp/gpt_resume_equivalence_test";
  std::filesystem::remove_all(dir);

  DeviceBatch batch;
  batch.allocate();

  train::TrainerConfig config = TinyConfig();
  train::Trainer uninterrupted;
  auto status = uninterrupted.init(config);
  ASSERT_TRUE(status.ok()) << status.message();
  std::vector<float> uninterrupted_prefix = RunSteps(uninterrupted, batch, 2);
  std::vector<float> uninterrupted_suffix = RunSteps(uninterrupted, batch, 3);
  ASSERT_EQ(uninterrupted.current_step(), 5);

  train::Trainer checkpointed;
  status = checkpointed.init(config);
  ASSERT_TRUE(status.ok()) << status.message();
  std::vector<float> checkpointed_prefix = RunSteps(checkpointed, batch, 2);
  ASSERT_EQ(checkpointed.current_step(), 2);
  status = checkpointed.save_checkpoint(
      dir, /*dataset_cursor=*/16, "{\"test\":\"resume_equivalence\"}");
  ASSERT_TRUE(status.ok()) << status.message();

  train::Trainer resumed;
  status = resumed.init(config);
  ASSERT_TRUE(status.ok()) << status.message();
  checkpoint::CheckpointMetadata meta;
  status = resumed.load_checkpoint(dir, meta);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(meta.step, 2);
  EXPECT_EQ(meta.dataset_cursor, 16);
  EXPECT_EQ(resumed.current_step(), 2);
  std::vector<float> resumed_suffix = RunSteps(resumed, batch, 3);
  EXPECT_EQ(resumed.current_step(), 5);

  ASSERT_EQ(checkpointed_prefix.size(), uninterrupted_prefix.size());
  for (size_t i = 0; i < checkpointed_prefix.size(); ++i) {
    EXPECT_FLOAT_EQ(checkpointed_prefix[i], uninterrupted_prefix[i]);
  }
  ASSERT_EQ(resumed_suffix.size(), uninterrupted_suffix.size());
  for (size_t i = 0; i < resumed_suffix.size(); ++i) {
    EXPECT_FLOAT_EQ(resumed_suffix[i], uninterrupted_suffix[i]);
  }

  batch.release();
  std::filesystem::remove_all(dir);
}
