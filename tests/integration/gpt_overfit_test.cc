// Tier C — Integration: GPT overfit-one-batch test.

#include "gtest/gtest.h"
#include "src/train/trainer.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

using namespace gpt;

TEST(GPTOverfitTest, FixedBatchLossDecreases) {
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

  train::Trainer trainer;
  auto status = trainer.init(config);
  ASSERT_TRUE(status.ok()) << status.message();

  const int64_t B = 2;
  const int64_t T = 4;
  std::vector<int32_t> h_input = {
      0, 1, 2, 3,
      3, 2, 1, 0,
  };
  std::vector<int32_t> h_targets = {
      1, 2, 3, 4,
      2, 1, 0, 7,
  };

  int32_t* d_input = nullptr;
  int32_t* d_targets = nullptr;
  cudaMalloc(&d_input, B * T * sizeof(int32_t));
  cudaMalloc(&d_targets, B * T * sizeof(int32_t));
  cudaMemcpy(d_input, h_input.data(), B * T * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, h_targets.data(), B * T * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  Tensor2D<const int32_t> input{d_input, {B, T}, {T, 1}};
  Tensor2D<const int32_t> targets{d_targets, {B, T}, {T, 1}};

  train::StepMetrics metrics;
  status = trainer.train_step(input, targets, metrics);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_TRUE(std::isfinite(metrics.loss));
  const float initial_loss = metrics.loss;

  float min_loss = initial_loss;
  for (int step = 0; step < 39; ++step) {
    status = trainer.train_step(input, targets, metrics);
    ASSERT_TRUE(status.ok()) << status.message();
    ASSERT_TRUE(std::isfinite(metrics.loss));
    min_loss = std::min(min_loss, metrics.loss);
  }

  EXPECT_EQ(trainer.current_step(), 40);
  EXPECT_LT(metrics.loss, initial_loss);
  EXPECT_LT(min_loss, initial_loss);

  cudaFree(d_input);
  cudaFree(d_targets);
}
