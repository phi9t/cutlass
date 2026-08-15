// Tier B — GPU unit tests for Trainer train-step orchestration.

#include "gtest/gtest.h"
#include "src/train/trainer.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace gpt;
using namespace gpt::train;

TEST(TrainerGPUTest, TrainStepRunsBackwardAndAdvancesStep) {
  TrainerConfig config;
  config.model_config.vocab_size = 8;
  config.model_config.max_seq_len = 2;
  config.model_config.n_layers = 0;
  config.model_config.n_heads = 1;
  config.model_config.d_model = 4;
  config.model_config.mlp_hidden = 8;
  config.model_config.tie_embeddings = false;
  config.model_config.use_bias = false;
  config.optimizer_config.lr = 1e-3f;
  config.optimizer_config.weight_decay = 0.0f;
  config.nccl_config.world_size = 1;

  Trainer trainer;
  auto status = trainer.init(config);
  ASSERT_TRUE(status.ok()) << status.message();

  const int64_t B = 1;
  const int64_t T = 2;
  std::vector<int32_t> h_input = {1, 2};
  std::vector<int32_t> h_targets = {2, 3};

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
  StepMetrics metrics;
  status = trainer.train_step(input, targets, metrics);
  ASSERT_TRUE(status.ok()) << status.message();

  EXPECT_EQ(trainer.current_step(), 1);
  EXPECT_TRUE(std::isfinite(metrics.loss));
  EXPECT_GT(metrics.loss, 0.0f);

  StepMetrics second_metrics;
  status = trainer.train_step(input, targets, second_metrics);
  ASSERT_TRUE(status.ok()) << status.message();
  EXPECT_EQ(trainer.current_step(), 2);
  EXPECT_TRUE(std::isfinite(second_metrics.loss));
  EXPECT_GT(second_metrics.loss, 0.0f);

  cudaFree(d_input);
  cudaFree(d_targets);
}
