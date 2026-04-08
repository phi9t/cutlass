// Tier C — Integration: GPT overfit-one-batch test.
//
// Initialize a tiny GPT model, run N training steps on a fixed batch,
// and verify that the loss decreases reliably.

#include "gtest/gtest.h"

#include "src/core/stream.h"
#include "src/model/gpt_config.h"
#include "src/model/gpt_model.h"
#include "src/model/gpt_params.h"
#include "src/ops/loss.h"
#include "src/train/optimizer.h"
#include "tests/integration/gpt_test_helpers.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace gpt;
using namespace gpt::model;

class GPTOverfitTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto s = CudaStream::Create();
    ASSERT_TRUE(s.ok());
    stream_ = s.take();
  }
  CudaStream stream_;
};

TEST_F(GPTOverfitTest, LossDecreases) {
  // Tiny GPT: 1 layer, D=32, V=100, T=8, H=4
  GPTConfig cfg;
  cfg.vocab_size   = 100;
  cfg.max_seq_len  = 16;
  cfg.n_layers     = 1;
  cfg.n_heads      = 4;
  cfg.d_model      = 32;
  cfg.mlp_hidden   = 128;
  cfg.tie_embeddings = false;
  cfg.use_bias     = true;
  cfg.layernorm_eps = 1e-5f;

  const int64_t B = 2, T = 8;

  // Allocate model.
  GPTParams params;
  float* param_buf = test::allocate_params(cfg, params);
  ASSERT_NE(param_buf, nullptr);

  // Initialize parameters with small random values on host, then copy.
  int64_t param_count = cfg.approx_param_count();
  std::vector<float> h_params(param_count);
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  for (auto& v : h_params) v = dist(rng);

  // Set LN gammas to 1.0 and betas to 0.0 for stable init.
  // We'd need to know the exact offsets... for simplicity, just use the
  // random init — the model should still overfit with small weights.
  cudaMemcpy(param_buf, h_params.data(), param_count * sizeof(float),
             cudaMemcpyHostToDevice);

  // Allocate gradients.
  GPTGrads grads;
  float* grad_buf = test::allocate_grads(cfg, grads);

  // Allocate forward state.
  GPTForwardState fwd_state;
  float* fwd_buf = test::allocate_forward_state(cfg, B, T, fwd_state);

  // Create a fixed batch of input/target token IDs.
  std::vector<int32_t> h_ids(B * T), h_targets(B * T);
  std::uniform_int_distribution<int32_t> token_dist(0, cfg.vocab_size - 1);
  for (auto& id : h_ids) id = token_dist(rng);
  for (auto& id : h_targets) id = token_dist(rng);

  int32_t *d_ids, *d_targets;
  cudaMalloc(&d_ids, B * T * sizeof(int32_t));
  cudaMalloc(&d_targets, B * T * sizeof(int32_t));
  cudaMemcpy(d_ids, h_ids.data(), B * T * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, h_targets.data(), B * T * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  Tensor2D<const int32_t> input_ids{d_ids, {B, T}, {T, 1}};
  Tensor1D<const int32_t> targets_flat{d_targets, {B * T}, {1}};

  // Allocate device scalar for loss.
  float* d_loss = nullptr;
  cudaMalloc(&d_loss, sizeof(float));

  // Allocate d_logits buffer for backward.
  float* d_logits_buf = nullptr;
  cudaMalloc(&d_logits_buf, B * T * cfg.vocab_size * sizeof(float));

  // Optimizer.
  train::AdamWConfig opt_cfg;
  opt_cfg.lr = 1e-3f;
  opt_cfg.weight_decay = 0.0f;  // no decay for overfitting test
  train::AdamW optimizer;
  ASSERT_TRUE(optimizer.init(param_count, opt_cfg).ok());

  // Training loop.
  const int N = 50;
  std::vector<float> losses;
  losses.reserve(N);

  for (int step = 0; step < N; ++step) {
    // Zero grads.
    cudaMemset(grad_buf, 0, param_count * sizeof(float));

    // Forward.
    auto fwd_status = gpt_forward(input_ids, cfg, params, fwd_state, stream_);
    ASSERT_TRUE(fwd_status.ok()) << fwd_status.message();

    // Loss.
    Tensor2D<const float> logits_view{fwd_state.logits.data,
                                       fwd_state.logits.shape,
                                       fwd_state.logits.stride};
    auto loss_status = ops::cross_entropy_forward(
        logits_view, targets_flat, d_loss, stream_);
    ASSERT_TRUE(loss_status.ok()) << loss_status.message();

    stream_.synchronize();
    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    losses.push_back(h_loss);

    // d_logits from cross_entropy_backward.
    Tensor2D<const float> logits_for_bwd{fwd_state.logits.data,
                                          fwd_state.logits.shape,
                                          fwd_state.logits.stride};
    Tensor2D<float> d_logits_view{d_logits_buf,
                                   {B * T, cfg.vocab_size},
                                   {cfg.vocab_size, 1}};
    auto ce_bwd = ops::cross_entropy_backward(
        logits_for_bwd, targets_flat, d_logits_view, stream_);
    ASSERT_TRUE(ce_bwd.ok()) << ce_bwd.message();

    // Model backward.
    Tensor2D<const float> d_logits_const{d_logits_buf,
                                          {B * T, cfg.vocab_size},
                                          {cfg.vocab_size, 1}};
    auto bwd_status = gpt_backward(
        d_logits_const, input_ids, cfg, params, fwd_state, grads, stream_);
    ASSERT_TRUE(bwd_status.ok()) << bwd_status.message();

    // Optimizer step.
    auto opt_status = optimizer.step(param_buf, grad_buf, param_count, stream_);
    ASSERT_TRUE(opt_status.ok()) << opt_status.message();
    optimizer.advance_step();

    stream_.synchronize();
  }

  // Verify: loss should decrease from first to last.
  ASSERT_GT(losses.size(), 10u);
  float first_loss = losses[0];
  float last_loss = losses.back();

  // The loss should decrease significantly when overfitting a tiny model on
  // a fixed batch.
  EXPECT_LT(last_loss, first_loss)
      << "Loss did not decrease: first=" << first_loss
      << " last=" << last_loss;

  // More stringently: the final loss should be at most 80% of the initial.
  EXPECT_LT(last_loss, 0.8f * first_loss)
      << "Loss did not decrease enough: first=" << first_loss
      << " last=" << last_loss;

  // Cleanup.
  optimizer.release();
  cudaFree(d_logits_buf);
  cudaFree(d_loss);
  cudaFree(d_targets);
  cudaFree(d_ids);
  cudaFree(fwd_buf);
  cudaFree(grad_buf);
  cudaFree(param_buf);
}

// Multi-layer variant: validates that block_inputs saving works correctly
// for models with more than one transformer layer.
TEST_F(GPTOverfitTest, MultiLayerLossDecreases) {
  GPTConfig cfg;
  cfg.vocab_size   = 100;
  cfg.max_seq_len  = 16;
  cfg.n_layers     = 2;   // 2 layers — exercises block_inputs saving
  cfg.n_heads      = 4;
  cfg.d_model      = 32;
  cfg.mlp_hidden   = 128;
  cfg.tie_embeddings = false;
  cfg.use_bias     = true;
  cfg.layernorm_eps = 1e-5f;

  const int64_t B = 2, T = 8;

  GPTParams params;
  float* param_buf = test::allocate_params(cfg, params);
  ASSERT_NE(param_buf, nullptr);

  int64_t param_count = cfg.approx_param_count();
  std::vector<float> h_params(param_count);
  std::mt19937 rng(123);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  for (auto& v : h_params) v = dist(rng);
  cudaMemcpy(param_buf, h_params.data(), param_count * sizeof(float),
             cudaMemcpyHostToDevice);

  GPTGrads grads;
  float* grad_buf = test::allocate_grads(cfg, grads);

  GPTForwardState fwd_state;
  float* fwd_buf = test::allocate_forward_state(cfg, B, T, fwd_state);

  std::vector<int32_t> h_ids(B * T), h_targets(B * T);
  std::uniform_int_distribution<int32_t> token_dist(0, cfg.vocab_size - 1);
  for (auto& id : h_ids) id = token_dist(rng);
  for (auto& id : h_targets) id = token_dist(rng);

  int32_t *d_ids, *d_targets;
  cudaMalloc(&d_ids, B * T * sizeof(int32_t));
  cudaMalloc(&d_targets, B * T * sizeof(int32_t));
  cudaMemcpy(d_ids, h_ids.data(), B * T * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, h_targets.data(), B * T * sizeof(int32_t),
             cudaMemcpyHostToDevice);

  Tensor2D<const int32_t> input_ids{d_ids, {B, T}, {T, 1}};
  Tensor1D<const int32_t> targets_flat{d_targets, {B * T}, {1}};

  float* d_loss = nullptr;
  cudaMalloc(&d_loss, sizeof(float));

  float* d_logits_buf = nullptr;
  cudaMalloc(&d_logits_buf, B * T * cfg.vocab_size * sizeof(float));

  train::AdamWConfig opt_cfg;
  opt_cfg.lr = 1e-3f;
  opt_cfg.weight_decay = 0.0f;
  train::AdamW optimizer;
  ASSERT_TRUE(optimizer.init(param_count, opt_cfg).ok());

  const int N = 50;
  std::vector<float> losses;
  losses.reserve(N);

  for (int step = 0; step < N; ++step) {
    cudaMemset(grad_buf, 0, param_count * sizeof(float));

    auto fwd_status = gpt_forward(input_ids, cfg, params, fwd_state, stream_);
    ASSERT_TRUE(fwd_status.ok()) << fwd_status.message();

    Tensor2D<const float> logits_view{fwd_state.logits.data,
                                       fwd_state.logits.shape,
                                       fwd_state.logits.stride};
    auto loss_status = ops::cross_entropy_forward(
        logits_view, targets_flat, d_loss, stream_);
    ASSERT_TRUE(loss_status.ok()) << loss_status.message();

    stream_.synchronize();
    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    losses.push_back(h_loss);

    Tensor2D<float> d_logits_view{d_logits_buf,
                                   {B * T, cfg.vocab_size},
                                   {cfg.vocab_size, 1}};
    auto ce_bwd = ops::cross_entropy_backward(
        logits_view, targets_flat, d_logits_view, stream_);
    ASSERT_TRUE(ce_bwd.ok()) << ce_bwd.message();

    Tensor2D<const float> d_logits_const{d_logits_buf,
                                          {B * T, cfg.vocab_size},
                                          {cfg.vocab_size, 1}};
    auto bwd_status = gpt_backward(
        d_logits_const, input_ids, cfg, params, fwd_state, grads, stream_);
    ASSERT_TRUE(bwd_status.ok()) << bwd_status.message();

    auto opt_status = optimizer.step(param_buf, grad_buf, param_count, stream_);
    ASSERT_TRUE(opt_status.ok()) << opt_status.message();
    optimizer.advance_step();

    stream_.synchronize();
  }

  ASSERT_GT(losses.size(), 10u);
  float first_loss = losses[0];
  float last_loss = losses.back();

  EXPECT_LT(last_loss, first_loss)
      << "Multi-layer loss did not decrease: first=" << first_loss
      << " last=" << last_loss;

  EXPECT_LT(last_loss, 0.8f * first_loss)
      << "Multi-layer loss did not decrease enough: first=" << first_loss
      << " last=" << last_loss;

  optimizer.release();
  cudaFree(d_logits_buf);
  cudaFree(d_loss);
  cudaFree(d_targets);
  cudaFree(d_ids);
  cudaFree(fwd_buf);
  cudaFree(grad_buf);
  cudaFree(param_buf);
}
