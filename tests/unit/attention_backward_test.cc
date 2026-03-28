// Tier B — GPU unit tests for attention backward.

#include "gtest/gtest.h"
#include "src/attention/attention.h"

using namespace gpt;
using namespace gpt::attention;

TEST(AttentionBackwardTest, NotYetImplemented) {
  // Placeholder: attention_backward returns kNotImplemented.
  // This test documents the expected behavior and will be expanded
  // in Milestone 5 with finite-difference gradient checks.
  CudaStream stream;  // default (null stream)
  AttentionConfig cfg;
  cfg.d_model = 32;
  cfg.n_heads = 2;
  cfg.head_dim = 16;
  cfg.use_bias = false;
  cfg.causal = true;

  AttentionParams params{};
  AttentionForwardState state{};
  AttentionGrads grads{};

  Tensor3D<const float> dO{nullptr, {1, 4, 32}, {128, 32, 1}};
  Tensor3D<const float> X{nullptr, {1, 4, 32}, {128, 32, 1}};
  Tensor3D<float> dX{nullptr, {1, 4, 32}, {128, 32, 1}};

  auto status = attention_backward(dO, X, cfg, params, state, dX, grads, stream);
  EXPECT_EQ(status.code(), StatusCode::kNotImplemented);
}
