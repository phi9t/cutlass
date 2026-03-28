// Tier B — GPU unit tests for attention forward.

#include "gtest/gtest.h"
#include "src/attention/attention.h"

using namespace gpt;
using namespace gpt::attention;

TEST(AttentionConfigTest, ScaleComputation) {
  AttentionConfig cfg;
  cfg.d_model = 64;
  cfg.n_heads = 4;
  cfg.head_dim = 16;
  // scale = 1/sqrt(16) = 0.25
  EXPECT_NEAR(cfg.scale(), 0.25f, 1e-6f);
}

TEST(AttentionConfigTest, HeadDimFromModel) {
  AttentionConfig cfg;
  cfg.d_model = 768;
  cfg.n_heads = 12;
  cfg.head_dim = 64;
  EXPECT_NEAR(cfg.scale(), 1.0f / 8.0f, 1e-6f);
}

// NOTE: Full forward integration tests require allocated GPU buffers for
// all intermediate tensors in AttentionForwardState.  These will be added
// as part of Milestone 5 when workspace allocation is implemented.
