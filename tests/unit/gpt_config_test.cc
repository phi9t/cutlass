// Tier A — Host-only unit tests for GPT config.

#include "gtest/gtest.h"
#include "src/model/gpt_config.h"

using namespace gpt::model;

TEST(GPTConfigTest, DefaultValues) {
  GPTConfig c;
  EXPECT_EQ(c.vocab_size, 50257);
  EXPECT_EQ(c.d_model, 768);
  EXPECT_EQ(c.n_heads, 12);
  EXPECT_EQ(c.head_dim(), 64);
}

TEST(GPTConfigTest, ParamCountPositive) {
  GPTConfig c;
  EXPECT_GT(c.approx_param_count(), 0);
}

TEST(GPTConfigTest, SmallConfig) {
  GPTConfig c;
  c.vocab_size = 100;
  c.max_seq_len = 16;
  c.n_layers = 2;
  c.n_heads = 2;
  c.d_model = 32;
  c.mlp_hidden = 128;
  EXPECT_EQ(c.head_dim(), 16);
  EXPECT_GT(c.approx_param_count(), 0);
}
